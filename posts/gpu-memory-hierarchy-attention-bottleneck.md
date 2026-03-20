## 核心结论

Attention 在 GPU 上慢，常见原因不是乘法不够快，而是显存搬运太多。

先定义两个术语。

- **HBM**：GPU 外部的大显存，容量大，但相对更慢。
- **SRAM**：GPU 芯片上的片上存储，容量很小，但带宽很高，适合做临时缓存。

以 A100 80GB 为例，官方规格里的 HBM 带宽约为 1.9 到 2.0 TB/s，FP16/BF16 Tensor Core 峰值算力约 312 TFLOP/s，因此一个常用的屋顶线门槛是：

$$
\text{ridge point} \approx \frac{312\ \text{TFLOP/s}}{2\ \text{TB/s}} \approx 156\ \text{FLOP/byte}
$$

这里的 **Arithmetic Intensity，算术强度**，白话讲就是“每搬 1 字节数据，顺手做了多少次浮点运算”：

$$
AI = \frac{\text{FLOPs}}{\text{Bytes}}
$$

如果 AI 低于这个门槛，程序通常是 **memory-bound**，白话讲就是“算力单元在等数据”；如果高于门槛，更可能是 **compute-bound**，也就是“真正被计算能力限制”。

标准 Attention 的问题在于，它会把中间的 $N\times N$ 分数矩阵 $S$ 和概率矩阵 $P$ 反复写回 HBM。结果是总 FLOPs 虽然不低，但总 Bytes 更夸张，整体 AI 经常只有几十，明显低于 156。于是 Tensor Core 不能持续吃满数据。

FlashAttention 没有改变 Attention 的数学结果，改变的是执行顺序：把 $Q,K,V$ 分块搬到 SRAM，在片上完成 `QK^T -> scale -> softmax -> PV` 的融合计算，不再把完整的 $S,P$ 物化到 HBM。计算量仍然是 $O(N^2d)$，但 IO 显著下降，AI 被抬高，Attention 才开始接近 GPU 擅长的工作方式。

---

## 问题定义与边界

本文讨论的是 **精确的 dense self-attention**，即：

$$
O=\text{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

其中：

- $N$ 是序列长度
- $d$ 是单头维度
- $Q,K,V\in\mathbb{R}^{N\times d}$
- $O\in\mathbb{R}^{N\times d}$

边界也要先说清楚。

1. 本文主要讨论单头 Attention 的 IO 分析，多头只是把结果按 head 数线性放大。
2. 本文关注训练或 prefill 阶段的主瓶颈，不重点展开 decode-only 场景。
3. 本文比较的是“标准实现”和“FlashAttention 一类 IO-aware 实现”，不是在比较 dense attention 与 sparse attention 的建模能力。

为什么标准实现会被带宽拖住，可以直接看一次典型流程：

1. 读入 $Q,K$，计算 $S=QK^\top$
2. 把 $S$ 写回 HBM
3. 再把 $S$ 读出来做 softmax，得到 $P$
4. 把 $P$ 写回 HBM
5. 再把 $P$ 读出来，与 $V$ 相乘得到 $O$

真正必须保留到最后的只有输入 $Q,K,V$ 和输出 $O$，规模是 $O(Nd)$；但标准实现多搬了两份 $N\times N$ 中间矩阵，规模变成 $O(N^2)$。当 $N \gg d$ 时，问题就从“矩阵乘很多”变成了“显存交通爆炸”。

下面用常见的例子量化。设 FP16，每个元素 2 字节，$N=4096,d=128$。

- $Q,K,V,O$ 各自大小：$4096\times128\times2 \approx 1$ MB
- 单个 $N\times N$ 矩阵大小：$4096^2\times2 \approx 32$ MB

于是标准实现的大头是：

| 步骤 | FLOPs | HBM Bytes | 说明 |
|---|---:|---:|---|
| $QK^\top$ | $2N^2d$ | 读 $Q,K$，写 $S$ | matmul 本身不坏 |
| softmax | 约 $5N^2$ | 读 $S$，写 $P$ | 算得少，搬得多 |
| $PV$ | $2N^2d$ | 读 $P,V$，写 $O$ | 又一次大矩阵读取 |

代入数值后：

- 总 FLOPs 约为 $4N^2d \approx 8.59$ GFLOP
- 总 HBM 流量约为 $132$ MB
- 所以

$$
AI_{\text{std}} \approx \frac{8.59\ \text{GFLOP}}{132\ \text{MB}} \approx 65\ \text{FLOP/byte}
$$

65 明显低于 156，所以标准 Attention 在 A100 上是典型的 memory-bound。

一个直观对比是：理想情况下，读 $Q,K,V$ 再写 $O$，总共大约只需要 4 MB；标准实现却做到了 132 MB，约 33 倍冗余。问题不在公式本身，而在数据移动路径。

---

## 核心机制与推导

FlashAttention 的核心不是“少算”，而是“少搬”。

### 1. 为什么 softmax 看起来迫使我们物化整行

softmax 的单行形式是：

$$
p_j=\frac{e^{x_j}}{\sum_k e^{x_k}}
$$

数值稳定版会先减去最大值：

$$
p_j=\frac{e^{x_j-m}}{\sum_k e^{x_k-m}},\quad m=\max_k x_k
$$

看起来要先看到整行，才能知道 $m$ 和分母，所以很多实现自然会先把整行分数 $S[i,:]$ 存下来。

FlashAttention 的关键观察是：这些统计量可以**在线更新**。

### 2. online softmax 递推

假设某一行分数被分成多个 tile。处理到第 $t$ 个 tile 时，维护两个量：

- $m^{(t)}$：到目前为止见过的最大值
- $\ell^{(t)}$：以这个最大值为基准的指数和

若当前 tile 的局部最大值和局部指数和分别为 $\tilde m,\tilde \ell$，则更新为：

$$
m^{(t+1)}=\max(m^{(t)},\tilde m)
$$

$$
\ell^{(t+1)}=
e^{m^{(t)}-m^{(t+1)}}\ell^{(t)}
+
e^{\tilde m-m^{(t+1)}}\tilde \ell
$$

输出累加项也同步缩放更新。设当前累计输出为 $o^{(t)}$，当前 tile 对应的局部加权和值为 $\tilde o$，则：

$$
o^{(t+1)}=
e^{m^{(t)}-m^{(t+1)}}o^{(t)}
+
e^{\tilde m-m^{(t+1)}}\tilde o
$$

最后一行输出为：

$$
O_i=\frac{o^{(T)}}{\ell^{(T)}}
$$

这就是 **online softmax**。白话讲，它让我们“边读一块，边更新全局归一化统计量”，不必先把整行分数存满再回来算。

### 3. 玩具例子

设一行分数是 `[1, 2, 10, 11]`，分成两块 `[1,2]` 和 `[10,11]`。

- 第一块的最大值是 2
- 第二块的最大值是 11
- 全局最大值显然是 11

如果先按第一块做过一遍指数和，第二块到来时并不需要重算整行，只需要把旧和乘上 $e^{2-11}$ 再加上新块的贡献。因为指数函数满足这种缩放重标定关系，所以累计量可以被修正到新的全局基准上。

这就是 FlashAttention 能“分块 yet exact”的数学基础。

### 4. IO 为什么降下来了

标准 Attention 物化了完整 $S,P$，所以 IO 主项是 $O(N^2)$。

FlashAttention 设片上可用 SRAM 大小为 $M$，一次能容纳的 tile 规模受 $M$ 限制。论文分析表明，其 HBM 访问可写成近似的：

$$
O\left(\frac{N^2d^2}{M}\right)
$$

在固定 $d$ 的常见场景下，常被简写理解为：

$$
O\left(\frac{N^2d}{M}\right)\ \text{个 tile 级访问}
$$

直观含义只有一个：**SRAM 越大，块越大，重复从 HBM 读取的次数越少**。

因此 FlashAttention 抬高 AI 的方式不是减少 FLOPs，而是减少 Bytes。很多工程文章用 $N=4096,d=128$ 的示意数字，把标准实现的 132 MB 压到十几 MB，甚至接近只读写输入输出的理想下界。于是：

| 实现 | 总 FLOPs | 总 Bytes | AI |
|---|---:|---:|---:|
| 标准 Attention | 8.59 GFLOP | 132 MB | 约 65 |
| FlashAttention 示意值 | 8.59 GFLOP | 17 MB | 约 506 |

506 已经越过 A100 的 156 门槛，因此更接近 compute-bound。

---

## 代码实现

下面给一个可运行的 Python 版本。它不是高性能 CUDA，而是把 FlashAttention 的两个关键点写清楚：

1. 不物化完整 $N\times N$ 矩阵
2. 用 online softmax 分块得到与标准实现一致的结果

```python
import math

def matmul_transpose_row(q_row, K):
    return [sum(a * b for a, b in zip(q_row, k_row)) for k_row in K]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def attention_naive(Q, K, V, scale):
    out = []
    for q in Q:
        scores = [s * scale for s in matmul_transpose_row(q, K)]
        probs = softmax(scores)
        row = []
        for j in range(len(V[0])):
            row.append(sum(p * v[j] for p, v in zip(probs, V)))
        out.append(row)
    return out

def attention_blockwise(Q, K, V, scale, block_size=2):
    n = len(Q)
    d = len(V[0])
    out = []

    for q in Q:
        m = float("-inf")   # running max
        l = 0.0             # running denominator
        acc = [0.0] * d     # running numerator

        for start in range(0, n, block_size):
            K_blk = K[start:start + block_size]
            V_blk = V[start:start + block_size]

            scores = [scale * sum(a * b for a, b in zip(q, k)) for k in K_blk]
            blk_m = max(scores)
            blk_exp = [math.exp(x - blk_m) for x in scores]
            blk_l = sum(blk_exp)

            blk_acc = [0.0] * d
            for w, v in zip(blk_exp, V_blk):
                for j in range(d):
                    blk_acc[j] += w * v[j]

            new_m = max(m, blk_m)
            old_factor = 0.0 if m == float("-inf") else math.exp(m - new_m)
            blk_factor = math.exp(blk_m - new_m)

            l = old_factor * l + blk_factor * blk_l
            for j in range(d):
                acc[j] = old_factor * acc[j] + blk_factor * blk_acc[j]

            m = new_m

        out.append([x / l for x in acc])
    return out

def almost_equal(a, b, eps=1e-9):
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for xa, xb in zip(ra, rb):
            if abs(xa - xb) > eps:
                return False
    return True

# 玩具例子
Q = [[1.0, 0.0], [0.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
V = [[2.0, 1.0], [1.0, 3.0], [4.0, 0.0], [0.0, 2.0]]

scale = 1.0 / math.sqrt(2.0)
naive = attention_naive(Q, K, V, scale)
block = attention_blockwise(Q, K, V, scale, block_size=2)

assert almost_equal(naive, block)

# 算术强度示意
std_flops = 8.59e9
std_bytes = 132e6
fa_bytes = 17e6
std_ai = std_flops / std_bytes
fa_ai = std_flops / fa_bytes

assert std_ai < 156
assert fa_ai > 156
assert fa_ai > std_ai
```

真正的 GPU 实现会把上面的逻辑改写成 fused kernel。伪代码结构通常是：

```text
for q_tile in Q_tiles:
    load q_tile to SRAM
    init m, l, acc

    for kv_tile in KV_tiles:
        load k_tile, v_tile to SRAM
        scores = q_tile @ k_tile^T
        online_update(m, l, acc, scores, v_tile)

    write output_tile to HBM
```

真实工程例子是 GPT 类模型训练。FlashAttention 论文报告过在 GPT-2 等场景获得数倍级注意力层加速，本质原因不是近似，而是 IO 路径被重写了。

---

## 工程权衡与常见坑

FlashAttention 不是“自动更快”，它依赖一组严格的工程条件。

| 项目 | 传统物化 $S/P$ | FlashAttention |
|---|---|---|
| 中间矩阵 | 存完整 $N\times N$ | 不落 HBM，只保 tile |
| HBM 流量 | 高，主项是 $O(N^2)$ | 低，接近 IO 下界 |
| 显存占用 | 容易随序列长爆炸 | 显著更稳 |
| 反向传播 | 直接读缓存 | 常用 recomputation |
| 速度瓶颈 | 带宽 | 更接近算力/调度 |

常见坑主要有五类。

1. **tile 设计不匹配 SRAM**
   片上存储很小，块太大放不下，块太小又会增加循环次数和调度开销。

2. **只做了分块，没做融合**
   如果 `matmul`、`softmax`、`PV` 还是三个独立 kernel，中间结果照样回 HBM，收益会被吃掉。

3. **bank conflict 和 occupancy 不好**
   术语解释：**occupancy** 是“一个 SM 上同时挂了多少活”，太低会让硬件空转。shared memory 访问冲突也会把理论带宽打折。

4. **反向传播缓存策略错误**
   FlashAttention 通常会保存每行的少量统计量，如 `max` 和 `sum`，在 backward 时重算局部分数，而不是缓存完整 $S,P$。这是典型的“用少量 FLOPs 换大量 Bytes”。

5. **误把所有场景都当成长序列训练**
   对很短序列、很小 batch 或 decode 单 token 场景，瓶颈可能转向 launch overhead、KV cache 访问或跨卡通信，此时收益模式会变。

一个实际判断标准是：如果你的 profiler 里 attention kernel 显示 Tensor Core 利用率低、HBM 带宽高、而且随序列长度增长几乎线性恶化为“等内存”，那就是典型的 IO 问题，不是“矩阵乘法库不够快”。

---

## 替代方案与适用边界

FlashAttention 解决的是“精确 dense attention 的 IO 问题”，不是所有 attention 问题的总解。

### 1. FlashAttention-2

FlashAttention-2 沿用相同思想，但进一步优化了线程块划分、warp 分工和 shared memory 路径，在 A100 上报告过最高约 230 TFLOP/s。它适合：

- 长序列训练
- 大 batch prefill
- 多头 dense attention 的主流训练场景

### 2. LeanAttention

LeanAttention 更偏向 decode-phase，也就是生成阶段。它把 online softmax 的结合性进一步当作并行归约来用，目标是长上下文生成时把并行度拉起来。适合：

- 超长上下文推理
- 单步解码延迟敏感
- 多 GPU 推理并行

### 3. 传统 Attention 仍有边界内价值

如果序列很短，或者运行环境没有高质量 fused kernel，传统实现可能因为代码简单、调试容易，仍然是合理选择。此时主要矛盾不是 $N^2$ IO。

可以用一个简单决策表：

| 场景 | 更合适的方案 |
|---|---|
| 短序列、调试优先 | 标准 Attention |
| 中长序列、训练/预填充 | FlashAttention / FlashAttention-2 |
| 超长上下文、decode 延迟敏感 | LeanAttention 或面向 decode 的变体 |
| 稀疏模式明确 | block-sparse 等近似/结构化方案 |

结论很直接：当问题是“密集 Attention 太吃带宽”时，应该优先改算法的数据流；当问题已经转成“并行划分、解码调度、跨卡协同”时，再考虑 FlashAttention-2、LeanAttention 或稀疏变体。

---

## 参考资料

- [NVIDIA A100 Tensor Core GPU 官方规格](https://www.nvidia.com/en-us/data-center/a100/)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://crfm.stanford.edu/2023/07/17/flash2.html)
- [Hugging Face: Standard Attention — The IO Problem](https://huggingface.co/blog/atharv6f/standard-attention-drawbacks)
- [Hugging Face: FlashAttention — IO Analysis and Evolution](https://huggingface.co/blog/atharv6f/flash-attention-io-analysis)
- [LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers](https://www.microsoft.com/en-us/research/publication/lean-attention-hardware-aware-scalable-attention-mechanism-for-the-decode-phase-of-transformers/)

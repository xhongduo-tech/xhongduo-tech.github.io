## 核心结论

FlashAttention-2 的价值不在于“把注意力公式改了”，而在于把同一个精确注意力计算，重新排成更适合 GPU 的执行顺序。核心有三点：

1. 把 softmax 的重缩放延后到循环结束后统一做，减少非矩阵乘 FLOP。
2. 把 Q 的分块循环放外层、K/V 的分块循环放内层，暴露序列维度并行。
3. 在 causal mask 下做块级早停，整块为 0 的区域直接跳过。

这三点叠加后，FlashAttention-2 更接近 GEMM。GEMM 指通用矩阵乘，白话就是 GPU 最擅长、Tensor Core 最容易跑满的那类计算。Hazy Research 给出的 A100 数据显示，FlashAttention-2 在端到端 GPT 训练中可达到约 225 TFLOPs/s，对应约 72% 的模型 FLOPs 利用率，GPT3-2.7B、8k 上下文场景下相对 FlashAttention v1 约有 1.3× 训练加速。

| 项目 | FlashAttention v1 | FlashAttention-2 |
|---|---:|---:|
| 理论峰值利用率 | 约 25% 到 40% | 约 50% 到 73% |
| GPT3-2.7B, 8k, A100 | 约 175 TFLOPs/s | 约 225 TFLOPs/s |
| 端到端训练加速 | 基线 | 相对 v1 约 1.3× |
| 关键瓶颈 | 重缩放多、并行度不足、warp 通信多 | 重缩放减少、序列并行、warp 通信少 |

一句话概括就是：

$$
\text{softmax delay} + \text{Q 外层并行} + \text{causal 块跳过}
\Rightarrow \text{更高 Tensor Core 利用率}
$$

---

## 问题定义与边界

标准注意力是：

$$
O=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

这里的难点不是公式本身，而是当序列长度 $N$ 很大时，$QK^\top$ 是 $N\times N$，显存和带宽都扛不住。FlashAttention v1 的做法是分块计算，不把完整注意力矩阵写回显存，而是在片上缓存里一块一块地算。

问题在于，v1 虽然已经把 IO 成本降下来了，但还有三个硬件层面的浪费：

1. 每处理一个 K/V 块，都要做一次 softmax 相关重缩放。
2. 一个 attention head 往往只对应一个 thread block。thread block 白话说就是 GPU 调度的一支计算小队。若 batch 小、head 少、序列长，很多 SM 会闲着。
3. 在 causal mask 下，明知某些分块全被遮挡，仍可能进入计算路径。

新手可以这样理解：

- v1 像“每搬来一箱货，就重新结一次账”。
- GPU 只能派一个小队处理一个 head，其他小队可能无事可做。
- 自回归场景里，上三角很多位置本来就不允许看见，却还在做无意义检查甚至计算。

分块视角下，可以把一个 head 的注意力矩阵看成很多 tile：

| 分块位置 | 是否可能有有效值 | v1 常见处理 | v2 优化方向 |
|---|---|---|---|
| 对角线附近块 | 有 | 正常计算 | 正常计算 |
| 部分遮挡块 | 有一部分 | 计算并加 mask | 计算并加 mask |
| 完全遮挡块 | 全为 0 | 可能仍进入路径 | 直接跳过 |

这里的边界也要说清楚。FlashAttention-2 解决的是“精确注意力在 GPU 上怎么更高效地排布”，不是稀疏注意力、线性注意力那类“换公式”路线。它不改变输出数学意义，优化的是执行顺序、并行拆分和 kernel 内部通信。

---

## 核心机制与推导

FlashAttention-2 的核心改动，是把“每块都归一化一次”改成“先累计无缩放输出，最后统一归一化”。

softmax 的在线更新里，会维护两个统计量：

- $m^{(j)}$：当前扫描到第 $j$ 个 K/V 块时的行最大值。
- $\ell^{(j)}$：指数和。
- $L^{(j)} = m^{(j)} + \log \ell^{(j)}$：log-sum-exp 形式，白话就是把 softmax 分母压缩成一个更稳定、也更省存储的量。

FlashAttention-2 维护无缩放输出 $\tilde O^{(j)}$：

$$
\tilde{O}^{(j)}
=
\exp(m^{(j-1)}-m^{(j)})\,\tilde{O}^{(j-1)}
+
\exp(S^{(j)}-m^{(j)})V^{(j)}
$$

其中 $S^{(j)}$ 是当前 Q 块与第 $j$ 个 K 块的分数。最后统一做一次归一化：

$$
O=\mathrm{diag}(\exp(-L^{(T_c)}))\,\tilde{O}^{(T_c)}
$$

和 v1 的差别是：v1 在块内、块间会反复做输出重缩放；v2 把这件事推迟到最后一次。这样做的收益不是“大 O 复杂度突然变了”，而是把很多标量操作、边界检查和共享内存读写从热路径里挪走了。对于 GPU 来说，这种非 matmul 操作很贵，因为 Tensor Core 主要加速的是矩阵乘，不是这些零散标量算子。

### 玩具例子

假设只有 1 行 Q，要看 2 个 K/V 块，每块 2 个 token。

第一块分数是：
$$
S^{(1)}=[1,2]
$$

第二块分数是：
$$
S^{(2)}=[3,0]
$$

对应的 V 分别是：
$$
V^{(1)}=\begin{bmatrix}10\\20\end{bmatrix},\quad
V^{(2)}=\begin{bmatrix}30\\40\end{bmatrix}
$$

处理第一块后，可以得到一版 $\tilde O^{(1)}$ 和 $L^{(1)}$。再处理第二块时，不需要先把第一块已经算好的输出完整归一化再重算，只要用新的 $m^{(2)}$ 调整旧值和新值的尺度，然后继续累积。直到第二块处理完，再统一乘一次 $\exp(-L^{(2)})$，就得到最终输出。

直观看，流程是：

1. 读入一块 K/V。
2. 更新当前最大值和 log-sum-exp。
3. 把这块对输出的贡献累积进 $\tilde O$。
4. 所有块结束后，统一归一化。

这就是“无缩放输出先累计，最后一次缩放”。

### 真实工程例子

以 GPT3-2.7B、8k 上下文、A100 为例，长序列通常伴随更小的 batch。此时如果仍然只按 `batch × heads` 发 thread block，GPU 的 SM 很难占满。FlashAttention-2 把 Q 切成更多行块，让一个 head 内部也能分成多个 block 并行跑。这样同样是 8k 上下文，GPU 不再因为“head 数不够多”而空转。Hazy Research 给出的结果是这一场景从约 175 TFLOPs/s 提升到约 225 TFLOPs/s。

---

## 代码实现

工程上可以把 FlashAttention-2 的前向 kernel 理解成下面这段伪代码：

```python
for q_block in Q_blocks:              # Q 在外层：暴露序列并行
    init m, l, O_tilde

    for kv_block in KV_blocks:        # K/V 在内层：逐块累积
        if causal_mask and block_is_fully_masked(q_block, kv_block):
            continue                  # 整块被遮挡，直接跳过

        S = q_block @ k_block.T * sm_scale
        m, l = update_logsumexp_stats(m, l, S)
        O_tilde = update_unscaled_output(O_tilde, m, S, v_block)

    O_block = final_normalize(O_tilde, m, l)   # 最后统一归一化
    write_back(O_block)
```

新手版理解：

- “Q 在外”不是数学要求，而是调度要求。
- “K/V 在内”表示同一个 Q 块会扫过所有 K/V 块。
- “最后一起算 softmax”不是偷懒，而是利用在线 softmax 的等价变形，把多次缩放变成一次。

下面用一个可运行的 Python 玩具实现，验证“延后归一化”和普通 softmax attention 结果一致：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def standard_attention_row(scores, values):
    probs = softmax(scores)
    return sum(p * v for p, v in zip(probs, values))

def flash2_style_row(scores_blocks, values_blocks):
    m = float("-inf")
    l = 0.0
    o_tilde = 0.0

    for s_block, v_block in zip(scores_blocks, values_blocks):
        block_max = max(s_block)
        new_m = max(m, block_max)

        # 旧块对新尺度的贡献
        old_scale = 0.0 if m == float("-inf") else math.exp(m - new_m)

        # 新块在新尺度下的贡献
        exps = [math.exp(s - new_m) for s in s_block]
        block_l = sum(exps)
        block_o = sum(e * v for e, v in zip(exps, v_block))

        l = l * old_scale + block_l
        o_tilde = o_tilde * old_scale + block_o
        m = new_m

    return o_tilde / l

# 玩具例子：两块 K/V，每块两个 token
scores = [1.0, 2.0, 3.0, 0.0]
values = [10.0, 20.0, 30.0, 40.0]

std = standard_attention_row(scores, values)
fa2 = flash2_style_row(
    scores_blocks=[[1.0, 2.0], [3.0, 0.0]],
    values_blocks=[[10.0, 20.0], [30.0, 40.0]],
)

assert abs(std - fa2) < 1e-9
print(round(std, 6), round(fa2, 6))
```

如果把并行调度也写成更贴近 kernel 的结构，关键是三点：

1. Q 行块要能独立分配给多个 thread block。
2. warp 用 split-Q，不用 split-K。
3. causal mask 要在块级先判断，而不是把无效块送进主计算路径。

---

## 工程权衡与常见坑

FlashAttention-2 的收益来自“更接近 GPU 喜欢的执行形状”，所以常见坑基本都和调度、块形状、mask 判定有关。

最常见的第一个坑，是还按 v1 的思路写循环。如果 Q 仍在内层、K/V 在外层，就很难把一个 head 拆成多个独立 row block，也就失去长序列下最关键的并行来源。

第二个坑，是 causal mask 只做元素级判断，不做块级早停。这样虽然结果对，但完全遮挡块仍会经历访存、边界判断甚至部分算子调用，吞吐率会掉得很快。

第三个坑，是忘了非 matmul FLOP 的代价。A100 上 matmul 峰值远高于普通 FP32 标量运算，所以“多几次缩放无所谓”在 GPU kernel 里并不成立。

在 causal mask 下，常见会遇到三类块：

| 块类型 | 例子 | 处理逻辑 | 风险 |
|---|---|---|---|
| 完整有效块 | 对角线左下区域 | 正常 matmul + 更新统计量 | 无 |
| 部分覆盖块 | 穿过对角线的块 | 逐元素加 mask 后计算 | 分支和边界判断多 |
| 完全遮挡块 | 对角线右上区域 | 直接 `continue` | 若未跳过会白算 |

对应的伪逻辑通常是：

```python
if kv_end <= q_start:
    # 完整有效块
    compute_full_block()
elif kv_start > q_end:
    # 完全遮挡块
    continue
else:
    # 部分覆盖块
    compute_partial_block_with_mask()
```

再补两个工程上的细节：

- warp 是 GPU 中 32 个线程的执行组，白话就是最小协同作战单位。v1 的 split-K 需要 warp 之间汇总中间结果，通信多；v2 的 split-Q 让每个 warp 算自己负责的 Q 子块，通信更少。
- tile 形状不能只看理论访存量，还要看寄存器压力和 occupancy。occupancy 白话就是 GPU 上能同时挂起多少活跃计算块。tile 过大可能反而让 occupancy 下降。

---

## 替代方案与适用边界

FlashAttention-2 不是所有场景都绝对最优。

如果序列很短、batch 很大、head 也多，v1 按 `batch × heads` 已经能把 GPU 填满，这时 v2 额外的调度复杂度和块级判断不一定值得。换句话说，FlashAttention-2 最擅长的是“长上下文 + 并行度天然不足”的区间。

如果硬件对 Tensor Core 友好度不高，或者实现环境不方便做复杂 kernel 重排，也可以退一步用 fused softmax + mask + matmul 的常规融合方案。这类方案实现简单，但通常拿不到 v2 那种序列并行和块跳过收益。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| FlashAttention v1 | 实现成熟，已显著省显存 | 长序列下并行度不足 | 中长序列，通用训练 |
| FlashAttention-2 | 吞吐更高，长序列更强 | kernel 更复杂 | 长上下文训练与推理 |
| 普通 fused attention | 接入简单 | IO 和并行优化弱 | 短序列或工程快速落地 |
| 稀疏/线性注意力 | 可进一步降复杂度 | 改变原始注意力形式 | 可接受近似或结构改动 |

一个实用判断标准：

```python
def should_use_flashattention2(seq_len, batch_size, num_heads):
    # 粗略启发式，不是论文公式
    return seq_len >= 4096 and batch_size * num_heads < 80

assert should_use_flashattention2(8192, 1, 32) is True
assert should_use_flashattention2(512, 16, 16) is False
```

这段代码表达的是经验而不是定理：当序列足够长，而 `batch_size × num_heads` 又不足以占满 GPU 时，FlashAttention-2 的序列并行更容易体现价值。

---

## 参考资料

下面三份材料分工很清楚：

| 资料 | 侧重点 | 适合怎么读 |
|---|---|---|
| [FlashAttention-2 论文（arXiv）](https://arxiv.org/abs/2307.08691) | 正式定义、实验、算法边界 | 先看摘要和方法，再看实验表 |
| [Hazy Research 官方博客](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) | 并行设计、硬件视角、A100 性能数据 | 先建立直觉，再回论文看细节 |
| [EmergentMind 综述](https://www.emergentmind.com/topics/flashattention2) | 数学推导整理、公式重述 | 用来补在线 softmax 推导 |

如果你是第一次读这个主题，推荐顺序是：

1. 先读 Hazy 博客，理解“为什么 v1 还不够快”。
2. 再读论文摘要与方法部分，确认三项优化的正式表述。
3. 最后看 EmergentMind 的公式整理，把 $\tilde O$、$L^{(j)}$ 和最终归一化串起来。

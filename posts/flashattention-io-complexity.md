## 核心结论

FlashAttention 的核心不是“少算”，而是“少搬数据”。

标准 Attention 在实现时，通常会先算出 $S = QK^\top$，再做 softmax，最后与 $V$ 相乘得到输出。这里的 $S$ 是一个 $N \times N$ 的注意力分数矩阵。$N$ 是序列长度，也就是 token 数；当 $N$ 很大时，这个矩阵本身就会变成显存读写瓶颈。

FlashAttention 的做法是 IO-aware，即“按内存层级设计计算顺序”。白话说，它优先考虑 GPU 里哪一级内存最贵、哪一级最便宜，再决定算子怎么排。它把 $Q/K/V$ 分成能放进 SRAM 的小块，点积、缩放、softmax、加权求和都在同一个 tile 内完成，避免把完整的 $N \times N$ 中间结果写回 HBM。

因此，标准 Attention 的 HBM IO 可以概括为：

$$
\Theta(Nd + N^2)
$$

而 FlashAttention 将它压到：

$$
\Theta\left(\frac{N^2 d^2}{M}\right)
$$

其中 $d$ 是每个 token 的 hidden dimension，$M$ 是片上 SRAM 容量。这个式子的含义不是“没有 $N^2$ 了”，而是“$N^2$ 级别的数据访问被 tile 大小吸收了一部分，不再需要把整张注意力矩阵来回写到 HBM”。

一个新手版直觉可以这样记：不是一次处理整张 $N \times N$ 表，而是把它切成很多 $128 \times 128$ 的小格子，每次只把一个小格子相关的 $Q/K/V/O$ 搬进 SRAM，算完就累计到输出，不生成全局 attention 矩阵。

---

## 问题定义与边界

先明确讨论对象。

Attention 这里指标准 scaled dot-product attention：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

其中：

- $Q$ 是 query，表示“当前 token 想查什么”
- $K$ 是 key，表示“每个 token 提供什么索引”
- $V$ 是 value，表示“每个 token 真正携带的信息”

“IO复杂度”讨论的是数据在 HBM 和 SRAM 之间搬运了多少次，不是算术乘加次数。HBM 可以理解为显卡的大显存，容量大但访问慢；SRAM 可以理解为芯片上的小缓存，容量小但访问快。GPU 上很多 Transformer kernel 不是被 FLOPs 卡住，而是被 HBM 带宽卡住。

标准 Attention 的边界问题主要有两个：

| 项目 | 标准 Attention | FlashAttention |
|---|---|---|
| 是否显式保存 $N \times N$ 分数矩阵 | 是 | 否 |
| HBM IO 主项 | $\Theta(Nd + N^2)$ | $\Theta(N^2 d^2 / M)$ |
| 长上下文是否容易爆显存带宽 | 是 | 明显缓解 |
| 主要优化方向 | 算子分别执行 | 算子融合 + 分块 |

这里的 $\Theta(Nd)$ 对应读入 $Q/K/V/O$，$\Theta(N^2)$ 对应 attention 分数和概率矩阵的中间读写。序列越长，$N^2$ 项越快压过其他项。

玩具例子：如果 $N=4$，注意力矩阵只有 $4\times4=16$ 个元素，显式存下来没问题。但如果 $N=32768$，则矩阵有：

$$
32768^2 \approx 1.07 \times 10^9
$$

这已经是十亿级 entry。即使每个 entry 只用 2 字节的 half precision，也要大约 2GB，只是存一个中间矩阵就已经不便宜，更别说前向和反向还要反复读写。

所以本文讨论边界是：GPU 上的长上下文 Transformer，瓶颈主要来自显存带宽和中间张量落盘，而不是 CPU 推理、小模型短文本、或者极端稀疏 attention。

---

## 核心机制与推导

FlashAttention 的关键是 tiling，也就是“分块处理”。白话说，就是把大矩阵拆成很多能放进 SRAM 的小矩阵块。

设：

- 序列长度为 $N$
- hidden dimension 为 $d$
- SRAM 可用容量为 $M$
- 列块大小为 $B_c$
- 行块大小为 $B_r$

为了让一个 tile 内同时放下 $Q$ 块、$K$ 块、$V$ 块和输出累加块 $O$，通常需要满足近似约束：

$$
4B_c d \le M
$$

于是可得一个常见估计：

$$
B_c \approx \left\lfloor \frac{M}{4d} \right\rfloor
$$

这不是精确硬件公式，而是工程上很有用的上界估算。因为真实实现还要给 softmax 的中间统计量留空间，比如每一行的最大值 $m_i$ 和归一化因子 $l_i$。

为什么这样能降 IO？

标准做法会显式生成 $S=QK^\top$，它大小是 $N \times N$，因此会出现 $\Theta(N^2)$ 级别 HBM 读写。FlashAttention 则把整个注意力矩阵视为很多 tile：

$$
\frac{N}{B_r} \times \frac{N}{B_c}
$$

个小块。每次只处理一个 $(Q_i, K_j, V_j)$ 组合：

1. 载入一个 query block 到 SRAM
2. 依次载入多个 key/value block
3. 在 SRAM 内做点积、缩放、局部 softmax 更新、对输出做累计
4. 不把这个 attention tile 写回 HBM

因为 attention tile 从未完整落到 HBM，所以原本的 $\Theta(N^2)$ 中间矩阵读写消失了，代价变成“为了扫完整个序列，需要重复载入多少个 tile”。

如果把块大小代入总 tile 数，可以得到论文中的 IO 上界量级：

$$
\Theta\left(\frac{N^2 d^2}{M}\right)
$$

直观推导是：tile 越大，需要处理的块数越少；而 tile 大小受 SRAM $M$ 限制。由于 $B_c \propto M/d$，总块数大约与 $N^2/B_c$ 成正比，代入后就得到 $N^2 d^2 / M$ 这个形式。

A100 的玩具算例很适合理解这个式子。假设：

- $M \approx 192\text{KB}$
- $d = 128$
- 取 $B_c=B_r=128$

则单个块元素数约为 $128 \times 128 = 16384$。若用 fp16，每块约 32KB。四块 $Q/K/V/O$ 大约需要 128KB，再加 softmax 统计量和缓冲区，整体仍能控制在 192KB 左右。这说明 128 这个 tile 能在 SRAM 内闭环完成计算，不需要反复把 attention 分数写回 HBM。

真实工程例子：32K token 的长上下文推理里，如果用朴素 attention，注意力分数矩阵规模会直接打到十亿级元素，HBM 带宽会成为主瓶颈。FlashAttention 的实现则是“固定 query tile，顺序扫描所有 KV tile 并累计输出”，显存占用不会随着完整 $N \times N$ 中间矩阵线性膨胀。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不是 CUDA kernel，但保留了 FlashAttention 最重要的结构：按 tile 遍历、在线 softmax、只累计输出、不保存完整分数矩阵。

```python
import math
import numpy as np

def naive_attention(Q, K, V):
    scores = Q @ K.T / math.sqrt(Q.shape[1])
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs @ V

def flash_attention_tiled(Q, K, V, block=2):
    n, d = Q.shape
    O = np.zeros((n, d), dtype=np.float64)
    m = np.full((n,), -np.inf, dtype=np.float64)   # 每行历史最大值
    l = np.zeros((n,), dtype=np.float64)           # 每行历史归一化因子

    scale = 1.0 / math.sqrt(d)

    for j in range(0, n, block):
        Kj = K[j:j + block]
        Vj = V[j:j + block]

        for i in range(0, n, block):
            Qi = Q[i:i + block]
            scores = (Qi @ Kj.T) * scale

            row_max = scores.max(axis=1)
            new_m = np.maximum(m[i:i + block], row_max)

            exp_old = np.exp(m[i:i + block] - new_m) * l[i:i + block]
            exp_new = np.exp(scores - new_m[:, None])
            new_l = exp_old + exp_new.sum(axis=1)

            old_o = O[i:i + block] * (exp_old / new_l)[:, None]
            new_o = (exp_new @ Vj) / new_l[:, None]
            O[i:i + block] = old_o + new_o

            m[i:i + block] = new_m
            l[i:i + block] = new_l

    return O

Q = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
K = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
V = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])

out1 = naive_attention(Q, K, V)
out2 = flash_attention_tiled(Q, K, V, block=2)

assert np.allclose(out1, out2, atol=1e-8)
print(out2)
```

这段代码体现了三个关键点：

| 步骤 | 标准 Attention | FlashAttention 风格 |
|---|---|---|
| 分数计算 | 直接生成完整 $N \times N$ | 只生成 tile 内局部 scores |
| softmax | 对整行一次完成 | 用在线更新合并多个 tile |
| 输出计算 | 概率矩阵再乘 $V$ | 每处理一个 tile 就直接累加到 $O$ |

如果写成伪代码，结构会更接近 GPU kernel：

```python
for each Q_block in Q:
    load Q_block to SRAM
    init O_block, m_block, l_block

    for each KV_block in K, V:
        load K_block, V_block to SRAM
        S_block = Q_block @ K_block.T
        update m_block, l_block with online softmax
        O_block = update(O_block, S_block, V_block)

    write O_block back to HBM
```

这里“在线 softmax”是关键。它的意思是：softmax 不需要一次看到整行所有分数才能算，可以维护历史最大值和归一化分母，逐块合并结果。这样就允许系统在没保存完整 attention 行的情况下得到精确输出。

---

## 工程权衡与常见坑

FlashAttention 不是“块越大越好”。tile 大小是一个典型的工程平衡问题。

| $B_c$ | SRAM 占用 | tile 数量 | KV 重复读取 | 典型问题 |
|---|---|---|---|---|
| 64 | 低 | 多 | 高 | IO 次数偏多，吞吐不高 |
| 128 | 中 | 适中 | 适中 | 常见平衡点 |
| 256 | 高 | 少 | 低 | 容易 shared memory overflow 或寄存器压力过大 |

第一个坑是共享内存不够。很多 GPU 一个 SM 可用的 shared memory 只有不到 256KB，实际还要和寄存器、warp 调度资源一起竞争。你按 $M/(4d)$ 粗算出来能放下，不代表编译后 occupancy 仍然合理。

第二个坑是 register pressure，也就是“每个线程手里临时变量太多”。白话说，线程为了少访存，把更多数据放寄存器，但寄存器一多，并发线程数就下降，最终可能让吞吐反而变差。

第三个坑是 softmax 统计量空间。除了 $Q/K/V/O$，还要存：

- 每行最大值
- 每行归一化系数
- 局部累加缓冲

很多实现初看只算四个主块容量，结果上线就发现 kernel 资源超了。

第四个坑是因果 mask 和变长序列。真实模型里常常不是完整双向 attention，而是 causal attention。tile 到右上角时会出现部分 block 无效，需要在 kernel 内正确 mask，否则会引入数值错误。

第五个坑是数值稳定性。在线 softmax 必须维护：

$$
m_i = \max_j s_{ij}
$$

并用重标定形式累计分母和输出，否则长序列下很容易溢出。FlashAttention 的难点之一就在于：它不是近似算法，但要在“分块”和“数值稳定”之间同时成立。

一个常见误区是“把块改成 256 应该更快，因为块更大”。这不一定。若 256 让 SRAM 溢出，系统就会降 occupancy，甚至无法启动 kernel。相反，若改成 64，虽然每块更安全，但 tile 数翻倍，KV 被重复加载更多次，HBM IO 又回来了。

---

## 替代方案与适用边界

FlashAttention 解决的是 IO 问题，不是所有 attention 问题。

有些替代方案主要减少计算量，有些主要减少参数量，但它们不一定优化了 HBM 访问模式。

| 方法 | 主要优化对象 | 是否显式针对 IO | 优点 | 局限 |
|---|---|---|---|---|
| 标准 Attention | 无 | 否 | 实现简单，通用性强 | 长上下文 IO 和显存压力大 |
| FlashAttention | HBM IO | 是 | 精确 attention，长上下文高效 | 依赖 GPU kernel 实现，工程复杂 |
| Block-Sparse Attention | 计算量 + 部分 IO | 部分 | 对局部结构数据有效 | 稀疏模式设计复杂，精度依赖任务 |
| Linformer | 降低 $N^2$ 计算 | 否 | 理论复杂度更低 | 属于近似方法，可能损伤精度 |
| xFormers/融合算子 | 工程加速 | 部分 | 接入方便 | 不一定达到 FlashAttention 的 IO 最优 |

新手最容易混淆的一点是：Linformer 之类的方法通过低秩近似减少计算规模，但并不等价于 IO-aware。也就是说，计算量少了，不代表 HBM 读写已经最优。FlashAttention 则是在“仍然算精确 attention”的前提下，把重读写压到 SRAM 内解决。

它的适用边界也很明确：

- 适合：长上下文训练、长上下文推理、GPU 上的 Transformer 主路径
- 一般：中等长度上下文，但仍希望提升吞吐
- 不适合：CPU 场景、极短序列、没有高效 kernel 支持的环境

真实工程里，如果上下文只有 128 或 256，attention 矩阵本来就不大，FlashAttention 的额外 kernel 复杂度未必值得；但当上下文来到 8K、16K、32K 时，收益通常会迅速变得明显。

---

## 参考资料

1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
   https://arxiv.org/abs/2205.14135

2. Hugging Face: Flash Attention Basics / The Tiling Strategy  
   https://huggingface.co/blog/atharv6f/flash-attention-basics

3. Emergent Mind: FlashAttention 论文解读  
   https://www.emergentmind.com/articles/2205.14135

4. Emergent Mind: Block-Sparse FlashAttention / IO-Awareness and Tiling  
   https://www.emergentmind.com/topics/block-sparse-flashattention

5. Memory-efficient GPU tiling explanation for FlashAttention  
   https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling

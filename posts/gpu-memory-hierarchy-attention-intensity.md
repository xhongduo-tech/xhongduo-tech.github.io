## 核心结论

Attention 的算术强度，英文是 arithmetic intensity，记作 AI，定义是“每搬运 1 字节数据，能够完成多少浮点运算”。它是判断一个 kernel 更像“被算力限制”还是“被带宽限制”的直接指标。

对单头、head dimension 为 $d$、序列长度为 $N$ 的标准 scaled dot-product attention，核心计算量近似为

$$
\text{FLOPs} \approx 4N^2d
$$

这里的两项主成本分别来自：

- $QK^\top$，成本约 $2N^2d$
- $PV$，成本约 $2N^2d$

如果只看理想化的最小 HBM 流量，只把 $Q,K,V$ 从 HBM 读入，并把最终输出 $O$ 写回，那么总流量是 $O(Nd)$，因此有

$$
\text{AI}=\Theta(N)
$$

若按 FP16 的 2 字节精确估算，并把 $Q,K,V,O$ 各读写一次计入，可写成

$$
\text{AI}_{\text{ideal}}
\approx
\frac{4N^2d}{8Nd}
=
\frac{N}{2}\ \text{FLOP/Byte}
$$

这说明“AI 约等于 $N$”说的是增长趋势，不是唯一常数。常数项取决于：

- 数据类型是 FP16、BF16 还是 FP32
- 是否把中间结果写回 HBM
- softmax、mask、dropout、transpose 等步骤是否融合
- kernel 是否存在额外的读写与同步开销

真实 kernel 通常比这个理想值低，因为标准 attention 往往会把 $S$ 和 $P$ 这样的中间矩阵显式写回 HBM。这样一来，HBM 流量不再是 $O(Nd)$，而会接近 $O(N^2)$，于是有效 AI 被明显拉低。

以 NVIDIA A100 40GB 为参照，FP16 Tensor Core 峰值约 312 TFLOPS，HBM 带宽约 1.555 TB/s，对应 roofline 拐点约为

$$
I^\*
\approx
\frac{312}{1.555}
\approx
200\ \text{FLOP/Byte}
$$

这个拐点的含义很直接：

- 当有效 AI 远低于 200 时，kernel 更可能是 memory-bound
- 当有效 AI 高于这个量级时，kernel 才更可能逐步转为 compute-bound

因此，短序列 attention 即使 FLOPs 总量不高，也可能很慢。原因不是“乘加太多”，而是“显存搬运占了主时间”。这也是 FlashAttention 能成立的核心背景：它不改变 attention 的精确数学结果，主要改变的是数据流，把大量原本要写回 HBM 的中间态留在片上完成。

| 项目 | 标准 Attention | FlashAttention |
|---|---:|---:|
| 核心 FLOPs | $4N^2d$ | $4N^2d$ |
| 是否改变数学结果 | 否 | 否，仍是 exact attention |
| HBM 中间态 | 通常显式写回 $S,P$ | 不写回完整 $S,P$ |
| HBM 流量趋势 | 容易被 $N^2$ 中间态放大 | 尽量压回 $O(Nd)$ 主项 |
| 短序列瓶颈 | 常见为带宽和 launch overhead | 仍可能带宽受限，但显著缓解 |
| 长序列瓶颈 | 部分场景转向算力 | 更接近 Tensor Core 上限 |

玩具例子：设 $N=256,d=128$，则核心主乘法 FLOPs 约为

$$
4\times 256^2\times 128
=
33{,}554{,}432
\approx 3.36\times 10^7
$$

这个数量本身并不夸张。但如果实现里把 $N\times N$ 的 score 矩阵和 softmax 概率矩阵都写回 HBM，再读出来继续算，时间就会主要花在数据搬运，而不是乘加。

真实工程例子：在 GPT 类模型训练里，单层 attention 会被拆成很多 head。单个 head 的 matmul 常常不够大，而 softmax、mask、transpose、dropout 又可能拆成多个 kernel。此时即使理论 FLOPs 不大，Tensor Core 利用率仍然可能很低，因为流水线被 HBM 带宽、kernel launch 和中间张量读写一起拖慢。

---

## 问题定义与边界

先固定讨论边界，避免把几类不同问题混在一起。

本文讨论的是单层、多头 self-attention 中“单个 head 的核心计算”。记

- $Q,K,V \in \mathbb{R}^{N\times d}$
- score 矩阵 $S=QK^\top \in \mathbb{R}^{N\times N}$
- 概率矩阵 $P=\text{softmax}(S) \in \mathbb{R}^{N\times N}$
- 输出 $O=PV \in \mathbb{R}^{N\times d}$

其中：

- $N$ 是序列长度，表示这次一起处理多少个 token
- $d$ 是单个 attention head 的通道宽度，也就是每个 token 在该 head 内部的向量维度

为了防止术语堆在一起，先把几个常见词说清楚：

| 术语 | 含义 | 在本文中的角色 |
|---|---|---|
| FLOPs | 浮点运算次数 | 描述“算了多少” |
| Bytes | 数据搬运字节数 | 描述“搬了多少” |
| Arithmetic Intensity | FLOPs / Bytes | 描述“每搬 1 字节能算多少” |
| Memory-bound | 受带宽限制 | 算力还没吃满，主要在等数据 |
| Compute-bound | 受算力限制 | 数据够快，主要卡在乘加吞吐 |
| Materialize | 将中间结果真实写成显存张量 | attention 中通常指写回 $S$ 或 $P$ |
| Tile | 切成能放进片上缓存的小块 | FlashAttention 的基本组织单位 |

标准 attention 的问题不在数学公式，而在数据路径。数学上它只是：

1. 计算相似度 $QK^\top$
2. 对每一行做 softmax
3. 用 softmax 权重对 $V$ 加权求和

真正昂贵的是工程实现。如果把 $S$ 和 $P$ 两个 $N\times N$ 中间矩阵都显式写回 HBM，读写成本很容易压过计算成本。

A100 这类 GPU 的内存是分层的。这个层次不是“背景知识”，而是 attention 性能分析的基础，因为你必须先知道数据可能待在哪里。

| 存储层级 | 位置 | 典型容量 | 访问特征 | 对 Attention 的意义 |
|---|---|---:|---|---|
| Register | 每个线程私有 | 最小 | 最快 | 存最热的标量、局部向量、累计状态 |
| Shared Memory / SRAM / L1 | SM 内部 | A100 上统一 L1/Shared 最多 192KB，单 block 可用 shared 最高约 163KB | 很快 | 放 tile 和片上中间结果 |
| L2 | 全卡共享 | 40MB | 比 HBM 快，但不是私有缓存 | 跨 block 或跨 SM 的有限复用 |
| HBM | 显卡外部主存 | 40GB/80GB | 容量最大，延迟最高 | 放模型主张量与最终结果 |

这四层可以理解成四种“距离不同的仓库”：

- Register 和 Shared Memory 是近仓，容量小但拿货快
- L2 是中仓，能缓冲一部分热点数据
- HBM 是远仓，容量大但每次搬运都贵

attention kernel 的设计，本质上就是决定：

- 哪些数据必须进近仓
- 哪些中间态可以在片上直接消费掉
- 哪些结果不得不回写到 HBM

为什么标准 attention 在短序列也慢？原因不是“短序列就一定轻松”，而是短序列对应的小矩阵经常不足以把 Tensor Core 跑满。比如 $N=256,d=64$ 时，单次 matmul 的规模不算大，但 softmax、mask、layout 转换、中间张量读写这些动作仍然一个不少。你实际得到的不是“一个饱和的大 GEMM”，而是一串小 kernel 轮流访问 HBM。

本文还需要明确两个边界：

1. 只讨论精确 attention，不讨论近似 attention  
   也就是说，FlashAttention 优化的是 IO 路径，不是把数学结果近似掉。

2. 只讨论单卡上的单层核心机制  
   不展开张量并行、流水并行、序列并行、多 GPU KV 分片等系统级问题。

这个边界很重要。否则容易把“attention 数学本身的复杂度”与“分布式训练系统的通信复杂度”混为一谈。

---

## 核心机制与推导

先从 FLOPs 推导开始。

### 1. 核心计算量为什么是 $4N^2d$

矩阵 $QK^\top$ 的形状是 $(N,d)\times(d,N)$。按密集矩阵乘法估算，乘加总成本量级是

$$
2N^2d
$$

这里的系数 2 来自“一次乘法加一次加法”这类常见 GEMM 记法。随后矩阵 $PV$ 的形状是 $(N,N)\times(N,d)$，同样成本约为

$$
2N^2d
$$

因此 attention 的两次主乘法合起来是

$$
\text{FLOPs}_{\text{core}}
\approx
4N^2d
$$

注意，这里还没有把以下部分算进去：

- softmax 的指数、归约、除法
- mask 应用
- scale（例如除以 $\sqrt d$）
- dropout
- layout 转换

这些部分通常不会改变主阶数量级，但会影响真实运行时间，尤其是在短序列或小 batch 情况下。

### 2. 理想情况下的 HBM 流量为什么是 $O(Nd)$

若完全不把 $S$ 和 $P$ 落到 HBM，而只：

- 从 HBM 读取 $Q,K,V$
- 把最终输出 $O$ 写回 HBM

那么理想化最少流量为

$$
\text{Bytes}_{\text{ideal}}
\approx
(Q+K+V+O)\times \text{bytes/elem}
=
4Nd\times \text{bytes/elem}
$$

于是算术强度为

$$
\text{AI}_{\text{ideal}}
\approx
\frac{4N^2d}{4Nd\times \text{bytes/elem}}
=
\frac{N}{\text{bytes/elem}}
$$

对于 FP16，每个元素 2 字节，因此

$$
\text{AI}_{\text{ideal}}
\approx
\frac{N}{2}\ \text{FLOP/Byte}
$$

这正是“Attention 的 AI 随序列长度近似线性增长”的来源。

### 3. 为什么标准实现的真实 AI 会更低

问题在于，标准 attention 往往不会做到“只读写 $Q,K,V,O$”。它常见的数据路径是：

1. 读入 $Q,K$
2. 计算 $S=QK^\top$
3. 将 $S$ 写回 HBM
4. 再读回 $S$，应用 mask、scale、softmax
5. 将 $P$ 写回 HBM
6. 再读回 $P$ 和 $V$
7. 计算输出 $O=PV$
8. 写回 $O$

如果把 $S$ 和 $P$ 都视为需要 materialize 的中间态，那么额外流量大约会包含两个 $N\times N$ 级别的大矩阵。粗略写成主项就是

$$
\text{Bytes}_{\text{std}}
=
O(Nd)+O(N^2)
$$

于是有效 AI 会变成

$$
\text{AI}_{\text{std}}
\approx
\frac{4N^2d}{c_1Nd+c_2N^2}
$$

这里 $c_1,c_2$ 是与数据类型、读写次数和实现细节相关的常数。这个式子最重要的不是精确系数，而是趋势：

- 理想 FlashAttention 路径尽量保留 $O(Nd)$ 流量主项
- 标准 materialized 路径会引入 $O(N^2)$ 额外流量主项

这就是两者在 roofline 图上位置明显不同的原因。

### 4. 用数字读一下数量级

下表给出理想化估算。这里的 AI 是“只读写 $Q,K,V,O$”时的上界，不代表标准实现真实达到的数值。

| $N$ | $d$ | 理想核心 FLOPs $4N^2d$ | 理想 Bytes（FP16）$8Nd$ | 理想 AI（FP16）$N/2$ | 对 A100 的直观判断 |
|---:|---:|---:|---:|---:|---|
| 128 | 128 | 8.39 MFLOPs | 131 KB | 64 FLOP/Byte | 明显偏 memory-bound |
| 256 | 128 | 33.55 MFLOPs | 262 KB | 128 FLOP/Byte | 仍偏 memory-bound |
| 512 | 128 | 134.22 MFLOPs | 524 KB | 256 FLOP/Byte | 理想值已接近跨过 roofline |
| 1024 | 128 | 536.87 MFLOPs | 1.0 MB | 512 FLOP/Byte | 理想化下可进入 compute-bound 区域 |
| 4096 | 128 | 8.59 GFLOPs | 4.0 MB | 2048 FLOP/Byte | 更容易转向 compute-bound |

这张表要这样读：

- 它反映的是趋势，而不是标准实现的真实性能
- 即使理想 AI 已高于 roofline 拐点，真实实现仍可能因为额外 IO 而被拉回 memory-bound
- 因此“$N=1024$ 就一定 compute-bound”这种说法并不严谨，更准确的表述是“1K token 量级开始有机会跨过拐点，但取决于实现是否足够 IO-aware”

### 5. FlashAttention 的关键机制是什么

FlashAttention 不是换了公式，而是换了数据流。它把 attention 的计算拆成多个 tile，也就是把大矩阵切成能放进片上缓存的小块。

对每个 query block，它会循环扫描多个 key/value block。在每一轮中，只把当前所需的 $K,V$ 子块拉进 SRAM，在片上完成：

- 该子块的 score 计算
- mask 和 scale
- softmax 的逐块更新
- 与 $V$ 的乘法累积

完成后立刻丢弃这个 score 子块，不把完整 $S$ 或 $P$ 写回 HBM。

### 6. 为什么 online softmax 能保证精确结果

难点在于 softmax 看起来需要整行数据。对于某一行 score，普通 softmax 形式是

$$
\text{softmax}(s_i)_t
=
\frac{e^{s_{it}}}{\sum_{u=1}^{N} e^{s_{iu}}}
$$

为了数值稳定，通常改写为减去行最大值：

$$
\text{softmax}(s_i)_t
=
\frac{e^{s_{it}-m_i}}{\sum_{u=1}^{N} e^{s_{iu}-m_i}},
\qquad
m_i=\max_u s_{iu}
$$

表面看，你似乎必须先拿到整行所有元素，才能知道 $m_i$ 和分母。但 online softmax 的做法是：逐块扫描时，只维护每一行两个运行状态：

- $m_i$：当前已扫描部分的最大值
- $\ell_i$：按该最大值重标定后的指数和

假设当前扫描到第 $j$ 个 block，这个 block 内该行的局部最大值是 $m_{i,j}$。更新规则是

$$
m_i^{\text{new}}=\max(m_i^{\text{old}}, m_{i,j})
$$

$$
\ell_i^{\text{new}}
=
\ell_i^{\text{old}} e^{m_i^{\text{old}}-m_i^{\text{new}}}
+
\sum_{t\in \text{block }j} e^{s_{it}-m_i^{\text{new}}}
$$

输出向量也要同步重标定。若把该行输出写成 $O_i$，则有

$$
O_i^{\text{new}}
=
\frac{
O_i^{\text{old}}\ell_i^{\text{old}}e^{m_i^{\text{old}}-m_i^{\text{new}}}
+
\sum_{t\in \text{block }j} e^{s_{it}-m_i^{\text{new}}}V_t
}{
\ell_i^{\text{new}}
}
$$

这个更新的意义是：虽然你没有一次拿到整行 $N$ 个分数，但你在每次块更新后都保持了“到目前为止的精确 softmax 状态”。因此最终扫完整行时，得到的就是精确结果，而不是近似值。

### 7. 一个直观的小例子

设 $N=8$，block size 为 4，那么某一行 query 只需分两轮扫描：

- 第 1 轮看前 4 个 key
- 第 2 轮看后 4 个 key

在第 1 轮结束时，你记录该行当前的：

- 最大值 $m$
- 指数和 $\ell$
- 输出累计向量 $O$

到了第 2 轮，如果出现更大的 score，你就把第 1 轮累计的结果按新的最大值重新缩放，再合并第 2 轮的贡献。这样直到扫描完所有 block，都不需要把一整行 $S$ 写出去。

### 8. 一个更贴近工程的例子

设 $N=256,d=128$，tile 大小选为 $64\times 64$。那么：

- 一共有 4 个 query tile
- 每个 query tile 要扫描 4 个 KV tile
- 每次只在片上构造一个 $64\times64$ 的 score 子块

也就是说：

- 不会生成完整的 $256\times256$ score 矩阵
- 不会生成完整的 $256\times256$ softmax 概率矩阵
- 在 HBM 中常驻的只有输入张量和最终输出

这正是 FlashAttention 能把 HBM 主流量重新压回线性级别的原因。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，用来验证 online softmax 的数学机制。它不模拟 GPU，也不追求性能，只验证一件事：

**在不保存完整 $S$ 的前提下，逐块更新的结果可以与标准 attention 精确一致。**

代码包含三部分：

1. 标准 attention 的单行实现
2. 使用 online softmax 的单行实现
3. 一个小测试，比较两者输出误差并验证 AI 公式

```python
import math
import random
from typing import List


Vector = List[float]
Matrix = List[Vector]


def dot(x: Vector, y: Vector) -> float:
    return sum(a * b for a, b in zip(x, y))


def standard_attention_row(q: Vector, K: Matrix, V: Matrix) -> Vector:
    """标准单行 attention：先算完整 scores，再做 softmax，再加权求和。"""
    scores = [dot(q, k) for k in K]

    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = sum(exps)

    probs = [e / denom for e in exps]

    out = [0.0 for _ in range(len(V[0]))]
    for p, v in zip(probs, V):
        for j, value in enumerate(v):
            out[j] += p * value
    return out


def flash_attention_row(q: Vector, K: Matrix, V: Matrix, block_size: int = 2) -> Vector:
    """使用 online softmax 的单行 attention，不保存完整 score 矩阵。"""
    if len(K) != len(V):
        raise ValueError("K and V must have the same number of rows")
    if len(K) == 0:
        raise ValueError("K and V must be non-empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    value_dim = len(V[0])

    # online softmax 的运行状态
    m = float("-inf")   # 当前已见最大值
    l = 0.0             # 当前已见 exp 和（按 m 重标定）
    acc = [0.0 for _ in range(value_dim)]  # 未归一化的输出累计量

    for start in range(0, len(K), block_size):
        K_blk = K[start:start + block_size]
        V_blk = V[start:start + block_size]

        scores_blk = [dot(q, k) for k in K_blk]
        m_blk = max(scores_blk)
        m_new = max(m, m_blk)

        old_scale = 0.0 if m == float("-inf") else math.exp(m - m_new)
        exp_blk = [math.exp(s - m_new) for s in scores_blk]

        l_new = l * old_scale + sum(exp_blk)

        acc_new = [x * old_scale for x in acc]
        for weight, v in zip(exp_blk, V_blk):
            for j, value in enumerate(v):
                acc_new[j] += weight * value

        m = m_new
        l = l_new
        acc = acc_new

    return [x / l for x in acc]


def ideal_ai_fp16(N: int) -> float:
    """
    理想化 FP16 attention 的 arithmetic intensity:
    FLOPs = 4N^2 d
    Bytes = (Q + K + V + O) * 2 bytes = 8Nd
    AI = FLOPs / Bytes = N / 2
    """
    return N / 2.0


def max_abs_diff(x: Vector, y: Vector) -> float:
    return max(abs(a - b) for a, b in zip(x, y))


def demo() -> None:
    random.seed(0)

    N = 6
    d = 4
    value_dim = 4

    q = [random.uniform(-1.0, 1.0) for _ in range(d)]
    K = [[random.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(N)]
    V = [[random.uniform(-1.0, 1.0) for _ in range(value_dim)] for _ in range(N)]

    out_std = standard_attention_row(q, K, V)
    out_flash = flash_attention_row(q, K, V, block_size=2)

    err = max_abs_diff(out_std, out_flash)

    print("standard:", [round(x, 8) for x in out_std])
    print("flash   :", [round(x, 8) for x in out_flash])
    print("max abs error:", err)

    assert err < 1e-12

    print("ideal_ai_fp16(256)  =", ideal_ai_fp16(256))
    print("ideal_ai_fp16(1024) =", ideal_ai_fp16(1024))

    assert ideal_ai_fp16(256) == 128.0
    assert ideal_ai_fp16(1024) == 512.0


if __name__ == "__main__":
    demo()
```

这段代码的关键点有两个：

- `standard_attention_row` 会先拿到完整 `scores`
- `flash_attention_row` 只按块处理 `scores_blk`，并用 `m, l, acc` 维护全局正确状态

如果运行它，两者输出会一致，误差只来自浮点舍入。

如果把这个思路翻译为 GPU kernel，伪代码大致如下：

```python
for Q_block in Q_tiles:                     # Q_block 尽量常驻 SRAM / registers
    load(Q_block)
    init row_max, row_sum, O_accum

    for KV_block in KV_tiles:              # K/V block 流式载入片上
        load(K_block, V_block)
        S_block = Q_block @ K_block.T      # Tensor Core 主算子
        apply_scale_and_mask(S_block)
        update_online_softmax(row_max, row_sum, O_accum, S_block, V_block)

    store(O_accum)                         # 只把最终输出写回 HBM
```

这个伪代码真正重要的不是循环写法，而是“每类数据放在哪一层存储”。

| 数据/步骤 | 最好停留的位置 | 原因 |
|---|---|---|
| $Q$ 子块 | Shared Memory / Register | 会被多个 KV tile 重用 |
| $K,V$ 子块 | Shared Memory | 当前扫描轮次频繁访问 |
| $S$ 子块 | Register / Shared Memory | 用完即丢，不应 materialize 到 HBM |
| $m,\ell$ | Register | 每行少量标量，最热状态 |
| 输出累计 $O_{\text{accum}}$ | Register 优先，必要时部分 shared | 高频更新，不能反复回写 HBM |
| 最终 $O$ | HBM | query block 完成后一次写回 |

为了帮助新手把“代码结构”和“性能含义”对应起来，下表把标准实现和 FlashAttention 风格的主要差异直接并列：

| 维度 | 标准实现 | FlashAttention 风格 |
|---|---|---|
| score 计算后 | 常写回完整 $S$ | 子块上即时消费 |
| softmax | 常作为独立阶段 | 融入块扫描 |
| 概率矩阵 $P$ | 常写回完整矩阵 | 不生成完整矩阵 |
| 输出累计 | 依赖读回 $P$ 再乘 $V$ | 在块内直接累计 |
| HBM 访问模式 | 多次往返 | 以流式读入和最终写回为主 |

工程里通常还会把以下步骤进一步融合进同一个 kernel：

- scale
- causal mask / padding mask
- softmax
- dropout
- $PV$ 乘法

fusion 的意思不是“代码写在一个函数里”，而是“原本会拆成多次 HBM 往返的步骤，被合成一次片上流水线”。这对短序列尤其关键，因为短序列除了受带宽限制，还容易被 kernel launch overhead 拖慢。

---

## 工程权衡与常见坑

第一个坑是重复 materialize 中间态。对 attention 来说，最昂贵的 materialize 通常就是把 $S$ 和 $P$ 真的写成显存张量。只要这一步发生，HBM 流量就很容易接近 $N^2$ 级，AI 会明显下降。

可以把这个问题写成一个简单对照：

| 做法 | 后果 |
|---|---|
| $S$ 只在片上存在 | 计算完子块即可丢弃 |
| $S$ 写回 HBM | 之后 softmax 又要再读一遍 |
| $P$ 只作为片上权重参与累计 | 无需形成完整概率矩阵 |
| $P$ 写回 HBM | 乘 $V$ 前又要再读一遍 |

第二个坑是 tile 过大。很多人第一次接触 tiled kernel 时会直觉地认为“tile 越大，复用越多，所以越快”。这只说对了一半。tile 过大时，会立刻碰到 shared memory 和 register 的硬上限，结果可能是：

- occupancy 降低
- 可并发的 block 数减少
- 编译器溢出到 local memory
- 片上双缓冲空间不够

在 A100 上，单个 block 的 shared memory 上限大约为 163KB。设计 tile 时通常要满足类似约束：

$$
\text{mem}(Q_{\text{tile}})
+
\text{mem}(K_{\text{tile}})
+
\text{mem}(V_{\text{tile}})
+
\text{mem}(\text{scratch})
\le
\alpha \cdot \text{available\_shmem}
$$

其中 $\alpha$ 常取小于 1 的经验安全系数，例如 0.8 到 0.9，用于给：

- 对齐开销
- 中间缓冲
- 双缓冲
- warp 级归约临时区

留出余量。

第三个坑是只盯着 matmul，不看非 matmul 部分。很多新手会认为 attention 的优化等价于“把 GEMM 跑快”。这是不完整的。attention 里真正容易拖慢流水线的，恰恰常常是这些部分：

- softmax 的归约和指数
- mask 应用
- dropout
- layout 转换
- block 间同步
- shared memory 往返

也就是说，attention 的性能问题并不只是“Tensor Core 利用率够不够高”，而是“非 matmul 部分是否把整体 roofline 位置拉偏了”。

第四个坑是把 L2 当成稳定缓存。L2 是全卡共享资源，不是私有二级 SRAM。下面几种情况都会让 L2 复用明显变差：

- 多头并发很多
- batch 变大
- 还有别的 kernel 同时运行
- 序列长度继续拉长
- KV cache 很大，工作集超过 L2 容量

因此，工程上不能把“L2 可能命中”当成设计前提。更稳妥的思路是：让关键子流程尽量在 Shared Memory 和 Register 内闭环完成。

第五个坑是 roofline 只看理论峰值，不看有效值。A100 的理论拐点可粗看为

$$
I^\*\approx 200\ \text{FLOP/Byte}
$$

但真实有效拐点会受很多因素影响：

- 实际 kernel 没法达到 312 TFLOPS 峰值
- 实际 HBM 带宽未必能完全打满
- 访存不是完全理想顺序
- warp 调度与寄存器压力会降低吞吐

因此“超过 200 就一定 compute-bound”也不严谨。更好的用法是把它当成一条判断基线，而不是精确边界。

下面把常见坑集中整理成一张表：

| 常见坑 | 直接后果 | 典型症状 | 规避方式 |
|---|---|---|---|
| 显式写回 $S,P$ | HBM 流量爆炸 | profiler 中 global memory bytes 很高 | 用 on-chip softmax，避免中间态落盘 |
| tile 过大 | occupancy 降低 | SM 利用率下降，shared memory 紧张 | 依据 shared memory 与寄存器上限反推 tile |
| softmax 独立成 kernel | 读写次数增加 | kernel 数量多，launch overhead 明显 | 与 matmul、mask、dropout 融合 |
| 过度依赖 L2 命中 | 长序列性能波动 | 序列拉长后吞吐掉得快 | 让关键路径尽量在 SRAM 内完成 |
| 只看 FLOPs 不看 Bytes | 错判瓶颈 | 以为算力不足，实际是带宽不足 | 用 roofline 同时分析运算与流量 |
| 忽略寄存器压力 | spilling 到 local memory | 指令数和内存访问异常增多 | 平衡 tile、warp 分工与累计状态大小 |

真实工程例子：在 LLM 推理里，prefill 阶段通常是“大 $N_q$ 乘大 $N_k$”，而 decode 阶段往往是“小 $N_q$ 扫长 KV cache”。后者并不是完整的方阵 self-attention，而更接近“少量 query 读取很长的历史 KV”。这时即使用了 FlashAttention 风格优化，瓶颈也常常更多落在 KV cache 的读取路径上，而不是训练时那套完整 $N\times N$ 推导本身。

因此，训练场景和推理场景必须分开分析：

| 场景 | 常见形状 | 更可能的瓶颈 |
|---|---|---|
| Training / Prefill | $N_q \approx N_k \approx N$ | attention 主体的 IO 与 Tensor Core 利用率 |
| Decode | $N_q$ 很小，$N_k$ 持续增长 | KV cache 带宽与缓存命中 |

---

## 替代方案与适用边界

FlashAttention 不是唯一方案，但它是当前“精确 attention + IO 优化”最常见、最有代表性的基线。

先把几条路线分清：

| 方案 | 主要思想 | 是否保持 exact attention | 优势 | 边界 |
|---|---|---:|---|---|
| 标准 Attention | 按数学步骤直接实现 | 是 | 简单、通用、易教学 | 中间态多，IO 成本高 |
| FlashAttention-1 | IO-aware + online softmax | 是 | 把 HBM 流量从中间态主导改为流式主导 | kernel 设计复杂 |
| FlashAttention-2 | 改 work partitioning 与并行策略 | 是 | 更高 occupancy，更少非 matmul 开销 | 仍受寄存器和 shared memory 约束 |
| FlashAttention-3 | 利用 Hopper 新硬件特性 | 是 | 在 H100 上更高吞吐，支持更激进流水 | 硬件门槛高 |
| 近似 Attention | 改写计算图或采样/稀疏化 | 否 | 可进一步降复杂度 | 数学结果改变 |

### 1. FlashAttention-1 的边界

FlashAttention-1 解决的是核心 IO 问题：不把 $S,P$ 写回 HBM，而是在片上完成 softmax 和加权累积。它最本质的收益是：

- 降低 HBM 流量
- 提高有效 AI
- 减少中间张量占用

但它并没有消除所有瓶颈。例如：

- 很短序列时，launch overhead 仍然显著
- 特殊 mask、padding 处理可能影响并行效率
- 极长序列时，L2/HBM 访问模式依然会成为新瓶颈

### 2. FlashAttention-2 的边界

FlashAttention-2 的重点不再只是“有没有在线 softmax”，而是“线程块和 warp 怎么分工更合理”。它通过改进 work partitioning，让 attention 更像一个高效 GEMM，而不是一个带大量同步与 shared memory 往返的复杂内核。

对新手来说，可以把它理解成：

- FlashAttention-1 先解决“不要把中间态写回 HBM”
- FlashAttention-2 再解决“即使不写回 HBM，片上也要分工合理，否则还是浪费”

### 3. FlashAttention-3 的边界

FlashAttention-3 主要面向 Hopper 架构，例如 H100。它会更深地利用：

- TMA
- 异步流水
- 新 Tensor Core 特性
- 更适合 FP8 的执行路径

因此它的收益高度依赖硬件。把它直接类比到 A100 或更老的 GPU 上，并不准确。

### 4. 与近似 attention 的关系

近似 attention 的目标通常是进一步降低时间或显存复杂度，例如利用稀疏、低秩、局部窗口或检索机制。它和 FlashAttention 不属于同一条路线：

- FlashAttention 优化的是实现路径，保留 exact attention
- 近似 attention 优化的是数学问题本身，通常会改变结果

因此，二者的适用问题不同。一个常见误区是把两者混称为“更快的 attention”。这种说法太宽泛，必须先区分：

- 你是想保留完全相同的输出
- 还是允许引入可控误差来换速度和显存

### 5. 按序列长度看适用边界

可以用一个更接近工程的视角总结：

| 序列长度区间 | 常见状态 | 优化重点 |
|---|---|---|
| 很短，例如 $N\le 256$ | 常见 memory-bound，且 launch overhead 显著 | 融合 kernel、减少中间态、减少调度开销 |
| 中等，例如 $N\approx 1\text{K}$ | 可能接近 roofline 转折区 | IO-aware 实现是否成熟成为关键 |
| 很长，例如 $N\ge 8\text{K}$ | attention 主体更大，缓存层次压力上升 | tile 设计、KV 布局、并行切分同时重要 |
| 超长，例如 $64\text{K}$ 或 $128\text{K}$ | 单卡缓存和显存访问模式恶化 | 需要系统级方案，不只是替换 kernel |

玩具例子：在教学代码或小模型实验里，$N=512,d=64$ 时直接用框架内置标准 attention 也完全合理。因为这个场景的主要目标可能是验证正确性、快速迭代，而不是把每个 kernel 压到极限。

真实工程例子：在 A100 上训练 8K 上下文的 GPT 类模型时，FlashAttention-2 往往已经能带来明显收益；但当上下文继续扩到 128K，问题就不再只是“有没有 FlashAttention”，而是：

- KV cache 如何布局
- 序列如何分块
- 是否需要序列并行
- 是否需要跨 GPU 分片

也就是说，FlashAttention 是高性能 exact attention 的基础组件，但不是超长上下文系统设计的全部答案。

---

## 参考资料

1. NVIDIA A100 官方规格页：A100 40GB/80GB 的 Tensor Core 峰值与 HBM 带宽  
   https://www.nvidia.com/en-us/data-center/a100/

2. NVIDIA Ampere Tuning Guide：Ampere 架构下 unified L1/shared memory、shared memory 上限、异步拷贝等实现细节  
   https://docs.nvidia.com/cuda/archive/12.0.0/ampere-tuning-guide/index.html

3. NVIDIA Ampere Architecture Whitepaper：A100 的 40MB L2、HBM2 带宽、SM 与缓存组织  
   https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf

4. FlashAttention 原论文：IO-aware exact attention 的基本定义、在线 softmax 机制与 IO 复杂度分析  
   https://arxiv.org/abs/2205.14135

5. FlashAttention-2 论文：改进 work partitioning、减少非 matmul 开销、提升并行效率  
   https://arxiv.org/abs/2307.08691

6. FlashAttention-2 项目说明页：实现思路、吞吐数据与适用背景  
   https://princeton-nlp.github.io/flash-atttention-2/

7. Dao-AILab 官方仓库：工程实现、安装方式、支持硬件与接口说明  
   https://github.com/Dao-AILab/flash-attention

8. 补充阅读：online softmax 与 FlashAttention 分块机制的直观解释  
   https://huggingface.co/blog/atharv6f/flash-attention-online-softmax

9. 补充阅读：roofline model 的经典背景，帮助理解“算力受限”和“带宽受限”的判别方法  
   https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/

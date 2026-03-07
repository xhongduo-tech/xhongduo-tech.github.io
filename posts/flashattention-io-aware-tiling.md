## 核心结论

FlashAttention 的核心不是“把 softmax 写得更快”，而是**把注意力的执行路径改成了 IO 感知的分块流式计算**。这里的 IO 指的不是磁盘读写，而是 GPU 上两级内存之间的数据搬运：

- HBM：显存，容量大，但访问代价高
- SRAM：片上共享内存和寄存器，容量小，但访问极快

标准注意力通常按下面的逻辑执行：

$$
S = QK^\top,\qquad
P = \mathrm{softmax}(S),\qquad
O = PV
$$

这个写法在数学上没有问题，但在工程实现上会诱导出一个关键瓶颈：**长度为 $N$ 的序列会产生一个 $N\times N$ 的中间分数矩阵**。即使你不把它暴露为 Python 张量，底层 kernel 也常常要为它付出接近的 HBM 读写代价。

FlashAttention 的策略是：

1. 把 $Q/K/V$ 按块装入 SRAM
2. 在片上完成这一块的分数计算、softmax 归一化和输出累积
3. 只保留每一行必须延续到下一块的统计量
4. 避免把完整的 $N\times N$ 分数矩阵写回 HBM

因此它优化的重点不是“减少算术公式的项数”，而是**减少最昂贵的显存往返**。

| 项目 | 传统注意力 | FlashAttention |
|---|---|---|
| 是否显式或等价地产生 $N\times N$ 分数矩阵 | 通常会 | 不会 |
| softmax 执行方式 | 全量分数算完后统一做 | 按块在线更新 |
| SRAM 的角色 | 辅助缓存 | 主计算现场 |
| HBM 主要压力来源 | 中间矩阵反复读写 | 只读写必要块和状态 |
| 适合的场景 | 短序列、实现简单优先 | 长上下文、带宽受限场景 |

从 IO 复杂度看，FlashAttention 论文给出的主结论可以概括为：

- 标准注意力的 HBM IO 主项接近 $\mathcal{O}(N^2)$ 级别中间量搬运
- FlashAttention 将 HBM IO 主项降低到与块大小 $M$ 相关的形式，常写为：

$$
\mathcal{O}\left(\frac{N^2 d^2}{M}\right)
$$

这里 $d$ 是单头维度，$M$ 是可用于该 kernel 的片上 SRAM 容量。**$M$ 越大，单次能处理的块越大，HBM 往返就越少。**

一句话概括：**FlashAttention 用更复杂的片上调度，换掉了最贵的显存读写。**

---

## 问题定义与边界

先把问题写清楚。标准缩放点积注意力是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

其中：

| 符号 | 形状 | 含义 |
|---|---|---|
| $Q$ | $N\times d$ | Query，表示当前位置要“查询什么” |
| $K$ | $N\times d$ | Key，表示每个位置可被匹配的“索引” |
| $V$ | $N\times d_v$ | Value，表示被取出的实际内容 |
| $N$ | 标量 | 序列长度 |
| $d$ | 标量 | 单头维度 |
| $d_v$ | 标量 | Value 维度，常与 $d$ 相同 |

把 $QK^\top$ 展开后得到：

$$
S\in\mathbb{R}^{N\times N}
$$

这意味着对每个 query token，都要和所有 key token 计算一次分数。问题从这里开始变得昂贵：

- 计算量随 $N^2$ 增长
- 中间分数矩阵规模也随 $N^2$ 增长
- 当 $N$ 很大时，真正先撞上的常常不是算力，而是内存带宽和中间存储

举一个具体数量级。若 $N=65536$，则：

$$
N^2 = 65536^2 = 4{,}294{,}967{,}296
$$

也就是约 $4.29\times 10^9$ 个分数。若仅以 `fp16` 存储，每个元素 2 字节，那么光这个分数矩阵理论上就需要约：

$$
4.29\times 10^9 \times 2 \approx 8.59\ \text{GB}
$$

这还没有算：

- 多头
- batch
- softmax 中间状态
- dropout / mask
- 反向传播需要保存的额外信息

所以，**问题的实质不是公式写不出来，而是直接实现会把 HBM 带宽和容量迅速吃满。**

FlashAttention 建立在一个明确的硬件边界上：

| 存储层级 | 特点 | 对算法的含义 |
|---|---|---|
| HBM | 大，但慢 | 适合放完整输入输出，不适合频繁回写巨型中间结果 |
| SRAM | 小，但快 | 适合做块内闭环计算 |

常见资料中，A100 的量级常被概括为：

- 单个 SM 可用片上存储量级约数百 KB
- HBM 带宽约 1.5 到 2 TB/s
- 片上 SRAM 总带宽量级远高于 HBM，可达十几到二十 TB/s 级别

这里不需要死记具体数字，真正重要的是结论：**两级内存的速度差足够大，算法必须围绕“少碰 HBM”来设计。**

因此 FlashAttention 采用分块。常见记法是：

$$
B_c=\left\lfloor \frac{M}{4d} \right\rfloor,\qquad
B_r=\min(B_c,d)
$$

其中：

- $B_c$：列块大小，一次处理多少个 key/value
- $B_r$：行块大小，一次处理多少个 query
- $M$：当前 kernel 可用的 SRAM 容量
- 分母中的 $4d$：表示需要同时为 $Q$ 块、$K$ 块、$V$ 块以及输出与统计量预留空间

这不是唯一写法，但足够表达核心思想：**块大小由片上存储反推，而不是任意拍脑袋设定。**

对新手来说，可以把它理解成“分批查表”：

- 不先生成整张“所有 query 对所有 key 的匹配大表”
- 而是拿一小批 query，扫描一小批 key/value
- 扫一块，更新一次结果
- 最终拼出的输出与全量 softmax 数学等价

FlashAttention 的适用边界也需要提前说清楚：

- 它解决的是 IO 主导问题
- 它不改变注意力的数学语义
- 它不是所有长度下都更快
- 当序列较短、并行度较低时，块调度开销可能抵消收益

所以它最适合的不是“任何注意力”，而是**长上下文、显存带宽敏感、需要精确 attention 的场景**。

---

## 核心机制与推导

FlashAttention 的执行顺序可以概括成一个双层循环：

1. 外层遍历 $K/V$ 的列块 $j$
2. 内层遍历 $Q$ 的行块 $i$

每次只把 $(Q_i, K_j, V_j)$ 这组当前需要的数据放入 SRAM，在片上完成三件事：

1. 计算局部分数
2. 在线完成 softmax 合并
3. 立即把这一块对输出的贡献累积进去

写成块级公式：

$$
S_{ij}=\frac{Q_iK_j^\top}{\sqrt d}
$$

这里：

- $Q_i\in\mathbb{R}^{B_r\times d}$
- $K_j\in\mathbb{R}^{B_c\times d}$
- $V_j\in\mathbb{R}^{B_c\times d_v}$
- $S_{ij}\in\mathbb{R}^{B_r\times B_c}$

### 为什么普通 softmax 不能直接照搬

对完整一行分数 $x=(x_1,\dots,x_N)$，softmax 是：

$$
\mathrm{softmax}(x_k)=\frac{e^{x_k}}{\sum_{t=1}^{N} e^{x_t}}
$$

数值稳定实现通常会先减去该行最大值：

$$
\mathrm{softmax}(x_k)=\frac{e^{x_k-m}}{\sum_{t=1}^{N} e^{x_t-m}},
\qquad m=\max_t x_t
$$

问题在于，分块后你一次看不到整行所有分数。第 1 个块只看见这行的一部分，第 2 个块又看到另一部分。也就是说：

- 当前块内的最大值不一定是全局最大值
- 当前块内的指数和也不是最终分母
- 不能把各块各自 softmax 后直接相加

### 在线 softmax 的状态变量

FlashAttention 为每个 query 行维护两个状态：

- $m_i$：到当前为止见过的最大分数
- $\ell_i$：在该最大值坐标系下的指数和

若当前处理的新块为 $S$，则对该行有：

$$
m_i'=\max\left(m_i,\max_k S_{ik}\right)
$$

然后把“旧块累计结果”和“新块局部结果”统一到新的最大值 $m_i'$ 下：

$$
\ell_i'=
\ell_i\cdot e^{m_i-m_i'}
+
\sum_k e^{S_{ik}-m_i'}
$$

这一式是 FlashAttention 能成立的关键。它表达的是：

- 旧分母 originally 以 $m_i$ 为基准
- 新块以 $m_i'$ 为基准
- 合并前必须先把旧分母重新缩放到新坐标系

### 输出为什么也要同步重标定

假设当前行已有输出向量 $O_i$，表示此前处理过的所有块对结果的累积。新块对应的 value 为 $V_j$。那么输出更新为：

$$
O_i'=
\frac{
\ell_i e^{m_i-m_i'} O_i
+
\sum_k e^{S_{ik}-m_i'}V_{j,k}
}{
\ell_i'
}
$$

其中 $V_{j,k}$ 表示当前块第 $k$ 个 key 对应的 value 向量。

这一步很容易被初学者忽略：**旧输出不仅要保留，还要先按新基准重缩放后再参与合并。**  
如果你只更新分母 $\ell_i$，不重标定旧输出 $O_i$，结果就不再等价于完整 softmax。

### 一个两块示例

假设同一行分数被拆成两块：

- 第 1 块最大值是 8
- 第 2 块最大值是 11

处理完第 1 块时，你保存的是“以 8 为基准”的指数和与加权输出。  
当读到第 2 块后，新的全局最大值变成 11，那么第 1 块的旧统计必须先乘上：

$$
e^{8-11}=e^{-3}
$$

这样才能和以 11 为基准的新块结果相加。  
这就是在线 softmax 的本质：**每一块都只贡献局部信息，但所有局部信息始终被折算到同一个全局坐标系中。**

### IO 复杂度为什么会下降

标准注意力的典型执行路径是：

1. 从 HBM 读取 $Q,K$
2. 计算并写回 $S=QK^\top$
3. 再读取 $S$ 做 softmax，写回概率矩阵 $P$
4. 再读取 $P,V$ 得到输出 $O$

即使具体实现做了融合优化，底层也往往绕不开对大规模中间量的反复搬运。于是 HBM 压力与 $N^2$ 中间矩阵强相关。

而 FlashAttention 不物化完整的 $S$ 和 $P$，只做：

- 分块读取 $Q/K/V$
- 在 SRAM 中算完一块的分数、归一化和输出贡献
- 把必要状态写回
- 不把完整 $N\times N$ 中间结果落到 HBM

设列块大小约为：

$$
B_c\approx \frac{M}{4d}
$$

则列块数约为：

$$
T_c\approx \frac{N}{B_c}\approx \frac{4Nd}{M}
$$

每个列块都要扫描所有 query 行块。把块大小代入后，论文分析可整理出 HBM IO 主项：

$$
\mathcal{O}\left(\frac{N^2 d^2}{M}\right)
$$

这个式子的直觉非常明确：

- $N$ 越大，问题越偏向长序列
- $d$ 越大，每块容纳的 token 越少
- $M$ 越大，块越大，需要的 HBM 往返越少

所以 FlashAttention 的收益不是无条件出现的，而是在**$N$ 足够大、HBM 足够贵、$M$ 又足以容纳合理块大小**时最明显。

### 用 A100 量级做直观估算

假设：

- $M=192\text{KiB}=196608$ 字节
- $d=64$

则近似有：

$$
B_c=\left\lfloor \frac{196608}{4\times 64} \right\rfloor=768
$$

再取：

$$
B_r=\min(B_c,d)=64
$$

这表示一次可以处理大致如下规模的数据块：

- 一个 $64\times 64$ 的 $Q$ 块
- 一个 $768\times 64$ 的 $K$ 块
- 一个 $768\times 64$ 的 $V$ 块

直观上，你不再构造完整的 $65536\times 65536$ 注意力图，而是让某个 query 小块在扫描不同的 $K/V$ 块时不断更新自己的：

- 当前最大值 $m$
- 当前分母 $\ell$
- 当前输出 $O$

这就是“IO 感知”的具体落点：**围绕片上内存能装下什么，重写整个计算顺序。**

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，用 CPU 模拟 FlashAttention 的核心思想：

- 按块扫描 $K/V$
- 对每个 $Q$ 块维护在线 softmax 状态
- 最终结果与标准注意力对齐

这个实现不追求速度，目的是把算法逻辑讲清楚。

```python
import math
import numpy as np


def standard_attention(Q, K, V, causal=False):
    """
    Reference implementation of scaled dot-product attention.
    Q: [N, d]
    K: [N, d]
    V: [N, dv]
    """
    n, d = Q.shape
    scores = Q @ K.T / math.sqrt(d)

    if causal:
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    row_max = np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores - row_max)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def flash_attention_toy(Q, K, V, br, bc, causal=False):
    """
    Blocked attention with online softmax.
    This mirrors the FlashAttention update rule on CPU.

    Q: [N, d]
    K: [N, d]
    V: [N, dv]
    br: row block size
    bc: column block size
    """
    n, d = Q.shape
    dv = V.shape[1]

    O = np.zeros((n, dv), dtype=np.float64)
    m = np.full((n,), -np.inf, dtype=np.float64)
    l = np.zeros((n,), dtype=np.float64)

    scale = 1.0 / math.sqrt(d)

    for j in range(0, n, bc):
        j_end = min(j + bc, n)
        Kj = K[j:j_end]
        Vj = V[j:j_end]

        for i in range(0, n, br):
            i_end = min(i + br, n)
            Qi = Q[i:i_end]

            Oi = O[i:i_end]
            mi = m[i:i_end]
            li = l[i:i_end]

            S = (Qi @ Kj.T) * scale

            if causal:
                q_idx = np.arange(i, i_end)[:, None]
                k_idx = np.arange(j, j_end)[None, :]
                causal_mask = k_idx > q_idx
                S = np.where(causal_mask, -np.inf, S)

            block_row_max = np.max(S, axis=1)
            new_m = np.maximum(mi, block_row_max)

            old_scale = np.exp(mi - new_m)
            P = np.exp(S - new_m[:, None])

            new_l = old_scale * li + np.sum(P, axis=1)

            weighted_old = (old_scale * li)[:, None] * Oi
            weighted_new = P @ Vj
            new_O = (weighted_old + weighted_new) / new_l[:, None]

            O[i:i_end] = new_O
            m[i:i_end] = new_m
            l[i:i_end] = new_l

    return O


def main():
    rng = np.random.default_rng(0)

    N, d, dv = 8, 4, 3
    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, dv))

    ref = standard_attention(Q, K, V, causal=False)
    out = flash_attention_toy(Q, K, V, br=2, bc=3, causal=False)
    np.testing.assert_allclose(ref, out, atol=1e-10, rtol=1e-10)

    ref_causal = standard_attention(Q, K, V, causal=True)
    out_causal = flash_attention_toy(Q, K, V, br=2, bc=3, causal=True)
    np.testing.assert_allclose(ref_causal, out_causal, atol=1e-10, rtol=1e-10)

    print("non-causal check: ok")
    print("causal check: ok")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出应为：

```text
non-causal check: ok
causal check: ok
```

### 代码和公式是一一对应的

下面把代码中的关键变量与公式对应起来：

| 代码变量 | 数学含义 |
|---|---|
| `m` | 每一行到当前块为止的最大值 $m_i$ |
| `l` | 每一行到当前块为止的指数和 $\ell_i$ |
| `S` | 当前块的局部分数矩阵 $S_{ij}$ |
| `old_scale` | 重标定因子 $e^{m_i-m_i'}$ |
| `P` | 当前块在新基准下的指数项 $e^{S-m_i'}$ |
| `weighted_old` | 旧输出按新基准重缩放后的贡献 |
| `weighted_new` | 当前块对输出的新贡献 |

这也是理解 FlashAttention 的最短路径：  
**它不是先算完整概率矩阵，再乘上 $V$；而是每看完一个块，就立刻把这个块对输出的贡献折算进去。**

### 用一个最小例子看执行顺序

假设：

- $N=8$
- `br=2`
- `bc=3`

那么执行顺序大致是：

1. 取 `K[0:3]` 和 `V[0:3]`
2. 依次处理 `Q[0:2]`、`Q[2:4]`、`Q[4:6]`、`Q[6:8]`
3. 更新所有这些 query 行的 `m / l / O`
4. 再取 `K[3:6]` 和 `V[3:6]`
5. 再扫描一遍全部 query 块
6. 最后处理 `K[6:8]` 和 `V[6:8]`

写成伪代码就是：

```python
for K_block, V_block in KV_blocks:
    load_block_to_sram(K_block, V_block)

    for Q_block in Q_blocks:
        S = Q_block @ K_block.T / sqrt(d)
        update_m_l_O_in_sram(S, V_block)
```

注意两个初学者最容易误解的点。

第一，`m` 和 `l` 不是局部临时变量，而是**每个 query 行跨块延续的全局状态**。  
第二，`O` 也不是最后一步才计算，而是**每来一个新块就要重新折算并累积**。

### 这个玩具实现和真实 GPU kernel 的差别

这个 Python 版本只保留了算法骨架，没有体现真实 kernel 的硬件细节。真正的 FlashAttention 实现还要处理：

- warp / thread block 映射
- shared memory 布局
- 寄存器压力
- 向量化加载
- Tensor Core 使用
- mask、dropout、mixed precision 融合
- forward/backward 的不同存储策略

但它已经足够说明最核心的一点：**FlashAttention 的突破首先是数据流设计，而不是某个单独公式。**

---

## 工程权衡与常见坑

FlashAttention 的难点不在“知道要分块”，而在“分块后如何仍然正确、稳定、高效”。

下面先看常见问题表。

| 常见坑 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| 块大小设得过大 | 性能下降、occupancy 变差，甚至 kernel fallback | 共享内存或寄存器超预算 | 理论块大小只作上界，实测调优 |
| 在线 softmax 合并错误 | 输出和标准注意力不一致 | $m,\ell,O$ 没有同步重标定 | 严格按在线更新公式实现 |
| 只更新分母不更新旧输出 | 数值接近但误差逐块累积 | 旧输出仍停留在旧基准下 | 更新 $\ell$ 时必须同步更新 $O$ |
| mask 处理位置错误 | causal attention 行为异常 | 先做合并后加 mask，会破坏概率定义 | 在块内分数阶段就应用 mask |
| 混合精度下直接 `exp` | 长序列时 NaN 或 inf | 指数动态范围过大 | 先减最大值，再用更高精度维护状态 |
| 短序列强行启用 | 没明显收益 | 控制流和块调度开销盖过收益 | 让 kernel 根据长度做选择 |
| 训练时 dropout 融合不一致 | 与基线实现结果偏差大 | 随机掩码应用顺序不同 | 在块内与 softmax 同步融合 |

### 块大小不是越大越好

理论上，块越大，HBM 往返越少。但工程上还要同时看：

- shared memory 是否装得下
- 寄存器是否爆掉
- 编译器是否还能生成高效指令
- occupancy 是否下降
- Tensor Core 对齐是否满足

所以像

$$
B_c=\left\lfloor \frac{M}{4d} \right\rfloor
$$

这种公式更像是“容量上界估计”，不是“无条件最优解”。  
真实工程里，最佳块大小通常要结合：

- head dimension
- 数据类型（fp16 / bf16 / fp8）
- 是否 causal
- 是否训练
- 当前 GPU 架构

一起调。

### 数值稳定性不是附属问题，而是主问题

在线 softmax 的全部设计，都是为了解决这个问题：

- 分块后你看不到完整一行
- 不同块的分数尺度可能差很多
- 若直接把局部指数和相加，会立刻失真

因此需要始终维护：

- 当前全局最大值 $m$
- 在这个最大值坐标系下的指数和 $\ell$

如果这个逻辑有一处写错，错误往往不会在短例子里立刻爆炸，而是会在：

- 更长的序列
- 更大的分数范围
- 更低精度
- causal mask 边界处

逐渐累积成明显偏差。

### “不显式保存矩阵”不等于 FlashAttention

这是一个常见误解。很多实现看起来没有输出完整的 attention matrix，但它们仍然可能：

- 在 kernel 之间回写局部中间结果到 HBM
- 多次读取等价规模的临时张量
- 让 softmax 和 matmul 分成多个阶段完成

如果中间量仍然反复离开片上内存，那么 IO 压力并没有真正解决。  
因此 FlashAttention 的判断标准不是“代码里有没有一个叫 `attn` 的大张量”，而是：

**一整个块的分数计算、归一化与输出累积，是否在片上闭环完成。**

### 新手最容易踩的两个概念坑

第一个坑是把 `m` 看成“当前块最大值”。  
实际上它是**到当前块为止的全局最大值**。

第二个坑是把 `O` 看成“已经最终确定的输出”。  
实际上在扫描完所有 $K/V$ 块之前，`O` 只是**在当前归一化坐标系下的阶段性累计结果**，后续仍可能因为更大的分数出现而被重标定。

把这两点记住，FlashAttention 的更新公式就不会再显得突兀。

---

## 替代方案与适用边界

FlashAttention 很重要，但它不是唯一优化路线，也不是任何情况下都该优先使用。

| 方案 | 是否保持精确 attention | HBM 压力 | 额外通信 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|---|
| 传统 fused attention | 是 | 高 | 低 | 低 | 短序列、实现简单优先 |
| FlashAttention | 是 | 明显更低 | 低 | 中到高 | 中长到超长上下文 |
| Tensor Parallel / Sequence Parallel | 是 | 取决于切分 | 高 | 高 | 单卡放不下、需多卡扩展 |
| 稀疏注意力 | 否，通常改变连接模式 | 更低 | 取决于实现 | 高 | 超长上下文、结构先验明显 |
| 线性/近似注意力 | 否，通常不再与原式等价 | 更低 | 低 | 高 | 可接受近似误差的极长序列 |

### 第一层边界：短序列不一定有收益

当序列长度只有几百到几千时，很多框架里的 fused attention 已经足够高效。  
这时 FlashAttention 还要承担：

- 分块调度
- 状态维护
- 更复杂的 kernel 约束

收益可能不明显，甚至在某些配置下更慢。

### 第二层边界：它优化 IO，不直接优化通信

如果你在做多卡并行，瓶颈可能来自：

- all-reduce
- all-gather
- KV cache 跨设备同步

FlashAttention 主要减少的是**单设备内部的 HBM 往返**，不是跨设备通信。  
所以它和 tensor parallel、pipeline parallel、sequence parallel 常常是互补关系，而不是替代关系。

### 第三层边界：它保持精确语义

FlashAttention 的一个重要价值在于：  
**它没有改变标准 attention 的数学结果，只是换了执行顺序。**

这与很多近似方案不同。后者常通过：

- 核函数近似
- 稀疏模式裁剪
- 低秩分解
- 状态压缩

进一步降低复杂度，但代价是模型行为本身发生变化。  
如果你的要求是“必须与标准 attention 等价”，FlashAttention 的优先级就很高。

### 一个实用选择规则

可以把方案选择粗略地写成下面这个判断表：

| 场景 | 优先选择 |
|---|---|
| 序列较短，追求实现简单 | 普通 fused attention |
| 序列较长，需要精确 attention | FlashAttention |
| 单卡放不下，需要多卡扩展 | FlashAttention + 并行方案 |
| 要把上下文继续推得更长，且可接受近似 | 稀疏/线性/其他近似注意力 |

实际工程中，一个常见组合是：

- 用 FlashAttention 降低单卡 attention 的 IO 压力
- 用 KV cache 处理自回归推理
- 再用 tensor parallel 或 paged KV cache 解决容量问题

换句话说，**FlashAttention 往往是长上下文系统里的基础模块，而不是全部答案。**

---

## 参考资料

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.  
   链接：https://arxiv.org/abs/2205.14135  
   用途：原始论文，给出 IO-aware 视角、在线 softmax 推导与复杂度分析。

2. Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024.  
   链接：https://arxiv.org/abs/2307.08691  
   用途：理解第一代 FlashAttention 之后，进一步看并行划分、warp 级工作分配和实际 kernel 优化方向。

3. FlashAttention 官方仓库。  
   链接：https://github.com/Dao-AILab/flash-attention  
   用途：查看真实工程实现、支持的 GPU/精度、安装方式以及与 PyTorch 集成的接口形式。

4. NVIDIA A100 Tensor Core GPU Architecture Whitepaper.  
   链接：https://resources.nvidia.com/en-us-tensor-core/nvidia-ampere-architecture-whitepaper  
   用途：理解 A100 的 HBM、片上存储和硬件层级背景，帮助把“IO 感知”放回真实 GPU 架构里理解。

5. AndoLogs: *AI: Flash Attention in a Flash*.  
   链接：https://blog.ando.ai/posts/ai-flash-attn-in-a-flash/  
   用途：适合入门阅读，便于把论文中的块划分、在线 softmax 和伪代码连起来看。

6. aman.ai Primer: *FlashAttention*.  
   链接：https://aman.ai/primers/ai/flashattention/  
   用途：从长上下文训练和推理的工程视角解释 FlashAttention 为什么重要。

7. Stanford CS336 / 相关课程与博客中的 attention kernel 讲解材料。  
   用途：帮助把“公式层 attention”与“kernel 层数据流”对应起来，理解为什么实际瓶颈常是内存搬运而不是 FLOPs。

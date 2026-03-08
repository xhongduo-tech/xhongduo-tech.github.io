## 核心结论

Mamba-2 的 SSD 层可以看成两种完全等价的东西，而且这个等价不是“直觉上相似”，而是同一个算子在两种记号下的精确重写。

第一种看法是选择性 SSM。SSM 是状态空间模型，核心对象是随时间递推的隐藏状态 $h_t$。最基础的离散形式写成

$$
h_t = A_t h_{t-1} + B_t x_t,\qquad y_t = C_t^\top h_t.
$$

这里 $x_t$ 是第 $t$ 个 token 的输入，$h_t$ 是当前状态，$y_t$ 是输出。它的含义很直接：过去的信息先被压到状态里，再由当前的读出向量 $C_t$ 取出需要的部分。

第二种看法是一个特殊的因果 masked attention。这里的 attention 不是标准 softmax attention，而是“把整段序列的 token mixing 写成一个下三角矩阵乘法”。Mamba-2 的关键限制是把每一步的状态转移矩阵约束成

$$
A_t = a_t I.
$$

这句话的技术含义是：状态向量的所有维度在时间上共享同一个衰减系数 $a_t$，但输入注入 $B_t$ 和输出读出 $C_t$ 仍然保留内容相关性。正是这个限制，让原本更一般的选择性 SSM 落到一个非常特殊、但结构极好的矩阵类上。

这个结果带来的直接后果是：整条序列对应的 token-mixing 矩阵成为半可分矩阵（semiseparable matrix）。半可分矩阵不是稠密矩阵，但它的非对角块可以写成低秩乘积，因此有两个同时成立的性质：

| 性质 | 含义 | 对 SSD 的价值 |
|---|---|---|
| 因果性 | 只依赖过去位置 | 适合自回归建模 |
| 非对角块低秩 | 远距离交互可压缩 | 可以改写成 batched matmul |
| 可块分解 | 大矩阵可分成小块处理 | 训练时更适合 GPU |
| 保留递推语义 | 仍然有状态传递 | 推理时仍可单步更新 |

于是，Mamba-2 可以把 Mamba-1 里以并行 scan 为主的实现，改写成“块内 batched matmul + 块间短递推”的 SSD 算法。训练时，大部分 FLOPs 落在 tensor core 友好的矩阵乘法上，因此比 Mamba-1 更容易把 GPU 吞吐吃满。

从工程角度看，SSD 不是把递推消掉，而是把长序列上的顺序依赖压缩到 chunk 之间。块内全部并行，块间只传压缩状态。这就是它在长序列训练上通常更快的根本原因。

---

## 问题定义与边界

先把边界说清楚。本文讨论的是 **Mamba-2 的 SSD 层**，不是一般 SSM，也不是标准 Transformer attention。为了避免符号过多，下面先用单头、单输入通道的记号说明主线；多头和高维输入只是把标量乘法替换成向量或小矩阵乘法，结构不变。

SSD 层接收长度为 $T$ 的序列 $x_1,\dots,x_T$，并保持因果约束，也就是第 $j$ 个位置只能依赖 $1,\dots,j$。把递推反复展开，有

$$
h_j
= A_j A_{j-1}\cdots A_1 h_0
+ \sum_{i=1}^{j}\left(\prod_{k=i+1}^{j} A_k\right) B_i x_i.
$$

若把初始状态 $h_0$ 先设为 0，只看输入对输出的贡献，则

$$
y_j=\sum_{i=1}^{j} C_j^\top\left(\prod_{k=i+1}^{j} A_k\right)B_i x_i.
$$

因此可以定义一个下三角矩阵 $M$：

$$
M_{ji}=C_j^\top\left(\prod_{k=i+1}^{j} A_k\right)B_i,\qquad i\le j,
$$

并写成整段序列的矩阵形式

$$
y = Mx.
$$

这一步很关键，因为它说明：递推系统不是矩阵乘法的“替代品”，它本身就是一种特殊的结构化矩阵乘法。只是这个矩阵不会显式 materialize 成普通的 $T\times T$ 稠密矩阵而已。

SSD 的额外限制是

$$
A_t = a_t I.
$$

把它代回上式，得到

$$
M_{ji}
=
\left(\prod_{k=i+1}^{j} a_k\right)\left(C_j^\top B_i\right).
$$

现在矩阵项被清楚地拆成了两部分：

1. $\prod_{k=i+1}^{j} a_k$：只负责“位置之间如何衰减”。
2. $C_j^\top B_i$：只负责“当前位置和过去位置是否内容匹配”。

于是可写成

$$
M = L \circ (CB^\top),
$$

其中 $\circ$ 是 Hadamard 乘，即逐元素相乘，$L$ 是因果衰减 mask，满足

$$
L_{ji}=
\begin{cases}
\prod_{k=i+1}^{j} a_k, & i\le j,\\
0, & i>j.
\end{cases}
$$

这和线性 attention 的结构已经非常接近。区别不是“一个是 attention，一个不是 attention”，而是 mask 的具体结构不同。SSD 的 mask 不是任意因果 mask，而是由连续衰减链构成的乘法型 mask。

为了防止把边界说大了，下面给一个更精确的比较：

| 模型 | 序列混合形式 | 时间结构 | 训练并行性 | 推理单步更新 | GPU 适配 |
|---|---|---|---:|---:|---|
| 标准 softmax attention | 稠密下三角矩阵 | 全局两两交互 | 高 | 中 | 很好 |
| 线性 attention | 结构化矩阵 | 可重写为前缀统计 | 高 | 低 | 很好 |
| Mamba-1 selective scan | 一般选择性 SSM | 长递推 | 中 | 很低 | 一般 |
| Mamba-2 SSD | 半可分矩阵 / 标量-单位阵 SSM | 块内并行 + 块间短递推 | 高 | 低到中 | 很好 |

因此，SSD 不是“比 attention 全面更强”，也不是“完全没有顺序依赖”。它做的是一个很明确的折中：保留线性时间的状态递推结构，同时把训练期的大头计算改造成 GPU 擅长的大矩阵乘法。

---

## 核心机制与推导

先从最小的玩具例子开始。设序列长度 $T=4$，chunk size $Q=2$，于是序列被切成两块：$(x_1,x_2)$ 和 $(x_3,x_4)$。

如果从递推角度看，它就是

$$
h_1=a_1 h_0 + B_1 x_1,\qquad
h_2=a_2 h_1 + B_2 x_2,
$$

$$
h_3=a_3 h_2 + B_3 x_3,\qquad
h_4=a_4 h_3 + B_4 x_4,
$$

输出为

$$
y_t=C_t^\top h_t.
$$

如果先忽略 $B_t,C_t$，只看最核心的衰减传播，那么递推

$$
s_t = a_t s_{t-1} + x_t
$$

对应的下三角矩阵是

$$
L=
\begin{bmatrix}
1 & 0 & 0 & 0 \\
a_2 & 1 & 0 & 0 \\
a_3a_2 & a_3 & 1 & 0 \\
a_4a_3a_2 & a_4a_3 & a_4 & 1
\end{bmatrix}.
$$

它的第 $j$ 行第 $i$ 列表示“位置 $i$ 的输入传播到位置 $j$ 时累计乘了多少次衰减”。因此

$$
s = Lx.
$$

把它写开：

$$
s_1=x_1,
$$

$$
s_2=a_2x_1+x_2,
$$

$$
s_3=a_3a_2x_1+a_3x_2+x_3,
$$

$$
s_4=a_4a_3a_2x_1+a_4a_3x_2+a_4x_3+x_4.
$$

这和递推逐步算出来的结果完全一致。这里最重要的不是公式本身，而是它揭示了一个事实：

| 视角 | 计算对象 | 顺序性在哪里 |
|---|---|---|
| 递推视角 | 每步更新状态 $h_t$ | 时间轴上逐步传递 |
| 矩阵视角 | 一次性乘结构化下三角矩阵 | 藏在矩阵结构里 |
| SSD 视角 | 对下三角矩阵做块分解 | 只剩 chunk 间短依赖 |

接着把 $B_t,C_t$ 放回来。对任意 $j\ge i$，有

$$
M_{ji}=C_j^\top B_i \cdot a_{i+1}a_{i+2}\cdots a_j.
$$

若把序列切成大小为 $Q$ 的块，那么整个下三角半可分矩阵可以分成四类计算：

1. 对角块  
   每个块内部仍然是一个较小的因果半可分矩阵，可以直接用局部矩阵乘法完成。这部分对应“块内输入对块内输出的影响”。

2. 块末状态  
   对每个块，计算“假设块初始状态为 0 时，本块结束后的状态”。这一步本质上是在提取每个块的压缩摘要。

3. 块间状态传递  
   把上一步得到的块级状态沿 chunk 维度做一次短递推。这一步仍然有顺序依赖，但长度从 $T$ 变成了 $T/Q$。

4. 初始状态补回输出  
   拿到每个块真实的初始状态后，再把这个状态对块内所有位置的贡献加回去。

可以把这四步和论文里的 block decomposition 一一对应：

| 论文直观颜色 | 作用 | 算法含义 | 常见实现原语 |
|---|---|---|---|
| 对角块 | 块内因果 mixing | intra-chunk output | batched matmul |
| 绿色块 | 输入到块末状态 | chunk state | batched matmul |
| 黄色块 | 块间状态递推 | state passing | short scan / recurrence |
| 蓝色块 | 初始状态到块内输出 | output from true init state | batched matmul |

因此，SSD 的本质不是“把 SSM 变成 attention”，而是“利用半可分矩阵结构，把一条长递推拆成局部二次项和全局线性项”。局部二次项对应块内矩阵乘，全局线性项对应块间状态传递。

这也是为什么 SSD 更适合 GPU。以长度 16K、chunk size 128 为例，块数约为

$$
16\,384 / 128 = 128.
$$

这意味着：

- 块内有 128 个 token，可以组成较规则的小矩阵，适合 tensor core。
- 块间只剩 128 个 chunk 级状态要传，而不是 16K 个 token 级状态。
- 大部分 FLOPs 被转移到高吞吐 matmul，短递推只占很小一部分。

这个分解解释了一个常见误解：SSD 快，不是因为它“没有递推”，而是因为它把 **长递推缩成短递推**，再把剩余计算尽量改写成矩阵乘。

---

## 代码实现

下面给一个最小可运行版本，验证三件事：

1. 逐步递推 `scan` 的结果；
2. 显式构造下三角矩阵 `M` 后做 `y = M @ x` 的结果；
3. 按 chunk 分块后，“块内局部贡献 + 块间状态传递 + 初始状态补回”的结果；

这三者完全一致。代码使用标量状态，目的不是追求性能，而是把 SSD 的数学结构拆开给新手看清楚。

```python
import numpy as np


def ssm_scan(a, b, c, x, h0=0.0):
    """
    标量状态版本
    h_t = a_t * h_{t-1} + b_t * x_t
    y_t = c_t * h_t
    """
    T = len(x)
    h = float(h0)
    y = np.zeros(T, dtype=np.float64)

    for t in range(T):
        h = a[t] * h + b[t] * x[t]
        y[t] = c[t] * h

    return y


def build_ssd_matrix(a, b, c):
    """
    构造显式下三角矩阵 M
    M[j, i] = c[j] * (prod_{k=i+1..j} a[k]) * b[i], for i <= j
    """
    T = len(a)
    M = np.zeros((T, T), dtype=np.float64)

    for j in range(T):
        for i in range(j + 1):
            decay = 1.0
            for k in range(i + 1, j + 1):
                decay *= a[k]
            M[j, i] = c[j] * decay * b[i]

    return M


def ssm_matrix(a, b, c, x):
    M = build_ssd_matrix(a, b, c)
    return M @ x, M


def local_chunk_matrix(a_chunk, b_chunk, c_chunk):
    """
    构造单个 chunk 内部的局部矩阵。
    只计算“块内输入 -> 块内输出”的贡献，默认 chunk 初始状态为 0。
    """
    q = len(a_chunk)
    M_local = np.zeros((q, q), dtype=np.float64)

    for j in range(q):
        for i in range(j + 1):
            decay = 1.0
            for k in range(i + 1, j + 1):
                decay *= a_chunk[k]
            M_local[j, i] = c_chunk[j] * decay * b_chunk[i]

    return M_local


def propagate_state_only(a_chunk, h_init):
    """
    只传播状态，不注入当前 chunk 输入。
    返回每个位置看到的“来自块初始状态”的隐藏状态贡献。
    """
    q = len(a_chunk)
    h = float(h_init)
    states = np.zeros(q, dtype=np.float64)

    for t in range(q):
        h = a_chunk[t] * h
        states[t] = h

    return states


def full_chunk_end_state(a_chunk, b_chunk, x_chunk, h_init):
    """
    计算一个 chunk 处理完成后的最终状态。
    """
    h = float(h_init)
    for t in range(len(a_chunk)):
        h = a_chunk[t] * h + b_chunk[t] * x_chunk[t]
    return h


def ssd_chunked(a, b, c, x, chunk_size, h0=0.0):
    """
    用 SSD 的分块思路计算：
    Step 1: 块内局部输出（假设块初始状态为 0）
    Step 2: 计算每个块在零初始状态下的结束状态
    Step 3: 在 chunk 维度上传递真实状态
    Step 4: 把真实初始状态对块内输出的贡献补回
    """
    T = len(x)
    assert T % chunk_size == 0, "为了演示简洁，这里要求 T 能被 chunk_size 整除"
    num_chunks = T // chunk_size

    # Step 1: 块内局部输出
    y_local = np.zeros(T, dtype=np.float64)
    zero_init_end_states = np.zeros(num_chunks, dtype=np.float64)

    for chunk_idx in range(num_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        a_chunk = a[s:e]
        b_chunk = b[s:e]
        c_chunk = c[s:e]
        x_chunk = x[s:e]

        M_local = local_chunk_matrix(a_chunk, b_chunk, c_chunk)
        y_local[s:e] = M_local @ x_chunk

        zero_init_end_states[chunk_idx] = full_chunk_end_state(
            a_chunk, b_chunk, x_chunk, h_init=0.0
        )

    # Step 2 + Step 3: 先求每个 chunk 的真实初始状态
    chunk_init_states = np.zeros(num_chunks, dtype=np.float64)
    h = float(h0)

    for chunk_idx in range(num_chunks):
        chunk_init_states[chunk_idx] = h
        s = chunk_idx * chunk_size
        e = s + chunk_size
        h = full_chunk_end_state(a[s:e], b[s:e], x[s:e], h_init=h)

    # Step 4: 把“真实初始状态 -> 块内输出”的贡献补回
    y_state = np.zeros(T, dtype=np.float64)

    for chunk_idx in range(num_chunks):
        s = chunk_idx * chunk_size
        e = s + chunk_size

        a_chunk = a[s:e]
        c_chunk = c[s:e]
        h_init = chunk_init_states[chunk_idx]

        state_contrib = propagate_state_only(a_chunk, h_init)
        y_state[s:e] = c_chunk * state_contrib

    return y_local + y_state


def main():
    a = np.array([0.7, 0.8, 0.9, 0.6], dtype=np.float64)
    b = np.array([1.0, 0.5, 1.2, 0.3], dtype=np.float64)
    c = np.array([1.5, 0.8, 1.1, 2.0], dtype=np.float64)
    x = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float64)

    y_scan = ssm_scan(a, b, c, x)
    y_mat, M = ssm_matrix(a, b, c, x)
    y_chunk = ssd_chunked(a, b, c, x, chunk_size=2)

    print("M =")
    print(M)
    print()
    print("y_scan =", y_scan)
    print("y_mat  =", y_mat)
    print("y_chunk=", y_chunk)

    assert np.allclose(y_scan, y_mat, atol=1e-12)
    assert np.allclose(y_scan, y_chunk, atol=1e-12)
    assert np.allclose(M, np.tril(M)), "因果矩阵必须是下三角"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
```

对这段代码，建议按下面的阅读顺序看：

| 函数 | 作用 | 对应理论步骤 |
|---|---|---|
| `ssm_scan` | 直接按递推定义算 | 原始 SSM 视角 |
| `build_ssd_matrix` | 显式写出下三角矩阵 | 结构化矩阵视角 |
| `local_chunk_matrix` | 只看块内输入的局部贡献 | SSD Step 1 |
| `full_chunk_end_state` | 计算块末状态 | SSD Step 2 |
| `chunk_init_states` 那段循环 | 把真实状态跨块传递 | SSD Step 3 |
| `propagate_state_only` + `y_state` | 将块初始状态对输出的影响补回 | SSD Step 4 |

如果你实际运行，会得到三组完全相同的输出。这说明“递推”和“矩阵乘”不是近似关系，而是精确等价。

再给一个小公式，把代码和理论对齐。对于第 $m$ 个 chunk，记其起始状态为 $\tilde h_m$。那么块内第 $\tau$ 个位置的输出可以拆成

$$
y_{m,\tau}
=
\underbrace{\sum_{r=1}^{\tau} C_{m,\tau}^\top
\left(\prod_{k=r+1}^{\tau} A_{m,k}\right)
B_{m,r} x_{m,r}}_{\text{块内输入贡献}}
+
\underbrace{C_{m,\tau}^\top
\left(\prod_{k=1}^{\tau} A_{m,k}\right)\tilde h_m}_{\text{块初始状态贡献}}.
$$

这正是上面代码里 `y_local + y_state` 的来源。

要强调的是，真正的高性能实现 **不会** 显式构造整个 $T\times T$ 的矩阵 `M`，也不会在 Python for-loop 里逐块算。工业实现通常会把 Step 1、2、4 写成 batched matmul kernel，把 Step 3 写成短 scan 或 state pass kernel。逻辑和上面完全一样，只是底层算子换成了 GPU 友好的形式。

---

## 工程权衡与常见坑

SSD 的核心工程参数是 `chunk_size = Q`。它同时控制两种代价：

1. 块内计算代价，大致随 $Q^2$ 增长；
2. 块间递推长度，大致随 $T/Q$ 增长。

因此它不是越大越好，也不是越小越好。可以把这个关系理解成一个简单的张力：

| `Q` 的取值 | 好处 | 坏处 |
|---|---|---|
| 大 | chunk 少，块间递推短 | 块内矩阵更重，寄存器/共享内存压力更大 |
| 小 | 块内更轻，更容易塞进 kernel | chunk 多，state passing 比例更高 |
| 中间值 | matmul 与递推较平衡 | 需要结合硬件 autotune |

这也是为什么论文和后续系统实现都强调 autotuning。最优 `chunk_size` 取决于很多具体因素：

- GPU 架构是 A100、H100 还是更新一代；
- 状态精度是 fp16、bf16 还是 fp32；
- head 维度和状态维度多大；
- kernel 是否融合；
- batch size 和序列长度处在哪个区间。

这一点在 2026 年 2 月 6 日发布的 PyTorch 博客里也有明确现象：融合后的 Triton SSD kernel 在一些配置下的最优 chunk size 与原始 unfused 实现不同，说明 `Q` 不是理论常数，而是实现相关的硬件参数。

初学者最常见的坑不是公式看不懂，而是把四步算法少实现一步。下面是最常见的错误：

| 坑 | 表面现象 | 真正原因 | 修复方式 |
|---|---|---|---|
| 漏掉 Step 3 | 结果和 scan 对不上 | 跨 chunk 依赖被切断 | 显式实现 state passing |
| 漏掉 Step 4 | 块内结果像是“少了一截” | 真实初始状态贡献没补回 | 把块初始状态映射到块内各位置 |
| `Q` 过大 | 理论上更并行，实际上更慢 | matmul 变大，资源占用过高 | autotune，不要拍脑袋固定 |
| `Q` 过小 | GPU 很忙但吞吐不高 | 短 scan 和 launch 开销占比上升 | 结合序列长度一起调参 |
| 状态精度太低 | 长序列漂移、数值不稳 | 连续衰减乘积对舍入敏感 | 状态用更高精度或谨慎混精 |

还有一个更隐蔽的问题是数值稳定性。因为衰减项里存在很多连乘：

$$
\prod_{k=i+1}^{j} a_k.
$$

若序列很长，且 $a_k$ 接近 0 或 1，这个乘积可能极小，也可能在 log 空间里更容易处理。实际实现里常把它改写成 log-domain 的 segment sum，再指数化回来，而不是直接做大段 `cumprod`。原因不是数学变了，而是浮点误差会在长上下文下累计。

另一个常被忽略的事实是：SSD 很并行，但 **不是所有块都能完全独立执行到结束**。块内独立，块间不独立。真正消不掉的依赖只被压缩到了 chunk 级别。如果把跨块状态传播做错，模型短上下文上可能还能训，长上下文性能通常会明显掉，因为信息链被你人为截断了。

---

## 替代方案与适用边界

如果目标是训练吞吐，SSD 很有吸引力，因为它把大量工作转成 matmul，可以直接复用 Transformer 世界里已经成熟的系统基础设施，比如 tensor parallel、sequence parallel、变长序列处理和 kernel fusion。

如果目标是自回归单步解码延迟，纯 scan 仍然很强。原因并不复杂：解码时一次只新增一个 token，此时 chunk 内大矩阵乘的优势并不明显，而递推状态天然适合单步更新。SSD 在推理时依然可以很快，但它最突出的系统优势主要体现在 prefill 和训练，而不是每一步都压到极限的 decode latency。

如果目标是“我已经有一套 attention 框架，只想小改”，那么 RetNet 式 chunkwise attention，或者一些 Gated Linear Attention 变体，也值得考虑。它们可以视作 SSD 框架中的更特殊情形：mask 更简单，接入成本更低，但表达空间也通常更受限。

可以用下面这张表来做选择：

| 方案 | 计算主形态 | 优势 | 推荐场景 | 不适合场景 |
|---|---|---|---|---|
| 纯 scan SSM | 全递推 | 单步更新直接、状态缓存清晰 | 低延迟解码、流式推理 | 追求训练期最大 tensor core 吞吐 |
| SSD | 块内 matmul + 块间递推 | 训练更适合 GPU，系统优化空间大 | 长序列训练、prefill、混合架构 | 极短序列、只看单步 decode latency |
| RetNet / chunkwise 线性 attention | 特定 mask 的块计算 | 更容易塞进已有 attention stack | 已有 attention 基建、低改造成本 | 需要 SSD 完整灵活性的场景 |
| 标准 softmax attention | 稠密两两交互 | 表达直接、生态成熟 | 中短上下文、通用基线 | 超长序列成本高 |

所以，SSD 不是“取代 attention 的统一答案”，也不是“取代所有 SSM 的终点”。更准确的说法是：它在结构化矩阵这个共同语言下，把 SSM 和线性 attention 的交集显式写了出来，并给出了一个对现代 GPU 更友好的实现路径。

从这个角度看，“Mamba-2 的 SSD 层与矩阵分解”真正重要的不是某个具体 kernel，而是这条链条：

$$
\text{选择性 SSM}
\;\Longleftrightarrow\;
\text{半可分矩阵}
\;\Longleftrightarrow\;
\text{块分解算法}
\;\Longrightarrow\;
\text{大部分计算变成 matmul}.
$$

这条链条同时解释了三件事：

1. 为什么 SSD 有清楚的理论定义；
2. 为什么它和线性 attention 会发生重合；
3. 为什么它在工程上会比纯 scan 更适合现代 GPU。

---

## 参考资料

1. Tri Dao, Albert Gu. *State Space Duality (Mamba-2) Part I: The Model*. https://tridao.me/blog/2024/mamba2-part1-model/
2. Tri Dao, Albert Gu. *State Space Duality (Mamba-2) Part II: The Theory*. https://tridao.me/blog/2024/mamba2-part2-theory/
3. Tri Dao, Albert Gu. *State Space Duality (Mamba-2) Part III: The Algorithm*. https://tridao.me/blog/2024/mamba2-part3-algorithm/
4. Tri Dao, Albert Gu. *Mamba-2: Algorithms and Systems*. Princeton Language and Intelligence, June 3, 2024. https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems
5. Graphcore Research Blog. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. https://graphcore-research.github.io/mamba2/
6. Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024, PMLR 235:10041-10071. https://proceedings.mlr.press/v235/dao24a.html
7. PyTorch Blog. *Accelerating Mamba2 with Kernel Fusion*. February 6, 2026. https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/

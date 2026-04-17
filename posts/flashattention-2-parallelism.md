## 核心结论

FlashAttention-2 的并行化改进，核心不是“换了一个更快的注意力公式”，而是**重排了 GPU 上谁负责什么工作**。更准确地说，它把块级计算的主并行维度改成沿着 $Q$ 方向切分：每个 warp 独立负责一个完整的 $Q$ 块输出 $O_i$，然后在 warp 内顺序扫描该块需要看到的所有 $K/V$ 块。warp 可以理解为 GPU 上一组同步执行同一条指令的线程小队，NVIDIA GPU 上通常是 32 个线程。

这个调整带来三个直接结果。

1. $Q$ 块只需要加载一次，并在片上反复复用，避免旧版里随着不同 $K/V$ 块反复回读同一段 $Q$。
2. 每个 warp 自己维护完整的输出块和 softmax 统计量，不再把一个输出块拆给多个 warp 协作，因此大幅减少跨 warp 同步。
3. 因果 mask 下可以直接跳过整块无效区域，减少无意义的分数计算和 softmax 更新。

注意力的目标没有变化，仍然是：

$$
O_i=\sum_j \mathrm{softmax}\left(\frac{Q_iK_j^T}{\sqrt d}\right)V_j
$$

变化点只在于“怎么把这件事排到 GPU 上”。FlashAttention-2 关注的是**工作划分、访存路径、同步边界**，而不是改写注意力定义本身。

在论文给出的 A100 测试中，FlashAttention-2 在非因果场景可达到约 225 TFLOPs/s，接近 A100 理论峰值的 72%；在因果场景中，由于可以跳过完全无效的块，吞吐还能进一步提升，常见测量约有 1.7x 的额外加速。这些数字说明它解决的是内核组织问题，而不是只做了一个小的常数优化。

下面先用一个最小对比看并行化收益。设序列长度 $N=4096$，块大小 $B=64$，则块数为：

$$
T=\frac{N}{B}=\frac{4096}{64}=64
$$

如果旧版按 $K/V$ 块做外层循环，那么每处理一个新的 $K/V$ 块，都要重新把对应的 $Q$ 块拉进来参与计算；FlashAttention-2 则是把一个 $Q$ 块先放进片上，再顺序扫描完整段 $K/V$。

| 对比项 | 原版 FlashAttention | FlashAttention-2 |
|---|---:|---:|
| 外层循环主体 | $K/V$ 块 | $Q$ 块 |
| 单个 $Q$ 块被加载次数 | 64 次 | 1 次 |
| 单个输出块 $O_i$ 的中间回写 | 多次 | 1 次最终写出 |
| warp 间同步需求 | 高 | 低 |
| 因果 mask 空块跳过收益 | 有，但调度不够直接 | 更直接、更稳定 |

如果把“算一个注意力块”想成“处理一份完整任务单”，旧版更像“多人分段接力处理同一张单据”，中间不断交接；FlashAttention-2 更像“一个人负责一整张单据，只是不断查阅外部参考资料”。后者的交接和回写成本更低。这里的“查阅外部参考资料”对应 $K/V$ 块流式进入片上，而“负责一整张单据”对应 warp 独立维护一个 $Q$ 块的输出状态。

---

## 问题定义与边界

问题定义可以压缩成一句话：**在长上下文下，注意力已经是高计算密度算子，而传统块化实现还会引入额外的显存访问与同步开销；FlashAttention-2 要解决的正是这部分“本来可以省掉的成本”。**

标准自注意力写成矩阵形式是：

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\mathrm{softmax}(S),\qquad O=PV
$$

其中：

- $Q,K,V\in\mathbb{R}^{N\times d}$
- $N$ 是序列长度
- $d$ 是每个 attention head 的维度
- $S\in\mathbb{R}^{N\times N}$ 是注意力分数矩阵
- $P\in\mathbb{R}^{N\times N}$ 是 softmax 后的注意力权重
- $O\in\mathbb{R}^{N\times d}$ 是输出

如果直接显式构造 $S$ 和 $P$，显存代价会非常高。FlashAttention 系列的基本思想就是：**不存完整的 $N\times N$ 中间矩阵，而是按块计算、按块归并。**

把序列分块后，记第 $i$ 个 $Q$ 块为 $Q_i\in\mathbb{R}^{B_q\times d}$，第 $j$ 个 $K/V$ 块为：

$$
K_j,V_j\in\mathbb{R}^{B_k\times d}
$$

则块级输出可以写成：

$$
O_i=\sum_j \mathrm{softmax}\left(\frac{Q_iK_j^\top}{\sqrt d}\right)V_j
$$

式子本身并不复杂，真正的难点在实现边界。下面这张表把新手最容易混淆的几个约束列出来。

| 边界 | 含义 | 对实现的影响 |
|---|---|---|
| 长序列 $N$ 很大 | 块数 $T$ 增多，循环层数上升 | 重复访存和重复同步会被放大 |
| 因果 mask | 位置 $t$ 只能看见 $\le t$ 的 token | 一部分块完全无效，可整块跳过 |
| 片上存储有限 | SRAM、寄存器都很贵 | 块过大时会压低 occupancy |
| warp 独立输出 | 一个 warp 要保住完整输出状态 | 需要精细控制寄存器压力 |
| 数值稳定性 | softmax 不能直接逐块生算生加 | 必须用 online softmax 维护统计量 |

这里先解释两个容易跳过去的术语。

**HBM**：High Bandwidth Memory，高带宽显存。容量大、带宽高，但延迟和能耗都高于寄存器和共享内存。  
**occupancy**：一个 SM（Streaming Multiprocessor）上同时能驻留多少活跃线程块/warp。occupancy 太低时，即使单个 warp 很快，也可能因为没有足够多的并发任务来掩盖访存延迟而造成硬件空转。

因果 mask 是这里最关键的边界。它的语义是：**第 $t$ 个 query 不能访问未来位置，即不能看列索引大于 $t$ 的 key。**

若一个块满足“该块中所有 key 位置都严格大于当前 query 块的所有位置”，那么这一整块的注意力分数都应视为 $-\infty$，softmax 后就是 0，继续计算它没有意义。

举一个最小例子。假设块大小为 4：

- 当前 $Q$ 块覆盖位置 8 到 11
- 某个 $K$ 块覆盖位置 12 到 15

则该块中的任意分数都满足“列索引 $>$ 行索引”，整块都落在因果 mask 的无效上三角区域，可以直接跳过。若不跳过，内核仍会执行：

1. 整块矩阵乘
2. mask 写入
3. 局部 softmax 更新
4. 局部输出归并

最后得到的贡献却全为 0。这类计算纯粹是浪费。

用块边界写，这个整块可跳过条件是：

$$
k_s > q_e
$$

其中：

- $[q_s,q_e]$ 是当前 $Q$ 块覆盖的行范围
- $[k_s,k_e]$ 是当前 $K$ 块覆盖的列范围

只要 key 块的最左列都已经落在 query 块最下行的右侧，整块就全无效。

反过来，只要：

$$
k_s \le q_e
$$

就不能整块跳过，因为这个块中仍可能有部分元素合法，必须保留计算。

因此本文的讨论边界很明确：

- 讨论对象是**块化的精确注意力**
- 关注点是**GPU 内核中的并行组织与访存路径**
- 不讨论近似注意力
- 不讨论 CPU 实现
- 不讨论训练框架外层的流水并行或张量并行

---

## 核心机制与推导

FlashAttention-2 的核心机制可以压缩为一句话：**让每个 warp 对一个 $Q$ 块的输出负责到底。**

先看旧版思路。如果外层按 $K/V$ 块循环，它的控制流可以抽象为：

$$
\text{for } j \text{ in KV-blocks:}
\quad
\text{for } i \text{ in Q-blocks:}
\quad
O_i \leftarrow \text{update}(O_i,Q_i,K_j,V_j)
$$

这个组织方式有两个直接后果。

1. 同一个 $Q_i$ 会随着不同 $j$ 被反复加载。
2. 同一个 $O_i$ 经常不是在单个 warp 内完整维护，而是在多个执行单元之间反复更新、暂存、再读回。

FlashAttention-2 把循环交换成：

$$
\text{for } i \text{ in Q-blocks:}
\quad
\text{load } Q_i \text{ once}
$$

$$
\text{for } j \text{ in KV-blocks:}
\quad
O_i \leftarrow \text{update}(O_i,Q_i,K_j,V_j)
$$

表面上只是交换了 $i,j$ 的顺序，但工程意义非常大。因为：

- $Q_i$ 在整个内层循环期间都可以留在寄存器或共享内存中
- $O_i$ 的累计状态只在 warp 内更新
- softmax 需要维护的统计量也不必频繁回写到 HBM

这里会用到 online softmax。它的目的不是“改变 softmax”，而是**在分块扫描过程中仍然得到与全量 softmax 等价的结果，同时避免保存完整得分矩阵**。

对某一行来说，设旧状态为：

- $m$：当前已经扫描过部分的最大值
- $l$：归一化分母
- $o$：当前累计输出向量

新块产生的局部统计量为：

- $\tilde m$：新块中的局部最大值
- $\tilde l$：按 $\tilde m$ 归一化后的局部分母
- $\tilde o$：新块对应的局部输出

则更新公式为：

$$
m'=\max(m,\tilde m)
$$

$$
l' = e^{m-m'}l + e^{\tilde m-m'}\tilde l
$$

$$
o' = \frac{e^{m-m'}l\cdot o + e^{\tilde m-m'}\tilde l\cdot \tilde o}{l'}
$$

这组更新式有两个重要含义。

第一，**不需要先拿到整行所有分数才能做 softmax**。  
第二，**输出块在扫描过程中可以一直停留在片上，只在最后写回一次**。

### 为什么数值稳定

新手常见疑问是：“softmax 不是依赖全行分数吗，怎么能一块一块算？”

关键在于 softmax 有如下平移不变性。对任意常数 $c$，

$$
\mathrm{softmax}(x)_t=\frac{e^{x_t}}{\sum_s e^{x_s}}
=\frac{e^{x_t-c}}{\sum_s e^{x_s-c}}
$$

因此每次只要维护“当前全局最大值”，就能把指数计算保持在稳定区间，避免直接对很大的正数做 $\exp$，也避免对很小的负数全部下溢到 0。

### 读写次数的简化推导

设总块数为：

$$
T=\frac{N}{B}
$$

对某个固定的 $Q_i$：

- 原版：每遍历一个 $K/V$ 块，$Q_i$ 都要再次参与计算，因此近似读取 $T$ 次
- FA-2：$Q_i$ 读取 1 次，在片上复用 $T$ 次

因此 $Q_i$ 的读取次数从 $T$ 降到 1，下降比例为：

$$
\frac{T-1}{T}=1-\frac{1}{T}
$$

当 $N=4096,B=64$ 时，$T=64$，即单个 $Q$ 块读取次数降到原来的：

$$
\frac{1}{64}
$$

再看输出块 $O_i$ 的中间状态。

| 项目 | 原版 FlashAttention | FlashAttention-2 |
|---|---|---|
| 中间输出 $O_i$ | 常需多轮读写 | 尽量常驻片上 |
| softmax 状态 $(m,l)$ | 多轮归并 | warp 内持续维护 |
| 最终写回 | 多次中间写回 + 最终写回 | 1 次最终写回 |

这就是“每个 warp 维护完整输出块”的真正收益。不是只少了一两个 barrier，而是把**中间状态的反复 HBM 往返**尽量移出了主路径。

### 用执行视角理解

从执行视角看，FlashAttention-2 更接近下面的结构：

1. warp 载入一个 $Q$ 小块
2. 初始化该块所有行的 $(m,l,o)$
3. 顺序读取第 1 个 $K/V$ 小块
4. 计算局部分数、更新局部 softmax 统计量
5. 顺序读取第 2 个、第 3 个，直到最后一个 $K/V$ 小块
6. 所有块扫完后，一次性写回 $O_i$

这就是常说的“**Q 常驻，K/V 流过**”。  
$Q$ 是驻留数据，$K/V$ 是流式数据。

### 为什么同步更少

如果一个输出块由多个 warp 协作完成，那么每一轮局部结果归并都可能需要：

- barrier
- warp 间共享内存通信
- 中间结果对齐

而在 FA-2 中，一个 warp 对一个输出块负责到底，输出状态不需要频繁跨 warp 合并，所以同步点更少。barrier 可以理解为“所有执行单元必须一起停下来，等最慢的一组追上再继续”。同步点一多，GPU 的流水就会出现更多空洞。

### 一个更具体的工作量对比

假设有 64 个 $Q$ 块和 64 个 $K/V$ 块。

- 原版组织方式下，执行上更像“沿列推进，每推进一列都要重新触碰很多行块”
- FA-2 下，执行上更像“每个 warp 拿定一行块，从左到右扫完整个版面”

前者的主要成本是反复调入同一个 $Q$ 块、反复回写同一个输出块；后者的主要成本则集中在一次性保持住当前工作集。

这也是为什么它在长上下文上收益更明显：块数越多，原版重复访存和同步的累计成本越大，而 FA-2 的组织优势会被进一步放大。

---

## 代码实现

下面先给一个新手友好的伪代码，表达 FlashAttention-2 的控制流。它不是 CUDA 代码，但和真实内核的结构是一致的。

```text
for each q_block i:
    load_q_block_once(i)
    init m, l, o for rows in this q_block

    for each kv_block j:
        if causal and block_is_fully_masked(i, j):
            continue

        load_k_block(j)
        load_v_block(j)

        scores = q_block @ k_block.T / sqrt(d)
        apply_elementwise_mask_if_needed(scores)
        update_online_softmax(m, l, o, scores, v_block)

    write_o_block_once(i, o)
```

这个伪代码只强调三件事。

1. `Q` 块在最外层，只加载一次。
2. `K/V` 块在内层顺序流过。
3. 输出块与 softmax 状态由同一个工作单元持续维护到结束。

为了让结构更直观，下面给一个**可直接运行**的 Python 示例。它做三件事：

- 给出标准 dense attention 作为参考实现
- 给出按 `Q` 块外层扫描的 FA-2 风格块实现
- 验证 causal / non-causal 两种情况下，块实现与 dense 实现数值一致

```python
import math


NEG_INF = -1e30


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    denom = sum(exps)
    return [x / denom for x in exps]


def dense_attention(q, k, v, causal=False):
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0] * d for _ in range(n)]

    for i in range(n):
        logits = []
        for j in range(n):
            score = sum(q[i][t] * k[j][t] for t in range(d)) * scale
            if causal and j > i:
                score = NEG_INF
            logits.append(score)

        probs = softmax(logits)
        for j, p in enumerate(probs):
            for t in range(d):
                out[i][t] += p * v[j][t]

    return out


def online_softmax_update(m, l, o, block_scores, block_values):
    """
    对单行做一次块级 online softmax 更新。
    m: 旧的行最大值
    l: 旧的归一化分母
    o: 旧的输出向量
    block_scores: 当前块的分数列表
    block_values: 当前块对应的 value 向量列表
    """
    block_m = max(block_scores)
    block_exps = [math.exp(x - block_m) for x in block_scores]
    block_l = sum(block_exps)

    d = len(o)
    block_o = [0.0] * d
    for p, vec in zip(block_exps, block_values):
        for t in range(d):
            block_o[t] += p * vec[t]
    block_o = [x / block_l for x in block_o]

    new_m = max(m, block_m)
    old_scale = math.exp(m - new_m) * l
    block_scale = math.exp(block_m - new_m) * block_l
    new_l = old_scale + block_scale

    new_o = [0.0] * d
    for t in range(d):
        numerator = old_scale * o[t] + block_scale * block_o[t]
        new_o[t] = numerator / new_l

    return new_m, new_l, new_o


def block_is_fully_masked(q_start, q_end, k_start, causal):
    """
    因果 mask 下，若 K 块的最左列都在 Q 块最下行右侧，则整块可跳过。
    条件：k_start > q_end
    """
    if not causal:
        return False
    return k_start > q_end


def fa2_style_block_attention(q, k, v, block_size=2, causal=False):
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0] * d for _ in range(n)]

    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size - 1, n - 1)

        # 一个 Q 块对应一组独立输出状态
        running_m = [float("-inf")] * (q_end - q_start + 1)
        running_l = [0.0] * (q_end - q_start + 1)
        running_o = [[0.0] * d for _ in range(q_end - q_start + 1)]

        for k_start in range(0, n, block_size):
            k_end = min(k_start + block_size, n)

            if block_is_fully_masked(q_start, q_end, k_start, causal):
                continue

            for local_row, row in enumerate(range(q_start, q_end + 1)):
                block_scores = []
                block_values = []

                for col in range(k_start, k_end):
                    score = sum(q[row][t] * k[col][t] for t in range(d)) * scale
                    if causal and col > row:
                        score = NEG_INF
                    block_scores.append(score)
                    block_values.append(v[col])

                running_m[local_row], running_l[local_row], running_o[local_row] = (
                    online_softmax_update(
                        running_m[local_row],
                        running_l[local_row],
                        running_o[local_row],
                        block_scores,
                        block_values,
                    )
                )

        for local_row, row in enumerate(range(q_start, q_end + 1)):
            out[row] = running_o[local_row]

    return out


def almost_equal(a, b, eps=1e-9):
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if abs(x - y) > eps:
                return False
    return True


def main():
    q = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ]
    k = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ]
    v = [
        [1.0, 2.0],
        [0.0, 1.0],
        [3.0, 1.0],
        [2.0, 0.0],
    ]

    dense_full = dense_attention(q, k, v, causal=False)
    fa2_full = fa2_style_block_attention(q, k, v, block_size=2, causal=False)

    dense_causal = dense_attention(q, k, v, causal=True)
    fa2_causal = fa2_style_block_attention(q, k, v, block_size=2, causal=True)

    assert almost_equal(dense_full, fa2_full)
    assert almost_equal(dense_causal, fa2_causal)

    print("non-causal output:")
    for row in fa2_full:
        print([round(x, 6) for x in row])

    print("\ncausal output:")
    for row in fa2_causal:
        print([round(x, 6) for x in row])

    print("\nverification passed")


if __name__ == "__main__":
    main()
```

如果你本地运行，这段代码应该打印两组输出，并在最后打印：

```text
verification passed
```

### 这段代码对应了哪些 GPU 思想

| Python 结构 | 对应 GPU 思想 | 含义 |
|---|---|---|
| 外层 `for q_start` | 以 $Q$ 块为并行单元 | 保证 Q 只加载一次 |
| 内层 `for k_start` | 顺序扫描 $K/V$ 块 | 让 K/V 成为流式输入 |
| `running_m/running_l/running_o` | warp 本地状态 | 持续维护 online softmax |
| `block_is_fully_masked` | 因果 mask 整块跳过 | 删除全无效块计算 |
| 末尾写 `out[row]` | 最终一次写回 | 避免中间结果反复落到显存 |

### 新手最容易误解的地方

第一，**这里的 Python 外层按 Q，不代表 GPU 只会有一个 warp 工作**。  
真实 GPU 上会有很多 warp 并行地各自处理不同的 $Q$ 块。这里的“按 Q 外层”说的是**单个工作单元的任务边界**，不是整个设备只串行执行。

第二，**online softmax 不是近似算法**。  
只要更新公式实现正确，它与一次性对完整分数向量做 softmax 是等价的。

第三，**整块跳过不等于块内有无效元素就跳过**。  
只有当整个块全部无效时才能 `continue`。否则就必须进入块内做逐元素 mask。

### 再给一个小的块跳过例子

设 `block_size = 4`，当前正在处理：

- $Q$ 块：位置 `8, 9, 10, 11`
- $K$ 块：位置 `12, 13, 14, 15`

则：

$$
k_s = 12,\qquad q_e = 11,\qquad k_s > q_e
$$

整块满足跳过条件，内层直接 `continue`。

但如果当前 $K$ 块是位置 `10, 11, 12, 13`，则：

$$
k_s = 10,\qquad q_e = 11,\qquad k_s \le q_e
$$

这时不能整块跳过，因为块内位置 `10, 11` 仍可能是合法的，只能做块内 mask。

---

## 工程权衡与常见坑

FlashAttention-2 不是“只要切成按 Q 外层就一定更快”。它真正依赖的是一种平衡：**片上数据复用变多，但片上资源压力不能把并发度压垮。**

最核心的平衡项有三个：

- 共享内存占用
- 寄存器占用
- occupancy

如果块太小，Q/K/V 的加载和调度开销相对变大；如果块太大，寄存器和共享内存消耗过高，SM 上挂不住足够多的活跃 warp，延迟无法被掩盖。

| 块大小趋势 | SRAM/寄存器占用 | 访存次数 | 并发度 | 常见结果 |
|---|---|---|---|---|
| 太小 | 低 | 高 | 高 | 访存偏多，算力利用不满 |
| 适中 | 平衡 | 平衡 | 平衡 | 通常最优 |
| 太大 | 高 | 低 | 低 | 容易寄存器溢出或 occupancy 下降 |

### 坑 1：块大小只看“复用”，不看“驻留成本”

很多人直觉上会觉得“块越大，复用越强，所以越快”。这不完整。因为更大的块意味着：

- 单个 warp 或线程块要保留更多中间状态
- 每行的 softmax 统计量和部分输出向量要占更多寄存器
- 共享内存中缓存的 K/V 数据也会变大

当资源占用超过阈值时，编译器可能把部分寄存器溢出到本地内存，本来想减少 HBM 访问，结果却从另一条路径把额外访存又引回来了。

### 坑 2：因果 mask 的整块判断写错

正确逻辑不是“块里有些元素无效就整块跳过”，而是“只有当整块全部无效时才跳过”。

设当前 $Q$ 块行范围为 $[q_s, q_e]$，当前 $K$ 块列范围为 $[k_s, k_e]$。因果 mask 下：

整块可跳过的条件是：

$$
k_s > q_e
$$

不能跳过的条件是：

$$
k_s \le q_e
$$

这是一个很容易写错的边界。错误方式常见有两种。

| 错误类型 | 结果 |
|---|---|
| 把“部分无效块”误判为“整块无效” | 丢失合法注意力连接，结果错误 |
| 把“整块无效”漏判成“需要计算” | 正确但浪费算力 |

第一种是功能错误，第二种是性能错误。前者更危险，因为它未必会立刻报错，可能只是训练变差、loss 异常、收敛变慢。

### 坑 3：只关注 GEMM，不关注中间状态读写

FlashAttention-2 的收益不是“某次矩阵乘更快”这么简单，而是：

- 少读了很多次 $Q$
- 少写了很多次 $O$
- 少做了很多次跨 warp 归并

如果实现中仍然频繁把部分输出写回显存，再从显存读回来继续累计，那么你虽然名义上采用了“按 Q 外层”的组织方式，实际仍然保留了旧版的大部分开销。

下面这张表可以帮助区分“真正省掉了什么”。

| 场景 | 是否执行块乘法 | 是否更新 softmax 统计 | 是否产生非零贡献 |
|---|---|---|---|
| 整块可跳过 | 否 | 否 | 否 |
| 整块未跳过但实际全无效 | 是 | 是 | 否 |
| 部分有效块 | 是 | 是 | 是 |

这里的关键不是“跳过后某个公式更短”，而是**连矩阵乘和 softmax 更新都不做了**。这就是因果场景下额外 1.7x 到 1.8x 提速的来源之一。

### 坑 4：mask 元数据和块调度边界没对齐

真实工程中，输入不一定是规则的满长序列，还可能包括：

- padding
- 变长序列
- packed sequence
- grouped-query attention
- multi-query attention

这时“逻辑 token 边界”和“物理 block 边界”可能不完全对齐。如果 mask 元数据仍按满长规则生成，就可能把一个“部分有效块”误判成“全无效块”。这类问题通常不会以崩溃形式出现，而是表现为：

- 训练不稳定
- 验证精度异常
- 特定 batch 下出现偏差

### 坑 5：数值稳定更新只写对一半

online softmax 看起来公式不长，但实现容易出错的点不少：

- 忘了在块内先减局部最大值
- 把 `block_o` 当成未归一化值使用
- 合并旧块和新块时缩放因子写错
- $l'$ 更新后，$o'$ 没有同步按新分母归一化

一旦这些细节有误，短序列和小数值样例可能看不出来，但长序列和大 logits 时就会暴露。

### 一个实用判断原则

如果你在 profile 中看到下面几类现象，通常说明 FA-2 风格重排是值得考虑的。

| 现象 | 含义 |
|---|---|
| HBM 读写占主导 | 说明中间状态和重复读取成本过高 |
| 同步等待明显 | 说明输出块被过度拆分，需要更多归并 |
| causal 场景下计算量仍接近 full attention | 说明无效块跳过做得不够彻底 |
| 长上下文吞吐下降过快 | 说明块数增加后重复访存成本被放大 |

---

## 替代方案与适用边界

FlashAttention-2 很强，但不是唯一答案，也不是没有边界。判断是否适合它，关键不在“它是不是最新”，而在于：**你要解决的瓶颈到底是显存、同步，还是算法复杂度本身。**

### 方案一：原版 FlashAttention

原版 FlashAttention 同样属于精确注意力，也同样依赖块化和 online softmax 来降低显存压力。它的主要差别不在数学定义，而在并行工作划分没有 FA-2 那么激进。

它的优点是：

- 结构更早被广泛理解和实现
- 某些复杂 mask 逻辑下更容易调试
- 对“先求稳再求极致吞吐”的场景更友好

它的局限是：

- 重复读取 $Q$ 的问题更明显
- 同一个输出块更可能被多个 warp 协作更新
- 长上下文下同步与中间状态回写成本更高

### 方案二：近似注意力

另一类替代方案是近似注意力，例如：

- 稀疏注意力
- 低秩近似
- 核方法近似
- 状态空间模型替代注意力

这类方案追求的是**从算法复杂度层面减少计算量**，例如把某些场景下的二次复杂度压到近线性或亚二次。它们解决的问题和 FlashAttention-2 不同。

| 类别 | 目标 | 是否精确等价 |
|---|---|---|
| FlashAttention / FA-2 | 把精确注意力排得更顺 | 是 |
| 稀疏 / 低秩 / SSM 替代 | 减少理论复杂度 | 通常否 |

如果你的上下文长度已经长到“即使把精确注意力排得再顺也难以承受”，那么该考虑的可能就不是 FA-2，而是近似或结构替代。

### 方案三：受硬件限制的保守切分

FA-2 依赖较强的片上缓存利用与 warp 级工作分配。如果硬件资源较紧，例如：

- 共享内存预算小
- 可用寄存器少
- SM 数量有限
- Tensor Core 路径受限

那么“一个 warp 负责完整 Q 块”可能不得不缩小块长，甚至退回到更保守的切分方式。此时 FA-2 的思路仍然成立，但最佳参数会变化，收益也可能没有在 A100/H100 这类卡上那么明显。

### 三类方案的对比

| 方案 | 适用上下文长度 | GPU 资源要求 | mask 复杂度适配 | 适用边界 |
|---|---|---|---|---|
| 原版 FlashAttention | 1k 到中长上下文 | 中等 | 较直观 | 实现更保守、调试更容易 |
| FlashAttention-2 | 中长到长上下文 | 较高 | 对标准 causal/non-causal 很强 | 追求吞吐、减少同步 |
| 近似注意力 | 极长上下文 | 视算法而定 | 多样 | 可降复杂度，但非精确等价 |

### 一个更实用的选择表

| 场景 | 更适合的方案 |
|---|---|
| 上下文约 1k，mask 很复杂，调试成本优先 | 原版 FlashAttention |
| 上下文 2k、4k、8k，标准 causal 或 non-causal，目标是提高吞吐 | FlashAttention-2 |
| 上下文继续拉长，精确注意力本身已难承受 | 近似注意力或结构替代 |

因此，FlashAttention-2 的适用边界不是“所有注意力实现都应该切过去”，而是：

**当你仍然需要精确注意力，并且长上下文已经让访存与同步成为主瓶颈时，FlashAttention-2 往往是更合理的并行组织方式。**

换句话说，它最适合回答的问题是：

> 精确注意力不改，怎么让 GPU 少搬数据、少等同步、少做无效块？

而不是：

> 怎么把注意力的理论复杂度彻底改掉？

---

## 参考资料

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. 论文原文，核心来源，包含并行划分、online softmax 组织、A100 上约 225 TFLOPs/s 等数据。https://tridao.me/publications/flash2/flash2.pdf
- FlashAttention 项目仓库。适合对照论文看真实工程接口、支持的 mask 类型、head dim 与硬件约束。https://github.com/Dao-AILab/flash-attention
- NVIDIA CUDA C++ Programming Guide。用于理解 warp、shared memory、occupancy、同步语义等 GPU 基础概念。https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA Best Practices Guide。适合理解为什么“减少 HBM 往返”和“降低同步点”会直接影响内核吞吐。https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Jalexine 的 FlashAttention 1 到 2 解读。适合辅助理解“外层从 $K/V$ 改到 $Q$”后，为什么 $Q$ 重读减少、warp 协作边界改变。https://jalexine.github.io/lab/2026-01-20-tiled-attention-flashattention-1-to-2
- Shenh10 的论文笔记。对块级公式、online softmax 状态更新做了较统一的符号整理，适合快速回忆推导。https://shenh10.github.io/papercache/papers/llm/engineering/attention/2023/07/01/flashattention-2-faster-attention-with-better-parallelism-and-work-partitioning.html
- DigitalOcean 的 FlashAttention-2 教程。偏入门视角，适合建立“每个 warp 负责完整 Q 输出块”的直觉。https://www.digitalocean.com/community/tutorials/flashattention2
- ICLR Blogposts 关于 FlashAttention 演进的文章。可辅助理解因果 mask、块跳过与工程收益之间的关系。https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/
- M. Brenndoerfer 的实现分析。适合理解块大小、HBM 访问、Q 重读次数之间的关系。https://mbrenndoerfer.com/writing/flashattention-implementation-gpu-memory-optimization

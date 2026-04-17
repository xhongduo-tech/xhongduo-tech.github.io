## 核心结论

FlashAttention 的因果 Mask 并不是先构造一个 $N \times N$ 的上三角布尔矩阵，再把被遮住的位置填成 $-\infty$。它的做法更接近“按块计算时直接不看那些必然无效的区域”。

具体说，序列被按固定大小切成 block，查询矩阵 $Q$ 和键矩阵 $K$ 的乘法不再一次性生成完整注意力分数矩阵，而是只计算需要的 block。对于因果注意力，位于主对角线右上方、完全属于“未来 token”区域的 block，可以直接跳过；只有穿过对角线的边界 block，才需要在 block 内做逐行 mask。

这件事的意义有两层：

| 维度 | 标准 causal attention | FlashAttention causal |
|---|---|---|
| Mask 表示 | 显式构造或等价应用 $N \times N$ mask | 不显式存整个三角矩阵 |
| 被遮挡区域 | 通常仍参与分数计算，再被 mask 掉 | 完全遮挡的 block 直接不算 |
| 计算量 | 仍接近完整 $O(N^2)$ 分数计算 | 因果场景约为非因果的一半 |
| 访存压力 | 需要读写更大的中间结果 | 中间结果按块流式处理 |

结论可以压缩成一句话：FlashAttention 的因果 Mask，本质是“块级跳过 + 边界块行级遮挡”，而不是“先算完整矩阵再裁掉上三角”。

---

## 问题定义与边界

先定义问题。因果注意力，英文是 causal attention，意思是“当前位置只能看见自己和过去，不能看未来”。如果第 $i$ 个 query 对第 $j$ 个 key 计算注意力，那么只允许 $j \le i$。

标准公式是：

$$
A_{ij} =
\begin{cases}
\frac{Q_i K_j^\top}{\sqrt{d}} & j \le i \\
-\infty & j > i
\end{cases}
$$

再对每一行做 softmax：

$$
P_{ij} = \text{softmax}(A_i)_j
$$

问题在于，很多实现虽然语义上“屏蔽了未来”，但工程上仍然先算了大量本来就不该存在的 $QK^\top$ 项，再把它们盖掉。这会浪费两种资源：

| 资源 | 为什么浪费 |
|---|---|
| FLOPs | 被 mask 的位置仍然做了乘加 |
| 带宽 | 中间分数矩阵需要更多读写 |

FlashAttention 的边界也要说清楚。它特别适合“规则、可预测”的 mask，尤其是自回归生成里的 causal mask。因为这种 mask 的形状固定，block 是否可跳过可以提前由几何关系判断。

它不天然解决所有稀疏 mask 问题。比如：

| Mask 类型 | 是否容易用 block 跳过 |
|---|---|
| 因果 mask | 容易，结构固定 |
| 局部窗口 mask | 较容易，可按区间判断 |
| 文档段落稀疏 mask | 取决于模式是否规则 |
| 任意动态逐元素 mask | 不容易，往往需要更通用方案 |

所以本文边界是：讨论 FlashAttention 在因果 Mask 下的实现，不展开任意稀疏模式的通用优化。

---

## 核心机制与推导

### 1. 从逐元素 mask 变成按块判断

假设序列长度为 $N$，block 大小为 $B$，那么整个注意力矩阵会被切成 $\lceil N/B \rceil \times \lceil N/B \rceil$ 个 block。

第 $i$ 个 query block 覆盖的行区间是：

$$
[q_s, q_e) = [iB, \min((i+1)B, N))
$$

第 $j$ 个 key block 覆盖的列区间是：

$$
[k_s, k_e) = [jB, \min((j+1)B, N))
$$

对因果 mask，有三种 block 状态：

| block 类型 | 条件 | 处理方式 |
|---|---|---|
| 全遮挡 | $k_s \ge q_e$ | 整块跳过 |
| 部分遮挡 | $k_s < q_e$ 且 $k_e > q_s$ | 计算，但做行级 mask |
| 完全可见 | $k_e \le q_s$ | 正常计算，无需 mask |

这里的“行级 mask”意思是：block 内不是所有元素都被遮挡，而是某些 query 行只能看到该 block 的前半部分 key。

### 2. 玩具例子：长度 8，块大小 2

令 $N=8, B=2$，那么共有 $4 \times 4 = 16$ 个 block。

block 坐标矩阵可以写成：

| q\k | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | partial | full-mask | full-mask | full-mask |
| 1 | visible | partial | full-mask | full-mask |
| 2 | visible | visible | partial | full-mask |
| 3 | visible | visible | visible | partial |

原因很直接：

- 对角线上的 block 是 partial，因为其中一部分元素满足 $j \le i$，另一部分满足 $j > i$
- 左下角 block 完全可见
- 右上角 block 完全被遮挡

所以 16 个 block 中：
- 4 个 partial
- 6 个 visible
- 6 个 full-mask

如果 block 足够大、序列足够长，因果场景下需要处理的区域会逼近整个矩阵的一半，这就是“FLOPs 约为非因果 attention 的 50%”的来源。

### 3. 为什么 partial block 必须逐行 mask

对一个 partial block，不能因为“这个块大部分有效”就整块保留。因为块内每一行的可见 key 上界不同。

设 query 全局索引为 `q_idx`，key 全局索引为 `k_idx`，则 mask 规则是：

$$
\text{mask}(q\_idx, k\_idx)=
\begin{cases}
0 & k\_idx \le q\_idx \\
-\infty & k\_idx > q\_idx
\end{cases}
$$

于是 block 内分数是：

$$
S = \frac{Q_{\text{block}} K_{\text{block}}^\top}{\sqrt{d}} + M_{\text{block}}
$$

然后再做 softmax。这里关键是：只有 partial block 需要加 $M_{\text{block}}$；full-mask block 直接不算，visible block 直接走普通路径。

### 4. 为什么数值上仍然正确

FlashAttention 的核心不是改数学定义，而是改计算顺序。它通过 streaming softmax，也就是“流式 softmax”，在分块遍历时维护每一行的局部最大值与归一化和，最后得到与 dense softmax 一致的结果。

因此，因果 mask 的优化只改变“哪些块参与计算”，不改变最终语义。只要 partial block 的逐行 mask 正确，结果就应与显式三角 mask 的 dense 实现一致。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用来验证“块级分类 + partial 行级 mask”与 dense causal mask 结果一致。

```python
import math

def matmul(a, b):
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i][t] * b[t][j]
            out[i][j] = s
    return out

def transpose(x):
    return [list(row) for row in zip(*x)]

def softmax_row(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [v / s for v in exps]

def dense_causal_attention(q, k, v):
    n, d = len(q), len(q[0])
    scores = matmul(q, transpose(k))
    scale = 1.0 / math.sqrt(d)
    for i in range(n):
        for j in range(n):
            scores[i][j] *= scale
            if j > i:
                scores[i][j] = float("-inf")
    probs = [softmax_row(row) for row in scores]
    return matmul(probs, v)

def flash_like_causal_attention(q, k, v, block_size):
    n, d = len(q), len(q[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0 for _ in range(len(v[0]))] for _ in range(n)]

    for qi in range(0, n, block_size):
        q_end = min(qi + block_size, n)
        scores_rows = [[] for _ in range(qi, q_end)]

        for kj in range(0, n, block_size):
            k_end = min(kj + block_size, n)

            # full-mask block: key 起点已经在 query block 结束右侧
            if kj >= q_end:
                continue

            q_block = q[qi:q_end]
            k_block = k[kj:k_end]
            local_scores = matmul(q_block, transpose(k_block))

            for r, q_idx in enumerate(range(qi, q_end)):
                row = []
                for c, k_idx in enumerate(range(kj, k_end)):
                    val = local_scores[r][c] * scale
                    # partial block 内按全局索引做行级 mask
                    if k_idx > q_idx:
                        val = float("-inf")
                    row.append(val)
                scores_rows[r].extend(row)

        probs_rows = [softmax_row(row) for row in scores_rows]

        # 按相同顺序收集有效的 V
        visible_v = []
        visible_indices = []
        for kj in range(0, n, block_size):
            k_end = min(kj + block_size, n)
            if kj >= q_end:
                continue
            for k_idx in range(kj, k_end):
                visible_v.append(v[k_idx])
                visible_indices.append(k_idx)

        block_out = matmul(probs_rows, visible_v)
        for r, q_idx in enumerate(range(qi, q_end)):
            out[q_idx] = block_out[r]

    return out

q = [
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 2.0],
]
k = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
]
v = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
]

dense = dense_causal_attention(q, k, v)
flash = flash_like_causal_attention(q, k, v, block_size=2)

for i in range(len(dense)):
    for j in range(len(dense[0])):
        assert abs(dense[i][j] - flash[i][j]) < 1e-9

print("ok")
```

这个例子故意没有写 GPU kernel，而是把逻辑拆开，保留两个最重要的判断：

| `block_type` | 处理路径 |
|---|---|
| `FULL_MASKED` | `continue` |
| `PARTIAL` | 构造行级 mask 后参与 softmax |
| `VISIBLE` | 直接参与普通 attention 计算 |

如果把它压缩成伪代码，就是：

```python
for q_block in query_blocks:
    for k_block in key_blocks:
        if k_start >= q_end:
            continue
        elif crosses_diagonal(q_block, k_block):
            scores = q_block @ k_block.T
            scores += row_causal_mask(q_indices, k_indices)
            update_streaming_softmax(scores)
        else:
            scores = q_block @ k_block.T
            update_streaming_softmax(scores)
```

真实工程里，这段逻辑通常在 CUDA 或 Triton kernel 中实现，并且会把 matmul、mask、softmax、与 $V$ 的乘法融合到更少的访存轮次中。

### 真实工程例子

以长度 2048、block 大小 128 为例，block 网格大小是 $16 \times 16$。非因果 attention 需要遍历全部 256 个 block 对。因果 attention 下，只需要遍历左下三角及对角线附近，总量约为：

$$
\frac{16 \times 17}{2} = 136
$$

其中：
- 严格左下角的大部分 block 是 visible
- 对角线 block 是 partial
- 右上角 120 个 block 可直接跳过

这就是部署长上下文生成模型时，FlashAttention 在因果场景能明显减少算力和带宽消耗的直接原因。

---

## 工程权衡与常见坑

FlashAttention 的优势不是“数学更省”，而是“实现更接近真正需要计算的区域”。但这种优化带来了一些严格的工程要求。

| 失败情境 | 后果 | 规避动作 |
|---|---|---|
| partial block 没做逐行 mask | 未来 token 泄露，结果错误 | 用全局 `q_idx/k_idx` 判定，而不是局部列号猜测 |
| block 分类条件写错 | 本该跳过的块被计算，吞吐下降 | 统一使用区间条件：`k_start >= q_end` 判 full-mask |
| 对角线 block 当成 visible | 结果错误，不只是慢 | 单独测试 diagonal blocks |
| metadata 没预处理 | kernel 内分支过多，调度变差 | 预先生成 block 类型摘要 |
| 最后一块不足 B 大小处理错误 | 尾块越界或 mask 错位 | 区间全部用 `min((i+1)B, N)` |

一个常见误解是：“既然有 mask，直接在 kernel 里每个元素判断一下不就行了？”理论上可以，但这样会损失 FlashAttention 的核心收益。因为你还是在访问、计算、判断那些原本完全可以跳过的区域。

另一个常见坑是把“partial”理解成“整个对角线块都要慢路径”。更准确的说法是：只有对角线块内部分元素需要遮挡。工程实现通常会把这种慢路径局部化，只让必要的行进入 mask 分支，而不是把全部块都按最坏情况处理。

---

## 替代方案与适用边界

如果 mask 不是标准因果三角形，FlashAttention 的 block skip 思路仍然有参考价值，但未必够用。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| Dense mask | 任意 mask | 实现最直接 | 计算和访存最重 |
| FlashAttention causal | 静态因果 mask | 对生成式推理非常高效 | 主要适合规则模式 |
| FlashMask | 规则稀疏区间 mask | 可按区间摘要跳过 block | 需要额外 metadata |
| FlexAttention | 更灵活的 mask 函数 | 适合动态规则和实验 | 实现复杂，性能依赖模式 |

可把选择原则记成一句话：

- mask 规则固定且强结构化，优先用块级跳过
- mask 稍复杂但可压缩成区间摘要，可考虑 FlashMask 类方案
- mask 高度动态或逐元素变化，往往需要更通用的框架，不能只靠 causal tiling

因此，FlashAttention 的因果 Mask 不是“通用稀疏注意力”的终点，而是“规则 mask 场景下最成功的一类专用优化”。

---

## 参考资料

- Hugging Face, “FlashAttention basics”: https://huggingface.co/blog/atharv6f/flash-attention-basics
- NVIDIA Developer Blog, “Tuning FlashAttention for Peak Performance in NVIDIA CUDA Tile”: https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
- PaddleNLP 文档, “FlashMask”: https://paddlenlp.readthedocs.io/en/latest/llm/docs/flashmask.html
- Emergent Mind, “FlashMaskedAttention / FlexAttention”: https://www.emergentmind.com/topics/flashmaskedattention

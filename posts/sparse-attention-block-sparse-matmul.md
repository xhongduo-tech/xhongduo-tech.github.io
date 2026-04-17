## 核心结论

块稀疏注意力的核心不是“把注意力矩阵里一部分元素置零”，而是把序列按固定大小切成 block，然后只对被允许的 block 对做普通的 dense attention。真正被 GPU 执行的仍然是规则的矩阵乘，只是执行对象从完整的 `N×N` token 关联矩阵，变成了少量被选中的 `B×B` 子块。

这里先统一记号：

| 记号 | 含义 |
|---|---|
| $N$ | 序列长度 |
| $d$ | 单个 attention head 的 hidden 维度 |
| $B$ | block 大小，表示每个 block 包含多少个 token |
| $n_b = N/B$ | block 总数 |
| $k$ | 每个 query block 保留的 key block 数 |

在 dense attention 中，所有 query token 都要和所有 key token 计算相似度，因此主计算量近似为：

$$
\text{FLOP}_{\text{dense}} \approx O(N^2 d)
$$

在 block-sparse attention 中，若每个 query block 只保留 $k$ 个 key block，则有效 block 对数大约是 $n_b \cdot k$，每个 block 内部仍要做一次 `B×d` 与 `B×d` 的 dense 乘法，因此主计算量近似为：

$$
\text{FLOP}_{\text{block-sparse}} \approx O\left(\frac{N}{B}\cdot k \cdot B^2 \cdot d\right)=O(NkBd)
$$

把两者相除：

$$
\frac{\text{FLOP}_{\text{block-sparse}}}{\text{FLOP}_{\text{dense}}}
\approx
\frac{kB}{N}
=
\frac{k}{N/B}
$$

这个式子比口头描述更重要。它说明收益主要取决于“每一行 block 保留多少列”，也就是 $k$ 相对于总 block 数 $N/B$ 的比例，而不是只取决于 block 大小本身。

玩具例子：

设 $N=4096$，$B=64$，则总 block 数是：

$$
n_b = \frac{4096}{64} = 64
$$

如果是 dense attention，每个 query block 都看全部 64 个 key block，总 block 对数是：

$$
64 \times 64 = 4096
$$

如果改成 block-sparse，并且每行只保留 $k=16$ 个 block，那么总 block 对数变成：

$$
64 \times 16 = 1024
$$

这时只计算原来四分之一的 block 对。进一步把 block 内部的 dense score 计算也算进去，则：

$$
\text{work}_{\text{dense}} = 64 \times 64 \times 64^2 = 16{,}777{,}216
$$

$$
\text{work}_{\text{sparse}} = 64 \times 16 \times 64^2 = 4{,}194{,}304
$$

二者比例仍然是：

$$
\frac{4{,}194{,}304}{16{,}777{,}216} = 0.25
$$

结论可以压缩成一句话：块稀疏注意力的理论收益来自“减少有效 block 对数”，工程收益来自“让这些 block 仍然足够规则，能映射到 GPU tile、warp 和 tensor core”。这也是为什么 `64×64` 或 `128×128` 这类块大小在工程里常见，它们更容易和底层 kernel 的 tile 对齐。

还要补一个现实判断。理论上 75% 的稀疏意味着主计算量下降到 25%；但真实时延并不会严格按这个比例下降，因为 gather、scatter、mask 元数据、kernel dispatch、缓存命中率和 occupancy 都会影响最终结果。公开资料和社区实践里，4K 及以上长序列在稀疏模式规整时，看到约 2 到 3 倍收益并不罕见；而短序列时，固定开销往往会吞掉理论优势。

---

## 问题定义与边界

先把问题说清楚。标准 attention 的核心步骤是：

1. 用 $QK^\top$ 计算 query 与 key 的相关分数。
2. 对分数做 softmax，把每一行归一化成概率分布。
3. 用这些概率去加权求和 $V$，得到输出。

对新手来说，可以把它理解成“每个 token 都问一遍：序列里哪些位置和我最相关”。如果序列长度是 $N$，那么每个 token 都要和另外 $N$ 个 token 建立关系，总关系数是 $N^2$ 量级，所以复杂度是：

$$
O(N^2 d)
$$

这就是 dense attention 的根本瓶颈。长度从 512 增长到 4096，不是增加 8 倍，而是相关对数增加到原来的 $8^2=64$ 倍。GPU 在这里不只是在做更多乘加，还要搬运更多中间分数、更大的 mask、更大的 softmax 工作集。

块稀疏 attention 的问题定义则不同。它先把 token 轴切成固定大小的 block，再让 kernel 在 block 层面决定“算还是不算”。于是基本计算单元不再是单个 token 对，而是整个 `B×B` 子矩阵。

| 维度 | Dense Attention | Block-Sparse Attention |
|---|---|---|
| 计算单元 | 单个 token 对 | 一个 `B×B` block |
| 主要数据结构 | 连续 `Q/K/V` 张量 | `Q/K/V` + block mask / block index |
| 主复杂度 | $O(N^2 d)$ | $O(\frac{N}{B}\cdot k\cdot B^2 d)$ |
| 元数据开销 | 近乎无 | 需要记录 block 是否有效 |
| GPU 访存 | 连续、规则 | 若设计得当仍可规则；若设计差会碎片化 |
| 长序列表现 | 稳定但贵 | 稀疏率高时更有机会加速 |
| 短序列表现 | 往往更稳 | 常被元数据和调度开销反超 |

如果你刚接触这个概念，可以用下面这个判断替代抽象术语：

- dense attention 问的是：“第 137 个 token 要不要看第 2890 个 token？”
- block-sparse attention 先问的是：“第 3 个 query block 要不要看第 45 个 key block？”
- 只有 block 被允许时，block 内部的 token 才进入真正的 attention 计算。

这就是“结构化稀疏”和“随机稀疏”的本质差别。前者先在粗粒度上裁剪，再在保留下来的区域里做普通 dense 运算；后者则可能在元素级别随机保留关系，数学上更细，但 GPU 更难高效执行。

这里有两个边界需要明确。

第一，PyTorch 文档中的 `flex_attention` 内核示例常见 `BLOCK_M=64`、`BLOCK_N=64`，但 `create_block_mask(..., BLOCK_SIZE=...)` 这个工具接口在文档里给出的默认值是 `128`。所以“工程里经常手动指定 64×64”成立，但“PyTorch 默认就是 64×64”不成立。两者不是一回事。

第二，块稀疏不是在任何长度下都划算。PyTorch issue #141129 中，开发者给出的案例明确指出：在较短序列上，特别是低于约 1000 的长度时，block sparse mask 可能明显慢于标准 `scaled_dot_product_attention`。这不是算法错误，而是固定成本还没被长序列摊薄。

---

## 核心机制与推导

块稀疏实现难的地方，不在于“哪些位置是 0”，而在于“怎么让跳过这些位置的过程本身也高效”。

如果按元素级稀疏去保存 attention 矩阵，会立刻遇到两个 GPU 不喜欢的问题：

1. 非零元素分布不连续，访存不规则。
2. 每个非零元素都要带索引，元数据成本很高。

所以工程实现通常不使用元素级 COO/CSR 式稀疏，而是使用 block mask。block mask 可以直观理解成一个大小为 $n_b \times n_b$ 的布尔矩阵：

$$
M \in \{0,1\}^{n_b \times n_b}, \quad n_b = \frac{N}{B}
$$

其中：

- $M_{ij}=1$ 表示第 $i$ 个 query block 允许访问第 $j$ 个 key block
- $M_{ij}=0$ 表示这一整块都跳过

如果第 $i$ 行只保留 $k$ 个 key block，那么总有效 block 数约为：

$$
n_b \cdot k = \frac{N}{B}\cdot k
$$

而每个有效 block 内部，仍然执行标准 attention 的局部子问题。设：

- $Q_i \in \mathbb{R}^{B \times d}$
- $K_j \in \mathbb{R}^{B \times d}$
- $V_j \in \mathbb{R}^{B \times d}$

则一个有效 block 对应的 score 计算是：

$$
S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B \times B}
$$

单个 block 的乘法代价近似是：

$$
O(B^2 d)
$$

于是总主计算量就是：

$$
\text{FLOP}_{\text{block-sparse}}
\approx
\frac{N}{B}\cdot k \cdot B^2 d
=
NkBd
$$

与 dense 对比：

$$
\frac{\text{FLOP}_{\text{block-sparse}}}{\text{FLOP}_{\text{dense}}}
\approx
\frac{NkBd}{N^2d}
=
\frac{kB}{N}
=
\frac{k}{N/B}
$$

这个式子可以再换一种写法。定义 block 密度 $s$ 为保留的 block 比例：

$$
s = \frac{k}{n_b} = \frac{k}{N/B}
$$

那么复杂度也可写成：

$$
O(sN^2d)
$$

但这个写法容易隐藏掉一个工程事实：即使 block 密度相同，不同的 $B$ 也会改变元数据规模、访存连续性和 kernel tile 的映射效果。因此在实现层面，通常仍然显式保留 $B$ 和 $k$ 两个量来分析。

再把玩具例子完整展开一次。设：

- $N=4096$
- $B=64$
- $n_b=64$
- 每行保留 $k=16$

则理论比例是：

$$
\frac{k}{N/B}=\frac{16}{64}=0.25
$$

也就是主算量约为 dense 的 25%。这时如果每个 query block 都看“前后 2 个邻居 block + 第 0 个全局 block + 若干检索回来的内容块”，你就得到一个同时兼顾局部上下文和少量远程依赖的结构化稀疏模式。

kernel 级数据流可以用一个更完整的伪代码概括：

```text
for each q_block i:
    cols = kv_indices[i]                 # 当前 query block 允许访问的 key/value block 列表
    Q_i = load(Q block i)
    state_m, state_l, state_o = init()   # 流式 softmax 所需状态

    for each kv_block j in cols:
        K_j, V_j = load(K/V block j)
        S_ij = Q_i @ K_j^T
        S_ij = apply_scale_and_mask(S_ij)
        state_m, state_l, state_o = online_softmax_update(S_ij, V_j, state_m, state_l, state_o)

    O_i = finalize(state_o, state_l)
    store(O_i)
```

这里有三个要点。

第一，真正高性能的实现不会先把所有有效 `S_ij` 全部落到显存，再统一做 softmax。它更常见的做法是像 FlashAttention 那样，边读块边维护在线 softmax 状态 `m/l/O`，尽量把中间结果留在寄存器或片上 SRAM 中。

第二，`gather` 和 `scatter` 不是可有可无的细节。它们决定了 K/V block 是否能连续读入，以及输出是否能顺序写回。如果 block 索引设计得太碎，理论 FLOP 降了，实际吞吐却未必提高。

第三，block-sparse 的优势来自“保留规则块”，而不是来自“把矩阵变稀”。同样的稀疏率下，随机 token-sparse 往往比 block-sparse 更难跑快，因为它破坏了 GPU 喜欢的连续 tile 结构。

真实工程里，长上下文推理是最典型的适用场景。例如文档问答、检索增强生成、长摘要等任务中，很多 token 并不需要看完整历史，而只需要：

- 当前附近的局部窗口
- 系统提示或开头摘要
- 少量检索命中的远程段落
- 少量结构锚点，如标题、分节符、表格开头

这些访问模式天然更适合表示成“局部带状 block + 少量全局 block”，而不是元素级随机稀疏。

---

## 代码实现

下面先给一个完全可运行的 Python 版本。它不依赖 GPU，目标不是测速，而是把 block-sparse 的计数逻辑、mask 结构和一个正确的前向实现讲清楚。你可以直接复制运行。

第一段代码只验证前文公式。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BlockStats:
    seq_len: int
    block_size: int
    n_blocks: int
    dense_pairs: int
    sparse_pairs: int
    dense_score_work: int
    sparse_score_work: int
    ratio: float


def count_dense_block_pairs(seq_len: int, block_size: int) -> int:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    n_blocks = seq_len // block_size
    return n_blocks * n_blocks


def count_sparse_block_pairs(seq_len: int, block_size: int, k: int) -> int:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    n_blocks = seq_len // block_size
    if not (0 <= k <= n_blocks):
        raise ValueError("k must satisfy 0 <= k <= n_blocks")
    return n_blocks * k


def estimate_score_work(seq_len: int, block_size: int, k: int) -> BlockStats:
    n_blocks = seq_len // block_size
    dense_pairs = count_dense_block_pairs(seq_len, block_size)
    sparse_pairs = count_sparse_block_pairs(seq_len, block_size, k)
    block_area = block_size * block_size
    dense_score_work = dense_pairs * block_area
    sparse_score_work = sparse_pairs * block_area
    ratio = sparse_score_work / dense_score_work if dense_score_work else 0.0
    return BlockStats(
        seq_len=seq_len,
        block_size=block_size,
        n_blocks=n_blocks,
        dense_pairs=dense_pairs,
        sparse_pairs=sparse_pairs,
        dense_score_work=dense_score_work,
        sparse_score_work=sparse_score_work,
        ratio=ratio,
    )


if __name__ == "__main__":
    stats = estimate_score_work(seq_len=4096, block_size=64, k=16)

    assert stats.n_blocks == 64
    assert stats.dense_pairs == 64 * 64
    assert stats.sparse_pairs == 64 * 16
    assert stats.dense_score_work == 16_777_216
    assert stats.sparse_score_work == 4_194_304
    assert abs(stats.ratio - 0.25) < 1e-12

    print(stats)
```

输出会告诉你三件事：

1. 总共有多少个 block。
2. dense 需要计算多少个 block 对。
3. sparse 之后主 score 计算缩减到什么比例。

第二段代码实现一个“能算出正确结果”的 NumPy 版 block-sparse attention。它比前面的计数脚本更接近真实实现，因为它真的会根据 block 索引去 gather K/V，再做 softmax 和加权求和。

```python
from __future__ import annotations

import math
from typing import List

import numpy as np


def build_local_global_block_index(n_blocks: int, local_radius: int) -> List[List[int]]:
    """
    每个 query block 保留:
    1. 自己前后 local_radius 个邻居
    2. 第 0 个全局 block
    """
    rows: List[List[int]] = []
    for q_block in range(n_blocks):
        cols = {0}
        left = max(0, q_block - local_radius)
        right = min(n_blocks - 1, q_block + local_radius)
        for kv_block in range(left, right + 1):
            cols.add(kv_block)
        rows.append(sorted(cols))
    return rows


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    q, k, v: [seq_len, dim]
    """
    dim = q.shape[-1]
    scores = q @ k.T / math.sqrt(dim)
    probs = softmax(scores, axis=-1)
    return probs @ v


def block_sparse_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int,
    kv_indices: List[List[int]],
) -> np.ndarray:
    """
    q, k, v: [seq_len, dim]
    kv_indices[q_block] = 当前 query block 可以访问的 key/value block 列表
    """
    seq_len, dim = q.shape
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")

    n_blocks = seq_len // block_size
    if len(kv_indices) != n_blocks:
        raise ValueError("len(kv_indices) must equal n_blocks")

    out = np.zeros_like(q)

    for q_block in range(n_blocks):
        q_start = q_block * block_size
        q_end = q_start + block_size
        q_block_tensor = q[q_start:q_end]  # [B, d]

        cols = kv_indices[q_block]
        gathered_k = []
        gathered_v = []

        for kv_block in cols:
            kv_start = kv_block * block_size
            kv_end = kv_start + block_size
            gathered_k.append(k[kv_start:kv_end])
            gathered_v.append(v[kv_start:kv_end])

        cat_k = np.concatenate(gathered_k, axis=0)  # [active_blocks * B, d]
        cat_v = np.concatenate(gathered_v, axis=0)  # [active_blocks * B, d]

        scores = q_block_tensor @ cat_k.T / math.sqrt(dim)
        probs = softmax(scores, axis=-1)
        out[q_start:q_end] = probs @ cat_v

    return out


if __name__ == "__main__":
    np.random.seed(0)

    seq_len = 256
    dim = 32
    block_size = 64
    n_blocks = seq_len // block_size

    q = np.random.randn(seq_len, dim).astype(np.float32)
    k = np.random.randn(seq_len, dim).astype(np.float32)
    v = np.random.randn(seq_len, dim).astype(np.float32)

    kv_indices = build_local_global_block_index(n_blocks=n_blocks, local_radius=1)

    out_sparse = block_sparse_attention(q, k, v, block_size, kv_indices)

    # 这里只验证 shape 和数值稳定性；它不会等于 dense，
    # 因为 sparse 故意丢弃了一部分远程连接。
    assert out_sparse.shape == (seq_len, dim)
    assert np.isfinite(out_sparse).all()

    print("n_blocks =", n_blocks)
    print("kv_indices =", kv_indices)
    print("output shape =", out_sparse.shape)
```

这段代码的价值在于把工程里的三个动作拆开让你看清楚：

- `kv_indices[q_block]`：元数据，说明当前 query block 能看哪些 key/value block
- `np.concatenate(gathered_k, axis=0)`：按索引 gather 出有效 K/V
- `probs @ cat_v`：在被保留的区域里做标准 dense attention

如果你想把它和 dense 版本直接对照，可以把 block-sparse 的 mask 设成“每个 query block 都能看所有 block”。这时 block-sparse 退化为 dense，输出应与 dense attention 数值一致。

下面给一个更接近实际训练/推理环境的 PyTorch 示例。这个示例依赖 CUDA 与 `torch.nn.attention.flex_attention`。不同 PyTorch 版本接口可能有细微差异，但文档中的核心思路是稳定的。

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

BATCH = 1
HEADS = 8
SEQ = 4096
DIM = 64
BLOCK = 64
DEVICE = "cuda"
DTYPE = torch.float16

def local_plus_global_mask(batch_idx, head_idx, q_idx, kv_idx):
    q_block = q_idx // BLOCK
    kv_block = kv_idx // BLOCK

    local_ok = (kv_block >= q_block - 2) & (kv_block <= q_block + 2)
    global_ok = kv_block == 0
    causal_ok = kv_idx <= q_idx  # 如果需要因果掩码，可加这一项

    return (local_ok | global_ok) & causal_ok

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    q = torch.randn(BATCH, HEADS, SEQ, DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn(BATCH, HEADS, SEQ, DIM, device=DEVICE, dtype=DTYPE)
    v = torch.randn(BATCH, HEADS, SEQ, DIM, device=DEVICE, dtype=DTYPE)

    block_mask = create_block_mask(
        local_plus_global_mask,
        B=BATCH,
        H=HEADS,
        Q_LEN=SEQ,
        KV_LEN=SEQ,
        device=DEVICE,
        BLOCK_SIZE=BLOCK,  # 显式指定，避免把工具默认值当成工程默认值
    )

    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "PRESCALE_QK": True,
    }

    out = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        kernel_options=kernel_options,
    )

    print("output shape:", out.shape)
    print("dtype:", out.dtype)
    print("finite:", torch.isfinite(out).all().item())

if __name__ == "__main__":
    main()
```

如果你对 `BlockMask` 的内部结构不熟，最低限度只需要记住下面这张表：

| 结构 | 含义 |
|---|---|
| `kv_num_blocks` | 每个 query block 保留多少个 key/value block |
| `kv_indices` | 这些保留 block 的列索引 |
| `q_num_blocks` | 反向传播时，从列方向回看需要的 query block 数 |
| `q_indices` | 与 `q_num_blocks` 对应的 query block 索引 |
| `full_kv_indices` | 可选优化，表示整块都有效、无需额外元素级判断的区域 |

这也是 PyTorch 文档把 `BlockMask` 设计成“面向 attention 访问模式的专用结构”的原因。它不是通用稀疏矩阵格式，而是围绕 attention 的前向与反向数据流组织的。

---

## 工程权衡与常见坑

块稀疏注意力是否真的更快，不能只看稀疏率，还要同时看 block 大小、序列长度、索引规律性、mask 生成代价和底层 kernel 是否足够融合。

先给一张经验表：

| 因素 | 好处 | 风险 |
|---|---|---|
| block 大 | 元数据少，访存更连续，更容易吃到 tensor core | 掩码更粗，可能保留很多无效 token |
| block 小 | 稀疏更细，表达能力更强 | 索引多、调度碎、kernel 开销更高 |
| 稀疏率高 | 理论主算量下降明显 | gather/scatter 与 mask 成本占比上升 |
| 序列长 | 更容易摊薄固定开销 | block 数变多，索引管理复杂 |
| 模式规整 | GPU 友好，缓存命中更好 | 表达灵活性稍弱 |
| 模式动态 | 能自适应内容 | 前处理和重排成本可能过高 |

最常见的坑有四类。

第一，短序列翻车。  
这是最容易误判的一类问题。你看到公式下降了，就以为时延也会线性下降，但短序列里常常相反。原因通常不是算子主体慢，而是这些固定成本没被摊薄：

- mask 构造
- block 索引生成与压缩
- gather/scatter 访存
- autotune 或 kernel dispatch
- 稀疏格式与 dense 格式之间的边界处理

PyTorch issue #141129 就明确给出了这类现象的案例：在低于约 1000 的序列长度时，block sparse mask 可能明显慢于标准 SDPA。

第二，把 FLOP 节省直接等价为时延节省。  
GPU 程序不只受乘加数影响，还受以下因素影响：

| 指标 | 对时延的影响 |
|---|---|
| Memory coalescing | 访存越连续，吞吐越稳定 |
| Occupancy | 活跃线程束不足会浪费 SM |
| Warp divergence | 不同线程路径分叉会降低效率 |
| Cache locality | K/V block 复用高时更有利 |
| Kernel fusion | 中间张量少落地时更快 |

因此“主计算量下降 75%”更准确的表达应该是：“有条件获得显著加速”，而不是“必然获得 4 倍提速”。

第三，block 太大导致表达过粗。  
这是模型效果层面的风险。某些任务的重要依赖可能只落在远处很短的一段 token 上。如果 block 太大，为了保住这几个 token，你不得不保留整个 block，于是：

- 稀疏率下降
- 不相关 token 被一起算进去
- 理论节省被 block 粗粒度稀释

第四，mask 本身生成太慢。  
动态稀疏模式最容易踩这个坑。比如你希望每步都根据当前 hidden state 决定保留哪些块，那么“选块”这一步本身可能就要做额外计算，甚至引入新的排序、打分和重排。结果是 kernel 虽然省了算力，前处理却把节省吃掉了。很多线上系统最后会退回以下结构，就是因为它们更稳定、更便于缓存和复用：

- 固定局部窗口
- 局部窗口 + 少量全局块
- 条带状或对角带状 block pattern
- 静态模板 + 少量动态修正

经验上可以把适用范围写成一个更实用的判断表：

| 序列长度 | 典型建议 |
|---|---|
| `< 1000` | 优先 dense attention 或 FlashAttention |
| `1000 - 4096` | 不做假设，直接 benchmark |
| `>= 4096` | 若每行只保留少量规整 block，block-sparse 更有机会赢 |
| `>= 8192` | 若模式稳定，block-sparse 的收益通常更容易体现 |

如果要给新手一个更务实的工程判断，可以直接用下面这条：

> 先看序列是否足够长，再看稀疏模式是否足够规整，最后才看理论稀疏率有多高。

---

## 替代方案与适用边界

块稀疏不是“更高级的 attention”，而是“在特定长度和访问模式下，更合算的实现组织方式”。如果目标不同，最优方案也不同。

如果你的目标是短序列低时延，通常更稳的选择是 dense attention 搭配 FlashAttention 一类 IO-aware kernel。这里的 IO-aware 指的是：实现会尽量减少中间分数矩阵的显存读写，把更多状态保留在寄存器或片上 SRAM 中，从而降低实际带宽压力。

如果你的目标是长序列省算力，可以把常见路线概括成三类：

| 方案 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| Dense / FlashAttention | 短序列、中等序列 | 稳定、实现成熟、效果无损 | 长序列成本仍高 |
| Block-Sparse | 长序列、结构化稀疏 | 规则、GPU 友好、易映射到 tile | 需要设计和维护 block mask |
| Token-wise Sparse | 稀疏模式高度不规则 | 粒度细、表达力强 | GPU 上通常不如 block 规整 |

还可以再加一条很多人容易忽略的对比：

| 问题 | 更适合的方案 |
|---|---|
| 只是想把 512 或 1K 序列跑快 | 先尝试 dense + FlashAttention |
| 需要支撑 4K 到 16K 长上下文 | 优先考虑 block-sparse 或 hybrid |
| 稀疏模式极动态、内容强依赖 | 需要评估动态选块成本是否值得 |
| 模型效果不能接受粗粒度裁剪 | 不要只盯着 block-sparse，可能要回到 dense 或更细粒度方案 |

实际工程里最常见的不是纯 dense，也不是纯 sparse，而是 hybrid：

1. 当 `seq_len < threshold` 时直接走 dense。
2. 当 `seq_len >= threshold` 且稀疏模式稳定时走 block-sparse。
3. 对极少数关键位置保留全局块，其余区域用局部或条带稀疏。
4. 若稀疏模式过于动态，则保留 dense fallback。

一个更接近生产环境的例子是在线推理服务。假设同一套模型同时服务三类请求：

- 聊天：几百 token
- 检索问答：2K 到 8K token
- 长摘要：8K 到 16K token

这时统一强制使用 block-sparse 往往不是最优选择。更合理的路由通常是：

| 条件 | 路由策略 |
|---|---|
| `N < 1024` | 直接 dense |
| `1024 <= N < 4096` | 做 A/B benchmark 决定 |
| `N >= 4096` 且模式稳定 | block-sparse |
| `N >= 4096` 但模式非常动态 | hybrid 优先 |
| 高质量优先于吞吐 | 保守使用 dense 或更宽松的稀疏模式 |

因此，块稀疏注意力更准确的定位不是“attention 的新范式”，而是“长序列下，一种把稀疏模式转换成 GPU 可高效执行 kernel 的方法”。

---

## 参考资料

1. PyTorch `flex_attention` 文档：给出 `flex_attention`、`create_block_mask`、`BlockMask.from_kv_blocks`、`BLOCK_M/BLOCK_N` 等接口说明。文档中 `create_block_mask(..., BLOCK_SIZE=128)` 的默认值与 `kernel_options` 示例里的 `BLOCK_M=64, BLOCK_N=64` 需要区分。  
   https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html  
   访问日期：2026-03-08

2. PyTorch Issue #141129：展示 block sparse mask 在较短序列，尤其低于约 1000 时，可能明显慢于标准 `scaled_dot_product_attention` 的实际案例。这个结论来自社区 issue，不是官方 benchmark 结论，但足以说明短序列边界。  
   https://github.com/pytorch/pytorch/issues/141129  
   访问日期：2026-03-08

3. Emergent Mind, “Block-Sparse Attention in Transformers”：汇总 block-sparse attention 的复杂度写法、block mask 定义以及与 GPU 友好的规则化稀疏之间的关系。属于综述型二手资料，适合做全局背景。  
   https://www.emergentmind.com/topics/block-sparse-attention  
   访问日期：2026-03-08

4. Emergent Mind, “Block-Sparse FlashAttention”：总结 block-sparse 与 IO-aware kernel 结合后的实现方向与经验收益。适合用来理解为什么“只做稀疏”不够，还需要 kernel 层面的融合与流式 softmax。  
   https://www.emergentmind.com/topics/block-sparse-flashattention  
   访问日期：2026-03-08

5. Michael Brenndoerfer, “Sparse Attention Patterns: Local, Strided & Block-Sparse Approaches”：用更直观的可视化方式解释局部、条带和 block-sparse 模式，并强调 block 结构对 GPU 并行和访存连续性的价值。适合作为入门补充。  
   https://mbrenndoerfer.com/writing/sparse-attention-patterns-efficient-transformers  
   访问日期：2026-03-08

6. 如果继续深入实现细节，下一步应直接阅读 PyTorch `flex_attention` 的源码与 Triton kernel 相关实现，而不是只停留在稀疏矩阵的抽象层。原因很简单：真正决定性能的不是公式本身，而是 block 排布、在线 softmax、片上缓存利用和访存路径。

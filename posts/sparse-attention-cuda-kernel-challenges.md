## 核心结论

稀疏注意力的理论收益来自“少算很多无用的 $QK^\top$ 元素”，但 CUDA kernel 的真正瓶颈常常不是算力，而是访存和调度。术语先解释一下：CUDA kernel 就是一段在 GPU 上并行执行的小程序；访存是线程去显存取数据；调度是 GPU 把很多线程块安排到硬件上执行的过程。

第一条结论：把注意力复杂度从 $O(N^2)$ 降到 $O(N\cdot k)$，不等于端到端延迟也按同样比例下降。若稀疏模式导致随机 gather、索引跳转、warp 发散，那么省下的 FLOP 很容易被额外的内存事务吃掉。warp 是 GPU 一次同步执行的 32 个线程，最怕每个线程走不同分支或读不同位置。

第二条结论：工程上真正可用的路径通常不是“任意稀疏”，而是“块稀疏”。也就是把 token 按固定块大小 $b$ 分组，只在块和块之间做稀疏选择。这样每个非零块内部仍然可以走密集矩阵乘法，才能较好复用 Tensor Core。Tensor Core 可以理解为 GPU 专门做小矩阵乘法的高速单元。

一个玩具例子：$N=1024,b=64,k=4$。理论计算量约为
$$
\frac{N}{b}\cdot k\cdot b^2 = N\cdot k\cdot b = 1024\cdot 4\cdot 64 = 262{,}144
$$
而密集注意力需要 $N^2=1{,}048{,}576$ 次配对，纸面上少了 4 倍。但如果每个块都要额外查一次索引表、做一次不连续加载，再单独维护 softmax 状态，实际延迟未必能赢。

一个真实工程例子：FlashInfer 0.2 在 FlashAttention-3 模板上做 block/vector sparsity，公开说明了 `plan/run` 分离、任意 block size、`page_size=1` 的向量稀疏支持。这说明现代实现的关键不只是“有稀疏 mask”，而是先把调度计划整理好，再让运行阶段尽量沿连续维度执行。

---

## 问题定义与边界

本文讨论的是“稀疏注意力的 CUDA kernel 为什么难做快”，不是讨论稀疏模式是否提升模型效果。边界先收紧，否则容易把算法问题和硬件问题混在一起。

我们关心的对象是单层 self-attention 或 prefilling 阶段的 attention kernel。输入通常是 $Q,K,V\in\mathbb{R}^{N\times d}$，序列长度为 $N$，head dimension 为 $d$。把序列按块大小 $b$ 划分后，每个 Query 块只访问 $k$ 个 Key/Value 块。这里的 $k$ 是每个块真正保留的连接数，不是总块数。

衡量指标至少有四个：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| FLOP | 理论计算量 | 反映“少算了多少” |
| HBM 访问量 | 显存读写总量 | 现代 GPU 常常先被内存限制 |
| Achieved bandwidth | 实际有效带宽 | 稀疏访问容易把带宽打散 |
| End-to-end latency | 真实延迟 | 最终用户只关心这个 |

这里有两个容易误解的边界。

第一，$O(N\cdot k)$ 常见于“以 token 为单位计数”的口语说法；严格到 block-sparse kernel，计算量更接近 $O(N\cdot k\cdot b)$。因为每命中一个块，仍要做一个 $b\times b$ 的小密集乘法。若把 $b$ 视为常数，才可进一步写成线性复杂度。

第二，短序列下 dense FlashAttention 往往仍很强。原因不是它理论复杂度更低，而是它的 IO 路径非常顺，tile、softmax、重排都已经高度融合。FlashAttention 论文的多组表格已经说明：很多“理论更省”的近似/稀疏方案，并不会自动换来 wall-clock speedup。对当代工程实现来说，是否在 $N<2048$ 时变快，通常依赖 GPU 架构、前向还是反向、是否 dropout、mask 结构是否规整，不能一概而论。

---

## 核心机制与推导

先看理论。把序列切成 $N/b$ 个块，每个 Query 块访问 $k$ 个 KV 块，每对块内部做 $b^2$ 次点积，则总工作量为
$$
\left(\frac{N}{b}\right)\cdot k\cdot b^2 = N\cdot k\cdot b
$$
若 $k,b$ 不随 $N$ 增长，复杂度对 $N$ 线性。

但 GPU 上还有第二层约束：内存访问必须尽量连续。连续的意思是，相邻线程最好读取相邻地址，这样硬件能把多个请求并成少量大事务。若稀疏索引是乱的，就会出现下面的问题：

| 机制 | 纸面效果 | GPU 上的代价 |
|---|---|---|
| 随机稀疏 | 跳过大量计算 | gather/scatter 多，访存不连续 |
| block-sparse | 少算整块 | 容易映射到 tile，访存更规整 |
| vector-sparse | 剪得更细 | 元数据更多，调度更难 |
| dense tile | 不剪枝 | 最容易跑满 Tensor Core |

为什么 block size 常要对齐 16、32、64？因为 Tensor Core 和 warp 级矩阵乘法的高效路径依赖规则 tile。OpenAI 早期 block-sparse 实现就支持 `8/16/32/64` 等 block size，本质上是在稀疏掩码层面做硬件对齐。块对齐以后，一个 warp 或多个 warp group 可以稳定处理同样形状的子问题，避免“这一半线程有活，那一半线程没活”的分支发散。

再看 softmax。标准注意力需要对每个 Query 行做
$$
\mathrm{softmax}(S_i)_j=\frac{e^{S_{ij}}}{\sum_t e^{S_{it}}}
$$
稀疏后，分母只在保留的块上累计。但如果这些块分散在不同位置，kernel 需要维护跨块的局部最大值和局部和，最后再归并。FlashAttention 的优势在于它把这套在线 softmax 设计成 tile 内流式更新；稀疏版本若 tile 次序混乱，在线归并就更难高效。

玩具例子可以把这个问题看得很直白。假设一个 Query 块要看 4 个 KV 块，其中 4 个块在显存中分别相隔很远。理论上只是“看 4 次”，但硬件上却可能是 4 次独立显存事务，加上 4 次索引解码。若换成 4 个连续 page，虽然数学上没有变化，延迟会明显下降。

真实工程里，FlashInfer 的 `plan/run` 分离就是在解决这个问题。`plan` 阶段先把长度、块索引、页表这类元数据整理成更适合 GPU 执行的布局；`run` 阶段再尽量走模板化、规则化的 kernel。白话说，就是“先把路修直，再开车”，否则车本身再快也跑不起来。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，用来说明 block-sparse 的元数据和复杂度估算。它不是 CUDA kernel，只是把“哪些块会被算”这件事讲清楚。

```python
from math import ceil

def build_block_mask(num_blocks: int, local_k: int):
    """每个 query block 只关注自己之前最近的 local_k 个块（含自身）"""
    mask = [[0] * num_blocks for _ in range(num_blocks)]
    for q in range(num_blocks):
        start = max(0, q - local_k + 1)
        for kv in range(start, q + 1):
            mask[q][kv] = 1
    return mask

def sparse_flops(seq_len: int, block_size: int, k: int) -> int:
    assert seq_len % block_size == 0
    return seq_len * k * block_size

def dense_flops(seq_len: int) -> int:
    return seq_len * seq_len

N = 1024
B = 64
K = 4
num_blocks = N // B

mask = build_block_mask(num_blocks, K)
nonzero_blocks = sum(sum(row) for row in mask)

assert num_blocks == 16
assert nonzero_blocks > 0
assert sparse_flops(N, B, K) == 262144
assert dense_flops(N) == 1048576
assert dense_flops(N) // sparse_flops(N, B, K) == 4

print("num_blocks =", num_blocks)
print("nonzero_blocks =", nonzero_blocks)
print("dense_flops =", dense_flops(N))
print("sparse_flops =", sparse_flops(N, B, K))
```

真正的 CUDA 实现通常分成四步。

第一步，生成 block metadata。也就是每个 Query 块对应哪些 KV 块，通常会保存成 CSR、paged list 或者压缩索引数组。metadata 就是“描述数据怎么排布的数据”。

第二步，重排与分桶。把相似长度、相似块数、相同 head pattern 的请求放在一起，尽量让一个 kernel 内的线程做相似工作。

第三步，执行块内密集计算。对每个命中的 $(q\_block, kv\_block)$，在 shared memory 中加载 tile，调用 Tensor Core 路径做 $QK^\top$、softmax 更新和 $PV$ 乘法。

第四步，写回结果并处理剩余状态。包括跨块归并 softmax 的统计量、边界 block 的 mask、以及 causal 约束。

如果写成结构化伪代码，大致如下：

```python
def sparse_attention_plan(q_lens, kv_lens, block_map):
    # 1. 统计每个请求的非零块数
    # 2. 按块数和长度排序，减少 warp 内负载不均
    # 3. 生成连续的 block schedule 与 gather index
    return schedule, gather_idx, row_ptr

def sparse_attention_run(Q, K, V, schedule, gather_idx, row_ptr):
    # 1. 一个 CTA 处理一个或多个 query block
    # 2. 根据 gather_idx 取出连续/半连续 KV page
    # 3. 对每个命中块做 tile matmul + online softmax
    # 4. 累计输出并写回
    return O
```

这里最重要的不是伪代码本身，而是“plan 和 run 分离”。如果每次 kernel 内部再去临时查复杂索引，性能通常会很差。

---

## 工程权衡与常见坑

第一类坑是“算少了，但搬更多了”。很多新手会把优化目标只写成“减少 attention score 计算量”，然后忽略 K/V gather、索引转置、mask 解码、prefix sum 等辅助步骤。结果是 FLOP 下降了，HBM 访问量反而上升。

第二类坑是 block 太小。小块在算法上更细，可以更精准地剪枝；但在 GPU 上，小块往往意味着更多 metadata、更碎的 memory transaction、更低的 Tensor Core 利用率。FlashMoBA 一类工作之所以强调小块的挑战，就是因为统计上更优的块大小，不一定是硬件上更优的块大小。

一个常见经验是：不要默认更稀疏就更快。比如 $b<128$ 时，如果命中的块分布还不连续，kernel 很容易从“算力受限”变成“访存延迟受限”。这时 profiler 里常见的现象是 SM 利用率不低，但 achieved bandwidth 和 tensor utilization 都上不去。

真实工程例子可以看成三种典型代价：

| 代价来源 | 具体表现 | 规避建议 |
|---|---|---|
| metadata 交换 | host/device 同步、索引表搬运 | 预规划、缓存计划、批量提交 |
| K/V 重排 | gather 过多、L2 命中率差 | 让 page 或 block 尽量连续 |
| 负载不均 | 有的 warp 很忙，有的 warp 空转 | 先按块数分桶，再发射 kernel |

还有两个很容易踩的实现细节。

一是边界块。序列长度不整除 block size 时，最后一个块往往需要额外 mask 分支。如果大量请求都落在边界块，warp 发散会明显增加。

二是 softmax 数值稳定性。跨多个稀疏块做在线 softmax 时，必须正确维护每一行的 running max 和 running sum，否则块与块之间的概率归一化会错。这类 bug 在小样例上可能不明显，但长序列和 bf16 下会快速放大。

---

## 替代方案与适用边界

如果目标是“短序列稳定提速”，很多时候 dense FlashAttention 仍是默认首选。因为它已经把 IO-aware tiling、在线 softmax、kernel 融合做得非常成熟，尤其在 $N$ 不够长时，稀疏调度的固定成本未必值得。

如果目标是“长上下文并且稀疏模式规则”，block-sparse 更合适。规则的 local/window/block mask 最容易对齐硬件，也最容易复用现有 tile kernel。

如果目标是“极细粒度剪枝”，vector-sparse 或 token-sparse 理论上剪得更多，但实现门槛更高，通常需要更复杂的 gather/scatter 和调度重排。它更适合已经有成熟 kernel 基础设施的推理系统，而不是第一次自写 kernel 的团队。

可以用下面这张表做快速判断：

| 方案 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| Dense FlashAttention | 短到中等序列，通用训练/推理 | 稳定、成熟、无需复杂元数据 | 复杂度仍是 $O(N^2)$ |
| Block-sparse | 长序列，规则稀疏模式 | 易于硬件对齐，能接近线性扩展 | 需设计 block layout 与调度 |
| Vector-sparse | 精细剪枝、KV cache 管理 | 剪枝粒度细 | 访存最不规则，工程复杂 |
| 低秩/线性注意力 | 允许改变注意力形式 | 理论和内存更省 | 行为与标准注意力差异更大 |

一句话判断边界：当序列不够长、稀疏模式不够规整、块大小不够友好时，dense kernel 往往更稳；当 $N$ 很长、块稀疏结构稳定、metadata 可提前规划时，block-sparse 才更有机会把理论优势兑现成真实吞吐。

---

## 参考资料

1. FlashAttention 论文：Tri Dao 等，*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*。核心贡献是把 attention 设计成 IO-aware 的 tile 流程，并明确指出很多近似注意力虽然减少 FLOP，但未必带来 wall-clock speedup。  
   https://deepsense.ai/wp-content/uploads/2023/04/2205.14135.pdf

2. FlashInfer 0.2 发布文。说明了基于 FlashAttention-3 模板的 block/vector sparsity、`page_size=1`、以及 `plan/run` 分离等现代工程做法。  
   https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html

3. OpenAI `sparse_attention` 仓库。展示了 block-sparse attention 的早期工程接口，支持 `8/16/32/64` block size，是“块对齐到硬件粒度”这一思路的典型实现。  
   https://github.com/openai/sparse_attention

4. Michael Brenndoerfer 的稀疏注意力文章。优点是把 block-sparse 的复杂度 $O(n\cdot k\cdot b)$ 和硬件友好性讲得比较直观。  
   https://mbrenndoerfer.com/writing/sparse-attention-patterns-efficient-transformers

5. Next Electronics 的稀疏注意力硬件分析。适合理解 coalesced access、warp divergence 与有效带宽下降的关系。  
   https://test.next.gr/ai/large-language-models/sparse-attention-techniques

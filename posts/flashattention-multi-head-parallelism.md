## 核心结论

FlashAttention 的关键优化不是“少算”，而是“边算边丢中间结果”，也就是把注意力矩阵按块处理，避免把完整的 $QK^\top$ 存回显存。这里的“显存”就是 GPU 上存放张量的内存，容量比主存小得多，但带宽更高。

FlashAttention-2 在这个基础上继续优化并行策略。原始 FlashAttention 主要在 `batch × head` 这个外层维度并行，也就是通常把一个 attention head 交给一个线程块。这里的“线程块”可以理解为 GPU 一次派出去执行同一段 kernel 的一个工作小组。这样做在 `B × H` 很大时没有问题，但在长序列训练里常见的是 `B` 很小、`H` 也有限，此时 GPU 的很多 SM 会空闲。这里的“SM”是 Streaming Multiprocessor，可以理解为 GPU 上真正执行计算的核心工作单元。

FlashAttention-2 的核心变化是：把查询序列再按行切块，让一个 head 不再只对应一个线程块，而是对应多个线程块。于是线程块总数从

$$
T_{\text{old}} = B \times H
$$

变成

$$
T_r = B \times H \times \lceil N / B_r \rceil
$$

其中 $N$ 是序列长度，$B_r$ 是查询方向的 tile 高度，也就是每个线程块负责多少行查询。只要 `T_r` 足够大，例如接近或超过 A100 的 108 个 SM，就能显著提升长序列场景下的硬件利用率。

玩具例子可以这样理解：原来是一整块披萨只给 12 个人分，每个人都要自己切完整张；现在改成先把每张披萨切成 64 小块，再同时分给更多人处理，等待时间会明显缩短。这个类比不涉及硬件细节，但能抓住并行度上升的本质。

---

## 问题定义与边界

先明确问题。多头注意力的“多头”是把同一段序列投影到多个子空间，各个 head 独立计算注意力，再拼接结果。FlashAttention 解决的是注意力算子在显存带宽上的瓶颈，不是改变注意力数学定义。

本文讨论的边界很具体：

| 讨论对象 | 范围 |
|---|---|
| 算子类型 | 标准 scaled dot-product attention |
| 关注点 | GPU 上 kernel 的并行映射与 SM 占用 |
| 重点场景 | 长序列、小 batch、head 数有限 |
| 不展开内容 | 稀疏注意力、近似注意力、跨卡通信细节 |

原始 FlashAttention 的一个常见并行映射是：每个 `(batch, head)` 对应一个线程块，线程块内部再循环处理这个 head 的整段序列。问题在于，如果 `B × H` 太小，那么可同时发出去的线程块数量也太小。

例如：

- `B = 1`
- `H = 12`
- 那么原始外层并行度只有 `12`

如果目标 GPU 是 A100，常见说法是至少需要大几十到上百个线程块，才能把 108 个 SM 基本喂满。此时只有 12 个线程块，就会出现“长序列很多，但 GPU 还是闲着”的反直觉现象。原因不是总工作量不够，而是工作被装进了太少的并行单元里。

这可以用一个小白版本的例子说明：长序列训练像请了 108 个员工干活，但你只派出 12 个人，每个人负责啃完整棵树，其余人虽然在场，却拿不到任务。

下面用一个对比表看这个问题更直观。假设 `SM = 108`，`B_r = 128`：

| 场景 | `B` | `H` | `N` | 原始并行 `B×H` | 序列切分并行 `B×H×⌈N/B_r⌉` | 对 SM 占用的影响 |
|---|---:|---:|---:|---:|---:|---|
| 短序列、大 batch | 8 | 16 | 512 | 128 | 512 | 原始方式已较充足 |
| 中序列、中 batch | 2 | 16 | 2048 | 32 | 512 | 序列切分明显改善 |
| 长序列、小 batch | 1 | 12 | 8192 | 12 | 768 | 原始方式严重不足，切分后充足 |
| 极长序列、超小 batch | 1 | 8 | 65536 | 8 | 4096 | 并行足够，但还要关注 launch 开销 |

所以问题不是“FlashAttention 快不快”，而是“它的并行策略在什么输入形状下会失速”。当 `B × H` 足够大时，旧策略就够用；当 `N` 很大但 `B × H` 很小时，必须把序列维度也变成并行来源。

---

## 核心机制与推导

先定义几个符号：

- `B`：batch size，一次并行处理多少条样本
- `H`：head 数，多头注意力里的头数
- `N`：序列长度，也就是 token 数
- `B_r`：row tile 大小，每个线程块负责多少行查询

这里“tile”就是一个小矩形分块，目的是让计算和片上缓存匹配，而不是一次处理整个大矩阵。

### 1. 原始策略为什么受限

在 FlashAttention 的基础实现里，一个线程块往往绑定一个 `(b, h)`。它会读取这个 head 的查询块、键块、值块，逐块扫描 K/V，并维护在线 softmax 需要的中间量。这里的“在线 softmax”是指不必先得到完整分数矩阵，再统一做 softmax，而是边扫描边更新最大值和归一化因子。

这样做有两个优点：

- 显存访问局部性好
- 不需要落地完整注意力矩阵

但并行度基本上锁死在 `B × H`。

### 2. FlashAttention-2 的改法

FlashAttention-2 把查询序列按行切开。也就是说，一个 `(b, h)` 不再只有一个线程块，而是拆成多个 row-block：

- 第 0 块负责 `Q[0:B_r]`
- 第 1 块负责 `Q[B_r:2B_r]`
- 第 2 块负责 `Q[2B_r:3B_r]`
- 以此类推

每个线程块仍然会遍历所有 K/V 块，但只负责自己那一段查询行的输出。因为注意力的每一行输出只依赖该行 query 与全部 key/value，所以不同 query 行块之间天然独立，不需要块间同步。

于是总线程块数变成：

$$
T_r = B \times H \times \lceil N / B_r \rceil
$$

这就是并行度提升的来源。

### 3. 数值推导

以一个常见长上下文场景为例：

- `B = 1`
- `H = 12`
- `N = 8192`
- `B_r = 128`

则

$$
\lceil N / B_r \rceil = \lceil 8192 / 128 \rceil = 64
$$

所以

$$
T_r = 1 \times 12 \times 64 = 768
$$

原始策略只有 `12` 个线程块，现在变成 `768` 个线程块。即使不能全部同时常驻，也已经足够让 108 个 SM 持续拿到工作。

这就是论文和工程实现里常说的“让每个 head 分布到多个线程块上”的真正含义。

### 4. 文字图解

可以把一个 head 的查询序列看成一列很长的行：

| row-block 编号 | 覆盖的 query 行 | 映射到的线程块 |
|---|---|---|
| 0 | `0 ~ B_r-1` | block 0 |
| 1 | `B_r ~ 2B_r-1` | block 1 |
| 2 | `2B_r ~ 3B_r-1` | block 2 |
| ... | ... | ... |
| `⌈N/B_r⌉-1` | 最后一段 | 最后一个 block |

这相当于把 `N=8192` 的长跑切成很多段，每个选手只跑 128 步，多个选手同时跑。原来一整个 head 只上一个选手，现在一个 head 能上 64 个选手，108 条跑道就更容易排满。

### 5. forward/backward 为什么还能继续优化

在前向传播里，按 query 行切块很自然，因为每个输出行独立。反向传播更复杂，因为梯度会涉及对 `Q/K/V` 的回传，某些写回路径可能产生冲突。FlashAttention-2 的工程优化之一，是在 forward 和 backward 里选用更适合的数据切分方式，例如前向按 row slices，反向可按 column slices 或重新组织归约，从而减少原子操作。这里的“原子操作”是指多个线程同时更新同一位置时必须串行保证正确性，会拖慢性能。

核心思想没有变：把数学上可独立的部分，映射成硬件上可独立的线程块。

---

## 代码实现

下面先给一个玩具级 Python 例子，只模拟“线程块数量如何变化”，不模拟 GPU 执行。它能直接运行，并用 `assert` 验证结论。

```python
import math

def flashattention_blocks_old(batch: int, heads: int) -> int:
    return batch * heads

def flashattention_blocks_fa2(batch: int, heads: int, seqlen: int, br: int) -> int:
    assert batch > 0 and heads > 0 and seqlen > 0 and br > 0
    return batch * heads * math.ceil(seqlen / br)

# 玩具例子：长序列 + 小 batch
B, H, N, Br = 1, 12, 8192, 128
old_blocks = flashattention_blocks_old(B, H)
new_blocks = flashattention_blocks_fa2(B, H, N, Br)

assert old_blocks == 12
assert new_blocks == 768
assert new_blocks > old_blocks

# 短序列 + 大 batch：原始并行已经不差
B, H, N, Br = 8, 16, 512, 128
old_blocks = flashattention_blocks_old(B, H)
new_blocks = flashattention_blocks_fa2(B, H, N, Br)

assert old_blocks == 128
assert new_blocks == 512
assert old_blocks >= 108  # 对 A100 这类 108 SM GPU 来说，原始并行已较充足
```

真正的 kernel launch 伪代码可以写成这样：

```python
# pseudocode

def launch_flashattention2(Q, K, V, B, H, N, Br):
    num_row_blocks = ceil_div(N, Br)

    # 把 grid 从 (B, H) 扩展成 (B, H, num_row_blocks)
    gridDim = (B, H, num_row_blocks)
    blockDim = (num_warps * 32, )

    launch kernel_flashattention_fa2<<<gridDim, blockDim>>>(Q, K, V, N, Br)


def kernel_flashattention_fa2(Q, K, V, N, Br):
    b = blockIdx.x
    h = blockIdx.y
    rb = blockIdx.z

    row_start = rb * Br
    row_end = min(row_start + Br, N)

    # 当前 block 只负责一个 head 的一个 row-block
    # 但要遍历所有 K/V tiles
    m = init_minus_inf(row_end - row_start)   # online softmax max
    l = init_zero(row_end - row_start)        # online softmax denom
    O = init_zero(row_end - row_start, d)

    for col_start in range(0, N, Bc):
        K_tile = load_k_tile(b, h, col_start)
        V_tile = load_v_tile(b, h, col_start)
        Q_tile = load_q_rows(b, h, row_start, row_end)

        S = matmul(Q_tile, transpose(K_tile))
        m, l, O = online_softmax_update(S, V_tile, m, l, O)

    store_output(b, h, row_start, row_end, O)
```

这段伪代码表达了两个关键点：

- `gridDim` 不再只跟 `B` 和 `H` 有关，而是扩展到 `B × H × ⌈N / B_r⌉`
- 每个线程块只负责 `row_start ~ row_end` 这一段 query 行

真实工程例子是 GPT 风格长上下文训练。假设你在单张 A100 上训练 8k、16k 或 64k token 的上下文窗口，显存压力会逼着你把 `batch` 压到很小，甚至只有 1 或 2。如果此时仍使用只在 `B × H` 上并行的策略，吞吐会被 SM 空转拖住；改用 FlashAttention-2 的 sequence parallelism 后，一个 head 可以派生出几十到几百个 row-block，GPU 更容易接近 GEMM 类算子的利用率。

---

## 工程权衡与常见坑

FlashAttention-2 不是“切得越碎越好”。它是在并行度、共享内存占用、寄存器压力、kernel launch 开销之间找平衡。

先看最直接的权衡表。假设 `B=1, H=12, N=8192`：

| `B_r` | `⌈N/B_r⌉` | 总线程块数 `T_r` | 并行度变化 | 工程判断 |
|---:|---:|---:|---|---|
| 256 | 32 | 384 | 已显著提升 | 常见可选值 |
| 128 | 64 | 768 | 更容易填满 SM | 长序列常用 |
| 64 | 128 | 1536 | 并行更多 | 可能增加调度与开销 |
| 32 | 256 | 3072 | 极高并行 | 可能切得过细，不一定更快 |

### 常见坑 1：只看到公式，忽略 tile 过小的成本

`T_r = B × H × ⌈N / B_r⌉` 告诉你调小 `B_r` 会增加线程块数，但线程块数不是唯一目标。`B_r` 太小会带来：

- 每块做的有效计算变少
- kernel launch 和调度开销占比上升
- 寄存器与共享内存复用效率下降

所以正确策略不是盲目把 `B_r` 调到最小，而是先确保并行度足够，再看吞吐拐点。

### 常见坑 2：只改 grid，不改 block 内部分工

FlashAttention-2 的提升不只是“序列维度多开块”，还包括更合理的 warp 分工。这里的“warp”是 GPU 中 32 个线程的基本执行单位。新版实现通常让不同 warp 更高效地共享 K/V、分担 Q 的子任务，减少 shared memory 通信。这里的“shared memory”是一个线程块内部共享的片上高速缓存。

如果只把 grid 改大，但 block 内仍沿用旧的工作划分，性能不一定达到预期。

### 常见坑 3：反向传播写回冲突

前向传播中的 row-block 独立性很好理解，但反向传播会涉及多个路径的梯度累加。若设计不当，容易出现：

- 原子加过多
- 写回冲突严重
- 数值归约顺序影响性能

工程上常见规避方式是：前向和反向使用不同切分方向，或者重新组织中间量，使得回传时的归约更局部化。

### 常见坑 4：把“SM 数量”当作绝对门槛

“线程块数大于 108 就能喂满 A100”只是粗略经验，不是严格定理。真实占用率还取决于：

- 每个 block 用多少寄存器
- 每个 block 用多少 shared memory
- 一个 SM 能同时常驻多少个 block
- kernel 是否受带宽而非计算限制

所以更准确的说法是：增大 `T_r` 能显著提高“可调度并行度”，但最终吞吐还受 occupancy 和 memory traffic 约束。这里的“occupancy”就是一个 SM 上同时挂着多少可运行线程块和 warp。

可操作的避坑策略如下：

| 问题 | 现象 | 优先策略 |
|---|---|---|
| `B×H` 太小 | 长序列下 GPU 空闲 | 启用序列切片并行 |
| `T_r` 仍不足 | SM 吃不满 | 适当调小 `B_r` |
| 切分后收益一般 | launch 开销上升 | 回退到更大的 `B_r` |
| block 内效率低 | 共享内存通信重 | 配合新版 warp 分工 |
| backward 慢 | 原子操作多 | 调整 forward/backward 切分方向 |

可以把它理解成多工位流水线：线槽切得太少，会排队；切得稍细一些，大家都能同时工作；但切得过碎，搬运和切换的成本又会反过来吞掉收益。

---

## 替代方案与适用边界

FlashAttention-2 的序列并行不是唯一选择，它只是对“长序列 + 小 batch”特别有效。

| 场景 | 特征 | 推荐策略 |
|---|---|---|
| 短序列、大 batch | `B×H` 本身已大 | 原始 batch×head 并行通常够用 |
| 长序列、小 batch | `N` 大，`B×H` 小 | 优先 FlashAttention-2 序列并行 |
| 超长序列、单卡吃紧 | `N` 极大，显存和 launch 都紧张 | FlashAttention-2 + 模型并行/流水线并行 |
| 需要进一步降复杂度 | 不是只想提速，而是算不动 | 考虑稀疏注意力或近似注意力 |

对新手来说，可以记成一句话：

- 短文章批量发，继续用老方案通常就够。
- 长小说但 batch 很小，就需要 FlashAttention-2 把序列切开并行。
- 如果小说长到单卡已经很难管理，就要再配合多卡并行。

这里还要强调一个边界：FlashAttention-2 主要解决的是算子级并行不足，不解决整个训练系统的所有瓶颈。如果你的训练已经被跨卡通信、embedding、optimizer step 或数据加载拖住，那么单独换 attention kernel 不会线性提升整机吞吐。

另一个边界是：当 `B×H` 已经很大、`N` 又不长时，序列再切分带来的收益可能有限，甚至被额外开销抵消。这也是为什么工程实现通常会根据输入形状动态选择 kernel，而不是永远强制使用最细粒度并行。

---

## 参考资料

- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, arXiv:2307.08691  
  https://arxiv.org/abs/2307.08691
- FlashAttention-2 Explained, DigitalOcean 教程  
  https://www.digitalocean.com/community/tutorials/flashattention2
- The Evolution of FlashAttention, ICLR Blogposts 2026  
  https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/

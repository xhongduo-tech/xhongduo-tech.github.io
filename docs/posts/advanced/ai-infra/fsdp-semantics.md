---
title: FSDP 的语义：AllGather 参数 + ReduceScatter 梯度
date: 2026-08-07
---

# FSDP 的语义：AllGather 参数 + ReduceScatter 梯度

<div class="epigraph">
<p>分片的唯一目的，是让每个进程只在需要时才看到它需要的那部分。</p>
<footer>—— 赵岩（Yanli Zhao），FSDP 论文第一作者</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ FSDP 论文（PyTorch DDP + ZeRO 研究）· 并行策略篇 ｜ 2026-08-07</p>
</div>

## 为什么从 FSDP 开始

上一节讲完 ZeRO-3，你知道了「参数、梯度、优化器状态都分片」的显存收益。但 ZeRO 是 DeepSpeed 的私有实现，PyTorch 生态需要一套**原生、易用、与 `torch.compile` 和自动微分无缝协作**的等价方案——这就是 **FSDP（Fully Sharded Data Parallel）**。

FSDP 的语义可以浓缩成一句口诀：**前向时 AllGather 取回全量参数，后向时 ReduceScatter 汇总并摊薄梯度**。它把 ZeRO-3 的每一个动作映射成 PyTorch 用户可感知的 API 行为。理解 FSDP，你就同时理解了 ZeRO-3 的运行时、以及今天 PyTorch 训练大模型的标准姿势。

## 1 FSDP 的切分单元：把参数 flatten 再分片

FSDP 与 ZeRO 的一个关键差别在于**切分的粒度**。ZeRO-3 按「参数张量」逐个分片；FSDP 则先把一组参数**压平（flatten）成一个一维张量**，再按数据并行规模切分。

- **Flatten**：一个 FSDP 单元内的所有参数被拼接成一个大的一维 buffer。好处是内存连续、通信一次搞定、分片索引简单。
- **Shard**：把压平后的一维 buffer 切成 $N$ 段，每张卡持有 $1/N$。
- **状态一并分片**：参数的梯度、优化器状态都建在这个分片 buffer 上，天然也分片。<span class="marginnote">flatten 的副作用是「一个参数可能被切断，分布在两张卡上」。对用户透明，但调试时要知道：FSDP 单元里的参数在内存里不是独立对象，而是共享一个被切分的大 buffer。</span>

**FSDP 的分片粒度是可配置的**：`sharding_strategy` 可以选 `FULL_SHARD`（参数、梯度、优化器状态全分片，即 ZeRO-3）、`SHARD_GRAD_OP`（保留全量参数、只切梯度与优化器状态，即 ZeRO-2）、`NO_SHARD`（纯 DDP 语义）。默认 `FULL_SHARD`。

## 2 前向：AllGather 取回全量参数

FSDP 单元的前向分三小步：

1. **AllGather**：把分散在 $N$ 张卡上的参数分片收集齐，拼成完整参数，放到一个**临时 buffer**。
2. **本地前向**：用完整参数计算本卡这份 batch 的前向。
3. **丢弃全量参数**：前向算完，立刻释放临时 buffer 中的非本地分片，只保留自己那 $1/N$。

为什么不一直保留全量参数？因为那样就退化成 DDP 了。FSDP 的精髓是**用完即弃**——参数只在算子执行的那一瞬间完整存在于显存，执行完立刻回到分片态。<span class="marginnote">这一步对应 ZeRO-3 的「每个算子前 AllGather 参数」。多个 FSDP 单元可以按需依次 AllGather，不用同时持有所有层的全量参数——显存峰值因此被压到「一层」的量级，而不是「整个模型」。</span>

## 3 后向：ReduceScatter 汇总并摊薄梯度

反向传播到 FSDP 单元时：

1. 本卡计算本地 batch 产生的**局部梯度**。
2. 由于参数是分片的，梯度也必须分片才能与参数对齐。FSDP 用 **ReduceScatter**：跨卡求和（reduce）同位置梯度，同时把结果**散射（scatter）**到持有对应参数片的卡上。
3. 更新：每张卡只更新自己那 $1/N$ 的优化器状态与参数，更新前需要把 FP16 分片转成 FP32 主权重。

一个易被忽略的细节：**后向的 AllGather**。因为后向要用到前向的激活与参数，FSDP 在计算某一层梯度前，会再次 AllGather 该层参数（与 DDP 的「后向只需本地参数」不同——DDP 参数一直完整，FSDP 参数是分片的）。所以完整地说，FSDP 是**「前向 AllGather + 后向 AllGather + 后向 ReduceScatter」**三次集合通信。<span class="marginnote">这就是 FSDP 比 DDP 通信量大的根本原因：DDP 每个梯度只需一次 AllReduce；FSDP 除了 ReduceScatter（等价于 AllReduce 的通信量），还多了前后向两次 AllGather 参数。</span>

## 4 通信量定量对比

设模型参数量 $\Psi$（字节），DP 规模 $N$。一次迭代的跨卡通信量：

- **DDP**：梯度 AllReduce，通信量 $2 \times 2\Psi$（AllReduce 的通信量是数据量的 2 倍，含发送与接收）。
- **FSDP**：前向 AllGather 参数 $+$ 后向 AllGather 参数 $+$ ReduceScatter 梯度。AllGather 与 ReduceScatter 的通信量约为「数据总量」的量级（每个 rank 收发共 $O(\Psi)$）。

于是 FSDP 相对 DDP 的通信开销大约变为 $1.5$–$3$ 倍（取决于实现是否复用前向的 AllGather 结果、是否开启通信计算重叠）。**FSDP 用「更多的通信」换「更省的显存」**，当显存是瓶颈时这笔交易很划算；当网络带宽紧张时，这个代价必须靠重叠与低精度来掩盖。

## 5 公式解析：为什么 AllGather+ReduceScatter 是分片的标准搭配

考虑一个 FSDP 单元的参数 $W$ 被切成 $N$ 片，$W_i$ 在第 $i$ 张卡上。

**前向时**，每张卡都需要完整 $W = [W_0, W_1, \ldots, W_{N-1}]$：

$$W^{(k)} = \text{AllGather}(W_0^{(k)}, W_1^{(k)}, \ldots, W_{N-1}^{(k)}) \tag{1}$$

- **$W^{(k)}$（第 $k$ 张卡上的全量参数）**：AllGather 的语义是「每张卡送出自己那份，同时收齐所有人的份」，所以 $N$ 张卡上最终各有一份完整 $W$。
- **下标 $0..N-1$（分片索引）**：每张卡只「拥有」自己的 $W_i$，其余都是借来的、用完即还。

**后向时**，设本卡 batch 算出的局部梯度为 $G^{(k)}$，它是对完整参数 $W$ 的梯度，也要被分摊：

$$G_i = \text{ReduceScatter}(G^{(0)}, G^{(1)}, \ldots, G^{(N-1)}) \quad \text{使得 } G = \frac{1}{N}\sum_k G^{(k)} \tag{2}$$

- **ReduceScatter 的 Reduce**：跨卡把同位置梯度求和（sum），等价于 AllReduce 的 reduce 半步。
- **Scatter**：把求和结果中属于第 $i$ 片的那部分，留在第 $i$ 张卡上。这样第 $i$ 张卡得到 $G_i$，恰好是它持有的参数片 $W_i$ 的梯度。

**关键洞察**：AllGather 的逆运算正是 ReduceScatter。参数怎么「展开」出去的，梯度就怎么「收拢」回来——**AllGather 与 ReduceScatter 是一对互逆的通信原语**，这正是「AllGather 参数 + ReduceScatter 梯度」能配对成一套完整训练语义的数学原因。<span class="marginnote">如果你学过第二篇的集合通信，会发现这背后是分片矩阵的「收集-汇总」对偶。ZeRO/FSDP 的设计本质上就是把「一次 AllReduce」拆成「AllGather（取参数）+ ReduceScatter（汇梯度）」，从而让内存也像通信一样分片。</span>

## 6 辨析｜易错点：FSDP、ZeRO-3、DDP 三者边界

| 特性 | DDP | ZeRO-3（DeepSpeed） | FSDP |
| --- | --- | --- | --- |
| 参数存储 | 每卡全量 | 分片 | 分片 |
| 梯度同步 | AllReduce | ReduceScatter | ReduceScatter |
| 参数取回 | 不需要 | AllGather | AllGather |
| 生态 | PyTorch 内置 | DeepSpeed 框架 | PyTorch 原生 |
| 灵活度 | 低 | 高（Offload 等） | 中（与 torch.compile 结合好） |

**辨析｜易错点：**
- **FSDP 不是新的并行策略**：它是 ZeRO-3 的实现，本质仍是「参数分片的数据并行」。
- **FSDP 不等于 DDP**：DDP 从不 AllGather 参数；FSDP 每个算子前后都要 AllGather。别把两者混成同一种通信开销。
- **FSDP 的 AllGather 不是免费的**：通信量比 DDP 大，显存省下的空间需要靠重叠（通信与计算并行）来补吞吐。
- **FSDP 分片沿 DP 维**：若同时用 TP/PP，FSDP 的分片组大小要等于 DP 组，而不是总卡数（与 ZeRO 同理）。

## 7 小结

- **FSDP = ZeRO-3 的 PyTorch 原生实现**：flatten 参数、分片、状态一并分片。
- **前向语义**：AllGather 取回全量参数 → 本地计算 → 用完即丢。
- **后向语义**：再次 AllGather 参数 → 算梯度 → ReduceScatter 汇总并摊薄梯度 → 本地更新。
- **通信对偶**：AllGather 与 ReduceScatter 互逆，构成完整的分片训练循环。
- **权衡**：显存省（每卡 $\approx 16\Psi/N$）但通信增（约 DDP 的 1.5–3 倍），需通信计算重叠兜底。

## 8 进阶与延伸

**动手看 FSDP 的通信量**：用 PyTorch Profiler（第十篇）抓一个 FSDP 训练 step 的 trace，数一数 AllGather 与 ReduceScatter 各出现几次——你会发现每个 FSDP 单元前向一次、后向一次 AllGather，正是本篇公式描述的三次集合通信。

**几个值得进一步挖的方向**：

- **FSDP 与 DDP 的切换点**：模型多大时 FSDP 优于 DDP？答案在「省下的显存能换多大的 batch」与「增加的通信代价」的权衡——这是成本工程（第十二篇）的典型算账。
- **sharding_strategy 的三个档**：`FULL_SHARD`、`SHARD_GRAD_OP`、`NO_SHARD` 对应 ZeRO-3/2/1——三档的显存与通信曲线各是什么形状？按模型规模选哪档。
- **FSDP 与 PP 的组合**：FSDP 分片沿 DP 维、PP 切层——两者叠时，通信组怎么建？这预告了下一节的 3D 混合并行。

**自测题**：FSDP 的「前向 AllGather 参数」与 ZeRO-3 的「每算子 AllGather」是同一件事吗？如果你能指出「都沿 DP 维分片、都用 AllGather 取回」的共同本质，就抓住了两个框架的同一性。

## 9 动手实践清单

- 用 `fully_shard` 包装一个模型，打印参数的 `placements` 确认分片声明。
- 用 profiler 数一数一个 step 里 AllGather 与 ReduceScatter 的次数。
- 对比 FSDP 与 DDP 的每步耗时与显存，画「模型规模 vs 谁更优」的切换点。
- 调整 `sharding_strategy` 三个档位，观察显存与通信的权衡。
- 验证「FSDP 沿 DP 维分片」——叠加 TP 时检查通信组是否正确。
- 开启 `torch.compile`，对比编译前后的每步耗时。
- 用 16Ψ/z 公式反推「FSDP 后每卡显存」，与实测核对。

在下一节，我们把 DP、TP、PP（以及 SP/EP/CP）拼成一张完整的拼图，看 **3D 混合并行**如何在一个真实集群上分配通信组。

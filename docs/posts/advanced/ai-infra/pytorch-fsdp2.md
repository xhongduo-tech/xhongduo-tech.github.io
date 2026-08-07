---
title: PyTorch FSDP2（fully_shard）的设计：per-parameter sharding
date: 2026-08-07
---

# PyTorch FSDP2（fully_shard）的设计：per-parameter sharding

<div class="epigraph">
<p>分片的粒度，决定了你能省多少显存、又浪费多少带宽。</p>
<footer>—— 安德鲁 · 古（Andrew Gu），FSDP 原作者与 FSDP2 设计者</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PyTorch FSDP2 设计文档（2024）· 训练框架篇 ｜ 2026-08-07</p>
</div>

## 为什么从 FSDP2 开始

FSDP1（`FullyShardedDataParallel`）已在 PyTorch 服役多年，能跑 7B/13B 模型。但它的内部设计有一个被人诟病的点：**把所有参数 flatten 成一个巨大 buffer 再分片**。这个设计让通信粗、内存峰值高、还和 `torch.compile` 处处作对。

PyTorch 在 2.3 之后引入了新一代 **FSDP2（`torch.distributed.fsdp.fully_shard`）**，核心转变就一句话：**从「模块级 flatten 分片」改为「参数级（per-parameter）分片」**。别小看这个粒度变化——它带来更细的通信、更低的峰值内存、以及与编译器更好的协作。理解 FSDP2，你就站在了 PyTorch 分布式训练的最前沿。

## 1 FSDP1 的包袱：flatten 的代价

回忆《FSDP 语义》那篇：FSDP1 把一个模块的所有参数 **flatten 成一维 buffer**，再按 DP 规模切开。flatten 的好处是「一次 AllGather 拿全模块参数」，但代价明显：

- **通信粒度粗**：无论是否用得上，FSDP1 都要对**整个模块**做一次 AllGather。模块内一个参数动了，全部参数跟着全量取回。
- **峰值内存高**：unsharded 的全量 buffer 在执行期间必须完整存在，峰值 = 整个模块的参数量。
- **与 `torch.compile` 不兼容**：flatten 的 buffer 破坏了编译器对参数视图的静态分析；FSDP1 与 compile 结合常报错或退化为 eager。
- **可组合性差**：每个 FSDP 模块的 unsharded buffer 是独立的，多层嵌套时内存无法共享，优化器、梯度也绑死在模块边界。<span class="marginnote">FSDP1 的「模块级」设计带来一个著名痛点：如果某个模块的参数特别大（比如超大 embedding），一次 AllGather 就要暂存整块参数，峰值内存直接爆掉。FSDP2 的 per-parameter 设计从根上消除了这个问题。</span>

## 2 FSDP2 的核心：per-parameter sharding

FSDP2 不再 flatten。**每个参数都被单独表示为一个 `DTensor`，其分片布局用 `Shard` 或 `Replicate` 显式声明**：

$$W \mapsto \text{DTensor}(\text{local\_shard}_i, \text{placements}=[\text{Shard}(0)])$$

- **`DTensor`**：PyTorch 2.x 引入的分布式张量抽象，描述「这个全局张量在 $N$ 卡上怎么分布」。
- **`Shard(0)`**：沿第 0 维切成 $N$ 片，每卡持有 $1/N$——per-parameter 的分片声明。
- **`Replicate`**：全量复制到每张卡——per-parameter 的「不分片」声明。

于是「哪个参数分片、哪个参数复制」变成**张量级属性**，而不是模块级开关。模型可以精细到「embedding 复制、FFN 分片」这种混合布局。<span class="marginnote">DTensor 是 FSDP2 的底层语言：分片、AllGather、ReduceScatter 都变成 DTensor 上的算子。它让「分片」从「框架实现细节」上升为「张量的可查询属性」——调试时能直接打印某个参数的 placements 看它分没分片。</span>

## 3 关键机制一：按需 AllGather，粒度到参数

FSDP2 的 AllGather 不再按模块触发，而是**按需、按参数触发**。某个参数要用于计算时，才把它的分片 AllGather 成完整参数；算完即释放。效果：

- 每个参数独立决定「何时取回、取回多大」，通信粒度 = 参数，而非模块。
- 一个超大 embedding 的 AllGather 只占它自己那么大，不再连累同模块其他参数。
- 多个参数的 AllGather 可以交错进行，通信与计算重叠的窗口更细。

**FSDP2 把「分片单元」与「通信单元」统一到参数级**：分片是按参数的，取回也是按参数的——这消除了 FSDP1 里「分片按模块、通信被迫跟随模块」的错配。

## 4 关键机制二：与 torch.compile 原生协作

FSDP2 在设计中把「编译器友好」作为一等目标。它不依赖 flatten buffer，而是用 DTensor 的标准算子表达分片逻辑，这恰好是 `torch.compile` 能静态分析的形态：

- **编译后的 AllGather**：DTensor 的 AllGather 被编译进计算图，与前后算子融合，减少 kernel 启动。
- **更少的显式同步**：编译器能识别「这个 AllGather 的结果只在某个算子用」，自动安排最晚取回、最早释放。
- **fallback 干净**：编译不了的部分退回 eager，不会整体崩坏。<span class="marginnote">体验上，FSDP2 + `torch.compile` 的组合比 FSDP1 + compile 稳定得多——社区报告里 FSDP1 与 compile 的兼容问题占了相当比例，而 FSDP2 从设计上就规避了。这也是 FSDP2 被定位为「未来默认」的原因。</span>

## 5 公式解析：通信粒度的变化

设一个模块有 $k$ 个参数，参数量分别为 $w_1, w_2, \ldots, w_k$，模块总参数 $W = \sum_j w_j$，DP 规模 $N$。

FSDP1 对模块做一次 AllGather，临时显存与通信量：

$$\text{Mem}_{\text{FSDP1}} = W \text{（全量 unsharded buffer）}, \qquad \text{Comm} \propto W$$

FSDP2 对每个参数独立 AllGather，且**只取当前计算需要的参数**。假设一次前向实际用到参数子集 $S$：

$$\text{Mem}_{\text{FSDP2}} = \sum_{j \in S} w_j \le W, \qquad \text{Comm} \propto \sum_{j \in S} w_j$$

- **$w_j$（单参数大小）**：per-parameter 的取回单位。
- **$S$（本次实际使用集）**：FSDP2 允许「只取用到的参数」，FSDP1 做不到（模块级必然全取）。
- **$W$（模块总量）**：FSDP1 的固定成本。

关键结论：**FSDP2 的峰值显存与通信量都从「模块总量 $W$」降为「实际使用子集」**。对 Transformer 这种「一层只用一个 attention + 一个 MLP」的模型，这通常意味着单次取回的量就是「一层的参数」，而不是「整个模型或整个大模块」。<span class="marginnote">直觉类比：FSDP1 像「借一整本书，看完一页就还」，FSDP2 像「只借当前要读的那一页」。当模型很大、或参数访问很稀疏（MoE 的路由、条件分支）时，per-parameter 的优势被急剧放大。</span>

## 6 辨析｜易错点：FSDP2 的常见误区

**辨析｜易错点：**
- **FSDP2 ≠ 新并行策略**：它仍是「参数分片的数据并行」，只是实现粒度和张量表达不同。
- **`fully_shard` 是函数不是类**：用法是 `fully_shard(module, ...)` 逐层包裹，替代 FSDP1 的 `FullyShardedDataParallel(module)` 类实例化。
- **per-parameter 不意味着没有 AllGather**：参数仍要 AllGather 才能计算，只是粒度变细、按需取回。
- **DTensor 分片是「声明式」的**：`Shard(0)` 声明了布局，但实际通信由算子执行；改布局要改 placements，不是改数据本身。
- **FSDP2 仍在演进**：API 以 beta 形式存在，生产使用前确认 PyTorch 版本与已知 bug。

## 7 小结

- **FSDP1 的包袱**：模块级 flatten 分片，通信粗、峰值内存高、与 torch.compile 不兼容。
- **FSDP2 的核心**：per-parameter sharding，用 DTensor 表达每个参数的分片布局。
- **按需 AllGather**：通信粒度降到参数级，取用实际子集，峰值与通信量从 $W$ 降为 $\sum_{j \in S} w_j$。
- **编译器友好**：DTensor 算子可静态分析，FSDP2 与 torch.compile 原生协作。
- **定位**：PyTorch 分布式训练的未来默认形态，仍属「分片式数据并行」。

## 8 进阶与延伸

**动手看 DTensor 的分片布局**：用 `fully_shard` 包装一个模型后，打印某个参数的 `distributed_tensor.placements`——你会看到 `Shard(0)` 的声明。试着把 embedding 改成 `Replicate`，体会「per-parameter 混合布局」的灵活性。

**几个值得进一步挖的方向**：

- **FSDP2 与 `torch.compile` 的配合**：DTensor 算子被编译进图后，AllGather 与前后算子融合——用 `torch.compile` 前后各剖析一次，量化编译带来的提升。
- **per-parameter 与优化器分片**：每个参数独立分片后，优化器状态也跟着分——这与 FSDP1 的「大 buffer 分片」在内存碎片上的差异是什么？
- **FSDP2 的梯度累积语义**：per-parameter 分片让「按需 AllGather」更精细——梯度累积时，哪些参数可以不取回？省下的通信量怎么算。

**自测题**：FSDP2 把「分片单元」从模块降到参数，为什么通信反而更省？如果你的回答是「只取用到的子集」，你抓住了一个关键——但别忘了「通信粒度变细」的另一面（更碎的 AllGather 可能有更高的延迟开销）。

## 9 动手实践清单

- 用 `fully_shard` 包装模型，打印参数的 `distributed_tensor.placements`。
- 把 embedding 设成 `Replicate`，观察混合布局的显存差异。
- 对比 FSDP1 与 FSDP2 在同一模型上的每步耗时。
- 开 `torch.compile`，量化编译对 FSDP2 的加速。
- 观察「per-parameter 分片」下超大 embedding 的 AllGather 粒度。
- 用 profiler 检查 FSDP2 的「按需取回」是否减少通信量。
- 验证「FSDP2 仍属分片式数据并行」——叠加 TP 时的行为。
- 观察 per-parameter 分片下的显存碎片情况。

在下一节，我们把几大框架摆上台面，做一场 **Megatron-LM vs DeepSpeed vs FSDP2** 的选型对决。

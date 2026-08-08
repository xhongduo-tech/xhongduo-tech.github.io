---
title: 显存碎片与 PyTorch 的 caching allocator 机制
date: 2026-08-07
---

# 显存碎片与 PyTorch 的 caching allocator 机制

<div class="epigraph">
<p>系统崩溃往往不是没内存，而是内存被拆得没法用。</p>
<footer>—— 尤纳斯 · 格里索拉（Jonas Geiping，ML 系统实践者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PyTorch 显存管理与 CUDA 分配器文档 · 显存优化篇 ｜ 2026-08-07</p>
</div>

## 为什么从显存碎片开始

很多训练工程师见过这种诡异场景：nvidia-smi 显示显存还有 20GB 空闲，torch.cuda.mem_get_info() 也说有 20GB 可用，可一分配 10GB 的张量就 OOM。**显存总量明明够，为什么还爆？** 答案往往不是「不够」，而是**碎片**——空闲的显存被拆成无数不相邻的小块，最大的连续块不够用。

理解碎片，就要先理解 PyTorch 的 **caching allocator（缓存分配器）**：它如何分配、为何产生碎片、以及用什么旋钮缓解。这是「显存够却 OOM」类问题排查的必备知识。

## 1 caching allocator 的工作方式

cudaMalloc/cudaFree 每次都要和驱动、和其他进程协调，开销大且慢。PyTorch 的做法是**缓存**：

- 第一次需要 8MB 时，真的向驱动要 8MB；用完不还，留在「缓存池」里。
- 下次再要 8MB（或相近大小），**直接复用缓存里的块**，不触发驱动调用。
- 只有在缓存里找不到合适块时，才向驱动申请新块。

这套机制把「分配」从「系统调用」变成了「池内查找」，速度提升几个数量级。代价是：**分配过的块被保留，内存不再完全交还操作系统**——nvidia-smi 里看着占满，其实是缓存。<span class="marginnote">这也是「PyTorch 进程退出前显存不降」的原因：caching allocator 缓存着那些块，即使张量已释放。看到显存占用高不要慌，先看 memory_allocated()（真占用）与 memory_reserved()（含缓存）的差。</span>

## 2 碎片是怎么产生的

碎片来自「分配—释放不同大小块」的次序错配。想象一个 1GB 的连续显存池：

1. 分配 A（400MB）、B（400MB）、C（200MB）——池满。
2. 释放 B——出现一段 400MB 空闲。
3. 想要一个 500MB 的张量——**失败**！虽然总空闲 400+200=600MB，但最大连续块只有 400MB。<span class="marginnote">这就是碎片的定义：空闲总量足够，但没有「够大且连续」的块。PyTorch 会尝试把相邻空闲块合并，但合并要求地址相邻；如果 A 和 C 仍被占用，B 的空闲块就孤零零地卡在中间，合并不了。</span>

训练里碎片尤其严重，因为：

前向/后向的激活张量**大小不断变化**（不同 batch、不同算子）。
梯度累积、重计算造成**反复分配释放**。
多流（stream）的并发分配让块地址交错。

**碎片率 = 「最大可用连续块」与「总空闲」的背离程度**——碎片越重，可用显存越多却越难用。

## 3 两个池与大小分桶

PyTorch 的 caching allocator 还做了两件事对抗碎片：

**大块池与小块池分离**：>1MB 的分配走「大块池」（stream 1），小分配走「小块池」。避免小块混在大块池里打散大块。
**向上取整对齐**：请求大小对齐到 512B 的整数倍，让「相同大小」的请求更容易复用同一块。

分池 + 对齐让「同尺寸张量」的复用好很多，但仍挡不住「大小各异的激活」带来的碎片。<span class="marginnote">一个经典观察：同样一个模型，batch size 设成 8 和 9 的碎片率可能天差地别——因为 9 的激活大小与默认对齐模式更「不合群」，产生更多不可合并的空洞。调 batch 大小有时能意外解决 OOM，就是这个道理。</span>

## 4 缓解手段：四件工具

PyTorch 提供四个主要旋钮对抗碎片：

**expandable_segments**：用 **VMM（虚拟内存管理）段**技术，让缓存块在虚拟地址上可扩展，物理页按需提交。碎片率显著下降，代价是少量显存开销与兼容性限制。<span class="marginnote">expandable_segments 是 2023 年后最推荐的碎片解决方案：它把「一大块虚拟地址空间」预留出来，块与块可以地址无关地伸缩，消除了「地址相邻才能合并」的约束。缺点是某些老 CUDA 版本或特殊 API 不兼容。</span>
- **max_split_size_mb**：设定「多大的块才允许被拆开使用」。防止一个大块被拆成碎片后没法再做大分配。
- **garbage_collection_threshold**：设定「缓存池利用率超过阈值就触发 cudaFree 归还」——显存紧张时让缓存块提前还给驱动。
- **roundup_pow2**：让分配向上取整到 2 的幂，提高同尺寸复用率。

## 5 公式解析：碎片率的定量估算

设缓存池中有 $n$ 个空闲块，大小为 $s_1, s_2, \ldots, s_n$（按地址顺序）。总空闲 $S_{\text{free}} = \sum s_i$，最大连续块 $S_{\max} = \max_i s_i$。定义碎片率：

$$\text{Fragmentation} = 1 - \frac{S_{\max}}{S_{\text{free}}}$$

- **$S_{\max}/S_{\text{free}}$（可利用率）**：空闲显存里「最大一块」的占比。等于 1 表示无碎片（所有空闲连成一片）。
- **$1 - S_{\max}/S_{\text{free}}$（碎片率）**：0 表示无碎片；接近 1 表示显存被切成无数小片。
- **OOM 判据**：请求大小 $R > S_{\max}$ 时必 OOM——**即使 $R \le S_{\text{free}}$**。

代入例子：池中空闲块 400MB + 200MB，$S_{\text{free}}=600$，$S_{\max}=400$，碎片率 $1 - 2/3 = 33\%$。想要 500MB 就 OOM。<span class="marginnote">这个公式也给出了排查路径：OOM 时先看 torch.cuda.memory_summary()——它列出最大可用块与碎片统计。如果「空闲大但最大块小」，就是碎片问题，上 expandable_segments；如果「真的一点空闲都没有」，才是容量问题。</span>

## 6 辨析｜易错点：显存排查的常见误区

**辨析｜易错点：**
- **nvidia-smi 占用高 ≠ 泄露**：caching allocator 缓存着块，真释放了多少要对比 memory_allocated() 与 memory_reserved()。
- **OOM ≠ 总量不够**：可能是碎片。先跑 torch.cuda.memory_summary() 判断，再决定加显存还是开 expandable_segments。
- **empty_cache 不是万能药**：它只把缓存归还驱动，不解决「已分配但不用」的张量，也压不住碎片（块还是碎的）。
- **expandable_segments 不是零成本**：它可能增加少量虚拟地址开销，且与部分自定义 CUDA 扩展不兼容。
- **别把所有显存问题都推给 allocator**：先确认是不是激活真的大、是不是忘了释放，再谈碎片。

## 7 小结

- **caching allocator**：缓存分配过的块，分配从系统调用变为池内查找；代价是显存不完全归还。
- **碎片成因**：不同大小张量的反复分配释放，最大连续块变小，总量够也 OOM。
- **分池与对齐**：大块池/小块池分离、512B 对齐，提高复用但挡不住激活碎片。
- **缓解四件套**：expandable_segments、max_split_size_mb、garbage_collection_threshold、roundup_pow2。
- **碎片率公式**：$1 - S_{\max}/S_{\text{free}}$，OOM 判据是「请求 > 最大连续块」。

## 8 进阶与延伸

**动手诊断一次 OOM**：故意用一个「显存勉强够」的配置训练，OOM 后立刻跑 torch.cuda.memory_summary()——看「最大可用块」与「碎片率」，判断是容量问题还是碎片问题，再决定上 expandable_segments 还是减 batch。

**几个值得进一步挖的方向**：

- **expandable_segments 的代价**：它用虚拟内存段消除碎片，但可能增加「分配粒度」的显存开销——什么场景下它反而更费显存？
- **max_split_size_mb 的语义**：设了它，小于该值的分配不再拆分大块——这防止「大块被拆碎」，但也可能让小块分配不到缓存，怎么平衡？
- **多流（stream）下的碎片**：计算流与通信流并发分配时，块地址交错更严重——碎片率怎么随 stream 数变化？

**自测题**：empty_cache 为什么「压不住碎片」？如果你能说清「empty_cache 只归还缓存、不合并空闲块」，就理解了碎片问题的本质在「分配模式」而非「有没有归还」。

## 9 动手实践清单

- OOM 后立刻跑 torch.cuda.memory_summary()，区分「容量问题 vs 碎片问题」。
- 对比 memory_allocated() 与 memory_reserved()，判断缓存占用。
- 开 expandable_segments，对比碎片率与 OOM 频率。
- 调 max_split_size_mb，观察大块是否被拆碎。
- 用 torch.cuda.memory_snapshot() 画一张「显存块分布」图。
- 试 garbage_collection_threshold，观察显存紧张时的行为。
- 验证「empty_cache 不解决碎片」——释放后看最大连续块是否变大。
- 对比「分池 + 对齐」对复用率的影响。
- 验证「碎片率公式」与 memory_summary 的一致性。

在下一节，我们把显存工程的每一笔都串起来——**实战演练：估算训练一个模型到底要多少显存**。

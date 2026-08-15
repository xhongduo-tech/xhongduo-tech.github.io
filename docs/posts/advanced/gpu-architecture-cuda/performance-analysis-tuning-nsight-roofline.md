---
title: 性能分析与调优（Nsight、Roofline、内存/计算受限判定）
date: 2026-08-07
---

# 性能分析与调优（Nsight、Roofline、内存/计算受限判定）

<div class="epigraph">
<p>程序优化的第一条规则：不要做。第二条规则（仅供专家）：先别做。</p>
<footer>—— 迈克尔 · 杰克逊（Michael A. Jackson），1975</footer>
</div>

<div class="article-byline">
<p>第四级 · GPU 架构与 CUDA 并行编程 ｜ Kirk & Hwu, Programming Massively Parallel Processors, 4e, Ch6；Roofline：Williams, Waterman & Patterson, 2009 ｜ 2026-08-07</p>
</div>

## 为什么从性能分析开始

前八篇给了你一整箱武器：SIMT、内存层次、共享内存、Tensor Core、流。但**武器多不代表打得好**——本专题反复强调「先测量、再优化」，现在到了兑现这句话的时候。

性能调优不是「凭感觉加关键字」，而是三件事：**测量（哪里慢）→ 建模（瓶颈是什么类型）→ 对症（怎么改）。** 这一篇把前面所有量化工具（占用率、有效带宽、算术强度）收拢成一张统一的性能地图——Roofline，并给出标准工作流。<span class="marginnote">Roofline 模型出自 Williams、Waterman 与 Patterson 2009 年的论文，是「内存受限 vs 计算受限」的最简洁数学表达；PMPP 第 6 章把这些概念落成 GPU 上的具体指标，NVIDIA 的 Nsight Compute 则把它做成可点的 UI。</span>

## 1 先测量：Nsight 全家桶

优化第一步永远是「让数据说话」。NVIDIA 提供的两大分析器分工明确：

- **Nsight Systems**：系统级时间线。看全局——kernel、拷贝、CPU 启动开销如何排队，**哪里在等、哪里是空洞**。适合回答「瓶颈在 GPU 内部还是 CPU/传输」。
- **Nsight Compute**：单 kernel 级剖析。一个 kernel 的完整画像——占用率、内存吞吐、指令吞吐、**瓶颈分析（Memory Workload Analysis / Speed of Light）**。适合回答「这个 kernel 为什么这么慢」。

工作流上的区别很重要：**先用 Nsight Systems 找到「时间去哪了」，再用 Nsight Compute 钻进「最慢的那个 kernel」。** 很多人一上来就抠 kernel 细节，却忘了全局时间线可能显示「CPU 拷贝占了 80%」——那再优化 kernel 也白搭。<span class="marginnote">一个常被忽略的点：Nsight 报的「瓶颈」是聚合统计，不代表每个 warp 都那样。性能调优要按「最慢的路径」下结论，而不是平均——与《SIMT 执行模型》篇讲的 warp 分歧是同一套逻辑。</span>

## 2 Roofline：一张图看懂一切

**Roofline 模型**用一条折线刻画一台机器的性能上界：横轴是**算术强度（arithmetic intensity，每搬 1 字节数据做多少次浮点运算）**，纵轴是**可达到的浮点吞吐（FLOPs/s）**。任何 kernel 的性能都不可能超出这条「屋顶」。

- **屋顶的平顶**：计算峰值 $\pi_{peak}$——纯计算时能达到的 FLOPs/s 上限。
- **屋顶的斜坡**：带宽 $\beta \times I$——带宽受限时，性能 = 带宽 × 算术强度。
- **折点（ridge point）**：两段交界，对应的算术强度叫临界强度 $I_{ridge}$。

kernel 落在屋顶上（性能被硬件封顶）才算「跑满了」；落在屋顶下方（性能低于上界），说明还有资源没利用——**调优就是不断把 kernel 往屋顶上顶。**

## 3 公式解析：Roofline 的数学

Roofline 的核心就是一个 min：

$$
\text{Achievable} = \min\left( \pi_{peak},\; \beta \times I \right), \qquad
I = \frac{\text{FLOPs}}{\text{Bytes}}
$$

逐项拆解：

- **$\pi_{peak}$**：计算峰值 FLOPs/s，硬件常量。如 A100 的 FP32 约 19.5 TFLOPS。
- **$\beta$**：显存带宽 Bytes/s，硬件常量。如 A100 约 1.6–2 TB/s（峰值），H100 约 3.35 TB/s。
- **$I$（算术强度）**：程序自己的属性——总浮点运算量除以总访存字节数。$I$ 大 = 计算密集，$I$ 小 = 访存密集。
- **临界强度 $I_{ridge} = \pi_{peak} / \beta$**：A100 上约 $19.5 / 2 \approx 10$ FLOPs/字节（不同精度/配置略有出入）。

代入判定：若 $I < I_{ridge}$，$\beta \times I < \pi_{peak}$，**内存受限（memory-bound）**——性能卡在带宽，提高算力没用；若 $I > I_{ridge}$，**计算受限（compute-bound）**——性能卡在算力，带宽再宽也没用。

**这就是前面所有「计算受限 vs 内存受限」判定的数学依据。** 拿 GEMM 举例：$I \approx 683$（见《Tensor Core》篇），远超临界强度——所以它是计算受限，优化方向是喂满 Tensor Core 而不是「再省点带宽」。

## 4 受限判定：对症下药

判定出受限类型后，优化方向完全不同，这是调优最重要的分岔口：

| 受限类型 | 症状（Nsight） | 优化方向 |
| --- | --- | --- |
| 内存受限 | Memory Throughput ≈ 100%，Compute < 100% | 减字节：合并访问、数据复用进共享内存、降低精度、减少跨端拷贝 |
| 计算受限 | Compute Throughput ≈ 100%，Memory < 100% | 减运算：向量化（float4）、用 Tensor Core、减少冗余计算、更优算法 |
| 延迟受限 | 两者都不满，但 warp 不足 / 占用率低 | 提并行：调 block 大小、压寄存器、`__launch_bounds__` |

**关键认知：内存受限时你花大力气优化「指令数」是白费的——瓶颈在带宽不在算力。** 相反，计算受限时省带宽也无用。判断错了方向，优化就是缘木求鱼。<span class="marginnote">Nsight Compute 的 "Speed of Light" 面板直接给你两个百分比（Compute 与 Memory），谁接近 100% 谁是瓶颈——这是把 Roofline 的公式落成可操作判断的最快路径。</span>

## 5 调优工作流：五个步骤

把前面的内容串成一个可执行的工作流：

1. **建立基线**：先跑对，记录当前耗时与指标。
2. **看全局（Nsight Systems）**：时间花在 kernel、拷贝还是 CPU 启动？先解决「没在算」的时间。
3. **钻最慢 kernel（Nsight Compute）**：看 Speed of Light、内存/计算吞吐，判定受限类型。
4. **对症修改**：按上表方向改，**一次只改一个变量**，记录前后对比。
5. **回归验证**：确认没有引入正确性问题；把改动沉淀为「为什么有效」的笔记。

**纪律最重要：一次只动一处。** 同时改占用率、共享内存、精度，即使性能变好你也不知道是哪个起的作用——后面想复用经验就无从谈起。<span class="marginnote">这也是为什么本专题前几篇刻意把「占用率」「共享内存」「流」分开讲：每个都是独立可调的旋钮，只有独立旋钮才能做干净的 A/B。</span>

特别提醒：**先把「CPU 与 GPU 之间拷贝」这条路径优化好，再看 kernel**——很多「GPU 慢」的真相是「数据在 PCIe 上往返太多」，Nsight Systems 时间线里的红色空洞一眼就能看出。这正呼应《流、事件与并发执行》篇：先让数据流动起来，再谈算得快。

## 6 Roofline 实战：两个典型 kernel 的对标

把理论落到两个具体例子，看 Roofline 怎么指导决策。以 A100（FP32 峰值约 19.5 TFLOPS、带宽约 2 TB/s）为基准，临界强度 $I_{ridge} \approx 10$ FLOPs/字节。

**例子一：向量求和（memory-bound）**。`sum += a[i]` 每个元素读 4 字节、做 1 次加法，算术强度 $I = 1/4 = 0.25$ FLOPs/字节，远低于 10——**内存受限**。Roofline 上它落在斜坡段：能达到约 $0.25 \times 2\text{ TB/s} = 500$ GFLOPs/s，而 19.5 TFLOPS 的算力峰值完全用不上。优化方向：合并访问把带宽用满、降低精度减字节——而不是「怎么加快加法」。

**例子二：GEMM（compute-bound）**。《Tensor Core》一篇算过 $I \approx 683$ FLOPs/字节，远高于 10——**计算受限**，落在屋顶平顶段。优化方向：喂满 Tensor Core、分块复用——省带宽没有意义。

把两个 kernel 画在 Roofline 上，决策一目了然：

| Kernel | $I$（FLOPs/字节） | 受限类型 | 该优化什么 |
| --- | --- | --- | --- |
| 向量求和 | 0.25 | 内存受限 | 带宽、合并、降精度 |
| GEMM | ≈683 | 计算受限 | 算力、Tensor Core、分块 |

**Roofline 的实践价值就在这里：它逼你先算 $I$，再决定方向**——避免了「在内存受限的 kernel 上优化指令数」这种最浪费的功夫；这也正是本专题从《内存层次》到《Tensor Core》每一篇都在铺垫的「先分类型、再动手」。拿到 Nsight 的「内存吞吐 / 计算吞吐」两个百分比，再套上 $I$ 与 $I_{ridge}$，五分钟就能定位一个 kernel 的优化主攻方向；本专题前面各篇讲的每一件优化手段（合并访问、共享内存、张量核心、流），都只是在这个框架下「选对了方向之后的执行」。

### 从 Roofline 到 Nsight：两个百分比

理论算完，回到工具：Nsight Compute 的 **Speed of Light** 面板给两个数字——Compute Throughput 与 Memory Throughput。**谁接近 100%，谁就是瓶颈；Roofline 已经告诉了你该信谁**：$I < I_{ridge}$ 的 kernel 看 Memory，$I > I_{ridge}$ 的看 Compute。两者都不满 80% 时，多半是延迟受限（占用率不足），回到《占用率与延迟隐藏》篇调并行度。

这一段的要点是「理论与工具对账」：**Roofline 负责预测瓶颈类型，Nsight 负责确认并量化**——两者一致，优化方向就锁定了；两者打架，先怀疑自己是不是算错了 $I$ 或量错了带宽。把这条习惯内化，你就真正拥有了本专题最值钱的技能：**不被直觉带偏、永远让数据说话。** 至此，本专题从 SIMT 到多 GPU 的所有优化手段，都收进了这一套「算强度 → 判受限 → 对症 → 实测」的闭环。

（一句话复盘本专题的优化观：先让数据流动起来（流/拷贝重叠），再让内存喂得满（合并访问/共享内存），再让算力用得上（张量核心/占用率），最后用 Roofline 判定每一步到底该不该做。）

## 7 小结

- 先测量再优化：**Nsight Systems** 看全局时间线，**Nsight Compute** 钻单 kernel 瓶颈。
- Roofline：$\text{Achievable} = \min(\pi_{peak}, \beta \times I)$，临界强度 $I_{ridge} = \pi_{peak} / \beta$。
- $I < I_{ridge}$ → **内存受限**（省字节），$I > I_{ridge}$ → **计算受限**（减运算），两者都低 → **延迟受限**（提并行）。
- 对症下药按受限类型，**一次只改一个变量**，用 Nsight 做干净 A/B。
- 判断错受限类型 = 缘木求鱼：内存受限时优化指令数、计算受限时省带宽，都是白费。

在下一节，我们把单 GPU 的能力扩展到「一张卡不够」的规模：**多 GPU 编程（NVLink/NVSwitch、NCCL、与集群博文衔接）**。

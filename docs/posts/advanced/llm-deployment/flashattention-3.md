---
title: FlashAttention-3：Hopper 架构的异步与 FP8
date: 2026-08-07
---

# FlashAttention-3：Hopper 架构的异步与 FP8

<div class="epigraph">
<p>当硬件开始理解你的工作负载，软件就能借力打力。</p>
<footer>—— FlashAttention-3 团队（Shah et al., 2024）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ FlashAttention-3 论文（Shah et al., 2024） ｜ 2026-08-07</p>
</div>

## 为什么从 FlashAttention-3 开始

FA1 解决了 IO、FA2 解决了并行，但两者都是为 Ampere（A100）设计的思想。Hopper（H100）带来了一批新硬件特性：**TMA（张量内存加速器）、异步执行、warp 专用化、FP8 张量核心**——如果内核还是「同步 + FP16」的老写法，这些硬件能力全浪费。FlashAttention-3 是**为 Hopper 重写的注意力内核**：它把「等待」变成「重叠」，把「FP16」换成「FP8」，在 H100 上再次把注意力提速约 1.5–2 倍。<span class="marginnote">FA3 的核心思想是<strong>异步化（asynchrony）</strong>：让数据的搬运（TMA）与计算（Tensor Core）并行发生，而不是「搬完再算、算完再搬」的串行节奏。这需要硬件支持，Hopper 恰好给了。</span>

本篇讲 FA3 依赖的 Hopper 特性、warp 专用化与异步流水线、以及 FP8 在注意力中的应用与精度问题。

## 1 Hopper 的新武器：TMA、Warpgroup 与异步

Hopper 架构相比 Ampere 的三个关键差异，是 FA3 设计的输入：

- **TMA（Tensor Memory Accelerator）**：一个专门负责「把数据从 HBM 搬到 SMEM、再搬回」的硬件单元。以前搬运要占用线程，TMA 让搬运**脱离线程、后台进行**——线程只管算。<span class="marginnote">TMA 的意义：<strong>搬运不占计算资源</strong>。内核可以把「下一块数据的搬运」与「当前块的计算」重叠起来，流水线几乎不空转。</span>
- **Warpgroup**：Hopper 把 4 个 warp 组成一个 warpgroup，可以做跨 warp 的协作与异步调度。FA3 用 warpgroup 分工：一组负责 GEMM，一组负责 softmax，两组异步流水。
- **异步指令（`cp.async` / 异步 GEMM）**：数据加载与 GEMM 发射可以「不等待」，靠事件/barrier 控制节奏。

这些特性让「搬运与计算重叠」从「程序员手动插桩」变成「硬件原生能力」——FA3 只是把流水线编排得更好。

## 2 Warp 专用化：生产者-消费者流水线

FA3 把注意力内核组织成**生产者-消费者流水线**：

- **生产者 warpgroup**：负责加载下一块 $K$、$V$（TMA 搬运）并启动 GEMM（$S = QK^T$）；
- **消费者 warpgroup**：负责对已算出的 $S$ 做在线 softmax、归一化，并计算 $PV$ 更新输出。

两组 warpgroup 之间用**异步 barrier** 同步，形成「搬运第 $j+1$ 块的同时，计算第 $j$ 块」的重叠。<span class="marginnote">这是经典的<strong>软件流水线（software pipelining）</strong>思想：把一个循环的「取数」与「计算」错开一拍。FA1/FA2 的循环是「取一块→算一块→取下一块」，FA3 是「边取边算」。</span>

相比 FA2 的「线程分工轮换」，FA3 的 warpgroup 专用化更进一步：**专职生产者不再切换角色**，它在整个循环里只做搬运与 GEMM 发射，专注度更高、流水线更稳定。

## 3 FP8 注意力：更快的 GEMM，更小心地处理

FA3 支持 FP8 输入（E4M3）：$Q$、$K$、$V$ 用 FP8 存储，$QK^T$ 与 $PV$ 的 GEMM 用 FP8 Tensor Core，累加保持 FP32。<span class="marginnote">FP8 让注意力 GEMM 的访存减半、Tensor Core 吞吐翻倍——但注意力的 FP8 比权重 FP8 <strong>更微妙</strong>：$QK^T$ 的数值范围动态变化大，量化误差会直接进入 softmax。</span>

FA3 处理 FP8 精度问题的方法：

- **per-block / per-warpgroup 缩放**：给每个 query/key 块单独算 scale，比整层一个 scale 更细；
- **FP32 累加**：GEMM 内部用 FP32 累加，避免中间精度损失累积；
- **「先算后缩放」策略**：$QK^T$ 在高精度下算完，再统一缩放到 FP8 做 softmax，把量化误差限制在可控范围。

**辨析｜易错点：FP8 注意力 ≠ 权重量化 FP8。** 权重量化是静态的（权重固定、一次校准）；注意力的 $Q$、$K$ 每步都在变，FP8 缩放必须**在线动态计算**。这比离线校准难得多——FA3 的贡献之一正是把「在线 per-block 缩放」做进了内核。

## 4 公式解析：异步重叠的加速

设每块注意力的计算时间 $T_{\text{comp}}$ 与搬运时间 $T_{\text{load}}$。同步实现（FA2 风格）每个循环步耗时 $T_{\text{sync}} = T_{\text{load}} + T_{\text{comp}}$（串行）。异步流水（FA3）让搬运与计算重叠：

- **第一步，写同步耗时**：循环 $M$ 步，总耗时 $M(T_{\text{load}} + T_{\text{comp}})$——搬运时间被完整计入。
- **第二步，写异步耗时**：流水线启动后，第 $k$ 步的搬运与第 $k-1$ 步的计算重叠。总耗时约 $T_{\text{start}} + \max(T_{\text{load}}, T_{\text{comp}}) \cdot M$——**搬运时间被「藏」在计算下面**。
- **第三步，比加速**：

$$\text{Speedup} \approx \frac{T_{\text{load}} + T_{\text{comp}}}{\max(T_{\text{load}}, T_{\text{comp}})}$$

当 $T_{\text{load}} \approx T_{\text{comp}}$ 时加速约 2 倍；若搬运远快于计算，加速趋近 1（瓶颈已不是搬运）。**异步化让瓶颈从「串行和」变成「两者较大值」**——这正是 FA3 相对 FA2 提速约 1.5–2 倍的机制。

叠加 FP8（GEMM 翻倍）后，FA3 在 H100 上相对 FA2 的整体收益可达 2–3 倍。

## 5 小结

- **FA3 为 Hopper 重写**：TMA、warpgroup、异步指令把「搬运与计算重叠」变成硬件原生能力。
- **生产者-消费者流水线**：一组 warpgroup 专职搬运与 GEMM、另一组专职 softmax，异步 barrier 同步。
- **FP8 支持**：$Q$/$K$/$V$ 用 FP8、累加 FP32，per-block 在线缩放控制量化误差。
- **FP8 注意力 ≠ 权重量化**：注意力动态在线缩放，比静态权重量化更复杂。
- **异步重叠的加速**：耗时从「搬运+计算」串行变为「两者较大值」，配合 FP8 整体提速约 2–3 倍。

在下一节，我们把注意力优化带到 decode 阶段的极限——**FlashDecoding 与长序列 Decode 加速**。

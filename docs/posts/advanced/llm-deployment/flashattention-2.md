---
title: FlashAttention-2：更好的并行与工作分配
date: 2026-08-07
---

# FlashAttention-2：更好的并行与工作分配

<div class="epigraph">
<p>把每个线程都用起来，把每块芯片都点亮。</p>
<footer>—— FlashAttention-2 优化哲学（Dao et al., 2023）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ FlashAttention-2 论文（Dao et al., 2023） ｜ 2026-08-07</p>
</div>

## 为什么从 FlashAttention-2 开始

FlashAttention-1 解决了「IO 太多」的问题，但在 GPU 的并行利用上还有明显短板：它在某些阶段只用一个 thread block 算一行、矩阵乘法用的不是最先进的算子（未充分利用 Tensor Core）、还把「非矩阵」的 softmax/归一化步骤放在了线程数很少的循环里。FlashAttention-2 不动算法的数学框架（仍是分块 + 在线 softmax），而是**重排了并行与工作分配**，让每个 SM、每个线程都满负荷。效果：相比 FA1 再提速约 2 倍，同时把「只能前向」扩展到「前向 + 反向」。<span class="marginnote">FlashAttention 三连的第二篇。记住主线：<strong>FA1 解决「搬得太多」，FA2 解决「跑得太空」</strong>——同一个内核，从「能跑」到「跑满」。</span>

本篇讲 FA2 的三大并行优化：序列维并行、更细的线程块划分、以及把非矩阵运算「矩阵化」。

## 1 并行度的重新设计：每个输出行一个 block

FA1 的一个设计弱点是：对每个 query 块，输出行的计算被串行化——同一个 thread block 要循环遍历所有 $K/V$ 块。当 batch × head 数不足以填满 GPU 的 SM 数量时，大量 SM 闲置。

FA2 的改进：**保持「一个 thread block 负责一个 query 块（一行）」的划分，但让这个 block 内部并行化程度更高**——把序列维、头维、batch 维都映射到并行网格上。<span class="marginnote">划分原则的对比：<strong>FA1 一个 block 处理一行里的一块，FA2 一个 block 处理一整行的计算</strong>（循环仍在，但每步都充分利用 block 内所有线程做 GEMM）。这让「并行度 = batch × heads」而不是「batch × heads × 序列块」。</span>

这个改变的本质是把「序列维并行」换成「更彻底的 head/batch 并行」：当 batch × heads 大于 SM 数时，每个 SM 都拿到活；而序列维的循环在 block 内部高效串行。**对在线推理（batch 小、head 多）尤其有利**。

## 2 线程块内的职责划分

FA2 对 thread block 内部的 128/256 个线程做了精细分工：

**一半线程算 $S = QK^T$ 的 GEMM**；
**另一半线程在算完的 $S$ 上做 softmax 与归一化**；
两部分轮流切换，避免「算 GEMM 时 softmax 线程闲置、softmax 时 GEMM 线程闲置」。

这个「两半轮换」设计让 block 内的每个线程在整个循环里都有活干，而不是「GEMM 阶段全员做 GEMM、softmax 阶段只有少数线程忙」。

另一个关键优化：**避免 FA1 中共享内存的读写序列化**。FA1 在 softmax 后要把结果写回共享内存再做 $PV$；FA2 通过更优的寄存器与共享内存分配，把「写回再读」的乒乓降到最低。<span class="marginnote">共享内存（SMEM）是 block 内的高速缓存，但读写也有带宽限制。<strong>FA2 减少了 SMEM 的中间写回次数</strong>，让数据更多地留在寄存器里——这是它在 kernel 层面的微观收益。</span>

## 3 把注意力「矩阵化」：启用 Tensor Core

FA1 的 $S = QK^T$ 虽然用了 GEMM，但 softmax、缩放、$P$ 的归一化等步骤是逐元素的手写循环——**没有用上 Tensor Core（张量核心）**，而 Tensor Core 是 A100/H100 算力的大头。

FA2 把所有能变成矩阵乘的步骤都变成 GEMM：

$QK^T$ 是 GEMM；
softmax 的「除以 $l$、乘 $e^{m}$」被吸收进下一次 GEMM 的缩放；
$PV$ 是 GEMM（在 FA1 中其实是「逐元素乘加」，FA2 把它规整成标准 GEMM）。

**尽量让计算落在 Tensor Core 上**，是 FA2 把速度推高的核心手段。矩阵乘法用 CUTLASS 的高性能 tile 配置，而非手写低效循环。<span class="marginnote"><strong>Tensor Core 擅长「大矩阵乘」，不擅长「小循环逐元素」</strong>。FA2 的策略是「把所有步骤都塑造成 Tensor Core 喜欢的形状」——这是所有高性能内核的共同哲学。</span>

**辨析｜易错点：FA2 不只是「把 FA1 改快点」。** 它同时支持了**前向 + 反向**（训练需要梯度），且反向传播也做分块与在线 softmax——这让 FlashAttention 从「推理专用」变成「训练/推理通用」。对部署而言，这意味着训练出的模型可以直接在 FA2 内核上推理，行为一致。

## 4 公式解析：并行度与加速比

设 GPU 有 $S$ 个 SM，注意力 batch × heads = $B_h$，序列长度 $N$。FA1 与 FA2 的并行度差异：

- **第一步，写 FA1 的并行占用**：FA1 可并行启动的 block 数 $\approx B_h \times (N / d_m)$（每 query 块一个 block）。当 $B_h$ 小（推理常见）时，序列块数成为并行度来源，但每个 block 内的 SM 利用率受「循环串行」限制。
- **第二步，写 FA2 的并行占用**：FA2 的 block 数 $\approx B_h$，但每个 block 内线程**全程同时做 GEMM 与 softmax**，SM 利用率更高。加速比的粗略分解：

$$\text{Speedup} = \frac{T_{\text{FA1}}}{T_{\text{FA2}}} \approx \frac{T_{\text{SMEM-overhead}} + T_{\text{low-occupancy}} + T_{\text{non-TC}}}{T_{\text{minimal}}}$$

- **第三步，看实际收益**：论文报告 FA2 相对 FA1 在 A100 上加速约 1.7–2.0 倍（前向），训练端到端约 1.3 倍。**收益来源是「并行度 + Tensor Core + 减 SMEM 乒乓」三者的叠加**，不是单一优化。

## 5 数值算例：FA1 vs FA2 的并行账

把「并行度从哪来」算成具体数字。设 A100（108 SM）、推理 batch=4、head=32（如 7B 模型），即 $B_h = 128$。

**FA1 的并行占用**：每个「query 块」一个 block。query 块沿序列维切（如 64 token 一块），block 数 $= B_h \times (N/64)$。对短序列（$N=2048$），$N/64 = 32$，block 数 $= 128 \times 32 = 4096$——**block 数看似充足，但每个 block 内「算 GEMM 时 softmax 线程闲置」，SM 利用率被 block 内部拖低**。

**FA2 的并行占用**：一个 block 负责一整行 query（head × batch），block 数 $= B_h = 128$（与序列块数无关）。每个 block 内 256 线程全程同时做 GEMM 与 softmax——**SM 利用率显著更高**。

| 对比项 | FA1 | FA2 |
| --- | --- | --- |
| block 数 | batch×heads×序列块 | batch×heads |
| block 内线程利用 | GEMM/softmax 轮流闲置 | 全程同时忙碌 |
| 计算载体 | 部分手写循环 | 尽量 Tensor Core |
| 短序列并行 | 依赖序列块 | 依赖 batch×heads |

**读这张表**：FA2 的并行哲学是「**让每个 block 干满一整行，而不是干半行等下一块**」——对推理（batch 小、head 多、序列长）尤其有利。这就是为什么 FA2 在「长序列 decode/prefill」场景的收益比 FA1 更明显。<span class="marginnote">并行哲学的直觉：<strong>FA1 像「每个工人只搬一块砖、搬完等下一块」，FA2 像「每个工人搬一整车砖、搬完再装车」</strong>——工人数相同，但每个工人的活儿更满。</span>

## 6 FA2 与其他优化的搭配

FA2 作为「注意力内核」，与上层优化天然互补：

| 搭配 | 效果 |
| --- | --- |
| FA2 + PagedAttention | 注意力快 + KV 内存高效，vLLM 标配 |
| FA2 + FlashDecoding | FA2 管 prefill、FlashDecoding 管 decode |
| FA2 + 量化（INT8/FP8） | 低精度注意力 + 高速内核，吞吐再翻 |
| FA2 + KV 量化 | 访存减半 + 内核高效 |

**读这张表**：FA2 是「内核层」优化，不排斥任何上层优化——**它是引擎的「默认地基」**，现代推理引擎（vLLM、TensorRT-LLM、SGLang）都默认启用 FA2 系内核。

**辨析｜易错点：FA2 不是「全能的注意力加速」。** 它对「标准注意力」最优；对稀疏注意力、MoE 路由后的变体，可能需要专用内核。**别把「FA2 加速了标准注意力」当成「任何注意力都快」**——用对内核，才能吃到 FA2 的红利。

**FA2 收尾一句**：它没有发明新数学，而是把「并行、分工、矩阵化」做到了极致——**高性能内核的通用哲学：让硬件每个部件都忙、都干它最擅长的活**。

把 FlashAttention 三代的演进摊开，看「主线」：

| 版本 | 解决的核心问题 | 关键手段 | 相对上一代 |
| --- | --- | --- | --- |
| FA1 | IO 太多（访存瓶颈） | 分块 + 在线 softmax，重算不存 | 2–4 倍（大序列） |
| FA2 | 并行不足、没吃满 | 更细并行 + Tensor Core + 分工 | 约 2 倍 |
| FA3 | Hopper 异步 + FP8 | 异步流水、FP8 注意力 | 约 2 倍（Hopper） |

**读这张表**：三代的主线是「**先解决访存，再解决并行，再压榨硬件特性**」——每一代都在「让硬件更忙、更高效」。理解这条主线，就理解了「内核优化往哪走」。

**FA2 在部署中的意义**：它是现代引擎的「默认内核」，推理与训练共用——**意味着你在 vLLM/TensorRT-LLM 里部署的模型，用的就是 FA2 系内核**。理解 FA2 的并行设计，也就理解了「为什么小 batch 推理时引擎仍能高效」。

## 7 小结

- **FA2 重排并行与工作分配**：一个 block 负责一整行 query，循环在 block 内高效执行，并行度 = batch × heads。
- **block 内线程分工轮换**：一半算 GEMM、一半算 softmax，轮流切换，消灭线程闲置。
- **把非矩阵运算矩阵化**：缩放、归一化吸收进 GEMM，尽量落在 Tensor Core 上。
- **支持前向 + 反向**：训练与推理共用内核，行为一致。
- **相对 FA1 加速约 2 倍**：并行度、Tensor Core、SMEM 优化三合一。
- **是引擎默认地基**：与 PagedAttention、FlashDecoding、量化均互补；推理场景 batch×heads 高时收益最大。

在下一节，我们登上 Hopper 架构的舞台——**FlashAttention-3：Hopper 架构的异步与 FP8**。

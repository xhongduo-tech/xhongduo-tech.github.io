---
title: Tensor Memory Accelerator（TMA）
date: 2026-08-07
---

# Tensor Memory Accelerator（TMA）

<div class="epigraph">
<p>最好的搬运工，是那个不需要你操心它怎么搬的人。</p>
<footer>—— 异步数据搬运的设计哲学</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ Hopper 论文 §4 ｜ 2026-08-07</p>
</div>

## 为什么从 TMA 讲起

从 Roofline 模型看，GPU 性能的天花板由「算力」和「带宽」共同决定；而实际性能能否触顶，取决于**数据能不能被及时搬进 SM**。传统 GPU 的数据搬运方式有一个尴尬：**搬运数据这件事本身消耗计算资源**——每个线程都要自己算地址、自己发访存指令、自己处理边界。Hopper 用一个专门的硬件单元来解决这个问题，这就是 **Tensor Memory Accelerator（TMA）**。它是 Hopper「异步化」设计理念的核心组件，也是理解 FlashAttention、cuBLAS 这些高性能库为什么能在 H100 上跑满的关键。<span class="marginnote">TMA 的定位可以类比 CPU 里的 DMA（直接内存访问）控制器：以前数据搬运要 CPU 一条条指令地搬，现在有专门的硬件负责，CPU 可以去算别的。TMA 就是 GPU 版 DMA，但做得远比 DMA 智能——它认识「张量」的结构。</span>

## 1 传统数据搬运的三大痛点

在理解 TMA 之前，先看清传统方式（`cp.async` 与普通 `ld` 指令）的问题。以「把 HBM 里的一块矩阵搬到共享内存」为例：

**痛点一：地址计算占用计算资源。** 一块 $128 \times 128$
---
title: GPU 体系结构：从图形流水线到通用计算（GPGPU）
date: 2026-08-07
---

# GPU 体系结构：从图形流水线到通用计算（GPGPU）

<div class="epigraph">
<p>CPU 是一辆赛车，GPU 是一支车队——单个比不过，但上千辆一起运货，总吞吐碾压。</p>
<footer>—— GPU 设计哲学的通俗比喻</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机体系结构 ｜ Hennessy & Patterson《Computer Architecture: A Quantitative Approach》附录 G ｜ 2026-08-07</p>
</div>

## 为什么 GPU 长这样

GPU 生来是为了**图形**：画面里数百万个像素/顶点，每个都做几乎一样的运算——这是 [[data-level-parallelism-overview]] 的终极形态。2006 年 NVIDIA 推出 **CUDA**，让程序员能把这套「海量并行」用于通用计算（**GPGPU**），GPU 从此成为 AI 与科学计算的主力。<span class="marginnote">GPU 的核心哲学是「<strong>延迟隐藏</strong>」：单个运算很慢（不做乱序、不深流水），但并行线程足够多，一个在等、另一个在算——<strong>吞吐碾压延迟</strong>。这与 CPU 的思路截然相反。</span>

## 1 GPU 的硬件组织：SM 与 SP

**核心概念**：现代 GPU 由**数十到上百个流式多处理器（SM, Streaming Multiprocessor）**组成，每个 SM 内含**数百个精简 ALU（CUDA core / SP）**、共享存储与调度器。

```
GPU：
  SM 0 ─ 数百个 ALU + 共享内存 + 调度器
  SM 1 ─ 数百个 ALU + 共享内存 + 调度器
  ...
  SM 127 ─ ...
```

- 单个 ALU：非常简陋，无乱序、无分支预测、缓存极小——**把面积省给 ALU 数量**。
- 总 ALU 数以万计：一块旗舰 GPU 有 16,000+ 个浮点单元。

## 2 CPU vs GPU：两种处理器的坐标

| 维度 | CPU | GPU |
| --- | --- | --- |
| 目标 | 单线程低延迟 | **海量吞吐** |
| 核心 | 少数（4–64）高性能核 | 极多（数千）简朴核 |
| 隐藏延迟 | 乱序、深流水 | **线程切换** |
| Cache | 大、多级、复杂 | 小、侧重吞吐与共享 |
| 控制流 | 复杂分支预测 | 掩码执行（[[gpu-branch-divergence]]） |
| 功耗预算 | 给「快」 | 给「多」 |

**核心概念**：CPU 用「一个核跑得快」换低延迟，GPU 用「几千个核一起跑」换高吞吐。**各有各的「以大概率事件为快」**：单线程程序 CPU 赢，数据并行程序 GPU 碾压。

## 3 CUDA 编程模型：GPU 的并行层级

CUDA 把并行组织成三层：

```
网格（grid）→ 块（block）→ 线程（thread）
   ↓            ↓             ↓
整个任务      一个 SM        一个 SP
```

- **线程（thread）**：最小执行单元，一个数据元素一个线程。
- **块（block）**：一组线程，驻留在**同一个 SM**，可协作（共享内存、同步）。
- **网格（grid）**：所有块，对应整个任务；不同块在不同 SM 上并行调度。

写代码：

```cuda
__global__ void add(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];          // 每个线程处理一个元素
}
```

## 4 GPU 的高带宽：内存不是障碍是设计对象

GPU 的另一个杀手锏是**内存带宽**：显存位宽极宽（如 384/512 bit）、GDDR/HBM 高频——一块 GPU 的带宽可达 CPU 的 5–10 倍。**数据并行负载大多是带宽敏感**，GPU 用「宽总线 + 海量并发访存」把带宽榨干。<span class="marginnote">这正是 [[simd-programming-autovectorization]] 里「带宽是 SIMD 加速天花板」的应对：GPU 把「天花板」整体抬高——<strong>它天生为带宽敏感负载设计</strong>。AI 训练、渲染、科学模拟全吃这一口。</span>

## 5 核心对比表：GPU 的「为什么」

> 本节为纯概念主题，以核心对比表替代公式解析。

| 设计选择 | CPU | GPU | GPU 的逻辑 |
| --- | --- | --- | --- |
| 单核性能 | 极致 | 普通 | 单核不重要 |
| 核数量 | 少 | 极多 | 并行度才是生产力 |
| 隐藏延迟 | 乱序/预测 | 线程切换 | 并行覆盖等待 |
| Cache | 大而复杂 | 小而为吞吐 | 少缓存、多 ALU |
| 频率 | 高 | 中 | 功耗给数量 |

## 6 一个实例：用数字看 GPU

以 NVIDIA A100（Ampere）为例，把本节的概念落成具体数字：

- **108 个 SM**，每 SM 128 个 FP32 CUDA core → 全芯片约 **6912 个核心**。
- **432 个 Tensor Core**（每 SM 4 个，见 [[tensor-core-matrix-acceleration]]）。
- **80 GB HBM2e 显存**，带宽约 **2 TB/s**——同时代 CPU 内存带宽的 10 倍以上。
- 功耗约 400 W：接近一个 CPU 插槽的 5 倍，但吞吐是数十倍。

这些数字印证了本节的核心结论：**GPU 把功耗预算全部砸向「核数量 × 带宽」，而不是单核速度**。对比 [[intel-core-arm-cortex-microrch]] 里 CPU 的 8–16 个核，数量级的差别就是「延迟机 vs 吞吐机」的差别。

## 7 小结

- GPU 从**图形流水线**走来，CUDA 把它变成**通用计算平台（GPGPU）**。
- 硬件组织：**SM（数十个）→ ALU（每 SM 数百个）**，总数以万计。
- **吞吐优先**哲学：单线程普通、并行度碾压；用线程切换隐藏延迟。
- CUDA 并行层级：**grid → block → thread**，对应任务 → SM → SP。
- GPU 内存带宽是 CPU 的 5–10 倍，为带宽敏感负载而生。
- CPU vs GPU = **低延迟 vs 高吞吐**，各有主战场。

在下一节，我们看 GPU 最独特的执行模型——**SIMT：Warp 与线程束调度**。

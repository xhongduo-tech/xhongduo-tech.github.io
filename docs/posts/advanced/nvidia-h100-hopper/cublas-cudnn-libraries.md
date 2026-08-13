---
title: cuBLAS / cuDNN 性能库
date: 2026-08-07
---

# cuBLAS / cuDNN 性能库

<div class="epigraph">
<p>最好的优化，是你不需要自己写的优化。</p>
<footer>—— 性能库存在的理由</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ NVIDIA cuBLAS / cuDNN 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从性能库讲起

前面十几节教的是「自己写 kernel 时怎么把 H100 用满」。但绝大多数人并不会直接写 kernel——他们用 **PyTorch、TensorFlow**，而这些框架底层调用的是 NVIDIA 的**性能库（libraries）**：cuBLAS（线性代数）、cuDNN（深度网络）。可以说，**你写的 AI 代码最终跑在 cuBLAS/cuDNN 的 kernel 上**。理解这两个库，你就理解了「为什么框架里的同一个算子，在 A100 上和在 H100 上会快这么多」——因为性能库会针对每一代新硬件重写内核。本节讲清楚它们各自管什么、如何在 Hopper 上把 TMA/wgmma/FP8 吃到极致。<span class="marginnote">性能库在软件栈中的位置：应用（PyTorch）→ 框架算子层（ATen）→ 厂商库（cuBLAS/cuDNN/cuBLASLt）→ CUDA 驱动 → 硬件。框架层的优化（算子融合）与厂商库的优化（kernel 选择）相互配合，共同决定端到端性能。</span>

## 1 为什么矩阵乘不能自己写

理论上，一个 $M \times N \times K$ 的矩阵乘用三循环就能写：

```cuda
for (i = 0; i < M; i++)
  for (j = 0; j < N; j++)
    for (k = 0; k < K; k++)
      C[i][j] += A[i][k] * B[k][j];
```

但这段代码在 GPU 上会慢得离谱——它完全没利用共享内存的分块复用、没利用 TMA 的异步搬运、没利用 wgmma 的矩阵级指令。**一个「能用」的 GEMM kernel 和一个「吃满 Tensor Core」的 GEMM kernel，性能差距可达 10–50 倍。**

一个工业级 GEMM 要考虑的维度包括：

**分块（tiling）**：把矩阵切成能装进共享内存/寄存器的小块，最大化数据复用；
**流水线**：TMA 预取下一块，wgmma 算当前块，重叠搬运与计算；
- **精度**：FP8/FP16/TF32 的缩放与累积策略；
- **形状启发式**：不同的 $M/N/K$ 与不同的 SM 数对应不同的最优分块；
- **尾块处理**：矩阵维度不是分块倍数时的边界逻辑。

这些优化互相纠缠，且随硬件换代而失效。**性能库的价值就是把这份复杂性封装起来，并用海量自动化测试保证「任意形状下都接近最优」。** 自己写 GEMM 很容易写成「能用但慢一半」——而这在千卡集群上意味着白白浪费几千万的电费与机时。<span class="marginnote">NVIDIA 在 Hopper 上把 cuBLAS 的 FP8 GEMM 用 TMA + wgmma + 双缓冲流水线重写，相对「Ampere 风格」的实现有数倍提升——这就是为什么「换 GPU 后库也要升级」。新硬件的能力，通常要等库更新才真正落地到你的模型。</span>

## 2 cuBLAS：线性代数的事实标准

**cuBLAS（CUDA Basic Linear Algebra Subroutines）**：NVIDIA 的 GPU 线性代数库，实现 BLAS 规范，核心是矩阵乘 GEMM 与一系列分解、求解、向量操作。

对 AI 最关键的是它的 **cuBLASLt（lightweight）** 变体——专门针对「一次配置、反复执行」的 GEMM 场景做了轻量化设计：

**plan 机制**：把「选择算法 + 分配 workspace」放在 plan 阶段，运行时反复执行同一 plan，避免重复决策开销；
- **算法启发式**：对给定形状，在数十种 kernel 变体（不同分块、不同流水线深度）里自动选优；
- **split-k**：当 $K$ 维过大时，把 $K$ 切成多段并行算，再归约——用小批量换取更多并行度。

在 H100 上，cuBLAS 支持 FP8 GEMM（配 Transformer Engine 的缩放）、TF32、以及面向 Hopper 的 wgmma 路径。框架里一个 `torch.matmul` 落到 FP16 GEMM 时，背后就是 cuBLASLt 在几十个候选 kernel 里挑最快的一个。

**辨析｜易错点：** 很多人以为「调 cuBLAS 就是调个参数」，其实它的性能高度依赖**批次形状**。矩阵太小（如 $M=8$
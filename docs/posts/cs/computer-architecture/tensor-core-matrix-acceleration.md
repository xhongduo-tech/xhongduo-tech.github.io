---
title: 张量核心（Tensor Core）与矩阵运算加速
date: 2026-08-07
---

# 张量核心（Tensor Core）与矩阵运算加速

<div class="epigraph">
<p>当 99% 的算力都花在矩阵乘法上时，给矩阵乘法盖一座专门的高速公路，比优化普通车道聪明得多。</p>
<footer>—— 张量核心的设计逻辑</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机体系结构 ｜ Hennessy & Patterson《Computer Architecture: A Quantitative Approach》附录 G ｜ 2026-08-07</p>
</div>

## 为什么 GPU 给矩阵开「专门车道」

深度学习把 GPU 的算力几乎全部消耗在**矩阵乘加（GEMM）**上——卷积、全连接、注意力全是它。2017 年 NVIDIA 在 Volta 架构引入 **Tensor Core（张量核心）**：在 SM 里塞进专门的**矩阵乘加单元**，一条指令干一整块矩阵的活。<span class="marginnote">这是「[[dsa-design-principles]] 领域专用」思想的 GPU 版预演：<strong>把最热的操作做成专用硬件</strong>，换来 10 倍吞吐。它也是 [[google-tpu-systolic-array]] 脉动阵列的「同路人」。</span>

## 1 一条指令：D = A×B + C

**核心概念**：**张量核心（Tensor Core）**执行**矩阵乘加**：$D = A \times B + C$，一整块矩阵运算用**一条指令**完成。例如 Ampere 的 **HMMA（Half-precision Matrix Multiply-Accumulate）**指令：

```asm
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1},         // D/C：16×8 的 f32 累加器（2 个寄存器）
    {a0, a1, a2, a3}, // A：16×16 的 f16 输入（4 个寄存器）
    {b0, b1},         // B：16×8 的 f16 输入（2 个寄存器）
    {c0, c1};         // C 初值
```

对比 CUDA core（普通 ALU）：它们一条指令只做**一个标量**乘加；Tensor Core 一条指令做**数千个**乘加——**指令条数不变，吞吐差一个数量级**。

## 2 混合精度与数值

Tensor Core 的核心优势之一在**混合精度（mixed precision）**：

| 精度 | A/B 输入 | 累加 C/D | 用途 |
| --- | --- | --- | --- |
| FP16 | FP16 | FP32 | 训练/推理 |
| TF32 | 19 位截断 FP32 | FP32 | 免改模型的加速 |
| INT8 | INT8 | INT32 | 推理量化 |
| BF16 | BF16 | FP32 | 训练大模型 |

**关键设计**：**输入可以窄（FP16/INT8），累加器保持宽（FP32/INT32）**——既快又不损失精度。这是矩阵运算「低精度输入 + 高精度累加」的标准配方，也是 [[sparsity-quantization-dsa]] 里量化的硬件基础。

## 3 吞吐对比：CUDA Core vs Tensor Core

| 单位 | 每条指令 | 相对吞吐 |
| --- | --- | --- |
| CUDA Core | 1 次标量乘加 | 基准 |
| Tensor Core（Volta） | 4×4×4 矩阵乘加（128 FLOP） | ~8 倍 |
| Tensor Core（Ampere） | 16×8×16 矩阵乘加（4096 FLOP） | ~10 倍+ |

**核心概念**：Tensor Core 的吞吐优势来自**复用**：一次矩阵乘加，**中间的乘累加结果在片上复用**，不来回搬数据——计算密度（FLOP/访存）远高于标量 ALU。<span class="marginnote">这就是 [[nn-dataflow-stationary]] 里「数据流」思想的体现：矩阵运算把「每个结果要用多个数据」的复用最大化，让 ALU 时刻在算而不是在等数据。</span>

## 4 与 DSA 的关系：GPU 的「领域专用化」

Tensor Core 本质上是一次**领域专用化**：

- **通用性下降**：它只做矩阵乘加，别的运算用不上。
- **性能暴增**：在它的领域里（GEMM），吞吐碾压通用 ALU。
- **编程门槛**：要用专用 API（wmma/mma 内联、cuBLAS、cuDNN），编译器不自动用。

这正好预示了 [[dsa-design-principles]] 的全部逻辑：**当某个操作是压倒性的热点时，专用硬件值得**。Tensor Core 是「GPU 向 DSA 靠拢」的桥头堡，而 TPU 把这条路走到底。

## 5 公式解析：一次矩阵乘加的算力

$$
\text{FLOPs} = 2 \cdot m \cdot n \cdot k
$$

（$m \times k$ 的 $A$ 乘 $k \times n$ 的 $B$，累加进 $m \times n$ 的 $C$）

- **第一步，看乘法次数**：$A$ 的 $m$ 行每行与 $B$ 的 $n$ 列各做 $k$ 次乘 → $m \cdot n \cdot k$ 次乘法。
- **第二步，看加法**：每对乘积还要累加 → 再来 $m \cdot n \cdot k$ 次加法。
- **第三步，代入 HMMA**：$m=16, n=8, k=16$ → $2 \times 16 \times 8 \times 16 = 4096$ FLOP/指令——**一条指令完成 4096 次浮点运算**，而 CUDA core 一条指令只有 2 FLOP。

## 6 程序怎么用上 Tensor Core

普通 CUDA 代码**不会自动**用上张量核心——必须显式调用库或内联指令：

- **cuBLAS/cuDNN**：GEMM/卷积库函数，内部自动把乘加调度到 Tensor Core。
- **wmma/mma 内联（PTX）**：手写矩阵分块，把 A/B/C 声明成「片段（fragment）」后，一条指令做一次 16×8×16 乘加。
- **框架层**：PyTorch/TensorFlow 开启**混合精度（AMP）**后，把卷积/全连接/注意力重写成 GEMM，喂给 Tensor Core。

**辨析｜易错点：** Tensor Core 要求数据布局满足特定分块（如 m16n8k16 的 A 按行主序连续），且输入多为 FP16/BF16。**开不开启混合精度，可能差一个数量级的训练速度**——「硬件有 Tensor Core」和「程序真用上 Tensor Core」是两回事。

## 7 小结

- **Tensor Core** 是 SM 里的专用**矩阵乘加单元**，一条指令 $D = A×B + C$ 完成整块矩阵。
- 混合精度：**窄输入（FP16/INT8）+ 宽累加（FP32/INT32）**，快且准。
- 吞吐比 CUDA core 高一个数量级，靠的是**数据复用**（算多搬少）。
- Tensor Core = GPU 的「领域专用化」，是 [[dsa-design-principles]] 的预演。
- 编程要专用 API（cuBLAS/wmma），编译器不会自动用。

在下一节，我们回到 CPU 侧看循环怎么变并行——**循环级并行与依赖分析**。

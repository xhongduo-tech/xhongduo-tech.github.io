---
title: Tensor Core 与矩阵运算（WMMA/MMA）
date: 2026-08-07
---

# Tensor Core 与矩阵运算（WMMA/MMA）

<div class="epigraph">
<p>因为摩尔定律式微，领域专用架构应运而生——它们在能效与性能上比通用处理器高一个数量级。</p>
<footer>—— 大卫 · 帕特森（David Patterson），2018 年图灵奖讲座</footer>
</div>

<div class="article-byline">
<p>第四级 · GPU 架构与 CUDA 并行编程 ｜ Kirk & Hwu, Programming Massively Parallel Processors, 4e, Ch16；CUDA C++ Programming Guide（WMMA/矩阵运算） ｜ 2026-08-07</p>
</div>

## 为什么从 Tensor Core 开始

前面几篇讲的都是「通用」GPU 运算：一个线程算一个数。但大模型时代的 GPU 把大量晶体管压在了**一个专门的运算**上——矩阵乘法。这个专用的加速单元就是 **Tensor Core（张量核心）**。

理解 Tensor Core，等于理解「从极限到大模型」里算力的真正形态：现代 GPU 的峰值浮点，绝大多数来自 Tensor Core 而非普通 CUDA 核心；训练与推理的性能，几乎全看矩阵乘法喂得快不快。这篇还会与《NVIDIA H100/Hopper》《NVIDIA Blackwell/B200》专题联动。<span class="marginnote">PMPP 第 16 章 "Deep Learning" 讲深度学习在 GPU 上的优化，其中矩阵乘法分块正是 Tensor Core 的用武之地；官方 API 定义见《CUDA C++ Programming Guide》的 wmma 与 mma 指令。本专题另设《NVIDIA H100/Hopper》《NVIDIA Blackwell/B200》专题深入硬件细节，本篇聚焦概念与编程。</span>

## 1 为什么矩阵乘法值得专用硬件

大模型的一切核心运算几乎都是**矩阵乘法（GEMM，general matrix multiply）**：Transformer 的 QKV 投影、注意力打分、FFN，全是 $D = A \times B + C$ 的形式。它有两个特点：

- **算术强度高**：数据能复用多次，计算比访存多得多。
- **形状规整**：本质是「乘累加（FMA）」的批量打包，非常适合流水线化。

于是 NVIDIA 从 Volta 架构（2017）起为它定制了 **Tensor Core**：一个执行单元专门做「小矩阵 × 小矩阵 + 累加」，一条指令完成 SIMT 的 CUDA 核心几十条指令才能完成的工作。A100 的 Tensor Core FP16 峰值约 312 TFLOPS，是其 CUDA 核心 FP32（约 19.5 TFLOPS）的十几倍——**张量核心把「通用计算」和「矩阵计算」在硬件上彻底分了家。**<span class="marginnote">这就是帕特森所说的「领域专用架构」在 GPU 内部的自我演化：先有通用 GPU，再在 GPU 内部长出更专用的 Tensor Core。H100 进一步把 FP16 峰值提到约 989 TFLOPS（稀疏后近 2 PFLOPs），B200 再翻倍——详见对应专题。</span>

## 2 从标量到张量：一条指令算一个块

普通 CUDA 核心的乘累加是标量级的：`d = a * b + c` 一次算一对数。Tensor Core 的乘累加是**张量级**的：一条 **MMA（matrix multiply accumulate）** 指令算的是「一个小矩阵 × 一个小矩阵」，比如 Volta 的 `HMMA.16816` 一次完成 $16 \times 16 \times 16$ 的 FP16 矩阵乘加。

于是编程模型里多了一层抽象：程序员（或库）把大矩阵切成 **tile（块）**，每个 tile 交给一个 warp，warp 里的 32 个线程协作完成这块 tile 的矩阵乘法。**一个 warp 不再「一个线程算一个数」，而是「一个 warp 算一个块」。**

```c
// wmma：Warp-Level Matrix Multiply-Accumulate（CUDA 官方 API）
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag, b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, A, lda);   // 从共享内存加载矩阵块
wmma::load_matrix_sync(b_frag, B, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // 一次算一个 16×16 块
```

<strong>辨析｜易错点：</strong>Tensor Core 并不是「自动加速一切代码」。只有符合它规格的运算（FP16/BF16/TF32/INT8，且 tile 尺寸匹配）才能吃到加速；普通 `float` 标量代码根本走不到 Tensor Core。**加速是有条件的，前提是把你把数据组织成它要的形状。**<span class="marginnote">多数时候你不需要手写 wmma——调用 cuBLAS、cuDNN、CUTLASS 或框架（PyTorch、DeepSeek 系的 MLA 实现等），库内部已经用 Tensor Core 帮你算好了。手写 wmma/mma 的场景是：库不满足你的特殊布局、或者你在写自定义算子。</span>

## 3 公式解析：GEMM 的 FLOPs 与算术强度

为什么 GEMM 能吃满张量核心？用两个数字说话。对 $n \times n \times n$ 的矩阵乘法 $C = A \times B$：

$$
\text{FLOPs} = 2n^3, \qquad I = \frac{2n^3}{3n^2 \times s} = \frac{2n}{3s}
$$

拆解这条公式：

- **$\text{FLOPs} = 2n^3$**：$n \times n$ 的每个输出元素要算 $n$ 次乘加，每次乘加算 2 次浮点运算（1 乘 1 加），共 $n^2 \times n \times 2 = 2n^3$。
- **$3n^2$**：三个 $n \times n$ 矩阵的总元素数（A、B 各 $n^2$，C 也是 $n^2$，但 C 可复用）。
- **$s$**：每个元素的字节数。FP32 时 $s=4$，FP16 时 $s=2$。
- **$I$（算术强度）**：每搬 1 字节数据能做多少次浮点运算，是 Roofline 模型的关键输入。

代入 $n = 4096$：$I = 2 \times 4096 / (3 \times 4) \approx 683$ FLOPs/字节（FP32）。而 H100 的显存带宽约 3.35 TB/s——**要喂饱 Tensor Core，靠通用访存根本不够，必须靠「分块 + 共享内存复用」让数据在片上反复用**：把一块 A 和一块 B 读进共享内存，算这一块能做的全部乘加，再换下一块。分块策略（tiling）正是 PMPP 第 16 章与 cuBLAS 的核心内容。<span class="marginnote">这也解释了为什么 GEMM 是「计算受限」的典型：算术强度高达数百，远在 Roofline 的「临界强度」之上，瓶颈是计算而非带宽——与上一篇讲的内存受限 kernel 正好相反。</span>

## 4 混合精度：用精度换速度

Tensor Core 的另一个关键点是**精度**。早期 Tensor Core 主打 FP16 输入、FP32 累加——输入用半精度省带宽、减一半寄存器压力，累加用全精度保住数值质量。这被称为**混合精度（mixed precision）**。

为什么能这么用？深度学习的权重与梯度天然有「容忍噪声」的特性——FP16 的约 10 位尾数对大部分训练场景足够，累加误差则被 FP32 累加器压住。于是：

- **FP16**：输入 2 字节，Tensor Core 吞吐最高（A100 312 TFLOPS）。
- **TF32**：Volta 之后引入，输入仍是 FP32 的内存布局，但 Tensor Core 内部只用 19 位精度——为了「不换代码就吃到部分张量加速」。
- **BF16**：尾数更少、指数范围更大，训练大模型时的首选低精度之一。

**代价是数值风险**：用错精度可能让 loss 发散。正确的做法不是「全低精度」，而是「输入低精度、累加高精度 + 定期检查数值」——现代框架（PyTorch AMP、DeepSeek 系训练）内置了这套流程。

## 5 与 H100/B200 的联动：专用化的继续

Tensor Core 不是终点，而是「专用化」路线的起点。沿着这条线：

- **Hopper（H100）**：引入 **WGMMA（warpgroup MMA）**，warp 组共享一个更大的累加器，把张量核心的利用率再推高一档；同期还有针对稀疏性的 2:4 结构化稀疏支持。
- **Blackwell（B200）**：把张量核心的 FP4/FP8 能力推向推理主力，配合 NVLink 互联（见《多 GPU 编程》篇）。
- **GPGPU 之外**：TPU、昇腾等 AI 芯片干脆把「矩阵乘加阵列」做成整颗芯片——GPU 的 Tensor Core 正是这条路的硬件先驱。

读到这里，你应该形成一条完整的因果链：**功耗墙 → 吞吐并行（SIMT）→ 内存带宽成为瓶颈 → 用共享内存分块复用 → 把「复用最多的运算」固化进硬件（Tensor Core）→ 再让这一级专用化继续推进（WGMMA、FP4）。** 本专题后面的《性能分析与调优》篇，正是量化这条链上每一环的得失。

## 6 什么时候你才需要手写张量核心

读到 wmma 代码后，最常见的困惑是「我也要这么写吗」。绝大多数时候——**不用**。真实生产链路上，Tensor Core 是这样被用起来的：

| 入口 | 抽象层次 | 谁在用 |
| --- | --- | --- |
| cuBLAS / cuDNN | 函数级 | 几乎所有框架与库 |
| CUTLASS | 模板级 | 追求极致性能的团队 |
| wmma（CUDA 内置） | 指令级 | 自定义算子作者 |
| mma（PTX 内联） | 汇编级 | 极少数极致定制 |

- **cuBLAS**：`cublasGemmEx` 一行调用，内部自动选择张量核心路径与分块策略，性能由 NVIDIA 专家打磨。
- **cuDNN**：卷积被隐式 GEMM（implicit GEMM）转成矩阵乘法，框架调用即吃到张量核心。
- **PyTorch / TensorFlow**：`torch.matmul`、`nn.Linear` 底层走 ATen → cuBLAS/cuDNN → 张量核心，用户零感知。
- **CUTLASS**：NVIDIA 的开源 GEMM 模板库，给「要极致性能或特殊布局」的团队当脚手架。

手写 wmma/mma 的真正场景只剩三类：① 库不覆盖的自定义算子（如 MoE 的专家分组、MLA 的合并投影）；② 需要把「分块 + 共享内存 + 张量核心」按特殊数据布局拼装；③ 教学与研究——正如本篇所做。

**辨析｜易错点：** 调 cuBLAS 时若「精度参数」没配对，可能悄悄回落到普通 CUDA 核心路径，性能掉一个数量级而你却不自知——**验证「是否真的用了张量核心」，要看 Nsight Compute 的 Tensor 指令计数，而不是猜。** 这也是《性能分析与调优》篇「先测量」的又一次现身。

一句话：**工程师的默认动作是「调库」，而不是「造轮子」。** 只有当库给不了你想要的形状或性能时，CUTLASS 与 wmma 才值得你出手——理解 Tensor Core 的原理，正是为了在需要出手时知道自己在做什么。

（FP8 从 H100 起成为推理主力格式，这一趋势在《NVIDIA H100/Hopper》《NVIDIA Blackwell/B200》专题有完整数据与算例。）

## 7 小结

- **Tensor Core** 是为矩阵乘法定制的专用执行单元，一条 MMA 指令算一个矩阵块，而非一对数。
- A100 Tensor Core FP16 峰值约 312 TFLOPS，远高于其 CUDA 核心 FP32（约 19.5 TFLOPS）。
- GEMM 的 FLOPs = $2n^3$，算术强度 $I = 2n/(3s)$：计算受限，靠分块 + 共享内存复用喂满带宽。
- **混合精度**：FP16/BF16 输入 + FP32 累加，用精度换速度，需防数值发散。
- 专用化继续演进：Hopper 的 WGMMA、Blackwell 的 FP4/FP8，以及 TPU/昇腾的整芯片矩阵阵列。
- 多数场景直接用 cuBLAS/cuDNN/框架即可吃到 Tensor Core，手写 wmma 属高级定制。

在下一节，我们把「CPU 拷贝数据的同时 GPU 算」这件事做成工程：**流、事件与并发执行（计算/传输重叠）**。

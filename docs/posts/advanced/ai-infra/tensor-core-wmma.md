---
title: Tensor Core 与 WMMA/mma 指令编程
date: 2026-08-07
---

# Tensor Core 与 WMMA/mma 指令编程

<div class="epigraph">
<p>任何足够先进的技术，都与魔法无异。</p>
<footer>—— 阿瑟 · C · 克拉克（Arthur C. Clarke）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Tensor Core 开始

上一课我们把矩阵乘法的算术强度抬过了 ridge point，但还差最后一口气：即使「算力打满」，用的是 CUDA Core 的 FP32——而大模型训练真正依赖的是 GPU 里一块**专门为矩阵乘法定制的硬件：Tensor Core**。它一次指令就能完成整块矩阵的乘加，把 FP16 算力推到 FP32 的十几倍。不理解 Tensor Core，就看不懂为什么 H100/A100 如此昂贵、为什么混合精度是训练的默认选项、为什么 cuBLAS 能接近硬件峰值。<span class="marginnote">Tensor Core 的思路与 CPU 里的 SIMD 一脉相承：<strong>与其让几十个线程各算一个标量，不如一条指令算一整块矩阵</strong>——这是「数据级并行」在硬件上的终极形态。它对标的数学对象，就是基础数学里「矩阵乘法是 $C_{ij}=\sum_k A_{ik}B_{kj}$ 的批量乘加」这一条。</span>

## 1 Tensor Core：为矩阵乘法量身定做的硬件单元

**Tensor Core（张量核心）**：GPU SM 内专门执行矩阵乘加 $D = A \times B + C$ 的硬件单元。它不是「用指令模拟矩阵乘」，而是把**一次指令、一整块矩阵**直接烧进电路。程序员不能按标量方式逐个调用它，必须用**warp 集体（warp-collective）**方式：一个 warp 的 32 个线程合作执行一条 mma 指令，每个线程贡献矩阵的一小片（称为 **fragment**，寄存器中的固定布局）。

PTX 的 `mma.sync` 指令定义了若干**形状（shape）**，即一次指令算多大一块：

| 指令形状 | 典型类型 | 最小架构 |
| --- | --- | --- |
| `m16n8k8` | FP16/BF16 | Turing（sm_75） |
| `m16n8k16` | FP16/BF16/INT8 | Ampere（sm_80，A100） |
| `m16n8k32` | INT8 / FP8 | Ampere / Ada+ |
| `wgmma.m64nNk16` | FP16/BF16 | Hopper（sm_90，H100） |

形状名里三个数字是 **m × n × k**：m 与 n 决定输出块的大小，k 决定一次乘加的归约深度。以 Ampere 最常用的 `m16n8k16` 为例：**一次指令算 $16 \times 8$ 的输出块，沿 k 归约 16**，即 128 个输出、每个累加 16 次乘加。<span class="marginnote">为什么是这些形状？它们对应<strong>寄存器与数据搬移的最优折中</strong>：m16n8 让 32 个线程恰好人手一份输出、k16 让 A 的 fragment 正好 4 个 32 位寄存器。形状一旦定死，程序员就得按它的寄存器布局喂数据——这就是 WMMA 存在的意义。</span>

## 2 性能的量化意义：算力翻一个数量级

把数字摆出来，Tensor Core 的分量一目了然：

| 芯片 | FP32 CUDA Core | FP16 Tensor Core（稠密） | 加速比 |
| --- | --- | --- | --- |
| A100 | 约 19.5 TFLOPS | 312 TFLOPS | 约 16× |
| H100 | 约 67 TFLOPS | 约 990 TFLOPS | 约 15× |

**FP16 的 Tensor Core 算力比同一块芯片的 FP32 高一个数量级。** 这正是大模型训练默认走混合精度（FP16/BF16）的硬件原因——用 FP16 喂给 Tensor Core，训练吞吐立刻翻十几倍，代价是额外的精度管理（loss scaling、主权重），那是本主题第四篇《混合精度训练》的内容。<span class="marginnote">Ampere 的 Tensor Core 还支持 <strong>2:4 结构化稀疏</strong>：权重四个里恰有两个非零时，吞吐再翻倍（A100 FP16 稀疏 624 TFLOPS）。稀疏是「用硬件换算力」的极致，但在大模型里由于需要专门剪枝训练，应用不如稠密普遍。</span>

## 3 两级编程入口：WMMA 与 mma.sync

在 CUDA 里用 Tensor Core 有三条路，从易到难：

1. **cuBLAS / cuDNN / cuBLASLt**：闭源库，调 `cublasGemmEx` 一行搞定，NVIDIA 帮你榨干硬件。绝大多数生产代码停在这一层。
2. **CUTLASS**：NVIDIA 开源的模板库，把 GEMM 拆成 tile、fragment、指令三级，是「要写自己的高性能 kernel 又不想碰裸汇编」的主力。
3. **WMMA（`nvcuda::wmma`）与内联 PTX `mma.sync`**：手写 kernel 时直接控制 Tensor Core 的两级接口。

本课聚焦第三层，因为它是理解「Tensor Core 到底怎么被驱动」的必经之路。**WMMA** 是 C++ API，用 `fragment` 抽象帮你管理寄存器布局；**mma.sync** 是内联汇编级的裸指令，寄存器布局完全由你负责，性能上限最高，复杂度也最高。<span class="marginnote">cuBLAS 的高性能内核（cuBLASLt 的 splitK、CUTLASS 的 persistent kernel）本质上都在 <strong>mma / wgmma 指令之上拼装配器</strong>。看懂 mma，你就看懂了闭源库最内层的那个循环。</span>

## 4 WMMA 编程模型：fragment 的装载、计算与写回

WMMA 的核心抽象是 **fragment**：矩阵块在 32 个线程寄存器里的分布，由编译器按硬件布局填好。程序员只做三件事：**装载（load）→ 乘加（mma_sync）→ 写回（store）**。

```cpp
#include <mma.h>
using namespace nvcuda;

// 每个 block 算一个 16×16 输出块（简化版，未处理越界）
__global__ void wmma_matmul(const __half* A, const __half* B, float* C,
                            int K) {
    // 声明三个 fragment：A 取 16×16（row-major），B 取 16×16（col-major），C 累加器
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);               // 累加器清零

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + blockIdx.y * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * 16 + blockIdx.x * 16, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // D = A×B + C
    }
    wmma::store_matrix_sync(C + blockIdx.y * 16 * 16 + blockIdx.x * 16,
                            c_frag, 16, wmma::mem_row_major);
}
```

三个关键点：

- **fragment 的维度参数必须与指令形状匹配**：上面 `fragment<matrix_a, 16, 16, 16, half, row_major>` 声明的就是「m16n8k16 形状里 A 的那一份」。
- **布局是硬件规定的，不可假设**：fragment 里第几个元素对应矩阵哪一行哪一列，由编译器/硬件决定；你只能通过 `load_matrix_sync` 装载、通过 `mma_sync` 计算，**不要手写寄存器布局去猜**。
- **累加器用 FP32**：即使输入是 FP16，累加器也用 `float` 保精度——这是混合精度稳定性的根基。

更低一层，内联 PTX 直接暴露 fragment 的寄存器布局。`m16n8k16` 下每个线程：A fragment 4 个 32 位寄存器（装 8 个 FP16），B fragment 2 个寄存器（装 4 个 FP16），累加器 4 个 FP32 寄存器：

```cuda
// A 的 4 个 b32（各装 2 个 half）与 B 的 2 个 b32，累加器 4 个 f32
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
```

喂给这些寄存器的数据，通常用 `ldmatrix` 指令从 Shared Memory 批量装载——这正是上一课 tiling 的产物在 Tensor Core 层级的落点。<span class="marginnote">注意分工：<strong>mma 指令只负责「算」</strong>，把数据搬进 Shared Memory、再搬进 fragment 寄存器的脏活全要你干。高性能 GEMM 的优化重点，恰恰在「怎么把数据以正确的布局喂进去」——CUTLASS 大半的复杂度都在这。</span>

## 5 公式解析：用 FLOPs 反推硬件规格

Tensor Core 的标称算力（312 TFLOPS）看起来像魔法，但用一次反推就能验证它来自何处，顺便让「一次 mma 算多少」变得可感。

**A100：108 个 SM，Boost 时钟约 1.41 GHz。** 若 FP16 稠密 312 TFLOPS 为真，则每个 SM 每周期要完成

$$
\frac{312 \times 10^{12}}{108 \times 1.41 \times 10^{9}} \approx 2048 \ \text{FLOPs/SM/cycle}
$$

做三步拆解：

- **第一步，换算单位**：把 TFLOPS 除以（SM 数 × 时钟频率），得到「每个 SM 每周期平均做多少次浮点运算」。这是所有「规格反推」的第一步。
- **第二步，对齐指令形状**：一条 `m16n8k8` 的 mma 做 $16 \times 8 \times 8 = 1024$ 次乘加，每次乘加算 2 次浮点（一乘一加），共 $1024 \times 2 = 2048$ FLOPs——**恰好等于一个 SM 每周期的预算**。
- **第三步，读结论**：也就是说，A100 的标称算力等价于「每 SM 每周期流水线化地完成一次 m16n8k8」。硬件规格不是拍脑袋，而是与指令形状严格自洽的——这也是「算力规格」背后真正的含义：**它在告诉你，这条指令可以每周期来一次**。<span class="marginnote">同样的反推可以用于估算任何 kernel 的算力天花板：<strong>先数清一次 mma 的 FLOPs，再乘以每周期能发几条，就得到峰值</strong>。Roofline 模型（本主题收尾篇）把这套逻辑系统化。</span>

再对比一条：FP16 Tensor Core 与 FP32 CUDA Core 的差距 $\frac{312}{19.5} \approx 16$，说明**同样一块硅，走专用矩阵硬件比走通用计算单元多出 16 倍吞吐**——这就是「专用化（specialization）」换取性能的极致样本。

## 6 辨析｜易错点

- **「fragment 的寄存器布局可以自己猜」**——错。布局由硬件固定且未公开为简单规律，擅自手工填充会得到错位结果。要么用 `load_matrix_sync`，要么查证精确的 `ldmatrix` + 索引公式。
- **「输入是 FP16，累加也应该是 FP16」**——错。**累加器用 FP32**（`f32.f16.f16.f32` 末尾的 f32），否则 K 一大精度崩溃。混合精度的「混合」正体现在这里。
- **「mma 指令自己会把数据搬进 Shared Memory」**——错。mma 只算不算搬；数据从全局到 shared、从 shared 到 fragment 都要你显式完成。
- **「矩阵很小也能白赚 16 倍」**——不行。mma 的最小粒度是 m16n8，加上 k 对齐与 warp 集体开销，M/N/K 过小（如几×几）时，数据搬运与对齐成本远超收益。小矩阵要用别的策略（如直接乘在共享内存、或 padding 到大块）。
- **「FP16 Tensor Core 的 16 倍是白给的」**——不白给。它需要你同时承担混合精度的工程复杂度（loss scaling、BF16 与 FP16 的选择），详见第四篇《混合精度训练》。

## 7 小结

- **Tensor Core** 是 SM 内专做 $D = A\times B + C$ 的硬件单元，以 **warp 集体**方式执行 `mma` 指令，形状如 m16n8k8 / m16n8k16 / m16n8k32。
- FP16 Tensor Core 算力约为同芯片 FP32 的 **16 倍**（A100 312 vs 19.5 TFLOPS），是大模型训练默认混合精度的硬件根源。
- 编程入口从易到难：**cuBLAS → CUTLASS → WMMA / mma.sync**；WMMA 用 `fragment` + `load_matrix_sync` / `mma_sync` / `store_matrix_sync` 三步驱动。
- mma 指令只负责计算，数据搬移（global → shared → fragment）是程序员的职责；累加器必须用 FP32。
- 用 `312e12 / (108 × 1.41e9) ≈ 2048` 可反推：A100 的规格等价于「每 SM 每周期完成一次 m16n8k8」。

在下一节，我们把前几课的所有工具（算术强度、tiling、Tensor Core）汇成一把尺子——**Roofline 模型**：一眼判断一个 kernel 到底卡在计算还是访存，以及离硬件峰值还差多远。

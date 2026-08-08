---
title: Kernel 融合与自定义 CUDA Kernel
date: 2026-08-07
---

# Kernel 融合与自定义 CUDA Kernel

<div class="epigraph">
<p>数据在哪里，计算就应该在哪里；搬数据的代价比算数据贵。</p>
<footer>—— 常见 GPU 优化口诀（源自 NVIDIA 性能优化指南）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ NVIDIA TensorRT 文档与 CUDA 编程指南 ｜ 2026-08-07</p>
</div>

## 为什么从 Kernel 融合开始

上一篇讲 TensorRT 的图优化是「消层」——把多余的层删掉、把便宜的替代换上。但 LLM 里真正的性能黑洞，是那些**拆不开的、密集出现的 elementwise 小算子**：残差加、LayerNorm、GeLU、SiLU、逐元素乘法。它们单独看每层都要访问一遍全量数据、发射一次内核；合起来看，同样的数据被反复从显存搬进搬出。<span class="marginnote">本专题《访存瓶颈：为什么 Decode 是 Memory-Bound》已经算过：<strong>GPU 每秒能搬的字节数远小于能做的浮点运算数</strong>，elementwise 算子几乎全是带宽型。</span>

**Kernel 融合（kernel fusion）**的朴素想法是：把「对同一块数据做一串操作」的多个内核，合并成一个内核，数据只从显存读一次、写一次。中间结果放在寄存器或共享内存里，而不是显存。这个看似简单的想法，在 LLM 推理里是把延迟打下来的头号手段——FlashAttention 本质上也是一个巨型融合内核。本篇讲融合的动机、分类、实现手段，以及为什么某些融合必须靠自定义 CUDA Kernel。

## 1 为什么融合能快：访存的账

假设要依次执行加、乘、GeLU 三个算子，数据大小为 $M$ 字节。不融合时，每个算子都要：从显存读输入、写入输出。总访存量约为 $6M$ 字节（三次读、三次写，近似）。

融合成一个内核后：从显存读一次 $x$（$M$ 字节），在寄存器/共享内存里做完三步，写一次 $z$（$M$ 字节）。总访存约 $2M$ 字节。

**访存从 $6M$ 降到 $2M$，节省 3 倍**——而这三步运算本身的浮点开销，现代 GPU 一眨眼就做完了。融合的收益几乎全部来自「少搬数据」。<span class="marginnote">更精确的账见下一篇公式解析：设访存带宽为 $B$，<strong>访存型算子的耗时 ≈ 数据量 ÷ 带宽</strong>，与算力无关。带宽就是融合要省的那个「水龙头」。</span>

## 2 融合的分类：elementwise、IO-Bound 与通用

按融合的粒度与对象，可以分成三类：

**Elementwise 融合**：把一串逐元素算子（加、乘、激活、缩放）合并。这类算子没有跨元素依赖，最容易融合，收益也最直接。Transformer 里的残差加 + LayerNorm + GeLU 就是教科书案例。
**IO-Bound 融合**：把「计算很小、纯访存」的算子（如 Reshape、Transpose、Concat）合并进周围的访存流程，消灭中间张量。Reshape 本身不改变数据内容，独立成一个内核就是白读白写一整份显存。
**通用融合**：需要跨块通信或共享中间结果的融合，如 Softmax 的行归约 + 归一化、FlashAttention 的注意力 + 归一化。这类融合要仔细设计线程块划分，通常必须手写 CUDA。

**辨析｜易错点：不是融合越多越好。** 融合会破坏算子的可组合性，也会增加内核里寄存器压力——融合过度导致寄存器溢出（spill）到局部内存，反而变慢。TensorRT 的做法是**按成本模型选**：对每个候选融合，估算「省下的访存」与「多付的寄存器/共享内存」，只在净收益为正时融合。

## 3 从图优化到自定义 Kernel

图优化（上一篇）负责在 DAG 上发现「哪些层可以融合」；**真正执行融合要靠自定义 CUDA Kernel**——标准的 PyTorch 算子库里没有「残差+LayerNorm+GeLU 三合一」的算子。

一个典型的融合内核长这样（伪代码）：

```cpp
// 融合内核：y = GELU(LayerNorm(x + residual))，逐块处理，数据只读一次、写一次
__global__ void fused_residual_layernorm_gelu(const float* x,
                                              const float* residual,
                                              const float* gamma,
                                              const float* beta,
                                              float* y, int N) {
    extern __shared__ float smem[];       // 共享内存暂存本块数据
    // 1. 残差加：r[i] = x[i] + residual[i]
    // 2. block 内归约求 mean 与 rstd（两次归约，见 LayerNorm 实现）
    // 3. 归一化：n[i] = (r[i] - mean) * rstd * gamma[i] + beta[i]
    // 4. GeLU（tanh 近似）：y[i] = 0.5 * n[i] * (1 + tanh(0.79788456 * (n[i] + 0.044715 * n[i]^3)))
    // 5. 写回 y —— 中间结果 r、n 只存在于寄存器/共享内存，不落地显存
}
```

真实工程里比这复杂得多：LayerNorm 的均值和方差需要在 block 内做归约，数据要在共享内存里暂存，还要考虑向量化访存（`float4` 128-bit 加载）与内存对齐。这正是 TensorRT 用大量手写 CUDA kernel（或借助 CUTLASS、FlashAttention 库）的原因。<span class="marginnote">注意上面 GeLU 用了 tanh 近似。工程中还有 Erf 精确版，两者对训练无差别、对量化模型有微小影响——<strong>内核里的数学近似必须与训练时一致，否则精度评测会翻车</strong>。</span>

## 4 公式解析：融合后的加速比

设一次推理中被融合的一组算子，原本需要 $K$ 次内核执行、总访存量 $Q$ 字节，融合后为 1 次内核、访存量 $Q'$ 字节，且访存带宽 $B$ 为瓶颈（elementwise 算子的常态）。原本耗时约为：

$$T_{\text{orig}} \approx \frac{Q}{B} + K \cdot s$$

其中 $s$ 是单次内核启动开销（约 2–10 微秒）。三步拆解：

- **第一步，理解 $Q/B$**：带宽瓶颈下，数据搬运时间 = 字节数除以带宽。对 40 GB 显存、带宽约 2 TB/s 的 A100，搬一整份 14 GB 权重要 7 毫秒量级；搬一层激活则只要微秒级——**数据越小的算子，越不该独立成内核**。
- **第二步，理解 $K \cdot s$**：每个内核的发射都有固定开销。$K$ 个内核的启动开销累加，在小算子主导的层里不容忽视。
- **第三步，对比融合后**：$T_{\text{fused}} \approx Q'/B + s$。当 $Q' \ll Q$（中间结果不再落地显存）且 $K \gg 1$ 时，

$$\frac{T_{\text{orig}}}{T_{\text{fused}}} \approx \frac{Q + K \cdot s \cdot B}{Q' + s \cdot B}$$

对典型的 LayerNorm 融合场景，$Q'/Q$ 可达 1/3 到 1/5，$K$ 从 5–8 降到 1，实测加速 2–4 倍。**融合的核心就是在 $Q$ 和 $K$ 两个维度同时做减法**。

## 5 小结

- **融合消灭中间张量的显存往返**：把 $K$ 个访存型小内核合并为 1 个大内核，数据只读一次、写一次。
- **三类融合粒度**：elementwise 融合最易做、IO-Bound 融合消灭透明传输、通用融合（softmax/注意力）需精心设计线程块。
- **融合要手写 CUDA Kernel**：标准算子库没有「残差+LayerNorm+GeLU」组合算子，TensorRT 借助 CUTLASS 与手写内核实现。
- **数学近似必须与训练一致**：GeLU 的 tanh 近似、LayerNorm 的 eps 取值都要和训练期对齐，否则影响量化与精度评测。
- **不是越多越好**：寄存器溢出与可组合性损失让「过度融合」反而变慢，要用成本模型把关。

在下一节，我们讨论让 TensorRT-LLM 在**动态请求流**下保持高吞吐的关键——**In-flight Batching 的原理**，它与 vLLM 的 Continuous Batching 是一对孪生兄弟。

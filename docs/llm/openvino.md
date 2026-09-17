---
title: OpenVINO
date: 2026-09-08
section: llm
---

# OpenVINO

<div class="epigraph">
<p>把模型编译到 Intel CPU、核显与独显的统一运行时：设备叫 CPU/GPU/NPU，编译器做的是图优化与量化，而不是再训一份权重。</p>
<footer>—— Intel OpenVINO 文档与 NNCF 量化工具链</footer>
</div>

[上一课](/llm/onnx-runtime)用 EP 把图画到多种后端。OpenVINO 是 Intel 硬件上的专用栈：IR（XML+bin 或从 ONNX/PyTorch 读入），`compile_model` 到指定 device，编译器在此完成图规范化、算子融合与 ISA 降级。LLM 路径有 Optimum-Intel、GenAI 库一类封装，处理 tokenizer 与逐步 generate。本课写它相对 ORT 的差：更贴 Intel ISA（AVX、[AMX](/llm/amx-kernel)）、核显共享内存，以及 NPU 的形状限制。不把某一代酷睿的 token/s 写成常数。

## 问题

目标设备是「没有独享 NVIDIA HBM 的机器」：服务器至强、笔记本核显、客户端 NPU。会计改写：带宽是 DDR 或 LPDDR，不是 HBM；算术是 AMX/AVX 或核显 EU。缺口是编译器必须把 Transformer 降到这些 ISA，并处理 KV 放哪——统一内存上 CPU 与 iGPU 共享，拷贝语义与 CUDA 不同：零拷贝省的是设备间搬运，省不掉 DDR 带宽本身这道关。动态 $n$ 在 NPU 上往往要形状桶或 pad 到最大值，容量与碎片问题以另一种硬件回来。

量化几乎是默认：INT8/INT4 权重量化才能让 7B 进 16GB 笔记本。只压权重与全量化是两条不同的账：权重量化把激活留在 BF16，GEMM 吃不到 AMX 的 INT8 吞吐，但 decode 本来带宽受限，权重流减半就接近直接翻 token/s；权重与激活都落 INT8，GEMM 才能走上 AMX 矩阵扩展，收益集中在 prefill 的胖 GEMM。这与 GPU FP8 服务不是同一套核，质量要在目标设备上验收。

<span class="marginnote">OpenVINO 的「GPU」多指 Intel 核显/独显，不是 CUDA。文档里的 GPU 插件不要当成能跑 CUDA FA。</span>

## 方法

导出或用 Optimum 转 IR，指定 `device=CPU|GPU|NPU`。LLM 用官方 GenAI / pipeline 管 KV：逐步 generate 的调度、KV 生命周期与采样都在运行时里；应用层拿 numpy 自己拼会绕开 AMX 核与 KV 管理——运行时的核选择按设备与形状做过离线调优，绕开即退回通用路径，token/s 反而降。批处理在端侧常是 1，工作点永远在带宽区；优化是减 $W_{\mathrm{bytes}}$（量化层数与分组粒度）与更好的 matmul 核，不是堆 $B$。与 ORT 选择：Intel 硬件优先 OpenVINO；要可移植到 NVIDIA 再 ORT/CUDA。

```mermaid
flowchart TD
  SRC["PyTorch / ONNX"] --> IR["OpenVINO IR"]
  IR --> CMP["compile_model"]
  CMP --> CPU["CPU AMX/AVX"]
  CMP --> IGPU["Intel GPU"]
  CMP --> NPU["客户端 NPU"]
```

## 机制

图捕获是第一道收益。IR 把算子图规范化后做常量折叠与融合：矩阵乘后的归一化与激活并入同一个核，attention 的多个小算子并成大核。为什么：每融合一层，中间张量就少一次 DDR 往返；端侧带宽贵，小算子各自读写内存时算得再快也在等内存。不做融合的图，GEMM 的提速会被访存账整段吃掉，这是「换了运行时却没快」最常见的原因。

AMX 把 decode 的瘦 GEMM 从 AVX 点积换成 tile 乘，每周期乘加数约为 BF16 tile 的两倍。它改变 token 吞吐的哪一段要拆开看：prefill 的胖 GEMM 计算受限，吃满 AMX 即近线性提速；decode 的瘦 GEMV 算术强度低、阵列喂不满，收益主要来自 INT8 权重把每步权重流量砍半。所以「INT8 加速几倍」必须拆 prefill 与 decode 分别报，混报会把带宽账记进算术账。强度仍受 DDR 限制，但常数更好。核显共享内存减少拷贝，但算力与带宽都低于独享 HBM GPU。NPU 对静态图友好，对投机、动态掩码不友好。工作点回到[屋顶线](/llm/arithmetic-intensity-decode)，只是换了峰值数字。

## 边界

不要用数据中心 GPU 的 TPOT SLA 套笔记本：功耗墙与内存通道数不同，同一份 IR 在两边的瓶颈不在同一处。不要假设 NPU 支持任意自定义采样，投机解码的动态分支先查算子覆盖。下一课：Apple 的 [MLX](/llm/mlx-apple)，统一内存故事更完整。

## 小结

- OpenVINO 把模型编译到 Intel CPU/iGPU/NPU；LLM 靠 GenAI 管 KV。
- 端侧 $B=1$，优化靠量化与 ISA 核，不靠堆并发。
- 设备名 GPU ≠ CUDA。
- 动态形状在 NPU 上要桶或 pad。
- 质量在目标量化路径上验收；INT8 收益拆 prefill（算术）与 decode（带宽）报。
- 出处：Intel OpenVINO 与 NNCF 文档；AMX 见 Intel SDM。

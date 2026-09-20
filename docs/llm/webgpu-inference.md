---
title: WebGPU 推理
date: 2026-09-08
section: llm
---

# WebGPU 推理

<div class="epigraph">
<p>浏览器里的 GPU 没有 CUDA 运行时：计算着色器加上严格的内存配额，能跑的是量化后的小模型，不是把 vLLM 搬进标签页。</p>
<footer>—— W3C WebGPU；浏览器 LLM 见 MLC WebLLM 与 transformers.js 一类 TVM / ONNX Runtime Web 路径</footer>
</div>

[上一课](/llm/mlx-apple)还有原生进程与统一内存。WebGPU 把计算放进沙箱：着色器、缓冲配额、无自定义特权内核。MLC 的 WebLLM 用 TVM 把模型编译到 WebGPU；transformers.js 常用 ONNX Runtime Web 或 WASM。本课写约束：权重要下载到浏览器、KV 占 JS/GPU 缓冲、decode 在带宽与启动开销双重劣势下跑。隐私（权重与数据不出页）是产品动机，不是性能动机。

## 问题

下载 4-bit 7B 仍是数 GB，首次 TTFT 含网络。配额与移动浏览器会直接拒绝。逐步：没有 [FlashAttention](/llm/flashattention) 那种 CUDA 核，注意力是着色器或退化实现，$n$ 稍长即炸。缺口是承认 Web 路径的模型级（1B–8B 量化、短上下文），以及预填充与 decode 的着色器启动开销。采样在 JS 或 WASM，词表扫描相对 GPU 着色器可能更刺眼——先修采样器内核的反面。

<span class="marginnote">TTFT（Time To First Token）指从发出请求到看见第一个字的等待时间。数字感受一下：7B 参数做 4-bit 量化，每参数约 0.5 字节，仅权重就约 3.5 GB——相当于先下完一部标清电影，模型才说得出第一个字。所以浏览器产品必须把「加载中」当成一等交互来设计。</span>

没有连续批：一页一个用户，$B=1$，工作点永远在带宽区。优化是量化、减层、短上下文，不是堆并发。

<span class="marginnote">WebGPU 与 WebGL 不同。旧的 WebGL 路径不适合通用计算。检测能力失败时应回退 WASM CPU，并明确更慢。</span>

## 方法

编译：TVM / ORT 把图降到 WGSL。分发：分片下载、缓存到 Origin Private File System。KV：固定最大 $n$ 预留，碎片回到 OpenVINO NPU 那类问题。流式：id 在 Worker 里 generate，主线程 detokenize，注意 UTF-8 稳定前缀。投机在浏览器上草稿也得是小模型或早退，下一课端侧投机再写；Web 上内存更紧。

<span class="marginnote">「着色器」原是显卡里给每个像素上色的小程序，WebGPU 把它推广成通用并行小程序：把一万条数据切一万份，每份由一个着色器实例同时处理。浏览器里的矩阵乘法全靠它跑，这也是为什么没有定制 CUDA 核时，注意力只能写成退化实现。</span>

<span class="marginnote">Origin Private File System（OPFS）可类比为浏览器分给每个网站的秘密仓库：用户在文件管理器里看不到里面的内容，但页面读写自己的权重分片不用征询下载文件夹。模型缓存在这里，第二次打开页面就不必重新下载数 GB。</span>

```mermaid
flowchart TD
  DL["分片下载权重"] --> COMP["TVM/ORT → WebGPU"]
  COMP --> STEP["着色器一步 decode"]
  STEP --> JS["JS/WASM 采样"]
  JS --> UI["稳定前缀上屏"]
```

## 机制

沙箱禁止持久内核与无限显存。每步着色器启动相对 CUDA Graph 更贵，小核更多伤。会计公式仍用，$B_{\mathrm{HBM}}$ 换成共享显存或系统内存。安全：模型文件来自源站，仍要完整性校验，避免恶意着色器或权重。这与 safetensors 的供应链同一类，只是运行在浏览器官辖。

```mermaid
flowchart TD
  REQ["一条生成请求"] --> B{"能凑批吗?"}
  B -->|"一页一用户, B=1"| BW["每步都要重读全部权重 → 卡显存带宽"]
  B -->|"服务器凑大批"| COMP["多个请求分摊一次权重读取 → 卡算力"]
  BW --> OPTW["优化: 量化 / 减层 / 短上下文"]
  COMP --> OPTC["优化: 连续批处理调度"]
```

<span class="marginnote">常见误区：把服务器吞吐优化照搬到浏览器。服务器靠大批请求分摊权重读取，算的是算术强度；浏览器永远 B=1，每个字都要重新搬一遍几 GB 的权重，瓶颈只剩带宽。所以网页端该做的是把权重变小（量化、减层），而不是想办法凑批。</span>

## 边界

不要承诺「在标签页跑 70B」。不要在 iOS Safari 的不稳定 WebGPU 上无检测直跑。下一课：真正的手机 NPU 部署，权限与工具链又不同。

出处：W3C WebGPU；MLC WebLLM；ORT Web。

## 小结

- WebGPU 推理是沙箱着色器 + 下载配额；$B=1$、短上下文、强量化。
- 无 CUDA FA；长 $n$ 先炸内存再谈算法。
- 首次延迟含下载；要用分片缓存。
- 采样与 detokenize 在 Worker；注意稳定前缀。
- 产品动机常是隐私与零安装，不是吞吐。
- 下一课：移动端 NPU。
- 出处：W3C WebGPU；MLC WebLLM。

---
title: INT4 权重量化的工程实践
date: 2026-08-07
---

# INT4 权重量化的工程实践

<div class="epigraph">
<p>理论给出可能，工程给出可行。</p>
<footer>—— 工程实践共识（借自 Linus Torvalds 精神）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ vLLM / TensorRT-LLM / llama.cpp 工程文档 ｜ 2026-08-07</p>
</div>

## 为什么从 INT4 工程实践开始

前面 GPTQ、AWQ 讲了 INT4 量化的**算法**——怎么把权重压到 4 bit。但「算法能算出 INT4 权重」与「INT4 权重能在 GPU 上跑起来」之间，隔着一整层工程。4 bit 不是硬件的原生类型：GPU 的最小存储与计算单元是字节（8 bit），INT4 必须「打包」进 INT8/FP16 容器；反量化时机、内存布局、kernel 选择，处处是坑。<span class="marginnote">本专题从量化原理一路到这里，接下来这篇讲<strong>怎么把 INT4 模型真正部署</strong>：显存账、打包格式、kernel 路径与精度取舍，是对 GPTQ/AWQ 算法的工程落地。</span>

本篇讲 INT4 权重量化在主流引擎里的工程实践：显存与吞吐的真实收益、INT4 的打包与反量化时机、以及「为什么 INT4 常见于权重而非激活」。

## 1 INT4 能省多少：显存账

先算最实在的账：INT4 权重比 FP16 省多少显存。设模型参数量 $P$（十亿为单位），FP16 权重体积 $V_{16} = 2P$ GB，INT4 权重体积 $V_4 = 0.5P$ GB（4 bit = 0.5 字节/参数）。

一个 70B 模型：

FP16：140 GB；
INT8：70 GB；
INT4：35 GB。

**70B 模型 INT4 后，一张 80 GB 的 A100/H100 就能放下，还余下空间给 KV Cache。** 这正是 INT4 大行其道的第一驱动力。<span class="marginnote">对比 FP16 需要 2 张 80 GB 卡做张量并行才能装下——<strong>INT4 把「必须多卡」变成「单卡可跑」</strong>，直接改写部署架构的形态。</span>

但显存账要算全：INT4 省的是**权重**的存储，KV Cache 与激活仍在增长。长上下文场景下 KV Cache 很快追上权重体积，INT4 的省显存收益会被缓存吃掉一部分——这也是后面《KV Cache 量化》与《PD 分离》的动机。

这里也回答了另一个高频问题：为什么 INT4 用于**权重**而不用于**激活**？权重是一次性、离线的，量化慢一点、用复杂校准都无妨，且权重分布相对平坦；激活是每步在线计算，量化必须快，且激活分布有离群值（见 SmoothQuant 篇），INT4 的粗粒度会直接毁掉激活。所以「INT4 权重 + FP16/FP8 激活」是主流，「INT4 激活」很罕见。

## 2 INT4 的打包与内存布局

GPU 没有原生 INT4 类型，工程上把 4 个 INT4 值打包进 1 个 32 bit 整数（或 2 个进 1 字节）。打包的方式直接影响 kernel 访存效率：

- **连续打包（packed）**：一个通道组的 4 个 INT4 连续排布，适合「整组反量化」的 kernel；**group-wise 量化天然适配这种布局**——因为 scale 是 128 个权重一组共享，加载时整组取、整组反量化。
- **位宽交错**：有些格式把 INT4 对放在一个字节的高 4 位/低 4 位，配合专门的位操作指令（如 CUDA 的 `__byte_perm` / `prmt`）在寄存器里展开。

**反量化的时机（dequantization timing）**是工程决策的核心：

- **先反量化再 GEMM（dequant-to-FP16）**：kernel 加载 INT4 权重 → 展开成 FP16 → 用 FP16 GEMM。实现简单、兼容性最好，吞吐约等于「FP16 吞吐 × 访存节省」。vLLM 的 Marlin、TensorRT-LLM 的 INT4 路径都属于此类（Marlin 更激进）。
- **INT4 融合 GEMM**：在 GEMM 内部边反量化边乘，避免展开中间张量。更省内存带宽，但 kernel 复杂。

**辨析｜易错点：INT4 的「吞吐提升」不等于「算力提升」。** INT4 GEMM 通常仍是 FP16 算力（因为累加/乘法走 FP16），提升来自**访存带宽的节省**（权重字节减到 1/4）。当推理是 Memory-Bound（decode 阶段）时收益显著；当是 Compute-Bound（大 batch prefill）时收益有限。**所以 INT4 适合 decode 为主的在线服务，不是万能药**。

## 3 kernel 路径：Marlin、GPTQ、AWQ 与 GGUF

INT4 权重的加载路径在主流引擎里各有实现，性能差异明显：

| 引擎/格式 | 打包方式 | 反量化时机 | 特点 |
| --- | --- | --- | --- |
| vLLM + Marlin | 按 4 位组打包 | GEMM 内部分块反量化 | 高吞吐、需特定 shape 对齐 |
| vLLM + GPTQ/AWQ | group-wise 打包 | 反量化后 FP16 GEMM | 兼容性好、支持动态 shape |
| TensorRT-LLM | 自定义 INT4 路径 | 融合进选定的 tactic | 构建期自动选最优 kernel |
| llama.cpp GGUF Q4_K | 超组+子组两级 scale | CPU 反量化 | 见本专题《llama.cpp 量化方案》 |

Marlin 是 vLLM 社区的高性能 INT4 kernel：它利用 4 位打包 + 权重重排，让内存访问几乎连续、把 INT4 的带宽优势吃满。**选 kernel 的准则是「看它是访存优化还是兼容优先」**——访存优化型（Marlin）吞吐高但有约束，兼容型（通用 GPTQ/AWQ kernel）灵活但略慢。

**辨析｜易错点：INT4 不等于「精度同比减半」。** INT4 只有 16 个取值格点，远少于 INT8 的 256 个；对数值范围大、分布不均匀的权重，量化误差可能集中爆发。INT4 是否可用不能只看 PPL 涨了多少，还要看关键任务的掉点（见本专题《量化模型的精度评测方法》）。

## 4 数值算例：不同规模的显存账

把「INT4 省多少」落成一张覆盖常见规模的表。模型权重体积随精度与参数量变化：

| 模型规模 | FP16 权重 | INT8 权重 | INT4 权重 | 单卡可跑（80 GB）? |
| --- | --- | --- | --- | --- |
| 7B | 14 GB | 7 GB | 3.5 GB | 是，余量充足 |
| 13B | 26 GB | 13 GB | 6.5 GB | 是 |
| 70B | 140 GB | 70 GB | 35 GB | INT4 刚好，需留 KV 余量 |
| 405B | 810 GB | 405 GB | 202 GB | 否，需多卡 |

读这张表，两个直接结论：

- **INT4 把「多卡才能装」变成「单卡可跑」**：70B 从「2 张 80 GB 卡的 FP16 张量并行」变成「单卡 INT4 推理」——部署架构的形态被彻底改写。
- **容量账要加 KV**：70B INT4 权重 35 GB，4k 上下文的 KV 还要额外约 1–2 GB；若做 128k 长上下文，KV 会涨到几十 GB，INT4 省下的空间很快被吃掉——这正是 KV Cache 量化与 PD 分离存在的理由。

另一个常被忽略的账是「精度与吞吐的联合收益」：INT4 不只省显存，还因为权重字节降到 1/4，decode 阶段（Memory-Bound）的每 token 访存时间同比例下降，吞吐近似翻 2–3 倍（见下节公式）。部署 INT4 前，这四个工程提醒值得过一遍：

- **对齐限制**：Marlin 等高性能 kernel 对张量形状有对齐要求，shape 不对会回退到慢路径。
- **校准集影响**：INT4 的 scale 由校准集决定，校准集分布失配会放大精度损失。
- **Kernel 选择**：先反量化再 GEMM 与融合 GEMM 各有适用，以实测为准。
- **渐进式部署**：先 INT8 试水、再 INT4，用评测漏斗把关（见量化评测篇）。

## 5 公式解析：INT4 部署的吞吐收益

用 Roofline 模型算 INT4 的真实收益。设权重字节数 $W_{\text{bytes}}$，访存带宽 $B$，一次 decode 的权重访存时间 $T_w = W_{\text{bytes}}/B$；设单 token decode 的计算量为 $C_{\text{flops}}$，算力 $F$，计算时间 $T_c = C_{\text{flops}}/F$。

- **第一步，写 Roofline 条件**：decode 的 token 量小，$T_w > T_c$，即 Memory-Bound（见本专题《decode-memory-bound》）。总耗时 $T \approx T_w$。
- **第二步，代入位宽**：FP16 权重 $W_{16}$，INT4 权重 $W_4 = W_{16}/4$，故

$$T_4 \approx \frac{W_{16}/4}{B} = \frac{T_{16}}{4}$$

- **第三步，看理想与现实的差距**：理想情况下 INT4 decode 比 FP16 快 4 倍。现实中：反量化开销、kernel 效率、非权重访存（KV Cache）的存在，把实际加速比拉到 2–3 倍。**当 KV Cache 访存占比升高（长上下文）时，INT4 的收益被稀释**——因为省掉的只是权重那部分字节。

这个模型也解释了为什么「INT4 权重 + 长上下文」组合需要额外手段（KV Cache 量化）：**两个瓶颈串行，只砍掉一个，总收益有限**。

## 6 INT4 与 FP8 的选型对照

INT4 与 FP8 是当前部署圈最常纠结的两位，先看清它们的定位差异：

| 维度 | INT4 权重（W4A16） | FP8（W8A8） |
| --- | --- | --- |
| 权重体积 | 0.5 字节/参数 | 1 字节/参数 |
| 激活精度 | FP16（计算仍 FP16） | FP8（走 INT8/FP8 Tensor Core） |
| 主要收益 | 显存与 decode 带宽 | 全链路吞吐（prefill 也快） |
| 精度损失 | 中（需校准集） | 低（FP8 动态范围大） |
| 典型引擎 | vLLM Marlin、llama.cpp | TensorRT-LLM、vLLM FP8 |
| 适合场景 | 在线 decode、显存紧张 | prefill 密集、大批量 |

**选型的核心判据是「瓶颈在哪」**：显存快爆了选 INT4（省一半权重字节）；带宽是瓶颈且 prefill 占比高选 FP8（激活也压缩、全链路加速）。两者可以组合——业界已有「INT4 权重 + FP8 KV」的混血方案，各取所长。这与本专题《FP8 量化》一篇相互印证：FP8 是浮点域的新格式，INT4 是整数域的极致压缩，它们解决的是不同侧重的瓶颈。

## 7 小结

- **INT4 让 70B 模型单卡可跑**：权重从 140 GB 降到 35 GB，部署架构从「多卡 TP」变成「单卡推理」。
- **INT4 打包进原生类型**：4 个值包进 32 bit，group-wise 量化天然适配连续打包；反量化时机（先反量化 vs GEMM 内融合）是核心工程决策。
- **收益来自带宽而非算力**：decode 阶段 Memory-Bound 时收益明显，Compute-Bound 场景收益有限。
- **kernel 有路径之分**：Marlin 吃满带宽但有 shape 约束，通用 GPTQ/AWQ kernel 兼容优先。
- **长上下文会稀释 INT4 收益**：KV Cache 成为新瓶颈，需要配合 KV Cache 量化。

在下一节，我们顺着「长上下文稀释 INT4 收益」这个话题，专门讲缓存侧的量化——**KV Cache 量化的收益与精度损失**。

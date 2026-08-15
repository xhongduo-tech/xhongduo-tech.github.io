---
title: 量化感知与 TensorRT-LLM 的低精度支持
date: 2026-08-07
---

# 量化感知与 TensorRT-LLM 的低精度支持

<div class="epigraph">
<p>用最少的位，表达最多的信息——这是量化与信息论共同的信条。</p>
<footer>—— 克劳德·香农（Claude Shannon）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ NVIDIA TensorRT-LLM 文档（Quantization 章节） ｜ 2026-08-07</p>
</div>

## 为什么从量化感知开始

TensorRT 与 vLLM 的一个本质差异，是它对**精度**的态度：vLLM 把权重加载成什么精度，就按什么精度算；TensorRT-LLM 则把「低精度化」当作构建期的**一等公民**，从图构建到内核选择全程为 INT8/FP8/INT4 做准备。这篇讲 TensorRT-LLM 的**量化感知（quantization-aware）** 体系：它支持哪些量化格式、如何把量化参数（scale/zero-point）嵌进引擎、以及为什么「量化」在 TensorRT-LLM 里不是事后压缩而是先验设计。<span class="marginnote">本专题《量化》篇会逐个展开 GPTQ/AWQ/SmoothQuant 等<strong>具体算法</strong>；本篇聚焦 TensorRT-LLM 这个引擎<strong>怎么支持</strong>它们。</span>

## 1 TensorRT-LLM 支持的量化格式

TensorRT-LLM 把量化表达为「张量的数据格式（dtype）组合」，主流的几类：

**W4A16**：权重 INT4、激活 FP16。权重省显存、计算走 FP16 累加，是「显存不够」场景的主力。<span class="marginnote">命名规则 <code>WnAm</code>：W 是权重位宽，A 是激活位宽。<strong>W4A16 表示「4 位权重 + 16 位激活」</strong>，A 一般指激活的输入，累加器通常保持更高精度。
<strong>W8A8</strong>：权重、激活都 INT8。利用 INT8 Tensor Core，吞吐翻倍，是「算力瓶颈」场景的主力；需要 SmoothQuant 式校准。
<strong>W4A16/AWQ、W8A16</strong>：分别对应 AWQ 量化与权重量化 + FP16 激活，TensorRT-LLM 内置了对 AWQ 与 GPTQ 两种权重量化的加载支持。
<strong>FP8（W8A8 FP8）</strong>：Hopper 及之后架构上最流行的格式，权重与激活用 E4M3、累加用 FP32/FP16。见本专题《FP8 量化》一篇。</span>

**量化方式（quantization mode）**在引擎里是一个显式的配置，例如 `--use_fp8`、`--quantize_weights`、`--quantize_activations` 等命令行开关，或 `QuantConfig` 对象。引擎构建时按这些配置走不同的内核选择路径。

## 2 量化参数如何进引擎：Scale 与 per-channel/per-tensor

量化不是一个「把 FP16 权重四舍五入成 INT8」这么简单。关键是**缩放因子（scale）与零点（zero-point）**要跟着权重一起进引擎，推理时先反量化再计算（或融合进计算）。<span class="marginnote">本专题《量化的基本原理》会系统推导对称/非对称量化公式；这里只给结论：<strong>反量化 <code>w_deq = w_q * scale</code>，scale 决定量化步长，是误差的唯一来源。</strong></span>

TensorRT-LLM 对 scale 的粒度有严格区分：

**per-tensor**：整层权重共用一个 scale，最省存储、误差最大。
**per-channel（逐列）**：权重的每个输出通道一个 scale，是权重量化的默认选择；激活则常做 per-token（每个 token 一个 scale）。
**group-wise**：GPTQ 的 128 个权重一组共享 scale，误差进一步下降。

scale 作为**权重的一部分**被序列化进引擎文件，推理内核在 GEMM 前把 INT 权重按 scale 展开到累加精度。**scale 不参与训练，但必须在校准阶段精确保存**——scale 差一点点，整个量化的误差就会放大。

## 3 量化感知的引擎构建流程

TensorRT-LLM 的量化工作流是「**先校准，后构建**」：

1. **校准（calibration）**：拿一小批代表性数据（calibration set）跑一遍 FP16 模型，统计权重/激活的数值分布，据此算出 scale。对 GPTQ/AWQ 这类需要反向传播迭代的量化，还要在 GPU 上跑优化循环（见量化篇对应文章）。
2. **导出量化模型**：把 FP16 权重、scale、zero-point 一起导出成 TensorRT-LLM 的检查点格式，常见的是带 `safetensors`/`npz` 标记的权重文件（也可能来自 HF 的 GPTQ/AWQ 量化检查点）。
3. **构建引擎**：`trtllm-build` 读入量化模型，按 QuantConfig 走对应的量化 kernel 实现，产出 `.engine` 文件。

**辨析｜易错点：calibration set 决定量化成败。** 校准数据若与真实推理分布偏差太大（比如用英语数据校准、线上全是代码），算出的 scale 会在实际分布上产生巨大误差。**量化后的模型必须用「贴近线上」的校准集 + 全面的精度评测验证**（见量化篇《量化模型的精度评测方法》），否则「量化省了显存、精度崩了」会成为线上事故。

## 4 公式解析：INT8 Tensor Core 的吞吐增益

为什么 W8A8 能「吞吐翻倍」？用吞吐公式推导。设 GPU 的 FP16 峰值算力为 $C_{16}$，INT8 峰值算力约 $2C_{16}$（Tensor Core 每周期处理两倍数据）。一次 GEMM 的计算量为 $2 \cdot M \cdot N \cdot K$ 次乘加（$M\times K$ 乘 $K\times N$ 得 $M\times N$）。<span class="marginnote">$2MNK$ 这个因子来自「每个输出元素要累加 K 次乘法」，是矩阵乘的经典复杂度。见本专题《算术强度与 Roofline 模型》。</span>

- **第一步，算 FP16 耗时**：$T_{16} = 2MNK / C_{16}$。
- **第二步，算 INT8 耗时**：若计算完全走 INT8 路径，$T_8 = 2MNK / (2C_{16}) = T_{16}/2$——**同尺寸 GEMM 时间减半**。
- **第三步，看现实修正**：INT8 路径的累加精度、scale 反量化开销会让实际加速比在 1.5–2 之间；若访存（权重搬移）才是瓶颈，则 FP16→INT8 把权重字节减半，同样能带来约 2 倍的访存收益。**无论算力瓶颈还是带宽瓶颈，INT8 都有近 2 倍的理论收益**——这是它成为主流的原因。

W4A16 的路子不同：权重占显存减到 1/4，主要解决「放得下」；计算仍走 FP16，吞吐提升有限。**选哪种量化，取决于瓶颈是显存容量、带宽还是算力**。

## 5 数值算例：一次 W8A8 的完整构建

把「先校准后构建」落到一个具体例子上。设 7B 模型、FP16 权重 14 GB，目标 W8A8（INT8 权重 + INT8 激活）：

1. **校准**：拿 128 条贴近线上的提示词（客服对话）跑 FP16，统计每层权重与激活的数值分布，算出每层权重 per-channel 的 scale、激活 per-token 的 scale。**校准集越贴近线上，scale 越准**。
2. **导出**：把 FP16 权重按 scale 转成 INT8（14 GB → 7 GB），连同 scale 存成 TensorRT-LLM 检查点。
3. **构建**：`trtllm-build --quant_config=W8A8` 读入，走 INT8 内核路径，产出 `.engine`。
4. **评测**：对量化引擎跑精度评测（困惑度、下游任务），与 FP16 基线对比——**掉点可接受才上线**（见量化评测篇）。

**算收益**：权重 14 → 7 GB（省 7 GB 显存）；GEMM 走 INT8 Tensor Core，prefill 吞吐理论上翻倍；权重访存减半，decode 每 token 搬的字节减半。**一个 W8A8 同时改善「容量、算力、带宽」三个瓶颈**——这就是它成为通用默认档的原因。

**scale 在哪一层**：权重 per-channel（每输出通道一个 scale，共 $N_{\text{out}}$ 个）、激活 per-token（每 token 一个 scale）。两个 scale 都随引擎文件序列化，推理时从显存读取。

## 6 量化格式选型：一张对照表

不同业务诉求选不同格式，对照选择：

| 场景 | 推荐格式 | 理由 |
| --- | --- | --- |
| 显存不够装模型 | W4A16 / INT4 | 权重 1/4，模型装得下优先 |
| prefill 吞吐不够 | W8A8 / FP8 | INT8/FP8 Tensor Core 吞吐翻倍 |
| decode 访存太重 | W8A8（权重减半） | 每 token 搬的权重字节减半 |
| 新架构（Hopper+） | FP8 | 原生支持、精度好于 INT8 |
| 精度敏感、想保守 | W8A16 | 只量化权重，激活保持 FP16 |

**选型逻辑**：先想清楚瓶颈是「容量、算力还是带宽」，再选对应的格式——**这正好呼应本专题《GPU 推理的关键指标》的三个指标**。TensorRT-LLM 的价值在于：这些格式不是「论文里的想法」，而是「构建期开关 + 对应内核」，选完配置就能产出引擎。

**辨析｜易错点：别把所有层用同一个量化配置。** 有些层（如 attention 的投影层、最后的输出层）对精度特别敏感，全部 INT4 可能掉点明显。**TensorRT-LLM 支持「按层指定量化格式」（混合精度量化）**——敏感层用 W8A16、其余层用 W4A16，是「省显存」与「保精度」兼顾的做法。

量化构建常遇到的几个问题，速查：

| 现象 | 常见原因 | 对策 |
| --- | --- | --- |
| 量化后精度崩 | calibration set 偏差大 | 换贴近线上的校准集，重跑校准 |
| 构建报「unsupported」 | 该层没有对应量化内核 | 该层退回高精度（混合量化） |
| 推理比 FP16 还慢 | 量化 kernel 未生效 / 图未走 INT8 路径 | 检查构建日志确认走了量化路径 |
| 显存省了但吞吐没涨 | 瓶颈在别处（KV/带宽） | 结合 KV 量化与访存优化 |
| scale 精度不足 | per-tensor 粒度太粗 | 换 per-channel / group-wise |

**读这张表的要点**：量化问题一半出在「校准」、一半出在「没走对路径」——**排查先看构建日志里量化开关是否真的生效**，再看校准集质量。这两步对了，剩下多是参数微调。

**一个「量化不生效」的排查顺序**：构建日志是否出现量化 kernel → 运行日志是否走 INT8 路径 → 显存是否真的降下来 → 吞吐是否真的涨上去。四步逐项验证，**「显存降了但吞吐没涨」多半是内核没吃满，不是量化失败**。

TensorRT-LLM 的量化体系收尾一句：**它不是「一堆量化算法」，而是一个「精度档位 + 内核选择」的工程框架**。

算法（GPTQ/AWQ/SmoothQuant）由外部检查点提供，引擎负责「认出来、选对内核、跑起来」——理解了这个分工，就知道「量化」在 TensorRT-LLM 里为什么是构建期的一等公民，也知道了「换一种量化」只是「换一个检查点 + 改一个配置」。

## 7 小结

- **量化在 TensorRT-LLM 是一等公民**：`QuantConfig`/构建开关决定内核选择路径，量化不是事后压缩而是构建期设计。
- **格式体系**：W4A16、W8A8、FP8 覆盖「省显存」「提吞吐」「新架构」三种诉求；GPTQ/AWQ 等算法以检查点形式被引擎加载。
- **scale 随权重进引擎**：per-tensor / per-channel / group-wise 三种粒度权衡误差与存储，scale 的精度决定量化成败。
- **工作流是「先校准后构建」**：calibration set 必须贴近线上分布，量化模型上线前要跑完整精度评测。
- **INT8 的理论收益近 2 倍**：无论算力瓶颈还是带宽瓶颈，位宽减半都带来接近翻倍的收益。
- **选型看瓶颈**：容量不够选 W4A16、吞吐不够选 W8A8/FP8、精度敏感选 W8A16；可混合精度逐层指定。

在下一节，我们把整个构建流程串起来——**引擎构建、序列化与部署流程**，看 `.engine` 文件产出之后发生了什么。

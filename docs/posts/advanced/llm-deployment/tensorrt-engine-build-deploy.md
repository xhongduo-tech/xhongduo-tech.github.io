---
title: 引擎构建、序列化与部署流程
date: 2026-08-07
---

# 引擎构建、序列化与部署流程

<div class="epigraph">
<p>好的构建是安静的：一次构建，到处运行，多年不改。</p>
<footer>—— 软件工程箴言（借自 Java 精神）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ NVIDIA TensorRT-LLM 文档 ｜ 2026-08-07</p>
</div>

## 为什么从引擎构建开始

前面三篇讲了 TensorRT 的图优化、内核融合与量化支持——它们都发生在**构建期（build time）**。本篇把这些步骤收束成一条可执行的流水线：从原始权重出发，到产出一个 `.engine` 文件，再到把它部署成在线服务。理解这条流水线，才能明白为什么 TensorRT-LLM 部署的「启动快」与「构建慢」是同一枚硬币的两面，也才能回答「为什么改了模型结构就得重新构建」。<span class="marginnote">与 vLLM/SGLang 的「<strong>即插即用</strong>」路线对比：那两家从 PyTorch 权重直接跑，TensorRT 需要一次重量级编译。代价与收益见本专题《主流引擎对比》。</span>

## 1 从权重到引擎：构建的三阶段

TensorRT-LLM 的构建流水线可以拆成三个阶段：

**阶段一：准备模型定义与权重。** 来源通常是 Hugging Face 权重。TensorRT-LLM 提供了从 HF 转换的脚本（`convert_checkpoint.py`），把 HF 的 `safetensors`/`bin` 权重转成 TensorRT-LLM 检查点格式（`*.safetensors` + `config.json`）。转换时同时可以固化量化决定——例如直接产出 INT4 权重的检查点。<span class="marginnote">转换脚本会做<strong>权重布局重排</strong>：MoE 专家权重按专家排列、attention 权重按头重排，为后续内核的连续访存做准备。布局错误是引擎构建失败的高频原因。</span>

**阶段二：构建引擎。** 命令核心是 `trtllm-build`。这一步内部发生：图构建（network definition）→ 图优化（本专题图优化篇）→ 自动调优（跑 benchmark 为每个层挑最快 tactic）→ 序列化。构建引擎的耗时从几分钟到几十分钟不等，取决于模型大小与自动调优的搜索广度。

**阶段三：验证与部署。** 用 `trtllm-run`/`tensorrt_llm.LLM` 或自己写 C++/Python runtime 加载引擎，跑通自检（`--input_text` 参数跑一个示例输入），确认输出与 FP16 参考一致后，再接入服务框架（Triton、自研 HTTP server）。

## 2 引擎文件的格式与序列化

构建的产物是 `.engine` 文件——一个**序列化后的可执行计划**，包含：

优化后的计算图与每个层的选定 tactic（内核实现）；
权重张量（含量化后的 INT/FP8 权重与 scale）；
元数据：输入/输出的 shape、dtype、动态维度（如 `batch_size`、`input_len`、`output_len`）。

**引擎是可移植但绑定的**：引擎与**特定 GPU 架构（compute capability）**绑定。为 A100（sm_80）构建的引擎，无法在 H100（sm_90）上直接加载——这是部署时最常见的「引擎不兼容」错误来源。<span class="marginnote">此外引擎还绑定构建时使用的 TensorRT 版本。升级 TensorRT 通常必须<strong>重新构建</strong>引擎，不能只替换库文件。</span>

引擎文件里有一个容易被忽略的维度：**动态 shape 范围**（min/opt/max 三元组）。动态 batch、动态序列长度都需要在构建时声明范围，超出范围会在运行时被拒绝。**范围开大了，图优化会退化**（有些优化依赖精确 shape）；范围开小了，业务高峰期直接崩。这是构建参数里最需要反复权衡的一组。

## 3 部署流程：从引擎到在线服务

引擎构建完成后，部署层可以有几种形态：

**TensorRT-LLM 自带 runtime**：Python `LLM` 类或 C++ runtime 直接加载引擎、提供服务，支持 In-flight Batching、流式输出。
**Triton Inference Server**：NVIDIA 官方的服务化方案，TensorRT-LLM 后端（`triton_tensorrtllm`）把引擎包成可推理的模型，Triton 负责并发调度、动态批、健康检查与度量上报。<span class="marginnote">Triton 的 <code>dynamic batching</code> 与引擎内的 In-flight Batching 是<strong>两层调度</strong>：Triton 在请求层做攒批，引擎内部在 token 层做调度。分清这两层对排查性能问题很重要。
<strong>自定义服务</strong>：自己写 gRPC/HTTP 层，把引擎当黑盒调用，适用于深度定制场景（如与现有网关深度集成）。</span>

一个完整的生产部署还包括：模型仓库管理（版本化引擎文件）、GPU 分配、健康检查（引擎自检）、监控指标（吞吐、延迟、显存）与滚动更新（新引擎灰度替换旧引擎）。

**辨析｜易错点：引擎构建的自动调优（tactic 搜索）在 CI 里是昂贵的。** 每次代码或模型变更都触发完整自动调优会让 CI 慢得不可接受。工程实践是：**把 tactic 搜索的结果缓存**（`--timing_cache_file`、`--tactic_sources` 等开关复用历史最佳 tactic），或者用「快速构建 + 定期深度调优」的双轨策略——快速构建保证迭代速度，深度调优保证最终性能。

## 4 公式解析：构建时间 vs 运行时间的权衡

TensorRT 的全部取舍，可以凝练成一个「总拥有成本」式：

$$T_{\text{total}} = T_{\text{build}} + \sum_{\text{inferences}} T_{\text{infer}} = T_{\text{build}} + R \cdot t_{\text{infer}}$$

其中 $R$ 是总推理次数，$t_{\text{infer}}$ 是单次推理（含图优化与融合后的）延迟。

- **第一步，读出两端的杠杆**：$T_{\text{build}}$ 是一次性投入，$R \cdot t_{\text{infer}}$ 是持续成本。构建优化在「构建侧」多花的时间，能在「运行侧」省回来多少，取决于 $R$。
- **第二步，代入量级**：设构建从 10 分钟优化到 1 小时（多搜出 10% 的核选），换来单次推理延迟降低 10%。当 $R$ 巨大（在线服务，每秒几十次推理）时，这点 $t_{\text{infer}}$ 的降低乘以千万级 $R$，远超构建多花的 50 分钟——**深度调优的引擎在长期服务里几乎总是划算**。
- **第三步，看反例**：若 $R$ 很小（一次性离线跑批、实验验证），多花的构建时间毫无回报。**构建优化的深度应与使用频次匹配**——这也是「快速构建 + 定期深调」双轨策略的理论依据。

## 5 小结

- **构建流水线三段**：HF 权重转换 → `trtllm-build`（图优化 + 自动调优 + 序列化）→ 加载验证与部署。
- **`.engine` 文件是可执行计划**：含选定内核、量化权重与元数据；**绑定 GPU 架构与 TensorRT 版本**，跨架构需重建。
- **动态 shape 范围要在构建期声明**：范围过窄运行期崩溃，过宽使图优化退化。
- **部署形态**：自带 runtime / Triton / 自定义服务；Triton 的请求级批与引擎的 token 级调度是两层。
- **构建深度与使用频次匹配**：$T_{\text{build}} + R \cdot t_{\text{infer}}$ 模型说明长生命周期服务值得深度调优。

在下一节，我们开启**第六篇 量化**，从最基础的概念讲起——**量化的基本原理：对称与非对称量化**，为后面 GPTQ、AWQ、SmoothQuant 打地基。

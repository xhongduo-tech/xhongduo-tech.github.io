---
title: 推理算法与服务系统
description: 按生成正确性、缓存、数值和服务调度组织推理研究，逐项留下可检查的证据。
kind: overview
status: planned
---

# 推理算法与服务系统

::: info 研究地图 · 待实践
这里组织待完成的专题、阅读与实验，不代表已经掌握全部算法。CPU 机制验证、源码分析和真实硬件实测分别记录；目前没有本人的性能结果。
:::

推理优化先回答三个问题：输出是否正确，时间花在哪里，什么资源限制了可服务的请求量。一次前向更快、草稿接受更多、显存占用更低，分别是局部证据；它们需要放回完整请求中判断。

## 先分清问题的层次

| 层次 | 要解决的问题 | 本目录入口 |
| --- | --- | --- |
| 生成与验证 | 怎样减少目标模型的串行调用，同时保持规定的采样分布？ | [投机解码](/inference/speculative-decoding) |
| 训练与草稿 | 多个预测目标怎样训练，预测模块何时能够作为草稿？ | [MTP](/inference/mtp) |
| 验证预算 | 哪些草稿值得验证，验证长度怎样随置信度和负载调整？ | [DSpark](/inference/dspark) |
| 内存与复用 | KV 怎样存储、寻址、复用和回收？ | [KV Cache](/inference/kv-cache) |
| 数值与算子 | 降低位宽怎样影响误差、实际内存和内核执行？ | [量化](/inference/quantization) |
| 请求与资源 | 请求怎样进入 batch，排队、并行和通信怎样影响尾延迟？ | [服务系统](/inference/serving-systems) |
| 运行时实现 | 上述机制怎样在一个固定版本的引擎中衔接？ | [DeepInfer](/inference/deepinfer) |

几组常见名词处理的是不同问题：

- **PagedAttention**涉及分页 KV 的存储与访问；**FlashAttention**通过分块降低精确注意力的存储器 IO。它们不是同一种算法。[PagedAttention 论文](https://arxiv.org/abs/2309.06180v1)、[FlashAttention 论文](https://arxiv.org/abs/2205.14135v2)
- **continuous batching**调整请求在执行迭代间的加入与退出；**chunked prefill**切分输入处理；**PD 分离**把 prefill 与 decode 放到不同资源上。[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、[vLLM 调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)、[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- **prefix caching**复用相同有效前缀的计算；**投机解码**生成草稿并验证；**TP／DP／EP 等并行**分布计算与状态。组合是否有益取决于模型、内核和工作负载，不能把这些名词合并为一种“部署算法”。

## 两条可以逐步推进的路径

先具备 Python 数组操作、张量 shape、矩阵乘法、softmax、条件概率和最基本的计时能力；不会的部分按[基础地图](/foundations/)补齐，再执行一个小实验。

**生成路径：** 自回归与采样 → [投机解码](/inference/speculative-decoding) → [MTP](/inference/mtp) → [DSpark](/inference/dspark) → [DeepInfer](/inference/deepinfer)。先解释正确性，再研究调度与执行；MTP 是具体训练设计，DSpark 还有草稿模型和置信度估计等额外依赖。

**系统路径：** prefill／decode 与因果注意力 → [KV Cache](/inference/kv-cache) → [服务系统](/inference/serving-systems) → 选择[量化](/inference/quantization)或 DeepInfer 的一个模块深入。可先从熟悉的部署现象切入，不必等生成路径全部完成。

同一轮只选一个主问题。例如“分页是否改变计算结果”可以先做 CPU reference；“某量化格式能否降低并发服务成本”则需要支持该格式的真实内核与硬件。

## 统一的测量语言

把一次请求的端到端时间写成分析用的分解：

$$
T_{\mathrm{request}}
=T_{\mathrm{queue}}+T_{\mathrm{prefill}}
+T_{\mathrm{decode}}+T_{\mathrm{other}}.
$$

其中通信、缓存迁移、调度和同步可能彼此重叠，计入哪一项必须说明，不能把各设备耗时直接相加。进一步区分：

| 指标 | 回答的问题 | 必须同时说明 |
| --- | --- | --- |
| 正确性／质量 | 机制或数值变化是否破坏预期行为？ | 分布、容差、数据与评价方法 |
| TTFT | 从请求到达至首 token 要多久？ | 是否含排队、分词和网络 |
| TPOT／ITL | 后续 token 多快出现，是否有长停顿？ | token 数、流式事件粒度与分位数 |
| 吞吐／goodput | 单位时间完成多少有效工作？ | 负载、成功率、延迟目标与统计窗口 |
| 显存／缓存 | 什么状态占用空间，何时释放？ | 权重、KV、临时区、预留与实测峰值 |

[服务系统页](/inference/serving-systems)给出统一指标定义。论文或仓库报告的性能属于其特定配置，不能替代本人的同条件测量。

## 第一组低成本实验

计划先完成以下任意一项，再展开下一项：

1. 两 token 词表的投机采样：枚举概率质量，随后用固定随机种子的 Monte Carlo 检查频率。
2. 短序列 attention：比较每步全量重算、连续 KV 和分页 KV 的输出。
3. 合成请求队列：比较整批等待与迭代加入，保存每个请求的事件时间线。

这组实验只验证定义和性质。后续选择一个受支持的小模型、一种 GPU 与一个固定工作负载，先建立未优化基线，再只改变一个条件；预算不足时保留待实测状态。

## 每个专题应交付什么

- 一份能解释输入、输出、状态和适用条件的机制笔记。
- 固定论文版本、源码 commit、模型 revision 与环境记录。
- 可独立运行的小脚本、断言和失败用例；有随机过程时说明抽样误差。
- 原始数据、绘图脚本和结论边界；没有硬件测量时明确写“CPU 模拟”。

对应[推理算法与服务实验](/projects/serving-lab)。常见错误是只读概念、只贴仓库入口，或用 CPU 玩具的耗时推断 GPU 生产吞吐；验收应能由他人重新检查。

## 版本与来源约定

资料核验日期：**2026-11-16**。本目录以 V3 报告解释固定的 MTP 设计，以 DSpark v1 和 DeepInfer 固定快照展开后续问题；V3／R1不是“当前最新模型”的代称。[DeepSeek-V4.1-Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)已存在，后续架构的缓存和执行方式需要单独核对，不能直接套用 V3 的 MLA 或通用 MHA 估算。

论文版本见各页；源码链接固定 commit。标记为 latest 的文档是核验当日的滚动资料，实际实验仍须冻结安装版本。页面写的是待验证计划，阅读、代码分析、模拟、硬件复现四种进度不会互相替代。

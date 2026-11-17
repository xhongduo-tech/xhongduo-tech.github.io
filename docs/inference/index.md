---
title: 推理算法与服务系统
description: 从解码正确性到缓存、算子、并行和集群调度。
kind: overview
status: published
---

# 推理算法与服务系统

部署能力需要解释算法、模型结构与硬件之间的关系。这里按问题分层组织，代码仓库和产品名称作为交叉索引。

## 算法与系统地图

| 层次 | 核心问题 | 代表主题 |
| --- | --- | --- |
| 生成与验证 | 怎样减少昂贵目标模型的串行前向调用，同时保持正确的验证与采样？ | [投机解码](/inference/speculative-decoding)、[MTP](/inference/mtp)、[DSpark](/inference/dspark) |
| 注意力与上下文 | 怎样减少注意力的计算与访存，管理长上下文？ | Attention、MLA、SWA、FlashMLA |
| 缓存与内存 | 哪些计算可以复用，缓存何时保留或迁移？ | KV Cache、分页、prefix cache、offload |
| 算子与数值 | 哪些瓶颈在矩阵计算、数据移动或低精度？ | GEMM、DeepGEMM、量化、融合与数值误差 |
| 并行与通信 | 数据和专家如何分布，通信怎样影响吞吐？ | TP、DP、CP、EP、DeepEP、EPLB |
| 批处理与调度 | 请求异质、负载变化时怎样组织执行？ | continuous batching、chunked prefill、异步调度 |
| 服务架构 | prefill与decode如何分工，怎样联合分配资源？ | PD分离、负载均衡、缓存系统 |
| 接口与正确性 | 协议、模板、流式输出和确定性怎样衔接？ | structured output、deepseek-recipe |

## 第一条专题路径

先理解自回归与采样，再读[投机解码](/inference/speculative-decoding)，区分[MTP的训练与草稿作用](/inference/mtp)，随后研究[DSpark的草稿与验证调度](/inference/dspark)，最后追踪[DeepInfer中的系统实现](/inference/deepinfer)。

这些页面目前是专题提纲，不计入已完成博文。源码分析、CPU机制实验和真实硬件结果会分别标注。

## 深入的标准

每个主题留下机制解释、固定版本的源码入口、至少一个可以检查的性质或实验，以及适用边界。接受率、单用户速度、吞吐和服务延迟分别衡量；不能只看某个局部指标就宣称整体提速。

## 对应项目

[推理算法与服务实验](/projects/serving-lab)负责形成模型、工作负载和系统的实际证据。

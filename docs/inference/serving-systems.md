---
title: 批处理、并行与服务调度
description: 从请求时间线理解continuous batching、chunked prefill、PD分离和并行的不同职责。
kind: topic
status: planned
---

# 批处理、并行与服务调度

::: info 专题计划 · 待研究
先计划 CPU 离散事件模拟，再进行受支持引擎的负载实验。以下不包含本人的吞吐或尾延迟成绩。
:::

## 要解决的问题与前置依赖

模型单次执行很快，服务仍可能因排队、长短请求混合、缓存不足和通信等待而变慢。本页研究如何组织**请求、执行迭代和资源**；先定义延迟目标，再选择策略。

先理解自回归、prefill／decode、[KV Cache](/inference/kv-cache)，会记录事件时间和统计分位数。算法比较需要固定输入长度、输出长度与到达模式，不能只用一个短 prompt。

## 统一指标和测量边界

本页负载实验拟以客户端发送请求为 $t_0$，以接收首个、最后一个有效输出 token 为 $t_1,t_N$：

$$
\mathrm{TTFT}=t_1-t_0,\qquad
\mathrm{TPOT}=\frac{t_N-t_1}{N-1}\quad(N>1).
$$

单 token 输出的 TPOT 记为不适用。平均 TPOT 不等于每个 inter-token latency 的尾部表现；流式事件可能聚合多个 token，必须记录 tokenizer 和事件粒度，不能把 SSE chunk 直接当 token。服务端排队与设备时间单独打点，解释与客户端时间的关系。

在固定统计窗口内，可以把 goodput 定义为：

$$
G=\frac{\#\{\text{成功且同时满足 TTFT、TPOT 目标的请求}\}}
{\text{统计时长}}.
$$

这是本页采用的请求／秒口径；若报告 token goodput 须另写分子。[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)将两阶段延迟目标与有效服务能力共同考虑，而非只追求无约束吞吐。

## 几种机制解决不同问题

| 机制 | 调整的对象 | 代价和适用边界 |
| --- | --- | --- |
| continuous batching | 在执行迭代间加入新请求、移除已结束请求 | 必须维护变动的状态与缓存；不是等待整个 batch 全部结束 |
| chunked prefill | 把长输入的 prefill 拆成若干片段 | 调整 TTFT、decode 干扰和利用率；chunk 过小也有调度开销 |
| prefix caching | 命中有效前缀后减少重复 prefill | 收益取决于命中和状态保留，不等于批处理 |
| PD 分离 | prefill、decode 使用不同执行资源 | 可独立配置，但需要传输 KV、路由及负载平衡 |
| 投机解码 | 每轮提出并验证多个候选位置 | 改变执行批量与缓存回退，仍需纳入服务调度 |

迭代级调度的原始系统入口是 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)；chunked prefill 的具体实现与参数见 [vLLM 调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)；PD 的独立资源配置与通信约束见 [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)。PD 分离不是所有规模都更优，尤其要检查传输成本、流量结构和资源闲置。

## 并行不等于调度策略

| 方式 | 分布什么 | 需要观察什么 |
| --- | --- | --- |
| TP | 一个层或算子的参数／张量计算 | collective 通信、同步和每设备内存 |
| PP | 不同层或阶段 | 流水线空泡、阶段不平衡和跨阶段传输 |
| DP | 模型执行副本，各自接收请求 | 路由、负载和缓存命中的不均衡 |
| EP | MoE 专家及 token 到专家的分发 | token 路由、all-to-all、热点与专家副本 |
| CP | 上下文切片及相关 attention 计算 | 长上下文分布、通信和额外执行开销 |

TP／PP／DP／EP 的具体支持见 [vLLM 并行文档](https://docs.vllm.ai/en/latest/configuration/optimization/#parallelism-strategies)；DeepInfer 的 Attention DP／balanced CP、MoE EP／EPLB 是[特定模型与系统的组合](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md)，不能视为所有模型的部署模板。EPLB 是专家负载平衡机制，和入口请求路由也不是同一层。

## CPU 小实验：请求队列与执行时间线

计划写一个标准库离散事件模拟器，使用合成的长短输入、不同输出长度和突发到达：

1. 先用固定、公开的合成执行成本，比较整批执行与允许迭代加入两种策略；同一批输入保持不变。
2. 保存 arrival、queue、prefill、decode step、finish 事件，检查请求只完成一次、状态不丢失和资源上限。
3. 加入简化 token 预算与 prefill chunk，观察改变粒度怎样影响排队与交错；仅解释模型中的因果关系。
4. 输出每个请求的 TTFT、TPOT、完成时间及 p50／p95，记录失败或未完成请求；不只统计成功且快的样本。

合成成本不是 GPU profile，模拟里的吞吐数值不能当作真实引擎成绩。第一轮无需实现所有并行与 PD 机制。

## 后续 GPU 实测

先在单设备、单一模型和固定引擎版本建立稳定基线，然后只选 batching 或 chunk 大小做一个对照。分别使用低负载、饱和附近和突发请求；区分开放到达率与固定并发的闭环压力，两者不能互换。

保存模型 revision、硬件、内核、缓存设置、输入输出长度分布、到达率、预热、重复次数、错误率、TTFT／TPOT／ITL 分位数及 goodput。之后若有多设备预算，再研究并行或 PD；报告 KV 传输与设备利用率，不能省略其成本。

## 验收产物与常见错误

交付可复现的请求集、调度模拟器、事件时间线、指标脚本和结果模板；后续硬件报告附原始数据及无优化基线。

常见错误是把并发数等同于到达率、把 continuous batching 当作 PD、把 prefix 命中当作算子加速，或只测平均值却声称改善尾延迟。若输出质量、错误率或拒绝率变化，性能比较也必须披露。

## 一手资料与版本

核验日期：**2026-11-16**。

- [Orca，OSDI 2022 正式论文与报告](https://www.usenix.org/conference/osdi22/presentation/yu)：iteration-level scheduling。
- [DistServe，OSDI 2024 正式论文与报告](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)：PD 分离、资源与通信放置、goodput。
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)：滚动文档，核验当日的 chunked prefill 和并行策略说明；实际实验冻结安装版本。
- [DeepInfer 部署问答固定快照](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)：用于对照特定架构中的并行、缓存与负载平衡边界，不把其生产数据写成个人结果。

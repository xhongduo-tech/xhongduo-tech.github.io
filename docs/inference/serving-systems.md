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


## 迭代级加入到底改变了什么：一张手算时间线

先隔离批次组织这一因素：只模拟 decode，假设请求到达时所需 prompt KV 已就绪；最多同时执行两个请求，每步无论含一个还是两个请求都耗时 1 个单位，每个活跃请求每步产出一个 token。忽略网络、EOS 之外的截断与缓存压力。

A 在 $t=0$ 到达，需要 4 个输出 token；B 在 $t=1$ 的执行步边界到达，需要 1 个。若当前 batch 完成前不能加入新请求：

| 方案 | A 的输出时刻 | B 的首次／完成时刻 | B 的 TTFT | A 的完成时刻 |
| --- | --- | --- | ---: | ---: |
| A 完成后才接纳 B | 1、2、3、4 | 5 | $5-1=4$ | 4 |
| 每步边界可接纳 B | 1、2、3、4 | 2 | $2-1=1$ | 4 |

第二种方案在区间 $[1,2]$ 同时推进 A、B。收益来自等待时间减少，不是 B 的模型层或采样公式变了；它不需要把 A 整个请求暂停。这对应 [Orca 的 iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)所要解决的批次固定问题，但表中的成本模型是教学假设，不是 Orca 或 GPU 实测。

**反向改动同样重要：** 若两个请求同批的那一步实际要 5 个单位，则 B 在 $t=6$ 才得到输出，TTFT 变成 5；A 后续在 7、8 完成剩余 token，比原来更慢。是否值得合批要由真实的执行成本曲线判断，不能只由“batch 更大”推出。

### 从请求生命期看调度

<figure class="research-figure">

~~~mermaid
flowchart TD
    Q["排队"] --> P{"Prompt KV 就绪？"}
    P -->|"否"| C["推进 Prefill chunk"]
    C --> P
    P -->|"是"| B["步边界组 batch"]
    B --> E["执行／接收结果"]
    E --> F{"结束或取消？"}
    F -->|"否"| U["更新有效状态"]
    U --> B
    F -->|"是"| R["移除／释放引用"]
~~~

<figcaption>请求先满足数据就绪条件，再进入执行步；未结束的请求继续推进，结束后按生命周期释放引用。</figcaption>
</figure>

图只表达状态依赖，不是某个调度器的完整控制流，也不表示 prefill、CPU 和 GPU 必须全部串行。异步执行时，调度可以提前准备，但不能把尚未产生的数据当作已就绪输入；[KV 页的所有权与完成条件](/inference/kv-cache)继续成立。

## Chunked prefill：切的是输入工作，不是请求身份

设本轮总 query token 预算为 $M$，已安排的 decode 消耗为 $q$，则 prefill 可使用的预算最多为 $M-q$。它是调度容量约束，不是延迟公式：不同模型、长度和硬件下，相同 token 数可能有不同代价。

例如固定 $M=8$，每轮有 3 个普通单步 decode 请求，剩余可安排 5 个 prefill token。一个尚有 11 个输入 token 待计算的请求，若每轮条件不变，至少分 5、5、1 三次推进。这改善长输入与其他请求交错的粒度，但该请求的完整前缀必须按正确位置和 mask 建立后，才具备相应生成条件。

反例是始终让 $q=M$：在这个简单策略里 prefill 永远得不到预算。它暴露了准入、公平性或预算保留规则的必要性；不代表实际 vLLM 必然采用这一会饥饿的策略。

可沿固定版本 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L179)核对 token budget、请求进度和分配，再看[总调度预算断言](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/core/sched/scheduler.py#L545)。不要把“未超过预算”当作公平性、延迟目标或状态正确性已经证明。

## PD 分离的代价必须回到关键路径

PD 分离允许两阶段采用不同资源与并行方案，同时引入 KV 传输、路由和独立排队。若假设传输在关键路径上串行发生，最低数据传输时间是：

$$
T_{\mathrm{transfer}}\ge
\frac{M_{\mathrm{KV,transfer}}}{BW_{\mathrm{effective}}},
$$

还未计固定启动、同步和额外排队。一个纯教学数值是 64 MB 数据通过有效 16 GB/s 链路传输，使用十进制单位时下限为 4 ms。若分离只消除了 2 ms 的干扰，却没有把这 4 ms 隐藏到其他工作中，这个假设下就没有延迟收益。

真实系统可能通过流水线、缓存命中或不同资源配置改变关键路径，所以要记录**实际传输字节、是否重叠、两侧排队和利用率**，而非机械相加所有设备耗时。不同架构的 KV 大小也不同，不能沿用一个 MHA 估算替全部模型作决策。[DistServe 的阶段配置与带宽放置](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)是对应的一手入口。

## 从现象定位瓶颈，再选对照

| 观察到的限制 | 可能相关的机制 | 能否支持该解释的独立证据 |
| --- | --- | --- |
| 请求等待长，但设备存在可利用容量 | 迭代接纳、准入与批次组织 | 同到达序列下，排队减少且执行成本没有等量增加 |
| 长 prefill 使 decode 出现长停顿 | chunk 粒度或 PD 组织 | 同工作负载下，分别记录 prefill 进度、ITL 和额外调度／传输成本 |
| KV 不足导致无法接纳或重算 | 缓存布局、共享与回收 | 有效状态、占用、抢占和重算日志相互对应 |
| 已有多设备但计算在等通信／热点 | 并行划分与路由负载 | 计算／通信时间线和负载分布支持判断，不能仅看总 GPU 利用率 |
| 每轮产出低、目标执行串行性突出 | 投机及验证预算 | 草稿、接受长度、执行代价与缓存回退同时纳入 |

这些机制不互相代替验收。消融时只改一个主要因素，保留取消、超长、突发和资源不足等负对照；若策略改善均值却恶化尾延迟或拒绝更多请求，应如实收窄结论。

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

资料与新增源码入口核验日期：**2026-11-17**。时间线、预算与传输数值均为教学解析，不是个人实测。

- [Orca，OSDI 2022 正式论文与报告](https://www.usenix.org/conference/osdi22/presentation/yu)：iteration-level scheduling。
- [DistServe，OSDI 2024 正式论文与报告](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)：PD 分离、资源与通信放置、goodput。
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)：滚动文档，核验当日的 chunked prefill 和并行策略说明；实际实验冻结安装版本。
- [DeepInfer 部署问答固定快照](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)：用于对照特定架构中的并行、缓存与负载平衡边界，不把其生产数据写成个人结果。

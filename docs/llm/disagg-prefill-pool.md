---
title: 分离 Prefill 资源池
date: 2026-09-07
section: llm
---

# 分离 Prefill 资源池

<div class="epigraph">
<p>前填工人是可以独立扩缩的计算池，不是解码卡上的一种阶段标签；池的大小由 TTFT 预算、前缀命中与 KV 出口带宽共同决定，而不是由 GPU 总数均分。</p>
<footer>—— 对照 Zhong et al., DistServe, OSDI 2024；Patel et al., Splitwise, ISCA 2024；Qin et al., Mooncake, FAST 2025</footer>
</div>

把 prefill 与 decode 拆开之后，出现两个**池**，而不是两道工序。解码池按 KV 容量与 TBT 扩；前填池按提示到达率、平均前缀长度和命中率扩。Colocate 系统没有这个旋钮——加卡只能两阶段一起加。[PD 分离](/llm/pd-disaggregation) 写为何拆、[KV 传输](/llm/pd-kv-transfer) 写怎么搬。本篇写前填侧作为资源池的编制：xPyD 比例、异构硬件、突发阀、缓存命中如何改变「需要多少前填卡」。

## 问题

前填是计算密的，见 [Prefill 计算特征](/llm/prefill-compute)：提示长度 $s$ 上，GEMM 强度随 $s$ 升，注意力还有 $O(s^2)$ 项。到达过程是随机的，长文档与短聊天混在一起。若前填能力按「平均 $s$ × QPS」配置，长尾会把 TTFT 打爆；按 P99 配置，平时空转。解码侧却可能同时空着——它的墙是 HBM 与逐步读权重，吃不下前填的峰值 FLOPs。

第二变量是缓存。全局前缀命中（[Mooncake Store](/llm/mooncake-store)、[LMCache](/llm/lmcache)、引擎 radix）把有效前填量从「每个请求的 $s$」变成「未命中后缀」。池的需求随命中率一阶变化。第三变量是出口：前填实例必须把 KV 送给解码池，带宽不够时前填卡变成带 HBM 的队列，见 DistServe 的 pull 缓冲。于是「前填池」同时是计算器、缓存客户和 KV 生产者，只按 GPU 利用率扩缩会扩错。

### Goodput 约束下的 xPyD

记 $x$ 为前填工人数、$y$ 为解码工人数。DistServe 优化的是 TTFT 与 TPOT **同时**达标时的每 GPU 请求率。搜索先在单副本上为两阶段分别选卡数与并行度，再按流量复制。比例 $x:y$ 不是 1:1：解码算力利用率低时，多个前填对应一个解码，让解码侧堆起更大连续批。Mooncake 生产把两类机器编成分离集群；Dynamo 把同一思想写成运行时可改的 xPyD，Planner 可在突发时把工人从一池挪到另一池或临时聚合。Splitwise 再允许异构：前填池用高算力卡，解码池用更便宜或更低功耗的卡，并保留 mixed 池做突发阀。

<span class="marginnote">工人（worker/instance）是一份完整权重对应的资源，内部可以 TP/PP。说「四张前填卡」时要声明是四个实例还是一个 TP4 实例。池的货币是实例，不是卡数裸加。</span>

## 方法

编制从三条曲线出发。**计算**：未命中前填的 FLOPs 约 $\propto$ QPS $\times \bar{s}_{\text{miss}} \times d^2$ 再加注意力二次项。命中率 $h$ 把 $\bar{s}_{\text{miss}}$ 拉到 $(1-h)$ 量级（精确关系取决于命中的是整段还是最长前缀）。**延迟**：单实例 TTFT 还含排队；Erlang 式的直觉是利用率靠近 1 时排队发散，前填池要留余量，不能按均值 100% 规划。**出口**：KV 产率约 QPS $\times$ 每请求字节，须小于 P 侧到 D 侧的有效带宽；放不下就增加副本、就近放置，或降低未命中 QPS（更好的缓存），而不是只加前填卡——加卡会进一步提高 KV 产率。

并行策略按池独立。前填可用更大 TP 或序列并行换 TTFT；解码可用不同 TP/PP/复制换 TBT 与 KV 容量。Dynamo 博客的 DeepSeek-R1 例子：聚合时一套并行，拆分后前填 EP4DP16、解码 EP64DP3，比例与并行都变了。vLLM / SGLang / TRT-LLM 作为工人引擎时，池的控制面是 [Dynamo](/llm/nvidia-dynamo)、[SGLang Router](/llm/sglang-router)、[Ray Serve LLM](/llm/ray-serve-llm) 或 Mooncake Conductor，工人内部仍是连续批。

突发阀有三件套。DistServe 的 pull：解码按需取 KV，前填 HBM 当队列，峰值表现为 P 侧缓冲涨。Splitwise 的 mixed：允许部分机器两阶段都做，避免硬隔离下一侧空转。Dynamo Planner：监控队列与传输时间，决定临时聚合或挪卡。没有阀门的纯硬隔离，在前填突发时比 colocate 更脆。

```mermaid
flowchart LR
  ARR["到达 · 提示长度混合"] --> CACHE{"前缀命中"}
  CACHE -->|命中| SHORT["短前填 / 跳过"]
  CACHE -->|未命中| POOL["Prefill 池 x 实例"]
  SHORT --> POOL
  POOL --> KV["KV 出口带宽"]
  KV --> D["Decode 池 y 实例"]
  POOL -.->|"突发"| MIX["mixed / 临时聚合"]
  MIX --> D
```

### 缓存如何改写编制表

设无缓存时需要 $x_0$ 个前填实例才能稳住 TTFT。命中率 $h$、命中平均跳过长度占提示的比例为 $\alpha$ 时，计算量大约降到 $1-\alpha h$ 倍（数量级；精确要按长度分布积分）。Mooncake 在高重复对话上把有效容量抬到数倍，含义是同样 TTFT 预算下前填池可以更小，或同样的池可以吃更高 QPS。反过来，关闭全局缓存却仍按「有缓存」编制，池会在真实未命中率下过载。编制表必须把 $h$ 当输入，定期用线上命中重估 $x$，而不是一次性按模型 FLOPs 算完。

前缀感知路由（SGLang Router、Dynamo Smart Router、Ray 前缀路由）提高的是 $h$ 的实现值。池扩缩若只看 GPU 利用率，命中变好时利用率下降，自动缩容会把 $h$ 再打下去（缓存随实例消失）——经典的缓存与弹性互相拆台。正确信号是「未命中前填排队」和「KV 槽」，不是平均 SM 利用率。

## 机制

池化的机制是**把阶段耦合从时间轴挪到队列**。Colocate 时前填插入解码步，干扰发生在 SM 与 HBM 带宽上。分离后干扰变成前填队列长度和解码侧 KV 流入。前者用 $x$ 与 chunked prefill 控制；后者用 $y$、pull 与传输路径控制。chunked prefill 在分离后不再承担「保护同卡解码」的任务，但仍能限制单次作业长度，使前填池的迭代时间有上界，便于稳定 TTFT 排队模型。

异构池的机制是屋顶线匹配：前填买 FLOPs，解码买容量与功耗效率。Splitwise 报告在当时 colocate 对照上约 1.4× 吞吐并降约 20% 成本，或同功耗成本预算下约 2.35× 吞吐。DistServe 报告 goodput 相对当时 SOTA 最高约 7.4× 请求率或约 12.6× 更紧 SLO。Mooncake 报告 SLO 内有效容量 +59%–498%。这些数字不可比成一张表，因为指标（goodput / 成本 / 有效容量）与是否含全局 KV 池不同；它们共同证明：**前填作为独立池是一等编制对象**。

<span class="marginnote">TTFT 在分离后包含排队、前填计算、KV 传输。把传输算进 TBT，会把本该加在互联或放置上的预算错误地加到解码池。</span>

### 与 colocate 优化共存

连续批、分页、GQA/MLA、投机解码在分离后仍然要。投机打在解码池；前填池一般不跑草稿树。MoE 上两池可取不同 EP 度。前填池的专家缓存命中往往低于稳态解码，TTFT 可能被冷专家搬运主导——编制时这是另一条与稠密 FLOPs 不同的曲线。不要假设「拆开之后前填池只是若干张算力卡」。

## 边界与工程取舍

权重要复制至少两份，这是池化的固定税。流量极低、提示极短时，税超过干扰收益，应保持聚合或 mixed。xPyD 在线重配要迁移或丢弃进行中的 KV，频率过高会抖 TTFT。自动扩缩必须钉住缓存亲和：缩掉持有热前缀的前填实例，等于主动降低 $h$。

观测要分池打点：前填排队、未命中 $s$ 分布、KV 出口利用率、解码 KV 占用、各自并行度。一张「集群 GPU 利用率」图无法判断该加 $x$ 还是加 $y$。控制面选 Dynamo、Router 还是 Conductor 可以不同，编制账必须同一套公式。

不要把「分离 Prefill 资源池」写成某一家产品的别名。它是 DistServe / Splitwise / Mooncake / Dynamo 共用的运维对象：一组专门做提示前填、可独立扩缩、以 KV 为出口的实例。实现可以是物理集群分区，也可以是同一调度器下的两个 Deployment。

<span class="marginnote">出处钉 Zhong et al., *DistServe*, OSDI 2024；Patel et al., *Splitwise*, ISCA 2024（arXiv:2311.18677）；Qin et al., *Mooncake*, FAST 2025（arXiv:2407.00079）；NVIDIA Dynamo 文档中的 runtime-reconfigurable xPyD。传输公式见 PD KV 传输篇。</span>

## 小结

- 前填池按未命中计算、TTFT 排队和 KV 出口带宽编制，与解码池独立扩缩。
- $x:y$ 来自双 SLO 搜索或在线 Planner，不是卡数均分；异构与 mixed 是合法阀门。
- 缓存命中一阶降低前填需求；扩缩信号要用未命中排队，避免弹性拆掉热前缀。
- 突发时用 pull 缓冲、mixed 或临时聚合，硬隔离会在峰值上更脆。
- 论文数字分别钉 goodput、成本与有效容量，证明池化本身，不互相连乘。
- 出处：DistServe、Splitwise、Mooncake、Dynamo xPyD。

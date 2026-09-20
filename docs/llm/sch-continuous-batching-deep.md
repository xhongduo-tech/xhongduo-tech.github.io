---
title: 连续批处理深入
date: 2026-09-18
section: llm
---

# 连续批处理深入

<div class="epigraph">
<p>成员可变只是解除了批的寿命约束；每一次迭代怎么组——塞多少、先装谁、一步多长——才决定 ITL 的分布。</p>
<footer>—— 据 Yu et al., OSDI 2022 与 Agrawal et al., Sarathi-Serve, OSDI 2024 的调度口径整理</footer>
</div>

[上一课](/llm/sch-basics-metrics)把合同钉在双分位数与 goodput 上，本课开始填机制：批怎么组。主干已写连续批处理的定义——成员在迭代边界进出（[连续批处理](/llm/continuous-batching)、[Orca 迭代级调度论文](/llm/orca-iteration)）；本课深入定义之外的三件事：一次迭代的 token 预算、prefill 与 decode 的混批次序、以及步时延为什么必须有上界。投机解码与 MoE 改变「一步的工作量」，也一并入账。后课的抢占、PD 分离都默认你已会做「每步组批」这笔账。

## 问题

「成员可变」只回答了谁在场，没回答本步做什么。一条 8k 提示的前填若整步执行，批内所有 decode 请求的 ITL 直接多出这一次前填的墙钟；新请求的 TTFT 也压在队头提示的长度上。反过来，为了护住 decode 把 prefill 全推后，新请求的 TTFT 无界，算力在大段空闲。两个方向都错，错的根源相同：**步时延没有上界**。组批问题于是可以精确表述：给定合同给 ITL 留下的步时延预算，每一步在预算内装下尽可能多的 token，并决定 prefill 与 decode 的次序。投机解码再添一层不确定：接受数是随机的，同一步的工作量按请求波动。

### token 预算与装填次序

工程解是本步 token 预算（`max_num_batched_tokens` 一类旋钮，[vLLM 调度器](/llm/vllm-scheduler) 已点过它的角色）：decode 优先装满——每条运行中请求至多一个 token，代价几乎只是 KV 读取；剩余预算给 prefill，长提示切成不超过余量的块跨多步执行（[chunked prefill](/llm/chunked-prefill)）。次序背后的账是两条 roofline：decode 是带宽活，prefill 是算力活（[Prefill 计算特征](/llm/prefill-compute)）；decode 先装保证 ITL 稳，prefill 用余量填算力空隙。步时延上界于是由「decode 步加一个 prefill 块」决定，是常数而不是最长提示的函数。

```mermaid
flowchart TD
  STEP["新迭代开始"] --> DEC["装 decode：每请求 1 token"]
  DEC --> BUD{"剩余 token 预算"}
  BUD -->|有| PRE["装 prefill 块：切到预算内"]
  BUD -->|无| RUNB["引擎：一次前向"]
  PRE --> RUNB
  RUNB --> ITL{"步时延在预算内"}
  ITL -->|是| NEXT["记账，进下一迭代"]
  ITL -->|否| TUNE["缩小块或预算"]
  TUNE --> NEXT
```

## 机制

吞吐的来源在算术强度：单条 decode 每读一遍权重只产一个 token，批内并发是唯一摊薄权重读取的手段（[解码的算术强度](/llm/arithmetic-intensity-decode)、[Decode 的显存墙](/llm/decode-memory-wall)）；所以 ITL 的地板是权重读取时间，批加大先摊薄地板、再撞算力墙，拐点见 [批大小与 roofline 拐点](/llm/batch-roofline-knee)。组批的艺术全在这个拐点之前，翻译成预算语言：预算给小了，decode 摊不薄权重，吞吐亏；给大了，步时延超预算，ITL 合同破。投机解码让「decode 先装」的代价不再恒定：接受 $k$ 个 token 的请求本步多算约 $k$ 倍，接受率高的批步时延更长、但产 token 更多——ITL 与吞吐要按接受长度联评（[投机解码原理](/llm/speculative-decoding)）。MoE 再叠一层：步时延取决于激活了哪些专家，批的构成影响负载，见 [MoE 推理批处理](/llm/moe-inference-batching)。

<span class="marginnote">ITL 有个不可再压的地板：decode 步至少要读一遍权重。13B 模型 fp16 权重约 26 GB，在 3.3 TB/s 的 HBM 上光权重读取约 8 ms——任何「每步 2 ms」的承诺都物理不可行。步时延预算要从这条地板往上留，而不是从零往下压。</span>

## 边界

预算调优是 TTFT 与 ITL 的换汇：预算大，前填走得快（TTFT 好）但步时延尖（ITL 差）。换汇点没有普适值，要从合同反推，且随负载漂移，要看分位数而不是均值。齐整离线负载（同长度分类、embedding）上静态批更省心，连续批的调度税没有对应收益。CUDA Graph 按（批大小，token 预算）桶捕获，桶太细内存爆、太粗每步多算 padding——桶距是又一个工程参数。本课的默认场景是单池混批；把 prefill 与 decode 拆成两池是另一条路，[PD 分离](/llm/pd-disaggregation) 之后课展开。

## 小结

- 连续批处理只定义成员可变；每步的 token 预算与装填次序才决定延迟分布。
- decode 先装、prefill 用余量切块执行，步时延上界是「decode 步加一块」。
- ITL 的地板是权重读取时间；批大小在该地板与算力墙之间调。
- 投机解码与 MoE 让「一步的工作量」随机化，预算要按最坏步留。
- 预算是 TTFT 与 ITL 的换汇点，按分位数调，不按均值。
- 出处：Yu et al., OSDI 2022；Agrawal et al., Sarathi-Serve, OSDI 2024；预算旋钮的工程口径对照 Kwon et al., SOSP 2023。

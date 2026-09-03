---
title: 流水线并行推理
date: 2026-09-03
section: llm
---

# 流水线并行推理

<div class="epigraph">
    <p>训练用微批填满管道；decode 一步一个 token、管道里没有微批可填时，气泡接近阶段数减一，流水线从吞吐工具变成尾延迟税。</p>
    <footer>—— 对照 Huang 等 GPipe、Megatron 1F1B，以及 DistServe 对 inter-op 并行与速率扩容的讨论</footer>
</div>

[流水线并行](/llm/pipeline-parallel) 按深度切层，阶段之间点对点传激活。训练侧用 $M$ 个微批压气泡，1F1B 再压激活内存。推理没有反向，看似更简单，却少了「用微批填管道」这个前提。Prefill 一个请求内部有长序列，或多请求并发时，管道里可以同时有多个前向；decode 每个请求每步只推一个 token，若没有足够的连续批，阶段 $2..P$ 都在等阶段 1，利用率 $1/P$。DistServe 把 inter-operator 并行写成：适度增加执行时间，但随 GPU 数近线性扩大速率容量。本篇写推理期 PP 何时有用、气泡从哪来、以及与 [PD 分离](/llm/pd-disaggregation) 之后的阶段划分如何咬合。

## 问题

权重大到 TP 组已经占满节点内 NVLink，还是放不下，就得沿深度再切。PP 的通信是边界激活，体积 $b\times s\times d$，次数少，跨节点 InfiniBand 比层内 All-Reduce 更扛得住。这在训练里成立，在 prefill 里也往往成立。Decode 把 $s$ 变成 1，$b$ 若也小，通信延迟相对计算更大，更致命的是调度：阶段必须等上一阶段的该微批（此处即该 decode 步）完成。单请求、同步流水线的气泡比例约 $(P-1)/1$，几乎不可用。

Colocate 服务还要把 prefill 与 decode 塞进同一条管道，长 prefill 占满阶段时，decode 步被堵住，TPOT 炸裂；chunked-prefill 只能缓和，不能取消依赖。问题是：PP 作为「扩容轴」与作为「单请求加速轴」在推理里必须拆开讲。

### 气泡在推理里长什么样

训练气泡来自预热与冲刷，稳定段 $M\gg P$ 时可忽略。推理 prefill：若同时有 $R$ 个独立请求（或把一条长提示切成可流水的块），$R$ 扮演 $M$。$R\ge P$ 时管道可满，TTFT 的排队与执行要分开算——后进请求要等气泡填满。推理 decode：连续批处理把许多请求的一步拼成一个 iteration。Iteration 级调度下，一个 decode batch 流过 $P$ 段，只要 batch 里请求数稳定，管道可以保持满；新请求加入、旧请求结束造成 batch 形状抖动，气泡再现。单请求流式对话、$P=8$，几乎是最坏情况。

<span class="marginnote">微批在推理里常常不叫微批，而叫「连续批里的请求数」或「chunked prefill 的块」。会计相同：管道深度 $P$ 需要足够多的独立前向填充。名称变了，公式没变。</span>

## 方法

Prefill 实例：层按 FLOPs 均分到 $P$ 段（输出头往往更重，不要按层数均分）。多请求或 chunk 作为填充物。紧 TTFT 时 DistServe 更倾向 intra-op（[推理 TP](/llm/infer-tp)）来砍执行时间；TTFT 宽松、要拉高每卡 goodput 时，inter-op 用更多卡换速率，单请求执行时间只温和上升。Decode 实例：优先用连续批把 $b$ 做大，PP 用来在 batch 已大、还要加卡时扩吞吐；TPOT SLO 很紧时仍要 intra-op 降步延迟，而不是盲目加 $P$。

### DistServe 的 inter-op 与同节点分段

DistServe 还有一层与 [KV 传输](/llm/pd-kv-transfer) 有关的用法：按层分段后，把 prefill 与 decode **同一逻辑阶段**放进同一节点，迫使 KV 走 NVLink 而不是跨机。此时 $P$ 不只是吞吐旋钮，还是放置约束：阶段数、每节点 GPU 数、P/D 实例的段要对齐。低跨节点带宽集群上，这个动机可能强过「用 PP 扩容」。高 InfiniBand 集群上，KV 传输不是墙，PP 回归普通的速率轴。

```mermaid
flowchart LR
  Pref["Prefill 请求"] --> S1["阶段 1"]
  S1 --> S2["阶段 2"]
  S2 --> SP["阶段 P + 首 token"]
  SP --> KV["KV 交给 decode 实例"]
  KV --> D1["Decode 阶段 1"]
  D1 --> D2["Decode 阶段 2"]
  D2 --> DP["阶段 P 逐步吐 token"]
```

### 与 chunked-prefill、1F1B 的关系

Chunked-prefill 是 colocate 下把长提示切开、与 decode 拼批，减轻干扰。它不消除干扰，还让 prefill 重复读 KV。PD 分离之后，prefill 实例内部仍可用切块来限制单次激活内存，但不再需要为了 decode 而 piggyback。训练的 1F1B 在推理里没有反向可交替；对应的是「一个 iteration 推完所有阶段再进入下一 iteration」，或异步把阶段重叠。异步重叠会引入乱序与更复杂的 KV 版本，在线延迟服务通常保持同步 iteration，用 batch 填气泡，而不是用异步乱序填。

## 机制

PP 减的是每卡权重 $\Phi/P$，通信是点对点激活而不是每层 All-Reduce。Prefill 执行时间随 $P$ 温和增加（气泡与边界通信），吞吐随实例内 GPU 数近线性——DistServe 对 decode 在大 batch 下也给出类似图像：inter-op 扩吞吐，intra-op 降延迟、收益递减。单请求延迟的下界仍包含流水线深度：第一 token 必须流过 $P$ 段。因此 PP 几乎从不单独用来优化「一个用户的 TTFT」，除非同时用 TP 砍每段时间。

负载不均在推理里同样无法 All-Reduce 掉：最慢阶段决定 iteration 时间。MoE 层、超宽 FFN、LM 头应摊开。Decode 步时间短，阶段不均比训练更刺眼。数值上 PP 不改变层内求和顺序，比改 TP 更接近可复现；但连续批的组成变化会改算子形状，延迟曲线仍会抖。

<span class="marginnote">「线性扩吞吐」的前提是管道已满且没有 KV 传输墙。空管道时加 $P$ 是线性加气泡。测 PP 收益必须报当时的 batch 与填充率，否则数字没有意义。</span>

### PD 分离后两条管道

分离后 prefill 管道与 decode 管道可以不同 $P$、不同段映射。KV 必须按层对齐传输：P 侧阶段 $i$ 的缓存送到 D 侧阶段 $i$。若两边 $P$ 不同，要有重切分或在传输层做层到设备的映射，否则缓存会对错卡。这是推理 PP 独有的约束，训练没有「另一条管道要接住 KV」。

## 边界与工程取舍

单请求、低并发的交互式服务，PP 不是默认项；复制或小 $T$ 更干净。离线批处理、长 prefill、大连续批的 decode 池，PP 才像训练时那样划算。$P$ 受层数整除约束；余下层归属要写死。检查点与导出仍按阶段分片，扩缩 $P$ 要重切。

不要把训练 $P$ 搬到在线 decode。也不要把 pipeline bubble 和 PD 干扰当成一件事：前者是阶段依赖，后者是 prefill/decode 抢同一 GPU。对策分别是填管道与拆池子。引用 GPipe / Megatron 说明气泡公式；引用 DistServe（arXiv:2401.09670）说明推理 inter-op 的速率含义与放置。Splitwise 的异构池会进一步让 P 侧与 D 侧选不同的卡型，PP 深度也可以不同。

<span class="marginnote">流式输出时用户按 token 感知延迟。PP 把「一步」定义成「所有阶段跑完」。若实现成阶段 1 先出部分结果，那是另一套异步语义，不再是标准同步流水线，延迟统计也要重做。</span>

## 小结

- 推理 PP 仍按深度切层；prefill 可用多请求或切块填管道，decode 必须靠连续批，否则气泡约 $P-1$。
- Inter-op 擅长扩吞吐，不擅长砍单请求执行时间；紧 SLO 时先考虑 TP。
- PD 分离后两条管道的 $P$ 可不同，但 KV 要按层映射；同阶段共节点可把传输限制在 NVLink。
- Chunked-prefill 是 colocate 的干扰缓解，不是 PP 的替代品。
- 单请求交互式 decode 默认不该上大 $P$。
- 出处：Huang 等 GPipe；Megatron 流水线调度；Zhong et al., DistServe，arXiv:2401.09670。Splitwise（Patel et al., arXiv:2311.18677）给出异构池上阶段分离的对照。

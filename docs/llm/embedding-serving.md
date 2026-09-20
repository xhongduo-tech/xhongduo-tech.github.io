---
title: 嵌入与重排模型服务
date: 2026-09-08
section: llm
---

# 嵌入与重排模型服务

<div class="epigraph">
<p>双编码器一次前向吐向量，没有逐步 decode，也没有 KV 墙；批处理回到普通 BERT 式 GEMM，调度却仍常被误套成自回归引擎。</p>
<footer>—— Reimers & Gurevych, Sentence-BERT, EMNLP 2019；重排器是交叉编码器，计算随候选对数涨</footer>
</div>

[上一课](/llm/vision-encoder-pipeline)仍接到自回归 LLM。检索栈里还有嵌入模型与重排模型：Sentence-BERT 把句子映到向量，用余弦召回；交叉编码器把 (查询, 文档) 一起编码打分。本课写服务画像差异：无 KV 缓存、无 TPOT、吞吐随 batch 与序列长度走普通屋顶线。用 vLLM 的连续批去套嵌入，会买一套用不上的分页 KV。Hugging Face TEI 一类专用服务才对症。

## 问题

嵌入请求是一次 prefill 式前向，输出 $d$ 维向量，可大批、可padding。重排是对每个候选一次交叉编码，$k$ 候选就是 $k$ 次（或拼 batch）。缺口是不要用 decode 会计：这里没有 $\mathrm{KV}(n)$ 随生成增长，有的是短序列的高 QPS。延迟 SLA 是 p99 毫秒级向量，不是 TPOT。把嵌入与 LLM 放同一连续批，形状与掩码都不同，核融合也不同。

<span class="marginnote">直觉类比：LLM 服务要为每个请求留一块随生成不断变长的「草稿纸」（KV 缓存）；嵌入请求一次前向就交卷，根本不用草稿纸。为嵌入建 KV 池，等于给交卷即走的考生留永久座位，钱花在没人坐的地方。</span>

动态 padding：批内按最长序列 pad，过长尾巴浪费。应按长度分桶，与训练不同，这是服务分桶。

<span class="marginnote">重排器的 FLOPs 随候选数线性，召回 $k=200$ 再重排 50 是典型。成本模型应写在检索路径上，不要记进「每个用户问题一个 LLM token」。RAG 的隐藏账单往往在这里。</span>

## 方法

嵌入：静态或半静态形状，CUDA Graph 友好，`torch.compile` 收益比 LLM decode 更干净。大 batch 直到拐点。多卡用数据并行复制模型，不必 TP——模型往往放得进单卡。重排：候选 batch 维，注意注意力掩码。与 LLM 协同：异步队列，不要在 LLM 的 GPU 上同步插一条嵌入前向（打乱 decode 池工作点）。

向量维与归一化是契约（是否 L2），服务端必须与训练一致，否则检索静默变差。

<span class="marginnote">为什么这一步不能错：训练时做了 L2 归一化、服务时忘了做，点积就会带上长度偏差——长文档的向量普遍更长，会系统性地排到前面。整个检索质量悄悄下滑，日志里却没有任何报错。</span>

```mermaid
flowchart TD
  Q["查询"] --> EMB["双编码器 batch"]
  DOC["文档库"] --> ANN["ANN 召回"]
  EMB --> ANN
  ANN --> RER["交叉编码器重排"]
  RER --> LLM["可选: 生成"]
```

## 机制

算术强度随 $B$ 与 $n$ 升，容易 compute-bound，MFU 可比 LLM decode 高一个数量级。这解释了为什么同一张卡跑嵌入「看起来利用率很好」、跑聊天 decode「利用率很差」——不是嵌入实现更强，是工作点不同。成本 $C$ 按请求或按 token 计都可以，但不要用 LLM 的 $C_{\mathrm{tok}}$ 乘嵌入 token。

<span class="marginnote">数字实例：7B 模型 BF16 权重约 14 GB，聊天 decode 每生成一个 token 都要把整份权重读一遍，算术强度只有每字节约两次运算；而嵌入一批几百条短句走稠密 GEMM，权重读一次摊给大量计算，算术强度高一个数量级。所以两种「利用率」根本不是一回事。</span>

```mermaid
flowchart TD
  SUB["三类请求进站"] --> LLM["LLM decode: KV 随步增长, 看 TPOT"]
  SUB --> EMB["嵌入: 一次前向出向量, 高 QPS"]
  SUB --> RE["重排: 每候选一次交叉编码, 成本随 k 线性"]
  LLM --> POOL1["LLM 池: 连续批 + 分页 KV"]
  EMB --> POOL2["嵌入池: 长度分桶 + CUDA Graph"]
  RE --> POOL2
  POOL1 --> MIX["分池调度, 互不打乱对方工作点"]
  POOL2 --> MIX
```

## 边界

不要为嵌入建 KV 池。不要把交叉编码器当双编码器用（QPS 会垮）。下一课回到 LLM：权重如何加载与流式进 GPU，这才是自回归服务启动的账单。

出处：Reimers & Gurevych, EMNLP 2019。TEI 等为工程实现。

## 小结

- 嵌入无 decode KV；服务画像是短序列高 QPS GEMM。
- 重排成本随候选数线性，是 RAG 隐藏账单。
- 长度分桶；CUDA Graph / compile 比 LLM decode 更适用。
- 与 LLM 分池，避免打乱 $B^\star$。
- 归一化与维是检索契约。
- 下一课：权重加载与流式。
- 出处：Reimers & Gurevych, 2019。

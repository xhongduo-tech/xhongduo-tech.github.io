---
title: In-flight Batching 的原理
date: 2026-08-07
---

# In-flight Batching 的原理

<div class="epigraph">
<p>让 GPU 永远有事做，是推理服务的第一要义。</p>
<footer>—— 工程箴言，源自 NVIDIA Triton / TensorRT-LLM 团队实践</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ NVIDIA TensorRT-LLM 文档 ｜ 2026-08-07</p>
</div>

## 为什么从 In-flight Batching 开始

静态批处理（static batching）时代，推理服务把请求攒够一批（比如 32 个）再一起前向，这一批里所有请求必须**同步**完成。问题立刻来了：LLM 生成的序列长度千差万别，一个 10-token 的短回答要等一个 1000-token 的长回答走完才能结束——短回答的算力全浪费在「陪跑」上，批内的无效填充（padding）暴涨。本专题《Continuous Batching》已经用 vLLM 讲过解法：**批的成员随时可变**。<span class="marginnote">In-flight Batching 是 TensorRT-LLM 与 NVIDIA Triton 对同一思想的命名，<strong>与 vLLM 的 Continuous Batching 是同一个概念的两种工程实现</strong>。读本篇时记得对照那篇。</span>

本篇讲 In-flight Batching 的机制：如何在**解码过程中**动态增删请求、如何组织请求状态、如何用「调度」而非「等待」来填满 GPU。它是 TensorRT-LLM 能在在线服务里压出高吞吐的支柱。

## 1 静态批处理的浪费在哪里

先量化「陪跑」的浪费。假设一批 8 个请求，每个都解码 100 步，但其中 2 个在第 20 步就结束。静态批处理必须等满 100 步，因为矩阵运算是**矩形**的——batch 维度固定为 8，已结束的请求也要占着矩阵的行，用 padding token 填满。

<div class="marginnote">填充带来的浪费分两层：<strong>显存浪费</strong>（padding 也占 KV Cache 的槽位）与<strong>算力浪费</strong>（padding 也参与矩阵乘法）。短请求越多、长短差距越大，浪费越严重。</div>

用数字说话：批大小 $B=8$，平均有效长度 $\bar{L}=50$，最长的完成长度 $L_{\max}=100$。整个批的「有效计算量」与「实际计算量」之比（填充效率）为：

$$\eta = \frac{\sum L_i}{B \cdot L_{\max}} = \frac{8 \times 50}{8 \times 100} = 50\%$$

一半的算力花在 padding 上。若批内长短更悬殊，$\eta$ 可以跌到 20% 以下。

## 2 In-flight Batching 的核心机制

**In-flight Batching（飞行中批处理）**的核心主张：批不是一个「固定集合」，而是一个「在解码过程中持续进出的流」。它的机制可以拆成三块：

- **请求状态机**：每个请求有三个状态——`WAITING`（排队中）、`RUNNING`（解码中）、`COMPLETED`（已完成）。调度器每步（iteration）都检查请求状态，动态决定谁进批、谁出批。<span class="marginnote">这与 vLLM 调度器里 <code>waiting</code>`/<code>running</code>`/<code>swapped</code> 的状态机同构，只是换了个名字。见本专题《vLLM 调度器源码分析》两篇。
<strong>迭代级调度（iteration-level scheduling）</strong>：传统框架按「请求」调度（一个请求占满一次前向），In-flight Batching 按「token 步」调度——<strong>每一步都重新组成一个批</strong>。某请求这步 decode 完、达到了停止条件，它立刻离开；新等待的请求立刻顶上空位。
<strong>KV Cache 的动态分配</strong>：批成员变化意味着 KV Cache 的分配与释放要逐 token 进行，由内存管理器（类似 PagedAttention 的页式分配）负责，见本专题《PagedAttention》。</span>

一个典型调度回合：GPU 完成一步 decode → 收到结果 → 有请求结束 → 释放其 KV Cache 与 batch 槽位 → 从等待队列补入新请求（做 prefill）→ 组成新批 → 发射下一步。

## 3 Prefill 与 Decode 的混合调度

In-flight Batching 的难点在于：新进入的请求要做 **prefill**（计算量密集、矩阵大），已在跑的请求要做 **decode**（访存密集、矩阵窄）。两者放进同一个批，会面临计算形状的巨大差异。

<div class="marginnote">vLLM 的应对是 <code>chunked prefill</code>：把 prefill 切成小块，插进 decode 的空隙，见本专题《Chunked Prefill》。TensorRT-LLM 同样允许 prefill 与 decode 混合调度，但默认策略更保守：<strong>prefill 与 decode 分阶段或分块交替</strong>，避免大 prefill 把整批的 decode 延迟全部拖高。</div>

混合调度要回答一个权衡问题：**这一步是优先让排队中的 prefill 进批（换取更高的吞吐），还是优先让 running 的 decode 不被拖慢（保住延迟）？** 工程上通常用「prefill 块大小上限 + 批内 decode 数上限」来约束，让每一步的延迟抖动可控。

**辨析｜易错点：In-flight Batching 不改变单请求的延迟上限，只改变吞吐。** 它让 GPU 的利用率提高，单位时间完成的请求变多；但单个长请求的端到端延迟，主要由它自身长度与批内排队决定。把「吞吐上去了」误读成「每个请求都快了」，是运维推理服务最常见的认知偏差。

## 4 公式解析：In-flight Batching 的吞吐提升

设静态批处理与 In-flight Batching 用同一批请求，静态批固定大小 $B$、步数 $L_{\max}$；In-flight 的有效平均批大小随时间从 $B$ 衰减到接近 $B_{\text{eff}} = \sum L_i / L_{\max}$。

- **第一步，读静态批的耗时**：静态批完成全部请求需要 $L_{\max}$ 步，每步算 $B$ 个序列，总计算量 $B \cdot L_{\max}$。
- **第二步，读 In-flight 的耗时**：请求随完成陆续离开，GPU 每步实际算的序列数递减。总计算量约为 $\sum L_i = B \cdot \bar{L}$，且每步的 padding 接近 0。
- **第三步，比吞吐**：两步都完成同一批请求，In-flight 的计算量是静态批的 $\bar{L}/L_{\max}$。吞吐反比于计算量，因此加速比为：

$$\text{Speedup} \approx \frac{L_{\max}}{\bar{L}}$$

$\bar{L}$ 远小于 $L_{\max}$ 时（长短混合的真实负载），加速比可达 2–3 倍。**这是「算力不浪费在陪跑上」的数学表达**。实际增益还要扣除调度与内存管理开销，但量级方向不变。

## 5 数值算例：填充效率的账

把「陪跑浪费」算成不同负载的填充效率。设批大小 $B=32$，批内请求长度分布不同：

| 负载画像 | 平均长度 $\bar{L}$ | 最长 $L_{\max}$ | 填充效率 $\eta$ | In-flight 加速比 $L_{\max}/\bar{L}$ |
| --- | --- | --- | --- | --- |
| 长度均匀（50±10） | 50 | 60 | 83% | 1.2 倍 |
| 长短混合（10–500） | 150 | 500 | 30% | 3.3 倍 |
| 极悬殊（5–1000） | 200 | 1000 | 20% | 5 倍 |

**读这张表**：**负载越「长短悬殊」，静态批的浪费越大、In-flight 的收益越高**。长度均匀的批（如离线同长批量处理）收益小；真实在线流量（聊天、问答）长短悬殊，收益巨大。**这也是为什么「动态批处理」成为在线推理引擎的标配**——在线负载的形状天然适合它。

**一个数字上的细节**：In-flight 的加速比是「对同一批请求」而言。真实服务里，In-flight 还让 GPU「空闲期」被新请求填满（静态批要等攒满一批），这部分额外收益更大——**静态批是「等满才跑」，In-flight 是「来了就干」**。

## 6 In-flight Batching 的工程实现要点

从原理到落地，工程上有几个关键决策：

**请求状态的粒度**：状态机按「请求」还是按「token 步」推进？——**必须按 token 步（iteration）**。每步调度器检查全部请求，决定谁进批、谁出批、谁继续。这是「迭代级调度」的落地。

**KV Cache 的动态性**：批成员每步变化，KV Cache 的分配/释放要逐 token 进行。**必须用页式/分块分配**（见 PagedAttention 篇），否则批成员变化时「按连续序列预分配」的方式会崩溃。

**prefill 与 decode 的共存**：新进批的请求要 prefill（大矩阵），在跑的请求要 decode（窄矩阵）。**工程上用「prefill 块大小上限 + 批内 decode 数上限」约束**，让每步延迟抖动可控（见 Chunked Prefill 篇）。

**辨析｜易错点：In-flight Batching 不是「批越大越好」。** 批的成员可以动态进出，但**批的大小仍受显存与延迟约束**——批越大，每步 decode 越慢、KV 占用越多。In-flight 解决的是「算力不浪费」，不是「批可以无限大」——批大小上限仍是调度器的核心参数（见 max-num-seqs 篇）。

**In-flight 与其他优化的关系**：它和 PagedAttention（内存）、Chunked Prefill（prefill 切块）、投机解码（步数减少）是**互补**的——In-flight 管「批怎么组织」，其他优化管「每步怎么更快」。**理解 In-flight 是理解「引擎为什么能压出高吞吐」的地基**。

把静态批与 In-flight 的差异摊开对比：

| 维度 | 静态批 | In-flight Batching |
| --- | --- | --- |
| 批成员 | 固定，同步完成 | 每步动态进出 |
| 调度粒度 | 按请求 | 按 token 步 |
| padding | 有（等最长的） | 几乎为零 |
| 短请求 | 陪跑等长请求 | 完成即离开 |
| 空闲时 | 等攒满一批 | 有请求就干 |
| 工程复杂度 | 低 | 高（状态机 + 页式 KV） |

**读这张表**：In-flight 全面优于静态批——代价是「工程复杂度」（请求状态机、动态 KV 分配、混合调度都要实现）。**这也是为什么 In-flight 是「现代引擎标配」而「老框架没有」**：它需要的内存与调度基础设施（PagedAttention 等）是一整套工程投入。

**一个工程现实的补充**：In-flight 的收益在「在线、动态、长短混合」的负载上最大；纯离线、长度固定的批处理（如批量打分），静态批的损失很小，In-flight 的复杂度收益有限。

**「按负载选批策略」，而不是「一律用 In-flight」**——在线流量用动态批吃满吞吐，离线固定长度负载用静态批简单直接。工程上没有银弹，只有「匹配负载」的取舍。

## 7 小结

- **静态批处理把算力浪费在 padding 上**：填充效率 $\eta = \sum L_i / (B \cdot L_{\max})$，长短混合时可低到 20% 以下。
- **In-flight Batching 是「按 token 步调度」而非「按请求调度」**：请求每步都可能进出批，GPU 永远装满有效工作。
- **请求状态机**：`WAITING` / `RUNNING` / `COMPLETED` 三态，与 vLLM 的调度器同构；KV Cache 用页式内存动态分配。
- **prefill 与 decode 混合调度**需要控制延迟抖动，常以 chunk 化 prefill + 批大小上限约束。
- **收益是吞吐而非单请求延迟**：加速比约为 $L_{\max}/\bar{L}$，来自消除「陪跑」的 padding 计算。
- **负载越悬殊收益越大**：在线流量长短混合是 In-flight 的主场；与 PagedAttention、Chunked Prefill 互补。

在下一节，我们进入 TensorRT-LLM 与量化结合的部分——**量化感知与 TensorRT-LLM 的低精度支持**，看它如何把 INT8/FP8 内建进引擎。

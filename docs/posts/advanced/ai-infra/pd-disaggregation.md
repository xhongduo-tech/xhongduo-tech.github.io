---
title: PD 分离（prefill/decode disaggregation）架构
date: 2026-08-07
---

# PD 分离（prefill/decode disaggregation）架构

<div class="epigraph">
<p>当两种负载互相拖累时，把它们分开是最彻底的解法。</p>
<footer>—— 马丁 · 福勒（Martin Fowler，软件架构家）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ PD 分离论文（DistServe, Splitwise）与推理架构实践 · 推理基础设施篇 ｜ 2026-08-07</p>
</div>

## 为什么从 PD 分离开始

前面我们反复遇到同一个矛盾：**prefill 要算力、decode 要带宽**，同一块 GPU 无法两全。continuous batching 用「调度」缓解，但调度只能在「同一张卡」内腾挪——prefill 一旦涌入，decode 的带宽还是被挤。**PD 分离（prefill/decode disaggregation）** 是更彻底的答案：**把 prefill 和 decode 放到不同的 GPU 上**，各用各的卡、各算各的活。

这是 2023–2024 年推理架构最大的演进方向之一（DistServe、Splitwise、以及 vLLM 的 PD 分离支持）。理解它，就理解了「推理系统的下一形态」——为什么分离、怎么分离、代价是什么。

## 1 为什么分离：调度解决不了的矛盾

continuous batching 的局限在于「同一张卡上，prefill 与 decode 仍互相挤占」：

- prefill 是「算力瞬间拉满」的重活——它一来，decode 的带宽份额被抢。
- decode 是「长期占卡」的轻活——它不走，prefill 的算力没处用。
- 两者在同一张卡上，**任何一个的高峰都会伤害另一个的延迟**。

更本质的矛盾是**缩放方向相反**：

- prefill 需要「算力强的卡」——多几个 TFLOPS 就快一点。
- decode 需要「带宽高的卡」——多几 TB/s 就快一点。
- 一张卡同时放大算力和带宽，成本是指数级上升的。

**分离的逻辑**：prefill 与 decode 是两种「完全不同的商品」，为什么要用同一种机器生产？<span class="marginnote">「不同负载用不同硬件」是分布式系统的一贯思路：正如 CPU 与 GPU 分工、训练与推理分离，prefill 与 decode 的分离是「按负载特征切分硬件」的又一次应用。分离后，prefill 卡可以少配带宽、多配算力，decode 卡反之——<strong>每张卡都只为自己的负载优化，成本大幅下降</strong>。</span>

## 2 分离架构：prefill 卡与 decode 卡各司其职

PD 分离的架构：

1. **Prefill 池**：若干算力强的 GPU，专职处理 prompt 的 prefill，产出「KV Cache + 第一个 token」。
2. **KV 传输**：把 prefill 产出的 KV Cache 通过网络传给 decode 卡。
3. **Decode 池**：若干带宽高的 GPU，接收 KV Cache，专职逐步 decode，直到生成完。

一个请求的生命周期：进 prefill 池 → KV Cache 转移到 decode 池 → decode 池逐步生成 → 完成退出。**prefill 卡腾出来了，可以立刻服务下一个请求；decode 卡持续做它最擅长的带宽密集活**。<span class="marginnote">分离后的规模可按负载独立伸缩：prompt 多的场景扩 prefill 池，输出多的场景扩 decode 池——「按需扩缩」从「整模型副本」细化为「分阶段副本」。这比整体扩缩容灵活得多，也省钱得多。</span>

## 3 关键技术：KV 传输与调度

PD 分离的工程难点不在「分」，而在「传」与「调」：

- **KV 传输开销**：KV Cache 要跨网络传给 decode 卡。KV 大小 = 每 token 的 KV 字节 × prompt 长度，可能几十 MB 到几 GB——**网络是分离架构的咽喉**。
- **KV 压缩/量化**：传输前把 KV 量化（如 KV INT8），传输量减半——带宽换质量的经典权衡。
- **调度与亲和**：请求路由到「有 KV 缓存的 decode 卡」还是「任意 decode 卡」？多轮对话中，同一会话的后续请求应路由到持有其 KV 的卡。

**「KV 跨卡」是把 prefill 与 decode 分离的代价**：传输时间必须远小于 decode 时间，否则分离得不偿失。<span class="marginnote">KV 传输的优化方向：要么压缩（KV 量化、低精度 KV），要么「就近」——prefill 卡与 decode 卡在同一机架/交换机下，网络延迟与带宽都友好。工程上「PD 共置（co-locate）+ 高速网络」是常见妥协：分离逻辑但不分离物理距离。</span>

## 4 收益与代价

**收益**：

- **互不拖累**：prefill 高峰不再挤爆 decode 的带宽——TTFT 与 TPOT 各自稳定。
- **硬件优化**：prefill 卡少带宽多算力、decode 卡多带宽少算力——单位成本产出更高。
- **独立扩缩**：按 prompt/输出的流量比例，分别扩两个池子。

**代价**：

- **KV 传输**：额外的网络开销与延迟。
- **复杂度**：两套池子、两套调度、KV 路由——系统复杂度大增。
- **显存翻倍**：两个池子各自加载一份权重（prefill 卡与 decode 卡都要权重），总显存需求上升。

**何时值得**：当并发高、prompt 长、或 TTFT/TPOT 要求苛刻时，收益远大于代价。<span class="marginnote">PD 分离的「显存翻倍」是它最常被质疑的点——两套权重确实是双份。但算总账：分离后每张卡只干一种活，利用率提升带来的吞吐收益，通常远超「多一份权重」的显存成本。这也是为什么大厂的高端推理服务普遍走向 PD 分离。</span>

## 5 公式解析：分离的收益判据

设模型权重 $2N$，prefill 耗时 $T_p = \frac{2NP}{C_p}$，decode 每步 $T_d = \frac{2N}{B_d}$，KV 传输 $T_{\text{kv}} = \frac{\text{KV}}{B_{\text{net}}}$。

**分离后单请求总延迟**：

$$T_{\text{sep}} = T_p + T_{\text{kv}} + S \cdot T_d$$

**不分离**（同一张卡，算力 $C$、带宽 $B$，串行承担）：

$$T_{\text{joint}} = \frac{2NP}{C} + S \cdot \frac{2N}{B}$$

- **分离的收益来源**：$C_p > C$（prefill 卡算力专配）、$B_d > B$（decode 卡带宽专配）——每个阶段都比「折中卡」快。
- **分离的代价**：$T_{\text{kv}}$ 是额外项。
- **判据**：当 $T_{\text{kv}} \ll S \cdot T_d$（KV 传输远小于 decode 总时长）时，分离划算——**长输出场景分离收益最大**。<span class="marginnote">代入数字：70B 模型，prefill 卡算力 2×、decode 卡带宽 1.5×，输出 500 token。不分离 decode 部分 500×42ms=21s；分离后 decode 500×28ms=14s，KV 传输 1GB/25GB/s=40ms（可忽略）——总延迟从 21s+ 降到 14s+。输出越长、并发越高，分离越值。</span>

## 6 辨析｜易错点：PD 分离的常见误区

**辨析｜易错点：**
- **「分离就是两套独立部署」不完整**：分离的核心是「KV 高效传递 + 分阶段扩缩」，不是简单复制两套。
- **「KV 传输免费」是错觉**：KV 是分离的咽喉，不压缩/不就近，传输可能吃掉全部收益。
- **「分离只适合大厂」不绝对**：单集群、单机也可以逻辑分离（同机不同卡），收益取决于负载特征。
- **「显存翻倍所以不划算」是静态看法**：算总账要看「利用率提升带来的吞吐」，通常净收益为正。
- **别忽略「多轮对话的 KV 亲和」**：同一会话的请求必须路由到持有 KV 的 decode 卡，否则 KV 白传、prefill 重来。

## 7 小结

- **分离动机**：prefill 要算力、decode 要带宽，同一张卡无法两全，缩放方向相反。
- **架构**：prefill 池（算力强）+ KV 传输 + decode 池（带宽高），按阶段独立扩缩。
- **关键技术**：KV 量化/压缩、就近部署、KV 亲和路由。
- **收益**：互不拖累、硬件专配、独立扩缩；**代价**：KV 传输、复杂度、双份权重。
- **判据**：$T_{\text{kv}} \ll S \cdot T_d$ 时分离划算——长输出、高并发场景收益最大。

## 8 进阶与延伸

**动手画一张 PD 分离的时序图**：画一个请求在「prefill 池 → KV 传输 → decode 池」的生命周期，标出每个阶段的耗时（用本篇公式代入数字）——你会看到「KV 传输」这一项在什么场景下可忽略、什么场景下成了瓶颈。

**几个值得进一步挖的方向**：

- **KV 量化的收益**：KV Cache 用 INT8 存储，传输量减半——KV 量化对「PD 分离的咽喉」有多大的缓解？量化 KV 的精度代价怎么评估？
- **prefill 池与 decode 池的配比**：负载「prompt 长 / 输出短」时，prefill 池该多配；「prompt 短 / 输出长」时反之——怎么根据「prompt:输出 比例」算两池的 GPU 配比？
- **PD 分离的容错**：decode 池的某个实例挂了，正在 decode 的请求怎么办？「KV 缓存丢失 → 重新 prefill」的代价——这是分离架构的容错难点。

**自测题**：为什么「长输出场景 PD 分离收益最大」？如果你能说清「KV 传输 $T_{\text{kv}}$ 相对 $S \cdot T_d$ 越小、分离越划算」，就抓住了分离的判据。

在下一节，我们进入**第十篇 监控与性能剖析**的第一课——**训练任务监控体系**：GPU 利用率、显存、网络带宽与功耗。

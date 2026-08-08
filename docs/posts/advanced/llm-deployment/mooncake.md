---
title: Mooncake：以 KV Cache 为中心的分离式架构
date: 2026-08-07
---

# Mooncake：以 KV Cache 为中心的分离式架构

<div class="epigraph">
<p>缓存不再住在计算旁边，而是成为集群的一等公民。</p>
<footer>—— Mooncake 架构理念（Mooncake 团队, 2024）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ Mooncake 论文（Qin et al., 2024） ｜ 2026-08-07</p>
</div>

## 为什么从 Mooncake 开始

PD 分离把 KV Cache 从「prefill 节点挪到 decode 节点」，但每次传输都是「临时抱佛脚」——传完就完，下一个请求还要重来。Mooncake 把这个思路推到极致：**把 KV Cache 从「跟随计算的附属品」升级为「集群级的一等资源」**。它用一层**以 KV Cache 为中心的分离式架构**，把 KV 缓存放在统一的存储池里，prefill 与 decode 节点通过它共享与复用缓存，而不只是「传输」。<span class="marginnote">Mooncake 的名字来自月饼——它诞生于月之暗面（Moonshot AI）的 Kimi 大模型部署实践。它的核心口号：<strong>把「计算为中心」改成「以 KV 为中心」</strong>。</span>

本篇讲 Mooncake 的动机、分离式 KV 池的设计、以及它如何用「缓存复用」改写 PD 分离的传输代价。

## 1 从「传输 KV」到「共享 KV」

PD 分离的朴素实现里，KV Cache 是**一次性消费**的：prefill 算完、传给 decode、decode 用完即弃。下一个相同前缀的请求，又要重新 prefill。Mooncake 的观察：

**多轮对话与系统提示词**让「相同前缀」的请求极多——每轮追问都带着同一个长前缀；
这些 KV Cache 若**持久化**下来，后续请求直接复用，能省掉大部分 prefill 计算与 KV 传输；
KV Cache 不再「跟着请求走」，而是「留在池子里，谁需要谁取」。

于是 Mooncake 引入**统一的 KV 缓存池（KV Cache Pool）**：一个分布式的、以 KV 块为单位的存储系统。prefill 节点算出的 KV 写入池，decode 节点从池读取，相同前缀的请求直接命中池里的块——**复用取代重算**。<span class="marginnote">这与本专题《RadixAttention》《Prefix Caching》是同一思想，只是从「单实例内」放大到「跨实例集群」：<strong>实例内的前缀树，Mooncake 把它变成集群级的 KV 池</strong>。</span>

## 2 分离式架构的组件

Mooncake 的架构由三类角色构成：

**Prefill 实例**：负责把 prompt 变成 KV Cache（算力密集）；产出后写入 KV 池，同时把首 token 交给 decode 实例。
**Decode 实例**：从 KV 池拉取前缀 KV，持续生成 token；生成中新增的 KV 也持续回写池子（供后续复用）。
**KV 池 / 调度器**：统一管理 KV 块的分配、放置、复制与调度。调度器决定「请求去哪、KV 放哪、谁复用谁」。

调度器的关键决策：**当两个请求共享长前缀时，让它们尽量落在同一组 decode 实例上**——这样 KV 池中的块可以跨请求共享，命中率高。<span class="marginnote">这又回到《Cache-aware 路由》：路由与缓存是同一枚硬币。<strong>Mooncake 把「路由决策」和「KV 放置决策」耦合在一起</strong>——请求去哪，取决于 KV 池里谁有它要的前缀。</span>

## 3 以 KV 为中心的调度收益

「以 KV 为中心」的收益可以从三个角度看到：

**Prefill 计算量下降**：相同前缀的请求命中缓存，不再重复 prefill。多轮对话场景中，实测可省去 50%–90% 的重复 prefill。<span class="marginnote">缓存复用的收益有上界：<strong>首轮（冷缓存）必须全量 prefill</strong>，且 KV 池的容量、驱逐策略决定命中率上限。多轮对话越深、系统提示词越长，收益越大。
<strong>传输开销降低</strong>：命中缓存的请求只需从池「取已存的块」，而不是「传全新算出的块」——网络负载下降。
<strong>吞吐与延迟双升</strong>：TTFT 因省去 prefill 而下降；整体吞吐因「算力不再重复浪费」而上升。</span>

Mooncake 论文报告：在长上下文、多轮对话的高并发负载下，它相对传统 PD 分离可提升约 2 倍的吞吐，同时显著改善 TTFT。

**辨析｜易错点：Mooncake 不是「把 KV 放到硬盘」。** KV 池是**内存级**的分布式存储（跨节点的 DRAM/HBM），不是磁盘缓存。KV Cache 的访问频率极高，放磁盘会慢几个数量级。**「分离」指的是与计算解耦、跨实例共享，不是降级到慢速存储**。

## 4 公式解析：缓存复用改写成本

设请求流中相同前缀出现 $R$ 次，每次 prefill 该前缀的成本为 $C_p$，KV 传输成本 $C_t$。无复用时总成本：

$$C_{\text{no-share}} = R \cdot (C_p + C_t)$$

有 KV 池复用后：

- **第一步，写复用成本**：第一次 prefill 付全价 $C_p$，之后每次只需从池取 KV（成本 $C_{\text{read}} \ll C_p$）加传输 $C_t$。总成本：

$$C_{\text{share}} = C_p + (R-1)(C_{\text{read}} + C_t)$$

- **第二步，算节省**：节省量 $= (R-1)(C_p - C_{\text{read}})$。**每复用一次，就省一次完整的 prefill**。
- **第三步，看规模效应**：当 $R$ 大（多轮对话轮次深）时，总成本从 $R \cdot C_p$ 量级降到 $C_p + (R-1) \cdot C_t$ 量级——**prefill 的重复计算被缓存替代**，这就是「省 50%–90% prefill」的数学来源。代价是 KV 池的存储成本与调度复杂度，但相比重复计算，内存远比算力便宜。

## 5 小结

- **Mooncake 把 KV Cache 升级为集群一等资源**：统一的 KV 池，prefill 写入、decode 读取、多请求共享。
- **组件三角色**：Prefill 实例、Decode 实例、KV 池 + 调度器；调度把「路由」与「KV 放置」耦合。
- **核心收益**：相同前缀的 KV 被复用，省去重复 prefill，TTFT 与吞吐双升（长上下文多轮场景约 2 倍）。
- **KV 池是内存级**：跨实例共享、不是磁盘缓存，访问频率决定了它必须在高速存储上。
- **复用改写成本**：总成本从「每次全价 prefill」降到「首次全价 + 后续廉价读取」。

在下一节，我们看支撑 KV 高速传输的底层网络技术——**跨节点 KV Cache 传输与 RDMA**。

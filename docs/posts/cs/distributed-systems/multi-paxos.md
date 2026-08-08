---
title: Multi-Paxos 与 Leader 选举优化
date: 2026-08-07
---

# Multi-Paxos 与 Leader 选举优化

<div class="epigraph">
<p>共识本身不是终点，共识的高吞吐复用才是——Multi-Paxos 让「每次决策」共享同一次选主，把共识从协议变成通道。</p>
<footer>—— 参照 Lamport 的 Paxos 论文与 Chubby 论文（Burrows 2006）</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ MIT 6.824 第7讲 / Lamport 2001 ｜ 2026-08-07</p>
</div>

## 为什么从 Multi-Paxos 开始

Basic Paxos 每达成一个值就要跑一轮完整的 Prepare + Accept——两阶段、两轮多数派往返，吞吐低得没法用。真实系统需要的是「**一条日志**」：第 1 条、第 2 条、第 3 条……每条是一个共识值，应用按顺序应用它们。**Multi-Paxos** 的洞察是：**选主（leader）这件事只需做一次，之后的每一轮共识都可以跳过 Prepare 阶段**。一次选主，换来持续的低成本拍板。<span class="marginnote">Multi-Paxos 是 Lamport 论文里的「Paxos 用于复制日志」章节，Google 的 Chubby、Megastore 都基于它。Raft 可以看作 Multi-Paxos 的「说人话重写版」——它们解决的问题几乎相同，只是工程取舍不同（见下一节对照）。</span>

## 1 从「每轮共识」到「一条日志」

Basic Paxos 被描述成「议长投票」：每个议题（一个值）都要两阶段。Multi-Paxos 的升级是把议题排成**日志序号**（log index）：第 1 个议题、第 2 个议题……每个议题仍是一次 Basic Paxos，但有了固定 leader 之后：

**选主阶段**：先跑一轮完整的两阶段，选出唯一的 leader（proposer 里轮次最高者胜出）。
**稳态阶段**：此后 leader 对每个日志序号直接发 Accept，**省略 Prepare**——因为唯一的 leader 不再需要探测「谁持有旧值」，它自己就是那个「知道所有已定值」的人。

省略 Prepare 的前提是：**同一个时刻只有一个活跃 leader**。若两个 leader 同时活跃（网络分区等），它们各自发 Accept，可能产生冲突——此时退回到带 Prepare 的完整两阶段来恢复一致性。这是 Multi-Paxos 的「降级通道」：正常时快，异常时慢但安全。<span class="marginnote"><strong>辨析｜易错点：</strong>Multi-Paxos 不是「每轮都选一个新 leader」，而是「选一次 leader 用很久」。选主是低频的（仅在 leader 崩溃或过期时），日志复制是高频的——把高频路径上的 Prepare 省掉，就是它比 Basic Paxos 吞吐高的全部秘密。</span>

## 2 Leader 选举：谁是 leader，如何判定失效

选主（leader election）是 Multi-Paxos 的关键件。选举机制要回答两个问题：

**怎么选**：leader 向多数派提出「我要当 leader」（用更高的轮次号 Prepare，获得多数派承诺即当选）。拿到多数派承诺 = 多数派承诺不再接受旧轮次的提议 = 自己是当前唯一合法的 leader。
**怎么失效**：leader 周期性发心跳（续期）；follower 若在超时时间内没收到心跳，就认为 leader 失效，发起新一轮选举（提升自己的轮次号，重新 Prepare）。

**选举的活性依赖超时**：FLP 告诉我们，没有超时就没有活性保证——Raft/Paxos 的超时时间要足够大，避免频繁误判（抖动），又要足够小，让故障恢复及时。<span class="marginnote">选举有个微妙要求：<strong>新 leader 必须知道「哪些日志序号已经被提交」</strong>。做法是新 leader 在选主时读取多数派里「已接受的最高日志序号」，并强制所有未提交位置失效——保证它不会在旧日志上「分叉」。这一规则在 Raft 里被具体化为「新 leader 的日志必须是最新的」（选举限制，见 Raft 安全篇）。</span>

## 3 复制日志：把状态机喂饱

一旦 leader 就位，日志复制的流程固定成：

1. 客户端把操作发给 leader（如 `SET x = 3`）。
2. leader 把操作追加到自己的日志序号 $i$，向所有 follower 广播 Append。
3. follower 收到后本地追加并 ack；leader 收到多数派 ack 后，标记序号 $i$ 为**已提交（committed）**，应用该操作，并向客户端返回成功。
4. follower 之后也会在「已知多数派已提交」时应用该操作——提交点由 leader 决定并传播。

这就是**状态机复制（state machine replication）**的雏形：所有副本以**相同顺序**应用**相同操作**，最终状态必然一致。Multi-Paxos 把「每个操作的共识」折叠成「每个日志序号的共识」，状态机复制是它的消费者。<span class="marginnote">「日志序号 = 共识的每次实例」是理解复制日志的关键：第 3 篇 Lamport 时钟里说过「共识序号 = Lamport 时间戳」，这里就是它的正式登场——日志序号的单调递增就是全局全序的骨架，副本按同一序号应用操作，等价于按同一全序执行。</span>

## 4 公式解析：稳态路径的往返次数

量化 Multi-Paxos 相比 Basic Paxos 的吞吐提升。设多数派大小为 $q$，网络往返为 RTT：

$$
T_{\text{Basic}} = 4 \times \text{RTT} \quad (\text{Prepare 2 个 RTT + Accept 2 个 RTT})
$$

$$
T_{\text{Multi（稳态）}} = 2 \times \text{RTT} \quad (\text{Accept 一个 RTT 到多数派 + 一个 RTT 回 leader})
$$

逐项拆解：

- **Basic**：Prepare（leader → 多数派 → leader）+ Accept（leader → 多数派 → leader），两轮都是完整的两个单程，共 4 个单程 ≈ 2 个 RTT 每值。
- **Multi 稳态**：Accept 到多数派（1 单程）+ ack 回 leader（1 单程）≈ 1 个 RTT 每值——**省掉了 Prepare 的两个单程**。
- 在局域网里 RTT 约 0.5–2ms，Multi 的吞吐可以到 Basic 的 2 倍以上；在广域网里差距更大（每省一个 RTT 都值钱）。

这就是 Multi-Paxos 的工程价值：**把「每个共识值」的代价从两阶段压到一阶段**，让共识从「协议」变成可高吞吐复用的「通道」。吞吐瓶颈从此不在协议，而在磁盘 fsync 与网络带宽。<span class="marginnote">注意稳态路径的简化是「已选主 + 无冲突」的理想情况：一旦出现两个 leader（分区恢复期），系统临时退回完整两阶段，吞吐跳水但安全不破——这个「正常快、异常慢但正确」的模式是共识系统普遍接受的工程权衡。</span>

## 5 小结

- **Multi-Paxos** = Basic Paxos 的日志化：每个日志序号一次共识，一次选主长期复用。
- **选主**：一次完整两阶段选出唯一 leader，靠多数派承诺保证唯一性；心跳超时驱动失效与重选。
- **稳态**：leader 对每个序号直接 Accept 多数派确认即提交——**省略 Prepare**，每值只花 1 个 RTT。
- **恢复**：出现竞争 leader 时退回完整两阶段，安全不破、吞吐让位。
- 副本按相同顺序应用相同日志 → **状态机复制**，这是所有强一致复制的统一模型。
- Multi-Paxos 是理论原型，工程实现细节多（leader 换届、日志空洞、快照）——Raft 把它们全显式化了。

在下一节，我们进入 Raft 三部曲的第一部——**Raft 的 Leader 选举**，看它如何用「任期 + 随机超时」把选主做成简单可靠的显式模块。

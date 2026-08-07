---
title: etcd 与基于 Raft 的协调服务
date: 2026-08-07
---

# etcd 与基于 Raft 的协调服务

<div class="epigraph">
<p>etcd 把 Raft 从论文变成生产级服务——Kubernetes 整个控制面的心脏，就是一撮撮键值对。</p>
<footer>—— 参照 etcd 官方文档与 Raft 论文（Ongaro & Ousterhout 2014）</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ etcd 文档 / Raft 工程 ｜ 2026-08-07</p>
</div>

## 为什么从 etcd 开始

ZooKeeper 用 Zab，etcd 用 **Raft**——它是「基于 Raft 的协调服务」的代表，也是 Kubernetes、分布式锁、配置中心、服务发现（Consul 也用类似技术）的现代底座。理解 etcd = 理解「如何把第 5 篇的 Raft 论文变成一个生产级系统」，以及「协调服务」在现代云原生世界的形态。<span class="marginnote">etcd 由 CoreOS 在 2013 年开发，是「云原生」运动的基础组件：Kubernetes 把集群的所有状态（节点、Pod、Service、配置）存在 etcd 里。它的 API 是「键值 + 事务 + watch」——比 ZooKeeper 的 znode 树更简单，靠 Raft 提供强一致。</span>

## 1 为什么 Kubernetes 选 etcd

Kubernetes 需要一个「整个集群的控制面存储」，要求：

- **强一致**：集群状态不能有分歧——两个节点对「Pod 该跑在哪」必须有一致答案。
- **高可用**：etcd 挂了 = Kubernetes 控制面瘫痪（无法调度、无法更新）。
- **watch**：Kubernetes 需要「实时感知状态变化」——Pod 被删、节点变化都要立刻通知控制器。
- **事务**：多步状态更新要原子（如「创建 Deployment 同时记录版本」）。

etcd 的「Raft 强一致 + 键值事务 + watch」恰好满足——Kubernetes 的所有控制器（deployment、service、node）本质上都是「watch etcd 的键值变化 → 收敛实际状态到期望状态」的循环。<span class="marginnote"><strong>辨析｜易错点：</strong>Kubernetes 用 etcd 做「控制面存储」，不是「业务数据存储」——Pod 的镜像、ConfigMap 可以放 etcd，但「业务数据库」绝不放 etcd。etcd 适合「少量、关键、强一致、频繁读」的协调数据；海量业务数据放它是错误用法（性能与成本都不合适）。</span>

## 2 etcd 的 API：键值 + 事务 + watch

etcd 的核心 API 三件套：

- **键值（KV）**：`put(key, value)`、`get(key)`、`delete(key)`——带**版本号（revision）**，每个键有多个版本（MVCC，第 8 篇快照隔离的兄弟）。
- **事务（txn）**：`if ... then ... else ...`——**比较（compare）+ 提交**的原子操作。比较可以是「键的版本 == x」「值 == y」等；满足才执行 then 分支，否则 else 分支。这是实现「乐观锁、分布式锁、leader 选举」的基础。
- **watch**：订阅键/前缀的变更——变更推送（带版本号），支持「从历史版本开始 watch」。

**事务的价值**：etcd 的事务是「单节点原子」的（走 Raft 提交）——这让「读-比较-写」可以做成原子操作。分布式锁、选主、乐观并发控制全建立在它之上。对比 ZooKeeper 的「临时节点 + 顺序节点」拼装，etcd 的「事务 + watch」更通用、更直接。<span class="marginnote">etcd 的 MVCC（多版本）让它能「时间旅行」：`get` 可以指定版本号读历史值，watch 可以从任意版本开始。这与第 8 篇的快照隔离同源——etcd 每个键的版本号是全局单调的 revision，相当于「数据库的事务 ID」，提供一致的快照视图。</span>

## 3 用 etcd 实现协调模式

ZooKeeper 的四块积木在 etcd 里都有对应，但用「事务」实现更直接：

- **分布式锁**：`if create(key, value, ttl) succeeds → 拿到锁`——用事务 + 租约（lease）实现；`delete(key)` 释放；`lease` 保证崩溃自动释放。
- **选主**：多个候选 `put` 同一个 key（带租约），谁成功谁当主；主崩溃 → 租约过期 → key 消失 → 其他候选 watch 到后重新竞选。
- **配置中心**：`put` 配置 + watch 订阅变更——配置更新实时推送。
- **分布式队列**：`put` 时带「有序键」（如 `task/0001`），读时取最小键——用 revision 排序。

与 ZooKeeper 的差异：etcd 没有「临时节点」这个原生概念，用「**租约（lease）+ 自动过期 key**」替代——更灵活（租约可关联任意 key、可续租、可关联多个 key）。<span class="marginnote">etcd 的租约 vs ZooKeeper 的临时节点：临时节点绑「会话」，会话断才删；etcd 的租约绑「时间」，到期才删。前者语义是「会话存活」，后者语义是「定期续租」——etcd 的模型更精细（不同 key 可以不同 TTL），也更接近第 6 篇「租约」的原始定义。</span>

## 4 Raft 在 etcd 里的工程化

etcd 内置的 Raft 实现（etcd/raft 库）是 Raft 论文的「标准工程实现」，被 TiKV、Dragonboat 等大量项目复用。它的工程细节：

- **WAL 持久化**：所有 Raft 日志写 WAL（预写日志），fsync 后 ack——「先落盘、再确认」。
- **快照（snapshot）**：状态机定期生成快照，压缩旧日志；慢 follower 用快照追赶。
- **选举超时随机化**：Raft 论文要求「随机化选举超时」防平票——etcd 的实现遵守。
- **只读一致性**：etcd 支持三种读模式——`serializable`（任意副本读，可能旧）、`linearizable`（走 leader + 仲裁读，保证最新）。Kubernetes 默认用 `linearizable` 读关键状态。
- **成员变更**：用 Raft 的单节点变更（joint consensus 的简化）安全扩缩容。

**etcd 的调优参数**（`--election-timeout`、`--heartbeat-interval`）直接对应 Raft 的时间常数——理解 Raft（第 5 篇）就理解了 etcd 的每个配置项。<span class="marginnote">「线性化读」的细节：etcd 的 `linearizable` 读要等 leader 确认「自己还是 leader」（通过跟多数派通信），才能保证读到的不是「过期 leader 的旧状态」——这是第 5 篇「只读请求」问题的生产答案。性能敏感场景可以用 `serializable` 读（快但可能旧），正确性敏感场景必须 `linearizable`。</span>

## 5 公式解析：租约与 TTL 的过期语义

把「etcd 租约的过期如何保证锁安全」量化。设租约 TTL 为 $L$、持有者续约周期 $C$、续约往返延迟 $R$：

$$
\text{锁被误夺条件：} \underbrace{\text{连续错过续约} \times R > L}_{\text{网络持续断开超过 } L} \;\Rightarrow\; P \approx \left(\frac{R}{L}\right)^{\text{错过的续约次数}}
$$

拆解：

- **安全条件**：只要持有者能在「租约到期前」完成续约（$C + R < L$），锁就一直在手——续约周期远小于 TTL 是基本要求。
- **误夺条件**：网络断开导致连续多次续约失败，且断开总时长超过 $L$——锁被释放、他人获取。
- **概率**：单次续约失败概率取决于网络；「连续错过到超过 $L$」的概率随 $L/C$ 增大而指数下降——**TTL 越长，误夺概率越低**，但「故障接管速度」也越慢。
- **工程权衡**：Kubernetes 给 etcd 设 `lease` 通常几十秒——在「误夺风险」与「接管速度」之间平衡。

这条式子的工程含义：**租约 TTL 是「锁安全性」与「故障可用性」之间的旋钮**——TTL 长保锁不被误夺（安全），但主故障后要等更久才有人接管（可用性差）。etcd 的默认值体现「宁可慢接管、不愿锁被打断」的取舍，与 Chubby 的几十秒租约如出一辙。<span class="marginnote">这也是「分布式锁」与「Raft 租约」的区别：Raft 的 leader 租约是「内部机制」（防双主），etcd 对外的 lease 是「用户机制」（防锁过期）；两者的过期都会导致「持有者被取代」——所以用户拿锁后的操作仍要短、要配围栏令牌（第 6 篇）。</span>

## 6 小结

- **etcd**：基于 Raft 的键值协调服务，Kubernetes 控制面的心脏。
- API 三件套：**KV（MVCC 多版本）+ 事务（比较-提交原子）+ watch（变更推送）**。
- 协调模式：分布式锁（租约 + 事务）、选主（租约过期 + watch）、配置中心（watch）、队列（有序键）。
- 与 ZooKeeper 的差异：没有「临时节点」，用**租约 + 自动过期 key** 替代——更灵活。
- Raft 工程化：WAL 持久化、快照压缩、选举超时随机化、线性化读、成员变更。
- 租约 TTL 是「锁安全」与「接管速度」之间的旋钮——几十秒默认是「宁慢勿夺」的取舍。

在下一节，我们看分布式锁领域最著名的一场论战——**Redlock 的争议与基于数据库锁的替代方案**。

---
pageClass: plain-doc
---

# 分布式系统

本篇对标 MIT 6.824 课程与《数据密集型应用系统设计》（DDIA）的章节体系，系统梳理分布式系统的核心理论与经典工程实践，从通信、时钟、一致性到共识、复制、分区、事务与容错，并逐一剖析 GFS、MapReduce、Spanner、Dynamo 等里程碑系统。

## 主题规划

<ProgressGrid cat="cs/distributed-systems" />


### 第 1 篇 分布式系统概述

- [x] [什么是分布式系统：动机与定义](./what-is-distributed-systems)
- [x] [分布式系统的核心挑战：部分失败与不确定性](./partial-failure-and-uncertainty)
- [x] [分布式计算的谬误（Fallacies of Distributed Computing）](./fallacies-of-distributed-computing)
- [x] [系统模型：同步、异步与部分异步](./system-models)
- [x] [故障模型：崩溃故障、拜占庭故障与网络分区](./failure-models)
- [x] [可扩展性、可用性与性能度量](./scalability-availability-performance)
- [x] [CAP 定理及其正确解读](./cap-theorem-revisited)
- [x] [BASE 与最终一致性](./base-eventual-consistency)

### 第 2 篇 RPC 与通信

- [x] [远程过程调用（RPC）的基本原理](./rpc-basics)
- [x] [接口定义语言（IDL）与序列化：Protocol Buffers 与 Thrift](./idl-serialization-protobuf-thrift)
- [x] [gRPC 的设计与流式调用](./grpc-design-streaming)
- [x] [消息传递模型：点对点、发布订阅与消息队列](./message-passing-models)
- [x] [超时、重试与幂等性](./timeout-retry-idempotency)
- [x] [网络分区与超时检测的两难](./network-partition-timeout-detection)
- [x] [REST 与 RPC 的取舍](./rest-vs-rpc)

### 第 3 篇 逻辑时钟与事件排序

- [x] [物理时钟的局限与时钟同步（NTP）](./physical-clocks-ntp)
- [x] [发生在先关系（happens-before）与偏序](./happens-before-partial-order)
- [x] [Lamport 时钟（Lamport Timestamps）](./lamport-clocks)
- [x] [向量时钟（Vector Clocks）与因果检测](./vector-clocks-causality)
- [x] [版本向量（Version Vectors）在副本系统中的应用](./version-vectors)
- [x] [混合逻辑时钟（HLC）与 TrueTime](./hlc-truetime)

### 第 4 篇 一致性问题

- [x] [一致性模型概览：从强到弱](./consistency-models-overview)
- [x] [线性一致性（Linearizability）的定义与验证](./linearizability)
- [x] [顺序一致性（Sequential Consistency）](./sequential-consistency)
- [x] [因果一致性（Causal Consistency）](./causal-consistency)
- [x] [最终一致性与会话保证](./eventual-consistency-session-guarantees)
- [x] [一致性与可用性的权衡实践](./consistency-availability-tradeoffs)

### 第 5 篇 共识算法

- [x] [共识问题的形式化定义与 FLP 不可能定理](./flp-impossibility)
- [x] [Paxos 的直观理解：Basic Paxos 详解](./basic-paxos)
- [x] [Multi-Paxos 与 Leader 选举优化](./multi-paxos)
- [x] [Raft 算法详解：Leader 选举](./raft-leader-election)
- [x] [Raft 算法详解：日志复制](./raft-log-replication)
- [x] [Raft 算法详解：安全性与成员变更](./raft-safety-membership-change)
- [x] [Paxos 与 Raft 的对比与工程实现要点](./paxos-vs-raft)
- [x] [拜占庭容错与 PBFT 简介](./pbft-byzantine-fault-tolerance)

### 第 6 篇 复制

- [x] [复制的目标与难点](./replication-goals-challenges)
- [x] [主从复制：同步与异步](./single-leader-replication)
- [x] [复制延迟与读己之写一致性](./replication-lag-read-your-writes)
- [x] [多主复制与写冲突处理](./multi-leader-replication)
- [x] [冲突解决：最后写入获胜（LWW）与冲突自由复制数据类型（CRDT）](./lww-crdt)
- [x] [无主复制：Quorum、读修复与反熵](./leaderless-replication-quorum)
- [x] [脑裂（Split-Brain）问题与围栏（Fencing）](./split-brain-fencing)

### 第 7 篇 分区与再平衡

- [x] [分区的动机与分区键选择](./partitioning-motivation-partition-keys)
- [x] [按键范围分区与按哈希分区](./range-vs-hash-partitioning)
- [x] [一致性哈希与虚拟节点](./consistent-hashing-virtual-nodes)
- [x] [次级索引的分区：局部索引与全局索引](./secondary-index-partitioning)
- [x] [再平衡策略：固定分区、动态分区与按节点比例](./rebalancing-strategies)
- [x] [请求路由与服务发现](./request-routing-service-discovery)

### 第 8 篇 分布式事务

- [x] [分布式事务的动机与局限](./distributed-transactions-motivation)
- [x] [两阶段提交（2PC）详解及其阻塞问题](./two-phase-commit)
- [x] [三阶段提交（3PC）与其假设](./three-phase-commit)
- [x] [快照隔离（Snapshot Isolation）与 MVCC](./snapshot-isolation-mvcc)
- [x] [快照隔离的写倾斜（Write Skew）问题](./write-skew)
- [x] [可串行化与 SSI（可串行化快照隔离）](./serializability-ssi)
- [x] [Saga 模式与补偿事务](./saga-pattern)
- [x] [Percolator 式乐观分布式事务](./percolator)

### 第 9 篇 容错与故障检测

- [x] [故障检测器：心跳与超时](./failure-detectors-heartbeat)
- [x] [Phi 累积故障检测器](./phi-accrual-failure-detector)
- [x] [成员管理：Gossip 协议](./gossip-protocol)
- [x] [检查点（Checkpoint）与日志恢复](./checkpoint-log-recovery)
- [x] [状态机复制（State Machine Replication）](./state-machine-replication)
- [x] [混沌工程：故障注入与系统韧性验证](./chaos-engineering)

### 第 10 篇 经典系统案例

- [x] [GFS：Google 文件系统的架构与一致性模型](./gfs)
- [x] [MapReduce：编程模型与容错机制](./mapreduce)
- [x] [Bigtable：数据模型与 SSTable](./bigtable)
- [x] [Dynamo：最终一致性的键值存储设计](./dynamo)
- [x] [Spanner：TrueTime 与全球分布式数据库](./spanner)
- [x] [Kafka：分布式日志与流存储架构](./kafka)
- [x] [Chubby：基于 Paxos 的锁服务](./chubby)

### 第 11 篇 分布式锁与协调服务

- [x] [分布式锁的正确性问题：租约与 Fencing Token](./distributed-locks-fencing)
- [x] [ZooKeeper 数据模型与 Zab 协议](./zookeeper-data-model-zab)
- [x] [ZooKeeper 的典型应用：选主、配置与队列](./zookeeper-applications)
- [x] [etcd 与基于 Raft 的协调服务](./etcd-raft)
- [x] [Redlock 的争议与基于数据库锁的替代方案](./redlock)

### 第 12 篇 流处理与批处理

- [x] [批处理模型回顾：从 MapReduce 到 Spark/Flink](./batch-processing-spark-flink)
- [x] [流处理语义：事件时间与处理时间](./event-time-processing-time)
- [x] [窗口（Window）与水位线（Watermark）](./windows-watermarks)
- [x] [恰好一次语义（Exactly-Once）的实现](./exactly-once-semantics)
- [x] [流批一体与 Lambda/Kappa 架构](./lambda-kappa-architecture)

### 第 13 篇 分布式系统的形式化验证

- [x] [为什么需要形式化验证](./why-formal-verification)
- [x] [TLA+ 初步：状态机与时序逻辑](./tla-plus-basics)
- [x] [用 TLA+ 描述共识协议](./tla-plus-consensus)
- [x] [模型检验（Model Checking）与 TLC](./model-checking-tlc)
- [x] [PlusCal 与工程实践中的形式化方法](./pluscal-practice)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

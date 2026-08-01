---
pageClass: plain-doc
---

# 分布式系统

本篇对标 MIT 6.824 课程与《数据密集型应用系统设计》（DDIA）的章节体系，系统梳理分布式系统的核心理论与经典工程实践，从通信、时钟、一致性到共识、复制、分区、事务与容错，并逐一剖析 GFS、MapReduce、Spanner、Dynamo 等里程碑系统。

## 主题规划

<ProgressGrid cat="cs/distributed-systems" />


### 第 1 篇 分布式系统概述

- [ ] 什么是分布式系统：动机与定义
- [ ] 分布式系统的核心挑战：部分失败与不确定性
- [ ] 分布式计算的谬误（Fallacies of Distributed Computing）
- [ ] 系统模型：同步、异步与部分异步
- [ ] 故障模型：崩溃故障、拜占庭故障与网络分区
- [ ] 可扩展性、可用性与性能度量
- [ ] CAP 定理及其正确解读
- [ ] BASE 与最终一致性

### 第 2 篇 RPC 与通信

- [ ] 远程过程调用（RPC）的基本原理
- [ ] 接口定义语言（IDL）与序列化：Protocol Buffers 与 Thrift
- [ ] gRPC 的设计与流式调用
- [ ] 消息传递模型：点对点、发布订阅与消息队列
- [ ] 超时、重试与幂等性
- [ ] 网络分区与超时检测的两难
- [ ] REST 与 RPC 的取舍

### 第 3 篇 逻辑时钟与事件排序

- [ ] 物理时钟的局限与时钟同步（NTP）
- [ ] 发生在先关系（happens-before）与偏序
- [ ] Lamport 时钟（Lamport Timestamps）
- [ ] 向量时钟（Vector Clocks）与因果检测
- [ ] 版本向量（Version Vectors）在副本系统中的应用
- [ ] 混合逻辑时钟（HLC）与 TrueTime

### 第 4 篇 一致性问题

- [ ] 一致性模型概览：从强到弱
- [ ] 线性一致性（Linearizability）的定义与验证
- [ ] 顺序一致性（Sequential Consistency）
- [ ] 因果一致性（Causal Consistency）
- [ ] 最终一致性与会话保证
- [ ] 一致性与可用性的权衡实践

### 第 5 篇 共识算法

- [ ] 共识问题的形式化定义与 FLP 不可能定理
- [ ] Paxos 的直观理解：Basic Paxos 详解
- [ ] Multi-Paxos 与 Leader 选举优化
- [ ] Raft 算法详解：Leader 选举
- [ ] Raft 算法详解：日志复制
- [ ] Raft 算法详解：安全性与成员变更
- [ ] Paxos 与 Raft 的对比与工程实现要点
- [ ] 拜占庭容错与 PBFT 简介

### 第 6 篇 复制

- [ ] 复制的目标与难点
- [ ] 主从复制：同步与异步
- [ ] 复制延迟与读己之写一致性
- [ ] 多主复制与写冲突处理
- [ ] 冲突解决：最后写入获胜（LWW）与冲突自由复制数据类型（CRDT）
- [ ] 无主复制：Quorum、读修复与反熵
- [ ] 脑裂（Split-Brain）问题与围栏（Fencing）

### 第 7 篇 分区与再平衡

- [ ] 分区的动机与分区键选择
- [ ] 按键范围分区与按哈希分区
- [ ] 一致性哈希与虚拟节点
- [ ] 次级索引的分区：局部索引与全局索引
- [ ] 再平衡策略：固定分区、动态分区与按节点比例
- [ ] 请求路由与服务发现

### 第 8 篇 分布式事务

- [ ] 分布式事务的动机与局限
- [ ] 两阶段提交（2PC）详解及其阻塞问题
- [ ] 三阶段提交（3PC）与其假设
- [ ] 快照隔离（Snapshot Isolation）与 MVCC
- [ ] 快照隔离的写倾斜（Write Skew）问题
- [ ] 可串行化与 SSI（可串行化快照隔离）
- [ ] Saga 模式与补偿事务
- [ ] Percolator 式乐观分布式事务

### 第 9 篇 容错与故障检测

- [ ] 故障检测器：心跳与超时
- [ ] Phi 累积故障检测器
- [ ] 成员管理：Gossip 协议
- [ ] 检查点（Checkpoint）与日志恢复
- [ ] 状态机复制（State Machine Replication）
- [ ] 混沌工程：故障注入与系统韧性验证

### 第 10 篇 经典系统案例

- [ ] GFS：Google 文件系统的架构与一致性模型
- [ ] MapReduce：编程模型与容错机制
- [ ] Bigtable：数据模型与 SSTable
- [ ] Dynamo：最终一致性的键值存储设计
- [ ] Spanner：TrueTime 与全球分布式数据库
- [ ] Kafka：分布式日志与流存储架构
- [ ] Chubby：基于 Paxos 的锁服务

### 第 11 篇 分布式锁与协调服务

- [ ] 分布式锁的正确性问题：租约与 Fencing Token
- [ ] ZooKeeper 数据模型与 Zab 协议
- [ ] ZooKeeper 的典型应用：选主、配置与队列
- [ ] etcd 与基于 Raft 的协调服务
- [ ] Redlock 的争议与基于数据库锁的替代方案

### 第 12 篇 流处理与批处理

- [ ] 批处理模型回顾：从 MapReduce 到 Spark/Flink
- [ ] 流处理语义：事件时间与处理时间
- [ ] 窗口（Window）与水位线（Watermark）
- [ ] 恰好一次语义（Exactly-Once）的实现
- [ ] 流批一体与 Lambda/Kappa 架构

### 第 13 篇 分布式系统的形式化验证

- [ ] 为什么需要形式化验证
- [ ] TLA+ 初步：状态机与时序逻辑
- [ ] 用 TLA+ 描述共识协议
- [ ] 模型检验（Model Checking）与 TLC
- [ ] PlusCal 与工程实践中的形式化方法

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

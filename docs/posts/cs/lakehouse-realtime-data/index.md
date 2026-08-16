---
pageClass: plain-doc
---

# 数据湖仓与实时流处理（Iceberg/Flink/Kafka）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Kleppmann, "Designing Data-Intensive Applications" (2017)
- Karim et al., "Flink 实战与流处理" 及 Apache Flink 官方文档体系
- Narkhede, Shapira & Palino, "Kafka: The Definitive Guide" (2nd ed., 2021)

## 主题规划

<ProgressGrid cat="cs/lakehouse-realtime-data" />

### 第1篇

- [x] [数据架构演进（数仓→数据湖→湖仓一体的必然性）](./data-architecture-evolution)
- [x] [存储引擎基础（LSM-Tree vs B+Tree、列存 Parquet/ORC）](./storage-engine-lsm-btree-columnar)
- [x] [表格式（Iceberg/Hudi/Delta 的 ACID 与时间旅行）](./table-formats-iceberg-hudi-delta)
- [x] [消息系统（Kafka 日志抽象、副本协议 ISR、精确一次语义）](./kafka-log-abstraction-isr-exactly-once)
- [x] [流处理模型（事件时间/水位线、窗口语义、Exactly-Once）](./stream-processing-event-time-watermarks)
- [x] [Flink 内核（状态后端、Checkpoint 算法、异步屏障）](./flink-internals-state-backend-checkpoint)
- [x] [流批一体（统一 SQL、Kappa 架构的复兴）](./stream-batch-unification-kappa)
- [x] [OLAP 引擎（ClickHouse/Doris/StarRocks 的向量化执行）](./olap-engine-vectorized-execution)

### 第2篇

- [x] [数据编排与治理（调度系统、血缘追踪、数据质量）](./data-orchestration-governance-lineage)
- [x] [实时数仓实践（CDC 入湖、维表关联、双流 Join）](./realtime-warehouse-cdc-lookup-dual-stream-join)
- [x] [AI 时代的数据底座（特征平台、向量索引与湖仓融合）](./ai-data-foundation-feature-store-vector-index)
- [x] [成本与性能工程（存算分离、冷热分层、查询下推）](./cost-performance-engineering)

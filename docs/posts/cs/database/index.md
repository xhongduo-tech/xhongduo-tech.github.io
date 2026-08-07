---
pageClass: plain-doc
---

# 数据库

数据库篇覆盖大学数据库课程的全部内容，以《数据库系统概念》（Database System Concepts, Silberschatz）为主线，外加分布式数据库专题。目标：学完每一章，写完对应的每一节博文。

## 主题规划

<ProgressGrid cat="cs/database" />


### 第一篇 关系数据库与 SQL

#### 第 1 章 数据库系统引论

- [x] [数据库系统的目标：从文件系统到 DBMS](./from-filesystem-to-dbms)
- [x] [数据视图：数据抽象、实例与模式、数据模型](./data-abstraction-and-data-models)
- [x] [数据库语言：DDL 与 DML](./database-languages)
- [x] [关系数据库：表、DML 与数据库设计](./relational-databases)
- [x] [数据库引擎：存储管理器与查询处理器](./database-engine)
- [x] [数据库与应用架构：两层与三层架构](./application-architecture)
- [x] [数据库用户与管理员（DBA）](./database-users-and-dba)
- [x] [数据库系统的历史与发展](./database-system-history)

#### 第 2 章 关系模型

- [x] [关系数据库的结构：关系、元组与属性](./relational-structure)
- [x] [数据库模式与关系实例](./database-schema-and-instances)
- [x] [码：超码、候选码、主码与外码](./keys-super-candidate-primary-foreign)
- [x] [模式图与关系模式设计](./schema-diagram-and-relation-design)
- [x] [关系查询语言概览](./relational-query-language-overview)
- [x] [关系代数：基本运算](./relational-algebra-basic-operations)
- [x] [关系代数：附加运算与扩展运算](./relational-algebra-extra-operations)

#### 第 3 章 SQL 基础

- [x] [SQL 查询语言概览与数据定义（DDL）](./sql-overview-and-ddl)
- [x] [单关系查询：SELECT 基本结构](./sql-single-relation-select)
- [x] [多关系查询：连接与笛卡儿积](./sql-multi-relation-joins)
- [x] [集合运算：并、交、差](./sql-set-operations)
- [x] [聚集函数与分组（GROUP BY / HAVING）](./sql-aggregation-group-by)
- [x] [嵌套子查询与集合成员比较](./sql-nested-subqueries)
- [x] [空值与三值逻辑](./sql-null-and-three-valued-logic)
- [x] [数据库修改：INSERT、UPDATE、DELETE](./sql-modification-insert-update-delete)

#### 第 4 章 中级 SQL

- [x] [连接表达式：内连接、外连接与自然连接](./sql-join-expressions)
- [x] [视图：定义、查询与更新](./sql-views)
- [x] [事务的 SQL 语义](./sql-transaction-semantics)
- [x] [完整性约束：主码、外码、CHECK 与断言](./sql-integrity-constraints)
- [x] [SQL 的数据类型与模式](./sql-data-types-and-schemas)
- [x] [授权：权限、角色与收回](./sql-authorization)
- [x] [视图与授权的递归](./sql-views-and-authorization-recursion)

#### 第 5 章 高级 SQL

- [x] [函数与过程：PL/SQL 风格的存储过程](./sql-functions-and-procedures)
- [x] [触发器：定义、事件与语义](./sql-triggers)
- [x] [递归查询：WITH RECURSIVE](./sql-recursive-queries)
- [x] [高级聚集：窗口函数、排名与分桶](./sql-window-functions)
- [x] [OLAP 操作：CUBE、ROLLUP 与数据立方体](./olap-cube-rollup)

#### 第 6 章 形式化关系查询语言

- [x] [元组关系演算](./tuple-relational-calculus)
- [x] [域关系演算](./domain-relational-calculus)
- [x] [关系代数与关系演算的表达能力等价性](./relational-algebra-calculus-equivalence)

### 第二篇 数据库设计

#### 第 7 章 实体-联系模型（ER 模型）

- [x] [设计过程概览与需求分析](./database-design-process-overview)
- [x] [实体集、属性与码](./er-entity-sets-attributes-keys)
- [x] [联系集与映射基数](./er-relationship-sets-mapping-cardinality)
- [x] [参与约束与弱实体集](./er-participation-constraints-weak-entity)
- [x] [消除冗余：属性设计与复合属性](./er-attribute-design-composite)
- [x] [ER 图符号与示例](./er-diagram-notation)
- [x] [扩展 ER 特性：特化、泛化、聚集](./eer-specialization-generalization-aggregation)
- [x] [从 ER 图到关系模式的转换](./er-to-relational-mapping)
- [x] [数据库设计的其他问题与 UML](./database-design-uml)

#### 第 8 章 规范化理论

- [x] [好的关系设计的特点与反例](./good-relation-design-characteristics)
- [x] [函数依赖：定义、闭包与平凡依赖](./functional-dependencies)
- [x] [码与函数依赖的关系](./keys-and-functional-dependencies)
- [x] [范式概览：1NF、2NF、3NF、BCNF](./normal-forms-overview)
- [x] [函数依赖理论：Armstrong 公理与正则覆盖](./armstrong-axioms-canonical-cover)
- [x] [无损连接分解与依赖保持](./lossless-join-and-dependency-preserving)
- [x] [BCNF 分解算法](./bcnf-decomposition-algorithm)
- [x] [第三范式分解算法](./3nf-decomposition-algorithm)
- [x] [多值依赖与第四范式](./multivalued-dependencies-4nf)
- [x] [其他范式与设计权衡：反规范化](./denormalization-and-design-tradeoffs)

### 第三篇 数据存储与查询引擎

#### 第 9 章 数据存储与文件组织

- [x] [物理存储介质概览：磁盘、SSD、内存层次](./physical-storage-media)
- [x] [磁盘结构与磁盘块存取的代价](./disk-structure-and-access-cost)
- [x] [文件组织：堆文件与顺序文件](./file-organization-heap-sequential)
- [x] [数据字典与系统目录](./data-dictionary-system-catalog)
- [x] [缓冲区管理：缓冲池与替换策略](./buffer-management)
- [x] [列式存储与行式存储的组织方式](./row-vs-columnar-storage)

#### 第 10 章 索引与哈希

- [x] [索引基本概念：顺序索引与辅助索引](./index-basics)
- [x] [B+ 树的结构与查找](./bplus-tree-structure-search)
- [x] [B+ 树的插入与删除](./bplus-tree-insertion-deletion)
- [x] [B+ 树变体：B 树、B\* 树与内存优化](./btree-variants)
- [x] [LSM 树：原理、Compaction 与读写放大](./lsm-tree)
- [x] [静态哈希：哈希函数与桶溢出](./static-hashing)
- [x] [动态哈希：可扩展哈希与线性哈希](./dynamic-hashing)
- [x] [位图索引与其他辅助索引](./bitmap-indexes)

#### 第 11 章 查询处理

- [x] [查询处理步骤与执行计划](./query-processing-steps)
- [x] [查询代价的度量](./query-cost-measurement)
- [x] [选择运算的算法：线性扫描与索引扫描](./selection-algorithms)
- [x] [排序：外部归并排序](./external-merge-sort)
- [x] [连接运算：嵌套循环与块嵌套循环](./nested-loop-join)
- [x] [连接运算：索引嵌套循环、归并连接与哈希连接](./join-algorithms-index-merge-hash)
- [x] [聚集与去重的实现](./aggregation-and-distinct)
- [x] [表达式求值：物化与流水线](./expression-evaluation-materialization-pipelining)
- [x] [内存中查询与列存执行模型](./in-memory-query-columnar-execution)

#### 第 12 章 查询优化

- [x] [查询优化概览：逻辑优化与物理优化](./query-optimization-overview)
- [x] [关系表达式的转换规则（等价变换）](./relational-equivalence-rules)
- [x] [查询重写：谓词下推与连接顺序](./query-rewriting-predicate-pushdown)
- [x] [统计信息与代价估算：基数估计](./statistics-and-cardinality-estimation)
- [x] [代价估算：选择率、直方图与采样](./cost-estimation-selectivity-histogram)
- [x] [基于代价的优化器与动态规划枚举](./cost-based-optimizer-dp)
- [x] [物化视图与查询结果缓存](./materialized-views)

### 第四篇 事务与故障恢复

#### 第 13 章 事务

- [x] [事务概念与 ACID 特性](./transaction-concept-acid)
- [x] [事务状态模型](./transaction-state-model)
- [x] [并发执行的必要性与问题](./concurrency-problems)
- [x] [可串行化：冲突可串行化与视图可串行化](./serializability)
- [x] [可恢复调度与无级联调度](./recoverable-schedules)
- [x] [事务隔离级别：读未提交到可串行化](./isolation-levels)

#### 第 14 章 并发控制

- [x] [基于锁的协议：共享锁与排他锁](./lock-based-protocols)
- [x] [两阶段锁协议（2PL）及其变体](./two-phase-locking)
- [x] [死锁处理：检测、预防与等待图](./deadlock-handling)
- [x] [锁的粒度与意向锁](./lock-granularity-intention-locks)
- [x] [多版本并发控制（MVCC）：快照隔离](./mvcc-snapshot-isolation)
- [x] [快照隔离的写偏斜问题](./write-skew)
- [x] [时间戳排序协议](./timestamp-ordering)
- [x] [乐观并发控制（有效性检查）](./optimistic-concurrency-control)
- [x] [谓词读、幻影现象与索引锁](./phantom-problem-index-locks)

#### 第 15 章 恢复系统

- [x] [故障分类与存储器层次](./failure-classification)
- [x] [基于日志的恢复：WAL（预写式日志）](./wal-log-based-recovery)
- [x] [延迟修改与立即修改](./deferred-vs-immediate-modification)
- [x] [检查点机制](./checkpointing)
- [x] [恢复算法：撤销（UNDO）与重做（REDO）](./undo-redo-recovery)
- [x] [缓冲管理策略：STEAL 与 NO-FORCE](./steal-no-force)
- [x] [ARIES 恢复算法：日志记录、分析与重做阶段](./aries-log-analysis-redo)
- [x] [ARIES 恢复算法：撤销阶段与补偿日志记录（CLR）](./aries-undo-clr)
- [x] [模糊检查点与高可用：备份与远程容灾](./fuzzy-checkpoint-high-availability)

### 第五篇 分布式数据库专题

#### 第 16 章 复制与共识

- [x] [复制的动机：可用性、延迟与读扩展](./replication-motivation)
- [x] [主从复制与多主复制](./single-multi-leader-replication)
- [x] [复制日志与复制滞后](./replication-logs-lag)
- [x] [共识问题与 FLP 不可能性](./consensus-flp-impossibility)
- [x] [Paxos 算法：角色、提案与多数派](./paxos-algorithm)
- [x] [Raft 算法：领导者选举与日志复制](./raft-algorithm)
- [x] [Raft 的成员变更与安全保证](./raft-membership-change-safety)

#### 第 17 章 分片与分布式事务

- [x] [数据分区：范围分片、哈希分片与一致性哈希](./data-partitioning)
- [x] [再平衡与路由：元数据管理](./rebalancing-routing)
- [x] [分布式事务与原子提交问题](./distributed-transactions-atomic-commit)
- [x] [两阶段提交（2PC）：流程与故障处理](./two-phase-commit)
- [x] [三阶段提交（3PC）与其局限](./three-phase-commit)
- [x] [TCC：Try-Confirm-Cancel 补偿模型](./tcc-compensation-model)
- [x] [Saga 模式：编排式与协同式](./saga-pattern)
- [x] [分布式死锁检测与全局快照](./distributed-deadlock-global-snapshot)

#### 第 18 章 CAP 与一致性模型

- [x] [CAP 定理：含义与常见误解](./cap-theorem)
- [x] [线性一致性与顺序一致性](./linearizable-sequential-consistency)
- [x] [因果一致性与会话保证](./causal-consistency-session-guarantees)
- [x] [最终一致性与收敛](./eventual-consistency)
- [x] [一致性模型的谱系与取舍](./consistency-model-spectrum)

#### 第 19 章 NoSQL 与 NewSQL

- [x] [NoSQL 的兴起与数据模型分类](./nosql-rise-and-data-models)
- [x] [键值存储与文档数据库](./key-value-document-stores)
- [x] [宽列存储：BigTable 与 Cassandra](./wide-column-stores)
- [x] [图数据库与图查询](./graph-databases)
- [x] [NewSQL：横向扩展的关系数据库](./newsql)
- [x] [Google Spanner：TrueTime 与外部一致性](./google-spanner-truetime)
- [x] [TiDB：架构、TiKV 与分布式 SQL 执行](./tidb-architecture)

#### 第 20 章 列存与 OLAP

- [x] [OLTP 与 OLAP 的工作负载差异](./oltp-vs-olap)
- [x] [列式存储：压缩、延迟物化与向量化执行](./columnar-storage-vectorized)
- [x] [数据仓库、数据湖与数据湖仓](./data-warehouse-lakehouse)
- [x] [星型模式与雪花模式](./star-snowflake-schema)
- [x] [列存数据库实践：ClickHouse 与 DuckDB](./clickhouse-duckdb)
- [x] [HTAP：混合事务分析处理](./htap)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

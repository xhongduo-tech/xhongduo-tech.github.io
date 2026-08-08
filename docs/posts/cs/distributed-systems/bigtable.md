---
title: Bigtable：数据模型与 SSTable
date: 2026-08-07
---

# Bigtable：数据模型与 SSTable

<div class="epigraph">
<p>Bigtable 是一张「稀疏的、分布式的、持久化的多维有序映射表」——Google 几乎一半产品的数据都住在里面。</p>
<footer>—— 张帆（Fay Chang）等，Bigtable: A Distributed Storage System for Structured Data，OSDI 2006</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ MIT 6.824 第5讲 / Bigtable 论文 2006 ｜ 2026-08-07</p>
</div>

## 为什么从 Bigtable 开始

GFS 存文件、MapReduce 算数据，可 Google 的多数产品（搜索、地图、Gmail）需要的是「结构化数据的随机读写」——这正是 Bigtable 的领地。**Bigtable** 在 GFS 之上构建了一张**多维稀疏映射表**，用**SSTable** 组织存储，成为 HBase、Cassandra 数据模型的共同祖先。<span class="marginnote">Bigtable 论文（Chang et al., OSDI 2006）描述了 Google 内部使用最广泛的存储系统：从 Web 索引到地图数据、从邮件到数据分析，几百个服务跑在它上面。它确立了「LSM 树 + 列式稀疏表」的 NoSQL 范式，HBase（开源克隆）、Cassandra、甚至 TiKV 都是它的后代。</span>

## 1 数据模型：一张巨大的稀疏表

Bigtable 的数据模型是一个**稀疏的、分布式的、多维有序映射**：

$$
(row:\ \mathrm{string},\ column:\ \mathrm{string},\ timestamp:\ \mathrm{int64}) \mapsto \mathrm{string}
$$

- **行键（row key）**：主键，按字典序排序——行键决定数据的物理位置（按行键范围分区到 tablet）。
- **列族（column family）**：列的集合，如 `anchor`（外链）、`contents`（页面内容）——列族是访问与存储的基本单元。
- **时间戳（timestamp）**：同一单元格可存多个版本，按时间戳排序——最新版本默认返回。

**稀疏**是关键：一行可以有任意多列，没有的列不占存储——所以「用户表」可以给每个用户存任意多的属性，不必预先定义 schema。

Bigtable 的行键设计直接影响性能：**相邻行键的数据落在同一 tablet**（按范围分区），所以「批量取同一前缀的行」是单 tablet 操作——例如把网页 URL 反转后作行键（`com.cnn.www`），同域名的页面就聚集在一起。<span class="marginnote"><strong>辨析｜易错点：</strong>Bigtable 不是「关系表」——它没有 SQL、没有 JOIN、没有强 schema。它是「宽列存储」（wide-column store）：行可以有无穷多列，列按族组织。Cassandra 的 column family、HBase 的 column family 都是这个模型的直接继承——「稀疏宽表」是它与关系型（行存）和键值（纯 KV）的本质区别。</span>

## 2 底层存储：SSTable

**SSTable（Sorted String Table，排序字符串表）**是 Bigtable 的存储基石：一个**按键排序的、不可变的、键值对集合**文件。三个性质：

**有序**：键按字典序排列——支持二分查找、支持「范围扫描」。
**不可变**：写入后不再修改——天然支持顺序写、无随机写、无碎片。
**紧凑**：一次顺序写生成，压缩率高。

SSTable 怎么支持「随机读写」？答案是 **LSM 树（Log-Structured Merge Tree）**结构：

**写入**：追加到内存中的 **memtable**（有序缓冲），memtable 满后刷成 SSTable（顺序写，快）。
**读取**：先查 memtable，再查最近的 SSTable…… 一层层查（可用布隆过滤器跳过无关层）。
**合并**：后台**压缩（compaction）**把多个 SSTable 合并成一个——清理冗余、删除过期数据。

这个设计的精髓：**把「随机写」变成「顺序写」**（写入只碰内存 + 顺序落盘），换取极高的写入吞吐；代价是「读要查多层」（读放大）与「后台压缩」的开销。LSM 树是 Bigtable/HBase/Cassandra/RocksDB 吞吐的秘密，也是它们与 B+ 树数据库（MySQL）在读写性格上的分野。<span class="marginnote">LSM 树的「读放大 vs 写放大」是存储引擎设计的核心权衡：SSTable 层数多则写快读慢，合并频繁则读快写慢。RocksDB（LSM 树的极致工程化）把「层数、合并策略、布隆过滤器」全部做成可调参数——理解 LSM 树，就理解了现代分布式存储引擎的性能心智模型。</span>

## 3 分布式架构：Tablet 与三层定位

Bigtable 的分布式结构把数据切成**tablet**（行键范围的分片），用三层系统定位：

**Chubby 服务**（基于 Paxos 的锁服务，第 10 篇 Chubby 展开）：保存元数据根表位置，提供分布式锁与选主——Bigtable 的「引导与协调」层。
**主节点（master）**：管理 tablet 分配、负载均衡、垃圾回收。master 不承担数据读写——数据流直接走 tablet server。
**Tablet server**：每个管理若干 tablet，处理其上的读写请求。

定位流程（三层索引）：客户端先问 Chubby 拿「根 tablet 位置」→ 根 tablet 指到「元数据 tablet」→ 元数据 tablet 指到「用户 tablet」。这棵「B+ 树式」的定位索引让任意行键都能在常数级跳数内被找到。

**tablet 的分裂与合并**：tablet 超过阈值自动分裂成两个（动态分区，第 7 篇讲过）；master 负责把 tablet 分配到各 tablet server 做负载均衡——这是 Bigtable 可扩展性的机制。<span class="marginnote">Bigtable 的三层定位（Chubby → 根 tablet → 元数据 tablet → 用户 tablet）是「分级索引」的经典工程案例——它把「任意键的定位」从「全局广播」变成「常数跳数的查找」。后来 HBase 的 ZooKeeper → root → META 结构几乎原样继承了它。</span>

## 4 一致性模型与事务能力

- **单行事务**：Bigtable 支持**单行的原子读-修改-写**（受 tablet server 上的锁与日志保护）——跨行事务不支持（当时）。
- **列族级控制**：访问控制（ACL）按列族配置——敏感列（如密码）与普通列有不同的权限。
- **弱一致场景的补救**：跨行一致性留给应用层（后来 Spanner 才解决跨行强一致）。

**单行原子 + 稀疏宽表** 的组合支撑了 Google 大量「单实体为中心」的应用（一个用户、一个网页、一个文档的所有数据在一行里）——这再次印证第 7 篇的原则：**把「一起操作的数据」设计进同一行/同一分区**，跨行事务的需求就被消化了。

## 5 公式解析：LSM 树的写入放大

把「LSM 树为什么写入快」量化。设 memtable 容量 $M$，SSTable 每层容量放大因子 $k$（如 10），写一次随机写转化为顺序写：

$$
\text{写放大（compaction）} \approx \frac{k}{k-1} \approx 1.1 \text{（k=10 时）}, \qquad
\text{读放大（查层数）} \approx O(\log_k N)
$$

拆解：

- **写放大**：每次数据最终被压缩 $\log_k N$ 次左右，但每次都是**顺序写**——顺序写的吞吐比随机写高 1–2 个数量级，所以「放大但顺序」仍然快。
- **读放大**：读要查 memtable + 每层 SSTable，最坏 $\log_k N$ 层；布隆过滤器把「某层必然没有」的查询 O(1) 跳过，实际读放大远小于理论值。
- **关键对比**：B+ 树（MySQL）读快（O(log) 单次定位）但随机写慢（页级随机 IO）；LSM 树写快（顺序落盘）但读要多层（靠布隆过滤补偿）。
- **工程读数**：LSM 树是「用读的复杂性换写的吞吐」——**写入密集的负载选 LSM，读密集 + 点查负载选 B+ 树**，没有绝对优劣。

这条式子的工程含义：**Bigtable 及其后代（HBase/Cassandra/RocksDB）是「写优化」的存储引擎**——它们的架构假设是「写入比读取更难伺候」（日志、事件、爬虫数据都是写密集）。理解写放大与读放大的互换，你就理解了为什么不同数据库引擎有截然不同的性能性格。<span class="marginnote">这也可以解释 NoSQL 与关系型的另一个分野：关系型（InnoDB）为「事务性点查」优化（B+ 树 + 随机写也能扛，因为有 WAL 缓冲），NoSQL 为「海量追加写」优化（LSM 树 + 顺序写）。选数据库引擎前先看你的负载是「写密集」还是「读密集 + 事务」。</span>

## 6 小结

- **Bigtable** 是「稀疏、分布式、多维有序映射表」：以 (row, column, timestamp) 三元组定位一个单元格，值是不解释的字节串。
- **SSTable**（有序不可变文件）+ **LSM 树**（内存缓冲 + 多层落盘 + 后台压缩）——把随机写变顺序写，写吞吐极高。
- 分布式架构：Chubby（引导）→ master（tablet 分配）→ tablet server（数据读写）；三层定位索引。
- tablet 按行键范围分区，自动分裂合并——动态分区 + 负载均衡。
- **单行原子事务** + 稀疏宽表：把「一起操作的数据」放进同一行，消化跨行事务需求。
- 写放大 vs 读放大：LSM 用读的复杂性换写的吞吐——写密集负载的引擎选择。

在下一节，我们看「最终一致」路线的另一座里程碑——**Dynamo**，Google 的对立面亚马逊如何设计去中心化存储。

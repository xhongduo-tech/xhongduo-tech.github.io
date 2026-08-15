---
title: Percolator 式乐观分布式事务
date: 2026-08-07
---

# Percolator 式乐观分布式事务

<div class="epigraph">
<p>给每个键配一把锁、给每个事务配一个时间戳——Percolator 用「锁 + 版本」在无中心协调下把跨行事务做成了现实。</p>
<footer>—— 参照 Peng & Dabek 2010（Large-scale Incremental Processing Using Distributed Transactions and Notifications）</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ Percolator 论文 2010 / TiDB 事务模型 ｜ 2026-08-07</p>
</div>

## 为什么从 Percolator 开始

2PC 阻塞、Saga 无原子性——有没有第三条路？**Percolator** 是 Google 2010 年公开的分布式事务实现：它不引入独立的协调者，而是**把事务信息写进数据本身**——每个键旁边存锁、存版本、存写记录，靠 Bigtable 的单行原子性做基础，实现跨行、跨表的可串行化事务。TiDB、CockroachDB（早期）、很多 NewSQL 的事务层都借鉴了它。<span class="marginnote">Percolator 论文的标题是《用分布式事务做大规模增量处理》——它最初是 Google 网页索引的增量更新系统，却因为事务机制太经典而被广泛学习。它的核心洞察：<strong>用「分布式锁表」把 2PC 的协调逻辑摊进数据面</strong>，避免独立的协调者。</span>

## 1 基础：Bigtable 的单行原子性

Percolator 构建在 **Bigtable** 之上。Bigtable 保证**单行内的读写是原子的**（一个行键的一列族内的操作不可分割）。Percolator 的全部技巧，就是把这个「单行原子性」放大成「跨行事务」——它把「锁」和「事务元数据」作为**普通的列**存进数据行，单行原子性让「读锁、写数据、清锁」成为原子操作。

每行数据有三组列：

**data 列**：真正的业务数据（value）。
**lock 列**：该行当前的锁——记录「哪个事务锁着这行、锁的类型（写锁/读锁）」。
**write 列**：该行已提交的版本记录——指向 data 列的某个版本。

事务的操作全部通过「读写这几个列」完成——协调逻辑被「物化」成数据，这就是 Percolator 不需要独立协调器的秘密。

## 2 事务流程：两阶段提交的 Percolator 版

一个 Percolator 事务分两个阶段，但协调者是「事务自己」而非独立节点。

**阶段一：写（Prewrite，准备）**

1. 事务为每个要写的键分配一个**全局递增的 start_ts（开始时间戳）**——作为快照点。
2. 对每个键执行 **Prewrite**：检查该键的 lock 列是否为空（无冲突）→ 把新值写入 data 列（带 start_ts 版本）→ 把 lock 列设为「我锁着这行」（记录事务 ID 与 start_ts）。
3. 任一键 Prewrite 失败（有锁）→ 整个事务中止。

**阶段二：提交（Commit）**

4. 事务选一个键作为**主键（primary）**，其余为从键（secondary）。
5. 先提交主键：把主键的 lock 列改为 write 列（记录 commit_ts，删除锁）——**这一步是原子的**。
6. 再提交所有从键：把各从键的 lock 改为 write。
7. 主键提交成功后，事务已「生效」；从键提交失败可由后台进程**继续补完**——因为主键的 write 记录了事务已提交，从键必须跟上。

**关键设计：主键 = 事务的「真相」**。任何事务/进程看到主键的 write 存在，就知道整个事务已提交，可以安全地补完从键或让读看到结果；主键的 lock 还在，就知道事务可能未完成。<span class="marginnote"><strong>辨析｜易错点：</strong>Percolator 的主键机制是「2PC 协调者的去中心化替代」：2PC 的协调者决策「commit/abort」存在协调者日志里，Percolator 把同样的决策存在主键的 lock/write 列里——决策信息从「协调者私有状态」变成「数据面的公共状态」。任何节点都能读主键判断事务状态，这就是「无独立协调者」的原理。</span>

## 3 冲突检测与事务中止

**写-写冲突**：Prewrite 时发现目标键已有 lock（别的活跃事务锁着）→ 提交失败。若有死锁（两个事务互相等锁），靠**锁的超时**打破——锁有持有者信息，超时后其他事务可以**接管并回滚**死锁事务。
**读-写冲突**：读事务按 start_ts 读快照——读到该版本的 write 记录即可，不碰 lock。若读到的是「已写 data 但未提交」的版本（有 lock），读事务**等待锁释放**或读到旧版本——具体策略由隔离级别决定。

事务中止（Rollback）：把已 Prewrite 的键的 data 版本与 lock 清除——与提交一样，先清主键的锁（让「是否中止」有唯一真相），再清从键。

## 4 Percolator 的代价与适用

Percolator 不是免费午餐：

**写放大**：每个键的每次事务写要写 data、lock、write 三处——Bigtable 的存储与 IO 开销翻倍。
**延迟**：每步都要访问 Bigtable（多次 RPC）；Prewrite 遍历所有键、Commit 再遍历——大事务延迟显著。
**后台清理**：锁超时接管、从键补完、孤儿锁清理都需要后台进程——实现复杂。
**可用性**：依赖底层 Bigtable 的可用性；Percolator 本身不提供跨数据中心事务。

**适用场景**：Google 用它做网页索引的增量更新——**大批量、低冲突、可后台重试**。TiDB 用它做 OLTP 事务——**读多写少、冲突稀疏**。Percolator 是「乐观 + 可恢复」的典范：冲突少时吞吐极高，冲突多时靠回滚与重试兜底。<span class="marginnote">Percolator 对 NewSQL 的影响深远：TiDB 的 Percolator 事务层、CockroachDB 的「并行提交（parallel commits）」优化，都在它的骨架上改进——主要优化「从键提交的多轮 RPC」（用异步 + 并行压缩成一轮）。理解 Percolator 就理解了 TiDB 事务的底层逻辑。</span>

## 5 公式解析：Percolator 的快照隔离保证

Percolator 的隔离级别是**可重复读/快照隔离**（论文称为「可串行化」但实际是快照隔离的变体）。核心是 start_ts/commit_ts 两个时间戳的配合：

$$
\text{读操作 } R \text{ 看到版本 } v \iff v.\text{start\_ts} \le R.\text{start\_ts} < v.\text{commit\_ts}
$$

拆解：

- 每个版本 $v$ 有 [start_ts, commit_ts] 区间：start_ts 是写入事务的开始，commit_ts 是提交时刻。
- 读事务 $R$ 只看到「$R$ 开始之前已提交」的版本：$v.\text{commit\_ts} \le R.\text{start\_ts}$。
- 正在提交中的版本（$v.\text{commit\_ts}$ 尚未来得及写）对 $R$ 不可见——快照冻结。
- 两个时间戳的**全局单调性**由「时间戳分配器」（一个单点或 HLC）保证——这是 Percolator 唯一的「中心化」部件。

这条式子的工程含义：**Percolator 的一致性基石 = 全局单调时间戳 + 单行原子提交**。时间戳提供快照的「世界线」，主键的原子提交提供「决策的真相」。两者合起来，跨行事务的一致性就从「协调者协商」退化为「时间戳 + 原子位」——这就是它不需要独立 2PC 协调者的完整逻辑。<span class="marginnote">时间戳分配器是 Percolator 的隐藏单点：Google 用一套中心化的时间戳服务器，TiDB 用 PD（Placement Driver）分配 TSO——它必须单调、快速、可用。这也是 NewSQL 的共性：<strong>强一致事务总需要一个「单调性源头」</strong>，区别只是这个源头是显式的协调者还是隐式的时间戳。</span>

## 6 数值走查：一次跨行转账事务

把流程落到具体数据上。设要执行「从账户 $X$ 转 100 元到账户 $Y$」的事务，两个键 `X`、`Y` 初始值均为 500，start_ts = 42。

**Prewrite 阶段**：

1. 事务为 `X` 写 data(X, 400, ver=42)，设 lock(X) = 「事务 T，start_ts=42」；为 `Y` 写 data(Y, 600, ver=42)，设 lock(Y) 同理。
2. 若 `Y` 已被别的活跃事务锁住 → Prewrite 失败 → 整个事务中止，`X` 的预写与锁回滚。
3. 假设两键都无锁，预写成功。

**Commit 阶段**：选 `X` 为主键。

1. 提交主键：把 lock(X) 原子地改写为 write(X, commit_ts=45)——此刻事务**已生效**。
2. 提交从键：把 lock(Y) 改为 write(Y, commit_ts=45)。
3. 若第 2 步的 RPC 失败（比如 `Y` 所在节点暂时不可达），后台进程看到主键 `X` 已有 write → 知道事务已提交 → **补完** `Y` 的提交。

**读侧视角**：一个 start_ts=44 的读事务读 `Y`：`Y` 的 write(commit_ts=45) 大于 44，所以 45 版本对 44 的读不可见——读到旧值 500。而 start_ts=46 的读事务能看到 commit_ts=45 的版本，读到 600。

这个走查把三个要点钉在一起：**主键是原子决策点**（write 一写事务即生效）；**从键可补完**（后台进程依据主键状态续命）；**快照靠时间戳**（commit_ts 与读事务 start_ts 的先后决定可见性）。Percolator 的全部一致性，就在这三个要点的配合里。

### Percolator 与 2PC 的对照

把 Percolator 与 2PC 放一起看，两者的「骨架」惊人相似：Percolator 的 Prewrite ≈ 2PC 的 prepare（投票意愿），Commit 主键 ≈ 2PC 的 commit 决策，主键原子提交 ≈ 2PC 协调者的持久化决策。差别在于：**2PC 的协调者是独立节点，Percolator 的「协调者」是主键本身**——决策信息从协调者日志搬到了数据面。这带来两个直接好处：没有协调者单点（主键状态可被任何节点读取），从键可异步补完（2PC 的参与者必须同步等协调者）。代价是「锁 + 版本」的写放大——这是「去中心化协商」的典型取舍。

## 7 小结

- **Percolator** 把 2PC 的协调逻辑「物化」进数据面：锁、版本、写记录作为普通列存进 Bigtable。
- 流程：**Prewrite（写 data + 锁）→ 选主键 → 提交主键（原子）→ 补完从键**。
- **主键 = 事务真相**：主键的 lock/write 状态让任何节点都能判断事务是否已提交——无独立协调者。
- 冲突：写-写靠 lock 冲突检测，读按快照时间戳，死锁靠锁超时接管回滚。
- 代价：写放大、多次 RPC、后台清理复杂；适用「低冲突、可重试」场景。
- 一致性基石 = 全局单调时间戳 + 单行原子提交；TiDB 等 NewSQL 的底层逻辑。

至此，分布式事务篇收官。下一章进入**容错与故障检测**——从心跳超时到 Gossip，看系统如何「感知死亡」并「自我修复」。

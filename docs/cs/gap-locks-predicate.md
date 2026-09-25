---
title: 间隙锁与谓词锁
date: 2026-09-08
section: cs
---

# 间隙锁与谓词锁

<div class="epigraph">
<p>幻象来自范围内新插入。谓词锁锁住「满足条件的未来行」；间隙/next-key 用索引空隙近似。</p>
<footer>—— 据 Eswaran et al. 谓词锁；InnoDB next-key；主干幻读课的锁实现</footer>
</div>

[上一课](/cs/intention-locks)把锁挂在层次上。本课不画 IX 矩阵。缺口是幻读：主干已定义；2PL 行锁锁不住「还不存在的行」。谓词锁（predicate lock）在理论上锁 $\sigma_\theta$；工程用索引间隙锁、next-key 锁近似。

## 问题

`SELECT … WHERE id BETWEEN 10 AND 20` 可重复读下，另一事务插入 id=15 会幻。谓词锁：对 $\theta$ 加锁，插入者检查新行是否满足任何活跃谓词。实现难：谓词任意、相交判定贵。间隙锁：在 B+ 叶上锁 (10,20) 之间的间隙，插入必须过间隙锁。next-key：间隙+记录，避免幻与丢失更新的组合缝。

无索引：可能退回表级锁。这是计划与并发的耦合——没有合适索引，可串行更贵。

<span class="marginnote">Eswaran, Gray, Lorie, Traiger，CACM 1976 事务一致与谓词。InnoDB REPEATABLE READ 实际用 next-key，与标准 RR 不完全同。本课机制。</span>

## 方法

扫描带索引条件则对碰到的间隙加锁（隔离级别够高时）。插入：定位叶，检查间隙冲突。哈希索引无序，间隙难定义，幻象保护弱或表锁。OCC/SSI 用读范围记录代替间隙锁。

死锁：间隙锁与行锁互相等待，wait-for 要包含它们。

```mermaid
flowchart TD
  RNG["范围扫描"] --> GAP["锁住叶上间隙"]
  INS["插入"] --> CHK["过间隙锁"]
  PRED["任意谓词"] --> TBL["或表锁 / 真谓词锁"]
```

## 机制

性能：范围越大锁越多，并发下降。这是「可串行的税」。覆盖扫描仍要锁间隙，不只盖列。分区：间隙在分区内，跨分区插入要各间隙或更粗锁。

一个具体的插入是怎么被间隙锁挡住的？关键在「锁的是空隙，不是行」。

```mermaid
flowchart TD
  EX["表里已有 id=10 与 id=20"] --> SCAN["事务 A：SELECT id BETWEEN 10 AND 20"]
  SCAN --> LOCK["A 锁住 (10,20) 之间的间隙"]
  INS["事务 B：INSERT id=15"] --> POS["定位到同一间隙"]
  POS --> WAIT["撞上 A 的间隙锁：等待"]
  ALT["若 B 插的是 id=25"] --> OTHER["落在别的间隙，不受阻"]
```

<span class="marginnote">直觉类比：行锁是给某张已摆好的桌子挂「已订」牌；间隙锁是把整块区域圈起来，不许任何新桌子摆进来——哪怕要防的「桌子」（id=15 那行）现在还不存在。幻读防的就是这些未来的行。</span>

<span class="marginnote">为什么重要：如果查询列上没有索引，InnoDB 找不到可用的窄间隙，可能退化为锁住更大范围甚至全表——一个范围查询就把所有插入堵死。并发表现直接取决于索引设计，不只是隔离级别设置。</span>

<span class="marginnote">常见误区：初学者容易以为「设了 REPEATABLE READ 就绝没有幻读」。InnoDB 的普通快照读靠 MVCC 看不见新行，但 SELECT … FOR UPDATE 这类当前读必须靠 next-key 锁挡住并发插入，两套机制各管一段。</span>

与 MVCC：快照读可避免部分幻，写偏斜与插入幻仍要 SSI 或间隙。Postgres RR 用快照，serializable 用 SSI，间隙模型与 InnoDB 不同。

## 边界

本课不画 wait-for 算法。也不把间隙当空间 R 树锁。谓词完全一般不可判定相交，工程必近似。

后课默认：防幻要锁范围或记谓词；无索引则粗锁。wait-for 图：检测锁等待环。

间隙是索引序上的谓词近似，不是 SQL 谓词的完备实现。

## 小结

- 谓词锁理论防幻；间隙/next-key 用 B+ 空隙实现。
- 无序索引与无索引使保护退化。
- wait-for 图下一课：死锁检测。
- 出处：Eswaran et al. 1976；InnoDB next-key；Gray and Reuter。

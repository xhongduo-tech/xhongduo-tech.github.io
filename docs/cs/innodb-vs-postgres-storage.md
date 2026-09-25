---
title: InnoDB 对 Postgres 堆
date: 2026-09-08
section: cs
---

# InnoDB 对 Postgres 堆

<div class="epigraph">
<p>InnoDB 主键即聚簇：行跟在主键叶上。Postgres 堆是无序页，索引一律二次查找。两种页合同决定回表、覆盖与 vacuum 的形状。</p>
<footer>—— 据 Ramakrishnan and Gehrke 聚簇；InnoDB 与 PostgreSQL 存储文档；Gray</footer>
</div>

[上一课](/cs/tde-compression)把加密放在 I/O 路径。本课打开引擎对照：同样关系代数，行放哪。缺口不是再定义 B+，而是两条工业路径——聚簇索引表 vs 堆+二级索引——对覆盖、范围扫描、MVCC 版本落点的影响。细节 undo/redo、vacuum 后课补。

## 问题

聚簇（索引组织）：按主键序存整行，主键范围扫描是顺序 I/O，二级索引存主键（再回聚簇）或存 RID。堆：插入常追加，物理序与主键无关，主键也是二级结构，RID 指堆槽。缺口是**默认访问路径**：`WHERE pk BETWEEN` 在 InnoDB 走叶链；在 Postgres 走主键索引再随机堆，除非额外 `CLUSTER`（一次性，不维持）。

<span class="marginnote">数字实例：`WHERE pk BETWEEN 1 AND 10000`——InnoDB 沿聚簇叶链顺序读，接近顺序 I/O；Postgres 从索引拿到一串按索引序、不按堆序的 TID，到堆里最坏是上万次随机跳转。同一条 SQL，物理形状完全不同。</span>

更新主键在聚簇里等于搬行；堆上改主键只改索引项。宽行、溢出页两边都有，但叶页变宽使 InnoDB 扇出下降。

<span class="marginnote">教材里 index-organized vs heap-organized。本课用两家引擎当实例，不把版本号当理论。覆盖索引课已有逻辑，这里是物理落点。</span>

## 方法

选主键时，InnoDB 要认真：单调键减少页分裂；UUID 随机插入打散叶、碎缓冲。Postgres 主键选择不决定堆序，要用 `BRIN`/分区/CLUSTER 表达序。二级索引：InnoDB 回表是回主键；Postgres 回表是回堆——随机 I/O 模型不同，校准分开。

MVCC：Postgres 版本在堆行头（多版本行）；InnoDB 旧版本常进 undo，聚簇叶上是当前行。下一课与 vacuum 课分别钉。

```mermaid
flowchart TD
  INN["InnoDB"] --> CL["主键叶上即行"]
  INN --> SEC["二级键到主键"]
  PG["Postgres"] --> HEAP["堆槽存行版本"]
  PG --> IDX["所有索引到 TID"]
```

## 机制

缓冲池：聚簇把行与主键叶同一页，点查 pk 少一次 I/O；二级仍两次。堆点查主键两次（索引+堆）。延迟物化：堆上 TID 列表再堆取，即 bitmap heap scan 一类。

锁：聚簇间隙与记录锁贴在主键序上；堆上锁贴 TID 与索引。后课间隙锁。

同一条二级索引点查，在两家引擎里各怎么走：

```mermaid
flowchart TD
  Q["二级索引等值查询"] --> I1["InnoDB：查二级树得到主键值"]
  I1 --> I2["回聚簇树：沿主键叶取整行"]
  Q --> P1["Postgres：查索引得到 TID"]
  P1 --> P2["按 TID 跳到堆槽取行"]
  I2 --> R["返回行"]
  P2 --> R
```

<span class="marginnote">直觉类比：聚簇表像按拼音排架的书库，书就立在目录所指的位置上；堆表像仓库散堆货物，每张索引卡只写「第几排第几格」，取货得按格子单独跑一趟。目录越准，跑腿次数越少。</span>

## 边界

本课不讲 doublewrite。也不把 MyISAM 写进对照。索引组织表课会抽象 InnoDB 这一侧，Oracle IOT 点名。

后课默认：谈到行位置，先问聚簇还是堆。undo/redo 与 doublewrite：日志与页撕裂如何配这两套页。

没有谁绝对更快；范围主键 vs 随机主键把结论反转。

<span class="marginnote">常见误区：初学者容易把「Postgres 主键不决定堆序」听成「Postgres 做不了范围扫描」。能做，只是物理序不保证、靠预读与位图扫描补救；`CLUSTER` 能一次性重排但不会自动维持。差别在默认代价，不在能不能。</span>

## 小结

- InnoDB 聚簇主键存行；Postgres 堆无序，索引都回 TID。
- 回表次数与页分裂模式因此不同。
- undo/redo 与 doublewrite 下一课。
- 出处：Ramakrishnan and Gehrke；InnoDB/Postgres 存储文档。

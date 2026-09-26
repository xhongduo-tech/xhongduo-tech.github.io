---
title: 影子分页对照
date: 2026-09-08
section: cs
---

# 影子分页对照

<div class="epigraph">
<p>影子分页写新页并切换根：提交即新树可见，无需 redo 前滚。随机写与大目录更新是代价，现代 OLTP 主路径仍是 WAL 原地。</p>
<footer>—— 据 Lorie 影子分页；Gray and Reuter 对照；SQLite 回滚模式点名</footer>
</div>

[上一课](/cs/logging-granularity)选了记什么。本课对照**不原地更新**的恢复：影子分页（shadow paging）。主干 WAL 是主路径；进阶用对照理解为何 ARIES 赢在随机 I/O 与细粒度。组提交下一课把 WAL 刷盘摊掉。

## 问题

每页有当前与影子。事务写时拷页到新位置（COW），修改页表（或文件 inode 树）。提交：原子切换根指针（一次 I/O）。崩溃：根仍指旧树，新页丢弃。缺口是**页表本身很大**：更新一条索引可能级联大量目录页，写放大；难以 steal 单页而不暴露。并发：两事务 COW 同一页要合并。

SQLite 历史上 rollback journal / WAL 模式切换，是影子与日志在嵌入式上的产品对照。LMDB 一类 COW B+ 也是影子亲戚。

<span class="marginnote">数字实例：提交只需 fsync 一次根指针，看起来便宜；但一个 8 KB 页哪怕只改了 100 字节，也得整页另写 8 KB——写放大约 80 倍，B+ 树上层目录页还会级联 COW。这就是大型 OLTP 弃用它的第一原因。</span>

<span class="marginnote">Raymond Lorie 影子分页。Gray and Reuter 分析为何大型系统偏 WAL。本课不是教「用影子替换 InnoDB」。</span>

## 方法

读：跟随当前根。写：分配新页号，拷旧，改，更新父（可能再 COW）。提交：fsync 新页与新根。检查点：可丢弃不可达旧页当 GC——与 MVCC 版本回收神似，粒度为页。

与 WAL 混合：写前日志仍可有，影子管页原子。现代常用 WAL+原地+doublewrite，少用纯影子。

```mermaid
flowchart TD
  OLD["旧根树"] --> COW["写时拷页"]
  COW --> NEW["新根"]
  NEW --> COM["原子切换根"]
  CRASH["崩溃"] --> OLD
```

## 机制

优点：恢复快（几乎不用 redo 长扫描），实现直观。缺点：随机分配页破坏聚簇、碎片、并发粒度粗、目录更新重。LSM 不可变文件+元数据指针切换，是文件级影子，吸收了「切换根」思想而把随机写变成顺序文件。

ARIES 用顺序日志换随机数据页延迟写，与影子的随机新页对偶。

```mermaid
flowchart TD
  CR["崩溃后重启"] --> Q{"采用哪种机制?"}
  Q -->|影子分页| R["读根指针指向哪棵树"]
  R --> A["新树完整即已提交, 否则弃用"]
  A --> D["几乎无需 redo 扫描"]
  Q -->|WAL+原地| L["从检查点起前滚 redo"]
  L --> U["重放日志尾到一致点"]
```

<span class="marginnote">直觉类比：影子分页像编辑器的「另存为」——所有修改写进一份新副本，保存（提交）只是把快捷方式指过去；崩溃最多丢掉那份没被指到的新副本，原文件完好无损。所以它恢复极快，快在「根本不需要修复」。</span>

## 边界

本课不讲组提交。也不把 ZFS CoW 当数据库事务。RTO 下一课之后量化。

后课默认：大型并发库用 WAL+ARIES；影子/COW 出现在嵌入式或文件级元数据。组提交：多事务共享一次 fsync。

影子的原子单位是根切换；WAL 的原子单位是日志尾。

## 小结

- 影子 COW+切根，恢复简单，随机写与目录放大差。
- LSM/文件元数据吸收了切根思想。
- 组提交下一课：摊掉 WAL 的 fsync。
- 出处：Lorie；Gray and Reuter；SQLite/LMDB 对照。

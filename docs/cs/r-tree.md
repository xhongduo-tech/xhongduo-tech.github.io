---
title: R 树
date: 2026-09-08
section: cs
---

# R 树

<div class="epigraph">
<p>每个节点存若干子矩形的最小外包框；查询框与 MBR 不相交则整页剪掉，相交则下降——为磁盘页而不是为内存二分。</p>
<footer>—— 据 Guttman, R-Trees: A Dynamic Index Structure for Spatial Searching, SIGMOD 1984；Beckmann et al., R*-tree, SIGMOD 1990 整理</footer>
</div>

[上一课](/cs/kd-tree) 适合内存点集，节点太瘦，外存一次 I/O 只比较一个分裂值。[B+](/cs/bplus-split) 已说明页要高扇出。本课不轮换轴。缺口是 R 树：页内多条记录，各带最小外包矩形（MBR），插入选「扩大面积最小」的子树，溢出则分裂。

## 问题

空间对象（点、框、多边形外包）要按页索引。查询矩形 $Q$：根到叶，只进入 $MBR\cap Q\neq\emptyset$ 的孩子。MBR 可重叠，故可能多路下降，最坏仍扫很多页。缺口不是新的几何谓词，而是**把 B 树的页占用率思想搬到矩形键**，分裂启发式决定重叠多少。

<span class="marginnote">Guttman SIGMOD 1984。R* 调整插入与强制重插以减重叠。本课钉 MBR 与剪枝，不把 R* 全部规则抄成词条。</span>

## 方法

查找：与 $Q$ 相交则递归。插入：从根选子树（最小面积增量等），叶满则分裂成两组，使两组 MBR 面积和或重叠小，再向上插索引项。删除：欠载则合并或再分配，类似 B+。

```mermaid
flowchart TD
  Q["查询框 Q"] --> MBR["孩子 MBR"]
  MBR --> SKIP["不相交: 剪枝"]
  MBR --> DOWN["相交: 下降, 可多路"]
```

与 k-d：R 树允许 MBR 重叠、面向块；k-d 空间不相交、面向内存。与四叉树：规则网格分裂 vs 数据自适应矩形。

<span class="marginnote">初学者容易以为 R 树像 k-d 那样空间不重叠、一次下降即可。实际上相邻 MBR 常常同时接到查询框，必须多路下降——重叠越多，要访问的磁盘页越多，这正是分裂启发式要压的东西。</span>

## 机制

I/O 次数 ≈ 访问节点数。重叠大则剪枝弱，这是调分裂的原因。并发与 B 树锁耦合类似，本课串行。不要把 R 树写成神经网络空间；就是外存空间索引。

<span class="marginnote">MBR（最小外包矩形）翻译过来就是「能把这个对象整个装进去的最小长方形」。树里存的不是多边形本身，而是这个外壳；查询框连外壳都不碰，里面的形状一眼都不用看。</span>

```mermaid
flowchart TD
  NEW["新矩形到来"] --> ROOT["从根开始"]
  ROOT --> CHOICE{"哪个孩子 MBR 扩大面积最小?"}
  CHOICE --> DESCEND["选增量最小者下降"]
  DESCEND --> FULL{"叶页已满?"}
  FULL -->|"否"| PUT["直接放入叶"]
  FULL -->|"是"| SPLIT["分裂成两组: 使 MBR 面积和与重叠小"]
  SPLIT --> UP["向上插入索引项"]
```

<span class="marginnote">「扩面积最小」代入一个数：新点已落在孩子 A 的框内（增量 0），落在孩子 B 的框外（要把 B 扩 3 个单位面积），就选 A——让一次插入尽量不把树的长相带歪。</span>

## 边界

本课不写 Hilbert 打包的具体码（下一课空间填充曲线会给序），不写三维 GIS 全流程。点集规则划分可用四叉/八叉，不一定要 R。

后课默认：外存空间范围查询默认 R 树家族。规则空间二分用四叉树。

## 小结

- R 树：分页 MBR 树，查询靠框相交剪枝。
- 分裂启发式决定重叠；最坏可退化。
- 规则网格划分是四叉/八叉树。
- 出处：Guttman, *SIGMOD*, 1984；Beckmann et al., *SIGMOD*, 1990。

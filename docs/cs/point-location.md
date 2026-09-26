---
title: 点定位
date: 2026-09-08
section: cs
---

# 点定位

<div class="epigraph">
<p>平面剖分成面后，询问点落在哪一面：梯形图或 Kirkpatrick 层次，$O(n)$ 空间 $O(\log n)$ 查询。</p>
<footer>—— 据 Kirkpatrick, Optimal Search in Planar Subdivisions, 1983；Mulmuley、Seidel 随机增量梯形图整理</footer>
</div>

上一课[Voronoi 与 Delaunay](/cs/voronoi-delaunay)给出剖分。缺口是查询：点 $q$ 在哪个面/哪条 Delaunay 边的哪侧。朴素 $O(n)$ 走。本课点定位结构。不重写 Fortune。后课几何精度。

## 问题

梯形图：过每个顶点作竖线直到撞边，面变梯形。随机增量期望 $O(n)$ 大小、$O(\log n)$ 历史 DAG 查询。Kirkpatrick：独立集收缩层次，$\log$ 层。步行（沿 Delaunay 走）实践快但最坏差。

缺口是数据结构，不是再三角化。

### 不是最近邻本身

最近站点可用 Voronoi 定位，或 k-d 树。本课一般剖分。k-d 树是正交，点名。

<span class="marginnote">Kirkpatrick 1983。CGAL 实现随机梯形图。后课鲁棒性：定位依赖谓词符号。</span>

<span class="marginnote">直觉类比：梯形图像在剖分的每个路口都拉起两根垂直的「灯柱」，灯柱一碰到边就停。奇形怪状的面全被切成上下底水平的梯形，而「这个点在梯形里吗」只需要一两次坐标比较。</span>

## 方法

预处理剖分。询问沿 DAG 或层次下降。动态插入更重（随机增量在线）。

```mermaid
flowchart TD
  SUB["平面剖分"] --> TRAP["梯形图 / 层次"]
  TRAP --> Q["O(log n) 点定位"]
```

与扫描线：梯形图常随机增量不是从左扫一次完。

## 机制

每层面数减常因子，查询下降对数。历史 DAG 记录「被谁切开」。与二叉搜索树：几何键是区域包含。与半平面：对偶可把某些定位变成凸包切。

竖线切割到底把查询变成什么样的一路下降？

```mermaid
flowchart TD
  IN["任意平面剖分"] --> V["过每个顶点作竖直线"]
  V --> STOP["竖线上下延伸，撞到第一条边就停"]
  STOP --> TRAP["所有面被切成梯形"]
  TRAP --> DAG["每个梯形对应 DAG 一个节点"]
  DAG --> SEARCH["查询：比较左与右、上与下，逐层下降"]
  SEARCH --> LOG["比较次数与层数同阶，对数级"]
```

<span class="marginnote">数字实例：100 万个顶点的地图，朴素逐面检查要约一百万次判断；$O(\log n)$ 定位只需 20 次上下比较（$2^{20}\approx 10^6$）——从「翻遍全书」变成「对半猜页码」。</span>

<span class="marginnote">常见误区：以为点定位的难点是查询慢。真正的坑是浮点：竖线算不算撞上这条边、询问点恰好在边上，符号判断在舍入误差下会翻转，DAG 走错一层结果就错——这正是后课几何鲁棒性要接的盘。</span>

## 边界

本课不写三维点定位。动态删点不写。后课默认：静态剖分 $O(\log n)$ 定位。下一课几何精度与鲁棒性。

## 小结

- 剖分查询用梯形图或 Kirkpatrick。
- $O(n)$ 空间、$O(\log n)$ 查询。
- 步行实用、最坏弱。
- 出处：Kirkpatrick, 1983；随机增量梯形图。

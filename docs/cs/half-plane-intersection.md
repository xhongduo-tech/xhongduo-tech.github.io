---
title: 半平面交
date: 2026-09-08
section: cs
---

# 半平面交

<div class="epigraph">
<p>线性不等式的可行域是凸多边形（可无界）；增量或极角排序 + 双端队列 $O(n\log n)$ 求出交。</p>
<footer>—— 据 Preparata, Muller 与计算几何教材；LP 二维对照[单纯形](/cs/simplex) 整理</footer>
</div>

上一课[线段相交](/cs/segment-intersection)报交点。半平面 $ax+by\le c$ 的交是凸集。缺口是求交多边形：极角排序后 deque 增量，类似 Andrew 凸包对偶。不重写 BO。后课 Voronoi。二维 LP 也可随机增量线性期望。

## 问题

$n$ 个半平面，交可能空、点、多边形、无界。对偶：点凸包 $\leftrightarrow$ 半平面交。算法：排序方向，维护凸链，新半平面切掉尾部。$O(n\log n)$。

<span class="marginnote">直觉类比：每个半平面就是一刀——沿一条直线把纸切开、只留一侧。所有刀切完后纸上剩下的那块就是半平面交；直刀只能切出平底，所以结果必然是凸的（或空、或无限延伸）。</span>

<span class="marginnote">数字实例：$x\ge0$、$x\le4$、$y\ge0$、$y\le3$ 四刀交出矩形 $(0,0)$–$(4,3)$。再加一刀 $x+y\le5$，右上角被切掉，变成 $(0,0)$、$(4,0)$、$(4,1)$、$(2,3)$、$(0,3)$ 五个顶点的五边形。</span>

缺口是凸可行域边界，不是单纯形表（高维）。

### 无界要加哨兵

无穷射线用很大边界或方向无穷。空交要检测矛盾（平行反向）。不要假定一定有界。

<span class="marginnote">计算几何教材半平面交。Megiddo/Seidel 二维 LP 线性。后课 Voronoi 可看成半平面交的轨迹。</span>

## 方法

规范化（左侧为可行）。按法向极角排序。双端队列增量。输出顶点。

```mermaid
flowchart TD
  HP["半平面"] --> ANG["极角排序"]
  ANG --> DQ["deque 切尾"]
  DQ --> POLY["凸交"]
```

平行半平面先合并最紧。

## 机制

新约束只切凸多边形的连续一段，故从两端弹。与 Graham 对偶。与 LP：目标可当再一个扫描方向。高维半空间交是 $O(n^{\lfloor d/2\rfloor})$ 级，本课平面。

```mermaid
flowchart TD
  NEW["取出下一条半平面"] --> CHK{"当前交还剩点吗？"}
  CHK -- "空" --> EMPTY["提前判定空交"]
  CHK -- "非空" --> BACK{"队尾顶点在新线不可行侧？"}
  BACK -- "是" --> POPB["弹队尾，再查"]
  BACK -- "否" --> FRONT{"队头顶点也不可行？"}
  FRONT -- "是" --> POPF["弹队头，再查"]
  FRONT -- "否" --> PUSH["把新边接进 deque"]
```

<span class="marginnote">常见误区：不是每条约束最后都贡献一条边。某条刀如果完全落在别的刀切过的区域内（冗余约束），弹两端的过程会自动把它淘汰——这正是「排序 + deque」比逐对求交快的原因。</span>

## 边界

本课不写三维多面体。浮点切点收到鲁棒性课。后课默认：平面半平面交 $O(n\log n)$。下一课 Voronoi 与 Delaunay。

## 小结

- 半平面交 = 凸多边形（可空、可无界）。
- 极角 + deque $O(n\log n)$。
- 对偶于点凸包。
- 出处：计算几何标准算法；二维 LP 对照。

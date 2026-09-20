---
title: A* 与启发式搜索
date: 2026-09-08
section: cs
---

# A* 与启发式搜索

<div class="epigraph">
<p>$f(n)=g(n)+h(n)$：已付代价加到达目标的估计；一致启发下 A* 展开的结点不超过任何同等启发的最优搜索。</p>
<footer>—— 据 Hart, Nilsson and Raphael, A Formal Basis for the Heuristic Determination of Minimum Cost Paths, 1968；Dijkstra 1959 对照整理</footer>
</div>

上一课[图同构直觉](/cs/graph-isomorphism)收束连通与匹配单元。主干[Dijkstra](/cs/dijkstra)是 $h=0$ 的最短路。本课缺口是**启发式** $h$：低估到 $t$ 的距离时，优先队列按 $g+h$ 取出仍正确。不重证非负权取出即钉死的那套，只把它改成带势的比较。后课双向与 ALT 把 $h$ 造得更狠。

## 问题

状态图（网格、配置）上单源到 $t$。$g(n)$ 为从 $s$ 到 $n$ 的已知代价。$h(n)$ 估计 $n$ 到 $t$。可采纳（admissible）：$h(n)\le\delta(n,t)$。一致（consistent）：$h(u)\le w(u,v)+h(v)$。一致 $\Rightarrow$ 可采纳；此时 A* 与带势 Dijkstra 相同：$\hat w(u,v)=w(u,v)+h(v)-h(u)\ge 0$，按 $g+h$ 取出即钉死。

缺口是 $h$，不是新的堆。$h=0$ 即 Dijkstra。过高估计可能丢最优，变成贪心最佳优先搜索。

### 启发来自松弛问题

网格用曼哈顿；图上用欧氏（若边权 $\ge$ 直线）。可采纳常来自删除约束后的真距离。不要把机器学习的启发式当本课正确性条件——除非仍可采纳。

<span class="marginnote">Hart–Nilsson–Raphael 1968。Pearl 的启发式搜索书是后续。一致启发下不会 decrease-key 到已钉死结点（与 Dijkstra 同）。后课 ALT 用路标算 $h$。</span>

<span class="marginnote">直觉类比：把 $g(n)$ 想成已经烧掉的油钱，$h(n)$ 想成导航估计的剩余油钱。A* 每次都挑「已花 + 预计」总账最便宜的方向探路；只要导航从不虚报（可采纳），总账最便宜的那条路就真是全局最便宜。</span>

<span class="marginnote">数字实例：网格上曼哈顿距离就是一个现成的可采纳 $h$——若从当前格到目标横向差 3 格、纵向差 4 格，就估 $h=3+4=7$。每步代价至少 1，真实路程必不小于 7，所以这个估计永远只低不高，用它剪枝不会剪掉真正的最短路。</span>

## 方法

优先队列键 $f=g+h$。取出 $u$，若 $u=t$ 停（一致时）。松弛邻接，更新 $g$ 与父指针。启发式预计算或闭式。

```mermaid
flowchart TD
  S["s"] --> G["g 已付"]
  H["h 启发"] --> F["f=g+h 出队"]
  G --> F
  F --> T["到达 t"]
```

不一致但可采纳时需允许重开结点，最坏仍可能指数展开，本课默认一致。

## 机制

势函数把原权变成非负，Dijkstra 正确性平移。展开结点集：任何 $f(n)\lt \delta(s,t)$ 的 $n$ 都必须展开，故 $h$ 越大（仍可采纳）剪掉越多。与 DFS/BFS：无权且 $h=0$ 不是 A* 的主场景；网格上 A* 典型。

不要在负权图上直接 A*：先 Bellman–Ford 或禁负。

第一张图画的是 $f=g+h$ 怎么算、怎么出队；这张图回答第二个问题：同一张图上，$h$ 取什么值直接决定要展开多少结点、以及会不会丢最优解。

```mermaid
flowchart TD
  G["同一张图"] --> H0["h = 0"]
  H0 --> EXPALL["向四面八方均匀展开，即 Dijkstra"]
  G --> HADM["可采纳 h，指向目标"]
  HADM --> FEW["只展开朝目标方向的结点"]
  G --> HBIG["h 高估真实距离"]
  HBIG --> SKEW["被吸向看似更近的路"]
  SKEW --> MISS["提前停在次优路径上"]
```

<span class="marginnote">常见误区：初学者容易把 A* 当成「另一种更快的 Dijkstra」。实际上 $h=0$ 时它就退化成 Dijkstra；加速完全来自 $h$ 剪掉的那些「怎么走都不会更优」的方向。$h$ 越准（但不高估）越快，一旦高估就会理直气壮地给出一条错误的「最短路」。</span>

## 边界

本课不写 IDA*、不写 SMA*。不把博弈树的 $\alpha\beta$ 当同一算法。后课默认：一致 $h$ 下 A* 最优；实现是带势 Dijkstra。下一课双向搜索与 ALT 路标。

## 小结

- 可采纳 $h$ 保证最优；一致则取出即钉死。
- $h=0$ 退回 Dijkstra；过大 $h$ 丢最优。
- 启发来自松弛真距离。
- 出处：Hart, Nilsson and Raphael, 1968。

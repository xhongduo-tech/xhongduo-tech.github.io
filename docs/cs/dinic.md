---
title: Edmonds–Karp 与 Dinic
date: 2026-09-08
section: cs
---

# Edmonds–Karp 与 Dinic

<div class="epigraph">
<p>最短增广把 Ford–Fulkerson 变成 $O(VE^2)$；分层网络上的阻塞流把迭代降到 $O(V^2 E)$。</p>
<footer>—— 据 Edmonds and Karp, Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems, 1972；Dinic, Algorithm for Solution of a Problem of Maximum Flow, 1970；CLRS 第 26 章整理</footer>
</div>

上一课[最大流 Ford–Fulkerson](/cs/max-flow-ff)给出残量增广与最大流最小割，并警告：任意增广路可使次数随容量指数。本课不重证割。缺口是**选哪条路**：Edmonds–Karp 每次 BFS 最短（边数最少）增广；Dinic 一次建分层，再推阻塞流。整数容量下二者都多项式。后课二分图匹配把单位容量网络当特例。

## 问题

Ford–Fulkerson 是框架。容量很大且每次 $\Delta=1$ 时，增广次数可达 $\Theta(|f^*|)$。Edmonds–Karp：残量上 BFS 找 $s$–$t$ 最短路再增。关键事实：最短路长度不减，每条边作为临界边的次数 $O(V)$，故增广 $O(VE)$ 次，每次 $O(E)$，总 $O(VE^2)$。缺口是这份选路规则，不是新的流定义。

Dinic：按残量 BFS 分层，$d(v)$ 为边数距离。只沿层间边 $d(w)=d(v)+1$ 推流，直到 $s$ 到 $t$ 不再连通（阻塞流）。然后重建分层。层数增加，至多 $V-1$ 相。单位容量或简单图上实现可到 $O(V^2 E)$ 量级（经典界）；更紧的实现不在本课展开。

<span class="marginnote">术语翻译：「增广路」就是一条从 $s$ 到 $t$、每段都还有剩余容量的路径；一次能塞的流量等于路上最窄那段的剩余量（瓶颈）。数字实例：路径 $s\to a\to t$ 上 $s\to a$ 剩 3、$a\to t$ 剩 5，则本次只能增广 $\min(3,5)=3$，且 $a\to t$ 还剩 2 给下一条路用。</span>

### 阻塞流不是一次增广

一次阻塞流可以同时填满分层里的许多条路，相当于一批最短增广。Dinic 1970 的「功率」估计即分层推进。不要把 Dinic 写成「再跑一遍 Edmonds–Karp」。

<span class="marginnote">Edmonds–Karp 1972 证明最短增广多项式。Dinic（Dinitz）1970 更早给出分层。CLRS 以 Edmonds–Karp 为定理级实现，Dinic 作进阶。本课两条都要：一条证明墙是多项式，一条给更常用的推进。</span>

## 方法

EK：循环 BFS，沿树增广，更新残量，直至 $t$ 不可达。Dinic：循环 { BFS 分层；若 $t$ 不在分层则停；DFS/指针沿层推阻塞流 }。

```mermaid
flowchart TD
  FF["残量可增广"] --> EK["BFS 最短增广 O(VE^2)"]
  FF --> DIN["分层 + 阻塞流"]
  DIN --> POLY["多项式相数"]
```

容量有理先化为整数。无理容量仍可能病态，主干用整数。

## 机制

最短增广使「距离标号」单调，分析才能数临界边。Dinic 的当前弧优化避免反复扫邻接表，是实现零件，不改变分层正确性。单位容量网络（匹配）上界更好，下一课用，本课不把匹配写完。

```mermaid
flowchart LR
  S["s：第 0 层"] --> A["a：第 1 层"]
  S --> B["b：第 1 层"]
  A --> C["c：第 2 层"]
  B --> C
  A --> T["t：第 2 层"]
  C --> T
  A -.->|"同层边：禁走"| B
```

这张图回答「分层网络长什么样」：BFS 给每个点标上到 $s$ 的边数距离，阻塞流只允许沿「第 $i$ 层 $\to$ 第 $i+1$ 层」的边推；同层边和往回走的边一律不碰。当 $t$ 被推到与 $s$ 断开（图中 $c\to t$、$a\to t$ 全饱和），这一相结束，残量重建分层后层数至少加一。

<span class="marginnote">直觉类比：分层像消防演习里按离出口的步数排队，水流只许从第 $i$ 排传给第 $i+1$ 排，绝不许在同学之间横着传、也不许往后传。这样每条被推的路都恰好是最短路长度，$t$ 的层数只增不减，所以至多 $V-1$ 相就穷尽。</span>

不要在有负容量的「流」上套本课；模型仍是上一课的容量约束与守恒。

## 边界

本课不写推送重标、不写费用流。最小割在增广结束时仍是 $s$ 侧可达集。后课默认：要多项式最坏，最短增广或 Dinic；二分图匹配化成单位容量最大流。

<span class="marginnote">常见误区：初学者容易以为最小割要在最大流之外「再单独算一次」。实际上最大流一结束，在残量网络上从 $s$ 还能走到的那个点集，把它射向外的所有边割断，恰好就是一个最小割——最大流最小割定理保证两边数值相等，不用再找。</span>

## 小结

- Edmonds–Karp：最短增广，$O(VE^2)$。
- Dinic：分层阻塞流，相数 $O(V)$。
- 仍是残量增广；选路决定次数。
- 出处：Dinic, 1970；Edmonds and Karp, 1972；CLRS 第 26 章。

---
title: 最近公共祖先
date: 2026-09-08
section: cs
---

# 最近公共祖先

<div class="epigraph">
<p>树上两点的最近公共祖先是同时盖住二者的最深顶点；离线并查集或在线倍增、欧拉序加 RMQ，都能答。</p>
<footer>—— 据 Tarjan, Applications of Path Compression on Balanced Trees, 1979；Harel and Tarjan, Fast Algorithms for Finding Nearest Common Ancestors, 1984；Bender and Farach-Colton, The LCA Problem Revisited, 2000 整理</footer>
</div>

上一课[哈密顿与 TSP 精确解](/cs/hamiltonian-tsp-exact)在一般图上指数。本课把图收成**树**（或有根树）：两点之间唯一路径，询问最近公共祖先（LCA）。不重做子集 DP。缺口是查询结构——离线一次扫完，或预处理后在线。后课倍增把同一技巧用到路径聚合与树上差分。

## 问题

有根树，$lca(u,v)$ 为 $u$、$v$ 的公共祖先中深度最大者。路径 $u$–$v$ 就是 $u$ 到 $lca$ 再下到 $v$。朴素双指针上跳 $O(n)$ 每次。缺口是更快：离线 $m$ 次询问，或预处理后 $O(1)$/$O(\log n)$。

Tarjan 离线：DFS 时用并查集维护「已处理子树的代表」；回溯时把询问另一端已访的点与当前点求 LCA。一次 DFS + 几乎 $O(n+m)$。在线：预处理深度与父，或欧拉序（进出把树摊成序列，LCA 变成序列上的 RMQ）。

<span class="marginnote">术语翻译：离线/在线说的是询问何时可知——离线 = 所有询问提前拿全，一次 DFS 走完顺便把每条都答掉（Tarjan 并查集）；在线 = 询问随时插进来，必须先花预处理时间备好一张随时可查的表（倍增、RMQ）。竞赛读入方式常决定选哪族算法。</span>

### RMQ 不是另一道题

欧拉序上相邻深度差 $1$，$\pm 1$ RMQ 可 $O(n)$ 预处理、$O(1)$ 查询（Bender–Farach-Colton）。一般 RMQ 稀疏表 $O(n\log n)$–$O(1)$。本课认「LCA $\leftrightarrow$ RMQ」，不把稀疏表写成另一课程。

<span class="marginnote">Tarjan 1979 离线并查集。Harel–Tarjan 1984 讨论在线线性预处理。Bender–Farach-Colton 2000 把 $\pm 1$ RMQ 写清楚。后课树上倍增是同一父指针的对数张表。</span>

## 方法

小 $n$ 可倍增：`fa[k][u]` 为 $u$ 的 $2^k$ 祖先，预处理 $O(n\log n)$，查询先对齐深度再一起跳，$O(\log n)$。离线用并查集。需要常数查询再用欧拉序 RMQ。

```mermaid
flowchart TD
  T["有根树"] --> OFF["离线：DFS + 并查集"]
  T --> BIN["在线：倍增 fa[k][u]"]
  T --> RMQ["欧拉序 + RMQ"]
  OFF --> LCA["lca(u,v)"]
  BIN --> LCA
  RMQ --> LCA
```

森林：先找根；不同树无 LCA，或虚根。

## 机制

括号序/欧拉序：第一次访问时刻之间的点，深度最小者即 LCA（更精确：欧拉序区间内深度最小的顶点）。倍增：深度差的二进制分解是在链上跳，与[二进制快速幂](/cs/fast-exponentiation)同一「倍增」念头，后课才把矩阵幂展开；本课只跳父指针。

<span class="marginnote">数字实例：两结点深度差为 13 = 8+4+1，即按 $2^3, 2^2, 2^0$ 三步跳平；$n=10^5$ 的树倍增表只需 $\lceil \log_2 n \rceil = 17$ 层，任何一次查询至多跳 $2\times 17$ 次——这就是「预处理 $O(n\log n)$、查询 $O(\log n)$」里那些对数的真身。</span>

<span class="marginnote">直觉类比：欧拉序像把树「压扁」成一份巡逻足迹——DFS 每次经过一个点就记一笔。两个点的 LCA，就是足迹里这两点之间海拔（深度）最低的那个点：想上树顶必须先经过它。LCA 于是变成「区间最小值」，RMQ 的全部机器都能搬过来用。</span>

```mermaid
flowchart TD
  Q["询问 lca(u, v)"] --> C1{"u 与 v 深度相同?"}
  C1 -->|"否"| UP["较深者按深度差的二进制逐位上跳"]
  UP --> C2{"跳平后 u == v?"}
  C2 -->|"是"| A1["u 本身就是 LCA"]
  C2 -->|"否"| C3["两指针从大 k 到小同步上跳"]
  C3 --> A2["首次父指针相同的前一步: 即 LCA"]
```

这张图回答的问题：一次倍增查询内部到底怎么跳。两步走——先把深者按深度差的二进制对齐到同层，若还没相遇，就让两者同速按 $2^k$ 往上试跳；每层只决定「跳或不跳」一位，所以查询是纯对数次查表。

不要在有环图上定义 LCA：最近公共祖先依赖树（或 DAG 上要另定义，本课不写）。

## 边界

本课不写重链剖分（下一课的下一课）、不写点分治。动态加点、换根在线是另一模型。后课默认：树上两点路径经过 LCA；倍增表是标准预处理。下一课把倍增接到路径和与树上差分。

## 小结

- LCA 是两点路径的最高点；树上路径唯一。
- 离线并查集；在线倍增或欧拉序 RMQ。
- 一般图的「祖先」没有这一定义。
- 出处：Tarjan, 1979；Harel and Tarjan, 1984；Bender and Farach-Colton, 2000。

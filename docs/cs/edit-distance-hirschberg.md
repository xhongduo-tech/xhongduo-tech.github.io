---
title: 编辑距离与 Hirschberg
date: 2026-09-08
section: cs
---

# 编辑距离与 Hirschberg

<div class="epigraph">
<p>Levenshtein 距离是插、删、改的格图最短路；$O(nm)$ 时间，Hirschberg 用分治把空间降到线性并还原路径。</p>
<footer>—— 据 Levenshtein, 1966；Hirschberg, A Linear Space Algorithm for Computing Maximal Common Subsequences, 1975；CLRS 第 15 章整理</footer>
</div>

上一课[LIS 与 LCS](/cs/lis-lcs)给了 LCS 表。编辑距离：对角匹配代价 0，水平/垂直插删 1，对角替换 1（或另设）。缺口是同一格图的最短路，以及 Hirschberg：两边 DP 在中线相遇，递归还原，空间 $O(n+m)$。不重写 LCS 转移形状。后课序列比对与位并行。

## 问题

$dp[i][j]=\min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+[a_i\neq b_j])$。LCS 长度 $k$ 时，若只插删，距离 $n+m-2k$。只要距离：滚动两行 $O(\min(n,m))$ 空间。要操作序列：朴素 $O(nm)$ 空间存前驱。Hirschberg：对中行分别正反 DP，找分割点使两侧距离和最小，递归。

<span class="marginnote">数字实例：比对两条 10000 字符的序列，全表约 10000 × 10000 = 1 亿格，每格 4 字节就是约 400 MB；只留几行的滚动写法只要几十 KB。空间差四个数量级，时间都停在同一个 $O(nm)$ 量级。</span>

缺口是空间，不是新的编辑操作。

<span class="marginnote">直觉类比：只要距离不要路径时，滚动数组像结账只保留「上一行小票」——算第 $i$ 行只用到第 $i-1$ 行，更早的行算完即丢。空间因此从 $O(nm)$ 降到两行；代价是丢掉了「这一步怎么走来的」的记录，想要路径就得另想办法。</span>

### 不是近似匹配通配符课

通配、仿射缺口罚分（Gotoh）后课点名。本课单位代价 Levenshtein。不要把 KMP 当编辑距离。

<span class="marginnote">Levenshtein 1966。Hirschberg 1975 原为 LCS，编辑距离同构。后课 Myers 位并行、$O(nd)$ 差分。</span>

## 方法

先写标准 DP。只要长度用滚动。要路径用 Hirschberg 或存前驱。对角线优先可提前停（阈值）。

```mermaid
flowchart TD
  GR["n×m 格图"] --> DP["插删改最短路"]
  DP --> HB["中线分治还原"]
```

边界：空串对长度为对方长度。

## 机制

格图 DAG，边权 0/1，DP 即 DAG 最短路。Hirschberg：最优路径必过中线某点，该点最小化左半+右半。与分治优化不同：这里分治的是路径空间不是决策单调。

<span class="marginnote">常见误区：把线性空间当成「免费优化」。Hirschberg 是用约两倍的时间常数（正反各 DP 一遍）换空间的；反过来，只要距离不要路径时滚动数组就够，不必动用分治。先问「要不要操作序列」，再选方案。</span>

不存整张表，Hirschberg 怎么把路径找回来？

```mermaid
flowchart TD
  MID["在中线处把格图切开"] --> F["上半：正向 DP 一行"]
  MID --> B["下半：反向 DP 一行"]
  F --> SUM["逐列相加两侧值"]
  B --> SUM
  SUM --> CUT{"取总和最小的列作为分割点"}
  CUT --> P1["上半递归求解"]
  CUT --> P2["下半递归求解"]
  P1 --> PATH["拼出完整编辑路径"]
  P2 --> PATH
```


## 边界

本课不写 DNA 仿射缺口全文。不写块编辑。后课默认：编辑距离 $O(nm)$；线性空间还原用 Hirschberg。下一课比对与位并行。

## 小结

- 编辑距离 = 格图最短路 $O(nm)$。
- Hirschberg 线性空间还原路径。
- 与 LCS 同一张格，代价不同。
- 出处：Levenshtein, 1966；Hirschberg, 1975。

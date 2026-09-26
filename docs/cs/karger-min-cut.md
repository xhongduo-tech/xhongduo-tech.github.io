---
title: Karger 最小割
date: 2026-09-08
section: cs
---

# Karger 最小割

<div class="epigraph">
<p>随机收缩边直到两点，得到的割以 $\Omega(1/n^2)$ 概率为全局最小；重复 $O(n^2\log n)$ 次高概率成功。</p>
<footer>—— 据 Karger, Global Min-cuts in RNC, 1993；Karger and Stein, 1996；CLRS 第 23 章对照整理</footer>
</div>

上一课[几何精度](/cs/geometric-robustness)收束几何。主干[最大流最小割](/cs/max-flow-ff)是 $s$–$t$ 割。全局最小割：无指定 $s,t$，无向边连通性。缺口是 Karger 随机收缩。不重写 Dinic。后课 Las Vegas / Monte Carlo 分类。

## 问题

无向（多重）图。收缩边：两端合并，环丢掉。直到剩 2 顶点，其间重边为割。最小割边数 $c$，成功概率 $\ge 1/\binom{n}{2}$ 量级：每次收缩不碰最小割边的条件概率。Karger–Stein：递归到 $n/\sqrt 2$ 再分支，时间更好。

<span class="marginnote">直觉类比：收缩边像把两座城市合并成「一个都市场」——桥还在，只是变成了市内桥。缩到只剩两个都市圈时，圈与圈之间剩下的几座桥就是候选割；全程没拆掉跨圈桥，这局就赢了。</span>

缺口是随机收缩，不是 $n$ 次最大流（也可 $O(n)$ 次流，更重）。

### 不是 $s$–$t$ 最小割

全局割可小于某对 $s$–$t$。指定 $s,t$ 用流。本课全局。

<span class="marginnote">Karger 1993。Karger–Stein 1996。后课把成功概率收进 Monte Carlo 定义。</span>

## 方法

实现用并查集 + 随机边（按重边权或邻接表）。重复独立试验取最小。Karger–Stein 递归写清分支。

```mermaid
flowchart TD
  G["多重图"] --> CTR["随机收缩"]
  CTR --> CUT["两点间重边"]
  CUT --> REP["重复取最小"]
```

加权边：按权随机，或先离散。

## 机制

最小割边少，随机边更常落在块内。分析用 $\prod (1-c/(m_i))$。与最大流：确定多项式，常数大。Karger 简单、随机。与 Borůvka：都收缩，一个为 MST 最轻、一个为均匀随机。

<span class="marginnote">数字实例：$n=100$ 时单次成功率约 $1/\binom{100}{2} \approx 0.02\%$，看似没救；但独立跑 2 万次，全败概率约 $(1-1/4950)^{20000} \approx e^{-4} \approx 1.8\%$，跑 5 万次降到约 $0.004\%$——重复独立试验是 Karger 唯一的放大器。</span>

```mermaid
flowchart TD
  S["剩 k 个顶点时随机收缩一条边"] --> Q["这条边有多危险？"]
  Q --> B["最小割共 c 条边，图里至少还有 k·c/2 条边"]
  B --> A["抽中最小割边的概率 ≤ 2/k"]
  A --> OK["不碰最小割的概率 ≥ 1 − 2/k"]
  OK --> CH["从 k=n 一路乘到 k=3"]
  CH --> E["单次存活率 ≥ 2/(n(n−1))，即 1/C(n,2)"]
```

## 边界

本课不写近线性确定全局最小割的最新结果全文。有向全局割不同。后课默认：全局最小割可用随机收缩。下一课 Las Vegas 与 Monte Carlo。

<span class="marginnote">常见误区：以为「随机算法」就不可信。Karger 是 Monte Carlo 型——输出偶有错，但重复能把错误概率压到任意小；而且拿到一个割后，检查它是否连通只需一次遍历，答案的好坏可以廉价验证。</span>

## 小结

- 随机收缩以多项式分之一得到最小割。
- 重复放大成功概率。
- 全局，不是 $s$–$t$。
- 出处：Karger, 1993；Karger and Stein, 1996。

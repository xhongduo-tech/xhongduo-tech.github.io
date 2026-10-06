---
title: Borůvka 与次小生成树
date: 2026-09-08
section: cs
---

# Borůvka 与次小生成树

<div class="epigraph">
<p>每个连通块同时抓住出块最轻边，一轮收缩；次小生成树则在 MST 上换一条边：树上路径最大边与非树边的最小非负替换。</p>
<footer>—— 据 Borůvka, 1926；Tarjan, Sensitivity Analysis of Minimum Spanning Trees, 1982；CLRS 第 23 章整理</footer>
</div>

上一课[Kruskal 重构树](/cs/kruskal-reconstruction)把 MST 加边史做成查询树。主干 Kruskal/Prim 已给一棵 MST。缺口有两块：Borůvka 的**并行长树**（也是切分定理），以及**严格次小生成树**——边集不同、权和次小。不重写切分定理。后课离开生成树，进入费用流。

## 问题

Borůvka：初始每个点一块。同步：每块选一条跨块最轻边（平权要定规则以免环）。加入这些边，收缩。至多 $O(\log n)$ 轮，每轮 $O(E)$，总 $O(E\log V)$。适合并行；也是 Prim/Kruskal 之外第三种经典 MST。

次小：在 MST $T$ 上，对每条非树边 $(u,v)$，替换 $T$ 上 $u$–$v$ 路径的最大边，得到另一棵生成树。所有这种替换里权和最小者即次小——边集与 $T$ 不同，权和允许仍等于 $w(T)$；若要权和严格大于 $w(T)$，平权时才要改删路径的次大边。实现：树上倍增维护路径 $\max$，扫非树边 $O(E\log V)$。

缺口是「换一条边」，不是再求 MST。

### 次小不是次小边

次小生成树与「第二小的那条边」无关。可能换掉的是内部一条大边。也不要 Borůvka 每块选次轻当次小树——那一般错。

<span class="marginnote">Borůvka 1926（Otakar Borůvka）早于 Kruskal/Prim。次小生成树的替换分析见 Tarjan 等关于 MST 敏感度的工作。CLRS 23 点名 Borůvka。后课最小费用流换容量与费用，不再是无向生成树。</span>

<span class="marginnote">直觉类比：Borůvka 像**村庄并网**——每个村各自拉一条通往邻村的 cheapest 电线，一轮下来村庄数至少减半；重复几轮，全国电网成型。各「村」互不商量，所以天然适合并行。</span>

## 方法

MST：Borůvka 或沿用 Kruskal。次小：先 MST；预处理树路径 $\max$；枚举非树边算 $\Delta=w(e)-\maxPath$；取最小非负 $\Delta$ 加上 $w(T)$。$\Delta=0$ 表示有另一棵同权 MST。

```mermaid
flowchart TD
  B["Borůvka 同步加最轻出边"] --> MST["最小生成树 T"]
  MST --> REP["非树边替换路径 max"]
  REP --> SEC["次小生成树"]
```

不连通则生成森林，次小说同一连通块内。

非树边 $(u,v)$ 进场时，树上 $u$–$v$ 路径该删谁？

```mermaid
flowchart LR
  ADD["加非树边 e：成环"] --> FIND["找环上 u–v 路径的最大边 m"]
  FIND --> CMP{"比较 w(e) 与 w(m)"}
  CMP -->|"更小"| SW["删 m：权变小 → 更优树"]
  CMP -->|"相等"| EQ["同权另一棵 MST"]
  CMP -->|"更大"| BAD["删 m 反而更贵：放弃"]
```

<span class="marginnote">数字实例：MST 上 $u$–$v$ 路径边权为 $\{3,8,5\}$，路径最大边 $=8$。某条权为 $6$ 的非树边换掉它，新树权 $=w(T)+6-8=w(T)-2$，更小——但 MST 已最小，说明此情形不可能；合法替换的最小正 $\Delta$ 才给出次小树。</span>

## 机制

Borůvka 每轮块数至少减半（每块至少合一次，若图连通），故对数轮。安全边仍来自切分。替换定理：任一与 $T$ 差一条边的生成树都是「加非树边、删圈上另一边」；权和变小当且仅当删的比加的重。故最优次小在「删路径最大边」里产生。

与重构树：LCA 给出路径 $\max$，正是本课替换要用的量。

## 边界

本课不写 $k$ 小生成树枚举全套、不写有向 Chu–Liu。动态 MST 不写。后课默认：MST 三种经典算法；次小 = 树上路径 max 替换。下一课最小费用最大流，边有容量与费用。

<span class="marginnote">常见误区：把「次小生成树」理解成「用第二便宜的边拼树」，或者 Borůvka 每块顺手存次轻边当次小树——都不对。次小是**整树权**排名第二，可能动的是树内部某条大边，且必须先有完整 MST 再做替换。</span>

## 小结

- Borůvka：块同步抓最轻出边，对数轮。
- 次小生成树：MST 上非树边替换路径最大边。
- 切分定理共用；与瓶颈重构树接口是路径 $\max$。
- 出处：Borůvka, 1926；MST 敏感度见 Tarjan, 1982；CLRS 第 23 章。

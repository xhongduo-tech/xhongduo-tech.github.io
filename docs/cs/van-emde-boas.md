---
title: van Emde Boas
date: 2026-09-08
section: cs
---

# van Emde Boas

<div class="epigraph">
<p>把 $[0,U)$ 对半切成 $\sqrt{U}$ 个簇，再递归；前驱后继跟着高位走簇、低位走内部，时间 $O(\log\log U)$。</p>
<footer>—— 据 van Emde Boas, Preserving Order in a Forest, FOCS 1975 / Math. Systems Theory 1977；Cormen, Leiserson, Rivest and Stein 第 20 章整理</footer>
</div>

[上一课](/cs/double-ended-pq) 与比较堆在任意有序域上是 $\Theta(\log n)$。键若是机器字宇宙 $[0,U)$，$n$ 可以远小于 $U$，比较模型不是唯一选择。[Trie](/cs/trie) 按比特下降是 $O(w)$。本课不维护堆序。缺口是 van Emde Boas 树：递归平方根切分，`insert`/`delete`/`successor` $O(\log\log U)$。

## 问题

需要有序整数集：成员、前驱、后继。平衡 BST 不知 $U$。vEB：结构含 `min`、`max`、一个 `summary` vEB（哪些簇非空）、以及 $\sqrt{U}$ 个簇各一棵子 vEB。高位 $\lfloor x/\sqrt{U}\rfloor$ 选簇，低位 $x\bmod\sqrt{U}$ 在簇内。后继：簇内有则递归；否则 summary 上找下一个非空簇再取其 min。缺口是**宇宙大小进递归深度** $\log\log U$，不是 $n$。

<span class="marginnote">空结构只存 min/max 的原型可降空间。完整数组簇是 $\Theta(U)$ 空间，须用哈希簇才接近 $O(n)$，那是 y-fast 的动机之一。</span>

## 方法

插入：空则只写 min；否则维护 min 不变式（更小的键顶替 min，旧 min 插进簇），再更新 summary。删除对称，注意 min 无簇副本的设计变体。递归底：$U=2$ 存两比特。

```mermaid
flowchart TD
  X["键 x"] --> HI["簇号 = 高位"]
  X --> LO["簇内 = 低位"]
  SUM["summary: 哪些簇非空"] --> NXT["跨簇后继"]
```

与位图：位图后继是扫字，$\Theta(U/w)$ 最坏一段；vEB 保证 $\log\log U$。$U=2^{32}$ 时 $\log\log U=5$，常数与实现相关。

## 机制

时间递推 $T(U)=T(\sqrt{U})+O(1)$ 得 $O(\log\log U)$。空间朴素 $\Theta(U)$。不要在 $U$ 超大时无哈希地分配簇数组。比较模型下信息下界仍 $\log n$；vEB 用了键是整数这一限制。

一次 `successor(x)` 的完整行走路线——先看本簇右侧，不行才跳到 summary：

```mermaid
flowchart TD
  S["successor(x)"] --> SPLIT["高位 h 选簇, 低位 l"]
  SPLIT --> C1{"l \lt 本簇 max ?"}
  C1 -->|是| IN["簇内递归求 l 的后继<br/>拼回 h·√U + 答案"]
  C1 -->|否| C2{"x \lt 全局 max ?"}
  C2 -->|否| NONE["无后继, 返回空"]
  C2 -->|是| SUM["summary 上求 h 的后继 h'"]
  SUM --> MIN["第 h' 簇非空, 取其 min"]
  MIN --> ANS["答案 = h'·√U + min"]
```

<span class="marginnote">数字实例：取 $U=2^{32}$（32 位整数宇宙），$\log\log U=\log 32=5$——无论集合里存 $10$ 个还是 $10^6$ 个键，一次后继最多下探约 5 层；同规模的平衡 BST 要走 $\log_2 10^6\approx 20$ 层。这是用「键必须是小整数」换来的量级差。</span>

<span class="marginnote">直觉类比：把宇宙想成一栋 $\sqrt U$ 层、每层 $\sqrt U$ 个柜子的楼。`summary` 是大厅的楼层指示牌，只记「哪层还有东西」。找后继时先翻自己这层（簇内递归），翻不到就抬头看指示牌跳到下一个有货的层，直接取那层最靠前的柜子（簇的 min）——不用一层层扫楼。</span>

<span class="marginnote">常见误区：按定义直接实现，空间是 $\Theta(U)$——$U=2^{32}$ 时哪怕只存一个键，簇数组也按 40 多亿个槽铺开。工程做法是用哈希表只存非空簇，空间落到接近 $O(n)$；这正是后课 y-fast trie 改造 vEB 的动机。</span>

## 边界

本课不把 x-fast trie 的全部分层位图写完——下一课 y-fast 用 x-fast 加平衡树块把空间收到 $O(n)$。也不把融合树的 $O(\log n/\log\log n)$ 写进来。

后课默认：宇宙 $U$ 已知时后继可以 $\log\log U$。线性空间用 y-fast trie。

## 小结

- vEB：平方根分簇，前驱后继 $O(\log\log U)$。
- 朴素空间 $\Theta(U)$；要 $O(n)$ 空间看 y-fast。
- 出处：van Emde Boas, 1975/1977；Cormen et al. 第 20 章。

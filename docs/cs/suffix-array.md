---
title: 后缀数组
date: 2026-09-08
section: cs
---

# 后缀数组

<div class="epigraph">
<p>把文本每一个后缀的起点按字典序排好，模式查找就变成在这张起点表上二分。</p>
<footer>—— 据 Manber and Myers, Suffix Arrays: A New Method for On-Line String Searches, SIAM J. Comput. 1993；Kärkkäinen and Sanders, Simple Linear Work Suffix Array Construction, ICALP 2003 整理</footer>
</div>

[上一课](/cs/space-filling-curve) 把网格线性化。字符串要的是「所有后缀谁更小」。[KMP](/cs/kmp) 只预处理模式；文本多次查询或子串统计需要文本侧索引。[Trie](/cs/trie) 可挂全部后缀但空间差。本课不写 Hilbert。缺口是后缀数组 $SA[i]$ = 第 $i$ 小后缀的起点。

## 问题

文本 $T[0..n)$（常加哨兵）。$n$ 个后缀，比较两后缀最坏 $\Theta(n)$，朴素排序 $\Theta(n^2\log n)$。后缀数组存排列，空间 $\Theta(n)$ 整数，远小于后缀树节点。查找 $P$：在 $SA$ 上二分，每次用 $T[SA[\cdot]..]$ 与 $P$ 比较，时间 $O(|P|\log n)$。缺口是**用排列代替显式树**，以及后课 LCP 把比较加速。

<span class="marginnote">初学者容易以为 SA 存的是后缀字符串本身——$n$ 个后缀合计 $n(n+1)/2$ 个字符，早就爆了。实际只存 $n$ 个起点下标，$\Theta(n)$ 个整数；字符串本体始终只有原文一份，要读时拿下标回去取。</span>

<span class="marginnote">Manber–Myers 倍增 $O(n\log n)$。DC3/skew（Kärkkäinen–Sanders）线性时间。本课钉语义与二分查找；构造选倍增即可讲清。</span>

## 方法

倍增：先按首字符秩，再按 $(rank[i],rank[i+2^k])$ 当对排序，k 增加。名次相同则后缀相等前缀更长。实现用整数排序。不要把后缀树 Ukkonen 提前当必须。

<span class="marginnote">术语翻译：哨兵就是在文本末尾补一个比所有字符都小的终结符（如 `$`）。它让长短后缀的比较总能分出胜负——短后缀先「用完」字符时，哨兵替它认输，不必为「一个前缀是另一个的前缀」写特判。</span>

```mermaid
flowchart TD
  T["文本 T"] --> SUF["n 个后缀"]
  SUF --> SA["SA: 字典序下的起点"]
  P["模式 P"] --> BIN["在 SA 上二分"]
```

与哈希：期望匹配可以 Karp–Rabin，后课再收；后缀数组确定性、支持所有子串的序结构。

## 机制

$SA$ 上相邻后缀可能公共前缀很长，朴素二分反复重比——这正是下一课 LCP 与 Kasai 的缺口。逆数组 $ISA[SA[i]]=i$ 把位置映回秩，构造与 LCP 都要用。

倍增构造为什么 $\log n$ 轮就能收敛，看每一轮各做了什么。

```mermaid
flowchart TD
  R0["k=0：按单个字符排名"] --> D["每轮用二元组排序"]
  D --> PAIR["(rank[i], rank[i+2^(k-1)])"]
  PAIR --> NR["排序得新一轮 rank"]
  NR --> Q{"2^k 覆盖全串了吗"}
  Q -- "否" --> D
  Q -- "是" --> OUT["rank 即 SA，共 log n 轮"]
```

<span class="marginnote">数字实例：$n=8$ 的文本只需 $k=0,1,2$ 共 3 轮——第 1 轮比较前 2 个字符、第 2 轮前 4 个、第 3 轮前 8 个。每轮排序 $O(n\log n)$，总复杂度 $O(n\log^2 n)$，换基数排序可到 $O(n\log n)$。</span>

本课不把 BWT 全文写完，只承认 BWT 是 $SA$ 的邻近字符排列。

## 边界

本课不要求实现 DC3。动态插入字符使 $SA$ 全失效，属动态后缀结构，主干不提前。LCP 数组是下一课必做预处理。

后课默认：静态文本子串查找可用 $SA$ 二分。相邻后缀的公共前缀用 LCP。

## 小结

- 后缀数组：后缀的字典序排列，$\Theta(n)$ 空间。
- 模式查找：$O(|P|\log n)$ 二分。
- 相邻 LCP 下一课线性算。
- 出处：Manber and Myers, 1993；Kärkkäinen and Sanders, ICALP 2003。

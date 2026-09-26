---
title: 字符串哈希
date: 2026-09-08
section: cs
---

# 字符串哈希

<div class="epigraph">
<p>把串看成 $b$ 进制整数模素数；子串哈希用前缀一次减一次乘，相等则以高概率当相等。</p>
<footer>—— 据 Karp and Rabin, Efficient Randomized Pattern-Matching Algorithms, IBM J. Res. Develop. 1987；Cormen, Leiserson, Rivest and Stein 第 32 章整理</footer>
</div>

[上一课](/cs/palindromic-tree) 与 [SAM](/cs/suffix-automaton)、[SA](/cs/suffix-array) 都是确定性索引，空间或预处理不菲。只需比较若干子串是否相等时，滚动哈希期望 $O(1)$。[前缀和](/cs/prefix-sum-difference) 的形状几乎相同。本课不建 fail。缺口是多项式哈希 / Karp–Rabin：前缀 $H[i]$，子串 $[l,r)$ 为 $H[r]-H[l]\cdot b^{r-l}$。

## 问题

朴素比子串 $\Theta(长度)$。哈希：选底 $b$ 与模 $M$（或双模），碰撞概率约 $\Theta(q^2/M)$ 于 $q$ 次比较（生日）。缺口不是通用散列族全文（后课），而是**串上的滑动与前缀**，使模式匹配、LCP 二分、去重期望变快。

<span class="marginnote">多项式哈希翻译成白话：把字符串当成 b 进制数读出来再取模——「abc」被压成一个数字。比数字只需一次，比逐字符比字符串快；代价是不同串可能压成同一个数，那就是碰撞，概率由模数大小决定。</span>

<span class="marginnote">Karp–Rabin 1987。模 $2^{64}$ 自然溢出不是素数域，对抗输入可打；教学合同写随机模或双质数。</span>

## 方法

预处理 $H$ 与幂 $p[i]=b^i$。比较：哈希相等再可选逐字符验证（确定性保险）。滚动：窗口右移减最左字符加最右。KMP 确定性线性；KR 期望线性、实现短。

```mermaid
flowchart LR
  S["串 S"] --> H["前缀哈希"]
  H --> SUB["H[r] - H[l] * b^{r-l}"]
  SUB --> EQ["相等? 高概率"]
```

与后缀数组：哈希可二分 LCP（每次中点比哈希），期望 $O(|P|\log n)$ 类，常数小；最坏碰撞要承认。

<span class="marginnote">数字实例：取 b=131、M=1e9+7，串「abc」的哈希是 ((1×131 + 2)×131 + 3) mod M——每来一个字符做一次乘加。M 取大素数让碰撞概率约 q²/M，q=10 万次比较时约百万分之一量级。</span>

## 机制

必须固定一种下标与 $b^{len}$。溢出与负差要加 $M$。不要把哈希当加密；本课是数据结构加速。字母映射成 $1..|\Sigma|$，避免前导零歧义。

下一课 Rope 处理的是可编辑大串，不是哈希；编辑会让前缀 $H$ 失效，除非树节点存哈希——那是结合。

```mermaid
flowchart LR
  W["当前窗口哈希 h"] --> SUB["减最左字符：h 减 s[l] 乘 b 的 len-1 次方"]
  SUB --> MUL["整体乘底：h 乘 b"]
  MUL --> ADD["加最右字符：h 加 s[r]"]
  ADD --> MOD["取模：h 对 M 取余"]
  MOD --> CMP{"与模式哈希相等？"}
  CMP -->|"相等"| VER["高概率匹配，可补逐字符验证"]
  CMP -->|"不等"| SHIFT["窗口右移一格重来"]
```

<span class="marginnote">常见误区：把哈希相等当绝对相等。它只是高概率相等——写死不变的模数会被对抗输入构造卡掉，所以合同常用双模数或随机种子；要确定性答案，就在哈希相等后补一次逐字符验证。</span>

## 边界

本课不证明通用散列全部定理。完美散列给静态键集零碰撞，后单元。对抗哈希要用随机种子，合同写期望。

后课默认：静态子串相等可用哈希。可持久大串编辑用 Rope。

## 小结

- 多项式哈希：前缀 $O(n)$，子串 $O(1)$，期望无碰撞。
- 模与底要随机化；可验证。
- 可编辑串结构是 Rope。
- 出处：Karp and Rabin, 1987；Cormen et al. 第 32 章。

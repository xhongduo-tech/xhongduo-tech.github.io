---
title: 正则匹配与回溯爆炸
date: 2026-09-08
section: cs
---

# 正则匹配与回溯爆炸

<div class="epigraph">
<p>正则对应 NFA，可线性于文本；回溯实现在嵌套 `*` 上可指数，ReDoS 是算法选择不是语言必然。</p>
<footer>—— 据 Thompson, Regular Expression Search Algorithm, 1968；Aho, Sethi, Ullman 龙书；[Chomsky](/cs/chomsky-hierarchy) 对照整理</footer>
</div>

上一课[Lyndon 分解](/cs/lyndon-duval)收束线性串技法。正则匹配在主干词法会再遇。本课缺口是**复杂度**：Thompson NFA 模拟 $O(nm)$，回溯引擎最坏指数。不重写正则→NFA 全构造。后课平面几何。线性串课序在此结束。

## 问题

正则语言可用 NFA，文本长 $n$、表达式长 $m$，Thompson 模拟 $O(nm)$。POSIX/Perl 回溯：每个 `*` 分叉，`a?^n a^n` 一类使路径指数。恶意正则 + 文本 = ReDoS。捕获组、回溯引用超出正则，更难。

缺口是实现模型，不是泵引理。

### 不是「正则很慢」

语言类是正则，算法可以是 NFA。慢来自回溯实现。不要把正则语言判成 NPC。

<span class="marginnote">Thompson 1968。Russ Cox 对回溯 vs NFA 的工程论述广泛引用。龙书词法扫描。后课凸包换几何。</span>

<span class="marginnote">数字实例：对 `(a+)+` 匹配 20 个 `a` 后接 `X`。回溯引擎在每个 `a` 上都面临「这层吃还是留给外层吃」的分叉，路径数约 $2^{20}\approx 10^6$；文本翻倍到 40 个 `a`，路径数涨到 $2^{40}\approx 10^{12}$——慢上一百万倍。</span>

## 方法

要最坏线性：Thompson NFA 或 DFA（DFA 可能 $2^m$ 状态）。需要回溯特性才用回溯，并限制或超时。诊断：嵌套量词 + 长失败文本。

```mermaid
flowchart TD
  RE["正则"] --> NFA["Thompson NFA O(nm)"]
  RE --> BT["回溯：最坏指数"]
  BT --> REDOS["ReDoS"]
```

DFA 最小化接[DFA 最小化](/cs/dfa-minimization)。

## 机制

NFA 位集或列表模拟：当前位置集合，每字符扫边。回溯是 DFS 这条 NFA，最坏走遍路径。与 CYK：上下文无关才 $O(n^3)$。与 KMP：单模式无 `*`。

<span class="marginnote">术语翻译：Thompson 模拟就是「把 NFA 所有可能的当前位置收成一个集合，每读一个字符，整组集合一起前进一步」。直觉类比：回溯是派一个人每条岔路都试一遍；Thompson 是每条岔路口各站一个人同时走——步数永远等于文本长度。</span>

```mermaid
flowchart TD
  P["模式 (a+)+ 对文本 aaaaX"] --> S1["回溯：第一个 a+ 吃 1 个 a"]
  S1 --> S2["失败回退：改吃 2 个 a"]
  S2 --> S3["再回退：改吃 3 个 a ……"]
  S3 --> EXP["路径数约 2^n：指数爆炸"]
  P --> T["Thompson：维护同时活的状态集"]
  T --> SET1["读完 a 后：{s2,s3}"]
  SET1 --> SET2["再读 a：{s3,s4}，每步只扫一遍"]
  SET2 --> LIN["O(n·m)，与分叉数无关"]
```

## 边界

本课不写 PCRE 全语义。不写递归正则。后课默认：最坏要线性用 NFA；回溯当工程风险。下一课凸包 Graham / Andrew。

<span class="marginnote">常见误区：初学者以为正则慢是「表达式表达力太强」。正则语言类恰好是能线性匹配的那一层，慢的只是回溯实现。给引擎加超时是止血，换 NFA/DFA 引擎才是根治。</span>

## 小结

- 正则语言可 $O(nm)$ NFA。
- 回溯实现可指数，ReDoS。
- 超正则特征另计代价。
- 出处：Thompson, 1968；龙书。

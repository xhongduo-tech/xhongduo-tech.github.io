---
title: 回文自动机
date: 2026-09-08
section: cs
---

# 回文自动机

<div class="epigraph">
<p>每个本质不同回文一个节点；失配跳到当前回文的最长真回文后缀，加字符只在两端对称时生长。</p>
<footer>—— 据 Rubinchik and Shur, EERTREE: An Efficient Data Structure for Processing Palindromes in Strings, 2018（Eur. J. Comb. / 会议前置版本）；Manacher 线性回文半径整理</footer>
</div>

[上一课](/cs/suffix-automaton) 的状态按结束集合走，不专门压缩回文。[KMP](/cs/kmp) 的 $\pi$ 也不是回文。Manacher 给出每个中心的最长半径，但不索引「本质不同回文」。本课不克隆 SAM。缺口是回文树（eertree / palindromic automaton）：本质不同回文 $O(n)$ 个。

## 问题

串的本质不同回文个数 $\le n+1$（奇偶各一棵树的经典界）。节点 = 一个回文串；转移 $c$ 表示两侧加 $c$。fail 指向最长真回文后缀。在线加字符：从 last 沿 fail 找到能加该字符的回文，没有则新建。缺口是**把回文集合做成自动机**，从而计数、最长回文、回文分割 DP 的转移可沿节点走。

<span class="marginnote">数字实例：串 aaa 的本质不同回文只有 a、aa、aaa 共 3 个，不超过 $3+1$。直觉原因：每在末尾新增一个字符，至多诞生一个「新的」本质不同回文——就是以新字符结尾的最长那个；更短的都会以真回文后缀的形式早出现过。所以节点只有 $O(n)$ 个。</span>

<span class="marginnote">Manacher 1975 线性最长回文。eertree 由 Rubinchik–Shur 系统化（会议 CPM 等，期刊 2018 左右）。本课不发明 arXiv 号。</span>

## 方法

两棵根：偶空与奇「虚」。维护 last。加字符与 SAM 同风格但谓词是回文可扩展。每个节点 `len`。出现次数可挂 fail 树求和。

```mermaid
flowchart TD
  PAL["本质回文"] --> NODE["树节点"]
  NODE --> CH["两侧加字符"]
  NODE --> FAIL["最长回文后缀"]
```

与 Manacher：Manacher 要每个中心的最长；eertree 要集合与转移。两者都线性，问题不同。与 SAM：回文约束更强，节点更少类。

## 机制

新建节点时 fail 必须存在且唯一最长。实现注意奇偶长度。不要用中心枚举 $+ $ 哈希当本课的结构定义——那是后课哈希的应用。

<span class="marginnote">fail 指针就是「最长真回文后缀」指针，可以类比 KMP 的 $\pi$：当前回文匹配不上，就退而求其次跳到「我还是回文」的最短退路。沿着 fail 一路跳，等于把以当前位置结尾的所有回文从长到短串成一条链——这条链正是回文 DP 计数的骨架。</span>

逐字符插入时，沿 fail 链的查找与新建流程如下；它回答的是「一个字符进来，树到底动了哪几处」，与上面那张概念图不同。

```mermaid
flowchart TD
  ADD["读入新字符 c"] --> W["从 last 沿 fail 链走"]
  W --> OK{"当前回文两侧加 c 合法?"}
  OK -->|"是"| EX{"转移 c 已存在?"}
  EX -->|"存在"| LAST["last 移到该子节点"]
  EX -->|"不存在"| NEW["新建节点 记 len"]
  NEW --> F2["再沿 fail 跳 定其最长回文后缀"]
  OK -->|"走完链也没有"| R2["挂到奇根 新建 len=1 节点"]
  LAST --> NEXT["处理下一个字符"]
  F2 --> NEXT
  R2 --> NEXT
```

字符串索引课序接下来从确定性自动机转到期望相等：滚动哈希。

## 边界

本课不把回文树与 SAM 联合结构写完。二维回文不是一维自动机。哈希冲突合同与本课无关。

后课默认：本质回文用 eertree 或 Manacher。要期望比较子串，用字符串哈希。

## 小结

- 回文自动机：本质回文 $O(n)$ 节点 + fail。
- 在线加字符沿 fail 扩展。
- 下一课滚动哈希，不保证零错误。
- 出处：Manacher；Rubinchik and Shur, eertree。

<span class="marginnote">常见误区：初学者容易以为统计回文得把 $O(n^2)$ 个子串全枚举一遍再逐个判回文。实际上本质不同回文只有 $O(n)$ 个，回文树把它们每个收成一个节点——枚举子串是和「按出现位置枚举」比，建树是和「按本质不同集合」比，量级完全不同。</span>

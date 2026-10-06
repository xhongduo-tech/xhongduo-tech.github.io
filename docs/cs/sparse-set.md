---
title: 稀疏集合
date: 2026-09-08
section: cs
---

# 稀疏集合

<div class="epigraph">
<p>稠密数组记下标，稀疏数组存元素；成员测试是一次下标往返，清集只要把 $n$ 置零。</p>
<footer>—— 据 Briggs and Torczon, An Efficient Representation for Sparse Sets, ACM Letters on Programming Languages and Systems, 1993；Knuth 整理</footer>
</div>

[上一课](/cs/bitset)在 $U$ 大时要 $\Theta(U/w)$ 空间与清零。编译器着色、图算法里常有「当前工作集很小、论域下标却到 $10^6$」。[数组](/cs/array-random-access)随机访问仍 $O(1)$。本课不把位图扫字当默认。缺口是 Briggs–Torczon 稀疏集合：两数组不初始化全体，成员测试仍 $O(1)$，清空 $O(1)$。

## 问题

需要：`add`/`remove`/`contains`、迭代当前 $n$ 个成员、以及反复 `clear`。位图 `clear` 是 $\Theta(U/w)$；哈希常数大且不保插入序。稀疏集合：`dense[0..n)` 存元素，`sparse[x]` 若有效则指向 `dense` 中位置。不变式：

$$
x\in S \iff 0\le \mathrm{sparse}[x]\lt n \ \land\ \mathrm{dense}[\mathrm{sparse}[x]]=x.
$$

`sparse` 的垃圾值只要不满足往返就不会被当成成员——因此**不必初始化 `sparse`**。缺口是这个往返测试，不是新的散列。

<span class="marginnote">数字实例：论域 $U=10^6$、当前集合只有 100 个元素：位图清空要扫约 $10^6/64\approx1.6$ 万个机器字，稀疏集合把 $n$ 置零，一条指令；迭代成员也只扫 `dense[0..100)`，与 $U$ 完全无关。</span>

<span class="marginnote">Briggs and Torczon 1993 原文用于编译器活跃集。与并查集的「父指针未初始化」同类：用往返或版本号避免 $\Theta(U)$ 填表。</span>

## 方法

`contains(x)`：读 $i=\mathrm{sparse}[x]$，判断 $i\lt n$ 且 `dense[i]==x`。`add`：已在则返回；否则 `dense[n]=x`，`sparse[x]=n`，$n{+}{+}$。`remove`：与末尾交换并改 `sparse`。`clear`：$n\leftarrow 0$，旧 `sparse` 槽全部失效。迭代扫 `dense[0..n)`。

```mermaid
flowchart TD
  X["候选 x"] --> SP["i = sparse[x]"]
  SP --> CK["i ＜ n 且 dense[i] = x"]
  CK --> IN["在集合中"]
```

空间仍 $\Theta(U+n)$ 的数组容量，但清零与迭代按 $n$。$U$ 必须能分配两个数组；只是不付初始化与按 $U$ 扫描的时间。

<span class="marginnote">直觉类比：`sparse[x]` 指着一张便签；验证 x 在集合里，就顺着便签去 dense 的那个格子看，里面写的必须还是 x。便签是旧垃圾没关系——只要格子里的名字对不上，这张便签就自动作废，无需打扫。</span>

## 机制

与位图：精确、无假阳性；交并要按较小集迭代再 contains，不是字级。与哈希：下标即键，无碰撞。版本号变体：`sparse` 存（位置, 世代），`clear` 只加世代，可避免交换删除时的部分麻烦，本课点名不写完。

```mermaid
flowchart TD
  RM["remove(x)"] --> I["定位 i = sparse[x]"]
  I --> LAST["取 dense 末尾元素 y"]
  LAST --> SW["dense[i] = y 且 sparse[y] = i"]
  SW --> DEC["n 减一"]
  DEC --> INV["往返不变式仍成立"]
  CL["clear"] --> Z["n = 0，旧 sparse 槽全部失效"]
```

这张图回答的是：删除与清空如何在不打扫 `sparse` 的前提下维持不变式——交换删除只改 `dense[i]` 与 `sparse[y]` 两处，`clear` 干脆让全部旧槽因 $n=0$ 失效。

区间查询课序在此收束：序列上从可逆前缀、幂等倍增、可更新树、扫描单调、分块离线，收到位级与稀疏下标。下一单元回到有序字典与堆的对数结构，不重写[红黑直觉](/cs/rbtree-intuition) 的着色 case。

## 边界

键必须是 $[0,U)$ 整数。字符串键先编号。并发无锁稀疏集不是本课。不要把未初始化 `sparse` 读进未定义行为——语言合同须保证读任意位型合法（或先用 mmap 等零页）。

<span class="marginnote">常见误区：初学者容易以为「不初始化 `sparse`」是 bug。恰恰相反：任何垃圾值都过不了往返验证，所以初始化 $\Theta(U)$ 的时间根本不用付——这正是它对比位图、对比 memset 清零最大的优势。</span>

后课默认：小工作集、大下标论域，用稀疏集合。平衡树与随机 BST 的续篇从 Treap 开始。

## 小结

- 稀疏集合：dense/sparse 往返，$O(1)$ 成员与清空。
- 不初始化 sparse；迭代按 $n$。
- 序列区间课序结束；下一课 Treap 接对数字典。
- 出处：Briggs and Torczon, *LOPLAS*, 1993。

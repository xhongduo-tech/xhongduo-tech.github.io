---
title: 树状数组
date: 2026-09-08
section: cs
---

# 树状数组

<div class="epigraph">
<p>每个下标负责一段以自己为右端、长度为最低位 1 的前缀碎片；改一点只走祖先，问前缀只走低位清零。</p>
<footer>—— 据 Fenwick, A New Data Structure for Cumulative Frequency Tables, Software: Practice and Experience, 1994；Cormen, Leiserson, Rivest and Stein 整理</footer>
</div>

[上一课](/cs/sparse-table)把静态幂等查询钉成 $O(1)$，数组一改就整表作废。[前缀和](/cs/prefix-sum-difference)同样经不起单点修改。本课不重讲倍增窗口。缺口是：**可更新的前缀和**——树状数组（Fenwick / binary indexed tree），单点加与前缀和都是 $O(\log n)$。

## 问题

需要维护 $A[1..n]$，操作是 $A[i]\mathrel{+}=d$ 以及查询 $S[r]=\sum_{i=1}^{r}A[i]$。线段树也能做，但节点数约 $4n$、常数更大。Fenwick 用 $n$ 个格子：下标 $i$ 存 $\sum A$ 在区间 $(i-\mathrm{lowbit}(i),i]$ 上的和，其中 $\mathrm{lowbit}(i)=i\,\&\,{-i}$。缺口不是新的群，仍是加法；是**用二进制进位结构代替递归线段**。

区间和仍是 $S[r]-S[l-1]$。区间加可再套一层差分，本课先钉点修前缀查。

<span class="marginnote">数字实例：$\mathrm{lowbit}$ 就是取二进制里最低的那个 1：$6 = 110_2$，lowbit 为 $2$；$12 = 1100_2$，lowbit 为 $4$。于是 $C[12]$ 负责从第 9 到第 12 这 4 个数的和——负责多长，正好等于下标二进制里末尾那个「1」的分量大小。</span>

<span class="marginnote">lowbit 来自补码：$-i=\sim i+1$，与 $i$ 只在最低的 1 及其右侧为零处相交。</span>

## 方法

更新 $i$：当 $i\le n$ 时把 $C[i]$ 加上 $d$，再 $i\mathrel{+}=\mathrm{lowbit}(i)$。查询前缀 $r$：累加 $C[r]$，再 $r\mathrel{-}=\mathrm{lowbit}(r)$，直到 $0$。两次都 $O(\log n)$ 步。下标从 $1$ 起；不要用 $0$，lowbit 会停住。

```mermaid
flowchart TD
  I["下标 i"] --> LB["lowbit = i AND -i"]
  LB --> UP["更新: i += lowbit"]
  LB --> Q["查询: i -= lowbit"]
```

初始化可逐点更新 $\Theta(n\log n)$，或按 $A$ 扫一遍把贡献加到 $C[i]$ 再向上，$O(n)$。

## 机制

每个 $A[j]$ 出现在哪些 $C[i]$：所有 $i\ge j$ 且 $(i-\mathrm{lowbit}(i),i]$ 盖住 $j$ 的那些。沿 $i\mathrel{+}=\mathrm{lowbit}$ 恰好走完覆盖 $j$ 的祖先。查询沿清低位走的是一段段不相交、并起来等于 $[1,r]$ 的块。

```mermaid
flowchart TD
  Q["查询 S 13"] --> C13["13=1101: 取 C13, 覆盖 (12,13]"]
  C13 --> C12["13 清掉最低位得 12: 取 C12, 覆盖 (8,12]"]
  C12 --> C8["12 清掉最低位得 8: 取 C8, 覆盖 (0,8]"]
  C8 --> Z["到 0 停: 三块不相交, 拼出 1..13"]
  U["单点加在 5"] --> C5["C5 加 d"]
  C5 --> C6["跳到 6 = 5 + lowbit 1"]
  C6 --> C8B["跳到 8 = 6 + lowbit 2"]
  C8B --> C16["跳到 16 = 8 + lowbit 8, 停"]
```

<span class="marginnote">直觉类比：查询像用二进制面额的硬币凑零钱——凑 $13 = 8 + 4 + 1$，一次取走面额合适的整块，最多 $\log n$ 枚。更新则相反：往 1 块钱投进点数后，要给所有「装有这枚硬币的钱箱」（高位祖先）都补记账。一个向下拆块，一个向上补账，方向正好相反。</span>

空间 $\Theta(n)$，比线段树瘦。缓存：跳 $\mathrm{lowbit}$ 比连续扫跳得散，但步数只有 $\log n$。二维 Fenwick 同构嵌套，本课不展开。

与[数组随机访问](/cs/array-random-access)：$C$ 仍是数组，只是语义按二进制区间切。

## 边界

运算须可结合、通常要可逆（区间用两个前缀）。最值也可以做，但「单点改成任意值」比「只加」麻烦，实践多换线段树。本课不写树上差分、不写权值树状数组套主席树。

后课默认：点修前缀和用 Fenwick 即可。要任意结合律、区间改、或存整段懒更新，下一课线段树。

<span class="marginnote">常见误区：初学者容易把更新和查询的跳法记反，或用下标 $0$ 起头。$\mathrm{lowbit}(0) = 0$，两个方向都会原地打转死循环——这正是树状数组约定下标从 1 开始的原因。方向记反则更隐蔽：查询会漏块或重复，结果看似合理却不等。</span>

## 小结

- Fenwick：lowbit 切前缀碎片，点修与前缀查询 $O(\log n)$。
- 空间 $n$；下标从 1。
- 更一般的区间运算交给线段树。
- 出处：Fenwick, *Software: Practice and Experience*, 1994；Cormen et al.。

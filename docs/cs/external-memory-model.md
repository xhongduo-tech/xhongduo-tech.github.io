---
title: 外存算法与 I/O 模型
date: 2026-09-08
section: cs
---

# 外存算法与 I/O 模型

<div class="epigraph">
<p>内存 $M$、块 $B$，代价是块传输次数；排序 $\Theta(\frac{N}{B}\log_{M/B}\frac{N}{B})$ 次 I/O，不是 RAM 的 $n\log n$ 比较。</p>
<footer>—— 据 Aggarwal and Vitter, The Input/Output Complexity of Sorting and Related Problems, 1988；Vitter 外存算法综述整理</footer>
</div>

上一课[Misra–Gries](/cs/misra-gries)假定流过 CPU。数据在盘上：随机访问一次块与扫一块同量级贵。缺口是 I/O 模型：扫描 $O(N/B)$，排序多路归并。不重写 RAM 快排。后课 PRAM 是并行，不是盘。

## 问题

参数 $N,M,B$，常 $M\gg B$。扫描顺序文件最优。排序：$\Theta(M/B)$ 路归并，I/O 如题记。B 树、外存优先队列、缓存无关模型（[cache-oblivious](/cs/cache-oblivious) 若已写则引用）点名。图外存更难（边定向）。

缺口是计数 I/O，不是 CPU 比较下界（后课对手）。<span class="marginnote">数字实例：顺序扫 100 GB 盘约几分钟；若改成逐条随机 4 KiB 读，$100\,\text{GB}\div 4\,\text{KiB}\approx 2.4\times 10^7$ 次，按每次 0.1 ms 算就是约 40 分钟。同一批数据，访问模式一换，时间差了好几倍——这正是模型只数「块传输次数」的原因。</span>

### 不是「内存不够就虚拟内存」

OS 分页是 LRU 在线；算法应显式扫块。虚拟内存可能抖动。外存算法自己安排传输。

<span class="marginnote">Aggarwal–Vitter 1988。Vitter 综述。后课 PRAM 前缀和。</span>

## 方法

顺序扫描、多路归并、分块矩阵乘（块 $B^{1/2}$）。图：扫描边列表多次，避免随机点。

```mermaid
flowchart TD
  DISK["N 条记录"] --> BLK["块 B"]
  BLK --> SORT["多路归并 I/O"]
```

缓存无关：不把 $B$ 写入代码，用递归分治。

## 机制

传输一次 $B$ 个元素均摊。比较在内存免费（相对 I/O）。与流：流 $B=1$ 或单次扫描；外存可多遍但要少遍。与 CH：路网可外存预处理。

```mermaid
flowchart TD
  IN["盘上 N 条乱序记录"] --> RUN["第一遍：每次装 M 条进内存排序，写出 N/M 个有序段"]
  RUN --> MG["每一遍归并：同时打开 M/B 个段，各取最小"]
  MG --> DEC["段数每遍除以 M/B，每遍 I/O 为 O(N/B)"]
  DEC --> FIN{"只剩一个有序段？"}
  FIN -->|"否，继续下一遍"| MG
  FIN -->|"是"| OUT["完成：总 I/O 约为遍数 × N/B"]
```

<span class="marginnote">直觉类比：多路归并像在一张小桌子上整理扑克——桌面（内存 $M$）一次只能摊开 $M/B$ 堆牌，每堆各翻最上面一张、挑最小的放走；堆的「路数」受桌面宽度限制，所以 $M/B$ 直接决定要来回几趟。</span>

## 边界

本课不写并行盘阵列全文。不写具体文件系统。后课默认：大数据排序用 I/O 公式。下一课 PRAM 与前缀和。<span class="marginnote">常见误区：初学者容易以为「内存装不下，交给虚拟内存就行」；但换页由 OS 的 LRU 盲目决定，算法的访问模式它看不见，容易来回抖动。外存算法的要义是自己按块安排读写，把 I/O 次数当成显式代价来优化。</span>

## 小结

- 代价是块 I/O；$M,B$ 是参数。
- 外存排序多路归并。
- 随机访问按块计，极贵。
- 出处：Aggarwal and Vitter, 1988。

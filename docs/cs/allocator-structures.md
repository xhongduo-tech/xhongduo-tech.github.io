---
title: 分配器内部结构
date: 2026-09-08
section: cs
---

# 分配器内部结构

<div class="epigraph">
<p>空闲块用隐式边界标签、显式空闲链、或按 size class 的 slab；分配器是数据结构，不是魔法系统调用。</p>
<footer>—— 据 Knuth, The Art of Computer Programming 卷 1；Wilson, Johnstone, Neely and Boles, Dynamic Storage Allocation: A Survey and Critical Review, IWMM 1995；[buddy 分配器](/cs/buddy-allocator) 整理</footer>
</div>

[上一课](/cs/rcu-data-structures) 延迟 `free`，块回到分配器。[buddy](/cs/buddy-allocator) 已给页框。[数组](/cs/array-random-access) 是用户对象布局。本课不 synchronize_rcu。缺口是用户级堆：边界标签、空闲双向链表、分离空闲表、slab/size class。

## 问题

`malloc(n)` / `free(p)` 要 $\Theta(1)$ 或对数找块，减少碎片。隐式：块头存 size 与 busy 位，空闲靠扫描（首次适应）。显式：空闲块内嵌指针成链或树（按大小的 BST/Treap）。分离：每个 size class 一条链，小对象 O(1)。slab：同尺寸对象槽位图或空闲栈。缺口是**把堆管理写成这些结构的组合**，而不是「向 OS 要页」一句。

<span class="marginnote">直觉类比：size class 像餐馆按人数备桌——4 人桌、8 人桌、16 人桌。来了 6 位客人只能给 8 人桌，空两个座就是内碎片；好处是新客人一来，从对应桌子直接领位，不用每次现场把椅子重新拼，这就是小对象 $O(1)$ 分配的来源。</span>

<span class="marginnote">Knuth 边界标签。Wilson et al. 1995 综述适应策略与碎片。页仍可用 mmap/brk；本课内部。</span>

## 方法

分配：size class 上 pop；没有则向更粗粒度要（buddy 页或新 span）。释放：压回 class，或合并相邻空闲（边界标签看前后块 busy）。线程缓存：每线程 magazine 减锁，耗尽再碰中心堆——并发哈希课的分段思想同构。

<span class="marginnote">数字实例：请求 100 字节、按 16 字节对齐量化，class 给你 112 字节，多付 12 字节内碎片；若 class 按 2 的幂划分则给 128 字节，浪费 28 字节（约 22%）。class 划得越密碎片越少，但链表与元数据越多——这是每个分配器都要做的取舍。</span>

```mermaid
flowchart TD
  MALLOC["malloc n"] --> SC["size class 空闲栈"]
  SC --> SPAN["没有则向页/span 要"]
  FREE["free"] --> PUSH["压回 class 或合并邻块"]
```

与 hazard：延迟释放的块仍占 class，直到真正 free。与 LSM：堆不是日志结构文件，但 thread cache 批量交还类似。

## 机制

元数据放块头则用户越界会毁链——这是安全课边界，本课点名。对齐与 size class 量化造成内碎片。本课不写利用分配器的 exploit。

free 的一瞬间，分配器怎么知道能不能把相邻的空闲块拼回去？答案是块头块尾的边界标签：

```mermaid
flowchart TD
  F["free(p)：读块头得本块大小"] --> TAG["边界标签报出左右邻居的忙闲"]
  TAG --> Q{"邻居里有空闲块吗？"}
  Q -->|"左右都忙"| I1["直接按原大小插回空闲链"]
  Q -->|"一侧或两侧空闲"| M["把相邻空闲块拼成一个大块"]
  M --> I2["按新大小插进对应的 class 链"]
  I1 --> DONE["供下次 malloc 领走"]
  I2 --> DONE
```

<span class="marginnote">常见误区：初学者以为堆内存越界写几个字节「顶多脏一点数据」。实际上写坏的是下一个块头里的 size 与 busy 位——分配器之后 free 或合并时会顺着这份被篡改的元数据去改链表指针，崩溃点往往离肇事代码很远，排查起来非常费劲。</span>

最后一课：对象图的引用计数遇环不能只靠分配器 free。

## 边界

本课不把 jemalloc/tcmalloc 源码当词条。内核 SLAB 与用户堆同族。GC 标记清除是另一策略，下一课只钉引用计数与环。

后课默认：堆 = size class + 页 span + 可选合并。对象寿命用引用计数时必须处理环。

## 小结

- 分配器：边界标签、空闲链、size class/slab。
- 并发靠线程缓存与分中心堆。
- 引用计数的环是最后一课。
- 出处：Knuth 卷 1；Wilson et al., 1995；buddy 课。

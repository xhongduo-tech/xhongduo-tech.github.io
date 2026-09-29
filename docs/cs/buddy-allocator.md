---
title: buddy
date: 2026-09-08
section: cs
---

# buddy

<div class="epigraph">
<p>物理内存按 2 的幂分块；分配时把大块劈成一对伙伴，释放时再与伙伴合并，减少外碎片。</p>
<footer>—— 据 Knowlton, A fast storage allocator, CACM 1965；Knuth TAOCP 对 buddy 系统的整理</footer>
</div>

[上一课](/cs/mprotect)改的是用户虚权限。缺页、页表、[大页](/cs/huge-pages) 都要内核拿出**连续物理页**。[置换](/cs/page-replace)释放的也是帧。缺口是内核的页框分配器：不是 `malloc`，是按阶（order）管理空闲块。buddy 是经典实现。

## 问题

若用一张位图每次找 N 个连续帧，扫描太慢。若只用单页空闲表，2MB 大页经常拼不出。buddy：每阶一条空闲链表，块大小 2^k 页，地址对齐。分配 order=n：若该阶没有，从更高阶劈成两个伙伴。释放：看伙伴是否空闲，合并升阶。缺口不是虚地址布局，而是物理连续与对齐。

本课不把每节点 NUMA 的 zonelist 调参写完，只承认 DMA/普通区等 zone。

<span class="marginnote">伙伴关系由地址决定：块基址 xor 块大小即伙伴。合并是 O(阶数)，不是扫全部内存。内碎片：只要了 3 页也得给 4 页。</span>

## 方法

`alloc_pages(order)` 从对应 zone 取块；失败则唤醒回收（后课 shrinker）或对调用者失败。大页需要高阶成功。单页（order 0）是缺页的常见路径。释放必须用原来的 order，否则破坏伙伴不变量。与用户堆对照：用户 brk 切的是虚区间；这里切的是 DRAM 帧。

<span class="marginnote">内碎片的数字实例：DMA 只要 3 页，buddy 只能给 4 页的块（order 2）——第 4 页整页闲置。粒度是 2 的幂，最坏浪费近一半，这笔「空间换速度」的交易换来的是 $O(\text{阶数})$ 的查找与合并，不用扫内存。</span>

```mermaid
flowchart TD
  REQ["要 2^n 页"] --> HIT{"该阶空闲?"}
  HIT -->|有| GIVE["交出对齐块"]
  HIT -->|无| SPLIT["劈更高阶"]
  FREE["释放"] --> MERGE["伙伴空则合并"]
```

## 机制

buddy 让物理连续成为一等公民：DMA、大页、某些内核栈都要它。外碎片表现为高阶链表空、低阶很碎——回收与压缩（后课）试图拼回高阶。不要把 buddy 写成文件系统的块分配；inode 的块位图是另一层。

[分页](/cs/paging-vm) 把这些帧填进 PTE；分配器不关心 VPN。

下图回答一个具体问题：低阶链全空、只剩一块 order-3 空闲时，要 1 页会发生什么。

```mermaid
flowchart TD
  A["要 1 页：order 0 链空"] --> B["取 order-3 块：页 0 到 7"]
  B --> C["劈：右半 页4-7 入 order-2 链"]
  C --> D["左半 页0-3 继续劈"]
  D --> E["右半 页2-3 入 order-1 链"]
  E --> F["劈 页0-1：页 1 入 order-0 链"]
  F --> G["交出页 0"]
```

<span class="marginnote">直觉类比：buddy 像切生日蛋糕只准沿中线切——要小块就把大块对半切到合适为止，切下的每一半都记得「另一半」是谁；将来两半都在空闲链里就能拼回整块。因为切口永远落在 2 的幂位置，另一半的位置用一次异或就算出来，不必查任何账本。</span>

## 边界

本课不引入 CMA、内存热插拔的全部状态机。不保证碎片在长时间运行后仍能给出 1GB 页。内核小对象若直接用 order 0 再自己切，浪费页表级对齐——下一课 slab 在页内切对象。

后课默认：内核能按 2^n 页拿到帧。频繁的几十字节对象不应各占一页，下一课 slab。

<span class="marginnote">常见误区：初学者以为释放时把 order 写错「顶多浪费点空间」。实际 buddy 靠「块大小决定伙伴地址」这条不变量定位另一半；order 错了算出的伙伴是错位地址，会把不相干的块错误合并或漏合并，空闲链表从此悄悄损坏，之后表现成莫名其妙的高阶分配失败。</span>

## 小结

- buddy 用伙伴合并管理 2^n 物理块，服务大页与 DMA。
- 内碎片换来分配与合并的速度。
- 页内小对象缓存是 slab 的缺口。
- 出处：Knowlton, *CACM* 1965；Knuth, *TAOCP*；Bovet and Cesati, *ULK*。

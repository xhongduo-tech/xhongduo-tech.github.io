---
title: 大页
date: 2026-09-08
section: cs
---

# 大页

<div class="epigraph">
<p>用 2MB 或 1GB 的页覆盖同一段虚区间，TLB 里一条就能翻译许多 4K 页曾经占用的项。</p>
<footer>—— 据 Hennessy and Patterson, Computer Architecture: A Quantitative Approach；Navarro 等对超级页的整理</footer>
</div>

[上一课](/cs/page-fault-path)按 4K 补页。[TLB](/cs/tlb-translate) 条目有限：工作集一宽，翻译缺失比数据缺失更先爆。[分页](/cs/paging-vm)允许页大小是 2 的幂，硬件用页表中间层的「叶子」指向大块物理内存。缺口是**何时用大页**：减 TLB 压力，而不把按需策略推倒重来。

## 问题

数据库缓冲池、虚拟机客户内存、科学计算数组，往往是连续 GB 级。若全用 4K，TLB 覆盖范围只有「项数 × 4K」。缺口不是新的 MMU 周期公式，而是：内核把若干连续物理页合成更大粒度，页表少走一层，TLB 以大页标签命中。代价是内碎片、分配失败时的回退，以及缺页一次要填的块变大。

<span class="marginnote">直觉类比：TLB 像你桌面上的便签墙，每张便签只记一个「地址→位置」。4K 页是每 4 KB 的书都要占一张便签；大页是整排书架只贴一张标签。便签数量有限时，标签越粗，不查目录（页表）就能直接拿到的书越多。</span>

本课不把每代 x86 的 PSE/PDPE 控制位写成手册。

<span class="marginnote">透明大页（THP）在运行中折叠或拆开；显式 `mmap`/`hugetlbfs` 由程序申请。折叠需要连续物理内存，与后课 buddy 的高阶块相关。</span>

## 方法

硬件：某级页表项置「大页」位，该项的物理基址对齐到 2MB/1GB，偏移字段变长。[缺页路径](/cs/page-fault-path)若决定用大页，一次分配整块并填一项。用户可 `madvise` 提示；也可映射 hugetlbfs。失败则退回 4K，正确性不变，只是 TLB 覆盖变差。

```mermaid
flowchart TD
  WS["大块连续工作集"] --> HP["大页: 一项覆盖"]
  WS --> BASE["4K: 多项占满 TLB"]
  HP --> TLB["TLB 命中率升"]
```

## 机制

大页把局部性从「页内」扩到「段内」：顺序扫描 2MB 不再换 512 条 4K 翻译。它不改变用户/内核分裂，也不取消按需——可以按大页粒度按需。与 Cache 行无关：大页不是把 Cache 变大，只是翻译变粗。

同一个地址，4K 页与大页的翻译各走多深？

```mermaid
flowchart TD
  VA["同一虚拟地址"] --> W4["4K 页：PGD→PUD→PMD→PTE 四级全走"]
  VA --> W2["2MB 大页：到 PMD 即叶，少走一层"]
  W4 --> M4["TLB 一项只管 4KB"]
  W2 --> M2["TLB 一项管 2MB"]
  M4 --> T["命中率高、遍历更短，翻译成本更低"]
  M2 --> T
```

<span class="marginnote">数字实例：一个典型 TLB 约有 1500 多项。按 4K 算，全部命中也只能覆盖约 6 MB 的工作集；换成 2MB 大页，同样项数覆盖约 3 GB。数据库缓冲池动辄几十 GB，这就是「翻译缺失比数据缺失先爆」的直观账。</span>

多核上拆大页或改权限，要冲掉的 TLB 项更「宽」，下一课 shootdown 会更疼。本课只承认粒度变了。

## 边界

本课不保证大页降低每一种负载：指针追逐、稀疏堆可能浪费物理内存。也不把设备 DMA 对齐问题写完。交换大页要拆或整块写出，实现复杂，主干留到 swap 课点名。不要在此重导多级页表。

后课默认：翻译项可以覆盖 2MB 级块。改页表后其他 CPU 的 TLB 如何作废，下一课 TLB shootdown。

## 小结

- 大页用更少 TLB 项覆盖连续工作集。
- 按需仍在，只是分配粒度变大；可回退 4K。
- 多核作废翻译是 shootdown 的缺口。
- 出处：Hennessy and Patterson, *CA:AQA*；Navarro et al., superpages；Tanenbaum *MOS*。

<span class="marginnote">常见误区：初学者容易以为「开大页必然更快、内存也省」。若访问模式稀疏（比如指针追逐散布在整个堆），2MB 页哪怕只用到几 KB 也独占整块物理内存——内碎片上去了，TLB 收益却没有。大页是连续大工作集的优化，不是无条件开关。</span>

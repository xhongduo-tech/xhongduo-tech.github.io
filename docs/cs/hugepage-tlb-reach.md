---
title: 大页对 TLB 覆盖
date: 2026-09-08
section: cs
---

# 大页对 TLB 覆盖

<div class="epigraph">
<p>一项 2MiB 映射等于 512 个 4KiB 项；TLB miss 与 PWC 压力一起下降，但内部碎片和预留代价交给 OS。</p>
<footer>—— 据 Navarro et al., Practical, Transparent Operating System Support for Superpages, OSDI 2002；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/tlb-hierarchy-pwc) 用更多项和 PWC 扩大翻译覆盖。项数受面积限制。本课不重画四级 walk。缺口是**大页（superpage / hugepage）：增大页大小，让同一 TLB 几何覆盖数兆字节到数吉字节。** 顺带放宽 [VIPT](/cs/vipt-cache) 的索引约束。

## 问题

数据库、JVM、科学数组的工作集远大于 L2 TLB × 4KiB。于是大量时间花在 walk 上，[CPI](/cs/cpi-amdahl) 的翻译项显形。缺口不是再加 4 倍 TLB 项，而是**让一项代表连续对齐的大物理区**。OS 必须找到连续物理内存并维护拆页/合并。

<span class="marginnote">拿数字感受缺口：一个 8 GiB 工作集的数据库，用 4KiB 页需要约两百万个 TLB 项才能全装下——而典型 L2 TLB 只有一两千项；换成 2MiB 大页，4096 项就够了。这就是为什么加几倍 TLB 项救不了它。</span>

<span class="marginnote">RISC-V Sv39 有 2MiB、1GiB 叶。x86 有 2MiB/1GiB。透明大页（THP）尝试在运行时提升，失败则保持 4KiB。</span>

## 方法

硬件：TLB 项带页大小字段，比较时掩码 VA 低位。同时查找多种大小（分裂阵列或并行 CAM）。缺失 walk 在叶上发现大页则填大项。I/D 与 ASID 规则同小页。

<span class="marginnote">「掩码 VA 低位」可以想象成 TLB 项自带一把可调齿距的梳子：页越大齿距越粗。比较地址时先按页大小把低位抹掉，剩下高位相同就算命中，于是一项天然覆盖大页内的所有地址——齿距换了，梳子还是那把。</span>

```mermaid
flowchart TD
  VA["VA"] --> LKP["按多种页大小查 TLB"]
  LKP -->|"大页命中"| PA["少项覆盖大区"]
  MISS["缺失"] --> WALK["walk 遇大叶则填大项"]
```

## 机制

覆盖：512× 或 262144× 的数量级变化。VIPT：页内位移变长，L1 可以在不引入 [别名](/cs/cache-aliasing) 的前提下做得更大。代价：写保护、COW、NUMA 迁移要以大页为粒度，细粒度保护变难；内部碎片浪费 DRAM。预取与 [行大小](/cs/line-size-sector) 仍按 cache 行，大页不改变 64B 传输。

<span class="marginnote">初学者容易以为大页能让缓存和预取一起提速——实际上 cache 行仍是 64B，预取器照旧按行工作；大页省的只是翻译（TLB 查找和页表遍历），不改变数据在缓存里的布局。内部碎片的账也很好算：为 100 KB 的数组升到 2MiB 大页，约 1.9 MiB 物理内存被白白占住。</span>

```mermaid
flowchart LR
  T["同一颗 1536 项 L2 TLB"] --> A["全部装 4KiB 页"]
  T --> B["全部装 2MiB 页"]
  A -->|"1536 × 4KiB"| C["覆盖约 6 MiB"]
  B -->|"1536 × 2MiB"| D["覆盖约 3 GiB"]
```

混合大小：L1 TLB 常把 4KiB 与 2MiB 分阵列并行查，避免一种大小的 CAM 把另一种挤掉。1GiB 项很少，多在 L2 TLB。拆页（promoted 页被写保护切碎）会瞬间把覆盖吐回去，PMU 上表现为 TLB miss 尖峰。

## 边界

本课不把 Linux `mmap`/`madvise` 写成教程。也不保证大页降低一致性流量——假共享仍按 cache 行。下一课回到多核副本：MESI 不够用时引入 Owned 与转发。

后课默认：翻译覆盖可用大页指数级放大。一致性协议的状态机下一课从 MESI 加 O/F。

## 小结

- 大页用一项覆盖连续大区，TLB/PWC 压力下降。
- OS 负责连续物理页与碎片；硬件负责多大小查找。
- MESI 之上的 Owned/Forward 是下一课。
- 出处：Navarro et al., *OSDI*, 2002；Hennessy and Patterson, *CA:AQA*。

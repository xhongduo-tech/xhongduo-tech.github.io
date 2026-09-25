---
title: IOMMU
date: 2026-09-08
section: cs
---

# IOMMU

<div class="epigraph">
<p>IOMMU 给设备一张自己的页表：DMA 使用 I/O 虚地址，内核限制设备能写到哪些物理页。</p>
<footer>—— 据 AMD/Intel IOMMU 架构概述；Tanenbaum 对 DMA 保护的整理</footer>
</div>

[上一课](/cs/io-dma)让设备按物理地址写帧，并点到 IOMMU。[分页](/cs/paging-vm) 保护的是 CPU 路径；设备作为总线主设备不走用户页表。缺口是**设备翻译与隔离**：错误的描述符或恶意设备不应写任意 DRAM。本课讲机制，不写攻击步骤。

## 问题

无 IOMMU：DMA 地址 = 物理地址（或 bounce）。外设或驱动 bug 可以覆写内核。有 IOMMU：驱动为一次传输建立 IOVA→PA 映射，设备只看见 IOVA。传输结束解映射。缺口不是重导 CPU 页表格式，而是：另一套翻译对象，权限更粗，与 [TLB](/cs/tlb-translate) 类似也有 IOTLB，失效要 shootdown 设备侧缓冲。

本课不把 VT-d 与 AMD-Vi 的寄存器级编程当考纲。

<span class="marginnote">VFIO/设备直通把 IOMMU 分组借给客机，虚拟化课会再提。这里只要求：CPU 分页管 CPU，IOMMU 管设备。</span>

## 方法

驱动：dma_map 把 bio 的页做成 IOVA，填入设备描述符。IOMMU 硬件按设备源 ID 选上下文表，走 I/O 页表。非法 IOVA 可中止并中断。完成：dma_unmap，IOTLB 作废。与 buddy 钉页规则同时成立：映射期间页不得被置换。

```mermaid
flowchart TD
  DEV["设备 DMA"] --> IOVA["I/O 虚地址"]
  IOVA --> IOPT["IOMMU 页表"]
  IOPT --> PA["允许的物理页"]
```

## 机制

IOMMU 把 DMA 从「信任设备」改成「设备也有地址空间」。这与用户/内核分裂平行：都是翻译上的权限，主体不同。性能代价是映射建立与 IOTLB 缺失。不要在此写如何绕过 IOMMU。

```mermaid
flowchart TD
  SUB["无 IOMMU: DMA 地址=物理地址"] --> ANY["坏描述符/恶意设备可写任意 DRAM"]
  MAP["dma_map: 建立映射"] --> TBL["按设备源 ID 查上下文表"]
  TBL --> OPT["I/O 页表: IOVA->PA + 权限"]
  OPT --> HIT{"IOVA 映射过?"}
  HIT -->|"是"| OK["设备只能写允许的页"]
  HIT -->|"否"| FAULT["中止事务 + 上报内核"]
```

<span class="marginnote">术语翻译：IOVA（I/O 虚拟地址）就是「设备眼里的假地址」。设备在描述符里填 IOVA，IOMMU 在总线上把它翻译成真物理地址——和 CPU 的 MMU 用虚拟地址换物理地址是同一个把戏，只是查表的主语从 CPU 换成了外设。</span>

<span class="marginnote">常见误区：初学者容易以为有了 IOMMU 性能白赚。映射和解除映射本身是软件开销，IOTLB 未命中还会拖慢每次 DMA 传输；所以驱动对同一块网卡的接收环通常映射一次反复用，而不是每包都 map/unmap。</span>

<span class="marginnote">为什么重要：一个 DMA 描述符填错地址，设备不会报错，而是安静地把数据覆写到内核任意物理页上——可能覆盖到别的进程甚至内核代码。没有 IOMMU 时，这种「写飞」是最难排查的安全漏洞来源之一。</span>

块层 bio 仍描述页；IOMMU 是 DMA API 之下的翻译。

## 边界

本课不引入 ATS/PRI 等 PCIe 扩展全文。不保证所有平台有 IOMMU 或已开启。下一课离开块与 DMA，进入进程间的异步通知：信号。

后课默认：设备 DMA 可被限制在映射过的页。针对进程的软件中断，下一课信号。

## 小结

- IOMMU 为 DMA 提供 IOVA 翻译与设备隔离。
- 映射期必须钉页；IOTLB 要作废。
- 进程级异步事件是信号课的缺口。
- 出处：Intel VT-d / AMD-Vi 概述；Tanenbaum *MOS*；Linux DMA API。

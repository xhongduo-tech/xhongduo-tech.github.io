---
title: MSI / MSI-X 中断
date: 2026-09-08
section: cs
---

# MSI / MSI-X 中断

<div class="epigraph">
  <p>设备不再拉一根 INTx 线，而是写一个约定的内存地址：这条 Memory Write TLP 被主机构成中断，向量在消息数据里。</p>
  <footer>—— 据 PCI-SIG PCI Express Base Specification；Intel 64 and IA-32 Architectures Software Developer’s Manual 整理</footer>
</div>

[上一课](/cs/pcie-tlp-bar)让设备能发 Memory Write。组成/OS 课的 [PLIC](/cs/plic-irq) 是线汇聚模型。缺口是 PCI **消息信号中断**：MSI 与 MSI-X 用写事务代替边沿/电平线，向量可以很多。

## 问题

传统 INTx 共享、电平、桥上虚拟线，扩展性差。MSI：配置能力里写 Message Address/Data，设备发一条写。MSI-X：独立表（常在 BAR）每向量各地址/数据，可掩码，向量数远大于 MSI。缺口不是 TLP 头格式重讲，而是：**中断=定向写**，与 DMA 写同类，只是地址落在中断控制器接收窗口（如 x86 的 FEE0_xxxx 或 IOMMU/APIC 区域）。

RISC-V 平台可用 MSI 接到 IMSIC 或经转换接到 PLIC，细节因平台而异，本课钉 PCI 侧合同。

### MSI 不是「更快的轮询」

没有消息时 CPU 不因该设备进中断；轮询是软件读门铃。把 MSI-X 理解成硬件自动 poll BAR，中断亲和与节能模型全错。过多向量浪费表项；过少则共享处理函数。

<span class="marginnote">MSI 就像把「按门铃」换成「往指定信箱投一张写着编号的纸条」：门铃只有响与不响，多户人家还得开门猜是谁；纸条自带身份，主机一看编号就知道是哪个设备、哪个事件，不用共享、不用轮询猜人。</span>

<span class="marginnote">PCI-SIG 定义 MSI/MSI-X 能力结构。Intel SDM 描述本地 APIC 如何接收这些写。下一课 APIC 路由。Linux 驱动写 `pci_alloc_irq_vectors` 一类，本课不背 API。</span>

## 方法

枚举时 OS 分配向量，编程 MSI-X 表，使能。设备事件：写对应表项的地址/数据。主机：APIC 投递到目标 CPU，软件在处理函数里读设备原因寄存器（仍经 BAR）。与 DMA 写序：描述符写完再发 MSI，通常靠 PCIe 的同路径 Posted 序或显式围栏——后课 DMA 再钉。

```mermaid
flowchart TD
  EVT["设备事件"] --> WR["Memory Write 到 MSI 地址"]
  WR --> APIC["主机中断控制器窗口"]
  APIC --> CPU["目标 CPU 向量"]
  CPU --> LATER["后课：APIC 如何选核"]
```

错误消息（AER）也走 Message TLP，不是 MSI 表，点名区分。

## 机制

下一课 APIC/IOAPIC/x2APIC 决定向量落到哪颗核，与 [irq affinity](/cs/irq-affinity) 软件策略衔接。本课只把 PCI 设备从「拉线」换成「写消息」。

给个量级：INTx 一共只有 4 条传统线，多个设备共享一条就得靠软件逐一排查是谁；MSI 一个设备最多 32 个向量，MSI-X 表可到 2048 项——一张多队列网卡给每个收包队列单独配一个中断，都绰绰有余。这是「向量可以很多」的具体含义。

```mermaid
flowchart TD
  TBL["MSI-X 表（在 BAR 内，OS 已编程）"] --> E0["表项 0：地址 A0 / 数据 D0"]
  TBL --> E1["表项 1：地址 A1 / 数据 D1"]
  TBL --> E2["表项 2：地址 A2 / 数据 D2，已掩码"]
  RXQ["网卡收包队列 0 完成"] -->|"写 A0 配 D0"| E0
  TXQ["发送队列空"] -->|"写 A1 配 D1"| E1
  E0 --> C0["投递到核 0 的向量"]
  E1 --> C1["投递到核 1 的向量"]
  E2 -.->|"被掩码：不发消息"| NONE["不产生中断"]
```

初学者容易以为 MSI 是设备「抢总线插嘴喊一嗓子」的特殊操作——实际上它就是一次普通的 Memory Write TLP，照常排队、照常受流控；主机唯一 special 的地方，是「写的地址落在中断接收窗口」这一条被认成中断，其余与 DMA 写别无二致。

<span class="marginnote">常见误区：把 MSI-X 的 2048 项当成「性能上限」——上限意义在「每个队列独立中断、免共享一条线的锁」，队列用不满 32 项的设备选 MSI 已足够；为用满表项而人为拆分队列纯属浪费。</span>

## 边界

本课不写 IOMMU 中断重映射表的全部字段，不把虚拟 MSI 注入写成完整 VFIO 课。不讨论中断风暴的全部缓解。

后课默认：PCIe 设备用 MSI/MSI-X 写消息投递中断；向量在表里。

## 小结

- INTx 共享线被 MSI 写事务替代。
- MSI-X 每向量独立地址/数据，表在 BAR。
- 投递窗口由主机中断控制器定义。
- 出处：PCI-SIG PCIe Base Spec；Intel SDM。

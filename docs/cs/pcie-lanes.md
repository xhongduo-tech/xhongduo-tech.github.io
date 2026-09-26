---
title: PCIe 层次与 lane
date: 2026-09-08
section: cs
---

# PCIe 层次与 lane

<div class="epigraph">
  <p>PCIe 不是共享并行总线：点对点串行 lane 组成链路，分层为物理、数据链路与事务，设备各占自己的宽度。</p>
  <footer>—— 据 PCI-SIG PCI Express Base Specification；Hennessy and Patterson, CA:AQA；Patterson and Hennessy, Computer Organization and Design 整理</footer>
</div>

[上一课](/cs/hdd-geometry)把块设备停在内部几何。组成课的[总线与 MMIO](/cs/bus-mmio) 是共享地址数据总线教学模型。缺口是当代互连 **PCIe**：lane、链路宽度 ×N、分层，让 SSD、GPU、网卡接到 CPU 根复合体。

## 问题

PCI 并行总线仲裁、反射、频率上不去。PCIe：每 lane 差分串行，×1/×4/×16 并行多 lane。分层：物理层（串扰、训练、8b/10b 或 128b/130b）、数据链路（ACK/重传、DLLP）、事务层（TLP，下一课）。缺口不是再画教学总线，而是**点对点交换机**把根端口接到端点。

链路训练协商速率与宽度。热插拔与电源状态点名。本课不把 SerDes 模拟前端画成光刻。

### lane 不是「内存通道」

DRAM 通道有命令/地址/数据时序；PCIe lane 是包交换串行链路。把 ×16 当成另一条 DDR 通道，一致性与地址译码会错——设备用 BAR 映射，不是 JEDEC 行地址。HBM 更不是 PCIe。

<span class="marginnote">PCI-SIG 基规范是权威。CA:AQA 与 COD 讨论 I/O 互连演进。本课用规范层次，不背每代 GT/s 表当作业。</span>

## 方法

根复合体发出配置与内存 TLP。交换机基于地址/ID 路由。端点（SSD 控制器、NIC）实现 BAR。带宽粗算：每 lane 有效速率 × 宽度 × 方向（全双工）。开销：编码、DLLP、TLP 头，使有效载荷低于标称。

<span class="marginnote">数字实例：PCIe 3.0 每 lane 标称 8 GT/s，经 128b/130b 编码后约 985 MB/s。你插一张 ×4 的 NVMe SSD，理论上限约 985 × 4 ≈ 3.9 GB/s——再快的盘插在 ×1 的槽上也只有约 1 GB/s，宽度直接封顶带宽。</span>

```mermaid
flowchart TD
  RC["根复合体"] --> PHY["物理 lane"]
  PHY --> DL["数据链路 ACK"]
  DL --> TL["事务层 TLP"]
  TL --> LATER["后课：TLP 类型与 BAR"]
```

NVMe SSD 是端点，内部再接 NAND 并行；PCIe 只负责到主机内存/CPU。

## 机制

下一课事务：Cfg/Mem/IO/Message，BAR 窗口把 MMIO 落到设备寄存器。MSI-X 用 TLP 写中断。本课先把「串行分层链路」钉住，替代教学共享总线的频率神话。

```mermaid
flowchart LR
  subgraph OLD["旧世界：共享并行总线"]
    CPU1["CPU"] --- BUS["一条共享总线"]
    BUS --- D1["设备 1"]
    BUS --- D2["设备 2"]
    BUS --- D3["设备 3"]
  end
  subgraph NEW["新世界：PCIe 点对点"]
    CPU2["根复合体"] ---|专用 lane| SW["交换机"]
    SW ---|lane ×N| E1["SSD"]
    SW ---|lane ×1| E2["网卡"]
    SW ---|lane ×16| E3["GPU"]
  end
```

这张图回答的问题是：为什么需要一个「交换机」？共享总线上所有设备抢同一组线，必须先仲裁；PCIe 里每个端点独享一条到交换机的链路，可以同时各自收发。交换机按地址决定每个包往哪个下游端口转发——这就是「点对点」三个字的实际含义。

<span class="marginnote">直觉类比：lane 就像一条双向两车道的专用匝道——发送一对差分线走一个方向、接收一对走另一方向，永远不用等对面来车；链路宽度 ×N 就是并排修 N 条这样的匝道。</span>

## 边界

本课不讲 CXL 全缓存一致协议（点名为 PCIe 物理上的近亲即可）。不把以太网 MAC 当 PCIe。不写光模块。不进入大模型多卡拓扑品牌。

<span class="marginnote">常见误区：以为「×16 的槽插 ×4 的盘会烧坏」或「插上就一定跑满 ×16」。实际上链路宽度是开机时训练协商出来的：双方按各自支持的最大宽度取交集，×4 设备在 ×16 槽里就跑 ×4，互不伤害，只是带宽没吃满。</span>

后课默认：PCIe 是分层点对点串行互连；宽度用 lane 数表示。

## 小结

- 共享并行 PCI 被点对点 PCIe 替代。
- 物理 / 数据链路 / 事务三层；lane 可聚合。
- 下一课才是 TLP 与 BAR 地址窗口。
- 出处：PCI-SIG PCIe Base Spec；Hennessy and Patterson, CA:AQA；Patterson and Hennessy, COD。

---
title: 时钟虚拟化
date: 2026-09-08
section: cs
---

# 时钟虚拟化

<div class="epigraph">
<p>客户 TSC 与时钟设备若按真实硬件跑，迁移与偷时间会让时间跳；KVM 用 pvclock/kvm-clock 或缩放 TSC 给一个稳定故事。</p>
<footer>—— 据 Linux kvmclock；Xen pvclock；[tickless](/cs/tickless-nohz) 与 [cyclictest](/cs/latency-measurement) 为时间先修</footer>
</div>

[posted interrupt](/cs/interrupt-virtualization) 处理 IRQ。[timer wheel](/cs/timer-wheel) 在客户内核另有一份。缺口是 **客户看见的时间**：TSC 偏移、偷时间、迁移。

## 问题

TSC 在迁移到另一台主机时不同步。无 pv：NTP 疯。kvm-clock：共享页写宿主持续时钟。缺口：`clocksource` 选择；与 [NO_HZ](/cs/tickless-nohz) 客户。本课不把 NTP 算法重写。<span class="marginnote">术语翻译：TSC（时间戳计数器）是 CPU 里一个随时钟周期不断加一的寄存器，软件读它拿高精度时间。麻烦在于它数的是「这台物理 CPU 的周期」——虚拟机一搬到别的主机，计数起点就换了，客户看见的时间会突然跳一大截。</span>

<span class="marginnote">TSC scaling 硬件可乘偏移。对象是客户时间，不是 RTC 芯片更换。</span>

## 方法

QEMU/KVM 提供 kvm-clock 页或 PIT/HPET 模拟。对照 [半虚拟](/cs/paravirtualization)：时钟也是 PV。对照 DAX：无关。对照 [cpuidle](/cs/cpuidle-cstate)：宿主编排 halt，客户 TSC 仍应前进策略一致。

```mermaid
flowchart TD
  H["宿主稳定时钟"] --> PV["pvclock 共享页"]
  PV --> G["客户 clocksource"]
  MIG["迁移"] --> ADJ["TSC 偏移调整"]
```

## 机制

时钟虚拟化把「时间是共享真理」在 VM 里重新定义，使迁移与超售（偷时间）不毁掉客户内核会计。超售过度仍会让客户觉得 CPU 被偷。不要写成相对论。与 [fairness](/cs/fairness-metrics)：客户 CFS 看自己的时钟。

```mermaid
flowchart TD
  MIG["VM 从主机 A 迁到主机 B"] --> DIFF{"两台主机的 TSC 读数是否相同？"}
  DIFF -->|"否"| OFF["算出偏移量写进 pvclock 共享页"]
  OFF --> SW["客户每次读钟统一加上偏移"]
  SW --> SAME["客户内核看见时间连续，会计不乱"]
  DIFF -->|"是"| RAW["裸 TSC 可直接沿用"]
```

<span class="marginnote">这张图回答「迁移那一刻时间为什么没有跳」：宿主在共享页里交出「偏移量 + 缩放系数」这两个数，客户每次读钟都按它折算。前提是切换要原子——偏移换到一半被客户读到，时间照样倒流。</span>

错误 clocksource 导致客户 hang 或时间倒流检测。<span class="marginnote">常见误区：以为客户里选哪种 `clocksource` 只是性能口味。选错时钟源（比如在一台 TSC 不可信的机器上硬用 TSC），轻则 NTP 反复拉锯校不准，重则时间倒流触发内核告警，定时器、超时、负载统计全面失灵——时间源是内核会计的地基。</span>

<span class="marginnote">直觉类比「偷时间」：你按小时租了跑步机，前台却时不时把你挪开给别人让位——你没跑满时长，账单却照算。宿主超售 CPU 时客户被挂起而不自知，所以要把 steal 时间显式报给客户，客户内核的负载统计才不至于把「没分到 CPU」误判成「程序自己慢」。</span>


实现上：偷时间（steal）要报给客户，否则负载计算偏了。迁移瞬间 TSC 偏移必须原子切换。客户选 tsc 还是 kvm-clock 决定能否跨机。 读法上只引用[上一课](/cs/interrupt-virtualization)的结论，不把对象换成训练推理或限价簿。

本课在操作系统进阶的「虚拟化与隔离进阶 / Hypervisor」课序里，对象是 **时钟虚拟化**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入所有 VDSO 时钟。不保证嵌套 VM 的 TSC。下一课内存超售：气球。


版本字段会变，课序钉的是机制对象「时钟虚拟化」，不是某一主线内核的结构体名。
后课默认：客户时钟靠 PV 或 TSC 缩放。用气球收回客户内存，下一课。

## 小结

- 客户时钟需 PV 或 TSC 偏移，才能迁移与超售。
- kvm-clock 是共享页协议。
- 内存气球是下一课。
- 出处：kvmclock；Xen pvclock；KVM。

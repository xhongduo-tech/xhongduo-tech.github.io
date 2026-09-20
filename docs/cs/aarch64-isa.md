---
title: ARM AArch64
date: 2026-09-08
section: cs
---

# ARM AArch64

<div class="epigraph">
  <p>A64 定长 32 位，31 个通用寄存器加零寄存器，load/store，条件码与可选的条件执行残留在比较与分支上，而不是 x86 那种变长外壳。</p>
  <footer>—— 据 ARM Architecture Reference Manual for A-profile；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/x86-microcode-decode)展示了 CISC 前端税。缺口是另一条工业主线 **AArch64**：定长、宽寄存器堆，作为后课与 RISC-V 对比的一端。不把 ARMv7 Thumb 变长当本课主体。

## 问题

A64：4 字节对齐指令，`Xn`/`Wn`，`XZR`/`WZR` 读零写丢弃（类似 `x0`）。寻址：基址+偏移、预/后变址、PC 相对。条件码 NZCV 在 `PSTATE`，分支可条件。缺口不是 μop ROM，而是这套**程序员模型**：异常级 EL0–EL3、SPSR、VBAR，对照 RISC-V 的特权与 `mtvec`。

NEON/SVE 是 SIMD/向量，后课再接。本课钉整数与系统寄存器骨架。

### AArch64 不是「安卓专用 ISA」

它是 A-profile 64 位架构，服务器与桌面同样存在。把 ARM 当成手机品牌或 GPU，对照课会失去指令级内容。微结构（乱序宽度）与 ISA 分开，和 x86 一样。

<span class="marginnote">ARM ARM 是规范。CA:AQA 用 ARM 作 RISC 例。PCS（Procedure Call Standard）是 ABI，最后一课边界再收。本课不背全部系统寄存器名。</span>

<span class="marginnote">术语翻译：XZR/WZR 就是「读它永远得到 0、往它写任何值都被丢弃」的零寄存器。编译器拿它当免费垃圾桶——比如把不想要的结果写进 WZR 丢掉，省下一个真正的通用寄存器。</span>

<span class="marginnote">数字实例：定长 32 位意味着取到第 N 条指令后，PC 直接加 4 就是下一条的位置；x86 指令从 1 到 15 字节不等，必须逐字节译码完才知道下一条从哪开始。这就是上一课「CISC 前端税」在这边的对应减免。</span>

## 方法

取指对齐 4 字节，译码组合字段（比 x86 短）。`ldr`/`str` 可带扩展/移位的索引。立即数逻辑编码是特殊位模式，不是任意 32 位——编译器选 `mov` 序列。原子：`ldxr`/`stxr` 在后课 LR/SC。

```mermaid
flowchart TD
  A64["定长 32 位"] --> GPR["X0-X30 + XZR"]
  A64 --> LS["load/store 变址"]
  A64 --> EL["EL0-EL3"]
  EL --> LATER["后课：与 RISC-V 并排对照"]
```

复位：实现定义入口，固件（TF-A）类似 UEFI 角色。GIC 对照 APIC。

## 机制

下一课明确 ARM vs RISC-V：特权、压缩、向量哲学、生态。本课只让 A64 成为可引用的 ISA，而不是「另一种 x86」。条件码相对 RISC-V 的显式比较分支是差异点，谓词课再扩。

第一张图画的是 A64 这个 ISA 静态上有哪些部件；这张图回答第二个问题：一条带变址的 `ldr` 在硬件里走一条什么路，定长译码在哪个环节省事。

```mermaid
flowchart TD
  PC["PC 对齐取指：一次 4 字节"] --> DEC["定长译码：按字段直接切分"]
  DEC --> REG["读寄存器堆：基址 X1"]
  REG --> AGU["地址生成：基址加偏移，回写变址"]
  AGU --> MEM["访存"]
  MEM --> WB["写回 X0，NZCV 不动"]
  WB --> NEXT["PC 加 4 即下一条"]
```

<span class="marginnote">常见误区：初学者容易把异常级 EL0–EL3 想成「性能挡位」。实际上它是一条特权阶梯：EL0 是普通应用，EL1 是操作系统内核，EL2 是虚拟机监控器，EL3 负责安全与非安全世界之间的切换。层级越高能碰的系统寄存器和内存越多，与跑得快慢无关。</span>

## 边界

本课不写 M-profile 微控制器，不把 TrustZone 世界切换写成完整安全课。不列每家微结构的流水线。

后课默认：AArch64 是定长 RISC 式 load/store ISA，带 NZCV 与异常级。

## 小结

- A64：32 位定长、31 GPR、load/store。
- 异常级与系统寄存器构成特权。
- 前端税低于 x86 变长。
- 出处：ARM ARM A-profile；Hennessy and Patterson, CA:AQA。

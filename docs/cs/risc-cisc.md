---
title: RISC 与 CISC
date: 2026-09-08
section: cs
---

# RISC 与 CISC

<div class="epigraph">
<p>把内存操作收成 load/store、指令定长、寻址很少，译码才能是组合；复杂指令把这些藏进微码，换的是编码密度与历史兼容。</p>
<footer>—— 据 Patterson and Hennessy, Computer Organization and Design (RISC-V) 整理</footer>
</div>

上一课[寻址方式](/cs/addressing-modes)钉死了 RV32I 只有很少几种有效地址。本课不重列基址+偏移，也不从 1970 年代 ISA 战争写综述。缺口是把「为什么故意少」收成原则：**RISC** 让后课单周期/流水线可画；CISC 用微码或多拍吃掉复杂寻址。

## 问题

CISC：变长编码、一条指令里存储器到存储器、丰富寻址，硬件译码或微程序。<span class="marginnote">常见误区：把 CISC 读成「复杂所以落后」。变长编码换来的是代码密度——同一段逻辑 CISC 常用更少字节表达，早期内存贵、今天 I-cache 紧张时都有价值。这是一笔交换，不是输赢。</span>[多周期与微程序](/cs/multicycle-microcode)已有直觉，本课提前用它当对照。RISC：定长、load/store、运算只在寄存器、简单寻址、可见寄存器多。<span class="marginnote">「运算只在寄存器」可以想象成厨房规矩：食材必须先搬上灶台（寄存器），锅里的操作只许在灶台上进行；RISC 不允许「边切菜边从冰柜取肉」（存储器操作数直接进运算）。</span>缺口不是品牌点名，而是组成后果：RISC 的关键路径短、控制表小；CISC 的 CPI 与取指都更烦。

当代 x86 在内部译成 RISC 式微操作，程序员模型仍 CISC。本课承认这层，不把解码缓存写完。

### RISC 不是「指令条数少所以快」

指令条数可能更多（要把复杂操作拆开）。快来自容易流水、易编译、易满足时序。用「精简=更少 opcode」衡量，会把 RISC-V 的扩展误解成背叛。

<span class="marginnote">Patterson/Hennessy 用 RISC-V 贯穿。本栏从此以 RV32I 为运行 ISA。CISC 只作对照，不把 x86 当主干指令表。</span>

## 方法

对照表：编码长度、访存是否掺进运算、寻址种类、控制实现（组合 vs 微码）。RISC-V 再加：`opcode` 位置固定，便于[译码器](/cs/decoder-encoder)硬连。

```mermaid
flowchart TD
  MODES["寻址与编码复杂度"] --> CISC["微码或多拍译码"]
  MODES --> RISC["定长 load/store"]
  RISC --> RV["下一课：RV32I 语义"]
```

## 机制

后课整数指令只给寄存器与简单访存的语义，硬件才能在单周期图上接线。伪指令、CSR 是汇编与特权的薄层，不把 ISA 变回 CISC。性能比较必须用同一基准上的 CPI×$T$×指令数，不能只比指令条数。<span class="marginnote">数字实例：程序 A 用 100 亿条指令、CPI 1.2；程序 B 用 80 亿条、CPI 1.8；主频同为 3 GHz。A 的用时 $=100\mathrm{e}9\times1.2/3\mathrm{e}9=40$ 秒，B $=80\mathrm{e}9\times1.8/3\mathrm{e}9=48$ 秒——指令更少反而更慢。</span>

```mermaid
flowchart TD
  P["同一程序"] --> I1["RISC：指令条数偏多"]
  P --> I2["CISC：指令条数偏少"]
  I1 --> C1["CPI 低，T 可拉高：易流水、易定时序"]
  I2 --> C2["CPI 高，取指与译码复杂"]
  C1 --> EQ["总时间 = 指令数 × CPI × 1/T"]
  C2 --> EQ
  EQ --> W["只比指令条数会得出相反结论"]
```

## 边界

本课不评哪家赢了市场，不把 VLIW、栈机展开。压缩 16 位 `C` 是密度扩展，语义仍 RISC，不在本课。

后课默认：主干 CPU 是 RISC、load/store、定长 32 位；下一课给出 RV32I 每条做什么。

## 小结

- RISC：定长、load/store、少寻址、组合译码。
- CISC：复杂编码与寻址，代价在微码/多拍。
- 「精简」指硬件规则，不是 opcode 最少。
- 出处：Patterson and Hennessy, COD (RISC-V)。

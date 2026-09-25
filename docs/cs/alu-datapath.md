---
title: ALU 数据通路
date: 2026-09-08
section: cs
---

# ALU 数据通路

<div class="epigraph">
<p>ALU 不是「会算的云」，而是加法器、逻辑单元与若干 MUX 接在同一组操作数上，由操作码选出哪一路成为结果。</p>
<footer>—— 据 Harris and Harris, Digital Design and Computer Architecture；Patterson and Hennessy, Computer Organization and Design (RISC-V) 整理</footer>
</div>

上一课[移位与桶形移位](/cs/barrel-shifter)钉死了加减组合块。本课不重推 $g,p$，也不从 ISA 操作码表另起。缺口是：CPU 还要与、或、异或、移位、置位。需要一条**数据通路**：操作数进入，功能块并行（或复用），MUX 选出结果，并给出零、负、溢出等旗标。

## 问题

加法器只有和。RISC-V 整数指令还要 `and`/`or`/`xor`/`slt`。缺口因此不是再造一种进位，而是把若干组合功能并在 A、B 上：逻辑按位、加减走加法器、比较可借减法的符号与溢出。`ALUOp` 控制内部 MUX 与加法器的 $c_0$、B 反相。本课只钉这条通路；哪条指令产生哪组控制，是[控制器真值表](/cs/control-truth-table)。

<span class="marginnote">术语翻译：MUX（多路选择器）就是数字电路里的多选一开关——多路输入同时摆在门口，由几根控制线决定放哪一路通过，像列车道岔决定火车走哪条轨。ALU 里「算什么」由 MUX 说了算：所有功能块其实每拍都在并行出结果。</span>

移位可以是桶形移位器（MUX 树），延迟另计。本课承认移位是 ALU 家族，不把 $n$ 位桶形的每一级画出。

### ALU 不是微程序仓库

本课的 ALU 是组合：输入稳定后 $t_{pd}$ 内出结果。没有内部逐步微操作。多周期里 ALU 被多次使用，仍是同一组合块加寄存器在外——那是后课。把 ALU 当成「一台小 CPU」，会与存储程序混层。

<span class="marginnote">Harris 与 Patterson/Hennessy 都把 ALU 画成带 ALUControl 的框。`slt` 用减法后看负异或溢出，把比较并进算术通路。</span>

## 方法

A、B 入。B 经 MUX 选原值或 $\bar B$。加法器算 $A+B_{\mathrm{sel}}+c_0$。逻辑单元算 $A\land B$ 等。结果 MUX 按 `ALUControl` 选。零旗标是结果各位或非。

<span class="marginnote">数字实例：算 $A-B$ 就是令 B 走反相 MUX、$c_0=1$（补码里 $A+\bar B+1=A-B$）。比如 $7-7$：差的全部位为 0，零旗标（各位取或再取非）得 1——`beq` 不用专门比较器，靠的就是这条减法通路吐出的 Z。</span>

```mermaid
flowchart TD
  AB["操作数 A, B"] --> LOG["按位逻辑"]
  AB --> ADD["加减 CLA"]
  ADD --> MUX["结果 MUX"]
  LOG --> MUX
  MUX --> LATER["后课：比较器细化"]
```

## 机制

后课寄存器堆的两个读口接到 A、B；写口接结果（或更后的 MemtoReg）。关键路径包含 ALU 的 $t_{pd}$。`sub` 与 `beq` 都可能用减法：零旗标给相等。本课不接 PC 与立即数——那是单周期数据通路把 ALUSrc 再加一层 MUX。

一条减法通路凭什么同时伺候 `sub`、`slt`、`beq` 三种指令？差、符号、溢出、零旗标各取所需：

```mermaid
flowchart TD
  CALC["共用加法器：B 反相、c0=1，算出 A − B"] --> OUT["差值 + 符号位 + 溢出旗标 + 零旗标"]
  OUT --> SUB["sub：直接取 32 位差写回"]
  OUT --> SLT["slt：看 符号⊕溢出，A 小于 B 则写 1"]
  OUT --> BEQ["beq：零旗标为 1 才跳转"]
```

<span class="marginnote">常见误区：初学者容易想象 ALU 里「住着一段小程序」，接到指令后逐步演算。实际上 ALU 是纯组合逻辑：加法器、逻辑块在同一瞬间并行算完各自结果，MUX 只是当场挑一份；连「逐步」的时间感都没有，从输入稳定到输出可用只隔一个 $t_{pd}$。</span>

## 边界

本课不实现除法、不把浮点 FPU 画进同一框。乘法可视为后课独立单元。比较器下一课可以强调无加减的幅度比较；本课已允许用减法实现 `slt`。

后课默认：整数 ALU 是组合，功能由一小组控制位选择；加减共用 CLA。

## 小结

- ALU = 算术块 + 逻辑块 + 结果选择。
- 控制位来自后课译码，本课只留端口。
- 仍是组合，延迟计入 CPU 关键路径。
- 出处：Harris and Harris；Patterson and Hennessy, COD (RISC-V)。

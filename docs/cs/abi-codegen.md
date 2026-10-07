---
title: ABI 与代码生成
date: 2026-09-08
section: cs
---

# ABI 与代码生成

<div class="epigraph">
<p>应用程序二进制接口规定寄存器谁保存、参数怎么传、栈帧怎么长；代码生成必须按同一份文档发指令，函数才能跨编译单元互调。</p>
<footer>—— 据 System V Application Binary Interface, AMD64 Architecture Processor Supplement；RISC-V ELF psABI 整理</footer>
</div>

上一课[窥孔与窥视窗](/cs/peephole)清理了函数体内的短序列。[调用约定与栈](/cs/calling-convention-stack)在组成课已讲帧与返回地址；[RISC-V 整数指令](/cs/riscv-int-isa)已给出 `jal`/`jalr` 与整数寄存器名。本课不重画五级流水线。缺口是：把那些硬件事实**写成跨编译器的合同**——哪几个是参数寄存器、哪几个 callee-save、栈对齐、谁清参数区——并在序言/尾声里真正发射。

## 问题

着色已选物理寄存器，但「`x10` 是第一个整数参数」不是 ISA 强制，是 ABI。两个目标文件若一个把返回值放 `x10`、一个去栈上找，链接起来仍崩。缺口是代码生成器遵守一份 ABI 文本：System V AMD64 或 RISC-V psABI，而不是发明第四种约定。

序言：保存返回地址与 callee-save、调整 `sp`、对齐。尾声：恢复、返回。可变参数、结构体返回按文档走间接。本课不重讲异常入口。

### ABI 不是 ISA 子集

ISA 说指令编码；ABI 说软件约定。无 ABI 的「合法机器码」仍无法与 libc 互调。组成课的调用约定课给了栈直觉；本课把寄存器号与对齐钉到可汇编。

<span class="marginnote">System V AMD64 ABI 与 RISC-V ELF psABI 是平台文档。本课引用它们的规则层级，不抄整张寄存器表。</span>

<span class="marginnote">术语翻译：callee-save（被调者保存）寄存器就是「谁想用谁先备份」的寄存器——函数 B 若要动它，必须先把旧值存进栈、返回前恢复。与之相对的 caller-save 则由调用方在 call 之前自己备份。分工写死在 ABI 里，两边才不会都以为对方保存了。</span>

<span class="marginnote">常见误区：初学者容易以为「同一 ISA 编译出的目标文件就能互相链接」。实际上 Windows x64 与 Linux 的 System V 都是 x86-64、指令编码相同，但参数放哪些寄存器、栈怎么对齐、异常怎么展开完全不同——ISA 相同、ABI 不同，照样链接崩。</span>

## 方法

按目标选 ABI。参数：前几个进规定寄存器，其余入栈。生成 call：把实参挪到合同位置，`jal`/`call`。被调者按合同保存。栈槽给溢出与局部，对齐 16 字节等条款写进帧布局。

```mermaid
flowchart TD
  IRCALL["IR call"] --> ABI["psABI / SysV 合同"]
  ABI --> ARG["参数寄存器或栈"]
  ABI --> PRO["序言 / 尾声"]
  PRO --> OBJ["可重定位指令"]
```

窥孔不得破坏合同处的 `sp` 与保存集。

## 机制

同一 ISA、不同 OS 可以不同 ABI（Windows x64 对 System V）。交叉编译必须选对三元组。叶子函数可简化序言，仍须对齐与红区规则（若文档有红区）。本课点名，不把红区当所有目标的默认。

与特权级、系统调用号：那是另一份 ABI（syscall ABI），后课操作系统再接；本课是用户态函数调用。

第一张图画的是编译期：IR call 如何按合同落到指令。这张图回答第二个问题：运行时一次调用里，调用者与被调者按合同各干各的活，帧是怎么长出来又收回去的。

```mermaid
flowchart TD
  CALLER["调用者"] --> PUT["实参挪进合同寄存器"]
  PUT --> JAL["call 跳转，压入返回地址"]
  JAL --> PRO["被调者序言：保存 callee-save，对齐 sp"]
  PRO --> BODY["函数体"]
  BODY --> EPI["尾声：按原样恢复寄存器与 sp"]
  EPI --> RET["ret，返回值在合同位置"]
  RET --> BACK["调用者从下一条继续"]
```

<span class="marginnote">数字实例：System V AMD64 要求 call 那一刻栈按 16 字节对齐。调用者进入时 sp 是 16 的倍数，call 压 8 字节返回地址后被调者序言再压一个寄存器（又 8 字节），sp 回到对齐——序言里少存一个寄存器，后面的对齐访存指令就可能当场崩溃。</span>

## 边界

本课不解析 ELF 重定位项、不加载共享库。不实现完整 libc 启动（`_start`、`argc`）。后课默认：每个函数的入口出口遵守目标 ABI。链接把符号引用变成地址。

## 小结

- 代码生成按 ABI 放参数、保存寄存器、对齐栈。
- 先修：调用约定课 + RISC-V 整数 ISA；合同文本是 SysV/psABI。
- ISA 合法不等于能与其他目标文件互调。
- 出处：System V AMD64 ABI；RISC-V ELF psABI；Aho et al., 龙书第 7–8 章。

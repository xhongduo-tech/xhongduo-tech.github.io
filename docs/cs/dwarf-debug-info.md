---
title: 调试信息 DWARF
date: 2026-09-08
section: cs
---

# 调试信息 DWARF

<div class="epigraph">
<p>优化把值放进寄存器并删语句，源级调试靠 DWARF：行号表、位置列表、类型 DIE。信息必须随变换更新，否则断点落空。</p>
<footer>—— 据 DWARF 标准；Tichy 等调试优化代码的讨论；主干编译器通行证整理</footer>
</div>

上一课[unwind](/cs/stack-unwinding)用了 DWARF CFI。缺口是**源级调试**：哪一行、变量现在哪、类型布局。优化编译器要么保守少优化，要么发复杂 location list。本课钉 DIE 与行表，不写 GDB 协议。符号 mangling 下一课。

## 问题

DCE 删赋值，变量「当前无位置」。内联：同一源行多个实例，需 `DW_TAG_inlined_subroutine`。缺口是**元数据与 IR 同步**，不是 FDE 查找。

<span class="marginnote">DIE（Debugging Information Entry，调试信息条目）是 DWARF 描述世界的最小单位：一个变量、一个类型、一个函数各是一条，彼此用父子关系连成树——整个程序的类型世界就是一棵 DIE 树。可以类比 JSON：每条是一个带 tag 的对象，字段描述它的属性。</span>

行号：`is_stmt` 标记语句边界，方便断点。调度打乱序，行表可非单调。

<span class="marginnote">为什么优化必须同步更新行表：下断点时，调试器要在行表里找那一行对应的第一条指令；优化若把语句删了或挪了而不更新表，断点要么落不上、要么停在别的语句上——你以为是调试器坏了，其实是元数据过期。这就是正文「信息必须随变换更新」的具体含义。</span>

### DWARF 不是「关掉优化」

`-O2 -g` 合法：信息尽量描述优化后的机器。质量参差。不要假设 `x` 总在栈槽。

<span class="marginnote">DWARF 5。GCC/LLVM 的 DI 元数据。本课不把 CodeView 写完，只点名 MSVC 另一格式。</span>

## 方法

前端发 `dbg.value`/`dbg.declare`。优化传递：复制、删除时更新。后端：寄存器分配后写成 location list（按 PC 区间）。类型：从语言类型降到 DIE。

```mermaid
flowchart TD
  SRC["源位置"] --> DI["IR 调试元数据"]
  DI --> OPT["随优化更新"]
  OPT --> DW["DWARF 节"]
```

与 SROA：字段拆开要多个 location。与合并：两变量同寄存器，调试器按 PC 区分。

## 机制

体积：`-g` 显著增大。分离：`.dwo`/dsym。strip 后只留 unwind 或全无。不要把密钥写进调试字符串。

```mermaid
flowchart TD
  Q["调试器：现在变量 x 在哪？"] --> PC["先看当前停在哪条指令（PC）"]
  PC --> R1["PC 在 0x100-0x1FF：查表得 x 在寄存器 RDI"]
  PC --> R2["PC 在 0x200-0x2FF：x 被溢出，躺在栈槽里"]
  PC --> R3["PC 在 0x300 之后：x 已被优化删除，无位置"]
  R1 --> A["同一变量，位置随指令区间切换"]
  R2 --> A
  R3 --> A
```

<span class="marginnote">strip 直译「剥离」：把调试节从二进制里剪下来单独存放（.dwo 或 dSYM 一类），发行物变小，出问题时再把两半拼回去调。常见误区：strip 不改变程序行为——剪掉的只是「给人和调试器看的注释」，机器执行根本不读它们。</span>

ASan 改变布局，调试信息要跟。

## 边界

本课不写表达式求值器全部 DWARF ops。后课默认：优化代码用位置列表。下一课符号与 mangling：链接与调试都看见的名字。

也不把 DWARF 当核心转储格式的全部（ELF note 另有）。

## 小结

- DWARF：行号、类型、变量位置，随优化更新。
- 内联与 SSA 析构是主要难点。
- CFI 是 DWARF 的一部分，服务展开。
- 出处：DWARF 标准；对照 Appel/龙书调试支持。

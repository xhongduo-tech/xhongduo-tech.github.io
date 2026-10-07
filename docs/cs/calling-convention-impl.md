---
title: 调用约定的实现
date: 2026-09-08
section: cs
---

# 调用约定的实现

<div class="epigraph">
<p>约定规定参数在哪些寄存器或栈槽、谁保存、返回值怎么走。后端 lowering 必须与 ABI 文档一致，否则不能与已编译的库链接。</p>
<footer>—— 据 System V ABI；ARM AAPCS；主干[调用约定与栈](/cs/calling-convention-stack)；Appel 整理</footer>
</div>

上一课[目标描述](/cs/target-description)有寄存器类。组成课已讲约定直觉。缺口是**实现**：ISel 把 `call` 降成拷贝到参数寄存器、栈对齐、red zone、可变参。本课钉 lowering，不重画栈帧全图——下一课布局。

## 问题

IR 的 `call @f(i32, double)` 要按 ABI 分类（整数/浮点/聚合）。聚合：寄存器对或内存。缺口是**分类算法 + 插入拷贝**，不是 lex。

可变参：固定参数后的区域；调用方与 `va_list` 一致。调用者保存：跨 call 的活寄存器要存——分配器已见预着色。

### ABI 不是「编译器随便选」

与操作系统、语言运行时绑定。自创约定只能闭世界 LTO 静态全程序。不要改 `printf` 的约定。

<span class="marginnote">SysV AMD64 ABI。AAPCS64。Itanium C++ ABI 的 this 指针。本课 C 为主，C++ 额外 this、返回槽。</span>

## 方法

实现 `LowerCall`/`LowerFormalArguments`：把参数映射到物理寄存器或 `FrameIndex`。返回：寄存器或 sret 隐藏指针。对齐：栈 16 字节等。

<span class="marginnote">术语翻译：sret 就是用「隐藏的第一个指针参数」的手段来做「返回大结构体」的事——调用方先在栈上挖好一块空间，把地址当参数递进去，函数往里写；调用结束后那块栈空间里就是返回值。</span>

```mermaid
flowchart TD
  IR["IR 参数"] --> CLS["ABI 分类"]
  CLS --> REG["寄存器"]
  CLS --> STK["栈槽"]
  REG --> CALL["call 序列"]
  STK --> CALL
```

与内联：内联后约定消失。与尾调用：须约定兼容才能 jmp。

<span class="marginnote">数字实例：SysV AMD64 下整数参数依次走 rdi、rsi、rdx、rcx、r8、r9 共 6 个寄存器。调用 f(1,2,3,4,5,6,7) 时前 6 个进寄存器，第 7 个参数 7 被压到栈上——多传一个参数，性能特征就可能变。</span>

## 机制

浮点与 SIMD 寄存器类独立。整数 6 个参数寄存器满则上栈。不要把 `float` 误分类到 GPR 除非 ABI 如此（某些软浮点）。

```mermaid
flowchart TD
  A["一个待传参数"] --> C{"ABI 分类"}
  C -->|"整数 / 指针"| G{"整数寄存器还有空位？"}
  C -->|"浮点 / SIMD"| X{"浮点寄存器还有空位？"}
  G -->|"是"| R["放入下一个 GPR，如 rdi"]
  G -->|"否"| S["压到调用方栈上按对齐排布"]
  X -->|"是"| XR["放入 xmm 寄存器"]
  X -->|"否"| S
  R --> CALL["call 指令"]
  XR --> CALL
  S --> CALL
```

异常：调用是 invoke 时还要 landing pad，展开表后课。

<span class="marginnote">常见误区：初学者容易以为「函数传参就是压栈」，实际上现代 x86-64 约定前几个参数走寄存器，栈只是寄存器用完后的溢出区；老的 32 位 cdecl 才是全部压栈、由调用方清理。</span>

## 边界

本课不写完整 DWARF。后课默认：call lowering 实现 ABI。下一课栈帧布局与帧指针。

也不把 ABI 当 REST API。

## 小结

- 调用约定 lowering：分类 → 寄存器/栈 → 拷贝。
- 必须与平台 ABI 一致才能链接 libc。
- 可变参、聚合、sret 是细节坑。
- 出处：System V ABI；AAPCS；对照龙书/Appel 调用序列。

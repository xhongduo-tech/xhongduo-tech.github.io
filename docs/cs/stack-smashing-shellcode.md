---
title: 栈溢出到 shellcode
date: 2026-09-08
section: cs
---

# 栈溢出到 shellcode

<div class="epigraph">
<p>返回地址也是栈上的数据。写过边界的字节若覆盖了它，CPU 就会跳到源码从未写过的地方。这是控制流完整性的开口，不是又一种密码分析。</p>
<footer>—— Aleph One, Smashing the Stack for Fun and Profit, Phrack 49, 1996；对照 Cowan 等对 canary 的后续</footer>
</div>

上一课[Needham–Schroeder](/cs/needham-schroeder-lowe)假设进程按规范走。缺口是**实现把控制数据与普通对象放在同一可写栈上**。本课只说明机制与为何出现 NX/canary，不提供可运行载荷或注入步骤。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

C 的数组与 `gets` 一类接口不携带长度。局部缓冲之后是保存的帧指针与返回地址。越界写一旦到达返回地址，函数返回就离开编译器安排的图。<span class="marginnote">直觉类比：返回地址和数据住同一间宿舍，就像快递单贴在货箱同一面墙上——货装得太满溢到单子上，收件人一栏被改，货就被送到别人家。</span>历史叙述里这曾被用来把处理器引向注入的字节——今日有 NX 等缓解，本课不教如何做。

### 课程边界

禁止 shellcode、禁止复现步骤。需要的只是：空间安全失败 ⇒ 控制流不再等于源码。防御在后课层层加。

<span class="marginnote">Aleph One 是历史文献，本课当反例引用，不当实验指导。Anderson 把这类失败归为实现与语言。</span>

## 方法

画栈帧槽位：局部变量、控制数据。指出语言与编译器可插入界检查或把返回地址移走（后课影子栈）。对照：协议逻辑再对，这一个写仍能接管进程。<span class="marginnote">常见误区：以为协议逻辑审对了就安全。内存不安全语言里，一个不带长度检查的读入就能把控制流改走——逻辑层与实现层是两张考卷，只答对一张不算及格。</span>

```mermaid
flowchart TD
  BUF["可写缓冲"] --> OOB["越界写"]
  OOB --> RET["返回地址被改"]
  RET --> CF["控制流离开源码"]
  NX["不可执行栈"] -.->|"缓解, 非根除"| CF
```

## 机制

进程作为 TCB 的一部分：用户输入变成了跳转目标。下一课 canary：用秘密值检测「帧被踏过」，不修复语言。<span class="marginnote">术语翻译：TCB（可信计算基）就是「整套安全赖以成立的那一小圈组件」——把进程也算进 TCB，用户输入就等于能改写跳转目标，安全边界随之从协议层塌到实现层。</span>

```mermaid
flowchart TD
  RISK["越界写破坏控制流"] --> D1["语言层：界检查与安全类型"]
  RISK --> D2["编译器层：canary 与影子栈"]
  RISK --> D3["系统层：NX 与 ASLR"]
  D1 --> C1["堵源头：这个写发不出去"]
  D2 --> C2["堵后果：改动退出前被查出"]
  D3 --> C3["堵利用：字节不可执行、地址不定"]
```

## 边界

本课不给任何 payload。canary 下一课是第一道常见编译器缓解。

## 小结

- 协议正确仍可能被内存破坏接管。
- 返回地址与数据同栈，越界写破坏控制流。
- 不提供注入方法；NX 等是缓解。
- 下一课栈 canary。
- 出处：Aleph One, Phrack 49（历史）；Anderson, *Security Engineering*；对照 Cowan et al.。

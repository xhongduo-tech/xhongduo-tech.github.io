---
title: 操作语义：进展与保持
date: 2026-09-08
section: cs
---

# 操作语义：进展与保持

<div class="epigraph">
<p>小步：一次改写一项。大步：直接给值。上下文指出下一归约点。类型定理建立在这份关系上，而不是建立在编译器实现上。</p>
<footer>—— 据 Plotkin, A Structural Approach to Operational Semantics；Pierce TAPL；Wright and Felleisen, 1994 整理</footer>
</div>

上一课[类型健全性](/cs/type-soundness)已经调用了「一步」与「值」。缺口是把**操作语义**写清楚：小步 SOS、求值上下文、调用约定（值调用 vs 名调用）如何改变进展陈述。不重写 λ 的 β 定义，只把它收成关系 $t \to t'$。也不写指称——下一课。

## 问题

「程序做什么」若只靠编译器，无法证保持。[λ 演算](/cs/lambda-calculus)有 β；STLC 要规定应用序还是正规序。小步：$\mathrm{E}[(\lambda x.t)\,v] \to \mathrm{E}[t[v/x]]$。大步：$\langle t,\sigma\rangle \Downarrow v$。缺口是**关系与上下文**，不是 lex。

状态：引用语言要堆 $\sigma$。进展变成：良型配置或能一步，或已是值。并发：一步是线程交错，保持更难。

### 小步不是「解释器源码」

SOS 是数学关系。定义式解释器（Reynolds）是函数，可作可执行规范，但仍要与 SOS 对齐。本课以 SOS 为主。

<span class="marginnote">Plotkin SOS。Felleisen 的求值上下文。Pierce TAPL 贯穿小步。Appel 的编译器正确性用语义对齐。本课不写 CompCert 的 Clight，那在课程末尾。</span>

<span class="marginnote">术语翻译：CBV（值调用）就是先把参数算成值再代入函数体——C、Java、Python 都是这个脾气；CBN（名调用）是把参数表达式原封不动代入，用到才算。同一个程序在两种策略下，求值路径甚至会不会停机都可能不同，所以「选策略」就是选规则。</span>

## 方法

列出语法、值、一步规则。选策略：CBN/CBV 改变哪条规则。证明：同一套进展保持。实现对照：字节码解释器应仿真小步（运行时单元）。

```mermaid
flowchart TD
  T["项"] --> CTX["求值上下文"]
  CTX --> STEP["小步 →"]
  STEP --> V["值或继续"]
```

<span class="marginnote">直觉类比：求值上下文 $\mathrm{E}[\,\square\,]$ 像一张「带洞的模板」，洞里填什么（哪个子项），哪里就是下一步的归约点——相当于把「先找位置、再改写」拆成填空题，规则就不用为每种嵌套写一遍。</span>

与效应：一步可带事件标签，得带标签的转换系统，供效应类型解释。

## 机制

合流：无类型 λ 有；加状态后一般没有「同一值」的合流，只有类型保证的不卡住。不要用 Church–Rosser 当命令式语言的健全性。

大步对不终止无推导，难以谈进展；小步更适合卡住分析。大步适合证明编译器「若源停机则目标得相同值」。

小步与大步各自适合回答什么问题：

```mermaid
flowchart TD
  T["项 t"] -->|"小步: 每次改写一项"| S1["t1 → t2 → t3 → …"]
  S1 -->|"看得见每一步中间态"| G1["可分析卡住: 证进展"]
  T -->|"大步: 直接给值"| B1["t ⇓ v"]
  B1 -->|"只有起点与终点"| G2["适合证停机则等值"]
```

<span class="marginnote">数字实例：把死循环 $\Omega$ 当参数传给 $\lambda x.\,3$——CBV 先算参数，永不停止；CBN 不算参数，一步得到 $3$。一个「参数根本没被用到」的例子，就能看出策略选择改变的是可停机性，不只是快慢。</span>

## 边界

本课不写抽象机器（SECD、CEK）的全部转移。后课默认：类型定理相对 SOS。下一课指称语义：用数学对象解释项，对照操作。

也不把 CPU 流水线当操作语义。

## 小结

- 小步/大步是项（加状态）上的关系。
- 进展保持建立在小步上更自然。
- 策略（CBV/CBN）是规则选择。
- 出处：Plotkin；Felleisen；Pierce TAPL；Wright–Felleisen。

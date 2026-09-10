---
title: 语言内存模型与 data race
date: 2026-09-08
section: cs
---

# 语言内存模型与 data race

<div class="epigraph">
<p>没有 data race 时，程序等价于某种交错的顺序一致执行。有 race 时，C/C++ 是未定义，Java 仍限制可见性。编译器重排必须遵守 happens-before。</p>
<footer>—— 据 Adve and Hill；Boehm and Adve, Foundations of the C++ Concurrency Memory Model；Java Memory Model（JSR-133）整理</footer>
</div>

上一课[反射](/cs/reflection-metadata) 可在运行时改字段，并发更乱。缺口是**语言级内存模型**：什么重排合法，data race 是不是 UB。接[UB 与优化](/cs/undefined-behavior-opt) 与组成课一致性，但不重讲总线。WebAssembly 下一课当目标。

## 问题

优化想把 load 提出循环；另一线程在写。无模型则要么禁止优化要么允许乱看。JMM/C++11：原子与锁建立 happens-before。缺口是**编译器合同**，不是硬件 MESI 细节。

Data race：冲突访问、至少一个写、无排序。C++：UB。Java：不崩型，但值可撕（除 long/double 特殊）——仍几乎不可写。

### 内存模型不是 GC 写屏障

GC 屏障维持三色；内存模型屏障维持可见性。JIT 可能同一条 store 两件事，概念分开。

<span class="marginnote">Boehm–Adve 2008。JSR-133。Adve–Gharachorloo 综述硬件。本课语言层。不写利用级竞态教程。</span>

## 方法

IR：`atomic` 带序（relaxed/acquire/release/seqcst）。优化：不跨原子乱序普通访问。逃逸到多线程的对象：锁消除要证明无逃逸。

```mermaid
flowchart TD
  RACE["data race"] --> C["C++：UB"]
  RACE --> J["Java：弱保证"]
  SYNC["锁 / 原子"] --> HB["happens-before"]
  HB --> SC["DRF ⇒ SC"]
```

与 fast-math 无关；与并发 GC 的对象发布有关：必须安全发布。

## 机制

编译器+硬件共同实现模型。`volatile` 在 Java 与 C 意义不同——点名坑。不要用普通 load 当旗。

DRF-SC：无 race 则程序员可当顺序一致想。有 race 的 C++ 程序优化可删除看起来「有用」的代码。

## 边界

本课不写全部内存序表。后课默认：优化受内存模型约束。下一课 WebAssembly 作目标：另一 ABI+线性内存。

也不把模型当经济学博弈。

## 小结

- DRF ⇒ 顺序一致交错；C++ race 是 UB。
- 原子序限制重排；与 GC 屏障不同。
- 锁消除、LICM 必须看逃逸与同步。
- 出处：Boehm and Adve；JSR-133；Adve and Hill。

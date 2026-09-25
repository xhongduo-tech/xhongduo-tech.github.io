---
title: 内联缓存
date: 2026-09-08
section: cs
---

# 内联缓存

<div class="epigraph">
<p>调用点记住上次接收者的类与方法地址：同类则直跳，不同类则回慢速查找。单态缓存命中时，动态分派接近静态调用。</p>
<footer>—— 据 Deutsch and Schiffman；Hölzle, Chambers and Ungar, Optimizing Dynamically-Typed Object-Oriented Languages with PICs 整理</footer>
</div>

上一课[追踪 JIT](/cs/tracing-jit) 沿路径投机类型。缺口是**调用点投机**：内联缓存（IC）。Smalltalk/Self/JS：方法查找贵。单态 IC：一个类；PIC：几个类的跳表。本课钉 IC，去优化下一课收失败。

## 问题

vtable 对静态类层次快；动态语言形状（hidden class）运行时变。IC：调用点状态机。缺口是**把查找结果焊在调用点**，不是整段 trace。

<span class="marginnote">直觉类比：动态方法查找像每次打电话都要翻通讯录；内联缓存像把「上次拨的号码」贴在话机上——对面还是同一个人（同一个类）就直接拨，换了人才重新翻本子。大多数调用点反复调的都是同一个类，所以这张便签命中率很高。</span>

与[类型类字典](/cs/typeclass-dictionary)：都是间接调用，IC 可改成直跳并内联。

### IC 不是 CPU 的 I-cache

Inline cache 是语言实现；instruction cache 是硬件。不要混名。

<span class="marginnote">Deutsch–Schiffman。Hölzle–Chambers–Ungar PIC（OOPSLA）。V8/IC 状态机是工程后裔。</span>

## 方法

未初始化：查表，填缓存。单态：比较类字，命中则调用缓存地址。失配：变多态或 MEGAMORPHIC（回哈希）。JIT 可对单态直接内联，失配去优化。

```mermaid
flowchart TD
  CALL["调用点"] --> MONO["单态 IC"]
  MONO --> PIC["多态 IC"]
  PIC --> MEGA["哈希查找"]
```

与逃逸/形状：对象布局稳定 IC 才稳。隐藏类转移会刷缓存。

## 机制

线程：IC 更新要原子或每线程。不要无限 PIC 变大。统计：调用点的多态度决定是否内联。

与方法 JIT：编译时把 IC 变成 cmp+je 序列。

单态 IC 命中与失配时各自发生什么：

```mermaid
flowchart TD
  C["调用点：obj.f()"] --> CMP["比较 obj 的隐藏类 == 缓存里记的类？"]
  CMP -- "相同" --> J["直跳上次查到的方法地址"]
  CMP -- "不同" --> LOOK["回慢路径：重新哈希查找"]
  LOOK --> UP["缓存记下新出现的类"]
  UP --> M["调用点升级为多态 PIC"]
```

<span class="marginnote">术语翻译：「多态」指同一个调用点见过 2 到 4 个不同的接收者类，缓存扩成几个槽的跳表；见过的类多到跳表不划算（如超过 4 个）就叫「超多态」，此时回退到哈希查找。多态度是运行时统计出来的，不是声明出来的。</span>

## 边界

本课不写去优化栈重写。后课默认：动态调用用 IC。下一课去优化：投机失败如何回解释器状态。

也不把 IC 当 HTTP 缓存。

<span class="marginnote">常见误区：别把 inline cache 和 CPU 的指令缓存（I-cache）当成一回事。前者是语言运行时焊进调用点的一小段「比较类字 + 直跳」机器码；后者是硬件取指部件缓存指令字节的部件。名字像，层次完全不同。</span>

## 小结

- IC：调用点缓存接收者类与目标。
- 单态可内联；超多态回慢路径。
- 与 vtable 静态分派 complementary。
- 出处：Deutsch and Schiffman；Hölzle, Chambers and Ungar。

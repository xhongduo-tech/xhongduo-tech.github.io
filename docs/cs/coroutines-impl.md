---
title: 协程与生成器实现
date: 2026-09-08
section: cs
---

# 协程与生成器实现

<div class="epigraph">
<p>生成器是半协程：`yield` 保存 PC 与局部，下次从该点继续。实现或把状态机编到堆帧，或切换独立栈。</p>
<footer>—— 据 Moura and Ierusalimschy, Revisiting Coroutines；C# / Python 生成器实现；对照 CPS 整理</footer>
</div>

上一课[CPS](/cs/cps-continuations) 把控制变成函数。生成器不必全 CPS：编译器把函数切成状态机。缺口是**暂停点的帧**。蹦床下一课。不写异步 IO 框架百科。

## 问题

`yield x` 返回值给调用者，但局部还要活。方案 A：变换成类，字段=局部，`switch(state)`。方案 B：独立栈，切换 SP。缺口是**与 GC 根、展开表的关系**，不是 call/cc 全权。

async/await 是生成器+将来值的语法，实现同类。

### 生成器不是操作系统线程

用户态、协作式。不要和抢占线程混。切换代价应远小于 `pthread`。

<span class="marginnote">Moura–Ierusalimschy。Python 的 frame 对象。LLVM 协程表示（CoroSplit）。本课两种实现对照。</span>

<span class="marginnote">PC（程序计数器）翻译过来就是「下一条执行哪条指令」的游标。yield 的本质是把游标连同局部变量打包存进协程帧：函数并没有真正结束，下次 resume 把游标拨回暂停处，从半句话中间接着讲。</span>

## 方法

前端标暂停点。CoroSplit：把 SSA 跨 yield 的值溢到协程帧。调用：resume 恢复 PC。完成：销毁帧。

```mermaid
flowchart TD
  YLD["yield"] --> SAVE["保存 PC 与活值"]
  SAVE --> RET["返回调用者"]
  RET --> RES["resume 恢复"]
```

与闭包：协程帧是环境的亲戚。与 DWARF：展开要认协程帧布局。

## 机制

栈满协程：每个协程一页栈，内存多。状态机：内存少，不能任意深度互递归 yield 除非堆分配。不要在持锁时 yield 而不文档。

```mermaid
flowchart TD
  CH["选协程实现"] --> SM["无栈状态机"]
  CH --> SF["有栈协程"]
  SM --> SMI["只存协程帧 极省内存 · 不能深层互 yield"]
  SF --> SFI["独立栈可任意嵌套 · 每个协程约一页内存"]
```

<span class="marginnote">直觉类比：栈满协程像给每个任务一间独立办公室，随时走人随时回来，东西都留在桌上；状态机像让员工把进度填进一张表格再走，回来照表继续——便宜得多，但表上没登记的东西就丢了，所以编译器必须精确算出哪些活值要溢进协程帧。</span>

异常：在 yield 对面抛，状态机要能传播。

<span class="marginnote">常见误区：以为协程切换像函数调用一样免费且无副作用。实际上 yield 点是「世界可能变了」的时刻——你持有的锁、正在遍历的容器都可能被对面协程动过；持锁 yield 引发的死锁没有时钟中断来解围，全靠自己约定避免。</span>

## 边界

本课不写蹦床。后课默认：生成器=可暂停帧。下一课蹦床与尾递归：另一套无栈控制。

也不把协程当化学课。

## 小结

- 生成器：状态机或独立栈保存暂停点。
- 与 CPS 同类控制，实现更特化。
- GC、展开、锁是正确性边界。
- 出处：Moura and Ierusalimschy；语言实现（Python/C#）；LLVM CoroSplit。

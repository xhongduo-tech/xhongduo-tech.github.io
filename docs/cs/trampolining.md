---
title: 蹦床与尾递归
date: 2026-09-08
section: cs
---

# 蹦床与尾递归

<div class="epigraph">
<p>宿主语言不保证尾调用时，把「下一步计算」做成 thunk 返回给循环：蹦床反复调用，栈深度保持常数。</p>
<footer>—— 据 Steele 尾调用讨论；Baker, Cheney on the M.T.A.；对照[尾调用优化](/cs/tail-call-opt) 与 CPS 整理</footer>
</div>

上一课[协程](/cs/coroutines-impl) 用帧暂停。缺口是在**无 TCO 的宿主**（Java、C# 历史、JS 引擎不完全保证）实现 Scheme 式互递归：蹦床。虚表下一课。不重写 CPS 变换全文。

## 问题

CPS 后全是尾调用，若 `call` 真的建帧则爆栈。蹦床：函数返回 `Bounce { thunk }` 或 `Done { value }`，驱动循环 `while (bounce) thunk()`。缺口是**驱动器**，不是 yield 状态机。

<span class="marginnote">术语翻译：thunk 就是「打包好但还没执行的一小步计算」——函数不直接调用下一个函数，而是把「帮我调 f(参数)」装进信封返回，由驱动循环拆信执行，栈于是不深。</span>

Cheney on the MTA：用垃圾回收当栈——极端蹦床，点名。

### 蹦床不是协程的 yield

yield 把控制交回调用者并保存局部；蹦床通常不把中间结果交给用户，只给驱动器。用户看来仍是一次 `eval`。

<span class="marginnote">Baker 的 MTA。Steele。SML 编译到 C 的历史技巧。本课工程补丁，优先仍是真 TCO。</span>

## 方法

约定：尾位置不调用而返回闭包。驱动：循环直到 `Done`。与闭包转换：thunk 即闭包。性能：间接多，可批处理若干步。

```mermaid
flowchart TD
  F["尾调用"] --> TH["返回 thunk"]
  TH --> LOOP["蹦床循环"]
  LOOP --> D["Done 值"]
```

与 JIT：宿主若能识别驱动循环，仍难内联互递归。真 TCO 更好。

<span class="marginnote">数字实例：JVM 默认栈大约撑 1 万层递归就溢出；同样的逻辑改成蹦床，跑 100 万步栈深也不变——代价是每步多一次闭包分配加一次间接调用，通常慢数倍。</span>

## 机制

调试栈无意义。异常：要穿过驱动器。不要在 thunk 里无意非尾调用。

与生成器：可把蹦床当无用户可见暂停的协程驱动。

<span class="marginnote">常见误区：以为写成尾递归编译器就一定会优化。JVM 与 CPython 默认都不做 TCO；不确认宿主行为就依赖它，小数据测试没事，上生产遇到长输入直接爆栈。</span>

```mermaid
flowchart TD
  ORD["普通递归调 100 万层"] --> SO["调用栈堆 100 万帧<br/>StackOverflow"]
  TRAMP["蹦床版调 100 万步"] --> STEP["每步：函数返回下一个 thunk"]
  STEP --> EXE["驱动器调用 thunk<br/>栈上始终只有循环这一帧"]
  EXE --> FIN["收到 Done 返回结果<br/>栈深仍是常数"]
  SO -.->|"对照"| STEP
```

## 边界

本课不写虚表。后课默认：无 TCO 时用蹦床保栈。下一课虚表与动态分派。

也不把蹦床当体操。

## 小结

- 蹦床：返回 thunk + 循环，模拟 TCO。
- 性能差于真尾调用；用于宿主限制。
- thunk 是闭包。
- 出处：Steele；Baker MTA；对照 Appel CPS。

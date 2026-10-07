---
title: 去优化
date: 2026-09-08
section: cs
---

# 去优化

<div class="epigraph">
<p>投机编译假设类型、别名、类层次；失败时必须把 native 帧翻译回解释器（或低层 JIT）状态，而不能假装没发生过。</p>
<footer>—— 据 Hölzle, Chambers and Ungar, Debugging Optimized Code with Dynamic Deoptimization, 1992；HotSpot 去优化说明整理</footer>
</div>

上一课[内联缓存](/cs/inline-caching) 在失配时要退出投机码。缺口是 **deopt**：映射寄存器与 spill 到字节码栈、PC、锁状态。本课钉为什么必须、元数据，类型反馈下一课是投机的来源。

## 问题

内联+类型假设后，CFG 不再对应源方法。失败：类加载打破 CHA、IC 失配、守卫失败。必须恢复：解释器能继续。缺口是**栈重写与副作用已发生**（不能回滚 IO），只恢复虚拟机状态。

<span class="marginnote">术语翻译：去优化就是「把跑在快道上、但假设已被打破的代码，安全地搬回慢道」。搬家不靠重算，靠编译器预先留下的映射表，把寄存器与栈槽翻译回解释器认识的字节码栈、PC 与锁状态，从失败点无缝继续。</span>

OSR（on-stack replacement）是反方向：解释器帧换成 compiled 帧。

### 去优化不是「反编译」

不是从机器码还原源。是编译器预先留下的映射表：`pc → 字节码索引 + 位置列表`。与 DWARF 同类，服务 VM 而非 GDB。

<span class="marginnote">Hölzle et al. 1992 PLDI。HotSpot Deoptimization。本课不写全部帧描述符格式。</span>

## 方法

编译时发 deopt 元数据。运行时：save 寄存器、walk 内联帧链、物化对象（逃逸分析撤销）、跳入解释器。延迟去优化：标记代码为非入口，现有帧在安全点再转。

```mermaid
flowchart TD
  FAIL["守卫 / IC 失败"] --> MAP["帧描述符"]
  MAP --> INT["重建解释器状态"]
  INT --> CONT["继续执行"]
```

与[逃逸分析](/cs/escape-analysis)：栈化对象在 deopt 时要重新堆分配（物化）。

## 机制

安全点：GC 与 deopt 只能在约定点停，否则映射不存在。不要在任意机器指令去优化。锁：膨胀、消除的锁要恢复。

把投机代码的整个生命周期摆出来，deopt 只是其中一环，之后还要回到编译策略上：

```mermaid
flowchart TD
  INT["解释执行：收集类型 profile"] --> HOT["热点：JIT 编译并内联"]
  HOT --> RUN["运行投机代码：守卫在场"]
  RUN -->|"假设一直成立"| OK["持续快速执行"]
  RUN -->|"守卫失败"| DEOPT["去优化：按映射表重建解释器状态"]
  DEOPT --> INT2["解释器继续，profile 更新"]
  INT2 --> RECOMP["再编译：收敛假设或放弃优化"]
```

<span class="marginnote">直觉类比：投机编译像外卖员猜「你家大门常开」径直送到门口——平时省去按门铃（类型检查），一旦猜错，守卫（门铃）响了只好退回前台重走流程（去优化）。注意：已经塞进门缝的东西（已发生的写操作、IO）收不回来，重走的是「接下来的路」，不是「来时的路」。</span>

<span class="marginnote">常见误区：初学者容易把 deopt 当成「反编译」或「回滚」。它既不从机器码还原源码（只是查编译期留下的映射表），也不能撤销已执行的副作用——只恢复虚拟机状态让程序继续。另外，deopt 本该稀少；若某段代码频繁去优化，说明它太投机，正确的反应是降级假设而不是忍着。</span>

性能：deopt 应稀。频繁则代码太投机，要降级。

## 边界

本课不写分层编译阈值。后课默认：投机必须可逆到 VM 状态。下一课类型反馈：给投机提供数据。

也不把 deopt 当异常 unwind 的别名（可共用展开，语义不同）。

## 小结

- 去优化：投机 native → 解释器/低层状态。
- 靠编译期映射，不是反编译。
- 物化、锁、安全点是实现难点。
- 出处：Hölzle, Chambers and Ungar, 1992；HotSpot。

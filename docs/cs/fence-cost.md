---
title: fence 的代价
date: 2026-09-08
section: cs
---

# fence 的代价

<div class="epigraph">
<p>全序栅栏要排空写缓冲、禁止越过它的 load 提前，等于主动关掉乱序与 MLP；正确的程序少用，用错的程序把它当注释重复插。</p>
<footer>—— 据 Hennessy and Patterson, CA:AQA；Intel SDM 对 mfence/sfence/lfence 的描述 整理</footer>
</div>

[上一课](/cs/transactional-memory) 用 abort 代替部分锁。[acquire/release](/cs/acquire-release) 已经是单向的。程序员和编译器仍会放出 `fence` / `mfence` / `dmb`。本课不重画写集。缺口是**把栅栏换成流水线里的停顿：SQ drain、load 禁止提前、甚至指令串行化**，并收束本单元。

## 问题

弱序核上，没有栅栏则 [LSQ](/cs/lsq-disambiguation) 可以让年轻 load 越过年长 store（地址不同时）。设备驱动、无锁、与 GPU 共享的页需要更强顺序。缺口不是新一致性态，而是**fence 在微结构上等于什么，以及它如何毁掉 [MLP](/cs/mlp-memory-parallelism) 与 store 缓冲。**

<span class="marginnote">术语翻译：MLP（内存级并行）就是「同时让多个 cache miss 在路上飞」的能力。栅栏一插，核必须把年长的访存全部等完才放行年轻的，等于把并排的高速车道并成了单车道——这就是本文说 fence「毁掉 MLP」的直觉。</span>

<span class="marginnote">x86：`sfence` 主要排 store；`lfence` 阻止后续指令在先前 load 前执行；`mfence` 更强。RISC-V `fence rw,rw` 的前/后集合可配，不必每次全排空。</span>

## 方法

实现：fence 在 ROB 里占一项，直到指定集合里更年长的访存完成一致性动作（store 成为 M 可见、load 数据已返回）。更年轻的同类型访存不得越过。可选：冲刷 [写合并](/cs/write-combining)。串行化 fence（如 `cpuid` 一类）还要等整个 ROB 退休。

```mermaid
flowchart TD
  F["fence 进入 ROB"] --> DRAIN["排空指定 SQ/LQ"]
  DRAIN --> WCB["可选冲刷 WCB"]
  DRAIN --> GO["后续访存才发"]
```

## 机制

代价以「本可以飞行的 miss 被拦住」计，不是 ALU 的一拍。高争用锁若每层都 seq-cst，会把核变成近似顺序机。[CPI 与阿姆达尔](/cs/cpi-amdahl) 在这里极狠：一小段同步代码的 $f$ 被放大。正确用法是：数据竞争自由的程序把 fence 藏进锁/原子原语，而不是散落在每个共享读。

```mermaid
flowchart TD
  A["无栅栏: load x 未命中"] --> B["load y 也未命中, 与 x 同时在飞"]
  B --> C["两个 miss 重叠, 耗时约一次访存"]
  D["有栅栏: fence"] --> E["load x 未命中, 后续被拦"]
  E --> F["栅栏排空期间流水线闲置"]
  F --> G["load y 才发, 耗时约两次访存"]
```

<span class="marginnote">数字实例：设一次 miss 约 100 ns。两个地址无关的 load 若能并行在飞，总耗时接近 100 ns；中间插一道栅栏后只能串行，变成约 200 ns。若这道栅栏又嵌在高争用锁里、每次加锁都执行，几纳秒的「语义代价」就放大成可观测的吞吐损失。</span>

本单元收束：替换、预取、VIPT、一致性五态与目录、伪共享、原子、模型、事务、fence。下一单元从并行分类（Flynn）转向向量、SIMT 与互连，不再加宽单核 cache 协议。

## 边界

本课不把所有 ISA fence 的真值表抄完。也不把 Spectre 的 `lfence` 串行化当性能优化推荐——那是后课安全与推测，动机相反。Flynn 分类下一课从「单核 ILP」抬到「多指令多数据怎么摆」。

后课默认：fence 是主动放弃 ILP/MLP 换顺序。并行形态先按 Flynn 分箱，再谈 VLIW 与 GPU。

<span class="marginnote">常见误区：初学者容易把 fence 当注释——「多插几条总没错，保险」。实际上每一条 fence 都是真金白银的停顿：排空写缓冲、拦住所有更年轻的访存。大多数场景 acquire/release 甚至数据竞争自由代码已经足够，多余的 fence 只是把多核亲手退化回顺序机。</span>

## 小结

- fence 排空指定访存集合，代价是 MLP 与写缓冲。
- 能用 acquire/release 就不要用全序。
- 下一课序从 Flynn 分类开始谈并行形态。
- 出处：Hennessy and Patterson, *CA:AQA*；Intel SDM；RISC-V 内存模型。

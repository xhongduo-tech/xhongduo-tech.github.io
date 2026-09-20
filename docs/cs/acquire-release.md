---
title: acquire / release
date: 2026-09-08
section: cs
---

# acquire / release

<div class="epigraph">
<p>释放之前的写必须在锁释放这一刀之前对全局可见；获取之后的读必须在锁获取这一刀之后才开始。不必对全世界做全序栅栏。</p>
<footer>—— 据 Boehm and Adve, Foundations of the C++ Concurrency Memory Model, PLDI 2008；SPARC RMO 整理</footer>
</div>

[上一课](/cs/atomic-cache-impl) 保证单个地址的 RMW 不被切开。程序员写互斥时还需要：**解锁之前对共享数据的 store，要被下一位加锁成功的人的 load 看见。** [MESI](/cs/mesi-protocol) 不管两个地址之间的顺序。本课不重做 cache lock。缺口是**acquire/release：比 seq-cst 全栅栏弱、比 relaxed 强的单向屏障。**

## 问题

核 A：`data=1; unlock`。核 B：`lock; x=data`。若 `unlock` 的 store 与 `data` 的 store 在写缓冲里重排，B 可能看到锁已开、`data` 仍是 0。seq-cst fence 能修，但把无关的访存也挡住，[MLP](/cs/mlp-memory-parallelism) 与 store 缓冲收益尽失。缺口不是更重的原子，而是**释放：此前写不能越过释放；获取：此后读不能越过获取。**

<span class="marginnote">C++11 / RISC-V 的 `release`/`acquire`、ARM 的 `dmb ishld` 一类，都是这个单向约束。x86 TSO 对 store 已经很强，许多释放几乎免费，获取仍要限制 store 前的 load 重排——实现上常已满足。</span>

<span class="marginnote">直觉类比：release 像发车前「把所有已装箱的货物先装上车」——它之前写的每一件都必须先出去；acquire 像「签收之后才开始拆箱」——它之后的读不许提前翻墙。这一刀把时间切成两半，两侧的访存不能越境。</span>

<span class="marginnote">为什么重要：没有这对语义，核 A 写完 data=1 再解锁，核 B 可能拿到锁却读回旧值 0——锁形同虚设。配上 release 解锁、acquire 加锁，B 只要加锁成功就必然看见 A 解锁前的所有写，共享数据本身用普通访存即可。</span>

## 方法

微结构：release store 在 SQ 里等到更年长的 store 都变成一致性可见（drain 到某一点）再发出；acquire load 之后的访存等该 load 完成（至少数据返回且获得足够权限）。不必等全世界的无关行。与 [写合并](/cs/write-combining)：WC 区域的 drain 规则更严，release 往往强制冲刷 WCB。

```mermaid
flowchart TD
  ST["更年长的 store"] --> REL["release"]
  REL --> UNL["锁字可见"]
  ACQ["acquire 锁字"] --> LD["更年轻的 load"]
```

## 机制

这是内存模型与 cache 协议的接口：协议提供单行上的「可见」，模型把多行用 acquire/release 串起来。乱序核的 [LSQ](/cs/lsq-disambiguation) 必须禁止跨越这些点的投机重排，或在发现违规时 replay。比下一课事务内存弱：这里没有多行原子提交，只有顺序约束。

relaxed 原子只保证该地址 RMW 的原子性，不提供这把「刀」。seq-cst 在所有核上插一把全序，实现上接近更重的 fence。锁的正确写法是 release 解锁、acquire 加锁，数据本身用普通访存即可——数据竞争自由程序的编译器契约。

第一张图画的是单核里 release 与 acquire 各自约束哪侧访存；这张图回答第二个问题：两个核对同一把锁一交一接时，配对如何把 data=1 送到下一位手里。

```mermaid
flowchart TD
  A["核 A"] --> W["data = 1，普通写"]
  W --> REL["release 解锁：先前的写先全局可见"]
  REL --> OPEN["锁字变为已解锁"]
  OPEN --> ACQ["核 B acquire 加锁成功"]
  ACQ --> R["读 data：保证看到 1"]
```

<span class="marginnote">常见误区：初学者容易把 relaxed 原子当成「又便宜又安全」的万能药。relaxed 只保证那一个地址自身的读改写不被切开，跨地址的先后完全不管——拿它保护共享数据，上面「读到 0」的事故照样发生。</span>

## 边界

本课不把 C++ 的 memory_order 矩阵整页抄来。也不把 Linux `smp_mb` 的各种变体列全。事务内存试图把临界区里的多行读写作一个原子块，失败则 abort，下一课。fence 指令的微结构代价再下一课单独记账。

后课默认：锁配对用 acquire/release 即可表达可见性，不必每条访存 seq-cst。把整个临界区包成硬件事务是另一条路。

## 小结

- acquire/release 是单向屏障，服务锁的可见性，弱于全序 fence。
- 实现上主要约束 SQ drain 与后续访存的发射。
- 多行投机原子块是下一课事务内存。
- 出处：Boehm and Adve, *PLDI*, 2008；SPARC RMO；Hennessy and Patterson, *CA:AQA*。

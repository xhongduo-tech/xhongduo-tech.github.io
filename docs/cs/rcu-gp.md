---
title: RCU 宽限期
date: 2026-09-08
section: cs
---

# RCU 宽限期

<div class="epigraph">
<p>宽限期结束，意味着开始等待之后启动的每个 CPU 都已经过了一次静止状态——于是那时还活着的 RCU 读者都已离开。</p>
<footer>—— 据 McKenney, RCU 实现与《Is Parallel Programming Hard》中对 grace period 的整理</footer>
</div>

[上一课](/cs/rcu-read)禁止在读侧释放旧结点。缺口是检测「世界上没有读者还拿着旧指针」：**grace period（宽限期）**。写者 `synchronize_rcu` 阻塞到一个宽限期结束，或 `call_rcu` 把释放挂到结束后的回调。本课钉静止状态直觉，不当成内核源码。

## 问题

不能给每个读者发票再回收：那等于又一把读锁。RCU 用 CPU 的静止：不在 `rcu_read_lock` 区间内（对抢占 RCU：该核上读侧嵌套为零）。若每个 CPU 都经历过一次静止，则任何在宽限期开始前已进入的读者都已退出——因为读者不可睡、不可被迁走而一直占着读侧（经典模型）。缺口不是再写 `rcu_dereference`，而是这条**全局静止论证**。

<span class="marginnote">直觉类比：把每个 CPU 想成教室里的一排座位，宽限期像下课清点——不追踪「谁借走了哪本书」，只要求每排座位至少空过一次。只要每排都空过一次，那么清点开始前坐在任何座位上的人都必定已经起身离开，旧讲义（旧节点）就可以收走。</span>

<span class="marginnote">`synchronize_rcu` 可能等若干毫秒，写路径不能在硬 IRQ 里做。`call_rcu` 把 free 推迟到软中断或工作队列。</span>

## 方法

写者发布新指针（release）后，等待宽限期，再 `kfree` 旧结点。实现：每 CPU 记 qs 位，时钟或上下文切换报告静止，协调者收集全 1 则 GP 结束。多个写者可共享同一 GP。与[idle](/cs/idle-thread)：idle 通常算静止，否则空闲核会挡住回收。<span class="marginnote">数字实例：`synchronize_rcu` 在典型内核上要等约几毫秒到几十毫秒（至少跨一个全核时钟周期）。写 1000 次配置就是秒级延迟，所以热路径的写者用 `call_rcu` 把释放挂成回调，自己不等——省下的时间照样要用内存峰值来还。</span>

```mermaid
flowchart TD
  PUB["发布新指针"] --> GP["等所有 CPU 静止一次"]
  GP --> FREE["释放旧结点"]
```

与[completion](/cs/completion) 不同：不是等一个生产者，而是等所有可能读者的核过静止。

## 机制

宽限期把内存回收延迟变成可证明的安全点。它是写者的税、读者的零税。过长的读侧临界区拖长 GP，表现为内存峰值——纪律仍是读侧要短。抢占 RCU、SRCU 改静止定义，本课只要求：释放前必须经过实现所承认的 GP。

```mermaid
flowchart TD
  T0["t=0：宽限期开始，某 CPU 上还有老读者"] --> Q1["该 CPU 稍后退出读区，报告一次静止"]
  T0 --> Q2["无读者的 CPU：时钟滴答即记静止"]
  T0 --> Q3["idle 的 CPU：也算静止"]
  Q1 --> ALL["协调者收齐全核静止位"]
  Q2 --> ALL
  Q3 --> ALL
  ALL --> SAFE["老读者必已退出：释放旧结点安全"]
```

<span class="marginnote">常见误区：初学者容易以为 GP 时长由写者决定。实际上它由**最慢的那群读者**决定——一个长时间抱着读侧临界区不放的路径（比如在中断里忘了退出）会独占整条宽限期，其余核全在陪等，内存里的待释放旧节点随之越积越多。</span>

NMI 中的读者要用特殊登记，否则静止检测看不见它们；点到即可。

## 边界

本课不把 Expedited GP 的 IPI 风暴写成运维事故分析，不列举所有 `rcu_barrier` 语义。无锁 CAS 循环另有 ABA，下一课。也不把 GC 分代与 RCU 混成一种运行时。

后课默认：RCU 释放发生在宽限期后。无锁结构用 CAS 换头时，节点被复用会对 CAS 撒谎，下一课 ABA。

## 小结

- 宽限期：所有 CPU 静止过一次 ⇒ 旧读者已离开，可释放。
- 写者付延迟；读侧过长会拖回收。
- 无锁 CAS 的 ABA 是下一课。
- 出处：McKenney，RCU 文献与 *Is Parallel Programming Hard, and, If So, What Can You Do About It?*。

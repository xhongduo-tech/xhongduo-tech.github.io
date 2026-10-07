---
title: 原子读改写
date: 2026-09-08
section: cs
---

# 原子读改写

<div class="epigraph">
<p>测试并置位、比较并交换、取并加，把「读—改—写」收成一条对总线可见的操作；没有它，锁字本身会竞争。</p>
<footer>—— 据 Dijkstra 之后的硬件实践；Love, Linux Kernel Development 整理</footer>
</div>

[上一课](/cs/memory-barrier-os)钉住可见序，但锁字的「若为 0 则置 1」仍是两条访存，中间可被他核切开。[竞争](/cs/race-critical) 的计数器例子正是这种切开。缺口是硬件 **RMW 原子**：`swap`、`cmpxchg`、`fetch_add`，在缓存一致性协议下对该地址表现为不可分。本课只钉对象，不写微架构延迟表。

## 问题

纯软件 Peterson 在 SC 上可互斥，在放宽模型加编译器下脆弱。内核与用户运行时默认用原子指令做锁与引用计数。缺口不是再讲 fence 的分类，而是：**哪一类更新必须是 RMW**——锁字、引用计数、无锁链表的头指针。失败的 CAS 还给出无锁算法的循环，后课再收。

<span class="marginnote">RMW 通常对同一 cache 行上锁（MESI 的 M），他核要等行转手。这就是自旋锁缓存行乒乓的来源。</span>

<span class="marginnote">数字实例：5 个线程各给引用计数加 1，若读改写被切开，两个线程可能同时读到 6、各自写回 7，最终得 9 而不是 10。计数少了就提前释放，正好砸向还在使用的对象——use-after-free 的一种生成方式就这么朴素。</span>

## 方法

`test_and_set`：读旧值并置 1，返回旧值，用来做最简自旋。`fetch_add`：引用计数。`cmpxchg`：期望值匹配才写成新值，否则返回当前。内核把它们包成 `atomic_t` 与 `atomic_long`，并规定是否自带屏障。本课不把每种 ISA 的 `lock` 前缀写完。

```mermaid
flowchart TD
  RMW["原子读改写"] --> TAS["测试并置位: 锁"]
  RMW --> ADD["取并加: 计数"]
  RMW --> CAS["CAS: 条件更新"]
```

单核关中断可以代替短 RMW，但挡不住多核；[锁与关中断](/cs/lock-irq) 下一课把两者搭配。

<span class="marginnote">直觉类比：`fetch_add` 像柜台叫号机——按一下（原子加），机器吐给你一个唯一旧值当号牌，两个人绝不可能拿到同一个号；普通「看一眼当前号、自己心里 +1 再写回去」就必然撞号，这正是丢失更新。</span>

## 机制

原子性相对的是该地址上的其它访存，不是全内存全序——全序仍要屏障。CAS 的「成功」只说明这一刻值匹配，不说明指针所指对象没被回收，那是 ABA，更后一课。引用计数用 fetch_add 避免丢失更新，仍要配对的屏障才能与对象内容同步。

切开的读改写怎么丢更新，一条原子怎么保住：

```mermaid
flowchart TD
  subgraph BAD["普通 读 再 写 两条指令"]
    B1["核 1 读到 5"] --> B2["核 2 也读到 5"] --> B3["各自写回 6 白加一次"]
  end
  subgraph GOOD["fetch_add 一条原子"]
    G1["核 1 原子加 返回旧值 5"] --> G2["核 2 原子加 排队后执行 结果 7"]
  end
```

## 边界

本课不引入 LL/SC 的伪失败与循环重试的全部细节，只承认有的 ISA 用 LL/SC 模拟 CAS。也不把业务逻辑写成「一条 CAS 事务」。用户态关中断不能代替 RMW：那是特权。

<span class="marginnote">常见误区：初学者容易把「原子」听成「事务」。一条 CAS 只保住一个地址的一次更新；「查余额、扣款、记账」这类多步逻辑的整体原子性要靠锁或真事务包住，硬塞进一条 CAS 是无锁算法课里最贵的坑。</span>

后课默认：锁字用原子 RMW 更新。内核里还要决定是否关本地 IRQ，下一课[锁与关中断](/cs/lock-irq)。

## 小结

- RMW 让锁字与计数的读改写不可切。
- CAS/TAS/fetch_add 是三类常用形态；跨核仍碰 cache 行。
- 与 IRQ 的搭配是下一课。
- 出处：Love, *LKD*；Silberschatz et al., *OSC*。

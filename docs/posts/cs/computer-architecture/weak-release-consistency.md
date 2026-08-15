---
title: 弱一致性模型与 Release Consistency
date: 2026-08-07
---

# 弱一致性模型与 Release Consistency

<div class="epigraph">
<p>不是所有内存操作都需要全局顺序——只有同步操作需要，其余交给硬件自由发挥。</p>
<footer>—— 弱内存模型的设计思想</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机体系结构 ｜ Hennessy & Patterson《Computer Architecture: A Quantitative Approach》第 5 章 ｜ 2026-08-07</p>
</div>

## 为什么「全部保序」是浪费

顺序一致性（[[memory-consistency-sequential]]）对**每一次**内存操作都要求全局顺序——但绝大多数访存操作**根本不需要**与其他线程交互。弱模型的核心洞察：**把操作分成「普通操作」与「同步操作」两类，只对同步操作强约束**，普通操作可以自由重排。<span class="marginnote">这又是「以大概率事件为快」：<strong>同步操作是少数（1%），普通访存是多数（99%）——让 99% 自由、1% 严格</strong>，性能与正确性兼得。</span>

## 1 弱一致性（Weak Ordering）

**核心概念**：**弱一致性（weak ordering）**把内存操作分为**普通（data）操作**与**同步（synchronization）操作**，规定：

1. 同步操作之间**按程序序**，且有全局顺序。
2. 普通操作的**重排不能越过同步操作**（同步是分界线）。
3. 处理器的停顿只发生在**同步点**——普通访存之间不要求任何顺序。

效果：两次同步之间的普通读写，硬件与编译器可以**尽情重排**——只要不跨越同步边界。性能大幅释放。

把「自由」的量级算出来。假定一次临界区里有 100 次普通读写、同步操作只有抢锁/放锁 2 次：SC 模型下这 102 次全都要有全局可见顺序；弱模型只有 2 次需要。若单次同步开销是普通读写的 20 倍，弱模型的同步成本约是 SC 的 $\frac{2 \times 20}{102} \approx 0.4$ 倍——**省下的不是一点点，是数量级**。这正是多核时代「性能要由内存模型来换」的经济学：约束越少，重排自由越大，流水线越满。

## 2 释放一致性（Release Consistency, RC）

弱模型把同步操作「一视同仁」；**释放一致性（RC）**进一步把同步操作分两类：

**acquire（获取）**：读类同步，如**抢锁**。其后的普通操作不得越过它（acquire 之后的操作必须看到 acquire 之前同步的可见效果）。
**release（释放）**：写类同步，如**放锁**。其前的普通操作不得拖到它之后（release 之前写的东西必须先于 release 可见）。

用锁的例子（[[synchronization-primitives]] 的 spinlock）：

```cpp
// 抢锁 = acquire：其后的普通读不得越过抢锁点
while (lock.test_and_set(std::memory_order_acquire)) { }   // 自旋等待

// 临界区：普通读写可自由重排，但不能越过 acquire/release 边界
shared_data += 1;

// 放锁 = release：其前的普通写在放锁前必须已可见
lock.clear(std::memory_order_release);
```

**核心概念**：acquire/release 是**不对称**的——acquire 管「进来之后」，release 管「出去之前」。一对 acquire/release 就足以把临界区「框」住：里面的普通读写既不被移出（release 挡前）、也不被外人看见「半成品」（acquire 挡后）。

直观理解这两个名字：acquire 是「**获取**访问权限」——抢到锁的那一刻，之前别人 release 出来的所有写，都必须对我可见；release 是「**释放**」——我把锁让出去的瞬间，我改过的共享数据必须全部已写回、对下一个 acquire 者可见。**一进一出之间，临界区的内容被「框」住**，这就是 RC 的语义骨架，也解释了为什么 C++ 把 `std::memory_order_acquire` 与 `std::memory_order_release` 设计成一对。

## 3 RC 的编程纪律与收益

**纪律**：所有共享数据的访问**必须**放在 acquire/release 对之间；裸的共享访问 = 数据竞争 = 未定义（[[memory-consistency-sequential]] 的 DRF 契约）。
**收益**：临界区内部的读写完全自由重排——**同步代码的性能接近「没有内存模型负担」**。
**代价**：写错就静默出错，调试极难。所以现代语言（C++ 的 `std::memory_order`、Rust 的 `Ordering`、Java 的 `VarHandle`）把 acquire/release 包装成语义清晰的 API。

DRF 契约再强调一遍：**只要程序没有裸共享访问，任何弱模型都能给出 SC 语义的结果**。正确性由「无数据竞争」保证，性能由「弱模型」释放——**两件事，分开卖**。这也是 C++ 内存模型设计时反复强调的一句话：`data-race-free` 的程序在 C++ 里永远表现为顺序一致。

一个经典的 RC 用例是**生产者—消费者**：生产者把数据写入缓冲区，再 release 一个「数据就绪」标志；消费者 acquire 这个标志后读数据。release 保证「缓冲区写入先于标志可见」，acquire 保证「标志可见后数据必然可见」——**一进一出之间，生产者写的数据被原封不动地递到了消费者手里**，而缓冲区本身的普通读写全程自由重排，不必等 SC 的全局顺序。这套模式就是 C++ `std::atomic`/`std::mutex` 的底层语义，也是 Rust `mpsc` 通道、Java `VarHandle` 释放获取语义的实现基础——**「无锁但有 RC 纪律」的队列，正是从这里长出来的**。

## 4 内存屏障指令：手动恢复顺序

弱模型下，需要时用**屏障指令（fence）**手动恢复顺序：

**全屏障（full fence）**：其前后所有读写都不越过（最贵）。
**acquire 屏障 / release 屏障**：只管一侧，更便宜。
RISC-V 的 `fence`、ARM 的 `dmb`/`dsb`/`isb`、x86 的 `lfence`/`sfence` 都是这套。

**辨析｜易错点：** 屏障与原子是两件事。原子（AMO/CAS）保证**不可分割**；屏障保证**顺序**。一个原子操作如果没带 release/acquire 语义，它旁边的普通读写照样可能乱序——**原子 + 屏障（或带语义的原子）才构成完整同步**。

RISC-V 的 `fence` 用两个四位掩码描述前后序：`fence rw, rw` 表示其前所有读写在 fence 前完成、其后所有读写等待——即全屏障。实现 release 只需 `fence rw, w`（只管自己写的发布），实现 acquire 只需 `fence r, rw`（只管自己读的获取）——比全屏障便宜。ARM 的 `dmb`/`dsb`、x86 的 `lock` 前缀对应同一套「不对称屏障」：

```asm
# release：fence rw, w —— 之前的写在 fence 前必须完成
fence rw, w
sw   t0, (a0)        # 放锁（store）
```

**不对称屏障只约束一侧，代价更低，正确用法配 acquire/release 语义**——这也是 `fence.rw` 这类指令在现代 ISA 里被「细粒度化」的原因。

把四种屏障/原子家族速查如下：

| 原语 | 管什么 | 典型指令 |
| --- | --- | --- |
| 全屏障 | 前后读写都不越过 | x86 `mfence`、ARM `dsb` |
| acquire 屏障 | 只管读获取一侧 | ARM `ldar`、RISC-V `fence r, rw` |
| release 屏障 | 只管写发布一侧 | ARM `stlr`、RISC-V `fence rw, w` |
| 原子（AMO/CAS） | 不可分割性（不含顺序） | x86 `lock` 前缀、RISC-V `amoswap` |

**记忆口诀：原子管「不可分割」，屏障管「先后顺序」，两者配齐才是一次完整同步**。

## 5 TSO 与其他模型的位置

内存模型是一个谱系：

| 模型 | 保序程度 | 代表 | 编程难度 |
| --- | --- | --- | --- |
| **SC** | 全部保序 | 教学 | 最低 |
| **TSO** | 写仍保序（store 缓冲友好） | **x86** | 中 |
| **弱一致性/RC** | 只保同步点 | **ARM、RISC-V** | 高 |

**核心概念**：x86 的 TSO（Total Store Order）介于 SC 与弱模型之间：它允许「读越过未提交的写」（store buffer 造成），但保持「写-写」顺序——所以 x86 上朴素代码比 ARM/RISC-V 更容易正确。<span class="marginnote">这就是「x86 上能跑的并发程序搬到 ARM 上就错」的根源：<strong>ARM/RISC-V 是弱模型，需要显式屏障；x86 的 TSO 替你保了更多序</strong>。写可移植并发代码，一律按弱模型写。</span>

## 6 核心对比表

> 本节为纯概念主题，以核心对比表替代公式解析。

| 维度 | SC | TSO（x86） | 弱一致性 / RC |
| --- | --- | --- | --- |
| 普通操作顺序 | 全部保序 | 写保序 | 自由 |
| 同步操作 | 就是普通操作 | 普通 + lock | **单独一类** |
| acquire/release | 不需要 | 需要（lock 已带） | 需要显式 |
| 性能 | 最差 | 中 | 最好 |
| 正确性负担 | 无 | 小 | 大（要懂模型） |

## 7 小结

- **弱一致性**只约束同步操作，普通访存自由重排——性能与正确性的分水岭。
- **释放一致性（RC）**把同步分成 **acquire（管后）/ release（管前）**，锁的语义由此建立。
- 纪律：共享访问必须放在 acquire/release 对之间，否则是数据竞争。
- **fence 指令**手动恢复顺序；原子管「不可分割」、屏障管「顺序」——两者互补。
- 模型谱系 SC → TSO → 弱/RC；**ARM/RISC-V 是弱模型，写并发代码按最弱模型写最保险**。

在下一节，我们把多核的性能账算起来——**多核处理器的性能建模与扩展性限制**。

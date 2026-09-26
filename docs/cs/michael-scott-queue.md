---
title: Michael–Scott 无锁队列
date: 2026-09-08
section: cs
---

# Michael–Scott 无锁队列

<div class="epigraph">
<p>哨兵节点永远在；enqueue CAS 尾的 next，再帮着推进 tail；dequeue CAS 头。线性化在成功的 CAS 上。</p>
<footer>—— 据 Michael and Scott, Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms, PODC 1996；Herlihy and Shavit 整理</footer>
</div>

[上一课](/cs/lsm-tree-ds) 的并发在文件层。[函数式队列](/cs/functional-queue) 无共享写。[无锁与 ABA](/cs/lockfree-aba) 已钉 CAS 语义。[原子 RMW](/cs/atomic-rmw) 现成。本课不 compaction。缺口是 Michael–Scott 队列：实用无锁 FIFO。

## 问题

多生产者多消费者 FIFO。加锁队列简单但有锁。MS 队列：`head`、`tail` 指向节点链表，首节点是哨兵（dummy）。入队：在 `tail->next` 上 CAS 从空到新节点，成功后再 CAS `tail` 前移；若看见别人已接上 next，则帮推进 tail 再试。出队：读 `head->next` 为真头，CAS `head` 前进，旧哨兵可回收。缺口是**两步 CAS 与「帮忙」以保持无锁**。

<span class="marginnote">术语翻译：哨兵（dummy）节点是不装数据的占位节点，永远待在队头。有了它，「队列空」与「队列非空」结构一致——真头永远是 head 的 next——enqueue/dequeue 就不必为空队列写特殊分支。</span>

<span class="marginnote">直觉类比：CAS（比较并交换）像贴封条——「门上还是我看到的旧封条，就换成我的；不是，就宣告失败」。失败不改任何东西，读一眼重贴即可；无锁算法就是靠反复「读-贴」而不是排队等锁前进的。</span>

<span class="marginnote">Michael and Scott, *PODC*, 1996。ABA：节点回收后复用骗 CAS，需 hazard、epoch 或带 tag 指针——下一课。</span>

## 方法

节点含 `next` 与值。空队列：`head==tail` 且 next 空。实现必须先写节点内容再 CAS 链入（发布）。失败重试是无锁允许的活锁风险，实践有界。

```mermaid
flowchart TD
  ENQ["enqueue"] --> CASN["CAS tail.next: null → node"]
  CASN --> CAST["CAS 推进 tail"]
  DEQ["dequeue"] --> CASH["CAS 推进 head"]
```

与环形数组 SPSC：单生产者可无 CAS 只屏障，更快；MS 面向 MPMC。与跳表：队列无序键，只 FIFO。

## 机制

线性化：入队成功挂 next 的 CAS；出队成功移 head 的 CAS。帮忙保证 tail 不落后太多，避免无锁失败。本课不给可复现的破坏性时序作业。

<span class="marginnote">常见误区：以为出队成功就能立刻 `free` 旧哨兵。别的线程可能正握着指向它的旧指针在读 next——一 free 就是 UAF，节点地址复用还会骗过 CAS 形成 ABA。安全回收要 hazard pointer 或 epoch，下一课。</span>

```mermaid
flowchart TD
  READ["读 tail 与 tail.next"] --> LAG{"tail 落后了? next 已非空?"}
  LAG -->|"落后"| HELP["帮别人 CAS 推进 tail"]
  HELP --> RETRY["重读 tail 再试"]
  LAG -->|"没落后"| CAS["CAS tail.next 挂上新节点"]
  CAS --> ADV["再 CAS 推进 tail"]
  RETRY --> LAG
```

回收：出队后节点不能立刻 `free`，否则 ABA 与 UAF。下一课 hazard pointer。

## 边界

本课不写完整可移植 C++ 实现当唯一答案。阻塞队列（两锁）是论文另一半，点名。栈的无锁更易 ABA，与 hazard 一起下一课。

后课默认：MPMC 无锁 FIFO 用 MS 队列。回收与栈用 hazard pointer。

## 小结

- MS 队列：哨兵链表，CAS next 与 head/tail。
- 无锁靠帮忙推进 tail；注意发布序。
- 回收与 ABA 交给 hazard。
- 出处：Michael and Scott, *PODC*, 1996。

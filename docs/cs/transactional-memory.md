---
title: 事务内存
date: 2026-09-08
section: cs
---

# 事务内存

<div class="epigraph">
<p>用 cache 的独占与共享跟踪读集写集：提交时若没有冲突则一次性对全局可见，否则丢掉推测写，像误预测一样恢复。</p>
<footer>—— 据 Herlihy and Moss, Transactional Memory: Architectural Support for Lock-Free Data Structures, ISCA 1993 整理</footer>
</div>

[上一课](/cs/acquire-release) 用锁的两端串可见性，锁本身仍是真共享行上的争用。[原子](/cs/atomic-cache-impl) 一次只罩一个地址。本课不重讲 dmb。缺口是**硬件事务内存（HTM）：把临界区里的多行读写当成可提交或可放弃的一块**，用 cache 状态当读集/写集。

## 问题

细粒度锁难写；粗粒度锁把伪共享变成真串行。无锁算法用 CAS 循环，可组合性差。Herlihy–Moss 的提案：指令标明事务开始/结束，硬件保证「看起来原子」。缺口不是更强的 fence，而是**用已有的一致性探询检测冲突：他核对你读集的写、或对你写集的任何访问，导致 abort。**

<span class="marginnote">术语翻译：读集是事务读过、不许别人中途改的地址清单；写集是写过、提交前不许别人碰的清单。硬件借 cache 行的共享/独占状态顺手记账，软件不必挨个登记。</span>

<span class="marginnote">Intel TSX（RTM/HLE）把 L1 当写缓冲：容量溢出、异常、某些指令都会 abort，软件必须有回退路径。这是有限 HTM，不是无限理想事务。</span>

## 方法

事务中：load 把行留在 S/E 并加入读集；store 把行升到 M 但**不对外提交**（可把数据放在 L1 私有副本）。探询打中读集或写集则 abort：冲刷写集，像 [推测恢复](/cs/speculation-recovery)。提交：写集一次性变成全局 M 可见，或靠「提交前检查读集仍有效」。

<span class="marginnote">数字实例：L1 通常只有 32–64 KB。事务里碰的行一超过 L1 装得下的量——比如遍历几 MB 的哈希表——硬件直接 abort。所以 TSX 文档劝你把临界区写小，别把整段循环塞进事务。</span>

```mermaid
flowchart TD
  BEGIN["事务开始"] --> RS["读集 S"]
  BEGIN --> WS["写集私有 M"]
  PROBE["他核探询冲突"] --> ABORT["丢写集，回退"]
  END["提交"] --> PUB["写集一次性可见"]
```

## 机制

与 ROB 推测的差别：事务跨越的是**内存行**，可以比 ROB 窗口长，但受 L1 容量限制。与锁：无冲突时无锁字颠簸；有冲突时 abort 重试可能活锁，需要指数退避或回退到锁。不能把 HTM 当成可以省略 [acquire/release](/cs/acquire-release) 的魔法——失败路径仍是锁。

禁止在事务里做的事（I/O、某些特权、过深嵌套）来自「写集无法回滚外部世界」。

<span class="marginnote">常见误区：把 abort 当出错。abort 恰是 HTM 的正常工作方式——冲突就整体回滚重试，与分支误预测冲刷流水线同理；真正要防的是 abort 率高到重试活锁，那时该退避或退回锁。</span>

```mermaid
flowchart TD
  TRY["尝试事务<br/>读写走 L1 私有副本"] --> PROBE{"探询撞上冲突？＜br/＞或容量溢出？"}
  PROBE -->|"都没有"| COMMIT["提交：写集一次性全局可见"]
  PROBE -->|"撞上了"| ABORT["abort：丢弃私有写"]
  ABORT --> RETRY{"重试次数＜br/＞超过阈值？"}
  RETRY -->|"没有"| BACK["指数退避等待"]
  BACK --> TRY
  RETRY -->|"超过"| LOCK["回退到传统锁路径"]
```

## 边界

本课不把 STM（纯软件）当微结构主干。也不提供可运行的锁省略攻击或侧信道利用；事务 abort 会改 cache，安全课可再引用。下一课把显式 fence 的周期数钉下来：没有事务、没有锁时，程序员仍会下栅栏。

后课默认：HTM 用 cache 做多行推测原子，容量与指令集有限。显式 fence 仍是最可控的顺序工具，但很贵。

## 小结

- HTM 用读集/写集与探询做冲突检测，提交或 abort。
- 容量与禁止指令强制软件回退。
- 显式 fence 的微结构代价是下一课。
- 出处：Herlihy and Moss, *ISCA*, 1993；Intel TSX 公开文档。

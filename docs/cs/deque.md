---
title: 双端队列
date: 2026-09-08
section: cs
---

# 双端队列

<div class="epigraph">
<p>两端都能进、两端都能出；栈和队列是它在操作集上的两个真子集。</p>
<footer>—— 据 Knuth, The Art of Computer Programming 卷 1；Cormen, Leiserson, Rivest and Stein 整理</footer>
</div>

[上一课](/cs/queue-buffer)钉了 FIFO：尾进头出，并声明双端「主干不提前」。栈是一端 LIFO。本课不重讲环形模运算。缺口是 deque 合同：`push_front`/`push_back`/`pop_front`/`pop_back` 均摊还或最坏 $\Theta(1)$，用来做滑动窗口、两端生长的工作列。表示可以是双向链表，也可以是后课环形数组的两端指针。

## 问题

只提供队列则无法在队头插入；只提供栈则无法在底端取。算法里「最新的从一端进、过期的从另一端丢」（单调队列）需要两侧。缺口不是新的节点类型，而是**把序列的两端都暴露成 $O(1)$ 端点操作**，中间按下标仍可昂贵。

<span class="marginnote">直觉类比：deque 像一条两端都开了门的走廊——元素从前门或后门进出都行；栈是只开后门的走廊（后进先出），队列是后门进、前门出的走廊（先进先出）。合同上看，栈与队列不过是把 deque 的四个门各关掉两个剩下的用法。</span>

空时四端操作无定义或失败；实现不得把「头超过尾」留成未定义环。

<span class="marginnote">STL deque 常用分块数组，使两端增长不必整表复制。本课先钉合同；块列是表示优化。</span>

## 方法

链表表示：双向表的哨兵两侧即两端，四个操作都是改常数条边。数组表示：头尾两个下标，向左增长时 `head = (head-1+cap) mod cap`——这已是环形缓冲的雏形，下一课专收。

```mermaid
flowchart LR
  PF["pop_front"] --> SEQ["序列"]
  SEQ --> PB["pop_back"]
  PUSHF["push_front"] --> SEQ
  PUSHB["push_back"] --> SEQ
```

栈 = 只开同一端的 push/pop。队列 = 只开 `push_back` + `pop_front`。合同包含关系让测试用例可以共享。

## 机制

与[局部性原理](/cs/locality-principle)：链表 deque 两端操作仍是指针追逐；要扫中间更差。高频滑动窗口若元素连续，数组环形更好。本课允许两种，后课把环形从队列里提出来当独立表示。

单调队列是 deque 两端各司其职的典型用户，两端动作分工明确：

```mermaid
flowchart TD
  NEW["新元素到达窗口右端"] --> CMP["从队尾弹出比它小的下标"]
  CMP --> IN["push_back：新下标入队尾"]
  OLD["队头下标滑出窗口范围"] --> OUT["pop_front：丢掉过期者"]
  QUERY["要当前窗口最大值"] --> PEEK["读队头，一次即答"]
```

<span class="marginnote">数字实例：滑动窗口最大值里，每个元素至多 `push_back` 一次、至多被从队尾或队头弹掉一次，总操作不超过 $2n$ 次，均摊 $\Theta(1)$。若改成每个窗口重扫一遍取最大，就是 $n \times k$ 次——$n=10^6$、$k=1000$ 时差出三个数量级。</span>

<span class="marginnote">常见误区：初学者容易以为「双端都能 $O(1)$」等于中间也能 $O(1)$。deque 的合同只钉两端，按下标访问中间在链表表示下是 $O(n)$；把随机访问当成免费能力，是性能分析里最晚被抓到的那口锅。</span>

中间插入删除若也要 $\Theta(1)$，调用方必须持有节点句柄，那是链表合同，不是下标 deque。混用会把随机访问假装成 $O(1)$。

## 边界

本课不引入优先双端（两端按键），那是堆的变体。也不写无锁 deque。扩容数组在两端同时增长时复制规则更烦，摊还课用单端表扩张即可，不必先在这里证明。

后课默认：需要两侧端点用 deque。定长高性能缓冲把数组收成环，头尾追逐，下一课只谈那一种布局。

## 小结

- deque 四端 $\Theta(1)$；栈与队列是操作子集。
- 链表或环形数组都能表示；局部性偏向数组。
- 环形布局与满空区分是下一课。
- 出处：Knuth, *TAOCP* 卷 1；Cormen et al. 对双端队列的接口。

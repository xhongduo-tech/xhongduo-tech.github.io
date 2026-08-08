---
title: 队列的顺序表示与循环队列
date: 2026-08-07
---

# 队列的顺序表示与循环队列

<div class="epigraph">
<p>先来先服务，后来后服务——公平是最简单的调度策略。</p>
<footer>—— 排队论格言（First come, first served）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据结构 ｜ 严蔚敏《数据结构》 §3.5 ｜ 2026-08-07</p>
</div>

## 为什么从队列讲起

栈限制「一端操作」，队列则限制「两端分工」：**一端只负责进（队尾），另一端只负责出（队头）**。于是它给出后进先出完全相反的纪律——**先进先出（FIFO, First In First Out）**。操作系统里的任务调度、打印机缓冲、BFS 遍历、消息队列，全是队列的战场。本节实现顺序队列，并解决它最经典的工程问题——**假溢出**：普通数组做队列，出队后「空出的头位置」无法复用，空间用一点废一点。循环队列用「取模」让数组首尾相接，一举化解。

## 1 队列的定义

**队列（queue）**：只允许在表的一端进行插入、在另一端进行删除的线性表。插入的一端叫**队尾（rear）**，删除的一端叫**队头（front）**。

队列的运算与栈对称：

- 入队（enqueue / push）：在队尾插入元素；
- 出队（dequeue / pop）：删除队头元素并返回；
- 读队头（getfront）：只读不删；
- 判空 / 判满 / 取长度。

队列用于「按到达顺序处理」的场景，与栈的「按逆序处理」形成互补。<span class="marginnote">栈与队列是同一枚硬币的两面：栈<strong>后进先出</strong>，队列<strong>先进先出</strong>。判断该用哪个，只需问一句：「先来的是先处理，还是后处理？」——递归与 DFS 选栈，层次遍历与调度选队列。</span>

## 2 朴素顺序队列的缺陷：假溢出

顺序队列用数组 + 两个下标 front（队头）与 rear（队尾）：

初始：`front = rear = 0`；
入队：`data[rear] = x; rear++`；
出队：`front++`。

这个朴素版本有个致命缺陷。反复入队出队后，`rear` 顶到数组末尾，`front` 却已越过若干「空出来的」位置——此时数组前半段全空，但 `rear == MAXSIZE`，判成「满」无法再入队。**明明有空位却装不下，这就是假溢出（false overflow）**。<span class="marginnote">假溢出的根源是<strong>指针只增不减</strong>：front 和 rear 都单调前进，出队的空间再也回不到入队端。解决思路就一个——让下标在数组范围内「绕圈」，即循环队列。</span>

## 3 循环队列：取模绕圈

循环队列把数组看成首尾相接的环：`front`/`rear` 到末尾后，下一步回到下标 0。所有下标移动都用取模：

$$
\text{rear} = (\text{rear} + 1) \% \text{MAXSIZE}, \qquad \text{front} = (\text{front} + 1) \% \text{MAXSIZE}
$$

这样出队腾出的头端空间可以被后续入队复用，假溢出消失。

**但循环带来一个新问题：判空与判满。** 空队列时 `front == rear`；入队把队列填满后，`rear` 绕一圈又追上 `front`，同样有 `front == rear`——同一个条件对应两种状态，必须区分。严蔚敏教材的处理是**牺牲一个存储单元**：

$$
\text{队空}:\ front = rear \qquad\qquad \text{队满}:\ (rear + 1) \% \text{MAXSIZE} = front
$$

即 `rear` 即将追上 `front` 时就视为满，留一个空位做「哨兵」。<span class="marginnote">这是「<strong>用空间消歧义</strong>」的典型：牺牲一格，换来 front==rear 唯一对应队空。另一种做法是加一个 `count` 计数器或 `flag` 标志，牺牲一字节换满状态判定——三种方案选哪种，取决于空间与可读性的权衡。</span>

## 4 公式解析：循环队列的长度

给定 `front` 与 `rear`，队列中元素的个数为：

$$
\text{length} = (\text{rear} - \text{front} + \text{MAXSIZE}) \% \text{MAXSIZE}
$$

- **第一步，读「直接相减」为什么错**：未绕圈时长度就是 `rear - front`；但绕圈后 `rear < front`（如 `rear = 1, front = 6`），直接相减得负数，毫无意义。
- **第二步，读懂「加 MAXSIZE 再取模」**：先补上 `MAXSIZE` 把负数扳正，再对 `MAXSIZE` 取模，把任何越界的结果拉回 $[0, \text{MAXSIZE}-1]$。$(1 - 6 + 10) \% 10 = 5$，长度 5，正确。
- **第三步，这是一个通用技巧**：凡是「环形」下标差，都要用「加 MAXSIZE 再取模」归一化。环形缓冲区、循环队列、哈希表线性探测回绕，共用这一条公式。

## 5 循环队列的入队与出队

```c
Status EnQueue(SqQueue &Q, ElemType x) {
    if ((Q.rear + 1) % MAXSIZE == Q.front) return ERROR;  /* 判满（牺牲一格） */
    Q.data[Q.rear] = x;
    Q.rear = (Q.rear + 1) % MAXSIZE;                      /* 队尾取模前进 */
    return OK;
}

Status DeQueue(SqQueue &Q, ElemType &x) {
    if (Q.front == Q.rear) return ERROR;                  /* 判空 */
    x = Q.data[Q.front];
    Q.front = (Q.front + 1) % MAXSIZE;                    /* 队头取模前进 */
    return OK;
}
```

**重点：循环队列的两个指针「只增不减」，靠取模回绕，因此不存在假溢出；同时每次操作都是 $O(1)$，且不搬动任何元素。** 队满与队空必须各判一次，判反任何一个都会让数据错位。

**辨析｜易错点：牺牲单元法的「浪费」是多少？** 该方案实际能容纳 `MAXSIZE - 1` 个元素，最多浪费 1 格。与之对比，朴素顺序队列最坏浪费近乎整个数组——一格的代价换来整个数组的复用，这笔账非常划算。<span class="marginnote">环形缓冲区是循环队列的直系后代：生产者写入尾端、消费者读走头端，双方只碰自己的指针，天然无锁可近似单生产者单消费者场景。操作系统课程里的「生产者-消费者」、网络栈的环形发送缓冲，都在用这个结构。</span>

## 6 小结

- 队列是**一端入、一端出**的线性表，特征为**先进先出（FIFO）**。
- 朴素顺序队列的**假溢出**：指针只增不减，出队空间无法复用。
- 循环队列用取模绕圈，一举消除假溢出。
- 判空 `front == rear`；判满 `(rear + 1) % MAXSIZE == front`（牺牲一格）。
- 长度公式 `(rear - front + MAXSIZE) % MAXSIZE`，环形下标的通用归一化技巧。
- 循环队列操作均 $O(1)$、不搬元素，是环形缓冲区的原型。

在下一节，我们用链表实现队列——**链队列**，它不设上限、无需判满，是深度不可预估场景下的第一选择。

---
title: Paxos 的直观理解：Basic Paxos 详解
date: 2026-08-07
---

# Paxos 的直观理解：Basic Paxos 详解

<div class="epigraph">
<p>Paxos 难懂，不是因为算法复杂，而是因为它刻意剥掉了一切偶然的细节，只留下共识的骨架。</p>
<footer>—— 莱斯利 · 兰波特（Leslie Lamport），Paxos Made Simple，2001</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ MIT 6.824 第7讲 / Lamport 2001 ｜ 2026-08-07</p>
</div>

## 为什么从 Basic Paxos 开始

FLP 说「完全异步不可解」，可 Raft、Chubby、etcd 都在跑——因为现实世界介于同步与异步之间。**Paxos** 是第一个真正解决共识的算法：它用一个优雅的两阶段协议，在「多数派健康」的前提下，保证一致性永不破坏、并最终能达成决定。<span class="marginnote">Paxos 由 Lamport 于 1989 年提出，1998 年正式发表，2001 年写了《Paxos Made Simple》试图让它可读。它是 Chubby、Google 内部几乎所有一致性组件的底层，也是 Raft 的前身。理解 Basic Paxos = 理解共识的「最小骨骼」。</span>

## 1 角色与法定人数

Basic Paxos 把节点分成三个逻辑角色：

**提议者（proposer）**：提出要共识的值（谁当 leader、日志第 N 条写什么）。
**接受者（acceptor）**：对提议投票并存储结果——绝大多数节点都当 acceptor。
**学习者（learner）**：观察共识结果（可以就是接受者自己）。

关键机制是**法定人数（quorum）**：一个提议要「成功」，必须被**严格多数**的 acceptor 接受。为什么多数派够？因为**任意两个多数派必然相交**——3 节点中 2+2 相交 1，5 节点中 3+3 相交 1。这个相交的节点，是前后两次决策之间传递「记忆」的桥梁。<span class="marginnote"><strong>辨析｜易错点：</strong>Paxos 的多数派是「过半」，不是「全员」。3 节点集群挂 1 个仍能共识，挂 2 个就停摆。多数派相交保证「新决策必然知道旧决策」，这是 Paxos 全部正确性的几何基础。</span>

## 2 两阶段协议：Prepare 与 Accept

Basic Paxos 的一轮共识分两个阶段，用**轮次号（ballot number）**防止乱序：

**阶段一（Prepare）**：

1. 提议者选一个递增的轮次号 $n$，向所有 acceptor 发送 Prepare($n$)。
2. acceptor 收到后，若 $n$ 大于它见过的最大轮次号，则承诺「不再接受轮次号小于 $n$ 的提议」，并返回它**已接受的最高轮次号**对应的值（若有）；否则拒绝。

**阶段二（Accept）**：

3. 提议者收集到多数派的 Prepare 应答后，从中选出「已接受的值」里轮次号最高的那个作为本轮提议值 $v$（若无则用自己新提的值），向所有 acceptor 发送 Accept($n, v$)。
4. acceptor 若仍遵守对 $n$ 的承诺（$n$ 是它见过的最大号），就接受 $v$，并广播「已接受」给学习者。

多数派收到 Accept 并接受后，**$v$ 就是共识结果**。<span class="marginnote">两条规则是全部精髓：<strong>acceptor 的承诺</strong>（不再接受更小轮次）+ <strong>提议者取最高已接受值</strong>。前者防止旧轮次覆盖新轮次，后者保证「如果已经有值被多数派接受，新提议者必然重新提出那个值」——共识因此不会被推翻。</span>

## 3 为什么多数派相交能保证安全

Paxos 的安全证明（不变式）可以浓缩成一句话：**一旦某个值 $v$ 被多数派接受，之后所有成功的新提议都必须是 $v$**。

推理链：

假设 $v$ 已被某个多数派 $Q_1$ 接受，轮次号 $n_1$。
任何新提议者想成功，必须通过 Prepare 获得某个多数派 $Q_2$ 的应答。
$Q_1 \cap Q_2 \neq \emptyset$：相交节点「知道」$v$ 已在 $n_1$ 被接受。
新提议者取「已接受值里轮次号最高者」，而 $v$ 的轮次 $n_1$ 是它可见的最高者（或之一）→ 新提议值就是 $v$。
于是新 Accept 提议 $v$，acceptor 因承诺接受 → 决策仍是 $v$。

这条链每一环都靠「多数派相交」的几何事实撑住——**没有相交，就没有记忆传递，共识就退化成各自为政**。<span class="marginnote">注意安全证明<strong>不依赖任何时间假设</strong>：无论消息多慢、多乱，只要多数派相交，决策不可回退——Paxos 的安全是「无条件」的。而<strong>活性（liveness）</strong>——一定能达成共识——才依赖超时等同步假设（FLP 的阴影只落在活性上）。</span>

## 4 公式解析：轮次号与承诺

把 acceptor 的承诺规则写成形式化不变式。设 acceptor $a$ 已承诺的最小轮次下界为 $m_a$，已接受的最高轮次为 $n_a$：

$$
\text{Promise: } \forall n < m_a:\; a \text{ 不接受 } (n, \cdot) \qquad
\text{Accept 条件: } n \ge m_a \;\Rightarrow\; a \text{ 接受 } (n, v)
$$

拆解：

- $m_a$ 是「承诺线」：acceptor 在 Prepare($n$）应答后，$m_a \leftarrow n$，从此**拒绝任何轮次小于 $n$ 的 Accept**——防止旧轮次死灰复燃。
- 接受条件：只有轮次号 $n \ge m_a$ 的 Accept 才被接受。
- 两者合起来：**轮次号单调递增的提议才可能成功**，而轮次号是提议者全局递增分配的——乱序、重试、迟到的消息都无法把已定决策推翻。

这条不变式的工程含义：**轮次号就是 Paxos 的时间轴**。实现时要保证轮次号全局唯一且递增（节点 ID + 计数器），否则两个提议者可能撞号，承诺与接受都失去意义——这是 Paxos 实现最常见的 bug 源头之一。

## 5 Basic Paxos 的代价：活性问题

Basic Paxos 是安全的，但**活性**有缺陷：多个提议者同时抢着提，可能形成「活锁」——每个提议者都不断升高轮次号，压过对方，结果谁也达不成共识。工程解法：

- 引入**唯一的 leader（领导者）**：同一时刻只让一个提议者活跃，其他人等待。
- leader 的轮次号持续递增，不再有竞争——这就是 **Multi-Paxos** 的优化方向。

Basic Paxos 的价值在于它是「共识的原子操作」：安全无条件、活性靠同步假设。真实系统（Raft、Chubby）都是在此基础上加 leader 与日志复制的工程化封装。<span class="marginnote">Lamport 常说 Paxos 的精髓被形式化表述掩盖了。一个实用的学习路径是：先把 Basic Paxos 的「两阶段 + 多数派 + 承诺」画成时序图跑通，再去看 Multi-Paxos 和 Raft 如何把「每轮都跑两阶段」优化成「一次选主、持续复制」。</span>

## 6 小结

- **Basic Paxos** 三角色：提议者、接受者、学习者；决策靠**严格多数**接受。
- **两阶段**：Prepare（探测 + 承诺）→ Accept（提交 + 接受）；轮次号单调递增。
- 核心不变式：**一旦多数派接受了 $v$，之后所有成功提议都是 $v$**——靠多数派相交的几何事实保证。
- 安全（一致性）无条件成立，活性（终止）依赖同步假设——这是 FLP 在 Paxos 上的落点。
- Basic Paxos 的活性缺陷（活锁）→ 引入 leader → **Multi-Paxos** 与 Raft 的登场。

在下一节，我们把「每轮共识」优化成「一次选主 + 持续拍板」——**Multi-Paxos 与 Leader 选举**。

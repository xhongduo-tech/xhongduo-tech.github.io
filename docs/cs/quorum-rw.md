---
title: Quorum 读写
date: 2026-09-08
section: cs
---

# Quorum 读写

<div class="epigraph">
<p>写集与读集必须相交。相交点上带着足够新的时间戳，读就能拼出最近一次完成的写。</p>
<footer>—— 据 Gifford, Weighted Voting for Replicated Data, SOSP 1979；Attiya, Bar-Noy and Dolev, Sharing Memory Robustly in Message-Passing Systems, JACM 1995 整理</footer>
</div>

上一课[链式复制](/cs/chain-replication)用拓扑固定提交点。缺口是**集合相交**：不排成链，也能让每次读撞上每次写。本课钉 $R+W\gt n$（及权重推广），不把 Dynamo 的 sloppy quorum 提前当默认。后课反熵处理「写没到齐」。

## 问题

$n$ 个副本，写等 $W$ 个 ack，读问 $R$ 个，取时间戳最大者。若 $R+W\gt n$，任一读集与任一写集共享至少一节点，该节点若保存了那次写，读就能看见。Attiya–Bar-Noy–Dolev（ABD）在异步崩溃下用两轮读/写给出线性一致寄存器：读也要再传播，防止旧值回写。

缺口：相交是安全的组合学；延迟是等最慢的那个法定人数成员。$W=n,R=1$ 写最慢读最快；$W=1,R=n$ 相反；$W=R=\lfloor n/2\rfloor+1$ 对称。这是[PACELC](/cs/cap-pacelc)无分区时的旋钮。

<span class="marginnote">Gifford 加权投票：节点带票数，法定人数是票和，不是台数。ABD 说明消息传递上模拟共享内存要两轮。</span>

## 方法

每值带单调时间戳（Lamport 钟或主序）。写：把新值送到 $W$ 个。读：收 $R$ 份，选最大时间戳；若要线性一致，把该值再写回 $W$（ABD 的 read-repair/传播）。时间戳不可比时不能只靠墙钟——接[物理时钟](/cs/clock-drift)的 $\varepsilon$。

```mermaid
flowchart TD
  WSET["写集 |W|"] --> INT["非空相交"]
  RSET["读集 |R|"] --> INT
  INT --> TS["最大时间戳"]
  TS --> VAL["读值 / 写回"]
```

分区：多数派只在一侧，另一侧 $W$ 凑不齐则停写（CP）。若降低 $W$ 使 $R+W\le n$，相交消失，进入最终一致。

## 机制

Sloppy quorum（后课 Dynamo）：写进提示节点，相交暂时不成立，靠反熵补。那是故意放弃 ABD 安全换可用。严格 quorum 下，失败的写可能只到 $W-1$：读可能看见也可能看不见，线性一致要求把不确定的写当成未提交或用两阶段。工程上常用「写满 W 才 ack 客户端」，未 ack 的写以重试+幂等消化。

```mermaid
flowchart TD
  N["5 个副本"] --> W2{"参数怎么配?"}
  W2 -->|"W=5, R=1"| C1["读最快, 写要等全部"]
  W2 -->|"W=1, R=5"| C2["写最快, 读要等全部"]
  W2 -->|"W=3, R=3"| C3["两侧对称, 都要过半"]
  C1 --> I["R+W=6 > 5, 相交成立"]
  C2 --> I
  C3 --> I
  I --> OK["任一读必撞上最近写"]
```

<span class="marginnote">数字实例：$n=5$、$W=3$、$R=3$ 时，写集占 3 个节点、读集也占 3 个，5 个位置装不下两组互不相交的 3 元集——鸽笼原理保证至少重合 1 个。若把 $W$ 降到 2，则 $2+3\le 5$，读集恰好落在没收到写的那两个节点上时，就会读到旧值。</span>

<span class="marginnote">直觉类比：可以把它想象成往 5 个柜子里同时塞留言，写操作只要塞进 3 个柜子就算完成；读操作随机打开 3 个柜子找最新留言。因为「3+3 比 5 多」，打开的柜子里必定有一个见过最新那条留言。</span>

<span class="marginnote">常见误区：凑齐多数派写才 ack，不代表每个写都一定成功——3 个里挂 1 个，写就停在 2 份，读可能看见也可能看不见。线性一致的正确处理是把它当未提交，靠重试与幂等消化，而不是当作「最终会到齐」。</span>

权重：慢盘少投票，SSD 多投票，仍要最坏相交。不要把权重当成负载均衡的充分条件。

本课不把 Raft 多数派日志当寄存器 quorum：共识是对**操作序列**的相交，寄存器是对**单值**的相交。后课 RSM 会接上。

## 边界

本课不写 Merkle 树，不引入 LWW 细节。不把「三个九」当 $n=3$ 的证明。后课默认：$R+W\gt n$ 加单调时间戳给崩溃异步寄存器；要可用性就放松相交并接受冲突。Dynamo 把放松做成系统。

相交是集合论，时间戳是打破平局的全序。缺一则读无「最近」。

## 小结

- $R+W\gt n$（或加权票和）保证读写集相交。
- ABD：异步线性一致寄存器要读传播。
- 相交放松即放弃该寄存器的线性一致。
- 出处：Gifford, SOSP 1979；Attiya, Bar-Noy and Dolev, JACM 1995。

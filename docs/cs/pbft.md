---
title: PBFT
date: 2026-09-08
section: cs
---

# PBFT

<div class="epigraph">
<p>部分同步下用 $n=3f+1$ 做拜占庭状态机：预准备、准备、提交三阶段，证书交叉验证，视图更换换主。</p>
<footer>—— 据 Castro and Liskov, Practical Byzantine Fault Tolerance, OSDI 1999 整理</footer>
</div>

上一课[VR](/cs/viewstamped-replication)的视图更换假定备份只崩溃或慢。缺口是**备份撒谎**：准备证书必须防伪造、防分裂。本课接[拜占庭将军](/cs/byzantine-generals)的 $n\gt 3f$，时间用部分同步，不把口头 OM 递归再写一遍。后课对照 2PC：2PC 连崩溃协调者都堵，更不抗撒谎。

## 问题

主（primary）在视图里给请求编号并广播 pre-prepare。备份验证后广播 prepare。收集 $2f$ 个匹配 prepare（加自己）形成 prepared 证书——保证诚实节点对「这个序号是这个请求」有交集。再广播 commit，收集 $2f+1$ 个 commit 后执行。三阶段让诚实节点在执行前对请求与序号有法定人数交叉。

<span class="marginnote">数字实例：$f=1$ 时 $n=3f+1=4$ 个副本——容忍 1 个撒谎；prepare 证书要 $2f=2$ 个匹配加自己，commit 要 $2f+1=3$ 个。$f=2$ 时就要 7 个副本。副本数随容错数线性涨、通信还按平方涨，这正是 PBFT 适合小规模委员会而不是几千节点的原因。</span>

主坏：备份超时触发 view change，新主带着 prepared 证书证明必须带着走的请求，避免分叉执行。

<span class="marginnote">OSDI 1999 强调实用：MAC 而不是每次数字签名、垃圾回收、窗口。安全在异步下保持，活靠部分同步。</span>

## 方法

认证信道：点对点 MAC 或签名。摘要链减少载荷。状态检查点做垃圾回收，类似快照但要证书证明检查点合法。客户端自己收集 $f+1$ 个相同应答才接受——因为主可能对客户端撒谎。

<span class="marginnote">为什么客户端要 $f+1$ 份相同应答：最多 $f$ 个节点会撒谎，收到 $f+1$ 份一模一样的结果时，其中至少 1 份出自诚实节点，而诚实节点只在证书齐了之后才执行——于是这个结果必是「真相」而非伪造。1 份都不够信：那可能正是主单方面编的。</span>

```mermaid
flowchart TD
  PP["pre-prepare"] --> P["prepare 证书"]
  P --> C["commit 证书"]
  C --> EX["执行"]
  TO["超时"] --> VC["view change"]
  VC --> PP
```

与 Raft：多了撒谎，法定人数从 $f+1$ 变成 $2f+1$ 这一档（$n=3f+1$）。不能用「多数派心跳」当正确性。

## 机制

prepared 证书相交：两个不同请求不能在同一序号都达到 prepared（诚实节点不协助）。commit 让节点知道「足够多诚实者已 prepared」，于是执行后崩溃恢复仍能证明。这比崩溃 Paxos 多一轮，税来自拜占庭。

<span class="marginnote">常见误区：以为拜占庭容错只是「多备几台」。实际上代价是三件套：副本从 $2f+1$ 涨到 $3f+1$（崩溃容忍 3 台的事这里要 4 台）、每步法定人数翻倍（$f+1$ 变 $2f+1$）、外加一轮通信与认证开销——「税」买的是对面会主动撒谎这种最强敌手模型。</span>

```mermaid
flowchart TD
  CMP{"敌手模型？"}
  CMP -- "只会崩溃" --> CF["Raft / Paxos：n = 2f+1"]
  CF --> CF2["多数派 f+1 即可提交"]
  CMP -- "会撒谎" --> BF["PBFT：n = 3f+1"]
  BF --> BF2["prepare 证书 2f，commit 证书 2f+1"]
  BF2 --> BF3["客户端再收 f+1 份相同应答"]
  CF2 --> OUT["多出的副本与轮次：拜占庭税"]
  BF3 --> OUT
```

[FLP](/cs/flp-impossibility)仍在：异步下 view change 可能永远抖。随机与超时是活性。恶意主可以在被换掉前拖延迟，但不能让两个诚实节点提交不同的同一序号请求。

本课不把比特币当 PBFT。也不写 BFT-SMaRt 的全部优化。权益与委员会是后课 PoW 直觉的对照对象，不是 PBFT 本身。

## 边界

本课不证全部不变量，不引入异步 BFT 的心跳复杂度下界细节。后课默认：崩溃 RSM 用 Raft/Paxos；$f$ 撒谎用 $3f+1$ 与三阶段。下一课把共识与两阶段提交的阻塞差讲清，避免把协调者当法定人数。

交叉证书替代「信任主」。主只是性能优化，不是信任根。

## 小结

- PBFT：$n=3f+1$，pre-prepare / prepare / commit。
- 安全异步；视图更换给活性。
- 客户端收 $f+1$ 份相同应答。
- 出处：Castro and Liskov, OSDI 1999。

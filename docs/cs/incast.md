---
title: incast
date: 2026-09-08
section: cs
---

# incast

<div class="epigraph">
<p>许多发送方同时打向同一接收方的浅缓冲，瞬时队列溢出；丢的是尾，TCP 超时把好链路当成死。 </p>
<footer>—— 据 Chen, Braud 等数据中心 TCP incast 研究；Alizadeh 等对 DCTCP 动机整理</footer>
</div>

[上一课](/cs/datacenter-clos) 的多对一仍汇到叶子出口。[输出排队](/cs/output-queue-hol) 已说热点。缺口是**应用同步扇入**：MapReduce shuffle、存储多块并发读。本课不把 DCTCP 标记写完。

## 问题

RTO 最小在毫秒，数据中心 RTT 在微秒。incast：N 个流的初始窗口同时到达，叶子出端口缓冲只有几十到上百 KB，丢包 → 超时 → 吞吐塌成「好网很慢」。不是 $C$ 不够，是瞬态超订。巨帧让一包占更多缓冲。PFC 可避免丢，但暂停波及 Clos 其它流。

<span class="marginnote">数字实例：数据中心 RTT 约 50–100 微秒，TCP RTO 下限却在百毫秒量级——一次超时够正常往返几千次。32 个发送方各带 64 KB 初始窗口同时涌入，约 2 MB 要在几十微秒内挤过一个几百 KB 的出端口缓冲，尾丢几乎必然发生。</span>

不要把 incast 写成广域拥塞崩溃：时间尺度与扇入结构不同。

<span class="marginnote">文献用 barrier 同步流量复现。本课不给攻击。对策方向：更深缓冲（加重 bufferbloat）、减小 IW、AQM、DCTCP、应用限流。</span>

### 同步扇入打浅缓冲

不是平均 $C$ 不够。短流凑不够 dupACK 只剩 RTO。ECMP 不拆同一出口。加大缓冲会转向膨胀。

## 方法

画：N 发送 → 一出口浅队列 → 丢 → RTO。对照 HOL：结构内部也可丢；incast 强调应用同步。ECMP 帮不上：目的是同一叶子口。

<span class="marginnote">直觉类比：incast 像散场时一千人同时挤向一个旋转门——不是门不够宽（平均带宽够），而是所有人同一秒涌到。被挤掉票的人（丢包）要等「下一个开放时段」（RTO）才能重进，可门后明明早就没人排队了。</span>

```mermaid
flowchart TD
  N["N 个发送方"] --> Q["浅出队列"]
  Q --> DROP["尾丢"]
  DROP --> RTO["超时"]
  RTO --> THR["吞吐塌陷"]
```

## 机制

Clos 过订购比放大汇聚。RoCE 后课更怕丢，故无损。TCP 快速重传要三个 dupACK，短流可能根本凑不够——只剩超时。这把[RTO](/cs/tcp-rto) 课的下限暴露成数据中心 bug。

```mermaid
flowchart TD
  L["某流丢了一个包"] --> A["第 1 个后续包到达：1 个 dupACK"]
  A --> B["第 2 个到达：2 个 dupACK"]
  B --> C["第 3 个到达：触发快速重传"]
  C --> OK["微秒内恢复，无需超时"]
  B -.->|"短流已发完，凑不齐 3 个"| T["只能干等 RTO：百毫秒级"]
```

<span class="marginnote">PFC（优先级流控暂停）的代价直觉：它不丢包，而是让上游「全体闭嘴」；在 Clos 里这条暂停会一站站向回传染，把毫不相干流量的队列也一起憋住——堵换了个地方，还可能憋出死锁。所以「无损网络」不是免费的。</span>

应用层：限制并发、请求错开，是端到端对策，符合 Saltzer。

## 边界

本课不引入每种存储协议的扇入参数。DCTCP 是下一课。后课默认：incast = 同步多对一打浅缓冲。

加大缓冲能盖 incast，会把延迟交给后课缓冲膨胀。

下一课[DCTCP](/cs/dctcp)。

## 小结

- 同步扇入溢出浅队列，RTO 主导。
- ECMP 不拆同一出口。
- 对策在 AQM/DCTCP/应用限流，不只加带宽。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：数据中心 incast 文献；DCTCP 动机。

---
title: 延迟型拥塞 Vegas / Swift
date: 2026-09-08
section: cs
---

# 延迟型拥塞 Vegas / Swift

<div class="epigraph">
<p>用 RTT 超过基线的量估计队列，提前减速；Vegas 在公网脆弱，Swift 把同一思想收进数据中心目标延迟。</p>
<footer>—— 据 Brakmo and Peterson, TCP Vegas, IEEE JSAC 1995；Kumar et al., Swift, SIGCOMM 2020 整理</footer>
</div>

[BBR](/cs/bbr-v2-v3) 用 minRTT。[ECN](/cs/ecn) 用标记。[上一课](/cs/bbr-v2-v3) 不讲 Vegas 控制律。缺口是**延迟作为主信号**：Vegas 窗口规则、反向流量噪声、Swift 的 target delay。本课不把无线误码写完。

## 问题

AIMD 等丢。Vegas：比较期望吞吐与实际，RTT 升则减窗口。公网路由变化、ACK 压缩、反向拥塞让基线漂，Vegas 常输给 Reno。数据中心 RTT 干净，Swift（及 TIMELY）用 delay 梯度或目标延迟调速，服务尾延迟，可与 ECN 双信号。与 CoDel：一个在端测 RTT，一个在箱测驻留。

<span class="marginnote">Vegas 在公网输给 Reno 的根源是基线会漂：路由一换，新的最小 RTT 变大，Vegas 把多出来的传播延迟误判成「队列积压」，于是主动减窗；而 Reno 只认丢包，不吃这个亏。同一个瓶颈上两家共存时，Vegas 一再让路，带宽就被 Reno 抢走——不是延迟信号没用，是它怕噪声。</span>

不要把延迟 CC 写成「不需要 AQM」的普适解。

<span class="marginnote">Vegas 1995。Swift 2020 面向 DC。本课不调每个 $\alpha,\beta$。</span>

### 延迟信号怕噪声

Vegas 在公网基线漂。Swift 把目标延迟放进数据中心。卫星传播淹没队列信号。可与 ECN 组合。

## 方法

对照：丢包 / ECN / 延迟。画：测 RTT → 相对 base → 调窗。卫星：base 已是 600 ms，队列信号淹没在传播里，延迟 CC 难。

```mermaid
flowchart TD
  BASE["基线 RTT"] --> DEL["超额延迟"]
  DEL --> DEC["减窗口"]
  NOISE["路由/反向噪声"] --> FAIL["公网 Vegas 失败"]
```

## 机制

DCTCP 用 CE 比例，Swift 用 delay，都可浅队列。WFQ 隔离后延迟信号更干净。MPTCP 子流 RTT 不同，延迟信号要分路径。QUIC 可在用户态实现 Vegas 变体。

公平：延迟 CC 对短 RTT 仍可能敏，但机制与 AIMD 的 $1/\mathrm{RTT}^2$ 不同。

Vegas 的控制律可以把「测到的延迟」一步步换成「加减窗的决策」：

```mermaid
flowchart TD
  M["每个 RTT 测当前 RTT 与吞吐"] --> D["Diff = cwnd × (1 − 基线RTT/当前RTT)"]
  D -->|低于 α：队列空| UP["窗口加 1"]
  D -->|介于 α 与 β：轻度积压| HOLD["窗口不动"]
  D -->|高于 β：积压过多| DOWN["窗口减 1"]
```

<span class="marginnote">数字实例：设基线 RTT 为 10 ms、当前 RTT 为 12 ms、cwnd 为 10 个包，则 Diff = 10 × (1 − 10/12) ≈ 1.7，意思是「瓶颈队列里大约积压了 1.7 个包」。若取 α=1、β=3，就有 1 ≤ 1.7 ≤ 3，落在「轻度积压」区间，这一轮窗口维持不动。</span>

<span class="marginnote">直觉类比：Swift 的目标延迟像给交换机队列画了一条「水位线」。水位（排队延迟）漫过线就拧小阀门（降速），低于线就稍微拧大。数据中心里所有流守同一条水位线，排队的苗头一露头就被压回去，深队列根本长不起来——这正是它能服务尾延迟的原因。</span>

## 边界

本课不引入 COPA 的全部效用函数。无线上的 TCP 是下一课。后课默认：延迟信号在可控 RTT 环境有用；公网噪声大。

把 Vegas 当默认公网 CC 会吃亏。

下一课[无线上的 TCP](/cs/tcp-wireless)。

## 小结

- Vegas 用超额 RTT 调窗，公网不稳。
- Swift 等把延迟 CC 放进数据中心。
- 与 ECN/AQM 可组合，不互相取消。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Brakmo–Peterson, 1995；Kumar et al., 2020。

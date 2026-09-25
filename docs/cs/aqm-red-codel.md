---
title: AQM：RED 与 CoDel
date: 2026-09-08
section: cs
---

# AQM：RED 与 CoDel

<div class="epigraph">
<p>尾丢把队列撑满才说话；主动队列管理提前丢或标记，给 AIMD 信号，并压住延迟。</p>
<footer>—— 据 Floyd and Jacobson, RED, IEEE/ACM ToN 1993；Nichols and Jacobson, CoDel, ACM Queue 2012；RFC 8289 整理</footer>
</div>

[RTT 不公平](/cs/cc-fairness-rtt) 与 [incast](/cs/incast) 都怪队列。[上一课](/cs/cc-fairness-rtt) 留下调度器与 AQM 两条路。缺口是 **RED/CoDel**：按平均队列或驻留时间丢/标。本课不把 ECN 协商写完。

## 问题

尾丢 = 满了才丢，信号晚、同步多流一起 MD。RED：平均队列在 min/max 间概率丢。难调，平均队列不是延迟。CoDel：测包的驻留，超过 target 一段时间则丢，参数按时间不是按包数，适配带宽变化。PIE 同类。数据中心浅阈标记更近 DCTCP，公网盒常 CoDel/FQ-CoDel。

不要把 AQM 写成提高 $C$：它管的是延迟与信号时机。

<span class="marginnote">直觉类比：尾丢像餐厅直到撞门才赶人——门口一满就把在场所有人一起赶走（全局同步），下一波又同时涌回。AQM 是提前小规模劝退几位：队伍始终流动，没人需要整场推倒重来，这正是 AIMD 想要的早期信号。</span>

<span class="marginnote">RFC 7567 AQM 建议。RFC 8289 CoDel。本课不把 RED 权值当考试。</span>

### 早期信号换浅延迟

不增加 $C$。RED 按深度难调；CoDel 按驻留。FQ-CoDel 顺带隔离流。极浅 cut-through 无队列可管。

<span class="marginnote">术语翻译：驻留时间就是一张快递单在网点积压的时长——入队到出队之间隔了多久。CoDel 不数仓库堆了多少件（深度），只看件压了多久：超过 target（默认 5 ms）还降不下来才动手。参数以时间为单位，带宽换了也不用重调，这是它比 RED 好养的地方。</span>

## 方法

对照：尾丢 / RED / CoDel。画：入队 → 测延迟 → 超阈则丢或 CE。与 PFC：一个丢/标，一个暂停，对象都是队列，哲学相反。

```mermaid
flowchart TD
  Q["队列"] --> TAIL["尾丢: 满才丢"]
  Q --> RED["RED: 平均深度"]
  Q --> CD["CoDel: 驻留时间"]
```

## 机制

FQ-CoDel 每流队列再 CoDel，顺带修 RTT 不公平与缓冲膨胀。交换机 ASIC 未必能每流，数据中心用简单阈。卫星：CoDel target 要大于固有 RTT 的排队部分，不能把传播时延当膨胀。P4 可实现自定义 AQM。

同步：随机早期丢减少全局同步，几何上仍 AIMD。

<span class="marginnote">常见误区：初学者容易以为队列越长吞吐越高。实际上队列只在瓶颈短暂空闲时兜缓冲，长过头的队列只是把包泡在延迟里（缓冲膨胀）：10 Mbps 链路上 1 MB 缓冲能多压约 0.8 秒延迟，吞吐却一点不涨。AQM 治的正是这个，而不是提速。</span>

```mermaid
flowchart TD
  DQ["出队一个包"] --> SOJ["现在时间减入队时间 = 驻留"]
  SOJ --> CMP{"驻留 ≤ target?"}
  CMP -->|是| KEEP["正常发送"]
  CMP -->|否| W{"超 target 已持续超过 interval?"}
  W -->|否| KEEP
  W -->|是| DROP["丢包或打 CE 标记"]
  DROP --> NEXT["进入丢包状态, 按递减间隔丢, 直到驻留降回 target 下"]
```

## 边界

本课不引入 L4S 双队列全文。ECN 是下一课。后课默认：AQM 用早期信号换浅延迟；参数以时间为宜。

在极浅缓冲的 cut-through 盒上 AQM 无队列可管。

下一课[ECN](/cs/ecn)。

## 小结

- 尾丢信号晚且易同步。
- RED 按深度，CoDel 按驻留。
- 不增加容量，改善延迟与反馈。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Floyd–Jacobson, 1993；Nichols–Jacobson；RFC 8289。

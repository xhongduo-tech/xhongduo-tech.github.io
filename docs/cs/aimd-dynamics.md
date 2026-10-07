---
title: AIMD 与 Chiu–Jain
date: 2026-09-08
section: cs
---

# AIMD 与 Chiu–Jain

<div class="epigraph">
<p>加性增、乘性减使共享瓶颈的流在公平与效率之间收敛；Chiu–Jain 用二维相位图说明为何 MIMD 会漂。</p>
<footer>—— 据 Chiu and Jain, Analysis of the Increase and Decrease Algorithms, Computer Networks 1989；Jacobson, 1988 整理</footer>
</div>

主干[拥塞控制](/cs/tcp-congestion) 已用 AIMD。[上一课](/cs/roce-lossless) 结束转发面。缺口是**为什么是 AI 而不是 MI**：两用户相位图、公平线与效率线。本课不把快重传状态机写完。

## 问题

两流共享容量 $C$。效率线 $x_1+x_2=C$，公平线 $x_1=x_2$。加性增：平行于公平线走向效率；乘性减：沿过原点射线退回。MIMD 增沿射线走，到不了公平。AIAD 在效率线附近振荡不收敛。数据中心 DCTCP 仍是乘性，只是减得更温和。广域卫星 BDP 大使 AIMD 探到 $C$ 的时间变长，不改几何。

<span class="marginnote">直觉类比：把相位图想成两人分一块大小为 $C$ 的蛋糕——效率线是「别浪费」，公平线是「要平分」。加性增是两人每次各多要一块，差距不变、整体逼近蛋糕边；乘性减是两人把手里的量同时砍半，比例不变、落点回到过原点的射线上，反而更靠近平分线。一增一减，锯齿越锯越靠交点。</span>

不要把 Jain 公平指数写成必须每课计算，这里只要相位图直觉。

<span class="marginnote">Chiu–Jain 1989。同步假设理想化；延迟 ACK、不同 RTT 后课会破公平。本课钉同 RTT 同步。</span>

### 同 RTT 才有那张相位图

AI 走向效率，MD 走向公平。MIMD 漂。不同 RTT 与 ACK 压缩会偏离交点，后课再破。

## 方法

画相位图：效率、公平、AIMD 轨迹。对照 PFC：链路暂停不是 AIMD，无公平定理。对照 WFQ：调度器显式公平，AIMD 是端到端隐式。

<span class="marginnote">术语翻译：MIMD 就是「乘性增、乘性减」——每轮按固定比例放大或缩小速率（如都乘 2 或乘 0.5）。听起来与 AIMD 相似，但比例操作不改变两流的差距比例，落点始终在过原点的同一条射线上，永远碰不到公平线 $x_1=x_2$。</span>

```mermaid
flowchart TD
  AI["加性增"] --> EFF["靠近效率线"]
  MD["乘性减"] --> FAIR["靠近公平线"]
  AIMD["交替"] --> CONV["收敛交点"]
```

## 机制

Reno 每 RTT 约加一 MSS，丢则减半，是 AIMD 实例。CUBIC 用时间立方，仍有乘性事件。ECN 把「减」从丢包提前，几何类似。多瓶颈与不同 RTT 使交点偏离，后课 RTT 不公平。

落到一条真实 TCP 流上，相位图的锯齿就是下面这个不断重复的周期：

```mermaid
flowchart TD
  S["cwnd 较小，慢涨阶段"] --> G["每 RTT 加 1 MSS（AI：平行公平线上移）"]
  G --> Q{"顶到效率线并收到丢包/ECN 信号？"}
  Q -->|"还没到"| G
  Q -->|"到了"| H["cwnd 减半（MD：沿过原点射线退回）"]
  H --> S
```

<span class="marginnote">数字实例：cwnd 为约 29 KB（20 个 1460 字节的 MSS）时丢包，减半后只剩 10 个 MSS；之后每 RTT 只加回 1 个，爬回 20 个要 10 个 RTT。「涨得慢、砍得快」正是网速测试里锯齿波形的来源。</span>

与交换机 VOQ 匹配对照：一个在端，一个在跳；都在分配 $C$。

## 边界

本课不证一般 $n$ 流的全部 Lyapunov。快速重传与恢复是下一课。后课默认：同 RTT 下 AIMD 收敛到公平有效；其它增减组合不。

实际 ACK 压缩会让「每 RTT 加一」变成突发，几何仍近似。

下一课[快速重传与恢复](/cs/fast-retransmit-recovery)。

## 小结

- AIMD 在效率–公平图上收敛。
- MIMD/AIAD 不提供同样保证。
- TCP 减半是 MD；DCTCP 是更小的 MD。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Chiu–Jain, 1989；Jacobson, 1988。

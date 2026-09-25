---
title: BBR 对照
date: 2026-09-08
section: cs
---

# BBR 对照

<div class="epigraph">
<p>把速率对准估计的瓶颈带宽，把飞行数据对准最小 RTT 下的 BDP；拥塞信号从「队列溢出」转到「模型」。 </p>
<footer>—— Cardwell, Cheng, Gunn, Jacobson and Yeganeh, BBR: Congestion-Based Congestion Control, ACM Queue 2016</footer>
</div>

[上一课](/cs/reno-cubic)在丢包驱动下比较了 Reno 与 Cubic。本课不重画立方曲线。缺口是：浅缓冲或无线误码时，丢包不等于瓶颈满；深缓冲时丢包来得太晚，队列已经很大。BBR 用交付率估带宽、用最小 RTT 估延迟，对照而非替换 TCP 可靠性。

## 问题

AIMD 把溢出当探针。缓冲很小则永远探溢；缓冲很大则延迟主导。BBR 的缺口是控制量改成两个估计：瓶颈带宽 BtlBw、往返传播 RTprop，发送速率与 `cwnd` 按 BDP 操作，并周期性探测是否变高/变低。本课不把版本号演进（BBR v2/v3）写成标准战争。

<span class="marginnote">BBR 仍跑在 TCP（或 QUIC）上，序号与重传不变。它改变 pacing 与窗口，不改变 RFC 793 的字节流合同。</span>

<span class="marginnote">直觉类比：AIMD 像「水漫出来才知道关小」，BBR 像调水龙头——稍微拧大一点，看水流是否真的变快：变快说明管子还能承受，不变快就把多拧的收回来。它始终盯着「管子多粗」和「水走完全程要多久」这两个数。</span>

## 方法

探测带宽：以略超估计的速率发，看交付率是否上升。探测 RTT：短暂降速看最小 RTT 是否更小。稳态：pacing 接近 BtlBw，飞行数据略高于 BDP 以保持管道满。与 Cubic 对照：后者填缓冲直到丢；BBR 声称把队列压在小幅度。共存时可能不公平，这是对照的边界，不是本课调参。

```mermaid
flowchart TD
  ACK["交付样本"] --> BW["估计瓶颈带宽"]
  ACK --> MIN["估计最小 RTT"]
  BW --> PACE["按模型 pacing"]
  MIN --> PACE
```

## 机制

模型控制把「共享队列的博弈」从丢包博弈改成估计博弈。估计错（路径变化、聚合链路）会过冲或欠载。端到端正确性仍靠校验与重传；BBR 不提供新的 CIA 性质。后课 QUIC 可以换传输实现，仍可跑同类控制。

```mermaid
flowchart TD
  SS["稳态: pacing 约等于 BtlBw"] --> PB["周期试探: 1.25 倍速发一阵"]
  PB --> Q{"交付率跟着上升?"}
  Q -->|"是"| HI["带宽真的变高, 巡航到新值"]
  Q -->|"没有"| DL["降到 0.75 倍排水, 收回多发的"]
  DL --> RT["探测 RTT: 短暂降速找最小值"]
  RT --> SS
  HI --> SS
```

<span class="marginnote">数字实例：链路 100 Mb/s、RTT 40 ms，则 BDP $= 10^8 \times 0.04 / 8 = 0.5$ MB。BBR 巡航时把飞行数据维持在略高于 0.5 MB（乘个保管道满的小系数）；Reno 若靠丢包探窗口，可能先把几 MB 的缓冲灌满才回头，队列与延迟都大了。</span>

## 边界

本课不引入 XCP 一类路由器协助协议当主干，不把内核开关当教材命令。应用层名字如何映射到 IP，下一课 DNS。

<span class="marginnote">常见误区：初学者容易以为 BBR 是一套新传输协议。它仍跑在普通 TCP 或 QUIC 之上，序号、确认、重传一字不改；换的只是「什么时候发多少」——pacing 节奏与窗口大小，字节流合同还是原来那份。</span>

后课默认：拥塞控制可以不把丢包当唯一信号。人可读的主机名是另一缺口。

## 小结

- BBR 用带宽与最小 RTT 建模，对照 Reno/Cubic 的丢包 AIMD。
- 可靠性合同不变。
- 名字解析下一课 DNS。
- 出处：Cardwell 等, *ACM Queue* 2016；对照 RFC 5681；Kurose and Ross。

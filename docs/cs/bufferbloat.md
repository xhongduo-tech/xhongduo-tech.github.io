---
title: 缓冲膨胀
date: 2026-09-08
section: cs
---

# 缓冲膨胀

<div class="epigraph">
<p>过大的瓶颈缓冲让 AIMD 在已满 $C$ 之后仍堆延迟；吞吐看起来很好，交互 RTT 变成秒级。</p>
<footer>—— 据 Gettys and Nichols, Bufferbloat, ACM Queue 2011；RFC 7567 整理</footer>
</div>

[BDP](/cs/bandwidth-delay-product) 说缓冲约一 BDP。[AQM](/cs/aqm-red-codel) 是药。[上一课](/cs/ecn) 给标记。缺口是**命名膨胀**：家用路由、LTE 基站、Cable 调制解调器里的巨缓冲。本课不把 WFQ 写完。

## 问题

网卡和驱动按「丢包坏」堆兆字节。TCP 填满缓冲才丢，稳态延迟 = 缓冲/$C$，可远超传播。测速工具显示线速，ssh 却卡——因为共享同一队列。卫星固有延迟不是膨胀；膨胀是可消灭的排队。FQ-CoDel 把大象流隔开，老鼠流不排队。

不要把所有高 RTT 都叫膨胀：先减传播与无线调度。

<span class="marginnote">数字实例：100 Mbit/s 链路 $C \approx 12.5$ MB/s，路由器塞了 4 MB 缓冲。TCP 灌满后排队延迟 = $4\,\text{MB} \div 12.5\,\text{MB/s} = 320$ ms——每个包都得排这么长的队。若缓冲只有一个 BDP（RTT 20 ms 时约 250 KB），排队只添 20 ms，交互不受影响。</span>

<span class="marginnote">Gettys 普及该词。本课不点名某调制解调器固件。</span>

### 吞吐满而交互死

排队延迟 = 缓冲/$C$。灌流前后 ping 对照。卫星固有传播不是膨胀。过浅则 incast。

## 方法

对照：浅缓冲丢包 vs 深缓冲延迟。画：cwnd 超过 BDP 的部分 = 队列。测：空闲 ping vs 灌流 ping。incast 是过浅；膨胀是过深；数据中心与接入网方向相反。

```mermaid
flowchart TD
  FILL["TCP 填缓冲"] --> DELAY["排队延迟"]
  DELAY --> SIG["很晚才丢/标"]
  AQM["AQM/FQ"] --> SHAL["压回浅队列"]
```

## 机制

窗口缩放「成功」后更容易堆满巨型缓冲。PFC 无损也会堆延迟，只是不丢。5G 切片可给 URLLC 独立浅队列，避免被 eMBB 膨胀。测量后课 iperf 要同时看 RTT。

与 HOL：膨胀是单队列深度；HOL 是结构。

下图回答一个具体问题：同一条链路、同一个灌流，深缓冲与浅缓冲加 AQM 的表现差在哪。

```mermaid
flowchart TD
  subgraph DEEP["深缓冲：4 MB 队列"]
    D1["队列灌满"] --> D2["排队延迟 320 ms"]
    D2 --> D3["吞吐满：测速好看"]
    D2 --> D4["ping/ssh 延迟数百 ms"]
  end
  subgraph SHALLOW["浅缓冲 + AQM"]
    S1["BDP 级队列，早丢/早标记"] --> S2["排队延迟约 20 ms"]
    S2 --> S3["吞吐略降，交互流畅"]
  end
```

<span class="marginnote">直觉类比：深缓冲像收费站修了超长引道——拥塞时车全堵在引道里，主路「吞吐」依旧满负荷，可新来的车要排三百米才到闸口。浅缓冲加 AQM 像在引道口放信号灯提前拦车，队伍永远短，每辆车都快点过闸。</span>

## 边界

本课不引入 BQL 网卡字节队列的全部。WFQ 是下一课。后课默认：过大缓冲把延迟藏进吞吐数字；AQM 或每流排队是解。

给接入网「更大缓冲抗抖动」要有上界，否则交互死。

下一课[WFQ](/cs/wfq-fair-queue)。

## 小结

- 膨胀：队列延迟主导 RTT，吞吐仍满。
- 测灌流前后的 ping。
- AQM/FQ 针对它；incast 是反面。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Gettys–Nichols, 2011；RFC 7567。

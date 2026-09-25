---
title: BBRv2 / v3
date: 2026-09-08
section: cs
---

# BBRv2 / v3

<div class="epigraph">
<p>BBR 用带宽与 RTT 估计管道，不靠丢包减半；v2/v3 修过估、公平与 ECN 反应，仍在模型与测量之间摇摆。</p>
<footer>—— 据 Cardwell et al., BBR, ACM Queue 2016；BBRv2/v3 公开设计说明整理</footer>
</div>

主干[BBR 对照](/cs/bbr) 已给 v1 直觉。[缓冲膨胀](/cs/bufferbloat) 是 v1 在深缓冲上的痛。[上一课](/cs/quic-migration) 不改控制律。缺口是 **v2/v3 补什么**：带宽过估、与 CUBIC 共存、ECN。本课不把 Vegas 写完。

## 问题

v1 ProbeBw 可把队列推高，深缓冲上仍膨胀；对 CUBIC 不公平。v2：更保守的带宽头、对丢失/ECN 作出反应、量化公平。v3 继续调探测与启动。模型：maxBW × minRTT ≈ BDP，窗口绕此值。误码链路把丢当拥塞仍可能，但比 AIMD 少抽空——卫星动机还在。

不要把版本号写成已在 RFC 终稿冻结；课钉对象与问题。

<span class="marginnote">v1：Cardwell 2016。后续版本以公开幻灯与代码为准，不把未冻结的草案号当标准。</span>

<span class="marginnote">直觉类比：v1 像只看导航的司机——自己测路况（带宽与 RTT）决定油门，不因为远处别人追尾（丢包）就急刹；v2/v3 则学会了听到喇叭（ECN 标记、丢包上升）也松一点油门，免得一直压着同路的车。</span>

### 模型不是 AIMD 换皮

v1 在深缓冲仍可膨胀。v2/v3 更听丢与 ECN，并修公平。版本以公开设计说明与代码为准，课钉对象不钉某年冻结稿。

## 方法

对照 AIMD 相位图 vs BBR 管道估计。画：Probe → Drain → Cruise。与 DCTCP：都利用 ECN，一个比例减，一个进模型。

<span class="marginnote">术语翻译：ECN（Explicit Congestion Notification，显式拥塞通知）就是路由器在包头上盖一枚「快堵了」的章，接收方把这个记号回传给发送端——让它在丢包发生之前就知道减速，比「拿丢失当烟雾报警」早一步。</span>

```mermaid
flowchart TD
  BW["估计带宽"] --> BDP["乘 minRTT"]
  BDP --> CAP["巡航窗口"]
  SIG["丢/ECN"] --> V2["v2 更听信号"]
```

## 机制

QUIC 与 Linux TCP 都有实现。多路径下每条路径一套估计。AQM 浅队列让 minRTT 更真。P4 不实现 BBR，BBR 在端。RoCE 用 DCQCN 不是 BBR。

过估：把突发当可用带宽，会伤同队列邻居——v2 要压这点。

```mermaid
flowchart TD
  subgraph V1["v1 的问题路径"]
    A["突发被当成可用带宽"] --> B["带宽过估, 窗口偏大"]
    B --> C["队列被推高, 同队邻居受压"]
  end
  subgraph V2V3["v2 / v3 的补丁"]
    D["更保守的带宽上限"] --> E["对丢包与 ECN 也要反应"]
    E --> F["与 CUBIC 做量化公平"]
  end
  C --> D
```

<span class="marginnote">数字实例：若瓶颈缓冲深达 4 个 BDP，v1 的周期探测可能把队列一路推满：12.5 MB 的 BDP 配上 50 MB 缓冲，排队包要多等几个 RTT，交互流量的延迟从 100 ms 涨到几百 ms——这就是 v1 在深缓冲上仍会「膨胀」的量感，也是 v2 收着走的原因。</span>

## 边界

本课不引入 PCC 等学习型 CC 全文。延迟型 Vegas/Swift 是下一课。后课默认：BBR 族是基于模型的 CC；v2/v3 针对公平与膨胀补丁。

与不可 ECN 的公网 CUBIC 混跑，结果仍依赖缓冲。

下一课[延迟型拥塞 Vegas / Swift](/cs/delay-based-cc)。

## 小结

- BBR 用 BW 与 RTT 估 BDP，而非纯 AIMD。
- v2/v3 加强对丢/ECN 与公平的反应。
- 深缓冲与误码仍是模型边界。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Cardwell et al., 2016；BBRv2/v3 说明。

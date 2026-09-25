---
title: ECN
date: 2026-09-08
section: cs
---

# ECN

<div class="epigraph">
<p>路由器把 IP 头 CE 位置上，收端在 ACK 里回 ECE，发送方当拥塞处理而不必丢包；要两端与路径都同意。</p>
<footer>—— 据 RFC 3168；RFC 8311；DCTCP 课对照整理</footer>
</div>

[AQM](/cs/aqm-red-codel) 可以丢也可以标。[DCTCP](/cs/dctcp) 已用 CE。[上一课](/cs/aqm-red-codel) 留下标记通道。缺口是 **RFC 3168 协商与语义**：ECT、CE、ECE、CWR。本课不把 bufferbloat 写完。

## 问题

丢包既是信号也是损伤。ECN：拥塞时标 CE，TCP 用 ECE 通知，发端减窗并回 CWR。握手用 ECN-Echo 与 CWR 位协商能力。中间盒把 CE 清零或丢 ECT 包，则静默失败。L4S 后用不同码点，本课点名。与 PFC：ECN 端到端（路径上最挤跳），PFC 一跳。

<span class="marginnote">直觉类比：丢包像收费站用「扔掉你的车」告诉你前面堵了——信号有效但代价惨重；ECN 像收费员在车窗上贴一张「前方拥堵请减速」的条子，消息照样送到，车也没少一台。对短连接密集的数据中心流量，省下的重传相当可观。</span>

不要把 CE 当成应用错误码。

<span class="marginnote">常见误区：CE（Congestion Experienced）不是给应用看的错误码，而是给 TCP 栈看的拥塞信号，应用完全不感知；它也不表示包损坏——包照常送达，只是顺手捎了一句「路上有点挤」。把它当应用层报错去处理，方向就错了。</span>

<span class="marginnote">RFC 3168。Classic ECN 一次拥塞一窗口事件；DCTCP 用比例。本课钉经典语义再对照。</span>

### 拥塞与丢包分离

CE 沿路径最挤跳；ECE/CWR 走 TCP。中间清洗则静默失败。Classic 一次一事件，DCTCP 用比例。

## 方法

画：ECT 发送 → AQM 标 CE → ECE → MD → CWR。对照无 ECN：丢 → 快重传。QUIC 也有 ECN 计数，后课。

```mermaid
flowchart TD
  ECT["ECT 包"] --> CE["路由器标 CE"]
  CE --> ECE["ACK 带 ECE"]
  ECE --> MD["发送方减窗"]
  MD --> CWR["通知已响应"]
```

## 机制

ROCE/DCQCN 用 ECN 调速率，不是 TCP ECE。无线链路层丢包不是 CE，后课 TCP 无线会混。PMTUD 与 ECN 正交。安全：伪造 CE 可降速，需路径可信或统计。

部署：要主机栈、AQMs、不清洗的中间盒同时在。

一条路径凭什么算「支持 ECN」？任何一环不配合会退到哪里？

```mermaid
flowchart TD
  SYN["握手时协商 ECN 能力"] --> BOTH{"两端都同意?"}
  BOTH -->|"否"| OFF["不用 ECN：靠丢包报拥塞"]
  BOTH -->|"是"| RUN["数据包标 ECT 上路"]
  RUN --> MB{"中间盒清洗 CE / 丢 ECT?"}
  MB -->|"不清洗"| GOOD["拥塞走标记：不丢包就降速"]
  MB -->|"清洗"| FAIL["静默失败：标记到不了发端"]
  FAIL --> OFF
```


## 边界

本课不引入 AccECN 的全部。缓冲膨胀是下一课。后课默认：ECN 把拥塞与丢包分离；协商失败则退回丢包信号。

只在交换机开标记、主机不开，等于没开。

<span class="marginnote">为什么重要：ECN 是端到端合同，任何一环缺席整条失效——中间盒清掉 CE 位，发端永远等不到 ECE，只能退回丢包信号。这也是它提出二十多年仍未全面普及的主因：路径上任何一台不认识它的老设备，都能无声无息地把它废掉。</span>

下一课[缓冲膨胀](/cs/bufferbloat)。

## 小结

- CE 是路径拥塞信号，ECE/CWR 走 TCP。
- 需端到端支持，中间不可清洗。
- DCTCP 解释不同，码点可同。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 3168；RFC 8311。

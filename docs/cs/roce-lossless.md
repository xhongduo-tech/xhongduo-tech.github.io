---
title: RoCE 与无损网络
date: 2026-09-08
section: cs
---

# RoCE 与无损网络

<div class="epigraph">
<p>RDMA 把远程内存操作放到网卡，绕过内核；RoCEv2 用 UDP/IP 承载，依赖 PFC 等把以太网做成「不丢」的假象。</p>
<footer>—— 据 InfiniBand 架构对照；RoCEv2 规范；IEEE 802.1Qbb 实践整理</footer>
</div>

[PFC](/cs/pause-pfc) 与 [DCTCP](/cs/dctcp) 已给反压与浅队列。[上一课](/cs/dctcp) 仍是 TCP。缺口是 **RoCE**：网卡 QP、无损以太网、与 TCP 栈对照。本课结束转发面课序。

## 问题

存储与 GPU 集合通信要微秒级、低 CPU。[套接字](/cs/socket-api) 系统调用与拷贝太重。RDMA：RNIC 直接读写对端内存。InfiniBand 原生有基于信用的链路；RoCEv2 把 IB 运输装进 UDP，走 IP Clos——但 UDP 不重传，丢包让 QP 停顿昂贵，于是要 PFC 无损 + 可选 DCQCN。这是把端到端可靠性下放到链路与网卡，Saltzer 警告的那种优化：一旦无损失败，没有 TCP 兜底。

不要把 RoCE 写成「更快的 TCP」：语义是 RDMA，不是字节流。

<span class="marginnote">RoCEv2 目的端口 4791。PFC 死锁与暂停风暴是已知运营风险。本课不写 verbs API 全集。</span>

### RDMA 怕丢

RoCEv2 走 UDP/IP，靠 PFC/DCQCN 维持无损假象。失败恢复昂贵。语义不是字节流。留在数据中心。

## 方法

对照：TCP 拷贝 vs RNIC 零拷贝。画：应用注册内存 → QP → RoCEv2 包 → PFC 队列。VXLAN 上跑 RoCE 要内层无损与外层 DSCP 映射，困难。

<span class="marginnote">数字实例：MTU 从 1500 开到 9000 字节的巨帧，传同样 1 GB 的消息，包数从约 70 万降到约 12 万。包少了，包头开销、中断次数和收端软件处理都按比例缩水——这正是数据中心 RDMA 默认开巨帧的原因。</span>

```mermaid
flowchart TD
  APP["注册内存"] --> RNIC["RNIC QP"]
  RNIC --> UDP["RoCEv2 over IP"]
  UDP --> PFC["优先级暂停"]
  DROP["仍丢则 QP 昂贵恢复"]
```

## 机制

incast 对 RoCE 更致命，故 PFC+ECN。HOL：PFC 整类暂停会冻无关 QP。ECMP 乱序对 RDMA 不友好，常要保序或网卡重排。MTU 用巨帧摊头。与 5G URLLC 对照：都是用资源换尾延迟，一层在机房以太网，一层在空口。

多台发送方同时压向一个收端时，ECN 先降速、PFC 兜底防丢：

```mermaid
flowchart TD
  M["多发送方同时压向一台收端"] --> Q["收端交换机队列堆积"]
  Q --> T{"水位越过 ECN 阈值？"}
  T -->|"未越"| OK["正常转发"]
  T -->|"已越"| ECN["交换机在包上打 ECN 标记"]
  ECN --> CN["收端回传拥塞通知"]
  CN --> RATE["发送方降低发送速率"]
  Q -->|"继续逼近线速水位"| PFC["PFC 对该优先级下发 PAUSE"]
  PFC --> HOLD["上游停发该类帧，保证不丢"]
```

<span class="marginnote">术语翻译：ECN 就是「在包上贴一张『前面堵了』的便签、让发送方提前减速」的手段；PFC 则是「直接喊停一类帧」的硬反压。前者柔和但慢半拍，后者保丢零却会连累同优先级的无关流量。</span>

<span class="marginnote">常见误区：初学者容易以为「无损网络」就是整个网络不丢任何帧。实际 PFC 只对配置了的优先级类生效——被暂停的那一类会冻结整条队列，无关 QP 跟着遭殃（队头阻塞）；而没配 PFC 的流量照样丢。</span>

## 边界

本课不引入 iWARP 的全部。AIMD 与 Chiu–Jain 是下一单元第一课。后课默认：RoCE 依赖无损与网卡卸载；失败模式与 TCP 不同。

互联网广域不假设 PFC，RoCE 留在数据中心。

下一课[AIMD 与 Chiu–Jain](/cs/aimd-dynamics)。

## 小结

- RoCEv2：RDMA over UDP/IP，怕丢。
- PFC/DCQCN 维持无损假象，有暂停风险。
- 不替代公网 TCP。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RoCEv2；802.1Qbb；InfiniBand 对照。

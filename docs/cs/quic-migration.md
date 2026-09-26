---
title: QUIC 连接迁移
date: 2026-09-08
section: cs
---

# QUIC 连接迁移

<div class="epigraph">
<p>连接 ID 与地址解耦：路径变了用 PATH_CHALLENGE 验证新路，字节流不断；NAT 重绑与 Wi‑Fi/蜂窝切换是同一机制。</p>
<footer>—— 据 RFC 9000 连接迁移；RFC 9002 恢复对照整理</footer>
</div>

[上一课](/cs/quic-streams-0rtt) 的连接还钉在四元组上叙事。TCP 换 IP 必断。[MPTCP](/cs/mptcp) 用子流。缺口是 **QUIC 迁移**：CID、路径验证、连接 ID 隐私。本课不把 BBRv2 写完。

## 问题

手机从 Wi‑Fi 到 LTE，TCP 四元组变，状态机作废。QUIC：包头 CID 让服务器认出连接，再验证新路径防劫持。旧路径可并存短暂。与蜂窝核心锚点对照：那是网络侧保 IP；迁移是端侧换 IP 仍保运输。Anonymizing CID 轮换防链路追踪。

不要把迁移写成 0-RTT 的一部分：0-RTT 是早数据，迁移是路径。

<span class="marginnote">RFC 9000 第 9 节。禁用迁移的中间盒存在。本课不写负载均衡怎么粘 CID，后课 Maglev 会沾边。</span>

### CID 解耦地址

新路径要挑战验证。拥塞状态不可盲目继承。服务器须能按 CID 路由到同一实例。与 0-RTT 不是一件事。

## 方法

画：旧路径 → 新地址 → PATH_CHALLENGE/RESPONSE → 切主路径。对照 MPTCP ADD_ADDR。卫星切换网关类似。

```mermaid
flowchart TD
  CID["连接 ID"] --> REC["对端认出"]
  NEW["新四元组"] --> CH["路径验证"]
  CH --> SW["切换发送路径"]
```

## 机制

拥塞状态是否继承是实现：新路径 BDP 不同，盲目继承会炸。保活用 PING 帧，短于 TCP keepalive。SYN cookies 无对应，Retry 与 CID 由服务器发。ECMP 哈希因四元组变而换路，本就可能，迁移只是显式。

安全：未验证的新路径不能立刻收非探测数据，防迁移劫持。

```mermaid
flowchart TD
  A["手机在 Wi-Fi 上发数据"] --> B["切换到 LTE，四元组变了"]
  B --> C["新包仍带同一个 CID"]
  C --> D{"服务器认出连接?"}
  D -->|"是"| E["发 PATH_CHALLENGE 探测帧"]
  E --> F["客户端回 PATH_RESPONSE"]
  F --> G{"响应内容对得上?"}
  G -->|"是"| H["新路径生效，切换发送"]
  G -->|"否"| I["维持旧路径，丢弃迁移"]
```

<span class="marginnote">PATH_CHALLENGE 就是服务器的「考题」：发出一串随机字节，看新地址上的客户端能不能原样答回来。答得回来，说明客户端真的在那个新地址上活着，而不是有人伪造源地址冒充。</span>

<span class="marginnote">数字实例：PATH_CHALLENGE 携带 8 至 16 字节随机数。若中间人想在 LTE 上劫持连接，它猜中这串随机数再回包的概率约为 $2^{-64}$ 量级——比中彩票难得多，这就是路径验证防劫持的底气。</span>

<span class="marginnote">常见误区：初学者容易以为迁移后 TCP 拥塞窗口也能「跟着搬」。实际上新路径的带宽时延积可能差一个数量级——从家里 Wi-Fi 切到地铁蜂窝，盲目继承满窗口只会把新路径打爆，所以 RFC 要求拥塞状态不得盲目继承。</span>

## 边界

本课不引入 QUIC 多路径扩展的全部。BBRv2/v3 是下一课。后课默认：CID 使运输跨地址存活；要验证路径。

服务器集群必须能按 CID 路由到同一实例，否则迁移失败。

下一课[BBRv2 / v3](/cs/bbr-v2-v3)。

## 小结

- CID 解耦连接与地址。
- 新路径要挑战验证。
- 拥塞状态不可盲目带着走。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9000。

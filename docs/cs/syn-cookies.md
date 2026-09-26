---
title: SYN cookies
date: 2026-09-08
section: cs
---

# SYN cookies

<div class="epigraph">
<p>不把半开连接存进队列，而把状态编码进 SYN-ACK 的序号；ACK 回来再重建。抗泛洪，但可能丢掉选项。</p>
<footer>—— 据 Bernstein 的 SYN cookies 笔记；RFC 4987 TCP SYN Flooding 整理</footer>
</div>

[上一课](/cs/tcp-state-machine) 的 SYN-RECEIVED 要占 TCB。[握手](/cs/tcp-handshake) 假定队列够。缺口是 **SYN flood**：伪造源填满半开。cookies 用密码学或哈希把 MSS 等塞进 seq。本课不把 keepalive 写完。

## 问题

攻击者发大量 SYN，源不可达，队列满则合法 SYN 被拒——可用性。Cookie：SYN-ACK 的初始序号 = 哈希(密钥, 四元组, 时间, MSS 编码)，不存 TCB。最终 ACK 校验哈希再创建 ESTABLISHED。代价：时间戳/SACK/缩放可能无法从 32 位里恢复，长肥管道受害者。只在队列压力下启用是常见政策。

不要把 cookie 写成 TLS cookie 或 QUIC retry 的拷贝，对象类似、编码不同。

<span class="marginnote">TCB（传输控制块）是内核为每条 TCP 连接保存的全部状态：序号、窗口、定时器。半开连接也要占一个 TCB——SYN flood 的本质就是用几十字节的假包骗走几百字节的内核内存。</span>

<span class="marginnote">RFC 4987 描述攻击与对策。Linux 实现细节不背。密钥要轮换。</span>

### 半开搬进序号

抗泛洪，可能丢掉缩放等选项。任播要同一密钥验证。压力下启用是常见政策，永久开启要接受降级。

## 方法

对照：存 TCB vs 编码进序号。画：SYN → cookie SYN-ACK → ACK 验证 → 建连。与 RPKI 对照：都是用密码学/哈希抗伪造，一层在路由，一层在传输。

```mermaid
flowchart TD
  SYN["SYN 不存表"] --> CK["序号=cookie"]
  CK --> ACK["回 ACK"]
  ACK --> VER["验证后建 TCB"]
```

## 机制

AIMD 尚未开始，cookies 在握手。MSS 钳制仍可发生在 SYN 上。Anycast 与 ECMP：ACK 必须回到能验证同一密钥的实例，否则失败——接住拓扑课。卫星长 RTT 使 cookie 时间窗要宽容。

```mermaid
flowchart TD
  S["SYN 到达且队列压力大"] --> C["现算 cookie：哈希(密钥, 四元组, 分钟计数, MSS)"]
  C --> A["SYN-ACK 携带 cookie 作初始序号"]
  A --> R["ACK 回来：重算哈希比对"]
  R -->|"匹配且在时间窗内"| E["此刻才建 TCB，进入 ESTABLISHED"]
  R -->|"不匹配或已过期"| D["静默丢弃，零内存占用"]
  FAKE["伪造源永不回 ACK"] --> NOPE["没有任何 TCB 生成：队列不涨"]
```

<span class="marginnote">数字实例：cookie 里编入「分钟计数」，服务器只接受当前与上一分钟生成的 cookie。若不编码时间，攻击者可以攒一万个旧 cookie 慢慢回放；有了时间窗，过期即作废，重放失去意义。</span>

合法瞬时高峰也会触发 cookies，表现为偶发无窗口缩放。

## 边界

本课不引入 SYNPROXY 的全部。保活与半开是下一课。后课默认：压力下用 cookie 抗泛洪；选项可能丢。

把 cookies 当默认永久开启要接受功能降级。

<span class="marginnote">常见误区：以为开了 cookie 就万事大吉。32 位序号塞不下 MSS 之外的 SACK、窗口缩放等选项，长肥管道（大带宽高时延链路）会因此跑不满——这是「可用性换功能」的定价，不是 bug。</span>

下一课[保活与半开](/cs/keepalive-half-open)。

## 小结

- 半开状态搬进 SYN-ACK 序号。
- 抗 SYN 泛洪，可能丢 TCP 选项。
- 验证要密钥与时间窗，任播要同源。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 4987；Bernstein cookies。

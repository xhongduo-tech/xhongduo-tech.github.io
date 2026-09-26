---
title: TCAM 查表
date: 2026-09-08
section: cs
---

# TCAM 查表

<div class="epigraph">
<p>三态内容寻址把前缀与掩码一次并行匹配，再按优先级编码出最长前缀；容量贵，所以 FIB 与 ACL 都在抢这块硅。</p>
<footer>—— 据 McAuley and Francis, Fast Routing Table Lookup, INFOCOM 1993；交换芯片查表实践整理</footer>
</div>

[上一课](/cs/router-architecture) 假定 LPM 线速。主干[最长前缀匹配](/cs/lpm) 给算法直觉。缺口是**硬件怎么一次比完**：TCAM 的 0/1/X、优先级编码器、与 SRAM 流水对照。本课不把 OpenFlow 流表写完。

## 问题

软件树/哈希在最坏深度上抖动。TCAM：每个比特存值与掩码，包键并行比较，命中集用优先级取最长或最先。ACL 五元组同样是三态。代价：功耗、容量（数十万量级则贵）、更新要插空洞避免打断线速。算法类 LPM（树、 trie）用 SRAM 换更新友好，最坏延迟要工程保证。

不要把 TCAM 写成「无限智能」：表满则无法再装更细前缀，运营要聚合。

<span class="marginnote">术语翻译：三态就是「每个比特除了存 0、1 还能存 X（不关心）」——X 相当于通配符，/8 这种粗前缀把低 24 位全标成 X，就能和 /32 一起被同一次并行比较比出来。</span>

<span class="marginnote">TCAM 也用于 MAC、流表。本课不讲电路晶体管。更新与查表并发是芯片内部课题。</span>

### 并行匹配有容量税

三态一次比完，再取 LPM/优先级。表满无法再装细前缀。IPv6 更宽的键更吃硅。

## 方法

对照：TCAM 并行 vs trie 多级。画：键 → 匹配向量 → 优先级 → 动作（下一跳、ACL drop）。RPKI 无效丢弃是动作之一，仍占表项。

```mermaid
flowchart TD
  KEY["包键"] --> TCAM["并行三态匹配"]
  TCAM --> PRI["优先级/LPM"]
  PRI --> ACT["下一跳或 ACL"]
```

## 机制

EVPN MAC 规模可压垮表，要聚合或层次。ECMP 下一跳组在 SRAM，TCAM 只指向组号。OpenFlow 后课把多级流表暴露给控制面，硬件仍常是 TCAM 级联。与容量 $C$ 无关：查表是转发税，不是信道。

安全：故意装大量更长前缀可耗尽 TCAM，像 MAC 耗尽。

<span class="marginnote">数字实例：TCAM 位元比 SRAM 位元多晶体管，功耗常是其数倍到数十倍，单片容量也就存几万到几十万条表项——所以交换机把最贵的三态硅留给路由前缀和 ACL，MAC 与邻居表等大表放 SRAM。</span>

<span class="marginnote">常见误区：初学者以为表能一直装下去。实际上每条 /24 级细前缀都实打实占一行三态位元，攻击者可以批量灌入更长前缀把表塞满，让新路由或安全规则装不进来——聚合与容量防护是运营必修课。</span>

```mermaid
flowchart TD
  PKT["目的地址 10.1.2.3"] --> M1["表项 10.0.0.0/8：命中"]
  PKT --> M2["表项 10.1.0.0/16：命中"]
  PKT --> M3["表项 172.16.5.0/24：未命中"]
  M1 --> ENC["优先级编码器"]
  M2 --> ENC
  M3 --> ENC
  ENC --> WIN["取最长 /16 → 下一跳 B"]
```

## 边界

本课不引入算法 LPM 的全部压缩 trie。SDN 与 OpenFlow 是下一课。后课默认：线速 LPM/ACL 靠 TCAM 或等价并行匹配，容量是硬约束。

IPv6 更宽的键更吃 TCAM，这是过渡课留下的硬件账。

下一课[SDN 与 OpenFlow](/cs/sdn-openflow)。

## 小结

- TCAM 并行匹配三态键，再取最优。
- 容量与功耗限制细前缀与 ACL。
- SRAM 算法查表换更新，要保证最坏时延。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：McAuley–Francis, 1993；芯片实践。

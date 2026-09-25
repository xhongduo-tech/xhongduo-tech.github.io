---
title: 数据中心 Clos / 胖树
date: 2026-09-08
section: cs
---

# 数据中心 Clos / 胖树

<div class="epigraph">
<p>叶子–脊线全互联把东西向带宽做成多级 Clos；ECMP 把流洒在多条等代价边上，取代大二层生成树。</p>
<footer>—— 据 Clos, A Study of Non-blocking Switching Networks, BSTJ 1953；Al-Fares et al., SIGCOMM 2008 胖树整理</footer>
</div>

[交换机 HOL](/cs/output-queue-hol) 与 [ECMP](/cs/ecmp-hashing) 已备好。[上一课](/cs/p4-programmable) 不改拓扑。缺口是**数据中心怎么连**：leaf-spine、过订购比、L3 到叶子。本课不把 incast 扇入写完。

## 问题

接入–汇聚–核心树在东西向拥挤，STP 还要阻塞边。胖树/Clos：每叶子连所有脊，脊再可选超脊。带宽由「叶子上联数 × 口速」决定，过订购是刻意的经济。转发：每叶子一个子网，主机默认网关在叶子（IRB/EVPN），中间纯 L3 ECMP，无 STP 大域。VXLAN 覆盖跑在这张 IP 网上。

<span class="marginnote">过订购用数字算：一片叶子下面接 48 个 10G 主机口，上联到脊只有 4 条 10G，过订购比就是 48:4 = 12:1。它赌的是「不是所有人同时满发」；若 48 台机器同时往外拷数据，每台平均只分到不到 1G 的东西向带宽，排队不可避免。</span>

不要把 Clos 写成无阻塞的数学保证：流量矩阵一偏，脊仍可满。

<span class="marginnote">Clos 原是电话交叉。Al-Fares 把胖树搬到以太网。本课不点名某云的内部名。</span>

### 东西向用 ECMP

leaf-spine 多条等代价，大 STP 退出。过订购是经济，流量一偏脊仍满。故障重哈希会短暂乱序。

<span class="marginnote">ECMP（等价多路径）可以想成收费站并开的一排等长车道：路由器把去往同一目的地的每条流按哈希固定分给某条路，流内不换道（免得包乱序），不同流之间摊匀负载；哪条车道先满，取决于流量分得匀不匀。</span>

## 方法

画：leaf 与 spine 全网格。对照校园三层：这里每跳都 ECMP。LACP 可把多口当一条上联，但叶子到脊更常用独立 ECMP 边。

```mermaid
flowchart TD
  H["主机"] --> L["叶子"]
  L --> S1["脊"]
  L --> S2["脊"]
  S1 --> L2["另一叶子"]
  S2 --> L2
```

## 机制

TE 在数据中心常简化为 ECMP + 尽量对称布线；极化仍在。BGP 可作 IGP（每盒一个 ASN 或私有 AS），收敛与策略比 OSPF 更熟——这是运营选择。5G 核心 UPF 也可落在 Clos 上，无线切片不改机房拓扑。

一条脊挂掉之后，网络是怎么收敛的：

```mermaid
flowchart TD
  A["一条脊交换机故障"] --> B["叶子收不到它的路由"]
  B --> C["BGP 撤销这条路径"]
  C --> D["ECMP 下一跳集合收缩"]
  D --> E["受影响的流重新哈希到剩余脊"]
  E --> F["换了路径的流短暂乱序"]
```

故障：一条脊挂了，哈希重映射，短暂乱序，与 LACP 成员 down 同类。

<span class="marginnote">初学者容易把「无阻塞 Clos」理解成永不拥塞。实际上无阻塞只保证拓扑里存在足够多的可走路径；流量矩阵一偏——比如十台服务器同时拷同一台——共享的脊照样打满、排队甚至丢包，拓扑救不了偏斜的流量。</span>

## 边界

本课不引入 Jupiter 一类专有细节。incast 是下一课。后课默认：东西向用 Clos + L3 ECMP，大 STP 退出。

光模块与 100G lane 决定实际能铺多少脊。

下一课[incast](/cs/incast)。

## 小结

- leaf-spine Clos 提供多条等代价东西向路径。
- 过订购是容量规划，不是协议字段。
- 用 ECMP 替代阻塞冗余边。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Clos, 1953；Al-Fares et al., 2008。

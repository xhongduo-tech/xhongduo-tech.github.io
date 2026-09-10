---
title: WFQ
date: 2026-09-08
section: cs
---

# WFQ

<div class="epigraph">
<p>加权公平排队按权仿真 GPS：每流自己的队列，调度器近似按权分享 $C$，隔离大象与老鼠。</p>
<footer>—— 据 Demers, Keshav and Shenker, SIGCOMM 1989；Parekh and Gallager, IEEE/ACM ToN 1993 GPS/WFQ 整理</footer>
</div>

[RTT 不公平](/cs/cc-fairness-rtt) 与 [缓冲膨胀](/cs/bufferbloat) 指出单队列的病。[上一课](/cs/bufferbloat) 点名 FQ-CoDel。缺口是 **WFQ/GPS 对象**：虚时间、权重、与 AIMD 的关系。本课不把令牌桶写完。

## 问题

单 FIFO 里短 RTT 大象压死一切。GPS：无穷可分流体按权分带宽。WFQ：包调度近似 GPS，复杂度高；DRR/FQ 是近似。每流隔离后，AIMD 在自己份额里收敛，RTT 偏差不再偷别人的定额。状态：流数 × 队列，ASIC 难，故数据中心少用完整 WFQ，接入网与 Linux qdisc 常见。

不要把 WFQ 写成「保证延迟上限」除非再加令牌桶准入（Parekh–Gallager）。

<span class="marginnote">DKS 1989。PG 1993 给端到端延迟界条件。本课不证界。</span>

### 隔离份额

近似 GPS，AIMD 不再互偷。完整延迟界还要整形。每流状态在核心不可扩展，用类或 FQ 近似。

## 方法

对照 FIFO / PQ / WFQ。画：分类 → 每流队列 → 按虚时间选包。与 ECMP：ECMP 选路径，WFQ 在一跳内分份额。

```mermaid
flowchart TD
  CL["分类到流队列"] --> GPS["按权仿真"]
  GPS --> OUT["出端口"]
  ISO["隔离"] --> AIMD["各流自己的 AIMD"]
```

方法止于选定对象与对照；机制才说它如何嵌入已有分层与主干课。

## 机制

CoDel 可跑在每个 WFQ 队列上。DiffServ 后课用有限类而不是每流，状态少。RoCE 优先级是 PQ 不是 WFQ。5G 切片在无线侧隔离，核心仍可用类排队。

哈希流识别错误会把两用户捆一队列，隔离失败。

## 边界

本课不引入 WF2Q 的全部。令牌桶与整形是下一课。后课默认：每流公平排队隔离份额；完整延迟界还要整形。

核心路由器按流状态不可扩展，用类或抽样。

上一课留下的缺口在本课收口；「WFQ」进入后课词汇表后只引用。文献用来钉对象与边界，不把本课写成该主题的独立综述。下一课[令牌桶与整形](/cs/token-bucket-shaping)。

## 小结

- WFQ 近似 GPS 按权分享。
- 隔离后 AIMD 不再互偷。
- 每流状态是扩展边界。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Demers et al., 1989；Parekh–Gallager, 1993。

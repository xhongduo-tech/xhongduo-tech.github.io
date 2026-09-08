---
title: RTO 与 Karn
date: 2026-09-08
section: cs
---

# RTO 与 Karn

<div class="epigraph">
<p>重传超时必须跟着测量到的往返时间走；一次重传之后的 ACK 不能用来更新估计，否则会把超时越调越乱。</p>
<footer>—— 据 RFC 793 的超时重传；Karn and Partridge, 1987；Kurose and Ross</footer>
</div>

[上一课](/cs/tcp-seq-rexmit)给出序号、累计 ACK 与「丢了就重发」。本课不重讲滑动前缀。缺口是**何时**重发：超时太短则假重传挤网络，太长则空等。[拥塞控制](/cs/tcp-congestion)还没开课，但 RTO 已经会制造重复段。Karn 算法处理「这段是第一次发还是重发的」的歧义。

## 问题

RTT 随路径变。RFC 793 已有重传，工程上用平滑 RTT 与均值偏差（Jacobson）设 RTO。若某段重传过，回来的 ACK 无法对应哪一次发送，把这次样本算进估计会系统性偏低或偏高。Karn：忽略重传段的 RTT 样本，并在重传时指数退避 RTO。缺口是**定时器与估计的一致性**，不是选择确认。

<span class="marginnote">时间戳选项后来让重传段也可测 RTT，那是对 Karn 的补充，不是取消「测不准就别更新」。假超时仍会进入拥塞响应，后课再收。</span>

## 方法

对未重传的段：采样 RTT，更新平滑值与偏差，RTO = 平滑 + 系数×偏差，并设下限。超时：重传最早未确认段，RTO 加倍，直到收到新 ACK 再按测量恢复。快速重传（三重复 ACK）是另一条丢包信号，其定时器交互留给后课 SACK/拥塞。

```mermaid
flowchart TD
  SAMPLE["未重传段的 RTT"] --> EST["平滑与偏差"]
  EST --> RTO["重传超时"]
  REX["已重传"] --> KARN["不采用该 ACK 作样本"]
  KARN --> BACK["RTO 指数退避"]
```

## 机制

RTO 把「IP 会丢」翻译成发送方的时钟。估计错误时，[端到端](/cs/layering-e2e) 仍最终可靠（重传到对），但延迟与负载变差。Karn 避免正反馈：超时 → 重传 → 错误样本 → 更小 RTO → 更多超时。

## 边界

本课不把 Linux 的 min RTO 默认值当 RFC 正文，不引入 F-RTO 全部细节。累计 ACK 仍不能告诉发送方「中间哪块到了」，下一课 SACK。

后课默认：超时按测量走，重传段不污染 RTT。块状丢失的信息下一课补。

## 小结

- RTO 跟随 RTT；Karn 丢掉有歧义的样本并退避。
- 假超时会制造重复段。
- 选择确认下一课。
- 出处：RFC 793；Karn and Partridge, 1987；Kurose and Ross。

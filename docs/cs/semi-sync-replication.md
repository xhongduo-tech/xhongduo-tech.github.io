---
title: 半同步复制
date: 2026-09-08
section: cs
---

# 半同步复制

<div class="epigraph">
<p>提交返回前至少一备库收到日志：RPO 相对异步收紧，延迟相对全同步放松。仍可能在窗口里丢，合同要写「几个 ACK」。</p>
<footer>—— 据 Gray 复制；MySQL 半同步；Postgres 同步复制级别</footer>
</div>

[上一课](/cs/logical-replication-cdc)把流送给异构下游。本课回到同构 HA：异步复制 RPO 大（主挂丢未传送日志）；全同步每备都 fsync 则延迟差。半同步（semi-sync）：等待 $k$ 个备库 ACK（常 $k{=}1$）再让提交返回。持久性等级课的③/④ 落地。

## 问题

ACK 语义：收到内存 vs 已 fsync，差一档 RPO。缺口是**主挂时选谁**：已 ACK 的备库有提交，未 ACK 的可能落后——升主要选最前，并处理裂脑。半同步在备库全挂时的降级：等还是退异步，上一课 D 等级已问。

性能：提交延迟 ≈ 组提交 + 网络 RTT + 备库刷盘。跨 AZ 明显。组提交窗口可等一批一起发，与半同步叠加。

<span class="marginnote">术语翻译：ACK 有两档含义——「收到内存」是备库把日志放进接收缓冲就算数，备库机器若此时断电就丢；「已 fsync」是日志真正写到盘上，断电也不丢。合同写「k 个 ACK」时若不写档位，RPO 完全是两个等级。</span>

<span class="marginnote">数字实例：本地组提交加 fsync 约 1 ms；跨可用区 RTT 常到 1-2 ms、备库 fsync 再 1 ms——半同步把每次提交从约 1 ms 拉到约 3-4 ms，对每秒上万次提交的库就是吞吐掉一档的量级。</span>

<span class="marginnote">MySQL after_sync/after_commit 差异是产品坑。Postgres `synchronous_commit` 与 `synchronous_standby_names`。本课 k-ACK 机制。</span>

## 方法

配置 k 与名单。监控复制延迟、ACK 超时。超时降级要告警。只读备库可异步，不进 k。

与 2PC：半同步不是跨分片原子，只是单主日志的耐久扩展。跨分片仍 2PC 或 Calvin。

```mermaid
flowchart TD
  COM["主 commit 记录"] --> SEND["传 WAL"]
  SEND --> ACK["k 个备库确认"]
  ACK --> RET["返回客户端"]
  FAIL["备库全超时"] --> POL["降级或拒绝"]
```

## 机制

乱序：备库 apply 可落后于收到；若 ACK 只表示收到，升主后还要 apply 完才服务——RTO 含追平。若 ACK=apply+fsync，延迟更大、RTO 更干净。

脑裂：旧主复活仍接写，需 fencing（STONITH、时间线 id）。后课分布式会再遇。

上面那张图回答「一次提交怎么等 ACK」；下面这张回答主挂之后的事——升主不是随便挑一台，选错备库会把已 ACK 的提交丢掉。

```mermaid
flowchart TD
  A["主库宕机"] --> B["收集各备库日志位置"]
  B --> C["选最前的已 ACK 备库"]
  C --> D["提升为新主"]
  D --> E["fencing 旧主: STONITH 或时间线"]
  D --> F["落后备库追平日志"]
  F --> G["重新加入 k 名单"]
```

<span class="marginnote">直觉类比：fencing 是给旧主发一张「通行证作废令」——STONITH（Shoot The Other Node In The Head）干脆直接断它的电。宁可旧主彻底缺席，也不能让它带着过期数据继续接客；「我还没死」的旧主比「已死」的更危险。</span>

## 边界

本课不量化读己之写的会话粘滞——下一课。也不把仲裁多数派当半同步（那是共识）。Spanner 后课。

后课默认：要小 RPO 用 k-ACK 同步复制；要低延迟用异步并接受丢失窗口。复制延迟与读己之写：会话读备库看见旧值。

半同步收紧的是提交日志的故障域，不是隔离级别。

## 小结

- k 个备库 ACK 再返回，RPO/延迟可调。
- ACK 是收到还是 fsync 必须写清；降级要告警。
- 复制延迟与读己之写下一课。
- 出处：Gray；MySQL/Postgres 同步复制。

---
title: 模糊检查点
date: 2026-09-08
section: cs
---

# 模糊检查点

<div class="epigraph">
<p>检查点不冻结世界：记下当时的脏页与活跃事务，后台继续刷；恢复用这份快照当分析起点，而不是要求检查点瞬间数据文件干净。</p>
<footer>—— 据 Mohan ARIES 检查点；Gray and Reuter；主干 checkpoint 课的模糊补全</footer>
</div>

[上一课](/cs/aries-phases-clr)从检查点 LSN 开分析。本课不写 CLR。缺口是检查点本身：若采取 sharp（停更新、刷光脏页）则吞吐归零。模糊检查点（fuzzy checkpoint）允许检查点期间页继续变脏，只保证记录了「这些页可能脏、这些事务活跃」，redo 仍从足够早的 LSN 开始。

## 问题

主干 [检查点](/cs/checkpoint) 缩短恢复。模糊：写 begin-checkpoint，拷贝脏页表与事务表，写 end-checkpoint，再把检查点指针进控制文件。拷贝期间新脏页不在表里？规则是：拷贝后变脏的页，其 recoveryLSN 仍 ≥ 检查点，redo 从检查点前足够早处扫会覆盖。细节在 ARIES：脏页表含当时未刷页。

缺口是**与刷脏并行**：检查点不代替刷脏；后台写者降低 min recoveryLSN，使下次检查点更紧。两者一起决定 RTO。

<span class="marginnote">模糊不是「随便丢日志」。WAL 仍先于页。检查点只限制崩溃后必须扫描的日志前缀。</span>

## 方法

周期或按日志体积触发。太频：写检查点记录与控制文件成为瓶颈。太稀：恢复扫太多。与组提交：检查点 LSN 之前的日志必须已持久。

<span class="marginnote">直觉类比：sharp 检查点像商店打烊盘点——关门、清点、再开张；模糊检查点像「边营业边盘货」：先给账本拍张快照，之后照常做生意，快照只保证记下「当时哪些账可能还没结」。</span>

<span class="marginnote">术语翻译：RTO（Recovery Time Objective）是恢复时间目标——崩溃后允许多久恢复服务。检查点做得越勤、脏页刷得越紧，崩溃后要重放的日志就越短，RTO 就越小；代价是平时性能被摊薄。</span>

复制：备库的 replay 位置类似检查点水位。备份常与检查点对齐拿一致点。

```mermaid
flowchart TD
  BEG["begin checkpoint"] --> SNAP["拷贝脏页表 / 事务表"]
  SNAP --> END["end checkpoint"]
  END --> CTL["控制文件指针"]
  BG["后台刷脏"] --> MIN["推进 min recoveryLSN"]
```

## 机制

恢复分析从最近完成的 end-checkpoint 走。半截检查点忽略，用上一个。与 TDE：检查点记录可含密钥版本。与缓冲 2Q：刷脏偏好老脏页，不是刚扫描的一次性页——策略与检查点目标对齐。

模糊检查点期间的崩溃：未完成检查点无效，正确性靠上一完成点+后续日志。

检查点写一半崩溃时恢复从哪里起步？答案是完全丢弃半成品、退回上一个完整检查点。

```mermaid
flowchart TD
  CRASH["崩溃时检查点只写了一半"] --> HALF["begin 有、end 没有"]
  HALF --> INVALID["该检查点整体无效"]
  INVALID --> PREV["回退用上一个完整的 end-checkpoint"]
  PREV --> ANALYZE["按它的脏页表/事务表开始分析"]
  ANALYZE --> REDO["redo 从足够早的 LSN 向后扫"]
```

<span class="marginnote">数字实例：若每 10 分钟做一次检查点，崩溃平均要重放约 10 分钟的日志；把周期缩到 1 分钟，重放量降到约 1 分钟，代价是每分钟多写一份快照并多刷一批脏页——恢复时间与运行开销就是这笔交换。</span>

## 边界

本课不比生理 vs 逻辑日志粒度。也不把检查点当 vacuum。RTO 课把时间预算量化。

后课默认：生产用模糊检查点+后台刷脏。日志粒度：记字节差、记操作、还是整页。

检查点是恢复的索引，不是把数据库变成只读。

## 小结

- 模糊检查点记录脏页与事务快照，不停机刷光。
- 后台刷脏推进 redo 起点；频率权衡恢复与运行。
- 日志粒度下一课。
- 出处：ARIES；Gray and Reuter；主干 checkpoint。

---
title: 组提交
date: 2026-09-08
section: cs
---

# 组提交

<div class="epigraph">
<p>许多事务的 commit 记录挤进同一次日志 fsync：延迟换吞吐，持久性不等式仍是「提交返回 ⇒ 日志已落」。</p>
<footer>—— 据 Gray and Reuter 组提交；Mohan；磁盘与 NVMe 上的 WAL</footer>
</div>

[上一课](/cs/shadow-paging)对照切根。本课回到 WAL 路径的刷盘：每个提交都 `fsync` 则磁盘 IOPS 打满，吞吐上不去。组提交（group commit）把一个日志缓冲里的多个 commit 一起刷。持久性等级下一课把「刷到哪」再分级。

## 问题

提交延迟 ≈ 等刷盘。组：领导者收集一个窗口内的提交者，一次 I/O，大家一起返回。缺口是**窗口**：等太久延迟差；不等则退化成每提交一刷。自适应窗口看队列长度。与并行：组提交是日志尾的串行点，是 OLTP 扩展上限之一——可分区日志缓解，复杂度高。

steal/no-force 仍成立：数据页不必在组里刷。只刷日志。

<span class="marginnote">直觉类比：组提交像拼车去机场。一个人叫一趟车（一次 fsync）太贵；等一小段凑满一车人再出发，车费（I/O）大家平摊。等车的时间就是提交延迟，车开得越满、整体越省。</span>

<span class="marginnote">数字实例：设一次 fsync 耗 1 ms、盘的 IOPS 上限约 1000，逐笔刷盘的提交速率就卡在约 1000 TPS。若每 50 个 commit 拼一组，同样的 IOPS 就能支撑约 50 倍的提交带宽——吞吐换来的是最多一个窗口的等待；且这不是放松持久性，没赶上这一刷的事务会整体 abort，不会半提交。</span>

<span class="marginnote">Gray and Reuter。MySQL/Postgres 都有组提交实现。本课不调某个 `commit_delay` 参数表。</span>

## 方法

日志缓冲满或超时或 N 个 commit → flush。返回后事务可见（按隔离：提交可见性还要看锁释放点）。备库半同步后课会等备库 ACK，组提交窗口与复制 ACK 可叠。

崩溃：组里已刷的全提交，未进这次刷的全部 abort——原子的是这一刷。

```mermaid
flowchart TD
  C1["commit"] --> BUF["日志尾缓冲"]
  C2["commit"] --> BUF
  BUF --> FS["一次 fsync"]
  FS --> OK["整组返回"]
```

## 机制

与 TDE：加密日志页再刷，CPU 进组路径。与双写：数据页路径独立。校准：优化器不管组提交，但基准 TPC-C 对它极敏感——后课基准。

无组提交时，用户同步提交延迟=单次 fsync；SSD 仍有上限。

```mermaid
flowchart TD
  T1["事务 A 已进组"] --> FL["这一组 fsync 落盘"]
  T2["事务 B 已进组"] --> FL
  T3["事务 C 还在缓冲"] --> NX["没赶上这一刷"]
  FL --> A1["A、B：整组提交返回"]
  NX --> A2["C：abort，重提交"]
  CR["崩溃"] --> FL
  CR --> NX
```

<span class="marginnote">为什么重要：崩溃后数据库只认「最后一次成功的那一刷」这条线。线上的全部提交、线下的全部作废，没有中间态——判断边界的不是事务提交的先后，而是它有没有赶上这一次 I/O。</span>

## 边界

本课不定义 RPO 的全部等级。也不把组提交当 2PC。异步提交（先返回再刷）属于下一课耐久性放松。

后课默认：同步提交走组刷日志。持久性等级：同步、异步、备库参与。

组提交不改变 WAL 不等式，只改变每次 I/O 摊到多少事务。

## 小结

- 多 commit 共享一次日志 fsync，吞吐升、尾延迟可能升。
- 崩溃原子单位是这一刷。
- 持久性等级下一课。
- 出处：Gray and Reuter；ARIES 实践。

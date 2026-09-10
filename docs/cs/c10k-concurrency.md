---
title: C10K 与并发模型
date: 2026-09-08
section: cs
---

# C10K 与并发模型

<div class="epigraph">
<p>一万并发连接的瓶颈在每连接的内核与用户态开销；事件驱动、高效多路复用与少拷贝把 C10K 变成 C10M 量级的工程，而不是新协议。</p>
<footer>—— 据 Kegel, The C10K Problem；libevent/nginx 实践整理</footer>
</div>

[Reactor](/cs/reactor-proactor) 给模式。[套接字](/cs/socket-api) 给 API。[上一课](/cs/reactor-proactor) 留下规模。缺口是 **C10K 问题**：fd 上限、epoll vs select、内存。本课不把粘包写完。

## 问题

2000 年代：select 线性扫描、每连接一进程、内核套接字缓冲默认大，一万连接吃光 RAM。对策：epoll/kqueue、降低缓冲、事件循环、零拷贝 sendfile。今日 C10M 还要网卡多队列、RPS、用户态栈。这与 $C$ 公式无关，是主机实现税。Maglev 把连接摊到多机，是水平解。

不要把 C10K 写成必须用某种语言。

<span class="marginnote">Kegel C10K。本课不点名基准广告。</span>

### 每连接开销 × N

事件化与少拷贝，不是新协议。H2 多流也减连接数。fd 耗尽是可用性攻击面。业务同步塞进循环会更早失败。

## 方法

对照：进程 / 线程 / 事件 / 协程。画：连接数 × 每连接开销。与 incast：那是网络队列；这里是主机 fd。

```mermaid
flowchart TD
  N["连接数"] --> COST["每连接内存与系统调用"]
  COST --> LIM["CPU/RAM 上限"]
  EV["事件循环"] --> LOW["降每连接开销"]
```

方法止于选定对象与对照；机制才说它如何嵌入已有分层与主干课。

## 机制

HTTP 持久减少握手但仍占 fd。H2 多流在一条连接上减轻 C10K。QUIC 用户态更灵活也更吃 CPU。SYN cookies 保护握手，不降低已建立连接数。测量：ss -s，不是 ping。

安全：fd 耗尽是拒绝服务，要 ulimit 与限连。

## 边界

本课不引入 DPDK 的全部。粘包与帧定界是下一课。后课默认：万级并发靠事件化与减每连接状态。

把业务逻辑同步塞进事件循环会在远低于 C10K 时失败。

上一课留下的缺口在本课收口；「C10K 与并发模型」进入后课词汇表后只引用。文献用来钉对象与边界，不把本课写成该主题的独立综述。下一课[粘包与帧定界](/cs/message-framing)。

## 小结

- 瓶颈是每连接开销 × N。
- 高效多路复用 + 少线程。
- 协议多路（H2）也减连接数。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Kegel C10K；服务器实践。

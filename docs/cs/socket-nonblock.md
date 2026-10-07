---
title: 阻塞与非阻塞套接字
date: 2026-09-08
section: cs
---

# 阻塞与非阻塞套接字

<div class="epigraph">
<p>默认 read 会把进程睡到有字节；非阻塞让调用立即返回「现在没有」，由事件通知再来读。</p>
<footer>—— 据 Stevens, UNIX Network Programming；Kurose and Ross 对套接字的整理</footer>
</div>

[上一课](/cs/socket-api)把 UDP/TCP 接到文件描述符：`connect`、`accept`、`read`、`write`。本课不重列调用。缺口是等待：阻塞套接字上，对端慢则线程卡在系统调用里；并发连接若一人一线程，[进程/线程](/cs/thread-shared-addr) 成本会先爆。非阻塞加多路复用是同一 API 上的调度合同。网络课在此结束，下一课转入物理层：香农容量在链路。

## 问题

Berkeley 套接字默认阻塞：无数据则睡眠，缓冲满则写睡眠。这与[文件字节流](/cs/file-bytestream)一致，但网络延迟是对端与路径，不是磁盘臂。服务器要同时服务许多已建立连接，不能为每个 `read` 配一个内核线程当默认。缺口是**调用是否等待就绪**，不是新的 TCP 状态。

<span class="marginnote">`select`/`poll`/`epoll`/`kqueue` 都是「等这些描述符可读可写」。本课只钉合同，不选哪一个可移植层。</span>

<span class="marginnote">阻塞 read 可以类比成快餐店点单后一直站在柜台前干等，汉堡做好才挪步；非阻塞是点完单拿个叫号器就走——叫号器一响（事件通知）再回来取餐。等餐时间你可以干别的事，于是一个服务员（线程）就能同时照看几十个号。</span>

## 方法

设非阻塞：`read` 无数据返回 EAGAIN；`connect` 可在进行中返回。事件循环：注册描述符，醒来后在就绪集合上读写，直到再次 EAGAIN。阻塞套接字仍适合简单客户端。半关闭与 [TIME_WAIT](/cs/tcp-time-wait) 不因非阻塞消失。

```mermaid
flowchart TD
  BLK["阻塞 read"] --> SLEEP["无数据则睡眠"]
  NB["非阻塞 read"] --> AGAIN["立即 EAGAIN"]
  AGAIN --> MUX["事件多路再读"]
```

<span class="marginnote">EAGAIN 的字面意思是「资源暂不可用，请待会儿再试」。它不是错误，而是非阻塞世界里的一句「现在没有」——初学者容易把它当异常去重试或报错，实际上事件循环遇到它就安静跳过，等下次就绪通知再来。</span>

## 机制

非阻塞把等待从「藏在系统调用里」变成「进程自己调度」，与内核[时间片](/cs/timeslice-cfs) 分工：内核仍唤醒，用户决定先服务谁。它不改 TCP 窗口与拥塞；只改线程是否被占住。数据库在后课另起：套接字送来的是字节，还没有表。

```mermaid
flowchart TD
  REG["注册关心的描述符集合"] --> WAIT["epoll_wait：<br/>无线程空转地睡"]
  WAIT -->|"内核唤醒"| RDY["拿到就绪集合"]
  RDY --> DISP["逐个分发处理器"]
  DISP --> IO["read / write 直到 EAGAIN"]
  IO -->|"本轮处理完"| WAIT
  IO -->|"连接仍有未读完数据"| REG
```

<span class="marginnote">为什么不能一人一线程：每条线程默认占 1-8 MB 栈，一万条连接就是 10-80 GB 内存——只为「等着读数据」这一个动作。换成非阻塞加一个事件循环，同样一万条连接只要一个线程加一张就绪表，内存从 GB 级降到 MB 级。这就是本课说的「线程成本先爆」的具体含义。</span>

## 边界

本课不把异步 I/O 的全部 POSIX 接口写完，不引入用户态协议栈。字节如何变成元组集合，是后课数据库的起点。

后课默认：网络端点可以不阻塞线程。持久共享数据需要关系，而不是再开一种套接字。

## 小结

- 阻塞等待就绪；非阻塞立即返回，靠多路复用再来。
- 不改变 TCP 语义，只改变进程调度。
- 网络课到此收束；下一课转入物理层的香农容量。
- 出处：Stevens *UNIX Network Programming*；Kurose and Ross。

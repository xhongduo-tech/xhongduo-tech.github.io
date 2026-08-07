---
title: select 与 poll 的原理与局限
date: 2026-08-07
---

# select 与 poll 的原理与局限

<div class="epigraph">
<p>select 与 poll 是 I/O 多路复用的拓荒者——它们让「一个进程管一堆连接」成为可能，也留下了「扫全表、传全集」的青春印记。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Unix 网络编程》§6.3、§6.4 与 Linux 内核 ｜ 2026-08-07</p>
</div>

## 为什么从 select/poll 开始

epoll 不是凭空出现的——它的前辈 `select` 与 `poll` 首先实现了「一个进程监视多个 fd」。理解它们的**原理与局限**，才能理解 epoll 针对性地解决了什么。这一节看这两个「经典多路复用」的工作方式与瓶颈。<span class="marginnote">回顾《I/O 模型》：多路复用 = 一个进程等一堆 fd 就绪。select/poll 是第一代实现——<strong>内核扫描 fd 集合、通知就绪，但每次调用都要「全量传入、全量扫描」</strong>。epoll 正是为消除这些 O(n) 成本而生。</span>

## 1 select：三组位图

**select**：监视**读、写、异常**三组 fd，等其中任意就绪。

```c
fd_set readfds, writefds;
FD_ZERO(&readfds); FD_SET(sock, &readfds);
select(max_fd + 1, &readfds, &writefds, NULL, &timeout);
if (FD_ISSET(sock, &readfds)) { /* sock 可读 */ }
```

**select 的工作方式**：

- 用户传入**三组 fd 位图**（`fd_set`，每位一个 fd）。
- 内核**遍历位图**，检查每个 fd 是否就绪。
- 返回时，位图被修改为「就绪的 fd 集合」——用户用 `FD_ISSET` 逐个检查。

**select 的三个局限**：

1. **fd 数量上限**：`FD_SETSIZE` 默认 1024——最多监视 1024 个 fd（可改但有限）。
2. **O(n) 全量扫描**：每次调用内核要**遍历所有 fd** 检查就绪——fd 多时开销大。
3. **全量拷贝 + 修改**：三组位图每次调用要**从用户态拷进内核、内核改完再拷回**——大数据量下拷贝开销显著。

## 2 poll：链表取代位图

**poll**：用**数组/链表**取代位图，突破 1024 上限。

```c
struct pollfd fds[] = { {sock, POLLIN, 0}, ... };
poll(fds, nfds, timeout);
if (fds[0].revents & POLLIN) { /* sock 可读 */ }
```

**poll 的改进**：

- **无 1024 上限**：`nfds` 任意。
- **每个 fd 一个 `pollfd` 结构**：`events`（关注的事件）、`revents`（实际发生的事件）。

**poll 的局限**：

- **仍是 O(n) 扫描**：每次调用内核遍历所有 fd 检查就绪——**fd 上万时每次调用都是大扫描**。
- **仍是全量传入**：每次调用传入全部 `pollfd` 数组——**即使只有几个 fd 就绪，也要处理全部**。

**select 与 poll 的共同痛点**：**「每次调用全量传入 + 全量扫描 + 返回后全量检查」**——fd 集合大时，这些 O(n) 成本累积，成为性能瓶颈。

## 3 为什么高并发下 select/poll 不行

设监视 $N$ 个 fd，每次 select/poll 调用的开销：

$$\text{每次调用成本} \approx O(N) \quad \text{（内核扫描）} + \text{拷贝量} \approx O(N) \quad \text{（位图/数组）}$$

- $N = 10000$：每次调用扫 10000 个 fd——**即使只有 1 个就绪，也要扫全表**。
- 连接数上万、事件频繁：**每次循环都付出 O(N)**——总开销 $O(N \times \text{事件数})$，性能随连接数线性恶化。

**对比 epoll 的 O(事件数)**：epoll 只返回「就绪的 fd」——**成本与就绪数成正比，与总 fd 数无关**。这是 epoll 高并发的根本优势（见下篇）。

**公式解析：select 的拷贝开销**

```c
FD_SET 三组位图拷贝：大小 = 3 × FD_SETSIZE/8 字节 ≈ 384 字节
```

- 位图本身不大（1024 fd = 128 字节/组），**真正的痛是内核扫描**。
- 但每次 select **都重新传一遍、扫一遍**——**重复劳动**是 select 的顽疾。

**直觉**：select/poll 的问题是「**每次调用都从头再来**」——没有「记住上次的状态」。epoll 的核心创新就是「**先注册（记住）、再等待（只回报变化）**」——把 O(n) 的重复劳动变成 O(1) 的增量更新。

## 4 核心对比表：select vs poll

| 维度 | select | poll |
| --- | --- | --- |
| 数据结构 | 三组位图 | pollfd 数组 |
| fd 上限 | 1024（FD_SETSIZE） | **无硬上限** |
| 事件类型 | 读/写/异常 | 更细（POLLIN/POLLOUT...） |
| 就绪检查 | FD_ISSET | revents 字段 |
| O(n) 扫描 | 有 | 有 |
| 全量传入 | 有 | 有 |
| 跨平台 | 广（含 Windows） | 广（Unix） |

**辨析｜易错点：** 「poll 比 select 快」是常被夸大的说法。**poll 只是「突破上限 + 接口更好」，复杂度仍是 O(n) 扫描**——它与 select 一样要全量传入、全量扫描。**「poll 更快」只在「fd 超过 1024」时成立（select 用不了），不是「同样的 fd 数 poll 更快」。**

## 5 小结

- **select**：三组位图监视读/写/异常，上限 1024，O(n) 扫描 + 全量拷贝。
- **poll**：pollfd 数组突破上限，但仍是 O(n) 扫描 + 全量传入。
- 共同痛点：**每次调用「全量传入 + 全量扫描 + 返回全查」**——fd 多时成本 O(n)。
- 高并发下 select/poll 的成本随连接数线性恶化。
- epoll 的核心创新：**先注册、只回报变化**——把 O(n) 变成 O(事件数)。

在下一节，我们看 epoll 的三剑客——**epoll 三剑客：epoll_create/epoll_ctl/epoll_wait**。

---
title: epoll 三剑客：epoll_create/epoll_ctl/epoll_wait
date: 2026-08-07
---

# epoll 三剑客：epoll_create/epoll_ctl/epoll_wait

<div class="epigraph">
<p>select 每次都问「我的这些 fd 好了吗」，epoll 则说「先把 fd 交给我，好了我通知你」——一个被动查，一个主动报。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《Unix 网络编程》与 Linux epoll 手册 ｜ 2026-08-07</p>
</div>

## 为什么从 epoll 三剑客开始

上一节讲了 select/poll 的 O(n) 痛点，这一节看 epoll 如何用**三个函数**解决它：`epoll_create()`（建表）、`epoll_ctl()`（登记 fd）、`epoll_wait()`（等就绪）。epoll 的核心思想是「**先注册、只回报变化**」——把 select/poll 的「每次全量扫描」变成「注册后内核替你盯着」。<span class="marginnote">回顾 select/poll 的痛点：每次调用「全量传入 + 全量扫描」。epoll 把流程拆成<strong>三阶段</strong>——<strong>创建（一次）、注册（增量）、等待（只回报变化）</strong>。注册后 fd 就「登记在案」，内核持续盯着，就绪时只返回就绪列表。</span>

## 1 epoll 三剑客：API 概览

**① epoll_create：创建 epoll 实例**

```c
int epfd = epoll_create(1024);   /* 创建一个 epoll 实例，返回 epfd */
```

- 创建一张**事件表**（内核维护），返回 epfd。
- epfd 本身也是一个 fd——用 `close(epfd)` 关闭。

**② epoll_ctl：注册/修改/删除 fd**

```c
struct epoll_event ev;
ev.events = EPOLLIN;                          /* 关注可读事件 */
ev.data.fd = fd;
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);      /* 把 fd 登记进 epoll */
```

- 操作：`EPOLL_CTL_ADD`（添加）、`EPOLL_CTL_MOD`（修改）、`EPOLL_CTL_DEL`（删除）。
- **增量操作**——只告诉内核「新加了谁」，不用重传全部。

**③ epoll_wait：等待就绪**

```c
struct epoll_event events[MAX_EVENTS];
int n = epoll_wait(epfd, events, MAX_EVENTS, -1);  /* 阻塞等待，返回就绪数 */
for (int i = 0; i < n; i++) {
    handle(events[i].data.fd);
}
```

- **只返回「就绪的 fd」**——n 个就绪事件。
- 成本与**就绪数**成正比，与**总 fd 数**无关——这是 vs select 的本质区别。

## 2 事件驱动：epoll 的「登记-回报」模型

**epoll 的事件模型**与 select 的「查询模型」对比：

| | select/poll | epoll |
| --- | --- | --- |
| 模式 | 每次调用传全部 fd 查询 | **先注册，等待回报** |
| 内核工作 | 每次全量扫描 | 注册后持续跟踪 |
| 返回 | 全部 fd（含未就绪） | **只返回就绪 fd** |
| 复杂度 | O(总 fd 数) | **O(就绪数)** |

**典型服务端循环**：

```c
for (;;) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; i++) {
        if (events[i].events & EPOLLIN) {
            read(events[i].data.fd, buf, sizeof(buf));
        }
    }
}
```

**这个循环就是事件驱动编程（Reactor 模式）的核心**——单线程轮询 epoll，处理所有连接（见《Reactor 模式》）。

## 3 公式解析：epoll 的复杂度优势

设总 fd 数 $N$，每次等待返回就绪数 $k$：

$$\text{select/poll 每轮成本} = O(N), \qquad \text{epoll 每轮成本} = O(k)$$

- **select/poll**：每轮要检查全部 $N$ 个 fd——即使 $k=1$ 也要扫 $N$。
- **epoll**：内核维护就绪链表，就绪 fd 直接挂链——返回就绪链表 $O(k)$。
- $N = 100000$、$k = 10$：select 扫 10 万，epoll 取 10 个——**差 1 万倍**。

**epoll 的三重优化**（针对 select 的三个痛点）：

| select 痛点 | epoll 解法 |
| --- | --- |
| 全量拷贝 | 注册一次，之后不再传全集 |
| O(n) 全量扫描 | 就绪链表只扫就绪的 |
| 1024 上限 | 无上限（受 fd 数限制） |

**直觉**：epoll 的胜利在于「**把 O(n) 的重复劳动摊薄成一次注册 + O(k) 的增量回报**」——**成本从「总 fd 数」变成「就绪数」**。高并发下绝大多数 fd 大部分时间不就绪，epoll 的优势随连接数放大。

**辨析｜易错点：** 「epoll 一定比 select 快」是过度简化。**fd 少（<几百）时 select 的开销可忽略，epoll 的创建/注册反而有固定成本**——**「fd 多」才是 epoll 的用武之地**。且 epoll 是 Linux 专有（非 POSIX 标准），跨平台要加抽象层（如 libevent）。**「量大用 epoll，量小 select 也够」是务实判断。**

## 4 核心对比表：select vs epoll

| 维度 | select | epoll |
| --- | --- | --- |
| 调用模型 | 每次全量查询 | 注册 + 等待回报 |
| 复杂度 | O(N) | **O(k)**（k=就绪数） |
| fd 上限 | 1024 | **无（受系统限制）** |
| 每次拷贝 | 全量 | 无（注册一次） |
| 可移植性 | 广 | **Linux 专有** |

## 5 小结

- **epoll 三剑客**：`epoll_create()`（建表）、`epoll_ctl()`（登记 fd）、`epoll_wait()`（等就绪）。
- 模型：**先注册、只回报变化**——成本从 O(N) 降到 O(就绪数)。
- 事件驱动循环：**epoll_wait 阻塞 → 遍历就绪事件 → 处理**——Reactor 的核心。
- 三重优化：**不拷全集、不扫全表、无 1024 上限**。
- epoll 是 Linux 专有、量大才显优势——跨平台需抽象层。

在下一节，我们钻进 epoll 的内核实现——**epoll 的内核实现：红黑树与就绪链表**。

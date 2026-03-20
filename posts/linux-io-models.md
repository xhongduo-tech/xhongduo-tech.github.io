## 核心结论

Linux I/O 模型解决的是同一个问题：当程序要从文件、socket、管道等 I/O 对象里拿数据时，线程应该怎么等、内核应该怎么通知、数据什么时候真正拷贝到用户空间。

Linux 常见的五种 I/O 模型可以归纳为下表：

| 模型 | 线程怎么等 | 数据就绪后谁继续推动 | 典型接口 | 适合场景 |
| --- | --- | --- | --- | --- |
| 阻塞 I/O | 当前线程睡眠等待 | 线程被唤醒后自己 `read`/`recv` | `read` `recv` | 简单脚本、低并发 |
| 非阻塞 I/O | 不睡眠，反复试 | 线程自己轮询 | `fcntl(O_NONBLOCK)` + `read` | 连接少、临时状态机 |
| I/O 多路复用 | 在一个等待点统一等多个 fd | 线程收到事件后自己读写 | `select` `poll` `epoll_wait` | 高并发网络服务 |
| 信号驱动 I/O | 先返回，等信号通知 | 线程在信号处理或主循环里继续 | `SIGIO` | 很少单独使用 |
| 异步 I/O | 提交后立即返回 | 内核把操作做完再通知 | `aio_*`、`io_uring` | 高吞吐、低系统调用开销 |

先给结论：

1. 阻塞、非阻塞、多路复用、信号驱动，本质上都要求“应用自己把最后的读写动作做完”。
2. 真正的异步 I/O 强调“提交请求后，内核完成等待和数据搬运，再回传完成结果”。
3. `select`/`poll` 的核心问题是每次都要扫描整个 fd 集合，复杂度接近 $O(n)$。
4. `epoll` 通过“只返回已就绪 fd”把开销压到更接近 $O(k)$，其中 $k$ 是本次活跃 fd 数。
5. `io_uring` 再往前走一步：把“提交请求”和“收取结果”做成共享环，减少 syscall、减少上下文切换，并支持更像异步的批量 I/O。

交通路口的比喻可以帮助理解：

- 阻塞 I/O：你站在红灯前不动，直到灯变绿。
- 非阻塞 I/O：你每隔几秒自己抬头看一次灯。
- I/O 多路复用：后台保安同时盯着很多路口，哪个变绿就通知你。
- 信号驱动 I/O：交警直接打电话告诉你“现在可以走了”。
- 异步 I/O：你把任务交给代办，对方把事办完后通知你结果。

这个比喻只用于帮助建立直觉，正式定义仍然看“谁等待、谁搬数据、谁发完成通知”。

---

## 问题定义与边界

I/O 模型讨论的不是“磁盘快还是网卡快”，而是程序和内核围绕 I/O 事件如何配合。这里的 I/O 事件，白话说就是“数据什么时候可读、缓冲区什么时候可写、请求什么时候真正完成”。

要把问题拆成两个阶段：

1. 等待数据就绪。
2. 把数据从内核缓冲区拷贝到用户空间。

很多初学者以为“`select`/`epoll` 已经把数据读出来了”，这是错的。它们只告诉你“现在可以读了”，真正的数据搬运仍要靠 `read`/`recv`。

以“处理 100 个 socket”为例：

- `select`：每次调用都要把 100 个 fd 从头检查一遍。
- `poll`：也要扫描 100 个条目，只是数据结构比 `select` 更灵活。
- `epoll`：如果这次只有 3 个连接真有数据，就只返回那 3 个。
- `io_uring`：如果你提前提交了一批读请求，内核完成后把 3 个结果写进完成队列，你直接收结果。

可以写成简化公式：

$$
\text{select/poll 事件检查成本} \approx O(n)
$$

$$
\text{epoll/io\_uring 活跃处理成本} \approx O(k),\quad k \ll n
$$

这里的边界也很重要，不要把它们混成一个概念：

| 技术 | 解决的核心问题 | 没解决的问题 | 主要限制 |
| --- | --- | --- | --- |
| 阻塞 I/O | 代码简单 | 一个线程容易被一个连接卡住 | 并发差 |
| 非阻塞 I/O | 不会卡死在单次调用 | 会忙轮询，CPU 浪费 | 代码碎片化 |
| `select`/`poll` | 一个线程等多个 fd | 仍然要扫描全集 | 高并发下退化明显 |
| `epoll` | 避免重复全量扫描 | 仍需用户态显式读写 | ET 模式容易漏读 |
| `io_uring` | 提交/完成分离，批量异步 | 编程模型更复杂 | 依赖较新内核 |

一个玩具例子是聊天室服务器：如果只有 3 个客户端，阻塞 I/O 甚至多线程阻塞 I/O 都能工作。但如果连接数涨到 3 万，线程数、上下文切换、fd 扫描成本都会压垮吞吐，这时模型选择才真正决定架构上限。

---

## 核心机制与推导

先看 `select`/`poll` 为什么慢。

它们的共同点是：用户把一组 fd 交给内核，内核睡眠等待；一旦超时或有事件，就遍历整组 fd，把状态写回给用户。下一轮还要重新传入，再扫描一遍。

如果有 100 个 socket，其中只有 2 个活跃，那么每轮有效工作只有 2 个，但扫描工作还是 100 个。假设一秒轮询 1000 次，检查次数大约是：

$$
100 \times 1000 = 100000
$$

如果活跃 fd 只有 2 个，真正有效的比例只有：

$$
\frac{2}{100} = 2\%
$$

这就是 `select`/`poll` 在高并发空闲连接场景下效率低的根本原因。

`epoll` 的思路不同。它把流程拆成两部分：

1. `epoll_ctl` 注册关注哪些 fd 和哪些事件。
2. 内核在 fd 状态变化时，把就绪项挂入 ready list，也就是“就绪链表”。
3. `epoll_wait` 直接从 ready list 拿事件，而不是每次全表扫描。

可以用简化流程图表示：

```text
select/poll:
用户传入全集 fd -> 内核逐个检查 -> 返回就绪 fd -> 用户再次 read/write

epoll:
用户注册 fd -> 内核在事件发生时标记就绪 -> epoll_wait 取就绪集合 -> 用户 read/write

io_uring:
用户写 SQE -> 内核消费提交队列 -> 完成 I/O -> 内核写 CQE -> 用户读完成队列
```

这里的 SQE 是 Submission Queue Entry，白话说就是“一个提交请求槽位”；CQE 是 Completion Queue Entry，白话说就是“一个完成结果槽位”。

`io_uring` 的核心机制是两组共享环形队列：

- SQ：用户写入请求，内核读取。
- CQ：内核写入完成结果，用户读取。

这意味着“提交”和“收割结果”不一定都要立即进入内核。多个请求可以先批量写进 SQ，再一次性 `io_uring_enter` 提交，降低 syscall 次数。

简化伪代码如下：

```text
用户:
  sqe = get_sqe()
  sqe.op = READ
  sqe.fd = fd
  sqe.buf = buf
  sqe.len = 4096
  submit()

内核:
  从 SQ 取请求
  等待 fd 可读
  把数据拷到 buf
  生成 CQE(status, res)
  写入 CQ

用户:
  cqe = peek_cqe()
  读取 res
  recycle cqe
```

这套模型的关键不是“完全没有拷贝”，而是“提交路径更轻、可批量、某些场景可结合注册缓冲区和 zero-copy”。zero-copy，白话说就是“尽量不在不同内存区域之间重复复制同一份数据”。

因此，`epoll` 和 `io_uring` 的本质区别不是“一个先进一个落后”，而是：

- `epoll` 擅长事件通知。
- `io_uring` 擅长把事件等待、提交、完成、批量化放进一个统一框架。

真实工程例子是高吞吐数据加载服务。假设一个服务要同时从 NVMe 磁盘读取大块数据，再发往网络。如果使用 `epoll`，你通常仍要管理“磁盘读状态 + socket 可写状态 + 用户态缓冲区队列”。如果使用 `io_uring`，可以把文件读、网络发、超时控制都组织成完成队列驱动的流水线，减少线程唤醒和 syscall 压力。

---

## 代码实现

先看最简单的阻塞 I/O。阻塞的意思是“调用线程会停在系统调用里，直到结果可用”。

```python
import time

class FakeSocket:
    def __init__(self, ready_after, data):
        self.ready_after = ready_after
        self.data = data
        self.start = time.time()

    def recv_blocking(self):
        while time.time() - self.start < self.ready_after:
            time.sleep(0.01)
        return self.data

    def recv_nonblocking(self):
        if time.time() - self.start < self.ready_after:
            raise BlockingIOError("EAGAIN")
        return self.data

sock = FakeSocket(ready_after=0.05, data=b"hello")

# 阻塞读取：线程一直等到数据就绪
data = sock.recv_blocking()
assert data == b"hello"

sock2 = FakeSocket(ready_after=0.05, data=b"world")
seen_eagain = False
while True:
    try:
        data2 = sock2.recv_nonblocking()
        break
    except BlockingIOError:
        seen_eagain = True
        time.sleep(0.01)

assert seen_eagain is True
assert data2 == b"world"
```

这个玩具代码展示了两个关键事实：

1. 阻塞 I/O 简单，但线程会一直卡着。
2. 非阻塞 I/O 会返回 `EAGAIN`，白话说就是“现在还不行，稍后再试”。

再看几种常见控制流的伪代码。

阻塞 I/O：

```c
int n = read(fd, buf, sizeof(buf));   // 没数据就睡眠
handle(buf, n);
```

非阻塞 I/O：

```c
fcntl(fd, F_SETFL, O_NONBLOCK);

while (1) {
    int n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        handle(buf, n);
        break;
    }
    if (errno == EAGAIN) {
        // 还没准备好，稍后重试
        continue;
    }
    // 其他错误
}
```

`epoll` LT 模式。LT 是 Level Trigger，白话说是“只要还可读，就不断提醒你”。

```c
int epfd = epoll_create1(0);
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &(struct epoll_event){ .events = EPOLLIN });

while (1) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; i++) {
        if (events[i].data.fd == fd) {
            int m = read(fd, buf, sizeof(buf));
            if (m > 0) handle(buf, m);
        }
    }
}
```

`epoll` ET 模式。ET 是 Edge Trigger，白话说是“只在状态变化那一刻提醒一次”。

```c
fcntl(fd, F_SETFL, O_NONBLOCK);
epoll_ctl(epfd, EPOLL_CTL_ADD, fd,
          &(struct epoll_event){ .events = EPOLLIN | EPOLLET });

while (1) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; i++) {
        while (1) {
            int m = read(events[i].data.fd, buf, sizeof(buf));
            if (m > 0) {
                handle(buf, m);
                continue;
            }
            if (m == -1 && errno == EAGAIN) {
                break; // 已经读干净
            }
            if (m == 0) {
                close(events[i].data.fd);
                break;
            }
            // error
            break;
        }
    }
}
```

ET 模式必须循环读到 `EAGAIN`，否则剩余数据可能没人再提醒你。

`io_uring` 的简化伪代码如下：

```c
struct io_uring ring;
io_uring_queue_init(256, &ring, 0);

struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
io_uring_prep_recv(sqe, fd, buf, 4096, 0);
sqe->user_data = REQ_ID_1;

io_uring_submit(&ring);

struct io_uring_cqe *cqe;
int ret = io_uring_wait_cqe(&ring, &cqe);
if (ret == 0) {
    if (cqe->res > 0) {
        handle(buf, cqe->res);
    }
    io_uring_cqe_seen(&ring, cqe);
}
```

常见 syscall 与模型关系如下：

| syscall / 接口 | 所属模型 | 作用 |
| --- | --- | --- |
| `read` `write` `recv` `send` | 所有模型都会用到 | 真正执行数据读写 |
| `fcntl(O_NONBLOCK)` | 非阻塞、`epoll` ET 常配套 | 设置非阻塞 |
| `select` | 多路复用 | 等多个 fd 就绪 |
| `poll` | 多路复用 | `select` 的可变长版本 |
| `epoll_ctl` | `epoll` | 注册/修改关注事件 |
| `epoll_wait` | `epoll` | 取就绪事件 |
| `io_uring_enter` / `io_uring_submit` | `io_uring` | 提交一批请求 |
| `io_uring_wait_cqe` | `io_uring` | 获取完成结果 |

---

## 工程权衡与常见坑

在真实服务器里，重点从来不是“会不会用接口”，而是“高并发下有没有遗漏事件、有没有引入额外 CPU 开销、有没有把内核优势用出来”。

先看风险矩阵：

| 坑 | 影响 | 规避方式 |
| --- | --- | --- |
| `select` 每轮重建 fd 集合 | 连接数上来后 CPU 明显上涨 | 连接多时改用 `epoll` |
| 非阻塞 I/O 忙轮询 | 空转耗 CPU | 配合事件通知，不要裸轮询 |
| `epoll` ET 只读一次 | 数据残留在内核缓冲区，后续不再通知 | 一直读到 `EAGAIN` |
| 没把 fd 设成非阻塞却用 ET | 读循环卡死线程 | ET 必配 `O_NONBLOCK` |
| `io_uring` 队列深度太小 | 吞吐起不来 | 根据设备并发能力调 queue depth |
| `io_uring` 缓冲区管理混乱 | 内存仍被内核占用，甚至数据错乱 | 用注册缓冲区、明确回收时机 |
| 盲目追求 zero-copy | 代码复杂度大增，但收益有限 | 只在大块数据、高吞吐链路启用 |

Nginx 一类高并发服务常用 `epoll`，很多场景偏向 ET 模式，因为这样可以减少重复事件通知，降低 event loop 压力。但代价是你必须自己把 socket 一次性读干净、写尽可能多，直到返回 `EAGAIN`。

这类代码的核心模式不是“收到事件就读一次”，而是：

```c
while ((n = read(fd, buf, sizeof(buf))) > 0) {
    process(buf, n);
}
if (n == -1 && errno != EAGAIN) {
    // error
}
```

`io_uring` 的坑更多出在资源生命周期。比如注册缓冲区时，程序把一块用户内存告诉内核“以后可以直接往这里写”。如果请求还没完成你就重用或释放这块内存，结果会不可控。很多高性能系统因此会做两件事：

1. 预注册固定缓冲区，避免频繁 pin/unpin。
2. 按完成事件回收，而不是按“我感觉已经发完了”回收。

简化配置思路如下：

```c
io_uring_register_buffers(&ring, iovecs, buf_count);
/* 提交若干 recv/send/read/write 请求 */
/* 只在收到 CQE 后回收对应 buffer slot */
```

对于初级工程师，一个务实原则是：先把 `epoll` LT 写对，再学 ET；先把 `io_uring` 当“更强的异步提交框架”，不要一上来就同时引入固定文件、固定缓冲区、SQPOLL、zero-copy 全套优化。

---

## 替代方案与适用边界

没有一种 I/O 模型在所有场景都最好，只有“在当前连接规模、内核版本、维护成本约束下最合适”的选择。

先给一张决策表：

| 方案 | 性能上限 | 兼容性 | 编程复杂度 | 适合场景 |
| --- | --- | --- | --- | --- |
| 阻塞 I/O | 低 | 最高 | 最低 | 小工具、离线脚本 |
| 非阻塞 I/O | 低到中 | 高 | 中 | 少量连接、简单状态机 |
| `select` | 低到中 | 很高 | 低 | 兼容老系统、fd 少 |
| `poll` | 中 | 高 | 低 | 比 `select` 灵活，但连接不算特别多 |
| `epoll` | 高 | Linux 专属 | 中 | 大量长连接服务 |
| `io_uring` | 很高 | 依赖新内核 | 高 | 高吞吐文件/网络 I/O、批量异步流水线 |

可以把选择边界粗略理解为：

```text
连接数少 + 兼容性优先          -> select / poll
连接数多 + 事件驱动服务器      -> epoll
高吞吐 + 低 syscall 开销优先   -> io_uring
```

再给两个具体判断：

- 如果你维护的是一个内部管理后台，最多几十个连接，部署环境复杂，还要兼容旧内核，那么 `select` 或 `poll` 仍然完全合理。
- 如果你做的是网关、代理、消息服务器，核心压力来自成千上万空闲连接上的少量活跃事件，那么 `epoll` 是主流选择。
- 如果你做的是高吞吐日志写入、数据库页读取、对象存储传输，且运行环境可控、内核较新，那么 `io_uring` 值得投入，尤其在批量提交和 SQPOLL 场景下更明显。

这里的 SQPOLL 可以理解为“让一个内核轮询线程代替应用频繁提交 syscall”。它的收益来自更少的用户态到内核态切换，但前提是你的请求足够密集，否则收益未必覆盖复杂度。

因此，工程上更常见的路线不是“直接从 `select` 跳到 `io_uring`”，而是：

1. 先理解阻塞与非阻塞。
2. 再掌握 `select/poll` 和 `epoll` 的事件模型。
3. 最后在明确吞吐瓶颈后，引入 `io_uring`。

---

## 参考资料

1. Linux `man` 手册：`select(2)`、`poll(2)`、`epoll(7)`、`io_uring_enter(2)`、`io_uring_register(2)`。用途：接口语义、错误码、触发方式的官方定义。
2. 张震，《Linux 系统 I/O 模型及 select/poll/epoll 详解》，2023。用途：梳理五种 I/O 模型与 `select/poll/epoll` 的实现差异。
3. Linux kernel documentation 与 `liburing` 示例。用途：理解 `io_uring` 的 Submission Queue / Completion Queue 共享环模型。
4. Nginx 事件驱动相关文章与源码实现。用途：理解高并发服务器为何偏向 `epoll`，以及 ET 模式下循环读写到 `EAGAIN` 的工程实践。
5. io_uring 与高性能数据库/存储系统相关论文和工程博客。用途：理解批量提交、固定缓冲区、SQPOLL、zero-copy 在高吞吐场景中的收益和代价。
6. Linux 内核网络编程资料中关于 readiness notification 与 async completion 的讨论。用途：区分“事件就绪通知”和“真正异步完成”的边界。

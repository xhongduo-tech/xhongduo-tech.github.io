## 核心结论

`socket` 是操作系统提供的网络通信抽象，可以把一条 TCP 连接理解成“可读可写的文件句柄”。白话说，程序不需要直接操作网卡，而是像读文件一样读写一个编号为 `fd` 的对象。

`epoll` 是 Linux 下的事件通知机制，可以把它理解成“内核帮你盯住很多连接，只有真的有事才叫你”。它的核心价值不是让单次 `recv` 更快，而是让海量连接的等待成本更低。

对初级工程师最重要的结论有三条：

| 结论 | 直接含义 | 工程后果 |
|---|---|---|
| 默认 `socket` 是阻塞的 | `accept/recv/send` 可能让线程睡眠 | 连接多时会逼出大量线程 |
| `epoll` 适合“连接多、活跃少”的场景 | 等待交给内核，用户态只处理就绪连接 | 单线程可管理上万连接 |
| `ET` 必须配合非阻塞和“读到 `EAGAIN`” | 不把缓冲区读空，可能再也收不到通知 | 代码复杂，但空闲 CPU 更省 |

一个玩具例子：聊天室服务器有 10K 条 TCP 连接，但同一时刻真正发消息的也许只有几十个。阻塞模型下，常见做法是“一连接一线程”；`epoll` 模型下，可以用一个事件循环线程统一处理这些连接，线程数从 10000 降到 1 到几个，内存和上下文切换都会明显下降。

如果按每个线程预留约 1MB 栈空间粗略估算，那么线程模型的额外栈内存约为：

$$
M \approx N \times 1\text{MB}
$$

当 $N=10000$ 时，

$$
M \approx 10000 \times 1\text{MB} \approx 10\text{GB}
$$

这还没有算线程调度和上下文切换成本。`epoll` 的价值，首先就体现在这里。

---

## 问题定义与边界

讨论 `socket` 和 `epoll`，先把边界讲清楚。

第一，本文以 Linux 下的 TCP 服务器为主。TCP 是“面向连接、可靠传输”的协议。白话说，它会先把连接建立好，再保证数据按顺序、尽量不丢地送到对端。UDP 不走连接建立流程，使用方式和问题模型都不同，这里不展开。

第二，本文主要关心 I/O 多路复用。所谓“I/O 多路复用”，就是一个线程同时等很多个 I/O 事件，而不是每个连接单独等。白话说，就是“一个人看很多门铃，哪个门响了就去开哪个”。

第三，`epoll` 解决的是“等待谁有数据”的问题，不解决协议解析、业务计算、磁盘瓶颈、数据库慢查询这些问题。也就是说，它能降低连接管理成本，但不能替代完整的服务端架构设计。

TCP 在传数据前，先要做三次握手：

1. 客户端发 `SYN`
2. 服务端回 `SYN-ACK`
3. 客户端回 `ACK`

握手完成后，连接才进入“已建立”状态。可以把这个过程理解成：双方先确认“我能听见你、你也能听见我、序号从哪里开始记”。只有完成这一步，后面的 `recv/send` 才有意义。

下面用一个阶段图把握手和 `epoll` 注册放到一起看：

```text
客户端                                  服务端
  | ---- SYN --------------------------> |
  | <--- SYN-ACK ---------------------- |
  | ---- ACK --------------------------> |
  |                                      |
  |         连接 established             |
  |                                      |
  |            accept() 返回 fd          |
  |            setnonblocking(fd)        |
  |            epoll_ctl(ADD, fd)        |
  |                                      |
  | ---- 数据 -------------------------> |
  | <--- 数据 -------------------------- |
```

这里有一个常见误解：不是“握手前就把未来的连接放进 `epoll`”，而是监听套接字先在 `epoll` 中等新连接事件；当监听套接字可读时，调用 `accept` 拿到新的已建立连接，再把这个连接对应的 `fd` 注册进 `epoll`。对新手来说，记住一句话就够了：`epoll` 监控的是“已经存在的文件描述符”，不是“还没创建出来的未来连接”。

---

## 核心机制与推导

Linux 里，网络连接最终都落到文件描述符 `fd` 上。`epoll` 的工作方式可以概括成三步：

1. `epoll_create` 创建一个 `epoll` 实例
2. `epoll_ctl` 把要监控的 `fd` 加入兴趣列表
3. `epoll_wait` 等待就绪事件返回

可以把内核内部粗略理解成两类集合：

| 结构 | 作用 | 白话解释 |
|---|---|---|
| interest list | 记录“你关心哪些 fd、关心什么事件” | 通讯录 |
| ready list | 记录“哪些 fd 现在已经就绪” | 待办事项 |

当某个连接从“不可读”变成“可读”时，内核会把它标记为就绪；`epoll_wait` 返回后，用户态只处理这些就绪项，而不是把所有连接逐个扫一遍。

这就是 `select/poll` 与 `epoll` 的本质差别之一。前者更像“每轮考试把全班点名一遍，问谁到了”；后者更像“谁到了就把名字放到前台，老师直接看前台名单”。

### LT 与 ET

`epoll` 有两种常用触发模式。

`LT`，Level Trigger，水平触发。白话说，只要条件还成立，就反复提醒你。

`ET`，Edge Trigger，边缘触发。白话说，只在状态发生变化的那一下提醒你。

以“连接可读”为例：

- `LT`：接收缓冲区里还有没读完的数据，下一次 `epoll_wait` 还会继续告诉你“这个 fd 可读”
- `ET`：从“不可读”变成“可读”时提醒一次，之后如果你没把数据读空，通常不会重复提醒

这就是为什么 ET 模式下必须一直读到 `EAGAIN`。`EAGAIN` 的意思是“现在没有更多数据了，先别读了”。白话说，就是“这次已经榨干了，等下次再来”。

核心读循环通常长这样：

```c
while (1) {
    ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
        // 处理读到的数据
        continue;
    }
    if (n == 0) {
        // 对端关闭连接
        close(fd);
        break;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // 当前已读空，回到 epoll_wait
        break;
    }
    // 其他错误
    close(fd);
    break;
}
```

一个玩具例子能把 ET 的坑看得很清楚。

假设内核缓冲区里本来积压了 8KB 数据，程序一次 `recv` 只读了 1KB 就返回业务层处理。此时缓冲区还剩 7KB：

- 在 `LT` 下：下次 `epoll_wait` 还会继续返回这个 `fd`
- 在 `ET` 下：如果没有新的边缘变化，这个 `fd` 可能再也不通知了，7KB 就卡在缓冲区里

所以 ET 的正确心智模型不是“收到通知就读一下”，而是“收到通知就把当前能读的都读干净”。

---

## 代码实现

下面先给一个可运行的 Python 玩具程序，模拟 `LT` 和 `ET` 在“未读空缓冲区”时的差异。它不是操作真实内核，而是帮助理解触发语义。

```python
class FakeSocket:
    def __init__(self, chunks):
        self.buffer = list(chunks)
        self.ready_notified = False

    def recv(self, n):
        if not self.buffer:
            raise BlockingIOError("EAGAIN")
        data = self.buffer.pop(0)
        return data[:n]

def epoll_lt(sock):
    # LT: 只要缓冲区非空，每轮都报告可读
    return bool(sock.buffer)

def epoll_et(sock):
    # ET: 从空到非空时通知一次，读不干净也不重复通知
    if sock.buffer and not sock.ready_notified:
        sock.ready_notified = True
        return True
    return False

# 玩具例子：缓冲区里有两段数据
sock1 = FakeSocket([b"abc", b"def"])
assert epoll_lt(sock1) is True
sock1.recv(2)  # 只读一部分
assert epoll_lt(sock1) is True  # LT 还会继续通知

sock2 = FakeSocket([b"abc", b"def"])
assert epoll_et(sock2) is True
sock2.recv(2)  # 只读一部分
assert epoll_et(sock2) is False  # ET 不会再次通知，直到状态重新变化
```

这个例子只做一件事：证明“ET 下不读空，后续可能没通知”这个结论。

下面给出一个更接近真实工程的 `epoll` 服务器骨架，重点看流程而不是细节：

```c
int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
setnonblocking(listen_fd);
bind(listen_fd, ...);
listen(listen_fd, SOMAXCONN);

int epfd = epoll_create1(0);

struct epoll_event ev;
ev.events = EPOLLIN;
ev.data.fd = listen_fd;
epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

while (1) {
    struct epoll_event events[1024];
    int nready = epoll_wait(epfd, events, 1024, -1);

    for (int i = 0; i < nready; i++) {
        int fd = events[i].data.fd;

        if (fd == listen_fd) {
            while (1) {
                int conn_fd = accept(listen_fd, NULL, NULL);
                if (conn_fd < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    // 其他错误处理
                    break;
                }

                setnonblocking(conn_fd);
                struct epoll_event cev;
                cev.events = EPOLLIN | EPOLLRDHUP | EPOLLET;
                cev.data.fd = conn_fd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, conn_fd, &cev);
            }
            continue;
        }

        if (events[i].events & EPOLLRDHUP) {
            close(fd);
            continue;
        }

        if (events[i].events & EPOLLIN) {
            while (1) {
                char buf[4096];
                ssize_t n = recv(fd, buf, sizeof(buf), 0);
                if (n > 0) {
                    // 解析请求，写入业务缓冲区
                    continue;
                }
                if (n == 0) {
                    close(fd);
                    break;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                close(fd);
                break;
            }
        }

        if (events[i].events & EPOLLOUT) {
            while (has_pending_output(fd)) {
                ssize_t n = send(fd, out_ptr(fd), out_len(fd), 0);
                if (n > 0) {
                    consume_output(fd, n);
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                close(fd);
                break;
            }

            if (!has_pending_output(fd)) {
                // 写完后取消 EPOLLOUT，避免空转
                mod_epoll(epfd, fd, EPOLLIN | EPOLLRDHUP | EPOLLET);
            }
        }
    }
}
```

这里有两个实现原则最容易被忽略：

第一，监听套接字也要用非阻塞，并在 ET 下把 `accept` 循环到 `EAGAIN`。否则一次事件里只接了一个连接，剩余已完成握手的连接可能滞留在队列中。

第二，`EPOLLOUT` 不要默认常驻监听。因为大多数 TCP 连接在“当前可写”上通常成立，如果一直监听可写事件，事件循环会被无意义地频繁唤醒。正确做法是：只有发送缓冲区里确实有待发送数据时，才临时打开 `EPOLLOUT`。

真实工程例子：一个 HTTP 网关可能同时维持 5 万条长连接，但高峰时刻真正有请求包进入的连接占比并不高。实践中常见设计是“主线程或少量 I/O 线程负责 `epoll_wait` + 收发包，协议解析后把业务任务投递到工作线程池”。这样做的原因是：网络等待和业务计算是两类不同瓶颈，前者适合事件驱动，后者适合并行计算。

---

## 工程权衡与常见坑

`epoll` 不是“总是更好”，而是“在高并发连接管理上更合适”。工程里最常见的权衡集中在 LT 和 ET 的选择。

| 维度 | LT | ET |
|---|---|---|
| 编程难度 | 低 | 高 |
| 通知行为 | 条件成立就持续通知 | 状态变化时通知一次 |
| 漏读风险 | 较低 | 高，不读到 `EAGAIN` 易出错 |
| 空闲 CPU | 可能更高 | 通常更低 |
| 适合阶段 | 入门、简单服务 | 高并发、成熟事件循环 |

最常见的坑有下面几类。

第一，ET 下没有读干净或写干净。这个问题最致命，因为它不会立刻报错，而是表现为“连接偶发卡死”。排查时常见现象是：客户端明明还连着，但服务端迟迟收不到后续消息。根因通常就是某次可读事件只读了一点点。

第二，长期监听 `EPOLLOUT`。很多新手把 `EPOLLIN | EPOLLOUT` 一起长期注册，结果空闲连接也不断被唤醒。因为“可写”在很多时候天然成立，事件循环就会空转。正确策略是“按需打开、写完关闭”。

第三，忘记处理对端关闭。`recv` 返回 0 表示对端已经优雅关闭；`EPOLLRDHUP` 常用于更早感知半关闭。如果不清理连接，`fd` 泄漏会慢慢堆满进程的文件描述符上限。

第四，只关注网络，不关注应用层背压。背压就是“下游处理不过来，上游就不能无限灌数据”。白话说，如果业务线程很慢，I/O 线程再快也不能一直把消息塞进内存队列，否则会造成内存暴涨。高并发服务器设计里，网络收包、业务队列、发送缓冲必须一起考虑。

一个真实工程场景：某推送服务维持大量空闲长连接，LT 模式下把 `EPOLLOUT` 常驻打开，结果即使没有真正要发送的数据，事件循环仍频繁返回写就绪，CPU 持续偏高。改成“默认只监听 `EPOLLIN`，有待发送数据时再追加 `EPOLLOUT`，写空后立刻移除”，CPU 使用率通常会明显下降。

可以直接使用下面这份避坑清单：

| 检查项 | 为什么重要 |
|---|---|
| 所有连接都设为非阻塞 | ET 不配非阻塞几乎必出问题 |
| `accept/read/write` 都循环到 `EAGAIN` | 防止遗漏事件 |
| 写完后取消 `EPOLLOUT` | 防止事件循环空转 |
| 处理 `recv == 0` 与 `EPOLLRDHUP` | 及时释放连接 |
| 给每个连接维护收发缓冲区 | 应对半包、粘包、部分写 |
| 限制连接数、队列长度、缓冲区大小 | 防止内存被慢客户端拖垮 |

---

## 替代方案与适用边界

`select`、`poll`、`epoll` 都属于 I/O 多路复用方案，但它们的伸缩性差别很大。

| 方案 | 数据结构特点 | 每次等待前的内核交互 | 扫描成本 | 适用范围 |
|---|---|---|---|---|
| `select` | 位图集合，`fd` 数量受上限约束 | 每次都要重新传入集合 | $O(n)$ | 连接少、兼容性优先 |
| `poll` | 数组，无 `select` 那样的固定上限 | 每次都要重新传入数组 | $O(n)$ | 中等规模、实现直观 |
| `epoll` | 内核维护兴趣列表和就绪列表 | 注册一次，之后等通知 | 接近按就绪数返回 | 大量连接、高并发服务器 |

一个新手容易理解的对比是：

- `select/poll`：假设你管 1000 个连接，即使只有 3 个连接有数据，也往往要把这 1000 个都检查一遍
- `epoll`：同样 1000 个连接，如果只有 3 个就绪，`epoll_wait` 主要返回这 3 个

所以 `epoll` 的优势并不是“单次系统调用绝对更少”，而是“等待规模随活跃连接数变化，而不是随总连接数线性放大”。

但也不能把它绝对化。下面这些场景不一定要上 `epoll`：

| 场景 | 更合适的方案 |
|---|---|
| 连接数很少，逻辑简单 | 阻塞 I/O + 少量线程 |
| 跨平台要求高 | 用更高层的网络库，屏蔽平台差异 |
| 主要瓶颈在 CPU 计算，不在连接等待 | 线程池/进程池优化更关键 |
| 需要更现代的异步接口 | 可考虑 `io_uring`，但复杂度更高 |

换句话说，`epoll` 的最佳适用边界是：Linux 平台、连接数大、空闲连接多、需要精细控制性能和资源消耗的网络服务。比如网关、聊天室、即时通信、反向代理、RPC 框架底层、长连接推送服务。

如果只是写一个内部小工具，最多几十个连接，阻塞模型往往更容易写对，也更容易维护。工程上，先选“足够简单且满足规模”的方案，通常比盲目追求高性能接口更重要。

---

## 参考资料

- eXpServer, Linux epoll Tutorial: https://expserver.github.io/guides/resources/linux-epoll-tutorial.html
- eXpServer, Introduction to Linux epoll: https://expserver.github.io/guides/resources/introduction-to-linux-epoll.html
- eXpServer, Roadmap Phase 1 Stage 9: https://expserver.github.io/roadmap/phase-1/stage-9.html
- Tianbo, Epoll Trigger Mode: https://tianbo.io/2025/01/14/epoll-trigger-mode/
- SystemOverflow, Blocking vs Non-Blocking I/O, Memory and Threading Trade-offs: https://www.systemoverflow.com/learn/os-systems-fundamentals/io-models/blocking-vs-non-blocking-io-memory-and-threading-trade-offs
- MDN, TCP handshake: https://developer.mozilla.org/en-US/docs/Glossary/TCP_handshake
- Linux man-pages, `epoll(7)`: https://man7.org/linux/man-pages/man7/epoll.7.html

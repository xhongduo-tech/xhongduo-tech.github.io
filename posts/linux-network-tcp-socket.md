## 核心结论

TCP 可以理解为“内核负责可靠传输的一套状态机”。状态机就是“系统根据当前状态和收到的事件决定下一步行为”的规则表。应用程序并不直接控制三次握手、重传、乱序重组、四次挥手，而是通过 `socket` 把“我要连接、我要发送、我要关闭”交给内核，再由内核按 TCP 规则执行。

对工程实践最重要的结论有三条。

第一，TCP 的可发送数据量不是拍脑袋决定，而是同时受接收端窗口和发送端拥塞窗口限制。接收端窗口 `rcv_wnd` 是“对方还能接多少”的通告；拥塞窗口 `cwnd` 是“网络当前允许发多少”的估计。简化后可写成：

$$
\text{FlightSize} \le \min(\text{rcv\_wnd}, \text{cwnd})
$$

第二，`socket` 默认是阻塞的。阻塞就是“调用如果暂时做不成，就停在这里等”。高并发服务通常改成非阻塞，再交给 `epoll` 统一等待事件。否则一个慢连接就会卡住一个线程，线程数一高，调度和内存成本都会上升。

第三，`epoll` 的 LT 和 ET 不是“性能开关”，而是两种通知语义。LT（Level Triggered，电平触发）是“缓冲区里还有数据就反复提醒”；ET（Edge Triggered，边沿触发）是“状态发生变化时提醒一次”。ET 更省唤醒，但要求用户态在一次被唤醒后持续读写直到 `EAGAIN`，否则容易漏处理。

---

## 问题定义与边界

这篇文章只讨论 Linux 传输层到 socket API 这一段，不展开网卡驱动、ARP、路由表、HTTP 语义，也不讨论 TLS 细节。目标是回答三个基础问题：

1. 一个 TCP 连接在内核里到底经历了哪些状态。
2. 为什么 TCP 既要做流量控制，又要做拥塞控制。
3. 为什么高并发网络程序几乎都使用非阻塞 socket 加 `epoll`。

先看边界。TCP 解决的是“面向连接、可靠、有序、字节流”的传输问题。字节流就是“发送方写入的是一串字节，接收方读到的也是字节序列，没有天然消息边界”。这意味着应用层不能把一次 `send()` 和一次 `recv()` 强行对应起来。

连接生命周期的典型状态如下表。状态名可以先记住，不必一次全部吃透。

| 状态 | 含义 | 常见触发事件 |
|---|---|---|
| `CLOSED` | 没有连接 | 新建 socket、关闭完成 |
| `LISTEN` | 服务端等待新连接 | `listen()` 后进入 |
| `SYN_SENT` | 客户端已发 SYN | `connect()` 发起连接 |
| `SYN_RCVD` | 服务端收到了 SYN，已回 SYN+ACK | 等待客户端最终 ACK |
| `ESTABLISHED` | 双方连接建立完成 | 可双向收发数据 |
| `FIN_WAIT_1` | 主动关闭方已发 FIN | 等待 ACK 或对方 FIN |
| `FIN_WAIT_2` | 主动关闭方收到 ACK | 等待对方 FIN |
| `CLOSE_WAIT` | 被动关闭方收到 FIN | 应用还没真正 `close()` |
| `LAST_ACK` | 被动关闭方已发 FIN | 等待最终 ACK |
| `TIME_WAIT` | 主动关闭方等待一段时间 | 防止旧报文干扰新连接 |

一个常见误区是把“服务端调用了 `listen()`”理解成“连接已经建立”。不对。`listen()` 只是进入 `LISTEN`，表示内核开始接受新的 SYN；真正进入 `ESTABLISHED` 还要走完整个握手过程。

玩具例子可以这样理解：客户端像打电话，先说“喂，在吗”；服务端回“在，你听得到吗”；客户端再回“听得到”。三次不是礼貌问题，而是为了确认双方的收发能力和初始序列号都被对方接受。

---

## 核心机制与推导

### 1. 三次握手和四次挥手为什么存在

三次握手的核心不是“建立连接”这四个字，而是同步双方的序列空间并确认双向可达。序列号可以理解为“字节编号”。如果没有这个编号，接收方无法判断哪些字节丢了、哪些乱序了、哪些是旧连接残留的数据。

客户端典型路径是：

`CLOSED -> SYN_SENT -> ESTABLISHED`

服务端典型路径是：

`LISTEN -> SYN_RCVD -> ESTABLISHED`

四次挥手比三次握手多一步，是因为 TCP 是全双工的。全双工就是“两个方向独立关闭”。A 说“我这边发完了”对应一个 FIN，B 回 ACK；等 B 也发完，再发自己的 FIN，A 再回 ACK。

`TIME_WAIT` 值得特别强调。它不是浪费，而是防止旧连接的延迟报文污染新连接，并确保最后一个 ACK 丢失时对端还能收到重传确认。

### 2. 滑动窗口为什么能持续发送

滑动窗口可以理解为“允许未确认数据在网上飞行的区间”。它不是“发一个包等一个 ACK”，而是“在窗口允许范围内连续发送，ACK 到来后窗口向前滑动”。

最简公式是：

$$
\text{SendWindow} = \min(\text{rcv\_wnd}, \text{cwnd})
$$

其中：

- `rcv_wnd`：接收端剩余缓冲能力。
- `cwnd`：发送端对网络承载能力的估计。

玩具例子：

接收端通告窗口为 5000 字节，发送端每个 TCP 段按 1000 字节发送。此时最多可并行飞行 5 段。假设发送端先发了 3 段，当第 1 段 ACK 到达时，窗口左边界右移，发送端马上可以补发第 4 段，而不需要等第 2、第 3 段都确认完。

这就是“流水线发送”。它把等待 RTT 的空档填满，从而提高吞吐。

### 3. 流量控制和拥塞控制为什么不能混为一谈

流量控制管的是“接收方吃不吃得下”，拥塞控制管的是“网络扛不扛得住”。

如果只有流量控制，没有拥塞控制，那么接收方缓冲很大时，发送端可能把中间链路打爆。  
如果只有拥塞控制，没有流量控制，那么网络没问题时，发送端仍可能把慢接收端淹没。

Linux 常见拥塞控制算法有 CUBIC 和 BBR。

CUBIC 的核心增长函数可以写成：

$$
W_{cubic}(t) = C(t-K)^3 + W_{max}
$$

其中 $W_{max}$ 是上次拥塞前的窗口峰值，$K$ 决定曲线拐点。它的特点是对长距离、高带宽链路更友好，窗口增长以时间为自变量，不完全依赖 ACK 频率。

BBR 的思路不同。它不主要把“丢包”当作拥塞信号，而是估计瓶颈带宽和最小 RTT。可以粗略记成：

$$
\text{cwnd} \approx \text{bottleneck\_bw} \times RTT
$$

这接近链路的 BDP（带宽时延积）。BDP 就是“让链路刚好装满需要多少在途数据”。BBR 因此更强调 pacing，也就是“按速率平滑发送”，而不是一股脑把包塞出去。

### 4. 用户态和内核如何配合

内核负责 TCP 状态、重传、窗口、拥塞控制；用户态负责何时调用 `read`/`write`/`accept`/`connect`。两边通过 socket 缓冲区和事件通知配合。

这也是为什么 `epoll` 不会替你“读完数据”。它只告诉你“现在值得读了”。真正把内核缓冲区搬到用户缓冲区，仍然要靠你的代码循环读取。

真实工程例子：Nginx、Redis 这一类高并发服务通常把监听 socket 和连接 socket 都设为非阻塞，再交给 `epoll`。这样一个线程可以管理大量连接，内核继续负责 TCP 可靠性，用户态只做事件驱动调度。连接数上来后，系统瓶颈往往转向业务逻辑、内存分配、上下文切换和网卡吞吐，而不是“握手怎么写”。

---

## 代码实现

先用一个可运行的 Python 小程序模拟“窗口推进”的核心逻辑。它不是完整 TCP，只是帮助你验证滑动窗口为什么能在 ACK 到来后继续补发。

```python
def sliding_window_send(total_segments, seg_size, rcv_wnd, cwnd, acked_sequence):
    window_bytes = min(rcv_wnd, cwnd)
    max_inflight = window_bytes // seg_size
    sent = 0
    inflight = 0
    history = []

    # 初始尽量发满窗口
    while sent < total_segments and inflight < max_inflight:
        sent += 1
        inflight += 1
        history.append(f"send-{sent}")

    # 每收到一个 ACK，就释放一个位置并继续发送
    for ack in acked_sequence:
        if inflight > 0:
            inflight -= 1
            history.append(f"ack-{ack}")
        while sent < total_segments and inflight < max_inflight:
            sent += 1
            inflight += 1
            history.append(f"send-{sent}")

    return history, max_inflight


history, max_inflight = sliding_window_send(
    total_segments=6,
    seg_size=1000,
    rcv_wnd=5000,
    cwnd=3000,
    acked_sequence=[1, 2, 3]
)

assert max_inflight == 3
assert history[:3] == ["send-1", "send-2", "send-3"]
assert "ack-1" in history
assert "send-4" in history
print(history)
```

上面例子里 `rcv_wnd=5000`，`cwnd=3000`，因此有效发送窗口是 3000 字节，只能同时飞行 3 个 1000 字节段。收到第一个 ACK 后，马上补发第 4 段，这就是窗口滑动。

下面看 Linux 风格的非阻塞 `connect + epoll` 伪代码。重点不是语法，而是时序。

```python
# 伪代码，展示关键流程
sock = socket()
set_nonblocking(sock)

ret = connect(sock, server_addr)
if ret == 0:
    state = "connected"
elif errno == EINPROGRESS:
    epoll_ctl_add(epfd, sock, EPOLLOUT | EPOLLIN | EPOLLET)
    state = "connecting"
else:
    raise ConnectError()

while True:
    events = epoll_wait(epfd)

    for ev in events:
        if ev.writable:
            err = getsockopt(sock, SOL_SOCKET, SO_ERROR)
            if err == 0 and state == "connecting":
                state = "connected"

            while True:
                n = write(sock, outbuf)
                if n > 0:
                    consume_outbuf(n)
                elif errno == EAGAIN:
                    break
                else:
                    close(sock)

        if ev.readable:
            while True:
                n = read(sock, inbuf)
                if n > 0:
                    process(inbuf[:n])
                elif n == 0:
                    close(sock)   # 对端关闭
                    break
                elif errno == EAGAIN:
                    break
                else:
                    close(sock)
                    break
```

这段流程有三个初学者必须记住的点。

第一，非阻塞 `connect()` 返回 `EINPROGRESS` 不表示失败，而是“连接还在进行中”。真正是否成功，要等 `EPOLLOUT` 后再用 `getsockopt(SO_ERROR)` 确认。

第二，ET 模式下必须循环读到 `EAGAIN`。`EAGAIN` 可以理解为“现在资源暂时不可用，你先别忙了”。在非阻塞 socket 上，它不是异常，而是一次正常的停止信号。

第三，非阻塞 `write()` 可能只写入一部分数据。部分写入就是“你给我 100KB，我现在内核发送缓冲只塞得下 32KB”。剩余数据必须继续保存在应用层发送缓冲，等下次可写事件再发。

LT 和 ET 的差异可以用表格直接记忆：

| 模式 | 通知方式 | 是否要读到 `EAGAIN` | 适合场景 |
|---|---|---|---|
| LT | 只要缓冲区仍可读/可写，就可能继续通知 | 建议读尽，但漏读通常还有下次提醒 | 逻辑简单、易调试 |
| ET | 只在状态从“不可用”变“可用”时通知一次 | 必须读写到 `EAGAIN` | 高并发、追求更少唤醒 |

---

## 工程权衡与常见坑

最常见的坑不是“不会写三次握手”，而是错判内核和用户态的分工。

第一类坑是 ET 没有读尽。结果是内核缓冲区里明明还有数据，但因为状态没有再次变化，应用再也等不到下一次通知。这个问题在压测下很典型，表现为“连接没断、CPU 不高、就是偶发卡死”。

第二类坑是把非阻塞写当成阻塞写来用。`send()` 返回了 20KB，并不代表你的 100KB 消息已经发送完成。应用层如果没有维护发送队列，就会造成截断、乱序拼接或协议层解析失败。

第三类坑是误解 `CLOSE_WAIT`。`CLOSE_WAIT` 持续很久，往往不是内核没回收，而是应用收到了对端 FIN 后，没有及时调用 `close()`。这类问题常常意味着连接泄漏。

第四类坑是把 CUBIC 和 BBR 看成“哪个更先进就全局切哪个”。不对。拥塞控制算法的收益依赖链路特征、RTT、丢包模式、队列管理策略以及应用是否长期跑满链路。

常见错误和规避方式如下表：

| 常见错误 | 直接后果 | 规避措施 |
|---|---|---|
| ET 模式下只读一次 | 缓冲区残留数据，后续可能不再通知 | 循环 `read()` 直到 `EAGAIN` |
| 非阻塞写未处理部分写入 | 消息截断或协议错位 | 维护发送缓冲和写偏移 |
| `connect()` 返回 `EINPROGRESS` 就判失败 | 误报连接失败 | 等 `EPOLLOUT` 后检查 `SO_ERROR` |
| 长期 `CLOSE_WAIT` | fd 泄漏、连接堆积 | 收到 EOF 后走完整关闭流程 |
| 只看 `rcv_wnd` 不看 `cwnd` | 高估可发送量 | 用 `min(rcv_wnd, cwnd)` 理解有效窗口 |
| BBR 场景下忽略 app-limited | 带宽估计偏低 | 识别应用未打满链路的测量区间 |

再看一个真实工程判断：如果你的服务是管理后台、几十到几百连接、逻辑主要花在数据库或磁盘 IO 上，那么阻塞 socket 加线程池可能已经够用，代码也更直。  
但如果你的服务是代理、网关、缓存、消息转发这类“连接多、单连接业务轻、网络 IO 占主导”的场景，非阻塞加 `epoll` 几乎是默认选择。

---

## 替代方案与适用边界

阻塞、非阻塞、`select`、`epoll LT`、`epoll ET` 之间没有绝对优劣，只有规模和复杂度上的取舍。

| 方案 | 优点 | 缺点 | 典型场景 |
|---|---|---|---|
| 阻塞 + 单线程 | 最易理解 | 一个慢连接卡住整体 | 教学、一次只处理一个连接 |
| 阻塞 + 多线程 | 编程模型直观 | 线程多时内存和调度成本高 | 中低并发内部工具 |
| `select`/`poll` | 兼容性好 | 扫描开销更高 | 老系统、小规模多路复用 |
| `epoll LT` | 易写、稳妥 | 唤醒次数更多 | 大多数通用 Linux 服务 |
| `epoll ET` | 更少重复通知 | 编程要求高，易出错 | 高并发、性能敏感服务 |

可以给出一个经验边界，而不是僵硬阈值：

- 连接数很少时，优先选最简单的模型。
- 连接数到几千以上，通常就要认真考虑事件驱动。
- 当业务处理远比网络 IO 重时，ET 的收益可能不明显。
- 当网络 IO 本身就是主成本时，ET、批量收发、零拷贝等优化才更值得。

拥塞控制同样如此。

| 算法 | 优势 | 风险或限制 | 更适合 |
|---|---|---|---|
| CUBIC | Linux 默认、兼容性强、长肥管道表现稳定 | 某些场景排队延迟更高 | 通用互联网服务 |
| BBR | 常能降低排队时延、提升链路利用率 | 依赖准确带宽/RTT估计，部署需观察公平性 | 长距离、高带宽、低延迟敏感业务 |

因此，替代方案不是“谁淘汰谁”，而是“哪种复杂度换来哪种收益”。对零基础工程师来说，最稳妥的学习顺序通常是：

1. 先理解 TCP 状态机和滑动窗口。
2. 再掌握阻塞与非阻塞差异。
3. 再理解 `epoll LT` 和 `ET` 的通知语义。
4. 最后再去研究 CUBIC、BBR、pacing、队列管理这些性能主题。

---

## 参考资料

1. Unix Network Programming：TCP 状态图与连接生命周期，适合建立整体状态机认识。  
2. RFC 9438：CUBIC 正式规范，适合理解窗口增长公式和设计目标。  
3. IETF BBR draft：BBR 的带宽模型、ProbeBW/ProbeRTT 状态与 pacing 思路。  
4. `man 7 epoll`：Linux `epoll` 的官方语义说明，尤其是 LT/ET 差异。  
5. Blocking & Non-Blocking Sockets 指南：非阻塞 socket、`EINPROGRESS`、`EAGAIN` 的入门解释。  
6. TCP Flow Control 相关文章：用例子解释 `rcv_wnd`、ACK 推进与滑动窗口。

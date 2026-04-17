## 核心结论

TCP 是传输层协议，传输层就是“负责让两台机器上的程序互相收发数据”的这一层。它的目标不是尽快发出去，而是尽量保证数据按序、完整、可确认地到达。为此，TCP 把可靠性拆成四组机制：三次握手建立连接、滑动窗口控制发送量、拥塞控制适应网络容量、四次挥手安全关闭连接。

UDP 也是传输层协议，但它是“尽快送出，不负责善后”的模型。它没有连接状态，不做重传，不保证顺序，也不感知拥塞。代价是应用自己承担可靠性；收益是延迟低、实现简单、适合实时场景。

理解 TCP 最重要的公式是：

$$
SW = \min(cwnd, rwnd)
$$

其中 `SW` 是发送窗口，表示发送方当前最多还能发多少未确认数据；`cwnd` 是拥塞窗口，白话讲就是“网络此刻允许你发多少”；`rwnd` 是接收窗口，白话讲就是“对方缓冲区还能接多少”。TCP 的真实发送能力，取这两个约束中的较小值。

对初级工程师，结论可以直接记成三句话：

| 主题 | 结论 |
| --- | --- |
| 建连 | TCP 需要三次握手，因为双方要同步“我是谁、你是谁、初始序号是多少、你能接多少” |
| 传输 | TCP 不是一直猛发，而是受 `cwnd` 和 `rwnd` 双重限制 |
| 关连 | TCP 用四次挥手和 `TIME_WAIT` 防止旧报文污染新连接 |

玩具例子：客户端给服务端发一句 `"hello"`，如果第一段丢了，TCP 会重传，应用通常无感知；如果用 UDP，应用层必须自己决定要不要重发、怎么去重、如何排序。

真实工程例子：Web 服务重启时，旧连接可能仍停留在 `TIME_WAIT`，导致新进程短时间内无法重新 `bind` 监听端口。这时通常会配置 `SO_REUSEADDR`，必要时再结合连接池、长连接或系统级端口复用策略处理。

---

## 问题定义与边界

本文讨论的是 TCP/IP 协议栈里“传输层”的核心机制。TCP/IP 协议栈就是“从应用到网络传输的一套分层规则”。这里重点放在 TCP 与 UDP，不展开物理层、链路层、ARP、交换机转发等主题，也不深入 Linux 内核源码实现。

问题可以分成两类：

1. 为什么 TCP 需要这么多控制机制？
2. 在工程里，什么时候该用 TCP，什么时候该用 UDP？

先给出边界。TCP 解决的是“可靠字节流”问题。字节流就是“应用看到的是连续字节，而不是离散消息包”。它关注的是：
- 建立双方一致的连接状态
- 让数据尽量不丢、不乱序、不重复
- 让快的发送方不要压垮慢的接收方
- 让单个连接不要把网络挤爆

UDP 的边界完全不同。它解决的是“尽快把一个报文发出去”，不承诺送达质量。它关注的是低开销、低延迟、无连接和多播能力。

下面这个表格可以把边界看清楚：

| 维度 | TCP：有控制 | UDP：少控制 |
| --- | --- | --- |
| 连接状态 | 有，需握手和挥手 | 无，直接发送 |
| 可靠性 | 有确认、重传、排序 | 默认无 |
| 延迟 | 较高，因握手和控制开销 | 较低 |
| 流量控制 | 有，靠 `rwnd` | 无 |
| 拥塞控制 | 有，靠 `cwnd` | 无 |
| 典型场景 | 浏览器请求、文件传输、数据库连接 | 实时语音、直播、在线游戏、DNS |

为什么三次握手不能省？核心原因不是“礼貌”，而是“同步状态”。TCP 每个方向都要维护序号，序号就是“当前这条连接里，哪一字节算作第几个字节”。如果客户端只发一次 SYN 就开始传数据，服务端可能不知道客户端是否收到自己的确认，也无法确认双方是否对初始序号达成一致。三次握手的目标，就是让双方都确认：

- 我能发给你
- 你能发给我
- 我知道你的初始序号
- 你知道我的初始序号

如果这一步不完整，后面就无法判断一个 ACK 是属于旧连接还是新连接，也无法安全重传。

---

## 核心机制与推导

三次握手的最小流程如下：

```text
客户端                    服务端
  | ---- SYN(seq=x) ----> |
  | <--- SYN+ACK(y,x+1) --|
  | ---- ACK(y+1) ------> |
```

- `SYN` 是同步序号标志，意思是“我要建连，并告诉你我的初始序号”
- `ACK` 是确认标志，意思是“你前面的数据我已经收到”
- `seq` 是本报文的起始序号
- `ack` 是“我下一次期望收到的序号”

为什么是三次而不是两次？因为服务端不仅要回“我收到了你的 SYN”，还要把“我的初始序号”告诉客户端；客户端最后那个 ACK 再证明“你的这个初始序号我也收到了”。双向信息都要闭环。

四次挥手则是因为 TCP 是全双工。全双工就是“两个方向都可以独立发送”。关闭连接时，一个方向说“我不再发了”，不等于另一个方向也立刻不发，所以常见流程是：

```text
客户端                    服务端
  | ---- FIN -----------> |
  | <--- ACK ------------ |
  | <--- FIN ------------ |
  | ---- ACK -----------> |
```

`FIN` 表示“我这边数据发完了”。之所以通常是四次，是因为“确认对方关闭”和“自己也关闭”是两个独立动作。

`TIME_WAIT` 是很多人第一次上线排障时才真正理解的状态。它是“主动关闭连接的一方在最后一个 ACK 发出后，仍保留一段时间的状态”。保留的目的有两个：

1. 保证最后一个 ACK 丢失时，对方还能重发 `FIN`，本端可以再次回复。
2. 等待网络中的旧报文自然过期，避免它们跑到后续重建的新连接里。

这就是 `TIME_WAIT` 的根本价值：防旧报文干扰。

接着看滑动窗口。发送方不能无限发，而是要满足：

$$
SW = \min(cwnd, rwnd)
$$

含义很直接：
- `rwnd` 限制“对方接不接得住”
- `cwnd` 限制“网络扛不扛得住”

如果接收方很慢，即使网络很空，`rwnd` 也会把发送速率压下来；如果接收方很快，但网络发生拥塞，`cwnd` 也会收缩。

玩具数值例子：假设 MSS 为 1460B。MSS 就是“单个 TCP 段里可承载的最大应用数据长度”。设：
- `rwnd = 10 MSS`
- 初始 `cwnd = 1 MSS`
- `ssthresh = 4 MSS`

那么一开始：

$$
SW = \min(1, 10) = 1 \text{ MSS}
$$

第 1 个 RTT 后收到 ACK，慢启动阶段：

$$
cwnd \leftarrow cwnd + MSS
$$

由于每收到一个 ACK 都加一个 MSS，而一个 RTT 内 ACK 数量近似等于已发送段数，所以按 RTT 看，`cwnd` 近似翻倍：

- RTT1 结束：`cwnd = 2 MSS`
- RTT2 结束：`cwnd = 4 MSS`

达到阈值后进入拥塞避免。拥塞避免就是“不要再指数增长，改成线性探测”。其经典近似写法是：

$$
cwnd \leftarrow cwnd + \frac{MSS^2}{cwnd}
$$

按 RTT 观察，相当于每个 RTT 大约只增加 `1 MSS`：

- RTT3 结束：`cwnd ≈ 5 MSS`
- RTT4 结束：`cwnd ≈ 6 MSS`

如果这时丢包，有两种典型反应。

第一种，超时重传。超时说明网络问题可能较严重，TCP 会更保守：

$$
ssthresh \leftarrow \max(\frac{SW}{2}, 2 \cdot MSS)
$$

$$
cwnd \leftarrow 1 \cdot MSS
$$

也就是阈值砍半，窗口回到 1，从慢启动重新探测。

第二种，重复 ACK 触发快重传。重复 ACK 就是“接收方反复确认同一个序号”，白话讲是“后面的包到了，但中间某个包没到”。这说明网络还在工作，只是出现了局部丢包，所以不必像超时那样把 `cwnd` 直接打回 1。此时会进入快重传/快恢复，更温和地调整窗口。

慢启动和拥塞避免的区别可以直接看表：

| 阶段 | 目标 | 增长规律 | 典型触发 |
| --- | --- | --- | --- |
| 慢启动 | 快速探测可用带宽 | 按 RTT 近似翻倍 | 新连接开始、超时后恢复 |
| 拥塞避免 | 平稳增加吞吐 | 按 RTT 近似 +1 MSS | `cwnd` 达到 `ssthresh` 后 |
| 快重传/快恢复 | 处理局部丢包 | 不回到最小值，温和收缩 | 多个重复 ACK |

把这些机制连起来，TCP 的逻辑其实很统一：先安全建连，再谨慎放量，发现拥塞就收缩，最后安全收尾。

---

## 代码实现

先看一个最小发送侧模型。它不是完整 TCP，而是把“发送窗口受 `cwnd`/`rwnd` 共同约束”这件事写成可运行代码。

```python
MSS = 1460

def send_window(cwnd, rwnd):
    return min(cwnd, rwnd)

def on_ack_slow_start(cwnd):
    return cwnd + MSS

def on_ack_congestion_avoidance(cwnd):
    # 经典近似：每收到一个 ACK 增长 MSS^2 / cwnd
    return cwnd + (MSS * MSS) // cwnd

def on_timeout(sw):
    ssthresh = max(sw // 2, 2 * MSS)
    cwnd = MSS
    return cwnd, ssthresh

def simulate():
    rwnd = 10 * MSS
    cwnd = 1 * MSS
    ssthresh = 4 * MSS

    history = []

    # RTT1: 慢启动
    sw = send_window(cwnd, rwnd)
    history.append(sw)
    cwnd = on_ack_slow_start(cwnd)

    # RTT2: 慢启动
    sw = send_window(cwnd, rwnd)
    history.append(sw)
    cwnd = on_ack_slow_start(cwnd)

    # 达到阈值后进入拥塞避免
    sw = send_window(cwnd, rwnd)
    history.append(sw)
    cwnd = on_ack_congestion_avoidance(cwnd)

    # 假设发生超时
    sw = send_window(cwnd, rwnd)
    new_cwnd, new_ssthresh = on_timeout(sw)

    assert history[0] == 1 * MSS
    assert history[1] == 2 * MSS
    assert history[2] == 4 * MSS
    assert new_cwnd == 1 * MSS
    assert new_ssthresh >= 2 * MSS

    return history, new_cwnd, new_ssthresh

result = simulate()
assert result[0] == [1460, 2920, 5840]
print(result)
```

这个玩具实现只表达三件事：

- 当前能发多少，不看心情，只看 `min(cwnd, rwnd)`
- 慢启动先快涨
- 超时后必须显著收缩

如果把它写成更接近发送循环的伪代码，结构通常是这样：

```python
while bytes_left > 0:
    sw = min(cwnd, rwnd)
    if bytes_in_flight < sw:
        send_segment()
        bytes_in_flight += MSS

    event = wait_event()

    if event == "ACK":
        bytes_in_flight -= acked_bytes
        if cwnd < ssthresh:
            cwnd += MSS
        else:
            cwnd += MSS * MSS / cwnd

    elif event == "TIMEOUT":
        ssthresh = max(sw / 2, 2 * MSS)
        cwnd = MSS
        retransmit_lost_segment()

    elif event == "DUP_ACK_3":
        ssthresh = max(sw / 2, 2 * MSS)
        fast_retransmit()
```

真实工程里，应用一般不自己实现 TCP，而是调用操作系统的 socket。下面是一个接近 Python 标准库接口的示意：

```python
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", 8080))
server.listen(128)

conn, addr = server.accept()
data = conn.recv(4096)
conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
conn.close()
server.close()
```

`SO_REUSEADDR` 的作用要说准：它主要用于“允许本地地址更快重新绑定”，常见于服务重启场景。它不是“无条件复用所有旧连接”，也不是“绕过 TCP 状态机”。如果本地端口还被其他活动监听占用，或平台语义不同，仍可能不能复用。

真实工程例子：一个短连接 Web 服务每秒建立和关闭大量 TCP 连接。如果服务进程异常退出后立刻重启，端口上可能仍存在大量 `TIME_WAIT` 记录。没有 `SO_REUSEADDR` 时，`bind()` 可能失败；加上它后，监听 socket 通常可以更快恢复服务。

---

## 工程权衡与常见坑

TCP 的可靠性不是免费午餐，它带来状态、内存、时延和排障复杂度。新手最常见的坑有三类。

第一类是 `TIME_WAIT` 误解。很多人看到大量 `TIME_WAIT` 就以为“系统出故障了”。其实它通常说明“这里有很多主动关闭的 TCP 连接”。问题不在于状态本身，而在于它是否造成端口耗尽、连接回收慢或服务重启失败。

第二类是把 `SO_REUSEADDR` 当万能开关。它解决的是“监听地址重绑定”这一类问题，不会消除高频短连接带来的所有副作用。真正的优化路径通常是：
- 减少不必要的主动关闭
- 用长连接或连接池降低建连/挥手频率
- 对高并发短连接场景，再评估系统级复用参数

第三类是只看应用吞吐，不看拥塞控制副作用。比如批量任务在短时间同时发起大量 TCP 连接，可能造成突发拥塞，结果整体吞吐反而下降，尾延迟上升。

常见坑与规避如下：

| 问题 | 现象 | 根因 | 常见规避 |
| --- | --- | --- | --- |
| `TIME_WAIT` 过多 | 端口占用、重启后难以绑定 | 短连接过多，主动关闭频繁 | 长连接、连接池、`SO_REUSEADDR` |
| 重复 ACK 很多 | 吞吐下降、重传增多 | 局部丢包或乱序 | 观察链路质量、调优队列与拥塞参数 |
| 拥塞爆发 | RTT 抖动、丢包集中 | 多连接同时增大发送速率 | 限流、平滑启动、连接复用 |
| 误用 UDP | 数据乱序、丢消息 | 应用没补齐可靠性逻辑 | 在应用层补 ACK、重传、去重 |
| 误以为 TCP 保证消息边界 | 一次 `recv` 读不全或读多条 | TCP 是字节流，不是消息队列 | 自定义长度字段或分隔符协议 |

这里再给一个新手容易遇到的真实例子。某个 Web 服务监听 `80` 或 `8080` 端口，运维脚本执行“停止后立即启动”。如果旧进程关闭时留下大量 `TIME_WAIT`，新进程可能在 `bind()` 阶段报地址占用。此时：
- `SO_REUSEADDR` 常用于加快监听恢复
- 长连接可以减少短时间内的大量关闭
- 连接池可以减少每次请求都重新建连
- 盲目缩短状态保留时间则可能引入旧报文污染风险

工程上，核心不是“消灭 `TIME_WAIT`”，而是“理解它为什么存在，并只在合适边界内优化”。

---

## 替代方案与适用边界

TCP 和 UDP 不是谁先进谁落后，而是目标函数不同。

如果业务首要目标是完整性，TCP 仍然是默认选择。文件下载、网页访问、数据库协议、消息队列，大多数都更关心“不能错、不能乱、最好别丢”，因此适合 TCP。

如果业务首要目标是时效性，UDP 更有优势。实时语音就是典型例子。晚到 300 毫秒的语音包，很多时候还不如直接丢掉，因为人耳更在意连续性而不是每个包都补回来。在线游戏也类似，角色当前位置比 1 秒前某一帧的位置更有价值。

但“用 UDP”不等于“不需要可靠性”。很多现代协议会走“UDP + 应用层控制”路线，代表就是 QUIC。它把连接管理、重传、流控等逻辑搬到用户态协议层，换取更快的迭代和更灵活的行为。可以把它理解成：底层借用 UDP 的无连接低延迟外壳，上层再重建需要的可靠机制。

下面这个表格适合做选型起点：

| 场景 | 更适合 TCP | 更适合 UDP |
| --- | --- | --- |
| 文件传输 | 是，完整性优先 | 否 |
| 浏览器请求 | 是，生态成熟且可靠 | 否 |
| 数据库连接 | 是，顺序和一致性重要 | 否 |
| 实时语音 | 一般否 | 是，低延迟优先 |
| 直播互动 | 一般否 | 是，容忍少量丢包 |
| 在线游戏状态同步 | 视协议设计而定 | 常见选择 |
| 多播/广播 | 不支持 | 支持，适合发现类或分发类场景 |

`tcp_tw_reuse` 和 `SO_REUSEADDR` 这类参数，属于操作系统级权衡。它们更适合短连接、高并发、端口复用压力明显的场景，不适合在不了解风险时机械开启。原因很简单：它们优化的是“状态复用速度”，而 TCP 维护这些状态本来就是为了避免语义错误。任何绕过都必须建立在充分理解之上。

一句话总结适用边界：
- 文件下载、网页、数据库，优先 TCP
- 实时语音、直播、游戏状态同步，优先 UDP 或基于 UDP 的上层协议
- 高交互、高性能场景，可评估 QUIC 一类替代方案
- 只要你不想自己处理丢包、重排、重传，默认先选 TCP

---

## 参考资料

1. MDN, TCP handshake: https://developer.mozilla.org/en-US/docs/Glossary/TCP_handshake
2. RFC 2001, TCP Slow Start, Congestion Avoidance, Fast Retransmit, and Fast Recovery: https://www.rfc-editor.org/rfc/rfc2001.html
3. IBM Cloud 文档, TCP flow control and sliding window
4. LinuxVox, `SO_REUSEADDR` on Linux: https://linuxvox.com/blog/what-is-the-meaning-of-so-reuseaddr-setsockopt-option-linux/
5. Connected, TIME_WAIT state explanation: https://www.connected.app/library/tcp-and-udp/tcp/articles/time-wait-state-idoy6cu
6. Linode 指南, TCP vs UDP 对比
7. RFC 793, Transmission Control Protocol
8. RFC 1122, Requirements for Internet Hosts

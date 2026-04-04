## 核心结论

Linux 网络协议栈可以理解为一条固定的数据通路：用户态通过 `socket` 发起 `send()` / `recv()`，内核把数据包装进 `sk_buff`。`sk_buff` 可以理解为“内核里表示一个网络包的统一对象”，后续 TCP、IP、设备驱动都围绕它处理。发送时，它一路经过传输层、网络层、设备队列，最后进入网卡的环形描述符；接收时，路径反过来，由网卡触发中断和软中断，再把包交回 socket。

对工程性能影响最大的，不只是“协议是否高效”，而是三件事是否同时成立：队列是否合理、SoftIRQ 是否扛得住、拷贝路径是否过长。`epoll` 只是把“发现哪个连接可读可写”的成本降下来，不能替代内核收发路径本身的优化。

下表先给出关键路径：

| 阶段 | 作用 | 典型对象 | 主要优化点 |
|---|---|---|---|
| 用户态系统调用 | 发起收发 | `send()` / `recv()` | 减少无意义 syscall 次数 |
| socket 层 | 连接状态与缓冲区管理 | `struct sock` | 调整收发 buffer |
| 传输层 | 可靠传输、重传、窗口 | TCP | MSS、窗口、拥塞控制 |
| 网络层 | 路由、分片、IP 头 | IP | 路由缓存、避免分片 |
| 设备队列 | 等待驱动发包 | qdisc / tx queue | 队列长度、BQL |
| 驱动与网卡 | DMA 与描述符收发 | ring buffer | 中断合并、RSS、GSO |

一个新手版玩具流程如下：

```text
用户进程 send()
  -> 内核创建/填充 sk_buff
  -> TCP 加头并排入发送队列
  -> IP 选路由与下一跳
  -> 驱动 ndo_start_xmit()
  -> NIC 发送环形缓冲区
  -> 网线
```

---

## 问题定义与边界

本文只讨论 Linux 原生网络栈，也就是 socket、`sk_buff`、TCP/IP、SoftIRQ、驱动队列和网卡之间的配合关系。SoftIRQ 是“内核延后执行的一类软中断任务”，网络收发大量依赖它来避免在硬中断里做太重的工作。

边界要先说清楚，否则容易把问题谈散：

| 项目 | 内容 |
|---|---|
| 关注层 | socket 接口、TCP/IP、队列、SoftIRQ、驱动、NIC |
| 排除层 | HTTP 语义、RPC 框架、应用业务逻辑、数据库 |
| 关键指标 | 吞吐、尾延迟、丢包率、重传率、CPU 占用 |
| 主要对象 | `sk_buff`、socket buffer、qdisc、ring buffer |
| 典型场景 | Linux 上的高并发 HTTP / RPC 服务器 |

真实边界例子：一个高并发 HTTP 服务器有 5 万个连接，业务线程只关心“哪个 fd 可读可写”，这部分是 `epoll` 负责；但真正决定 99 线延迟的，常常是内核里收包 backlog 是否溢出、`ksoftirqd` 是否被打满、网卡队列是否堆积，而不是 HTTP 路由代码本身。

所以，本文的问题定义不是“如何写一个 Web 服务”，而是“Linux 怎样把一个包可靠、低延迟地穿过内核与网卡，并在高并发下尽量不堵住”。

---

## 核心机制与推导

`sk_buff` 是主线。它像一个“带元数据的包容器”，既包含实际数据，也记录协议头位置、长度、校验信息、设备信息。发送路径里，`tcp_sendmsg()` 会把用户数据拷到内核缓冲，再组织成一个或多个 `sk_buff`；接收路径里，驱动把 DMA 收到的数据挂到 `sk_buff`，交给上层继续处理。

### 1. 发送与接收的分层流动

发送方向：

```text
用户态 buffer
  -> socket 发送缓冲
  -> sk_buff
  -> TCP 输出队列
  -> IP 路由
  -> qdisc / dev queue
  -> 驱动 ring
  -> NIC DMA 发出
```

接收方向：

```text
NIC 收包
  -> 硬中断确认
  -> NET_RX_SOFTIRQ
  -> 驱动分配/回填 sk_buff
  -> IP 校验与路由判断
  -> TCP 按序重组、ACK
  -> socket 接收缓冲
  -> 用户态 recv()
```

### 2. 滑动窗口

滑动窗口可以理解为“允许发送端在未收到确认前，最多有多少数据在路上”。它不是无限发，而是受两种窗口共同约束：

$$
\text{发送上限} = \min(\text{rwnd}, \text{cwnd})
$$

其中 `rwnd` 是接收窗口，表示接收方还能装多少；`cwnd` 是拥塞窗口，表示网络当前允许你放多少。前者防止对端被压垮，后者防止链路被挤爆。

玩具例子：

- 接收窗口 `rwnd = 16KB`
- MSS = 1500B，MSS 可以理解为“单个 TCP 段可携带的最大有效负载”
- 假设当前 `cwnd` 足够大

那么一次最多发送的段数约为：

$$
\left\lfloor \frac{16 \times 1024}{1500} \right\rfloor = 10
$$

也就是先发 10 段，累计约 15000B，等待 ACK 后窗口向前滑动，再继续发。

| 时间点 | 已发 bytes | 未确认 bytes | 可继续发送? | 说明 |
|---|---:|---:|---|---|
| T0 | 0 | 0 | 是 | 初始可发 |
| T1 | 15000 | 15000 | 否 | 接近 16KB 上限 |
| T2 | 收到 6000B ACK | 9000 | 是 | 窗口向前滑动 |
| T3 | 再发 6000B | 15000 | 否 | 回到上限附近 |

### 3. 拥塞控制与重传定时器

TCP 不只靠窗口，还靠拥塞控制判断“现在网络是不是堵了”。最粗略的理解是：

- 正常收到 ACK，`cwnd` 逐步增加
- 出现丢包，`cwnd` 通常减半
- 超时未确认，则认为网络更严重拥塞，进入重传并退避

常见的教学表达可以写成：

$$
cwnd_{new} = cwnd_{old} + 1 \quad (\text{每个 RTT 成功推进时})
$$

$$
cwnd_{new} = \frac{cwnd_{old}}{2} \quad (\text{检测到丢包时})
$$

这里 RTT 是“一个包发出去再收到确认的往返时间”。

重传定时器 RTO 的核心公式来自 RFC 6298：

$$
RTO = SRTT + \max(G, 4 \cdot RTTVAR)
$$

- `SRTT`：平滑后的 RTT 平均值
- `RTTVAR`：RTT 波动程度
- `G`：时钟粒度

它表达的不是“固定等 1 秒”，而是“按历史网络延迟和波动动态估计超时阈值”。若连续超时，RTO 会指数退避，避免在拥堵时继续猛发。

### 4. 队列为什么决定延迟

Linux 网络栈里有多个队列：backlog 队列、socket buffer、qdisc、驱动发送队列、NIC ring。只要任何一层积压，包就会排队。排队本身不是错，但队列太长会导致 bufferbloat，也就是“缓冲区太大，吞吐没掉，延迟却暴涨”。

因此，高性能网络不是让队列无限大，而是让队列“够用但不过量”。

---

## 代码实现

先看简化后的发送路径伪代码。下面不是内核源码逐行翻译，而是核心逻辑压缩版：

```python
from math import floor

def max_segments(rwnd_bytes: int, mss: int, cwnd_segments: int) -> int:
    by_rwnd = rwnd_bytes // mss
    return min(by_rwnd, cwnd_segments)

def update_cwnd_on_ack(cwnd: int) -> int:
    return cwnd + 1

def update_cwnd_on_loss(cwnd: int) -> int:
    return max(1, cwnd // 2)

# 玩具例子：16KB 接收窗口，MSS 1500B，cwnd=20 段
rwnd = 16 * 1024
mss = 1500
cwnd = 20

segs = max_segments(rwnd, mss, cwnd)
assert segs == 10

cwnd = update_cwnd_on_ack(cwnd)
assert cwnd == 21

cwnd = update_cwnd_on_loss(cwnd)
assert cwnd == 10
```

再看发送主线的伪代码：

```text
send(fd, user_buf)
  -> sock_sendmsg()
  -> tcp_sendmsg()
      1. 从用户态复制数据到内核发送缓冲
      2. 分配或复用 sk_buff
      3. 按 MSS 切分负载
      4. 挂入 TCP 发送队列
  -> tcp_transmit_skb()
      1. 填 TCP 头
      2. 交给 IP 层
  -> ip_queue_xmit()
      1. 查路由
      2. 填 IP 头
      3. 进入 qdisc / dev queue
  -> dev_queue_xmit()
  -> driver.ndo_start_xmit(skb)
      1. 把 skb 映射成 DMA 描述符
      2. 放入 NIC 发送 ring
      3. 通知网卡发送
```

接收侧的逻辑对应为：

```text
NIC 收到包
  -> 触发硬中断
  -> 中断处理函数做最少工作，调度 NET_RX_SOFTIRQ
  -> poll/NAPI 批量取包
  -> 构造/回收 sk_buff
  -> 交给 ip_rcv()
  -> tcp_v4_rcv()
  -> 放入 socket 接收队列
  -> 用户进程 recv() 取走
```

这里的 NAPI 可以理解为“内核在高流量下改为批量轮询收包的机制”，它的目标是减少中断风暴。

真实工程例子：一个反向代理服务器用 `epoll` 管理数万个连接。应用层看见的是“某些 fd ready 了”，但底层实际发生的是：

1. 网卡把收到的包 DMA 到内存。
2. `NET_RX_SOFTIRQ` 批量处理包，转成 `sk_buff`。
3. TCP 按序重组，把数据挂到某个 socket 的接收队列。
4. `epoll` 发现该 socket 变成可读，把事件交给用户态线程。
5. 线程 `recv()` 取数据，解析 HTTP，再 `send()` 响应。
6. 响应包再次经历 `sk_buff -> qdisc -> 驱动 ring -> NIC`。

这就是为什么 `epoll` 虽然重要，但它只处理“事件分发”，不处理“包是如何在内核里流动”的核心成本。

---

## 工程权衡与常见坑

高并发场景下，网络瓶颈常常不是协议规范本身，而是实现细节的堆积。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| `netdev_max_backlog` 太小 | 突发流量时丢包 | 结合压测逐步调大并观察丢包 |
| `tx_queue_len` 太大 | 队列堆积，尾延迟升高 | 不追求盲目大队列，结合 BQL 调整 |
| SoftIRQ 饱和 | `ksoftirqd` 占满 CPU，应用线程被挤压 | 做 CPU 亲和、RSS/RPS 调整 |
| 拷贝次数过多 | CPU 消耗大，吞吐下降 | `sendfile`、GSO、zero-copy |
| 忽略网卡能力 | 分段、校验全交给 CPU | 开启 TSO/GSO/GRO 等卸载能力 |
| 只盯 `epoll` | 误判瓶颈位置 | 同时看软中断、重传、队列长度 |

常见调优参数可以先从这些开始：

| 参数 | 意义 | 调整方向 | 监控指标 |
|---|---|---|---|
| `net.core.netdev_max_backlog` | 设备收包 backlog 上限 | 突发场景可适当增大 | 丢包、软中断积压 |
| `tx_queue_len` | 设备发送队列长度 | 过大时下调 | 尾延迟、qdisc backlog |
| `rmem_max` / `wmem_max` | socket 最大收发缓冲 | 大连接或高 BDP 链路增大 | 吞吐、内存占用 |
| `somaxconn` | 监听队列上限 | 高并发接入时提高 | accept backlog |
| `busy_poll` 相关参数 | 忙轮询时长 | 低延迟场景按需开启 | CPU 占用、RTT |

几个典型坑需要单独指出。

第一，bufferbloat。很多人看到丢包就只会加大队列，但队列越长，排队时延越高，最后表现为“吞吐不低，但页面很卡”。

第二，SoftIRQ 亲和性。网卡中断和应用线程如果频繁跨核迁移，缓存命中率会变差，额外引入调度成本。高流量机器通常要看 RSS、RPS、XPS 和 CPU 绑定。

第三，盲目追求 zero-copy。零拷贝不是任何场景都更快。若数据要修改、压缩、加密，很多路径仍然需要进入用户态处理，收益会下降。

第四，把 `epoll` 说成 O(1) 就以为全链路 O(1)。`epoll` 降的是“事件感知成本”，不是“收发包总成本”。真正的 CPU 时间仍花在协议处理、软中断、内存管理和拷贝上。

---

## 替代方案与适用边界

默认的 Linux 内核栈适合绝大多数通用服务，因为它完整提供 TCP 可靠性、拥塞控制、流量控制、成熟驱动适配和标准 socket 接口。只有在极低延迟或极高包率场景下，才值得考虑替代方案。

| 方案 | 降低拷贝 | 遗失特性 | 适用情况 |
|---|---|---|---|
| 常规内核 TCP/IP | 一般 | 几乎无 | 通用 Web、RPC、数据库 |
| `sendfile` / zero-copy | 部分降低 | 灵活性下降 | 静态文件、代理转发 |
| `SO_BUSY_POLL` | 不直接降拷贝，但降唤醒延迟 | 更耗 CPU | 极低延迟服务 |
| XDP | 可在更早阶段处理包 | 需自己处理更多逻辑 | 丢弃、过滤、简单转发 |
| DPDK | 大幅绕开内核路径 | 失去标准 socket/TCP 便利 | 高频交易、超高 PPS |
| 用户态 TCP 栈 | 可定制 | 复杂度高，兼容性差 | 特殊专用场景 |

真实工程例子：一个静态资源分发服务已经用了 `epoll + sendfile + GSO`，NIC 也支持分段卸载，但在高 PPS 的短连接压测下，`ksoftirqd` 仍然偏高。这时可以考虑把最前面的包过滤逻辑下沉到 XDP，先在驱动更靠前的位置丢掉无效流量，再把真正业务流量交给内核 TCP 栈。这种做法的前提是：你只把“简单且高频”的逻辑下沉，而不是重写整套可靠传输。

选择边界可以简化成一句话：如果你需要完整 TCP 语义和成熟生态，先把 Linux 原生栈调到位；只有当 profiler 明确证明瓶颈就在内核通路本身，才去考虑 XDP、DPDK 或用户态协议栈。

---

## 参考资料

| 来源名 | 类型 | 贡献 | 链接 |
|---|---|---|---|
| Queueing in the Linux Network Stack | 专题文章 | 解释 Linux 网络栈中的排队、驱动队列、SoftIRQ 与拥塞现象 | https://www.linuxjournal.com/content/queueing-linux-network-stack |
| RHEL Chapter 34 Tuning the network performance | 官方文档 | 给出网络性能调优的系统化参数与监控思路 | https://docs.redhat.com/ |
| RFC 6298: Computing TCP's Retransmission Timer | RFC | 给出 TCP 重传超时 RTO 的标准计算方法 | https://datatracker.ietf.org/doc/html/rfc6298 |
| Sliding Window Protocol Overview | 技术资料 | 说明滑动窗口的基本约束与吞吐关系 | https://www.sciencedirect.com/topics/computer-science/sliding-window-protocol |
| TCP Notes | 学习资料 | 用较直观的方式解释 TCP、ACK、窗口滑动 | https://notes.eddyerburgh.me/computer-networking/internet/tcp |
| epoll | 资料页 | 说明 epoll 的事件模型与数据结构特点 | https://en.wikipedia.org/wiki/Epoll |

这些资料的分工很清楚：Linux Journal 适合理解“为什么会排队、为什么 SoftIRQ 会成为瓶颈”；RFC 6298 提供 RTO 的正式公式；Red Hat 文档更适合落地调优；滑动窗口资料帮助建立吞吐和未确认数据上限之间的直觉。

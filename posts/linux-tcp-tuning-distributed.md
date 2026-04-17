## 核心结论

分布式训练里的“网络慢”，很多时候不是网卡标称带宽不够，而是 Linux 默认 TCP 参数让单连接窗口太小，导致链路根本灌不满。TCP 窗口可以理解成“发送方在等确认前，最多能在路上挂着多少数据”。如果窗口小于带宽延迟积 BDP（Bandwidth-Delay Product，白话说就是“这条链路在一个 RTT 内最多能装下多少数据”），吞吐就会被窗口直接卡死。

判断一条链路能否接近物理上限，最先看两个式子：

$$
BDP = 带宽 \times RTT
$$

$$
吞吐量 \approx \frac{TCP\ 窗口}{RTT}
$$

因此，10 Gbps、40 Gbps 这类训练网络要想接近线速，通常至少要同时做四件事：

| 项目 | 为什么要做 | 不做的直接后果 |
| --- | --- | --- |
| 调大 `net.core.rmem_max/wmem_max` 与 `tcp_rmem/tcp_wmem` | 给 TCP 自动扩窗留足上限 | 窗口停在几 MB，吞吐只有几百 Mbps 到几 Gbps |
| 统一配置 MTU 9000 | 减少每 GB 数据对应的包数和中断数 | CPU 开销偏高，吞吐波动明显 |
| 配置 IRQ affinity 与网卡多队列 | 让多个 CPU 分担收发队列 | 单核打满，其余核心空闲 |
| 显式设置 `NCCL_SOCKET_IFNAME` | 保证 NCCL 走对网卡 | 训练进程误走 1 Gb 或虚拟网卡 |

一个最小玩具例子就能说明问题。假设链路是 10 Gbps，RTT 是 100 ms，如果发送窗口只有 4 MiB，那么理论吞吐上限大约是：

$$
4\ MiB / 0.1s = 40\ MiB/s \approx 320\ Mb/s
$$

也就是说，即使你插着 10 Gb 网卡，也可能只跑出 320 Mb/s 这种远低于线速的结果。把窗口上限扩到 32 MiB 后，理论上限才来到约 2.56 Gb/s；如果目标是逼近 10 Gbps，窗口还要继续向 BDP 对齐。

---

## 问题定义与边界

本文讨论的是普通以太网环境里的 Linux TCP 调优，主要面对分布式训练中 NCCL 走 socket 通信、或训练前用 `iperf3` 做链路验收的场景。这里的“调优”不是魔法提速，而是消除明显的软件瓶颈，让网络表现尽量接近硬件和拓扑允许的上限。

边界先说清楚：

| 边界 | 属于本文 | 不属于本文 |
| --- | --- | --- |
| 10/25/40 Gb 以太网 TCP 通信 | 是 |  |
| `iperf3` 端到端验证 | 是 |  |
| NCCL 通过 socket 选网卡 | 是 |  |
| RoCE、InfiniBand、RDMA verbs 细节 |  | 不是本文主线 |
| 交换机缓冲、ECN、PFC 深度调参 |  | 不是本文主线 |
| 云厂商专用加速协议 |  | 不展开 |

这里还有一个容易混淆的点：`net.core.rmem_max/wmem_max` 不是“当前所有连接都立刻占满这么多内存”，它是单个 socket 缓冲区允许达到的上限；`tcp_rmem/tcp_wmem` 是 TCP 自动调节窗口时可使用的最小值、默认值和最大值。只有把这两组参数一起放宽，Linux 才可能在高 RTT 或高带宽环境下把窗口真正拉起来。

真实工程里，问题往往不是单一参数失效，而是多个瓶颈叠加。例如一台训练节点有两块网卡：`eth0` 是 10 Gb 训练网，`eno1` 是 1 Gb 管理网。如果没有显式设置 `NCCL_SOCKET_IFNAME=eth0`，即使你把 `wmem_max`、MTU、IRQ 都调好了，训练仍可能走错接口，最终结果还是慢。

---

## 核心机制与推导

先从最核心的量讲起：RTT。RTT 是 round-trip time，白话说就是“一个数据包发出去，再收到确认，大概要多久”。只要 RTT 不为零，发送方就必须在这段等待时间里持续把数据堆在链路上，否则链路会出现“空转”。

这就是 BDP 的意义。假设带宽为 $B$，RTT 为 $T$，那么为了让链路始终满载，至少需要：

$$
窗口 \ge B \times T
$$

### 玩具例子

假设：

- 带宽 = 10 Gbps
- RTT = 10 ms

则：

$$
BDP = 10 \times 10^9 \times 0.01 = 10^8\ bits \approx 12.5\ MB
$$

这意味着，TCP 有效窗口至少要到 12.5 MB 左右，单连接才有可能接近 10 Gbps。若实际最大窗口只有 4 MiB，则上限近似是：

$$
4\ MiB / 0.01s \approx 400\ MiB/s \approx 3.2\ Gb/s
$$

这已经不是“调得不够精细”，而是数学上就跑不满。

很多新手看到这里会问：Linux 不是会自动调节窗口吗？会，但自动调节也有天花板。`tcp_rmem` 和 `tcp_wmem` 的第三个值就是最大上限，而这个上限还会被 `net.core.rmem_max` 和 `net.core.wmem_max` 再卡一道。上层想开更大的 socket buffer，如果系统全局上限太小，也开不上去。

MTU 机制是第二个关键点。MTU 是 maximum transmission unit，白话说就是“一个以太网帧一次最多装多少字节”。标准 MTU 1500 时，大文件会被拆成更多包；如果全链路支持 9000 的 jumbo frame，大数据传输会减少包数量、协议头占比和中断压力。它不改变 BDP 公式，但会降低 CPU 侧开销，所以对 10/40 Gb 链路常常有实用价值。

第三个关键点是 IRQ affinity。IRQ 是硬件中断，白话说就是“网卡通知 CPU 来处理收发事件的机制”。现代网卡通常有多个 RX/TX 队列，如果所有中断都堆到同一个 CPU 上，那么你看到的不是“网卡不行”，而是“单核被打满”。IRQ affinity 的作用，就是把不同队列的中断绑到不同核心，让吞吐更稳定。

### 真实工程例子

一组 8 卡训练节点，每台机器有：

- 1 张 10 Gb 训练网卡 `eth0`
- 1 张 1 Gb 管理网卡 `eno1`
- 默认 Linux TCP 缓冲参数
- NCCL 未显式指定接口

现象通常是：

- `nvidia-smi` 里 GPU 利用率有周期性掉坑
- `sar -n DEV 1` 看到 `eno1` 也有流量
- `iperf3` 单流只有几百 Mbps 到 2 Gbps
- 某一个 CPU 核长期 100%，其余核心较闲

这类问题不是只改一个 `sysctl` 就结束。正确路径通常是：

1. 先用 `iperf3` 单流和多流确认裸链路极限。
2. 再调大 `rmem/wmem` 和 `tcp_rmem/tcp_wmem`。
3. 再统一 MTU 到 9000，并确认交换机与对端都支持。
4. 再检查 `ethtool -l`、`/proc/interrupts` 和 IRQ 绑核。
5. 最后显式导出 `NCCL_SOCKET_IFNAME=eth0`，避免选错口。

只有这几个层次都打通，训练吞吐才能稳定接近物理极限。

---

## 代码实现

下面给出两个层面的实现：一个是计算和校验 BDP 的 `python` 小脚本，一个是 Linux 上的实际调参命令。

```python
def bdp_bytes(bandwidth_gbps: float, rtt_ms: float) -> float:
    bits_per_sec = bandwidth_gbps * 1_000_000_000
    rtt_sec = rtt_ms / 1000
    return bits_per_sec * rtt_sec / 8

def throughput_gbps(window_mib: float, rtt_ms: float) -> float:
    bytes_per_sec = window_mib * 1024 * 1024 / (rtt_ms / 1000)
    return bytes_per_sec * 8 / 1_000_000_000

# 玩具例子：10 Gbps, 10 ms
bdp = bdp_bytes(10, 10)
assert 12_000_000 < bdp < 13_500_000

# 4 MiB 窗口在 100 ms RTT 上只能到约 0.32 Gbps
tp = throughput_gbps(4, 100)
assert 0.30 < tp < 0.35

# 32 MiB 窗口在 100 ms RTT 上约 2.68 Gbps
tp2 = throughput_gbps(32, 100)
assert 2.6 < tp2 < 2.8

print("BDP(bytes)=", int(bdp))
print("4MiB@100ms throughput(Gbps)=", round(tp, 3))
print("32MiB@100ms throughput(Gbps)=", round(tp2, 3))
```

如果你不知道该设多大窗口，可以先按下面这个原则粗估：

| 链路 | RTT | 目标窗口量级 |
| --- | --- | --- |
| 10 Gb | 0.1 ms 到 0.5 ms，机房内短链路 | 几百 KB 到数 MB |
| 10 Gb | 1 ms 到 10 ms，跨机架或跨园区 | 数 MB 到十几 MB |
| 40 Gb | 1 ms 到 10 ms | 十几 MB 到几十 MB |

Linux 实际调参可先从保守值开始：

```bash
sysctl -w net.core.rmem_max=33554432
sysctl -w net.core.wmem_max=33554432
sysctl -w net.ipv4.tcp_rmem="4096 87380 33554432"
sysctl -w net.ipv4.tcp_wmem="4096 65536 33554432"
sysctl -w net.ipv4.tcp_mtu_probing=1

ip link set eth0 mtu 9000
ip link set eth0 txqueuelen 10000

export NCCL_SOCKET_IFNAME=eth0

iperf3 -s
# 对端执行
iperf3 -c 10.0.0.2 -w 32M -P 4 -t 30
```

这组命令的逻辑是：

- `net.core.*_max` 先把系统允许的单 socket 上限放宽。
- `tcp_*mem` 再把 TCP 自动扩窗的最大值放宽。
- `tcp_mtu_probing=1` 允许内核在 MTU 不确定时做探测，减少 PMTU 问题。
- `mtu 9000` 提高单帧载荷，但前提是全链路一致。
- `txqueuelen` 提高设备发送队列长度，减少高吞吐下的排队抖动。
- `iperf3 -w 32M -P 4` 用大窗口和多流测试是否逼近物理带宽。

如果你只做训练不做基线测试，后续一旦发现吞吐异常，就无法判断是 NCCL、模型并行、交换机、还是 Linux TCP 参数导致的。因此 `iperf3` 不是附属步骤，而是验收步骤。

---

## 工程权衡与常见坑

第一个常见坑是 jumbo frame 半配置。也就是主机网卡已经设到 9000，但交换机端口、对端机器、虚拟交换层或某一跳设备还停在 1500。这样不但不会稳定提速，还可能出现分片、丢包、吞吐抖动。正确做法是全链路统一验证，并用：

```bash
ping -M do -s 8972 <peer-ip>
```

确认大包确实能通过。

第二个常见坑是把 `tcp_wmem` 调大了，却忘了 `net.core.wmem_max`。前者像“TCP 想申请的额度”，后者像“系统批准的上限”。只改前者不改后者，很多场景下效果并不会落地。

第三个常见坑是忽略多队列和中断分布。你可能已经把 `iperf3 -P 4` 跑到了 8 Gbps，但 CPU 其中一个核 100%，另外几个核 20%。这说明瓶颈已经转到中断和 softirq 处理，而不是窗口。此时要看 `ethtool -l eth0`、`ethtool -L eth0 combined N`、`/proc/interrupts` 和 IRQ 亲和性配置。

第四个常见坑是 NCCL 选错接口。`NCCL_SOCKET_IFNAME` 是“告诉 NCCL 优先或只用哪些网卡”的环境变量。白话解释，就是“不要让通信库自己猜网卡”。多网卡机器上，如果不显式指定，它可能走到管理网、容器桥接网甚至回环接口过滤逻辑，训练吞吐会显著下降。

| 场景 | 典型现象 | 根因 | 处理方式 |
| --- | --- | --- | --- |
| 只改 `tcp_wmem` 未改 `net.core.wmem_max` | 吞吐改善很小 | 系统上限仍然太低 | 两组参数一起改 |
| MTU 9000 只改了单机 | 大包不通、吞吐抖动 | 路径 MTU 不一致 | 全链路统一 MTU，再做 `ping -M do` |
| 单流跑不满，多流才上来 | `iperf3 -P 1` 慢，`-P 4` 快 | 单连接受窗口或拥塞控制影响 | 先确认 BDP，再决定是否多流 |
| 某个 CPU 核打满 | 吞吐不稳，softirq 高 | IRQ/队列未分散 | 配多队列和 IRQ affinity |
| 训练带宽远低于 `iperf3` | 裸链路快，训练慢 | NCCL 走错接口或框架配置不一致 | 显式设 `NCCL_SOCKET_IFNAME` |

---

## 替代方案与适用边界

本文主线是“普通 TCP 环境先把基础瓶颈消掉”。这条路线最通用，因为不依赖专门硬件，也不要求你改应用协议。但它不是唯一方案。

一种替代思路是换拥塞控制算法，比如 BBR 或 HTCP。拥塞控制可以理解成“TCP 决定发多快、收多稳的策略”。在一些高带宽、高 RTT、轻微丢包的环境里，BBR 可能比默认算法更容易维持高吞吐。不过这不替代窗口调优，只是把“如何接近可用带宽”的策略换了一种。若系统窗口上限本身太小，BBR 也救不了。

另一种常见手段是多流。`iperf3 -P 4` 的本质不是“把网卡变快”，而是让多个 TCP 连接共同占满链路。当单流受限于窗口、拥塞控制或应用模型时，多流能更快打满物理带宽。但它带来的问题也很直接：更难诊断、对公平性不友好、与真实训练流模型不完全一致。

再往上是 RDMA、RoCE、InfiniBand、GPUDirect 这类方案。它们的核心价值是绕开普通 TCP/IP 栈的一部分开销，用更低延迟、更低 CPU 占用的路径传输数据。如果你的训练集群已经完整部署这套硬件和驱动，本文这套 TCP 调优就不是主战场，而更像 fallback 路径或故障回退路径。

| 方案 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| TCP + BDP 调优 | 常规以太网训练集群 | 最通用，落地门槛低 | 仍受内核栈与 CPU 开销影响 |
| TCP + 多流 + 更强拥塞控制 | 单流跑不满但又必须用 TCP | 提升上限较快 | 诊断复杂，行为不总等价于真实业务 |
| RDMA / GPUDirect | 大规模高性能训练网络 | 延迟和 CPU 开销更优 | 依赖硬件、交换机和驱动生态 |

实务上，建议顺序是：

1. 先用 TCP 基线把链路跑到该有的水平。
2. 再确认训练框架是否走对接口。
3. 只有在 TCP 已接近硬件极限后，才讨论是否需要 RDMA 或更复杂的传输栈。

---

## 参考资料

- NASA 关于 TCP 窗口与吞吐关系的教程：说明 `Throughput ≈ Window / RTT` 的来源。  
- Red Hat 网络性能调优文档：涵盖 `tcp_rmem/tcp_wmem`、MTU 与验证方法。  
- SUSE 性能调优白皮书：介绍 IRQ affinity、队列与 CPU 分担关系。  
- NVIDIA NCCL 环境变量文档：解释 `NCCL_SOCKET_IFNAME` 等接口选择规则。  
- iperf3 官方文档：说明窗口、并行流和端到端带宽验证方式。  
- S3NS/Google Cloud 相关经验文档：总结高带宽链路上的 TCP buffer 调优原则。  
- OnnoWiki 与 10/40 Gb 调优经验材料：提供常见 `sysctl` 推荐值与实践经验。

## 核心结论

Docker 常见的 `bridge`、`host`、`overlay` 三种网络模式，本质上对应三种完全不同的转发路径。

第一，`bridge` 是单机默认方案。它依赖 Linux bridge，也就是内核里的二层虚拟交换机；Docker 默认创建 `docker0`，再给每个容器接一对 `veth pair`，也就是“成对出现的虚拟网线”。容器出站流量经过宿主机上的 `iptables` 做源地址转换，外部访问容器端口时再做目标地址转换，因此它解决的是“单主机上容器之间互通”和“把容器服务暴露给外部”这两个问题。

第二，`host` 是最短路径方案。容器直接共享宿主机网络命名空间，也就是“直接用宿主机那套网卡、路由表和端口空间”，因此没有 bridge 转发、没有额外 NAT、没有 VXLAN 封装。代价是隔离弱，端口冲突风险高，但性能和时延通常最好。在分布式训练、低延迟数据库、需要大量端口的服务里，`host` 往往比默认 bridge 更合适。

第三，`overlay` 是跨主机方案。它在多台机器之间用 VXLAN，也就是“把二层以太网帧再套一层 UDP 外壳发出去”，构造一个逻辑上的同一网段。它解决的是“容器不在同一台宿主机上，但仍要像在一个二层网络里那样通信”。代价是封装开销、控制平面复杂度、MTU 调整和运维门槛都明显更高。

可以先记一个工程上足够实用的选型表：

| 模式 | 是否 NAT | 是否跨主机 | 典型适用场景 | 经验性能特征 |
| --- | --- | --- | --- | --- |
| bridge | 是 | 否 | 单机微服务、开发环境、常规 Web 服务 | 接近宿主，但有 NAT 和转发路径 |
| host | 否 | 否 | 高性能服务、分布式训练、大端口范围应用 | 路径最短，通常最好 |
| overlay | 常伴随 NAT/隧道转发 | 是 | Swarm 多主机通信、跨节点服务发现 | 受 VXLAN 封装和 MTU 影响明显 |

如果只需要单机跑服务，先选 `bridge`。如果性能和低延迟比隔离更重要，选 `host`。如果必须多机组网，才进入 `overlay` 的问题域。

---

## 问题定义与边界

这篇文章要解决的不是“Docker 网络命令怎么背”，而是三个更底层的问题：

1. 容器为什么默认能拿到 `172.17.x.x` 这类地址，并且还能访问外网？
2. `docker run -p 8080:80` 到底改了什么规则，为什么访问宿主机 `8080` 会进到容器 `80`？
3. 为什么一旦跨主机，就要从简单的 bridge 进入 VXLAN、MTU、数据平面端口这些更复杂的约束？

先明确边界。

`bridge` 处理的是单主机网络问题。容器 IP 通常是私网地址，不能直接被外部路由到，所以出站需要 SNAT，也就是“把源地址改成宿主机地址”；入站端口映射需要 DNAT，也就是“把目标地址改成容器地址”。

`host` 基本不处理这些问题。因为它没有独立容器网络栈，容器就是“站在宿主机网卡旁边收发包”。所以 `-p` 在 `host` 模式下没有意义，端口是否可用完全取决于宿主机当前占用情况。

`overlay` 处理的是跨主机容器互通问题。不同宿主机上的容器，本来处在不同的二层广播域里，Linux bridge 无法直接跨机工作，所以需要一个隧道层把远端容器网络“叠”到本地。Docker 的 overlay 选择的是 VXLAN，它的典型额外头部开销约为 50 bytes，因此一个非常关键的公式是：

$$
\text{Effective MTU} = \text{Underlying MTU} - 50
$$

意思是：底层物理网络如果是 MTU 1500，那么 overlay 网络里真正安全可用的负载大小大约只有 1450。若底层网络本来就只有 1470，再套 VXLAN 还按 1500 发，很容易碎片化，或者被中间设备直接丢弃。

一个玩具例子可以帮助理解这三个边界。

假设你在一台笔记本上运行：

```bash
docker run --rm -d --name web -p 8080:80 nginx
```

现在浏览器访问的是宿主机 `8080`，不是容器 IP。这个请求的边界很清楚：

- 浏览器发给宿主机
- 宿主机 `iptables` 在 `nat` 表里做 DNAT
- 目标改写成容器 IP:80
- 回包时再经连接跟踪返回给浏览器

但如果是另一台服务器上的容器要访问这个容器，仅靠单机 bridge 就不够了，因为中间少了“跨主机二层连通性”。这时问题已经从 NAT 进入隧道和跨节点转发。

一个真实工程例子是分布式训练。训练框架往往会同时拉起大量进程，使用大量端口，并对延迟和抖动敏感。如果还叠加 Docker bridge、NAT、端口映射甚至 overlay 封装，网络路径会被拉长，吞吐和尾延迟都会更差。所以这类场景常直接用 `--network host`，目的不是“图省事”，而是减少网络层可变因素。

---

## 核心机制与推导

### 1. bridge 的工作链路

先看 `bridge`。

Docker 启动后，默认会创建 `docker0`。它是一个 Linux bridge，可以把它理解成“宿主机里的虚拟交换机”。每新建一个 bridge 网络容器，Docker 会创建一对 `veth pair`：

- 一端放进容器的网络命名空间，通常命名为 `eth0`
- 另一端留在宿主机，并接入 `docker0`

这样，同一台宿主机上的多个容器就像都插在同一台交换机上一样，可以二层互通。

但容器用的是私网地址，比如 `172.17.0.2`。如果它访问公网，公网不会认识这个地址，所以要在宿主机出口做源地址伪装 `MASQUERADE`。这是一种特殊的 SNAT，意思是“自动改成出口网卡当前的地址”。

外部请求进入容器，则反过来做 DNAT。以 `-p 8080:80` 为例，访问宿主机 `8080` 时，规则会把目标改写到容器 `172.17.0.2:80`。

典型规则大致长这样：

```text
*nat
:PREROUTING ACCEPT [..]
:POSTROUTING ACCEPT [..]
:DOCKER - [0:0]
-A PREROUTING -m addrtype --dst-type LOCAL -j DOCKER
-A DOCKER ! -i docker0 -p tcp --dport 8080 -j DNAT --to-destination 172.17.0.2:80
-A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE
COMMIT
```

可以把它翻译成三步：

1. 发往本机地址的入站流量，先跳到 `DOCKER` 链。
2. 目标端口如果是 `8080`，改写成容器 `172.17.0.2:80`。
3. 容器流量如果要离开 `docker0` 出去，就把源地址改成宿主机地址。

这就是 bridge 模式里“端口映射”和“容器访问外网”的底层原因。

### 2. host 为什么更快

`host` 模式几乎没有上面这套路径。容器共享宿主机网络命名空间，也就是不再有独立容器网卡、独立容器 IP、独立 NAT 出口。应用在容器里监听 `0.0.0.0:8080`，本质上就是在宿主机监听 `8080`。

所以它快，不是因为 Docker 做了某种神秘优化，而是因为绕开了额外网络抽象层：

- 没有 `docker0`
- 没有 `veth pair`
- 没有容器侧独立地址
- 一般不需要 `-p` 带来的 DNAT 路径

路径短，规则少，可观测性也更直。代价同样直接：谁占了端口，谁就赢；容器之间也不再有网络级隔离边界。

### 3. overlay 为什么复杂

`overlay` 的难点在于，它要让不在同一台机器上的容器“看起来像在同一张二层网络里”。

Linux bridge 无法跨机，所以 Docker 在每个节点上除了本地 bridge，还会创建 VXLAN 设备。VXLAN 可以理解为“用 UDP 包装二层帧再发往远端宿主机”。默认数据平面端口常见为 `4789/UDP`。

发送路径大致是：

1. 容器把帧发到本地 bridge。
2. 本地 bridge 判断目标在远端节点。
3. 帧被交给 VXLAN 设备封装。
4. 外层加上 UDP/IP 头发往目标宿主机。
5. 目标宿主机解封装，再交给本地 bridge 和目标容器。

这比 bridge 至少多了两件事：

- 额外封装/解封装
- 远端节点寻址和控制平面同步

所以 overlay 一定比 bridge 复杂，也更容易受 MTU、交换机、云厂商 VPC 限制影响。

吞吐差异的方向也容易推出来。若底层网络吞吐是 $B$，额外头部开销占比近似为 $\frac{H}{P+H}$，其中 $H$ 是隧道头部，$P$ 是有效负载。负载越小，头部占比越高；路径中每多一层 bridge、iptables、VXLAN 处理，CPU 和缓存开销也会增加。因此小包、高并发、跨机转发时，overlay 退化最明显。

下面这个 Python 玩具程序不是在模拟 Linux 内核，而是在模拟“有效载荷比例”这个核心现象：

```python
def effective_payload_ratio(underlay_mtu: int, vxlan_overhead: int = 50) -> float:
    assert underlay_mtu > vxlan_overhead
    return (underlay_mtu - vxlan_overhead) / underlay_mtu

ratio_1500 = effective_payload_ratio(1500)
ratio_9000 = effective_payload_ratio(9000)

assert round(ratio_1500, 4) == 0.9667
assert round(ratio_9000, 4) == 0.9944
assert ratio_9000 > ratio_1500

print(ratio_1500, ratio_9000)
```

它说明一个简单事实：同样 50 bytes 头部开销，在 MTU 1500 下占比明显高于 jumbo frame。真实系统里还叠加转发、校验、封装和中断处理，所以实际损耗往往比这个模型更复杂，但方向一致。

### 4. NAT 端口映射的推导

为什么 `-p 8080:80` 必须经过 NAT？因为外部世界只知道宿主机 IP，不知道容器 IP。设宿主机是 `10.0.0.5`，容器是 `172.17.0.2`。

外部客户端访问：

$$
10.0.0.5:8080
$$

宿主机收到后，在 `PREROUTING` 执行：

$$
(10.0.0.5:8080) \Rightarrow (172.17.0.2:80)
$$

回包时，连接跟踪系统根据 NAT 状态表恢复会话，让客户端仍然感觉自己在和 `10.0.0.5:8080` 通信。

因此 `-p` 的本质不是“Docker 帮你开了个端口”，而是“Docker 写了一组 NAT 规则，并把连接状态维护交给内核”。

---

## 代码实现

先给最小可操作命令，再解释每种模式的验证点。

### 1. bridge 模式

```bash
docker network create -d bridge demo-br
docker run --rm -d --name demo-nginx --network demo-br -p 8080:80 nginx
docker inspect demo-nginx --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
```

你应该看到一个类似 `172.18.0.x` 的地址。此时有两个验证点：

- 容器确实在 Docker bridge 私网里
- 宿主机 `8080` 通过 DNAT 转到容器 `80`

如果宿主机允许查看规则，还可以执行：

```bash
iptables -t nat -L DOCKER -n --line-numbers
```

重点不是记住所有输出，而是确认“宿主机端口 -> 容器 IP:端口”的 DNAT 规则真实存在。

### 2. host 模式

```bash
docker run --rm -d --name host-nginx --network host nginx
ss -ltnp | grep ':80'
```

这个模式下，Nginx 会直接占用宿主机 `80`。此时再写 `-p 8080:80` 没意义，因为不存在“宿主机端口再映射到容器端口”的那层地址边界。

一个玩具例子是本机开发时跑性能压测工具。若你只是想比较“网络模式本身”的差异，`host` 往往能更接近裸机表现，因为少了 bridge 和 NAT 变量。

### 3. overlay 模式

```bash
docker swarm init
docker network create -d overlay --attachable demo-ov
docker service create --name web --network demo-ov -p 8080:80 nginx
```

这里要多注意三件事：

- 必须先进入 Swarm，因为 overlay 依赖控制平面
- 节点间要开放 `2377/TCP`、`7946/TCP+UDP`、`4789/UDP`
- MTU 要和底层网络匹配，否则可能出现跨节点能解析服务但实际传输异常

如果底层 MTU 小于 1500，创建 overlay 时应显式指定：

```bash
docker network create -d overlay \
  --opt com.docker.network.driver.mtu=1450 \
  demo-ov
```

### 4. 一个真实工程例子

假设你在两台 GPU 服务器上跑分布式训练。每台机器 8 张卡，训练框架需要多进程互联，端口范围大、通信频繁、对抖动敏感。

这时典型做法是：

- 训练容器使用 `--network host`
- 直接绑定宿主机高速网卡
- 如需 RDMA、NCCL、多端口 rendezvous，避免再叠加 bridge NAT 或 overlay 封装

原因不是 Docker 官方“推荐了某个命令”，而是网络数据路径越短，越容易稳定吃满 10G/25G 甚至更高带宽；同时也避免 `-p` 大规模端口映射的规则维护成本。

---

## 工程权衡与常见坑

实际工程里，出问题最多的往往不是“模式选错”，而是“知道概念，但没把规则、端口、MTU 一起看”。

| 问题 | 常见表现 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| `-p` 映射后仍访问失败 | 本机可起服务，外部打不通 | `DOCKER-USER` 或 FORWARD 策略拦截 | 先检查 `iptables` 链顺序和默认策略 |
| host 模式端口冲突 | 容器启动失败或服务互相覆盖 | 共享宿主端口空间 | 启动前显式核查端口占用 |
| overlay 跨节点不通 | 同机正常，跨机失败 | `4789/UDP`、`7946`、`2377` 未开放 | 先通端口，再查路由和服务发现 |
| overlay 吞吐差、偶发超时 | 大包更明显，小包抖动上升 | VXLAN 封装和 MTU 不匹配 | 统一 `mtu`，必要时抓包验证 |
| 误以为 bridge 很“透明” | 只看容器 IP，不看 NAT 状态 | 忽略了 DNAT/SNAT 和连接跟踪 | 排障时同时看 `docker inspect` 与 `iptables` |

这里有几个坑值得单独展开。

第一，`bridge` 模式的问题经常被误判成“容器没起来”。其实很多时候容器服务正常，问题出在宿主机防火墙链顺序。Docker 会插入自己的链，但企业环境里往往还存在额外防火墙策略，特别是 `DOCKER-USER` 链。如果默认策略是 DROP，`-p` 看起来映射了，流量实际上被拦在更前面。

第二，`host` 模式最容易让人高估。它解决的是网络路径问题，不解决资源隔离、端口治理、噪声邻居、宿主机安全暴露等问题。如果你在一台混部机器上同时跑数据库、监控、业务和训练任务，全部切 host 往往不是优化，而是把边界全部拆掉。

第三，`overlay` 的坑大多集中在“看起来配置没错，但就是慢或偶发失败”。这种情况优先查 MTU。因为 overlay 的问题不是简单的“连或不连”，而常常表现为：

- 小包通，大包不通
- 心跳正常，吞吐异常
- 同节点正常，跨节点抖动大

本质原因是二层帧被封装进 UDP 之后，链路预算变了，但底层网络设备不一定帮你兜底。

---

## 替代方案与适用边界

如果这三种模式都不合适，还有两个常见替代方向：`macvlan` 和 `ipvlan`。

`macvlan` 的思路是让容器在二层网络里看起来像独立物理机，也就是“给容器一个真实可见的 MAC 和网段身份”。这适合一些老系统、网络探针、旁路流量采集场景，因为它们希望容器像普通主机一样被交换机和上游设备识别。

示例命令如下：

```bash
docker network create -d macvlan \
  --subnet=172.16.86.0/24 \
  --gateway=172.16.86.1 \
  -o parent=eth0 \
  macnet

docker run --rm --network macnet alpine ip addr
```

但它的边界也很明确：

- 依赖底层交换网络配合
- 常见环境需要开启混杂模式或满足上游 MAC 学习能力
- 容器与宿主机通信并不天然顺畅，很多场景还要额外补路由

所以不要把 `macvlan` 理解成“高级版 bridge”。它解决的是“让容器直接进入物理网络身份体系”，不是“替代所有 Docker 网络问题”。

再给一个选型收敛表：

| 需求 | 更合适的模式 | 不合适的原因 |
| --- | --- | --- |
| 单机开发、普通服务发布 | bridge | host 隔离太弱，overlay 过重 |
| 单机高性能服务、训练任务 | host | bridge 多一层转发，overlay 更慢 |
| 跨主机容器互通 | overlay | bridge 不能跨机，host 不提供虚拟二层 |
| 容器需要像真实主机上网段 | macvlan/ipvlan | bridge 地址不在物理网络中 |

最后给一个实用判断规则。

如果你的问题是“容器为什么不能访问外部”，先想 NAT。  
如果你的问题是“为什么访问宿主端口进不到容器”，先想 DNAT 和防火墙链。  
如果你的问题是“为什么跨节点慢而且诡异”，先想 VXLAN 和 MTU。  
如果你的问题是“为什么这类高性能任务都用 host”，答案通常是：它不是最优雅，但它最接近真实网络路径。

---

## 参考资料

- Docker Bridge 网络驱动说明文档。([docs.docker.com](https://docs.docker.com/engine/network/drivers/bridge/?utm_source=openai))
- Docker Host 网络驱动说明文档。([docs.docker.com](https://docs.docker.com/engine/network/drivers/host/?utm_source=openai))
- Docker Overlay 网络驱动与端口说明。([docs.docker.com](https://docs.docker.com/engine/network/drivers/overlay/?utm_source=openai))
- Docker `iptables` 规则与 `DOCKER-USER` 链说明。([docs.docker.com](https://docs.docker.com/engine/network/firewall-iptables/?utm_source=openai))
- Swarm/Overlay MTU 调整建议。([swarmkit.org](https://swarmkit.org/t/overlay-network-mtu/14?utm_source=openai))
- Overlay 与 host 吞吐对比实验讨论。([groups.google.com](https://groups.google.com/g/docker-dev/c/NR0GkxWmc20?utm_source=openai))
- Docker Labs 对 overlay/VXLAN 架构的拆解。([dockerlabs.collabnix.com](https://dockerlabs.collabnix.com/networking/concepts/06-overlay-networks.html?utm_source=openai))
- Macvlan 驱动官方说明。([docs.docker.com](https://docs.docker.com/network/drivers/macvlan/?utm_source=openai))

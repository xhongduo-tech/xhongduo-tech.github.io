## 核心结论

Docker 网络模式的本质，不是“给容器分一个 IP”这么简单，而是把两件事组合起来：

1. **网络命名空间**：可以理解为“给进程单独切一套网络视图”，里面有自己的网卡、路由表、端口监听状态。
2. **虚拟交换结构**：可以理解为“宿主机内部的虚拟连线和转发设备”，决定容器流量怎么出去、怎么互通、要不要跨主机。

因此，选择 `bridge`、`host`、`overlay`，本质上是在选择三种不同的连接方式：

| 维度 | bridge | host | overlay |
|---|---|---|---|
| 网络隔离 | 有，容器在独立网络命名空间 | 弱，共享宿主网络栈 | 有，且支持跨主机逻辑隔离 |
| 是否需要端口映射 | 常见，需要 `-p` | 不需要，也不能依赖 `-p` 提供隔离 | 服务间通常不靠 `-p`，对外暴露再单独设计 |
| 跨主机通信 | 不直接支持 | 不直接支持 | 支持 |
| 性能路径 | 多一层 veth/bridge/NAT | 最短，通常更优 | 多 VXLAN 封装，开销更高 |
| 典型场景 | 单机部署、开发环境、默认方案 | 单机高性能、低延迟 | Swarm 多节点服务互通 |

给零基础读者一个直观图景：

- `bridge` 像宿主机里有一只虚拟交换机 `docker0`，每个容器插上一根虚拟网线，先在本机局域网里说话，再通过 NAT 跟外界通信。
- `host` 像容器不再插自己的虚拟网线，而是直接借用宿主机的网口和端口。
- `overlay` 像多台宿主机之间又拉出一张“逻辑局域网”，让不同机器上的容器看起来像在同一个二层网络里。

如果只记一条：**单机隔离优先选 `bridge`，单机性能优先看 `host`，跨主机互通才选 `overlay`。**

---

## 问题定义与边界

讨论 Docker 网络模式，先把问题问对。真正要回答的是：

1. 容器是否需要**独立网络隔离**？
2. 容器是否要**对外暴露端口**？
3. 容器是否要和**其他主机上的容器互通**？
4. 你是在**单机 Docker**，还是在 **Swarm 集群**里运行？

这四个问题决定了模式选择边界。

### 适用场景与不适用场景

**`bridge` 适用：**
- 单机部署
- 希望容器有独立 IP 和独立端口空间
- 需要用 `-p 8080:80` 这样的方式对外暴露服务
- 希望默认隔离较好，减少宿主端口冲突

**`bridge` 不适用：**
- 明确追求极低网络开销
- 需要跨主机二层互通
- 需要容器直接出现在物理网络同一网段

**`host` 适用：**
- 单机部署
- 服务性能敏感，想减少 NAT 和桥接转发
- 容器数量不多，端口规划清晰
- 需要直接使用宿主网络能力

**`host` 不适用：**
- 多个服务容易争抢同一端口
- 需要强隔离
- 希望一个宿主上重复启动多份同端口服务

**`overlay` 适用：**
- 多主机部署
- Swarm 服务之间要按名字互访
- 希望跨节点容器“像同一个内网”一样通信

**`overlay` 不适用：**
- 只有单机
- 团队对 VXLAN、MTU、隧道排障没有准备
- 对极致网络性能要求高但不接受额外封装成本

新手可以先用一个最小决策法：

- 只跑一台机器，而且要隔离：`bridge`
- 只跑一台机器，而且更看重性能：`host`
- 跑多台机器，而且容器要互通：`overlay`

边界还要补两条：

- `overlay` 通常依赖 **Swarm 控制平面** 来管理网络成员和服务发现，不是单机随便一开就等价完成。
- `bridge` 的对外访问通常依赖 **iptables NAT**，所以“能上网”和“能被外面访问”不是同一件事，前者是出站转发，后者通常还要端口映射。

---

## 核心机制与推导

### 1. bridge 模式为什么能隔离又能上网

`bridge` 是 Docker 默认网络驱动。每个容器放在自己的网络命名空间里，再通过一对 **veth** 接口连接到宿主机的 `docker0` 网桥。

**veth** 可以理解为“一根虚拟网线的两端”，一端在容器里，一端在宿主里。

流量路径可以写成：

$$
\text{bridge: Container} \rightarrow veth_c \leftrightarrow veth_h \rightarrow docker0 \rightarrow eth0
$$

如果容器要访问外网，还会经过 NAT：

$$
\text{源地址 }172.17.0.2 \xrightarrow{\text{MASQUERADE}} \text{宿主机出口地址}
$$

这里的 `MASQUERADE` 是 iptables 的一种源地址转换规则，白话讲就是“出门前把容器地址换成宿主地址”。

一个玩具例子：

- 宿主机 `docker0` 网桥地址：`172.17.0.1/16`
- 容器 A 地址：`172.17.0.2`
- 容器 B 地址：`172.17.0.3`

A 和 B 在同一台宿主机上时，可以像同一局域网机器那样互通；但外部机器不能直接访问 `172.17.0.2`，因为这只是宿主内部虚拟网络地址。若执行 `-p 8080:80`，外部访问的是宿主 `8080`，再转发到容器 `80`。

这就是为什么 `bridge` 同时满足两件事：

- 容器之间彼此隔离，有自己的 IP 和端口空间
- 通过端口映射，外部仍然能访问容器服务

### 2. host 模式为什么更快但更“裸露”

`host` 模式下，容器不再创建独立的网络路径去接桥和 NAT，而是直接复用宿主机网络栈。

网络栈可以理解为“IP、端口、路由、连接跟踪这一整套网络处理逻辑”。

路径可以近似写成：

$$
\text{host: Container process} \rightarrow \text{Host network stack} \rightarrow eth0
$$

这意味着：

- 没有独立容器 IP
- 没有 `docker0` 这层桥接转发
- 通常没有端口映射这一层
- 容器监听的端口，实际就是宿主在监听

例如容器内应用绑定 `0.0.0.0:80`，宿主机的 `80` 端口就会被占用。再启动第二个同样监听 `80` 的 `host` 容器，直接冲突。

所以 `host` 的性能优势来自路径更短，少了一些转发和 NAT；它的代价则是隔离边界变弱，端口管理难度变高。

### 3. overlay 为什么能跨主机互通

`overlay` 要解决的问题是：容器不只在一台机器上，怎么还像在同一个逻辑局域网里通信？

答案是 **VXLAN**。

**VXLAN** 可以理解为“把二层以太网帧封装进 UDP 包里，在三层物理网络上传输”。

路径可以写成：

$$
\text{overlay: Container} \rightarrow veth \rightarrow \text{overlay switch} \rightarrow \text{VTEP封装} \rightarrow \text{物理网} \rightarrow \text{目标VTEP解封装}
$$

其中 **VTEP** 是 VXLAN Tunnel Endpoint，可理解为“隧道出入口”。

假设：

- 宿主机 A 上有服务 `web`
- 宿主机 B 上有服务 `db`
- 二者加入同一个 overlay 网络

那么 `web` 发出的报文，会先在本机进入 overlay 网络，再被封装成 VXLAN 包，经物理网发到 B，B 的 VTEP 解封装后再交给 `db` 容器。

这让不同主机的容器看起来像在同一个二层广播域里，但要付出两个代价：

1. 多了一层封装与解封装
2. MTU 变得更敏感

### 4. 为什么 overlay 常见 MTU 问题

标准以太网 MTU 常是 `1500` 字节。VXLAN 额外增加大约 `50` 字节封装头。于是：

$$
1500 + 50 = 1550
$$

如果底层物理网络还是 `1500`，那原本刚好不分片的包，经过封装后就可能超限，导致：

- 分片
- 丢包
- 延迟升高
- 某些服务“偶发超时”

因此 overlay 实战里常把有效 MTU 调到 `1450` 左右，给 VXLAN 预留头部空间。

给新手一个接力图景：

- `bridge`：容器先把包交给 veth，再交给 `docker0`，再交给宿主网卡
- `host`：容器直接站在宿主跑道上发包
- `overlay`：容器把包交给本机隧道入口，封装后穿过物理网，到另一台机器再拆包

---

## 代码实现

先给一个最小可运行的“模式差异模拟”，它不真的操作 Docker，而是把三种网络模式抽象成路径和额外开销，方便理解。

```python
from dataclasses import dataclass

@dataclass
class NetworkMode:
    name: str
    hops: int
    nat: bool
    encapsulation_bytes: int
    isolated: bool
    cross_host: bool

def estimate_payload_mtu(base_mtu: int, mode: NetworkMode) -> int:
    """返回应用可安全使用的近似有效 MTU。"""
    nat_overhead = 0  # NAT 主要改地址，不直接像 VXLAN 那样固定占 50B 头
    return base_mtu - mode.encapsulation_bytes - nat_overhead

bridge = NetworkMode(
    name="bridge",
    hops=3,              # veth -> bridge -> host nic
    nat=True,
    encapsulation_bytes=0,
    isolated=True,
    cross_host=False,
)

host = NetworkMode(
    name="host",
    hops=1,              # host stack -> nic
    nat=False,
    encapsulation_bytes=0,
    isolated=False,
    cross_host=False,
)

overlay = NetworkMode(
    name="overlay",
    hops=4,              # veth -> overlay -> vtep -> nic
    nat=False,
    encapsulation_bytes=50,   # 近似 VXLAN 额外头
    isolated=True,
    cross_host=True,
)

assert bridge.isolated is True
assert host.nat is False
assert overlay.cross_host is True
assert estimate_payload_mtu(1500, overlay) == 1450
assert estimate_payload_mtu(1500, bridge) == 1500
```

这个例子表达的不是内核真实实现细节，而是三个判断：

- `bridge` 有隔离，常见本机转发和 NAT
- `host` 路径最短，但不隔离
- `overlay` 支持跨主机，但有效 MTU 变小

### bridge 模式

查看默认桥接网络：

```bash
docker network inspect bridge
```

启动一个 `nginx` 容器，并把宿主 `8080` 映射到容器 `80`：

```bash
docker run -d --name web1 --network bridge -p 8080:80 nginx
```

访问逻辑是：

1. 客户端访问 `宿主IP:8080`
2. 宿主上的端口映射规则命中
3. 流量被转发到容器 `80`

这是最常见的新手入门例子。容器本身仍是独立网络命名空间，只是宿主额外帮你开了一个入口。

### host 模式

```bash
docker run -d --name web2 --network host nginx
```

此时如果 `nginx` 在容器里监听 `80`，宿主的 `80` 就直接被占用。你通常不会再写 `-p 8080:80`，因为 `host` 模式下没有必要，也无法靠它获得隔离。

### overlay 模式

先初始化 Swarm：

```bash
docker swarm init
```

创建 overlay 网络：

```bash
docker network create -d overlay myoverlay
```

在该网络中部署服务：

```bash
docker service create \
  --name web \
  --network myoverlay \
  --replicas 2 \
  nginx
```

真实工程例子：两台宿主机构成一个 Swarm 集群，`api` 服务和 `redis` 服务都加入 `myoverlay`。此时 `api` 容器通常可以直接通过服务名 `redis` 做 DNS 解析并访问，而不需要手工维护对端 IP。对微服务部署来说，这比单纯 bridge 网络更适合多节点环境。

---

## 工程权衡与常见坑

选网络模式，不是背命令，而是做工程权衡。

### 1. 三种模式的核心权衡

| 模式 | 优点 | 代价 | 最常见误用 |
|---|---|---|---|
| bridge | 默认可用、隔离好、端口映射清晰 | 多一层转发，排障要看 bridge/NAT | 以为容器 IP 能被外网直接访问 |
| host | 路径短、性能通常更好 | 端口冲突、隔离弱 | 一台机上起多个相同端口服务 |
| overlay | 跨主机互通、服务发现方便 | VXLAN 开销、MTU 敏感、排障复杂 | 单机场景也强行用 overlay |

### 2. 常见坑与规避策略

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| host 端口冲突 | 第二个容器启动失败或服务异常 | 容器直接占用宿主端口 | 统一端口规划；同类服务避免 `host` 并发 |
| host 下误用 `-p` | 以为做了映射，实际无意义 | 宿主网络栈已直接暴露 | 把 `host` 理解成“没有独立容器端口空间” |
| bridge 出不了网 | 容器能启动但访问外网失败 | NAT 或转发规则缺失 | 检查 iptables、IP 转发、宿主防火墙 |
| bridge 对外访问失败 | 容器内能 curl，本机外部不能访问 | 没做 `-p` 映射 | 明确“出站”和“入站”是两件事 |
| overlay 丢包或超时 | 跨节点偶发失败 | VXLAN 封装触发 MTU/分片问题 | 先检查 MTU，常见调整到 `1450` |
| overlay 排障困难 | 服务名能解析但连不通 | VTEP、隧道、节点连通性异常 | 从节点互通、隧道状态、MTU 三处排查 |

### 3. 新手最容易混淆的三件事

第一，**容器有 IP，不等于外部可达**。  
`bridge` 下的容器 IP 多数只是宿主内部虚拟网络地址。外部通常看不到，也不会路由到那里。

第二，**`host` 不是“更高级的 bridge”**。  
它不是桥接优化版，而是直接绕过独立容器网络栈的一部分隔离设计。

第三，**`overlay` 不是“跨主机 bridge”那么简单**。  
它底层依赖封装隧道和控制平面维护状态，所以性能特征、故障模式、运维难度都不一样。

### 4. 一个真实工程排障例子

某个 Swarm 集群里，`order` 服务调用 `payment` 服务时，平时正常，只有大报文或 TLS 握手阶段偶发超时。应用日志看不出问题，DNS 解析也正常。最终排查发现是宿主物理网络 MTU 为 `1500`，overlay 保持默认配置，VXLAN 封装后部分报文超出链路承载能力，产生分片与丢包。

这种问题的特征是：

- 小请求常正常
- 大响应更容易出错
- 重试后偶尔恢复
- 单机压测看不出，跨节点更明显

处理顺序通常是：

1. `docker network inspect myoverlay`
2. 核对宿主网卡 MTU
3. 核对节点间 VXLAN/UDP 通路
4. 把 overlay 有效 MTU 调整到约 `1450`
5. 再做跨节点压测验证

---

## 替代方案与适用边界

当 `bridge`、`host`、`overlay` 不够合适时，还有替代选择，但每个替代都意味着新的约束。

| 方案 | 应用场景 | 优点 | 代价 |
|---|---|---|---|
| macvlan | 容器需要物理网络同网段 IP | 容器像独立物理主机 | 配置复杂，宿主与容器互通需额外设计 |
| none | 完全自定义网络 | 最干净，适合实验或安全隔离 | 一切都要自己接 |
| CNI 插件网络 | 更复杂的容器编排环境 | 能力强、策略丰富 | 需要额外控制平面和运维体系 |
| Kubernetes 网络方案 | 大规模集群 | 服务发现、策略、网络模型更系统 | 学习成本和组件复杂度更高 |

### macvlan 什么时候值得考虑

**macvlan** 可以理解为“直接让容器在物理二层网络里拥有自己的身份”。  
它适合的典型场景是：老系统按 IP 白名单接入，或者某些设备只接受同网段真实地址，不适合 NAT 或 overlay 抽象。

但它不适合默认拿来替代 `bridge`，因为：

- 配置门槛更高
- 宿主与容器互通经常需要额外处理
- 对交换机、安全策略、网络规划要求更严格

### none 什么时候有意义

`none` 模式几乎就是“只给你一个网络命名空间，别的什么都不接”。  
适合教学实验、特殊安全隔离、或者由你自己接入自定义虚拟网络。对初级工程师来说，它更多是理解边界用，而不是日常业务首选。

### 更大规模时为什么常转向 CNI/Kubernetes

当需求从“几台 Docker 主机”变成“几十上百节点、网络策略、服务治理、可观测性”时，单纯依赖 Docker 原生网络驱动往往不够。此时会转向 CNI 体系或 Kubernetes 网络方案，因为它们不只是“连通”，还要解决：

- 大规模 IP 管理
- 网络策略
- 跨节点服务发现
- 可观测性
- 安全边界

所以替代方案的选择规则很简单：

- 单机默认先看 `bridge`
- 单机极致性能再评估 `host`
- 多节点服务互通先看 `overlay`
- 需要物理网络真实身份再看 `macvlan`
- 需要更强集群网络能力再上 CNI/Kubernetes

---

## 参考资料

1. Docker 官方文档，《Network drivers》
2. Docker 官方文档，《Bridge network driver》
3. Docker 官方文档，《Host network driver》
4. Docker 官方文档，《Overlay network driver》
5. Docker 官方文档，《Networking tutorials》
6. Linux 容器网络相关资料，关于 network namespace、veth、bridge、iptables 的基础机制
7. VXLAN 相关实践资料，关于封装开销、VTEP、MTU 调整与跨主机排障

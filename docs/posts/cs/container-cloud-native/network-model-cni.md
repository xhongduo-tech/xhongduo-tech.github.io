---
title: 网络模型（CNI/Service Mesh）
date: 2026-08-07
---

# 网络模型（CNI/Service Mesh）

<div class="epigraph">
<p>Kubernetes 的网络模型是一句非常简单的承诺：每个 Pod 都像一个独立的机器，拥有自己的 IP。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.15 ｜ 2026-08-07</p>
</div>

## 为什么网络模型先于实现

存储我们靠 CSI 解决了「谁都能接入」，网络的问题更尖锐：容器被调度到任何一台节点，Pod 随时新建、销毁、搬家，**IP 随手分配**。如果网络地址随节点/位置漂移，服务发现、防火墙、监控全都无从谈起。Kubernetes 先立下一条**网络模型**（一组承诺），再由不同实现去兑现——**先定语义，再谈实现**，这正是插件化哲学（CSI/CNI/CRI）的统一思路。

## 1 网络模型的四条承诺

Kubernetes 网络模型（flat network）规定：

1. **每个 Pod 一个独立 IP**，无需 NAT，Pod 之间可以直接用这个 IP 通信。
2. **Pod 与节点上的 IP 可以互相路由**——从节点访问 Pod 与从 Pod 访问节点都直接可达。
3. **不靠 IP 伪装（masquerading）**：通信双方看到的是**真实的**源/目标 IP——这让审计、策略、遥测都可信。
4. **Pod 内的容器共享该 IP**（同一 net namespace，见《容器原理》）。

**核心概念：扁平网络**：所有 Pod 都在同一个二层可达的「虚拟大网」里，无论它们跑在哪台节点上。<span class="marginnote">「Pod 就像一台机器」的承诺，让应用可以按「写单机程序」的方式思考分布式：本地 socket、localhost、固定端口都仍然成立。网络模型的伟大之处，就是把「分布式的复杂性」隔离在平台层。</span>

**补充｜延伸：Pod 网段要提前规划**。模型承诺「每 Pod 一个 IP」，意味着 Pod 网段（如 `10.0.0.0/16`，约 65534 个地址）必须与节点网段、Service 网段（如 `10.96.0.0/12`）互不重叠，且预留扩展空间——集群建好后改网段是灾难级的操作。**地址规划是网络模型的地基工程**，在建集群第一天就该定下来。

## 2 CNI：网络插件的统一接口

**CNI（Container Network Interface）**：kubelet 在创建 Pod 时调用的标准接口，规定「给容器接入网络」的动作（ADD/DEL/CHECK）与入参（网络配置 JSON）。任何实现只要遵循 CNI，就能决定「Pod 的 IP 从哪来、流量怎么走」。主流实现：

- **Bridge（桥接）**：每个 Pod 一个 `veth` 网线，接入节点上的 Linux bridge——同节点 Pod 直接桥接互通。
- **Overlay（覆盖网络）**：节点间用隧道封装（**VXLAN**）把 Pod 流量「打包」在宿主机网络里传输——解耦 Pod 网络与底层网络，最灵活（Flannel、部分 Calico 模式）。
- **Calico**：基于 **BGP** 的纯三层路由，不封装，性能好，也支持 NetworkPolicy。
- **Cilium**：基于 **eBPF** 的数据面，性能与可编程性俱佳，是当前社区最活跃的方向。

一个 Pod 拿到 IP 的完整过程（bridge 方案为例）：kubelet 调度器确认 Pod 落在节点上 → 运行时创建容器 → 调用 CNI 插件 `ADD` → 插件创建 `veth` 对（一端在容器 net namespace，一端挂到节点 bridge）→ 分配一个 IP → 写入路由与 ARP 表 → 回告 CNI 成功。**「一个 Pod 一个 IP」的实现，就是把一段普通 Linux 网络的搭建自动化**。

| 实现 | 原理 | 封装 | 网络策略 | 适合场景 |
| --- | --- | --- | --- | --- |
| Flannel | VXLAN 覆盖 | 有 | 无 | 快速起步 |
| Calico | BGP 三层路由 | 无 | 有 | 性能敏感 |
| Cilium | eBPF 数据面 | 无/有 | 有 | 高性能 + 可编程 |

排障时最常用的两个命令：节点上 `ip route` 看路由表、`bridge link` 看桥接端口；Pod 里 `ip a` 看本 Pod 的 IP、`ip route` 看默认路由。对 overlay 方案，`ip link` 里会出现 `vxlan0`、`flannel.1` 之类的隧道接口——MTU 是否调对、隧道接口是否 UP，一眼可见。**网络排障的起点永远是「先把数据平面的每一跳看清」**，而不是凭感觉猜。

**公式解析：覆盖网络的封装代价**：VXLAN 在 Pod 包外再包一层宿主机可路由的包头：

$$
P_{\text{wire}} = \text{VXLAN}(P_{\text{pod}}), \qquad \text{MTU}_{\text{eff}} = \text{MTU}_{\text{host}} - 50
$$

- $P_{\text{pod}}$：Pod 发出的原始数据包（其目标 IP 是另一个 Pod）。
- $\text{VXLAN}(\cdot)$：在包外加 VXLAN/外层 IP/UDP 头（约 50 字节）。
- $\text{MTU}_{\text{eff}}$：Pod 网卡实际可用 MTU。

三步拆解：

- **第一步，封装是「套娃」**：宿主机看到的是外层包（可正常路由），Pod 看到的是内层包（语义不变）。覆盖网络让「Pod 网络」与「底层网络」彻底解耦——这也是它好部署的原因。
- **第二步，封装有代价**：每包多 ~50 字节，MTU 缩小。如果 Pod 侧 MTU 没调对，就会触发分片，性能雪崩——这是 overlay 方案最常见的排障点。
- **第三步，无封装 vs 有封装是根本取舍**：BGP 直路（Calico）快但依赖底层网络支持；VXLAN（Flannel/Cilium 隧道）通用但有开销。**eBPF（Cilium）试图两者兼得：直路转发的速度 + 可编程的控制**。

算一个具体数字：宿主机网卡 MTU 1500，VXLAN 封装吃掉约 50 字节，Pod 网卡的有效 MTU 应设 1450。如果 Pod 侧仍用默认 1500，超过 1450 字节的包就会在隧道出口被分片——流量一大，转发性能断崖下跌，且表现为「偶发超时、重传激增」这类难以定位的症状。**集群里「调了 MTU 才能解的小概率故障」，几乎是 overlay 方案的必修课**。

## 3 节点上的数据面：kube-proxy 与 DNS

Pod 之间的直连有了，但「服务发现」还要落地（见《Pod 与服务发现》）：

- **kube-proxy**：在每台节点上把「发往 Service clusterIP 的包」改写目标为某个后端 Pod IP。现代实现用 **iptables**（规则链）或 **IPVS**（内核级负载均衡）。**IP 伪装（DNAT）是这里唯一允许的地址改写**——Service 是网络模型的一个刻意例外。<span class="marginnote">Service 的 DNAT 是网络模型里唯一的「作弊」：Pod 间直连不 NAT，但访问 Service 时必然发生目标改写。理解这一点，你就理解了为何 Service 只做 L4 分发——L7 的精细控制是服务网格的活。</span>
- **CoreDNS**：集群内 DNS，解析 `service.namespace.svc.cluster.local` 名字，返回 clusterIP 或 Pod IP 列表（headless）。

iptables 与 IPVS 的差别值得量化：iptables 方案下，每加一条 Service 规则都要在链里多匹配一次，规则多到几千条时性能明显下滑；IPVS 把负载均衡搬进内核的哈希表，转发性能稳定得多。**Service 数量多的集群，通常从 iptables 切到 IPVS**——代价是 IPVS 的会话保持行为与 iptables 略有不同，需要单独验证。

一个完整的服务名解析流程值得走一遍：Pod 里的进程请求 `web.default.svc.cluster.local` → CoreDNS 返回 `10.96.0.10`（clusterIP）→ 包发往该 IP → 节点上的 kube-proxy 把目标改写为某个后端 Pod IP → 到达真正的 Pod。**这条链路里，DNS 提供「名字到 VIP」，kube-proxy 提供「VIP 到真实 Pod」**——两段合起来，客户端才始终只看到一个稳定的名字。若把 `clusterIP: None` 设为 headless，DNS 则直接返回全部后端 Pod IP 列表，供有状态应用按名字逐个访问。

## 4 辨析｜易错点：Service 与网络模型的关系

**最常见的误解**：以为「Pod 访问 Service 也遵守『不 NAT』承诺」。错——模型承诺的是 **Pod 之间**直连不 NAT；**Service 的 clusterIP 本身就是 NAT 的入口**，这是设计内例外。

**第二个易错点**：**扁平网络 ≠ 安全网络**。所有 Pod 互通是默认状态——「Pod 能互访」是**网络模型的一部分**，不是 bug。要限制谁访问谁，靠 **NetworkPolicy**（见《安全》一课），它是叠加在扁平网络之上的**策略层**，不是网络模型本身。

**第三个易错点**：CNI 插件管「Pod 网络」，服务网格管「服务间流量」，Ingress 管「外部进集群」——三者分工不同、可以叠加：**CNI 让 Pod 有网，Ingress 让外面能进来，服务网格让内部流量被治理**。三者叠加后的完整链路是：外部请求 → Ingress（L7 入口）→ Service（L4 稳定入口）→ 边车 Envoy（网格治理）→ Pod 容器。每一层解决一个问题，各管一段——排障时先问「流量卡在哪一段」，就知道该看哪一层的数据。

**辨析｜易错点：网络模型只管「集群内」**。四条承诺描述的是 Pod 之间、Pod 与节点之间的连通性；**集群外的访问**（用户浏览器 → 服务）不属于网络模型——那是 Ingress 的职责（见《Ingress 与流量管理》）；Pod 要访问公网，往往也需要 egress 策略或 NAT。**「Pod 直连」是内网承诺，「对外可达」是另一个话题**，两者别混为一谈。

## 5 网络模型与分布式模式

回看《分布式系统设计模式》的复制/分片/事件驱动：复制要求「任意副本可直接被访问」（扁平网络保证）、分片要求「流量可路由到任一片」（IP 直连保证）、事件驱动要求「异步消息可达」（网络可达性保证）。**网络模型是一切分布式模式的物理前提**——它把「机器之间互连」这件大事，压缩成了「每个 Pod 一个 IP」这一条承诺。

跨集群的场景需要补一句：网络模型的「每 Pod 一 IP」假设集群是**单一边界**。跨集群（联邦、多云）时，Pod IP 可能重叠，需要 **服务网格多集群** 或 **Cilium ClusterMesh** 这类方案，把「集群内名字」翻译成「跨集群地址」。**网络模型的边界，就是编排边界的边界**——理解它，你才能判断哪些功能天然跨集群、哪些需要额外机制。

把本课的高频术语收进一张速查表：

| 术语 | 含义 |
| --- | --- |
| veth | 连接两个网络命名空间的虚拟网线 |
| bridge | 内核里的二层交换机 |
| VXLAN | 在三层之上封装的二层隧道 |
| BGP | 边界网关协议，三层的路由宣告 |
| eBPF | 内核内的高性能可编程数据面 |
| MTU | 最大传输单元，决定单包上限 |

## 6 小结

- 网络模型四条承诺：**每 Pod 一 IP、直连无 NAT、真实地址、Pod 内共享**。
- **CNI** 是网络插件接口；实现分 **桥接、覆盖（VXLAN）、BGP（Calico）、eBPF（Cilium）**，各有性能与复杂度的取舍。
- 覆盖网络每包 ~50 字节封装代价，MTU 要配套调整。
- **kube-proxy**（iptables/IPVS）做 Service 的 DNAT 改写，**CoreDNS** 解析服务名。
- Service 的 NAT 是模型内例外；**扁平网络默认全通，隔离靠 NetworkPolicy**。
- 服务名解析 = **DNS（名字→VIP）+ kube-proxy（VIP→真实 Pod）** 两段；headless 服务直接返回 Pod IP 列表。
- 网络模型只管**集群内**连通；对外访问靠 Ingress，跨集群靠网格多集群等机制。
- 排障三件套：`ip route`、`bridge link`、`ip a`——先把数据平面的每一跳看清。

在下一节，我们把「外部流量进集群」这一环补齐——进入 **Ingress 与流量管理/负载均衡**。

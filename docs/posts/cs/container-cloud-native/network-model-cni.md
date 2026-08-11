---
title: 网络模型（CNI/Service Mesh）
date: 2026-08-11
---

# 网络模型（CNI/Service Mesh）

<div class="epigraph">
<p>Kubernetes 的网络模型是一句非常简单的承诺：每个 Pod 都像一个独立的机器，拥有自己的 IP。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
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

## 2 CNI：网络插件的统一接口

**CNI（Container Network Interface）**：kubelet 在创建 Pod 时调用的标准接口，规定「给容器接入网络」的动作（ADD/DEL/CHECK）与入参（网络配置 JSON）。任何实现只要遵循 CNI，就能决定「Pod 的 IP 从哪来、流量怎么走」。主流实现：

- **Bridge（桥接）**：每个 Pod 一个 `veth` 网线，接入节点上的 Linux bridge——同节点 Pod 直接桥接互通。
- **Overlay（覆盖网络）**：节点间用隧道封装（**VXLAN**）把 Pod 流量「打包」在宿主机网络里传输——解耦 Pod 网络与底层网络，最灵活（Flannel、部分 Calico 模式）。
- **Calico**：基于 **BGP** 的纯三层路由，不封装，性能好，也支持 NetworkPolicy。
- **Cilium**：基于 **eBPF** 的数据面，性能与可编程性俱佳，是当前社区最活跃的方向。

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

## 3 节点上的数据面：kube-proxy 与 DNS

Pod 之间的直连有了，但「服务发现」还要落地（见《Pod 与服务发现》）：

- **kube-proxy**：在每台节点上把「发往 Service clusterIP 的包」改写目标为某个后端 Pod IP。现代实现用 **iptables**（规则链）或 **IPVS**（内核级负载均衡）。**IP 伪装（DNAT）是这里唯一允许的地址改写**——Service 是网络模型的一个刻意例外。<span class="marginnote">Service 的 DNAT 是网络模型里唯一的「作弊」：Pod 间直连不 NAT，但访问 Service 时必然发生目标改写。理解这一点，你就理解了为何 Service 只做 L4 分发——L7 的精细控制是服务网格的活。</span>
- **CoreDNS**：集群内 DNS，解析 `service.namespace.svc.cluster.local` 名字，返回 clusterIP 或 Pod IP 列表（headless）。

## 4 辨析｜易错点：Service 与网络模型的关系

**最常见的误解**：以为「Pod 访问 Service 也遵守『不 NAT』承诺」。错——模型承诺的是 **Pod 之间**直连不 NAT；**Service 的 clusterIP 本身就是 NAT 的入口**，这是设计内例外。

**第二个易错点**：**扁平网络 ≠ 安全网络**。所有 Pod 互通是默认状态——「Pod 能互访」是**网络模型的一部分**，不是 bug。要限制谁访问谁，靠 **NetworkPolicy**（见《安全》一课），它是叠加在扁平网络之上的**策略层**，不是网络模型本身。

**第三个易错点**：CNI 插件管「Pod 网络」，服务网格管「服务间流量」，Ingress 管「外部进集群」——三者分工不同、可以叠加：**CNI 让 Pod 有网，Ingress 让外面能进来，服务网格让内部流量被治理**。

## 5 网络模型与分布式模式

回看《分布式系统设计模式》的复制/分片/事件驱动：复制要求「任意副本可直接被访问」（扁平网络保证）、分片要求「流量可路由到任一片」（IP 直连保证）、事件驱动要求「异步消息可达」（网络可达性保证）。**网络模型是一切分布式模式的物理前提**——它把「机器之间互连」这件大事，压缩成了「每个 Pod 一个 IP」这一条承诺。

## 6 小结

- 网络模型四条承诺：**每 Pod 一 IP、直连无 NAT、真实地址、Pod 内共享**。
- **CNI** 是网络插件接口；实现分 **桥接、覆盖（VXLAN）、BGP（Calico）、eBPF（Cilium）**，各有性能与复杂度的取舍。
- 覆盖网络每包 ~50 字节封装代价，MTU 要配套调整。
- **kube-proxy**（iptables/IPVS）做 Service 的 DNAT 改写，**CoreDNS** 解析服务名。
- Service 的 NAT 是模型内例外；**扁平网络默认全通，隔离靠 NetworkPolicy**。

在下一节，我们把「外部流量进集群」这一环补齐——进入 **Ingress 与流量管理/负载均衡**。

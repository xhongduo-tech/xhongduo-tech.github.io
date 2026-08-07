---
title: K8s 网络与存储
date: 2026-08-07
---

# K8s 网络与存储

<div class="epigraph">
<p>Pod 生而短暂，数据必须永恒。</p>
<footer>—— Kubernetes 存储设计原则</footer>
</div>

<div class="article-byline">
<p>第三级 · 云计算 ｜ Kubernetes 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 K8s 网络与存储开始

编排三件套解决了「应用怎么跑」，但生产环境还要回答两个更底层的问题：**Pod 之间怎么互通**（网络）、**Pod 死了数据怎么办**（存储）。K8s 对这两者都有一套抽象：网络侧是「每个 Pod 一个 IP」的扁平模型 + CNI 插件；存储侧是 **PV/PVC** 两级抽象。这一篇把这两块拼图补上，你就有了一幅完整的 K8s 全景图。

## 1 K8s 网络模型：扁平网络三原则

K8s 对集群网络提出三条硬性要求，任何 CNI 插件都必须满足：

1. **每个 Pod 都有一个独立的 IP**，Pod 之间可以直接互通（无需 NAT）。
2. **节点上的 Pod 与节点本身互通**，Pod 可以访问节点。
3. **Pod 看到的 IP 就是它自己的 IP**，不做地址转换——保证网络对应用透明。

这三条合起来叫**扁平网络模型（flat network）**：整个集群就像一个大二层网络，任何 Pod 通过 IP 直接访问任何其他 Pod，跟传统物理机房的「机器之间直连」完全一样。好处是应用不需要感知「自己在容器里」，网络行为与裸机一致。<span class="marginnote">「Pod 即 IP」让服务发现变得简单：无需端口映射、无需 NAT，Pod 之间就像局域网里的主机。代价是 IP 数量爆炸——集群里几千个 Pod 就要几千个 IP，这正是 Overlay 网络（VXLAN）与 IP 段规划要解决的问题，可回看《网络虚拟化》一篇的 VXLAN。</span>

## 2 CNI 插件：网络的「司机」

K8s 本身**不实现网络**，只规定模型。真正实现网络的是 **CNI（Container Network Interface）插件**——它负责在 Pod 创建时为它分配 IP、配置虚拟网卡与路由。常见 CNI 分两类：

- **Overlay 型**：用 VXLAN 隧道构建虚拟网络。如 **flannel**（简单、易用）、**Weave**。
- **路由型**：用 BGP 等路由协议把各节点的 Pod 网段广播到物理网络，Pod 直接走物理路由。如 **Calico**（性能高、支持网络策略）、**Cilium**（基于 eBPF，现代高性能方案）。

**辨析｜易错点：** 别把 CNI 与 kube-proxy 混为一谈。CNI 解决「Pod 之间怎么连通」（给 Pod 发 IP、配路由），kube-proxy 解决「Service 的流量怎么转发到 Pod」。前者是**网络连通**，后者是**负载均衡**——两个层面，两类组件。生产实践里常搭配使用（如 Calico 做连通 + kube-proxy 做转发）。

## 3 K8s 存储：PV、PVC 与 StorageClass

容器是无状态的，但数据库、文件服务必须有状态。K8s 存储的核心抽象是**存储卷（Volume）**，其中最关键的三个对象：

- **PV（PersistentVolume，持久卷）**：集群管理员预先准备好的一块存储（可以是云盘、NFS、本地盘），是**资源**。
- **PVC（PersistentVolumeClaim，持久卷声明）**：应用提出的「我要一块 100 GB 可读写存储」的**需求**。
- **StorageClass**：PV 的「模板与工厂」，按需动态创建 PV（如指定云盘类型、性能等级）。

**核心机制：PVC 绑定 PV**——应用声明需求（PVC），系统找一个满足需求的 PV 绑定（或由 StorageClass 动态创建）。应用只与 PVC 打交道，不关心底层是云盘还是 NFS。这又是一层典型的抽象：**需求与供给解耦，声明与实现分离**。<span class="marginnote">把 PV/PVC 类比成「租房」：PV 是「房源」，PVC 是「租房需求」，StorageClass 是「房屋开发商」。租客（应用）只看需求（我要多大、多快），不关心房子在哪个小区——这是存储的「接口即承诺」。</span>

## 4 核心要点对比：网络插件与存储抽象

| 层次 | 对象/插件 | 职责 | 典型实现 |
| --- | --- | --- | --- |
| 网络连通 | CNI 插件 | Pod 分配 IP、配置路由 | flannel、Calico、Cilium |
| 流量转发 | kube-proxy | Service → Pod 负载均衡 | iptables、IPVS |
| 存储供给 | PV | 一块准备好的持久卷 | 云盘、NFS、本地盘 |
| 存储需求 | PVC | 应用声明存储需求 | 100GB、ReadWriteOnce |
| 动态供给 | StorageClass | 按需创建 PV | 按云盘类型模板 |

**辨析｜易错点：** 存储卷的生命周期**独立于 Pod**——Pod 重建、迁移、删除，卷都还在，新 Pod 可以重新挂载同一块卷。这是「有状态应用上 K8s」的关键前提。但要注意：普通 PV 只能被一台节点上的 Pod 挂载（ReadWriteOnce），跨节点共享需要专门的共享存储（如 NFS、CephFS）。「以为所有 PV 都能多机共享」是上云排障中最常见的误解之一。<span class="marginnote">有状态应用（数据库）在 K8s 上运行通常需要 StatefulSet + 动态 PVC 的组合——StatefulSet 保证「稳定的身份与稳定的存储绑定」，每个副本固定一块自己的卷。这是比 Deployment 更进阶的一课，了解存在即可。</span>

## 5 Ingress：集群的流量入口

有了 Pod 互通与 Service 转发，集群对外的流量还差最后一环——**Ingress**。

**Service 的局限**：Service 的 LoadBalancer 类型每暴露一个服务就要创建一个云负载均衡器，服务一多，LB 数量爆炸、成本失控、管理混乱。**Ingress** 用「一个入口 + 规则路由」解决这个问题：

- **Ingress（对象）**：声明「外部请求按什么规则路由到哪个 Service」——按域名、按路径、按 TLS。
- **Ingress Controller（实现）**：真正执行路由的组件（Nginx Ingress、ALB Ingress），读取 Ingress 规则并实现转发。

一个典型规则：`api.example.com/* → api-service`，`www.example.com/static/* → static-service`——**一个公网入口，按域名与路径分发到多个内部 Service**。<span class="marginnote">Ingress 与 Service 的分工可以记成：<strong>Service 是「集群内的路标」，Ingress 是「大门口的收发室」</strong>。外部流量先到 Ingress（收发室），它按规则（这封信给谁）转发到对应 Service（路标），再由 Service 分发给具体 Pod。这层「统一入口 + 规则路由」正是《负载均衡》七层能力在 K8s 内的体现。</span>

**Ingress 还能做**：TLS 终止（统一挂证书，后端不用管 https）、限流、重写路径、灰度分流。它让集群的「南北向流量」集中管理，是微服务对外暴露的标准姿势。

**辨析｜易错点：** Ingress 只是「**定义规则的对象**」，真正干活的是 Ingress Controller——**没有装 Controller，Ingress 规则形同虚设**（只创建 Ingress 对象但不安装 Nginx Ingress，流量不会自己路由）。初学 K8s 网络最容易踩的坑就在这：对象有了，实现没装，于是一头雾水「为什么规则不生效」。

## 6 有状态应用与 StatefulSet：数据库怎么上 K8s

Deployment 适合无状态应用，但数据库、消息队列这类**有状态应用**需要「稳定的身份」与「独立的存储」——K8s 用 **StatefulSet** 满足它们。

**有状态应用的两大需求**：

1. **稳定网络身份**：数据库副本需要固定的名字（`db-0`、`db-1`），重启后名字不变，客户端才能稳定连接。
2. **稳定存储绑定**：每个副本对应一块**自己的 PVC**，重建后仍挂载同一块卷——数据不随 Pod 消失。

**StatefulSet 与 Deployment 的对比**：

| 维度 | Deployment | StatefulSet |
| --- | --- | --- |
| Pod 命名 | 随机（`app-abc123`） | 有序固定（`db-0`） |
| 存储 | 可选共享卷 | 每副本独立 PVC |
| 扩缩容 | 任意并发 | 按序进行（0,1,2…） |
| 典型负载 | 无状态微服务 | 数据库、缓存、ZooKeeper |

**辨析｜易错点：** 有状态应用上 K8s 的**复杂度远高于无状态**——主从切换、数据备份、扩缩容时的数据再平衡，都要自己设计。这也是为什么许多团队「数据库仍用云托管（RDS）」，只在 K8s 上跑无状态应用——**托管数据库把有状态的痛苦外包给了云**（见《云数据库服务》）。K8s 跑有状态是「能」与「愿不愿承担复杂度」的两回事。<span class="marginnote">StatefulSet 的「有序扩缩容」是深思熟虑的：数据库扩一个副本，要等它同步完成再扩下一个，防止「一堆副本同时抢数据」——<strong>用「有序」换「安全」</strong>。这种「为数据一致性放弃并行性」的设计，与分布式系统里「单点定序」（见《分布式文件系统》）是同一个思想。</span>

## 7 NetworkPolicy：集群内的「防火墙」

K8s 的 Pod 之间默认**全互通**——任何 Pod 能访问任何 Pod。这在安全上不可接受（比如数据库 Pod 不该被随便访问），于是有了 **NetworkPolicy（网络策略）**。

**NetworkPolicy**：声明「哪些 Pod 能被哪些来源访问」的规则对象，由网络插件（Calico/Cilium）强制执行。一个典型规则：`允许只来自「标签为 app=web」的 Pod 访问数据库 Pod 的 3306 端口`——按**标签**而非 IP 做访问控制。

**辨析｜易错点：** NetworkPolicy 有两个「坑」：

1. **默认不生效**：只有定义了 NetworkPolicy，流量才会被限制；**不定义 = 全放行**（默认宽松）。
2. **依赖网络插件**：flannel 这类简单 Overlay 插件**不支持** NetworkPolicy；要用它得选 Calico、Cilium 等支持策略的插件。

| 插件 | 支持 NetworkPolicy | 特点 |
| --- | --- | --- |
| flannel | 否 | 简单 Overlay |
| Calico | 是 | 路由型 + 策略 |
| Cilium | 是 | eBPF + 精细策略 |

**NetworkPolicy 是「零信任」在集群内的落地**：不再假设「集群内都是自己人」，而是「默认拒绝、显式放行」——这与《云安全与合规》的最小权限、以及云 VPC 的安全组（《云网络》）是同一思想在不同层级的表现。<span class="marginnote">一层很清晰的「防火墙谱系」：<strong>云 VPC 安全组管「实例间」，K8s NetworkPolicy 管「Pod 间」</strong>——都是「按标签/角色、默认最小开放」的访问控制。理解安全组，就理解 NetworkPolicy 的一半；安全思想在不同抽象层反复复用的规律，在这里再次显现。</span>

## 8 小结

- K8s 网络模型三原则：**Pod 独有 IP、Pod 直连、无 NAT 透明**。
- CNI 插件实现网络（flannel/Calico/Cilium），kube-proxy 实现 Service 转发——**两层职责别混淆**。
- 存储三件套：**PV**（资源）、**PVC**（需求）、**StorageClass**（动态工厂）。
- 卷生命周期**独立于 Pod**，有状态应用靠 StatefulSet + 动态 PVC。
- 网络与存储的抽象共同保证：**Pod 随时可换，网络随时可通，数据永不丢失**。

- 网络与存储的共性：**两者都是「让 Pod 可任意替换」的前提**——网络保证新 Pod 可通，存储保证新 Pod 数据还在；没有这两条，Pod 的「可替换性」只是空谈。

在下一篇，我们看向集群的「大脑」如何做决策——**K8s 调度与伸缩**，Pod 被派到哪台节点、业务高峰怎么自动加副本。网络与存储回答「Pod 怎么活得好」，调度与伸缩回答「Pod 被派到哪、业务高峰怎么扛」——**从「单 Pod 的健康」走向「整个集群的理智」**。

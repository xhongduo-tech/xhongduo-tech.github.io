---
title: Pod 与服务发现
date: 2026-08-07
---

# Pod 与服务发现

<div class="epigraph">
<p>Pod 是 Kubernetes 的最小部署单元，Service 是它永恒的地址；一个负责存在，一个负责被找到。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.7 ｜ 2026-08-07</p>
</div>

## 为什么 Pod 是「最小的」

上节说「一切皆对象」，但调度器调度的最小单位不是容器，而是 **Pod**。这是新手最容易产生的一个问号：为什么还要在容器外面再包一层？答案是：容器只有**进程级**的隔离，而很多场景需要一组进程**共享同一个 IP、共享本地存储、共享生命周期**——代理进程跟在主进程身边，缓存进程要读主进程写的数据。Pod 就是「必须同生共死、必须住在一起」的这组容器。

## 1 Pod：一组共享命名的容器

**核心概念：Pod**：Kubernetes 中最小的调度与部署单元，包含一个或多个容器。Pod 内的所有容器**共享同一个网络命名空间**（同一 IP、同一端口空间、同一回环接口）与可选的**共享存储卷**。<span class="marginnote">Pod 内容器共享网络命名空间意味着：容器 A 监听 8080，容器 B 可以直接用 <code>localhost:8080</code> 访问它——它们在同一台「逻辑主机」上。这在普通容器（各有各的 net namespace）里是做不到的。</span>

Pod 是一台「逻辑机器」：

**一个 Pod 一个 IP**：Pod 拥有集群可路由的独立 IP，Pod 之间可以用该 IP 直连（网络模型保证这一点，见《网络模型》）。
- **共享卷**：Pod 内容器可挂载同一卷，读同一份数据。
- **同一生命周期**：Pod 内所有容器一起被调度、一起被销毁。一个容器崩了，kubelet 会重启它；Pod 本身仍在。
- **探针与生命周期钩子**：kubelet 通过 `livenessProbe`（活着吗）、`readinessProbe`（能接流量吗）、`startupProbe`（启动好了吗）管理 Pod 状态。

Pod 的典型副作用是**边车（sidecar）模式**：主容器 + 日志采集容器、网络代理容器（如 Istio 的 Envoy）、配置刷新容器——它们共享网络与卷，形成一个「团队」。

三种探针的分工值得单独记牢：

| 探针 | 回答的问题 | 失败后果 |
| --- | --- | --- |
| `livenessProbe` | 进程还活着吗 | 杀掉容器重启 |
| `readinessProbe` | 能接流量吗 | 从 Service 后端摘除（不杀） |
| `startupProbe` | 启动好了吗 | 慢启动保护（防止前两者误判） |

**补充｜延伸：探针的「诚实度」决定 Service 的稳定性**。`readinessProbe` 的返回是 EndpointSlice 增删的依据——探针写得太松，挂了还接流量；写得太紧，启动正常却被误摘。常见实践：HTTP 探针要检查「真的能服务请求」的端点（而不是根路径），并给足 `initialDelaySeconds` 与 `periodSeconds`。**探针不是摆设，它是服务发现正确性的第一道闸**。与探针常被放在一起的还有 **initContainer**——在业务容器启动前先跑一段「准备就绪」逻辑（拉取依赖、等数据库就绪），它做完才轮到主容器，是「启动就绪」的另一种表达。

## 2 Pod 的生命周期

Pod 的状态机：

| 状态 | 含义 |
| --- | --- |
| `Pending` | 已接受，等待调度或拉取镜像 |
| `Running` | 至少一个容器在运行 |
| `Succeeded` | 所有容器正常退出（Job 用） |
| `Failed` | 有容器以非零码退出 |
| `Unknown` | 状态无法获取（节点失联） |

**辨析｜易错点**：Pod 里的 `restartPolicy`（默认 `Always`）控制容器**重启**，但重启的是容器，不是 Pod。如果整个 Pod 需要被替换（节点挂了），那是**控制器**（Deployment、StatefulSet）的职责，Pod 本身不会「自我复活」。**Pod 是「可再生」的——它随时可能被调度器/控制器销毁并重建，所以 Pod 的 IP 与名字都是易变的。** 这也直接引出了 Service。

**补充｜延伸：静态 Pod**。还有一种绕过控制器的 Pod：**静态 Pod**——由 kubelet 直接从节点上的清单目录（`/etc/kubernetes/manifests`）拉起，不经 API Server。控制平面的 `kube-apiserver`、`etcd` 通常就以静态 Pod 方式运行：**即使整个集群的 API 还没起来，kubelet 也能先拉起这些关键组件**。它是集群「自举」的机制，也解释了为什么控制平面组件能这么早就位。

## 3 Service：不变的名字，可变的 Pod

Pod 会消失、会被重建（新名字、新 IP）。如果客户端记的是 Pod IP，Pod 一死就全断。**Service** 是解决这个问题的抽象：给一组 Pod 提供一个**稳定入口**——固定的 `clusterIP` 与固定的 DNS 名字，把流量转发给此刻「真正活着」的 Pod。

Service 与 Pod 的连接同样靠标签：`selector` 选中一组 Pod，Service 后端列表就是它们。后端由 **Endpoint / EndpointSlice** 对象维护——kubelet 汇报的 Pod IP 会动态刷新到这个列表里。<span class="marginnote">Service 的语义是「一个稳定的名字，指向一组易变的 Pod」。这与域名解析（DNS）的哲学一致：名字是长期资产，IP 是临时指环。Service 名字经由集群内 DNS 解析——<code>web.default.svc.cluster.local</code>——形成完整的服务发现链路。</span>

Service 的几种类型：

- **ClusterIP**（默认）：集群内部可访问的稳定虚拟 IP。
- **NodePort**：在每个节点上开一个端口，集群外可经 `nodeIP:port` 访问。
- **LoadBalancer**：由云厂商分配负载均衡器（云控制器实现）。
- **Headless**（`clusterIP: None`）：不做负载均衡，直接返回后端 Pod IP 列表——供有状态应用按名字逐个访问。

| 类型 | 谁能访问 | 特点 |
| --- | --- | --- |
| ClusterIP | 集群内 | 默认，稳定虚拟 IP |
| NodePort | 集群外（经节点端口） | 简单，端口管理粗 |
| LoadBalancer | 集群外（云 LB） | 一个服务一个 LB |
| Headless | 按名字逐个 | 无 clusterIP，直接给 Pod IP |

Service 的 DNS 全名值得拆开看：`web.default.svc.cluster.local` 从左到右是 `服务名.命名空间.svc.cluster.local`——**命名空间进了域名**，所以跨命名空间访问要写 `svcB.otherns.svc`，同命名空间内可只写 `svcB`。DNS 把「命名空间分层」翻译成了域名分层，这是《对象模型》里「分层命名空间」在服务发现上的落地。

## 4 公式解析：服务发现的完整地址

服务发现的全链路是一个「名字 → 地址」的复合映射，可以写成一个嵌套函数：

$$
\text{Addr}(\text{svc}) = \text{clusterIP}(\text{svc}) : \text{port}, \qquad
\text{Pods}(\text{svc}) = \{ p \mid \text{match}(p.\text{labels}, \text{svc}.\text{selector}) \}
$$

- $\text{clusterIP}(\text{svc})$：Service 创建时分配、生命周期内不变的虚拟 IP。
- $\text{Pods}(\text{svc})$：由标签选择器在任一时刻决定的**活的**后端集合。
- 二者结合：**名字稳定，转发目标实时变化**。

三步拆解：

- **第一步，名字查 IP**：客户端访问 `web.default.svc.cluster.local`，集群内 DNS（CoreDNS）返回 `clusterIP`——这一步名字变 IP，稳定。
- **第二步，IP 转发到 Pod**：`kube-proxy` 在节点上把发往 `clusterIP` 的包按负载均衡规则改写目标为某个后端 Pod IP——这一步 IP 变「当前活着的 Pod」，实时。
- **第三步，后端持续刷新**：Pod 挂了，EndpointSlice 移除它；新 Pod 起来，加入它。**客户端全程只看到那个不变的名字。**

走一遍具体场景：`web` 服务 3 个副本，Pod IP 是 `10.1.0.5`、`10.1.0.6`、`10.1.0.7`，Service clusterIP 是 `10.96.0.10`。客户端访问 `web` → DNS 回 `10.96.0.10` → kube-proxy 从 EndpointSlice 里挑一个（比如 `10.1.0.6`）改写目标 → 请求到达 Pod。若此时 `10.1.0.6` 挂了，EndpointSlice 下一轮就把它摘掉，新请求只落到剩下两个——**客户端一行代码都不用改**。<span class="marginnote">「摘除不健康的后端」这一步靠的是 readinessProbe：Pod 没通过就绝不进 EndpointSlice。所以 Service 的「稳定性」最终由探针的诚实程度决定——探针撒谎，Service 就撒谎。</span>

## 5 从「单 Pod」到「一群 Pod」

现在你有了两个基本概念：Pod 管「一个实例」，Service 管「发现一群实例」。但「谁保证始终有 3 个实例、挂了自动补」还不属于这两者——那是**控制器**的职责。调度器把 Pod 放到节点上，Service 让它们被找到，控制器保证数量——三者合起来，才是编排的完整画面。

**补充｜延伸：会话保持与拓扑感知**。默认的 Service 负载均衡是无状态的轮询/随机——同一用户的不同请求可能落到不同后端，这通常没问题（应用应无状态）。若确有需要，可开 `sessionAffinity: ClientIP` 让同源请求粘到同一后端，或以 `topologyKeys` 做「就近优先」。这与《Ingress》一课里讨论的「黏连 vs 均衡」是同一枚硬币——**无状态应用不依赖它，有状态应用才需要它**。

**辨析｜易错点：Headless Service 不是「没有 Service」**。`clusterIP: None` 只是**不做集群级负载均衡**，DNS 直接返回后端 Pod 的 IP 列表——客户端自己决定连谁。这让「按名字逐个访问」成为可能，但也把负载均衡的责任还给了客户端。**StatefulSet 的有状态应用用它做主从发现，无状态应用几乎总该用普通 Service。**

**补充｜延伸：kube-proxy 怎么把包送到 Pod**。它有三种实现：`userspace`（最慢，已废弃）、`iptables`（默认，规则链 NAT 改写）、`ipvs`（哈希查表，支持加权轮询、最小连接等更复杂的调度）。`iptables` 模式把转发规则写进内核 NAT 表，包进来按规则链逐条匹配、改写目标地址——**这是「数据平面」在 Kubernetes 内的最早形态**，比服务网格的 Envoy 更底层。

## 6 小结

- **Pod** 是最小调度单元：一组容器共享网络命名空间与卷，同生共死，IP 与名字**易变**。
- Pod 状态机：`Pending → Running → Succeeded/Failed/Unknown`；容器可重启，Pod 由控制器重建。
- 探针三分工：`liveness`（杀与重启）、`readiness`（摘出后端）、`startup`（慢启动保护）。
- **Service** 提供**稳定名字 + clusterIP**，按标签选中 Pod，经 EndpointSlice 维护**动态**后端。
- 服务发现链路：**DNS 名字 → clusterIP → 当前存活的 Pod**，三层映射，前两层稳定、最后一层实时。
- 边车模式让「辅助进程」与主进程共享网络与卷；会话保持只在有状态需要时开启。
- **静态 Pod** 由 kubelet 直接拉起，不经 API Server——控制平面组件（apiserver/etcd）靠它实现集群自举。
- kube-proxy 用 `iptables`（规则链 NAT）或 `ipvs`（哈希表）改写包目标地址，是 K8s 里最底层的「数据平面」。
- Headless Service 不做集群级负载均衡、直接给 Pod IP，是 StatefulSet 主从发现的基础；无状态应用几乎总该用普通 Service。

在下一节，我们回答「谁保证副本数量、谁做滚动升级」——进入 **控制器与声明式 API**。

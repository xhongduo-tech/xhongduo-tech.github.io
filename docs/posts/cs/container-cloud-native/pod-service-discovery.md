---
title: Pod 与服务发现
date: 2026-08-11
---

# Pod 与服务发现

<div class="epigraph">
<p>Pod 是 Kubernetes 的最小部署单元，Service 是它永恒的地址；一个负责存在，一个负责被找到。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
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

## 3 Service：不变的名字，可变的 Pod

Pod 会消失、会被重建（新名字、新 IP）。如果客户端记的是 Pod IP，Pod 一死就全断。**Service** 是解决这个问题的抽象：给一组 Pod 提供一个**稳定入口**——固定的 `clusterIP` 与固定的 DNS 名字，把流量转发给此刻「真正活着」的 Pod。

Service 与 Pod 的连接同样靠标签：`selector` 选中一组 Pod，Service 后端列表就是它们。后端由 **Endpoint / EndpointSlice** 对象维护——kubelet 汇报的 Pod IP 会动态刷新到这个列表里。<span class="marginnote">Service 的语义是「一个稳定的名字，指向一组易变的 Pod」。这与域名解析（DNS）的哲学一致：名字是长期资产，IP 是临时指环。Service 名字经由集群内 DNS 解析——<code>web.default.svc.cluster.local</code>——形成完整的服务发现链路。</span>

Service 的几种类型：

- **ClusterIP**（默认）：集群内部可访问的稳定虚拟 IP。
- **NodePort**：在每个节点上开一个端口，集群外可经 `nodeIP:port` 访问。
- **LoadBalancer**：由云厂商分配负载均衡器（云控制器实现）。
- **Headless**（`clusterIP: None`）：不做负载均衡，直接返回后端 Pod IP 列表——供有状态应用按名字逐个访问。

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

## 5 从「单 Pod」到「一群 Pod」

现在你有了两个基本概念：Pod 管「一个实例」，Service 管「发现一群实例」。但「谁保证始终有 3 个实例、挂了自动补」还不属于这两者——那是**控制器**的职责。调度器把 Pod 放到节点上，Service 让它们被找到，控制器保证数量——三者合起来，才是编排的完整画面。

## 6 小结

- **Pod** 是最小调度单元：一组容器共享网络命名空间与卷，同生共死，IP 与名字**易变**。
- Pod 状态机：`Pending → Running → Succeeded/Failed/Unknown`；容器可重启，Pod 由控制器重建。
- **Service** 提供**稳定名字 + clusterIP**，按标签选中 Pod，经 EndpointSlice 维护**动态**后端。
- 服务发现链路：**DNS 名字 → clusterIP → 当前存活的 Pod**，三层映射，前两层稳定、最后一层实时。
- 边车模式让「辅助进程」与主进程共享网络与卷。

在下一节，我们回答「谁保证副本数量、谁做滚动升级」——进入 **控制器与声明式 API**。

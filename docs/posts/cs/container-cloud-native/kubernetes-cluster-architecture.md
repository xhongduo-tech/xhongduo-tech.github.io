---
title: Kubernetes 集群架构与控制平面
date: 2026-08-07
---

# Kubernetes 集群架构与控制平面

<div class="epigraph">
<p>一个编排系统，就是一台持续把「我想要的状态」翻译成「现在该做的事」的机器。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.3 ｜ 2026-08-07</p>
</div>

## 为什么先看集群骨架

前两节我们让单个容器能跑起来，但生产环境要面对的是「上百个容器、几十台机器、随时有机器挂掉」。把容器交给一个**编排系统**统一调度、恢复、升级，就进入了 Kubernetes。它首先是一个**架构问题**：谁掌握状态、谁做决策、谁在节点上执行——这四种角色必须分清楚，之后的 Pod、控制器、网络才有落点。

## 1 控制平面：集群的大脑

Kubernetes 集群分两半：**控制平面（control plane）** 与 **工作节点（worker nodes）**。控制平面由四个组件构成，它们共同回答「集群应该处于什么状态」：

- **API Server（kube-apiserver）**：集群唯一的**前门**。所有对集群的读写都经过它：先做认证（你是谁）、授权（你能干什么）、准入（这份请求是否合法），再落地。它是唯一会直接读写 etcd 的组件。<span class="marginnote">「唯一入口」是 API Server 的设计哲学：校验、审计、访问控制都集中在这一个点。任何组件——kubectl、kubelet、控制器——都不许绕过它去摸 etcd，这保证了集群状态变更全程可审计。</span>
- **etcd**：分布式**键值存储**，集群状态的唯一事实来源（single source of truth）。所有对象（Pod、Service、ConfigMap……）的持久化状态都存在这里，靠 **Raft 共识**保证多副本一致。
- **Scheduler（kube-scheduler）**：负责为新 Pod **挑选节点**。先按硬约束过滤（predicates：资源够不够、节点是否可调度、亲和性），再按软偏好打分（priorities：分散放置还是装箱）。只做「决定」，不负责执行。
- **Controller Manager（kube-controller-manager）**：运行着一批**控制器**——Deployment、ReplicaSet、Namespace、Endpoint 等各有一个控制器。它们是声明式模型的执行者：盯住实际状态，往期望状态拉。

四个组件分工放在一张表里：

| 组件 | 角色 | 一句话 |
| --- | --- | --- |
| kube-apiserver | 唯一入口 | 认证、授权、准入、读写 etcd |
| etcd | 事实来源 | Raft 共识下的键值存储 |
| kube-scheduler | 决策者 | 过滤 + 打分，挑选节点 |
| kube-controller-manager | 执行者 | 跑控制器，往期望状态拉 |

四个组件里，API Server、Scheduler、Controller Manager 都是**无状态**的——它们不自己保存数据，状态全在 etcd。因此控制平面的高可用通常这样搭：**etcd 用 3 或 5 副本做主从共识，其余三个组件各跑 2 个实例做负载均衡**。无状态组件想加几份就加几份，有状态的 etcd 才需要精打细算副本数——「把有状态的集合收窄到最小」，是分布式系统里反复出现的架构原则。

## 2 数据平面：工作节点

工作节点上的组件负责「把决策变成现实」：

- **kubelet**：每台节点上的**驻守代理**，唯一持续与 API Server 对话的节点组件。它向 API Server 汇报本节点状态，通过 **CRI** 调用容器运行时（containerd）启停容器，执行存活/就绪探针。
- **kube-proxy**：实现 **Service 的虚拟 IP**——把发往 Service 的流量转发到真正的 Pod。现代实现基于 iptables 或 IPVS（见《网络模型》）。
- **容器运行时**：`containerd` / `CRI-O`，上一节的分层运行时栈在这里落地。

**核心概念：控制平面与数据平面**：控制平面**做决策但不碰流量**，数据平面**转发流量但不做决策**。<span class="marginnote">「控制/数据平面」是计算机网络里的老概念（路由器、SDN 都有）——控制平面负责「拓扑、路由怎么算」，数据平面负责「每个包怎么转发」。Kubernetes 把这套分工搬进了编排系统，你会在网络模型与 Ingress 课里反复见到这对词。</span>

![Kubernetes 集群架构：控制平面与工作节点](/images/container-cloud-native/kubernetes-cluster-architecture-1.svg)

## 3 声明式模型：用户只说要什么

Kubernetes 最大的范式转变，是用户不再下命令（imperative：「把 Pod 删掉」「把副本数改成 3」），而是**提交期望状态（desired state）**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: web
          image: nginx:1.27
```

用户只写了「我要 3 个带 `app: web` 标签的 nginx Pod」。至于哪台机器跑、容器挂了怎么拉起、镜像怎么拉——全部由系统代劳。**把「怎么做」交给控制器，是声明式模型的核心。**

## 4 公式解析：控制回环

声明式模型背后是一个永不停止的**控制回环（reconcile loop）**，可以用一个简洁的式子刻画：

$$
\Delta(t) = \text{Desired} - \text{Observed}(t), \qquad \lim_{t \to \infty} \Delta(t) = 0
$$

- $\text{Desired}$：对象 `spec` 里用户声明的期望状态（常量）。
- $\text{Observed}(t)$：控制器从 API Server watch 到的当前实际状态（随时间变化）。
- $\Delta(t)$：期望与实际的**差距（误差）**。
- 控制器的职责：不断执行「补差动作」$\text{act}(\Delta)$，让差距收敛到 0。

三步拆解：

- **第一步，差距驱动动作**：观察到有 2 个 Pod，期望 3 个，$\Delta = 1$，控制器就创建一个新 Pod；观察不到差，就什么都不做。
- **第二步，动作与观察解耦**：控制器只负责「提交动作」，然后重新观察。这给了系统**自愈**能力——任何意外（机器宕机、Pod 被杀、容器崩溃）都表现为 Observed 偏离 Desired，回环会自动把它拉回来。
- **第三步，收敛而非瞬达**：$\Delta \to 0$ 是渐进的。升级时的滚动更新、扩容时的逐步创建，都只是把「一次大动作」拆成多个小动作，让回环有节拍地收敛——这正是《控制器与声明式 API》里滚动更新的数学直觉。

实现上还有一个细节：控制器不是「每次变化都立刻执行动作」，而是对**变更事件做去重与限速**——短时间内的大量变更会被合并成一次调和，失败的动作按指数退避重试。这避免了「一堆事件同时到达时，控制器手忙脚乱地各打一发」。**回环的节奏感，一半来自用户声明的目标，一半来自控制器自己踩的刹车**。

## 5 辨析｜易错点：etcd 不是「运行时数据库」

**最常见的误解**：以为 etcd 里存了集群的一切运行数据。真相是：

- etcd 只存**声明式状态**（对象的 `spec` 与 `status`）与集群元数据，**不存**容器日志、指标、对象存储数据。
- 应用数据走持久化存储（卷），日志走可观测性通道，指标走 Prometheus——它们**不进 etcd**。
- **另一个易错点**：`kubectl get` 看到的 Pod 状态来自 etcd 中 `status` 字段，而这个字段由 kubelet 持续汇报更新；但「Pod 进程此刻还活着吗」的实时性问题，靠的是探针与健康检查，不是 etcd。
- 生产集群里 etcd 必须独立、多副本、快照备份——它是「丢了就丢了一切」的组件。

**补充｜延伸：Raft 怎么保证一致**。etcd 靠 3 或 5 个副本组成 Raft 组：写入必须先获得**多数派**（过半）副本确认才提交，读也走多数派（或加租约）。所以即使某个 etcd 副本失联，只要多数还在，集群照常工作；反之，副本数不达标（如 3 副本只剩 1），etcd 会拒绝写入以保护一致性。**「奇数副本 + 多数派」是分布式存储的标准姿势**，你会在《分布式系统》与数据库课程里再见到它。

## 6 小结

- 集群分**控制平面**（API Server / etcd / Scheduler / Controller Manager）与**数据平面**（kubelet / kube-proxy / 运行时）。
- **API Server 是唯一入口**，负责认证、授权、准入，且是唯一读写 etcd 的组件。
- **声明式模型**：用户提交期望状态，控制器持续**调和**实际状态。
- 控制回环 $\Delta = \text{Desired} - \text{Observed} \to 0$ 是自愈与滚动更新的底层机制。
- etcd 只存声明式状态，不是运行时数据库；Raft 多数派保证其一致性。
- 控制平面高可用：无状态组件多实例负载均衡，etcd 用 3/5 副本做共识。

在下一节，我们把「期望状态」翻译成 API 世界的语法——进入 **Kubernetes 对象模型**。

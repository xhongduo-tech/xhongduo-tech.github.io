---
title: Pod/Service/Deployment 编排
date: 2026-08-07
---

# Pod/Service/Deployment 编排

<div class="epigraph">
<p>声明期望，剩下的交给集群。</p>
<footer>—— Kubernetes 使用者的共同体验</footer>
</div>

<div class="article-byline">
<p>第三级 · 云计算 ｜ Kubernetes 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从三大对象编排开始

上一篇立起了 K8s 的骨架，这一篇把三大核心对象——**Pod、Service、Deployment**——放进一个完整的「部署应用」流程里跑一遍。你会看到它们如何各司其职、如何配合完成一次「无中断的版本升级」：这是 K8s 最日常也最体现设计思想的工作流。读完这一篇，你会对「声明式编排」有切身体感，而不再只是概念。

一个预习心态：这一篇的主题是「**三个对象的一次合奏**」——Pod 是乐手、Deployment 是乐队的编制表（要几个人）、Service 是演奏给观众的门票渠道。单独认识每个对象不难，难的是看懂它们「各自独立又协同完成一次发布」的配合——这正是 K8s 编排的精髓。

## 1 一次部署的完整生命周期

部署一个「跑 3 个副本的 nginx 服务」，最小需要的声明是两个对象：一个 **Deployment**（描述要多少 Pod、什么镜像、怎么更新），一个 **Service**（描述如何把流量路由到这些 Pod）。流程如下：

1. **创建 Deployment**：用户 <code>kubectl apply -f</code> 提交声明，API server 写入 etcd。
2. **控制器介入**：Deployment 控制器看到「期望 3 副本、实际 0」，创建 3 个 Pod 对象；ReplicaSet 确保 Pod 数量恒为 3。
3. **调度器分配**：scheduler 为每个 Pod 挑一台资源充足的节点，kubelet 拉镜像、起容器。
4. **Service 暴露**：Service 控制器创建稳定入口，kube-proxy 下发转发规则，流量被路由到这三个 Pod。

整个过程中用户**只做了第 1 步**——其余全部由控制回路自动收敛。这就是声明式与「一步步手动敲命令」的本质区别。<span class="marginnote">注意第 2 步里出现了 ReplicaSet：Deployment 并不直接管 Pod，而是管 ReplicaSet，ReplicaSet 再管 Pod 数量。这个「中间层」让滚动更新成为可能——每轮升级用新 ReplicaSet 逐步替换旧 ReplicaSet，新旧交替过程天然可控。这也是 K8s 设计里典型的<strong>分层解耦</strong>。</span>

## 2 Deployment：副本、滚动更新与回滚

**Deployment** 的声明里最关键的两个字段是 <code>spec.replicas</code>`（期望副本数）与 <code>spec.template</code>`（Pod 模板）。它的核心能力是**滚动更新（rolling update）**：

修改镜像版本 → Deployment 创建新 ReplicaSet → 逐个**增量替换**旧 Pod（默认先起新 Pod，等健康就绪再杀旧 Pod）→ 全程服务不中断。
新版本出问题 → <code>kubectl rollout undo</code> **一键回滚**到上一个版本，Pod 反向替换。
日常扩容：<code>kubectl scale --replicas=5</code>，控制器立刻补足副本。

**辨析｜易错点：** Deployment 只保证「副本数量」这个期望，它**不关心流量**。更新时新旧 Pod 会短暂共存，但谁对外提供流量、流量怎么切，是 Service 的事。很多人刚接触时把「Deployment 更新」误当成「流量已切换」，实际上两者是不同控制面管的——这正是「关注点分离」在编排层的体现。

## 3 Service：把流量送进 Pod

**Service** 提供一个**稳定的虚拟 IP（ClusterIP）与 DNS 名**，把对它的访问负载均衡到一组 Pod 上。它通过**标签选择器（label selector）**决定把流量转发给哪些 Pod——只要 Pod 的标签匹配，无论 Pod 被重建到哪台节点、IP 变成什么，Service 都能找到它们。

Service 的几种类型各有用途：

| 类型 | 作用 | 典型场景 |
| --- | --- | --- |
| ClusterIP | 集群内虚拟 IP，仅集群内可访问 | 微服务间内部调用 |
| NodePort | 在每台节点上开一个固定端口 | 简单对外暴露 |
| LoadBalancer | 对接云负载均衡器，分配公网 IP | 对外提供服务 |

## 4 核心要点对比：三大对象的职责边界

| 对象 | 管什么 | 不管什么 | 一句话 |
| --- | --- | --- | --- |
| Pod | 容器的运行单元 | 数量、入口 | 「住在哪」 |
| Deployment | 副本数量与版本 | 流量路由 | 「有多少、什么版本」 |
| Service | 流量路由与稳定入口 | 副本管理 | 「怎么找到」 |

把这张表读熟，K8s 的「编排三件套」就不再是三个名词，而是一套分工明确的小组织：**Deployment 决定跑什么跑几个，Pod 是跑起来的实例，Service 是把用户请求安全送到实例的向导**。<span class="marginnote">一个实用的进阶提示：Pod 有生命周期（Pending → Running → Succeeded/Failed），Deployment 靠「就绪探针（readiness probe）」判断 Pod 是否健康、决定是否把流量切给它——探针是 K8s 自愈与滚动更新正确性的关键，值得在实战中重点练习。</span>

## 5 探针与自愈：让系统自己照顾自己

编排不只是「部署」，更是「持续守护」。守护靠两类**探针（probe）**：

**就绪探针（readiness probe）**：问「这个 Pod 准备好接收流量了吗？」——不通过，Service 就不把流量分给它；通过后恢复转发。用于**滚动更新**：新 Pod 未就绪，流量不切换。
**存活探针（liveness probe）**：问「这个 Pod 还活着吗？」——探测失败，kubelet 按策略**重启容器**。用于**自愈**：进程卡死、死锁时，自动杀掉重启。

配合**重启策略（restartPolicy）**，K8s 的自愈闭环就完整了：Pod 里的容器崩溃 → 存活探针探测失败 → kubelet 重启容器 → 流量重新就绪 → 用户无感。整个过程**无需人工介入**。<span class="marginnote">探针是「<strong>把人类的健康检查自动化</strong>」：运维过去要盯着监控、判断「这台要不要重启」，现在 K8s 用探针自动判断。但探针设计有讲究——就绪探针要探「真的能干活」（如能连数据库），而不是探「进程在跑」，否则会出现《负载均衡》里提到的「显示健康、实际超时」。</span>

**自愈的边界**：K8s 能自动处理的是「**实例级故障**」（容器崩溃、节点宕机导致 Pod 被杀），靠控制器重建 Pod；它不自动处理「**应用级故障**」（代码 bug 导致所有副本一起错）——那需要回滚版本，靠的是《CI/CD 与 DevOps》里的发布机制。**自愈管「活得起来」，不保证「业务是对的」**。

**辨析｜易错点：** 探针不是越多越好。探针太频繁会消耗资源、误杀慢启动应用；探针太宽松则失去意义。一个经典折中：给慢启动的容器配 <code>startupProbe</code>（启动宽限期），让探针等到应用真正就绪再开始探测——否则启动中的应用会被「误诊死亡」连环重启。

## 6 配置与密钥：ConfigMap 与 Secret

12 要素（《云原生概念与 12 要素》）说「配置进环境变量」，K8s 把它落地成两个专门对象：

**ConfigMap**：存放**非敏感配置**——URL、开关、日志级别。把配置从镜像里「抽出来」，同一镜像配不同 ConfigMap 就能跑不同环境。
**Secret**：存放**敏感信息**——数据库密码、API 密钥、证书。内容以 base64 存储，并可加密。

**用法**：Deployment 通过「环境变量」或「挂载文件」把 ConfigMap/Secret 注入容器。**镜像保持纯净，配置从外部注入**——这就是「一份镜像，处处可跑」的具体实现。

| 对象 | 存什么 | 是否敏感 | 注入方式 |
| --- | --- | --- | --- |
| ConfigMap | URL、开关、日志级别 | 否 | 环境变量 / 挂载 |
| Secret | 密码、密钥、证书 | 是 | 环境变量 / 挂载 |

**辨析｜易错点：** Secret 只是「比 ConfigMap 稍安全」，**不是加密保险箱**——默认 base64 只是编码、不是加密，任何能读取 Secret 的人都能解开。真正的密钥安全要配合**加密（KMS）与访问控制（RBAC）**，并且**不要把 Secret 提交进 git**（历史上无数密钥泄露事故源于此）。Secret 的治理，是《云安全与合规》「密钥管理」在集群内的延伸。<span class="marginnote">一个反直觉但重要的点：把配置做成「对象」而不是「文件」的意义在于——<strong>配置变更可以触发滚动更新</strong>。改 ConfigMap → Deployment 自动滚动重启 Pod 应用新配置，无需手动重建。配置与代码同等对待、可版本化、可回滚——这正是「配置即代码」的 K8s 表达。</span>

## 7 滚动更新的两个旋钮：maxUnavailable 与 maxSurge

滚动更新「不中断」的细节，藏在 Deployment 更新策略的两个参数里——它们决定了「新旧交替的节奏」。

- **maxUnavailable**：更新过程中**最多允许有多少旧副本不可用**。默认 25%——更新 4 个副本时，最多 1 个先停，其余继续服务。
- **maxSurge**：更新过程中**最多允许超出期望副本数的量**。默认 25%——可以先多起 1 个新副本，等它就绪再停旧副本。

两个参数共同控制「先起新、后停旧」的节奏：<code>maxSurge</code>` 决定「多快能上新」，<code>maxUnavailable</code>` 决定「最多允许缺多少」：

| 参数 | 控制什么 | 调大效果 | 调小效果 |
| --- | --- | --- | --- |
| maxSurge | 额外起的副本 | 更新快 | 更保守 |
| maxUnavailable | 允许缺的副本 | 停得更快 | 服务更稳 |

**辨析｜易错点：** 追求「零中断」不等于把两个参数都设成 0——<code>maxUnavailable=0</code>` 确实保证「旧副本一个不少」，但若 <code>maxSurge</code>` 也小，更新会非常慢（一个个来）。**「不中断」与「更新快」之间的平衡，正是这两个旋钮的用途**——理解它们，你就能读懂任何一次滚动更新的行为。

## 8 小结

- 部署流程：**提交声明 → 控制器建 ReplicaSet → 调度器分配 → kubelet 起容器 → Service 路由**。
- **Deployment** 管副本与滚动更新，中间经 ReplicaSet 间接管 Pod。
- 滚动更新 = 新 ReplicaSet 增量替换旧 ReplicaSet，**服务不中断、可一键回滚**。
- **Service** 用标签选择器路由流量，类型有 ClusterIP / NodePort / LoadBalancer。
- 职责划分：Deployment 管「多少与版本」、Pod 管「运行」、Service 管「找到」。

- 本节的心智模型：**Deployment 管「多少与版本」、Pod 管「运行」、Service 管「找到」**——三者协作，构成一次「无中断部署」的最小闭环。

在下一篇，我们深入 K8s 的另外两块拼图——**K8s 网络与存储**：Pod 之间怎么互通、容器数据如何持久化。你刚学到的「Pod 随时可换」正是网络与存储要回答的起点——**因为 Pod 会被不断重建，所以必须有一套机制保证「新的 Pod 依然能通、数据依然在」**。

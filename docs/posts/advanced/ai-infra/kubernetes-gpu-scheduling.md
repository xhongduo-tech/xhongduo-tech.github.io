---
title: Kubernetes 在 AI 训练中的角色：GPU 调度与 Volcano/ gang scheduling
date: 2026-08-07
---

# Kubernetes 在 AI 训练中的角色：GPU 调度与 Volcano/ gang scheduling

<div class="epigraph">
<p>调度的本质，是决定谁在什么时候用哪块算力。</p>
<footer>—— 马尔文 · 康威（Melvin Conway，调度与并发研究者）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ Kubernetes 文档与 Volcano 项目 · 集群调度篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Kubernetes 开始

网络拓扑解决「怎么连」，接着要解决「怎么调度」——几千块 GPU 上同时跑着几十个训练任务，谁来分配、怎么避免互相挤占？**Kubernetes（K8s）** 是云原生世界的调度底座，但 K8s 原生调度是为「无状态 Web 服务」设计的，直接拿来调度训练任务会遇到两个硬伤：**GPU 资源如何暴露**、以及**训练任务的「全员到齐才能开工」**。

本篇讲 K8s 在 AI 训练里的两个核心问题：GPU 作为资源的调度方式，以及 gang scheduling（组调度）为什么是训练的刚需。

## 1 K8s 调度的基本模型：Pod 与资源请求

K8s 把工作负载抽象成 **Pod**（一组容器的调度单元），Pod 声明要多少 CPU、内存，调度器把它放到满足资源要求的节点上。

对 GPU 训练，第一件事是**让 K8s 认识 GPU**。标准做法：

- **Device Plugin**：NVIDIA/k8s-device-plugin 把 GPU 作为「扩展资源」暴露给 K8s（如 `nvidia.com/gpu`）。
- **Pod 申请**：容器声明 `nvidia.com/gpu: 4`，调度器就会把它放到有 4 块空闲 GPU 的节点。

但问题来了：K8s 的 GPU 调度只保证「数量」，不保证「拓扑」。TP=8 的任务拿到 8 块 GPU，可能分布在 8 个不同节点——**NVLink 优势全丢**。这就引出了节点亲和性与拓扑约束。<span class="marginnote">原生 K8s 的 GPU 调度在训练场景的第一个坑：它不感知 NVLink 域。需要<strong>节点亲和性（`nodeSelector`/`nodeAffinity`）</strong> / <strong>拓扑约束（`topologySpreadConstraints`）</strong> 把任务钉在同一个节点（或同一组 NVLink 相连的 GPU）上，否则 TP 通信走网卡，性能崩。这也是「K8s + AI」需要专门组件的原因。</span>

## 2 训练任务的特殊需求：gang scheduling

训练任务与 Web 服务最大的不同：**多卡任务必须「同时就位」才能启动**。

一个 TP=8 的训练任务要 8 个 Pod 同时运行，形成分布式训练组。
若只调度到 7 个，第 8 个卡住，训练无法开始，那 7 个 Pod 就**白白占着资源等死**。
更糟：多个任务互相抢资源，可能造成**死锁**——任务 A 等 B 释放、B 等 A 释放。

**组调度（gang scheduling）** 的语义：**「要么全部 Pod 同时被调度，要么一个都不调度」**。这是训练任务调度与普通负载调度的分水岭。<span class="marginnote">「全部就位才启动」在 K8s 原生里没有，因为 K8s 是为「可独立运行的服务」设计的——每个 Pod 都能自给自足。训练任务的 Pod 是「协作群体」，单独跑没有意义。这个差异是 AI 调度器（Volcano 等）存在的根本理由。</span>

## 3 Volcano：组调度的实现

**Volcano** 是 CNCF 的批处理调度器，专为 AI/大数据设计。它的核心机制：

**Queue + PodGroup**：把一组相关 Pod（一个训练任务）声明为一个 **PodGroup**，配一个 Queue（队列）。
**Gang Scheduling**：调度器检查一个 PodGroup 是否「能全部放下」——能则一次全调度，不能则一个不调。
**抢占与优先级**：高优先级任务可抢占低优先级任务已占的资源。
**Task 拓扑约束**：感知 GPU 拓扑，尽量把同组 Pod 放到 NVLink 域内。

**PodGroup** 的语义就是「gang」：**`minMember`** 字段指定最小成员数，不足则不调度——**把「全有或全无」写进调度器**。<span class="marginnote">Volcano 还解决「K8s 原生调度器的性能问题」：K8s 默认调度器在大规模（数千 Pod）下调度吞吐不够，Volcano 用更高效的算法和批量调度，支撑大规模 AI 训练集群的「一次调度几百个 Pod」。</span>

## 4 其它关键组件：Kueue、MPI Operator、节点池

围绕 K8s 的训练调度生态还有几个重要角色：

**Kueue**：CNCF 的「作业排队」项目，管「任务谁先谁后」——与 Volcano 的「怎么放」互补。Kueue 提供配额（quota）、队列、抢占策略。
**MPI Operator / Kubeflow Training Operator**：把「一个分布式训练任务」抽象成一个 K8s 自定义资源（如 `MPIJob`、`PyTorchJob`），自动拉起 worker、launcher 等 Pod 并协调 rendezvous。
**节点池（Node Pools）**：把集群分成「TP 专用节点池」「通用节点池」等，用节点亲和性把任务导向合适的硬件组合。<span class="marginnote">生态分工一句话：Training Operator 负责「把训练任务翻译成 K8s 对象」，Volcano/Kueue 负责「排队与组调度」，Device Plugin 负责「暴露 GPU」——四件套拼起来，K8s 才能当训练集群的调度底座。</span>

## 5 公式解析：组调度的资源利用

设集群总 GPU 数 $C$，任务 $i$ 需要 $g_i$ 块 GPU、运行 $T_i$ 时间。组调度的「全部就位才启动」保证每块 GPU 只要被分配就是「被使用」（不空等）。

**无 gang（部分调度）** 时的资源浪费：任务只分到 $g'_i < g_i$ 块 GPU 就启动了，但这 $g'_i$ 块 GPU 因任务无法运行而空转，直到补足或超时。浪费期望：

$$\text{Waste}_{\text{partial}} = \sum_{\text{partial jobs}} g'_i \cdot t_{\text{wait}}$$

**有 gang**：调度器要么给满 $g_i$、要么不给，浪费为 0（不给就不占资源）。<span class="marginnote">Gang 的代价是「等待时间」：高优先级的 gang 任务若卡在「等最后一块 GPU」，会让整组干等。所以工业调度器都会加「超时 + 降级」：组调度等太久就降级为「部分调度先跑」，或者让调度器优先凑小任务。组调度不是免费——它是「宁等勿占」与「宁占勿等」的取舍。</span>

## 6 辨析｜易错点：K8s 调度的常见误区

**辨析｜易错点：**
- **「K8s 原生就能调度 GPU 训练」不完整**：原生只管「数量」，不管「拓扑」与「组」，需要 Device Plugin + 调度器扩展。
- **「gang scheduling = 排队」是混淆**：排队是「谁先谁后」（Kueue），组调度是「一群 Pod 必须同时就位」（Volcano）——两个不同概念。
- **「GPU 资源只按整数分配」局限**：MIG（多实例 GPU）可切分 GPU，但要专门的 device plugin 支持。
- **别忽略「节点亲和性」**：TP 任务必须钉在同一节点/NVLink 域，否则调度对了也白搭。
- **「调度越满越好」不成立**：过度调度导致 gang 死锁（任务互相等），调度器要有抢占与超时机制。

## 7 小结

- **K8s 的角色**：训练集群的调度底座，Device Plugin 暴露 GPU 为扩展资源。
- **训练与 Web 服务的差异**：训练 Pod 是协作群体，必须「全员就位」才能开工。
- **Gang scheduling**：全有或全无的组调度，杜绝「部分资源空等」与调度死锁。
- **Volcano**：PodGroup + 队列 + 抢占 + 拓扑感知，AI 批处理调度的核心实现。
- **生态四件套**：Training Operator（翻译任务）、Volcano/Kueue（排队与组调度）、Device Plugin（暴露 GPU）。

## 8 进阶与延伸

**动手配一个 Volcano 的 PodGroup**：给一个 8 卡训练任务写一个 `PodGroup` + `Queue` 的 YAML，设置 `minMember`——观察调度器「要么全调度、要么不调度」的行为，再试试把 `minMember` 调小看区别。

**几个值得进一步挖的方向**：

- **GPU 拓扑的节点亲和**：TP=8 的任务怎么保证 8 块 GPU 落在同一节点/NVLink 域？`nodeSelector` + Device Plugin 的拓扑信息怎么配合？
- **MIG 与整卡**：MIG 切分的 GPU 在 K8s 里怎么暴露？`nvidia.com/gpu` 的计数粒度与 MIG 实例的映射——多租户小模型部署的性价比工具。
- **Volcano vs Kueue 的分工**：Volcano 管「怎么放」（组调度），Kueue 管「谁先谁后」（排队）——两者共存时，PodGroup 与 Workload 的对接怎么配置？

**自测题**：为什么「gang scheduling」能防止「资源死锁」？如果你能说清「两个任务互相等对方释放资源」，就理解了「全有或全无」调度的必要性。

## 9 动手实践清单

- 给一个 8 卡任务写 `PodGroup` + `Queue` 的 YAML，设 `minMember`。
- 观察 `minMember` 从 8 降到 4 时调度行为的变化。
- 用 `nodeSelector` 把 TP=8 的任务钉在同一节点。
- 试 MIG 切分，观察 `nvidia.com/gpu` 的计数粒度。
- 对比 Volcano（组调度）与 Kueue（排队）的职责边界。
- 用 Device Plugin 验证「GPU 作为扩展资源」的暴露方式。
- 模拟「两个任务互相等资源」的死锁，验证 gang 的防死锁。
- 观察「minMember 不足」时 Pod 是否全部等待。
- 用 nodeSelector 把 TP 任务钉在 NVLink 域。
- 对比 Volcano 与原生 K8s 的大规模调度吞吐。
- 验证 Device Plugin 的 GPU 计数与 MIG 实例。
- 画「K8s + AI」四件套（Operator/Volcano/Kueue/Device Plugin）架构图。
- 观察「gang 等待」时资源是否被无效占用。
- 验证「节点亲和」对 TP 通信性能的影响。
- 观察「gang scheduling」在部分资源不足时的等待行为。

在下一节，我们对比另一条调度路线——**Slurm 作业调度**：HPC 世界的经典方案如何管理 GPU 集群。

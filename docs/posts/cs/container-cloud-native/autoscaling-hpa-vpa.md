---
title: 弹性伸缩 HPA/VPA
date: 2026-08-11
---

# 弹性伸缩 HPA/VPA

<div class="epigraph">
<p>弹性不是「够用就好」，而是「刚好够用」，且这个「刚好」是自动算出来的。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么要把「调副本数」交给机器

上一节的控制器保证了「数量恒定」；但生产流量有峰谷——白天十万 QPS、夜里两万。如果副本数永远按峰值配，就是白烧钱；按谷值配，高峰期就崩。**弹性伸缩**让副本数与负载自动匹配。它也是「从极限到大模型」主线里「让系统随需求自适应」的第一课：系统不再是被静态配置钉死的机器，而是一个会呼吸的活物。

## 1 三个维度的弹性

Kubernetes 的弹性按「缩什么」分三种：

- **HPA（Horizontal Pod Autoscaler，水平伸缩）**：改**副本数**——多开/少开 Pod。最常用。
- **VPA（Vertical Pod Autoscaler，垂直伸缩）**：改**单个 Pod 的资源请求**——大/小 Pod。适合「不好水平拆」的负载。
- **Cluster Autoscaler / Karpenter**：改**节点数**——集群容量按需增减（通常由云厂商配合）。

三者配合的典型链路：HPA 先加 Pod → 节点不够 → Cluster Autoscaler 加节点 → 新 Pod 调度上去。<span class="marginnote">HPA 管「应用层弹性」，Cluster Autoscaler 管「基础设施层弹性」——就像水位上涨先多开抽水机，再决定要不要增建电厂。两层错开，既保时延又控成本。</span>

## 2 HPA 的决策回路

HPA 是一个控制器，它周期性地（默认每 15 秒）拉取指标，算出一个「应该有多少副本」，再更新 Deployment 的 `spec.replicas`。Deployment 控制器看到新的期望值，就按上一节的滚动更新方式补齐/裁剪。

核心公式（来自官方算法）：

$$
\text{desiredReplicas} = \left\lceil \text{currentReplicas} \times \frac{\text{currentMetricValue}}{\text{desiredMetricValue}} \right\rceil
$$

当同时观测多个指标时，取计算结果中的**最大值**（哪个指标「更饿」就按谁扩）。

## 3 公式解析：HPA 的目标副本数

我们把上面的公式一步步拆开：

$$
n_{\text{desired}} = \left\lceil n_{\text{cur}} \cdot \frac{m_{\text{cur}}}{m_{\text{target}}} \right\rceil
$$

- $n_{\text{cur}}$：当前副本数。
- $m_{\text{cur}}$：当前观测到的指标均值（如所有 Pod 的 CPU 平均利用率）。
- $m_{\text{target}}$：用户设的目标值（`targetAverageUtilization`，如 50%）。
- $\lceil \cdot \rceil$：向上取整——保证「略多」而不是「略少」。

三步拆解：

- **第一步，看比值**：$\frac{m_{\text{cur}}}{m_{\text{target}}}$ 是「当前饥饿度」。比值 = 1 表示刚刚好，副本数不变；比值 > 1 表示超载，要扩容；< 1 表示闲置，可缩容。
- **第二步，按比例推算副本**：当前 10 个 Pod、CPU 用到 80%，目标 50%，那么 $10 \times \frac{0.8}{0.5} = 16$ 个 Pod——线性外推，10 个能消化 5 份工作，现在有 8 份，就要 16 个。
- **第三步，两个防抖机制**：**冷却（cooldown）**——HPA 的 `--horizontal-pod-autoscaler-sync-period` 与缩容等待，防止指标抖动造成「疯狂加减副本」（thrashing）；**伸缩边界**——`minReplicas`/`maxReplicas` 硬性兜底，避免缩到 0 或扩到失控。还要注意：**用 CPU 均值的 HPA，对瞬时尖峰反应慢**（均值平滑了尖峰）；对秒级突发敏感的应用，应考虑基于自定义指标（如队列长度、QPS）的伸缩。<span class="marginnote">「疯狂加减副本」（thrashing）是自动伸缩的头号病：指标在目标值上下抖动，HPA 每次都取整加一、过会儿又减一，集群在扩容与缩容之间反复横跳。冷却期与边界就是给这个回路装上的阻尼器。</span>

**辨析｜易错点：HPA 与 VPA 不能对同一指标同时生效**。VPA 会改 `requests`，HPA 算副本数时用的是相对 `requests` 的利用率——两套机制同时改基线会导致互相打架（VPA 把 Pod 改大 → 利用率下降 → HPA 缩容）。实践规则：**HPA 管副本，VPA 管单 Pod 大小，二者对同一 workload 不要同时配置**。

## 4 VPA：把「请求多少资源」也自动化

VPA 解决的是另一个浪费：开发者给 Pod 设的 `requests` 常常拍脑袋（设大怕 OOM，设小怕被杀）。VPA 的三个组件：

- **Recommender**：根据历史用量，算出每个容器的建议 `requests`（下界、目标、上界）。
- **Updater**：对需要调整的 Pod，驱逐它让重建时带上新值（Pod 会被重启——所以 **VPA 与「不能重启」的负载不兼容**）。
- **Admission Plugin**：在 Pod 创建时把建议值写进 `requests`。

VPA 的价值不是精确，而是**把「资源预算」变成一个持续校准的过程**——这和 HPA 一样，都是让系统自我调节的一部分。

## 5 弹性伸缩的工程边界

- **指标质量决定一切**：HPA 只对**可靠、及时**的指标有意义。基于陈旧或高噪声指标做自动决策，比不自动更糟。
- **下限兜底**：`minReplicas` 至少要能扛住单 Pod 故障时的重分配（一般建议 ≥2，跨可用区分布）。
- **弹性是「设计出来的」**：无状态应用天然可水平伸缩；有状态应用（数据库）扩容要重分片，通常交给专门的算子——**伸缩之前，先问负载能不能被安全地拆分**。

## 6 小结

- 弹性分三层：**HPA（副本数）**、**VPA（单 Pod 大小）**、**Cluster Autoscaler（节点数）**。
- HPA 核心公式：$n_{\text{desired}} = \lceil n_{\text{cur}} \cdot m_{\text{cur}}/m_{\text{target}} \rceil$，多指标取最大值。
- 防抖靠**冷却**与**min/max 边界**；均值指标对尖峰反应慢。
- **VPA 会重启 Pod**，且与 HPA 不能同打一个指标。
- 弹性上限由「负载能否安全拆分」决定：无状态优先，有状态需专门机制。

在下一节，我们把目光从「集群内部」移到「微服务之间的通信治理」——进入 **服务网格 Istio**。

---
title: 服务网格 Istio
date: 2026-08-07
---

# 服务网格 Istio

<div class="epigraph">
<p>当微服务之间互相调用成了家常便饭，治理这种调用的基础设施就值得被专门造出来。</p>
<footer>—— 意译自 Lee Atchison（《Architecting for the Cloud》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Atchison §8 ｜ 2026-08-07</p>
</div>

## 为什么微服务还要「一张网」

前面几节解决了「容器怎么跑、怎么发现、怎么伸缩」，但微服务落地后出现了一个新痛点：**服务之间的通信质量**。调用超时了怎么办？要不要重试？要不要熔断？哪些服务可以互相访问？mTLS 加密谁来做？全链路追踪的 trace 由谁传播？如果把这些逻辑写进每个服务，代码就废了——**服务网格把「服务间通信的治理」从业务代码里抽出来，下沉为基础设施**。

## 1 数据平面与控制平面

**核心概念：服务网格（service mesh）**：一个专门负责服务间通信的**基础设施层**，分两半：

- **数据平面**：一组**边车代理（sidecar proxy）**。Istio 用的是 **Envoy**。每个服务实例旁边都跑着一个 Envoy，所有进出该实例的流量都**先经过 Envoy**——透明代理，业务代码完全无感知。
- **控制平面**：Istio 的 **istiod**，负责向所有 Envoy 下发配置（路由规则、流量策略、证书），并收集遥测数据。

<span class="marginnote">环境里没有服务网格的专图，但《集群架构》一课那张「控制器 + 执行者」的图能帮你定位：istiod 相当于控制平面，每个 Pod 里的 Envoy 相当于数据平面——控制平面只管下规则，真正的包转发都在边车。</span>

流量路径：`客户端 → 客户端旁的 Envoy → 网络 → 服务端旁的 Envoy → 服务端`。每跳流量都有两个 Envoy 守护，于是**双向**都可观察、可控制。<span class="marginnote">「代理 + 控制器」的结构我们并不陌生——《集群架构》一课的 kubelet + API Server、HPA 的指标回路都是同一套骨架：一个持续盯梢的控制器 + 一组忠实的执行者。Istio 只是把这个模式搬到了「服务间流量」上。</span>

每个 Envoy 与 istiod 的通信方式值得点一句：Envoy 通过 **xDS 协议**（Discovery 服务）订阅配置——路由、监听器、集群、端点各自是一类资源，istiod 一有变更就增量推送。所以「几秒内全局生效」不是靠轮询，而是**服务端主动推送**；这保证了成千上万个边车能同时收到同一条规则。

## 2 流量管理：把路由规则写进配置

Istio 让你用声明式配置控制流量，核心资源：

**VirtualService**：定义「去往某个 host 的流量怎么路由」——按权重分流、按 header 条件路由（金丝雀、A/B 测试）。
- **DestinationRule**：定义「后端实例的负载均衡策略」与**熔断**（`maxConnections`、`maxPendingRequests`、`outlierDetection`）。
- **Gateway**：网格边缘的入口/出口代理（通常配 Ingress Gateway）。
- **ServiceEntry**：把网格外的服务（外部 API、遗留系统）注册进网格，让规则也能作用到它们。

| 资源 | 管什么 | 例子 |
| --- | --- | --- |
| VirtualService | 流量怎么路由 | 按权重/header 分流 |
| DestinationRule | 后端策略 | 负载均衡、熔断、TLS |
| Gateway | 边缘出入口 | Ingress/Egress Gateway |
| ServiceEntry | 外部服务入网格 | 注册外部 API |

用权重把 10% 流量切到新版本，只需：

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts: [reviews]
  http:
    - route:
        - destination: { host: reviews, subset: v2 }
          weight: 10
        - destination: { host: reviews, subset: v1 }
          weight: 90
```

无需改任何服务代码，流量就按比例分流了。**控制平面把这份配置推给所有相关 Envoy，几秒内全局生效。**

这份 YAML 值得逐行看一遍：`hosts` 说明「管的是去往 `reviews` 的流量」；`route` 下列出两个 `destination`（带 `subset` 标签的 v2 与 v1）；`weight` 决定比例。`subset` 由配套的 `DestinationRule` 定义（把带 `version: v2` 标签的 Pod 归为一组）——**VirtualService 描述「怎么分」，DestinationRule 描述「分给谁」**，两者常成对出现。金丝雀发布就是不断调整 `weight`，从 10 → 50 → 100。

**流量治理除了路由，还有三件套：超时（timeout）、重试（retry）、熔断（circuit breaker）**。超时防止「上游慢了，下游跟着全卡死」；重试消化瞬时故障；熔断（DestinationRule 的 `outlierDetection`）在连续失败达到阈值后把实例**踢出负载池**一段时间。三者配合，是微服务「自愈」的第一道防线——它们都是配置，不写业务代码。把四个核心资源串起来记忆：VirtualService 讲「路由」，DestinationRule 讲「策略」，Gateway 讲「边界在哪」，ServiceEntry 讲「网外的服务算不算」。

## 3 安全：网格内的零信任

服务网格顺带解决了微服务安全的三大问题：

- **mTLS（双向 TLS）**：Istio 自动为服务间通信签发证书并做双向认证——不仅加密，还**验证对端身份**。默认配置下，网格内流量可全部加密。
- **AuthorizationPolicy**：基于身份（`source.principal`、命名空间、标签）的细粒度授权，替代「靠 IP 判断」的脆弱做法。
- **证书轮换**：istiod 作为 CA 自动下发并轮换证书，运维免手抖。<span class="marginnote">零信任（zero trust）思想在网格里落到实操：<strong>不信任网络位置，只信任身份</strong>。这在分布式系统里尤其重要——IP 可以伪装、网络可以被攻破，但双向 TLS 的身份不行。</span>

mTLS 的工作流值得展开：istiod 作为 CA 给每个 Envoy 签发「服务身份证书」（SPIFFE 格式，如 `spiffe://cluster.local/ns/default/sa/reviews`）→ 服务端 Envoy 校验客户端证书，客户端 Envoy 也校验服务端证书 → 两端各验证一次，随后对流量加密。**「调用方是谁」由证书里的 ServiceAccount 决定**——这比「这个 IP 是不是数据库」可信得多。

## 4 可观测性：网格就是一张「流量传感器网」

因为所有流量都经过 Envoy，网格天生拥有全量遥测：

- **指标（metrics）**：每个服务对的 RPS、时延、错误率（对应 RED 方法论，见《可观测性》一课）。
- **追踪（traces）**：Envoy 自动传播与生成 trace，配合 Jaeger/Tempo 可以查看一次跨 5 个服务的调用链。
- **访问日志**：每个请求的详细记录。

**服务网格让「你不知道服务之间发生了什么」变成「你比服务自己还清楚」**——这是它最被低估的价值。

网格观测与《可观测性》一课对接的方式很具体：Envoy 生成的是标准 Prometheus 指标（`istio_request_duration_seconds` 等），trace 以 OTLP 格式发给 Collector，访问日志也可路由到 Loki。**网格把「每对服务」的通信数据变成了一等公民**——你在面板上看到的「service_a → service_b 的 P95」，网格和可观测性各出一半力。

## 5 辨析｜易错点：服务网格 ≠ Service

**最容易混的一对概念**：

- **Service**（Kubernetes）解决的是**服务发现**——给一组 Pod 一个稳定名字，做 L4 负载均衡（IP/端口层）。
- **服务网格 / Istio**解决的是**流量治理**——在 Service 之上做 L7（HTTP 层）的路由、重试、熔断、mTLS、观测。

二者是**叠加**关系：Istio 的 Envoy 在 Service 提供的入口后面做更细的控制。同理，**服务网格 ≠ Ingress**：Ingress 管「网格外的流量进集群」，服务网格管「集群内的服务间流量」（详见《Ingress 与流量管理》）。

**另一个易错点**：服务网格不是银弹。它带来可观测与可控，也带来复杂度与性能开销（每跳多两个 Envoy、延迟与内存成本）。**在服务数量不多、治理需求不重时，引入网格往往是过度设计**——先让 Kubernetes Service + Ingress 干好它们的活。

量化一下代价：每个 Envoy 边车要额外吃几十 MB 内存、几十毫秒级别的 CPU，每跳时延增加个位数毫秒级。一个 200 个 Pod 的集群，网格的内存开销可能到 GB 量级。**网格的价值在「服务足够多、治理足够复杂」时才回本**——这是一个清晰的成本-收益权衡，而不是默认选项。

**再辨析一组：服务网格与 API 网关**。API 网关（Ingress Gateway、Kong、Traefik）在**边缘**做「对外流量」的聚合、鉴权、限流；服务网格在**内部**做「服务到服务」的路由与治理。两者常被混谈，但边界清晰：网关管「南北向」（进集群），网格管「东西向」（集群内）。Istio 的 Ingress Gateway 正是「网关形态、网格内核」——把南北向流量也交给网格治理。

## 6 小结

- 服务网格 = **数据平面（Envoy 边车）** + **控制平面（istiod）**，把服务间通信治理下沉为基础设施。
- **流量管理**：VirtualService（路由/分流）、DestinationRule（负载均衡/熔断）、Gateway（边缘入口）、ServiceEntry（外部服务）。
- **安全**：mTLS 双向认证 + AuthorizationPolicy 基于身份的授权，落地零信任。
- **可观测**：网格内每跳流量自带指标、追踪、日志。
- 服务网格解决 **L7 流量治理**，叠加在 Service（L4 服务发现）之上，与 Ingress 分工不同；复杂度有代价，按需采用。
- 流量治理三件套——**超时、重试、熔断**——都是配置、不写业务代码，是微服务自愈的第一道防线。
- Envoy 经 **xDS 协议**向 istiod 订阅配置，变更即推，这是「几秒内全局生效」的机制。
- **南北向（Ingress 网关）与东西向（网格）**分工不同；API 网关管边界、网格管内部，Istio Ingress Gateway 把两者合一。
- mTLS + SPIFFE 证书让「调用方是谁」由 ServiceAccount 决定，落地零信任；选型边界：服务足够多、治理足够复杂时网格才回本。

在下一节，我们跳出 Kubernetes 回到应用的视角，看「什么样的应用才配得上云」——进入 **云原生 12 因素应用**。

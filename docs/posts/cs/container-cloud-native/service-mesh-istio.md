---
title: 服务网格 Istio
date: 2026-08-11
---

# 服务网格 Istio

<div class="epigraph">
<p>当微服务之间互相调用成了家常便饭，治理这种调用的基础设施就值得被专门造出来。</p>
<footer>—— 意译自 Lee Atchison（《Architecting for the Cloud》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么微服务还要「一张网」

前面几节解决了「容器怎么跑、怎么发现、怎么伸缩」，但微服务落地后出现了一个新痛点：**服务之间的通信质量**。调用超时了怎么办？要不要重试？要不要熔断？哪些服务可以互相访问？mTLS 加密谁来做？全链路追踪的 trace 由谁传播？如果把这些逻辑写进每个服务，代码就废了——**服务网格把「服务间通信的治理」从业务代码里抽出来，下沉为基础设施**。

## 1 数据平面与控制平面

**核心概念：服务网格（service mesh）**：一个专门负责服务间通信的**基础设施层**，分两半：

- **数据平面**：一组**边车代理（sidecar proxy）**。Istio 用的是 **Envoy**。每个服务实例旁边都跑着一个 Envoy，所有进出该实例的流量都**先经过 Envoy**——透明代理，业务代码完全无感知。
- **控制平面**：Istio 的 **istiod**，负责向所有 Envoy 下发配置（路由规则、流量策略、证书），并收集遥测数据。

<span class="marginnote">环境里没有服务网格的专图，但《集群架构》一课那张「控制器 + 执行者」的图能帮你定位：istiod 相当于控制平面，每个 Pod 里的 Envoy 相当于数据平面——控制平面只管下规则，真正的包转发都在边车。</span>

流量路径：`客户端 → 客户端旁的 Envoy → 网络 → 服务端旁的 Envoy → 服务端`。每跳流量都有两个 Envoy 守护，于是**双向**都可观察、可控制。<span class="marginnote">「代理 + 控制器」的结构我们并不陌生——《集群架构》一课的 kubelet + API Server、HPA 的指标回路都是同一套骨架：一个持续盯梢的控制器 + 一组忠实的执行者。Istio 只是把这个模式搬到了「服务间流量」上。</span>

## 2 流量管理：把路由规则写进配置

Istio 让你用声明式配置控制流量，核心资源：

**VirtualService**：定义「去往某个 host 的流量怎么路由」——按权重分流、按 header 条件路由（金丝雀、A/B 测试）。
- **DestinationRule**：定义「后端实例的负载均衡策略」与**熔断**（`maxConnections`、`maxPendingRequests`、`outlierDetection`）。
- **Gateway**：网格边缘的入口/出口代理（通常配 Ingress Gateway）。
- **ServiceEntry**：把网格外的服务（外部 API、遗留系统）注册进网格，让规则也能作用到它们。

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

## 3 安全：网格内的零信任

服务网格顺带解决了微服务安全的三大问题：

- **mTLS（双向 TLS）**：Istio 自动为服务间通信签发证书并做双向认证——不仅加密，还**验证对端身份**。默认配置下，网格内流量可全部加密。
- **AuthorizationPolicy**：基于身份（`source.principal`、命名空间、标签）的细粒度授权，替代「靠 IP 判断」的脆弱做法。
- **证书轮换**：istiod 作为 CA 自动下发并轮换证书，运维免手抖。<span class="marginnote">零信任（zero trust）思想在网格里落到实操：<strong>不信任网络位置，只信任身份</strong>。这在分布式系统里尤其重要——IP 可以伪装、网络可以被攻破，但双向 TLS 的身份不行。</span>

## 4 可观测性：网格就是一张「流量传感器网」

因为所有流量都经过 Envoy，网格天生拥有全量遥测：

- **指标（metrics）**：每个服务对的 RPS、时延、错误率（对应 RED 方法论，见《可观测性》一课）。
- **追踪（traces）**：Envoy 自动传播与生成 trace，配合 Jaeger/Tempo 可以查看一次跨 5 个服务的调用链。
- **访问日志**：每个请求的详细记录。

**服务网格让「你不知道服务之间发生了什么」变成「你比服务自己还清楚」**——这是它最被低估的价值。

## 5 辨析｜易错点：服务网格 ≠ Service

**最容易混的一对概念**：

- **Service**（Kubernetes）解决的是**服务发现**——给一组 Pod 一个稳定名字，做 L4 负载均衡（IP/端口层）。
- **服务网格 / Istio**解决的是**流量治理**——在 Service 之上做 L7（HTTP 层）的路由、重试、熔断、mTLS、观测。

二者是**叠加**关系：Istio 的 Envoy 在 Service 提供的入口后面做更细的控制。同理，**服务网格 ≠ Ingress**：Ingress 管「网格外的流量进集群」，服务网格管「集群内的服务间流量」（详见《Ingress 与流量管理》）。

**另一个易错点**：服务网格不是银弹。它带来可观测与可控，也带来复杂度与性能开销（每跳多两个 Envoy、延迟与内存成本）。**在服务数量不多、治理需求不重时，引入网格往往是过度设计**——先让 Kubernetes Service + Ingress 干好它们的活。

## 6 小结

- 服务网格 = **数据平面（Envoy 边车）** + **控制平面（istiod）**，把服务间通信治理下沉为基础设施。
- **流量管理**：VirtualService（路由/分流）、DestinationRule（负载均衡/熔断）、Gateway（边缘入口）、ServiceEntry（外部服务）。
- **安全**：mTLS 双向认证 + AuthorizationPolicy 基于身份的授权，落地零信任。
- **可观测**：网格内每跳流量自带指标、追踪、日志。
- 服务网格解决 **L7 流量治理**，叠加在 Service（L4 服务发现）之上，与 Ingress 分工不同；复杂度有代价，按需采用。

在下一节，我们跳出 Kubernetes 回到应用的视角，看「什么样的应用才配得上云」——进入 **云原生 12 因素应用**。

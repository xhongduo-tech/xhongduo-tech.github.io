---
title: Ingress 与流量管理/负载均衡
date: 2026-08-11
---

# Ingress 与流量管理/负载均衡

<div class="epigraph">
<p>集群内部有了一套完整的寻路能力，但「外面的流量怎么进来」是另一扇门——而门要能上锁、能分流。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 容器与云原生 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么集群「外部」是另一个世界

前两节的网络模型解决了「Pod 之间怎么通信」，Service 解决了「集群内服务发现」。但互联网用户不在集群里——**外部流量怎么进集群**是又一个独立问题。而且它比集群内流量更讲究：外部流量要按域名、按 URL 路径分发到不同服务，要 TLS 终结、要限流、要金丝雀。这一层在 Kubernetes 里由 **Ingress** 及配套的负载均衡机制负责。

## 1 从 Service 到集群外的三条路

Service 的三种对外暴露方式，是「外部流量进集群」的第一层：

| 类型 | 暴露方式 | 特点 |
| --- | --- | --- |
| `ClusterIP` | 集群内虚拟 IP | 集群外不可直接访问 |
| `NodePort` | 每节点开一个端口 `nodeIP:nodePort` | 简单，但端口管理混乱，直接暴露节点 |
| `LoadBalancer` | 云厂商分配 LB，指向各节点 NodePort | 一个 Service 一个 LB，贵且慢（每个 LB 有成本与预热时间） |

NodePort 与 LoadBalancer 的转发链：`外部 → LB → nodeIP:nodePort → Service clusterIP → Pod`。**每层都是地址改写，链路越长，可治理的点越多**——而治理这些点，正是 Ingress 诞生的原因。

## 2 Ingress：七层流量的统一入口

**核心概念：Ingress**：一种对象，描述「**外部 HTTP(S) 流量按什么规则路由到集群内服务**」。它只描述「规则」，真正执行的是 **Ingress Controller**（如 ingress-nginx、Traefik、HAProxy Ingress）。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: main
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /orders
            pathType: Prefix
            backend:
              service:
                name: orders
                port:
                  number: 8080
          - path: /users
            pathType: Prefix
            backend:
              service:
                name: users
                port:
                  number: 8080
```

这份 Ingress 说：「`api.example.com` 下，`/orders` 开头去 orders 服务，`/users` 开头去 users 服务。」**Ingress 让「一个入口，多套规则」成为可能**——这是它相对「一个 Service 一个 LoadBalancer」的核心优势。<span class="marginnote">Ingress Controller 本身也是跑在集群里的 Pod（通常是 Deployment + Service），它 watch 集群里的 Ingress 对象，把规则实时翻译成 nginx/HAProxy 的配置并重载——又是「控制器 + 执行者」的那套骨架。</span>

**按 host 路由**：多个域名共享一个入口（虚拟主机）。
- **按 path 路由**：一个域名按路径分到不同后端。
- **TLS 终结**：证书配置在 Ingress 上，解密在入口完成，后端跑明文 HTTP。
- **IngressClass**：一个集群可以有多个 Ingress Controller（nginx + 私有网关），Ingress 用 `ingressClassName` 指定用哪个。

**辨析｜易错点**：Ingress 是**控制面的声明**，Ingress Controller 是**数据面的执行者**——这和 Service 与 kube-proxy、Service Mesh 与 Envoy 是完全同构的两件套。**只写了 Ingress 而没有装 Ingress Controller，流量不会路由**——这是新手最常见的「写了没反应」事故。

## 3 负载均衡：从四层到七层

「负载均衡」这个词横跨两个层次，必须分清：

- **L4（传输层）**：按 IP + 端口分发，不读报文内容（Service 的 kube-proxy 就是 L4）。快、通用，但无法按 URL/Header 分流。
- **L7（应用层）**：读 HTTP 报文，按路径、Header、Cookie 路由（Ingress Controller、服务网格）。精细，但更贵。<span class="marginnote">一个直觉记忆法：L4 只看「信封」——把包裹送到哪栋楼；L7 拆开信封看「内容」——根据内容决定放到哪个房间。能读到内容，才能做内容级分流。</span>

**公式解析：会话保持与粘性**：L7 负载均衡常需要把「同一个用户」的请求固定到同一后端（session 黏连）：

$$
\text{sticky}(r) = \text{hash}(\text{session}(r)) \bmod N
$$

- $\text{session}(r)$：请求 $r$ 的会话标识（Cookie 或 Header）。
- $N$：后端实例数。
- $\text{sticky}(r)$：该请求应去往的后端编号。

三步拆解：

- **第一步，哈希让「同会话 → 同后端」**：相同 Cookie 哈希到相同编号——用户中途换后端时 session 就丢了，黏连解决这个问题。
- **第二步，代价是「亲和性破坏了均衡」**：热用户会压向单点，负载不再均匀。所以现代架构倾向**无状态 + 任意后端可服务**（12 因素第 6 条），让黏连变成可选项而非必需。
- **第三步，为什么这里提公式**：负载均衡的本质是一个**哈希/权重映射**——L4 的 IP 哈希、L7 的 URL 哈希、Ingress 的路径匹配，全都是在做「把请求映射到后端」。看懂这个映射，就懂了所有负载均衡器。

## 4 流量治理：金丝雀、灰度与限流

Ingress 的规则之上，还能叠加更细的流量治理（尤其在服务网格里，见《服务网格 Istio》）：

- **金丝雀发布**：新版本先收 5% 流量，验证后再逐步放大。
- **A/B 测试**：按 Header/参数分流到不同版本。
- **限流与熔断**：入口限流保护后端，异常时快速失败而非拖垮。

这些能力在不同层都能做：Ingress Controller 能做基础版，服务网格做最强版，应用网关（如 APISIX、Kong）是独立的中间态。**选择哪一层做流量治理，是架构权衡**：越靠下越高效、越靠上越精细。

## 5 辨析｜易错点：一张「流量地图」防止搞混

把本专题讲过的三层流量控制放在一张表里：

| 层 | 管什么 | 典型实现 |
| --- | --- | --- |
| 服务发现/分发（L4） | 集群内 Pod 之间 | Service + kube-proxy |
| 集群入口（L7） | 外部流量进集群 | Ingress + Controller |
| 服务间治理（L7） | 集群内服务调用 | 服务网格（Istio） |

- **不是互斥而是叠加**：Ingress 管「门」，Service 管「楼内通道」，Service Mesh 管「楼内每一通电话的治理」。
- **别用 Ingress 管服务间流量**（那是服务网格的活）；**别用 Service 当外部入口**（除非是简单 NodePort/LB）。

## 6 小结

- 外部进集群的三层路径：`LoadBalancer/NodePort → Service → Pod`；每层一次地址改写。
- **Ingress** 描述七层路由规则（按 host/path），**Ingress Controller** 执行；只写 Ingress 不装 Controller 不生效。
- **L4 与 L7 负载均衡**是两回事：四层看 IP/端口，七层读 HTTP 报文。
- 负载均衡的本质是「请求 → 后端」的哈希/权重映射；黏连与均衡是此消彼长的权衡。
- 三层流量控制各有分工：**Ingress 管入口，Service 管分发，Service Mesh 管治理**。

在下一节，我们给整个系统加上「约束与访问控制」——进入 **安全（RBAC/NetworkPolicy/Secrets）**。

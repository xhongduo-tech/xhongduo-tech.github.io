## 核心结论

Kubernetes 里，`Service` 和 `Ingress` 解决的是两类不同问题。

`Service` 是四层抽象。四层的意思是“按 IP 和端口转发”，不理解 HTTP 里的域名、路径、Cookie。它给一组会变化的 Pod 提供一个稳定入口，通常表现为 `clusterIP:port`。只要前端连这个虚拟地址，就不用关心 Pod 扩缩容、重建、漂移。其核心链路可以写成：

$$
clusterIP:port + selector \rightarrow kube\text{-}proxy \rightarrow EndpointSlice/Endpoints \rightarrow Pod
$$

`Ingress` 是七层抽象。七层的意思是“理解 HTTP/HTTPS 请求语义”，能按 `Host`、`Path`、TLS 证书来路由。它本身只是 API 对象，真正干活的是 Ingress Controller，比如 NGINX Ingress、Traefik。其核心链路可以写成：

$$
Ingress.rules(host,path) + spec.tls(secret) \rightarrow Service \rightarrow Pod
$$

对新手可以这样理解：Service 像集群内部的固定 VIP，Ingress 像放在最前面的反向代理。前者解决“Pod 不稳定但入口要稳定”，后者解决“外部流量如何按域名、路径和证书进入系统”。

最常见的组合是：

`外部 HTTP 请求 -> Ingress -> Service -> Pod`

不是：

`外部请求直接进 Pod`

---

## 问题定义与边界

先把边界讲清楚，否则很容易把两个对象混着用。

| Component | 作用 | 可控范围 | 关键限制 |
| --- | --- | --- | --- |
| Service | 给 Pod 提供稳定访问入口与 L4 负载均衡 | 集群内访问为主，也可经 NodePort/LoadBalancer 暴露 | 不理解 Host/Path；默认不处理 TLS 终止 |
| Ingress | 给 HTTP/HTTPS 提供统一入口与 L7 路由 | 集群外到集群内的 Web 流量 | 只对 HTTP/HTTPS 生效；必须有 Ingress Controller |
| EndpointSlice | 记录某个 Service 当前后端地址集合 | 控制面到各节点 | 不是直接给业务访问的对象 |
| Secret(TLS) | 存证书和私钥 | 同命名空间内由控制器引用 | 证书更新是否生效取决于控制器是否正确监听/重载 |

一个玩具例子：

你有 3 个 `web` Pod，IP 分别是 `10.244.1.5`、`10.244.2.9`、`10.244.3.2`。这些 IP 会变，但你创建了 `web-svc` 后，集群给它分配了固定 `ClusterIP`，比如 `10.96.0.10:4200`。此后，其他服务只连 `10.96.0.10:4200`，无需知道后面 Pod 变成 2 个还是 5 个。

再往外一层，如果你希望用户访问 `https://foo.bar.com/foo`，那就不是 Service 单独能解决的事了，因为这里涉及：

- 域名 `foo.bar.com`
- 路径 `/foo`
- TLS 证书
- 可能还要按不同路径转发到不同后端

这些属于 Ingress 的职责。

一个真实工程例子：

`api` 命名空间中有 `api-svc`，对内端口是 `8080`。外部希望通过 `https://foo.bar.com/api` 访问它。典型做法是：

- `api-svc` 负责把流量稳定送到 `app=api` 的 Pod
- Ingress 负责把 `foo.bar.com/api` 转发给 `api-svc:8080`
- cert-manager 负责签发和续期 `foo.bar.com` 对应证书，写入 TLS Secret

这里的关键边界是：Service 解决“稳定后端寻址”，Ingress 解决“外部 Web 入口治理”。

---

## 核心机制与推导

先看 Service。

Pod 天生不稳定。Deployment 滚动发布时，旧 Pod 会删，新 Pod 会起，Pod IP 会变。如果客户端直接记 Pod IP，那么任一重建都可能让连接目标失效。于是 Kubernetes 引入一个稳定中间层：Service VIP。

Service 的工作不是“自己转发包”，而是让每个节点上的 `kube-proxy` 按 Service 和 EndpointSlice 状态去写内核转发表规则。官方文档里明确说明，`kube-proxy` 监听 Service 和 EndpointSlice，对流向 `clusterIP:port` 的流量做重定向。Linux 上常见模式是 `iptables`、`ipvs`、`nftables`。截至 Kubernetes 1.35，`nftables` 已稳定，`ipvs` 反而被标记为 deprecated，因此“Service 一定靠 IPVS”已经不是准确说法。

推导过程如下：

1. 你创建 Service，并写 `selector: app=web`
2. 控制面找到所有匹配标签的 Pod
3. 这些 Pod 被组织进 EndpointSlice
4. 节点上的 kube-proxy 观察到变化，更新转发规则
5. 访问 `clusterIP:port` 的 TCP/UDP 流量被转发到某个后端 Pod

因此 Service 的本质不是 DNS，也不是应用层代理，而是“稳定 VIP + 节点级转发表更新”。

再看 Ingress。

Ingress 本身不转发流量，它描述“应该怎么转发 HTTP/HTTPS 请求”。真正执行规则的是 Ingress Controller。它会把 Ingress 资源翻译成 NGINX、Traefik 或云负载均衡器的配置。

链路通常是：

`External HTTP -> Ingress Controller -> Service -> Pod`

当请求进入时，控制器先看两件事：

- `Host` 是否匹配，例如 `foo.bar.com`
- `Path` 是否匹配，例如 `/foo`

若命中规则，就把请求发给指定 Service。若还配置了 `spec.tls.secretName`，控制器会在入口处做 TLS 终止。TLS 终止的意思是“在入口把 HTTPS 解密成明文 HTTP，再转给后端”，所以很多集群里 Ingress 到 Service 之间是明文，除非你再额外做 mTLS 或后端 TLS。

这里可以把 Service 与 Ingress 的差异压缩成一个判断表：

| 判断问题 | Service | Ingress |
| --- | --- | --- |
| 后端 Pod 在变，入口要稳定 | 是核心能力 | 不是核心能力 |
| 按域名区分流量 | 不支持 | 支持 |
| 按 URL 路径区分流量 | 不支持 | 支持 |
| TLS 证书挂载与终止 | 不是主要职责 | 主要职责之一 |
| 面向 TCP/UDP | 是 | 通常否 |
| 面向 HTTP/HTTPS | 只能透传端口 | 是核心场景 |

所以常见误区是把 Ingress 理解成“更高级的 Service”。这不准确。它们不是上下替代关系，而是分层组合关系。

---

## 代码实现

下面给一个最小可工作的 Service + Ingress + cert-manager 示例。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-svc
  namespace: api
spec:
  selector:
    app: web
  sessionAffinity: None
  ports:
    - name: http
      protocol: TCP
      port: 4200
      targetPort: 4200
```

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  namespace: api
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - foo.bar.com
      secretName: foo-tls
  rules:
    - host: foo.bar.com
      http:
        paths:
          - path: /foo
            pathType: Prefix
            backend:
              service:
                name: web-svc
                port:
                  number: 4200
```

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: foo-cert
  namespace: api
spec:
  secretName: foo-tls
  dnsNames:
    - foo.bar.com
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
```

这个配置表达的意思是：

- `web-svc` 选中 `app=web` 的 Pod
- Ingress 把 `foo.bar.com/foo` 的请求发到 `web-svc:4200`
- cert-manager 负责把证书写入 `foo-tls`
- Ingress Controller 读取 `foo-tls` 做 TLS 终止

下面用一个可运行的 Python 玩具程序模拟“按规则匹配并转发到 Service”的过程：

```python
from dataclasses import dataclass

@dataclass
class Rule:
    host: str
    path_prefix: str
    service: str
    port: int

def route_request(host: str, path: str, rules: list[Rule]):
    for rule in rules:
        if host == rule.host and path.startswith(rule.path_prefix):
            return f"{rule.service}:{rule.port}"
    return "default-backend"

rules = [
    Rule(host="foo.bar.com", path_prefix="/foo", service="web-svc", port=4200),
    Rule(host="foo.bar.com", path_prefix="/api", service="api-svc", port=8080),
]

assert route_request("foo.bar.com", "/foo/list", rules) == "web-svc:4200"
assert route_request("foo.bar.com", "/api/users", rules) == "api-svc:8080"
assert route_request("bar.com", "/foo", rules) == "default-backend"
print("ok")
```

这段代码不是 Kubernetes 实现，只是帮助理解 Ingress Controller 的核心判断逻辑：先匹配主机名，再匹配路径，最后决定发给哪个 Service。

---

## 工程权衡与常见坑

第一类坑是会话保持。

Service 支持 `sessionAffinity: ClientIP`。白话说法是“同一个客户端 IP 尽量一直打到同一个 Pod”。它能缓解把登录态放内存时的抖动，但代价明显：

- 单个 NAT 出口后的大量用户会被当成同一个客户端 IP
- 流量容易倾斜到某个 Pod
- Pod 重建后，会话仍然丢
- 本质上只是“把状态问题往后拖”，没有真正解决状态外置问题

所以如果业务依赖登录态，优先考虑 Redis、数据库或签名 Token，而不是指望 `ClientIP` 粘性兜底。

第二类坑是跨命名空间引用。

经典 Ingress 的后端 Service 一般要求同命名空间理解最稳妥。更现代的 Gateway API 允许 Route 跨命名空间引用 Service，但必须由目标命名空间显式声明 `ReferenceGrant`，本质上是一个“双方同意”的授权握手。很多团队迁移到 Gateway API 时，最常见故障就是“路由写对了，但后端引用被拒绝”。

第三类坑是证书更新。

Ingress 只引用 Secret，不负责签发证书。证书轮转通常交给 cert-manager。cert-manager 默认会根据证书实际有效期计算续期时间；若证书是 90 天且未显式设置 `renewBefore`，通常会在走完约 $2/3$ 生命周期时触发，也就是大约到期前 30 天左右尝试续签。这里必须注意：Secret 更新后，是否立刻对外生效，取决于 Ingress Controller 是否正确监听并重载配置。主流控制器通常支持，但你不能把它当成必然行为，最好在测试环境验证一次真实换证链路。

| 坑 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| `sessionAffinity=ClientIP` | 单 Pod 过热 | NAT 或热点客户端导致粘性过强 | 状态外置，默认用 `None` |
| 证书续期后仍返回旧证书 | TLS 没及时更新 | 控制器未正确重载 Secret | 用 cert-manager 并验证控制器热更新能力 |
| 只创建 Ingress 不生效 | 外部始终 404 或无地址 | 没有安装 Ingress Controller | 先确认控制器和 `ingressClassName` |
| 误把 Service 当 HTTP 网关 | 无法按路径分流 | Service 只做 L4 | 需要 Host/Path/TLS 时使用 Ingress |
| 跨 namespace 引用失败 | 配置合法但不转发 | 缺少显式授权 | 在 Gateway API 中补 `ReferenceGrant` |

真实工程里最常见的稳定方案是：Service 保持无状态负载均衡，Ingress 负责 L7 规则，登录态和灰度状态落到外部系统，不把路由粘性当业务正确性的前提。

---

## 替代方案与适用边界

不是所有外部访问都必须走 Ingress。

如果你只有一个简单 Web 服务，没有多域名、没有路径分发、没有统一证书治理，`Service type=LoadBalancer` 往往就够了。云厂商会直接给你一个公网 IP 或负载均衡器，再配 ExternalDNS 解析域名，链路更短，故障点也更少。

如果你要暴露的不是 HTTP，而是 TCP、UDP、数据库协议、游戏协议，Ingress 也不适合。此时更常见的是：

- `NodePort`
- `LoadBalancer`
- 或直接用 Gateway API / 专用四层网关

Gateway API 是比 Ingress 更新的模型。它把入口、路由、授权拆得更清楚，比如 `Gateway`、`HTTPRoute`、`GRPCRoute`、`ReferenceGrant`。优点是表达能力更强，缺点是理解门槛更高，控制器兼容性也要单独核对。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Service `ClusterIP` | 纯集群内访问 | 简单、稳定 | 不能直接给公网用 |
| Service `LoadBalancer` | 单服务直接对外 | 链路短，维护简单 | 不擅长多域名、多路径汇聚 |
| Ingress | 多个 HTTP/HTTPS 服务统一入口 | 支持 Host/Path/TLS，适合 Web | 依赖 Controller，只覆盖 HTTP/HTTPS |
| Gateway API | 需要更细粒度流量治理 | 模型清晰、跨团队边界更强 | 更复杂，学习和落地成本更高 |
| Headless Service | 客户端自己做发现或 StatefulSet | 直接拿 Pod 地址 | 不提供 VIP 和标准负载均衡 |

可以把选型原则压缩成一句话：

- 只要“稳定内部入口”，用 Service。
- 只要“外部 Web 流量治理”，在 Service 前面加 Ingress。
- 只要“更强路由模型和跨命名空间授权”，考虑 Gateway API。

---

## 参考资料

| 来源 | 内容 | 用途 |
| --- | --- | --- |
| [Kubernetes 官方 Service 与 kube-proxy 文档](https://kubernetes.io/docs/reference/networking/virtual-ips/) | 说明 Service VIP、kube-proxy、iptables/IPVS/nftables、session affinity 的工作机制 | 支撑 Service 原理、代理模式、会话保持部分 |
| [Kubernetes 官方 Service 概念文档](https://kubernetes.io/docs/concepts/services-networking/service/) | 说明 Service 类型、EndpointSlice、Headless Service 等 | 支撑 Service 边界、替代方案部分 |
| [Kubernetes 官方 Ingress 概念文档](https://kubernetes.io/docs/concepts/services-networking/ingress/) | 说明 Ingress 暴露 HTTP/HTTPS、Host/Path 规则、TLS Secret、Ingress Controller 前提 | 支撑 Ingress 定义、TLS、实现部分 |
| [Gateway API GEP-709](https://gateway-api.sigs.k8s.io/geps/gep-709/) | 说明跨命名空间引用需要 `ReferenceGrant` 授权握手 | 支撑跨 namespace 边界与常见坑部分 |
| [cert-manager Certificate 文档](https://cert-manager.io/docs/usage/certificate/) | 说明 `Certificate` 续期机制、`renewBefore`、默认生命周期计算 | 支撑证书自动续期与工程例子部分 |
| [cert-manager FAQ](https://cert-manager.io/docs/faq/) | 补充默认 `duration` 与续期时间计算规则 | 支撑证书续期细节校验 |

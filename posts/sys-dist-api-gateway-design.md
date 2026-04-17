## 核心结论

API 网关是放在客户端和后端服务之间的统一入口。统一入口的意思是：客户端只访问一个域名或一组受控入口，不直接面对用户服务、订单服务、支付服务这些内部地址。它的核心价值不是“帮你多转发一次请求”，而是把认证、路由、限流、熔断、日志、监控、协议转换这些横切能力集中到边界层。

“横切能力”可以理解为很多服务都要做、但又不属于某个具体业务的公共能力。把它们集中到网关，前端接入会更简单，后端服务也能更专注于业务本身。

但 API 网关不是没有代价。它会引入一跳额外网络路径，也会形成明显的中心节点。如果只部署一个实例，它就是单点故障；如果把业务编排、字段拼装、复杂判断都塞进去，它又会变成新的巨石系统。正确做法是让网关保持“薄而强”：强在通用治理能力，薄在不承载核心业务规则。

可以把常见职责先压缩成一张表：

| 能力 | 白话解释 | 适合放在网关吗 |
| --- | --- | --- |
| 路由转发 | 决定请求该去哪个后端 | 是 |
| 认证鉴权 | 先检查你是谁、能不能访问 | 是 |
| 限流熔断 | 流量太大时先挡住，避免压垮后端 | 是 |
| 日志监控 | 统一记录谁在什么时候调了什么 API | 是 |
| 协议转换 | 比如 HTTP 转 gRPC，或统一响应格式 | 视复杂度而定 |
| 订单折扣计算 | 真正的业务规则 | 否 |
| 库存预占逻辑 | 业务状态变更 | 否 |
| 跨服务复杂编排 | 多服务协同业务流程 | 通常否 |

一个面向初学者最直观的理解方式是“前台门童”模型：Web 和移动端都只跟门口的网关说话，网关根据路径、身份和策略决定能不能进、该去哪个房间、是否要限速、是否要记账。

---

## 问题定义与边界

在单体应用里，客户端通常只面对一个服务地址，问题不明显。进入微服务架构后，问题会迅速暴露：

1. 服务数量增加，客户端要维护多个地址。
2. 不同服务可能有不同认证方式。
3. 日志、限流、追踪分散在各处，排障困难。
4. 某些服务对外暴露并不安全，应该只允许内网访问。
5. Web、移动端、小程序对接口形态的要求并不相同。

API 网关要解决的是“外部访问如何稳定、有序、可治理地进入系统”这个边界问题，而不是“业务逻辑在哪里写”这个内部问题。

所以边界要先划清：

| 网关负责 | 网关不负责 |
| --- | --- |
| 暴露统一入口 | 实现核心业务规则 |
| 校验令牌、API Key、JWT | 修改订单状态机 |
| 统一限流、熔断、超时 | 承担复杂业务编排 |
| 记录访问日志、指标、Trace | 持久化业务数据 |
| 路由到正确服务实例 | 替代服务层领域模型 |
| 基础响应转换与版本管理 | 变成“万能后端” |

“JWT”是 JSON Web Token，一种把用户身份信息打包进签名令牌里的常见认证格式。网关验证 JWT，后端就不用每个服务都重复做同样的身份校验。

### 玩具例子

假设只有两个后端服务：

- `/users/*` 走用户服务
- `/orders/*` 走订单服务

如果没有网关，前端需要记住：

- `https://user.example.com/profile`
- `https://order.example.com/list`

如果加上网关，前端只记住：

- `https://api.example.com/users/profile`
- `https://api.example.com/orders/list`

前端少知道了一层内部结构，后端以后迁移服务地址、扩容实例、替换协议，都可以尽量不影响客户端。

这个边界非常重要。网关的目标是降低耦合，而不是吸收所有逻辑。

---

## 核心机制与推导

API 网关有很多能力，但最适合拿来讲清原理的，是限流。限流是“控制单位时间内允许多少请求通过”的机制。最常见的实现之一是令牌桶。

“令牌桶”可以理解为一个会自动补充通行证的桶。每来一个请求，就消耗一个通行证；桶空了，请求就被拒绝或排队。

定义两个参数：

- $r$：令牌补充速率，单位通常是 tokens/s，也就是每秒补充多少个令牌
- $b$：桶容量，也就是最多能攒多少令牌

如果观察窗口是 $T$ 秒，那么理论上最多允许通过的请求数为：

$$
\text{max requests} = r \times T + b
$$

这里的 $b$ 表示“可突发能力”。系统平时空闲时，桶会积满；一旦突然来一波流量，先消耗积累的令牌，再回到稳定速率 $r$。

看一个最小数值例子：

- 持续速率 $r = 1000 \text{ req/s}$
- 桶容量 $b = 2000$
- 观察窗口 $T = 3s$

那么 3 秒内理论最多可放行：

$$
1000 \times 3 + 2000 = 5000
$$

这 5000 不是说“每秒都能打到 5000”，而是说在 3 秒窗口内，允许先吃掉积攒的 2000，再按每秒 1000 的速度持续补充。

用表格看更直观：

| 参数 | 含义 | 例子 |
| --- | --- | --- |
| $r$ | 持续处理能力对应的放行速率 | 1000 req/s |
| $b$ | 可接受的突发窗口 | 2000 |
| $T$ | 观察时间窗口 | 3 s |
| $r \times T + b$ | 窗口内理论可通过总量 | 5000 |

为什么 $b$ 不能无限大？因为桶太大，相当于允许客户端一次性把很大的脉冲流量砸向后端，保护作用会变弱。工程上常用经验是：$b$ 先设成 $0.5r$ 到 $1.0r$ 左右，再根据真实流量调优。

### 推导思路

限流不是拍脑袋设数字，而是从下游能力反推。假设订单服务稳定能处理 800 req/s，超过 1000 req/s 就明显变慢，那么网关给这个路由的持续速率就不能轻易配成 2000 req/s。否则网关不是保护后端，而是在稳定制造过载。

可以把关系写成近似式：

$$
r \le C_{downstream} \times \alpha
$$

其中：

- $C_{downstream}$ 是下游稳定处理能力
- $\alpha$ 是安全系数，通常小于 1

比如下游稳定能力 800 req/s，安全系数取 0.8，那么可先设：

$$
r = 800 \times 0.8 = 640
$$

这样给后端留出余量，便于处理网络抖动、慢查询、偶发尖峰。

### 真实工程例子

电商系统在大促时，移动端会短时间集中刷新“商品详情”和“库存状态”。如果没有网关，所有流量直接打到商品服务，热点商品很容易把缓存和数据库都拖慢。加上网关后，通常会做三层控制：

1. 在网关先做每用户、每 IP、每路由限流。
2. 对高价值路由设置更小的突发桶，避免瞬时冲击。
3. 对下游异常比例升高的路由启用熔断或快速失败。

“熔断”可以理解为：发现后端已经明显异常时，暂时不再继续把流量打过去，而是快速返回错误或降级结果，防止故障扩大。

这说明 API 网关不是只负责“找路”，它还负责“在入口处调节流量形状”。

---

## 代码实现

工程上一个可维护的 API 网关，通常遵循两个原则：

1. 配置驱动路由
2. 插件化处理横切能力

配置驱动的意思是“路由、认证、限流规则主要靠配置表达，而不是写死在代码里”。插件化的意思是“认证、限流、日志、熔断这些能力可以按顺序挂在请求链上”。

先看一个简化配置：

```yaml
routes:
  - path: /users/**
    target: http://user-service
    auth: jwt
    rate_limit:
      rate: 500
      burst: 800

  - path: /orders/**
    target: http://order-service
    auth: jwt
    rate_limit:
      rate: 200
      burst: 300
```

处理流程通常是：

`请求进入 -> 匹配路由 -> 认证 -> 限流 -> 记录日志/指标 -> 转发到下游 -> 返回响应`

下面用一个可运行的 Python 玩具实现演示“配置路由 + 令牌桶限流”的核心思路：

```python
from dataclasses import dataclass

@dataclass
class TokenBucket:
    rate: float
    burst: int
    tokens: float
    last_ts: float

    def allow(self, now: float, cost: int = 1) -> bool:
        elapsed = max(0.0, now - self.last_ts)
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_ts = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

@dataclass
class Route:
    prefix: str
    target: str
    auth: str
    bucket: TokenBucket

class Gateway:
    def __init__(self, routes):
        self.routes = routes

    def match(self, path: str):
        for route in self.routes:
            if path.startswith(route.prefix):
                return route
        return None

    def handle(self, path: str, token: str, now: float) -> str:
        route = self.match(path)
        if route is None:
            return "404"

        if route.auth == "jwt" and token != "valid-jwt":
            return "401"

        if not route.bucket.allow(now):
            return "429"

        return f"200 -> {route.target}"

user_route = Route(
    prefix="/users/",
    target="http://user-service",
    auth="jwt",
    bucket=TokenBucket(rate=2.0, burst=3, tokens=3, last_ts=0.0),
)

gw = Gateway([user_route])

assert gw.handle("/users/profile", "bad-token", 0.0) == "401"
assert gw.handle("/users/profile", "valid-jwt", 0.0) == "200 -> http://user-service"
assert gw.handle("/users/profile", "valid-jwt", 0.0) == "200 -> http://user-service"
assert gw.handle("/users/profile", "valid-jwt", 0.0) == "200 -> http://user-service"
assert gw.handle("/users/profile", "valid-jwt", 0.0) == "429"
assert gw.handle("/users/profile", "valid-jwt", 0.5) == "200 -> http://user-service"
```

这个例子故意没有实现真正的网络转发，只保留了网关最关键的入口控制逻辑。它说明三件事：

1. 认证失败，应尽早在边界拒绝。
2. 限流应在转发前执行，否则保护不了下游。
3. 路由规则与治理规则适合用配置挂接，而不是散落在业务代码里。

真实工程里，网关还需要支持：

- 配置热更新
- 多副本无状态部署
- 健康检查与摘流
- OpenTelemetry 追踪
- 超时、重试、熔断策略
- 灰度发布与版本路由

“无状态”是指单个网关实例不依赖本地内存保存关键业务状态。这样实例坏了可以直接替换，也更容易水平扩容。

---

## 工程权衡与常见坑

API 网关最常见的问题，不是“不会用”，而是“用过头”。

第一类坑是把业务写进网关。比如把“折扣计算”“优惠券组合校验”“订单合并规则”直接写成网关脚本。短期看省事，长期会出现三个后果：

- 网关变成最难测试的地方
- 业务规则和领域服务分裂
- 每次改流量治理都要碰业务逻辑，风险叠加

第二类坑是单实例部署。网关是统一入口，一台挂掉，所有 API 一起挂。正确做法通常是多副本、跨可用区、前挂负载均衡，并配健康探针自动摘除异常实例。

第三类坑是过度定制。很多团队一开始选了网关产品，后面大量写自定义插件，最后升级困难、迁移困难、性能不可预测。插件应服务于通用治理，不应把产品特性锁死在私有扩展里。

可以把典型风险压缩成一张对策表：

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 业务编排塞进网关 | 网关越来越大，变更越来越慢 | 业务留在后端服务，网关只做边界治理 |
| 单实例部署 | 一个实例故障导致全站 API 不可用 | 无状态多副本、负载均衡、跨 AZ |
| 配置变更靠人工 | 易出错、难回滚 | 配置中心、版本化、IaC |
| 限流值拍脑袋 | 要么误伤用户，要么保护不了后端 | 从下游容量反推，逐步调优 |
| 过度聚合响应 | 网关响应越来越重、延迟升高 | 只做必要聚合，复杂编排下沉 |
| 日志只记入口不记链路 | 出问题时找不到瓶颈 | 统一日志 + Trace ID 透传 |

“IaC”是 Infrastructure as Code，白话解释就是“用代码和配置文件管理基础设施”，而不是纯手工点控制台。对网关来说，IaC 很重要，因为路由、证书、限流、域名、灰度策略本质上都属于基础设施配置。

一个真实工程场景是：团队最初只部署了一台网关，平时压测也没出大问题；双十一预热时，一个插件内存泄漏把进程拖死，外部用户会看到所有接口都超时。后面改成 active-active 多副本、跨可用区、自动扩缩和健康探针后，即使单个实例异常，也只是局部摘流，不会让整个入口消失。

---

## 替代方案与适用边界

API 网关不是唯一方案，也不是所有场景都该重用同一种网关形态。

### 1. 单一 API 网关

最常见。适合外部流量入口统一治理，尤其是面向 Web、App、合作方开放接口时。优点是统一、直观、治理集中；限制是容易成为能力堆积点。

### 2. BFF

BFF 是 Backend For Frontend，白话解释是“按客户端类型拆分后端入口”。比如移动端一个网关，Web 端一个网关。这样移动端可以拿更小的 payload，Web 端拿更丰富的数据，不必在一个公共网关里互相妥协。

### 3. 服务网格

服务网格主要处理服务和服务之间的内部通信，也就是东西向流量，而 API 网关主要处理外部进入系统的南北向流量。服务网格更适合做内部 mTLS、重试、超时、熔断和可观测性，不直接替代对外 API 治理。

对比看更清楚：

| 方案 | 适用场景 | 优势 | 限制 |
| --- | --- | --- | --- |
| 单一 API 网关 | 统一外部入口 | 管理集中、接入简单 | 易堆积过多职责 |
| BFF 多入口 | Web、移动端诉求差异大 | 按终端定制接口更合理 | 网关数量增加，治理更复杂 |
| 服务网格 | 内部服务通信治理 | 对内可靠性和安全性强 | 不等于对外 API 产品层治理 |

选择边界可以简单记成一句话：

- 外部入口治理，优先看 API 网关。
- 不同客户端诉求差异明显，考虑 BFF。
- 内部服务通信可靠性、安全性、可观测性，考虑服务网格。

很多成熟系统最终会同时使用三者：外部用 API 网关，客户端差异大的地方加 BFF，内部东西向通信交给服务网格。这不是重复建设，而是边界分工。

---

## 参考资料

| 文献 | 关注点 | 适配章节 |
| --- | --- | --- |
| AWS API Gateway 文档: Throttle requests to your REST APIs | 官方说明 API Gateway 使用 token bucket、触发 429、rate 与 burst 的含义 | 核心机制与推导、工程权衡与常见坑 |
| AWS API Gateway Quotas | 官方配额与区域级限流说明 | 核心机制与推导 |
| System Overflow: Sizing and Tuning Token Bucket | `max = r × T + b` 公式与 `b` 的经验取值 | 核心机制与推导 |
| Zylos Research: API Gateway Patterns and Architecture | 网关职责、BFF、服务网格边界、避免单点与业务下沉 | 问题定义与边界、工程权衡、替代方案 |
| APIPark: Essential Guide / Main Concepts | 用统一入口解释 API 网关的角色，以及认证、路由、日志等典型能力 | 核心结论、问题定义与边界 |

- AWS API Gateway Throttling: https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html
- AWS API Gateway Quotas: https://docs.aws.amazon.com/apigateway/latest/developerguide/limits.html
- System Overflow Token Bucket: https://www.systemoverflow.com/learn/rate-limiting/token-bucket/sizing-and-tuning-choosing-r-and-b-with-real-numbers
- Zylos Research API Gateway Patterns: https://zylos.ai/research/2026-02-15-api-gateway-patterns
- APIPark Essential Guide: https://apipark.com/techblog/en/essential-guide-api-gateway-main-concepts/

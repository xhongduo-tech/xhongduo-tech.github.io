## 核心结论

RESTful API 和 gRPC 不是“二选一”的关系，而是两种面向不同边界的接口协议。RESTful API 是一种基于 HTTP 资源语义设计接口的方式，白话说，就是“用浏览器和各种客户端都容易调用的 Web 接口”；gRPC 是基于 HTTP/2 和 Protocol Buffers 的远程过程调用协议，白话说，就是“让服务像调用本地函数一样调用远端服务，并且类型更严格、传输更高效”。

对大模型平台或一般后端平台，最常见、也最稳妥的做法是：

1. 对外开放接口用 RESTful API，返回 JSON，兼容浏览器、移动端、第三方开发者和调试工具。
2. 对内服务间通信用 gRPC，获得更低延迟、更小报文、更清晰的接口约束。
3. API Gateway 作为统一入口，集中处理认证、限流、参数校验、缓存、错误翻译、监控和降级。

新人视角可以先这样理解：外部用户通过 HTTPS 发 REST 请求拿到 JSON；内部多个微服务之间通过 gRPC 传递推理请求、向量检索请求或流式结果；网关站在中间，负责“先拦住不合法请求，再把合法请求转成内部调用，并在后端失败时给出统一错误”。

| 维度 | RESTful API | gRPC |
|---|---|---|
| 底层协议 | 主要基于 HTTP/1.1，也可跑在 HTTP/2 | 基于 HTTP/2 |
| 常见数据格式 | JSON | Protocol Buffers |
| 可读性 | 高，抓包和调试直观 | 低，二进制不可直接读 |
| 类型约束 | 较弱，依赖文档或 schema | 强，接口定义先行 |
| 性能 | 较通用，序列化成本较高 | 通常更高效 |
| 流式能力 | 可做，但实现不统一 | 原生支持单向/双向流 |
| 客户端兼容性 | 非常广泛 | 浏览器直连受限，需额外方案 |
| 典型场景 | 对外开放 API、管理后台、第三方接入 | 内部微服务、高频低延迟调用 |
| 主要优点 | 生态成熟、上手简单、文档友好 | 高性能、强类型、IDL 驱动 |
| 主要缺点 | 报文大、约束弱、接口演进容易混乱 | 调试门槛更高、外部兼容性较差 |

---

## 问题定义与边界

这篇文章讨论的不是“哪种协议更高级”，而是“如何为大模型业务设计一套安全、可靠、可扩展、可监控的 API 与服务架构”。

边界要先划清：

1. 外部边界：面向浏览器、App、第三方平台、企业客户的公开接口。
2. 内部边界：网关到模型服务、向量检索服务、计费服务、审计服务等内部通信。
3. 控制边界：认证、鉴权、配额、限流、日志、告警、缓存、熔断、降级这些“控制能力”放在哪里。

如果边界不清，系统很快会变成两种混乱：

1. 外部接口泄露内部实现，客户端被迫理解内部服务细节。
2. 内部服务各自重复实现认证、限流、错误码，导致规则不一致。

这里需要明确几个基础约束。

HTTPS 是加密传输协议，白话说，就是“保证请求内容在传输途中不被随便看见或篡改”。OAuth2 是授权框架，白话说，就是“规定第三方如何拿到合法访问权限”。JWT 是一种令牌格式，白话说，就是“把用户身份和权限信息装进一个签名后的字符串里”。

因此，外部请求通常至少要满足：

| 约束项 | 作用 | 为什么必须有 |
|---|---|---|
| HTTPS | 保护传输链路 | 防止中间人窃听或篡改 |
| OAuth2 / API Key / JWT | 标识调用方身份 | 防止匿名滥用 |
| Schema 校验 | 验证字段、类型、长度、枚举值 | 防止脏数据和畸形请求 |
| 注入防护 | 限制 prompt、SQL、命令等危险输入 | 防止恶意 payload |
| 统一错误码 | 让客户端可编程处理失败 | 防止“失败了但不知道怎么处理” |
| 速率限制 | 控制单位时间请求量 | 防止突发流量拖垮集群 |
| 审计日志 | 记录谁在什么时间做了什么调用 | 便于排障和追责 |

“外部调用像一扇带锁的门”这个说法可以更精确一点：门锁是 HTTPS + 认证；门口安检是 schema 校验、长度限制、内容过滤；门卫记录是审计日志；限流闸机决定同一时间能进多少人。

简化流程图可以写成：

```text
客户端
  |
  | HTTPS + Token
  v
API Gateway
  |-- 认证/鉴权
  |-- schema 校验
  |-- 限流/配额
  |-- 统一日志与错误翻译
  v
外部 REST API
  |
  v
内部 gRPC 服务
  |-- 模型推理
  |-- 向量检索
  |-- 缓存
  |-- 计费/审计
```

这个边界设计有一个核心目的：外部可见接口尽量稳定，内部实现允许演进。客户端只关心“调用什么接口、传什么字段、拿到什么结果”，不应被迫知道你内部是单体、微服务，还是多个模型路由。

---

## 核心机制与推导

服务架构最终要回答两个问题：

1. 服务是否能扛住流量和故障。
2. 故障发生后，影响会不会被迅速放大。

可靠性里常用一个公式：

$$
Availability \approx \frac{MTBF}{MTBF + MTTR}
$$

MTBF 是 Mean Time Between Failures，平均无故障间隔，白话说，就是“两次故障之间平均能稳定运行多久”；MTTR 是 Mean Time To Repair，平均修复时间，白话说，就是“出故障后平均多久能恢复”。

这个公式可以直接推导理解。一个服务在一个长期周期里，总时间约等于：

$$
总时间 = 正常运行时间 + 故障恢复时间
$$

如果平均每轮正常运行时间是 $MTBF$，每轮故障恢复时间是 $MTTR$，那么可用时间占比近似就是：

$$
可用性 = \frac{正常运行时间}{正常运行时间 + 故障恢复时间}
= \frac{MTBF}{MTBF + MTTR}
$$

玩具例子：

假设一个推理服务平均 500 小时坏一次，坏了以后 5 小时恢复，那么：

$$
Availability \approx \frac{500}{500 + 5} \approx 99.01\%
$$

这说明问题不只是“少出故障”，也包括“出故障后要恢复得快”。很多系统可用性低，不是因为每天都坏，而是因为一旦坏了恢复太慢。

下面是一段可运行的 Python 代码，直接把这个概念算出来：

```python
def availability(mtbf_hours: float, mttr_hours: float) -> float:
    assert mtbf_hours > 0
    assert mttr_hours >= 0
    return mtbf_hours / (mtbf_hours + mttr_hours)

a = availability(500, 5)
assert round(a, 4) == 0.9901

year_hours = 365 * 24
downtime_hours = year_hours * (1 - a)
assert 86 < downtime_hours < 87

print(a, downtime_hours)
```

但公式只告诉你“结果”，不能告诉你“怎么做”。工程上真正起作用的是 API Gateway 的控制回路。

网关至少承担三类机制：

1. 入口治理：认证、鉴权、参数校验、限流。
2. 调用治理：超时、重试、熔断、负载均衡、缓存。
3. 结果治理：统一错误码、监控埋点、审计日志、降级响应。

熔断器是一个故障隔离机制，白话说，就是“当后端坏得太厉害时，暂时别再继续打它”。如果没有熔断，坏掉的服务会收到更多请求，响应更慢，调用方不断重试，最后把整条链路拖垮。

指数退避重试也是类似逻辑。它不是“失败后立刻狂打三次”，而是“失败后逐步拉长等待时间”，比如第 1 次等 100ms，第 2 次等 200ms，第 3 次等 400ms。这样做的目的，是给后端恢复窗口，避免放大瞬时流量。

一个简化的控制伪代码如下：

```text
if token_invalid:
    return 401

if request_schema_invalid:
    return 400

if rate_limit_exceeded:
    return 429

if circuit_open(service_name):
    return fallback_response()

for attempt in [1, 2, 3]:
    resp = call_backend_with_timeout()
    if resp.success:
        return resp
    record_failure()
    sleep(backoff(attempt))

if recent_failure_rate(service_name) > threshold:
    open_circuit(service_name)

return unified_error()
```

真实工程例子可以放到大模型平台里看。外部客户调用 `/v1/chat/completions` 这类 REST 接口，提交 JSON 消息；Gateway 校验 token、模型权限、输入大小、敏感字段；然后把请求分发给内部的路由服务；路由服务再通过 gRPC 调模型服务、embedding 服务、向量检索服务、缓存服务；如果向量检索超时，系统可直接降级为“无检索回答”而不是整体失败。这就是“对外稳定接口 + 对内可控调用链”的典型架构。

---

## 代码实现

先看一个最小化的 REST handler。这里不追求框架细节，而是突出顺序：认证、校验、限流、内部调用、错误翻译。

```python
import time
from typing import Dict, Any

RATE_BUCKET = {}
WINDOW_SECONDS = 60
LIMIT_PER_MINUTE = 3

def verify_jwt(token: str) -> bool:
    return token == "valid-token"

def validate_payload(data: Dict[str, Any]) -> None:
    assert "prompt" in data, "missing prompt"
    assert isinstance(data["prompt"], str), "prompt must be str"
    assert 1 <= len(data["prompt"]) <= 2000, "prompt length invalid"
    if "model" in data:
        assert data["model"] in {"gpt-basic", "gpt-pro"}, "invalid model"

def rate_limit(key: str) -> bool:
    now = int(time.time())
    window = now // WINDOW_SECONDS
    bucket_key = f"{key}:{window}"
    RATE_BUCKET[bucket_key] = RATE_BUCKET.get(bucket_key, 0) + 1
    return RATE_BUCKET[bucket_key] <= LIMIT_PER_MINUTE

def grpc_infer(prompt: str, model: str) -> Dict[str, Any]:
    if "timeout" in prompt:
        raise TimeoutError("backend timeout")
    return {"text": f"[{model}] {prompt[::-1]}", "usage": {"input_tokens": len(prompt)}}

def rest_chat_handler(headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    token = headers.get("Authorization", "").replace("Bearer ", "")
    if not verify_jwt(token):
        return {"status": 401, "error": {"code": "UNAUTHORIZED", "message": "invalid token"}}

    try:
        validate_payload(body)
    except AssertionError as e:
        return {"status": 400, "error": {"code": "BAD_REQUEST", "message": str(e)}}

    user_id = headers.get("X-User-Id", "anonymous")
    if not rate_limit(user_id):
        return {"status": 429, "error": {"code": "RATE_LIMITED", "message": "too many requests"}}

    try:
        result = grpc_infer(body["prompt"], body.get("model", "gpt-basic"))
        return {"status": 200, "data": result}
    except TimeoutError:
        return {"status": 503, "error": {"code": "UPSTREAM_TIMEOUT", "message": "please retry later"}}

ok = rest_chat_handler(
    {"Authorization": "Bearer valid-token", "X-User-Id": "u1"},
    {"prompt": "hello", "model": "gpt-basic"}
)
assert ok["status"] == 200
assert ok["data"]["text"] == "[gpt-basic] olleh"

bad = rest_chat_handler(
    {"Authorization": "Bearer invalid", "X-User-Id": "u1"},
    {"prompt": "hello"}
)
assert bad["status"] == 401
```

上面这段代码体现了一个关键原则：外部接口不要把内部错误原样抛出去。内部抛 `TimeoutError`，对外统一成 `503 + UPSTREAM_TIMEOUT`。这样客户端就能稳定处理错误，而不会因为后端某个库换了异常类型而崩掉。

再看 gRPC 的接口定义和调用关系。Protocol Buffers 是接口描述语言，白话说，就是“先写清楚字段和类型，再由工具生成代码”。

```proto
syntax = "proto3";

package inference;

service InferenceService {
  rpc Generate (GenerateRequest) returns (GenerateResponse);
}

message GenerateRequest {
  string prompt = 1;
  string model = 2;
  string request_id = 3;
}

message GenerateResponse {
  string text = 1;
  int32 input_tokens = 2;
}
```

如果把它翻成伪代码，Gateway 的职责会更清楚：

```python
def gateway_to_grpc(body, grpc_client):
    req = {
        "prompt": body["prompt"],
        "model": body.get("model", "gpt-basic"),
        "request_id": body.get("request_id", "generated-id"),
    }
    # 实际工程里这里会设置 deadline / metadata / trace id
    resp = grpc_client.Generate(req, timeout=1.5)
    return {
        "output": resp["text"],
        "usage": {"input_tokens": resp["input_tokens"]},
    }
```

新人最容易忽略的一点，是 REST 和 gRPC 的边界不只是“协议转换”，还包括“语义转换”。

例如：

1. REST 用 HTTP 状态码表达结果，gRPC 通常用 status code 和消息体。
2. REST 习惯资源路径和 JSON 字段命名，gRPC 习惯方法签名和 proto message。
3. REST 面向开放客户端，字段兼容性要更保守；gRPC 面向内部，版本演进可以更严格。

一个玩具例子是“字符串反转服务”。对外接口：

```http
POST /v1/reverse
Content-Type: application/json

{"text":"abc"}
```

返回：

```json
{"result":"cba"}
```

对内其实可以是 gRPC 的 `Reverse(text)`。这个例子足够小，便于看清：外部关心的是 JSON 格式和错误码；内部关心的是强类型定义和调用效率。

一个真实工程例子是“大模型问答 + 向量检索”。对外 REST 接口可能是：

`POST /v1/chat/completions`

输入包含 `messages`、`model`、`temperature`。Gateway 校验这些字段后，内部会拆成多个 gRPC 调用：

1. `AuthService.CheckQuota`
2. `Retriever.SearchDocuments`
3. `ModelRouter.SelectModel`
4. `InferenceService.Generate`
5. `BillingService.RecordUsage`

客户端完全不需要知道这 5 步，只看到一个稳定的 REST API。

---

## 工程权衡与常见坑

架构设计里最大的误区，是只比较协议性能，而忽略系统控制面。很多线上事故不是“JSON 比 protobuf 慢”，而是“没有超时、没有熔断、没有限流、重试策略乱配”。

常见问题可以直接列出来：

| 坑 | 后果 | 缓解措施 |
|---|---|---|
| 没有网关级超时 | 后端一慢，前端请求全部堆住 | 为每一跳设置 deadline，整体请求设置总超时 |
| 没有熔断 | 故障持续放大，健康实例被拖垮 | 按故障率/慢请求比例打开熔断 |
| 无限制重试 | 瞬时流量倍增，形成重试风暴 | 幂等接口才重试，配合指数退避和最大次数 |
| 限流只做在应用层 | 多实例下规则不一致 | 网关统一限流，必要时配合集中式配额 |
| 直接暴露内部错误 | 客户端无法稳定处理，泄露内部细节 | 统一错误码和错误结构 |
| 参数不做 schema 校验 | 畸形请求进入后端，排障困难 | 在入口校验类型、长度、枚举值、必填项 |
| 缓存策略过粗 | 返回过期数据或缓存击穿 | 区分静态配置、热数据、不可缓存数据 |
| 日志没有 request_id | 无法串联完整调用链 | 每个请求生成并透传 trace/request id |
| REST 和 gRPC 版本策略混乱 | 客户端升级困难，接口演进失控 | 外部接口显式版本化，内部 proto 做向后兼容 |

默认缓存和优雅降级值得单独强调。

缓存不是“所有东西都缓存”，而是“只缓存不会破坏正确性的结果”。例如模型列表、价格配置、公开元数据适合缓存；用户私有会话、实时计费、权限判断通常不适合直接缓存。缓存的本质是用空间换时间，但如果键设计错误，可能把 A 用户的数据返回给 B 用户，这是严重事故。

优雅降级是“部分能力失效时，返回可接受但较弱的结果”。例如：

1. 向量检索服务超时，降级为仅模型直答。
2. 推荐解释服务失败，只返回主结果，不返回扩展分析。
3. 非关键统计埋点失败，不影响主流程。

新人可以这样理解：如果没有熔断，某个 gRPC 服务慢了，REST 接口也会一直卡住；如果没有降级，原本“少一个辅助功能”会变成“整个接口都不可用”。

---

## 替代方案与适用边界

什么时候只用 REST？答案很直接：当你的主要目标是开放性和兼容性，而不是极致性能。比如公开 API、后台管理系统、第三方集成平台、浏览器直连场景，REST 通常更合适，因为文档、调试、跨语言支持和接入成本都更低。

什么时候只用 gRPC？当系统几乎完全是内部服务调用，而且请求频繁、链路长、字段结构稳定、类型约束强时，gRPC 会更合适，比如推荐系统、检索系统、内部计算平台。

双轨架构为什么常见？因为它刚好把两类问题拆开了：

1. 对外解决“谁都能接得上”的问题。
2. 对内解决“跑得快、约束清楚、链路稳定”的问题。

可以用一个决策矩阵快速判断：

| 决策维度 | 更偏 RESTful API | 更偏 gRPC |
|---|---|---|
| 客户端类型复杂 | 是 | 否 |
| 浏览器直接调用 | 是 | 否 |
| 低延迟要求极高 | 否 | 是 |
| 强类型约束重要 | 一般 | 很重要 |
| 流式通信频繁 | 一般 | 是 |
| 抓包调试频繁 | 是 | 否 |
| 第三方开发者接入 | 是 | 否 |
| 内部服务高频互调 | 一般 | 是 |
| 文档优先 | 是 | 一般 |
| 统一代码生成优先 | 一般 | 是 |

场景匹配可以这样看：

1. 开放给生态合作伙伴的模型调用平台：优先 REST。
2. 企业内部推理编排、检索、重排序服务：优先 gRPC。
3. 既有开放 API，又有复杂内部服务网格的大模型平台：REST + gRPC 双轨。

“开放 API 用 REST，内部微服务用 gRPC”不是口号，而是因为两者优化目标不同。可以把它理解成前台和后厨的分工，但更精确的说法是：前台负责稳定交互协议，后厨负责高效执行流程。前台不应暴露后厨工序，后厨也不该为适配所有外部客人而牺牲效率。

---

## 参考资料

1. Toptal, `gRPC vs REST: Understanding gRPC, OpenAPI and REST and When to Use Them`
   用途：REST 与 gRPC 的协议、数据格式、性能和典型场景对比。  
   链接：https://www.toptal.com/developers/grpc/grpc-vs-rest-api

2. Zigpoll, `Best Practices for Designing RESTful APIs to Ensure Security and Scalability`
   用途：REST API 的安全性、参数校验、错误处理、扩展性和监控实践。  
   链接：https://www.zigpoll.com/content/what-are-some-best-practices-for-designing-restful-apis-to-ensure-both-security-and-scalability

3. ReliabilityCalc, `Reliability Formulas`
   用途：MTBF、MTTR 与可用性公式，用于可靠性推导和 SLA 理解。  
   链接：https://reliabilitycalc.com/formulas/

4. Gravitee, `Choosing the Right API Architecture`
   用途：API 架构选择、网关角色、内部高性能通信与外部开放接口的组合实践。  
   链接：https://www.gravitee.io/blog/choosing-right-api-architecture

5. OpsMoon, `API Gateway Best Practices`
   用途：API Gateway 的限流、熔断、缓存、错误治理和常见故障规避。  
   链接：https://opsmoon.com/blog/api-gateway-best-practices

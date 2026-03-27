## 核心结论

LLM API 调用的最佳实践，不是“把请求发出去”这么简单，而是把失败当成系统的常规输入来设计。这里的“失败”指请求没有按预期完成，例如限流、超时、服务端报错或上下文超限。一个可上线的调用层，至少要同时处理四件事：错误分类、重试与退避、限流与并发控制、降级与回退。

可以先看一个结论公式。设单次调用成功率为 $p$，最多允许重试 $n$ 次，那么整体成功率近似为：

$$
P_{\text{success}} = 1 - (1-p)^{n+1}
$$

这个式子说明，只要失败是短暂的、可恢复的，少量重试就能明显提升整体可用性。但前提很严格：只能重试“暂态错误”，也就是短时间后可能恢复的错误，例如 429、5xx、网络抖动。对 401、403、参数错误这类“确定性错误”重试，只会放大流量和成本。

对初学者可以用一句白话理解：LLM API 像一家会偶尔暂停接单的餐厅，客户端必须会排队、会等一会再试、会换备用菜单，还要记下每次出问题的原因。这样系统才不会因为一次高峰流量直接失控。

下表可以作为最小稳定性设计的骨架：

| 故障类别 | 常见触发条件 | 是否适合重试 | 主控策略 |
|---|---|---:|---|
| 429 限流 | QPS 或 TPM 超标 | 是 | 指数退避、抖动、限流、排队 |
| 5xx 服务端错误 | 提供方暂时异常 | 是 | 短重试、超时控制、备用模型 |
| 网络超时 | 网络抖动、连接中断 | 是 | 幂等重试、连接超时、读超时 |
| 400/上下文超限 | prompt 太长、参数非法 | 否 | token 预检、裁剪、拆分 |
| 401/403 鉴权失败 | key 无效、权限不足 | 否 | 立即告警、停止重试 |
| 模型不可用 | 模型下线、区域故障 | 视情况 | provider fallback、缓存、降级 |

---

## 问题定义与边界

“最佳实践”讨论的不是提示词写法，而是调用层的可靠性设计。这里的调用层，是业务代码与模型提供方之间的一层适配逻辑，负责统一处理请求构造、错误识别、重试、日志、指标和降级。没有这一层，业务代码会在多个地方散落 `try/except`，最终既难维护，也无法稳定扩展。

边界需要先划清。不是所有失败都能靠重试解决，至少要分成三类：

| 错误类型 | 典型状态码/现象 | 本质 | 责任边界 | 是否重试 |
|---|---|---|---|---:|
| 暂态故障 | 429、502、503、超时 | 服务暂时忙或链路抖动 | 客户端与提供方共同承受 | 是 |
| 永久故障 | 401、403、模型不存在 | 配置或权限错误 | 客户端 | 否 |
| 请求缺陷 | 400、上下文超限、JSON 格式错 | 输入不合法 | 客户端 | 否 |
| 业务级失败 | 输出质量差、结构不符合预期 | 结果不满足业务要求 | 业务系统 | 通常不靠 HTTP 重试 |

这里有一个新手容易忽略的点：HTTP 成功不等于业务成功。比如接口返回 200，但模型输出的 JSON 缺字段，或者回答偏题，这属于“语义失败”，不能简单按网络失败处理。否则会出现“重复烧钱但没有更好结果”的情况。

玩具例子可以用“排队买咖啡”来理解。假设店里一次最多只能处理 10 个订单，第 11 个人立刻去抢窗口，得到的就是“稍后再来”。如果 200 个人都在同一秒重新冲进去，门口只会更堵。这对应真实系统中的 429 风暴。正确做法不是不停点刷新，而是限流、等待、错峰重试。

真实工程里更复杂。比如一个客服系统同时服务几千个在线用户，白天流量高峰时，摘要、改写、问答都共用同一组 API 配额。如果没有客户端侧的令牌预算、并发上限和错误分级，一次营销活动就可能把全站的 LLM 能力一起拖垮。此时出问题的不是“模型不够聪明”，而是调用系统没有边界控制。

---

## 核心机制与推导

最核心的机制有三个：重试、退避、降级。

第一，重试提升的是“短时恢复概率”。如果单次成功率为 $p$，单次失败率为 $1-p$，那么连续 $n+1$ 次都失败的概率是 $(1-p)^{n+1}$，因此至少成功一次的概率就是：

$$
P_{\text{success}} = 1 - (1-p)^{n+1}
$$

举一个最小数值例子。若单次成功率 $p=0.95$，最多重试 3 次，则：

$$
P_{\text{success}} = 1 - 0.05^4 = 0.99999375
$$

这说明在“错误确实可恢复”的前提下，少量重试会显著提高表面可用性。但这不是免费午餐，因为每次重试都会增加延迟和成本。

第二，退避控制的是“重试对系统的伤害”。最常见的是指数退避：

$$
delay_k = \min(base \times 2^k,\ max)
$$

其中 $k$ 是第几次重试，`base` 是初始等待时间，`max` 是等待上限。再加入 jitter，也就是抖动，目的是让不同请求不要在同一时刻一起重试。一个常见形式是：

$$
sleep_k = delay_k \times (0.5 + U(0, 1))
$$

这里 $U(0,1)$ 表示 0 到 1 的均匀随机数。白话解释：本来大家都准备 2 秒后一起重试，现在每个人多一点随机偏移，避免再次同时撞上限流阈值。

第三，降级处理的是“重试也救不回来”的情况。典型链路可以写成：

`主模型 -> 短重试 -> 备用模型 -> 缓存结果 -> 人工介入/友好失败`

这不是为了让所有请求都必须成功，而是为了让系统在故障态下仍然可预测。一个摘要接口在主模型故障时，切到便宜一点的小模型，质量可能略降，但比完全不可用更符合业务目标。

这三者的关系可以概括为：

| 机制 | 解决什么问题 | 代价 | 适用前提 |
|---|---|---|---|
| 重试 | 短暂失败 | 延迟、额外 token 成本 | 错误可恢复 |
| 退避+抖动 | 防止重试洪峰 | 响应更慢 | 并发请求较多 |
| 降级 | 长时间故障或成本超限 | 质量可能下降 | 业务允许结果分层 |

如果要再压缩成一句规则，就是：先判断错在哪，再决定要不要等、等多久、失败后退到哪一层。

---

## 代码实现

实现时，不要把重试逻辑写进每个业务函数里。更稳的做法是抽象一个 Provider 接口，再在外层包上 retry、fallback、observability。这里的 “observability” 指可观测性，也就是日志、指标、追踪，用来回答“哪里慢、哪里错、错了多少次”。

下面给一个可运行的 Python 示例，演示三件事：

1. 只对可重试错误重试  
2. 使用指数退避与 jitter  
3. 主 provider 失败后切到 fallback provider

```python
import random
import time


class RetryableError(Exception):
    pass


class NonRetryableError(Exception):
    pass


class FakeProvider:
    def __init__(self, outcomes, name="primary"):
        self.outcomes = list(outcomes)
        self.name = name
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if not self.outcomes:
            return f"{self.name}:ok:{prompt}"

        outcome = self.outcomes.pop(0)
        if outcome == "429":
            raise RetryableError("rate limited")
        if outcome == "500":
            raise RetryableError("server error")
        if outcome == "timeout":
            raise RetryableError("timeout")
        if outcome == "400":
            raise NonRetryableError("bad request")
        return f"{self.name}:ok:{prompt}"


def backoff_delay(base: float, attempt: int, max_delay: float) -> float:
    raw = min(base * (2 ** attempt), max_delay)
    jitter = 0.5 + random.random()  # [0.5, 1.5)
    return raw * jitter


def call_with_retry(provider, prompt, max_retries=3, base=0.1, max_delay=1.0, sleep=False):
    attempts = 0
    while True:
        try:
            return provider.generate(prompt), attempts
        except RetryableError:
            if attempts >= max_retries:
                raise
            delay = backoff_delay(base, attempts, max_delay)
            attempts += 1
            if sleep:
                time.sleep(delay)
        except NonRetryableError:
            raise


def call_with_fallback(primary, secondary, prompt):
    try:
        result, attempts = call_with_retry(primary, prompt, max_retries=2, sleep=False)
        return {"provider": primary.name, "text": result, "attempts": attempts}
    except RetryableError:
        result, attempts = call_with_retry(secondary, prompt, max_retries=1, sleep=False)
        return {"provider": secondary.name, "text": result, "attempts": attempts}


# 玩具例子：主服务先 429 再成功
p1 = FakeProvider(["429", "ok"], name="primary")
result, attempts = call_with_retry(p1, "hello", max_retries=3, sleep=False)
assert result == "primary:ok:hello"
assert attempts == 1

# 非重试错误不应重复调用
p2 = FakeProvider(["400"], name="primary")
try:
    call_with_retry(p2, "bad", max_retries=3, sleep=False)
    assert False, "should not reach"
except NonRetryableError:
    assert p2.calls == 1

# 真实工程缩略例子：主服务连续失败，切换备用服务
primary = FakeProvider(["429", "500", "timeout"], name="primary")
secondary = FakeProvider(["ok"], name="fallback")
final = call_with_fallback(primary, secondary, "summarize this")
assert final["provider"] == "fallback"
assert final["text"] == "fallback:ok:summarize this"
```

这段代码故意没有接真实 SDK，因为重点是结构。真正接生产 API 时，至少还要补上四类能力：

| 能力 | 为什么要有 | 最低要求 |
|---|---|---|
| 超时控制 | 防止请求无限挂起 | 连接超时、读取超时分开配置 |
| `Retry-After` 支持 | 尊重服务端建议等待时间 | 优先使用响应头 |
| 指标埋点 | 判断是否需要扩容或调策略 | 成功率、P95 延迟、重试次数、fallback 次数 |
| 幂等性设计 | 防止重试导致重复副作用 | 为写操作设置 request id |

真实工程例子可以看在线客服系统。假设用户发来一段工单文本，系统先做“分类 + 摘要 + 回复草稿”。这三个步骤不该各自散写调用代码，而应统一走一个 LLM 网关层：

1. 网关先做 token 预估，超限则切分输入  
2. 再做全局限流，避免某个租户吃光配额  
3. 请求主模型，遇到 429/5xx 则重试  
4. 达到阈值后切小模型，或直接返回缓存模板  
5. 全程记录 `trace_id`、模型名、token 数、重试次数、最终耗时

这样后续排查时，才能区分是“主模型今天不稳定”，还是“某个租户 prompt 异常增长”，还是“fallback 触发太频繁导致质量投诉”。

---

## 工程权衡与常见坑

最大误区是把“重试”当成万能药。实际上，重试只适合暂态故障，不适合输入错误和权限错误。把 400、401 也放进重试，会同时造成三种坏结果：用户等待更久、费用更高、系统更堵。

下面这个表可以直接当 checklist：

| 常见坑 | 现象 | 后果 | 规避策略 | 关键指标 |
|---|---|---|---|---|
| 重试范围过大 | 400/401 也重试 | 无效放大流量 | 只重试 429/5xx/超时 | retry_success_rate |
| 没有 jitter | 同步重试 | 二次限流洪峰 | 指数退避加随机抖动 | requests_per_second |
| 忽略 token 限制 | 上下文超限 | 非重试失败 | token 预检、裁剪、分块 | input_tokens, 400_rate |
| 没有超时 | 请求长期挂起 | 线程/连接占满 | 分层超时 | timeout_rate |
| 没有 fallback | 主模型故障即全挂 | 可用性差 | 备用模型/缓存 | fallback_count |
| 没有幂等 | 重试导致重复写入 | 数据污染 | request id、去重 | duplicate_write_count |
| 只看 HTTP 状态 | 200 但输出不可用 | 业务失败被漏掉 | 结果校验、结构校验 | schema_error_rate |

另一个常见问题是只做“接口级稳定”，不做“业务级稳定”。比如一个结构化抽取接口，返回 200，但 JSON 缺字段。如果你直接把结果写数据库，后面的流程照样崩。这里要加输出校验，必要时把“结构不合法”视为可恢复的业务失败，再决定是否重试或切备用模型。

还要注意成本。假设一次请求平均消耗 $c$，重试上限是 $n$，那么单用户请求的最坏成本上界接近：

$$
Cost_{\max} \approx (n+1) \times c
$$

所以“可用性更高”通常意味着“成本更高、尾延迟更长”。工程上不能只问是否成功，还要问是否值得。一个面向 C 端实时聊天的接口，可能更重视低延迟，会把重试次数压到 1 到 2 次；一个离线批处理摘要任务，则可以接受更长等待，用更多重试换吞吐完成率。

---

## 替代方案与适用边界

不是所有场景都适合“主模型失败后继续重试”。有些场景更适合直接走替代方案。

最常见的替代链有三种：

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 备用 provider | 主 provider 故障或区域异常 | 可用性高 | 输出风格和质量可能漂移 |
| 备用小模型 | 对实时性敏感、质量可略降 | 成本低、切换快 | 复杂任务效果可能明显变差 |
| 缓存结果 | 相似问题高频重复 | 响应极快、最省钱 | 新鲜度有限 |
| 人工介入 | 高价值、低容错业务 | 质量可控 | 人力成本高、吞吐低 |

可以把决策过程理解成一个简化流程：

1. 请求进入，先做输入校验和 token 预检  
2. 主模型调用失败，若是 429/5xx/超时，则短重试  
3. 若累计等待超过阈值，比如 2 秒或 3 次尝试，则进入降级  
4. 优先切备用模型；若业务允许，可直接返回缓存  
5. 若是高风险场景，如金融审核、医疗建议，则转人工或拒答  
6. 主模型恢复后，再逐步回切，不要瞬间全部流量打回去

这里的“回切”也要谨慎。很多系统只设计了 fallback，没有设计 recovery。结果是主模型已经恢复，系统还长期跑在贵的备用 provider 上，或者继续使用质量较差的小模型。正确做法是持续打点：记录 fallback 触发原因、持续时长、回切时间、回切后的成功率变化。

适用边界也要说清。以下场景不能只靠本文这些模式解决：

1. 输出必须绝对一致的场景。多 provider 切换会引入风格漂移。  
2. 有强副作用的写操作。比如“自动下单”“自动发邮件”，重试必须有更强的幂等保障。  
3. 超低延迟场景。若接口预算只有 300ms，多次重试本身就不现实。  
4. 超长上下文任务。这里更关键的是检索、分块、压缩，而不是重试。  

所以，最佳实践不是固定模板，而是围绕业务目标做约束设计。实时系统优先低延迟，批处理优先完成率，高风险系统优先可审计性。重试、退避、fallback 只是工具，不是目的。

---

## 参考资料

- Statsig, Provider fallbacks: Ensuring LLM availability  
- Grizzly Peak Software, LLM API Error Handling and Retry Patterns  
- BackendBytes, LLM API Integration Patterns for Backend Engineers  
- Fast.io, AI Agent Retry Patterns  
- Artificial Intelligence Wiki, LLM Error Handling Guide  
- Dev.to, Prompt Rate Limits & Batching: How to Stop Your LLM API from Melting Down

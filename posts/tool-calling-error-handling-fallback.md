## 核心结论

工具调用的错误处理，不是“失败了就重试”，而是先分型，再决定回退动作。对智能体系统来说，工具调用就是“让模型去执行外部能力”，比如查数据库、调用支付接口、读对象存储。只要进入外部系统，失败就不再只有一种。

一个实用的最小结论是：把错误稳定地归入五类，然后给每类绑定默认策略。

| 错误类型 | 常见状态/信号 | 典型原因 | 首选回退策略 |
|---|---:|---|---|
| 参数校验失败 | 400 | 必填字段缺失、类型不对、枚举值非法 | 参数修正，不直接重试 |
| API 超时/率限 | 408/429/超时异常 | 网络抖动、服务繁忙、配额触顶 | 指数退避重试 |
| 权限不足 | 401/403 | API Key 失效、角色权限不够 | 替代工具或降级为只读 |
| 资源不存在 | 404 | ID 错误、路径错误、资源已删除 | 用户确认，不盲目重试 |
| 服务端内部错误 | 500/502/503 | 下游异常、服务不稳定、临时故障 | 有上限的重试，最终失败外显 |

如果只记一个设计原则，可以记这个：

$$
\text{处理动作} = f(\text{错误类别}, \text{是否可恢复}, \text{是否可替代}, \text{是否需要用户确认})
$$

这里“可恢复”意思是系统能否自己修复；“可替代”意思是能否换一个工具或降级能力；“需要用户确认”意思是继续执行前，必须让用户补充真实世界信息。

这个公式可以直接落成一张决策表：

| 错误类别 | 系统能否自修复 | 是否存在替代路径 | 是否必须问用户 | 默认动作 |
|---|---|---|---|---|
| `validation_error` | 是 | 通常否 | 有时否 | 修正参数后重发 |
| `timeout_retryable` | 是 | 不一定 | 否 | 重试 |
| `permission_denied` | 否 | 常常是 | 有时否 | 降级或换工具 |
| `not_found` | 通常否 | 否 | 常常是 | 停下并确认 |
| `server_error` | 有时是 | 不一定 | 否 | 有上限重试 |

玩具例子：一个天气工具要求 `city` 字段，模型却传了 `ctiy`。这是参数校验失败，正确动作不是重试三次，而是修正参数名再发一次。

真实工程例子：对象存储列表接口返回 403，并提示缺少 `storage.buckets.list`。这不是网络问题，也不是服务挂了，而是权限边界不满足。正确动作通常是切到只读视图、提示联系管理员，或者改用一个不需要该权限的替代工具。

再强调一次：错误处理的目标不是“把所有失败都藏起来”，而是把失败变成可解释、可继续、可观测的控制流。

---

## 问题定义与边界

“工具调用的错误处理”指的是：当智能体调用外部 API、数据库、文件系统或内部服务失败时，系统如何把失败转换成可继续执行、可定位、可观察的控制流。

这里的“回退策略”就是失败后的备用动作。白话说，它不是把错误吞掉，而是给系统一条次优但合理的出路。

这个问题的边界要先讲清楚，否则很容易把所有失败都丢给“重试”：

1. 参数错误不是瞬时错误。缺字段、字段类型错、格式错，这类问题靠时间不会变好。
2. 权限错误不是网络错误。没有权限时，再重试十次，结果通常还是 401/403。
3. 资源不存在和资源暂不可用不是一回事。404 常常说明你找错对象了，不是服务在抖。
4. 5xx 和超时才是典型的瞬时错误。瞬时错误的特点是“稍后重试可能成功”。
5. 率限 429 介于两者之间。它不是参数错，但也不是服务坏了，而是“你现在发得太快”。

因此，一个稳定系统通常先做分类，再决定是否自动处理。常见结构化错误对象可以长这样：

```json
{
  "error": {
    "type": "permission_denied",
    "code": "MISSING_PERMISSION",
    "message": "Missing required permission: storage.buckets.list",
    "details": {
      "required_permission": "storage.buckets.list",
      "resource": "projects/demo-project"
    },
    "request_id": "req_9f2c1a"
  }
}
```

这类结构里的术语含义要明确：

| 字段 | 作用 | 白话解释 |
|---|---|---|
| `type` | 错误类别 | 让程序先知道这是什么大类问题 |
| `code` | 细粒度代码 | 让程序区分同一类里的具体场景 |
| `message` | 人类可读描述 | 给开发者和用户看的短说明 |
| `details` | 结构化上下文 | 放具体缺哪个字段、哪个权限、哪个资源 |
| `request_id` | 请求追踪号 | 方便日志和平台侧排查同一次请求 |

新手最容易混淆的是两层语义：

| 层次 | 例子 | 用途 |
|---|---|---|
| 协议层信号 | HTTP 400、403、429、500 | 说明接口层面发生了什么 |
| 业务层语义 | 参数缺失、权限不足、速率限制、下游故障 | 决定系统下一步该怎么做 |

协议层和业务层不能直接画等号。比如两个接口都返回 400，一个可能是字段缺失，另一个可能是查询条件冲突。前者适合自动补字段，后者可能必须重新规划调用路径。

边界还包括“不要过度智能”。例如 404 时，系统可以提示“资源不存在，是否确认 ID”，但不应该擅自创建一个新资源，除非业务规则明确允许。再比如支付、扣费、发通知这类写操作，即使遇到 503，也不能只因为“看起来像瞬时故障”就无限重试，因为它们可能带来重复副作用。

一句话概括边界：错误处理是在正确性约束下做恢复，不是在所有场景里追求继续执行。

---

## 核心机制与推导

错误处理的核心机制有两层：分类器和决策器。

分类器负责把失败映射到统一语义。决策器负责根据语义执行动作。可以把它理解成：

$$
\text{error} \rightarrow \text{normalize} \rightarrow \text{policy} \rightarrow \text{fallback action}
$$

“normalize”是归一化，白话说就是把不同供应商、不同 SDK、不同状态码风格，统一翻译成你系统内部能理解的错误模型。

一个常见内部模型可以写成：

| 内部类别 | 识别条件 | 默认动作 |
|---|---|---|
| `validation_error` | 400 或字段校验异常 | 修正参数 |
| `timeout_retryable` | 超时、408、429 | 指数退避 |
| `permission_denied` | 401/403 | 降级或替代 |
| `not_found` | 404 | 用户确认 |
| `server_error` | 500/502/503 | 有上限重试 |

这一步的关键不是“把名字改好看”，而是建立稳定的策略接口。只要归一化后的类别不变，你可以更换底层供应商、SDK，甚至把 HTTP 工具换成数据库驱动，策略层都不用重写。

为什么重试要用指数退避，而不是固定 1 秒重试一次？因为固定间隔容易让大量客户端同时回来，形成“惊群”。“惊群”就是一群请求在同一时刻再次撞向服务，把原本短暂的故障放大成持续拥塞。

指数退避公式通常写成：

$$
delay = \min(cap,\ base \times 2^{attempt}) + jitter
$$

其中：

- `base` 是初始等待时间
- `cap` 是最大等待上限
- `attempt` 是第几次失败
- `jitter` 是随机扰动，白话说就是再加一点随机数，避免大家同一秒重试

玩具例子：令 `base = 1s`，`cap = 32s`。

| 失败次数 `attempt` | 指数部分 | 截断后等待 | 若 `jitter \in [0, 1)` |
|---|---:|---:|---:|
| 0 | $1 \times 2^0$ | 1s | 1s 到 2s 之间 |
| 1 | $1 \times 2^1$ | 2s | 2s 到 3s 之间 |
| 2 | $1 \times 2^2$ | 4s | 4s 到 5s 之间 |
| 3 | $1 \times 2^3$ | 8s | 8s 到 9s 之间 |
| 5 | $1 \times 2^5$ | 32s | 32s 到 33s 之间 |

这样做的作用不是“更慢”，而是“更稳”。大量客户端不会在同一秒再次压向同一个下游。

但这个公式只适合可重试错误。对参数校验失败，数学上等待时间再漂亮也没有意义，因为输入本身是错的。

真实工程里，决策逻辑一般还要加三个约束：

1. 最大尝试次数。常见是 3 到 5 次。
2. 总重试预算。单次调用不能把整条链路的超时预算吃光。
3. 幂等性检查。幂等性就是“同样请求执行多次，结果和一次相同”。

可以把决策器再写得更完整一点：

$$
\text{是否重试} =
\mathbf{1}[
\text{错误可重试}
\land \text{attempt} < \text{max\_retries}
\land \text{幂等}
\land \text{剩余超时预算} > 0
]
$$

这里的 `幂等` 很重要。读操作通常更适合重试，写操作必须谨慎。如果一个写操作不是幂等的，比如“重复扣款”，那就不能简单套用重试模板，而要配合幂等键、去重表或事务语义。

新手常见误区是：把“可重试”和“值得重试”当成一回事。并不是所有 5xx 都值得重试；如果本次用户请求的总超时只剩 200ms，而你下一次退避就要等 1 秒，那这次重试在系统层面已经不划算了。

---

## 代码实现

下面给一个可运行的 Python 示例。它展示四件事：

1. 先把原始错误归一化。
2. 再按类别选择参数修正、重试、降级或用户确认。
3. 在 429/503 时优先尊重服务端给出的 `retry_after`。
4. 记录结构化结果，便于前端和日志统一消费。

```python
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolError(Exception):
    status: int
    error_type: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = "req_demo"


@dataclass
class NormalizedError:
    category: str
    retryable: bool
    user_action_required: bool
    details: dict[str, Any]
    message: str
    request_id: str


def compute_delay(
    *,
    attempt: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter_max: float = 0.5,
    retry_after: float | None = None,
) -> float:
    # 如果服务端明确告诉你多久后再试，优先尊重它。
    if retry_after is not None:
        return max(0.0, min(cap, retry_after))

    delay = min(cap, base * (2 ** attempt))
    delay += random.uniform(0.0, jitter_max)
    return delay


def normalize_error(exc: ToolError) -> NormalizedError:
    if exc.status == 400:
        return NormalizedError(
            category="validation_error",
            retryable=False,
            user_action_required=False,
            details=exc.details,
            message=exc.message,
            request_id=exc.request_id,
        )

    if exc.status in (408, 429):
        return NormalizedError(
            category="timeout_retryable",
            retryable=True,
            user_action_required=False,
            details=exc.details,
            message=exc.message,
            request_id=exc.request_id,
        )

    if exc.status in (401, 403):
        return NormalizedError(
            category="permission_denied",
            retryable=False,
            user_action_required=False,
            details=exc.details,
            message=exc.message,
            request_id=exc.request_id,
        )

    if exc.status == 404:
        return NormalizedError(
            category="not_found",
            retryable=False,
            user_action_required=True,
            details=exc.details,
            message=exc.message,
            request_id=exc.request_id,
        )

    if exc.status in (500, 502, 503):
        return NormalizedError(
            category="server_error",
            retryable=True,
            user_action_required=False,
            details=exc.details,
            message=exc.message,
            request_id=exc.request_id,
        )

    return NormalizedError(
        category="unknown_error",
        retryable=False,
        user_action_required=False,
        details=exc.details,
        message=exc.message,
        request_id=exc.request_id,
    )


def repair_params(params: dict[str, Any], err: NormalizedError) -> dict[str, Any]:
    repaired = dict(params)
    missing_fields = err.details.get("missing_fields", [])

    # 一个最小但常见的自动修正：把明显拼错的键纠正回来。
    if "city" in missing_fields and "city" not in repaired and "ctiy" in repaired:
        repaired["city"] = repaired.pop("ctiy")

    return repaired


class FakeWeatherTool:
    def __init__(self) -> None:
        self.rate_limit_counter = 0

    def call(self, params: dict[str, Any], mode: str) -> dict[str, Any]:
        if mode == "bad_param":
            if "city" not in params:
                raise ToolError(
                    status=400,
                    error_type="invalid_request",
                    code="MISSING_FIELD",
                    message="city is required",
                    details={"missing_fields": ["city"]},
                )
            return {"weather": "sunny", "city": params["city"]}

        if mode == "timeout_then_success":
            self.rate_limit_counter += 1
            if self.rate_limit_counter < 3:
                raise ToolError(
                    status=429,
                    error_type="rate_limit",
                    code="RATE_LIMITED",
                    message="too many requests",
                    details={"retry_after": 1.0},
                )
            return {"weather": "cloudy", "city": params.get("city", "Shanghai")}

        if mode == "forbidden":
            raise ToolError(
                status=403,
                error_type="permission_denied",
                code="MISSING_PERMISSION",
                message="missing required permission",
                details={"required_permission": "storage.buckets.list"},
            )

        if mode == "not_found":
            raise ToolError(
                status=404,
                error_type="not_found",
                code="RESOURCE_NOT_FOUND",
                message="resource does not exist",
                details={"resource_id": "doc_123"},
            )

        raise ToolError(
            status=503,
            error_type="service_unavailable",
            code="UNAVAILABLE",
            message="service temporarily unavailable",
            details={"retry_after": 0.2},
        )


def call_with_fallback(
    tool: FakeWeatherTool,
    params: dict[str, Any],
    mode: str,
    *,
    max_retries: int = 3,
    is_idempotent: bool = True,
    sleeper: Callable[[float], None] = lambda _: None,
) -> dict[str, Any]:
    working = dict(params)

    for attempt in range(max_retries + 1):
        try:
            data = tool.call(working, mode)
            return {
                "status": "ok",
                "data": data,
                "attempts": attempt + 1,
                "fallback": None,
            }
        except ToolError as exc:
            err = normalize_error(exc)

            if err.category == "validation_error":
                repaired = repair_params(working, err)
                if repaired != working:
                    working = repaired
                    continue
                return {
                    "status": "failed",
                    "error_category": err.category,
                    "action": "fix_parameters",
                    "user_message": err.message,
                    "details": err.details,
                    "request_id": err.request_id,
                }

            if err.category == "permission_denied":
                return {
                    "status": "degraded",
                    "error_category": err.category,
                    "action": "fallback_read_only",
                    "user_message": "权限不足，已切换为只读模式",
                    "details": err.details,
                    "request_id": err.request_id,
                }

            if err.category == "not_found":
                return {
                    "status": "blocked",
                    "error_category": err.category,
                    "action": "ask_user_confirmation",
                    "user_message": "资源不存在，请确认 ID 或路径",
                    "details": err.details,
                    "request_id": err.request_id,
                }

            if err.retryable and is_idempotent and attempt < max_retries:
                retry_after = err.details.get("retry_after")
                delay = compute_delay(attempt=attempt, retry_after=retry_after)
                sleeper(delay)
                continue

            return {
                "status": "failed",
                "error_category": err.category,
                "action": "retry_exhausted" if err.retryable else "surface_error",
                "user_message": err.message,
                "details": err.details,
                "request_id": err.request_id,
            }

    raise AssertionError("unreachable")


def main() -> None:
    tool = FakeWeatherTool()

    # 1. 参数自动修正
    result1 = call_with_fallback(tool, {"ctiy": "Shanghai"}, "bad_param")
    assert result1["status"] == "ok"
    assert result1["data"]["city"] == "Shanghai"

    # 2. 429 后按 retry_after 重试，最终成功
    result2 = call_with_fallback(tool, {"city": "Shanghai"}, "timeout_then_success")
    assert result2["status"] == "ok"
    assert result2["attempts"] == 3

    # 3. 权限不足时降级
    result3 = call_with_fallback(tool, {}, "forbidden")
    assert result3["status"] == "degraded"
    assert result3["action"] == "fallback_read_only"

    # 4. 资源不存在时阻塞并要求用户确认
    result4 = call_with_fallback(tool, {}, "not_found")
    assert result4["status"] == "blocked"
    assert result4["action"] == "ask_user_confirmation"

    # 5. 503 在达到上限后失败外显
    result5 = call_with_fallback(tool, {}, "server_error", max_retries=2)
    assert result5["status"] == "failed"
    assert result5["action"] == "retry_exhausted"

    print("all checks passed")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，执行结果应为：

```text
all checks passed
```

这段实现故意保持最小，但它已经覆盖了文章主线：

| 错误类别 | 代码里怎么识别 | 采取的动作 |
|---|---|---|
| `validation_error` | `status == 400` | 先尝试 `repair_params()` |
| `timeout_retryable` | `status in (408, 429)` | 按 `retry_after` 或指数退避重试 |
| `permission_denied` | `status in (401, 403)` | 返回 `degraded`，切只读 |
| `not_found` | `status == 404` | 返回 `blocked`，要求确认 |
| `server_error` | `status in (500, 502, 503)` | 有上限重试，超限后外显 |

玩具例子在这里很直观：`ctiy` 拼错后，系统通过 `missing_fields` 推断出该修正参数，而不是向用户直接报一个生硬的 400。

真实工程例子可以这样理解。假设一个智能体帮用户列出云存储桶：

1. 模型决定调用 `list_buckets`。
2. 后端返回 403，`details.required_permission = storage.buckets.list`。
3. 系统把错误归类为 `permission_denied`。
4. 前端不显示“系统异常”，而是显示“当前账号没有列出存储桶权限，已切换为只读模式”。
5. 同时日志记录 `request_id`、用户 ID、工具名、所需权限。
6. 上层编排器继续后续流程，但只允许读取缓存摘要，不再尝试写操作。

这一步的价值在于：失败没有让整个智能体流程中断，而是被转成了一个可解释的降级状态。

如果你要把这段代码真正放进生产系统，还要再补三件事：

| 需要补的能力 | 为什么要补 |
|---|---|
| 超时预算传递 | 避免单个工具把整条链路耗尽 |
| 熔断或限流 | 避免下游故障时持续放大流量 |
| 结构化埋点 | 便于统计哪类错误在上涨 |

---

## 工程权衡与常见坑

错误处理做得越自动，越要小心误判。很多线上问题不是“不会处理错误”，而是“处理得太想当然”。

下面是常见坑：

| 常见坑 | 结果 | 更合理的做法 |
|---|---|---|
| 所有错误一律重试 | 参数错、权限错被重复放大 | 先分类，再决定 |
| 不设最大重试次数 | 形成 retry storm | 限制 3-5 次并设置 cap |
| 不加 jitter | 大量客户端同一时刻回打 | 为每次重试加入随机扰动 |
| 忽略 `Retry-After` | 和服务端节流信号冲突 | 优先尊重服务端建议 |
| 只记录文本日志 | 后续无法按类别聚合分析 | 记录结构化字段 |
| 把 403 当网络问题 | 用户无法知道真实阻塞点 | 明确提示缺失权限 |
| 把 404 自动重试 | 无意义消耗资源 | 改为用户确认 |
| 写请求直接重试 | 可能产生重复副作用 | 要求幂等键或补偿机制 |

最容易踩的第一个坑，是“统一异常捕获后 sleep 一下再来”。这看起来简单，但实际会把错误语义全丢掉。一个成熟系统要关心的不只是“有没有失败”，还要关心“失败是否值得继续尝试”。

第二个坑是忽略可观测性。“可观测性”就是系统是否能从日志、指标、追踪里看出问题。白话说，就是出了故障后，你能不能知道哪类错误在涨、集中在哪个工具、影响了多少用户。

建议至少记录这些字段：

| 字段 | 说明 |
|---|---|
| `tool_name` | 哪个工具失败 |
| `error_category` | 归一化后的错误类别 |
| `status/code` | 原始协议层信号 |
| `request_id` | 外部服务追踪号 |
| `attempt` | 第几次尝试 |
| `fallback_action` | 最终执行了什么回退 |
| `latency_ms` | 调用耗时 |
| `is_idempotent` | 本次调用是否允许重试 |
| `degraded` | 是否进入降级路径 |

如果要做指标面板，最少看三组数：

| 指标 | 用途 |
|---|---|
| `error_rate by category` | 看哪类错误在上涨 |
| `retry_success_rate` | 看重试是否真的有价值 |
| `degrade_rate` | 看有多少请求靠降级撑住 |

第三个坑是忽略用户体验。工程师常说“报错了”，但用户真正想知道的是“我下一步该做什么”。所以同一个错误，需要分成机器可处理信息和用户可理解信息：

- 机器读：`type/code/details/request_id`
- 用户读：补什么字段、联系谁、要不要重试、是否已降级

下面是一组更实用的用户提示模板：

| 错误类别 | 不推荐文案 | 推荐文案 |
|---|---|---|
| 参数错误 | 请求失败 | 缺少 `city` 字段，已尝试自动修正；若仍失败，请补充城市名 |
| 权限不足 | 系统异常 | 当前账号缺少 `storage.buckets.list` 权限，已切换为只读模式 |
| 资源不存在 | 未知错误 | 资源 `doc_123` 不存在，请确认 ID 或路径 |
| 率限 | 请稍后再试 | 请求过于频繁，系统将在 1 秒后自动重试 |
| 服务错误 | 请求失败 | 下游服务暂时不可用，系统已重试 3 次，稍后可再次发起 |

真实工程里，最差的体验是返回一句“请求失败，请稍后再试”。这句话对 400、403、404、500 都一样，等于没有信息。

还有一个经常被忽略的坑：多层重试叠加。假设网关重试 3 次、编排器重试 3 次、底层 SDK 再重试 3 次，最坏情况下会把一次下游失败放大成 $3 \times 3 \times 3 = 27$ 次请求。如果链路更深，放大效应会更明显。解决办法不是“每层都关掉”，而是明确哪一层拥有最终重试权。

---

## 替代方案与适用边界

不是所有场景都要自己写一整套回退框架。是否需要复杂策略，取决于工具的重要性、调用频率和副作用强弱。

一种替代方案是“失败即外显”。也就是不自动回退，直接把错误透给上层。这适合内部工具、低频运维脚本、人工值守系统，因为人能直接读日志修复。

另一种替代方案是“只做有限降级，不做自动修正”。例如权限不足时只切只读，参数错误时直接要求模型重新规划，而不是尝试自动补字段。这适合高风险写操作，因为自动修正可能修错对象。

第三种替代方案是“策略分层”：

| 层次 | 负责内容 | 适合放什么策略 |
|---|---|---|
| SDK 层 | 单次请求 | 超时、签名、基础重试 |
| 工具适配层 | 供应商差异归一化 | 错误分类、字段修正 |
| 编排层 | 任务是否继续 | 换工具、降级、用户确认 |
| 产品层 | 最终交互 | 友好提示、人工介入入口 |

这比把所有逻辑塞进一个 `try/except` 更稳定，因为每层只处理自己最了解的问题。

还可以按触发条件选策略：

| 触发条件 | 推荐策略 | 适用边界 |
|---|---|---|
| 400 参数错误 | 参数修正或重新生成参数 | 参数结构清晰、修正规则明确 |
| 401/403 权限不足 | 替代工具、只读模式、人工申请权限 | 存在能力降级路径 |
| 404 资源不存在 | 用户确认、展示新建入口 | 资源标识来自用户输入 |
| 429/超时 | 指数退避重试 | 读操作、幂等写操作 |
| 500/502/503 | 有上限重试，最终失败外显 | 临时性下游故障 |
| 高副作用写请求 | 谨慎重试，优先幂等机制 | 支付、扣费、发消息 |

新手可以直接用下面这张最小选型表：

| 你的场景 | 建议 |
|---|---|
| 内部脚本、人工值守 | 失败外显即可 |
| 读操作很多、下游偶发抖动 | 加重试和退避 |
| 有权限边界、角色复杂 | 加降级和权限提示 |
| 写操作有副作用 | 先做幂等，再谈自动重试 |
| 面向最终用户 | 必须区分机器信息和用户文案 |

玩具例子：记事本工具 `get_note(note_id)` 返回 404。若 `note_id` 来自用户手输，系统应提示确认 ID，而不是继续重试。因为不存在的信息不会因等待 8 秒而出现。

真实工程例子：企业内部智能体调用 IAM 受控接口时，403 说明角色边界不足。此时最好的策略通常不是“继续重试”，而是：

1. 记录缺失权限。
2. 切到只读或摘要模式。
3. 告诉用户需要管理员授予什么权限。
4. 保留当前上下文，等权限恢复后继续。

这类边界说明一个事实：回退策略不是“让每次调用都成功”，而是“让系统在失败时仍然保持正确、稳定、可理解”。

---

## 参考资料

- RFC 9110, HTTP Semantics: https://www.rfc-editor.org/rfc/rfc9110
- RFC 6585, Additional HTTP Status Codes（含 429）: https://www.rfc-editor.org/rfc/rfc6585
- MDN, Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Retry-After
- AWS Builders' Library, Timeouts, retries, and backoff with jitter: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/
- Google Cloud IAM, Troubleshoot permission error messages: https://cloud.google.com/iam/docs/permission-error-messages
- Stripe API, Idempotent requests: https://docs.stripe.com/api/idempotent_requests
- Google Cloud Storage JSON API status and error codes: https://cloud.google.com/storage/docs/json_api/v1/status-codes
- AWS Well-Architected Framework, Mitigate interaction failure: https://docs.aws.amazon.com/wellarchitected/latest/framework/rel_mitigate_interaction_failure.html

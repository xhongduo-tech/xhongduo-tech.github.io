## 核心结论

Agent 的工具定义规范，本质上不是“给模型一点提示词”，而是定义一份机器可执行的契约。这里的“契约”可以直白理解为：调用前双方先把规则写清楚，谁都不能靠猜。最常见的表达方式就是 JSON Schema，它明确字段名、类型、是否必填、取值范围、格式约束，以及工具返回结果应该长什么样。

把这个问题说成公式就是：

$$
执行成功 \equiv Schema(Input)\ \land\ TypeConvert(Input)\ \land\ ToolReturn\ \land\ ErrorPolicy
$$

意思是，一次自动调用是否真正成功，不只看“工具跑没跑起来”，而要同时满足四件事：

1. 输入先通过 Schema 校验。
2. 必要的类型转换成功，比如把 `"49.95"` 转成 `49.95`。
3. 工具返回值本身也符合输出约束。
4. 出错时按既定策略处理，比如超时重试、429 退避、不可恢复错误直接降级。

一个新手最容易理解的玩具例子是支付工具。它只接受：

```json
{"amount": 49.95, "currency": "USD"}
```

其中 `amount` 必须是数字且 $\ge 0.01$，`currency` 必须是三位大写货币码。如果模型生成了 `"49.95"` 这种字符串，只要 Schema 配合转换规则允许，它仍然可以被安全地改成数字后执行。反过来，如果把 `"forty nine"` 也放过去，那就不是“模型灵活”，而是系统边界失效。

下面这个表，是工具 Schema 里最核心的四类信息：

| field | type | constraints | description |
|---|---|---|---|
| `amount` | `number` | `minimum: 0.01` | 支付金额，最小 0.01 |
| `currency` | `string` | `pattern: ^[A-Z]{3}$` | 三位大写货币码，如 `USD` |
| `order_id` | `string` | `minLength: 1` | 订单编号，不能为空 |
| `quantity` | `integer` | `minimum: 1` | 商品数量，必须是正整数 |

工程上最重要的结论是：只定义输入 Schema 还不够。真正稳定的 Agent 工具系统，必须同时定义输入、输出、错误、重试和降级路径。否则它只是“能演示”，不是“能上线”。

---

## 问题定义与边界

Agent 和工具之间的关系，可以理解为“自然语言世界”和“程序世界”的接口层。Agent 擅长从用户的话里提取意图，但工具只接受结构化参数。工具定义规范，就是把这层接口从“概率性理解”变成“确定性边界”。

输入边界可以简化写成：

$$
输入边界 = \{required\ fields\} \land type\ constraints
$$

这里有三个边界必须写清楚。

第一，哪些字段必须出现。`required` 的白话解释是“缺了就不能调用”。例如转账工具里，`amount`、`currency`、`recipient` 通常都是必填。模型少生成一个，系统应该立刻拦截，而不是带着半残参数往下游冲。

第二，字段类型必须明确。`number` 不是 `string`，`integer` 不是任意数字，`boolean` 也不是 `"yes"`。如果没有明确类型，模型可能产出“人能看懂、程序不能跑”的 JSON。

第三，格式和取值范围要有边界。比如邮箱要符合邮件格式，日期要统一为 `YYYY-MM-DD`，货币码要匹配 `^[A-Z]{3}$`。这类约束不是“吹毛求疵”，而是在阻止错误进入系统深处。

一个典型新手例子是：用户说“转 50 美元”。Agent 可能先生成：

```json
{"amount": 50}
```

但工具 Schema 规定 `currency` 必须存在。如果它缺失，正确做法不是默默默认成 `USD`，而是立即报验证错误，或者回问用户，或者根据业务上下文补全后重新验证。是否允许默认值，必须由工具定义，而不是由模型临场猜测。

下面这个表，总结了常见输入边界与失败后的处理：

| 边界类型 | 示例 | 不满足时的行为 |
|---|---|---|
| `required` | 缺少 `currency` | 直接报错，阻止调用，必要时回问 |
| 类型约束 | `amount: "abc"` | 校验失败，不进入执行阶段 |
| 格式约束 | `currency: "usd"` | 可选自动标准化；否则报错 |
| 范围约束 | `amount: -10` | 直接报错，视为非法输入 |
| 枚举约束 | `status` 只能是 `pending/success/failed` | 非法值拒绝执行 |

问题边界还包括输出边界。很多系统只校验输入，不校验输出，这是常见设计缺口。因为只要工具返回的数据继续喂给另一个工具，输出就已经变成下一个工具的输入。如果输出没有 Schema，错误会在后面一层才爆炸，定位成本更高。

所以更准确的定义是：工具规范要覆盖输入结构、输出结构、错误类型，以及允许的自动修复范围。超出边界就拒绝，不在边界内就不要执行。这不是保守，而是让 Agent 系统可预测。

---

## 核心机制与推导

完整的工具调用通常分成四步：选择工具、构造输入、执行工具、验证输出。每一步都要靠 Schema 和策略约束，而不是只靠模型“尽量生成对的东西”。

核心公式可以写成：

$$
调用成功 \Leftrightarrow validate(input)\ \land\ convert(types)\ \land\ validate(output)\ \land\ policy(error)
$$

### 1. 调用前：根据 metadata 选择工具

工具的 `name` 和 `description` 是给模型看的决策信息。白话解释就是：模型先看工具说明，决定该不该用、该填什么字段。这里最常见的问题不是“Schema 写错”，而是“描述写空了”。如果你只写一个 `payment_tool`，却不说明它适用于单笔支付、币种限制、是否需要 order_id，模型就会在选择阶段出错。

### 2. 构造输入：先生成，再验证，再转换

Schema 不只是“检查对不对”，也可以承担一部分“转换成对的”。例如：

- `"3"` 转成 `3`
- `"USD "` 去掉前后空格后再校验
- 缺省字段注入默认值，比如 `retry_count = 0`

这里的重点是，转换必须是显式的、可预测的。可以安全转换的才转，不安全的直接报错。比如 `"49.95"` 转数字通常是安全的，但 `"almost fifty"` 不是。

玩具例子：库存查询工具要求 `quantity` 是整数。如果模型输出 `"3"`，系统可以先转换成 `3`，再校验 `quantity >= 1`。如果网络调用返回 504，错误策略规定指数退避重试 3 次。这个流程才算闭环。

### 3. 调用后：输出也必须校验

工具返回值不是天然可信。哪怕是你自己的内部服务，也会出现字段缺失、字段重命名、状态码和 body 不一致等问题。因此 Output Schema 的作用是把“工具内部异常”拦在当前层，而不是污染整个编排链路。

例如支付工具预期输出：

```json
{"payment_id":"p_123","status":"success"}
```

如果实际返回：

```json
{"id":"p_123","ok":true}
```

这并不是“差不多能用”，而是严格意义上的协议破坏。正确做法是输出校验失败，进入错误策略，而不是让后续步骤自己猜。

### 4. 错误策略：决定是否重试、降级还是人工接管

Error Policy 的白话解释是“出错后按什么规矩收场”。它至少要区分三类错误：

| 错误类型 | 典型原因 | 推荐策略 |
|---|---|---|
| 校验错误 | 缺字段、类型不对、格式非法 | 不重试，直接回问或拒绝 |
| 瞬时错误 | 超时、429、502、504 | 指数退避重试 |
| 业务错误 | 余额不足、库存不足、权限拒绝 | 不重试，返回明确业务结果 |

真实工程例子：一个电商 Agent 需要“查库存 → 锁库存 → 创建订单 → 发起支付”。这其实是四个工具，每个工具都有输入输出 Schema。编排层依赖上一步输出的 `sku_id`、`reservation_id`、`order_id` 往下传。如果“创建订单”阶段返回 5xx，系统可以按策略重试；如果“库存不足”，则不能重试，而应直接给用户解释。这里真正保证流程稳定的不是模型，而是每一步清晰的契约和错误分流。

---

## 代码实现

下面用一个可运行的 Python 例子演示四件事：输入校验、类型转换、工具执行、错误重试。这里不依赖第三方库，目的是把机制讲清楚。工程中可以替换成 JSON Schema、Pydantic、Zod 或 Agent SDK 自带能力。

```python
import re
import time
from typing import Any, Dict

class ValidationError(Exception):
    pass

class RetryableError(Exception):
    pass

def validate_and_convert_payment(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = ["amount", "currency"]
    for key in required:
        if key not in payload:
            raise ValidationError(f"missing required field: {key}")

    amount = payload["amount"]
    if isinstance(amount, str):
        try:
            amount = float(amount.strip())
        except ValueError as exc:
            raise ValidationError("amount must be a number") from exc

    if not isinstance(amount, (int, float)):
        raise ValidationError("amount must be numeric")

    if amount < 0.01:
        raise ValidationError("amount must be >= 0.01")

    currency = payload["currency"]
    if not isinstance(currency, str):
        raise ValidationError("currency must be string")
    currency = currency.strip().upper()

    if not re.fullmatch(r"^[A-Z]{3}$", currency):
        raise ValidationError("currency must be 3 uppercase letters")

    return {"amount": float(amount), "currency": currency}

def call_payment_tool(validated: Dict[str, Any], fail_times: int = 0) -> Dict[str, Any]:
    if fail_times > 0:
        raise RetryableError("upstream timeout")
    return {
        "payment_id": "pay_001",
        "status": "success",
        "amount": validated["amount"],
        "currency": validated["currency"],
    }

def validate_output(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result.get("payment_id"), str) or not result["payment_id"]:
        raise ValidationError("invalid output: payment_id")
    if result.get("status") not in {"success", "failed"}:
        raise ValidationError("invalid output: status")
    return result

def execute_with_retry(payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    validated = validate_and_convert_payment(payload)

    for attempt in range(max_retries + 1):
        try:
            simulated_failures = 1 if attempt == 0 else 0
            result = call_payment_tool(validated, fail_times=simulated_failures)
            return validate_output(result)
        except RetryableError:
            if attempt == max_retries:
                raise
            time.sleep(0.01 * (2 ** attempt))

    raise RuntimeError("unreachable")

ok = execute_with_retry({"amount": "49.95", "currency": "usd"})
assert ok["status"] == "success"
assert ok["amount"] == 49.95
assert ok["currency"] == "USD"

try:
    execute_with_retry({"amount": "-1", "currency": "USD"})
    raise AssertionError("should fail")
except ValidationError:
    pass
```

这个例子里有几个关键点。

第一，`validate_and_convert_payment` 先做输入守门。`"49.95"` 可以转换成数字，因此允许；`"-1"` 虽然也是数字字符串，但违反金额下限，因此拒绝。

第二，`currency` 先标准化为大写，再匹配正则。这种转换是安全转换，因为不会改变语义，只是统一格式。

第三，`call_payment_tool` 之后还要做 `validate_output`。即使工具函数本身是你写的，也不能假设它永远返回符合协议的结果。

第四，重试只针对 `RetryableError`。如果是校验错误，就不重试，因为参数错了，重试一百次也不会变对。

如果用 JSON 形式表达工具定义，大致会长这样：

```json
{
  "name": "payment",
  "description": "Create a single payment in a supported currency.",
  "schema": {
    "type": "object",
    "properties": {
      "amount": { "type": "number", "minimum": 0.01 },
      "currency": { "type": "string", "pattern": "^[A-Z]{3}$" }
    },
    "required": ["amount", "currency"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "payment_id": { "type": "string" },
      "status": { "type": "string", "enum": ["success", "failed"] }
    },
    "required": ["payment_id", "status"]
  },
  "errorPolicy": {
    "retryOn": ["timeout", "429", "502", "504"],
    "maxRetries": 3,
    "backoff": "exponential"
  }
}
```

工程里可以把它抽象成统一管道：

```text
自然语言 -> 工具选择 -> 输入生成 -> 输入校验/转换 -> 工具执行 -> 输出校验 -> 编排层消费
```

这条管道的每一段都不应依赖“模型大概率会做对”，而应依赖显式规则。

---

## 工程权衡与常见坑

工具定义规范最容易踩的坑，不在复杂场景，而在“看起来能跑”的简化实现。

下面这个表列出几类高频问题：

| 常见坑 | 直接后果 | 规避策略 |
|---|---|---|
| 忘记写 `required` | 模型漏字段，调用时才崩 | 显式声明必填字段 |
| 类型定义过宽 | `"3"`、`3`、`"three"` 混进同一路径 | 仅允许可安全转换的输入 |
| 不校验输出 | 下游收到脏数据后崩溃 | 增加 `outputSchema` 再验证 |
| 没有超时与重试 | 网络抖动直接失败 | 指数退避 + 最大重试次数 |
| 错误不分层 | 业务失败和系统失败混在一起 | 区分验证错误、瞬时错误、业务错误 |
| 描述写得过短 | 模型选错工具或填错字段 | 写清用途、限制和字段语义 |

### 坑 1：缺少 `required`

如果 `amount` 没被标成必填，模型在某些上下文下就可能省略它。你会看到一个“结构上合法、业务上无意义”的 JSON。修复方式很简单：把必须存在的字段全部列进 `required`，不要依赖描述文本暗示。

### 坑 2：把宽松当鲁棒

有些系统喜欢“尽量都接”，例如任何数字字段都先 `str()` 再 `float()`，任何布尔字段都把 `"yes"`、`"ok"`、`1` 当真。短期看成功率高，长期看错误更难追踪。规范的做法是：只做可证明安全的转换，其他情况直接失败，让错误停在边界。

### 坑 3：只管输入，不管输出

这是最隐蔽也最危险的坑。输入校验只是防止你把错参数送进去，输出校验才是在防止错误结果继续扩散。尤其在多工具编排里，输出不校验，相当于把一个不可信对象当成下一步的可信输入。

### 坑 4：没有明确降级路径

不是所有失败都该自动重试。参数缺失不该重试，权限不足不该重试，库存不足也不该重试。真正需要重试的是超时、限流、短暂上游故障。超过最大重试后，应进入降级路径，比如转人工审查、记录死信队列、或返回用户可理解的失败信息。

下面是一段简单伪代码，展示“输出校验失败后退到人工审查”的流程：

```python
def run_tool_with_guard(tool, payload):
    validated_input = validate_input(tool.input_schema, payload)
    result = tool.execute(validated_input)

    try:
        validated_output = validate_output(tool.output_schema, result)
        return {"mode": "auto", "data": validated_output}
    except ValidationError as err:
        enqueue_human_review(
            tool_name=tool.name,
            payload=validated_input,
            raw_result=result,
            reason=str(err),
        )
        return {"mode": "human_review", "status": "pending"}
```

这段逻辑的关键不是“退人工”本身，而是把系统状态定义清楚。自动化失败并不等于系统崩溃，只要状态机仍然可预测。

---

## 替代方案与适用边界

不是所有工具都需要同样严格的 Schema。规范设计也有成本，因此要根据风险等级和工作流复杂度做权衡。

先给出多步工作流的表达式：

$$
Workflow\ 成功 \Leftrightarrow \forall step:\ Schema(inputs)\ \land\ Schema(outputs)
$$

意思是，一个工作流只有在每一步的输入输出都满足各自契约时，整体才算可靠。只要其中一步输出结构漂移，后面的链路就不再可信。

### 严格 Schema + 自动重试

适用场景：

- 支付、下单、退款、权限变更等高风险操作
- 多工具串联的自动化流程
- 结果会直接写数据库、调用外部 API 或触发财务动作的链路

优点是稳定、可监控、可审计。缺点是定义成本更高，前期会暴露更多“模型原本会蒙混过去”的错误，但这是好事，因为这些错误迟早会出现在生产环境。

### 简化 Schema + 人工复审

适用场景：

- 低风险信息查询
- 内容整理、摘要、标签提取
- 允许人工最终确认的半自动流程

这类方案可以减少必填字段，放宽部分类型约束，让模型更容易生成可接受输入，但必须把最后一道校验交给人工或离线审查，不适合直接执行强副作用操作。

下面是两种方案的对比：

| 方案 | 适用场景 | 优点 | 风险 |
|---|---|---|---|
| 严格 Schema + 自动重试 | 支付、订单、权限、工作流编排 | 自动化程度高，失败边界清晰 | 定义成本高，早期更“严格” |
| 简化 Schema + 人工复审 | 低风险查询、内容处理 | 接入快，对模型更宽容 | 自动化闭环弱，人工成本高 |

真实工程例子可以看 MCP 风格的 Workflow Tool。比如：

1. `create_order` 输出 `order_id`
2. `reserve_inventory` 输出 `reservation_id`
3. `create_payment` 输入前两步结果
4. `notify_user` 根据最终状态发送消息

编排层并不应该把这些步骤写成一串“自由文本推理”，而应该让每个步骤都有自己的输入输出 Schema。这样 step A 的 `order_id` 才能可靠传给 step B。若某一步返回 5xx，编排层按重试策略回退；若返回业务错误，则终止后续步骤并生成明确状态。

适用边界可以总结成一句话：风险越高、链路越长、自动化副作用越强，Schema 就越要严格；风险越低、允许人工复核，Schema 才可以适度放宽。

---

## 参考资料

| 资源 | 内容摘要 | 适用场景 |
|---|---|---|
| [Tool Augmented AI: Best Practices for Schema](https://toolaugmented.ai/best-practices-for-schema/?utm_source=openai) | 讲清工具字段、类型、严格约束与描述写法 | 完整 Schema 设计 |
| [Orbit Agent Blueprints Schema](https://csiorbit.com/docs/agent-services/agent-blueprints/schema?utm_source=openai) | 从 Agent Blueprint 角度说明 Schema 契约 | Agent 平台接入 |
| [Compozy Tools Schema Docs](https://compozy.com/docs/schema/tools?utm_source=openai) | 说明输入输出 Schema 与执行约束 | 工具调用机制 |
| [Superjson: OpenAI Agents SDK JSON Tool Schemas Guide](https://superjson.ai/blog/2025-09-01-openai-agents-sdk-json-tool-schemas-guide/?utm_source=openai) | 解释 JSON Tool Schema、类型转换与 SDK 落地 | 工程实现参考 |
| [Tray.ai Workflow Tools](https://docs.tray.ai/platform/artificial-intelligence/agent-gateway/workflow-tools?utm_source=openai) | 多步工作流工具封装与编排思路 | Workflow Tool 场景 |
| [Arun Baby: Tool Design Principles](https://arunbaby.com/ai-agents/0026-tool-design-principles/?utm_source=openai) | 常见设计坑、错误处理与工具边界 | 工程权衡与避坑 |

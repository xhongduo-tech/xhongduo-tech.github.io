## 核心结论

工具描述不是装饰文本，而是智能体调用工具时的提示契约。这里的“提示契约”，可以直接理解为模型与工具之间的操作协议：模型依赖这份文本判断这个工具是什么、什么时候该用、参数应该怎么填、哪些情况不能用。只要系统里存在“多工具选择”和“结构化参数填写”，工具描述就不再是可有可无的说明文，而是决定调用成功率的控制面。

一套可用的工具契约，至少应覆盖三层文本：`name`、`description`、`parameter description`。在工程实践里，如果再补上 `when to use` 和 `parameter example`，效果通常会明显更稳定。原因很直接。工具调用往往包含两个连续判断：

1. 先判断“这一步要不要调用工具，以及该调用哪个工具”
2. 再判断“参数应该如何组织，才能形成合法请求”

第一步主要受 `name`、`description`、`when to use` 影响；第二步主要受 `parameter description`、参数示例、Schema 约束影响。任意一步失误，整次调用就会失败。于是，工具描述优化的本质不是“把文案写得更漂亮”，而是“降低系统在决策链上的累计误差”。

一个足够实用、也最容易落地的结论是：

- `when to use` 应该写成排他条件，而不是泛泛的主题描述
- `parameter example` 应该写成可以直接模仿的合法样例，而不是抽象提示

前者的作用是缩小工具选择空间，减少误调用；后者的作用是给模型提供一份参数模板，减少格式错误、字段遗漏和字段幻觉。很多团队在生产环境里观察到的规律都很接近这一点：当描述中补充清晰的使用条件时，误调用会出现显著的相对下降；当参数中补充最小合法示例时，参数格式错误会继续下降。若任务本身是多步工具链，这些改进不是停留在单点，而会通过连乘机制被放大成整体收益。

下表把不同字段的职责拆开看会更清楚：

| 层级/字段 | 解决的核心问题 | 模型主要学到什么 | 典型风险 |
|---|---|---|---|
| `name` | 这是什么工具 | 工具的大类身份与唯一职责 | 命名过泛，和别的工具语义重叠 |
| `description` | 这个工具做什么 | 功能范围与正向职责 | 写得过宽导致误用，写得过短导致理解不足 |
| `parameter description` | 参数怎么填 | 字段含义、类型、约束、是否必填 | 漏字段、错类型、格式不合法 |
| `when to use` | 什么时候才该调用 | 使用边界与触发条件 | 不写会误调，写得不排他会冲突 |
| `parameter example` | 合法输入长什么样 | 可直接模仿的输入模板 | 示例缺失时模型只能猜 |

可以直接看一个玩具例子。假设有个工具叫 `lookup_order`。如果你只写一句“查询订单”，模型很可能把“退款进度”“投诉记录”“售后状态”“物流位置”都当成它可能负责的事情。但如果改写成：

- `description`：只查询订单当前状态，包括待发货、运输中、已签收，不处理退款、投诉、售后
- `when to use`：仅当用户询问发货进度、物流位置、是否签收时使用
- `parameter example`：`{"order_id": "ORD-20260224-1234"}`

那么模型面前的决策空间会立即收缩。它不仅更容易选对工具，也更容易生成一个可以直接提交给后端的参数对象。

---

## 问题定义与边界

本文讨论的是依赖工具链的智能体，而不是只输出自然语言的普通文本生成模型。这里的“工具链”，可以白话理解为：模型不是只靠生成文本回答问题，而是必须在合适的时机调用函数、API、数据库查询、检索接口或业务服务。只要系统里存在“从多个工具中选择一个”与“向工具填写结构化参数”这两个步骤，工具描述就会直接影响成功率。

从故障现象看，常见错误大致可以分成三类：

| 错误类型 | 白话解释 | 常见表现 | 主要缺陷层级 |
|---|---|---|---|
| Wrong Tool | 选错工具 | 问物流却调用退款工具，问库存却调用订单工具 | `name` / `description` / `when to use` |
| Missing Tool | 该用工具却没用 | 明明要查数据库，却直接编答案 | `description` / `when to use` |
| Hallucination | 编造结果或参数 | 随机补不存在字段、伪造返回值、乱填格式 | `parameter description` / 示例 / 校验 |

这三类错误里，前两类主要发生在“要不要调用”和“调用哪个”的阶段，第三类主要发生在“参数怎么填”和“结果怎么解释”的阶段。它们经常连锁出现：选错工具之后，参数自然也更容易错；没有明确示例时，模型就会开始猜测字段名和格式；一旦后端不做校验，错误就会穿透到业务层。

边界也需要明确，不然讨论会变形。

第一，本文关注的是提示层的契约设计，即如何把工具元数据写成更容易被模型正确消费的文本结构。它不试图展开模型微调、强化学习、检索排序、候选工具召回等更底层的问题。

第二，本文说的是“工具调用场景”的优化，不是所有 LLM 任务都会等比例受益。纯写作、摘要、开放问答并不一定需要这些字段；但凡进入函数调用、工作流编排、Agent 路由，收益就会迅速增大。

第三，“描述越详细越好”是错误直觉。描述过短，模型看不出边界；描述过长，关键信息会被淹没。真正有用的不是字数，而是信息密度和结构质量。模型需要的是高区分度信号，而不是更多噪声。

看一个最小对比就很容易明白。

版本 A 只有一句：

```text
description: 查询订单
```

版本 B 则补全了边界和示例：

```text
description: 只查询订单当前状态，包括待发货、运输中、已签收，不处理退款、投诉、售后。
when_to_use: 当用户询问物流、发货进度、是否签收时调用。
parameter example: {"order_id": "ORD-20260224-1234"}
```

如果用户问的是“我的包裹到哪了？”，版本 A 可能出现几种结果：

- 调用 `get_order_detail`，因为“订单详情”与“订单状态”语义相近
- 调用 `refund_status`，因为“订单问题”也可能触发售后链路
- 不调用任何工具，直接用语言模型猜一个答案

而版本 B 更容易把意图稳定压到 `lookup_order` 上。关键不在于它更长，而在于它补上了两个原本缺失的控制信号：

- 范围：它明确告诉模型“这个工具只处理物流状态，不处理别的问题”
- 触发条件：它明确告诉模型“只有在这些具体问题出现时才调用”

对新手更重要的一点是：工具描述的目标不是“把功能介绍完整”，而是“帮助模型在分叉路口做出正确选择”。把它理解成 API 文档往往不够，把它理解成决策路标才更接近真实用途。

---

## 核心机制与推导

工具调用流程本质上是多步决策链。每一步都可能成功，也可能失败。只要系统不是一步完成，而是要连续经过“是否调用”“选哪个工具”“抽取实体”“填参数”“解释结果”等环节，最终成功率就不是看某一个步骤有多强，而是看每一步能不能稳住。

先定义最简单的形式。假设总共有 $N$ 步，每一步成功率为 $s_i$，错误率为 $e_i$，则：

$$
s_i = 1 - e_i
$$

整个工作流的总成功率为：

$$
S = \prod_{i=1}^{N} s_i = \prod_{i=1}^{N}(1-e_i)
$$

这个公式有两个直接含义。

第一，多步系统是连乘，不是加法。某一步如果明显偏弱，会把整条链拖得很低。

第二，小幅优化单步准确率，往往会在整体指标上放大。因为你优化的不是某一个输出，而是整个链条中的一个乘数。

对新手来说，可以先用一个日常化理解：如果一个流程要连续通过 5 个关卡，每个关卡都只有 90% 概率通过，那总通过率不是 90%，而是：

$$
0.9^5 \approx 0.59
$$

也就是 59% 左右。单步看起来已经“不差”，串起来却明显不够稳。这正是很多 Agent 系统“单点测试都还行，上线后整体表现却差很多”的原因。

下面看一个 5 步玩具例子。假设某客服智能体需要完成以下流程：

1. 判断当前问题是否需要调用工具
2. 从多个工具中选出正确工具
3. 从用户话语中抽取订单号
4. 按正确格式组装参数
5. 根据工具返回结果生成最终回答

优化前，各步成功率假设如下：

| 步骤 | 成功率 |
|---|---:|
| 是否调用工具 | 0.92 |
| 选择正确工具 | 0.80 |
| 抽取订单号 | 0.90 |
| 组装参数 | 0.89 |
| 解释结果 | 0.95 |

则总成功率为：

$$
0.92 \times 0.80 \times 0.90 \times 0.89 \times 0.95 \approx 0.56
$$

也就是说，大约只有 56% 的任务能完整、正确地走完全链路。

现在做两项改进：

- 给工具补上明确的 `when to use`，把“选择正确工具”从 0.80 提高到 0.88
- 给参数补上合法样例，把“组装参数”从 0.89 提高到 0.94

其余步骤不变，则：

$$
0.92 \times 0.88 \times 0.90 \times 0.94 \times 0.95 \approx 0.65
$$

总成功率上升到约 65%。从单步看，0.80 到 0.88、0.89 到 0.94 都不算夸张；但放到连乘系统里，最终提升已经很明显。

再看一个更贴近工程复盘的表示方式：

| 场景 | Step1 | Step2 | Step3 | Step4 | Step5 | 整体成功率 |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 0.93 | 0.84 | 0.86 | 0.90 | 0.91 | 0.55 |
| 加 `when to use` | 0.93 | 0.90 | 0.86 | 0.90 | 0.91 | 0.59 |
| 再加参数示例 | 0.93 | 0.90 | 0.86 | 0.96 | 0.91 | 0.63 |

如果继续把工具重名、边界冲突、参数错误提示不清晰等问题一起收敛，整体成功率继续上探是合理的。这里的关键不是“某个字段神奇地提升了模型能力”，而是：工具描述通过提供更强的约束和更清晰的模板，减少了模型在中间步骤中的分叉和猜测。

这个机制可以进一步写成误差传播的形式。设第 $i$ 步错误率为 $e_i$，则整体失败率为：

$$
1 - S = 1 - \prod_{i=1}^{N}(1-e_i)
$$

当若干个步骤都有中等错误率时，整体失败率会上升得很快。于是，一份好的工具契约的价值，不是把某一步做到 100%，而是把最容易失控的几个步骤压回到可控范围。

把这个抽象公式落到真实场景里会更容易理解。假设客服系统里有四个工具：

- `lookup_order`
- `refund_status`
- `create_ticket`
- `query_inventory`

用户问题是“订单什么时候到？”这句话与订单、物流、售后都存在一定语义邻近。如果 `lookup_order` 的描述只有“查询订单”，那么模型看到的是一个模糊类别；它无法知道这是“订单状态查询”，还是“订单详情查询”，还是“订单相关问题入口”。

但如果改成：

- `description`：只查询订单状态，不处理退款、售后、投诉
- `when to use`：仅在用户询问物流、发货、签收状态时调用
- `parameter example`：`{"order_id":"ORD-20260224-1234"}`

则模型的决策空间会同时从两端收缩：

- 在工具选择阶段，它更少考虑不相关工具
- 在参数填写阶段，它更少猜测字段结构

因此，错误率下降并不神秘。模型并不是突然“理解更深了”，而是系统向它提供了更明确的选择边界与模仿模板。

---

## 代码实现

工程上更可维护的做法，是把工具元数据显式建模，而不是把所有信息混成一段自由文本。这样做有三个好处：

1. 可以把同一份元数据同时用于模型提示和运行时校验
2. 可以单独更新某一层信息，例如只改 `when to use` 而不影响参数定义
3. 可以在多工具场景里做一致性检查，例如检测不同工具的边界是否重叠

下面先给一个 `lookup_order` 的 JSON 定义：

```json
{
  "name": "lookup_order",
  "description": "只查询订单当前状态，包括待发货、运输中、已签收。不处理退款、投诉、售后。",
  "whenToUse": "当用户询问物流、发货进度、是否签收时调用。若问题是退款、售后、投诉，禁止调用。",
  "parameterDescriptions": {
    "order_id": {
      "type": "string",
      "description": "订单编号，格式为 ORD-YYYYMMDD-XXXX。",
      "required": true
    }
  },
  "parameterExample": {
    "order_id": "ORD-20260224-1234"
  }
}
```

这份结构里最关键的不是字段名，而是它把五类信息拆开了：

- 身份：`name`
- 职责：`description`
- 触发条件：`whenToUse`
- 参数规则：`parameterDescriptions`
- 合法样例：`parameterExample`

一旦拆开，运行时就能做更多事情。例如：

- 把 `description` 和 `whenToUse` 拼进工具提示
- 把 `parameterDescriptions` 编译成 JSON Schema 或校验函数
- 把 `parameterExample` 用作 few-shot 样板
- 在日志里单独统计“误选率”和“参数错误率”

下面给出一个可运行的 Python 示例。它实现四件事：

1. 定义工具元数据
2. 判断当前用户问题是否适合调用 `lookup_order`
3. 校验 `order_id` 是否存在且格式合法
4. 模拟一次工具调用与结果解释

```python
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


ORDER_ID_RE = re.compile(r"^ORD-\d{8}-\d{4}$")


@dataclass
class ToolSpec:
    name: str
    description: str
    when_to_use: str
    parameter_descriptions: Dict[str, Dict[str, Any]]
    parameter_example: Dict[str, Any]


LOOKUP_ORDER = ToolSpec(
    name="lookup_order",
    description="只查询订单当前状态，包括待发货、运输中、已签收。不处理退款、投诉、售后。",
    when_to_use="当用户询问物流、发货进度、是否签收时调用。若问题是退款、售后、投诉，禁止调用。",
    parameter_descriptions={
        "order_id": {
            "type": "string",
            "description": "订单编号，格式为 ORD-YYYYMMDD-XXXX",
            "required": True,
        }
    },
    parameter_example={"order_id": "ORD-20260224-1234"},
)


def detect_intent(user_query: str) -> str:
    shipping_keywords = ["物流", "发货", "到哪", "签收", "快递", "运输", "配送"]
    refund_keywords = ["退款", "退货", "投诉", "售后"]
    ticket_keywords = ["人工", "客服", "工单", "申诉"]

    if any(k in user_query for k in shipping_keywords):
        return "shipping_status"
    if any(k in user_query for k in refund_keywords):
        return "refund_or_support"
    if any(k in user_query for k in ticket_keywords):
        return "ticket_or_agent"
    return "unknown"


def should_call_lookup_order(user_query: str) -> bool:
    return detect_intent(user_query) == "shipping_status"


def validate_lookup_order_params(params: Dict[str, Any]) -> Dict[str, str]:
    if "order_id" not in params:
        raise ValueError(
            "缺少字段 order_id。合法示例: {'order_id': 'ORD-20260224-1234'}"
        )

    order_id = params["order_id"]

    if not isinstance(order_id, str):
        raise TypeError("order_id 必须是字符串")

    if not ORDER_ID_RE.fullmatch(order_id):
        raise ValueError("order_id 格式错误，应类似 ORD-20260224-1234")

    return {"order_id": order_id}


def call_lookup_order(order_id: str) -> Dict[str, str]:
    # 这里用静态数据模拟后端返回，真实场景下可替换为数据库/API 调用
    mock_db = {
        "ORD-20260224-1234": {
            "status": "运输中",
            "carrier": "SF Express",
            "last_event": "2026-02-25 14:30 已离开发货城市",
        },
        "ORD-20260226-5678": {
            "status": "已签收",
            "carrier": "YTO",
            "last_event": "2026-02-27 18:05 已由前台代收",
        },
    }

    if order_id not in mock_db:
        raise LookupError(f"未找到订单: {order_id}")

    return mock_db[order_id]


def explain_lookup_result(result: Dict[str, str]) -> str:
    return (
        f"订单当前状态：{result['status']}；"
        f"承运方：{result['carrier']}；"
        f"最新轨迹：{result['last_event']}。"
    )


def handle_user_query(user_query: str, params: Optional[Dict[str, Any]] = None) -> str:
    if not should_call_lookup_order(user_query):
        return "当前问题不适合调用 lookup_order，应考虑其他工具或继续澄清意图。"

    if params is None:
        return (
            "可以调用 lookup_order，但缺少参数。"
            "请提供类似 {'order_id': 'ORD-20260224-1234'} 的输入。"
        )

    validated = validate_lookup_order_params(params)
    result = call_lookup_order(validated["order_id"])
    return explain_lookup_result(result)


# 合法路径
assert should_call_lookup_order("我的订单物流到哪了？") is True
assert validate_lookup_order_params(
    {"order_id": "ORD-20260224-1234"}
) == {"order_id": "ORD-20260224-1234"}

response = handle_user_query(
    "我的订单物流到哪了？",
    {"order_id": "ORD-20260224-1234"}
)
assert "订单当前状态：运输中" in response

# 非法路径 1：意图不匹配
assert should_call_lookup_order("我要申请退款") is False

# 非法路径 2：参数格式错误
try:
    validate_lookup_order_params({"order_id": "1234"})
    raise AssertionError("这里应该抛出异常")
except ValueError as e:
    assert "格式错误" in str(e)

# 非法路径 3：缺少字段
try:
    validate_lookup_order_params({})
    raise AssertionError("这里应该抛出异常")
except ValueError as e:
    assert "缺少字段 order_id" in str(e)

# 非法路径 4：订单不存在
try:
    handle_user_query("我的快递到哪了？", {"order_id": "ORD-20260101-0001"})
    raise AssertionError("这里应该抛出异常")
except LookupError as e:
    assert "未找到订单" in str(e)
```

这个示例比单纯的“校验一个正则”多做了一步：它把“提示契约”和“运行时护栏”放到了同一个对象模型里。这正是工程实现里最重要的点。工具描述负责把模型引向正确轨道，运行时校验负责在模型偏离轨道时拦截错误。两者不是替代关系，而是前后两层防线。

如果再进一步，可以把这套结构接入自动化提示拼装。例如：

```python
def render_tool_prompt(spec: ToolSpec) -> str:
    lines = [
        f"Tool name: {spec.name}",
        f"Description: {spec.description}",
        f"When to use: {spec.when_to_use}",
        "Parameters:",
    ]

    for field, meta in spec.parameter_descriptions.items():
        required = "required" if meta.get("required") else "optional"
        lines.append(
            f"- {field} ({meta['type']}, {required}): {meta['description']}"
        )

    lines.append(f"Example: {spec.parameter_example}")
    return "\n".join(lines)


prompt_text = render_tool_prompt(LOOKUP_ORDER)
assert "Tool name: lookup_order" in prompt_text
assert "Example: {'order_id': 'ORD-20260224-1234'}" in prompt_text
```

这样做的意义在于：提示内容不再由人工到处复制粘贴，而是从同一份工具定义自动生成。它可以减少文档和实现不一致的问题，例如示例已经更新了，但系统提示里还是旧格式。

对新手来说，这里有一个很重要的工程习惯：示例一定要“能跑”。很多团队会写出这种看起来合理、其实无效的例子：

```json
{"order_id": "<你的订单号>"}
```

这对人类读者可能没问题，但对模型不够好，因为它并没有展示真实的合法模式。更好的写法是：

```json
{"order_id": "ORD-20260224-1234"}
```

它直接体现了前缀、日期段和流水号段。模型更容易从中学到格式，而不是只知道“这里应该填点什么”。

如果要进一步提升鲁棒性，可以把错误回传设计成可用于重试的提示，而不仅仅是程序报错。例如：

- 第一次失败返回：`缺少字段 order_id`
- 第二次失败返回：`order_id 格式错误，应类似 ORD-20260224-1234`
- 第三次失败返回：`当前问题不属于物流状态查询，请不要调用 lookup_order`

这种错误信息实际上是在给模型提供“下一次应如何修正”的最短路径。很多参数重试机制的提升，靠的不是更复杂的推理，而是更具体的纠错反馈。

---

## 工程权衡与常见坑

最常见的误区，是把“详细”误认为“有效”。真实情况通常更接近下面这句话：越结构化越安全，越冗长越容易失焦。

当描述太短时，模型无法区分工具边界；当描述太长时，关键边界又会被背景材料、例外情况、FAQ 和叙述性文字稀释。尤其在多工具场景里，如果每个工具都写成大段说明，模型看到的并不是“更多信息”，而是“一堆互相竞争的信息”。

很多团队在复盘时都会看到类似现象：同一个工具，极短描述导致误选；补充职责、边界、参数示例后准确率上升；继续往里塞背景知识、业务规则、长篇说明后，准确率又开始下降。趋势通常像一个倒 U 型，而不是一路上升。

可以用伪数据表示这个关系：

| 描述长度（token） | 准确率 |
|---:|---:|
| 80 | 0.87 |
| 150 | 0.95 |
| 300 | 0.94 |
| 500 | 0.91 |
| 1200 | 0.86 |
| 2000 | 0.80 |

如果把它画成折线图，典型形状是先升后降。最佳区间通常不是最短，也不是最长，而是“足以定义职责和边界，但还没有开始啰嗦”的范围。很多时候，约 100 到 250 token 的高密度描述，比 1000 token 的大段说明更有效。

从机制上看，原因并不复杂：

- 模型的注意力预算有限，工具描述之间会竞争
- 越长的文本越容易引入非关键语义
- 触发条件若淹没在长段落里，模型很难在选择时抓住它
- 多工具总长度膨胀后，还会挤压用户上下文和系统规则的空间

因此，优化方向不应是“继续加信息”，而应是“保留最有判别力的信息”。通常最值得保留的是：

- 这个工具做什么
- 这个工具不做什么
- 什么时候调用
- 参数怎么填
- 合法输入长什么样

常见坑可以归纳如下：

| 问题 | 原因 | 对策 |
|---|---|---|
| 工具名过泛 | 多个工具语义重叠，名称不能体现唯一职责 | 名称尽量体现唯一行为，如 `lookup_order_status` |
| 描述只写功能不写边界 | 模型知道能做什么，不知道哪些情况不能做 | 在 `description` 或 `when to use` 中补排他条件 |
| 参数说明没有示例 | 模型知道字段名，但不知道格式模板 | 给出最小合法 JSON 示例 |
| 多工具都写“订单相关问题时调用” | 条件重叠，工具相互竞争 | 让条件互补，明确切分物流、退款、投诉 |
| 只靠提示不做校验 | 模型一旦漂移，错误直接到后端 | 用 Schema、类型检查、正则做二次拦截 |
| 描述过长 | 注意力稀释、上下文膨胀 | 压缩到“职责 + 边界 + 示例” |

还有一个特别容易被忽略的坑，是 `when to use` 写成开放句式，而不是排他句式。比如：

```text
当用户询问订单相关问题时调用
```

这句话看起来像条件，实际上没有缩小任何决策空间。因为“订单相关问题”太宽，退款、物流、售后、投诉都可以被算进去。

更好的写法是：

```text
仅当用户询问物流、发货进度、签收状态时调用；退款、售后、投诉一律不要调用。
```

两者的差别在于：

- 前者是在描述主题
- 后者是在切分边界

主题描述帮助模型“模糊联想”，边界描述帮助模型“排除错误分支”。在工具路由问题上，后者通常更重要。

另一个常见问题是工具之间的职责切分不对称。比如：

- `lookup_order` 写得很宽
- `refund_status` 写得很窄
- `create_ticket` 又写得像兜底入口

这样即使单个工具文本都没错，整体工具池仍会出现竞争。解决方法不是只改某一个工具，而是把一组相关工具放在一起做边界检查。一个简单原则是：如果两个工具的 `when to use` 可以同时命中同一个用户问题，那它们就很可能还没有切干净。

---

## 替代方案与适用边界

如果你暂时无法修改现有工具元数据，也并不意味着完全无解。一个常见替代方案，是在 dispatcher 层增加 pre-filter。这里的“dispatcher”可以直接理解为路由器：它先判断问题属于哪一类，再决定把哪些工具暴露给模型。

最简单的流程如下：

1. 先把用户问题粗分类为 `shipping`、`refund`、`complaint`
2. 如果类别是 `shipping`，只向模型暴露 `lookup_order`
3. 如果类别是 `refund`，只向模型暴露 `refund_status`
4. 最后让模型只在缩小后的工具集合里填参数

这个方案的优势很明显：它可以先从系统层面裁掉一批不相关工具，从而降低误选概率。即使工具描述本身写得一般，模型也至少不会在一个过大的候选池里乱选。

下面给一个极简示例：

```python
from typing import List


def route_candidate_tools(user_query: str) -> List[str]:
    intent = detect_intent(user_query)

    if intent == "shipping_status":
        return ["lookup_order"]
    if intent == "refund_or_support":
        return ["refund_status", "create_ticket"]
    if intent == "ticket_or_agent":
        return ["create_ticket"]
    return []


assert route_candidate_tools("我的订单到哪了？") == ["lookup_order"]
assert route_candidate_tools("我要退款") == ["refund_status", "create_ticket"]
```

但它只是补救，不是根治。原因主要有两个。

第一，pre-filter 本身也是一个分类器。只要它会犯错，就仍可能把模型送进错误分支。例如用户说“包裹一直没到，我要投诉”，这句话同时包含物流和投诉信号，粗分类本身就可能摇摆。

第二，pre-filter 主要解决的是“选哪个工具”，但对“参数怎么填”帮助有限。即使候选工具只剩一个，模型仍然可能：

- 漏掉必填字段
- 把字符串写成对象
- 拼错日期格式
- 额外编造不存在的字段

因此，只要工具涉及强结构参数，最终仍需要参数示例与运行时校验。

下表可以概括两种路径的差异：

| 方案 | 开发成本 | 准确率提升 | 扩展性 | 主要局限 |
|---|---:|---:|---:|---|
| 完整提示契约 | 中等 | 高 | 高 | 需要维护工具元数据 |
| 外部规则 pre-filter | 低到中 | 中等 | 中等 | 主要缓解误选，难解决参数错误 |

适用边界也可以更明确一些：

| 场景 | 推荐做法 |
|---|---|
| 多工具、高频调用、参数结构强 | 完整提示契约 + Schema/运行时校验 |
| 工具很少、调用低频 | 可简化字段，但仍建议保留边界和示例 |
| 无法改工具定义 | 先用 dispatcher pre-filter 作为补救 |
| 参数是 JSON、SQL、DSL 等强格式输入 | 必须给出示例，并做严格校验 |
| 高风险业务，如下单、扣费、权限变更 | 除提示契约外，还要增加确认与审计机制 |

还需要补一句边界条件：如果任务本身是纯检索、纯摘要，或者只有一个工具且参数极简单，例如 `get_current_time({"timezone": "Asia/Shanghai"})`，那么 `when to use` 与复杂示例的收益不会像多工具客服系统那样大。这不是因为它们没用，而是因为问题本身的分叉更少、出错面更窄。

所以，替代方案确实存在，但大多属于系统外补洞。只要你有权设计工具元数据，最稳妥的做法仍然是把 `name`、`description`、`parameter description`、`when to use`、`parameter example` 视为一个整体来设计。工具越多、工作流越长、参数越强结构，这个整体设计就越重要。

---

## 参考资料

1. Thread Transfer, *Tool Use in AI Agents: Production Best Practices*  
说明：给出生产环境中工具误调用率下降的案例，可用于理解“明确工具边界”对路由准确率的影响。阅读时应重点关注它如何定义“误调用”、如何统计成功率，以及是否区分了工具选择错误与参数填写错误。

2. Playbooks, *Tool Design*  
说明：强调 `name`、`description` 和参数说明共同构成工具契约，并建议补充 `when to use` 与参数示例。对本文最有帮助的部分，不是具体措辞模板，而是它把工具定义看成“行为约束”而不只是“接口注释”。

3. Anthropic Developer Documentation / 相关文章，*Advanced Tool Use*  
说明：这一类资料通常展示参数示例、工具模式和结构化调用设计对稳定性的帮助。阅读时建议重点看示例是如何约束输出格式的，而不只是看模型能力展示。

4. 关于 Prompt 长度与质量关系的讨论资料，如 VAHU / David Veksler 相关文章  
说明：可用于理解“上下文不是越长越好”。真正需要关注的是：额外长度提供的是高判别信息，还是只是在堆叠背景材料。对工具描述来说，长度的收益通常在一定区间后开始递减。

5. 关于多步工作流成功率分析的工程文章或实践复盘  
说明：这些材料通常会从流程指标而不是单步指标出发，解释为什么一个 5 步系统里每一步只错一点，整体就会明显下滑。本文中的
$$
S=\prod_{i=1}^{N}s_i
$$
可以作为阅读这类材料时的统一分析框架。

建议按需回到原文确认具体实验设置、模型版本、任务口径和数据定义。不同平台对“成功率”“参数正确率”“误调用率”的定义往往并不一致。有的平台把“调用了正确工具但参数有误”算作失败，有的平台会单独拆分统计。若不先统一口径，同一组数字很容易被误读。

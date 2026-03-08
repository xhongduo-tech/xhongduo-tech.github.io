## 核心结论

长时间工具调用如果只在最后一次性返回结果，用户感知通常不是“系统正在努力”，而是“页面没反应”。  
解决方式不是单纯加一个转圈，而是把一次工具调用拆成三个可见阶段：

| 阶段 | 用户看到什么 | Anthropic 侧常见事件 | OpenAI 侧常见事件 |
| --- | --- | --- | --- |
| 调用发起 | 将要使用哪个工具、已知参数有哪些 | `message_start`、`content_block_start`、`content_block_delta` 中出现 `input_json_delta` | `response.output_item.added` |
| 执行中 | 参数还在补全、系统仍在处理 | `content_block_delta` | `response.function_call_arguments.delta` |
| 结果返回 | 参数定稿、工具执行完成、结果摘要 | `content_block_stop`、`message_stop`，随后通常进入工具执行与结果回传 | `response.function_call_arguments.done`、`response.output_item.done` |

这里的“流式响应”是指服务端把事件分片持续推给前端，而不是等完整 JSON 或完整结果生成后再一次性返回。  
对用户体验最有价值的收益不是“真实耗时更短”，而是“等待过程持续可见、可解释、可确认”。

对于数据库查询超过 5 秒、外部 API 超过 10 秒、聚合多个供应商接口的场景，推荐使用三卡片视图：

1. 参数卡片：展示即将调用的工具名、已知参数和参数是否仍在补全。
2. 进度卡片：展示“已收到多少片段”“当前处于参数生成还是工具执行阶段”。
3. 结果卡片：先展示结果摘要，再按需展开原始返回，避免一开始把大段 JSON 直接砸给用户。

新手版玩具例子：查询航班时，不要让页面空白等待 12 秒，而是依次显示：

1. “将查询：北京 -> 旧金山，日期 2026-04-02”
2. “正在获取价格与舱位，已联系 3 个供应商”
3. “找到 8 个航班，最低价 4120 元”

这三句话没有减少后端 12 秒的真实耗时，但显著降低了用户反复点击、刷新页面和怀疑系统失效的概率。

---

## 问题定义与边界

“工具调用”是指模型不直接输出自然语言答案，而是先生成一个结构化请求，再由外部程序去查数据库、调 API、执行检索、读文件或运行某段业务逻辑。  
“中间状态展示”是指在工具真正返回前，把这段等待过程可视化，而不是让用户只看到一个静止页面。

这个问题的本质不是前端动画是否好看，而是事件模型是否足够细。只有后端能持续收到增量事件，前端才知道系统现在到底处于：

1. 正在决定调用哪个工具
2. 正在补全工具参数
3. 参数已经完成，后端正在执行工具
4. 工具结果已经返回，前端正在渲染摘要

Anthropic 和 OpenAI 的共同点是：都支持把工具参数以增量形式流出来。  
差异主要在事件命名、对象层级和聚合方式。

| 维度 | Anthropic | OpenAI |
| --- | --- | --- |
| 流协议感知 | 以 `message_start` 到 `message_stop` 为一条消息流 | 以 `response.*` 为事件流 |
| 工具参数增量 | `content_block_delta` 中的 `input_json_delta.partial_json` | `response.function_call_arguments.delta` |
| 完成信号 | `content_block_stop` 表示该内容块结束，随后可将累计片段解析为完整输入 | `response.function_call_arguments.done` 给出完整参数 |
| 多调用区分 | 常按 `content` 的 `index` 区分内容块 | 常按 `output_index` 或 `item_id` 聚合同一调用 |
| 风险点 | 可能收到不完整或无效 JSON 片段，尤其在细粒度工具流中更明显 | 一个响应里可能有多个函数调用，若不分组会串流混淆 |

边界也要提前讲清楚，否则设计会走偏：

- 如果工具调用本身只要 1 到 2 秒，三阶段 UI 往往是过度设计。
- 如果业务不希望暴露参数细节，可以只展示“正在查询”和最终结果，不展示具体参数。
- 如果接入层不支持 SSE 或 WebSocket，只能退回任务轮询，粒度会明显更粗。
- 如果参数包含敏感字段，例如邮箱、手机号、内部检索语句、数据库主键，不应原样展示，应做脱敏或摘要化。
- 如果用户只关心“有没有结果”，而不关心“系统正在查什么”，则不必强行做参数级可视化。

因此，这类设计不是“所有工具调用都要三阶段”，而是“长耗时、强交互、用户容易焦虑的调用才值得三阶段展示”。

---

## 核心机制与推导

核心机制可以写成一个直接的公式：

$$
A_{\text{final}} = \bigoplus_{i=1}^{n} \Delta_i
$$

其中：

- $A_{\text{final}}$ 表示最终完整参数字符串
- $\Delta_i$ 表示第 $i$ 个增量片段
- $\bigoplus$ 表示按接收顺序拼接

如果最终要解析成 JSON 对象，那么过程通常是：

$$
\text{args\_obj} = \operatorname{parse}\left(\bigoplus_{i=1}^{n} \Delta_i\right)
$$

这两个公式表达的意思很朴素：  
工具参数往往不是“一次生成完成”，而是“先来一段，再来一段，最后拼起来才是完整 JSON”。

对于前端来说，还需要一个更接近工程实现的状态机：

$$
\text{UI State} \in \{\text{准备调用},\ \text{参数生成中},\ \text{工具执行中},\ \text{结果已返回},\ \text{失败}\}
$$

也就是说，前端真正要维护的不是“有没有返回结果”，而是“系统当前在等待链路的哪个位置”。

ASCII 流程图如下：

```text
用户提问
  |
  v
模型决定是否调用工具
  |
  +--> 阶段1：发起
  |     创建调用卡片
  |     显示工具名、已知参数、调用ID
  |
  +--> 阶段2：参数流入
  |     持续接收 delta
  |     拼接参数片段
  |     更新参数快照和片段计数
  |
  +--> 阶段3：参数完成
  |     解析 JSON
  |     标记“参数已定稿”
  |
  +--> 阶段4：工具执行
  |     调用数据库/API/检索系统
  |     更新“执行中”状态
  |
  +--> 阶段5：结果返回
        展示结果摘要
        允许展开完整结果
```

这里有一个经常被忽略的细节：  
“参数流结束”不等于“工具执行结束”。

很多文章会把这两步揉在一起，但在真实系统里，它们是不同阶段：

| 阶段 | 谁在工作 | 前端该显示什么 |
| --- | --- | --- |
| 参数生成中 | 模型 | 已知参数、片段计数、参数还未定稿 |
| 工具执行中 | 后端业务系统 | 正在查询数据库 / 正在请求外部 API |
| 结果整理中 | 后端或前端 | 正在生成结果摘要、准备展示 |

玩具例子：目标参数是 `{"location":"San Francisco"}`。  
OpenAI 风格事件链可以理解为：

1. `response.output_item.added`
2. `response.function_call_arguments.delta` 传来 `{"location":"San`
3. `response.function_call_arguments.delta` 传来 ` Francisco`
4. `response.function_call_arguments.delta` 传来 `"}`
5. `response.function_call_arguments.done` 给出完整参数

这 5 步在 UI 上不是“等待同一个完成事件”，而是应当逐步投射成：

| 事件 | 参数卡片 | 进度卡片 | 结果卡片 |
| --- | --- | --- | --- |
| `response.output_item.added` | 将调用 `get_weather` | 已创建调用 | 空 |
| 第 1 个 delta | 已识别到 `location` 字段 | 已收 1 个片段 | 空 |
| 第 2 个 delta | 参数值继续补全 | 已收 2 个片段 | 空 |
| 第 3 个 delta | 参数接近完整 | 已收 3 个片段 | 空 |
| `response.function_call_arguments.done` | `{"location":"San Francisco"}` | 参数已完成，准备执行工具 | 可进入结果摘要阶段 |

Anthropic 的思路相同，只是参数增量通常出现在 `content_block_delta` 事件内部，具体字段是 `delta.type = input_json_delta` 和 `delta.partial_json`。

真实工程例子：订票智能体调用外部航班 API，总耗时 12 秒。  
如果没有中间状态，用户看到的是“提交后一直空白”，主观感受接近故障。  
如果有分阶段展示，时间线可以变成：

| 时间点 | 可见状态 | 用户理解 |
| --- | --- | --- |
| 第 1 秒 | 准备查询航班，北京 -> 旧金山，日期 2026-04-02 | 系统理解了我的意图 |
| 第 3 秒 | 正在请求供应商 A / B / C | 系统确实开始工作了 |
| 第 8 秒 | 已收到部分报价，正在合并价格 | 不是卡死，而是在处理中 |
| 第 12 秒 | 共 8 个结果，最低价 4120 元 | 等待有结果且过程可信 |

同样是 12 秒，系统解释得越清楚，用户越不容易中途打断任务或重复发起调用。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例。  
它做了三件事：

1. 把 Anthropic 和 OpenAI 的事件统一映射成同一种内部事件。
2. 为每个工具调用维护独立状态，避免多调用串流混淆。
3. 把状态投射为三张卡片：参数卡片、进度卡片、结果卡片。

这个示例不依赖第三方库，直接 `python demo.py` 即可运行。

```python
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class ToolCallState:
    call_id: str
    tool_name: str = ""
    buffer: str = ""
    fragment_count: int = 0
    final_args: Optional[dict] = None
    phase: str = "created"  # created -> args_streaming -> args_done -> running -> finished -> failed
    result_summary: str = ""
    error: str = ""

    def cards(self) -> Dict[str, str]:
        if self.phase == "failed":
            return {
                "card1": f"工具: {self.tool_name or 'unknown'}",
                "card2": f"失败: {self.error}",
                "card3": "可重试或回退到普通加载态",
            }

        param_text = self.buffer if self.buffer else "{}"
        if self.final_args is not None:
            param_text = json.dumps(self.final_args, ensure_ascii=False)

        progress_map = {
            "created": "已创建调用，等待参数",
            "args_streaming": f"参数生成中，已接收 {self.fragment_count} 个片段",
            "args_done": "参数已定稿，等待执行工具",
            "running": "工具执行中",
            "finished": "结果已返回",
        }

        return {
            "card1": f"工具: {self.tool_name or 'unknown'} | 参数: {param_text}",
            "card2": progress_map.get(self.phase, self.phase),
            "card3": self.result_summary,
        }


class ToolStreamAggregator:
    def __init__(self) -> None:
        self.calls: Dict[str, ToolCallState] = {}

    def _get_or_create(self, call_id: str) -> ToolCallState:
        if call_id not in self.calls:
            self.calls[call_id] = ToolCallState(call_id=call_id)
        return self.calls[call_id]

    def on_normalized_event(self, event: Dict[str, Any]) -> Dict[str, str]:
        event_type = event["type"]
        call_id = event["call_id"]
        state = self._get_or_create(call_id)

        if event_type == "call_started":
            state.tool_name = event["tool_name"]
            state.phase = "created"
            return state.cards()

        if event_type == "args_delta":
            state.phase = "args_streaming"
            state.buffer += event["delta"]
            state.fragment_count += 1
            return state.cards()

        if event_type == "args_done":
            raw = event.get("full_args") or state.buffer
            try:
                state.final_args = json.loads(raw)
                state.phase = "args_done"
            except json.JSONDecodeError as exc:
                state.phase = "failed"
                state.error = f"参数不是完整 JSON: {exc.msg}"
            return state.cards()

        if event_type == "tool_running":
            state.phase = "running"
            return state.cards()

        if event_type == "tool_result":
            state.phase = "finished"
            state.result_summary = event["summary"]
            return state.cards()

        if event_type == "tool_error":
            state.phase = "failed"
            state.error = event["error"]
            return state.cards()

        raise ValueError(f"unknown event type: {event_type}")


def normalize_openai_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = event["type"]

    if t == "response.output_item.added":
        return {
            "type": "call_started",
            "call_id": event["item"]["id"],
            "tool_name": event["item"]["name"],
        }

    if t == "response.function_call_arguments.delta":
        return {
            "type": "args_delta",
            "call_id": event["item_id"],
            "delta": event["delta"],
        }

    if t == "response.function_call_arguments.done":
        return {
            "type": "args_done",
            "call_id": event["item"]["id"],
            "full_args": event["item"]["arguments"],
        }

    return None


def normalize_anthropic_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = event["type"]

    if t == "content_block_start" and event["content_block"]["type"] == "tool_use":
        return {
            "type": "call_started",
            "call_id": event["content_block"]["id"],
            "tool_name": event["content_block"]["name"],
        }

    if t == "content_block_delta" and event["delta"]["type"] == "input_json_delta":
        return {
            "type": "args_delta",
            "call_id": event["id"],
            "delta": event["delta"]["partial_json"],
        }

    if t == "content_block_stop":
        return {
            "type": "args_done",
            "call_id": event["id"],
        }

    return None


def demo_openai() -> None:
    aggregator = ToolStreamAggregator()

    openai_events = [
        {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_1",
                "name": "get_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": '{"location":"San',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "delta": ' Francisco","unit":"celsius"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item": {
                "id": "fc_1",
                "name": "get_weather",
                "arguments": '{"location":"San Francisco","unit":"celsius"}',
            },
        },
    ]

    for raw_event in openai_events:
        normalized = normalize_openai_event(raw_event)
        if normalized:
            cards = aggregator.on_normalized_event(normalized)
            print(cards)

    cards = aggregator.on_normalized_event(
        {"type": "tool_running", "call_id": "fc_1"}
    )
    print(cards)

    cards = aggregator.on_normalized_event(
        {
            "type": "tool_result",
            "call_id": "fc_1",
            "summary": "天气晴，18°C，风速 3 m/s",
        }
    )
    print(cards)

    state = aggregator.calls["fc_1"]
    assert state.final_args == {"location": "San Francisco", "unit": "celsius"}
    assert state.phase == "finished"
    assert "天气晴" in state.result_summary


if __name__ == "__main__":
    demo_openai()
    print("demo passed")
```

这段代码解决了原始玩具代码里几个常见问题：

| 原问题 | 为什么不稳 | 改进方式 |
| --- | --- | --- |
| 只维护一个全局 `buffer` | 多个工具并发时会串流混淆 | 用 `call_id` 区分每个调用 |
| 假设所有 delta 字段都叫 `delta` | Anthropic 的参数片段字段名不同 | 先归一化，再进入统一状态机 |
| `done` 时直接 `json.loads(self.buffer)` | 有些平台会在完成事件直接给完整 arguments，更可靠 | 优先使用完成事件里的完整参数，没有再回退到缓存 |
| 没有“工具执行中”阶段 | 参数结束不代表结果已经回来 | 增加 `tool_running` 状态 |
| 没有错误路径 | 一旦 JSON 不完整，前端不知道如何显示 | 显式进入 `failed` 状态 |

前端真正落地时，可以把 `cards()` 的三个字段直接映射到三张卡片：

| 卡片 | 建议内容 | 为什么这么设计 |
| --- | --- | --- |
| 参数卡片 | 工具名、参数快照、脱敏字段 | 帮用户确认系统到底要查什么 |
| 进度卡片 | 参数生成中 / 工具执行中 / 已完成 | 把等待拆成可理解阶段 |
| 结果卡片 | 先摘要，后展开详情 | 避免一上来渲染大段 JSON 或长表格 |

如果你确实想在参数还没定稿时就给用户“预览感”，更稳妥的方式不是每收到一片就 `JSON.parse`，而是分层处理：

1. 第一层：只显示原始片段拼接后的字符串。
2. 第二层：在界面上把它标记为“草稿参数”。
3. 第三层：只有收到完成事件后，才把它提升为“最终参数”。

这个约束很重要，因为“看起来像 JSON 的字符串”不等于“已经是可信 JSON”。

---

## 工程权衡与常见坑

最常见的坑，是把“增量字符串”误当成“完整 JSON”，然后让整条链路在边界情况下变得脆弱。

| 坑 | 原因 | 规避策略 |
| --- | --- | --- |
| 每个 delta 都直接解析 | 片段天然可能截断，尤其在细粒度工具流中 | 先累积，完成后再解析 |
| 只在最终完成时更新 UI | 用户在前 5 到 10 秒看到的是空白 | delta 阶段至少更新片段计数和阶段状态 |
| 一个页面多个工具调用时串流混淆 | 没按 `call_id`、`item_id`、`output_index` 或内容块索引分组 | 为每个调用维护独立状态 |
| 把参数展示得过细 | 可能暴露敏感字段、内部查询语句或业务主键 | 对用户展示层做脱敏和摘要化 |
| 把“参数完成”当成“结果完成” | 模型生成参数和后端执行工具是两段不同耗时 | UI 状态至少拆成参数生成中、工具执行中、结果已返回 |
| 只做流式，不做失败态 | 网络中断、超时、JSON 截断时界面会卡在半成品 | 增加超时、失败、可重试状态 |
| 用片段数量假装百分比进度 | 片段多少和真实剩余时间没有线性关系 | 用阶段提示代替虚假百分比 |
| 认为“有流式就一定更快” | 真实后端耗时可能完全没变 | 把目标定义为“降低等待焦虑和误操作率” |

还要注意两个新手最容易忽略的判断。

第一，参数流完成，不代表工具一定成功。  
例如参数 `{"location":"San Francisco"}` 已经组装成功，但天气 API 可能超时、限流或返回 500。  
因此结果卡片不能直接写成“查询成功”，更合理的写法是：

1. 参数已定稿
2. 正在请求天气服务
3. 查询完成 / 查询失败

第二，前端展示的是“用户可理解的状态”，不是“底层事件原样转储”。  
用户真正关心的通常不是 `response.function_call_arguments.delta` 这串事件名，而是：

- 系统准备查什么
- 现在是在生成参数还是执行查询
- 还要不要继续等
- 最后查到了什么

所以工程上最好做一层“事件 -> 业务状态 -> UI 文案”的映射，不要把底层协议直接暴露给用户界面。

可以把这层映射写成一个简单表：

| 底层信号 | 业务状态 | 用户文案 |
| --- | --- | --- |
| 收到开始事件 | 准备调用 | 正在准备查询 |
| 收到参数 delta | 参数生成中 | 正在补全查询条件 |
| 参数完成 | 准备执行工具 | 查询条件已确认 |
| 工具已发出请求 | 工具执行中 | 正在访问外部服务 |
| 工具返回结果 | 结果已返回 | 已获得结果 |
| 超时 / 异常 | 调用失败 | 查询失败，可重试 |

这层抽象的价值在于：底层供应商可以换，UI 逻辑不用重写。

---

## 替代方案与适用边界

如果拿不到 Anthropic 或 OpenAI 的流式事件，也可以退回“后台作业模式”。  
这里的“轮询”是指前端每隔几秒请求一次后端，询问某个任务当前处于排队、处理中还是已完成。

| 方案 | 延迟感知 | 实现复杂度 | 用户透明度 | 适用场景 |
| --- | --- | --- | --- | --- |
| 流式 delta | 最细 | 中到高 | 高 | 长耗时、希望展示参数与进度的对话式智能体 |
| 后台作业 + 轮询 | 中 | 中 | 中 | 无法接入 SSE/WebSocket，但任务状态可持久化 |
| 只显示加载中 | 最差 | 低 | 低 | 小于 3 秒的短调用 |
| 立即返回“任务已受理”+ 异步通知 | 中 | 中到高 | 中 | 超长任务，如转码、批处理、离线索引 |

真实工程里，后台作业模式适合“视频转码、批量报表、长检索、离线 embedding 构建”这类任务：

1. 前端发起任务，后端返回 `job_id`
2. 前端轮询 `/jobs/{id}`
3. 状态依次展示“已排队 -> 处理中 -> 完成 / 失败”
4. 完成后再拉取结果

这种方式的优点是边界清晰：

- 后端可以把任务状态落库
- 前端刷新页面后仍能恢复进度
- 不依赖模型厂商的流式参数能力

但它也有明确缺点：

- 用户看不到参数生成过程
- 看不到“系统到底在查什么”
- 粒度通常只有排队、处理中、完成，解释性明显弱于参数级流式展示

因此可以用下面这个判断表做技术选型：

| 问题 | 更适合流式 delta | 更适合作业轮询 |
| --- | --- | --- |
| 用户想实时看到系统在查什么 | 是 | 否 |
| 参数生成过程本身有解释价值 | 是 | 否 |
| 任务非常长，可能跨页面恢复 | 否 | 是 |
| 基础设施不方便接 SSE / WebSocket | 否 | 是 |
| 任务主要是后端重计算，不是模型补全参数 | 否 | 是 |

简单说：

- 如果重点是“解释等待过程”，优先流式 delta。
- 如果重点是“稳定跑完超长任务”，优先后台作业。
- 如果调用只要 1 到 3 秒，直接普通加载态通常就够了。

---

## 参考资料

| 资料 | 对应主题 |
| --- | --- |
| Anthropic Streaming Messages: https://platform.claude.com/docs/en/build-with-claude/streaming | `message_start`、`content_block_*`、`input_json_delta.partial_json` 的基础流式事件模型 |
| Anthropic Fine-grained tool streaming: https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming | 细粒度工具参数流、`eager_input_streaming`、不完整 JSON 的边界 |
| OpenAI Function Calling Guide: https://platform.openai.com/docs/guides/function-calling | `response.output_item.added`、`response.function_call_arguments.delta`、`response.function_call_arguments.done` 的聚合方式 |

这些资料在工程上对应三条非常直接的实现原则：

1. 参数增量本质上是字符串流，不要把每个片段当成完整 JSON。
2. 多个工具调用必须按调用标识分组，不能共享一个全局缓冲区。
3. 对用户展示的是“业务状态”，不是供应商协议名本身。

## 核心结论

API-Bank 是一个面向工具增强型大语言模型的运行级评测基准。运行级，意思不是只看模型“会不会说”，而是看它能不能真的把 API 选对、参数填对、调用顺序排对，并且处理返回结果。它的核心价值不在“题目多”，而在“任务链条完整”。

更具体地说，API-Bank用 73 个可执行 API、314 组工具使用对话、753 次真实 API 调用，把模型的工具能力拆成三层：

| 评测层级 | 是否需要检索 API | 是否需要规划步骤 | 典型难点 |
|---|---:|---:|---|
| Call | 否 | 否 | 选对工具、填对参数 |
| Retrieve+Call | 是 | 否 | 从候选 API 中找到正确接口 |
| Plan+Retrieve+Call | 是 | 是 | 先规划，再多步执行，并处理中间状态 |

对新手可以这样理解：它像一个“API 超市”。用户提出任务，模型要先判断拿哪个工具，再看说明书，再按顺序调用。如果任务简单，只拿一次工具；如果任务复杂，要先查工具，再规划“先做什么、后做什么”。

API-Bank的重要结论不是“某个模型分数更高”这么简单，而是它证明了工具使用能力至少由三个部分组成：

$$
\text{ToolUseAbility} \approx f(\text{规划}, \text{检索}, \text{执行})
$$

论文给出的结果显示，GPT-4 在规划稳定性上更强，GPT-3.5 相比 GPT-3 在调用成功率上有明显提升，但随着任务从单步变成多步、从少量 API 变成大量候选 API，性能会明显下滑。这说明“会函数调用”不等于“具备稳定 Agent 能力”。

---

## 问题定义与边界

API-Bank想回答的问题很明确：当模型面对真实工具时，它到底能不能完成一个从理解需求到执行 API 的闭环。

这里的“工具增强型大语言模型”，白话讲，就是不只会生成文本，还会调用外部工具来完成任务的模型。比如查天气、创建日程、搜索信息、生成图像，都不该只靠语言模型自己瞎编，而应通过 API 获取真实结果。

它的评测边界也很清楚：

| 维度 | API-Bank 的范围 | 不覆盖的范围 |
|---|---|---|
| 工具数量 | 73 个可运行 API | 开放世界中无限工具 |
| 对话数据 | 314 组标注对话 | 任意开放式闲聊 |
| 调用记录 | 753 次真实调用 | 只做文本打分、不执行调用 |
| 目标 | 衡量规划、检索、执行能力 | 只看文笔、流畅度、人格化回复 |

因此，API-Bank测的不是“模型有没有常识”，而是“模型能否在定义明确的工具空间里，做出正确、可执行、可验证的动作”。

一个玩具例子很适合说明边界：

- 用户说：“帮我查杭州今天的天气。”
- 如果系统已经把天气 API 明确给到模型，这属于 `Call`。
- 如果系统只给一堆 API 名称，模型要先找出哪个能查天气，这属于 `Retrieve+Call`。
- 如果用户说：“把我明天下午空闲时间找出来，再安排一个和客户的会议，并提醒我提前 30 分钟”，模型可能先取身份令牌、再读日程、再创建新日程、再设置提醒，这才是 `Plan+Retrieve+Call`。

这也是它和普通问答 benchmark 的分界线。普通问答可以接受“答案大致正确”，但 API-Bank 不行。API 名拼错一个字符、参数类型错一次、调用顺序颠倒一次，都可能直接失败。

---

## 核心机制与推导

API-Bank 的设计重点是“分层评测”和“复杂度分桶”。

第一层是 `Call`。这是最直接的情况：API 已经给定，模型主要负责把用户需求映射成正确的参数。这里考的是“执行能力”，也就是能不能把话变成正确的结构化调用。

第二层是 `Retrieve+Call`。这里的“检索”不是信息检索里的全文搜索，而是在候选工具集合中找到目标 API。白话讲，就是先找到正确说明书，再按说明书调用。

第三层是 `Plan+Retrieve+Call`。这里增加了“规划”要求。规划，白话讲，就是把一个复杂目标拆成若干有先后顺序的步骤。模型不仅要知道“用什么工具”，还要知道“先做什么，后做什么”。

这三层不是并列关系，而是逐层叠加：

$$
\text{Call} \subset \text{Retrieve+Call} \subset \text{Plan+Retrieve+Call}
$$

从能力需求看，也可以写成：

$$
S = w_1 C + w_2 R + w_3 P
$$

其中：

- $C$ 表示调用正确性，能否选对 API 并构造正确参数
- $R$ 表示检索正确性，能否在候选 API 中找到目标接口
- $P$ 表示规划正确性，能否生成合理步骤并按顺序执行

当任务进入多步阶段时，整体成功率通常不是单步成功率的简单复制，而更接近乘法衰减。假设每一步成功率为 $p$，总共需要 $n$ 步，那么端到端成功率大致会接近：

$$
P(\text{success}) \approx p^n
$$

这就是为什么看起来“每一步都还行”的模型，一到多步 Agent 任务就明显掉线。比如单步成功率 0.9，三步任务端到端成功率约为 $0.9^3=0.729$；五步任务则降到约 0.59。

API-Bank 还引入了两个复杂度维度：

| 复杂度维度 | 低复杂度 | 高复杂度 | 影响 |
|---|---|---|---|
| API 候选规模 | Few API | Many API | 检索难度上升 |
| 调用次数 | Single Call | Multiple Calls | 状态管理与顺序依赖上升 |

这使它不只是“分三类题”，而是形成一个二维难度网格。比如：

- `Call + Few API + Single Call`：最基础，类似直接查天气
- `Retrieve+Call + Many API + Single Call`：要先在很多工具里找对 API
- `Plan+Retrieve+Call + Many API + Multiple Calls`：最接近真实 Agent 任务

一个真实工程例子是“获取令牌后创建日程”。用户目标看起来只有一句话：“帮我安排下周三下午三点和王工开会。”但系统可能要求：

1. 先调用 `GetUserToken`
2. 再验证用户身份或账户状态
3. 然后调用 `AddAgenda`
4. 如参数不全，再向用户补问
5. 最后返回可读结果

这里每一步都可能失败，而且失败原因不同。模型如果跳过令牌步骤，即使后面的日程参数全对，服务仍然会拒绝请求。

---

## 代码实现

如果要自己实现一个最小版 API-Bank 风格评测器，核心流程可以压缩为四步：读任务、选工具、执行调用、记录结果。

下面这个 Python 玩具实现展示了最基本的调用与评分逻辑。它不依赖真实网络，但保留了 API 选择、参数校验和成功率计算这三个关键环节。

```python
from dataclasses import dataclass

@dataclass
class APICall:
    api_name: str
    params: dict

API_SPECS = {
    "GetWeather": {"required": ["city"]},
    "GetUserToken": {"required": ["user_id"]},
    "AddAgenda": {"required": ["token", "title", "time"]},
}

def validate_call(call: APICall) -> bool:
    if call.api_name not in API_SPECS:
        return False
    required = API_SPECS[call.api_name]["required"]
    return all(k in call.params and call.params[k] for k in required)

def run_call(call: APICall):
    if not validate_call(call):
        return {"ok": False, "error": "invalid_call"}

    if call.api_name == "GetWeather":
        return {"ok": True, "result": f"{call.params['city']} 晴 24C"}

    if call.api_name == "GetUserToken":
        return {"ok": True, "result": f"TOKEN-{call.params['user_id']}"}

    if call.api_name == "AddAgenda":
        if not str(call.params["token"]).startswith("TOKEN-"):
            return {"ok": False, "error": "unauthorized"}
        return {
            "ok": True,
            "result": f"agenda<{call.params['title']}>@{call.params['time']}"
        }

    return {"ok": False, "error": "unknown_api"}

def evaluate(calls):
    results = [run_call(c) for c in calls]
    success = sum(1 for r in results if r["ok"])
    return {
        "total": len(results),
        "success": success,
        "success_rate": success / len(results) if results else 0.0,
        "results": results,
    }

# 玩具例子：单步调用
weather_case = [APICall("GetWeather", {"city": "Hangzhou"})]
report = evaluate(weather_case)
assert report["success_rate"] == 1.0
assert report["results"][0]["result"] == "Hangzhou 晴 24C"

# 真实工程风格例子：先取 token，再加日程
token_resp = run_call(APICall("GetUserToken", {"user_id": "alice"}))
token = token_resp["result"]
agenda_resp = run_call(APICall("AddAgenda", {
    "token": token,
    "title": "和王工开会",
    "time": "2026-04-16 15:00"
}))
assert token.startswith("TOKEN-")
assert agenda_resp["ok"] is True

# 常见错误：跳过 token
bad_resp = run_call(APICall("AddAgenda", {
    "token": "alice",
    "title": "和王工开会",
    "time": "2026-04-16 15:00"
}))
assert bad_resp["ok"] is False
assert bad_resp["error"] == "unauthorized"
```

这个例子虽然简单，但已经体现了 API-Bank 的核心思想：

| 步骤 | 输入 | 输出 | 失败点 |
|---|---|---|---|
| 选择 API | 用户需求 | API 名 | 选错工具 |
| 参数填充 | 对话上下文 | 结构化参数 | 漏字段、类型错 |
| 执行调用 | API 调用串 | 返回结果 | 鉴权失败、接口失败 |
| 结果评估 | 调用日志 | 成功/失败标签 | 评估口径不一致 |

如果把这个流程扩展成完整评测器，通常还需要记录以下字段：

- `task_id`：任务编号
- `level`：属于 `Call`、`Retrieve+Call` 还是 `Plan+Retrieve+Call`
- `api_pool_size`：Few 还是 Many
- `call_count`：Single 还是 Multiple
- `gold_calls`：标准 API 调用序列
- `pred_calls`：模型输出的调用序列
- `success`：端到端是否成功
- `error_type`：失败类别

可以把最小评分逻辑写成：

```python
def score_case(level, api_pool_size, call_count, success):
    difficulty = 1
    difficulty += 1 if level != "Call" else 0
    difficulty += 1 if api_pool_size == "Many" else 0
    difficulty += 1 if call_count == "Multiple" else 0
    return difficulty if success else 0

assert score_case("Call", "Few", "Single", True) == 1
assert score_case("Plan+Retrieve+Call", "Many", "Multiple", True) == 4
assert score_case("Plan+Retrieve+Call", "Many", "Multiple", False) == 0
```

这不是论文原始公式，而是一个工程上容易实现的近似打分思路，用来说明：难度越高，成功案例越有信息量。

---

## 工程权衡与常见坑

API-Bank 的工程价值很大，但实现和使用时有几个典型坑，几乎每个工具型 Agent 系统都会踩到。

首先是 API 名称和参数错误。这类错误最“低级”，但在评测里杀伤力最大。因为工具调用不是自然语言，没有“大意对了”这一说。

其次是调用顺序错误。很多任务存在强依赖关系，比如必须先拿 token，再写日程；必须先查资源 id，再执行删除；必须先登录，再做账户操作。顺序一错，后面全错。

再次是中间状态丢失。多步任务要求模型记住上一步返回值，并把它带入下一步。比如把 `GetUserToken` 返回的 token 传给 `AddAgenda`。如果模型只会“每一步单独看起来合理”，而不会维护状态，就很难通过多步评测。

常见错误可以整理成表：

| 错误类型 | 表现 | 根因 | 应对方式 |
|---|---|---|---|
| API 名拼写错 | 调用不存在的接口 | 检索或复制不稳定 | 使用受控 API 枚举 |
| 参数缺失 | 少填必填字段 | 槽位抽取不完整 | 参数 schema 校验 |
| 参数格式错 | 时间、ID、枚举值不合法 | 类型约束弱 | 强制类型检查与重试 |
| 顺序错乱 | 先写后验、先用后取 | 规划能力不足 | 显式步骤计划检查 |
| 状态丢失 | 忘记 token、id、session | 上下文管理差 | 中间结果结构化缓存 |
| 恢复失败 | 一步失败后整体崩溃 | 缺少异常分支 | 增加补问与重试机制 |

真实工程例子最能说明问题。假设用户说：“帮我安排明天下午 3 点和客户开会。”系统要求先鉴权。一个粗糙模型可能直接调用 `AddAgenda(title, time)`，结果返回 401。更稳妥的 Agent 应该识别出依赖链：

1. 检查当前是否已有 token
2. 如果没有，调用 `GetUserToken`
3. 若缺少用户标识，向用户补问
4. 拿到 token 后调用 `AddAgenda`
5. 如果时间冲突，再触发查询空闲时间或要求改期

恢复逻辑通常要显式写出来，而不是指望模型“自己悟出来”：

```python
def robust_add_agenda(user_id, title, time):
    token_resp = run_call(APICall("GetUserToken", {"user_id": user_id}))
    assert token_resp["ok"], "token 获取失败"

    agenda_resp = run_call(APICall("AddAgenda", {
        "token": token_resp["result"],
        "title": title,
        "time": time,
    }))

    if agenda_resp["ok"]:
        return agenda_resp["result"]

    # 简化恢复策略：只要鉴权失败就终止并暴露错误
    return f"failed: {agenda_resp['error']}"

assert "agenda<" in robust_add_agenda("alice", "客户会议", "2026-04-16 15:00")
```

工程上真正重要的不是“让模型偶尔成功”，而是把错误变成可观察、可归因、可恢复的失败。否则 benchmark 分数看起来不错，上线后仍然会频繁出事故。

---

## 替代方案与适用边界

API-Bank 不是唯一的 Agent 评测方法，但它有鲜明定位：强调真实 API 调用和端到端执行，而不是只做静态文本评分。

和常见替代方案对比，可以看得更清楚：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 纯文本问答评测 | 便宜、快、易批量运行 | 不测真实执行 | 语言理解、知识问答 |
| 函数调用格式评测 | 便于对齐 JSON 结构 | 往往不含真实后端 | 测结构化输出能力 |
| 模拟 API 评测 | 可控、稳定、低成本 | 真实性不足 | 早期原型验证 |
| API-Bank | 有真实调用、分层任务、多域对话 | 更重、更复杂、维护成本高 | 工具型 Agent 评估与调优 |

因此，它特别适合以下场景：

- 评估一个模型是否具备“从需求到执行”的工具使用能力
- 比较不同模型在检索、规划、执行上的短板
- 做 Agent 系统回归测试，观察升级后是否退化
- 为指令微调、工具微调或策略优化提供可验证目标

它不太适合以下场景：

- 只想看模型回答是否流畅
- 只评估单轮问答
- 工具环境变化极快，无法维持稳定可复现实验
- 只关心函数调用语法，不关心真实后端执行

对新手来说，可以用一句话抓住边界：如果你只是想测“模型会不会回答”，API-Bank 太重；如果你想测“模型能不能把 `GetUserToken -> AddAgenda` 这样的链路真的跑通”，API-Bank 很合适。

---

## 参考资料

1. API-Bank 原始论文：Li et al., *API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs*，给出基准定义、73 个 API、314 组对话、753 次调用及主要实验结果。  
   https://arxiv.gg/abs/2304.08244?utm_source=openai

2. ScienceStack 论文摘要页：整理了 `Call`、`Retrieve+Call`、`Plan+Retrieve+Call` 三层结构，以及 Few/Many API、Single/Multiple Calls 的复杂度维度，适合理解评测框架。  
   https://www.sciencestack.ai/paper/2304.08244v2?utm_source=openai

3. ADS / arXiv 索引：提供论文元数据与 DOI，可用于核对论文发表信息。  
   https://ui.adsabs.harvard.edu/abs/2023arXiv230408244L/abstract

4. CSDN 技术解读：适合作为入门解释，帮助理解“API 超市”式直觉和任务层级。  
   https://blog.csdn.net/m0_60388871/article/details/144033307?utm_source=openai

5. KuxAI 文章：提供 `GetUserToken`、`AddAgenda` 一类多步工具调用场景，适合补充工程直觉。  
   https://www.kuxai.com/article/1007?utm_source=openai

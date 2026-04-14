## 核心结论

ToolBench 是一个面向“工具使用能力”的评测基准。这里的工具使用能力，白话说，就是模型看到自然语言任务后，能不能选对 API、按对顺序调用、把参数写对。它的核心价值不在于“再造一个聊天数据集”，而在于把真实 API 调用过程拆成可检查的 JSON 序列，从而把原本很难量化的代理行为，变成可以打分的结构化输出。

它覆盖 16,464 个真实 REST API，涉及天气、金融、购物、地图、办公等多个领域，并支持单步任务与多步任务。单步任务，白话说，就是只需要一次 API 调用；多步任务，就是需要多次调用并保持顺序关系。对于想评估“大模型是否真的会用工具”的团队，这比只看最终自然语言回答更有判别力，因为模型可能“会说”，但不会“调”。

一个最小玩具例子是天气查询。用户说“查询北京今天的天气”，评测期望模型输出：

```json
[
  {
    "name": "weather/today",
    "parameters": {
      "location": "Beijing"
    }
  }
]
```

如果工具名、参数名、参数层级都对，这条样本的工具选择准确率就是 1，结构匹配也可能为 1；如果模型写成 `"weather": "Beijing"`，它看起来接近，但在评测里仍然是错的。

ToolBench 常见的四个核心指标如下：

| 指标 | 关注点 | 白话解释 |
|---|---|---|
| Pass Rate | 整条调用链是否通过 | 参考序列要点基本都对，才能算“做成了” |
| Win Rate | 与基线模型相比谁更好 | 同一道题，当前模型是否比对手更强 |
| AST Accuracy | 调用结构是否对齐 | 不只看字面值，还看 JSON 层级结构是否合理 |
| Tool Selection Accuracy | 工具是否选对 | 选出来的工具里，有多少是真的需要的 |

其中最直观的公式是工具选择准确率：

$$
\text{Tool Selection Accuracy} = \frac{\text{正确选出的工具数}}{\text{模型选出的工具总数}}
$$

如果模型本该只选 `weather/today`，却额外多选了 `map/search_city`，那分数就会下降。这个指标的意义是：即使最终某一步侥幸接近正确，模型若存在“乱调工具”的问题，也会被单独暴露出来。

---

## 问题定义与边界

ToolBench 要回答的问题很明确：如何在真实 API 背景下，量化模型把自然语言任务映射成工具调用序列的能力。这里的“映射”，白话说，就是把“帮我查苹果股价然后下单”翻译成一串机器可执行的 JSON 调用步骤。

它评测的不是“模型最终回答得像不像人”，而是“模型有没有正确规划工具链”。因此，它的检查对象是工具调用序列，而不是自然语言回复质量。这个边界很关键，因为很多代理系统的失败，并不是解释能力弱，而是中间调用链有一步错了，后面就全错。

一个简单的两步例子是金融助手任务：

```json
[
  {"name": "stock/query", "parameters": {"symbol": "AAPL"}},
  {"name": "trade/execute", "parameters": {"symbol": "AAPL", "side": "buy", "amount": 10}}
]
```

如果模型先下单再查价，或者第二步漏了 `amount`，都属于失败。也就是说，ToolBench 不只看“用了哪些工具”，还看“顺序是否正确”。

边界可以概括为下表：

| 评测边界 | ToolBench 的做法 | 含义 |
|---|---|---|
| 单步 vs 多步 | 同时覆盖 | 既看一次调用是否正确，也看链式规划是否稳定 |
| 真实 API 覆盖范围 | 使用大规模真实 REST API | 不是手写几个玩具函数，而是接近真实外部服务世界 |
| 是否执行真实请求 | 通常不追踪真实网络执行结果 | 重点是“指令到调用序列”的映射，而不是在线系统成败 |

这意味着 ToolBench 更像“离线结构评测”，而不是“在线端到端产品压测”。它适合回答“模型会不会调用工具”，但不直接回答“工具返回异常时模型怎么恢复”“接口限流时是否重试”“真实支付流程是否安全”。这些问题属于更高一层的工程验证。

---

## 核心机制与推导

ToolBench 的基本流程可以概括成三步。第一步，基于真实 API 文档生成任务；第二步，为每个任务构造参考调用序列；第三步，把模型输出的 JSON 与参考序列做结构化比对。这里的参考调用序列，也叫 ground truth，白话说，就是评测器认为“正确或标准”的那条工具调用路径。

它的关键设计是“真实 API + 参考序列”。如果只有 API 文档，没有标准序列，评测就会变成主观判断；如果只有玩具工具而不是真实 API，结果又很难外推到真实代理系统。ToolBench 试图在两者之间取平衡。

以天气任务为例，参考序列可能是：

```json
[
  {"name":"weather/today","parameters":{"location":"NYC"}}
]
```

若模型输出：

```json
[
  {"name":"weather/today","parameters":{"location":"NYC"}}
]
```

那么工具名、参数键、参数层级都对，AST Accuracy 很高。AST 是抽象语法树，白话说，就是把 JSON 看成一个有层级的结构树，而不是简单字符串。这样做的目的，是避免“表面像、结构错”的情况漏判。

例如下面两个输出，字符串差异不大，但结构意义完全不同：

```json
{"name":"weather/today","parameters":{"location":"NYC"}}
{"name":"weather/today","location":"NYC"}
```

第一个表示 `location` 属于 `parameters`；第二个则把 `location` 提到了顶层。对人来说可能“能看懂”，但对工具系统来说通常就是错误结构。因此可以把 AST Accuracy 理解为“结构层面的正确率”，一个简化写法是：

$$
\text{AST Accuracy} = \frac{\text{结构匹配的调用节点数}}{\text{总调用节点数}}
$$

而工具选择准确率则更关注“有没有挑对工具”：

$$
\text{Tool Selection Accuracy} = \frac{|T_{\text{pred}} \cap T_{\text{gold}}|}{|T_{\text{pred}}|}
$$

其中 $T_{\text{pred}}$ 是模型选出的工具集合，$T_{\text{gold}}$ 是标准工具集合。

玩具例子可以帮助理解。假设标准答案只需要 `weather/today`，模型却输出了 `weather/today` 和 `calendar/create_event` 两个工具。即使第一个工具是对的，第二个多余工具也会把工具选择准确率拉低，因为模型表现出“过度调用”。

真实工程里，这个机制非常有用。比如做一个金融代理，用户说“先看特斯拉股价，再决定是否买入 5 股”。模型不仅要能选中金融查询工具，还要知道交易动作不能抢跑。只要链条里某一步错位，最终代理行为就不可控。ToolBench 的价值就在于，它把这种不可控拆成“选错工具”“顺序错”“参数错”“结构错”几个可定位问题。

---

## 代码实现

下面给出一个简化版评测脚本，用来说明 ToolBench 风格的评分思路。它不依赖真实网络，只比较模型输出与标准序列的结构和内容。

```python
from typing import List, Dict, Any, Set

Call = Dict[str, Any]

def tool_set(seq: List[Call]) -> Set[str]:
    return {item["name"] for item in seq if "name" in item}

def tool_selection_accuracy(pred: List[Call], gold: List[Call]) -> float:
    pred_tools = tool_set(pred)
    gold_tools = tool_set(gold)
    if not pred_tools:
        return 0.0
    return len(pred_tools & gold_tools) / len(pred_tools)

def ast_equal(pred: Call, gold: Call) -> bool:
    # 简化版 AST 对齐：要求 name、parameters 键存在且层级一致
    if set(pred.keys()) != set(gold.keys()):
        return False
    if pred.get("name") != gold.get("name"):
        return False
    if not isinstance(pred.get("parameters"), dict):
        return False
    if set(pred["parameters"].keys()) != set(gold["parameters"].keys()):
        return False
    return pred["parameters"] == gold["parameters"]

def ast_accuracy(pred: List[Call], gold: List[Call]) -> float:
    if not gold:
        return 0.0
    matched = 0
    for p, g in zip(pred, gold):
        if ast_equal(p, g):
            matched += 1
    return matched / len(gold)

def pass_rate(pred: List[Call], gold: List[Call]) -> float:
    return 1.0 if pred == gold else 0.0

gold = [
    {"name": "weather/today", "parameters": {"location": "Beijing"}}
]

pred_ok = [
    {"name": "weather/today", "parameters": {"location": "Beijing"}}
]

pred_bad = [
    {"name": "weather/today", "location": "Beijing"}
]

assert pass_rate(pred_ok, gold) == 1.0
assert tool_selection_accuracy(pred_ok, gold) == 1.0
assert ast_accuracy(pred_ok, gold) == 1.0
assert pass_rate(pred_bad, gold) == 0.0
assert ast_accuracy(pred_bad, gold) == 0.0
```

这个脚本体现了四个关键动作：

| 步骤 | 做什么 | 对应意义 |
|---|---|---|
| 读取标准序列 | 加载 ground truth JSON 数组 | 获得标准工具链 |
| 解析模型输出 | 把模型文本转成 `name + parameters` 结构 | 统一进入评测器 |
| 顺序比较 | 逐步检查每一步是否对齐 | 发现规划或链路顺序错误 |
| 更新指标 | 计算 Pass、AST、Selection 等 | 从多个角度定位能力缺口 |

如果要写成更接近评测流水的伪代码，通常是下面这种形式：

```python
for step, ground_truth in enumerate(gold_sequence):
    assert step < len(model_output)
    current = model_output[step]
    update_tool_selection(current, ground_truth)
    update_ast_score(current, ground_truth)

final_pass = 1 if model_output == gold_sequence else 0
```

真实工程例子是：你在做一个“差旅助理”，任务是“先查明天上海到北京的机票，再把最低价结果写入报销系统”。如果模型生成：

1. `flight/search`
2. `expense/create_record`

而且第二步参数里包含第一步得到的票价信息，那么这条链就能被结构化验证。相反，如果模型直接调用报销系统，说明它虽然“知道最终目标”，但不会正确使用中间工具。这正是 ToolBench 想测出来的能力缺口。

---

## 工程权衡与常见坑

ToolBench 的优点是严格，缺点也来自严格。它要求模型输出 JSON 数组，工具名和参数层级都要稳定。这种约束让评测可重复，但也让一些“看起来差不多”的输出直接判错。换句话说，它更像编译器，不像聊天评委。

第一个权衡是格式严格性。严格格式能捕捉结构错误，但会放大输出不稳定带来的误差。新手常见错误是只输出字符串，或者忘记 `parameters` 这个字段，例如：

```python
prompt_template = """
You must output in JSON array.
Each item must contain:
- name: tool name
- parameters: object
Do not output plain text.
"""
```

如果模型输出 `"weather/today Beijing"`，评测器根本无法抽取调用信息，这时 Tool Selection Accuracy 往往直接为 0。

第二个权衡是多工具链复杂度。链路越长，错误传播越明显。一条四步调用链只要第一步参数错了，后面即使格式都对，也可能整体失败。因此实践中通常先按 API 分类跑简单任务，再扩展到跨域多工具任务，而不是一开始就上最复杂场景。

常见坑可以总结如下：

| 常见坑 | 典型表现 | 后果 | 规避方式 |
|---|---|---|---|
| JSON 格式错误 | 输出自然语言或对象而非数组 | 评测器无法解析 | 强制提示 `output JSON array only` |
| 缺少 `parameters` | 只给工具名，不给参数对象 | AST 直接失败 | 固定模板要求 `name + parameters` |
| 参数格式错 | 日期、枚举、字段名不符合要求 | 工具选对但调用错 | 复用样例并对参数做 schema 检查 |
| 多工具链超时或截断 | 只输出前几步 | 链条不完整，Pass 下降 | 从短链开始训练与测试 |
| 顺序错误 | 先执行后查询 | 规划能力被判失败 | 在提示中明确“按依赖顺序输出” |

真实工程里，另一个坑是“评测好看，系统不好用”。原因是 ToolBench 主要检查离线结构，不模拟线上接口异常、权限失败、空返回、重试退避等问题。所以它很适合做能力基准，不适合单独作为上线依据。做产品时，离线结构评测和在线鲁棒性测试必须分开。

---

## 替代方案与适用边界

如果你的目标只是验证一个固定工作流是否通顺，自建模拟工具链往往更便宜。所谓模拟工具链，白话说，就是自己写几组简化 API 和标准答案，例如“天气查询 + 日程创建”两步流程。这种方式便于快速迭代提示词，也容易控制数据质量，但覆盖面很窄。

例如你可以自己做一套 2 类任务、20 条样本的小基准，专门检查公司内部助手是否会调用 CRM 和工单系统。这对单一业务很有效，但不能说明模型在开放工具集上是否有泛化能力。因为从 2 类任务泛化到 49 类真实 API，是完全不同的难度层级。

另一类替代是 AgentBench 等更偏“代理行为”的平台。它们更强调多轮决策、环境交互和整体任务完成度，适合测“代理是否能坚持完成任务”。但如果你关心的是“工具名选没选对”“参数层级对不对”，ToolBench 这种结构化基准通常更直接。

可以做一个横向比较：

| 方案 | 覆盖范围 | 是否有标准调用序列 | 评价重点 | 适用场景 |
|---|---|---|---|---|
| ToolBench | 大规模多域真实 API | 有 | 工具选择、结构对齐、链式规划 | 通用工具学习能力评测 |
| 自建 mock 工具链 | 小范围、强定制 | 通常有 | 业务闭环正确性 | 内部场景快速验证 |
| AgentBench 类平台 | 更偏任务环境 | 不一定聚焦调用序列 | 多轮代理策略与完成度 | 端到端代理能力测试 |

所以边界很清楚：如果你要回答“模型在开放工具世界里会不会选、会不会调、会不会排顺序”，ToolBench 很合适；如果你要回答“线上系统稳定不稳定、异常恢复行不行”，还需要补充在线测试、沙箱执行和真实观测。

---

## 参考资料

下表列出三个常用来源及其用途：

| 来源 | 覆盖内容 | 用途说明 |
|---|---|---|
| ToolBench GitHub README | 数据规模、API 来源、任务构造方式 | 用于确认 16,464 个真实 API 与整体数据设计 |
| EmergentMind ToolBench Evaluation | 评价流程、指标解释、典型用法 | 用于理解 Pass Rate、Win Rate、AST Accuracy 等含义 |
| mcpbr ToolBench 指南 | 输出格式、最佳实践、常见错误 | 用于参考 JSON 数组输出形式与评测实践建议 |

1. ToolBench GitHub README：说明基准的真实 API 规模、领域覆盖和任务生成方式，适合理解它为什么能被视为工具学习 benchmark。  
2. EmergentMind 的 ToolBench Evaluation 综述：适合快速建立指标层面的整体认识，尤其是结构化评测与工具选择的区别。  
3. mcpbr 的 ToolBench 页面：更偏实操，能帮助理解输出格式约束、常见错误和提示模板设计。

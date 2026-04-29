## 核心结论

BFCL 和 ToolBench 都在评测大模型的工具调用能力，但它们回答的不是同一个问题。

一句话结论：**BFCL 更适合判断“模型会不会按正确结构调用工具”，ToolBench 更适合判断“模型能不能把真实任务链真正完成”。**

这里的“工具调用能力”，白话说就是模型不只会聊天，还会像程序一样去调用外部函数、API 或系统能力。真正有价值的评测，不是看它有没有说出某个函数名，而是看它是否同时做到四件事：选对工具、填对参数、按对顺序调用、失败后能恢复。

先看一个新手版理解：

| 评测 | 可以把它理解成什么 | 主要看什么 |
|---|---|---|
| BFCL | 结构化单元测试 | 函数名、参数名、参数值、调用结构是否正确 |
| ToolBench | 任务回放测试 | 一串 API 调用后，事情是否真的办成 |

玩具例子很简单。假设任务是“查北京明天天气”。  
如果模型输出 `weather(city="Beijing", date="tomorrow")`，BFCL 更关心这条调用的结构是不是和参考答案一致。  
如果任务变成“查北京明天天气，再在晚上 9 点创建提醒”，ToolBench 更关心的是整个流程是否跑通，而不只是第一步有没有写出 `weather(...)`。

真实工程例子更能说明差别。企业助手里，一个用户说：“查明天北京天气，再创建提醒。”  
BFCL 看的是模型是否正确生成类似 `weather(city="北京", date="tomorrow")` 和 `create_reminder(time="21:00", content="查看北京天气")` 的调用结构。  
ToolBench 看的是模型能不能先查询天气、再基于结果组织提醒内容、再成功调用提醒接口，最终把任务完成。

所以两者不是谁替代谁，而是职责不同：

| 你想回答的问题 | 更合适的评测 |
|---|---|
| 模型会不会调用工具 | BFCL |
| 模型能不能完成真实工作流 | ToolBench |

---

## 问题定义与边界

先把“工具调用评测”定义清楚。

普通问答评测只关心答案对不对。比如“北京是中国首都吗”，只看模型回答“是”是否正确。  
工具调用评测不同，它评的是一个过程。输入通常是用户自然语言请求和一组可用工具定义，输出不是一段自由文本，而是一条或多条结构化调用，以及调用后的任务结果。

可以用下面这个表格固定定义边界：

| 项目 | 普通问答 | 工具调用评测 |
|---|---|---|
| 评测对象 | 文本回答能力 | 工具使用能力 |
| 输入 | 问题文本 | 问题文本 + 工具 schema |
| 输出 | 自然语言答案 | 函数调用、参数、调用序列 |
| 判定方式 | 答案是否正确 | 工具选择、参数、顺序、结果是否正确 |

这里有几个术语需要先定边界。

“结构化函数调用”是指模型输出的不是一句描述，而是机器可解析的调用格式，比如 JSON 或函数签名。白话说，就是程序能直接拿去执行的调用。  
“单轮调用”是用户给一个请求，模型给出一次调用，评测立即结束。  
“多轮调用”是模型需要根据前一步工具返回结果继续决定下一步。  
“真实 API 任务链”是多个工具按顺序协作，直到完成一个完整任务，而不是只验证某个局部步骤。

这也是 BFCL 和 ToolBench 的主要边界差异：

| 维度 | BFCL | ToolBench |
|---|---|---|
| 核心对象 | 结构化调用 | 真实任务链 |
| 常见场景 | 单轮或局部调用正确性 | 多步 API 规划与执行 |
| 主要风险 | 结构看似正确但任务不一定完成 | 任务完成但局部步骤未必最优 |

还要明确一个容易被忽略的边界：这两类评测都不是“通用智能”的完整测量。它们测的是模型在给定工具定义、给定任务分布、给定调用预算下的工具使用表现。换句话说，分数高不等于模型在所有业务里都可靠。

举个边界例子。假设某天气工具原来的参数名叫 `city_name`，后来改成 `location`。同一个模型的底层能力没变，但如果它仍输出旧参数名，BFCL 分数会明显下降。这个结果说明它对当前 schema 的适配差，而不是说明它突然不会理解天气任务了。ToolBench 也类似，如果任务集偏向搜索类 API，模型在搜索任务上得分高，不代表它在办公自动化或数据库操作场景也同样强。

---

## 核心机制与推导

### 1. BFCL 在测什么

BFCL 的核心不是字符串匹配，而是结构化比对。  
AST，抽象语法树，白话说就是把“函数名、参数名、参数值、嵌套关系”拆成一棵结构树，再判断两棵树是否一致。

这件事为什么重要？因为同一个调用可以有不同表面写法，但结构可能一样；反过来，表面上都提到了同一个函数名，结构却可能错得很严重。

最小例子：

- 参考答案：`weather(city="北京", date="tomorrow")`
- 预测一：`weather(date="tomorrow", city="北京")`
- 预测二：`weather(location="北京", date="tomorrow")`

如果评测器只做字符串精确匹配，预测一可能被误判为错，因为参数顺序不同。  
如果评测器做 AST 匹配，预测一通常可判为对，因为函数和键值对结构一致。  
预测二则要看 schema 是否允许 `location` 这个参数名；如果不允许，结构上就是错的。

BFCL 常用抽象可以写成：

\[
s_i=1 \iff \text{pred\_AST}_i = \text{ref\_AST}_i,\qquad
\mathrm{Acc}=\frac{1}{N}\sum_{i=1}^{N}s_i
\]

其中 $s_i$ 表示第 $i$ 个样本是否匹配成功，$N$ 是总样本数。  
如果 5 条样本里 4 条 AST 完全一致，那么：

\[
\mathrm{Acc}=\frac{4}{5}=80\%
\]

这就是一个标准的玩具例子。它说明 BFCL 的分数本质上是在回答：**模型输出的调用结构和标准答案有多一致。**

### 2. ToolBench / ToolEval 在测什么

ToolBench 的重点不是结构本身，而是任务完成度。  
可以把它理解成一个带工具环境的任务集合，模型需要在调用预算内选择工具、组织步骤、执行多次 API 调用，最后把任务做完。

这里的“调用预算”，白话说就是允许模型最多调用多少次工具，防止它无限尝试。  
一个任务是否成功，往往取决于最终状态，而不只是中间某一步是否漂亮。

常见抽象公式是：

\[
\mathrm{PassRate}=\frac{N_{\text{success}}}{N},\qquad
\mathrm{WinRate}=\frac{N_{\text{preferred}}}{N_{\text{pairs}}}
\]

其中：

- $N_{\text{success}}$ 是成功完成的任务数
- $N$ 是总任务数
- $N_{\text{preferred}}$ 是在两份候选解中被偏好评审器选中的次数
- $N_{\text{pairs}}$ 是比较对数

如果 100 个任务中有 68 个在调用预算内完成，那么：

\[
\mathrm{PassRate}=68\%
\]

这就是 ToolBench 风格的最小数值例子。

### 3. 为什么两者会得出不同结论

因为它们的“正确”定义不同。

| 评测 | 正确的含义 | 典型失败 |
|---|---|---|
| BFCL | 调用结构与参考答案一致 | 函数名对但参数错 |
| ToolBench | 整体任务完成 | 中间步骤合理但最终没完成 |

还是用“查明天北京天气，再创建提醒”这个真实工程例子说明。

在 BFCL 里，你可能把两条函数调用都写出来了，名字和参数也都对，得分不错。  
但在 ToolBench 里，如果你先创建了提醒、后查天气，或者天气查询失败后没有重试，最终任务就可能失败。也就是说，**结构正确不等于任务完成。**

反过来，ToolBench 里也可能出现另一种情况：模型通过一次模糊搜索拿到了结果，最终任务成功，但中间调用并不规范。如果只看最终任务，它可能过关；但从工程治理角度看，这种“歪打正着”不一定值得鼓励。

如果要画成流程图，最简单的路径是：

用户请求 -> 工具选择 -> 参数填充 -> 调用结果 -> 下一步决策 -> 最终任务完成

BFCL 主要盯前半段的结构正确性，ToolBench 更重后半段的链路完成度。

---

## 代码实现

评测实现的重点不是训练模型，而是把“输入、解析、比对、统计”这四步做扎实。只要这四步设计得清楚，工具 schema 变化时，脚本才不至于整套失效。

先看一个 BFCL 风格的最小可运行实现。这个例子不追求覆盖全部语法，只演示“把函数调用解析为结构，再比较”的核心思想。

```python
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class ToolCall:
    name: str
    args: tuple

def parse_tool_call(text: str) -> ToolCall:
    m = re.fullmatch(r"\s*([a-zA-Z_]\w*)\((.*)\)\s*", text)
    if not m:
        raise ValueError(f"invalid call: {text}")
    name, raw_args = m.group(1), m.group(2).strip()
    if not raw_args:
        return ToolCall(name=name, args=tuple())

    parts = [p.strip() for p in raw_args.split(",")]
    kvs = []
    for part in parts:
        k, v = [x.strip() for x in part.split("=", 1)]
        # 去掉简单字符串引号
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
            v = v[1:-1]
        kvs.append((k, v))

    # 按参数名排序，避免参数顺序影响结构判断
    return ToolCall(name=name, args=tuple(sorted(kvs)))

def bfcl_score(pred: str, ref: str) -> int:
    return int(parse_tool_call(pred) == parse_tool_call(ref))

# 玩具样本
pred = 'weather(date="tomorrow", city="北京")'
ref = 'weather(city="北京", date="tomorrow")'
assert bfcl_score(pred, ref) == 1

pred2 = 'weather(location="北京", date="tomorrow")'
assert bfcl_score(pred2, ref) == 0

samples = [
    ('weather(city="北京", date="tomorrow")', 'weather(city="北京", date="tomorrow")'),
    ('weather(date="tomorrow", city="北京")', 'weather(city="北京", date="tomorrow")'),
    ('create_reminder(time="21:00", content="看天气")', 'create_reminder(time="21:00", content="看天气")'),
    ('weather(location="北京", date="tomorrow")', 'weather(city="北京", date="tomorrow")'),
    ('calendar_add(title="开会", time="10:00")', 'calendar_add(title="开会", time="10:00")'),
]

acc = sum(bfcl_score(p, r) for p, r in samples) / len(samples)
assert acc == 0.8
print("BFCL-style accuracy:", acc)
```

这段代码做了四件事：

| 评测步骤 | 输入 | 输出 | 常见失败点 |
|---|---|---|---|
| 读入模型输出 | 原始字符串 | 待解析文本 | 输出不是合法调用格式 |
| 解析调用 | 函数签名文本 | 结构化对象 | 参数分隔、引号、嵌套处理不完整 |
| 与参考答案比对 | 预测结构 + 参考结构 | `0/1` 分数 | 忽略 schema 约束导致误判 |
| 汇总统计 | 每条样本分数 | 总准确率 | 混合不同任务类型导致平均值失真 |

再看一个 ToolBench 风格的极简伪实现。重点是记录调用链和预算控制，而不是把任务描述写得多漂亮。

```python
def run_api_chain(task, model_plan, call_budget=10):
    calls = 0
    state = {"weather": None, "reminder_created": False}

    for step in model_plan:
        if calls >= call_budget:
            return False
        calls += 1

        if step["tool"] == "weather":
            if step["args"] == {"city": "北京", "date": "tomorrow"}:
                state["weather"] = "sunny"
            else:
                return False

        elif step["tool"] == "create_reminder":
            if state["weather"] is None:
                return False
            if "content" in step["args"] and "time" in step["args"]:
                state["reminder_created"] = True
            else:
                return False

        else:
            return False

    return state["weather"] is not None and state["reminder_created"]

plan = [
    {"tool": "weather", "args": {"city": "北京", "date": "tomorrow"}},
    {"tool": "create_reminder", "args": {"time": "21:00", "content": "查看北京天气"}},
]

assert run_api_chain("查明天北京天气，再创建提醒", plan, call_budget=5) is True
```

真实工程中，建议把评测模块拆开：

1. `schema_loader`：固定并加载工具定义版本  
2. `call_parser`：把模型输出解析成统一结构  
3. `executor`：在沙盒环境里执行工具调用  
4. `judge`：判断 AST 匹配或任务成功  
5. `reporter`：汇总准确率、成功率、失败原因

如果团队经常改工具接口，最好额外维护一个配置片段来锁定版本，例如：

```python
EVAL_CONFIG = {
    "schema_version": "weather_v3_2026_04_01",
    "taskset_version": "tool_tasks_r12",
    "call_budget": 10,
    "metrics": ["tool_select_acc", "arg_acc", "order_acc", "pass_rate"],
}
assert EVAL_CONFIG["call_budget"] > 0
```

这样做的价值很直接：同一轮评测到底是在比较模型能力，还是在比较接口变化，后续就能说清楚。

---

## 工程权衡与常见坑

最常见的误区，是把“输出了函数名”误当成“会用工具”。这会显著高估模型能力。

比如模型输出了 `weather(...)`，但把 `date="tomorrow"` 写成了 `date="next day"`，或者把城市写成英文缩写而后端只接受中文，真实任务仍然会失败。对线上系统来说，参数错误和顺序错误的破坏性，通常比“函数名完全错了”还更隐蔽。

下面是工程里常见的坑：

| 常见坑 | 影响 | 规避方式 |
|---|---|---|
| 只统计函数名是否命中 | 高估能力 | 分开统计工具选择、参数、顺序 |
| schema 改名未锁版本 | 分数波动难解释 | 固定工具定义并记录版本号 |
| 只测单轮不测恢复 | 低估线上失败率 | 增加失败重试与异常返回样本 |
| 任务集过于单一 | 结果无法外推 | 按场景分层抽样 |
| 只看平均分 | 掩盖局部灾难 | 分任务类型、分工具族群报告 |

可以把离线样例和真实失败恢复对比着看：

| 场景 | 离线样例常见设置 | 真实工程常见情况 |
|---|---|---|
| 天气查询 | 参数总是干净完整 | 城市名别称、时间表达含糊 |
| 多步链路 | 每步都成功 | 中间接口超时、限流、空结果 |
| 提醒创建 | 接口一定可用 | 用户权限不足、字段校验失败 |

一个典型真实业务例子是客服办公助手。用户说：“查一下客户 A 上周工单状态，再给我生成跟进提醒。”  
离线样例里，客户名称、时间范围、工单系统接口都很标准。  
线上却可能出现三种问题：

1. 客户 A 有重名，先要 disambiguation，也就是先澄清到底是哪一个  
2. 上周的时间范围要根据时区和周起始日转换  
3. 工单系统查询为空时，模型不能直接编造提醒内容，而要降级回复或要求补充信息

如果评测里没有这些失败恢复样本，模型离线分数再高，上线也可能很脆。

实际落地时，建议至少分开统计四类指标：

| 指标 | 含义 |
|---|---|
| 选对工具 | 该用哪个工具时是否选对 |
| 参数正确 | 参数名、参数值、类型是否正确 |
| 顺序正确 | 多步任务里先后顺序是否合理 |
| 任务完成率 | 最终目标是否完成 |

这四类指标不能互相替代。原因很简单：任务完成率低时，你需要知道问题出在工具选择、参数填充，还是失败恢复；否则团队只会看到一个总分，却不知道该修哪里。

---

## 替代方案与适用边界

BFCL 和 ToolBench 更准确的关系，不是竞品，而是两个层次。

BFCL 解决的是“结构正确性”问题。  
ToolBench 解决的是“任务完成度”问题。

如果你的目标只是做函数调用回归测试，比如升级模型版本后，检查它是否仍然按既定 schema 输出，那么 BFCL 这类评测性价比很高。它快、可重复、结果清晰，适合做 CI 里的结构化检查。  
如果你的目标是验证企业助手、办公代理、API 编排代理是否真的能把工作流跑通，那么 ToolBench 类评测更接近生产现实。

可以用一个决策表来选：

| 目标 | 推荐评测 | 理由 |
|---|---|---|
| 检查函数调用格式是否回归 | BFCL | 直接比较结构，定位清晰 |
| 检查多步任务是否跑通 | ToolBench | 覆盖调用链与执行结果 |
| 接口改版后的兼容性验证 | 先 BFCL | 先看 schema 对齐是否出问题 |
| 上线前业务验收 | 先 BFCL，再 ToolBench | 先保结构，再验任务完成 |

这里再给一个新手版理解：

- 如果你只想确认“模型会不会按格式吐出正确调用”，BFCL 足够。
- 如果你要确认“企业助手能不能真的把事办完”，ToolBench 更合适。

也存在替代或补充路线，但都各有边界：

| 替代思路 | 能解决什么 | 不能替代什么 |
|---|---|---|
| 手写单元测试集 | 验证少量关键调用 | 覆盖面有限，难测复杂任务链 |
| 线上 A/B 观测 | 看到真实用户效果 | 成本高，定位慢，风险大 |
| 合成任务集 | 快速扩充样本 | 容易和真实分布脱节 |

更稳妥的组合方式通常是两段式：

1. 先用 BFCL 做结构回归，保证函数名、参数名、参数值和基础调用结构没有明显退化。  
2. 再用 ToolBench 类任务做链路验证，确认多步 API 任务在预算内真的能完成。

这个组合有现实意义。API 调用框架改版后，先跑 BFCL 可以很快发现“参数名变了、字段漏了、格式错了”这类低层问题；产品上线前，再跑 ToolBench 风格任务，可以发现“顺序不对、遇错不恢复、预算耗尽”这类高层问题。

结论还是那句：**BFCL 更像结构层体检，ToolBench 更像流程层验收。生产级 agent 往往两者都需要。**

---

## 参考资料

下面这些资料最好分两轮读：先读论文理解“为什么这样测”，再读仓库 README 理解“实际怎么跑”。

| 资料名 | 作用 | 推荐阅读顺序 |
|---|---|---|
| BFCL 论文 | 理解定义、动机、指标 | 1 |
| BFCL README | 看数据格式和运行逻辑 | 2 |
| ToolBench README | 看任务集、工具集、评测流程 | 3 |
| ToolLLM 论文 | 补背景，理解 ToolBench 所在脉络 | 4 |

1. [BFCL: The Berkeley Function-Calling Leaderboard](https://proceedings.mlr.press/v267/patil25a.html)
2. [BFCL 官方仓库 README](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md)
3. [ToolBench 官方仓库 README](https://github.com/OpenBMB/ToolBench)
4. [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)

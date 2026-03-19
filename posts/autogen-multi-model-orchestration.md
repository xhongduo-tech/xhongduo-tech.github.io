## 核心结论

AutoGen 里所谓“多模型混合编排”，核心不是让框架自动判断哪个模型最聪明，而是开发者先把“谁负责什么、优先用谁、失败后换谁”写进配置。`config_list` 本质上就是一个按顺序排列的模型候选池；`filter_dict` 本质上就是筛选条件，用来把候选池切成“推理专用”“编码专用”“摘要专用”等子集。

如果你把推理 Agent 配成 `GPT-4 -> GPT-3.5`，把编码 Agent 配成 `Claude -> GPT-3.5`，把摘要 Agent 只配成 `GPT-3.5`，那么系统行为就很稳定：高难任务优先吃高能力模型，简单收尾任务固定走便宜模型，主模型超时或限流时再顺位回退。这样做的目标不是“绝对最优”，而是把质量、成本、可用性放进同一个可控结构里。

下表是最常见的配置方式：

| Agent | 主模型 | 备选顺位 | 主要目标 |
|---|---|---|---|
| 推理 Agent | GPT-4 级模型 | GPT-3.5 级模型 | 保证复杂分析质量 |
| 编码 Agent | Claude 级模型 | GPT-3.5 级模型 | 提升代码生成结构性 |
| 摘要 Agent | GPT-3.5 级模型 | 无或同级备份 | 压低成本 |
| 编排/审稿 Agent | 高质量模型 | 中质量模型 | 汇总多 Agent 结果 |

从机制上可以写成：

$$
L_F = \mathrm{filter}(L, F)
$$

其中 $L$ 是全局模型列表，$F$ 是某个 Agent 的筛选条件。真正调用时按顺序执行：

$$
\text{for } i=1..|L_F|,\ \text{若 } call(L_F[i]) \text{ 成功则返回，否则继续}
$$

结论很直接：顺序决定优先级，过滤决定权限，失败回退决定系统韧性。

---

## 问题定义与边界

要先把问题说清楚。这里讨论的不是“一个 Agent 如何思考”，而是“多个 Agent 在协作时，各自应该拿到什么模型资源”。“编排”这个词的白话解释是：提前规定任务分工和调用路径，而不是临场乱选。

多 Agent 系统里，至少有三个矛盾同时存在：

| 目标 | 需要什么 | 常见冲突 |
|---|---|---|
| 推理质量 | 更强模型 | 更贵、更慢 |
| 工程可用性 | fallback | 逻辑更复杂 |
| 成本控制 | 低价模型 | 容易拉低上游质量 |

因此，多模型混排解决的是一个资源分配问题，而不是一个提示词问题。你要回答的是：

1. 哪类 Agent 必须用强模型？
2. 哪类 Agent 可以稳定降级？
3. 哪类失败允许回退，哪类失败必须中断？

玩具例子最容易理解。假设你只有三个 Agent：

- `reasoner`：负责解数学题思路
- `coder`：负责把思路写成 Python
- `summarizer`：负责把结果压缩成 3 句话

这时共享配置池可以是：

- `gpt-4`，标签 `["reasoner"]`
- `claude-3.5`，标签 `["coder"]`
- `gpt-3.5-turbo`，标签 `["reasoner", "coder", "summarizer", "fallback"]`

推理 Agent 过滤后看到 `[gpt-4, gpt-3.5-turbo]`；编码 Agent 过滤后看到 `[claude-3.5, gpt-3.5-turbo]`；摘要 Agent 过滤后只看到 `[gpt-3.5-turbo]`。这已经足够形成一套可运行的多模型策略。

边界也要讲清楚。AutoGen 0.2 文档里，`llm_config + config_list` 是标准做法；在较新的 stable/0.4 系列文档里，更常见的是直接给每个 Agent 传入不同的 `model_client`。两种方式表达的是同一个工程思想：不同 Agent 绑定不同模型资源。本文重点分析前者，因为它最直接体现“排序、过滤、回退”三件事。

---

## 核心机制与推导

先看最小机制。设全局配置列表为：

$$
L=[C_1, C_2, \dots, C_n]
$$

每个配置 $C_i$ 至少包含 `model`，通常还会带 `api_key`、`base_url`、`tags`。某个 Agent 的筛选条件为 $F$，那么它实际能用的列表是：

$$
L_F=\{C_i \in L \mid C_i \text{ 满足 } F\}
$$

这里“满足”的白话解释是：配置项里的字段和 Agent 要求对得上。比如 `filter_dict={"tags":["reasoner"]}` 表示只保留带 `reasoner` 标签的配置。

接下来是优先级。AutoGen 0.2 官方文档明确说明，Agent 会先使用 `config_list` 中第一个可用模型；若该模型失败，再尝试第二个，以此类推。也就是说，列表顺序本身就是策略。

伪代码可以写成：

```text
enabled_list = filter(config_list, filter_dict)
if enabled_list is empty:
    raise ConfigError

for config in enabled_list:
    try:
        result = call_llm(config)
        return result
    except transient_error:
        continue

raise AllModelsFailed
```

这里有两个容易忽略的推论。

第一，`filter` 比 `fallback` 更基础。因为如果筛选错了，Agent 根本看不到正确模型，后面的回退机制没有意义。比如摘要 Agent 不应该能看到昂贵推理模型，否则它会把预算偷偷吃掉。

第二，顺序只在“筛选之后”才有意义。全局列表里把 `gpt-4` 写在第一位，并不代表每个 Agent 都会先调用它；只有当该 Agent 的过滤结果里仍包含它，它才是第一顺位。

再看一个玩具例子。推理 Agent 的有效列表是：

$$
L_{\text{reasoner}}=[\text{GPT-4}, \text{GPT-3.5}]
$$

如果 `call(GPT-4)` 成功，流程结束；如果限流、超时或临时失败，再尝试 `GPT-3.5`。这里的回退不是“质量等价替换”，而是“服务连续性优先”的替换。所以你不能把 fallback 当成无损替身，它只是让系统不断电。

真实工程例子更能看出价值。假设你在做一个代码助理工作流：

1. 需求拆解 Agent 负责理解用户需求。
2. 实现 Agent 负责写代码。
3. Review Agent 负责找 bug 和边界条件。
4. Summary Agent 负责生成变更摘要。

如果全部都上最强模型，质量高但费用高；如果全部都上便宜模型，成本低但 Review 和需求拆解容易失真。更合理的做法是：

- 拆解 Agent：高质量模型优先，保证问题定义准确
- 实现 Agent：偏代码能力强的模型优先
- Review Agent：高质量模型优先，因为它负责兜底
- Summary Agent：低成本模型即可

这时，多 Agent 协作质量并不只取决于单个模型强弱，而取决于“高能力模型是否放在真正高杠杆的位置上”。

---

## 代码实现

下面先用纯 Python 写一个可运行的简化版，模拟 `filter + 顺位回退` 机制。这个例子不依赖真实 API，但逻辑和 AutoGen 的配置思想一致。

```python
from typing import List, Dict, Any

config_list = [
    {"model": "gpt-4", "tags": ["reasoner"], "ok": False},
    {"model": "gpt-3.5-turbo", "tags": ["reasoner", "summarizer", "fallback"], "ok": True},
    {"model": "claude-3.5", "tags": ["coder"], "ok": True},
]

def filter_config(configs: List[Dict[str, Any]], filter_dict: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    if "tags" not in filter_dict:
        return configs[:]
    wanted = set(filter_dict["tags"])
    return [c for c in configs if wanted.intersection(set(c.get("tags", [])))]

def call_model(config: Dict[str, Any], prompt: str) -> str:
    if not config["ok"]:
        raise RuntimeError(f"{config['model']} temporary failure")
    return f"{config['model']} handled: {prompt}"

def run_with_fallback(configs: List[Dict[str, Any]], prompt: str) -> str:
    if not configs:
        raise ValueError("empty config list")
    last_error = None
    for config in configs:
        try:
            return call_model(config, prompt)
        except RuntimeError as e:
            last_error = e
    raise RuntimeError(f"all models failed: {last_error}")

reasoner_list = filter_config(config_list, {"tags": ["reasoner"]})
coder_list = filter_config(config_list, {"tags": ["coder"]})
summarizer_list = filter_config(config_list, {"tags": ["summarizer"]})

assert [c["model"] for c in reasoner_list] == ["gpt-4", "gpt-3.5-turbo"]
assert [c["model"] for c in coder_list] == ["claude-3.5"]
assert [c["model"] for c in summarizer_list] == ["gpt-3.5-turbo"]

result = run_with_fallback(reasoner_list, "solve 2x+3=7")
assert result.startswith("gpt-3.5-turbo handled:")
print(result)
```

上面这个例子说明三件事：

- `reasoner` 会先看到 `gpt-4`
- `gpt-4` 失败后才会退到 `gpt-3.5-turbo`
- `summarizer` 根本看不到 `gpt-4`

再看贴近 AutoGen 0.2 的写法。下面是结构示意，重点是配置思想：

```python
import autogen

all_configs = autogen.config_list_from_json("OAI_CONFIG_LIST")

reasoner_configs = autogen.filter_config(
    all_configs,
    {"tags": ["reasoner", "fallback"]}
)

coder_configs = autogen.filter_config(
    all_configs,
    {"tags": ["coder", "fallback"]}
)

summarizer_configs = autogen.filter_config(
    all_configs,
    {"tags": ["summarizer"]}
)

reasoner = autogen.AssistantAgent(
    name="reasoner",
    llm_config={"config_list": reasoner_configs}
)

coder = autogen.AssistantAgent(
    name="coder",
    llm_config={"config_list": coder_configs}
)

summarizer = autogen.AssistantAgent(
    name="summarizer",
    llm_config={"config_list": summarizer_configs}
)
```

对应的 `OAI_CONFIG_LIST` 可以长这样：

```json
[
  {"model": "gpt-4", "api_key": "...", "tags": ["reasoner"]},
  {"model": "claude-3.5", "api_key": "...", "base_url": "...", "tags": ["coder"]},
  {"model": "gpt-3.5-turbo", "api_key": "...", "tags": ["reasoner", "coder", "summarizer", "fallback"]}
]
```

真实工程例子可以设成“PR 自动处理流水线”：

- `reasoner`：读需求单和历史讨论，产出改动计划
- `coder`：生成代码补丁
- `reviewer`：检查回归风险
- `summarizer`：输出变更说明和发布摘要

这类场景里，`reasoner` 和 `reviewer` 的错误代价高，应该优先高质量模型；`summarizer` 只是压缩信息，低价模型就足够。这样分配，比“一刀切全用同一个模型”更接近实际团队的资源配置方式。

---

## 工程权衡与常见坑

多模型编排不是白拿收益，它本质上是在“策略复杂度”换“成本和可用性控制”。常见坑通常不在 API 层，而在配置设计层。

| 常见坑 | 结果 | 规避措施 |
|---|---|---|
| 没有给 Agent 做标签过滤 | 廉价任务误用高价模型，成本失控 | 每类 Agent 单独定义 `tags` |
| 主备顺序写反 | 高能力 Agent 长期跑在低能力模型上 | 把主模型放在筛选结果首位 |
| fallback 过宽 | 失败时掉到完全不适合的模型 | fallback 只给能力接近或可接受的替代项 |
| 配置筛选后为空 | 运行时报错或启动即失败 | 初始化阶段校验 `L_F` 非空 |
| 误把 fallback 当质量兜底 | 结果可用但精度下降 | 对关键节点增加 review 或终止条件 |

最常见的错误就是“顺序错位”。比如你想让推理 Agent 主要用 GPT-4，但列表写成 `[GPT-3.5, GPT-4]`。在官方 0.2 机制下，第一个能成功的模型就会被使用，后面的 GPT-4 根本不会出场。这不是“偶尔降级”，而是“永久主力写错了”。

第二个高频问题是标签设计过粗。很多团队一开始只写 `["prod"]`、`["backup"]` 这种标签，看起来简单，实际上没有表达任务差异。更实用的标签通常是按职责写，例如：

- `reasoner`
- `coder`
- `reviewer`
- `summarizer`
- `fallback`

“职责标签”比“环境标签”更能直接决定 Agent 行为。

还有一个现实问题是能力差异会放大协作误差。多 Agent 系统不是线性流水线，而是误差传播链。上游推理 Agent 若输出错误计划，下游编码 Agent 再强，也只能把错误方案写得更完整。于是高质量模型最该放的位置通常不是“字写得最多的那个 Agent”，而是“决定方向的那个 Agent”。

---

## 替代方案与适用边界

如果你的系统很小，只有一个 Agent，那么多模型混排通常不值得。单模型直连更简单，监控也更清楚。只有当任务被明确拆成不同职责，而且这些职责对模型能力的需求明显不同，多模型编排才开始划算。

一种替代方案是“单模型固定绑定”。比如所有 Agent 都只用一个高质量模型。优点是行为一致、调试简单；缺点是贵，而且没有天然的跨模型 fallback。适合高价值、低并发、对延迟不敏感的场景。

另一种替代方案是新版本 AutoGen 常见的写法：不给每个 Agent 传 `config_list`，而是直接给不同 Agent 传不同 `model_client`。这等价于在代码层显式绑定模型。优点是边界清楚；缺点是共享配置池和顺位回退没有 0.2 那么直观，需要你自己组织更多结构。

再往上走，就是 Mixture of Agents 这类分层架构。“分层”的白话解释是：第一层先各自做初步回答，第二层再综合上一层结果，像多轮精炼。官方示例里，worker agent 和 orchestrator agent 都是显式绑定 `model_client` 的；如果你要在工程里混用不同模型，可以让不同 worker 类型绑定不同 client，或者在旧式 `config_list` 思路里为每一层准备不同过滤规则。

例如两层结构可以这样理解：

- 第一层 `layer1_filter={"tags":["reasoner"]}`：负责拆题、提出方案
- 第二层 `layer2_filter={"tags":["coder", "summarizer"]}`：负责实现和收敛

适用边界可以概括为：

| 方案 | 适合场景 | 不适合场景 |
|---|---|---|
| 单模型直连 | 系统简单、要求稳定 | 成本敏感、职责差异大 |
| `config_list` 多模型编排 | 0.2 风格项目、需要排序和 fallback | 团队不愿维护配置策略 |
| 显式多 `model_client` | 新版 AutoGen、边界明确 | 想复用统一候选池 |
| 分层 Mixture of Agents | 复杂任务、多阶段精炼 | 小任务、低收益场景 |

最终判断标准不是“能不能混模型”，而是“模型差异是否刚好对应任务差异”。如果没有这种对应关系，多模型只会带来额外复杂度。

---

## 参考资料

- AutoGen 0.2 官方 LLM Configuration：`config_list`、`filter_config`、顺位调用与失败回退  
  https://microsoft.github.io/autogen/0.2/docs/topics/llm_configuration/

- AutoGen 0.2 官方 `OpenAIWrapper` 参考：`config_list` 作为多个配置项传入  
  https://autogenhub.github.io/autogen/docs/reference/oai/client/

- AutoGen stable 官方 Model Clients：新版更常见的是按 Agent 直接绑定 `model_client`  
  https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-clients.html

- AutoGen stable 官方 Mixture of Agents：分层 worker/orchestrator 设计模式  
  https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html

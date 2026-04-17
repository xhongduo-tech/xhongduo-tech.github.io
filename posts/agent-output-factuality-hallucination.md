## 核心结论

Agent 的事实性验证不是只检查“最后一句话对不对”，而是要同时检查两层对象：

1. 工具调用是否合理，也就是参数、目标对象、返回解释是否和外部事实一致。
2. 最终回答是否有依据，也就是回答里的结论能不能被工具结果、知识库或多次独立采样支持。

这里的“幻觉”可以理解为：模型给出了看起来完整、语气确定、但没有可靠依据的调用或结论。它不只出现在自然语言里，也会出现在工具参数里。例如把 `city=Shanghai` 写成 `city=ShangHaii`，或者把本该查天气的任务错误地路由到汇率工具，这些都属于 Agent 层面的事实性错误。

从工程效果看，三类检测手段最常见，也最互补：

| 检测策略 | 检查对象 | 擅长覆盖的场景 | 延迟影响 |
|---|---|---|---|
| 知识库/世界模型核查 | 工具调用、最终结论 | 稳定事实、结构化参数、工具元数据 | 中等 |
| 自一致性 SC@k | 最终结论、推理路径 | 可重复求解的问题、偶发性幻觉 | 高 |
| token 级置信度 | 生成文本局部片段 | 编造式续写、低把握表达 | 低 |

一个适合初学者的玩具例子是天气查询。Agent 回答“上海今天晴”，但世界知识缓存里记录 `weather(Shanghai)=Rainy`，同时 SC@5 的五次采样里有四次都回答“Rainy”。这时即使单次生成看上去很流畅，也应判为高风险输出，而不是直接发给用户。

真实工程里，更稳妥的做法不是三选一，而是组合：先做知识库校验拦截显式错误，再对高风险任务做 SC@k 投票，最后用 token 级置信度给出局部风险提示。这样做的目标不是“彻底消灭幻觉”，而是把错误压缩到可监控、可回滚、可人工介入的范围。

---

## 问题定义与边界

先定义对象。设 Agent 在一次任务中生成了若干工具调用和若干文本结论。把每一次工具调用记为 $z$，把世界知识或外部可验证事实集合记为 $\mathcal{K}$。最简单的事实一致性判断可以写成：

$$
\mathbbm{1}_{z \in \mathcal{K}} =
\begin{cases}
1, & z\ \text{能被知识库或世界描述支持} \\
0, & z\ \text{无法被支持或与之冲突}
\end{cases}
$$

这里的“知识库”不是玄学概念，它就是一份可查证的事实集合。对白话解释来说，它相当于“系统当前愿意信任的外部真相来源”，可以是工具 schema、缓存结果、数据库记录、知识图谱，或者人工维护的白名单规则。

问题边界必须讲清楚，否则检测系统会被误用。

第一，知识库不等于真理本身。它只代表“当前系统可验证的真相范围”。如果知识库过期，系统会把正确的新信息误判为幻觉。比如天气缓存两小时前是 `Rainy`，现在天气已经转晴，但知识库没更新，这时模型说“晴”可能是对的，知识库却会给出反对票。

第二，自一致性不等于真实。自一致性，白话说就是“让模型独立做多次，再看多数答案是什么”。常见决策写法是：

$$
\hat{y}=\arg\max_y \#\{y_i = y\}_{i=1}^{k}
$$

其中 $k$ 是采样次数，$\#\{y_i = y\}_{i=1}^{k}$ 表示在 $k$ 次结果里答案 $y$ 出现了多少次。这个方法能抑制偶发错误，但如果模型整体偏见一致，多数票也可能一起错。

第三，token 级置信度不直接判断真假。token 可以理解为模型一次生成的最小文本片段，比如一个词、一个子词或一个符号。它更像“模型自己对这段文本熟不熟”的信号，而不是外部事实核查。常见打分写法是：

$$
L=\frac{1}{M}\sum_{m=1}^{M}\log p_{\mathrm{token}_m}
$$

其中 $M$ 是输出长度，$p_{\mathrm{token}_m}$ 是第 $m$ 个 token 被模型生成的概率。若平均对数概率 $L$ 显著低于历史基线，通常说明这段输出更像“硬编出来的”。

所以，Agent 幻觉检测的边界可以总结成一句话：它不是证明“绝对真实”，而是在有限验证资源下，尽量找出“没有依据、与外部冲突、或内部不稳定”的调用与结论。

---

## 核心机制与推导

### 1. 知识库核查：先检查“说了什么”

知识库核查的核心做法，是把 Agent 的调用或断言拆成结构化事实，再和 $\mathcal{K}$ 对齐。

例如一次工具调用：

`weather(city="Shanghai", date="2026-03-20") -> "Sunny"`

可以被拆成若干可验证单元：

| 单元 | 含义 | 可验证来源 |
|---|---|---|
| `tool=weather` | 调用了天气工具 | 工具注册表 |
| `city=Shanghai` | 参数指向上海 | 参数 schema、地名库 |
| `date=2026-03-20` | 查询日期正确 | 时间规则 |
| `result=Sunny` | 返回解释为晴 | 天气 API/缓存 |

只要其中某一步无法被支持，就应该触发警报。这里最容易被忽略的是：工具调用本身也会幻觉。很多系统只在最终文本做审核，却默认“模型调用工具一定是正确的”，这在多工具 Agent 里通常不成立。

### 2. 自一致性：再检查“是不是稳定”

自一致性 SC@k 的思想很直接：同一个问题，多做几次独立采样。如果某个答案只有少量路径支持，而另一个答案在大多数路径中稳定出现，那么多数答案更可靠。

玩具例子如下。用户问：“上海今天什么天气？”系统做 5 次采样，得到：

| 采样编号 | 结果 |
|---|---|
| 1 | Rainy |
| 2 | Rainy |
| 3 | Sunny |
| 4 | Rainy |
| 5 | Rainy |

根据多数投票：

$$
\hat{y}=\arg\max_y \#\{y_i = y\}_{i=1}^{5} = \text{Rainy}
$$

这里 Sunny 只出现 1 次，所以单次输出 Sunny 时，应被视为不稳定答案。SC 的价值不在于“证明 Rainy 必然正确”，而在于暴露单次采样的偶然性。

### 3. token 置信度：最后检查“像不像在编”

token 级置信度常被放在后处理阶段。它不要求外部知识，也不要求多次采样，只读取生成时模型给出的概率。若一段结论的平均 log-prob 很低，就说明模型在这些词上并不确定。

例如一句话是：

“根据官方天气数据，上海今天是晴天。”

如果“官方”“天气数据”“晴天”这些关键片段的 token 概率整体偏低，就说明模型可能在用高确定性的语气包装低依据的信息。这种现象在长回答、摘要生成、工具结果复述中都很常见。

### 4. 三种机制为什么互补

这三种机制分别对应三种不同失败模式：

| 失败模式 | 典型例子 | 最有效的检测手段 |
|---|---|---|
| 外部事实冲突 | 天气库写 Rainy，模型说 Sunny | 知识库核查 |
| 单次推理偶发偏离 | 5 次里只有 1 次答对或答错 | 自一致性 |
| 语言表面流畅但内部把握低 | 复述工具结果时擅自补充细节 | token 置信度 |

真实工程例子是 ToolBench 一类多工具任务。这里问题不只是“答案文本是否正确”，还包括“是否选对工具、是否填对参数、是否解释对返回值”。研究里常把工具调用正确率与 hallucination rate 绑定评估。经验上，SC@5 能显著压低工具调用幻觉率，例如从 18% 降到 7%，但代价是大约 5 倍采样成本。这说明它不是默认开关，而是分层启用的高成本保险。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它模拟三件事：

1. 工具调用后做知识库核查。
2. 对同一问题做 SC@5 投票。
3. 用平均 log-prob 做简单置信度判定。

```python
from collections import Counter
from math import fsum

WORLD_KB = {
    ("weather", "Shanghai", "2026-03-20"): "Rainy"
}

def check_world(tool_name: str, city: str, date: str, answer: str) -> bool:
    expected = WORLD_KB.get((tool_name, city, date))
    return expected == answer

def self_consistency_vote(samples: list[str]) -> tuple[str, float]:
    counter = Counter(samples)
    winner, count = counter.most_common(1)[0]
    confidence = count / len(samples)
    return winner, confidence

def avg_logprob(token_logprobs: list[float]) -> float:
    return fsum(token_logprobs) / len(token_logprobs)

def detect_hallucination(
    tool_name: str,
    city: str,
    date: str,
    single_answer: str,
    sc_samples: list[str],
    token_logprobs: list[float],
    lp_threshold: float = -2.5,
):
    world_ok = check_world(tool_name, city, date, single_answer)
    voted_answer, sc_conf = self_consistency_vote(sc_samples)
    lp = avg_logprob(token_logprobs)

    risk_flags = []
    if not world_ok:
        risk_flags.append("world_conflict")
    if voted_answer != single_answer:
        risk_flags.append("sc_disagree")
    if lp < lp_threshold:
        risk_flags.append("low_confidence_tokens")

    return {
        "world_ok": world_ok,
        "voted_answer": voted_answer,
        "sc_confidence": sc_conf,
        "avg_logprob": lp,
        "risk_flags": risk_flags,
    }

result = detect_hallucination(
    tool_name="weather",
    city="Shanghai",
    date="2026-03-20",
    single_answer="Sunny",
    sc_samples=["Rainy", "Rainy", "Sunny", "Rainy", "Rainy"],
    token_logprobs=[-2.9, -2.7, -2.6, -2.8],
)

assert result["world_ok"] is False
assert result["voted_answer"] == "Rainy"
assert "world_conflict" in result["risk_flags"]
assert "sc_disagree" in result["risk_flags"]
assert "low_confidence_tokens" in result["risk_flags"]
```

这个例子故意做得很小，但流程已经完整：

工具调用 -> 世界检查 -> 多次采样投票 -> token 置信度告警

放到真实系统里，需要把“世界检查”从字典替换成正式的数据源，把“SC 采样”接到模型推理接口，把“log-prob 阈值”换成历史统计基线。

再给一个更接近真实工程的流程图式伪代码：

```python
def agent_answer(query):
    plan = llm_plan(query)
    tool_call = llm_generate_tool_call(plan)

    tool_result = execute_tool(tool_call)

    world_status = world_checker.verify(tool_call, tool_result)

    if world_status == "conflict":
        return escalate("工具调用或返回与知识库冲突")

    sc_outputs = [run_independent_sample(query) for _ in range(5)]
    voted_answer = majority_vote(sc_outputs)

    final_answer, token_logprobs = llm_finalize(query, tool_result, voted_answer)
    score = mean(token_logprobs)

    if score < historical_baseline(query.type):
        final_answer = add_warning(final_answer, "该回答置信度偏低")

    return final_answer
```

这个流程里，世界检查和 SC 并不是互斥关系。前者更像“硬约束”，后者更像“稳定性测量”。如果系统资源有限，优先保留世界检查；如果任务代价高、错误不可接受，再加 SC@k。

---

## 工程权衡与常见坑

最关键的权衡是准确率和成本的交换。SC@5 常能显著降低幻觉率，但它天然要多跑 5 次采样，所以延迟和 token 消耗也接近 5 倍。对在线客服、搜索补全这类低时延场景，这个代价通常过高；对代码审查、批量报告生成、后台工作流，这个代价往往是可接受的。

下面列常见坑。

| 常见坑 | 表现 | 后果 | 规避措施 |
|---|---|---|---|
| 过期知识库 | 缓存仍是旧天气、旧价格、旧 schema | 把正确结果误判为幻觉 | 给知识源加时间戳与刷新策略 |
| 工具元数据不同步 | schema 已改，校验器还按旧字段检查 | 大量误报 | 工具注册表和校验规则共用同一来源 |
| SC 成本过高 | 高并发下 k 次采样拖慢服务 | 延迟和费用失控 | 只对高风险任务启用 SC |
| token 阈值固定不变 | 所有任务共用同一个阈值 | 误杀正常长文本 | 按任务类型建立历史基线 |
| 只审最终答案，不审工具调用 | 参数错但文本看起来通顺 | 错误被包装后外泄 | 把工具调用纳入第一层审计 |

一个初学者容易踩的误区是，把知识库核查理解成“查百科”。实际上工程里更重要的是查“系统自己依赖的世界状态”。例如天气、库存、API 参数约束、账号权限、时间窗口，这些往往比通用百科更关键。

再看一个真实工程例子。一个电商 Agent 要调用 `get_inventory(sku, warehouse)` 判断能否下单。模型把 `warehouse="CN-East-1"` 误写成 `warehouse="CN-East"`，接口却做了模糊回退，返回了另一个仓的库存。此时最终文本“有货，可下单”语义上通顺，但从系统角度已经是严重事实错误。这个问题单靠 token 置信度几乎抓不住，自一致性也未必有效，因为模型可能会稳定地重复同一个错误。最有效的还是参数级知识核查。

---

## 替代方案与适用边界

如果资源有限，工程上常见三种简化部署方式。

| 方案 | 成本 | 适合任务 | 主要限制 |
|---|---|---|---|
| 仅 token 置信度 | 低 | 长文本生成、摘要、低资源场景 | 不能直接验证外部事实 |
| 仅知识库对齐 | 中 | 稳定事实、工具参数、结构化调用 | 依赖知识库完整且新鲜 |
| 仅 SC@k | 高 | 可重复求解、允许并行采样的任务 | 成本高，且多数票可能一起错 |

如果只有少量工具，而且这些工具有明确 schema、稳定返回格式，优先做知识库或规则对齐。这是收益最高的一步，因为它直接约束工具调用本身，而 Agent 最危险的错误恰恰常出在这里。

如果任务是批量离线推理，例如每天生成数千份分析摘要，那么可以加 SC@5 或 SC@3，把偶发性错误压下去。这类场景对单次延迟不敏感，更适合用多采样换稳定性。

如果既没有可靠知识源，也不能承受多次采样，那就只能退到 token 置信度。但要明确：它只能告诉你“模型像不像没把握”，不能告诉你“事实究竟是真是假”。所以它更适合作为提示灯，而不是最终裁判。

实践里更合理的分层策略通常是：

1. 默认启用知识库核查。
2. 对高风险任务启用 SC@k。
3. 对所有输出保留 token 风险分数，用于日志、告警和人工审核排序。

这样的结构符合成本曲线，也符合错误分布。因为大部分明显错误可以在第一层被拦住，真正需要高成本 SC 的，只是剩下那部分“看起来合理但不够稳”的结果。

---

## 参考资料

- World-Based Checker：把模型断言或工具调用映射到世界描述，再做事实一致性验证，适合理解“为什么工具参数也需要核查”。
- HaluCheck：面向幻觉检测的事实核查流程，可帮助理解如何把自然语言结论拆成可验证单元。
- Self-Consistency：经典多次采样投票方法，核心思想是用多数结果抑制单次推理偏差。
- ToolBench：多工具调用评测基线，适合理解 Agent 场景里“工具选择、参数填写、结果解释”为什么都可能出现幻觉。
- token-level confidence / log-prob 方法：通过平均 log-prob、熵等指标估计生成不确定性，适合理解“模型像不像在编”。

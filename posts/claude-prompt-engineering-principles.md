## 核心结论

Claude 的 Prompt Engineering，本质上不是“写提示词技巧合集”，而是**通过组织输入 token 序列，改变模型后续每一步生成概率**。这里的 token 可以先理解为“模型看到的一小段文本单位”，一句 prompt 会先被切成 token，再进入模型计算。Claude 是**自回归模型**，白话说就是“每次只预测下一个 token”，因此前面写进 prompt 的角色、上下文、格式要求、示例，都会持续影响后面的输出。

对初级工程师最重要的结论有三条：

1. prompt 不会修改 Claude 的参数，它只是在固定模型上重排上下文，从而改变 $P(\text{output}\mid\text{prompt})$。
2. 高质量 prompt 的核心不是“词藻”，而是**结构**：角色设定、任务边界、输入数据、输出格式、示例、推理与最终答案分离。
3. Claude 对粗糙 prompt 往往有一定容错，常会主动补全语义或保持格式一致；但只要任务涉及 JSON、审计、提取、分类、代码生成，仍然应优先使用结构化 prompt，而不是一段自然语言堆要求。

一个适合新手记忆的最小模板是：

```text
<role>你是一个资深数据分析师</role>
<context>这里是待分析的业务数据和背景</context>
<instructions>先判断异常，再按固定字段输出 JSON</instructions>
<examples>这里给一个输入输出示例</examples>
<answer>只输出最终结果，不附加解释</answer>
```

这不是因为 XML 标签本身有“魔法”，而是因为它让模型更容易区分“身份”“背景”“规则”“示例”“最终答复”，从而减少语义混淆。

---

## 问题定义与边界

本文讨论的问题是：**怎样设计 prompt，才能让 Claude 在格式、角色、内容范围、推理步骤上更稳定地满足 success criteria**。success criteria 可以先理解为“判定这次输出算不算成功的标准”，例如“必须输出合法 JSON”“必须分三点回答”“必须引用输入中的原文”。

边界也必须先说清：

| 约束对象 | 能否靠 Prompt 控制 | 常见手段 | 说明 |
|---|---|---|---|
| 角色风格 | 可以部分控制 | `<role>`、语气示例 | 能显著影响措辞与视角，但不是硬约束 |
| 输出格式 | 可以较强控制 | `<output_format>`、JSON schema、示例 | 对结构化任务最有效 |
| 内容范围 | 可以部分控制 | `<scope>`、禁止项、引用要求 | 仍可能出现越界补充 |
| 输出长度 | 可以弱到中等控制 | “不超过 N 字”“限定条数” | 不是逐 token 强限制 |
| 事实正确性 | 不能仅靠 prompt 保证 | 检索、工具调用、后校验 | prompt 只能降低偏差，不能替代事实源 |
| 模型能力上限 | 不能控制 | 无 | 不会因 prompt 变成另一个模型 |
| 参数与权重 | 不能控制 | 无 | prompt 不是微调 |

因此，Prompt Engineering 的边界是：**它控制输入，不控制模型本身**。如果任务要求“百分之百合法 JSON”“字段类型绝不出错”“禁止幻觉”，那么 prompt 只能作为第一层控制，还需要 API 级结构化输出、程序校验或检索增强。

一个玩具例子很直观。

不好的写法：

```text
帮我分析这个季度情况，像专业人士一样，顺便给我一个表格和结论，不要太长，最好是 JSON。
```

这里把角色、格式、长度、任务目标混在一个段落里，模型要自己猜哪些要求更重要。

更好的写法：

```text
<role>你是财务分析师</role>
<context>下面是季度收入、成本、利润数据</context>
<instructions>
1. 识别异常变化
2. 给出三条结论
3. 输出为 JSON
4. 每条结论不超过 50 字
</instructions>
<output_format>
{"findings":[{"title":"string","risk":"low|medium|high","reason":"string"}]}
</output_format>
```

这里的边界非常明确：任务是什么、格式是什么、长度怎么控制、字段类型是什么。Claude 更容易稳定命中目标。

---

## 核心机制与推导

Claude 这类大语言模型的生成过程可写成：

$$
P(x_1, x_2, \dots, x_T)=\prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

这条公式的意思是：整个输出序列的概率，等于“每一步在前文条件下生成当前 token 的概率”不断相乘。也就是说，只要你改动 prompt 前面的 token，后面所有位置的条件概率都会变。

这里的**条件概率**，白话说就是“在已有上下文下，下一个词出现的可能性”。Prompt Engineering 正是在改这些条件。

再看一个最小玩具例子。假设你希望模型输出“解法如下：先排序，再双指针”。

Prompt A：

```text
<role>你是算法讲师</role>
<instructions>请用教学风格解释解法</instructions>
```

Prompt B：

```text
请回答下面的问题。
```

假设在某一步，模型要决定是否输出“解法”这个 token：

| 场景 | $P(\text{“解法”}\mid\text{当前上下文})$ |
|---|---:|
| Prompt A | 0.60 |
| Prompt B | 0.20 |

如果后续关键 token 的联合概率相同，前者整体输出目标序列的概率就是后者的 3 倍。这个例子虽然简化，但说明了核心事实：**prompt 不是装饰，它直接参与概率计算**。

更进一步，为什么 XML 标签常有效？因为 Transformer 的注意力机制会在上下文中寻找“哪些片段和当前预测相关”。**注意力机制**可以先理解为“模型在生成下一个词时，动态查看前文重点位置的方法”。当 prompt 被写成 `<role>`、`<context>`、`<instructions>` 这些分段形式时，模型更容易学到边界：哪个片段是身份，哪个片段是输入数据，哪个片段是必须遵守的规则。

可以把这个过程理解为一次“上下文重加权”：

- `<role>` 提高特定语气、知识风格、专业词汇的概率。
- `<context>` 提高与当前任务输入相关的信息被引用的概率。
- `<instructions>` 提高格式遵守、步骤顺序、禁止事项被执行的概率。
- `<examples>` 提高输出模仿示例结构的概率。
- `<thinking>` 与 `<answer>` 分离时，可降低把中间分析直接混入最终答案的概率。

真实工程里，最常见的问题不是“模型完全不会做”，而是**它会做，但会在边界处漂移**。例如：

- 本来要 JSON，却多输出一句解释。
- 本来要引用原文，却加入未经输入提供的补充。
- 本来要先分类再总结，却只给总结。

这些偏移并不神秘，本质上是你没有把目标序列的高概率路径压得足够明显。

---

## 代码实现

工程上最实用的做法是把 prompt 拆成模板片段，而不是在代码里手写一整段长字符串。**模板化**，白话说就是“先定义固定骨架，再把业务数据填进去”。

下面先给一个最小 Python 示例，用玩具概率模型演示“前缀变化会改变整体序列概率”。

```python
from math import prod

def sequence_probability(step_probs):
    assert step_probs, "step_probs 不能为空"
    assert all(0.0 <= p <= 1.0 for p in step_probs), "概率必须在 [0, 1] 区间"
    return prod(step_probs)

# Prompt A：有 role + instructions，目标序列每一步更容易被选中
prompt_a_probs = [0.60, 0.85, 0.90]

# Prompt B：只有普通提问，第一步更不容易命中目标表达
prompt_b_probs = [0.20, 0.85, 0.90]

pa = sequence_probability(prompt_a_probs)
pb = sequence_probability(prompt_b_probs)

assert abs(pa - 0.459) < 1e-9
assert abs(pb - 0.153) < 1e-9
assert pa / pb == 3.0

print("Prompt A 序列概率:", pa)
print("Prompt B 序列概率:", pb)
print("A 相对 B 的倍数:", pa / pb)
```

这段代码当然不是 Claude 的真实内部实现，但它抓住了最关键的机制：前面某一步概率从 0.2 变成 0.6，最终目标序列概率就会连带被放大。

下面是一个更接近真实工程的 prompt 模板：

```python
def build_prompt(user_query: str, raw_data: str) -> str:
    prompt = f"""
<role>
你是企业数据分析师，擅长识别异常并输出结构化结论。
</role>

<context>
用户问题：{user_query}
原始数据：
{raw_data}
</context>

<instructions>
1. 先识别最重要的三个异常点
2. 每个异常点必须引用 context 中的具体数据
3. 最终只输出 JSON
4. 不要输出额外解释
</instructions>

<output_format>
{{
  "findings": [
    {{
      "title": "string",
      "evidence": "string",
      "risk_level": "low|medium|high",
      "action": "string"
    }}
  ]
}}
</output_format>

<answer>
只输出符合 output_format 的 JSON。
</answer>
""".strip()
    return prompt


example = build_prompt(
    user_query="分析本季度经营异常",
    raw_data="Q1利润率 18%，Q2利润率 9%，退款率从 2.1% 升至 5.8%"
)

assert "<role>" in example
assert "<output_format>" in example
assert "只输出 JSON" in example
print(example[:200])
```

真实工程例子：假设你在做一个“客服工单结构化提取”服务。输入是一段用户投诉文本，输出必须写入数据库。此时自然语言 prompt 不够，你通常会组合三层控制：

| 层级 | 作用 | 示例 |
|---|---|---|
| Prompt 结构 | 让模型理解任务与边界 | `<role>`、`<context>`、`<instructions>` |
| 输出格式约束 | 限定字段和类型 | JSON schema / structured outputs |
| 程序后校验 | 防止脏数据入库 | `json.loads`、字段枚举检查、缺失重试 |

一个典型请求会长这样：

```json
{
  "prompt": "<role>你是售后质检助手</role><context>这里是工单原文...</context><instructions>抽取问题类型、严重等级、是否退款</instructions><output_format>{...schema...}</output_format>"
}
```

这里真正可复用的不是某一句“万能提示词”，而是这套结构。

---

## 工程权衡与常见坑

Claude 在实际使用中通常比“极端脆弱”的想象要稳，但这不代表 prompt 可以随便写。常见坑主要有五类：

| 常见坑 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 不分块 | 输出夹杂解释、格式漂移 | 模型分不清规则和背景 | 用 `<role>`、`<context>`、`<instructions>` 分段 |
| 指令与示例混写 | 模型把示例内容当必须复述文本 | 边界不清 | 单独放 `<examples>` |
| 只说“按 JSON 输出” | 结果字段缺失或多字段 | 结构不够明确 | 给 schema 或最小样例 |
| 复杂任务不给推理路径 | 直接跳到结论，遗漏中间判断 | 模型倾向走最短输出路径 | 显式要求分步分析，再单独输出答案 |
| 没有后校验 | 偶发脏数据进入系统 | prompt 不是硬约束 | 程序校验 + 失败重试 |

这里有一个非常常见的误区：把“Claude 对 prompt 波动更耐受”理解成“结构无所谓”。这是错误的。更准确的说法是，**Claude 往往能在较弱提示下维持基本语义连贯，但在格式可靠性和边界遵守上，结构化 prompt 仍然显著更稳**。尤其当任务从“聊天”变成“系统组件”时，这个差异会立刻放大。

再看一个真实工程例子。某团队做“合同条款抽取”：

- 第一版 prompt：一大段自然语言，要求抽取甲方、乙方、生效日期、违约责任。
- 结果：大部分样本可用，但有时字段名漂移，例如输出成 `effective_time` 而不是 `effective_date`。
- 第二版 prompt：改成 XML 分段 + JSON schema + 必填字段校验。
- 结果：可解析率提升，重试次数下降，线上异常更少。

这类问题的本质不是“模型突然变聪明”，而是**你把目标接口定义清楚了**。

还有一个坑是把 `<thinking>` 直接展示给最终用户。如果场景只是内部推理控制，可以要求模型“先在内部完成分析，再在 `<answer>` 输出最终结果”，但工程上通常不应依赖暴露完整思维链作为产品接口，因为你真正需要的是**稳定结果**，不是一大段中间痕迹。

---

## 替代方案与适用边界

不是所有场景都值得上 XML 分段和 schema。要按任务强度选方案。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 结构化 prompt + schema | 提取、分类、审计、写库、API 对接 | 稳定、可校验、便于维护 | 编写成本更高 |
| 自然语言 prompt | 聊天、头脑风暴、开放写作 | 快、灵活、开发成本低 | 格式可靠性差 |
| 自然语言 + 示例 | 中等复杂度生成任务 | 比纯自然语言更稳 | 仍可能边界漂移 |
| 后处理校验 | 任何需要上线的场景 | 能兜底 | 不能替代上游 prompt 设计 |

如果调用环境只支持纯文本，仍然可以退化为：

```text
你是财务顾问。阅读以下数据后，输出三条建议。每条不超过 50 字，并包含 risk_level 字段。最终仅输出合法 JSON。
```

这比随意提问强很多，但可靠性仍不如明确分块。原因很简单：自然语言里的多个要求容易互相竞争，模型会自行判断“哪个更重要”。

什么时候可以简化 prompt？

- 任务是开放写作，不要求固定字段。
- 输出只给人看，不进下游程序。
- 错一两次可以接受。
- 用户交互是单轮探索，不是生产流程。

什么时候必须强化 prompt？

- 输出要被程序解析。
- 字段缺失会造成业务错误。
- 输入较长，包含多段上下文。
- 需要引用证据，不能自由发挥。
- 需要稳定复现，而不是“差不多能用”。

最后补一个边界判断：关于“Claude 对 prompt 格式的敏感性低于 GPT-4”，目前更适合写成**一些第三方实测中，Claude 对 prompt 质量波动表现出较强耐受性**。这能作为经验参考，但不应替代你自己的任务评测。因为不同模型版本、采样参数、上下文长度、任务类型都会改变结果。

---

## 参考资料

| 资料 | 链接 | 重点 |
|---|---|---|
| Anthropic: Use XML tags to structure your prompts | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags | 说明如何用 XML 标签分隔角色、上下文、指令、示例 |
| Anthropic: Structured outputs | https://docs.claude.com/en/docs/build-with-claude/structured-outputs | 说明如何用结构化输出约束 JSON 格式和字段类型 |
| Claude Prompting Best Practices | https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices | 总结 Claude 常见 prompt 组织方式与实践建议 |
| Decoder architecture / autoregressive transformers | https://mbrenndoerfer.com/writing/decoder-architecture-causal-masking-autoregressive-transformers | 解释自回归 Transformer 与链式概率公式的机制 |
| Claude vs. GPT-4 prompt sensitivity（第三方实测） | https://ai-prompts-pro.com/blog/claude-vs-chatgpt-prompts | 提供对两类模型 prompt 稳定性的经验性比较 |

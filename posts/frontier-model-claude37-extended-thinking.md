## 核心结论

Claude 3.7 Sonnet 的 Extended Thinking，本质上不是“换了一个更聪明的模型”，而是给同一个模型分配更多推理预算。推理预算可以理解为模型在正式回答前，先做分解、规划、验证的内部计算额度。

它适合复杂任务，尤其是多步推理、调试、规划、代码分析、长链路问题定位。这类任务往往不是“知道答案”就够，而是需要把问题拆成若干中间步骤。Extended Thinking 的价值，主要体现在把这部分中间计算做得更充分。

它不适合简单任务。对事实检索、格式改写、短问答、简单总结，额外思考通常不会带来等比例收益，反而只会增加延迟和成本。

可以先用一个最小判断标准：

| 任务类型 | 推荐模式 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| 复杂多步推理 | Extended Thinking | 更强的拆解、规划、验证 | 更高延迟、更高 token 成本 |
| 代码调试/架构分析 | Extended Thinking | 更完整的问题树和修复路径 | 响应更慢 |
| 简单检索/格式化 | 普通模式 | 足够快，性价比高 | 推理深度有限 |
| 日常闲聊/轻写作 | 普通模式 | 速度快 | 额外思考基本浪费 |

对初学者，可以把它理解成“慢速模式”。难题时打开，让模型先认真想；简单题时关闭，直接回答。

---

## 问题定义与边界

Extended Thinking 要解决的问题是：普通单轮回答里，模型很容易直接生成一个“看起来像答案”的输出，但对复杂任务，真正困难的部分往往是中间推导，而不是最后一句结论。

这里的“推导”不是玄学，而是明确的计算预算分配问题。Anthropic 的 Messages API 里，通常通过：

```json
"thinking": {
  "type": "enabled",
  "budget_tokens": 2000
}
```

来告诉模型：这一轮回答里，你最多可以拿出 2000 个 token 用于内部思考。

边界也很明确：

1. `budget_tokens` 必须不小于最低要求。官方文档给出的最小值是 `1024`。
2. `budget_tokens` 必须小于 `max_tokens`。
3. `max_tokens` 本身包含“思考 token + 最终输出 token”。
4. 整个请求还要满足上下文窗口限制，即：

$$
prompt\_tokens + max\_tokens \le context\_window
$$

白话说，输入已经很长时，你给输出和思考预留的空间就会被压缩；不是想给多少就给多少。

还要注意一个常被误解的边界：思考不会无限继承到后续对话轮次。Claude 3.7 Sonnet 的 Extended Thinking 主要作用于“当前 turn 的推理质量”，不是把前面所有内部思考永久叠加保存。

玩具例子可以这样看：

你问：“请分三步证明为什么二分查找要求有序数组，并给出错误示例。”

普通模式可能直接给一个结论型回答。  
Extended Thinking 模式更可能先经历这样的内部路线：

1. 定义二分查找依赖的单调性条件
2. 说明为什么无序数组破坏单调性
3. 构造反例
4. 再输出最终答案

这就是它的边界价值：不是让答案“更长”，而是让中间步骤更完整。

---

## 核心机制与推导

Claude 3.7 Sonnet 的 Extended Thinking 机制，可以抽象成一个两阶段过程：

1. 内部思考阶段
2. 最终回答阶段

在 API 响应结构里，官方文档显示它会先返回 `thinking` 内容块，再返回 `text` 内容块。对工程系统来说，这意味着“推理”和“回答”在协议层就是分开的，而不是只在概念上分开。

简化后的请求形态如下：

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 8000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 2000
  },
  "messages": [
    {
      "role": "user",
      "content": "Explain the debugging plan for a flaky distributed job."
    }
  ]
}
```

这里最关键的推导关系是：

$$
final\_answer\_tokens \le max\_tokens - thinking\_tokens
$$

如果你把 `max_tokens` 设成 8000，把 `budget_tokens` 设成 2000，那么可以粗略理解为：模型最多有 2000 的思考空间，剩余输出空间最多约 6000。实际使用中，模型不一定把预算全部花完，但上限关系存在。

为什么这能提升复杂题效果？因为复杂题的错误，很多不是出在“不会说”，而是出在“过早下结论”。更多思考预算，相当于允许模型在内部走更长的问题树：

- 多列几个假设
- 多检查几个分支
- 多做一步自我校验
- 先形成计划，再组织答案

这与传统算法里的“搜索更深”很接近。虽然大模型不是显式树搜索程序，但从工程效果看，额外预算常常等价于更充分的候选路径评估。

可以用一个非常小的数学化抽象理解：

设任务难度为 $D$，思考预算为 $B$，回答质量为 $Q$，那么经验上存在：

$$
Q = f(D, B)
$$

其中当任务较简单时，$\frac{\partial Q}{\partial B}$ 很小，也就是预算增加，质量提升有限；当任务复杂且需要多步推理时，$\frac{\partial Q}{\partial B}$ 往往更大，也就是预算更值钱。

真实工程例子：  
一个微服务系统里，订单创建偶发超时。根因可能在 API 网关、消息队列、库存服务、数据库连接池、重试风暴中的任意一环。

普通模式容易直接给“增加超时、加日志、排查慢查询”这类通用建议。  
Extended Thinking 更适合先建立排障路径：

1. 确认故障是否集中在某一依赖
2. 区分同步链路超时还是异步堆积
3. 判断是容量瓶颈还是重试放大
4. 再给出日志埋点、指标、修复建议

这不是因为模型突然“懂了分布式系统”，而是因为它有机会先组织排查结构，再输出方案。

---

## 代码实现

API 侧的最小实现并不复杂，关键是三个字段：

- `model`
- `max_tokens`
- `thinking`

下面先给一个可运行的 Python 玩具实现。它不调用 Claude API，而是模拟“预算是否有效”和“剩余回答空间”的检查逻辑，帮助理解参数关系。

```python
def validate_extended_thinking(prompt_tokens: int, context_window: int,
                               max_tokens: int, budget_tokens: int) -> dict:
    assert prompt_tokens >= 0
    assert context_window > 0
    assert max_tokens > 0
    assert budget_tokens >= 0

    if budget_tokens < 1024:
        raise ValueError("budget_tokens too small: minimum is 1024")
    if budget_tokens >= max_tokens:
        raise ValueError("budget_tokens must be less than max_tokens")
    if prompt_tokens + max_tokens > context_window:
        raise ValueError("prompt_tokens + max_tokens exceeds context window")

    return {
        "thinking_budget": budget_tokens,
        "max_answer_tokens": max_tokens - budget_tokens,
        "request_ok": True,
    }


toy = validate_extended_thinking(
    prompt_tokens=1000,
    context_window=200000,
    max_tokens=8000,
    budget_tokens=2000,
)

assert toy["thinking_budget"] == 2000
assert toy["max_answer_tokens"] == 6000
assert toy["request_ok"] is True

try:
    validate_extended_thinking(
        prompt_tokens=1000,
        context_window=200000,
        max_tokens=8000,
        budget_tokens=500,
    )
    assert False, "should have raised"
except ValueError as e:
    assert "minimum is 1024" in str(e)

print("ok")
```

如果你直接调用 Anthropic Messages API，核心请求通常类似这样：

```bash
curl https://api.anthropic.com/v1/messages \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --data '{
    "model": "claude-3-7-sonnet-20250219",
    "max_tokens": 16000,
    "thinking": {
      "type": "enabled",
      "budget_tokens": 5000
    },
    "messages": [
      {
        "role": "user",
        "content": "Plan a distributed rate limiter and explain failure modes."
      }
    ]
  }'
```

字段含义直接对应：

| 字段 | 作用 | 工程含义 |
| --- | --- | --- |
| `model` | 指定模型 | 这里是 Claude 3.7 Sonnet 的版本名 |
| `max_tokens` | 本轮输出总额度 | 包含思考和最终回答 |
| `thinking.type` | 是否启用思考 | `enabled` 表示开启 |
| `thinking.budget_tokens` | 思考预算 | 复杂题可适当增大 |
| `messages` | 输入消息 | 用户问题本体 |

如果你使用流式返回，前端或中间层最好把“思考中”和“最终回答中”作为两个状态处理。原因很简单：用户对等待更敏感，看到系统正在做哪一阶段，体验会更稳定。

一个常见工程做法是：

1. 收到 `thinking` 相关事件时，界面显示“分析中”
2. 收到正式 `text` 内容时，切换到“生成答案”
3. 记录 thinking token 使用量，供成本分析

这对调试尤其有用，因为你能区分“模型还没想完”与“模型已经开始答复但输出很慢”。

---

## 工程权衡与常见坑

最核心的权衡只有两个：成本和延迟。

官方 Cookbook 明确说明，Extended Thinking 产生的 thinking tokens 按输出 token 计费，并计入 rate limit。也就是说，打开它不是“免费多想一会儿”，而是真实消耗资源。

可以用一个简单表看清楚：

| 场景 | 推荐 Budget | 额外收益 | 额外代价 |
| --- | --- | --- | --- |
| 跨服务 Debug | 2000-8000 | 更完整的排障路径 | 首字更慢，成本更高 |
| 算法题/证明题 | 2000-16000 | 更稳定的多步推导 | 可能等待明显增加 |
| PR 审查/代码分析 | 2000-6000 | 更容易发现隐含问题 | 对简单变更不划算 |
| 简单问答 | 0 | 几乎无 | 纯浪费 |

常见坑主要有五类。

第一，预算设得太小。  
`budget_tokens < 1024` 会直接报错。这不是建议值，而是接口约束。

第二，把 `budget_tokens` 误当成额外额度。  
不是。它是 `max_tokens` 的子集。很多人把 `max_tokens=4000`、`budget_tokens=3000` 设上去，然后抱怨最终答案太短，原因就在这里。

第三，在简单任务上滥用。  
比如“把这段话改成更正式语气”“列出 HTTP 状态码”“总结这篇短文”。这类任务主要靠信息提取和表述，不靠长链推理。此时开 Extended Thinking，通常只会增加等待。

第四，忽略兼容性限制。  
Anthropic Cookbook 说明，thinking 与 `temperature`、`top_p`、`top_k` 的修改不兼容，也不能预填充响应。工程上如果你已有一套采样参数模板，需要单独为 thinking 模式分支处理。

第五，把“看到思考摘要”误解成“拿到了完整内部推理”。  
官方文档在 Claude 4 系列里强调的是 summarized thinking，即返回摘要而非全部内部推理。对 Claude 3.7 Sonnet 的历史能力，工程上也不应该把可见 thinking 当成完整可审计日志，更不能把它作为安全边界本身。

真实工程例子：  
做一次线上事故复盘助手。输入包括报警日志、trace 片段、最近变更记录。  
如果不开 Extended Thinking，模型可能给出“增加日志、回滚发布、检查依赖”的模板化答案。  
如果开到 3000 或 5000 budget，模型更可能先构造“变更影响范围 -> 故障传播路径 -> 观测缺口 -> 修复建议”的结构化分析。  
但如果你把每一条普通告警都走这条链路，吞吐会明显下降，账单也会放大。

---

## 替代方案与适用边界

Extended Thinking 不是唯一办法。它只是在“同一个模型内，用更多预算换更强复杂推理”的方案。

有三类常见替代路线。

第一类，普通模式 + 更好的任务拆分。  
如果你把问题拆成多个明确子任务，例如“先找根因候选，再评估证据，再给修复方案”，即使不开 Extended Thinking，效果也常常能提升。优点是便宜、快、可控。缺点是需要调用方自己设计流程。

第二类，普通模式 + 多轮交互。  
先让模型列计划，再让它逐步展开。它本质上是把“内部思考”改成“外部分步”。优点是透明，缺点是要多轮调用，整体编排更复杂。

第三类，迁移到更新模型的 adaptive thinking。  
截至 2026 年 3 月 12 日，Anthropic 官方文档已经把 Claude 3.7 Sonnet 标为 deprecated，而 Claude 4 系列支持 extended/adaptive thinking。也就是说，如果你在做新系统，工程上更值得优先看当前模型族，而不是围绕 3.7 Sonnet 做长期设计。

可以用一个决策矩阵快速判断：

| 任务类型 | 普通模式 | Extended Thinking | 更适合的理由 |
| --- | --- | --- | --- |
| 简单信息查询 | 是 | 否 | 推理深度不是瓶颈 |
| 文本润色/格式化 | 是 | 否 | 主要是表达，不是规划 |
| 多步技术诊断 | 可用 | 是 | 需要建立问题树 |
| 算法证明/复杂代码分析 | 可用 | 是 | 中间推导质量决定结果 |
| 高吞吐批处理 | 是 | 谨慎 | 预算和延迟压力更大 |
| 新系统长期接入 | 视场景 | 视场景 | 优先评估 Claude 4 的现行能力 |

所以最实用的落地原则不是“默认开启”，而是：

- 先把任务按复杂度分层
- 简单层走普通模式
- 中等复杂度从最小预算 `1024` 起试
- 只有在复杂任务上确实带来正确率或稳定性提升时，再增加 budget

这比“一把梭全开”更符合工程成本结构。

---

## 参考资料

- Anthropic 官方文档：Building with extended thinking  
  https://platform.claude.com/docs/en/build-with-claude/extended-thinking

- Anthropic 官方 Help Center：Using extended thinking  
  https://support.claude.com/en/articles/10574485-using-extended-thinking-on-claude-3-7-sonnet

- Anthropic Cookbook：Extended thinking  
  https://platform.claude.com/cookbook/extended-thinking-extended-thinking

- Anthropic API Reference：Thinking configuration / Messages  
  https://platform.claude.com/docs/en/api/typescript/messages

- Simon Willison：Claude 3.7 Sonnet, extended thinking and long output, llm-anthropic 0.14  
  https://simonwillison.net/2025/Feb/25/llm-anthropic-014/

## 核心结论

Claude 的 Chain-of-Thought，简称 CoT，可以理解为“模型先把中间推理拆开，再给最终答案”。它不是单独的一条提示词技巧，而是提示、后训练和推理预算共同作用的结果。

对复杂任务，CoT 的主要价值不是“显得更聪明”，而是把原本一次性生成的答案，改成多步条件生成。对数学、逻辑、长文分析这类需要多跳推理的任务，这通常能提升正确率，也更容易审计错误落在哪一步。Anthropic 官方文档把这种能力进一步产品化为 `thinking`/extended thinking：模型先产生 `thinking` 内容块，再输出最终 `text`。

常见的公开材料会用类似 “GSM8K 从 41% 到 78%” 的数字说明 CoT 的收益，但这些数字高度依赖模型版本、提示模板、是否 few-shot、是否使用可见推理。更稳妥的结论是：CoT 对多步推理通常有显著帮助，对简单问答则常常只增加延迟和成本。

一个最小玩具例子就够说明问题：题目是“30% 的人看电视，其中 24% 还拥有 4 台电视，这类人占总人数多少？”直接回答容易跳步；CoT 会先写出 $0.3 \times 0.24 = 0.072$，再得出 7.2%。

| 模式 | 输出方式 | 复杂任务准确率 | 延迟 | 可审计性 |
| --- | --- | --- | --- | --- |
| 直接答题 | 直接结论 | 基线 | 低 | 低 |
| 标准 CoT | 分步推理后结论 | 通常更高 | 中 | 中 |
| Extended thinking | thinking 块/摘要 + 结论 | 复杂任务通常最高 | 高 | 高 |

---

## 问题定义与边界

这里说的 CoT，是指 Claude 在回答前显式展开推理路径。显式的意思是“中间步骤以文本形式出现，用户或系统能看到一部分或摘要”。它和“模型内部一定在算什么”不是同一个概念。

术语首次说明：
“Rationale” 指推理说明，也就是模型把“为什么这样答”写出来的那部分文本。
“Extended thinking” 指 Anthropic 提供的更长推理模式，本质是给模型更多推理预算。
“Budget tokens” 指 thinking 可消耗的 token 上限，可以理解为“允许模型思考多长”。

适用边界很明确：如果任务需要多步计算、条件分解、文档比对、方案规划，CoT 值得开；如果只是 FAQ、定义查询、普通润色，直接答题通常更划算。

官方 API 文档要求 thinking 的最小预算是 1,024 tokens；官方提示文档建议从最小预算开始调，超过 32K 时要警惕超时和连接问题。对零基础读者，可以把它理解成：推理不是免费午餐，步骤越多，时间和费用越高。

| 任务类型 | 是否适合 CoT | 预期收益 | 说明 |
| --- | --- | --- | --- |
| GSM8K 这类数学题 | 高 | 高 | 需要显式中间计算 |
| ProofWriter/逻辑题 | 高 | 中到高 | 需要链式条件推导 |
| 财务分析/合规审查 | 高 | 高 | 需要可检查的依据链 |
| 简单 FAQ | 低 | 低 | 额外步骤通常浪费 |
| 文风改写 | 低 | 低 | 推理链很少带来收益 |

---

## 核心机制与推导

Claude 的 CoT 能力，不是只靠“请一步一步思考”这句提示词长出来的。更核心的是后训练。Anthropic 在 Constitutional AI，简称 CAI，和 RLHF/RLAIF 路线里，用“模型生成回答，再被偏好模型或规则体系打分，再继续优化”的方式，鼓励更有用、更合规的回答形式。

白话解释：
“CAI” 可以理解为“先写一套原则，再让模型按这些原则批评和修正自己”。
“RLAIF” 可以理解为“不是完全靠人打分，而是部分用 AI 反馈当奖励信号”。

公开讨论里常见一个 PPO-ptx 形式的目标函数写法：

$$
L(\phi)=\mathbb{E}\left[r_\theta(x,y)-\beta \log \frac{\pi_\phi^{RL}(y|x)}{\pi^{SFT}(y|x)}\right]+\gamma \mathbb{E}[\log \pi_\phi^{RL}(x)]
$$

它的直觉是：

| 项 | 作用 | 直白解释 |
| --- | --- | --- |
| $r_\theta(x,y)$ | 奖励项 | 奖励“更好”的回答或推理 |
| $\beta KL$ 项 | 约束项 | 防止模型为了高分跑偏原始语言分布 |
| $\gamma$ 预训练保留项 | 稳定项 | 尽量别把原有语言能力练坏 |

所以 CoT 的提升并不是“模型突然会思考了”，而是训练目标更偏好“先拆步骤、再下结论”的行为模式。Anthropic 官方还特别提醒一个边界：可见的 CoT 不一定完全忠实反映模型真实内部依据。也就是说，thinking 块有用，但不能把它当成完美的脑内录像。

从训练到推理可以简化为：

训练数据与原则约束 → 偏好/奖励建模 → PPO 类更新 → 推理时允许 thinking 块展开 → 最终答案

真实工程例子是财务分析。给 Claude 一份长财报和一组问题，如果直接让它下结论，容易遗漏脚注、时间范围或口径差异；开启 thinking 后，模型通常会先列假设、识别指标定义、再做交叉核对。这样不一定绝对正确，但更容易让工程师定位错误是在“数据抽取”还是“推理合并”。

---

## 代码实现

现在的 Claude API 不是传 `thinking=true`，而是传 `thinking` 对象。对 Claude 4 系列，文档已同时支持 manual thinking 和 adaptive thinking；对老的 Claude Sonnet 3.7，返回的是完整 thinking；对 Claude 4，多数场景返回的是 thinking 摘要。

```python
def toy_cot_ratio(watch_tv_ratio: float, own_four_ratio_among_watchers: float) -> float:
    result = watch_tv_ratio * own_four_ratio_among_watchers
    assert 0.0 <= result <= 1.0
    return result

ratio = toy_cot_ratio(0.30, 0.24)
assert abs(ratio - 0.072) < 1e-9
assert f"{ratio*100:.1f}%" == "7.2%"
print(f"拥有4台电视的人占比: {ratio*100:.1f}%")
```

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 4000
  },
  "messages": [
    {
      "role": "user",
      "content": "先列出推理依据，再给最终结论。比较两份财报里现金流变化的主要原因。"
    }
  ]
}
```

实践上，提示词应少写“机械步骤指令”，多写目标和检查标准，例如：

```text
请先给出推理依据，再给结论。
要求：
1. 明确列出使用了哪些事实
2. 如果有假设，单独标出
3. 最终结论单独成段
```

---

## 工程权衡与常见坑

最大的三个问题不是“答不答得出”，而是“贵不贵、慢不慢、能不能完整返回”。

| 常见坑 | 表现 | 缓解方式 |
| --- | --- | --- |
| token 成本高 | 输出变长，计费上升 | 从 1,024 起逐步调预算 |
| 超时/长连接不稳 | 32K 以上更明显 | 用 batch、流式返回、拆任务 |
| 安全截断 | thinking 显示未完成 | 重写提示，降低敏感描述，分两轮问 |

对新手最重要的一条是：如果界面提示 thinking 未完成，不一定是模型坏了，很多时候是安全系统把后续 thinking 隐去了。此时不要盲目加大预算，先改写问题边界。

另一个常见误区是“所有复杂任务都该开最长 thinking”。不对。过长推理会带来收益递减，甚至把简单问题拖慢。经验上应先用最小预算验证收益，再逐步增加。

---

## 替代方案与适用边界

不是所有任务都要用 CoT。短问题、确定性查表、固定格式抽取，往往直接答题更好。真正需要 CoT 的，是“步骤本身决定答案质量”的任务。

| 模式 | 适合任务 | 响应时间 | token 成本 | 备注 |
| --- | --- | --- | --- | --- |
| No CoT | FAQ、改写、简单检索 | 最快 | 最低 | 默认首选 |
| Standard CoT | 中等复杂分析 | 中 | 中 | 适合多数工程任务 |
| Extended thinking | 高复杂度推理、规划、代码/数学难题 | 最慢 | 最高 | 只在收益明确时启用 |

如果你更关心“快”和“便宜”，可以不用 extended thinking，只在 prompt 里要求“先列依据后下结论”。如果你更关心“复杂问题成功率”，再考虑提高 budget 或使用支持更强 thinking 的模型版本。

需要纠正一个常见过时说法：并不是 Claude 3.5 Sonnet 支持 extended thinking。Anthropic 当前官方文档列出的 thinking 支持模型，重点是 Claude 4 系列以及已弃用的 Claude Sonnet 3.7（`claude-3-7-sonnet-20250219`）。写工程文档时，模型名必须按官方当前接口写。

---

## 参考资料

1. [Anthropic: Let Claude think (chain of thought prompting)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-of-thought)  
用途：说明 CoT 在提示工程中的定位和适用场景。

2. [Anthropic: Building with extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)  
用途：说明 `thinking` 内容块、`budget_tokens`、流式输出和模型版本差异。

3. [Anthropic: Extended thinking tips](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/extended-thinking-tips)  
用途：说明最小 1,024 token 预算、32K 以上的超时风险与调参建议。

4. [Claude Help Center: Using extended thinking](https://support.claude.com/en/articles/10574485-using-extended-thinking)  
用途：说明产品侧“Thinking”指示器、未完成 thinking 与安全截断。

5. [Anthropic: Measuring Faithfulness in Chain-of-Thought Reasoning](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)  
用途：说明可见 CoT 有用，但不一定完全忠实反映真实内部依据。

6. [Anthropic / arXiv: Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)  
用途：说明 CAI、RLAIF、自我批评与偏好优化的训练思路。

7. [RLHF Book: Synthetic Data & CAI](https://rlhfbook.com/c/13-cai.html)  
用途：补充 PPO-ptx、KL 正则和预训练保留项的直观解释。

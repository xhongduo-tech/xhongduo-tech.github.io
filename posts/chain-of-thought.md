## 核心结论

Chain-of-Thought，简称 CoT，中文通常叫“推理链”，意思是让模型把中间思考步骤显式写出来，而不是只吐出最终答案。它的核心作用不是“让模型更会说”，而是“让模型多占用一些输出 token 作为草稿纸”，从而把原本压缩在一次前向生成里的多步计算展开。

Wei 等人在 2022 年的结论可以概括成一句话：在 few-shot prompt 里放入“题目 + 中间推理步骤 + 最终答案”的示例后，超大模型会模仿这种格式，先写步骤，再给答案，复杂推理任务的准确率明显上升。few-shot 的意思是“给少量示例”，也就是不重新训练模型，只在提示词里示范几题。

Kojima 等人随后提出 Zero-shot CoT。zero-shot 的意思是“不给示例，直接做题”。它的关键发现是，只要在答案前加一句 `Let's think step by step.`，大模型就可能自动展开推理链。这说明模型内部本来就存在某种可被触发的多步推理能力，只是默认不一定显式输出。

再往前一步，Self-Consistency 的思路是：既然一条推理链可能偶然走偏，那就采样多条不同的推理链，再对最终答案做多数投票。它本质上是“多次独立解题，再选共识结果”，通常能继续提升 5% 到 10% 左右的准确率，但代价是 token 和延迟线性增加。

下面这个玩具例子可以直接说明 CoT 的作用：

| 提示方式 | 输入片段 | 典型行为 |
|---|---|---|
| 无 CoT | `Q: 2,4,8,14,22, ?` | 直接猜下一项，容易错 |
| Few-shot CoT | `A: 先看差值，再找规律` | 倾向输出“差值为2,4,6,8，下一项加10” |
| Zero-shot CoT | `A: Let's think step by step.` | 模型自己尝试展开步骤 |
| Self-Consistency | 重复采样多条链 | 对多个候选答案投票 |

例如数列 `2,4,8,14,22`，正确思路是先看相邻差值：$2,4,6,8$，差值每次增加 2，因此下一次应加 10，答案是 $22+10=32$。如果没有中间步骤，模型可能只凭表面模式猜成 30 或 28；如果先写出“差值序列”，正确率通常更高。

---

## 问题定义与边界

CoT 要解决的问题，不是“让模型说得更长”，而是“让模型把原本隐含的多步计算，显式展开成可跟踪的中间状态”。这里的“中间状态”可以理解成草稿过程，例如先列条件、再算中间值、最后得结论。

它适合的任务有一个共同特征：最终答案不是靠单次模式匹配就能稳定得到，而需要若干步变换。典型任务包括算术、多跳问答、符号推理、逻辑题和带约束的规划问题。反过来说，如果任务本身只是事实回忆，比如“巴黎是哪国首都”，CoT 往往只会增加输出长度，不一定提升正确率。

边界同样重要。CoT 的经典结果主要出现在超大参数模型上，尤其是论文中反复提到的 100B 以上级别模型。原因不是“小模型不会输出长文本”，而是“小模型往往还没有学会把自然语言推理链真正当作计算过程”。它可能会生成看起来像解释的话，但这些话只是叙述噪声，不是有效计算。

下面这张表可以把边界说清楚：

| 输入类型 | 触发方式 | 对模型规模的要求 | 主要风险 |
|---|---|---|---|
| 无 CoT | 直接问答 | 低 | 多步题目容易直接猜 |
| Few-shot CoT | 给示例推理链 | 中到高，经典结果多见于 >100B | 示例写得差会诱导错误格式 |
| Zero-shot CoT | `Let's think step by step.` | 更依赖模型已具备推理能力 | 小模型常输出空洞步骤 |
| Self-Consistency | 多次采样后投票 | 需要基础 CoT 已有效 | 成本和延迟按采样次数上升 |

一个反例很关键：如果你在 15B 或 20B 级模型上直接塞入完整 CoT，常见输出是“我先分析条件，然后做计算，最后得到结果”，但中间并没有真正算对。这类文本形式上像推理，功能上却只是解释模板。Zero-shot CoT 在这种模型上也容易退化成“自然语言碎片”，并不自动转化为更强的求解能力。

所以，问题定义要收紧成一句话：CoT 不是通用增益开关，而是一种“在足够强的模型上，把内部潜在推理路径转成外显草稿”的触发技术。

---

## 核心机制与推导

Few-shot CoT 的机制很直接：先给模型几个“题目 + 逐步解答 + 最终答案”的示例，模型会模仿这种结构继续生成。这里的“模仿”不是浅层复制，而是利用大模型的 in-context learning，中文可以理解为“在上下文里临时学会一种解题格式”。

最小模板通常长这样：

```text
Q: 12 × 13
A: 12 × 10 = 120，12 × 3 = 36，120 + 36 = 156，所以答案是 156。

Q: 27 + 38 - 15
A: 27 + 38 = 65，65 - 15 = 50，所以答案是 50。

Q: [新问题]
A:
```

模型看到前面两道题的“骨架”后，会倾向于延续同一种格式。这就是 prompt skeleton，中文可以理解为“提示骨架”，即输出结构被前文定型。

Zero-shot CoT 更极端。它不提供示例，只提供一个触发语：

```text
Q: 12 × 13
A: Let's think step by step.
```

这句话的作用不是传递知识，而是切换输出模式。它告诉模型：“不要直接给结论，先展开步骤。”为什么一句短语能触发？一种常见解释是，大模型在预训练语料中已经见过大量“逐步解释”的文本模式，因此这个短语像一个模式选择器，把生成分布推向“先推理、再总结”的区域。

Self-Consistency 则是在此基础上再加一层聚合。设第 $i$ 次采样得到的最终答案为 $a_i$，那么最终输出定义为：

$$
\hat{a}=\arg\max_a \sum_{i=1}^{k}\mathbb{1}[a_i=a]
$$

其中 $\mathbb{1}[a_i=a]$ 是指示函数，意思是“如果第 $i$ 条链的答案等于 $a$，就记 1，否则记 0”。这个公式本质上就是多数投票。它的假设是：错误推理路径比较分散，而正确路径更容易在多次采样中重复出现。

为什么 CoT 会有效，常见理论解释是 token budget hypothesis，可以翻译成“token 预算假说”。token 是模型处理文本的最小片段，可以粗略理解为“文字计算单位”。这个假说认为，中间 token 相当于额外的草稿空间，延长了模型的有效计算深度。不是模型参数突然变多了，而是原本必须压缩在一步完成的推理，被展开成多步序列。

可以把这个关系写成一个非常粗略的工程直觉：

$$
\text{有效推理容量} \approx f(\text{模型能力}, \text{可用中间 token 数})
$$

这不是严格论文公式，但能帮助理解：当中间 token 太少时，模型只能“凭感觉近似”；当允许它多写几步时，就有空间保存中间结果。

下面用表格描述 token 预算与输出行为的关系：

| 预算控制方式 | 典型指令 | 结果 | 风险 |
|---|---|---|---|
| 不限制 | `Think step by step.` | 推理更充分 | 容易冗长，成本高 |
| 软限制 | `Use about 80 tokens.` | 兼顾步骤与成本 | 可能不稳定 |
| 强限制 | `Use <40 tokens.` | 输出更短 | 复杂题可能推不完 |
| 多采样 | `n=8` 条链 | 稳定性更高 | 成本乘以 8 |

再看一个玩具例子。题目是：`Q: 小明有 12 支笔，又买了 13 支，送人 5 支，还剩多少？`

无 CoT 时，模型可能直接给 20，也可能给 25。  
有 CoT 时，过程会更像：

1. 先算总数：$12+13=25$
2. 再减去送出的：$25-5=20$
3. 所以答案是 20

这里提升的关键不是语言更“像老师”，而是中间值 25 被明确写出来了，后续步骤不必完全依赖隐藏状态记忆。

---

## 代码实现

工程里真正可复用的部分通常有三层：prompt 拼接、调用模型、抽取最终答案。CoT 的推理文本可以保留做调试，但系统最终往往只消费结构化答案。

先看一个可运行的 Python 玩具实现。它不调用真实 LLM，只模拟 Self-Consistency 的投票逻辑：

```python
from collections import Counter
import re

def extract_answer(chain: str) -> str:
    match = re.search(r"answer is ([^.\n]+)", chain.lower())
    if not match:
        raise ValueError(f"cannot extract answer from: {chain}")
    return match.group(1).strip()

def majority_vote(chains):
    answers = [extract_answer(c) for c in chains]
    votes = Counter(answers)
    return votes.most_common(1)[0][0]

chains = [
    "Let's think step by step. 12*10=120, 12*3=36, total 156. Therefore, the answer is 156.",
    "Let's think step by step. Break 13 into 10 and 3, then 120+36=156. Therefore, the answer is 156.",
    "Let's think step by step. I made an error and got 154. Therefore, the answer is 154.",
]

final = majority_vote(chains)
assert extract_answer(chains[0]) == "156"
assert final == "156"
print(final)
```

这个例子有两个要点：

1. `extract_answer` 负责从整段推理链里抽出最终答案。
2. `majority_vote` 不关心中间步骤写得多漂亮，只对最终候选答案计票。

如果接真实模型，流程通常是这样：

```python
from collections import Counter

def build_prompt(question: str, mode: str = "zero_shot") -> str:
    if mode == "few_shot":
        demos = """
Q: 12 * 13
A: 12 * 10 = 120, 12 * 3 = 36, 120 + 36 = 156. Therefore, the answer is 156.

Q: 27 + 38 - 15
A: 27 + 38 = 65, 65 - 15 = 50. Therefore, the answer is 50.
""".strip()
        return f"{demos}\n\nQ: {question}\nA:"
    if mode == "zero_shot":
        return f"Q: {question}\nA: Let's think step by step."
    raise ValueError("unknown mode")

def self_consistency_generate(model, question: str, k: int = 5):
    prompt = build_prompt(question, mode="zero_shot")
    chains = [model.generate(prompt, temperature=0.7) for _ in range(k)]
    answers = [extract_answer(chain) for chain in chains]
    final = Counter(answers).most_common(1)[0][0]
    return {
        "prompt": prompt,
        "chains": chains,
        "answers": answers,
        "final": final,
    }
```

真实工程例子可以看“金融分析助手”。假设用户问：“某公司未来三年自由现金流是否覆盖债务偿付？”这类问题不是单步检索，而是多步计算：读取现金流、拆分资本开支、比较到期债务、考虑利率变化。系统可以这样设计：

| 步骤 | 动作 | 目的 |
|---|---|---|
| 1 | 拼接 zero-shot CoT prompt | 触发逐步分析 |
| 2 | 调用模型生成 8 条链 | 提高鲁棒性 |
| 3 | 提取每条链的最终判断 | 转成结构化结果 |
| 4 | 多数投票 | 降低单次偶发错误 |
| 5 | 保留原始 reasoning text | 方便审计与排错 |

如果任务需要固定风格，例如“先列假设，再算指标，再给风险结论”，则 few-shot CoT 更合适，因为示例本身就把输出骨架锁定了。

---

## 工程权衡与常见坑

CoT 的第一笔成本不是模型费，而是输出长度。原本一句能答完的问题，现在要多写一段中间过程。若平均输出 token 从 80 增加到 240，账单和延迟都可能接近 3 倍。Self-Consistency 再乘上采样次数，整体成本很快变成主约束。

下面这张表最实用：

| 机制 | 成本负担 | 适合何时开启 | 缓解策略 |
|---|---|---|---|
| Few-shot CoT | prompt 长、输出也长 | 任务模式固定，示例可复用 | 精简示例数量 |
| Zero-shot CoT | 输出变长 | 先快速验证模型是否会推理 | 加 token 预算限制 |
| Self-Consistency | 调用次数线性增加 | 高价值问题，容错要求高 | 减少采样数，先筛问题 |
| Token budget | 可能损失部分推理细节 | 成本敏感场景 | 对复杂度分级设置预算 |

第一个常见坑是“把解释当推理”。一段很像解释的话，不等于它真的完成了计算。判断标准不是文风，而是中间变量是否可验证、步骤是否闭合、最终结论是否和中间结果一致。

第二个坑是“在不合适的模型上强推 CoT”。如果基础模型太小，CoT 可能让输出更长，但答案更差。原因是额外文本把生成空间扩得更大，却没有对应的计算能力去利用它。

第三个坑是“只看最终准确率，不看可运维性”。真实系统里，推理链还影响日志体积、审计风险、前端展示和缓存命中率。很多团队上线后才发现，同样 QPS 下，CoT 版本把延迟从 2 秒拉到 8 秒，或者把每次调用成本放大 5 倍。

第四个坑是“答案抽取不稳”。模型可能写：
- `Therefore, the answer is 156`
- `Final answer: 156`
- `So 156 is correct`

如果没有稳定的抽取协议，Self-Consistency 的投票层会直接失效。工程上通常要么要求固定输出格式，要么在后处理阶段做更强的解析。

金融分析助手是典型权衡案例。zero-shot CoT 加 8 次 Self-Consistency 可能显著提升关键报告的一致性，但它意味着 8 倍采样成本。若场景是实时对话，这通常不可接受；若场景是离线生成高价值报告，这种成本可能合理。实务中的折中做法通常是：

1. 先用单次 zero-shot CoT 跑普通请求。
2. 只对高风险、高金额、高不确定性问题启用多采样。
3. 对每类任务设置 token budget，比如 `<120 tokens>`。
4. 对明显简单题直接关闭 CoT。

---

## 替代方案与适用边界

如果模型还不具备稳定的 CoT 能力，替代方案通常比“硬加一句 step by step”更可靠。最常见的是问题拆分，也就是 decomposition，中文可理解为“把大问题手工切成几个小问题”。

例如模型只有 20B 左右，你不要要求它一次完成“读题、建模、计算、核验”。更稳的做法是分成两步甚至三步：
1. 先抽取已知条件。
2. 再根据条件做单步计算。
3. 最后汇总答案。

这种方法的核心优势是：步骤来自系统流程设计，而不是依赖模型自己学会生成高质量推理链。

另一类替代方案是 Retrieval + decomposition，也就是“检索增强 + 分步求解”。retrieval 的白话解释是“先从外部知识库拿资料，再让模型处理”。当任务难点在知识缺失而不是计算缺失时，它往往比 CoT 更有效。

下面做一个直接对比：

| 方案 | 优点 | 缺点 | 更适合什么边界 |
|---|---|---|---|
| CoT | 对已具备推理能力的大模型提升明显 | 成本高，小模型不稳定 | >100B 级模型，多步推理任务 |
| 手动问题拆分 | 可控、稳定、易调试 | 流程设计成本高 | 中小模型、固定任务流 |
| Retrieval + decomposition | 补足外部知识，减少幻觉 | 依赖检索质量 | 知识密集型任务 |
| 审稿机制/规则复核 | 易审计，适合高风险场景 | 复杂度更高 | 金融、医疗、合规等场景 |

在敏感场景里，Self-Consistency 也不是唯一办法。比如合规审核、医疗辅助、财务报告，很多团队更偏向“单次生成 + 审稿模型复核 + 人工抽检”。原因很现实：多采样投票虽然能提准确率，但不一定便于审计；而规则复核和审稿链路更容易挂接现有治理系统。

所以适用边界可以压缩成一句工程判断：

- 如果模型足够大，问题需要多步推理，且允许更高 token 成本，优先考虑 CoT。
- 如果模型偏小，或者系统要求强可控、低延迟、强审计，优先考虑问题拆分、检索增强或审稿机制。

---

## 参考资料

1. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*，2022。用于 CoT 原始定义、few-shot 模板与大模型规模边界说明。  
2. Kojima et al., *Large Language Models are Zero-Shot Reasoners*，2022。用于 Zero-shot CoT 的触发短语与实验结果说明。  
3. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*，2023。用于多数投票公式与多采样机制说明。  
4. Han et al., *Token-Budget-Aware LLM Reasoning*，2024/2025。用于 token-budget 假说与预算控制的工程讨论。  
5. arxiv.gg 上述论文索引页。用于快速核对论文摘要和结论脉络。  
6. Galileo.ai 关于 Chain-of-Thought 的工程博客。用于真实系统中 few-shot、zero-shot、Self-Consistency 的实践视角。  
7. Emergent Mind 关于 Zero-shot CoT、few-shot CoT、token budget 的整理文章。用于补充术语解释与工程边界。  
8. ICML 2022 Kojima 论文展示页。用于 Zero-shot CoT 在算术与推理基准上的代表性结果说明。

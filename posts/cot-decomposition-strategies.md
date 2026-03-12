## 核心结论

多步推理里的 CoT 分解策略，核心不是“让模型多想一会”，而是把原问题 $T$ 拆成一串更容易判断、检查、拼接的子问题 $\{S_1,\dots,S_n\}$。这样做的直接收益是：每一步的目标更明确，中间结果可以被复用，最终答案不再完全依赖一次性长输出。

三种常见策略可以先按一句话区分：

| 策略 | 一句话定义 | 拆解形式 | 子步骤关系 | 适合场景 |
|---|---|---|---|---|
| Least-to-Most Prompting | 先做最简单的子问题，再把前面答案喂给后面步骤 | 有序串行 | 强依赖前序答案 | 规则明确、步骤顺序固定的任务 |
| Decomposed Prompting | 把任务拆给不同模块或不同 prompt 分别处理 | 模块化 | 可串行也可并行 | 任务可分成检索、计算、抽取等不同能力块 |
| Successive Prompting | 模型先提出下一问，再回答，再继续 | 互动式迭代 | 每轮动态决定下一步 | 需要验证、追问、外部工具配合的 agent 场景 |

Least-to-Most Prompting，简称 LtM，可以把它理解为“从最小可解单元开始累积”。它在 SCAN 这类组合泛化任务上的代表性结果很突出：论文报告中，复杂长度拆分场景下准确率可从 16% 提升到 99.7%。这说明问题分解不是装饰，而是显著改变了模型处理复杂组合任务的方式。

先看一个玩具例子。任务是把命令 `TURN RIGHT ×2 + JUMP` 展开成动作序列。直接让模型一次性输出，常见错误是漏掉重复、顺序错、把 `×2` 作用范围看错。LtM 的做法是：

1. 先问：`TURN RIGHT ×2` 展开后是什么？
2. 再问：把上一步结果和 `JUMP` 拼起来是什么？
3. 最后输出完整动作序列。

于是中间过程变成：

- 子问题 1：`TURN RIGHT ×2`  
  答案：`TURN RIGHT, TURN RIGHT`
- 子问题 2：把 `TURN RIGHT, TURN RIGHT` 与 `JUMP` 连接  
  答案：`TURN RIGHT, TURN RIGHT, JUMP`

新手需要看到的重点是：LtM 不是“多写几句话”，而是“拆问题 -> 逐步答 -> 累积输出”。

真实工程里，DROP 这类多步阅读理解问题更能体现价值。DROP 是一个需要从段落中做多跳抽取、计数、比较的问答数据集。一个问题可能同时需要先找事件、再找时间、再做差值。若直接一次性回答，模型很容易在中间推理环节丢信息；若拆成子问，系统就能分别做证据定位、局部计算和最终汇总。

---

## 问题定义与边界

形式化地说，原任务 $T$ 被拆成若干子任务：

$$
T \rightarrow \{S_1, S_2, \dots, S_n\}
$$

第 $k$ 个子答案记为 $A_k$。如果后续步骤依赖前面的结果，那么最终答案并不是单独由某一个子问题决定，而是由整条依赖链决定：

$$
A_k = f(S_k, \{A_1,\dots,A_{k-1}\})
$$

这里的“依赖”可以用白话解释为：后一步要拿前一步的结果当输入，不是彼此独立的。

因此，解答顺序会影响最终答案 $A$。一个直观依赖图可以写成：

$$
S_1 \rightarrow A_1 \rightarrow S_2 \rightarrow A_2 \rightarrow \cdots \rightarrow S_n \rightarrow A
$$

这类分解策略适合下面几类任务：

| 任务类型 | 是否适合分解式 CoT | 原因 |
|---|---|---|
| 多步问答 | 适合 | 可以把“抽取证据、比较、计算”拆开 |
| 指令序列生成 | 适合 | 动作展开有天然步骤依赖 |
| 工具调用 agent | 适合 | 每一步都可验证并决定下一步 |
| 单步分类 | 通常收益有限 | 问题本身没有明显中间状态 |
| 开放式长文写作 | 收益不稳定 | 子问题边界模糊，拆分未必带来更强约束 |

判断“什么时候该启用分解”，一个简单标准是：如果你能明确指出至少两个不可省略的中间问题，那么这个任务通常值得拆。

例如一个 DROP 风格问题：

> 段落中说，A 球队在 2018 年得分 24 分，在 2019 年得分 31 分。问题：A 球队两年得分相差多少？

这个问题虽然短，但它至少包含三个子动作：

1. 定位 2018 年对应得分
2. 定位 2019 年对应得分
3. 做减法

如果直接提问“相差多少”，模型可能抓错年份或把比较方向弄反。分解后则更稳：

- 子问 1：2018 年得分是多少？
- 子问 2：2019 年得分是多少？
- 子问 3：两者差值是多少？

边界也要说清楚。分解式 CoT 不是万能方法。若任务本身就是单步映射，比如“判断这句话情感是正面还是负面”，强行拆分可能只会增加 token 成本，并不提高准确率。另一个边界是连续生成任务，例如文学创作，很多“中间子问题”并没有客观可验证答案，此时分解未必比直接生成更好。

---

## 核心机制与推导

### 1. Least-to-Most Prompting

LtM 的核心规则是：按难度或依赖顺序，把问题拆成一串更小的问题，并让每一步显式接收之前答案。其计算形式可以写成：

$$
A_k = f(S_k,\{A_{<k}\})
$$

其中 $\{A_{<k}\}$ 的意思是“前面所有已经得到的答案”。

还是看 `TURN RIGHT ×2 + JUMP` 这个玩具例子：

| 步骤 | 子问题 | 中间答案 | 下一步输入如何构造 |
|---|---|---|---|
| 1 | `TURN RIGHT ×2` 应展开成什么？ | `TURN RIGHT, TURN RIGHT` | 把这个结果作为已知答案传给步骤 2 |
| 2 | 已知前面是 `TURN RIGHT, TURN RIGHT`，再接 `JUMP` 后完整序列是什么？ | `TURN RIGHT, TURN RIGHT, JUMP` | 作为最终输出 |
| 3 | 无 | 最终答案 | 无 |

这里的关键不在“分两步”，而在“第二步必须看到第一步答案”。如果第二步看不到前文，它就不是 LtM，只是两个孤立 prompt。

对于更真实的 DROP 问题，LtM 可以这样拆：

- 子问 1：段落中 2018 年 A 球队得分是多少？
- 子问 2：段落中 2019 年 A 球队得分是多少？
- 子问 3：已知 2018 年是 24、2019 年是 31，相差多少？

这样第三步已经不是“从长段落里继续找答案”，而是把前两步的结构化结果做计算。复杂度被转移到了更可控的中间状态上。

### 2. Decomposed Prompting

Decomposed Prompting 的重点不是固定顺序，而是“不同子任务可以由不同处理器负责”。这里的“处理器”用白话讲，就是不同 prompt 模板、不同子模型，甚至不同工具。

它可以写成：

$$
A_k = f_k(S_k, C_k)
$$

其中 $f_k$ 表示第 $k$ 类子任务自己的解法，$C_k$ 是该任务需要的上下文。比如：

- 检索子任务：用“抽取证据” prompt
- 计算子任务：用“只输出计算过程和结果” prompt
- 汇总子任务：用“把前面结构化结果写成自然语言答案” prompt

DROP 示例可以拆成下面三个模块：

| 模块 | 任务 | prompt 风格 |
|---|---|---|
| Evidence Finder | 从段落里抽取相关句子 | 只返回证据句和对应字段 |
| Reasoner | 对抽取结果做比较或计算 | 只基于结构化输入推理 |
| Verbalizer | 生成最终自然语言答案 | 不再重新阅读原文 |

这种方式的优势是模块边界更清楚，甚至可并行。比如一个问题同时需要找“球队得分”和“比赛日期”，这两个抽取子任务可以并发完成，再在后面汇总。

更进一步，某个模块内部还可以继续使用 LtM。也就是说，Decomposed Prompting 是一个更外层的框架，不排斥内部递归分解。

### 3. Successive Prompting

Successive Prompting 适合交互式推理。它不是提前写好完整分解，而是让系统在每一轮决定“下一步该问什么”。

典型形式是两个函数交替工作：

$$
q_t = f_{\text{QD}}(H_{t-1}), \quad a_t = f_{\text{QA}}(q_t, H_{t-1})
$$

其中：

- $f_{\text{QD}}$：Question Decomposer，负责生成下一问
- $f_{\text{QA}}$：Question Answerer，负责回答当前问
- $H_{t-1}$：到当前轮为止的历史

当系统生成终止标记 $\langle \text{EOQ} \rangle$ 时，循环结束。

一个 DROP 风格伪对话可以写成：

- QD：为了回答主问题，先找 2018 年 A 球队得分是多少？
- QA：24
- QD：再找 2019 年 A 球队得分是多少？
- QA：31
- QD：根据 24 和 31，差值是多少？
- QA：7
- QD：`<EOQ>`

这种方法的优点是灵活。它不像 LtM 那样要求你在一开始就把所有步骤设计好，而是边推理边决定下一步。缺点也明显：如果没有终止条件和历史校验，它很容易无限追问或者在错误中间答案上越走越远。

---

## 代码实现

下面给一个最小可运行的 Python 示例，用来演示 LtM 和 Successive 的基本控制逻辑。这里不用真实大模型，而是用规则函数模拟，目的是让流程可运行、可断言。

```python
from typing import List, Dict

def ltm_expand_command(command: str) -> List[str]:
    """
    玩具例子:
    输入: 'TURN RIGHT x2 + JUMP'
    输出: ['TURN RIGHT', 'TURN RIGHT', 'JUMP']
    """
    # 子问题1: 展开重复
    parts = [p.strip() for p in command.split("+")]
    expanded = []

    for part in parts:
        if "x" in part:
            action, times = part.rsplit("x", 1)
            action = action.strip()
            times = int(times.strip())
            expanded.extend([action] * times)
        else:
            expanded.append(part)

    return expanded

def successive_drop_reasoner(facts: Dict[str, int]) -> int:
    """
    真实工程例子的简化版:
    已知两个年份的得分，按 Successive 风格逐轮提问并回答。
    """
    history = []
    step_budget = 5

    for _ in range(step_budget):
        # QD: 根据历史决定下一问
        if "score_2018" not in history:
            q = "2018 score?"
            a = facts["2018"]
            history.append("score_2018")
        elif "score_2019" not in history:
            q = "2019 score?"
            a = facts["2019"]
            history.append("score_2019")
        elif "diff" not in history:
            q = "difference?"
            a = abs(facts["2019"] - facts["2018"])
            history.append("diff")
        else:
            q = "<EOQ>"
            break

    assert q == "<EOQ>" or "diff" in history
    return abs(facts["2019"] - facts["2018"])

# LtM 玩具例子
result = ltm_expand_command("TURN RIGHT x2 + JUMP")
assert result == ["TURN RIGHT", "TURN RIGHT", "JUMP"]

# Successive 真实工程例子
diff = successive_drop_reasoner({"2018": 24, "2019": 31})
assert diff == 7

print(result)
print(diff)
```

如果你把它映射到真实 LLM 调用，LtM 的 prompt 结构通常长这样：

```python
def solve_with_ltm(model, original_task, subquestions):
    answers = []

    for sq in subquestions:
        prompt = f"""
你正在做多步推理。请只回答当前子问题。

原任务:
{original_task}

当前子问题:
{sq}

已知前置答案:
{answers}
"""
        # 这里调用模型，并把前面的答案拼进去
        a = model(prompt)
        answers.append(a)

    return answers[-1]
```

上面最重要的工程点只有一个：`已知前置答案` 必须结构化传入。不要只把整个历史对话原样拼进去，否则模型会把哪些是“已确认结果”、哪些是“待判断文本”混在一起。

Successive 的控制循环则更像一个 agent：

```python
def solve_with_successive(qd_model, qa_model, main_question, context, max_steps=8):
    history = []

    for step in range(max_steps):
        qd_prompt = f"""
主问题:
{main_question}

上下文:
{context}

历史:
{history}

请生成下一个最必要的子问题。
如果已经足够回答主问题，只输出 <EOQ>。
"""
        subq = qd_model(qd_prompt).strip()

        if subq == "<EOQ>":
            break

        qa_prompt = f"""
主问题:
{main_question}

当前子问题:
{subq}

上下文:
{context}

历史:
{history}

请只回答当前子问题。
"""
        ans = qa_model(qa_prompt).strip()

        # 这里应插入答案验证，例如格式检查、证据校验、置信度判断
        history.append({"subq": subq, "ans": ans})

    return history
```

这个循环里最重要的不是“能跑”，而是三件事：

1. 每轮必须有 `max_steps`
2. 必须识别 `<EOQ>`
3. 最好对 `ans` 做验证后再写入历史

否则它很容易变成“自己不断追问自己”的死循环。

---

## 工程权衡与常见坑

分解式 CoT 提高稳定性，但代价也很明确：调用次数更多、模板更复杂、错误传播更长。工程实现时，常见问题通常不是“模型不会推理”，而是“分解流程没有被严格执行”。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| LtM 缺少示例 | 模型直接跳到最终答案 | 没学会“先拆再答”的格式 | 提供 1 到 3 个标准拆解示例 |
| 子问题写得过大 | 每一步仍然很难 | 只是形式上拆分，没有降低复杂度 | 让每个子问题只做一种动作 |
| 中间答案未结构化 | 后续步骤误读前文 | 历史混成自然语言段落 | 用 JSON、字段列表或编号答案 |
| Successive 无终止符 | 无限追问 | 系统不知道何时结束 | 设 `<EOQ>` 和最大步数 |
| 中间错误累积 | 后面步骤全错 | 错答案被当作真前提 | 每步做校验或回滚 |
| token 成本过高 | 延迟和费用上升 | 多轮调用天然更贵 | 合并低价值子任务，限制步数 |

两个小场景最常见。

场景一：LtM 输入缺少示例，模型跳过子问题。  
比如你只写“请分步回答”，但没有给任何示范，模型常常会直接输出最终答案，因为在它看来“分步”只是表面要求，不是必须遵守的执行协议。解决方法不是重复说“请务必”，而是给清晰 few-shot 示例，并把输出格式卡死。

场景二：Successive 没设终止条件，系统一直生成新问题。  
例如它可能不断问“还有没有别的年份”“还要不要再验证一次”。如果没有 `<EOQ>`、`max_steps`、历史去重，这种循环在 agent 系统里非常常见。真正可用的实现通常还要加两层保护：子问题去重、连续低增益步数停止。

还有一个容易被忽略的坑：不是每个子问题都值得单独调用模型。若某一步只是简单字符串拼接或整数减法，应优先交给程序完成，而不是继续消耗 LLM token。Decomposed Prompting 在工程上最有价值的一点，就是把“需要模型的部分”和“规则可解的部分”分开。

---

## 替代方案与适用边界

分解式 CoT 不是唯一方案。它和标准 CoT、Tree-of-Thought（ToT，树式思考，可以并行探索多条推理分支）相比，各有适用边界。

| 方法 | 交互程度 | 可验证性 | 成本 | 适合任务 |
|---|---|---|---|---|
| 标准 CoT | 低 | 低到中 | 低 | 简单到中等难度推理 |
| Least-to-Most | 低到中 | 高 | 中 | 顺序明确的多步任务 |
| Decomposed Prompting | 中 | 高 | 中到高 | 模块清晰、可分治任务 |
| Successive Prompting | 高 | 很高 | 高 | agent、工具调用、需动态追问 |
| Tree-of-Thought | 很高 | 中到高 | 很高 | 搜索空间大、需多路径探索 |

一个典型失败对比很能说明问题。

直接 CoT 失败案例：  
主问题是“把 `TURN RIGHT ×2 + JUMP` 展开”。模型可能直接写成 `TURN RIGHT, JUMP, TURN RIGHT`，因为它在一次性生成时把重复和连接同时处理，发生了顺序混淆。

LtM 成功案例：  
先只做 `TURN RIGHT ×2`，得到 `TURN RIGHT, TURN RIGHT`；再拼接 `JUMP`。因为每一步只处理一种关系，错误机会明显减少。

所以可以用一句实用判断来选方法：

- 如果任务是固定顺序、多步依赖，优先 LtM。
- 如果任务能自然拆成“检索、抽取、计算、汇总”等模块，优先 Decomposed。
- 如果任务过程中需要动态决定下一步，或者要与工具、数据库、搜索接口互动，优先 Successive。
- 如果问题本身就是单步判断，先别急着上分解式 CoT。
- 如果你怀疑存在多条竞争性推理路径，才考虑 ToT 这类更重的搜索框架。

对零基础到初级工程师来说，最重要的不是记住术语，而是建立一个判断：分解的价值来自“中间状态可控且可验证”。一旦中间状态既不稳定也不可检查，分解本身就可能退化成更长、更贵的普通生成。

---

## 参考资料

1. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models  
   URL: https://www.emergentmind.com/papers/2205.10625  
   贡献：提出 LtM，核心结论是把复杂任务拆成按顺序求解的子问题，可显著提升组合泛化能力。文中 SCAN 实验是该方法最常被引用的结果之一。

2. Learn Prompting: Decomposed Prompting  
   URL: https://learnprompting.org/docs/advanced/decomposition/decomp  
   贡献：用教程形式解释任务分解、模块化 prompt 和递归分解，适合工程视角入门。

3. Successive Prompting Methods Overview  
   URL: https://www.emergentmind.com/topics/successive-prompting  
   贡献：总结 successive prompting 的交替问答机制、典型应用和实验收益，适合理解其 agent 化特征。

4. OwnYourAI: Least-to-Most Prompting Enables Complex Reasoning  
   URL: https://www.ownyourai.com/research-papers-3/least-to-most-prompting-enables-complex  
   贡献：提供 LtM 论文的易读摘要，适合先建立整体直觉，再回到原论文看实验细节。

建议阅读顺序是：先看 LtM 原论文理解“为什么拆分能提高组合泛化”，再看 Decomposed 和 Successive 的综述，最后回到你自己的任务，把子问题定义、依赖关系、终止条件写成明确模板。实验数字、数据集设定和评测口径仍应以原始文献为准。

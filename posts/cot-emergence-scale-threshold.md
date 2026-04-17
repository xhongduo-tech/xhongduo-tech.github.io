## 核心结论

思维链提示（Chain-of-Thought，简称 CoT）指的是：**在提示词中要求模型先写出中间推理步骤，再给出最终答案**。它不是对所有模型、所有任务都有效的通用技巧。当前较一致的实验结论是：**只有当模型规模达到一定阈值后，CoT 才会稳定提高复杂推理任务的正确率；在较小模型上，它常常没有收益，甚至会让结果更差。**

这件事容易被一个直觉误导：人类做多步题时经常写草稿，所以模型也应该“先写步骤再作答”。问题在于，**模型生成的步骤文本，不等于模型真的完成了可靠推理**。小模型经常只是生成“像推理”的语言外壳，句子完整、格式像样，但中间计算、条件转化或逻辑约束已经出错。步骤一旦出错，后续答案通常会被整条错误链拖偏。

把这种现象抽象成规模趋势，可以写成一个简化表达：

$$
\mathrm{Acc}_{\mathrm{CoT}}(s) \not\propto s,\qquad
\mathrm{Acc}_{\mathrm{CoT}}(s) \approx \text{U-shaped over some tasks}
$$

其中 $s$ 表示模型规模，$\mathrm{Acc}_{\mathrm{CoT}}(s)$ 表示使用 CoT 后的准确率。它的含义不是“所有任务都严格呈 U 形”，而是说在一类需要多步推理、且容易受错误中间步骤干扰的任务上，常能观察到下面这种趋势：

1. 小模型阶段，CoT 几乎没有帮助，甚至带来负收益。
2. 中等规模阶段，模型能写更长、更像样的步骤，但还不能稳定控制推理质量，容易掉进准确率谷底。
3. 更大模型阶段，模型开始具备更强的子任务分解和步骤一致性能力，CoT 才明显生效。

可以用更直观的玩具例子理解：

- 直接问：`12 × 47 = ?`
- CoT 问法：`请一步步计算 12 × 47，并给出最终答案`

这道题本身很短。对有能力的模型来说，直接回答已经足够；对能力不够的模型来说，强行展开步骤只是在增加出错机会。例如它可能写成：

```text
12 × 40 = 420
12 × 7 = 84
420 + 84 = 504
```

这里第一步就错了，后面的格式再完整也没有意义。也就是说，**CoT 有价值的前提，不是“模型会写步骤”，而是“模型能稳定完成步骤中涉及的子任务”**。

下表概括了最常见的三种情况：

| 场景 | 常见效果 | 本质原因 |
| --- | --- | --- |
| Standard prompt | 基线稳定 | 不强制生成中间链，错误传播路径更短 |
| CoT on 小模型 | 可能下降 | 生成的是“像推理的文本”，不是可靠推理 |
| CoT on 大模型 | 明显提升 | 能把复杂任务拆解为子问题，并维持步骤一致性 |

如果只记一个结论，应记这句：**思维链提示的收益是“能力阈值触发”的，不是“提示词自动附魔”的。**

---

## 问题定义与边界

定义先说清楚。**思维链提示**是指在输入中显式要求模型先写出中间推理过程，再输出最终答案。最常见的结构是：

```text
Prompt -> Chain -> Answer
```

也就是：

1. 输入问题；
2. 要求模型生成推理链；
3. 再由推理链得到最终答案。

它的目标不是把回答写长，而是**让模型在多步任务上显式展开中间状态**。所谓“多步任务”，通常包括：

| 任务类型 | 为什么需要多步 |
| --- | --- |
| 多步算术 | 需要中间量计算与累计 |
| 符号推理 | 需要按照规则逐步变换 |
| 组合规划 | 需要枚举、筛选、约束检查 |
| 条件问答 | 需要跨句整合条件并排除冲突 |

这里要区分两个概念：

| 概念 | 含义 | 是否等同 |
| --- | --- | --- |
| 长回答 | 输出很多文字 | 否 |
| 推理链 | 输出与求解过程对应的中间步骤 | 否 |

很多新手把这两者混在一起，以为“回答更长”就等于“推理更强”。这并不成立。模型可以写很多话，但这些话未必对求解有帮助。

CoT 的适用边界通常可以从四个维度判断：

| 维度 | 适合 CoT | 不适合 CoT |
| --- | --- | --- |
| Model size | 大模型，尤其是已表现出较强推理能力的模型 | 小模型或中小模型 |
| Task complexity | 多步数学、逻辑推理、规划问题 | 单步判断、事实检索、短分类 |
| Token budget | 允许更长输入输出 | 成本和时延敏感 |
| Reliability need | 可配合解析器、验证器、投票机制 | 直接把长文本当最终结果 |

再看最基础的算术题：

- 直接问：`12 × 47 = ?`
- 展开步骤：`12 × 47 = 12 × (40 + 7) = 480 + 84 = 564`

这个例子能说明三个边界。

第一，**收益空间取决于题目复杂度**。单步题本来就不需要太多中间状态，CoT 的提升空间天然有限。  
第二，**错误面会放大**。直接回答只需要一个结果；CoT 则要求模型先生成多个中间步骤，每一步都可能出错。  
第三，**文本可读性不等于推理可靠性**。模型可能把句子写得很顺，但关键一步算错。

因此，CoT 的边界可以更精确地概括为：

$$
\text{Use CoT if } 
\bigl(\text{task is multi-step}\bigr)
\land
\bigl(\text{model can handle subproblems}\bigr)
\land
\bigl(\text{budget allows longer outputs}\bigr)
\land
\bigl(\text{verification is available}\bigr)
$$

只要其中两三项不满足，CoT 的收益就会明显打折，甚至转负。

---

## 核心机制与推导

为什么 CoT 的效果不是“模型越大越一直变好”，而常表现出阈值和谷底？核心原因可以概括成两个能力阶段。

第一阶段，模型先学到的是**表面推理模板**。也就是它知道什么样的句子“像在解题”，比如“先算 A，再算 B，最后得到 C”。  
第二阶段，模型才逐渐学到**真正有用的子任务分解能力**。也就是它不仅会写“先算什么”，而且真的知道“为什么该先算它、它和后续步骤如何衔接”。

这两种能力差别很大：

| 能力 | 表现 | 对 CoT 的意义 |
| --- | --- | --- |
| 表面模板能力 | 会生成“首先、然后、最后”这类解释结构 | 让答案看起来更像推理 |
| 子任务分解能力 | 能找到求解所需的关键中间量 | 让步骤真正提高正确率 |

中等规模模型最容易出问题。因为它已经足够强，可以生成完整、流畅、结构清楚的推理文本；但它还不够强，无法稳定区分“关键步骤”和“干扰模式”。于是会出现一种典型失败：**说得越来越像推理，错得也越来越系统**。

把这个过程写成简化函数，可以表示为：

$$
\mathrm{Acc}_{\mathrm{CoT}}
=
f\bigl(
\text{model size},
\text{task complexity},
\text{subproblem skill},
\text{interference}
\bigr)
$$

其中：

- `model size`：模型参数规模或有效能力规模；
- `task complexity`：任务需要多少中间状态；
- `subproblem skill`：模型对关键子任务的掌握程度；
- `interference`：题目中会诱导错误模式的干扰强度。

这里最关键的是 `subproblem skill` 和 `interference`。如果模型只能写步骤模板，不能稳定解决子任务，那么 CoT 只是在把错误写出来；如果题目带有高干扰，错误链还会被持续放大。

可以把准确率写成一个更具解释性的分解：

$$
\mathrm{Acc}_{\mathrm{CoT}}
\approx
P(\text{decompose correctly})
\times
P(\text{execute each step correctly} \mid \text{correct decomposition})
\times
P(\text{preserve consistency})
$$

这个式子说明了 CoT 为什么难。模型不仅要：

1. 找对拆解方式；
2. 每一步都执行正确；
3. 还要保证前后步骤一致。

这比“直接输出一个答案”要求更高。

研究中常被引用的现象是，一些反向缩放或高干扰推理任务上，**中等规模模型可能比更小模型更容易被错误链条拖垮，而更大模型才重新回升**。例如在相关文献中，PaLM 某些干扰型任务的表现曾出现过近似这样的对比：**62B 规模在某些设置下大约只有 26.8% 左右准确率，而 540B 规模可回升到约 59.9%**。这组数字的价值不在于给出一个固定阈值，而在于说明：**中等模型不是“再大一点的小模型”，它可能正位于最容易被错误 CoT 干扰的区间。**

把这种阶段差异总结成表格更清楚：

| 阶段 | 模型典型行为 | CoT 效果 |
| --- | --- | --- |
| 小模型 | 步骤文本本身就不稳定 | 无收益或轻微负收益 |
| 中等模型 | 能写长步骤，但关键判断不稳 | 容易落入 U 形谷底 |
| 大模型 | 能识别关键中间量并保持步骤一致 | CoT 开始明显提升准确率 |

再看一个对新手更友好的例子。假设题目要求保留有效数字：

```text
把 0.004860 保留三位有效数字
```

正确求解至少要完成三个子任务：

| 子任务 | 正确要求 |
| --- | --- |
| 识别有效数字起点 | 前导零不计入有效数字 |
| 统计位数 | `4、8、6、0` 中取前三位有效数字 |
| 按规则舍入 | 第四位决定是否进位 |

中等模型常见的问题不是“不会写解释”，而是会把“前导零”“末尾零”“小数点位置”混成一团，最后给出形式完整但规则错误的链条。更大模型则更可能先锁定“有效数字定义”这一核心子任务，再展开后续判断。

这也是为什么在 GSM8K 这类小学奥数风格数据集上，few-shot CoT 常对大模型有效。任务真正需要的是：

1. 从题目中抽取已知条件；
2. 构造中间量；
3. 做连续运算；
4. 回到题目所问对象。

CoT 之所以帮助大模型，不是因为“输出更长”，而是因为**更长的输出中承载了可控的中间状态表示**。一旦模型具备这种能力，CoT 才会从“表面解释”变成“有效分解”。

---

## 代码实现

工程里不应把 CoT 理解成一句 `Let's think step by step` 就结束。可落地的做法通常是：**只在合适模型和合适任务上启用 CoT，并对输出做结构化解析和程序校验**。因为线上系统要对结果负责，不能把长推理文本直接当成可信真相。

一个最小流程通常是：

```text
prompt -> model output -> parse -> verify -> final answer
```

下面给出一个可以直接运行的 Python 示例。它做三件事：

1. 构造 few-shot CoT prompt；
2. 解析模型返回的 `CHAIN` 和 `ANSWER`；
3. 用程序验证算术题的最终答案是否正确。

```python
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelOutput:
    chain: str
    answer: str


def build_prompt(question: str) -> str:
    demos = [
        {
            "q": "一个盒子里有 3 袋糖，每袋 4 颗，又放入 2 颗。一共有多少颗？",
            "chain": "先算 3 × 4 = 12，再加上 2，得到 14。",
            "a": "14",
        },
        {
            "q": "一本书 25 元，买 4 本，再加一个 6 元书签，一共多少钱？",
            "chain": "先算 25 × 4 = 100，再加上 6，得到 106。",
            "a": "106",
        },
    ]

    parts = []
    for demo in demos:
        parts.append(
            f"Q: {demo['q']}\n"
            f"CHAIN: {demo['chain']}\n"
            f"ANSWER: {demo['a']}\n"
        )

    parts.append(f"Q: {question}\nCHAIN:")
    return "\n".join(parts)


def parse_output(text: str) -> ModelOutput:
    chain_match = re.search(r"CHAIN:\s*(.*?)\s*ANSWER:", text, re.S)
    answer_match = re.search(r"ANSWER:\s*([^\n]+)", text)

    chain = chain_match.group(1).strip() if chain_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return ModelOutput(chain=chain, answer=answer)


def extract_mul_operands(question: str) -> Optional[tuple[int, int]]:
    match = re.search(r"(\d+)\s*\*\s*(\d+)", question)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def verify_mul_question(question: str, answer: str) -> bool:
    operands = extract_mul_operands(question)
    if operands is None:
        return False

    x, y = operands
    expected = x * y

    try:
        return int(answer) == expected
    except ValueError:
        return False


def simulate_model_response(question: str) -> str:
    # 这里只是模拟一个“模型输出”，便于演示 parse + verify 流程。
    if question == "12 * 47 = ?":
        return (
            "Q: 12 * 47 = ?\n"
            "CHAIN: 12 * 40 = 480\n"
            "12 * 7 = 84\n"
            "480 + 84 = 564\n"
            "ANSWER: 564\n"
        )
    return (
        f"Q: {question}\n"
        "CHAIN: 无法解析\n"
        "ANSWER: unknown\n"
    )


def answer_with_verification(question: str) -> dict:
    prompt = build_prompt(question)
    raw_output = simulate_model_response(question)
    parsed = parse_output(raw_output)
    passed = verify_mul_question(question, parsed.answer)

    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "chain": parsed.chain,
        "answer": parsed.answer,
        "verified": passed,
    }


if __name__ == "__main__":
    result = answer_with_verification("12 * 47 = ?")

    assert "12 * 40 = 480" in result["chain"]
    assert result["answer"] == "564"
    assert result["verified"] is True

    bad = parse_output(
        "Q: 12 * 47 = ?\n"
        "CHAIN: 12 * 40 = 420\n"
        "12 * 7 = 84\n"
        "420 + 84 = 504\n"
        "ANSWER: 504\n"
    )
    assert verify_mul_question("12 * 47 = ?", bad.answer) is False

    print("Prompt sent to model:")
    print(result["prompt"])
    print("\nParsed answer:", result["answer"])
    print("Verified:", result["verified"])
```

这段代码虽然没有接入真实 API，但逻辑是完整可运行的。它把工程里最关键的三个部件拆开了：

| 部件 | 作用 | 为什么需要 |
| --- | --- | --- |
| Prompt 模板 | 约束输出格式 | 便于后续程序解析 |
| Parser | 从长文本中抽取结构化字段 | 避免业务系统直接读整段自然语言 |
| Verifier | 检查最终答案是否满足题目约束 | 防止错误链条进入线上结果 |

如果接入真实模型，流程通常会长这样：

```python
def call_llm(question: str, llm) -> dict:
    prompt = build_prompt(question)
    raw_text = llm.generate(prompt)  # 伪代码
    parsed = parse_output(raw_text)
    ok = verify_mul_question(question, parsed.answer)

    return {
        "question": question,
        "chain": parsed.chain,
        "answer": parsed.answer,
        "verified": ok,
        "raw_text": raw_text,
    }
```

新手在这里最容易忽略的一点是：**解析和校验不是“附加优化”，而是 CoT 工程化的核心组成部分。**  
原因很简单。CoT 会让模型输出更多内容，内容越多，错误落点就越多。没有结构化处理，系统就只能“读一段很像推理的话，然后选择相信它”，这在生产环境里风险很高。

还可以把是否接受答案写成一个简单决策规则：

$$
\text{Accept} =
\begin{cases}
1, & \text{if parse succeeds and verify(answer)=True} \\
0, & \text{otherwise}
\end{cases}
$$

即使将来你把 verifier 换成更复杂的约束检查、SQL 执行、代码单测或符号求解器，工程思想也是一样的：**CoT 负责提供候选推理，验证模块负责决定是否信任结果。**

实践中还有两个常用原则：

| 原则 | 解释 |
| --- | --- |
| few-shot 示例要短 | 示例太长会挤占上下文，还会增加时延和费用 |
| 优先校验最终答案 | 中间链条可能只是解释文本，不一定可靠映射内部推理 |

因此，真正可落地的 CoT 系统不是“让模型说得更多”，而是“让模型在受控格式内给出可检查的候选解”。

---

## 工程权衡与常见坑

CoT 的第一笔成本是 token。无论是 few-shot 示例，还是更长的输出链条，都会直接推高调用费用和响应时延。一个常见的粗略成本公式是：

$$
\mathrm{Cost} \approx (T_{\text{input}} + T_{\text{output}})\times \mathrm{price}
$$

其中：

- $T_{\text{input}}$：输入 token 数；
- $T_{\text{output}}$：输出 token 数；
- `price`：模型按 token 计费的价格。

CoT 同时会增加这两项：

1. few-shot 示例会拉高 $T_{\text{input}}$；
2. 中间推理链会拉高 $T_{\text{output}}$。

如果系统还有重试、投票、多样本采样，成本会进一步放大：

$$
\mathrm{Total\ Cost}
\approx
n \times (T_{\text{input}} + T_{\text{output}})\times \mathrm{price}
$$

其中 $n$ 是采样次数。  
所以工程上的关键问题从来不是“CoT 能不能提高一点准确率”，而是：

**这点提升，是否足以覆盖新增的 token 成本、延迟成本和失败路径复杂度？**

可以用下面这个判断表做初筛：

| 场景 | 结论 | 原因 |
| --- | --- | --- |
| 大模型 + GSM8K 类多步推理 | 通常值得尝试 CoT | 收益空间大，步骤能承载中间状态 |
| 小模型 + 复杂算术 | 更适合短 prompt + verifier | 长链条更容易放大错误 |
| 简单分类/信息抽取 | CoT 基本浪费 | 任务本来就不需要多步求解 |
| 严格延迟 SLA 场景 | 先算 token 预算，再决定是否启用 | CoT 往往显著拉长响应时间 |

常见坑至少有四类，而且它们都很典型：

| 坑 | 表面现象 | 实际后果 | 规避策略 |
| --- | --- | --- | --- |
| 小模型强行加 CoT | 回答更长、更像推理 | 错误链条拖低准确率 | 改用短提示 + 程序验证 |
| 简单任务也加 CoT | “感觉更认真” | 成本上升但收益几乎为零 | 仅对多步任务启用 |
| few-shot 示例过长 | 提示词看起来很完整 | 上下文被挤占，时延上升 | 控制示例数量和长度 |
| 直接信任链条 | 解释很流畅 | 错误步骤直接进入业务结果 | 抽取最终答案并做校验 |

还有一个很常见、但不容易被察觉的问题：**长链条会让错误更难被人类肉眼发现**。  
如果模型直接给出一个错误答案，你往往一眼就能看到它错了；但如果模型先给你一大段“像模像样”的推理，读者很容易被语气、格式和局部正确步骤迷惑，忽略真正出错的关键点。这在人工审核流程中尤其危险。

新手需要特别警惕下面这组“假信号”：

| 假信号 | 为什么不可靠 |
| --- | --- |
| 解释很长 | 长度不代表正确性 |
| 语气很自信 | 自信只是语言风格，不是证据 |
| 公式写出来了 | 公式可能套对了，代值仍然错 |
| 前两步是对的 | 后面关键一步错了，结论依然错 |

因此，在成本敏感或小模型场景里，**错误链条往往比没有链条更危险**。  
没有链条时，系统只需要校验一个最终结果；有链条时，系统会收到一大段表面可信、但可能在关键处失真的文本，这会增加审核难度和集成风险。

---

## 替代方案与适用边界

如果模型规模不足，或者任务不适合长链条，就不应机械追求 CoT。更稳妥的替代路线通常有三类：

| 方案 | 规模要求 | 代表任务 | 优点 | 局限 |
| --- | --- | --- | --- | --- |
| 短 prompt + verifier | 低到中 | 算术、格式化提取、规则判断 | 成本低、实现简单、稳定 | 适用任务范围有限 |
| self-consistency + verification | 中到高 | 多步推理、确定性较强问题 | 可降低单次采样偶然误差 | 成本高于单次调用 |
| 微调/专项训练 | 视数据与资源而定 | 垂直领域推理、固定任务流 | 线上行为更稳定 | 前期投入高、需要数据 |

这里先解释一下 self-consistency。它的核心思想是：**对同一个问题采样多次，再用投票或验证器选出更可信的答案**。它关注的是“多次独立尝试后的稳定答案”，而不是“单次输出必须写出一条完美长链”。

一个简化流程可以写成：

$$
\hat{y}
=
\operatorname*{argmax}_{y}
\sum_{i=1}^{n}
\mathbf{1}(y_i = y)
$$

这里 $y_1,\dots,y_n$ 是模型多次采样得到的候选答案，$\hat{y}$ 是出现次数最多的结果。若再结合验证器，可以变成“先过滤掉不满足约束的候选，再投票”。

这对一些任务很有用，因为模型的错误往往具有随机性，而正确答案在多次采样里更可能重复出现。

对于小模型做复杂算术，一个很实用的工程模式是：

1. 用非常短的 prompt，直接要求只输出答案；
2. 采样多次，得到若干候选；
3. 用 verifier 检查每个候选是否满足约束；
4. 在通过验证的候选中选择结果。

例如：

```text
只输出最终整数答案，不要解释。
问题：12 * 47 = ?
```

这种写法虽然“没有推理过程”，但在小模型上反而更稳。原因是你把模型的任务压缩成“给候选结果”，而把“判断结果是否可靠”交给程序。

而在大模型做复杂数学、组合规划、代码推理时，更合适的模式通常是：

1. few-shot CoT；
2. 给予适度的推理空间；
3. 抽取最终答案；
4. 使用校验器或外部工具兜底。

两种方案的对比如下：

| 条件 | 更合适的方法 |
| --- | --- |
| 模型较小，任务规则清晰，可程序校验 | 短 prompt + verifier |
| 模型较强，任务是多步推理，单次输出不稳定 | self-consistency + verification |
| 模型很强，任务复杂且需要中间状态 | few-shot CoT + verification |
| 任务长期固定、调用量大 | 微调或专项训练 |

所以边界非常明确：**CoT 不是推理任务的默认总开关，而是“大模型 + 多步任务 + 有预算 + 有校验”条件下的一种高收益策略。**  
当这些条件不满足时，验证、采样、微调往往比长思维链更可靠，也更容易工程化。

---

## 参考资料

下表列出理解这个主题时最值得优先阅读的资料。它们分别回答三个问题：CoT 何时有效、为什么会出现规模阈值或 U 形现象、工程上还能怎么做。

| 资料 | 类型 | 核心贡献 |
| --- | --- | --- |
| Wei et al., 2022, *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | 论文 | 系统展示 few-shot CoT 在大模型上的推理提升，并强调规模是关键前提 |
| Wang et al., 2022, *Self-Consistency Improves Chain of Thought Reasoning in Language Models* | 论文 | 提出多次采样后投票的 self-consistency，说明“多样解再聚合”可提升推理稳定性 |
| BIG-Bench / inverse scaling 相关论文与项目页面 | 基准与论文 | 展示某些任务上更大中等模型可能先变差，再在更大规模上回升 |
| Emergent Mind: *U-Shaped Reasoning Performance* | 综述页面 | 总结推理性能随规模出现 U 形或反向缩放现象，帮助理解谷底区 |
| Emergent Mind: *Chain-of-Thought Prompting* | 综述页面 | 梳理 CoT、self-consistency、验证机制及其适用边界 |

如果继续深入，建议按下面顺序读：

1. 先读 Wei 等人的 CoT 论文实验部分，重点看“标准提示 vs few-shot CoT”的对比设置。
2. 再读 self-consistency 论文，理解为什么“同题多采样再聚合”有时比单条长链更稳。
3. 最后看 U 形或反向缩放相关综述，理解为什么“模型更大一点”并不自动意味着“更适合 CoT”。

对初学者来说，最重要的不是记住某个具体参数规模，而是建立这个判断框架：

| 判断问题 | 应该先问什么 |
| --- | --- |
| 要不要用 CoT？ | 任务是不是多步推理？ |
| CoT 会不会有收益？ | 模型是否已经具备稳定子任务能力？ |
| 线上能不能落地？ | token、时延、解析、校验是否可接受？ |
| 如果 CoT 不稳怎么办？ | 能否改用验证、采样或专项训练？ |

最终应回到本文开头的核心结论：**思维链提示的有效性取决于能力阈值，而不是提示词表面形式。**  
模型规模不足时，长链条往往只是把错误写得更完整；模型规模足够大时，CoT 才可能真正把复杂问题拆成可控的中间步骤，并转化为准确率提升。

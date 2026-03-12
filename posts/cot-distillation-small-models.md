## 核心结论

思维链蒸馏（Chain-of-Thought Distillation, CoT Distillation）指的是：先让大模型为题目生成中间推理过程，再把“题目 + 推理链 + 最终答案”作为监督数据，用来微调小模型。它迁移的不是某个单独答案，而是“如何分步求解”的输出模式。

第一，CoT 蒸馏确实能迁移一部分推理能力，但前提不是“有 CoT 就够”，而是“高质量 CoT + 与学生容量匹配的难度”。Magister 等人在 ACL 2023 中报告，T5 XXL 在 GSM8K 上的基线准确率是 8.11%，用 PaLM 540B 生成的 CoT 数据微调后提升到 21.99%；若再用外部计算器修正中间算术错误，可到 38.21%；只用 20% 的蒸馏数据时也能到 11.22%。这说明推理能力可以迁移，但高度依赖数据质量、样本量和后处理机制。来源：<https://aclanthology.org/2023.acl-short.151/>

第二，小模型并不总能从更长、更强的推理中受益。Li 等人在 Findings of ACL 2025 提出 small model learnability gap：参数规模不大的学生模型，往往更适合较短、较稳定、分布更接近自己的 CoT 数据。对 3B 级学生，直接蒸馏超长推理链，或者直接模仿更强教师的输出分布，可能让平均成绩下降而不是上升。来源：<https://aclanthology.org/2025.findings-acl.1301/>

第三，工程上最稳的做法通常不是盲目追求“最强教师”，而是做混合蒸馏。Mix-Long 把长链和短链混合，Mix-Large 把强教师和弱教师混合。Li 等人的实验里，Qwen2.5-3B 用 Mix-Long 和 Mix-Large 后，平均分分别到 45.9 和 45.8，都高于单独长链或单独强教师。结论不是“长链没用”，而是“长链要和学生容量配比后才有用”。

| 结果片段 | 设置 | 指标 |
|---|---|---:|
| T5 XXL baseline | GSM8K | 8.11 |
| PaLM 540B CoT 蒸馏 | GSM8K | 21.99 |
| PaLM 540B CoT 蒸馏 + calculator | GSM8K | 38.21 |
| 20% CoT 数据蒸馏 | T5 XXL / GSM8K | 11.22 |
| Mix-Long | Qwen2.5-3B 平均分 | 45.9 |
| Mix-Large | Qwen2.5-3B 平均分 | 45.8 |

如果只记一句话，可以记成：CoT 蒸馏有效，但对小模型最关键的不是“推理写得多长”，而是“推理是否正确、是否稳定、是否在学生学得动的复杂度范围内”。

---

## 问题定义与边界

问题可以形式化为一个标准的监督学习过程。给定教师模型 $T$、学生模型 $S$、输入问题 $x$、教师生成的推理链 $r$、最终答案 $y$，蒸馏阶段的目标是让学生学习：

$$
x \rightarrow (r, y)
$$

也就是，输入一道题，输出一段中间推理，再给出最终答案。这里的训练对象不是隐式“思考能力”，而是显式的序列生成行为。

Magister 等人的关键做法之一，是在教师生成 CoT 时使用目标提示：先把正确答案给教师，再让教师围绕正确答案补出推理过程。这样做的作用不是“作弊”，而是降低错误链比例。因为蒸馏阶段最怕的不是答案错，而是“答案对但过程错”，这种样本会把错误步骤当成监督信号传给学生。

边界也很明确。CoT 蒸馏不是任何场景下都稳定有效，它受到四类变量约束：

| 变量 | 过强时的问题 | 过弱时的问题 |
|---|---|---|
| CoT 长度 | 小模型容易学到冗长、绕圈、终止不稳 | 中间信息太少，学不到步骤模板 |
| 教师强度 | 输出分布太复杂，学生吸收不了 | 上限偏低，蒸馏收益受限 |
| 数据规模 | 生成、清洗、训练成本明显上升 | 容易记题型，泛化差 |
| 数据质量 | 错链被当真理学走，放大错误模式 | 过度过滤会损失覆盖面与多样性 |

对新手最容易混淆的一点是：CoT 蒸馏优化的是“可模仿的过程”，不是抽象的“会不会思考”。如果教师给出的是一条人类看起来很漂亮、但对小模型来说过长过难的题解，那么学生学到的往往不是更强的推理，而是更长的输出、更高的解码不稳定性，以及更差的终止行为。

玩具例子很直观。题目是“3 个苹果加 2 个苹果等于几”。

- 短链：`3 + 2 = 5，所以答案是 5。`
- 长链：定义变量、解释加法意义、拆分集合、验证结果、再复述答案。

对人来说，第二种也能看懂；但对一个 1B 到 3B 的学生模型，第一种更像可以重复学习的稳定模板，第二种则引入了额外语言负担。

真实工程里，很多团队一开始都会犯同一个错误：直接拿 70B 或更强模型生成长推理链，再整批做 SFT。训练 loss 往往会下降，但线上推理可能出现三种退化：答案不稳定、输出过长、容易停不下来。原因通常不是“模型没学会”，而是“模型学到了不适合自己容量的输出分布”。

---

## 核心机制与推导

Li 等人把这个问题拆成两个差值，分别衡量“长链是否真的更适合学生”和“强教师是否真的更适合学生”：

$$
\Delta_{Long} = P_{Long} - P_{Short}
$$

$$
\Delta_{Large} = P_{Large} - P_{Small}
$$

其中：

- $P_{Long}$：学生用长 CoT 数据微调后的平均成绩
- $P_{Short}$：学生用短 CoT 数据微调后的平均成绩
- $P_{Large}$：学生用强教师数据训练后的平均成绩
- $P_{Small}$：学生用较弱教师数据训练后的平均成绩

这两个量的解释很直接：

- 如果 $\Delta_{Long} > 0$，长链对该学生是净收益。
- 如果 $\Delta_{Long} < 0$，长链带来的噪声和学习负担超过了有效信息。
- 如果 $\Delta_{Large} > 0$，强教师的分布对学生是可吸收的。
- 如果 $\Delta_{Large} < 0$，强教师太强，学生反而学不进去。

在 Qwen2.5-3B 上，论文报告 $P_{Long}=40.3$、$P_{Short}=43.4$，所以：

$$
\Delta_{Long} = 40.3 - 43.4 = -3.1
$$

这表示对这个 3B 学生，单独使用长链数据是负收益。类似地，$P_{Large}=39.7$、$P_{Small}=39.4$，则：

$$
\Delta_{Large} = 39.7 - 39.4 = 0.3
$$

这不是明显优势，说明“更强教师”并没有转化成显著更好的学生表现。

从学习目标看，学生优化的是条件概率：

$$
p_\theta(r, y \mid x)
$$

常见的监督损失写成：

$$
\mathcal{L}_{SFT} = - \sum_{t=1}^{n}\log p_\theta(z_t \mid x, z_{<t})
$$

其中 $z = [r; y]$，即把推理链和最终答案拼成一个目标序列。问题在于，当 $r$ 很长、风格很复杂、分支很多时，学生要同时学习四种东西：

- 语言表述风格
- 步骤分解模式
- 局部算术或符号操作
- 何时结束输出

对小模型来说，这四项经常会互相竞争参数容量。于是模型可能“学会了写得像在推理”，却没有真正提高答案正确率。

混合蒸馏的核心，就是别让训练数据集中在单一复杂度上。可以把训练分布写成：

$$
D_{mix} = \lambda D_{long} + (1-\lambda)D_{short}
$$

或者：

$$
D_{mix} = \alpha D_{strong} + (1-\alpha)D_{weak}
$$

Li 等人的实验中，一个有效配置是让混合比例偏向更容易学的样本，例如在 Qwen2.5-3B 上，Mix-Long 使用长链与普通短链约 1:4 的混合。直观理解是：少量高复杂度样本提供上限，多数低复杂度样本保证学生先学稳基本模式。

再看一个新手更容易代入的例子。题目是“一个数加 4 等于 9，求这个数”。

- 短链：`9 - 4 = 5，所以答案是 5。`
- 长链：设未知数 $x$，写出 $x+4=9$，移项得 $x=9-4$，再讨论等式变形，再代回验证。

如果学生模型本来就弱，第二种链条并不一定比第一种更有价值。因为对于这类题目，真正要迁移的是“从等式反推未知数”的模板，而不是完整教材式展开。

---

## 代码实现

最小可用实现通常分三步：教师生成、样本过滤、混合构造。下面给出一个可以直接运行的 Python 示例，不依赖第三方库，演示四件事：

1. 解析最终答案  
2. 过滤答案不一致或 CoT 过长的样本  
3. 构造 Mix-Long / Mix-Large 数据  
4. 输出可用于后续 SFT 的训练样本  

```python
from dataclasses import dataclass
from typing import Iterable, List, Optional
import re


@dataclass
class Example:
    question: str
    cot: str
    answer: str
    teacher: str          # e.g. "strong" / "weak"
    cot_type: str         # e.g. "long" / "short"


ANSWER_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")


def normalize_answer(text: str) -> str:
    text = text.strip()
    match = ANSWER_RE.search(text)
    return match.group(1) if match else text


def token_count(text: str) -> int:
    return len(text.split())


def final_answer_from_cot(cot: str) -> Optional[str]:
    patterns = [
        r"answer is\s+(-?\d+(?:\.\d+)?)",
        r"答案是\s*(-?\d+(?:\.\d+)?)",
        r"therefore[, ]+\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cot, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def keep_example(ex: Example, max_len: int = 120) -> bool:
    predicted = final_answer_from_cot(ex.cot)
    if predicted is None:
        return False
    if normalize_answer(predicted) != normalize_answer(ex.answer):
        return False
    if token_count(ex.cot) > max_len:
        return False
    return True


def filter_examples(examples: Iterable[Example], max_len: int = 120) -> List[Example]:
    return [ex for ex in examples if keep_example(ex, max_len=max_len)]


def build_mix_long(examples: Iterable[Example], long_ratio: int = 1, short_ratio: int = 4) -> List[Example]:
    longs = [ex for ex in examples if ex.cot_type == "long"]
    shorts = [ex for ex in examples if ex.cot_type == "short"]
    if not longs or not shorts:
        return []
    k_long = min(len(longs), long_ratio)
    k_short = min(len(shorts), short_ratio)
    return longs[:k_long] + shorts[:k_short]


def build_mix_large(examples: Iterable[Example], strong_ratio: int = 1, weak_ratio: int = 1) -> List[Example]:
    strong = [ex for ex in examples if ex.teacher == "strong"]
    weak = [ex for ex in examples if ex.teacher == "weak"]
    if not strong or not weak:
        return []
    k_strong = min(len(strong), strong_ratio)
    k_weak = min(len(weak), weak_ratio)
    return strong[:k_strong] + weak[:k_weak]


def to_sft_record(ex: Example) -> dict:
    target = f"{ex.cot}\nFinal answer: {normalize_answer(ex.answer)}"
    return {"input": ex.question, "target": target}


if __name__ == "__main__":
    raw_examples = [
        Example(
            question="Roger has 5 tennis balls and buys 2 cans with 3 balls each. How many balls now?",
            cot="Roger starts with 5. 2 cans * 3 balls = 6. 5 + 6 = 11. The answer is 11.",
            answer="11",
            teacher="strong",
            cot_type="short",
        ),
        Example(
            question="Roger has 5 tennis balls and buys 2 cans with 3 balls each. How many balls now?",
            cot="Let x be the original count. He has 5. Each can has 3. Two cans make 6. "
                "Then 5 + 6 = 12. The answer is 12.",
            answer="11",
            teacher="strong",
            cot_type="long",
        ),
        Example(
            question="A number plus 4 equals 9. What is the number?",
            cot="9 - 4 = 5. The answer is 5.",
            answer="5",
            teacher="weak",
            cot_type="short",
        ),
        Example(
            question="A number plus 4 equals 9. What is the number?",
            cot="Set x + 4 = 9. Subtract 4 from both sides to get x = 5. The answer is 5.",
            answer="5",
            teacher="strong",
            cot_type="long",
        ),
    ]

    clean = filter_examples(raw_examples, max_len=40)
    mix_long = build_mix_long(clean, long_ratio=1, short_ratio=2)
    mix_large = build_mix_large(clean, strong_ratio=1, weak_ratio=1)

    print("clean examples:", len(clean))
    print("mix_long size:", len(mix_long))
    print("mix_large size:", len(mix_large))
    print("first sft record:", to_sft_record(clean[0]))
```

这段代码故意保持简单，但已经覆盖了真实流程中的三个关键判断：

- 最终答案是否和标注一致
- 推理链是否过长
- 是否按数据复杂度做混合

如果要进一步接近论文做法，可以继续加三类过滤器：

| 过滤器 | 目的 | 适用任务 |
|---|---|---|
| 答案一致性检查 | 去掉答案错的样本 | 所有任务 |
| 长度阈值检查 | 防止小模型被长链拖垮 | 小模型蒸馏 |
| 程序执行检查 | 去掉“答案对但过程错”的样本 | 数学、代码、符号推理 |

训练目标仍然是标准 SFT，只是目标序列通常写成 `cot + answer`。如果任务可程序验证，工程上更稳的做法不是保留自然语言 CoT 原文，而是把关键步骤转成可执行程序或结构化中间表示，再做验证和蒸馏，这就是 PaD 的核心思路。来源：<https://aclanthology.org/2024.naacl-long.142/>

---

## 工程权衡与常见坑

最常见的坑有四个，而且都不是“多加点数据就能解决”的问题。

第一，错误 CoT 比没有 CoT 更危险。普通 SFT 里，脏标签已经会伤模型；CoT 蒸馏里更严重，因为错误会出现在中间步骤，学生会把这些步骤学成模板。Magister 的做法是先给目标答案，再剔除错误链。PaD 更进一步，用程序替代自然语言链，并在解码中做逐步验证。

第二，小模型容易被长链拖垮。Li 等人的结果显示，Qwen2.5-3B 单独用 Long CoT 的平均分是 40.3，反而低于 Short CoT 的 43.4。这里的退化不是偶然噪声，而是一个结构性现象：学生模型容量有限，长链把一部分参数预算浪费在表述冗余上。

第三，数据太少时，蒸馏很容易过拟合。Magister 在 GSM8K 上给出的 4% 数据结果只有 6.29%，甚至低于 8.11% 的基线；20% 数据才到 11.22%。这说明“CoT 质量高”并不自动等于“少量样本足够”。

第四，答案正确不等于过程可学。现实中经常存在这种样本：教师最后答对了，但中间某一步算错，后面又靠语言模式绕回正确答案。对于学生，这类样本是有毒的，因为它会把错误推理和正确答案绑定在一起。

下面这张表能直接看出混合蒸馏对 3B 学生的稳定性收益：

| 方法 | MATH | AMC | GSM8K | OlympiadBench | AIME | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B Long CoT | 56.2 | 37.5 | 80.0 | 24.4 | 3.3 | 40.3 |
| Qwen2.5-3B Short CoT | 61.0 | 37.5 | 82.0 | 26.4 | 10.0 | 43.4 |
| Qwen2.5-3B Strong Teacher CoT | 57.5 | 35.0 | 80.0 | 25.9 | 0.0 | 39.7 |
| Mix-Long | 64.7 | 45.0 | 81.4 | 28.6 | 10.0 | 45.9 |
| Mix-Large | 65.8 | 42.5 | 81.7 | 29.0 | 10.0 | 45.8 |

这张表的含义不是“以后都不用长链”或“强教师没用”，而是：对小模型，单一复杂度的数据分布往往不稳，混合后更接近学生可学习区间。

可以把落地时的检查项记成一个最小清单：

| 检查项 | 建议 |
|---|---|
| 教师生成 | 尽量约束输出格式，避免自由发挥过长 |
| 样本过滤 | 至少做答案一致性过滤 |
| 长度控制 | 对 1B 到 3B 先从短链开始 |
| 配比实验 | 至少比较短链、长链、Mix-Long 三组 |
| 可验证任务 | 能程序检查就不要只信自然语言 CoT |
| 线上观察 | 重点看正确率、平均输出长度、停止稳定性 |

---

## 替代方案与适用边界

标准 CoT 蒸馏适合先做出一个能跑通的 baseline。它实现简单，数据格式直接，适用于大多数监督微调管线；但缺点也直接，就是对数据质量非常敏感。

Mix Distillation 适合参数规模较小的学生模型，尤其是 1B 到 7B 这个区间。它不要求你发明新的训练目标，只要求你承认一件事：不是所有教师样本都该被同等对待。长链、短链、强教师、弱教师，都可能有价值，但价值取决于学生容量。

PaD 则更适合可以程序校验的任务，例如算术、符号推理、结构化操作、部分代码任务。它的核心不是“让程序替代语言模型”，而是“让中间步骤进入可执行、可验证、可过滤的空间”。这是对 CoT 蒸馏最实际的质量增强之一。来源：<https://aclanthology.org/2024.naacl-long.142/>

| 方案 | 适用模型 | 数据要求 | 质量保障 | 局限 |
|---|---|---|---|---|
| 标准 CoT 蒸馏 | 所有规模 | 中到高 | 人工筛错、答案校验 | 错链容易混入 |
| Mix Distillation | 小到中模型 | 中等以上 | 长短链、强弱教师混合 | 需要配比实验 |
| PaD | 可程序验证任务 | 中等 | 编译、执行、逐步验证 | 不适合纯开放式主观推理 |

适用边界可以总结成四条经验法则：

- 学生极小，先短链，再尝试混合教师。
- 学生在 3B 左右，优先比较 `Short CoT`、`Mix-Long`、`Mix-Large`。
- 任务能执行校验，优先考虑 PaD，而不是直接相信自然语言步骤。
- 如果目标任务是开放式复杂思辨，而不是算术、符号或结构化推理，CoT 蒸馏收益会更不稳定，因为“过程正确”本身更难定义和验证。

---

## 参考资料

| 资料 | 出版信息 | 核心贡献 |
|---|---|---|
| [*Teaching Small Language Models to Reason*](https://aclanthology.org/2023.acl-short.151/) | Magister et al., ACL 2023 | 证明高质量 CoT 蒸馏可显著提升 T5 系列在算术、常识、符号任务上的表现，并给出数据规模与模型规模的关系 |
| [*Small Models Struggle to Learn from Strong Reasoners*](https://aclanthology.org/2025.findings-acl.1301/) | Li et al., Findings of ACL 2025 | 提出 small model learnability gap，定义 $\Delta_{Long}$ 与 $\Delta_{Large}$，并验证 Mix Distillation 对小模型更稳 |
| [*PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning*](https://aclanthology.org/2024.naacl-long.142/) | Zhu et al., NAACL 2024 | 用可执行程序替代自然语言 CoT，并加入自动过滤、自我修正与逐步验证，提高蒸馏质量 |
| [*Large Language Models Are Reasoning Teachers*](https://aclanthology.org/2023.acl-long.830/) | Ho et al., ACL 2023 | 系统讨论大模型作为推理教师时的数据构造、不同推理样本的多样性，以及对小模型迁移的价值 |

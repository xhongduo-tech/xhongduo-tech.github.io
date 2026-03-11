## 核心结论

C-Eval 是中文大语言模型的综合评测基准。它用同一套中文选择题、同一套提示模板和同一类计分方式，比较不同模型在中文知识理解与推理上的稳定性。与“让模型自由写一段答案”不同，C-Eval 把输出约束为 A、B、C、D 四选一，因此评分标准清晰，适合做离线对比和工程回归检测。

它的核心价值有三点。

第一，多学科覆盖。C-Eval 包含 52 个学科、13,948 道题，难度从初中延伸到专家级。你拿到的不是一个孤立总分，而是一张能力剖面图，可以继续看 STEM、社科、人文以及具体学科的强弱分布。

第二，计分低歧义。所有题目都是单项选择题，最常用指标是准确率。准确率的定义直接、可复现，适合纳入自动化流水线，而不需要人工判断“这段开放式回答到底算不算对”。

第三，few-shot 评测通常更接近真实工程使用方式。few-shot 的意思是，在正式题目前先放入少量示例题，让模型知道输出格式和答题方式。对于很多基础模型，这比 zero-shot 更稳定；但对部分已经强指令化的模型，zero-shot 也可能更好，因此必须固定模板后再比较，不要混着看。

可以先看一眼它的基本规模：

| 维度 | 数值 | 含义 |
| --- | --- | --- |
| 学科数量 | 52 | 覆盖 STEM、社科、人文、其他四大类 |
| 难度等级 | 4 | 初中、高中、大学、专家 |
| 题量 | 13,948 | 全部为中文四选一选择题 |
| 常用指标 | Accuracy | 正确题数占总题数比例 |
| 常见切分 | dev / val / test | few-shot 示例、调参集、评测集 |

一个最小例子就够理解这个基准。假设有 10 道题，模型答对 7 道，那么准确率是：

$$
\frac{7}{10}=70\%
$$

C-Eval 的本质，只是把这个最基本的计分方法扩展成跨 52 门学科、跨多个难度层级的标准化考试。它不直接回答“模型是否会聊天”，而是回答“模型在中文知识和推理题上是否稳定”。

---

## 问题定义与边界

C-Eval 要解决的问题是：如何在中文环境下，用一套低歧义、可复现的方法，评估大语言模型的知识掌握和推理能力。

这里有三个关键词。

第一，中文环境。很多通用基准以英文为主，中文题面的知识表达、术语分布、选项干扰方式并不完全相同。C-Eval 的价值首先在于，它把评测语言固定为中文。

第二，知识与推理。C-Eval 的题目不是简单的字面匹配。很多题要求模型理解题干、区分干扰项、结合常识或学科知识做推断。但这种推理主要发生在“选择题约束”之内，而不是长篇生成任务里。

第三，可复现。所有模型面对的是相同题集、相近模板和统一指标，这使得结果可以横向比较，也方便做版本回归。

最基础的评价公式是准确率：

$$
\text{Accuracy}=\frac{\text{正确答案数}}{\text{总题数}}
$$

如果要看单个学科，例如“计算机网络”一科，公式一样：

$$
\text{Accuracy}_{\text{计算机网络}}=\frac{\text{该学科答对题数}}{\text{该学科总题数}}
$$

工程上还常见两种聚合口径：

| 聚合方式 | 计算方式 | 适合场景 |
| --- | --- | --- |
| Overall / Micro Accuracy | 把所有题放在一起算总准确率 | 看总体回归 |
| Subject Average / Macro Average | 先算每学科准确率，再对学科平均 | 看是否被大题量学科“稀释” |

新手要先建立一个边界意识：C-Eval 测的是“选择题上的中文知识与推理”，不是“模型全部能力”。

它**不直接覆盖**以下能力：

| 能力 | C-Eval 是否直接覆盖 | 原因 |
| --- | --- | --- |
| 长文写作 | 否 | 输出被约束为选项，不看文章结构与表达质量 |
| 多轮对话 | 否 | 题目通常是单轮问答 |
| 工具调用 | 否 | 不要求搜索、代码执行或外部 API |
| 代码生成 | 否 | 不以程序正确性为评分标准 |
| 长上下文处理 | 有限 | 题目长度有限，不能代表超长文档场景 |

切分边界也要说清楚。官方仓库中每个学科通常包含 `dev`、`val`、`test` 三个部分：

| 切分 | 作用 | 应该怎么用 |
| --- | --- | --- |
| `dev` | few-shot 示例题，通常带解析 | 组装 prompt，不用于正式计分 |
| `val` | 调参和本地回归 | 比较模型版本、比较提示模板 |
| `test` | 正式评测或独立报告 | 固定方案后一次性评估，不反复调参 |

这里要补一个时间信息。C-Eval 论文和早期评测流程强调 test 标签不公开、通过排行榜系统提交预测；官方 GitHub 在 **2025 年 7 月 27 日** 更新过“完整 test set 已向社区发布”的消息。因此今天再写工程流程时，最稳妥的说法不是“test 永远无法本地运行”，而是：

1. 历史论文和早期 leaderboard 采用隐藏测试集流程。
2. 现在复现实验时，需要明确你使用的是哪一版数据和哪一种评测协议。
3. 即使 test 数据可获得，也不应把它当作反复调 prompt 的开发集，否则比较会失真。

一个直观例子是：你先在 `val` 上测试了 12 个 prompt，选出最优模板；然后又在 `test` 上继续微调提示词，直到分数最高。此时 `test` 已经被你当成开发集使用，结果不再能代表真实泛化能力。

因此，C-Eval 的正确边界通常是：用 `val` 定模板、定温度、定输出解析规则；用固定好的方案跑最终结果；对外报告时写明数据版本、切分和 prompt 设定。

---

## 核心机制与推导

C-Eval 的工作机制可以拆成三层：题库设计、提示模板、结果聚合。

### 1. 题库设计：用统一框架覆盖多层级、多学科

C-Eval 不是单一学科 benchmark。它把 52 个学科放进同一套评测框架中，并覆盖四个难度层级。这种设计直接带来两个结果：

| 结果 | 含义 |
| --- | --- |
| 可以看总分 | 快速判断模型总体是否退化 |
| 可以看分学科结果 | 定位退化发生在哪个知识域 |

这点在工程上很重要。一个模型的总分不变，不等于能力没变。它可能是“法律下降 8%，物理上升 8%”，最后在总平均上互相抵消。如果你的业务是法务问答，这种总分“稳定”没有意义。

### 2. 提示模板：同一模型，模板不同，分数可能不同

C-Eval 常见的三种运行方式如下：

| 方式 | 模板特点 | 成本 | 输出稳定性 | 常见用途 |
| --- | --- | --- | --- | --- |
| zero-shot | 直接给题目和作答指令 | 最低 | 依赖模型指令跟随能力 | 快速基线 |
| answer-only five-shot | 先给 5 道示例题和标准答案 | 中 | 通常较稳定 | 默认离线评测 |
| chain-of-thought five-shot | 先给 5 道含推理过程的示例 | 最高 | 对复杂题可能更强，但更贵 | 深测推理能力 |

few-shot 的作用不是“教模型新知识”，而是给模型一个局部任务格式。它告诉模型三件事：

1. 这是哪一类考试题。
2. 你应该用什么格式输出。
3. 看到类似结构时，应优先沿用同一答题模式。

可以把它理解成条件约束。模型原本面对的是“任何中文文本都可能继续生成”；加上 few-shot 后，模型面对的是“这是标准化考试，输出应像前面的样例一样”。这会改变它的输出分布。

下面用一个玩具例子说明：

| 方式 | 答对题数 | 总题数 | 准确率 |
| --- | --- | --- | --- |
| zero-shot | 58 | 100 | 58% |
| answer-only five-shot | 64 | 100 | 64% |
| chain-of-thought five-shot | 67 | 100 | 67% |

这个表不说明 “CoT 永远最好”，它只说明：**模板本身就是变量**。因此比较两个模型前，必须先固定模板；否则你比较的是“模型 + 模板”的组合，而不是模型本身。

### 3. 结果聚合：从单题正确，汇总到学科和总体

对每一道题，评测器只关心最终选项是否与标准答案一致。假设第 $i$ 道题答对记为 1，答错记为 0，那么总体准确率可以写成：

$$
\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i=y_i)
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $N$ | 总题数 |
| $\hat{y}_i$ | 模型对第 $i$ 题预测的选项 |
| $y_i$ | 第 $i$ 题的标准答案 |
| $\mathbf{1}(\cdot)$ | 指示函数，条件成立记 1，否则记 0 |

如果还要看某个学科 $s$ 的表现，只需要把求和范围限制在该学科对应的题目集合：

$$
\text{Accuracy}_s=\frac{1}{|D_s|}\sum_{i\in D_s}\mathbf{1}(\hat{y}_i=y_i)
$$

其中 $D_s$ 表示学科 $s$ 的题目集合。

这一层的工程意义很直接：C-Eval 不是为了给出一个漂亮单分，而是为了让你形成“总分 + 学科分 + 难度分层”的观察面板。

一个真实感更强的例子是：你把底座模型从版本 A 切到版本 B。人工试用后，B 的对话更流畅，拒答也更自然；但 C-Eval 的 `法律职业资格`、`医学统计`、`高中物理` 三科同时下降。此时结论不能是“B 更会聊天，所以整体更强”，而应该是“B 在交互表现上可能更好，但知识密集型学科存在明显退化”。C-Eval 的任务，就是尽快把这个退化显式化。

---

## 代码实现

工程上最实用的写法不是“只算一个总准确率”，而是同时做到四件事：

1. 解析模型输出中的最终选项。
2. 统计总体准确率。
3. 统计分学科准确率。
4. 对比上一版本结果，决定是否触发回归门禁。

下面给出一个**标准库即可运行**的 Python 示例。它不依赖第三方库，复制后可直接执行。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re


CHOICE_RE = re.compile(r"\b([ABCD])\b")


@dataclass(frozen=True)
class Example:
    subject: str
    answer: str
    prediction_text: str


@dataclass(frozen=True)
class EvalSummary:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            raise ValueError("total must be > 0")
        return self.correct / self.total


def extract_choice(text: str) -> str:
    """
    从模型输出中提取最后出现的 A/B/C/D。
    规则足够简单，适合 answer-only 场景。
    """
    matches = CHOICE_RE.findall(text.upper())
    if not matches:
        raise ValueError(f"cannot extract choice from: {text!r}")
    return matches[-1]


def evaluate(examples: Iterable[Example]) -> tuple[EvalSummary, dict[str, EvalSummary]]:
    total = 0
    correct = 0
    per_subject_count: dict[str, int] = {}
    per_subject_correct: dict[str, int] = {}

    for ex in examples:
        pred = extract_choice(ex.prediction_text)
        is_correct = int(pred == ex.answer)

        total += 1
        correct += is_correct

        per_subject_count[ex.subject] = per_subject_count.get(ex.subject, 0) + 1
        per_subject_correct[ex.subject] = per_subject_correct.get(ex.subject, 0) + is_correct

    overall = EvalSummary(correct=correct, total=total)
    by_subject = {
        subject: EvalSummary(
            correct=per_subject_correct[subject],
            total=per_subject_count[subject],
        )
        for subject in per_subject_count
    }
    return overall, by_subject


def should_fail_gate(
    current: EvalSummary,
    previous: EvalSummary,
    max_drop: float = 0.02,
) -> bool:
    """
    如果当前版本相对上一版本下降超过阈值，则触发失败。
    max_drop=0.02 表示最多允许下降 2 个百分点。
    """
    return (previous.accuracy - current.accuracy) > max_drop


def should_fail_subject_gate(
    current_by_subject: dict[str, EvalSummary],
    previous_by_subject: dict[str, EvalSummary],
    max_drop: float = 0.03,
    watched_subjects: set[str] | None = None,
) -> bool:
    subjects = watched_subjects or set(current_by_subject) & set(previous_by_subject)
    for subject in subjects:
        current = current_by_subject[subject].accuracy
        previous = previous_by_subject[subject].accuracy
        if previous - current > max_drop:
            return True
    return False


if __name__ == "__main__":
    prev_examples = [
        Example("法律", "B", "答案：B"),
        Example("法律", "C", "我认为答案是 C"),
        Example("医学", "A", "A"),
        Example("医学", "D", "答案：D"),
        Example("计算机", "B", "最终答案 B"),
        Example("计算机", "A", "A"),
    ]

    curr_examples = [
        Example("法律", "B", "答案：B"),
        Example("法律", "C", "答案：D"),
        Example("医学", "A", "A"),
        Example("医学", "D", "答案：C"),
        Example("计算机", "B", "最终答案 B"),
        Example("计算机", "A", "A"),
    ]

    prev_overall, prev_subject = evaluate(prev_examples)
    curr_overall, curr_subject = evaluate(curr_examples)

    print(f"prev accuracy: {prev_overall.accuracy:.2%}")
    print(f"curr accuracy: {curr_overall.accuracy:.2%}")
    print("fail overall gate:", should_fail_gate(curr_overall, prev_overall, max_drop=0.10))
    print(
        "fail subject gate:",
        should_fail_subject_gate(curr_subject, prev_subject, max_drop=0.40),
    )

    for subject, summary in sorted(curr_subject.items()):
        print(f"{subject}: {summary.correct}/{summary.total} = {summary.accuracy:.2%}")
```

这段代码覆盖了最核心的离线评测逻辑。

第一，`extract_choice()` 负责从模型输出中提取 A/B/C/D。对于 answer-only 模板，这通常已经够用。  
第二，`evaluate()` 同时统计总体和分学科准确率。  
第三，`should_fail_gate()` 和 `should_fail_subject_gate()` 可以直接接到 CI 流水线中。

运行这段代码，你会得到两个层面的结果：

| 输出 | 含义 |
| --- | --- |
| `prev accuracy` / `curr accuracy` | 当前版本相对上一版本是否退化 |
| 每个学科的正确率 | 是否有关键领域单独退化 |

如果你要把它接到真实数据上，最常见的输入记录至少要包含这些字段：

| 字段 | 说明 |
| --- | --- |
| `subject` | 学科名或学科 ID |
| `question` | 题干 |
| `A/B/C/D` | 四个选项 |
| `answer` | 标准答案 |
| `prediction_text` | 模型原始输出 |

下面再给一个更接近真实 C-Eval few-shot 的提示模板。这个模板本身不是“评分代码”，但它决定了模型输出是否稳定，因此在工程中必须版本化管理。

```text
以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。

{demo_question_1}
A. {demo_A_1}
B. {demo_B_1}
C. {demo_C_1}
D. {demo_D_1}
答案：{demo_answer_1}

{demo_question_2}
A. {demo_A_2}
B. {demo_B_2}
C. {demo_C_2}
D. {demo_D_2}
答案：{demo_answer_2}

... 省略其余 few-shot 示例 ...

现在回答下面这道题：

{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案：
```

如果是 chain-of-thought five-shot，few-shot 部分会包含解析过程，但正式题目的输出是否允许模型显式展开推理，需要你在实验设定里写清楚。否则会出现两类常见问题：

| 问题 | 后果 |
| --- | --- |
| 输出太长 | 成本和延迟明显上升 |
| 最终答案位置不稳定 | 解析脚本更容易抽取失败 |

再给一个常见的配置示意。它不绑定某个特定框架，但字段足够接近现有评测工具：

```json
{
  "dataset_name": "ceval/ceval-exam",
  "split": "val",
  "prompt_template": "answer_only",
  "num_fewshot": 5,
  "temperature": 0.0,
  "metric": "accuracy",
  "save_predictions": true
}
```

这里几个字段需要固定：

| 字段 | 为什么要固定 |
| --- | --- |
| `split` | 不同切分用途不同，不能混用 |
| `prompt_template` | 模板变化会直接影响分数 |
| `num_fewshot` | 0-shot 和 5-shot 不能直接混比 |
| `temperature` | 随机性会影响结果稳定性 |
| `save_predictions` | 方便复盘错误样本 |

一个完整的工程流程通常是：

1. 训练或微调完成后，导出候选模型。
2. 在固定模板、固定温度、固定切分下跑 C-Eval `val`。
3. 输出总体准确率、分学科准确率、与上一个线上版本的差值。
4. 如果总分下降超过阈值，或关键学科下降超过阈值，则 CI 失败。
5. 保存预测结果，供错误分析使用。

这样做的价值不是“追排行榜”，而是把“模型是否变差”变成一个可执行、可追踪、可回归的工程规则。

---

## 工程权衡与常见坑

C-Eval 非常适合做中文知识与推理回归，但它不是万能指标。工程上至少有六类常见坑需要提前规避。

| 坑 | 现象 | 风险 | 规避策略 |
| --- | --- | --- | --- |
| 混用 `val` 与 `test` | 在同一份结果上反复调模板 | 结果失真，不可复现 | `val` 调参，`test` 只做最终评估 |
| 只看总分 | 总分变化不大就放行 | 关键学科可能大幅退化 | 同时看总分、学科分、难度层分数 |
| 模板未固定 | 不同实验用不同 prompt | 分数不可比 | 把模板和 few-shot 示例一并版本化 |
| 输出解析不稳定 | 模型输出“我认为是 B，但也可能是 C” | 抽取答案出错，污染分数 | 限制输出格式，保存原始预测 |
| few-shot 过长 | 输入上下文过大 | 成本上升、延迟增加、甚至超窗口 | 日常回归优先 answer-only five-shot |
| 误解单次波动 | 0.2% 变化就得出强结论 | 误判模型优劣 | 固定温度、固定随机种子、重复测量 |

### 1. 不要把“高分”直接等价为“产品更好”

如果你的产品是客服对话、代码助手、文案生成器，那么 C-Eval 只能回答其中一部分问题。它能告诉你模型在知识密集型中文题上是否退化，但不能替代对话体验、工具调用、执行成功率等指标。

一个常见误判是：

- 模型 A 的 C-Eval 比模型 B 高 3 分，于是断言 A 全面更强。

这个结论不成立。更准确的说法是：

- 模型 A 在 C-Eval 所覆盖的中文知识与推理任务上表现更强；
- 是否更适合你的产品，还要看你的业务指标。

### 2. token 成本是实实在在的工程变量

few-shot 尤其是 CoT few-shot，会显著拉长输入长度。设一题原始题面是 300 tokens，5 个示例每个 180 tokens，那么单题输入可能接近：

$$
300 + 5 \times 180 = 1200 \text{ tokens}
$$

如果 CoT 示例再带解析，长度还会继续增大。题量一上来，成本会很快膨胀。对在线评测平台或日常回归任务来说，这不是细节，而是预算问题。

常见折中方式如下：

| 场景 | 推荐模板 |
| --- | --- |
| 每日回归 | answer-only five-shot 或 zero-shot |
| 发版前候选评估 | answer-only five-shot + 关键集加测 |
| 深入分析复杂推理 | 额外加跑 CoT five-shot |

### 3. 数据泄漏必须作为风险写进结论

公开 benchmark 都面临一个现实问题：模型可能在预训练或后训练语料中见过题目、答案或高度相似的材料。C-Eval 也不例外。

这意味着高分的解释必须克制。更稳妥的表述是：

- 该模型在 C-Eval 上表现更好；
- 这说明它对该基准所覆盖分布的适应性更强；
- 但不能把这个结果无限外推为“真实世界所有中文知识任务都更强”。

### 4. 学科样本量不一致，读分要看口径

有的学科题更多，有的学科题更少。如果只看总体准确率，大题量学科会占更大权重；如果看学科平均，不同学科又会被等权处理。两种口径都合理，但含义不同。

因此，实验报告至少应写清：

| 报告项 | 是否必须写明 |
| --- | --- |
| 数据版本 | 是 |
| 使用的切分 | 是 |
| prompt 模板 | 是 |
| few-shot 数量 | 是 |
| 温度 / 解码设置 | 是 |
| 聚合口径 | 是 |

没有这些信息，分数本身的解释价值会明显下降。

---

## 替代方案与适用边界

如果你的目标是评估“中文多学科知识与推理”，C-Eval 很合适；如果目标换了，工具也应该换。

| 基准 | 覆盖重点 | 输出形式 | 常见指标 | 更适合什么 |
| --- | --- | --- | --- | --- |
| C-Eval | 中文多学科知识与推理 | 选择题 | Accuracy | 中文综合能力回归 |
| GSM8K | 数学文字题、多步算术推理 | 生成式答案 | Exact Match 等 | 算术链式推理 |
| MMLU | 多学科通识知识，英文生态更成熟 | 选择题 | Accuracy | 国际通用知识基线 |

三者的差别可以直接概括为：

| 基准 | 优势 | 局限 |
| --- | --- | --- |
| C-Eval | 中文场景直接、覆盖广、适合回归 | 不能代表开放式生成和工具使用 |
| GSM8K | 对多步数学推理敏感 | 任务面窄，不代表综合知识 |
| MMLU | 国际上使用广，便于横向对照 | 中文适配性通常不如 C-Eval 直接 |

所以选择逻辑很简单：

1. 你要看中文知识面是否退化，优先 C-Eval。
2. 你要看多步数学推理是否增强，补充 GSM8K。
3. 你要做跨语言或对标国际常用基准，再加入 MMLU。

一个具体例子是：某模型在 C-Eval 上从 62% 提升到 68%，但在 GSM8K 上几乎不动。最合理的解释不是“模型全面变强”，而是“它在中文题面理解、中文知识召回或中文考试型作答上更强了，但多步数学推理未必同步提升”。

同理，如果你的业务是“中文教育问答”“法律检索问答”“学科分类”，C-Eval 的工程价值很高；如果你的业务是“自动写代码”“多轮 agent 工具执行”“长文摘要”，C-Eval 只能作为补充，不应当作主指标。

---

## 参考资料

C-Eval 的参考资料应分成四类看：官方仓库、官方站点、论文、工程接入工具。它们解决的问题不同，不应混为一谈。

| 资料类型 | 用途 | 建议优先看什么 |
| --- | --- | --- |
| 官方 GitHub 仓库 | 确认数据结构、prompt、切分与示例 | README、`subject_mapping.json`、示例数据格式 |
| 官方网站 | 看榜单、任务说明、提交或版本信息 | leaderboard、任务介绍、公告 |
| 论文 | 看设计目标、构建方法和实验设置 | 数据来源、难度分层、基线结果 |
| 评测框架文档 | 看如何接入自动化评测 | task 名、数据适配、批量评测方式 |

建议优先阅读以下资料：

| 名称 | 链接 | 你能得到什么 |
| --- | --- | --- |
| C-Eval 官方仓库 | https://github.com/hkust-nlp/ceval | 52 学科、13,948 题、prompt 模板、数据使用方式 |
| C-Eval 官方网站 | https://cevalbenchmark.com/ | leaderboard、任务说明、更新信息 |
| C-Eval 论文 | https://arxiv.org/abs/2305.08322 | 设计动机、构建方法、基线实验 |
| LM Evaluation Harness | https://github.com/EleutherAI/lm-evaluation-harness | 自动化评测接入方式 |

结合官方资料，可以把工程上的最小实践总结成下面两条。

第一，日常开发时，用 `val` 做回归，固定 prompt 和 few-shot 数量，保存预测结果。  
第二，对外报告时，写清你使用的数据版本、切分、模板和聚合口径；不要只给一个孤立分数。

最后给一个比原文更完整的配置示意。它仍然是“示意”，不是绑定某一个框架的唯一格式，但已经覆盖了离线评测所需的关键信息。

```json
{
  "dataset_name": "ceval/ceval-exam",
  "split": "val",
  "prompt_template": "answer_only",
  "num_fewshot": 5,
  "temperature": 0.0,
  "max_new_tokens": 8,
  "save_predictions": true,
  "save_raw_generations": true,
  "metric": "accuracy",
  "report_by_subject": true
}
```

如果你要保留一份正式评测记录，建议至少连同下面这些内容一起存档：

| 存档项 | 原因 |
| --- | --- |
| 模型版本 | 便于回溯 |
| 数据版本 | 避免不同版本混比 |
| prompt 模板全文 | 模板变化会影响结果 |
| few-shot 示例内容 | few-shot 本身也是实验条件 |
| 原始预测 | 便于错误分析 |
| 汇总脚本版本 | 避免统计口径变化 |

这样做的意义不只是“让实验更规范”，而是让 C-Eval 真正成为一个可复现、可比较、可进入 CI 的工程基准，而不是一次性截图分数。

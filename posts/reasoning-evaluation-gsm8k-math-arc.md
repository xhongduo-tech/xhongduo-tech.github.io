## 核心结论

GSM8K、MATH、ARC 不是对同一件事做三次重复测量，而是在测三种不同结构的推理任务。

GSM8K 的重点是多步小学数学文字题。它考的是“把自然语言条件拆成若干算术步骤，并得到最终数值答案”的能力。这里的“多步”可以白话理解为：不是一步心算，而是要先列出中间量，再继续算下去。

MATH 的重点是竞赛级数学推导。它考的是更长链路的符号操作、分情况讨论、公式变形和启发式搜索。这里的“启发式”可以白话理解为：不是机械套公式，而是要先判断往哪个方向推更可能成功。

ARC 的重点是科学与常识场景下的多项选择推理。它考的是读题、找证据、排除干扰项，而不只是算数。这里的“干扰项”可以白话理解为：看起来像对，但和题干条件并不完全匹配的错误答案。

把三者放在一起看，才能区分一个模型到底是：
1. 只会短链路算术和常见模板；
2. 能做长链路数学推导；
3. 还是也能处理跨句证据整合与常识/科学选择题。

下面这张表先给出总览。

| 基准 | 任务类型 | 输入形式 | 输出形式 | 主要考察能力 | 常见评分方式 |
| --- | --- | --- | --- | --- | --- |
| GSM8K | 多步算术文字题 | 自然语言题干 | 最终数值/短文本答案 | 条件拆解、短链路计算、基础代数 | 最终答案 exact match |
| MATH | 竞赛级数学题 | 自然语言 + 数学表达式 | 归一化后的最终答案 | 长链路推导、符号变换、问题分解 | 最终答案归一化后 exact match |
| ARC | 科学/常识多选题 | 题干 + 选项 | 选项标签或选项内容 | 常识推理、证据整合、排除干扰项 | 多项选择 accuracy |

一个直接的诊断例子是：某模型在 GSM8K 很高，但在 MATH 和 ARC 都低。这通常说明它擅长基础算术和套路化步骤，不足以说明它具备更一般的推理能力。

---

## 问题定义与边界

讨论分数之前，必须先定义评测对象。否则最容易犯的错，就是把不同题型下的 accuracy 当成同一种能力刻度。

设第 $i$ 道题为 $x_i$，标准答案为 $y_i$，模型输出为 $\hat y_i$。注意，这里的“答案”在三个数据集里并不一样：

| 符号 | 含义 | GSM8K | MATH | ARC |
| --- | --- | --- | --- | --- |
| $x_i$ | 第 $i$ 道题目 | 文字题 | 数学题 | 选择题题干与选项 |
| $y_i$ | 标准答案 | 数值文本 | 归一化后的数学答案 | 正确选项标签 |
| $\hat y_i$ | 模型输出 | 自由文本 | 自由文本 | 自由文本或选项 |
| $\mathrm{norm}(\cdot)$ | 归一化函数 | 提取最终数值 | 提取最终答案并标准化格式 | 映射到 A/B/C/D |
| 判定规则 | 如何算对 | 数值匹配 | 归一化后匹配 | 选项命中 |

这里的“归一化”可以白话理解为：把不同写法但语义相同的答案，变成统一格式再比较。

例如，同样是“答对 2 题”：
- 在 GSM8K，可能是两个最终数值和标准答案完全一致；
- 在 MATH，可能是两个 `\boxed{...}` 里的最终答案经标准化后相等；
- 在 ARC，可能只是两个选项字母命中。

表面上都叫 accuracy，但判定对象完全不同，所以不能把 80% 的 GSM8K 和 80% 的 ARC 当成同一尺度上的“80 分”。

边界也要说清楚。GSM8K、MATH、ARC 都主要评测“有标准答案的离线推理任务”。它们不直接评测：
- 长文检索后的开放问答；
- 工具调用能力；
- 多轮交互中的自我修正；
- 真实产品里的延迟、成本、稳定性。

因此，它们更适合做“分层诊断”，不适合单独代表“通用智能”。

---

## 核心机制与推导

三者虽然题型不同，但评分可以统一写成一个公式：

$$
\mathrm{Acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\big[\mathrm{norm}(\hat y_i)=\mathrm{norm}(y_i)\big]
$$

其中 $\mathbf{1}[\cdot]$ 是指示函数。白话说就是：条件成立记 1 分，不成立记 0 分，最后取平均。

关键不在 `accuracy` 这个词，而在两个前置步骤：

1. 先把模型输出变成“可判定答案”；
2. 再按数据集规则比较。

三个基准的核心差异，可以压缩成下面这张对比表。

| 基准 | 先做什么 | 再做什么 | 典型难点 |
| --- | --- | --- | --- |
| GSM8K | 从长文本里提取最终数值 | 与标准数值比较 | 模型会输出过程但最后答案格式不稳定 |
| MATH | 提取最终答案并做符号/格式标准化 | 与标准答案比较 | `\boxed{}`、分数、小数、等价表达式处理复杂 |
| ARC | 把输出映射到选项标签 | 与正确选项比较 | 模型可能复述选项文本而不是直接给字母 |

玩具例子很简单。假设有三道题，标准答案分别是 `5`、`12`、`B`。模型输出分别是 `The answer is 5.`、`10`、`Option B`。如果归一化后得到 `5`、`10`、`B`，那么三题中答对两题：

$$
\mathrm{Acc} = \frac{2}{3} = 66.7\%
$$

这说明“先抽取答案再评分”才是核心机制。如果直接拿全文字符串比较，第一题和第三题都会被误判。

还可以把文中常见写法统一起来：

```text
通用准确率:
Acc = (1/N) Σ_i 1[norm(\hat y_i)=norm(y_i)]

GSM8K:
c* = argmax_j v(c_j)

MATH:
final answer exact match after normalization

ARC:
multiple-choice accuracy on option labels
```

其中 $c^*=\arg\max_j v(c_j)$ 出现在带 verifier 的 GSM8K 设置里。这里的“verifier”可以白话理解为：不是直接相信第一份解答，而是让一个额外评分器在多个候选答案里选最可信的那个。它说明 GSM8K 评测里，生成和判别有时会被拆开。

真实工程例子更能看出三者为什么要一起看。假设你在评测一个通用助手模型：
- GSM8K 高，说明短链路算术和文字题拆解稳定；
- MATH 低，说明复杂推导、符号变换、长链路规划容易失稳；
- ARC 低，说明常识与科学题里的证据整合和排除干扰项不可靠。

这时结论不是“模型会推理”，而是“模型在基础算术模板上表现不错，但一般推理能力仍不完整”。

---

## 代码实现

实现时最重要的原则是：把“答案抽取/归一化”和“评分”拆开。不要把原始输出直接拿去比较。

下面是一段最小可运行的 Python 示例，分别演示 GSM8K/MATH 风格的最终答案抽取，以及 ARC 风格的选项映射。

```python
import re

def normalize_numeric(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        text = m.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        value = nums[-1]
        if "." in value:
            value = str(float(value)).rstrip("0").rstrip(".")
        return value.lstrip("0") or "0"
    return text.lower().strip()

def normalize_choice(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    for ch in ["A", "B", "C", "D"]:
        if f"OPTION {ch}" in text or f"ANSWER: {ch}" in text:
            return ch
    return text

def score_exact(pred: str, gold: str, task: str) -> int:
    if task in {"gsm8k", "math"}:
        return int(normalize_numeric(pred) == normalize_numeric(gold))
    if task == "arc":
        return int(normalize_choice(pred) == normalize_choice(gold))
    raise ValueError("unknown task")

def eval_dataset(samples):
    correct = 0
    for s in samples:
        correct += score_exact(s["model_output"], s["label"], s["task"])
    return correct / len(samples)

samples = [
    {"task": "gsm8k", "model_output": "The answer is 5.", "label": "5"},
    {"task": "math", "model_output": "\\boxed{05}", "label": "5"},
    {"task": "arc", "model_output": "Option B", "label": "B"},
    {"task": "gsm8k", "model_output": "4", "label": "5"},
]

acc = eval_dataset(samples)
assert score_exact("The answer is 5.", "5", "gsm8k") == 1
assert score_exact("\\boxed{5.0}", "5", "math") == 1
assert score_exact("Option B", "B", "arc") == 1
assert score_exact("4", "5", "gsm8k") == 0
assert abs(acc - 0.75) < 1e-9
print(acc)
```

这段代码故意做得很小，但已经体现了评测实现的两个必要步骤：
1. 先提取最终答案；
2. 再按任务类型比较。

真正工程里还会更复杂。例如 MATH 里你可能需要处理分数、根式、负号位置、等价表达式；ARC 里你可能要从“完整句子答案”反推出选项编号，而不是只匹配单个字母。

---

## 工程权衡与常见坑

最常见的错误，不是模型本身，而是评测脚本写错。

| 坑点 | 错误做法 | 正确做法 | 影响 |
| --- | --- | --- | --- |
| 直接横比三者分数 | 把三个 accuracy 当同一标尺 | 先在各自任务内解读，再看差异模式 | 会误判模型能力结构 |
| 不做答案归一化 | 直接全文字符串比较 | 先抽取最终答案，再标准化格式 | 大量“本来答对”被算错 |
| 只看总分 | 只报一个平均值 | 分开报 GSM8K、MATH、ARC，必要时分层 | 无法定位具体短板 |
| 过度依赖 CoT 文本 | 觉得推理链像样就算强 | 最终仍以标准答案计分 | 容易把“会写解释”当“会解题” |
| 忽略数据污染 | 看到高分就下结论 | 检查训练泄漏、提示复用和近似题记忆 | 高分可能不代表泛化 |

一个典型误判例子是：模型输出 `\boxed{5}`、`5.0`、`05`、`5 `。如果你的脚本不做标准化，这四种写法可能被当成四个不同答案；但从任务语义上看，它们通常都应该对应同一个最终值。

另一个常见坑是 ARC。很多实现只接受单字母 `A/B/C/D`，但真实模型经常输出“我选择 B，因为……”。如果不先做选项映射，准确率会被平白拉低。

工程上还要权衡“规则是否过强”。规则写得太松，可能把本不等价的答案误判成正确；规则写得太严，又会把合理等价答案判错。这个边界在 MATH 尤其敏感，因为数学表达式的等价性远比字符串相等复杂。

---

## 替代方案与适用边界

如果目标是“评测一个基础推理切面”，GSM8K、MATH、ARC 很有用；如果目标是“提升真实系统效果”，它们还不够。

| 场景 | 更适合的基准或方法 | 原因 |
| --- | --- | --- |
| 看基础算术与短链路拆解 | GSM8K | 题型集中，误差来源清晰 |
| 看长链路数学推导 | MATH | 难度更高，能拉开复杂推导能力差异 |
| 看科学常识与干扰项排除 | ARC | 多选结构更接近证据整合 |
| 看生成阶段是否可改进 | verifier rerank / pass@k / self-consistency | 可能“会做但第一次没说对” |
| 看真实助手能力 | 工具调用、检索问答、长上下文任务 | 离线标准题无法覆盖产品行为 |

这里的 `pass@k` 可以白话理解为：一次生成 $k$ 个候选答案，只要其中一个正确就算通过。它适合分析“模型有没有潜在解题能力，但首答不稳定”。

真实工程里，一个很常见的现象是：模型在 MATH 上原始分数不高，但引入 verifier 重新排序后显著提升。这通常说明问题不完全在“不会推”，而在“第一次采样出的最终答案不稳定”。相反，如果 verifier 和多样本投票都救不回来，才更可能是底层推理能力本身不足。

因此，GSM8K、MATH、ARC 最好的用法不是争论“谁更代表智能”，而是把它们当作三把不同的尺子：
- GSM8K 看短链路算术；
- MATH 看长链路数学；
- ARC 看知识型选择推理。

三者联合，能帮助你判断模型是在某一类模板上强，还是具备更一般的推理迁移能力；但三者本身仍然不能覆盖工具使用、开放环境决策、多轮交互纠错等更接近真实系统的问题。

---

## 参考资料

1. [OpenAI: Solving math word problems](https://openai.com/research/solving-math-word-problems)
2. [Cobbe et al., 2021: Training Verifiers to Solve Math Word Problems](https://huggingface.co/papers/2110.14168)
3. [Hendrycks et al., 2021: Measuring Mathematical Problem Solving With the MATH Dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf)
4. [Clark et al., 2018: ARC, the AI2 Reasoning Challenge](https://a11y2.apps.allenai.org/paper?id=88bb0a28bb58d847183ec505dda89b63771bb495)
5. [Ai2 Open Data: ARC dataset](https://allenai.org/open-data)
6. [MATH official repository](https://github.com/hendrycks/math)

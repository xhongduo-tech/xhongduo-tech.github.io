## 核心结论

MMLU 的目标不是测某一个任务做得有多极致，而是测一个语言模型在很多学科里是否都具备“够用的知识面”和“基础推理能力”。这里的“广覆盖”可以直白理解为：不是只看数学、代码或法律，而是把 57 个学科放在同一张成绩单里一起看。

它的标准输入形式是多项选择题，常见做法是 few-shot 提示。few-shot 的白话解释是：先给模型看少量带答案的示例，再让它做新题。标准输出通常只允许模型回答一个选项，如 `A/B/C/D`。标准口径是准确率，也就是答对题数占总题数的比例：

$$
\mathrm{Accuracy}=\frac{N_{\text{correct}}}{N_{\text{total}}}\times 100\%
$$

玩具例子可以这样理解：把 SAT、MCAT、法学考试、医学考试、计算机课程题混在一起，先给模型 5 道示例题，再给一题新题，要求只选 `A-D`。如果总共 100 题答对 72 题，MMLU 准确率就是 $72\%$。

MMLU 的价值在于覆盖面广、实现简单、容易复现，适合做大模型版本回归测试。但它也有明确局限：多选题格式会把开放式能力压缩成“选项判断”，可能高估真实能力；题库公开，存在训练集污染风险；提示模板、示例顺序、上下文长度都可能明显影响结果。

---

## 问题定义与边界

MMLU 评测的问题可以表述为：在不做任务微调的条件下，一个语言模型在 57 个开放学科上的单轮多选问答表现，是否足以反映它的跨领域知识泛化能力。这里“泛化”可以直白理解为：不是只会做训练过的某一类题，而是在很多不同题型和学科里都能保持相对稳定的表现。

它的边界也很清楚。

| 维度 | MMLU 的标准边界 |
|---|---|
| 输入形式 | zero-shot 或 few-shot prompt |
| 输出形式 | 单个选项，通常是 `A/B/C/D` |
| 训练依赖 | 原则上不针对该任务单独微调 |
| 交互方式 | 单轮，不测多轮追问 |
| 评估粒度 | 整体准确率 + 学科级准确率 |
| 主要能力 | 广覆盖知识与基础推理 |
| 不擅长反映 | 长链推理、多轮规划、工具使用 |

新手版理解：像是把 57 本教材里每门课都抽一些四选一题，不让模型边答边学，也不允许它追问，只看它第一次回答能否选中最合理的那个选项。

这里要特别区分“能做选择题”和“真正理解问题”。MMLU 更接近前者。因为模型只要在给定选项中识别相对最优答案，就可能拿到高分；但真实工程里，很多任务并没有现成选项，也不能靠排除法解决。这意味着：MMLU 是一个有用的能力切片，不是完整能力画像。

---

## 核心机制与推导

MMLU 的基本统计方式并不复杂。最直接的是整体准确率，也就是把全部题目放在一起，统计总命中率。另一个常见口径是学科宏平均。这里“宏平均”可以直白理解为：先算每个学科自己的准确率，再让每个学科等权平均，避免题量多的学科把总分“拉着走”。

设第 $i$ 个学科有 $n_i$ 道题，答对 $c_i$ 道，则该学科准确率为：

$$
\mathrm{Accuracy}_i=\frac{c_i}{n_i}
$$

57 个学科的宏平均为：

$$
\mathrm{MacroAcc}=\frac{1}{57}\sum_{i=1}^{57}\mathrm{Accuracy}_i
$$

玩具例子：某模型在“计算机安全”5 道题中答对 4 道，那么该学科准确率就是 $4/5=80\%$。如果另一个学科 10 道题只答对 5 道，则该学科准确率为 $50\%$。宏平均时，这两个学科各占一半权重；整体准确率时，第二个学科因为题更多，会占更大权重。

这两种口径都合理，但回答的问题不同：

| 口径 | 强调什么 | 风险 |
|---|---|---|
| Overall Accuracy | 全体题目的总命中率 | 题量大的学科影响更大 |
| Macro Accuracy | 各学科是否均衡 | 可能弱化热门学科的重要性 |

few-shot 的机制也值得单独说明。模型不是直接看题，而是先看若干个“题目 + 正确答案”的示例，再看待测题。理论上，这会帮助模型理解格式与作答方式；工程上，它也会引入额外变量。示例顺序不同、题目截断不同、选项排版不同，都可能让分数变化几个百分点，严重时甚至更多。

真实工程例子：团队在发布新版本模型前，把 MMLU 作为回归门槛。版本 A 的 overall accuracy 是 71%，版本 B 是 72%。如果你没有固定 prompt 模板，这 1 个百分点可能只是模板变化，不一定代表模型真的变强了。只有把示例数、示例顺序、系统提示、温度、最大输出长度都固定下来，版本对比才有意义。

---

## 代码实现

下面给一个最小但可运行的 Python 示例。它不依赖真实模型 API，而是用一个假的预测函数演示 MMLU 的计算流程：构造 few-shot prompt、遍历学科、记录 overall accuracy 和 macro accuracy，并用 `assert` 验证结果。

```python
from collections import defaultdict

# 一个极小的玩具数据集，结构模仿 MMLU
dataset = [
    {
        "subject": "computer_security",
        "question": "Which protocol is mainly used for secure web browsing?",
        "choices": ["FTP", "HTTP", "HTTPS", "SMTP"],
        "answer": "C",
    },
    {
        "subject": "computer_security",
        "question": "What does SQL injection target?",
        "choices": ["CPU cache", "Database queries", "DNS records", "Image compression"],
        "answer": "B",
    },
    {
        "subject": "high_school_biology",
        "question": "Which organelle is the main site of ATP production?",
        "choices": ["Ribosome", "Mitochondrion", "Golgi apparatus", "Lysosome"],
        "answer": "B",
    },
    {
        "subject": "high_school_biology",
        "question": "DNA stands for?",
        "choices": [
            "Deoxyribonucleic acid",
            "Dynamic ribonuclear acid",
            "Double nitrogen acid",
            "Digital nucleic array",
        ],
        "answer": "A",
    },
]

few_shot_examples = [
    {
        "question": "What is 2 + 2?",
        "choices": ["1", "2", "4", "8"],
        "answer": "C",
    },
    {
        "question": "Which letter comes first in the English alphabet?",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
    },
]

def build_prompt(examples, item):
    prompt_parts = []
    for ex in examples:
        prompt_parts.append(
            f"Question: {ex['question']}\n"
            f"A. {ex['choices'][0]}\n"
            f"B. {ex['choices'][1]}\n"
            f"C. {ex['choices'][2]}\n"
            f"D. {ex['choices'][3]}\n"
            f"Answer: {ex['answer']}\n"
        )
    prompt_parts.append(
        f"Question: {item['question']}\n"
        f"A. {item['choices'][0]}\n"
        f"B. {item['choices'][1]}\n"
        f"C. {item['choices'][2]}\n"
        f"D. {item['choices'][3]}\n"
        f"Answer:"
    )
    return "\n".join(prompt_parts)

def mock_model_call(prompt, item):
    # 演示用“模型”：固定返回正确答案，最后一题故意答错
    if "DNA stands for?" in prompt:
        return "B"
    return item["answer"]

subject_total = defaultdict(int)
subject_correct = defaultdict(int)
overall_total = 0
overall_correct = 0

for item in dataset:
    prompt = build_prompt(few_shot_examples, item)
    pred = mock_model_call(prompt, item)

    overall_total += 1
    subject_total[item["subject"]] += 1

    if pred == item["answer"]:
        overall_correct += 1
        subject_correct[item["subject"]] += 1

overall_accuracy = overall_correct / overall_total
subject_accuracy = {
    subject: subject_correct[subject] / subject_total[subject]
    for subject in subject_total
}
macro_accuracy = sum(subject_accuracy.values()) / len(subject_accuracy)

print("overall_accuracy =", overall_accuracy)
print("subject_accuracy =", subject_accuracy)
print("macro_accuracy =", macro_accuracy)

assert overall_total == 4
assert overall_correct == 3
assert abs(overall_accuracy - 0.75) < 1e-9
assert abs(subject_accuracy["computer_security"] - 1.0) < 1e-9
assert abs(subject_accuracy["high_school_biology"] - 0.5) < 1e-9
assert abs(macro_accuracy - 0.75) < 1e-9
```

如果换成真实工程实现，核心流程仍然是这四步：

| 步骤 | 工程动作 |
|---|---|
| 1 | 按学科加载题库，例如 CSV 或 JSON |
| 2 | 用固定模板构造 zero-shot 或 5-shot prompt |
| 3 | 逐题调用模型，只保留 `A/B/C/D` |
| 4 | 汇总 overall、macro、各学科差值 |

真实工程例子：你在模型发布流水线里加入一个 nightly job，每晚对固定版本的 MMLU-Cleaned 题集跑一次，生成 `overall_accuracy`、`macro_accuracy`、`top_drop_subjects` 三类指标。如果“法律”学科比昨天跌了 8 个百分点，而其他学科稳定，就要优先排查最近是否改了法律相关数据混合比例或提示模板。

---

## 工程权衡与常见坑

MMLU 的第一个大坑是 benchmark contamination，也就是基准污染。白话说，就是模型训练时可能已经见过题库或高相似版本，导致分数看起来很高，但那不一定代表真实泛化能力。对公开题库来说，这是长期存在的问题。

第二个大坑是题目质量。公开资料里反复提到，一些题存在错标、多义、歧义或答案不唯一的问题。这样的问题一多，评测结果就会混入“题库噪声”，你看到的分数变化未必全是模型变化。

第三个大坑是 prompt drift，也就是提示漂移。白话说，同样一套题，因为示例顺序、系统指令、分隔符、是否要求“只输出字母”这些细节变化，分数可能明显波动。很多团队以为自己在比较模型，实际上比较的是模板。

| Pitfall | Description | Mitigation |
|---|---|---|
| 题库污染 | 训练集见过题目或高度相似文本 | 使用 Redux/Cleaned 版本，补充自建 hold-out 集 |
| 题目噪声 | 错题、歧义题、多解题影响统计 | 记录数据版本，优先采用清洗版 |
| Prompt drift | 模板、顺序、格式变化导致波动 | 固定模板、示例顺序、解码参数 |
| 输出不规范 | 模型输出解释文本而非单字母 | 增加严格解析器，只接受 `A-D` |
| 只看总分 | 某些学科退化被 overall 掩盖 | 同时看宏平均和学科分布 |
| 误读高分 | 把选择题高分当作通用智能高分 | 联合开放问答、长推理、多轮任务一起看 |

一个常见误区是“模型 MMLU 高，所以它就更适合线上客服、代码代理或医学问答”。这个推理并不成立。因为 MMLU 更偏向静态知识检索和基础判断，而真实产品往往需要多轮上下文维护、拒答策略、工具调用、格式稳定输出，这些都不是 MMLU 的强项。

---

## 替代方案与适用边界

如果你的目标是“尽量干净地比较模型在广覆盖知识上的能力”，可以优先看 MMLU 的清洗变体，如 MMLU-Redux、MMLU-CF。它们的核心价值不是改评分公式，而是尽量减少错题、歧义和污染带来的偏差。

如果你的目标是特定领域能力，而不是 57 科平均水平，那么更合适的做法是使用领域 benchmark。比如医学模型更适合看 USMLE 类评测，法律模型更适合法律问答或法条检索基准，代码模型则需要结合 HumanEval、MBPP 或仓库级任务。

| 基准 | 焦点 | 污染控制 | 适用场景 |
|---|---|---|---|
| MMLU | 57 科广覆盖知识与基础推理 | 一般 | 大模型通用回归、横向比较 |
| MMLU-Redux / Cleaned | 广覆盖，但更重视题目质量 | 更强 | 需要更稳定可比的离线评测 |
| 医学/法律/代码专用 benchmark | 单领域深度能力 | 视数据集而定 | 产品明确服务某专业用户 |
| 多轮代理型 benchmark | 长流程决策、工具调用 | 视任务设计而定 | 测 agent 或 workflow 系统 |

玩具例子：如果你只做医学问答助手，MMLU 里历史、物理、会计这些学科的高低，对产品价值并不直接。你仍然可以保留一次 MMLU 作为“知识面下限检查”，但主指标应该换成医学领域基准。

真实工程例子：一家做企业法务助手的团队，发现模型在 MMLU 上提升了 3 个百分点，但法律场景用户满意度没有改善。排查后发现，新版本主要提升的是理工类学科，而法条检索稳定性和引用准确率没有变化。这说明 broad benchmark 可以作为背景指标，但不能替代领域指标。

---

## 参考资料

- AI Wiki: MMLU 条目，适合用来确认定义、题目结构、57 学科范围与标准准确率口径。
- EmergentMind: MMLU benchmark 解读，适合查看 few-shot 常见设置、overall 与 macro 的说明，以及对评测机制的概括。
- Galileo: MMLU 工程指南，适合参考模型发布前如何把 MMLU 用作回归测试、如何记录学科差值与模板变量。
- Artificial Intelligence Wiki: MMLU 知识评测条目，适合了解题库污染、题目质量问题、提示模板导致波动等工程风险。
- 原始论文与官方题库说明: 适合确认数据集设计初衷、标准题型、评测口径，以及后续清洗版本的来源说明。
- 使用建议：真正落地时，应优先获取最新版题库说明和清洗版数据集页面，明确你使用的是原始版、Redux 版还是其他修订版，否则不同实验结果可能不可比。

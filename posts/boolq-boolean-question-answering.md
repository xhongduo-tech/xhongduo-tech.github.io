## 核心结论

BoolQ 是一个段落级别的布尔问答基准。布尔问答，意思是答案只能是 `True` 或 `False`。它包含 15,942 个自然语言问题，配套上下文来自 Wikipedia 段落，目标不是生成一段解释，而是判断“这段话是否支持这个问题里的断言”。

它适合作为问答系统、阅读理解模型、检索增强问答链路的最低门槛校验集。原因很直接：问题表面上只是二分类，实际却要求模型处理否定、时间条件、隐含前提、术语对应、常识压缩表达等语义现象。一个只能靠关键词匹配的系统，往往在 BoolQ 上很快暴露问题。

BoolQ 的核心指标通常只有一个：

$$
Accuracy = \frac{\#\text{正确预测}}{N}
$$

这里的 Accuracy 就是准确率，意思是“所有样本里答对了多少比例”。如果开发集有 100 条样本，模型答对 78 条，那么准确率就是 $78/100=0.78$，也就是 78%。

但 BoolQ 不能只看“跑通”。它有明显的多数类偏置，也就是某一类样本天然更多。公开资料里常提到 `True` 占比大约 62%，这意味着一个很差的系统只要大量猜 `True`，都可能碰到一个不低的基线。所以工程上不能把“比随机好”当作结论，必须设定明确门槛，例如：

- 新模型 Dev Accuracy 至少不低于稳定版
- 如果要上线，要求至少高出稳定版 2 个百分点
- 若低于 `baseline - 0.02`，触发回滚

玩具例子最容易理解这个任务。问题是：

`Is Mount Everest the tallest mountain on Earth?`

配套段落如果写着“Mount Everest is Earth's highest mountain above sea level”，那么标签就是 `True`。这已经是完整的 BoolQ 任务，不需要生成解释，不需要外部搜索，只判断段落是否支持问题中的断言。

---

## 问题定义与边界

一个 BoolQ 样本只有三个核心字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `question` | string | 自然语言疑问句 |
| `context` | paragraph | 来自 Wikipedia 的段落 |
| `label` | `{True, False}` | 段落是否支持该断言 |

这里的 `context` 就是上下文，也就是给模型看的证据段落。`label` 是监督信号，表示标准答案。任务本质上是“给定证据做二分类”，不是开放式问答。

边界要先说清楚，否则很多新手会把 BoolQ 理解错。

第一，它不考外部知识。模型不能因为“自己知道”答案就直接答。评测定义要求依据给定段落作答。如果段落没提供支撑，即使常识上你知道答案，也不能算答对。

第二，它不要求生成自然语言解释。你可以在产品里额外生成解释，但评测本身只看布尔标签。

第三，它只关心段落是否支持问题断言，不等于“段落中出现了同样的词”。支持关系可能来自转述、否定、时间限定或上位词下位词关系。

一个新手版本的例子：

- `question`: `Is aspirin a type of antibiotic?`
- `context`: `Aspirin is a medication used to reduce pain, fever, or inflammation. It is an analgesic and antipyretic.`
- `label`: `False`

这里段落没有出现 “antibiotic” 的反义定义，但已经给出 aspirin 是 analgesic。analgesic 指镇痛药，不是抗生素。模型必须基于段落语义做排除判断，而不是只做词面命中。

因此，BoolQ 的能力边界可以概括成一句话：它测试“基于给定段落的支持性判断”，尤其对隐式信息敏感，但不覆盖开放生成、多跳检索和长文档推理的全部问题。

---

## 核心机制与推导

从建模角度看，BoolQ 最自然的写法是二元分类。二元分类，意思是输出空间只有两个类别。输入是一对文本：`question` 和 `context`，模型把它们编码后输出两个分数，也就是 logits。logits 指模型未归一化的类别分值。

形式上可以写成：

$$
logits = Model(question, context)
$$

再通过 Softmax 得到两个类别的概率。Softmax 是把一组分数变成概率分布的函数。

$$
p(True) = softmax(logits)[True]
$$

最后取概率更高的类别作为预测结果：

$$
prediction = \arg\max p
$$

如果把问题写得更工程化，一次推理链路通常是：

1. 读取 `question` 和 `context`
2. 按模板拼接，例如 `[CLS] question [SEP] context [SEP]`
3. 送入 Transformer 编码器
4. 取分类头输出 `True/False`
5. 用 Accuracy 汇总整个 Dev 集表现

这里的 Transformer 可以简单理解为“能让问题和上下文互相看见彼此”的编码器。它不是只分别看两个句子，而是学习它们之间的对应关系，例如否定词修饰了谁，时间条件作用在哪个事件上。

玩具例子可以写成：

- `question`: `Does the paragraph state that apples are red?`
- `context`: `Many apples are red, though some varieties are green or yellow.`
- 模型输出概率：`[0.2, 0.8]`

如果我们约定 `[False, True]` 的顺序，那么 `0.8` 对应 `True`，预测正确。这个例子看起来简单，但已经体现了“不是逐词复制，而是判断段落是否支持命题”。

评测阶段还要注意一个细节：答案比对前通常先做规范化，也就是 Quasi-exact match。它的意思是“不是完全逐字符比较，而是先做轻量标准化，再比对内容是否一致”。常见写法是：

$$
normalize(answer) = answer.strip().lower()
$$

`strip()` 去掉首尾空格，`lower()` 统一为小写。这样 `True`、` true `、`TRUE` 会被视为同一结果。对 BoolQ 这种布尔标签来说，这个步骤不复杂，但在自动化评测里很关键，因为格式差异不应被误判为能力下降。

---

## 代码实现

工程实现建议拆成两部分：数据读取与规范化、模型推理与打分。这样做的好处是职责清晰，后续更容易复用到 CI。

下面给一个可运行的最小 Python 版本。它没有依赖真实大模型，而是用一个规则函数模拟预测流程，目的是把评测框架讲清楚。

```python
from dataclasses import dataclass

@dataclass
class Sample:
    question: str
    context: str
    label: str  # "true" or "false"

def normalize(text: str) -> str:
    # Quasi-match: 统一大小写与首尾空格
    return text.strip().lower()

def predict_boolq(question: str, context: str) -> str:
    q = normalize(question)
    c = normalize(context)

    # 一个玩具规则模型：仅用于演示评测流程
    if "tallest mountain on earth" in q and "highest mountain above sea level" in c:
        return "true"
    if "aspirin a type of antibiotic" in q and "analgesic" in c:
        return "false"
    if "apples are red" in q and "many apples are red" in c:
        return "true"

    # 默认猜多数类，模拟弱模型常见行为
    return "true"

def evaluate(samples):
    correct = 0
    for sample in samples:
        pred = predict_boolq(sample.question, sample.context)
        if normalize(pred) == normalize(sample.label):
            correct += 1
    return correct / len(samples)

def check_gate(acc: float, baseline: float, rollback_delta: float = 0.02):
    # 若低于稳定版 2 个百分点，则视为不可发布
    if acc < baseline - rollback_delta:
        raise RuntimeError(
            f"RollbackError: accuracy={acc:.3f} < baseline-threshold={baseline - rollback_delta:.3f}"
        )

dev_set = [
    Sample(
        question="Is Mount Everest the tallest mountain on Earth?",
        context="Mount Everest is Earth's highest mountain above sea level.",
        label="True",
    ),
    Sample(
        question="Is aspirin a type of antibiotic?",
        context="Aspirin is an analgesic and antipyretic medication.",
        label="False",
    ),
    Sample(
        question="Does the paragraph state that apples are red?",
        context="Many apples are red, though some varieties are green or yellow.",
        label="True",
    ),
]

acc = evaluate(dev_set)
assert abs(acc - 1.0) < 1e-9

baseline = 0.75
check_gate(acc, baseline)

print(f"dev accuracy = {acc:.3f}")
```

这段代码里有三个关键点。

第一，`normalize()` 必须独立成函数，并在评估、推理输出格式化、可能的人类标注回收流程里复用。否则同一条样本可能因为大小写差异被重复算错。

第二，`evaluate()` 只做一件事：统计准确率。统计逻辑越简单，越不容易在 CI 里藏 bug。

第三，`check_gate()` 明确写出可回滚条件。回滚条件，意思是“模型结果差到什么程度时，自动阻止发布”。例如稳定版 Dev Accuracy 是 0.78，阈值设为 2 个百分点，那么新模型低于 0.76 就不应继续上线。

真实工程例子通常不是手写规则，而是这样一条链路：

- 从 Dev 集读取 `question/context/label`
- 调用线上候选模型或离线 checkpoint
- 输出 `true/false`
- 跑统一 normalization
- 计算 Accuracy
- 与稳定版指标对比
- 低于阈值则阻断部署或回滚

这个流程足够轻，适合做成 CI 每次自动执行。它不替代更复杂的离线分析，但非常适合做“最低质量红线”。

---

## 工程权衡与常见坑

BoolQ 的工程价值很高，但前提是你知道它哪里容易骗人。

最常见的问题是多数类基线。多数类基线，意思是“只猜样本里出现更多的那一类，也能拿到的分数”。如果 `True` 约占 62%，那么一个只会偏向 `True` 的模型就能轻松拿到 60% 左右。这说明“有分数”不等于“懂推理”。

下面这个表最实用：

| 坑 | 原因 | 规避 |
| --- | --- | --- |
| 多数类基线 | `True` 占比偏高，弱模型靠猜测也能过 60% | 同时看类别分布、混淆矩阵，并设高于多数类的 gate |
| 忽略 normalization | 空格、大小写差异导致误判 | 统一使用 Quasi-match 规范化 |
| 只看总 Accuracy | 看不出否定、时间、指代等薄弱点 | 增加错误切片分析 |
| 训练评估不一致 | 训练输入模板、推理模板、评测脚本不同 | 固化模板与标签映射 |
| 把 BoolQ 当开放问答 | 生成解释质量掩盖了分类错误 | 把布尔标签评测与解释生成分开 |

新手容易踩的一个坑是把模型输出写成自然语言，例如输出 `"Yes, according to the passage..."`。如果评测脚本只接受 `true/false`，那就会直接算错。解决方法不是放宽标准，而是明确在后处理阶段把模型输出映射为标准标签。

另一个典型坑是“表面关键词学习”。比如模型看见问题里有 `is`、`does`、`can` 这类模式，再结合上下文里某些高频词，就机械地偏向 `True`。在 Dev 集上它可能从 62% 提到 68%，看起来有提升，但本质仍然没有理解段落。

举一个更贴近真实系统的例子。你做了一个检索增强问答服务，先用搜索召回 Wikipedia 片段，再让模型回答 yes/no。某次你升级了提示词，Dev Accuracy 从 0.79 降到 0.77。表面上只掉 2 个百分点，但如果你的上线门槛是“不得低于稳定版 0.02”，这已经刚好踩线。此时正确做法不是凭感觉上线，而是：

- 先检查 normalization 是否一致
- 再看检索片段是否变短或截断
- 最后看错误是否集中在否定、时间条件和跨句指代

因为 BoolQ 的很多错例并不是“模型完全不会”，而是链路前面某一步轻微退化后，语义证据不完整了。

---

## 替代方案与适用边界

BoolQ 适合做 yes/no 推理的快速质量门槛，但它不是万能评测。

如果你的目标是快速回归控制，BoolQ Dev 非常合适。样本结构简单、指标明确、自动化成本低。尤其在模型迭代频繁时，它能作为第一道门，及时发现明显退化。

但如果你要验证更深层的推理，BoolQ 就不够了。它的标签空间太小，很多复杂错误被压缩成一个 `False`，很难看出错在证据缺失、逻辑链断裂，还是歧义处理失败。这时可以考虑其他数据集。

| 数据集 | 目标 | 适用场景 |
| --- | --- | --- |
| BoolQ | yes/no 推理 | CI gate、快速迭代、最低能力校验 |
| MultiRC | 多问题、多选项、跨句理解 | 更深的逻辑推理与证据整合 |
| ReCoRD | 指代与抽取 | 检查实体解析与候选选择 |
| NLI 数据集 | 蕴含/矛盾/中立判断 | 更细粒度的支持关系分析 |

这里的 NLI 是自然语言推断，意思是判断一个句子是否蕴含、矛盾或中立于另一个句子。它比 BoolQ 更细，因为 BoolQ 只有两类，而 NLI 通常至少三类。

一个新手版对比很清楚：如果你只是想知道“系统是否能根据段落判断是或否”，用 BoolQ 就够；如果你想知道“系统能不能跨多个句子拼出完整逻辑，并定位关键证据”，那么 MultiRC 更合适。

所以更稳的工程组合往往是：

- BoolQ 做日常 CI gate
- 更复杂的数据集做阶段性回归
- 在线指标验证最终用户价值

不要试图让一个数据集覆盖所有能力。评测集的意义不是“代替真实世界”，而是“稳定地暴露某一类问题”。

---

## 参考资料

- Hugging Face Dataset Card: BoolQ，包含样本规模、字段结构与数据集说明  
  https://huggingface.co/datasets/google/boolq
- AI Wiki: BoolQ，整理了任务定义、split、常见基线与使用背景  
  https://aiwiki.ai/wiki/boolq
- FlagEval Evaluation Metrics，说明 Accuracy 与 Quasi-exact match 的评测规范  
  https://flageval.baai.ac.cn/docs/en/nlp/metrics.html

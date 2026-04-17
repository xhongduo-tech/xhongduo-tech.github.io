## 核心结论

弱监督标注策略的核心不是“用规则直接代替人工标注”，而是把多个带噪声的标注源，统一聚合成一个概率标签系统。这里的“带噪声”指每个标注源都可能频繁出错，而且一开始并不知道它到底有多准。Snorkel 这类框架的关键贡献，是把这些标注源写成可执行的标注函数（Labeling Function, LF，可以理解成“会投票或弃权的小程序”），再用标签模型（Label Model，可以理解成“学习谁更可靠、谁在重复发声的聚合器”）估计每条样本的真实标签分布。

玩具例子最能说明问题。假设有 10 个 LF：

- 前 5 个 LF 来自同一条规则的不同改写，彼此高度相关，单个准确率只有 50%
- 后 5 个 LF 来自不同知识源，彼此近似独立，单个准确率 99%

如果模型错误地把这 10 个 LF 都当成独立投票者，那么前 5 个“重复噪声”会被误认为是强证据，后 5 个真正高质量信号反而可能被低估。建模相关性之后，系统才会意识到“这 5 票其实更像 1 票”。

| 场景 | 前 5 个相关 LF 的估计准确率 | 后 5 个独立 LF 的估计准确率 | 结果 |
|---|---:|---:|---|
| 不建模相关性 | 被高估，可能接近 100% | 被低估，可能接近 50% | 重复噪声被当成高置信度 |
| 建模相关性 | 回到接近真实的 50% | 回到接近真实的 99% | 聚合结果恢复正常 |

所以，弱监督真正解决的问题是：在没有逐条人工标注的前提下，如何从“多源、低质、相关、覆盖不均”的投票中恢复出可用于训练模型的概率标签。它不是自动化替代标注员，而是把人工逐条标注，替换成“规则工程 + 小规模验证集 + 概率建模”。

---

## 问题定义与边界

问题定义可以写成一句话：给定未标注数据 $X=\{x_1,\dots,x_n\}$，以及多个会输出标签或弃权的 LF，如何在没有真实标签 $Y$ 的情况下估计每条样本的标签概率 $P(y_i\mid \Lambda_i)$。这里的 $\Lambda_i$ 是第 $i$ 条样本收到的全部 LF 投票。

对白话一点的理解是：不要把字典匹配、依存路径规则、关键词命中、近义词替换、知识库查询这些信号当成最终标签，而是统一看成一群“投票者”。每来一条数据，就让所有投票者回答“正类”“负类”或“我不知道”，最后再交给标签模型做裁决。

一个简化流程如下：

`数据 --> 多个 LF --> 投票矩阵 Λ --> 标签模型 --> 概率标签 --> 下游判别模型`

这里的几个术语，新手需要先区分清楚：

| 术语 | 数学对象 | 白话解释 |
|---|---|---|
| 样本 $x_i$ | 一条输入数据 | 例如一段文本、一张图像、一条日志 |
| LF | 函数 $ \lambda_j(x_i)$ | 一段规则程序，输出标签或弃权 |
| 弃权 | 特殊值 `abstain` | 这条规则对当前样本没有把握 |
| 投票矩阵 $\Lambda$ | $n \times m$ 矩阵 | 每行是一条样本，每列是一个 LF |
| 概率标签 | $P(y_i \mid \Lambda_i)$ | 不是“绝对正确”，而是“有多大概率正确” |

这里有几个边界条件必须说清楚：

| 条件 | 含义 | 为什么重要 |
|---|---|---|
| LF 可以 `abstain` | 弃权，表示“不确定” | 避免为了覆盖率强行乱标 |
| LF 需要互补覆盖 | 不同 LF 覆盖不同样本片段 | 否则样本大量无人投票 |
| LF 不要求高精度 | 单个 LF 可以很弱 | 聚合后的整体质量才是重点 |
| 需要评估集 | 至少留一小部分人工标注验证 | 否则无法判断系统是否在放大偏差 |
| 标签定义要稳定 | 任务边界必须明确 | 否则 LF 会互相冲突但不是“噪声”，而是任务本身没定义清楚 |

这意味着弱监督不是“零人工成本”。更准确的定位是：把大规模逐条标注，替换成少量验证集 + 一批规则工程。对新手最常见的误解，是把它理解成“自动打标签工具”。其实它是“程序化构造噪声标签，再用模型校正噪声”的方法。

真实工程例子是 NLP 关系抽取。假设任务是识别“公司收购公司”关系，你可以设计这样的 LF：

- 文本出现 `acquired`、`bought`、`purchased` 等收购触发词
- 两个实体都出现在公司词典中
- 依存路径中出现收购类谓词
- 新闻标题中出现并购模板句式
- 若句子是否定语境，如 `did not acquire`，则弃权或投反票
- 若句子谈论的是融资、合作、投资而不是收购，则投负类

这些 LF 单独看都不可靠，但合起来可以在没有大规模人工标注的情况下，先造出一批可用训练数据。

再看一个更具体的句子级例子：

| 句子 | 关键词 LF | 公司词典 LF | 否定 LF | 直觉判断 |
|---|---:|---:|---:|---|
| `Microsoft acquired GitHub in 2018.` | 正 | 正 | 弃权 | 很可能正类 |
| `Microsoft did not acquire OpenAI.` | 正 | 正 | 负 | 存在冲突，需聚合 |
| `Microsoft partnered with OpenAI.` | 弃权 | 正 | 弃权 | 不能因为都是公司就判收购 |
| `The acquisition rumor was denied.` | 正 | 弃权 | 负 | 关键词命中但语义是负类 |

这个例子说明：LF 的价值不在于“单条规则决定真相”，而在于“把不同角度的弱证据组织起来”。

---

## 核心机制与推导

Snorkel 的标签模型通常从投票矩阵 $\Lambda \in \{-1,0,1\}^{n\times m}$ 出发。这里 $n$ 是样本数，$m$ 是 LF 数，$0$ 表示弃权，$\pm1$ 表示类别投票。对多分类任务，也可以扩展成更多标签值。

核心思想是：把“是否投票”“是否正确”“LF 之间是否相关”写成因子函数（factor）。因子函数可以理解成“把一个复杂系统拆成多个可学习的小判断项”。

对二分类任务，一个常见写法是：

$$
\phi^{(\text{Lab})}_{i,j}=\mathbf{1}\{\Lambda_{i,j}\neq 0\}
$$

$$
\phi^{(\text{Acc})}_{i,j}=\mathbf{1}\{\Lambda_{i,j}=y_i\}
$$

$$
\phi^{(\text{Corr})}_{i,j,k}=\mathbf{1}\{\Lambda_{i,j}=\Lambda_{i,k}\neq 0\},\quad (j,k)\in C
$$

其中：

- $\phi^{(\text{Lab})}$ 表示第 $j$ 个 LF 在第 $i$ 条样本上是否发声
- $\phi^{(\text{Acc})}$ 表示该 LF 的投票是否与真实标签一致
- $\phi^{(\text{Corr})}$ 表示两个 LF 是否表现出相关性
- $C$ 是候选相关对集合
- 参数向量 $w$ 负责给这些因子分配权重

把这些因子合在一起，可以写成一个生成模型：

$$
P_w(\Lambda, Y)\propto \exp\left(\sum_{i=1}^{n}\sum_{f} w_f \phi_f(\Lambda_i, y_i)\right)
$$

如果写成单条样本的后验形式，标签模型关心的是：

$$
P(y_i \mid \Lambda_i)=\frac{P(\Lambda_i, y_i)}{\sum_{y' \in \mathcal{Y}} P(\Lambda_i, y')}
$$

训练时并没有真实 $Y$，所以要通过最大似然估计，在只观测到 $\Lambda$ 的前提下反推最合理的参数 $w$。直觉上，它在回答两个问题：

1. 哪些 LF 平均更可靠
2. 哪些 LF 其实在重复表达同一来源的信息

推断完成后，标签模型输出的不是硬标签，而是概率标签。例如：

$$
P(y_i=1\mid \Lambda_i)=0.87,\qquad P(y_i=0\mid \Lambda_i)=0.13
$$

这一步很重要，因为下游判别模型通常更适合吃“软标签”。所谓软标签，就是不直接说“它一定是正类”，而是说“它有 87% 概率是正类”。这比硬标签更符合弱监督场景下的真实不确定性。

再看一个矩阵示意：

| 样本 | LF1 | LF2 | LF3 | LF4 | 含义 |
|---|---:|---:|---:|---:|---|
| $x_1$ | 1 | 1 | 0 | -1 | 两票正类，一票反类，一票弃权 |
| $x_2$ | 0 | 1 | 0 | 1 | 两个 LF 支持正类 |
| $x_3$ | -1 | -1 | -1 | 0 | 多个 LF 支持反类 |
| $x_4$ | 0 | 0 | 0 | 0 | 无人覆盖，无法可靠推断 |

如果 LF1 和 LF2 总是一起输出 1，那么 $\phi^{(\text{Corr})}$ 会频繁被触发。模型就会学到：这两个 LF 不是两份独立证据，而是高度重复的同一类信号。于是它会降低“它们各自都很准”的置信度，避免把重复投票误当成真值。

从新手角度看，可以把“相关性建模”理解成一句话：  
`五条抄同一个来源的规则，不应被当成五个独立专家。`

这也是为什么结构学习（structure learning，可以理解成“自动找出哪些 LF 对应该显式连边”）很关键。没有它，系统默认独立，容易高估重复规则的价值；有了它，模型才知道哪些票应该“打折”。

实际系统里，还常看三个统计量：

| 指标 | 含义 | 典型用途 |
|---|---|---|
| Coverage | LF 覆盖了多少样本 | 看是否大量样本无人投票 |
| Overlap | 多个 LF 同时命中同一样本的比例 | 看信号是否会发生交互 |
| Conflict | 多个 LF 在同一样本上投不同标签的比例 | 看噪声和任务边界是否有问题 |

它们不是最终目标，但对诊断 LF 系统非常有用。经验上，先看这三个数，再看模型精度，排障效率最高。

---

## 代码实现

工程上通常分四步：定义 LF、生成投票矩阵、训练 `LabelModel`、再用概率标签训练下游判别模型。

先看一个可运行的玩具版 Python 示例。它不依赖 Snorkel，但能把 `abstain`、冲突、加权聚合这些核心概念说明白。

```python
from math import exp
from dataclasses import dataclass

ABSTAIN = 0
NEG = -1
POS = 1

@dataclass
class Example:
    text: str

def lf_keyword(example: Example) -> int:
    text = example.text.lower()
    keywords = ("acquired", "bought", "purchased")
    return POS if any(word in text for word in keywords) else ABSTAIN

def lf_company_pair(example: Example) -> int:
    text = example.text.lower()
    companies = ("google", "microsoft", "openai", "github", "apple")
    hit = sum(name in text for name in companies)
    return POS if hit >= 2 else ABSTAIN

def lf_negation(example: Example) -> int:
    text = example.text.lower()
    negative_patterns = (
        "did not acquire",
        "didn't acquire",
        "denied acquiring",
        "acquisition rumor",
    )
    return NEG if any(p in text for p in negative_patterns) else ABSTAIN

def lf_partnership(example: Example) -> int:
    text = example.text.lower()
    non_acquisition_patterns = ("partnered with", "invested in", "collaborated with")
    return NEG if any(p in text for p in non_acquisition_patterns) else ABSTAIN

def sigmoid(score: float) -> float:
    return 1.0 / (1.0 + exp(-score))

def aggregate(votes: list[int], weights: list[float]) -> float:
    if len(votes) != len(weights):
        raise ValueError("votes and weights must have the same length")

    score = 0.0
    for vote, weight in zip(votes, weights):
        if vote == ABSTAIN:
            continue
        score += vote * weight
    return sigmoid(score)

def apply_lfs(example: Example) -> list[int]:
    lfs = [lf_keyword, lf_company_pair, lf_negation, lf_partnership]
    return [lf(example) for lf in lfs]

example_pos = Example("Microsoft acquired GitHub in 2018.")
example_neg = Example("Microsoft did not acquire OpenAI.")
example_other = Example("Microsoft partnered with OpenAI on infrastructure.")

votes_pos = apply_lfs(example_pos)
votes_neg = apply_lfs(example_neg)
votes_other = apply_lfs(example_other)

weights = [1.2, 0.8, 1.5, 1.3]

prob_pos = aggregate(votes_pos, weights)
prob_neg = aggregate(votes_neg, weights)
prob_other = aggregate(votes_other, weights)

assert votes_pos == [POS, POS, ABSTAIN, ABSTAIN]
assert votes_neg == [POS, POS, NEG, ABSTAIN]
assert votes_other == [ABSTAIN, POS, ABSTAIN, NEG]

assert prob_pos > prob_neg
assert prob_other < 0.5
assert 0.0 < prob_pos < 1.0
assert 0.0 < prob_neg < 1.0
assert 0.0 < prob_other < 1.0

print("votes_pos =", votes_pos, "prob_pos =", round(prob_pos, 4))
print("votes_neg =", votes_neg, "prob_neg =", round(prob_neg, 4))
print("votes_other =", votes_other, "prob_other =", round(prob_other, 4))
```

这个示例为什么可运行、而且有解释力：

| 设计点 | 作用 |
|---|---|
| `ABSTAIN = 0` | 明确把“没把握”与“负类”分开 |
| `lf_negation` / `lf_partnership` | 展示反向证据不是异常，而是正常组成部分 |
| `aggregate` 跳过弃权 | 弃权不应被误算成负样本 |
| 三条断言 | 保证例子行为符合预期 |

上面只是简化版。真实实现通常直接用 Snorkel。下面给出一个更接近工程实践的最小示例：

```python
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.analysis import LFAnalysis

ABSTAIN = -1
NEG = 0
POS = 1

COMPANY_KB = {"google", "microsoft", "openai", "github", "apple"}

@labeling_function()
def lf_keyword(x):
    text = x.text.lower()
    if any(word in text for word in ("acquired", "bought", "purchased")):
        return POS
    return ABSTAIN

@labeling_function()
def lf_kb_match(x):
    text = x.text.lower()
    hit = sum(name in text for name in COMPANY_KB)
    if hit >= 2:
        return POS
    return ABSTAIN

@labeling_function()
def lf_negative_pattern(x):
    text = x.text.lower()
    if any(p in text for p in ("did not acquire", "denied acquiring", "acquisition rumor")):
        return NEG
    return ABSTAIN

@labeling_function()
def lf_non_acquisition_event(x):
    text = x.text.lower()
    if any(p in text for p in ("partnered with", "invested in", "collaborated with")):
        return NEG
    return ABSTAIN

train_df = pd.DataFrame(
    {
        "text": [
            "Microsoft acquired GitHub in 2018.",
            "Microsoft did not acquire OpenAI.",
            "Apple bought a small AI startup.",
            "Google partnered with OpenAI.",
            "The acquisition rumor was denied by the company.",
        ]
    }
)

lfs = [lf_keyword, lf_kb_match, lf_negative_pattern, lf_non_acquisition_event]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=train_df)

analysis_df = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(analysis_df[["Polarity", "Coverage", "Overlaps", "Conflicts"]])

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=300, log_freq=100, seed=42)

probs_train = label_model.predict_proba(L=L_train)
preds_train = probs_train.argmax(axis=1)

print("L_train =")
print(L_train)
print("probs_train =")
print(probs_train)
print("preds_train =", preds_train.tolist())
```

这个版本比玩具版多了三件关键事情：

1. 用 `PandasLFApplier` 把多条 LF 批量作用到数据集上，得到投票矩阵 `L_train`
2. 用 `LFAnalysis` 看每个 LF 的覆盖率、重叠率、冲突率
3. 用 `LabelModel.predict_proba` 输出软标签，而不是只输出硬标签

常见 LF 类型可以这样组织：

| LF 类型 | 示例 | 何时 `abstain` |
|---|---|---|
| 规则型 | 关键词、正则、句法模板 | 没命中规则时 |
| 知识库型 | 实体在词典或 KB 中共现 | 实体解析失败或证据不足时 |
| 模板型 | 标题句式、公告格式、网页结构模式 | 句式不完整或上下文缺失时 |
| 统计型 | TF-IDF、相似度阈值、共现分数 | 分数落在灰区时 |
| 模型型 | 小模型或旧模型打分 | 置信度不够时弃权 |

真实工程流程通常不会停在 `LabelModel`。更常见的做法是：先用几十个 LF 给 10 万条文本生成概率标签，再训练一个 Transformer、BiLSTM 或其他判别模型。原因很直接：

- LF 擅长快速扩大标签规模
- 标签模型擅长把多源噪声整合成软标签
- 判别模型擅长从上下文中学习 LF 没显式编码的模式

因此一个完整流程通常是：

`未标注语料 --> LF --> LabelModel --> 软标签 --> 判别模型 --> 人工误差分析 --> 回写新 LF`

这也是弱监督真正能落地的地方。它不是终点模型，而是训练数据工厂。

---

## 工程权衡与常见坑

弱监督最常见的误区，不是“LF 写得不够多”，而是“LF 太像了却没建模相关性”。比如做实体关系抽取时，词典 LF 和同义词 LF 总是一起命中，它们本质上可能来自同一知识源。若模型把这两票当成独立强证据，就会把重复误报放大成高置信伪标签。

另一个常见坑是 label density 过高。所谓 label density，可以理解成“每条样本平均会收到多少 LF 投票”。很多新手为了追求覆盖率，让 LF 尽量少弃权，结果每个 LF 都在大面积发声，噪声会迅速累积，标签模型反而更容易过拟合。

| LF 密度 | 风险 | 使用 `abstain` 后的改善 |
|---|---|---|
| 低 | 覆盖不足，很多样本没有标签 | 需要补更多互补 LF |
| 中 | 通常最好 | 平衡覆盖率与噪声 |
| 高且很少弃权 | 容易过拟合、放大误报 | 通过弃权保留高精度区域 |

实际调参时，可以按这个顺序排查：

1. 先看每个 LF 的覆盖率、冲突率、重叠率
2. 再对比 `MajorityLabelVoter` 和 `LabelModel`
3. 如果提升不明显，优先检查高相关 LF，而不是盲目增加数量
4. 需要时启用结构学习，或手动移除明显重复 LF
5. 最后用一小部分人工验证集看精度、召回率、校准情况

这里 `MajorityLabelVoter` 的作用很像基线模型，也就是“简单多数投票”。如果复杂的 `LabelModel` 还不如多数投票，通常说明两种情况之一：

- LF 质量整体太差，模型没有可学习信号
- 相关性和覆盖结构没有被正确处理

再给出一组更实用的排障表：

| 症状 | 常见原因 | 处理方式 |
|---|---|---|
| 大量样本全是弃权 | LF 覆盖面太窄 | 补充更宽覆盖的规则，但保留弃权机制 |
| 覆盖率很高但精度很差 | LF 太激进、误报多 | 提高阈值，扩大灰区，增加反向 LF |
| `LabelModel` 不如多数投票 | LF 高相关但未处理 | 合并相似 LF，删除重复源，检查结构学习 |
| 训练集效果高、验证集差 | 伪标签偏差被下游模型放大 | 减少低精度 LF 权重，增大人工验证集 |
| 某一类几乎预测不到 | 类别不平衡或 LF 偏向单边 | 专门补少数类 LF，显式检查类先验 |

对新手尤其重要的一点是：  
`abstain` 不是“失败”，而是弱监督里最重要的质量控制手段之一。  
宁可少投高质量票，也不要让每条 LF 对所有样本都发声。

还有一个容易被忽略的问题是“任务定义漂移”。例如你一开始把“收购传闻”当成正类，后来又想改成负类，那 LF 冲突会显著上升，但这不是模型问题，而是标签定义变了。弱监督系统会把定义混乱放大成工程混乱。

---

## 替代方案与适用边界

弱监督不是唯一方案。它和远程监督、众包标注、自训练各自解决的是不同成本结构下的问题。

| 方案 | 人工成本 | 噪声控制 | 所需先验 | 适用场景 | 常见验证手段 |
|---|---:|---:|---|---|---|
| 弱监督 | 低到中 | 中到高，可建模 | 规则、KB、启发式 | 有领域知识但缺标签 | 对照 `MajorityLabelVoter`、人工验证集 |
| 远程监督 | 低 | 较弱 | 需要现成知识库对齐 | 关系抽取、实体链接 | 抽样人工审核 |
| 众包标注 | 中到高 | 通常较强 | 任务说明清晰 | 需要稳定高质量标签 | 多标注者一致性 |
| 自训练 | 低 | 依赖初始模型质量 | 需要一个可用种子模型 | 已有少量标注、模型可迁移 | 留出集性能变化 |

几个边界需要明确：

- 若你没有任何规则、词典、知识库、模板信号，弱监督很难启动
- 若任务标签定义极其主观，比如“这段评论是否有轻微讽刺”，LF 往往难写，众包或专家标注更稳
- 若已有高质量知识库且匹配关系清晰，远程监督可能更直接
- 若已经有一个不错的小样本监督模型，自训练可能更省工程成本
- 若上线要求强可解释、强审计，弱监督通常比黑盒自训练更容易追责，因为每条 LF 都可检查
- 若领域变化非常快，LF 维护成本会上升，此时要评估规则工程是否值得持续投入

所以弱监督最适合的条件是：你手里没有足够人工标签，但有可程序化表达的领域知识，并且能接受先造一批带噪训练数据，再用下游模型和人工验证逐步修正。

一个实用判断标准是：

| 你手里有什么 | 更适合的方法 |
|---|---|
| 大量专家知识、少量人工时间 | 弱监督 |
| 高质量知识库、关系映射清楚 | 远程监督 |
| 标签定义清晰、预算充足 | 众包或人工标注 |
| 已有不错的小模型 | 自训练或半监督 |

工程上，弱监督常被放在“第一阶段提速”位置：先把无标签数据变成可训练数据，再决定是否追加人工精标。它不是和人工标注对立，而是减少人工标注的总量，并把人工集中在验证和误差分析上。

---

## 参考资料

1. [**Snorkel: Rapid Training Data Creation with Weak Supervision**](https://link.springer.com/article/10.1007/s00778-019-00552-1)  
   - 作者：Alexander Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, Christopher Ré  
   - 发表：*The VLDB Journal*, 2020（在线发表于 2019-07-15）  
   - 核心贡献：系统化介绍程序化标注、LF、生成式标签模型和端到端弱监督流水线  
   - 用途：理论总览与系统设计入口

2. [**Data Programming: Creating Large Training Sets, Quickly**](https://snorkel.ai/research-paper/data-programming-creating-large-training-sets-quickly/)  
   - 作者：A. Ratner 等  
   - 发表：NeurIPS 2016  
   - 核心贡献：提出 data programming 范式，说明如何在没有真实标签时，从多个噪声标注源中恢复训练标签  
   - 用途：理解弱监督的生成建模基础

3. [**Get Started - Snorkel**](https://snorkelproject.org/get-started/)  
   - 核心内容：官方最小工作流，覆盖 `labeling_function`、`PandasLFApplier`、`LabelModel`、软标签到下游模型训练  
   - 用途：工程入门和 API 对照

4. [**Intro Tutorial: Data Labeling / Labeling Functions**](https://snorkelproject.org/use-cases/01-spam-tutorial/)  
   - 核心内容：用垃圾评论分类任务演示如何写 LF、看覆盖率与冲突、训练标签模型  
   - 用途：新手上手示例，适合理解 LF 应该怎么设计

5. [**snorkel-tutorials**](https://github.com/snorkel-team/snorkel-tutorials)  
   - 核心内容：官方教程仓库，包含 `spam`、`spouse`、视觉关系等任务  
   - 用途：查找更完整的 notebook、数据集和端到端例子

6. **The Data Programming Book / Snorkel 相关综述与结构学习论文**  
   - 核心贡献：进一步解释 LF 相关性、结构学习、概率标签推断和误差分析方法  
   - 用途：补足“为什么相关 LF 不能当独立证据”的理论部分  
   - 说明：如果只读一篇，先读上面的 VLDB Journal 论文；如果要深入机制，再补这些材料

7. **弱监督在 NLP 信息抽取中的实验论文与案例**  
   - 典型方向：关系抽取、事件抽取、文本分类、医学文本抽取  
   - 核心贡献：展示弱监督相对远程监督、纯规则系统和少量人工监督的性能位置  
   - 用途：判断你的任务是否适合上弱监督，而不是只看概念演示

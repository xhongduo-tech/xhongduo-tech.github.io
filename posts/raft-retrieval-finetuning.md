## 核心结论

RAFT（Retrieval-Augmented Fine-tuning，检索增强微调）不是把 RAG 和 SFT 机械拼接，而是把“检索到的文档”直接写进监督微调样本，让模型在训练阶段反复学习两件事：

1. 有证据时，先定位证据，再回答。
2. 没证据时，明确拒答，而不是补全成一个看起来合理的答案。

标准 RAG 的主要问题，通常不是“检索不到文档”，而是“模型不一定会用文档”。推理时把 top-k 文档塞进上下文，并不等于模型稳定学会了证据定位、关键信息抽取、噪声过滤和低证据拒答。RAFT 的核心改动是把这些能力提前放进训练分布里。

它的训练目标本身没有变，仍然是标准的下一个词预测，即最小化负对数似然：

$$
\mathcal{L}_{\text{RAFT}}
=
-\sum_{t=1}^{T}\log p_\theta(y_t \mid y_{<t}, Q, C)
$$

其中：

- $Q$ 是问题
- $y_t$ 是第 $t$ 个目标 token
- $C$ 是检索上下文

RAFT 的关键在于：这个上下文 $C$ 不再只是普通提示词，而是被设计成两种训练形态之一：

$$
C \in \{D^* \cup D_k,\; D_k\}
$$

其中：

- $D^*$ 表示包含答案证据的 oracle 文档
- $D_k$ 表示若干个干扰文档（distractor documents）

第一种情况表示“有真证据，但被噪声包围”；第二种情况表示“只有噪声，没有支持证据”。这正是在模拟真实部署里的 top-k 检索结果。

对新手，最重要的理解是：RAFT 训练的不是“领域知识背诵”，而是“在领域文档中找证据并据此作答”的能力模式。  
因此，在医疗、法律、企业知识库、金融合规这类高风险问答里，RAFT 往往比纯 RAG 更稳，因为它更强调“证据使用行为”而不是“答案记忆”。

一个最小玩具例子如下。

- 问题：`轻度肾功能不全是否需要调整药物 X 剂量？`
- 正样本上下文：药品说明书剂量段落 + 儿童用药段落 + 不良反应段落 + 储存条件段落
- 负样本上下文：儿童用药段落 + 不良反应段落 + 储存条件段落 + 联用禁忌段落

RAFT 希望模型学会：

- 在正样本里只引用“剂量调整”段落
- 在负样本里输出“当前检索结果不足以支持结论”

这就是它和“把文档拼进 prompt”之间的本质差异。

---

## 问题定义与边界

先把问题说清楚。

RAG（检索增强生成）解决的是“模型参数中没有最新知识，或不应把全部知识写进参数”这个问题。  
SFT（监督微调）解决的是“模型回答格式、语气、任务边界不符合目标任务”这个问题。  
RAFT 解决的是第三个问题：模型虽然已经拿到了资料，但没有稳定学会如何从资料中找证据、忽略噪声，并在缺乏依据时拒答。

因此，RAFT 面向的是一个更窄、更明确的任务定义：

| 输入集合类型 | 训练目标 | 模型应学会的行为 |
| --- | --- | --- |
| $Q + (D^* \cup D_k)$ | 生成答案并引用支持证据 | 证据定位、证据抽取、带依据回答 |
| $Q + D_k$ | 生成拒答或证据不足说明 | 低证据克制、避免幻觉、拒答格式 |
| 仅 $Q$ | 不是 RAFT 的核心训练形态 | 不强调 |

这里的几个符号需要先解释。

| 符号 | 含义 | 对新手的直白解释 |
| --- | --- | --- |
| $Q$ | Question | 用户真正提出的问题 |
| $D^*$ | Oracle document | 确实包含答案依据的文档 |
| $D_k$ | Distractor documents | 看起来相关，但无法支持答案的文档 |
| Top-k | 检索返回的前 k 个结果 | 线上系统最常见的候选文档集合 |

这一定义有三个明确边界。

第一，RAFT 只适合“答案必须能在文档里找到依据”的任务。  
如果任务本身不强调证据，例如营销文案、创意写作、闲聊陪伴、通用润色，那么 RAFT 额外引入的检索上下文和样本构造成本通常不划算。

第二，RAFT 依赖可标注的 oracle 文档。  
也就是说，你至少要能回答：“这个问题的标准答案，究竟由哪份文档支持？”  
如果连这一点都无法确定，就没法稳定构造正样本和负样本。开放域闲聊通常不满足这个条件，法规问答、药品问答、企业制度问答则更容易满足。

第三，RAFT 不是检索器替代品。  
它只能提高“模型如何使用检索结果”的能力，不能替代“把正确文档找回来”这件事。如果 retriever 的召回率很差，正确文档长期排不进 top-k，RAFT 也不能凭空修复整个系统。

一个真实工程例子可以说明边界。

问题：`某药物对轻度肾功能不全患者是否需要调整剂量？`

如果线上检索结果包含：

- 药品说明书第 4.2 节“剂量与给药”
- 儿童患者剂量说明
- 肝功能异常警示
- 联合用药禁忌

那么这是 RAFT 很适合的场景，因为：

- 答案必须来自文档
- 可以标注出第 4.2 节是 oracle
- 其余几段都属于“语义接近但不支持当前问题”的干扰项

相反，如果问题是：`请写一段鼓励团队士气的周会发言稿`，RAFT 就不是优先方法，因为这里没有稳定的“证据引用”要求。

---

## 核心机制与推导

RAFT 的机制可以压缩成一句话：  
用和部署阶段相似的检索上下文做监督微调，并故意混入噪声，让模型学会“先判断证据是否成立，再决定如何回答”。

### 1. 样本分布设计

设训练样本总数为 $N$，正样本比例为 $P$，则：

$$
N_{\text{pos}} = P \cdot N,\qquad
N_{\text{neg}} = (1-P)\cdot N
$$

其中：

- 正样本：问题 $Q$ + 1 个 oracle 文档 $D^*$ + 若干干扰文档 $D_k$
- 负样本：问题 $Q$ + 若干干扰文档 $D_k$，但不提供 oracle 文档

经验上，很多实现会从 $P \approx 0.8$ 起步，也就是：

- 80% 左右样本用于学习“如何基于正确证据回答”
- 20% 左右样本用于学习“没有证据时如何拒答”

这不是固定定律，但它通常是一个比较稳的起点。

### 2. RAFT 改变的是条件分布，不是损失形式

普通 SFT 学的是：

$$
p(y \mid Q)
$$

也就是“给定问题，预测答案”。

普通 RAG 在线推理时，实际条件变成：

$$
p(y \mid Q, \text{top-k docs})
$$

问题在于：如果训练时几乎没见过这种输入形式，模型未必会稳定利用这些文档。它可能只是把文档当背景噪声，仍然按参数记忆作答。

RAFT 则直接训练下面两类条件分布：

$$
p_\theta(y \mid Q, D^*, D_k)
$$

以及

$$
p_\theta(y \mid Q, D_k)
$$

这意味着模型学到的行为不再是“看到问题就答”，而是：

1. 先检查上下文里是否存在支持证据
2. 有证据时抽取并组织答案
3. 无证据时保持克制

### 3. 为什么负样本很关键

只用正样本训练，会有一个典型副作用：模型学到“只要给了文档，就默认答案一定能在里面找到”。  
这种模型上线后容易出现两类错误：

- 引用了不支持当前问题的段落
- 把局部相关信息拼成一个错误结论

因此 RAFT 必须显式加入负样本，让模型学习：

$$
\text{If } \text{support}(Q, C)=0,\quad y \rightarrow \text{refusal}
$$

这里的 $\text{support}(Q, C)=0$ 表示上下文 $C$ 中没有足以支持问题 $Q$ 的证据。  
虽然真实训练里不会直接写这条规则，但负样本监督会把这种行为逐步写进参数。

### 4. 一个数字化玩具例子

假设有 10 个 QA 样本，设置：

- 正样本比例 $P=0.8$
- 每个样本使用 4 个干扰文档

则训练样本分布可以写成：

| 样本类型 | 数量 | 输入上下文 | 目标输出 |
| --- | --- | --- | --- |
| 正样本 | 8 | 1 个 $D^*$ + 4 个 $D_k$ | 给出答案并引用证据 |
| 负样本 | 2 | 4 个 $D_k$ | 拒答或声明证据不足 |
| 总计 | 10 | 模拟真实 top-k 检索环境 | 学会“有据则答，无据则拒” |

如果把检索文档看成按顺序拼接的块，则一条正样本通常类似于：

$$
[Q]\;[D_1]\;[D_2]\;[D_3]\;[D_4]\;[D_5]
$$

其中仅有一个文档是 $D^*$，其余都是 $D_k$。  
模型必须在有限上下文内先找到那个真正支持答案的文档，再生成回答。

### 5. RAFT 学到的不只是“答对”，还包括“怎么答”

这是很多初学者会忽略的一点。

如果目标模板是：

- `Answer`
- `Reasoning`
- `Reference`

那么模型学到的就不只是“结论是什么”，还包括：

- 先给结论还是先给解释
- 引用是否必须出现
- 引用放在哪个字段
- 缺乏依据时是否必须显式说明

这意味着 RAFT 实际上同时学习了两层能力：

| 层次 | 学习内容 |
| --- | --- |
| 证据层 | 从候选文档中选出真正支持答案的内容 |
| 输出层 | 按领域要求组织答案、解释和引用 |

例如：

- 医疗问答通常强调结论、适用条件、剂量依据、风险提示
- 法律问答通常强调条文编号、适用范围、例外情况
- 企业知识库问答通常强调制度版本、章节号、生效日期

因此，RAFT 的价值不仅是降低幻觉，还在于把“证据驱动的回答格式”一并学进去。

### 6. 和纯 RAG 的核心差异

用一句话概括：

- 纯 RAG：把文档交给模型，希望它会用
- RAFT：在训练时反复教模型怎么用文档，以及什么时候不要乱用文档

这就是两者效果差异的根本来源。

---

## 代码实现

工程上最关键的部分，不是训练循环本身，而是样本构造。  
因为 loss 没变，真正决定 RAFT 成败的是四件事：

1. 怎么找 oracle 文档
2. 怎么挑难负样本
3. 怎么组织输入模板
4. 怎么定义负样本的目标输出

下面给一个可运行的 Python 玩具实现。它只依赖标准库，不依赖深度学习框架，目的是完整演示 RAFT 样本构造、简单检索、正负样本生成和基本校验逻辑。

```python
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict


random.seed(7)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    supports_answer: bool = False


@dataclass(frozen=True)
class QAItem:
    question: str
    answer: str
    oracle_doc_id: str
    refusal: str


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def vectorize(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(question: str, corpus: List[Document], top_k: int = 5) -> List[Document]:
    qv = vectorize(question)
    scored = []
    for doc in corpus:
        score = cosine(qv, vectorize(doc.text))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_raft_sample(
    qa: QAItem,
    retrieved_docs: List[Document],
    positive_prob: float = 0.8,
    distractor_k: int = 4,
) -> Dict[str, object]:
    assert 0.0 <= positive_prob <= 1.0

    oracle_doc = None
    distractors = []

    for doc in retrieved_docs:
        if doc.doc_id == qa.oracle_doc_id:
            oracle_doc = doc
        else:
            distractors.append(doc)

    assert len(distractors) >= distractor_k, "Not enough distractors in retrieved docs"

    sampled_distractors = random.sample(distractors, distractor_k)
    use_positive = (oracle_doc is not None) and (random.random() < positive_prob)

    if use_positive:
        docs = [oracle_doc] + sampled_distractors
        random.shuffle(docs)
        target = (
            f"Answer: {qa.answer}\n"
            f"Reasoning: The answer is directly supported by the retrieved evidence.\n"
            f"Reference: {qa.oracle_doc_id}"
        )
        label = "positive"
    else:
        docs = sampled_distractors
        target = (
            f"Answer: {qa.refusal}\n"
            "Reasoning: None of the provided documents contains sufficient support.\n"
            "Reference: NONE"
        )
        label = "negative"

    prompt_lines = [f"Question: {qa.question}", "Documents:"]
    prompt_lines.extend([f"- [{doc.doc_id}] {doc.text}" for doc in docs])

    return {
        "label": label,
        "prompt": "\n".join(prompt_lines),
        "target": target,
        "doc_ids": [doc.doc_id for doc in docs],
    }


def main() -> None:
    corpus = [
        Document(
            "DOC_ORACLE",
            "drug x mild renal impairment no dose adjustment required section 4_2 dosage",
            supports_answer=True,
        ),
        Document(
            "DOC_A",
            "drug x pediatric dosage for children under twelve years old",
        ),
        Document(
            "DOC_B",
            "drug x liver toxicity warning and hepatic monitoring recommendations",
        ),
        Document(
            "DOC_C",
            "drug x storage temperature and packaging instructions",
        ),
        Document(
            "DOC_D",
            "drug x anticoagulant interaction and bleeding risk",
        ),
        Document(
            "DOC_E",
            "drug x severe allergy contraindication and emergency treatment",
        ),
        Document(
            "DOC_F",
            "drug x pregnancy and lactation precautions",
        ),
    ]

    qa = QAItem(
        question="Does drug X require dose adjustment for mild renal impairment?",
        answer="No dose adjustment is required for mild renal impairment.",
        oracle_doc_id="DOC_ORACLE",
        refusal="Insufficient evidence from retrieved documents.",
    )

    retrieved = retrieve(qa.question, corpus, top_k=6)
    assert any(doc.doc_id == "DOC_ORACLE" for doc in retrieved)

    samples = [
        build_raft_sample(
            qa=qa,
            retrieved_docs=retrieved,
            positive_prob=0.8,
            distractor_k=4,
        )
        for _ in range(200)
    ]

    pos = sum(1 for sample in samples if sample["label"] == "positive")
    neg = sum(1 for sample in samples if sample["label"] == "negative")

    assert pos > neg
    assert 130 <= pos <= 190
    assert all(
        len(sample["doc_ids"]) == 5 for sample in samples if sample["label"] == "positive"
    )
    assert all(
        len(sample["doc_ids"]) == 4 for sample in samples if sample["label"] == "negative"
    )
    assert any(
        "Reference: NONE" in sample["target"]
        for sample in samples
        if sample["label"] == "negative"
    )

    one_positive = next(sample for sample in samples if sample["label"] == "positive")
    one_negative = next(sample for sample in samples if sample["label"] == "negative")

    print("positive_samples =", pos)
    print("negative_samples =", neg)
    print()
    print("=== POSITIVE SAMPLE ===")
    print(one_positive["prompt"])
    print(one_positive["target"])
    print()
    print("=== NEGATIVE SAMPLE ===")
    print(one_negative["prompt"])
    print(one_negative["target"])


if __name__ == "__main__":
    main()
```

这段代码对应的真实工程动作如下：

| 代码部分 | 真实系统中的对应环节 | 作用 |
| --- | --- | --- |
| `retrieve()` | Retriever 检索器 | 从候选库中返回 top-k 文档 |
| `oracle_doc_id` | Oracle 标注 | 标记哪份文档真正支持答案 |
| `sampled_distractors` | 难负样本选择 | 构造“看起来相关但不支持”的干扰文档 |
| `build_raft_sample()` | 样本模板构造 | 生成训练输入和监督目标 |
| `Reference: NONE` | 负样本目标 | 显式教模型在无证据时拒答 |

如果迁移到真实训练流程，逻辑通常可以写成：

```python
for qa in dataset:
    retrieved = retriever.search(qa.question, top_k=20)

    oracle = match_oracle_doc(qa, retrieved)
    distractors = select_distractors(
        retrieved_docs=retrieved,
        oracle_doc=oracle,
        k=4,
    )

    if oracle is not None and random.random() < P:
        context = shuffle([oracle] + distractors)
        target = format_answer(
            answer=qa.answer,
            reasoning=qa.reasoning,
            reference=oracle.doc_id,
        )
    else:
        context = distractors
        target = format_refusal(
            message="The retrieved documents do not provide sufficient evidence."
        )

    model.train_step(
        input_text=render_prompt(question=qa.question, context=context),
        target_text=target,
    )
```

### 目标模板为什么要固定

对 RAFT 而言，模板固定通常比模板花哨更重要。一个简洁而稳定的模板常见如下：

| 字段 | 作用 | 为什么重要 |
| --- | --- | --- |
| `Answer` | 最终结论 | 便于直接评估答对率 |
| `Reasoning` | 说明结论如何由证据推出 | 便于检查模型是否真的在读文档 |
| `Reference` | 文档编号、章节号、段落号 | 便于评估引用命中率 |

如果模板经常变化，模型很容易学到多套输出风格，结果是：

- 引用字段缺失
- 拒答格式漂移
- 评估脚本难以稳定解析

### 一个真实工程例子

企业内部运维知识库常见的问题是：  
`跨地域容灾切换的 RTO 是多少？`

你可以构造：

- oracle：`容灾规范 v3.2` 第 5.1 节
- distractor：
  - 监控告警规则
  - 值班制度
  - 密码轮换规范
  - 备份保留周期说明

正样本要求模型输出：

- 结论：RTO 数值
- 原因：来自哪条规范
- 引用：规范版本和章节号

负样本则只给干扰文档，要求模型明确说明“当前检索结果没有覆盖 RTO 定义”。  
这比直接把答案背进参数更接近真实线上环境。

### 一个实用起始配置

| 参数 | 常见起点 | 说明 |
| --- | --- | --- |
| 正样本比例 `P` | 0.8 | 先保证模型经常看到真证据 |
| 干扰文档数 `k` | 2 到 4 | 先控制噪声难度，不要一开始过高 |
| 检索候选 `top_k` | 10 到 20 | 便于从中选 oracle 和 distractor |
| 输出模板 | `Answer + Reasoning + Reference` | 便于训练和评估 |
| 负样本目标 | 拒答或证据不足说明 | 必须显式写进监督数据 |

---

## 工程权衡与常见坑

RAFT 的主要收益，是提升“证据选择能力”和“拒绝噪声能力”；它的主要成本，是数据准备复杂度、训练成本和评估复杂度都明显上升。

最常见的问题往往不在模型结构，而在样本分布设计。

| 常见坑 | 后果 | 规避策略 |
| --- | --- | --- |
| 只有 oracle，没有负样本 | 模型默认“给了文档就一定能答” | 必须加入纯 distractor 样本 |
| distractor 太弱 | 训练过于简单，线上泛化差 | 选语义接近但不支持问题的文档 |
| 负样本全是完全无关文档 | 模型学会“看到明显不相关就拒答”，但学不会难判断场景 | 优先构造伪相关负样本 |
| `P` 太高 | 拒答能力不足，容易胡引用 | 保留 10% 到 30% 负样本 |
| `P` 太低 | 模型很少看到正证据，训练不稳定 | 从 0.8 左右起步 |
| 模板不固定 | 输出格式漂移，难以自动评估 | 固定 `Answer/Reasoning/Reference` |
| oracle 标注不准 | 训练信号本身错误 | 先做人审和小规模抽检 |
| 检索器召回低 | RAFT 样本构造依赖的 top-k 本身就是错的 | 先验证 retriever 的 Recall@k |

### 1. 难负样本比“随便负样本”更重要

这是 RAFT 中最容易被低估的一点。

如果负样本只是“问药品剂量，却给一篇足球新闻”，模型当然会拒答，但这种训练价值很低。  
真正有价值的负样本应当满足：

- 主题相关
- 词面相近
- 看起来像能回答
- 实际上不能支持结论

以药品问答为例，更好的负样本包括：

- 儿童剂量说明
- 肝功能异常警示
- 联合用药风险
- 孕期用药说明

它们都和“药物使用”高度相关，但仍不能回答“轻度肾功能不全是否要调剂量”。

### 2. 评估不能只看答案正确率

如果只看最终答案是否“像是对的”，会漏掉很多关键错误。  
RAFT 至少应同时评估三类指标：

| 指标 | 一个可操作定义 | 工程意义 |
| --- | --- | --- |
| Answer Accuracy | 最终结论是否正确 | 基础正确性 |
| Citation Hit Rate | 引用是否命中 oracle 文档 | 检查是否真的用对证据 |
| Refusal Accuracy | 无证据时是否正确拒答 | 检查模型克制能力 |

其中引用命中率可以写成：

$$
\text{Citation Hit Rate}
=
\frac{\#\{\text{引用命中 oracle 的样本}\}}
{\#\{\text{需要引用的样本}\}}
$$

拒答准确率可以写成：

$$
\text{Refusal Accuracy}
=
\frac{\#\{\text{负样本上正确拒答}\}}
{\#\{\text{负样本总数}\}}
$$

如果一个模型答对率很高，但引用命中率很低，通常说明它仍在“凭感觉答题”，而不是“按证据答题”。

### 3. 成本要单独核算

与纯 RAG 相比，RAFT 多出来的成本主要有三类：

| 成本项 | 纯 RAG | RAFT |
| --- | --- | --- |
| 检索系统建设 | 需要 | 需要 |
| Oracle / distractor 标注 | 通常不需要 | 必须重点投入 |
| 微调训练 | 可选 | 通常需要 |
| 评估体系 | 可较简单 | 必须包含引用与拒答评估 |

因此，RAFT 并不是“总是更好”，而是“在错误引用代价高时更值”。  
医疗、法律、金融合规、企业制度问答，通常都属于这种情况。

### 4. 一个常见错误认知

很多团队会说：“我们已经做了 RAG，所以模型应该知道怎么引用。”  
这个推理并不成立。

RAG 只保证了“文档被送进上下文”；  
RAFT 才是在训练中显式优化“如何使用这些文档”。  
这两者不是一回事。

---

## 替代方案与适用边界

如果你的系统目标只是“尽量回答对”，而不是“稳定带依据回答，并在无依据时拒答”，那就不一定要上 RAFT。

最常见的替代方案有三类。

| 方案 | 最适合的问题 | 主要优点 | 主要缺点 |
| --- | --- | --- | --- |
| 纯 RAG | 文档更新快、上线周期短、引用要求一般 | 实时性高、训练成本低 | 模型未必会稳定使用证据 |
| 纯 SFT | 模板稳定、知识变化慢、任务格式固定 | 实现简单、输出稳定 | 容易把答案背进参数，更新慢 |
| 负采样增强 SFT | 重点是提升拒答能力 | 比完整 RAFT 轻量 | 不一定能学到完整的证据引用行为 |
| RAFT | 高风险专业问答、强依据需求 | 引用稳定性和拒答能力更强 | 数据准备和训练成本最高 |

### 1. 纯 RAG 什么时候就够了

如果你的场景是：

- 产品帮助中心
- 公开文档问答
- 一般 FAQ
- 普通客服知识查询

那么先把下面几件事做好，通常比直接上 RAFT 更划算：

1. 提升检索召回率
2. 优化 chunk 切分
3. 增加 reranker
4. 约束 prompt 中的引用格式

这些改动的成本通常比完整 RAFT 低很多，而且对很多中低风险场景已经足够。

### 2. 纯 SFT 什么时候更合适

如果任务的核心不是“依据检索文档作答”，而是“输出风格和格式稳定”，那么纯 SFT 往往更直接。例如：

- 客服固定话术
- 公司内部代码风格助手
- 标准格式摘要
- 固定模板邮件生成

这类任务并不强调“引用外部文档作为证据”，因此 RAFT 的额外训练分布设计不一定带来明显收益。

### 3. RAFT 最适合什么条件同时成立

可以用下面这张表做快速判断：

| 条件 | 是否关键 | 说明 |
| --- | --- | --- |
| 有检索系统 | 是 | 没有检索就没有 RAFT 的上下文来源 |
| 能标注 oracle 文档 | 是 | 否则无法构造正样本 |
| 错误引用代价高 | 是 | 这是 RAFT 最值得投入的前提 |
| 需要输出依据 | 是 | 不要求依据时，RAFT 的价值会下降 |
| 知识频繁更新 | 常常是加分项 | 说明不能只靠参数记忆 |

只有当这些条件大体成立时，RAFT 才通常是合理优先级。

### 4. 一个边界很清楚的医疗例子

如果系统只是生成“常见用药注意事项”，那么：

- 纯 SFT 可能够用
- 纯 RAG 也可能够用

但如果系统需要回答：

- `这个结论依据哪篇指南？`
- `是药品说明书第几节？`
- `当前检索结果是否足以支持剂量建议？`

那么系统需要的就不只是“答得像”，而是：

- 选对证据
- 给出引用
- 无证据时闭嘴

这正是 RAFT 的核心适用边界。

一句话总结：  
如果任务关键在“会查资料”，优先把 RAG 做好；  
如果任务关键在“会按证据回答”，RAFT 才是更合适的方法。

---

## 参考资料

1. Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG*, 2024。  
核心贡献：提出把“oracle 文档 + 干扰文档”直接写入监督微调样本，让模型在训练阶段学习证据抽取与无证据拒答。  
建议阅读方式：先看任务定义和数据构造部分，再看实验里与纯 RAG、纯 SFT 的对比。

2. 论文 PDF: https://shishirpatil.github.io/publications/raft-2024.pdf  
核心价值：适合直接查看样本构造、训练设定、实验表格和附录细节。  
建议阅读方式：先读 Figure 和实验设置，再回头看方法章节，理解正负样本比例和 distractor 设计。

3. CatalyzeX 论文页面: https://www.catalyzex.com/paper/raft-adapting-language-model-to-domain  
核心价值：便于快速获取论文摘要、代码线索和相关实现讨论。  
建议阅读方式：用于论文导航，不建议替代原文。

4. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020。  
核心贡献：系统化提出 RAG，把检索分布与生成分布结合起来，是理解“为什么仅在推理时拼接文档仍然不够”的重要背景材料。  
建议阅读方式：重点看 RAG-Sequence 与 RAG-Token 的区别，再回头理解 RAFT 为什么要把检索上下文前移到训练阶段。

5. 企业知识库或领域 QA 的内部评估实践资料。  
核心价值：RAFT 成败高度依赖工程细节，尤其是 oracle 标注质量、难负样本构造和拒答指标设计。  
建议阅读方式：不要只看最终答对率，必须同时检查引用命中率与负样本拒答准确率。

## 核心结论

RAG 的事实性与忠实度评测，核心不是判断答案“像不像对”，而是判断答案里的每个命题，是否真的有证据支撑。

`factuality` 可以直译为“事实性”，意思是答案是否符合一个更可靠、更新、被当作真值的外部证据集。`faithfulness` 可以直译为“忠实度”，意思是答案是否忠实于当前检索到的上下文，没有越界编造。两者不是一回事。

对零基础读者，最重要的一句白话是：答案看起来很合理，不等于它真的被证据支持；检索到了相关材料，也不等于模型不会顺手补出材料里没有的话。

RAG 评测至少要拆成两层看：

| 对象 | 关注问题 | 典型输入 | 常见指标 |
|---|---|---|---|
| `retrieval` 检索 | 该找的证据有没有找回来 | `q, R_K, E*` | `Recall@K`、Context Recall |
| `generation` 生成 | 回答有没有乱编、漏答、答偏 | `q, C, y, E*` | Faithfulness、Factuality、Answer Relevance |
| `faithfulness` 忠实度 | `y` 是否被当前 `C` 支持 | `C, y` | claim-level support |
| `factuality` 事实性 | `y` 是否符合更可靠的 `E*` | `E*, y` | atomic fact support |

结论也可以压缩成一句工程判断：RAG 的评测要把“证据有没有取对”和“模型有没有超出证据”分开算，否则分数高低没有诊断价值。

---

## 问题定义与边界

先把对象定义清楚。这里用一套最常见、最方便工程实现的记号：

| 记号 | 定义 | 白话解释 |
|---|---|---|
| `q` | 用户问题 | 用户到底问了什么 |
| `C` | 当前检索上下文 | 这次 RAG 真正喂给模型的材料 |
| `E*` | 参考证据集 | 你认为更可靠的“标准答案证据” |
| `y` | 模型回答 | 最终输出给用户的文字 |
| `c_i` | 第 `i` 条原子命题 | 把一句长回答拆成最小可核验判断 |

“原子命题”这个术语第一次看会有点抽象。白话说，它就是一条可以单独判断真假的最小陈述，比如“首届超级碗在 1967 年举行”就是一条原子命题；“首届超级碗在 1967 年举行，地点在洛杉矶，而且由 AFL 和 NFL 冠军参加”其实已经混了三条判断。

边界必须先说清楚，否则很容易把指标用错。

第一，`faithfulness` 只回答“答案有没有超出当前上下文”，不回答“当前上下文本身是不是最新、最真”。如果知识库里是一份过期政策，模型逐字忠实复述旧政策，这个答案可能 `faithful`，但并不 `factual`。

第二，`factuality` 只回答“答案是否符合你选定的参考证据”，不自动保证“业务上就可用”。例如客服系统要求答案同时满足“事实正确”和“语气合规”，事实性高也不代表合规性高。

第三，一些框架会把 `faithfulness` 叫作 `groundedness`，本质都在问“回答是否扎根于证据”。术语名称可能不同，但判断对象基本一致。

下面这张表是常见误用边界：

| 指标 | 不能单独回答的问题 | 典型误用 |
|---|---|---|
| Faithfulness | 外部世界是否真实正确 | 把“上下文支持”误当成“事实正确” |
| Factuality | 检索模块是否工作良好 | 只看最终正确率，忽略检索缺失 |
| Recall@K | 模型有没有幻觉扩写 | 以为“证据找到了”就等于“答案不会乱编” |
| Answer Relevance | 回答是否被证据支持 | 只看答题相关性，不看证据基础 |

---

## 核心机制与推导

最常见、也最稳妥的做法，是把回答 `y` 拆成一组原子命题 $c_1, c_2, \dots, c_m$，然后逐条判断支持关系。

如果上下文 `C` 能推出命题 `c_i`，记为 $C \vDash c_i$；如果参考证据 `E*` 能推出命题 `c_i`，记为 $E^* \vDash c_i$。于是可以得到两个核心分数：

$$
Faithfulness(C, y) = \frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[C \vDash c_i]
$$

$$
Factuality(E^*, y) = \frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[E^* \vDash c_i]
$$

这里的 $\mathbf{1}[\cdot]$ 是指示函数。白话说，成立记 1，不成立记 0，最后求平均。

检索侧至少还要配一个召回指标，例如：

$$
Recall@K = \frac{|E^* \cap R_K|}{|E^*|}
$$

其中 `R_K` 是前 `K` 个检索结果。这个公式的含义很直接：标准证据里该找回来的内容，前 `K` 条结果里到底找回了多少。

一个玩具例子最容易看清机制。

问题：`q = “首届超级碗何时举行，在哪里举行？”`

参考证据 `E*`：
1. 首届超级碗于 1967-01-15 举行。
2. 比赛地点是 Los Angeles Memorial Coliseum。

模型回答 `y`：
“首届超级碗在 1967 年 1 月 15 日举行，地点在洛杉矶，并且是 NFL 单独举办的赛事。”

拆成原子命题后：

| 原子命题 | `C` 支持？ | `E*` 支持？ | 说明 |
|---|---|---|---|
| 首届超级碗于 1967-01-15 举行 | 1 | 1 | 时间正确 |
| 地点在洛杉矶 | 1 | 1 | 地点正确 |
| 它是 NFL 单独举办的赛事 | 0 | 0 | 超出证据，属于幻觉扩写 |

所以两个分数都先得到 $2/3$。如果这第三条在外部真相里其实成立，但当前检索没找回来，那么 `factuality` 可能是 1，而 `faithfulness` 是 $2/3$。这正是两者必须分开的原因。

再看一个真实工程例子。企业内网知识库中，员工问“差旅报销上限是多少”。检索到的是 2024 年政策，里面写“国内住宿上限 500 元”；但 2025 年最新版政策已经改成 650 元。模型忠实复述“500 元”。这时：

- 对当前上下文看，回答是 `faithful`
- 对最新版政策真值看，回答不是 `factual`
- 如果系统监控里只有忠实度，你会误以为系统没有问题
- 真正的问题其实是知识库版本过期，属于检索数据治理问题

这就是为什么 RAG 评测不能只算一个总分。

---

## 代码实现

工程上可以把实现拆成四步：

1. 把回答切成原子命题。
2. 为每条命题收集可判定的证据片段。
3. 判断“是否支持”。
4. 聚合成回答级分数，并和检索指标一起报表。

下面给一个最小可运行的 Python 玩具实现。它不是生产级 NLI，只是演示流程。为了让代码可跑，支持判定先用“标准化后做子串匹配”代替。

```python
import re
from dataclasses import dataclass

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[，。、“”‘’：:；;,.()（）\s]+", " ", text)
    return text.strip()

def split_into_atomic_claims(answer: str):
    parts = re.split(r"[；;。.!?\n]+", answer)
    claims = [p.strip() for p in parts if p.strip()]
    return claims

def supports(evidence_list, claim: str) -> bool:
    c = normalize(claim)
    return any(c in normalize(ev) for ev in evidence_list)

def score_claims(evidence_list, answer: str):
    claims = split_into_atomic_claims(answer)
    flags = [1 if supports(evidence_list, c) else 0 for c in claims]
    score = sum(flags) / len(claims) if claims else 0.0
    return claims, flags, score

def recall_at_k(reference_evidence, retrieved_evidence):
    ref = {normalize(x) for x in reference_evidence}
    ret = {normalize(x) for x in retrieved_evidence}
    return len(ref & ret) / len(ref) if ref else 0.0

reference_evidence = [
    "首届超级碗于 1967-01-15 举行",
    "比赛地点是 Los Angeles Memorial Coliseum",
    "该比赛由 AFL 与 NFL 冠军参与"
]

retrieved_context = [
    "首届超级碗于 1967-01-15 举行",
    "比赛地点是 Los Angeles Memorial Coliseum"
]

answer = (
    "首届超级碗于 1967-01-15 举行；"
    "比赛地点是 Los Angeles Memorial Coliseum；"
    "该比赛由 AFL 与 NFL 冠军参与"
)

claims, faith_flags, faithfulness = score_claims(retrieved_context, answer)
_, fact_flags, factuality = score_claims(reference_evidence, answer)
retrieval_recall = recall_at_k(reference_evidence, retrieved_context)

assert claims == [
    "首届超级碗于 1967-01-15 举行",
    "比赛地点是 Los Angeles Memorial Coliseum",
    "该比赛由 AFL 与 NFL 冠军参与"
]
assert faith_flags == [1, 1, 0]
assert abs(faithfulness - 2/3) < 1e-9
assert fact_flags == [1, 1, 1]
assert factuality == 1.0
assert abs(retrieval_recall - 2/3) < 1e-9

print({
    "faithfulness": faithfulness,
    "factuality": factuality,
    "recall_at_k": retrieval_recall
})
```

这段代码表达了一个关键诊断：如果 `factuality = 1.0`、`faithfulness = 2/3`、`Recall@K = 2/3`，问题主要不在“答案世界知识错了”，而在“参考证据中有一部分没被检索回来，所以模型在当前上下文视角下无法被判为忠实”。

生产环境一般不会用子串匹配，而会用两类判定器：

| 判定器 | 白话解释 | 优点 | 缺点 |
|---|---|---|---|
| LLM Judge | 让另一个模型当裁判 | 灵活、覆盖长文本和复杂语义 | 成本高，结果有抖动 |
| NLI 模型 | 自然语言推断模型，专门判断“能否推出” | 稳定、便宜、快 | 对领域术语和长上下文可能不够强 |

无论选哪种，工程上都要固定提示词、缓存 judge 结果、保存命题与证据对、记录模型版本。否则今天 0.82、明天 0.77，你很难知道是系统退化，还是裁判变了。

---

## 工程权衡与常见坑

RAG 评测里最危险的事，是拿一个单一分数做产品决策。高忠实度不代表知识库没过期，高召回也不代表答案没扩写，高相关性更不代表事实正确。

常见坑可以直接列成表：

| 坑位 | 后果 | 规避方式 |
|---|---|---|
| `chunk` 粒度过粗或过细 | 过粗时噪声大，过细时证据断裂 | 基于句群或段落做切分，并做离线抽检 |
| 知识库版本过期 | 忠实度高但事实性低 | 给文档加版本、生效日期、失效日期 |
| 只看单指标 | 无法定位是检索错还是生成错 | 至少同时看 `Recall@K + Faithfulness + Factuality` |
| judge 抖动 | 分数不稳定，A/B 结论不可信 | 固定 judge、提示词、温度和缓存 |
| 只评单轮问答 | 多轮系统上线后表现失真 | 覆盖追问、改写、省略指代场景 |
| 命题拆分过粗 | 一条错半句却被整体判真 | 尽量拆成最小可核验事实 |
| 只存总分不存证据 | 无法复盘和修 bug | 每条 claim 保存支持片段和判定理由 |

真实工程里，最常见的错位是：团队发现用户投诉“回答不准”，于是调大 `top_k`。结果召回上去了，但噪声也上去了，模型反而更容易把不相关内容缝进答案。这个时候如果你只盯 `Recall@K`，会以为系统在变好；如果同时看 `faithfulness`，可能会发现它在下降。

另一个高频坑是把“无法从上下文推出”直接当成“错误”。严格说，这只是“当前证据不足”。模型可能恰好说对了，但你这次没取到证据；也可能模型说错了。前者影响 `faithfulness`，后者影响 `factuality`。这两类问题的修法完全不同。

---

## 替代方案与适用边界

不是所有任务都需要细粒度事实核验。封闭答案、短文本、答案形式固定的场景，简单指标就够；开放问答、企业知识库、长答案、多跳推理场景，必须上“检索 + 生成 + 人审抽检”的组合。

常见替代方案对比如下：

| 方法 | 适用场景 | 优点 | 缺点 | 是否需要标注 | 能否定位检索/生成错误 |
|---|---|---|---|---|---|
| Exact Match | 选择题、短答案、标准答案唯一 | 简单直接 | 对同义表达不友好 | 需要 | 否 |
| ROUGE / BLEU | 摘要、改写、模板化输出 | 便宜，批量快 | 和事实支持弱相关 | 需要参考答案 | 否 |
| 人工评审 | 高风险业务、上线验收 | 最可靠 | 慢且贵 | 需要人力 | 可以，但成本高 |
| RAGAS | 通用 RAG 快速评测 | 接入快，工程友好 | 指标解释需谨慎 | 可弱标注 | 部分可以 |
| TruLens RAG Triad | 诊断上下文相关、扎根性、答案相关性 | 结构清晰 | 更偏框架化评估 | 可弱标注 | 部分可以 |
| RAGChecker | 细粒度诊断 RAG | 能拆检索与生成，诊断强 | 流程更复杂 | 通常不必强标注 | 可以 |
| FActScore | 长文本事实核验 | 原子事实视角清晰 | 更偏长文本 factuality | 依赖知识源 | 不专门定位检索 |

对初级工程师，一个实用判断规则是：

- 如果任务像考试判卷，答案短且标准，先用 `Exact Match`
- 如果任务像知识库问答，优先看证据支持链
- 如果任务输出长答案，必须拆原子命题
- 如果业务有合规或高风险要求，自动分数之外一定要加人工抽检

换句话说，RAG 不是“有检索就更真”，而是“你终于有机会把错误拆解并归因”。评测体系的价值就在这里。

---

## 参考资料

| 标题 | 来源 | 用途 |
|---|---|---|
| Faithfulness | Ragas 文档 | 了解忠实度定义与计算方式 |
| The RAG Triad | TruLens 文档 | 理解检索、扎根性、答案相关性的三角诊断 |
| FActScore | arXiv 论文 | 学习原子事实级事实性评测 |
| RAGChecker | Amazon Science / arXiv | 学习如何拆分检索与生成诊断 |

1. [Faithfulness - Ragas Documentation](https://docs.ragas.io/en/v0.3.3/concepts/metrics/available_metrics/faithfulness/)
2. [The RAG Triad - TruLens Documentation](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
3. [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251)
4. [RAGChecker: A fine-grained framework for diagnosing retrieval-augmented generation](https://www.amazon.science/publications/ragchecker-a-fine-grained-framework-for-diagnosing-retrieval-augmented-generation)
5. [RAGChecker GitHub Repository](https://github.com/amazon-science/RAGChecker)

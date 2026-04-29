## 核心结论

RAG 是“检索增强生成”（先找资料，再让模型作答）。它真正难的地方，不是把几个链接挂到答案后面，而是让答案里的每个关键结论都能被某段原始证据直接支持，并且别人能按位置复核出来。

这里要区分四个容易混在一起的概念：链接、引用、对齐、可追溯。链接只是“来源大概在这里”；引用是“答案声称用了这里”；对齐是“答案这句话和这段证据在语义跨度上匹配”；可追溯则更严格，要求“结论、位置、支持关系”三件事同时成立。缺任何一个，系统都可能制造“看起来有出处”的伪证据感。

一个玩具例子最容易看清这个区别。答案说：“公司报销必须提供电子发票。” 如果它给出的证据原文其实是“建议优先提供电子发票”，那就不是可追溯。因为“建议”不能推出“必须”。链接是真的，引用位置也许也是真的，但支持关系是假的。

| 概念 | 关注点 | 是否可复核 | 是否足以支撑结论 |
|---|---|---:|---:|
| 链接 | 文档来源 | 否 | 否 |
| 引用 | 引到某个位置 | 部分 | 不一定 |
| 对齐 | 结论与证据跨度匹配 | 是 | 可能 |
| 可追溯 | 结论、位置、支持关系都成立 | 是 | 是 |

因此，RAG 引用对齐与可追溯性的核心不是“加引用样式”，而是建立一条可检查的证据链：答案中的关键结论可以拆开，拆开后每一条都能找到准确证据，且证据确实支持这条结论。

---

## 问题定义与边界

这篇文章讨论的是“结论级可追溯性”。意思是：把答案看成若干条独立结论，逐条检查其证据来源。它不等于检索命中率，也不等于“整体看起来像真话”。

需要分清三个层次。

第一层是检索相关。相关的意思是“这段材料和问题有关系”。比如问“巴黎是不是法国首都”，检索到了 “France is in Western Europe.”，这段材料和法国有关，所以相关。

第二层是语义支持。支持的意思是“这段材料足以推出这条结论”。刚才那句 “France is in Western Europe.” 并不能推出“巴黎是法国首都”，所以它相关但不支持。

第三层是位置可定位。可定位的意思是“别人能准确找到你依赖的是哪一句、哪一段、哪个版本”。如果系统只保存网页 URL，不保存段落号、chunk 编号、偏移量，那么文档一更新，原答案即使没变，也可能失去复核能力。

| 层次 | 问题 | 典型指标 | 常见误区 |
|---|---|---|---|
| 检索 | 有没有找对材料 | Recall / Precision | 只看召回 |
| 支持 | 材料是否真的支撑结论 | Entailment / Faithfulness | 把相关当支持 |
| 对齐 | 引用位置是否对得上 | Span-level citation accuracy | 只给 URL |
| 可追溯 | 人能否复核 | 可定位引用 | 没有版本号、段落号 |

边界也要说清楚。本文不讨论“模型是否知道世界知识”，而讨论“模型这次回答是否有证据可查”。即使模型说对了，如果它给出的引用位置不对，或者证据并不支持结论，那仍然不是一个可信的 RAG 系统。

真实工程里，这个边界尤其重要。企业知识库问答场景中，用户问“报销是否必须提供电子发票”。如果系统只是给出“是，必须提供”，再附一个制度 PDF 链接，这仍然不够。因为真正需要的是：哪一版制度、哪一节、哪一句，是否真的写了“必须”。

---

## 核心机制与推导

要把可追溯性做实，第一步不是优化排版，而是把答案拆成原子结论。原子结论就是“不能再拆而仍保持明确真值”的最小结论单元。比如“法国位于西欧，巴黎是法国首都”其实包含两条原子结论，必须分开验证。

形式化地写，可以把系统中的对象定义为：

$$
P = \{p_1,\dots,p_m\}
$$

表示检索到的证据片段集合；把答案拆成原子结论集合：

$$
C = \{c_1,\dots,c_n\}
$$

对于每条结论 $c_i$，系统声称它对应某个证据跨度 $s_i$，且 $s_i \subset p_{j(i)}$。然后定义：

$$
u_i = 1[c_i \vdash s_i]
$$

这里的 $\vdash$ 可以理解为“被支持”。若证据跨度真的支持结论，则 $u_i=1$，否则为 0。最后把整段答案的可追溯性写成：

$$
T = \frac{1}{n}\sum_{i=1}^{n}u_i
$$

这个公式的意义很直接：可追溯性不是“整段答案一次性判真伪”，而是逐条结论算分，再求平均。

玩具例子如下。

检索到两段证据：

- `p1`: “Paris is the capital of France.”
- `p2`: “France is in Western Europe.”

答案是：“France is in Western Europe, and Paris is the capital of France.”

拆成两条结论后：

| 结论 | 引用证据 | 是否支持 | `u_i` |
|---|---|---:|---:|
| France is in Western Europe. | `p2` | 是 | 1 |
| Paris is the capital of France. | `p2` | 否 | 0 |

于是：

$$
T=\frac{1+0}{2}=0.5
$$

如果第二条改引 `p1`，则 $T=1.0$。这个例子说明，一段看起来“整体大差不差”的答案，实际上可能只有一半结论可追溯。

从流水线角度，完整过程应当是：

1. 检索证据 `P`
2. 生成候选答案 `A`
3. 拆分原子结论 `C`
4. 为每个结论定位证据跨度 `s_i`
5. 判断支持关系 `u_i`
6. 汇总得到整体分数 `T`

这里最容易犯的新手错误，是把“和主题相关”误认为“支持结论”。相关性只能说明材料值得看，不能说明这句话已经被证明。

---

## 代码实现

工程实现的重点，不是给句子后面插 `[1][2][3]` 这样的编号，而是建立“结论拆分、证据定位、支持验证”的闭环。没有这个闭环，引用只是 UI 装饰。

下面先给一个最小可运行的 Python 例子。它不是生产级自然语言推理器，但足够说明核心流程：先拆结论，再找证据，再判断是否被支持。

```python
from dataclasses import dataclass

@dataclass
class Evidence:
    doc_id: str
    version: str
    chunk_id: str
    offset_start: int
    offset_end: int
    text: str
    source_url: str

def split_into_atomic_claims(answer: str):
    parts = [p.strip(" .") for p in answer.split(" and ") if p.strip()]
    return parts

def locate_supporting_span(claim: str, retrieved_chunks: list[Evidence]):
    keywords = {
        "France is in Western Europe": ["France", "Western Europe"],
        "Paris is the capital of France": ["Paris", "capital of France"],
        "报销必须提供电子发票": ["必须", "电子发票"],
    }
    need = keywords.get(claim, claim.split())
    for chunk in retrieved_chunks:
        if all(k in chunk.text for k in need):
            return chunk
    return None

def entailment_check(claim: str, span: Evidence):
    text = span.text
    rules = {
        "France is in Western Europe": "France is in Western Europe" in text,
        "Paris is the capital of France": "Paris is the capital of France" in text,
        "报销必须提供电子发票": ("必须" in text and "电子发票" in text),
    }
    return rules.get(claim, False)

def evaluate_traceability(answer: str, retrieved_chunks: list[Evidence]):
    claims = split_into_atomic_claims(answer)
    scores = []
    for claim in claims:
        span = locate_supporting_span(claim, retrieved_chunks)
        if span is None:
            scores.append(0)
            continue
        supported = entailment_check(claim, span)
        scores.append(1 if supported else 0)
    return sum(scores) / len(scores) if scores else 0

chunks = [
    Evidence("geo", "v1", "p1", 0, 35, "Paris is the capital of France.", "https://example.com/p1"),
    Evidence("geo", "v1", "p2", 36, 70, "France is in Western Europe.", "https://example.com/p2"),
]

score = evaluate_traceability(
    "France is in Western Europe and Paris is the capital of France",
    chunks
)

assert score == 1.0
```

生产系统里，证据键至少要保存下面这些字段，否则后续很难复核：

| 字段 | 作用 |
|---|---|
| `doc_id` | 定位文档 |
| `version` | 防止引用旧版本 |
| `chunk_id` | 定位片段 |
| `offset_start` / `offset_end` | 定位句子或跨度 |
| `text` | 供人工复核 |
| `source_url` | 方便跳转 |

一个更接近真实系统的记录结构可以是：

```json
{
  "claim": "报销必须提供电子发票",
  "evidence": {
    "doc_id": "expense_policy_2025",
    "version": "v3.2",
    "chunk_id": "sec-4-2",
    "offset_start": 128,
    "offset_end": 164,
    "text": "报销需提交符合要求的电子发票。",
    "source_url": "https://example.com/policy"
  },
  "supported": true
}
```

这里要注意，只有 `doc_url` 没有意义。因为 URL 只能说明“来自哪个页面”，不能说明“来自页面的哪一句”。一旦页面内容更新，表面上引用还在，实际已经失去可追溯性。

真实工程例子是企业制度问答。用户提问：“差旅报销是否必须提供电子发票？” 一个合格系统至少要做到三件事：返回结论；绑定到制度 `v3.2` 的具体条款；允许前端高亮出原句。这样审计、财务、员工三方都能复核。

---

## 工程权衡与常见坑

真正难做的地方，不在“有没有引用”，而在“引用粒度”和“支持判断成本”之间的平衡。粒度太粗，容易失真；粒度太细，存储、计算、标注和前端展示成本都会上升。

最常见的坑如下：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 只给文档 URL | 无法复核到句子 | 保存 `doc_id + chunk_id + offset` |
| 一条引用支撑多条结论 | 结论对不齐 | 先拆原子结论 |
| 证据只是相关 | 伪证据感 | 做 entailment / faithfulness 检查 |
| 生成时改写过度 | 语义漂移 | 保留短引文或可定位摘要 |
| 引用旧版本 | 表面正确、实际失真 | 证据键纳入版本号 |

其中“过度改写”尤其危险。原文写“建议优先使用电子发票”，模型输出“必须提供电子发票”，这时哪怕引用位置完全正确，答案仍然不可信。因为它把建议性措辞改成了强制性措辞，语义发生了等级跃迁。

可以把这个错误看成一个简单的蕴含关系问题。若原文强度低于结论强度，则通常不能推出：

$$
\text{建议} \not\Rightarrow \text{必须}
$$

再看一个对照：

| 表达 | 能否支撑“必须提供电子发票” | 原因 |
|---|---:|---|
| 报销需提交电子发票。 | 是 | 明确要求 |
| 建议优先使用电子发票。 | 否 | 只是建议 |
| 报销可能需要相关票据。 | 否 | 过于模糊 |

另一个工程坑是文档版本漂移。比如 2025 年制度写“必须”，2026 年制度改成“优先推荐”。如果系统只保存文档标题，不保存版本号，那么旧答案会继续显示“有引用”，但它引用的是过期事实。这类问题不会在 demo 中暴露，却会在正式环境中直接破坏信任。

---

## 替代方案与适用边界

并不是所有系统都要做到句子级强对齐。要做到什么粒度，取决于业务风险、延迟预算、人工审核能力和错误后果。

| 方案 | 特点 | 适用场景 | 不足 |
|---|---|---|---|
| 文档级引用 | 实现简单 | 低风险检索 | 粒度太粗 |
| 段落级引用 | 平衡性较好 | 企业知识库 | 仍可能不够精确 |
| 句子级引用 | 可复核性强 | 高风险领域 | 实现和维护成本高 |
| 人工审核 | 最稳 | 合规场景 | 成本高、吞吐低 |

低风险场景，例如内部知识检索预览、学习型问答、文章推荐，段落级引用通常已经够用。因为这里的目标是“帮助定位材料”，不是“直接作为正式依据”。

高风险场景，例如财务、法务、医疗、合规，就不能只给文档级引用。至少要做到段落级，理想情况下做到句子级或更细粒度，并保留版本号、位置索引和原文高亮。否则系统不应被包装成“可信引用系统”。

还要注意一个现实边界：有些问题天生需要跨段推理甚至多证据合成。例如“某政策是否适用于外包员工”可能要同时依赖定义条款、适用范围条款和例外条款。这时“一条结论对应一条证据”未必够用，需要把结论对应到证据集合而非单一 span。但即使如此，原则不变：每一步推理都应该能回到原始证据，而不是凭生成模型自行补全。

所以，替代方案不是“放弃可追溯”，而是在不同风险等级下调整实现强度。如果做不到强对齐，就应该诚实地把系统定义为“辅助检索”或“摘要生成”，而不是“可审计回答”。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | 说明 RAG 基本范式 |
| Asai et al., *Evidentiality-guided Generation for Knowledge-Intensive NLP Tasks* | 说明证据驱动生成与证据对齐思路 |
| Niu et al., *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models* | 说明可信 RAG 的幻觉评测 |
| Ragas Faithfulness | 说明答案是否忠实于证据 |
| Ragas Context Precision | 说明检索上下文是否真的有用 |

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://discovery.ucl.ac.uk/id/eprint/10100504/)
2. [Evidentiality-guided Generation for Knowledge-Intensive NLP Tasks](https://aclanthology.org/2022.naacl-main.162/)
3. [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/)
4. [Ragas Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
5. [Ragas Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)

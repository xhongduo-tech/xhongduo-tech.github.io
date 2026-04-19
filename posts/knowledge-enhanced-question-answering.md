## 核心结论

知识增强问答是把外部知识库中的相关证据先检索出来，再让模型基于证据回答问题的问答方法。它的核心目标不是让模型“更会聊天”，而是让模型在回答事实问题时更准确、更容易更新、更容易追溯来源。

普通生成模型主要依赖参数记忆。参数记忆指模型在训练过程中把知识压进参数里，回答时不再实时查资料。这种方式对常识问题有效，但对冷门事实、最新政策、企业内部文档容易失效。知识增强问答把“查资料”放到回答前面：先从 Wikipedia、知识图谱、企业 FAQ、政策文档、论文库或工单规范中找证据，再组织答案。

| 对比项 | 纯生成模型 | 知识增强问答 |
|---|---|---|
| 知识来源 | 模型参数 | 外部知识库 + 模型参数 |
| 更新方式 | 通常需要重新训练或微调 | 更新文档和索引即可 |
| 可追溯性 | 较弱，通常没有来源 | 可返回引用段落 |
| 适合问题 | 常识、改写、总结、创作 | 开放域事实问答、企业知识库问答 |
| 主要风险 | 幻觉、过时知识 | 检索失败、证据使用错误 |

玩具例子：问“法国首都是哪座城市？”纯生成模型可能直接从记忆中答“巴黎”。知识增强问答会先检索到“Paris is the capital of France”这样的证据，再回答“巴黎”。

真实工程例子：用户问“某产品退款规则是什么？”系统先查 FAQ、售后政策、工单处理规范，再生成答案并附上来源。如果退款规则改了，只需要更新文档和索引，不必立刻重训大模型。

RAG 是最典型的知识增强问答实现路线，它把检索与生成组合起来。REALM 是更进一步的路线，它把知识检索放进预训练过程，让检索器也能被训练目标影响。

---

## 问题定义与边界

知识增强问答可以形式化为：给定问题 $x$，系统从知识库中检索证据 $z$，再输出答案 $y$。这里的知识库不是固定格式，可以是百科网页、企业文档、论文、FAQ、数据库导出的文本、客服工单规范，也可以是结构化知识图谱。

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $x$ | question | 用户提出的问题 |
| $z$ | evidence document | 被检索出来的证据文本或知识片段 |
| $y$ | answer | 系统最终输出的答案 |
| TopK | top-k evidence | 按相关性排序后取前 k 个证据 |

这个方法主要解决开放域事实问答。开放域事实问答指答案依赖外部世界知识，而且问题范围不固定。例如“法国首都是哪座城市？”“某 API 的退款策略是什么？”“这篇论文提出了什么方法？”都属于这一类。

它不适合所有问题。问“把这段话改写得更正式”时，任务核心是语言改写，不需要外部知识库。问“写一首科幻诗”时，任务核心是创作，检索事实资料也不是必要步骤。

| 任务类型 | 是否适合知识增强问答 | 原因 |
|---|---:|---|
| 百科事实问答 | 适合 | 答案可从外部资料中找到 |
| 企业政策问答 | 适合 | 知识经常更新，需要可追溯 |
| 论文内容问答 | 适合 | 需要依据论文原文 |
| 文本改写 | 不适合 | 主要依赖语言生成能力 |
| 创意写作 | 不适合 | 目标不是事实准确 |
| 复杂数学证明 | 不一定适合 | 检索只能提供资料，不能替代推理 |

边界需要明确：知识增强问答提升的是“基于证据回答事实问题”的能力，不自动保证复杂推理正确。它能把正确资料放到模型面前，但模型仍可能误读证据、忽略限制条件，或者把多个文档中的信息错误拼接。

---

## 核心机制与推导

知识增强问答通常包含两个核心模块：检索器和生成器。检索器负责从知识库中找出与问题最相关的 top-k 证据；生成器负责在问题和证据的条件下输出答案。

检索器会给问题 $x$ 和候选文档 $z$ 一个相关性分数：

$$
s_\eta(x, z)
$$

其中 $\eta$ 表示检索器参数。分数越高，说明文档越可能包含答案。然后用 softmax 把分数转成概率分布：

$$
p_\eta(z|x)=softmax(s_\eta(x,z))
$$

生成器在问题和证据条件下生成答案：

$$
p_\theta(y|x,z)
$$

其中 $\theta$ 表示生成模型参数。RAG 的关键思想是：最终答案不是只由一个文档决定，而是对多个候选证据做加权：

$$
p(y|x) \approx \sum_{z \in TopK(x)} p_\eta(z|x)p_\theta(y|x,z)
$$

这个公式的含义是：一个答案的概率，等于“证据被检索到的概率”乘以“在该证据下生成该答案的概率”，再对 top-k 证据求和。

玩具数值例子：问题是“法国首都是哪座城市？”

| 证据 | $p_\eta(z|x)$ | $p_\theta(y=Paris|x,z)$ |
|---|---:|---:|
| $z_1$：“Paris is the capital of France” | 0.7 | 0.9 |
| $z_2$：“Lyon is a city in France” | 0.3 | 0.1 |

则：

$$
p(Paris|x) \approx 0.7 \times 0.9 + 0.3 \times 0.1 = 0.66
$$

这说明答案质量同时受检索器和生成器影响。检索到好证据但生成器不用，答案会错；生成器很强但证据没找对，也会错。

如果按 token 逐个生成答案，RAG 还可以写成：

$$
p(y|x) \approx \prod_i \sum_{z \in TopK(x)} p_\eta(z|x)p_\theta(y_i|x,z,y_{<i})
$$

$y_i$ 表示第 $i$ 个 token，$y_{<i}$ 表示前面已经生成的 token。token 是模型处理文本的最小片段，可以是一个字、一个词，或者词的一部分。

REALM 的机制更靠近训练阶段。它不是简单地在推理时外接检索器，而是在语言模型预训练时引入检索模块。训练目标会判断检索到的文档是否帮助模型预测文本。如果某些文档能降低语言建模损失，梯度就会推动检索器以后更倾向找这类文档。梯度是训练中用来更新参数的信号，表示当前参数应该往哪个方向调整。

RAG 更像工程组合：已有检索器加已有生成器。REALM 更像端到端学习：让检索器和语言模型在训练目标下共同变好。

---

## 代码实现

一个最小知识增强问答系统可以拆成四步：分块、建索引、检索、生成。分块是把长文档切成较短片段；索引是为了快速查找相关片段而建立的数据结构；检索是按问题找 top-k 片段；生成是把问题和证据拼成 prompt 后交给模型。

最小伪代码如下：

```text
documents = load_documents()
chunks = split_documents(documents)
embeddings = encode(chunks)
index = build_vector_index(embeddings)

question = input()
query_vector = encode(question)
top_chunks = search(index, query_vector, top_k=3)

prompt = build_prompt(question, top_chunks)
answer = generate(prompt)
return answer_with_citations(answer, top_chunks)
```

下面是一个可运行的 Python 玩具实现。它没有调用大模型，而是用词重叠做检索，再用规则模拟“基于证据回答”。真实系统会把 `retrieve` 替换成 BM25、向量检索或混合检索，把 `generate_answer` 替换成生成模型调用。

```python
from collections import Counter
import re

def tokenize(text):
    return re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", text.lower())

def score(query, doc):
    q = Counter(tokenize(query))
    d = Counter(tokenize(doc))
    return sum(q[t] * d[t] for t in q)

def retrieve(question, docs, top_k=2):
    ranked = sorted(docs, key=lambda doc: score(question, doc["text"]), reverse=True)
    return ranked[:top_k]

def generate_answer(question, evidences):
    joined = " ".join(e["text"] for e in evidences)
    if "refund" in question.lower() and "7 days" in joined:
        return {
            "answer": "The product supports refunds within 7 days if it is unused.",
            "citations": [e["id"] for e in evidences if "refund" in e["text"].lower()]
        }
    if "法国首都" in question and "巴黎" in joined:
        return {
            "answer": "法国首都是巴黎。",
            "citations": [e["id"] for e in evidences if "巴黎" in e["text"]]
        }
    return {"answer": "没有找到足够证据回答。", "citations": []}

docs = [
    {"id": "policy-1", "text": "Refund policy: unused products can be refunded within 7 days."},
    {"id": "geo-1", "text": "法国首都是巴黎，巴黎也是法国最大的城市。"},
    {"id": "policy-2", "text": "Shipping usually takes 3 to 5 business days."},
]

evidences = retrieve("What is the refund rule?", docs, top_k=2)
result = generate_answer("What is the refund rule?", evidences)

assert result["answer"] == "The product supports refunds within 7 days if it is unused."
assert "policy-1" in result["citations"]
```

工程中通常会把模块拆开，避免检索和生成强耦合。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 文档切分 | 原始文档 | chunks | 控制证据粒度 |
| 向量化 | chunks 或 query | embedding | 把文本变成可检索向量 |
| 检索器 | query embedding | top-k chunks | 找候选证据 |
| 重排器 | question + chunks | reranked chunks | 提高证据排序质量 |
| 生成器 | question + evidence | answer | 基于证据生成答案 |
| 引用模块 | answer + evidence | citation | 让答案可追溯 |

真实工程例子：企业知识库问答系统会把 FAQ、政策文档、工单说明统一切分入库。用户问“会员产品退款规则是什么？”系统先召回退款相关段落，再生成带引用的答案：

```json
{
  "answer": "会员产品在未使用权益的情况下支持 7 天内退款，超过 7 天需人工审核。",
  "citations": [
    {
      "doc_id": "refund-policy-2026",
      "section": "2.1 退款条件"
    }
  ]
}
```

这种结构让前端能展示答案，也能展示“答案来自哪份文档”。

---

## 工程权衡与常见坑

知识增强问答的核心工程问题不是“模型能不能生成文字”，而是“系统能不能找对证据、用对证据、稳定输出”。多数失败来自检索链路，而不是生成模型本身。

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| 检索召回差 | 答案在库里，但 top-k 没找到 | 使用 `BM25 + dense retriever`，提高召回 |
| chunk 切分不合理 | 太长稀释信号，太短丢上下文 | 按标题、段落、语义边界切分 |
| 索引过时 | 文档更新了，系统仍答旧规则 | 定期重建索引或做增量刷新 |
| 生成器幻觉 | 有证据仍然编造 | 引用约束、拒答阈值、证据一致性校验 |
| 只看最终 EM/F1 | 不知道是检索错还是生成错 | 同时评估 retrieval recall 和 answer quality |
| top-k 太小 | 相关证据覆盖不足 | 先扩召回，再 rerank 压缩噪声 |

EM 是 exact match，表示答案是否和标准答案完全匹配。F1 是衡量预测答案和标准答案词重叠程度的指标。它们能评估最终答案，但不能单独定位失败原因。一个系统回答错，可能是证据没检索到，也可能是证据找到了但生成器没用对。

新手常见误区是直接调大模型参数，忽略文档处理。实际项目中，chunk 大小、标题保留、表格解析、索引刷新、权限过滤都可能决定成败。比如政策文档从“7 天无理由退款”改成“未使用权益 7 天内可退款”，但向量索引没有刷新，系统仍会输出旧规则。这不是模型智力问题，而是数据同步问题。

另一个关键权衡是 top-k。top-k 太小会漏证据，top-k 太大会把无关文本塞进 prompt，增加误导。常见做法是第一阶段扩大召回，例如取 top-50；第二阶段用重排器筛到 top-5；最后只把最有用的证据交给生成器。

权限也是工程边界。企业知识库中不同用户能看的文档不同，检索阶段必须做权限过滤。否则模型可能把用户无权查看的内部信息生成出来。

---

## 替代方案与适用边界

不是所有问答都必须使用知识增强问答。正确选择方案要看事实依赖、更新频率、可追溯要求和成本。

| 方案 | 机制 | 适用场景 | 边界 |
|---|---|---|---|
| 纯生成模型 | 直接用模型回答 | 改写、总结、常识问答、创作 | 容易过时，来源不可追溯 |
| 纯检索 | 只返回相关文档 | 法规、合同、精确查找 | 用户需要自己读材料 |
| RAG | 检索证据后生成答案 | 企业知识库、开放域事实问答 | 依赖检索质量和 prompt 设计 |
| REALM | 检索参与预训练 | 研究型系统、大规模预训练 | 训练成本高，工程复杂 |
| 规则/模板系统 | 按固定规则输出 | 强约束流程、表单、状态查询 | 灵活性差，维护成本高 |

新手版判断方式：如果只是问“这段文字是什么意思”，直接让模型总结通常就够了；如果问“公司最新退款规则是什么”，就应该查知识库再回答。

工程版判断方式：如果业务要求强事实性、可追溯、可更新，知识增强问答更合适。例如客服问答、内部制度查询、研发文档助手、论文问答、合规政策问答，都天然适合 RAG。如果任务要求严格执行固定流程，例如“订单状态只能从数据库字段返回”，规则系统或工具调用可能比 RAG 更可靠。

知识增强问答不适合三类任务。第一类是低事实依赖任务，例如润色、翻译、风格改写。第二类是纯创造性写作，检索会增加不必要复杂度。第三类是强逻辑推演任务，例如复杂证明、严谨规划、代码形式验证。检索能提供材料，但不能保证推理链条正确。

结论是：知识增强问答不是“给所有大模型接一个向量库”，而是在事实知识重要、知识变化频繁、答案需要来源时，用检索证据约束生成。

---

## 参考资料

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)  
   RAG 原论文，主要提出把检索文档作为隐变量，并让生成模型在多个检索证据上边缘化生成答案。

2. [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/pdf/2002.08909)  
   REALM 原论文，主要解决如何把知识检索放入语言模型预训练，使检索器能从训练目标中学习。

3. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906)  
   DPR 原论文，主要介绍用稠密向量检索改进开放域问答中的证据召回。

4. [Hugging Face Transformers RAG Examples](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)  
   RAG 工程示例，适合继续理解检索器、生成器和数据处理如何在代码中组合。

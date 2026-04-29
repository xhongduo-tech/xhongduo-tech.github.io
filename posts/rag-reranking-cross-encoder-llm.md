## 核心结论

RAG 重排序的本质是“检索后精排”。白话说，第一阶段先尽量把可能相关的材料多找回来，第二阶段再把最该看的证据排到前面。它不负责扩大知识覆盖面，而是负责提高证据进入上下文之前的精度。

在工程系统里，重排序的价值不能只看“回答更像对了”。更准确的判断方式是看它是否同时改善了三类结果：第一，排序指标是否上升，比如 `MRR@10`、`nDCG@10`；第二，端到端任务成功率是否上升，比如问题是否真正解决；第三，额外成本是否可接受，比如 `p95 latency`、`token_in + token_out`、人工接管率是否失控。

一个足够实用的总览公式是：

$$
s_i = f(q, d_i)
$$

这里的 $q$ 是查询，$d_i$ 是第 $i$ 个候选文档，$s_i$ 是相关性分数。Cross-Encoder 的作用，就是给每个候选算出这个分数，再按分数排序。

如果把整个 Agent 或 RAG 系统看成一条流水线，成功率更接近下面这个分解：

$$
S \approx R_{plan} \times R_{retrieve} \times R_{rerank} \times R_{tool} \times R_{recover}
$$

其中：
| 环节 | 白话解释 | 主要作用 |
|---|---|---|
| `R_plan` | 规划是否把问题拆对 | 决定后面走哪条路径 |
| `R_retrieve` | 召回是否把候选找回来 | 决定是否“看得到” |
| `R_rerank` | 重排序是否把好证据排前面 | 决定是否“排得准” |
| `R_tool` | 工具调用是否执行成功 | 决定能不能完成外部动作 |
| `R_recover` | 失败后是否能恢复继续 | 决定异常时损失多大 |

玩具例子很直观。用户问“如何处理订单退款失败”。检索阶段先找回 20 段文档，其中既有“退款失败原因说明”，也有“售后总政策”“会员规则”“物流异常说明”。如果不重排，模型可能把泛泛政策当主证据；如果重排后把“退款失败原因说明”和“工单处理流程”放到前 5 段，回答通常会更稳，因为进入上下文的证据更接近问题本身。

---

## 问题定义与边界

本文讨论的是“召回之后的排序问题”。更具体地说，输入是一个查询和一批已经找回来的候选文档，输出是这些候选的优先级顺序。它不讨论向量库怎么建，不讨论 embedding 模型怎么训，也不讨论最终答案怎么生成。

这条边界很重要，因为很多系统把所有收益都算到 rerank 头上，最后得出错误结论。假设检索阶段根本没找回“退款失败的真实原因”这类文档，那么重排序无论多强，也只能把一堆不相关材料重新排队，不能凭空补出正确知识。

下面这个表可以直接划清边界：

| 阶段 | 解决什么 | 不解决什么 |
|---|---|---|
| 召回 | 尽量找回可能相关的候选 | 候选之间谁更相关 |
| 重排序 | 把更相关的候选排前面 | 知识缺失、工具失败 |
| 生成 | 组织自然语言答案 | 证据本身是否可靠 |
| 工具调用 | 执行数据库、日志、工单等动作 | 检索排序问题 |
| 状态持久化 | 保存中间状态并支持恢复 | 证据相关性判断 |

本文重点比较两类 reranker。

| 类型 | 白话解释 | 常见输入 |
|---|---|---|
| Cross-Encoder | 把“问题+单篇文档”一起看，再给一个分数 | `query + one doc` |
| LLM reranker | 让大模型直接比较一组候选，输出排序或偏好 | `query + many docs` |

它们都适用于 RAG、知识库问答、客服检索、运维助手和多工具 Agent，但适用边界不同：Cross-Encoder 更像一个稳定的精排器，LLM reranker 更像一个昂贵但可能更灵活的裁判。

---

## 核心机制与推导

Cross-Encoder 的核心机制是联合编码。联合编码的白话解释是：它不是先分别把问题和文档变成向量再比相似度，而是把两者一起送进模型，让模型直接判断“这段文档对这个问题到底有多相关”。因此它更擅长捕捉细粒度语义关系，比如步骤顺序、否定词、条件限制和术语对齐。

形式上可以写成：

$$
s_i = f(q, d_i)
$$

对每个候选文档单独算一个 $s_i$，然后按 $s_i$ 降序排列。注意这里的代价也很清楚：如果候选有 $K$ 条，就要算 $K$ 次。

LLM reranker 更接近 listwise 思路。listwise 的白话解释是：不是一条一条打分，而是让模型一次看一组候选，然后直接判断整体名次。形式上可以写成：

$$
rank = g(q, d_1, d_2, \dots, d_K)
$$

它的优势是能利用“候选之间的相对比较”。比如两段文档都提到“API 密钥”，但一段是“查看权限说明”，另一段是“重置流程”。在组内比较时，模型可能更容易直接发现“重置流程”才是更贴近用户意图的答案来源。

玩具例子如下。用户问“如何重置 API 密钥”。召回结果有 20 段文档，其中真正相关的只有 3 段，其余是“权限管理”“账号安全”“错误码说明”。

- Cross-Encoder 的做法：分别判断“查询 vs 每段文档”的相关性，得到 20 个分数，再排序。
- LLM reranker 的做法：一次看 5 到 10 段候选，让模型直接输出“最相关的前 5 名”。

这两类方法的关键差异不是“谁更高级”，而是“谁把计算预算花在什么地方”。

从评估角度，排序提升必须用排序指标表达。最常见的是：
- `MRR@10`：第一个正确结果出现得越靠前越好。
- `nDCG@10`：多条相关结果的整体排序质量越高越好。
- `MAP`：平均精度，适合看一批查询上的整体检索排序表现。

但只看排序指标还不够，因为 Agent 场景通常还有工具调用与恢复链路。一个真实工程例子是客服/运维 Agent：先做向量召回，再重排序，再决定是否调用工单系统、日志系统或数据库查询。这里成功率的提升，可能分别来自不同环节：

- 证据命中率上升，来自 `R_rerank`。
- 工具调用成功率上升，来自更好的规划和路由。
- 人工接管率下降，来自 checkpoint、重试和恢复。

如果没有把这些环节拆开看，就容易把所有收益都误记到 reranker 身上。

---

## 代码实现

实现时最重要的原则是把“召回”和“重排序”拆开。重排序只处理候选列表，不负责全库搜索。这样接口清晰，也方便做消融实验。

下面先给一个最小可运行的 Cross-Encoder 风格玩具实现。这里不用真实模型，而是用关键词重合模拟分数，目的是把流程讲清楚。

```python
from dataclasses import dataclass

@dataclass
class Doc:
    doc_id: str
    text: str

def simple_scorer(query: str, doc: str) -> float:
    q_terms = set(query.lower().split())
    d_terms = set(doc.lower().split())
    overlap = len(q_terms & d_terms)
    return overlap / max(len(q_terms), 1)

def rerank(query: str, candidates: list[Doc], top_k: int = 3):
    scored = [(doc, simple_scorer(query, doc.text)) for doc in candidates]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

query = "reset api key"
candidates = [
    Doc("d1", "api key reset steps and security check"),
    Doc("d2", "billing refund policy and invoice"),
    Doc("d3", "permission management for admin users"),
    Doc("d4", "how to reset api key from dashboard"),
]

topk = rerank(query, candidates, top_k=2)
ids = [doc.doc_id for doc, score in topk]

assert ids[0] in {"d1", "d4"}
assert len(topk) == 2
assert topk[0][1] >= topk[1][1]
print(ids)
```

这段代码表达了四个工程上必须显式存在的步骤：输入候选、计算分数、排序、截断。真实系统会把 `simple_scorer` 换成 Cross-Encoder 模型推理。

如果是 LLM reranker，通常不会返回连续分数，而是返回排序结果、偏好对或前几名编号。一个极简接口可以长这样：

```python
def build_rerank_prompt(query: str, candidates: list[str]) -> str:
    lines = [f"Query: {query}", "Rank the documents by relevance:"]
    for i, doc in enumerate(candidates, start=1):
        lines.append(f"[{i}] {doc}")
    return "\n".join(lines)

def parse_topk(output: str, top_k: int = 3) -> list[int]:
    ids = [int(x) for x in output.split() if x.isdigit()]
    result = []
    for x in ids:
        if x not in result:
            result.append(x)
    return result[:top_k]

mock_output = "4 1 3"
assert parse_topk(mock_output, top_k=2) == [4, 1]
```

真实工程里，除了排序本身，还要补齐四类日志字段，否则上线后很难定位收益来源：

| 字段 | 作用 |
|---|---|
| `query_id / session_id` | 关联一次完整请求 |
| `candidate_ids` | 记录召回候选集合 |
| `rerank_scores` 或 `rank_order` | 记录重排序输出 |
| `selected_topk` | 记录进入上下文的证据 |
| `tool_route` | 记录后续调用了哪些工具 |
| `latency_ms / token_usage` | 记录成本 |
| `human_takeover` | 记录是否人工接管 |

一个真实工程例子可以这样设计链路：

1. 规划器判断问题属于“退款失败排查”。
2. 检索器从知识库召回 20 条候选。
3. 重排序器选出前 5 条。
4. 生成器基于前 5 条先给出解释。
5. 路由器判断是否还需要查工单系统或支付日志。
6. 状态层把 `query / candidates / scores / tool_results / retry_count` 写入 thread state。
7. 若日志系统超时，从最近 checkpoint 恢复，而不是从头重新检索和重排。

这个链路里，重排序只是中间一环，但它直接决定后续模型和工具看到什么证据。

---

## 工程权衡与常见坑

Cross-Encoder 和 LLM reranker 的取舍，本质上是在准确率、延迟、成本和稳定性之间做工程平衡。

| 维度 | Cross-Encoder | LLM reranker |
|---|---|---|
| 准确率 | 通常较高且稳定 | 可能更高，但波动更大 |
| 延迟 | 较低，近似随 `K` 线性增长 | 较高，受 token 和模型推理影响 |
| 成本 | 低到中 | 高 |
| 可解释性 | 较强，能输出逐条分数 | 较弱，常只给排序结果 |
| 并发适配 | 好 | 一般 |
| 适合任务 | 高频、稳定精排 | 高价值、歧义大问题 |

先看一个简单的数值估算。假设召回 `K_0 = 20`，最终放进上下文 `K_1 = 5`。
如果 Cross-Encoder 每对 `(q, d_i)` 打分约 `3 ms`，那么 rerank 总耗时约是：

$$
20 \times 3 = 60 \text{ ms}
$$

如果改成 LLM reranker，每段平均 `180 token`，那么光输入候选就约有：

$$
20 \times 180 = 3600 \text{ tokens}
$$

再加上查询、系统提示和输出 token，总成本会明显上升。对高并发客服系统来说，这通常不是小事，因为它会直接放大服务成本和尾延迟。

常见坑主要有五类。

第一，`K` 开太大。很多团队以为“多排一点总没坏处”，但 Cross-Encoder 的推理次数和 LLM reranker 的 token 长度都会随 `K` 增长。召回窗口过大，收益常常先饱和，成本却继续上涨。

第二，只看主观体验，不看指标。如果只凭“答案感觉更顺了”来决定是否上线，很容易误判。正确做法是同时看 `MRR@10 / nDCG@10 / MAP` 和端到端成功率。

第三，没有做消融实验。消融实验的白话解释是：一次只增加一个模块，观察收益来自哪里。推荐至少比较四组：`base`、`+rerank`、`+plan+tools`、`+checkpoint`。否则最后只知道系统变好了，不知道到底是哪一环有效。

第四，没有状态持久化。工具失败后如果整个流程从头开始，rerank 带来的收益会被异常重试吞掉。尤其是多工具 Agent，重排一次并不便宜，最好把中间结果可恢复地保存下来。

第五，误把 rerank 当成万能修复器。若问题本质是知识库缺失、文档切片过碎、工具路由错误，继续打磨 reranker 的边际收益会很低。

---

## 替代方案与适用边界

重排序不是唯一方案，也不是所有系统都必须上。是否需要它，取决于你的目标是“把前几条文档排得更准”，还是“把整个任务真正做成”。

下面这个边界表比较实用：

| 方案 | 适用条件 | 不适用条件 |
|---|---|---|
| 只召回不重排 | 低要求、低成本、结果可容忍 | 高精度问答、证据容易混淆 |
| Cross-Encoder | 中高并发、稳定需求、预算有限 | 需要复杂全局推理 |
| LLM reranker | 高歧义、高价值、低频查询 | 高并发、低预算 |
| 计划+工具+checkpoint | Agent 执行链长、失败代价高 | 仅静态问答 |

如果系统只是一个低频内部知识库搜索，且用户能接受自己点开前几条结果，那么向量召回加一点规则排序可能已经够用。规则排序的白话解释是：用发布时间、点击率、文档类型、业务优先级这类显式规则做简单加权，而不引入额外模型推理。

但如果系统是客服 Agent、运维助手或企业知识执行器，目标已经不只是“把文档找出来”，而是“让系统做对动作”。这时就不能只讨论 rerank，而要把规划、检索、工具路由、状态持久化和失败恢复串起来分析。

一个实用的决策顺序是：

1. 先确认召回是否足够。如果正确证据经常根本找不回来，先修召回。
2. 再确认是否存在“找到了但排不准”的问题。如果有，再考虑 rerank。
3. 若系统需要查日志、查数据库、提工单，继续引入工具路由。
4. 若工具链路长且失败常见，再引入 checkpoint 与人工接管。

换句话说，Cross-Encoder 与 LLM reranker 解决的是“证据排序”这一层，而不是整个 Agent 可靠性问题。对零基础到初级工程师来说，最重要的判断不是“哪种模型最先进”，而是“我的瓶颈到底在召回、排序、工具还是恢复”。

---

## 参考资料

1. [Sentence Transformers: CrossEncoder Usage](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
2. [Sentence Transformers: CrossEncoder Evaluation](https://www.sbert.net/docs/package_reference/cross_encoder/evaluation.html)
3. [Cohere Rerank Overview](https://docs.cohere.com/docs/rerank-overview)
4. [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
5. [LangChain Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
6. [A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE](https://europe.naverlabs.com/research/publications/a-thorough-comparison-of-cross-encoders-and-llms-for-reranking-splade/)
7. [Zero-Shot Listwise Document Reranking with a Large Language Model](https://dblp.org/rec/journals/corr/abs-2305-02156)

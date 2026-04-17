## 核心结论

RAG，Retrieval-Augmented Generation，中文可理解为“检索增强生成”，意思是模型在回答前先查外部资料，再基于资料生成答案。它解决的不是“模型不会说话”，而是“模型缺少最新知识、私有知识、可追溯证据”。

一套可落地的 RAG 系统，最小闭环就是 6 步：

1. 文档加载
2. 文档分块
3. 计算嵌入
4. 写入向量存储
5. 检索相关片段
6. 把片段拼进 Prompt 交给 LLM 生成

可以把它理解成“问答前先查资料再作答”。例如用户问“最新安全政策是什么”，系统不会直接让 LLM 硬答，而是先从政策库中检索出最近版本的政策段落，再把这些段落连同问题一起发给模型，最后输出带来源的回答。这就是企业知识问答、内部搜索、文档助手的标准做法。

| 方案 | 知识来源 | 时效性 | 可引用证据 | 典型问题 |
| --- | --- | --- | --- | --- |
| 传统问答 | 模型参数内部记忆 | 低 | 弱 | 容易过时 |
| RAG | 模型参数 + 外部文档 | 高 | 强 | 需要额外检索链路 |

一个最常见的 2-Step RAG 流程可以写成：

`文档入库 -> 用户提问 -> 检索 top-k -> 组装上下文 -> LLM 生成答案`

结论先给出：对“知识频繁更新、需要引用原文、数据不适合微调”的场景，RAG 通常比直接微调更便宜，也更容易维护。

---

## 问题定义与边界

RAG 的核心问题不是“怎么接一个向量库”，而是“什么问题值得做检索增强”。如果问题本身不依赖外部知识，RAG 反而会增加复杂度和延迟。

适合做 RAG 的场景有三类：

| 数据种类 | 更新频率 | 应对策略 |
| --- | --- | --- |
| 企业制度、产品文档、FAQ | 中到高 | 定期增量入库 |
| 法规、政策、公告 | 高 | 按日期版本化 + 新文档优先 |
| 长篇技术文档、论文、手册 | 低到中 | 分块 + 语义检索 |

边界也要讲清楚。

第一，RAG 不是数据库查询的替代品。像“订单总额是多少”这类精确数值问题，应该查 SQL 或 API，不应只靠向量检索。

第二，RAG 不是万能纠错器。如果检索召回错了，生成阶段通常也会跟着错。很多人把 hallucination，中文常译为“幻觉”，理解成模型乱编；在 RAG 里更常见的根源其实是“查错资料”或“没查到关键资料”。

第三，非结构化文档越多、主题越混杂，越需要更强的检索增强策略，例如混合检索、rerank。rerank 可以理解为“二次精排”，先粗召回，再让更强的模型重新排序。

一个新手能理解的玩具例子是：你有 10 篇公司制度文档，用户问“请假最多能请几天”。如果你把整篇 PDF 一次性喂给模型，成本高且容易漏；如果你先把文档切成多个段落，再找最像“请假制度”的几个段落，模型就更容易答对。

一个真实工程例子是：企业内部安全政策库，每周更新一次。文档先被下载并转成纯文本，再按约 800 token 分块，写入向量库。在线查询时，系统先执行“关键词召回 + 向量召回”，合并后做 rerank，只把前 5 段送进 LLM，并要求答案附带文档标题与生效日期。这种方案比“把所有文档塞进上下文”稳定得多。

---

## 核心机制与推导

Embedding，中文通常叫“嵌入向量”，就是把一段文本映射成一个高维数字数组，使语义相近的文本在向量空间里更接近。RAG 的检索通常依赖它。

最常用的相似度之一是余弦相似度：

$$
\cos(\theta)=\frac{\mathbf{A}\cdot\mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}
$$

它只比较方向，不强调向量长度。这一点对语义检索很重要，因为一段文本“信息量更大”不应该自动意味着“更相关”。

玩具例子如下。设：

- $\mathbf{A}=[3,4,0]$
- $\mathbf{B}=[1,2,2]$

则点积为 $3\times1+4\times2+0\times2=11$，模长分别为 $\|\mathbf{A}\|=5$、$\|\mathbf{B}\|=3$，所以：

$$
\cos(\theta)=\frac{11}{5\times3}=0.73
$$

0.73 可以粗略理解为“两个段落语义较接近”。

如果用 L2 距离，即欧氏距离，也就是“直线距离”，结果是：

$$
\|\mathbf{A}-\mathbf{B}\|_2=\sqrt{(3-1)^2+(4-2)^2+(0-2)^2}=\sqrt{12}\approx3.46
$$

L2 没错，但它受模长影响更明显。在很多 embedding 模型输出上，先归一化再做余弦或点积排序，通常更稳定。

RAG 的在线检索链路通常分两层：

1. 粗召回：向量索引找 top-k 候选
2. 精排序：用 rerank 模型重新排序

其中近似最近邻，Approximate Nearest Neighbor，简称 ANN，可以理解为“用更少计算量近似找到最近向量”。它牺牲少量精度，换取更快搜索速度，这也是 FAISS、HNSW 这类库的价值。

下面这张表说明为什么很多系统会加 rerank：

| 检索方案 | Top-5 Precision | Top-5 Recall | MRR |
| --- | --- | --- | --- |
| 仅向量召回 | 0.62 | 0.81 | 0.68 |
| 向量召回 + rerank | 0.74 | 0.82 | 0.79 |

这里的 MRR，Mean Reciprocal Rank，中文可理解为“首个正确结果排名的倒数均值”，越高表示正确结果排得越靠前。

---

## 代码实现

下面给出一个纯 Python 的最小 RAG 原型。它不依赖外部模型，用词袋向量模拟 embedding，目的是让输入输出关系足够清楚。生产环境只需要把 `embed_text` 换成真实 embedding API，把内存列表换成 FAISS 或别的向量库即可。

```python
import math
import re
from typing import List, Dict, Tuple

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z\u4e00-\u9fff0-9]+", text.lower())

def split_chunks(text: str, chunk_size: int = 40, overlap: int = 10) -> List[str]:
    words = tokenize(text)
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab = {}
    for text in texts:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def embed_text(text: str, vocab: Dict[str, int]) -> List[float]:
    vec = [0.0] * len(vocab)
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def index_chunks(chunks: List[str]) -> Tuple[Dict[str, int], List[Dict]]:
    vocab = build_vocab(chunks)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": i,
            "text": chunk,
            "vector": embed_text(chunk, vocab)
        })
    return vocab, records

def retrieve(query: str, vocab: Dict[str, int], records: List[Dict], top_k: int = 3) -> List[Dict]:
    qv = embed_text(query, vocab)
    scored = []
    for rec in records:
        score = cosine(qv, rec["vector"])
        scored.append({**rec, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

docs = [
    "安全政策 2026 版本规定 所有生产环境访问必须开启多因素认证 并保留审计日志",
    "请假制度规定 年假需要提前申请 病假需要提交证明",
    "研发规范要求 代码合并前必须通过测试和代码评审"
]

all_chunks = []
for doc in docs:
    all_chunks.extend(split_chunks(doc, chunk_size=12, overlap=3))

vocab, records = index_chunks(all_chunks)
results = retrieve("最新安全政策要求什么", vocab, records, top_k=2)

assert len(results) == 2
assert "安全政策" in results[0]["text"]
assert results[0]["score"] >= results[1]["score"]

for item in results:
    print(item["score"], item["text"])
```

这个原型对应的模块边界如下：

| 组件 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| Loader | 文件路径、URL、数据库记录 | 原始文本 | 统一加载数据 |
| Splitter | 原始文本 | chunk 列表 | 控制粒度 |
| Embedder | chunk / query | 向量 | 进入向量空间 |
| Indexer | 向量 + 元数据 | 可检索索引 | 支持检索 |
| Retriever | query 向量 | top-k chunk | 找候选证据 |
| LLM Prompt | 问题 + chunk | 最终答案 | 生成可读回复 |

如果换成真实工程实现，伪码通常是：

```text
docs = load_texts()
chunks = split_chunks(docs)
vectors = embedding_model.embed(chunks)
vector_store.add(vectors, metadata=chunks)

query_vec = embedding_model.embed(user_query)
candidates = vector_store.search(query_vec, top_k=20)
reranked = rerank(user_query, candidates)[:5]
prompt = build_prompt(user_query, reranked)
answer = llm.generate(prompt)
```

如果使用 FAISS，可以把“内存列表检索”替换成它的索引结构。FAISS 本质上就是高效相似度搜索库，适合从几千到上亿向量的场景。

---

## 工程权衡与常见坑

RAG 的效果，往往不是卡在模型，而是卡在分块、召回和评估。

第一个权衡是 chunk size。chunk 就是“切出来的小文本块”。太大，容易混入多个主题；太小，上下文不完整。常见起点是 500 到 1000 token，再按文档结构调。

| Chunk 大小 | 优点 | 风险 | 常见结果 |
| --- | --- | --- | --- |
| 200-400 | 定位精细 | 语义断裂 | Recall 偏低 |
| 500-1000 | 平衡较好 | 需要调 overlap | 常用默认值 |
| 1200-1500 | 上下文更完整 | 主题混杂、浪费窗口 | Precision 偏低 |

真实工程里经常出现这种现象：把 chunk 从 1500 token 调到 800 token，Top-5 precision 提升约 8%，因为每块主题更集中，召回更准。

第二个坑是只做向量检索，不做混合检索。很多专有名词、版本号、错误码，本质上更适合关键词匹配。最稳妥的做法通常是 hybrid retrieval，也就是“关键词检索 + 语义检索”的混合方案。

第三个坑是没有评估。RAG 不评估，就很难知道问题出在“没召回”还是“答错了”。

常见指标如下：

$$
Precision@k=\frac{TP@k}{k}
$$

$$
Recall@k=\frac{TP@k}{\text{relevant docs}}
$$

$$
MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}
$$

其中 $TP$ 是真正例，可理解为“确实相关的结果数”；$rank_i$ 是第一个正确结果的排名。

一个新手能落地的监控脚本，是每周抽 50 个问题，人工标注正确文档，然后计算 top-5 hit rate，也就是“前 5 个结果里是否出现正确文档”。这比一开始就追求复杂评测平台更实用。

还要注意生成阶段的约束。Prompt 最好明确要求：

- 只能基于给定上下文回答
- 不知道就回答不知道
- 输出引用片段或文档编号

否则即使检索对了，模型也可能把多个片段拼错。

---

## 替代方案与适用边界

并不是所有知识问答都要上完整 RAG。

如果你只有 10 篇手册，先做 keyword match，也就是“关键词匹配”，通常更划算。比如只对标题、摘要和一级小节标题做 BM25 或简单倒排索引，很多 FAQ 已经够用。

如果数据规模中等、查询既有关键词又有语义表达变化，混合 RAG 往往是最稳妥的工程选择。

如果任务是高度结构化、答案必须精确到字段级，优先考虑数据库查询、知识图谱或工具调用，而不是只靠向量检索。

微调也不是 RAG 的直接替代。微调更适合固定格式输出、风格学习、领域术语适配；它不擅长承载频繁更新的知识库。

| 方案 | 适用场景 | 优点 | 代价 |
| --- | --- | --- | --- |
| 纯关键词检索 | 小文档集、FAQ、标题明确 | 简单便宜 | 语义泛化差 |
| Hybrid RAG | 企业知识库、政策库、技术文档 | 精度与维护性平衡 | 系统复杂度中等 |
| Full LLM Fine-tune | 风格统一、任务固定 | 输出稳定 | 更新知识成本高 |

企业级真实场景里，如果是全量政策库、客服知识库、售后排障手册，RAG 很适合；如果只是几十条 FAQ，传统检索就足够；如果是股票价格、订单状态、库存余量，应该接实时数据源并支持增量更新，而不是把所有状态做成静态 chunk。

---

## 参考资料

| 资源名称 | 链接 | 主要贡献 |
| --- | --- | --- |
| Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | https://arxiv.org/abs/2005.11401 | RAG 经典论文，定义“参数记忆 + 非参数记忆”的基本框架 |
| LangChain Retrieval Docs | https://docs.langchain.com/oss/python/langchain/retrieval | 给出 2-Step RAG、Agentic RAG、检索组件拆分方式 |
| LangChain RAG Tutorial | https://docs.langchain.com/oss/python/langchain/rag | 提供完整索引与问答链路示例 |
| FAISS GitHub | https://github.com/facebookresearch/faiss | 说明向量索引的能力边界与检索 trade-off |
| FAISS Getting Started Wiki | https://github.com/facebookresearch/faiss/wiki/Getting-started | 展示 Python 中 `add` 与 `search` 的最小用法 |
| Ragas Metrics Docs | https://docs.ragas.io/en/latest/concepts/metrics/ | 说明 faithfulness、context precision、context recall 等评估指标 |
| RAG Survey 2023 | https://arxiv.org/abs/2312.10997 | 系统总结 Naive RAG、Advanced RAG、Modular RAG 的演进 |

可参考的文末引用格式示例：

- [1] Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- [2] LangChain Docs. Retrieval.
- [3] Meta AI. FAISS Documentation.
- [4] Ragas Docs. Metrics for RAG Evaluation.

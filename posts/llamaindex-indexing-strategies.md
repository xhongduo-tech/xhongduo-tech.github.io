## 核心结论

LlamaIndex 的“索引策略”本质上不是只选一种索引，而是把不同索引当成不同能力层来组合。向量索引负责语义相似，意思是“就算词不一样，也能找出表达接近的内容”；关键词索引负责精确命中，意思是“用户问了哪个词，就优先抓哪个词”；知识图谱索引负责实体关系，意思是“把文本拆成谁和谁有什么关系”；树状索引负责长文层次摘要，意思是“先总结局部，再总结整体”。

对新手可以直接这样理解：LlamaIndex 就像同时启用 FAISS 向量检索、知识图谱问答、层级摘要和 BM25 精确命中，然后把它们的输出交给一个统一的响应合成模块整理成最终回答。官方文档里，响应合成器（Response Synthesizer）就是“把检索出来的文本块变成答案”的组件；多路检索则可以通过 RouterRetriever、QueryFusionRetriever 或自定义流程来实现。

四类索引不是互斥关系，而是针对不同查询失败模式的补丁。只做向量检索，容易漏掉专有名词；只做 BM25，容易漏掉同义表达；只做知识图谱，难处理开放式长文语义；只做树状摘要，又不适合精准定位细节。工程上更稳妥的方案通常是“向量 + BM25”打底，再按业务是否存在实体关系问答或长文总结需求，决定是否加图谱或树状摘要。

| 索引类型 | 输入形式 | 核心输出 | 最适场景 |
| --- | --- | --- | --- |
| 向量索引 | 文本 embedding，意思是数值化语义表示 | 相似文本块 | 同义改写、自然语言问答 |
| 关键词索引 / BM25 | 分词后的词项统计 | 关键词命中文本块 | 专有名词、报错码、接口名 |
| 知识图谱索引 | 文本中抽出的三元组 | 实体与关系路径 | “谁依赖谁”“A 和 B 什么关系” |
| 树状索引 | 分块文本及各层摘要 | 多层级总结结果 | 超长文档总结、章节归纳 |

---

## 问题定义与边界

问题不是“哪种索引最好”，而是“一个服务要同时回答哪些类型的问题”。如果查询类型混杂，只靠一种索引通常会出现系统性漏召回。

一个简单分类如下：

1. 语义型查询：用户说法和原文不一致，但意思接近。
2. 术语型查询：用户必须命中特定词，如产品名、报错码、表字段名。
3. 关系型查询：用户问的是实体之间的连接关系。
4. 长文归纳型查询：用户问整篇文档的主线，而不是某一句。

玩具例子：知识库里有一句“Redis 持久化会带来额外 I/O 开销”。用户问“为什么缓存落盘会变慢”。这里“落盘”和“持久化”不是同一个词，但语义接近，向量索引更容易召回。

真实工程例子：员工满意度平台里，用户可能问“希腊团建反馈怎么样”，也可能问“Pinecone 在员工建议里出现在哪些语境”。前者偏语义，因为原文未必写“希腊团建”，可能写“雅典 offsite”“公司出游”；后者偏术语，必须抓住 “Pinecone” 这个明确词项。如果系统只做向量检索，第二类问题容易被无关但语义近似的文本稀释；如果只做 BM25，第一类问题又很难覆盖同义表达。

因此边界很清楚：

- 轻量知识库：优先解决语义召回和精确命中，通常“向量 + BM25”足够。
- 实体关系密集：再加知识图谱。
- 文档极长且结构明显：再加树状索引。
- 纯结构化数据查询：优先 SQL 或图数据库，不应强行用 RAG 索引替代。

各类索引负责的查询类型，可以用一条简单分工线来记：

- 向量索引：负责“意思像不像”
- BM25：负责“词出现没出现”
- 知识图谱：负责“实体怎么连起来”
- 树状索引：负责“整体怎么概括”

---

## 核心机制与推导

混合检索最常见的核心是 dense + sparse。dense 指稠密向量，也就是 embedding；sparse 指稀疏向量，也就是词项权重。Pinecone 文档给出的典型做法是用 $\alpha$ 控制二者权重，本质是线性加权：

$$
score_{\text{hybrid}}=\alpha \cdot score_{\text{dense}}+(1-\alpha)\cdot score_{\text{sparse}}, \quad \alpha \in [0,1]
$$

白话解释：先用 dense 向量找“意思接近”的文本，再用 sparse 向量抓住“词本身要对上”的文本，最后算一个折中分数。$\alpha$ 越大，越偏语义；越小，越偏关键词。Pinecone 官方示例也给出相同的凸组合思路，即对 dense 向量乘 $\alpha$，对 sparse 值乘 $(1-\alpha)$ 后再查询。

玩具例子：

- 查询：“pinecone athens offsite”
- 文本 A：“Visiting the Parthenon during the Pinecone offsite...”
- 文本 B：“Last time I visited Greece was on my own.”

A 和查询同时包含 “Pinecone” 与 “offsite”，语义和关键词都强；B 只和 “Greece/Athens” 语义相关。此时混合检索比纯向量更容易把 A 放到前面。

树状索引的机制不同。它不强调“找一段最像的文本”，而强调“逐层压缩长文信息”。LlamaIndex 的 TreeIndex 与 TreeSummarize 都采用自底向上策略：先把叶子 chunk 摘要，再把多个摘要继续汇总，直到根节点。白话解释就是“先归纳局部，再归纳整体”。

可以写成简化伪代码：

```text
chunks = split(document)
level = chunks
while len(level) > 1:
    groups = pack(level, num_children)
    level = [summarize(group) for group in groups]
return level[0]
```

这套机制适合长制度文档、需求文档、会议纪要，因为回答“这份文档整体结论是什么”时，不能只看 top_k 相似片段，而要让多层信息都参与。

知识图谱索引的核心是三元组抽取。三元组就是 `(主体, 关系, 客体)`，例如 `(Pinecone, supports, hybrid search)`。LlamaIndex 旧接口里是 `KnowledgeGraphIndex`，但官方 API 已标注该类已弃用，更推荐 `PropertyGraphIndex`。机制没有变：依赖 LLM 按 prompt 从非结构化文本抽关系，然后在查询时按实体路径召回相关节点。白话解释：不是按句子相似度找答案，而是先把“谁和谁有关”抽成图，再沿着图找。

所以四类机制的数学与工程含义可以压缩成一句话：

- 向量索引解决语义空间邻近问题。
- BM25 解决词项统计排序问题。
- 知识图谱解决实体关系显式建模问题。
- 树状索引解决长文多层信息聚合问题。

---

## 代码实现

下面先给一个可运行的玩具代码，只演示混合分数如何工作。它不依赖外部库，但足以说明 $\alpha$ 的作用。

```python
from math import isclose

def hybrid_score(score_dense: float, score_sparse: float, alpha: float) -> float:
    assert 0.0 <= alpha <= 1.0
    return alpha * score_dense + (1 - alpha) * score_sparse

# 玩具例子：文档A语义强、关键词一般；文档B语义一般、关键词强
doc_a = hybrid_score(score_dense=0.92, score_sparse=0.30, alpha=0.5)
doc_b = hybrid_score(score_dense=0.65, score_sparse=0.88, alpha=0.5)

assert isclose(doc_a, 0.61)
assert isclose(doc_b, 0.765)
assert doc_b > doc_a  # 在 alpha=0.5 时，关键词优势更明显的 B 排前

# 如果更偏语义
doc_a_semantic = hybrid_score(0.92, 0.30, alpha=0.9)
doc_b_semantic = hybrid_score(0.65, 0.88, alpha=0.9)

assert doc_a_semantic > doc_b_semantic
print("hybrid scoring demo passed")
```

接着看接近真实工程的示意框架。这里重点不是完整跑通，而是理解组件怎么拼起来。

```python
# 示意代码：包路径可能随 LlamaIndex 版本有调整
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

# 1. 读取文档并切块
docs = SimpleDirectoryReader("./data").load_data()

# 2. FAISS 向量索引
dim = 1536  # embedding 维度，需要与所选 embedding 模型一致
faiss_index = faiss.IndexFlatL2(dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
)

vector_retriever = vector_index.as_retriever(similarity_top_k=5)

# 3. BM25 关键词检索
nodes = vector_index.docstore.docs.values()
bm25_retriever = BM25Retriever.from_defaults(
    nodes=list(nodes),
    similarity_top_k=3,
)

# 4. 多路融合检索
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,
    use_async=False,
)

# 5. 统一响应合成
response_synthesizer = get_response_synthesizer(response_mode="compact")

retrieved_nodes = hybrid_retriever.retrieve("Pinecone hybrid search 的优势是什么？")
response = response_synthesizer.synthesize(
    query="Pinecone hybrid search 的优势是什么？",
    nodes=retrieved_nodes,
)

print(response)
```

如果后端换成 Pinecone，工程思路通常是：

1. 用 dense embedding 建语义向量。
2. 用 BM25 或稀疏编码器建 sparse 向量。
3. 查询时用 $\alpha=0.5$ 作为起点。
4. 如果业务术语极多，就把 $\alpha$ 下调到 0.3 到 0.6 区间做实验。
5. 最终仍由 response synthesizer 统一整理答案。

如果要加入知识图谱与树状摘要，一般不是替换前面的向量/BM25，而是作为补充路由：

- 知识图谱：回答“谁依赖谁”“某实体出现在哪些关系里”
- 树状摘要：回答“这份长文整体结论是什么”

真实工程里，一个常见组合是：

- 默认入口：向量检索 + BM25
- 当查询含实体关系模式：转知识图谱
- 当用户问“总结/概括/主线”：转树状摘要
- 最后统一走响应合成器输出

---

## 工程权衡与常见坑

第一类坑是把底层向量库能力想得太全。FAISS 很快，也适合本地和单机，但 LlamaIndex 官方 FAISS 实现明确写了 metadata filters 尚未实现。如果你直接期待 `query.filters` 生效，会报错。

真实工程里，一个常见修复是外层维护 `id -> metadata` 映射，先做向量召回，再在应用层过滤：

```python
id_to_meta = {
    "101": {"team": "search", "lang": "zh"},
    "102": {"team": "infra", "lang": "en"},
}

retrieved_ids = ["101", "102"]
filtered_ids = [i for i in retrieved_ids if id_to_meta[i]["team"] == "search"]

assert filtered_ids == ["101"]
```

第二类坑是把知识图谱当成“自动就准”的能力。三元组抽取强依赖 prompt 和原始文本质量。句子里代词多、格式乱、表格转文本脏，都会导致主体和客体识别错误。再加上 `KnowledgeGraphIndex` 已被官方标记为弃用，当前新项目更应关注 `PropertyGraphIndex`，否则后续升级会有成本。

第三类坑是混合检索参数误设。Pinecone 的 hybrid 查询本质是线性融合。如果 sparse 侧没有做好归一化，或者 $\alpha$ 过低，检索结果会被关键词噪声拉偏；如果 $\alpha$ 过高，又会退化成几乎纯语义搜索，专有名词命中率下降。

第四类坑是树状摘要层级过深。树过深意味着摘要调用次数上升，成本和延迟都上升，而且多层摘要会引入信息损耗。文档本来就不长时，强行上树状摘要通常不划算。

| 问题 | 根因 | 规避方式 |
| --- | --- | --- |
| FAISS 过滤失败 | 官方实现不支持 metadata filters | 外层维护 ID 到 metadata 映射，召回后过滤 |
| 图谱关系错乱 | 三元组抽取 prompt 不稳，原文噪声大 | 清洗文本，限制每 chunk 三元组数，人工抽检 |
| hybrid 结果偏移 | sparse 未归一化或 $\alpha$ 失衡 | 先用 0.5 起步，再按评测集调参 |
| 树状摘要太慢 | chunk 太碎、层级太深 | 控制 chunk 数量和 `num_children` |
| top_k 设太大 | 召回冗余，合成阶段噪声增加 | 先小范围评测，一般从 3 到 10 调整 |

一个经验判断标准是：如果你没有离线评测集，就不要频繁谈“最优索引策略”，因为你无法证明召回率、精确率和响应延迟之间的平衡是否真的更好。

---

## 替代方案与适用边界

不是每个项目都值得上“全家桶”。索引体系越复杂，数据同步、调参、评测和运维成本越高。

小团队知识库就是典型例子。假设只有几千篇内部文档，主要问题是“概念问答 + 专有名词命中”，那么 FAISS + BM25 往往已经够用，不必立刻引入知识图谱或树状摘要。因为图谱要解决的是实体关系推理，树状摘要要解决的是超长文档多层归纳；如果业务没有这两类痛点，引入它们只会增加复杂度。

反过来，如果你做的是法规库、论文库、企业流程文档，且用户常问“总结这份材料的主要变化”“供应商 A 和组件 B 有什么依赖关系”，那单一向量库就不够了。此时应考虑树状摘要或图结构检索。

| 方案 | 适用场景 | 成本 | 复杂度 |
| --- | --- | --- | --- |
| 单一向量索引 | FAQ、普通知识库、语义问答为主 | 低 | 低 |
| BM25 / 倒排表 | 专有词、编号、日志、报错码检索 | 低 | 低 |
| 向量 + BM25 | 大多数中小型 RAG 项目 | 中 | 中 |
| 向量 + BM25 + 图谱 | 实体关系密集、知识连接重要 | 中高 | 高 |
| 向量 + BM25 + 树状摘要 | 长文总结、报告归纳 | 中高 | 高 |
| 全组合方案 | 查询类型很多且预算足够 | 高 | 高 |

可以把适用边界总结为：

- 没有明显关系查询，不必先上图谱。
- 没有超长文总结，不必先上树状摘要。
- 没有专有名词痛点，纯向量可以先跑。
- 没有语义改写需求，纯 BM25 也可能足够。
- 只有当多种失败模式同时存在时，混合索引才真正体现价值。

---

## 参考资料

1. LlamaIndex Response Synthesizer 文档：说明响应合成器在检索后如何把文本块组织成答案  
   https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/

2. LlamaIndex FaissVectorStore API：说明 `FaissVectorStore` 的基本用法，以及 `IndexFlatL2(d)` 这类初始化方式  
   https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/faiss/

3. LlamaIndex BM25 Retriever 文档：说明 `BM25Retriever` 的接口与 `similarity_top_k` 等参数  
   https://docs.llamaindex.ai/en/stable/api_reference/retrievers/bm25/

4. LlamaIndex TreeIndex API：说明树状索引是自底向上构建、节点是子节点摘要  
   https://docs.llamaindex.ai/en/stable/api_reference/indices/tree/

5. LlamaIndex TreeSummarize 文档：说明递归 repack 与 summarize 的树状摘要过程  
   https://docs.llamaindex.ai/en/stable/api_reference/response_synthesizers/tree_summarize/

6. LlamaIndex Knowledge Graph 文档：说明三元组抽取式知识图谱索引；同时可注意官方已标记 `KnowledgeGraphIndex` 弃用，建议关注 `PropertyGraphIndex`  
   https://docs.llamaindex.ai/en/stable/api_reference/indices/knowledge_graph/

7. Pinecone Hybrid Search 文档：说明 dense 与 sparse 的混合检索方式，以及按 $\alpha$ 做线性加权的示例  
   https://docs.pinecone.io/guides/search/hybrid-search

8. Pinecone 与 LlamaIndex 集成文档：说明在 Pinecone 上接入 LlamaIndex 的典型流程  
   https://docs.pinecone.io/integrations/llamaindex

9. Pinecone Hybrid Search 博客：给出“员工满意度平台”“Greece offsite / Pinecone”这类真实检索场景示例  
   https://www.pinecone.io/blog/hybrid-search/

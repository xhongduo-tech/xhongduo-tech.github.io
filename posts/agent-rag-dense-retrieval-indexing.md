## 核心结论

Dense Retrieval 的索引构建，本质上是在离线阶段把文档切成可检索的片段，再把每个片段编码成固定维度的向量，最后连同元数据一起写入近似最近邻索引。这里的“向量”可以理解为一串能表示语义特征的数字；“元数据”是附加约束信息，比如时间、地域、权限、版本。

它成立的关键原因是：查询和文档被映射到同一个向量空间后，语义接近的内容在空间里也更接近，所以可以用内积或余弦相似度近似“是否相关”。最常见的打分形式是：

$$
s(q,d)=\langle \mathbf q,\mathbf d \rangle
$$

其中 $\mathbf q=f_Q(q)$ 是查询向量，$\mathbf d=f_D(d)$ 是文档向量，$\langle \cdot,\cdot \rangle$ 表示内积。

但工程上不能把“向量化”理解成全部工作。索引质量往往主要取决于三件事：

| 环节 | 决定什么 | 典型失败 |
|---|---|---|
| Chunk 切分 | 检索粒度是否合适 | 一段里混入多个主题，召回不准 |
| Metadata 设计 | 能否满足权限、时间、地域约束 | 召回到不合规或过期内容 |
| 索引更新 | 检索是否反映最新知识 | 文档已更新，索引仍指向旧版本 |

对 RAG 来说，Dense Retrieval 适合作为第一阶段召回器。它擅长找“语义相近”的内容，但对精确编号、精确年份、精确 ID 往往不够稳定，所以真实系统通常会叠加 metadata filter、hybrid retrieval 或 reranker。

---

## 问题定义与边界

索引构建的输入，不是原始 PDF 或网页本身，而是经过清洗后的 chunk 集合。这里的“chunk”就是切分后的文本片段，目的是让检索单元既足够小，能精确命中，又足够完整，不丢上下文。输出则是一组：

1. 向量
2. 稳定 ID
3. 原文片段
4. 可过滤的元数据

这件事的边界要先说清。Dense Retrieval 解决的是“在大规模文本集合里先找出可能相关的若干片段”，不是直接回答问题，也不是自动保证正确性。它尤其不保证以下事情：

| 需求 | Dense Retrieval 是否单独解决 |
|---|---|
| 精确关键词匹配 | 不稳定 |
| 权限隔离 | 不能，必须靠 metadata 或索引分片 |
| 最新版本优先 | 不能，必须靠版本字段和更新策略 |
| 最终答案正确 | 不能，后面还需要 rerank 或生成约束 |

一个典型的玩具例子是“2026 年税务模板”。用户问的是模板，但系统里可能同时有“2024 报税说明”“2026 税务变更解读”“2026 申报表格样例”。如果只看语义，相近内容都可能被召回。真正可用的系统必须在召回后继续按 `doc_type=template`、`year=2026`、`status=published` 过滤，否则很容易把解释文档当成模板文档。

chunk 大小也不是越大越好。太大时，一个 chunk 里可能同时讲“退税条件”和“申报流程”，查询只关心其中一半，却被整段召回，噪声会变多。太小时，又可能把“例外条款”切掉，导致生成阶段丢关键信息。对合同、制度、说明文档，常见做法是按标题和段落边界切分，长度控制在 500 到 800 token，并保留 50 到 100 token overlap。这里的“overlap”就是相邻 chunk 的重叠区域，用来防止一句话或一个条款恰好被切断。

---

## 核心机制与推导

Dense Retrieval 的核心结构通常是双编码器。所谓“双编码器”，就是查询和文档分别经过编码模型，得到同维度向量：

$$
\mathbf q=f_Q(q),\quad \mathbf d=f_D(d),\quad \mathbf q,\mathbf d\in\mathbb R^d
$$

检索时不再扫描全文，而是在向量空间中比较相似度。若使用内积，则：

$$
s(q,d)=\mathbf q^\top \mathbf d=\sum_{i=1}^{d} q_i d_i
$$

为什么这能工作？因为训练目标会把“相关查询-文档对”拉近，把“不相关对”推远。于是，查询里没出现过的同义表达，也可能被召回。例如“怎么报销出差费”和“差旅费用报销流程”词面不同，但向量可能很近。

看一个最小数值例子。设查询向量：

$$
\mathbf q=[0.9,0.1,0.0]
$$

两个候选文档向量分别是：

$$
\mathbf d_1=[0.8,0.2,0.0],\quad \mathbf d_2=[0.3,0.3,0.9]
$$

则有：

$$
s_1=\mathbf q^\top \mathbf d_1=0.9\times0.8+0.1\times0.2=0.74
$$

$$
s_2=\mathbf q^\top \mathbf d_2=0.9\times0.3+0.1\times0.3=0.30
$$

所以 $\mathbf d_1$ 更相关。这个例子很小，但说明了检索排序的本质：不是按关键词出现次数，而是按向量打分排序。

真实工程里不会线性扫描全部向量，而会把向量放进 ANN 索引。ANN 是 Approximate Nearest Neighbor，白话说就是“用更快但允许少量近似误差的方法找最近邻”。常见结构包括 HNSW、IVF、PQ。它们的目标不是改变语义打分，而是把“全库遍历”变成“近似地只查一小部分候选”。

一个常见的在线流程可以写成：

`query -> embedding -> ANN top-k -> metadata filter -> reranker -> LLM`

注意 metadata filter 的位置在工程上有两种实现：预过滤和后过滤。若约束是硬性的，例如租户、地域、权限、版本，应该尽量前置；否则可能先召回到不合规文档，再在后面丢弃，造成有效召回下降。

真实工程例子更能说明问题。金融级 RAG 往往不能把全球文档放在同一个统一索引里直接搜。原因不是模型不够强，而是合规要求先天存在。更稳妥的设计是按 `region` 或 `tenant` 分片建索引，例如 APAC、EU、US 各自独立 embedding 和索引。查询进来后先根据用户身份选择索引，再做相似度搜索，再按版本和可见级别过滤。这样做的代价是维护复杂度提高，但可以显著降低跨区域误召回的风险。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，把“chunk -> embed -> metadata filter -> top-k”串起来。这里不用真实大模型，而是用一个简化词表向量来演示机制。代码的重点不是效果，而是索引构建的数据结构。

```python
from math import sqrt

docs = [
    {
        "id": 1,
        "text": "2026 税务模板 个人所得税申报表 标准版",
        "meta": {"year": 2026, "doc_type": "template", "region": "CN"}
    },
    {
        "id": 2,
        "text": "2024 税务模板 个人所得税申报表 旧版",
        "meta": {"year": 2024, "doc_type": "template", "region": "CN"}
    },
    {
        "id": 3,
        "text": "2026 税务政策解读 个税申报流程说明",
        "meta": {"year": 2026, "doc_type": "guide", "region": "CN"}
    },
]

vocab = ["2026", "税务", "模板", "申报表", "流程", "解读"]

def embed(text: str):
    vec = [text.count(token) for token in vocab]
    norm = sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# 离线索引构建
index = []
for doc in docs:
    index.append({
        "id": doc["id"],
        "vector": embed(doc["text"]),
        "text": doc["text"],
        "meta": doc["meta"],
    })

# 在线查询
query = "查 2026 年税务模板"
qvec = embed(query)

# 先过滤硬约束，再排序
candidates = [
    item for item in index
    if item["meta"]["year"] == 2026 and item["meta"]["region"] == "CN"
]

ranked = sorted(
    candidates,
    key=lambda item: dot(qvec, item["vector"]),
    reverse=True
)

top1 = ranked[0]
assert top1["meta"]["doc_type"] == "template"
assert "模板" in top1["text"]
print(top1["id"], top1["text"])
```

这个例子虽然简化，但已经包含了 Dense Retrieval 索引构建里最重要的几个步骤：

| 字段 | 作用 | 工程要求 |
|---|---|---|
| `id` | 稳定定位 chunk | 不要用会变化的临时序号 |
| `vector` | 用于相似度检索 | 必须与在线查询使用同一 embedding 模型 |
| `text` | 返回给 reranker 或 LLM | 保留原文，不要只存向量 |
| `meta` | 过滤与审计 | 字段名固定，值域可控 |

如果换成真实系统，流程通常是：

1. 按文档结构切 chunk，例如标题、段落、列表边界。
2. 为每个 chunk 生成唯一 ID，并记录 `source_id`、`chunk_id`、`version`、`region`、`updated_at`。
3. 用统一的 embedding 模型批量生成向量。
4. 把向量和 ID 写入 FAISS、HNSW 或向量数据库。
5. 把文本和 metadata 写入可回表的存储层。
6. 查询时先做权限与租户约束，再进行 ANN 召回，再 rerank。

这里最容易被忽视的一点是“稳定 ID”。如果每次重建索引都重新分配 ID，就很难做增量更新、问题追踪和缓存复用。另一个关键点是日志。至少应该记录 query、过滤条件、top-k 命中片段、最终采用片段、索引版本号。否则你很难判断问题出在切分、向量、过滤，还是 reranker。

---

## 工程权衡与常见坑

Dense Retrieval 的主要工程难点不在“怎么调 API”，而在“怎么控制误召回和漏召回”。常见坑可以直接列出来：

| 常见坑 | 现象 | 原因 | 缓解策略 | 监控指标 |
|---|---|---|---|---|
| Chunk 过大 | 召回结果主题混杂 | 一个 chunk 含多个主题 | 按结构切分，降低 chunk 长度 | top-k 平均长度、人工相关率 |
| Chunk 过小 | 回答不完整 | 关键上下文被切断 | 增加 overlap，按语义边界切 | no-answer 率、补充召回率 |
| 缺少 metadata | 召回到旧版或越权文档 | 只有语义，没有硬约束 | 建立版本、权限、地域字段 | 合规拦截率、错误命中率 |
| 索引陈旧 | 文档已更新但结果仍旧 | 没有增量重建 | 建立 CDC 或定时增量 embedding | 索引滞后时间 |
| 只用 dense | 编号、年份命中差 | 语义近但词面不精确 | 引入 BM25 或 hybrid | exact-match query 成功率 |

其中最常见的误区是以为“embedding 模型越强，检索就一定越准”。实际上，很多线上故障来自数据治理而不是模型本身。比如同一份制度文档有多个版本，索引里没有 `version` 和 `effective_date`，那么系统可能稳定地召回“语义最像的旧版本”。从模型角度它没错，从业务角度却是严重错误。

另一个常见坑是过滤顺序。假设一个查询是“查 2025 Q4 APAC 报表”。如果先在全库做 dense 搜索，再在结果里过滤 `year=2025, quarter=Q4, region=APAC`，那么 top-k 很可能被别的年份挤占，最后过滤后一个都不剩。此时用户会误以为知识库没有数据，实际问题是过滤太晚。对这种硬条件，应优先缩小候选集合，再做语义排序。

真实工程里，索引分片也是权衡。按地域、租户、业务线拆多个索引，会增加构建和运维成本，但可以降低延迟并强化隔离。全部数据堆进一个全局索引，管理简单，却更容易在权限和召回精度上出问题。没有绝对最优，只有是否符合约束。

---

## 替代方案与适用边界

Dense Retrieval 不是通用最优解。对于精确编号、法规条文号、错误码、零件号这类“词面必须一致”的任务，BM25 这类 sparse retrieval 往往更稳。这里的“sparse”可以理解为基于词项出现情况打分，而不是基于语义向量。

因此很多生产系统采用 hybrid retrieval，也就是 dense 和 sparse 结合。一个简单形式是线性加权：

$$
s_{\mathrm{hybrid}} = \alpha s_{\mathrm{dense}} + (1-\alpha)s_{\mathrm{sparse}}
$$

其中 $\alpha \in [0,1]$。如果查询偏语义问答，可以提高 $\alpha$；如果查询包含强精确约束，比如年份、编号、SKU，可以降低 $\alpha$。

适用边界可以概括为：

| 场景 | 更适合的方法 |
|---|---|
| FAQ、知识问答、长文本语义匹配 | Dense Retrieval |
| 精确编号、法规条款、日志错误码 | Sparse / BM25 |
| 企业知识库、既要语义又要精确过滤 | Hybrid + Metadata |
| 强合规、多租户、多地域 | 分片索引 + Metadata 预过滤 |

继续看一个实际查询：“查 2025 Q4 报表”。如果直接用 dense，可能召回“季度财务分析”“2025 年经营总结”“APAC 财报说明”，但这些不一定是 Q4 报表本身。更稳妥的做法是：

1. 先用 sparse 找出含 `2025`、`Q4`、`报表` 的候选。
2. 再用 dense 在候选中补充语义相关背景。
3. 再按 `year=2025`、`region=APAC`、`access_level=user_scope` 做过滤。
4. 最后用 reranker 排最终顺序。

所以 Dense Retrieval 的适用边界很明确：当你的主要目标是“语义召回”时，它是核心工具；当你的主要目标是“硬约束精确命中”时，它必须退到组合系统中的一个组件，而不是唯一方案。

---

## 参考资料

1. EmergentMind，《Dense Retrieval Models》，关于双编码器、MIPS、ANN 与向量检索公式的整理。  
2. EmergentMind，《Dense-Sparse Hybrid Retrieval》，关于 dense 与 sparse 融合、加权与排序策略。  
3. Unstructured，《RAG Systems Best Practices: Unstructured Data Pipeline》，关于 chunking、metadata 与增量更新。  
4. Dextra Labs，《Implementing Retrieval-Augmented Generation with Real-World Constraints》，关于分片索引、合规和延迟权衡。  
5. AILog，《Dense Retrieval Guide》，关于生产级 dense retrieval 流程、chunk 切分与向量索引实现。

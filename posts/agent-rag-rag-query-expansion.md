## 核心结论

RAG 的查询扩展，本质是在“检索之前”把一个用户问题拆成多个更容易命中文档的查询视角。这里的“检索”指从知识库里找相关资料；“扩展”不是把问题写得更长，而是让系统同时覆盖同义词、上下位词、时间约束、部门标签、安全标签等信息。它成立的原因很直接：真实知识库的写法不统一，用户提问也常常过于笼统，单个 query 往往只能命中知识的一部分。

在智能体级 RAG 中，查询扩展通常不是一个孤立技巧，而是 pre-retrieval 阶段的一部分。所谓 pre-retrieval，就是正式检索前的准备步骤。智能体会先判断问题是否模糊、是否缺实体、是否需要时间范围、是否需要安全过滤，再决定生成多少个扩展 query，并把这些 query 并行发给向量检索和关键词检索。向量检索就是按语义相似度找内容相近的文档；关键词检索就是按字面词项匹配找明确提到这些词的文档。两者互补，才能提升召回率。

它的主要收益不是“答案更聪明”，而是“证据更完整”。RAG 最怕的不是模型不会写，而是根本没把关键文档找出来。查询扩展优先解决这个问题，所以它通常先提升 recall，再通过 rerank 控制 precision。recall 可以理解为“该找到的有没有找到”；precision 可以理解为“找到的结果里有多少真的相关”。

下面这个表可以先建立直观感觉：

| 场景 | 扩展方式 | 直接收益 |
|---|---|---|
| 普通宽泛问题 | 补充关键词、时间、部门 metadata | 减少原始 query 过泛导致的漏召回 |
| 多术语同义问题 | 生成同义词、产品别名、英文名 | 提升跨团队文档命中率 |
| 混合检索场景 | 同时跑向量 query 与关键词 query | 兼顾语义覆盖与字面精确匹配 |
| 安全敏感场景 | 加入权限、部门、版本过滤条件 | 降低错误文档进入上下文的概率 |

玩具例子：用户问“公司 Copilot 的安全策略有哪些？”如果只搜这句，可能拿到泛化介绍页，漏掉“Copilot Studio 安全默认值”“内部数据边界”“2025 安全基线”这类关键文档。扩展后可以并行生成“Copilot 安全策略 2025”“Copilot Studio 安全默认”“Copilot 数据隔离 部门规范”等 query，候选集合会明显更完整。

---

## 问题定义与边界

查询扩展要解决的问题，不是“让提问更好看”，而是“让检索输入更适合知识库的组织方式”。知识库中的文档标题、正文术语、缩写、版本号、部门标签，通常和用户自然语言提问并不完全一致。用户说“产品性能监控怎么做”，文档可能写的是“APM 仪表盘配置”“可观测性基线”“SLO 告警策略”。如果不扩展，检索系统很可能只命中“性能”而漏掉“可观测性”与“SLO”。

边界也很明确。第一，查询扩展不能无限生成。query 数量一旦膨胀，延迟、成本、重复候选和错误上下文会一起上升。第二，metadata 必须真实。metadata 就是附加给检索的结构化标签，例如部门、时间、权限、产品线。如果系统胡乱补“2025”“安全策略”“财务部”，会把不相关文档推高。第三，它不能替代事实校验。扩展只提升“找到证据”的概率，不保证“证据被正确理解”。

一个简单判断标准是：当问题满足以下任一条件时，适合做扩展。
1. 术语可能有多种写法。
2. 问题缺少业务上下文。
3. 问题包含时间、版本、地区、权限等隐含条件。
4. 需要同时覆盖关键词匹配和语义相似匹配。

玩具例子可以写成这样：原始问题是“产品性能监控怎么做”。系统扩展成“产品性能实时监控 仪表盘”“应用性能监控 APM 告警策略”“2025 性能基线 安全要求”。这不是在“猜答案”，而是在补足检索条件。

真实工程例子：企业内部知识库里，安全团队文档写“安全基线”，平台团队写“默认配置”，业务团队写“最佳实践”。用户问“Copilot 的安全策略有哪些”，如果没有扩展，搜索会偏向某一团队的话术。加入部门标签、产品子模块、年份限制后，系统更容易把跨团队文档一起召回。

---

## 核心机制与推导

典型流程是：原始 query 进入重写器，生成多个扩展 query；每个 query 分别走关键词检索和向量检索；然后把多路结果融合，再选 top-k 文档送给 LLM。这里的“融合”不是简单拼接，而是做排序合并。最常见的方法之一是 RRF，中文可理解为“倒数排序融合”。

它的核心公式是：

$$
\mathrm{score}(d)=\sum_{q}\frac{1}{k+\mathrm{rank}_q(d)}
$$

其中，$d$ 是文档，$\mathrm{rank}_q(d)$ 是文档在某一路 query 结果里的排名，$k$ 是平滑常数，工程上常取 60。这个公式的含义是：如果一个文档在多路检索中都靠前，它的总分就会更高；如果它只在某一路特别靠前、其他路完全没有出现，它不会被无条件放大。

为什么这种方法适合查询扩展？因为扩展后的多个 query，本来就代表多个视角。真正高价值的证据，通常会在多个视角里都被命中。RRF 恰好把“多路一致出现”转成更高分数。

看一个最小数值例子。假设某文档在“向量检索 + 扩展 query A”里排名第 2，在“关键词检索 + 扩展 query B”里排名第 5，取 $k=60$，则：

$$
\mathrm{score}(d)=\frac{1}{60+2}+\frac{1}{60+5}\approx 0.01613+0.01538=0.03151
$$

这个分数不大，但关键不是绝对值，而是相对比较。另一个只在单一路里排名第 1 的文档，得分可能是 $\frac{1}{61}\approx 0.01639$。前者虽然没有任何一路拿到第一，但因为跨路稳定出现，最终更可能排在前面。这正符合 RAG 的目标：优先给 LLM 提供多证据一致的上下文。

需要注意，查询扩展不等于“越多越好”。从推导角度看，增加 query 数量会提高某些文档累计得分的机会，但也会让噪声文档在更多路里“偶然出现”。所以工程上通常会限制扩展条数，比如 3 到 8 条，再配合过滤、去重和 rerank。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实向量库，但把“重写、多路检索、RRF 融合、去重”的主干流程表达清楚了。

```python
from collections import defaultdict

DOCS = {
    "d1": "Copilot 安全策略 2025 数据隔离 部门权限",
    "d2": "Copilot Studio 安全默认 配置 访问控制",
    "d3": "产品性能监控 仪表盘 APM 告警",
    "d4": "可观测性 SLO 指标 延迟 错误率",
    "d5": "Copilot 简介 与 功能总览",
}

def rewrite_query(user_query, department=None):
    rewrites = [user_query]
    if "Copilot" in user_query:
        rewrites.append(f"{user_query} 2025")
        rewrites.append("Copilot Studio 安全默认")
    if "性能监控" in user_query:
        rewrites.append("APM 仪表盘 告警策略")
        rewrites.append("可观测性 SLO")
    if department:
        rewrites.append(f"{user_query} {department}")
    return rewrites[:5]

def keyword_search(query, docs):
    terms = query.split()
    scored = []
    for doc_id, text in docs.items():
        score = sum(term in text for term in terms)
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in scored]

def pseudo_vector_search(query, docs):
    # 用字符重叠数模拟“语义相近”
    qchars = set(query.replace(" ", ""))
    scored = []
    for doc_id, text in docs.items():
        overlap = len(qchars & set(text.replace(" ", "")))
        if overlap > 0:
            scored.append((doc_id, overlap))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in scored]

def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for ranked_docs in rank_lists:
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

def retrieve(user_query, department=None, top_k=3):
    queries = rewrite_query(user_query, department=department)
    rank_lists = []
    for q in queries:
        rank_lists.append(keyword_search(q, DOCS))
        rank_lists.append(pseudo_vector_search(q, DOCS))
    fused = rrf_fuse(rank_lists)
    return queries, fused[:top_k]

queries, results = retrieve("Copilot 安全策略 有哪些", department="安全部")
top_ids = [doc_id for doc_id, _ in results]

assert "Copilot 安全策略 有哪些" in queries
assert "d1" in top_ids
assert len(results) <= 3
print(queries)
print(results)
```

这段代码里有几个关键点。

第一，`rewrite_query` 负责控制扩展策略。真实系统中它可以来自规则、LLM、分类器，或者三者结合。对于零基础读者，先把它理解成“把模糊问题补成更像知识库语言的查询”。

第二，`keyword_search` 和 `pseudo_vector_search` 代表两条不同召回路径。真实工程里通常是 BM25 加向量索引。BM25 是一种经典关键词排序算法，适合处理明确术语；向量检索适合处理同义表达和语义近邻。

第三，`rrf_fuse` 负责把多路结果合并。它不关心原始分值是否同量纲，只看名次，这正是它工程上很稳的原因。不同检索器的原始分数往往不可直接比较，但名次可以融合。

真实工程例子可以写成一条流水线：
`Agent.ask(question) -> rewrite -> parallel retrieve -> filter -> RRF -> dedupe -> top-k -> LLM`
其中 `filter` 可以插入权限白名单、时间版本限制、脱敏规则；`dedupe` 用于消除同一文档不同切片的重复命中。

---

## 工程权衡与常见坑

查询扩展最常见的正面效果是 recall 上升，最常见的副作用是上下文膨胀。上下文膨胀就是送进 LLM 的文档太多、太重复、太杂，导致真正关键的信息被淹没。这一点对新手很重要：RAG 失败常常不是“没搜到”，而是“搜到太多且没收好”。

常见坑可以直接看表：

| 坑 | 现象 | 缓解措施 |
|---|---|---|
| 候选文档重复 | 同一主题片段反复进入 prompt | 文档级去重、段落级合并、限制 top-k |
| 扩展过度 | 延迟和成本上升，噪声变多 | 限制 query 数量，做 query 质量评估 |
| metadata 失真 | 错误部门、错误年份把结果带偏 | metadata 只来自可信结构化数据 |
| 旧信息泄露 | 过期政策被高分召回 | 版本过滤、时间衰减、有效期字段 |
| 检索对了但答案错了 | LLM 误读文档或错误归纳 | 增加 citation、verification、答案约束 |

一个典型失败模式是：系统把 query 扩展做得很复杂，却没有做好过滤。结果是“2023 旧版策略”“草稿规范”“无权限部门文档”一起进入上下文，模型反而更容易答错。也就是说，查询扩展必须和权限控制、版本控制、rerank 一起设计，不能单独上线。

另一个常见误区是把扩展当成纯生成问题。实际上它更像受约束的搜索优化问题。约束包括延迟预算、检索接口 QPS、上下文窗口大小、权限边界、审计要求。如果你的系统只能接受 300ms 检索延迟，那就不可能无限并行十几个 query。

对零基础工程师来说，可以记一个简单原则：先保证“少量、高质量、可解释”的扩展，再追求“更多、更广”。如果连为什么要生成某个扩展 query 都解释不清，这个扩展很可能不应该出现在生产系统里。

---

## 替代方案与适用边界

查询扩展不是唯一方案。它适合“问题模糊、术语多样、知识分散”的场景，但并不总是最优。

第一种替代方案是单 query + 强 rerank。也就是不扩展太多查询，而是先用一个较稳的 query 召回更多候选，再用更强的排序模型精排。这适合简单事实问答、延迟严格、知识库术语比较统一的系统。

第二种替代方案是 plan-based agent。plan-based 的意思是“先规划查询步骤，再逐步执行”。它不把问题一次性扩展成很多 query，而是先判断要查哪些子问题，再对每个子问题分别检索。这更适合财务审计、合规核查、多部门流程确认这种需要顺序推理和审计记录的场景。

下面做个对比：

| 方案 | 适用边界 | 优点 | 局限 |
|---|---|---|---|
| 单 query + rerank | 简单问题、低延迟要求 | 实现简单、成本低 | 对模糊问题漏召回明显 |
| Query extension | 多义、宽泛、跨术语问题 | 提升覆盖率，适合混合检索 | 易膨胀，需要强过滤 |
| Plan-based agent | 多步推理、审计敏感任务 | 可解释、可追踪 | 实现复杂，链路更长 |

真实工程里，这三者常常组合使用，而不是互斥。比如“确认 2025 年财务审计流程”这种任务，可以先由 plan-based agent 拆成“总流程、部门职责、时间节点、例外审批”四个子问题，再在每个子问题内部做少量 query extension。这样既保留步骤化控制，又能提升每一步的检索覆盖。

所以适用边界可以总结成一句话：查询扩展适合解决“单次检索视角不足”，不适合单独解决“多步规划、严格验证、复杂推理”。当问题本身需要顺序执行和状态跟踪时，单靠扩展 query 不够。

---

## 参考资料

- Pinecone. Retrieval-Augmented Generation. https://www.pinecone.io/learn/retrieval-augmented-generation/
- Microsoft Cloud Blog. Common retrieval augmented generation (RAG) techniques explained. 2025-02-04. https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/04/common-retrieval-augmented-generation-rag-techniques-explained/
- Azure AI Search Learn. Hybrid search ranking with Reciprocal Rank Fusion (RRF). https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- PromptQL Blog. Fundamental Failure Modes in RAG Systems. https://promptql.io/blog/why-rag-fails

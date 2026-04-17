## 核心结论

实体链接的候选生成，本质上是把文本里的 mention 先映射到一个足够小、但仍然包含正确实体的候选集合，再交给排序器做最终消歧。mention 可以理解为“文本里被识别出来、准备去链接的一段字符串”，比如“胰岛素”“ACE2”“苹果”。

对初学者，最容易理解的两阶段流程是：先用别名字典拿到一批可能实体，再用 cross-encoder 这种“把 mention、上下文、候选实体一起读”的精排模型选最匹配的那个。候选生成决定上限，精排决定最终选择；如果正确实体没进候选集，后面的模型再强也补不回来。

设知识库实体全集为 $\mathcal{E}$，候选生成函数为 $f_{CG}(m, x)\subseteq \mathcal{E}$，其中 $m$ 是 mention，$x$ 代表可用信号，例如别名、上下文、类型。若只保留前 $K$ 个候选，则数据集上的召回可写为：

$$
\mathrm{Recall@K}=\frac{1}{|D|}\sum_{(m,e^\*)\in D}\mathbf{1}\left[e^\*\in f_{CG}(m,x)_{1:K}\right]
$$

其中 $e^\*$ 是 gold 实体，也就是标注答案。两阶段系统的最终准确率满足：

$$
\mathrm{Acc}_{final}@K \le \mathrm{Recall@K}
$$

所以工程目标不是盲目增大 $K$，而是让 $\mathrm{Recall@K}\approx 1$ 的同时保持 $K$ 尽量小。原因很直接：$K$ 越大，后续 cross-encoder 需要打分的 mention-entity 对越多，延迟、显存和吞吐压力都会上升。

---

## 问题定义与边界

实体链接任务可以拆成四个对象：

| 对象 | 含义 | 例子 |
|---|---|---|
| mention | 文本中的待链接字符串 | “胰岛素” |
| 知识库 $\mathcal{E}$ | 可选实体全集 | 药物库、疾病库、通用百科库 |
| 候选集 $C_K$ | 候选生成输出的前 $K$ 个实体 | 20 个“可能是胰岛素”的实体 |
| gold 实体 | 正确答案 | `INSULIN (CHEBI:145810)` |
| 精排器 | 在候选集内选最终实体的模型 | cross-encoder、pairwise ranker |

数据流很固定：`mention + 上下文 -> 候选生成 -> 候选集 -> 精排 -> 最终实体`。候选生成不负责“最后选谁”，它只负责“不要把正确答案漏掉”。

玩具例子先看一个简单版本。句子是“患者开始使用胰岛素控制血糖”。系统先识别到 mention=`胰岛素`。候选生成可能先查别名字典，得到“Insulin”“Insulin therapy”“Insulin receptor”等别名相关实体；再用 BM25 在实体名称和定义里搜一次；最后保留前 20 个。精排器再根据上下文“控制血糖”判断这里更像药物或激素，而不是受体蛋白。

这里的边界要讲清楚：

1. 候选生成追求的是高召回，不是高精度。
2. 候选生成通常不直接输出单一答案，而是输出一个有限集合。
3. 如果知识库缺实体，候选生成再好也无法命中，这属于知识库覆盖问题，不是召回算法本身的问题。
4. 如果 mention 检测错了，比如把“ACE2 受体”只截成“受体”，后续候选生成也会被拖垮，这属于上游 NER 边界误差。

---

## 核心机制与推导

实际系统很少只靠一种召回方式，而是把多种高召回信号并起来。原因是不同方法覆盖不同失败模式。

第一类是别名字典。别名就是“同一个实体在文本里的不同叫法”。它对标准别名、缩写、规范词最有效，速度也最快。比如 mention=`IL-6`，字典能直接映射到 `Interleukin-6`。缺点是覆盖不到新写法、错拼、上下文依赖名称。

第二类是编辑距离。编辑距离可以理解为“两个字符串改成一样最少要几步”。它适合处理错拼、连字符变化、大小写变化，例如 `insuline` 召回 `insulin`，`covid19` 召回 `COVID-19`。但它只看字面，不看语义。

第三类是 BM25 或字符 n-gram 检索。BM25 可以理解为“关键词匹配打分器”，会考虑词频和区分度；字符 n-gram 则把字符串切成连续字符片段，比如 3-gram，把 `insulin` 切成 `ins`、`nsu`、`sul` 等。这两类方法对长尾别名、词序变化、部分重叠都很有用，常被用作第一阶段主力召回。

第四类是稠密向量检索。稠密向量就是“把 mention 和上下文编码成连续向量，再按向量相似度搜实体”。它能补上字面不近但语义很近的候选。比如 mention=`ACE2`，上下文是“病毒通过该受体进入肺泡上皮细胞”。即使字符串本身很短，向量检索也能借助上下文把候选往“血管紧张素转化酶 2”而不是其他缩写实体上拉。

这几类方法协同的思路是：

$$
C_K(m)=\mathrm{TopK}\left(
C_{alias}\cup C_{edit}\cup C_{bm25}\cup C_{char}\cup C_{dense}
\right)
$$

其中每一路先各自召回一批，再做去重、打粗分、截断。若把 gold 实体命中的事件记成 $A_i$，那么整体漏召回事件是 $\cap_i \overline{A_i}$。只要各路错误不完全重合，混合召回就能显著提高 Recall@K。工程上常见的效果不是“单路特别强”，而是“多路互补后漏召回少很多”。

玩具例子可以这样看。mention=`苹果`，上下文 A 是“苹果发布了新款 M 芯片”，上下文 B 是“他每天吃一个苹果”。字典检索会同时召回公司和水果；BM25 也可能混在一起；真正拉开差距的是上下文向量和类型约束。A 应该偏向组织机构，B 应该偏向食物。候选生成阶段如果已经利用上下文和类型削掉明显不对的候选，精排负担会小很多。

真实工程例子看生物医学。句子是“ACE2 expression increased in lung tissue after infection”。仅凭 mention=`ACE2`，知识库里可能有基因、蛋白、受体相关条目。生产系统常见做法是：先用别名字典和字符级索引召回标准名、同义词和缩写扩展；再用上下文向量检索补充语义相关候选；最后用类型约束保留“gene/protein”一类实体。这样可以把几百万实体的知识库，压缩到几十或上百个候选，再送给精排器。

---

## 代码实现

下面的代码不是工业级实现，但包含了候选生成的核心骨架：加载别名字典、做多路召回、合并去重、按分数排序，并验证 gold 是否进入前 $K$。

```python
from math import log

KB = {
    "E1": {"name": "Insulin", "aliases": ["胰岛素", "insulin"], "type": "drug"},
    "E2": {"name": "Insulin receptor", "aliases": ["胰岛素受体", "insulin receptor"], "type": "protein"},
    "E3": {"name": "ACE2", "aliases": ["ACE2", "血管紧张素转化酶2"], "type": "gene"},
    "E4": {"name": "Apple Inc.", "aliases": ["苹果公司", "Apple"], "type": "org"},
    "E5": {"name": "Apple (fruit)", "aliases": ["苹果", "apple fruit"], "type": "food"},
}

alias_index = {}
for eid, meta in KB.items():
    for a in meta["aliases"]:
        alias_index.setdefault(a.lower(), set()).add(eid)

idf = {
    "控制": 1.3, "血糖": 1.5, "受体": 1.4, "病毒": 1.2,
    "进入": 1.0, "肺泡": 1.6, "芯片": 1.7, "发布": 1.1
}

entity_docs = {
    "E1": "胰岛素 控制 血糖 激素 药物",
    "E2": "胰岛素 受体 蛋白 信号通路",
    "E3": "ACE2 病毒 进入 肺泡 受体 基因",
    "E4": "苹果 发布 芯片 公司 科技",
    "E5": "苹果 水果 食物",
}

dense_hint = {
    ("ACE2", "病毒通过该受体进入肺泡上皮细胞"): {"E3": 0.95, "E2": 0.35},
    ("苹果", "苹果发布了新款芯片"): {"E4": 0.92, "E5": 0.15},
}

def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]

def bm25_like_score(query_tokens, doc):
    doc_tokens = doc.split()
    score = 0.0
    for t in query_tokens:
        if t in doc_tokens:
            score += idf.get(t, 1.0)
    return score

def candidate_generate(mention, context, topk=3, allowed_types=None):
    scores = {}

    # 1) 别名字典精确召回
    for eid in alias_index.get(mention.lower(), set()):
        scores[eid] = scores.get(eid, 0.0) + 5.0

    # 2) 编辑距离模糊召回
    for eid, meta in KB.items():
        best = min(edit_distance(mention.lower(), a.lower()) for a in meta["aliases"])
        if best <= 2:
            scores[eid] = scores.get(eid, 0.0) + (2.5 - best)

    # 3) BM25 风格关键词召回
    q_tokens = context.split()
    for eid, doc in entity_docs.items():
        scores[eid] = scores.get(eid, 0.0) + bm25_like_score(q_tokens, doc)

    # 4) 稠密检索补充
    for eid, sim in dense_hint.get((mention, context), {}).items():
        scores[eid] = scores.get(eid, 0.0) + sim * 3.0

    items = []
    for eid, s in scores.items():
        if allowed_types and KB[eid]["type"] not in allowed_types:
            continue
        items.append((eid, round(s, 3)))

    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topk]

cands1 = candidate_generate("胰岛素", "控制 血糖", topk=3, allowed_types={"drug", "protein"})
cands2 = candidate_generate("ACE2", "病毒通过该受体进入肺泡上皮细胞".replace("。", ""), topk=3, allowed_types={"gene", "protein"})
cands3 = candidate_generate("苹果", "苹果发布了新款芯片", topk=2, allowed_types={"org", "food"})

assert any(eid == "E1" for eid, _ in cands1)
assert any(eid == "E3" for eid, _ in cands2)
assert cands3[0][0] == "E4"
```

这段代码对应的工程流程可以概括成：

1. 用别名字典做高置信直达召回。
2. 用编辑距离补错拼和形态变化。
3. 用 BM25 或字符检索覆盖长尾字面匹配。
4. 用上下文向量检索补语义相近但字面不近的候选。
5. 合并、去重、加权打分。
6. 用类型约束、黑名单、长度规则做过滤。
7. 截断到前 $K$，交给精排接口。

和精排器对接时，接口通常长这样：输入 `(mention, context, [candidate_1, ..., candidate_K])`，输出每个候选的相关性分数。候选生成模块要保证格式稳定、候选实体元数据完整，并且尽量让每个候选都带上标题、别名、定义、类型这些精排特征。

---

## 工程权衡与常见坑

候选生成最常见的权衡是 Recall@K 与吞吐的平衡。$K$ 太小，gold 容易漏掉；$K$ 太大，cross-encoder 成本急剧上升，而且大候选池会引入更多极难负样本和噪声候选，实际精排效果可能反而下降。

下面给一个工程化的示意表，数值用于说明趋势：

| 候选数 K | Recall@K | 单条 mention 精排延迟 | 每秒吞吐 |
|---|---:|---:|---:|
| 50 | 0.91 | 18 ms | 220 |
| 100 | 0.95 | 35 ms | 112 |
| 200 | 0.96 | 69 ms | 57 |
| 300 | 0.961 | 103 ms | 38 |

新手可以把它理解成：把 $K$ 从 300 减到 100，cross-encoder 批次处理速度可能接近翻倍，但召回只掉大约 1 个百分点。若业务要求是高吞吐在线服务，这通常是值得的；若业务是离线知识库构建，可能会接受更大的 $K$。

常见坑主要有五类。

第一，别名字典脏。知识库别名里如果混入停用词、通用词、旧缩写，会导致候选爆炸。比如把“IT”“CAT”直接当强别名，会让很多 mention 失控。

第二，只做精确匹配。真实文本里存在错拼、简称、连字符变化、大小写变化，只靠 exact match 会漏大量 gold。

第三，只做稠密检索。稠密方法在零样本和语义泛化上强，但对标准代码、化学式、缩写、版本号这类高度字面敏感场景，稀疏召回往往更稳。

第四，类型约束过硬。类型预测错了，会把 gold 直接过滤掉。实践里更稳的办法通常是“软约束加分”或“保留少量跨类型候选”，而不是一刀切。

第五，评估口径混乱。候选生成阶段应优先看 Recall@K、Mean Candidate Size、平均延迟，而不是直接拿最终 F1 评价。否则你分不清问题出在候选生成，还是出在精排。

真实工程例子里，RAG 或知识图谱入库流水线常见做法是：第一阶段用别名索引 + BM25/字符 n-gram 把知识库压到大约 100 个候选；第二阶段用 cross-encoder 打分。如果把第一阶段放宽到 500 个候选，理论上召回可能略涨，但线上 GPU 批次更难打满，尾延迟会明显变差。

---

## 替代方案与适用边界

候选生成没有唯一标准答案，关键看知识库规模、实体类型和线上约束。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 仅别名字典 | 极快、可解释 | 长尾和新写法召回差 | 术语规范、别名维护强的领域 |
| 别名 + BM25/字符检索 | 便宜、稳定、覆盖面广 | 语义歧义处理弱 | 多数工业系统的默认起点 |
| 仅稠密检索 | 上下文泛化强，适合 zero-shot | 对字面细节敏感任务不稳 | 开放域、长文本上下文丰富 |
| 混合召回 | 召回最高、抗失败模式更强 | 系统更复杂，维护成本更高 | 中大型生产系统 |

如果知识库很规范，比如企业内部产品库、药品标准词表，别名字典加字符检索通常已经够用。如果是开放域百科、零样本实体链接、跨领域问答，往往需要稠密检索补足语义覆盖。研究结果也表明，在 Zero-shot EL 设定下，仅用稠密候选生成并加入类型信号，在 $K=50$ 时也能达到 84.28% 的 gold recall，这说明小规模候选并不必然意味着低召回。

但这不代表“以后都只用稠密检索”。边界在于：

1. 字面精确性极强的场景，稀疏和字符召回依旧必要。
2. 上下文很短甚至没有上下文时，稠密检索优势会下降。
3. 知识库定义字段很差、别名稀少时，向量空间质量也会受影响。
4. 极低延迟场景下，纯规则或纯稀疏索引更容易落地。

工程上最稳的结论是：先用简单混合召回把 Recall@K 做到足够高，再决定是否继续引入更重的稠密模块，而不是一开始就把系统做成全稠密、全神经、全黑盒。

---

## 参考资料

1. Evan French, Bridget T. McInnes. *An overview of Biomedical Entity Linking throughout the years*. 这篇综述系统梳理了生物医学实体链接的发展，明确指出候选生成常采用 BM25、Lucene、编辑距离、表示相似度和语义类型约束等高召回手段。  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC9845184/

2. Haodi Ma, et al. *A Comprehensive Evaluation of Biomedical Entity Linking Models*. 这篇评测把实体链接明确拆成 CG 与 NED 两阶段，并强调候选生成决定是否把正确实体送入后续消歧阶段。  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11097978/

3. Eleni Partalidou, et al. *Increasing Entity Linking upper bound through a more effective Candidate Generation System*. 这篇工作展示了在 Zero-shot EL 中，基于稠密双编码器并加入类型信息，可以在仅保留 50 个候选时达到 84.28% 的 gold recall。  
   https://openreview.net/forum?id=gG0lBcnXCQw

4. Mathew Jacob, et al. *Drowning in Documents: Consequences of Scaling Reranker Inference*. 这篇预印本讨论了 reranker 在大规模候选上会出现质量下降，支持“候选集不能盲目做大”的工程结论。  
   DOI: https://doi.org/10.48550/arXiv.2411.11767

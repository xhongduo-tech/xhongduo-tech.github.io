## 核心结论

RAG 中的 `chunk` 是检索前预先切好的最小知识单元。白话说，模型不是直接在整篇文档里找答案，而是在很多小片段里挑最相关的片段再拼进提示词。这个切法会直接决定 Agent 能不能“捞到对的上下文”。

结论先给出：

1. `chunk` 策略通常比“换一个更强的大模型”更先影响 Agent 的检索质量。检索没拿对，后面的生成再强也只能在错误上下文上继续推理。
2. 固定长度分块最容易实现，但最容易把完整语义拆断。比如一个 FAQ 的问题在上一个 chunk，答案在下一个 chunk，检索命中时就只拿到半截。
3. 递归分块是工程上的常见折中。它优先保留段落、标题、句子边界，再补 token 长度约束，通常能在复杂度和效果之间取得平衡。
4. 语义分块会额外计算句子之间的相似度，用“主题有没有跳变”来决定何时新开 chunk。代价是预处理更慢，但在高召回场景里常常更值。
5. 在公开案例和经验总结里，语义或结构化 chunk 相比固定 512 token 切分，`Recall@5`、`nDCG`、grounded answer rate 常见可提升约 12% 到 30%；研究摘要里给出的典型数字是 Recall@5 提升约 18 个百分点。

一个最小对比可以先看：

| 策略 | 基本规则 | 优点 | 缺点 | 典型场景 |
| --- | --- | --- | --- | --- |
| 固定长度分块 | 每 256/512/1024 token 切一次 | 简单、快、稳定 | 容易截断语义 | FAQ、格式规整手册 |
| 递归分块 | 先按段落/句子，再按长度补切 | 保留结构，成本适中 | 对隐含主题切换不敏感 | 大多数通用文档 |
| 语义分块 | 按句向量相似度判断是否换块 | 语义完整，召回更高 | 预处理慢，实现复杂 | 合同、法规、论文、知识库 |

玩具例子很直观。假设文档里有两句：

- “退货条件：商品需未拆封。”
- “退款时效：审核后 3 个工作日到账。”

如果固定 512 token 切分时刚好把这两句和上下文一起切乱，检索“退款多久到账”时，可能拿到“退货条件”附近的片段而不是“退款时效”。语义分块会更倾向于把“退款时效”相关句子聚在一起，检索返回的片段更完整。

---

## 问题定义与边界

这里的问题不是“怎么把文档切开”，而是“怎么切，才能让向量检索和后续提示词看到的是同一段完整意思”。

更严格地说，`chunk` 有两个边界：

1. 语义边界：一个 chunk 内最好只包含同一主题或同一局部论点。
2. 长度边界：一个 chunk 不能无限大，因为嵌入模型、向量库和 LLM 上下文窗口都有成本限制。

如果切分太粗，多个主题会被塞进同一个 chunk，向量表示被平均化，检索时难以精确对齐问题。  
如果切分太细，相关信息又会被打散，需要 top-k 返回多个碎片，Agent 还要自己拼装，容易漏掉关键约束。

因此 chunk 策略本质上是在优化下面这个张力：

$$
\text{Chunk Quality} \approx f(\text{semantic coherence}, \text{size constraint}, \text{retrieval cost})
$$

其中：

- `semantic coherence` 是语义一致性，白话说就是“这一块是不是在讲同一件事”。
- `size constraint` 是长度约束，白话说就是“不能长到影响检索和上下文装载”。
- `retrieval cost` 是检索和预处理代价，白话说就是“别为了效果把系统拖慢到不可用”。

实践里常见的长度范围是 256 到 1024 token。研究摘要里提到的经验值是：语义分块判断主题跳变时，同时保持 `chunk_size ≤ 512~1024`，这是因为检索得到的 chunk 还要进 prompt，太长会浪费上下文预算。

一个真实工程例子是合同问答。用户问“违约后的赔偿责任上限是什么”。如果合同被机械地按 512 token 切开，“赔偿责任”条款可能在 chunk A，“责任上限”定义在 chunk B。检索 top-3 只命中 A 时，Agent 会给出不完整答案。这个错误不是模型不理解法律，而是输入上下文先天缺损。

所以本文的边界也很明确：

- 讨论对象是文档预处理阶段的 chunk 策略。
- 不展开重排器、混合检索、多跳检索的细节。
- 默认下游是“embedding 检索 + top-k chunk 注入 prompt”的典型 RAG 管线。
- 关注指标是 `Recall@k`、`nDCG` 和 grounded answers，而不是开放式生成文风。

---

## 核心机制与推导

语义分块的核心机制可以写成一句话：逐句编码，比较相邻句子的相似度，检测主题是否跳变。

设句子序列为 $s_1, s_2, \dots, s_n$，嵌入模型输出向量 $v_i = E(s_i)$。  
相邻句子的余弦相似度定义为：

$$
\cos(v_{i-1}, v_i)=\frac{v_{i-1}\cdot v_i}{\|v_{i-1}\|\|v_i\|}
$$

如果这个值高，说明两句讲的是相近主题；如果这个值低，说明很可能发生了主题切换。于是可以定义一个阈值 $\tau$：

$$
\text{new\_chunk}(i)=
\begin{cases}
1, & \cos(v_{i-1}, v_i) < \tau \\
1, & \text{size}(chunk) > L \\
0, & \text{otherwise}
\end{cases}
$$

其中：

- $\tau$ 是相似度阈值，常见经验范围约为 0.6 到 0.8。
- $L$ 是 chunk 最大长度，常见为 512 到 1024 token。

白话解释就是：每来一句话，都问一次“它和上一句还在说同一件事吗”。如果不像了，或者当前块已经太大，就另起一个 chunk。

这个机制为什么有效？因为 embedding 检索依赖向量表示。如果一个 chunk 只包含一条完整思想，它的向量更容易和用户问题对齐；如果一个 chunk 里混了多个主题，向量就会被“拉平均”，检索信号变弱。

可以把机制理解成下面这条链路：

`句子 -> embedding -> 相似度判断 -> chunk_id -> 向量入库 -> 检索 -> prompt`

这里 `chunk_id` 是块的唯一标识，白话说就是“这段文本在知识库里的编号”。工程上还应该同时记录 `embedding_version` 和 `threshold`，否则以后调了阈值重跑分块，会很难回滚和比对。

再看一个玩具例子。文档句子如下：

1. “Redis 支持字符串、列表、哈希等数据结构。”
2. “Redis 适合高频读写和低延迟缓存。”
3. “向量数据库用于相似度检索。”
4. “RAG 系统会把 chunk embedding 写入向量库。”

前两句主题是 Redis，后两句主题是向量检索。固定分块可能把 2 和 3 塞在一起；语义分块则更可能形成：

- chunk A: 1, 2
- chunk B: 3, 4

这样用户问“RAG 为什么需要向量库”时，检索命中的 chunk B 更纯净。

---

## 代码实现

下面给一个可运行的简化 Python 示例。它不依赖真实 embedding API，而是用一个小词袋向量模拟“语义相似度”，目的是把核心逻辑讲清楚。

```python
import math
import re

VOCAB = [
    "退款", "到账", "退货", "条件", "缓存", "redis",
    "向量", "检索", "合同", "赔偿", "责任", "上限"
]

def tokenize(text: str):
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())

def embed(text: str):
    tokens = tokenize(text)
    return [tokens.count(word) for word in VOCAB]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def semantic_chunk(sentences, threshold=0.3, max_sentences=2):
    chunks = []
    current = [sentences[0]]
    prev_vec = embed(sentences[0])

    for sent in sentences[1:]:
        vec = embed(sent)
        sim = cosine(prev_vec, vec)

        if sim < threshold or len(current) >= max_sentences:
            chunks.append(current)
            current = [sent]
        else:
            current.append(sent)

        prev_vec = vec

    if current:
        chunks.append(current)
    return chunks

sentences = [
    "退货条件是商品未拆封",
    "退款到账通常需要三个工作日",
    "Redis 适合做缓存",
    "向量检索用于RAG"
]

chunks = semantic_chunk(sentences, threshold=0.2, max_sentences=2)

assert len(chunks) >= 2
assert "Redis 适合做缓存" in chunks[-1] or "Redis 适合做缓存" in chunks[-2]
assert any("退款到账通常需要三个工作日" in chunk for chunk in chunks)

print(chunks)
```

这段代码展示了最小闭环：

1. 先把文档拆成句子。
2. 每句生成向量。
3. 计算当前句与前一句的相似度。
4. 如果相似度低于阈值，或者当前块太长，就新开一个 chunk。
5. 输出 chunk 列表。

在真实工程里，流程通常会再多几步：

| 固定分块 | 语义分块 |
| --- | --- |
| 按 token 窗口切 | 先切句，再算 embedding |
| 可直接批量处理全文 | 需要逐句或逐段比较相似度 |
| metadata 较少 | metadata 要记录阈值、版本、父文档位置 |
| ingestion 快 | ingestion 慢，但召回通常更高 |

真实工程版的 chunk 对象建议至少带这些字段：

```python
chunk = {
    "chunk_id": "doc123#chunk07",
    "doc_id": "doc123",
    "text": "...",
    "start_sentence": 42,
    "end_sentence": 47,
    "threshold": 0.72,
    "embedding_version": "bge-m3-v2",
}
```

真实工程例子可以看企业知识库。假设每天新增 500 份制度、产品说明和工单总结。原先固定 512 token 切分，预处理 1 小时完成，但客服 Agent 对“退款时效”“责任边界”“例外条件”类问题频繁答非所问。改成递归 + 语义边界后，预处理变慢，但 top-5 命中的片段更完整，Agent 输出中的“引用原文可追溯性”明显变好。这类收益通常直接体现在 grounded answer rate 上。

---

## 工程权衡与常见坑

chunk 策略没有“永远最优”的答案，只有按文档类型、延迟预算和召回目标做权衡。

先看典型权衡：

| 方案 | Recall 表现 | ingestion 时间 | 实现复杂度 | 主要风险 |
| --- | --- | --- | --- | --- |
| 固定 512 token | 基线 | 1x | 低 | 语义断裂 |
| 递归字符分块 | 中高 | 1.2x~1.8x | 中 | 对隐含主题切换不够敏感 |
| 语义分块 | 高 | 2.5x~3x，极端可更高 | 高 | 处理慢、版本管理复杂 |

常见坑主要有五类。

第一，chunk 太小。  
太小会把完整上下文撕裂，导致检索结果碎片化。用户问“赔偿责任上限及例外条款”，top-k 可能只返回“上限”，没返回“例外”，回答就会过度简化。

第二，chunk 太大。  
太大会让单个向量混入多个主题，检索精度下降。即使 top-1 命中，LLM 也要在长文本里自己找重点，容易漏条件句。

第三，只调模型，不调 chunk。  
很多 Agent 表现差，根因不是 LLM 推理弱，而是 RAG 检索喂错了上下文。先检查 chunk 质量，通常比直接升级模型更便宜。

第四，语义分块没有缓存。  
语义分块要大量调用 embedding。如果每次重建索引都全量重算，吞吐会非常差。应当按句子内容哈希做 embedding cache，并支持 batch 请求。

第五，没有记录版本。  
阈值从 0.7 改到 0.75，或者 embedding 模型从 `v1` 升到 `v2`，chunk 边界就可能整体变化。如果不记录 `chunk_id`、`threshold`、`embedding_version`，线上效果回退时几乎无法定位原因。

下面这个表可以帮助做实验：

| chunk_size / 策略 | 处理时间 | Recall@5 倾向 | 适合文档 |
| --- | --- | --- | --- |
| 256 固定 | 低 | 中 | 短 FAQ |
| 512 固定 | 低 | 中 | 规整说明书 |
| 512 递归 | 中 | 中高 | 普通知识库 |
| 512~1024 语义 | 高 | 高 | 合同、法规、论文 |

一个很实际的工程案例是：原来 500 份文档固定 chunk 需要 1 小时；改成语义分块后涨到 2.5 小时；加入“句子切分缓存 + embedding batch + 增量更新”后，重新压回约 1.2 小时。这说明语义分块慢，不代表不可落地，关键在于是否把预处理链路工程化。

---

## 替代方案与适用边界

如果把三种主流策略放在一起看，选择标准其实很明确：你的目标到底是“先上线”，还是“高召回”。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 固定长度分块 | FAQ、格式高度统一的内容 | 实现最简单，成本最低 | 常把问答、定义、条件拆断 |
| 递归字符分块 | 通用知识库、博客、产品文档 | 保留标题和段落结构，效果稳定 | 遇到主题在段内突变时不够敏感 |
| 语义分块 | 法规、合同、论文、复杂工单 | 最能保持思想完整，召回高 | 依赖 embedding，预处理慢 |

可以用一句工程化判断来选：

- 如果文档天然结构清晰，先用固定或递归分块。
- 如果问题通常依赖局部精确语义，比如“例外条款”“责任边界”“限制条件”，优先考虑语义分块。
- 如果系统是高吞吐、低成本优先，递归分块通常是更稳妥的默认值。
- 如果系统目标是高价值问答、低容错，语义分块比盲目升级 LLM 更值得先做。

还有两个常见替代思路。

一是“固定分块 + overlap”。`overlap` 是重叠窗口，白话说就是前后 chunk 留一部分重复内容，减少被截断的概率。它能缓解边界问题，但不能真正识别主题切换。

二是“递归分块 + reranker”。先用较便宜的结构化分块保证吞吐，再用重排模型提高 top-k 质量。这个组合常常比纯语义分块更适合线上系统，因为它把成本放在查询时，而不是索引时。

所以适用边界不是“语义分块一定最好”，而是“当召回损失主要来自错误边界时，语义分块最值得投入”。

---

## 参考资料

| 资料 | 主要主张 | 贡献 |
| --- | --- | --- |
| Qdrant chunking strategies | chunk 是检索最小单元，边界决定向量对齐质量 | 给出固定、递归、语义分块的机制框架 |
| 阿里云相关技术文章 | 递归与语义分块更适合保留结构和主题完整性 | 对新手解释清楚，强调速度与质量权衡 |
| Amir Teymoori, *RAG Text Chunking Strategies* (2025) | 语义边界分块可带来约 12%~18% 检索提升 | 提供实验型结论，适合做效果参考 |
| OptyxStack case study | 结构化/语义 chunk 可提升 Recall、nDCG、grounded answer | 说明真实工程里 chunk 改造能直接提升 Agent 可用性 |
| David Koh, *The Art of the Split* | chunk 切法决定检索行为，而非纯粹文本裁剪 | 强调 chunk 是检索系统设计的一部分 |

这些资料的共同结论很一致：RAG 的失败常常不是“模型不会答”，而是“系统没把正确片段拿给模型”。其中最值得复查的数据点，是语义分块相对固定分块在 Recall@5 上的提升，以及其带来的 2.5 倍到 3 倍以上预处理成本。这个对比恰好说明，chunk 策略是一个典型的工程优化问题，不是抽象的文本清洗问题。

参考链接：

- Qdrant: https://qdrant.tech/course/essentials/day-1/chunking-strategies/
- 阿里云开发者社区: https://developer.aliyun.com/article/1685312
- Amir Teymoori: https://amirteymoori.com/rag-text-chunking-strategies/
- OptyxStack case study: https://optyxstack.com/case-studies/rag-low-recall-retrieval-miss

## 核心结论

私有知识库 RAG 的目标，不是“把所有文件丢给大模型”，而是把企业自有文档先变成可检索、可授权、可增量维护的数据层，再把检索结果作为上下文交给大模型生成答案。这里的 RAG，白话解释就是“先查资料，再让模型回答”。

它的最小闭环可以写成一条流水线：

`文件解析 → chunk 分块 → embedding 向量化 → vector store 存储 → 权限过滤检索 → LLM 生成`

其中真正决定效果和成本的，不是 LLM 本身，而是两件事：

1. 文档是否被正确解析，包括 PDF 正文、表格、图片 OCR。
2. chunk 策略和 embedding 模型是否合适。

一个典型工程例子是：某企业把安全手册 PDF 用 Mineru 解析，抽出章节、表格和图片文字，再按 POMA 风格的层级分块结合滑动窗口切片，使用 `chunk_size=500`、`stride=250` 生成向量，最后在检索时先做权限判断，只把授权用户可见的 chunk 送给问答系统。这样得到的不是“全公司共享的聊天机器人”，而是“带权限边界的企业知识问答”。

若文档总长度为 $T$，chunk 大小为 $C$，重叠为 $O$，滑动窗口产生的 chunk 数可近似写成：

$$
\text{chunk 数} \approx \left\lceil \frac{T}{C-O} \right\rceil
$$

这个公式直接决定索引规模、embedding 成本和召回覆盖率。

---

## 问题定义与边界

私有知识库 RAG 处理的是企业自有资料，不是开放互联网网页。输入通常包括 PDF、Excel 导出的表格、扫描图片、内部 Wiki、数据库导出文本。目标是让系统在不泄露隐私的前提下，回答“基于企业资料”的问题。

边界要先说清：

| 输入源 | 常见问题 | 解析工具 | 输出 |
|---|---|---|---|
| PDF 手册 | 双栏、页眉页脚、表格混排 | Mineru | 段落、标题、表格、图片引用 |
| 扫描图片/截图 | 无法直接复制文字 | OCR | 可检索文本 |
| 表格/报表 | 单元格关系容易丢失 | 表格提取器 | 结构化表格文本或 JSON |
| 数据库导出 | 字段名和业务语义脱节 | 自定义 ETL | 记录级文本 + 元数据 |

这里的“元数据”，白话解释就是“描述这段内容属于谁、来自哪里、版本是什么的附加信息”。

新手可以把它理解成这样：公司有一批厂商手册和安全规范，系统先用 OCR 和 PDF 解析把内容抽出来，再切成很多小段，每一段都带上 `doc_id`、`page`、`department`、`access_level` 之类的标签。用户提问时，检索层先判断“你能看哪些段”，只返回可访问 chunk，然后再交给 LLM 组织答案。

因此，本文讨论的边界包括：

1. 文档来源是企业自有资产。
2. 需要支持增量更新，而不是每次全量重建。
3. 需要权限控制，避免敏感 chunk 被检出。
4. 重点是检索增强问答，不展开训练私有大模型。

---

## 核心机制与推导

先看分块。chunk，白话解释就是“把长文档切成模型容易处理的小段”。如果切得太大，检索定位不准，成本也高；如果切得太小，上下文会断裂，答案容易缺信息。

最常见的做法有两类：

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 固定滑动窗口 | 按固定 token 长度切，并保留重叠 | 实现简单，适合长 PDF | 容易切断标题和语义边界 |
| 层级语义分块 | 先按标题、段落、表格等结构切，再补窗口 | 语义更完整 | 实现更复杂，依赖解析质量 |

文档长为 $T$，chunk 大小为 $C$，重叠为 $O$，则近似有：

$$
\text{chunk 数} \approx \left\lceil \frac{T}{C-O} \right\rceil
$$

这意味着窗口真正前进的步长不是 $C$，而是 $C-O$。例如 `C=500`、`O=250`，每次只前进 250 token。

玩具例子：一段 1000 token 文本，按 `chunk_size=500`、`stride=250` 切分。

- 第 1 段覆盖 `0..499`
- 第 2 段覆盖 `250..749`
- 第 3 段覆盖 `500..999`

你会得到 3 个 chunk，而不是 2 个。原因是中间保留了重叠区，避免一句话前半句在上一个 chunk、后半句在下一个 chunk 时丢失语义。

真实工程例子：制造厂商的安全手册每天都有小改动，通常只影响少量章节。如果采用增量索引，流程应是：

| 阶段 | 动作 | 输出 |
|---|---|---|
| 检测 | 比较时间戳、哈希或 CDC 记录 | 找到新增/修改/删除文档 |
| chunk | 只重切受影响文档 | 新旧 chunk 差异 |
| embed | 只对变动 chunk 重算向量 | 新向量 |
| upsert/delete | 写入或删除向量库记录 | 最新索引 |
| 版本记录 | 保存索引版本、提示词版本、镜像版本 | 可回滚状态 |

这里的 CDC，白话解释就是“记录数据哪里变了的机制”。它的工程价值是避免每天重建全库。实际项目里，每天可能只需要重建 5 到 20 个 chunk，向量库执行 upsert 后刷新缓存即可，不需要停机做全量 rebuild。

权限控制是另一条主线。ReBAC，白话解释就是“基于关系来判断谁能访问什么”。例如“张三属于安全部，安全部能看 A 文档的 2 级内容”。如果不在检索前做过滤，那么向量检索会先返回相似 chunk，再把它们传给 LLM，敏感内容已经泄露，后面再遮盖就晚了。

所以正确顺序应该是：

`用户身份 → 权限过滤候选集合 → 向量检索 → 重排 → LLM 生成`

而不是：

`向量检索 → 找到敏感 chunk → 再尝试拦截`

---

## 代码实现

下面给出一个可运行的简化版本，重点展示滑动窗口分块、增量更新和权限过滤。这里不依赖真实向量库，用内存结构模拟核心逻辑。

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    version: str
    access_level: int

def sliding_window(tokens: List[str], chunk_size: int = 500, stride: int = 250) -> List[Tuple[int, int, List[str]]]:
    assert chunk_size > 0
    assert stride > 0
    assert stride <= chunk_size

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + chunk_size, n)
        out.append((i, j, tokens[i:j]))
        if j == n:
            break
        i += stride
    return out

def fake_embed(text: str) -> List[float]:
    # 用长度和字符和模拟 embedding，真实工程中替换为 sentence-transformers/BGE/OpenAI
    return [float(len(text)), float(sum(ord(c) for c in text) % 1000)]

def can_access(user_level: int, chunk: Chunk) -> bool:
    return user_level >= chunk.access_level

def upsert_chunks(doc_id: str, version: str, tokens: List[str], access_level: int) -> Dict[str, Dict]:
    store = {}
    for idx, (start, end, part) in enumerate(sliding_window(tokens, chunk_size=500, stride=250)):
        text = " ".join(part)
        chunk = Chunk(
            chunk_id=f"{doc_id}:{version}:{idx}",
            doc_id=doc_id,
            text=text,
            start=start,
            end=end,
            version=version,
            access_level=access_level,
        )
        store[chunk.chunk_id] = {
            "chunk": chunk,
            "vector": fake_embed(text),
            "metadata": {"doc": doc_id, "version": version},
        }
    return store

# 玩具例子：1000 token -> 3 个 chunk
tokens = [f"t{i}" for i in range(1000)]
windows = sliding_window(tokens, chunk_size=500, stride=250)
assert len(windows) == 3
assert windows[0][:2] == (0, 500)
assert windows[1][:2] == (250, 750)
assert windows[2][:2] == (500, 1000)

# 权限检查
store = upsert_chunks("safety-manual", "v1", tokens, access_level=2)
sample_chunk = next(iter(store.values()))["chunk"]
assert can_access(2, sample_chunk) is True
assert can_access(1, sample_chunk) is False
```

如果接到真实工程里，可以把流程理解为下面的伪代码：

```python
chunks = chunker.parse(pdf).sliding_window(chunk_size=500, stride=250)
vectors = embedder.encode([c.text for c in chunks])

for chunk, vector in zip(chunks, vectors):
    if user.can_access(chunk):
        vector_store.upsert(
            id=chunk.id,
            vector=vector,
            metadata={"doc": chunk.doc_id, "page": chunk.page, "acl": chunk.acl}
        )
```

更完整的职责分工如下：

| 工具/库 | 作用 | 典型位置 |
|---|---|---|
| Mineru | PDF 解析、表格提取、图片 OCR | ingestion 层 |
| sentence-transformers / BGE | 生成 embedding 向量 | embedding 层 |
| OpenAI / Cohere / ChatNexus | 商业 embedding 或生成模型 | 托管能力层 |
| Pinecone / pgvector | 向量存储与检索 | retrieval 层 |
| SpiceDB | ReBAC 权限判断 | access control 层 |

工程里建议把 chunk 元数据至少设计成：

- `doc_id`
- `chunk_id`
- `version`
- `source_page`
- `department`
- `access_level`
- `created_at`
- `hash`

这样增量更新和权限排查才有依据。

---

## 工程权衡与常见坑

第一类权衡是 embedding 模型。

| 方案 | 优点 | 代价 | 适用场景 |
|---|---|---|---|
| 开源 embedding | 隐私好、成本低、可本地部署 | 调参与运维成本高 | 内网、敏感数据、预算有限 |
| 商业 embedding | SLA 稳定、即开即用、常见任务效果好 | 持续付费、数据出域风险 | 快速上线、对稳定性要求高 |

第二类权衡是索引更新方式。

| 方案 | 优点 | 风险 | 适用场景 |
|---|---|---|---|
| 全量重建 | 逻辑简单、一致性强 | 慢、贵、可能停机 | 小规模数据集 |
| 增量更新 | 快、成本低、适合持续变化 | 需要版本和删除策略 | 企业知识库、日更文档 |

常见坑主要有四个。

1. PDF 解析质量差。双栏、脚注、页眉页脚混进正文后，后续 chunk 再精致也没用。应优先保证解析层产出结构化段落和表格。
2. chunk 只看长度，不看结构。标题、表格说明、代码块被切断后，召回结果虽然“相似”，但回答不完整。
3. 检索前不做权限过滤。没有 ReBAC 或行级权限策略时，任意用户都可能通过相似查询拉出敏感 chunk。
4. 只做 upsert，不处理 delete。文档已删除或版本过期，但旧向量仍在库里，系统会回答陈旧内容。

真实工程里，制造厂商的增量索引通常每天只重建 5 到 20 个 chunk。这样做的关键不是“快一点”，而是让知识库持续新鲜，同时避免全量重建导致的检索中断。若再配合缓存刷新和版本标签，就能做到更新后快速生效、回滚也有路径。

---

## 替代方案与适用边界

没有一种 chunk 策略和模型组合适合所有文档。应根据文档结构、预算和合规要求选择。

| 方案 | 优点 | 适用场景 |
|---|---|---|
| POMA 层级 chunk | 保留标题、段落、表格结构，语义更完整 | 结构化文档、手册、制度文件 |
| 固定滑动窗口 | 简单稳定，容易量化成本 | 超长 PDF、格式混乱文本 |
| 开源 embedding + 本地向量库 | 数据不出域，成本可控 | 企业内网、隐私优先 |
| 商业 embedding + 托管检索 | 上线快，SLA 明确 | 需要快速交付的业务线 |

新手版选择建议可以直接写成：

- 想省成本、强调隐私：`sentence-transformers/BGE + 本地 vector store`
- 想追求托管能力和 SLA：商业 embedding + 托管向量库
- 文档层次清楚：优先层级语义分块
- 文档是长篇连续文本：优先滑动窗口

它的适用边界也要明确。私有知识库 RAG 适合“答案就在文档里”的任务，例如制度问答、产品手册问答、售后知识查询。若问题需要复杂计算、跨系统实时事务、或者答案根本不在文档中，仅靠 RAG 不够，需要接数据库查询、工作流引擎或外部工具。

---

## 参考资料

1. POMA 文档：说明 PDF 解析、结构识别与分块流程，适合理解“先解析再 chunk”的链路。  
2. DataCamp 关于 chunking 的文章：给出滑动窗口思路和 $ \lceil T/(C-O) \rceil $ 的近似公式，也提供了 1000 token 示例。  
3. Pinecone 关于 RAG 访问控制的资料：强调检索前权限过滤，说明向量检索不能替代授权系统。  
4. SpiceDB 相关资料：用于实现 ReBAC，把“用户-角色-资源”的关系落成可查询权限图。  
5. Mineru 项目资料：用于 PDF、表格、图片 OCR 解析，是 ingestion 层的典型工具。  
6. sentence-transformers / BGE 文档：用于本地 embedding 方案评估，适合隐私优先场景。  
7. OpenAI、Cohere、ChatNexus 等商业方案资料：用于比较托管 embedding 或企业检索服务的 SLA、成本和精度。  
8. 增量更新相关文章，如知识衰减与持续刷新实践：用于理解“只重建变化 chunk”而非全量重建的工程策略。  
9. AI Wiki、Git 与 Docker 版本化实践：用于记录 embedding、索引和提示模板版本，支持回滚与审计。

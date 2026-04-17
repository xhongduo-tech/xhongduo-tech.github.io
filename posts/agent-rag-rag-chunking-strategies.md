## 核心结论

RAG 的 Chunk 策略，本质上是在回答一个工程问题：**把原始文档切成多大、按什么边界切，才能让“检索到的片段”既足够准，又足够完整**。Chunk 可以理解为“送去做向量检索的文档片段”；如果这个片段本身语义不完整，后面的召回、重排、生成都会跟着失真。

最实用的结论有三个。

第一，**优先保证语义完整，再谈固定长度**。一个 chunk 最好是“能独立解释一个小事实”的最小单元，比如一个段落、一个小节、一个表格说明，而不是机械地从第 1 个 token 切到第 512 个 token。原因很直接：embedding 是“把文本压缩成向量表示”的过程，如果一个向量里混了多个主题，相似度检索就会变钝。

第二，**把上下文预算当成硬约束**。最常见的估算式是：

$$
context\_usage = chunk\_size \times top_k
$$

并要求：

$$
chunk\_size \times top_k < context\_limit
$$

这里 `chunk_size` 是单个片段的大致 token 数，`top_k` 是一次检索送给大模型的片段数，`context_limit` 是模型可接受的上下文上限。这个式子不是严格物理定律，因为系统提示词、用户问题、元数据、引用格式都会占 token，但它足够适合作为一阶近似。

第三，**overlap 或 parent-child 不是装饰，而是边界补偿机制**。overlap 就是“相邻 chunk 之间重复一小段内容”；它的作用是避免一句话刚好被切断，导致前一个 chunk 只有主语，后一个 chunk 只有结论。对大多数新系统，`512 token + 10%~15% overlap` 是一个常见起点，不一定最优，但便于验证。

一个最小数值例子：把一份约 2,048 token 的文档按 `512 token` 切分，并给每段保留约 `50 token` 重叠。直观上它会形成 4 个主 chunk。若检索时 `top_k=4`，则主占用约为：

$$
512 \times 4 = 2048
$$

在 `4096 token` 的上下文窗口内仍可放下。这就是为什么 chunk 策略不能脱离上下文预算单独讨论。

---

## 问题定义与边界

Chunking 是“把原始知识源切成适合 embedding 和检索的小段”的过程。这里的边界要说清楚：**它不是摘要，不是改写，也不是训练模型**；它只是决定“知识库里最小可检索单元长什么样”。

为什么这件事这么关键？因为 RAG 的回答质量，往往不是先败在大模型，而是先败在检索输入。如果 chunk 过大，一个向量里会同时包含定义、示例、例外条件、广告语、导航文本，查询命中时会出现“相关但不聚焦”。如果 chunk 过小，检索虽然可能命中关键词，却丢掉了前提条件和约束，最后让模型在残缺证据上生成答案。

可以用一个玩具例子看这个边界。

假设原文是：

“Redis 的持久化有 RDB 和 AOF。RDB 适合周期性快照，恢复速度通常更快；AOF 记录写命令，数据完整性通常更好，但文件可能更大。”

如果你按固定字数错误切开：

- chunk A：“Redis 的持久化有 RDB 和 AOF。RDB 适合周期性快照”
- chunk B：“恢复速度通常更快；AOF 记录写命令，数据完整性通常更好”

当用户问“为什么有人用 AOF”时，B 可能被检索到，但它没有“AOF 是什么”的前文；当用户问“RDB 有什么优点”时，A 可能被召回，但“恢复速度更快”恰好落在 B。问题不是模型不会回答，而是**证据单元被切坏了**。

新人可以先用下表建立直觉：

| chunk_size | 语义完整性 | 上下文覆盖 | 检索准确率 | 典型问题 |
|---|---|---|---|---|
| 小 | 弱，容易缺前因后果 | 弱，需要更多 top_k | 高精度但易碎 | 命中局部词，缺结论或条件 |
| 中 | 通常较平衡 | 中等 | 通常最好调参 | 需要少量 overlap 补边界 |
| 大 | 强，但容易混主题 | 强，单段信息多 | 召回广但不聚焦 | 向量语义被稀释，重排压力大 |

这里的“小、中、大”不是固定数字，因为法律文档、API 文档、FAQ、技术博客的自然结构完全不同。**真正的边界不是 token 数本身，而是“一个片段能否作为独立证据被引用”**。

---

## 核心机制与推导

Chunk 策略为什么成立，可以从两个约束推导出来：**向量表达约束**和**上下文预算约束**。

先看向量表达。embedding 可以理解为“把一句话或一段文字映射到高维空间中的一个点”。检索时不是按字面逐字匹配，而是在这个空间里找“意思相近”的点。如果一个 chunk 同时装了三个主题，比如“安装方式、错误排查、性能调优”，这个向量就变成多方向混合，和任何一个具体问题都只能“有点像”，这就是大 chunk 的稀释问题。

再看上下文预算。RAG 的流程不是只检索，还要把检索结果拼进 prompt 交给 LLM。因此 chunk 不是越完整越好，而是在完整和可装载之间找平衡。最常用的估算就是：

$$
context\_usage = chunk\_size \times top_k
$$

如果：

$$
context\_usage \ge context\_limit
$$

那就算召回正确，最终也塞不进模型窗口，或者只能截断，导致最相关证据反而被丢掉。实际工程中还应预留提示词、问题、系统规则、引用格式的空间，所以更安全的经验做法是：

$$
chunk\_size \times top_k \le 0.5 \sim 0.7 \times context\_limit
$$

这不是论文级严格界限，而是部署时更稳的预算方式。

继续看题目要求里的数值演示。文档长约 2,048 token，按 `512 token` 切分，配 `50 token overlap`。主 chunk 数可近似看作 4 段。若 `top_k=4`，主载荷约为 2,048 token，仍低于 4,096 token 窗口。此时 overlap 的意义不在于“增加信息量”，而在于**防止跨段句子失真**。也就是说，overlap 是为了修复分段带来的边界损伤。

真实工程例子比玩具例子更明显。假设你在做企业帮助中心，原始页面是 HTML 结构：

- H1: 退款政策
- H2: 企业版
- H3: 月付退款
- 表格: 退款时限、手续费、适用国家

如果只按固定长度切，检索到的片段可能只包含“30 天内可申请”，但丢了“企业版”“月付”“部分国家不适用”这些限制条件。更好的做法是按结构切：**以 H3 内容为主体，同时把 H1/H2 标题、表格原文、URL、锚点等 metadata 一起挂到 chunk 上**。这样向量既聚焦，又保留层级语义，生成答案时更容易引用正确条件。

因此，chunk 策略的核心机制可以概括为一句话：**用尽可能小、但仍然语义自洽的片段作为检索单元，再用 overlap、metadata、parent-child 去补偿局部片段缺少的全局上下文。**

---

## 代码实现

对新系统，基线方案应当足够简单：**固定大小切分 + overlap + 基本 metadata**。原因不是它一定最好，而是它最容易排查问题。你需要先知道系统在“能跑通”的情况下表现如何，再决定是否引入语义切分或父子检索。

下面是一个可运行的 Python 玩具实现。这里不用真正 tokenizer，而是用空格分词模拟 token，目的是把滑动窗口逻辑说清楚。

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start: int
    end: int

def fixed_chunk_with_overlap(text: str, chunk_size: int = 8, overlap: int = 2):
    assert chunk_size > 0
    assert 0 <= overlap < chunk_size

    tokens = text.split()
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(tokens), step):
        piece = tokens[i:i + chunk_size]
        if not piece:
            break
        chunks.append(Chunk(
            text=" ".join(piece),
            start=i,
            end=i + len(piece)
        ))
        if i + chunk_size >= len(tokens):
            break
    return chunks

sample = (
    "Redis persistence includes RDB and AOF . "
    "RDB is snapshot based and usually restores faster . "
    "AOF logs write commands and usually preserves changes better ."
)

chunks = fixed_chunk_with_overlap(sample, chunk_size=10, overlap=3)

assert len(chunks) >= 2
assert chunks[0].end > chunks[1].start   # overlap 生效
assert "RDB" in chunks[0].text
assert "AOF" in " ".join(c.text for c in chunks)

for c in chunks:
    print(c)
```

这个实现体现了三件事：

1. `step = chunk_size - overlap`，说明相邻窗口不是完全分离的。
2. 每个 chunk 记录 `start/end`，便于后续做高亮、回溯、去重。
3. `assert` 明确约束：`overlap` 不能大于等于 `chunk_size`，否则滑窗就会卡死或重复过多。

如果放到真实 ingestion pipeline，通常还会补两类元数据：

| 元数据 | 作用 | 例子 |
|---|---|---|
| 结构元数据 | 恢复文档层级 | `h1=退款政策, h2=企业版, h3=月付退款` |
| 定位元数据 | 便于引用和回溯 | `url, anchor, paragraph_id, table_id` |

真实工程里常见的做法是：先把 HTML、Markdown 或 PDF 解析成结构节点，再对每个节点内部做固定窗口切分。这样不是“固定切分”和“结构切分”二选一，而是两层组合：**先按结构定边界，再按长度控预算**。

---

## 工程权衡与常见坑

Chunk 策略最常见的失败，不是参数没调到最优，而是**根本没有建立评估闭环**。你不能只看“模型回答看起来还行”，而要看检索是否真正召回了支撑答案的证据。

先看常见坑：

| 坑 | 直接后果 | 规避方法 |
|---|---|---|
| 不设 overlap | 句子在边界断裂，证据残缺 | 加 10%~15% overlap，重点看长句和表格说明 |
| 全部文档统一固定切分 | FAQ、教程、API 文档被同一规则误伤 | 先按文档类型分类，再分别设策略 |
| chunk 太小 | precision 高但 recall 差 | 提高 chunk_size，或引入 parent-child |
| chunk 太大 | 向量混主题，检索不聚焦 | 按段落或标题切，减少单段主题数 |
| 不做评估就上线 | 参数只能凭感觉调整 | 建一个问答评估集，比召回率和答案引用率 |
| 忽略结构元数据 | 检索到正文却缺限制条件 | 把标题层级、表格、URL 一并存入 metadata |

一个典型坑就是“没有 overlap 的边界断裂”。例如：

- chunk A：“JWT 的 `exp` 字段表示令牌过期时间，服务端验证时应同时检查”
- chunk B：“签名是否合法与过期时间是否已到，否则会接受失效令牌”

用户问“为什么要检查 exp”，如果只召回 A，句子没说完；只召回 B，则连主语都没有。模型即使语言能力很强，也只能猜测。很多团队把这种回答错误归因于 hallucination，实际上是**证据切分缺陷**。

另一个现实权衡是成本。更细粒度的 chunk 往往意味着更多 embedding 数量、更多向量存储、更多召回噪声、更多重排开销。文献和行业文章里经常提到，细切分可能让索引规模膨胀到原来的数倍。于是一个务实策略通常是：先从 `512 token + 10%~15% overlap` 起步，建立评估集，再逐步加结构感知、语义切分、父子检索，而不是一开始就堆复杂方案。

还有一个值得注意的研究信号：有工作报告过 growing-window semantic chunking 在评估中让“进入生成阶段的回答概率”提升约 4%。这个数字本身不该被机械套用，因为语料、任务、检索器、重排器都不同，但它说明了一点：**chunk 不是静态预处理细节，而是可测量影响整体系统表现的核心变量**。

---

## 替代方案与适用边界

固定大小切分不是终点，而是基线。是否升级，取决于你的知识源结构和查询模式。

| 策略 | 适用条件 | 优点 | 代价 |
|---|---|---|---|
| fixed + overlap | 新系统、快速上线、文档结构一般 | 简单、稳定、易调试 | 对结构复杂文档不够聪明 |
| parent-child | 需要“小片段精准检索，大片段完整生成” | 检索准，生成时上下文更完整 | 索引和映射更复杂 |
| late chunking | 需要先保留全局语义再细分 | 有机会改善长文语义表达 | 实现复杂，依赖具体模型方案 |
| proposition-based | 文档里事实点很密，适合拆成命题 | 精细到“单事实证据” | 切分和标注成本高，噪声也高 |
| dynamic/query-aware | 查询差异很大，静态 chunk 不适配 | 可针对问题动态组织上下文 | 在线成本高，系统复杂度高 |
| structure-aware | HTML、Markdown、手册、知识库有明显层级 | 保留标题、表格、列表等结构 | 需要稳定解析器和元数据设计 |

什么时候该用 parent-child？当你发现“小 chunk 检索更准，但送给模型后信息不够”时。做法通常是：用小 chunk 建向量索引命中位置，再回表取对应的大段父节点供模型生成。这样检索层保持高精度，生成层又不至于因为证据太碎而失去上下文。

什么时候该用 structure-aware？当你的数据天然带层级，例如产品文档、SOP、帮助中心、法规页面。比如一个工业 HTML 站点，把 H3 正文作为主体，把 H1/H2、表格原文、图片说明、链接锚点一起写入 metadata。这样当用户问“企业版退款手续费怎么算”，召回结果不仅有句子本身，还有“这是企业版、月付、某国家范围内的规则”这些边界条件，生成可引用答案的概率会明显提高。

但并不是所有场景都值得上复杂策略。若你的知识库主要是短 FAQ，且每条本来就天然独立，那再做语义切分、命题切分，收益可能很小，反而增加索引噪声。**是否升级 chunk 策略，应该由评估结果驱动，而不是由术语新旧驱动。**

---

## 参考资料

- ByteTools, *RAG Document Chunking Strategies: Complete Guide for 2026*  
  https://bytetools.io/guides/rag-chunking-strategies
- AppearMore, *Chunking Strategies in Retrieval-Augmented Generation Architecture*  
  https://appearmore.com/geo-knowledge-base/retrieval-augmented-generation-rag/rag-architecture/chunking-strategies/
- ScienceDirect, *Optimising retrieval performance in RAG systems: A new growing window semantic chunking strategy...*  
  https://www.sciencedirect.com/science/article/pii/S0950705125019343
- Viqus, *RAG Chunking Strategies That Actually Work in 2026*  
  https://viqus.ai/blog/rag-chunking-strategies-2026

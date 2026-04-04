## 核心结论

企业知识库问答的核心，不是“把文档接进大模型”，而是把企业内部已经存在但分散、异构、权限不一致、版本不一致的知识，变成一条可审计的回答流水线。这里的 RAG，意思是 Retrieval-Augmented Generation，中文通常叫“检索增强生成”。它的基本动作不是“让模型直接回答”，而是先检索证据，再基于证据生成回答，并把出处一并返回。

对零基础到初级工程师，最重要的理解有三点。

1. 这个系统替代的不是传统搜索框本身，而是员工“先找资料，再读几份，再整理成一句可执行答案”的重复劳动。
2. 一个可用系统必须同时满足“能找对证据”和“不会越权引用”两个条件，只做其中一半都不够。
3. 大模型在这里主要承担“组织答案”的角色，不应该承担“凭记忆猜制度”的角色。

这件事为什么有业务价值，可以先看一个最小例子。假设公司里有三份资料：财务制度、差旅规范、OA 提交流程。员工问“如何申请报销？”。传统搜索通常返回几个文件名，用户还得自己打开阅读；知识库问答则会先找到最相关的几段材料，再生成一句可执行的话，例如：

“先在 OA 发起报销单，上传电子发票与审批截图；单笔金额超过 5000 元时，需要部门负责人复核。”

同时系统会附带出处，比如“财务制度第 3.2 节”“差旅规范第 2.1 节”“OA 指南第 4 节”。这样用户得到的不是一组候选文档，而是一段带证据的可操作答案。

下面这张表可以先把三类方案分开：

| 方案 | 输入 | 处理方式 | 输出 |
|---|---|---|---|
| 传统搜索 | 关键词，如“报销 发票” | 匹配标题、正文关键词 | 文档列表，用户自己阅读 |
| 人工答复 | 同事自然语言提问 | 人回忆经验、翻文档、整理答案 | 一段答案，但可能无出处、难复用 |
| AI 知识库问答 | 自然语言问题，如“如何申请报销？” | 查询重写、混合检索、权限过滤、重排序、生成回答 | 一段带引用的答案，可追踪、可复查 |

如果团队每周因为找资料浪费 12 小时，其中 40% 是重复问题，那么理论上可回收的时间约为：

$$
T_{saved} = 12 \times 40\% = 4.8 \text{ 小时/周}
$$

如果这是一个 10 人团队，则一周约为：

$$
10 \times 4.8 = 48 \text{ 人小时/周}
$$

按 8 小时工作日估算，约等于：

$$
48 \div 8 = 6 \text{ 个工作日/周}
$$

这也是为什么企业知识库问答的价值通常不是“展示模型能力”，而是回收组织的注意力。

---

## 问题定义与边界

“构建企业知识库问答”更准确的定义是：把企业内部多源知识转成统一索引，让系统能够基于自然语言问题检索相关证据，在权限和版本约束下生成带引用的回答。

这个定义里有四个边界，必须在开始阶段讲清楚。

第一，它解决的是“知识定位 + 证据整合 + 自然语言表达”的组合问题，不只是关键词查找。  
用户真正想问的往往不是“哪个文档提到报销”，而是“基于现行制度，报销超过 5000 元时审批链是什么”。前者只需要搜索，后者需要跨文档取证与整合。

第二，它处理的是结构化知识与非结构化知识的混合场景。  
结构化知识，白话说是字段明确、行列清晰的数据，比如 Excel、数据库表、工单状态；非结构化知识是大段自然语言文本，比如制度文档、会议纪要、聊天记录、售后记录。企业真实问题通常同时依赖两者，例如“某客户是否已完成验收并满足付款条件”，可能既要查合同条款，也要查工单状态。

第三，它不是万能问答。  
如果公司根本没有文档记录，只有少数老员工知道答案，那么系统很难稳定回答。RAG 更适合回答“企业已有依据但分散”的问题，不适合替代需要审批流、事务写入、强责任归属的业务系统。

第四，它必须处理权限。  
ACL 是 Access Control List，中文常翻成访问控制列表，可以直接理解成“谁能看什么”。企业知识库问答不是“搜到即可见”，而是“只允许检索和引用用户有权访问的内容”。如果系统先把无权内容喂给模型，再在最后一层“隐藏一下”，风险已经发生了。

这个目标可以先写成一个极简公式：

$$
Answer = LLM(RetrievedChunks, Question)
$$

但这只是教学版。真实系统至少还要加上查询改写、权限过滤、有效期过滤和重排序：

$$
Answer = LLM(Rerank(ValidityFilter(ACLFilter(Retrieve(Rewrite(Question))))))
$$

如果再把“带引用”写进去，表达会更接近真实产品：

$$
Output = \{answer,\ sources,\ confidence\}
$$

对新手来说，一个常见误区是把问题理解成“我要先知道文档在哪里”。  
在企业知识库问答里，用户心智应该被改写成：

“我把自然语言问题交给系统，系统负责找到最相关、最新、且我有权限查看的证据，再把证据组织成答案。”

这才是产品边界。

举一个更接近真实工程的例子。售后团队问：

“某型号设备的付款流程和验收条件是否绑定？”

这不是单文档问题。答案可能分散在合同模板、销售 SOP、财务结算制度、法务批注和项目验收记录里。知识库问答系统的目标，不是返回其中一份文档，而是把这些分散证据组合成一个可核对的回答，并明确告诉用户证据来自哪里、缺少什么、哪些结论只能推断不能确定。

---

## 核心机制与推导

一个可用的企业知识库问答，通常会经过五步：

1. 查询重写
2. 混合检索
3. ACL 过滤
4. 重排序
5. 生成回答

先看阶段表：

| 阶段 | 目标 | 为什么需要 | 新手可直接理解成 |
|---|---|---|---|
| 查询重写 | 把口语问题改写成可检索表达 | 用户问题常省略主语、术语不统一 | “把人话翻译成搜索更容易命中的问法” |
| 混合检索 | 同时做语义召回和关键词命中 | 只靠单一路径容易漏召回 | “既按意思找，也按关键字找” |
| ACL 过滤 | 去掉无权访问的片段 | 防止越权泄露 | “先看权限，再给模型看内容” |
| 重排序 | 从候选片段中选出最有用的前几段 | 候选多但质量不齐 | “把最值得引用的证据排到前面” |
| 生成回答 | 基于证据组织答案并附引用 | 用户需要结论，不想自己拼接 | “把证据整理成人能直接读的答案” |

### 1. 查询重写

用户经常不会使用文档中的官方术语。  
例如用户问“我们有没有付款流程”，真正涉及的文档可能写的是“付款审批”“应付账款”“供应商结算”“打款申请”。如果系统只用原问句检索，召回会不稳定。

因此系统常常先做查询重写，把用户问题补全成多个检索友好的表达：

- 原问题：`我们有没有付款流程`
- 重写后：
  - `付款审批流程`
  - `供应商打款申请条件`
  - `付款节点 责任人`
  - `财务结算 SOP`

这一步不是为了把问题变复杂，而是为了补足检索所需的语义约束。

### 2. 混合检索

混合检索通常是“向量检索 + 关键词检索”。

向量检索可以理解成“按语义相近度找文本”。系统先把问题和文档片段映射成向量，再按距离找相近内容。它的优势是能处理“词不同但意思接近”的情况。  
关键词检索常见实现是 BM25。BM25 可以简单理解成“看关键词命中得准不准、密不密、是不是关键位置”。它对产品型号、术语缩写、流程编号、时间、金额等精确词非常重要。

为什么两者都要有？因为它们各自会漏掉一类问题：

| 只用什么 | 容易出现的问题 |
|---|---|
| 只用向量检索 | 找到语义相近但关键条件不准确的段落 |
| 只用关键词检索 | 漏掉同义表达、缩写展开、口语化问法 |

一个常见的工程写法是先分别得到两份候选，再合并。用公式表示：

$$
Candidates = DenseRetrieve(Q') \cup SparseRetrieve(Q')
$$

如果需要简单说明“为什么要合并分数”，可以写成：

$$
Score(d, q) = \alpha \cdot Score_{dense}(d, q) + \beta \cdot Score_{sparse}(d, q)
$$

其中：

- $d$ 表示候选文档或片段
- $q$ 表示问题
- $\alpha,\beta$ 表示两路检索权重

真实系统里也常用 RRF，即 Reciprocal Rank Fusion。可以把它理解成“把多种排序结果按名次合并”，而不是简单相加原始分数。这样做的原因是不同检索器的分数尺度不一样，直接相加不稳。

### 3. ACL 过滤

ACL 过滤应该发生在模型看到内容之前，而不是回答之后。  
原因很直接：只要敏感内容已经进入提示词，就已经造成潜在泄露，即使最后答案里没直接显示，也可能通过归纳、转述、摘要等形式泄露出去。

可以把它写成：

$$
VisibleCandidates = \{c \in Candidates \mid user \in ACL(c)\}
$$

如果系统还管理制度版本和有效期，通常还会再做一次有效性过滤：

$$
ValidCandidates = \{c \in VisibleCandidates \mid expire(c)=null \ or\ expire(c)\ge today\}
$$

### 4. 重排序

企业问答的常见问题不是“完全搜不到”，而是“搜出来 20 段，里面只有 3 段真正回答问题”。这时就需要重排序。

重排序可以理解成一个更精细的二次判断过程。它不负责大范围召回，而负责在候选中判断：

- 这段是否真正回答了问题
- 这段是否包含关键条件
- 这段是否来自高可信来源
- 这段是否是现行版本
- 这段是否与其他证据冲突

对新手来说，可以记一个简单原则：

“检索解决覆盖率，重排序解决准确率。”

### 5. 生成回答

最后一步才是大模型生成。  
在这个位置上，大模型最适合做三件事：

1. 把多段证据组织成顺畅的人类语言
2. 补齐连接词、条件句、例外说明
3. 按统一格式输出结论、步骤、引用和不确定项

它不应该擅自补出证据中没有的审批规则，更不应该把不同制度年代的内容混成一句话。

因此，真实系统的提示词通常会带上硬约束，例如：

- 只能基于给定证据回答
- 若证据不足，明确说“无法确认”
- 每个关键结论必须附出处
- 若多个文档冲突，优先引用最新版本并标注冲突

把整个流程串起来，可以写成：

$$
Q \rightarrow Q' \rightarrow Retrieve_{dense+bm25} \rightarrow ACL \rightarrow Validity \rightarrow Rerank \rightarrow LLM \rightarrow Answer + Sources
$$

下面用“报销打车费需要什么材料？”做一个完整玩具例子。

| 阶段 | 输入/输出 |
|---|---|
| 用户问题 | `报销打车费需要什么材料？` |
| 查询重写 | `打车费报销 发票 行程单 审批记录 夜间加班` |
| 混合检索 | 找到差旅制度、OA 指南、财务 FAQ 中相关片段 |
| ACL 过滤 | 去掉仅财务可见的内部复核规则 |
| 有效期过滤 | 去掉已经废止的“纸质发票原件必须线下提交” |
| 重排序 | 选出最能回答“材料要求”的前三段 |
| 生成回答 | “通常需要电子发票和行程明细；若为夜间加班打车，还需补充主管审批记录。” |

这时系统输出的就不是“文档 A、文档 B、文档 C”，而是一句可执行结论加三条出处。

这一点和早期论文中的 RAG 定义是一致的：检索的价值，不只是提升准确率，还在于给回答提供外部依据与可更新知识源。对企业场景来说，这比“模型记住多少知识”更重要，因为企业知识本来就是不断变化的。

---

## 代码实现

最小可用版本至少需要三类对象：

1. 文档对象：记录标题、来源、更新时间、权限等元数据
2. 片段对象：记录切片后的文本块和它所属文档
3. 流水线函数：按固定顺序执行重写、检索、过滤、重排序和回答生成

下面给出一个可以直接运行的 Python 示例。它不依赖第三方库，只用标准库模拟一个最小 RAG 流程。为了方便新手理解，这里没有真正接入向量数据库，而是用“关键词重叠 + 简单同义词扩展”来模拟混合检索的思想。代码重点不在性能，而在把链路跑通。

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import log
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    acl_roles: frozenset[str]
    updated_at: date
    valid_until: date | None
    source_url: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def rewrite_query(question: str) -> list[str]:
    """
    返回多条检索表达，模拟“查询重写”。
    """
    rewrites = [question]
    expansions = {
        "报销": ["费用报销", "差旅报销", "发票", "审批"],
        "打车": ["出租车", "网约车", "行程单"],
        "付款": ["打款", "应付账款", "付款审批", "结算"],
        "合同": ["协议", "模板", "法务审查"],
    }
    extra_terms: list[str] = []
    for word, related in expansions.items():
        if word in question:
            extra_terms.extend(related)

    if extra_terms:
        rewrites.append(question + " " + " ".join(extra_terms))

    return rewrites


def acl_filter(chunks: Iterable[Chunk], user_roles: set[str]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.acl_roles & user_roles]


def validity_filter(chunks: Iterable[Chunk], today: date) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.valid_until is None or chunk.valid_until >= today
    ]


def compute_idf(chunks: list[Chunk]) -> dict[str, float]:
    documents = [set(tokenize(chunk.title + " " + chunk.text)) for chunk in chunks]
    n_docs = len(documents)
    df: dict[str, int] = {}

    for doc_tokens in documents:
        for token in doc_tokens:
            df[token] = df.get(token, 0) + 1

    return {
        token: log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for token, freq in df.items()
    }


def sparse_score(query: str, chunk: Chunk, idf: dict[str, float]) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(chunk.title + " " + chunk.text)

    if not q_tokens or not d_tokens:
        return 0.0

    tf: dict[str, int] = {}
    for token in d_tokens:
        tf[token] = tf.get(token, 0) + 1

    score = 0.0
    doc_len = len(d_tokens)
    avg_doc_len = 30.0
    k1 = 1.5
    b = 0.75

    for token in q_tokens:
        if token not in tf:
            continue
        term_tf = tf[token]
        term_idf = idf.get(token, 0.0)
        denom = term_tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        score += term_idf * (term_tf * (k1 + 1)) / denom

    return score


def dense_score(query: str, chunk: Chunk) -> float:
    """
    用 Jaccard 重叠率粗略模拟“语义召回”。
    真实系统这里通常会调用 embedding 模型和向量索引。
    """
    q_set = set(tokenize(query))
    d_set = set(tokenize(chunk.title + " " + chunk.text))
    if not q_set or not d_set:
        return 0.0
    return len(q_set & d_set) / len(q_set | d_set)


def hybrid_retrieval(rewrites: list[str], chunks: list[Chunk], top_k: int = 8) -> list[tuple[Chunk, float]]:
    idf = compute_idf(chunks)
    scored: dict[str, tuple[Chunk, float]] = {}

    for rewrite in rewrites:
        for chunk in chunks:
            score = 0.7 * sparse_score(rewrite, chunk, idf) + 0.3 * dense_score(rewrite, chunk)
            current = scored.get(chunk.chunk_id)
            if current is None or score > current[1]:
                scored[chunk.chunk_id] = (chunk, score)

    ranked = sorted(scored.values(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def rerank(question: str, candidates: list[tuple[Chunk, float]]) -> list[Chunk]:
    """
    简化版重排序：
    1. 先看与原问题的 token 重叠
    2. 再加一点“更新时间更近”的偏好
    """
    q_tokens = set(tokenize(question))

    def rerank_score(item: tuple[Chunk, float]) -> float:
        chunk, retrieval_score = item
        c_tokens = set(tokenize(chunk.title + " " + chunk.text))
        overlap = len(q_tokens & c_tokens)
        freshness = chunk.updated_at.toordinal() / 1_000_000
        return retrieval_score + 0.2 * overlap + freshness

    ranked = sorted(candidates, key=rerank_score, reverse=True)
    return [chunk for chunk, _ in ranked]


def build_answer(question: str, chunks: list[Chunk]) -> str:
    if not chunks:
        return "未找到可用依据，请联系文档负责人或补充更具体的问题。"

    lines = []
    if "报销" in question and "打车" in question:
        lines.append("打车报销通常需要电子发票和行程明细。")
        if any("夜间加班" in chunk.text for chunk in chunks):
            lines.append("若属于夜间加班场景，还需要补充主管审批记录。")
        lines.append("建议先在 OA 中发起报销单，再按页面要求上传材料。")
    else:
        lines.append("已根据当前可见且有效的知识片段整理答案，建议同时核对引用来源。")

    return "".join(lines)


def attach_sources(answer: str, chunks: list[Chunk]) -> str:
    refs = "\n".join(
        f"- [{chunk.title}]({chunk.source_url})，文档 ID: {chunk.doc_id}，更新时间: {chunk.updated_at.isoformat()}"
        for chunk in chunks
    )
    return f"{answer}\n\n参考来源:\n{refs}"


def answer_question(question: str, user_roles: set[str], all_chunks: list[Chunk], today: date) -> dict:
    rewrites = rewrite_query(question)
    retrieved = hybrid_retrieval(rewrites, all_chunks, top_k=8)
    visible = acl_filter((chunk for chunk, _ in retrieved), user_roles)
    valid = validity_filter(visible, today)
    top_chunks = rerank(question, [(chunk, 0.0) for chunk in valid])[:3]

    answer = build_answer(question, top_chunks)
    final_text = attach_sources(answer, top_chunks)

    return {
        "question": question,
        "answer": answer,
        "final_text": final_text,
        "sources": [chunk.source_url for chunk in top_chunks],
        "chunks": [chunk.chunk_id for chunk in top_chunks],
    }


def demo() -> None:
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="travel-policy-v3",
            title="差旅制度",
            text="员工申请打车报销时，需要提供电子发票和行程明细。",
            acl_roles=frozenset({"employee", "finance"}),
            updated_at=date(2026, 3, 1),
            valid_until=date(2026, 12, 31),
            source_url="/docs/travel-policy-v3",
        ),
        Chunk(
            chunk_id="c2",
            doc_id="oa-guide-v2",
            title="OA 报销指南",
            text="报销流程为：在 OA 发起报销单，上传发票、审批截图和必要附件。",
            acl_roles=frozenset({"employee", "finance"}),
            updated_at=date(2026, 2, 18),
            valid_until=None,
            source_url="/docs/oa-guide-v2",
        ),
        Chunk(
            chunk_id="c3",
            doc_id="night-taxi-faq",
            title="财务 FAQ",
            text="夜间加班产生的打车费用，除电子发票外，还需补充主管审批记录。",
            acl_roles=frozenset({"employee", "finance"}),
            updated_at=date(2026, 1, 10),
            valid_until=None,
            source_url="/docs/night-taxi-faq",
        ),
        Chunk(
            chunk_id="c4",
            doc_id="finance-internal-rules",
            title="财务内部复核规则",
            text="单据复核的内部打分规则仅财务组可见。",
            acl_roles=frozenset({"finance"}),
            updated_at=date(2026, 2, 1),
            valid_until=None,
            source_url="/docs/finance-internal-rules",
        ),
        Chunk(
            chunk_id="c5",
            doc_id="old-travel-policy",
            title="旧版差旅制度",
            text="旧制度要求纸质发票原件线下提交。",
            acl_roles=frozenset({"employee", "finance"}),
            updated_at=date(2024, 6, 1),
            valid_until=date(2024, 12, 31),
            source_url="/docs/old-travel-policy",
        ),
    ]

    result = answer_question(
        question="报销打车费需要什么材料？",
        user_roles={"employee"},
        all_chunks=chunks,
        today=date(2026, 4, 4),
    )

    print(result["final_text"])

    # 基本断言，验证 ACL 和有效期过滤确实生效
    assert "/docs/finance-internal-rules" not in result["final_text"]
    assert "/docs/old-travel-policy" not in result["final_text"]
    assert len(result["sources"]) >= 2


if __name__ == "__main__":
    demo()
```

这段代码解决了原型阶段最容易忽略的三件事：

| 约束 | 代码位置 | 作用 |
|---|---|---|
| 权限过滤 | `acl_filter` | 无权限片段不进入候选 |
| 有效期过滤 | `validity_filter` | 过期制度不会继续参与回答 |
| 带出处返回 | `attach_sources` | 每次回答都可复查 |

如果你运行这段代码，输出会接近下面这样：

```text
打车报销通常需要电子发票和行程明细。若属于夜间加班场景，还需要补充主管审批记录。建议先在 OA 中发起报销单，再按页面要求上传材料。

参考来源:
- [差旅制度](/docs/travel-policy-v3)，文档 ID: travel-policy-v3，更新时间: 2026-03-01
- [OA 报销指南](/docs/oa-guide-v2)，文档 ID: oa-guide-v2，更新时间: 2026-02-18
- [财务 FAQ](/docs/night-taxi-faq)，文档 ID: night-taxi-faq，更新时间: 2026-01-10
```

### 为什么这个示例虽然简单，但方向是对的

因为它已经体现了企业知识库问答里最核心的顺序约束：

```python
rewrite -> retrieval -> acl -> validity -> rerank -> answer -> sources
```

这个顺序不能随意颠倒。最常见的错误有两个。

第一，把 ACL 放在生成之后。  
这会导致模型先读到敏感内容，再“尽量不说出来”。这在安全上是错误顺序。

第二，把有效期过滤放得过晚。  
这样旧制度可能已经参与排序和生成，即使最后没直接引用，也会污染回答。

### 从玩具版走向工程版，下一步该补什么

当你把最小链路跑通后，通常会逐步替换下面几个模块：

| 原型实现 | 工程实现 |
|---|---|
| 手写 `rewrite_query` | 用 LLM 或规则模板做查询扩展 |
| Jaccard 模拟语义检索 | 用 embedding + 向量数据库 |
| 简化 BM25 | 用搜索引擎的倒排索引 |
| 手写 `rerank` | 用交叉编码器或专门 reranker |
| 模板式 `build_answer` | 用 LLM 按证据生成并强制附引用 |

对初学者，建议顺序是：

1. 先跑通 Markdown 文档 + 权限表 + 引用返回
2. 再补混合检索
3. 再补重排序
4. 最后再接入更多知识源，如聊天记录、Excel、工单系统

不要一上来就接十几个系统。最小版本能稳定回答、不会越权、能显示出处，比“大而全但经常错”更有价值。

---

## 工程权衡与常见坑

企业知识库问答最难的部分，不是“让模型能说话”，而是让它在知识混乱、版本频繁变化、权限规则复杂的现实环境里稳定说对话。

下面这张表是最常见的失败模式：

| 风险/坑 | 表现 | 为什么会发生 | 防范措施 |
|---|---|---|---|
| 知识碎片化 | 答案只覆盖半条流程 | 同一问题分散在多份资料里 | 统一采集入口，建立跨文档链接 |
| 过期文档 | 模型持续引用废弃 SOP | 索引里没有版本与失效控制 | 建 `valid_until`、版本号、状态字段 |
| 权限越权 | 用户看到不该看的规则 | 只同步了文档权限，没同步 chunk 权限 | 片段级 ACL，检索前过滤 |
| 切片过粗 | 一段里混多个主题，召回不准 | 一个 chunk 包含太多无关信息 | 按语义边界切片，保留标题 |
| 切片过细 | 证据碎裂，模型无法理解上下文 | 每段太短，没有条件和例外 | 保留上下文窗口或父段落信息 |
| 只做向量检索 | 型号、编号、术语漏召回 | 向量检索不擅长精确词匹配 | 加 BM25 或倒排索引 |
| 不附引用 | 用户不信，也无法追责 | 回答虽流畅但无法核对 | 强制返回来源链接、版本和日期 |
| 无质量闭环 | 错答反复出现 | 没有日志和坏例分析机制 | 记录问答日志，持续优化检索和重排 |

### 坑 1：过期制度没有真正下线

这是最常见、也最容易被低估的问题。  
例如旧版制度写着“纸质发票必须线下提交”，但公司已经改成电子发票。如果索引里没有 `valid_until` 或版本字段，旧文档可能仍然参与召回，模型也就会持续输出过时答案。

这里的关键点是：  
解决方案不是在提示词里提醒模型“尽量优先用新制度”，而是让过期内容在检索阶段就退出候选集。因为一旦过期内容已经进了上下文，后面就很难完全消除它的影响。

### 坑 2：权限只做到文档级，没有做到片段级

很多系统最初只同步“这个文档谁能看”，但企业场景中一份文档里常常同时包含公开内容和敏感内容。  
例如《财务制度》正文可公开，附件里的某些复核规则只对财务组开放。如果权限只做到文档级，系统可能把敏感附件一起切进可见 chunk，最后在回答里泄露内部规则。

新手可以直接记住一句话：

“企业问答的权限粒度，至少要和检索粒度一致。”

如果你按 chunk 检索，就应该按 chunk 控权限。

### 坑 3：切片策略错误

切片不是“按 500 字硬切一下”这么简单。  
如果切得太粗，一个片段里会混进多个主题，导致召回命中但答案不聚焦；如果切得太细，审批条件、例外条款、流程顺序会被拆散，模型拿到的只是碎片，难以稳定回答。

一个更实际的切片原则是：

| 文档类型 | 更合适的切片方式 |
|---|---|
| 制度/SOP | 按标题层级、条款编号切片 |
| FAQ | 一问一答为一个 chunk |
| 会议纪要 | 按议题或决议项切片 |
| 工单/客服记录 | 按会话轮次或事件阶段切片 |
| 表格/Excel | 先转为结构化记录，再决定是否拼接为文本 |

### 坑 4：答案看起来很对，其实证据不足

这类错误比明显答错更危险。  
例如用户问“海外供应商付款是否需要法务审查”，系统检索到付款制度和合同模板，但漏掉了最新的跨境审查补充条款。此时模型可能凭语言流畅性给出一段“听起来正确”的答案，却没有足够证据支撑。

所以工程上通常要加三条硬约束：

1. 证据不足时允许回答“无法确认”
2. 每个关键结论必须能映射到至少一个引用
3. 质量评估不能只看答案像不像，还要看依据是否正确

可以把这件事抽象成一个简单判断：

$$
ReliableAnswer = CorrectEvidence + CorrectPermission + CorrectGeneration
$$

只要其中任意一项失败，答案就不可靠。

### 坑 5：没有质量闭环

如果系统上线后只看“调用量”“响应时间”，却不看“是否引用对了”“是否越权”“是否用了过期制度”，那么错误会反复出现。

一个基本的评估面板至少要看下面四项：

| 指标 | 含义 |
|---|---|
| Evidence Hit Rate | 回答是否命中正确证据 |
| Permission Error Rate | 是否发生越权引用 |
| Stale Citation Rate | 是否引用了过期内容 |
| Abstain Rate | 证据不足时是否正确选择“不确定” |

对企业场景来说，这些指标通常比通用模型分数更重要。因为用户真正关心的不是“语言是否优美”，而是“这个回答能不能拿去执行”。

---

## 替代方案与适用边界

不是所有知识问答都要上 RAG + LLM。  
方案选择取决于四个维度：知识规模、变化频率、合规强度、回答复杂度。

先看对比表：

| 方案 | 适用场景 | 优势 | 限制 |
|---|---|---|---|
| 全文搜索/关键词检索 | 文档少、术语稳定 | 成本低、实现简单 | 用户仍需自己读和整合 |
| FAQ Bot + 规则引擎 | 问题固定、答案固定 | 一致性高、可控性强 | 难覆盖长尾问题和跨文档组合 |
| RAG + LLM | 多源知识、问题开放、需整合证据 | 可处理自然语言和多片段总结 | 成本更高，依赖知识治理质量 |
| 人工客服/专家答复 | 高风险、低频、强责任场景 | 判断力强，可处理模糊上下文 | 不可扩展，难沉淀为系统能力 |

对新手，一个很实用的判断方法是看问题是否满足下面四个信号。

1. 资料量已经超过人工能记住的范围，例如上百份文档
2. 同一个问题往往需要拼接多份资料才能回答
3. 用户提问方式和文档术语不一致
4. 回答必须带出处，且不能越权

如果这四个信号大多成立，那么 RAG + LLM 通常是合理方案。  
如果只成立一条，比如“文档不多，但想做一个聊天界面”，那往往还没到需要复杂架构的时候。

下面给三个典型边界。

### 边界 1：文档少且问题固定，不必上复杂 RAG

例如公司只有十几条行政 FAQ，问题也很固定：

- 年假几天
- 会议室怎么预约
- 工卡丢了怎么办

这种场景用全文搜索、关键词匹配，甚至一份规则表就能解决。上 RAG 的收益不一定能覆盖复杂度。

### 边界 2：高合规场景，生成权应该收缩

例如法务合规、医疗建议、强监管审批这类场景，系统可以采用保守模式：

- 规则引擎负责得出结论
- 检索系统负责找依据
- LLM 只负责把结果解释成人话

也就是“规则决定能不能做，LLM 负责说明为什么”。  
这样会牺牲一部分灵活性，但能显著提升确定性。

### 边界 3：知识主要是隐性经验，RAG 不是第一解

如果企业里真正关键的信息根本不在文档里，而在资深员工脑子里，那么 RAG 的第一步不是建问答，而是先做知识沉淀。  
否则系统只能对“已经写下来的知识”表现良好，对真正高价值但未显化的经验基本无能为力。

所以，RAG + LLM 不是默认答案，而是在“知识多源、表达不统一、问题开放、且需要证据整合”同时成立时，最合理的工程解。

---

## 参考资料

| 来源 | 年份 | 重点贡献 |
|---|---:|---|
| Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | 2020 | 经典 RAG 论文，明确提出“参数知识 + 外部检索”的生成框架，并强调可更新知识与出处的重要性 |
| Microsoft Learn, [*Hybrid search using vectors and full text in Azure AI Search*](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview) | 2026 | 说明企业检索中为什么要把向量检索和关键词检索结合起来，并给出混合检索与 RRF 的工程做法 |
| Microsoft Learn, [*Document-level access control in Azure AI Search*](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview) | 2025 | 说明企业检索系统为什么必须在查询阶段执行文档级权限控制，适用于 RAG 和企业搜索场景 |
| Microsoft Learn, [*Security filters for trimming results in Azure AI Search*](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search) | 2026 | 给出基于用户或组身份做安全过滤的实现模式，适合作为 ACL 过滤的工程参考 |
| Yu et al., [*EKRAG: Benchmark RAG for Enterprise Knowledge Question Answering*](https://aclanthology.org/2025.knowledgenlp-1.13/) | 2025 | 提供企业知识问答的评测基准，说明企业内容与通用开放域问答在检索和评估上存在明显差异 |

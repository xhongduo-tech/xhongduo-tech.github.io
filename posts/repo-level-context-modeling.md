## 核心结论

仓库级上下文建模，指模型在补全代码时，不只看当前文件，还显式利用同一仓库里的相关文件、符号定义、调用链和依赖关系。白话解释：模型不是“只盯着眼前这一屏代码写下去”，而是先去仓库里找“这个函数到底依赖谁、被谁调用、类型从哪里来”，再决定如何补全。

这个方向的结论已经比较清楚：

| 方案 | 看当前文件 | 看跨文件上下文 | 典型收益 | 代价 |
| --- | --- | --- | --- | --- |
| 传统 in-file 补全 | 是 | 否 | 延迟低，实现简单 | 缺定义时容易瞎猜 |
| 仓库级检索后补全 | 是 | 是 | 跨文件任务准确率显著提升 | 检索与拼接增加延迟 |
| 训练期就融合仓库上下文 | 是 | 是 | 小模型更容易学会用上下文 | 训练成本更高 |

一个足够直观的数值是：如果单文件基线 exact match 只有 50%，加入有效 cross-file context 后提升到 67% 并不夸张，这对应约 34% 的相对提升。研究里常见的仓库级 FIM 训练或检索增强，对跨文件补全带来的相对提升大致落在 15% 到 30%，某些 benchmark 上更高。

“新手版”理解可以压缩成一句话：当你在一个类里补缺失函数时，系统不应该只看当前文件，而应该先把相关类型定义、工具函数、被调用 API 和测试用例找出来，再一起交给模型，这才接近真实开发。

从方法论上看，当前最有代表性的三条线分别是：

| 方法 | 核心思想 | 代表工作 |
| --- | --- | --- |
| 检索增强生成 | 先查仓库，再补代码 | RepoCoder |
| 静态依赖图增强 | 用跨文件依赖图找上下文 | CoCoMIC |
| 训练期多源融合 | 训练时就让模型消化多路上下文 | RepoFusion |

---

## 问题定义与边界

本文讨论的不是“让模型读完整个仓库”，而是“在代码缺失片段需要补全时，怎样选择最相关的仓库级上下文”。这里的补全主要是 FIM，Fill-in-the-Middle，中间填充；白话解释：已知前半段和后半段，让模型补中间缺掉的部分。

基本目标是：

$$
\hat{M} = \arg\max_M P(M \mid P, S)
$$

其中：

- $P$ 是 prefix，也就是缺失片段前面的代码
- $S$ 是 suffix，也就是缺失片段后面的代码
- $M$ 是模型要补出的中间代码

仓库级上下文建模把目标扩展为：

$$
\hat{M} = \arg\max_M P(M \mid P, S, C)
$$

这里的 $C$ 是 cross-file context，跨文件上下文；白话解释：除了当前文件前后文，再额外给模型一组来自仓库其他位置的证据。

边界要讲清楚，否则工程上很容易失控：

| 边界问题 | 本文讨论 | 不重点讨论 |
| --- | --- | --- |
| 任务类型 | 代码补全、FIM、跨文件 infilling | 通用聊天、长文档问答 |
| 上下文来源 | 仓库内文件、符号图、调用链、测试 | 外部互联网知识 |
| 目标优化 | EM、编辑正确率、延迟、token 成本 | 开放式创造性生成 |
| 使用时机 | 仅在当前上下文不够时触发 | 每次无差别检索全仓库 |

玩具例子最容易说明这个边界。

假设当前文件里有一段缺失代码：

```python
# service.py
from repo import UserRepo

class UserService:
    def __init__(self, repo: UserRepo):
        self.repo = repo

    def get_active_usernames(self):
        # 中间缺失
        pass
```

如果只看当前文件，模型可能猜出一个遍历逻辑，但不知道 `repo.list_users()` 返回什么结构，也不知道 `User` 对象的字段名是 `active` 还是 `is_active`。这时只靠 $P$ 和 $S$ 不够，必须引入 `models.py`、`repo.py` 里的定义作为 $C$。

真实工程例子更典型。比如在支付服务里补全 `refund_order(order_id)` 的中间逻辑，当前文件只看得到函数名，但真正需要的约束往往散落在：

- `order_model.py` 里的状态枚举
- `payment_gateway.py` 里的退款 API 包装
- `audit.py` 里的审计日志接口
- `tests/test_refund.py` 里的行为预期

这就是仓库级上下文建模的实际边界：不是为了“多喂一点 token”，而是为了把任务真正需要的依赖补齐。

---

## 核心机制与推导

核心机制可以概括成一句话：先决定哪些跨文件证据值得看，再让模型在 $P,S,C$ 条件下生成 $M$。

如果把补全过程写成依赖关系，它更接近下面这个形式：

$$
P(M \mid P,S,C) \propto P(C \mid P,S) \cdot P(M \mid P,S,C)
$$

这里第一项并不是严格训练目标，而是工程上的两阶段近似：先根据当前缺口估计“哪些上下文最相关”，再在这些上下文条件下生成答案。白话解释：先查资料，再写代码。

### 1. RepoCoder：检索-生成闭环

RepoCoder 的关键不是“检索一次”，而是“生成过程中可反复检索”。这意味着模型先根据当前缺口拿到一批候选片段，生成一部分内容后，再依据新暴露出的标识符或 API 继续检索。这样做的原因很直接：很多跨文件线索不是在补全开始前就完全可见的。

“新手版”理解：你写到一半发现需要 `serialize_event()`，这时系统再去搜谁定义了它，而不是一开始盲目把整个仓库塞进 prompt。

### 2. CoCoMIC：静态依赖图提供结构化线索

静态依赖图，指通过 import、调用、类型引用等静态关系构造的图；白话解释：把“谁依赖谁”画成一张图。CoCoMIC 用这类图帮助模型找到跨文件但结构上强相关的上下文，而不是只靠文本相似度。

这种方法的优势在于，当符号名不相似时，静态图仍可能找到正确文件。例如当前要补全的是 `OrderService.cancel()`，真正相关的是 `StateTransitionGuard`，两者词面不相近，但调用图里可能直接相连。

### 3. RepoFusion：训练期融合多源上下文

RepoFusion 的关键点是：不是把检索器当成推理时外挂，而是在训练阶段就让模型见过多种来源的仓库上下文，比如 Prompt Proposal、BM25 候选、近邻样本。这样模型会学会“如何消费噪声上下文”和“如何在多路证据之间取舍”。

这对小模型尤其重要。因为大模型即使面对脏 prompt，也可能凭参数记忆和强推理能力兜底；小模型如果没被专门训练过，常常会被多源上下文直接扰乱。

### 4. BM25 与 Dense Retrieval 的差异

BM25 是基于词项匹配的传统检索；白话解释：看关键词和标识符是否真的对得上。Dense Retrieval 是向量检索；白话解释：把代码编码成语义向量，再找意思相近的片段。

两者不是替代关系，而是互补关系：

| 检索源 | 优势 | 弱点 | 适合场景 |
| --- | --- | --- | --- |
| BM25 | 稀有标识符、精确 API 名称命中率高 | 词不一样就容易漏 | 查类名、函数名、配置键 |
| Dense | 语义泛化能力强 | 容易召回“看起来像但不能用”的抽象代码 | 查模式相近实现 |
| 静态依赖图 | 结构相关性强 | 需要额外分析过程 | 查 import、调用、类型依赖 |

一个简单推导是：如果目标片段依赖一个稀有符号 `LedgerReconciliationPolicy`，BM25 的召回概率通常高；如果目标要实现“分页查询后聚合结果”的逻辑，但函数名不同，dense 更可能召回相似实现。于是融合检索更合理，常见做法是先 union 候选，再重排序。

---

## 代码实现

下面给出一个可运行的玩具实现，演示“BM25 风格打分 + dense 风格语义打分 + 融合排序”的最小闭环。这里不用外部库，只做概念验证。

```python
from math import log, sqrt
from collections import Counter, defaultdict

docs = {
    "repo.py": "class UserRepo def list_users return users",
    "models.py": "class User name is_active email",
    "service.py": "def get_active_usernames repo list_users",
    "tests.py": "test active usernames should include only active user names",
}

query = "get active usernames from list_users using is_active"

def tokenize(text: str):
    return [t.lower() for t in text.split()]

def bm25_scores(query, docs, k1=1.5, b=0.75):
    tokenized_docs = {k: tokenize(v) for k, v in docs.items()}
    q = tokenize(query)
    N = len(docs)
    avgdl = sum(len(v) for v in tokenized_docs.values()) / N

    df = defaultdict(int)
    for terms in tokenized_docs.values():
        for term in set(terms):
            df[term] += 1

    scores = {}
    for doc_id, terms in tokenized_docs.items():
        tf = Counter(terms)
        dl = len(terms)
        score = 0.0
        for term in q:
            if term not in df:
                continue
            idf = log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            if denom > 0:
                score += idf * (freq * (k1 + 1)) / denom
        scores[doc_id] = score
    return scores

def dense_like_scores(query, docs):
    # 玩具版“语义”分数：词袋余弦相似度
    qv = Counter(tokenize(query))
    scores = {}
    for doc_id, text in docs.items():
        dv = Counter(tokenize(text))
        dot = sum(qv[t] * dv[t] for t in qv)
        qn = sqrt(sum(v * v for v in qv.values()))
        dn = sqrt(sum(v * v for v in dv.values()))
        scores[doc_id] = dot / (qn * dn) if qn and dn else 0.0
    return scores

def reciprocal_rank_fusion(*rankings, k=60):
    score = defaultdict(float)
    for ranking in rankings:
        ordered = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ordered, start=1):
            score[doc_id] += 1.0 / (k + rank)
    return dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

bm25 = bm25_scores(query, docs)
dense = dense_like_scores(query, docs)
fused = reciprocal_rank_fusion(bm25, dense)

top_doc = next(iter(fused))
assert top_doc in {"repo.py", "models.py", "service.py"}
assert fused["repo.py"] > 0
assert fused["models.py"] > 0

print("BM25:", bm25)
print("Dense:", dense)
print("Fused:", fused)
```

这个例子是玩具版，但核心结构已经完整：

1. 先根据当前缺口构造 query。
2. 用不同检索器召回候选。
3. 用融合排序决定哪些片段进入 prompt。
4. 交给代码模型做 FIM 补全。

把它写成更接近 RepoCoder 的伪代码，大致如下：

```python
def repo_level_complete(prefix, suffix, repo_index, model, max_rounds=2):
    partial = ""
    for _ in range(max_rounds):
        query = build_query(prefix, partial, suffix)
        bm25_hits = bm25_retrieve(query, repo_index, topk=8)
        dense_hits = dense_retrieve(query, repo_index, topk=8)
        graph_hits = static_graph_retrieve(query, repo_index, topk=4)

        context = rerank_and_pack(bm25_hits, dense_hits, graph_hits, budget=3000)
        draft = model.fill_in_middle(prefix, suffix, context)

        if is_confident_enough(draft) or no_new_symbol(draft, partial):
            return draft
        partial = draft

    return partial
```

这里有两个工程要点。

第一，`rerank_and_pack` 很重要。它决定哪些 snippet 进入有限 prompt budget。真实系统通常不会简单拼接，而会按文件粒度或符号粒度切片，再按相关性、去重率、token 开销重新排序。

第二，`is_confident_enough` 决定要不要继续检索。Repoformer 一类工作指出，很多时候额外检索没有收益，说明“检索是否触发”本身也是一个学习问题。

真实工程例子可以看一个 API 服务：

| 文件 | 作用 |
| --- | --- |
| `handlers/refund.py` | HTTP 接口入口 |
| `services/refund_service.py` | 退款业务逻辑 |
| `gateways/payment_gateway.py` | 第三方支付调用 |
| `models/order.py` | 订单状态定义 |
| `tests/test_refund_flow.py` | 行为约束 |

当开发者在 `refund_service.py` 中补全“退款前状态检查”逻辑时，最有价值的上下文常常不是同文件上一段代码，而是：

- `models/order.py` 里的合法状态枚举
- `tests/test_refund_flow.py` 里的预期失败条件
- `payment_gateway.py` 里的异常类型和返回结构

这就是仓库级补全优于单文件补全的直接原因。

---

## 工程权衡与常见坑

仓库级上下文建模不是“检索越多越好”。它的核心矛盾是准确率、延迟和噪声三者之间的平衡。

### 1. 检索频率过高

最常见的坑是每次补全都触发检索。问题不在能不能做，而在大部分补全根本不需要做。比如补一个局部变量名、一个简单循环、一个当前类里的私有方法，跨文件检索几乎没有收益。

| 策略 | 风险 | 对策 |
| --- | --- | --- |
| 每次都检索 | 延迟高，token 浪费 | 先做 selective RAG 判定 |
| 只检索一次 | 后续暴露的新符号可能漏掉 | 允许有限轮次迭代检索 |
| 只靠模型自判断 | 容易错过边界场景 | 增加启发式规则和兜底阈值 |

一个常见阈值逻辑是：

```python
def should_retrieve(prefix, suffix):
    signals = {
        "has_undefined_symbol": detect_undefined_symbol(prefix, suffix),
        "has_import_reference": detect_import_gap(prefix),
        "low_confidence": estimate_model_confidence(prefix, suffix) < 0.45,
    }
    return any(signals.values())
```

白话解释：如果当前缺口里出现“未知类型”“跨模块 API”“模型把握不高”这几类信号，再触发检索。

### 2. 纯 BM25 或纯 dense 都不稳

纯 BM25 的问题是过度依赖词面。代码里存在大量别名、包装层和语义相似实现，词不一样就会漏召回。纯 dense 的问题是过度抽象，容易把“模式像”但接口不兼容的代码拿进来，反而污染 prompt。

更稳妥的方案通常是：

- 候选召回阶段：BM25 + dense + 静态图并行
- 重排序阶段：RRF 或学习到的 reranker
- 拼接阶段：优先保留定义、类型、测试约束，少放无关实现细节

### 3. 上下文切片过粗

很多系统直接按整个文件召回。这在仓库小的时候还能忍，大仓库里问题很明显：token 大量浪费在无关内容上。更合理的是按符号切片，至少切到类、函数、常量定义级别。

### 4. 训练分布与推理分布不一致

如果模型训练时只见过干净的单文件 FIM，推理时突然给它多路跨文件噪声，它未必会正确利用上下文。RepoFusion 类方法的价值就在这里：提前让模型学会面对不完美检索结果。

---

## 替代方案与适用边界

不是所有任务都需要仓库级上下文。一个成熟系统通常会按任务类型选择策略，而不是统一上最大配置。

| 任务类型 | 是否常需跨文件 | 推荐策略 | 资源/延迟建议 |
| --- | --- | --- | --- |
| 单文件局部补全 | 低 | 仅 in-file FIM | 最低延迟 |
| 类内方法补全 | 中 | in-file 优先，低置信度再检索 | 低到中延迟 |
| 跨文件 infilling | 高 | selective retrieval + rerank | 中延迟 |
| 缺定义的 API 调用补全 | 高 | 静态图 + BM25 融合 | 中延迟 |
| 仓库级重构建议 | 很高 | 更大上下文或多轮 agent | 高延迟 |

几种替代方案可以这样理解：

| 方案 | 适用边界 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 全仓库长上下文直接塞入 | 小仓库、离线分析 | 实现直观 | token 成本高，噪声大 |
| 严格静态分析检索 | 类型语言、依赖明确 | 精确性高 | 对动态语言不稳定 |
| Selective Retrieval | 在线补全场景 | 准确率和延迟平衡较好 | 需要额外触发策略 |
| 训练期仓库级 FIM | 有训练资源的平台 | 小模型收益明显 | 数据构造复杂 |

“新手版”的适用边界判断可以压缩成两条：

- 如果当前文件已经包含全部定义，任务只是补一段局部实现，可以跳过跨文件检索。
- 如果你发现缺失片段依赖外部类型、工具函数、测试约束或状态机定义，就必须把相关文件拉进来。

换句话说，仓库级上下文不是默认配置，而是对“信息缺口”的定向修复手段。

---

## 参考资料

下表列出本文涉及的核心工作。可以把它们视为仓库级代码补全研究的主线资料。

| 文献 | 贡献 | 重点数据或结论 |
| --- | --- | --- |
| [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://aclanthology.org/2023.emnlp-main.151/) | 提出检索-生成迭代闭环与 RepoEval benchmark | 证明跨文件检索能显著提升仓库级补全 |
| [RepoFusion: Training Code Models to Understand Your Repository](https://arxiv.org/abs/2306.10901) | 训练期融合 Prompt Proposal、BM25、近邻上下文 | 小模型可借训练策略逼近更大模型效果 |
| [CoCoMIC: Code Completion By Jointly Modeling In-File and Cross-File Context](https://assets.amazon.science/4c/d0/fbb252f64d9f879159040c14cc33/cocomic-code-completion-by-jointly-modeling-in-file-and-cross-file-context.pdf) | 用 CCFinder 与跨文件依赖图联合建模 | 引入 cross-file context 后 EM 相对提升显著 |
| [CrossCodeEval](https://crosscodeeval.github.io/) | 提供强调跨文件检索能力的评测框架 | 说明只测单文件会低估真实难度 |
| [Fill-in-the-Middle (FIM) 综述条目](https://www.emergentmind.com/topics/fill-in-the-middle-fim-993a87a4-2524-4748-943d-0b0ae4448bd1) | 汇总 FIM 目标、训练形式与扩展方向 | 给出 $P(M \mid P,S)$ 到 $P(M \mid P,S,C)$ 的统一视角 |
| [REPOFUSE Technical Report](https://www.emergentmind.com/papers/2402.14323) | 从系统角度讨论仓库上下文融合与效率优化 | 在准确率提升同时降低推理延迟 |
| [Repoformer / selective retrieval 相关综述资料](https://cloud.tencent.com/developer/article/2630515) | 强调不是每次检索都有收益 | 额外检索中大比例可能无效，需阈值控制 |

这些资料的阅读顺序建议是：先看 CrossCodeEval 理解问题边界，再看 RepoCoder 理解检索闭环，再看 CoCoMIC 和 RepoFusion 理解“静态图增强”和“训练期融合”这两条路线。

## 核心结论

上下文窗口填充策略，白话说，就是“在模型能读的有限字数里，决定把哪几段记忆塞进去”。它不是简单的检索后拼接，而是一个受限集合选择问题：在固定 token 预算下，同时最大化相关性与信息覆盖。

三类常见策略可以按能力分层理解：

| 策略 | 选择规则 | 优点 | 主要问题 | 适合场景 |
| --- | --- | --- | --- | --- |
| 贪心填充 | 按相关性排序，直到窗口满 | 实现最简单，延迟低 | 容易塞入重复片段 | 问题单跳、语料重复少 |
| MMR 多样性填充 | 每次选“相关且不重复”的片段 | 覆盖更好，适合多跳 | 参数需要调，过强会漏细节 | 企业知识库、FAQ、事故分析 |
| 分层摘要填充 | 先文档级筛选，再章节级，再片段级压缩 | 适合长文档和复杂问题 | 实现复杂，摘要本身可能丢信息 | 法律、论文、多文档综述 |

如果只看工程结论，优先级通常是这样的：

1. 先把“检索 top-k 后直接全塞进去”的做法替换掉。
2. 语料存在明显重复时，先上 MMR 或 AdaGReS 一类的冗余惩罚。
3. 文档很长、问题需要跨章节时，再上分层摘要。

一个足够稳定的经验是：有限预算下，正确目标不是“拿到最像查询的几个片段”，而是“拿到一组能共同回答问题的片段”。公开工程案例里，AdaGReS 相比传统 top-k 在冗余企业语料上报告了约 30% 的 token 节省和约 15% 的整体正确率提升；多跳问题提升更明显。对于初级工程师，这个结论比任何花哨检索名词都更重要。

---

## 问题定义与边界

上下文窗口，白话说，就是模型这一次回答前能看到的输入容量。它是硬约束，不会因为你“还有很多相关资料”就自动变大。于是问题变成：

给定候选记忆片段集合 $D=\{d_1,d_2,\dots,d_n\}$，在预算 $B$ 内选择子集 $S$，使得

$$
\sum_{d_i \in S} tokens(d_i) \le B
$$

并且 $S$ 对当前问题既相关，又尽量覆盖不同事实。

这里有三个边界最容易被忽略。

第一，预算边界。`top-k=5` 不等于“最终放 5 段”。如果 5 段里有 3 段在重复同一事实，那么预算虽然花掉了，信息覆盖却没有增加。

第二，冗余边界。冗余，白话说，就是“看起来是新段落，其实在重复老信息”。向量检索特别容易把相邻、改写、转述后的相似块一起召回。

第三，多跳边界。多跳，白话说，就是答案要跨两个或多个事实链路才能拼出来。例如“上周二故障的原因和业务影响”，至少要覆盖“原因”与“影响”两个 hop。只覆盖一个 hop，答案就会不完整。

下面这个小表可以直接用来判断候选片段是否值得保留：

| 候选片段 | 相关性 | 来源 | hop | 是否应优先覆盖 |
| --- | --- | --- | --- | --- |
| 数据库超时根因说明 | 高 | 事故复盘 A | 原因 | 是 |
| 数据库超时重复描述 | 高 | 值班群摘录 | 原因 | 否，可能冗余 |
| 订单失败率上升 18% | 中高 | 影响评估报告 | 影响 | 是 |
| 上月另一次故障记录 | 中 | 历史周报 | 无关 hop | 否 |

玩具例子可以更直观。假设 token 预算只能放 3 段，每段约 500 token：

- 贪心 top-3：原因 A、原因 B、原因 C
- 更好的集合：原因 A、影响 D、修复动作 E

前者相关性分数可能更高，但后者更接近“可回答”。这就是上下文填充和单纯检索排序的本质区别。

---

## 核心机制与推导

MMR，白话说，就是“每次挑一个既像问题、又别太像已选内容的片段”。它的经典形式是：

$$
MMR(d_i)=\lambda \cdot sim(q,d_i) - (1-\lambda)\cdot \max_{d_j \in S} sim(d_i,d_j)
$$

其中：

- $sim(q,d_i)$ 表示候选片段对查询的相关性
- $\max_{d_j \in S} sim(d_i,d_j)$ 表示它和已选集合里最相似片段的重复程度
- $\lambda$ 控制相关性与多样性的平衡

如果 $\lambda=1$，它退化成纯相关性排序；如果 $\lambda=0$，它几乎只追求差异性，容易选到不相关内容。公开实践资料里，$\lambda=0.7$ 常被当作偏稳妥的默认值，因为它仍然优先相关性，但已经开始惩罚重复。

可以把它理解成拼图问题。贪心填充是在找“和盒盖最像的三块拼图”；MMR 是在找“既像盒盖，又能补上新区域的下一块拼图”。

AdaGReS 这类策略再往前走一步，不只看单个片段分数，而是把上下文视为一个集合优化问题。它常写成：

$$
Score(S)=Rel(S)-\beta \cdot Redundancy(S)
$$

这里的 $\beta$ 是冗余惩罚强度。它的工程价值在于：同样是冗余语料，有的查询需要强惩罚，有的查询不能惩罚太重。于是 $\beta$ 可以按候选池统计量自适应调整，例如基于相关性均值和方差近似估计：

$$
\beta \approx \frac{\mu_{rel}}{\sigma_{rel}+\epsilon} \cdot budget\_factor
$$

这个式子不是唯一标准答案，但表达了一个实用思想：候选越同质，越需要抑制重复；候选差异越大，惩罚就要更保守。

一个玩具数值例子：

| 候选 | 对查询相关性 | 与已选最大相似度 | $\lambda=0.7$ 时 MMR 分数 |
| --- | --- | --- | --- |
| A: 数据库超时 | 0.92 | 0.00 | 0.644 |
| B: 数据库超时改写版 | 0.90 | 0.95 | 0.345 |
| C: 业务影响统计 | 0.78 | 0.20 | 0.486 |
| D: 修复动作 | 0.70 | 0.25 | 0.415 |

第一轮会先选 A。第二轮如果只看相关性会继续选 B；但 MMR 会发现 B 与 A 高度重复，于是更可能选 C。这样，窗口里第一次同时出现“原因”和“影响”。

这也是为什么在多跳推理里，MMR 类方法常比贪心填充更稳。它不是让每个片段都最相关，而是让最终集合更完整。

---

## 代码实现

下面给出一个可运行的 Python 版本。它不依赖真实向量库，只用手写相似度分数来演示“贪心填充”和“MMR 填充”的差异。

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Chunk:
    chunk_id: str
    text: str
    tokens: int
    rel: float
    hop: str
    source: str

def greedy_fill(chunks: List[Chunk], budget: int) -> List[Chunk]:
    ordered = sorted(chunks, key=lambda x: x.rel, reverse=True)
    selected = []
    used = 0
    for c in ordered:
        if used + c.tokens <= budget:
            selected.append(c)
            used += c.tokens
    return selected

def mmr_fill(
    chunks: List[Chunk],
    budget: int,
    lambda_: float,
    pair_sim: dict[Tuple[str, str], float],
) -> List[Chunk]:
    selected = []
    remaining = chunks[:]
    used = 0

    def sim(a: str, b: str) -> float:
        if (a, b) in pair_sim:
            return pair_sim[(a, b)]
        if (b, a) in pair_sim:
            return pair_sim[(b, a)]
        return 0.0

    while remaining:
        best = None
        best_score = float("-inf")
        best_idx = -1
        for i, c in enumerate(remaining):
            if used + c.tokens > budget:
                continue
            redundancy = max((sim(c.chunk_id, s.chunk_id) for s in selected), default=0.0)
            score = lambda_ * c.rel - (1 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best = c
                best_idx = i
        if best is None:
            break
        selected.append(best)
        used += best.tokens
        remaining.pop(best_idx)
    return selected

chunks = [
    Chunk("A", "数据库超时根因", 500, 0.92, "cause", "postmortem"),
    Chunk("B", "数据库超时改写版", 500, 0.90, "cause", "ticket"),
    Chunk("C", "业务影响: 订单失败率上升18%", 500, 0.78, "impact", "impact-report"),
    Chunk("D", "修复动作: 连接池参数回滚", 500, 0.70, "fix", "runbook"),
]

pair_sim = {
    ("A", "B"): 0.95,
    ("A", "C"): 0.20,
    ("A", "D"): 0.25,
    ("B", "C"): 0.22,
    ("B", "D"): 0.24,
    ("C", "D"): 0.30,
}

budget = 1500

greedy = greedy_fill(chunks, budget)
mmr = mmr_fill(chunks, budget, lambda_=0.7, pair_sim=pair_sim)

greedy_hops = {c.hop for c in greedy}
mmr_hops = {c.hop for c in mmr}

assert len(greedy) == 3
assert len(mmr) == 3
assert "impact" not in greedy_hops
assert "impact" in mmr_hops

print("greedy:", [c.chunk_id for c in greedy])
print("mmr:", [c.chunk_id for c in mmr])
```

这个例子里，贪心填充会优先拿 `A+B+C` 或 `A+B+D`，取决于排序细节，但它很容易先塞进两个“原因”片段。MMR 则更容易拿 `A+C+D`，因为 `B` 与 `A` 的重复度过高。

真实工程里，代码会多两层：

1. `retrieve_candidates(query, k=20)` 先拉大候选池，为后续集合优化留空间。
2. `select_context(...)` 再按相关性减去冗余惩罚做迭代选择，直到 token 用完。

如果你的系统已经有 embedding 检索，改造成本通常不高。难点不在代码量，而在你是否开始把“选上下文”当成独立模块来评估。

---

## 工程权衡与常见坑

真实工程例子最典型的是企业事故分析。问题是：“上周二故障的原因和业务影响是什么？”如果知识库里同时有事故复盘、值班群记录、影响评估表、周报摘要，top-k 很可能召回 3 段都在讲“数据库超时”，却把“订单失败率上升”“退款延迟”这类影响信息挤掉。公开案例中，AdaGReS 在这类冗余企业语料上报告了约 30% token 节省与约 15% 整体正确率提升，核心原因不是检索更准，而是窗口里终于同时装进了互补事实。

但多样性不是越强越好，常见坑主要有四个。

第一，过度去重。两个高相似片段里，可能一个带表格，一个带结论。只因为“长得像”就删掉，会丢关键细节。

第二，候选池过小。你如果只先检索 5 条，再做 MMR，算法再聪明也无事可做。集合优化需要候选冗余里存在“可替换项”。

第三，只监控最终答案，不监控选择过程。工程上至少要记录 `selected_id`、`rel_score`、`redundancy`、`used_tokens`。否则答错时你根本不知道是没召回、选错了，还是摘要时丢了信息。

第四，把摘要当万能补丁。分层摘要能压缩上下文，但摘要模型本身也会漏掉边界条件、数字和否定关系。对高风险问答，原文片段和摘要片段最好混合保留。

可以把调参思路整理成一张决策表：

| 现象 | 更可能的问题 | 调整方向 |
| --- | --- | --- |
| 回答重复一个事实，缺少第二个事实 | 冗余过高，覆盖不足 | 降低 top-k 直塞比例，增加 MMR/集合选择 |
| 回答漏掉细小关键条件 | 冗余惩罚过强 | 降低 $\beta$ 或加“每个子问题至少一段”约束 |
| 上下文仍然很长 | 候选过多但过滤太弱 | 提高冗余惩罚，增加预算感知 |
| 长文档答案碎片化 | 平面检索不够 | 切换到分层摘要或层级 RAG |

工程上最稳的做法通常不是“只信一个分数”，而是同时加两个约束：相关性分数负责保底，覆盖约束负责防漏。

---

## 替代方案与适用边界

如果问题主要是“重复片段太多”，MMR 或 AdaGReS 足够有效；如果问题主要是“材料太长、结构太深”，分层摘要更合适。

层级 RAG，白话说，就是“先看目录和摘要，再深入到章节和具体段落”。它通常分三层：文档级检索、章节级检索、片段级检索，然后在进入模型前做压缩。它适合论文综述、法规检索、医疗指南这类长文档场景，因为问题往往跨章节，且原文顺序本身有意义。

Adaptive-k，白话说，就是“不预先写死拿几段，而是看相似度曲线在哪儿突然掉下去”。它的好处是几乎不增加延迟，适合对时延敏感的线上系统。它解决的是“拿多少段”的问题，不直接解决“这几段彼此是否重复”的问题，所以和 MMR 组合通常比单独使用更稳。

三类方案可以这样比较：

| 方案 | 解决的主问题 | 优点 | 代价 | 适用边界 |
| --- | --- | --- | --- | --- |
| MMR / AdaGReS | 重复片段挤占窗口 | 提升覆盖，改造成本低 | 需要调惩罚项 | 冗余知识库、多跳问答 |
| 层级 RAG | 长文档、跨层级推理 | 结构清晰，适合长上下文 | 系统复杂，摘要有误差 | 法律、论文、复杂报告 |
| Adaptive-k | 固定 k 不稳 | 延迟低，容易接入 | 不处理重复 | 在线检索、预算敏感系统 |

适用边界也要说清楚。若你的问题本身是单跳事实查询，例如“某 API 的返回字段是什么意思”，简单贪心填充未必差；若你的语料高度结构化且文档很短，复杂集合优化可能收益有限。真正值得投入的场景，是“有限预算 + 明显冗余 + 需要跨事实拼接答案”。

---

## 参考资料

1. Micheal Lanham, “Your Agent’s RAG Is Bleeding Tokens: How AdaGReS Beats Top-K Retrieval”, Medium, 2026-01-10. 介绍将上下文选择视为集合优化问题，并给出约 30% token 节省、约 15% 正确率提升的公开工程案例。  
   链接：https://medium.com/%40Micheal-Lanham/your-agents-rag-is-bleeding-tokens-how-adagres-beats-top-k-retrieval-8b8e50870e56

2. Wayland Zhang, “Chapter 8: Memory Architecture”, AI Agent Architecture, 2026. 给出 MMR 的实用公式与 `lambda=0.7` 的默认建议。  
   链接：https://www.waylandz.com/ai-agent-book-en/chapter-08-memory-architecture/

3. Atharva Khollam, “Hierarchical RAG: Multi-Level Retrieval and Context Condensation for Long-Context Reasoning”, Medium, 2025-10-20. 介绍文档级到片段级的多层检索与逐层压缩。  
   链接：https://medium.com/%40atharvadude617/hierarchical-rag-multi-level-retrieval-and-context-condensation-for-long-context-reasoning-dce197a5da45

4. Indranil Chandra, “Adaptive-k Retrieval: A Smarter Way to Trim Context in RAG”, Medium. 说明如何依据相似度分布的最大落差动态决定检索深度。  
   链接：https://indranildchandra.medium.com/adaptive-k-retrieval-a-smarter-way-to-trim-context-in-rag-687a399a67fe

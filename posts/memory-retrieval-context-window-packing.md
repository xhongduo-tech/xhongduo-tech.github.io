## 核心结论

记忆检索不只是在“找什么”，更是在“有限窗口里先放什么、舍弃什么、放到多细”。这一步叫上下文窗口填充策略，指把候选记忆片段塞进模型输入的规则。对多跳推理任务来说，它往往比单次召回分数更关键，因为多跳问题需要多条互补证据，而不是一堆意思接近的高分段落。

三类策略最常见：

| 策略 | 决策原则 | 优点 | 主要风险 | 适合场景 |
|---|---|---|---|---|
| 贪心填充 | 按相关性从高到低塞到满 | 简单、快、易实现 | 冗余高，容易挤掉第二条线索 | 问题单跳、候选很短 |
| MMR 多样性填充 | 同时考虑相关性与去重 | 多跳效果通常更稳 | 需要调 $\lambda$，实现略复杂 | 多跳问答、Agent 记忆检索 |
| 分层摘要填充 | 先粗摘要，再按需展开细节 | 节省窗口，适合长文 | 摘要质量差会丢关键信息 | 长期记忆、超长上下文 |

核心判断标准不是“拿到多少段”，而是“窗口内是否保留了互补线索”。一个常见经验是：如果任务需要两条以上证据链，单纯 top-k 往往不够，MMR 往往比贪心更稳；如果单条文档本身很长，分层摘要比继续加 chunk 更有效。

---

## 问题定义与边界

问题定义可以写成一句话：在固定 token 预算下，从检索候选池中决定谁先进入上下文窗口、谁被跳过、是否保留相似片段，以及要保留到什么粒度。

这里有三个边界必须先定清楚。

第一，`token 上限`。token 可以理解成模型读入文本时的容量单位。窗口再大也不是无限的，记忆片段多了以后，任何一个新片段都会挤占已有空间。

第二，`候选粒度`。粒度就是切分得多细。你可以拿 article 级整篇文章，也可以拿 paragraph 级段落，还可以拿 chunk 级小片段。粒度越细，精确命中越容易；粒度越细，也越容易出现重复片段淹没窗口。

第三，`任务需要的证据条数`。单跳问题常常一条证据就够，多跳问题需要两条或更多互补证据。如果系统不知道这一点，就会把窗口当作普通排序题处理，结果把同一路径的相似片段塞满。

对零基础读者，可以用“背包”来理解。上下文窗口像背包，token 是容量。贪心 top-k 相当于只按“价值分”往里装；MMR 则是在每次装入时额外问一句：“这个东西是不是和包里已有物品太像了？”分层摘要则是“先装目录和摘要，真的需要时再展开正文”。

一个实用的流程可以抽象成下面这样：

```text
用户问题
  ↓
向量检索/混合检索
  ↓
候选池 candidates
  ↓
重排与填充策略
  ├─ 贪心：按 sim(q, d) 排序
  ├─ MMR：按 relevance - redundancy 排序
  └─ 分层：先 coarse summary，再 fine chunk
  ↓
上下文窗口 context
  ↓
模型推理
```

“填充策略”讨论的边界并不包含底层 embedding 模型训练，也不包含模型本体参数更新。它讨论的是推理时的输入组织问题，属于典型的 context engineering，也就是上下文工程。

---

## 核心机制与推导

MMR 的全称是 Maximal Marginal Relevance，可以直译为“最大边际相关性”。白话解释是：每次新选一个候选时，不只看它和问题有多像，还要看它是不是跟已选内容太重复。

公式是：

$$
\mathrm{MMR}_i=\lambda \cdot \mathrm{sim}(q,i)-(1-\lambda)\cdot \max_{j\in S}\mathrm{sim}(i,j)
$$

其中：

- $q$ 是查询问题
- $i$ 是当前候选片段
- $S$ 是已经选入窗口的片段集合
- $\mathrm{sim}$ 是相似度函数，常用 cosine similarity
- $\lambda \in (0,1)$ 是权重，控制“相关性”和“多样性”的平衡

这里的 trade-off 可以直接理解成拉锯：

- $\lambda$ 越大，越偏向“和问题最像”
- $\lambda$ 越小，越偏向“和已选内容不重复”

经验上，$\lambda \approx 0.7$ 常作为一个稳定起点，因为它保留了相关性优先，但已经开始抑制冗余。

### 玩具例子：A、B、C 三段候选为什么会选 A→C→B

假设用户问：“某模型为什么在长上下文推理中失败？”

检索后得到三段候选：

- A：讲注意力退化，和问题相似度 0.92
- B：也讲注意力退化，但只是 A 的近义重复，相似度 0.88
- C：讲检索片段排序错误导致关键证据缺失，相似度 0.85

再设：

- $\mathrm{sim}(B, A)=0.9$
- $\mathrm{sim}(C, A)=0.5$
- $\lambda=0.7$

第一次选择时，$S$ 为空，通常直接拿最高相关的 A。

第二次对 B、C 打分：

$$
\mathrm{MMR}_B=0.7 \times 0.88 - 0.3 \times 0.9 = 0.346
$$

$$
\mathrm{MMR}_C=0.7 \times 0.85 - 0.3 \times 0.5 = 0.445
$$

所以第二个会选 C，而不是 B。原因非常直接：B 虽然更相关，但它和 A 太像；C 稍微没那么像问题，却带来了另一条因果路径。多跳任务依赖的正是这种“第二条路线索”。

这也是为什么 top-k 容易失败。top-k 看的是：

$$
\text{score}(i)=\mathrm{sim}(q,i)
$$

它不会惩罚重复，因此经常选出一串同义片段。

### 分层摘要为什么有效

分层摘要的核心不是“把文章缩短”，而是“先保留结构，再按需下钻”。粗粒度摘要可以理解为目录级信息，细粒度 chunk 是正文级信息。对于长文档或长期记忆，先把 coarse summary 放进窗口，只有当模型确认某个主题需要展开时，再加载对应细节。

这背后的思路是：

$$
\text{总窗口预算} = \text{全局概览预算} + \text{局部细节预算}
$$

如果一开始就把细节全部灌入，那么全局概览会缺失，模型常常知道很多局部句子，却不知道这些句子在整体任务里各自扮演什么角色。

可以把它理解为“先看目录再翻章节”。对短文没必要，但对长文、会话记忆、项目文档库非常重要。

---

## 代码实现

下面给一个适合初学者理解的可运行 Python 版本。它不依赖向量数据库，用字典模拟“与查询的相似度”和“片段之间的相似度”，重点展示 MMR 的循环逻辑。

```python
from typing import List, Dict, Tuple

def mmr_select(
    candidates: List[str],
    query_sim: Dict[str, float],
    pair_sim: Dict[Tuple[str, str], float],
    token_len: Dict[str, int],
    max_tokens: int,
    lam: float = 0.7,
) -> List[str]:
    selected = []
    used_tokens = 0
    remaining = set(candidates)

    def sim(a: str, b: str) -> float:
        if (a, b) in pair_sim:
            return pair_sim[(a, b)]
        if (b, a) in pair_sim:
            return pair_sim[(b, a)]
        return 0.0

    while remaining:
        best_item = None
        best_score = float("-inf")

        for item in remaining:
            if used_tokens + token_len[item] > max_tokens:
                continue

            if not selected:
                score = query_sim[item]
            else:
                redundancy = max(sim(item, s) for s in selected)
                score = lam * query_sim[item] - (1 - lam) * redundancy

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break

        selected.append(best_item)
        used_tokens += token_len[best_item]
        remaining.remove(best_item)

    return selected


candidates = ["A", "B", "C"]
query_sim = {"A": 0.92, "B": 0.88, "C": 0.85}
pair_sim = {
    ("A", "B"): 0.90,
    ("A", "C"): 0.50,
    ("B", "C"): 0.40,
}
token_len = {"A": 80, "B": 80, "C": 80}

result = mmr_select(
    candidates=candidates,
    query_sim=query_sim,
    pair_sim=pair_sim,
    token_len=token_len,
    max_tokens=160,
    lam=0.7,
)

assert result == ["A", "C"], result
print(result)
```

这段代码体现了一个关键实现原则：`retrieve candidates -> while window not full -> compute MMR -> append best -> update selected set`。真正工程里通常会再做两件事：

1. 预先缓存候选间相似度，避免每轮重复算 embedding。
2. 在填充前先做 chunk 截断，避免单个片段过长直接吃掉预算。

输入输出结构可以先固定成下面这样：

| 结构 | 类型 | 含义 |
|---|---|---|
| `candidates` | `List[Chunk]` | 检索返回的候选片段 |
| `selected` | `List[Chunk]` | 已经放进窗口的片段 |
| `window` | `str` 或 token buffer | 最终给模型的上下文 |
| `query_sim` | `Dict[id, float]` | 候选与问题的相似度 |
| `pair_sim` | `Dict[(id,id), float]` | 候选之间的相似度 |

如果要加分层摘要，常见做法是把候选分成两层：

- coarse 层：文档摘要、段落主题句、章节标题
- fine 层：原始 chunk、句级证据、代码片段

填充时先拿 coarse 层判断“哪些主题值得展开”，再到 fine 层补细节，而不是直接在所有细粒度 chunk 上暴力 top-k。

---

## 工程权衡与常见坑

真实系统里，填充策略的目标不是单独优化一个排序分数，而是同时平衡正确率、延迟、token 成本和稳定性。

先看一个高层对比：

| 策略 | 冗余比例 | 多跳 F1 表现 | 实现复杂度 |
|---|---|---|---|
| top-k 贪心 | 高 | 基线 | 低 |
| MMR | 中低 | 常见可提升约 11% 左右 | 中 |
| 分层摘要 | 低 | 长文场景更稳 | 中高 |

这里“冗余比例”指被选中的上下文中，语义高度重复的片段占比。它一高，窗口就会出现“看起来很多，实际上有效信息很少”的问题。

### 真实工程例子

在多 Agent RAG 场景里，像 SQuAD、HotpotQA 这类任务经常要求模型综合多段证据。若只用 top-k，相似片段容易集中在一条路径上，例如都在解释“人物 A 的背景”，却漏掉“人物 A 与事件 B 的关系”。一些工程报告和论文结果显示，引入混合召回、rerank、MMR 或近似的多样性填充后，相比纯 top-k 流水线，F1 可能出现双位数提升，研究摘要里给出的量级约为 11.5%。这个数字的含义不是“MMR 永远 +11.5%”，而是说明填充策略足以决定多跳问题能否过线。

常见坑可以直接列出来：

| 坑 | 触发条件 | 结果 | 规避方法 |
|---|---|---|---|
| 只按 top-k 填充 | 候选高度相似 | 冗余塞满窗口，第二条证据缺失 | 用 MMR 或 rerank 去重 |
| $\lambda$ 过高 | 过度追求相关性 | 退化成贪心 | 从 0.7 起调，小批量验证 |
| $\lambda$ 过低 | 过度追求多样性 | 选入不够相关的片段 | 观察准确率和人工可读性 |
| chunk 太大 | 单片段过长 | 一段占满窗口 | 先做 chunk 上限裁剪 |
| 不做分层摘要 | 长期记忆或长文档 | token overflow、信息冲突 | 先 coarse，再 fine |
| 只评估召回率 | 忽略最终回答质量 | 检索看似好，回答仍错 | 直接评估最终任务 F1/EM |

一个常被忽略的事实是：多跳任务不是“把最多信息放进去”，而是“把足够且互补的信息放进去”。因此工程上应先估算任务需要几条独立线索，再决定窗口里最多留多少同类片段。否则系统会把全部预算浪费在同一路径的反复确认上。

---

## 替代方案与适用边界

MMR 很实用，但不是唯一方案。它解决的是“已召回候选如何去重并保留互补信息”，如果你的问题出在召回阶段本身，单靠 MMR 不够。

可以把常见方案放到一张决策表里看：

| 方案 | 适用场景 | 优点 | 局限 | 复杂度 |
|---|---|---|---|---|
| top-k 贪心 | 单跳、小窗口、低成本系统 | 最简单 | 冗余高 | 低 |
| MMR | 多跳问答、Agent 记忆注入 | 去重有效，易落地 | 需要候选间相似度 | 中 |
| rerank + hybrid recall | 检索源复杂，既有关键词又有语义 | 召回更全，排序更稳 | 组件更多 | 中高 |
| 分层摘要 | 长文、长期记忆、超长对话 | 节省窗口，保留全局结构 | 依赖摘要质量 | 中高 |
| 多 Agent 分工检索 | 问题复杂、知识源异构 | 能拆路径并行找证据 | 调度和成本更高 | 高 |

几个边界判断很重要：

第一，短文档场景未必需要分层摘要。如果单条证据本身只有几百 token，先摘要再展开可能纯属额外开销，MMR 就够了。

第二，MMR 不能替代混合召回。若 dense retrieval 漏掉关键词精确匹配项，MMR 只是对错误候选做更好排序，无法凭空补回遗漏文档。此时要先做 dense + sparse 的 hybrid recall，再谈填充。

第三，多 Agent 系统里，MMR 更像“窗口整理器”，不是总调度器。多个 Agent 分头检索后，仍然需要一个最终的汇总阶段去控制冗余，否则每个 Agent 都可能把自己最相关的材料重复送进来。

对新手来说，可以这样记：

- 只有短答案、单跳问题：先用 top-k，足够简单
- 需要两条以上证据：优先试 MMR
- 文档很长、记忆很多：考虑分层摘要
- 数据源复杂、召回不稳：先补 hybrid recall 或 rerank

---

## 参考资料

1. **Context Engineering Guide**  
   用途：支撑上下文窗口填充的定义、MMR 思路、$\lambda \approx 0.7$ 的经验设定，以及 relevance 与 diversity 的权衡框架。  
   链接：https://www.linkedin.com/pulse/context-engineering-guide-cost-latency-optimization-agent-khatri-thg9c?utm_source=openai

2. **MMR Diversify Search Results**  
   用途：支撑 MMR 的直观解释、A/B/C 玩具例子、为什么 top-k 容易选出重复证据，以及何时使用多样性重排。  
   链接：https://app.ailog.fr/en/blog/guides/mmr-diversification?utm_source=openai

3. **Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization**  
   用途：支撑分层摘要填充、粗粒度到细粒度的多层检索结构，以及长上下文场景下“先保留结构再展开细节”的工程思路。  
   链接：https://www.microsoft.com/en-us/research/publication/improving-long-context-summarization-with-multi-granularity-retrieval-optimization/?utm_source=openai

4. **Multi-agent Retrieval-Augmented Generation for Enhancing Answer Generation and Knowledge Retrieval**  
   用途：支撑多 Agent、多跳问答场景下，混合召回与重排策略相对传统 top-k 流水线可带来双位数量级 F1 提升的工程证据。  
   链接：https://link.springer.com/chapter/10.1007/978-3-032-05179-0_22?utm_source=openai

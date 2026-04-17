## 核心结论

ICL，in-context learning，即“不改模型参数，只靠提示词里的少量示例让模型临时学会任务”，它的性能对示例选择非常敏感。对大多数 few-shot 场景，第一条经验不是“多给例子”，而是“给对例子”。

更准确地说，示例选择至少有三个层次：

1. 先选相关样本。最基本且常常有效的方法，是从候选库里检索与当前输入语义最接近的 $k$ 个示例。Liu 等人的工作说明，语义检索通常明显优于随机抽样。
2. 再控顺序。已有研究和后续分析都表明，大模型对靠近 prompt 末尾的示例更敏感，这可以理解为近因偏置，即“最后看到的模式更容易被直接套用”。因此，把“最相关的示例放最后”通常比随机顺序更稳。
3. 最后控覆盖。只按相似度排序，容易拿到多个“几乎同义”的例子，导致 prompt 重复展示同一种模式，却漏掉关键推理片段。Coverage-based 方法把“示例集合是否覆盖测试输入的重要成分”显式写成目标函数，尤其适合语义解析、组合推理、数据到文本这类任务。

一句话总结：ICL 不是单点检索问题，而是“相关性 + 顺序 + 多样性”的联合优化问题。只做相似度检索，通常已经比随机好；但想把性能做稳，必须进一步处理顺序和覆盖。

---

## 问题定义与边界

先把问题写清楚。设示例集合为

$$
D = \{(x_i, y_i)\}_{i=1}^{n}
$$

其中 $x_i$ 是输入，$y_i$ 是答案。给定一个排列 $\pi = (\pi_1,\pi_2,\dots,\pi_n)$，我们把示例按顺序拼成上下文：

$$
C_{\pi}=\Omega(x_{\pi_1}, y_{\pi_1}) \oplus \cdots \oplus \Omega(x_{\pi_n}, y_{\pi_n})
$$

这里 $\Omega(\cdot)$ 表示模板化，即“把样本写成模型能读的自然语言格式”；$\oplus$ 表示拼接。对测试输入 $x_t$，模型预测为：

$$
y_{t,\pi}=\arg\max_y P\big(v(y)\mid C_\pi \oplus \Omega(x_t, *)\big)
$$

这两个式子已经给出本文的边界：ICL 的目标不是训练一个新分类器，而是在给定候选示例池时，决定“选谁、放哪、放几个”。

这个问题容易被误解成“找最近邻”就结束了。实际上它至少受四个约束：

| 约束 | 含义 | 对策略的影响 |
|---|---|---|
| 上下文长度 | prompt 可容纳的 token 有上限 | $k$ 不能无限增大 |
| 任务类型 | 分类、抽取、语义解析、生成任务差异很大 | 相似度和覆盖的收益不同 |
| 候选池大小 | 可检索示例多还是少 | 决定是否值得建索引或聚类 |
| 输出结构性 | 答案是否有固定模式 | 结构越强，覆盖越重要 |

玩具例子先看最简单版本。

假设当前测试问题是“把 `Add a meeting with Jim and his manager for tomorrow` 转成结构化指令”。候选库里有三个示例，它们和测试输入的语义相似度分别是：

| 示例 | 相似度 |
|---|---:|
| “Add an appointment with Jim for tomorrow” | 0.92 |
| “Schedule a meeting with Doug and his boss for next week to review” | 0.88 |
| “Book a restaurant for tonight” | 0.45 |

如果只做 top-2 检索，会选 0.92 和 0.88。接下来顺序就重要了。把 0.88 放前、0.92 放后，等于把更接近测试输入的模式留在末尾；模型更容易把最后这个模板直接映射到当前问题。这个例子不复杂，但足够说明：示例选择不是单个分数，而是“集合 + 顺序”的问题。

真实工程里更明显。比如做 4 类客服情绪分类，用户输入可能是“我已经扣款两次了还没人回复”。若随机选 4 个示例，可能 3 个都是“退款抱怨”，1 个是“物流延迟”，模型会被上下文分布带偏。若先检索语义相关样本，再把与当前投诉最接近的案例放在最后，准确率和稳定性通常都会提升。

本文讨论的是推理时的示例选择，不讨论微调后再检索，也不讨论训练一个专用 reranker 的重方案。原因很简单：对零基础到初级工程师，最先落地的通常是训练免费或轻训练的方法。

---

## 核心机制与推导

### 1. 为什么“最近邻”有效

Liu 等人 2022 的核心发现很朴素：和当前测试输入语义更接近的示例，通常更能提示模型“这是哪一类任务、该用什么输出模式”。这和传统检索类似，但 ICL 更敏感，因为示例不仅提供信息，还在提示模型“如何推理”。

如果把每个输入编码成向量，最常见的相关性分数是余弦相似度：

$$
\text{sim}(x, z)=\frac{e(x)\cdot e(z)}{\|e(x)\|\|e(z)\|}
$$

这里 $e(\cdot)$ 是 embedding，也就是“把文本压成一个定长语义向量”。最近邻方法的局限也正来自这里：一个向量只能表达整体相似，不能说明“测试输入里哪些关键 token 被哪个示例覆盖了”。

### 2. 为什么“只看整体相似”会重复

Gupta 等人 2023 提出的思路更细。BERTScore-Recall，简称 BSR，可以理解成“测试句里的每个 token，在候选示例里能不能找到足够像的对应片段”。它不是把整句压成一个点，而是逐 token 看匹配。

其定义可写为：

$$
\text{BSR}(x,z)=\sum_{x_i\in x} w(x_i)\max_j x_i^\top z_j
$$

其中：

- $x_i$ 是测试输入的第 $i$ 个 token 表示
- $z_j$ 是候选示例的第 $j$ 个 token 表示
- $w(x_i)$ 是 token 权重，可简单取均匀值，也可用 IDF
- $\max_j x_i^\top z_j$ 表示“测试 token 在示例中能找到的最佳匹配”

白话说，BSR 不是问“这两个句子整体像不像”，而是问“测试输入的重要成分，在这个示例里有没有被覆盖到”。

这就解释了为何有时第二相似的例子比第一相似的例子更有用。整体上它可能没那么像，但它补上了一个关键推理部件。

### 3. 为什么要从单示例扩展到集合

单个示例的相关性再高，也可能只覆盖一种模式。于是 Gupta 等人进一步定义集合覆盖分数：

$$
\text{setcov}(x_{\text{test}}, Z)=\sum_{s\in S_{x_{\text{test}}}} \max_{z\in Z} c(s,z)
$$

这里：

- $Z$ 是已选示例集合
- $S_{x_{\text{test}}}$ 是测试输入的“显著成分集合”，可理解成 token、n-gram 或结构片段
- $c(s,z)$ 是某个成分 $s$ 被示例 $z$ 覆盖的程度

它的意思很直接：对测试输入里的每个关键成分，只记“当前集合中谁覆盖得最好”，然后把这些最佳覆盖求和。这样一来，重复示例的边际收益会下降，因为它们覆盖的是同一批成分。

这类目标常见于子模优化。子模可以白话理解成“先加入的元素更值钱，后加入的类似元素越来越不值钱”。因此贪心算法往往够用：每一步加入能让集合覆盖增益最大的示例。

### 4. 顺序为什么还单独重要

即使你选对了集合，顺序仍然会影响输出。ACL 2024 关于 example order 的研究总结了一个很关键的现象：Causal LLM 往往对后部示例更敏感。对实际工程，这意味着一个简单策略常常有效：

先检索出 top-$k$，再按相关性升序排列，把最相关示例放在最后。

注意这里不是说“永远最后最好”，而是说在很多 few-shot 任务里，这是一条成本极低、收益稳定的启发式。尤其当你不能为每个任务单独调顺序时，这个规则很实用。

### 5. 数据到文本任务为什么更强调多样性

DCCS，Double Clustering-based In-Context Example Selection，可以理解成“双阶段聚类选例”。第一阶段按输入数据相似度聚类，保证相关；第二阶段在簇内按参考文本聚类，保证表达多样性。

它解决的是另一类常见问题：如果做数据到文本生成，只给模型多个语义接近、措辞也接近的示例，模型容易学到单一表述风格，导致输出模式塌缩。DCCS 的思路等于先找“同类问题”，再从同类问题里挑“不同写法”。

因此，最近邻、coverage、双聚类三种方法并不冲突。它们对应的是三个不同目标：

| 方法 | 核心目标 | 最适合的任务 |
|---|---|---|
| kNN 检索 | 找最相关示例 | 通用分类、QA、基础抽取 |
| Set-BSR / coverage | 覆盖关键推理成分 | 语义解析、组合推理 |
| DCCS | 同时保相关与表述多样性 | 数据到文本生成 |

---

## 代码实现

下面给一个可运行的玩具实现。它不依赖深度学习库，只用人工构造的 token 相似度矩阵，模拟 Set-BSR 的贪心选择和“最相关样本放末尾”的排序规则。

```python
from typing import Dict, List, Tuple

def bsr_recall(test_tokens: List[str], example_tokens: List[str], sim: Dict[Tuple[str, str], float]) -> float:
    # 对测试句每个 token，找示例中的最佳匹配
    total = 0.0
    for t in test_tokens:
        best = 0.0
        for e in example_tokens:
            best = max(best, sim.get((t, e), sim.get((e, t), 0.0)))
        total += best
    return total / max(len(test_tokens), 1)

def setcov(test_tokens: List[str], chosen_examples: List[List[str]], sim: Dict[Tuple[str, str], float]) -> float:
    total = 0.0
    for t in test_tokens:
        best = 0.0
        for ex in chosen_examples:
            for e in ex:
                best = max(best, sim.get((t, e), sim.get((e, t), 0.0)))
        total += best
    return total / max(len(test_tokens), 1)

def greedy_select(test_tokens: List[str], candidates: Dict[str, List[str]], sim: Dict[Tuple[str, str], float], k: int):
    selected = []
    remaining = dict(candidates)
    while len(selected) < k and remaining:
        best_name = None
        best_score = -1.0
        for name, tokens in remaining.items():
            score = setcov(test_tokens, [candidates[n] for n in selected] + [tokens], sim)
            if score > best_score:
                best_score = score
                best_name = name
        selected.append(best_name)
        remaining.pop(best_name)
    return selected

def reorder_by_relevance(test_tokens: List[str], chosen: List[str], candidates: Dict[str, List[str]], sim: Dict[Tuple[str, str], float]):
    scored = []
    for name in chosen:
        score = bsr_recall(test_tokens, candidates[name], sim)
        scored.append((score, name))
    # 升序排列，最相关的示例放最后
    scored.sort()
    return [name for _, name in scored]

test = ["meeting", "manager", "tomorrow"]

candidates = {
    "A": ["appointment", "jim", "tomorrow"],
    "B": ["meeting", "boss", "next_week", "review"],
    "C": ["restaurant", "tonight"],
    "D": ["meeting", "manager", "tomorrow"],
}

sim = {
    ("meeting", "appointment"): 0.6,
    ("meeting", "meeting"): 1.0,
    ("manager", "boss"): 0.8,
    ("manager", "manager"): 1.0,
    ("tomorrow", "tomorrow"): 1.0,
    ("tomorrow", "next_week"): 0.2,
}

selected = greedy_select(test, candidates, sim, k=2)
ordered = reorder_by_relevance(test, selected, candidates, sim)

assert len(selected) == 2
assert "D" in selected  # 完全匹配必须被选中
assert ordered[-1] == "D"  # 最相关示例被放到末尾
assert setcov(test, [candidates[n] for n in selected], sim) >= bsr_recall(test, candidates["C"], sim)

print("selected:", selected)
print("ordered:", ordered)
```

这段代码体现了两个核心步骤：

1. `greedy_select` 做集合选择，目标是最大化覆盖。
2. `reorder_by_relevance` 做最终排序，让最相关示例出现在 prompt 末尾。

真实工程例子可以这样落地。假设你在做客服问答：

1. 离线把历史 `问题 -> 标准回复` 编码，存到向量索引。
2. 在线对当前用户问题做 embedding 检索，先拿到 20 个近邻。
3. 若任务只是分类或简答，直接取 top-4 到 top-8。
4. 若任务含明显的多步推理或结构映射，再跑一次 coverage rerank，过滤重复模式。
5. 最后按相关性升序拼接，让最相关示例贴近当前 query。

工程伪代码如下：

```python
query_emb = embed(query)
pool = vector_index.search(query_emb, top_k=20)
chosen = coverage_rerank(query, pool, k=6)   # 简单任务可跳过
chosen = sort_by_similarity_asc(query, chosen)
prompt = concat_demos(chosen) + format_query(query)
answer = call_llm(prompt)
```

如果任务是数据到文本，比如“把结构化订单数据生成人类可读摘要”，则可把第 2 步换成 DCCS 风格流程：先按输入结构检索近簇，再在簇内选不同表述的中心样本，避免 prompt 中全是同一种写法。

---

## 工程权衡与常见坑

最常见的坑不是“模型太弱”，而是“把检索当成唯一问题”。

第一，重复示例会浪费上下文。你拿到 6 个 top-k 近邻，看上去都很像，但其中 4 个可能只是同义改写。对于分类任务，这有时问题不大；对语义解析或生成任务，这会让模型只看到一种推理路径。Set-BSR 的价值就在这里，它强迫你问“这个新例子补了什么”。

第二，顺序不能忽略。很多团队会做检索，但把结果按数据库返回顺序直接拼进去。这样等于把一个本来可能稳定增益的系统，变成带随机噪声的系统。若没有更复杂的 order search，至少做一条简单规则：按相似度升序排，最高相似的放最后。

第三，$k$ 不是越大越好。上下文里示例一多，收益会边际递减，甚至下降。原因通常有三个：噪声进入、示例彼此冲突、当前 query 离末尾太远。实际项目里，4 到 8 shot 是最常见的有效区间，先验证这个范围，再考虑扩展。

第四，任务不同，最优策略不同。Gupta 等人的结果里，Set-BSR 在组合性强的语义解析任务上收益很大，但在 IID 的简单分类或推理数据集上，单例 BSR 已经很强，集合覆盖不一定继续提升。不要把某篇论文的最佳方法机械复制到所有任务。

第五，生成任务还要考虑 token 成本。DCCS 的一个现实价值是降低“每个样本都做全量相似度检索”的成本。论文里对 E2E 数据集报告了明显的检索时间节省，DCCS-Batch 在 batch size 5 时还带来较大的 token 节省。这说明有些场景下，选例策略不只是为了准确率，也是为了吞吐和成本。

一个简化对比：

| 策略 | 优点 | 代价 | 常见失败方式 |
|---|---|---|---|
| 随机 few-shot | 实现最简单 | 性能波动大 | 示例不相关、顺序不稳 |
| 纯 kNN | 上线快，通常显著优于随机 | 易重复 | 覆盖不足 |
| kNN + 末尾排序 | 成本极低，常有稳定增益 | 仍可能重复 | 只修顺序，不修集合 |
| Set-BSR | 解释性强，适合组合任务 | 计算更重 | 候选池太小则优势有限 |
| DCCS | 兼顾相关与多样性，且更省检索成本 | 更适合特定任务形态 | 泛化到普通分类未必划算 |

---

## 替代方案与适用边界

如果你的任务是标准文本分类、FAQ 匹配、简单抽取，优先级通常是：

`kNN 检索 -> 末尾排序 -> 小规模 A/B 测试`

这是性价比最高的路径。实现复杂度低，不依赖额外训练，且通常已经能把随机 few-shot 的不稳定性压下去。

如果任务是语义解析、工具调用参数生成、需要覆盖多个约束条件的组合推理，优先考虑 Set-BSR 或类似 coverage 方法。原因不是它“更高级”，而是这些任务里“缺一个关键部件就全错”，覆盖比平均相似更重要。

如果任务是数据到文本生成，例如结构化商品信息生成描述、表格生成摘要、报表转自然语言，DCCS 更合适。它的核心不是证明最近邻没用，而是指出“只要相关，不要多样”会让输出模式变窄。论文中的结果说明，DCCS-Batch 在若干数据到文本基准上能在保留生成质量的同时降低检索和 token 成本。

什么情况下该考虑微调而不是继续折腾 ICL 选例？

1. 输出格式长期稳定，且有足够高质量标注数据。
2. 你对延迟特别敏感，不想每次都注入很多上下文。
3. 你的知识变化不频繁，否则微调后的知识很快过期。
4. 你需要的是固定风格和一致性，而不是动态知识接入。

因此，一个实用判断是：

- 动态知识、多变问题：优先 ICL / 检索增强
- 结构推理、多约束映射：优先 coverage
- 数据到文本、批量生成：优先 DCCS 类方法
- 稳定格式、大规模重复任务：再考虑微调

最重要的边界是，不存在“普适最优”的示例选择器。你真正该优化的是任务失败模式：是“不相关”、是“重复”、是“顺序错”，还是“成本太高”。定位错了，方法选得再复杂也没用。

---

## 参考资料

- Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, Weizhu Chen. 2022. *What Makes Good In-Context Examples for GPT-3?* ACL Anthology: https://aclanthology.org/2022.deelio-1.10/
- Qi Guo, Leiyu Wang, Yidong Wang, Wei Ye, Shikun Zhang. 2024. *What Makes a Good Order of Examples in In-Context Learning*. ACL Anthology: https://aclanthology.org/2024.findings-acl.884/
- Shivanshu Gupta, Matt Gardner, Sameer Singh. 2023. *Coverage-based Example Selection for In-Context Learning*. ACL Anthology: https://aclanthology.org/2023.findings-emnlp.930.pdf
- Yicheng Li et al. 2025. *How to quickly select good in-context examples in large language models for data-to-text tasks*. Cambridge University Press: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/EB6EAFC6A135AE4DD80A0EC110311F69/S2977042425100101a.pdf/div-class-title-how-to-quickly-select-good-in-context-examples-in-large-language-models-for-data-to-text-tasks-div.pdf

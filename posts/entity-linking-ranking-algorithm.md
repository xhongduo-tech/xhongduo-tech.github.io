## 核心结论

实体链接排序算法的任务，不是“找出所有可能实体”，而是“在已经召回的候选集合里，选出与当前 mention 最一致的那个实体”。`mention` 就是文本里指向某个实体的那段字符串，比如句子里的 “Jordan” 或 “Apple”。

一个稳定的排序器，通常不会只看单一信号。工程上最常融合四类特征：

1. 局部上下文语义：当前句子前后到底在说什么。
2. mention 与实体名称相似度：字面上像不像这个实体。
3. 候选实体先验流行度：这个 mention 过去最常指向谁。
4. 文档内实体共现一致性：同一篇文档里的其他实体是否支持这个选择。

因此，排序阶段常写成一个打分问题：

$$
s(m,e\mid D)=\lambda_c s_{ctx}(c_m,d_e)+\lambda_n s_{name}(m,e)+\lambda_p \log p(e\mid m)+\lambda_g s_{coh}(e,E_D)
$$

最终输出分数最高的候选：

$$
\hat e=\arg\max_e s(m,e\mid D)
$$

现代系统里，传统线性加权和学习排序仍然有价值，但高精度场景更常见的做法是 `cross-encoder`。`cross-encoder` 可以理解为“把 mention 上下文和实体描述拼在一起，让模型直接判断二者是否匹配”的联合编码器。如果文档里多个 mention 相互关联，还可以再做 `collective linking`，也就是“让整篇文档里的实体选择彼此约束，尽量落在同一主题子图上”。

---

## 问题定义与边界

实体链接通常分成三步：`mention 检测 → 候选召回 → 候选排序`。本文只讨论第三步。

排序阶段的输入和输出可以先定义清楚：

| 项 | 含义 | 例子 |
|---|---|---|
| mention | 待消歧的文本片段 | `Apple` |
| 候选实体 | 召回模块给出的若干实体 | `Apple Inc.`、`Apple (fruit)` |
| 局部上下文 | mention 周围的词、句子、段落 | “发布新款芯片” |
| 文档级信息 | 同文档其他 mention 的候选与已选实体 | `Tim Cook`、`Cupertino` |
| 输出标签 | 最终选中的实体 | `Apple Inc.` |

这一步的边界很重要：

- 排序器无法补救“正确实体根本没被召回”的问题。
- 排序器不负责实体库构建，也不负责别名清洗。
- 排序器不等于文本分类，它必须在“给定候选集合”内做选择。
- 排序器也不直接解决开放世界问题，比如文本提到的实体根本不在知识库里。

看一个最短例子。句子是：“Apple 发布新款芯片。”  
候选召回先给出两个候选：`Apple Inc.` 和 `Apple (fruit)`。  
排序器这时做的不是重新去全库搜索，而是在这两个候选里比较谁更合理。因为上下文里出现“发布”“芯片”这类偏公司和产品发布的词，所以公司实体得分更高。

用流程图式文字描述就是：

`原文 → 检测 mention → 召回 top-k 候选 → 对每个候选打分 → 排序 → 输出 top-1 或 top-k`

这里的 `top-k` 指“保留分数最高的前 k 个候选”，常用于后续精排或人工审核。

---

## 核心机制与推导

先看为什么要多信号融合。因为每个信号都只解决问题的一部分。

- 上下文分数：解决“这句话到底在讨论什么主题”。
- 名称分数：解决“字符串本身像不像这个实体”。
- 先验分数：解决“这个词平时最常指向谁”。
- 一致性分数：解决“同文档其他实体是否支持这个选择”。

### 玩具例子：Jordan

句子是：“Jordan 与 Israel 的边境局势再次升温，Amman 已召开紧急会议。”  
`Jordan` 的候选有两个：

1. `Jordan`，中东国家。
2. `Michael Jordan`，篮球运动员。

设权重为 $\lambda_c=0.5,\lambda_n=0.2,\lambda_p=0.1,\lambda_g=0.2$。  
这里 $\lambda$ 是“权重”，白话讲就是“每种信号在总分里占多大比重”。

| 候选 | $s_{ctx}$ | $s_{name}$ | $\log p(e\mid m)$ | $s_{coh}$ | 总分 |
|---|---:|---:|---:|---:|---:|
| Jordan（国家） | 0.8 | 0.1 | 0.3 | 0.9 | 0.63 |
| Michael Jordan | 0.2 | 0.6 | 0.7 | 0.1 | 0.31 |

为什么会这样：

- 名称分数上，`Michael Jordan` 更像，因为字面包含完整人名。
- 先验上，很多语料里 “Jordan” 也确实经常指向球员，所以它不低。
- 但上下文里有 `Israel`、`Amman`、`边境局势`，明显偏国家与地缘政治。
- 文档级一致性里，`Israel` 和 `Amman` 进一步支持国家语义。

所以总分最高的是国家 `Jordan`。

### 从线性加权到学习排序

最基础的做法是人工设计特征，再用线性模型或树模型学习权重。优点是可解释，缺点是表达能力有限。它隐含的思想很直接：正确实体应当在各个关键信号上整体占优。

进一步，可以把问题写成 pairwise ranking。`pairwise ranking` 白话讲就是“不直接学绝对分数，而是学正确候选要比错误候选高多少”。

常见损失函数是：

$$
L=\max(0,\gamma-s(m,e^+\mid D)+s(m,e^-\mid D))
$$

其中：

- $e^+$ 是正确实体。
- $e^-$ 是错误实体。
- $\gamma$ 是间隔，表示“正确候选至少要比错误候选高出这么多”。

它的含义很简单：如果正确候选没有比错误候选高 enough，就继续罚；如果已经高出足够间隔，就不罚。

这种训练方式有两个好处：

1. 更贴近排序目标，而不是普通分类目标。
2. 更适合“同一个 mention 下多个候选竞争”的结构。

### 为什么 cross-encoder 往往更强

传统特征模型会把上下文、名称、先验拆开算，再做融合。`cross-encoder` 则直接把它们一起送入模型。常见输入形式类似：

`[CLS] mention 左右上下文 [SEP] 实体标题 [SEP] 实体描述 [SEP]`

例如：

`[CLS] Apple 发布新款芯片，Tim Cook 在发布会上强调能效 [SEP] Apple Inc. [SEP] American multinational technology company ... [SEP]`

这样模型可以显式建模“发布会”“芯片”“Tim Cook”与“technology company”之间的细粒度对齐关系，而不是先分别编码再粗糙地做相似度。它的代价是慢，因为每个 mention 和每个候选都要联合过一遍模型。

### 真实工程例子：新闻聚合

在新闻聚合系统中，一篇稿件里可能同时出现：

- `Apple`
- `Tim Cook`
- `Cupertino`
- `iPhone`

如果只看 `Apple` 自己的局部窗口，排序器多半已经能选到 `Apple Inc.`。但如果再引入 `collective linking`，系统会发现这几个实体在知识图谱里高度相关，于是公司候选的全局一致性分数继续提高。反过来，如果一篇文档里只有一个很短的句子：“Apple 很甜。” 那么全局一致性几乎帮不上忙，局部上下文和名称分数更关键。

---

## 代码实现

先给一个适合初学者理解的最小流程：输入 mention 和候选，给每个候选算分，然后排序输出。下面代码可直接运行，演示一个简化版排序器。

```python
from math import log

def token_overlap_score(context, entity_desc):
    context_tokens = set(context.lower().split())
    entity_tokens = set(entity_desc.lower().split())
    if not context_tokens:
        return 0.0
    return len(context_tokens & entity_tokens) / len(context_tokens)

def name_score(mention, entity_name):
    m = mention.lower()
    e = entity_name.lower()
    if m == e:
        return 1.0
    if m in e or e in m:
        return 0.6
    return 0.0

def prior_score(prior_prob):
    return log(prior_prob + 1e-8)

def coherence_score(entity_name, doc_entities):
    related = {
        "Jordan (country)": {"Israel", "Amman", "Middle East"},
        "Michael Jordan": {"NBA", "Chicago Bulls", "basketball"},
    }
    support = related.get(entity_name, set()) & set(doc_entities)
    return min(len(support) / 2.0, 1.0)

def score_candidate(mention, context, candidate, doc_entities, weights):
    s_ctx = token_overlap_score(context, candidate["desc"])
    s_name = name_score(mention, candidate["name"])
    s_prior = prior_score(candidate["prior"])
    s_coh = coherence_score(candidate["name"], doc_entities)
    total = (
        weights["ctx"] * s_ctx
        + weights["name"] * s_name
        + weights["prior"] * s_prior
        + weights["coh"] * s_coh
    )
    return total

def rank_entities(mention, context, candidates, doc_entities):
    weights = {"ctx": 0.5, "name": 0.2, "prior": 0.1, "coh": 0.2}
    scored = []
    for c in candidates:
        s = score_candidate(mention, context, c, doc_entities, weights)
        scored.append((c["name"], s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

mention = "Jordan"
context = "Jordan discussed border tensions with Israel and officials in Amman"
doc_entities = ["Israel", "Amman", "Middle East"]

candidates = [
    {
        "name": "Jordan (country)",
        "desc": "Middle East country bordering Israel with capital Amman",
        "prior": 0.35,
    },
    {
        "name": "Michael Jordan",
        "desc": "American basketball player associated with NBA and Chicago Bulls",
        "prior": 0.55,
    },
]

ranked = rank_entities(mention, context, candidates, doc_entities)
assert ranked[0][0] == "Jordan (country)"
assert ranked[0][1] > ranked[1][1]
print(ranked)
```

这个例子故意简化了四个点：

1. `candidate_generator` 没展开，只假设候选已经给好。
2. `ctx_score` 用词重叠代替真实语义模型。
3. `prior_score` 直接吃一个先验概率。
4. `coherence_score` 用手写关联表代替知识图谱推断。

工程实现通常是两段式：

1. 粗排：快速筛出 top-k 候选。
2. 精排：对 top-k 用更重的模型重算分数。

伪代码可以写成：

```text
candidates = candidate_generator(mention, doc)
topk = recall_topk(candidates, k=50)

for e in topk:
    e.score = score_candidate(mention, local_context, e, doc_entities)

reranked = sort_by_score(topk)
best = reranked[0]
```

如果用神经排序器，精排输入常是：

`mention 上下文 + [SEP] + 实体标题 + [SEP] + 实体描述`

这样做的原因不是形式美观，而是为了让模型在 token 级别同时看到“当前文本在说什么”和“这个实体是什么”。

---

## 工程权衡与常见坑

排序阶段最常见的误解，是“只要换更强模型就能显著提升效果”。实际并不总是如此。很多线上问题根本不在排序器，而在召回、先验质量、实体库更新和上下文截断。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| `top-k` 召回不足 | 正确实体不在候选中，排序必错 | 先优化别名表、召回模型、候选覆盖率 |
| 先验过强 | 高频实体压制上下文，歧义词易误连 | 对先验降权、做温度校准、按领域重估先验 |
| 长上下文截断 | 关键信号落在窗口外，语义分数失真 | 分句打分、滑窗、层级聚合 |
| 噪声共现 | collective linking 被弱相关实体带偏 | 只保留高置信边，限制图传播深度 |
| 训练线上不一致 | 离线指标高，线上错误多 | 混入目标域数据，做线上日志回流训练 |

几个典型坑值得展开说。

先验过强时，系统会“想当然”。例如 `Jordan` 在训练语料中经常是球员，模型就容易忽略当前文档其实在谈中东政治。这个问题在新闻、金融、医疗领域特别明显，因为领域文本的实体分布和百科语料差异很大。

长上下文截断也常被低估。`cross-encoder` 虽然强，但输入长度有限。如果 mention 真正相关的信息出现在前后两段之外，模型看到的只是一个被切断的局部片段，性能会下降。长文档场景里，常见做法不是“无脑加长输入”，而是“先抽取 mention 所在句、相邻句、标题、摘要，再做层级融合”。

再看真实工程例子。新闻聚合中，如果同文档同时出现 `Tim Cook`、`Cupertino`、`iPhone`，那么 `Apple Inc.` 的一致性分数会被显著抬高。但如果系统把某个不相关的 `Apple Records` 也高置信加入文档图，噪声边就会把整体图结构带偏。全局方法的收益和风险是绑定的：它很强，但前提是图边质量足够好。

---

## 替代方案与适用边界

排序不是“越复杂越好”，而是“在成本、延迟、可解释性、精度之间做平衡”。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 规则/线性加权 | 可解释、便宜、上线快 | 表达能力有限，对复杂语义弱 | 小规模系统、冷启动、强规则领域 |
| pairwise learning-to-rank | 目标贴近排序，较稳健 | 依赖标注质量，特征工程仍重 | 中等规模、已有标注样本 |
| cross-encoder 精排 | 精度高，语义建模强 | 慢，算力成本高 | 高价值查询、离线处理、top-k 精排 |
| collective linking | 能利用文档全局一致性 | 易受噪声共现影响，实现复杂 | 长文档、多 mention 强关联场景 |
| bi-encoder + cross-encoder | 兼顾召回速度与精排效果 | 系统链路更复杂 | 大规模知识库、工业级 EL |

三种方案放在同一句子里看差异最清楚。句子：“Apple 在 Cupertino 发布新款 iPhone。”

- 规则/线性模型会显式看 `发布`、`iPhone`、`Cupertino` 这些词，并给 `Apple Inc.` 更高分。
- `cross-encoder` 会把整句与实体描述联合编码，直接学习“发布会 + 产品 + 地点”与科技公司的语义一致性。
- `collective linking` 则进一步利用同文档其他 mention，如果同时还有 `Tim Cook`，公司实体会继续增强。

什么时候不该上复杂模型？至少有三类情况：

1. 候选很少，且歧义低。比如内部系统里的标准设备名映射，线性模型就够。
2. 实体库变化极快。复杂神经模型的重训、蒸馏、部署成本可能高于收益。
3. 延迟极敏感。在线广告、实时风控场景里，几十毫秒的额外精排代价未必能接受。

所以更常见的工业实践不是二选一，而是分层：  
`高召回候选生成 + 轻量粗排 + top-k cross-encoder 精排 + 可选 collective rerank`。  
这比“一开始就对全量候选跑大模型”更现实。

---

## 参考资料

1. [Deep Joint Entity Disambiguation with Local Neural Attention](https://aclanthology.org/D17-1277/)：这篇主要说明局部上下文、实体先验和文档级联合推断如何组合，是现代神经实体消歧的重要起点。
2. [Scalable Zero-shot Entity Linking with Dense Entity Retrieval](https://aclanthology.org/2020.emnlp-main.519/)：这篇主要说明 `bi-encoder + cross-encoder` 两阶段架构怎样兼顾大规模召回与精排效果。
3. [Neural Collective Entity Linking](https://aclanthology.org/C18-1057/)：这篇主要解决图结构中的全局一致性建模问题，适合理解 collective linking。
4. [Neural entity linking: A survey of models based on deep learning](https://journals.sagepub.com/doi/10.3233/SW-222986)：这篇是综述，适合先建立候选生成、局部排序、全局推断的整体地图。
5. [facebookresearch/BLINK](https://github.com/facebookresearch/BLINK)：这是公开工程实现，最有参考价值的点是两阶段检索与重排如何落到可运行系统。
6. [Learning to Rank for Information Retrieval](https://www.nowpublishers.com/article/Details/INR-016)：这份资料主要帮助理解 pairwise ranking、listwise ranking 等排序学习基本思想，虽然不只讲实体链接，但方法层非常通用。

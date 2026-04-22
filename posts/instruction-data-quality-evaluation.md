## 核心结论

指令数据质量评估，是判断一条 `instruction-response` 对是否值得进入微调训练集。`instruction` 是用户给模型的任务说明，`response` 是期望模型学会的回答，二者合起来才是一条训练样本。

核心判断不是“这条回答读起来顺不顺”，而是它能不能教模型做对事。高质量指令数据通常同时满足四点：指令清晰、回复正确、指令与回复相关、和其他样本有足够差异。只满足其中一两点，训练价值就会明显下降。

| 维度 | 高质量样本 | 低质量样本 |
|---|---|---|
| 指令清晰度 | “用 3 句话解释 Transformer 中自注意力的作用” | “讲讲这个模型” |
| 回复正确性 | 说明 token 间相关性、权重计算和上下文聚合 | 只说“它让模型更聪明” |
| 相关性 | 回答紧扣 Transformer | 转去介绍 CNN 或泛泛谈 AI |
| 多样性 | 覆盖不同任务、领域、格式和难度 | 大量样本只是换词复述同一模板 |

新手版玩具例子：同样是“解释一下 Transformer”，一条回答如果准确解释自注意力、位置编码和训练目标，通常比只说“Transformer 很强大、效果很好”更有训练价值。后者看起来流畅，但信息密度低，模型学不到稳定的任务能力。

真实工程例子：做 SFT，也就是监督微调时，团队可能先收集 30 万条指令数据，再用质量评分器打分，保留高分样本，然后用 embedding 做相似度去重，最后只用 6k 到 10k 条数据训练。这样做的目标是降低噪声、减少重复、节省训练成本。

一个实用判断框架是：

```text
输入样本
  ↓
质量是否足够高？——否→ 丢弃
  ↓ 是
是否与已选样本过于相似？——是→ 降权或丢弃
  ↓ 否
进入训练集
```

质量分可以写成：

$$
q = \sum_{k=1}^{6} k \cdot p_k
$$

其中 $p_k$ 表示样本属于第 $k$ 个质量等级的概率，$q$ 是最终质量分。

---

## 问题定义与边界

指令数据质量评估 = 判断一条 `instruction-response` 对对微调是否有用。

这里评估的对象不是单独的指令，也不是单独的回答，更不是最终模型效果。它评估的是“这条输入和输出放在一起，能不能成为一条好的训练样本”。

| 评估对象 | 不评估什么 | 为什么 |
|---|---|---|
| `instruction-response` 对 | 只看指令 | 指令清楚但回复错误，仍然不能训练 |
| 样本训练价值 | 最终模型能力 | 模型效果还受基座模型、训练参数、数据配比影响 |
| 领域内可用性 | 绝对真理 | 有些任务允许多个正确答案 |
| 整体数据分布 | 单条样本孤立好坏 | 高分重复样本太多也会伤害泛化 |

术语说明：

| 术语 | 白话解释 |
|---|---|
| 指令 | 用户希望模型完成的任务 |
| 回复 | 训练时希望模型输出的答案 |
| 样本 | 一组指令和回复 |
| 多样性 | 数据之间不要都长得一样，任务、表达、领域和难度要有覆盖 |

新手版例子：同样一句“总结这篇文章”，如果回答准确覆盖核心信息、结构清楚、没有编造内容，就适合进入训练集；如果只是在复述标题，或者把文章没有说的内容补进去，就不适合。

边界示例：文案生成、头脑风暴、开放式问答这类任务不一定有唯一标准答案。此时“是否唯一正确”不是关键，但“是否满足指令、是否有用、是否重复、是否违反领域约束”仍然重要。

所以质量评估不能简单等同于事实核查。事实核查只问“对不对”，指令数据质量评估还要问“清不清楚、相关不相关、对训练有没有贡献、和已有数据是不是太像”。

---

## 核心机制与推导

Deita 的一个重要思路是：不要把“好坏”当成一个神秘连续分数，而是先把样本质量分成 1 到 6 级，再计算期望分数。期望分数是按概率加权后的平均等级。

具体形式是：

$$
p_k = \mathrm{softmax}(z_k), \quad k \in \{1,2,3,4,5,6\}
$$

$$
q = \sum_{k=1}^{6} k \cdot p_k
$$

其中 `logits` 是模型输出的原始分数，还没有归一化；`softmax` 是把一组原始分数转换成概率分布的方法；$z_k$ 是第 $k$ 级的 logits，$p_k$ 是第 $k$ 级概率，$q$ 是样本质量分。

玩具数值例子：

```text
p = [0.03, 0.07, 0.10, 0.20, 0.30, 0.30]
q = 1×0.03 + 2×0.07 + 3×0.10 + 4×0.20 + 5×0.30 + 6×0.30
q = 4.57
```

这说明模型认为它大概率属于 5 级或 6 级，也有一部分概率属于中等质量，因此整体是“偏高质量，但不是满分”。

完整流程通常不是只按质量分排序。只看质量分会筛出很多风格相近的高分样本，例如大量“请解释 X 的定义、特点和应用”的模板。更稳妥的方式是先筛质量，再控制多样性。

```text
输入样本池
  ↓
质量评分器打分
  ↓
按阈值保留高分样本
  ↓
用 embedding 表示语义
  ↓
相似度去重或覆盖控制
  ↓
输出最终训练集
```

`embedding` 是把文本转换成向量的方法，语义相近的文本向量距离通常更近。余弦相似度是常用的向量相似度指标，值越接近 1，表示两个文本越相似。

| 步骤 | 输入 | 输出 | 目的 |
|---|---|---|---|
| 样本读取 | 原始指令池 | 标准化样本 | 统一字段 |
| 质量打分 | 指令和回复 | 质量分 `q` | 去除明显低质数据 |
| 阈值筛选 | 带分样本 | 候选集 | 控制噪声 |
| embedding 计算 | 候选样本文本 | 向量 | 度量语义相似 |
| 多样性过滤 | 向量和样本 | 最终训练集 | 减少重复、扩大覆盖 |

---

## 代码实现

最小实现可以拆成两段：先计算质量分，再做多样性筛选。下面代码不依赖深度学习框架，直接用模拟 logits 和简单词袋 embedding 展示完整逻辑，可直接运行。

```python
import math
from collections import Counter

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def quality_score(logits):
    probs = softmax(logits)
    return sum((i + 1) * p for i, p in enumerate(probs))

def tokenize(text):
    return [w.lower() for w in text.replace(",", " ").replace(".", " ").split()]

def embed(sample):
    text = sample["instruction"] + " " + sample["response"]
    return Counter(tokenize(text))

def cosine(a, b):
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def diversity_filter(samples, threshold=0.75):
    selected = []
    selected_vecs = []
    for s in samples:
        v = embed(s)
        if all(cosine(v, old_v) < threshold for old_v in selected_vecs):
            selected.append(s)
            selected_vecs.append(v)
    return selected

dataset = [
    {
        "id": "a",
        "instruction": "Explain self attention in Transformer.",
        "response": "Self attention lets each token compute weights over other tokens and aggregate context.",
        "logits": [-2.0, -1.2, -0.6, 0.2, 1.1, 1.1],
    },
    {
        "id": "b",
        "instruction": "Explain self attention.",
        "response": "It is very powerful and useful.",
        "logits": [0.4, 0.6, 0.2, -0.1, -0.5, -0.8],
    },
    {
        "id": "c",
        "instruction": "Summarize why data diversity matters for instruction tuning.",
        "response": "Diversity reduces overfitting to repeated templates and improves coverage of tasks.",
        "logits": [-1.8, -1.0, -0.2, 0.5, 1.0, 0.9],
    },
]

for sample in dataset:
    sample["q"] = quality_score(sample["logits"])

kept = [s for s in dataset if s["q"] >= 4.0]
selected = diversity_filter(kept, threshold=0.75)

assert round(dataset[0]["q"], 2) >= 4.5
assert [s["id"] for s in kept] == ["a", "c"]
assert len(selected) == 2
print([(s["id"], round(s["q"], 2)) for s in selected])
```

工程版接入时，评分器通常不是手写 logits，而是来自强模型、专门训练的质量评分器，或 LLM-as-a-judge。LLM-as-a-judge 是让大模型充当评审器，对样本质量给出分数或等级。

一个真实工程流程可以是：

```python
for batch in load_jsonl("raw_instruction_pool.jsonl"):
    scores = quality_scorer.batch_score(batch)
    save_jsonl("scored.jsonl", attach_scores(batch, scores))

candidates = load_where("scored.jsonl", lambda x: x["quality"] >= 4.5)
vectors = embedding_model.encode([x["instruction"] + "\n" + x["response"] for x in candidates])
selected = diversity_select(candidates, vectors, cosine_threshold=0.88, max_samples=10000)
save_jsonl("sft_train_selected.jsonl", selected)
```

| 模块 | 输入 | 输出 | 关键参数 |
|---|---|---|---|
| 质量评分器 | `instruction`, `response` | `quality_score` | 等级数、提示词、阈值 |
| embedding 模型 | 样本文本 | 向量 | 模型类型、最大长度 |
| 多样性筛选 | 候选样本和向量 | 去重后样本 | 相似度阈值、最大样本数 |
| 人工抽检 | 抽样数据 | 校准结果 | 抽样比例、错误类型 |

对 `posts.json` 或内部样本池接入时，关键是先统一字段。例如把 `title`、`summary`、正文片段拼成指令，把标准答案或目标输出放进 `response`。不要在评分阶段混入训练标签之外的元信息，否则评分器可能学到“格式好看就是质量高”的错误信号。

---

## 工程权衡与常见坑

最大的问题是把“流畅”误当成“高质量”。流畅只说明语言表面顺滑，不说明事实正确、任务完成、领域约束满足。一个回答可以语气完整、格式漂亮，但内容空泛或编造事实。

错误案例：

```text
指令：从准确性、成本、延迟三个角度比较 RAG 和微调。
回答：RAG 和微调都很重要。它们可以提升模型效果，也能帮助业务落地。选择时要综合考虑。
```

这段回答没有明显语病，但没有完成“三个角度比较”。正确做法应该逐项比较准确性、成本、延迟，并说明适用条件。

更好的回答结构：

| 角度 | RAG | 微调 |
|---|---|---|
| 准确性 | 依赖检索质量，适合更新快的知识 | 适合学习稳定格式和任务行为 |
| 成本 | 推理时多一次检索和上下文开销 | 训练成本更高，推理流程更简单 |
| 延迟 | 受检索链路影响 | 通常不需要额外检索 |

常见坑如下：

| 坑 | 造成的问题 | 规避方法 |
|---|---|---|
| 只看流畅度 | 幻觉回答被保留 | 加入事实正确性和任务相关性检查 |
| 只按高分排序 | 数据模板化 | 增加 embedding 去重和覆盖采样 |
| 长度偏置 | 长回答更容易得高分 | 按任务类型控制长度，做人工校准 |
| 格式偏置 | 表格、编号被误判为更优 | 让评分标准关注内容而非样式 |
| 位置偏置 | A/B 对比中前者更占优 | 打乱顺序或双向评测 |
| 跨领域漂移 | 通用评分器误判专业数据 | 用领域样本校准或重训评分器 |

新手版反例：指令要求“给出三个不同角度分析”，但数据池里 500 条回答都套用“首先、其次、最后”的固定句式，只替换几个关键词。它们单条看都完整，放在一起却高度重复。模型学到的不是分析能力，而是固定话术。

真实工程中还要注意抽检。自动评分器可以处理大规模数据，但不能替代质量闭环。比较稳的做法是保留一小批人工标注样本，定期检查评分器与人工判断的一致性。若发现评分器长期偏爱长答案、偏爱固定格式、偏爱某个领域，就要调阈值、改提示词，或重新训练评估器。

---

## 替代方案与适用边界

质量评分器适合“数据量大、需要自动筛选”的场景，但不是唯一方案。数据规模小、领域风险高、任务边界清楚时，人工审核和规则过滤往往更可靠。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 自动质量评分器 | 数十万到数百万条样本 | 成本低、速度快、可批处理 | 可能有领域漂移和评分偏置 |
| LLM-as-a-judge | 开放式问答、写作、对话任务 | 标准灵活，解释性较好 | 受位置、长度、格式偏置影响 |
| 规则过滤 | 格式固定、错误模式明确 | 可解释、稳定、便宜 | 覆盖不了语义质量 |
| 人工全审 | 高风险、小规模数据 | 准确、可控 | 成本高、速度慢 |
| 规则过滤 + 人工抽检 | 中等规模、要求稳妥 | 成本和质量平衡 | 需要设计抽样策略 |
| 覆盖度采样 | 多领域、多任务训练集 | 能控制数据分布 | 需要可靠标签或 embedding |

什么时候用自动评分器，可以按下面决策表判断：

| 条件 | 建议 |
|---|---|
| 数据超过 10 万条 | 优先自动评分，再人工抽检 |
| 数据只有几千条且风险高 | 优先人工审核 |
| 数据格式噪声很多 | 先规则清洗，再质量评分 |
| 任务覆盖面很广 | 质量评分后必须做多样性控制 |
| 医疗、法律、金融等高风险领域 | 自动评分只能辅助，必须人工校准 |
| 代码、问答、客服、摘要等常见任务 | 可以使用自动评分器作为第一层筛选 |

开放域问答和代码任务可以用自动评分器先筛出明显低质样本，再通过测试、静态检查或人工抽样提高可靠性。医疗、法律、金融类数据则不能只依赖强模型评分，因为一个看起来流畅的错误回答可能带来严重后果。

最终原则是：自动评分器解决规模问题，人工校准解决可信度问题，多样性控制解决泛化问题。三者缺一项，训练集都容易偏。

---

## 参考资料

1. [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://openreview.net/forum?id=BTKAeLqLMw)
2. [hkust-nlp/deita](https://github.com/hkust-nlp/deita)
3. [hkust-nlp/deita-quality-scorer](https://huggingface.co/hkust-nlp/deita-quality-scorer)
4. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
5. [Data Diversity Matters for Robust Instruction Tuning](https://aclanthology.org/2024.findings-emnlp.195/)

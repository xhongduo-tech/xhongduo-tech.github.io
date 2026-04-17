## 核心结论

Unigram Language Model 分词的核心不是“从小词拼成大词”，而是“先准备一个足够大的候选子词集合，再删掉不重要的子词”。这里的“子词”可以理解为介于单字和整词之间的文本片段，比如 `ing`、`tion`、`机器`、`学习`。这里的“语言模型”不是完整句子生成模型，而是一个只给子词分配概率的概率模型。

它的基本假设很直接：如果一句话被切成一串子词 $S=[s_1,\dots,s_K]$，那么这条切分路径的概率等于各子词概率的乘积：

$$
P(S)=\prod_{k=1}^K P(s_k)
$$

对同一个词或句段，通常不止一种切分，因此观测串 $w$ 的概率是所有合法切分路径概率之和：

$$
P(w)=\sum_{S\in \mathrm{Seg}(w)} P(S)
$$

这两个式子决定了 Unigram 的训练方式：先允许很多候选切分，再用 EM 算法反复估计每个子词的概率，最后按“删掉谁对整体损失最小”来剪枝。EM 算法可以先理解成“先估计每个子词大概用了多少次，再按次数更新概率”的迭代过程。

和 BPE 相比，Unigram 的方向正好相反。BPE 是从字符出发不断合并高频相邻片段，属于自下而上的贪心合并；Unigram 是从大词表出发不断裁剪低贡献片段，属于自上而下的全局剪枝。这个差异带来两个直接结果：

| 维度 | Unigram LM | BPE |
|---|---|---|
| 训练方向 | 大词表逐步裁剪 | 小词表逐步合并 |
| 切分视角 | 比较整条切分路径概率 | 比较局部合并频次 |
| 多候选分词 | 天然支持 | 通常最终只保留确定合并规则 |
| 解码方式 | 常用 Viterbi 找最优切分 | 按合并规则直接切 |

玩具例子可以直接看字符串 `▁ab`。如果当前词表里有 `▁ab`、`▁a`、`b` 三个子词，那么它至少有两条切分路径：

- $[\text{▁ab}]$
- $[\text{▁a}, \text{b}]$

若概率分别为 $P(\text{▁ab})=0.3$、$P(\text{▁a})=0.5$、$P(\text{b})=0.2$，则两条路径概率为：

- $P([\text{▁ab}])=0.3$
- $P([\text{▁a},\text{b}])=0.5\times 0.2=0.1$

所以模型更偏向把它当成一个整体子词。Unigram 的训练目标就是让这种“哪种切分更能解释数据”变成可计算、可优化的过程。

---

## 问题定义与边界

Unigram LM 分词要解决的问题，可以严格表述为：

给定训练语料和目标词表大小 $|V|=M$，在大量候选子词中选择一个固定大小的子词集合，并为每个子词分配概率，使训练语料在该模型下的总似然尽可能高。

这里的“似然”可以理解为“这套词表和概率参数解释训练语料的能力”。解释得越好，训练语料出现的概率越高。

问题边界同样重要。Unigram 并不是在真实语言学意义上寻找“正确词边界”，它做的是面向模型训练的统计切分。它不保证切分符合人类直觉，也不保证每个子词有独立语义，只保证在当前假设下整体概率较优。

可控制因素与受限条件可以分开看：

| 可控制因素 | 受限条件 |
|---|---|
| 初始候选词表大小 | 假设子词彼此独立 |
| 目标词表大小 | 只能从候选集中剪枝，不能凭空生成任意新片段 |
| 每轮剪枝比例 | 切分质量依赖候选生成是否充分 |
| EM 迭代轮数 | 似然优化是近似的，不保证全局最优 |
| 最小子词长度、字符覆盖策略 | 真实语义依赖未被显式建模 |

这意味着 Unigram 的效果高度依赖两个前提。

第一，初始候选必须够全。也就是训练一开始就要把可能有价值的片段尽量放进来，否则后续再聪明的剪枝也无法保留一个根本不存在的子词。

第二，独立假设必须能接受。所谓“独立假设”，就是把一条切分路径的概率写成各子词概率的乘积，默认相邻子词之间没有更复杂的条件依赖。这显然是近似，但换来的好处是训练和解码都能用动态规划高效完成。

对零基础读者，可以把这个边界理解成一句话：模型不是在“理解句子”，而是在“用一套概率化积木，尽量便宜地拼出训练语料”。

真实工程里，SentencePiece 之所以适合多语种和无空格语言，关键也在这里。它不要求先做空格分词或人工词典切词，而是直接在原始文本上建立候选子词，再让概率和剪枝机制决定保留哪些片段。这种做法对中文、日文、韩文以及混合语料都更统一。

---

## 核心机制与推导

Unigram 的训练可拆成四步：候选生成、E 步、M 步、剪枝。

### 1. 候选生成

候选生成就是先得到一个远大于目标大小的子词表。常见做法是从语料中抽取频繁子串，再加入所有基础字符，保证任何文本都能被切开。基础字符的作用很关键，它是“兜底片段”，否则有些字符串可能完全无法切分。

### 2. E 步：计算期望计数

E 步的目标是计算每个子词的期望计数 $C(s)$。这里的“期望计数”可以理解为：考虑所有可能切分后，一个子词平均相当于在语料中出现了多少次。

对一个样本 $w$，某个切分 $S$ 的后验概率为：

$$
P(S\mid w)=\frac{P(S)}{P(w)}
$$

于是子词 $s$ 在整个语料上的期望计数可写为：

$$
C(s)=\sum_{w\in \mathcal{D}}\sum_{S\in \mathrm{Seg}(w)} P(S\mid w)\cdot n(s,S)
$$

其中 $n(s,S)$ 表示子词 $s$ 在切分 $S$ 中出现的次数。

这一步不能暴力枚举所有切分，因为路径数通常指数增长，所以实际实现会用前向-后向算法。前向-后向可以理解成“用动态规划把所有路径概率压缩计算出来”，避免重复枚举。

### 3. M 步：更新子词概率

得到 $C(s)$ 后，M 步就是归一化更新：

$$
P(s)=\frac{C(s)}{\sum_{t\in V} C(t)}
$$

含义非常直白：谁在所有可能切分里承担了更多“解释责任”，谁的概率就上升。

如果继续用 `▁ab` 这个玩具例子，假设语料中反复出现 `▁ab`，而且每次作为整体切分的后验概率都更高，那么经过几轮 E/M 迭代后，`▁ab` 的期望计数就会高于 `▁a` 和 `b` 的组合贡献，它的概率会继续上升。这样会形成正反馈：越能解释数据的子词，越容易在下一轮继续被选中。

### 4. 剪枝：删除损失最小的子词

只做 EM 不够，因为初始词表太大。Unigram 的关键在于每轮训练后都评估“删掉哪个子词损失最小”。

理想做法是计算删除某子词 $s$ 后总似然的下降量：

$$
\Delta_s = \mathcal{L}(V) - \mathcal{L}(V\setminus \{s\})
$$

其中 $\mathcal{L}(V)$ 是当前词表下训练语料的对数似然。$\Delta_s$ 越小，说明这个子词越不重要，越适合删掉。

工程上常用近似指标，比如根据期望计数、替代路径是否存在、或者删除后最优路径变化程度来估计。也可能设置计数阈值 $\tau_e$，例如仅保留 $C(s)\ge \tau_e$ 的子词，再结合目标词表大小做排序裁剪。

这一步和“直接删概率最小的子词”不同。概率小不等于贡献小。一个低频长词可能只在少量样本中出现，但一旦删掉，它所在样本只能退化成很多短片段，整体似然会明显下降。它像桥梁构件，出现次数不多，但结构价值高。

真实工程例子是多语种预训练模型。比如用 SentencePiece 为中英日混合语料训练固定大小词表时，英语高频片段容易压制低资源语言。如果只看局部频率，低资源语言的关键子词会被快速吞没；而 Unigram 用整条路径似然和剪枝损失来评估，更有机会保留“低频但不可替代”的片段，这也是它常被用于统一多语种词表的原因之一。

最终解码时，常用 Viterbi 算法。Viterbi 可以理解成“只找概率最大的那条路径”的动态规划算法。训练时要考虑所有路径；推理切分时通常只取最优路径，速度更高。

---

## 代码实现

下面给一个可运行的简化版实现。它不包含完整前向-后向和真实剪枝评分，但能准确展示 Unigram 的核心结构：候选词表、路径概率、期望计数近似、剪枝和重新归一化。

```python
from collections import defaultdict

def segmentations(text, vocab):
    # 返回 text 的所有合法切分
    results = []

    def dfs(i, path):
        if i == len(text):
            results.append(path[:])
            return
        for token in vocab:
            if text.startswith(token, i):
                path.append(token)
                dfs(i + len(token), path)
                path.pop()

    dfs(0, [])
    return results

def segmentation_prob(seg, probs):
    p = 1.0
    for token in seg:
        p *= probs[token]
    return p

def expected_counts(corpus, probs):
    counts = defaultdict(float)
    total_loglik = 0.0

    for text in corpus:
        segs = segmentations(text, probs.keys())
        path_probs = [segmentation_prob(seg, probs) for seg in segs]
        z = sum(path_probs)
        assert z > 0, f"{text} 无法被当前词表切分"
        total_loglik += z

        for seg, p in zip(segs, path_probs):
            posterior = p / z
            for token in seg:
                counts[token] += posterior

    return counts, total_loglik

def normalize(counts):
    total = sum(counts.values())
    return {token: c / total for token, c in counts.items() if c > 0}

def prune(probs, counts, keep_size, protected_tokens=None):
    protected_tokens = set(protected_tokens or [])
    items = []
    for token, prob in probs.items():
        score = counts.get(token, 0.0)
        items.append((token, score, prob))

    # 简化策略：优先保留受保护 token，再按期望计数排序
    keep = {token for token in protected_tokens if token in probs}
    candidates = sorted(
        [x for x in items if x[0] not in keep],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    for token, _, _ in candidates:
        if len(keep) < keep_size:
            keep.add(token)

    new_probs = {token: probs[token] for token in keep}
    z = sum(new_probs.values())
    new_probs = {token: p / z for token, p in new_probs.items()}
    return new_probs

# 玩具语料
corpus = ["ab", "ab", "ab", "a"]
candidate_vocab = {
    "a": 0.25,
    "b": 0.15,
    "ab": 0.45,
    "ba": 0.05,
    "aa": 0.10,
}

# 基础字符作为兜底，避免剪枝后无法切分
protected = {"a", "b"}

counts, likelihood = expected_counts(corpus, candidate_vocab)
probabilities = normalize(counts)
probabilities = prune(probabilities, counts, keep_size=3, protected_tokens=protected)

assert "a" in probabilities and "b" in probabilities
assert abs(sum(probabilities.values()) - 1.0) < 1e-9

best_ab = max(
    segmentations("ab", probabilities.keys()),
    key=lambda seg: segmentation_prob(seg, probabilities)
)

assert best_ab in (["ab"], ["a", "b"])
print("updated_probs =", probabilities)
print("best segmentation for 'ab' =", best_ab)
```

这段代码有三个值得注意的点。

第一，`candidate_vocab` 明确体现“从大词表开始”。这和 BPE 从字符逐步合并完全不同。

第二，`expected_counts` 中的 `posterior = p / z` 对应 E 步。它不是简单把最优路径计 1 次，而是按所有路径的相对概率分摊责任。

第三，`protected_tokens` 用来保护基础字符。这是工程实现中常见的安全措施，否则词表一旦剪得过猛，部分样本可能失去可切分路径。

如果把它写成训练流程伪代码，就是：

```python
candidate_vocab = build_large_seed_vocab(corpus)
probabilities = init_uniform(candidate_vocab)

while len(candidate_vocab) > target_vocab_size:
    counts = expectation_step(corpus, candidate_vocab, probabilities)
    probabilities = maximization_step(counts)
    candidate_vocab = prune_low_impact_tokens(candidate_vocab, probabilities, counts)
    probabilities = renormalize(probabilities, candidate_vocab)
```

真实系统会把 `expectation_step` 和 `prune_low_impact_tokens` 做得更复杂，例如：

- 用前向-后向替代暴力枚举所有切分
- 用 Viterbi 或近似损失评估删除某个子词的影响
- 记录每轮被剪掉的子词，便于回滚和诊断
- 保留特殊 token，如 `<unk>`、`<s>`、`</s>`

---

## 工程权衡与常见坑

Unigram 在工程上不是“理论对了就一定好用”，它的关键难点主要集中在初始候选、剪枝节奏和多语种平衡。

最常见的误区，是把它当成“概率版 BPE”。它们都输出子词词表，但训练目标和风险点不同。Unigram 更依赖全局概率估计，因此对实现细节更敏感。

| 常见坑 | 触发条件 | 规避策略 |
|---|---|---|
| 初始候选不充分 | 候选只来自很短高频片段，长词覆盖差 | 用 suffix array、LCP 或频繁子串方法扩大候选覆盖 |
| 剪枝过猛 | 每轮删除比例过高 | 分阶段剪枝，保留缓冲区，逐轮复训 |
| 只按概率删词 | 忽略删除后的替代路径质量 | 使用似然降幅、期望计数或 Viterbi 变化量综合评估 |
| 未保护基础字符 | 罕见样本无合法切分 | 始终保留基础字符和必要特殊 token |
| 多语种失衡 | 大语种高频片段主导训练 | 做采样平衡、分语种温度采样或最小覆盖约束 |
| 训练慢 | 枚举切分过多 | 用动态规划、Trie、并行化和缓存 |
| 推理与训练不一致 | 训练考虑多路径，推理却用不同规则切分 | 明确以 Viterbi 最优路径作为推理标准 |

一个直观比喻是：如果剪枝时只看“当前谁最细、最瘦”，就像砍树时只看树干粗细，可能会先砍掉稀有但支撑局部结构的树。Unigram 之所以强调损失增量，就是为了避免这种局部视角。

真实工程中还有两个具体问题。

第一是内存和速度。对于长文本和大词表，候选匹配数会很多。如果没有 Trie 或前缀索引，前向-后向的常数开销很高。

第二是词表可解释性。Unigram 为了提升似然，可能保留一些“看起来怪异但统计上有效”的片段，例如跨词边界碎片或混合标点片段。这对人类阅读不友好，但对下游模型并不一定是坏事。所以评价它时要以下游任务和压缩率为准，不要只看“分出来像不像词”。

---

## 替代方案与适用边界

Unigram、BPE、WordPiece 都在解决“固定词表下如何覆盖开放词汇”的问题，但出发点不同。

WordPiece 可以理解成“选一个新片段，使训练目标提升最大”的方法，常见于早期 BERT 系列；BPE 更像“每次合并最常一起出现的相邻片段”；Unigram 则是“假设已有很多候选，再从中留下最能解释整语料的一组”。

| 方法 | 训练步骤 | 核心假设 | 词表变化方式 | 更适合的场景 |
|---|---|---|---|---|
| Unigram | 大词表初始化 → EM → 剪枝 | 切分路径概率是子词概率乘积 | 自上而下裁剪 | 多语种、需要多候选分词、关注全局概率 |
| BPE | 从字符开始统计相邻对并合并 | 高频相邻对值得合并 | 自下而上扩张 | 实现简单、训练快、工业部署成熟 |
| WordPiece | 按目标函数选择新片段 | 新片段应最大化语料解释力 | 逐步扩张 | 与既有 Transformer 体系兼容时常用 |

对新手最容易理解的区别是：

- BPE 像“不断把常一起出现的砖块粘起来”
- Unigram 像“先准备很多砖块，再挑一套最能解释整栋房子的砖”

什么时候优先选 Unigram？

- 需要统一处理中英文、日文等不同语言
- 希望保留多候选切分，而不是固定合并规则
- 更关心整体路径概率，而不是局部频次
- 使用 SentencePiece 体系，希望直接在原始文本上训练

什么时候不一定要选 Unigram？

- 训练速度和实现复杂度比切分质量更重要
- 现有系统已深度绑定 BPE 词表
- 语料单一、语言边界清晰、分词需求简单
- 不需要采样式分词或概率化切分

因此，Unigram 的适用边界很明确：它不是所有场景都优于 BPE，而是在“多路径、全局概率、统一文本预处理”这些目标上更自然。

---

## 参考资料

| 资源 | 链接 | 核心内容 |
|---|---|---|
| SentencePiece GitHub | https://github.com/google/sentencepiece | SentencePiece 项目说明，包含 Unigram 和 BPE 两种后端背景 |
| Hugging Face Course: Unigram Tokenization | https://huggingface.co/docs/course/en/chapter6/7 | 用教学方式解释 Unigram 的训练流程、EM 和实际分词行为 |
| Emergent Mind: SentencePiece Unigram Model | https://www.emergentmind.com/topics/sentencepiece-unigram-model | 对 Unigram 模型定义、公式、EM 与剪枝过程的综述 |
| Emergent Mind: Unigram Tokenization Algorithm | https://www.emergentmind.com/topics/unigram-tokenization-algorithm | 给出玩具例子、训练直觉和工程注意事项 |
| Kudo, SentencePiece 相关论文与文档 | https://arxiv.org/abs/1808.06226 | SentencePiece 的设计目标、无预分词训练与子词建模背景 |

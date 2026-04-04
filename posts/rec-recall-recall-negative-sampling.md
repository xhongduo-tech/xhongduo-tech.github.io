## 核心结论

召回层的负采样，本质上是在“算不动全量候选”与“仍然要学出有效边界”之间做折中。负采样不是训练时的附属技巧，而是直接决定模型看到哪些反例、因此学出什么判别边界的核心机制。

对召回模型来说，候选池常常是百万级到千万级 item。假设总共有 $N=10^6$ 个 item，某次训练里用户的真实正例只有 1 个。若使用全量 softmax，模型需要对 $10^6$ 个 item 都算 score，再做归一化，单步复杂度接近 $O(N)$。如果改成采样 $k=64$ 个负例，那么这一步只需要比较 $1+64$ 个 score，复杂度近似变成 $O(k)$，训练才进入可落地区间。

但复杂度降下去，不代表问题解决了。关键在于：你抽到的这 64 个负例，是否真的能代表“模型该学会区分的边界”。随机负样本简单、稳定、覆盖广，但通常太容易；批内负采样高效，因为直接把同一个 batch 里的其他正例当负例，但会被 batch 分布绑架；hard negative 能让边界更紧，却更容易引入假负例，也就是“这个样本其实可能是用户会喜欢的，只是当前日志里没出现”。

因此，召回层的负采样要和损失函数一起看。Sampled Softmax、BPR、InfoNCE 都是在“有限负例”条件下近似原始目标，只是近似方式不同。Sampled Softmax 会显式引入采样分布修正：

$$
o'_j = o_j - \log(k q_j)
$$

$$
\mathcal{L}=-o'_y+\log\sum_{j\in S\cup\{y\}}\exp(o'_j)
$$

这里 $q_j$ 是负例被采到的概率，白话说就是“这个 item 本来就容易被抽到吗”。如果不做这个修正，热门 item 因为更常出现，会在训练中被过度放大，模型学到的是采样器的偏差，不是用户偏好本身。

结论可以压缩成一句话：召回层的负采样决定了学习边界，$k$ 决定计算预算与梯度质量，采样分布决定你在逼近哪个真实目标，损失函数决定这种近似是否可控。

---

## 问题定义与边界

先定义问题。召回层的目标不是最终排序，而是从海量 item 中先筛出一个较小候选集。这里的“召回”可以理解为第一道粗筛，把不可能的内容先去掉，再交给后续精排模型做细致比较。

形式化地说，给定用户或上下文表示 $x$，真实交互 item 记为 $x^+$，模型学习一个打分函数 $f(x, item)$。训练目标通常写成：

$$
\mathcal{L} = \mathbb{E}_{(x,x^+)} \, \mathbb{E}_{x^-\sim p_n}\, \ell(f(x,x^+), f(x,x^-))
$$

这里的 $p_n$ 是负样本分布，白话说就是“你打算从哪里抽负例”。这个分布不是中性选择，它直接规定了模型眼里的“非目标”到底长什么样。

对零基础读者，可以这样理解：你有一个用户，他昨天点了一条篮球视频，这是正例。系统里有一千万条视频，不可能每次训练都把一千万条都拿来和篮球视频比较。所以你每轮只从“用户没点的内容里”抽 $k$ 条出来当负例，让模型学会“这条篮球视频比这 $k$ 条更像用户想要的”。

这个定义有三个边界要讲清楚。

第一，未交互不等于负例。用户没点，不一定是不喜欢，也可能是没看到。所以召回训练里很多“负例”其实是“未观测样本”，也就是没有被证明喜欢、也没有被证明讨厌的样本。

第二，召回层学的是粗粒度边界，不要求严格排序全部 item。它要解决的是“把相关的先捞进来”，而不是“把第 3 名和第 4 名绝对排准”。所以负采样的重点是塑造大范围区分能力，而不是逼近最终业务指标的所有细节。

第三，采样分布与线上分布通常不一致。训练里你可能按流行度采样、按批内样本采样、按相似度采样；线上候选却来自真实内容池。两者差得太远时，训练和部署之间就会出现边界错位。

一个玩具例子可以说明这个边界问题。假设候选池只有 6 个 item：

| item | 类型 | 用户真实偏好 |
|---|---|---|
| A | 篮球集锦 | 喜欢 |
| B | 篮球教学 | 喜欢 |
| C | 足球新闻 | 一般 |
| D | 搞笑短片 | 一般 |
| E | 美食探店 | 不喜欢 |
| F | 财经解读 | 不喜欢 |

如果训练时总是随机抽到 E、F 这种很远的负例，模型很容易学会“A 比 E 好”。但这没什么价值，因为真正难的是把 A 和 C、D、甚至 B 这类近邻内容区分清楚。也就是说，负采样的难度决定了模型在什么边界上变强。

---

## 核心机制与推导

### 1. 为什么全量 softmax 不现实

设 query 表示为 $q$，每个 item 表示为 $v_j$，打分为 $o_j = q^\top v_j$。全量 softmax 的目标是让正例 $y$ 的概率最大：

$$
P(y|q)=\frac{\exp(o_y)}{\sum_{j=1}^{N}\exp(o_j)}
$$

问题在于分母要遍历全部 $N$ 个 item。当 $N$ 是百万级时，训练成本过高，参数更新也会非常慢。

负采样的核心做法是：不再对全部 item 归一化，而只在正例和少量采样负例上近似这个目标。

### 2. Sampled Softmax：有限采样下逼近分类目标

Sampled Softmax 先采样一个负例集合 $S$，大小为 $k$，再只在 $S\cup\{y\}$ 上做 softmax。但如果直接做，会有偏差，因为有些 item 比另一些 item 更容易被抽到。于是要做 logQ 修正：

$$
o'_j = o_j - \log(k q_j)
$$

其中 $q_j$ 是 item $j$ 被采样到的概率。修正后的损失为：

$$
\mathcal{L}=-o'_y+\log\sum_{j\in S\cup\{y\}}\exp(o'_j)
$$

直观理解是：如果某个热门 item 因为出现频繁而更容易进入负例集合，那么它原始 score 的影响要扣掉一部分，避免它仅仅因为“被采得多”就主导训练。

### 3. BPR：只比较正负相对次序

BPR 是 pairwise loss，白话说就是“不要求算概率，只要求正例分数高于负例分数”。常见形式是：

$$
\mathcal{L}_{BPR}=-\log \sigma(s(q,i^+)-s(q,i^-))
$$

如果扩展到 $k$ 个负例，就是把多个负例分别与正例做比较。它的优点是简单，特别适合隐式反馈场景；缺点是没有显式归一化，负例质量对训练影响非常大。

### 4. InfoNCE：把召回问题写成对比学习

InfoNCE 可以看成“有限负例 softmax”的另一种写法：

$$
\mathcal{L}=-\log\frac{\exp(q\cdot k^+/\tau)}{\exp(q\cdot k^+/\tau)+\sum_{i=1}^k \exp(q\cdot k_i^-/\tau)}
$$

这里 $\tau$ 是温度，白话说就是“控制分数差异被放大还是压平的参数”。$\tau$ 越小，模型越强调 hardest negatives；$\tau$ 越大，分布越平缓。

InfoNCE 常被用在双塔召回中，因为它天然支持批内负样本。一个 batch 中每条样本的正例，可以被别的样本当作负例复用，因此效率很高。

### 5. 负例数量 $k$ 为什么重要

$k$ 太小，模型每次看到的反例太少，梯度噪声大，训练出来的边界会松。$k$ 太大，计算成本会上升，而且更容易把假负例塞进训练，导致优化不稳定。

还是用一个玩具例子。假设 query 是“用户最近连续看了篮球战术拆解”，正例是“NBA 季后赛分析”。如果只采 2 个负例，而且恰好是“做菜教程”和“旅游 vlog”，模型几乎不需要学习就能分开；如果采 50 个负例，其中包含“CBA 集锦”“篮球装备评测”“足球技战术讨论”，模型才真正被迫学习语义边界。

所以 $k$ 不只是算力参数，也是任务难度参数。它决定模型到底是在学“明显不一样”，还是在学“看起来很像但仍要分清”。

### 6. 真实工程例子

在短视频或视频平台的召回阶段，用户塔输出一个 embedding，内容塔为每个视频输出一个 embedding。训练时，一个用户样本可能只有 1 个点击视频作为正例。若直接和百万级视频库做全量 softmax，训练吞吐量无法接受。工程上常见做法是：

1. 用随机负采样保证覆盖面。
2. 用批内负样本提升吞吐量。
3. 再从近邻索引或曝光未点日志里补一部分 hard negatives。
4. 对热门 item 做 logQ 或 importance correction，避免采样偏差过大。

这个流程的意义不是“多堆几个技巧”，而是分层控制边界：先学粗边界，再逐步拉紧近邻边界。

---

## 代码实现

下面给一个可运行的 Python 最小实现。它演示三件事：采样负例、做 logQ 修正、计算一个 sampled softmax 风格的 loss。代码是玩具级实现，但结构与工程实践一致。

```python
import math
import random

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def sampled_softmax_loss(query, item_vecs, pos_id, q_dist, k=3, seed=0):
    random.seed(seed)
    all_ids = list(item_vecs.keys())
    neg_pool = [i for i in all_ids if i != pos_id]

    # 按给定分布采样负例；真实工程里通常会去重、避开曝光正例、支持混合采样池
    negatives = random.choices(
        population=neg_pool,
        weights=[q_dist[i] for i in neg_pool],
        k=k
    )

    pos_logit = dot(query, item_vecs[pos_id])

    corrected_logits = [pos_logit]
    labels = [pos_id]

    # 正例也做对应修正，这里为了演示直接使用其分布概率
    corrected_logits[0] = pos_logit - math.log(k * q_dist[pos_id])

    for neg_id in negatives:
        neg_logit = dot(query, item_vecs[neg_id])
        neg_corrected = neg_logit - math.log(k * q_dist[neg_id])
        corrected_logits.append(neg_corrected)
        labels.append(neg_id)

    probs = softmax(corrected_logits)
    loss = -math.log(probs[0])

    return {
        "negatives": negatives,
        "labels": labels,
        "corrected_logits": corrected_logits,
        "probs": probs,
        "loss": loss,
    }

# 玩具向量：query 更接近篮球内容
query = [1.0, 0.8]
item_vecs = {
    "basketball_analysis": [1.0, 0.9],   # 正例
    "basketball_news": [0.9, 0.7],       # hard negative
    "soccer_news": [0.7, 0.2],
    "cooking": [0.1, 0.0],
    "finance": [0.0, 0.2],
}

# 采样分布：热门内容更容易被抽到
q_dist = {
    "basketball_analysis": 0.20,
    "basketball_news": 0.30,
    "soccer_news": 0.25,
    "cooking": 0.15,
    "finance": 0.10,
}

result = sampled_softmax_loss(
    query=query,
    item_vecs=item_vecs,
    pos_id="basketball_analysis",
    q_dist=q_dist,
    k=3,
    seed=42,
)

assert len(result["negatives"]) == 3
assert len(result["probs"]) == 4
assert abs(sum(result["probs"]) - 1.0) < 1e-9
assert result["loss"] > 0

print(result)
```

这段代码虽然简化了很多细节，但已经反映了召回训练的主干流程：

| 步骤 | 作用 | 工程含义 |
|---|---|---|
| 构建负例池 | 从候选中排除正例 | 真实系统里还要排除同 session 已点击项、未来正反馈项 |
| 按 $q_j$ 采样 | 控制负例出现频率 | 可接入随机、流行度、批内、缓存近邻等策略 |
| 计算 corrected logits | 做 logQ 修正 | 避免高频 item 仅因易采样而主导梯度 |
| 在小集合上算 loss | 近似全量目标 | 把训练复杂度从 $O(N)$ 降到 $O(k)$ |

真实工程里还会加入以下组件。

第一，去重和去假负。一个用户昨天点过的视频，今天不能因为本轮未点就直接当负例。

第二，混合采样池。常见配比是“随机负例 + 批内负例 + 近邻 hard negative”。

第三，难度调度。训练初期更多用随机负例，模型稳定后再逐步提高 hard negative 比例。

第四，采样分布持久化。$q_j$ 往往来自 item 频次、曝光分布或某种平滑后的 unigram 分布，不是每步临时现算。

---

## 工程权衡与常见坑

负采样的真正难点，不是“会不会写采样器”，而是“采样器是否把模型推向了错误边界”。下面是召回系统里最常见的坑。

| 坑 | 说明 | 规避 |
|---|---|---|
| 随机负样本偏差 | 抽到的大多是很远的样本，模型只学会区分热门与冷门，边界过松 | 使用 logQ 校正，引入 popularity-aware 分布，并混入少量近邻负例 |
| 批内负样本偏强 | 同 batch 的其他正例被当作负例，若 batch 偏热门或偏单一主题，会放大分布偏差 | 控制 batch 构造策略，跨主题混合，必要时做 debias |
| Hard negative 过难 | 很相似的样本可能其实也是潜在正例，训练会误伤召回能力 | 用 curriculum 策略逐步加难，监控 false positive 比例 |
| $k$ 盲目增大 | 负例更多不一定更好，可能增加噪声、拉低吞吐、破坏稳定性 | 做分桶实验，联合看 recall@K、训练 loss 波动和吞吐 |
| 采样分布不一致 | 训练采样分布与线上候选分布差太远，离线指标和线上效果脱节 | 让负采样部分贴近线上曝光或检索分布，必要时做 importance correction |

这里重点说两个容易被忽视的问题。

### 1. 假负例比“简单负例”更危险

简单负例最多是训练收益小，假负例则会直接把本来该靠近的内容推远。比如电影推荐里，一个 batch 里有很多热门电影，《星际穿越》的正例样本可能会把《盗梦空间》当成负例。对于某些用户，这其实是高度相关内容。模型如果不断收到这种信号，就会学出错误边界。

### 2. batch 分布会悄悄决定你的模型偏好

批内负采样效率很高，但它有个隐含假设：同 batch 其他正例可以安全地当作当前样本负例。这个假设在分布均匀时还成立，在“热门内容集中刷入一个 batch”时就会失效。于是你以为自己在做高效训练，实际上是在反复惩罚热门近邻内容。

工程上常见做法不是二选一，而是混合：随机负例保底覆盖，批内负例保效率，hard negative 保边界强度，再用监控和调度控制副作用。

---

## 替代方案与适用边界

负采样不是所有场景都必须上的默认答案，它的价值取决于候选规模、任务目标和训练资源。

| 方法 | 适用边界 | 说明 |
|---|---|---|
| Full Softmax | $N \lesssim 10^4$ | 候选量较小可直接精确归一化，目标最干净，但复杂度是 $O(N)$ |
| Sampled Softmax / logQ | $N \gg 10^5$ | 召回主流方案，用有限负例逼近分类目标，并修正采样偏差 |
| BPR | 隐式反馈、pairwise 排序强相关场景 | 直接优化正例比分数高于负例，简单有效，但依赖负例质量 |
| InfoNCE / 批内负样本 | 双塔、对比学习、表示学习任务 | 复用 batch 内样本作为负例，吞吐高，但强依赖 batch 分布 |
| Hierarchical Softmax | 标签树结构明确的场景 | 用树形路径降低计算量，但要求类别或 item 有稳定层级结构 |

什么时候可以不用负采样？如果 catalog 只有 8000 个 item，而且训练资源充足，全量 softmax 往往更直接，目标更准确，也少了采样偏差问题。

什么时候必须认真设计负采样？当候选池到 $10^6$ 量级以上时，负采样已经不是优化技巧，而是训练定义的一部分。你不定义好负例来源，等于没有定义清楚模型到底在学什么。

一个真实工程边界是这样的：

1. 小型课程推荐站，课程量 5000，直接 full softmax 可行。
2. 中型资讯站，文章量 20 万，通常会转向 sampled softmax 或 BPR。
3. 大型视频平台，内容量百万以上，基本都会采用“随机 + 批内 + hard negative”的混合策略，并配合 ANN 近邻索引、曝光日志和 false positive 监控。

如果你的业务目标主要是 top-K 召回，hard negative 往往有明显收益；如果你的系统数据噪声很大、未曝光样本很多，那么 hard negative 的使用必须更保守，否则很容易过拟合到伪边界。

---

## 参考资料

- Sampled Softmax Loss Overview（2025）
- Negative Sampling in Model Training（2026）
- Contrastive InfoNCE Loss Overview（2024）
- Correcting the LogQ Correction（RecSys 2025）
- On the Theories Behind Hard Negative Sampling for Recommendation（WWW 2023）
- Candidate Sampling Algorithms（TensorFlow candidate sampling）
- 推荐系统 Candidate Generation / 双塔召回相关工程资料

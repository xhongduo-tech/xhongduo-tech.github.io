## 核心结论

YouTube 深度召回网络的核心，是把推荐系统里的“下一次最可能观看哪个视频”建模成一个极端多分类问题。极端多分类指类别数量非常大，不是几十类，而是百万级、千万级视频都作为候选类别。

它不是一个“用户喜不喜欢某个视频”的二分类模型，而是一个“在全量视频库里，哪个视频最可能成为下一次观看目标”的多分类模型。模型先把用户历史和上下文编码成用户向量 $u$，再把每个视频表示成视频向量 $v_i$，用点积打分：

$$
s_i = u^\top v_i
$$

如果对全量视频做 Softmax，某个视频 $i$ 被观看的概率可以写成：

$$
p(i\mid u)=\frac{e^{s_i}}{\sum_{j\in\mathcal V} e^{s_j}}
$$

其中 $\mathcal V$ 是全部视频集合。分母要遍历所有视频，所以训练和线上都不能直接暴力计算。YouTube 的做法是：训练阶段用负采样近似 Softmax，线上阶段用 ANN 检索 Top-K 视频。

ANN 是 Approximate Nearest Neighbor，意思是近似最近邻检索。它不保证每次都找到数学上绝对精确的最近向量，但能在巨大向量库中快速找到足够接近的候选。

| 维度 | 二分类推荐 | 多分类召回 |
|---|---|---|
| 问题形式 | 判断用户是否喜欢某个视频 | 从全量视频中预测下一个观看视频 |
| 输出 | 一个点击率或偏好分数 | 一批 Top-K 候选视频 |
| 类别规模 | 单个样本二分类 | 百万级视频类别 |
| 典型位置 | 排序层、精排层 | 召回层 |
| 主要目标 | 更准地排序 | 更全地找回可能相关内容 |

一个玩具例子：假设视频库只有 3 个视频，用户向量是 $u=[1,2]$，三个视频向量分别是 $v_1=[2,0]$、$v_2=[0,1]$、$v_3=[1,1]$。点积分数分别是 $2,2,3$，所以第三个视频更可能被召回。

一个真实工程例子：首页推荐先用召回模型从千万级视频库中找出 200 个候选视频，再交给排序模型结合缩略图、频道偏好、地域、设备、时效性等特征重排。召回层的目标不是最终展示顺序，而是尽量把用户可能感兴趣的视频先找出来。

---

## 问题定义与边界

召回是推荐系统的第一层筛选。它的白话定义是：先从巨大内容库里找出一批“可能相关”的候选项，缩小后续排序模型的处理范围。

Top-K 是指按分数取前 K 个结果。例如 Top-200 就是取分数最高的 200 个视频。候选集是召回层输出的一批待排序内容。排序是指在候选集内部重新打分，决定最终展示顺序。

YouTube 深度召回网络的输入不是单个行为，而是一组用户历史和上下文特征。典型输入包括观看历史、搜索词、设备、地域、年龄段、时间等。输出也不是最终推荐列表，而是一批候选视频。

用户表示可以抽象为：

$$
u = f(h, q, d, r, t, \cdots)
$$

其中 $h$ 是观看历史表示，$q$ 是搜索词表示，$d$ 是设备特征，$r$ 是地域特征，$t$ 是时间特征，$f$ 是多层神经网络。

| 项目 | 内容 | 说明 |
|---|---|---|
| 输入特征 | 历史观看、搜索词、设备、地域、时间 | 用来描述用户当前状态 |
| 输出目标 | 下一次观看的视频 | 训练标签来自真实观看行为 |
| 召回层职责 | 找出一批可能相关候选 | 追求覆盖率和检索效率 |
| 排序层职责 | 对候选视频精细排序 | 追求最终点击、观看时长、满意度等目标 |

边界要划清：召回层不负责解决所有推荐问题。它不应该直接承担缩略图吸引力、频道疲劳、内容安全、广告混排、精细时效性等复杂目标。它只负责在大规模视频库里把可能相关的视频找出来。

例如，一个用户刚连续看了多个篮球集锦。召回层不需要立刻判断他到底喜欢“篮球运动”“某个球星”还是“某个频道”。它只需要先把篮球、体育、相似频道、相似观看人群喜欢的视频召回出来。排序层再结合更细特征决定哪些视频排在前面。

---

## 核心机制与推导

YouTube 深度召回网络的核心机制是向量匹配。向量是由一组数字组成的表示，例如 $[0.2, -0.7, 1.3]$。模型把用户编码成向量 $u$，把视频编码成向量 $v_i$，然后用点积计算匹配分数：

$$
s_i = u^\top v_i
$$

点积越大，说明用户向量和视频向量方向越接近，模型认为该视频越可能被观看。

观看历史通常先做平均池化。池化是把多个向量合成一个固定长度向量的方法。若最近观看序列为 $(x_1,\dots,x_m)$，每个视频的 embedding 是 $e(x_t)$，历史表示可以写成：

$$
h=\frac{1}{m}\sum_{t=1}^{m} e(x_t)
$$

embedding 是离散对象的稠密向量表示。白话说，就是把视频 ID、搜索词、地域这类离散符号转换成模型能计算的一串数字。

完整训练目标接近全量 Softmax：

$$
p(i^+\mid u)=\frac{e^{u^\top v_{i^+}}}{\sum_{j\in\mathcal V}e^{u^\top v_j}}
$$

其中 $i^+$ 是真实被观看的视频。但当 $\mathcal V$ 有百万级视频时，每个训练样本都计算全部视频的分母成本太高。

所以训练时使用 sampled softmax。sampled softmax 是一种近似训练方法：每次只取真实正样本和一小批负样本来估计全量 Softmax。形式上可以写成：

$$
p(i^+\mid u,\mathcal N)\approx
\frac{e^{u^\top v_{i^+}}}
{e^{u^\top v_{i^+}}+\sum_{j\in\mathcal N}e^{u^\top v_j}}
$$

其中 $\mathcal N$ 是采样出来的负样本集合。为了减轻采样分布和真实分布不一致的问题，工程上常加入抽样概率修正，例如对候选 $j$ 使用：

$$
\tilde{s}_j = u^\top v_j - \log Q(j)
$$

其中 $Q(j)$ 是视频 $j$ 被采样为负样本的概率。

上线阶段也不能对全量视频逐个点积。做法是提前把所有视频向量建成 ANN 索引，然后把用户向量 $u$ 放进索引里查 Top-K：

$$
\operatorname{TopK}(u)=\arg\max_{i\in\mathcal V}^{K} u^\top v_i
$$

| 阶段 | 计算方式 | 为什么这样做 |
|---|---|---|
| 训练阶段 | 正样本 + 采样负样本 | 避免每步计算百万级 Softmax |
| 离线评估 | 可用候选集指标或近似全量评估 | 检查召回质量，不等同于线上效果 |
| 上线阶段 | ANN 检索 Top-K | 在低延迟下搜索巨大视频库 |
| 后续排序 | 精排模型重打分 | 用更丰富特征优化最终展示 |

流程可以写成：

```text
用户历史观看 -> embedding lookup -> 平均池化 -> 拼接上下文特征
-> MLP 得到用户向量 -> ANN 匹配视频向量 -> 返回 Top-K 候选
```

---

## 代码实现

下面是一个可运行的最小 Python 例子。它不依赖深度学习框架，只演示三件事：历史平均池化、点积打分、Top-K 召回。

```python
import math

def mean_pool(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(scores):
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    return [x / total for x in exps]

def top_k(user_vec, video_vecs, k):
    scored = [(video_id, dot(user_vec, vec)) for video_id, vec in video_vecs.items()]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

video_embeddings = {
    "basketball_1": [2.0, 0.0],
    "programming_1": [0.0, 1.0],
    "basketball_mix": [1.0, 1.0],
}

watched = [
    video_embeddings["basketball_1"],
    video_embeddings["basketball_mix"],
]

user_vec = mean_pool(watched)
scores = [dot(user_vec, v) for v in video_embeddings.values()]
probs = softmax(scores)
candidates = top_k(user_vec, video_embeddings, k=2)

assert user_vec == [1.5, 0.5]
assert candidates[0][0] == "basketball_1"
assert len(probs) == 3
assert abs(sum(probs) - 1.0) < 1e-9
```

更接近工程实现时，可以拆成训练路径和上线路径：

```python
# 伪代码：训练路径
history_emb = mean(lookup(video_embedding_table, watched_video_ids))
search_emb = lookup(search_embedding_table, search_tokens)
device_emb = lookup(device_embedding_table, device_id)
region_emb = lookup(region_embedding_table, region_id)

context_emb = concat([history_emb, search_emb, device_emb, region_emb])
user_vec = mlp(context_emb)

pos_video_vec = lookup(video_embedding_table, watched_next_video_id)
neg_video_vecs = sample_negative_video_vectors(batch_size=100)

pos_score = dot(user_vec, pos_video_vec)
neg_scores = matmul(neg_video_vecs, user_vec)
loss = sampled_softmax_loss(pos_score, neg_scores)
```

```python
# 伪代码：上线路径
history_emb = mean(lookup(video_embedding_table, recent_watched_video_ids))
context_emb = concat([history_emb, search_emb, device_emb, region_emb])
user_vec = mlp(context_emb)

candidates = ann_index.search(user_vec, top_k=200)
return candidates
```

| 环节 | 训练时 | 上线时 |
|---|---|---|
| 输入 | 历史行为、上下文、下一次观看标签 | 当前历史行为、当前上下文 |
| 标签 | 真实下一个观看视频 | 无标签 |
| 负样本 | 从视频库采样 | 不采样 |
| 损失 | sampled softmax loss | 不计算 loss |
| 输出 | 更新模型参数 | Top-K 候选视频 |
| 检索方式 | 正负样本打分 | ANN 索引检索 |

这段拆分很重要。训练阶段的 sampled softmax 是为了让模型学到向量空间；上线阶段的 ANN 是为了从向量空间中快速找候选。两者都在近似全量计算，但近似目的不同。

---

## 工程权衡与常见坑

第一类常见坑是负样本分布偏差。训练时采样的负样本集合 $\mathcal N$ 通常不等于线上真实视频分布。如果采样方式过于简单，模型可能只学会区分正样本和“很容易的负样本”，离线 loss 很好看，但线上召回质量不高。

一种修正思路是使用抽样概率：

$$
\tilde{s}_j = s_j - \log Q(j)
$$

其中 $Q(j)$ 表示样本 $j$ 被采中的概率。高频视频更容易被采中，如果不修正，模型可能对热门视频产生额外偏置。

第二类常见坑是平均池化丢失顺序。平均池化稳定、简单、计算快，但它把早期行为和最近行为一视同仁。若用户上午看篮球，晚上连续看编程教程，简单平均可能把兴趣混在一起。

可以使用时间衰减池化。时间衰减是指越新的行为权重越大：

$$
h=\frac{\sum_{t=1}^{m} w_t e(x_t)}{\sum_{t=1}^{m}w_t},\quad w_t=e^{-\lambda \Delta t}
$$

其中 $\Delta t$ 是行为距离当前的时间间隔，$\lambda$ 控制衰减速度。

```python
import math

def weighted_pool(items):
    # items: [(embedding, hours_ago)]
    weights = [math.exp(-0.1 * hours_ago) for _, hours_ago in items]
    dim = len(items[0][0])
    total = sum(weights)
    pooled = []
    for i in range(dim):
        pooled.append(sum(vec[i] * w for (vec, _), w in zip(items, weights)) / total)
    return pooled

old_basketball = ([2.0, 0.0], 24)
recent_programming = ([0.0, 2.0], 1)

h = weighted_pool([old_basketball, recent_programming])
assert h[1] > h[0]
```

第三类常见坑是训练集泄漏未来信息。泄漏未来信息是指模型训练时看到了预测时本不该知道的数据。例如预测用户第 10 次观看，却把第 11 次、第 12 次观看也放进历史输入。这会让离线指标虚高。

| 常见坑 | 产生原因 | 影响 | 规避方式 |
|---|---|---|---|
| 负样本分布偏差 | 采样分布不等于真实分布 | 离线指标虚高 | 使用采样修正，做线上 A/B |
| 平均池化丢失顺序 | 所有历史同权 | 短期兴趣被稀释 | 时间衰减、分桶、序列模型 |
| 未来信息泄漏 | 构造样本时用了目标之后的行为 | 评估不可信 | rollback history，只用目标前历史 |
| 训练和推断混用近似 | 把 sampled softmax 当线上检索 | 行为不一致 | 训练、评估、ANN 推断分开 |
| 召回和排序目标混淆 | 用召回层承担精排目标 | 系统复杂且效果不稳 | 两阶段建模，职责分离 |

rollback history 是论文中很重要的样本构造思想。白话说，就是把用户历史回滚到目标观看发生之前，只用当时已经发生的行为预测下一次观看。

真实工程里，首页召回通常不会只依赖一个召回源。YouTube 深度召回网络可以提供强个性化候选，但系统还会混入热门内容、订阅频道更新、地域热点、冷启动内容等召回源。这样做的原因是单一神经召回模型很难同时覆盖新视频、突发热点和长尾兴趣。

---

## 替代方案与适用边界

YouTube 深度召回网络适合内容库很大、用户行为丰富、需要个性化召回的场景。它不是所有推荐系统的默认答案。

如果内容库只有几千条，比如一个内部知识库或小型课程平台，直接用规则检索、关键词检索、向量检索可能更简单。深度召回网络需要训练数据、特征管道、向量索引、在线服务和监控体系，工程成本并不低。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 规则召回 | 业务规则明确、规模小 | 可解释、上线快 | 泛化能力弱 |
| 协同过滤 | 用户-物品交互矩阵较稳定 | 实现简单，效果基线强 | 冷启动困难 |
| 双塔召回 | 用户侧和物品侧可分开编码 | 适合 ANN 检索 | 交叉特征表达弱 |
| YouTube 深度召回 | 大规模内容库和丰富行为日志 | 工业可扩展，个性化强 | 训练和系统复杂 |
| 序列模型 | 短期兴趣、会话意图重要 | 能捕捉行为顺序 | 推理成本和训练复杂度更高 |
| 混合召回 | 大型线上推荐系统 | 覆盖更全面 | 合并、去重、配额策略复杂 |

YouTube 方案本质上也是双塔召回的一种经典形态：用户塔输出用户向量，视频塔或视频 embedding 表输出视频向量，再通过向量相似度检索。双塔模型是指用户侧和物品侧分别编码，最后用点积或余弦相似度匹配。

当业务强依赖行为顺序时，可以考虑序列模型。序列模型是把用户行为按时间顺序建模的模型，例如 RNN、Transformer 或 DIN/DIEN 一类结构。它比平均池化更能表达“刚刚发生的行为”。

当业务目标是新闻、短视频热点、直播等强时效内容时，单纯依赖历史兴趣可能不够。需要额外加入实时召回、热门召回和新内容探索。否则模型会更偏向历史稳定兴趣，错过突发内容。

结论是：当你有百万级以上内容库、足够行为数据、需要低延迟个性化召回，并且后面还有排序层时，YouTube 深度召回网络是一个合理方案。当内容规模小、数据少、排序目标简单，或者主要依赖关键词匹配时，不必强行使用这套架构。

---

## 参考资料

- 论文原文：Deep Neural Networks for YouTube Recommendations  
  https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/  
  用来理解 YouTube 两阶段推荐架构、候选生成模型和工程动机。

- 论文 PDF：Deep Neural Networks for YouTube Recommendations  
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf  
  用来查看候选生成网络、样本构造、负采样和线上服务细节。

- TensorFlow 文档：tf.nn.sampled_softmax_loss  
  https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss  
  用来理解 sampled softmax 在训练大规模分类模型时的接口和约束。

- ScaNN：Google Research 的向量近邻检索工具  
  https://github.com/google-research/google-research/blob/master/scann/README.md  
  用来理解 ANN 检索如何服务于大规模向量召回。

- Faiss：Meta 开源的向量检索库  
  https://github.com/facebookresearch/faiss/wiki  
  用来了解工业里常见的向量索引、近邻搜索和相似度检索实现。

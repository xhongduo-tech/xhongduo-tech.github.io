## 核心结论

负采样是“给模型提供反例”的过程。Easy Negative 指和锚点差得很远、模型一眼就能分开的负样本；Hard Negative 指和锚点很像、容易让模型分不清的负样本。两者的核心差别，不在“难不难看”，而在“是否还能产生有效梯度”。

在嵌入模型、召回模型、对比学习里，随机负采样通常简单、便宜、稳定，但大量样本很快变成“无效题”。以 Triplet Loss 为例：

$$
L=\max(0,\ d(a,p)-d(a,n)+m)
$$

其中 $a$ 是 anchor，指当前查询或样本；$p$ 是 positive，指正确匹配；$n$ 是 negative，指错误匹配；$m$ 是 margin，指希望正负之间至少拉开的安全间隔。只有当

$$
d(a,n)<d(a,p)+m
$$

时，损失才大于 0，模型才会更新。随机采到的大多数 easy negative 往往满足不了这个条件，所以训练效率低。

一个玩具例子最直观。设 $m=0.2$，$d(a,p)=0.4$：

- easy negative: $d(a,n)=0.9$，则 $L=\max(0,0.4-0.9+0.2)=0$
- hard negative: $d(a,n)=0.55$，则 $L=\max(0,0.4-0.55+0.2)=0.05$
- 更难的 hard negative: $d(a,n)=0.35$，则 $L=\max(0,0.4-0.35+0.2)=0.25$

结论很直接：随机负采样保证“有负样本”，Hard Negative Mining 保证“有用的负样本”。工程上通常不选纯 easy，也不直接全量 hardest，而是采用混合策略：warm-up 阶段以 easy 为主，稳定后逐步提高 hard 比例，再结合批内和批外负样本一起训练。

---

## 问题定义与边界

这类问题主要出现在检索、推荐、向量召回、句向量训练、图像匹配。目标不是“找任何负样本”，而是“找能逼近当前判别边界的负样本”。

如果把向量空间想成一张地图，模型训练的目标就是让正确配对更近，错误配对更远。random negative 的问题是，它常常离得太远，模型已经会做，继续喂这些样本没有信息增量。hard negative 的价值在于，它落在模型当前还不稳定的边界附近。

但 hard 不等于越近越好。边界必须讲清楚：

| 类型 | 与 anchor 的距离/相似度特征 | 是否常产生梯度 | 主要风险 | 是否需要额外验证 |
|---|---|---:|---|---|
| Easy Negative | 距离远，相似度低 | 否，常为 0 | 训练效率低 | 一般不需要 |
| Hard Negative | 距离近，相似度高，但确实是负样本 | 是 | 训练不稳定 | 建议需要 |
| False Negative | 看起来很近，实际上应当是正样本 | 会，但方向错误 | 标签噪声、模型学坏 | 必须需要 |

这里的 false negative 是“被误当成负样本的真相近样本”。白话说，就是你以为它是反例，实际上它本来就应该匹配。比如推荐系统里，用户没点过某商品，不代表它就是负样本；用户可能只是没看到。

因此，问题定义不是“怎么挖 hardest sample”，而是“怎么在可控噪声下，持续提供有效梯度”。这决定了几个工程边界：

- 数据噪声高时，hard negative 要更保守。
- 训练早期模型还不稳定，hard 比例不能太高。
- 业务里“未点击=负样本”不总成立，必须结合曝光日志、规则过滤或复核模型。
- 批内负采样适合低成本启动，批外负采样适合补充更强的 hard 信号。

---

## 核心机制与推导

先看 Triplet Loss。它本质上要求：

$$
d(a,p)+m<d(a,n)
$$

也就是正样本比负样本更近，且近得足够多。如果这个不等式已经成立很多，损失就是 0；如果不成立，模型才需要继续学。

所以采样策略的本质，是控制 $d(a,n)$ 的分布，让更多负样本落入“还会触发损失”的区间。把随机采样写成概率问题更容易理解。假设候选负样本距离分布大多集中在 $[0.8,1.2]$，而有效区间只有 $[0.0,\ d(a,p)+m)$。若当前 $d(a,p)+m=0.6$，那么随机抽到有效负样本的概率就很低。Hard Negative Mining 做的事情，就是提升“采到落在有效区间内样本”的概率。

对比学习里的 InfoNCE 也是同样逻辑。公式是：

$$
L=-\log \frac{e^{\operatorname{sim}(a,p)/\tau}}{e^{\operatorname{sim}(a,p)/\tau}+\sum_{n}e^{\operatorname{sim}(a,n)/\tau}}
$$

其中 $\operatorname{sim}$ 是相似度，常用余弦相似度；$\tau$ 是温度系数，可以理解为“softmax 的放大倍率”。$\tau$ 越小，hard negative 的权重越大，因为和正样本更接近的负样本会在分母里占更高比例。于是模型会更强烈地被这些难负样本驱动。

一个玩具例子。设正样本相似度为 0.8，两个负样本相似度分别是 0.1 和 0.75：

- 当 $\tau$ 较大时，两者在分母中的差异被压平，easy negative 仍占一部分权重。
- 当 $\tau$ 较小时，0.75 这个 hard negative 的指数项显著变大，loss 几乎主要由它决定。

这就是为什么“低温度 + hard negative”通常更快，但也更容易不稳定。因为模型被迫把注意力集中在少量最难样本上，一旦这些样本里混入 false negative，就会学偏。

采样实现上，常见来源有四类：

| 采样来源 | 含义 | 难度水平 | 成本 |
|---|---|---|---|
| In-batch | 同一个 batch 里其他样本充当负例 | 中等 | 最低 |
| Memory bank / Queue | 历史 batch 的向量池 | 中高 | 较低 |
| ANN / KNN 检索 | 从全局向量库找最近邻负样本 | 高 | 中等 |
| Graph-based / ProSampler | 用近邻图、共现图找结构上接近的负样本 | 高 | 较高 |

in-batch negative 是“同批其他样本当反例”。白话说，不额外找数据，直接把别人的正样本当我的负样本。它实现简单，但 hard 程度受 batch 大小和数据分布限制。批外 hard negative 则能更主动地把“真正接近的错误候选”找出来。

---

## 代码实现

下面给一个可运行的简化 Python 示例，演示三件事：

1. 计算 triplet loss  
2. 按 epoch 动态提高 hard 比例  
3. 用简单规则过滤可能的 false negative

```python
import math
import random

def triplet_loss(d_ap, d_an, margin=0.2):
    return max(0.0, d_ap - d_an + margin)

def mix_ratio(epoch, start=0.3, step=0.05, cap=0.8):
    return min(cap, start + epoch * step)

def sample_negatives(anchor_id, easy_pool, hard_pool, epoch, total_k=6):
    hard_ratio = mix_ratio(epoch)
    hard_k = int(total_k * hard_ratio)
    easy_k = total_k - hard_k

    easy_samples = random.sample(easy_pool, min(easy_k, len(easy_pool)))
    hard_samples = random.sample(hard_pool, min(hard_k, len(hard_pool)))
    return easy_samples + hard_samples, hard_ratio

def is_possible_false_negative(anchor_text, neg_text):
    # 简化规则：文本完全相同或主实体相同，认为风险高
    return anchor_text.strip().lower() == neg_text.strip().lower()

# 玩具数据：distance 越小越难
easy_pool = [
    {"id": "n1", "text": "apple phone case", "distance": 0.92},
    {"id": "n2", "text": "running shoes men", "distance": 0.88},
    {"id": "n3", "text": "wireless keyboard", "distance": 0.95},
]

hard_pool = [
    {"id": "n4", "text": "iphone 15 pro max case", "distance": 0.55},
    {"id": "n5", "text": "iphone magsafe case", "distance": 0.48},
    {"id": "n6", "text": "iphone case", "distance": 0.35},  # 可能是 false negative
]

anchor = {"id": "a1", "text": "iphone case"}
positive = {"id": "p1", "distance": 0.40}

loss_easy = triplet_loss(positive["distance"], 0.92, margin=0.2)
loss_hard = triplet_loss(positive["distance"], 0.55, margin=0.2)

assert loss_easy == 0.0
assert abs(loss_hard - 0.05) < 1e-9

selected, ratio = sample_negatives(anchor["id"], easy_pool, hard_pool, epoch=6, total_k=4)
filtered = [x for x in selected if not is_possible_false_negative(anchor["text"], x["text"])]

# 至少还能保留部分样本
assert len(filtered) >= 1
assert 0.3 <= ratio <= 0.8

batch_loss = sum(triplet_loss(positive["distance"], x["distance"], margin=0.2) for x in filtered)
assert batch_loss >= 0.0

print("hard_ratio=", ratio)
print("selected_ids=", [x["id"] for x in filtered])
print("batch_loss=", round(batch_loss, 4))
```

上面代码故意保持简单，但结构已经接近真实工程：

- `easy_pool` 可以来自 in-batch negatives
- `hard_pool` 可以来自 ANN 检索、KNN 图、曝光日志
- `mix_ratio` 控制 hard/easy 比例随 epoch 上升
- `is_possible_false_negative` 对应真实系统里的 cross-encoder 复核、规则过滤或标签校验

真实工程例子可以看推荐召回。比如搜索“iPhone 手机壳”，正样本是用户点击或购买的商品。负样本的构造方式可以是：

- easy：同批次里其他完全无关商品，如跑鞋、键盘
- hard：文本相近但未点击的商品，如“iPhone MagSafe 壳”
- false negative 风险样本：用户没点但其实高度相关、只是排序位置太靠后或未曝光充分的商品

一种常见训练流水线如下：

1. 先用 in-batch negative 训练基础模型，快速获得可用向量。
2. 用基础模型离线召回每个 query 的近邻候选，构建 hard pool。
3. 训练时按比例混合 easy 与 hard，例如 `20% hard -> 50% hard -> 80% hard`。
4. 对 hard pool 中最高相似度样本做交叉编码器复核，剔除 false negative。
5. 监控 recall、loss 分布、embedding 范数和正负相似度间隔。

如果是双塔检索模型，这套逻辑尤其常见，因为双塔上线后本身就能提供“当前模型最容易混淆的候选”，天然适合反哺 hard negative 挖掘。

---

## 工程权衡与常见坑

hard negative 的价值很高，但副作用也集中。实践里最常见的不是“没效果”，而是“短期看起来效果很好，随后训练开始发散或线上退化”。

| 风险 | 原因 | 典型表现 | 规避方式 |
|---|---|---|---|
| False Negative | 把潜在正样本当成负样本 | recall 下降，长尾查询退化 | cross-encoder 复核、规则过滤、人工抽检 |
| Loss 崩溃 | hardest 样本过多，梯度过激 | loss 抖动大，训练不稳 | warm-up、逐步提 hard 比例、梯度裁剪 |
| 维度坍缩 | 模型被少数难例牵引过度 | 向量分布集中，相似度失真 | 加强 easy mix、正则化、监控 embedding norm |
| 训练变慢 | 批外采样成本高 | 离线构图和检索耗时 | 先 in-batch，后离线增量更新 hard pool |
| 线上不一致 | 训练负样本和线上候选分布不同 | 训练指标好，线上收益弱 | 用曝光日志或真实召回候选构建 hard pool |

“维度坍缩”是指不同样本的向量越来越像，模型虽然还能输出数值，但区分能力消失。对初学者来说，可以把它理解成：模型为了满足少量极难样本，把整个向量空间压扁了。

一个常被引用的真实工程思路，是从日志里挖 hard negative，而不是只从随机样本里挖。比如 Pinterest 一类的推荐场景，会利用视觉相似、跳过行为、近似查询等信号构造更难的负样本。好处是这些负样本更接近真实线上竞争关系，不再是“明显无关”的 easy case。公开资料中，这类方法可以把召回指标显著抬高，同时减少下游排序阶段需要处理的候选量，带来延迟收益。这里的核心不是某个具体数字，而是方法论：hard negative 最有价值的来源，往往是“真实系统差一点就会选中的错误候选”。

一个实用原则是：不要直接追 hardest，优先追 useful hardest。也就是先保证负样本真的错，再追求它足够难。

---

## 替代方案与适用边界

不是所有场景都要重度 hard negative mining。下面是几种常见替代方案。

| 策略 | 做法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| Hard Negative Mining | 主动找最接近的负样本 | 收敛快，边界清晰 | 噪声风险高，系统复杂 | 大规模检索、推荐召回 |
| Semi-hard Negative | 选比正样本远，但仍违反 margin 的负样本 | 梯度稳定，噪声更低 | 难度不如 hardest | 标签不完美的数据 |
| In-batch Only | 只用批内其他样本作负例 | 实现简单，吞吐高 | hard 程度有限 | 训练初期、资源有限 |
| Soft Weighted Negative | 不硬筛 hardest，而按相似度加权 | 更平滑，抗噪声 | 收敛可能更慢 | false negative 成本高 |

semi-hard negative 很重要。它指“比 positive 远，但还不够远”的负样本。以 Triplet Loss 看，它满足：

$$
d(a,p) < d(a,n) < d(a,p)+m
$$

它既不是最危险的错误样本，也不是完全无效的 easy 样本。白话说，就是“有点难，但没难到可能是错标”。很多团队会优先用 semi-hard，而不是 hardest-first。

如果数据规模小、标注噪声高、误伤成本高，soft weighted 往往比强硬 hard mining 更稳。做法是不给负样本简单打“选中/不选中”，而是按相似度赋权。相似度高的负样本权重大，相似度低的权重小。这样模型仍会关注难例，但不会被个别可疑样本强行拖偏。

在高吞吐系统里，in-batch + 大 batch + momentum queue 也可能已经够用。原因很简单：只要 batch 足够大，批内就自然会出现一些高相似负样本；再加一个历史队列，难样本覆盖面会进一步扩大。这种方案不一定最强，但在延迟、存储、实现复杂度上更平衡。

因此，选择策略的依据不是“哪种最先进”，而是三件事：

- 你的负样本标签是否可靠
- 你的训练系统能否承受批外采样成本
- 你的线上候选竞争是否真的需要更细的判别边界

---

## 参考资料

| 来源 | 核心贡献 | 适用章节 |
|---|---|---|
| [SystemOverflow: What is Hard Negative Mining?](https://www.systemoverflow.com/learn/ml-embeddings/hard-negative-mining/what-is-hard-negative-mining?utm_source=openai) | 给出 hard negative 的定义、混合 easy/hard 的直观训练思路，以及推荐系统案例 | 核心结论、问题定义、工程权衡 |
| [SystemOverflow: Triplet Loss and Contrastive Loss Formulations](https://www.systemoverflow.com/learn/ml-embeddings/hard-negative-mining/triplet-loss-and-contrastive-loss-formulations?utm_source=openai) | 说明 Triplet Loss 与 InfoNCE 的公式、梯度触发条件和数值例子 | 核心机制与推导 |
| [Sentence Transformers Documentation](https://www.sbert.net/examples/sentence_transformer/training/README.html) | 展示 in-batch negative 在句向量训练中的常见实现方式 | 代码实现、替代方案 |
| [ProSampler / Graph-based Sampling Paper](https://arxiv.org/abs/2307.07240) | 提供基于近邻图或结构信息的批外 hard negative 采样思路 | 代码实现、工程权衡 |
| [Hugging Face: Mitigating False Negatives in Retriever Training](https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training?utm_source=openai) | 讨论 false negative 风险，以及用 cross-encoder 复核和过滤的做法 | 工程权衡与常见坑 |
| [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) | 经典 triplet loss 与 semi-hard negative 训练策略来源 | 核心机制、替代方案 |

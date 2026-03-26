## 核心结论

DSSM 可以理解为一种“双塔召回”架构：用户塔负责把用户信息编码成向量，物品塔负责把物品信息编码成向量，二者落到同一个语义空间。语义空间可以直白理解为“相似内容会靠近、不相似内容会远离”的向量坐标系。召回时不再对每个 user-item 组合逐条跑复杂模型，而是先算用户向量，再去向量索引里找最相近的物品向量。

它的核心打分公式很简单：

$$
score(u,i)=f_u(u)^\top f_i(i)
$$

其中 $f_u$ 是用户塔，$f_i$ 是物品塔，$\top$ 表示转置，最终就是向量内积。内积可以直白理解为“两个方向有多一致”。如果先把向量归一化到单位长度，内积就等价于余弦相似度。

这套架构的工程价值不在“公式新”，而在“计算拆分”。物品塔可以离线批量算好全部 item embedding 并建 ANN 索引，ANN 是近似最近邻搜索，用较小误差换大幅提速；用户塔只需在请求到来时在线算一次。于是海量召回从“遍历上亿物品逐条打分”变成“生成一个用户向量，再做一次近邻搜索”。

一个最直观的玩具例子是：用户塔实时读取“最近 50 次点击 + 当前时段 + 设备类型”，item 塔离线读取“标题 + 类目 + 价格区间”，分别输出 128 维向量，最终在向量库里按点积找 Top-K。新手只要抓住一句话就够了：先独立编码，再快速比相似度。

---

## 问题定义与边界

推荐系统通常不是一步完成，而是多阶段流水线。召回阶段的目标不是最终排序，而是从百万到亿级候选里先找出几百到几千个“值得进一步判断”的候选。这个阶段本质上是检索问题，不是精排问题。

双塔适合这个阶段，因为它把计算拆成两侧：

| 维度 | 用户塔输入 | 物品塔输入 |
|---|---|---|
| 主要信息 | 行为序列、画像、上下文 | 标题、类目、价格、内容特征 |
| 数据更新频率 | 高，常常实时变化 | 相对低，通常分钟级到天级 |
| 计算方式 | 在线计算 | 离线预计算 |
| 目标 | 生成用户查询向量 | 生成候选物品向量 |

这里有一个重要边界：双塔每个塔都只看自己这一侧的输入，直到最后一步才用内积相遇。这意味着它擅长“快速筛候选”，但不擅长细粒度交叉特征建模。交叉特征可以直白理解为“这个用户的这个具体特征，和这个物品的那个具体特征之间的直接组合关系”。

例如，用户“价格敏感”且商品“高价但限时折扣”，这种细粒度交互，双塔只能通过各自向量间接表达，不能像 cross-encoder 那样逐对展开。因此双塔通常用于召回层，而不是最终排序层。排序层往往会接一个更重的模型，对少量候选做逐条精打分。

真实工程里经常是这样的流水线：

1. 双塔从一亿商品里召回 500 个。
2. 粗排模型把 500 个缩到 100 个。
3. 精排模型对这 100 个做复杂交叉打分。
4. 重排模块再考虑多样性、去重、业务约束。

所以，双塔的边界不是“精度最高”，而是“在极大规模下，以可接受精度换取可接受延迟”。

---

## 核心机制与推导

双塔的相似度函数通常写成：

$$
s(u,i)=u^\top i=\sum_{k=1}^{d}u_k i_k
$$

其中 $u$ 是用户向量，$i$ 是物品向量，$d$ 是向量维度。维度可以直白理解为“模型用多少个数字来表示一个对象”。这个公式的好处是可以批量矩阵乘法，也可以直接接 ANN 检索。

训练时不能只告诉模型“哪些是正样本”，还要告诉它“哪些不该靠近”。因此常用对比学习目标。对比学习可以直白理解为“让正确配对更近，让错误配对更远”。一个常见形式是：

$$
L=-\log \frac{\exp(s(u,i^+)/\tau)}{\exp(s(u,i^+)/\tau)+\sum_{i^-}\exp(s(u,i^-)/\tau)}
$$

符号含义如下：

| 符号 | 含义 |
|---|---|
| $u$ | 用户向量 |
| $i^+$ | 正样本物品向量 |
| $i^-$ | 负样本物品向量 |
| $s(\cdot,\cdot)$ | 相似度函数，通常是内积 |
| $\tau$ | 温度参数，控制分数拉开的敏感度 |

玩具例子可以直接算。设：

- 用户向量 $u=[0.1,0.3,0.4]$
- 正样本 $i^+=[0.2,0.5,0.1]$
- 负样本 $i^-=[0.7,-0.1,0.2]$

则：

$$
s(u,i^+)=0.1\times0.2+0.3\times0.5+0.4\times0.1=0.21
$$

$$
s(u,i^-)=0.1\times0.7+0.3\times(-0.1)+0.4\times0.2=0.12
$$

正样本得分比负样本高，loss 就会更小；如果反过来，梯度就会推动用户向量更靠近正样本、更远离负样本。这就是双塔训练的核心驱动力。

再往前推一步，为什么它能快？因为排序分解了。假设有 $N$ 个物品，如果在线逐个打分，复杂度接近 $O(N)$。而双塔把 item 向量提前算好后，线上只做：

1. 计算一次 $u=f_u(x)$
2. 在 ANN 索引中查 Top-K

这样真正在线重算的只剩用户侧。

真实工程例子：一个 App 商店做推荐。用户塔输入“近 7 天安装序列、最近搜索词、设备型号、网络状态、国家地区”；物品塔输入“App 标题、描述、类别、包体大小、价格、历史转化统计”。线上请求到来时，系统实时算出用户 embedding，然后在预构建的 HNSW 或 Faiss 索引中搜索最接近的 300 个 App，交给后续排序。这里双塔不是整个推荐系统，但它决定了后面是否还有好候选可排。

---

## 代码实现

下面用一个最小可运行的 Python 例子说明内积打分、softmax 概率和 Top-K 召回。它不是训练框架代码，但能把双塔最核心的计算路径跑通。

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(xs):
    exps = [math.exp(x) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

# 玩具用户向量与候选物品向量
user_vec = [0.1, 0.3, 0.4]
items = {
    "positive_item": [0.2, 0.5, 0.1],
    "negative_item": [0.7, -0.1, 0.2],
    "other_item": [0.0, 0.1, 0.8],
}

scores = {name: dot(user_vec, vec) for name, vec in items.items()}

# 计算分数
assert round(scores["positive_item"], 2) == 0.21
assert round(scores["negative_item"], 2) == 0.12

# 按分数排序，模拟召回 Top-K
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top2 = [name for name, _ in ranked[:2]]

assert top2[0] == "other_item"
assert "positive_item" in top2

# 模拟对比学习中的 softmax 概率
logits = [scores["positive_item"], scores["negative_item"]]
probs = softmax(logits)

assert len(probs) == 2
assert probs[0] > probs[1]
assert abs(sum(probs) - 1.0) < 1e-9

print("scores:", scores)
print("top2:", top2)
print("softmax probs:", probs)
```

如果写成训练与推理流程，结构通常如下：

```python
# 训练阶段
for batch in train_loader:
    user_embed = user_tower(batch.user_features)
    pos_embed = item_tower(batch.pos_item_features)
    neg_embed = item_tower(batch.neg_item_features)

    pos_score = (user_embed * pos_embed).sum(axis=1)
    neg_score = user_embed @ neg_embed.T

    loss = contrastive_loss(pos_score, neg_score)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 离线阶段
for item in all_items:
    item_embed = item_tower(item.features)
    ann_index.add(item.id, item_embed)

# 在线召回阶段
user_embed = user_tower(current_user_features)
candidate_ids = ann_index.search(user_embed, k=100)
```

实现时常见拆分是：

| 模块 | 职责 |
|---|---|
| `user_tower` | 编码用户行为、上下文、画像 |
| `item_tower` | 编码物品内容、类目、统计特征 |
| `sampler` | 产出正负样本，决定训练分布 |
| `contrastive_loss` | 拉近正样本、拉远负样本 |
| `ann_index` | 存储 item embedding 并支持近邻搜索 |

新手最容易忽略的一点是：双塔的“模型”不只是一段神经网络代码，离线 embedding 生成、索引构建、在线特征服务、ANN 检索都属于系统的一部分。只把网络写出来，不等于把召回系统建出来。

---

## 工程权衡与常见坑

双塔上线后，常见问题通常不在“模型会不会收敛”，而在“训练分布和线上分布是否一致”。

| 常见坑 | 具体表现 | 常见规避方式 |
|---|---|---|
| 负采样偏差 | 模型过度学热门 item，长尾召回差 | 混合负采样、硬负采样、logQ 校正 |
| embedding 过期 | 商品已改价或下架，向量仍是旧的 | 增量刷新、TTL 过期、事件触发重算 |
| 训练/Serving 分布失配 | 离线指标好，线上点击差 | 对齐特征口径、统一归一化、回放验证 |
| 冷启动 | 新用户或新物品缺少行为数据 | 用内容特征、热门兜底、探索策略 |
| 向量坍塌 | embedding 区分度低，大家都挤在一起 | 调维度、温度参数、正则、采样优化 |

负采样最关键。负采样就是“给模型看哪些反例”。如果只用 batch 内负样本，很多负样本来自曝光偏置；如果只用流行度采样，又容易让模型记住热门分布。Google 在 WWW 2020 提出的 Mixed Negative Sampling，本质是把 batch 内负样本和均匀采样负样本混起来，缓解隐式反馈中的选择偏差。选择偏差可以直白理解为“你看到的数据，不等于真实偏好分布”。

真实工程例子：App 推荐里，正样本来自用户点击后安装的日志，负样本一部分来自同 batch 其他曝光 App，一部分来自全库均匀抽样未安装 App。这样训练出的模型通常比单一负采样更稳，因为它同时见到了“容易混淆的近邻负样本”和“覆盖更广的全局负样本”。

另一个大坑是 item embedding 过期。双塔依赖“item 可离线预计算”，但这不等于“可以很久不更新”。商品标题、库存、价格、内容审核状态都可能变化。如果索引里还是旧向量，召回就会把错误候选带给排序层。工程上通常需要：

1. 定期全量重建索引。
2. 对高频变更 item 做增量更新。
3. 对下架或风控命中 item 做即时删除。
4. 给 embedding 设置版本号和过期策略。

最后，别把离线 Recall@K 看成全部。双塔在线效果还受 ANN 召回误差、特征延迟、候选去重、业务规则过滤影响。很多项目离线好看、线上一般，问题不在塔，而在系统链路没对齐。

---

## 替代方案与适用边界

双塔不是唯一方案。它的优势是快，代价是交互表达能力有限。可以用一张表看清边界：

| 方案 | 计算方式 | 精度潜力 | 延迟 | 典型用途 |
|---|---|---|---|---|
| Dual-Tower / DSSM | 先独立编码，再算相似度 | 中 | 低 | 大规模召回 |
| Cross-Encoder | user-item 拼接后逐对打分 | 高 | 高 | 精排、重排 |
| Siamese | 两侧结构对称，常共享参数 | 中 | 低 | 文本检索、问答匹配 |

Cross-Encoder 适合需要强交叉特征的任务。例如“这个用户对这个具体商品标题中的某个词是否敏感”，它可以直接在同一网络里建模，但它必须对每个候选逐条计算，无法承担海量候选召回。双塔则相反，先牺牲一部分交互能力，换取大规模检索速度。

Siamese 可以看作双塔的一个特例。它通常指两侧结构相同、参数共享。参数共享可以直白理解为“左右两边用同一套编码器权重”。如果输入两侧模态相近，比如问答检索中的“问题”和“候选问题”，共享参数往往合理；如果两侧模态差异很大，比如“用户行为序列”和“商品内容特征”，通常更适合非对称双塔，各自编码。

因此适用边界可以概括为：

- 候选量巨大、延迟严格、允许后续再精排：优先双塔。
- 候选量小、需要精细交互、可接受较高延迟：考虑 cross-encoder。
- 两侧输入语义和结构高度一致：可以考虑 Siamese 共享参数。
- 新物品很多且内容特征充分：双塔对冷启动更友好，因为 item 塔可直接利用内容侧特征。
- 强规则约束、多目标排序复杂：双塔只负责第一阶段，不能替代后续排序与重排。

---

## 参考资料

- SystemOverflow, *How Two Tower Architecture Works*  
  https://www.systemoverflow.com/learn/ml-recommendation-systems/two-tower-models/how-two-tower-architecture-works

- Shaped Docs, *Two-Tower (Neural Retrieval)*  
  https://docs.shaped.ai/docs/model_library/two_tower/

- Emergent Mind, *Two-Tower Retrieval Architecture*  
  https://www.emergentmind.com/topics/two-tower-retrieval-architecture

- Google Research, *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations*  
  https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/

- Microsoft Research, *DSSM*  
  https://www.microsoft.com/en-us/research/project/dssm/

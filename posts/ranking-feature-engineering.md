## 核心结论

排序模型的特征工程，就是把用户、物品、上下文和交叉信号，转换成稳定、可学习、可在线复用的数值输入。

排序模型的目标是对候选物品打分，例如预测点击率、购买率、停留时长或综合收益。模型本身只接收数字向量，不会直接理解“27 岁用户”“3 号商品”“晚上 8 点”“最近点击过手机壳”这些原始业务字段。特征工程的作用，是把这些字段处理成模型能使用的形式。

典型输入可以写成：

$$
\mathbf{z}=[\mathbf{x}'_u,\mathbf{x}'_i,\mathbf{x}'_c,\mathbf{e}_u,\mathbf{e}_i,\mathbf{e}_{u\times i}]
$$

其中 $\mathbf{x}'_u$ 表示处理后的用户数值特征，$\mathbf{x}'_i$ 表示处理后的物品数值特征，$\mathbf{x}'_c$ 表示处理后的上下文特征，$\mathbf{e}_u$、$\mathbf{e}_i$、$\mathbf{e}_{u\times i}$ 表示用户、物品和交叉特征的 Embedding 向量。Embedding 是一种可训练的查表表示，用稠密向量表示离散 ID。

| 原始字段 | 特征处理 | 模型输入 |
|---|---|---|
| 年龄 27 | 标准化：$x'=(x-\mu)/\sigma$ | `-0.3` |
| 商品价格 37 | 分箱 | `price_bin=3` |
| 类目 ID 12 | Embedding：$\mathbf{e}=E[id]$ | `[0.2, -0.5]` |
| 用户 ID × 类目 ID | 哈希交叉 | `cross_bucket=814` |
| 最近 20 次点击 | 序列池化 | `seq_embedding` |
| 小时、城市、网络 | one-hot、Embedding 或分箱 | `context_features` |

核心目标不是“堆更多特征”，而是让特征满足四个条件：稳定、可复用、无泄漏、能泛化。稳定表示统计口径不会频繁漂移；可复用表示离线训练和在线推理使用同一套逻辑；无泄漏表示训练特征只使用样本发生时刻以前的信息；能泛化表示模型不只记住历史样本，还能处理新用户、新物品和新场景。

---

## 问题定义与边界

排序模型通常位于推荐系统的后半段。召回阶段先从海量物品中选出几百到几千个候选，排序阶段再对候选逐个打分。特征工程就是为这个打分函数准备输入。

特征可以分为五类：

| 特征类型 | 典型字段 | 可用时机 | 常见处理方式 |
|---|---|---|---|
| 用户特征 | 年龄、性别、会员等级、7 天点击数 | 请求时刻已知 | 标准化、分箱、Embedding |
| 物品特征 | 类目、价格、品牌、历史 CTR | 请求时刻已知 | 分箱、Embedding、统计平滑 |
| 上下文特征 | 小时、星期、城市、设备、网络类型 | 请求时刻已知 | one-hot、分箱、Embedding |
| 交叉特征 | 用户 × 类目、城市 × 类目 | 请求时刻由已有字段计算 | 显式交叉、哈希交叉、交叉 Embedding |
| 序列特征 | 最近点击、最近购买、最近搜索词 | 请求时刻以前的行为 | 池化、RNN、Transformer |

这里的边界很重要：只讨论请求时刻可见的特征。请求时刻是系统准备给用户排序候选商品的那一刻。任何发生在这个时刻之后的信息，都不能作为训练特征。

电商排序里的可用特征包括：用户最近 7 天点击数、最近 20 次点击序列、常买类目；物品类目、价格、历史点击率；上下文小时、星期、城市、网络类型；交叉特征如 `user_id × item_category`、`city × category`。但“未来 7 天点击数”“曝光后是否收藏”“次日是否购买”不能作为输入特征。它们最多可以作为标签或评估目标，不能作为模型预测时的输入。

| 特征 | 可在线复用 | 原因 |
|---|---:|---|
| 用户过去 7 天点击数 | 是 | 请求时刻以前已经发生 |
| 商品过去 1 天 CTR | 是 | 如果统计截止到请求前 |
| 用户未来 7 天点击数 | 否 | 使用了未来信息 |
| 商品全量历史 CTR | 不一定 | 如果包含样本之后的数据，会泄漏 |
| 当前小时 | 是 | 请求时刻直接可见 |
| 曝光后是否点击 | 否 | 这是标签，不是输入 |

排序特征工程有三个关键约束。

第一，训练和线上一致性。训练时怎么标准化、怎么分箱、怎么哈希，线上必须完全一致。否则模型在训练和推理时看到的是两个分布。

第二，时点一致性。样本发生在 2026-04-01 10:00，就只能使用这个时间点之前的统计。不能用 2026-04-10 汇总出来的全量统计去回填历史样本。

第三，可解释性要求。排序模型可以很复杂，但基础特征必须能追溯。线上 CTR 下降时，工程师需要知道是用户特征缺失、物品统计延迟、上下文异常，还是交叉特征冲突变多。

---

## 核心机制与推导

连续特征是可以直接用数值表示的字段，例如年龄、价格、点击次数、停留时长。连续特征通常需要标准化：

$$
x'=\frac{x-\mu}{\sigma}
$$

其中 $\mu$ 是训练集均值，$\sigma$ 是训练集标准差。标准化的作用是降低尺度差异。比如年龄大多在 0 到 100，价格可能从 1 到 100000，点击次数可能长尾分布很重。如果直接输入，模型训练会更容易被大尺度字段影响。

分箱是把连续值映射到离散区间：

$$
b=\mathrm{bucket}(x;\beta_1,\dots,\beta_m)
$$

例如价格 37，分箱边界是 `[10, 20, 50]`，则 37 落在 `(20, 50]`，可以记为第 3 个区间。分箱适合非线性明显的字段。价格从 10 到 20 的变化，和从 1000 到 1010 的变化，对点击率的影响通常不同。

类别特征是取值来自有限集合的字段，例如用户 ID、商品 ID、城市、类目、品牌。低基数类别可以 one-hot，高基数类别通常用 Embedding：

$$
\mathbf{e}=E[id]
$$

其中 $E\in \mathbb{R}^{|V|\times d}$ 是 Embedding 表，$|V|$ 是 ID 总数，$d$ 是向量维度。Embedding 的好处是能把离散 ID 转成可训练的稠密表示。

交叉特征用于表达“两个字段组合后才有意义”的关系。比如用户喜欢数码类商品，不代表他喜欢所有商品；城市用户对某些类目的偏好也可能不同。哈希交叉可以写成：

$$
k=\mathrm{hash}(a,b)\bmod B
$$

其中 $B$ 是桶数量。哈希交叉不需要维护完整字典，适合高基数组合，但会有桶冲突。

序列特征表示用户过去行为的顺序，例如最近点击过的商品列表。序列不能简单当成无序集合，否则“刚刚点击”和“三周前点击”会被混在一起。最简单的处理是池化：

$$
\mathbf{e}_{seq}=\mathrm{Pool}(E[v_1],\dots,E[v_T])
$$

Pool 是池化，白话说就是把多个向量压成一个向量，例如取平均、加权平均或最大值。复杂一些可以用 GRU、Transformer 等序列模型。

| 特征类型 | 处理方法 | 优点 | 风险 |
|---|---|---|---|
| 连续特征 | 标准化、归一化 | 训练更稳定 | 统计量口径错误会造成偏移 |
| 连续特征 | 分箱 | 捕捉非线性 | 边界设计粗糙会损失信息 |
| 类别特征 | one-hot | 简单、可解释 | 高基数时维度过大 |
| 类别特征 | Embedding | 表达能力强 | 冷启动和维度选择困难 |
| 交叉特征 | 哈希交叉 | 节省字典空间 | 哈希冲突不可避免 |
| 序列特征 | 池化或序列建模 | 使用行为历史 | mask、长度截断容易出错 |

一个玩具例子可以完整说明这条链路。用户年龄 27，训练均值 30，标准差 10，则：

$$
x'=\frac{27-30}{10}=-0.3
$$

商品价格 37，分箱边界 `[10, 20, 50]`，落入第 3 个区间。类目 ID 为 12，Embedding 查表得到 `[0.2, -0.5]`。用户 ID 和类目 ID 做哈希交叉，映射到 1024 个桶中的第 814 个桶。最终输入可以表示为：

```text
[-0.3, price_bin=3, 0.2, -0.5, cross_bucket=814]
```

真实工程例子是电商首页排序。用户侧使用最近 7 天点击数、最近 20 次点击序列、常买类目；物品侧使用标题类目、品牌、价格、历史 CTR；上下文侧使用小时、星期、城市、网络类型；交叉侧使用 `user_id × item_category`、`city × category`、`recent_click_category × item_category`。这些特征拼接后输入排序模型，模型输出每个候选商品的点击概率或综合得分。

---

## 代码实现

下面的代码展示一个可运行的最小特征层。它覆盖数值标准化、类别 Embedding、哈希交叉、序列池化，并用同一个 `FeatureProcessor` 同时服务训练和推理。

```python
import hashlib
import numpy as np


class FeatureProcessor:
    def __init__(self, stats, embeddings, bucket_boundaries, hash_buckets=1024):
        self.stats = stats
        self.embeddings = embeddings
        self.bucket_boundaries = bucket_boundaries
        self.hash_buckets = hash_buckets

    def normalize(self, name, value):
        mu = self.stats[name]["mean"]
        sigma = self.stats[name]["std"]
        return (value - mu) / sigma

    def bucketize(self, name, value):
        boundaries = self.bucket_boundaries[name]
        for idx, boundary in enumerate(boundaries, start=1):
            if value <= boundary:
                return idx
        return len(boundaries) + 1

    def embedding_lookup(self, table_name, item_id):
        table = self.embeddings[table_name]
        return np.array(table.get(item_id, table["<UNK>"]), dtype=float)

    def hashed_cross(self, *values):
        raw = "|".join(map(str, values)).encode("utf-8")
        digest = hashlib.md5(raw).hexdigest()
        return int(digest, 16) % self.hash_buckets

    def pooled_sequence_embedding(self, table_name, ids, max_len=20):
        ids = ids[-max_len:]
        if not ids:
            return self.embedding_lookup(table_name, "<UNK>")
        vectors = [self.embedding_lookup(table_name, x) for x in ids]
        return np.mean(vectors, axis=0)

    def transform(self, sample):
        age_norm = self.normalize("age", sample["age"])
        price_bin = self.bucketize("price", sample["price"])

        category_emb = self.embedding_lookup("category", sample["category_id"])
        user_emb = self.embedding_lookup("user", sample["user_id"])

        cross_bucket = self.hashed_cross(sample["user_id"], sample["category_id"])
        cross_scaled = cross_bucket / (self.hash_buckets - 1)

        seq_emb = self.pooled_sequence_embedding(
            "category",
            sample["recent_click_categories"],
        )

        return np.concatenate([
            np.array([age_norm, price_bin, cross_scaled], dtype=float),
            user_emb,
            category_emb,
            seq_emb,
        ])


stats = {
    "age": {"mean": 30.0, "std": 10.0},
}

bucket_boundaries = {
    "price": [10, 20, 50],
}

embeddings = {
    "user": {
        "u1": [0.1, 0.3],
        "<UNK>": [0.0, 0.0],
    },
    "category": {
        12: [0.2, -0.5],
        8: [0.4, 0.1],
        "<UNK>": [0.0, 0.0],
    },
}

processor = FeatureProcessor(stats, embeddings, bucket_boundaries)

sample = {
    "user_id": "u1",
    "age": 27,
    "price": 37,
    "category_id": 12,
    "recent_click_categories": [8, 12],
}

features = processor.transform(sample)

assert round(processor.normalize("age", 27), 2) == -0.30
assert processor.bucketize("price", 37) == 3
assert processor.embedding_lookup("category", 12).tolist() == [0.2, -0.5]
assert len(features) == 9
assert np.isfinite(features).all()
```

这段代码里，`stats` 和 `bucket_boundaries` 代表离线训练阶段产出的统计量。线上推理不能重新计算一套均值、方差或分箱边界，而应该加载同一份配置。否则训练时年龄 27 被映射成 `-0.3`，线上可能被映射成另一个值，模型输入分布就会变化。

一个常见工程结构是：

```text
offline_data -> fit_feature_stats -> save_feature_config
training_sample -> FeatureProcessor(config).transform -> train_model
online_request -> FeatureProcessor(config).transform -> model_predict
```

关键点是 `FeatureProcessor(config)` 必须共用。训练和推理可以运行在不同系统里，但特征逻辑、统计量版本、哈希方式、默认值策略必须一致。

---

## 工程权衡与常见坑

特征泄漏是排序特征工程里最高优先级的风险。泄漏是指训练时使用了预测时不可见的信息。比如把“未来 7 天点击数”作为训练输入，模型离线 AUC 可能明显变高，但线上 CTR 下降。原因很直接：线上没有这个字段，或者线上只能拿到过去点击数，模型训练出的规律无法复用。

统计口径不一致也很常见。CTR 是点击率，定义是点击次数除以曝光次数。商品历史 CTR 如果用全量数据计算，就可能包含样本发生之后的点击。正确做法是按样本时间截断，或至少按训练窗口固定统计版本。

| 常见坑 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 特征泄漏 | 离线 AUC 高，线上 CTR 低 | 使用未来信息 | 所有统计按样本时点截断 |
| 统计口径不一致 | 训练稳定，线上分布漂移 | 离线和线上计算逻辑不同 | 特征层共用，配置版本化 |
| 哈希桶冲突 | 部分组合效果异常 | 多个交叉映射到同一桶 | 增大桶数，监控冲突率 |
| 序列长度和 mask 错误 | 空序列用户表现差 | padding 被当成真实行为 | 明确 mask 和默认向量 |
| Embedding 维度不当 | 欠拟合或内存过大 | 维度和基数、样本量不匹配 | 按基数、频次和实验调参 |

高基数交叉要谨慎。`user_id × item_id` 这种交叉可能极度稀疏，模型容易记住历史曝光，而不是学习可泛化规律。`user_id × category`、`city × category` 通常更稳，因为它们粒度更粗，重复样本更多。

Embedding 维度也不是越大越好。维度过小，复杂偏好表达不出来；维度过大，参数量增加，冷门 ID 学不充分，还会拖慢训练和推理。常见做法是让高频、高价值、高基数字段使用更高维度，让低频或低基数字段使用较低维度。

线上 CTR 下降时，可以按下面路径排查：

| 排查项 | 重点问题 |
|---|---|
| 特征缺失率 | 是否某些字段线上为空或默认值暴涨 |
| 分布对比 | 线上均值、分位数是否偏离训练集 |
| 统计版本 | CTR、热度、均值方差是否加载错版本 |
| 序列处理 | 空序列、截断、mask 是否符合预期 |
| 交叉特征 | 哈希桶冲突率是否异常 |
| 模型输入 | 拼接顺序是否和训练时一致 |

排序系统里，特征质量问题经常比模型结构问题更难发现。模型报错通常很直接，特征错了往往还能正常输出，只是输出变差。

---

## 替代方案与适用边界

不是所有特征都必须 Embedding，也不是所有字段都值得交叉。特征方案要按数据规模、延迟、可解释性和稳定性选择。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| One-hot | 低基数类别，如星期、性别、网络类型 | 简单、可解释 | 高基数时维度爆炸 |
| Embedding | 高基数类别，如用户、商品、类目 | 表达能力强 | 需要足够样本训练 |
| 显式交叉 | 组合数量可控，如城市 × 类目 | 可解释、可监控 | 字典维护成本高 |
| 哈希交叉 | 极高基数组合 | 无需完整字典 | 有冲突，解释性差 |
| 统计特征 | 点击数、CTR、转化率 | 稳定、便宜 | 容易泄漏，需要平滑 |
| 序列建模 | 行为顺序重要的场景 | 能表达兴趣变化 | 延迟和实现复杂度高 |
| 手工分箱 | 非线性明显、需解释 | 稳定、鲁棒 | 边界需要维护 |
| 连续输入 | 数值关系较平滑 | 信息损失少 | 对尺度和异常值敏感 |

低基数类别可以优先 one-hot。例如网络类型只有 Wi-Fi、4G、5G、未知几类，one-hot 足够清晰。高基数类别更适合 Embedding，例如商品 ID、用户 ID、品牌 ID。

极高基数交叉可以优先哈希，而不是维护显式字典。例如 `user_id × category_id` 在大系统里可能有千万级组合，显式字典更新成本高。哈希交叉更工程化，但要接受冲突风险。

简单场景下，规则特征加线性模型可能比复杂深度特征更稳定。例如小规模 B2B 推荐、内容量有限的内部系统、强解释要求的金融排序，都不一定需要复杂序列模型。统计特征、分箱特征、少量交叉特征，配合逻辑回归或 GBDT，可能更容易上线和排查。

可以用四个维度做选择：

| 决策维度 | 优先考虑 |
|---|---|
| 数据规模小 | one-hot、统计特征、手工分箱 |
| 数据规模大 | Embedding、哈希交叉、序列建模 |
| 延迟要求高 | 轻量统计、预计算 Embedding、短序列 |
| 可解释性要求高 | 分箱、显式交叉、线性模型 |
| 稳定性优先 | 少量强特征、固定统计口径、严格特征监控 |

排序模型特征工程的实践原则可以压缩成一句话：先保证特征正确，再追求特征丰富；先保证训练线上一致，再追求模型表达能力。

---

## 参考资料

1. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)：支持本文关于稀疏交叉特征与深度表示结合的讨论。
2. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)：支持本文关于推荐系统中类别特征、Embedding 和交互建模的讨论。
3. [Keras preprocessing layers guide](https://www.tensorflow.org/guide/keras/preprocessing_layers)：支持本文关于训练和推理共用预处理逻辑的讨论。
4. [Keras Normalization layer](https://keras.io/2/api/layers/preprocessing_layers/numerical/normalization/)：支持本文关于数值特征标准化和统计量复用的讨论。
5. [Keras HashedCrossing layer](https://keras.io/api/layers/preprocessing_layers/categorical/hashed_crossing/)：支持本文关于哈希交叉特征的讨论。
6. [TensorFlow Recommenders: Taking advantage of context features](https://www.tensorflow.org/recommenders/examples/context_features)：支持本文关于上下文特征进入推荐模型的讨论。
7. [TensorFlow Embedding layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding)：支持本文关于 Embedding 查表机制的讨论。

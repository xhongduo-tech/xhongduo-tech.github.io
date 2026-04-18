## 核心结论

Wide & Deep 是一种面向推荐、广告 CTR、排序任务的联合学习框架。它的核心不是“把两个模型拼起来”，而是在同一个预测目标下，让 Wide 分支负责记忆高频显式规则，让 Deep 分支负责泛化稀疏特征之间的相似关系。

CTR 是 Click-Through Rate，指用户点击某个候选物品的概率。排序模型通常不是直接决定“推荐什么”，而是给每个候选物品打分，再按分数排序。

Wide & Deep 的预测形式可以写成：

$$
p(y=1|x)=\sigma(z_w+z_d+b)
$$

其中 $z_w$ 是 Wide 分支输出的 logit，$z_d$ 是 Deep 分支输出的 logit，$b$ 是偏置，$\sigma$ 是 sigmoid 函数。logit 是进入 sigmoid 前的原始分数，数值越大，预测概率通常越高。

| 分支 | 作用 | 优势 | 风险 |
|---|---|---|---|
| Wide | 记忆显式交叉 | 可解释、命中高频规则快 | 特征爆炸 |
| Deep | 学习稀疏表示 | 泛化强、可处理相似模式 | 过度泛化 |

玩具例子：如果“学生用户 + 编程学习 App”在历史数据里经常一起出现，Wide 分支可以直接记住这个组合，给它较高分数。若遇到“刚毕业用户 + AI 工具 App”这种没见过的新组合，Deep 分支仍能根据用户画像相似、App 类别相似给出合理分数。

真实工程例子：Google Play 排序里，候选 App 很多，用户、设备、地域、历史安装、App 类别等特征高度稀疏。Wide 侧可以记住“某类用户 + 某类 App”的高频安装模式，Deep 侧可以学习“相似用户可能喜欢相似 App”的泛化规律。

---

## 问题定义与边界

Wide & Deep 解决的不是所有分类问题，而是高维稀疏特征下的排序、CTR 预估、推荐预测问题。稀疏特征是指大部分取值都为空或为 0 的特征，例如用户 ID、商品 ID、城市、设备类型、App 类别。每个样本只命中极少数取值，但总词表可能非常大。

| 维度 | 典型输入 | 典型输出 |
|---|---|---|
| 业务场景 | 推荐、广告 CTR、排序 | 点击概率、安装概率、转化概率 |
| 数据特征 | 用户、上下文、候选物品、交叉特征 | 稀疏为主 |
| 目标函数 | 二分类或排序损失 | 概率分数/排序分数 |

在电商推荐中，用户性别、年龄桶、设备、地域、品类点击历史、商品 ID 都可以是稀疏特征。模型既要学会“老用户 + 熟悉品类”的历史规则，也要判断“没见过的新组合是否相近”。

Wide & Deep 适合三个条件同时存在的场景：

| 条件 | 含义 | 为什么重要 |
|---|---|---|
| 稀疏特征占主导 | 输入里大量是 ID、类别、分桶特征 | Deep 需要 embedding 学习表示 |
| 显式交叉有价值 | 某些组合本身强于单个特征 | Wide 可以直接记住规则 |
| 训练与线上特征一致 | 离线训练和线上服务使用同一套特征生成逻辑 | 否则离线指标会失真 |

它不适合把所有任务都替代掉。纯文本理解通常依赖语言模型或序列模型；纯视觉任务通常依赖卷积网络或视觉 Transformer；连续数值特征为主的表格任务可能用树模型更简单。Wide & Deep 的优势来自“稀疏特征 + 高价值交叉 + 需要泛化”的组合。

---

## 核心机制与推导

Wide 侧是线性模型加显式交叉特征。线性模型是把每个特征乘上一个权重后求和的模型，结构简单，但可解释性强。显式交叉特征是人为构造的组合特征，例如 `user_age_bucket=18_24 AND app_category=education`。如果这个组合经常对应点击或安装，Wide 分支可以直接给它较大权重。

Deep 侧是 embedding 加 MLP。embedding 是把离散 ID 映射成低维向量的方法，例如把一个 App 类别从 one-hot 向量变成 16 维或 32 维稠密向量。MLP 是多层感知机，即多层全连接神经网络，用来学习非线性组合关系。

设原始特征为 $x$，交叉特征为 $\phi(x)$。Wide 分支输出为：

$$
z_w = w_{wide}^T [x,\phi(x)]
$$

其中 $[x,\phi(x)]$ 表示把原始特征和交叉特征拼在一起，$w_{wide}$ 是 Wide 侧权重。

Deep 分支先把稀疏特征映射为 embedding，再经过多层网络，最后一层表示为 $h_L$：

$$
z_d = w_{deep}^T h_L
$$

最终输出为：

$$
p(y=1|x)=\sigma(z_w+z_d+b)
$$

结构可以按下面理解：

```text
输入特征
  |-- Wide: 原始特征 + 交叉特征 -> 线性层 -> z_w
  |-- Deep: 稀疏特征 -> embedding -> MLP -> z_d
                         |
                    z_w + z_d + b
                         |
                      sigmoid
                         |
                    点击/转化概率
```

为什么需要交叉特征：单个特征可能不够表达业务规则。“北京用户”不一定点击教育 App，“教育 App”也不一定被点击，但“北京学生用户 + 考研 App”可能是强信号。这个组合信号如果只靠单个特征，很难表达完整含义。

为什么 Wide 适合记忆：Wide 分支直接给交叉特征分配权重。只要历史数据里某个组合出现频率足够高，模型就可以稳定学到这个组合的正负影响。

为什么 Deep 适合泛化：Deep 分支不会只依赖固定组合是否出现过，而是学习向量空间里的相似性。即使“年轻用户 + AI 笔记 App”没有大量历史样本，只要它和“年轻用户 + 效率工具 App”的 embedding 接近，模型仍能给出合理估计。

为什么必须联合训练：Joint Training 是同一个损失同时更新 Wide 和 Deep 两路参数，不是先训练两个模型再投票。ensemble 是多个模型各自训练后再融合；Wide & Deep 是一个模型里两条分支共同优化同一个目标。这样 Wide 学到的强规则和 Deep 学到的泛化信号会在同一次反向传播里被校准。

数值例子：假设某个样本命中了高频交叉特征，Wide 给出 $z_w=1.5$，Deep 给出 $z_d=0.2$，偏置 $b=-0.3$。总 logit 为：

$$
z=1.5+0.2-0.3=1.4
$$

所以：

$$
p=\sigma(1.4)\approx 0.80
$$

如果只看 Wide：

$$
\sigma(1.5-0.3)=\sigma(1.2)\approx 0.77
$$

如果只看 Deep：

$$
\sigma(0.2-0.3)=\sigma(-0.1)\approx 0.48
$$

这个例子说明：Wide 把历史强规则推高，Deep 提供额外泛化信号，最终分数由两路共同决定。

---

## 代码实现

工程实现要把特征处理、Wide 分支、Deep 分支、损失函数、训练流程拆开。不要把所有逻辑混成一个不可检查的黑盒。

常见伪代码如下：

```python
wide_logit = linear(concat(raw_features, crossed_features))
deep_input = embed(sparse_features)
deep_logit = mlp(deep_input)
logit = wide_logit + deep_logit + bias
prob = sigmoid(logit)
loss = binary_cross_entropy(prob, label)
```

| 模块 | 常见实现 | 说明 |
|---|---|---|
| Wide | 线性层 + 交叉特征 | 记忆显式规则 |
| Deep | embedding + MLP | 泛化稀疏特征 |
| Loss | BCE / logloss | 端到端优化 |

BCE 是 Binary Cross Entropy，中文常叫二分类交叉熵，用来衡量预测概率和真实 0/1 标签之间的差距。推荐和广告 CTR 里，点击是 1，未点击是 0，BCE 是很常见的训练目标。

下面是一个可运行的最小 Python 例子。它不依赖深度学习框架，只演示 Wide logit、Deep logit、sigmoid、BCE 的核心计算：

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def binary_cross_entropy(prob, label):
    eps = 1e-12
    prob = min(max(prob, eps), 1 - eps)
    return -(label * math.log(prob) + (1 - label) * math.log(1 - prob))

def wide_and_deep_predict(wide_logit, deep_logit, bias):
    logit = wide_logit + deep_logit + bias
    return sigmoid(logit)

# 玩具样本：历史高频交叉命中，Deep 也给出轻微正向信号
z_w = 1.5
z_d = 0.2
b = -0.3

prob = wide_and_deep_predict(z_w, z_d, b)
loss = binary_cross_entropy(prob, label=1)

assert round(prob, 2) == 0.80
assert loss < 0.25

wide_only = sigmoid(z_w + b)
deep_only = sigmoid(z_d + b)

assert round(wide_only, 2) == 0.77
assert round(deep_only, 2) == 0.48
```

如果落地到推荐系统，输入可以是：

| 输入字段 | 示例 | 进入分支 |
|---|---|---|
| 用户 ID | `user_1024` | Deep embedding |
| 年龄桶 | `age_18_24` | Wide + Deep |
| 设备类型 | `ios` | Wide + Deep |
| App 类别 | `education` | Wide + Deep |
| 交叉特征 | `age_18_24_x_education` | Wide |
| 输出 | 点击/安装概率 | 排序分数 |

训练步骤可以简化为：

```python
for batch in training_data:
    raw_features = build_raw_features(batch)
    crossed_features = build_cross_features(batch)
    sparse_features = build_sparse_features(batch)

    wide_logit = wide_branch(raw_features, crossed_features)
    deep_logit = deep_branch(sparse_features)

    logit = wide_logit + deep_logit + bias
    loss = binary_cross_entropy_with_logits(logit, batch.labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

原始 Wide & Deep 工作中，Wide 侧常配 FTRL + L1。FTRL 是 Follow-The-Regularized-Leader，一种适合大规模稀疏线性模型的优化算法；L1 正则会鼓励大量权重变成 0，便于特征选择。Deep 侧常配 AdaGrad 或类似优化器。AdaGrad 会根据历史梯度调整学习率，对稀疏特征较友好。

最关键的工程要求是：训练和线上特征生成链路必须一致。离线训练时怎么分桶、怎么构造交叉、怎么处理 OOV，线上服务也必须一样。OOV 是 Out Of Vocabulary，指线上出现了训练词表里没有的新类别。

---

## 工程权衡与常见坑

Wide & Deep 的最大工程风险通常不是模型不够复杂，而是特征工程失控、离线线上不一致、词表更新不稳定。模型结构只是第一层，真正影响效果的是数据和特征链路。

| 常见坑 | 现象 | 规避方法 |
|---|---|---|
| 当成 ensemble | 训练目标不统一 | 确认同一损失、同一次反传 |
| 交叉特征过多 | 维度爆炸、过拟合 | 只保留高价值 cross |
| Deep 过度泛化 | 长尾规则被抹平 | 增加稀疏覆盖、分桶、约束 |
| 特征不一致 | 线上效果掉 | 统一特征生成链路 |
| OOV / 冷启动差 | 新词表失效 | 默认桶 + 稳定更新策略 |

交叉特征不是越多越好。如果把所有二阶、三阶交叉都塞进 Wide，维度会迅速膨胀。二阶交叉是两个特征组合，三阶交叉是三个特征组合。假设有 1000 个类别特征，两两组合的候选空间接近 $1000^2$，大量组合样本很少，训练会变慢，也容易记住噪声。

Deep 也不是越深越好。Deep 分支会把相似样本拉近，这是泛化能力的来源，但也可能把特殊高频组合抹平。例如某个小众设备型号对某类 App 转化特别高，纯 Deep 模型可能把它并入普通设备模式，导致排序损失。Wide 分支保留这类显式规则，可以减少这种问题。

线上/离线一致性可以按下面检查：

```text
原始日志
  -> 离线特征生成
  -> 训练样本
  -> 模型训练
  -> 模型发布

线上请求
  -> 在线特征生成
  -> 同一套分桶、词表、交叉逻辑
  -> 模型预测
  -> 排序结果
```

真实工程中，要特别关注这些细节：

| 检查项 | 错误后果 |
|---|---|
| 分桶边界是否一致 | 年龄、价格、时长等特征分布错位 |
| 词表版本是否一致 | embedding 查错或大量 OOV |
| 交叉规则是否一致 | Wide 侧命中特征变化 |
| 缺失值默认值是否一致 | 训练和线上统计分布不一致 |
| 特征时间窗是否一致 | 训练偷看未来或线上信号缺失 |

Wide & Deep 的权衡可以概括为：Wide 提供稳定、可解释、命中快的历史规则；Deep 提供可扩展、能处理新组合的泛化能力。代价是工程链路更复杂，特征治理要求更高。

---

## 替代方案与适用边界

Wide & Deep 不是推荐系统的默认最优解。它适合“既要记忆又要泛化”的场景；如果数据形态或业务目标不同，替代方案可能更简单、更稳。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 线性模型 | 简单、可解释 | 泛化弱 | 强规则、低复杂度 |
| 纯 MLP | 表达能力强 | 稀疏特征不稳 | 密集特征较多 |
| Wide & Deep | 记忆 + 泛化 | 工程复杂 | 推荐、广告 CTR |
| 树模型 | 非线性强 | 大规模稀疏特征弱 | 表格数据 |

新手版对比：只用线性模型，容易记住“某类用户 + 某类商品”的高频规则，但遇到没见过的复杂组合会弱。只用 MLP，能学习复杂关系，但对大规模稀疏 ID 特征常常不够稳定。Wide & Deep 同时保留两种能力，但需要更严格的特征工程和训练链路。

适用 Wide & Deep 前，可以先问三个问题：

| 问题 | 如果答案是“是” | 如果答案是“否” |
|---|---|---|
| 特征是否主要稀疏 | Wide & Deep 值得考虑 | 普通 MLP、树模型可能更简单 |
| 是否存在高频交叉规则 | Wide 侧有价值 | 交叉特征收益有限 |
| 是否要求线上低延迟 | 需要控制模型规模 | 可以考虑更复杂的重排序模型 |

如果特征主要是连续数值，且交叉关系不明显，树模型或普通深度模型可能更直接。如果目标是文本语义匹配，双塔模型或交叉编码器通常更合适。双塔模型是分别编码用户和物品，再计算相似度的结构；交叉编码器是把两侧输入放在一起，让模型直接学习细粒度匹配关系。

在大规模推荐系统里，Wide & Deep 常用于精排层。精排层是召回之后、最终展示之前的排序阶段，候选数量已经减少，可以使用更复杂的特征和模型。若候选数量特别大，通常会先用召回模型或粗排模型过滤，再交给 Wide & Deep 或更复杂模型打分。

---

## 参考资料

建议阅读顺序：

1. 先看原始论文，理解 Wide & Deep 为什么要同时解决 memorization 和 generalization。
2. 再看 Google Research 页面，确认它在 Google Play 推荐场景中的定位。
3. 最后看 TensorFlow 教程和博客示例，理解工程实现、训练、调参和 Keras Functional API 的落地方式。

| 资料 | 作用 |
|---|---|
| Wide & Deep Learning for Recommender Systems | 理解理论与结构 |
| Google Research: Wide & Deep Learning for Recommender Systems | 理解应用背景 |
| TensorFlow: Tuning a wide and deep model using Google Cloud | 理解工程训练 |
| TensorFlow Blog: Predicting the price of wine with the Keras Functional API and TensorFlow | 理解 Keras 落地 |

参考链接：

- 原始论文：<https://arxiv.org/abs/1606.07792>
- Google Research 页面：<https://research.google/pubs/wide-deep-learning-for-recommender-systems/>
- TensorFlow 官方教程：<https://www.tensorflow.org/cloud/tutorials/hp_tuning_wide_and_deep_model>
- TensorFlow 官方博客示例：<https://blog.tensorflow.org/2018/04/predicting-price-of-wine-with-keras-api-tensorflow.html>

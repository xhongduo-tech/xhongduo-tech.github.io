## 核心结论

Wide & Deep 是一种推荐排序框架：把 Wide 侧的显式规则记忆和 Deep 侧的低维表示泛化放进同一个模型里，使用同一个标签、同一个损失函数联合训练。

它不是“两个模型训练完再平均分数”，而是在预测时把两路 logit 先相加，再统一进入 sigmoid：

$$
p(y=1\mid x)=\sigma\big(w_w^\top [x,\phi(x)] + w_d^\top a^{(L)} + b\big)
$$

其中，logit 是进入 sigmoid 前的原始打分；$\phi(x)$ 是人工构造的交叉特征；$a^{(L)}$ 是 Deep 侧最后一层网络输出。

| 结构 | 主要作用 | 优点 | 主要风险 |
|---|---|---|---|
| Wide | 记住高频、明确的特征组合 | 可解释，能强记历史规则 | 泛化弱，容易只会记热门模式 |
| Deep | 学习稀疏特征的 embedding 表示 | 能处理未见过的新组合 | 可能把不该相似的样本学得太近 |
| Wide & Deep | 同时记忆高频规则和泛化长尾组合 | 排序更稳，适合推荐场景 | 特征治理成本高 |

玩具例子：用户 A 经常点击“程序员 + 机械键盘”这类商品。Wide 侧可以直接记住 `occupation=engineer AND category=keyboard` 这个强关联；Deep 侧会学习“程序员、开发者、数据工程师”这些用户特征在 embedding 空间里相近，从而对“数据工程师 + 人体工学键盘”这种少见组合也给出合理分数。

真实工程例子：应用商店首页排序里，输入包括用户历史安装、当前曝光应用、设备、地区、时间等特征。Wide 侧记住 `user_installed_app=Netflix AND impression_app=Pandora` 这类高频共现规则；Deep 侧学习用户、App、设备、地区之间更抽象的相似关系。这样模型既能利用历史强规则，也能处理新 App、新用户、新组合。

为什么需要它：推荐排序的数据通常稀疏、组合巨大、长尾样本多。只用 Wide 容易过拟合历史高频组合，只用 Deep 又可能欠拟合某些明确业务规则。Wide & Deep 的价值在于把这两类能力放进一个端到端目标里共同优化。

---

## 问题定义与边界

Wide & Deep 主要解决的是排序问题：给定用户、物品、上下文等特征，预测某个候选对象被点击、安装、购买或转化的概率，再按概率排序。

| 项目 | 定义 |
|---|---|
| 输入 | 用户特征、物品特征、上下文特征、历史行为特征 |
| 输出 | 点击率、转化率、安装率等概率 |
| 目标 | 提升排序质量，而不只是提升离线分类指标 |
| 典型任务 | 推荐系统排序、广告排序、搜索结果重排 |
| 非核心目标 | 通用图像分类、文本生成、无监督聚类 |

输入特征通常分为三类：

| 特征类型 | 白话解释 | 示例 |
|---|---|---|
| 稀疏类别特征 | 取值很多、每个样本只命中少数值的离散特征 | 用户 ID、商品 ID、城市、设备型号 |
| 数值特征 | 可以直接表示大小的连续或离散数值 | 年龄、价格、历史点击次数 |
| 交叉特征 | 多个特征组合后形成的新特征 | `city=Beijing AND category=phone` |

它适合“规则明显 + 长尾泛化并存”的场景。例如应用商店首页排序：用户历史安装、当前曝光 App、设备、地区、时间都很稀疏。单独 Wide 容易只记住热门 App 或常见地区规则；单独 Deep 可能把一些表面相似但业务含义不同的组合拉得太近；联合训练通常更稳定。

它不适合所有分类任务。如果数据量很小、特征很少、业务规则简单，逻辑回归或 GBDT 可能更稳。如果主要输入是图像、语音、长文本，Wide 侧手工交叉的价值会下降。如果系统没有稳定的特征生产管道，Wide & Deep 反而会增加训练和线上一致性的风险。

joint training 与 ensemble 的区别很关键。joint training 是 Wide 和 Deep 共用一个损失函数，参数一起反向传播更新；ensemble 是多个模型各自训练，再把输出平均、加权或投票。Wide & Deep 的核心不是“融合结果”，而是“融合训练目标下的两类特征表达”。

---

## 核心机制与推导

Wide 侧通常是线性模型。线性模型是指每个输入特征乘以一个权重后相加。它接收原始 one-hot 特征和人工交叉特征，用来显式记忆高价值规则：

$$
z_{wide}=w_w^\top [x,\phi(x)]
$$

Deep 侧通常是 embedding 加多层感知机。embedding 是把高维稀疏类别映射成低维稠密向量；多层感知机 MLP 是由多层线性变换和非线性激活组成的前馈网络。Deep 侧流程是：

```text
稀疏类别特征 -> embedding 查表 -> 向量拼接 -> MLP -> 最后一层表示 a^(L)
```

Deep 侧 logit 为：

$$
z_{deep}=w_d^\top a^{(L)}
$$

最终模型不是分别计算两个概率，而是先加总 logit：

$$
z=z_{wide}+z_{deep}+b
$$

再进入 sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

机制流程图可以写成：

```text
输入特征
  ├─ Wide 分支：one-hot + cross feature -> 线性 logit
  └─ Deep 分支：embedding -> MLP -> deep logit
                         ↓
                logits 相加 + bias
                         ↓
                      sigmoid
                         ↓
                    点击/转化概率
```

数值例子：假设某个样本的 Wide logit 为 $1.2$，Deep logit 为 $-0.4$，bias 为 $0$。总 logit 是：

$$
z=1.2+(-0.4)+0=0.8
$$

预测概率是：

$$
\sigma(0.8)=\frac{1}{1+e^{-0.8}}\approx 0.690
$$

如果真实标签 $y=1$，二分类交叉熵损失为：

$$
-\log(0.690)\approx 0.371
$$

这个推导说明两个事实。第一，Wide 侧和 Deep 侧不是各自输出一个概率再合并。第二，损失函数看到的是合并后的总 logit，所以训练会同时调整“规则记忆”和“泛化表示”。

为什么它能同时记忆和泛化：Wide 侧的交叉特征直接给高频组合分配参数，只要组合在历史中足够稳定，模型就能快速记住；Deep 侧把稀疏类别压到低维空间，让相似用户、相似商品、相似上下文共享统计强度，从而处理训练集中没充分出现过的新组合。

---

## 代码实现

最小实现重点有四个：Wide 特征、Deep embedding、两路 logit 融合、同一个 binary cross-entropy 损失。binary cross-entropy 是二分类常用损失，用来惩罚预测概率和真实 0/1 标签之间的差距。

| 原始信息 | Wide 输入 | Deep 输入 |
|---|---|---|
| 用户职业 | one-hot | occupation embedding |
| 商品类别 | one-hot | category embedding |
| 用户职业 + 商品类别 | 显式 cross feature | 由 MLP 自动学习交互 |
| 历史点击次数 | 数值特征 | 数值特征拼接或分桶后 embedding |

下面是一个可运行的 Python 玩具实现，不依赖深度学习框架，只演示“Wide logit + Deep logit + bias -> sigmoid -> loss”的核心计算。

```python
import math

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def binary_cross_entropy(y, p):
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))

# Wide 侧：显式记住一个交叉特征
wide_weights = {
    "occupation=engineer": 0.3,
    "category=keyboard": 0.4,
    "occupation=engineer&category=keyboard": 0.5,
}

# Deep 侧：这里用已经算好的 deep_logit 代表 embedding + MLP 的输出
features = ["occupation=engineer", "category=keyboard"]
cross = "occupation=engineer&category=keyboard"

wide_logit = sum(wide_weights.get(f, 0.0) for f in features)
wide_logit += wide_weights.get(cross, 0.0)

deep_logit = -0.4
bias = 0.0
logit = wide_logit + deep_logit + bias
prob = sigmoid(logit)
loss = binary_cross_entropy(1, prob)

assert abs(wide_logit - 1.2) < 1e-9
assert abs(logit - 0.8) < 1e-9
assert 0.68 < prob < 0.70
assert 0.36 < loss < 0.38

print(round(prob, 3), round(loss, 3))
```

工程伪代码通常是：

```python
wide_logit = WideFeatures(x, cross_features)
deep_logit = DeepMLP(embeddings(x))
logit = wide_logit + deep_logit + bias
prob = sigmoid(logit)
loss = binary_cross_entropy(y, prob)
```

训练时必须保证 Wide 和 Deep 共享同一批样本、同一标签、同一损失函数。不能先训练一个 Wide 模型，再训练一个 Deep 模型，然后把两个概率简单平均；那是 ensemble，不是 Wide & Deep 的联合训练。

训练和 serving 的特征一致性是硬约束。训练时怎么分桶、归一化、生成交叉特征，线上预测时就必须完全一致。尤其是交叉特征名称、缺失值处理、时间窗口统计、类别截断规则，只要有一个环节不一致，线上分数就会系统性偏移。

---

## 工程权衡与常见坑

Wide & Deep 的主要工程成本不是 MLP 有几层，而是特征设计、样本构造、时间窗口、线上线下一致性和监控。

| 问题 | 表现 | 原因 | 规避方法 |
|---|---|---|---|
| 交叉特征爆炸 | 参数量大，训练慢，长尾特征无效 | 组合空间远大于样本量 | 只保留高频、高价值、可解释的 cross |
| 数据泄漏 | 离线 AUC 很高，线上效果变差 | 使用了曝光后才发生的未来行为 | 严格按时间切分，检查特征可用时点 |
| 训练和线上不一致 | 离线稳定，线上分数漂移 | 特征生成逻辑不统一 | 复用同一套特征管道或同一份配置 |
| embedding 过拟合 | 训练集好，验证集差 | 维度过大或正则不足 | 控制维度、加正则、早停 |
| 只看 AUC | 离线提升，业务无收益 | 排序目标和业务指标不一致 | 同时看 CTR、CVR、留存、收入等指标 |
| 把 joint training 当 ensemble | 两路模型互相不影响 | 各自训练后才融合 | 确认端到端反传和统一损失 |

训练、验证、线上一致性检查清单：

| 检查项 | 必须确认的问题 |
|---|---|
| 时间切分 | 验证集是否晚于训练集 |
| 特征时点 | 每个特征在线上预测时是否已经可用 |
| 缺失值 | 训练和线上是否使用同一默认值 |
| 类别词表 | 未登录类别、新类别、低频类别如何处理 |
| 交叉特征 | cross 的拼接顺序和命名是否一致 |
| 指标 | 离线指标是否能解释线上业务目标 |

未来行为泄漏是最典型的坑。例如预测“用户是否点击当前曝光 App”，却把“曝光后 10 分钟内是否安装过类似 App”作为训练特征。模型离线看起来很强，因为它偷看了答案附近的信息；线上预测时这个特征根本不存在，效果会下降。

过拟合控制需要同时管 Wide 和 Deep。Wide 侧要限制交叉特征规模，避免每个罕见组合都有独立参数。Deep 侧要控制 embedding 维度、使用 L2 正则、dropout 或早停。早停是指验证集指标不再提升时停止训练，避免模型继续记住训练噪声。

cross feature 不应越多越好。更稳的做法是从强业务先验开始，例如“用户历史行为类别 × 当前物品类别”“地区 × 服务可用性”“设备 × App 兼容性”。如果一个交叉项既低频又难解释，通常不该优先加入。

---

## 替代方案与适用边界

Wide & Deep 不是推荐排序的唯一答案。它适合规则明显、稀疏特征多、长尾组合多的场景；如果特征关系更复杂，或者希望模型自动学习更多交互，其他结构可能更合适。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| Wide-only | 规则简单、业务先验强 | 可解释，部署简单 | 泛化弱 |
| Deep-only | 特征丰富、语义结构强 | 能学习低维表示 | 可能忽略明确规则 |
| Wide & Deep | 高频规则和长尾泛化并存 | 平衡记忆与泛化 | 特征工程成本高 |
| DeepFM / xDeepFM | 需要自动学习特征交互 | 减少人工 cross 设计 | 结构更复杂，调参成本更高 |
| GBDT + LR | 传统表格任务、可解释需求强 | 工程成熟，鲁棒 | 对超大规模稀疏 ID 泛化有限 |

适用边界可以按三点判断。第一，看业务目标：如果目标是推荐排序、广告点击、转化预测，Wide & Deep 有明确适配性。第二，看数据规模：如果样本少、类别少，复杂网络未必值得；如果样本大、稀疏类别多，embedding 的价值更明显。第三，看特征形态：如果有大量稳定交叉规则，同时也有新用户、新物品、新组合，Wide & Deep 更有优势。

为什么不总是越复杂越好：复杂模型会增加训练成本、调参成本、解释成本和线上延迟。推荐系统最终看的是业务收益，不是结构是否先进。如果一个 GBDT + LR 已经满足延迟、稳定性和可解释性要求，盲目升级到更复杂模型可能只会增加维护负担。

选择建议是：规则强、数据中等、解释要求高时，先用 Wide-only 或 GBDT + LR；稀疏 ID 多、样本规模大、有明显相似性结构时，考虑 Deep-only 或 Wide & Deep；既要记住强规则，又要覆盖长尾新组合时，Wide & Deep 是稳妥基线；如果人工交叉特征维护困难，再考虑 DeepFM、xDeepFM 这类自动交互模型。

---

## 参考资料

| 来源 | 类型 | 适合阅读内容 |
|---|---|---|
| Wide & Deep Learning for Recommender Systems | 论文 | 原始定义、公式、Google Play 实验 |
| Google Research Blog | 官方博客 | 直观动机、记忆与泛化的解释 |
| TensorFlow Cloud Tutorial | 官方教程 | 工程训练、调参流程 |
| TensorFlow Feature Columns Tutorial | 官方教程 | 结构化特征处理方法 |

1. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)：适合查看 Wide & Deep 的原始定义、公式和推荐系统实验设置。
2. [Wide & Deep Learning: Better Together with TensorFlow](https://research.google/blog/wide-amp-deep-learning-better-together-with-tensorflow/)：适合理解 Google 对 memorization 与 generalization 的工程动机解释。
3. [Tuning a wide and deep model using Google Cloud](https://www.tensorflow.org/cloud/tutorials/hp_tuning_wide_and_deep_model)：适合参考 Wide & Deep 模型训练和调参流程。
4. [Classify structured data with feature columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns)：适合了解结构化数据、类别特征、交叉特征在 TensorFlow 中的处理方式。

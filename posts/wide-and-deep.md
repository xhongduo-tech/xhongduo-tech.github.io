## 核心结论

Wide & Deep 是一种把“记忆”和“泛化”放进同一个模型里的推荐排序架构。这里的“记忆”指把历史上反复出现、已经被验证有效的规则直接学下来；“泛化”指面对训练里没见过的新组合，仍然能给出合理预测。它的核心不是让一个更大的模型去包办一切，而是明确拆成两条分支：

| 分支 | 主要输入 | 擅长能力 | 典型问题 |
|---|---|---|---|
| Wide | 稀疏 one-hot 特征、人工交叉特征 | 记住高频规则 | 手工特征多，容易膨胀 |
| Deep | embedding、MLP | 学习隐式模式、处理未见组合 | 参数多，容易过拟合 |

最终预测不是二选一，而是把两边的 logit 直接相加：

$$
p=\sigma(y_{\text{wide}}+y_{\text{deep}})
$$

其中 $\sigma(\cdot)$ 是 sigmoid，白话解释就是“把任意实数压到 0 到 1 之间，变成点击概率或转化概率”。

玩具例子可以直接理解成一句话：把“常见搭配”交给 wide 记住，把“新组合”交给 deep 推测，最后把两部分打分加起来。比如“女性用户 + 游戏类 App”在历史上就是高点击组合，那么 wide 可以直接给高权重；如果来了“从没在训练里成批出现过的新用户画像”，deep 还能通过 embedding 的相似性给出可用估计。

---

## 问题定义与边界

Wide & Deep 主要解决 CTR 预测、推荐排序、广告预估这一类问题：特征非常稀疏，用户和物料组合极多，线上既要稳定利用历史规律，又不能只会背训练集。

更具体地说，这类任务常同时存在两种约束：

| 业务问题 | 本质约束 | 只用 Wide 的问题 | 只用 Deep 的问题 |
|---|---|---|---|
| 高频规则必须稳定命中 | 历史上有强模式 | 可以处理 | 容易学到，但不一定稳定保留 |
| 新组合也要有分数 | 稀疏组合巨大 | 几乎无能为力 | 可以泛化 |
| 线上需要可控性 | 规则可解释 | 强 | 弱 |
| 数据分布会漂移 | 需要持续适配 | 手工维护成本高 | 对数据量更敏感 |

因此它适合这样的场景：

1. 类别特征很多，例如用户国家、设备、频道、品类、作者、广告位。
2. 高阶规则确实存在，例如“搜索词 A + 类目 B + 时段 C”这种组合有明显转化偏好。
3. 训练样本虽然大，但长尾仍然很多，不能指望所有组合都被充分看到。

不太适合的场景也要说清楚：

1. 如果特征大多是连续稠密特征，且显式规则不强，Wide 的收益会下降。
2. 如果业务没有明显的人工交叉经验，Wide 侧可能维护成本过高。
3. 如果数据极少，Deep 侧泛化能力也会有限，整体未必优于简单线性模型。
4. 如果更关注序列行为、上下文动态兴趣，单纯 Wide & Deep 往往不如 DIN、Transformer 类模型。

其训练目标通常写成：

$$
L=\text{Loss}(y,\hat{y})+\text{regularization}
$$

如果是二分类点击任务，$\text{Loss}$ 常用 log loss。这里“正则化”就是对参数大小加约束，防止模型把训练集记得过死。

一个新手版例子：做游戏推荐时，历史数据说明“学生用户 + 免费游戏”点击率很高，这类规则适合 wide 直接记住；但“第一次出现的机型 + 某个新游戏子类”没有足够历史时，就需要 deep 根据 embedding 的相似性去推测。

---

## 核心机制与推导

Wide 分支本质上是带交叉特征的线性模型：

$$
y_{\text{wide}}=w^\top x+b
$$

这里的 $x$ 不只包含原始 one-hot 特征，也可以包含人工构造的交叉项。交叉项可以写成：

$$
\phi_k(x)=\prod_i x_i^{c_{ki}}
$$

白话解释：只有当一组特征同时出现时，这个交叉特征才为 1。比如“gender=female”和“category=game”都出现时，交叉项 `female_AND_game=1`。

Deep 分支先把离散 ID 变成 embedding。embedding 可以理解成“把离散类别映射成低维稠密向量，让相似对象在空间里更接近”。之后再送入多层感知器 MLP：

$$
h_0=[e_1;e_2;\dots;e_m]
$$

$$
h_{l+1}=f(W_lh_l+b_l)
$$

$$
y_{\text{deep}}=w_{\text{deep}}^\top h_{\text{deep}}+b_{\text{deep}}
$$

最终总 logit 为：

$$
y=y_{\text{wide}}+y_{\text{deep}}, \quad p=\sigma(y)
$$

联合训练意味着两边共享同一个损失，而不是各训各的。以点击标签 $t\in\{0,1\}$ 为例：

$$
L=-t\log p-(1-t)\log(1-p)
$$

再加正则项：

$$
L_{\text{total}}=L+\lambda_{\text{wide}}\|w\|^2+\lambda_{\text{deep}}\|W\|^2
$$

这会带来一个重要结果：如果某类规则非常稳定，wide 可以快速学到；如果遇到没见过的组合，deep 仍可以通过参数共享和 embedding 相似性补上分数。

下面用一个数值化玩具例子说明。

设某次曝光有以下特征：

- `gender=female`
- `country=CN`
- `category=game`

Wide 侧特征与权重：

| 特征 | 值 | 权重 | 贡献 |
|---|---:|---:|---:|
| female | 1 | 0.3 | 0.3 |
| country_CN | 1 | 0.1 | 0.1 |
| category_game | 1 | 0.4 | 0.4 |
| female_AND_game | 1 | 1.2 | 1.2 |

则：

$$
y_{\text{wide}}=0.3+0.1+0.4+1.2=2.0
$$

Deep 侧经过 embedding 和 MLP 后输出：

$$
y_{\text{deep}}=0.5
$$

所以总 logit：

$$
y=2.0+0.5=2.5
$$

最终概率：

$$
p=\sigma(2.5)\approx 0.924
$$

这个流程可以压成一个表：

| 阶段 | 输出 |
|---|---|
| 原始特征 | `female, CN, game` |
| Wide | 根据线性权重和交叉项得到 `2.0` |
| Deep | 根据 embedding + MLP 得到 `0.5` |
| 合并 | `2.5` |
| Sigmoid | `0.924` |

真实工程例子是 Google Play 商店推荐。它面对的是海量用户、海量应用、极其稀疏的点击与安装日志。Wide 侧适合记住“某类用户在某类 app 上的稳定偏好”，Deep 侧则处理“未充分曝光的新 app、新用户组合”。这类场景里，如果只有线性模型，长尾泛化会很差；如果只用深度网络，训练和线上稳定性又不一定够好。Wide & Deep 的价值正是把这两类能力同时保留。

---

## 代码实现

下面给出一个可运行的极简 Python 示例。它不是生产级训练代码，但完整展示了 Wide logit、Deep logit、求和、sigmoid 和 log loss 的计算过程。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ---------- Wide part ----------
# 稀疏特征直接写成字典：feature -> value
wide_features = {
    "female": 1.0,
    "country_CN": 1.0,
    "category_game": 1.0,
    "female_AND_game": 1.0,
}

wide_weights = {
    "female": 0.3,
    "country_CN": 0.1,
    "category_game": 0.4,
    "female_AND_game": 1.2,
}

def wide_logit(features, weights, bias=0.0):
    return sum(features.get(k, 0.0) * weights.get(k, 0.0) for k in features) + bias

# ---------- Deep part ----------
# 这里用一个极简“embedding + 一层线性”来模拟 DNN 输出
embeddings = {
    "female": [0.8, 0.1],
    "country_CN": [0.2, 0.6],
    "category_game": [0.9, 0.7],
}

hidden_weights = [0.4, 0.2]  # 模拟最后一层投影
deep_bias = -0.05

def avg_embedding(keys):
    dim = len(embeddings[keys[0]])
    out = [0.0] * dim
    for k in keys:
        for i, v in enumerate(embeddings[k]):
            out[i] += v
    return [v / len(keys) for v in out]

def relu(x: float) -> float:
    return max(0.0, x)

def deep_logit(keys):
    e = avg_embedding(keys)
    # 简化版：平均 embedding 后做线性变换
    z = sum(w * v for w, v in zip(hidden_weights, e)) + deep_bias
    return relu(z)

def log_loss(y_true: int, p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return -(y_true * math.log(p) + (1 - y_true) * math.log(1 - p))

yw = wide_logit(wide_features, wide_weights)
yd = deep_logit(["female", "country_CN", "category_game"])
logit = yw + yd
prob = sigmoid(logit)
loss = log_loss(1, prob)

assert round(yw, 4) == 2.0
assert yd > 0
assert 0 < prob < 1
assert loss >= 0

print("wide_logit =", round(yw, 4))
print("deep_logit =", round(yd, 4))
print("prob =", round(prob, 4))
print("loss =", round(loss, 4))
```

这段代码对应的结构就是：

```python
wide_logit = linear(wide_features)
deep_logit = dense(relu(dense(embedding(features))))
logit = wide_logit + deep_logit
prob = sigmoid(logit)
```

生产实现通常会进一步拆成三部分：

1. Wide 输入层：one-hot、bucketized 数值特征、人工交叉特征或 hash crossing。
2. Deep 输入层：用户 ID、物料 ID、品类、上下文离散特征的 embedding。
3. 联合输出层：两路 logit 相加，统一优化目标。

常见超参数影响如下：

| 超参数 | 作用 | 过小风险 | 过大风险 |
|---|---|---|---|
| embedding size | 控制类别向量容量 | 表达力不足 | 过拟合、显存增大 |
| hidden units | 控制 Deep 分支容量 | 学不到复杂模式 | 训练慢、过拟合 |
| learning rate | 控制更新步长 | 收敛慢 | 不稳定、震荡 |
| wide L2 | 控制规则权重大小 | 规则过强 | 规则学不起来 |
| deep dropout | 抑制过拟合 | 效果有限 | 欠拟合 |

如果是 TensorFlow、PyTorch 或其他工业框架，核心都不会变：Wide 是显式规则通道，Deep 是隐式模式通道，联合损失一起训练。

---

## 工程权衡与常见坑

Wide & Deep 真正难的不是把论文结构写出来，而是把它稳定放进线上系统。

最常见的问题是 feature explosion，也就是“特征爆炸”。一旦人工交叉做得太多，特征空间会迅速膨胀。比如用户端 100 个离散域、物料端 100 个离散域，如果盲目两两交叉，存储、训练和推理都会变重。解决思路通常不是“多上机器”，而是先筛选交叉项：只保留高频、高收益组合，或者对交叉特征做 hashing。

第二类问题是 Deep 分支过强或过弱。过强时，模型看起来 AUC 很高，但 wide 的规则价值被淹没，线上表现不稳定；过弱时，模型退化成“带点 embedding 的线性模型”，对新组合帮助有限。因此很多工程团队会分别给 wide 和 deep 配不同正则、不同学习率，甚至对 deep 做 warmup。warmup 的白话解释是“让某一部分先慢一点学，避免一开始把整体训练方向带偏”。

第三类问题是训练分布与服务分布不一致。Wide 侧高度依赖特征工程，如果训练时的交叉规则在线上构造不一致，效果会直接坍塌。Deep 侧如果 embedding 的字典、截断、OOV 规则和线上不一致，也会产生严重偏移。

下面把常见坑压成表：

| 常见坑 | 现象 | 典型原因 | 对策 |
|---|---|---|---|
| 交叉特征过多 | 训练慢、模型大 | 人工枚举过度 | top-k 筛选、hash crossing |
| Deep 过拟合 | 离线高、线上掉 | embedding/hidden 太大 | dropout、L2、early stop |
| 一侧支配训练 | Wide 或 Deep 基本不起作用 | 学习率和正则失衡 | 分支级调参、warmup |
| 特征不一致 | 线上效果异常 | 训练/推理口径不同 | 特征平台统一生成 |
| 冷启动差 | 新物料没分数 | 历史行为不足 | 加内容特征、召回兜底 |

联合损失可以写成：

$$
L=L_{\text{sigmoid}}+\lambda_{\text{wide}}\|w\|^2+\lambda_{\text{deep}}\|W\|^2
$$

这个式子对应的工程含义非常直接：wide 和 deep 不是“都加个正则就完了”，而是两边往往需要不同的约束强度。

一个新手版反例：某团队把所有“用户属性 × 商品属性”都交叉进 wide，结果特征量涨了几十倍，训练明显变慢，线上更新周期拉长。后来只保留 top-k 高收益组合，wide 反而更稳定。另一个常见操作是把 deep 的 embedding 维度从 64 降到 16，AUC 可能只掉一点，但线上泛化和稳定性更好。

---

## 替代方案与适用边界

如果只看思想，Wide & Deep 不是唯一方案，而是“显式规则 + 隐式泛化”这一设计路线中的经典起点。

先和纯 Wide、纯 Deep 对比：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Pure Wide | 简单、可解释、稳定 | 泛化弱 | 强规则、低复杂度任务 |
| Pure Deep | 自动学习模式 | 依赖数据量、可控性弱 | 特征丰富、数据大 |
| Wide & Deep | 兼顾记忆与泛化 | 工程复杂度更高 | 推荐、CTR、排序 |
| DeepFM | 无需手工交叉太多 | 解释性进一步下降 | 高维稀疏推荐 |
| xDeepFM | 显式高阶交互更强 | 结构更复杂 | 交互模式复杂场景 |

FM 指 Factorization Machine，白话解释是“用向量内积自动建模两两特征交互”。它的 pairwise 项通常写成：

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
$$

而 Wide 的显式交叉更像是：

$$
\sum_k w_k \phi_k(x)
$$

两者差异在于：

1. Wide 需要你明确告诉模型“哪些特征要交叉”。
2. FM 自动学习两两交互，不必手工列出全部组合。

所以 DeepFM 可以看成“把 Wide 侧的显式交叉，换成 FM 的自动低阶交互”，然后再配一个 Deep 分支。它通常更省人工特征工程，但也更依赖数据和训练稳定性。

可以用简化流程理解几种架构：

| 架构 | 流程 |
|---|---|
| Wide & Deep | 显式交叉 -> Wide；embedding -> MLP -> Deep；两路求和 |
| DeepFM | FM 自动二阶交互 + Deep 共享 embedding |
| xDeepFM | 显式高阶交互模块 + Deep |
| 纯 Deep | embedding -> MLP -> 输出 |

适用边界可以概括为：

1. 如果业务强依赖可解释规则，Wide & Deep 比纯 Deep 更容易控。
2. 如果人工交叉维护成本太高，DeepFM 往往更划算。
3. 如果交互模式很复杂、二阶交互不够，xDeepFM 一类结构更强。
4. 如果任务已经被序列建模主导，例如短视频兴趣流，Wide & Deep 往往只是基础层，而不是主模型。

---

## 参考资料

| 资料 | 年份 | 类型 | 用途 |
|---|---:|---|---|
| Google: Wide & Deep Learning for Recommender Systems | 2016 | 论文/官方出版页 | 原始架构、Google Play 生产应用 |
| ResearchGate 镜像条目 | 2016 | 论文索引 | 公式与结构说明 |
| Emergent Mind: Wide and Deep Learning | 近年整理 | 教程/综述 | 联合训练、优缺点、工程问题 |
| TensorFlow Wide & Deep 教程 | 持续更新 | 官方教程 | 工程实现思路 |

1. Google Research, *Wide & Deep Learning for Recommender Systems*，2016。核心价值是提出联合训练的 Wide 与 Deep 架构，并报告了 Google Play 的生产效果。  
2. Heng-Tze Cheng 等人的论文镜像与索引页面，可用于核对公式、wide 交叉项定义和整体结构。  
3. Emergent Mind 的专题整理，适合补充“为什么要同时做记忆与泛化”以及常见工程坑。  
4. TensorFlow 官方 Wide & Deep 教程，适合对照具体实现方式，包括 wide 特征列、embedding 列和联合训练流程。

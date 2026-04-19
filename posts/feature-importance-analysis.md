## 核心结论

特征重要性，是把模型输出或指标变化分摊到各个特征上，用来回答“当前模型为什么这么判”。

它不回答“现实世界里这个特征是否天然重要”。在推荐系统里，这个区别很关键。模型可能高度依赖 `hour`，不是因为“小时”这个概念在业务上一定最重要，而是因为当前训练数据、样本分布、特征工程和模型结构让它在某些场景下更有解释力。例如同一个 `hour` 特征，在夜间流量里可能比白天更重要，因为夜间用户行为更集中，模型更容易从时间段中读出点击概率差异。

| 对比项 | 现实重要性 | 模型重要性 |
|---|---:|---:|
| 解释对象 | 业务事实或因果关系 | 当前模型的预测逻辑 |
| 典型问题 | “这个因素是否真的影响用户点击？” | “模型是否依赖这个特征做判断？” |
| 依赖条件 | 实验设计、因果识别、业务机制 | 数据、特征、模型、训练目标 |
| 推荐系统风险 | 把相关性误当因果 | 把日志泄漏误当有效信号 |

三类常见方法可以先用公式建立直觉：

参数法看模型参数大小：

$$
I_j = |\beta_j|
$$

置换法看打乱特征后的指标下降：

$$
I_j = S(D) - \frac{1}{K}\sum_{k=1}^{K} S(\tilde D_{k,j})
$$

SHAP 看单个样本预测值如何被特征分解：

$$
f(x)=\phi_0+\sum_{j=1}^{d}\phi_j
$$

推荐系统里不能只盯着单个维度。`item_id`、`user_id`、`category` 这类特征通常是高维稀疏特征，高维稀疏特征是指大多数样本只命中极少数取值的特征表示，例如 one-hot 编码后的商品 ID。实际分析时通常要同时看单特征、特征组和交互项。

---

## 问题定义与边界

先统一记号。设模型输入为 \(x=(x_1,x_2,\dots,x_d)\)，其中 \(d\) 是特征维度数量；模型输出为 \(f(x)\)；验证集为 \(D\)；验证集指标为 \(S(D)\)，例如 AUC、LogLoss 的负数、NDCG 或 Recall。

特征重要性分析要先说明三个问题。

第一，解释的是单样本还是全局。单样本解释回答“这一次为什么给这个用户推荐这个商品”。全局解释回答“模型整体最依赖哪些特征”。一个用户样本里 `item_id` 的 SHAP 值很大，只能说明这次预测里 `item_id` 贡献大，不能推出 `item_id` 字段在全局一定最重要。

第二，解释的是输出值还是指标。CTR 模型通常输出点击概率，也可能先输出 logit。logit 是概率经过对数几率变换后的模型原始打分，常见形式是 \(\log\frac{p}{1-p}\)。在 raw/logit 空间里解释更接近模型内部加法结构；在概率空间里解释更容易给业务方看，但非线性映射会压缩或放大局部变化。

第三，解释的是相关性还是因果。因果效应是指控制其他因素后，主动改变某个因素会导致结果如何变化。特征重要性不是因果效应。对于 CTR 模型，`exposure_count` 的重要性可能来自真实业务信号，也可能来自日志泄漏。日志泄漏是指训练特征中混入了预测时不应该知道的信息，例如用推荐后才产生的曝光统计去预测点击。两者都能让模型指标变高，但解释含义完全不同。

| 分析类型 | 回答的问题 | 常用方法 | 边界 |
|---|---|---|---|
| 单样本解释 | 某一次预测为什么高或低 | SHAP、梯度归因、局部遮蔽 | 不代表全局规律 |
| 全局解释 | 模型整体依赖哪些特征 | 置换重要性、平均 SHAP、参数法 | 受验证集分布影响 |
| 特征交互解释 | 哪些特征组合共同起作用 | TreeSHAP interaction、二阶交叉分析 | 成本高，解释口径要固定 |
| 因果解释 | 改变某个因素是否会改变结果 | A/B 实验、因果推断 | 不属于普通特征重要性 |

本文讨论的是模型解释，不讨论因果识别，也不把重要性直接等同于业务价值。一个特征对模型重要，可能只是因为它记录了错误的数据路径；一个特征业务上重要，也可能因为当前数据质量差而没有被模型用好。

---

## 核心机制与推导

参数法、置换法、SHAP 对应三种不同视角：参数大小、指标下降、边际贡献分解。

先看一个玩具例子。设线性打分模型为：

$$
f(x)=0.2+0.5x_1-0.1x_2
$$

输入 \(x=(2,3)\)，则：

$$
f(x)=0.2+0.5\times 2-0.1\times 3=0.9
$$

如果使用参数法，特征 \(x_1\) 的重要性为 \(|0.5|\)，特征 \(x_2\) 的重要性为 \(|-0.1|\)。因为 \(|0.5|>|-0.1|\)，所以在参数口径下 \(x_1\) 更重要。这个结论有前提：特征必须已经标准化。标准化是把不同量纲的特征变成可比较尺度，例如减去均值再除以标准差。否则 `user_age` 的系数和 `price` 的系数不能直接比较。

参数法公式是：

$$
I_j = |\beta_j|
$$

它适合线性模型、广义线性模型、FM 等参数有明确含义的模型。FM 是 Factorization Machine，即因子分解机，用低维向量表示特征并建模二阶交互。

置换法不看模型内部结构，只看“把某个特征打乱后指标掉多少”。设原始验证集指标为 \(S(D)\)，第 \(k\) 次把第 \(j\) 个特征打乱后的数据为 \(\tilde D_{k,j}\)，重复 \(K\) 次，则：

$$
I_j = S(D) - \frac{1}{K}\sum_{k=1}^{K} S(\tilde D_{k,j})
$$

例如原始 AUC 是 0.812，把 \(x_1\) 打乱后平均 AUC 降到 0.781，则置换重要性为 \(0.031\)。AUC 是衡量排序质量的指标，可以粗略理解为“随机抽一个正样本和负样本，模型把正样本排在前面的概率”。

SHAP 使用 Shapley value 思想，把预测值分解为基线值和各特征贡献：

$$
f(x)=\phi_0+\sum_{j=1}^{d}\phi_j
$$

其中 \(\phi_0\) 是基线输出，\(\phi_j\) 是第 \(j\) 个特征的贡献。如果上面的线性模型以 0.2 为基线，则可写成 \(\phi_0=0.2,\phi_1=1.0,\phi_2=-0.3\)。这里的贡献刚好等于各项加法贡献；在复杂模型里，SHAP 会通过不同特征子集上的边际贡献加权平均得到近似或精确结果。

TreeSHAP 是树模型上的 SHAP 高效算法。它还能扩展到交互解释，给出 \(\phi_{ij}\)。对角项 \(\phi_{ii}\) 表示主效应，非对角项 \(\phi_{ij}\) 表示两个特征之间的交互效应。交互效应是指两个特征一起出现时产生的影响，不等于它们单独影响的简单相加。

推荐系统中，高维稀疏特征通常要按 field 聚合。field 是一组语义相同的特征集合，例如 `item_id` 是一个 field，one-hot 后可能展开成几十万维。常见聚合口径是：

$$
I_{\text{field}}=\sum_{j\in \text{field}} |\phi_j|
$$

这不是唯一数学定义，而是一种工程上可读的聚合方式。否则逐维看 `item_id=938271` 的重要性，对模型排查和业务沟通都很难用。

| 方法 | 输入 | 输出 | 优点 | 局限 | 适用模型 |
|---|---|---|---|---|---|
| 参数法 | 训练后参数 | 特征系数大小 | 快、稳定、直观 | 依赖标准化和模型形式 | 线性模型、GLM、FM |
| 置换法 | 模型、验证集、指标 | 指标下降量 | 模型无关，贴近离线指标 | 高相关特征下会失真，计算较慢 | 几乎所有模型 |
| SHAP | 模型、样本 | 单样本或全局贡献 | 可解释单次预测，可聚合 | 成本高，依赖背景分布假设 | 树模型、部分深度模型 |
| 交互 SHAP | 树模型、样本 | 主效应与交互效应 | 能分析特征组合 | 结果量大，需要筛选 top-k | GBDT、XGBoost、LightGBM |

---

## 代码实现

工程实现可以分成两条路径：训练后解释和离线评估。训练后解释通常指 SHAP、参数法、交互分析；离线评估通常指置换重要性，因为它需要反复在验证集上计算指标。

真实工程例子：一个 CTR 排序模型使用 `user_id`、`item_id`、`category`、`hour`、`device`、`exposure_count`。先在验证集上计算 `hour` 的 permutation importance；再对树模型计算 TreeSHAP；最后按 field 汇总，观察 `hour`、`device` 的交互强度。如果 `exposure_count` 一被打乱 AUC 大幅下降，就要进一步排查它是否来自预测时不可用的曝光日志。

下面是一个可运行的最小实现，用纯 Python 演示置换重要性的核心逻辑。这里用准确率代替 AUC，目的是让代码不依赖第三方库。

```python
from copy import deepcopy

def score_accuracy(rows):
    # 一个玩具 CTR 规则：夜间且移动端更容易预测为点击
    correct = 0
    for r in rows:
        pred = 1 if (r["hour"] >= 20 and r["device"] == "mobile") else 0
        correct += int(pred == r["clicked"])
    return correct / len(rows)

def permute_feature(rows, feature):
    new_rows = deepcopy(rows)
    values = [r[feature] for r in new_rows]
    values = values[1:] + values[:1]
    for r, v in zip(new_rows, values):
        r[feature] = v
    return new_rows

def permutation_importance(rows, feature):
    base = score_accuracy(rows)
    shuffled = score_accuracy(permute_feature(rows, feature))
    return base - shuffled

valid_data = [
    {"hour": 21, "device": "mobile", "clicked": 1},
    {"hour": 22, "device": "mobile", "clicked": 1},
    {"hour": 10, "device": "desktop", "clicked": 0},
    {"hour": 14, "device": "desktop", "clicked": 0},
]

hour_importance = permutation_importance(valid_data, "hour")
device_importance = permutation_importance(valid_data, "device")

assert score_accuracy(valid_data) == 1.0
assert hour_importance >= 0
assert device_importance >= 0
print({"hour": hour_importance, "device": device_importance})
```

在真实项目中，代码骨架通常长这样：

```python
def load_model(path):
    """加载训练好的 CTR 排序模型。"""
    raise NotImplementedError

def load_valid_data(path):
    """加载验证集，返回 X_valid, y_valid 和字段映射。"""
    raise NotImplementedError

def compute_permutation_importance(model, X_valid, y_valid, scoring="roc_auc"):
    from sklearn.inspection import permutation_importance
    return permutation_importance(
        model,
        X_valid,
        y_valid,
        scoring=scoring,
        n_repeats=5,
        random_state=42,
    )

def compute_shap_values(model, X_valid):
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_valid)

def aggregate_by_field(feature_importance, feature_to_field):
    field_scores = {}
    for feature, value in feature_importance.items():
        field = feature_to_field[feature]
        field_scores[field] = field_scores.get(field, 0.0) + abs(value)
    return field_scores

def plot_top_features(result_table, top_k=20):
    top_rows = sorted(result_table, key=lambda r: r["importance"], reverse=True)[:top_k]
    for row in top_rows:
        print(row["rank"], row["feature"], row["field"], row["importance"])
```

输出表建议统一成下面的结构，方便下游做报表、报警和版本对比。

| feature | field | importance | importance_type | rank |
|---|---|---:|---|---:|
| hour | hour | 0.031 | permutation_auc_drop | 1 |
| exposure_count | exposure_count | 0.028 | permutation_auc_drop | 2 |
| item_id_938271 | item_id | 0.017 | mean_abs_shap | 3 |
| device_mobile | device | 0.011 | mean_abs_shap | 4 |

新手读结果时可以先按这个顺序判断：

| 结果现象 | 优先解释 | 下一步 |
|---|---|---|
| 置换重要性高，SHAP 也高 | 模型稳定依赖该特征 | 检查业务合理性 |
| 置换重要性高，但业务上不该高 | 可能有泄漏或分布偏差 | 查特征生成时间线 |
| 单样本 SHAP 高，全局均值不高 | 局部场景特征 | 分人群、分场景复查 |
| field 重要性高，但单维分散 | 高维稀疏特征组有效 | 做 field 级汇总展示 |

---

## 工程权衡与常见坑

不同方法的结果不能机械互比。参数法解释的是参数大小，置换法解释的是指标下降，SHAP 解释的是预测值分解。三者可以互相验证，但不是同一个单位。

| 常见坑 | 错误做法 | 后果 | 规避方式 |
|---|---|---|---|
| 原始系数不可直接比较 | 直接比较 `user_age` 和 `price` 的系数 | 量纲影响结论 | 先标准化再看系数 |
| 训练集上做置换 | 在训练集计算指标下降 | 过拟合特征被高估 | 只在验证集或测试集做 |
| 高维稀疏特征逐维解释 | 逐个解释几十万维 `item_id` | 结果不可读，成本高 | 优先按 field 聚合 |
| 强相关特征下 SHAP 不稳定 | 把 `item_id` 与 `category` 的归因当绝对结论 | 贡献在相关特征间分摊变化 | 说明依赖假设，做分组验证 |
| 重要性等于因果 | 看到重要就认为能驱动点击 | 错误业务决策 | 用实验验证因果 |

几个工程规则要固定下来。

第一，先标准化再看系数。线性模型里，`price` 的单位可能是元，`user_age` 的单位是岁，原始系数大小不具备可比性。

第二，只在验证集或测试集做置换。训练集已经被模型见过，指标变化会混入过拟合影响，不能代表上线后的泛化行为。

第三，优先按 field 聚合。推荐模型里的 `user_id`、`item_id`、`query_id` 可能展开成海量维度，单维重要性很难直接指导工程动作。

第四，先看 raw/logit，再映射到概率展示。很多模型内部是加法打分，logit 空间的贡献更容易相加；概率空间更直观，但容易让贡献看起来非线性。

排查日志泄漏时，特征重要性非常有工程价值。可以按三步走：先看置换重要性中异常靠前的特征，尤其是 `exposure_count`、`click_count_1h`、`rank_position`、`is_exposed` 这类和日志链路相关的字段；再检查特征生成时间是否早于预测时间；最后做时间切分验证，如果某个特征在随机切分下很强、在按时间切分下明显变弱，就要怀疑它依赖了未来信息或短期缓存状态。

---

## 替代方案与适用边界

没有一种方法能覆盖所有模型和所有业务问题。方法选择应该由模型类型和解释目标决定。

| 方法 | 适合什么模型 | 适合什么目标 | 不适合什么场景 |
|---|---|---|---|
| 参数法 | 线性模型、逻辑回归、FM | 快速解释全局方向和强弱 | 深度模型、未标准化特征、强共线特征 |
| 置换法 | 任意可预测模型 | 找可疑特征、做特征筛选、贴近离线指标 | 特征强相关、计算预算很紧 |
| SHAP | 树模型、部分深度模型 | 解释单次推荐、做 field 级贡献分析 | 超大规模深度模型、实时解释链路 |
| 梯度归因 | 深度模型 | 看输入变化对输出的局部影响 | 离散 ID 特征、梯度饱和场景 |
| 遮蔽法 | 任意模型 | 模拟删除某类特征后的影响 | 特征组合太多，成本高 |

如果模型是线性模型或 FM，优先用参数法。它解释直接、成本低，也方便排查符号是否符合常识。例如 `price` 系数为正还是负，`is_new_user` 是否符合预期，都可以快速检查。

如果模型是 GBDT、XGBoost 或 LightGBM，优先用 TreeSHAP 或 TreeExplainer。树模型天然包含分裂路径和非线性交互，TreeSHAP 比普通采样 SHAP 更适合工程使用。

如果模型是深度推荐模型，可以用 SHAP、梯度归因、集成遮蔽法，但要接受稳定性和成本问题。深度模型里的 embedding、attention、交叉网络层会让解释对象变得复杂。embedding 是把离散 ID 映射成低维向量的表示方法；它能提升模型效果，但不容易直接对应到人能读懂的字段解释。

按目标选择更直接：

如果目标是“找可疑泄漏特征”，优先置换法，因为它直接看特征被破坏后离线指标掉多少。

如果目标是“解释单次推荐结果”，优先 SHAP，因为它能把某个样本的预测拆成特征贡献。

如果目标是“做特征筛选”，可以用置换法结合稳定性统计。稳定性统计是指在不同时间窗口、不同随机种子、不同样本切片上重复计算重要性，只有持续靠前的特征才进入筛选候选。

不要直接只看 attention 权重。attention 是模型内部用于加权信息的机制，但 attention 权重不一定等于最终预测贡献。后续层、残差连接、归一化和非线性变换都可能改变最终输出。它可以作为诊断信号，但不是通用的重要性指标。

---

## 参考资料

上述参数法、置换法和 SHAP 公式分别来自线性模型解释、置换特征重要性和 Shapley value 分解思想。field 聚合是推荐系统工程中的常用解释口径，用于把高维稀疏维度合并成可读的字段级结果。

1. [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
2. [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
3. [scikit-learn: Permutation feature importance](https://scikit-learn.org/1.5/modules/permutation_importance.html)
4. [scikit-learn: Common pitfalls in the interpretation of coefficients of linear models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html)
5. [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://www.microsoft.com/en-us/research/publication/xdeepfm-combining-explicit-and-implicit-feature-interactions-for-recommender-systems/)

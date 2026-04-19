## 核心结论

NDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）是排序评估指标：它把一个排序结果的质量压到 `0~1` 附近，用来衡量“相关结果是否排在前面”。

它的核心不是只看相关结果有没有出现，而是看相关结果出现在什么位置。第 1 位命中的价值通常高于第 10 位命中，因为用户更容易看到前面的结果。

NDCG 由三部分组成：

| 组成 | 作用 | 白话解释 |
|---|---|---|
| DCG | 计算当前排序得分 | 相关性越高、位置越靠前，贡献越大 |
| IDCG | 计算理想排序得分 | 把所有结果按真实相关性从高到低排，得到理论上限 |
| NDCG | 归一化 | 用当前得分除以理想得分，让不同 query 可比较 |

公式是：

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

其中 `@k` 表示只看前 `k` 个位置。例如搜索页通常只关心前 10 条，就会看 `NDCG@10`。

位置衰减来自权重：

$$
\frac{1}{\log_2(i+1)}
$$

`i` 是排序位置，从 1 开始。位置越靠后，权重越小。

| 位置 i | 权重 $1 / \log_2(i+1)$ | 含义 |
|---:|---:|---|
| 1 | 1.000 | 第 1 位完整计分 |
| 2 | 0.631 | 第 2 位开始明显折损 |
| 3 | 0.500 | 第 3 位只剩一半贡献 |
| 4 | 0.431 | 后排继续下降 |
| 10 | 0.289 | 第 10 位贡献更低 |

玩具例子：同样命中 2 个相关结果，`[相关, 相关, 无关]` 比 `[相关, 无关, 相关]` 更好。后者虽然也找到了第二个相关项，但它被放到第 3 位，贡献被位置衰减压低。

---

## 问题定义与边界

在使用 NDCG 前，必须先定义“相关性”。相关性是指一个结果对当前 query 是否有用，可以是二值标注，也可以是多级标注。

二值相关性只区分相关和无关：

| label | 含义 |
|---:|---|
| 0 | 无关 |
| 1 | 相关 |

多级相关性会表达强弱：

| label | 含义 |
|---:|---|
| 0 | 无关 |
| 1 | 略相关 |
| 2 | 相关 |
| 3 | 高度相关 |

统一记号如下：

| 记号 | 含义 |
|---|---|
| $rel_i$ | 第 `i` 个位置结果的真实相关性 |
| $g_i$ | 第 `i` 个位置的增益，由相关性转换得到 |
| $k$ | 只评估前 `k` 个结果 |
| gain | 增益函数，把相关性标签转换成分数 |

常见 gain 有两种：

$$
g_i = rel_i
$$

或：

$$
g_i = 2^{rel_i} - 1
$$

第一种更直观，相关性是多少就加多少分。第二种会放大高相关结果的价值，例如 `rel=3` 会变成 `7`，适合业务上非常重视高质量结果的场景。

NDCG 适合评估顶部展示质量，不适合单独衡量“有没有找全”。搜索场景里，用户通常只看前 10 条，所以 `NDCG@10` 往往比 `NDCG@100` 更接近真实体验。但如果目标是覆盖尽可能多的相关结果，只看 NDCG 不够，还需要 Recall 等指标。

| 场景 | 适合看 NDCG | 原因 |
|---|---:|---|
| 搜索结果前 10 条质量 | 是 | 用户主要看顶部结果 |
| 推荐首页曝光位排序 | 是 | 曝光位有限，前排更重要 |
| 判断候选集中相关项是否全部召回 | 否 | NDCG 不直接衡量找全程度 |
| 只关心第一个结果是否正确 | 不一定 | MRR 或 Hit@1 可能更直接 |

边界条件也要提前约定：如果某个 query 没有任何相关结果，那么理想排序的 `IDCG=0`。此时 `NDCG = DCG / IDCG` 无法正常计算。工程上常见处理方式是返回 `0.0`，或者在整体评估时跳过这类 query。两种方式都可以，但必须在项目内固定。

---

## 核心机制与推导

NDCG 的计算顺序是：先算每个位置的 gain，再按位置折损得到 DCG，然后用理想排序的 DCG 做归一化。

完整公式如下：

$$
DCG@k = \sum_{i=1}^{k} \frac{g_i}{\log_2(i+1)}
$$

这里：

| 符号 | 解释 |
|---|---|
| $i$ | 排序位置，从 1 开始 |
| $k$ | 只看前 `k` 个结果 |
| $g_i$ | 第 `i` 位结果的增益 |
| $\log_2(i+1)$ | 位置折损分母 |

`DCG` 的含义是：每个位置的结果先根据相关性拿到 gain，然后根据位置打折。位置越靠前，折损越小；位置越靠后，折损越大。

理想排序的分数是：

$$
IDCG@k = \text{按真实相关性降序排列后的 } DCG@k
$$

最终：

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

玩具例子取二值相关性，`k=3`。真实最优排序是：

```text
[1, 1, 0]
```

模型输出排序是：

```text
[1, 0, 1]
```

使用 `g_i = rel_i`，模型排序的 DCG 是：

$$
DCG@3 = \frac{1}{\log_2(2)} + \frac{0}{\log_2(3)} + \frac{1}{\log_2(4)}
$$

$$
DCG@3 = 1 + 0 + 0.5 = 1.5
$$

理想排序的 IDCG 是：

$$
IDCG@3 = \frac{1}{\log_2(2)} + \frac{1}{\log_2(3)} + \frac{0}{\log_2(4)}
$$

$$
IDCG@3 \approx 1 + 0.631 + 0 = 1.631
$$

所以：

$$
NDCG@3 = \frac{1.5}{1.631} \approx 0.919
$$

结论很直接：第二个相关结果虽然出现了，但被放在第 3 位，所以它只拿到 `0.5` 的位置权重。如果模型把它放到第 2 位，分数会更高。

真实工程例子是推荐系统首页排序。假设每个用户请求会产生 200 个候选商品，模型给每个商品打一个排序分，页面只展示前 10 个。人工标注或日志反馈把商品相关性标成 `0/1/2/3`。此时离线评估常看 `NDCG@10`，因为它直接对应前 10 个曝光位的排序质量。第 1 位放错高相关商品，比第 80 位放错高相关商品更影响用户体验。

---

## 代码实现

实现 NDCG 时，要先统一输入格式，再统一 gain 公式，再统一 `k` 的截断方式。否则不同实现之间的数值不可比。

常见输入如下：

| 字段 | 含义 | 示例 |
|---|---|---|
| `query_id` | 一次搜索或推荐请求的 ID | `q_001` |
| `doc_id` | 被排序的文档、商品或内容 ID | `d_1024` |
| `label` | 真实相关性 | `0/1/2/3` |
| `score` | 模型预测分数 | `0.87` |
| `k` | 评估前几个结果 | `10` |

下面是一个最小可运行 Python 实现。输入是同一个 query 下每个结果的真实标签 `y_true` 和模型分数 `y_score`，输出 `NDCG@k`。

```python
import math

def gain(rel, method="linear"):
    if method == "linear":
        return rel
    if method == "exp2":
        return 2 ** rel - 1
    raise ValueError(f"unknown gain method: {method}")

def dcg(rels, k, gain_method="linear"):
    total = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        total += gain(rel, gain_method) / math.log2(i + 1)
    return total

def ndcg_score(y_true, y_score, k, gain_method="linear"):
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if k <= 0:
        return 0.0

    paired = list(zip(y_true, y_score))

    # 模型排序：score 越大越靠前
    ranked_by_model = [
        rel for rel, _ in sorted(paired, key=lambda x: x[1], reverse=True)
    ]

    # 理想排序：真实相关性越大越靠前
    ranked_ideal = sorted(y_true, reverse=True)

    dcg_val = dcg(ranked_by_model, k, gain_method)
    idcg_val = dcg(ranked_ideal, k, gain_method)

    return 0.0 if idcg_val == 0 else dcg_val / idcg_val

# 玩具例子：模型排序后的真实相关性等价于 [1, 0, 1]
y_true = [1, 1, 0]
y_score = [0.9, 0.1, 0.8]
value = ndcg_score(y_true, y_score, k=3)

assert round(value, 3) == 0.920
assert ndcg_score([0, 0, 0], [0.3, 0.2, 0.1], k=3) == 0.0
assert ndcg_score([3, 2, 0], [0.9, 0.8, 0.1], k=3, gain_method="exp2") == 1.0

print(value)
```

这段代码有三个关键点。

第一，`ranked_by_model` 是按模型分数排序后的真实相关性序列。NDCG 不是直接对模型分数求和，而是看模型排出来的位置上放了什么真实标签。

第二，`ranked_ideal` 是把真实标签从高到低排序，用来计算 `IDCG`。这里不能按模型分数排，否则理想上限就错了。

第三，`gain_method` 必须固定。`linear` 和 `exp2` 都是合法做法，但同一项目不能混用。否则某次实验的 `NDCG@10=0.82` 和另一次实验的 `NDCG@10=0.84` 可能没有可比性。

---

## 工程权衡与常见坑

NDCG 不是一个可以脱离上下文解释的绝对指标。它适合在同一任务、同一标注体系、同一 `k`、同一 gain 定义下比较模型。

例如同一个搜索项目里，模型 A 的 `NDCG@10=0.75`，模型 B 的 `NDCG@10=0.78`，通常可以认为 B 的顶部排序质量更好。但如果 A 用 `gain=rel`，B 用 `gain=2^rel-1`，这个比较就不成立。

常见坑如下：

| 坑 | 问题 | 规避做法 |
|---|---|---|
| 混用 DCG 和 NDCG | DCG 没有归一化，不同 query 难比较 | 跨 query 汇总时使用 NDCG |
| gain 定义不统一 | `rel` 和 `2^rel-1` 数值尺度不同 | 在评估配置中显式记录 gain 方法 |
| `k` 与业务目标不一致 | 指标优化方向偏离真实体验 | 首页看小 `k`，长列表看更大的 `k` |
| `IDCG=0` 未处理 | 出现除零或 NaN | 固定返回 `0.0` 或跳过该 query |
| ties 排序规则不一致 | 相同 score 的结果顺序影响分数 | 固定二级排序键或使用同一评估库 |
| 多 query 直接合并计算 | 热门 query 或长列表 query 权重异常 | 通常先按 query 算 NDCG，再求平均 |
| 离线标签过旧 | 指标反映旧偏好 | 定期刷新标注或结合在线实验 |

真实工程中，最大的问题通常不是公式写错，而是评估口径漂移。训练代码用一种 NDCG，离线评估平台用另一种 NDCG，线上报表又做了不同的 query 过滤，最后所有人都在看“NDCG”，但数值不是同一个东西。

更稳妥的做法是把评估口径写成配置：

| 配置项 | 推荐做法 |
|---|---|
| `k` | 明确写成 `10`、`20` 等固定值 |
| gain | 明确写成 `linear` 或 `exp2` |
| query 过滤 | 记录是否过滤无正例 query |
| ties | 固定按 `score desc, doc_id asc` 等规则排序 |
| 聚合方式 | 先 query 内计算，再对 query 求均值 |
| 版本 | 评估代码和指标配置一起版本化 |

NDCG 越高通常越好，但只能说明“按当前标注和当前位置衰减函数看，排序更接近理想排序”。它不能保证点击率一定提升，也不能保证业务收入一定提升。离线 NDCG 应该用于模型筛选和回归检查，最终仍要结合在线 A/B 实验验证。

---

## 替代方案与适用边界

如果目标是看“相关结果是否排在前面”，NDCG 很合适。如果目标是看“命中了多少相关项”，还需要 Recall、Precision、MAP、MRR 等指标配合。

| 指标 | 主要回答的问题 | 适用场景 |
|---|---|---|
| NDCG | 排序质量好不好，尤其是前排 | 搜索、推荐、广告排序 |
| MAP | 多个相关项的平均排序质量 | 信息检索、文档搜索 |
| Recall@k | 前 `k` 个结果找回了多少相关项 | 召回系统、候选生成 |
| Precision@k | 前 `k` 个结果里相关项比例多高 | 结果列表质量控制 |
| MRR | 第一个相关项出现得多快 | 问答、导航型搜索、只关心首个正确结果 |

当业务只关心第 1 个结果时，MRR 可能比 NDCG 更直接。例如用户输入一个明确问题，只需要第一个答案正确，那么“第一个相关结果的位置”比整个前 10 的排序质量更重要。

当业务关心候选集覆盖时，Recall@k 更直接。例如推荐系统的召回层需要从百万商品中找出尽可能多的潜在相关商品，此时只看 NDCG 可能会掩盖召回不足的问题。召回层常用 Recall@100、Recall@500；排序层再看 NDCG@10、NDCG@20。

NDCG 的另一个边界是它默认相关性标注可信，并且默认对数位置衰减符合业务。如果用户行为不是简单的“越靠前越重要”，例如瀑布流、短视频连续滑动、多列商品卡片、广告混排，那么固定的 `1/log2(i+1)` 可能不够准确。此时可以使用基于真实曝光和点击行为的指标，或者使用更贴近业务目标的学习目标和在线实验。

一个实际推荐系统可以这样组合指标：

| 模块 | 主要指标 | 原因 |
|---|---|---|
| 召回层 | Recall@100、Recall@500 | 先保证相关候选能进入后续排序 |
| 粗排层 | NDCG@50、AUC | 评估大规模候选的初步排序 |
| 精排层 | NDCG@10、Precision@10 | 对应最终曝光位置 |
| 在线实验 | CTR、CVR、停留时长、收入 | 验证真实用户行为和业务收益 |

因此，NDCG 是排序质量指标，不是完整评估体系。它在顶部展示质量评估中非常有用，但不应该替代召回、精度、业务指标和在线实验。

---

## 参考资料

1. [Järvelin & Kekäläinen, 2002, Cumulated Gain-based Evaluation of IR Techniques](https://doi.org/10.1145/582415.582418)
2. [scikit-learn: ndcg_score](https://sklearn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
3. [Wang et al., 2013, A Theoretical Analysis of NDCG Type Ranking Measures](https://proceedings.mlr.press/v30/Wang13.html)
4. [TensorFlow Model Analysis: tfma.metrics.NDCG](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/metrics/NDCG)

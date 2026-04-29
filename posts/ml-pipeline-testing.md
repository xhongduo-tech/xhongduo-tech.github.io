## 核心结论

ML Pipeline 测试的本质，不是证明“模型懂业务”，而是把数据契约、训练与服务一致性、模型质量阈值做成发布门禁。更准确地说，它是在回答三个问题：

1. 输入数据是否合法。
2. 线上数据是否已经偏离训练时的统计假设。
3. 新模型是否在整体和关键切片上都没有退化。

可以把它压缩成一句话：

`ML Pipeline 测试 = 数据门禁 + 模型门禁 + 漂移/倾斜检测`

这里的“门禁”就是一组可执行规则，意思是模型想上线，必须先通过这些检查。

传统软件测试和 ML 测试的关注点不同。传统测试主要验证“代码逻辑有没有写错”；ML 测试主要验证“数据和模型是否仍处在可接受边界内”。

| 维度 | 传统软件测试 | ML Pipeline 测试 |
|---|---|---|
| 核心对象 | 代码逻辑 | 数据、特征、模型行为 |
| 典型失败原因 | 条件分支错误、状态处理错误 | 数据漂移、特征倾斜、切片退化 |
| 主要问题 | 输出是否符合预期 | 输入是否越界、模型是否变坏 |
| 拦截对象 | bug 代码 | 脏数据、坏模型、线上线下不一致 |
| 常见工具思路 | 单测、集成测试、端到端测试 | schema 校验、drift 检测、切片评估、基线比较 |

玩具例子很简单。传统单测像检查“加法函数是不是把 `2+3` 算成 `5`”；ML 测试像检查“这批 `age` 和 `payment_type` 还像不像训练时见过的数据，以及新模型在 `new_user` 这类关键人群上有没有变差”。

真实工程里，很多事故并不是模型代码突然坏了，而是数据源升级、埋点改名、特征字典变化、线上出现训练时没见过的新枚举值。这也是为什么 ML 测试要把重点放在数据和分布，而不是只盯着代码。

---

## 问题定义与边界

先定义边界。ML 系统的主要风险，通常来自输入分布和特征实现，而不是某一行 Python 代码本身。因为模型是从历史数据中学习统计规律，一旦输入数据改变，模型即使代码完全没变，也可能立刻失效。

术语先解释：

- “schema”是数据结构约定，白话说就是字段名、类型、是否可空、取值范围这些基本规则。
- “drift”是分布漂移，白话说就是训练期和服务期的数据统计形状不一样了。
- “skew”是训练/服务倾斜，白话说就是同一个特征在线上和离线算出来不一致。

可以用两个基本约束描述它：

- 数据约束：$x_j \in S_j$
- 分布约束：$d(P_{train}^j, P_{serving}^j) \le \tau_j$

其中：

- $x_j$ 是第 $j$ 个特征的观测值。
- $S_j$ 是该特征允许的合法集合，比如范围、枚举或格式。
- $P_{train}^j$ 是训练集上第 $j$ 个特征的分布。
- $P_{serving}^j$ 是线上服务期该特征的分布。
- $d(\cdot,\cdot)$ 是分布距离函数，白话说就是衡量两个分布差了多少。
- $\tau_j$ 是阈值，超过就认为风险过高。

一个最小例子：

- 训练时 `payment_type ∈ {cash, card}`
- 线上突然出现 `apple_pay`

这时即使代码一行没改，模型输入也已经偏离原始假设。问题不在“推理函数报错”，而在“模型从来没学过这种值该怎么解释”。

测试能做的事情和不能做的事情要分开看：

| 类别 | 能测什么 | 不能直接测什么 |
|---|---|---|
| 数据层 | schema、空值、范围、枚举、分布漂移 | 数据是否完整表达了业务语义 |
| 特征层 | 训练/服务一致性、特征缺失、编码变化 | 特征是否一定是最优设计 |
| 模型层 | 整体指标、切片指标、基线回退 | 模型是否真正“理解”因果关系 |
| 系统层 | 回归风险、发布门禁、灰度前检查 | 长期策略收益、业务闭环效果 |

常见误区是把测试能力想得过大。离线测试能证明的是“没有明显越界”，不能证明“模型一定在真实业务里有效”。例如标签本身有系统性错误，或者业务目标定义错了，测试可能全部通过，但上线后依然没有价值。

真实工程例子：支付风控模型训练时 `merchant_region` 只有国内区域码，线上接入跨境支付后开始出现新的区域编码。此时模型并不一定崩溃，但输入语义已经变了。如果没有数据测试，这类问题很容易在 AUC 还没明显下降前就先污染评分结果。

---

## 核心机制与推导

ML Pipeline 测试通常是三层递进关系：

1. 数据门禁：输入是否合法。
2. 漂移/倾斜检测：输入是否仍和训练时一致。
3. 模型门禁：候选模型是否达到质量要求，且关键切片不退化。

这三层的顺序不能反。因为如果输入本身已经错了，再看模型指标往往没有意义。

### 1. 数据门禁

最基础的规则是：

$$
x_j \in S_j
$$

意思是每个特征值都要落在允许集合里。`age ∈ [0,120]`、`payment_type ∈ {cash, card}` 都属于这一类。

玩具例子：

- `age=170`
- `payment_type=apple_pay`

这不是“模型预测不准”，而是“输入先失败”。如果连输入是否合法都不保证，后面的模型评估只是对污染数据做计算。

### 2. 漂移与倾斜

数据合法，不代表分布稳定。训练时 `payment_type` 可能 80% 是 `cash`，20% 是 `card`；线上变成 60% `cash`、35% `card`、5% 新值。即使每条数据都能解析，模型面对的总体环境也变了。

常见约束写成：

$$
d(P_{train}^j, P_{serving}^j) \le \tau_j
$$

如果特征是离散枚举，可以用 $L_\infty$ 距离，也就是每个类别概率差值里的最大值：

$$
d_\infty(P,Q)=\max_i |P_i-Q_i|
$$

例如训练期 `payment_type=[0.8, 0.2, 0.0]`，服务期 `[0.6, 0.35, 0.05]`，则

$$
d_\infty = \max(0.2, 0.15, 0.05)=0.20
$$

如果阈值 $\tau=0.10$，就应直接阻断发布或报警。

这里要注意，drift 和 skew 不完全一样：

| 符号/术语 | 含义 |
|---|---|
| $S_j$ | 第 $j$ 个特征的合法约束集合 |
| $P_{train}^j$ | 训练期第 $j$ 个特征分布 |
| $P_{serving}^j$ | 服务期第 $j$ 个特征分布 |
| $m_c$ | 候选模型指标 |
| $m_b$ | 基线模型指标 |
| $S_k$ | 第 $k$ 个评估切片 |

- drift 关注“同一特征在不同时间分布变了没有”。
- skew 关注“同一时刻离线和线上算出来是不是一回事”。

真实工程里，feature skew 比纯 drift 更危险。比如训练时 `user_age_bucket` 用“当前日期 - 出生日期”计算，线上却缓存了旧逻辑，用了另一套分桶规则。此时即使原始年龄分布没变，模型看到的特征语义已经不一致。

### 3. 模型门禁与切片门禁

模型层通常至少有两个规则：

- 绝对阈值：$m_c(S_k) \ge T_k$
- 相对回退：$m_c(S_k)-m_b(S_k)\ge -\Delta_k$

这里：

- $m$ 是指标，比如 AUC、F1、LogLoss。
- $T_k$ 是某个切片必须达到的最低标准。
- $\Delta_k$ 是允许相对基线退化的最大幅度。

为什么要同时看“绝对值”和“相对回退”？因为只看绝对值，可能把弱退化放过去；只看相对值，又可能把本来就很差的模型接受进来。

更关键的是切片评估。切片就是把数据按业务维度拆开看，白话说是“不要只看总体平均，按关键人群单独算”。常见切片有：

- `new_user / old_user`
- `country`
- `device_type`
- `merchant_category`
- `hour_of_day`

玩具例子：

- 总体 AUC：基线 `0.92`，候选 `0.93`
- `new_user` 切片：基线 `0.89`，候选 `0.81`

如果只看总体，候选模型似乎更好；但对新用户这个关键切片，它已经明显退化。真实系统里，这类局部退化非常常见，因为整体指标会被大流量主群体“稀释”。

所以更合理的发布链路是：

`输入数据 -> schema/异常校验 -> drift/skew 检测 -> 模型整体评估 -> 切片门禁 -> 发布`

这个链路成立的原因很直接：每一层都在排除上一层无法发现的风险。

---

## 代码实现

实现时应明确分成两层：

1. 数据测试层：字段、类型、范围、枚举、漂移、倾斜。
2. 模型测试层：整体指标、切片指标、基线比较。

关键不是“写很多 `assert`”，而是把阈值、切片和失败动作配置化。

先看一个可运行的最小 Python 例子。它不依赖外部库，但已经覆盖数据门禁、漂移检测和模型门禁的核心思路。

```python
from collections import Counter

SCHEMA = {
    "age": {"type": int, "min": 0, "max": 120},
    "payment_type": {"type": str, "enum": {"cash", "card"}},
}

def assert_schema_and_values(batch, schema):
    for row in batch:
        for field, rule in schema.items():
            assert field in row, f"missing field: {field}"
            assert isinstance(row[field], rule["type"]), f"type mismatch: {field}"

            if "min" in rule:
                assert rule["min"] <= row[field] <= rule["max"], f"range error: {field}={row[field]}"
            if "enum" in rule:
                assert row[field] in rule["enum"], f"enum error: {field}={row[field]}"

def categorical_dist(rows, field, categories):
    cnt = Counter(row[field] for row in rows)
    n = len(rows)
    return {c: cnt.get(c, 0) / n for c in categories}

def linf_distance(p, q):
    keys = set(p) | set(q)
    return max(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def auc_from_scores(labels, scores):
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    assert pos and neg, "AUC requires both positive and negative samples"

    win = 0.0
    total = 0
    for sp, _ in pos:
        for sn, _ in neg:
            total += 1
            if sp > sn:
                win += 1.0
            elif sp == sn:
                win += 0.5
    return win / total

train_batch = [
    {"age": 20, "payment_type": "cash"},
    {"age": 35, "payment_type": "cash"},
    {"age": 41, "payment_type": "cash"},
    {"age": 29, "payment_type": "card"},
    {"age": 50, "payment_type": "cash"},
]

serving_batch = [
    {"age": 22, "payment_type": "cash"},
    {"age": 31, "payment_type": "card"},
    {"age": 44, "payment_type": "cash"},
    {"age": 27, "payment_type": "cash"},
    {"age": 39, "payment_type": "card"},
]

assert_schema_and_values(train_batch, SCHEMA)
assert_schema_and_values(serving_batch, SCHEMA)

train_dist = categorical_dist(train_batch, "payment_type", ["cash", "card"])
serving_dist = categorical_dist(serving_batch, "payment_type", ["cash", "card"])
drift = linf_distance(train_dist, serving_dist)
assert drift <= 0.40, f"drift too large: {drift}"

labels = [1, 0, 1, 0, 1, 0]
baseline_scores = [0.95, 0.70, 0.80, 0.40, 0.75, 0.30]
candidate_scores = [0.92, 0.68, 0.82, 0.38, 0.73, 0.35]

auc_baseline = auc_from_scores(labels, baseline_scores)
auc_candidate = auc_from_scores(labels, candidate_scores)

assert auc_candidate >= 0.90
assert auc_candidate - auc_baseline >= -0.05

print("all checks passed")
```

上面这段代码对应的是最小闭环：

- `assert_schema_and_values` 对应数据门禁。
- `linf_distance` 对应离散特征的分布漂移。
- `auc_from_scores` 和后续断言对应模型门禁。

如果写成更接近工程代码的伪结构，通常是这样：

```python
assert_schema(batch, schema)
assert_range(batch["age"], 0, 120)
assert_enum(batch["payment_type"], {"cash", "card"})
```

```python
drift = linf_distance(train_dist["payment_type"], serving_dist["payment_type"])
assert drift <= 0.10
```

```python
assert auc(candidate) >= 0.90
assert auc(candidate, slice="new_user") >= auc(baseline, slice="new_user") - 0.01
```

真实工程例子可以想成一个推荐或风控流水线：

1. 从训练产物里读取 schema 和参考分布。
2. 对待发布批次执行字段、类型、空值、范围、枚举校验。
3. 对核心特征执行 drift/skew 检查。
4. 在验证集和回放样本上计算整体指标。
5. 按 `country × device_type × new_user` 做切片指标。
6. 与生产基线模型比较。
7. 通过则进入 shadow 或 canary，否则回退。

建议把规则做成配置，而不是写死在代码里：

| 检查项 | 阈值示例 | 适用环境 | 失败动作 |
|---|---|---|---|
| `age` 范围 | `[0,120]` | 训练/预发布/线上批次 | 直接失败 |
| `payment_type` 枚举 | `{cash, card}` | 训练/预发布 | 失败并报警 |
| 分类分布漂移 | `L∞ <= 0.10` | 预发布/线上监控 | 阻断或人工审核 |
| 整体 AUC | `>= 0.90` | 训练评估/发布前 | 阻断发布 |
| `new_user` 切片回退 | `>= baseline - 0.01` | 发布前 | 阻断发布 |

如果使用 TFDV 和 TFMA，可以这样理解分层：

- TFDV 更偏“数据验证层”，负责 schema、异常、drift、skew。
- TFMA 更偏“模型验证层”，负责整体指标、切片指标、阈值门禁、候选/基线比较。

---

## 工程权衡与常见坑

ML 测试真正难的地方不在 API，而在权衡。规则太少，拦不住风险；规则太死，又会把正常变化都当事故。

最常见的坑是只看整体指标。总体 AUC 从 `0.92` 升到 `0.93`，看起来更好了；但 `new_user` 切片从 `0.89` 掉到 `0.81`，这在很多业务里已经不可接受。整体指标像平均数，白话说就是“会把局部问题抹平”。

另一个高频坑是时间泄漏。时间泄漏是指训练或验证时不小心看到了未来信息，白话说就是“考试时提前偷看答案”。如果你随机切分训练集和验证集，可能把未来样本分到训练里，导致离线评估非常漂亮，上线后却明显失真。对时间序列、风控、推荐排序，这个问题尤其严重。

下面是常见坑和规避方式：

| 常见坑 | 具体表现 | 风险 | 规避策略 |
|---|---|---|---|
| 只看整体指标 | 总体变好，关键切片变差 | 长尾用户受损被掩盖 | 对关键切片单独设门禁 |
| 随机切分导致时间泄漏 | 验证集掺入未来分布 | 离线指标虚高 | 按时间切分，并做回放验证 |
| schema 定义过死 | 合法新枚举值被当异常 | 运维噪声高，规则失信 | 按环境和版本维护 schema |
| 只做离线不做线上监控 | 发布时通过，线上逐渐漂移 | 问题滞后暴露 | 建立持续监控和报警 |
| 训练和服务特征逻辑不共享 | 同名特征语义不同 | feature skew，线上失真 | 共用一套特征计算路径 |

这里还有一个常被忽略的点：阈值不是越严越好。比如某些稳定业务，分类特征漂移阈值可以很小；但对快速变化的推荐流量，如果阈值设得过低，就会频繁误报，最后团队会绕过测试。工程上更合理的做法是按风险分级：

- 高风险业务：支付、授信、风控。强门禁，必须保留切片和基线比较。
- 中风险业务：搜索排序、推荐曝光。门禁加灰度。
- 低风险业务：实验性原型、内部分析模型。基础 schema 和整体指标即可。

完整工程链路通常不是“离线测试通过就上线”，而是：

`离线门禁 -> shadow -> canary -> 线上监控`

- shadow 是影子流量，白话说就是线上真实请求只拿来比对，不真正影响用户结果。
- canary 是灰度发布，白话说就是先放一小部分流量，确认没问题再扩大。

这条链路的重要性在于：离线测试只能看到历史样本，真正的分布变化往往要在接近线上时才能暴露。

---

## 替代方案与适用边界

ML 测试不是单独的万能方案，它只是保障链中的一段。更完整的视角是：离线测试负责拦截明显越界，shadow/canary 负责验证上线前行为，在线监控负责发现长期漂移和效果衰减。

不同方案覆盖的问题不同：

| 方案 | 适用阶段 | 能发现的问题 | 局限 |
|---|---|---|---|
| 离线单测 | 研发期 | 代码逻辑错误、基础函数回归 | 不覆盖数据分布风险 |
| 数据验证 | 训练前、发布前 | schema 错误、异常值、漂移、skew | 不直接判断模型效果 |
| 模型验证 | 发布前 | 整体指标不足、切片退化、基线回退 | 依赖验证集代表性 |
| shadow | 预发布 | 真实流量下的线上线下一致性 | 不直接暴露用户反馈 |
| canary | 上线期 | 小流量真实效果异常 | 仍需监控和回退机制 |
| 在线监控 | 长期运行 | 漂移、延迟、业务指标衰减 | 发现时问题已在生产 |

所以边界要说清楚：

- 如果数据结构长期稳定、业务风险低，可以简化漂移检测，重点放在 schema 和整体指标。
- 如果存在明显的关键人群，比如新用户、低频商户、特定国家，就必须做切片门禁。
- 如果业务代价高，比如误拒支付、错误授信，就必须保留候选/基线比较，并进入灰度发布。
- 如果是研究型原型，强依赖快速试错，可以只保留基础检查，不必一开始就堆满完整门禁链。

可以把决策逻辑理解成一个简化判断：

1. 线上数据是否经常变化？
2. 是否存在关键切片？
3. 失败代价是否高？
4. 是否已经有稳定基线可比较？

如果前三个问题里有两个以上回答“是”，那就不应只做离线指标检查，而应把数据验证、切片门禁、基线比较和灰度发布全部接上。

真实工程里，推荐系统和支付风控通常比实验性模型更需要强门禁。原因不是模型更复杂，而是错误成本更高。一个推荐模型退化，可能损失点击率；一个支付风控模型退化，可能直接带来拒付损失或误杀正常用户。

---

## 参考资料

| 类型 | 资料 | 用途 |
|---|---|---|
| 论文 | ML Test Score | 解释 ML 生产就绪需要哪些测试与监控 |
| 论文 | Hidden Technical Debt in ML Systems | 解释为什么数据依赖会积累技术债 |
| 官方文档 | TFDV Guide / Get Started | 数据 schema、drift、skew 的实现方式 |
| 官方文档 | TFMA Model Validations / Get Started | 切片评估、阈值门禁、基线比较 |

1. [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
2. [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
3. [TensorFlow Data Validation Guide](https://www.tensorflow.org/tfx/guide/tfdv)
4. [TFDV Get Started](https://www.tensorflow.org/tfx/data_validation/get_started)
5. [TensorFlow Model Analysis: Model Validations](https://www.tensorflow.org/tfx/model_analysis/model_validations)
6. [TFMA Get Started](https://www.tensorflow.org/tfx/model_analysis/get_started)

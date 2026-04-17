## 核心结论

异常数据检测不是“找离群点”这么简单。工程里真正要解决的问题是：用一套可解释、可配置、可复核的规则，把高风险样本从正常流量里筛出来，交给人工确认，而不是让单一规则直接决定“删除”或“放行”。

更稳妥的做法，是把三类信号统一进一个规则引擎：

1. 统计异常：看数值是否明显偏离整体分布。
2. 文本异常：看文本是否过长、过短、重复、乱码。
3. 结构异常：看字段、类型、Schema 是否偏离预期。

这里的规则引擎，可以理解成“把很多检查结果按条件组合起来的判定器”。它不只输出一个布尔值，还应该输出命中的规则、分数、证据和后续动作，比如“进入人工复核队列”。

玩具例子很直接：抓取网页商品价格时，正常价格大多在 10 到 18 之间，突然出现一个 `100`，同时该条记录还缺失 `title` 字段，这种样本就不该直接入库，而应该先让人看一眼。核心目标不是“把所有异常都自动拦住”，而是“把最值得怀疑的样本优先送去复核”。

---

## 问题定义与边界

异常数据，指的是在统计分布、文本形态或结构约束上明显偏离预期的数据。这里的“偏离预期”，不是泛指一切不同，而是指足以影响下游分析、模型训练、监控报表或业务决策的偏差。

边界必须先说清楚：

| 规则类型 | 主要对象 | 典型信号 | 常见触发条件 | 容忍范围 |
| --- | --- | --- | --- | --- |
| 统计规则 | 数值字段 | 极大值、极小值、分布突变 | `|Z|>2`、超出 IQR、超出分位数边界 | 可按字段单独配置 |
| 文本规则 | 标题、正文、描述 | 过短、过长、重复片段、乱码 | 长度 `< min`、长度 `> max`、重复率超阈值、非文本字符比率过高 | 与字段语义相关 |
| 结构规则 | 整条记录 | 缺字段、多字段、类型不一致、Schema 漂移 | 缺少必填字段、`price` 从 number 变 string、新增未知字段 | 允许白名单与版本升级 |

这里的 Schema，白话说就是“一条数据应该长什么样”。比如应该有哪些字段、字段是什么类型、哪些字段必须有。

新手最容易踩的坑，是把“统计异常”当成“全部异常”。实际上，`price=100` 可能真的是高价商品，不一定错；而 `price="未知"` 虽然不是统计极端，却是明显的结构异常。又比如正文从平时的 300 字突然变成 0 字，这更像文本抽取失败，而不是数值分布问题。

因此，边界应当明确为：

- 统计规则只回答“数值是否偏离整体”。
- 文本规则只回答“文本形态是否异常”。
- 结构规则只回答“记录结构是否违反约束”。
- 最终是否报警，由统一引擎联动判断，而不是某一类规则单独拍板。

---

## 核心机制与推导

统计异常检测常见有三种规则：Z-Score、IQR、百分位阈值。

Z-Score 可以理解为“一个值离平均值有多少个标准差”。标准差是“数据波动幅度”的度量。公式是：

$$
Z=\frac{x-\mu}{\sigma}
$$

其中，$x$ 是当前值，$\mu$ 是均值，$\sigma$ 是标准差。经验上，若 $|Z|>2$ 或 $|Z|>3$，就可认为该点明显偏离整体。

IQR 是“四分位距”，白话说就是“中间 50% 数据的宽度”。先求：

$$
IQR = Q3 - Q1
$$

再给出上下界：

$$
Lower = Q1 - 1.5 \times IQR
$$

$$
Upper = Q3 + 1.5 \times IQR
$$

超出这个区间的值，可视为异常候选。IQR 的优点是对极端值更稳，不容易被单个大值拉偏。

百分位阈值更直接：用第 5 百分位和第 95 百分位作为边界。它回答的问题是：“如果我们只保留中间 90% 的正常范围，边界在哪？”公式可以简单记为：

$$
Lower = P_{5}, \quad Upper = P_{95}
$$

玩具例子用数据集：

`[10,12,12,13,12,14,10,15,14,13,16,18,100]`

其均值约为 $19.92$，标准差约为 $25.43$。对 `100` 计算：

$$
Z=\frac{100-19.92}{25.43}\approx 3.15
$$

因为 $|3.15| > 2$，所以 `100` 被标记为统计异常。

但工程里不能只停在这里。真正有用的是把多个信号串联成表达式，例如：

| 样本字段 | 统计信号 | 文本信号 | 结构信号 | 最终判定 |
| --- | --- | --- | --- | --- |
| `price=100` | `|Z|>2` | 无 | 无 | 弱异常，先打标 |
| `price=100` 且 `text_len=5` | `|Z|>2` | 文本过短 | 无 | 进入人工复核 |
| `price="100元"` 且缺失 `title` | 不参与数值统计 | 可能正常 | 类型错误 + 缺字段 | 直接高优先级告警 |

一个典型规则表达式可以写成：

`(|Z_price| > 2 AND text_len < 10) OR schema_invalid`

这一步很关键。规则引擎真正做的，不是“算公式”，而是“把不同来源的异常证据合并成一个可执行判断”。

真实工程例子：网页爬虫抓取电商页面，过去 `price` 一直是数字，`body` 平均长度 500 到 2000 字。某次站点改版后，抓到的数据里出现：

- `price="缺货"`
- `body=""`
- 多了一个未登记的 `discountLabel` 字段

这时统计规则、文本规则、结构规则会同时触发。比起只看单个字段，统一引擎更容易定位为“页面结构变了，抓取逻辑失效”，而不是把每个字段分散处理。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它演示三件事：统计阈值计算、文本/结构传感器、规则引擎聚合。

```python
from statistics import mean, pstdev

def percentile(values, p):
    values = sorted(values)
    if not values:
        raise ValueError("empty values")
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)

def calc_z_score(x, values):
    mu = mean(values)
    sigma = pstdev(values)
    if sigma == 0:
        return 0.0
    return (x - mu) / sigma

def calc_iqr_bounds(values):
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def calc_percentile_bounds(values, low=0.05, high=0.95):
    return percentile(values, low), percentile(values, high)

def text_signals(text, min_len=10, max_len=5000):
    text = text or ""
    repeated = len(text) >= 6 and text[:3] * (len(text) // 3) == text[:3 * (len(text) // 3)]
    non_printable_ratio = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r") / max(len(text), 1)
    return {
        "too_short": len(text) < min_len,
        "too_long": len(text) > max_len,
        "repeated": repeated,
        "garbled": non_printable_ratio > 0.1,
    }

def schema_signals(record, required_fields, field_types):
    missing = [f for f in required_fields if f not in record]
    wrong_type = []
    for field, typ in field_types.items():
        if field in record and not isinstance(record[field], typ):
            wrong_type.append(field)
    return {
        "missing_fields": missing,
        "wrong_type_fields": wrong_type,
        "schema_invalid": bool(missing or wrong_type),
    }

def evaluate_record(record, price_history):
    z = calc_z_score(record["price"], price_history) if isinstance(record.get("price"), (int, float)) else None
    iqr_low, iqr_high = calc_iqr_bounds(price_history)
    p_low, p_high = calc_percentile_bounds(price_history)

    text_flags = text_signals(record.get("body", ""))
    schema_flags = schema_signals(
        record,
        required_fields=["title", "price", "body"],
        field_types={"title": str, "price": (int, float), "body": str},
    )

    stat_abnormal = (
        z is not None and abs(z) > 2
    ) or (
        isinstance(record.get("price"), (int, float)) and (record["price"] < iqr_low or record["price"] > iqr_high)
    ) or (
        isinstance(record.get("price"), (int, float)) and (record["price"] < p_low or record["price"] > p_high)
    )

    need_review = (
        (stat_abnormal and text_flags["too_short"])
        or schema_flags["schema_invalid"]
        or text_flags["garbled"]
        or text_flags["repeated"]
    )

    return {
        "z_score": z,
        "stat_abnormal": stat_abnormal,
        "text_flags": text_flags,
        "schema_flags": schema_flags,
        "need_review": need_review,
    }

history = [10, 12, 12, 13, 12, 14, 10, 15, 14, 13, 16, 18, 100]

record_ok = {"title": "商品A", "price": 16, "body": "这是一段正常的商品描述，长度足够。"}
record_bad = {"title": "商品B", "price": 100, "body": "短"}

ok_result = evaluate_record(record_ok, history)
bad_result = evaluate_record(record_bad, history)

assert ok_result["need_review"] is False
assert bad_result["stat_abnormal"] is True
assert bad_result["text_flags"]["too_short"] is True
assert bad_result["need_review"] is True
```

如果把它抽象成工程模块，可以分成三层：

1. `signals` 层：只负责产生信号，例如 `z_score`、`too_short`、`schema_invalid`。
2. `rules` 层：把信号组合成表达式，例如 `stat_abnormal AND too_short`。
3. `actions` 层：决定动作，例如 `log`、`alert`、`enqueue_review`。

配置可以做成 JSON：

```json
{
  "numeric_rules": {
    "price": {
      "zscore_threshold": 2.0,
      "percentile_low": 0.05,
      "percentile_high": 0.95
    }
  },
  "text_rules": {
    "body": {
      "min_len": 10,
      "max_len": 5000,
      "detect_repeated": true,
      "detect_garbled": true
    }
  },
  "schema_rules": {
    "required_fields": ["title", "price", "body"],
    "field_types": {
      "title": "str",
      "price": "number",
      "body": "str"
    }
  },
  "engine_rules": [
    "(price.stat_abnormal AND body.too_short) => review",
    "(record.schema_invalid) => alert",
    "(body.garbled OR body.repeated) => review"
  ]
}
```

新手要抓住一个重点：统计函数负责“算边界”，传感器负责“看字段”，规则引擎负责“做组合判断”。这三层拆开后，代码才容易维护。

---

## 工程权衡与常见坑

最常见的问题不是“规则太少”，而是“规则太激进”。阈值一旦设得过紧，误报会迅速上升，团队每天都在处理没价值的告警。

| 常见坑 | 现象 | 后果 | 应对策略 |
| --- | --- | --- | --- |
| 阈值过紧 | `|Z|>1.5` 到处触发 | 误报高，人工疲劳 | 先宽后紧，按字段校准 |
| 告警风暴 | 同一问题重复报警 | 监控噪声淹没有效信号 | 做频控、去重、聚合 |
| 缺少人工复核 | 规则永远不修 | 误报与漏报长期共存 | 建立复核反馈闭环 |
| 规则膨胀 | 每个异常都加一条 if | 配置失控，维护困难 | 分层设计，抽象公共规则 |

真实工程里，频控很重要。比如某字段过去一周持续触发 `|Z|>2`，如果每条都报警，值班系统就会被刷屏。更合理的是做聚合策略：同一字段、同一来源、同一小时内只报一次，并附带样本数和代表记录。

人工复核闭环可以简化成：

`异常命中 -> 进入复核队列 -> 人工标记真异常/假异常 -> 调整阈值或补充规则 -> 下次重新评估`

这里的 human-in-the-loop，白话说就是“让人参与最终判断，并把判断结果反过来喂给系统”。没有这个闭环，规则只能越堆越多，不能真正收敛。

还有一个常见坑：把不同字段混在一起算统计阈值。比如把“图书价格”和“手机价格”放进同一分布，得到的均值和分位数几乎没有意义。正确做法通常是按字段、按来源、按品类、按站点分桶计算。

---

## 替代方案与适用边界

不是所有场景都需要统一规则引擎，但数据越异构，越需要混合方案。

| 方法 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 纯统计方法 | 简单、快、易解释 | 无法处理文本和结构问题 | 数值监控、稳定指标流 |
| 纯规则方法 | 可控、可审计 | 覆盖有限，规则会膨胀 | 结构稳定、业务明确 |
| 轻量 ML/聚类 | 能发现复杂模式 | 解释性弱，运维成本更高 | 低频复杂样本、异构文本 |
| 混合引擎 | 兼顾解释性与覆盖面 | 设计复杂度更高 | 数据工程与质量平台 |

如果场景很简单，比如固定报表字段的范围校验，只用规则就够了。如果主要问题是极端数值波动，统计阈值就够了。但一旦出现 schema drift、文本抽取失败、字段类型漂移，仅靠 Z-Score 基本不够。

一个典型替代方案，是在规则引擎之外增加轻量聚类模型。聚类可以理解为“把相似样本自动分组”。如果某批记录同时远离已有群组中心，就作为“可疑新模式”送人工验证。它适合下面这种情况：

- 文本结构很复杂，手写规则难覆盖。
- 正常模式本身就有多种类型。
- 数据量不大，但每条记录价值高。

反过来，如果数据是高频、稳定、结构固定的日志流或指标流，坚持“规则 + 缓冲层 + 人工复核”通常更可靠。因为这类场景最看重的是低延迟、可解释和可审计，不一定需要模型。

结论是：ML 不是替代规则，而是补充规则。只有当规则已经稳定运行，但仍存在大量“规则描述不了”的异常时，再考虑加入聚类或轻量模型。

---

## 参考资料

- ManageEngine, Anomaly Detection  
  用于理解 Z-Score、IQR、百分位阈值这三类统计异常检测的基本思路，适合做阈值配置入门。

- Swiftorial, Statistical Methods for Anomaly Detection  
  给出了 Z-Score 与分位数方法的直观说明，也提供了适合新手理解的数值例子。

- DQOps, How to Detect Data Quality Issues in Text Fields  
  适合参考文本字段的长度、重复模式、异常变化等规则设计方式。

- Grepsr, Schema Drift in Web Data Pipelines  
  重点看结构异常与 schema drift 的工程监控思路，特别适合网页抓取和数据管道场景。

- Shaip, Human-in-the-Loop Approach for AI Data Quality  
  用于理解人工复核闭环、责任追踪和反馈机制在数据质量系统中的作用。

- The DataOps Organization, Data Quality Rules  
  适合补充数据质量规则体系、治理流程与告警管理的整体框架。

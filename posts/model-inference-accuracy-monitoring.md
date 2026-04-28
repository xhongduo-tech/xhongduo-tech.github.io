## 核心结论

模型推理精度监控，是在生产环境里持续估计或复算模型真实效果指标的工程系统。这里的“精度监控”不是泛指任何监控，而是明确指向 `accuracy / precision / recall / F1 / RMSE / FPR / AUROC` 这类和业务结果直接相关的指标。

离线测试分数只能说明模型在历史样本上的能力，不能直接代表上线后的真实表现。原因很简单：上线后的输入分布会变，用户行为会变，采样方式会变，标签还可能延迟几天甚至几周才回来。所以一个测试集上 `accuracy=95%` 的模型，上线后今天的真实精度完全可能不是 `95%`。

新手最容易混淆的一点是：输入漂移监控不是精度监控。所谓“漂移”，白话说就是线上数据长得和训练时不太一样；它只能说明“可能有风险”，不能直接说明模型已经变差。真正的性能裁决，要么依赖真实标签回流后的复算，要么在没有标签时依赖校准后的置信度做近似估计。

下面这张表先把“离线指标”和“在线监控”区分清楚：

| 对比项 | 离线指标 | 在线精度监控 |
|---|---|---|
| 目标 | 评估训练后模型能力 | 评估生产环境真实表现 |
| 数据来源 | 测试集、验证集 | 真实线上请求 |
| 标签状态 | 通常已知 | 可能延迟、缺失、稀疏 |
| 结论强度 | 训练阶段参考 | 运营阶段裁决 |
| 是否可直接告警 | 通常不能 | 应该能 |
| 核心风险 | 过拟合、数据泄漏 | 分布变化、标签延迟、切片失效 |

一个最小玩具例子可以直接说明问题。假设模型今天在线预测了 4 个样本，分数分别是 `0.9, 0.8, 0.4, 0.2`，阈值是 `0.5`。你现在还没有真实标签，因此不能直接算真实精度。但如果这些分数已经做过概率校准，就可以先估计“这 4 个预测大概有多少是对的”。等 7 天后真实标签回来，再用真实精度去校验这个估计有没有明显偏差。这个“先估计，后复算”的两阶段思路，就是很多真实系统的基本做法。

---

## 问题定义与边界

模型推理精度监控关注的是“推理结果的真实性能”，不是“模型服务是否活着”，也不是“输入分布是否变化”。“真实性能”这个词可以理解成：模型在真实线上请求上，究竟做对了多少、漏掉了多少、误报了多少。

不同任务监控的指标不同：

| 任务类型 | 常见指标 | 适用原因 |
|---|---|---|
| 分类任务 | `accuracy / precision / recall / F1 / FPR / AUROC` | 要区分预测是否命中、误报是否过多 |
| 回归任务 | `RMSE / MAE / MAPE / R^2` | 要衡量预测值与真实值偏差 |
| 排序任务 | `NDCG / MAP / Recall@K / CTR uplift` | 要衡量排序前列是否更相关 |
| 风控/审核 | `recall / precision / FPR / TPR` | 通常更关心漏放与误杀成本 |
| 推荐/广告 | `CTR / CVR / calibration / revenue per mille` | 线上结果与业务收益直接绑定 |

边界必须讲清楚，因为很多系统实际上不具备“实时真实标签”。常见有三种场景：

| 场景 | 能做什么 | 不能做什么 |
|---|---|---|
| 有标签 | 直接复算真实指标，作为最终裁决 | 不能忽略切片监控 |
| 延迟标签 | 先做无标签估计，后做标签复算校验 | 不能把估计值当最终事实 |
| 无标签 | 做风险监控、漂移监控、人工抽检 | 不能声称自己在做真实精度监控 |

这里有一个典型误区。垃圾邮件模型发现最近邮件标题分布变了，这只能说明 `P(X)` 可能变了。`P(X)` 白话说就是“输入长什么样”的概率分布。模型效果真正依赖的是 `P(Y|X)`，也就是“给定输入后，标签怎么生成”的关系。如果攻击者换了写法，但模型仍能识别，漂移会上升而精度不一定下降；反过来，如果输入分布变化很小，但标签机制变了，比如规则调整、人工标注口径变化，精度也可能明显下滑。

所以这篇文章的边界是明确的：

1. 讨论对象是生产推理结果的质量监控。
2. 漂移检测只作为预警，不作为最终裁决。
3. 只要没有真实标签，任何“精度”都只是估计值，而不是结论。
4. 无标签估计主要适用于二分类或能稳定输出可校准置信度的场景，对复杂排序、多阶段决策链的适用性更弱。

---

## 核心机制与推导

先看有标签的情况。二分类任务里，最基础的真实精度定义是：

$$
\mathrm{Acc}=\frac{TP+TN}{N}
$$

这里 `TP` 是真正例，白话说就是“预测为正且真实也为正”；`TN` 是真负例；`FP` 是假正例；`FN` 是假负例。它们组成混淆矩阵。

| 真实混淆矩阵 | 真实为正 | 真实为负 |
|---|---:|---:|
| 预测为正 | `TP` | `FP` |
| 预测为负 | `FN` | `TN` |

如果线上标签及时返回，事情很简单：按窗口把请求和标签对齐，然后直接算混淆矩阵，再推导出 `accuracy / recall / precision / FPR`。这是最可信的方法。

问题出在“标签还没回来”。这时只能估计。设模型对样本 $x_i$ 输出校准后的正类概率 $s_i=P(y_i=1|x_i)$，阈值为 $t$，预测标签是：

$$
\hat y_i = 1[s_i \ge t]
$$

如果 $s_i$ 已经校准得比较好，就能把每条样本看成一个“正确概率已知、真实标签未知”的随机变量。于是可以定义期望混淆矩阵：

$$
\widehat{TP}=\sum_{i:\hat y_i=1}s_i,\quad
\widehat{FP}=\sum_{i:\hat y_i=1}(1-s_i)
$$

$$
\widehat{TN}=\sum_{i:\hat y_i=0}(1-s_i),\quad
\widehat{FN}=\sum_{i:\hat y_i=0}s_i
$$

于是期望精度是：

$$
\widehat{\mathrm{Acc}}=\frac{1}{N}\sum_i \big[\hat y_i s_i + (1-\hat y_i)(1-s_i)\big]
$$

把它翻成白话其实不复杂：

1. 如果模型把某条样本判成正类，那么“这条判对的概率”就是它的正类概率 $s_i$。
2. 如果模型把某条样本判成负类，那么“这条判对的概率”就是 $1-s_i$。
3. 把每条样本“判对的概率”加起来，就是期望正确数。
4. 再除以样本总数，就是估计精度。

玩具例子如下。4 条请求的分数是 `0.9, 0.8, 0.4, 0.2`，阈值 `0.5`，于是预测标签为 `1,1,0,0`。估计正确数：

- 前两条预测为正，期望正确数是 `0.9 + 0.8`
- 后两条预测为负，期望正确数是 `(1-0.4) + (1-0.2) = 0.6 + 0.8`
- 总期望正确数 `= 3.1`
- 估计精度 `= 3.1 / 4 = 77.5%`

这就是“无标签精度估计”的基本逻辑。

再把真实混淆矩阵和期望混淆矩阵放在一起看：

| 项目 | 有真实标签时 | 无真实标签时 |
|---|---|---|
| `TP` | 直接计数 | 用 $\sum s_i$ 估计 |
| `FP` | 直接计数 | 用 $\sum (1-s_i)$ 估计 |
| `TN` | 直接计数 | 用 $\sum (1-s_i)$ 估计 |
| `FN` | 直接计数 | 用 $\sum s_i$ 估计 |
| 结论性质 | 最终裁决 | 近似估计 |

这个方法为什么能成立，取决于两个前提。

第一，`calibration` 必须足够好。`calibration` 白话说就是“模型打出 0.8 分时，现实里大约真的有 80% 会是正类”。如果分数只是排序分，不是概率分，那上面的期望推导就没有意义。

第二，监控窗口内数据分布不能偏离太远。如果训练时校准是在一种分布上做的，而现在线上已经换成另一种机制，$s_i$ 可能系统性偏高或偏低，估计值就会失真。

几个相关术语也要说明：

| 术语 | 白话解释 | 作用 |
|---|---|---|
| `calibration` | 分数和真实概率是否对得上 | 决定无标签估计能否成立 |
| `ECE` | Expected Calibration Error，分桶后比较预测概率和真实频率的平均偏差 | 衡量校准误差 |
| `Brier score` | 概率预测与真实标签之间的平方误差 | 同时看准确性和校准性 |

成立与不成立的边界如下：

| 条件 | 结论 |
|---|---|
| 概率已校准、样本量足够、窗口分布稳定 | 无标签估计可作为短期代理指标 |
| 概率未校准 | 估计值不可信 |
| 标签分布突变、业务规则变化 | 历史校准可能失效 |
| 样本量很小 | 单窗口波动过大，估计噪声很重 |
| 极度类别不平衡，只看 `accuracy` | 结论容易误导 |

标签回流前后的时序也应分开看：

```text
推理时刻 T0:
请求到达 -> 模型输出 prediction/score -> 写入推理日志 -> 生成实时估计指标

标签回流时刻 T+K:
业务结果产生 -> 生成 groundTruth -> 按 inferenceId 合并 -> 复算真实指标 -> 校验估计偏差
```

这意味着生产里的“精度监控”通常不是一个指标，而是两层指标：
一层是短期无标签估计，追求快；
另一层是延迟真实复算，追求准。

---

## 代码实现

工程上建议拆成三层：在线埋点、标签回流合并、指标计算与告警。重点不是算法多高级，而是日志字段是否完整、ID 是否能对齐、窗口是否合理、切片是否稳定。

先定义最关键的日志字段：

| 字段 | 含义 | 作用 |
|---|---|---|
| `inferenceId` | 每次推理唯一 ID | 关联标签的主键 |
| `eventTime` | 推理发生时间 | 做窗口聚合 |
| `prediction` | 最终预测标签 | 计算混淆矩阵 |
| `score` | 模型置信度或概率 | 做估计与分析 |
| `threshold` | 当前判定阈值 | 复现决策逻辑 |
| `modelVersion` | 模型版本号 | 版本对比 |
| `featureSnapshot` | 关键特征快照 | 排查问题与抽样分析 |
| `groundTruth` | 真实标签 | 真实复算必需 |
| `labelTime` | 标签到达时间 | 区分延迟层 |

真实工程例子可以看支付风控。授权时模型立即输出“拦截/放行”，但拒付或人工审核结果往往要 `T+7` 或更久。此时系统通常这么分层：

```text
实时层:
在线请求 -> 写 inference log -> 生成分群统计 -> 用校准 score 估计短期表现

延迟层:
标签回流 -> 按 inferenceId merge -> 计算真实 recall/FPR/accuracy -> 与估计值比对 -> 告警或触发重训
```

下面给一个可运行的 Python 示例，覆盖推理日志、标签合并、滚动窗口计算和告警。代码做了简化，但结构和真实系统是一致的。

```python
from collections import defaultdict
from datetime import datetime

def estimate_accuracy(rows):
    n = len(rows)
    assert n > 0
    correct_expectation = 0.0
    for r in rows:
        y_hat = r["prediction"]
        s = r["score"]
        assert 0.0 <= s <= 1.0
        assert y_hat in (0, 1)
        correct_expectation += y_hat * s + (1 - y_hat) * (1 - s)
    return correct_expectation / n

def merge_labels(pred_logs, label_logs):
    label_map = {x["inferenceId"]: x for x in label_logs}
    merged = []
    for p in pred_logs:
        item = dict(p)
        label = label_map.get(p["inferenceId"])
        item["groundTruth"] = None if label is None else label["groundTruth"]
        item["labelTime"] = None if label is None else label["labelTime"]
        merged.append(item)
    return merged

def true_metrics(rows):
    labeled = [r for r in rows if r["groundTruth"] is not None]
    assert labeled, "need at least one labeled sample"
    tp = fp = tn = fn = 0
    for r in labeled:
        y_hat = r["prediction"]
        y = r["groundTruth"]
        if y_hat == 1 and y == 1:
            tp += 1
        elif y_hat == 1 and y == 0:
            fp += 1
        elif y_hat == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    acc = (tp + tn) / len(labeled)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"accuracy": acc, "recall": recall, "fpr": fpr}

def group_by_day(rows):
    buckets = defaultdict(list)
    for r in rows:
        day = r["eventTime"][:10]
        buckets[day].append(r)
    return dict(buckets)

def consecutive_alert(values, threshold, k):
    streak = 0
    for v in values:
        if v < threshold:
            streak += 1
            if streak >= k:
                return True
        else:
            streak = 0
    return False

pred_logs = [
    {"inferenceId": "a", "eventTime": "2026-04-01T10:00:00", "prediction": 1, "score": 0.9, "threshold": 0.5, "modelVersion": "v3", "featureSnapshot": {"channel": "ios"}},
    {"inferenceId": "b", "eventTime": "2026-04-01T11:00:00", "prediction": 1, "score": 0.8, "threshold": 0.5, "modelVersion": "v3", "featureSnapshot": {"channel": "ios"}},
    {"inferenceId": "c", "eventTime": "2026-04-01T12:00:00", "prediction": 0, "score": 0.4, "threshold": 0.5, "modelVersion": "v3", "featureSnapshot": {"channel": "android"}},
    {"inferenceId": "d", "eventTime": "2026-04-02T09:00:00", "prediction": 0, "score": 0.2, "threshold": 0.5, "modelVersion": "v3", "featureSnapshot": {"channel": "android"}},
]

label_logs = [
    {"inferenceId": "a", "groundTruth": 1, "labelTime": "2026-04-08T10:00:00"},
    {"inferenceId": "b", "groundTruth": 0, "labelTime": "2026-04-08T11:00:00"},
    {"inferenceId": "c", "groundTruth": 0, "labelTime": "2026-04-08T12:00:00"},
    {"inferenceId": "d", "groundTruth": 0, "labelTime": "2026-04-09T09:00:00"},
]

merged = merge_labels(pred_logs, label_logs)
estimated_acc = estimate_accuracy(pred_logs)
metrics = true_metrics(merged)

assert round(estimated_acc, 3) == 0.775
assert round(metrics["accuracy"], 3) == 0.750
assert round(metrics["recall"], 3) == 1.000
assert round(metrics["fpr"], 3) == 0.333

daily_true_acc = []
for day, rows in sorted(group_by_day(merged).items()):
    labeled_rows = [r for r in rows if r["groundTruth"] is not None]
    if labeled_rows:
        daily_true_acc.append(true_metrics(labeled_rows)["accuracy"])

assert consecutive_alert([0.84, 0.81, 0.78], threshold=0.80, k=2) is True
```

这个例子里，估计精度是 `77.5%`，真实精度是 `75%`。两者很接近，说明在这个极小样本下估计没有明显跑偏。但真正工程里不能只看一次对齐，而要长期监控“估计值和真实值的偏差”。

如果把实现流程压成最小步骤，就是：

1. 推理时记录 `inferenceId / score / threshold / feature snapshot`
2. 标签回来后按 `inferenceId` 关联
3. 每天或每小时做窗口聚合
4. 按总量和切片计算 `accuracy / recall / FPR`
5. 若某个切片连续 3 个窗口低于阈值，则报警
6. 若估计值与真实值长期偏离，则检查校准或重做校准

---

## 工程权衡与常见坑

最常见的错误不是“不会算指标”，而是“算出来的指标不代表真实问题”。

第一类坑是只看整体均值。整体 `accuracy=92%` 并不说明系统健康，因为一个大流量切片可以掩盖多个小流量切片的失效。比如整体还是 `92%`，但 iOS 新版本只有 `78%`，如果不按设备版本切片，这个问题会被平均值直接吞掉。

第二类坑是 ID 没对齐。标签回流如果不能用 `inferenceId` 精确关联推理请求，后面的真实复算几乎都不可信。很多团队直到上线后才发现，只记了用户 ID、订单 ID，没记推理事件 ID，结果一条请求对应多次重试、多模型版本并存时，根本没法正确 join。

第三类坑是拿未校准的分数直接做无标签估计。排序模型常常能给出“高分更像正类”，但这个分数不一定是概率。若 `0.9` 并不真的意味着 90% 为正，那么 $\widehat{TP}$、$\widehat{FP}$ 的推导就没有统计意义。

常见坑与规避可以放在一张表里：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 只看漂移，不看真实指标 | 把风险信号当结论 | 漂移做预警，真实标签做裁决 |
| 标签回流未对齐 ID | 指标失真 | 强制记录 `inferenceId` |
| 只看整体均值 | 局部失效被掩盖 | 按渠道、地域、设备、版本分群 |
| 置信度未校准就做估计 | 估计值系统性偏差 | 先做 calibration，并跟踪 `ECE/Brier` |
| 类别不平衡只看 `accuracy` | 漏掉高误杀或高漏放 | 同时看 `precision/recall/FPR` |
| 窗口太短导致噪声过大 | 告警抖动严重 | 设最小样本量和滚动窗口 |

告警也不能只靠单阈值。更稳妥的设计通常分成几类：

| 告警类型 | 触发条件 | 适用场景 |
|---|---|---|
| 指标阈值告警 | 如 `recall < 0.85` | 明确业务底线 |
| 连续多窗口告警 | 连续 3 个窗口低于阈值 | 过滤偶发噪声 |
| 分群异常告警 | 某 slice 明显劣化 | 局部回归、客户端问题 |
| 估计与真实偏差告警 | $|\hat m - m|$ 长期过大 | 检查校准是否失效 |

一个常被忽略的权衡是窗口长度。窗口太短，告警会充满随机噪声；窗口太长，问题发现太晚。实践里通常会同时保留快慢两层窗口，比如 `1h` 看实时趋势，`1d` 看稳定结论，再叠加最小样本量门槛。

---

## 替代方案与适用边界

如果标签足够快回流，直接复算真实指标永远是首选。它简单、可信、可解释，不依赖额外统计前提。所谓“首选”，不是因为它更高级，而是因为它假设最少。

当标签存在延迟但最终能回来时，比较合理的方案是“两段式”：
前半段用校准后的置信度做估计，保证发现问题足够快；
后半段等标签回流后复算真实指标，并持续校验估计偏差。

如果标签长期缺失，事情就要讲清边界。你可以监控输入漂移、分数分布、异常率、人工抽检结果，但不能把这些东西直接命名为“精度监控结果”。那只是风险代理，不是真实效果。

几个常见方案对比如下：

| 方案 | 核心做法 | 优点 | 局限 |
|---|---|---|---|
| 直接复算 | 标签回流后直接算真实指标 | 可信、简单、可解释 | 依赖标签时效 |
| 校准置信度估计 | 用校准概率估计期望指标 | 快，适合延迟标签场景 | 依赖校准质量和分布稳定 |
| 输入漂移监控 | 监控 `P(X)` 变化 | 无需标签，发现风险早 | 不能证明精度下降 |
| 人工抽检 | 抽样人工标注 | 可用于冷启动和无标签场景 | 成本高、覆盖有限 |

适用边界也可以明确成表：

| 业务条件 | 推荐方案 |
|---|---|
| `T+1` 能拿到大部分标签 | 直接复算为主 |
| `T+7` 才能拿到标签 | 校准估计 + 延迟复算 |
| 几乎没有标签，但可人工抽样 | 人工抽检 + 风险监控 |
| 完全无标签且难以抽检 | 只能做漂移与异常监控，不能下精度结论 |
| 强监管、高风险业务 | 必须保留真实标签闭环，不应只依赖估计 |

新手可以把三种方案记成：

- 方案 A：`T+1` 有标签，直接每天算真实精度。
- 方案 B：`T+7` 才有标签，先估计，后复算。
- 方案 C：几乎没标签，只能看风险信号，不能把它当最终精度。

最后给一个明确结论：只要真实标签能稳定回流，无标签估计就不该替代真实评估；它只能缩短观测延迟，不能替代最终裁决。只要概率没有经过可靠校准，或者线上分布已经明显换挡，无标签估计就只能当参考，不能用来驱动高风险自动决策。

---

## 参考资料

1. [Amazon SageMaker Model Monitor: Model quality](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html) 这篇资料支持“在线模型质量监控”和“真实指标复算”的工程落地部分。  
2. [Amazon SageMaker: Ingest Ground Truth labels and merge them with predictions](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-merge.html) 这篇资料支持“标签回流合并”和 `inferenceId` 对齐的实现细节。  
3. [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) 这篇资料支持“概率校准、可靠性曲线、校准前提”的理论与实践部分。  
4. [Performance Estimation in Binary Classification Using Calibrated Confidence](https://link.springer.com/article/10.1007/s10994-025-06970-3) 这篇资料支持“用校准置信度做无标签性能估计”的核心机制与前提条件。  
5. [Vertex AI Model Monitoring overview](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview) 这篇资料支持“云上模型监控体系、漂移与质量监控分层”的工程背景。

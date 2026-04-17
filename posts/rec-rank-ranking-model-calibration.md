## 核心结论

排序模型的校准，目标不是把排序能力变强，而是把模型分数变成“可解释的概率”。这里的“概率”可以白话理解为：这个分数对应的真实发生率大约是多少。对推荐、广告、搜索来说，这一步很关键，因为策略层往往不是只看谁排前面，而是要根据概率去算期望收益、决定出价、设置阈值、做多目标融合。

一个常见误解是：“模型打了 80 分，所以点击概率就是 80%。”这通常是错的。很多排序模型只保证相对顺序，不能保证数值本身有概率含义。A 的分数比 B 高，说明 A 更值得排前，但不代表分数 0.8、2.3、15.7 可以直接当成真实命中率。

校准就是在原始分数 $f$ 之外，再学习一个映射 $g(f)$，让输出 $\hat p$ 更接近真实概率。策略层真正要用的是这个 $\hat p$，而不是未经处理的原始分数。

| 方法 | 典型公式 | 参数量 | 是否保序 | 过拟合风险 | 适用场景 |
|---|---|---:|---|---|---|
| Platt Scaling | $\hat p=\frac{1}{1+e^{Af+B}}$ | 2 个 | 是 | 低到中 | 数据不大，先求稳 |
| Temperature Scaling | $\hat p=\sigma(f/T)$ | 1 个 | 是 | 低 | 已有 logit，模型整体“过热/过冷” |
| Isotonic Regression | $\hat p=g(f),\ g \text{ 单调}$ | 非参数 | 是 | 中到高 | 数据足够多，映射明显非线性 |

玩具例子：某模型给一个商品打分 2.0，新手会以为“这是高分，大概 70% 会点”。但若校准后采用 Platt 参数 $A=-0.5, B=0$，则

$$
\hat p=\frac{1}{1+e^{Af+B}}=\frac{1}{1+e^{-1}}\Bigg? 
$$

注意这里要代入的是 $Af+B=-1$，因此

$$
\hat p=\frac{1}{1+e^{1}}\approx 0.27
$$

意思是：这个“高分”样本在历史上真实点击率只有约 27%。如果策略层原来按 70% 去出价，就会系统性高估收益；按 27% 去算，决策才是合理的。

---

## 问题定义与边界

排序问题关心“谁在前”，校准问题关心“分数能不能解释成概率”。这两个目标相关，但不是一回事。

可以把流程写成：

$$
\text{原始分数 } f \rightarrow \text{校准概率 } \hat p \rightarrow \text{策略决策}
$$

其中“策略决策”包括出价、阈值过滤、重排权重、多目标融合等。若中间这一步错了，后面即使逻辑完全正确，也会建立在错误置信度上。

| 环节 | 输入 | 输出 | 潜在影响 |
|---|---|---|---|
| 排序模型 | 特征、候选集 | 原始分数 $f$ | 只保证相对顺序，不保证概率解释 |
| 校准模块 | 原始分数、标注数据 | 概率 $\hat p$ | 决定期望收益是否可信 |
| 策略层 | 概率、业务约束 | 出价/重排/阈值 | 会放大上游概率误差 |

边界也要说清楚：

1. 校准不能凭空提升特征表达能力。模型本身区分不出好坏样本时，校准也救不了。
2. 校准通常只修“分数到概率”的映射，不直接改排序学习目标。
3. 校准依赖分布稳定。线上流量、位置偏差、样本选择方式变了，旧校准函数可能失效。

真实工程例子：Elasticsearch 里常把 BM25、稠密召回、semantic reranker 的分数一起用于融合。问题在于这些分数来自不同模型、不同尺度、不同分布。某个 reranker 可能因为分布偏移，经常打出很高分，但实际命中率并不高。如果不先校准，融合层会误以为“这个模型特别自信”，结果给它过高权重，导致整体精度下降。把各模型分数先映射到统一概率空间，再做阈值或加权，才有可比性。

---

## 核心机制与推导

### 1. Platt Scaling

Platt Scaling 本质上是在原始分数上再拟合一个 Logistic 回归。Logistic 可以白话理解为：把任意实数压到 0 到 1 之间的 S 形函数。

公式是：

$$
\hat p=\frac{1}{1+e^{Af+B}}
$$

其中 $f$ 是原始分数，$A,B$ 是在独立校准集上学出来的参数。因为这个函数单调，所以不会破坏原有排序顺序，只是重塑分数的概率含义。

推导思路很直接：

1. 假设分数和真实概率之间存在单调 S 形关系。
2. 用带标签的数据 $(f_i,y_i)$ 去拟合 $A,B$。
3. 最小化对数损失，使输出概率尽量匹配真实点击/转化标签。

还是用上面的玩具例子。若 $f=2.0,\ A=-0.5,\ B=0$：

$$
Af+B=-1
$$

$$
\hat p=\frac{1}{1+e^{1}}\approx 0.2689
$$

这个结果的含义不是“模型变差了”，而是“模型原先把 2.0 这个分数说得太满，现在被拉回到历史真实水平”。

### 2. Isotonic Regression

Isotonic Regression 是保序回归。保序的意思是：更高的原始分数，校准后的概率不能更低。它不预设 S 形，而是直接学习一个单调函数 $g$：

$$
\hat p=g(f), \quad g \text{ 为单调非降函数}
$$

常见优化目标写成：

$$
\min_{g \in \mathcal{M}} \sum_{i=1}^{n}(y_i-g(f_i))^2
$$

其中 $\mathcal{M}$ 表示所有单调非降函数。实际求解通常用 PAV 算法。PAV 可以白话理解为：如果相邻分桶违反单调性，就把它们合并，直到整体单调为止。

优点是灵活，能拟合复杂形状；缺点是样本少时容易把噪声当规律，得到一条锯齿状、泛化差的映射。

### 3. Temperature Scaling

Temperature Scaling 常用于已有 logit 的模型。logit 可以白话理解为：进入 sigmoid 或 softmax 之前的原始决策值。它的做法是只引入一个温度参数 $T>0$：

$$
\hat p=\sigma(f/T)=\frac{1}{1+e^{-f/T}}
$$

若 $T>1$，logit 被压缩，输出更平，模型“降温”，过度自信会减弱。  
若 $0<T<1$，logit 被放大，输出更尖，模型“升温”。

它的特点是结构极简，只调整整体置信度，不改变相对顺序，适合“大体形状对，但整体偏热或偏冷”的情况。

---

## 代码实现

工程上通常采用“三段式”：

1. 先训练原始排序模型。
2. 再用独立校准集拟合校准器。
3. 部署时把 `score -> calibrated_prob` 封装成函数或服务。

下面给一个可运行的 Python 示例，演示 Platt、Temperature 和分桶 ECE 计算。代码里的 `assert` 用来保证输出落在合法范围内。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def platt_score_to_prob(score: float, A: float, B: float) -> float:
    # 注意公式写成 1 / (1 + exp(Af + B))
    p = 1.0 / (1.0 + math.exp(A * score + B))
    assert 0.0 <= p <= 1.0
    return p

def temperature_score_to_prob(logit: float, T: float) -> float:
    assert T > 0.0
    p = sigmoid(logit / T)
    assert 0.0 <= p <= 1.0
    return p

def brier_score(probs, labels):
    assert len(probs) == len(labels) and len(probs) > 0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)

def ece(probs, labels, n_bins=5):
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    total = len(probs)
    err = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_y = sum(y for _, y in bucket) / len(bucket)
        err += len(bucket) / total * abs(avg_p - avg_y)
    return err

# 玩具例子：f=2.0, A=-0.5, B=0.0
p = platt_score_to_prob(2.0, -0.5, 0.0)
assert abs(p - 0.2689) < 1e-3

# 一个小样本：原始概率明显偏高
raw_probs = [0.90, 0.80, 0.70, 0.20, 0.10]
labels    = [1,    0,    0,    0,    0]

print("platt toy prob:", round(p, 4))
print("brier:", round(brier_score(raw_probs, labels), 4))
print("ece:", round(ece(raw_probs, labels, n_bins=5), 4))
```

部署接口通常很简单：

| 参数 | 类型 | 说明 |
|---|---|---|
| `score` | `float` | 排序模型原始分数或 logit |
| `model_id` | `string` | 指定使用哪套校准参数 |
| 返回 `prob` | `float` | 校准后的概率，可直接给策略层使用 |

真实工程里要注意两点。第一，不同模型要有各自的校准器，不要混用参数。第二，校准最好做成可热更新配置，因为线上分布漂移后，经常需要重训而不必重发整个排序模型。

---

## 工程权衡与常见坑

最大的坑是：只看 AUC，以为模型已经够好了。AUC 衡量排序区分能力，但不关心 0.9 到底是不是 90% 的真实发生率。排序好，不代表概率准。

| 指标 | 关注点 | 惩罚什么错误 |
|---|---|---|
| LogLoss | 概率是否与真实标签一致 | 高置信错判会被重罚 |
| Brier Score | 概率的均方误差 | 概率整体偏离真实频率 |
| ECE | 分桶后平均校准误差 | 各置信区间系统性失真 |
| MCE | 最坏分桶误差 | 某个置信区间特别不准 |

举个初级工程师容易理解的例子。两个模型 A 和 B，AUC 都差不多，但：

- 模型 A：经常把错样本打成 0.95
- 模型 B：错样本一般只打成 0.65

两者排序可能接近，但 A 的 LogLoss 往往更差，因为“高置信错判”在策略层最危险。广告出价、库存分配、通知发送都会因为这种过度自信而被放大损失。

假设某 CTR 排序模型校准前后指标如下：

| 状态 | AUC | LogLoss | Brier |
|---|---:|---:|---:|
| 校准前 | 0.781 | 0.462 | 0.148 |
| 校准后 | 0.780 | 0.421 | 0.131 |

这类结果很常见：AUC 几乎不动，但 LogLoss 和 Brier 明显改善。解释很简单，校准没改变“谁排前面”，但修正了“模型到底有多确定”。

常见风险与缓解方式如下：

| 风险 | 表现 | 缓解措施 |
|---|---|---|
| 用训练集直接做校准 | 线下特别好，线上掉得快 | 使用独立校准集 |
| Isotonic 样本过少 | 曲线锯齿化，线上不稳定 | 样本少时优先 Platt 或 Temperature |
| 分布漂移 | 老参数越来越不准 | 定期重训，按时间窗监控 ECE |
| 混用不同模型分数 | 融合权重失真 | 每个模型独立校准到统一概率空间 |
| 把位置偏差当真实概率 | 顶部样本天然点击高 | 做去偏或在一致曝光条件下校准 |

---

## 替代方案与适用边界

如果数据量小，优先考虑 Platt Scaling 或 Temperature Scaling。原因很实际：参数少，稳定，过拟合风险低。如果数据量足够大，而且你观察到“原始分数和真实概率关系明显不是 S 形”，再考虑 Isotonic Regression。

| 方案 | 适用条件 | 数据要求 | 优点 | 局限 |
|---|---|---|---|---|
| Temperature Scaling | 有 logit，整体置信度偏热/偏冷 | 低到中 | 最简单，稳定 | 只能做整体缩放 |
| Platt Scaling | 二分类排序分数可单调映射 | 中 | 易实现，解释直观 | 形状受 sigmoid 限制 |
| Isotonic Regression | 非线性关系明显，样本多 | 高 | 灵活，拟合能力强 | 小样本易过拟合 |
| 分桶校准 | 需要快速上线、可解释性强 | 中 | 简单直接，便于监控 | 分桶边界敏感，不平滑 |
| 概率化树模型/贝叶斯方法 | 训练阶段就要概率输出 | 高 | 模型层面统一处理不确定性 | 实现复杂，改造成本高 |

真实工程中还有一个边界：如果下游只做“取 TopK”，完全不依赖概率阈值、收益估算、竞价或跨模型融合，那么校准的重要性会下降。但只要策略层要计算

$$
\mathbb{E}[\text{收益}] = \hat p \times \text{单次收益}
$$

校准就会直接影响业务。因为这里错的不是排序，而是期望值本身。

因此选择方法时不要问“哪种最先进”，而要问三个问题：

1. 我的下游是否真的依赖概率？
2. 我的校准集是否足够独立且样本充足？
3. 当前误差是“整体过热”，还是“局部形状不对”？

---

## 参考资料

1. StatsTest, *Calibration Checks: Brier Score and Reliability Diagrams*  
   适合建立基础概念，重点解释 Brier Score、可靠性图以及“排序好不等于概率准”。

2. MetricGate, *Class Probability Calibration Calculator*  
   对 Platt Scaling、Isotonic Regression、Temperature Scaling 的公式和差异做了直接说明，适合作为实现前的对照材料。

3. Elastic Search Labs, *Improve search results by calibrating model scoring in Elasticsearch*  
   重点在搜索与多模型融合场景，说明为什么不同检索器和 reranker 的分数必须先映射到统一概率空间。

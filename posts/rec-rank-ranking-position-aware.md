## 核心结论

位置感知排序要解决的核心问题是：点击不等于相关性。相关性指“这个结果本身是否满足用户需求”；位置偏差指“结果排得越靠前，越容易被用户看到和点击”。如果训练时直接把点击当标签，模型会把“排在前面带来的曝光优势”误学成“内容本身更相关”，最终形成“高位更容易被点，被点更多又更容易继续排高位”的反馈闭环。

更准确的做法，是把点击拆成两个概率：

$$
P(\text{click}\mid q,d,k)=P(\text{exam}\mid q,k)\times P(\text{rel}\mid q,d)
$$

其中，$q$ 是 query，白话解释就是“用户这次搜的词”；$d$ 是 document，也就是候选内容；$k$ 是 position，也就是展示位次。$P(\text{exam})$ 表示“用户有没有看到这个位置”，$P(\text{rel})$ 表示“看到了以后，这个结果是不是足够相关”。

工程上常见路线有四类：

| 方案 | 解决什么问题 | 训练时能否用位置特征 | 推理时能否用位置特征 |
| --- | --- | --- | --- |
| 加 position bias 特征做辅助建模 | 先显式承认位置影响存在 | 可以，但要和主排序目标隔离 | 通常不应直接用 |
| PBM / Examination Model | 分解“看见”和“相关” | 可以用于估计偏差 | 主排序一般不用 |
| IPS 去偏估计 | 用逆倾向加权校正标签或损失 | 只用于训练校正 | 不用 |
| 联合建模 $P(click)=P(exam)\times P(rel)$ | 同时估计位置偏差和真实相关性 | 可以 | 线上通常只保留 $P(rel)$ 或 ranker-only 特征 |

结论可以压缩成一句话：训练阶段可以建模位置，推理阶段不要依赖位置本身做最终排序。

---

## 问题定义与边界

排序模型的位置感知，讨论的不是“要不要把位置放进特征”，而是“如何避免点击日志把位置优势伪装成相关性”。这两个问题必须分开。

先看一个玩具例子。假设同一篇文章被放到不同位置：

| 文档 | 位次 | 曝光数 | 点击数 | 观测 CTR |
| --- | --- | ---: | ---: | ---: |
| A | 1 | 1000 | 250 | 25% |
| A | 5 | 1000 | 50 | 5% |

如果只看 CTR，结论会变成“位置 1 的 A 比位置 5 的 A 更相关”。但这是错误的，因为文档没有变，变的只是位次。真正变的是用户看见它的概率。

在最基础的 Position-Based Model，简称 PBM，可以先假设“是否被看见”主要由位置决定。比如：

| 位次 | 估计 $P(exam)$ | 观测 CTR | 估计 $P(rel)=CTR/P(exam)$ |
| --- | ---: | ---: | ---: |
| 1 | 0.9 | 25% | 27.8% |
| 5 | 0.3 | 5% | 16.7% |

这个表已经说明了问题：直接 CTR 明显受位置影响。即使做了简单除法，两个位置得到的相关性估计仍不完全相同，原因可能是样本噪声、query 差异、snippet 展示差异、用户信任偏差等。它至少告诉我们一件确定的事：点击标签本身带偏。

这里的边界也要讲清楚。

第一，位置感知排序主要解决隐式反馈去偏。隐式反馈就是“点击、停留时长、跳出”这类用户行为日志，而不是人工标注。

第二，它不能凭空制造真实标签。去偏只能减少错误归因，不能保证完全恢复真值。

第三，它通常只讨论“展示后被看到”这一层偏差，不自动解决“召回阶段没拿到候选”“摘要质量影响点击”“品牌词天然更容易被点”等其他偏差。

---

## 核心机制与推导

最常见的推导起点是点击分解：

$$
P(C=1\mid q,d,k)=P(E=1\mid q,k)\cdot P(R=1\mid q,d)
$$

其中：

- $C$ 是 click，表示是否点击
- $E$ 是 examination，表示是否看见
- $R$ 是 relevance，表示是否相关

白话理解很直接：用户只有先看到，才有机会点击；看到之后，是否点击主要由内容相关性决定。

### 1. 为什么直接用点击会错

如果训练目标是“预测点击”，而输入里又混入位次或位次强相关特征，模型学到的很可能是：

$$
\text{high click} \Rightarrow \text{high rank}
$$

但这不是“高相关导致高点击”，而可能只是“高位置导致高点击”。这会把排序系统带入自强化循环：

1. 旧模型先把某些结果放高位
2. 高位结果得到更多曝光和点击
3. 新模型把这些点击当成更高相关
4. 这些结果继续排更高

这就是推荐和搜索里常说的 rich get richer。

### 2. IPS 的基本思想

IPS 是 Inverse Propensity Scoring，白话解释就是“谁更难被看到，就给谁更大的补偿权重”。如果位置 $k$ 的被看见概率是 $p_k$，那么这个样本的权重可以写成：

$$
w_k=\frac{1}{p_k}
$$

如果某条样本出现在低位，它天然更难获得点击，那么它一旦被点中，就应该被赋予更高权重。训练损失可写成：

$$
L=\sum_i \frac{c_i}{p_{k_i}}\cdot \ell(f(q_i,d_i), 1)
$$

这里 $\ell$ 是损失函数，白话解释就是“模型预测错了要罚多少”；$f(q_i,d_i)$ 是排序打分函数。

### 3. PBM、EM 和更复杂模型的关系

PBM 假设最强，适合做第一层理解：位置影响 examination，文档影响 relevance。但真实系统往往更复杂：

- 用户对头部结果有“系统推荐应该更靠谱”的信任，这叫 trust bias，白话解释就是“用户会额外相信前几名”。
- 不同 query 的浏览深度不同，长尾 query 和导航型 query 的 examination 分布可能完全不同。
- 移动端屏幕更短，位置 4 和位置 8 的可见性差距比 PC 更大。

所以工程中常见两类扩展：

| 方法 | 额外考虑了什么 | 适用情况 |
| --- | --- | --- |
| EM 联合估计 | 同时估计位置偏差与相关性隐变量 | 标签稀疏、想做概率分解 |
| TrustPBM | 在位置偏差外加入用户信任效应 | 用户明显偏信顶部结果 |
| Query-level Propensity | 倾向概率依赖 query 或人群 | query 差异很大 |
| DualIPW | 除位置外还考虑点击噪声或饱和 | 移动端、一次只点很少结果 |

---

## 代码实现

下面给一个最小可运行的 Python 例子。它不依赖第三方库，演示三件事：

1. 如何统计每个位次的 examination propensity
2. 如何用 IPS 生成去偏标签
3. 为什么线上推理不能直接依赖位置特征

```python
from collections import defaultdict

# 日志样本: query, doc, position, clicked
logs = [
    ("python sort", "doc_a", 1, 1),
    ("python sort", "doc_b", 2, 0),
    ("python sort", "doc_c", 3, 0),
    ("python sort", "doc_a", 4, 0),
    ("python sort", "doc_d", 5, 0),

    ("ranking bias", "doc_e", 1, 1),
    ("ranking bias", "doc_f", 2, 1),
    ("ranking bias", "doc_a", 5, 1),

    ("ctr model", "doc_g", 1, 0),
    ("ctr model", "doc_a", 3, 1),
    ("ctr model", "doc_h", 5, 0),
]

# 假设通过随机化实验或外部估计，得到每个位次的 examination 概率
propensity = {
    1: 0.90,
    2: 0.75,
    3: 0.55,
    4: 0.40,
    5: 0.25,
}

# 用 IPS 估计每个 doc 的去偏相关性得分
weighted_click = defaultdict(float)
weighted_exposure = defaultdict(float)

for _, doc, pos, clicked in logs:
    w = 1.0 / propensity[pos]
    weighted_click[doc] += clicked * w
    weighted_exposure[doc] += w

debias_score = {}
for doc in weighted_exposure:
    debias_score[doc] = weighted_click[doc] / weighted_exposure[doc]

# 观测点击率，用于对比
raw_click = defaultdict(int)
raw_exp = defaultdict(int)

for _, doc, _, clicked in logs:
    raw_click[doc] += clicked
    raw_exp[doc] += 1

raw_ctr = {doc: raw_click[doc] / raw_exp[doc] for doc in raw_exp}

# doc_a 在低位也被点到，IPS 后得分应高于其原始 CTR
assert raw_ctr["doc_a"] == 0.5
assert debias_score["doc_a"] > raw_ctr["doc_a"]

# 位置 5 的一次点击权重应大于位置 1 的一次点击权重
assert (1 / propensity[5]) > (1 / propensity[1])

def rank_online(doc_features):
    # 线上推理只使用内容相关特征，不使用当前展示位次
    # 这里用一个假的线性分数示意
    return (
        0.6 * doc_features["title_match"]
        + 0.3 * doc_features["semantic_match"]
        + 0.1 * doc_features["freshness"]
    )

score = rank_online({
    "title_match": 0.9,
    "semantic_match": 0.8,
    "freshness": 0.5,
})
assert 0.0 <= score <= 1.0
```

这个例子故意把“估计 propensity”和“训练 ranker”拆开，因为工程上这通常就是两阶段。

一个更接近真实工程的例子，是推荐流里的文章排序。假设首页卡片天然有首屏优势，用户对前两条更容易浏览。如果你直接用点击训练 CTR ranker，那么早期靠运营顶上去的内容，会因为高曝光继续积累更多点击，最终长期霸榜。正确做法通常是：

| 步骤 | 做什么 | 目的 |
| --- | --- | --- |
| 日志采集 | 记录 query/user/doc/position/click | 保留偏差来源 |
| 倾向估计 | 用随机打散、小流量实验或点击模型估计 $P(exam)$ | 获得去偏基础 |
| 标签生成 | 用 IPS 或 EM 得到去偏相关信号 | 避免把位置当质量 |
| 排序训练 | 只输入 ranker-only 特征 | 让模型学内容而不是位置 |
| 线上服务 | 只用 query-doc 特征打分 | 避免重新引入偏差 |

这里的 ranker-only 特征，白话解释就是“线上无论排第几都能事先知道的内容特征”，例如标题匹配度、向量召回分、作者质量分、发布时间衰减等。

---

## 工程权衡与常见坑

位置感知排序的难点不在公式，而在工程细节。

| 常见坑 | 为什么会出错 | 规避方式 |
| --- | --- | --- |
| 直接用 CTR 当标签 | CTR 混合了位置与相关性 | 做 propensity 去偏或显式点击分解 |
| 训练时输入 position，线上也输入 position | 模型会把“当前排位”当成预测依据 | 训练可用于辅助估计，主 ranker 推理不直接用 |
| propensity 只按位次统计 | 忽略 query、端类型、人群差异 | 做 query-aware 或 context-aware propensity |
| 随机化流量太少 | 倾向估计方差大，低位样本不稳定 | 延长采样周期，做平滑和截断 |
| IPS 权重过大 | 低 propensity 导致训练爆炸 | 做 clipping，例如 $w=\min(1/p, w_{max})$ |
| 点击很稀疏时强行点式估计 | 标签噪声高，排序不稳定 | 用 pairwise/listwise 或联合 EM |
| 把摘要质量变化当相关性变化 | snippet 改动会显著影响点击 | 将展示样式特征单独建模 |

一个常被忽略的问题，是“训练能不能用位置特征”。答案不是简单的能或不能，而是要看用途。

- 如果位置特征用于估计 propensity 或点击分解子模型，可以。
- 如果位置特征直接进入最终线上排序打分，通常不行。
- 如果你的任务本质上是“预测点击概率而非预测相关性”，可以保留位置特征，但那不等于相关性排序模型。

很多团队在这里混淆了“CTR 预估”和“相关性排序”。CTR 预估关心的是“这个展示条件下会不会点”；相关性排序关心的是“这个内容本身该不该排前面”。前者可以依赖展示上下文，后者必须尽量排除展示偏差。

---

## 替代方案与适用边界

PBM + IPS 是最容易落地的起点，但不是所有场景都够用。

| 方案 | 关注点 | 数据要求 | 适用边界 |
| --- | --- | --- | --- |
| PBM + IPS | 位置偏差基础去偏 | 需要 position 日志和 propensity | 冷启动简单、可快速上线 |
| EM 联合估计 | 同时估计 examination 与 relevance | 需要较多日志 | 点击噪声较大时更稳 |
| TrustPBM | 用户对头部结果的额外信任 | 需要更细行为建模 | 搜索结果头部信任强 |
| DualIPW | 同时处理位置与其他曝光偏差 | 需要更复杂实验设计 | 移动端、点击饱和明显 |
| 随机化 / Interleaving | 直接用实验获取无偏对比 | 需要线上流量 | 高精度评估或校准阶段 |

适用边界也要说清楚。

第一，如果你的业务完全拿不到位置扰动数据，propensity 很难估准。这时可以先做轻量随机化收集样本，再上去偏学习。

第二，如果排序目标就是“当前布局下的点击最大化”，而不是“抽象相关性最优”，那么保留位置上下文是合理的，但要明确这不是无偏相关性学习。

第三，如果系统已经强依赖多目标优化，比如同时优化点击、时长、转化，那么位置感知只是其中一层校正，不会替代完整的目标设计。

对于初级工程师，最实用的判断标准是：只要训练标签来自点击日志，就默认存在位置偏差；只要模型上线时能看到“它自己排出来的位置”，就默认存在泄漏风险。

---

## 参考资料

- System Overflow, “How Do Click Models Separate Examination from Attractiveness?”  
  https://www.systemoverflow.com/learn/ml-search-ranking/search-relevance-feedback/how-do-click-models-separate-examination-from-attractiveness?utm_source=openai
- System Overflow, “What is Position Bias and Why Does It Distort Ranking Systems?”  
  https://www.systemoverflow.com/learn/ml-search-ranking/search-relevance-feedback/what-is-position-bias-and-why-does-it-distort-ranking-systems?utm_source=openai
- Google Research, “Position Bias Estimation for Unbiased Learning to Rank in Personal Search”  
  https://research.google/pubs/position-bias-estimation-for-unbiased-learning-to-rank-in-personal-search/?utm_source=openai
- Google Research, “Addressing Trust Bias for Unbiased Learning-to-Rank”  
  https://research.google/pubs/addressing-trust-bias-for-unbiased-learning-to-rank/?utm_source=openai
- ResearchGate, “Unbiased Learning to Rank with Query-Level Click Propensity Estimation”  
  https://www.researchgate.net/publication/389089959_Unbiased_Learning_to_Rank_with_Query-Level_Click_Propensity_Estimation_Beyond_Pointwise_Observation_and_Relevance?utm_source=openai

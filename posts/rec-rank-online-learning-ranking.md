## 核心结论

在线学习排序，是指排序模型在系统运行过程中直接利用实时反馈持续更新参数，而不是等一批新数据积累完再统一重训。白话说，它不是“先收作业、晚上统一批改”，而是“用户刚点完，模型立刻微调一点”。

它最适合两类场景：一类是用户兴趣变化快，比如短视频、新闻、搜索联想；另一类是供给和热点变化快，比如电商促销、库存变化、突发事件。此时离线模型的主要问题不是“不会排”，而是“来不及变”。

一个对新手最直观的例子是搜索重排。用户搜“蓝牙耳机”，系统先召回一批商品，再按当前排序器打分展示。若促销时段某款耳机点击和加购突然升高，在线学习排序可以只基于这批刚刚曝光且被点击的数据，马上把它往前提，而不是等第二天离线训练完成。

传统离线训练与在线增量更新的差异可以先看下面这张表：

| 维度 | 传统离线训练 | 在线增量更新 |
| --- | --- | --- |
| 数据来源 | 周期性汇总日志 | 实时曝光与反馈流 |
| 更新频率 | 小时级、天级 | 秒级、分钟级、请求级 |
| 优点 | 稳定、可控、易回放 | 响应快、适应漂移 |
| 缺点 | 对热点反应慢 | 容易波动、偏差更难处理 |
| 典型风险 | 训练数据过时 | 反馈延迟、位置偏差、灾难性遗忘 |

结论可以压缩成一句话：在线学习排序的收益通常来自“更快适应”，而它的成本主要来自“更难稳定”。在生产里，真正可用的方案通常不是完全在线替代离线，而是“离线底座 + 在线微调 + 风险控制”。

---

## 问题定义与边界

设查询为 $q$，候选文档或商品为 $d$，排序模型参数为 $\theta$，用户反馈为 $y$。在线学习排序要解决的问题是：在没有人工预标注标签的情况下，仅根据实时反馈更新 $\theta$，让未来排序更符合用户偏好。

可以把单次训练目标写成：

$$
L(\theta; (q, d), y)
$$

其中 $L$ 是损失函数。损失函数就是“模型当前排法有多差”的量化规则。若是点击建模，常见做法是把点击看成二分类反馈；若是排序优化，也可以用 pairwise loss。pairwise 的意思是“不直接判断绝对分数，而是判断文档 A 是否应排在文档 B 前面”。

在线更新的边界比离线训练严格得多：

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t; (q_t, d_t), y_t)
$$

但这里的 $(q_t, d_t, y_t)$ 不是任意样本，而只能来自“当前真的被曝光过”的结果。系统看不到未曝光候选的真实反馈，因此天然存在选择偏差。选择偏差就是“你只能观察自己展示过的东西，所以数据本身已经被旧策略筛过一次”。

新手版本可以这样理解：排序器像一个边走边调整地图的导航，它每次只能看到自己刚走过的路，没走过的路没有反馈，所以不能假装自己知道全局最优。

因此，在线学习排序至少要面对三个约束：

| 约束 | 含义 | 后果 |
| --- | --- | --- |
| 只见曝光数据 | 未展示结果没有反馈 | 训练样本不完整 |
| 位置偏差 | 高位更容易被点 | 点击不等于真实相关性 |
| 反馈延迟 | 转化、停留时长可能晚到 | 更新目标和真实收益不同步 |

所以“收到点击就立刻更新”只是最外层动作，真正困难的是：你收到的反馈并不干净，也不完整。

---

## 核心机制与推导

最基础的在线更新是增量 SGD。SGD 是随机梯度下降，白话说就是“每来一个样本，沿着让损失变小的方向走一小步”。

其标准形式为：

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t; (q_t, d_t), y_t)
$$

如果直接这样用点击做更新，会遇到一个大问题：高位结果更容易被看到，也更容易被点击，因此点击包含位置偏差。位置偏差就是“用户先看到上面的，所以上面的点击天然占便宜”。

常见修正方式是重要性加权，英文常写 IPS，即 Inverse Propensity Scoring。propensity 可以理解为“被看到的概率”。若某个位置本来就很容易被看到，那么它的点击证据应当打折；若某个位置很难被看到但仍然被点，它的证据应当更重。写成式子是：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot w_t \cdot \nabla_\theta L(\theta_t; (q_t, d_t), y_t)
$$

其中 $w_t$ 是重要性权重。一个简单近似是 $w_t = \frac{1}{p(o_t=1 \mid \text{position})}$，即曝光概率的倒数。

研究摘要里给出的玩具例子很适合入门：

- 当前参数 $\theta_t = 0.5$
- 学习率 $\eta_t = 0.01$
- 当前样本梯度为 $2$

如果不做偏差校准：

$$
\theta_{t+1} = 0.5 - 0.01 \times 2 = 0.48
$$

如果该点击来自一个高偏置位置，只给权重 $w = 0.5$，则等效梯度变成 $2 \times 0.5 = 1$：

$$
\theta_{t+1} = 0.5 - 0.01 \times 1 = 0.49
$$

这表示同样一个点击，在校准后更新更保守，因为系统承认“这个点击有一部分是位置带来的，不全是相关性”。

数值过程可以压成一张表：

| 步骤 | 数值 | 含义 |
| --- | --- | --- |
| 初始参数 | $\theta_t = 0.5$ | 当前模型状态 |
| 学习率 | $\eta_t = 0.01$ | 本轮更新步长 |
| 原始梯度 | $g = 2$ | 当前样本给出的调整方向 |
| 无校准更新 | $0.5 - 0.01 \times 2 = 0.48$ | 完全相信该反馈 |
| 加权后梯度 | $g' = 2 \times 0.5 = 1$ | 承认位置偏差 |
| 校准后更新 | $0.5 - 0.01 \times 1 = 0.49$ | 更保守 |

真实工程里还会再加一层“时间衰减”或“滑动窗口”。滑动窗口就是只保留最近一段数据。白话说，它像一本只记最近 10 笔交易的账本，旧记录不会永久支配现在。这样做的原因是在线排序通常服务于非平稳环境，即数据分布会变，旧样本可能已经失效。

一个真实工程例子是电商促销排序。平时模型可能更重视长期转化率，但大促期间，库存、活动价、主推款都在快速变化。如果还是用昨天的统计量支配今天的排序，系统会明显迟钝。在线增量更新加上滑动窗口，可以让“最近几小时的点击和成交”更快影响权重，但仍保留离线主模型作为安全底座。

---

## 代码实现

下面给一个可运行的极简 Python 版本，演示三个核心模块：增量更新、重要性加权、滑动窗口。这里用逻辑回归式点击预测做例子，目标不是工业强度，而是让机制能跑通。

```python
from collections import deque
import math

class OnlineRanker:
    def __init__(self, dim, lr=0.1, window_size=5):
        self.theta = [0.0] * dim
        self.lr = lr
        self.buffer = deque(maxlen=window_size)
        self.backup_theta = list(self.theta)

    def score(self, x):
        return sum(w * xi for w, xi in zip(self.theta, x))

    def predict_prob(self, x):
        s = self.score(x)
        return 1.0 / (1.0 + math.exp(-s))

    def update(self, x, click, position_bias_weight):
        # logistic loss gradient: (p - y) * x
        p = self.predict_prob(x)
        grad_scale = (p - click) * position_bias_weight

        for i in range(len(self.theta)):
            self.theta[i] -= self.lr * grad_scale * x[i]

        self.buffer.append((x, click, position_bias_weight))
        return p

    def rollback(self):
        self.theta = list(self.backup_theta)

    def checkpoint(self):
        self.backup_theta = list(self.theta)

ranker = OnlineRanker(dim=2, lr=0.1, window_size=3)

# 玩具样本：特征 x，点击 y，位置校准权重 w
samples = [
    ([1.0, 0.0], 1, 0.5),
    ([0.0, 1.0], 0, 1.0),
    ([1.0, 1.0], 1, 0.8),
]

ranker.checkpoint()
for x, y, w in samples:
    ranker.update(x, y, w)

# 参数应已发生更新
assert ranker.theta != [0.0, 0.0]

# 概率应在 0 到 1 之间
p = ranker.predict_prob([1.0, 1.0])
assert 0.0 < p < 1.0

# buffer 只保留最近 window_size 条
assert len(ranker.buffer) <= 3
```

核心流程就是这几步：

```python
grad = compute_grad(theta, query, doc)
iw = position_weight(position)
theta -= eta * grad * iw
buffer.append((query, doc, feedback))
```

各部分输入输出可以再对照一下：

| 模块 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| 打分 | query/doc 特征 | score | 当前排序依据 |
| 梯度计算 | score 与 click | grad | 告诉参数往哪改 |
| 重要性加权 | 位置或曝光概率 | iw | 修正点击偏差 |
| 参数更新 | grad、iw、lr | 新 $\theta$ | 完成在线学习 |
| buffer | 最新反馈流 | 近期样本集合 | 控制记忆范围 |
| checkpoint/rollback | 当前参数快照 | 可恢复状态 | 防止线上劣化 |

真实工程不会只维护一个向量。通常还会有：
- 离线大模型作为 warm-start 初值。warm-start 就是“在线模型不是从零开始，而是从一个已经可用的离线模型继续学”。
- 实时特征服务，用于拿最新库存、价格、上下文。
- 监控与回滚，一旦 CTR、GMV、投诉率异常，自动切回备份模型。

---

## 工程权衡与常见坑

在线学习排序最容易被低估的，不是“怎么更新”，而是“什么时候不该更新”。更新太快，模型会抖；更新太慢，又失去在线价值。

常见坑与规避策略如下：

| 坑 | 具体表现 | 常见规避策略 |
| --- | --- | --- |
| 反馈延迟 | 点击快到，转化晚到 | 分层目标，点击实时更新，转化做延迟补账 |
| 位置偏差 | 高位天然更容易被点 | IPS、self-normalized IPS、随机化估计 |
| 选择偏差 | 只学到旧策略看过的样本 | 安全探索、bandit 重排、反事实估计 |
| 灾难性遗忘 | 新热点一来，老偏好全忘 | 滑动窗口 + 保留高质量样本 + 周期性 warm-start |
| 更新抖动 | 指标分钟级大起大落 | 自适应学习率、最小流量门槛、灰度发布 |
| 坏数据放大 | 埋点异常导致错误更新 | 校验、去重、异常阈值、快速回滚 |

“滑动窗口像只记最近 10 条交易的记账本”这个比喻在工程上很实用，但要补一句更准确的话：窗口不是越小越好。窗口太小会导致估计方差变大，系统对偶然点击过度敏感；窗口太大又会跟不上漂移。实际常用做法是让窗口长度、学习率、回滚阈值一起调。

还有一个常见误区是把在线排序理解成“点击越多的永远升得越高”。这会形成反馈回路。反馈回路就是“模型把某类内容排得更前，这类内容因曝光更多又得到更多点击，最后被误认为更好”。如果没有探索与去偏，这种回路会把热门项越推越热，把长尾项彻底压死。

---

## 替代方案与适用边界

在线学习排序不是默认最优方案，它只是对“变化快”这类问题更合适。若反馈稀疏、探索代价高、业务风险大，其他方案可能更稳。

| 方案 | 适用场景 | 限制 |
| --- | --- | --- |
| 离线批量更新 | 数据稳定、指标重可控 | 对热点和漂移反应慢 |
| 在线增量 LR/FTRL | 实时点击多、特征简单、更新需快 | 偏差处理和稳定性要求高 |
| Bandit 排序 | 需要探索新结果、平衡试错与收益 | 实现复杂，线上干预更敏感 |
| 强化学习排序 | 多步长期收益显著，如会话级优化 | 状态建模复杂，训练不稳定 |
| 在线校准 | 主模型稳定，只需修正分数偏差 | 只能局部修补，提升上限有限 |
| 周期性 warm-start | 想保留在线适应，又不想全时更新 | 对极快热点仍可能滞后 |

Bandit 可以理解为“带探索约束的在线决策器”。它不是只按当前最好结果展示，而是允许少量受控试错，用来发现新优项。BubbleRank 这类方法的重点不是“探索更多”，而是“安全探索”，即尽量不把明显差的结果直接顶到前面。

在线排序不适合的典型边界也很明确：
- 反馈天然延迟很长，比如人工审核、金融授信、B2B 长周期成交。
- 曝光极少，无法支撑稳定增量更新。
- 错误探索成本很高，比如医疗决策、高风险推荐。
- 业务更关心长期约束而不是短期点击，比如公平性、合规性、品牌安全。

对新手来说，可以记一个简单判断标准：如果你拿到的是“几乎实时、量足够、噪声可控”的反馈，在线学习排序值得做；如果你拿到的是“很慢、很稀、很贵”的反馈，优先考虑离线再训练或在线校准。

---

## 参考资料

- Shengyao Zhuang, Guido Zuccon. *Counterfactual Online Learning to Rank* (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7148247/
  作用：给出反事实在线排序框架，核心是用去偏估计处理点击反馈中的位置与选择偏差。

- Shengyao Zhuang, Zhihao Qiao, Guido Zuccon. *Reinforcement online learning to rank with unbiased reward shaping* (2022). https://doi.org/10.1007/s10791-022-09413-y
  作用：把强化学习引入在线排序，并用无偏 reward shaping 处理位置偏差。

- Baharan Mirzasoleiman et al. *BubbleRank: Safe Online Learning to Re-Rank via Implicit Click Feedback* (UAI 2019). https://sites.ualberta.ca/~szepesva/papers/UAI2019-BubbleRank.pdf
  作用：强调安全探索，在尽量不伤害线上体验的前提下做 bandit 式重排。

- Next Electronics. *Reranking Systems That Learn from Feedback Streams*. https://www.next.gr/ai/supervised-learning/reranking-systems-that-learn-from-feedback-streams
  作用：工程视角总结了实时反馈流、增量 SGD、偏差校准和系统架构，适合快速建立全局概念。

- Vladimir Braverman, Chen Wang, Liudeng Wang, Samson Zhou. *Online Learning with Recency: Algorithms for Sliding-window Streaming Multi-armed Bandits* (Submitted to ICLR 2026). https://openreview.net/forum?id=XuQfB6RqSd
  作用：从理论上研究滑动窗口 bandit 与有限内存下的 recent-data 学习。需要注意，这是一篇提交到 ICLR 2026 的论文，当前公开状态为 submitted，不应表述为已正式录用。

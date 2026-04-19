## 核心结论

冷启动 Bandit 是一种在线决策方法，用来解决“新物品缺少历史反馈，但又必须获得少量曝光验证质量”的问题。它的目标不是一次性找到最优物品，而是在有限曝光里尽快识别值得放量的新物品，同时控制试错成本。

一个玩具例子：新文章刚上线，系统只有 1 次曝光机会。如果直接全量推荐，可能把低质量文章推给大量用户；如果完全不推荐，它永远没有点击数据，也永远无法进入排序系统。Bandit 的作用是把这件事写成可计算的策略：给新物品分配受控探索流量，观察反馈，再逐步调整曝光。

核心指标通常不是单轮点击率，而是累计奖励和后悔值 `regret`。奖励是每次选择后得到的反馈，例如点击为 1、未点击为 0；后悔值是“当前策略相比总是选择最优动作少拿了多少收益”。

常见方法的差别如下：

| 方法 | 探索方式 | 是否使用特征 | 冷启动适配度 | 优点 | 缺点 |
|---|---:|---:|---:|---|---|
| `ε-Greedy` | 以概率 `ε` 随机探索 | 否 | 中 | 实现简单，容易上线 | 探索不区分对象，可能浪费流量 |
| `UCB` | 均值 + 不确定性奖励 | 否 | 中高 | 自动优先试曝光少的物品 | 对非平稳环境敏感 |
| `Thompson Sampling` | 按后验采样 | 否或可扩展 | 高 | 随机性自然，线上表现稳定 | 需要维护概率分布 |
| `LinUCB` | 线性预测 + 不确定性 | 是 | 高 | 能用特征迁移，适合新物品 | 依赖特征质量和线性假设 |

真实工程例子：资讯流中新文章刚上线时，点击历史为 0，但标题、作者、类目、发布时间、文本 Embedding 已知。系统可以先用 `LinUCB` 给少量流量，通过用户点击持续更新参数；当曝光足够后，再把文章交给常规排序模型或召回系统做大规模分发。

---

## 问题定义与边界

Bandit 问题不是离线排序问题。离线排序是在已有样本上训练一个模型，再对候选集打分；Bandit 是在线决策，每一轮只能选择一个或少数几个物品，反馈逐步到来，下一轮策略会受到上一轮反馈影响。

冷启动也不是“完全没有数据”。更准确的说法是：单个新物品缺少历史行为数据，但系统可能仍然拥有上下文特征、内容特征、类目先验或相似物品经验。

以资讯流为例，一篇新文章刚上线：

| 输入 | 示例 |
|---|---|
| 物品历史反馈 | 点击数 0，曝光数 0 |
| 内容特征 | 标题、正文、作者、类目、Embedding |
| 用户特征 | 年龄段、兴趣标签、历史点击主题 |
| 场景特征 | 时间、频道、设备、位置 |
| 输出 | 当前轮是否给这篇文章曝光，以及给多少探索流量 |

常用符号如下：

| 符号 | 含义 |
|---|---|
| $t$ | 第 $t$ 轮决策 |
| $a_t$ | 第 $t$ 轮选择的动作或物品 |
| $r_t$ | 第 $t$ 轮获得的奖励，例如点击为 1 |
| $N_a(t)$ | 到第 $t$ 轮前物品 $a$ 被选择的次数 |
| $\hat \mu_a(t)$ | 物品 $a$ 的历史平均奖励 |
| $x_{t,a}$ | 第 $t$ 轮物品 $a$ 的上下文特征 |
| $regret$ | 当前策略相比最优策略损失的累计收益 |

一个标准目标可以写成：

$$
R_T = \sum_{t=1}^{T} r_t
$$

最大化累计奖励 $R_T$，等价于在很多设定下最小化后悔值：

$$
Regret(T) = \sum_{t=1}^{T}(\mu^* - \mu_{a_t})
$$

其中 $\mu^*$ 表示最优物品的真实期望收益，$\mu_{a_t}$ 表示当前选择物品的真实期望收益。真实期望收益在上线时不可直接观察，只能通过逐步试探估计。

---

## 核心机制与推导

`ε-Greedy` 是最直接的策略：大部分时候选择当前平均收益最高的物品，小部分时候随机选一个物品。`ε` 是探索率，表示随机探索的概率。

$$
a_t =
\begin{cases}
\text{random arm}, & \text{with probability } \epsilon_t \\
\arg\max_a \hat\mu_a(t), & \text{with probability } 1-\epsilon_t
\end{cases}
$$

它的问题是探索没有方向。一个已经试过很多次但表现差的物品，和一个完全没试过的新物品，在随机探索时可能被同等对待。

`UCB` 的核心是“不确定性奖励”。它认为物品分数不只看历史均值，还要看估计是否可靠。曝光次数越少，置信区间越宽，探索加成越大。

$$
a_t = \arg\max_a \left[\hat \mu_a(t) + \sqrt{\frac{2\ln t}{N_a(t)}}\right]
$$

其中 $\sqrt{\frac{2\ln t}{N_a(t)}}$ 是探索项。$N_a(t)$ 越小，探索项越大，新物品更容易被试到。

数值例子：有 3 个物品，当前 $t=100$。

| 物品 | 平均奖励 $\hat\mu$ | 曝光次数 $N$ | UCB1 近似分数 |
|---|---:|---:|---:|
| a1 | 0.26 | 50 | $0.26+\sqrt{2\ln100/50}\approx0.69$ |
| a2 | 0.18 | 20 | $0.18+\sqrt{2\ln100/20}\approx0.86$ |
| a3 | 0.00 | 1 | $0.00+\sqrt{2\ln100/1}\approx3.03$ |

虽然 `a3` 目前点击率是 0，但它只曝光过 1 次，不确定性极高，所以会被优先探索。这就是 UCB 对冷启动物品友好的原因。

`Thompson Sampling` 是后验采样方法。后验是“观察到当前数据后，参数可能取什么值的概率分布”。它不直接给每个物品加探索项，而是从每个物品的收益分布中抽样，选择本轮抽样结果最好的物品。

$$
\tilde\theta_t \sim p(\theta \mid D_t)
$$

$$
a_t = \arg\max_a \mathbb{E}[r \mid a, \tilde\theta_t]
$$

对点击这种 0/1 奖励，常用 Beta-Bernoulli 建模。假设物品 $a$ 的点击率服从：

$$
\theta_a \sim Beta(\alpha_a, \beta_a)
$$

点击后更新 $\alpha_a \leftarrow \alpha_a + 1$，未点击后更新 $\beta_a \leftarrow \beta_a + 1$。曝光少的物品分布更宽，抽到高值的概率仍然存在，因此会自然获得探索机会。

`LinUCB` 属于 Contextual Bandit。上下文是指决策时可见的特征，例如用户、物品、场景信息。它假设奖励可以近似由特征线性解释：

$$
r_{t,a} = x_{t,a}^T\theta^* + \eta_t
$$

其中 $x_{t,a}$ 是特征向量，$\theta^*$ 是未知参数，$\eta_t$ 是噪声。LinUCB 的分数为：

$$
score(a)=x_{t,a}^T\hat\theta_t+\alpha\sqrt{x_{t,a}^TA_t^{-1}x_{t,a}}
$$

$$
A_t=\lambda I+\sum x x^T,\quad b_t=\sum r x,\quad \hat\theta_t=A_t^{-1}b_t
$$

第一项是利用，表示当前模型预测收益；第二项是探索，表示当前特征方向的不确定性。它的优势是可以从旧物品迁移到新物品：如果新文章和过去高点击文章在特征上相似，即使自身点击历史为 0，也能获得较合理的初始分数。

机制对比：

| 方法 | 评分或选择函数 | 探索来源 |
|---|---|---|
| `ε-Greedy` | 随机或 $\arg\max_a \hat\mu_a$ | 固定随机概率 |
| `UCB1` | $\hat\mu_a+\sqrt{2\ln t/N_a}$ | 曝光少带来的不确定性 |
| `Thompson Sampling` | $\arg\max_a \mathbb{E}[r\mid\tilde\theta]$ | 后验分布采样 |
| `LinUCB` | $x^T\hat\theta+\alpha\sqrt{x^TA^{-1}x}$ | 特征方向的不确定性 |

---

## 代码实现

实现时应先抽象统一接口，不要把探索逻辑散落在业务代码里。最小接口通常包括 `select_arm()` 和 `update()`：前者根据当前状态选择物品，后者根据反馈更新统计量。

伪代码如下：

```text
初始化每个物品的统计量
for 每一轮请求:
    读取候选物品
    调用 select_arm(candidates) 选择物品
    展示给用户
    收到点击、转化或停留时长反馈
    调用 update(arm, reward) 更新统计量
```

下面是一个可运行的 `UCB1` 最小实现：

```python
import math

class UCB1:
    def __init__(self, arms):
        self.arms = list(arms)
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}
        self.total = 0

    def select_arm(self):
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm

        scores = {}
        for arm in self.arms:
            mean = self.values[arm]
            bonus = math.sqrt(2 * math.log(self.total) / self.counts[arm])
            scores[arm] = mean + bonus
        return max(scores, key=scores.get)

    def update(self, arm, reward):
        self.total += 1
        n = self.counts[arm]
        old = self.values[arm]
        self.counts[arm] = n + 1
        self.values[arm] = old + (reward - old) / (n + 1)

bandit = UCB1(["new_article", "old_article"])
assert bandit.select_arm() == "new_article"
bandit.update("new_article", 1)
assert bandit.counts["new_article"] == 1
assert bandit.values["new_article"] == 1.0
```

`LinUCB` 的实现重点是维护矩阵 $A$ 和向量 $b$。每次选择时计算 $\hat\theta=A^{-1}b$，每次反馈后执行：

$$
A \leftarrow A + xx^T,\quad b \leftarrow b + rx
$$

简化代码片段如下：

```python
import numpy as np

class LinUCB:
    def __init__(self, dim, alpha=1.0, l2=1.0):
        self.alpha = alpha
        self.A = l2 * np.eye(dim)
        self.b = np.zeros(dim)

    def score(self, x):
        inv_A = np.linalg.inv(self.A)
        theta = inv_A @ self.b
        exploit = x @ theta
        explore = self.alpha * np.sqrt(x @ inv_A @ x)
        return float(exploit + explore)

    def update(self, x, reward):
        self.A += np.outer(x, x)
        self.b += reward * x

model = LinUCB(dim=2)
x = np.array([1.0, 0.5])
before = model.score(x)
model.update(x, 1.0)
after = model.score(x)
assert model.A.shape == (2, 2)
assert model.b.shape == (2,)
assert after != before
```

真实系统里还要考虑缓存和延迟更新。点击反馈可能几秒后到达，转化反馈可能几小时后到达，不能假设展示后立刻拿到奖励。常见做法是把曝光日志、反馈日志和策略版本号一起写入消息队列，再由流式任务异步更新 Bandit 状态。

---

## 工程权衡与常见坑

固定 `ε` 容易出问题。`ε` 太大，系统长期把流量浪费在低质量物品上；`ε` 太小，新物品拿不到足够曝光。常见策略是让 `ε_t` 随时间衰减：

$$
\epsilon_t = \max(\epsilon_{min}, \frac{c}{\sqrt{t}})
$$

其中 $\epsilon_{min}$ 是最小探索率，用来保证系统长期仍能发现新变化。

位置偏差是推荐系统最常见的坑。高位曝光的文章点击更多，不一定代表文章更好，可能只是因为它出现在第一个位置。若直接用点击率更新 Bandit，算法会把“位置优势”误判成“内容质量”。处理方式包括随机流量校正、位置归一化、逆倾向得分加权和固定实验桶。

| 问题 | 后果 | 处理方式 |
|---|---|---|
| 位置偏差 | 高位物品被误判为更优 | 随机桶、去偏点击率、IPS 加权 |
| 延迟反馈 | 更新使用了不完整奖励 | 设置归因窗口，区分点击和转化 |
| 时变环境 | 昨天的最优今天失效 | 滑动窗口、时间折扣、周期重置 |
| 特征泄漏 | 离线效果虚高，线上失效 | 禁用未来特征，按时间切分验证 |
| 稀疏无效特征 | `LinUCB` 接近随机 | 做特征筛选、降维、Embedding 预训练 |
| 流量过小 | 统计波动大 | 合并类目先验，延长探索期 |

另一个常见问题是奖励定义不稳定。点击、停留、收藏、下单都可以作为奖励，但它们优化方向不同。资讯场景只看点击，可能制造标题党；电商场景只看下单，反馈又太慢。工程上通常会定义加权奖励，例如：

$$
r = 1.0 \cdot click + 3.0 \cdot favorite + 10.0 \cdot purchase
$$

但权重不是越复杂越好。奖励越复杂，越需要离线回放、A/B 实验和业务约束校验。

---

## 替代方案与适用边界

不是所有冷启动问题都必须上 Bandit。如果流量极少、反馈极慢、特征极差，规则策略或离线预热可能更稳。

例如新品类刚上线，只有类目和标题，没有可靠内容特征，也没有相似历史物品。此时直接用 `LinUCB` 可能只是把噪声写进矩阵。更稳的做法是先按规则分流：每个新品给固定小流量，积累基础曝光和人工质量分，再切到 `UCB`、`Thompson Sampling` 或排序模型。

不同方案边界如下：

| 方法 | 适用边界 | 优点 | 缺点 | 落地成本 |
|---|---|---|---|---:|
| `Random` | 极早期验证、候选很少 | 无偏，简单 | 收益差 | 低 |
| `Round Robin` | 需要平均曝光 | 公平，易解释 | 不看反馈 | 低 |
| `Greedy` | 历史数据充分 | 收益稳定 | 新物品起不来 | 低 |
| `ε-Greedy` | 快速加探索 | 简单可控 | 探索粗糙 | 低 |
| `UCB` | 物品数适中，反馈快 | 自动照顾低曝光物品 | 不用特征 | 中 |
| `TS` | 点击、转化等概率反馈 | 探索自然，效果稳 | 分布建模更复杂 | 中 |
| `LinUCB` | 特征可迁移，冷启动明显 | 新物品启动快 | 依赖特征和线性假设 | 中高 |

纯探索策略适合物品少、特征弱、实现简单的场景；上下文 Bandit 适合用户、物品、场景特征比较充分，并且这些特征能迁移历史经验的场景。

最终落地时，Bandit 往往不是单独存在。更常见的结构是：召回系统给出候选，排序模型给出基础分，Bandit 只在新物品、小流量实验桶或重排阶段介入。这样既能控制风险，又能让冷启动物品获得必要的探索机会。

---

## 参考资料

1. [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352)  
Auer、Cesa-Bianchi、Fischer，2002。本文使用其中的 UCB1 思想和有限时间 regret 分析结论。

2. [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://www.microsoft.com/en-us/research/publication/a-contextual-bandit-approach-to-personalized-news-article-recommendation-3/)  
Li、Chu、Langford、Schapire，2010。本文使用其新闻推荐中的 Contextual Bandit 建模方式。

3. [Contextual Bandits with Linear Payoff Functions](https://proceedings.mlr.press/v15/chu11a.html)  
Chu、Li、Reyzin、Schapire，2011。本文使用其中线性收益假设和 LinUCB 形式。

4. [Analysis of Thompson Sampling for the Multi-armed Bandit Problem](https://proceedings.mlr.press/v23/agrawal12.html)  
Agrawal、Goyal，2012。本文使用其中 Thompson Sampling 的后验采样视角。

5. [Bandit Algorithms](https://www.cambridge.org/core/books/bandit-algorithms/contents/647794008F1BA21CD64EEC14A2527196)  
Lattimore、Szepesvári，2020。本文参考其对 Bandit 问题、regret 和算法边界的系统化整理。

6. [Reinforcement Learning: An Introduction](https://incompleteideas.net/book/the-book.html)  
Sutton、Barto。本文参考其对多臂老虎机问题与探索-利用权衡的基础解释。

推荐阅读顺序：先读 Sutton & Barto 的 bandit 章节建立直觉，再读 UCB1 原始论文理解不确定性奖励，然后读新闻推荐论文理解工程场景；LinUCB 和 Thompson Sampling 的理论论文可在实现对应算法时回看。

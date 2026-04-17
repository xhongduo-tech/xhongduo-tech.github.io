## 核心结论

Bandit 算法解决的是推荐系统里的“探索与利用”矛盾。探索，白话说就是主动试新内容；利用，白话说就是继续推当前看起来最能点开的内容。它不是离线训练一个固定模型再长期上线，而是在每次推荐后立刻用反馈修正下一次决策。

最常见的三类方法可以直接对应三种工程思路：

1. $\varepsilon$-Greedy：以 $\varepsilon$ 的概率随机探索，以 $1-\varepsilon$ 的概率选择当前估值最高的物品。优点是实现最简单，缺点是探索不够“聪明”。
2. UCB：在经验均值上加一个“不确定性奖励”。被看得少的内容虽然当前均值不高，但因为信息不足，会得到额外加分。
3. Thompson Sampling：把每个物品的点击率看成一个概率分布而不是一个固定数，每轮从分布里采样，再选采样值最大的物品。它会在“不确定但可能很强”的物品上自然多试几次。

如果只需要一个新手可落地的判断标准，可以直接用下面这张表：

| 方法 | 决策逻辑 | 优点 | 主要缺点 | 适合场景 |
|---|---|---|---|---|
| $\varepsilon$-Greedy | 随机探索 + 当前最优利用 | 最容易实现和解释 | 探索很粗糙，容易浪费流量 | 小规模推荐、冷启动基线 |
| UCB | 均值 + 置信加成 | 探索更有针对性 | 需要调探索系数，遇到分布漂移要小心 | 候选数量有限、奖励较稳定 |
| Thompson Sampling | 从后验分布采样 | 通常收敛快，参数少 | 需要维护后验，解释性略弱 | CTR 推荐、广告、内容流 |
| Contextual Bandit | 把用户/场景特征也纳入决策 | 能做个性化探索 | 实现复杂度明显更高 | 用户特征丰富的推荐系统 |

对推荐系统来说，Bandit 的真正价值不是“某个公式更高级”，而是它让系统具备在线持续学习能力。新内容上线、新用户进入、热点变化时，系统不会永远被旧热门锁死。

---

## 问题定义与边界

先把问题说清楚。Bandit 适合下面这类任务：

- 每一轮只能从有限个候选里选一个或几个。
- 很快能拿到反馈，比如点击、停留、转化、是否播放完成。
- 目标是让一段时间内的累计收益最大，而不是只看单次预测精度。

形式化地说，设有 $K$ 个候选臂，臂可以理解为“可被推荐的内容”。第 $t$ 轮选择动作 $a_t$，得到奖励 $r_t \in [0,1]$。推荐里常见的二元奖励就是点击记为 1，未点击记为 0。目标是最大化累计奖励：

$$
\max \sum_{t=1}^{T} r_t
$$

另一个核心量是 regret，中文常译为“遗憾值”。白话说，就是你和“如果一开始就知道最优内容是谁”之间损失了多少收益：

$$
R_T = T\mu^* - \sum_{t=1}^{T}\mu_{a_t}
$$

其中 $\mu^*$ 是最优臂的真实期望奖励。

这说明 Bandit 不是做“绝对正确预测”，而是在不确定下尽快逼近最优策略。

边界也必须说清楚，否则很容易误用：

| 维度 | 典型问题 | Bandit 是否适合 | 处理边界 |
|---|---|---|---|
| 奖励是否可量化 | 点击、播放完成率、是否购买 | 适合 | 奖励定义必须统一，避免一会儿看 CTR 一会儿看 GMV |
| 反馈是否及时 | 点击几秒内返回 | 适合 | 延迟太长时要做代理指标或延迟更新 |
| 是否有强上下文 | 用户画像、时间、设备、频道 | 适合升级成 Contextual Bandit | 无上下文时只能学“全局最优”，个性化能力弱 |
| 内容是否持续变化 | 新内容不断上线 | 很适合 | 需要强制探索，否则新内容永远没机会 |
| 是否要求长期序列优化 | 当前推荐会影响后续很多步 | 只部分适合 | 多步长期价值更适合强化学习，不是标准 Bandit |

玩具例子可以直接看一个新用户冷启动场景。首页有 5 篇文章，真实 CTR 分别是 $\{30\%,25\%,20\%,15\%,10\%\}$，但系统一开始并不知道。若完全利用，系统可能因为前几次偶然点击，把 25% 的文章误判成最好，后面长期错推。若完全探索，流量又浪费太多。Bandit 的作用就是在“尽快赚钱”和“尽快学会”之间找平衡。

真实工程例子更接近资讯流推荐。新内容每天大量进入候选池，旧热门有稳定点击，新内容几乎没有曝光。如果只按离线 CTR 排序，历史强内容会一直占据前排，新内容缺样本，估计值永远上不来，形成“曝光少 -> 数据少 -> 排名低 -> 更没曝光”的闭环。Bandit 就是用受控探索打破这个闭环。

---

## 核心机制与推导

### 1. $\varepsilon$-Greedy

定义最直接。记 $\hat{\mu}_a(t)$ 为第 $t$ 轮前对物品 $a$ 的经验均值估计，则：

$$
P(a_t=a)=
\begin{cases}
1-\varepsilon + \frac{\varepsilon}{K}, & a=\arg\max_j \hat{\mu}_j(t) \\
\frac{\varepsilon}{K}, & \text{otherwise}
\end{cases}
$$

白话解释：大多数时候用当前第一名，少数时候随机试别人。

它的核心问题也很明确。即使某个内容已经明显很差，它仍然会在探索阶段被均匀抽到。这种“平均主义探索”实现简单，但效率不高。

### 2. UCB1

UCB 是 Upper Confidence Bound，中文可理解为“置信上界”。白话说，就是不只看当前平均分，还看“它还有没有可能被低估”。

经典 UCB1 选择：

$$
a_t=\arg\max_a \left(\hat{\mu}_a(t) + c\sqrt{\frac{\ln t}{n_a(t)}}\right)
$$

其中：

- $\hat{\mu}_a(t)$：当前经验均值
- $n_a(t)$：该物品已被推荐的次数
- $t$：总轮数
- $c$：探索强度系数

为什么它会自动减少探索？因为探索项是：

$$
c\sqrt{\frac{\ln t}{n_a(t)}}
$$

当某个物品被看得越来越多时，$n_a(t)$ 变大，整个项会下降。也就是说，信息越充分，不确定性奖励越小。这就是“探索随样本积累而衰减”的数学原因。

玩具例子里，前 1 到 50 轮某个新内容可能只展示过 3 次，即使经验 CTR 还不高，它的 $n_a$ 很小，置信加成就会很大，所以系统会继续给它机会。一旦试验次数上来，发现真实表现一般，这个加成就会迅速缩小，流量回到更优内容上。

### 3. Thompson Sampling

Thompson Sampling 是贝叶斯采样。贝叶斯，白话说，就是把“参数不确定”也当成模型的一部分。对点击/未点击这类二元奖励，常用 Beta 分布做后验：

$$
\theta_a \sim \text{Beta}(\alpha_a,\beta_a)
$$

每轮做两件事：

1. 从每个物品的后验里采样一个 $\tilde{\theta}_a$
2. 选择采样值最大的物品

即：

$$
a_t=\arg\max_a \tilde{\theta}_a,\quad \tilde{\theta}_a \sim \text{Beta}(\alpha_a,\beta_a)
$$

更新规则也很直接：

- 点击：$\alpha_a \leftarrow \alpha_a + 1$
- 未点击：$\beta_a \leftarrow \beta_a + 1$

它为什么探索得自然？因为样本少的物品，后验分布更宽，随机采样时偶尔能采到很高的值，于是得到曝光机会；样本多后分布收窄，不确定性下降，探索自动减少。

### 4. 动态收敛过程

继续看 5 个候选内容真实 CTR 为 $\{30\%,25\%,20\%,15\%,10\%\}$、$\varepsilon=0.1$ 的例子。

- 第 1 到 100 轮：为了避免冷启动误判，可以人为加入 $\varepsilon$-first，让系统近似均匀探索，每个内容至少获得基础样本。
- 第 100 到 500 轮：系统开始主要利用当前估值最高的内容，但仍保留 10% 随机试探。此时 30% CTR 的内容通常会逐步领先，但 25% 内容仍会保留一部分流量。
- 第 500 到 1000 轮：大部分流量会集中到真实最优臂，探索比例依旧存在，但影响变小。累计 regret 的增长速度明显放缓。

这说明 Bandit 不是“一上来就找到最优”，而是通过早期试错换取中后期稳定收益。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它模拟 5 个候选内容的点击反馈，并比较 $\varepsilon$-Greedy、UCB 和 Thompson Sampling 的基本写法。

```python
import math
import random

TRUE_CTRS = [0.30, 0.25, 0.20, 0.15, 0.10]
BEST_CTR = max(TRUE_CTRS)

def reward(arm):
    return 1 if random.random() < TRUE_CTRS[arm] else 0

def run_epsilon_greedy(steps=2000, epsilon=0.1, seed=7):
    random.seed(seed)
    k = len(TRUE_CTRS)
    counts = [0] * k
    values = [0.0] * k
    total_reward = 0

    for t in range(steps):
        if 0 in counts:
            arm = counts.index(0)  # 先保证每个臂至少试一次
        elif random.random() < epsilon:
            arm = random.randrange(k)
        else:
            arm = max(range(k), key=lambda i: values[i])

        r = reward(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        total_reward += r

    return counts, values, total_reward

def run_ucb(steps=2000, c=2**0.5, seed=7):
    random.seed(seed)
    k = len(TRUE_CTRS)
    counts = [0] * k
    values = [0.0] * k
    total_reward = 0

    for t in range(steps):
        if 0 in counts:
            arm = counts.index(0)
        else:
            scores = [
                values[i] + c * math.sqrt(math.log(t + 1) / counts[i])
                for i in range(k)
            ]
            arm = max(range(k), key=lambda i: scores[i])

        r = reward(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        total_reward += r

    return counts, values, total_reward

def run_thompson(steps=2000, seed=7):
    random.seed(seed)
    k = len(TRUE_CTRS)
    alpha = [1] * k
    beta = [1] * k
    counts = [0] * k
    total_reward = 0

    for _ in range(steps):
        samples = [random.betavariate(alpha[i], beta[i]) for i in range(k)]
        arm = max(range(k), key=lambda i: samples[i])

        r = reward(arm)
        counts[arm] += 1
        total_reward += r
        if r == 1:
            alpha[arm] += 1
        else:
            beta[arm] += 1

    posterior_means = [alpha[i] / (alpha[i] + beta[i]) for i in range(k)]
    return counts, posterior_means, total_reward

eg_counts, eg_values, eg_reward = run_epsilon_greedy()
ucb_counts, ucb_values, ucb_reward = run_ucb()
ts_counts, ts_values, ts_reward = run_thompson()

assert sum(eg_counts) == 2000
assert sum(ucb_counts) == 2000
assert sum(ts_counts) == 2000

# 最优臂是索引 0，经过足够轮数后应获得最多曝光
assert eg_counts[0] == max(eg_counts)
assert ucb_counts[0] == max(ucb_counts)
assert ts_counts[0] == max(ts_counts)

# 累计奖励不应超过理论上界
assert eg_reward <= 2000
assert ucb_reward <= 2000
assert ts_reward <= 2000
```

如果要把它接到真实推荐服务里，最小实现通常只需要维护以下状态：

| 字段 | 含义 |
|---|---|
| `counts[a]` | 物品 `a` 被展示了多少次 |
| `clicks[a]` 或 `reward_sum[a]` | 累计奖励 |
| `values[a]` | 经验均值或估计 CTR |
| `alpha[a], beta[a]` | Thompson Sampling 的后验参数 |
| `last_update_time` | 用于做时间衰减或漂移监控 |

真实工程例子可以这样落地：首页候选先经过召回和粗排得到 50 篇文章，Bandit 不负责“从百万内容里找候选”，而是在这 50 篇里决定最终曝光顺序。也就是说，Bandit 常常不是替代整套推荐系统，而是作为最后一层在线探索策略。

---

## 工程权衡与常见坑

从工程角度看，Bandit 的核心权衡不是公式，而是“短期 CTR 损失换长期学习速度”。MCP Analytics 在 2026 年 2 月 13 日的白皮书摘要中给出的结果是：上下文 Bandit 相比 $\varepsilon$-greedy 基线和无上下文 UCB 有明显收益；在高流量内容推荐中，Thompson Sampling 的累计奖励比 LinUCB 高 8%，而 LinUCB 在金融等风险敏感场景中方差更低、更容易审计。这类结论的重点不是某个数字本身，而是说明“是否有上下文”“是否能接受随机性”比单纯争论算法名字更重要。

常见坑如下：

| 坑 | 原因 | 结果 | 缓解措施 |
|---|---|---|---|
| $\varepsilon$ 设太大 | 长期随机流量过多 | CTR 明显下降 | 冷启动期大，稳定后衰减 |
| $\varepsilon$ 设太小 | 新内容几乎没机会 | 被旧热门锁死 | 设最小探索下限，或加 $\varepsilon$-first |
| UCB 的 $c$ 过大 | 过度相信不确定性 | 探索浪费流量 | 离线回放 + 小流量线上调参 |
| UCB 的 $c$ 过小 | 不敢试新内容 | 早期误判后难纠正 | 对冷启动物品单独提高 bonus |
| Thompson 无先验 | 完全冷启动太慢 | 相似内容知识无法复用 | 用历史类目 CTR 设 informative prior |
| 奖励定义错误 | 只看点击，不看低质点击 | 学到标题党 | 奖励改成点击 + 停留时长加权 |
| 忽略延迟反馈 | 转化晚于曝光很多 | 在线估值失真 | 用代理指标，做延迟归因回填 |
| 分布漂移未处理 | 热点变化、节假日变化 | 老数据压制新趋势 | 时间衰减、滑动窗口、周期重置 |

一个非常常见的工程补丁是 optimistic initialization，也就是“乐观初始化”。白话说，就是给新内容一个偏高的初始估值，例如把新内容的初始 CTR 估计设成 0.2 而不是 0。这样系统会主动给它几次曝光，避免它因为没有历史数据直接沉底。

另一个常见补丁是分阶段策略：

1. 新用户前 $K$ 次强制均匀探索。
2. 新内容上线前 $M$ 次至少保证最小曝光。
3. 稳定期切到 UCB 或 Thompson Sampling。
4. 如果监控发现点击分布明显漂移，重置部分计数或加大时间衰减。

这里的关键不是“重置所有历史”，而是承认线上环境不稳定。推荐不是实验室里的静态分布。

---

## 替代方案与适用边界

Bandit 很有用，但它不是推荐系统的万能壳。

如果上下文很多，比如用户兴趣、设备、时间、来源频道、会话阶段都强相关，那么优先考虑 Contextual Bandit。它本质上是在“每个用户环境下”做探索与利用，而不是只学一个全局 CTR。LinUCB 是线性置信上界方法，解释性强；LinTS 或更一般的 Contextual Thompson Sampling 是贝叶斯版本，常在高维特征下表现更灵活。

如果问题更长链路，比如首页推荐会影响后续停留、搜索、关注、次日留存，那么标准 Bandit 就不够，因为它只优化即时奖励或短反馈。此时要考虑强化学习或至少做多目标重排。

下面给出一个工程上更实用的比较表：

| 方案 | 是否用上下文 | 优点 | 风险/成本 | 适用边界 |
|---|---|---|---|---|
| $\varepsilon$-Greedy | 否 | 简单、稳定、开发快 | 探索效率低 | 先做基线、流量不大 |
| UCB | 否 | 有原则地偏向低样本内容 | 对参数和非平稳性敏感 | 候选少、奖励较稳定 |
| Thompson Sampling | 否 | 探索自然、调参少 | 解释性略弱 | CTR 优化、内容流、广告 |
| LinUCB | 是 | 可解释、决策可复现 | 特征工程要求高 | 风险敏感、需审计 |
| Contextual Thompson Sampling | 是 | 个性化强、通常收敛更快 | 实现和监控更复杂 | 高流量个性化推荐 |
| 离线排序 + 轻量 Bandit | 部分 | 改造成本低 | 上限受离线模型限制 | 现有推荐系统增量升级 |
| 强化学习 | 是，多步 | 能优化长期价值 | 数据和系统成本高 | 明确存在长期序列影响 |

真实工程里常见的组合方案反而不是“全站纯 Bandit”，而是：

- 召回和粗排继续用离线模型。
- 精排输出候选分数。
- Bandit 在最终曝光位做轻量探索。
- 对新内容、新用户、新频道单独加探索配额。

例如风险敏感的广告定价系统，初期往往先用 LinUCB，原因不是它收益一定最高，而是它决策路径更稳定，更适合受监管场景；当日志充足、团队能接受更强随机性后，再逐步切到 Thompson Sampling 提高累计收益。这类迁移路线通常比“一步到位上最复杂模型”更现实。

---

## 参考资料

- System Overflow, Core Bandit Algorithms: Epsilon Greedy, UCB, and Thompson Sampling  
  https://www.systemoverflow.com/learn/ml-recommendation-systems/diversity-exploration/core-bandit-algorithms-epsilon-greedy-ucb-and-thompson-sampling
- System Overflow, Upper Confidence Bound (UCB): Optimism Under Uncertainty  
  https://www.systemoverflow.com/learn/ml-ab-testing/multi-armed-bandits/upper-confidence-bound-ucb-optimism-under-uncertainty
- System Overflow, Thompson Sampling: Bayesian Probability Matching  
  https://www.systemoverflow.com/learn/ml-ab-testing/multi-armed-bandits/thompson-sampling-bayesian-probability-matching
- MCP Analytics, Contextual Bandits: UCB vs Thompson Sampling in Prod, February 13, 2026  
  https://mcpanalytics.ai/whitepapers/contextual-bandits-whitepaper
- Artificial-Intelligence-Wiki, Multi-Armed Bandit Algorithms, Updated November 23, 2025  
  https://artificial-intelligence-wiki.com/agentic-ai/autonomous-decision-making-frameworks/multi-armed-bandit-algorithms/

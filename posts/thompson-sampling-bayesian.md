## 核心结论

Thompson Sampling（简称 TS）是一种贝叶斯探索策略。贝叶斯可以先理解成“用概率分布表达不确定性的方法”。它不直接拿每个候选臂的历史平均奖励做比较，而是先为每个臂维护一个“真实收益参数的后验分布”，后验可以理解成“结合历史观测后，对真实参数当前相信到什么程度”。每一轮，它从每个臂的后验里各采样一个值，再选择采样值最大的那个臂。

它的核心动作可以写成：

$$
\tilde \theta_i \sim p(\theta_i \mid D_t), \qquad
a_t = \arg\max_i \tilde \theta_i
$$

这里 $\theta_i$ 是第 $i$ 个臂的真实收益参数，$D_t$ 是到第 $t$ 轮为止的数据，$\tilde \theta_i$ 是一次采样得到的“当前可能真实值”。

这个策略的关键优点是：探索和利用不是人为硬切分出来的，而是由后验分布自动决定。数据少、后验宽的臂，不确定性大，更容易采样出一个偏高的值，因此会自然获得探索机会；数据多、后验窄的臂，采样结果更稳定，如果均值也高，就会越来越常被选中。这种机制常被称为“概率匹配”，意思是“一个臂被选中的概率，大致匹配它是当前最优臂的概率”。

一个直观判断是：如果两个广告里，一个样本很少但均值看起来高，另一个样本很多且稳定更好，TS 不会只盯着表面均值。它会把“均值”和“不确定性”一起算进去。因此，新广告不会因为样本少而永远没机会，老广告也不会因为偶然几次高点击就长期霸榜。

| 比较方式 | 决策依据 | 结果倾向 |
| --- | --- | --- |
| 只看历史均值 | 哪个平均值大就选哪个 | 容易过早锁死 |
| UCB | 均值 + 乐观置信上界 | 主动给不确定臂加奖励 |
| Thompson Sampling | 从后验采样后再比较 | 用概率方式平衡探索与利用 |

从理论上看，TS 在多臂老虎机问题上有渐近最优的对数悔恨界。悔恨可以先理解成“因为没一直选最优臂而损失掉的累计收益”。从实践上看，在广告投放、推荐系统、在线实验等场景中，它常常比简单贪心和不少 UCB 变体更稳，尤其适合冷启动明显、反馈噪声较大的问题。

---

## 问题定义与边界

Thompson Sampling 主要解决的是多臂老虎机问题。多臂老虎机可以理解成：你每一轮只能做一个动作，但每个动作的真实收益未知，而且收益带噪声。目标是在试验过程中边学边选，让累计收益尽量大。

标准设定是：

| 要素 | 含义 |
| --- | --- |
| 对象 | 多个候选臂，例如多个广告、标题、推荐内容 |
| 动作 | 每轮选择一个臂 |
| 奖励 | 该臂返回一个随机结果，如点击 `1/0`、收入、停留时长 |
| 目标 | 最大化累计奖励，或最小化累计悔恨 |
| 约束 | 每轮只能观测被选中臂的反馈，不能同时知道所有臂结果 |

这不是离线排序问题，而是在线决策问题。离线排序通常先收集完整数据，再训练一个排序模型；在线 bandit 是一边行动一边收集数据，决策本身会改变后续看到的数据。

玩具例子可以这样看。现在有 3 篇候选文章：首页每次只能展示 1 篇。用户点了记作 1，不点记作 0。系统刚开始不知道哪篇更吸引人，只能一边展示一边统计点击。这里每篇文章就是一个臂，点击就是伯努利奖励。伯努利分布可以先理解成“结果只有成功或失败，例如点或不点”的概率模型。

TS 的适用边界也要说清楚：

1. 奖励需要能被一个明确的概率模型描述，比如点击用伯努利，连续收入用高斯。
2. 需要能在线更新后验，至少要有近似更新办法。
3. 环境不能过于剧烈非平稳。非平稳可以理解成“真实分布会持续变化”。如果用户兴趣每天大幅切换，纯累计后验会迅速过时。
4. 奖励反馈最好不要无限延迟。延迟过长会让决策和更新脱节。
5. 如果动作不是少量离散臂，而是高维连续控制，TS 就不是直接可用的标准答案，通常要升级到 contextual bandit 或更完整的强化学习方法。

因此，TS 最适合的是：动作空间有限、反馈可观测、需要在线探索、又希望实现简单可控的场景。

---

## 核心机制与推导

TS 的机制可以拆成三步：

1. 为每个臂维护参数的后验分布。
2. 每轮从每个后验里采样一个候选值。
3. 选采样值最大的臂，并用真实反馈更新该臂后验。

核心不是“谁历史均值最高”，而是“谁在当前不确定性下更可能是最优”。

### 1. 伯努利奖励：Beta 后验

如果奖励只有成功和失败，比如点击与否，常用 Beta-Bernoulli 模型。Beta 分布可以先理解成“定义在 0 到 1 之间、常用来表示概率不确定性”的分布。

设第 $i$ 个臂的真实点击率是 $\theta_i$，先验为：

$$
\theta_i \sim \mathrm{Beta}(\alpha_i, \beta_i)
$$

如果观察到 $s_i$ 次成功、$f_i$ 次失败，那么后验为：

$$
\theta_i \mid D_t \sim \mathrm{Beta}(\alpha_i + s_i, \beta_i + f_i)
$$

这里的含义很直接：成功次数加到 $\alpha$，失败次数加到 $\beta$。如果初始取 $\mathrm{Beta}(1,1)$，它是一个弱先验，可以理解成“开始时没有明显偏好”。

玩具例子：

- 臂 A：先验 $\mathrm{Beta}(1,1)$，观察到 8 次成功、2 次失败，后验变成 $\mathrm{Beta}(9,3)$
- 臂 B：先验 $\mathrm{Beta}(1,1)$，观察到 3 次成功、7 次失败，后验变成 $\mathrm{Beta}(4,8)$

如果某一轮采样得到：

$$
\tilde \theta_A = 0.74,\qquad \tilde \theta_B = 0.31
$$

那就选 A。下一轮即使 A 的均值仍更高，采样也可能变成：

$$
\tilde \theta_A = 0.42,\qquad \tilde \theta_B = 0.55
$$

这时会选 B。原因不是 B 更好，而是 B 仍有一定概率被认为“可能更好”，算法就给它一次试探机会。

### 2. 为什么后验宽会促进探索

后验宽，表示不确定性大。数学上，采样分布更分散，更容易抽到极端值。极端值一旦超过其他臂，本轮就会被选中。

这正是 TS 的探索来源。它不是人工规定“每 10 次随机一次”，而是把探索强度绑定到不确定性。当某个臂样本很少时，分布宽，探索概率高；当样本越来越多时，分布收缩，探索自然减少。

这比固定 $\epsilon$ 的 $\epsilon$-greedy 更细。$\epsilon$-greedy 的意思是“以固定小概率随机探索，否则选当前最好”。它简单，但不知道谁更不确定，也不知道哪些臂已经试够了。

### 3. 高斯奖励：正态后验

如果奖励是连续值，比如每次展示后的收入、停留时长增量，可以用高斯模型。高斯分布先理解成“围绕某个均值波动的连续噪声模型”。

设奖励噪声方差已知为 $\sigma^2$，第 $i$ 个臂真实均值参数为 $\theta_i$，先验为：

$$
\theta_i \sim \mathcal N(m_i, v_i)
$$

观察到 $n_i$ 个样本，样本均值为 $\bar x_i$，则后验仍是正态分布，参数更新为：

$$
v_i' = \left(\frac{1}{v_i} + \frac{n_i}{\sigma^2}\right)^{-1}
$$

$$
m_i' = v_i' \left(\frac{m_i}{v_i} + \frac{n_i \bar x_i}{\sigma^2}\right)
$$

这个公式表达的是“精度相加”。精度可以理解成“方差的倒数，越大表示越确定”。先验精度和数据精度一起决定后验精度；先验均值和样本均值按各自精度加权，得到后验均值。

因此，高斯版 TS 仍然是同一个逻辑：后验均值代表当前估计，后验方差代表不确定性，采样把这两件事同时带进决策里。

### 4. 真实工程例子

新闻推荐是典型例子。假设首页有 20 个候选标题，每次只能展示 1 个，目标是提高点击。新标题刚上线时没有数据，老标题已经展示很多次。纯贪心策略会偏向老标题，因为它们的历史均值更稳定；但这会让新标题长期得不到曝光，系统就永远学不会它们是否更优。

TS 在这里的作用是：

- 新标题样本少，后验宽，会自动获得更多尝试。
- 老标题样本多，后验窄，如果效果确实稳定更高，就会被持续选中。
- 整个过程不需要人工写复杂的探索规则。

这就是它在推荐和广告系统里被广泛采用的根本原因。

---

## 代码实现

实现 TS 的重点不是复杂优化，而是把“状态维护”和“决策逻辑”拆开。对于伯努利奖励，每个臂只需要维护两个参数：`alpha` 和 `beta`。

| 字段 | 含义 |
| --- | --- |
| `alpha` | 成功计数加先验伪计数 |
| `beta` | 失败计数加先验伪计数 |
| `sample` | 本轮从后验采样出的值 |
| `count` | 被拉取次数，可选统计字段 |

最小流程如下：

```text
初始化每个臂的先验参数
循环每一轮：
  对每个臂从后验分布采样一个值
  选择采样值最大的臂
  观察奖励
  用奖励更新该臂的后验参数
```

下面给一个可运行的 Python 版本，模拟 3 个伯努利臂。代码里用了 `random.betavariate` 从 Beta 分布采样。

```python
import random

class BernoulliThompsonSampling:
    def __init__(self, n_arms, alpha=1.0, beta=1.0):
        self.alpha = [alpha] * n_arms
        self.beta = [beta] * n_arms
        self.count = [0] * n_arms

    def select_arm(self):
        samples = [
            random.betavariate(self.alpha[i], self.beta[i])
            for i in range(len(self.alpha))
        ]
        return max(range(len(samples)), key=lambda i: samples[i])

    def update(self, arm, reward):
        assert reward in (0, 1)
        self.count[arm] += 1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

def simulate(true_rates, rounds=5000, seed=7):
    random.seed(seed)
    ts = BernoulliThompsonSampling(len(true_rates))
    rewards = 0

    for _ in range(rounds):
        arm = ts.select_arm()
        reward = 1 if random.random() < true_rates[arm] else 0
        ts.update(arm, reward)
        rewards += reward

    best_arm = max(range(len(true_rates)), key=lambda i: true_rates[i])
    learned_best = max(range(len(true_rates)), key=lambda i: ts.alpha[i] / (ts.alpha[i] + ts.beta[i]))
    return ts, rewards, best_arm, learned_best

true_rates = [0.05, 0.08, 0.12]
ts, rewards, best_arm, learned_best = simulate(true_rates)

assert best_arm == 2
assert learned_best == 2
assert sum(ts.count) == 5000
assert rewards > 300  # 随机模拟下应明显大于纯随机的下界

print("alpha:", ts.alpha)
print("beta:", ts.beta)
print("count:", ts.count)
print("best_arm:", best_arm, "learned_best:", learned_best, "rewards:", rewards)
```

这段代码体现了两个实现原则：

1. 决策只依赖当前后验，不需要额外全局优化器。
2. 更新只发生在被选中的臂，因为 bandit 只能看到被执行动作的反馈。

如果要切到高斯奖励，也只需要把每个臂的状态改成 `mean`、`variance`、`count`，并用正态后验更新公式替换 `alpha/beta` 更新逻辑。也就是说，TS 的“外壳”基本不变，变化的是单臂奖励模型。

---

## 工程权衡与常见坑

TS 的代码不长，但线上效果高度依赖建模是否正确。很多失败案例不是算法错，而是目标定义和数据回流出了问题。

| 坑点 | 后果 | 规避方法 |
| --- | --- | --- |
| 先验过强 | 新臂长期得不到足够探索 | 从弱先验开始，按业务量级校准伪计数 |
| 奖励定义不稳 | 后验对应的目标混乱，决策漂移 | 固定单一目标，如只优化 CTR 或只优化 CVR |
| 非平稳环境 | 旧数据持续污染当前决策 | 用滑动窗口、折扣后验或周期性重置 |
| 忽略方差 | 连续奖励场景下不确定性估计失真 | 显式建模噪声，必要时用 Normal-Gamma |
| 延迟反馈 | 更新时点错位，误判臂质量 | 区分曝光时间与反馈时间，做异步更新 |

最常见的真实工程坑有三个。

第一，奖励口径混用。比如广告场景里，今天按点击更新，明天又把转化混进来，后验就不再表示同一个目标。一个后验分布只能对应一个明确问题，比如“该广告的点击率”或“该广告的转化率”，不能两种语义混在一起。

第二，延迟反馈。转化常常不是实时返回，可能在几小时甚至几天后才到账。如果系统在曝光后立刻把“未转化”当成负样本更新，会系统性低估某些长决策链条的广告。这时必须做异步更新：曝光先记日志，等真实反馈回来后再补更新，或者用延迟校正模型。

第三，非平稳。比如内容平台上，热点每天都在变。昨天用户爱看某类标题，不代表今天还爱看。如果还把一个月前的数据和今天等权相加，后验会变得“稳定但过时”。这时纯 TS 不够，需要给旧数据打折。常见做法是把成功和失败计数乘一个折扣因子 $\gamma \in (0,1)$，或者只保留最近 $N$ 天数据。

还有一个容易被忽视的问题是“方差视角”。初学者常以为 TS 只是“随机选一下”。其实它依赖的是后验方差。两个臂均值一样时，方差更大的臂应该更常被试探；如果实现里只记录均值、不记录不确定性，那就已经不是 TS 了。

---

## 替代方案与适用边界

TS 不是唯一可用的探索策略。它的优势在于：建模清楚时很自然，探索强度随不确定性自动变化，实现也不复杂。但如果场景不满足前提，就应考虑其他方法。

| 方法 | 探索方式 | 实现复杂度 | 对先验依赖 | 适合场景 |
| --- | --- | --- | --- | --- |
| Thompson Sampling | 从后验采样做概率匹配 | 中等 | 有一定依赖 | 奖励可建模、需要在线探索 |
| UCB | 均值加不确定性上界 | 中等 | 低 | 希望有明确乐观上界控制 |
| Epsilon-Greedy | 固定概率随机探索 | 低 | 无 | 基线系统、快速上线验证 |

TS 和这些方法的边界可以这样理解：

- 如果你只想先做一个能跑的在线探索基线，$\epsilon$-greedy 最简单，但效果通常粗糙。
- 如果你不想引入先验，只想用置信区间做“乐观探索”，UCB 更直接。
- 如果你能明确奖励模型，并且关心冷启动与不确定性表达，TS 往往更合适。

再往外扩一层，如果不同用户上下文会显著改变最优臂，比如“同一篇文章对新用户和老用户点击率差异很大”，那就不是无上下文 bandit 了，而该考虑 contextual bandit。上下文 bandit 可以先理解成“选择动作前还能看到用户特征，再根据特征做探索与利用”。

如果环境强非平稳，比如推荐池每天大换血，标准 TS 也不够。这时可以考虑：

- 滑动窗口 TS：只看最近一段数据。
- 折扣 TS：旧数据按时间衰减。
- 分段重启 TS：检测分布漂移后重置后验。

如果动作之间有长期依赖，比如今天推了什么会影响用户明天行为，那就超出了单步 bandit 假设，应该转向更完整的强化学习框架。

所以，TS 的真正适用边界不是“它强不强”，而是“你的问题是否满足单步在线决策、奖励可建模、反馈可更新这三个条件”。

---

## 参考资料

1. [Thompson, 1933. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples](https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862)
2. [Russo et al., 2018. A Tutorial on Thompson Sampling](https://www.nowpublishers.com/article/Details/MAL-070)
3. [Chapelle & Li, 2011. An Empirical Evaluation of Thompson Sampling](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling)
4. [Korda, Kaufmann, and Munos, 2013. Thompson Sampling for 1-Dimensional Exponential Family Bandits](https://proceedings.neurips.cc/paper/2013/hash/aba3b6fd5d186d28e06ff97135cade7f-Abstract.html)
5. [Bubeck and Liu, 2013. Prior-free and prior-dependent regret bounds for Thompson Sampling](https://proceedings.neurips.cc/paper/2013/hash/39461a19e9eddfb385ea76b26521ea48-Abstract.html)

## 核心结论

探索策略不是“多试几个动作”这么简单，而是在每一步决策里回答一个更具体的问题：现在应该优先赚眼前这一次，还是先花一点机会去减少不确定性。这里的不确定性，白话讲，就是“我还没看明白这个动作到底好不好”。

在多臂赌博机问题里，三类经典策略对应三种不同的控制方式：

| 策略 | 探索机制 | 典型参数/分布 |
|---|---|---|
| ε-greedy | 以 ε 概率随机选，剩余时间选当前最优 | ε≈0.05~0.15 |
| UCB | 在估计均值上加置信上界，少试过的动作额外加分 | $c,\ Q(a)+c\sqrt{\ln t/N(a)}$ |
| Thompson Sampling | 为每个动作维护后验分布并采样，采样值最大者被选中 | Beta$(\alpha,\beta)$ 等先验/后验 |

三者的本质差异不在“有没有探索”，而在“探索是否识别了不确定性的来源”。

- ε-greedy 最简单，但把所有非最优动作一视同仁。一个明显差的动作和一个只是样本少的动作，都会被随机探索到。
- UCB 把“样本少”直接转成探索奖金。动作被尝试次数 $N(a)$ 越少，奖金越大。
- Thompson Sampling 把“我对这个动作有多没把握”写成概率分布，再按分布采样，属于概率匹配，通常更平滑。

结论可以压缩成一句话：探索策略的设计核心，是把“不确定性”编码进动作选择概率，而不是单纯增加随机性。

---

## 问题定义与边界

先限定问题边界。这里讨论的是**标准多臂赌博机**：每轮从若干动作里选一个，只观察被选动作的奖励。多臂赌博机，白话讲，就是“有多个可选按钮，但每次只能按一个，而且只能看到这一个按钮的反馈”。

目标通常写成最小化**累计遗憾**。遗憾，白话讲，就是“如果每次都选最优动作，本来可以多赚多少”。

$$
R_T=\sum_{t=1}^{T}\left(\mu^*-\mu_{a_t}\right)
$$

其中，$\mu^*$ 是最优动作的真实期望奖励，$\mu_{a_t}$ 是第 $t$ 轮实际选中动作的真实期望奖励。这个定义强调的是长期损失，不是某一轮输赢。

这个问题有几个重要边界：

| 边界 | 含义 | 对策略的要求 |
|---|---|---|
| 只看到已选动作反馈 | 没选的动作没有新数据 | 必须保留探索通道 |
| 早期样本很少 | 当前“最好”可能只是偶然 | 不能过早锁死 |
| 奖励可能平稳也可能非平稳 | 真实均值可能固定，也可能漂移 | 平稳算法未必适合线上系统 |
| 目标是累计收益 | 不是单轮最优，而是长期最优 | 需要权衡短期与长期 |

一个新手容易忽略的边界是：**每个动作都至少要有被再次验证的机会**。否则，算法可能因为前几次偶然结果，把真正更优的动作永久放弃。

玩具例子：两臂老虎机，臂 A 前 5 次里赢了 4 次，臂 B 前 2 次里只赢了 0 次。纯贪心会一直选 A，因为当前估计胜率是 $0.8$ 对 $0$。问题在于，B 的信息量太少，不能据此断定它差。如果 B 的真实胜率其实是 $0.7$，只是前两次碰巧都输，那么过早放弃会持续积累遗憾。

真实工程例子：广告投放中有两个创意 A 和 B。A 已投放 10 万次，CTR 稳定在 0.8%；B 刚上线，只投了 200 次，CTR 只有 0.3%。如果系统只按当前均值排序，B 会被迅速下线。但 B 的样本太少，方差很大，现有数据不足以说明它长期更差。探索策略存在的意义，就是防止系统把“没看清”误判成“真的差”。

如果场景是**非平稳**的，也就是奖励分布会变，那么边界又变了。比如节假日、活动季、流量结构切换后，过去的赢家可能不再是赢家。此时，单调减小探索强度的策略会逐渐失去纠错能力，必须引入窗口、折扣或重置机制。

---

## 核心机制与推导

### 1. ε-greedy：用固定随机性保底

ε-greedy 的策略很直接：以概率 $1-\epsilon$ 选当前估计最好的动作，以概率 $\epsilon$ 随机选动作。

$$
\pi(a)=
\begin{cases}
1-\epsilon+\frac{\epsilon}{|A|}, & a=\arg\max Q(a) \\
\frac{\epsilon}{|A|}, & \text{otherwise}
\end{cases}
$$

这里的 $Q(a)$ 是动作价值估计，白话讲，就是“我目前认为这个动作平均能拿多少奖励”。

优点是实现成本极低，缺点也同样直接：它只知道“随机试一试”，不知道“该重点试谁”。一个已经试了很多次且明显差的动作，仍会和一个样本不足的动作获得同等级别的随机机会。

### 2. UCB：把不确定性显式加到分数里

UCB 的核心是：

$$
UCB(a)=Q(a)+c\sqrt{\frac{\ln t}{N(a)}}
$$

这里的置信上界，白话讲，就是“我先乐观地把这个动作估高一点，给它一次翻盘机会”。

- $Q(a)$ 是当前均值估计
- $N(a)$ 是动作被选次数
- $t$ 是总轮数
- $c$ 控制探索强度

这个式子的关键结构是：$N(a)$ 小，奖励加成大；$t$ 变大，整体探索加成只缓慢上升。于是，UCB 会优先把探索预算给“看起来还行但样本太少”的动作，而不是均匀随机撒出去。

这个形式来自浓缩不等式的思想。对白话理解来说，只要抓住一点：样本少，均值估计不稳，所以要给一个与不确定性成正比的补偿。

### 3. Thompson Sampling：把不确定性建模成分布

Thompson Sampling 不直接算一个确定分数，而是给每个动作维护一个后验分布。后验，白话讲，就是“看完历史数据后，我对这个动作真实好坏的概率判断”。

在 Bernoulli 奖励下，常用 Beta 后验：

$$
\theta_a \sim Beta(\alpha_a,\beta_a), \qquad a_t=\arg\max_a \theta_a
$$

如果动作成功一次，就把 $\alpha$ 加 1；失败一次，就把 $\beta$ 加 1。于是：

- 样本少时，分布宽，容易采到高值，也容易被探索
- 样本多时，分布窄，动作排序更稳定
- 不需要手动构造探索奖金，探索是从后验采样自然产生的

### 4. 一个数值推导例子

假设第 $t=5$ 轮，三臂的统计为：

| 动作 | $Q(a)$ | $N(a)$ |
|---|---:|---:|
| A | 0.60 | 10 |
| B | 0.55 | 2 |
| C | 0.50 | 1 |

取 $c=1$，则

$$
\ln 5 \approx 1.609
$$

所以：

- $UCB(A)=0.60+\sqrt{1.609/10}\approx 1.001$
- $UCB(B)=0.55+\sqrt{1.609/2}\approx 1.447$
- $UCB(C)=0.50+\sqrt{1.609/1}\approx 1.769$

结果是 C 被选中。原因不是它均值最高，而是它最不确定。

这正是 UCB 的设计目标：短期允许“看起来不优”的动作被重新验证，以避免长期把最优动作错杀。

再看两臂老虎机这个玩具例子。假设目前：

- 臂 A：8 次里成功 4 次，$Q(A)=0.5$
- 臂 B：2 次里成功 1 次，$Q(B)=0.5$

此时 ε-greedy 在利用阶段看不出区别，只能依赖随机探索；UCB 会偏向 B，因为它的 $N(B)$ 更小；Thompson Sampling 下，A 的后验是 $Beta(5,5)$，B 的后验是 $Beta(2,2)$，B 的分布更宽，因此更常出现高采样值。这三者体现了完全不同的探索力度。

---

## 代码实现

工程上要落地，核心是维护每个动作的统计量。对 Bernoulli 奖励，最常见的状态就是：

- 均值估计或成功率
- 被选择次数 $N(a)$
- Thompson Sampling 的 $\alpha,\beta$

下面这个 Python 例子可以直接运行，分别实现 ε-greedy、UCB 和 Thompson Sampling 的选臂逻辑。`assert` 用来确认更新后的统计没有出错。

```python
import math
import random

class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def pull(self):
        return 1 if random.random() < self.p else 0


class BanditStats:
    def __init__(self, k):
        self.counts = [0] * k
        self.values = [0.0] * k
        self.alpha = [1] * k
        self.beta = [1] * k

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


def select_epsilon_greedy(stats, epsilon=0.1):
    k = len(stats.values)
    if random.random() < epsilon:
        return random.randrange(k)
    best = max(range(k), key=lambda i: stats.values[i])
    return best


def select_ucb(stats, t, c=1.0):
    for i, n in enumerate(stats.counts):
        if n == 0:
            return i
    scores = [
        stats.values[i] + c * math.sqrt(math.log(t) / stats.counts[i])
        for i in range(len(stats.values))
    ]
    return max(range(len(scores)), key=lambda i: scores[i])


def select_thompson(stats):
    samples = [
        random.betavariate(stats.alpha[i], stats.beta[i])
        for i in range(len(stats.alpha))
    ]
    return max(range(len(samples)), key=lambda i: samples[i])


random.seed(7)
arms = [BernoulliArm(0.45), BernoulliArm(0.60)]
stats = BanditStats(k=2)

for t in range(1, 51):
    arm = select_ucb(stats, t=max(t, 2), c=1.0)
    reward = arms[arm].pull()
    stats.update(arm, reward)

assert sum(stats.counts) == 50
assert all(a >= 1 and b >= 1 for a, b in zip(stats.alpha, stats.beta))
assert max(stats.values) <= 1.0 and min(stats.values) >= 0.0
```

如果线上奖励是点击/不点击、转化/不转化，Thompson Sampling 常常最省事，因为 Beta-Bernoulli 更新非常自然：每个动作维护

$$
\alpha = accepts + 1,\qquad \beta = denies + 1
$$

也就是成功次数加 1、失败次数加 1，对应一个均匀先验。

下面是常见的 TypeScript 选择逻辑，适合直接嵌到推荐或投放服务里：

```ts
function selectArm(stats: ArmStats[]): string {
  if (Math.random() < EPSILON) {
    return stats[randIndex(stats.length)].id;
  }
  return stats
    .map((arm) => ({
      id: arm.id,
      score: sampleBeta(arm.accepts + 1, arm.denies + 1),
    }))
    .sort((a, b) => b.score - a.score)[0].id;
}
```

真实工程例子：信息流推荐有 20 个候选内容，每次只展示 1 个。若目标是点击率，系统可以把每个内容当成一个臂：

- `accepts` 记录点击次数
- `denies` 记录未点击次数
- 每次请求时用 Thompson Sampling 采样一次
- 采样值最大的内容进入排序前列

这套做法的优势不是“理论更高级”，而是它把冷启动内容自动放进了竞争池。新内容因为后验更宽，偶尔会被采到高分；老内容如果长期表现稳定，后验会变窄，排序会更稳。

---

## 工程权衡与常见坑

真正上线时，问题往往不在公式，而在数据分布和系统约束。

| 坑 | 原因 | 规避 |
|---|---|---|
| ε-greedy 冷启动低效 | 固定 ε 不区分“明显差”和“还不确定” | 改用 UCB/TS，或让 ε 随时间衰减 |
| UCB 非平稳失效 | 旧样本一直累计，置信项收缩后难以翻盘 | Sliding-window UCB、Discounted UCB |
| TS 历史滞后 | 后验持续累加旧数据，环境变化时反应慢 | 动态先验、折扣更新、窗口 TS |
| 稀疏奖励学习太慢 | 成功事件极少，早期信号噪声大 | 设计代理指标，如加购、停留时长 |
| 只看 CTR 不看业务目标 | 探索优化了局部指标，不等于收益最大 | 奖励定义对齐 GMV、留存、利润 |

先看 ε-greedy。它最大的问题不是“随机”，而是“随机得太粗”。如果有 100 个候选，其中 90 个已经明显差，固定 ε 仍会把探索预算分给这些坏动作，预算利用率很低。

UCB 的常见问题出现在非平稳环境。比如广告系统里，创意素材的有效期很短，用户群也会切换。标准 UCB 假设历史样本一直有用，因此会长期相信旧赢家。结果是新素材已经变好，但系统仍把流量压在旧素材上。

Thompson Sampling 在平稳 Bernoulli 场景里通常很强，但它也不是没有代价。首先，它依赖建模假设。点击任务适合 Beta-Bernoulli，但如果奖励是连续值、延迟值或强上下文相关，简单 Beta 后验就不够了。其次，TS 对“忘记旧信息”同样敏感，历史太长时也会滞后。

真实工程例子：广告投放里常见“早高峰”和“夜间流量”完全不同。上午 A 素材表现好，晚上 B 素材表现好。如果系统把全天历史混在一起，任何一种平稳 bandit 都会出问题。实践里通常会做以下处理：

- 分时段建模，而不是全局共享一套统计
- 对最近数据加更高权重
- 给新素材一个合理先验，而不是从绝对零开始
- 设置业务护栏，比如最低曝光、预算上限、频控

有些厂商案例会说 Thompson Sampling 相比固定 A/B 或静态规则能减少明显预算浪费。这个方向是可信的，但不能把某个百分比当成通用定律。因为收益取决于流量波动、候选数量、延迟反馈、回传噪声和先验设计。

---

## 替代方案与适用边界

如果场景不满足“少量动作、即时反馈、弱上下文”这些条件，探索策略也要升级。

| 替代 | 适用边界 | 限制 |
|---|---|---|
| 固定 A/B | 需要严格对照、结论可解释、预算充足 | 不能自适应，累计遗憾高 |
| 梯度带子 | 不直接估 $Q(a)$，而学习动作偏好 $H(a)$ | 对步长和数值稳定性敏感 |
| Bootstrap UCB | 非线性模型、复杂奖励结构 | 计算更重，调参更难 |
| Contextual Bandit | 动作效果依赖用户/上下文特征 | 特征偏差会直接影响探索 |
| 深度 RL 探索 | 长时序决策、状态会转移 | 训练复杂，样本成本高 |

梯度带子是一个值得单独提的替代方案。它不直接估计动作价值，而是学习动作偏好：

$$
\pi(a)=\frac{e^{H(a)}}{\sum_b e^{H(b)}}
$$

这里的偏好 $H(a)$，白话讲，就是“系统主观上更想选这个动作的程度”。它适合那些动作价值不好直接估、但偏好可以通过梯度更新的情况。不过它对学习率、基线项和数值稳定性更敏感，不如 Bernoulli 场景下的 TS 那么省心。

固定 A/B 也不是“落后方法”，而是目标不同。它适合需要明确统计检验、需要稳定对照实验的场景，比如功能上线评估、因果归因。但如果目标从“做实验”变成“边学边赚”，A/B 就不是最优工具。

最后要明确一条边界：本文讨论的是 bandit 级别的探索，不是完整强化学习里的长期状态探索。到了 MDP 场景，动作不只影响当前奖励，还会改变未来状态分布，此时仅靠 ε-greedy、UCB、TS 处理动作选择是不够的，往往需要结合值函数不确定性、参数噪声、内在奖励或乐观初始化。

---

## 参考资料

- Sutton, Barto. *Reinforcement Learning: An Introduction*，第 2 章 Bandits
- Auer, Cesa-Bianchi, Fischer. *Finite-time Analysis of the Multiarmed Bandit Problem*，2002
- Russo et al. *A Tutorial on Thompson Sampling*，2018
- System Overflow: *Core Bandit Algorithms: Epsilon Greedy, UCB, and Thompson Sampling*  
  https://www.systemoverflow.com/learn/ml-recommendation-systems/diversity-exploration/core-bandit-algorithms-epsilon-greedy-ucb-and-thompson-sampling
- Next Electronics: *Exploration vs Exploitation Strategies*  
  https://next.gr/ai/reinforcement-learning/exploration-vs-exploitation-strategies
- Microsoft Learn: *Test Run – The UCB1 Algorithm for Multi-Armed Bandit Problems*  
  https://learn.microsoft.com/en-us/archive/msdn-magazine/2019/august/test-run-the-ucb1-algorithm-for-multi-armed-bandit-problems
- Qi, Guo, Zhu. *Thompson Sampling for Non-Stationary Bandit Problems*，Entropy 2025  
  https://www.mdpi.com/1099-4300/27/1/51

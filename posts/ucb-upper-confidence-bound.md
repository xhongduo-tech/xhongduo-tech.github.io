## 核心结论

UCB（Upper Confidence Bound，上置信界）是一类多臂老虎机算法。它的核心做法不是“随机试一试”，而是每一轮都计算每个臂的**乐观估计**，再选分数最高的那个臂。公式写成：

$$
\text{UCB}_a(t)=\hat\mu_a(t)+c\sqrt{\frac{\ln t}{n_a(t)}}
$$

其中，$\hat\mu_a(t)$ 是臂 $a$ 当前的经验均值，白话说就是“这个动作到现在看起来平均有多好”；$n_a(t)$ 是它被选择过多少次；后面的加号项是**置信半径**，白话说就是“因为样本少，所以先给它一点乐观分”。

这套方法解决了 $\varepsilon$-greedy 的核心缺点：$\varepsilon$-greedy 会按固定概率乱试，但它不区分“谁更不确定”。UCB 会把探索预算优先给样本不足的臂，因此探索更有方向。

一个最小玩具例子：

- 广告 A：平均点击率 $0.62$，只展示了 $100$ 次
- 广告 B：平均点击率 $0.71$，已经展示了 $400$ 次
- 当前轮次 $t=1000$

则：

$$
\text{UCB}_A \approx 0.62+\sqrt{\frac{2\ln 1000}{100}} \approx 0.992
$$

$$
\text{UCB}_B \approx 0.71+\sqrt{\frac{2\ln 1000}{400}} \approx 0.896
$$

虽然 B 的当前均值更高，但 A 的样本更少，不确定性更大，所以 UCB 会先继续给 A 机会。这不是“偏心”，而是在避免过早把一个可能更优的臂判死。

一句话定义表如下：

| 术语 | 数学记号 | 白话解释 |
|---|---|---|
| 均值 | $\hat\mu_a(t)$ | 这个臂目前看起来平均有多好 |
| 置信半径 | $c\sqrt{\ln t / n_a(t)}$ | 因样本不足而额外补上的“乐观分” |
| 乐观估计 | $\hat\mu_a(t)+c\sqrt{\ln t / n_a(t)}$ | 假设这个臂可能比当前观测值更好一点 |

UCB 的理论基础来自 Hoeffding 不等式，因此它不是拍脑袋加一个 bonus，而是有明确概率解释的高概率上界。标准 UCB1 在经典假设下还能得到**对数级悔恨**，即长期损失增长很慢，这是它被广泛使用的核心原因。

---

## 问题定义与边界

多臂老虎机（multi-armed bandit）问题可以先这样理解：你面前有 $K$ 个按钮，每一轮只能按一个，按完只会看到这个按钮的奖励，其他按钮这一轮的结果你看不到。目标不是只让某一轮赢，而是在很多轮之后，让总奖励尽量高。

这类问题的标准形式是：

- 轮次：$t=1,2,\dots,T$
- 臂：$a\in\{1,\dots,K\}$
- 第 $t$ 轮只能选一个臂
- 如果选中臂 $a$，得到奖励 $r_{a,s}$
- 每个臂都有一个未知真实均值 $\mu_a$
- 目标是尽快逼近最优臂，同时减少试错损失

常用符号如下：

| 符号 | 含义 |
|---|---|
| $t$ | 当前轮次 |
| $n_a(t)$ | 到第 $t$ 轮时，臂 $a$ 已被拉取的次数 |
| $\hat\mu_a(t)$ | 臂 $a$ 的经验均值 |
| $r_{a,s}$ | 臂 $a$ 第 $s$ 次被选择时获得的奖励 |

“经验均值”这个词首次出现时可以直接理解成：把这个臂历史奖励求平均，作为当前对它真实价值的估计。

UCB 的适用前提要讲清楚，否则会误用。它通常假设：

- 奖励相对稳定，不会在短时间内剧烈漂移
- 同一个臂的奖励样本可以视为独立同分布，至少接近这个条件
- 奖励有界，经典 UCB1 常直接假设 $r\in[0,1]$
- 每轮只能看到自己选中的那个臂的反馈，也就是**单臂反馈**

一个新手版工程例子：推荐系统首页每次只能给用户展示一个商品位。你不能一开始就知道哪个商品点击率最高，只能一边曝光一边学。如果完全贪心，早期偶然跑高的数据会误导系统；如果完全随机，流量又浪费太多。UCB 就是在这两个目标之间做平衡。

下面这张表能快速说明“问题假设”和“不适用场景”的关系：

| 问题假设 | 典型含义 | 不适用场景 |
|---|---|---|
| 有界奖励 | 点击率、转化率这类值天然落在有限区间内 | 奖励无界且重尾，直接用 Hoeffding 版本不稳 |
| 独立同分布 | 同一臂的统计规律大致稳定 | 强漂移环境，例如热点内容昼夜切换非常快 |
| 单臂反馈 | 只知道自己选的那个动作结果 | 全信息反馈，例如每轮都能看到所有动作的真实收益 |

因此，UCB 不是“所有在线决策都能直接套”的万能模板。它更适合下面这类边界：

- 在线决策
- 每次试验有成本
- 反馈延迟不太长
- 希望尽快压低无效臂曝光
- 环境相对平稳，至少在一个时间窗口内近似平稳

---

## 核心机制与推导

UCB 的关键不是公式长什么样，而是理解“为什么 bonus 长这样”。

先看 Hoeffding 不等式。若奖励有界且样本独立，那么对任意臂 $a$，其真实均值 $\mu_a$ 与经验均值 $\hat\mu_a$ 之间有高概率界：

$$
P\left(\mu_a>\hat\mu_a+\sqrt{\frac{\ln(1/\delta)}{2n_a}}\right)\le \delta
$$

这句话的白话版本是：样本数为 $n_a$ 时，真实均值大幅高于经验均值的概率不会太大；而“可能高出去多少”，大致被一个 $\sqrt{1/n_a}$ 量级的项控制。

因此，一个自然的乐观估计就是：

$$
\mu_a \lesssim \hat\mu_a+\sqrt{\frac{\ln(1/\delta)}{2n_a}}
$$

再把失败概率 $\delta$ 设计成会随时间缩小的量，常见推导会得到 UCB1 形式：

$$
\text{UCB}_a(t)=\hat\mu_a(t)+\sqrt{\frac{2\ln t}{n_a(t)}}
$$

这里有三个非常重要的结构含义。

第一，bonus 随 $n_a$ 增大而缩小。  
因为 $\sqrt{1/n_a}$ 会下降，所以一个臂被拉得越多，不确定性越小，UCB 越接近它的真实均值估计。于是算法会从“积极探索”逐步过渡到“偏向利用”。

第二，bonus 随 $t$ 增大而缓慢增加。  
这里是 $\ln t$，不是 $t$。这表示随着总轮次变长，算法仍然会保留少量探索，但增长极慢。它的目的不是永远大量试错，而是防止某个臂因为早期偶然噪声被永久忽略。

第三，bonus 不是方差估计。  
很多初学者会把这项理解成“数据波动大，所以 bonus 大”，这不准确。原版 UCB1 的 bonus 主要来自**样本数量带来的置信区间宽度**，不是直接由经验方差算出来的。后者更接近 UCB-V 这类方差感知版本。

再看一个玩具例子。假设两个臂当前均值分别是：

- 臂 A：$\hat\mu_A=0.55,\; n_A=3$
- 臂 B：$\hat\mu_B=0.60,\; n_B=300$

如果当前 $t=1000$，则 A 的 bonus 会非常大，B 的 bonus 会非常小。即使 A 的当前均值低一点，UCB 仍可能选 A。原因不是“算法糊涂”，而是 3 次样本根本不足以说明 A 真差。这个机制本质上在说：**低样本不是低价值的证据，只是高不确定性的证据。**

从长期表现看，UCB1 的经典期望悔恨界为：

$$
\mathbb E[R_T]=O\left(\sum_{a:\Delta_a>0}\frac{\ln T}{\Delta_a}\right)
$$

其中 $\Delta_a=\mu^\*-\mu_a$，表示臂 $a$ 与最优臂之间的差距。  
“悔恨”这个词白话解释为：如果一开始就知道最优臂是谁，本来能拿到更多奖励；现在因为需要学习而损失掉的这部分，就是 regret。

这个式子告诉你两件事：

- 非最优臂被试错的总次数只会按 $\ln T$ 级别增长，不会线性爆炸
- 越接近最优臂的次优臂越难分辨，因此它们会被探索更多次，因为 $\Delta_a$ 越小，分母越小

如果把它画成图，通常会看到两条趋势线：

- 经验均值曲线：随着样本增多逐渐稳定
- bonus 曲线：开始很高，之后单调下降

这也是很多资料会建议配“均值曲线 + bonus 递减曲线”示意图的原因，因为它能直观看出 UCB 是怎么从探索走向利用的。

真实工程例子可以看广告冷启动。一个新广告刚上线，曝光数极少，CTR 的经验均值很不稳定。原版贪心法可能因为前几次没点就把它压到底；UCB 则会因为 $n_a$ 小而自动补足探索分，让新广告获得一定验证流量。这在新商品、新内容、新策略上线时非常常见。

---

## 代码实现

实现 UCB 时最常见的错误有两个：

- 忘记处理 $n_a=0$，导致除零
- 把 bonus 写成某种经验方差项，结果实现的已经不是 UCB1

最稳妥的初始化方式是：每个臂先拉一次。这样从第 $K+1$ 轮开始，每个臂都有定义良好的经验均值和拉取次数。

更新均值时不需要保存所有历史奖励，可以用增量更新：

| 变量 | 更新公式 |
|---|---|
| 拉取次数 | `pulls[a] += 1` |
| 经验均值 | `mean[a] = mean[a] + (reward - mean[a]) / pulls[a]` |

下面给出一个可运行的 Python 玩具实现。为了让例子可复现，代码里用固定随机种子，并在最后用 `assert` 做基本校验。

```python
import math
import random


class BernoulliBandit:
    def __init__(self, probs, seed=0):
        self.probs = probs
        self.rng = random.Random(seed)

    def pull(self, arm):
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0


def run_ucb1(probs, horizon, seed=0):
    bandit = BernoulliBandit(probs, seed=seed)
    k = len(probs)
    pulls = [0] * k
    means = [0.0] * k
    total_reward = 0.0
    chosen_arms = []

    # 初始化：每个臂先拉一次，避免 n_a = 0
    for arm in range(k):
        reward = bandit.pull(arm)
        pulls[arm] += 1
        means[arm] = reward
        total_reward += reward
        chosen_arms.append(arm)

    # 主循环
    for t in range(k + 1, horizon + 1):
        scores = []
        for arm in range(k):
            bonus = math.sqrt(2.0 * math.log(t) / pulls[arm])
            score = means[arm] + bonus
            scores.append(score)

        arm = max(range(k), key=lambda a: scores[a])
        reward = bandit.pull(arm)

        pulls[arm] += 1
        means[arm] += (reward - means[arm]) / pulls[arm]
        total_reward += reward
        chosen_arms.append(arm)

    return {
        "pulls": pulls,
        "means": means,
        "total_reward": total_reward,
        "chosen_arms": chosen_arms,
    }


if __name__ == "__main__":
    probs = [0.45, 0.50, 0.65]  # 第 2 号臂最优
    result = run_ucb1(probs=probs, horizon=2000, seed=42)

    best_arm = probs.index(max(probs))
    most_pulled_arm = max(range(len(probs)), key=lambda a: result["pulls"][a])

    print("pulls =", result["pulls"])
    print("means =", [round(x, 3) for x in result["means"]])
    print("total_reward =", result["total_reward"])

    assert sum(result["pulls"]) == 2000
    assert all(n >= 1 for n in result["pulls"])
    assert most_pulled_arm == best_arm
```

这段代码体现了两个实现要点。

第一，初始化和评分计算要分开。  
初始化是为了保证每个臂都至少有一次观测；评分阶段才真正使用

$$
\hat\mu_a+\sqrt{2\ln t / n_a}
$$

第二，经验均值更新和 UCB 评分计算要分开。  
`means` 反映数据本身；`bonus` 反映不确定性。两者职责不同，混在一起后很容易把算法逻辑写偏。

如果你不想“先拉一遍”，另一种写法是对未访问臂直接赋分 `+inf`，让它们优先被选中，直到每个臂都被访问过至少一次。这两种做法本质一致。

真实工程例子里，UCB 往往不是直接操作“硬币 0/1 奖励”，而是操作 CTR、CVR、播放完成率、停留时长等在线指标。只要这些指标能被合理压缩到有界区间，并且环境近似平稳，UCB 就可以作为一个简单而强的基线方法。

---

## 工程权衡与常见坑

UCB 的理论漂亮，但落地时要先看假设是否成立。很多“UCB 不好用”的案例，问题不在算法本身，而在数据条件和目标函数已经偏离了原始理论。

常见坑如下：

| 常见坑 | 后果 | 处理方式 |
|---|---|---|
| `n_a = 0` | 除零或未定义 | 每个臂先探索一次，或给未访问臂无限大优先级 |
| 奖励不在 `[0,1]` | Hoeffding 版理论不匹配 | 先归一化，或改用更匹配的浓缩界 |
| 非平稳环境 | 旧数据误导当前决策 | 用 sliding-window UCB 或 discount UCB |
| 把 bonus 当方差 | 错误理解机制，改错公式 | 明确它是置信半径，不是经验方差 |
| 忽略延迟反馈 | 统计口径错位 | 做延迟补账或改成异步更新策略 |

为什么“奖励不在 `[0,1]`”要特别强调？因为 UCB1 的经典推导依赖有界奖励。如果你直接把一个长尾收入值、无上界时长、或极端稀疏的大额回报丢进去，原有的高概率界就未必成立。实践中常见做法是：

- 对奖励做截断或归一化
- 使用次高斯假设对应的界
- 在高波动场景下改用更稳健的方法

非平稳环境是另一个高频问题。比如新品上架第一天表现一般，第二周因为活动流量暴涨而显著变好。原版 UCB 会把早期旧样本一直记在均值里，导致反应偏慢。此时更合适的是：

- 滑动窗口 UCB：只看最近一段时间的数据
- 折扣 UCB：越老的数据权重越低

还要专门解释一下为什么 $\ln t$ 不能省略。  
如果把 bonus 写成常数，比如 $c/\sqrt{n_a}$ 而没有 $\ln t$，那么随着总轮次增长，算法可能探索不足。原因是它没有随着“同时比较的次数增加”而调整置信要求。$\ln t$ 的作用，是让算法在长期运行时仍保留非常缓慢但必要的再探索能力，从而支撑对数悔恨这类理论结果。它不是装饰项，而是平衡“长期不漏掉优臂”和“不过度试错”的关键结构。

真实工程例子可以看推荐流量分发。假设一个视频推荐位在白天和夜间用户群差异极大。你若直接使用全量历史的原版 UCB，夜间可能仍被白天数据主导，导致当前最优内容上不来。这里问题不是 UCB 不会探索，而是它假设的“稳定分布”已经被打破。

---

## 替代方案与适用边界

UCB 最常被拿来和 $\varepsilon$-greedy、Thompson Sampling 做比较。

先说 $\varepsilon$-greedy。它的做法是：大部分时间选当前最好臂，少部分时间随机探索。优点是实现非常简单，缺点也直接：探索对象没有区分。一个已经试了很多次、明显很差的臂，仍可能因为随机探索被继续选中；而一个只试了几次、潜力未明的臂，并不会获得更高优先级。

UCB 的优点就在这里：它不是“随机撒点流量”，而是“把探索流量压到不确定的臂上”。

再看 Thompson Sampling。它的核心思想是：给每个臂维护后验分布，然后每轮从分布中采样一个价值，再按样本值选臂。它通常效果很强，尤其在贝叶斯建模合适时经常表现优秀。但它的动作选择带有显式随机性。若你需要向产品、运营、风控解释“为什么这一轮选了这个臂”，UCB 往往更容易说清楚，因为它的评分是确定性可分解的：均值多少，bonus 多少，一眼能拆开。

下面给出一个简表：

| 方法 | 核心特征 | 优点 | 局限 |
|---|---|---|---|
| $\varepsilon$-greedy | 固定概率随机探索 | 实现最简单 | 探索不分对象，浪费流量 |
| UCB | 均值 + 上置信界 | 确定性强、解释性好 | 依赖分布假设，对非平稳原版不友好 |
| Thompson Sampling | 从后验分布采样决策 | 常有很强经验效果 | 解释成本更高，需要先验/后验设计 |
| Sliding-window UCB | 只看最近窗口 | 适合漂移环境 | 窗口大小敏感，稳定期可能丢信息 |

一个新手版决策建议可以写得很直接：

- 想先有一个理论清楚、实现简单、解释方便的基线：用 UCB
- 环境变化快：考虑滑动窗口 UCB 或折扣 UCB
- 更重视经验性能，且能接受采样式策略：考虑 Thompson Sampling
- 只想快速搭一个最简基线：先上 $\varepsilon$-greedy，但别把它当长期方案

UCB 的适用边界也需要明确：

- 在线决策，离线批处理问题不一定适合
- 每次试验有成本，不能无限乱试
- 希望尽快压低明显无效的臂
- 需要较好的可解释性
- 环境至少在一段时间内近似稳定

因此，UCB 不是“最强 bandit 算法”的统一答案，但它是“最容易讲清楚，且理论与工程都足够扎实”的一类答案。对于零基础到初级工程师，理解 UCB 也有额外价值：它把“统计置信区间”和“在线决策”这两个常分开讲的主题连在了一起。

---

## 参考资料

先看论文理解理论来源，再看教材理解系统推导，最后看工程库理解接口和落地方式。下面这张表可作为阅读顺序建议：

| 来源类型 | 价值 | 适合阅读顺序 |
|---|---|---|
| 经典论文 | 看清 UCB1 的问题设定、算法与 regret 结果 | 1 |
| 教材 | 系统理解证明、变体和适用边界 | 2 |
| 课程资料 | 补 Hoeffding 不等式这类概率工具 | 3 |
| 工程库/代码 | 看实际 API、参数和示例 | 4 |

1. [Finite-time Analysis of the Multiarmed Bandit Problem - Auer, Cesa-Bianchi, Fischer (2002)](https://doi.org/10.1023/A:1013689704352)
2. [MIT OpenCourseWare - Hoeffding's Inequality](https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/resources/hoeffdings-inequality/)
3. [Bandit Algorithms - Lattimore & Szepesvari, Chapter 7: The Upper Confidence Bound Algorithm](https://www.cambridge.org/core/books/bandit-algorithms/upper-confidence-bound-algorithm/F84A107E30DB4B78A3323D6742971107)
4. [MABWiser: Contextual Multi-Armed Bandits Library](https://github.com/fidelity/mabwiser)

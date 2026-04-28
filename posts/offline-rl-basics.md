## 核心结论

离线 RL（Offline Reinforcement Learning）是只使用固定数据集 $D=\{(s,a,r,s')\}$ 学习策略 $\pi$ 的强化学习方法，训练阶段不再和环境交互。它的主线不是探索，而是控制分布偏移。

这里的分布偏移可以先用白话理解：训练数据里常见的状态和动作组合，模型见得多；训练数据里几乎没出现过的动作，模型只能“猜”。一旦策略 $\pi$ 倾向于选择这些没见过或很少见过的动作，价值函数 $Q$ 的误差就会被放大。

在线 RL 和离线 RL 的差异，可以先压缩成一张表：

| 维度 | 在线 RL | 离线 RL |
| --- | --- | --- |
| 是否与环境交互 | 是 | 否 |
| 数据来源 | 边采样边训练 | 固定历史数据 |
| 主要风险 | 探索成本高、样本效率低 | 分布偏移、外推误差 |
| 优化重点 | 平衡探索与利用 | 约束策略不要跑出数据支持集 |
| 常见失败模式 | 试错代价过高 | $Q$ 对数据外动作过度乐观 |

总判断很简单：离线 RL 的成败，主要取决于两件事。第一，critic 要保守，不能把数据外动作估得过高。第二，actor 要受约束，不能偏离行为策略 $\beta$ 太远。

---

## 问题定义与边界

先统一符号：

| 术语 | 定义 | 白话解释 |
| --- | --- | --- |
| $\beta$ | 行为策略 | 采集这批历史数据时真正执行动作的策略 |
| $\pi$ | 目标策略 | 我们希望学出来并最终部署的策略 |
| $Q(s,a)$ | 动作价值函数 | 在状态 $s$ 选动作 $a$，未来总回报有多大 |
| $\bar Q$ | target network | 用来稳定训练的延迟副本 |
| $\gamma$ | 折扣因子 | 未来奖励要不要打折，通常 $0<\gamma<1$ |
| OOD | Out-of-Distribution | 超出训练数据分布的状态或动作 |
| bootstrap | 自举更新 | 用当前模型的估计去构造下一步训练目标 |
| support set | 支持集 | 数据里真正覆盖过、可被经验支撑的区域 |

离线 RL 的输入是一个静态数据集：

$$
D=\{(s,a,r,s')\}
$$

有时还会附带 `done`、`timeout`、`info` 等字段，用来区分自然终止和时间截断。

它解决的问题是：给定这批固定轨迹，能不能学出一个比原行为策略 $\beta$ 更好的策略 $\pi$。它不解决的问题也必须说清楚：如果数据从来没覆盖过某些关键动作，那么算法不能凭空知道这些动作的真实价值。换句话说，离线 RL 不是“无数据也能推断世界真相”，而是“在已有数据支持的范围内尽量优化”。

一个玩具例子最容易说明边界。

设只有一个状态 $s_0$，两个动作 $a_1,a_2$。数据里只出现过 $(s_0,a_1,r=1)$，从没出现过 $a_2$。这时你最多能说：数据支持 $a_1$ 的价值大约是 1；至于 $a_2$，没有直接证据。离线 RL 的合理目标不是“猜出 $a_2$ 其实更好”，而是避免因为模型误差把 $a_2$ 错判为更优。

再用一张表概括问题边界：

| 能做什么 | 不能做什么 |
| --- | --- |
| 从静态数据中学习近似最优策略 | 对完全没覆盖的动作做可靠价值评估 |
| 复用昂贵或高风险场景中的历史日志 | 靠纯推断发现数据里从未验证过的新行为 |
| 在安全约束下做策略改进 | 弥补极差数据覆盖带来的不可辨识问题 |

因此，离线 RL 的前提不是“算法足够强”，而是“数据至少提供了可学习的支持集”。

---

## 核心机制与推导

离线 RL 仍然建立在 Bellman 备份上。Bellman 备份可以先白话理解为：当前动作的价值，等于眼前奖励加上下一步还能拿到的未来价值。

基础 TD 目标常写成：

$$
y=r+\gamma \mathbb E_{a'\sim \pi(\cdot|s')}[\bar Q(s',a')]
$$

对应的 critic 损失是：

$$
L_{TD}=\mathbb E_{(s,a,r,s')\sim D}\left[(Q(s,a)-y)^2\right]
$$

如果这是在线 RL，问题通常没那么尖锐，因为策略一旦去试某个动作，环境会给出真实反馈，错误估计会被后续样本修正。但离线 RL 不行。因为你不能再采样，$Q$ 对数据外动作的高估，一旦出现，就可能在 bootstrap 中一层一层传下去。

为什么会这样？看一个最小数值例子。

仍然在状态 $s_0$ 下有两个动作 $a_1,a_2$。数据里只有 $a_1$，且回报为 1。理想情况应该是：

$$
Q(s_0,a_1)\approx 1,\quad Q(s_0,a_2)\text{ 不确定}
$$

但如果函数逼近器因为初始化、噪声或泛化，把 $Q(s_0,a_2)$ 暂时估成 1.8，那么贪心策略就会倾向于选 $a_2$。接着，后续的 TD 目标又会把这个偏大的值反向喂给训练过程，于是“虚高”会被自举机制固化。这类由数据外动作引发的错误传播，通常称为外推误差。

所以离线 RL 一般沿两条线同时处理问题。

第一条线是约束 actor。actor 是策略网络，白话讲就是“真正决定选哪个动作的模块”。常见写法是：

$$
L_{\pi}=-\mathbb E_{s\sim D,a\sim \pi(\cdot|s)}[Q(s,a)] + \lambda D(\pi(\cdot|s)\|\beta(\cdot|s))
$$

第一项鼓励策略选择高价值动作，第二项惩罚它偏离行为策略 $\beta$。这里的 $D(\cdot\|\cdot)$ 可以是 KL 散度、MMD，或者其他距离。直观含义是：你可以比历史策略更好，但不能一下跳到历史数据从未支持的地方。

第二条线是让 critic 更保守。保守的意思不是故意把所有动作都估低，而是对数据外动作保持怀疑。CQL（Conservative Q-Learning）的典型思路是，在数据分布上的 $Q$ 值和更宽分布上的 $Q$ 值之间人为拉开差距，典型形式可以写成：

$$
\mathbb E_{s\sim D,a\sim \mu}[Q(s,a)]-\mathbb E_{s\sim D,a\sim \beta}[Q(s,a)]
$$

这里的 $\mu$ 是一个更宽的采样分布，可以覆盖潜在的 OOD 动作。这个差值被加入损失后，优化会压低那些“在宽分布里看起来很高、但数据里没有证据支持”的动作价值。

如果是离散动作空间，常用 `logsumexp` 近似宽分布上的上界：

$$
\log \sum_a \exp(Q(s,a))
$$

它的作用是把所有动作的高值都纳入惩罚，尤其是那些不在数据支持集里却被模型估高的动作。原因是 `logsumexp` 对较大的输入更敏感，相当于“盯着最可疑的高 Q 动作”。

把整个训练流程压缩成文字版机制图，就是：

数据集 $D$ $\rightarrow$ 更新 critic $\rightarrow$ 更新 actor $\rightarrow$ 检查策略是否偏离 $\beta$ 太远 $\rightarrow$ 继续训练。

真实工程例子也很典型。比如仓储机械臂抓取系统，历史数据来自人工示教和少量自动抓取日志。上线前不允许在线试错，因为撞坏货物或机械臂的代价太高。这时只能用离线 RL。从算法上看，核心不是让机械臂“想象一个全新抓取动作”，而是从历史上成功率较高的动作附近做受约束的改进。否则，$Q$ 一旦把某个极少见姿态估高，策略就可能输出一个从未被验证过的高风险动作。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是深度学习版本，而是用离散状态动作的表格方法演示“普通离线 Q 学习”和“带行为约束的保守更新”的差异。重点是把数据流和损失逻辑讲清楚。

```python
import math
from collections import defaultdict, Counter

# toy dataset: only a1 appears in state s0, reward is always 1
dataset = [
    ("s0", "a1", 1.0, "terminal", True),
    ("s0", "a1", 1.0, "terminal", True),
    ("s0", "a1", 1.0, "terminal", True),
    ("s0", "a1", 1.0, "terminal", True),
]

actions = ["a1", "a2"]
gamma = 0.99

def estimate_beta(dataset):
    counts = defaultdict(Counter)
    for s, a, r, s2, done in dataset:
        counts[s][a] += 1
    beta = {}
    for s, counter in counts.items():
        total = sum(counter.values())
        beta[s] = {a: counter[a] / total for a in actions}
    return beta

def fitted_q_iteration(dataset, actions, steps=20, alpha=0.5):
    Q = defaultdict(float)
    # simulate a bad initialization for unseen action
    Q[("s0", "a2")] = 1.8

    for _ in range(steps):
        for s, a, r, s2, done in dataset:
            target = r if done else r + gamma * max(Q[(s2, a2)] for a2 in actions)
            Q[(s, a)] = (1 - alpha) * Q[(s, a)] + alpha * target
    return Q

def conservative_policy(Q, beta, state, lam=4.0):
    scores = {}
    for a in actions:
        beta_prob = beta[state].get(a, 1e-8)
        # add log(beta) as a simple behavior regularizer
        scores[a] = Q[(state, a)] + lam * math.log(beta_prob + 1e-8)
    best_action = max(scores, key=scores.get)
    return best_action, scores

beta = estimate_beta(dataset)
Q = fitted_q_iteration(dataset, actions)

# ordinary greedy policy picks the unseen action because of optimistic Q
greedy_action = max(actions, key=lambda a: Q[("s0", a)])

# conservative policy is pulled back toward behavior support
safe_action, safe_scores = conservative_policy(Q, beta, "s0", lam=4.0)

assert round(Q[("s0", "a1")], 6) == 1.0
assert greedy_action == "a2"
assert safe_action == "a1"

print("Q(s0, a1) =", Q[("s0", "a1")])
print("Q(s0, a2) =", Q[("s0", "a2")])
print("greedy_action =", greedy_action)
print("safe_action =", safe_action)
print("safe_scores =", safe_scores)
```

这个例子里，`a2` 在数据中完全没出现，但我们故意给它一个偏高初始化，模拟外推误差。普通贪心策略会选 `a2`，而加了行为约束后，策略会被拉回到数据支持较强的 `a1`。

真正工程实现一般拆成四块。

第一块是数据加载。离线 RL 的 batch 不只是 `(s,a,r,s')`，通常还要区分 `done` 和 `timeout`。`done=True` 表示环境真正终止，TD 目标不应该再加未来价值；`timeout=True` 往往只是轨迹因为时间上限被截断，下一状态仍然有价值。如果把两者混为一谈，$Q$ 会系统性偏低。

第二块是 critic 更新。连续动作场景下通常会写成：

```python
for batch in dataloader:
    s, a, r, s2, done = batch

    with torch.no_grad():
        a2 = pi.sample(s2)
        y = r + gamma * (1 - done) * q_target(s2, a2)

    critic_loss = mse(q(s, a), y) + conservative_term(q, s, batch)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
```

第三块是 actor 更新。要么显式学习行为策略 $\beta$，再加 KL 约束；要么直接用生成式约束、动作候选筛选等办法限制策略偏移。

```python
a_pi = pi.sample(s)
actor_loss = -q(s, a_pi).mean() + lambda_ * kl_to_beta(pi, beta, s)
actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()
```

第四块是 target network 同步，也就是软更新：

$$
\bar \theta \leftarrow \tau \theta + (1-\tau)\bar \theta
$$

这里 $\tau$ 很小，作用是让训练目标变化更平滑。

如果是离散动作场景，CQL 的保守项常用 `logsumexp(Q(s, ·)) - Q(s, a_{data})`。如果是连续动作场景，不能枚举所有动作，就常从当前策略、随机动作分布或行为模型附近采样一批动作，近似估计宽分布上的高 Q 区域。

工程上建议至少记录以下指标：

| 指标 | 作用 |
| --- | --- |
| 数据内动作的平均 $Q$ | 看模型是否学到基础价值结构 |
| 随机或策略采样动作的平均 $Q$ | 检查是否存在 OOD 高估 |
| 策略熵 | 看策略是否过早塌缩 |
| $\pi$ 与 $\beta$ 的距离 | 监控策略偏移是否失控 |
| 成功率/回报的离线评估值 | 作为趋势参考，不应单独相信 |

---

## 工程权衡与常见坑

离线 RL 最危险的地方，不是训练不收敛，而是“看起来收敛了，但策略很自信地做错事”。因为没有在线反馈，很多错误不会被自动纠正。

最常见的坑如下：

| 问题 | 后果 | 规避方式 |
| --- | --- | --- |
| $Q$ 过于乐观 | 策略被虚高动作吸引 | 使用保守 critic，如 CQL 风格惩罚 |
| $\pi$ 偏离 $\beta$ 太远 | 进入数据外区域，性能崩溃 | 加 KL/MMD/行为生成约束 |
| $\beta$ 估得过平滑 | 真正有价值的稀有动作也被压掉 | 调整行为模型容量和正则强度 |
| 混淆 `terminal` 与 `timeout` | TD 目标错误，Q 值漂移 | 数据预处理时显式区分终止原因 |
| 数据覆盖差 | 再强算法也学不出可靠策略 | 先做 coverage 检查，再决定是否训练 |
| 用同一批轨迹做训练和结论性评估 | 指标虚高，部署失败 | 留出独立验证集，结合 OPE 或安全回放 |

有一个误区要单独说：把离线 RL 当成普通监督学习。比如只学习 $(s,a)\to r$，或者直接模仿动作。这种做法当然有用，但它解决的是短期奖励拟合或行为复现，不是长期回报优化。强化学习中的“长期”来自折扣累积：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

如果只盯着单步奖励，很容易错过“眼前差一点、长期更好”的策略。

再看一个真实工程坑。推荐系统里常用历史曝光日志训练策略。如果日志只记录了系统当时展示过的内容，那么未曝光内容几乎没有反馈。此时若直接做离线 RL 并最大化 $Q$，模型可能把缺少反馈的内容误估成高收益，从而在部署时大量推荐未验证物料。工程上通常会先做强行为约束，甚至先从行为克隆或保守策略改进起步，而不是直接追求“看起来更优”的激进策略。

一条实用顺序是：

1. 先检查数据覆盖。
2. 再做保守 critic。
3. 最后加 policy constraint。

这个顺序重要，因为如果第一步不过关，后面只是把不可靠训练过程包装得更复杂。

---

## 替代方案与适用边界

离线 RL 方法的差异，主要体现在两件事上：怎么限制策略偏移，怎么限制 $Q$ 的外推误差。

常见方法可以先横向比较：

| 方法 | 核心思想 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| BC | 直接模仿数据中的动作 | 稳定、简单、容易训练 | 不能超越数据平均行为太多 | 数据质量高，目标是安全复现 |
| Behavior Regularized Offline RL | 在策略优化时显式贴近 $\beta$ | 兼顾改进和保守 | 若正则过强，提升有限 | 行为策略较稳定、希望小步改进 |
| BCQ | 生成接近数据分布的候选动作，再从中选优 | 明确限制支持集外动作 | 连续动作实现较复杂 | 连续控制、需要强动作约束 |
| CQL | 对数据外动作的 Q 值施加保守惩罚 | 对 OOD 高估控制直接 | 过度保守时可能压低真实好动作 | 数据噪声较大、需要稳健性 |

BC 是行为克隆，白话讲就是“看历史怎么做，我就怎么做”。它通常是最稳的基线，也经常比实现不当的复杂离线 RL 更可靠。但它的上限受限于数据本身，很难主动做策略改进。

Behavior Regularized Offline RL 的思路更像“在模仿附近优化”。它承认 $Q$ 可以指导改进，但要求改进幅度受控。

BCQ（Batch-Constrained Q-learning）强调候选动作必须来自接近数据分布的生成器，再在这些候选里挑高 Q 动作。它对“支持集约束”表达得很明确。

CQL 则更关注 critic 侧，核心是“宁可低估一点，也不要对 OOD 动作盲目乐观”。如果数据噪声大、覆盖杂乱，CQL 往往比纯 actor 约束更稳。

什么时候不该优先上离线 RL，也要说清楚：

| 场景 | 原因 |
| --- | --- |
| 数据极少 | 连行为策略都估不稳，更别提策略改进 |
| 覆盖极差 | 无法支撑可靠的价值学习 |
| 任务依赖强探索 | 固定数据无法发现全新有效行为 |
| 回报延迟极长且日志质量差 | bootstrap 误差和标注噪声会叠加 |
| 只要求安全复现现有流程 | 行为克隆可能更便宜、更稳 |

所以，离线 RL 不是“只要有日志就能学策略”的通用答案。它更适合这样一类问题：在线试错成本高，但历史数据足够多，且覆盖了关键决策区域的一部分。此时，保守地从历史行为中榨取增益，才是它最合理的位置。

---

## 参考资料

1. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
2. [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
3. [CQL 论文 PDF](https://arxiv.org/pdf/2006.04779.pdf)
4. [Off-Policy Deep Reinforcement Learning without Exploration (BCQ)](https://proceedings.mlr.press/v97/fujimoto19a.html)
5. [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://github.com/Farama-Foundation/D4RL)
6. [Behavior Regularized Offline Reinforcement Learning](https://www.acml-conf.org/2021/conference/accepted-papers/156/)

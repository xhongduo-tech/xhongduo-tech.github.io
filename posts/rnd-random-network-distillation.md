## 核心结论

RND（Random Network Distillation，随机网络蒸馏）是一种给强化学习智能体添加“内在奖励”的探索方法。内在奖励的白话解释是：环境本身不给分时，先用一个额外规则鼓励智能体主动去没见过的地方。RND 的规则很直接：如果当前状态很少见，模型对它的预测误差就大；误差越大，探索奖励越高。

它的核心公式是：

$$
r_t^{int} = \|f^*(s_t) - f_\theta(s_t)\|^2
$$

其中，$f^*$ 是固定随机网络，意思是参数随机初始化后永远不更新；$f_\theta$ 是可训练预测器，意思是它会不断学习去模仿 $f^*$ 对状态 $s_t$ 的输出。对同一个状态看得越多，$f_\theta$ 越容易学会，误差越小；第一次见到的新状态，误差通常更大。

这意味着 RND 奖励的不是“做对任务”，而是“到达不熟悉的状态”。它特别适合稀疏奖励任务。稀疏奖励的白话解释是：环境很少直接告诉你“你做对了”，例如走了几百步以后才第一次拿到钥匙。

一个最小判断表可以先记住：

| 状态类型 | 预测误差 | 内在奖励 | 对策略的作用 |
| --- | --- | --- | --- |
| 反复见过的状态 | 小 | 低 | 不再鼓励继续停留 |
| 刚到达的新状态 | 大 | 高 | 鼓励继续探索 |
| 完全随机噪声状态 | 可能一直很大 | 可能虚高 | 可能误导策略 |

---

## 问题定义与边界

RND 解决的问题不是“如何理解任务”，而是“如何在没有明显奖励时继续探索”。更准确地说，它处理的是探索不足问题：策略很容易卡在熟悉区域反复行动，因为外在奖励太稀少，优化器看不到继续往前走的信号。

总奖励通常写成：

$$
r_t = r_t^{ext} + \beta \cdot \mathrm{norm}(r_t^{int})
$$

这里 $r_t^{ext}$ 是环境给的外在奖励，$\beta$ 是缩放系数，控制内在奖励的权重；$\mathrm{norm}(\cdot)$ 表示归一化，白话解释是把数值尺度压到稳定范围，避免一个奖励源把另一个完全淹没。

RND 适合的边界很明确：

| 场景 | 是否适合 RND | 原因 |
| --- | --- | --- |
| 稀疏奖励、长时序探索 | 适合 | 外在奖励少，内在奖励能持续提供信号 |
| 状态较稳定、环境近似确定性 | 适合 | 预测误差更容易对应“是否见过” |
| 奖励已经很密集 | 一般 | 额外探索信号收益有限 |
| 强随机噪声环境 | 不适合或需谨慎 | 模型可能把噪声误认成新奇 |
| 高维连续观测但结构稳定 | 常见适用 | 神经网络可直接学表征 |

玩具例子可以看一个迷宫。迷宫里只有走到终点才有 $+1$，中间几百步都是 $0$。如果没有探索奖励，智能体可能只在起点附近乱转，因为这些动作的长期差别很难从样本里看出来。加上 RND 后，第一次走进新走廊、第一次进入新房间、第一次看到新布局，都会得到额外奖励。这样策略先学会“去新地方”，之后才更可能碰到真正的终点奖励。

真实工程例子是 Atari 的 `Montezuma's Revenge`。这个任务出名的原因不是动作复杂，而是有效事件太少。拿钥匙、跳平台、进新房间这些步骤之间可能隔着很长的无奖励过程。RND 与 PPO 结合后，可以把“第一次到达某个房间”也变成学习信号，因此显著改善探索。

但边界也必须说清：RND 不是严格的不确定性估计器。不确定性估计器的白话解释是：它要区分“模型还没学会”和“世界本身就随机”。RND 主要依赖预测误差，而预测误差在有噪声时不一定代表有价值的新知识。

---

## 核心机制与推导

RND 只有两个核心部件：

| 模块 | 是否更新 | 作用 |
| --- | --- | --- |
| $f^*$ 固定随机目标网络 | 否 | 给每个状态产生一个固定目标向量 |
| $f_\theta$ 预测器网络 | 是 | 学习逼近目标向量 |

机制可以分成三步。

第一步，给状态做编码。状态 $s_t$ 输入两个网络，分别得到两个向量：

$$
y_t^* = f^*(s_t), \quad \hat{y}_t = f_\theta(s_t)
$$

第二步，用均方误差计算新奇度：

$$
r_t^{int} = \|y_t^* - \hat{y}_t\|_2^2
$$

均方误差的白话解释是：把每个维度的差平方后再求和，数值越大代表预测越不准。

第三步，只训练预测器：

$$
\min_\theta \; \mathbb{E}_{s_t} \left[\|f^*(s_t)-f_\theta(s_t)\|^2\right]
$$

因为 $f^*$ 不动，学习过程只有一个方向：让 $f_\theta$ 对见过的状态越来越熟。于是“误差下降”可以近似看成“熟悉度上升”。这不是数学上对访问次数的严格计数，但在实践中经常能起到类似作用。

玩具数值例子最直观。设某个状态 $s$ 的目标输出和预测输出分别是：

- $f^*(s) = [1.2, -0.8]$
- $f_\theta(s) = [1.0, -0.5]$

那么它的内在奖励是：

$$
r^{int} = (1.2 - 1.0)^2 + (-0.8 - (-0.5))^2 = 0.04 + 0.09 = 0.13
$$

如果训练一段时间后，预测器学成：

- $f_\theta(s) = [1.2, -0.8]$

那么：

$$
r^{int} = 0
$$

这就是 RND 的关键逻辑：同一个状态不会永远带来高奖励，奖励会随“见过次数”逐渐衰减。

为什么随机网络也能工作？原因不是“随机有智慧”，而是随机映射在高维空间里能把不同状态投影成不同向量。预测器只要学会复现这些向量，就等于学会区分“哪些输入已经见过”。这里并不要求目标网络有语义，只要求它固定且足够区分输入。

和 ICM（Intrinsic Curiosity Module，内在好奇心模块）的差别也很关键。ICM 通常学习的是“给定当前状态和动作，预测下一个状态特征”，所以它依赖动作和动力学。RND 不需要逆动力学模型，也不要求知道“动作导致了什么变化”，它只看当前状态本身是否难预测，因此结构更简单。

---

## 代码实现

下面给一个可运行的极简 Python 版本。它不是完整 PPO 训练器，但能准确展示 RND 的核心计算：固定目标、训练预测器、误差下降、奖励衰减。

```python
import numpy as np

rng = np.random.default_rng(0)

class LinearNet:
    def __init__(self, in_dim, out_dim, trainable=True):
        self.W = rng.normal(0, 0.5, size=(out_dim, in_dim))
        self.b = rng.normal(0, 0.1, size=(out_dim,))
        self.trainable = trainable

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return self.W @ x + self.b

    def train_step(self, x, target, lr=0.05):
        assert self.trainable
        x = np.asarray(x, dtype=np.float64)
        pred = self(x)
        diff = pred - target
        loss = np.mean(diff ** 2)

        # d/dW mean((Wx+b-target)^2)
        grad_W = (2.0 / diff.size) * np.outer(diff, x)
        grad_b = (2.0 / diff.size) * diff

        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return loss

def rnd_reward(target_net, predictor_net, state):
    target = target_net(state)
    pred = predictor_net(state)
    return float(np.mean((target - pred) ** 2))

# 固定随机目标网络
target_net = LinearNet(in_dim=3, out_dim=4, trainable=False)
# 可训练预测器
predictor_net = LinearNet(in_dim=3, out_dim=4, trainable=True)

seen_state = np.array([1.0, 0.0, 0.0])
novel_state = np.array([0.0, 1.0, 1.0])

before_seen = rnd_reward(target_net, predictor_net, seen_state)
before_novel = rnd_reward(target_net, predictor_net, novel_state)

# 只反复训练 seen_state，模拟“这个状态已经见过很多次”
for _ in range(300):
    target = target_net(seen_state)
    predictor_net.train_step(seen_state, target, lr=0.05)

after_seen = rnd_reward(target_net, predictor_net, seen_state)
after_novel = rnd_reward(target_net, predictor_net, novel_state)

assert after_seen < before_seen
assert after_novel > 0.0
assert after_seen < after_novel

print("seen reward before:", round(before_seen, 6))
print("seen reward after :", round(after_seen, 6))
print("novel reward after:", round(after_novel, 6))
```

这段代码体现了三个工程事实：

1. `target_net` 固定不训练。
2. `predictor_net` 只在见过的状态上拟合目标输出。
3. 同一个状态被反复训练后，奖励会下降；没训练过的状态，奖励仍然偏高。

如果把它接进真实强化学习循环，流程通常是：

| 步骤 | 动作 |
| --- | --- |
| 1 | 从环境采样 `obs_t` |
| 2 | 用 RND 计算 `r_t^{int}` |
| 3 | 对 `r_t^{int}` 做 running normalization |
| 4 | 组合成总奖励或分开计算两套 return |
| 5 | 用 PPO/A2C 更新策略 |
| 6 | 用同批状态更新预测器 |

真实工程里常见的做法不是简单地把奖励硬相加，而是为外在奖励和内在奖励分别维护 value head。value head 的白话解释是：策略网络里专门预测未来回报的一支输出头。这样做的原因是两类奖励的统计性质不同，混在一个值函数里容易让训练不稳定。

以 Atari 为例，环境每步返回图像观测 `obs_t`。常见管线是：先经过卷积编码，再送入 RND 模块计算新奇度，同时 PPO 用策略头输出动作分布，用两套价值头分别估计 extrinsic return 和 intrinsic return。最后优势函数和损失项按各自尺度组合。

---

## 工程权衡与常见坑

RND 的难点从来不在公式，而在奖励语义是否稳定。

| 问题 | 表现 | 影响 | 常见处理 |
| --- | --- | --- | --- |
| 内在奖励尺度漂移 | 前期很大，后期骤降 | 压过外在奖励或训练震荡 | running mean/std 归一化 |
| noisy-TV 问题 | 噪声状态一直高奖励 | 策略追逐无意义随机性 | 选稳定特征、过滤噪声 |
| 预测器太弱 | 到处都预测不好 | 所有状态都像“新状态” | 增大容量或改善表征 |
| 预测器太强 | 很快全学会 | 探索奖励过早消失 | 控制容量、学习率、更新频率 |
| 输入表示差 | 关注像素噪声不关注结构 | 新奇信号失真 | 用共享编码器或稳定预处理 |
| 奖励直接硬加 | 策略目标冲突 | PPO value 学不稳 | 分离 value head 和折扣因子 |

一个常用归一化形式是：

$$
\tilde{r}_t^{int} = \frac{r_t^{int} - \mu_t}{\sigma_t + \epsilon}
$$

这里 $\mu_t$ 和 $\sigma_t$ 是运行中的均值与标准差。它的作用不是改变排序，而是防止某一段训练里内在奖励绝对值过大。

错误用法和推荐用法可以直接对照：

| 错误用法 | 推荐用法 |
| --- | --- |
| 直接把未归一化的 `r_int` 加到环境奖励 | 先做 running normalization 再缩放 |
| 把像素级随机闪烁直接送入 RND | 尽量使用稳定、可压缩的状态表示 |
| 只看总回报，不分内外奖励统计 | 单独监控 extrinsic / intrinsic return |
| 看到效果差就一味增大 `beta` | 先检查奖励尺度、预测器容量、状态噪声 |

真实工程里最常见的误解是：误差越大越好。这个判断是错的。RND 想要的不是“永远预测不好”，而是“对真正没见过的状态暂时预测不好”。如果预测器永远学不会任何状态，RND 就退化成常数噪声源；如果预测器几步就学会所有状态，探索信号又会过快消失。

---

## 替代方案与适用边界

RND 不是唯一探索方案，它和其他方法的区别主要在“奖励信号来自哪里”。

| 方法 | 核心信号 | 优势 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| RND | 状态预测误差 | 结构简单、易与 PPO 结合 | 对噪声敏感 | 稀疏奖励、状态可压缩 |
| ICM | 动力学预测误差 | 强调可控变化 | 依赖动作与动力学建模 | 交互结构明显的任务 |
| 计数法/伪计数 | 访问频率 | 解释性强 | 高维连续状态难直接计数 | 离散或可哈希状态 |
| 熵奖励 | 策略分布熵 | 实现简单 | 鼓励动作随机，不一定到新区域 | 早期探索增强 |
| 参数噪声 | 参数级随机扰动 | 行为探索更稳定 | 信号较粗 | 连续控制或简单探索 |

如果目标是“尽快找到没去过的区域”，RND 往往比 ICM 更直接，因为它不需要判断某个动作是否导致了可预测变化。如果目标更接近“学习可控动力学”，ICM 可能更贴近需求。

还要强调一个适用边界：RND 更像访问频率或可压缩性信号，而不是严格的 epistemic uncertainty。epistemic uncertainty 的白话解释是：模型因为数据不够而产生的“知识性未知”。RND 没有显式区分“我没见过”和“这个世界本来就乱”，所以在高噪声环境里不能把它当成严格不确定性估计来解释。

因此，选择 RND 的判断标准通常是四个问题：

1. 外在奖励是否稀疏？
2. 环境观测是否相对稳定，而不是充满不可约噪声？
3. 状态是否能被神经网络学成较稳定表征？
4. 你要的是“找新区域”，还是“建模动作导致的变化”？

前 3 个问题大多回答“是”时，RND 通常值得优先尝试。

---

## 参考资料

1. [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
2. [Reinforcement learning with prediction-based rewards](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/)
3. [openai/random-network-distillation](https://github.com/openai/random-network-distillation)
4. [Curiosity-driven Exploration by Self-supervised Prediction](https://proceedings.mlr.press/v70/pathak17a.html)

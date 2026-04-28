## 核心结论

TRPO（Trust Region Policy Optimization，信赖域策略优化）是一种**带约束的策略梯度方法**。策略梯度可以理解为“直接沿着能提高回报的方向更新策略参数”；TRPO 在这个基础上加了一条硬约束：**新策略不能离旧策略太远**。这里的“太远”通常用 KL 散度衡量，KL 散度可以理解为“两个概率分布差了多少”。

它解决的核心问题不是“怎样让每一步收益涨得最多”，而是“怎样避免一次更新把已经学到的策略毁掉”。普通策略梯度常见的问题是：方向可能没错，但步长太大，导致动作分布突变，训练回报突然掉下去。TRPO 的策略是：**先限制改动幅度，再在这个安全范围里尽量提升目标**。

它的核心优化问题是：

$$
\max_{\theta}\; \mathbb E_{s,a\sim \pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A_{\text{old}}(s,a)\right]
$$

$$
\text{s.t.}\;\mathbb E_{s\sim \pi_{\text{old}}}\left[D_{KL}\big(\pi_{\text{old}}(\cdot|s)\|\pi_\theta(\cdot|s)\big)\right]\le \delta
$$

其中，优势函数 $A(s,a)$ 可以理解为“某个动作比当前平均水平好多少”；$\delta$ 是允许的最大策略变化幅度。

下面这张表先把普通策略梯度和 TRPO 的差别定死：

| 方法 | 优化目标 | 是否限制策略变化 | 常见结果 |
|---|---|---:|---|
| Vanilla Policy Gradient | 直接增大期望回报 | 否 | 简单，但容易更新过猛 |
| TRPO | 增大代理目标 | 是，约束平均 KL | 更稳，但实现复杂 |
| PPO | 增大裁剪后目标 | 近似限制 | 工程上更常用 |

一个玩具例子就能说明 TRPO 的判断标准。假设旧策略在某状态下选动作 1 的概率是 0.8，选动作 0 的概率是 0.2。优势分别是：

- $A(s,1)=+2$
- $A(s,0)=-1$

如果候选新策略把概率改成 $(0.9, 0.1)$，代理目标提高了，而且 KL 很小，这种更新通常会被接受。  
如果候选新策略改成 $(0.99, 0.01)$，代理目标更高，但 KL 明显超限，TRPO 会拒绝。  
结论很直接：**TRPO 接受“提升但不过界”的更新，拒绝“提升更大但过界”的更新。**

---

## 问题定义与边界

TRPO 要解决的问题非常具体：**策略更新步长过大，导致性能崩溃**。

在强化学习里，策略是一个概率分布，输出“在状态 $s$ 下选动作 $a$ 的概率”。训练时我们不断调整参数 $\theta$，希望高回报动作概率变大、低回报动作概率变小。但普通策略梯度只告诉你“该往哪里走”，并不天然保证“别走太远”。这就会出现一个常见现象：一次梯度更新后，动作分布改得过猛，策略突然失去原本已经学会的行为。

对零基础读者，一个更具体的理解是：如果旧策略已经能让机器人勉强稳定地走，普通策略梯度可能因为某次大更新，把“抬腿”和“落脚”的概率结构一下改乱，结果机器人直接摔倒。TRPO 的作用不是让它学得更快，而是防止这类“从能走变成不会走”的退化。

TRPO 的适用边界也很明确：

| 维度 | 适合 | 不适合 |
|---|---|---|
| 数据来源 | on-policy，数据由当前策略采样 | off-policy，来自回放池或历史日志 |
| 动作空间 | 连续控制、较平滑策略 | 极大离散动作空间时代价偏高 |
| 目标 | 控制策略漂移、提升训练稳定性 | 极限吞吐、极低时延训练 |
| 价值估计 | 有相对可靠的 value/advantage 估计 | 优势噪声极大、value 函数很差 |

这里要强调一个边界：TRPO 是 **on-policy** 方法。on-policy 可以理解为“训练所用的数据，必须是当前这版策略自己采出来的”。如果你拿很久之前的旧日志数据，或者拿别的策略采样的数据硬套 TRPO，它依赖的代理目标近似和 KL 约束解释都会变弱，理论前提不再成立。

它也不是解决一切不稳定问题的通用药。以下问题不在 TRPO 的主要解决范围内：

- 奖励设计错误
- 状态观测信息不足
- value function 拟合严重失真
- 环境高噪声导致梯度方差爆炸
- 离线数据分布偏移

所以问题边界可以压缩成一句话：**TRPO 解决的是“当前策略梯度更新太激进”这个局部问题，不是整个强化学习系统的稳定性总开关。**

---

## 核心机制与推导

TRPO 的关键做法是把“直接最大化真实回报”改成“最大化一个局部可信的代理目标，并限制新旧策略差异”。

### 1. 代理目标

TRPO 不直接优化真实回报，而是优化下面这个代理目标：

$$
L(\theta)=\mathbb E_{s,a\sim \pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A_{\text{old}}(s,a)\right]
$$

这里的比值

$$
r_\theta(s,a)=\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}
$$

叫做**重要性采样比率**，白话解释就是：它用来衡量“新策略对旧样本里的这个动作，变得更偏爱还是更不偏爱”。

如果某个动作优势 $A_{\text{old}}(s,a)$ 为正，说明它比平均水平更好，那么 TRPO 倾向于让对应概率变大；如果优势为负，就倾向于压低它。

### 2. KL 约束

只优化代理目标还不够，因为代理目标的可信性来自“新策略离旧策略不远”。所以 TRPO 加上平均 KL 约束：

$$
\bar D_{KL}(\pi_{\text{old}}\|\pi_\theta)=\mathbb E_{s\sim \pi_{\text{old}}}\left[D_{KL}(\pi_{\text{old}}(\cdot|s)\|\pi_\theta(\cdot|s))\right]\le \delta
$$

这里必须注意方向：通常写的是 $D_{KL}(\pi_{\text{old}} \| \pi_\theta)$，不是反过来。因为 KL 不是对称的，方向写反，约束的几何意义就变了。

### 3. 二阶近似

在当前参数 $\theta_k$ 附近，用 $\Delta=\theta-\theta_k$ 表示更新量。TRPO 做两个局部近似：

- 代理目标的一阶近似：
$$
L(\theta_k+\Delta)\approx g^\top \Delta
$$

- KL 约束的二阶近似：
$$
\bar D_{KL}(\theta_k,\theta_k+\Delta)\approx \frac12 \Delta^\top H\Delta
$$

其中：

- $g=\nabla_\theta L(\theta)\vert_{\theta=\theta_k}$ 是策略梯度
- $H$ 是 KL 关于参数的 Hessian，可理解为“局部曲率矩阵”

于是原问题变成：

$$
\max_\Delta g^\top \Delta
\quad
\text{s.t.}\quad
\frac12\Delta^\top H\Delta\le \delta
$$

这是一个标准二次约束线性优化问题，解是：

$$
\Delta^*=\sqrt{\frac{2\delta}{g^\top H^{-1}g}}\,H^{-1}g
$$

这个方向 $H^{-1}g$ 就是**自然梯度方向**。自然梯度可以理解为“在参数空间里考虑分布几何之后，更合理的上升方向”，不是只看欧氏距离的普通梯度。

### 4. 为什么不用显式 Hessian

真实工程里，策略网络参数很多，$H$ 可能是百万维矩阵，不可能直接构造再求逆。TRPO 的关键工程技巧是只计算 **Hessian-vector product**，即 $Hv$。白话解释是：不把整张大矩阵摊开，而是只回答“这个矩阵乘一个向量的结果是什么”。

这样就能用**共轭梯度法**求解线性方程：

$$
Hx=g
$$

求得的 $x\approx H^{-1}g$，再按上面的系数缩放步长。

### 5. line search 为什么还要做

二阶近似只在局部成立。即使理论步长满足近似 KL 约束，真实更新后实际 KL 也可能超限，所以 TRPO 还会做 **backtracking line search**，也就是“从候选步长开始，不行就不断缩小，直到真实代理目标改善且真实 KL 合规”。

整个机制可以压缩成流程：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 采样 trajectories | 当前策略 | 状态、动作、回报 | 获得 on-policy 数据 |
| 估计优势 | 轨迹、value 函数 | $A_t$ | 判断动作好坏 |
| 计算梯度 $g$ | 旧样本、代理目标 | 一阶梯度 | 给出提升方向 |
| 定义 $Hv$ | KL 函数、向量 $v$ | 乘积结果 | 间接提供曲率 |
| 共轭梯度 | $Hv$、$g$ | $x\approx H^{-1}g$ | 求自然梯度方向 |
| 步长缩放 | $x$、$\delta$ | 候选步长 | 控制近似 KL |
| line search | 候选参数 | 接受/拒绝 | 保证真实约束 |

真实工程例子通常出现在 MuJoCo 连续控制任务，比如 Hopper、Walker2d、Ant。当前策略先采一批 rollout，再用 GAE 算优势，接着共轭梯度求自然梯度方向，最后 line search 检查真实 KL。和普通大步更新相比，TRPO 往往能让训练曲线少一些断崖式下跌。

---

## 代码实现

TRPO 的实现关键不在“把公式抄全”，而在把职责拆对。一个最小实现通常包括四块：

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `policy_loss` | 计算代理目标 | `logp_old`, `logp_new`, `adv` | 标量 loss |
| `kl_divergence` | 计算平均 KL | 旧策略分布、新策略分布 | 标量 KL |
| `conjugate_gradient` | 解 `Hx=g` | `Avp` 函数、梯度向量 | 方向向量 |
| `line_search` | 回溯找可接受更新 | 当前参数、候选步长 | 新参数 |

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示 TRPO 最关键的两个判断：代理目标是否提高、KL 是否超限。

```python
import math

def surrogate_objective(pi_old, pi_new, advantages):
    total = 0.0
    for a in pi_old:
        ratio = pi_new[a] / pi_old[a]
        total += pi_old[a] * ratio * advantages[a]
    return total

def kl_old_new(pi_old, pi_new):
    return sum(pi_old[a] * math.log(pi_old[a] / pi_new[a]) for a in pi_old)

def accept_update(pi_old, pi_new, advantages, delta):
    obj = surrogate_objective(pi_old, pi_new, advantages)
    kl = kl_old_new(pi_old, pi_new)
    return obj, kl, kl <= delta

pi_old = {0: 0.2, 1: 0.8}
advantages = {0: -1.0, 1: 2.0}
delta = 0.05

candidate_a = {0: 0.1, 1: 0.9}
candidate_b = {0: 0.01, 1: 0.99}

obj_a, kl_a, ok_a = accept_update(pi_old, candidate_a, advantages, delta)
obj_b, kl_b, ok_b = accept_update(pi_old, candidate_b, advantages, delta)

assert round(obj_a, 2) == 1.70
assert kl_a < delta
assert round(obj_b, 2) == 1.97
assert kl_b > delta
assert ok_a is True
assert ok_b is False

print("candidate_a:", obj_a, kl_a, ok_a)
print("candidate_b:", obj_b, kl_b, ok_b)
```

这个例子体现了 TRPO 的核心裁决逻辑：

- 候选 A：提升不少，KL 合规，接受
- 候选 B：提升更大，但 KL 超限，拒绝

再给一个更接近真实训练代码的伪代码框架：

```text
1. collect on-policy trajectories with current policy
2. fit value function and estimate advantages (often with GAE)
3. compute policy gradient g from surrogate objective
4. define Hessian-vector product function from mean KL
5. use conjugate gradient to solve Hx = g
6. scale step by sqrt(2 * delta / (g^T x))
7. run backtracking line search
8. if surrogate improves and actual KL <= delta: accept
9. else: shrink step or reject update
```

如果换成 PyTorch/JAX，实现上的重点通常是：

- `policy_loss` 只负责代理目标，不混进 KL 惩罚
- `kl_divergence` 单独算，方便约束检查
- Hessian 不显式构造，只通过自动微分实现 `Hv`
- line search 必须检查“真实 KL + 真实代理目标”，不能只看近似值

---

## 工程权衡与常见坑

TRPO 的优点和代价都很明确。优点是稳定性强、理论结构清晰；代价是实现复杂、每次更新更重、样本利用率不高。

最常见的坑如下：

| 常见坑 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| KL 方向写反 | 更新不稳或过保守 | 约束几何意义变了 | 固定使用 `D_KL(pi_old || pi_new)` |
| 不做 line search | 理论上没问题，实际一更新就崩 | 二阶近似不精确 | 必须检查真实 KL |
| `delta` 过大 | 回报暴涨后暴跌 | 允许步子太大 | 从小值开始调 |
| `delta` 过小 | 学习极慢 | 更新过于保守 | 逐步增大阈值 |
| 用 off-policy 数据 | 训练表现随机 | 代理目标假设失效 | 只用当前策略采样数据 |
| value 拟合差 | 优势高噪声，方向乱跳 | baseline 不准 | 先修 value 再调策略 |

调参上最重要的不是“把所有超参数都扫一遍”，而是先抓四个点：

| 项目 | 趋势 | 工程判断 |
|---|---|---|
| `delta` 小 | 更稳、更慢 | 适合先把训练跑稳 |
| `delta` 大 | 更快、更危险 | 只在已稳定后尝试 |
| 优势归一化 | 通常更稳 | 一般建议开启 |
| value function 拟合 | 影响极大 | 拟合差时先修这个 |

一个工程上很容易误解的点是“单调改进保证”。论文里的保证依赖一系列条件，比如局部近似、优势估计质量、KL 约束满足等。真实实现里不能把它理解成“只要用了 TRPO 就绝不会掉点”。更准确的说法是：**在这些近似和假设成立得足够好的情况下，TRPO 比普通策略梯度更容易保持不崩。**

另一个真实坑来自吞吐。TRPO 每轮都要做共轭梯度和 line search，通常比 PPO 更重。如果你的训练系统目标是极高并发、高吞吐、快速试错，TRPO 往往不是第一选择。

---

## 替代方案与适用边界

TRPO 最重要的历史价值，是把“限制策略漂移”这件事讲清楚。后来很多方法都在不同程度上继承了这个思想，最典型的是 PPO。

三者可以直接对比：

| 方法 | 核心思想 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| Vanilla PG | 直接按梯度更新 | 最简单 | 很不稳 | 教学、小实验 |
| TRPO | KL 约束的信赖域更新 | 稳定、理论清晰 | 重、复杂 | on-policy 连续控制 |
| PPO | 裁剪目标近似限制更新 | 简单、便宜、主流 | 约束不如 TRPO 直接 | 大多数工程训练 |

对初级工程师，判断标准可以很直接：

- 如果你在学习策略梯度为什么会崩，TRPO 值得认真读，因为它把“安全更新”讲得最清楚。
- 如果你要落地一个能训练、能迭代、维护成本低的工程系统，PPO 往往更现实。
- 如果你拿的是离线日志、回放池数据或强 off-policy 场景，TRPO 通常不是合适起点。

适用边界可以压缩成一组清单：

适合：

- on-policy 训练
- 连续控制任务
- 希望显式控制策略漂移
- 更重视训练稳定性而不是极限吞吐

不适合：

- 离线强化学习
- 强实时训练需求
- 超大规模高吞吐系统
- value 估计极不稳定、优势噪声很大的系统

一句话总结替代关系：**TRPO 是“约束更新”思想的代表方法，PPO 是更便宜的工程近似。**

---

## 参考资料

1. [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
2. [OpenAI Spinning Up: TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
3. [OpenAI Spinning Up Repository](https://github.com/openai/spinningup)
4. [ikostrikov/pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo)

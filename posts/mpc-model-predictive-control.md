## 核心结论

模型预测控制（MPC，Model Predictive Control，意思是“先用模型预测未来，再按预测结果选动作”）在强化学习里最重要的区别，是它**不先学一个固定策略再长期复用**，而是**每一步都重新解一个有限视野优化问题**。给定当前状态 $s_t$，它会在长度为 $H$ 的动作序列上搜索：

$$
a^*_{t:t+H-1}
=
\arg\max_{a_{t:t+H-1}}
\sum_{k=0}^{H-1} r(s_{t+k}, a_{t+k})
$$

其中未来状态由模型推进：

$$
s_{t+k+1}=f_\theta(s_{t+k}, a_{t+k})
$$

这里的 $f_\theta$ 可以是真实动力学，也可以是学习到的世界模型。世界模型的白话解释是：一个用数据拟合出来的“环境模拟器”，输入当前状态和动作，输出下一步会发生什么。

一句话理解：策略网络像“背熟一套动作映射”，MPC 像“每走一步都重新看前方几步，再临时决定下一步”。

下表是纯策略学习与 MPC 的核心差异：

| 维度 | 纯策略 RL | MPC |
|---|---|---|
| 决策方式 | 直接输出动作 $a_t=\pi(s_t)$ | 在线搜索动作序列，只执行第一步 |
| 是否依赖模型 | 通常不依赖显式模型 | 必须有真实模型或学习模型 |
| 约束处理 | 通常靠奖励塑形间接表达 | 可直接加入动作、状态、安全约束 |
| 计算位置 | 训练重，推理轻 | 推理阶段也要做优化 |
| 对分布外情况 | 容易失效 | 可借助重规划缓解 |
| 可解释性 | 较弱 | 较强，可检查候选轨迹 |

MPC 在 RL 中的价值不只是“算得更聪明”，而是把**模型预测、约束处理、在线重规划**统一到一个决策框架里。

---

## 问题定义与边界

MPC 要解决的问题是：**在模型足够可信、且在线算得过来的前提下，如何在有限未来内选出当前最优动作**。

先统一符号：

| 符号 | 含义 |
|---|---|
| $s_t \in \mathcal S$ | 时刻 $t$ 的状态，白话是“系统当前观测到的情况” |
| $a_t \in \mathcal A$ | 时刻 $t$ 的动作，白话是“当前要施加的控制输入” |
| $f_\theta$ | 动力学模型，白话是“状态如何随动作变化的规则” |
| $r(s,a)$ | 单步奖励，白话是“当前动作值不值得做” |
| $H$ | 规划视野，白话是“往前看多少步” |
| $\mathcal S,\mathcal A$ | 状态空间、动作空间 |

它不是求整条任务生命周期的全局最优，而是求一个**局部有限视野最优**。这点非常关键。因为真实工程里，模型会有误差，环境也会变，MPC 的做法是“只把短期算准，再靠下一步重算修正”。

面向新手的直观理解：开车时你不会每秒都重做一遍整趟旅行的全球规划，但你会持续看前方几十米，决定当前是轻踩刹车、保持车道，还是加一点方向盘。这就是有限视野控制。

它的边界也很明确：

| 适用前提 | 不适用场景 |
|---|---|
| 短期模型预测足够准 | 长期预测高度失真 |
| 在线优化能在时限内完成 | 毫秒级预算但优化太慢 |
| 约束需要显式处理 | 任务简单到直接策略就够 |
| 系统状态较可观测 | 严重部分可观测且无状态估计 |

所以，MPC 不是“比策略网络高级”的通用替代品，而是针对“能建模、要约束、能在线规划”的问题更合适。

---

## 核心机制与推导

MPC 的执行循环可以拆成三步：

1. 从当前状态 $s_t$ 出发，用模型 $f_\theta$ 展开未来。
2. 对很多候选动作序列计算累计回报。
3. 选出最优序列，但只执行第一步 $a_t^*$，到下一时刻再重算。

为什么只执行第一步？因为执行后会拿到新的真实观测，原来对未来的预测需要立刻修正。这叫**滚动时域优化**，白话就是“边走边改计划”。

状态展开公式是：

$$
s_{t+1}=f_\theta(s_t,a_t),\quad
s_{t+2}=f_\theta(s_{t+1},a_{t+1}),\quad \dots
$$

累计目标通常写成：

$$
J(a_{t:t+H-1}; s_t)
=
\sum_{k=0}^{H-1} r(s_{t+k}, a_{t+k})
\quad
\text{s.t.}
\quad
s_{t+k+1}=f_\theta(s_{t+k}, a_{t+k})
$$

如果还要考虑约束，可以写成：

$$
a_{t+k}\in \mathcal A_{\text{safe}},\quad
s_{t+k}\in \mathcal S_{\text{safe}}
$$

### 玩具例子

设一维系统：

$$
s_{t+1}=s_t+a_t
$$

奖励定义为：

$$
r_{t+1}=-(s_{t+1}^2+0.1a_t^2)
$$

含义很直接：希望状态尽快靠近 0，同时动作不要过大。初始状态 $s_t=2$，视野 $H=2$，候选动作只允许取 $\{-1,0,1\}$。

比较两条动作序列：

| 动作序列 | 状态变化 | 总回报 |
|---|---|---|
| $[-1,-1]$ | $2 \to 1 \to 0$ | $-(1^2+0.1)- (0^2+0.1) = -1.2$ |
| $[0,0]$ | $2 \to 2 \to 2$ | $-(4+0)- (4+0) = -8$ |

因为 $-1.2 > -8$，所以 MPC 会选择第一步 $a_t=-1$。这里最关键的不是“当前一步看起来怎样”，而是“这一步会把未来两步带到哪里”。

### CEM 如何做搜索

连续动作空间里，动作序列优化往往没法直接枚举。常用方法是 CEM（Cross-Entropy Method，交叉熵方法，白话是“先随机试很多方案，再只保留好的样本去更新采样分布”）。

CEM 的典型流程如下：

| 步骤 | 做法 |
|---|---|
| 采样 | 从当前高斯分布采样 $N$ 条动作序列 |
| 评估 | 用模型 rollout，计算每条序列的回报 |
| 排序 | 按回报从高到低排序 |
| 取 elite | 保留前 $\rho$ 比例的优秀样本 |
| 更新分布 | 用 elite 的均值和方差更新采样分布 |
| 重复 | 迭代若干轮，直到收敛或预算用完 |

它的本质不是梯度下降，而是**分布收缩**：把采样分布逐步收缩到高回报区域。这个方法对不可导奖励、复杂约束、黑盒模型都比较实用。

### 真实工程例子

移动机器人避障是 MPC 很典型的落地场景。状态包含位置、速度、朝向，动作是线速度和角速度。世界模型预测未来几步轨迹，奖励鼓励靠近目标，约束要求不能撞墙、不能超出转向角上限、不能超过加速度上限。

如果只用纯奖励而不加硬约束，优化器可能会找到“数值上回报高，但物理上不可执行”的动作序列，比如急转急停、贴边穿障碍。MPC 的优势就在于它可以把这些约束直接写进规划问题，而不是只靠奖励“暗示”安全。

---

## 代码实现

下面给一个可运行的最小 Python 版本。它不是完整 RL 训练系统，而是一个“给定模型、当前状态、奖励函数，就用 CEM 做 MPC 规划”的最小闭环。

```python
import numpy as np

def predict_next_state(state, action):
    # 一维积分系统：s_{t+1} = s_t + a_t
    return state + action

def reward_fn(next_state, action):
    # 希望状态接近 0，同时动作不要太大
    return -(next_state ** 2 + 0.1 * action ** 2)

def rollout_return(state, actions):
    total = 0.0
    s = state
    for a in actions:
        s = predict_next_state(s, a)
        total += reward_fn(s, a)
    return total

def cem_optimize(
    state,
    horizon=5,
    num_samples=256,
    num_elites=32,
    num_iters=5,
    action_low=-1.0,
    action_high=1.0,
):
    mean = np.zeros(horizon)
    std = np.ones(horizon)

    for _ in range(num_iters):
        samples = np.random.randn(num_samples, horizon) * std + mean
        samples = np.clip(samples, action_low, action_high)

        returns = np.array([rollout_return(state, seq) for seq in samples])
        elite_idx = np.argsort(returns)[-num_elites:]
        elites = samples[elite_idx]

        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-6

    best_action_sequence = mean
    return float(best_action_sequence[0]), best_action_sequence

# 基本正确性测试
a0, seq = cem_optimize(state=2.0, horizon=3)
assert isinstance(a0, float)
assert len(seq) == 3

# 玩具例子应偏向负动作，把状态往 0 拉回去
r_good = rollout_return(2.0, [-1.0, -1.0])
r_bad = rollout_return(2.0, [0.0, 0.0])
assert r_good > r_bad

# 规划出的第一步通常应为负数
assert a0 < 0.0
print("first action:", a0)
print("planned sequence:", seq)
```

这段代码对应的模块职责非常清楚：

| 接口 | 作用 |
|---|---|
| `predict_next_state(state, action)` | 用模型预测下一状态 |
| `reward_fn(next_state, action)` | 评估一步的收益 |
| `rollout_return(state, actions)` | 对一条动作序列做 rollout 并求总回报 |
| `cem_optimize(...)` | 用 CEM 搜索最优动作序列 |

从环境接入到动作输出的步骤可以写成：

| 步骤 | 输入 | 输出 |
|---|---|---|
| 读取当前观测 | 环境状态 $s_t$ | 当前状态 |
| 生成候选序列 | 采样分布 | $N$ 条动作序列 |
| 模型 rollout | $s_t$ 与候选动作 | 预测状态轨迹 |
| 计算回报 | 预测轨迹 | 每条序列的分数 |
| CEM 更新 | elite 样本 | 新的采样分布 |
| 输出控制 | 最优序列 | 只执行第一步 $a_t$ |

如果换成真正的 model-based RL，通常只需把 `predict_next_state` 换成学习得到的神经网络模型，把 `reward_fn` 换成任务定义，把状态从标量换成向量即可。

---

## 工程权衡与常见坑

MPC 在论文里常常很干净，但在工程里最难的是“模型、优化、约束、时延”四件事要同时成立。

| 常见坑 | 具体表现 | 规避策略 |
|---|---|---|
| 模型偏差 | 短期还准，滚几步后全错 | 用 ensemble、不确定性惩罚、缩短 $H$、高频重规划 |
| 采样太贵 | CEM 在高维动作上很慢 | warm start、GPU 并行、减少变量维度、动作分块 |
| 约束缺失 | 结果高回报但不可执行 | 把安全约束直接写入规划，必要时加 safety shield |
| 视野选择不当 | 过短短视，过长误差爆炸 | 按模型精度和实时预算联合选择 |
| 奖励设计失真 | 学会投机行为而非真实目标 | 奖励与约束分开：目标进奖励，底线进约束 |
| 状态估计不稳 | 观测噪声导致规划抖动 | 配合滤波器、belief state、动作平滑 |

这里最容易被初学者忽略的是：**奖励不是约束的替代品**。例如自动驾驶里，如果只把“碰撞惩罚”写成一个很大的负奖励，优化器仍可能在某些边界条件下选择高风险动作，因为它看到的是期望回报，不是硬安全保证。真正稳妥的做法是把碰撞距离、最大曲率、最大加速度写成显式约束。

另一个常见误区是盲目拉长视野。视野更长不一定更好，因为模型误差会沿 rollout 累积。若每步误差记为 $\epsilon_k$，则长视野预测误差常近似随 $\sum_k \epsilon_k$ 甚至更快放大。对学习模型来说，过长视野经常意味着“看得更远，但全是错的”。

---

## 替代方案与适用边界

MPC 很强，但并不总是第一选择。它更适合“需要约束、可建模、可在线算”的任务；不一定适合超高维感知输入、超长时序信用分配、强随机非平稳环境。

下面给一个常见方案对比：

| 方法 | 核心思想 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 纯策略 RL | 直接学 $\pi(a|s)$ | 推理快 | 约束难处理，分布外脆弱 | 大规模交互、对实时要求高 |
| MPC | 每步在线规划 | 可处理约束、可解释 | 推理成本高、依赖模型 | 机器人、控制、自动驾驶 |
| 价值函数引导规划 | 用 $V/Q$ 辅助搜索 | 比纯搜索更高效 | 值函数误差会误导规划 | 离散动作或混合规划 |
| 安全过滤器 / shield | 在动作输出后做安全修正 | 易与现有策略叠加 | 只能兜底，不一定最优 | 已有策略但要补安全层 |

可以用一个很直白的判断清单来决定要不要上 MPC：

| 判断问题 | 若回答是“是” |
|---|---|
| 你能得到短期可用的环境模型吗？ | 更适合 MPC |
| 你必须处理速度、碰撞、能耗等硬约束吗？ | 更适合 MPC |
| 你的推理时允许做几十到几百次模型 rollout 吗？ | 更适合 MPC |
| 你的任务主要难点是长期抽象决策而非短期控制吗？ | 未必适合 MPC |
| 你的输入是纯图像且模型很难学准吗？ | 更偏向策略或混合方法 |

同样是机器人控制，纯 policy network 更像“把观测直接映射到动作”，而 MPC 更像“每次都重新做一个局部控制问题”。前者便宜，后者稳健且更容易把工程规则写进去。

---

## 参考资料

1. [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)](https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) - 概率动力学模型结合 MPC 的代表性工作，适合理解现代 model-based RL 主线。
2. [The Cross-Entropy Method for Combinatorial and Continuous Optimization](https://econpapers.repec.org/article/sprmetcap/v_3a1_3ay_3a1999_3ai_3a2_3ad_3a10.1023_5fa_3a1010091220143.htm) - CEM 经典论文，适合理解“采样-筛选-更新分布”的原理。
3. [The Cross-Entropy Method for Optimization](https://www.sciencedirect.com/science/article/pii/B9780444538598000035) - 对 CEM 更系统的综述，适合补全理论背景。
4. [safe-mbrl: Constrained Model-based RL with Robust Cross Entropy Method](https://github.com/liuzuxin/safe-mbrl) - 一个把安全约束、世界模型和 CEM 结合起来的开源实现，适合对应工程落地。
5. [Model Predictive Control: Theory and Design](https://www.amazon.com/Model-Predictive-Control-Theory-Design/dp/0975937705) - 传统 MPC 理论入口，适合补控制视角中的稳定性与约束处理。

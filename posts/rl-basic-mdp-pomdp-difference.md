## 核心结论

MDP 和 POMDP 的根本区别，不在于“动作怎么选”，而在于“智能体到底看到了什么”。

MDP，Markov Decision Process，中文常叫“马尔可夫决策过程”，白话就是：系统的真实状态 $s$ 是直接可见的，当前状态已经包含了做决策所需的全部信息。于是策略可以直接写成 $\pi(a|s)$，转移满足 $P(s'|s,a)$。

POMDP，Partially Observable Markov Decision Process，中文常叫“部分可观测马尔可夫决策过程”，白话就是：真实状态还在，但智能体看不全，只能拿到带噪声、带遮挡、带信息缺失的观测 $o$。这时策略不能只看当前观测，而要维护一个“信念状态” $b(s)$。信念状态就是“系统当前处于各个真实状态的概率分布”。

最重要的结论有三点：

| 问题 | MDP | POMDP |
|---|---|---|
| 智能体看到什么 | 真实状态 $s$ | 观测 $o$ |
| 策略依赖什么 | $\pi(a|s)$ | $\pi(a|b)$ 或 $\pi(a|o_{1:t},a_{1:t-1})$ |
| 计算代价 | 通常较低 | 通常显著更高 |

一个玩具例子是完全可见小迷宫：你知道自己在第几格，直接选往左还是往右，这就是 MDP。一个真实工程例子是带模糊雷达的移动机器人：它只能看到“前方像是有墙”“左边可能有障碍”，却不知道自己百分之百在哪，这就是 POMDP。后者必须先估计“我大概率在哪”，再决定动作。

---

## 问题定义与边界

MDP 的标准定义是五元组：

$$
\langle S, A, T, R, \gamma \rangle
$$

其中：

- $S$ 是状态集合，白话就是系统可能处于哪些情况
- $A$ 是动作集合，白话就是你能做什么
- $T(s'|s,a)$ 是状态转移概率，白话就是做了动作后，系统怎么变化
- $R(s,a)$ 或 $R(s,a,s')$ 是奖励函数，白话就是这一步值不值
- $\gamma$ 是折扣因子，白话就是未来奖励在当前值多少钱

POMDP 在此基础上再加两项：

$$
\langle S, A, T, R, O, \Omega, \gamma \rangle
$$

这里：

- $O$ 是观测集合，白话就是传感器会输出哪些信号
- $\Omega(o|s',a)$ 是观测模型，白话就是系统到了状态 $s'$ 后，你看到观测 $o$ 的概率

边界要划清楚：POMDP 不是“状态没了”，而是“状态还在，但你看不见”。因此它不是把状态换成观测，而是在真实状态之上再叠一层不确定观测。

看一个最小例子。设：

- $S=\{s_1, s_2\}$
- $A=\{\text{left}, \text{right}\}$

在 MDP 里，如果你当前明确知道自己在 $s_1$，那就直接比较：
- 执行 `left` 的长期回报
- 执行 `right` 的长期回报

在 POMDP 里，你看到的可能只是一个观测，比如“仪表盘亮黄灯”。黄灯可能对应 $s_1$，也可能对应 $s_2$。于是同样一个观测，不能唯一确定状态，决策会出现不确定性。

这就是两者的边界：

- 如果“当前可见信息”已经足以唯一确定决策所需状态，用 MDP
- 如果“当前可见信息”只是带噪声线索，需要结合历史推断隐藏状态，用 POMDP

---

## 核心机制与推导

POMDP 的核心不是“多了一个观测”，而是“历史必须被压缩成信念状态”。

设历史为：

$$
h_t=(o_1,a_1,o_2,a_2,\dots,o_t)
$$

直接用全部历史做决策太贵，所以通常把历史压缩成信念状态：

$$
b_t(s)=P(s_t=s \mid h_t)
$$

这句话的白话是：在看完到当前为止的所有动作和观测后，系统处于状态 $s$ 的概率是多少。

### 从历史到信念

给定旧信念 $b(s)$、动作 $a$ 和新观测 $o$，新信念为：

$$
b'(s')=\tau(b,a,o)=\frac{\Omega(o|s',a)\sum_s T(s'|s,a)b(s)}{\eta(o,b,a)}
$$

其中 $\eta(o,b,a)$ 是归一化常数，作用是把所有概率加起来调成 1。

这个式子分成两步理解最容易：

1. 预测：先根据动作做状态转移  
   $$
   \hat b(s')=\sum_s T(s'|s,a)b(s)
   $$
2. 校正：再根据观测修正  
   $$
   b'(s') \propto \Omega(o|s',a)\hat b(s')
   $$

这就是贝叶斯滤波。贝叶斯滤波，白话就是：先按动力学猜下一步，再用传感器读数纠偏。

### 玩具例子

设初始信念：

$$
b=[0.8,0.2]
$$

表示系统有 80% 在 $s_1$，20% 在 $s_2$。

假设动作执行后状态不变，转移矩阵近似是恒等映射。又设观测为 $z$，观测模型为：

$$
\Omega(z|s_1)=0.9,\quad \Omega(z|s_2)=0.4
$$

那么预测后仍是：

$$
\hat b=[0.8,0.2]
$$

校正时：

$$
b'(s_1)\propto 0.9 \times 0.8 = 0.72
$$

$$
b'(s_2)\propto 0.4 \times 0.2 = 0.08
$$

归一化常数：

$$
\eta = 0.72 + 0.08 = 0.80
$$

所以：

$$
b'=\left[\frac{0.72}{0.80}, \frac{0.08}{0.80}\right]=[0.9,0.1]
$$

同样的动作，因为收到了观测 $z$，信念从 $[0.8,0.2]$ 更新到了 $[0.9,0.1]$。这就是 POMDP 与 MDP 的根本差异：动作之前要先“估状态”，而不是默认状态已知。

### 从状态价值到信念价值

MDP 的最优价值函数是：

$$
V^*(s)=\max_a \left[ R(s,a)+\gamma \sum_{s'} T(s'|s,a)V^*(s') \right]
$$

POMDP 则变成 belief MDP，也就是在信念空间上做动态规划：

$$
V^*(b)=\max_a \left[ R(b,a)+\gamma\sum_o P(o|b,a)V^*(\tau(b,a,o)) \right]
$$

这里的 $R(b,a)$ 是对真实状态奖励按信念求期望。问题也随之变难：原来状态空间大小是 $|S|$，现在信念状态落在概率单纯形上，是连续空间。换句话说，POMDP 可以更真实地建模遮挡、噪声和信息缺失，但计算代价显著上升。

---

## 代码实现

下面给一个可运行的最小实现。它不是完整求解器，但足够演示 POMDP 的关键步骤：预测、观测校正、归一化。

```python
from math import isclose

def belief_update(belief, transition, observation_prob, action, obs):
    """
    belief: dict[state] -> prob
    transition: dict[action][s][s_next] -> prob
    observation_prob: dict[action][s_next][obs] -> prob
    """
    predicted = {}
    for s, p_s in belief.items():
        for s_next, p_trans in transition[action][s].items():
            predicted[s_next] = predicted.get(s_next, 0.0) + p_s * p_trans

    unnormalized = {}
    for s_next, p_pred in predicted.items():
        p_obs = observation_prob[action][s_next][obs]
        unnormalized[s_next] = p_pred * p_obs

    z = sum(unnormalized.values())
    assert z > 0.0, "归一化常数为 0，说明观测模型或输入有问题"

    updated = {s: p / z for s, p in unnormalized.items()}
    assert isclose(sum(updated.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    return updated

belief = {"s1": 0.8, "s2": 0.2}
transition = {
    "stay": {
        "s1": {"s1": 1.0, "s2": 0.0},
        "s2": {"s1": 0.0, "s2": 1.0},
    }
}
observation_prob = {
    "stay": {
        "s1": {"z": 0.9},
        "s2": {"z": 0.4},
    }
}

updated = belief_update(belief, transition, observation_prob, "stay", "z")
assert round(updated["s1"], 4) == 0.9
assert round(updated["s2"], 4) == 0.1
print(updated)
```

如果想让新手先建立直觉，可以把“按 belief 选动作”理解成：先算出当前最可能在哪，再根据整个概率分布而不是单个观测做动作选择。伪代码如下：

```python
def expected_reward(belief, reward, action):
    return sum(belief[s] * reward[s][action] for s in belief)

def greedy_policy_on_belief(belief, reward, actions):
    scores = {a: expected_reward(belief, reward, a) for a in actions}
    best_action = max(scores, key=scores.get)
    return best_action, scores

reward = {
    "s1": {"left": 2.0, "right": 0.0},
    "s2": {"left": -1.0, "right": 1.0},
}

action, scores = greedy_policy_on_belief({"s1": 0.9, "s2": 0.1}, reward, ["left", "right"])
assert action == "left"
assert scores["left"] > scores["right"]
```

这段代码只做一步贪心，不是完整的 value iteration，但结构已经说明问题：在 POMDP 中，策略查询的输入应该是 belief，而不是单次观测。

工程里还要注意数值稳定性。如果状态多、序列长，概率连乘会变得很小，此时常改用对数空间或定期重归一化。

---

## 工程权衡与常见坑

真实系统里，POMDP 的难点通常不在公式，而在代价和近似。

一个真实工程例子是自主导航机器人。机器人在仓库里移动时，激光雷达会被货架遮挡，里程计会累积误差，摄像头会受光照影响。如果你硬把当前位置当成“完全已知状态”，相当于把问题错误建成 MDP，那么策略会过度自信：以为自己在走廊中心，实际上已经偏到边缘。结果就是转弯过早、避障失败、路径抖动。

更合理的做法是维护参数化 belief，比如只保留位置均值、协方差或少量足够统计量。足够统计量，白话就是：不保存完整历史，只保存对未来决策够用的压缩信息。这样虽然牺牲了部分精确性，但能把问题从不可算变成可算。

常见坑可以直接列出来：

| 常见坑 | 具体表现 | 后果 | 规避方法 |
|---|---|---|---|
| 忽略观测模型 $\Omega(o|s',a)$ | 只建转移，不建传感器噪声 | belief 无法正确更新 | 先标定传感器，再写观测概率 |
| 忘记归一化 | 更新后概率和不为 1 | 后续价值计算全错 | 每次 update 后强制检查和为 1 |
| 把观测当状态 | 直接学 $\pi(a|o)$ | 在部分可观测下常丢最优性 | 至少引入历史窗口或 belief |
| belief 空间连续 | 不能直接套普通离散 DP | 计算爆炸 | 用粒子滤波、网格近似、参数化 belief |
| 数值下溢 | 长序列概率越来越接近 0 | 更新失真或 NaN | 用 log-space 或稳定归一化 |
| 模型太精细 | 状态、观测维度暴涨 | 无法实时规划 | 先做抽象，再做在线近似 |

一个判断标准很实用：如果系统错误主要来自“未来随机”，MDP 往往够用；如果错误主要来自“当前看不清”，就应该认真考虑 POMDP。

---

## 替代方案与适用边界

不是所有“不确定”问题都值得上 POMDP。POMDP 更强，但也更贵。

可以用下面这张表快速判断：

| 场景特征 | 更适合 MDP | 更适合 POMDP |
|---|---|---|
| 状态是否直接可见 | 是 | 否 |
| 观测噪声 | 低 | 高 |
| 是否存在遮挡/缺失信息 | 少 | 多 |
| 实时性要求 | 极高 | 可以接受近似推断 |
| 规划复杂度预算 | 紧 | 相对宽松 |

简化成 checklist 就是：

1. 当前观测能否稳定唯一对应真实状态？
2. 如果不能，历史信息是否会显著提高判断准确率？
3. 如果会，维护 belief 的收益是否大于计算成本？

如果第 1 个问题回答“能”，通常优先用 MDP。  
如果第 1 个回答“不能”，第 2 个回答“会”，那已经进入 POMDP 的典型适用区。  
如果第 3 个问题回答“成本太高”，就不要硬上完整 POMDP，而要考虑替代方案。

常见替代方案有三类：

- 历史窗口策略：把最近几步观测拼起来近似隐藏状态，适合轻量系统
- 状态估计器 + MDP：先用卡尔曼滤波、粒子滤波或其他估计器得到状态分布，再在估计状态上规划
- 参数化或近似 POMDP：只保留 belief 的低维表示，用点值迭代、采样规划等方法近似求解

所以准确说，MDP 和 POMDP 不是“谁更高级”，而是“你对可观测性的假设是否成立”。当观测可信、状态可复现、实时性要求高时，MDP 是更稳妥的工程选择；当环境存在遮挡、噪声、误检、漏检，且这些不确定性会直接影响动作质量时，POMDP 才值得它的额外复杂度。

---

## 参考资料

- [Yale CS470: MDP and POMDP 概念整理](https://zoo.cs.yale.edu/classes/cs470/materials/hws/aima/mdp.html?utm_source=openai)  
  重点在 MDP/POMDP 的基本定义与状态可观测性差异，适合建立术语框架。

- [POMDP.org: Who needs POMDPs?](https://www.pomdp.org/talks/who-needs-pomdps/index.html?utm_source=openai)  
  重点在直观解释，适合理解“完全可见迷宫”和“模糊传感器机器人”这种区别。

- [POMDP.org Tutorial: Solving POMDPs](https://www.pomdp.org/tutorial/pomdp-solving.html?utm_source=openai)  
  重点在 belief update、belief MDP 和 value iteration，是本文公式部分的直接背景。

- [Emergent Mind: POMDP Framework](https://www.emergentmind.com/topics/pomdp-framework?utm_source=openai)  
  重点在把 belief 价值递归写清楚，适合从概念过渡到求解视角。

- [Parametric POMDPs for Robot Navigation](https://www.sciencedirect.com/science/article/abs/pii/S0921889006000960?utm_source=openai)  
  重点在机器人导航中的工程化视角，说明为什么现实噪声会逼着系统从 MDP 走向 POMDP。

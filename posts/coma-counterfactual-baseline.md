## 核心结论

COMA（Counterfactual Multi-Agent Policy Gradients，反事实多智能体策略梯度）是一种多智能体 actor-critic 方法。它解决的核心问题不是“团队最后得分高不高”，而是“在这一步里，某个 agent 的动作到底对团队结果贡献了多少”。

它的关键设计是反事实基线。反事实的白话意思是：保持其他条件不变，只改一个变量，观察结果会不会变。放到多智能体里，就是固定其他 agent 的动作 $a_{-i}$，只替换当前 agent $i$ 的动作 $a_i$，再看联合动作价值 $Q(s,a)$ 如何变化。这样得到的优势函数不是模糊的团队平均评价，而是更接近“这个 agent 这一步的边际贡献”。

核心公式有两个：

$$
b_i(s, a_{-i}) = \sum_{a_i'} \pi_i(a_i' \mid o_i)\, Q\bigl(s, (a_i', a_{-i})\bigr)
$$

$$
A_i(s, a) = Q(s, a) - b_i(s, a_{-i})
$$

其中，基线 $b_i$ 是“如果队友动作不变，agent $i$ 按自己当前策略平均会做到什么水平”；优势 $A_i$ 是“这次实际动作相对这个平均水平是更好还是更差”。

为什么需要 COMA，可以先看一个新手版例子。团队比赛赢了，普通共享奖励只会告诉你“球队赢了”。但训练算法真正需要知道的是：3 号球员这次传球是否关键，还是队友本来就能把球打进。COMA 的做法是先固定其他球员动作，再假设 3 号球员换一种动作，比较结果差异。这样才能把“团队赢了”拆成“谁在这一步做对了什么”。

一句话总结：COMA 的本质是用中心化 critic 评估联合动作，再用反事实基线把全局回报拆成单个 agent 可学习的信用信号。

---

## 问题定义与边界

多智能体强化学习里，最难的问题之一叫信用分配。信用分配的白话解释是：结果已经出来了，但不知道应该把功劳或责任分给谁。单智能体没有这个问题，因为动作和结果之间是一对一链路；多智能体里，多个动作共同决定一个共享奖励，链路会混在一起。

典型场景有三个特征：

| 场景特征 | 为什么会困难 | 为什么 COMA 合适 |
| --- | --- | --- |
| 共享奖励 | 所有 agent 拿到同一个回报，单看奖励无法区分个体贡献 | 反事实基线能给每个 agent 单独归因 |
| 部分可观测 | 每个 agent 只能看局部观测，无法独立重建全局 | 训练时用中心化 critic 补全全局信息 |
| 稀疏反馈 | 结果常常在很多步之后才出现 | 用 $Q(s,a)$ 建模中间动作对最终回报的影响 |

在 StarCraft II 微操任务里，这个问题尤其明显。每个单位都是一个 agent，最后系统可能只给“这波团战赢了”或“输了”的信号。但具体策略动作包括谁先手、谁后撤、谁集火、谁补伤害。普通 policy gradient 只能看到全局回报，难以判断是哪个单位的动作更关键。COMA 就是在这种“团队胜负明确，但个体功劳模糊”的任务里发挥作用。

COMA 通常适用于以下边界：

| 约束 | 说明 |
| --- | --- |
| 中心化训练 | 训练时 critic 能访问全局状态、联合动作、全队轨迹 |
| 分散执行 | 执行时每个 actor 只能访问自己的局部观测 $o_i$ |
| 离散动作 | 原版公式通过枚举动作求和，最适合离散动作空间 |
| 共享目标 | 多个 agent 共同优化团队回报，而不是彼此对抗 |

这里要强调“训练时集中，执行时分散”。这句话的意思是：学习阶段可以开上帝视角，但部署阶段不能依赖上帝视角。COMA 的 critic 是中心化的，因为它需要看全局状态 $s$ 和联合动作 $a$ 才能做信用分配；但 actor 必须是局部的，因为真实执行时每个 agent 只能根据自己的观测行动。

它也有明确的不适用边界。

第一，不适合直接照搬到连续动作。因为基线公式需要对 $a_i'$ 枚举求和，而连续动作空间没有有限个动作可直接遍历。第二，不适合纯局部可决策任务。如果每个 agent 的奖励本来就是独立且清晰的，那么 COMA 的额外复杂度没有必要。第三，不适合联合动作维度极端膨胀的任务，因为 critic 建模 $Q(s,a)$ 本身就会变难。

---

## 核心机制与推导

COMA 的结构可以拆成两部分。

一部分是 actor。actor 的白话解释是：负责做动作决策的策略网络。每个 agent 有自己的策略 $\pi_i(a_i \mid o_i)$，输入是局部观测 $o_i$，输出是动作分布。

另一部分是 critic。critic 的白话解释是：负责打分的价值网络。它评估联合动作在全局状态下的价值，即：

$$
Q(s, a), \quad a=(a_i, a_{-i})
$$

这里 $a=(a_i,a_{-i})$ 表示联合动作由两部分组成：当前 agent $i$ 的动作 $a_i$，以及其他所有 agent 的动作 $a_{-i}$。

普通 actor-critic 会用一个基线减少梯度方差，但 COMA 不是用状态值函数 $V(s)$ 做统一基线，而是构造只针对 agent $i$ 的反事实基线：

$$
b_i(s, a_{-i}) = \sum_{a_i'} \pi_i(a_i' \mid o_i)\, Q(s, (a_i', a_{-i}))
$$

这一步的关键不在“求平均”，而在“固定谁”。固定 $a_{-i}$，意味着队友当前动作保持不变；只对 $a_i$ 求策略期望，意味着只考察 agent $i$ 自己的动作变化。于是得到的优势函数：

$$
A_i(s, a) = Q(s, a) - b_i(s, a_{-i})
$$

这个 $A_i$ 表示：在队友动作已知的前提下，这次实际动作比“自己按当前策略平均会做出的动作”好多少或差多少。

接着代入策略梯度：

$$
\nabla_{\theta_i} J_i = \mathbb{E}\bigl[\nabla_{\theta_i}\log \pi_i(a_i \mid o_i)\cdot A_i(s,a)\bigr]
$$

这条式子的直观含义是：

1. 如果某个动作的优势为正，就增加它的概率。
2. 如果某个动作的优势为负，就降低它的概率。
3. 这个优势不是团队整体情绪分，而是已经扣除了“队友动作带来的背景分数”。

看一个玩具例子。假设 agent $i$ 只有两个动作 $0$ 和 $1$，队友动作固定为 $a_{-i}$。critic 给出：

- $Q(s,(0,a_{-i}))=10$
- $Q(s,(1,a_{-i}))=6$

当前策略是：

- $\pi_i(0|o_i)=0.7$
- $\pi_i(1|o_i)=0.3$

那么反事实基线为：

$$
b_i = 0.7\times 10 + 0.3\times 6 = 8.8
$$

如果这一步实际选了动作 $1$，那么：

$$
A_i = 6 - 8.8 = -2.8
$$

如果实际选了动作 $0$，那么：

$$
A_i = 10 - 8.8 = 1.2
$$

这里的结论不是“动作 1 绝对错误”，而是“在当前队友动作固定时，它低于 agent 自己按当前策略平均会做到的水平”。这就是 COMA 的归因方式。它不是拿动作和全局平均比，而是拿动作和“当前上下文里的自我反事实平均”比。

再看一个真实工程例子。假设是 StarCraft 微操中的三单位协作：坦克顶前排、远程输出、残血单位拉扯。共享奖励可能来自整场战斗结果。某一步里远程单位没有集火关键目标，导致后续整体输出不足。普通共享奖励只能在战斗结束后告诉系统“输了”；COMA 会在训练时固定另外两个单位当时的动作，只替换远程单位这一步的动作，比较 $Q$ 值变化。如果替换成“集火”后 $Q$ 明显更高，那么这个单位当前动作就会拿到负优势，策略会被纠正。

如果把这个流程画成结构图，可以理解为三层：

| 模块 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| Actor $i$ | 局部观测 $o_i$ | $\pi_i(a_i\mid o_i)$ | 决策 |
| Central Critic | 全局状态 $s$、联合动作 $a$ | $Q(s,a)$ | 评估团队动作价值 |
| Counterfactual Baseline | $Q(s,(a_i',a_{-i}))$、$\pi_i$ | $b_i(s,a_{-i})$ | 只计算 agent $i$ 的边际基线 |

所以 COMA 的核心不是“更复杂的 advantage”，而是“把 advantage 的比较对象改成了局部反事实世界”。

---

## 代码实现

实现 COMA 时，通常要分清三件事：收集联合轨迹、训练中心化 critic、计算反事实基线更新 actor。

联合轨迹的白话解释是：同一个时间步里，把所有 agent 的观测、动作、奖励、可行动作 mask 一起存下来。因为后续 critic 和基线都依赖“同一步里其他 agent 做了什么”，只存单个 agent 的局部日志不够。

先看输入输出结构：

| 项目 | 作用 |
| --- | --- |
| 全局状态 `state` | 训练 critic，用于恢复全局上下文 |
| 局部观测 `obs[i]` | 训练或执行 actor，仅 agent `i` 可见 |
| 联合动作 `joint_action` | 训练 critic，构造真实 $Q(s,a)$ |
| `action_mask[i]` | 标记哪些动作合法，避免非法动作进入基线 |
| 输出 `Q` | 当前联合动作价值 |
| 输出 `b_i` | agent `i` 的反事实基线 |
| 输出 `A_i` | actor 更新用的优势 |

为什么需要 action mask。因为许多环境里并不是每一步所有动作都可选，比如单位死亡、技能冷却、目标超出射程。如果把非法动作也纳入

$$
\sum_{a_i'} \pi_i(a_i'|o_i)\,Q(s,(a_i',a_{-i}))
$$

基线就会被错误抬高或拉低，优势会失真。

下面给一个可运行的 Python 玩具实现。它不是完整训练框架，但能准确演示 COMA 的反事实基线计算。

```python
from math import log

def critic_forward(q_table, state, joint_action):
    return q_table[(state, tuple(joint_action))]

def counterfactual_baseline(q_values_for_agent_i, policy_probs, action_mask):
    masked_probs = [
        p if m else 0.0
        for p, m in zip(policy_probs, action_mask)
    ]
    total = sum(masked_probs)
    assert total > 0.0, "at least one action must be legal"

    # 重新归一化，保证只在合法动作上求期望
    normalized_probs = [p / total for p in masked_probs]
    baseline = sum(p * q for p, q in zip(normalized_probs, q_values_for_agent_i))
    return baseline, normalized_probs

def coma_advantage(q_table, state, joint_action, agent_i, policy_probs, action_mask):
    q_sa = critic_forward(q_table, state, joint_action)

    q_values_for_agent_i = []
    for a_i in range(len(policy_probs)):
        counterfactual_action = list(joint_action)
        counterfactual_action[agent_i] = a_i
        q_values_for_agent_i.append(
            critic_forward(q_table, state, counterfactual_action)
        )

    baseline, normalized_probs = counterfactual_baseline(
        q_values_for_agent_i, policy_probs, action_mask
    )
    advantage = q_sa - baseline
    return q_sa, baseline, advantage, normalized_probs

# 玩具例子：2 个 agent，每个 agent 2 个动作
q_table = {
    ("s0", (0, 0)): 10.0,
    ("s0", (1, 0)): 6.0,
    ("s0", (0, 1)): 7.0,
    ("s0", (1, 1)): 4.0,
}

policy_probs = [0.7, 0.3]
action_mask = [1, 1]
joint_action = [1, 0]  # agent0 选 1，agent1 选 0

q_sa, baseline, advantage, normalized_probs = coma_advantage(
    q_table=q_table,
    state="s0",
    joint_action=joint_action,
    agent_i=0,
    policy_probs=policy_probs,
    action_mask=action_mask,
)

assert abs(q_sa - 6.0) < 1e-9
assert abs(baseline - 8.8) < 1e-9
assert abs(advantage - (-2.8)) < 1e-9
assert abs(sum(normalized_probs) - 1.0) < 1e-9

# 一个简化的 actor loss 形式
chosen_action = joint_action[0]
log_prob = log(policy_probs[chosen_action])
actor_loss = -(log_prob * advantage)

assert actor_loss < 0  # 负优势时，优化会倾向于降低该动作概率
print("COMA toy example passed.")
```

这个例子体现了实现上的几个关键点。

第一，critic 和 actor 的输入不同。critic 用 `state + joint_action`，因为它的任务是估计联合动作价值；actor 只用 `obs[i]`，因为执行时拿不到全局状态。

第二，基线的实现方式通常是“遍历当前 agent 的所有候选动作”。这是离散动作下最直接的方法：先用 critic 算出每个替代动作对应的 $Q$，再按当前策略概率加权。

第三，训练时必须保留联合动作轨迹。因为你在时间步 $t$ 更新 agent $i$ 时，需要知道同一步其他 agent 的真实动作 $a_{-i}$。如果轨迹里只保留局部动作，就无法构造反事实世界。

真实工程里，常见伪代码是这样的：

```python
# 1. 从轨迹取出 state, obs_i, joint_action, available_actions_i
q_sa = critic_forward(state, joint_action)

# 2. 枚举 agent i 的候选动作，其他 agent 动作固定
q_values = []
for candidate_a_i in valid_actions_i:
    joint_action_cf = replace(joint_action, i, candidate_a_i)
    q_values.append(critic_forward(state, joint_action_cf))

# 3. 用当前策略概率求反事实基线
baseline = sum(policy_prob[a_i] * q_values[a_i] for a_i in valid_actions_i)

# 4. 算优势
advantage = q_sa - baseline

# 5. 更新 actor
actor_loss = -(log_prob_taken_action * advantage).mean()
```

---

## 工程权衡与常见坑

COMA 的收益来自更清晰的信用分配，但代价也很明确：实现复杂、计算重、训练稳定性更敏感。尤其在动作空间较大时，基线的逐动作枚举会带来明显计算成本。

下面是常见坑：

| 常见坑 | 错误做法 | 正确做法 |
| --- | --- | --- |
| 基线写成全局均值 | 对所有 agent、所有动作一起平均 | 固定 $a_{-i}$，只对当前 agent 的 $a_i$ 求期望 |
| critic 退化成局部输入 | 只喂 `obs[i]` 给 critic | critic 需要中心化输入，至少能看到全局状态和联合动作 |
| 忽略非法动作 | 所有动作都进 softmax 和求和 | 用 action mask 过滤非法动作并重归一化 |
| 混淆 `Q` 和 `A` | 直接拿 `Q` 更新 actor | actor 更新应使用 $A_i = Q - b_i$ |
| 连续动作直接照搬 | 对连续动作硬枚举 | 改用采样近似、解析近似，或换方法 |
| 数据时间错位 | 当前策略配旧轨迹、旧 critic | 保证采样策略、critic 估值和轨迹时间尽量对齐 |

一个典型错误是把基线写成“所有动作的全局平均 Q”。这样做的问题是，它把“队友当前动作固定”这个条件删掉了。删掉后，优势就不再表示 agent $i$ 的边际贡献，而变成一个混杂着队友策略变化的平均值，信用分配会重新模糊。

另一个常见问题是 critic 质量不足。COMA 高度依赖 critic，因为反事实基线完全建立在 $Q(s,a)$ 的准确性之上。如果 critic 学偏了，基线和优势都会偏。结果往往不是“稍微差一点”，而是 actor 会被系统性误导。

工程排查时，可以按这个 checklist 走：

| 检查项 | 说明 |
| --- | --- |
| 时间步对齐 | `state`、`obs`、`joint_action`、`reward` 是否来自同一时间步 |
| 轨迹缓存完整 | 是否保留了所有 agent 的动作和合法动作集合 |
| 动作合法性检查 | mask 是否同时用于策略和基线计算 |
| 训练/执行一致性 | actor 执行时是否只依赖局部观测 |
| on-policy 程度 | 轨迹是否过旧，策略是否已经漂移太多 |

如果训练不稳定，常见排查顺序可以是：先查 mask，再查轨迹对齐，再查 critic 输出范围，再查 advantage 分布，最后再看学习率和探索参数。原因很简单，COMA 大多数“玄学不收敛”问题，最后都能落到数据错位或基线错误上。

---

## 替代方案与适用边界

COMA 不是多智能体问题的通用最优解。它更适合强协作、共享奖励、需要显式信用分配、且动作是离散的任务。如果任务结构不同，其他方法可能更简单，也更稳。

先看对比：

| 方法 | 是否中心化 critic | 是否显式信用分配 | 动作类型 | 计算成本 | 典型适用场景 |
| --- | --- | --- | --- | --- | --- |
| COMA | 是 | 是，靠反事实基线 | 离散优先 | 较高 | 强协作、共享奖励、归因困难 |
| MADDPG | 是 | 间接，不是反事实归因 | 连续更常见 | 中到高 | 混合协作/竞争、连续控制 |
| QMIX | 不是 actor-critic 式中心 critic，使用混合网络 | 不显式做单体反事实 | 离散 | 中等 | 协作任务，重视稳定值分解 |
| VDN | 否，值函数加和分解 | 不显式 | 离散 | 较低 | 结构较简单、快速基线方案 |

可以把它们理解成两类思路。

一类是 COMA 这种“直接归因”。它明确问：只改 agent $i$ 的动作，团队价值会变多少。优点是解释清楚，信用分配有针对性；缺点是 critic 难学、基线计算重。

另一类是 QMIX、VDN 这种“值分解”。它们不一定逐个体做反事实比较，而是通过网络结构把团队价值拆成若干局部值再组合。优点通常是更稳定、更容易训练；缺点是归因解释没有 COMA 那么直接。

再看两个应用判断。

玩具例子：两个机器人协同搬箱子，一个负责推，一个负责纠偏。如果箱子掉落，你希望知道是推力方向错误，还是纠偏时机错误，这种问题就适合 COMA，因为你确实关心单个体的边际责任。

真实工程例子：如果你做的是多单位协同战斗、无人机编队、仓储机器人协同搬运，且训练时能收集全局状态，那么 COMA 值得考虑。相反，如果任务更像“所有人各做各的局部子任务”，或者团队回报本身就能自然拆开，那么 COMA 往往不划算。

可以用一句选择指南收尾：

- 选 COMA：当你需要显式解决共享奖励下的信用分配，并且能接受中心化 critic 与离散动作枚举的成本。
- 不选 COMA：当任务本身不缺归因、动作连续且高维、或更看重训练稳定性与实现简单度时。

---

## 参考资料

阅读顺序建议：

| 顺序 | 资料类型 | 目的 |
| --- | --- | --- |
| 1 | 原始论文 | 先理解反事实基线的定义和公式出处 |
| 2 | 论文转写版 | 补方法段落和公式上下文 |
| 3 | PyMARL 实现 | 看工程入口和训练流程 |
| 4 | SMAC 环境 | 理解 COMA 常用实验设定 |
| 5 | 工程文档总结 | 对照实现细节和常见配置 |

公式出处说明：本文中的 $b_i(s,a_{-i})$、$A_i(s,a)$ 与策略梯度形式，核心来源于 COMA 原论文的方法定义。

1. [Counterfactual Multi-Agent Policy Gradients](https://www.cs.ox.ac.uk/publications/publication11394-abstract.html)
2. [Counterfactual Multi-Agent Policy Gradients（论文转写版）](https://awesomepapers.org/sandbox/papers/counterfactual-multi-agent-policy-gradients)
3. [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)
4. [oxwhirl/smac](https://github.com/oxwhirl/smac)
5. [DI-engine COMA 文档](https://di-engine-docs.readthedocs.io/en/latest/12_policies/coma.html)

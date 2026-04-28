## 核心结论

多智能体强化学习（Multi-Agent Reinforcement Learning, MARL，白话是“多个会自己学策略的决策体一起做长期决策”）的真正难点，不是 agent 数量变多，而是每个 agent 的回报会被其他 agent 的策略同时改变。单智能体强化学习里，环境通常近似固定；多智能体里，其他 agent 也是会更新的变量，所以训练目标会漂移。

对合作任务来说，最常见的两个极端方案都有限制。独立学习（Independent Learning，白话是“每个 agent 各学各的，把其他人当环境”）实现简单，但会遇到非平稳问题；完全联合学习（Joint Learning，白话是“把所有 agent 当成一个超大 agent 一起学”）能直接看联合动作，但动作空间会按乘积增长，规模稍大就不可用。工程上真正常落地的路线，通常是 CTDE 或 value factorization。CTDE 是 centralized training with decentralized execution，白话是“训练时看更多全局信息，执行时每个 agent 仍独立行动”；value factorization 是“把团队价值拆成多个局部价值再组合”。

一个最简单的玩具例子能说明核心矛盾。两个机器人一起搬箱子，各自动作只有 `L` 和 `R`。只有两人同时向左 `(L,L)` 才能让箱子移动，奖励为 1，其余联合动作奖励都为 0。这里问题不在于某个机器人单独选 `L` 是不是好动作，而在于两个动作组合起来是不是有效协调。多智能体 RL 学的本质，是“联合行为模式”，不是多个单体策略的简单相加。

| 维度 | 单智能体 RL | 多智能体 RL |
|---|---|---|
| 环境是否近似固定 | 通常是 | 往往不是 |
| 回报是否受他人策略影响 | 否 | 是 |
| 是否存在联合动作爆炸 | 否 | 是 |
| 主要难点 | 探索、稀疏奖励 | 非平稳、协调、信用分配 |

---

## 问题定义与边界

本文讨论的是合作型多智能体强化学习，即所有 agent 共享团队目标，不讨论主要以竞争为主的博弈场景。标准形式通常写成 Dec-POMDP。Dec-POMDP 是 decentralized partially observable Markov decision process，白话是“每个 agent 只能看到局部信息，但团队要在不完整观察下共同决策”。

形式化地，可以写成：

$$
\mathcal M=\langle S,\{A_i\},T,R,\{\Omega_i\},O,\gamma\rangle
$$

其中：

- $S$：全局状态，白话是“环境完整真相”
- $A_i$：第 $i$ 个 agent 的动作空间
- $A=\times_i A_i$：联合动作空间，白话是“所有 agent 动作组合起来的全集”
- $T$：状态转移函数，描述执行联合动作后环境怎么变
- $R$：共享团队奖励，所有 agent 拿到同一个团队回报
- $\Omega_i$：第 $i$ 个 agent 的局部观测空间
- $O$：观测函数，决定每个 agent 能看到什么
- $\gamma$：折扣因子，控制未来奖励的权重

合作场景的关键约束有两个。第一，奖励共享不等于最优策略能独立求解。因为“共享一个分数”只说明目标一致，不说明每个局部动作对最终分数的贡献容易归因。第二，局部观测意味着单个 agent 通常拿不到完整状态，因此即使团队目标明确，也可能因为信息不足而难以协调。

新手可理解版本可以看仓储 AGV 例子。假设有 3 台搬运车，每台车只能看到附近通道和邻近车位，但整体目标是“单位时间内完成尽可能多订单，同时避免碰撞和拥堵”。这就是典型 Dec-POMDP：全局地图和任务分布是真实状态 $S$，每辆车看到的局部栅格是观测 $\Omega_i$，所有车辆共享同一个吞吐量相关奖励 $R$。

本文边界如下：

| 任务类型 | 是否共享奖励 | 是否本文重点覆盖 |
|---|---|---|
| 合作任务 | 是 | 是 |
| 混合合作竞争任务 | 部分共享 | 部分提及，不展开 |
| 完全对抗任务 | 否 | 否 |

因此，后文所有机制都围绕“共享团队奖励 + 局部观测 + 多 agent 协同”展开，而不是一般博弈论全景。

---

## 核心机制与推导

合作型 MARL 的优化目标可以写成：

$$
J(\pi)=\mathbb E_\pi\Big[\sum_{t=0}^{T-1}\gamma^t r_t\Big],\quad r_t=R(s_t,\mathbf a_t)
$$

这里 $\pi$ 表示所有 agent 策略的集合，$\mathbf a_t$ 表示时刻 $t$ 的联合动作。白话说，团队想最大化整个轨迹上的累计回报，而奖励由“当前状态 + 所有 agent 的动作组合”共同决定。

第一层难点是非平稳。非平稳（non-stationarity，白话是“你在学的时候，训练对象本身也在变”）来自这样一个事实：如果 agent $i$ 把其他 agent 当环境的一部分，那么它实际面对的转移分布依赖别人当前策略：

$$
P(s_{t+1}\mid s_t,a_i^t)=\sum_{\mathbf a_{-i}}P(s_{t+1}\mid s_t,a_i^t,\mathbf a_{-i})\prod_{j\ne i}\pi_j(a_j^t\mid h_j^t)
$$

其中 $\mathbf a_{-i}$ 是除了 agent $i$ 之外其他 agent 的动作，$h_j^t$ 是第 $j$ 个 agent 到时刻 $t$ 的历史信息。只要其他 agent 在更新策略，上式右边就会变。于是对 agent $i$ 来说，同一个状态动作对，今天估出来的价值，明天可能已经对应另一套环境分布。独立 Q-learning 往往在这里变得不稳定。

玩具例子最直观。两个 agent 各有动作 $\{L,R\}$，且只有 $(L,L)$ 奖励为 1，其余为 0。若两个 agent 独立随机，各以 0.5 选 $L$，则：

$$
\mathbb E[r]=P(L,L)=0.5\times 0.5=0.25
$$

如果能稳定协调成总是选 $L$，则期望奖励变成 1。这里并不是某个 agent “没学会左转”，而是独立优化缺少对联合动作模式的建模。

第二层难点是联合动作爆炸。若有 $n$ 个 agent，每个 agent 有 $m$ 个离散动作，则联合动作空间大小是 $m^n$。当 $n=20,m=10$ 时，联合动作数是 $10^{20}$，无法直接枚举。完全联合学习虽然概念上最干净，但在规模稍大时就失去可计算性。

因此才需要折中路线。

CTDE 的思路是：训练期允许使用全局状态、所有 agent 动作、甚至团队历史，帮助价值估计更稳定；执行期仍要求每个 agent 只依赖本地观测决策。它解决的是“训练时需要更多信息，部署时拿不到那么多信息”的矛盾。

另一类常见方法是价值分解。以 QMIX 为例，核心形式是：

$$
Q_{tot}(s,\mathbf a)=f\big(Q_1(o_1,a_1),\dots,Q_n(o_n,a_n),s\big),\quad \frac{\partial Q_{tot}}{\partial Q_i}\ge 0
$$

这里 $Q_i$ 是第 $i$ 个 agent 的局部动作价值，$Q_{tot}$ 是团队总价值，$f$ 是混合器。单调约束 $\frac{\partial Q_{tot}}{\partial Q_i}\ge 0$ 的含义是：如果某个局部动作更好，混合后的团队价值不会反向变差。白话说，局部贪心和全局贪心保持方向一致，这样执行时每个 agent 只需根据自己的 $Q_i$ 选动作。

真实工程例子可以看仓储多机器人拣货。每辆 AGV 只看局部地图，但团队要同时优化订单完成数、碰撞率和通道拥堵。独立学习容易出现“局部绕路正确、全局形成堵塞”；完全联合优化又太大。CTDE 或 QMIX 这类方法，正好对应“训练时用全局车间状态做团队价值估计，执行时每台车仍只看本地传感器”。

| 方法 | 优点 | 缺点 | 适用条件 |
|---|---|---|---|
| 独立学习 | 简单、可并行 | 非平稳严重、协同弱 | 任务耦合弱、小规模基线 |
| 完全联合学习 | 直接建模联合最优 | 动作空间指数爆炸 | agent 很少、动作很小 |
| CTDE | 训练稳定、部署可分布式 | 需要训练期全局信息 | 合作任务、可收集全局日志 |
| VDN | 分解简单、易训练 | 表达能力有限 | 奖励结构较接近可加 |
| QMIX | 比 VDN 更强，兼顾分布式执行 | 仍受单调性约束 | 协作强、离散动作、需要扩展性 |

---

## 代码实现

落地实现时，先不要把算法名当成核心，先看数据流。一个最小合作 MARL 系统，通常包含四块：环境交互循环、每个 agent 的局部网络、训练期使用的 centralized critic 或 mixer、经验回放与参数更新。

先看一个可运行的玩具实现。它不依赖深度学习，只用表格法模拟“两个 agent 必须协调成 `(L,L)` 才有回报”的环境，用来验证联合价值与独立价值的差异。

```python
from itertools import product

ACTIONS = ["L", "R"]

def team_reward(a1, a2):
    return 1 if (a1, a2) == ("L", "L") else 0

# 联合动作价值表
joint_q = {(a1, a2): team_reward(a1, a2) for a1, a2 in product(ACTIONS, ACTIONS)}

# 独立边际“价值”：把对方动作视为均匀随机
ind_q_agent1 = {}
for a1 in ACTIONS:
    ind_q_agent1[a1] = sum(team_reward(a1, a2) for a2 in ACTIONS) / len(ACTIONS)

ind_q_agent2 = {}
for a2 in ACTIONS:
    ind_q_agent2[a2] = sum(team_reward(a1, a2) for a1 in ACTIONS) / len(ACTIONS)

best_joint = max(joint_q, key=joint_q.get)
best_local_1 = max(ind_q_agent1, key=ind_q_agent1.get)
best_local_2 = max(ind_q_agent2, key=ind_q_agent2.get)

assert best_joint == ("L", "L")
assert joint_q[best_joint] == 1
assert ind_q_agent1["L"] == 0.5
assert ind_q_agent1["R"] == 0.0
assert (best_local_1, best_local_2) == ("L", "L")

print("best_joint:", best_joint, "reward:", joint_q[best_joint])
print("local_values_1:", ind_q_agent1)
print("local_values_2:", ind_q_agent2)
```

这段代码说明两件事。第一，联合价值是直接定义在 $(a_1,a_2)$ 上的；第二，即使局部边际价值能在这个玩具问题里给出正确偏好，真实任务里也很容易因为状态依赖、部分观测和策略同时更新而失效，所以需要更稳定的训练结构。

如果用深度学习实现 CTDE/QMIX，训练循环通常长这样：

```python
for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        actions = []
        for i in range(n_agents):
            actions.append(agent_q[i].act(obs[i]))  # 执行时只看局部观测

        next_obs, reward, done, info = env.step(actions)
        replay_buffer.add(obs, actions, reward, next_obs, done, info["state"])
        obs = next_obs

    batch = replay_buffer.sample()

    # 训练时可读取全局状态、联合动作、团队奖励
    q_local = compute_local_q(batch.obs, batch.actions)
    q_tot = mixer(q_local, batch.state)            # QMIX / VDN
    target_q_tot = target_mixer(...)
    loss = ((q_tot - target_q_tot) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这里最容易理解错的点是“谁在什么时候看什么信息”。以 QMIX 为例：

1. 每个 agent 网络输入自己的局部观测 $o_i$，输出局部动作价值 $Q_i(o_i,a_i)$。
2. mixer 在训练期把所有 $Q_i$ 和全局状态 $s$ 组合成 $Q_{tot}(s,\mathbf a)$。
3. 反向传播时，团队损失会通过 mixer 回传到各个 agent 网络。
4. 执行期没有 mixer 参与决策，每个 agent 只根据本地观测独立选动作。

这就是“训练看全局、执行看局部”的最小结构。

真实工程例子里，仓储机器人系统常见做法是：局部输入包括自身速度、目标货架方向、周边占用栅格；团队奖励可能定义为“完成订单数 - 碰撞罚分 - 超时罚分 - 拥堵罚分”；训练时额外记录全局车间状态做 critic 或 mixer 输入。这样一来，部署端每台车无需全局地图推理器，只要保留本地策略网络即可。

| 维度 | 训练时 | 执行时 |
|---|---|---|
| 可见信息 | 可包含全局状态、联合动作、团队回报 | 通常只有局部观测 |
| 是否使用 mixer / centralized critic | 是 | 否 |
| 是否依赖通信 | 可选 | 取决于部署设计 |
| 是否适合真实部署 | 取决于是否满足执行侧信息约束 | 是，前提是只用可获得信息 |

---

## 工程权衡与常见坑

第一个常见误解是“既然共享奖励，那每个 agent 独立优化总能学到合作”。这不成立。共享奖励只说明目标一致，不说明信用分配（credit assignment，白话是“团队结果到底该归因给谁的哪个动作”）容易。比如一次任务失败，可能是导航 agent 走错，也可能是调度 agent 分配冲突，还可能是两者单独都没错，但时序组合错了。没有额外结构时，共享回报很难给出清晰梯度。

第二个坑是直接枚举联合动作。小规模玩具环境里看起来最自然，但一旦 agent 数量上去，计算和探索成本都会失控。这个问题通常不是“机器再大一点就能扛住”，而是复杂度形式本身就不对。

第三个坑是训练和执行的信息条件不一致。训练时如果用了全局地图、其他 agent 隐状态、未来信息或稳定通信，测试时却拿不到，那最终效果往往会断崖式下降。CTDE 的前提不是“训练时随便多看”，而是“训练期额外信息只用于辅助学习，执行策略本身不能依赖不可得信息”。

第四个坑是把通信当成万能药。通信能缓解部分可观测问题，但不能自动解决信用分配、奖励稀疏和联合探索困难。如果奖励本身不可归因，只增加消息通道，通常只会让训练更不稳定。

第五个坑是评估指标太单一。团队总回报高，不代表系统就能上线。仓储、车队、机器人编队这类任务，还必须同时看碰撞率、超时率、协作成功率等约束指标。例如可定义：

$$
\text{TeamReturn}=\sum_{t=0}^{T-1}\gamma^t r_t
$$

同时记录：

- 碰撞率：每百步碰撞次数
- 超时率：任务超出截止时间的比例
- 协作成功率：需要多 agent 联动的任务完成比例

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 把共享奖励理解成独立优化 | 学不到协同，信用分配失败 | 小规模先验证联合或 CTDE 基线 |
| 联合动作直接枚举 | 计算与探索爆炸 | 中大规模优先价值分解或 actor-critic |
| CTDE 训练/执行信息不一致 | 训练好、部署差 | 严格区分训练辅助信息与执行输入 |
| 只加通信，不做归因设计 | 效果不稳定 | 奖励设计先保证可归因 |
| 训练期通信与测试期通信不一致 | 评估虚高 | 测试时复现真实通信约束 |

实践上可以记三个原则。小规模任务先验证“联合优化是否真的必要”；中大规模合作任务优先考虑 CTDE；局部观测很强时再考虑通信，但通信设计必须与部署条件一致。

---

## 替代方案与适用边界

没有任何一种 MARL 方法适合所有任务，方法选择主要看四个条件：是否合作、是否部分可观测、agent 数量规模、执行时是否必须分布式。

如果任务规模很小，比如只有 2 到 3 个 agent，且每个 agent 动作空间很小，那么直接联合建模往往最直观，调试成本也最低。这时没必要为了“追论文名词”强行上复杂 mixer。相反，如果是 20 台机器人、每台 10 个动作的系统，联合动作空间立刻爆炸，必须转向 CTDE、价值分解或通信式方法。

Independent Q-Learning 适合当基线，不适合承担强协作主力。VDN 假设团队价值近似可加，结构简单，适合先跑通；QMIX 通过单调混合器提升表达能力，在离散合作任务里很常见；MADDPG 属于 centralized critic 的 actor-critic 路线，适合连续动作或混合协作竞争环境；通信式 MARL 适合局部观测极强的场景；而 centralized planning 更像“集中调度”，在执行端不要求分布式时反而更简单。

| 方法 | 合作任务 | 是否需要全局状态 | 支持分布式执行 | 大规模扩展性 | 信用分配难度 |
|---|---|---|---|---|---|
| Independent Q-Learning | 可用但弱 | 否 | 是 | 中 | 高 |
| VDN | 是 | 训练期通常需要 | 是 | 较好 | 中 |
| QMIX | 是 | 训练期需要 | 是 | 较好 | 中 |
| MADDPG | 是，也可用于混合任务 | 训练期 critic 需要 | 是 | 中 | 中 |
| Communication-based MARL | 是 | 不一定 | 是 | 取决于通信开销 | 中到高 |
| Centralized planning | 是 | 是 | 否或弱 | 差到中 | 低，但部署受限 |

适用边界可以直接这样判断：

- 小规模且动作空间小：可先试联合学习
- 强合作且共享奖励明确：优先 CTDE 或价值分解
- 部分可观测严重：考虑通信或记忆模块
- 需要分布式执行：避免执行时依赖全局状态
- 不允许稳定通信：不要把通信当必要前提

真实工程里，很多系统最终不会纯粹采用某一篇论文的方法，而是“CTDE + 简化奖励 + 局部通信 + 规则安全层”的混合方案。原因很直接：工程目标不是证明方法新，而是在资源、可解释性和部署约束下稳定完成任务。

---

## 参考资料

建议阅读顺序是：先看 Dec-POMDP 和复杂度，理解为什么多智能体比单智能体难；再看合作学习和信用分配；最后看 VDN、QMIX、MADDPG 和通信方法，这样理论和工程实现能连起来。

| 阅读目标 | 建议先读 |
|---|---|
| 理解问题为什么难 | Bernstein et al. (2002) |
| 理解合作学习早期思路 | Lauer & Riedmiller (2004) |
| 理解价值分解 | Sunehag et al. (2017), Rashid et al. (2018) |
| 理解 actor-critic 路线 | Lowe et al. (2017) |
| 理解通信机制 | Foerster et al. (2016) |

1. [The Complexity of Decentralized Control of Markov Decision Processes](https://doi.org/10.1287/moor.27.4.819.297)
2. [Reinforcement Learning for Stochastic Cooperative Multi-Agent Systems](https://doi.org/10.1109/AAMAS.2004.226)
3. [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
4. [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
5. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
6. [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676)

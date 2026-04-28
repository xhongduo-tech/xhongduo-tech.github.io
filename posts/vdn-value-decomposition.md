## 核心结论

VDN（Value Decomposition Networks，价值分解网络）是 cooperative MARL 中的一类 value decomposition 方法。cooperative MARL 的白话解释是：多个智能体不是互相对抗，而是共同追求一个团队目标。VDN 的核心假设非常直接：

$$
Q_{\text{tot}}(s,\mathbf a)=\sum_{i=1}^n Q_i(s_i,a_i)
$$

这里 $Q_{\text{tot}}$ 是联合动作价值，白话解释是“整个团队在当前状态下执行这组动作到底值多少”；$Q_i(s_i,a_i)$ 是第 $i$ 个 agent 的局部价值，白话解释是“这个 agent 在自己看到的信息下，这个动作值多少”。

这个加法假设带来一个关键结果：如果联合价值真能写成各 agent 局部价值之和，那么联合最优动作就可以拆成每个 agent 各自做局部最优选择。也就是：

$$
\arg\max_{\mathbf a} Q_{\text{tot}}(s,\mathbf a)
=
\left[
\arg\max_{a_1} Q_1(s_1,a_1),\ldots,\arg\max_{a_n} Q_n(s_n,a_n)
\right]
$$

这正是 VDN 支持 CTDE 的原因。CTDE（Centralized Training with Decentralized Execution，集中训练、分散执行）的白话解释是：训练时可以看全局信息，执行时每个 agent 只靠自己局部观测做决策。

新手可以先把它理解成一句话：团队总分等于个人贡献分数之和。如果每个人都把自己的分数做到最高，团队总分也会被推高。

---

## 问题定义与边界

VDN 解决的不是“一切多智能体协作问题”，而是“可加协作”问题。这里先把几个最小术语定义清楚：

| 术语 | 数学记号 | 白话解释 |
|---|---|---|
| 全局状态 | $s$ | 环境完整信息，训练时可能可见 |
| 局部状态/局部观测 | $s_i$ | 第 $i$ 个 agent 自己能看到的信息 |
| 局部动作 | $a_i$ | 第 $i$ 个 agent 的动作 |
| 联合动作 | $\mathbf a=(a_1,\dots,a_n)$ | 所有 agent 动作拼起来的整体动作 |
| 团队奖励 | $r$ | 团队共同收到的回报 |

在 cooperative MARL 里，目标通常是最大化团队期望回报：

$$
\max_\pi \mathbb E\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

其中 $\gamma$ 是折扣因子，白话解释是“未来奖励折算到现在的重要程度”。

VDN 的适用前提是：团队收益结构应该近似可加。也就是，团队做得好，基本可以理解成“每个 agent 各自贡献一部分，总体加起来更好”。如果收益主要来自动作之间的联动关系，而不是简单叠加，VDN 就会失真。

先看一个适合的真实工程例子。仓储多机器人搬运系统里，每台机器人负责一片区域，奖励可以定义成总搬运件数、总吞吐或总延迟下降。如果机器人之间耦合不强，那么整体收益通常可以近似写成“各机器人局部效率贡献之和”。这种场景中，VDN 往往足够。

再看一个不适合的玩具例子。两人协调游戏里，只有两个人选同一动作才得分：

| Agent 1 / Agent 2 | X | Y |
|---|---:|---:|
| A | 1 | 0 |
| B | 0 | 1 |

这个奖励不是“你贡献一点、我贡献一点”的结构，而是“只有配合关系成立才有奖励”。这种收益依赖联动，不是简单加法能表达的。

可以把任务边界总结成下面这张表：

| 任务类型 | 是否近似可加 | 是否适合 VDN | 原因 |
|---|---|---|---|
| 多机器人独立搬运 | 是 | 是 | 每台机器人贡献可近似相加 |
| 多路并行分拣 | 是 | 是 | 团队吞吐常可拆成局部吞吐之和 |
| 协调开门/同步抬运 | 否 | 否 | 必须联合动作同时成立才有收益 |
| 双车交叉避让 | 往往否 | 谨慎 | 收益依赖时序协调和相互让行 |
| 协调博弈类任务 | 否 | 否 | 奖励主要来自动作组合关系 |

边界结论可以直接写成一句话：VDN 解决的是“可加协作”，不是一般意义上的协作建模。

---

## 核心机制与推导

VDN 的机制可以拆成三步。

第一步，做价值分解。原本我们想学的是联合动作价值 $Q_{\text{tot}}(s,\mathbf a)$，但它的输入维度会随着 agent 数量增长而快速爆炸。VDN 直接假设它可以分解成若干局部价值之和：

$$
Q_{\text{tot}}(s,\mathbf a)=\sum_{i=1}^n Q_i(s_i,a_i)
$$

这一步的意义是降低复杂度。每个 agent 只需要学一个局部 Q 函数，而不是整个联合动作空间上的大 Q 函数。

第二步，说明为什么联合最大化可以拆成逐维最大化。因为

$$
Q_{\text{tot}}(s,\mathbf a)
=
Q_1(s_1,a_1)+Q_2(s_2,a_2)+\cdots+Q_n(s_n,a_n)
$$

每一项只依赖对应 agent 自己的动作，所以在独立动作空间里：

$$
\max_{a_1,\dots,a_n}\sum_i Q_i(s_i,a_i)
=
\sum_i \max_{a_i} Q_i(s_i,a_i)
$$

因此联合最优动作就是每一维单独取最大值对应的动作。这就得到：

$$
\arg\max_{\mathbf a} Q_{\text{tot}}(s,\mathbf a)
=
\left[
\arg\max_{a_1} Q_1(s_1,a_1),\ldots,\arg\max_{a_n} Q_n(s_n,a_n)
\right]
$$

第三步，说明为什么这保证了去中心化执行。执行时每个 agent 不需要知道别人的局部 Q，也不需要知道全局状态，只要对自己的 $Q_i(s_i,a_i)$ 做贪心选择即可。这就是“训练集中、执行分散”的结构基础。

看一个最小数值玩具例子。假设有两个 agent：

- Agent 1: $Q_1(A)=2,\ Q_1(B)=0$
- Agent 2: $Q_2(X)=1,\ Q_2(Y)=3$

则联合价值为：

- $Q_{\text{tot}}(A,X)=2+1=3$
- $Q_{\text{tot}}(A,Y)=2+3=5$
- $Q_{\text{tot}}(B,X)=0+1=1$
- $Q_{\text{tot}}(B,Y)=0+3=3$

显然最大的是 $(A,Y)$。而局部贪心也是 Agent 1 选 $A$、Agent 2 选 $Y$。局部最优组合就是联合最优，这不是巧合，而是加法结构的直接结果。

训练时通常不是直接拟合即时奖励，而是对 $Q_{\text{tot}}$ 做 TD 学习。TD（Temporal Difference，时序差分）的白话解释是：用“当前奖励 + 对未来价值的估计”来更新当前价值。一个典型目标是：

$$
y = r + \gamma \max_{\mathbf a'} Q_{\text{tot}}^{-}(s',\mathbf a')
$$

其中 $Q_{\text{tot}}^{-}$ 表示 target network，白话解释是“冻结一段时间不动、让训练更稳定的旧网络”。

这一机制可以压缩成下表：

| 项目 | VDN 中的含义 | 执行时是否需要全局状态 |
|---|---|---|
| 局部 Q | $Q_i(s_i,a_i)$，每个 agent 自己的动作价值 | 否 |
| 联合 Q | $Q_{\text{tot}}=\sum_i Q_i$ | 否，执行时不直接算也行 |
| TD target | $r+\gamma \max_{\mathbf a'}Q_{\text{tot}}^{-}(s',\mathbf a')$ | 训练时常需要 |
| mixer | 把多个局部 Q 合成总 Q 的模块 | 否 |
| 全局状态 | 用于训练辅助或 critic 输入 | 训练时可用，执行时不用 |

顺带对比一下 QMIX。QMIX 仍然保留“局部贪心能导出联合贪心”的约束，但不再要求纯加法，而是允许受全局状态条件控制的单调混合。VDN 是最简单的纯加法版本，QMIX 是它的表达力增强版。

| 方法 | 混合形式 | 表达能力 | 去中心化执行 |
|---|---|---|---|
| VDN | 纯加法 | 较弱 | 是 |
| QMIX | 状态条件下的单调非线性混合 | 更强 | 是 |

---

## 代码实现

代码层面，VDN 的 mixer 几乎是所有 value factorization 方法里最直接的。本质就是把每个 agent 选出的局部 Q 沿 agent 维度求和。

下面给一个可运行的最小 Python 例子。它不依赖深度学习框架，只演示 VDN 的核心逻辑，并用 `assert` 验证“局部贪心等于联合最优”这件事。

```python
from itertools import product

def vdn_total_q(local_qs, joint_action):
    return sum(local_qs[i][a] for i, a in enumerate(joint_action))

# 两个 agent 的局部 Q
local_qs = [
    {"A": 2.0, "B": 0.0},   # Agent 1
    {"X": 1.0, "Y": 3.0},   # Agent 2
]

# 每个 agent 局部贪心
greedy_actions = tuple(
    max(agent_q.items(), key=lambda kv: kv[1])[0]
    for agent_q in local_qs
)

# 穷举联合动作，找联合最优
all_joint_actions = list(product(local_qs[0].keys(), local_qs[1].keys()))
best_joint_action = max(all_joint_actions, key=lambda a: vdn_total_q(local_qs, a))

assert greedy_actions == ("A", "Y")
assert best_joint_action == ("A", "Y")
assert vdn_total_q(local_qs, best_joint_action) == 5.0

print("greedy_actions =", greedy_actions)
print("best_joint_action =", best_joint_action)
print("Q_tot =", vdn_total_q(local_qs, best_joint_action))
```

如果换成 PyMARL 风格的实现，`agent_qs` 常见形状是 `[batch, time, agents]`。VDN mixer 只需要在最后一个维度求和，得到 `[batch, time, 1]` 的团队总价值。伪代码如下：

```python
import torch
import torch.nn as nn

class VDNMixer(nn.Module):
    def forward(self, agent_qs, states=None):
        # agent_qs: [batch, time, agents]
        # output:   [batch, time, 1]
        return agent_qs.sum(dim=-1, keepdim=True)

# 简单校验
agent_qs = torch.tensor([[[2.0, 3.0], [1.0, 4.0]]])  # [1, 2, 2]
mixer = VDNMixer()
q_tot = mixer(agent_qs)

assert q_tot.shape == (1, 2, 1)
assert torch.allclose(q_tot, torch.tensor([[[5.0], [5.0]]]))
```

训练流程通常是：

1. 每个 agent 网络输出自己动作空间上的 Q 值。
2. 根据当前实际执行动作，取出各 agent 对应动作的局部 Q。
3. 用 VDN mixer 把这些局部 Q 相加，得到 $Q_{\text{tot}}$。
4. 用 target network 计算目标值 $y$。
5. 最小化 $(Q_{\text{tot}}-y)^2$。

下面这张表可以帮助把代码对象和数学对象对应起来：

| 代码对象 | 数学对象 | 含义 |
|---|---|---|
| `agent_qs` | $Q_i(s_i,a_i)$ | 每个 agent 的局部 Q |
| `states` | $s$ | 全局状态，VDN 混合时通常不用 |
| `sum(dim=-1)` | $\sum_i$ | 把各 agent 价值加起来 |
| `q_tot` | $Q_{\text{tot}}$ | 团队总价值 |
| `target_max_q_tot` | $\max_{\mathbf a'}Q_{\text{tot}}(s',\mathbf a')$ | 下一时刻目标总价值 |

真实工程里，最常见的实现方式不是“直接写联合动作网络”，而是“共享参数的 agent 网络 + 一个极简 mixer”。这也是 VDN 在早期 cooperative MARL 工程实践里很受欢迎的原因：结构简单，训练链路短，调试成本低。

---

## 工程权衡与常见坑

VDN 的强项是简单、稳定、执行高效。它的弱点也同样明确：表达能力有限。不是效果偶尔差，而是模型结构本身表达不了某些协作关系。

先看常见风险表：

| 风险点 | 现象 | 结果 | 规避方式 |
|---|---|---|---|
| 奖励不可加 | 只有特定动作组合才有高奖励 | 学不到真实协同模式 | 改用 QMIX、QPLEX 或通信方法 |
| 协调依赖全局关系 | 单个 agent 看不到决定性信息 | 执行期决策失真 | 增强观测、加入通信、重新设定任务 |
| 局部最优不等于联合最优 | 每人都“各做各的最好”但组合很差 | 收敛到伪协作策略 | 检查任务结构是否适合 VDN |
| 奖励设计过粗 | 总奖励反馈太弱 | credit assignment 困难 | 增加更细粒度的辅助奖励 |
| 误把能跑当能协作 | 训练曲线有提升，但策略靠偶然碰运气 | 泛化差、部署不稳 | 做对照实验和行为分析 |

第一类坑是奖励不可加。比如同步抬运，只有两台机器人同一时刻配合抬起货物才有效。单个机器人“自己做得好”没有意义，必须动作配合才有价值。这类任务用 VDN 往往会把“协同带来的额外收益”压平。

第二类坑是协调依赖全局关系。比如双车交叉避让，关键不在于每辆车局部效率高不高，而在于它们是否正确预判彼此轨迹、让行次序是否合适。即便总奖励是共享的，局部 Q 的简单相加也很难刻画这种相互约束。

第三类坑是局部最优与联合最优不一致。注意，这句话不是说 VDN 的数学推导错了，而是说真实环境的最优结构可能本来就不是加法型。模型内部会强制把问题投影到“可加世界”里，所以它学出来的“局部贪心一致性”只是对假设成立时有意义。

真实工程例子里，这个问题很常见。比如仓储系统前期做单机仿真时，VDN 可能效果不错；但一旦引入狭窄通道、会车、共享装卸口、同步换道等约束，收益就不再接近可加。此时继续调学习率、网络层数、epsilon 衰减，通常只是修表面，不会解决结构问题。

实操建议很简单：先判断任务是否近似可加，再决定是否用 VDN。不要先上 VDN，再把所有不收敛现象都归因于“超参数没调好”。

---

## 替代方案与适用边界

当纯加法不够时，正确方向通常不是继续扩大 VDN 的解释范围，而是换更合适的 factorization 结构。

可以先看整体对比：

| 方法 | 核心假设 | 优点 | 局限 | 适用场景 |
|---|---|---|---|---|
| VDN | 联合 Q 可写成局部 Q 之和 | 简单、稳定、易实现 | 表达不了非加法协作 | 近似可加任务 |
| QMIX | 联合 Q 是局部 Q 的单调混合 | 比 VDN 表达力强，仍支持去中心化执行 | 仍受单调性约束 | 中等复杂协作任务 |
| QPLEX | 更强的优势分解结构 | 更接近完整联合价值表达 | 结构更复杂，训练成本更高 | 强协作、复杂信用分配 |
| 显式通信方法 | agent 间通过消息传递协同 | 能显式处理信息缺失和协调 | 通信设计复杂，训练更难 | 强耦合、多阶段协作 |

怎么选可以用一个对比式判断：

- 如果任务是“各自贡献可叠加”，VDN 往往足够。
- 如果任务是“某些动作组合才有效”，QMIX 往往比 VDN 更合适。
- 如果任务里存在复杂信用分配，且单调混合仍不够，QPLEX 这类更强分解方法更值得考虑。
- 如果核心难点不是价值混合，而是 agent 之间信息不对称，需要彼此传递意图，那么显式通信方法通常更直接。

这里再强调一次边界：VDN 的边界不是“它效果差”，而是“它结构表达不够”。在结构匹配的任务上，VDN 依然是非常高效的基线；在结构不匹配的任务上，越努力调参，越可能只是把错误假设训练得更稳定。

---

## 参考资料

1. [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
2. [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
3. [PyMARL `vdn.py`](https://raw.githubusercontent.com/oxwhirl/pymarl/master/src/modules/mixers/vdn.py)
4. [PyMARL `qmix.py`](https://raw.githubusercontent.com/oxwhirl/pymarl/master/src/modules/mixers/qmix.py)
5. [The StarCraft Multi-Agent Challenge](https://arxiv.org/abs/1902.04043)

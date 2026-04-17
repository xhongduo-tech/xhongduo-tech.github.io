## 核心结论

Model-based RL 的规划，可以先理解成两步：先得到一个环境模型，再在模型里反复“预演”未来。环境模型指的是对“执行动作后会到哪里、拿到什么奖励”的预测器，常写成 $\hat{\mathcal{M}}=(\hat{P},\hat{R})$。这里的“规划”不是单纯做路径搜索，而是在模型内部生成 imagined trajectory，也就是“想象出来的状态转移序列”，再把这些虚拟经验拿来更新价值函数或策略。

它和纯 model-free 方法的核心差别，不在更新公式，而在样本来源。纯 model-free 只能依赖真实交互得到 $(s,a,r,s')$；Model-based RL 则把一次真实交互拆成两份价值：第一份用于直接学习，第二份用于更新模型，随后再从模型中额外生成 $k$ 次虚拟样本继续学习。典型代表是 Dyna。Dyna 的直观含义很简单：每走一步真实环境，就在“脑中复盘”若干步。

因此，Model-based RL 的主要优势是样本效率高。样本效率就是“同样数量的真实交互，谁学得更快”。在真实数据昂贵的场景，比如机器人、工业控制、在线推荐中的高风险实验，能不能少采几次真实样本通常比单次计算快一点更重要。Dyna 的价值正在这里：它把一次真实样本反复利用，让值函数更快逼近正确解。

但它不是“白拿收益”。模型一旦有偏差，规划会把偏差也一起重复传播。规划次数越多、rollout 越长，错误被放大的机会越大。这就是 Model-based RL 的核心交换关系：用更多计算和更强的模型假设，换更少的真实样本；同时承担模型偏差被规划放大的风险。

---

## 问题定义与边界

先把问题说清楚。强化学习中的环境通常建模为 MDP，中文叫“马尔可夫决策过程”，可以理解成“当前状态加上当前动作，就足以决定下一步概率分布和奖励规则”的决策系统。给定状态 $s$ 和动作 $a$，真实环境由 $P(s'|s,a)$ 和 $R(s,a)$ 决定，而 Model-based RL 只拥有它们的真实版本或估计版本 $\hat{P},\hat{R}$。

因此，Model-based RL 的规划问题不是“会不会更新 Q”，而是“在一个不完全准确的模型上，额外做多少次更新才划算”。这个问题有三个边界条件。

| 因素 | 它控制什么 | 过小的后果 | 过大的后果 |
| --- | --- | --- | --- |
| 模型准确性 | 虚拟样本是否可信 | 规划收益有限 | 偏差被当成真相传播 |
| 规划次数 $k$ | 每个真实样本被复用多少次 | 样本效率提升不明显 | 训练被虚假经验主导 |
| Rollout 长度 | 单次想象轨迹向前推多远 | 只能做很局部备份 | 长链误差快速累积 |

新手最容易混淆的一点是：规划次数 $k$ 增大，不等于效果一定更好。因为模型误差通常不是独立噪声，而会在多步预测里累积。假设真实环境里某动作有 $0.9$ 的概率转向目标状态，而模型错误地学成了 $0.95$。只错一步，影响可能不大；连续错五步，最后学到的值函数就可能明显偏离真实回报。

一个玩具例子可以说明这个边界。设只有两个状态 $s_0,s_1$，一个动作 $a$。在真实环境里，执行 $a$ 后必然从 $s_0$ 到 $s_1$，并得到奖励 $+1$，之后终止。此时如果真实样本只采到一次，那么纯 model-free 只更新一次；而 Dyna 可以在记录了这个转移后，从模型里继续抽取 $(s_0,a)$ 做多次备份，于是 $Q(s_0,a)$ 很快被推向正确值。这个例子说明规划确实能放大单个真实样本的作用。

但如果模型错了，比如它错误地认为到达 $s_1$ 后还可以持续拿奖励，那么规划就会稳定地把这个假信号灌进 Q 表。也就是说，Model-based RL 的边界不是“能不能规划”，而是“模型错到什么程度时，规划开始弊大于利”。

从理论上看，在有限 MDP 或更一般的线性表示设定里，基于模型的方法可以达到很好的样本复杂度。样本复杂度可以简单理解成“为了把策略学到足够好，大概需要多少真实样本”。这说明 model-based 路线不是经验技巧，而是有明确理论支撑的。但理论成立的前提通常包含模型误差可控、状态表示合理、估计集中性成立等条件，工程上不能直接把理论结论等价成“多做规划一定有利”。

---

## 核心机制与推导

Model-based RL 规划的数学核心，是把 Bellman 备份从真实环境搬到估计模型上。Bellman 备份可以理解成“用一步未来的最好结果，反推当前动作值应该是多少”的递推规则。

在模型 $\hat{\mathcal{M}}$ 下，最常见的动作价值更新写成：

$$
Q(s,a)\leftarrow (1-\alpha)Q(s,a)+\alpha\left[\hat{R}(s,a)+\gamma\sum_{s'}\hat{P}(s'|s,a)\max_{a'}Q(s',a')\right]
$$

其中：
- $\alpha$ 是学习率，可以理解成“新信息覆盖旧估计的速度”。
- $\gamma$ 是折扣因子，可以理解成“未来奖励在今天还值多少”。
- $\hat{P}(s'|s,a)$ 是模型预测的下一状态分布。
- $\hat{R}(s,a)$ 是模型预测的即时奖励。

如果用算子形式写，定义模型上的 Bellman 最优算子：

$$
\mathcal{T}_{\hat{\mathcal{M}}}Q(s,a)=\hat{R}(s,a)+\gamma\sum_{s'}\hat{P}(s'|s,a)\max_{a'}Q(s',a')
$$

那么做 $k$ 次规划，可以看成反复应用这个算子：

$$
Q \leftarrow \mathcal{T}_{\hat{\mathcal{M}}}^{k}Q
$$

这行式子的含义很重要：规划本质上不是另起一套学习规则，而是在模型上加速 Bellman 迭代。真实样本更新和 imagined 样本更新之所以可以并列放在同一个算法里，就是因为两者共享同一类目标，只是一个来自真实环境，一个来自估计模型。

Dyna 的机制因此非常直接：

1. 从真实环境执行动作，得到 $(s,a,r,s')$。
2. 用这条真实样本更新 Q。
3. 把这条样本写进模型，得到对 $\hat{P},\hat{R}$ 的最新估计。
4. 从模型中随机或按优先级抽样若干个状态动作对。
5. 对这些虚拟样本再做 $k$ 次 Q 更新。

对新手来说，可以把它想成“一个学生做题后立刻复盘”。真实做题只做了一次，但复盘时会重放解题步骤、替换一些分支、反复检查同类题，于是同样一次经历带来的学习信号更多。

再看一个更完整的玩具例子。还是两状态系统，终止状态 $s_1$ 的所有动作价值都为 $0$，在 $s_0$ 执行动作 $a$ 可得到奖励 $1$ 并转移到 $s_1$。若 $\gamma=0.9$，则最优值就是：

$$
Q^\*(s_0,a)=1+0.9\times 0 = 1
$$

若初始 $Q=0$，做一次真实更新后已经向 $1$ 靠近；如果接着做多次模型规划，即重复使用同一条可靠转移，那么 Q 会更快收敛。这个例子里，规划没有创造新信息，而是把已有信息传播得更彻底。

真实工程例子则更复杂。以机器人抓取为例，真实试错成本高，机械臂碰撞还可能损坏设备。工程上常先引入部分物理先验，例如近似动力学、摩擦项、关节约束，再结合真实日志学一个残差模型。残差模型就是“用学习器补上物理模型没解释到的那部分误差”。此时规划不一定做很长 rollout，而是常在短时域内生成 imagined 轨迹，用于价值学习或候选动作筛选。这样做的原因不是学术上的优雅，而是现实约束：真实交互太贵，纯 model-free 往往不够用。

---

## 代码实现

下面给一个可运行的 Dyna-Q 极简实现。它故意只保留最核心结构：真实更新、模型记录、规划更新。环境使用前面的两状态玩具例子，方便直接看出“规划放大样本”的效果。

```python
from collections import defaultdict

class TinyEnv:
    def reset(self):
        return "s0"

    def step(self, state, action):
        assert state == "s0"
        assert action == "a"
        return "s1", 1.0, True

class DeterministicModel:
    def __init__(self):
        self.memory = {}

    def record(self, s, a, r, s_next):
        self.memory[(s, a)] = (r, s_next)

    def sample_state_action(self):
        return next(iter(self.memory.keys()))

    def predict(self, s, a):
        return self.memory[(s, a)]

def dyna_q_once(k=5, alpha=0.5, gamma=0.9):
    env = TinyEnv()
    model = DeterministicModel()
    Q = defaultdict(float)

    s = env.reset()
    a = "a"
    s_next, r, done = env.step(s, a)

    # 1) 真实样本更新
    target = r + gamma * 0.0
    Q[(s, a)] += alpha * (target - Q[(s, a)])

    # 2) 写入模型
    model.record(s, a, r, s_next)

    # 3) 规划更新
    for _ in range(k):
        s_sim, a_sim = model.sample_state_action()
        r_sim, s_sim_next = model.predict(s_sim, a_sim)
        target_sim = r_sim + gamma * 0.0
        Q[(s_sim, a_sim)] += alpha * (target_sim - Q[(s_sim, a_sim)])

    return Q[(s, a)]

q_without_planning = dyna_q_once(k=0)
q_with_planning = dyna_q_once(k=5)

assert 0 < q_without_planning < 1.0
assert q_with_planning > q_without_planning
assert q_with_planning < 1.0 + 1e-9

print("no planning:", round(q_without_planning, 4))
print("with planning:", round(q_with_planning, 4))
```

这段代码有两个观察点。

第一，真实样本和虚拟样本共用同一个目标形式。代码里 `target` 和 `target_sim` 的写法完全一致，只是前者来自环境，后者来自模型。这正对应前面说的“规划是在模型上做 Bellman 迭代”。

第二，`k` 只改变信息复用次数，不改变单次真实交互数量。`q_with_planning > q_without_planning` 的 `assert` 正是在验证这一点：真实只交互了一次，但做了额外规划后，Q 更接近正确值 1。

如果把这个最小实现扩展到更一般的 Dyna-Q，通常会增加三类工程模块：

| 模块 | 最小版本怎么做 | 工程版本会怎么做 |
| --- | --- | --- |
| 模型存储 | 哈希表记住见过的转移 | 神经网络或集成模型预测分布 |
| 抽样策略 | 随机抽历史状态动作 | 按 TD error 做 prioritized sweeping |
| 规划深度 | 单步 imagined backup | 多步 rollout + 早停机制 |

Prioritized Sweeping 可以简单理解成“优先复习错得最厉害的题”。这里的 TD error，也就是时序差分误差，可以理解成“当前 Q 估计和新备份目标之间的差值”。误差大的状态说明价值还没传对，优先规划它们通常比均匀抽样更高效。

真实工程里，一个常见模式是：训练阶段做短 rollout 规划，部署阶段尽量减少长时搜索。原因也很现实，训练时可以多算，线上控制往往有严格延迟预算，不能每一步都做很深的模型搜索。

---

## 工程权衡与常见坑

Model-based RL 难点不在“公式复杂”，而在“错误会被复用”。下面这些坑几乎都会遇到。

| 常见坑 | 具体表现 | 典型缓解方式 |
| --- | --- | --- |
| 模型偏差积累 | rollout 越长，虚拟轨迹越像在离真实分布漂移 | 缩短 rollout，只做局部规划 |
| 规划次数过大 | 训练后期被模型生成数据主导，Q 学歪 | 动态调节 $k$，混入真实数据更新 |
| 只看平均误差 | 平均很准，但关键状态预测很差 | 做不确定性估计，重点监控危险状态 |
| 训练分布外规划 | 模型在没见过的状态上胡乱外推 | 限制规划起点来自真实回放缓冲区 |
| 线上推理过慢 | 每步都做复杂搜索，延迟不满足要求 | 训练期多规划，部署期更多依赖策略网络 |

最值得强调的是 rollout 长度。很多初学者会自然地想：“既然模型能预测，那就一直往后滚，滚得越长越好。”这通常是错的。因为多步 rollout 的误差近似不是线性叠加，而是会通过状态分布偏移不断放大。前一步错一点，下一步就在错误状态附近继续预测，后面会越来越偏。

一个常见做法是只做短 horizon rollout。horizon 就是“向前看多少步”。短 rollout 的含义不是保守，而是承认模型只在局部可靠。OpenReview 上关于 rollout 的一些工作也强调要根据不确定性决定何时终止 imagined rollout。不确定性可以白话理解为“模型自己也知道这一步没那么有把握”。当 epistemic uncertainty，也就是“由于数据不够而导致的模型未知性”较高时，继续规划通常风险更大。

真实工程例子还是机器人控制最典型。假设一个机械臂要把零件插入卡槽。物理模型能大致描述关节运动和接触趋势，但接触瞬间的摩擦、弹性、视觉误差都很难准确建模。如果此时做很长的模型 rollout，后面预测到的姿态会逐渐脱离真实可执行区域，最后策略学到一堆“模型里能成功、现实里会卡住”的动作。更稳健的做法往往是：用物理先验提供局部可行性，再用真实数据纠正关键误差，并把规划控制在短时域内。

另一个坑是把 Dyna 理解成“固定多复盘几次就行”。事实上，$k$ 更适合当作一个需要调度的超参数。训练早期模型很粗糙，过大的 $k$ 可能让错误信号迅速污染 Q；训练中后期模型更稳定时，再增加规划预算通常更划算。也就是说，规划强度不应该脱离模型质量单独讨论。

---

## 替代方案与适用边界

如果问题的真实样本很便宜，或者环境过于复杂、难以学出可用模型，那么纯 model-free 方案经常更直接。比如一些大规模在线系统可以持续收集海量交互数据，此时多花精力建模型不一定划算。Q-learning、SARSA、actor-critic 等方法虽然样本效率偏低，但实现和调参链路更成熟。

如果模型非常可靠，而且推理时允许在线优化，那么 MPC 或 CEM 这类“纯规划”方法也常是更合适的选择。MPC 是模型预测控制，可以理解成“每一步都用模型向前试很多动作序列，然后只执行当前最优那一步”。CEM 是交叉熵法，可以理解成“不断筛选更优的动作序列分布”。这类方法不一定显式学习一个长期值函数，而是直接用模型做在线决策。

更常见的现实答案其实是混合方案：模型、策略、价值函数一起用。

| 方法 | 适合什么场景 | 主要代价 |
| --- | --- | --- |
| 纯 model-free | 数据充足、环境复杂、模型难拟合 | 样本需求大 |
| 纯规划（MPC/CEM） | 模型可靠、在线算力足够 | 推理成本高 |
| Dyna 式 MBRL | 真实样本昂贵、需要复用经验 | 模型偏差会影响学习 |
| 混合方案 | 同时要样本效率和上线速度 | 系统复杂度最高 |

因此，Model-based RL 的适用边界可以总结成一句话：当真实交互贵、局部模型可学、并且你愿意为偏差控制投入工程复杂度时，它通常值得；反过来，当数据便宜、模型不稳定、线上延迟极严时，它未必是最优选。

对初级工程师来说，最实用的判断标准不是“哪条路线更先进”，而是问三个问题：

1. 真实数据是不是昂贵到必须复用？
2. 我能不能学到一个至少局部可信的模型？
3. 我有没有资源监控并约束模型偏差？

这三个问题里，只要后两项答案很弱，就不应该盲目增加规划强度。

---

## 参考资料

| 来源 | 内容 |
| --- | --- |
| [Emergent Mind: Model-Based Reinforcement Learning](https://www.emergentmind.com/topics/model-based-reinforcement-learning-mbrl?utm_source=openai) | MBRL 定义、样本效率优势、模型偏差与规划边界 |
| [GeeksforGeeks: Dyna Algorithm in Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/dyna-algorithm-in-reinforcement-learning-/?utm_source=openai) | Dyna 的基本流程与直观解释 |
| [Next.gr: Model-Based Reinforcement Learning](https://next.gr/ai/reinforcement-learning/model-based-reinforcement-learning?utm_source=openai) | Dyna、Prioritized Sweeping 与 Bellman 备份公式 |
| [OpenReview: On Rollouts in Model-Based Reinforcement Learning](https://openreview.net/forum?id=PmmxvSlVna&utm_source=openai) | rollout 误差传播、不确定性与早停思想 |
| [OpenReview: Physics-Informed Model and Hybrid Planning for Efficient Dyna-Style RL](https://openreview.net/forum?id=2T4p0wQxp9&utm_source=openai) | 物理先验、混合规划与工程案例 |
| [PMC9512142: Sample-Efficient Reinforcement Learning for Linear MDPs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9512142/?utm_source=openai) | 基于模型方法的样本复杂度理论背景 |

## 核心结论

`model-based RL` 的定义很直接：先学习一个环境近似模型 $\hat p_\phi(s',r\mid s,a)$，再在这个模型里做规划，最后再决定真实执行什么动作。这里的“环境模型”就是“给定当前状态和动作，预测下一步会到哪里、得到多少奖励”的函数。

它和 `model-free RL` 的根本区别，不在于谁“更智能”，而在于是否**显式**学习环境转移与奖励。`model-free` 直接学价值函数 $Q$ 或策略 $\pi$；`model-based` 先学规则，再用规则推演未来。核心关系可以写成：

$$
\hat p_\phi(s',r\mid s,a)\approx p(s',r\mid s,a)
$$

对零基础读者，最直观的玩具例子是踢球：先学会“球踢出去大概率飞向哪、能不能得分”，再在脑中先算一遍“左脚还是右脚更划算”，最后才真的踢。这就是“先建模，再规划”。

它通常比 `model-free` 更省真实样本，因为很多试错发生在“脑内模拟”里，而不是在真实环境里重复犯错。但代价同样明确：如果模型学歪了，规划会系统性放大这种错误。模型越被依赖，偏差越危险。

下面这张表先给出一眼能看懂的对比：

| 维度 | Model-Free RL | Model-Based RL |
|---|---|---|
| 学什么 | 直接学 $Q(s,a)$ 或 $\pi(a\mid s)$ | 先学 $\hat p_\phi(s',r\mid s,a)$，再规划 |
| 是否显式建模环境 | 否 | 是 |
| 样本效率 | 通常较低 | 通常较高 |
| 计算开销 | 训练期高，决策期相对轻 | 训练和决策都可能较重 |
| 主要风险 | 真实样本需求大 | 模型偏差被规划放大 |
| 典型代表 | DQN、PPO、SAC | Dyna、MPC、MuZero 一类路线 |

---

## 问题定义与边界

强化学习要解决的问题是：在状态 $s$ 下选择动作 $a$，使长期累计回报最大。所谓“长期回报”，就是不仅看眼前奖励，还看这个动作会怎样影响后面一连串状态。标准目标写成：

$$
\max_\pi \mathbb E\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

这里 $\gamma$ 是折扣因子，白话讲就是“未来奖励按多大权重计入今天的决策”。

`model-based RL` 的特殊点，不是“用了搜索”，而是**显式学习环境动力学或奖励结构**。如果一个方法从头到尾只学策略，不学“动作之后环境怎么变”，那它就不是典型 `model-based`。

边界最容易混淆的地方有两个。

第一，不是所有“用了搜索”的方法都在学模型。比如你已经有一个完全可信的游戏规则引擎，直接在规则上做搜索，这叫 `search + known simulator`，不是“先学环境模型”。

第二，不是所有“有模拟器”的系统都属于 `model-based RL`。如果模拟器是人写好的真实规则，你没有学习它，只是在上面反复推演，那重点是“已知规则搜索”，不是“学习模型”。

新手版可以这样理解：如果你能直接调用游戏引擎反复模拟下一步，那是“有模拟器”；如果你没有引擎，先从数据里学出一个近似引擎，再拿它做搜索，才是典型 `model-based RL`。

三类方法放在一起看更清楚：

| 方法类型 | 核心对象 | 是否学习环境模型 | 是否使用搜索/规划 | 典型场景 |
|---|---|---|---|---|
| `model-free` | 直接学 $Q$ 或 $\pi$ | 否 | 可无 | 样本便宜、实现要简单 |
| `model-based` | 先学 $\hat p$ 再规划 | 是 | 常有 | 样本昂贵、需要高样本效率 |
| `search + known simulator` | 在真实规则上搜索 | 否 | 是 | 棋类、规则明确的游戏 |

这也是为什么 AlphaZero 经常被误分类。AlphaZero 很强，但它的核心不是“先学环境模型”，而是“已知规则 + MCTS 搜索 + 价值/策略网络辅助”。真正更接近“学模型后再搜索”的代表是 MuZero。

---

## 核心机制与推导

`model-based RL` 最重要的闭环可以拆成三步：学模型、用模型规划、再用真实数据校正。Dyna 是这条路线最经典的骨架，因为它把“真实经验”和“模型生成样本”放进同一个训练循环。

先看规划为什么可行。假设模型已经学到：

$$
\hat p_\phi(s',r\mid s,a)\approx p(s',r\mid s,a)
$$

那么动作价值就不必只靠真实采样估计，也可以在模型里做递推：

$$
\hat Q(s,a)=\mathbb E_{\hat p_\phi}\left[r+\gamma \max_{a'}\hat Q(s',a')\right]
$$

这里“递推”这个词的白话意思是：先看走这一步的即时收益，再把下一步最优选择的价值折回来。规划本质上就是不断重复这个计算。

最小数值玩具例子如下。设 $\gamma=0.9$，起点状态是 $s_0$，有两个动作：

- 动作 `a`：模型预测到达 $s_1$，即时奖励为 0，且已知 $V(s_1)=1$
- 动作 `b`：直接终局，奖励为 0

那么：

$$
Q(s_0,a)=0+0.9\times 1=0.9
$$

$$
Q(s_0,b)=0
$$

所以规划会选 `a`。这说明规划不是看“这一步立刻赚多少”，而是看“这一步把我带去哪里”。

但同一个例子也能说明模型偏差为什么危险。假设真实世界里 `a` 并不会立刻得分，只是去到一个中间状态；而模型误学成“执行 `a` 直接得 2 分”。那搜索就会过度偏向 `a`，因为它是在错误世界里做最优决策。规划本身没错，错的是它信了一个偏掉的世界模型。

Dyna 的思想正是为了解这个张力：真实数据贵，所以尽量复用；模型可能偏，所以持续用真实交互纠正。伪公式可以写成：

$$
\text{真实样本} \rightarrow \text{更新模型}
$$

$$
\text{真实样本} \rightarrow \text{更新价值/策略}
$$

$$
\text{模型生成 imagined\ samples} \rightarrow \text{继续更新价值/策略}
$$

其中 `imagined samples` 可以理解成“模型脑补出来的训练样本”。它们不是免费午餐，但能显著提高数据复用率。

把这个闭环写成步骤表更直观：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 真实交互 | $(s,a)$ | $(r,s')$ | 获取可信数据 |
| 更新模型 | 真实转移 | $\hat p_\phi$ | 学环境规则 |
| imagined rollout | $\hat p_\phi$ + 历史状态 | 虚拟样本 | 在模型里扩充经验 |
| 更新价值/策略 | 真实样本 + 虚拟样本 | 新的 $Q$ 或 $\pi$ | 提高决策能力 |

真实工程例子是机器人抓取。真机试错成本高，还可能损坏设备，所以常见做法不是在真机上傻试几百万次，而是先用一批真实交互学一个动力学模型，再用短视距 `MPC` 控制。`MPC` 就是“每次只往前规划有限几步，执行第一步，再重新规划”的方法，白话讲是“边走边重算”，这样能减少模型误差在长链条里累积。

误差累积可以用一个简单关系理解：如果单步模型误差是 $\epsilon$，那么 rollout 越长，总误差通常越容易增长，粗略上会随步数放大。因此很多系统更偏好短 horizon 规划。不是因为远期目标不重要，而是因为“短视距 + 高频重规划”往往比“一口气想很远”更稳。

---

## 代码实现

最小可运行实现不需要复杂神经网络，只要把 Dyna 的数据流闭环写清楚就够了。下面这个极简 Python 例子用表格模型保存转移，用 Q-learning 更新价值，再用模型生成 imagined update。它不是工业级算法，但能跑通“真实样本学模型 + 模型样本继续训练”的核心思路。

```python
from collections import defaultdict
import random

gamma = 0.9
alpha = 0.5

# 一个极简环境
# s0 --a--> s1, reward 0
# s0 --b--> terminal, reward 0
# s1 --a--> terminal, reward 1
# s1 --b--> terminal, reward 0
env = {
    ("s0", "a"): ("s1", 0.0),
    ("s0", "b"): ("terminal", 0.0),
    ("s1", "a"): ("terminal", 1.0),
    ("s1", "b"): ("terminal", 0.0),
}

actions = {
    "s0": ["a", "b"],
    "s1": ["a", "b"],
    "terminal": [],
}

Q = defaultdict(float)
model = {}  # 学到的环境模型: (s,a) -> (s', r)
memory = []

def step(state, action):
    return env[(state, action)]

def max_q(state):
    if state == "terminal":
        return 0.0
    return max(Q[(state, a)] for a in actions[state])

def q_update(s, a, r, s_next):
    target = r + gamma * max_q(s_next)
    Q[(s, a)] += alpha * (target - Q[(s, a)])

# 真实交互若干轮
for _ in range(30):
    state = "s0"
    while state != "terminal":
        # epsilon-greedy 的极简替代：随机探索偏多一点
        action = random.choice(actions[state])

        next_state, reward = step(state, action)

        # 1) 用真实样本更新 Q
        q_update(state, action, reward, next_state)

        # 2) 用真实样本更新模型
        model[(state, action)] = (next_state, reward)
        memory.append((state, action))

        # 3) Dyna imagined planning
        for _ in range(5):
            s_plan, a_plan = random.choice(memory)
            s_next_plan, r_plan = model[(s_plan, a_plan)]
            q_update(s_plan, a_plan, r_plan, s_next_plan)

        state = next_state

# s0 下动作 a 应该优于 b，因为 a 通向后续奖励
assert Q[("s0", "a")] > Q[("s0", "b")]
assert Q[("s1", "a")] > Q[("s1", "b")]

print("Q(s0,a) =", round(Q[("s0", "a")], 3))
print("Q(s0,b) =", round(Q[("s0", "b")], 3))
```

这段代码对应四个部件：

| 组件 | 职责 | 例子里的对应物 |
|---|---|---|
| `environment` | 提供真实转移 | `env` 与 `step()` |
| `model` | 预测下一状态和奖励 | `model[(s,a)] = (s', r)` |
| `planner` | 用模型生成 imagined updates | `for _ in range(5)` 那段 |
| `value/policy` | 吸收经验并泛化 | `Q` 与 `q_update()` |

如果要进一步往工程系统靠近，常见会加一个极简 `MPC` 框架：从当前状态出发，枚举若干动作序列，用模型预测每条序列的累计回报，只执行第一步，然后在新状态重新规划。它的思路是：

1. 收集真实样本 $(s,a,r,s')$
2. 更新模型 $\hat p_\phi$
3. 从当前状态采样候选动作序列
4. 用 $\hat p_\phi$ rollout 每条序列的回报
5. 选回报最高的序列，只执行第一步
6. 获得新真实样本后重新规划

这类实现的重点不在“代码复杂”，而在“数据流闭环清楚”：真实环境负责纠偏，模型负责扩充试错，规划器负责把模型转化为行动。

---

## 工程权衡与常见坑

工程上最核心的权衡是：**样本效率 vs 模型误差**。

如果 horizon 很短，模型误差不容易累积，系统更稳；但你看到的未来太近，可能学不到远期收益。如果 horizon 很长，理论上更会“做长远打算”，但只要模型有偏，错误就会在 rollout 中层层放大。

第二个权衡是：**算力 vs 效果**。规划越深、候选动作越多、imagined samples 比例越高，计算成本就越大。很多 `model-based RL` 系统不是输在思路上，而是输在“想得太多，算不过来”。

机器人控制是很典型的真实工程例子。假设抓取系统把“向左移动 5cm”误估成“高概率碰撞”，规划器就会长期回避左侧动作，导致策略异常保守。这里的问题不在策略优化器，而在模型把某块动作空间学坏了。

常见坑和规避方式可以直接列出来：

| 常见坑 | 典型后果 | 常见规避方式 |
|---|---|---|
| 模型误差累积 | 长 rollout 后完全跑偏 | 短 horizon、receding horizon、频繁重规划 |
| 不确定区域被过度利用 | 策略钻模型漏洞 | ensemble、uncertainty filtering、保守更新 |
| imagined data 比例过高 | 训练越来越脱离真实分布 | 限制 imagined ratio，混合真实样本 |
| 规划成本过高 | 决策太慢，线上不可用 | 限制搜索深度、压缩动作候选、减少重规划频率 |
| 把 AlphaZero 当成学模型 | 方法分类混乱，设计误判 | 区分“已知规则搜索”和“学习模型后规划” |

如果画成概念示意，误差大致像这样增长：

$$
\text{rollout length} \uparrow \Rightarrow \text{model bias accumulation} \uparrow
$$

它不是严格线性定律，但工程上足够有指导意义。因此很多系统宁愿“短一点、勤一点、保守一点”，也不愿“一次性规划太深”。

还有一个容易被忽略的问题是分布偏移。模型训练时见过的状态区域，和规划器搜索时会访问的状态区域，往往不是一回事。规划器尤其擅长钻进模型最不熟悉、但又看起来高回报的角落。这时如果没有不确定性估计，系统会产生“纸面最优、现实崩溃”的现象。

---

## 替代方案与适用边界

并不是所有强化学习任务都应该优先考虑 `model-based RL`。

如果环境很简单、真实样本便宜、延迟不敏感，`model-free` 往往更省事。因为你省掉了模型学习、规划器设计、不确定性估计这一整套复杂度。很多在线推荐、简单控制或可高频采样的模拟环境，直接上 `model-free` 反而更实际。

如果你已经有一个完全可信的模拟器或规则系统，那也不一定要再学一个模型。棋类程序就是标准例子：规则明明已知，直接搜索通常更合理。新手版理解就是：你已经有官方引擎，就没必要先训练一个“仿官方引擎”再去搜索。

这几类方法的适用边界可以总结为：

| 路线 | 优点 | 缺点 | 更适合什么场景 |
|---|---|---|---|
| `model-free` | 实现相对简单，训练目标直接 | 样本需求大 | 模拟便宜、环境稳定 |
| `model-based` | 样本效率高，可利用结构信息 | 实现复杂，对模型误差敏感 | 真机昂贵、安全要求高 |
| `search-only` | 规则可信时效果强 | 强依赖 simulator/规则 | 棋类、组合搜索问题 |

三条代表路线也要分清：

- `Dyna`：真实经验学模型，再用模型生成 imagined samples 训练价值或策略。
- `AlphaZero`：已知规则下的 `MCTS + 策略/价值网络`，重点是搜索，不是学习环境动力学。
- `MuZero`：学习潜在空间模型，再在潜在模型上搜索，是“学模型后规划”的代表路线。

可以用一个简化概念式表示：

$$
\text{Dyna}: \text{real data} \rightarrow \hat p \rightarrow \text{imagined updates}
$$

$$
\text{AlphaZero}: \text{known rules} \rightarrow \text{MCTS}
$$

$$
\text{MuZero}: \text{learned latent model} \rightarrow \text{search}
$$

因此，`model-based RL` 不是默认更高级，而是更适合某一类约束条件：真实试错贵、环境结构强、需要样本效率、并且你愿意为规划和模型维护支付额外复杂度。

---

## 参考资料

把参考文献理解成“本文关键判断背后的证据来源”，而不是凑链接。经典起源负责定义框架，综述负责给出全景，代表性系统负责说明不同路线各自怎么落地。

1. [Sutton, 1991, Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://papersdb.cs.ualberta.ca/~papersdb/view_publication.php?pub_id=500)  
   Dyna 思想来源。本文“真实经验 + 模型生成样本 + 再训练”的框架主要对应这条脉络。

2. [Moerland et al., 2023, Model-based Reinforcement Learning: A Survey](https://www.emerald.com/ftmal/article/16/1/1/1331296/Model-based-Reinforcement-Learning-A-Survey)  
   `model-based RL` 全景综述。本文关于边界划分、样本效率优势和模型偏差风险的整体判断可从这里建立系统视角。

3. [Silver et al., 2017, Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)  
   AlphaZero 代表论文。它说明了 `MCTS + 已知规则` 的搜索范式，适合用来和“学习模型后规划”做边界区分。

4. [Schrittwieser et al., 2020, Mastering Atari, Go, chess and shogi by planning with a learned model](https://www.nature.com/articles/s41586-020-03051-4)  
   MuZero 代表论文。它展示了“学习潜在模型后再搜索”的路线，是本文区分 AlphaZero 与 MuZero 的关键依据。

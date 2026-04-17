## 核心结论

经验回放（Experience Replay）是 DQN 中最关键的稳定器之一。它把智能体与环境交互得到的转移样本 $(s,a,r,s')$ 存入一个固定容量的缓冲区，再从缓冲区中随机抽取小批量样本训练 Q 网络。这里的“转移样本”就是“一次状态、一次动作、一次奖励、以及跳到的新状态”的记录。

它解决两个核心问题：

1. 连续交互数据高度相关。相邻两帧往往几乎一样，直接按时间顺序训练会让梯度方向来回摆动，导致优化不稳定。
2. 在线采样成本高。环境交互很贵，尤其是 Atari、机器人控制或真实业务系统，必须让同一条样本被重复利用。

DQN 的 TD 误差定义为：

$$
\delta = r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)
$$

这里的 TD 误差可以理解为“当前 Q 估计和目标值之间的差值”。差值越大，说明这条样本越值得重新学习。

优先级经验回放（Prioritized Experience Replay, PER）在普通经验回放上再加一步：更频繁地抽取 TD 误差大的样本。直观上，它相当于把“错题”多做几遍。但因为这种抽样不再均匀，会引入偏差，所以还需要重要性采样权重做修正。

新手版理解可以直接用一句话概括：先把游戏每一帧“存起来”，训练时像抽扑克牌一样“洗一洗”再学；如果某些帧总是学不好，就多抽几次，但训练时要给它们乘一个校正权重，避免网络过度偏向这些难样本。

---

## 问题定义与边界

DQN 的目标不是“记住最近一次奖励”，而是逼近动作价值函数 $Q(s,a)$。问题在于，环境按时间产生的数据天然不满足 SGD 喜欢的独立同分布假设。独立同分布可以简单理解为“样本之间不要太像，而且最好来自同一稳定分布”。

如果直接用最近几步数据更新，常见现象是：

- 连续状态过于相似，训练批次信息量低
- 某一小段轨迹中的偶然波动被反复放大
- 网络参数一变，目标分布也跟着变，训练更容易震荡

经验回放的边界主要在两个维度：容量与采样策略。

| 维度 | 太小会怎样 | 合适时会怎样 | 太大会怎样 |
|---|---|---|---|
| 缓冲区容量 | 样本高度相关，像反复看同一段视频 | 兼顾新数据与多样性 | 数据陈旧，学到过时策略分布 |
| 采样策略 | 只看最近样本，梯度方差大 | 随机抽样更接近 i.i.d. | 若过度偏向旧样本，更新滞后 |
| 优先级强度 | 学不到关键边界样本 | 重点学习高误差样本 | 过拟合少数“难样本” |

常见经验是：容量太小，例如低于 $10^4$，很容易出现相关性强的问题；容量太大，例如超过 $10^6$，则要警惕“旧策略产生的样本”大量占据训练数据。这里的“数据陈旧”就是样本来自很早之前的策略，它们和当前策略分布差异太大。

PER 进一步引入三个常见超参数：

| 参数 | 含义 | 典型作用 |
|---|---|---|
| $\alpha$ | 控制优先级强度 | $\alpha=0$ 时退化为均匀采样 |
| $\beta$ | 重要性采样修正强度 | 训练后期逐步增大到 1 |
| $\varepsilon$ | 防止优先级为 0 的小常数 | 保证每条样本都有被抽中的机会 |

其采样概率为：

$$
P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}
$$

如果 $\alpha$ 太大，训练会过度集中在少数高误差样本；如果 $\beta$ 长期太小，偏差修正不足，学习目标会被采样分布扭曲。

玩具例子：假设你在打一个简单平台跳跃游戏，最近连续 20 帧都只是“向右走”，如果每次训练都用这 20 帧，网络几乎只能学到“向右没出错”，却学不到“跳崖会死”“吃到奖励怎么变好”这类稀疏但关键的信息。把这些帧先存入缓冲区，再随机抽历史中的不同片段，训练信号会明显更稳定。

---

## 核心机制与推导

普通经验回放的机制可以拆成三步：

1. 把每次交互得到的 $(s,a,r,s')$ 写入缓冲区
2. 缓冲区满了以后，用新样本覆盖最旧样本
3. 每次训练时随机抽取一个 batch 更新网络

这本质上是一个环形队列。环形队列就是“写到尾部后从头继续覆盖”的固定数组结构。

### 从 TD 误差到优先级

DQN 的训练目标来自 Bellman 目标：

$$
y = r + \gamma \max_{a'}Q(s', a'; \theta^-)
$$

所以单条样本的误差是：

$$
\delta_i = y_i - Q(s_i, a_i; \theta)
$$

PER 不直接按时间随机抽，而是先定义优先级：

$$
p_i = |\delta_i| + \varepsilon
$$

这里的绝对值表示“只关心误差有多大，不关心高估还是低估”。

再把优先级转成概率：

$$
P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}
$$

于是，误差越大的样本越常被抽到。

### 为什么还要重要性采样权重

问题在于，网络原本想拟合的是经验池中的整体分布，但现在我们故意放大了高误差样本的出现频率，目标分布变了。为减少这种偏差，PER 引入重要性采样权重：

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

其中 $N$ 是缓冲区大小。实际实现里通常还会做归一化：

$$
\hat{w}_i = \frac{w_i}{\max_j w_j}
$$

这样 $\hat{w}_i \in (0,1]$，数值更稳定。

推导逻辑可以压缩成一条链：

1. 用 $\delta_i$ 衡量“这条样本还有多少没学会”
2. 用 $p_i = |\delta_i| + \varepsilon$ 变成非负优先级
3. 用 $P(i)$ 定义抽样分布
4. 用 $w_i$ 修正“抽样不均匀”带来的偏差

新手版理解：TD 误差大的样本就是“错题”；PER 让错题多出现；但如果错题出现太频繁，模型会误以为“所有题都像错题一样难”，所以要用权重把这种偏差拉回来。

### 一个最小数值例子

假设缓冲区里有 100 条样本，大多数样本的 $|\delta| \approx 0.5$，只有一条样本的 $|\delta|=12$。设 $\alpha=0.6$，$\varepsilon=0.01$。

高误差样本的优先级约为 $12.01$，普通样本约为 $0.51$。两者的相对采样强度接近：

$$
\left(\frac{12.01}{0.51}\right)^{0.6} \approx 3.8
$$

这意味着那条高误差样本会比普通样本更频繁地进入 batch。这样做通常能更快修正边界状态、稀有奖励状态或终止状态附近的估计误差。

真实工程例子：在 Atari 的 DQN/Double DQN 系统里，智能体可能长时间都只在“普通帧”上移动，而真正关键的是“快撞墙”“刚吃到奖励”“即将死亡”这些稀有状态。PER 会让这些高误差转移更频繁参与训练，因此网络更快学会边界决策。

---

## 代码实现

下面先给一个可运行的普通经验回放实现，再给出一个最小化的 PER 思路。

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        assert capacity > 0
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        assert 0 < batch_size <= self.size
        idxs = random.sample(range(self.size), batch_size)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return self.size


buf = ReplayBuffer(capacity=4)
buf.push("s0", 0, 1.0, "s1", False)
buf.push("s1", 1, 0.0, "s2", False)
buf.push("s2", 0, -1.0, "s3", True)
assert len(buf) == 3

batch = buf.sample(2)
assert len(batch) == 2
assert all(len(x) == 5 for x in batch)

# 环形覆盖测试
buf.push("s3", 1, 0.5, "s4", False)
buf.push("s4", 0, 0.2, "s5", False)
assert len(buf) == 4
```

这段代码体现了两个关键点：

- `push` 使用 `pos` 指针实现环形写入
- `sample` 从已有样本中均匀随机抽样

训练循环通常长这样：

1. 与环境交互，得到一条转移
2. `push` 进缓冲区
3. 当缓冲区样本数足够时，`sample(batch_size)`
4. 计算目标值与 TD 误差
5. 反向传播更新 Q 网络

PER 的最小伪代码如下：

```python
# priority: p_i = abs(td_error) + eps
# prob: P(i) = p_i ** alpha / sum_j p_j ** alpha
# weight: w_i = (1 / (N * P(i))) ** beta

beta = min(1.0, beta + beta_increment)

samples, idxs, probs = per_buffer.sample(batch_size)
weights = [(1.0 / (len(per_buffer) * p)) ** beta for p in probs]
max_w = max(weights)
weights = [w / max_w for w in weights]

# 用 weights 缩放每条样本的 loss
# loss = mean(weights[i] * td_error[i]^2)

# 训练后根据新 td_error 更新 priority
for idx, td_error in zip(idxs, new_td_errors):
    per_buffer.update(idx, abs(td_error) + eps)
```

工程里如果要高效实现 `sample` 与 `update`，通常不会每次都全量归一化，而是用 SumTree 或 segment tree。SumTree 可以理解为“一棵树里每个父节点存子树优先级之和”，这样采样和更新都能做到 $O(\log N)$，否则每次都线性扫描会太慢。

---

## 工程权衡与常见坑

经验回放本身不复杂，难的是把它调到“稳定但不过度复杂”。

| 常见问题 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 缓冲区过小 | loss 抖动大，策略反复横跳 | 连续样本相关性太强 | 增大容量，至少覆盖多个行为阶段 |
| 缓冲区过大 | 收敛慢，学习滞后 | 大量旧策略数据占比过高 | 用环形队列，必要时限制可见窗口 |
| PER 过强 | 模型盯住少量高误差样本 | $\alpha$ 太大或未做权重修正 | 先从较小 $\alpha$ 开始，并归一化权重 |
| 权重不稳定 | loss 爆炸或梯度异常 | $P(i)$ 太小导致 $w_i$ 过大 | 用 $\hat{w}_i$ 归一化到 $(0,1]$ |
| priority 为 0 | 某些样本永远抽不到 | 未加 $\varepsilon$ | 使用 $p_i=|\delta_i|+\varepsilon$ |
| 新样本学不到 | 刚加入的数据长期不被抽中 | 初始 priority 太低 | 新样本 priority 设为当前最大值 |

一个常见误区是“PER 一定比均匀回放好”。这并不总成立。原因很直接：如果 Q 网络还很不稳定，TD 误差大的样本未必真的“更重要”，也可能只是网络当前估计噪声更大。此时过早上 PER，等于把噪声当成信号。

一个实用策略是 warm-up。warm-up 就是“先用普通回放把网络训练到有基本判断力，再打开更激进的机制”。例如前几万步先用 uniform replay，待 Q 值分布初步稳定后再启用 PER。

真实工程例子：在 Atari 多环境训练中，很多实现会把新样本的优先级直接设为当前最大值。目的不是说它一定最重要，而是保证“新数据至少先被看见一次”。否则，如果新样本初始优先级过低，它可能长期抽不到，策略更新就会滞后于环境变化。

---

## 替代方案与适用边界

经验回放不是唯一方案，更不是所有场景都要上 PER。

| 方案 | 适用场景 | 优点 | 代价或边界 |
|---|---|---|---|
| Uniform Replay | 大多数标准 DQN 入门场景 | 简单、稳定、易调参 | 对关键稀有样本不够敏感 |
| PER | 稀有奖励、边界状态重要、训练预算有限 | 更快利用高价值样本 | 实现复杂，参数敏感，有采样偏差 |
| Reservoir Sampling | 长时分布变化大，希望长期保留代表性样本 | 长期多样性更强 | 不直接针对 TD 误差优化 |
| n-step + Uniform | 需要更强回报传播能力 | 奖励传播更快 | 回报方差更高 |
| Rainbow 风格组合 | 追求 SOTA 或较强基线 | 多机制互补 | 系统复杂度显著上升 |

如果资源有限，优先顺序通常是：

1. 先把 uniform replay 做对
2. 再考虑加入 n-step return
3. 最后再评估是否值得接入 PER

对于连续控制或分布式训练，还会出现其他边界：

- 在 MuJoCo 这类连续控制任务里，经验回放仍然常用，但不一定必须用 PER，是否有收益取决于奖励稀疏程度与算法结构。
- 在分布式系统里，多个 actor 同时产生数据，通常会把样本汇总到共享 replay server，再统一维护 priority。这类系统能提高吞吐，但实现复杂度明显增加。

新手版建议很简单：先把“缓冲区 + 随机采样 + 目标网络”这三件事跑稳，再谈优先级回放。因为对初级工程师来说，最常见的问题不是“缺少 PER”，而是基础 replay 已经写错，比如索引覆盖错误、done 标志处理错误、或者 batch 构造维度错误。

---

## 参考资料

1. GenRL 官方文档《Prioritized Deep Q-Networks》  
   适合看 PER 的完整训练流程、TD 误差、priority 更新以及 SumTree 类实现思路。  
   https://genrl.readthedocs.io/en/latest/usage/tutorials/Deep/Prioritized_DQN.html

2. Next Electronics 关于 DQN / Prioritized Experience Replay 的说明  
   适合看 $\alpha$、$\beta$、$\varepsilon$ 的作用，以及为什么要用重要性采样做偏差修正。  
   https://www.next.gr/ai/reinforcement-learning/prioritized-experience-replay-in-dqn

3. SEOFAI 的 DQN Replay Buffer 条目  
   适合新手理解 replay buffer 的基本定义、容量取舍和为什么它能打破时间相关性。  
   https://seofai.com/ai-glossary/dqn-replay-buffer/

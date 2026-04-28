## 核心结论

Prioritized Experience Replay，简称 PER，本质上不是“把旧经验随机再学一遍”，而是“按照样本的 TD 误差大小，重新分配它们被回放的概率”。

如果把经验回放看成复习错题，均匀采样只是平均翻旧账，PER 则会优先复习错得最多的题。这里的 TD 误差，白话讲，就是“当前值函数对这条转移到底看错了多少”。误差越大，说明模型在这条样本上还有更明显的学习空间。

PER 的收益和代价必须一起看：

| 方案 | 抽样方式 | 主要收益 | 主要代价 |
|---|---|---|---|
| 均匀回放 | 每条经验等概率 | 简单、稳定、无额外偏差 | 高信息量样本可能被淹没 |
| PER | 按 $|\delta|$ 加权抽样 | 更快聚焦高信息量样本 | 引入采样偏差，需重要性采样修正 |

结论先说清楚：PER 常常能提升数据利用率，尤其适合“少量关键样本很重要”的强化学习任务；但它不是免费加速器，必须配合重要性采样权重，否则训练目标会偏。

---

## 问题定义与边界

经验回放的原始目标有两个：一是打破相邻样本的时间相关性，二是提高数据复用率。它解决的是“在线交互得到的数据太相关、太宝贵”的问题，不是自动识别“真正重要样本”的万能筛子。

普通 replay 的问题在于：所有转移一视同仁。环境里大量“平凡样本”会不断进入 batch，例如机器人已经稳定抓住物体后的连续状态，或者 Atari 游戏中没有危险、没有奖励变化的普通帧。这些样本当然也有用，但它们的信息密度往往低于少数关键失败样本。

PER 想改进的是“回放效率”，不是“样本真值判断”。

| 维度 | 普通 replay 的缺点 | PER 想改进什么 | PER 不能保证什么 |
|---|---|---|---|
| 数据利用 | 高价值样本被平均化 | 让高误差样本更常被训练 | 不保证高误差就是真重要 |
| 学习速度 | 纠错慢 | 更快修正明显错误 | 不保证最终一定更优 |
| 稳定性 | 稳但可能慢 | 聚焦难样本 | 噪声、离群点会被放大 |

边界要讲清楚。PER 默认接受一个前提：TD 误差可以近似反映学习价值。这个前提并不总成立。

同样是高 TD 误差，可能有两种完全不同的来源：

1. 模型真的还没学会，这条样本有价值。
2. 这条样本本身噪声很大，或者奖励设计不稳定，所以误差大但不值得反复学。

玩具例子可以这样看。假设有两条转移都出现了大误差：

- 样本 A：智能体第一次遇到“掉进坑里”的状态，值函数明显估错。
- 样本 B：传感器偶发抖动，把正常状态误读成极端值，导致 TD 误差异常大。

对 A，优先回放通常有帮助；对 B，优先回放可能会反复学习假信号。PER 能提高效率，但不负责鉴别噪声。

---

## 核心机制与推导

PER 的核心链路是：先算 TD 误差，再把误差转成优先级，再把优先级转成抽样概率，最后用重要性采样权重修正偏差。

先定义 TD 误差。对于 DQN 风格的一步目标：

$$
\delta_i = r_i + \gamma \max_{a'} Q(s'_i,a') - Q(s_i, a_i)
$$

其中 $\delta_i$ 表示第 $i$ 条转移的 TD 误差，也就是“目标值”和“当前估计值”的差。

接着定义优先级：

$$
p_i = |\delta_i| + \epsilon
$$

这里 $\epsilon > 0$ 的作用很直接：防止优先级变成 0，避免某些样本永远抽不到。

再把优先级转成抽样概率：

$$
P(i)=\frac{p_i^\alpha}{\sum_j p_j^\alpha}
$$

$\alpha$ 控制偏向强度：

- $\alpha=0$：退化成均匀采样。
- $\alpha$ 越大：越偏向高误差样本。
- $\alpha=1$：按优先级比例直接抽。

问题在于，一旦不是均匀采样，梯度估计就有偏。于是需要重要性采样权重：

$$
w_i=(N\cdot P(i))^{-\beta}
$$

其中 $N$ 是回放池大小，$\beta$ 控制“修正偏差修到什么程度”。

- $\beta=0$：完全不修正。
- $\beta=1$：做完整的一阶修正。
- 工程里常把 $\beta$ 从较小值逐步增大到 1，这叫 anneal，白话讲就是“前期先多学，后期再更严格地纠偏”。

很多实现还会再做归一化：

$$
\tilde{w}_i = \frac{w_i}{\max_j w_j}
$$

这样数值更稳定。

看一个最小数值例子。设三条样本优先级为 $p=[1,2,7]$，取 $\alpha=1$，则：

$$
P = [0.1,0.2,0.7]
$$

若 $N=3,\beta=1$，则未归一化权重为：

$$
w=[(0.3)^{-1},(0.6)^{-1},(2.1)^{-1}] \approx [3.33,1.67,0.48]
$$

这组数字表达的直觉很重要：第 3 条样本会被更频繁抽到，但它单次更新的梯度权重反而更小。也就是“抽得更多，但每次别学得太重”，这样才能减少偏差。

| 样本 | $p_i$ | $P(i)$ | 未归一化 $w_i$ |
|---|---:|---:|---:|
| 1 | 1 | 0.1 | 3.33 |
| 2 | 2 | 0.2 | 1.67 |
| 3 | 7 | 0.7 | 0.48 |

参数作用可以再压缩成一张表：

| 参数 | 数学作用 | 直觉解释 | 常见风险 |
|---|---|---|---|
| $\epsilon$ | 防止 $p_i=0$ | 给所有样本保底出场机会 | 取 0 会让样本永久消失 |
| $\alpha$ | 控制采样倾斜度 | 多偏向难样本 | 太大时会放大噪声 |
| $\beta$ | 控制偏差修正强度 | 把训练分布往真实分布拉回去 | 太小会偏，太大早期可能不稳 |

真实工程例子里，Atari DQN 或机器人抓取都很典型。比如机械臂抓取任务，大部分转移只是“手臂平稳移动”，而少数“抓空”“滑落”“接近碰撞”的转移往往产生更大的 TD 误差。PER 会更频繁地抽这些少数关键样本，因此通常比均匀 replay 更快修正错误估计。

---

## 代码实现

实现 PER 的核心不是采样循环本身，而是如何高效维护 priority。常见做法是 Sum Tree，或者更一般的 Segment Tree。它本质上是一棵二叉树，叶子存样本优先级，内部节点存子树优先级之和。

一个最简树形示意可以写成：

```text
                [sum=10]
               /        \
          [sum=3]      [sum=7]
           /   \        /   \
         [1]  [2]     [7]  [0]
```

如果总和是 10，就在区间 $[0,10)$ 上采一个随机数。例如随机到 8.4，就沿着前缀和往下走：

- 左子树和是 3，8.4 > 3，所以去右子树
- 右子树左叶子是 7，对应区间覆盖到 10
- 最终落到优先级为 7 的样本

这就是“按前缀和定位叶子节点”。树高是 $\log N$，所以采样和更新都能做到 $O(\log n)$。

流程拆开看最清楚：

```text
sample(batch_size)
  -> 从 sum tree 按前缀和采样 indices
  -> 取出 transitions
  -> 根据 P(i) 计算 IS weights
  -> 返回 batch, indices, weights

learn(batch)
  -> 计算 TD error
  -> 用 weights 缩放 loss
  -> 反向传播更新网络

update_priorities(indices, td_errors)
  -> priority = |td_error| + eps
  -> 回写 sum tree
```

复杂度对比如下：

| 操作 | 朴素线性实现 | Sum Tree / Segment Tree |
|---|---:|---:|
| 单次采样 | $O(n)$ | $O(\log n)$ |
| 单次 priority 更新 | $O(1)$ 或 $O(n)$ 视实现而定 | $O(\log n)$ |
| batch 采样 $k$ 次 | $O(kn)$ | $O(k\log n)$ |

下面给一个可运行的最小 Python 实现。它不是完整 replay buffer，但足够展示概率、权重和更新逻辑。

```python
import random
from math import isclose

class SumTree:
    def __init__(self, capacity):
        cap = 1
        while cap < capacity:
            cap *= 2
        self.capacity = cap
        self.tree = [0.0] * (2 * cap)

    def update(self, idx, value):
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def total(self):
        return self.tree[1]

    def get(self, idx):
        return self.tree[idx + self.capacity]

    def find_prefixsum_idx(self, mass):
        i = 1
        while i < self.capacity:
            left = 2 * i
            if mass <= self.tree[left]:
                i = left
            else:
                mass -= self.tree[left]
                i = left + 1
        return i - self.capacity

class PrioritizedSampler:
    def __init__(self, priorities, alpha=1.0, beta=1.0, eps=1e-8):
        self.n = len(priorities)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tree = SumTree(self.n)
        for i, p in enumerate(priorities):
            self.tree.update(i, (p + eps) ** alpha)

    def probs(self):
        total = self.tree.total()
        return [self.tree.get(i) / total for i in range(self.n)]

    def weights(self):
        probs = self.probs()
        raw = [(self.n * p) ** (-self.beta) for p in probs]
        m = max(raw)
        return [w / m for w in raw]

    def sample_one(self, rng=random.random):
        mass = rng() * self.tree.total()
        idx = self.tree.find_prefixsum_idx(mass)
        return idx, self.probs()[idx], self.weights()[idx]

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            p = abs(td) + self.eps
            self.tree.update(idx, p ** self.alpha)

sampler = PrioritizedSampler([1, 2, 7], alpha=1.0, beta=1.0)
probs = sampler.probs()
weights = sampler.weights()

assert all(isclose(a, b, rel_tol=1e-6) for a, b in zip(probs, [0.1, 0.2, 0.7]))
assert weights[2] < weights[1] < weights[0]  # 抽得越频繁，单次权重越小

sampler.update_priorities([0], [10.0])
new_probs = sampler.probs()
assert new_probs[0] > probs[0]
```

如果把它放回完整 DQN 训练环，接口一般就是：

```python
# pseudo code
batch, indices, weights = replay.sample(batch_size)
td_error = compute_td_error(batch, q_net, target_net)
loss = (weights * td_error.pow(2)).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
replay.update_priorities(indices, td_error.detach().abs())
```

---

## 工程权衡与常见坑

第一条原则：priority 不等于真实重要性。$|\delta|$ 大，只能说明“当前估计偏差大”，不能直接推出“这条样本就值得反复学”。

最常见的风险，是偶发极端样本被反复抽到。例如某次环境 reset 异常、奖励爆了一次、观测出现脏数据，这类样本会带来巨大 TD 误差。如果不做限制，它会被频繁采样，把模型拉向错误方向。

常见规避手段有三类：

- 对 priority 做 clip，限制上界。
- 控制 $\alpha$，不要过分倾斜。
- 每次训练后及时更新 priority，避免“老误差”长期支配采样。

坑点可以系统化地看：

| 坑点 | 现象 | 原因 | 常见处理 |
|---|---|---|---|
| priority 过大 | 极端样本被反复抽到 | 噪声或离群点被放大 | clip priority，限制 $\alpha$ |
| $\beta$ 设太小 | 学得快但目标偏 | 重要性采样修正不足 | 逐步 anneal 到 1 |
| priority 不更新 | 采样分布陈旧 | 样本“难度”已变化 | 每轮学习后回写 TD 误差 |
| $\epsilon$ 取 0 | 某些样本永远不再出现 | 优先级可能变成 0 | 保持小正数 |
| 树容量不补齐 | 下标或查询逻辑复杂 | 完全二叉树假设不成立 | 容量向上补到 2 的幂 |

还有两个新手容易忽略的点。

一是 batch 内权重通常要归一化，不然数值尺度会跳得很厉害。二是多步回报、序列回放时，priority 聚合规则要统一。比如一段序列里每步都有 TD 误差，你到底取 `max`、`mean`，还是最后一步误差？这不是细节问题，而是训练目标的一部分。

---

## 替代方案与适用边界

PER 不是唯一方案，也不是所有项目都值得上。

如果任务本身数据量不大、训练噪声高、实现成本敏感，均匀回放往往更稳。它虽然慢一点，但行为简单，可解释性更强，调参负担也更低。

还存在一些“按规则采样”的替代思路，例如按 reward 大小、按 episode 成败、按最近时间窗口、按序列边界做采样。它们未必像 PER 那样有完整的偏差修正公式，但有时更贴合任务结构。

| 方案 | 适用场景 | 优点 | 代价 |
|---|---|---|---|
| 均匀回放 | 基线训练、噪声高、小规模任务 | 最稳、最简单 | 关键样本利用不足 |
| PER | 稀有关键样本少、TD 误差有信息量 | 学习更快、数据效率高 | 需维护树结构和 IS 权重 |
| 其他规则采样 | 任务结构明确，如成败分明或序列强依赖 | 可结合业务先验 | 偏差控制通常更弱 |

边界条件可以用两个真实场景对比说明。

场景 A：稀疏奖励机器人抓取。绝大多数时间没有奖励变化，少数“接近成功”“抓空”“滑落”的样本特别关键。这里 TD 误差往往确实能反映学习进展，PER 通常很有帮助。

场景 B：高噪声环境中的不稳定奖励。比如观测抖动明显、奖励函数本身波动大，很多高 TD 误差只是噪声放大。这里 PER 可能被误导，不如均匀回放或更保守的采样策略稳定。

所以最终判断标准不是“PER 高级不高级”，而是这句话：你的任务里，TD 误差是否真的能近似代表样本的学习价值。如果答案偏是，PER 值得上；如果答案偏否，它可能只是把噪声学得更勤奋。

---

## 参考资料

如果只看一篇，先看原论文；如果要落地实现，优先看 TorchRL 和 RLlib 的接口与源码，它们把公式、数据结构和工程细节都串起来了。

1. [Prioritized Experience Replay, Schaul et al.](https://arxiv.org/abs/1511.05952)
2. [TorchRL PrioritizedSampler 官方文档](https://docs.pytorch.org/rl/main/reference/generated/torchrl.data.replay_buffers.PrioritizedSampler.html)
3. [Ray RLlib PrioritizedReplayBuffer 源码文档](https://docs.ray.io/en/latest/_modules/ray/rllib/utils/replay_buffers/prioritized_replay_buffer.html)
4. [AgileRL Segment Trees 文档](https://docs.agilerl.com/en/latest/api/components/segment_tree.html)

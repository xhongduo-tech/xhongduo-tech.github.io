## 核心结论

Noisy Nets 的核心不是“把随机性加大”，而是把探索的位置从动作空间移到参数空间。动作空间随机，指的是像 `ε-greedy` 那样，先按固定概率决定“这一步要不要乱选动作”；参数空间随机，指的是先给网络参数加入可学习噪声，再让网络像平常一样根据当前状态输出动作价值。

这两种方法的差别很大。`ε-greedy` 的随机发生在动作选择最后一步，所以它对所有状态使用同一种随机规则；Noisy Nets 的随机发生在网络内部，所以同样的噪声会系统性地影响一段时间内的决策方向，探索更连贯，也更容易和状态特征、Q 值结构绑定。

可以把两者做一个最短对比：

| 方法 | 随机发生位置 | 是否需要手工调探索率 | 探索是否与状态相关 | 典型问题 |
| --- | --- | --- | --- | --- |
| `ε-greedy` | 动作空间 | 需要，通常要设 `ε` 衰减 | 弱，规则基本全局一致 | 容易出现无意义抖动 |
| Noisy Nets | 参数空间 | 通常不需要单独调 `ε` 曲线 | 强，探索随网络输出结构变化 | 实现和评估切换更容易写错 |
| 固定贪心 | 不探索 | 不需要 | 无 | 很容易卡在局部最优 |

一个直观说法是：`ε-greedy` 像“正常思考，但偶尔乱按按钮”；Noisy Nets 像“先把模型内部判断轻微扰动，再按扰动后的判断稳定行动”。后者不是每一步独立乱跳，而是在一段时间内沿着某个偏移方向去试。

---

## 问题定义与边界

强化学习里的探索问题很直接：智能体必须在“利用当前已知最好动作”和“尝试未知动作”之间平衡。如果只利用，它学不到更好的策略；如果只探索，训练又会非常低效。

`ε-greedy` 是最常见的基线方法。它的规则简单：以概率 $1-\epsilon$ 选择当前 Q 值最大的动作，以概率 $\epsilon$ 随机选动作。这个方法容易实现，但它有两个明显局限。

第一，它的随机是“动作级别”的。模型本身没有变化，只是在最后一步被强行插入一次随机决策。第二，它的随机通常与状态无关。无论当前状态是“快找到奖励了”还是“完全没信息”，只要 `ε` 一样，随机机制就一样。

在稀疏奖励、长时序决策、离散动作很多的任务里，这种做法经常不够自然。比如 Atari 游戏中，智能体有时需要连续做对多个动作才会接近奖励，而 `ε-greedy` 的随机插入可能会让动作序列频繁断裂，表现成“抖一下、停一下、再抖一下”。

Noisy Nets 解决的是这个边界内的问题：它替代的是“动作空间随机探索”这一层，不是整个强化学习训练难题的总解。

| 问题 | Noisy Nets 是否直接解决 | 说明 |
| --- | --- | --- |
| 如何更自然地探索未知动作 | 是 | 通过参数噪声形成结构化探索 |
| 稀疏奖励下的探索效率 | 部分改善 | 有帮助，但不保证一定找到奖励 |
| 奖励函数设计错误 | 否 | 奖励错了，探索再好也会学偏 |
| 信用分配困难 | 否 | 长回报如何归因，仍是算法本身的问题 |
| 环境强非平稳 | 否 | 噪声不能替代稳定训练机制 |
| 安全约束严格的部署推理 | 通常不适合保留噪声 | 评估或上线通常要关噪声 |

适用边界也要说清楚：

| 更适合 | 较不适合 |
| --- | --- |
| DQN、Double DQN、Dueling DQN、Rainbow | 强安全约束的在线控制 |
| 离散动作空间 | 明确要求推理完全确定性的场景 |
| 稀疏奖励任务 | 噪声会直接触发危险动作的任务 |
| 希望减少 `ε` 调参负担的场景 | 主要依赖其他探索机制的算法体系 |

真实工程例子是 Atari DQN。把末端若干 `Linear` 层替换成 `NoisyLinear`，常见做法是同时去掉或显著弱化 `ε-greedy`。这样探索不再是“固定概率乱选”，而是“价值函数在当前参数扰动下偏向不同动作”。

---

## 核心机制与推导

Noisy Nets 最核心的一步，是把普通线性层的固定参数，改成“均值 + 噪声尺度 × 随机噪声”的形式：

$$
y=(\mu_w+\sigma_w\odot\epsilon_w)x+\mu_b+\sigma_b\odot\epsilon_b
$$

这里几个术语先用白话解释：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $\mu_w,\mu_b$ | 可学习均值参数 | 不带噪声时网络想学到的“中心值” |
| $\sigma_w,\sigma_b$ | 可学习噪声尺度 | 网络自己学“这个位置该不该保留随机性” |
| $\epsilon_w,\epsilon_b$ | 随机噪声 | 每次采样出来的扰动值 |
| $\odot$ | 按元素乘法 | 每个参数位置各自乘自己的噪声 |

普通线性层是 $y=Wx+b$。Noisy 层的意思不是把 $W$ 随机乱改，而是把 $W$ 写成：

$$
W=\mu_w+\sigma_w\odot\epsilon_w,\quad b=\mu_b+\sigma_b\odot\epsilon_b
$$

这样同一个状态 $x$，在两次前向传播里，哪怕输入完全一样，也可能得到不同的 Q 值排序，因此产生不同动作。探索来自网络本身的参数采样，而不是最后一步额外抛硬币。

先看一个玩具例子。设某个状态输入后，只比较两个动作 A 和 B。

第一次采样：
- 动作 A：$\mu=1.0,\sigma=0.2,\epsilon=+1$，得到 $Q_A=1.2$
- 动作 B：$\mu=0.9,\sigma=0.3,\epsilon=-1$，得到 $Q_B=0.6$

所以选 A。

第二次采样：
- 动作 A：$Q_A=1.0+0.2\times(-1)=0.8$
- 动作 B：$Q_B=0.9+0.3\times(+1)=1.2$

所以改选 B。

重点在于：状态没变，动作也不是手工随机切换，而是“参数采样后网络自然给出了不同偏好”。

为了降低大网络里的采样成本，论文常用 factorized Gaussian noise。它不用为每个权重单独采一个独立高斯，而是用输入维度和输出维度两个向量组合出矩阵噪声：

$$
\epsilon_w=f(\epsilon^p)\otimes f(\epsilon^q),\quad \epsilon_b=f(\epsilon^q),\quad f(u)=\operatorname{sgn}(u)\sqrt{|u|}
$$

这里：
- $\otimes$ 是外积，意思是把两个向量组合成一个矩阵。
- $\operatorname{sgn}(u)$ 是符号函数，只保留正负号。

为什么这样做有用？因为如果输入维度是 $p$、输出维度是 $q$，全独立噪声要采样 $p\times q$ 个随机数；factorized 只需要采样 $p+q$ 个，成本更低，同时仍然保留足够的随机结构。

训练和评估的处理也不同：

| 阶段 | 是否采样噪声 | 参数是否更新 | 常见目的 |
| --- | --- | --- | --- |
| 训练 | 是 | $\mu,\sigma$ 都更新 | 学会何时保留探索 |
| 评估 | 否，通常只用 $\mu$ | 不更新 | 测真实策略水平 |

这也解释了 Noisy Nets 和 `ε-greedy` 的本质差异。`ε-greedy` 的随机规则是外加的，Noisy Nets 的随机规则是网络自己学出来的。某些参数位置如果发现噪声长期有益，$\sigma$ 会保留较大；如果噪声妨碍收敛，$\sigma$ 会被学小。探索不再完全依赖人工调度，而是部分交给优化过程。

---

## 代码实现

工程上通常不是重写整套 DQN，而是把 `nn.Linear` 换成 `NoisyLinear`。下面给一个可运行的最小 PyTorch 实现。它展示三个关键点：可学习的 `mu/sigma`、factorized noise、训练和评估切换。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_w = self.sigma_init / math.sqrt(self.in_features)
        sigma_b = self.sigma_init / math.sqrt(self.out_features)
        self.weight_sigma.data.fill_(sigma_w)
        self.bias_sigma.data.fill_(sigma_b)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# 最小断言：训练态同一输入在重采样后应可能改变输出；评估态应稳定
torch.manual_seed(0)
layer = NoisyLinear(4, 2)
x = torch.ones(1, 4)

layer.train()
y1 = layer(x)
layer.reset_noise()
y2 = layer(x)
assert not torch.allclose(y1, y2), "训练态重采样后输出应变化"

layer.eval()
y3 = layer(x)
layer.reset_noise()
y4 = layer(x)
assert torch.allclose(y3, y4), "评估态关闭噪声后输出应稳定"
```

在 DQN 中的替换方式通常很直接：

```python
import torch.nn as nn
import torch.nn.functional as F

class NoisyDQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, action_dim)

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
```

典型训练循环里，每次参数更新前后都可以重采样噪声；而评估前要切到 `eval`：

```python
model.train()
model.reset_noise()
q_values = model(states)

model.eval()
with torch.no_grad():
    greedy_q = model(states)   # 此时只用 mu，不再带噪声
```

实现时最容易混淆的是这三件事：

| 场景 | 是否重采样 | 是否保留噪声参与前向 | `sigma` 是否有梯度 |
| --- | --- | --- | --- |
| 训练 | 是 | 是 | 是 |
| 验证/评估 | 可不重采样 | 否，通常只用 `mu` | 否 |
| 部署推理 | 否 | 否 | 否 |

---

## 工程权衡与常见坑

Noisy Nets 的收益通常来自更一致的探索，但前提是实现细节不能错位。它不是“换层就自动涨分”的魔法模块，很多失败案例其实是训练和评估逻辑没分清。

先看 `sigma` 初始化。$\sigma$ 太大，表示参数扰动幅度太大，早期策略会非常乱；$\sigma$ 太小，又几乎等于没探索。常见初始化会让它与输入维度成反比缩放，例如：

$$
\sigma_w \approx \frac{\sigma_0}{\sqrt{d_{in}}}
$$

其中 $\sigma_0$ 常取较小常数，如 $0.4$ 到 $0.5$ 一类的起始量级。直觉很简单：输入维度越大，单个权重不该带过猛的初始扰动，否则总噪声叠加会失控。

真实工程里最常见的坑如下：

| 坑点 | 症状 | 原因 | 修复方式 |
| --- | --- | --- | --- |
| 评估时没关噪声 | 分数波动大、复现差 | 测试阶段还在采样随机参数 | `model.eval()`，前向只用 `mu` |
| `sigma` 初始化过大 | 前期动作极乱，收敛慢 | 参数噪声压过了有效价值信号 | 降低 `sigma_init` |
| `sigma` 初始化过小 | 和普通 DQN 差不多 | 几乎没有探索增益 | 检查初始化和梯度更新 |
| 同时强保留 `ε-greedy` | 指标变化难解释 | 两种探索来源混杂 | 先单独验证 Noisy Nets |
| 忘记周期性重采样 | 探索不连续或近乎失效 | 噪声只采样一次，行为固化 | 明确 `reset_noise()` 调用点 |
| 训练态/评估态代码不一致 | 线上线下表现对不上 | 前向逻辑分支不统一 | 在层内部显式区分 `self.training` |

尤其要注意：Noisy Nets 追求的是“带结构的探索”，不是“让每一步都尽量不同”。如果你在 batch 训练时每个微小步骤都乱采样，但在线交互时却长时间不换噪声，训练分布和执行分布就会偏离，结果很难复现。

还有一个常见误判：有人发现评估分数上下抖动，就得出“这个方法不稳定”。很多时候问题不是方法不稳定，而是你在评估随机策略本身。对价值型方法来说，测试目标一般是“这套参数学到了什么策略”，所以应关闭噪声，只保留均值参数。

---

## 替代方案与适用边界

Noisy Nets 不是唯一的探索方法，它只是把探索绑定到参数不确定性的一种做法。和它最容易混淆的，是 `ε-greedy`、Boltzmann exploration、熵正则以及其他参数噪声方法。

| 方法 | 核心机制 | 更适合什么 | 主要代价 |
| --- | --- | --- | --- |
| `ε-greedy` | 以概率 `ε` 随机选动作 | 基线、实现极简、离散动作 | 需要手工调 `ε`，探索较粗糙 |
| Noisy Nets | 在参数空间加可学习噪声 | 值函数方法、离散动作、稀疏奖励 | 实现更复杂，评估切换要严谨 |
| Boltzmann exploration | 按 softmax 概率采样动作 | Q 值相对大小有意义的场景 | 温度参数敏感 |
| 熵正则 | 在目标里鼓励策略分散 | 策略梯度、Actor-Critic | 需调熵系数，未必适配值函数方法 |
| UCB/置信上界方法 | 显式利用不确定性奖励探索 | Bandit 或可估计置信界场景 | 额外统计或建模成本 |

还可以从适用边界再看一遍：

| 维度 | Noisy Nets 特征 |
| --- | --- |
| 任务类型 | 更偏值函数方法，如 DQN 系列 |
| 动作空间 | 更适合离散动作 |
| 是否易复现 | 中等，取决于噪声管理是否规范 |
| 额外超参 | 有，但通常少于手工设计完整 `ε` 曲线 |
| 推理是否确定 | 可以，前提是评估时关闭噪声 |

和熵正则的差异尤其要说清楚。熵正则的目标是“让策略分布别过早塌缩”，它直接作用在动作分布上；Noisy Nets 的目标是“让模型带着参数不确定性去决策”，它作用在网络表示和价值估计内部。两者都能增加探索，但机制不同，适用算法族也不同。

因此，一个实用判断标准是：
- 如果你在做 DQN、Double DQN、Rainbow 一类离散动作值函数方法，并且对 `ε` 衰减调参很烦，Noisy Nets 值得优先尝试。
- 如果你在做连续动作控制，或算法核心本来就是策略梯度加熵正则，那么 Noisy Nets 不一定是第一选择。
- 如果部署要求动作绝对稳定、不可带随机性，那么训练可以用 Noisy Nets，推理必须关噪声；如果连训练中的随机探索都受强安全约束，就要慎用。

---

## 参考资料

阅读顺序建议：

| 顺序 | 资料类型 | 作用 |
| --- | --- | --- |
| 1 | 原论文 | 确认方法定义、公式、实验结论 |
| 2 | 官方介绍页 | 快速把握方法定位 |
| 3 | 最小实现源码 | 理解 `NoisyLinear` 具体怎么写 |

下面这些资料里，论文和 Google Research 页面属于一手定义来源；代码仓库主要用于理解实现细节。文中“评估时关闭噪声”“先不要和强 `ε-greedy` 混用”等内容，属于工程实践中的常见做法，不是论文里唯一允许的写法。

1. [Noisy Networks for Exploration - OpenReview](https://openreview.net/forum?id=zgUEeXkag9)
2. [Noisy Networks for Exploration - Google Research](https://research.google/pubs/noisy-networks-for-exploration/)
3. [thomashirtz/noisy-networks - GitHub](https://github.com/thomashirtz/noisy-networks)
4. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

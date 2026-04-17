## 核心结论

Noisy Nets 的核心做法，是把线性层的权重和偏置从固定参数，改成“平均值 + 可学习噪声”的形式：

$$
W=\mu_W+\sigma_W\odot \epsilon_W,\quad b=\mu_b+\sigma_b\odot \epsilon_b
$$

这里的 $\mu$ 可以理解为“正常权重”，$\sigma$ 可以理解为“抖动幅度”，$\epsilon$ 是每次前向传播临时采样的随机噪声。这样一来，网络不再依赖手工设计的 $\epsilon$-greedy 去随机选动作，而是直接让整个值函数带着结构化扰动去探索。

对白话一点的理解是：每次前向时，模型都会在当前参数附近“轻微摇摆”一下；如果任务还没学明白，梯度会推动 $\sigma$ 变大，让探索更强；如果任务已经比较确定，$\sigma$ 会被学小，探索自然减弱。于是它把“探索强度调度”从人工规则，变成了可学习参数。

Noisy Nets 相比 $\epsilon$-greedy 的优势，不是“随机性更多”，而是“随机性更有结构”。$\epsilon$-greedy 只是在动作输出端偶尔乱选一下，而 Noisy Nets 会同时改变整张值函数的形状，因此更容易形成持续几个时间步的一致探索行为。这类探索通常被称为深度探索，意思是探索不是单步乱动，而是沿着某种新策略持续试下去。

| 维度 | $\epsilon$-greedy | Noisy Nets |
|---|---|---|
| 随机化位置 | 动作选择阶段 | 网络参数阶段 |
| 是否手工调度 | 通常要手工退火 $\epsilon$ | 通常不需要显式退火 |
| 扰动范围 | 只影响当前动作 | 影响整个值函数 |
| 探索持续性 | 往往是单步随机 | 更容易形成连贯策略 |
| 训练稳定性 | 相对直观 | 方差可能更大 |

---

## 问题定义与边界

强化学习里的探索，指的是“为了发现更优行为，愿意尝试当前看起来不一定最优的动作”。在 DQN 一类值函数方法中，最常见的做法是 $\epsilon$-greedy：大多数时候选当前 $Q$ 值最大的动作，小概率随机选一个动作。

这个办法简单，但有两个常见问题。

第一，探索是“动作级”的，不是“策略级”的。也就是说，智能体可能只是在某一帧突然乱按一下键，下一帧又恢复贪心选择。这样的随机动作常常不够连贯，尤其在需要连续多步试错的场景里效果有限。

第二，$\epsilon$ 的调度通常要人工设定。比如从 1.0 退火到 0.1，持续多少步、最终退到多少，都要人来调。调小了可能探索不足，调大了又可能长期收敛很慢。

Noisy Nets 讨论的是另一条路线：不在动作空间直接加随机性，而是在参数空间加随机性。参数空间噪声，指的是对网络参数本身做随机扰动，让网络输出整体改变。它适用于可以把核心决策模块写成线性层或仿射层的函数逼近器，最典型的就是 DQN、Double DQN、Dueling DQN 这类值方法。

原始线性层是：

$$
y=Wx+b
$$

Noisy Nets 把它替换成：

$$
y=(\mu_W+\sigma_W\odot\epsilon_W)x+(\mu_b+\sigma_b\odot\epsilon_b)
$$

边界也要说清楚。Noisy Nets 不是“所有强化学习都更好”的通用结论。它最自然的落点，是离散动作空间下的值函数近似，尤其是 DQN 系列。到了连续控制，探索往往和策略分布、熵正则、动作噪声等机制耦合更深，Noisy Nets 可能仍然有用，但通常不是唯一主角。

玩具例子可以这样看。假设一个小迷宫里，向右走两步才能拿到奖励。$\epsilon$-greedy 可能只在第一步随机向右，第二步又恢复贪心而折返，于是很难连续两步试对。Noisy Nets 则可能因为整张 $Q$ 函数被扰动，在一段时间内都更偏好“向右”这条策略，因此更有机会真正走到奖励点。

真实工程里，Atari 就是典型例子。很多游戏的高分策略依赖连续动作序列，而不是单次随机按键。Noisy Nets 替代 $\epsilon$-greedy 后，往往能更稳定地走出新的行为模式，尤其在 Dueling 和 Double DQN 组合下更常见。

---

## 核心机制与推导

Noisy Nets 的关键，不只是“给参数加噪声”，而是“让噪声幅度也参与学习”。

### 1. 参数化方式

对每个线性层，维护两组参数：

- $\mu_W,\mu_b$：均值参数，表示这层在不考虑噪声时的中心值
- $\sigma_W,\sigma_b$：尺度参数，表示这层允许多大程度的随机扰动

前向传播时再采样噪声：

$$
W=\mu_W+\sigma_W\odot\epsilon_W,\quad b=\mu_b+\sigma_b\odot\epsilon_b
$$

其中 $\odot$ 表示逐元素乘法，意思是每个参数位置都有自己的噪声振幅。

### 2. 为什么它可训练

目标不是优化某一次固定噪声下的损失，而是优化“对噪声取期望后的损失”：

$$
\bar{L}(\theta)=\mathbb{E}_{\epsilon}[L(\mu+\sigma\odot\epsilon)]
$$

这里 $\theta=\{\mu,\sigma\}$。直观上，训练在问一个问题：如果网络每次都会轻微抖动，那么平均而言，什么样的 $\mu$ 和 $\sigma$ 最好？

实际训练时，我们通常每次前向采样一份 $\epsilon$，用它构造本次网络，再做常规反向传播。这样梯度会同时流向 $\mu$ 和 $\sigma$。如果增大某个 $\sigma$ 能带来更好的长期回报，梯度就会把它推大；反之就会推小。

### 3. 因子化高斯噪声

如果直接给每个权重元素单独采样一个高斯噪声，成本会比较高。论文常用的做法是因子化高斯噪声。所谓因子化，指的是把二维权重噪声分解成输入维度和输出维度两个一维噪声向量的组合。

设输入维度是 $p$，输出维度是 $q$。先采样：

$$
\epsilon^i \in \mathbb{R}^p,\quad \epsilon^o \in \mathbb{R}^q
$$

再经过变换函数：

$$
f(x)=\operatorname{sign}(x)\sqrt{|x|}
$$

最后构造：

$$
\epsilon_W[j,i]=f(\epsilon^o_j)f(\epsilon^i_i),\quad \epsilon_b[j]=f(\epsilon^o_j)
$$

这样做的好处是，原本需要为 $q\times p$ 个权重单独采样噪声，现在只要采样 $p+q$ 个标量，再外积组合出来，采样成本更低，同时仍保留一定的结构化随机性。

### 4. 单层数值玩具例子

假设某个标量线性层只有一个权重，没有偏置：

- 输入 $x=2$
- $\mu_W=1.0$
- $\sigma_W=0.2$
- 一次采样得到 $\epsilon_W=0.5$

那么本次前向的实际权重是：

$$
w=1.0+0.2\times 0.5=1.1
$$

输出是：

$$
y=1.1\times 2=2.2
$$

如果训练后期学到 $\sigma_W=0.05$，同样的噪声只会产生：

$$
w=1.0+0.05\times 0.5=1.025
$$

这说明探索强度自然减弱了。它不是靠人工把 $\epsilon$ 从 1.0 调到 0.1，而是网络自己学会“这里还需要多大抖动”。

### 5. 梯度更新路径

| 步骤 | 做什么 | 作用 |
|---|---|---|
| 1 | 采样 $\epsilon^i,\epsilon^o$ | 生成本次结构化噪声 |
| 2 | 构造 $\epsilon_W,\epsilon_b$ | 得到本次扰动后的层参数 |
| 3 | 前向计算 $Q(s,a)$ | 得到值函数输出 |
| 4 | 计算 TD 损失 | 衡量当前值估计误差 |
| 5 | 反向传播到 $\mu,\sigma$ | 同时学习中心值和探索强度 |

这里有一个理解重点：Noisy Nets 不是“给训练加点随机扰动”这么简单，它是在把“探索策略”也参数化，然后交给梯度一起优化。

---

## 代码实现

下面给一个可运行的 Python 版本，重点展示 Noisy Linear 层的核心结构。为了便于阅读，这里用 PyTorch 风格实现。`mu` 是平均参数，`rho` 经过 `softplus` 后得到非负的 `sigma`，避免直接学习出负的噪声幅度。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3.0)

    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        sigma_w = F.softplus(self.weight_rho)
        sigma_b = F.softplus(self.bias_rho)

        eps_in = self.scale_noise(self.in_features).to(x.device)
        eps_out = self.scale_noise(self.out_features).to(x.device)

        eps_w = torch.outer(eps_out, eps_in)
        eps_b = eps_out

        weight = self.weight_mu + sigma_w * eps_w
        bias = self.bias_mu + sigma_b * eps_b

        return F.linear(x, weight, bias)

layer = NoisyLinear(3, 2)
inp = torch.tensor([[1.0, 2.0, 3.0]])
out1 = layer(inp)
out2 = layer(inp)

assert out1.shape == (1, 2)
assert out2.shape == (1, 2)
assert not torch.allclose(out1, out2), "两次前向应因重采样而不同"
assert torch.all(F.softplus(layer.weight_rho) >= 0)
```

如果把它接到 DQN 里，通常会替换最后一层或最后几层全连接层。新手可以先记住下面这句话：普通线性层输出是“固定权重乘输入”，Noisy Linear 输出是“带一次临时抖动的权重乘输入”。

一个最小 DQN 头部示意如下：

```python
class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = NoisyLinear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

net = QNet(state_dim=4, hidden_dim=16, action_dim=2)
q = net(torch.randn(5, 4))
assert q.shape == (5, 2)
```

真实工程例子里，Atari DQN 常见的做法是：卷积特征提取层保持不变，把全连接值头替换成 Noisy Linear。这样改动小，但能直接把探索机制从手工 $\epsilon$-greedy 改成可学习噪声。对于已有 DQN 代码库，这是较低侵入性的升级路径。

实现时有两个经验点。

第一，每次前向都要重新采样噪声，训练阶段尤其如此。否则噪声固定住，探索意义会迅速变弱。

第二，推理阶段是否保留噪声，要看目标。如果是评估最终策略，通常固定为均值参数或者关闭噪声；如果还在持续在线训练，则仍可保留采样。

---

## 工程权衡与常见坑

Noisy Nets 的收益来自更强的结构化探索，但代价是训练方差更大，因此工程上必须多看稳定性。

| 问题 | 典型症状 | 对策 |
|---|---|---|
| $\sigma$ 过大 | 得分大幅抖动，TD loss 不稳定 | 用 `softplus` 保证非负，并控制初始化 |
| 噪声过早衰减 | 很快退化成近乎贪心策略 | 检查 $\sigma$ 初始化是否过小 |
| 重采样频率不当 | 同一批样本梯度高方差 | 固定一次前向内部的噪声，按 step 重采样 |
| 目标网络不稳 | Q 值持续放大或震荡 | 配合 target network、Double DQN |
| Replay 失配 | 训练样本对应旧策略分布 | 依赖大 buffer 平滑，并避免过快参数漂移 |

最常见的坑，是把 $\sigma$ 当成普通权重随便训练。如果不做非负约束，$\sigma$ 可能出现符号翻转，虽然数学上某些情况下等价于吸收到 $\epsilon$ 里，但实现上会让解释和调参都变得混乱。更稳妥的方式，是让网络学习 `rho`，再通过 `softplus(rho)` 变成 $\sigma$，这样 $\sigma \ge 0$ 恒成立。

第二个坑，是高方差。参数空间噪声和动作空间噪声不同，它不是让某个动作偶尔乱跳，而是让整张网络一起“摆动”。白话说，$\sigma$ 大的时候，不是手指抖一下，而是整条神经回路都在抖。这样虽然可能跳出局部最优，但也更容易让 TD 目标波动放大。

这时常见的缓解手段有三类。

一类是结构上保守一点，比如只在 value head 上加噪声，不动前面的特征提取层。这样探索仍然存在，但对表示学习的破坏较小。

一类是训练上配套稳定器，比如 target network、Double DQN、Huber loss。这些机制本来就是为了解决值函数训练不稳，在 Noisy Nets 下更加重要。

一类是调初始化。很多实现会把 $\sigma$ 初始得不大不小，让早期足够探索，但不会一开始就把 Q 值打散得太夸张。如果发现训练前几万步分数完全乱飘，首先应检查的不是“算法错了”，而是噪声初始化过猛。

Replay Buffer 还有一个容易被忽略的点。经验回放里的样本来自旧策略分布，而当前训练用的是新参数和新噪声。这个分布失配本来就存在，在参数噪声下可能更明显。实际工程里不能彻底消除，但可以通过较大的 replay buffer、较平滑的目标网络更新、不要让 $\sigma$ 剧烈跳动来缓解。

---

## 替代方案与适用边界

Noisy Nets 不是唯一的探索方法。理解它的价值，最好把它放到几类常见替代方案里比较。

新手版可以这样记：

- 动作噪声：像“每次出手前手抖一下”
- 参数噪声：像“把整个大脑轻微调偏一点”
- Noisy Nets：是参数噪声里一种可学习、可端到端训练的实现

| 方法 | 随机化位置 | 适用场景 | 优势 | 局限 |
|---|---|---|---|---|
| $\epsilon$-greedy | 动作输出 | 离散动作、基线系统 | 简单、稳、易调试 | 探索不连贯 |
| 动作空间噪声 | 动作值本身 | 连续控制常见 | 直观，和策略输出直接配合 | 只改输出，不改内部价值结构 |
| 一般参数空间噪声 | 网络参数 | 策略梯度、值函数都可尝试 | 探索更持久 | 噪声尺度难手调 |
| Noisy Nets | 可学习参数噪声 | DQN 系列最常见 | 把探索强度交给梯度学习 | 方差较大，实现更复杂 |

从适用边界看，Noisy Nets 最合适的地方是离散动作值方法，尤其是 DQN、Double DQN、Dueling DQN、Rainbow 一类框架。因为这些方法本来就依赖值函数头部做动作打分，把普通线性层替换成 Noisy Linear 很自然。

在连续控制里，它并非不能用，但往往要和策略梯度方法一起设计。因为连续控制常见的是高斯策略、熵正则、目标策略平滑等机制，动作分布本身已经承担了大量探索职责。此时 Noisy Nets 可能是补充项，而不是主探索机制。

再说一个真实工程判断标准。如果你的系统已经有成熟的 DQN 训练链路，$\epsilon$ 调度总是难调，且任务需要多步连贯探索，那么 Noisy Nets 值得优先尝试。如果你的任务动作空间很小、奖励密集、基线 DQN 已经足够稳定，那么 Noisy Nets 的收益可能不大，反而会增加调参复杂度。

---

## 参考资料

1. 论文：Fortunato, M. et al. *Noisy Networks for Exploration*，ICLR 2018。主要贡献：提出用可学习参数噪声替代 $\epsilon$-greedy，并给出 Noisy Linear 与因子化高斯噪声的标准形式。新手阅读建议：先看摘要、方法图和公式部分，再回头看实验表。  
2. 文章：Emergent Mind 对 *Noisy Networks for Exploration* 的整理页。主要贡献：把论文核心定义、公式和实验背景压缩成更适合快速浏览的摘要。适合作为首次建立整体概念的入口。  
3. 文章：Emergent Mind 对 NoisyNet-DQN 主题的整理页。主要贡献：补充因子化高斯噪声、损失期望形式和实现细节，便于从“知道概念”过渡到“能写代码”。  
4. 论文：Plappert, M. et al. *Parameter Space Noise for Exploration*。主要贡献：系统比较参数空间噪声与动作空间噪声，解释为什么参数扰动更容易形成一致探索行为，也讨论了稳定性问题。  
5. 快速入门顺序：先读 Fortunato 论文建立定义，再读 NoisyNet-DQN 的实现整理，最后读 Parameter Space Noise for Exploration 理解它和其他探索方法的边界差异。对初学者来说，第一遍不要追求完整证明，先抓住“为什么参数噪声比动作随机更连贯”这个主线。

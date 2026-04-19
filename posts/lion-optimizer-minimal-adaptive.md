## 核心结论

Lion（EvoLved Sign Momentum）是一种只保留一阶动量、用 `sign` 决定更新方向的优化器。优化器是训练模型时负责更新参数的算法，目标是让损失函数逐步变小。和 AdamW 相比，Lion 不维护二阶矩状态，因此每个参数只额外保存一份动量，状态内存大约是 AdamW 的一半。

更直接地说，Adam 像“先看路面是否坑洼，再决定每一步迈多大”；Lion 像“先判断往左还是往右，然后每次迈固定长度的一步”。这个比喻只能帮助理解方向差异，不能替代定义：Lion 的关键不是估计每一维的自适应步长，而是先平滑梯度，再用平滑信号的符号做定幅更新。

| 对比项 | AdamW | Lion |
|---|---:|---:|
| 状态数 | 一阶动量 `m` + 二阶矩 `v` | 一阶动量 `m` |
| 更新依据 | 梯度均值与平方梯度均值 | 平滑方向信号的符号 |
| 步长是否自适应 | 是，每一维有不同缩放 | 否，主要由 `sign` 决定方向 |
| 显存占用 | 较高 | 较低 |
| 调参敏感性 | 相对稳健 | 对 `lr`、`wd`、batch size 更敏感 |

Lion 的优势主要来自工程效率：少一份状态，显存压力更低；更新规则简单，训练吞吐可能更高。论文报告在 ViT、Diffusion、LLM 等场景中有较好表现，速度提升常见范围约为 2%-15%。但它不是“绝对更优”的优化器，尤其不能把 AdamW 的配置原样搬过来。

---

## 问题定义与边界

模型训练通常可以写成一个优化问题：找到参数 $\theta$，让损失函数 $f(\theta)$ 尽可能小。

$$
\theta^* = \arg\min_{\theta} f(\theta)
$$

其中，$\theta$ 是模型参数，$f(\theta)$ 是损失函数，也就是模型预测结果和训练目标之间的差距。优化器每一步要解决的问题是：根据当前梯度，如何把 $\theta$ 更新到一个更好的位置。

标准梯度下降可以写成：

$$
\theta_t = \theta_{t-1} - \eta g_t
$$

这里 $\eta$ 是学习率，表示每一步走多远；$g_t = \nabla f(\theta_{t-1})$ 是当前梯度，表示损失上升最快的方向。因为训练时要降低损失，所以更新方向通常是梯度的反方向。

Lion 的目标不是替代所有优化器，而是在大 batch、长训练、显存敏感的场景里提供更高效率。batch size 是一次训练迭代中同时使用的样本数量。大 batch 通常梯度更稳定，也更适合 Lion 这种“按方向定幅移动”的更新方式。

| 场景 | 是否适合 Lion | 原因 |
|---|---|---|
| 视觉大模型预训练 | 适合尝试 | ViT 等任务中论文报告收益明显 |
| 扩散模型 | 适合尝试 | 训练长、显存压力高，状态减少有价值 |
| 大规模预训练 | 适合尝试 | 参数量大时，优化器状态内存很关键 |
| 资源紧张场景 | 适合尝试 | 少存二阶矩状态可降低显存占用 |
| 小 batch 高噪声场景 | 谨慎使用 | 梯度方向波动大，定幅更新可能不稳 |

玩具例子：如果只训练一个参数 $\theta$，AdamW 会根据梯度历史估计“这一维最近是否波动很大”，再调整这一步的有效步长；Lion 只判断平滑后的方向是正还是负，然后按固定幅度更新。

真实工程例子：如果你训练的是大 batch 的 ViT 预训练，Lion 值得试，因为这类任务训练步数多、参数量大、显存占用敏感。如果你训练的是小 batch 的检测或分割任务，先保守使用 AdamW 往往更稳，再用对照实验判断 Lion 是否值得切换。

---

## 核心机制与推导

Lion 的核心公式可以写成：

$$
g_t = \nabla f(\theta_{t-1})
$$

$$
c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
\theta_t = \theta_{t-1} - \eta_t \cdot \operatorname{sign}(c_t)
$$

$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$

其中，$g_t$ 表示当前梯度；$m_t$ 是一阶动量，也就是梯度的指数滑动平均；$c_t$ 是当前用于决定更新方向的平滑信号；$\operatorname{sign}(c_t)$ 只输出 `-1`、`0` 或 `+1`。$\beta_1$ 和 $\beta_2$ 是动量系数，用来控制历史梯度占多少权重。

Lion 有两个关键机制。

第一，`sign` 让每个参数维度的更新幅度一致。假设学习率是 `0.1`，那么某一维只要 `sign(c_t)=+1`，这一维参数就减少 `0.1`；只要 `sign(c_t)=-1`，这一维参数就增加 `0.1`。梯度是 `0.2` 还是 `20`，不会直接改变这一步的幅度。

第二，权重衰减通常采用解耦形式。权重衰减是一种正则化方法，用来限制参数过大，降低过拟合风险。解耦权重衰减指的是不把衰减项混进梯度里，而是在更新前或更新时单独缩放参数。常见实现顺序是：

$$
\theta \leftarrow \theta \cdot (1 - \eta \lambda)
$$

然后再执行：

$$
\theta \leftarrow \theta - \eta \cdot \operatorname{sign}(c_t)
$$

下面用一个两步数值例子说明 Lion 看的是方向，而不是梯度幅度。设 $\theta_0=1.00$，$m_0=0$，$\beta_1=0.9$，$\beta_2=0.99$，$\eta=0.10$，暂时不使用权重衰减。

第一步 `g1 = 0.20`：

$$
c_1 = 0.9 \times 0 + 0.1 \times 0.20 = 0.02
$$

所以 `sign(c1)=+1`，参数从 `1.00` 变成 `0.90`。

第二步 `g2 = -0.40`：

$$
c_2 = 0.9 \times 0.002 + 0.1 \times (-0.40) = -0.0382
$$

所以 `sign(c2)=-1`，参数从 `0.90` 又回到 `1.00`。

| t | `g_t` | `c_t` | `sign(c_t)` | `θ_t` | `m_t` |
|---:|---:|---:|---:|---:|---:|
| 0 | - | - | - | 1.0000 | 0.0000 |
| 1 | 0.20 | 0.0200 | +1 | 0.9000 | 0.0020 |
| 2 | -0.40 | -0.0382 | -1 | 1.0000 | -0.00202 |

这个例子很小，但足够说明核心差异：第二步的梯度幅度更大，Lion 并没有把步子放大到两倍，而是只根据平滑方向改变更新方向。

---

## 代码实现

Lion 的核心状态只有一份动量 `m`。AdamW 通常需要 `exp_avg` 和 `exp_avg_sq`，前者是一阶动量，后者是平方梯度的滑动平均。Lion 不需要 `exp_avg_sq`，所以每个参数少维护一份同形状张量。

最小伪代码如下：

```text
for each parameter p:
    if state[p] not exists:
        state[p].m = zeros_like(p)

    g = p.grad
    m = state[p].m

    p *= (1 - lr * weight_decay)

    u = beta1 * m + (1 - beta1) * g
    p -= lr * sign(u)

    m = beta2 * m + (1 - beta2) * g
    state[p].m = m
```

一个可运行的 Python 玩具实现如下，不依赖 PyTorch，只演示单个标量参数的更新逻辑：

```python
def sign(x):
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0

def lion_step(theta, grad, m, lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.0):
    theta = theta * (1 - lr * weight_decay)
    update_signal = beta1 * m + (1 - beta1) * grad
    theta = theta - lr * sign(update_signal)
    m = beta2 * m + (1 - beta2) * grad
    return theta, m, update_signal

theta = 1.0
m = 0.0

theta, m, c1 = lion_step(theta, grad=0.20, m=m)
assert round(c1, 4) == 0.02
assert round(theta, 4) == 0.9
assert round(m, 4) == 0.002

theta, m, c2 = lion_step(theta, grad=-0.40, m=m)
assert round(c2, 4) == -0.0382
assert round(theta, 4) == 1.0
assert round(m, 5) == -0.00202
```

接近 PyTorch 的结构可以写成：

```python
class LionLike:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.state = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            if p not in self.state:
                self.state[p] = {"m": torch.zeros_like(p)}

            grad = p.grad
            m = self.state[p]["m"]

            p.data.mul_(1 - self.lr * self.weight_decay)

            update = m.mul(self.beta1).add(grad, alpha=1 - self.beta1)
            p.data.add_(update.sign(), alpha=-self.lr)

            m.mul_(self.beta2).add_(grad, alpha=1 - self.beta2)
```

| 优化器 | 每个参数的额外状态 | 典型字段 |
|---|---:|---|
| SGD | 0 | 无 |
| SGD+Momentum | 1 | `momentum_buffer` |
| AdamW | 2 | `exp_avg`, `exp_avg_sq` |
| Lion | 1 | `exp_avg` 或 `m` |

实现顺序很重要。通常应先做解耦权重衰减，再计算 `sign` 更新，最后更新动量状态。如果把权重衰减混进梯度，或者先更新 `m` 再计算 `u`，训练行为就会和论文及官方实现不一致。

---

## 工程权衡与常见坑

Lion 的主要价值是效率，不是保证更高精度。它通常更省显存、吞吐更高，但对超参数更敏感，尤其是 `lr`、`wd`、`beta2` 和 batch size。

学习率 `lr` 控制每一步的基础步长。因为 Lion 的更新使用 `sign`，有效步幅不像 AdamW 那样被二阶矩缩放，所以直接沿用 AdamW 的学习率容易过大。权重衰减 `wd` 控制参数缩放强度。由于解耦衰减的实际强度和 `lr * wd` 相关，当 `lr` 降低时，`wd` 往往也需要重新调整。

错误做法：AdamW 用什么配置，就给 Lion 用什么配置。  
正确做法：先把 `lr` 降低，把 `wd` 提高，再重新扫描 `beta2` 和 batch size。

| 坑点 | 现象 | 规避建议 |
|---|---|---|
| 直接沿用 AdamW 学习率 | loss 抖动、发散或精度下降 | 先把 `lr` 降到 AdamW 的约 `1/3` 到 `1/10` |
| 不重新调 weight decay | 正则强度变化，泛化变差 | 随 `lr` 调整 `wd`，重新做网格搜索 |
| 小 batch 盲目使用 | 梯度方向噪声大，更新不稳定 | 小 batch 任务优先保留 AdamW 基线 |
| 忽略数值波动 | 早期训练指标看似异常 | 观察更长窗口，并记录梯度范数、loss 曲线 |
| 只看单次实验 | 误判收益 | 至少固定随机种子做 AdamW 与 Lion 对照 |

真实工程中，一个合理的切换流程是：先保留 AdamW 的完整基线，包括最终指标、训练速度、显存峰值和 loss 曲线；再把优化器换成 Lion，只调整最关键的 `lr` 和 `wd`；如果指标接近，再继续扫描 `beta2`、warmup 步数和 batch size。warmup 是训练初期逐步增大学习率的策略，用来避免刚开始更新过猛。

---

## 替代方案与适用边界

Lion 应该放在优化器谱系里理解。它最接近“动量 + 符号更新”的思路，不是 AdamW 的直接替身，也不是 SGD 的简单加强版。

| 优化器 | 状态量 | 步长策略 | 典型优点 | 典型风险 | 适合场景 |
|---|---:|---|---|---|---|
| AdamW | 2 | 二阶矩自适应缩放 | 稳定、默认选择强 | 状态内存较高 | Transformer、通用深度学习训练 |
| SGD+Momentum | 1 | 全局学习率 + 动量 | 简单、泛化常较好 | 调参依赖任务经验 | CNN、分类任务、成熟训练配方 |
| Lion | 1 | 平滑方向 + `sign` 定幅更新 | 省状态、吞吐高 | 对超参和 batch size 敏感 | 大 batch、长训练、显存敏感任务 |
| Adafactor | 低于 AdamW | 分解二阶矩近似 | 极大模型状态更省 | 行为更复杂，任务适配成本高 | 超大模型、内存极度受限场景 |

选择指南可以写得很明确：如果你优先考虑稳定性，先用 AdamW；如果你优先考虑吞吐和显存，试 Lion；如果你在极大模型上追求更低状态开销，再考虑 Adafactor 类方案；如果你训练的是传统视觉模型并且已有成熟配方，SGD+Momentum 仍然值得保留。

Lion 的边界也要清楚。它没有统一的“全局最优”地位。大 batch、长训练、参数量大、显存紧张时，它更有性价比；小 batch、高噪声、训练已经不稳定时，它可能不会带来收益。工程上判断一个优化器是否值得使用，不能只看论文结论，而要同时看最终指标、收敛速度、显存峰值、吞吐、调参成本和失败概率。

---

## 参考资料

1. [论文 PDF：Symbolic Discovery of Optimization Algorithms](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a39b4925e35cf447ccba8757137d84f-Paper-Conference.pdf)
2. [Google 官方实现：google/automl/lion/lion_pytorch.py](https://github.com/google/automl/blob/master/lion/lion_pytorch.py)
3. [NeurIPS Poster：Symbolic Discovery of Optimization Algorithms](https://nips.cc/virtual/2023/poster/70492)
4. [OpenReview 条目：Symbolic Discovery of Optimization Algorithms](https://openreview.net/forum?id=ne6zeqLFCZ)

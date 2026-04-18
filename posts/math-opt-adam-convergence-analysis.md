## 核心结论

Adam 是把一阶矩 `m_t` 和二阶矩 `v_t` 结合起来的自适应优化器。一阶矩是梯度的指数移动平均，用来估计更新方向；二阶矩是梯度平方的指数移动平均，用来估计每个参数维度的梯度尺度。它的核心不是“自动找到最优解”，而是“方向用平均梯度，步长按坐标自动缩放”。

如果把训练看成走山路，Adam 一边记住最近大致往哪边走，一边记住这条路最近有多陡。路很陡的方向就小步走，平缓的方向就大步走，所以它比“全局一个学习率”更灵活，但也可能因为记忆方式不当而走偏。

Adam 的标准更新公式是：

$$
g_t = \nabla f(\theta_{t-1})
$$

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
$$

其中 $\eta$ 是基础学习率，$\beta_1$ 控制方向记忆的衰减速度，$\beta_2$ 控制尺度记忆的衰减速度，$\varepsilon$ 用来避免分母为 0。

| 结论 | 含义 |
|---|---|
| 有优势 | 对噪声梯度、稀疏梯度、不同维度尺度不一致更友好 |
| 有风险 | 经典 Adam 并不天然保证收敛 |
| 工程建议 | 优先考虑 AdamW；收敛性风险较高时考虑 AMSGrad |

同样面对稀疏梯度，SGD 使用所有参数共享的学习率，某些很少出现梯度的维度可能更新太慢。Adam 会根据每个维度自己的二阶矩调整有效步长，稀疏维度的分母通常较小，因此有效更新可能更大。

---

## 问题定义与边界

本文讨论的“收敛性”不是简单看训练 loss 是否下降。训练 loss 暂时下降只能说明当前优化过程在某些 batch 或某段时间内有效，不能说明算法理论上一定会稳定到合理点。

在深度学习中，更常见的收敛目标是接近驻点。驻点是梯度接近 0 的点，白话说就是参数附近已经没有明显的下降方向。非凸优化里的目标通常不是保证找到全局最优，而是希望 $\|\nabla f(\theta)\|$ 逐渐变小，最后不要持续大幅震荡。

随机优化问题通常写成：

$$
\min_\theta f(\theta) = \mathbb{E}_\xi[F(\theta; \xi)]
$$

这里 $\xi$ 表示随机样本或 mini-batch，$F(\theta;\xi)$ 是单个样本或一批样本上的损失，$f(\theta)$ 是总体期望损失。

训练一个深度网络时，目标函数通常不是一条平滑的碗，而是很多坑和坡。这里说的“收敛”，不是保证到全局最优，而是希望最后别再明显乱跳，梯度也逐渐变小。

常见收敛性分析依赖以下假设：

| 假设 | 作用 |
|---|---|
| 目标函数下界存在 | 防止目标函数无限下降，保证优化问题有意义 |
| 平滑或 Lipschitz 梯度 | 控制单步更新对函数值的影响 |
| 随机梯度无偏 | 保证随机梯度在期望上指向真实梯度 |
| 方差或二阶矩有界 | 控制噪声不会无限放大 |
| 学习率按证明要求衰减 | 支撑理论收敛界，而不是只用固定学习率经验训练 |

本文的边界是非凸随机优化背景下的 Adam 收敛性。凸优化中的部分结论更强，但不能直接搬到深度网络。深度网络通常是非凸问题，且梯度噪声、归一化层、动量缓存、混合精度都会影响实际轨迹。

一个反例直觉是：即使 loss 在前几千步下降，也不能说明 Adam 在所有梯度序列下都收敛。有些构造出的梯度序列会让 Adam 的有效步长在后期反弹，导致参数持续震荡。

---

## 核心机制与推导

Adam 的机制可以拆成两层指数移动平均。指数移动平均，简称 EMA，是“越近的数据权重越大、越远的数据权重越小”的加权平均。

第一层是方向记忆：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$m_t$ 不是当前梯度，而是历史梯度的平滑版本。它降低了单个 batch 噪声对方向的影响。

第二层是尺度记忆：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$v_t$ 记录每个维度的梯度平方大小。梯度平方越大，说明这个维度最近变化更剧烈，更新时分母越大，有效步长越小。

第一步看到梯度很大，Adam 会记住这个方向；之后即使当前梯度变成 0，历史信息还在，所以参数还会继续动。它不是只看当下，而是看一段时间的平均状态。

偏差校正来自一个简单事实：初始化时 $m_0=0, v_0=0$，早期 EMA 会被 0 拉低。假设梯度均值稳定为 $\mu$，梯度平方均值稳定为 $\nu$，则有：

$$
\mathbb{E}[m_t] = (1-\beta_1^t)\mu,\quad \mathbb{E}[v_t] = (1-\beta_2^t)\nu
$$

所以需要除以 $1-\beta_1^t$ 和 $1-\beta_2^t$：

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

这就是偏差校正。它的作用是修复早期低估，而不是改变长期优化目标。

玩具例子：取 $\beta_1=0.9,\beta_2=0.999,\eta=10^{-3},\varepsilon\approx0$，一维梯度序列为 $g_1=1,g_2=0$。

| t | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | 更新量 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.1 | 0.001000 | 1.0000 | 1.0000 | 0.0010 |
| 2 | 0 | 0.09 | 0.000999 | 0.4737 | 0.4997 | 0.00067 |

第二步当前梯度已经是 0，但 $m_t$ 仍保留正方向，因此更新不会立刻停止。这是动量类方法的正常行为，也解释了为什么 Adam 可能在噪声环境中更平滑。

Adam 收敛分析的核心矛盾在于有效学习率。对第 $i$ 个参数维度，有效步长近似为：

$$
\alpha_{t,i} = \frac{\eta}{\sqrt{\hat v_{t,i}}+\varepsilon}
$$

如果 $\hat v_{t,i}$ 下降，有效步长就会上升。经典 Adam 的二阶矩分母不是单调不减的，因此某些梯度序列下会出现有效步长回弹，破坏理论收敛条件。

AMSGrad 的改动是：

$$
\tilde v_t = \max(\tilde v_{t-1}, v_t)
$$

然后用 $\tilde v_t$ 替代 $v_t$ 参与分母计算。这样分母单调不减，有效步长不会因为二阶矩下降而反弹，收敛性分析更容易成立。

| 步骤 | 作用 |
|---|---|
| 计算梯度 | 获取当前下降方向 |
| 更新 `m_t` | 平滑方向噪声 |
| 更新 `v_t` | 平滑尺度信息 |
| 偏差校正 | 修正早期低估 |
| 参数更新 | 按坐标自适应缩放 |

非凸收敛性结果通常不是说“Adam 一定找到最优解”，而是在额外条件下证明平均梯度范数会下降，例如达到某类 $O(\log T / T)$ 量级的界。这里的重点是条件，而不是只记住复杂度符号。

---

## 代码实现

代码里不会真的写“记忆方向”这句话，而是用 `exp_avg` 存 $m_t$，用 `exp_avg_sq` 存 $v_t$。每一步先更新这两个缓存，再算出真正要加到参数上的步子。

下面是一个可运行的一维 Adam 与 AMSGrad 最小实现：

```python
import math

def adam_steps(grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, amsgrad=False):
    theta = 0.0
    m = 0.0
    v = 0.0
    v_max = 0.0
    updates = []

    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        if amsgrad:
            v_max = max(v_max, v_hat)
            denom = math.sqrt(v_max) + eps
        else:
            denom = math.sqrt(v_hat) + eps

        step = lr * m_hat / denom
        theta -= step
        updates.append(step)

    return theta, updates

theta, updates = adam_steps([1.0, 0.0])
assert len(updates) == 2
assert abs(updates[0] - 1e-3) < 1e-8
assert 6.6e-4 < updates[1] < 6.8e-4
assert theta < 0
```

公式和代码变量可以这样对应：

| 公式符号 | 代码变量 |
|---|---|
| $g_t$ | `g` 或 `p.grad` |
| $m_t$ | `exp_avg` / `m` |
| $v_t$ | `exp_avg_sq` / `v` |
| $\hat m_t$ | bias-corrected `exp_avg` |
| $\hat v_t$ | bias-corrected `exp_avg_sq` |
| $\tilde v_t$ | `max_exp_avg_sq` / `v_max` |

PyTorch 风格的核心逻辑如下：

```python
for p in model.parameters():
    if p.grad is None:
        continue

    state = optimizer_state[p]
    grad = p.grad

    state["m"] = beta1 * state["m"] + (1 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1 - beta2) * (grad * grad)

    m_hat = state["m"] / (1 - beta1 ** step)
    v_hat = state["v"] / (1 - beta2 ** step)

    p.data -= lr * m_hat / (v_hat.sqrt() + eps)
```

加入 AMSGrad 时，只需要让二阶矩分母使用历史最大值：

```python
state["v_max"] = torch.maximum(state["v_max"], state["v"])
p.data -= lr * m_hat / (state["v_max"].sqrt() + eps)
```

真实工程例子：训练 Transformer 或 LLM 时，embedding 层梯度经常稀疏，attention 层梯度噪声可能较大，不同参数矩阵的尺度也不同。Adam 能减少手工为每类参数设计学习率的成本，因此常被作为默认起点。但如果验证集 loss 抖动、训练后期不稳定，通常要检查 warmup、学习率衰减、梯度裁剪、AdamW 或 AMSGrad，而不是只盯着 Adam 名字本身。

---

## 工程权衡与常见坑

工程上最常见的问题不是“Adam 不够聪明”，而是学习率、权重衰减、数值稳定性和训练计划没有配套处理。

`eps` 是数值稳定项。它避免 $\sqrt{\hat v_t}$ 过小时分母接近 0。如果 `eps` 设得太小，分母接近 0 时会把噪声放大；如果设得太大，又会削弱 Adam 的自适应缩放能力。

`weight_decay` 是权重衰减，白话说就是让参数不要无约束变大。经典 Adam 中把 weight decay 直接加进梯度，行为不等同于标准解耦权重衰减。AdamW 把参数衰减和梯度更新拆开，在 Transformer、LLM 等任务中更常用。

warmup 是训练初期逐渐增大学习率的策略。Adam 有偏差校正，但这不等于初期不会不稳定。大模型训练中，不做 warmup 可能让早期更新过猛。

梯度裁剪是限制梯度范数的操作。它的作用是防止少数异常 batch 产生过大的更新，尤其适合序列模型、强化学习、混合精度训练。

| 问题 | 后果 | 处理方式 |
|---|---|---|
| `eps` 过小 | 数值噪声被放大 | 调大 `eps` |
| 把 Adam weight decay 当 L2 | 正则行为偏差 | 改用 AdamW |
| 不做 warmup | 初期不稳定 | 加 warmup |
| 不做梯度裁剪 | 尖峰更新过大 | clip gradient |
| 稀疏梯度长期尖峰 | 后续步长偏小 | 检查 AMSGrad、学习率和裁剪 |
| 固定学习率训练到底 | 后期震荡 | 使用学习率衰减 |

一个常见场景是训练 Transformer：embedding 层某些 token 很少出现，梯度稀疏；attention 层受 batch 内容影响大，梯度噪声高。Adam 常能让模型先跑起来，但如果验证集开始抖动，优先检查学习率是否过大、warmup 是否太短、是否需要梯度裁剪，而不是直接换优化器。

另一个坑是把 Adam 看成“学习率免调”。Adam 只是按坐标调整有效步长，基础学习率 $\eta$ 仍然决定整体更新强度。对于同一个模型，`1e-3`、`3e-4`、`1e-4` 可能得到完全不同的稳定性和泛化结果。

---

## 替代方案与适用边界

Adam 不是唯一选择，也不是所有任务的最优选择。优化器选择应根据目标函数形态、梯度稀疏性、稳定性要求、泛化表现和工程成本决定。

如果 Adam 像一个会自动调步长的骑手，那么 SGD + Momentum 更像一个更朴素但可能更稳的骑手；AdamW 则是在 Adam 的基础上把“正则化”这件事拆开处理，更适合很多深度学习任务。

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| SGD + Momentum | 简单、泛化常较强 | 调参敏感，收敛初期可能慢 | 经典监督学习、CV 分类 |
| RMSProp | 自适应尺度，适合非平稳梯度 | 理论性质弱于部分改进版 | RNN、在线学习历史任务 |
| Adam | 收敛快，上手容易 | 某些序列下可能不收敛 | 默认实验起点 |
| AdamW | 解耦权重衰减，工程表现稳定 | 仍需调学习率和 warmup | Transformer、LLM、扩散模型 |
| AMSGrad | 更稳的理论性质 | 有时更新更保守 | 收敛性优先、二阶矩不稳定场景 |

训练大模型时，AdamW 通常是默认起点，因为它保留 Adam 的自适应优势，同时修正了 weight decay 的耦合问题。追求最终泛化时，SGD + Momentum 有时仍然更强，尤其在部分视觉监督学习任务中。遇到二阶矩不稳定、有效步长回弹或理论收敛性要求更高的任务时，可以考虑 AMSGrad。

边界需要说清楚：Adam 的优势主要来自局部缩放和动量平滑，不代表它理解目标函数结构。对于病态目标、错误归一化、错误损失设计、数据分布漂移，换 Adam 不能从根上解决问题。优化器只能改善搜索过程，不能替代建模、数据和训练目标本身。

---

## 参考资料

如果想继续往下看，先读 Adam 原论文理解算法本身，再读 AMSGrad 论文理解为什么经典 Adam 会出问题，最后看 PyTorch / TensorFlow 文档理解工程实现差异。

| 顺序 | 目的 |
|---|---|
| 1 | 先理解 Adam 更新公式 |
| 2 | 再看收敛性问题 |
| 3 | 再看 AMSGrad / AdamW |
| 4 | 最后看框架实现细节 |

原始算法：

- Kingma & Ba, *Adam: A Method for Stochastic Optimization*  
  https://arxiv.org/abs/1412.6980

收敛性改进：

- Reddi et al., *On the Convergence of Adam and Beyond*  
  https://openreview.net/forum?id=ryQu7f-RZ

- Chen et al., *On the convergence of a class of Adam-type algorithms for non-convex optimization*  
  https://experts.umn.edu/en/publications/on-the-convergence-of-a-class-of-adam-type-algorithms-for-non-con/

工程文档：

- PyTorch `torch.optim.Adam` 官方文档  
  https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html

- TensorFlow `ResourceApplyAdamWithAmsgrad` 官方文档  
  https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resource-apply-adam-with-amsgrad

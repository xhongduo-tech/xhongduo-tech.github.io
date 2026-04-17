## 核心结论

Adam 是一种自适应优化器，也就是“每个参数自己调学习率”的更新方法。它把两类历史信息合在一起用：

1. 一阶矩，直白说就是“平均梯度方向”，回答参数应该往哪边走。
2. 二阶矩，直白说就是“平均平方梯度大小”，回答这一步应该走多快。

它的核心公式是：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1}=\theta_t-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

这里 $g_t$ 是当前梯度，$\theta$ 是参数，$\alpha$ 是学习率，$\epsilon$ 是防止除零的小常数。

新手可以先用一个玩具比喻理解：方向盘控制“朝哪转”，油门控制“开多快”。Momentum 像方向盘，RMSProp 像油门，Adam 把两者放在同一辆车里，所以常被概括为 `Adam = Momentum + RMSProp`。

在现代工程里，更常用的是 AdamW。它和 Adam 的关键差异不是“更新更复杂”，而是把权重衰减从梯度更新中拆开，避免正则化强度被自适应缩放破坏。这也是 Transformer、GPT、ViT 一类模型训练中默认更常见的选择。

| 方法 | 是否使用方向动量 | 是否使用幅度自适应 | 权重衰减是否解耦 | 典型结论 |
|---|---|---|---|---|
| Adam | 是 | 是 | 否 | 收敛快，但 L2 正则可能被缩放 |
| AdamW | 是 | 是 | 是 | 更适合大模型和稳定正则化 |

---

## 问题定义与边界

问题本质是：不同参数维度的梯度统计并不一样，统一学习率往往不合理。

举一个最小直觉例子。假设两个参数的梯度分别是 $0.001$ 和 $10$。如果都用同一个学习率，前者几乎不动，后者可能直接震荡。Adam 的目标就是让每个参数维度根据自己的历史梯度，得到不同的有效步长。

它处理的是“参数级别步长不均衡”问题，不处理“目标函数本身不可优化”问题。比如学习率设得离谱、数据脏、梯度爆炸完全未控，这些不是 Adam 单独能救回来的。

Adam 的常见边界条件如下：

| 超参 | 默认值 | 白话解释 | 影响 |
|---|---:|---|---|
| $\alpha$ | 任务相关 | 基础学习率 | 控制整体更新尺度 |
| $\beta_1$ | 0.9 | 方向记忆长度 | 越大越平滑，方向更稳 |
| $\beta_2$ | 0.999 | 幅度记忆长度 | 越大越稳定，步长更保守 |
| $\epsilon$ | $10^{-8}$ | 防止分母为 0 | 影响数值稳定性 |
| $m_0,v_0$ | 0 | 初始历史统计 | 导致前几步有偏，需要修正 |

为什么要偏差修正？因为一开始 $m_0=v_0=0$，前几步的指数滑动平均会系统性偏小。偏差修正就是把这个“刚开机、历史不够”的误差补回来：

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

如果没有这一步，第一批更新常会比预期更小。对小模型可能只是慢一点，对大模型训练前期则可能直接影响 warmup 段的稳定性。

再看一个新手例子：如果某一步梯度突然变大，SGD 会按原学习率直接跟着冲；Adam 会因为 $\sqrt{\hat{v}_t}$ 变大而自动压小有效步长，相当于“梯度大时自动收油门”。

---

## 核心机制与推导

Adam 的机制可以拆成三层。

第一层是一阶矩 $m_t$。矩在这里可以简单理解成“统计量”。一阶矩记录梯度的指数滑动平均，也就是最近一段时间主要往哪个方向走。$\beta_1=0.9$ 表示保留大量历史方向，当前梯度只占约 10% 的新增权重。

第二层是二阶矩 $v_t$。它记录平方梯度的指数滑动平均。平方的作用是去掉正负号，只保留“幅度有多大”。如果某个参数最近总是出现很大的梯度，那么它的 $v_t$ 就会上升，更新时分母变大，步长自动减小。

第三层是把两者合并：

$$
\Delta\theta_t=-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

这里分子给方向，分母给尺度。于是就形成了一个自然逻辑：

1. 当前梯度 $g_t$ 进来。
2. 用 $m_t$ 平滑方向，防止一步左一步右。
3. 用 $v_t$ 估计幅度，防止某些维度走太猛。
4. 用偏差修正恢复前几步统计量的量级。
5. 最终得到方向稳定、幅度自适应的更新。

玩具例子最能看清这个过程。设：

- $g_1=0.2$
- $m_0=0,\ v_0=0$
- $\beta_1=0.9,\ \beta_2=0.999$

那么第一步有：

$$
m_1=0.9\cdot 0+0.1\cdot 0.2=0.02
$$

$$
v_1=0.999\cdot 0+0.001\cdot 0.2^2=0.00004
$$

偏差修正后：

$$
\hat{m}_1=\frac{0.02}{1-0.9}=0.2
$$

$$
\hat{v}_1=\frac{0.00004}{1-0.999}=0.04
$$

于是更新量为：

$$
\alpha\frac{0.2}{\sqrt{0.04}+\epsilon}\approx \alpha
$$

这说明第一步在偏差修正后，会恢复到和当前梯度量级一致的尺度。如果不修正，前几步会被严重低估。

再看一个两维参数的玩具例子。若某一维梯度长期是 $[0.1,0.1,0.1]$，另一维长期是 $[10,9,11]$，那么第二维的 $v_t$ 会远大于第一维，导致它的有效步长更小。Adam 就是在做这种“高波动维度更保守，低波动维度更敢走”的分维控制。

真实工程例子是训练 Transformer。词嵌入层、注意力投影层、LayerNorm 参数的梯度统计差异很大。如果统一用 SGD 学习率，往往需要非常细致地试错；而 AdamW 允许不同参数按历史幅度自动归一，配合 warmup 和 cosine decay 之类的调度器，训练会更稳。

---

## 代码实现

下面先给出一个最小可运行的 Python 实现，展示 Adam 与 AdamW 的关键区别。代码里只有标量参数，但逻辑和向量、张量版本完全一致。

```python
import math

def adam_step(theta, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # 一阶矩：方向的指数滑动平均
    m = beta1 * m + (1 - beta1) * grad
    # 二阶矩：平方梯度幅度的指数滑动平均
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # 偏差修正：补偿 m,v 从 0 开始时前几步偏小的问题
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps)
    return theta, m, v

def adamw_step(theta, grad, m, v, t, lr=1e-3, weight_decay=1e-2,
               beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # AdamW: 权重衰减直接作用在参数上，而不是混进梯度里
    theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps) - lr * weight_decay * theta
    return theta, m, v

# 玩具验证
theta, m, v = 1.0, 0.0, 0.0
theta2, m2, v2 = adam_step(theta, grad=0.2, m=m, v=v, t=1, lr=0.1)
assert theta2 < theta
assert m2 > 0
assert v2 > 0

theta_w, _, _ = adamw_step(1.0, grad=0.2, m=0.0, v=0.0, t=1, lr=0.1, weight_decay=0.01)
assert theta_w < theta2  # AdamW 多了一项参数衰减

# 无梯度时，Adam 不动；AdamW 仍会因权重衰减收缩参数
theta_zero, _, _ = adam_step(1.0, grad=0.0, m=0.0, v=0.0, t=1, lr=0.1)
theta_zero_w, _, _ = adamw_step(1.0, grad=0.0, m=0.0, v=0.0, t=1, lr=0.1, weight_decay=0.01)
assert abs(theta_zero - 1.0) < 1e-12
assert theta_zero_w < 1.0
```

变量作用可以用这个小表记住：

| 变量 | 作用 |
|---|---|
| `theta` | 当前参数 |
| `grad` | 当前梯度 |
| `m` | 一阶矩，保存方向趋势 |
| `v` | 二阶矩，保存幅度趋势 |
| `t` | 第几步，用于偏差修正 |
| `lr` | 基础学习率 |
| `weight_decay` | AdamW 的解耦权重衰减 |

两者的伪代码差异本质上只有一行：

```python
# Adam
theta -= lr * m_hat / (sqrt(v_hat) + eps)

# AdamW
theta -= lr * m_hat / (sqrt(v_hat) + eps)
theta -= lr * weight_decay * theta
```

这里要注意，AdamW 不是“把梯度改成 grad + lambda * theta”，而是把参数衰减单独做一次。这就是“解耦”的含义。

---

## 工程权衡与常见坑

第一个常见坑是把 Adam 和“L2 正则”简单等同。在线性回归或普通 SGD 里，把 $\lambda\theta$ 加进梯度通常能等价表达权重衰减；但在 Adam 里，这一项会被分母 $\sqrt{\hat{v}_t}$ 再缩放一次，于是不同参数受到的正则化强度不一致。

旧写法近似是：

$$
\theta_{t+1}=\theta_t-\alpha\frac{\hat{m}_t+\lambda\theta_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

AdamW 则改成：

$$
\theta_{t+1}=\theta_t-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}-\alpha\lambda\theta_t
$$

这两者不等价。前者的正则项会被自适应分母扭曲，后者才是真正稳定的参数收缩。

第二个坑是忽略偏差修正。有人自己手写优化器时只写 EMA，不写 $\hat{m}_t,\hat{v}_t$，结果前几步更新明显偏小，还以为是学习率太低。

第三个坑是误解默认超参的含义。$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$ 不是神奇数字，而是一组经验上平衡“方向平滑”和“幅度稳定”的默认组合。$\beta_2$ 很大，意味着平方梯度统计记忆更长，因为幅度波动通常比方向更噪。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 在 Adam 中直接加 L2 | 正则化效果不稳定 | 被 $\sqrt{\hat{v}_t}$ 缩放 | 用 AdamW |
| 忘记偏差修正 | 前几步更新太小 | $m,v$ 从 0 初始化 | 实现 $\hat{m}_t,\hat{v}_t$ |
| $\beta_2$ 设太小 | 步长剧烈波动 | 幅度统计记忆太短 | 从 0.999 起调 |
| 学习率照搬 SGD | 训练不稳 | Adam 的有效步长机制不同 | 单独搜索 Adam/AdamW 学习率 |
| 把所有参数都做衰减 | 训练质量下降 | bias、LayerNorm 通常不宜衰减 | 参数分组设置 weight decay |

真实工程例子：训练 GPT 或 ViT 时，优化器通常会把线性层权重做 weight decay，但不对 bias、LayerNorm、Embedding norm 这类参数做衰减。原因不是“社区习惯”，而是这些参数的尺度语义不同，盲目衰减常导致性能下降。

---

## 替代方案与适用边界

Adam 不是唯一可用方案，它是在多个优化器思想之间做折中。

Momentum SGD 只保留方向信息，不做参数级别自适应。优点是简单、泛化常常不错，缺点是对学习率更敏感。RMSProp 只保留幅度信息，不显式积累方向趋势，所以能抑制震荡，但没有动量那种“沿主方向持续推进”的效果。Adam 则把两者合并。

AdaGrad 也做自适应，但它会把历史平方梯度一直累加，不会遗忘。优点是稀疏特征任务上常很有效，因为频繁更新的维度会越来越保守；缺点是训练时间一长，学习率可能衰减得过头。

下面用表格压缩这些差异：

| 优化器 | 核心思想 | 适用场景 | 主要缺点 |
|---|---|---|---|
| SGD + Momentum | 方向动量 | 视觉分类、成熟配方训练 | 对学习率敏感 |
| RMSProp | 幅度自适应 | 非平稳目标、RNN 早期训练 | 缺少方向累积 |
| AdaGrad | 累积平方梯度 | 稀疏特征、浅层模型 | 学习率单调变小 |
| Adam | 动量 + 自适应 | 通用深度学习起点 | 正则化与缩放耦合 |
| AdamW | Adam + 解耦衰减 | Transformer、GPT、ViT | 仍需调学习率和衰减 |

新手可以这样记：

- 只想加方向惯性，用 Momentum SGD。
- 只想按梯度大小自动调速，用 RMSProp。
- 同时想要方向感和调速能力，用 Adam。
- 还想把正则化做对，尤其训练大模型，用 AdamW。

下面是 Adam 与 RMSProp 的最小流程对比：

```python
# RMSProp
v = beta2 * v + (1 - beta2) * grad**2
theta -= lr * grad / (sqrt(v) + eps)

# Adam
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
theta -= lr * m_hat / (sqrt(v_hat) + eps)
```

适用边界也要说清楚。若你在一个小型视觉任务上已经有成熟 SGD 配方，Adam 不一定带来更好泛化；若任务梯度极稀疏，AdaGrad 有时仍然是低成本可用选择；若是现代大模型预训练或微调，AdamW 基本是更稳妥的默认起点。

---

## 参考资料

- [DataCamp: Adam Optimizer Tutorial](https://www.datacamp.com/tutorial/adam-optimizer-tutorial?utm_source=openai)
  作用：解释 Adam 的一阶矩、二阶矩、偏差修正和最小数值例子，适合先把公式跑通。
- [Michael Brenndoerfer: AdamW Optimizer, Decoupled Weight Decay](https://mbrenndoerfer.com/writing/adamw-optimizer-decoupled-weight-decay?utm_source=openai)
  作用：说明为什么 Adam 中直接加 L2 会失真，以及 AdamW 的解耦权重衰减为何必要。
- [Keras API: AdamW](https://keras.io/2/api/optimizers/adamw/?utm_source=openai)
  作用：给出工程实现、默认超参与现代框架中的标准接口，适合对照实际训练代码。

如果要自己做实验，最简单的方法是复现文中的玩具例子：手工设定 $g_1=0.2$、$m_0=v_0=0$，把第一步 Adam 和 AdamW 都算一遍，再把 `weight_decay` 改成 0、0.01、0.1，观察参数收缩是否变化。这比直接记结论更容易建立直觉。

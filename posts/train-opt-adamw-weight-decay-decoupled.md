## 核心结论

AdamW 解决的不是“Adam 会不会收敛”，而是“Adam 里的权重衰减到底是不是你以为的那个权重衰减”。

传统做法把 L2 正则直接并入梯度：

$$
g_t \leftarrow \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t
$$

然后再把这个总梯度送进 Adam。问题在于，Adam 会用历史一阶矩 $m_t$ 和二阶矩 $v_t$ 对梯度做自适应缩放，白话说，就是“不同参数会被自动分配不同步长”。一旦把 $\lambda \theta_t$ 混进梯度，L2 惩罚也被这套自适应机制一起缩放了。结果不是“所有参数按同样频率缩小”，而是“大梯度参数衰减偏弱，小梯度参数衰减偏强”。

AdamW 把这两件事拆开：

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta \lambda \theta_t
$$

其中：
- $\eta$ 是学习率，白话说就是每一步走多远
- $\lambda$ 是权重衰减系数，白话说就是每一步把参数往 0 拉回多少
- $\hat m_t/\sqrt{\hat v_t}$ 只负责“按梯度方向更新”
- $\lambda \theta_t$ 只负责“统一收缩参数大小”

同一个式子也可以写成乘性形式：

$$
\theta_{t+1} = (1-\eta\lambda)\theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这个写法更直观。它表示每一步先把参数整体乘上一个略小于 1 的因子，再叠加梯度更新。

一个常见数值例子：取 $\eta=10^{-3}, \lambda=0.01$，则每步乘法因子是：

$$
1-\eta\lambda = 1-10^{-5}=0.99999
$$

单步看几乎没变化，但长期会累积。若忽略梯度项，只看纯衰减，经过 10000 步后：

$$
0.99999^{10000} \approx e^{-0.1} \approx 0.9048
$$

也就是参数大约缩小 9.5%。这正是 AdamW 的核心直觉：单步弱，长期稳，且对所有被衰减的参数一视同仁。

| 项目 | Adam + L2 | AdamW |
|---|---|---|
| L2 是否进入梯度统计 | 是 | 否 |
| 是否被 $\sqrt{\hat v_t}$ 自适应缩放 | 是 | 否 |
| weight decay 是否可直接解释 | 较差 | 较好 |
| 大梯度参数的实际衰减 | 往往更弱 | 与梯度幅值无关 |
| 在 Transformer/LLM 中的地位 | 已不推荐 | 标准选择 |

---

## 问题定义与边界

问题定义很具体：当优化器是 Adam 时，怎样实现“真正的权重衰减”。

“权重衰减”这个词的本意，是让参数在训练过程中持续向 0 收缩。白话说，就是防止模型把权重拉得过大，从而降低过拟合风险。它要表达的是一种独立的收缩机制，而不是“顺便往梯度里再加一点东西”。

在普通 SGD 里，把 L2 正则加进梯度，和做 weight decay 在很多场景下近似等价，因为 SGD 没有按参数维度做复杂的自适应缩放。但在 Adam 里，这种等价关系被破坏了。

传统 Adam 的更新可以写成：

$$
g_t = \nabla_\theta \mathcal{L}(\theta_t) + \lambda \theta_t
$$

$$
m_t=\beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

边界要说清楚：

1. 这里讨论的是 Adam、RMSProp、Adagrad 这一类“自适应优化器”。
2. 讨论重点不是正则化有没有用，而是“L2 并入梯度”在 Adam 下不再等价于“独立权重衰减”。
3. AdamW 只解耦 weight decay，不改变 Adam 的核心自适应更新结构。
4. 不是所有参数都应该衰减。工程上通常排除 bias、LayerNorm，很多实现也会排除 embedding。

先看一个玩具例子。假设两个参数 $w_1,w_2$ 的当前值都等于 1，权重衰减系数也相同：

- $w_1$ 对应的大梯度：$\nabla \mathcal{L}=10$
- $w_2$ 对应的小梯度：$\nabla \mathcal{L}=0.01$

在 Adam + L2 中，$\lambda w$ 会和任务梯度一起进入 $v_t$。由于 $w_1$ 的总梯度大，$v_t$ 也大，分母 $\sqrt{v_t}$ 更大，于是它的“L2 那一部分”会被额外压小。反过来，$w_2$ 的梯度本来就小，分母也小，L2 那部分反而更容易起作用。

可以把它理解成同样一把刷子在擦两个物体，但刷子会自动根据表面粗糙度改力度。结果不是“同样擦掉一层”，而是粗糙的地方擦得更轻，平滑的地方擦得更重。这就是耦合。

| 场景 | 传统 Adam + L2 | AdamW |
|---|---|---|
| 参数梯度很大 | 衰减常被分母稀释 | 衰减强度仍是 $\eta\lambda$ |
| 参数梯度很小 | 衰减常显得更强 | 衰减强度仍是 $\eta\lambda$ |
| $\lambda$ 的可迁移性 | 差 | 更好 |
| 调参含义 | 混杂 | 清晰 |

所以，AdamW 的问题边界不是“替代所有优化器”，而是“在自适应优化器里，把 weight decay 从梯度统计中拿出来”。

---

## 核心机制与推导

AdamW 的关键机制只有一句话：梯度更新和参数收缩分两步做。

第一步，按 Adam 的方式处理任务梯度：

$$
g_t = \nabla_\theta \mathcal{L}(\theta_t)
$$

$$
m_t=\beta_1 m_{t-1} + (1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\theta'_t = \theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

第二步，独立做权重衰减：

$$
\theta_{t+1}=\theta'_t-\eta\lambda\theta_t
$$

把两步合并，就得到常见写法：

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}-\eta\lambda\theta_t
$$

若把后两项重新整理，又可写成：

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这两个形式是等价的。

- 加法形式强调“weight decay 是独立加项”
- 乘法形式强调“每一步都在统一缩小旧参数”

这里有个很重要的推导直觉。传统 Adam + L2 的问题不是“多了一个 $\lambda\theta$”，而是这个项进入了 $m_t,v_t$ 的历史统计。白话说，优化器开始把“正则化项”误当成“任务梯度”来记忆和缩放。这样一来，weight decay 不再有自己独立的物理意义。

AdamW 则要求“遗忘这段历史”：二阶矩 $v_t$ 只能描述任务梯度的波动，不应该描述参数收缩本身。否则你会得到一种奇怪效果：参数收缩力度取决于过去梯度有多大，而不是当前设置的 $\lambda$ 有多大。

下面这个小表可以帮助区分两部分的分工：

| 项 | 作用 | 是否依赖梯度历史 |
|---|---|---|
| $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ | 决定往哪个方向更新、更新多大 | 是 |
| $\lambda \theta_t$ | 决定把参数往 0 拉回多少 | 否 |

再看一个两步对比的玩具例子。假设某一维参数当前为 $\theta_t=2.0$，梯度更新项给出的步长是 0.1，学习率和权重衰减满足 $\eta\lambda=0.01$。

AdamW：
1. 先按梯度更新：$2.0-0.1=1.9$
2. 再做衰减：减去 $0.01\times2.0=0.02$
3. 最终得到 $1.88$

传统 Adam + L2 的思想则是把 $0.02$ 混进梯度，再一起交给 Adam 的自适应分母处理。若该维历史梯度大，那个 0.02 很可能被缩小，最终就不是减去 0.02，而是更少。

这也是为什么论文强调“decoupled”，即解耦。解耦不是语法变化，而是恢复了超参数语义：
- 学习率 $\eta$ 控制任务更新步长
- 权重衰减 $\lambda$ 控制参数收缩时间尺度

如果两个量纠缠在一起，调参会变得很难解释。

真实工程例子可以看 BERT 或 GPT 微调。假设你只换了任务，从情感分类变成命名实体识别，梯度分布会变化很大。若使用 Adam + L2，同一个 $\lambda$ 的实际效果也会跟着漂移；若使用 AdamW，$\lambda$ 至少仍表示“每一步对参数做多大比例的统一收缩”，可迁移性更强。

---

## 代码实现

下面先给一个可运行的简化 Python 版本，只演示单参数 AdamW 的核心逻辑。它不是生产代码，但足够说明机制。

```python
import math

def adamw_scalar(theta, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    m = 0.0
    v = 0.0
    history = []

    for t, g in enumerate(grads, start=1):
        # 1) 只用任务梯度更新动量与二阶矩
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # 2) Adam 自适应梯度步
        theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps)

        # 3) 解耦的 weight decay
        theta = theta - lr * weight_decay * theta

        history.append(theta)

    return theta, history

# 玩具例子：梯度恒定时，参数应持续下降
final_theta, hist = adamw_scalar(theta=1.0, grads=[0.1] * 100)
assert final_theta < 1.0
assert len(hist) == 100

# 验证纯衰减近似：无梯度时应接近乘法收缩
def pure_decay(theta, steps, lr=1e-3, weight_decay=0.01):
    for _ in range(steps):
        theta = theta - lr * weight_decay * theta
    return theta

theta_after = pure_decay(1.0, 10000)
expected = (1 - 1e-3 * 0.01) ** 10000
assert abs(theta_after - expected) < 1e-12
```

如果把这个思路写成伪代码，核心就是两行：

```python
param -= lr * m_hat / (sqrt(v_hat) + eps)
param -= lr * weight_decay * param_old_or_param
```

在主流框架里，通常不手写 `step()`，而是通过参数分组告诉优化器“哪些参数衰减，哪些不衰减”。下面是新手更常见的 PyTorch 写法：

```python
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer_grouped_parameters = [
    {"params": decay_params, "weight_decay": 0.01, "lr": 2e-5},
    {"params": no_decay_params, "weight_decay": 0.0, "lr": 2e-5},
]

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

参数组装时，常见字段可以这样理解：

| 字段 | 含义 | 常见设置 |
|---|---|---|
| `lr` | 学习率 | `2e-5` 到 `3e-4` |
| `weight_decay` | 该组参数的衰减强度 | `0.0`、`0.01`、`0.1` |
| `params` | 这一组包含哪些张量 | decay / no_decay 分组 |
| `exclude` | 某些框架里的排除规则 | bias、LayerNorm、embedding |

真实工程例子：做 BERT 分类微调时，通常会让线性层权重参与衰减，但把 `bias` 和 `LayerNorm.weight` 排除；做 LLM 预训练时，很多实现还会把 embedding 一并排除，因为这些参数的尺度约束方式与普通线性层不同，粗暴衰减可能破坏训练稳定性。

---

## 工程权衡与常见坑

AdamW 在工程上已经非常成熟，但它不是“开了就万事大吉”。真正容易出问题的是参数分组、训练时长、以及对超参数语义的误解。

先给一个常见超参表：

| 场景 | 学习率 | weight decay | 梯度裁剪 | 备注 |
|---|---|---|---|---|
| BERT 微调 | `2e-5` 到 `5e-5` | `0.01` | `1.0` | 常排除 bias 和 LayerNorm |
| GPT/Transformer 中等规模训练 | `1e-4` 到 `3e-4` | `0.01` 到 `0.1` | `1.0` | AdamW 标准配置 |
| 大模型预训练 | `1e-4` 量级 | `0.1` 常见 | `1.0` | 需配合 warmup 和 lr schedule |
| 小模型快速实验 | `1e-3` 左右 | `0` 到 `0.01` | 可选 | 先验证任务是否收敛 |

常见坑主要有五类。

第一，忘记参数分组。  
如果你把所有参数一股脑交给优化器做衰减，bias 和 LayerNorm 也会被缩小。LayerNorm 是归一化层，白话说是专门调节激活尺度的参数；它的权重本来就承担“保持数值稳定”的职责，强行衰减往往会伤害表现。bias 也类似，它不是表达复杂模式的主力，衰减收益通常很低。

第二，把“AdamW”名字用上了，但代码路径里其实还是耦合实现。  
不少旧代码会写 `optimizer=Adam(...)`，同时在 loss 里再手动加 L2 项。这样得到的不是 AdamW，而是 Adam + L2。若你目标是解耦权重衰减，就不应该再手动把 $\lambda\|\theta\|^2$ 加回 loss。

第三，忽略训练长度对衰减总量的影响。  
每步衰减因子是 $(1-\eta\lambda)$，总训练越长，累计收缩越强。也就是说，即使 $\lambda$ 不变，训练 1 万步和 10 万步的总衰减效果也完全不同。实践里常说要关注 $\eta\lambda$ 的时间尺度，本质上就是不要只盯着单步超参数，还要看总步数。

第四，把 embedding 也默认衰减。  
这不是绝对错误，但需要明确判断。词向量层存储的是离散 token 的表示，白话说是“每个词对应的一张参数表”。在很多 Transformer 训练实践中，embedding 不做衰减更稳，尤其是预训练或词表较大时。

第五，只调学习率，不调 weight decay。  
AdamW 把两者语义拆开了，但这不等于两者完全独立。学习率决定你沿梯度走多快，weight decay 决定你往 0 拉多狠。任务变了、步数变了、batch size 变了，这两个量通常都要重新看。

一个真实工程经验是：Transformer 微调常见配置为 `lr=2e-5`、`weight_decay=0.01`、`max_grad_norm=1.0`。如果你发现训练集继续下降但验证集变差，优先检查三件事：
- 是否真的用了 AdamW 而不是 Adam + L2
- 是否正确排除了 bias 和 LayerNorm
- 训练总步数变长后，weight decay 是否也该重新调整

---

## 替代方案与适用边界

AdamW 很常用，但不是唯一答案。更准确地说，它是“自适应优化器 + 可解释权重衰减”这一组合里的默认基线。

下面把几种常见方案并列：

| Optimizer | weight decay 处理 | 适用场景 |
|---|---|---|
| Adam | 常与 L2 耦合进梯度 | 旧代码、快速实验，不建议作为新项目默认 |
| AdamW | 与自适应梯度解耦 | Transformer、BERT、GPT、T5、LLaMA 等主流训练 |
| SGDW / SGD + decay | 在 SGD 框架下做独立衰减 | 小模型、视觉任务、大 batch、强调泛化时 |
| LAMB | 在 AdamW 基础上再做层级信赖比率缩放 | 超大 batch、大模型训练 |

三者更新思想可以简化成：

1. Adam + L2  
   先把 $\lambda\theta$ 加进梯度，再整体做自适应缩放。

2. AdamW  
   任务梯度走 Adam，自身参数再额外衰减一次。

3. SGDW  
   不做 Adam 那种按维度自适应缩放，而是用 SGD 的更新，再独立做衰减。

如果模型不大、batch 很大、训练已经很稳定，SGD 系列仍然有竞争力，特别是在一些视觉任务中，SGD 的泛化有时更强。但在现代 NLP、LLM、Transformer 体系里，AdamW 基本已经成为默认选择，因为它兼顾了：
- Adam 的优化稳定性
- weight decay 的可解释性
- 对大规模参数训练的成熟工程经验

适用边界也要明确：

1. AdamW 不能替代学习率调度。warmup、cosine decay 这类调度仍然重要。
2. AdamW 不能自动修复糟糕的数据、错误的标签或不稳定的混合精度设置。
3. 若训练使用极大 batch，LAMB 之类方法可能更合适。
4. 若你刻意要把正则项并入目标函数进行理论分析，那么 L2 penalty 和 decoupled decay 不是同一个对象，不能混着讲。

结论很简单：如果你在训练现代 Transformer，又没有特别强的反例，先用 AdamW，而不是 Adam + L2。

---

## 参考资料

1. Loshchilov, Ilya; Hutter, Frank. *Decoupled Weight Decay Regularization*. ICLR 2019 / arXiv.  
说明：提出 AdamW 的原始论文，核心贡献是证明在 Adam 中应将 weight decay 与梯度更新解耦。

2. Hugging Face Transformers Documentation, *AdamWeightDecay / optimizer schedules*.  
说明：给出工程实现方式，尤其是参数分组、排除 `bias` 与 `LayerNorm` 的常见做法。

3. Metric Coders, *AdamW: The Gold Standard Optimizer for Training LLMs*.  
说明：从大模型工程实践角度解释 AdamW 为什么成为 LLM 训练默认优化器。

4. Michael Brenndoerfer, *AdamW Optimizer: Decoupled Weight Decay*.  
说明：对乘性衰减因子和长期累计效果有直观解释，适合理解 $1-\eta\lambda$ 的时间尺度。

5. Reiase 等技术博客关于 AdamW 的解析文章。  
说明：通常会补充“为什么 LayerNorm、bias、embedding 往往不做衰减”的实现经验。

6. Cross Validated 等社区讨论。  
说明：适合补充阅读工程争议点，例如 embedding 是否应该排除、不同模型结构下的经验差异。

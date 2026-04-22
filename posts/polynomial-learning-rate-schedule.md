## 核心结论

Polynomial 学习率调度是一种按幂函数平滑降低学习率的方法。它不是每隔几轮突然下降，也不是每一步固定乘一个比例，而是根据训练还剩多少进度来计算当前学习率。

标准公式是：

$$
\eta(t)=\eta_{\text{end}}+(\eta_0-\eta_{\text{end}})\left(1-\frac{t}{T}\right)^p,\quad 0\le t\le T
$$

$$
\eta(t)=\eta_{\text{end}},\quad t>T
$$

其中 $\eta(t)$ 表示第 $t$ 步的学习率，$\eta_0$ 是初始学习率，$\eta_{\text{end}}$ 是终止学习率，$T$ 是总衰减步数，$p$ 是 `power`。

`power` 决定曲线形状。需要特别注意：在上面这个标准公式下，`power=1` 是线性衰减；`power>1` 会让学习率在前中期更低，后期更平；`power<1` 会让学习率前期下降更慢，后期下降更陡。也就是说，不能只凭“power 更大”判断它一定“保留更久的高学习率”，必须看具体公式。

| `power` | 曲线特征 | 适合场景 |
|---|---|---|
| `1` | 线性下降 | 需要稳定、可解释的默认方案 |
| `>1` | 前中期更快降到低学习率，后期更平 | 希望较早降低更新幅度的训练 |
| `<1` | 前期保留更高学习率，后期下降更快 | 希望前期探索更久、后期快速收尾 |

新手版理解：训练快结束时，学习率通常要变小，否则模型参数还在大幅跳动，很难稳定收敛。Polynomial decay 的作用是给这条下降曲线一个可控形状。

具体版：同样从 `0.1` 降到 `0.01`，使用标准公式时，`power=2` 在中段会比 `power=1` 得到更小的学习率，而不是更大。

---

## 问题定义与边界

Polynomial decay 解决的问题是：如何在训练后期平滑降低学习率，同时让学习率下降过程可控。

学习率是优化器每次更新参数时的步长。步长太小，模型学得慢；步长一直太大，后期容易在较优区域附近震荡。Polynomial decay 的目标是在两者之间给出一条连续曲线：前期允许模型更新，后期逐渐减小更新幅度。

本文只讨论学习率调度器本身。调度器是控制学习率随训练进度变化的规则。它不等价于优化器，也不覆盖动量、权重衰减、梯度裁剪、混合精度、batch size 调整等训练策略。

| 参数 | 含义 |
|---|---|
| $\eta_0$ | 初始学习率 |
| $\eta_{\text{end}}$ | 终止学习率 |
| $T$ | 衰减总步数 |
| $p$ | 幂指数，决定曲线形状 |
| $t$ | 当前训练进度，通常是 step |

反例也要分清。Polynomial decay 不是“每个 epoch 乘以 0.9”的固定倍率衰减；那更接近 exponential decay。它也不是“训练到第 30、60、90 轮突然降一截”的阶梯策略；那是 step decay。Polynomial decay 的关键是连续、平滑、由幂函数控制。

---

## 核心机制与推导

Polynomial decay 的核心是“剩余进度的幂次缩放”。

令：

$$
r=1-\frac{t}{T}
$$

$r$ 表示剩余训练比例。训练刚开始时，$t=0$，所以 $r=1$；训练结束时，$t=T$，所以 $r=0$。学习率中真正参与变化的是：

$$
r^p=\left(1-\frac{t}{T}\right)^p
$$

当 `power=1` 时：

$$
\eta(t)=\eta_{\text{end}}+(\eta_0-\eta_{\text{end}})\left(1-\frac{t}{T}\right)
$$

这就是从 $\eta_0$ 到 $\eta_{\text{end}}$ 的线性插值，所以 `power=1` 退化为线性衰减。

玩具例子：假设总训练步数是 10，当前已经走到第 5 步，那么剩余进度是 0.5。设 $\eta_0=0.1$，$\eta_{\text{end}}=0.01$，$T=10$。

当 `p=2`：

$$
\eta(5)=0.01+(0.1-0.01)\times(1-0.5)^2=0.0325
$$

当 `p=1`：

$$
\eta(5)=0.01+(0.1-0.01)\times0.5=0.055
$$

所以在标准公式下，`p=2` 的中段学习率低于 `p=1`。这说明 `power` 不是一个“越大越慢衰减”的旋钮，而是改变曲线弯曲方式的参数。

| `t/T` | `p=1` | `p=2` | 直观解释 |
|---|---:|---:|---|
| 0.2 | 较高 | 更低 | `p=2` 前期下降更明显 |
| 0.5 | 中等 | 更低 | 中段差距变大 |
| 0.8 | 较低 | 接近终止值 | 后期进入低学习率区域 |

真实工程例子：在 ResNet 这类视觉模型训练中，论文或代码配置里经常能看到 “warmup + polynomial decay”。warmup 是训练初期把学习率从很小的值逐步升高，用来降低训练刚开始的不稳定性。某些 ResNet50 训练配置会使用 `power=2`，这意味着模型在 warmup 后较快进入更低学习率区间。Transformer 或 BERT 生态里更常见的是 `power=1`，也就是 warmup 后线性下降，因为它简单、稳定、容易复现实验设置。

---

## 代码实现

实现 Polynomial decay 时，最重要的是先确定单位：按 step 更新，还是按 epoch 更新。step 是一次参数更新；epoch 是完整遍历一遍训练集。大多数深度学习框架的调度器按 step 更自然，因为学习率变化应该和优化器更新次数对齐。

下面是一个可运行的 Python 实现：

```python
def polynomial_decay(step, eta_0, eta_end, total_steps, power):
    if step >= total_steps:
        return eta_end
    progress = 1.0 - step / total_steps
    return eta_end + (eta_0 - eta_end) * (progress ** power)


assert abs(polynomial_decay(0, 0.1, 0.01, 10, 2) - 0.1) < 1e-12
assert abs(polynomial_decay(5, 0.1, 0.01, 10, 2) - 0.0325) < 1e-12
assert abs(polynomial_decay(5, 0.1, 0.01, 10, 1) - 0.055) < 1e-12
assert abs(polynomial_decay(10, 0.1, 0.01, 10, 2) - 0.01) < 1e-12
assert abs(polynomial_decay(20, 0.1, 0.01, 10, 2) - 0.01) < 1e-12
```

PyTorch 风格的训练循环可以写成：

```python
import torch

model = torch.nn.Linear(4, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

total_steps = 1000
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=total_steps,
    power=1.0,
)

for step in range(total_steps):
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

TensorFlow / Keras 风格通常直接把 schedule 传给 optimizer：

```python
import tensorflow as tf

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    end_learning_rate=0.01,
    power=1.0,
    cycle=False,
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

`cycle=False` 表示超过 `decay_steps` 后保持 `end_learning_rate`。`cycle=True` 表示超过设定步数后扩展或重算周期，不适合所有训练；如果没有明确需要，默认保持 `False` 更容易控制结果。

---

## 工程权衡与常见坑

最常见的问题不是公式写错，而是训练粒度、warmup 边界、终止学习率和 `power` 取值没有统一设计。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| step 和 epoch 混用 | 曲线形状错误 | 统一按训练步数实现 |
| warmup 边界不连续 | 学习率突变 | 让 warmup 末值和 decay 起点一致 |
| $\eta_{\text{end}}$ 过低 | 后期几乎不学习 | 按任务设定合理下限 |
| `power` 盲目变大 | 过早进入低学习率，可能欠拟合 | 做小范围搜索 |
| `cycle` 误用 | 训练结束后学习率行为不符合预期 | 默认保持 `False` |
| 多个调度器叠加 | 出现双重衰减 | 明确只有一个组件控制学习率 |

新手常见误区是把 `end_learning_rate` 设成 0。数学上可以，但工程上不一定合理。如果训练最后 20% 的学习率已经接近 0，参数几乎不再变化，等价于后面的训练时间利用率很低。

另一个常见问题是 warmup 和 decay 接不上。比如 warmup 最后一刻学习率是 `0.05`，但 Polynomial decay 起点是 `0.1`，中间就会突然跳高。大模型、大 batch 或不稳定任务中，这种跳变可能带来 loss 震荡。

真实工程里，`power` 没有统一最优值。ResNet 训练中可以尝试 `1`、`1.5`、`2`；Transformer 训练中通常先从线性衰减开始，也就是 `power=1`。如果验证集指标对学习率很敏感，优先固定总步数、warmup 步数和终止学习率，再单独搜索 `power`，否则很难判断到底是哪一个因素起作用。

---

## 替代方案与适用边界

Polynomial decay 适合需要连续、平滑、可控下降的训练。它不是唯一方案，也不是所有任务的默认最优解。

| 调度器 | 特点 | 优势 | 局限 |
|---|---|---|---|
| Polynomial decay | 幂函数平滑下降 | 可控、直观 | 需要调 `power` |
| Cosine decay | 余弦曲线平滑收尾 | 常见、效果稳 | 曲线含义不如线性直观 |
| Step decay | 分段下降 | 简单直接 | 跳变明显 |
| Exponential decay | 固定倍率下降 | 计算简单 | 后期可能降太快 |

如果你只想要一个“平滑往下走”的学习率，Polynomial decay 是合理选择。如果你希望默认方案简单、论文复现清楚，`power=1` 的线性衰减通常更稳。如果你更看重周期性重启、长训练中的重新探索，cosine schedule with restarts 可能更合适。如果你只需要非常直接地在几个固定节点降低学习率，step decay 也能满足需求。

适用边界可以这样判断：当你的主要需求是“从初始学习率稳定降到一个终止学习率，并且希望用一个参数控制曲线形状”时，选择 Polynomial decay；当你不想额外调 `power`，或者已有社区标准配置时，优先跟随任务生态中的常用调度器。

---

## 参考资料

1. [TensorFlow PolynomialDecay 官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay)
2. [PyTorch PolynomialLR 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html)
3. [Hugging Face Transformers Optimization Schedules](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules)
4. [Google Research BERT optimization.py](https://github.com/google-research/bert/blob/master/optimization.py)
5. [Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC](https://openaccess.thecvf.com/content/CVPR2022/papers/An_Killing_Two_Birds_With_One_Stone_Efficient_and_Robust_Training_CVPR_2022_paper.pdf)

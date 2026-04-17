## 核心结论

梯度下降的三种常见形态，本质区别只有一个：**每次更新参数时，拿多少样本来估计梯度**。梯度可以理解为“参数该往哪个方向改，损失才会下降”的方向信息。

- **Full-batch Gradient Descent**：每次用全部 $N$ 个样本计算一次精确梯度，方向最稳定，但单步计算最重，内存压力最大，而且过于“平滑”时不容易借助随机扰动离开鞍点。
- **SGD**：每次只用 1 个样本，更新最频繁，梯度方差最大。方差就是“同一个真实方向附近的随机抖动幅度”。这种抖动会让优化轨迹更随机，常能帮助跳出局部不良区域，但后期收敛会持续震荡。
- **Mini-batch GD**：每次用 $B$ 个样本，通常是 32、64、128、256。它在稳定性、吞吐、显存占用之间做折中，因此是现代深度学习的默认选择。

统一更新公式是：

$$
\theta_{t+1}=\theta_t-\eta \cdot \frac{1}{B}\sum_{i\in \mathcal{B}_t}\nabla_\theta \ell(\theta_t;x_i,y_i)
$$

其中 $\theta$ 是模型参数，$\eta$ 是学习率，$\mathcal{B}_t$ 是当前 batch。

一个核心规律是：**batch size 越大，梯度噪声越小**。若样本梯度近似独立，则 batch 平均梯度的方差大约满足：

$$
\mathrm{Var}(\hat g_B)\propto \frac{1}{B}, \qquad \sigma_{\text{noise}}\propto \frac{1}{\sqrt{B}}
$$

这直接带来线性缩放经验法则：

$$
\eta_{\text{new}}=\eta_{\text{ref}}\cdot \frac{B_{\text{new}}}{B_{\text{ref}}}
$$

它的意思是：当 batch size 增大 $k$ 倍时，学习率也尝试增大 $k$ 倍，以补偿噪声下降后的“探索能力”损失。但这条规则只在一定范围内有效，通常写作 $B < B_{\text{critical}}$，也就是还没超过任务的临界批量。

| 方法 | batch size | 梯度噪声 | 单步成本 | 收敛轨迹 | 典型问题 |
| --- | ---: | --- | --- | --- | --- |
| SGD | 1 | 最高 | 最低 | 抖动大 | 后期难稳定 |
| Mini-batch | 32-512 常见 | 中等 | 中等 | 稳定与探索兼顾 | 需调 batch 和学习率 |
| Full-batch | $N$ | 最低 | 最高 | 最平滑 | 吞吐低，易停在坏区域附近 |

---

## 问题定义与边界

讨论对象是监督学习中的经验风险最小化：

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(\theta;x_i,y_i)
$$

目标是找到一组参数 $\theta$，让平均损失 $L(\theta)$ 尽量小。

这里的“边界”有三层。

第一层是**算法边界**。本文只讨论同一类一阶优化方法，也就是只用梯度，不直接计算 Hessian。Hessian 可以理解为“曲率矩阵”，描述损失面在各方向上有多陡。

第二层是**数据边界**。默认训练样本经过随机打乱，并且 batch 是从总体中近似独立采样。如果数据没打乱，例如前 1000 条全是猫、后 1000 条全是狗，那么无论是 SGD 还是 Mini-batch，梯度都会带偏。

第三层是**工程边界**。本文主要解释 batch size 对收敛、噪声、吞吐和泛化的影响，不展开 Adam、LAMB、二阶优化器的完整推导，只在“替代方案”中说明何时该换工具。

一个最小玩具例子可以帮助建立直觉。设我们要拟合一条直线 $y=w x$，数据只有 4 个点：

| 样本 | $x$ | $y$ |
| --- | ---: | ---: |
| 1 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 6 |
| 4 | 4 | 8 |

- Full-batch：每次都看 4 个点，算出的梯度方向稳定。
- SGD：可能这一步只看到点 $(1,2)$，下一步只看到点 $(4,8)$，两个样本的梯度大小和方向会明显不同。
- Mini-batch：比如每次看 2 个点，单步不如 Full-batch 精确，但比单样本更稳定。

所以问题不是“哪一种绝对更好”，而是：**在给定数据规模、硬件吞吐、模型曲率和目标泛化性能下，选多大的 batch 最合适**。

---

## 核心机制与推导

先看为什么 batch 会改变噪声。

设单样本梯度记作 $g_i=\nabla_\theta \ell(\theta;x_i,y_i)$，总体真实梯度记作：

$$
g=\mathbb{E}[g_i]
$$

若我们用一个大小为 $B$ 的 batch 求平均梯度：

$$
\hat g_B=\frac{1}{B}\sum_{i=1}^{B} g_i
$$

在独立同分布近似下，有：

$$
\mathbb{E}[\hat g_B]=g
$$

这说明 batch 平均梯度是无偏的，意思是“长期平均来看，方向没偏”。

再看方差：

$$
\mathrm{Var}(\hat g_B)=\frac{1}{B^2}\sum_{i=1}^{B}\mathrm{Var}(g_i)
\approx \frac{\sigma^2}{B}
$$

因此标准差满足：

$$
\sigma(\hat g_B)\approx \frac{\sigma}{\sqrt{B}}
$$

这就是常说的“batch 增大，噪声按 $1/\sqrt{B}$ 下降”。

### 玩具例子：为什么 batch 变大后常要增大学习率

假设参考配置是：

- $B_{\text{ref}}=128$
- $\eta_{\text{ref}}=0.02$

现在把 batch size 提高到 512，也就是增大 4 倍：

$$
\frac{B_{\text{new}}}{B_{\text{ref}}}=4
$$

噪声幅度会大约缩小为：

$$
\frac{1}{\sqrt{4}}=\frac{1}{2}
$$

如果学习率不变，参数每一步受到的随机探索会更弱，训练轨迹更容易变得“过于保守”。因此常用线性缩放规则：

$$
\eta_{\text{new}}=0.02\times 4=0.08
$$

这不是数学定理，而是工程经验：**当 batch 变大导致梯度更平滑时，适当放大学习率，可以让每次参数更新仍保持足够的推进速度**。

### 为什么 Full-batch 不一定最好

直觉上，精确梯度似乎最合理，但深度网络的损失面通常非凸。非凸的意思是“不是一个光滑大碗”，而是有平台、狭长谷地、鞍点和多个局部区域。

- 在鞍点附近，真实梯度可能很小，Full-batch 会稳定地停留在这里附近。
- SGD 和小 batch 的随机噪声，相当于给参数增加扰动，更容易跨过窄而浅的不良区域。

这也是为什么“更精确”不等于“泛化更好”。训练优化不仅是数值下降问题，也是“落到哪类极小值附近”的问题。

### 真实工程例子：ResNet-50 训练中的批量临界值

在 ImageNet 上训练 ResNet-50 这类模型时，工程上常观察到一个现象：当 batch 从 64、128、256 增大时，吞吐明显提升，训练也更稳定；但继续增到 1024、2048 甚至更大后，验证集收益开始变差，必须配合 warm-up、学习率衰减，甚至更适合大批量的优化器策略。

可以把这个现象理解为存在一个经验上的 $B_{\text{critical}}$：

- 当 $B < B_{\text{critical}}$ 时，增大 batch 主要是在降低无效噪声，训练通常更高效。
- 当 $B > B_{\text{critical}}$ 时，再继续增大 batch，收益递减，甚至因为噪声过低而更容易收敛到尖锐极小值。

尖锐极小值可以理解为“训练集上很低，但对参数扰动很敏感”的解，通常泛化更差。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，展示三种更新方式的差别，并顺手实现线性缩放规则。例子使用一维线性回归，目标是拟合 $y=2x+1$。

```python
import random

def make_data():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [2.0 * x + 1.0 for x in xs]
    return list(zip(xs, ys))

def mse_grad(w, b, batch):
    # 对损失 (1/|B|) sum (wx+b-y)^2 求梯度
    dw = 0.0
    db = 0.0
    for x, y in batch:
        err = (w * x + b) - y
        dw += 2.0 * err * x
        db += 2.0 * err
    n = len(batch)
    return dw / n, db / n

def loss(w, b, data):
    return sum(((w * x + b) - y) ** 2 for x, y in data) / len(data)

def scale_lr(ref_lr, ref_batch, new_batch):
    return ref_lr * (new_batch / ref_batch)

def train(batch_size, steps, base_lr, ref_batch=2, seed=0):
    random.seed(seed)
    data = make_data()
    w, b = 0.0, 0.0
    lr = scale_lr(base_lr, ref_batch, batch_size)

    for _ in range(steps):
        if batch_size >= len(data):
            batch = data[:]                 # Full-batch
        else:
            batch = random.sample(data, batch_size)  # SGD 或 Mini-batch

        dw, db = mse_grad(w, b, batch)
        w -= lr * dw
        b -= lr * db

    return w, b, loss(w, b, data)

# SGD: batch_size = 1
w1, b1, l1 = train(batch_size=1, steps=200, base_lr=0.01, ref_batch=1, seed=42)

# Mini-batch: batch_size = 2
w2, b2, l2 = train(batch_size=2, steps=200, base_lr=0.01, ref_batch=2, seed=42)

# Full-batch: batch_size = N
w3, b3, l3 = train(batch_size=6, steps=200, base_lr=0.01, ref_batch=2, seed=42)

assert l1 < 1e-2
assert l2 < 1e-4
assert l3 < 1e-4
assert abs(w3 - 2.0) < 1e-2
assert abs(b3 - 1.0) < 1e-2

print("SGD:", w1, b1, l1)
print("Mini-batch:", w2, b2, l2)
print("Full-batch:", w3, b3, l3)
```

这段代码刻意保留了最核心的结构：

- `batch_size=1` 时是 SGD。
- `1 < batch_size < N` 时是 Mini-batch。
- `batch_size=N` 时是 Full-batch。
- `scale_lr` 演示了线性缩放。

如果换成 PyTorch 风格，训练循环通常会写成这样：

```python
ref_batch = 128
base_lr = 0.02

for step, batch in enumerate(dataloader):
    batch_size = batch["x"].shape[0]
    optimizer.param_groups[0]["lr"] = base_lr * (batch_size / ref_batch)

    optimizer.zero_grad()
    loss = model.compute_loss(batch)
    loss.backward()

    # 梯度裁剪：防止梯度过大导致更新发散
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

如果想估计“当前梯度噪声是否太大”，一个简单做法是记录相邻若干步梯度范数的波动：

```python
grad_norm_ema = 0.0
beta = 0.9

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model.compute_loss(batch)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    grad_norm_ema = beta * grad_norm_ema + (1 - beta) * total_norm
    optimizer.step()
```

这不是严格的梯度噪声尺度估计，但足够作为工程监控信号：如果梯度范数波动异常大，常意味着学习率过高、batch 太小，或者数据分布出现了问题。

---

## 工程权衡与常见坑

真正训练模型时，batch size 从来不是孤立参数，而是和学习率、显存、通信、归一化层、数据顺序一起工作的。

| 问题 | 现象 | 常见原因 | 规避方式 |
| --- | --- | --- | --- |
| 收敛震荡 | loss 上下跳动明显 | batch 太小或学习率太大 | 增大 batch、减小学习率、加 momentum |
| 训练发散 | loss 直接爆炸或出现 `nan` | 曲率大、梯度过大 | gradient clipping、warm-up、lr decay |
| 泛化变差 | 训练集好，验证集差 | batch 过大，噪声过低 | 不盲目增大 batch，超阈值后改用衰减策略 |
| 吞吐低 | GPU 利用率不高 | batch 太小 | 增大 Mini-batch 或做梯度累积 |
| 梯度偏移 | 某些阶段持续朝错误方向更新 | 数据未打乱或非 IID | shuffle、分层采样、检查数据管线 |

### 常见坑 1：线性缩放不是无限有效

把 batch 从 128 提到 256、512，经常还能按比例放大学习率；但继续提到很大时，训练未必更快，验证集甚至变差。原因不是公式错了，而是噪声已经小到不再提供有效探索。

经验上可以这样理解：

- 小到中等 batch：噪声主要是“无害抖动”，降一点更好。
- 超大 batch：噪声开始变成“有益正则化”，再降就可能丢掉泛化能力。

### 常见坑 2：Full-batch 可能理论整洁，工程上却不划算

Full-batch 每一步都要扫全量数据。若数据集很大，一步更新时间会非常长，参数更新频率极低。即使每一步方向更准，总训练墙钟时间也可能更差。

更现实的问题是显存和内存。很多任务根本放不下全量数据，尤其是图像、序列和多模态模型。

### 常见坑 3：未打乱数据时，SGD 的噪声不是“好噪声”

很多初学者以为 SGD 自带随机性，所以天然鲁棒。这不对。若数据管线按时间、类别或来源排序，那么 SGD 每一步看到的不是随机样本，而是系统性偏差样本。此时梯度抖动不是探索，而是偏移。

### 常见坑 4：大 batch 下忘了 warm-up

warm-up 可以理解为“前几轮先用较小学习率热启动”。因为模型初期参数还很差，若一开始就对大 batch 直接套上线性缩放后的大学习率，容易发散。

常见写法如下：

```python
# 伪代码
if epoch < warmup_epochs:
    lr = target_lr * (epoch + 1) / warmup_epochs
else:
    lr = cosine_decay(target_lr, epoch)
```

---

## 替代方案与适用边界

当直接调 batch size 已经不能解决问题时，常见替代方案有三类。

| 方法 | 解决什么问题 | 适用场景 | 局限 |
| --- | --- | --- | --- |
| Warm-up + LR decay | 大 batch 初期不稳定、后期需细收敛 | 标准视觉/NLP 训练 | 仍需手调调度器 |
| Gradient Accumulation | 显存放不下大 batch | 单卡或小显存设备 | 不能完全等价于真实大 batch 的吞吐 |
| Hessian-aware / 大批量优化器 | 超大 batch 泛化变差 | 分布式大规模训练 | 实现更复杂，调参成本高 |

### 方案 1：先小 batch warm-up，再切大 batch

一个实用流程是：

1. 前期用 batch=64，学习率较小，让模型先进入稳定下降区间。
2. 中期切到 batch=256 或 512，并按线性规则放大学习率。
3. 接近收敛时停止继续放大学习率，改用 cosine decay 或 step decay。

这比“一开始就超大 batch”更稳，因为初期模型参数远离可行区域，最怕大步长直接越界。

### 方案 2：用梯度累积代替直接增大 batch

梯度累积的意思是：连续跑多个小 batch，只在累计若干次后再更新一次参数。对白话理解，它是在“显存装不下大 batch 时，用多次小步前向/反向，攒出一个等效大 batch”。

但要注意，它和真实大 batch 并不完全等价：

- 参数更新频率更低。
- BatchNorm 等依赖当前 batch 统计量的层，行为会不同。
- 吞吐未必和真实大 batch 一样高。

### 方案 3：超过临界批量后，不再继续线性放大学习率

如果已经观察到 $B > B_{\text{critical}}$，再强行按线性缩放增大学习率，往往收益有限。更合理的做法通常是：

- 固定 batch，不再增大。
- 进入学习率衰减阶段。
- 必要时引入更适合大批量的优化器或曲率感知方法。

所以，线性缩放规则不是“batch 越大越好”的许可证，而只是一个**在有效区间内的近似调参规则**。

---

## 参考资料

1. [TensorTonic: SGD, Mini-Batch & Learning Rate Schedules](https://www.tensortonic.com/ml-math/optimization/sgd-variants?utm_source=openai)  
   重点是统一更新公式、梯度噪声与 batch size 的关系，以及 SGD、Mini-batch、Full-batch 的基础对比。

2. [ML Journey: Gradient Noise Scale and Batch Size Relationship](https://mljourney.com/gradient-noise-scale-and-batch-size-relationship/?utm_source=openai)  
   重点是梯度噪声尺度、线性缩放经验法则，以及临界批量 $B_{\text{critical}}$ 的工程解释。

3. [Rohan Paul: ML Interview Q Series](https://www.rohan-paul.com/p/ml-interview-q-series-explain-how-058?utm_source=openai)  
   重点是训练中常见的学习率、数据打乱、稳定性与泛化陷阱，适合工程排障。

4. [张俊宏读书笔记：Fundamentals of Deep Learning](https://zhangjunhd.github.io/reading-notes/ml/FundamentalsOfDeepLearning.html?utm_source=openai)  
   用较直观的方式解释了 SGD、Mini-batch、Full-batch 的差异，适合作为入门补充。

5. Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*  
   经典论文，解释为什么过大的 batch 容易落入尖锐极小值，是理解“大 batch 泛化差异”的重要起点。

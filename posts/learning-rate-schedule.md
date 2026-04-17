## 核心结论

学习率调度的核心任务，是让参数更新的步子在不同训练阶段匹配不同目标。学习率就是每次参数更新走多远的系数，值太大容易震荡，值太小又会训练过慢。对现代深度学习训练，固定学习率通常不是一个好默认选项。

Warmup、余弦退火和 OneCycleLR 可以理解为三种不同的“步长控制策略”：

| 策略 | 白话解释 | 主要目标 | 典型阶段 |
|---|---|---|---|
| Warmup | 先从小步开始，逐步把步子放大 | 稳定训练初期 | 前几百到几千步 |
| 余弦退火 | 用平滑曲线慢慢把步子缩小 | 后期细致收敛 | 中后期 |
| OneCycleLR | 先升到峰值，再持续压低到很小 | 中期更强探索，末期更强收敛 | 整个训练过程 |

这三者不是完全互斥。实际工程里最常见的是“Warmup + 后续衰减”，而后续衰减常用余弦退火。OneCycleLR 则把“先升后降”做成一个完整周期，适合总步数已知、希望快速收敛的训练任务。

Transformer 训练中最经典的学习率公式是：

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5},\ step \cdot warmup^{-1.5})
$$

这里的 $d_{model}$ 是模型隐藏维度，白话说就是每个 token 内部表示的宽度；$warmup$ 是预热步数，表示前多少步要逐步增大学习率。这个公式本质上做了两件事：前期线性升高，后期按逆平方根下降。

一个直观的玩具例子是：你刚开始跑步时不会第一秒就全速冲刺，而是先热身，再进入主训练节奏，最后放慢整理。学习率调度也是同一个逻辑，只不过对象从人体变成了模型参数。

---

## 问题定义与边界

问题定义很简单：为什么固定学习率不够用？

如果一开始学习率就很大，模型还没有形成稳定的梯度统计，参数会被大幅更新，容易出现 loss 抖动、梯度爆炸、训练不收敛。梯度就是损失函数对参数的变化方向和幅度，白话说就是“往哪改、改多少”的信号。尤其在 Adam 这类自适应优化器中，虽然它会维护一阶矩和二阶矩估计，但在前几步这些统计量本身还不稳定。

如果训练后期学习率还维持较大值，模型就很难停在较好的极小值附近。极小值可以理解为损失函数较低的位置；学习率太大时，参数会一直在低谷附近来回跳，无法做精细调整。

训练阶段和目标可以概括为下表：

| 训练阶段 | 主要风险 | 需要的学习率行为 |
|---|---|---|
| 初期 | 梯度统计不稳定、loss 抖动 | 从小到大，逐步升高 |
| 中期 | 容易卡在次优区域，探索不足 | 允许较大步长探索 |
| 后期 | 已接近较优点，过大步长会来回跳 | 平滑下降到较小值 |

边界也必须说清楚。

第一，调度器解决的是“学习率随时间如何变化”，不是“学习率初值一定怎么设”。基准学习率、batch size、优化器类型仍然要一起考虑。batch size 就是一次前向和反向传播使用多少样本，越大通常梯度噪声越小，但也可能需要重新调整 warmup 和峰值学习率。

第二，不是所有任务都需要复杂调度。如果你只训练几百步做快速实验，复杂调度带来的收益可能不明显，甚至会因为超参数更多而更难调。

第三，OneCycleLR 要求你基本知道总步数。如果训练过程会随时提前停止，或者数据流长度动态变化，它的使用边界就比普通余弦退火窄。

一个新手常见场景是：训练一个小 Transformer，把 Adam 的学习率直接设成 `1e-3`，第一个 epoch 前几十个 batch 就出现 loss 飙升。Warmup 的作用，就是让这个 `1e-3` 不是第一步就用上，而是从更小值逐步升到目标值。

---

## 核心机制与推导

先看 Transformer 经典 warmup 公式：

$$
lr = d_{model}^{-0.5} \cdot \min(step^{-0.5},\ step \cdot warmup^{-1.5})
$$

这个公式可以分成两段理解。

当 $step \le warmup$ 时，取到的是：

$$
lr = d_{model}^{-0.5} \cdot step \cdot warmup^{-1.5}
$$

这是一段线性增长。因为 $step$ 每增加 1，学习率就按固定比例升高。作用是让优化器先建立较稳定的统计量，再逐步放大更新幅度。

当 $step > warmup$ 时，取到的是：

$$
lr = d_{model}^{-0.5} \cdot step^{-0.5}
$$

这时学习率按逆平方根衰减。逆平方根衰减的特点是：前期降得比线性慢，后期也不会骤降到极小值，适合长程训练。

一个玩具数值例子：

设 $d_{model}=256$，则 $256^{-0.5}=1/16=0.0625$。再设 `warmup=10`。

- 第 1 步：  
  $$
  lr = 0.0625 \cdot \min(1,\ 1 \cdot 10^{-1.5}) \approx 0.00198
  $$
- 第 5 步：  
  $$
  lr = 0.0625 \cdot \min(5^{-0.5},\ 5 \cdot 10^{-1.5}) \approx 0.00988
  $$
- 第 10 步：达到 warmup 顶点
- 第 12 步：已经进入衰减段，  
  $$
  lr = 0.0625 \cdot 12^{-0.5} \approx 0.0180
  $$

这说明 warmup 不只是“从 0 慢慢变大”，它还规定了何时切换到衰减阶段。

再看余弦退火。它的典型公式是：

$$
lr(t)=lr_{min}+0.5(lr_{max}-lr_{min})(1+\cos(\pi t/T))
$$

这里 $T$ 是退火周期长度，白话说就是从最高点降到最低点一共走多少步。这个公式的关键点不是“它用了余弦”，而是“它提供了平滑的一阶变化”。也就是说，学习率不会像阶梯衰减那样突然掉一截，而是连续、平缓地下行。

如果设 `lr_max=0.02`、`lr_min=0.002`、`T=100`：

- 第 0 步：$lr=0.02$
- 第 50 步：余弦项变成 $0$，中点约为 $0.011$
- 第 100 步：$lr=0.002$

这类曲线在后期很常用，因为它能避免“刚收敛一点就被突然大幅降速”或者“降速过猛导致训练停滞”。

OneCycleLR 的机制可以概括成分段函数。它不是单独只做衰减，而是让学习率完成一个完整周期：

$$
lr(step)=
\begin{cases}
\text{从 } lr_{start} \text{ 上升到 } lr_{max}, & 0 \le step < s_1 \\
\text{从 } lr_{max} \text{ 下降到 } lr_{final}, & s_1 \le step \le s_2
\end{cases}
$$

在 PyTorch 默认实现里，这个上升和下降过程通常都可以用 cosine 或 linear 完成。其思想是：先把学习率推高，迫使模型更积极地探索参数空间；再迅速而平滑地压低到很小值，完成收敛。

把三者放在一张图的概念上理解：

1. Warmup：起点很低，线性爬升。
2. Cosine：从高点平滑下滑。
3. OneCycle：前段继续冲高，中后段压到更低。

真实工程例子是机器翻译或大语言模型预训练。训练一个中等规模 Transformer 时，常见配置是前几千步 warmup，之后进入余弦衰减；如果是总步数明确的监督训练任务，例如图像分类或语音分类，OneCycleLR 常被用来缩短达到较好结果的时间。

---

## 代码实现

下面先用一个纯 Python 的可运行例子计算 warmup 和 cosine 的学习率。这个例子不依赖 PyTorch，重点是把公式落实成代码。

```python
import math

def transformer_lr(step: int, d_model: int = 256, warmup: int = 10) -> float:
    assert step > 0
    assert d_model > 0
    assert warmup > 0
    scale = d_model ** (-0.5)
    return scale * min(step ** (-0.5), step * (warmup ** -1.5))

def cosine_lr(t: int, T: int, lr_max: float, lr_min: float) -> float:
    assert 0 <= t <= T
    assert lr_max >= lr_min >= 0
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / T))

lr1 = transformer_lr(1)
lr5 = transformer_lr(5)
lr10 = transformer_lr(10)
lr12 = transformer_lr(12)

assert lr1 < lr5 < lr10
assert lr12 < lr10

c0 = cosine_lr(0, 100, 0.02, 0.002)
c50 = cosine_lr(50, 100, 0.02, 0.002)
c100 = cosine_lr(100, 100, 0.02, 0.002)

assert abs(c0 - 0.02) < 1e-12
assert abs(c100 - 0.002) < 1e-12
assert c0 > c50 > c100

print(round(lr1, 6), round(lr5, 6), round(lr10, 6), round(lr12, 6))
print(round(c0, 6), round(c50, 6), round(c100, 6))
```

如果你在工程里用 PyTorch，常见写法如下。这个例子展示 OneCycleLR，重点是调用顺序和步频。

```python
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

model = nn.Linear(16, 2)
optimizer = AdamW(model.parameters(), lr=1e-4)

epochs = 3
steps_per_epoch = 20
total_steps = epochs * steps_per_epoch

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=total_steps,
    pct_start=0.1,          # 前 10% 步数用于升高学习率
    anneal_strategy="cos",  # 用余弦方式下降
    div_factor=10.0,        # 初始 lr = max_lr / div_factor
    final_div_factor=100.0  # 最终 lr 更低
)

loss_fn = nn.CrossEntropyLoss()
lrs = []

for step in range(total_steps):
    x = torch.randn(8, 16)
    y = torch.randint(0, 2, (8,))

    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    optimizer.step()
    scheduler.step()  # 按 batch 调用，不要按 epoch 调

    current_lr = optimizer.param_groups[0]["lr"]
    lrs.append(current_lr)

assert len(lrs) == total_steps
assert max(lrs) <= 1.000001e-3
assert min(lrs) > 0
print([round(v, 6) for v in lrs[:10]])
```

如果你的目标是“先 warmup，再进入普通余弦退火”，更常见的是自己用 `LambdaLR` 写 warmup，再接 `CosineAnnealingLR`，或者直接写一个统一的 lambda 函数。关键点有两个：

1. `scheduler.step()` 的频率要和设计单位一致。若按 batch 设计，就必须每个 batch 调一次。
2. `total_steps`、`steps_per_epoch`、`warmup_steps` 必须和真实训练循环对齐。

真实工程例子：训练一个 300k step 的 Transformer 翻译模型，`warmup_steps=4000`，Adam 使用 `betas=(0.9, 0.98)`。前 4000 步线性升高学习率，之后按逆平方根或余弦衰减。这类配置之所以常见，不是因为它“神奇”，而是因为它对训练初期的数值稳定性更友好。

---

## 工程权衡与常见坑

学习率调度真正难的地方，不是公式本身，而是脚本实现细节。

| 常见坑 | 现象 | 原因 | 规避办法 |
|---|---|---|---|
| `total_steps` 算错 | 峰值提早或延后出现 | OneCycle 依赖总步数 | 训练前先精确计算真实总步数 |
| `scheduler.step()` 调错频率 | 曲线形状完全变形 | 本应按 batch，结果按 epoch | 明确调度单位，统一写在训练循环里 |
| warmup 太短 | 前几步 loss 爆炸 | 学习率过快升到峰值 | batch size 变化后同步调整 warmup |
| 峰值学习率过大 | loss 抖动甚至 nan | 探索过猛 | 先做小范围学习率扫描 |
| 断点恢复后调度器状态没恢复 | 恢复训练后 lr 异常 | 只恢复了 optimizer，没恢复 scheduler | checkpoint 同时保存两者状态 |

一个典型新手坑是：原来 `batch_size=32`，warmup 设成 1000 步；后来显存优化后把 `batch_size` 提到 128，但 warmup 仍然是 1000 步。这样每一步看到的数据量大了很多，等价于“按样本数计算的 warmup 变短了”，学习率可能比预期更早到达峰值，结果就是模型还没稳定就开始大步更新。

另一个坑是调用顺序。大多数 PyTorch 优化流程应写成：

1. `loss.backward()`
2. `optimizer.step()`
3. `scheduler.step()`

如果把顺序写反，部分调度器的第一步行为会和预期不一致。

还有一个现实权衡：调度器越复杂，超参数越多。Warmup 要选步数，Cosine 要选 `T_max` 或周期长度，OneCycle 还要选 `max_lr`、`pct_start`、`div_factor`、`final_div_factor`。如果模型本身、数据清洗、损失函数还没稳定，过早精调学习率策略，收益通常不如先修正基础问题。

---

## 替代方案与适用边界

不是每个项目都应该直接上 OneCycleLR。

| 方案 | 适用场景 | 不适用场景 |
|---|---|---|
| Warmup + CosineAnnealingLR | 总步数大致已知，训练较稳定 | 需要根据验证集动态降 lr |
| OneCycleLR | 总步数明确，希望较快达到较好结果 | 总步数不稳定、频繁早停 |
| ReduceLROnPlateau | 更关注验证集指标停滞 | 每 batch 都要细粒度调度 |
| LambdaLR 手写函数 | 需要精细控制任意曲线 | 团队不熟悉调度细节，维护成本高 |
| CosineAnnealingWarmRestarts | 希望周期性重启探索 | 只想单调下降到收敛 |

如果训练总步数无法精确提前知道，OneCycleLR 的边界就比较明显。因为它必须围绕一个完整周期安排升高和下降，一旦实际训练提前结束，可能停在错误阶段；如果训练额外延长，后半段学习率也可能已经降得过低。

这时更稳妥的替代方案是：

- 手动 warmup + `ReduceLROnPlateau`：前期保证稳定，后期根据验证集自动降速。
- `LambdaLR`：自己定义 $ \lambda(step) $，适合做短期实验和调参。
- `CosineAnnealingWarmRestarts`：适合希望多次“重新探索”的任务。

一个短训练场景的玩具例子是：你只打算跑 300 步验证模型能否工作。此时 OneCycleLR 往往不划算，因为 warmup、上升、下降三个阶段都挤在很短的窗口里，曲线形状未必合理。直接使用 `LambdaLR` 手写一个“前 30 步 warmup，后 270 步线性下降”的函数，往往更可控。

结论可以简化成一句话：

- 长训练、Transformer 类模型：优先考虑 Warmup + 后续衰减。
- 总步数明确、监督任务：OneCycleLR 值得优先尝试。
- 总步数不明确、验证集主导调参：Plateau 或自定义 Lambda 更稳。

---

## 参考资料

1. PyTorch `OneCycleLR` 文档：说明参数含义、`total_steps` 约束和按 batch 调用的要求。  
   https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

2. PyTorch `CosineAnnealingLR` 文档：用于确认余弦退火公式、`T_max` 的意义和调度行为。  
   https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

3. Harvard NLP Annotated Transformer 学习率调度说明：用于理解 Transformer 中经典的 $d_{model}^{-0.5}$ 与 warmup 公式。  
   https://deepwiki.com/harvardnlp/annotated-transformer/3.2-learning-rate-scheduling

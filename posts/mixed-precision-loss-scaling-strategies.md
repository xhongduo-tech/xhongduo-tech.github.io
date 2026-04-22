## 核心结论

`Loss Scaling` 是混合精度训练中的梯度缩放策略：先把损失函数值 `L` 乘上缩放因子 `S`，让反向传播得到更大的梯度，减少 `fp16` 下小梯度下溢为 `0` 的概率；真正更新参数前，再把梯度除以 `S` 还原。

核心公式是：

$$
L' = S \cdot L
$$

$$
g' = \frac{\partial L'}{\partial \theta} = S \cdot g
$$

$$
g = \frac{g'}{S}
$$

这里的 `g` 是原始梯度，`g'` 是被放大后的梯度，`θ` 是模型参数。

新手版玩具例子：如果一个梯度本来只有 `1e-8`，它在 `fp16` 中可能直接变成 `0`。如果先把 loss 放大 `32768` 倍，反向传播得到的梯度也会放大 `32768` 倍，变成 `3.2768e-4`。这个数更容易被 `fp16` 表示。优化器更新前再除回 `32768`，梯度仍然对应原来的 `1e-8`。

工程上更常用的是**动态 Loss Scaling**。动态策略会自动寻找“足够大但不溢出”的 `S`：检测到 `inf/nan` 就跳过本次参数更新，并把 `S` 减半；连续若干步没有溢出，就按倍率增大 `S`。静态策略使用固定 `S`，开销更小，但对梯度范围的先验要求更高。

---

## 问题定义与边界

混合精度训练是指训练时同时使用不同数值精度，例如大部分矩阵计算用 `fp16`，部分关键状态和累加仍保留 `fp32`。它的目标是提升吞吐、降低显存占用，但问题不是“能不能算”，而是“小梯度会不会在 `fp16` 中消失”。

`fp16` 是 16 位浮点数，`fp32` 是 32 位浮点数。白话说，`fp16` 用更少的位数表示数字，所以范围更窄、精度更低。小到一定程度的数会被舍入成 `0`，这叫**下溢**；大到超过最大可表示值时会变成 `inf`，这叫**上溢**。

| 类型 | 位宽 | 最大有限值 | 最小正规正数 | 典型用途 |
|---|---:|---:|---:|---|
| `fp16` | 16 bit | 约 `65504` | 约 `6.10e-5` | Tensor Core 计算、激活、部分梯度 |
| `fp32` | 32 bit | 约 `3.4e38` | 约 `1.18e-38` | 权重主副本、优化器状态、关键归约 |
| `bf16` | 16 bit | 接近 `fp32` | 接近 `fp32` | 更宽动态范围的混合精度训练 |

新手版解释：`fp16` 的表示范围比 `fp32` 窄。它不是不能表示小数，而是能可靠表示的有效范围更有限。非常小的梯度进入 `fp16` 后可能直接变成 `0`，对应参数就收不到更新信号。Loss Scaling 的作用是先把这些小梯度放大，让它们在反向传播过程中不那么容易丢掉。

Loss Scaling 的边界也要明确：

| 问题 | Loss Scaling 是否主要解决 | 说明 |
|---|---|---|
| 小梯度下溢为 `0` | 是 | 这是 Loss Scaling 的核心目标 |
| 缩放后梯度超过 `fp16` 最大值 | 否，反而可能引入 | `S` 过大会导致上溢，需要检测并回退 |
| 优化器超参数不合适 | 否 | 学习率、权重衰减、调度器仍需单独调整 |
| 模型结构本身不稳定 | 否 | 例如归一化、初始化、残差尺度问题 |
| `bf16` 模型强转 `fp16` 后溢出 | 不完全解决 | `bf16` 范围更宽，转 `fp16` 可能本身不安全 |

因此，Loss Scaling 主要适用于 `fp16` 混合精度训练。对于 `bf16`，因为指数范围接近 `fp32`，通常不需要同样强度的 loss scaling。

---

## 核心机制与推导

Loss Scaling 的数学链路很短，但顺序必须准确。

设原始 loss 为 `L`，参数为 `θ`，原始梯度为：

$$
g = \frac{\partial L}{\partial \theta}
$$

训练时不直接对 `L` 做反向传播，而是构造缩放后的 loss：

$$
L' = S \cdot L
$$

由于 `S` 对参数 `θ` 来说是常数，所以：

$$
g' = \frac{\partial L'}{\partial \theta}
= \frac{\partial (S \cdot L)}{\partial \theta}
= S \cdot \frac{\partial L}{\partial \theta}
= S \cdot g
$$

这说明：loss 被放大多少倍，梯度也会被放大多少倍。优化器真正更新参数前，需要执行反缩放：

$$
g = \frac{g'}{S}
$$

数值例子 1：假设某个梯度 `g = 1e-8`。直接放进 `fp16` 时，它可能因为太小而下溢成 `0`。取 `S = 32768`，则：

$$
g' = 32768 \cdot 10^{-8} = 3.2768 \times 10^{-4}
$$

`3.2768e-4` 在 `fp16` 中更容易表示。反缩放后：

$$
g = \frac{3.2768 \times 10^{-4}}{32768} = 10^{-8}
$$

数值例子 2：如果某层梯度约为 `3.0`，同样取 `S = 32768`，则：

$$
S \cdot g = 32768 \cdot 3.0 = 98304
$$

`98304` 超过 `fp16` 最大有限值 `65504`，会产生 `inf`。这时不能继续执行参数更新，而应该降低 `S`。

动态 Loss Scaling 本质上是一个闭环控制策略。它不断试探当前训练过程能承受的最大缩放因子：太大就回退，稳定一段时间就增长。

| 事件 | 动作 | 目的 |
|---|---|---|
| 检测到 `inf/nan` | 跳过 `optimizer.step()` | 避免坏梯度写入权重 |
| 检测到 `inf/nan` | `S ← S / 2` | 降低下一步溢出概率 |
| 连续若干步无溢出 | `S ← S · growth_factor` | 尽量保留更小梯度 |
| 当前步正常 | 先反缩放再更新 | 让优化器看到真实梯度尺度 |

常见默认参数如下：

| 参数 | 常见值 | 含义 |
|---|---:|---|
| `init_scale` | `65536` | 初始缩放因子，即 $2^{16}$ |
| `growth_factor` | `2` | 稳定后放大 `S` 的倍率 |
| `backoff_factor` | `0.5` | 溢出后缩小 `S` 的倍率 |
| `growth_interval` | `2000` | 连续多少步稳定后增长 |

真实工程例子：训练 Transformer 或 LLM 时，不同层、不同训练阶段的梯度范围会明显波动。训练早期 loss 下降快，梯度可能偏大；训练中后期很多梯度变小，容易下溢。动态 Loss Scaling 能在这些阶段自动调整 `S`，通常比手动固定一个值更稳。

---

## 代码实现

PyTorch 中，标准写法是 `autocast + GradScaler`。`autocast` 负责自动选择部分算子使用低精度执行；`GradScaler` 负责 loss 缩放、梯度反缩放、溢出检测、跳过更新和更新缩放因子。

典型训练循环如下：

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda", init_scale=65536) if device == "cuda" else None
loss_fn = nn.MSELoss()

x = torch.randn(8, 4, device=device)
y = torch.randn(8, 1, device=device)

optimizer.zero_grad(set_to_none=True)

if device == "cuda":
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = model(x)
        loss = loss_fn(pred, y)

    scaler.scale(loss).backward()

    # 如果要做梯度裁剪，必须先反缩放。
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
else:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

assert torch.isfinite(loss).item()
```

顺序重点如下：

| 步骤 | 操作 | 说明 |
|---:|---|---|
| 1 | `optimizer.zero_grad()` | 清空上一轮梯度 |
| 2 | `with autocast(...)` | 混合精度前向 |
| 3 | `scaler.scale(loss).backward()` | 对缩放后的 loss 反传 |
| 4 | `scaler.unscale_(optimizer)` | 把梯度除回真实尺度 |
| 5 | `clip_grad_norm_` | 在真实梯度上做裁剪 |
| 6 | `scaler.step(optimizer)` | 无溢出才执行更新 |
| 7 | `scaler.update()` | 根据本轮结果调整 `S` |

带梯度裁剪时，`unscale_` 的位置非常关键。梯度裁剪是限制真实梯度范数，如果先裁剪再反缩放，裁剪看到的是被放大后的梯度，阈值会失真。

梯度累积场景下，多个 micro-batch 合成一个有效 batch。此时同一个有效 batch 内必须共享同一个 scale，不能中途 `update()`。

| 阶段 | 操作 | 是否调用 `update()` |
|---|---|---|
| micro-batch 1 | `scale(loss / accum_steps).backward()` | 否 |
| micro-batch 2 | `scale(loss / accum_steps).backward()` | 否 |
| micro-batch N | `scale(loss / accum_steps).backward()` | 否 |
| 有效 batch 结束 | `unscale_`、裁剪、`step`、`update` | 是 |

原因很直接：梯度累积是把多次 backward 的梯度加在一起。如果累积到一半改变 `S`，不同 micro-batch 的梯度就来自不同缩放尺度，最后无法得到明确的真实梯度。

---

## 工程权衡与常见坑

Loss Scaling 最容易出错的地方不是公式，而是训练循环顺序。

| 常见坑 | 错误做法 | 正确做法 |
|---|---|---|
| 错误顺序 | 先裁剪再 `unscale_` | 先 `unscale_` 再裁剪 |
| 中途修改 `S` | 梯度累积到一半调用 `update()` | 有效 batch 结束后再 `update()` |
| 跳过检测 | 出现 `inf/nan` 仍然 `optimizer.step()` | 检测到溢出就跳过更新 |
| 误解 scale 范围 | 认为 `S` 一定大于 `1` | 动态策略可能把 `S` 降到 `1` 以下 |
| dtype 误用 | 把 `bf16` 预训练模型直接转 `fp16` | 先确认数值范围和溢出风险 |

新手版错误示例 1：先裁剪再 `unscale`。

假设真实梯度范数是 `0.5`，裁剪阈值是 `1.0`，本来不应该裁剪。但如果 `S = 32768`，裁剪前看到的范数会变成 `16384`。这时裁剪会强行缩小梯度，导致更新幅度远小于预期。

新手版错误示例 2：梯度累积到一半就 `update()`。

假设一个有效 batch 由 4 个 micro-batch 组成。前 2 个 micro-batch 用 `S = 65536`，中途调用 `update()` 后，后 2 个 micro-batch 改成 `S = 32768`。这些梯度被累加到同一组 `.grad` 里，但它们的缩放来源不同，最终反缩放不再对应同一个数学表达式。

另一个容易忽视的点是：`GradScaler` 允许 scale 降到 `1` 以下。这不是 bug，而是为了继续尝试找到可运行的尺度。官方也明确说明，某些模型在 `fp16` 范围内并不天然安全，尤其是从 `bf16` 训练得到的模型直接转成 `fp16` 时，可能更容易产生溢出。

工程上可以采用以下判断：

| 现象 | 优先检查 |
|---|---|
| loss 突然变成 `nan` | 是否有梯度溢出、学习率是否过大 |
| scale 一直下降 | 模型或输入是否产生过大激活/梯度 |
| 训练很慢但稳定 | 是否频繁跳过 `optimizer.step()` |
| 梯度裁剪后效果异常 | 是否在 `unscale_` 之前裁剪 |
| 梯度累积结果不稳定 | 是否在累积中途 `step/update` |

---

## 替代方案与适用边界

Loss Scaling 有两类主流策略：静态 scaling 和动态 scaling。

静态 scaling 是固定使用一个 `S`，例如始终使用 `32768` 或 `65536`。它的优点是实现简单、状态少、额外判断少；缺点是不自适应。如果梯度范围变化明显，固定 `S` 要么太小，无法充分减少下溢；要么太大，容易造成上溢。

动态 scaling 是根据训练过程自动调整 `S`。它的优点是鲁棒性更强，适合 Transformer、LLM、扩散模型等梯度波动明显的任务；缺点是要维护 scale 状态，并且需要做 `inf/nan` 检测。

| 策略 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 静态 Loss Scaling | 简单、开销小、行为可预测 | 依赖先验，不适合强波动 | 梯度分布稳定的小模型或固定任务 |
| 动态 Loss Scaling | 更稳，能自动回退和增长 | 状态更多，有检测开销 | 大模型、深层网络、不确定任务 |
| 不使用 Loss Scaling | 最简单 | `fp16` 小梯度容易丢失 | `bf16` 训练或已确认无需缩放的场景 |

适用场景判断：

| 条件 | 推荐策略 |
|---|---|
| 梯度范围长期稳定 | 静态 scaling |
| 梯度波动大或没有先验 | 动态 scaling |
| 追求最小实现复杂度 | 静态 scaling |
| 追求训练鲁棒性 | 动态 scaling |
| 使用 `bf16` 混合精度 | 通常不需要传统 fp16 loss scaling |
| 训练 LLM / Transformer | 优先动态 scaling |

新手版判断：如果你已经知道某个模型训练时梯度范围长期稳定，固定 `S` 可能就够了。例如一个小型 CNN 在固定数据集上训练，多次实验都没有溢出，静态 scale 可以减少一些控制逻辑。相反，如果模型很深、层数多、训练阶段变化明显，例如 Transformer 或 LLM，优先使用动态策略。

真实工程里，推荐默认从动态 Loss Scaling 开始。只有在已经观察到梯度范围稳定、性能瓶颈明确、并且希望减少控制逻辑时，再考虑静态 scaling。推荐初始 `S` 通常取 $2^{15}$ 到 $2^{16}$，也就是 `32768` 到 `65536`。遇到 `inf/nan` 时减半，是简单且常用的回退策略。

---

## 参考资料

1. [NVIDIA: Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)  
   用途：解释混合精度训练的工程流程、fp16 数值范围、loss scaling 的动机和实践方法，适合作为机制和工程建议的主要来源。

2. [PyTorch: Automatic Mixed Precision package - torch.amp](https://docs.pytorch.org/docs/2.9/amp.html)  
   用途：确认 `torch.amp.autocast`、`GradScaler`、默认参数、scale 增长和回退行为，适合作为代码实现与 API 行为的依据。

3. [PyTorch: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)  
   用途：确认 `scaler.scale(loss).backward()`、`scaler.unscale_(optimizer)`、梯度裁剪、梯度累积、`step()` 和 `update()` 的正确顺序。

4. [Micikevicius et al.: Mixed Precision Training](https://arxiv.org/abs/1710.03740)  
   用途：理解混合精度训练和 loss scaling 的理论背景，适合作为公式推导和数值机制的论文来源。

| 正文位置 | 建议引用 |
|---|---|
| `fp16` 范围与下溢问题 | NVIDIA 文档、Mixed Precision Training 论文 |
| `L' = S·L` 与反缩放机制 | Mixed Precision Training 论文 |
| `GradScaler` 默认参数 | PyTorch AMP 文档 |
| `unscale_`、梯度裁剪、梯度累积顺序 | PyTorch AMP examples |
| 动态 scaling 的溢出检测和回退 | PyTorch AMP 文档、NVIDIA 文档 |

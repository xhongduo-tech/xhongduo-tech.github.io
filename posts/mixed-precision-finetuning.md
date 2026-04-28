## 核心结论

混合精度微调的定义很直接：训练时把大多数前向计算和反向计算放到 `FP16` 或 `BF16`，把主权重、优化器状态以及少数数值敏感操作保留在 `FP32`。这样做的目标不是“统一降成 16 位”，而是在显存、速度、稳定性之间取一个工程上可用的平衡。

对初学者来说，可以先记住两个结论。

第一，`FP16` 和 `BF16` 都是 16 位浮点数，但不是一回事。`FP16` 的动态范围更小。动态范围的白话解释是：一个数值格式能同时表示多大和多小的数。如果范围太小，梯度这种很小的数就可能直接变成 0，这叫下溢。`BF16` 的动态范围接近 `FP32`，所以大模型训练时通常更稳。

第二，混合精度训练的核心收益通常来自三件事：显存更低、吞吐更高、可训练更大的 batch 或更大的模型。但代价也明确：数值稳定性管理更复杂，尤其是 `FP16` 常常需要 `loss scaling`，也就是先把损失放大，再把梯度缩回去，避免梯度太小直接丢失。

一个玩具例子足够说明直觉。假设你在训练一个非常小的线性模型，真实梯度只有 `1e-8`。如果直接用 `FP16` 表示，这个梯度可能被冲成 `0`，这一步就等于没更新；如果先把 loss 乘以 `1024`，梯度也会同步放大为 `1.024e-5`，这时 `FP16` 更有机会保住它，最后更新前再除回去。混合精度训练并没有改变数学目标，它只是改变了“中间数值怎么存、怎么算”。

真实工程里，7B 或 13B 模型微调更能体现这个差异。全 `FP32` 常常吃不下显存；直接 `.half()` 又容易数值不稳。实际可用的方案通常是：矩阵乘法等大头算子走 `FP16/BF16`，主权重和优化器状态保留 `FP32`，如果是 `FP16` 再加 `GradScaler`。这就是“混合”的含义。

| 格式 | 位宽 | 动态范围 | 训练稳定性 | 是否常需 loss scaling | 常见定位 |
|---|---:|---:|---:|---:|---|
| `FP32` | 32 | 大 | 高 | 否 | 基线训练、调试 |
| `FP16` | 16 | 较小 | 中 | 是 | 通用混合精度 |
| `BF16` | 16 | 接近 `FP32` | 较高 | 通常否 | 大模型训练优先选择 |

---

## 问题定义与边界

本文讨论的是训练和微调阶段的混合精度，不是推理量化，也不是模型压缩。微调的白话解释是：在已有预训练模型上，继续用自己的数据做参数更新。只要发生参数更新，就必须关心梯度、优化器状态、数值误差累计这些问题，因此训练场景比推理场景更严格。

这里先划清三个边界。

| 场景 | 目标 | 是否有反向传播 | 主要关心点 | 能否直接照搬 16 位 |
|---|---|---|---|---|
| 推理 | 更快生成结果 | 否 | 吞吐、延迟、显存 | 相对更容易 |
| 训练 | 从零学参数 | 是 | 梯度稳定、收敛、更新精度 | 不能简单照搬 |
| 微调 | 在已有模型上继续训练 | 是 | 显存、吞吐、稳定性平衡 | 不能简单照搬 |

为什么训练阶段不能把“全模型转成 16 位”当作完整方案？因为训练不只是前向推理。前向得到 loss 后，还要反向求梯度、做归约、裁剪梯度、更新优化器状态，再把参数写回去。这里有很多步骤对数值范围敏感。

通常可以粗略分成两类区域。

| 组件 | 常见精度策略 | 原因 |
|---|---|---|
| 矩阵乘法、卷积 | `FP16/BF16` | 计算量大，低精度收益高 |
| 激活值缓存 | `FP16/BF16` | 节省显存 |
| 主权重副本 | `FP32` | 更新累计更稳 |
| 优化器状态 | `FP32` | 动量、二阶矩对误差敏感 |
| `LayerNorm`、`softmax`、归约 | 常保留或回退 `FP32` | 小误差可能被放大 |

这里的主权重副本可以理解成“真正拿来做参数更新的那份高精度参数”。你看到模型在低精度下前向跑，不代表所有参数都只剩低精度版本。混合精度训练的工程关键之一，就是保留一份高精度的更新基准。

再强调一个边界：混合精度不是量化。量化的白话解释是：把参数或激活映射到更低位宽的离散表示，例如 `INT8` 或 `INT4`。混合精度仍然是浮点计算，只是换成更低位宽的浮点格式。两者都能省显存，但解决的问题不完全一样。

---

## 核心机制与推导

混合精度训练真正难的点，不在“会不会调 API”，而在理解为什么 `FP16` 容易不稳、为什么 `BF16` 往往更稳、为什么要保留 `FP32` 更新。

先看浮点数的本质。浮点数可以粗略理解成：

$$
\text{value} = \text{sign} \times \text{mantissa} \times 2^{\text{exponent}}
$$

这里 `mantissa` 可以理解成有效数字部分，决定“刻度有多细”；`exponent` 可以理解成指数部分，决定“范围能铺多大”。`BF16` 和 `FP16` 都只有 16 位，但它们把位数分给了不同部分。`BF16` 给了更大的指数范围，所以它不一定更精细，但通常更不容易在大模型训练时出现溢出或下溢。

### 为什么 `FP16` 容易丢梯度

训练里很多梯度非常小，尤其在深层网络、归一化、长链路反向传播后更明显。假设某层真实梯度是：

$$
g = 10^{-8}
$$

如果这个值落在 `FP16` 难以稳定表示的区域，它可能直接被舍入成 0。结果不是“误差稍大一点”，而是“这一步完全没更新”。如果很多层、很多步都发生这种事，训练就会停滞或者变得极不稳定。

这就是 `loss scaling` 的动机。做法是先把 loss 乘上一个缩放因子 $S$：

$$
L_s = S \cdot L
$$

根据求导链式法则，梯度也会同比例放大：

$$
g_s = \frac{\partial L_s}{\partial w} = S \cdot g
$$

如果原始梯度很小，那么放大后的梯度更容易落在 `FP16` 可表示区域。等反向传播结束后，再除回去：

$$
\hat{g} = \frac{g_s}{S}
$$

最后更新主权重：

$$
w_{fp32} \leftarrow w_{fp32} - \eta \cdot \hat{g}
$$

这里 $\eta$ 是学习率，也就是每次更新走多大步。

### 玩具例子：下溢如何发生

假设：
- 原始梯度 $g = 10^{-8}$
- 缩放因子 $S = 1024$
- 学习率 $\eta = 10^{-3}$

则缩放后：

$$
g_s = 1024 \times 10^{-8} = 1.024 \times 10^{-5}
$$

这个数更容易被 `FP16` 保住。更新前再反缩放：

$$
\hat{g} = \frac{1.024 \times 10^{-5}}{1024} = 10^{-8}
$$

参数更新量仍然是：

$$
\Delta w = \eta \cdot \hat{g} = 10^{-3} \times 10^{-8} = 10^{-11}
$$

也就是说，`loss scaling` 并没有改变最终数学更新，只是帮助你在低精度中间过程里别把梯度丢掉。

### 为什么 `BF16` 通常不需要 loss scaling

`BF16` 的关键优势不是“更精确”，而是“更宽的数值范围”。这意味着很多原本会在 `FP16` 里下溢或上溢的中间值，在 `BF16` 里仍然能保住数量级。所以大模型训练里，`BF16` 常常可以省掉 `GradScaler`，流程更简单，调试更省心。

但要注意一个容易混淆的点：`BF16` 不是“全方面优于 `FP16`”。它主要优在范围，不是优在尾数精度。某些特别吃数值细粒度的操作，`BF16` 也不意味着绝对无风险。工程里真正稳定，还是依赖框架自动选择精度、保留敏感算子的高精度路径。

真实工程例子里，7B 模型在 A100 上做监督微调，常见配置是 `bf16 mixed precision`。因为这类模型层数深、激活多、梯度跨度大，`BF16` 的大范围更适合。若设备不支持 `BF16`，才回退到 `FP16 + GradScaler`。这不是“谁更先进”，而是谁更匹配当前硬件和任务。

---

## 代码实现

下面先给一个不依赖 GPU 的可运行玩具代码，用来验证 `loss scaling` 不会改变最终更新结果，并用 `assert` 检查结论。

```python
def mixed_precision_update(loss, grad, lr, scale):
    scaled_loss = loss * scale
    scaled_grad = grad * scale
    unscaled_grad = scaled_grad / scale
    new_weight = 1.0 - lr * unscaled_grad
    return scaled_loss, scaled_grad, unscaled_grad, new_weight

loss = 0.25
grad = 1e-8
lr = 1e-3
scale = 1024.0

scaled_loss, scaled_grad, unscaled_grad, new_weight = mixed_precision_update(
    loss=loss, grad=grad, lr=lr, scale=scale
)

assert abs(scaled_loss - 256.0) < 1e-12
assert abs(scaled_grad - 1.024e-5) < 1e-12
assert abs(unscaled_grad - grad) < 1e-20
assert abs(new_weight - (1.0 - 1e-11)) < 1e-18

print("loss scaling keeps the final update unchanged")
```

这个玩具例子只验证数学关系，不代表真实训练框架会用这种手写方式。真实工程中，PyTorch 会用 `autocast` 和 `GradScaler` 接管大部分细节。

### `FP16` 最小训练步

```python
import torch

model = torch.nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")

x = torch.randn(4, 8, device="cuda")
y = torch.randint(0, 2, (4,), device="cuda")

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda", dtype=torch.float16):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)

scaler.scale(loss).backward()

# 先反缩放，再做梯度裁剪，否则裁剪阈值会被放大后的梯度污染
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

这里有三个顺序不能乱。

1. `autocast` 只控制前向中适合低精度的算子怎么跑。
2. `scaler.scale(loss).backward()` 让反向传播使用放大的 loss。
3. `unscale_` 必须早于 `clip_grad_norm_`，因为裁剪应该针对真实梯度，而不是放大后的梯度。

### `BF16` 最小训练步

```python
import torch

model = torch.nn.Linear(8, 2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(4, 8, device="cuda")
y = torch.randint(0, 2, (4,), device="cuda")

optimizer.zero_grad(set_to_none=True)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`BF16` 版本看起来更短，核心原因是通常不需要 `GradScaler`。这不是因为训练没有风险，而是因为 `BF16` 较大的动态范围，已经解决了 `FP16` 最常见的“梯度太小直接归零”问题。

### 错误写法和正确写法

错误写法常见在初学者第一次接触混合精度时：

```python
model = model.half()
loss = model(x.half()).loss
loss.backward()
optimizer.step()
```

这段代码的问题在于，它把“模型变成半精度”和“混合精度训练”混为一谈。你没有显式处理主权重更新，也没有处理梯度缩放，也没有让框架自动判断哪些算子应回退高精度。这种写法可能能跑，但不等于稳。

更合理的思路是：

- 让 `autocast` 决定大多数算子用什么低精度。
- 让优化器继续围绕高精度主权重更新。
- 在 `FP16` 时使用 `GradScaler`。
- 对梯度裁剪、梯度累积、自定义算子做额外检查。

---

## 工程权衡与常见坑

混合精度不是“打开开关就只有收益”的功能。工程上至少要同时看四个维度：显存、吞吐、收敛、维护成本。

| 维度 | `FP32` | `FP16` 混合精度 | `BF16` 混合精度 |
|---|---|---|---|
| 显存占用 | 高 | 低 | 低 |
| 训练速度 | 基线 | 常更快 | 常更快 |
| 数值稳定性 | 最高 | 依赖缩放和实现 | 通常更稳 |
| 调试复杂度 | 低 | 中到高 | 中 |

下面是实际工程里最常见的坑。

| 问题 | 错误做法 | 正确做法 |
|---|---|---|
| 梯度下溢 | `FP16` 下直接反向传播 | 使用 `GradScaler` |
| 梯度裁剪顺序错误 | 先裁剪再 `unscale_` | 先 `unscale_` 再裁剪 |
| 把 `.half()` 当完整方案 | 参数、状态全转半精度 | 使用 `autocast` 管理计算精度 |
| 忽略敏感算子 | 强制所有算子低精度 | 允许 `LayerNorm`、`softmax` 等保留高精度 |
| 误解 `BF16` | 以为它比 `FP16` 更精细 | 理解它是“范围更大，不是更细” |

### 真实工程例子：7B 指令微调

假设你在单机多卡上做 7B 模型指令微调，已经用了梯度累积和检查点重计算，但 batch size 还是上不去。此时混合精度的收益通常不是“省出一点显存”，而是可能直接让训练配置从“跑不起来”变成“可稳定训练”。

常见实践是：

- A100/H100 等支持较好时优先 `BF16`。
- 旧卡或环境不完整时使用 `FP16 + GradScaler`。
- 优化器状态保留 `FP32`。
- 自定义 fused kernel、attention 实现、归约操作单独验证数值行为。
- 观察 loss 曲线是否突然 `nan`、突然抖动、或长时间不下降。

如果在 `FP16` 训练里出现 `nan`，不要第一时间把锅甩给学习率。要先检查三件事：是否用了 `GradScaler`，是否有算子被错误强制到半精度，是否在 `unscale_` 前就做了梯度裁剪。很多“玄学不收敛”其实是顺序错了。

---

## 替代方案与适用边界

混合精度很常见，但它不是唯一方案。选择标准不是“社区最流行什么”，而是“你现在的瓶颈到底是什么”。

| 方案 | 显存压力 | 稳定性 | 改动成本 | 适用场景 |
|---|---:|---:|---:|---|
| `FP32` | 高 | 高 | 低 | 小模型、调试、排查数值问题 |
| `FP16` 混合精度 | 中低 | 中 | 中 | 硬件不支持 `BF16` 的常见加速方案 |
| `BF16` 混合精度 | 中低 | 较高 | 中 | 大模型训练、支持 `BF16` 的设备 |
| LoRA | 更低 | 较高 | 中 | 只训练少量增量参数 |
| QLoRA | 最低之一 | 中到较高 | 中到高 | 显存极紧张的大模型微调 |

可以按下面的思路判断。

第一，如果模型小、显存够、你在排查训练 bug，先上 `FP32`。因为它最简单，最适合作为基线。

第二，如果你需要更高吞吐或更低显存，而硬件支持 `BF16`，优先考虑 `BF16` 混合精度。它通常是当前大模型微调中更稳的默认选择。

第三，如果设备不支持 `BF16`，再使用 `FP16` 混合精度，并明确加入 `GradScaler`。这时你要接受更高一点的调试成本。

第四，如果模型还是装不下，混合精度就不够了。此时应考虑 LoRA 或 QLoRA。LoRA 的白话解释是：冻结大部分原模型，只训练少量低秩适配参数。QLoRA 则进一步把底座模型以更低位宽存储，再在其上做参数高效微调。它们解决的是“显存根本不够”的问题，不只是“训练想更快一点”的问题。

所以，混合精度的适用边界可以概括成一句话：当你的目标是尽量保留全量训练流程，同时降低显存和提高吞吐，它是第一层工程优化；当你的目标是把本来根本装不下的模型塞进有限显存，参数高效微调或量化方案才是下一层手段。

---

## 参考资料

1. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
2. [PyTorch Automatic Mixed Precision Documentation](https://docs.pytorch.org/docs/stable/amp.html)
3. [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
4. [NVIDIA Transformer Engine: Low Precision Training Introduction](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/introduction/introduction.html)
5. [PyTorch Blog: Bfloat16 on Intel Xeon](https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/)

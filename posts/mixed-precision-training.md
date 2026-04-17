## 核心结论

混合精度训练的本质是：**把“便宜的大规模计算”和“可靠的小幅累积更新”分开处理**。前向传播和反向传播主要消耗算力与显存，适合用 FP16 或 BF16；参数更新和优化器状态需要长期累积误差，必须保留 FP32。

这里先解释三个术语。**前向传播**就是模型把输入变成输出的计算过程。**反向传播**就是根据损失函数把误差传回各层、求梯度的过程。**优化器状态**就是 Adam、SGD with momentum 这类优化器内部保存的历史统计量，例如动量和二阶矩。

三种常见浮点格式可以先看成“指数位决定数值范围，尾数位决定数值精细度”：

| 格式 | 位宽 | 符号位 | 指数位 | 尾数位 | 主要特点 |
|---|---:|---:|---:|---:|---|
| FP16 | 16 | 1 | 5 | 10 | 显存省、吞吐高，但动态范围窄 |
| BF16 | 16 | 1 | 8 | 7 | 动态范围接近 FP32，但精度更粗 |
| FP32 | 32 | 1 | 8 | 23 | 范围大、精度高，适合参数更新 |

关键结论有两条。

第一，**FP16/BF16 做前向与反向，FP32 保存 master weights 和 optimizer state**，通常可以在基本不损失收敛质量的前提下，显著减少显存占用并提升训练吞吐。

第二，**FP16 训练通常需要 Gradient Scaling**。Gradient Scaling 的白话解释是：先把 loss 放大，再算梯度，最后把梯度缩回去。这样做不是改变学习目标，而是防止梯度在 FP16 里因为太小而直接变成 0。

一个真实工程例子是：同一份 Transformer 训练任务，从全 FP32 切到 A100 上的 BF16 前向/反向，参数更新仍保留 FP32 master copy，常见现象是训练速度明显提高，而收敛曲线与全 FP32 基本重合。这也是当前大模型训练的标准做法之一。

---

## 问题定义与边界

问题并不是“低精度能不能算”，而是“低精度会不会把关键数值算坏”。

浮点数的两个风险是：

| 风险 | 白话解释 | 常见后果 |
|---|---|---|
| Underflow，下溢 | 数太小，小到格式表示不出来 | 梯度直接变成 0 |
| Overflow，上溢 | 数太大，大到格式表示不出来 | 结果变成 `inf` 或 `nan` |

混合精度训练要解决的边界条件是：

1. 前向/反向尽量放到低精度执行，获得吞吐和显存收益。
2. 参数更新不能因为低精度累积误差而漂移。
3. 数值异常要能检测并恢复，而不是静默污染训练。

FP16 的问题尤其典型。它只有 5 位指数，动态范围较窄。很多深层网络中的梯度本来就很小，例如某层真实梯度是 $5 \times 10^{-6}$。如果在低精度链路里继续被缩小，可能直接掉到无法有效表示的区间，最后看起来像“这层没学到任何东西”。

混合精度的数值流可以写成：

$$
\text{loss}_{scaled} = \text{loss} \times S
$$

$$
g_{scaled} = \frac{\partial \text{loss}_{scaled}}{\partial w} = S \cdot \frac{\partial \text{loss}}{\partial w}
$$

$$
g = \frac{g_{scaled}}{S}
$$

其中 $S$ 是 scale factor，也叫缩放因子。它的作用不是改变目标函数，而是把梯度临时抬高到 FP16 能安全表示的范围里。

所以本文讨论的边界很明确：

- 低精度计算主要针对前向和反向。
- FP32 负责“记账”，也就是 master weights 和 optimizer states。
- 如果硬件原生支持 BF16，很多场景可以不做 loss scaling。
- 如果主要依赖 FP16，loss scaling 通常不是可选优化，而是稳定训练的必要机制。

---

## 核心机制与推导

先看一个玩具例子。

设某个参数的真实梯度是：

$$
g = 5 \times 10^{-6}
$$

如果直接在 FP16 路径中传播，这个数可能非常接近危险区，后续再经过链式求导中的缩放，很容易继续减小并被冲成 0。现在设缩放因子 $S = 1024$，则：

$$
g_{scaled} = g \times S = 5 \times 10^{-6} \times 1024 = 5.12 \times 10^{-3}
$$

这个值对 FP16 来说安全得多。反向传播结束后再还原：

$$
g = \frac{g_{scaled}}{1024}
$$

这样，低精度计算时避免了下溢，真正更新参数时又回到了原始梯度尺度。

这就是 loss scaling 的核心机制。它只改变计算链路中的中间数值范围，不改变训练目标。

再往前推一步，为什么还需要 FP32 master weights？

因为参数更新不是一次性的，而是很多步的小量累加。以 SGD 为例：

$$
w_{t+1} = w_t - \eta g_t
$$

如果 $w_t$、$\eta g_t$、历史状态都用低精度保存，那么“小改动加到大数上”时，更新量可能直接被舍入掉。白话说就是：**模型明明在更新，但低精度记账本太粗，看起来像没记上**。

对 Adam 这类优化器更明显，因为它维护一阶矩和二阶矩：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

这里的 $m_t$ 和 $v_t$ 都是长期累计量。如果也放到 FP16 或 BF16，误差会不断积累，最后影响收敛。因此工程上通常要求：

- 模型参与算子的张量可以是 FP16/BF16
- 梯度在 unscale 后送入 FP32 optimizer
- master parameters 和 optimizer states 始终保持 FP32

动态 loss scaling 则解决“scale 取多少”的问题。静态 scale 的缺点是写死后不适配不同训练阶段。动态规则通常是：

| 事件 | 处理 |
|---|---|
| 检测到 overflow | 跳过本次参数更新，`scale = scale / 2` |
| 连续 N 步没有 overflow | `scale = scale * 2` |
| scale 太小仍频繁 underflow | 继续尝试增大，直到接近可用上界 |

这相当于自动寻找“当前训练阶段能承受的最大安全 scale”。最大安全值附近通常最好，因为它最能减少梯度下溢。

真实工程里常见的情况是：某一批次激活值突然变大，导致反向某层梯度溢出，训练框架检测到 `inf` 或 `nan`，就跳过这一步，不污染参数，并把 scale 从 8192 降到 4096。后续若长期稳定，再逐步升回去。这种“防爆机制”是混合精度真正能落地的关键。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用来演示“缩放前后梯度数值恢复一致”这一点。

```python
def scale_gradient(g, scale):
    g_scaled = g * scale
    g_unscaled = g_scaled / scale
    return g_scaled, g_unscaled

g = 5e-6
scale = 1024

g_scaled, g_unscaled = scale_gradient(g, scale)

assert abs(g_scaled - 5.12e-3) < 1e-12
assert abs(g_unscaled - g) < 1e-18

# 模拟动态 loss scaling 的一个极简策略
def update_scale(scale, overflow, growth_interval_reached):
    if overflow:
        return scale / 2
    if growth_interval_reached:
        return scale * 2
    return scale

assert update_scale(1024, overflow=True, growth_interval_reached=False) == 512
assert update_scale(1024, overflow=False, growth_interval_reached=True) == 2048
assert update_scale(1024, overflow=False, growth_interval_reached=False) == 1024
```

上面的代码只演示数值流程，不涉及深度学习框架。实际训练时，PyTorch 里最常见的写法是 `autocast + GradScaler`。`autocast` 的白话解释是“让框架自动选择适合的低精度算子类型”。`GradScaler` 的白话解释是“自动做 loss scaling，并负责检测 overflow”。

```python
import torch
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

x = torch.randn(64, 16, device=device)
y = torch.randn(64, 1, device=device)

model.train()
optimizer.zero_grad()

with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

assert loss.ndim == 0
```

这段代码里每一步的职责是：

| 步骤 | 作用 |
|---|---|
| `autocast()` | 前向中尽量使用低精度算子 |
| `scaler.scale(loss).backward()` | 先放大 loss，再反向传播 |
| `scaler.step(optimizer)` | 自动 unscale，检查 overflow，安全时再更新参数 |
| `scaler.update()` | 根据是否溢出动态调整 scale |

如果是 BF16，代码形式很像，但在很多支持 BF16 的 GPU 上通常不需要 `GradScaler`。例如 A100 上常见配置是前向/反向直接 BF16，优化器状态和 master copy 仍保留 FP32。原因是 BF16 的指数位和 FP32 一样，动态范围足够大，数值更不容易因为范围不足而下溢或上溢。

一个真实工程例子可以这样理解：训练一个十亿级参数 Transformer 时，激活和梯度张量占用非常大。如果全程 FP32，显存压力和带宽压力都会很高；切换到 BF16 后，张量体积接近减半，Tensor Core 吞吐更高，而 FP32 optimizer state 保证了长期更新不失真。这也是大模型训练普遍采用 BF16 的直接原因。

---

## 工程权衡与常见坑

混合精度不是“改个 dtype 就结束”，真正的坑主要集中在数值稳定性和状态保存上。

| 常见坑 | 现象 | 原因 | 应对 |
|---|---|---|---|
| scale 太高 | 出现 `nan`、`inf`，某步被跳过 | 梯度或中间结果 overflow | 动态缩放，overflow 后立即 `÷2` |
| scale 太低 | 梯度大量变 0，收敛慢 | 仍有 underflow | 提高 scale，或使用动态 scaling |
| optimizer state 用低精度 | loss 降不动或后期震荡 | 历史统计量累积误差太大 | 始终保留 FP32 |
| 误以为 BF16 等于 FP32 | 训练稳定但指标略差 | BF16 动态范围大，但尾数只有 7 位 | 关键累积量仍用 FP32 |
| 所有算子都强制低精度 | 少数算子不稳定 | 某些归一化、归约操作对精度敏感 | 依赖框架 autocast 的白名单/黑名单策略 |

这里要特别区分“范围”和“精度”。

- **范围**指能表示多大和多小的数，主要由指数位决定。
- **精度**指相邻两个可表示数之间有多密，主要由尾数位决定。

BF16 的优势是范围大，和 FP32 同级；但它的精度仍明显低于 FP32。因此不能得出“既然 BF16 很稳，那优化器状态也能改 BF16”这种结论。对长期累计的状态，FP32 仍是更稳妥的下界。

另一个容易忽略的点是：**overflow 不是坏事本身，检测不到 overflow 才是坏事**。在动态 loss scaling 中，偶发 overflow 代表当前 scale 接近上限，系统随后下调并继续训练，这是一种正常控制回路。真正危险的是静默出现 `nan` 后仍继续更新，把参数彻底污染。

---

## 替代方案与适用边界

混合精度不是唯一方案，不同硬件和任务边界下选择不同。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 全 FP32 | 老旧硬件、极端稳定性要求、排查数值 bug | 最直观，最稳 | 显存和吞吐成本高 |
| FP16 + Loss Scaling | 支持 FP16 Tensor Core、但 BF16 不友好或不可用 | 加速明显，生态成熟 | 需要管理 scale |
| BF16 + FP32 状态 | A100/H100 等原生 BF16 硬件 | 范围大，通常不需 scaling | 尾数少，部分任务仍不如 FP32 精细 |
| 纯低精度更新 | 极端压缩或实验性方案 | 理论上更省 | 收敛风险高，不适合作为默认工程方案 |

硬件上可以粗略对比：

| 硬件类型 | 常见低精度选择 | 是否通常需要 loss scaling | 典型判断 |
|---|---|---|---|
| A100/H100 | BF16 优先 | 通常不需要 | 稳定性和吞吐兼顾，适合大模型 |
| V100/T4 等偏旧平台 | FP16 常见 | 通常需要 | 生态成熟，但要处理 underflow/overflow |
| 无低精度加速硬件 | FP32 | 不适用 | 保守但成本更高 |

所以适用边界可以总结为：

1. 如果硬件原生支持 BF16，优先考虑 **BF16 前向/反向 + FP32 参数更新**。
2. 如果主要依赖 FP16，默认应配套 **GradScaler 或等价动态 scaling 机制**。
3. 如果任务对数值极端敏感，或者正在调试收敛异常，先回退到全 FP32 是合理策略。
4. 如果你看到训练“能跑但效果差”，先检查的不是吞吐，而是 optimizer state 是否保持 FP32、scale 是否合理、是否存在静默 `nan`。

---

## 参考资料

| 资料 | 类型 | 重点 |
|---|---|---|
| TensorFlow Mixed Precision Guide | 官方文档 | 混合精度基本原理、loss scaling、数值范围 |
| NVIDIA Train With Mixed Precision | 官方文档 | FP16/BF16 工程实践、overflow/underflow 处理 |
| Bitfern Mixed Precision Training Explained | 实战文章 | FP32 master weights、BF16 在大模型中的使用 |
| PyTorch AMP 文档 | 官方文档 | `autocast`、`GradScaler` 的标准用法 |

- TensorFlow: https://www.tensorflow.org/guide/mixed_precision
- NVIDIA: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
- Bitfern: https://bitfern.com/mixed-precision-training-explained/
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html

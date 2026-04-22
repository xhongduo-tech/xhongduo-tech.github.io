## 核心结论

Weight Standardization（WS，卷积权重标准化）是在卷积层前向计算时，对每个输出通道对应的卷积核权重做零均值、单位方差标准化。

它处理的是**权重**，不是输入图片，也不是中间激活。标准公式是：

$$
\mu_i = mean(W_i)
$$

$$
\sigma_i = \sqrt{var(W_i) + \epsilon}
$$

$$
W'_i = \frac{W_i - \mu_i}{\sigma_i}
$$

其中 $W_i$ 表示第 $i$ 个输出通道对应的卷积核权重，$W'_i$ 是标准化后的权重。

一句话结论：**WS 处理权重，GN 处理激活，二者配合用于微批次训练。**

BatchNorm（BN，批归一化）依赖当前 batch 的均值和方差。batch size 很小时，比如每卡只有 1-2 张图，这些统计量很容易抖动。GroupNorm（GN，组归一化）不依赖 batch，但它只处理激活分布。WS 补上另一侧：先让卷积核权重的尺度更稳定，再让 GN 处理卷积输出。

一个新手版例子：如果一层卷积核里的数值分布很乱，有的特别大，有的特别小，同一层输出就更容易不稳定。WS 先把每个输出通道的卷积核拉到接近同一尺度，再交给后面的 GN 处理激活。

| 方法 | 作用对象 | 是否依赖 batch | 典型用途 |
|---|---:|---:|---|
| BatchNorm | 激活 | 是 | 大 batch 分类训练 |
| GroupNorm | 激活 | 否 | 小 batch 检测、分割 |
| Weight Standardization | 卷积权重 | 否 | 配合 GN 稳定卷积训练 |

---

## 问题定义与边界

WS 要解决的问题不是“所有网络都需要归一化”，而是更具体的问题：在目标检测、实例分割、语义分割等任务中，输入分辨率高、模型大、显存压力大，单卡 batch size 常被压到 1-2。此时 BN 依赖的 batch 统计量代表性很差，训练会更不稳定。

BN 的统计对象是当前 batch 中的激活。激活是神经网络中间层的输出特征，可以理解为“输入经过若干层计算后的表示”。如果每次只喂 1 张图，BN 看到的“这一批图像均值”几乎没有代表性，统计会随样本变化明显漂移。

WS 的边界很明确：

| 方法 | 输入对象 | 统计范围 | 是否依赖 batch | 常见位置 |
|---|---|---|---:|---|
| BatchNorm | 激活 | batch + 空间维度 | 是 | Conv 后 |
| GroupNorm | 激活 | 通道分组 + 空间维度 | 否 | Conv 后 |
| Weight Standardization | 卷积权重 | 每个输出通道的卷积核 | 否 | Conv 内部 |

设卷积权重张量为：

$$
W \in R^{C_{out} \times C_{in} \times kH \times kW}
$$

其中 $C_{out}$ 是输出通道数，$C_{in}$ 是输入通道数，$kH$ 和 $kW$ 是卷积核高宽。WS 对每个输出通道单独处理，也就是对形状为 $C_{in} \times kH \times kW$ 的一组权重求均值和方差。

真实工程例子：在 Mask R-CNN、DeepLab 这类模型中，单卡 batch 常常被压到 1-2。继续使用 BN 时，统计量容易不稳定；改成 `Conv + GN` 通常更稳；再加入 WS，即 `Conv(WS) + GN + ReLU`，可以进一步稳定卷积权重尺度。

WS 的使用边界：

| 边界 | 说明 |
|---|---|
| 只针对卷积权重 | 不处理输入图片，也不处理激活 |
| 常用于后接归一化层的卷积 | 典型组合是 WS + GN |
| 不是 GN 的替代品 | GN 处理激活，WS 处理权重 |
| 不是所有网络都必须使用 | 大 batch 分类模型可能 BN 已经足够 |

---

## 核心机制与推导

WS 的计算对象是每个输出通道对应的卷积核。对普通二维卷积来说，一个输出通道会有一组权重，覆盖所有输入通道和卷积核空间位置。WS 对这组权重求均值和方差，然后标准化。

完整前向过程可以写成：

$$
\mu_i = \frac{1}{N}\sum_{j=1}^{N} W_{i,j}
$$

$$
\sigma_i = \sqrt{\frac{1}{N}\sum_{j=1}^{N}(W_{i,j} - \mu_i)^2 + \epsilon}
$$

$$
W'_{i,j} = \frac{W_{i,j} - \mu_i}{\sigma_i}
$$

$$
y = Conv(x, W')
$$

| 符号 | 含义 |
|---|---|
| $W_i$ | 第 $i$ 个输出通道对应的卷积核权重 |
| $\mu_i$ | 该卷积核权重的均值 |
| $\sigma_i$ | 该卷积核权重的标准差 |
| $\epsilon$ | 防止方差接近 0 时除法不稳定的小常数 |
| $W'_i$ | 标准化后的卷积核权重 |
| $y$ | 使用标准化权重卷积后的输出 |

玩具例子：假设某个输出通道的卷积核权重展开后是 `[1, 2, 3, 4]`。

均值：

$$
\mu = \frac{1 + 2 + 3 + 4}{4} = 2.5
$$

方差：

$$
var = \frac{(1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2}{4} = 1.25
$$

标准差：

$$
\sigma = \sqrt{1.25} \approx 1.118
$$

标准化后约为：

```text
[-1.342, -0.447, 0.447, 1.342]
```

结果是这组卷积核权重的均值接近 0，标准差接近 1。更准确地说，WS 不是让模型“更聪明”，而是改变优化时的参数尺度，让同一层不同输出通道的卷积核不至于尺度差异过大。

为什么通常和 GN 搭配？因为 WS 标准化权重，影响卷积计算本身；GN 标准化激活，影响卷积输出后的特征分布。二者处理的位置不同：

```text
Input -> Conv(WS) -> GroupNorm -> ReLU -> ...
```

---

## 代码实现

最小 PyTorch 实现如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        var = w.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

# toy check
conv = WSConv2d(1, 1, kernel_size=2, bias=False)
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))

w = conv.weight
mean = w.mean(dim=(1, 2, 3), keepdim=True)
var = w.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
w_std = (w - mean) / torch.sqrt(var + 1e-5)

assert torch.allclose(w_std.mean(dim=(1, 2, 3)), torch.zeros(1), atol=1e-6)
assert torch.allclose(w_std.var(dim=(1, 2, 3), unbiased=False), torch.ones(1), atol=1e-5)

x = torch.ones(1, 1, 3, 3)
y = conv(x)
assert y.shape == (1, 1, 2, 2)
```

这段代码的意思是：先把卷积核按输出通道标准化，再拿标准化后的卷积核做卷积。

`unbiased=False` 表示使用总体方差，而不是无偏估计方差。这里标准化的是当前权重本身，不是在用样本估计总体统计量，所以通常使用 `unbiased=False` 更符合计算目的。

与 GroupNorm 组合时，可以这样写：

```python
import torch.nn as nn

block = nn.Sequential(
    WSConv2d(64, 128, kernel_size=3, padding=1, bias=False),
    nn.GroupNorm(num_groups=32, num_channels=128),
    nn.ReLU(inplace=True),
)
```

这里 `num_groups=32` 表示把 128 个通道分成 32 组，每组 4 个通道。GN 的统计不依赖 batch size，所以 batch size 为 1 时也能正常工作。

分组卷积和深度可分离卷积也可以使用同样写法。关键点是统计轴仍然是 `(1, 2, 3)`，也就是对每个输出通道单独统计它对应的权重，不要把整层权重混在一起算。

训练和推理必须走同一条路径。WS 不像 BN 那样维护 running mean 和 running var，它每次前向都会从当前权重计算标准化结果。因此推理时也应该使用同一个 `WSConv2d.forward`，不能临时跳过标准化。

---

## 工程权衡与常见坑

WS 的价值主要出现在“小 batch + 卷积主导 + 后接归一化”的场景。它不是免费午餐，也不是所有模型都该加。

| 坑点 | 错误做法 | 正确做法 |
|---|---|---|
| 把 WS 当成 GN | 认为 WS 可以替代激活归一化 | WS 处理权重，GN 处理激活 |
| 统计维度写错 | 对整层权重求一个均值和方差 | 按输出通道分别统计 |
| 省略 $\epsilon$ | 直接除以标准差 | 使用 `sqrt(var + eps)` |
| 乱加到所有层 | Linear、Embedding 都套 WS | 主要用于卷积层 |
| 推理路径不一致 | 训练用 WS，推理不用 | 训练和推理使用同一实现 |
| 忽略分组卷积 | 按 groups 重新写复杂统计 | 仍按输出通道统计 |

新手容易混淆的一点是：WS 不是“把网络所有地方都归一化一下”。它只管卷积权重，不能替代激活归一化。卷积输出是否稳定，还需要 GN、BN 或其他激活归一化方法处理。

工程上还有一个常见错误：在分组卷积或深度可分离卷积里，把整层权重一起算均值和方差。这样会把不同输出通道混在一起，改变 WS 的定义。正确做法仍然是每个输出通道单独标准化。

三种结构的差异可以这样看：

| 结构 | batch 依赖 | 适用场景 | 说明 |
|---|---:|---|---|
| Conv + BN | 是 | 大 batch 分类 | 简单高效，但小 batch 易不稳 |
| Conv + GN | 否 | 小 batch 检测、分割 | 不依赖 batch 统计 |
| Conv + WS + GN | 否 | 微批次卷积模型 | 同时稳定权重尺度和激活分布 |

WS 也会增加一点前向计算开销，因为每次卷积前都要计算权重均值和方差。通常这点开销相对卷积本身不大，但在极端性能敏感场景中仍需评估。

---

## 替代方案与适用边界

如果 batch size 足够大，BN 仍然可能是最简单有效的选择。WS 的优势来自特定条件：微批次训练、卷积模型、归一化层配合。脱离这些条件，收益可能有限。

| 方案 | 依赖 batch | 适用 batch size | 优点 | 限制 |
|---|---:|---|---|---|
| BatchNorm | 是 | 较大 batch | 成熟、速度快、广泛验证 | 小 batch 统计不稳 |
| GroupNorm | 否 | 小 batch | 统计稳定，适合检测分割 | 可能不如大 batch BN 简单高效 |
| WS + GN | 否 | 微批次 | 权重和激活两侧都更稳定 | 增加实现复杂度 |
| LayerNorm | 否 | Transformer 更常见 | 不依赖 batch | CNN 中不一定合适 |
| InstanceNorm | 否 | 风格迁移等任务 | 按样本归一化 | 可能削弱实例间统计信息 |

新手版判断：如果你训练普通图像分类模型，batch 很大，BN 可能已经足够，不一定需要额外引入 WS。先把基线跑稳，再考虑替换归一化结构。

真实工程判断：如果你在训练检测或分割模型，显存紧张导致每卡 batch size 只有 1-2，并且 BN 表现不稳，可以优先考虑 `Conv + GN`。如果还想进一步改善卷积权重尺度，再引入 `Conv + WS + GN`。

WS 不是普适最优解，而是微批次卷积模型里的稳定化手段。它的判断标准不是“论文里效果好”，而是你的训练瓶颈是否真的来自小 batch 下的统计不稳定和卷积权重尺度问题。

---

## 参考资料

1. [Micro-Batch Training with Batch-Channel Normalization and Weight Standardization](https://arxiv.org/abs/1903.10520)
2. [joe-siyuan-qiao/WeightStandardization](https://github.com/joe-siyuan-qiao/WeightStandardization)
3. [Group Normalization](https://arxiv.org/abs/1803.08494)
4. [torch.nn.GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html)

论文用于理解原理，代码仓库用于确认实现细节，PyTorch 文档用于确认 GN 接口。

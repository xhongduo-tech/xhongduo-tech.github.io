## 核心结论

混合精度下的 BN 核心风险，是 batch 统计、running stats、前后向数值口径不统一。

BatchNorm，简称 BN，是一种用当前 batch 的均值和方差来稳定中间特征分布的归一化层。Mixed Precision，中文常叫混合精度，是指训练时让一部分计算使用 `fp16` 或 `bf16` 以提高速度、降低显存，同时保留一部分关键状态或计算在 `fp32` 中以维持数值稳定。

Mixed-Precision BatchNorm 的问题，不是“BN 不能混合精度”，而是统计量可能在不同精度下被计算、累积和使用。训练时当前 batch 的均值、方差可能来自半精度输入；running mean 和 running var 又常以 `fp32` 状态保存；推理时再用 running stats 替代当前 batch 统计。三者口径不一致时，归一化噪声会被放大。

| 口径 | `x_i` 的精度 | `μ_B / σ_B^2` 的精度 | `running mean/var` 的精度 | 训练与推理是否一致 |
|---|---:|---:|---:|---|
| 全 `fp32` BN | `fp32` | `fp32` | `fp32` | 较一致 |
| 半精度输入 BN | `fp16` | 可能受 `fp16` 量化影响 | 通常 `fp32` | 可能偏移 |
| AMP 默认 BN | 框架按算子策略处理 | 通常更保守 | 通常 `fp32` | 取决于实现和 batch 条件 |
| 手动 `.half()` BN | `fp16` | 更容易受舍入影响 | 可能被错误转成 `fp16` | 风险最高 |

新手版理解是：同一批数据里，`fp16` 可能把两个很接近的数看成同一个值，BatchNorm 算出来的均值和方差就会失真。训练时看起来还能跑，推理时却可能因为 running stats 的偏移变得不稳定。

真实工程里，`Mask R-CNN`、`Faster R-CNN`、`DeepLab` 这类检测和分割任务最容易踩坑。它们常见每卡 `batch=1~2`，BN 的 batch 统计本来就不稳；再叠加 AMP，常见现象是 loss 抖动、收敛变慢、验证指标上下波动。

---

## 问题定义与边界

本文讨论的是 BatchNorm 在 AMP / 半精度训练中的数值行为，不是泛指所有归一化层。问题成立的前提是：归一化依赖 batch 维统计，也就是每一步都要从当前 mini-batch 里估计均值和方差。

三层边界需要先说清楚：

| 边界 | 具体对象 | 为什么重要 |
|---|---|---|
| 统计对象 | 当前 batch 的均值和方差 | batch 越小，估计越不稳定 |
| 数值对象 | 输入、归一化中间量、running stats | 精度不同会造成口径不一致 |
| 场景对象 | 小 batch、多卡、检测、分割 | 这些任务最容易让 BN 统计不足 |

BatchNorm 每一步都在看“这一批样本”的均值和方差。batch 很小时，这个估计本来就不稳；再把数值精度降到 `fp16`，就像用更粗的刻度尺去量本来就很短的距离。这里的重点不是比喻，而是误差来源：样本数少导致统计误差大，半精度导致舍入误差大，二者叠加后会让归一化结果更抖。

几种归一化方式的边界如下：

| 方法 | 是否依赖 batch 统计 | 是否有 running stats | 是否受 mixed precision 影响 | 说明 |
|---|---:|---:|---:|---|
| BatchNorm | 是 | 是 | 是 | 当前 batch 统计和 running stats 都要关注 |
| SyncBatchNorm | 是，跨设备聚合 | 是 | 是 | 增大统计样本数，但仍有精度问题 |
| Ghost BatchNorm | 是，按小组统计 | 是 | 是 | 主动改变统计粒度 |
| GroupNorm | 否 | 否 | 较小 | 不依赖 batch 维，训练和推理行为一致 |

GroupNorm 是把通道分组后在每个样本内部做归一化的层，它不依赖 batch 统计，因此不属于本文问题的主体。SyncBatchNorm 是跨 GPU 聚合均值和方差的 BN，它可以缓解每卡 batch 太小的问题，但不消除半精度数值误差本身。

---

## 核心机制与推导

BatchNorm 的核心公式很简单：先对当前 batch 求均值和方差，再用它们把输入标准化，最后乘上可学习缩放参数 `γ`，加上可学习偏移参数 `β`。这里的 `γ` 是缩放系数，`β` 是偏移系数，它们让归一化后的特征仍然可以被模型调整。

设一个通道上参与统计的元素为 $x_1, x_2, ..., x_m$，其中 $m$ 是参与统计的元素数量。BatchNorm 的 batch 均值为：

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

batch 方差为：

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

归一化输出为：

$$
y_i = \gamma \cdot \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$

其中 $\epsilon$ 是一个很小的正数，用来避免除以零。

训练时，BN 还会维护 running mean 和 running var。running stats 是推理阶段使用的滑动平均统计量，因为推理时通常不再使用当前 batch 的统计。常见更新形式是：

$$
\hat{\mu}_t = (1 - \alpha)\hat{\mu}_{t-1} + \alpha\mu_B
$$

$$
\hat{\sigma}_t^2 = (1 - \alpha)\hat{\sigma}_{t-1}^2 + \alpha\sigma_B^2
$$

其中 $\alpha$ 是动量更新系数。它控制当前 batch 统计写入 running stats 的比例。

新手版推导可以按三步看：

| 步骤 | 操作 | 直白解释 |
|---|---|---|
| 1 | 计算均值 | 把这一批里的数加起来，再除以数量 |
| 2 | 计算方差 | 看每个数离均值有多远，再平方平均 |
| 3 | 标准化 | 减均值、除标准差，再乘 `γ` 加 `β` |

混合精度风险从第一步就开始。`fp16` 的表示范围和精度都比 `fp32` 粗。对于很接近的两个值，`fp32` 还能区分，`fp16` 可能直接舍入成同一个值。均值和方差一旦在这种输入上计算，就会出现更大的统计偏差。

玩具例子如下。取：

```text
x = [1.0, 1.00048828125]
```

在 `fp32` 下，这两个数可以区分：

```text
μ_B = 1.000244140625
σ_B^2 = 5.9604645e-8
```

但在 `fp16` 下，它们可能被舍入成非常接近甚至相同的表示。如果都变成 `1.0`，则：

```text
μ_B = 1.0
σ_B^2 = 0
```

这会让归一化分母退化到 $\sqrt{\epsilon}$。输出被压缩，梯度对微小扰动更敏感。小 batch 下这种问题更明显，因为参与统计的样本少，单个数值误差对整体均值和方差的影响更大。

可以把口径不一致看成下面的流程：

```text
输入 x
  -> autocast / half
  -> 当前 batch 统计 μ_B, σ_B^2
  -> 归一化前向与反向
  -> running mean/var 更新
  -> 推理阶段使用 running mean/var
```

如果训练时当前 batch 统计来自半精度口径，而 running stats 以更高精度累积，推理时又只使用 running stats，就会出现“训练前向看到的是一套统计，推理使用的是另一套统计”。这就是 Mixed-Precision BatchNorm 的核心挑战。

---

## 代码实现

代码目标不是手写一个完整 BN，而是展示 AMP 场景下更稳的使用方式。Autocast 是 PyTorch AMP 提供的自动精度选择机制，它会根据算子类型选择合适精度。GradScaler 是用于缩放 loss 的工具，主要用于避免半精度反向传播中的梯度下溢。

正确的 AMP 训练模板是：输入和大部分算子交给 `autocast`，BN 保持框架默认处理，不要手动把整个模型转成 `half`。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 2),
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

images = torch.randn(4, 3, 32, 32, device="cuda")
targets = torch.tensor([0, 1, 0, 1], device="cuda")

optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

assert torch.isfinite(loss).item()
assert outputs.shape == (4, 2)
```

错误用法是把 `autocast` 和手动 `.half()` 混在一起。手动 `.half()` 会把很多状态和模块也转成半精度，可能破坏框架对 BN、归约、loss 等算子的保守处理。

```python
# 反例：不要这样写
model = model.half()
images = images.half()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, targets)
```

如果 per-GPU batch 很小，并且你仍然想保留 BN 体系，可以考虑 SyncBatchNorm。DistributedDataParallel，简称 DDP，是 PyTorch 中常用的多进程多卡训练封装。SyncBatchNorm 需要在包 DDP 前转换。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
)

model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = model.cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
)
```

如果 batch 极小，例如检测或分割里每卡只有 `1`，更直接的做法是换成 GroupNorm。GroupNorm 不维护 running stats，训练和推理使用同一套计算逻辑。

```python
import torch.nn as nn

block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
    nn.GroupNorm(num_groups=32, num_channels=128),
    nn.ReLU(inplace=True),
)

assert isinstance(block[1], nn.GroupNorm)
```

不同做法的作用和风险如下：

| 做法 | 作用 | 风险 | 适用场景 |
|---|---|---|---|
| AMP + 默认 BN | 保留框架精度策略 | 小 batch 仍可能抖 | 普通分类、大 batch |
| 手动 `.half()` | 看似省显存 | 破坏 BN 和状态精度 | 不推荐 |
| SyncBatchNorm | 跨 GPU 聚合统计 | 通信成本增加 | 多卡、小 batch |
| GroupNorm | 去掉 batch 依赖 | 需要重调超参 | 检测、分割、小 batch |
| Ghost BatchNorm | 改变统计粒度 | running stats 口径变化 | 想模拟小组统计 |

---

## 工程权衡与常见坑

真正的工程问题不是“BN 理论上对不对”，而是训练稳定性、部署一致性、分布式成本之间的取舍。

你看到训练曲线一会儿上升一会儿下降，不一定是模型结构坏了。可能是每张卡上的 batch 太小，BN 统计本来就抖，再叠加 `fp16` 后抖得更厉害。检测和分割任务尤其常见，因为高分辨率图像占显存，每卡 batch 很难做大。

真实工程例子：训练 `Mask R-CNN` 时，每卡 `batch=2`，8 卡总 batch 是 16。普通 BN 如果只看每卡本地 batch，每个 BN 层统计只来自很少样本。AMP 开启后，卷积更快、显存更低，但 BN 的局部统计更容易影响 loss 曲线。工程上常见处理是：冻结 backbone BN、改用 SyncBatchNorm，或者在新结构中直接用 GroupNorm。

常见坑和规避方式如下：

| 坑点 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 手动 `.half()` + `autocast` | loss 抖动或 NaN | 精度策略被手动覆盖 | 只用 AMP 管理精度 |
| SyncBN 转换顺序错误 | 同步无效或报错 | 应在 DDP 前转换 | 先 `convert_sync_batchnorm()`，再包 DDP |
| 小 batch 继续用 BN | 验证指标波动 | batch 统计噪声过大 | 用 SyncBN、冻结 BN 或换 GN |
| 误解 GhostBN | 训练和普通 BN 不一致 | 它改变了统计分组 | 明确统计粒度再使用 |
| GroupNorm 不重调超参 | 收敛变慢或精度下降 | GN 不是 BN 等价替代 | 重新调学习率、weight decay、warmup |

工程判断准则可以压缩成三条：

| 条件 | 优先判断 |
|---|---|
| per-GPU batch 很小 | 先怀疑 BN |
| 多卡同步成本可接受 | 优先考虑 `SyncBatchNorm` |
| 对训练/推理一致性要求高 | 考虑 `GroupNorm` |

还有一个容易忽略的问题：训练和部署链路可能不完全一致。训练时使用 AMP，导出模型时使用 `fp32` 或 TensorRT 的 `fp16`，推理 batch 又可能是 `1`。如果 BN 的 running stats 本身已经偏了，部署侧再怎么调推理精度，也只能缓解，不能从根上修正训练阶段写入的统计偏差。

---

## 替代方案与适用边界

替代方案不是同一类问题的完全答案，而是不同约束下的折中。选择依据应是 batch 大小、分布式条件、推理一致性要求、迁移成本。

如果你做的是图像分类，大 batch 训练通常还能继续用 BN。分类任务输入分辨率较低，单卡 batch 可以比较大，BN 的统计估计相对稳定。此时优先使用框架默认 AMP，不要手动改半精度。

如果你做的是目标检测或语义分割，每卡 batch 只有 `1`，直接换 GroupNorm 往往更省事。GroupNorm 没有 running stats，训练和推理行为一致。代价是它不是 BN 的严格等价替代，学习率、正则、预训练权重适配都可能要重调。

| 方案 | 是否依赖 batch 统计 | 是否有 running stats | 训练/推理是否一致 | 代价 |
|---|---:|---:|---:|---|
| 继续用 BN | 是 | 是 | 不完全一致 | 对 batch 大小敏感 |
| SyncBatchNorm | 是，跨卡统计 | 是 | 不完全一致 | 增加通信开销 |
| Ghost BatchNorm | 是，小组统计 | 是 | 不完全一致 | 改变统计口径 |
| GroupNorm | 否 | 否 | 一致 | 可能需要重调超参 |
| 冻结 BN | 否，使用固定统计 | 是 | 较一致 | 依赖预训练统计质量 |

选择规则可以直接写成工程决策表：

| 条件 | 推荐方案 |
|---|---|
| batch 足够大 | 继续用 BN |
| 多卡训练且想保留 BN | `SyncBatchNorm` |
| batch 极小且想稳 | `GroupNorm` |
| 想改变统计粒度 | `Ghost BatchNorm` |
| 使用强预训练 backbone | 冻结 BN 或只训练 affine 参数 |

Ghost BatchNorm 是把一个大 batch 切成多个小组分别计算 BN 统计的方法。它不是“更快的 BN”，而是主动改变统计口径。适合想让模型看到更有噪声的小组统计，从而获得某种正则效果的场景；但它也会改变 running stats 的来源，不能当成普通 BN 的透明替代。

SyncBatchNorm 适合多卡训练，并且你希望继续使用 BN 体系的情况。它通过跨 GPU 聚合统计，让每层 BN 看到更大的有效 batch。但它需要额外通信，也受到分布式训练实现限制。

GroupNorm 适合小 batch、训练和推理一致性优先的情况。尤其在 detection 和 segmentation 中，GN 经常比硬撑 BN 更稳定。但它通常要重新调学习率、weight decay、warmup、归一化位置，不能只替换一行代码就默认性能不变。

---

## 参考资料

| 引用用途 | 资料 |
|---|---|
| AMP 文档 | 确认 mixed precision 的推荐使用方式 |
| BatchNorm2d 文档 | 确认 BN 参数、running stats 与训练推理行为 |
| SyncBatchNorm 文档 | 确认分布式同步约束 |
| GroupNorm 文档 | 确认替代方案接口 |
| Ghost Batch Normalization | 理解统计口径分组 |
| Group Normalization | 理解小 batch 下的替代思路 |

官方文档：

1. [PyTorch Automatic Mixed Precision package](https://docs.pytorch.org/docs/stable/amp.html?highlight=autocast)
2. [PyTorch BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm2d.html)
3. [PyTorch SyncBatchNorm](https://docs.pytorch.org/docs/main/generated/torch.nn.SyncBatchNorm.html)
4. [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html)

方法论文：

5. [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://huggingface.co/papers/1705.08741)
6. [Group Normalization](https://papers.cool/arxiv/1803.08494)

延伸阅读：

7. [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

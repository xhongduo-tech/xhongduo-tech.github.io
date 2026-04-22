## 核心结论

GroupNorm 是一种“按通道分组、在每个样本内部做归一化”的方法。它不使用 batch 维度上的统计量，因此训练行为不依赖 Batch Size。

一句话结论：`batch=1` 时，BatchNorm 会强依赖单个 batch 的均值和方差，统计容易波动；GroupNorm 只看当前样本内部的通道组，所以在小 batch 下仍能稳定工作。

归一化是深度学习里常见的稳定训练手段。白话说，它会把某一组特征的数值拉回到均值接近 0、方差接近 1 的范围，避免中间层特征尺度不断漂移。GroupNorm 的核心不是改变特征表达的语义，而是让每层输入的数值分布更稳定。

| 方法 | 统计范围 | 是否依赖 batch | 是否有 running stats | 典型适用场景 |
|---|---:|---:|---:|---|
| BatchNorm | 同一通道、跨 batch 和空间位置 | 是 | 是 | 大 batch 图像分类、常规 CNN |
| GroupNorm | 单样本内、通道分组后、组内空间位置 | 否 | 否 | 小 batch 检测、分割、视频任务 |
| LayerNorm | 单样本内、通常跨全部通道 | 否 | 否 | Transformer、MLP、序列模型 |
| InstanceNorm | 单样本内、单通道空间位置 | 否 | 否 | 风格迁移、生成模型、图像变换 |

真实工程例子：目标检测、语义分割、视频分析常因为输入分辨率高、模型大、显存紧，每张卡只能放 `1-2` 张图。此时 BatchNorm 的批统计不稳定，GroupNorm 经常是更稳妥的归一化层选择。

---

## 问题定义与边界

设 CNN 中某一层的输入为：

$$
x \in \mathbb{R}^{N \times C \times H \times W}
$$

其中 `N` 是 batch 大小，`C` 是通道数，`H` 和 `W` 是特征图高宽。GroupNorm 的目标是在不跨样本的前提下，对每个样本内部的通道组做归一化。

设通道数 `C` 被分成 `G` 个组，每组有：

$$
m = C / G
$$

个通道。对样本 `n` 和分组 `g`，定义统计集合：

$$
S_{n,g}=\{(n,c,h,w)\mid c \in [gm,(g+1)m),\,0 \le h < H,\,0 \le w < W\}
$$

白话解释：`S_{n,g}` 就是“第 `n` 个样本里，第 `g` 个通道组覆盖的所有位置”。GN 的统计只在组内、同一样本内计算，不看其他样本。

边界条件很明确：

| 条件 | 含义 |
|---|---|
| `C % G == 0` | 通道数必须能被分组数整除 |
| `G = 1` | 所有通道放进一个组，行为接近 LayerNorm |
| `G = C` | 每个通道单独成组，行为接近 InstanceNorm |
| `G = 32` | CNN 中常见默认起点，但不是硬规则 |

玩具例子：如果输入形状是 `N=1, C=4, H=W=1`，并设置 `G=2`，那么 4 个通道会被分成两组：第 1 组包含通道 0 和 1，第 2 组包含通道 2 和 3。它不会因为 `N=1` 就失去可用统计量，因为统计仍然可以在当前样本的通道组内部计算。

这正是小 batch 下 BN 和 GN 的关键差异。BatchNorm 需要跨 batch 统计同一通道的均值和方差；当 `batch=1` 时，批内样本太少，统计几乎完全由单个样本决定。GroupNorm 不跨样本，因此 `batch=1` 和 `batch=32` 在统计定义上没有本质差别。

---

## 核心机制与推导

GroupNorm 的计算分三步。

第一步，对每个样本、每个组计算均值和方差：

$$
\mu_{n,g}=\frac{1}{|S_{n,g}|}\sum_{i\in S_{n,g}} x_i,\quad
\sigma^2_{n,g}=\frac{1}{|S_{n,g}|}\sum_{i\in S_{n,g}}(x_i-\mu_{n,g})^2
$$

第二步，对组内每个元素做标准化：

$$
\hat{x}_i=\frac{x_i-\mu_{n,g}}{\sqrt{\sigma^2_{n,g}+\epsilon}}
$$

其中 $\epsilon$ 是一个很小的正数，用来避免方差接近 0 时除以 0。

第三步，使用逐通道的可学习参数做仿射变换：

$$
y_i=\gamma_i \hat{x}_i+\beta_i
$$

仿射变换是线性缩放加平移。白话说，标准化之后模型不一定希望所有通道都严格保持均值 0、方差 1，所以给每个通道一个可学习的缩放参数 $\gamma$ 和偏置参数 $\beta$，让网络自己决定最终尺度。

注意：GroupNorm 的 $\gamma$ 和 $\beta$ 与 BatchNorm 一样，都是逐通道参数，形状通常是 `[C]`。不同的是统计量来源不同。BN 使用 batch 统计，并在训练时维护 running mean 和 running variance；GN 不维护 running stats，训练和推理阶段都直接使用当前输入的组内统计。

用一个完整数值例子看机制。设 `N=1, C=4, H=W=1, G=2`，输入为：

```text
x = [1, 3, 2, 4]
```

分成两组：

| 组 | 原始值 | 均值 | 方差 | 标准化结果 |
|---|---:|---:|---:|---:|
| 第 1 组 | `[1, 3]` | `2` | `1` | `[-1, 1]` |
| 第 2 组 | `[2, 4]` | `3` | `1` | `[-1, 1]` |

如果 $\gamma=1,\beta=0$，最终输出就是：

```text
[-1, 1, -1, 1]
```

BN 和 GN 的统计范围可以这样对比：

| 输入元素 | BatchNorm 统计时看什么 | GroupNorm 统计时看什么 |
|---|---|---|
| 某个通道 `c` | 同一通道在整个 batch 内的值 | 当前样本内，该通道所在组的值 |
| 是否跨样本 | 跨样本 | 不跨样本 |
| batch 变小时 | 统计波动明显 | 定义不变 |
| 推理阶段 | 使用 running mean/var | 仍用当前输入统计 |

这也是论文中强调 GN 适合小 batch 的原因：它把归一化的统计对象从“这一批样本”换成了“单个样本内部的通道组”。

---

## 代码实现

PyTorch 里可以直接使用 `nn.GroupNorm`。最重要的两个参数是 `num_groups` 和 `num_channels`：前者是分组数，后者是通道数，并且 `num_channels` 必须能被 `num_groups` 整除。

```python
import torch
import torch.nn as nn

gn = nn.GroupNorm(num_groups=32, num_channels=128)
x = torch.randn(4, 128, 32, 32)
y = gn(x)

assert y.shape == x.shape
assert gn.num_groups == 32
assert gn.num_channels == 128
```

如果 `C=128, G=32`，每组有 `4` 个通道。对每个样本、每个组，GroupNorm 会在 `4 × H × W` 个数上计算均值和方差。

手写一个简化版有助于理解 reshape 逻辑：

```python
import torch

def simple_group_norm(x, num_groups, gamma=None, beta=None, eps=1e-5):
    # x: [N, C, H, W]
    N, C, H, W = x.shape
    assert C % num_groups == 0

    x_grouped = x.reshape(N, num_groups, C // num_groups, H, W)
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

    x_hat = (x_grouped - mean) / torch.sqrt(var + eps)
    x_hat = x_hat.reshape(N, C, H, W)

    if gamma is None:
        gamma = torch.ones(C, device=x.device, dtype=x.dtype)
    if beta is None:
        beta = torch.zeros(C, device=x.device, dtype=x.dtype)

    return x_hat * gamma.view(1, C, 1, 1) + beta.view(1, C, 1, 1)

x = torch.tensor([[[[1.0]], [[3.0]], [[2.0]], [[4.0]]]])
y = simple_group_norm(x, num_groups=2, eps=0.0)

expected = torch.tensor([[[[-1.0]], [[1.0]], [[-1.0]], [[1.0]]]])
assert torch.allclose(y, expected)
```

核心步骤只有四个：

```python
# x: [N, C, H, W]
# reshape -> [N, G, C//G, H, W]
# 对后 3 个维度求均值和方差
# 标准化后 reshape 回原形状
# 做逐通道 gamma/beta 仿射变换
```

真实工程里通常不手写这一层，直接用框架实现。原因是官方实现会处理 dtype、设备、反向传播、性能细节和边界情况。手写版适合理解机制，不适合作为生产训练代码的默认选择。

---

## 工程权衡与常见坑

第一类坑是分组数不合法。`num_channels % num_groups == 0` 是硬约束。例如 `C=96` 时，`G=32` 合法，每组 3 个通道；`C=100` 时，`G=32` 不合法，会直接报错。替换模型里的归一化层之前，应先检查每个 stage 的通道数。

第二类坑是认为 `G` 越大越好。不是。`G` 越大，每组通道越少，越接近 InstanceNorm；`G` 越小，每组通道越多，越接近 LayerNorm。不同设置会改变模型对通道统计的约束方式。

| `G` 设置 | 行为偏向 | 可能优势 | 主要风险 |
|---:|---|---|---|
| `G=1` | 接近 LayerNorm | 不要求通道分组，统计范围大 | 可能削弱 CNN 通道局部结构 |
| `G=8/16/32` | 标准 GroupNorm | 常见折中，适合 CNN 小 batch | 仍需验证任务效果 |
| `G=C` | 接近 InstanceNorm | 对每个通道独立归一化 | 可能丢失通道间幅值关系 |
| 自定义 `G` | 按网络通道数调整 | 可适配特殊结构 | 搜索成本更高 |

常见起点是 `G=32`，但它不是数学定律。如果某层通道数小于 32，或者不能被 32 整除，就需要调整分组数。工程上可以用“尽量接近 32，并保证能整除”的规则作为初始方案。

第三类坑是忽略 BN 到 GN 的替换成本。真实工程例子：把检测模型中 ResNet backbone 的 BatchNorm 全部替换成 GroupNorm 后，小 batch 训练稳定性可能提升；但学习率、权重衰减、预训练权重加载、初始化方式都可能需要重新评估。原因是归一化层改变了中间特征的数值行为，虽然网络结构看起来只替换了一行代码，但优化过程已经变了。

第四类坑是误以为 `eval()` 会改变 GN 的统计方式。BatchNorm 在训练时使用当前 batch 统计，并更新 running mean/var；在推理时使用 running mean/var。GroupNorm 没有 running stats，因此 `model.eval()` 不会让 GN 切换到滑动均值和滑动方差。`eval()` 仍会影响 Dropout 等层，但不会改变 GN 的统计来源。

第五类坑是性能判断过早。BatchNorm 在很多硬件和框架里优化很充分，GN 未必总是更快。尤其在大 batch 图像分类中，BN 可能同时有更好的速度和精度。是否替换应以任务、batch size、显存限制和 profiling 结果为准。

---

## 替代方案与适用边界

GroupNorm 不是“全面优于 BatchNorm”的替代品。更准确的说法是：当 batch 很小、跨设备同步成本高、输入尺寸大或任务导致显存紧张时，GN 更可靠；当 batch 足够大、BN 统计稳定且硬件优化充分时，BN 仍然是强基线。

| 方法 | 统计范围 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| BatchNorm | 同通道跨 batch 和空间 | 大 batch 下效果好，生态成熟，速度常有优势 | 小 batch 统计不稳，需要 running stats | 图像分类、大 batch CNN |
| LayerNorm | 单样本内跨特征维度 | 不依赖 batch，适合序列模型 | 在传统 CNN 中不一定最自然 | Transformer、MLP、NLP |
| InstanceNorm | 单样本单通道空间 | 强调单图像、单通道归一化 | 可能削弱通道间对比信息 | 风格迁移、生成式图像任务 |
| GroupNorm | 单样本通道分组 | 小 batch 稳定，无 running stats | 分组数需调，速度未必优于 BN | 检测、分割、视频、小 batch CNN |

任务选择建议可以直接按 batch 条件判断：

| 任务 | 常见 batch 情况 | 初始建议 |
|---|---:|---|
| ImageNet 图像分类 | batch 通常较大 | 先保留 BatchNorm |
| 目标检测 | 每卡 batch 常为 `1-2` | 优先评估 GroupNorm |
| 语义分割 | 高分辨率输入，显存压力大 | 优先评估 GroupNorm |
| 视频分类 | 输入含时间维，显存占用高 | 优先评估 GroupNorm |
| Transformer / MLP | 通常不是 CNN 通道分组结构 | LayerNorm 更自然 |

再回到一个玩具判断：如果你只有 `batch=1`，BN 的统计对象几乎只剩一张样本；GN 的统计对象仍是当前样本内部的通道组。因此在这个边界上，GN 的定义更稳定。

但在真实工程中，不能只看归一化层本身。同步 BatchNorm 也是一种选择，它会跨多张卡合并 batch 统计，缓解单卡 batch 小的问题。它的代价是通信开销更高，训练系统更复杂。对检测和分割任务，如果多卡同步成本可接受，SyncBN 和 GN 都值得实验；如果希望减少跨设备依赖，GN 更简单。

总结边界：GN 主要是 CNN 场景里小 batch 训练的稳健替代方案。在 Transformer 或纯 MLP 架构中，LayerNorm 通常更符合结构习惯；在大 batch CNN 分类任务中，BatchNorm 仍然应该作为优先基线。

---

## 参考资料

1. [Group Normalization, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)
2. [Group Normalization 论文 PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)
3. [PyTorch nn.GroupNorm 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html)
4. [PyTorch F.group_norm 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.group_norm.html)

本文代码行为以 PyTorch `GroupNorm` 为准；不同框架在参数命名、方差估计细节和默认 `eps` 上可能存在差异。

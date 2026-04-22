## 核心结论

DropBlock 是一种用于卷积神经网络（CNN）的正则化方法：训练时随机丢弃特征图上的连续矩形区域，而不是像普通 dropout 那样独立丢弃单个位置。

正则化是指在训练中主动加入约束或扰动，降低模型对训练集细节的过度依赖。DropBlock 的核心约束很直接：如果模型总是依赖某个局部区域的强响应，就随机遮掉这片区域，迫使模型从别的位置、别的通道、别的上下文中学习证据。

普通 dropout 更像零散丢弃：

```text
原特征图:
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1

普通 dropout:
1 0 1 1 1
1 1 1 0 1
0 1 1 1 1
1 1 0 1 1
1 1 1 1 0

DropBlock:
1 1 1 1 1
1 0 0 0 1
1 0 0 0 1
1 0 0 0 1
1 1 1 1 1
```

对 CNN 来说，相邻位置通常表达相近的局部视觉信息。普通 dropout 只删掉几个零散点，周围位置仍可能保留同一条边缘、纹理或目标部件的信息。DropBlock 直接删除一整块，更容易切断这种局部绕路。

| 方法 | 丢弃单位 | 主要适用对象 | 对 CNN 特征图的影响 |
|---|---:|---|---|
| Dropout | 独立神经元或独立位置 | 全连接层、通用网络 | 对空间相关性处理较弱 |
| Spatial Dropout | 整个通道 | 卷积特征图 | 强迫通道间不互相依赖 |
| DropBlock | 连续空间块 | CNN backbone、FPN、残差块 | 强迫模型不依赖局部强响应 |

玩具例子：一张特征图中，某个区域对“猫耳朵”有强激活。普通 dropout 可能只删掉耳朵边缘上的一个点，周围点仍然能补上。DropBlock 会删掉一小片耳朵区域，模型必须同时利用眼睛、轮廓、纹理等其他证据。

真实工程例子：在 `ResNet-50 + FPN` 的目标检测系统中，FPN 是特征金字塔网络，用来融合不同尺度的卷积特征。检测和分割任务依赖空间位置，DropBlock 放在 backbone 或残差分支中，通常比普通 dropout 更贴合二维特征图的结构。原论文报告，DropBlock 在 ResNet-50 的 ImageNet 分类上带来约 1% 到 2% 的精度提升，并能改善 COCO 检测任务表现。

---

## 问题定义与边界

CNN 特征图不是一组彼此独立的数字。特征图是卷积层输出的二维网格，每个位置表示输入图像某个局部区域的特征响应。由于卷积核在空间上滑动，相邻位置通常来自相邻感受野，携带高度相关的信息。

这会带来一个问题：普通 dropout 的独立点丢弃很难真正破坏局部证据。假设特征图上有一段目标边缘被激活，普通 dropout 删除其中一个点，网络仍能从附近点恢复相似信息。这不是模型更聪明，而是卷积特征本身存在空间冗余。

DropBlock 的问题边界是：它主要用于训练阶段的卷积特征图，不是为了替代所有场景中的 dropout。推理阶段应关闭 DropBlock，因为推理需要稳定输出，而不是继续随机遮挡特征。

| 场景 | 是否优先考虑 DropBlock | 原因 |
|---|---:|---|
| CNN backbone | 是 | 特征图有明显空间结构 |
| 残差块 | 是 | 可约束残差分支过度依赖局部响应 |
| FPN | 是 | 多尺度检测特征保留二维空间信息 |
| 检测 / 分割任务 | 是 | 输出强依赖空间定位和区域特征 |
| 小型全连接层 | 不优先 | 没有二维空间块的含义 |
| 极小特征图 | 谨慎 | 一个块可能遮掉过多信息 |
| 推理阶段 | 否 | 正则化只应在训练时启用 |

边界要说清楚：DropBlock 不是让模型“看不见图像”，而是在训练期间让模型不要只记住一小块最容易用的证据。它的目标是提高泛化能力。泛化能力是指模型在未见过的新样本上保持正确预测的能力。

---

## 核心机制与推导

DropBlock 的实现不是直接随机选择输出位置置零，而是分两步：先采样块中心，再把中心扩展成连续块。这样可以保证最终 mask 是连续区域。

设输入特征图空间尺寸为 $H \times W$，块大小为 $b$，保留概率为 `keep_prob`。丢弃概率定义为：

$$
p_{drop} = 1 - keep\_prob
$$

如果直接让每个中心点以 $p_{drop}$ 的概率命中，最终丢弃面积会远大于预期，因为每个中心点一旦命中，会扩展为 $b \times b$ 的块。因此需要计算中心采样率 $\gamma$：

$$
\gamma =
\frac{
p_{drop} \cdot H \cdot W
}{
b^2 \cdot (H - b + 1) \cdot (W - b + 1)
}
$$

这里的 $(H-b+1)(W-b+1)$ 是合法块中心或合法块起点数量的近似计数，$b^2$ 是每次命中影响的面积。公式的直觉是：目标丢弃总面积约为 $p_{drop}HW$，每次中心命中会贡献约 $b^2$ 个丢弃位置，所以中心命中概率必须更小。

最终输出通常写成：

$$
y = \frac{x \odot (1 - B)}{mean(1 - B)}
$$

其中 $x$ 是输入特征图，$B$ 是扩展后的 block mask，$\odot$ 表示逐元素相乘。`mean(1 - B)` 是保留下来的比例，用它缩放输出，是为了让训练时特征幅值的期望接近未丢弃时的幅值。

流程如下：

```text
1. 采样中心点
   center_mask ~ Bernoulli(gamma)

2. 扩展成块 mask
   每个命中中心扩展为 b x b 区域

3. 生成 block mask B
   B 中 1 表示要丢弃，0 表示保留

4. 归一化输出
   y = x * (1 - B) / mean(1 - B)
```

最小数值例子：取 `H=W=5`，`b=3`，`keep_prob=0.9`，则 `p_drop=0.1`。代入：

$$
\gamma =
\frac{0.1 \times 25}{9 \times 3 \times 3}
=
\frac{25}{810}
\approx 0.0309
$$

意思是中心点命中概率很低，但一旦命中，就会抹掉一个 `3x3` 区域。采样的是少量点，破坏的是连续面。

如果中心点 `(3,3)` 命中，按 1-based 坐标看，`3x3` 扩展结果是：

```text
中心 mask:
0 0 0 0 0
0 0 0 0 0
0 0 C 0 0
0 0 0 0 0
0 0 0 0 0

扩展后 B:
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

---

## 代码实现

实现 DropBlock 的重点是四件事：计算 `gamma`、采样中心、扩展块、只在训练态启用。训练态是指模型正在通过数据更新参数；推理态是指模型参数固定，用来预测新样本。

新手版伪代码如下：

```python
if training:
    center_mask = bernoulli(gamma, shape=(H - b + 1, W - b + 1))
    block_mask = expand_to_blocks(center_mask, block_size=b)
    y = x * (1 - block_mask) / mean(1 - block_mask)
else:
    y = x
```

下面是一个可运行的 NumPy 版本，输入为单张二维特征图，便于看清楚机制：

```python
import numpy as np

def dropblock_2d(x, block_size, keep_prob, training=True, seed=0):
    if not training:
        return x.copy()

    h, w = x.shape
    b = min(block_size, h, w)
    p_drop = 1.0 - keep_prob

    gamma = p_drop * h * w / (b * b * (h - b + 1) * (w - b + 1))

    rng = np.random.default_rng(seed)
    center_mask = rng.random((h - b + 1, w - b + 1)) < gamma

    block_mask = np.zeros((h, w), dtype=np.float32)

    # center_mask 中每个 True 位置扩展为 b x b 连续丢弃区域
    for i in range(h - b + 1):
        for j in range(w - b + 1):
            if center_mask[i, j]:
                block_mask[i:i + b, j:j + b] = 1.0

    keep_mask = 1.0 - block_mask
    keep_mean = keep_mask.mean()

    # 极端情况下全被丢弃，工程实现应避免除以 0
    if keep_mean == 0:
        return np.zeros_like(x)

    return x * keep_mask / keep_mean

x = np.ones((5, 5), dtype=np.float32)
y_eval = dropblock_2d(x, block_size=3, keep_prob=0.9, training=False)
assert np.array_equal(y_eval, x)

y_train = dropblock_2d(x, block_size=3, keep_prob=0.9, training=True, seed=7)
assert y_train.shape == x.shape
assert np.isfinite(y_train).all()
assert y_train.mean() >= 0.0
```

PyTorch 中通常写成模块，在 `forward()` 里判断 `self.training`：

```python
class DropBlock2D(nn.Module):
    def __init__(self, block_size, keep_prob):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training:
            return x
        # 这里省略具体 mask 生成逻辑：
        # 1. 根据 H, W, block_size, keep_prob 计算 gamma
        # 2. 采样中心点
        # 3. 用 max_pool 或卷积把中心点扩展成 block mask
        # 4. x * keep_mask / keep_mask.mean()
        return dropblock_impl(x, self.block_size, self.keep_prob)
```

工程实现常用 `max_pool2d` 扩展 mask。原因是：中心点为 1 后，最大池化可以把附近窗口都变成 1，相当于把点扩成块。

| 实现点 | 正确做法 | 原因 |
|---|---|---|
| `gamma` 计算 | 结合 `H, W, block_size, keep_prob` | 不同特征图尺寸下丢弃强度要可控 |
| `block_mask` 生成 | 中心采样后扩展 | 保证丢弃区域连续 |
| 输出缩放 | 除以保留比例 | 保持特征幅值大致稳定 |
| 训练态开关 | 只在 `train()` 启用 | 推理需要确定性输出 |
| 参数调度 | `p_drop` 从 0 线性升高 | 避免训练初期扰动过强 |

---

## 工程权衡与常见坑

DropBlock 比普通 dropout 更强，但更强不等于更好。它直接删除连续空间区域，参数设置错误时会破坏训练信号，尤其是在浅层和小特征图上。

第一个常见坑是 `block_size` 过大。比如特征图只有 `7x7`，还使用 `b=5`，一次命中就可能遮住大半张特征图。模型不是被迫学习更好的特征，而是经常拿不到足够信息，训练会变慢甚至不稳定。

第二个常见坑是把 `keep_prob` 和 `drop_prob` 混用。`keep_prob=0.9` 表示保留 90%，丢弃 10%；`drop_prob=0.9` 表示丢弃 90%。这两个数字含义相反，混用会让正则强度完全偏离预期。

第三个常见坑是没有绑定训练和推理状态。DropBlock 应该只在训练中启用。推理时仍随机遮挡特征，会导致同一张图多次预测结果不一致。

| 坑 | 后果 | 规避方式 |
|---|---|---|
| `block_size` 过大 | 特征图被过度破坏 | 保证 `b < H,W`，小图用小块 |
| `p_drop` 过大 | 训练初期不稳定 | 使用线性调度 |
| `keep_prob` / `drop_prob` 混用 | 实际丢弃强度反向 | 统一用 `p_drop = 1 - keep_prob` |
| 推理时未关闭 | 预测结果异常波动 | 只在 `model.train()` 中启用 |
| 小特征图硬用固定 `b` | mask 异常或删除过多 | 按层调整或 clamp |
| 当成普通 dropout | 场景不匹配 | 优先用于卷积特征图 |

线性调度是训练 DropBlock 时常用的做法。它表示前期扰动小，后期逐步增加到目标强度：

```text
p_drop
0.10 |                         *
0.08 |                    *
0.06 |               *
0.04 |          *
0.02 |     *
0.00 | *
     +-----------------------------
       early       middle      late
```

这个设计的原因是：训练初期模型还没有学到稳定特征，过早遮挡大块区域会让梯度信号很噪；训练后期模型已有较强拟合能力，再增加正则化更合理。

真实工程中，DropBlock 经常不是全网统一一个参数。浅层特征图大、语义弱，可以用较小 `p_drop`；深层特征图小、语义强，要谨慎控制 `block_size`。检测和分割模型里，backbone、neck、head 的特征尺寸不同，统一套一个 `b=7` 往往不是好选择。

---

## 替代方案与适用边界

DropBlock 的适用条件可以概括为四个问题：

| 判断问题 | 倾向使用 DropBlock 的答案 |
|---|---|
| 是否是卷积特征图？ | 是 |
| 是否存在明显空间相关性？ | 是 |
| 特征图尺寸是否足够大？ | 是 |
| 训练是否允许更强正则？ | 是 |

如果这些答案大多是否定的，DropBlock 就不一定是优先选择。

| 方法 | 机制 | 更适合的场景 | 局限 |
|---|---|---|---|
| Dropout | 随机丢弃独立点 | MLP、分类头、通用正则 | 不专门处理空间相关性 |
| DropBlock | 随机丢弃连续区域 | CNN backbone、检测、分割 | 小特征图上容易过强 |
| Stochastic Depth | 随机跳过整层或残差分支 | 很深的残差网络 | 粒度比 DropBlock 更粗 |
| Cutout / Random Erasing | 遮挡输入图像区域 | 图像数据增强 | 作用在输入层，不直接约束中间特征 |
| Spatial Dropout | 丢弃整个通道 | 通道冗余明显的卷积网络 | 不控制空间局部依赖 |

Stochastic Depth 是随机深度方法，训练时随机跳过某些层或残差分支。它解决的是“深网络中层级依赖过强”的问题，不是局部空间区域依赖问题。

Cutout 和 Random Erasing 是输入层数据增强方法。它们遮挡的是原始图像的一部分。DropBlock 遮挡的是中间特征图，两者可以同时使用，但正则强度需要重新评估。

新手版对比可以这样记：图像分类 CNN、目标检测、实例分割中，DropBlock 常常有价值；纯 MLP 或小型全连接分类头中，普通 dropout 更直接；如果训练中已经有很强的数据增强、强归一化、较浅网络，DropBlock 的边际收益可能下降。

工程选择不应只看论文结果。更稳妥的方式是做消融实验。消融实验是指只改变一个因素，观察指标变化。比如保持数据增强、学习率、训练轮数不变，只比较 `no DropBlock`、`p_drop=0.05`、`p_drop=0.1` 三组结果。这样才能判断收益来自 DropBlock，而不是其他训练配置。

---

## 参考资料

1. [DropBlock: A regularization method for convolutional networks](https://papers.nips.cc/paper_files/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html)
2. [Hugging Face Papers: 1810.12890](https://huggingface.co/papers/1810.12890)
3. [torchvision.ops.drop_block 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/drop_block.html)
4. [miguelvr/dropblock 参考实现](https://github.com/miguelvr/dropblock)

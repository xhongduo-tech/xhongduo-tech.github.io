## 核心结论

Progressive Resizing = 输入分辨率逐步增大、同一任务分阶段训练。

它是一种训练策略：先用低分辨率图像训练，让模型快速学到整体结构；再逐步切到更高分辨率，让模型补足细节。任务不变、标签不变、模型参数连续继承，变化的是输入图像尺寸。

玩具例子：训练猫狗分类时，先用 `128x128` 图像训练几轮，模型先学会猫和狗的大致轮廓；再切到 `224x224`、`384x384`，模型继续学习耳朵、胡须、毛发纹理等细节。这里没有把“猫狗分类”换成别的任务，只是让模型先看低清图，再看高清图。

它的核心价值不是提升模型表达能力，而是降低训练早期的计算成本。低分辨率图像像素少，单步训练更快；高分辨率图像保留更多细节，用于最后收敛到目标精度。典型配置是 `128 -> 224 -> 384`，每个阶段训练约 `1/3` 的 epoch。实际收益依赖任务和数据管线，常见结果是总训练时间减少 `30% - 50%`，最终精度损失通常小于 `1%`。

| 方法 | 变化内容 | 目标 |
|---|---|---|
| Progressive Resizing | 输入分辨率逐步升高 | 降低前期训练成本 |
| 数据增强 | 输入内容随机变换 | 提高泛化 |
| Curriculum Learning | 样本难度逐步升高 | 调整学习顺序 |

---

## 问题定义与边界

Progressive Resizing 主要解决的是图像训练中的计算效率问题。它适合这样的任务：低分辨率已经能提供足够的全局结构，高分辨率主要用于补充细节。

全局结构是指图像里大尺度的形状、布局和区域关系。比如猫狗分类里，身体轮廓、头部位置、姿态通常在低分辨率下仍然存在。细粒度信息是指很小的局部像素差异，比如文字笔画、微小缺陷、小目标边界。

真实工程例子：病理切片分类可以先用低倍图判断组织结构，再放大看细胞核、核分裂和微小病灶。遥感大图分类也类似，低分辨率能先学到道路、水体、建筑群等大范围模式，高分辨率再补细节。

但 OCR 不太适合从很低分辨率开始。OCR 是光学字符识别，目标是从图像中识别文字。字符笔画本身就是答案的一部分，缩图过度会直接抹掉判别信息。小目标检测也类似，目标在低分辨率下可能只剩几个像素，甚至完全消失。

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 图像分类 | 适合 | 低分辨率先学轮廓 |
| 病理切片分类 | 适合 | 先结构后细节 |
| 遥感大图分类 | 适合 | 大范围上下文重要 |
| OCR | 不太适合 | 字符细节依赖高分辨率 |
| 小目标检测 | 不太适合 | 目标在低分辨率下可能消失 |
| 超细边界分割 | 不太适合 | 边界信息会被抹掉 |

边界定义：如果任务的关键判别信息主要来自局部细节，而不是全局结构，Progressive Resizing 的收益会明显下降。

---

## 核心机制与推导

设训练分成 $K$ 个阶段，分辨率依次为 $r_1 < r_2 < \dots < r_K$。第 $i$ 个阶段不是重新训练一个新模型，而是继承上一阶段的参数：

$$
\theta_{i,0} = \theta_{i-1,E_{i-1}}
$$

其中 $\theta$ 表示模型参数，$E_{i-1}$ 表示上一阶段训练结束时的 epoch。白话解释：第一阶段训练出的模型权重，会直接作为第二阶段的初始权重。

每个阶段优化的是同一个任务损失：

$$
L_i(\theta)=\mathbb{E}_{(x,y)\sim D}\,\ell(f_\theta(T_{r_i}(x)),y)
$$

这里 $D$ 是训练数据集，$x$ 是输入图像，$y$ 是标签，$T_{r_i}(x)$ 表示把图像处理到分辨率 $r_i$，$f_\theta$ 是模型，$\ell$ 是损失函数。损失函数是衡量模型预测和真实标签差距的函数。公式的重点是：标签没变，任务没变，只是输入图像经过了不同分辨率的变换。

省时来自计算量。对常见卷积网络，单步计算量通常近似随图像面积增长：

$$
\text{FLOPs}(r)\propto r^2
$$

FLOPs 是浮点运算次数，常用来估算模型计算成本。分辨率从 `128` 增加到 `384`，边长变成 3 倍，面积约变成 9 倍，单步计算也会明显增加。

玩具数值例子：总共训练 9 个 epoch，按 `128 / 224 / 384` 各训练 3 个 epoch。假设 `384` 下单个 epoch 需要 9 分钟，并按面积粗略缩放，那么 `128` 约 1 分钟，`224` 约 3.4 分钟，`384` 约 9 分钟。总时间约为：

$$
3\times1 + 3\times3.4 + 3\times9 = 40.2
$$

如果全程使用 `384`，时间是：

$$
9\times9 = 81
$$

这个简化例子里，时间约省一半。真实训练中还会受到数据读取、增强、显存、并行效率影响，所以不能把这个数字当成保证值。

| 分辨率 | 单步代价 | 作用 |
|---|---|---|
| 低 | 低 | 快速探索、学粗特征 |
| 中 | 中 | 过渡、修正结构 |
| 高 | 高 | 学细节、收敛到目标精度 |

机制可以概括为：低分辨率阶段先做粗粒度参数搜索，让模型进入一个合理参数区域；高分辨率阶段在已有表示上细化。训练不是从零开始三次，而是逐步升级同一个模型。

---

## 代码实现

代码层面的核心不是动态改模型，而是在不同 epoch 段切换数据预处理、输入尺寸、batch size 和学习率。

对于支持可变输入的卷积网络，以及带自适应池化的分类模型，输入尺寸切换通常不需要改网络结构。自适应池化是指模型在分类头前把不同空间尺寸的特征图压到固定大小，从而允许输入图像尺寸变化。

下面是一个可运行的 Python 玩具实现，用面积近似估算不同训练计划的耗时：

```python
def relative_epoch_cost(size, target_size=384, target_minutes=9.0):
    return target_minutes * (size / target_size) ** 2

def schedule_minutes(stages):
    total = 0.0
    for stage in stages:
        total += stage["epochs"] * relative_epoch_cost(stage["size"])
    return total

progressive_stages = [
    {"size": 128, "epochs": 3, "lr": 1e-3},
    {"size": 224, "epochs": 3, "lr": 3e-4},
    {"size": 384, "epochs": 3, "lr": 1e-4},
]

fixed_high_res = [
    {"size": 384, "epochs": 9, "lr": 1e-4},
]

progressive_time = schedule_minutes(progressive_stages)
fixed_time = schedule_minutes(fixed_high_res)

assert round(progressive_time, 1) == 40.2
assert fixed_time == 81.0
assert progressive_time < fixed_time
```

真实训练伪代码如下：

```python
stages = [
    {"size": 128, "epochs": 3, "lr": 1e-3},
    {"size": 224, "epochs": 3, "lr": 3e-4},
    {"size": 384, "epochs": 3, "lr": 1e-4},
]

for stage_idx, stage in enumerate(stages):
    train_loader = make_dataloader(
        image_size=stage["size"],
        batch_size=get_batch_size(stage["size"])
    )

    optimizer = set_lr(optimizer, stage["lr"])

    if stage_idx > 0:
        warmup_short(optimizer)

    for epoch in range(stage["epochs"]):
        train_one_epoch(model, train_loader, optimizer)
```

| 实现点 | 为什么要做 |
|---|---|
| 分阶段 Resize | 控制输入分辨率 |
| 切阶段降学习率 | 减少震荡 |
| 短 warmup | 缓解分辨率切换冲击 |
| 调整 batch size | 高分辨率下显存更紧 |
| 监控吞吐 | 防止 CPU / I/O 成为瓶颈 |

学习率是每次参数更新的步长。分辨率切换后，输入分布发生变化，特征图大小、增强效果和梯度统计都可能变化，所以通常要降低学习率，必要时加一个很短的 warmup。warmup 是指训练阶段开始时先用较小学习率，再逐步升到目标学习率，用来降低训练初期震荡。

---

## 工程权衡与常见坑

Progressive Resizing 的收益来自低分辨率阶段更便宜，但它不是免费加速。分辨率切换会改变输入分布，训练曲线可能出现短期 loss 上升。loss 是损失值，越低通常表示模型在训练集上的预测误差越小。轻微上升是正常现象，关键看切换后是否能快速恢复并继续下降。

最常见的坑是切阶段后沿用原学习率。低分辨率阶段的较大学习率用于快速探索，高分辨率阶段继续使用同样步长，可能导致 loss 抖动甚至精度退化。更稳妥的做法是每次升分辨率时降低学习率，或者重新做一次短 warmup。

第二个坑是低分辨率阶段太长。如果前 `80%` 的训练都在低分辨率完成，模型会过度适应粗特征。最后突然看高清图时，它只能做局部补救，整体表示已经偏了。实践上，低分辨率通常只占前 `1/3` 到 `1/2` 训练。

第三个坑是数据管线瓶颈。图像变小后，GPU 单步计算变快，但 CPU 读取、JPEG 解码、随机增强可能跟不上，导致 GPU 空转。吞吐是单位时间处理的样本数，低分辨率阶段应该重点观察吞吐是否真的提升。

| 坑 | 表现 | 规避方式 |
|---|---|---|
| 切阶段不降学习率 | loss 震荡 | 重新设定更小 lr |
| 低分辨率阶段过长 | 后期精度补不回 | 低分辨率只占前 1/3 到 1/2 |
| 任务依赖细节 | 性能下降 | 提高最低分辨率或不用 |
| 数据管线变慢 | GPU 空转 | 缓存、加 worker、加速解码 |
| batch size 不重调 | 显存爆或吞吐差 | 分辨率升高时同步调 batch |

真实工程例子：工业缺陷检测中，如果缺陷是大面积划痕，Progressive Resizing 可能有效；如果缺陷是几个像素宽的裂纹，低分辨率阶段可能直接把裂纹抹掉。这时不应该从 `128` 开始，最低分辨率要提高，或者直接固定高分辨率训练。

---

## 替代方案与适用边界

Progressive Resizing 不是唯一的多尺度训练方法。它的特点是训练节奏固定：先低、再中、最后高。另一类方法是 Random Multi-scale Training，即随机多尺度训练，每个 batch 或每隔几步随机使用不同输入尺寸。

新手可以这样区分：Progressive Resizing 像先看缩略图，再看原图；随机多尺度训练像每次训练都随机看不同大小的图。二者都和尺度有关，但目标不同。前者更偏向节省前期计算，后者更偏向提升模型对尺度变化的鲁棒性。鲁棒性是指输入发生合理变化时，模型仍能保持稳定表现。

| 方法 | 核心思路 | 优点 | 缺点 |
|---|---|---|---|
| Progressive Resizing | 分阶段升分辨率 | 省前期计算 | 切换时要调参 |
| Random Multi-scale Training | 训练中随机尺度 | 鲁棒性更强 | 训练更不稳定 |
| 固定高分辨率训练 | 全程目标尺寸 | 简单直接 | 成本高 |
| 图像金字塔 / 多分支 | 多尺度并行 | 表达能力强 | 结构复杂 |

如果任务收益主要来自全局结构，优先考虑 Progressive Resizing。比如普通图像分类、病理切片分类、遥感大图分类，低分辨率阶段可以先学到有用表示。

如果任务收益主要来自细节精度，优先考虑固定高分辨率或更保守的多尺度策略。OCR 识别里字符笔画就是关键信息，太早缩图会直接丢失答案；小目标检测里目标可能在低分辨率下消失；超细边界分割里边界会被插值和缩放抹平。

工程上可以用一个简单判断：先把训练集样本缩到最低候选分辨率，人工看一批图。如果人眼已经难以判断标签，模型也很难从这个阶段学到正确信号。此时应该提高起始分辨率，或者放弃 Progressive Resizing。

---

## 参考资料

1. [EfficientNetV2: Smaller Models and Faster Training](https://research.google/pubs/efficientnetv2-smaller-models-and-faster-training/)：支持 progressive learning 与逐步增大图像尺寸可以加速训练的研究背景。
2. [fastai Book, Chapter 7: Training a State-of-the-Art Model](https://fastai.github.io/fastbook2e/book7.html)：提供 progressive resizing 的经典实践说明。
3. [fastai Book, Chapter 14: ResNets](https://fastai.github.io/fastbook2e/resnet.html)：解释卷积网络、全卷积结构和自适应池化为何能处理不同输入尺寸。
4. [MosaicML Composer: Progressive Image Resizing](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/progressive_resizing.html)：提供工程实现、训练策略和超参数建议。

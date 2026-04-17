## 核心结论

μP，全称 Maximal Update Parameterization，中文可理解为“让参数更新在放大模型时仍保持同一数量级的参数化方式”。它解决的不是“模型能不能继续训练”，而是“把宽度从小扩到大以后，原来调好的超参数为什么还能继续有效”。

核心结论有两条。

第一，μP的目标不是固定权重本身，而是固定各层表征的相对更新规模。表征，白话说，就是网络中间层提取出来的特征向量。若训练一步后中间激活从 $h$ 变成 $h+\Delta h$，μP希望 $\Delta h/h$ 不随宽度变化而漂移。这样，小模型和大模型虽然参数个数不同，但“每一步学到多少新特征”保持同一量级。

第二，只要模型是沿着“宽度”扩展，且初始化、读出层、优化器缩放满足μP规则，那么在小模型上搜到的学习率、权重衰减、warmup 等超参数，常常可以直接迁移到大模型。这种做法在论文里叫 μTransfer，即“零样本超参数迁移”。

先看一个最小规则表。这里的 width 指隐藏维度或通道数。

| 部分 | 标准做法直觉 | μP做法直觉 | 目的 |
| --- | --- | --- | --- |
| 隐藏层初始化 | 常见为 fan-in 缩放 | 保持与宽度一致的 $1/\sqrt{\text{width}}$ 量级 | 让前向激活稳定 |
| 隐藏层更新 | 容易随宽度变大而变弱或变强 | 让特征更新保持 $\Theta(1)$ | 保持 feature learning |
| 读出层学习率 | 常直接与别的层相同 | 需要额外按 width 缩小 | 防止输出层更新过猛 |
| 调参流程 | 每个大模型都重调 | 先调小模型，再迁移 | 降低调参成本 |

玩具例子：设小模型宽度是 64，大模型宽度是 4096。若你在宽度 64 上试出一个能稳定下降的学习率，标准参数化下直接复制到 4096 往往会失效，因为输出层和隐藏层的更新比例变了。μP的作用就是把这种“比例变形”消掉。

真实工程例子：Tensor Programs V 报告，把约 13M 参数的代理模型上调好的超参数迁移到 BERT-large 约 350M 参数模型；又把约 40M 代理模型的超参数迁移到 6.7B GPT-3 级别模型。重点不是某个数字本身，而是“大模型不再从头做整套搜索”。

---

## 问题定义与边界

问题定义很具体：当我们只放大模型宽度，不改网络家族时，能不能让“最优超参数”对宽度近似不敏感？如果可以，那么调参就不再需要围绕最大模型重复烧算力。

这里的超参数，白话说，就是训练前你先手工设定、训练过程中不会直接被梯度学习出来的量，比如学习率、权重衰减、batch size、warmup 步数。μP主要处理的是其中最容易随宽度失真的部分，尤其是学习率相关尺度。

它的边界也很明确。

| 条件 | 是否属于μP主要适用范围 | 说明 |
| --- | --- | --- |
| 只改宽度，深度不变 | 是 | 这是论文和工具支持最强的情况 |
| 同一网络家族内扩展 | 是 | 例如同一类 Transformer、MLP、ResNet |
| 深度也一起大改 | 部分适用 | 需重新做额外验证 |
| 不规则结构、很多自定义张量乘法 | 风险较高 | 需要手动指定哪些维度是“无限维” |
| 只想研究无限宽线性化极限 | 不一定首选 | 这类问题常用 NTK 参数化更直接 |

可把μP的核心约束写成一个缩放目标：

$$
\frac{\|\Delta h\|}{\|h\|} \approx \text{constant w.r.t. width}
$$

其中 $\Delta h$ 是一步训练带来的激活变化。若这个比值随着 width 变成接近 0，说明模型越来越像“只动最后一层”；若它爆炸，说明宽模型训练会变得不稳定。

常见的直观规则可以概括为：

$$
\sigma_{\text{hidden}} \propto \frac{1}{\sqrt{\text{width}}}, \qquad
\eta_{\text{readout}} \propto \frac{1}{\text{width}}
$$

这里 $\sigma$ 是初始化标准差，$\eta$ 是学习率。对初学者更重要的一句解释是：μP不是把所有层都用同一个裸学习率，而是让“有效更新尺度”一致。实际代码里，这件事通常交给专门的 μP 优化器和 `MuReadout` 层来做，而不是人工按层拍脑袋写死。

---

## 核心机制与推导

先定义 abc 参数化。它是一套把“前向缩放、初始化缩放、学习率缩放”统一写进宽度指数的方法。白话说，就是把每个量都写成 width 的幂次，看模型放大时哪部分会失衡。

把某层写成简化形式：

$$
h^{l+1} = n^{-a_l} W^l h^l
$$

其中 $n$ 是宽度，$a_l$ 控制前向时这层额外除以多少宽度因子；初始化可写成 $W^l_0 \sim n^{-b_l}$ 的量级；学习率写成 $\eta_l \sim n^{-c_l}$。于是，一次更新后的表征变化会带上某个关于 $n$ 的指数。μP要做的，就是把这个指数调成 0，也就是让更新规模是 $\Theta(1)$。

这里的 $\Theta(1)$，白话说，就是“不随宽度继续变大或变小，始终维持常数量级”。

因此，μP追求的是：

$$
\Delta h^l = \Theta(1), \qquad
\frac{\Delta h^l}{h^l} = \Theta(1)
$$

论文把这种“特征学习不会随宽度消失”的极限称为 maximal update。若把特征更新的宽度指数记成 $r$，那么 μP对应的关键选择就是：

$$
r = 0
$$

$r=0$ 的含义不是“更新等于 0”，而是“更新不随 width 缩成 0，也不随 width 爆掉”。这正是小模型和大模型能共享超参数的必要条件。

可用一个简化推导来理解。

1. 前向激活要稳定，所以隐藏层初始化通常保持 fan-in 归一化，即每个坐标的方差不因 width 增长而膨胀。
2. 反向梯度经过宽层求和后，会自然带上宽度因子。
3. 若学习率不做匹配，$\Delta W h$ 会随 width 偏大或偏小。
4. μP通过对读出层和部分隐藏权重使用不同的有效缩放，抵消这个宽度因子。
5. 结果是各层表征更新保持同一量级，超参数曲线在不同宽度上对齐。

可以把它压成一张“流程图”。

| 步骤 | 若不处理会怎样 | μP的处理 |
| --- | --- | --- |
| 宽度变大 | 梯度和激活统计改变 | 先标记哪些维度是 width |
| 初始化保持旧写法 | 某些层激活或梯度漂移 | 按 μP 初始化重标定 |
| 学习率仍统一 | 输出层更新过大最常见 | 读出层额外缩放 |
| 训练若看起来能跑 | 但最优 lr 不再可迁移 | 用 coord check 验证坐标统计 |

玩具例子：两层 MLP，宽度从 128 扩到 8192。如果最后一层仍用和隐藏层一样的裸学习率，那么 logits 的更新通常会比中间特征更敏感，表现为大模型最优学习率显著往左移。μP把读出层单独处理，就是为了把这条最优学习率曲线拉回重合。

真实工程例子：在 Transformer 里，除了读出层，注意力缩放也要匹配 μP 约定。`microsoft/mup` 文档明确提醒，注意力分数应按 $1/d$ 而不是常见的 $1/\sqrt{d}$ 形式做 μP 兼容缩放，否则宽度扩展时坐标统计会偏掉。

---

## 代码实现

实践里最稳妥的做法不是手搓一堆 if/else，而是使用 `mup` 包提供的三件事：`MuReadout`、`set_base_shapes`、`MuAdam` 或 `MuSGD`。`base shapes`，白话说，就是先定义一个基准小模型和一个变化模型，让库知道哪些维度代表“会被放大的宽度”。

先给一个可运行的玩具脚本，只演示“读出层学习率按 width 缩放”的核心想法：

```python
def mup_readout_lr(base_lr: float, width: int) -> float:
    assert width > 0
    return base_lr / width

def hidden_init_std(width: int) -> float:
    assert width > 0
    return width ** -0.5

base_lr = 2e-3

# 玩具例子：宽度越大，读出层学习率越小
assert abs(mup_readout_lr(base_lr, 512) - 3.90625e-6) < 1e-12
assert abs(mup_readout_lr(base_lr, 1024) - 1.953125e-6) < 1e-12

# 隐藏层初始化保持 1/sqrt(width) 量级
assert abs(hidden_init_std(4) - 0.5) < 1e-12
assert abs(hidden_init_std(100) - 0.1) < 1e-12

# 相同 base_lr 下，宽模型读出层更新更保守
assert mup_readout_lr(base_lr, 1024) < mup_readout_lr(base_lr, 128)
```

如果你不用官方库，最少也要把参数组拆开：

```python
width = 512
base_lr = 2e-3
readout_lr = base_lr / width

param_groups = [
    {"name": "hidden", "lr": base_lr},
    {"name": "readout", "lr": readout_lr},
]
```

但真正工程实现更推荐下面这种写法，因为它会根据 `infshape` 自动调整有效学习率，而不是只靠参数名匹配：

```python
from mup import MuReadout, set_base_shapes, MuAdam
import mup

# 1. 定义 base / delta / target 三个模型，深度一致，只改宽度
base_model = MyModel(width=128)
delta_model = MyModel(width=256)
model = MyModel(width=2048)

# 2. 尽早设置 base shapes，最好在重新初始化和构造 optimizer 之前
set_base_shapes(model, base_model, delta=delta_model)

# 3. 若手动初始化，使用 mup.init 中的版本
for p in model.parameters():
    mup.init.xavier_uniform_(p)

# 4. 输出层应使用 MuReadout，优化器用 MuAdam / MuSGD
optimizer = MuAdam(model.parameters(), lr=1e-3)
```

真实工程例子：如果你训练一个自定义 Transformer，需要同时检查三件事。

1. 读出层是否替换成 `MuReadout`。
2. 注意力缩放是否按 μP 规则调整。
3. scheduler 是否“相对地”改当前参数组学习率，而不是绝对覆盖。因为 `mup` 已经细分过参数组，若你直接把所有参数组都写回同一个裸 `lr`，等于把 μP 缩放又抹掉一次。

coord check 是另一个关键工具。它的作用是看不同 width 下，激活、梯度、更新统计是否画成近似水平线。水平线，白话说，就是“宽度变了，但每层坐标尺度没明显飘”。没有这个检查，就不能确认你实现的是 μP，而只是“看起来像”。

---

## 工程权衡与常见坑

μP的收益很大，但前提是实现严格。它最常见的问题不是理论错，而是工程上只做了一半。

| 常见坑 | 现象 | 后果 | 规避方式 |
| --- | --- | --- | --- |
| 只改初始化，不改优化器 | 小模型有效，大模型失灵 | 最优 lr 仍随 width 漂移 | 用 `MuAdam`/`MuSGD` |
| 输出层没用 `MuReadout` | logits 更新过猛 | 大模型更容易震荡 | 替换最后读出层 |
| 没跑 coord check | 误以为实现正确 | 扩宽后激活爆炸或衰减 | 先验证再正式训练 |
| scheduler 绝对覆盖 lr | 把 μP 参数组缩放抹掉 | 训练表现异常 | 只做相对乘法更新 |
| 自定义张量主维方向错了 | 宽度识别错误 | 某些层缩放反了 | 手动设置 `infshape.main_idx` |
| 用 `DataParallel` | 丢失 `infshape` 属性 | 训练中规则失效 | 用 `DistributedDataParallel` |

有一个常见误解需要单独说清：很多入门总结会写成“隐藏层 lr 恒定，输出层 lr 按 width 缩小”。这句话在概念层面帮助理解可以，但在真实 PyTorch 实现里不够精确。`mup` 官方实现会根据参数张量的 `infshape.width_mult()` 自动计算学习率缩放，尤其在 Adam 下，隐藏权重的有效学习率也可能体现为 `globalLR / width_mult()`。所以工程上不要把口号直接翻译成固定常数，而要让库根据张量角色自动处理。

---

## 替代方案与适用边界

最常被拿来比较的是 NTK parameterization。NTK，白话说，就是把网络放到“无限宽但特征几乎不动”的线性化近似里分析训练。

两者差异可以压成一张表：

| 方案 | 目标 | 特征学习能力 | 超参数迁移 | 调参成本 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| μP | 保持更新尺度不变 | 强 | 强 | 低 | 需要真正学新特征的大模型 |
| NTK 参数化 | 保持核极限稳定 | 弱到中 | 一般不是重点 | 中 | 理论分析、线性化近似 |
| 传统经验调参 | 直接在目标模型试 | 强 | 弱 | 高 | 小中型模型或预算充足 |
| AutoML / 大规模 sweep | 自动搜索 | 强 | 弱 | 很高 | 预算极高、追求全局最优 |

什么时候优先用 μP？

1. 宽度扩展明确，模型家族稳定。
2. 训练预算贵，不能在目标模型上反复试。
3. 任务依赖 feature learning，比如语言模型预训练、大型表征学习。

什么时候不要盲目上 μP？

1. 你主要在改深度、序列长度、数据配方，而不是宽度。
2. 模型结构很不规则，很多分支没有清楚的 width 维。
3. 只是做一个小模型实验，没有超参数迁移需求。
4. 你并不打算使用 coord check 验证实现。

一句话概括边界：μP最适合“同一架构按宽度放大”的训练体系；一旦你改变的是别的缩放轴，μP仍可能有帮助，但不能默认超参数必然零迁移。

---

## 参考资料

| 资料 | 适合用途 | 说明 |
| --- | --- | --- |
| [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) | 理论 + 实证 | μP 与 μTransfer 的核心论文，包含 BERT / GPT 迁移结果 |
| [microsoft/mup README](https://github.com/microsoft/mup) | 工程实践 | `MuReadout`、`set_base_shapes`、`MuAdam`、coord check 的官方入口 |
| [microsoft/mup CoordCheck 文档与示例](https://github.com/microsoft/mup) | 工具使用 | 观察激活、梯度、更新统计是否随 width 保持平稳 |
| [Microsoft 关于 μTransfer 的介绍](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/) | 概念入门 | 从训练成本角度解释为什么先调小模型再迁移可行 |

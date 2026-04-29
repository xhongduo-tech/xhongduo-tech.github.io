## 核心结论

μP（Maximal Update Parametrization）是一套按“宽度”统一缩放初始化与学习率的参数化框架。白话说，它先规定“模型变宽时每类参数该怎么变”，再去调超参数。这样，小模型上调好的学习率、初始化尺度、warmup、schedule，才有机会迁移到大模型上，而不是每放大一次模型就重新做一轮昂贵搜索。

它的价值不是“自动提升上限”，而是降低扩展成本。普通参数化下，宽度一变，训练动力学也会变，导致“小模型最优学习率”在大模型上失效。μP 的目标是让跨宽度的训练行为更稳定，从而支持 μTransfer，也就是“小模型调参，大模型零样本迁移”。

| 对比项 | 普通参数化 | μP |
|---|---|---|
| 宽度变化后最优超参数 | 常漂移 | 更稳定 |
| 小模型调参价值 | 只能部分参考 | 更接近大模型代理 |
| 关注点 | 先训练再补救 | 先规定缩放规则再训练 |
| 工程成本 | 接入简单 | 接入更严格 |

核心缩放关系可以写成：

$$
\eta_p=\eta_{\text{global}}/m(p)
$$

其中 $m(p)$ 是参数张量 $p$ 的宽度倍数。白话说，它表示“这个参数相对基准模型被放大了多少”。

---

## 问题定义与边界

问题先定义清楚。你在隐藏宽度为 128 的模型上找到一个很好用的学习率，换到宽度 512 时却训练发散、变慢，或者最优点完全换了位置。这通常不是“调参不够认真”，而是模型宽度变化后，参数尺度、激活尺度、梯度尺度一起变了，原来的超参数不再处在同一个训练 regime。regime 可以理解为“训练系统运行的整体状态区间”。

| 现象 | 原因 | 后果 |
|---|---|---|
| 大模型要重新搜学习率 | 宽度变了，更新尺度变了 | 调参成本高 |
| 小模型初始化稳定，大模型不稳定 | 激活或输出尺度漂移 | 前几步就炸掉 |
| 同一 schedule 在不同宽度效果差很多 | 训练动力学不一致 | 代理模型失真 |

μP 的边界也要说清楚。

1. 它主要解决“宽度扩展下的超参数迁移”问题，不是所有训练问题的万能解。
2. 它要求 base shape 和 delta shape 定义正确。base shape 可以理解为“基准模型尺寸说明书”，delta shape 可以理解为“告诉系统哪些维度会被放大”。
3. 它不是“只改学习率”的技巧，而是初始化、参数化、优化器三件事一起改。
4. 它对“主要沿宽度放大”的场景最有效；如果你同时大改深度、注意力结构、词表、归一化方式，迁移效果会下降。
5. 它也不保证大模型一定更好，只保证你更有机会把小模型上找到的好设置带过去。

一个新手常见误区是：把 μP 理解成“学习率除以宽度倍数”。这不够。因为学习率只是训练更新的一部分，初始化和层级缩放规则不一致，整体坐标量级仍然会漂。

---

## 核心机制与推导

μP 的核心思想不是固定“参数值”，而是尽量固定训练过程中各层的坐标量级。坐标量级可以理解为“某层单个神经元看到的典型数值尺度”。目标是让宽模型和窄模型在训练早期表现出相近的动力学：

$$
\text{preactivation}=\Theta(1),\quad \text{output}=O(1)
$$

这里 $\Theta(1)$ 表示“随宽度变化但量级保持常数”，$O(1)$ 表示“不会随着宽度无限增大”。

### 1. 用 base model 定义参照系

你先选一个基准模型 `base_model`，再定义一个 `delta_model`，它和基准模型结构相同，但所有想扩展的宽度维度都要变一下。这样系统才能知道：哪些维度是“有限维”，哪些是“无限维”。无限维的意思不是数学上真的无穷，而是“随着模型规模一起放大”。

### 2. 给每个参数定义宽度倍数

对参数张量 $p$，定义：

$$
m(p)=\mathrm{width\_mult}(p)
$$

对常见隐藏层权重，可以近似理解为：

$$
m(p)\approx \frac{\mathrm{fan\_in}(p)}{\mathrm{base\ fan\_in}(p)}
$$

`fan_in` 是输入通道数，白话说就是“有多少路输入汇到这个参数上”。

### 3. 按参数级缩放学习率

μP 的关键不是只有一个全局学习率，而是不同参数张量的实际学习率不同：

$$
\eta_p=\eta_{\text{global}}/m(p)
$$

这意味着模型从宽度 128 变到 512 时，某些层的学习率可能缩小 4 倍，但另一些不随宽度变的参数不一定跟着缩。

### 4. 初始化也要配套重标定

如果你只改 optimizer，不改初始化，那么训练第一步前的激活尺度就可能已经不对。μP 的库会在 `set_base_shapes` 后给参数挂上形状信息，再通过 `mup.init` 或模型自带的 μP 兼容初始化，让缩放规则贯穿到训练开始前。

### 玩具例子

假设基准模型隐藏宽度 $w_0=128$，目标模型隐藏宽度 $w=512$。某一层隐藏权重的 `fan_in` 随宽度同比放大，那么：

$$
m(p)=512/128=4
$$

如果代理小模型上该层最优学习率是 $3\times10^{-4}$，那么目标模型对应层的学习率应变为：

$$
\eta_p=3\times10^{-4}/4=7.5\times10^{-5}
$$

这不是经验拍脑袋，而是参数化规则的一部分。

### Transformer 里的特殊点

Transformer 的注意力分数缩放在普通实现里常用 $1/\sqrt d$。但按 `mup` 官方 README 的实现建议，做 μP 时要把注意力缩放改成与 $1/d$ 同阶的规则，仓库示例用的是 `8/d`，目的是在常见 `d=64` 头维度下与旧实现保持可比。这里的重点不是“8”这个常数，而是它不应继续沿用默认的 $1/\sqrt d$ 缩放逻辑，否则宽度放大后的训练动态会偏离 μP 设定。

| 部分 | 作用 | 结果 |
|---|---|---|
| base shape | 定义基准参照 | 知道什么叫“放大 1 倍” |
| delta model | 标出会变化的宽度维度 | 正确识别缩放方向 |
| 参数级学习率 | 每类参数按自己的 $m(p)$ 缩放 | 跨宽度更稳定 |
| 初始化重标定 | 保持训练前激活尺度一致 | 早期训练更稳 |
| Coord Check | 检查坐标量级是否水平 | 及时发现实现错误 |

---

## 代码实现

工程上最重要的不是 API 名字，而是顺序。正确顺序通常是：先 `set_base_shapes`，再初始化，再创建 optimizer。因为 optimizer 会记录参数状态，初始化也会决定训练起点，顺序错了，规则就断了。

先给一个可运行的玩具代码，只演示“按参数级学习率缩放”的最小逻辑：

```python
from dataclasses import dataclass

@dataclass
class ParamSpec:
    name: str
    fan_in: int
    base_fan_in: int

    def width_mult(self) -> float:
        return self.fan_in / self.base_fan_in

def mup_lr(global_lr: float, p: ParamSpec) -> float:
    return global_lr / p.width_mult()

base_width = 128
target_width = 512
global_lr = 3e-4

hidden_weight = ParamSpec(
    name="hidden.weight",
    fan_in=target_width,
    base_fan_in=base_width,
)

lr_hidden = mup_lr(global_lr, hidden_weight)

assert hidden_weight.width_mult() == 4
assert abs(lr_hidden - 7.5e-5) < 1e-12

# 不随宽度变化的参数，宽度倍数为 1
layernorm_bias = ParamSpec(
    name="ln.bias",
    fan_in=1,
    base_fan_in=1,
)

lr_ln = mup_lr(global_lr, layernorm_bias)
assert layernorm_bias.width_mult() == 1
assert abs(lr_ln - global_lr) < 1e-12

print("hidden lr:", lr_hidden)
print("layernorm lr:", lr_ln)
```

这个例子故意简单，目的是说明：μP 的“学习率迁移”不是所有参数统一除一个数，而是按张量所属角色分别处理。

如果用官方 `mup` 库，流程更接近下面这样：

```python
import torch
import torch.nn as nn
from mup import MuReadout, set_base_shapes, MuAdamW
import mup

class ToyMLP(nn.Module):
    def __init__(self, width, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.readout = MuReadout(width, d_out)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.readout(x)

base_model = ToyMLP(width=128)
delta_model = ToyMLP(width=256)
model = ToyMLP(width=512)

set_base_shapes(model, base_model, delta=delta_model)

for p in model.parameters():
    if p.dim() > 1:
        mup.init.xavier_uniform_(p)

optimizer = MuAdamW(model.parameters(), lr=3e-4)
```

实际接线时要检查三件事：

| 步骤 | 说明 | 常见错误 |
|---|---|---|
| `set_base_shapes` | 建立 base/delta 对照 | 忘记在初始化前调用 |
| μP 初始化 | 用 `mup.init` 或兼容初始化 | 仍沿用默认初始化 |
| μP 优化器 | 用 `MuAdam`/`MuAdamW`/`MuSGD` | 只改全局 lr，不改 optimizer 逻辑 |

### 真实工程例子

一个常见训练流程是：先在 13M 级代理 Transformer 上搜索学习率、warmup 步数、schedule、初始化尺度，再把这组超参数迁移到 350M 甚至更大的模型。`Tensor Programs V` 和微软博客给出的核心结论是，在 μP 下，很多最优超参数随宽度变化更稳定，因此小模型不只是“便宜试验田”，而是对大模型更可信的代理。

---

## 工程权衡与常见坑

μP 的难点主要不在理论，而在接线一致性。你只要混入一段默认参数化逻辑，训练行为就可能局部失真。

| 坑 | 表现 | 规避方式 |
|---|---|---|
| 只改学习率 | 大模型仍不稳定 | `set_base_shapes`、初始化、μP optimizer 一起上 |
| `delta_model` 少写了宽度维度 | 某些层缩放失真 | 所有会变宽的维度都要进入对照 |
| 顺序错误 | 参数已被错误初始化 | 先设 shape，再初始化，再建 optimizer |
| 保存后直接加载继续训 | `infshape` 信息丢失 | 加载后重新 `set_base_shapes(..., rescale_params=False)` |
| Transformer 仍用默认 attention scaling | 宽度扩展后动态不对 | 按 μP 规则改到 $1/d$ 同阶 |
| 不做 Coord Check | 训练几千步后才发现错 | 前几步就看激活、输出、梯度曲线 |

几个实用判断标准：

1. “能跑起来”不等于“μP 接对了”。
2. 小模型 loss 下降不等于大模型也会稳定迁移。
3. 宽度翻倍后，如果最佳学习率曲线位置还在大幅漂移，优先怀疑参数化实现，而不是先怪数据或随机种子。
4. Coord Check 很重要。它本质上是在看不同宽度下，某层激活或输出曲线是否大致水平。如果越宽越爆，通常说明实现不对。

工程上还有一个现实权衡：μP 接入成本高于默认训练脚手架。你需要维护 base/delta 模型定义、替换 readout、替换 optimizer、注意 checkpoint 恢复逻辑。但如果你的目标是从 100M 扩到 1B，或者反复做多轮大规模训练，这个额外成本通常能被节省的调参预算覆盖。

---

## 替代方案与适用边界

不是所有团队都必须上 μP。关键看你的目标到底是“先把模型训起来”，还是“让小模型调参结论尽可能稳定迁移到大模型”。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| 普通参数化 | 小模型、一次性实验 | 简单直接 | 宽度变化时超参数常漂 |
| μP | 明确沿宽度扩展、需要代理模型 | 超参数迁移性更强 | 接入复杂，要求严格 |
| 经验缩放规则 | 某个固定架构族 | 上手快 | 缺少统一理论与可迁移保证 |

什么时候值得用 μP？

1. 你有一个小模型，想把它当成大模型调参代理。
2. 你的扩展方向主要是宽度，而不是彻底换结构。
3. 你愿意统一处理初始化、readout、optimizer 和检查流程。
4. 你的大模型训练成本高到不能随便做随机搜索。

什么时候不一定值得？

1. 你只训练一个中小模型，没有扩展计划。
2. 结构变化比宽度变化更大，比如同时换注意力形式、归一化、激活函数、深度和数据配方。
3. 团队工程链路不稳定，无法保证所有层都按 μP 规则实现。此时理论收益会被工程误差抵消。

可以把它理解成：普通参数化适合“先把实验做出来”，μP 适合“把扩展路径制度化”。

---

## 参考资料

1. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/)
2. [Microsoft `mup` Repository README](https://github.com/microsoft/mup)
3. [µTransfer: A technique for hyperparameter tuning of enormous neural networks](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)
4. [u-μP: The Unit-Scaled Maximal Update Parametrization](https://huggingface.co/papers/2407.17465)

## 核心结论

`Layer Scaling` 是一类稳定深层神经网络训练的方法。它的核心动作只有一句话：**按层深把残差更新先压小**。做法可以是给残差分支乘一个很小的系数，也可以是在初始化时按深度缩小某些权重的尺度。

在 Transformer 里，`残差分支`可以理解为“每一层对主干表示做的一次增量修改”。如果网络很深，比如 100 层、200 层，且每层一开始都把这次修改完整加回去，很多小更新会在前期快速累积，结果就是 loss 抖动、梯度范数剧烈震荡，严重时直接发散。`Layer Scaling` 的目的不是提升单层表达能力，而是让深层网络在训练早期更像“接近恒等映射”的系统，先稳住，再学习更强的更新。

一个最简形式是：

$$
x_{l+1} = x_l + \gamma_l F_l(\mathrm{LN}(x_l))
$$

这里的 $\gamma_l$ 是缩放系数。白话说，它控制“这一层先往主干里加多少”。

下面这个表先给出两个最常见实现：

| 方法 | 核心形式 | 直接作用点 | 典型用途 |
|---|---|---|---|
| `LayerScale` | `x_{l+1}=x_l+\mathrm{Diag}(\gamma_l)F_l(\mathrm{LN}(x_l))` | 前向中的残差输出 | 深层 ViT、CaiT |
| `DeepNorm` | 对残差路径和初始化同时按深度重标定 | 初始化和残差合成规则 | 超深 Transformer |

玩具例子可以这样理解：做一锅汤时，每一层都像“再加一勺盐”。10 勺问题不大，100 勺就很容易失控。`Layer Scaling` 的意思不是不加，而是前面先按“几滴”加，等系统稳定后，再由训练自己决定哪些层该放大。

---

## 问题定义与边界

`Layer Scaling` 解决的是**深层残差网络训练不稳定**的问题。这里的“不稳定”主要指：

- 训练前几百到几千步 loss 爆炸或大幅震荡
- 梯度范数频繁出现尖峰
- 模型对学习率异常敏感，稍微大一点就发散
- 层数增加后，明明结构没变，但训练明显变脆弱

它不主要解决下面这些问题：

- 数据质量差
- 标签噪声大
- 模型容量不足
- 纯粹的过拟合
- 推理速度慢

`LayerNorm` 是“把一层输入的尺度规范化”；白话说，它把一批特征拉回比较稳定的统计范围。`Layer Scaling` 不是它的替代品。前者主要管输入分布，后者主要管**残差更新到底加多大**。两者经常同时使用，但不是一回事。

可以先看适用边界：

| 现象 | 是否适合优先考虑 Layer Scaling | 说明 |
|---|---|---|
| 100+ 层 Transformer 前期发散 | 是 | 典型场景 |
| 梯度范数剧烈震荡 | 是 | 常与残差累积过强有关 |
| 小模型收敛慢 | 通常不是 | 先看学习率、数据、优化器 |
| 只有验证集过拟合 | 不是 | 应优先看正则化与数据 |
| 偶发单次数值异常 | 不一定 | 可能先检查混精、裁剪、batch |

边界要说清楚：`Layer Scaling` 更适合**深层 Transformer、深层 ViT、超深编码器/解码器**。如果模型只有十几层，或者问题主要来自错误的数据管线，那么它通常不是第一优先级。

真实工程里，一个很典型的信号是：你把 24 层模型扩到 96 层，参数量确实更大了，但训练并没有更好，反而一开始就抖。这时先问的问题不是“是不是模型太大”，而是“是不是深层残差更新叠得太快”。

---

## 核心机制与推导

先从普通残差块开始。设第 $l$ 层输入是 $x_l$，子层变换是 $F_l$，那么标准残差写法是：

$$
x_{l+1}=x_l+F_l(x_l)
$$

这句话的含义很直接：主干保留原表示，再叠加当前层算出的更新。

如果层数浅，这种设计很好，因为梯度传播路径短，更新叠加也有限。但当层数很深时，问题会变成：即使每层更新都“不算特别大”，很多层连续相加后，总体扰动也可能迅速放大。用很粗的近似看，如果每层残差更新的典型尺度是 $\sigma$，累积后的整体扰动规模会随着层数 $L$ 增长，常见的直觉近似是接近 $\sqrt{L}\sigma$ 或更糟。训练前期参数还没进入稳定区间时，这种累积特别危险。

### 为什么“先小后大”更稳

如果把残差分支改成：

$$
x_{l+1}=x_l+\gamma_lF_l(x_l), \quad 0<\gamma_l \ll 1
$$

那么在训练刚开始时，网络更接近：

$$
x_{l+1}\approx x_l
$$

也就是接近恒等映射。白话说，模型一开始不会急着大改主干表示，而是先学“怎样在几乎不破坏已有信号的前提下做微调”。这能显著降低前期更新累积过快的问题。

### LayerScale 的机制

`LayerScale` 的代表形式是：

$$
x_{l+1}=x_l+\mathrm{Diag}(\gamma_l)F_l(\mathrm{LN}(x_l))
$$

这里 $\gamma_l \in \mathbb{R}^d$ 通常是一个长度为 hidden size 的可学习向量，`Diag` 表示按通道逐元素缩放。也就是说，不同通道可以学不同的残差强度。初值通常非常小，比如 `1e-5` 或 `1e-6`。

它的含义是：

1. 子层先正常计算更新
2. 但更新不会直接加回主干
3. 而是先乘一个很小的、可学习的门
4. 训练稳定后，这个门可以自己长大

所以 `LayerScale` 更像**局部、通道级、前向期直接控制残差幅度**。

### DeepNorm 的机制

`DeepNorm` 的思路更系统。它不是只在前向里加一个可学习小门，而是把**残差合成规则**和**初始化尺度**一起按深度重标定。一个常见表达是：

$$
x_{l+1}=\mathrm{LN}(\alpha x_l + F_l(\mathrm{LN}(x_l); \theta_l))
$$

其中 $\alpha$ 控制主干项的权重，$\beta$ 常用于初始化时缩放某些参数。白话说，`DeepNorm` 不仅规定“前向怎么加”，还规定“某些层刚出生时应该多大”。

在 encoder-only 或 decoder-only 情况下，常见经验公式是：

$$
\alpha = (2N)^{1/4}, \qquad \beta=(8N)^{-1/4}
$$

其中 $N$ 是层数。

这个式子的重要信息不在于必须背下来，而在于方向：**层数越深，初始化就要越克制**。因为 $N$ 变大时，$\beta$ 会变小，意味着某些投影层权重的初始化幅度更小。

### 一个玩具例子

假设有 3 层网络，每层子层输出都暂时看成常数更新：

- 第 1 层输出 `2`
- 第 2 层输出 `2`
- 第 3 层输出 `2`

普通残差：

- 初始输入 `x_0 = 0`
- `x_1 = 0 + 2 = 2`
- `x_2 = 2 + 2 = 4`
- `x_3 = 4 + 2 = 6`

如果加上统一缩放 $\gamma = 0.1$：

- `x_1 = 0 + 0.1 * 2 = 0.2`
- `x_2 = 0.2 + 0.1 * 2 = 0.4`
- `x_3 = 0.4 + 0.1 * 2 = 0.6`

这个例子当然过于简化，但它清楚说明了一点：**当每层都向主干加更新时，缩放系数直接决定累计扰动的增长速度。**

### 一个数值推导例子

考虑 24 层 encoder-only 的 `DeepNorm`：

$$
\alpha = (2\times24)^{1/4}=48^{1/4}\approx 2.63
$$

$$
\beta = (8\times24)^{-1/4}=192^{-1/4}\approx 0.27
$$

如果某个需要重标定的线性层，在标准 Xavier 初始化后权重标准差约为 `0.02`，那么乘上 $\beta$ 后变成：

$$
0.02 \times 0.27 = 0.0054
$$

这意味着网络刚开始训练时，这部分残差相关变换的输出会更小，从而减轻早期层层叠加导致的不稳定。

---

## 代码实现

代码实现要分两类看：

1. 前向里直接缩放残差输出
2. 初始化时按深度缩放参数

先给一个可以直接运行的 `python` 玩具实现。它不依赖深度学习框架，只用标量模拟“残差累积”和“缩放后更稳”的效果。

```python
import math

def forward_without_scaling(x0, residuals):
    x = x0
    traj = [x]
    for r in residuals:
        x = x + r
        traj.append(x)
    return traj

def forward_with_scaling(x0, residuals, gamma):
    x = x0
    traj = [x]
    for r in residuals:
        x = x + gamma * r
        traj.append(x)
    return traj

# 玩具例子：5层，每层都试图加较大的更新
residuals = [2.0, 2.0, 2.0, 2.0, 2.0]

plain = forward_without_scaling(0.0, residuals)
scaled = forward_with_scaling(0.0, residuals, gamma=0.1)

assert plain[-1] == 10.0
assert abs(scaled[-1] - 1.0) < 1e-12
assert scaled[-1] < plain[-1]

def deepnorm_beta(num_layers):
    return (8 * num_layers) ** (-0.25)

beta_24 = deepnorm_beta(24)
assert 0.26 < beta_24 < 0.28

std_before = 0.02
std_after = std_before * beta_24
assert std_after < std_before

print("plain:", plain)
print("scaled:", scaled)
print("beta_24:", round(beta_24, 4))
print("scaled std:", round(std_after, 6))
```

这段代码表达的就是：同样的每层更新，如果先乘一个小系数，最终累计扰动会小很多；而 `DeepNorm` 里初始化缩放的作用，本质也是把早期更新压住。

### LayerScale 的典型实现

在 PyTorch 里，`LayerScale` 常长这样：

```python
import torch
import torch.nn as nn

class LayerScaleBlock(nn.Module):
    def __init__(self, dim, sublayer, init_value=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = sublayer
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        y = self.sublayer(self.norm(x))
        return x + self.gamma * y
```

关键点只有两个：

- `gamma` 是可学习参数
- 初值很小，但不是固定死的

这里 `逐通道缩放` 的意思是 hidden dimension 上每个通道都有自己的系数。白话说，模型可以自己学习“哪些特征通道的残差该先开大一点，哪些该更保守”。

### DeepNorm 的实现位置

`DeepNorm` 更常见的做法不是加一个 `nn.Parameter gamma`，而是：

- 调整残差合成公式中的系数
- 在初始化时对部分线性层乘上 $\beta$

实践里通常会保留 `Q/K` 为标准初始化，而对 `V/O` 投影及 FFN 相关线性层做额外缩放。原因是这些分支更直接参与残差输出的幅度控制。

可以把二者差异总结成表：

| 实现点 | `LayerScale` | `DeepNorm` |
|---|---|---|
| 作用位置 | 前向残差输出 | 参数初始化 + 残差缩放 |
| 是否可学习 | 是 | 通常不是，主要是固定公式 |
| 粒度 | 通道级 | 全层级 / 全网络级 |
| 常见场景 | CaiT、深层 ViT | 超深 Transformer |

### 真实工程例子

假设你在训练一个 128 层的翻译 Transformer，前 800 step 里看到：

- training loss 明显锯齿震荡
- grad norm 周期性冲高
- 学习率已经不大，但仍不稳定

这时工程上常见做法是：

1. 保留 `Pre-LN` 结构
2. 开启 `DeepNorm`
3. 对需要的线性层使用 $\beta$ 缩放初始化
4. 监控前 1000 step 的 `loss`、`grad norm`、`update norm`

如果你训练的是 100+ 层 ViT，则更常见的是直接给 attention block 和 MLP block 的残差分支加 `LayerScale(init_values=1e-5)`。因为 ViT 场景里，前向中加一个轻量可学习残差门，通常改动小、见效直接。

---

## 工程权衡与常见坑

`Layer Scaling` 不是“加了就一定更强”，它解决的是稳定性问题，因此天然存在一个权衡：**越保守越稳，但前期也可能越慢**。

比如：

- `gamma = 1e-6` 往往比 `1e-4` 更稳
- 但也更可能让前几千步学习很慢
- `beta` 缩得过小，也会让模型像“手脚被绑住”，稳定但迟迟学不动

下面这个表是最常见的坑：

| 坑 | 后果 | 处理方式 |
|---|---|---|
| 混淆 `LayerScale`、`DeepNorm`、`LayerNorm scaling` | 配置错位，排障困难 | 先明确方法定义和作用位置 |
| 只降学习率，不改残差尺度 | 仍可能前期发散 | 同时检查初始化与残差更新幅度 |
| `gamma` 或 `beta` 过小 | 模型很稳但收敛慢 | 从论文或主流实现经验值起步 |
| 加了缩放后仍沿用原 `drop path` | 正则强度失衡 | 重新搜索正则超参 |
| checkpoint 没保存缩放配置 | 权重加载不匹配 | 保存完整结构与初始化参数 |
| 把所有投影层都统一缩放 | 与论文实现不一致 | 明确哪些层需要缩放 |

有几个工程判断非常实用。

第一，优先看前 1000 step，而不是只看最终 loss。因为 `Layer Scaling` 的价值主要体现在训练早期的稳定区间。你需要盯的不是单点 loss，而是三条曲线是否同时变得“更平”：

- `loss`
- `grad norm`
- `update norm`

第二，不要把它当成所有不稳定问题的总开关。如果混合精度配置错了，或数据里有异常样本，或 batch size 太小导致统计噪声很大，`Layer Scaling` 也救不了结构之外的问题。

第三，深层模型里“稳”通常比“前 100 步下降快”更重要。很多新手会看到加了缩放后前期 loss 下降没那么猛，就误以为效果更差。实际上，只要它显著降低发散概率，并让长程训练可持续，往往就是正确方向。

---

## 替代方案与适用边界

`Layer Scaling` 很重要，但不是唯一方案。它适合的是“深层残差更新过强”这类问题。如果问题根源不在这里，就应该优先用别的方法。

先看对比：

| 方法 | 解决的问题 | 适用场景 | 局限 |
|---|---|---|---|
| `LayerScale` | 残差输出过强 | 深层 ViT、Transformer | 主要解决残差幅度 |
| `DeepNorm` | 超深网络训练稳定性 | 100+ 层 Transformer | 依赖结构和初始化规则 |
| 梯度裁剪 | 梯度爆炸 | 通用训练 | 常治标，不一定治本 |
| 降学习率 | 更新步长过大 | 通用训练 | 可能把学习也一起压慢 |
| `Pre-LN` | 优化路径更稳 | 许多 Transformer | 对极深层可能仍不够 |
| 更保守初始化 | 参数初值过大 | 通用训练 | 不直接控制前向残差叠加 |

### 什么时候优先用替代方案

如果只是训练初期偶尔有一两个 step 抖动，但整体可训练，先试这些更便宜的手段通常更合理：

- 学习率略降
- 开梯度裁剪
- 检查 warmup 是否太短
- 检查混精和 loss scale

如果模型层数不深，比如 12 层、24 层，且训练一直稳定，那专门引入 `Layer Scaling` 的收益可能有限。

### 什么时候更该考虑 Layer Scaling

以下场景里，它通常更值得优先考虑：

- 你正在从 24 层扩到 96 层、192 层
- 模型结构本身没错，但层数一深就训练脆弱
- 你已经做了 `Pre-LN`、warmup、梯度裁剪，仍不稳定
- 你需要长期训练而不是只跑短实验

再强调一次真实工程边界：如果主要瓶颈是数据、tokenization、优化器设置或 batch 规模，先把这些基础问题处理对，再谈 `Layer Scaling`。它不是跳过基本功的捷径。

---

## 参考资料

| 类型 | 来源 |
|---|---|
| 原论文 | *DeepNet: Scaling Transformers to 1,000 Layers* |
| 原论文 | *Going deeper with Image Transformers* |
| 工程实现 | TorchScale 官方仓库 |
| 代码参考 | timm Vision Transformer 源码 |

1. Wang et al. *DeepNet: Scaling Transformers to 1,000 Layers*. 重点看 `DeepNorm` 的残差重标定和初始化规则。  
2. Touvron et al. *Going deeper with Image Transformers*. 重点看 CaiT 中 `LayerScale` 的设计动机和初值设置。  
3. Microsoft `TorchScale` 仓库。重点看 `deepnorm=True` 时的工程落地方式。  
4. `timm` 中 Vision Transformer 相关实现。重点看深层 ViT 场景下残差缩放的实际写法。

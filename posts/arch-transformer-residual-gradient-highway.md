## 核心结论

残差连接（residual connection，白话说就是“把输入原样绕过去，再和新计算结果相加”）之所以重要，不是因为它让表达式更复杂，而是因为它在每一层都保留了一条恒等路径。标准写法是：

$$
x_{l+1} = x_l + F_l(x_l)
$$

这里 $F_l$ 表示第 $l$ 层真正学习的变换。对损失函数 $L$ 求导可得：

$$
\frac{\partial L}{\partial x_l}
=
\frac{\partial L}{\partial x_{l+1}}
\left(1 + \frac{\partial F_l}{\partial x_l}\right)
$$

关键不是 $\frac{\partial F_l}{\partial x_l}$，而是前面的那个 $1$。这个常数项就是“梯度高速公路”：即使变换分支的梯度很小，恒等分支仍然能把梯度直接传回去。

如果没有残差，深层网络的梯度大致是很多层导数的连乘：

$$
\prod_i g_i
$$

只要大多数 $g_i<1$，梯度就会指数衰减。加上残差后变成：

$$
\prod_i (1+g_i)
$$

这时每层都多了一个保底通路，训练深层 Transformer 才变得现实。对 96 层甚至更深的模型来说，这不是“优化小技巧”，而是是否能稳定训练的前提条件之一。

一个最小玩具例子可以直接看出差异。若两层普通链式网络每层导数都是 $0.5$，则总梯度是：

$$
0.5 \times 0.5 = 0.25
$$

若两层都是残差块，则每层梯度因子是 $1+0.5=1.5$，总梯度变成：

$$
(1+0.5)^2 = 2.25
$$

这说明残差并不是“让梯度变大”这么简单，而是把“容易断掉的纯乘法链”改成了“带保底项的乘法链”。

---

## 问题定义与边界

问题定义很明确：深层 Transformer 在反向传播时，底层参数经常收不到足够强的梯度信号。梯度（gradient，白话说就是“参数该往哪个方向改、改多少”）一旦随着层数指数衰减，底层层几乎不更新，模型就会表现为收敛慢、训练不稳，甚至直接发散。

普通链式深层网络可写成：

$$
x_{l+1}=F_l(x_l)
$$

其梯度传播是：

$$
\frac{\partial L}{\partial x_l}
=
\frac{\partial L}{\partial x_{l+1}}
\cdot
\frac{\partial F_l}{\partial x_l}
$$

连续多层后，梯度就变成多个 Jacobian 的连乘。Jacobian（雅可比矩阵，白话说就是“这一层输入变化会怎样影响输出变化”）若谱半径普遍小于 1，梯度就会很快缩小。

对零基础读者，可以先用一个简化图像理解：把深层网络看成很多串联的水管，每一段都有弯头和阻力。没有残差时，水必须穿过全部弯管才能回流；有残差时，每一段旁边都多了一根直通支路，水不用完全穿过弯管，也能继续往回传。这不是严格数学定义，但足够帮助理解“旁路”为什么能保底。

这篇文章的边界有三点：

1. 讨论对象主要是 Transformer 中的残差结构，而不是所有深度网络。
2. 重点分析 Pre-LN 与 Post-LN。LayerNorm（层归一化，白话说就是“把每个 token 的特征重新拉回稳定尺度”）的位置会影响梯度是否真的沿恒等路径稳定传播。
3. 讨论残差缩放 $x_{l+1}=x_l+\alpha F(x_l)$。缩放因子 $\alpha$ 不是可有可无，它决定了“梯度不消失”和“激活不爆炸”之间的平衡。

下面用一个表先把三类结构放在一起：

| 结构 | 前向形式 | 反向梯度主因子 | 定性效果 |
|---|---|---|---|
| 普通链式 | $x_{l+1}=F(x_l)$ | $g_l$ | 易随深度指数衰减 |
| 残差 | $x_{l+1}=x_l+F(x_l)$ | $1+g_l$ | 有恒等保底路径，梯度不断层 |
| 残差+缩放 | $x_{l+1}=x_l+\alpha F(x_l)$ | $1+\alpha g_l$ | 保留高速公路，同时抑制累积放大 |

这里的“不断层”不等于“绝对稳定”。如果 $g_l$ 很大且同号，$\prod (1+g_l)$ 也可能迅速膨胀，所以工程上常常需要 LayerNorm、初始化策略、残差缩放一起工作。

---

## 核心机制与推导

设第 $l$ 层的局部导数记为：

$$
g_l=\frac{\partial F_l}{\partial x_l}
$$

先看无残差的情况：

$$
x_{l+1}=F_l(x_l)
$$

则从第 $L$ 层回传到第 $0$ 层，有：

$$
\frac{\partial L}{\partial x_0}
=
\frac{\partial L}{\partial x_L}
\prod_{l=0}^{L-1} g_l
$$

如果每层平均 $g_l=0.5$，10 层之后：

$$
0.5^{10}\approx 9.8\times 10^{-4}
$$

这意味着底层几乎感觉不到损失函数的变化。

再看残差：

$$
x_{l+1}=x_l+F_l(x_l)
$$

对 $x_l$ 求导：

$$
\frac{\partial x_{l+1}}{\partial x_l}=1+g_l
$$

于是总梯度变为：

$$
\frac{\partial L}{\partial x_0}
=
\frac{\partial L}{\partial x_L}
\prod_{l=0}^{L-1}(1+g_l)
$$

如果仍然令每层 $g_l=0.5$，10 层之后：

$$
(1+0.5)^{10}=1.5^{10}\approx 57
$$

这说明残差把“每层都在削弱梯度”的系统，改成了“每层至少带着一个恒等通道”的系统。

如果再加入残差缩放：

$$
x_{l+1}=x_l+\alpha F_l(x_l)
$$

则有：

$$
\frac{\partial x_{l+1}}{\partial x_l}=1+\alpha g_l
$$

总梯度变成：

$$
\frac{\partial L}{\partial x_0}
=
\frac{\partial L}{\partial x_L}
\prod_{l=0}^{L-1}(1+\alpha g_l)
$$

还是令 $g_l=0.5$，取 $\alpha=0.5$，则 10 层之后：

$$
(1+0.5\times 0.5)^{10}=1.25^{10}\approx 9.3
$$

这个数值很重要。它说明缩放不是把残差“削弱到无效”，而是把可能过度膨胀的乘积拉回更稳的范围。

可以把三种情况直接排成一个玩具例子：

| 设定 | 每层因子 | 10 层总梯度近似值 | 解释 |
|---|---|---:|---|
| Plain | $0.5$ | $9.8\times10^{-4}$ | 快速衰减 |
| Residual | $1.5$ | $57$ | 高速公路存在，但可能放大 |
| Residual + $\alpha=0.5$ | $1.25$ | $9.3$ | 通路保留，放大受控 |

真实工程例子可以看深层 Transformer。比如 96 层 Pre-LN 结构中，每个 block 若写成：

$$
x_{l+1}=x_l+\alpha \cdot \text{SubLayer}(\text{LN}(x_l))
$$

则梯度可以优先沿着 $x_l \to x_{l+1}$ 的恒等路径回传，而变换支路只负责提供修正量。这样底层 embedding 附近的层不会因为中间经过几十层注意力和 MLP 后就完全收不到更新。大模型训练中，这条性质直接影响“能否把层数堆上去”。

如果用文字画一个极简图，梯度流大致是这样：

```text
无残差:
x_l -> F_l -> x_{l+1} -> F_{l+1} -> ... -> L

有残差:
x_l ------> (+) ------> x_{l+1} ------> (+) -----> ...
   \-> F_l -/               \-> F_{l+1} -/
```

上面那条直线就是高速公路。它不是替代学习分支，而是给学习分支一个稳定的“地基”。

---

## 代码实现

下面先给一个不依赖 PyTorch 的可运行玩具实现，用来验证三种梯度乘积的差别：

```python
def plain_grad(gs):
    out = 1.0
    for g in gs:
        out *= g
    return out

def residual_grad(gs, alpha=1.0):
    out = 1.0
    for g in gs:
        out *= (1.0 + alpha * g)
    return out

# 玩具例子：两层，每层局部导数都是 0.5
gs = [0.5, 0.5]
assert abs(plain_grad(gs) - 0.25) < 1e-12
assert abs(residual_grad(gs) - 2.25) < 1e-12

# 10 层例子
gs10 = [0.5] * 10
plain = plain_grad(gs10)
res = residual_grad(gs10)
scaled = residual_grad(gs10, alpha=0.5)

assert plain < 1e-3
assert 56 < res < 58
assert 9 < scaled < 10

print(plain, res, scaled)
```

这段代码虽然简单，但它精确对应前面的推导：普通链式是 $\prod g_i$，残差是 $\prod (1+\alpha g_i)$。

在真实模型里，代码通常更接近下面这种 Pre-LN 写法。Pre-LN（白话说就是“先归一化，再做子层计算，再加回原输入”）是目前更常见的稳定方案：

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, alpha=1.0, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, dropout=dropout)
        self.alpha = alpha

    def forward(self, x):
        h = self.norm(x)      # Pre-LN: 先稳定输入尺度
        h = self.ffn(h)       # 学习残差分支
        return x + self.alpha * h  # 恒等路径 + 缩放后的更新量

# 形状检查
block = ResidualBlock(dim=8, hidden_dim=16, alpha=0.5, dropout=0.0)
x = torch.randn(2, 4, 8)
y = block(x)
assert y.shape == x.shape
```

这个结构有几个实际超参数：

| 参数 | 作用 | 常见取值 |
|---|---|---|
| `alpha` | 残差缩放，控制每层增量强度 | `1.0`、`0.5`、按层缩放 |
| `dropout` | 随机丢弃部分激活，缓解过拟合 | `0.0` 到 `0.1` 常见 |
| `hidden_dim` | 前馈网络中间维度 | 通常是 `dim` 的 4 倍左右 |

再看 Post-LN。Post-LN（白话说就是“先做子层和残差相加，再做归一化”）通常写成：

```python
def forward(self, x):
    h = self.ffn(x)
    return self.norm(x + self.alpha * h)
```

它也有残差，但梯度在回传时会经过后置归一化的影响，极深时更容易出现优化困难。因此很多深层 Transformer 训练更偏向 Pre-LN，或者在 Post-LN 上额外引入更强的 highway 设计。

---

## 工程权衡与常见坑

残差连接解决的是“梯度断流”，不是“一切训练问题”。真正的工程难点在于：你既不能把恒等通路拿掉，也不能让每层的残差无节制叠加。

真实工程例子中，一个 96 层 Pre-LN Transformer 如果保留残差并给出合适缩放，训练曲线通常会更平滑，因为底层参数始终能收到有效梯度。而同样深度若直接去掉残差，常见现象是前几千步 loss 明显震荡，随后优化停滞，严重时出现 NaN。这里 NaN 的直接原因不一定总是“没残差”，但没残差会让深层优化窗口变得很窄，训练鲁棒性显著下降。

最常见的坑可以直接列出来：

| 坑 | 现象 | 原因 | 缓解策略 |
|---|---|---|---|
| 去掉 residual | 底层几乎不学习，loss 降不动 | 梯度变成纯连乘，指数衰减 | 保留恒等旁路 |
| residual 不缩放 | 深层 activation 逐层累加偏大 | 每层都在往主干上加增量 | 用 $\alpha<1$、更稳初始化 |
| 深层 Post-LN 直接硬堆层数 | 底层梯度弱、训练不稳 | 归一化位置改变了梯度路径性质 | 改用 Pre-LN 或 highway 变体 |
| 残差维度不匹配 | 运行时报 shape error | 主干和分支无法相加 | 用 projection 对齐维度 |
| 以为残差能替代初始化和归一化 | 训练仍发散 | 旁路只解决一部分问题 | 联合考虑 LN、初始化、学习率 |

还有两个容易被忽略的点。

第一，残差“保梯度”不代表输出尺度天然稳定。若每层都向主干加上一个量级不小的更新，激活值会逐层漂移，最终导致注意力 logits、MLP 输出或归一化统计失控。因此深层模型会在初始化、学习率 warmup、残差缩放上做一整套配合。

第二，Pre-LN 和 Post-LN 不是“谁永远更好”，而是优化路径不同。Pre-LN 的优点是梯度更容易沿恒等主干传到低层，代价是部分配置下表示尺度可能更依赖后续头部处理。Post-LN 在一些设置下最终性能可能不差，但对深度和训练技巧更敏感。

一个实用判断标准是：如果你已经把深度推到几十层以上，且训练不稳定，优先检查残差主干是否保持干净、LayerNorm 是否放在合适位置、残差分支是否过强，而不是先去改复杂优化器技巧。

---

## 替代方案与适用边界

标准残差不是唯一方案，但它是默认基线。只有当标准残差已经无法满足收敛性、稳定性或表示控制需求时，才值得引入更复杂的旁路设计。

常见变体之一是 gated residual，也就是门控残差。gated（门控，白话说就是“让模型自己决定这层残差该放多少”）可写成：

$$
x_{l+1}=x_l+\sigma(a_l)\odot F_l(x_l)
$$

其中 $\sigma(a_l)$ 是 $0$ 到 $1$ 之间的门值。它比固定 $\alpha$ 更灵活，但也引入更多参数和额外训练不稳定因素。

另一类是 Highway Network 风格：

$$
x_{l+1}=T(x_l)\odot H(x_l) + (1-T(x_l))\odot x_l
$$

这里 $T(x_l)$ 是变换门，决定走多少新分支、保留多少旧分支。它本质上也是在显式控制“高速公路”的宽度。

还有 Stochastic Depth 或 LayerDrop。这类方法不是替代残差本身，而是让某些层在训练时随机跳过。它们的作用更偏正则化与深层训练稳定，而不是单独承担梯度保底职责。

下面做一个简表：

| 方案 | 公式特征 | 优点 | 代价 | 适用边界 |
|---|---|---|---|---|
| 标准 residual | $x+\!F(x)$ | 简单、稳定、默认首选 | 深层可能需额外缩放 | 大多数 Transformer |
| residual + scaling | $x+\alpha F(x)$ | 更稳，适合很深网络 | 需调 $\alpha$ | 64 层以上常见 |
| gated residual | $x+\sigma(a)\odot F(x)$ | 自适应控制残差强度 | 参数更复杂 | 深层且分支强弱差异大 |
| highway | $T\odot H +(1-T)\odot x$ | 显式建模通路比例 | 结构更重 | 特殊稳定性需求场景 |
| stochastic depth | 随机跳过层 | 深层正则化有效 | 训练/推理行为不同 | 超深网络训练 |

它们也有适用边界。比如在 CNN 中，如果输入输出通道数不同，残差主干不能直接相加，就必须加 projection，也就是投影层。projection（投影，白话说就是“先用一个线性映射把形状改对，再去相加”）通常用 $1\times1$ 卷积或线性层完成。否则“梯度高速公路”在数学上根本建不起来。

因此，残差高速公路成立有一个隐藏前提：主干支路和残差支路要能在同一空间里相加。Transformer 因为大多数 block 保持同维度，所以天然适合这套机制；跨尺度 CNN 或编码器-解码器结构则常常需要额外设计。

---

## 参考资料

1. SudoAll, “LLM Residual Connections”. 文章重点在于用链式法则解释 residual 作为 gradient highway 的推导，给出 $x_{l+1}=x_l+F(x_l)$ 与 $\prod(1+g_i)$ 的直观分析。  
   链接：https://sudoall.com/llm-residual-connections/

2. Orchestra Research, “Residual Connections”. 文章贡献主要是直观解释深层网络中残差如何保留信息与梯度路径，适合作为机制理解的补充材料。  
   链接：https://www.orchestra-research.com/intro-to-ai-research/residual-connections

3. Emergent Mind, “Post-LayerNorm Is Back” 相关页面，论文编号 `arXiv:2601.19895`。该材料讨论 Pre-LN / Post-LN 的训练行为差异，以及深层 Transformer 中 highway 设计对优化的影响。  
   链接：https://www.emergentmind.com/papers/2601.19895

4. “Post-LayerNorm Is Back” 论文原始版本。适合追溯更正式的实验设置、深层网络稳定性结论以及相关 highway 设计背景。  
   可按题名或 `arXiv:2601.19895` 检索原文。

5. Michael Brenndoerfer 关于 residual scaling 与深层稳定性的讲解材料。价值主要在工程经验层面：为什么仅有 residual 还不够，为什么极深模型往往还需要缩放、归一化和初始化策略配合。  
   可按作者名与 “residual scaling transformer” 检索。

## 核心结论

`Pre-Norm` 和 `Post-Norm` 的区别，不是有没有 `LayerNorm`，而是 `LayerNorm` 放在残差分支前还是后。`LayerNorm` 可以理解为“按特征维度做标准化，让每个位置的数值尺度更可控”的操作。对深层 Transformer 来说，`Pre-Norm` 一般更容易训练稳定，已经成为现代大模型的标准选择；`Post-Norm` 是原始 Transformer 的典型写法，能用，但通常更依赖 warmup 和更谨慎的超参数。

两种写法的核心公式只有一行之差：

$$
\text{Post-Norm: } y = LN(x + F(x))
$$

$$
\text{Pre-Norm: } y = x + F(LN(x))
$$

这里的 `F` 表示子层变换，通常是 self-attention 或前馈网络 `FFN`。`残差` 可以理解为“把原输入直接加回输出，保留一条捷径路径”。

新手可以先记一句话：

- `Post-Norm`：先更新，再归一化。
- `Pre-Norm`：先归一化，再更新。

它们的工程表现差异，主要来自梯度怎么穿过这条结构。`梯度` 就是反向传播时参数更新信号的大小和方向。`Pre-Norm` 保留了更直接的残差主干，反向时更容易把信号传回前层；`Post-Norm` 则把归一化放在残差相加之后，反向必须经过 `LN`，深层时更容易对训练早期造成不稳定。

| 对比项 | Pre-Norm | Post-Norm |
| --- | --- | --- |
| 公式 | `x + F(LN(x))` | `LN(x + F(x))` |
| LN 位置 | 子层前 | 残差相加后 |
| 残差主干 | 更直通 | 更受 LN 影响 |
| 深层训练稳定性 | 通常更强 | 通常更依赖 warmup |
| 现代 LLM 使用情况 | 主流 | 较少作为默认方案 |

如果只记一个结论：当模型层数变深、训练预算有限、希望尽快把训练跑稳时，优先选 `Pre-Norm`。如果你在严格复现早期 Transformer 论文或历史实现，才更可能保留 `Post-Norm`。

---

## 问题定义与边界

这篇文章只讨论 Transformer 层内部的归一化位置，不讨论卷积网络、RNN 或一般神经网络里 `LayerNorm` 的所有用法。关注点很窄：一层 Transformer 中，`子层 + 残差 + 归一化` 这三者到底怎么排顺序。

先统一符号：

| 符号 | 含义 |
| --- | --- |
| $x_l$ | 第 $l$ 层输入 |
| $F_l$ | 第 $l$ 层子层变换，可能是 attention 或 FFN |
| $LN$ | LayerNorm，对特征维度做标准化 |
| $x_{l+1}$ | 第 $l$ 层输出 |

于是两种结构可以写成：

$$
x_{l+1} = LN(x_l + F_l(x_l))
$$

$$
x_{l+1} = x_l + F_l(LN(x_l))
$$

这不是语法差异，而是优化行为差异。

边界要说清楚：

- 文章讨论的是 encoder-only 或 decoder-only Transformer 中常见的层内 norm 位置。
- 文章不展开 `RMSNorm`、`ScaleNorm`、`DeepNorm` 等替代归一化方案，后文只做边界比较。
- `norm_first=True` 只表示层内先做 norm，不等于整个模型不需要最终输出 norm。
- `Pre-Norm` 不是“免调参许可证”。学习率、warmup、初始化、梯度裁剪仍然重要。

一个新手版理解是：一层 Transformer 可以看成一条流水线，输入先进入某个计算模块，比如注意力；然后把输出和原输入相加；最后可能做一次归一化。争议点不是模块本身，而是 `LN` 放在“算之前”还是“加之后”。

为什么这个问题重要？因为 Transformer 往往要堆很多层。层数一深，单层里看起来很小的结构差异，会在几十层传播后变成明显的优化差异。对浅层模型，两种结构都可能工作；对深层模型，这个差异通常会放大成“能不能稳住训练”。

---

## 核心机制与推导

先看前向。

`Post-Norm` 是：

$$
x_{l+1} = LN(x_l + F_l(x_l))
$$

含义是先把子层输出加到残差上，再对结果整体做一次标准化。这样每层输出的尺度通常更规整，数值分布看起来更“整齐”。

`Pre-Norm` 是：

$$
x_{l+1} = x_l + F_l(LN(x_l))
$$

含义是先把输入喂给 `LN`，得到尺度更稳定的子层输入，再让子层去计算，最后加回原始残差。这样残差主干本身没有在层末尾被 `LN` 再处理一遍。

这会直接影响反向传播。

把一层当成函数，`J` 表示雅可比矩阵，可以理解为“局部梯度变换规则”。那么：

- `Post-Norm` 的局部梯度近似包含 $J_{LN}\cdot(I + J_F)$
- `Pre-Norm` 的局部梯度近似包含 $I + J_F \cdot J_{LN}$

这两个式子最关键的差别，不是乘法顺序本身，而是 `Pre-Norm` 明确保留了一条更直接的 $I$ 路径，也就是恒等映射路径。`恒等映射` 可以理解为“输入几乎原样通过”。深层网络里，这条路径很重要，因为它给梯度提供了一条不容易被压扁或放大的通道。

玩具例子可以看出两者输出形态差异。

设：

- $x = [1, 3]$
- $F(z) = z$
- 忽略 `eps`

先算 `Post-Norm`：

$$
x + F(x) = [2, 6]
$$

其均值为 $4$，标准差为 $2$，所以：

$$
LN([2, 6]) = \left[\frac{2-4}{2}, \frac{6-4}{2}\right] = [-1, 1]
$$

再算 `Pre-Norm`：

$$
LN([1, 3]) = [-1, 1]
$$

于是：

$$
x + F(LN(x)) = [1, 3] + [-1, 1] = [0, 4]
$$

这个例子不是为了说明谁更“正确”，而是展示数值形态：`Post-Norm` 会把层输出重新拉回标准化空间；`Pre-Norm` 则保留了一部分原始残差幅度。前者更整齐，后者更保留主干。

再把机制压缩成表格：

| 维度 | Pre-Norm | Post-Norm |
| --- | --- | --- |
| 前向路径 | 先标准化输入，再做子层更新 | 先更新并残差相加，再统一标准化 |
| 反向路径 | 更容易保留恒等残差通路 | 梯度必须穿过层末 LN |
| 残差是否直通 | 更接近直通 | 更受归一化扰动 |
| 训练早期风险 | 相对较低 | 更容易 loss spike |
| 深层堆叠表现 | 通常更稳 | 更敏感 |

为什么 `Post-Norm` 更容易在训练早期出问题？因为每层输出都经过一次层末 `LN`，反向传播也每层都要穿过它。`LN` 虽然不是坏操作，但它会重新缩放和中心化激活。层数一多，梯度会不断受到这些局部缩放规则影响。再叠加随机初始化和较大学习率，就更容易出现前几百步不稳定、梯度异常甚至 `NaN`。

真实工程例子更直观。假设你从零训练一个 24 层 decoder-only 语言模型。若采用 `Pre-Norm`，配合正常初始化、余弦退火和较短 warmup，训练通常能比较平稳进入下降阶段。若换成 `Post-Norm`，即便代码逻辑没错，也常见这些现象：

- 前几百步 loss 上下大幅抖动
- 梯度范数突然暴涨
- 个别 batch 直接出现 `NaN`
- 需要把学习率再降，或把 warmup 拉长

所以“Pre-Norm 更稳”不是口号，而是由残差主干和梯度路径共同决定的结果。

---

## 代码实现

在 PyTorch 里，这个差异已经被直接暴露为参数：

```python
import torch.nn as nn

layer = nn.TransformerEncoderLayer(
    d_model=768,
    nhead=12,
    dim_feedforward=3072,
    norm_first=True,
    batch_first=True,
)
```

这里的 `norm_first=True` 就是 `Pre-Norm`；`False` 对应 `Post-Norm`。

最容易记的伪代码如下：

```python
# Post-Norm
x = ln(x + attn(x))
x = ln(x + ffn(x))

# Pre-Norm
x = x + attn(ln(x))
x = x + ffn(ln(x))
```

注意这里有两个子层：attention 和 FFN。工程里最常见的错误之一，就是只改了 attention 的 norm 位置，FFN 还保留另一种写法。这样会让层内结构前后不一致，训练行为也会变得难分析。

下面给一个可运行的玩具实现，只保留结构，不实现真正多头注意力。代码目标是验证两种 block 的公式差异，而不是追求性能。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class ToySubLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model))

    def forward(self, x):
        return self.proj(x)

class PreNormBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = ToySubLayer(d_model)
        self.ffn = ToySubLayer(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class PostNormBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = ToySubLayer(d_model)
        self.ffn = ToySubLayer(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x

x = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]])

pre_block = PreNormBlock(d_model=2)
post_block = PostNormBlock(d_model=2)

pre_out = pre_block(x)
post_out = post_block(x)

assert pre_out.shape == x.shape
assert post_out.shape == x.shape
assert not torch.allclose(pre_out, post_out)
assert torch.isfinite(pre_out).all()
assert torch.isfinite(post_out).all()

print("pre_out =", pre_out)
print("post_out =", post_out)
```

这段代码有几个要点：

- `ToySubLayer` 用单位矩阵近似 `F(z)=z)`，方便观察结构差异。
- `PreNormBlock` 的残差主干没有在层末被再标准化。
- `PostNormBlock` 的每个子层输出都要经过一次层末 `LN`。

工程使用时可以这样理解：

| `norm_first` | 结构 | 常见表现 | 适用场景 |
| --- | --- | --- | --- |
| `True` | Pre-Norm | 更稳，更适合深层 | 从零训练中大型 Transformer |
| `False` | Post-Norm | 对超参数更敏感 | 复现原始论文或旧实现 |

如果训练脚本也要给最小建议，可以遵循这个方向：

- `Pre-Norm`：常规学习率、较短 warmup、标准初始化
- `Post-Norm`：更小学习率、更长 warmup、更保守初始化

---

## 工程权衡与常见坑

`Pre-Norm` 的优势是稳定性，但代价不是零。它并不自动保证“更快收敛”或“最终效果一定更好”。它解决的是优化入口问题，不是所有训练问题。

先看常见坑：

| 误区 | 后果 | 规避方法 |
| --- | --- | --- |
| 只改 attention，不改 FFN | 层内结构不一致，训练现象难解释 | attention 和 FFN 保持同一种 norm 位置 |
| 把 `norm_first=True` 当万能稳定器 | 仍可能爆梯度或 NaN | 继续检查学习率、初始化、梯度裁剪 |
| 混淆层内 norm 和最终输出 norm | 误判模型结构 | 区分 block 内 LN 与 stack 末尾 final norm |
| 深层 `Post-Norm` 直接沿用激进超参 | loss spike 或不收敛 | 减小 lr，拉长 warmup |
| 看到浅层模型能训稳就外推到深层 | 上层实验结论失效 | 分层数重新评估结构选择 |

真实工程里，最典型的失败现象通常不是“完全跑不动”，而是这些更隐蔽的信号：

- loss 在训练初期突然尖峰上冲
- 梯度范数偶发暴涨
- 训练一段时间后出现 `NaN`
- 收敛速度异常慢
- 对 batch size 或学习率非常敏感

对于零基础到初级工程师，一个实用判断标准是：如果你调的是一个十几层以上的 Transformer，且目标是尽快跑出稳定 baseline，就不要把 `Post-Norm` 当默认起点。`Post-Norm` 不是错误设计，但它对训练配置更挑剔。

再强调一次权衡：

- `Pre-Norm` 的优点是更稳、更省调参成本。
- `Post-Norm` 的优点是结构上更接近原始 Transformer 论文，某些旧实验复现必须保留。
- 当模型浅、任务简单、训练预算足够时，两者差距可能没那么大。
- 当模型深、训练资源贵、排障成本高时，稳定性优先级会明显高于“结构传统性”。

---

## 替代方案与适用边界

`Pre-Norm` 不是唯一解。它解决的是“深层残差网络如何让梯度更稳地流动”这个问题，而这个问题还有其他工程答案，比如 `RMSNorm`、`DeepNorm`、Residual Scaling。`Residual Scaling` 可以理解为“对残差分支再乘一个缩放系数，降低深层堆叠时的震荡风险”。

下面给一个压缩对比：

| 方案 | 稳定性 | 实现成本 | 收敛表现 | 复现难度 |
| --- | --- | --- | --- | --- |
| Pre-Norm | 高 | 低 | 通常稳 | 低 |
| Post-Norm | 中到低 | 低 | 对超参敏感 | 低 |
| RMSNorm | 高 | 低到中 | 现代 LLM 常用 | 中 |
| DeepNorm | 很高 | 中 | 面向更深层网络 | 中到高 |

适用边界可以直接记成三条：

- 如果目标是从零训练一个较深的模型，并且先要稳定，再谈细调，优先选 `Pre-Norm`。
- 如果目标是复现某篇早期论文、复现旧 checkpoint 对应结构，必须按原设计保留 `Post-Norm`。
- 如果模型特别深，或者你已经遇到明显的训练不稳，可以进一步考虑 `RMSNorm`、DeepNorm 或残差缩放。

这里还要补一个边界判断。很多人把“现代 LLM 普遍用 Pre-Norm”理解成“Post-Norm 已经过时且毫无价值”。这个理解不准确。正确说法是：在深层大模型训练这个主流场景下，`Pre-Norm` 的工程收益更明显，因此成为默认选择；但在论文复现、教学示例、浅层网络和某些历史代码库中，`Post-Norm` 仍然是合理结构。

所以不要问“谁绝对更先进”，而要问“我的目标是什么”：

- 要训稳深层模型：先选 `Pre-Norm`
- 要复现经典 Transformer：保留 `Post-Norm`
- 要追求现代大模型统一实现：通常会进一步转向 `Pre-Norm + RMSNorm` 这一类方案

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)  
  原始 Transformer 论文，适合理解残差、attention 和归一化在经典结构中的放置方式。

- [On Layer Normalization in the Transformer Architecture](https://www.microsoft.com/en-us/research/publication/on-layer-normalization-in-the-transformer-architecture/)  
  直接讨论 Transformer 中 LayerNorm 位置问题，适合在理解基础结构后阅读。

- [PyTorch `TransformerEncoderLayer` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.TransformerEncoderLayer.html)  
  工程实现入口，查看 `norm_first=True/False` 在框架中的实际语义。

- [PyTorch `torch/nn/modules/transformer.py` 源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py)  
  适合对照源码确认 Pre-Norm 和 Post-Norm 在标准实现中的组织方式。

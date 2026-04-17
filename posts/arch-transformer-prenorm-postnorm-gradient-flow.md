## 核心结论

Pre-Norm 和 Post-Norm 的差别，不在“是否用了归一化”，而在“归一化放在残差连接的哪一侧”。残差连接可以理解为“把旧表示直接传下去的捷径”；LayerNorm 可以理解为“把一层输入重新拉回到稳定尺度的操作”。这两个部件顺序一换，反向传播时的梯度路径就变了。

Pre-Norm 的形式是
$$
x_{l+1}=x_l+F(\mathrm{LN}(x_l))
$$
它的关键性质是：对 $x_l$ 求导时总会保留一个恒等映射 $I$。这意味着即使子层 $F$ 学得很差、很小，梯度仍然能沿着残差支路直接回到早期层。对训练来说，结论很直接：更稳定、更不依赖 warmup、深层更容易训起来。

Post-Norm 的形式是
$$
x_{l+1}=\mathrm{LN}(x_l+F(x_l))
$$
梯度必须先穿过每一层的 LayerNorm 雅可比矩阵。雅可比可以理解为“这一层对微小扰动的放大或缩小规则”。如果每层都带来一点收缩，那么多层连乘后，浅层参数收到的梯度就会快速变小。结论是：Post-Norm 在深层下更容易训练不稳，通常需要 warmup、更保守的学习率，或者额外的初始化/缩放技巧。

工程上可以先记一个简单判断：

| 结构 | 梯度直通性 | 训练稳定性 | 默认是否依赖 warmup | 常见用途 |
|---|---|---:|---:|---|
| Pre-Norm | 强 | 高 | 低 | 现代大语言模型主流 |
| Post-Norm | 弱 | 中到低 | 高 | 早期 Transformer、追求更强层间变换时 |
| Pre-RMSNorm | 强 | 高 | 低 | LLaMA、Mistral 一类实现 |

如果目标是“先把深层 Transformer 稳定训起来”，默认选 Pre-Norm；如果目标是“在已有稳定训练技巧下继续压榨表示能力”，Post-Norm 才值得认真考虑。

---

## 问题定义与边界

本文只讨论 Transformer block 内“归一化与残差的相对顺序”这一件事，不讨论注意力公式本身、激活函数差异、并行策略或数据规模变化。也就是说，我们把一个 block 抽象成“输入 $x_l$，经过某个子层 $F$，再加上残差”的过程。

两种定义如下：

- Pre-Norm：先归一化，再进入子层，最后和原输入相加。
- Post-Norm：先进入子层，与原输入相加，再做归一化。

写成式子就是：

$$
\text{Pre-Norm: } x_{l+1}=x_l+F(\mathrm{LN}(x_l))
$$

$$
\text{Post-Norm: } x_{l+1}=\mathrm{LN}(x_l+F(x_l))
$$

边界有两个。

第一，本文讨论的是“深度训练稳定性”，尤其是 12 层以上的网络。浅层模型里，Post-Norm 不一定明显出问题，因为链条还不够长，梯度衰减累计得没那么严重。

第二，本文说“Pre-Norm 更稳定”，不等于“Pre-Norm 永远更好”。稳定说的是优化过程更容易收敛，不是最终上限一定更高。很多实验都观察到：如果 Post-Norm 被 warmup、初始化缩放、结构改造成功扶稳，它在部分任务上可能比 Pre-Norm 高出约 0.5 到 1 个 BLEU，原因通常是深层表示被利用得更充分，而不是模型突然“更聪明”。

先看一个玩具例子。假设有 3 层网络，每层的 LayerNorm 对梯度都平均缩小到原来的 $0.7$。那么：

- Pre-Norm：梯度至少还保留一条“乘 1”的直通路。
- Post-Norm：梯度粗略变成 $0.7^3=0.343$。

如果是 12 层，就是
$$
0.7^{12}\approx 0.0138
$$
也就是只剩下原来的 1.38%。这不是精确推导，但足够说明“层数一深，Post-Norm 为什么难训”。

---

## 核心机制与推导

先看 Pre-Norm。定义
$$
x_{l+1}=x_l+F(\mathrm{LN}(x_l))
$$
对 $x_l$ 求导，链式法则给出
$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+\frac{\partial F(\mathrm{LN}(x_l))}{\partial x_l}
=I+B_l
$$
其中 $I$ 是恒等矩阵，$B_l$ 是子层和 LayerNorm 合起来的导数项。

这条式子的含义非常重要：不管 $B_l$ 怎么变，梯度传播里永远带着一个“直接通过”的分量。于是从第 $L$ 层回传到第 $1$ 层时，不会纯粹依赖所有归一化和子层导数的乘积。

再看 Post-Norm。定义
$$
x_{l+1}=\mathrm{LN}(x_l+F(x_l))
$$
令 $u_l=x_l+F(x_l)$，则
$$
x_{l+1}=\mathrm{LN}(u_l)
$$
于是
$$
\frac{\partial x_{l+1}}{\partial x_l}
=
\underbrace{\frac{\partial \mathrm{LN}(u_l)}{\partial u_l}}_{G_l}
\cdot
\underbrace{\left(I+\frac{\partial F(x_l)}{\partial x_l}\right)}_{I+F_l'}
$$
也就是
$$
\frac{\partial x_{l+1}}{\partial x_l}=G_l(I+F_l')
$$
跨越多层后，整体梯度变成
$$
\prod_{\ell=1}^{L} G_\ell (I+F_\ell')
$$

这里的关键不在 $F_\ell'$，而在 $G_\ell$。$G_\ell$ 来自 LayerNorm 的雅可比。LayerNorm 会按标准差 $\sigma$ 做缩放，所以它天然带着类似 $1/\sigma$ 的因子。若 $\sigma>1$，每一层都会有一点收缩，多层连乘后就可能指数下降。

可以把两者的梯度图画成下面这样：

| 结构 | 单层反向主路径 | 多层效果 |
|---|---|---|
| Pre-Norm | $I+B_\ell$ | 至少有恒等项直通 |
| Post-Norm | $G_\ell(I+F_\ell')$ | 每层都要乘一次 $G_\ell$ |

再给一个更具体的玩具推导。若每层平均有 $\|G_\ell\|\approx 1/1.4\approx 0.714$，12 层后：
$$
(1/1.4)^{12}\approx 0.016
$$
这意味着浅层只接收到约 1.6% 的梯度量级。相反，Pre-Norm 若每层的附加项大约是 $\|B_\ell\|\le 0.1$，那单层导数更像 $I+0.1$ 附近的扰动，而不是连续缩小的连乘。

真实工程例子更直观。以深层机器翻译 Transformer 为例，在 18+18 层编码器/解码器设置下，Pre-Norm 往往可以在标准初始化、较少甚至无 warmup 的条件下收敛；同配置的 Post-Norm 经常直接训练失败，或者前期 loss 大幅震荡。原因不是注意力算错了，而是浅层梯度在训练早期“几乎听不见”。

---

## 代码实现

实现层面，两者只差几行，但训练行为会明显不同。下面给出一个可运行的最小 Python 例子，用线性层代替注意力和前馈层，验证两种结构的前向定义与梯度可计算性。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class ToySubLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(x)

class PreNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = ToySubLayer(d)
        self.ffn = ToySubLayer(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class PostNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = ToySubLayer(d)
        self.ffn = ToySubLayer(d)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x

def grad_norm(model, x):
    x = x.clone().detach().requires_grad_(True)
    y = model(x).sum()
    y.backward()
    return x.grad.norm().item()

d = 8
x = torch.randn(2, 4, d)

pre = PreNormBlock(d)
post = PostNormBlock(d)

pre_g = grad_norm(pre, x)
post_g = grad_norm(post, x)

assert pre(x).shape == x.shape
assert post(x).shape == x.shape
assert pre_g > 0
assert post_g > 0

print("pre grad norm:", round(pre_g, 6))
print("post grad norm:", round(post_g, 6))
```

如果把它扩展成完整 Transformer block，结构可以写成：

```python
def pre_norm_block(x, attn, ffn, ln1, ln2):
    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))
    return x

def post_norm_block(x, attn, ffn, ln1, ln2):
    x = ln1(x + attn(x))
    x = ln2(x + ffn(x))
    return x
```

训练脚本上的差异通常比 block 本身更关键：

| 项目 | Pre-Norm 常见做法 | Post-Norm 常见做法 |
|---|---|---|
| 学习率起点 | 可以较大 | 通常更保守 |
| Warmup | 可短，可无 | 常常必需 |
| 深层扩展 | 更自然 | 常需额外缩放初始化 |
| 调参难度 | 低 | 高 |

真实工程里，LLaMA、Mistral 这类模型并没有继续用标准 LayerNorm，而是采用 Pre-RMSNorm。RMSNorm 可以理解为“只按均方根缩放，不再显式减均值”的轻量归一化版本。它保留了 Pre-Norm 的稳定训练特征，同时在实现和数值上更适合大模型。

---

## 工程权衡与常见坑

Pre-Norm 最大优点是“容易训练”，但它不是没有代价。最常见的问题叫表达坍缩。这个词的白话意思是：层数越来越深后，后面很多层虽然还在算，但对最终表示的新增贡献越来越小，看起来像接近恒等映射。

为什么会这样？因为每层都走
$$
x_{l+1}=x_l+\Delta x_l
$$
如果训练过程为了稳定性，逐渐把 $\Delta x_l$ 压得很小，那么深层 block 更像“微调输入”而不是“重构表示”。这会让模型很稳，但也可能限制深层表达力。

Post-Norm 的问题则更直接：不稳、难调、对 warmup 敏感。常见坑有三类。

第一，学习率一上来就过大，训练前几百步直接震荡甚至发散。  
第二，模型超过 12 层后，第一层到第三层几乎学不到东西。  
第三，看起来 loss 在降，但收敛速度慢，浅层参数更新极小。

一个常见误区是“Post-Norm 表现差，所以没有价值”。这不准确。更准确的说法是：Post-Norm 对优化条件要求高，但一旦通过 DeepNorm、Sandwich-LN、专门初始化等方法扶稳，它可能在表示多样性或某些翻译指标上拿到更高上限。

下面是工程取舍表：

| 方案 | 收敛稳定性 | 表达上限潜力 | 调参成本 | 常见坑 |
|---|---:|---:|---:|---|
| Pre-Norm | 高 | 中到高 | 低 | 深层增量变小，表示坍缩 |
| Post-Norm | 低到中 | 高 | 高 | 深层梯度衰减，严重依赖 warmup |
| Pre-Norm + RMSNorm | 高 | 中到高 | 低 | 仍需关注深层贡献衰减 |
| Post-Norm + DeepNorm | 中到高 | 高 | 中到高 | 初始化和缩放系数要匹配 |

如果你是零基础到初级工程师，最实用的经验是：

1. 自己从零写 Transformer，默认先用 Pre-Norm。
2. 发现模型很深但后几层“像没学到东西”，再考虑额外缩放或结构增强。
3. 不要在没有 warmup 和初始化设计的前提下直接上深层 Post-Norm。

---

## 替代方案与适用边界

现实里并不是只有“纯 Pre-Norm”和“纯 Post-Norm”两个按钮。很多现代工作其实是在做折中。

NormFormer 的思路是在 Pre-Norm 基础上继续修正梯度幅度不平衡，比如加额外归一化、head-wise scaling。它适合已经接受 Pre-Norm 主体、但又想提升大模型训练质量的场景。

DeepNorm 的思路是通过残差缩放系数 $\alpha,\beta$ 重新设计深层网络的数值范围。它不是简单换顺序，而是用初始化和缩放把“很深的 Post-Norm 风格网络”重新拉回可训练区间。适合上百层甚至更深的网络。

ResiDual 一类方法则尝试同时保留 Pre 和 Post 的优点，本质是在结构上增加双重信息通道：既保留梯度直达路径，又不完全放弃后归一化带来的表示约束。

可以用下面这张表快速判断：

| 替代方案 | 核心目标 | 适用场景 |
|---|---|---|
| Pre-RMSNorm | 保持稳定训练，降低实现负担 | 现代 LLM 默认选型 |
| NormFormer | 修正 Pre-Norm 梯度幅度分布 | 大模型继续提效 |
| DeepNorm | 让超深网络重新稳定 | 追求超深 Transformer |
| ResiDual | 折中梯度稳定与表示能力 | 研究型或高性能定制模型 |

最后给一个决策规则。

如果你在做通用语言模型、指令模型、代码模型，首选 Pre-Norm 或 Pre-RMSNorm。  
如果你在做超深翻译模型，且团队能承担较高调参成本，可以尝试 DeepNorm 或 Sandwich-LN 支持下的 Post-Norm。  
如果你只是想把一个教学版 Transformer 先跑通，不要一开始就挑战 Post-Norm 深层稳定性问题，因为那不是“模型理解”的主难点，而是“优化工程”的主难点。

---

## 参考资料

- Pre-LayerNorm Transformers 综述：https://www.emergentmind.com/topics/pre-layernorm-transformers
- Post-LayerNorm Transformers 分析：https://www.emergentmind.com/topics/post-layernorm-post-ln
- Pre-Norm 残差连接与翻译实验对比：https://www.emergentmind.com/topics/pre-norm-residual-connections-prenorm
- DeepNorm 相关资料：https://www.emergentmind.com/topics/deepnorm
- NormFormer 与相关扩展：https://www.emergentmind.com/topics/normformer
- LLaMA / Mistral / Gemma / Qwen 架构说明：https://nonlinear.technology/blog/llama-mistral-gemma-qwen-architecture-how-they-work/

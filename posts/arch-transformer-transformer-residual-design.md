## 核心结论

Transformer 的残差连接写成
$$x_{l+1}=x_l+F_l(x_l)$$
其中残差连接就是“把旧表示直接绕过去的加法旁路”，白话说是给每一层保留一条不必经过复杂变换的直通车。它的核心价值不是“让网络更深”这么简单，而是让梯度在反向传播时始终保留一条恒等路径：
$$\frac{\partial \mathcal L}{\partial x_l}=\frac{\partial \mathcal L}{\partial x_{l+1}}\left(I+\frac{\partial F_l}{\partial x_l}\right)$$
这里的 $I$ 是单位映射，白话说是“哪怕子层学得一般，梯度也不会完全断掉”。

对零基础读者，最实用的理解是一个玩具例子：把残差流看成高速公路，$F_l(x_l)$ 只是每一层在匝道口加的一点修正。高速公路本身负责把主干语义一路送到高层，匝道负责局部修补。因此深层模型能训练，首先依赖的不是每层都很强，而是主干信息不被层层改坏。

最近针对 Llama 3.1、Qwen 3 的分析还给出一个重要现象：模型后半段层对残差流的改变量明显变小，说明早层往往已经完成了大部分主干表示，后半层更多是在精修分布而不是重写语义。这进一步说明残差流不是“辅助结构”，而是 Transformer 的主干。

---

## 问题定义与边界

问题不是“残差连接有没有用”，而是“当 Transformer 深到 100 层、200 层甚至 1000 层时，残差流如何仍然稳定”。这里的稳定有两层含义：

| 目标 | 含义 | 失败现象 |
| --- | --- | --- |
| 前向稳定 | 表示幅度不随层数无控制地增大 | 激活爆炸、LayerNorm 前数值失控 |
| 反向稳定 | 梯度能从顶层回到底层 | 梯度消失、深层几乎学不到 |

没有残差时，深层网络等于不断复合非线性变换，梯度容易在连乘中衰减。只有残差但不控制尺度也不够，因为你在做的是
$$x_{l+1}=x_l+F_l(x_l)$$
如果 200 层都往同一个方向“多写一点”，总量仍然会失控。

新手可以用一个简单边界来理解：如果前 100 层已经写入主要信息，后 100 层就不该继续用同样强度重写主干；否则模型相当于反复覆盖旧表示，收益递减，甚至破坏已形成的结构。

这也是为什么“深层可训练”不等于“随便堆层”。残差连接提供了信息旁路，但还需要缩放、归一化和训练技巧来约束每层写入量。

---

## 核心机制与推导

先看梯度。把一层展开：
$$x_{l+1}=x_l+F_l(x_l)$$
则反向有
$$g_l=g_{l+1}\left(I+J_l\right),\quad J_l=\frac{\partial F_l}{\partial x_l}$$
即使 $J_l$ 很小，$I$ 仍然存在，所以不会像纯堆叠网络那样只剩连乘的微小导数。这就是“恒等梯度路径”。

但恒等路径只解决“能传回去”，不自动解决“数值会不会炸”。DeepNet 给出的 DeepNorm 做法是对白话里的“旧主干”和“新修正”重新配平。其编码器形式可写成：
$$x_{l+1}=\mathrm{LayerNorm}(\alpha x_l+G_l(x_l))$$
同时对子层内部部分权重乘以初始化系数 $\beta$。对 $N$ 层编码器，论文给出的稳定缩放是：
$$\alpha=(2N)^{1/4},\qquad \beta=(8N)^{-1/4}$$

这里的残差缩放就是“先把主干量级调到合适，再允许子层往上加修正”；$\beta$ 则是在初始化时压低子层写入强度，避免早期训练直接把残差流冲坏。

一个关键数值要算对。按 DeepNet 的编码器公式：

| 编码器层数 $N$ | $\alpha=(2N)^{1/4}$ | $\beta=(8N)^{-1/4}$ |
| --- | ---: | ---: |
| 100 | 3.761 | 0.188 |
| 200 | 4.472 | 0.158 |
| 1000 | 6.687 | 0.106 |

注意：$N=100$ 时 $\beta$ 约为 0.188，不是 0.37。这个差别不只是算术问题，而是直接影响子层初始写入量。

LoRA 可以从同一个“残差更新”视角理解。LoRA 的低秩适配就是“用很小的矩阵去近似原权重需要补上的改动”，即
$$W=W_0+\Delta W=W_0+BA$$
其中低秩是“用更少参数表示一个大矩阵更新”。前向变成
$$h=W_0x+BAx$$
这本质上仍然是“保留原路径，再叠加一个可控修正”。所以 LoRA 和残差连接虽然作用位置不同，一个在层输出上加，一个在权重上加，但设计思想一致：先保留基座，再叠加小更新。

真实工程例子是 DeepNet：它报告 200 层、3.2B 参数模型在大规模多语翻译上超过 48 层、12B 的基线。这说明深度不是不能扩，只是必须先把残差流稳定下来。

---

## 代码实现

先给一个可运行的玩具实现，用纯 Python 演示残差、DeepNorm 系数和 LoRA 更新：

```python
import math

def residual_stack(x, n_layers, delta):
    for _ in range(n_layers):
        x = x + delta
    return x

def deepnorm_coeff(n):
    alpha = (2 * n) ** 0.25
    beta = (8 * n) ** -0.25
    return alpha, beta

def matvec(W, x):
    return [sum(wij * xj for wij, xj in zip(wi, x)) for wi in W]

def lora_forward(W0, A, B, x):
    # W0: d x k, A: r x k, B: d x r
    base = matvec(W0, x)
    ax = matvec(A, x)
    bax = matvec(B, ax)
    return [u + v for u, v in zip(base, bax)]

alpha, beta = deepnorm_coeff(100)
assert round(alpha, 3) == 3.761
assert round(beta, 3) == 0.188

# 残差恒等路径：如果子层不写入，输出应保持不变
x = 5.0
assert residual_stack(x, 10, 0.0) == 5.0

# LoRA 玩具例子：W0x + BAx
W0 = [[1.0, 0.0], [0.0, 1.0]]
A = [[2.0, 0.0]]      # r=1
B = [[1.0], [3.0]]
x = [1.0, 2.0]
y = lora_forward(W0, A, B, x)
assert y == [3.0, 8.0]
```

上面这个玩具例子表达的是结构，不是训练速度。真正工程里会用 PyTorch，把残差块、LoRA 和 checkpoint 放在一起：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank

    def forward(self, x):
        delta = (x @ self.A.t()) @ self.B.t()
        return x @ self.weight.t() + self.scale * delta

class ResidualBlock(nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = LoRALinear(dim, dim, rank=rank)

    def _forward(self, x):
        h = self.proj(self.norm(x))
        return x + h

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
```

这里 checkpoint 的意思是“前向时少存中间激活，反向时重算一遍”；LoRA 的意思是“冻结原权重，只训练低秩增量”。

---

## 工程权衡与常见坑

真正训练超深 Transformer，不是只会写残差公式就够了。常见权衡如下：

| 技术 | 峰值显存 | 计算开销 | 主要作用 |
| --- | --- | --- | --- |
| Gradient checkpointing | 常见可降约 33% 左右 | 常见慢 20% 到 30%，有时更高 | 用重算换显存 |
| DeepNorm scaling | 几乎不增加显存 | 额外代价很小 | 控制深层残差写入量 |
| LoRA | 显著减少可训练参数 | 推理可合并，几乎无额外延迟 | 下游高效微调 |

常见坑有四个。

第一，只加残差、不做尺度控制。结果通常是浅层还能跑，层数一深就出现 loss spike，白话说是“每层都在往主干上用力写字，最后把纸写穿了”。

第二，把 DeepNorm 系数算错。尤其是 $\beta$，它直接决定子层初始化强度，数值错一倍，训练稳定性会明显变差。

第三，把 checkpoint 当成“白拿显存”。它本质是拿计算换内存，显存越紧张越值得开，但吞吐会下降。显存不是瓶颈时，盲目开启往往不划算。

第四，把 LoRA 当成“从零训练深模型”的替代品。它不是。LoRA 解决的是下游适配成本，不解决超深主干从头训练的稳定性。

---

## 替代方案与适用边界

如果任务是从零训练 100 层以上的 Transformer，优先考虑的是 DeepNorm 这类残差稳定方案，因为问题根源在主干深度扩展。

如果任务是已有大模型做下游微调，优先考虑 LoRA，因为它保留 $W_0$，只学习 $\Delta W=BA$，训练参数和显存都更省，合并后也不会引入额外推理路径。

如果任务受限于 GPU 显存，而不是参数量，那么 gradient checkpointing 往往最直接。它不改变模型表达能力，只改变训练时存激活的策略。

| 方法 | 核心改动 | 最适合的场景 | 不适合的场景 |
| --- | --- | --- | --- |
| DeepNorm | 调整残差和初始化的尺度 | 训练 100+ 层 Transformer | 只做轻量下游微调 |
| LoRA | 用低秩增量 $BA$ 叠加到 $W_0$ | PEFT、任务适配 | 替代深层主干稳定训练 |
| Gradient checkpointing | 反向时重算激活 | 显存受限训练 | 吞吐敏感、显存富余 |

所以实践上的组合通常是：超深从头训练用 DeepNorm 保残差流稳定；显存不够时叠加 checkpoint；下游任务再用 LoRA 做低成本适配。三者能组合，但解决的是三个不同层面的瓶颈。

---

## 参考资料

- DeepNet: Scaling Transformers to 1,000 Layers. arXiv:2203.00555. https://arxiv.org/abs/2203.00555
- LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Do Language Models Use Their Depth Efficiently? arXiv:2505.13898. https://arxiv.org/abs/2505.13898
- Hugging Face Transformers 文档，Gradient Checkpointing. https://huggingface.co/docs/transformers/en/main_classes/trainer
- Hugging Face Performance Guide，Gradient Checkpointing. https://huggingface.co/docs/transformers/v4.17.0/performance

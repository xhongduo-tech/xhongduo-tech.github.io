## 核心结论

SwiGLU 是 Transformer 前馈网络（FFN，指注意力层后面那段按位置独立计算的多层感知机）里常见的一种门控激活。它的标准形式是：

$$
\mathrm{SwiGLU}(x)=\mathrm{Swish}(xW_g)\odot(xW_v)
$$

其中，$\odot$ 表示逐元素乘，意思是同一位置上的两个数一一相乘；$W_g$ 是门分支的权重，$W_v$ 是值分支的权重。再接一层输出投影，就得到完整 FFN：

$$
y=(\mathrm{Swish}(xW_g)\odot(xW_v))W_o
$$

这里的 Swish，也常直接用 SiLU 实现，定义为：

$$
\mathrm{Swish}(z)=z\cdot \sigma(z), \quad \sigma(z)=\frac{1}{1+e^{-z}}
$$

核心结论有三点。

第一，SwiGLU 不是“把 ReLU 换成一个更平滑的函数”这么简单。它本质上是“双路投影 + 门控乘法”的结构变化，一路负责控制放行强度，一路负责提供内容值。

第二，它在现代大模型的 FFN 中通常比 ReLU/GELU 更有效。原因不是某个单独公式神奇，而是门控结构让网络能更细地决定“哪些维度该通过、放大、抑制或反向”。

第三，它更强通常也更贵。如果你把传统 FFN 原样替换成 SwiGLU，而中间维度不做调整，参数量和计算量往往会上升。所以 PaLM、LLaMA 这类模型通常会同步缩小 FFN 宽度，换取更好的效果/成本比。

用一句白话概括：SwiGLU 不是单路激活，而是让一条分支去“当门”，另一条分支去“装内容”，最后再把两者乘起来。

---

## 问题定义与边界

SwiGLU 要解决的问题，是传统 FFN 的单一路径非线性表达不够灵活。

先看普通 FFN。给定输入向量 $x$，传统两层 FFN 往往写成：

$$
y=\phi(xW_1)W_2
$$

其中 $\phi$ 是 ReLU 或 GELU 这样的激活函数。这里的特点是：只有一条主路径，先做一次线性变换，再统一过激活，最后映射回原维度。

这种结构足够简单，也很好用，但它有一个限制：每个中间维度是否被保留，主要由单一路径上的激活函数决定。ReLU 是硬截断，GELU 更平滑，但本质上都还是“一个向量自己决定自己”。

SwiGLU 的出发点是：不要只让一个中间表示单独过激活，而是拆成两路，让一路负责“值”，一路负责“门”。这样模型可以学到更细的条件控制。

这里要先把三个经常被混淆的概念分开。

| 名称 | 形式 | 是否门控 | 它是什么 |
|---|---|---:|---|
| SiLU / Swish | $z\cdot \sigma(z)$ | 否 | 单变量激活函数 |
| GLU | $A\odot \sigma(B)$ | 是 | 双路门控结构 |
| SwiGLU | $\mathrm{Swish}(A)\odot B$ | 是 | 用 Swish 做门的 GLU 变体 |

术语首次解释如下。

SiLU/Swish：一种平滑激活，意思是不会像 ReLU 那样在 0 点直接折断。  
GLU：Gated Linear Unit，门控线性单元，意思是“一路生成内容，一路生成门，再相乘”。  
SwiGLU：用 Swish 作为门分支非线性的 GLU 变体。

一个玩具例子可以先建立直觉。

假设你有两个长度都是 3 的向量：

- 值分支输出：$v=[10, 2, -4]$
- 门分支经过 Swish 后：$g=[0.9, 0.1, -0.5]$

逐元素相乘得到：

$$
g\odot v=[9, 0.2, 2]
$$

这说明门分支不只是“开或关”。它可以：

- 把第一维大部分放行
- 把第二维强烈压小
- 把第三维翻转符号并缩放

这就是“细粒度控制”。它不是简单删掉某些维度，而是在连续数值空间中调节通过量。

边界也要说清楚。本文讨论的是 Transformer 里的 FFN/MLP 层，不讨论以下内容：

- 注意力层内部的 Q/K/V 计算
- 卷积网络中的激活设计
- MoE、稀疏门控、路由器这类更复杂的门控系统
- 训练稳定性以外的系统优化细节，比如算子融合、内核实现

所以本文的范围很明确：SwiGLU 作为 FFN 激活结构，为什么有效，怎么实现，代价是什么，适合什么场景。

---

## 核心机制与推导

先从 Swish 开始。

Swish 的定义是：

$$
\mathrm{Swish}(z)=z\cdot \sigma(z)
$$

因为 sigmoid 的输出在 $(0,1)$ 之间，所以 Swish 可以理解为“让 $z$ 自己乘一个平滑的门值”。当 $z$ 很大时，$\sigma(z)\approx 1$，输出接近 $z$；当 $z$ 很小时，$\sigma(z)\approx 0$，输出接近 0，但不是像 ReLU 那样直接截断。

单看 Swish，它只是一个激活函数。SwiGLU 进一步把它放到双路结构里。

设输入是 $x\in\mathbb{R}^{d_{model}}$，先做两次线性投影：

$$
a=xW_g,\quad b=xW_v
$$

其中：

- $a$ 是门分支的预激活值
- $b$ 是值分支的内容值

然后对门分支做 Swish：

$$
g=\mathrm{Swish}(a)
$$

最后和内容值逐元素相乘：

$$
h=g\odot b
$$

若放回完整 FFN，再接输出投影：

$$
y=hW_o=(\mathrm{Swish}(xW_g)\odot(xW_v))W_o
$$

职责分工可以写成一张表。

| 分支 | 公式 | 职责 |
|---|---|---|
| 门分支 | $\mathrm{Swish}(xW_g)$ | 计算每个中间维度应该通过多少 |
| 值分支 | $xW_v$ | 提供被筛选的特征内容 |
| 合成输出 | $\mathrm{Swish}(xW_g)\odot(xW_v)$ | 得到有选择性的中间表示 |

这比普通 $\phi(xW)$ 多了一个关键自由度。普通 FFN 中，同一个中间向量既负责产生特征，又负责决定自己是否保留；SwiGLU 中，这两个任务拆开了。

看一个明确的数值例子。设：

$$
xW_g=[2,-1],\quad xW_v=[3,4]
$$

先算 sigmoid：

- $\sigma(2)\approx 0.881$
- $\sigma(-1)\approx 0.269$

所以：

$$
\mathrm{Swish}(2)=2\times 0.881\approx 1.762
$$

$$
\mathrm{Swish}(-1)=-1\times 0.269\approx -0.269
$$

于是门分支输出为：

$$
\mathrm{Swish}(xW_g)\approx [1.762,-0.269]
$$

再和值分支逐元素相乘：

$$
[1.762,-0.269]\odot[3,4]\approx[5.286,-1.076]
$$

这个例子说明两件事。

第一，门的值不局限于 0 到 1。因为是 Swish，不是纯 sigmoid，所以门分支自己也带幅值信息。  
第二，输出可以为负，说明门控不是简单掩码，而是连续可学习变换。

从推导角度，可以把普通 FFN 和 SwiGLU 并排看：

| 结构 | 中间表示 |
|---|---|
| ReLU FFN | $\mathrm{ReLU}(xW_1)$ |
| GELU FFN | $\mathrm{GELU}(xW_1)$ |
| SwiGLU FFN | $\mathrm{Swish}(xW_g)\odot(xW_v)$ |

差别不在“激活函数名字不同”，而在中间表示的构造方式不同。SwiGLU 的中间表示是双路交互后的结果，因此表达能力通常更强。

再看一个真实工程例子。大语言模型的每个 Transformer block 通常包含“注意力 + FFN”两大部分。注意力负责跨 token 混合信息，FFN 负责对每个 token 的通道维度做非线性变换。很多现代模型把 FFN 写成：

1. 输入 $x$ 先归一化
2. 通过两个上投影得到 `gate_proj` 和 `up_proj`
3. 做 `silu(gate_proj(x)) * up_proj(x)`
4. 再通过 `down_proj` 投影回模型维度

这就是工程代码里的 SwiGLU。LLaMA 系列里常见的 `gate_proj / up_proj / down_proj` 命名，正对应这三步。

为什么它适合 FFN？因为 FFN 的目标本来就不是跨位置通信，而是“对当前 token 的表示做通道重组与筛选”。门控结构恰好适合做这种按维度的细粒度控制。

---

## 代码实现

实现 SwiGLU 的最小思路只有三步：

1. 输入做两路线性投影
2. 门分支做 `SiLU/Swish`
3. 两路逐元素乘，再做输出投影

下面先给一个可运行的纯 Python 版本，便于验证数学定义。这里不用任何深度学习框架，只展示机制。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def swish(x: float) -> float:
    return x * sigmoid(x)

def swiglu(gate_vec, value_vec):
    assert len(gate_vec) == len(value_vec)
    return [swish(g) * v for g, v in zip(gate_vec, value_vec)]

# 玩具例子
gate = [2.0, -1.0]
value = [3.0, 4.0]
out = swiglu(gate, value)

assert len(out) == 2
assert abs(out[0] - 5.2848) < 1e-2
assert abs(out[1] - (-1.0758)) < 1e-2
```

这个例子对应前面的数值推导。你可以直接运行，检查公式是否一致。

再给一个更接近真实模型的 PyTorch 写法。这里的术语解释如下：

`d_model`：模型主通道维度，也就是 token 表示的宽度。  
`d_ff`：FFN 中间层维度，也叫 hidden size 或 intermediate size。  
`bias`：线性层偏置项，意思是额外加一个可学习常数向量。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_g = nn.Linear(d_model, d_ff, bias=bias)  # gate_proj
        self.w_v = nn.Linear(d_model, d_ff, bias=bias)  # up_proj
        self.w_o = nn.Linear(d_ff, d_model, bias=bias)  # down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.silu(self.w_g(x))
        v = self.w_v(x)
        h = g * v
        y = self.w_o(h)
        return y

# shape check
m = SwiGLUFFN(d_model=8, d_ff=16, bias=False)
x = torch.randn(2, 4, 8)  # batch=2, seq=4, d_model=8
y = m(x)

assert y.shape == (2, 4, 8)
```

代码对应关系很直接。

| 代码对象 | 数学含义 |
|---|---|
| `self.w_g` | $W_g$，门分支投影 |
| `self.w_v` | $W_v$，值分支投影 |
| `F.silu(...)` | $\mathrm{Swish}(\cdot)$ |
| `g * v` | 逐元素乘 $\odot$ |
| `self.w_o` | $W_o$，输出投影 |

如果你在复现 LLaMA 风格实现，还会遇到一个常见问题：中间维度不是传统 FFN 那个 `4 * d_model`，而是会缩小。

原因如下。普通两层 FFN 的主要参数量大致是：

$$
d_{model}\times d_{ff}+d_{ff}\times d_{model}=2d_{model}d_{ff}
$$

如果传统设定是 $d_{ff}=4d_{model}$，那么参数量约为：

$$
2d_{model}\cdot 4d_{model}=8d_{model}^2
$$

SwiGLU 有三层投影，参数量近似是：

$$
d_{model}d_{ff}+d_{model}d_{ff}+d_{ff}d_{model}=3d_{model}d_{ff}
$$

若想让它与传统 FFN 参数量接近，就需要：

$$
3d_{model}d_{ff}\approx 8d_{model}^2
$$

所以：

$$
d_{ff}\approx \frac{8}{3}d_{model}
$$

这就是很多实现把中间维度设成“大约原来 4 倍宽度的 $2/3$”的原因。它不是拍脑袋经验，而是参数预算平衡后的结果。

一个真实工程例子是：当你实现一个 GPT 类 block 时，如果你已经有 `Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)` 的 FFN，迁移到 SwiGLU 时，通常不能只把 `GELU` 替换成 `SiLU`。正确做法是把上投影拆成两层，并把中间维度改到合适值，否则模型配置、参数量和吞吐都会偏离目标。

---

## 工程权衡与常见坑

SwiGLU 的工程价值是“更好的表达能力”，代价是“更复杂的参数与计算结构”。这类设计没有免费午餐，权衡要明确。

先做一个对比。

| 方案 | 表达能力 | 计算成本 | 常见用途 |
|---|---|---|---|
| ReLU FFN | 中等 | 低 | 传统 MLP、早期网络 |
| GELU FFN | 较强 | 中等 | BERT 类 Transformer |
| SwiGLU FFN | 更强 | 较高，通常需调宽度 | PaLM、LLaMA 一类大模型 |

工程上最常见的收益，是在相近训练预算下获得更好的损失或下游效果。原因前面已经解释：双路门控让 FFN 对通道的选择更细。

但常见坑也很集中。

第一，把 `SiLU` 当成 `SwiGLU`。  
这是一类最常见的实现错误。`SiLU(xW)` 只是单路激活；`SiLU(xW_g) * (xW_v)` 才是 SwiGLU。少了一路投影和乘法，结构已经不是同一个东西。

第二，忘记输出投影 `W_o`。  
有些教程为了讲机制，只写中间激活 `Swish(xW_g) ⊙ (xW_v)`。如果你在模型里直接把它当最终输出，大概率会维度不匹配，因为 FFN 需要投影回 `d_model`。

第三，两路张量形状不一致。  
逐元素乘要求形状一致。比如一条分支输出 `[batch, seq, 4096]`，另一条是 `[batch, seq, 11008]`，就会直接报错。这通常来自 `d_ff` 配置不统一，或者切分权重时写错维度。

第四，不缩中间维度，导致成本上升。  
传统 FFN 只有两次大矩阵乘，SwiGLU 有三次。如果还保持同样的中间宽度，参数和 FLOP 都会上去。很多人做结构对比实验时没控制这个变量，最后得到的结论不可靠。

第五，`bias` 设置和目标实现不一致。  
有些论文或开源实现默认 `bias=False`，有些默认 `bias=True`。如果你的目标是复现某个已知模型，偏置项也要对齐。否则结果差异未必来自 SwiGLU 本身。

第六，把它当作“任何网络都更强”的通用替代。  
SwiGLU 在大型 Transformer 的 FFN 中很常见，不等于在所有轻量模型、所有数据规模、所有推理预算下都最优。小模型可能更受限于延迟、显存或实现复杂度。

可以用一个真实工程判断流程来理解。假设你在做一个面向线上服务的中型文本模型，目标是：

- 延迟不能明显增加
- 参数量最好维持原预算
- 想提升困惑度或指令跟随效果

这时你可以考虑：

1. 把 GELU FFN 替换为 SwiGLU
2. 将 `d_ff` 从 `4*d_model` 调到约 `8/3*d_model`
3. 保持其他 block 结构不变
4. 比较训练稳定性、验证集损失、推理吞吐

如果你直接把中间维度维持在 `4*d_model`，实验结果即使变好，也很难说明“是结构更优”，因为你同时给了它更多计算资源。

---

## 替代方案与适用边界

SwiGLU 不是唯一合理选择。它只是当前大模型 FFN 里非常主流的一种。

常见替代方案如下。

| 方案 | 门控 | 平滑性 | 典型场景 |
|---|---:|---:|---|
| ReLU | 否 | 否 | 简单 MLP、低成本场景 |
| GELU | 否 | 是 | 经典 Transformer |
| Swish / SiLU | 否 | 是 | 单路激活替代 |
| GeGLU | 是 | 是 | 门控 FFN |
| SwiGLU | 是 | 是 | 现代大模型 FFN |

这些名字容易混淆，区别可以压缩成一句话。

ReLU/GELU/SiLU：都是“单路激活”。  
GeGLU/SwiGLU/ReGLU：都是“门控 FFN 变体”。

其中 GeGLU 指门分支使用 GELU，ReGLU 指门分支使用 ReLU，SwiGLU 指门分支使用 Swish/SiLU。它们都属于 GLU 变体。

什么时候选 SwiGLU，什么时候不选，可以按目标判断。

如果你在做现代 Transformer 骨架，尤其是想靠近 PaLM、LLaMA 这类设计，SwiGLU 很合适。因为它已经在真实大模型中被广泛验证，且社区实现成熟。

如果你只是在做一个结构简单、预算紧张的小模型，GELU 往往更省心。原因很现实：代码更短，算子更少，调参经验更多。

如果你强调严格复现某篇论文或某个开源模型，那就不用抽象讨论，直接跟目标实现保持一致。模型结构上的一个小偏差，常常足以让训练曲线和最终指标偏掉。

还有一个边界：SwiGLU 的优势主要体现在 FFN 的通道表达，而不是替代注意力机制本身。它解决的是“每个 token 内部怎么变换通道”，不是“token 和 token 之间怎么交互”。

最后给一个简化选择规则。

| 场景 | 更合适的选择 |
|---|---|
| 教学、原型、小模型 | GELU 或 ReLU |
| 追求现代大模型骨架 | SwiGLU |
| 预算极紧、极端追求吞吐 | 先评估 GELU/ReLU |
| 复现特定论文 | 跟论文/官方实现一致 |

所以，SwiGLU 的适用边界不是“永远更好”，而是“在 Transformer FFN 里，常常是更好的工程选择，但前提是你愿意为它调整宽度并承担更复杂的实现”。

---

## 参考资料

1. Noam Shazeer, *GLU Variants Improve Transformer* (2020)  
   https://arxiv.org/abs/2002.05202  
   作用：给出 GLU、ReGLU、GeGLU、SwiGLU 等变体，并讨论它们在 Transformer FFN 中的效果。

2. Prajit Ramachandran, Barret Zoph, Quoc V. Le, *Swish: a Self-Gated Activation Function* (2017)  
   https://arxiv.org/abs/1710.05941  
   作用：提供 Swish/SiLU 的定义与动机，是理解 SwiGLU 中门分支非线性的基础。

3. Aakanksha Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022)  
   https://arxiv.org/abs/2204.02311  
   作用：展示 SwiGLU 在大规模语言模型中的工程落地。

4. Hugo Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023)  
   https://arxiv.org/abs/2302.13971  
   作用：展示现代开源大模型中基于 SwiGLU 的 FFN 设计实践。

5. 本文主要依赖的结论对应关系  
   - Swish 的数学定义：来自 Swish 论文  
   - SwiGLU 属于 GLU 变体、适合 Transformer FFN：来自 GLU Variants 论文  
   - PaLM、LLaMA 采用此类 FFN：来自各自模型论文

| 论文 | 主要贡献 | 对本文的作用 |
|---|---|---|
| Swish | 定义平滑自门控激活 | 解释门分支为何用 SiLU/Swish |
| GLU Variants | 系统比较门控 FFN 变体 | 解释 SwiGLU 的结构来源 |
| PaLM | 大模型工程采用 SwiGLU | 提供真实工程例子 |
| LLaMA | 开源大模型中的标准 FFN 实践 | 说明社区主流实现路径 |

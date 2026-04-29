## 核心结论

GeLU 和 SwiGLU 的比较，核心不是“哪个激活函数更高级”，而是“哪种 FFN 设计在给定预算下更值”。

先用一句话对比。GeLU 像“在同一条中间特征通路上，对每个数值做平滑缩放”；SwiGLU 像“先分出内容和闸门，再让闸门决定内容放行多少”。这里的 FFN 是 Transformer 里的前馈网络，白话说，就是注意力层后面那块负责做非线性变换和特征重组的模块。门控是指“用一条分支去控制另一条分支输出强弱”的机制。

在标准写法下，两者可以写成：

$$
\text{GeLU-FFN}: y = W_2 \cdot \text{GeLU}(W_1 x)
$$

$$
\text{SwiGLU-FFN}: y = W_o \cdot [\text{SiLU}(W_g x) \odot (W_v x)]
$$

其中 $\odot$ 表示逐元素相乘，意思是同一位置上的数一对一相乘。

结论有两层。

第一，GeLU 和 SwiGLU 不是同一层级的替换。GeLU 是单分支逐元素激活；SwiGLU 是双分支投影加门控相乘，属于 FFN 结构升级，而不只是把一个激活函数换成另一个。

第二，从算子效率看，GeLU 的路径更短、kernel 更少、实现更朴素；从单位预算效果看，SwiGLU 往往更强，但前提是比较方式公平，通常要把中间宽度按约 $2/3$ 缩放，并尽量使用 fused kernel。kernel 可以理解为“GPU 上一次具体执行的算子调用”；fused kernel 是把多步操作合并成一次更大的执行，以减少中间读写和调度开销。

总览如下：

| 项目 | GeLU-FFN | SwiGLU-FFN |
|---|---|---|
| 结构复杂度 | 低 | 更高 |
| 参数量 | 基线 | 若不缩放会更大 |
| kernel 数 | 少 | 更多 |
| 推理友好性 | 更好 | 依赖融合实现 |
| 表达能力 | 标准 | 通常更强 |

---

## 问题定义与边界

本文讨论的对象，不是孤立的 GeLU 激活函数和 SiLU 激活函数，而是 decoder-only Transformer 里的 FFN block 设计。decoder-only 的意思是“只有解码器堆叠、按自回归方式预测下一个 token”的语言模型结构，现代大语言模型大多属于这一类。

比较边界必须先说清楚，否则结论会失真。

如果你直接拿 `h=4d` 的 GeLU-FFN 和 `h=4d` 的 SwiGLU-FFN 比，SwiGLU 会天然多一条投影，因此参数更多、FLOPs 更高、显存流量更大。这样得出的“效果更好”没有太大说服力，因为它本来就更贵。FLOPs 是浮点运算次数，白话说，就是理论计算量。

更合理的比较方式，是让两者的预算接近，再看谁更值。本文默认采用以下边界：

| 维度 | 说明 |
|---|---|
| 模型块 | Transformer FFN |
| 对象 | GeLU-FFN vs SwiGLU-FFN |
| 比较标准 | 参数量、FLOPs、显存流量、延迟 |
| 不讨论 | 注意力层、归一化层、量化策略 |

预算对齐时，常用关系是：

$$
h_{\text{swiglu}} \approx \frac{2}{3} h_{\text{gelu}}
$$

原因很直接。GeLU-FFN 的参数主项是 $2dh$，而 SwiGLU-FFN 的参数主项是 $3dh$。如果要让两者参数量接近，就应让 SwiGLU 的中间宽度更窄。

一个玩具例子：令输入维度 $d=6$。若 GeLU 取 $h=24$，则参数主项为 $6\times24+24\times6=288$。若 SwiGLU 取 $h=16$，则参数主项为 $6\times16+6\times16+16\times6=288$。这时两者预算对齐，比较才公平。

所以，本文后续所有结论都应理解为：在参数量近似对齐、且实现合理的前提下，SwiGLU 往往能提供更好的表达能力和训练收益；但在纯算子简洁性上，GeLU 更有优势。

---

## 核心机制与推导

先看 GeLU。GeLU 的全称是 Gaussian Error Linear Unit，白话说，它是一种“对输入做平滑放行”的激活函数，不像 ReLU 那样一刀切地把负数清零。其定义常写为：

$$
\text{GeLU}(z)=z\Phi(z)
$$

这里 $\Phi(z)$ 是标准高斯分布的累积分布函数，白话说，就是“输入越大，被保留的比例越高”。

GeLU-FFN 的计算链路是：

$$
x \xrightarrow{W_1} h \xrightarrow{\text{GeLU}} h' \xrightarrow{W_2} y
$$

这条路径的特点是：只有一条中间表示通路，所有特征都在同一条路上被激活，再统一投回输出空间。它能表达非线性，但没有显式区分“内容”和“控制”。

再看 SwiGLU。SiLU 的定义是：

$$
\text{SiLU}(z)=z\sigma(z)
$$

其中 $\sigma(z)$ 是 sigmoid，白话说，是一个把数压到 $0$ 到 $1$ 附近的平滑函数。SwiGLU 的核心不在 SiLU 本身，而在“双分支 + 门控”：

$$
\text{SwiGLU}(x)=\text{SiLU}(W_gx)\odot (W_vx)
$$

这里 $W_gx$ 是门分支，决定放行比例；$W_vx$ 是值分支，也可以叫内容分支，承载真正要传递的信息。最后再经过 $W_o$ 投回模型维度。

最小数值例子可以直接说明门控在做什么。设 $x=2,\ W_g=0.5,\ W_v=1.5$。则：

$$
W_gx=1,\quad \text{SiLU}(1)\approx 0.731
$$

$$
W_vx=3
$$

于是门控后的值为：

$$
0.731 \times 3 \approx 2.19
$$

这表示内容分支本来想输出 `3`，但门分支认为只能放行约 `73.1%`，于是实际通过量变成 `2.19`。这不是“同一个值自己过激活”，而是“一个值被另一路信号控制”。这类显式分工，通常能给模型带来更细的特征选择能力。

参数量也能看出两者不是一个层级。

$$
\text{GeLU-FFN}: d\times h + h\times d = 2dh
$$

$$
\text{SwiGLU-FFN}: d\times h + d\times h + h\times d = 3dh
$$

令两者预算对齐：

$$
2d h_{\text{gelu}} \approx 3d h_{\text{swiglu}}
$$

可得：

$$
h_{\text{swiglu}} \approx \frac{2}{3} h_{\text{gelu}}
$$

如果传统 GeLU-FFN 常用 $h_{\text{gelu}}=4d$，那么对齐后通常取：

$$
h_{\text{swiglu}} \approx \frac{8d}{3}
$$

两者机制对比如下：

| 项目 | GeLU | SwiGLU |
|---|---|---|
| 线性层数 | 2 | 3 |
| 激活/门控 | 1 个激活 | 1 个激活 + 1 个乘法 |
| 计算图 | 简单 | 更复杂 |
| 表达能力 | 标准 | 更强 |

真实工程例子是在 decoder-only LLM 的 FFN 选型里。如果目标是“同等参数和训练算力下，把困惑度或下游指标再压一点”，团队通常不会只看单个激活函数，而会直接比较完整 FFN block。现代大模型大量采用 SwiGLU，本质原因不是它“看起来更复杂”，而是它常在相近预算下带来更好的训练收益。

---

## 代码实现

实现时最容易犯的错误，是把 SwiGLU 理解成“把 `gelu` 换成 `silu` 就行”。这是错的。SwiGLU 多了一条投影分支，维度关系也变了。

下面给一个可运行的最小 Python 例子。为了保证任何环境都能跑，这里先用纯 Python 写出玩具版，再给 PyTorch 结构版。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def silu(x: float) -> float:
    return x * sigmoid(x)

def gelu_tanh_approx(x: float) -> float:
    # 常见近似版 GeLU
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))

def gelu_ffn_scalar(x: float, w1: float, w2: float) -> float:
    hidden = w1 * x
    return w2 * gelu_tanh_approx(hidden)

def swiglu_ffn_scalar(x: float, wg: float, wv: float, wo: float) -> float:
    gate = silu(wg * x)
    value = wv * x
    return wo * (gate * value)

# 玩具例子
x = 2.0
out = swiglu_ffn_scalar(x, wg=0.5, wv=1.5, wo=1.0)
assert abs(out - 2.1931757359) < 1e-6

# 基本性质：当门接近 0 时，SwiGLU 输出应被压低
small_gate_out = swiglu_ffn_scalar(x, wg=-10.0, wv=1.5, wo=1.0)
assert small_gate_out < 0.01

# GeLU 与 SwiGLU 都应是确定性函数
assert gelu_ffn_scalar(2.0, 1.2, 0.8) == gelu_ffn_scalar(2.0, 1.2, 0.8)
```

PyTorch 版本更贴近实际模型：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        # 单分支 + 激活
        return self.w2(F.gelu(self.w1(x)))

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.wg = nn.Linear(d_model, hidden_dim)
        self.wv = nn.Linear(d_model, hidden_dim)
        self.wo = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        # 门控分支 + 内容分支 + 逐元素乘
        gate = F.silu(self.wg(x))
        value = self.wv(x)
        return self.wo(gate * value)
```

如果进入工程版本，通常会继续做两件事。

第一，把两路输入投影合并成一次更大的 matmul。matmul 是矩阵乘法，白话说，就是神经网络里最贵的主计算。比如把 `Wg` 和 `Wv` 在实现上拼接成一个更大的权重，一次算出两路结果，再切分。

第二，尽量复用中间张量并使用 fused kernel，把 `bias + silu + mul` 一类操作融合，减少显存往返。真正影响延迟的，往往不只是 FLOPs，还包括 kernel 调度和内存带宽占用。

---

## 工程权衡与常见坑

SwiGLU 的优势不是无条件的。它在“同等预算下更强”这件事上经常成立，但前提是实现栈足够成熟。如果没有 fused kernel，它额外的一次投影、一次逐元素乘和更多中间张量，都会增加显存流量和延迟。显存流量可以理解为“数据在显存和计算单元之间搬运的总量”。

一个典型工程决策场景是这样的。你在训练 decoder-only LLM，目标是同等预算下尽量提高训练收益，那么通常优先考虑 SwiGLU，并把中间维度缩到约 `2/3`。如果你的目标变成“推理链路最短、实现最稳、部署最简单”，GeLU 往往更合适，因为它的算子路径短，依赖的融合也更少。

常见坑如下：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 把 SwiGLU 当成“只换激活” | 误判计算图和成本 | 明确它是结构升级 |
| 不做 `2/3` 缩放 | 参数和算力膨胀 | 按预算重新设 hidden dim |
| 没有 fused kernel | 延迟变差 | 使用融合实现或高性能内核 |
| 混淆精确 GeLU 和近似 GeLU | 数值和性能对不上 | 比较时注明实现版本 |

还要注意 GeLU 自己也有实现差异。很多框架里，GeLU 既有精确版，也有 tanh 近似版。近似版速度常更好，但数值略有差别。如果你做基准测试，不说明版本，很容易把“实现差异”误当成“结构差异”。

从性能路径看，也可以粗略总结：

| 指标 | GeLU | SwiGLU |
|---|---|---|
| kernel 数 | 少 | 多 |
| 中间张量 | 少 | 多 |
| 显存流量 | 低 | 高 |
| 表达能力 | 中等 | 更强 |

所以，SwiGLU 的收益通常来自“更好的表达能力”，而它的代价通常落在“更多算子和更重内存路径”上。现代大模型之所以经常选它，是因为训练系统已经足够强，能把这部分代价压低；一旦系统栈不支持，理论上的优势不一定能稳定落到真实延迟上。

---

## 替代方案与适用边界

SwiGLU 不是唯一替代方案。GLU 是 Gated Linear Unit，白话说，就是“一类用门控乘法做特征筛选的前馈结构总称”。它下面还有 ReGLU、GEGLU 等变体，只是本文重点不在列举家族，而在解释为什么现代 LLM 常从 GeLU-FFN 走向 SwiGLU-FFN。

按目标选结构，通常比问“谁绝对更好”更有意义：

| 目标 | 更适合 |
|---|---|
| 最简单实现 | GeLU |
| 最短推理链路 | GeLU |
| 同预算更高效果 | SwiGLU |
| 大模型训练主流配置 | SwiGLU |

边界关系再重复一次：

$$
h_{\text{swiglu}} \approx \frac{2}{3} h_{\text{gelu}}
$$

若常见配置里 $h_{\text{gelu}}=4d$，则：

$$
h_{\text{swiglu}} \approx \frac{8d}{3}
$$

什么时候不优先选 SwiGLU？

第一，预算极紧，且你的部署目标更关心稳定延迟而不是训练收益上限。  
第二，实现栈不支持高质量融合，导致额外 GEMM 和中间张量开销明显。  
第三，已有成熟 GeLU 推理优化，切换成本大于潜在收益。  
第四，你在做论文复现或教学基线，希望结构最标准、最容易和历史工作对齐。

反过来，什么时候更应优先考虑 SwiGLU？

第一，你在做大模型训练，目标是提升单位参数或单位算力的效果。  
第二，你的训练和推理框架已经有 fused MLP、tensor parallel 等成熟支持。tensor parallel 是把大矩阵切开到多卡并行算，白话说，就是把一层的计算拆给多张卡分担。  
第三，你更在意最终模型质量，而不是单个 FFN block 的最简实现。

因此，GeLU 仍然是合理基线，SwiGLU 则更像现代大模型里的默认升级项。两者不是“旧方案”和“新方案”的简单关系，而是“更朴素的计算图”和“更强但更依赖工程实现的计算图”的权衡。

---

## 参考资料

1. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
2. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
3. [PyTorch 文档：torch.nn.GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)
4. [PyTorch 文档：torch.nn.functional.silu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
5. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

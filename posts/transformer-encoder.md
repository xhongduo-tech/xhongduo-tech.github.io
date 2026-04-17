## 核心结论

Transformer Encoder 里的前馈层（FFN，Feed-Forward Network）不是注意力后的附属模块，而是每层表达能力、参数容量和局部记忆能力的核心来源之一。标准写法是：

$$
\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
$$

其中 $\phi$ 通常是 ReLU 或 GELU。它先把每个 token 的向量从 $d_{model}$ 扩展到更宽的中间维度 $d_{ff}$，再压回 $d_{model}$。常见配置是：

$$
d_{ff}=4\times d_{model}
$$

例如 BERT-Base 的 FFN 维度是：

$$
768 \rightarrow 3072 \rightarrow 768
$$

忽略偏置时，FFN 的参数量约为：

$$
\#\mathrm{FFN}=2\times d_{model}\times d_{ff}
$$

当 $d_{ff}=4d_{model}$ 时，有：

$$
\#\mathrm{FFN}=8d_{model}^2
$$

而标准多头注意力里，$Q,K,V,O$ 四个线性投影的参数量大约是：

$$
\#\mathrm{MHA}\approx 4d_{model}^2
$$

这意味着在许多标准配置里，FFN 的参数量通常接近注意力模块的两倍，往往是单层参数的主要组成部分。直观地说，Self-Attention 负责决定“当前 token 应该从上下文哪里取信息”，FFN 负责把这些信息变成“下一层可直接使用的新表示”。

已有研究进一步把 FFN 解释为一种“键值记忆”。第一层线性变换像用一组可学习“键”去匹配输入模式，第二层线性变换像把匹配结果投影成“值”，写回 token 表示。因此，FFN 的作用不只是增加非线性，还承担了逐位置写入内部知识的工作。

残差连接（Residual Connection）也不是“顺手加一条捷径”这么简单。标准 Pre-LN 写法是：

$$
y=x+\mathrm{Sublayer}(\mathrm{LN}(x))
$$

它的关键作用是给反向传播保留一条恒等通路。即使子层的梯度变弱，残差分支仍然提供一个系数为 1 的直接梯度项，所以深层 Transformer 更容易训练稳定。对现代深层 Transformer 来说，Pre-LN 往往比 Post-LN 更容易训练成功。

---

## 问题定义与边界

这篇文章只讨论标准 Transformer Encoder 中两个问题：

1. 前馈层为什么不是“注意力后随便接一个 MLP”，而是每层表达和记忆能力的重要来源。
2. 残差连接为什么会直接影响深层训练稳定性，尤其是 Pre-LN 与 Post-LN 的差异。

这里讨论的对象是标准编码器层。用 Pre-LN 形式表示，可以写成：

$$
x \rightarrow \mathrm{LN} \rightarrow \mathrm{MHA} \rightarrow \mathrm{Residual}
\rightarrow \mathrm{LN} \rightarrow \mathrm{FFN} \rightarrow \mathrm{Residual}
$$

不展开以下变体：

| 范围内 | 不展开 |
|---|---|
| 标准 Encoder 的 Self-Attention | Cross-Attention |
| 逐位置 FFN | MoE、稀疏 FFN 的实现细节 |
| Pre-LN / Post-LN 的训练差异 | 卷积混合架构、状态空间模型 |
| 单层机制与参数规模 | 长上下文优化、KV Cache、推理系统工程 |

先给一个面向新手的最小框架。单层 Encoder 不是“一个注意力模块”这么简单，而是两个功能不同的子层叠在一起：

1. 注意力层负责跨 token 聚合信息。
2. FFN 负责对每个 token 单独重编码。
3. 残差与 LayerNorm 负责让这个过程能稳定堆很多层。

可以把它理解成两步：

1. 先“读”：注意力判断当前 token 该看上下文中的哪些位置。
2. 再“写”：FFN 把读到的信息写成新的内部表示。

如果只有注意力，没有 FFN，模型就更像一个会做加权平均的路由系统，擅长“从哪里拿信息”，但不擅长“把拿到的信息变成新的抽象特征”。如果只有 FFN，没有注意力，模型又只能逐位置独立处理 token，无法建模上下文依赖。

下表先把三个关键组件拆开：

| 模块 | 结构 | 主要作用 | 对新手的直观理解 |
|---|---|---|---|
| MHA | token 间交互 | 聚合上下文信息 | 决定当前词该看谁 |
| FFN | $d_{model}\to d_{ff}\to d_{model}$ | 做逐位置非线性变换，提升表达与记忆容量 | 把聚合到的信息加工成新特征 |
| 残差 + LN | $x+\mathrm{Sublayer}(\cdot)$，配合归一化 | 稳定深层训练 | 保留旧表示，再叠加本层修正 |

看一个极短例子：

| 输入序列 | 当前关注 token | 注意力可能做什么 | FFN 可能做什么 |
|---|---|---|---|
| `猫 追 老鼠` | `追` | 聚合 `猫` 和 `老鼠` 的语义关系 | 把“动作关系”“施事-受事结构”写入 `追` 的新表示 |
| `not good` | `good` | 感知前面的否定词 `not` | 把“极性反转”写成内部特征 |
| `for i in range` | `range` | 聚合代码上下文 | 把“循环结构”“Python 语法模式”写入表示 |

所以问题不只是“有没有非线性”，而是两个更具体的问题：

1. 这个非线性模块是否承担了稳定、可复用的模式写入工作。
2. 当层数很深时，这个写入过程能否在反向传播时保持可训练。

---

## 核心机制与推导

### 1. FFN 到底在做什么

标准 FFN 的形式是：

$$
\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
$$

其中：

| 符号 | 含义 | 形状 |
|---|---|---|
| $x$ | 单个 token 的输入向量 | $1 \times d_{model}$ |
| $W_1$ | 第一层权重 | $d_{model}\times d_{ff}$ |
| $b_1$ | 第一层偏置 | $d_{ff}$ |
| $\phi$ | 激活函数，如 ReLU、GELU | 逐元素 |
| $W_2$ | 第二层权重 | $d_{ff}\times d_{model}$ |
| $b_2$ | 第二层偏置 | $d_{model}$ |

“逐位置计算”这句话容易被初学者误解。它的准确含义是：

1. 每个 token 都经过同一套 FFN 参数。
2. FFN 在内部不让不同 token 直接相互通信。
3. token 间交互已经在注意力层完成。

假设输入批量形状为：

$$
X\in \mathbb{R}^{B\times T\times d_{model}}
$$

那么 FFN 的作用方式是对每个位置 $(b,t)$ 上的向量 $X_{b,t,:}$ 独立应用同一个函数 $\mathrm{FFN}(\cdot)$。可以写成：

$$
Y_{b,t,:}=\mathrm{FFN}(X_{b,t,:})
$$

这意味着 FFN 更像“对每个 token 的局部重编码器”，而不是“跨 token 的交互模块”。

### 2. 为什么先扩维再压回

第一层把向量从 $d_{model}$ 扩到 $d_{ff}$，不是为了形式好看，而是为了给模型更宽的中间特征空间。扩维后的每个通道都可以学习成一个模式探测器。

例如，一个中间维度可能对下面某类模式高响应：

| 模式类型 | 可能对应的输入特征 |
|---|---|
| 否定关系 | `not`, `never`, `no` 等词及其上下文 |
| 时态关系 | 动词过去式、时间副词 |
| 句法角色 | 主语、宾语、修饰关系 |
| 代码语法 | `if`, `for`, `class`, 括号配对 |
| 数学结构 | 运算符、括号层级、变量依赖 |

激活函数的作用是引入非线性选择。以 ReLU 为例：

$$
\mathrm{ReLU}(z)=\max(0,z)
$$

它会让不匹配模式的通道输出为 0，只保留匹配较强的通道。GELU 更平滑，常用于 BERT、GPT 一类模型。其常见近似写法为：

$$
\mathrm{GELU}(z)\approx 0.5z\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}\left(z+0.044715z^3\right)\right]\right)
$$

对新手来说，不必先记公式，只需要记住一点：ReLU 像“硬截断”，GELU 像“平滑门控”。两者都在做一件事，即让某些中间特征通过、另一些被压低。

### 3. 为什么说 FFN 像键值记忆

把第一层拆成若干行向量看，会更容易理解。对第 $j$ 个中间通道，有：

$$
h_j=\phi(x\cdot w_{1,j}+b_{1,j})
$$

其中 $w_{1,j}$ 是 $W_1$ 的第 $j$ 列或等价表示下的一组参数。它像一个“模式匹配器”：如果输入 $x$ 符合这个模式，则 $h_j$ 较大；否则较小。

第二层再把这些中间激活组合回输出空间：

$$
y=\sum_{j=1}^{d_{ff}} h_j w_{2,j}+b_2
$$

这里可以把 $h_j$ 看成“匹配强度”，把 $w_{2,j}$ 看成“写回模板”。于是整个 FFN 可以理解为：

1. 先查：当前 token 是否匹配若干模式。
2. 再写：把匹配到的模式转换成输出表示。

这就是“键值记忆”解释的直观版本。它不是说 FFN 真有一个显式哈希表，而是说它的结构功能类似“模式匹配后写值”。

### 4. 参数量为什么说明 FFN 很重要

忽略偏置时，FFN 的参数量是：

$$
\#\mathrm{FFN}=d_{model}d_{ff}+d_{ff}d_{model}=2d_{model}d_{ff}
$$

如果取标准配置：

$$
d_{ff}=4d_{model}
$$

则有：

$$
\#\mathrm{FFN}=8d_{model}^2
$$

标准多头注意力的线性投影参数量近似为：

$$
\#\mathrm{MHA}=d_{model}^2+d_{model}^2+d_{model}^2+d_{model}^2=4d_{model}^2
$$

所以在很多标准 Transformer 层中：

$$
\#\mathrm{FFN}\approx 2\times \#\mathrm{MHA\ projections}
$$

以 BERT-Base 为例：

$$
d_{model}=768,\quad d_{ff}=3072
$$

则单层 FFN 权重数约为：

$$
2\times 768\times 3072=4{,}718{,}592
$$

如果把偏置也算上：

$$
4{,}718{,}592 + 3072 + 768 = 4{,}722{,}432
$$

再看 BERT-Large：

$$
d_{model}=1024,\quad d_{ff}=4096
$$

则单层 FFN 权重数约为：

$$
2\times 1024\times 4096=8{,}388{,}608
$$

这个量级说明一个事实：很多人把 Transformer 简化成“注意力模型”，但从参数预算看，FFN 往往才是层内最重的部分之一。

### 5. 残差连接为什么能改善训练稳定性

Pre-LN 子层写成：

$$
y=x+F(\mathrm{LN}(x))
$$

对输入 $x$ 求导：

$$
\frac{\partial y}{\partial x}
=
I+\frac{\partial F(\mathrm{LN}(x))}{\partial x}
$$

这里的 $I$ 是恒等矩阵。它的含义很直接：

1. 即使子层梯度很弱，至少还有一条直接传回去的梯度。
2. 深层连乘时，不会完全依赖复杂子层的雅可比矩阵。
3. 网络更容易从“近似恒等映射”开始训练，再逐步学偏离。

如果没有残差，堆很多层时，梯度完全依赖各层复杂变换的连乘，更容易衰减或爆炸。

### 6. 为什么 Pre-LN 和 Post-LN 不一样

Post-LN 写法是：

$$
y=\mathrm{LN}(x+F(x))
$$

它和 Pre-LN 的关键差异不在“有没有 LayerNorm”，而在 LayerNorm 放在什么位置。

| 结构 | 公式 | 直观结果 |
|---|---|---|
| Pre-LN | $x+F(\mathrm{LN}(x))$ | 残差主干更接近裸露直连 |
| Post-LN | $\mathrm{LN}(x+F(x))$ | 梯度回传时需要先经过 LN |

对新手可以这样理解：

1. Pre-LN：先把子层输入规范好，再把结果加回原输入。
2. Post-LN：先相加，再整体做归一化。

从训练角度看，Pre-LN 通常更适合深层网络，因为恒等路径更清晰。Post-LN 并不是不能训练，但常常更依赖 warm-up、学习率和初始化技巧。

### 7. 用一个小例子串起来

假设某层里，token `book` 在句子 `I book a ticket` 中出现。注意力先利用上下文判断这里的 `book` 更像动词而不是名词。接下来 FFN 可能做的不是“再看别人”，而是把下面这些局部特征写进 `book` 的新表示：

| 可能被激活的中间特征 | 含义 |
|---|---|
| 动词用法特征 | 当前上下文中更像动作 |
| 及物性特征 | 后面跟宾语 `ticket` |
| 订票语义特征 | 与旅行、预约相关 |
| 句法角色特征 | 谓语中心 |

最终残差把旧表示和这些新写入特征相加，因此模型不是每层完全重写 token，而是在已有表示上逐步修正。

---

## 代码实现

下面先给一个可运行的纯 Python 版本。它不依赖任何深度学习框架，目的是把 FFN、LayerNorm 和 Pre-LN 残差的计算顺序写清楚。

```python
import math
import random


def layer_norm(vec, eps=1e-5):
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / len(vec)
    return [(x - mean) / math.sqrt(var + eps) for x in vec]


def linear(vec, weight, bias):
    # weight shape: [out_dim][in_dim]
    out = []
    for row, b in zip(weight, bias):
        value = sum(v * w for v, w in zip(vec, row)) + b
        out.append(value)
    return out


def relu(vec):
    return [max(0.0, x) for x in vec]


def add(a, b):
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]


def make_matrix(out_dim, in_dim, seed):
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(in_dim)
    return [
        [(rng.random() * 2.0 - 1.0) * scale for _ in range(in_dim)]
        for _ in range(out_dim)
    ]


def ffn(vec, w1, b1, w2, b2):
    hidden = linear(vec, w1, b1)
    hidden = relu(hidden)
    output = linear(hidden, w2, b2)
    return output


def ffn_block_pre_ln(x, w1, b1, w2, b2):
    normed = layer_norm(x)
    transformed = ffn(normed, w1, b1, w2, b2)
    return add(x, transformed)


def param_count(d_model, d_ff, with_bias=True):
    count = d_model * d_ff + d_ff * d_model
    if with_bias:
        count += d_ff + d_model
    return count


def main():
    d_model = 4
    d_ff = 8

    x = [1.0, 2.0, 3.0, 4.0]
    w1 = make_matrix(d_ff, d_model, seed=1)
    b1 = [0.0] * d_ff
    w2 = make_matrix(d_model, d_ff, seed=2)
    b2 = [0.0] * d_model

    y = ffn_block_pre_ln(x, w1, b1, w2, b2)

    print("input :", [round(v, 4) for v in x])
    print("output:", [round(v, 4) for v in y])
    print("params(64->128->64, with_bias=True):", param_count(64, 128))
    print("params(768->3072->768, with_bias=False):", param_count(768, 3072, with_bias=False))

    assert len(y) == d_model
    assert param_count(64, 128) == 16576
    assert param_count(768, 3072, with_bias=False) == 4718592


if __name__ == "__main__":
    main()
```

这段代码有几个点值得明确说明：

| 代码部分 | 作用 | 对应理论概念 |
|---|---|---|
| `layer_norm(x)` | 先做归一化 | Pre-LN |
| `linear -> relu -> linear` | 两层逐位置变换 | FFN |
| `add(x, transformed)` | 原输入和子层输出相加 | 残差连接 |
| `param_count` | 计算 FFN 参数量 | 说明 FFN 的容量占比 |

如果你运行这段代码，输出维度会保持不变。这一点很重要，因为 Transformer 的每个子层虽然内部会扩维，但最终都要回到 $d_{model}$，这样才能继续堆叠后续层。

再给一个更接近工程实践的 PyTorch 版本：

```python
import torch
import torch.nn as nn


class PreNormFFN(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return x + y


if __name__ == "__main__":
    x = torch.randn(2, 5, 768)
    block = PreNormFFN(d_model=768, d_ff=3072, p=0.1)
    y = block(x)

    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("ffn params   :", sum(p.numel() for p in block.parameters()))

    assert y.shape == x.shape
```

如果把它放进完整 Encoder Block，常见顺序是：

```python
x = x + dropout(attn(layer_norm1(x)))
x = x + dropout(ffn(layer_norm2(x)))
```

这里可以看出两个事实：

1. 注意力子层和 FFN 子层都共享同一种“Pre-LN + Residual”骨架。
2. 两个子层的功能不同，但训练稳定性依赖同一种残差设计。

为了避免初学者把“逐位置”理解错，再补一个形状表：

| 张量 | 形状 | 含义 |
|---|---|---|
| `x` | `[B, T, d_model]` | 一个批次的 token 表示 |
| `fc1(x)` | `[B, T, d_ff]` | 每个 token 被扩维 |
| `act(fc1(x))` | `[B, T, d_ff]` | 非线性筛选后的中间特征 |
| `fc2(...)` | `[B, T, d_model]` | 再压回模型维度 |
| `x + y` | `[B, T, d_model]` | 残差输出，能继续堆叠 |

---

## 工程权衡与常见坑

### 1. 误把“有残差”当成“自动稳定”

最常见的说法是“残差可以防止信息丢失”。这句话不算错，但太模糊。更准确的说法是：残差为反向传播提供了恒等梯度通路，而 Pre-LN 是否成立，决定这条通路够不够直接。

| 结构 | 形式 | 梯度通路 | 训练表现 |
|---|---|---|---|
| Pre-LN | $x+F(\mathrm{LN}(x))$ | 恒等项直接存在 | 深层更稳定，warm-up 压力更小 |
| Post-LN | $\mathrm{LN}(x+F(x))$ | 梯度需经过 LN | 深层更脆弱，更依赖超参 |
| 无残差 | $F(x)$ | 完全依赖子层梯度 | 深层通常更难训练 |

所以判断训练是否稳定，不能只问“有没有残差”，还要问“残差是不是被额外变换打断了”。

### 2. 只盯注意力头数，忽略 FFN 宽度

很多人讲 Transformer 时，会花大量篇幅讨论头数、注意力图、注意力模式，却低估 FFN 宽度的影响。实际上，缩小 $d_{ff}$ 往往会直接压缩模型的中间特征空间，降低表达能力。

下面是一个直观对比：

| 配置 | $d_{model}$ | $d_{ff}$ | FFN 权重数 |
|---|---:|---:|---:|
| 小模型示例 | 256 | 1024 | 524,288 |
| BERT-Base | 768 | 3072 | 4,718,592 |
| BERT-Large | 1024 | 4096 | 8,388,608 |

因此，FFN 宽度不是可有可无的细节，而是容量设计的重要旋钮。

### 3. 认为 Pre-LN 没有代价

Pre-LN 通常更容易训练，但不代表没有副作用。层数非常深时，可能出现后层表示变化越来越小、各层输出过于相似的现象，常被概括为表示塌缩或层间差异减弱。

工程上常见缓解方法包括：

| 方法 | 核心思想 | 代价 |
|---|---|---|
| 残差缩放 | 用 $x+\alpha F(\mathrm{LN}(x))$ 控制子层更新幅度 | 需要调 $\alpha$ |
| 更稳的初始化 | 降低深层早期更新震荡 | 训练配置更敏感 |
| 学习率与 warm-up 调整 | 让深层更新更平滑 | 收敛可能更慢 |
| NormFormer 类改法 | 在已有骨架上加更细的归一化/缩放 | 结构更复杂 |

对新手最实用的结论是：Pre-LN 常是更稳的默认起点，但模型很深时仍然要关心优化细节。

### 4. 误以为 FFN 能替代注意力

FFN 不负责 token 间通信。它只能对每个位置单独做变换。因此：

1. 注意力负责“从别的位置读取信息”。
2. FFN 负责“把读到的信息写成新的本地表示”。

两者缺一不可。只看其中之一，都会误判 Transformer 的能力来源。

### 5. 代码里常见的实现错误

下面这些错误在手写 Transformer 时很常见：

| 错误 | 后果 |
|---|---|
| 忘记把 `fc2` 输出压回 `d_model` | 残差无法相加，维度不匹配 |
| 把 FFN 写成跨 token 共享错误维度的线性层 | 逻辑混乱，破坏逐位置语义 |
| Post-LN/Pre-LN 顺序写反 | 训练行为和预期不一致 |
| 残差前后忘记 dropout 或位置放错 | 正则效果异常 |
| 误把“每个 token 独立处理”理解成“每个 token 用不同参数” | 与实际共享参数机制相反 |

---

## 替代方案与适用边界

标准 FFN 不是唯一选择，但它长期稳定存在，是因为它简单、并行友好、参数效率高，而且在大量任务上足够有效。

### 1. 残差缩放

一种常见改法是：

$$
y=x+\alpha\cdot F(\mathrm{LN}(x))
$$

其中 $\alpha<1$。它的作用不是改变 FFN 本身，而是限制每层对子表示的扰动幅度。对超深网络来说，这往往比直接硬堆层数更稳。

### 2. NormFormer 一类改进

这类方法不是推翻 Pre-LN，而是在原有结构上增加更精细的缩放或归一化控制，目标是同时保留梯度通路和层间有效更新。它更像是在标准骨架上做训练稳定性增强，而不是改掉 Encoder 的主逻辑。

### 3. 稀疏 FFN 与 MoE

MoE（Mixture of Experts）会在许多专家 FFN 中只激活少数几个，从而在不线性增加每 token 计算量的前提下扩大总参数量。它适合超大模型，但边界也很明确：

| 方案 | 适合场景 | 优点 | 风险 |
|---|---|---|---|
| 标准 FFN + Pre-LN | 通用 Encoder、BERT 类模型 | 简单稳定，工程成熟 | 超深时仍需优化技巧 |
| 残差缩放 | 深层 Transformer | 更易控更新幅度 | 需要调参 |
| NormFormer 类改进 | 层数更深、训练不稳 | 缓解训练脆弱性 | 实现更复杂 |
| 稀疏/MoE FFN | 超大参数模型 | 扩大容量更高效 | 路由、并行和系统复杂度高 |

### 4. 适用边界到底在哪里

如果你讨论的是“标准 Encoder 为什么能工作”，那么 FFN + 残差就是最核心的主干机制。如果你讨论的是“超深、超大模型怎么进一步优化”，那才需要引入残差缩放、NormFormer、MoE 等变体。

因此边界可以概括成一句话：

1. 先把标准 FFN 和残差讲清楚，才能理解后续改法到底在修什么问题。
2. 大多数入门和中等规模工程场景，标准 FFN + Pre-LN 已经是足够合理的默认方案。

---

## 参考资料

| 资料 | 核心贡献 | 如何阅读 |
|---|---|---|
| Vaswani et al., *Attention Is All You Need* | 给出 Transformer 基本结构，明确 FFN 是逐位置两层网络 | 先看 Encoder Block 图，再看公式里的 position-wise feed-forward networks |
| Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | 展示 $768 \to 3072 \to 768$ 与 $1024 \to 4096 \to 1024$ 等标准配置 | 重点看模型结构表和隐藏维度设置 |
| Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* | 把 FFN 从“非线性层”推进到“键值记忆”解释 | 重点看 FFN 如何把模式匹配映射成输出值 |
| Xiong et al., *On Layer Normalization in the Transformer Architecture* | 系统分析 Pre-LN 与 Post-LN 的训练差异 | 重点看梯度稳定性和 warm-up 需求的讨论 |
| PyTorch `TransformerEncoderLayer` 文档与源码 | 展示工业实现中的层顺序、dropout 和归一化封装方式 | 把源码里的张量形状和论文公式一一对照 |

下面补一个更适合入门者的阅读顺序：

1. 先看 Vaswani 论文，确认 FFN 在原始 Transformer 里就不是可选组件。
2. 再看 BERT，理解真实模型为什么普遍采用 $4\times d_{model}$ 的 FFN 宽度。
3. 然后看 Geva 等人的工作，理解 FFN 为什么不只是“附加 MLP”。
4. 最后看 Pre-LN / Post-LN 的分析，理解残差路径为什么直接影响深层训练。

如果只选一个机制论文切入，优先看 Geva 等人的工作，因为它最直接回答“FFN 到底在干什么”。如果只选一个工程对照对象，优先看 BERT-Base，因为它的维度配置和模块顺序已经成为许多 Encoder 设计的事实参考。

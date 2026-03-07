## 核心结论

激活函数的作用，是在神经网络里引入非线性。非线性可以理解为“模型不再只是做一层层线性加权”，否则再深的网络也等价于一次线性变换，表达能力会被锁死。

Sigmoid、ReLU、GELU、SwiGLU 分别对应四种不同机制：

| 激活函数 | 定义 | 输出特性 | 梯度特性 | 常见位置 |
|---|---|---|---|---|
| Sigmoid | $\sigma(x)=\frac{1}{1+e^{-x}}$ | 压到 $(0,1)$ | 最大导数仅 $0.25$，易饱和 | 二分类输出层、门控单元 |
| ReLU | $\mathrm{ReLU}(x)=\max(0,x)$ | 负数截断，正数直通 | 正区梯度为 1，负区为 0 | CNN、MLP 基础层 |
| GELU | $\mathrm{GELU}(x)=x\Phi(x)$ | 平滑保留小负值 | 平滑、连续，梯度更柔和 | Transformer FFN |
| SwiGLU | $\mathrm{Swish}(xW)\odot (xV)$ | 双路投影后门控 | 同时保留线性路与非线路 | 大模型 FFN |

结论先给清楚：

1. 深层网络不要把 Sigmoid 当默认隐藏层激活。它的梯度上界太低，层数一深就容易训练不动。
2. ReLU 仍然是“便宜、快、足够有效”的默认选项，尤其适合 CNN 和预算敏感场景。
3. GELU 是 Transformer 时代的主流折中方案。它比 ReLU 更平滑，通常更适合大规模语言模型和预训练模型。
4. SwiGLU 不是单个激活函数，而是“门控前馈结构”。它在大模型 FFN 中常优于普通 GELU，但代价是参数、显存和算力更高。

一个新手能马上记住的版本是：Sigmoid 像把数压进一个小盒子，ReLU 像只让正数通过，GELU 像按概率软通过，SwiGLU 则是在两路特征之间加一道可学习的门。

---

## 问题定义与边界

问题的核心不是“哪一个激活函数最先进”，而是“在当前网络深度、任务类型、硬件预算下，哪一个激活函数最不容易成为瓶颈”。

激活函数主要影响三件事：

| 影响项 | 白话解释 | 典型风险 |
|---|---|---|
| 非线性表达 | 模型能否表示复杂模式 | 过弱则模型学不出复杂关系 |
| 梯度传播 | 误差能否从输出层传回前面层 | 梯度消失或梯度断流 |
| 计算与部署成本 | 前向和反向传播是否便宜 | 训练慢、推理慢、显存吃紧 |

边界要分清：

1. 如果你讨论的是输出层，Sigmoid 很常见，因为二分类需要把输出解释成概率。
2. 如果你讨论的是隐藏层，Sigmoid 通常不是默认答案，因为它很容易进入饱和区。饱和区就是输入很大或很小时，函数输出几乎不再变化，导数接近 0。
3. 如果你讨论的是 CNN 或传统 MLP，ReLU 往往先试。
4. 如果你讨论的是 BERT、GPT 这一类 Transformer 前馈层，GELU 更常见。
5. 如果你讨论的是 LLaMA 一类大语言模型的 FFN，很多实现会用 SwiGLU 或相关 GLU 变体。

玩具例子先看一个 10 层网络。假设每层都用 Sigmoid，而且每层传回去的梯度都最多乘上 $0.25$，那么输入侧接收到的梯度上界近似为：

$$
0.25^{10}\approx 9.54\times 10^{-7}
$$

这已经接近 $10^{-6}$。白话说，前几层几乎听不到损失函数的“训练指令”。

真实工程例子更直观：做一个图像分类小模型时，如果隐藏层全部改成 Sigmoid，你常会看到训练 loss 下降很慢；换成 ReLU 后，前几轮就能明显收敛。做 Transformer 时，如果把 FFN 的 GELU 粗暴换成 ReLU，模型未必完全失效，但通常会损失一部分稳定性和最终精度，因为 FFN 的连续门控能力变弱了。

---

## 核心机制与推导

### 1. Sigmoid：压缩强，但梯度天花板低

Sigmoid 定义为：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

导数是：

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

因为 $\sigma(x)\in(0,1)$，所以 $\sigma'(x)$ 的最大值出现在 $\sigma(x)=0.5$，也就是 $x=0$ 附近，此时：

$$
\sigma'(0)=0.25
$$

这意味着即使在最理想的位置，它每层最多也只保留四分之一梯度。只要输入远离 0，导数会更小，进入饱和区后几乎等于 0。

所以 Sigmoid 的数学问题不是“不能用”，而是“深层传播太吃亏”。

### 2. ReLU：传播直接，但负区会断

ReLU 定义为：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

导数可写成分段形式：

$$
\mathrm{ReLU}'(x)=
\begin{cases}
1, & x>0 \\
0, & x<0
\end{cases}
$$

在 $x>0$ 时，梯度直接通过，不缩小；这就是它在深层网络里更容易训练的原因。

但问题也同样清楚：只要神经元长期落在负区，梯度恒为 0，这个单元就可能不再更新。这叫 Dead ReLU，白话解释就是“神经元死掉了，之后一直没反应”。

### 3. GELU：概率门控的平滑版本

GELU 定义为：

$$
\mathrm{GELU}(x)=x\Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数，白话解释是“随机变量小于当前值的概率”。

直觉上，GELU 不是像 ReLU 那样把负数硬切掉，而是按输入大小给一个连续的通过概率。输入越大，越容易通过；输入略小于 0 时，也不是直接归零，而是保留一部分信息。

工程中常用近似式：

$$
\mathrm{GELU}(x)\approx 0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right)
$$

这个近似的意义不是改变理论，而是减少直接计算 CDF 的成本。

### 4. SwiGLU：用门控替代单路激活

SwiGLU 常写成：

$$
\mathrm{SwiGLU}(x)=\mathrm{Swish}(xW)\odot (xV)
$$

再经过一个输出投影矩阵 $W_2$：

$$
y=\big(\mathrm{Swish}(xW)\odot (xV)\big)W_2
$$

其中：

$$
\mathrm{Swish}(u)=u\sigma(u)
$$

$\odot$ 表示逐元素乘法，也就是同位置相乘。

这里的关键不是“又造了一个激活函数”，而是结构发生了变化：一条支路负责产生门控，一条支路保留线性信息，然后两者相乘。这样做的好处是，模型不必把所有表达都挤在一条激活曲线里，而是通过双路分工增强 FFN 的表示能力。

如果用一句话概括这四者的推导逻辑：

- Sigmoid：通过压缩输出得到概率感，但代价是梯度上界低。
- ReLU：通过硬截断换取简单稳定的正区传播。
- GELU：把硬截断改成平滑概率门控。
- SwiGLU：进一步把“激活”升级为“门控结构”。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，只依赖标准库：

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def relu(x: float) -> float:
    return max(0.0, x)

def gelu_tanh_approx(x: float) -> float:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))

def swish(x: float) -> float:
    return x * sigmoid(x)

def swiglu_scalar(x: float, w: float, v: float) -> float:
    # 标量版玩具示例：真实模型里通常是向量和矩阵
    return swish(x * w) * (x * v)

def chained_sigmoid_grad_bound(layers: int) -> float:
    # 每层都取 Sigmoid 导数理论上界 0.25
    return 0.25 ** layers

assert abs(sigmoid(0.0) - 0.5) < 1e-12
assert relu(-3.0) == 0.0
assert relu(2.5) == 2.5
assert abs(chained_sigmoid_grad_bound(10) - (0.25 ** 10)) < 1e-18
assert gelu_tanh_approx(3.0) > gelu_tanh_approx(1.0)
assert swiglu_scalar(2.0, 1.0, 1.0) > 0.0

print("sigmoid(0) =", sigmoid(0.0))
print("10层 Sigmoid 梯度上界 =", chained_sigmoid_grad_bound(10))
print("gelu_tanh_approx(1) =", gelu_tanh_approx(1.0))
print("swiglu_scalar(2,1,1) =", swiglu_scalar(2.0, 1.0, 1.0))
```

这个代码说明了三个关键点：

1. `sigmoid(0)=0.5`，对应导数峰值点。
2. `0.25 ** 10` 很快衰减到 $10^{-6}$ 量级。
3. GELU 和 SwiGLU 都不是简单的“过零截断”，而是连续变化。

如果换成框架 API，常见形式如下：

| 框架/实现 | 写法 | 说明 |
|---|---|---|
| PyTorch Sigmoid | `torch.sigmoid(x)` | 常用于输出层或门控 |
| PyTorch ReLU | `torch.relu(x)` 或 `nn.ReLU()` | 简单高效 |
| PyTorch GELU | `nn.GELU()` | 可选精确版或近似版 |
| TensorFlow ReLU | `tf.nn.relu(x)` | 常规默认项 |
| TensorFlow GELU | `tf.nn.gelu(x)` | Transformer 常用 |
| 手写 GELU 近似 | `0.5*x*(1+tanh(...))` | 部署时常见 |

真实工程例子可以看 Transformer 的 FFN。一个标准 FFN 大致是：

$$
\mathrm{FFN}(x)=W_2(\phi(W_1x+b_1))+b_2
$$

这里 $\phi$ 可以是 ReLU 或 GELU。若换成 SwiGLU，则通常变成两路线性投影再门控相乘，计算图更复杂，但表达力更强。

---

## 工程权衡与常见坑

工程上真正重要的不是“函数曲线好不好看”，而是训练有没有稳定收益。

| 方案 | 常见坑 | 典型症状 | 规避策略 |
|---|---|---|---|
| Sigmoid | 饱和导致梯度消失 | 深层训练很慢，前层几乎不更新 | 隐藏层避免全用；只在输出层或门控处用 |
| ReLU | Dead ReLU | 一批神经元长期输出 0 | 降学习率、改初始化、尝试 Leaky ReLU |
| GELU | 计算更重 | 训练吞吐下降，部署端不友好 | 使用近似实现，评估硬件支持 |
| SwiGLU | 参数和显存增加 | FFN 更吃显存，延迟更高 | 调整中间维度，确认收益再上 |

### Sigmoid 的坑

它最典型的问题是“层层缩小”。如果一层梯度平均只有 0.1，十层后就是 $10^{-10}$。即使按最乐观的 0.25 算，十层后也只有约 $10^{-6}$。这会导致前面层几乎学不到东西。

所以 Sigmoid 更适合做输出概率，而不是做深层隐藏层默认激活。

### ReLU 的坑

ReLU 的问题不是梯度太小，而是有些神经元直接没梯度。比如一个神经元参数初始化后总让输入落在负区，再加上大学习率把它继续往负方向推，它就可能长时间输出 0。

这类情况在日志里常表现为：某些通道激活分布极度偏零，训练精度卡住，换学习率或初始化后恢复。

### GELU 的坑

GELU 理论上更平滑，但你要为这个平滑付费。它涉及 CDF 或近似计算，吞吐通常不如 ReLU。对超大批量训练和部署端推理来说，这个成本不是理论问题，而是真金白银的算力问题。

### SwiGLU 的坑

SwiGLU 常常带来更好的效果，但它的本质是“多开一条路，再做门控”。这意味着中间维度、矩阵乘法、显存访问都更重。大模型里 FFN 本来就是主要算力来源之一，改成 SwiGLU 后收益是否值得，必须看整机吞吐和最终指标。

一个实际判断规则是：

- 预算紧、小模型、上线快：优先 ReLU。
- 预训练 Transformer：优先 GELU。
- 大模型 FFN、追求效果上限：评估 SwiGLU。

---

## 替代方案与适用边界

除了本文主角，工程里还有一些常见替代品。它们不是“冷门知识”，而是用于补 Sigmoid 或 ReLU 的短板。

| 方案 | 核心特性 | 适用场景 |
|---|---|---|
| Leaky ReLU | 负区保留很小斜率 | 需要缓解 Dead ReLU 的 CNN/MLP |
| ELU | 负区平滑且非零 | 希望输出更接近零均值 |
| Swish | $x\sigma(x)$，平滑非单调 | 想要 ReLU 的效果又要更柔和 |
| GLU | 一路线性一路门控 | 序列模型、门控结构 |
| GEGLU | GELU 门控版本 | Transformer FFN 变体 |
| SwiGLU | Swish 门控版本 | 大语言模型 FFN |

可以把选择过程压缩成一个简单决策表：

| 条件 | 更合适的选择 |
|---|---|
| 二分类输出层 | Sigmoid |
| 普通 CNN、预算低 | ReLU |
| ReLU 出现大量死亡单元 | Leaky ReLU |
| Transformer 编码器/解码器 FFN | GELU |
| 大模型、追求更强 FFN 表达 | SwiGLU |

新手可以用一句实用判断：

- 想要最便宜、最稳的基础方案，用 ReLU。
- 想要更平滑的 Transformer 默认方案，用 GELU。
- 想要更强的门控表达，并能接受更高算力成本，用 SwiGLU。
- 想保留 ReLU 的简单结构但减少“死神经元”，试 Leaky ReLU。

这里还有一个边界常被忽略：激活函数不是独立决定效果的。初始化、归一化、残差连接、学习率调度，会和激活函数一起作用。比如带残差和 LayerNorm 的 Transformer，对 GELU 更友好；而没有这些稳定器的深层 MLP，光换激活函数不一定救得回来。

---

## 参考资料

1. Britannica: Sigmoid function。适合查 Sigmoid 的数学定义与基本性质。  
2. Charu C. Aggarwal 课程讲义：介绍激活函数、梯度传播和深层网络训练问题。  
3. Built In: ReLU activation function。适合看 ReLU 的工程直觉与常见使用场景。  
4. MBrenndoerfer 激活函数综述：适合横向比较 GELU、Swish、GLU 类方法。  
5. LLaMA 架构说明：可查看大模型 FFN 中 SwiGLU 的实际使用方式。  
6. AIML / Dead ReLU 相关文章：适合排查 ReLU 死亡单元的工程问题。

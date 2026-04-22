## 核心结论

Stochastic Depth 是一种用于残差网络的训练期正则化方法。正则化是指在训练时故意加入约束或噪声，减少模型只记住训练数据的风险。它的做法很直接：训练时随机跳过一部分残差块，测试时使用完整网络，并把每个残差分支按存活概率缩放。

核心公式是：

$$
\text{训练: } x_{l+1}=x_l+z_lF_l(x_l),\quad z_l\sim Bernoulli(p_l)
$$

$$
\text{测试: } x_{l+1}=x_l+p_lF_l(x_l)
$$

这里 \(x_l\) 是第 \(l\) 个残差块的输入，\(F_l(x_l)\) 是残差分支的输出，\(p_l\) 是该块在训练时被保留的概率，\(z_l\) 是一个只取 0 或 1 的随机变量。

它的核心收益有两个。第一，训练时网络的有效深度变浅，反向传播路径更短，超深网络更容易优化。第二，每次训练看到的都是不同深度的子网络，相当于在许多子网络之间做隐式集成，从而改善泛化。

一个新手版本可以这样理解：每个残差块都有一个“临时开关”。训练时某些开关关闭，但主干 identity shortcut 始终保留；推理时所有开关打开，不过残差分支按训练时的平均工作强度进行缩放。

---

## 问题定义与边界

Stochastic Depth 主要解决两个问题：极深残差网络训练困难，以及深层模型过拟合。它不是为了替代所有正则化方法，也不是部署期压缩技术。

残差网络是带有 shortcut 的神经网络结构。shortcut 是一条把输入直接加到输出上的路径，让网络学习“在原输入基础上修改多少”，而不是每层都重新生成完整表示。一个标准残差块通常写成：

$$
x_{l+1}=x_l+F_l(x_l)
$$

Stochastic Depth 只作用在 \(F_l(x_l)\) 这条残差分支上，不删除 \(x_l\) 这条 identity shortcut。也就是说，网络主体仍然连通，随机性来自“本次前向传播中这个残差分支是否参与计算”。

它和 Dropout 容易混淆。Dropout 是随机丢弃神经元或特征通道；Stochastic Depth 是随机丢弃整个残差分支。前者改变一层内部的信息，后者改变网络本次训练经过的有效深度。

| 方法 | 随机对象 | 训练/测试行为 | 是否保留主干 |
|---|---|---|---|
| Dropout | 神经元或通道 | 测试时缩放激活 | 是 |
| Stochastic Depth | 残差块 | 测试时缩放残差分支 | 是 |
| Layer Pruning | 固定删除层 | 结构永久改变 | 视情况而定 |
| Early Exit | 早退分支 | 推理阶段动态停止 | 是 |

边界要说清楚：它适合残差结构，尤其适合很深的网络；如果模型本身不深，或者没有清晰的残差分支，收益可能有限。

---

## 核心机制与推导

每个残差块引入一个 Bernoulli 门控变量。Bernoulli 分布是只产生 0 或 1 的随机分布，常用于表示“发生”或“不发生”。

$$
z_l\sim Bernoulli(p_l)
$$

训练时：

$$
x_{l+1}=x_l+z_lF_l(x_l)
$$

当 \(z_l=1\) 时，残差分支正常参与计算；当 \(z_l=0\) 时，该残差分支被跳过，输出就是 \(x_l\)。

测试时不再随机采样，而是用期望行为替代随机行为：

$$
E[z_lF_l(x_l)] = p_lF_l(x_l)
$$

所以测试公式变成：

$$
x_{l+1}=x_l+p_lF_l(x_l)
$$

论文中常用线性衰减的存活概率：

$$
p_l=1-\frac{l-1}{L-1}(1-p_{\min})
$$

其中 \(L\) 是残差块总数，\(p_{\min}\) 是最后一个残差块的存活概率。若 \(p_{\min}=0.5\)，第一块的 \(p_1=1\)，最后一块的 \(p_L=0.5\)。越靠后的层越容易被跳过。

玩具例子如下。设 \(L=3\)，存活概率为 \(p=[1,0.75,0.5]\)，输入 \(x_0=10\)，三个残差分支输出分别为 \(F_1=2\)、\(F_2=4\)、\(F_3=6\)。

训练时如果采样到 \(z=[1,0,1]\)，则：

$$
x_3=10+2+0+6=18
$$

测试时不用随机采样，而是计算期望路径：

$$
x_3=10+1\times2+0.75\times4+0.5\times6=17
$$

这两个数不要求每次相等。训练阶段是随机样本，测试阶段是平均效果。它们在统计意义上对齐。

为什么它能帮助训练更深的网络？因为一次训练中只有部分残差块参与，反向传播不必总是穿过完整深度。若某次采样只保留了 \(k\) 个残差块，那么梯度的有效路径约从 \(L\) 变成 \(k\)。路径变短后，梯度更容易到达前面的层，优化难度降低。

真实工程例子是训练超深 ResNet 视觉骨干。原论文报告了 110 层到 1200+ 层残差网络的实验结果，作者实现中也说明 1202 层网络可以通过该方法训练。在这种场景里，固定全深度训练会更慢，也更容易优化不稳定；Stochastic Depth 通过随机浅化训练路径降低了训练成本，同时保留测试时的完整深度。

---

## 代码实现

实现重点不是“随机删层”，而是“只对残差分支做门控”。训练和测试的缩放规则也必须一致。

最小伪代码是：

```python
def stochastic_depth_residual(x, F, p, training):
    if training:
        z = bernoulli_sample(p)
        return x + z * F(x)
    else:
        return x + p * F(x)
```

下面是一个可运行的 Python 玩具实现。为了让代码不依赖深度学习框架，这里把 \(F(x)\) 简化为固定残差值。

```python
import random

def survival_prob(l, L, p_min=0.5):
    if L == 1:
        return 1.0
    return 1.0 - ((l - 1) / (L - 1)) * (1.0 - p_min)

def stochastic_depth_forward(x0, residuals, training, seed=None, p_min=0.5):
    if seed is not None:
        random.seed(seed)

    x = x0
    L = len(residuals)
    gates = []

    for i, r in enumerate(residuals, start=1):
        p = survival_prob(i, L, p_min)

        if training:
            z = 1 if random.random() < p else 0
            gates.append(z)
            x = x + z * r
        else:
            x = x + p * r

    return x, gates

# 玩具例子：L=3, p=[1, 0.75, 0.5]
assert survival_prob(1, 3) == 1.0
assert survival_prob(2, 3) == 0.75
assert survival_prob(3, 3) == 0.5

test_out, _ = stochastic_depth_forward(10, [2, 4, 6], training=False)
assert test_out == 17.0

train_out, gates = stochastic_depth_forward(10, [2, 4, 6], training=True, seed=1)
assert gates[0] == 1
assert train_out in {12, 16, 18, 22}
```

实际框架里接口命名不完全一致，必须先确认参数语义：

| 参数名 | 含义 | 常见坑 |
|---|---|---|
| `p` / `survival` | 块被保留的概率 | 和 drop rate 弄反 |
| `drop_rate` | 块被丢弃的概率，等于 \(1-p\) | 传参时方向相反 |
| `training` | 是否处于训练模式 | 推理时忘记关闭随机采样 |
| `per-batch` | 一个 batch 共用一个门控值 | 复现论文时粒度不一致 |
| `per-sample` | batch 内每个样本单独采样 | 和论文设定可能不同 |

PyTorch 或 TensorFlow 风格的伪接口通常类似：

```python
# 伪接口示意：注意这里的 p 可能是 drop probability，也可能是 survival probability
out = stochastic_depth(x, p=0.2, mode="row", training=True)
```

工程中不能只看参数名，要看文档定义。有些库的 `p=0.2` 表示丢弃概率，有些实现可能把它解释为保留概率，两者正好相反。

---

## 工程权衡与常见坑

Stochastic Depth 能提升深层网络的可训练性，但不是越激进越好。存活概率过低会导致后层长期缺少训练，特征表达变弱。对新手来说，先从线性衰减到 \(p_{\min}=0.5\) 这种保守设置开始，更容易得到稳定结果。

最常见的错误是把整个残差块都随机删掉，连 identity shortcut 也断掉。这样网络不再保持残差结构，训练行为会变成另一种随机断图，优化稳定性通常会变差。正确做法是：shortcut 一直保留，只随机控制残差分支。

另一个高频问题是训练和测试缩放不一致。训练时用了 \(z_lF_l(x_l)\)，测试时就应该使用 \(p_lF_l(x_l)\) 对齐期望。如果测试时直接使用完整残差分支 \(F_l(x_l)\)，输出分布会比训练时更强，可能造成精度下降。

| 常见坑 | 正确做法 |
|---|---|
| 把 drop rate 和 survival probability 混淆 | 先确认库里 `p` 的定义 |
| 测试时忘记按 \(p_l\) 缩放残差 | 推理阶段使用期望路径 |
| 误删整个残差块的 shortcut | 只门控残差分支，保留 identity |
| 采样从 per-batch 改成 per-sample | 复现论文前先确认原实现粒度 |
| 最后几层 \(p_l\) 太低 | 从 \(p_{\min}=0.5\) 起步更稳妥 |
| 在很浅的网络上强行使用 | 先确认过拟合或优化困难是否真实存在 |

工程上还要注意 BatchNorm。BatchNorm 是根据 batch 统计量归一化特征的层。如果某个残差分支经常被跳过，它内部的统计量更新也会变少。通常成熟框架和主流模型实现已经处理了这些细节，但自写网络时要确认训练模式、归一化层、随机门控三者行为一致。

---

## 替代方案与适用边界

Stochastic Depth 适合很深的残差网络，尤其是视觉骨干、超深 ResNet、ConvNeXt、Vision Transformer 中带残差连接的模块。虽然最初论文讨论的是 ResNet，但后续很多架构也采用类似思想，常叫 drop path。Drop path 是随机丢弃一条计算路径的通用说法，Stochastic Depth 可以看作它在残差深度上的典型形式。

如果网络不深，或者主要问题不是过拟合和深层优化困难，它不一定是首选。如果目标是降低推理成本，它也不是直接答案，因为测试时通常仍使用完整网络。它减少的是训练期有效计算，并改善泛化，不等于部署期固定剪枝。

| 方案 | 适合场景 | 主要收益 |
|---|---|---|
| Stochastic Depth | 超深残差网络训练 | 更稳、更易训、泛化更好 |
| Dropout | 通用过拟合抑制 | 简单、通用 |
| Layer Pruning | 需要压缩推理成本 | 直接减小模型 |
| Knowledge Distillation | 想保留精度并压缩小模型 | 学生模型更轻 |
| Early Exit | 想减少平均推理延迟 | 动态停止 |
| Weight Decay | 大多数神经网络训练 | 抑制权重过大 |

决策可以简化为三句话。训练 1000+ 层 ResNet，或者很深的残差视觉模型，可以考虑 Stochastic Depth。想让推理模型更小、更快，优先看剪枝、蒸馏或更小主干。只是普通浅层网络过拟合，先用数据增强、weight decay、Dropout 等更通用手段。

---

## 参考资料

1. [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

2. [Stochastic Depth 作者 GitHub 仓库](https://github.com/yueatsprograms/Stochastic_Depth)

3. [TensorFlow Addons: tfa.layers.StochasticDepth](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth)

4. [Torchvision: torchvision.ops.stochastic_depth](https://pytorch.org/vision/stable/generated/torchvision.ops.stochastic_depth.html)

如果只想确认实现细节，优先看框架文档；如果想确认方法出处、公式和实验设定，优先看论文和作者仓库。

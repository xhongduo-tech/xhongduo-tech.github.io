## 核心结论

Dropout 的直接定义是：训练时以概率 $p$ 随机把一部分神经元输出置零，保留的部分再乘上 $\frac{1}{1-p}$，使整体期望不变。这里的“期望”就是大量重复采样后的平均输出。最常见的写法是

$$
z_i \sim \mathrm{Bernoulli}(1-p), \qquad
y_j=\frac{1}{1-p}\sum_i w_{ji} z_i x_i
$$

其中 Bernoulli 分布可以理解为“抛一次只会得到 0 或 1 的硬币”。

它首先是正则化方法。正则化的白话解释是：通过主动制造训练扰动，减少模型死记硬背训练集。更进一步，Dropout 还可以从贝叶斯视角解释为一种变分推理近似。变分推理的白话解释是：原本精确求后验分布太难，于是用一个更容易计算的分布去逼近它。Gal 和 Ghahramani 给出的结论是，带 Dropout 的神经网络可以看成在近似对深度高斯过程做贝叶斯推断。

这件事的实际价值不只在“防过拟合”。如果在预测时继续保留 Dropout 并做多次前向，也就是 MC Dropout，那么模型不只给一个点预测，还能给一个预测分布近似：

$$
q(y^*|x^*) \approx \frac{1}{T}\sum_{t=1}^{T} f_{W_t}(x^*)
$$

这里 $T$ 是采样次数，$W_t$ 表示第 $t$ 次随机 mask 下的有效参数。它的工程含义很直接：均值可当预测结果，方差可当不确定性。

一个新手版直觉可以这样记：训练时把每层的输出“随机关灯”一部分，剩下的灯统一调亮一点；推理时不再随机，而是直接用平均亮度工作。普通 Dropout 解决泛化问题，MC Dropout 额外解决“模型对自己有多确定”的问题。

---

## 问题定义与边界

Dropout 解决的问题不是“让网络更强”，而是“让网络在训练时不要过度依赖少数固定通路”。所谓通路，就是从输入到输出的一条激活路径。若某些神经元总是一起出现，模型就容易形成脆弱共适应。共适应的白话解释是：多个特征彼此捆绑，单独拿掉任意一个就失效。Dropout 通过随机失活打断这种捆绑。

但它有明确边界，主要有两层。

第一层边界是训练和推理必须一致。训练阶段使用随机采样，推理阶段如果不做期望补偿，输出分布就会漂移。常见做法有两种：

| 方案 | 训练阶段 | 推理阶段 | 结果 |
|---|---|---|---|
| 标准 Dropout | 随机置零，不缩放 | 权重乘 $(1-p)$ 或输出乘 $(1-p)$ | 训练推理期望一致 |
| Inverted Dropout | 随机置零，并乘 $\frac{1}{1-p}$ | 不再缩放 | 现代框架默认采用 |
| MC Dropout | 随机置零，并乘 $\frac{1}{1-p}$ | 仍然保留随机采样，多次前向 | 得到预测均值和不确定性 |

现代 Keras / TensorFlow 默认是 inverted dropout，也就是训练时补偿，推理时直接关掉 Dropout 层。

第二层边界是贝叶斯解释并不等于“严格精确的贝叶斯后验”。它是近似。更准确地说，是对某个变分分布 $q(W)$ 的近似优化。这个结论足够支持工程使用，但不能把 MC Dropout 当成任何任务上的最优不确定性估计器。

一个玩具例子可以说明边界。假设你拟合一条曲线，如果训练时某几个隐藏单元总被强依赖，模型可能在训练点附近非常贴合，但换一个输入就大幅波动。Dropout 相当于每一步训练都随机删掉部分连接，逼迫网络学到更稳健的表示；而推理时再回到“平均网络”。这就是它防过拟合的边界：它控制的是表示冗余和鲁棒性，不是直接提升模型容量。

普通 Dropout 和 MC Dropout 的差别也要分清：

| 视角 | 普通 Dropout | MC Dropout |
|---|---|---|
| 目标 | 提升泛化 | 提升泛化并估计不确定性 |
| 推理是否采样 | 否 | 是 |
| 输出 | 单次点预测 | 均值、方差、置信区间近似 |
| 贝叶斯解释 | 隐含存在 | 显式利用 |

---

## 核心机制与推导

### 1. 训练阶段到底做了什么

设某层输入为 $x$，权重为 $W$，mask 为 $z$。对第 $i$ 个输入单元：

$$
z_i \sim \mathrm{Bernoulli}(1-p)
$$

意思是：以概率 $1-p$ 保留，以概率 $p$ 置零。于是单个输出单元 $j$ 的线性部分可写为：

$$
y_j=\frac{1}{1-p}\sum_i w_{ji} z_i x_i
$$

为什么要乘 $\frac{1}{1-p}$？因为保留下来的单元变少了，如果不放大，输出期望会整体变小。这个缩放叫期望校正。

我们可以直接算期望：

$$
\mathbb{E}[z_i] = 1-p
$$

所以

$$
\mathbb{E}[y_j]
= \frac{1}{1-p}\sum_i w_{ji} x_i \mathbb{E}[z_i]
= \sum_i w_{ji} x_i
$$

这说明使用 inverted dropout 时，训练阶段每一步虽然是随机子网络，但从期望上看，和原始网络的输出尺度一致。

### 2. 玩具例子

设单输入 $x=1$，权重 $w=2$，dropout rate 为 $p=0.5$。

训练时有两种可能：

- mask 为 0，输出是 $0$
- mask 为 1，输出是 $\frac{2\times 1}{1-0.5}=4$

所以训练输出集合是 $\{0,4\}$，期望为：

$$
\frac{0+4}{2}=2
$$

而原始无 Dropout 的输出本来就是 $2$。这就是“随机训练，期望不变”。

很多教材会写“推理时把权重乘以 $(1-p)$”。这对应另一种等价实现：训练时不做 $\frac{1}{1-p}$ 放大，推理时再缩权重。现代框架通常反过来做，也就是训练时放大，推理时不动参数。两者数学上是等价的，只是工程实现不同。

### 3. 为什么它像模型集成

模型集成的白话解释是：把多个模型结果平均，通常比单个模型更稳。Dropout 的关键洞见是，训练时每个 mask 都对应一个“共享参数的子网络”。你并没有真的训练指数级数量的独立模型，但在优化过程中，网络相当于学习了大量子结构的共享表示。推理时取期望，本质上接近这些子网络的平均效果。

可以把机制压缩成下表：

| 步骤 | 数学对象 | 作用 |
|---|---|---|
| 采样 mask | $z_i \sim \mathrm{Bernoulli}(1-p)$ | 随机删掉部分神经元 |
| 保留期望 | $\frac{1}{1-p}$ | 保持激活尺度稳定 |
| 多轮训练 | 不同 mask 的子网络 | 抑制共适应 |
| 推理聚合 | 期望或多次采样平均 | 近似集成效果 |

### 4. 贝叶斯解释

Gal 和 Ghahramani 的核心观点是：在神经网络中使用 Dropout，可以解释为在某个变分分布下近似优化后验。后验的白话解释是：看完数据后，对参数还剩下多少不确定性的分布。精确算后验很难，于是引入近似分布 $q(W)$。

在这个视角里，Dropout mask 相当于对权重或激活引入随机性，训练过程相当于最小化一个与变分下界相关的目标。结论不是“Dropout 就是贝叶斯网络”，而是“Dropout 提供了一个便宜的贝叶斯近似入口”。

因此，预测阶段如果继续采样不同 mask，就能近似参数后验诱导出的预测分布：

$$
q(y^*|x^*) \approx \frac{1}{T}\sum_{t=1}^{T} f_{W_t}(x^*)
$$

进一步可以估计均值与方差：

$$
\mu \approx \frac{1}{T}\sum_{t=1}^{T}\hat y_t,\qquad
\sigma^2 \approx \frac{1}{T}\sum_{t=1}^{T}(\hat y_t-\mu)^2
$$

这里的方差可以粗略理解为“模型犹豫程度”。若多次采样结果很分散，说明模型对该输入缺乏把握。

### 5. 真实工程例子

一个典型场景是风险敏感预测，例如销量预测、医疗辅助评分、异常检测阈值判断。假设你做电商需求预测，模型对常见商品给出稳定结果，但对长尾新商品的数据很少。若只输出一个点预测，系统无法区分“预测值本身低”还是“模型其实很不确定”。MC Dropout 可以在不改动太多训练流程的前提下，为每个样本增加一个方差指标。工程上常见策略是：

- 均值作为主预测
- 方差超过阈值时触发人工审核或保守策略
- 把高不确定样本优先送去补数据或主动标注

这比单纯依赖 softmax 最大概率更稳，因为 softmax 往往在分布外样本上仍可能过度自信。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，验证“训练时随机失活但期望不变”。

```python
import random

def dropout_forward(x, w, p, trials=20000):
    total = 0.0
    for _ in range(trials):
        z = 1 if random.random() > p else 0
        y = (w * z * x) / (1 - p)
        total += y
    return total / trials

x = 1.0
w = 2.0
p = 0.5

estimated_expectation = dropout_forward(x, w, p)
baseline = w * x

assert abs(estimated_expectation - baseline) < 0.1
print("estimated:", estimated_expectation, "baseline:", baseline)
```

这个例子里，`assert` 检查的是大量随机采样后的平均输出是否接近无 Dropout 时的输出。

如果把玩具例子写成可枚举版本，会更直观：

```python
def toy_outputs(x=1.0, w=2.0, p=0.5):
    y_drop = 0.0
    y_keep = (w * x) / (1 - p)
    expected = 0.5 * y_drop + 0.5 * y_keep
    assert y_drop == 0.0
    assert y_keep == 4.0
    assert expected == 2.0
    return y_drop, y_keep, expected

print(toy_outputs())
```

实际工程里，Keras 的 Dropout 写法通常是：

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
```

这里 `Dropout(0.5)` 的含义是训练时丢弃 50% 的输入单元。Keras 会根据 `training` 标志自动决定是否启用随机失活。

如果你要做 MC Dropout，关键不是换层，而是在预测时显式保留 `training=True`：

```python
import tensorflow as tf

def mc_dropout_predict(model, x, T=30):
    preds = []
    for _ in range(T):
        y = model(x, training=True)  # 保持 dropout 生效
        preds.append(y)
    stacked = tf.stack(preds, axis=0)
    mean = tf.reduce_mean(stacked, axis=0)
    var = tf.math.reduce_variance(stacked, axis=0)
    return mean, var
```

训练和推理的行为差异可以压缩成一张表：

| 场景 | 调用方式 | Dropout 是否采样 | 常见用途 |
|---|---|---|---|
| 训练 | `model(x, training=True)` | 是 | 参数更新 |
| 普通推理 | `model(x, training=False)` | 否 | 点预测 |
| MC Dropout 推理 | 多次 `model(x, training=True)` | 是 | 不确定性估计 |

如果是卷积网络，还要考虑 `SpatialDropout`。它不是按单个像素或单个激活置零，而是按整张特征图置零。特征图的白话解释是：卷积层输出的一整个通道。这样做是因为卷积层相邻位置高度相关，随机删掉几个单点通常破坏不了冗余，删整张通道更有正则化意义。

---

## 工程权衡与常见坑

Dropout 看起来简单，但真正用在工程里，经常踩的是“位置、概率、推理行为”三个坑。

| 常见坑 | 现象 | 原因 | 解决方案 |
|---|---|---|---|
| 在卷积前几层直接用普通 Dropout | 效果弱，甚至伤害特征提取 | 相邻像素强相关，单点失活意义有限 | 优先用 `SpatialDropout` 或把 Dropout 放到高层 |
| $p$ 设太大 | 训练和验证都差，明显欠拟合 | 信息被删太多，偏差增大 | 从 0.1, 0.2, 0.3, 0.5 网格搜索 |
| 推理时忘记区分 `training` | 线上结果不稳定或与离线不一致 | 训练/推理行为混淆 | 普通预测必须 `training=False`，MC Dropout 才保留 `True` |
| 与 BatchNorm 混用不当 | 收敛慢，统计量不稳 | 两者都在改变激活分布 | 通常先按现有架构验证，不要机械叠加大 dropout |
| 所有层都加同样的 Dropout | 效果不可控 | 不同层冗余程度不同 | 只在高层全连接或任务头优先尝试 |

### 1. CNN 里的坑

新手常见误区是把普通 Dropout 直接加在卷积开头，以为“越早正则化越好”。这往往不成立。原因是图像中邻近像素和相邻卷积响应高度相关，你删掉一个位置，周围位置还能补回来，因此正则化信号很弱；但如果删得太多，又会破坏低层局部结构。于是更常见的方案是：

- 低层卷积少用或不用普通 Dropout
- 中高层用 `SpatialDropout`
- 分类头或全连接层再用标准 Dropout

可以把它理解成：在卷积早期，整张特征图是更合理的“失活单位”。

### 2. $p$ 怎么选

$p$ 不是越大越强。它控制的是噪声强度。噪声太小，正则化不足；噪声太大，模型学不到稳定信号。经验上：

| 场景 | 常见 $p$ 范围 |
|---|---|
| 输入层附近 | 0.05 到 0.2 |
| 全连接隐藏层 | 0.2 到 0.5 |
| 小数据集且模型大 | 可适当增大 |
| 已有强数据增强/强正则 | 可适当减小 |

如果任务需要稳定的不确定性估计，仅靠手工调 $p$ 往往不够。Concrete Dropout 试图解决这个问题。它通过连续松弛让 dropout rate 可学习，再结合 KL 正则共同优化。连续松弛的白话解释是：把原本不可导的 0/1 采样，改成可微近似，便于反向传播。

### 3. 训练与推理一致性原则

最容易被忽略的一条原则是：你必须清楚自己要的是“点预测”还是“分布预测”。

- 如果要稳定线上推理，用 `training=False`
- 如果要估不确定性，用多次 `training=True`
- 不要把单次 `training=True` 当正常线上输出，它只是一次随机样本

这条规则听起来简单，但在实际服务化时经常被包装层、导出图或推理脚本绕丢。

---

## 替代方案与适用边界

Dropout 不是唯一的随机正则化方法。不同变体的主要区别，在于“随机失活”的粒度不同。

| 方法 | 随机作用粒度 | 优点 | 局限 | 适用场景 |
|---|---|---|---|---|
| Dropout | 激活值/神经元 | 简单、通用、框架支持完善 | 对卷积低层常不够理想 | MLP、分类头、通用深度网络 |
| DropConnect | 权重 | 更细粒度，约束连接本身 | 计算和实现更复杂 | 小模型、研究型实验 |
| SpatialDropout | 整个特征图/通道 | 适合卷积相关性强的场景 | 不适合普通 MLP | CNN、时序卷积 |
| Concrete Dropout | 可学习的 dropout rate | 能自动调 $p$，利于不确定性建模 | 训练更复杂，需要额外正则项 | 数据稀缺、需可靠置信度的任务 |

### 1. DropConnect

DropConnect 可以理解为“不是随机关神经元，而是随机关权重”。这意味着某个神经元仍存在，但它连向上一层的部分连接被置零。它比 Dropout 更细粒度，但工程实现更复杂，也不如普通 Dropout 那样被主流框架广泛原生支持。

新手版直觉是：Dropout 像“随机让一些灯泡熄灭”，DropConnect 像“灯泡还亮着，但部分电线被随机断开”。

### 2. SpatialDropout

当数据具有强空间结构时，SpatialDropout 往往比普通 Dropout 更合理。图像、语音谱图、部分时序卷积任务都属于这一类。它的本质不是改目标函数，而是把“随机单位”从单个激活改成整张通道。

### 3. Concrete Dropout

Concrete Dropout 适合需要更稳健不确定性估计、又不想手动反复调参的任务。它让网络自己学会该关多少。代价是目标函数更复杂，训练超参数更多，调试成本也更高。因此它不一定适合简单分类任务，但在小样本回归、主动学习、风险敏感决策里更有价值。

### 4. 适用边界总结

如果只是常规监督学习中的过拟合问题，普通 Dropout 足够。

如果是卷积网络且 Dropout 效果不稳定，先试 SpatialDropout。

如果你需要预测不确定性，又不想改成完整贝叶斯神经网络，MC Dropout 是性价比很高的折中。

如果你连 dropout rate 也想一起学习，并愿意接受更高训练复杂度，再考虑 Concrete Dropout。

---

## 参考资料

1. Gal, Y. and Ghahramani, Z. 2016. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML 2016.  
   说明：提出 Dropout 的贝叶斯解释，并给出 MC Dropout 的不确定性估计方法。  
   链接：https://proceedings.mlr.press/v48/gal16.html

2. Wan, L. et al. 2013. *Regularization of Neural Networks using DropConnect*. ICML 2013.  
   说明：提出 DropConnect，在权重级别做随机失活。  
   链接：https://proceedings.mlr.press/v28/wan13.html

3. Keras Documentation. *Dropout layer*.  
   说明：给出 Keras 中 Dropout 的标准接口与训练/推理行为。  
   链接：https://keras.io/api/layers/regularization_layers/dropout/

4. TensorFlow Documentation. *tf.keras.layers.Dropout*.  
   说明：解释 `training=True/False` 下 Dropout 的执行方式。  
   链接：https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

5. TensorFlow Documentation. *tf.keras.layers.SpatialDropout1D*（以及对应 2D/3D 变体）。  
   说明：说明按特征图或通道丢弃的空间型 Dropout。  
   链接：https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D

6. Gal, Y., Hron, J., and Kendall, A. 2017. *Concrete Dropout*.  
   说明：通过连续松弛让 dropout rate 可学习。  
   链接：https://arxiv.org/abs/1705.07832

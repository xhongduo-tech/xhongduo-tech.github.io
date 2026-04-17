## 核心结论

BatchNorm，中文常写作“批归一化”，是指：在训练阶段，针对一个 mini-batch 内的每个特征维度或通道，先做标准化，再用可学习参数恢复合适的尺度与偏移。这里的“标准化”就是把一组数变成“均值接近 0、方差接近 1”的形式，常见写法是 z-score。

它解决的核心问题通常表述为“内部协变量偏移”。这个词的白话解释是：前面层的参数一更新，后面层看到的输入分布就变了，后面层只能不断重新适应。BatchNorm 的作用不是让输入永远不变，而是把这种漂移控制在更稳定的范围内，让优化更容易。

最重要的工程结论只有三条：

| 结论 | 训练阶段 | 推理阶段 |
|---|---|---|
| 用什么统计量 | 当前 mini-batch 的均值、方差 | 训练过程中累积的 running mean / running variance |
| 是否有可学习参数 | 有，$\gamma$ 和 $\beta$ | 有，仍然使用训练好的 $\gamma$ 和 $\beta$ |
| 主要收益 | 提高训练稳定性，允许更大学习率，减轻初始化敏感性 | 保持输出分布稳定，避免因 batch 大小变化导致结果漂移 |

一个最小数值例子可以直接说明它在做什么。设输入是 $[2,4]$，均值 $\mu=3$，方差 $\sigma^2=1$，$\varepsilon=10^{-5}$，缩放参数 $\gamma=2$，偏移参数 $\beta=1$。标准化后约为 $[-1,1]$，再缩放和平移得到：

$$
y = \gamma \hat{x} + \beta = [2 \times (-1) + 1,\ 2 \times 1 + 1] = [-1, 3]
$$

这说明 BatchNorm 不是把特征“压平”后就结束，它还保留了重新定义输出尺度的能力。$\gamma$ 控制“放大还是缩小”，$\beta$ 控制“整体平移”。

---

## 问题定义与边界

先把问题讲清楚。所谓“内部协变量偏移”，这里可以理解为：网络中某一层的输入分布，会随着前面层参数更新而不断变化。分布就是数据的大致统计形态，比如平均值、波动范围、偏斜程度。对后续层来说，这相当于训练目标一直在轻微移动。

玩具例子如下。假设某层在第 100 次迭代时，输入激活均值大约是 0.3；到了第 500 次迭代，因为前一层权重变了，均值变成了 0.7。后面的线性层原本学会了“如何处理均值 0.3 附近的数据”，现在它又得重新适应均值 0.7。这个过程会拖慢训练，严重时还会让梯度传播更不稳定。

下面用示意统计说明有无 BatchNorm 时的差异：

| 训练轮次 | 无 BatchNorm：某通道均值 | 无 BatchNorm：某通道方差 | 有 BatchNorm：输出均值 | 有 BatchNorm：输出方差 |
|---|---:|---:|---:|---:|
| step 100 | 0.31 | 0.18 | 0.00 附近 | 1.00 附近 |
| step 500 | 0.68 | 0.42 | 0.00 附近 | 1.00 附近 |
| step 1000 | 0.52 | 0.27 | 0.00 附近 | 1.00 附近 |

这个表不是说真实训练里永远精确等于 0 和 1，而是说明 BatchNorm 会把每一批数据重新拉回一个较稳定的统计范围。

边界也必须说清楚。BatchNorm 不是“任何场景都该默认加”的万能层，它最适合以下情况：

| 适合 | 不适合或效果受限 |
|---|---|
| CNN 中按通道处理激活 | batch 很小，统计量噪声大 |
| 训练时 batch 相对稳定 | 序列模型中时间步差异大 |
| 可以明确区分 train / eval 模式 | 在线推理时 batch 大小不断变化 |
| 需要更稳的优化过程 | Transformer 中通常更偏向 LayerNorm |

因此，题目里说“RNN/Transformer 中不适用”，更准确的表达是：不是主流做法，往往不如 LayerNorm 合适。原因不在于公式不能写，而在于统计维度和训练/推理一致性处理更麻烦。

---

## 核心机制与推导

BatchNorm 的核心流程可以概括为三步：统计、标准化、缩放偏移。

设一个 mini-batch 中某个通道有 $m$ 个值，记为 $x_1, x_2, \dots, x_m$。训练阶段先计算该批次的均值和方差：

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

然后做标准化：

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}
$$

这里的 $\varepsilon$ 是一个很小的正数，白话解释是“防止分母接近 0 导致数值不稳定”。

最后再做可学习的线性变换：

$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\gamma$ 是缩放参数，$\beta$ 是平移参数。它们是模型训练出来的，不是手工指定的常数。

把上面的机制放到一个完整例子里更容易理解。设某一批输入为：

$$
x = [2, 4]
$$

则：

- $\mu_B = (2+4)/2 = 3$
- $\sigma_B^2 = ((2-3)^2 + (4-3)^2)/2 = 1$

所以标准化结果约为：

$$
\hat{x} = \left[\frac{-1}{\sqrt{1+10^{-5}}}, \frac{1}{\sqrt{1+10^{-5}}}\right] \approx [-1, 1]
$$

若 $\gamma=2,\ \beta=1$，则输出为：

$$
y \approx [-1, 3]
$$

可以把流程写成一个表：

| 步骤 | 公式 | 作用 |
|---|---|---|
| 统计 | $\mu_B,\ \sigma_B^2$ | 估计当前 batch 的中心和波动 |
| 标准化 | $\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}$ | 把特征拉到统一尺度 |
| 缩放偏移 | $y_i=\gamma \hat{x}_i+\beta$ | 恢复模型需要的表达能力 |

推理阶段是另一个关键点。训练时我们每次只看到一个 batch，batch 统计量会抖动。如果推理时还继续使用“当前 batch 的均值和方差”，那么同一个样本在不同批次里可能得到不同结果。为了解决这个问题，训练过程中会维护运行统计量：

$$
\text{running\_mean} \leftarrow (1-\alpha)\cdot \text{running\_mean} + \alpha \cdot \mu_B
$$

$$
\text{running\_var} \leftarrow (1-\alpha)\cdot \text{running\_var} + \alpha \cdot \sigma_B^2
$$

这里的 $\alpha$ 可以理解为更新速度。推理时不再看当前 batch，而是固定使用 running mean 和 running variance。这样训练和部署就能衔接起来。

真实工程例子是卷积网络中的图像分类。对一个形状为 `(N, C, H, W)` 的张量，BatchNorm2d 通常按通道 `C` 统计，把 `N*H*W` 上的数据看作这个通道的一批样本。这样做的效果是：第 17 个卷积通道的激活不会因为前一层参数轻微更新，就在不同训练步之间大幅漂移。Intel OneDNN 这类推理库在部署时也会围绕这一套 running 统计做优化。

---

## 代码实现

先给一个可运行的 Python 玩具实现。它只实现单通道的一维 BatchNorm，但逻辑是完整的：训练阶段用 batch 统计，推理阶段用 running 统计，并带有 `assert` 验证。

```python
import math

class SimpleBatchNorm1D:
    def __init__(self, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = 1.0
        self.beta = 0.0
        self.running_mean = 0.0
        self.running_var = 1.0
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        n = len(x)
        assert n > 0

        if self.training:
            mean = sum(x) / n
            var = sum((xi - mean) ** 2 for xi in x) / n

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        y = [
            self.gamma * ((xi - mean) / math.sqrt(var + self.eps)) + self.beta
            for xi in x
        ]
        return y

# 玩具例子：输入 [2,4]，均值=3，方差=1，gamma=2，beta=1
bn = SimpleBatchNorm1D()
bn.gamma = 2.0
bn.beta = 1.0

out_train = bn.forward([2.0, 4.0])
assert abs(out_train[0] - (-1.0)) < 1e-3
assert abs(out_train[1] - 3.0) < 1e-3

# 训练一次后，running 统计应被更新
assert bn.running_mean != 0.0
assert bn.running_var != 1.0

# 切到推理模式，输出要使用 running 统计
bn.eval()
out_eval = bn.forward([2.0, 4.0])
assert len(out_eval) == 2
```

如果用 PyTorch，代码通常更短。关键点是：`train()` 和 `eval()` 会改变 BatchNorm 的行为。

```python
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)

model = ConvNet()
x = torch.randn(8, 3, 32, 32)

model.train()   # 用 batch 统计
y1 = model(x)

model.eval()    # 用 running 统计
with torch.no_grad():
    y2 = model(x)
```

手动实现时，逻辑通常可以写成如下伪代码：

```python
if training:
    mean = batch_mean(x)
    var = batch_var(x)
    x_hat = (x - mean) / sqrt(var + eps)
    y = gamma * x_hat + beta
    running_mean = (1 - momentum) * running_mean + momentum * mean
    running_var = (1 - momentum) * running_var + momentum * var
else:
    x_hat = (x - running_mean) / sqrt(running_var + eps)
    y = gamma * x_hat + beta
```

这里最容易写错的不是公式，而是模式切换。训练时和推理时使用的统计量不同，这是 BatchNorm 与很多普通层最大的区别。

---

## 工程权衡与常见坑

BatchNorm 的第一大问题是对 batch size 敏感。batch 太小，均值和方差估计会非常吵。所谓“吵”，白话解释是：这一批和下一批的统计量差异很大，导致归一化结果忽上忽下。

例如 batch size 只有 2：

- 第一个 batch 某通道激活是 `[0.2, 0.8]`，均值是 0.5
- 第二个 batch 某通道激活是 `[0.1, 1.1]`，均值是 0.6
- 第三个 batch 某通道激活是 `[-0.2, 1.4]`，均值又变成 0.6，但方差已经明显更大

在这么小的样本下，统计量受单个样本影响极强，训练会抖。

第二大问题是序列模型。RNN 和 Transformer 处理的是时间步或 token 序列。对于这类模型，“按 batch 统计”往往不是最自然的归一化方式。因为不同时间步、不同长度样本、不同 mask 状态下，统计分布不稳定，训练和推理也不容易保持完全一致。这也是为什么 Transformer 主流使用 LayerNorm，而不是 BatchNorm。

第三大问题是部署阶段忘记切换模式。假设训练完成后直接拿 `train()` 模式跑验证集，那么验证结果会依赖当前 batch 的组成；如果最后一个 batch 很小，还可能出现明显波动。这类 bug 非常常见，而且表面上看像“模型随机不稳定”。

下面给一个常见坑对照表：

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| batch 太小 | loss 抖动、收敛慢 | 均值方差估计噪声大 | 增大 batch，或改用 GroupNorm / LayerNorm |
| 推理忘记 `eval()` | 同一输入在不同批次结果不同 | 仍在用 batch 统计 | 推理、验证、导出模型前显式调用 `eval()` |
| 分布切换明显 | 训练好、线上差 | running 统计与真实线上分布不匹配 | 用更贴近线上分布的数据训练或重估统计 |
| 序列模型硬套 BN | 收敛不稳、实现复杂 | 归一化维度不自然 | Transformer 优先用 LayerNorm |
| 极小方差通道 | 数值不稳 | 分母过小 | 保留 $\varepsilon$，不要随意删掉 |

真实工程例子是多卡训练图像模型。单卡 batch 只有 2，总 batch 虽然是 16，但如果每张卡各自算 BatchNorm，单卡统计仍然很差。这时常用 SyncBatchNorm，把多卡上的统计同步后再做归一化。代价是通信开销变大，吞吐下降。也就是说，BatchNorm 的稳定性和系统成本往往要一起看，不能只看数学公式。

---

## 替代方案与适用边界

当 BatchNorm 不合适时，最常见的替代方案是 LayerNorm 和 GroupNorm。

LayerNorm，中文常叫“层归一化”，意思是：不跨 batch 维度统计，而是在单个样本内部，对一个 token 或一层特征做归一化。白话解释是“每个样本自己归一化，不依赖别的样本”。这让它天然适合 Transformer，因为推理时即使 batch size 从 64 变成 1，行为也不会改变。

GroupNorm，中文常叫“组归一化”，意思是：把通道分成若干组，在每个样本内部按组统计。它兼顾了一部分通道结构信息，同时不依赖 batch 维度，因此在小 batch 的视觉任务里常比 BatchNorm 更稳。

对比可以写成表：

| 方法 | 统计维度 | 是否依赖 batch size | 典型场景 | 主要优点 | 主要限制 |
|---|---|---|---|---|---|
| BatchNorm | 跨 batch、空间维，按通道统计 | 是 | CNN、较大 batch 训练 | 优化稳定，卷积场景成熟 | 小 batch 不稳，推理需 running 统计 |
| LayerNorm | 单样本内部按特征统计 | 否 | Transformer、RNN | 训练推理一致，适合序列 | 对卷积通道结构利用较弱 |
| GroupNorm | 单样本内部按通道组统计 | 否 | 小 batch 视觉任务 | 不依赖 batch，视觉上常有效 | 需要选择组数 |

Transformer 的真实工程例子最典型。输入是若干 token，每个 token 对应一个隐藏向量。若用 LayerNorm，就对单个 token 的隐藏维度做归一化；这样每个 token 的处理与当前 batch 里还有哪些样本无关，因此更稳定，也更容易部署。

还有一个部署相关边界值得单独说明：BatchNorm 在推理图中常被折叠进前一层线性算子，比如卷积或全连接。所谓“折叠”，白话解释是：把归一化和缩放偏移提前算进卷积权重与偏置中，推理时就不必再单独执行一个 BatchNorm 层。

若前一层是卷积：

$$
z = W * x + b
$$

再接 BatchNorm：

$$
y = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2+\varepsilon}} + \beta
$$

则可以改写为新的卷积：

$$
y = W' * x + b'
$$

其中

$$
W' = \frac{\gamma}{\sqrt{\sigma^2+\varepsilon}} W
$$

$$
b' = \frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}(b-\mu) + \beta
$$

这就是很多推理库和编译器做的优化，包括 OneDNN 一类高性能后端。它的前提是：使用的是固定的 running mean / variance，而不是运行时 batch 统计。因此，这种折叠只适用于推理，不适用于训练。

---

## 参考资料

- Ioffe, Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*  
  侧重点：原始论文，给出 BatchNorm 的动机、公式和训练收益。

- Wikipedia: Batch normalization  
  侧重点：概念总览，解释内部协变量偏移、训练与推理阶段统计量的差异。

- Aman.ai 关于 BatchNorm 的技术笔记  
  侧重点：面向学习者解释 running statistics、训练时 batch 统计与推理时固定统计的区别。

- TensorTonic 关于 Batch Normalization 的推导  
  侧重点：公式拆解清楚，适合对 $\mu_B$、$\sigma_B^2$、$\hat{x}$、$\gamma/\beta$ 的数值过程做核对。

- Intel OneDNN Developer Guide: Batch Normalization  
  侧重点：工程实现与部署优化，尤其是推理阶段 scale+shift、与卷积融合、性能考虑。

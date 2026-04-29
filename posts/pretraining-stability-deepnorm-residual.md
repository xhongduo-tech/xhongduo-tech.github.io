## 核心结论

DeepNorm 和 ResiDual 都在解决同一类问题：**Transformer 变得很深以后，训练容易不稳定**。这里的“不稳定”，白话讲就是 loss 早期剧烈震荡、梯度突然爆掉、或者模型明明没报错却怎么也学不动。

先给结论：

1. **DeepNorm = 深度相关残差缩放 + 配套初始化修正**。它的重点不是“把学习率调小一点”，而是直接修改每一层残差的组合方式，让主干信号在超深网络里更稳，同时把某些权重初始化缩小，避免一开始更新过猛。
2. **ResiDual = 双残差路径**。它不是简单二选一地站在 Pre-LN 或 Post-LN 一边，而是同时保留两条残差流，一条更利于梯度回传，一条更利于表达能力。
3. 如果模型只有 12 到 24 层，很多时候 **Pre-LN + warmup + 梯度裁剪** 就够了。DeepNorm 和 ResiDual 的价值，主要出现在 **100 层上下甚至更深** 的场景。
4. 从工程视角看，DeepNorm 更像“把超深训练先稳住”；ResiDual 更像“在稳定之外，尽量保住深层结构的表达优势”。

下面这张表先建立整体印象：

| 结构 | 核心公式特点 | 稳定性 | 表达能力倾向 | 适用深度 |
|---|---|---:|---:|---:|
| Post-LN | 残差后做 LN | 较弱 | 较强 | 浅层到中层 |
| Pre-LN | 先 LN 再进子层 | 较强 | 常被认为略保守 | 浅层到较深 |
| DeepNorm | 对残差主干做深度相关缩放，并改初始化 | 很强 | 保持深层可训练性 | 很深到超深 |
| ResiDual | 同时保留两条残差路径 | 很强 | 稳定性与表达兼顾 | 深层到超深 |

可以把它们浓缩成两句话：

$$
\text{DeepNorm} = \text{残差缩放} + \text{初始化修正}
$$

$$
\text{ResiDual} = \text{双残差路径}
$$

---

## 问题定义与边界

先把问题说准。深层 Transformer 的麻烦，不是“层数多”这四个字本身，而是：

1. **残差会逐层累加**。残差，白话讲就是“这一层在原输入上额外加的改动”。
2. 每层都在加一点东西，浅层时通常还可控；层数一多，累计效应可能让表示尺度漂移，或者让早期更新过大。
3. 梯度在深层网络里回传时，还会遇到路径过长、某些分支放大或缩小过头的问题。

统一记号如下：

- $x_l$：第 $l$ 层的隐藏状态，也就是第 $l$ 层输入的向量表示
- $F_l(\cdot)$：第 $l$ 层子模块的变换，可以是自注意力或 FFN
- $LN(\cdot)$：LayerNorm，白话讲就是“把一层激活的尺度整理到更稳定的范围”

一个很常见的误解是：**只要优化器够强，结构问题就能靠调参解决**。这通常不成立。因为优化器主要控制“怎么更新参数”，而 DeepNorm、ResiDual 处理的是“网络内部信号怎么流动”。

下面这张表说明边界：

| 现象 | 可能原因 | 只靠优化器能否解决 | 是否考虑结构改造 |
|---|---|---|---|
| loss 早期剧烈震荡 | 残差累计过猛、初始化过大 | 有时缓解，但不稳 | 是 |
| 梯度范数偶发爆炸 | 深层残差与参数更新耦合失控 | 常不够 | 是 |
| 模型能跑但很难扩到更深 | 路径设计不利于深层训练 | 通常不够 | 是 |
| 12 层模型训练正常 | 深度压力不大 | 往往够用 | 否，先别复杂化 |

一个“玩具例子”可以帮助理解边界。

假设每层都在主干上加一个固定偏移 $\Delta = 0.1$。那么 8 层以后累计是 $0.8$，通常问题不大；如果是 128 层，累计就是 $12.8$。真实模型当然不是简单常数相加，但它说明一件事：**深度带来的核心风险是累计效应，不是某一层单独出错**。

真实工程里也是同样逻辑。比如你训练一个 12 层文本分类模型，最先该做的是：

- 用 Pre-LN baseline
- 设合理 warmup
- 加梯度裁剪
- 检查 batch size 和学习率

只有当这些都正常，但模型一加深就明显发散，才说明你遇到的是**结构级稳定性问题**，这时 DeepNorm 或 ResiDual 才有意义。

---

## 核心机制与推导

先看三种基础形式。

**Post-LN**：

$$
x_{l+1} = LN(x_l + F_l(x_l))
$$

它的特点是：先把残差和新信息合并，再做归一化。白话讲，就是“先把旧内容和新改动糊在一起，再统一整理”。这类结构在表达上往往不差，但层数变深以后，训练稳定性比较敏感。

**Pre-LN**：

$$
x_{l+1} = x_l + F_l(LN(x_l))
$$

它的特点是：先把输入整理好，再送进子层，最后直接做残差相加。白话讲，就是“先把输入洗干净，再加工，再加回主干”。Pre-LN 的优点是梯度路径更顺，深层训练通常更稳，所以它成了很多大模型的默认起点。

**DeepNorm**：

$$
x_{l+1} = LN(\alpha x_l + F_l(x_l))
$$

这里的 $\alpha$ 不是固定常数，而是和深度 $L$ 相关。它的直觉不是“让更新更小”，而是**让主干残差更强，同时把参数初始更新幅度缩小**。这两件事必须一起发生。

以常见的 decoder-only 情况为例，论文里使用的典型系数可写成：

$$
\alpha = (2L)^{1/4}, \quad \beta = (8L)^{-1/4}
$$

其中 $\beta$ 用于某些权重矩阵的初始化缩放。白话讲，$\alpha$ 决定“旧信息保留多强”，$\beta$ 决定“新子层一开始出手多重”。

取 $L=64$：

$$
\alpha = (128)^{1/4} \approx 3.36,\quad \beta = (512)^{-1/4} \approx 0.21
$$

再做一个最小数值例子。若 $x_l=1$，$F_l(x_l)=0.2$：

- 普通 Post-LN 合并前是 $1 + 0.2 = 1.2$
- DeepNorm 合并前是 $3.36 \times 1 + 0.2 = 3.56$

这说明 DeepNorm 的方向不是“把一切都压小”，而是**把残差主干抬强，把新更新缩稳**。真正要控制的是“每一层对整个网络行为的扰动大小”。

下面把几种结构并排比较：

| 机制 | 残差是否缩放 | 是否改初始化 | 梯度路径特点 | 对深层稳定性的影响 |
|---|---|---|---|---|
| Post-LN | 否 | 通常否 | 路径较敏感 | 深层常不稳 |
| Pre-LN | 否 | 通常否 | 梯度更直通 | 明显更稳 |
| DeepNorm | 是 | 是 | 主干更强，子层更克制 | 超深训练更稳 |
| ResiDual | 是，双路径视角 | 依实现而定 | 一条保梯度，一条保表达 | 深层兼顾稳定和表达 |

再看 ResiDual。它的关键不是“Pre-LN 和 Post-LN 都来一份”这么粗糙，而是**保留两条功能不同的残差流**。

可以把它理解成：

- 主链保留更像 Post-LN 的表达路径
- 旁路保留更像 Pre-LN 的稳定梯度路径
- 两路最后合并，让深层网络既不容易把梯度堵死，也不必完全退化到只追求稳定

如果用极简形式表达，可以写成：

$$
h_{l+1} = \text{MainPath}(h_l) + \text{AuxResidual}(h_l)
$$

这里的 MainPath 和 AuxResidual 并不是随便叠两个 block，而是有明确分工的两条残差支路。

用直观类比说明四者差异：

- Post-LN：先合并，再统一整理
- Pre-LN：先整理，再合并
- DeepNorm：主干加粗，子层初始动作变小
- ResiDual：两条路同时走，一条保训练，一条保表达

真实工程例子：做机器翻译或多语种预训练时，当 encoder-decoder 从 24 层扩到 100 层以上，很多团队会发现，原来在中等深度可用的结构突然开始出现早期 loss 抖动、训练窗口极窄、seed 很敏感。这个阶段，问题已经不只是“多跑几个 warmup 步”能解决，而是需要改 block 级设计。DeepNorm 和 ResiDual 都属于这个层面的解法。

---

## 代码实现

先强调一个工程事实：**DeepNorm 不是只改 forward 里一行残差公式**。如果你只把 $\alpha$ 加上去，却不改初始化，常常会得到一个看起来更奇怪、而不是更稳定的模型。

下面给一个可运行的 Python 玩具实现，用来验证系数和前向组合逻辑：

```python
import math

def deepnorm_coeff_decoder_only(L: int):
    assert L > 0
    alpha = (2 * L) ** 0.25
    beta = (8 * L) ** -0.25
    return alpha, beta

def post_ln_merge(x, fx):
    return x + fx

def deepnorm_merge(x, fx, alpha):
    return alpha * x + fx

alpha, beta = deepnorm_coeff_decoder_only(64)

assert round(alpha, 2) == 3.36
assert round(beta, 2) == 0.21

x_l = 1.0
f_x = 0.2

plain = post_ln_merge(x_l, f_x)
deep = deepnorm_merge(x_l, f_x, alpha)

assert abs(plain - 1.2) < 1e-9
assert round(deep, 2) == 3.56
assert deep > plain
```

上面这段代码没有实现完整 Transformer，但它验证了两个关键事实：

1. DeepNorm 的主干系数随深度变化。
2. 合并前的主干量级确实更强。

再给一个 PyTorch 风格的简化伪代码。先看 DeepNorm block：

```python
import torch
import torch.nn as nn

class DeepNormBlock(nn.Module):
    def __init__(self, hidden_size, alpha, attn, ffn):
        super().__init__()
        self.alpha = alpha
        self.attn = attn
        self.ffn = ffn
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.ln1(self.alpha * x + self.attn(x))
        x = self.ln2(self.alpha * x + self.ffn(x))
        return x
```

这里只展示结构意图：**残差主干乘以 `alpha` 再与子层输出相加**。

初始化同样关键。简化理解如下：

| 参数 | 是否缩放 | 缩放系数 | 作用位置 |
|---|---|---:|---|
| `W_q` | 通常否 | 标准 Xavier | Query 投影 |
| `W_k` | 通常否 | 标准 Xavier | Key 投影 |
| `W_v` | 是 | `beta` | Value 投影 |
| `W_o` | 是 | `beta` | Attention 输出投影 |
| `W_ffn``(部分层)` | 是 | `beta` | FFN 相关线性层 |

一个简化初始化伪代码：

```python
def init_weights(module, beta):
    for name, p in module.named_parameters():
        if p.dim() < 2:
            continue
        if any(key in name for key in ["w_v", "out_proj", "ffn"]):
            nn.init.xavier_normal_(p, gain=beta)
        else:
            nn.init.xavier_normal_(p)
```

这段代码不是官方实现原样，但表达的是工程原则：**哪些矩阵要缩，哪些不要缩，必须和论文定义一致**。

再看 ResiDual 的极简结构意图：

```python
class ResiDualBlock(nn.Module):
    def __init__(self, hidden_size, attn, ffn):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x, aux):
        main = self.post_ln(x + self.attn(x))
        aux = aux + self.ffn(self.pre_ln(x))
        out = main + aux
        return out, aux
```

这段伪代码强调三件事：

1. 它有额外状态 `aux`
2. 不是单路输入单路输出的最普通 block
3. 状态字典、前向接口、层间累计逻辑都可能要改

这也是为什么 ResiDual 不能被理解为“把 Pre-LN block 和 Post-LN block 首尾相接”。真正的难点在于：**你要维护双路径的一致性，而不是多堆几层模块**。

工程上常见配置项可以整理为：

| 配置项 | 含义 |
|---|---|
| `depth` | 网络深度，决定是否真的需要结构级稳定化 |
| `hidden_size` | 隐藏维度，影响参数规模与训练噪声 |
| `alpha` | DeepNorm 残差主干缩放系数 |
| `beta` | DeepNorm 初始化缩放系数 |
| `use_residual_dual` | 是否启用双残差路径 |

如果是已有项目改造，建议顺序是：

1. 先把 baseline 跑稳
2. 单独引入 DeepNorm 或 ResiDual，不要一次混多个结构技巧
3. 明确哪些层必须重初始化
4. 检查 checkpoint 是否还能兼容旧结构

---

## 工程权衡与常见坑

DeepNorm 和 ResiDual 都不是“打开就涨点”的通用插件，它们的代价不同。

DeepNorm 的主要代价在于：

- 你要严格匹配残差缩放和初始化策略
- 已有模型结构若很多地方复用默认初始化，改造会牵涉多个模块
- 与其他归一化技巧混用时，收益可能不再可解释

ResiDual 的主要代价在于：

- 结构更复杂
- 前向图和状态传递不再是标准 Transformer
- checkpoint、蒸馏、并行切分、导出部署都可能要多处理一层兼容性

下面是常见坑：

| 错误做法 | 后果 | 修正方式 |
|---|---|---|
| 只调 `alpha`，不调 `beta` | 训练更不稳，早期更新过猛 | 残差缩放和初始化成套修改 |
| 把 DeepNorm 与其他 LN 技巧随意叠加 | 效果不可控，难定位问题 | 一次只改一个主机制 |
| 把 ResiDual 当成简单串联 | 前向逻辑与状态字典错位 | 按双路径设计实现 |
| 模型只有十几层就硬上复杂结构 | 增加维护成本，收益很小 | 先用 Pre-LN baseline |
| 没看梯度和激活监控，只看 loss | 容易误判问题来源 | 同时监控梯度范数与激活漂移 |

这里可以给一个真实工程判断框架。

如果你训练 18 层或 24 层模型，loss 前期有点抖，但：

- 梯度范数整体可控
- 调小学习率后可收敛
- 不同 seed 结果差异不大

那大概率还是优化问题，先别动结构。

如果你把模型从 48 层推到 128 层以后出现这些现象：

- warmup 很长仍然发散
- 梯度范数周期性飙升
- 深层激活均值或方差持续漂移
- 不同随机种子差异巨大

那更像是结构级稳定性问题。这时只继续磨学习率，通常性价比不高。

训练监控可以重点看三类指标：

| 指标 | 观察什么 | 说明什么 |
|---|---|---|
| loss 曲线 | 是否早期剧烈震荡 | 更新是否失控 |
| 梯度范数 | 是否偶发尖峰 | 深层回传是否不稳 |
| 层间激活统计 | 均值/方差是否持续漂移 | 残差累计是否失衡 |

一句话概括：**warmup 解决的是“更新节奏”，DeepNorm/ResiDual 解决的是“网络结构本身的信号流”**。两者不是互斥关系，但作用层级不同。

---

## 替代方案与适用边界

先给实用建议。不要一看到“超深稳定训练”就默认上 DeepNorm 或 ResiDual。大多数初级工程场景，模型压根没深到需要它们。

可以按深度粗略分层：

| 方法 | 主要优点 | 主要代价 | 适用深度 | 是否需要改初始化 |
|---|---|---|---|---|
| Pre-LN | 简单、稳、成熟 | 可能牺牲部分结构特性 | 12-48 层常见 | 否 |
| warmup + 梯度裁剪 | 成本低，先验强 | 只治更新节奏 | 各深度都该先试 | 否 |
| 初始化调优 | 改动小 | 往往不够系统 | 中浅层优先 | 是 |
| DeepNorm | 超深训练稳定性强 | 要改残差和初始化 | 100 层以上更有价值 | 是 |
| ResiDual | 稳定性与表达兼顾 | 结构复杂、维护重 | 深层到超深 | 视实现而定 |

边界判断清单可以直接这么用：

1. 模型是否已经深到 100 层上下？
2. 现有 Pre-LN baseline 是否已经充分调过？
3. 发散是否稳定复现，而不是偶发坏 seed？
4. 是否能接受更复杂的实现和 checkpoint 管理？
5. 任务更优先要“先训稳”，还是要“稳定和表达兼顾”？

对应选择通常是：

- **12 到 24 层**：优先 Pre-LN、warmup、梯度裁剪、标准初始化
- **24 到 64 层**：先做成熟 baseline，再判断是否真有结构瓶颈
- **100 层以上**：DeepNorm 开始变得有明确价值
- **既要很深，又不想明显牺牲表达路径设计**：ResiDual 更值得评估

最后再给一个直白判断。

如果你的目标只是把小模型训出来，DeepNorm 和 ResiDual 往往太重；如果你的目标是把 Transformer 从“能训”推进到“几百层也能稳训”，那它们就不是可选优化，而是结构设计的一部分。

---

## 参考资料

| 来源 | 适合查什么 | 读者优先级 |
|---|---|---|
| DeepNet 论文 | 理论动机、公式、稳定性分析 | 高 |
| ar5iv 全文 | 更易读的论文正文 | 高 |
| TorchScale | 工程实现与配置方式 | 很高 |
| ResiDual 论文 | 双残差结构设计 | 高 |
| ResiDual 仓库 | 实际代码与训练细节 | 中高 |

1. [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
2. [DeepNet 论文全文（ar5iv）](https://ar5iv.labs.arxiv.org/html/2203.00555)
3. [TorchScale 官方仓库](https://github.com/microsoft/torchscale)
4. [ResiDual: Transformer with Dual Residual Connections](https://arxiv.org/abs/2304.14802)
5. [ResiDual 官方仓库](https://github.com/microsoft/ResiDual)

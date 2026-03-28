## 核心结论

Transformer 的残差连接写成
$$
x_{l+1}=x_l+F_l(x_l)
$$
其中 $x_l$ 是第 $l$ 层输入，$F_l$ 是这一层的注意力或前馈网络变换。它的核心作用不是“多加一条路”这么简单，而是提供一条**恒等路径**。恒等路径就是输入几乎不变地直接传到下一层的通道。反向传播时，梯度可以沿着这条路径直接回到前面层，因此深层网络不会像普通堆叠那样很快出现梯度消失。

对初学者最重要的理解是：残差块不需要每层都“重写全部表示”，它只需要学习一个增量，也就是“在原表示上改一点点”。这也是为什么深层 Transformer 能堆起来。没有残差时，每层都要负责完整变换；有残差时，子层只负责局部修正，主干信息继续向前流。

一个玩具例子：把 100 层网络想成 100 次连续修稿。没有残差时，第 100 次修稿只能看到第 99 次版本，早期信息容易被反复改坏；有残差时，每次修稿都在“原稿 + 小修正”上进行，原稿主干始终还在，所以梯度和信息都更容易保留下来。

| 设计 | 前向信息流 | 反向梯度流 | 深层训练风险 |
| --- | --- | --- | --- |
| 无残差 | 每层都完全改写表示 | 梯度必须穿过所有非线性层 | 梯度消失/爆炸明显 |
| 有残差 | 主干表示直接传递，子层只加增量 | 存在近似恒等的直通路径 | 可训练层数显著增加 |

经验上，残差流在很多 Transformer 中是表示的主载体。也就是说，大部分信息沿主干流动，attention 和 FFN 更多是在主干上写入增量，而不是每层重新编码全部语义。因此，残差连接不是附属结构，它本身就是 Transformer 深度扩展的基本条件。

---

## 问题定义与边界

这里讨论的是**深层 Transformer**，尤其是几十层到上千层的模型。问题不在于“残差有没有用”，而在于“只有普通残差够不够”。答案是：层数一深，普通 Post-Norm 写法往往不够。

Post-Norm 指的是先做残差相加，再做 LayerNorm，即：
$$
x_{l+1}=\mathrm{LayerNorm}(x_l+F_l(x_l))
$$
Pre-Norm 则是先归一化再进子层。LayerNorm 可以理解成“把一层向量重新拉回稳定尺度”的归一化操作。两者都用残差，但训练稳定性差异很大。普通 Post-Norm 在深层时容易出现更新量累积失控，训练直接发散。DeepNorm 的工作正是解决这个问题。

第二个边界是显存。层数增加后，反向传播要保存更多中间激活。激活就是前向计算时每层的中间结果。即使梯度稳定了，显存也会先撑爆，所以工程上必须考虑 gradient checkpointing，也就是**激活检查点**。它的做法是少存一些中间结果，在反向时重算一部分前向。

第三个边界是微调。很多人把“残差设计”和“LoRA 微调”分开看，其实它们思想一致。LoRA 不是重训整个权重，而是在原权重上叠加一个低秩增量：
$$
\Delta W=BA
$$
低秩的意思是用两个更小的矩阵近似一个大矩阵更新，参数更省。它本质上也是“保留主干，只学差分”。

| 问题 | 边界 | 解决方案 |
| --- | --- | --- |
| 深层梯度消失 | 普通深堆叠难训练 | 残差提供恒等梯度路径 |
| Post-Norm 发散 | 层数很深时更新量累积过大 | DeepNorm 的 $\alpha,\beta$ 缩放 |
| 激活显存过高 | 层数和序列长度都大时显存先爆 | Gradient checkpointing |
| 全量微调成本高 | 大模型难以训练和部署 | LoRA 只学习 $\Delta W$ |

DeepNorm 常见写法会引入按层数定义的系数。对 encoder-only 或常见简化讨论，可以先抓住这两个量：
$$
\alpha=(2N)^{1/4},\qquad \beta=(8N)^{-1/4}
$$
其中 $N$ 是总层数。直觉上，$\alpha$ 用来增强主干残差，$\beta$ 用来压低子层更新。这样深层累积后的总更新量仍能维持在可控范围。

---

## 核心机制与推导

先看最基本的反向传播。若
$$
x_{l+1}=x_l+F_l(x_l)
$$
那么对上一层输入求导：
$$
\frac{\partial x_{l+1}}{\partial x_l}=I+\frac{\partial F_l(x_l)}{\partial x_l}
$$
这里的 $I$ 是单位映射，也就是“直接把自己传过去”的那部分。关键点在于，即使 $\frac{\partial F_l}{\partial x_l}$ 很小，梯度里仍然保留一条来自 $I$ 的直通项。这就是残差能缓解梯度消失的数学原因。

但到了深层 Post-Norm，仅有这条直通路径还不够。因为每层子模块输出如果尺度不受控，很多层叠加后总更新会越来越大。DeepNorm 的思路是把残差主干和子层增量明确区分开：
$$
x_{l+1}=\mathrm{LayerNorm}(\alpha x_l+G_l(x_l;\theta_l))
$$
同时在初始化或投影层上配合 $\beta$ 缩放。可以把它理解成“放大主干、压小支路”。这样训练时模型主要沿着稳定主干传播，子层做细粒度修正。

用 12 层做玩具数值例子：
$$
\alpha=(2\times 12)^{1/4}\approx 2.21,\qquad \beta=(8\times 12)^{-1/4}\approx 0.32
$$
不同结构下实现细节会略有差异，但直觉不变：主干被明确保住，增量被明确约束。很多面向工程的二手资料会给出近似“缩到一半”的口径，本质上是在强调“增量必须明显小于主干”，而不是要求你死记某个四舍五入数。

可以把这一层想成文字示意图：

`输入 x_l`  
`├─ 恒等路径: 乘 alpha，直接往后传`  
`└─ 子层路径: Attention/FFN 产生更新，再按 beta 缩放`  
`最后把两条路相加，再做 LayerNorm`

LoRA 也符合这个结构。原层是
$$
y=Wx
$$
LoRA 后变成
$$
y=(W+\Delta W)x=(W+BA)x
$$
其中 $W$ 冻结，只训练 $A,B$。这说明 LoRA 不是替换原有能力，而是在原能力之上叠加一个低秩修正。它和残差的共同思想是：**先保住主路径，再学习小更新**。

真实工程例子是训练超深翻译模型或大语言模型底座。假设你要训一个 1000 层 Post-Norm Transformer。只用普通 Xavier 初始化，子层输出会在深堆叠下逐步放大，loss 很容易在早期爆掉。工程上通常会同时做三件事：

1. 用 DeepNorm 或类似缩放控制残差更新。
2. 用 checkpointing 减少激活存储。
3. 微调阶段不用全量训练，而是在 Q/K/V/FFN 上挂 LoRA。

这三者不是彼此独立的技巧，而是围绕同一目标展开：让主干稳定，让增量可控，让资源可承受。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是完整 Transformer，而是把残差、DeepNorm 风格缩放和 LoRA 的核心结构抽出来，便于理解。

```python
import math

def layer_norm(vec, eps=1e-5):
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / len(vec)
    return [(x - mean) / math.sqrt(var + eps) for x in vec]

def linear(vec, weight):
    out = []
    for row in weight:
        out.append(sum(a * b for a, b in zip(row, vec)))
    return out

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def scale(a, s):
    return [x * s for x in a]

def deepnorm_block(x, sublayer, alpha, beta):
    # alpha 控制恒等路径强度
    residual = scale(x, alpha)
    # beta 控制子层更新幅度
    delta = scale(sublayer(x), beta)
    return layer_norm(add(residual, delta))

def make_lora_delta(x, A, B):
    low_rank = linear(x, A)      # 降维
    return linear(low_rank, B)   # 升维

def lora_linear(x, W, A, B, lora_scale=1.0):
    base = linear(x, W)
    delta = scale(make_lora_delta(x, A, B), lora_scale)
    return add(base, delta)

# 玩具输入
x = [1.0, 2.0, -1.0]

# 3 -> 3 的子层
W_sub = [
    [0.2, 0.0, 0.1],
    [0.1, 0.3, -0.2],
    [0.0, 0.2, 0.4],
]

def sublayer(v):
    return linear(v, W_sub)

N = 12
alpha = (2 * N) ** 0.25
beta = (8 * N) ** (-0.25)

y = deepnorm_block(x, sublayer, alpha, beta)
assert len(y) == 3
assert abs(sum(y) / len(y)) < 1e-4  # LayerNorm 后均值接近 0

# LoRA: W 是冻结主权重，A/B 是可训练低秩增量
W = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]

A = [
    [0.1, 0.0, 0.2],
]  # 1 x 3

B = [
    [0.5],
    [-0.3],
]  # 2 x 1

out = lora_linear(x, W, A, B, lora_scale=0.8)
assert len(out) == 2

base = linear(x, W)
assert out != base  # LoRA 确实添加了增量
```

如果换成 PyTorch，核心 forward 结构通常类似这样：

```python
def forward(self, x):
    x_main = self.alpha * x
    x_delta = self.beta * self.sublayer(x)
    return self.ln(x_main + x_delta)
```

LoRA 则常见为：

```python
def forward(self, x):
    return self.base(x) + self.scale * self.B(self.A(x))
```

这两段代码背后的共同点很重要：原路径始终存在，训练只是在原路径上叠加受控变化。

---

## 工程权衡与常见坑

残差连接能让模型训起来，但工程上最常见的问题不是“有没有残差”，而是“残差尺度、显存策略、微调范围是否匹配”。

| 坑 | 后果 | 规避 |
| --- | --- | --- |
| 深层 Post-Norm 直接套普通初始化 | 训练前期更新爆炸，loss 发散 | 用 DeepNorm 或等价缩放 |
| 子层输出不做深度相关缩放 | 后层更新累积失控 | 用 $\alpha,\beta$ 控制主干和增量 |
| checkpoint 过密 | 重算太多，训练明显变慢 | 按 $\sqrt{n}$ 量级分段更常见 |
| checkpoint 太稀 | 显存节省不明显 | 结合层数和序列长度调粒度 |
| LoRA rank 太小 | 表达能力不足，任务学不动 | 从小 rank 起步，再按层数/任务加大 |
| LoRA rank 太大 | 参数和显存回升，失去 PEFT 意义 | 只在关键投影层上加适中 rank |
| 误把残差当“可有可无” | 深层性能和稳定性一起掉 | 不要轻易删主干路径 |

显存方面，gradient checkpointing 的理论经典结果是把激活存储从 $O(n)$ 降到 $O(\sqrt{n})$。工程里常见做法不是每层都 checkpoint，而是每隔若干层打一段。经验公式可以写成：
$$
\text{checkpoint interval}\approx \sqrt{n}
$$
其中 $n$ 是总层数。它不是精确最优解，而是常用起点。代价是反向传播需要重算部分前向，因此训练时间通常会上升，经典论文和很多工程经验都报告过大约 30% 左右额外开销，实际会随框架、序列长度和 kernel 优化波动。

还有一个经常被忽略的点：如果残差流承载了大部分稳定信息，那么 FFN 和 attention 的职责更像“增量编辑器”。这意味着做剪枝、早停、低秩适配时，不能只盯 attention，FFN 容量也很关键。否则主干虽然还在，但缺少足够的增量编辑能力，模型表达会明显下降。

---

## 替代方案与适用边界

如果你不是在训练极深 Post-Norm 模型，未必一定要上 DeepNorm。不同方案的适用边界很明确。

| 方案 | 适用场景 | 优点 | 代价/边界 |
| --- | --- | --- | --- |
| Pre-Norm | 中深层 Transformer，训练稳定优先 | 好训，工程成熟 | 有时最终性能不如优化好的 Post-Norm |
| Post-Norm + DeepNorm | 很深的 Transformer，想保留 Post-Norm 优势 | 可把层数推到很深 | 需要特殊缩放和初始化配套 |
| LoRA 微调 | 预训练模型下游适配 | 参数少，显存省，部署灵活 | 不解决从零训练的深层稳定性 |
| Checkpointing | 显存成为瓶颈时 | 不改模型结构，直接省显存 | 增加重算开销 |

简单决策可以这样看：

如果你是在训练几十层到一两百层模型，并且更看重稳定和简单，优先用 Pre-Norm。  
如果你明确要做很深的 Post-Norm 设计，或者要复现 DeepNet 一类路线，就必须把残差缩放和初始化一起考虑。  
如果你不是从零训练，而是在现有大模型上做任务适配，优先考虑 LoRA，而不是动全部参数。  
如果模型本身已经稳定，但显存不够，再加 checkpointing；它解决的是资源问题，不是残差稳定性本身。

要特别避免一个误区：LoRA 不能替代 DeepNorm。LoRA 解决的是“怎么高效改参数”，DeepNorm 解决的是“深层网络怎么稳定训练”。两者关注点不同，但它们都遵循“保主干、学增量”的残差思想。

---

## 参考资料

- Vaswani et al., *Attention Is All You Need*. Transformer 原始结构，定义了 attention/FFN 外的残差连接。https://arxiv.org/abs/1706.03762
- He et al., *Deep Residual Learning for Image Recognition*. 残差学习的经典起点，解释了恒等映射为何能帮助深层训练。https://arxiv.org/abs/1512.03385
- Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers*. DeepNorm 的核心来源，讨论深层 Post-Norm 的稳定训练与层数扩展。https://arxiv.org/abs/2203.00555
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*. 给出 $\Delta W=BA$ 的低秩增量微调形式。https://arxiv.org/abs/2106.09685
- Chen et al., *Training Deep Nets with Sublinear Memory Cost*. 激活检查点的经典论文，给出 $O(\sqrt{n})$ 内存-计算折中。https://arxiv.org/abs/1604.06174
- Anthropic, *Privileged Bases in the Transformer Residual Stream*. 讨论 residual stream 作为 Transformer 主信息通道的重要性。https://transformer-circuits.pub/2023/privileged-basis/index.html
- Guangxuan Xiao, *Stacking Your Transformer Layers? Better Keep the Residual*. 面向实践的经验分析，讨论 skip connection 对深层信息流和 FFN 依赖的启发。https://guangxuanx.com/blog/stacking-swa.html

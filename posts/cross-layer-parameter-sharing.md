## 核心结论

跨层参数共享指的是：**多个网络层反复使用同一组参数**。白话说，原本要造 12 套不同零件，现在只保留 1 套零件，前向传播时把这套零件重复用 12 次。

它解决的核心问题不是“减少计算”，而是**减少存储参数**。如果单层参数记为 $|\theta|$，层数为 $L$，普通堆叠的参数量近似是 $L\times |\theta|$；完全共享后变成：

$$
P_{\text{shared}} \approx |\theta|
$$

所以，**深度仍然存在，但参数不再随层数线性增长**。这也是 ALBERT 能在保持 12 层编码深度时，把参数量显著压缩的原因之一。

一个常见误解是：参数共享后推理会同等比例变快。这个结论通常不成立。因为前向仍然要跑 $L$ 次，同样要做 $L$ 次注意力和前馈网络计算，减少的是模型权重存储、显存占用和优化状态大小，不是把深度“删掉”。

先看最直观的量级对比：

| 模型 | 编码层数 | 是否跨层共享 | 参数量量级 | 主要减参来源 |
|---|---:|---|---:|---|
| BERT-base | 12 | 否 | ~110M | 无 |
| ALBERT-base | 12 | 是 | ~12M | 跨层共享 + embedding 分解 |
| ALBERT-xxlarge | 12 | 是 | ~235M | 共享层参数，扩大隐藏规模 |

这个表有两个新手常见误区需要提前说明：

| 误区 | 正确说法 |
|---|---|
| 参数量变小，计算量一定同比下降 | 不成立。层数不变时，主干计算仍然要执行多次 |
| 共享后就等于只剩 1 层 | 不成立。仍然是多层迭代，只是每层用同一组权重 |
| 共享后表达能力完全相同 | 不成立。参数自由度下降，层间差异通常会变少 |

因此，跨层参数共享最适合的判断标准不是“我要不要更快”，而是“**我能否接受层间表达多样性下降，以换取更低参数成本**”。

---

## 问题定义与边界

先把边界说清楚。

这里讨论的“跨层参数共享”，特指 **Transformer 编码层之间复用权重**。术语“权重”就是训练中被更新的数值矩阵，比如自注意力里的投影矩阵、前馈网络里的线性层参数。它不等于缓存、中间激活，也不等于输入 embedding。

普通 Transformer 的做法是：

- 第 1 层有自己的一套 attention 和 FFN 参数
- 第 2 层再有自己的一套
- 一直到第 $L$ 层

所以参数量随着层数近似线性增长。若每层参数为 $|\theta|$，则：

$$
P_{\text{stack}} \approx L \times |\theta|
$$

完全共享后，所有层共用同一个参数集 $\Theta$：

$$
h^{(\ell)} = f\!\left(h^{(\ell-1)}; \Theta\right), \quad \ell = 1,2,\dots,L
$$

这里的 $h^{(\ell)}$ 是第 $\ell$ 层的隐藏状态，白话说就是“这一层处理后的表示”；$f$ 是同一个 Transformer 块。

这个定义要和另外几类压缩手段区分开，否则很容易把不同问题混成一个问题：

| 方法 | 压缩对象 | 是否减少层间独立参数 | 典型做法 |
|---|---|---|---|
| 跨层参数共享 | 编码层权重 | 是 | 多层复用同一套 block |
| 低秩分解/矩阵分解 | 单层矩阵 | 不一定 | 把大矩阵拆成两个小矩阵 |
| 剪枝 | 参数连接 | 不一定 | 删除不重要权重 |
| 量化 | 参数数值表示 | 不一定 | FP32 改成 INT8/INT4 |
| 蒸馏 | 模型整体容量 | 否 | 用小模型拟合大模型 |

再补一个经常一起出现的边界：ALBERT 不只做了跨层共享，还做了 **embedding 分解**。embedding 就是把词 ID 映射成向量的查表矩阵。原始 BERT 词嵌入近似是：

$$
O(V\times H)
$$

其中 $V$ 是词表大小，$H$ 是隐藏维度。ALBERT 把它拆成两段：

$$
O(V\times E + E\times H), \quad H \gg E
$$

这里的 $E$ 是较小的 embedding 维度。拆分后的含义是：

- 先把词映射成较小维度的向量
- 再通过一个投影矩阵升到隐藏维度

这会进一步减少参数，但它和“跨层共享”不是同一个机制，分析时不要混在一起。否则会出现一种常见错误：把 ALBERT 的总减参效果全部归因到跨层共享。实际上它是两种机制叠加。

再看一个最小玩具例子。假设你有一个把数字“加 1 再乘 2”的函数块：

- 不共享：写 12 个函数，参数各不相同
- 共享：只写 1 个函数，重复调用 12 次

第二种做法函数定义更少，但输入仍然会经过 12 次变换。**共享的是规则，不是跳过步骤。**

如果把“参数”“激活”“计算”三个概念拆开，边界会更清楚：

| 对象 | 是否因跨层共享而减少 |
|---|---|
| 模型参数 | 通常显著减少 |
| 优化器状态 | 通常跟着减少 |
| 前向计算次数 | 通常不减少 |
| 中间激活数量 | 训练时通常不按参数量同比减少 |
| 推理深度 | 不减少 |

这也是为什么跨层参数共享更像一种**参数效率设计**，而不是一种**算力效率设计**。

---

## 核心机制与推导

机制可以概括成一句话：**把“深层网络”改写成“同一个变换块的迭代应用”**。

普通 12 层 Transformer 的表达是：

$$
h^{(1)} = f_1(h^{(0)}),\quad
h^{(2)} = f_2(h^{(1)}),\quad \dots,\quad
h^{(12)} = f_{12}(h^{(11)})
$$

其中每个 $f_\ell$ 都有不同参数。

跨层共享后变成：

$$
h^{(\ell)} = f(h^{(\ell-1)}; \Theta), \quad \forall \ell \in [1,L]
$$

所有层都调用同一个 $f$。这使模型更像一个“迭代精炼器”：每过一层，不是换一套新规则，而是用**同一套规则继续修正表示**。

### 为什么共享后仍然有“层次”？

因为虽然参数相同，但输入状态不同。形式上：

$$
h^{(0)} \neq h^{(1)} \neq h^{(2)} \neq \cdots \neq h^{(L)}
$$

所以，同一组参数作用在不同状态上，仍然会产生不同输出。把它写成复合函数更直观：

$$
h^{(L)} = \underbrace{f \circ f \circ \cdots \circ f}_{L\text{ 次}}(h^{(0)})
$$

这里的关键不是“函数有几个”，而是“函数被应用了几次”。即便函数本身一样，多次迭代后也会产生逐层变化的表示。

### 参数量为什么会下降？

设单层 attention、FFN、LayerNorm 等参数总量为 $|\theta|$。

不共享时：

$$
P_{\text{stack}} = L|\theta|
$$

完全共享时：

$$
P_{\text{shared}} = |\theta|
$$

如果每 $k$ 层共享一组，那么共有 $\frac{L}{k}$ 组参数：

$$
P_{\text{group}} = \frac{L}{k}|\theta|
$$

这说明参数共享不是非黑即白，还可以做**分组共享**。

### 计算量为什么通常不同比下降？

设单层计算量近似为 $C_{\text{layer}}$。那么：

- 不共享：总计算量约为 $L \cdot C_{\text{layer}}$
- 完全共享：总计算量仍约为 $L \cdot C_{\text{layer}}$

即：

$$
F_{\text{shared}} \approx F_{\text{stack}} \approx L \cdot C_{\text{layer}}
$$

因此，**参数量下降和 FLOPs 下降不是同一件事**。这一点必须单独记住。

### 表达能力会发生什么变化？

关键点在于：独立层允许每层学习不同功能，例如低层偏局部模式、高层偏语义组合；完全共享则强迫所有层围绕同一个变换结构工作。这会带来一种**隐式正则化**。术语“正则化”就是限制模型自由度，减少过拟合倾向。

可以把它理解成下面这张表：

| 结构 | 每层是否有独立参数 | 模型自由度 | 层间功能分化 |
|---|---|---|---|
| 普通堆叠 | 是 | 高 | 通常更强 |
| 完全共享 | 否 | 低 | 通常更弱 |
| 分组共享 | 部分有 | 中 | 中等 |

自由度下降的后果有两面：

| 影响 | 含义 |
|---|---|
| 好处 | 更省参数，更像一种结构化约束 |
| 风险 | 某些任务需要的层间差异可能学不出来 |

### 一个一维迭代玩具例子

设一个最简单的一维共享迭代：

$$
h^{(\ell)} = 0.8h^{(\ell-1)} + 1
$$

从 $h^{(0)}=0$ 开始：

$$
h^{(1)} = 1
$$

$$
h^{(2)} = 1.8
$$

$$
h^{(3)} = 2.44
$$

$$
h^{(4)} = 2.952
$$

参数完全没变，但状态不断精炼，逐步逼近稳定点。这个稳定点满足：

$$
h^\* = 0.8h^\* + 1
$$

解得：

$$
h^\* = 5
$$

这个例子说明：**共享参数不等于输出不变**。只要输入状态在变，层输出就会继续变化。

### 真实工程例子：ALBERT

ALBERT 的核心不是“把 BERT 砍成更浅”，而是：

- 编码层参数做跨层共享
- embedding 做分解，减少词表相关参数

这样会减少：

- 模型权重文件大小
- 优化器状态占用
- 多卡训练时的参数同步压力

但不会让每个 token 少做那几轮 attention 计算。对新手来说，最稳妥的记忆方式是：

> **ALBERT 主要省的是参数，不是层数。**

---

## 代码实现

实现层面最重要的一点是：**只实例化一个 block，多次调用它**。不要创建 `layers = [Block() for _ in range(L)]`，而是创建一个 `shared_block = Block()`，然后循环调用。

下面先给一个可以直接运行的 Python 玩具实现。它不依赖第三方库，运行后能看到两件事：

- 同一个 `block` 被重复使用
- 深度增加后，输出会变化，但参数数量不变

```python
import math
from dataclasses import dataclass

@dataclass
class SharedBlock:
    w_attn: float
    w_ffn: float

    def __call__(self, x: float) -> float:
        # 极简“残差 + 两段变换”
        attn_out = x * self.w_attn
        x = x + attn_out

        ffn_out = math.tanh(x * self.w_ffn)
        x = x + ffn_out
        return x


def forward_with_sharing(x: float, depth: int, block: SharedBlock) -> float:
    states = [x]
    for _ in range(depth):
        x = block(x)
        states.append(x)
    return x, states


def parameter_count_without_sharing(depth: int, params_per_block: int) -> int:
    return depth * params_per_block


def parameter_count_with_sharing(params_per_block: int) -> int:
    return params_per_block


def demo() -> None:
    block = SharedBlock(w_attn=0.1, w_ffn=0.3)

    y1, states1 = forward_with_sharing(1.0, depth=1, block=block)
    y3, states3 = forward_with_sharing(1.0, depth=3, block=block)

    assert y3 != y1
    assert parameter_count_without_sharing(depth=12, params_per_block=2) == 24
    assert parameter_count_with_sharing(params_per_block=2) == 2

    # 前向多次调用的是同一个对象
    first_id = id(block)
    _ = block(1.23)
    second_id = id(block)
    assert first_id == second_id

    print("depth=1 final:", round(y1, 4))
    print("depth=3 final:", round(y3, 4))
    print("states(depth=3):", [round(v, 4) for v in states3])
    print("params without sharing:", parameter_count_without_sharing(12, 2))
    print("params with sharing:", parameter_count_with_sharing(2))


if __name__ == "__main__":
    demo()
```

一组典型输出会类似这样：

```text
depth=1 final: 1.3185
depth=3 final: 2.3442
states(depth=3): [1.0, 1.3185, 1.7811, 2.3442]
params without sharing: 24
params with sharing: 2
```

这个玩具实现能说明结构，但还不够接近真实深度学习框架。下面给一个可以直接运行的 PyTorch 版本。它依然是简化模型，但已经具备“共享层反复调用”的工程写法。

```python
import torch
import torch.nn as nn


class SharedMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class SharedEncoder(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.block = SharedMLPBlock(dim)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.depth):
            x = self.block(x)
        return x


def count_unique_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    dim = 8
    depth = 6

    model = SharedEncoder(dim=dim, depth=depth)
    x = torch.randn(batch_size, seq_len, dim)
    y = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("unique parameter count:", count_unique_params(model))

    # 确认整个模型里只有一套 block 参数
    block_param_ids = [id(p) for p in model.block.parameters()]
    model_param_ids = [id(p) for p in model.parameters()]
    assert set(block_param_ids) == set(model_param_ids)

    # 做一次反向传播，确认代码可训练
    loss = y.mean()
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print("grad norm sum:", round(grad_norm, 6))


if __name__ == "__main__":
    main()
```

这个版本展示了一个关键事实：`depth=6` 只代表前向中执行 6 次，并不代表模型里有 6 套参数。

### 工程中最常见的两种写法

第一种写法是**单对象重复调用**：

```python
self.block = Block()
for _ in range(depth):
    x = self.block(x)
```

第二种写法是**显式存层号，但仍共享主参数**：

```python
class Encoder(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.block = block
        self.depth = depth

    def forward(self, x):
        for layer_idx in range(self.depth):
            x = self.block(x, layer_idx=layer_idx)
        return x
```

这里 `layer_idx` 的作用是给“共享但不完全相同”的设计留接口。例如：

- dropout 随层调整
- 部分 LayerNorm 不共享
- 根据层号注入位置偏置
- 给每层插入独立 adapter

### 一个常见错误写法

很多初学者会误写成下面这样：

```python
self.blocks = nn.ModuleList([Block() for _ in range(depth)])
for block in self.blocks:
    x = block(x)
```

这不是跨层参数共享。因为 `Block()` 被实例化了多次，每个对象都持有独立参数。

判断方法很简单：

| 写法 | 是否共享 |
|---|---|
| `self.block = Block()` 后循环调用 | 是 |
| `ModuleList([Block() for _ in range(L)])` | 否 |
| 多个层引用同一个 `Block` 对象 | 是 |
| 多个层结构长得一样，但各自单独初始化 | 否 |

真实工程例子可以看 ALBERT 类模型的实现思路：编码器堆栈不再保存 12 套独立层对象，而是保存 1 套共享层，前向中重复执行多次。这样 checkpoint 更小，参数同步更轻，但激活图和计算深度仍按实际层数保留。

---

## 工程权衡与常见坑

跨层参数共享的收益明确，但副作用也很稳定，不属于“调一调就没事”的小问题。

第一，**参数少不等于 FLOPs 少**。FLOPs 就是浮点运算量，白话说是“实际算了多少次乘加”。如果还是 12 层，就还是要算 12 次。省下来的主要是权重存储，不是前向次数。

第二，**层间特征多样性会下降**。普通堆叠里，第 3 层和第 11 层可以学完全不同的功能；完全共享后，它们只能在“同一套算子反复应用”的框架里分工。这对语言理解任务常常可接受，但对强依赖分层特征的任务，风险更大。

第三，**训练稳定性要单独观察**。共享会让梯度反复汇聚到同一组参数上，更新信号更集中。若学习率、归一化、dropout 设计不合适，容易出现训练震荡。

把这些权衡压缩成一张表，更容易判断：

| 维度 | 不共享 | 完全共享 | 影响 |
|---|---|---|---|
| 参数量 | 高 | 低 | 共享明显占优 |
| 计算量 | 高 | 高 | 通常没有本质变化 |
| 表达多样性 | 高 | 低 | 共享通常吃亏 |
| 训练显存 | 高 | 低 | 共享通常更友好 |
| 超参数敏感性 | 中 | 中到高 | 共享结构更需要重新调参 |

常见坑可以直接列出来：

1. 误把“模型更小”理解成“延迟更低”。在很多线上服务里，吞吐瓶颈仍然来自 attention 计算，不来自权重大小。
2. 把所有子模块都一刀切共享。实践里常见折中是 attention 和 FFN 分别处理，或者 LayerNorm 不共享。
3. 忽略任务类型差异。分类、句对匹配这类任务往往更容易接受共享；视觉、多模态、结构化预测任务常更依赖层次化表征。
4. 只看参数量，不看优化器状态。训练时 Adam 一类优化器还会给每个参数维护额外状态，参数共享的训练显存收益通常比“只看 checkpoint 大小”更明显。
5. 忽略调参迁移成本。一个在普通 Transformer 上稳定的超参数组合，放到共享结构上不一定仍然稳定。
6. 忽略激活显存。共享参数能减少权重显存，但深层网络训练时，激活保存仍然可能是大头。
7. 误把“共享”实现成“拷贝初始化相同”。初始值相同不代表训练过程中参数仍然共享。
8. 忽略诊断难度。多层共用同一套参数后，单层异常往往会传导到所有层，定位问题会更集中但也更难切分。

真实工程例子：假设你要把一个文本分类服务部署到显存紧张的单卡环境。原始 BERT-base 能跑，但 batch 稍微一大就溢出。此时换用 ALBERT 一类共享参数结构，常见收益是：

- 模型文件更小，加载更快
- 权重与优化器状态占用更低
- 显存压力下降，可容纳更大 batch 或更多并发
- 在 GLUE 类任务上，精度不一定明显掉

但如果你做的是文档结构解析、视觉语言联合编码，完全共享可能让中高层表示不够分化，最终精度回退比参数收益更明显。

对新手来说，有一个很实用的判断流程：

| 你最缺什么 | 优先考虑什么 |
|---|---|
| 显存/存储 | 跨层共享、embedding 分解、量化 |
| 推理速度 | 稀疏注意力、蒸馏、更浅模型、算子优化 |
| 极限精度 | 减少共享，保留更多独立层 |
| 训练稳定性 | 分组共享、独立 LayerNorm、较保守学习率 |

---

## 替代方案与适用边界

完全共享只是参数效率设计中的一个端点，不是唯一答案。

最常见替代方案是**分组共享**。例如 12 层分成 3 组，每组 4 层共享一套参数。这样参数量从 $12|\theta|$ 降到 $3|\theta|$，比完全共享多一些，但保留了组间差异。

| 策略 | 参数量近似 | 表达多样性 | 适用场景 |
|---|---|---|---|
| 不共享 | $L|\theta|$ | 高 | 追求容量上限 |
| 完全共享 | $|\theta|$ | 低 | 强参数约束、文本任务 |
| 分组共享 | $\frac{L}{k}|\theta|$ | 中 | 想折中参数与性能 |
| 只共享部分模块 | 介于两者之间 | 中高 | 需要保留层次结构 |

还有一种常见折中是**只共享部分参数**。例如：

- 共享 attention，不共享 FFN
- 共享 FFN，不共享 attention
- 共享主干层，但每层保留独立 LayerNorm 或 adapter

这里的 adapter 指插入在主干中的小型可训练模块，白话说就是“在共享主路旁边加一点低成本个性化参数”。

把这些折中放到同一张表里更容易看出差别：

| 方案 | 参数节省 | 性能风险 | 实现复杂度 |
|---|---|---|---|
| 完全共享 | 高 | 中到高 | 低 |
| 分组共享 | 中 | 中 | 低到中 |
| 仅共享 attention | 中 | 视任务而定 | 中 |
| 仅共享 FFN | 中 | 视任务而定 | 中 |
| 共享主干 + 独立 adapter | 中到高 | 通常较稳 | 中到高 |

ELECTRA 常被放进这个话题里，但要区分清楚。它更典型的共享点是**生成器和判别器的词嵌入共享**，不是“所有层完全共用一套编码参数”。它说明的不是“必须全层共享”，而是：**参数共享可以发生在不同位置**，包括层间、模型间、embedding 侧。

什么时候适合用跨层共享？

- 参数预算非常紧
- 任务以文本理解为主
- 更关心存储与训练显存，而不是单次前向延迟
- 可以接受少量精度波动，换取更大深度或更大 batch
- 你需要在同等显存下把模型做得更深

什么时候要谨慎？

- 任务依赖明显的多层级特征
- 模型是视觉 Transformer 或多模态结构
- 你需要分析每层表示并依赖层间分化
- 推理瓶颈主要在算力而不是显存
- 任务规模小，但需要高可解释性或精细中间表征

如果是初学者，一个实用原则是：**先尝试分组共享，再决定是否走向完全共享**。因为它通常比“一步到位全共享”更稳，更容易找到可接受的精度-参数平衡点。

还可以把选择逻辑压缩成一个决策表：

| 需求 | 更合适的方向 |
|---|---|
| 先保精度 | 不共享或分组共享 |
| 先保参数预算 | 完全共享 |
| 先保推理速度 | 不优先考虑跨层共享 |
| 先保训练显存 | 共享 + 混合精度 + 激活检查点 |
| 先保可解释的层间差异 | 不共享或少量共享 |

一句话总结适用边界：

> **跨层参数共享适合“参数预算紧、层数仍想保留”的场景，不适合把它当成通用加速器。**

---

## 参考资料

- Lan, Zhenzhong, et al. “ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.” arXiv:1909.11942. https://arxiv.org/abs/1909.11942
- Devlin, Jacob, et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” arXiv:1810.04805. https://arxiv.org/abs/1810.04805
- Clark, Kevin, et al. “ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.” arXiv:2003.10555. https://arxiv.org/abs/2003.10555
- ALBERT 主题综述与参数统计整理。https://www.emergentmind.com/topics/albert
- Amit Chaudhary, “ALBERT visual summary.” https://amitness.com/posts/albert-visual-summary
- ELECTRA 机制解读与嵌入共享说明。https://mbrenndoerfer.com/writing/electra-efficient-pretraining-replaced-token-detection
- 参数共享与递归/迭代网络的关系，可参考 Universal Transformer：Dehghani, Mostafa, et al. “Universal Transformers.” arXiv:1807.03819. https://arxiv.org/abs/1807.03819
- 跨层共享与工程副作用讨论。https://www.cnblogs.com/LAKan/p/16587407.html

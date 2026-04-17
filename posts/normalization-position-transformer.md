## 核心结论

Pre-Norm 和 Post-Norm 的差别只有一个：`LayerNorm` 放在子层前，还是放在残差相加后。`LayerNorm` 的作用是把每个 token 在特征维上的数值重新拉回到可控范围，避免某些维度过大、某些维度过小，进而让后续计算更稳定。

先给结论，再解释原因：

| 对比项 | Pre-Norm | Post-Norm |
|---|---|---|
| 公式 | $x_{l+1}=x_l+F(\mathrm{LN}(x_l))$ | $x_{l+1}=\mathrm{LN}(x_l+F(x_l))$ |
| 梯度主通路 | 残差分支保留近似恒等映射 | 梯度必须先穿过 LayerNorm |
| 深层训练稳定性 | 更强 | 更弱 |
| 学习率容忍度 | 更大 | 更小 |
| warmup 需求 | 往往可缩短甚至省掉 | 往往更依赖 warmup |
| 常见使用场景 | 大模型、深层网络、训练优先 | 浅层模型、部分精度优先场景 |

核心原因不是“Pre-Norm 更新”，而是它给反向传播保留了一条更直接的残差主路。对深层 Transformer 来说，这条路通常比局部模块写法更重要。

这也是为什么很多现代大模型优先使用 Pre-Norm：层数一旦上去，训练首先要解决的是“能不能稳定学起来”，其次才是“同等预算下能不能再多榨一点上限”。而较早的原始 Transformer 或部分 BERT 风格结构保留 Post-Norm，是因为它更接近 2017 年的初始设计，在层数不深、训练配方成熟时仍然可以取得很强结果。

如果只记一句话，可以记这个：

> Pre-Norm 的优势不在前向表达式更复杂，而在反向传播时更容易把梯度完整地送回前面层。

---

## 问题定义与边界

本文只讨论 Transformer block 内 `LayerNorm` 的放置位置，不讨论 BatchNorm、RMSNorm、DeepNorm，也不讨论 MoE、并行残差、缩放残差等额外技巧。这里的 `残差连接`，就是把输入直接加回输出的通路：

$$
\text{output} = \text{input} + \text{某个子层的变换}
$$

它的意义很直接：即使子层一开始没学到有用东西，原始输入仍然可以沿残差路径继续往后传。

一个标准子层可以是自注意力，也可以是前馈网络 FFN。若忽略 dropout，二者的区别只有顺序：

**Pre-Norm**
$$
x_{l+1}=x_l+F(\mathrm{LN}(x_l))
$$

**Post-Norm**
$$
x_{l+1}=\mathrm{LN}(x_l+F(x_l))
$$

在 6 层以内的小模型里，这个差异看起来像“实现风格不同”；但从训练角度看，它决定了梯度主路经过哪些算子。

| 位置 | Pre-Norm 的执行顺序 | Post-Norm 的执行顺序 |
|---|---|---|
| 注意力子层 | `LN -> Attention -> Add` | `Attention -> Add -> LN` |
| FFN 子层 | `LN -> FFN -> Add` | `FFN -> Add -> LN` |

为了让概念更清楚，先解释三个常见术语：

| 术语 | 精确定义 | 初学者可这样理解 |
|---|---|---|
| `LayerNorm` | 对每个 token 的隐藏向量按特征维做归一化 | 把这一行特征重新拉回可控尺度 |
| `残差连接` | 把输入直接加到子层输出上 | 给网络留一条“原样通过”的路 |
| `训练动力学` | 参数更新过程中梯度、激活、收敛速度的变化规律 | 模型到底是怎么一步步学会的 |

本文的讨论边界是深度至少在 6 层以上的 Transformer。层数越深，这个位置差异越明显。对 2 到 4 层的小网络，两种写法往往都能跑；对 24、48、96 层这种深层结构，差异会快速放大。

一个足够直观的玩具理解是：

- Pre-Norm：先把输入调到稳定尺度，再算修正量，再加回原输入。
- Post-Norm：先把修正量加回去，再把整层结果统一重缩放。

层数少时，这两种写法都可能没问题。层数一多，“每层都多做一次整体缩放”这件事就会累积成优化难度。

再看一个更具体的标量化类比。假设每一层都近似做：
$$
x_{l+1} \approx x_l + \epsilon_l
$$
其中 $\epsilon_l$ 是这一层学到的小修正。

- 对 Pre-Norm，残差主项 $x_l$ 会直接保留下来。
- 对 Post-Norm，输出近似变成
  $$
  x_{l+1} \approx \mathrm{LN}(x_l + \epsilon_l)
  $$
  这一步会把整体数值重新标准化，残差主项不再是“原封不动地保留”。

这就是后文所有训练差异的起点。

---

## 核心机制与推导

先看最关键的问题：反向传播时，梯度到底走哪条路。

`雅可比矩阵` 表示“输入微小变化会怎样影响输出”的线性近似。对初学者来说，不必把它想成抽象矩阵，只需要记住：它描述了一层对梯度的放大、缩小和方向扭曲。

### 1. Pre-Norm 的梯度主路

对 Pre-Norm，
$$
x_{l+1}=x_l+F(\mathrm{LN}(x_l))
$$

对 $x_l$ 求导，可得：
$$
\frac{\partial x_{l+1}}{\partial x_l}
=
I+\frac{\partial F(\mathrm{LN}(x_l))}{\partial x_l}
$$

因此反向传播满足：
$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_{l+1}}
\left(
I+\frac{\partial F(\mathrm{LN}(x_l))}{\partial x_l}
\right)
$$

这里最关键的是这个 $I$。它来自残差连接，表示即使子层部分在初始化时还很差，梯度仍然有一条近似恒等映射的通路可以直接往前传。

把多层展开更清楚。若总共有 $L$ 层，则：
$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{k=l}^{L-1}
\left(
I+\frac{\partial F_k(\mathrm{LN}(x_k))}{\partial x_k}
\right)
$$

只要每层附加项不是特别极端，这个乘积就仍然保留了“恒等项主导”的性质。工程上，这通常意味着更平稳的梯度流。

### 2. Post-Norm 的梯度主路

对 Post-Norm，
$$
x_{l+1}=\mathrm{LN}(x_l+F(x_l))
$$

记
$$
u_l = x_l + F(x_l)
$$
则
$$
x_{l+1} = \mathrm{LN}(u_l)
$$

对 $x_l$ 求导：
$$
\frac{\partial x_{l+1}}{\partial x_l}
=
J_{\mathrm{LN}}(u_l)
\left(
I+\frac{\partial F(x_l)}{\partial x_l}
\right)
$$

于是有：
$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_{l+1}}
\cdot
J_{\mathrm{LN}}(u_l)
\cdot
\left(
I+\frac{\partial F(x_l)}{\partial x_l}
\right)
$$

问题不在“LayerNorm 一定会把梯度变坏”，而在于：

1. 梯度主路无法绕开 `LayerNorm`
2. 深层堆叠时会出现多个 $J_{\mathrm{LN}}$ 连乘
3. 每层哪怕只带来一点点缩放偏差，累计后也会显著改变优化状态

把多层写开：
$$
\frac{\partial \mathcal{L}}{\partial x_l}
=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{k=l}^{L-1}
\left[
J_{\mathrm{LN}}(u_k)
\left(
I+\frac{\partial F_k(x_k)}{\partial x_k}
\right)
\right]
$$

和 Pre-Norm 相比，差别不是有没有残差项，而是残差项前面被乘上了 $J_{\mathrm{LN}}$。

### 3. LayerNorm 的导数为什么会改变训练难度

对单个 token 的隐藏向量 $h \in \mathbb{R}^d$，LayerNorm 可写作：
$$
\mathrm{LN}(h)=\gamma \odot \frac{h-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$
其中：
$$
\mu=\frac{1}{d}\sum_{i=1}^{d}h_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(h_i-\mu)^2
$$

这说明 LayerNorm 不是简单乘一个常数，而是：

- 先减去均值
- 再除以标准差
- 最后乘可学习缩放 $\gamma$、加平移 $\beta$

所以它的雅可比不是单位矩阵。它会改变梯度的尺度，也会耦合不同特征维度。单层看通常可控，但深层连乘后会放大这种影响。

### 4. 一个极简梯度流图

**Pre-Norm**
`x -> LN -> F -> + -> x'`

反向主路：
`x' -> + -> x`

**Post-Norm**
`x -> F -> + -> LN -> x'`

反向主路：
`x' -> LN -> + -> x`

差别就这一点：Pre-Norm 的梯度主路绕开了 LayerNorm，Post-Norm 没绕开。

### 5. 论文现象应该怎么读

Xiong 等人在 2020 年的工作中给出了更具体的理论和实验分析。结论不是“Post-Norm 一定训练失败”，而是：

- 在初始化附近，Post-LN 靠近输出层的梯度往往更大
- 因而它更依赖 warmup 去缓和前期不稳定更新
- Pre-LN 的梯度分布更平滑，对大学习率更宽容

对新手来说，可以把这个现象理解为：

> Pre-Norm 像是给每一层都保留了可退回的默认通道；Post-Norm 则要求所有层在一开始就更配合。

### 6. 为什么大模型更偏向 Pre-Norm

当模型从 12 层增加到 48 层、96 层，训练问题会从“局部模块是否好用”变成“整条梯度链能不能长期稳定”。这也是很多深层语言模型走向 Pre-Norm 的直接原因。

一个简化判断表：

| 层数 | 两者差异是否明显 | 常见现象 |
|---|---|---|
| 2 到 6 层 | 通常不明显 | 都可能稳定训练 |
| 12 到 24 层 | 开始明显 | Post-Norm 更吃 warmup 和学习率 |
| 24 到 48 层 | 明显放大 | Pre-Norm 更容易先训起来 |
| 96 层及以上 | 工程差异很大 | 训练优先时通常更偏 Pre-Norm |

---

## 代码实现

下面先给一个最小版本，只保留“归一化、子层、残差”三件事。

```python
def pre_norm_forward(x, sublayer, norm):
    return x + sublayer(norm(x))

def post_norm_forward(x, sublayer, norm):
    return norm(x + sublayer(x))
```

这两个函数已经表达了本质区别：

- Pre-Norm：`norm(x)` 进入子层，残差 `x` 直接加回
- Post-Norm：先相加，再对总结果做 `norm`

### 1. 一个可运行的 PyTorch block

下面的代码可以直接运行。它实现了一个最小化的 Transformer 风格子层，不依赖额外框架。

```python
import torch
import torch.nn as nn

class ToySublayer(nn.Module):
    def __init__(self, dim, hidden_mul=2):
        super().__init__()
        hidden = dim * hidden_mul
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

class PreNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = ToySublayer(dim)

    def forward(self, x):
        return x + self.sublayer(self.norm(x))

class PostNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = ToySublayer(dim)

    def forward(self, x):
        return self.norm(x + self.sublayer(x))
```

如果你是第一次看这类代码，可以按下面顺序理解：

| 代码部分 | 作用 |
|---|---|
| `ToySublayer` | 代替注意力或 FFN 的“某个变换” |
| `LayerNorm(dim)` | 对最后一维做归一化 |
| `x + ...` | 残差连接 |
| `self.sublayer(self.norm(x))` | Pre-Norm |
| `self.norm(x + self.sublayer(x))` | Post-Norm |

### 2. 一个输出行为对比实验

这个实验验证一个直观事实：Post-Norm 会把输出重新标准化，而 Pre-Norm 不会强制把最终输出拉成零均值。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

batch, dim = 2, 4
x = torch.randn(batch, dim)

norm = nn.LayerNorm(dim)
linear = nn.Linear(dim, dim, bias=False)

with torch.no_grad():
    linear.weight.copy_(torch.eye(dim))

def pre_norm(x):
    return x + linear(norm(x))

def post_norm(x):
    return norm(x + linear(x))

y_pre = pre_norm(x)
y_post = post_norm(x)

pre_mean = y_pre.mean(dim=-1)
post_mean = y_post.mean(dim=-1)

assert not torch.allclose(pre_mean, torch.zeros_like(pre_mean), atol=1e-4)
assert torch.allclose(post_mean, torch.zeros_like(post_mean), atol=1e-5)
assert not torch.allclose(y_pre, y_post)

print("input:")
print(x)
print("pre-norm output:")
print(y_pre)
print("post-norm output:")
print(y_post)
print("ok")
```

这个实验说明的不是“谁更好”，而是“它们前向行为本来就不同”。Post-Norm 每层都会显式把总输出重新归一化，Pre-Norm 不会。

### 3. 一个更关键的梯度实验

下面再给一个可以直接运行的小实验，观察深层堆叠时输入梯度的大小。它不是论文复现，但能帮助建立训练直觉。

```python
import copy
import torch
import torch.nn as nn

torch.manual_seed(42)

class ToySublayer(nn.Module):
    def __init__(self, dim, hidden_mul=4):
        super().__init__()
        hidden = dim * hidden_mul
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

class PreNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = ToySublayer(dim)

    def forward(self, x):
        return x + self.sublayer(self.norm(x))

class PostNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = ToySublayer(dim)

    def forward(self, x):
        return self.norm(x + self.sublayer(x))

def build_stack(block_cls, depth, dim):
    return nn.Sequential(*[block_cls(dim) for _ in range(depth)])

def grad_norm_of_input(model, x):
    x = x.clone().detach().requires_grad_(True)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    return x.grad.norm().item(), loss.item()

batch, seq_len, dim = 8, 16, 64
x = torch.randn(batch, seq_len, dim)

for depth in [2, 6, 12, 24]:
    torch.manual_seed(42)
    pre_model = build_stack(PreNormBlock, depth, dim)

    torch.manual_seed(42)
    post_model = build_stack(PostNormBlock, depth, dim)

    pre_grad, pre_loss = grad_norm_of_input(pre_model, x)
    post_grad, post_loss = grad_norm_of_input(post_model, x)

    print(
        f"depth={depth:>2d} | "
        f"pre_grad={pre_grad:.6f}, post_grad={post_grad:.6f} | "
        f"pre_loss={pre_loss:.6f}, post_loss={post_loss:.6f}"
    )
```

运行后你通常会看到：

- 小深度时，两者差异不一定大
- 深度上来后，梯度规模和 loss 行为开始出现更明显分化
- Post-Norm 对初始化、损失定义更敏感

这个实验不能代替完整训练，但能让你直接看到“结构顺序会改变梯度路径”这件事。

### 4. 一个更接近真实 Transformer 的写法

真实 decoder layer 往往至少有两个子层：注意力和 FFN。写法一般如下。

**Pre-Norm**
```python
x = x + attn(norm1(x))
x = x + ffn(norm2(x))
```

**Post-Norm**
```python
x = norm1(x + attn(x))
x = norm2(x + ffn(x))
```

也可以整理成表：

| 子层 | Pre-Norm 写法 | Post-Norm 写法 |
|---|---|---|
| Attention | `x = x + attn(norm1(x))` | `x = norm1(x + attn(x))` |
| FFN | `x = x + ffn(norm2(x))` | `x = norm2(x + ffn(x))` |

### 5. 初学者最容易漏掉的一点

很多 Pre-LN 实现会在所有 block 堆叠结束后，再加一个最终归一化：

```python
x = blocks(x)
x = final_norm(x)
```

这一步不是形式主义。因为 Pre-Norm 在每个 block 内部把输入先归一化，但 block 的总输出本身没有在残差相加后再次标准化，所以最后常常需要一个 `final_norm` 来整理输出分布。

---

## 工程权衡与常见坑

Pre-Norm 的最大优点是稳。这里的“稳”不是指最终指标必然更高，而是指更容易在训练初期维持可控状态，不容易一上来就出现 loss 抖动、梯度异常或学习率一调大就发散。

| 工程现象 | Pre-Norm | Post-Norm |
|---|---|---|
| 一上来用较大学习率 | 更可能成功 | 更容易不稳定 |
| warmup 步数 | 常可明显缩短 | 常需要更长 |
| 深层扩展到 24+ 层 | 风险较低 | 风险明显上升 |
| 调参成本 | 更低 | 更高 |
| 最终精度 | 稳定但不总是最佳 | 某些任务可能略优 |

这里补一个常被忽略的现实：工程上“更好”通常不是只看最终分数，而是看下面这几项总成本。

| 维度 | Pre-Norm 常见表现 | Post-Norm 常见表现 |
|---|---|---|
| 首次跑通成功率 | 更高 | 更依赖经验 |
| 对优化器容忍度 | 更强 | 更敏感 |
| 调学习率次数 | 更少 | 更多 |
| 训练前期监控压力 | 更小 | 更大 |

常见坑主要有六个。

### 1. 误把“Pre-Norm 更稳定”理解成“Post-Norm 没价值”

这不准确。Post-Norm 不是错误结构，而是更难训的结构。原始 Transformer 就是 Post-Norm，很多浅层模型、成熟配方和特定任务中它依然能工作得很好，有时最终性能并不差。

### 2. 改成 Pre-Norm，却忘了加 final norm

很多初学者只把 block 内部顺序改了，却没检查模型末尾是否还需要一个 `final_norm`。这会导致训练虽然不炸，但输出分布与预期不一致，尤其在语言模型头部前更明显。

### 3. 把“Pre-LN 可减少 warmup”误解成“永远不需要 warmup”

更准确的说法是：Pre-LN 在初始化附近梯度更健康，因此常常可以减少 warmup 依赖。它不是对所有任务、所有 batch size、所有优化器都完全不需要 warmup。

### 4. 以为问题一定出在结构，而忽略了训练配方

如果一个 Post-Norm 模型不稳定，问题不一定是“结构选错了”，也可能是以下因素没配套：

- 学习率过大
- warmup 太短
- 梯度裁剪缺失
- 初始化尺度不合适
- mixed precision 配置过激进

### 5. 只看单层公式，忽略深层累计效应

单看一层，Post-Norm 的 LayerNorm 并不吓人；真正的问题是深层连乘。很多训练问题都不是某一层突然坏掉，而是每层都带来一点点不利因素，最后在 24 层、48 层后累积成大问题。

### 6. 把经验口号当硬规则

“24 层 Post-Norm 一定炸”“Pre-Norm 一定比 Post-Norm 精度高”这类说法都不准确。更稳妥的表达是：

> 层数越深，Post-Norm 对训练超参数越敏感；Pre-Norm 在大多数深层场景里更容易稳定训练。

给一个实际决策例子。假设你在训练 GPT 风格解码器：

- 12 层，数据不大，预算有限：Post-Norm 可能也能顺利收敛。
- 36 层，想尽快把学习率拉起来：Pre-Norm 通常更省调参时间。
- 96 层级别：大多数团队会先选 Pre-Norm，因为首先要确保训练流程足够稳。

如果必须继续用 Post-Norm，至少同步检查四件事：

| 检查项 | 为什么重要 |
|---|---|
| 学习率上限 | 过大时最容易先把后层更新打坏 |
| warmup 步数 | 用来缓和初始化阶段的大梯度 |
| 梯度裁剪 | 防止偶发更新过猛 |
| 初始化尺度 | 影响前期激活和梯度分布 |

---

## 替代方案与适用边界

Pre-Norm 和 Post-Norm 不是二选一的立场问题，而是不同深度、不同训练预算、不同调参资源下的工程选择。

先给一个足够实用的决策表：

| 条件 | 更推荐 |
|---|---|
| 层数小于 12，任务稳定，追求贴近原始结构 | Post-Norm 可尝试 |
| 层数在 12 到 24，调参资源一般 | 优先 Pre-Norm |
| 层数大于 24，尤其是语言模型预训练 | 基本优先 Pre-Norm |
| 经常遇到梯度爆炸、loss 开局发散 | 先改 Pre-Norm |
| 已有成熟 warmup 和小学习率配方 | Post-Norm 仍可保留 |

再补一个“按目标选”的版本：

| 你的首要目标 | 更合适的选择 |
|---|---|
| 先把训练稳稳跑通 | Pre-Norm |
| 研究原始 Transformer 结构 | Post-Norm |
| 训练非常深的模型 | Pre-Norm |
| 复现已有成熟论文配方 | 跟论文保持一致 |
| 排查训练初期发散问题 | 先试 Pre-Norm |

除了这两种基本放置方式，还可以考虑两类替代路线。

### 1. 保留前归一化思路，但换归一化算子

例如 RMSNorm。它只按均方根做缩放，不显式减均值，形式可写作：
$$
\mathrm{RMSNorm}(x)=\gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2+\epsilon}}
$$

它的特点是：

- 计算更简单
- 保留“先归一化再进子层”的稳定思路
- 在很多现代 LLM 中使用广泛

对初学者，可以把它理解成：目标仍然是控制尺度，只是控制方式比 LayerNorm 更简化。

### 2. 保留后归一化，但补额外稳定技巧

例如：

- 更长 warmup
- 更小学习率
- 残差缩放
- 更保守的初始化
- DeepNorm 一类专门面向深层稳定性的设计

这些技巧的共同目标都是同一个：减轻深层堆叠时的梯度失衡问题。

### 3. 本文结论不应过度外推的边界

本文不讨论以下情况，因此结论不能直接机械套用：

| 场景 | 为什么不能直接套用 |
|---|---|
| 极浅网络 | 层数太少，差异可能不明显 |
| 非 Transformer 主体 | 结构不同，梯度路径也不同 |
| 使用 RMSNorm/DeepNorm 的模型 | 归一化与残差设计已改变 |
| 强依赖特定论文配方的复现 | 结构和超参数往往成套出现 |

对初学者，一个足够稳的规则是：

- 小模型、浅层、想先理解原始 Transformer：可以从 Post-Norm 学起。
- 大模型、深层、最怕训练不稳：优先用 Pre-Norm。
- 已经被 warmup、爆梯、loss 抖动折腾过一次：默认先改 Pre-Norm，再看别的技巧。

这条规则不保证全局最优，但在大多数工程场景里足够有效。

---

## 参考资料

1. Xiong et al., 2020, *On Layer Normalization in the Transformer Architecture*。核心贡献：从理论和实验两方面说明 Post-LN 在初始化附近更容易出现靠近输出层梯度偏大的问题，因此更依赖 warmup；Pre-LN 的梯度分布更平滑。链接：<https://proceedings.mlr.press/v119/xiong20b.html>

2. Vaswani et al., 2017, *Attention Is All You Need*。核心贡献：给出原始 Transformer 设计，是理解 Post-Norm 历史背景的起点。链接：<https://arxiv.org/abs/1706.03762>

3. Brown et al., 2020, *Language Models are Few-Shot Learners*。核心贡献：GPT-3 作为深层语言模型代表案例，体现了工程实践中对可训练性和可扩展性的优先考虑。链接：<https://arxiv.org/abs/2005.14165>

4. Michael Brenndoerfer, *Pre-Norm vs Post-Norm*。核心贡献：用非常直观的方式解释残差主路为什么决定训练稳定性，适合作为建立直觉的辅助材料。链接：<https://mbrenndoerfer.com/writing/pre-norm-vs-post-norm>

5. Yiqing Liang, *Why LayerNorm Won*。核心贡献：把 LayerNorm、RMSNorm 与现代大模型的归一化实践联系起来，适合理解为什么训练深层模型时“稳定性优先”会压过“形式贴近原始设计”。链接：<https://lynl7130.github.io/blog/posts/layernorm-vs-batchnorm.html>

6. Wang et al., 2022, *DeepNet: Scaling Transformers to 1,000 Layers*。核心贡献：说明如果坚持把网络做得极深，仅靠“换个位置”往往不够，通常还需要残差缩放等额外稳定化设计。链接：<https://arxiv.org/abs/2203.00555>

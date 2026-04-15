## 核心结论

Sandwich Norm 是一种残差块中的归一化写法，形式是：

$$
x_{l+1} = x_l + LN(F(LN(x_l)))
$$

这里的 `LayerNorm` 首次出现，指的是“按单个样本的特征维度做标准化，让均值接近 0、方差接近 1 的变换”。白话说，它会把一段向量重新拉回比较稳定的数值范围。

它和常见 `Pre-LN` 的区别不在“有没有在子层前做归一化”，而在“子层输出是否再做一次归一化”。`Pre-LN` 是：

$$
x_{l+1} = x_l + F(LN(x_l))
$$

Sandwich Norm 是：

$$
x_{l+1} = x_l + LN(F(LN(x_l)))
$$

核心作用不是提升模型表达能力上限，而是提高训练稳定性，尤其是在深层网络、混合精度训练、长序列或多模态场景中，尽量压住残差分支的尺度漂移。白话说，它更像“限幅器”而不是“加速器”。

| 方案 | 归一化位置 | 残差分支输出是否再校准 | 训练稳定性倾向 | 额外开销 |
| --- | --- | --- | --- | --- |
| Post-LN | 子层后、残差后 | 否 | 深层训练较敏感 | 低 |
| Pre-LN | 子层前 | 否 | 一般比 Post-LN 更稳 | 低 |
| Sandwich Norm | 子层前 + 子层后 | 是 | 通常更稳，尤其深层时 | 略高 |

如果只记一句话，可以记成：`Pre-LN` 解决“进子层前先整理输入”，Sandwich Norm 进一步解决“出子层后再把分支输出拉回稳定尺度”。

---

## 问题定义与边界

问题先说清楚。Transformer 一类模型的基本结构是“主干残差流 + 子层分支”。残差流可以理解为“跨层传递的主通道”，每一层都把一个分支输出加回主通道：

$$
x_{l+1} = x_l + \Delta_l
$$

其中 $\Delta_l$ 是注意力或前馈网络产生的更新量。问题在于，如果 $\Delta_l$ 的尺度在很多层里持续偏大，那么主干中的激活值就可能逐层放大。深层堆叠后，常见后果有三类：

| 现象 | 白话解释 | 常见后果 |
| --- | --- | --- |
| 激活放大 | 某些层输出越来越大 | 后续层输入失衡 |
| 数值溢出 | 半精度表示不下这么大的数 | `inf`、`NaN` |
| 训练发散 | 参数更新方向失控 | loss 突然爆炸 |

这就是 Sandwich Norm 要处理的核心问题：**残差分支的数值尺度失控**。

它属于 residual normalization，也就是“残差结构中的归一化变体”。白话说，不是改模型任务目标，而是改“信息在残差块里怎么被规范化”。

边界也要明确。Sandwich Norm 能改善的是数值稳定性，不是所有训练问题。下面这张表最重要：

| 类别 | 能否直接解决 | 说明 |
| --- | --- | --- |
| 激活逐层放大 | 能部分缓解 | 子层输出会被再次归一化 |
| 深层训练不稳 | 能部分缓解 | 残差分支更不容易失控 |
| fp16 overflow 倾向 | 能降低风险 | 但不能保证完全消失 |
| 数据噪声大 | 不能 | 这属于数据质量问题 |
| 损失函数设计错误 | 不能 | 归一化不改目标函数 |
| 学习率过大 | 不能根治 | 仍需 warmup 和调参 |
| 梯度爆炸 | 不能单独兜底 | 仍可能需要裁剪 |

所以它不是“加了就稳”，而是“在结构层面多加一层保险”。

一个真实工程边界是这样的：如果你训练的是 6 层到 12 层的小模型，数据干净，`bf16` 稳定，loss 曲线本来就平滑，那么额外一层 `LayerNorm` 很可能收益不明显。但如果你在做数十层以上的语言模型、视觉 Transformer、text-to-image 或其他多模态预训练，数值稳定本来就是系统性问题，这时 Sandwich Norm 的价值会明显提高。

---

## 核心机制与推导

先看单个子层。设第 $l$ 层输入是 $x_l$，子层函数是 $A(\cdot)$，它可以是 attention，也可以是 FFN。Sandwich Norm 写成三步：

$$
u_l = A(LN(x_l))
$$

$$
\tilde{u}_l = LN(u_l)
$$

$$
x'_l = x_l + \tilde{u}_l
$$

这三步的含义分别是：

| 步骤 | 数学式 | 白话解释 |
| --- | --- | --- |
| 前归一化 | $LN(x_l)$ | 先把输入整理到稳定尺度 |
| 子层变换 | $u_l = A(LN(x_l))$ | 让注意力或 FFN 计算更新量 |
| 后归一化 | $\tilde{u}_l = LN(u_l)$ | 再把更新量本身压回稳定尺度 |
| 残差相加 | $x'_l = x_l + \tilde{u}_l$ | 把受控的更新量加回主干 |

如果一个 block 里包含 attention 和 FFN 两个子层，则通常分别这样写：

$$
x'_l = x_l + LN(A(LN(x_l)))
$$

$$
x_{l+1} = x'_l + LN(F(LN(x'_l)))
$$

其中 `FFN` 首次出现，指的是“位置独立的前馈网络”，白话说就是“每个 token 各自经过的小型 MLP”。

### 玩具例子

先用一个极小数值例子看“第二次归一化到底在干什么”。

取：

$$
x = [1, 3]
$$

假设 `LayerNorm` 的缩放和平移参数分别是 $\gamma = 1, \beta = 0$，并且为了方便演示令 $\epsilon = 0$。则：

$$
LN([1, 3]) = [-1, 1]
$$

再假设子层函数是一个简单线性放大：

$$
A(z) = 2z
$$

那么：

$$
u = A(LN(x)) = 2 \cdot [-1, 1] = [-2, 2]
$$

如果这是普通 `Pre-LN`，残差更新会直接把 `[-2, 2]` 加回去：

$$
x' = [1,3] + [-2,2] = [-1,5]
$$

而 Sandwich Norm 会先再做一次归一化：

$$
LN([-2,2]) = [-1,1]
$$

于是：

$$
x' = [1,3] + [-1,1] = [0,4]
$$

这个玩具例子说明一件事：子层可能把输出幅度放大，但第二次 `LN` 会把它重新拉回稳定范围。它不改变“方向信息”的基本趋势，但会抑制“幅度无约束增长”。

### 为什么这对深层堆叠有帮助

看残差递推：

$$
x_{L} = x_0 + \sum_{l=0}^{L-1}\Delta_l
$$

如果每层的 $\Delta_l$ 都可能有较大尺度差异，那么和式会越来越不稳定。Sandwich Norm 的思路，是让每个 $\Delta_l$ 在进入主干前都经过一次尺度重整，使它们不至于因为某层子网络输出异常大而污染整个后续层。

当然，`LayerNorm` 不是硬截断，它不会简单把所有值裁成同一个区间。它做的是按样本内部统计量标准化：

$$
LN(x_i) = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

这里 $\mu$ 是均值，$\sigma^2$ 是方差，$\epsilon$ 是数值稳定项。白话说，它会按“当前样本自己的统计特征”重新缩放，而不是依赖 batch 里的其他样本。

### 真实工程例子

在大规模 text-to-image 训练中，模型往往同时处理文本 token、图像 token、交叉注意力和较深的堆叠层数。这类训练常见三个条件同时出现：

| 条件 | 影响 |
| --- | --- |
| 层数深 | 残差累积更明显 |
| mixed precision | 大值更容易溢出 |
| 多模态分布差异大 | 某些层激活范围更容易异常 |

这时，一旦某个 attention block 输出幅度异常大，它会立刻被加到主干里，后续层再在这个放大的基础上继续算，最后表现为 loss 抖动、局部 `NaN`、梯度异常，或者某几步后直接发散。

Sandwich Norm 在这里的价值很直接：它不是让模型“学到更多”，而是尽量确保“每层送回主干的增量不要突然过大”。这也是为什么它常被归类为训练稳定化技巧，而不是能力增强技巧。

---

## 代码实现

先给最小伪代码。一个标准的 Sandwich Norm block 可以写成：

```python
y = x + ln(attn(ln(x)))
x = y + ln(ffn(ln(y)))
```

真正写代码时，关键点有两个：

1. 注意力和 FFN 两个子层都要遵守相同结构。
2. 前后的 `LayerNorm` 最好拆开定义，避免把“共享一个 LN”误写成不清晰逻辑。

下面给一个可运行的 Python 版本。它不是完整 Transformer，只保留结构骨架，用来说明 Sandwich Norm 的执行顺序。

```python
import math

def layer_norm(vec, eps=1e-5, gamma=1.0, beta=0.0):
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / len(vec)
    denom = math.sqrt(var + eps)
    return [gamma * (x - mean) / denom + beta for x in vec]

def linear(vec, weight, bias):
    out = []
    for row, b in zip(weight, bias):
        out.append(sum(v * w for v, w in zip(vec, row)) + b)
    return out

def relu(vec):
    return [max(0.0, x) for x in vec]

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def attention_sublayer(x):
    # 玩具 attention：这里只用线性映射代替真正的 QKV 计算
    w = [
        [1.2, -0.2],
        [0.3, 0.8],
    ]
    b = [0.1, -0.1]
    return linear(x, w, b)

def ffn_sublayer(x):
    w1 = [
        [1.0, 0.5],
        [-0.5, 1.0],
    ]
    b1 = [0.0, 0.2]
    h = relu(linear(x, w1, b1))

    w2 = [
        [0.7, -0.4],
        [0.2, 0.9],
    ]
    b2 = [0.05, -0.05]
    return linear(h, w2, b2)

def sandwich_block(x):
    attn_in = layer_norm(x)
    attn_out = attention_sublayer(attn_in)
    y = add(x, layer_norm(attn_out))

    ffn_in = layer_norm(y)
    ffn_out = ffn_sublayer(ffn_in)
    z = add(y, layer_norm(ffn_out))
    return z

def pre_ln_block(x):
    attn_in = layer_norm(x)
    y = add(x, attention_sublayer(attn_in))

    ffn_in = layer_norm(y)
    z = add(y, ffn_sublayer(ffn_in))
    return z

x = [1.0, 3.0]
sandwich_out = sandwich_block(x)
preln_out = pre_ln_block(x)

assert len(sandwich_out) == 2
assert len(preln_out) == 2
assert all(math.isfinite(v) for v in sandwich_out)
assert all(math.isfinite(v) for v in preln_out)

# LayerNorm 结果应接近零均值
ln_x = layer_norm(x)
assert abs(sum(ln_x) / len(ln_x)) < 1e-6

# Sandwich 结构中的两次 LN 都应存在且输出可控
attn_branch = attention_sublayer(layer_norm(x))
attn_branch_normed = layer_norm(attn_branch)
assert abs(sum(attn_branch_normed) / len(attn_branch_normed)) < 1e-6
```

这个实现里每个张量的含义如下：

| 张量 | 含义 |
| --- | --- |
| `x` | 当前 block 输入，也就是主干残差流 |
| `layer_norm(x)` | 进入子层前被整理过的输入 |
| `attn_out` | 注意力子层产生的更新量 |
| `layer_norm(attn_out)` | 被再次校准后的注意力更新量 |
| `y` | attention 残差相加后的中间结果 |
| `ffn_out` | FFN 子层产生的更新量 |
| `z` | 一个完整 block 的输出 |

和 `Pre-LN` 的代码差异可以压缩成一张表：

| 项目 | Pre-LN | Sandwich Norm |
| --- | --- | --- |
| attention 分支 | `x + attn(ln(x))` | `x + ln(attn(ln(x)))` |
| FFN 分支 | `y + ffn(ln(y))` | `y + ln(ffn(ln(y)))` |
| 新增模块 | 无 | 每个子层多一个输出 LN |
| 目标 | 稳定输入分布 | 稳定输入分布 + 稳定残差增量 |

如果你在真实框架里实现，建议显式定义四个归一化模块：`attn_norm_in`、`attn_norm_out`、`ffn_norm_in`、`ffn_norm_out`。这样最不容易写乱，也方便做 ablation，也就是“控制变量实验，单独比较某一处设计是否有效”。

---

## 工程权衡与常见坑

Sandwich Norm 的代价很明确：每个子层多一次 `LayerNorm`。这意味着多一点算力、多一点内存访问、多一点延迟。在大模型里，这个成本通常可以接受；在极致轻量模型里，就不一定划算。

最常见的误区如下：

| 误区 | 后果 | 正确做法 |
| --- | --- | --- |
| 把 Sandwich Norm 当成 Pre-LN 同义词 | 结构理解错误，实验结论会错 | 明确区分“只有前 LN”和“前后都 LN” |
| 只在 attention 上加，不在 FFN 上加 | block 内部不一致，结果难解释 | 两个子层都统一处理 |
| 以为加了它就能取消 warmup | 训练前期仍可能不稳 | 继续保留 warmup |
| 以为它能替代梯度裁剪 | 极端梯度仍可能爆炸 | 需要时继续启用裁剪 |
| 盲目搬到小模型 | 增加开销但收益不明显 | 先看是否真的存在不稳定问题 |
| 不核对具体论文实现 | 名称相近但实现不同 | 按论文或代码逐项核对 |

这里要特别强调一个边界：有些文章会把“某些架构使用输出端额外归一化”笼统说成 Sandwich Norm，但工程里名称和实现常常不完全一致。比如谈到 RetNet 时，应该核对公开实现或论文描述，不要因为“都在残差周围加归一化”就把不同机制混成同一种写法。

做工程判断时，可以直接看下面这张清单：

| 判断问题 | 如果答案是“是” | 含义 |
| --- | --- | --- |
| 训练中出现 `NaN` 或 overflow 吗 | 值得尝试 | 说明数值稳定性是瓶颈 |
| 模型很深吗 | 值得尝试 | 残差累积风险更高 |
| 使用 fp16/bf16 吗 | 更值得尝试 | 半精度更怕极端大值 |
| 当前 Pre-LN 已经很稳吗 | 可不改 | 结构改动收益可能有限 |
| 对延迟和吞吐极度敏感吗 | 谨慎 | 额外 LN 不是零成本 |

一个真实工程经验是：当你已经有 `warmup + grad clip + loss scaling + Pre-LN`，但训练仍然偶发性炸掉，这时再试 Sandwich Norm 是合理的。反过来，如果你连学习率都还没调对，先加 Sandwich Norm 往往只是掩盖问题，不是解决问题。

---

## 替代方案与适用边界

Sandwich Norm 不是唯一方案。常见替代项包括 `Pre-LN`、`Post-LN`、残差缩放、梯度裁剪、学习率 warmup，以及其他归一化变体。

先做总表对比：

| 方案 | 核心思路 | 优点 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| Post-LN | 残差相加后归一化 | 经典写法 | 深层训练更敏感 | 早期 Transformer |
| Pre-LN | 子层前归一化 | 更容易训练深层模型 | 分支输出仍可能偏大 | 通用默认选择 |
| Sandwich Norm | 子层前后都归一化 | 更强的数值稳健性 | 多一层 LN 成本 | 深层、大模型、不稳定训练 |
| 残差缩放 | 给残差分支乘较小系数 | 简单直接 | 需要调缩放超参 | 很深网络 |
| 梯度裁剪 | 限制梯度范数 | 对梯度爆炸有效 | 不直接管激活尺度 | 训练兜底措施 |
| Warmup | 前期逐步升学习率 | 减少初期发散 | 不能单独解决结构问题 | 几乎所有大模型训练 |

怎么选，可以按问题类型判断：

### 场景一：小模型、短训练、稳定数据

如果是 6 到 12 层的小模型，训练数据分布稳定，`Pre-LN` 已经没有明显数值问题，那么继续用 `Pre-LN` 通常就够了。此时主要瓶颈更可能在数据、参数量或训练时长，而不是归一化位置。

### 场景二：深层大模型、长序列、多模态

如果模型很深，或者不同模态输入尺度差异明显，又或者你已经遇到不稳定现象，那么 Sandwich Norm 比较值得尝试。因为这里的问题往往不是“模型不会学”，而是“模型来不及学就炸了”。

### 场景三：极致轻量部署

如果你追求极限吞吐或边缘设备部署，额外 `LayerNorm` 的代价可能不值。此时更可能优先考虑较轻的稳定化手段，比如谨慎初始化、残差缩放、改学习率曲线，而不是直接增加每层归一化次数。

所以一句话总结适用边界：

| 问题 | 是否优先考虑 Sandwich Norm |
| --- | --- |
| 模型容易炸 | 是 |
| 模型能力不够 | 否 |
| 训练不稳且层数深 | 是 |
| 小模型训练稳定 | 否 |
| 对延迟极敏感 | 谨慎 |
| 想快速做稳健性 ablation | 是 |

不要把它当作“更先进的 Transformer 必备组件”，它只是一个在特定问题下很有价值的结构选择。

---

## 参考资料

1. `CogView: Mastering Text-to-Image Generation via Transformers`
用于追溯 Sandwich-LN 的论文背景和其在大规模训练稳定化中的用途。

2. `Attention Is All You Need`
用于对照 Transformer 原始残差块和归一化位置，理解 Post-LN 的来源。

3. `PyTorch torch.nn.LayerNorm` 文档
用于确认 `LayerNorm` 的精确定义、参数含义和数值行为。

4. `lucidrains/x-transformers`
用于查看社区实现中 `sandwich_norm` 的工程写法和开关方式。

5. `Microsoft Research RetNet` 相关公开介绍
用于了解相邻架构中“归一化位置”和“训练稳定性设计”的工程上下文，但应核对具体实现细节，不要直接等同于标准 Sandwich Norm。

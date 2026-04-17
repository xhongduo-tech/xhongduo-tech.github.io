## 核心结论

RoPE，Rotary Position Embedding，中文通常叫“旋转位置编码”，本质是把“位置信息”直接写进注意力里的 `query` 和 `key`。白话讲，它不是给 token 再额外加一段位置向量，而是让向量本身按位置旋转一个角度。

它最关键的性质是：位置 $i$ 的 query 和位置 $j$ 的 key 做点积时，结果主要由相对距离 $j-i$ 决定，而不是由两个绝对位置编号分别决定。公式写成：

$$
\widetilde q_i^\top \widetilde k_j
=
(R_i q_i)^\top (R_j k_j)
=
q_i^\top R_i^\top R_j k_j
=
q_i^\top R_{j-i} k_j
$$

这里的 $R_m$ 是“旋转矩阵”，可以理解为“把向量旋转 $m$ 倍角度的规则”。

这件事的工程价值很直接：

| 结论 | 含义 | 对模型的价值 |
|---|---|---|
| 相对位置自然进入注意力 | 得分依赖 $j-i$ | 语言中“前后关系”更容易建模 |
| 多频率旋转 | 低频看远处，高频看近处 | 同时保留全局结构和局部细节 |
| 实现便宜 | 只是在 attention 前做逐元素变换 | 不改 Transformer 主体结构 |

玩具例子可以先这样理解：把每个 token 想成一个二维小箭头。第 3 个位置旋转 $3\theta$，第 8 个位置旋转 $8\theta$。两个箭头最后的夹角差是 $5\theta$，所以它们的“对齐程度”只和距离 5 有关。

真实工程里，LLaMA、Mistral、Qwen 这类主流大模型都使用 RoPE。它已经不是实验性技巧，而是现代开源 LLM 的默认组件之一。

---

## 问题定义与边界

Transformer 本身没有顺序感。白话讲，如果只看 self-attention 的矩阵乘法，token 集合更像“袋子”而不是“序列”。所以模型必须引入位置编码。

最早常见方案是“绝对位置编码”，也就是给第 1 个 token、第 2 个 token、第 3 个 token 各加上不同的位置向量。这种方法能告诉模型“我在第几个位置”，但它对“相差几个位置”的表达不够直接，特别是在训练长度之外做推理时，行为容易变差。

RoPE 解决的是这个核心问题：让注意力更自然地表达相对位置。

比如两个 token 的距离是 8，无论它们是 `(2,10)` 还是 `(100,108)`，RoPE 关心的核心量都是这 8 的角度差：

$$
\Delta \phi_n = (j-i)\theta_n
$$

其中 $\theta_n$ 是第 $n$ 个频率基。白话讲，不同维度对会用不同速度旋转，所以同样的距离 8，在高频维度上会显得“转得很多”，在低频维度上会显得“转得较少”。

绝对位置编码和 RoPE 的差别可以先看这张表：

| 方案 | 相对不变性 | 长度外推时的一致性 | 主要问题 |
|---|---|---|---|
| 绝对位置编码 | 弱 | 通常较差 | 训练外位置分布不稳定 |
| RoPE | 强 | 较好，但有限 | 超长外推时角度超出训练分布 |

RoPE 也有边界，不是“无限长上下文”的万能解。它的限制主要来自两个因素。

第一，`base`。它控制频率分布。若 `base` 太小，高频维转得太快，远距离位置容易出现周期重叠；若 `base` 太大，局部区分能力可能下降。

第二，训练窗口。假设模型训练时最大长度是 4096，推理直接喂 8192，那么很多维度上的旋转角度已经落到训练时没见过的区域。即使公式仍然成立，模型参数也未必学会了如何在这些角度上工作。

所以本文讨论的边界是：RoPE 本身解释“为什么能表达相对位置”，以及“为什么直接外推会失效”，但它不意味着模型天然支持任意长上下文。

---

## 核心机制与推导

RoPE 的核心构造是：把 head 维度两两分组，每两个维度组成一个二维平面，然后在这个平面上做旋转。

如果 head 维度是 $d$，那么会有 $d/2$ 个二维块。第 $n$ 个块的频率通常定义为：

$$
\theta_n = 10000^{-2n/d}, \quad n=0,1,\dots,d/2-1
$$

位置为 $m$ 时，对应的二维旋转块是：

$$
\begin{pmatrix}
\cos(m\theta_n) & -\sin(m\theta_n) \\
\sin(m\theta_n) & \cos(m\theta_n)
\end{pmatrix}
$$

把所有块沿对角线拼起来，就得到整个位置 $m$ 的旋转矩阵 $R_m$。

为什么它能得到相对位置？因为旋转矩阵满足：

$$
R_i^\top R_j = R_{j-i}
$$

证明并不复杂。二维旋转矩阵的转置等于反向旋转：

$$
R_i^\top = R_{-i}
$$

而旋转可以叠加：

$$
R_{-i}R_j = R_{j-i}
$$

所以有：

$$
(R_i q_i)^\top (R_j k_j)
=
q_i^\top R_i^\top R_j k_j
=
q_i^\top R_{j-i} k_j
$$

这说明注意力分数天然带有“相对位移”结构。

下面给一个玩具例子。设 $d=4$，也就是两个二维块；`base=10000`。则：

$$
\theta_0 = 10000^0 = 1,\qquad
\theta_1 = 10000^{-1/2} = 0.01
$$

如果位置差 $\Delta m=2$，那么两个块的相对旋转角分别是：

$$
2\theta_0 = 2,\qquad 2\theta_1 = 0.02
$$

这正体现了 RoPE 的多尺度特征：

| 维度块 | 频率 | 距离为 2 时角度变化 | 作用倾向 |
|---|---|---|---|
| 第 1 块 | 高频 | 大 | 区分局部顺序 |
| 第 2 块 | 低频 | 小 | 保留远距稳定性 |

白话讲，高频块像“近距离放大镜”，低频块像“远距离地图”。

再看真实工程例子。假设一个 7B 模型训练时最大长度是 4096，推理想扩到 8192。直接用原始 RoPE，会让很多维度上的相对角度翻倍。模型虽然结构没变，但它在训练时没有见过这些更大的角度分布，所以困惑度可能明显上升，长文检索、跨段引用、代码补全等任务最容易先出问题。

这就是为什么后来会出现 NTK-aware scaling。它的想法不是改 attention 公式，而是改频率基的分布，把原来的 `base=b` 调整为：

$$
b' = b \cdot s^{d/(d-2)}, \qquad s=\frac{L_{\text{target}}}{L_{\text{train}}}
$$

白话讲，目标上下文越长，就把整体频率拉低一些，让角度增长变慢。这样扩到更长长度时，相邻位置在高频维上的“扭曲”不会突然跳太快，从而减少训练外分布漂移。

---

## 代码实现

工程里通常不会真的构造完整矩阵 $R_m$，因为那样既慢又占内存。实际做法是预先算好每个位置、每个频率的 `cos` 和 `sin`，再用向量重排实现旋转。

下面给一个可运行的 Python 版本，演示 RoPE 的最小实现和相对位置性质：

```python
import math


def build_inv_freq(head_dim, base=10000.0):
    assert head_dim % 2 == 0
    return [1.0 / (base ** (i / head_dim)) for i in range(0, head_dim, 2)]


def rope_angles(position, inv_freq):
    return [position * w for w in inv_freq]


def apply_rope(vec, position, base=10000.0):
    head_dim = len(vec)
    assert head_dim % 2 == 0
    inv_freq = build_inv_freq(head_dim, base)
    angles = rope_angles(position, inv_freq)

    out = []
    for pair_idx, angle in enumerate(angles):
        x1 = vec[2 * pair_idx]
        x2 = vec[2 * pair_idx + 1]
        c = math.cos(angle)
        s = math.sin(angle)
        out.extend([x1 * c - x2 * s, x1 * s + x2 * c])
    return out


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def approx_equal(x, y, eps=1e-9):
    return abs(x - y) < eps


q = [1.0, 2.0, 3.0, 4.0]
k = [0.5, -1.0, 2.0, 1.5]

# 绝对位置分别为 i 和 j
i, j = 3, 8
lhs = dot(apply_rope(q, i), apply_rope(k, j))

# 相对位置只与 j-i 有关：把 q 留在原位，只把 k 旋转到 j-i
rhs = dot(q, apply_rope(k, j - i))

assert approx_equal(lhs, rhs), (lhs, rhs)

# 同样的相对距离，换一组绝对位置，结果仍成立
i2, j2 = 10, 15
lhs2 = dot(apply_rope(q, i2), apply_rope(k, j2))
rhs2 = dot(q, apply_rope(k, j2 - i2))
assert approx_equal(lhs2, rhs2), (lhs2, rhs2)

print("RoPE relative position property holds.")
```

这段代码里最关键的是“二维旋转”：

$$
(x_1, x_2)
\mapsto
(x_1\cos\phi - x_2\sin\phi,\; x_1\sin\phi + x_2\cos\phi)
$$

在 PyTorch 实现里，通常会把偶数位和奇数位拆开，再做一次“90 度交换”。常见写法可以概括成：

| 步骤 | 操作 | 作用 |
|---|---|---|
| 1 | 生成 `inv_freq` | 定义每个二维块的频率 |
| 2 | `freqs = positions * inv_freq` | 得到每个位置的角度 |
| 3 | 计算 `cos/sin` | 做旋转所需的三角项 |
| 4 | `rotate_half(x)` | 把 `(x1,x2)` 变成 `(-x2,x1)` |
| 5 | `x * cos + rotate_half(x) * sin` | 完成旋转 |

简化后的伪代码如下：

```python
# x shape: [batch, heads, seq, head_dim]
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # (-x2, x1) 对应二维平面中的 90 度旋转基
    return interleave(-x2, x1)

inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
freqs = outer(positions, inv_freq)   # [seq, head_dim/2]
cos = repeat_interleave(cos(freqs), 2, dim=-1)
sin = repeat_interleave(sin(freqs), 2, dim=-1)

q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

复杂度仍然是 $O(Td)$，其中 $T$ 是序列长度，$d$ 是 head 维度。和 attention 主体的 $O(T^2)$ 相比，这部分额外开销非常小，所以 RoPE 才能在大模型里广泛落地。

---

## 工程权衡与常见坑

RoPE 的优点明确，但工程上最容易出问题的也是它的“角度分布”。

先看最常见的坑：

| 风险 | 症状 | 规避 |
|---|---|---|
| 直接超长外推 | PPL 上升，长文检索失败 | 用 NTK-aware scaling 或 YaRN |
| 高频维失效 | 邻近 token 区分变差 | 调整 `base` 或缩放策略 |
| 只看 PPL 不看任务 | 表面指标还行，召回很差 | 加长上下文检索与跨段引用评测 |
| 频率实现写错 | 输出抖动、训练不收敛 | 检查偶奇维配对与 `rotate_half` |

为什么直接外推会坏？可以把角度随距离增长想成一组不同斜率的直线：

$$
\phi_n(\Delta m)=\Delta m \cdot \theta_n
$$

高频维的斜率大，距离一长，角度就飞快增大。训练时如果模型只见过 4096 以内的角度区域，推理突然上到 8192，它在高频部分看到的是完全不同的分布。结果往往是：局部顺序判断先坏，随后长文对齐能力下降。

一个典型的真实工程例子是：团队训练了 4k 上下文模型，业务突然要求支持 8k 文档问答。如果只改推理端的最大长度参数，模型通常会出现“前文还能记住，后半段引用错位”的现象。更稳妥的做法是引入 RoPE 缩放，例如按 $s=2$ 设置动态 scaling，让频率整体变慢。很多实践报告里，这种无微调扩容能把困惑度增幅控制在较小范围内，同时显著改善长文检索。

但这里也要强调一个边界：NTK-aware scaling 不是免费午餐。它本质上是在“保持更长上下文稳定”与“保留原始局部细节”之间做折中。缩得太多，局部区分度会被压平；缩得太少，超长位置又会跳出训练分布。

另一个常见误区是误以为“RoPE 既然表达相对距离，就天然适合任意稀疏位置”。这不准确。标准 RoPE 假设位置编号是连续推进的。如果中间有大段缺失、压缩、拼接，或者跨文档拼装，原始位置编号未必仍然符合你的语义距离，此时可能需要额外的 mask、segment 设计，或更适合长上下文重排的变体。

所以工程上不要只问“能不能开到 32k”，还要问三件事：

1. 训练窗口是多少。
2. 评测任务是生成、检索还是代码补全。
3. 缩放后局部与远距能力分别损失了多少。

---

## 替代方案与适用边界

如果需求只是“正常的 LLM 位置编码”，原始 RoPE 已经足够好。但如果需求更极端，就要看变体。

下面先给一个对比表：

| 方案 | 上下文扩展能力 | 相对角度精度 | 额外计算/改动 | 适合场景 |
|---|---|---|---|---|
| 原始 RoPE | 中 | 高 | 低 | 标准 4k/8k/32k LLM |
| NTK-aware scaling / YaRN | 高 | 中 | 低到中 | 需要快速扩到 32k/64k/128k |
| DRoPE | 不以扩长为主 | 更强调方向一致性 | 中 | 需要几何方向语义的任务 |

YaRN 可以理解为“为了长上下文做的更系统缩放策略”。白话讲，它通过插值和外推组合，让原本训练在短窗口上的模型更平滑地适配超长上下文。它很适合业务上“先把 32k 或 128k 跑起来”的场景，但通常需要明确配置缩放倍率，且最好配合少量验证集调参。

DRoPE，Directional RoPE，可以理解为“更强调方向几何一致性”的版本。它不是主要为长上下文服务，而是更适合需要精确方向关系的任务，比如显式空间方向、多智能体朝向编码、结构化轨迹建模等。如果你做的是普通文本 LLM，DRoPE 通常不是第一选择。

可以用一个很短的决策逻辑来选：

| 问题 | 建议 |
|---|---|
| 是否需要把窗口从 4k 扩到 32k 以上？ | 先看 NTK-aware scaling 或 YaRN |
| 是否主要关心几何方向一致性，而不是长文本？ | 看 DRoPE |
| 是否只是标准文本建模，窗口需求在训练长度附近？ | 原始 RoPE 即可 |

所以适用边界可以概括成一句话：原始 RoPE 解决“相对位置建模”，缩放方案解决“超长上下文工程落地”，而几何变体解决“特殊结构任务”。

---

## 参考资料

1. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*.  
2. Michael Brenndoerfer, *NTK-aware Scaling: Extending Context Length in LLMs*.  
3. Emergent Mind 关于 RoPE、RoPE 长上下文扩展、base 与上下文长度关系的综述页面。  
4. LLaMA、Mistral、Qwen 等开源模型架构分析资料，关于其默认使用 RoPE 的说明。  
5. 关于 YaRN、DRoPE 等 RoPE 变体的技术综述与实现讨论。

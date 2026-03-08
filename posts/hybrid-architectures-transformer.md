## 核心结论

混合架构的本质，不是简单把 Transformer、RNN、CNN 叠在一起，而是把不同网络自带的结构偏置组合起来。这里的“结构偏置”，指模型天生更容易学习哪一类模式：CNN 偏向局部邻域，RNN/LSTM 偏向顺序状态，GNN 偏向图关系，Transformer 偏向大范围的信息交互。

对初学者，一个最容易建立直觉的视角是“三层滤镜”：

1. CNN 先看局部，提取短距离纹理、边缘、邻近 token 模式。
2. RNN/LSTM 再看顺序，保留时间方向上的状态记忆。LSTM 可以理解为“带可学习开关的记忆单元”，它会决定哪些旧信息保留，哪些旧信息丢弃。
3. Transformer 最后做全局整合，让远距离位置直接交互。

这种设计的目标，不是模块越多越好，而是职责划分更清楚。纯 Transformer 的优势是全局依赖建模，但在长序列、局部结构很强的数据、或者天然带图结构的数据上，只靠全连接注意力往往成本高，而且缺少明确的局部归纳偏置。混合架构的价值，就是让 CNN/RNN/GNN 先完成压缩、筛选、预编码，再把计算预算留给真正需要全局推理的位置。

从统一公式看，很多混合设计都可以写成“注意力分数 + 结构偏置 + 掩码限制”：

$$
S_{ij}=\frac{Q(i)K(j)^\top}{\sqrt{d_k}}+\text{bias},\quad
A_{ij}=\text{softmax}(S_{ij}+M_{ij})
$$

其中：

| 符号 | 含义 | 初学者理解 |
| --- | --- | --- |
| $Q(i)$ | 第 $i$ 个位置的 Query | 当前位置“拿着什么问题去找信息” |
| $K(j)$ | 第 $j$ 个位置的 Key | 第 $j$ 个位置“提供什么索引线索” |
| $\text{bias}$ | 额外结构偏置 | 人工加入“更偏好局部/图邻接/相对位置”的倾向 |
| $M_{ij}$ | 掩码 | 决定第 $i$ 个位置能不能看第 $j$ 个位置 |

Longformer 的滑窗加全局 token，是层内混合；CNN+Transformer 常见于视觉，是阶段混合；GNN+Transformer 常见于分子、知识图谱、交通网络，是结构混合。T5 体现的是阶段职责划分，GPT-J 则更接近纯 Transformer 工程化延展。这些设计共同说明一个现实结论：工程里真正有效的架构，通常不是“纯粹”，而是“分工明确”。

---

## 问题定义与边界

本文讨论的“混合架构”，指在 Transformer 主干之外，引入 CNN、RNN/LSTM 或 GNN，并通过串联、并联、交织三种方式之一完成融合。

边界先明确：

| 模块 | 主要职责 | 优势 | 常见代价 |
| --- | --- | --- | --- |
| CNN | 建模局部邻域模式 | 参数共享强，局部特征提取稳定 | 感受野有限，远距离关系弱 |
| RNN/LSTM | 建模时间顺序与状态传递 | 顺序信息明确，适合流式输入 | 长序列难并行，训练慢 |
| Transformer | 建模全局依赖 | 任意位置直接交互，表达力强 | $O(L^2)$ 注意力成本高 |
| GNN | 建模图结构关系 | 节点和边的归纳偏置明确 | 图过大时传播和采样复杂 |

这里最核心的问题不是“哪个模型更强”，而是“当前任务缺少什么偏置”。

举一个统一的任务设定。假设输入长度为 $L=1024$ 的文本序列，并且每个位置还绑定一个局部视觉块，或者一个图结构中的邻接节点。此时如果直接让 Transformer 全量建模，模型需要同时完成三件事：

1. 学局部模式，例如短语、边缘、纹理、邻域片段。
2. 学结构关系，例如先后顺序、节点连接、区域邻接。
3. 承担全局交互的计算成本。

更合理的做法通常是：

1. 先用 CNN 压缩局部纹理或短片段模式。
2. 再用 RNN 或 GNN 处理时间链路或显式图结构。
3. 最后交给 Transformer 做全局推理和跨块对齐。

这说明混合架构不是为了“把所有模型都用上”，而是为了限制每个模块的责任范围。任务没有顺序依赖，就没必要硬加 RNN；数据不是图，就不必强行接 GNN；输入很短，比如几十个 token，全局注意力本身就不贵，混合带来的收益通常也会下降。

可以用一个玩具例子帮助建立直觉。假设要判断一段长度为 20 的字符序列中，是否同时出现模式 `abc` 和很远位置的模式 `xyz`。

| 子任务 | 更自然的模块 | 原因 |
| --- | --- | --- |
| 找出 `abc`、`xyz` 这种短模式 | CNN | 卷积核天然擅长扫描局部片段 |
| 判断 `abc` 是否出现在 `xyz` 之前 | RNN/LSTM | 状态沿时间推进，顺序信息显式 |
| 判断两个远距离模式是否共同出现 | Transformer | 任意位置可直接交互 |

这就是职责分工的最小示例。混合架构的意义，在于把局部检测、顺序约束、全局关联拆开处理，而不是让一个模块同时学习全部规则。

---

## 核心机制与推导

混合机制通常有三类：串联、并联、交织。

第一类是串联。先由一个模块做预处理，再把结果交给另一个模块。例如 CNN 提取 patch 特征后送入 Transformer，或者 GNN 更新节点表示后送入 Transformer。这类方法实现最直接，适合“先局部、后全局”的任务。

第二类是并联。同一输入同时走多条分支，最后做拼接、加权或门控融合。门控可以理解为可学习的分流器，它决定当前样本更应该相信哪条分支。

第三类是交织。不同模块在层内或层间交替出现，例如几层局部卷积后插一层全局注意力，或者一层注意力后接图消息传递。交织最灵活，但训练也最容易不稳定，因为信号路径更多，梯度更难平衡。

为了把三种方式讲清楚，可以先看统一表达：

$$
h^{(l+1)} = \mathcal{F}_{\text{mix}}\left(h^{(l)}\right)
$$

其中 $\mathcal{F}_{\text{mix}}$ 可以有不同形式。

如果是串联：

$$
h^{(l+1)}=\mathcal{T}(\mathcal{C}(h^{(l)}))
$$

这里 $\mathcal{C}$ 表示 CNN/RNN/GNN 等局部或结构模块，$\mathcal{T}$ 表示 Transformer。意思是先局部编码，再全局整合。

如果是并联：

$$
h^{(l+1)}=\alpha \cdot \mathcal{C}(h^{(l)})+\beta \cdot \mathcal{R}(h^{(l)})+\gamma \cdot \mathcal{T}(h^{(l)})
$$

并且：

$$
\alpha,\beta,\gamma \ge 0,\quad \alpha+\beta+\gamma=1
$$

这就是门控融合的形式。三个权重由模型学习，分别表示局部分支、顺序分支、全局分支的贡献比例。

如果是交织：

$$
h^{(l+1)}=\mathcal{T}\big(\mathcal{C}(h^{(l)})\big),\quad
h^{(l+2)}=\mathcal{G}\big(h^{(l+1)}\big)
$$

这里可以先插入卷积层，再做注意力，再做图消息传递。交织的核心不是公式本身，而是“不同偏置在不同层反复出现”。

### Longformer：层内混合的典型例子

Longformer 是理解“层内混合”的典型案例。它不是简单把 Transformer 整体换掉，而是在单层注意力里同时保留局部窗口和少量全局 token。设序列长度为 $L$，窗口宽度为 $w$，全局 token 集合为 $\mathcal{G}$，则：

$$
S_{ij}=\frac{Q(i)K(j)^\top}{\sqrt{d_k}}+\text{bias}
$$

$$
M_{ij}=
\begin{cases}
0,& |i-j|\le w/2 \ \text{或}\ i\in\mathcal{G}\ \text{或}\ j\in\mathcal{G}\\
-\infty,& \text{otherwise}
\end{cases}
$$

然后：

$$
A_{ij}=\text{softmax}(S_{ij}+M_{ij})
$$

这三个式子表达的含义很直接：

1. 先计算普通注意力分数。
2. 再通过掩码决定哪些位置允许连接。
3. 不在窗口里、也不属于全局 token 的位置，分数直接变成 $-\infty$，softmax 后概率近似为 0。

于是原本完整的 $L\times L$ 交互，被限制成“局部滑窗 + 少量全局连接”。

### 复杂度为什么会下降

全连接注意力的主要成本是：

$$
O(L^2)
$$

如果 $L=1024$，则需要比较的配对数大约为：

$$
1024^2 = 1{,}048{,}576
$$

如果只保留窗口 $w=64$ 的局部交互，主要成本变成：

$$
O(Lw)=1024\times64=65{,}536
$$

二者相差约 16 倍。若再加入少量全局 token，复杂度近似变成：

$$
O(Lw + L|\mathcal{G}|)
$$

只要 $|\mathcal{G}| \ll L$，总成本仍显著低于全连接注意力。

### 为什么这仍然属于混合架构

Longformer 没有显式引入 CNN 或 RNN，但它把两种感受野放到了同一层里：

| 组成部分 | 承担角色 | 对应偏置 |
| --- | --- | --- |
| 滑动窗口注意力 | 局部交互 | 类似卷积式局部偏置 |
| 全局 token | 长距离汇聚 | 类似全局摘要节点 |

所以它本质上仍然是混合，只不过混合发生在注意力内部，而不是做成外部子网络。

### T5 与 GPT-J 的位置

再看 T5 和 GPT-J。T5 采用统一的 text-to-text 编码器/解码器框架，核心不是显式加入 CNN 或 RNN，而是通过编码器与解码器的职责划分，让输入表示构建与输出生成分开进行。GPT-J 则保持标准自回归 Transformer 主干和固定上下文窗口，更接近纯 Transformer 的工程扩展。

这两者提醒一个重要边界：不是每个成功模型都必须显式混合额外模块。只有当任务需要更强局部偏置、更低长序列成本、或者显式结构建模时，混合设计才会明显变得合理。

### 一个更接近真实业务的例子

以遥感或农业监测为例。卫星图像往往同时包含三类信息：

| 信息类型 | 例子 | 更适合的模块 |
| --- | --- | --- |
| 局部纹理 | 叶片边缘、病斑细节、道路纹理 | CNN |
| 全局布局 | 地块分布、区域关联、跨区域变化 | Transformer |
| 显式结构 | 灌溉连通、地块邻接、传感器网络 | GNN |

这时用单一模块很难同时高效处理三类结构。更自然的方案是：

1. CNN 先提取局部纹理。
2. Transformer 再处理大范围上下文。
3. GNN 额外建模区域节点之间的显式关系。

收益不在于某一个分支“更强”，而在于三种偏置分别负责自己最擅长的部分。

---

## 代码实现

下面给出一个可运行的玩具实现，用 Python 模拟“CNN 分支 + RNN 分支 + Transformer 分支”的门控融合。这里不依赖深度学习框架，但代码可以直接运行，重点是把“分支输出如何对齐、如何做门控、如何验证权重是否正常”写清楚。

```python
import math
from typing import List, Tuple


Vector = List[float]
Matrix = List[Vector]


def softmax(xs: Vector) -> Vector:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def check_same_length(*vectors: Vector) -> None:
    if not vectors:
        raise ValueError("at least one vector is required")
    n = len(vectors[0])
    for v in vectors[1:]:
        if len(v) != n:
            raise ValueError("all vectors must have the same length")


def weighted_sum(vectors: List[Vector], weights: Vector) -> Vector:
    if len(vectors) != len(weights):
        raise ValueError("number of vectors and weights must match")
    check_same_length(*vectors)

    dim = len(vectors[0])
    out = [0.0] * dim
    for w, vec in zip(weights, vectors):
        for i in range(dim):
            out[i] += w * vec[i]
    return out


def fuse_branches(
    cnn_feats: Vector,
    rnn_feats: Vector,
    transformer_feats: Vector,
    gate_logits: Vector,
) -> Tuple[Vector, Vector]:
    if len(gate_logits) != 3:
        raise ValueError("gate_logits must contain exactly 3 values")

    weights = softmax(gate_logits)
    fused = weighted_sum(
        [cnn_feats, rnn_feats, transformer_feats],
        weights,
    )
    return fused, weights


def pretty(xs: Vector, ndigits: int = 4) -> Vector:
    return [round(x, ndigits) for x in xs]


if __name__ == "__main__":
    # 三个分支都输出 4 维表示，现实中通常是同一个 hidden size
    cnn_feats = [0.8, 0.6, 0.1, 0.0]          # 局部模式更强
    rnn_feats = [0.2, 0.9, 0.3, 0.4]          # 顺序记忆更强
    transformer_feats = [0.7, 0.2, 0.8, 0.9]  # 全局关系更强

    fused, weights = fuse_branches(
        cnn_feats,
        rnn_feats,
        transformer_feats,
        gate_logits=[1.2, 0.8, 1.0],
    )

    assert len(fused) == 4
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in weights)

    # 当门控明显偏向 CNN 分支时，输出会更接近 cnn_feats
    fused2, weights2 = fuse_branches(
        cnn_feats,
        rnn_feats,
        transformer_feats,
        gate_logits=[5.0, 0.1, 0.1],
    )

    assert weights2[0] > 0.98
    assert abs(fused2[0] - cnn_feats[0]) < abs(fused[0] - cnn_feats[0])

    print("weights =", pretty(weights))
    print("fused   =", pretty(fused))
    print("weights2=", pretty(weights2))
    print("fused2  =", pretty(fused2))
```

这段代码的作用可以按四步理解：

1. 三个分支分别产生一个同维度向量。
2. `gate_logits` 表示模型对三条分支的原始偏好分数。
3. `softmax` 把分数转成和为 1 的权重。
4. 最终输出是三条分支的加权和。

如果手动计算第一组门控：

$$
\text{softmax}([1.2, 0.8, 1.0]) \approx [0.402, 0.269, 0.329]
$$

那么融合后的第一个维度约为：

$$
0.402\times0.8 + 0.269\times0.2 + 0.329\times0.7 \approx 0.606
$$

这说明门控融合不是“选一个分支”，而是按比例汇总信息。

### 一个最小可运行的 PyTorch 版本

下面再给一个真正可运行的 PyTorch 示例。它不是完整训练脚本，但模块本身可直接前向执行，适合初学者理解“如何对齐维度并做门控”。

```python
import torch
import torch.nn as nn


class HybridBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.local_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.trm_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.gate = nn.Linear(hidden_dim * 3, 3)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, input_dim]

        # 1. 局部分支：这里用线性层近似局部分支接口，真实工程可换成 Conv1d
        cnn_feats = torch.tanh(self.local_proj(x))  # [B, L, H]

        # 2. 顺序分支：GRU 输出每个时刻的隐藏状态
        rnn_feats, _ = self.rnn(x)  # [B, L, H]

        # 3. 全局分支：先映射到 hidden_dim，再做自注意力
        trm_in = self.trm_proj(x)
        trm_feats, _ = self.attn(trm_in, trm_in, trm_in)  # [B, L, H]

        # 4. 用序列平均表示产生门控
        pooled = torch.cat(
            [
                cnn_feats.mean(dim=1),
                rnn_feats.mean(dim=1),
                trm_feats.mean(dim=1),
            ],
            dim=-1,
        )  # [B, 3H]

        gate = torch.softmax(self.gate(pooled), dim=-1)  # [B, 3]

        # 5. 按门控融合三条分支
        hybrid = (
            gate[:, 0].unsqueeze(-1).unsqueeze(-1) * cnn_feats
            + gate[:, 1].unsqueeze(-1).unsqueeze(-1) * rnn_feats
            + gate[:, 2].unsqueeze(-1).unsqueeze(-1) * trm_feats
        )

        # 6. 残差保底，避免融合层一开始破坏主干
        out = self.norm(hybrid + trm_feats)
        return out, gate


if __name__ == "__main__":
    torch.manual_seed(0)

    model = HybridBlock(input_dim=8, hidden_dim=16, num_heads=4)
    x = torch.randn(2, 5, 8)  # batch=2, seq_len=5, input_dim=8
    out, gate = model(x)

    assert out.shape == (2, 5, 16)
    assert gate.shape == (2, 3)
    assert torch.allclose(gate.sum(dim=-1), torch.ones(2), atol=1e-6)

    print("out shape :", out.shape)
    print("gate      :", gate)
```

### 实现时最容易忽略的三个点

第一，先统一维度。CNN 输出常是空间张量，RNN 输出常是时间序列张量，Transformer 输出常是 token 表示。融合前必须对齐到同一个 hidden size，否则无法安全相加或拼接。

第二，要保留残差。残差可以理解为“给主干留一条直通车”，避免融合层在训练早期把原本可用的表示破坏掉。

第三，要让门控可学习。如果直接把三条分支简单相加，数值尺度更大的分支往往会主导输出，这不一定代表它真的更有用。门控的作用，就是让模型自己学习“当前样本该更相信谁”。

---

## 工程权衡与常见坑

混合架构最常见的问题，不是“完全不能跑”，而是“可以训练，但收益不稳定”。

| 坑 | 表现 | 原因 | 缓解手段 |
| --- | --- | --- | --- |
| CNN/RNN 分支参数过大 | Transformer 梯度变小，主干几乎不学 | 额外分支过强，主干被边缘化 | 缩小分支宽度，给主干单独学习率，加入门控 |
| 全局 token 过强 | 后层注意力被少数 token 垄断 | 模型把全局 token 当捷径 | 限制全局 token 数量，重采样全局集合，做 attention dropout |
| 多分支尺度不一致 | loss 波动大，训练前期不收敛 | 各分支输出分布差异太大 | 每个分支后加 `LayerNorm` 或投影层 |
| 串联过深 | 延迟高，显存占用大 | 中间特征分辨率太高、层数太多 | 将局部模块前移，尽早降采样 |
| GNN 传播层数过多 | 节点表示过平滑，区分度下降 | 多跳传播把节点表征拉得太近 | 减少消息传递层数，保留跳连 |
| 稀疏注意力窗口太小 | 长距离信息仍传不过去 | 局部窗口不足以跨段传递信息 | 增大全局 token 或增加跨层跳跃连接 |

实践里最先检查的，不应该只是最终指标，而应该先看中间统计量：

| 该看什么 | 为什么重要 | 异常时可能说明什么 |
| --- | --- | --- |
| 各分支梯度范数 | 看谁在主导训练 | 某分支吃掉了学习预算 |
| 各分支门控均值 | 看模型长期偏向哪条路 | 某条分支几乎被废弃 |
| 全局 token 注意力占比 | 看是否形成捷径 | 全局节点过强，普通 token 表达被压制 |
| 分支输出均值/方差 | 看尺度是否失衡 | 融合前归一化不足 |

另一个常见误区，是把混合架构当成“万能增强器”。如果数据规模很小，或者标签噪声很大，多加一个分支通常只会增加不稳定性。原因很简单：混合架构的自由度更高，意味着它对数据量、超参数、初始化方式都更敏感。

对于初级工程师，一个更稳妥的实践顺序通常是：

1. 先做最小可行版本，只保留一条额外分支。
2. 先固定 Transformer 主干，再逐步放开门控学习。
3. 记录各分支权重、梯度范数、注意力分布，不只盯着最终准确率。
4. 如果收益不稳定，先减模块，再谈加模块。

原因不复杂。混合系统的大多数问题，不是理论本身错了，而是某一条路径在训练中偷偷吃掉了容量、梯度或者注意力预算。

---

## 替代方案与适用边界

不是每个任务都需要显式混合 CNN/RNN/GNN。有时更轻量的替代方案已经足够。

| 替代方案 | 适用边界 | 代表模型 | 为什么可能更合适 |
| --- | --- | --- | --- |
| 轻量 Transformer + 局部模块 | 端侧部署、移动设备、低时延 | MobileViT | 把局部偏置做进轻量结构，推理成本更可控 |
| 稀疏注意力 Transformer | 长文档、长上下文文本 | Longformer、BigBird | 不必额外引入 RNN/CNN，也能降低长序列成本 |
| 线性注意力 Transformer | 极长序列、显存受限 | Performer 一类 | 主要目标是把注意力复杂度压低 |
| GNN + Transformer 两阶段 | 图结构明确，节点关系强 | Graphormer 方向 | 先吃掉图邻接，再做全局关系整合 |
| CNN 主干 + 全局注意力头 | 图像局部特征占主导 | ViT 混合变体 | 局部模式明显时，比三分支更简单稳妥 |

如果目标是边缘设备部署，MobileViT 这类模型通常比“大而全的 CNN+Transformer+RNN”更现实，因为它在结构设计阶段就考虑了局部偏置与推理成本，而不是事后拼接多个大模块。

如果任务是结构化图数据，例如分子图、交通网络、知识图谱，更自然的方案通常是先用 GNN 生成节点表示，再交给 Transformer 做全局关系建模。因为图的邻接关系本身就是一手先验，先用 GNN 吃掉这部分信息，比让 Transformer 从头学习邻接更高效。

如果数据本质上结构较弱，比如普通中短文本分类，RNN 的价值会明显下降。此时更实际的路线往往是稀疏注意力或线性注意力扩展，再配一个小型局部分支补短模式，而不是直接构造复杂三分支系统。

可以把适用边界压缩成一张判断表：

| 任务特征 | 更可能合适的方案 |
| --- | --- |
| 局部模式非常强，远距离依赖一般 | CNN + Transformer |
| 顺序约束强，且存在流式输入 | RNN/LSTM + Transformer |
| 图结构是核心信息源 | GNN + Transformer |
| 主要问题是上下文太长 | 稀疏/线性注意力 Transformer |
| 输入短、结构简单、数据量有限 | 纯 Transformer 或单一轻量增强 |

原则始终一致：只为真实缺失的偏置付费，不为“看起来更复杂”付费。

---

## 参考资料

1. Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer*. arXiv:2004.05150. https://arxiv.org/abs/2004.05150
2. Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
3. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. https://arxiv.org/abs/2010.11929
4. Wu et al. *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*. ICLR 2022. https://arxiv.org/abs/2110.02178
5. Ying et al. *Do Transformers Really Perform Bad for Graph Representation?* Graphormer. NeurIPS 2021. https://arxiv.org/abs/2106.05234
6. Emergent Mind, Hybrid Transformer Architectures: https://www.emergentmind.com/topics/hybrid-transformer-architectures
7. Emergent Mind, Hybrid Transformer-CNN Architectures: https://www.emergentmind.com/topics/hybrid-transformer-cnn-architectures
8. Emergent Mind, Longformer-based Encoder: https://www.emergentmind.com/topics/longformer-based-encoder
9. M. Brenndoerfer, Longformer Efficient Attention Notes: https://mbrenndoerfer.com/writing/longformer-efficient-attention-long-documents

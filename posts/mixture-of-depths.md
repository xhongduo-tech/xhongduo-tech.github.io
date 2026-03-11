## 核心结论

Mixture-of-Depths，简称 MoD，本质是“按层做条件计算”。更直白地说，不是每个 token 在每一层都执行同样多的计算，而是先由一个路由器判断：这个 token 在当前层值不值得花 attention 和 MLP 的成本。

它把 MoE 的路由思想从“把 token 分给哪个专家”改成“这个 token 要不要经过当前层”。具体做法是：每一层先对全部 token 打分，再只允许固定容量 $C$ 个 token 进入完整的 attention + MLP，其余 token 不做这一层的昂贵计算，直接通过残差连接把上一层表示传下去。

这个设计的核心结论有三点：

1. MoD 的主要收益不是减少参数量，而是把有限 FLOP 集中到更重要的 token 上。
2. 固定容量的 top-k 路由同时解决两件事：一是每层计算预算可控，二是输出张量形状保持稳定，因此仍然能用标准批处理方式训练和部署。
3. 原论文最关键的训练设计是：前向用硬 top-k 决定哪些 token 进入当前层，并把 router 分数乘到被选中的计算分支上，让 router 进入语言模型主损失的梯度路径。工程实现里，很多人还会额外加入 straight-through estimator，简称 STE。它的直白解释是“前向仍然用硬选择，反向用连续近似提供梯度”，作用是让训练更稳定，而不是改变前向语义。

可以先用下表抓住 MoD 和普通 Transformer 的差别：

| 方案 | 每层参与完整计算的 token 数 | FLOP 特征 | 梯度主要流向 | 张量形状 |
| --- | --- | --- | --- | --- |
| 全部 token 参与 | $S$ | 固定且最高 | 全部 token | 固定 |
| MoD top-k 参与 | $C$ | 固定且更低 | 主要流向入选 token 与 router | 固定 |

玩具例子可以这样理解：序列长度 $S=16$，容量 $C=4$。这一层只让 4 个分数最高的 token 进入 attention + MLP，另外 12 个 token 不做新计算，直接把上一层表示复制到下一层。于是这一层真正昂贵的计算只发生在 $4/16=25\%$ 的 token 上。

这件事的重点不是“删掉了 75% 的 token”，而是“这 75% 的 token 在当前层暂时不更新”。它们仍然存在于序列里，也仍然会出现在下一层，只是这一层没有被分配到额外算力。

---

## 问题定义与边界

MoD 解决的问题是：在 causal language model 里，在不改变层间张量尺寸的前提下，按 token 动态分配算力。

标准 Transformer 的问题很直接：不管 token 是句首标点、重复格式符号，还是真正承载语义和推理负担的关键词，它们在每一层都要支付同样的 attention 和 MLP 成本。这种“所有 token 同价”的计算方式在长序列下会浪费大量 FLOP。

MoD 的约束边界也很清晰：

| 符号 | 含义 | 作用 |
| --- | --- | --- |
| $S$ | 序列长度 | 当前层一共有多少个 token 可供选择 |
| $C$ | capacity，容量 | 当前层最多允许多少个 token 进入完整计算 |
| $r_i^l$ | 第 $l$ 层第 $i$ 个 token 的路由分数 | 决定该 token 是否入选 |
| $\beta=1-C/S$ | 百分位阈值对应比例 | 用来定义 top-k 的分界线 |

如果记第 $l$ 层所有 token 的路由分数集合为 $R^l=\{r_1^l,\dots,r_S^l\}$，那么阈值可以写成：

$$
\beta = 1 - \frac{C}{S}, \qquad \tau^l = P_\beta(R^l)
$$

其中 $P_\beta(R^l)$ 的含义是：把这一层全部 token 的分数排序后，取第 $\beta$ 百分位所对应的阈值。高于这个阈值的 token 被选中，数量约等于 $C$。在真正实现里，为了严格拿到固定数量，通常直接做 top-k，而不是依赖百分位数函数。

把这个过程写成二值门控更容易理解：

$$
m_i^l = \mathbf{1}[r_i^l \in \operatorname{TopK}(R^l, C)]
$$

这里 $m_i^l \in \{0,1\}$ 表示第 $i$ 个 token 在第 $l$ 层是否进入完整计算。于是每层都在做同一件事：先打分，再保留恰好 $C$ 个 token 的门为 1，其余门为 0。

这里有两个关键边界。

1. $C$ 一般是预先固定的，而不是每个 batch 动态变化。原因不是数学上不能变，而是固定容量更容易保证张量形状稳定、吞吐稳定、内核调度稳定。
2. top-k 依赖整段序列的全局排序，因此训练时比较自然，采样时却有非因果问题。因为自回归生成到第 $t$ 步时，第 $t+1$ 到第 $S$ 个 token 还不存在，你无法直接知道当前 token 在“全序列 top-k”里是否会入选。

再看一个具体量级。假设序列长度 $S=2048$，容量 $C=256$，那么该层只有：

$$
\frac{C}{S} = \frac{256}{2048} = 12.5\%
$$

的 token 会进入 attention + MLP。也就是说，87.5% 的 token 在这一层只做残差传递，不做新的昂贵计算。MoD 的工程价值就来自这种“按层稀疏、但张量形状不变”的节省方式。

---

## 核心机制与推导

第 $l$ 层输入记作 $X^l=\{x_i^l\}_{i=1}^S$，router 是一个线性打分器：

$$
r_i^l = w_\theta^\top x_i^l
$$

它的含义很简单：用一个可学习向量和 token 表示做内积，得到“这个 token 在当前层值不值得继续花算力”的分数。

选中规则可以写成四步：

1. 对全部 $S$ 个 token 计算分数 $r_i^l$
2. 从中选出 top-k，也就是容量为 $C$ 的那一组 token
3. 只有被选中的 token 进入当前层的 attention + MLP
4. 未入选 token 直接走残差连接

把它写成带门控的形式更清楚：

$$
x_i^{l+1} = x_i^l + m_i^l \cdot r_i^l \cdot f_i(\tilde X^l)
$$

其中：

- $m_i^l$ 是 top-k 产生的二值门控
- $\tilde X^l$ 是被选中的 token 子集
- $f_i(\tilde X^l)$ 表示该层真正昂贵的计算，即 self-attention 与 MLP 的组合输出

这一个式子里有三个最重要的点。

第一，未入选 token 是恒等映射。  
如果 $m_i^l=0$，那么：

$$
x_i^{l+1} = x_i^l
$$

这不是“删除 token”，也不是“截断梯度”，而是“这一层暂不更新该 token”。它依然保留在后续层的输入里，因此 MoD 是深度路径动态，而不是序列长度动态。

第二，入选 token 的分支前面要乘上 $r_i^l$。  
这一步非常关键，因为它让 router 进入语言模型主损失的梯度路径。否则路由决策只是一个离散选择，router 很难从主任务里收到有效信号。原论文强调的重点并不是“用软 top-k 取代硬 top-k”，而是“保留硬 top-k，同时让分数直接参与被选中分支的数值计算”。

第三，节省来自“只对子集做昂贵计算”。  
如果忽略常数和实现细节，标准层的主要开销大致是：

$$
\text{Attention} \sim O(S^2 d), \qquad \text{MLP} \sim O(S d^2)
$$

MoD 只对 $C$ 个 token 做完整块计算后，主开销可以近似写成：

$$
\text{Attention} \sim O(C^2 d), \qquad \text{MLP} \sim O(C d^2)
$$

因此当 $C \ll S$ 时，节省会非常明显。严格来说，真实 attention 还会涉及被选中 token 对上下文的访问方式、是否只在子集内部算 attention、以及实现上的 gather/scatter 成本，所以这里是主量级分析，不是完整的 kernel 级 FLOP 公式。但用它来理解 MoD 为什么省算力已经足够。

下面给一个完整的玩具例子。设：

- $S=8$
- $C=2$
- router 分数为 $[0.1, 0.7, 0.2, 1.3, -0.4, 0.5, 0.0, 0.9]$

排序后前 2 名是 1.3 和 0.9，对应第 4 和第 8 个 token，所以这一层只更新这两个 token，其余 6 个 token 直接跳过。你可以把它理解成：模型认为当前层最值得继续加工的是第 4 和第 8 个 token，其它 token 在这一层保留原状即可。

这和 early exit 不同。early exit 的意思是“某个 token 在较浅层已经足够确定，因此后续所有层都不再处理它”；MoD 则是“某个 token 是否在当前层被处理，是逐层重新决定的”。同一个 token 可能在第 3 层被跳过，在第 4 层又重新入选，因此它是动态深度，而不是永久退出。

为了帮助初学者区分几个容易混淆的概念，可以看这个对照表：

| 概念 | 问题形式 | 决策对象 | 结果 |
| --- | --- | --- | --- |
| MoD | 这层算不算 | token 是否进入当前层 | 深度路径变化 |
| MoE | 去哪个专家算 | token 选择专家子网络 | 宽度路径变化 |
| Early Exit | 后面还算不算 | token 是否提前结束后续层 | 计算终止 |
| Sparse Attention | 看哪些 token | attention 连接边 | 注意力图变化 |

---

## 代码实现

最小实现可以拆成四步：打分、选 top-k、只对子集做块计算、再散回原位置。下面先给一个可以直接运行的 NumPy 版本，再给一个接近真实训练代码的 PyTorch 版本。

### 1. 可直接运行的 NumPy 例子

这个版本故意不实现真正的 attention，而是用一个可验证的假计算 `residual_calc` 代替。目的不是复现论文结果，而是把 MoD 的数据流讲清楚。

```python
import numpy as np

def residual_calc(selected: np.ndarray) -> np.ndarray:
    """
    用一个确定性的假计算代替 attention + MLP。
    输入/输出形状都为 [C, D]。
    """
    return selected * 2.0 + 1.0

def topk_mask(scores: np.ndarray, capacity: int) -> np.ndarray:
    """
    返回 shape [S] 的布尔 mask，恰好有 capacity 个 True。
    """
    if capacity <= 0:
        return np.zeros_like(scores, dtype=bool)
    if capacity >= scores.shape[0]:
        return np.ones_like(scores, dtype=bool)

    top_idx = np.argpartition(scores, -capacity)[-capacity:]
    mask = np.zeros_like(scores, dtype=bool)
    mask[top_idx] = True
    return mask

def mod_layer(x: np.ndarray, router_w: np.ndarray, capacity: int):
    """
    x: shape [S, D]
    router_w: shape [D]
    返回:
      out:    shape [S, D]
      scores: shape [S]
      mask:   shape [S]
    """
    assert x.ndim == 2
    assert router_w.ndim == 1
    assert x.shape[1] == router_w.shape[0]

    scores = x @ router_w                       # [S]
    mask = topk_mask(scores, capacity)         # [S]
    selected = x[mask]                         # [C, D]
    selected_scores = scores[mask][:, None]    # [C, 1]

    transformed = residual_calc(selected)      # [C, D]
    updated = selected_scores * transformed    # [C, D]

    out = x.copy()
    out[mask] = x[mask] + updated              # 残差更新
    return out, scores, mask

def main():
    x = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 1.0],
        [3.0, 0.5],
    ], dtype=np.float64)

    router_w = np.array([1.0, 0.2], dtype=np.float64)
    out, scores, mask = mod_layer(x, router_w, capacity=2)

    print("scores:", scores)
    print("mask:", mask.astype(int))
    print("out:\n", out)

    assert mask.sum() == 2
    assert scores.shape == (4,)
    assert out.shape == x.shape
    assert np.allclose(out[~mask], x[~mask])   # 未入选 token 直接跳过
    assert not np.allclose(out[mask], x[mask]) # 入选 token 被更新

if __name__ == "__main__":
    main()
```

这个例子里如果你手算一下：

- 第 1 个 token 分数：$1.0 \times 1.0 + 0.0 \times 0.2 = 1.0$
- 第 2 个 token 分数：$0.0 \times 1.0 + 1.0 \times 0.2 = 0.2$
- 第 3 个 token 分数：$2.0 \times 1.0 + 1.0 \times 0.2 = 2.2$
- 第 4 个 token 分数：$3.0 \times 1.0 + 0.5 \times 0.2 = 3.1$

因此 top-2 会选中第 3 和第 4 个 token。程序里的 `mask` 应该正好反映这一点。

### 2. 接近真实训练代码的 PyTorch 版本

真实训练里一般是 batched 版本，输入形状为 `[B, S, D]`。下面这个例子省略多头 attention 和 layer norm，只保留 MoD 的核心路由逻辑，代码可直接运行。

```python
import torch
import torch.nn as nn

class ToyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class MoDLayer(nn.Module):
    def __init__(self, dim: int, capacity: int):
        super().__init__()
        self.router = nn.Linear(dim, 1, bias=False)
        self.block = ToyBlock(dim)
        self.capacity = capacity

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, D]
        return:
          out: [B, S, D]
          scores: [B, S]
          topk_idx: [B, C]
          mask: [B, S]
        """
        B, S, D = x.shape
        C = min(self.capacity, S)

        scores = self.router(x).squeeze(-1)           # [B, S]
        topk_scores, topk_idx = torch.topk(scores, k=C, dim=1)  # [B, C], [B, C]

        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, D)   # [B, C, D]
        selected = torch.gather(x, dim=1, index=gather_idx)     # [B, C, D]

        selected_out = self.block(selected)                     # [B, C, D]
        selected_out = topk_scores.unsqueeze(-1) * selected_out

        out = x.clone()
        residual_selected = selected + selected_out
        out.scatter_(dim=1, index=gather_idx, src=residual_selected)

        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_idx, True)

        return out, scores, topk_idx, mask

def main():
    torch.manual_seed(0)

    B, S, D, C = 2, 6, 4, 3
    x = torch.randn(B, S, D)

    layer = MoDLayer(dim=D, capacity=C)
    out, scores, topk_idx, mask = layer(x)

    print("scores shape:", scores.shape)
    print("topk_idx shape:", topk_idx.shape)
    print("mask:\n", mask.int())
    print("out shape:", out.shape)

    assert out.shape == x.shape
    assert scores.shape == (B, S)
    assert topk_idx.shape == (B, C)
    assert mask.shape == (B, S)
    assert torch.all(mask.sum(dim=1) == C)

if __name__ == "__main__":
    main()
```

这个 PyTorch 版本体现了三件关键工程事实：

1. `torch.topk` 保证每个样本都严格拿到 $C$ 个 token。
2. `gather` 和 `scatter_` 是 MoD 的核心张量操作。
3. 输出张量形状仍然是 `[B, S, D]`，所以它可以嵌回标准 Transformer 栈里，不需要改上层接口。

### 3. 关于 STE，该怎么讲才准确

这部分很容易被过度简化，所以需要说清楚。

- 原论文最核心的主设计是“硬 top-k + 被选中分支乘 router 分数 + 为采样阶段补辅助机制”。
- STE 不是论文里唯一且必须的定义性成分，更准确的说法是：很多工程实现会额外加入 STE 或其它连续近似，让 router 的训练更平滑。

一种常见写法如下：

```python
import torch

def ste_mask(scores: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    scores: [B, S]
    返回一个前向近似 hard top-k、反向带 soft 梯度的 mask。
    """
    _, topk_idx = torch.topk(scores, k=k, dim=1)

    hard = torch.zeros_like(scores)
    hard.scatter_(1, topk_idx, 1.0)

    soft = torch.sigmoid(scores / temperature)

    # 前向值等于 hard，反向梯度来自 soft
    mask = hard + soft - soft.detach()
    return mask
```

它的含义是：

- 前向时，`mask` 的数值表现和 `hard` 一样，看起来像离散选择。
- 反向时，`hard` 没梯度，但 `soft - soft.detach()` 会把 `soft` 的梯度保留下来。

因此 STE 的作用不是改变前向路由，而是给 router 提供一个近似连续的梯度通道，缓解“刚好卡在 top-k 边界附近的 token 很难学”的问题。

### 4. 初学者最容易误解的实现点

| 误解 | 正确理解 |
| --- | --- |
| 未入选 token 被删除了 | 没有删除，只是这一层不更新 |
| top-k 后输出长度会变短 | 中间子集变短，但散回后输出仍是原长度 |
| router 只负责选 token，不参与主任务 | 实际上 router 分数会乘进被选中的分支，从而进入主损失梯度 |
| 有了 STE 就不需要硬 top-k | 不对，STE 通常只是给硬 top-k 提供近似梯度，不改变容量约束 |

---

## 工程权衡与常见坑

MoD 不是“加一个 top-k”就结束，真正的难点在训练路径、采样路径、以及张量实现三者的一致性。

先看最常见的问题：

| 常见坑 | 本质问题 | 常见规避策略 |
| --- | --- | --- |
| top-k 与 causal 冲突 | 生成第 $t$ 个 token 时看不到未来 token，无法知道它是否属于全序列 top-k | 用辅助 loss 让 router 学成局部可判别，或加第二个 predictor |
| top-k 不可导 | 硬选择会让边界附近样本梯度不稳定 | 用 router 分数乘被选中分支；工程上可再加 STE 或 soft relaxation |
| 容量不平衡 | 动态 k 会破坏 batch 一致性和吞吐 | 固定 $C$，不要让每个样本自由涨跌 |
| 过度跳层 | router 学会把大量 token 都排到低分区，表示能力下降 | 只在部分层插入 MoD，并调 capacity |
| 索引散回开销 | gather/scatter 写不好会抵消理论收益 | 用张量化实现，减少 Python 循环与无谓拷贝 |

### 1. 训练能跑，不代表推理能跑

这是 MoD 最大的工程门槛。训练时，模型拿到的是完整序列，所以可以先看完所有 token，再决定谁属于 top-k。可一旦进入自回归生成，模型每次只看到当前前缀，这时“当前 token 最终是不是全序列 top-k”这个问题本身就不再因果。

这可以写成一个简单对比：

| 场景 | 能否看到完整序列 | 是否能直接做全序列 top-k |
| --- | --- | --- |
| 训练 | 能 | 能 |
| 自回归采样 | 不能 | 不能 |

原论文为这个问题给了两类办法。

1. 辅助 loss  
把“某个 token 在完整训练路径里是否属于 top-k”当作一个监督标签，训练 router 或额外分支去预测这个二值结果。这样推理时即使看不到未来，也能根据当前上下文近似判断。

2. 第二个 predictor  
再训练一个小模型或小分支，输入与 router 类似，但不承接主模型的全部梯度，只专门学习“这个 token 最终会不会被选中”。这样做的目的，是把“完整序列排序得到的事后标签”转换成“前缀可预测的局部判断”。

直白地说，训练时你是在“考试结束后看总分排名前四名是谁”；推理时你是在“考试进行中预测谁最后会进前四”。这两个问题不是同一个问题，所以不能直接把训练态逻辑搬到采样态。

### 2. top-k 不可导，但不是完全无解

硬 top-k 最大的问题是边界不连续。第 $C$ 名和第 $C+1$ 名的分数可能只差很小一点，但前者有完整梯度路径，后者几乎没有。这会让 router 的训练很抖。

MoD 的典型处理方式有三层：

1. 先接受前向必须是硬 top-k  
因为容量固定是它的工程前提。如果你完全改成软选择，计算预算就不再可控。

2. 让被选中分支乘上 router 分数  
这一步把 router 接入主任务损失，是最核心的稳定来源。

3. 视情况再加 STE 或软松弛  
如果训练仍然不稳，再通过近似连续梯度补充信号。

因此，正确的理解不是“MoD 靠 STE 才能训练”，而是“MoD 的定义性设计是硬 top-k 和固定容量；STE 是常见的工程补充”。

### 3. 理论节省不等于实际提速

这是另一个很容易误判的地方。理论上，只让 $C$ 个 token 进入 attention + MLP，FLOP 一定下降；但实际 wall-clock latency 是否下降，要看以下因素：

| 因素 | 会不会影响实际速度 | 原因 |
| --- | --- | --- |
| gather/scatter 的实现 | 会 | 内存访问不连续会拉低吞吐 |
| GPU kernel 是否适配固定小批子集 | 会 | 小张量可能利用率不足 |
| MoD 层插入频率 | 会 | 路由和散回本身也有开销 |
| capacity 设得是否过大 | 会 | 如果 $C$ 接近 $S$，节省不明显 |
| 推理是否实现因果路由 | 会 | 训练快不代表生成快 |

所以在工程上，最保守的做法通常不是“每一层都换成 MoD”，而是“只在部分层插入 MoD，并逐步调 capacity”。因为如果插得太密，路由开销和实现复杂度可能先把收益吃掉。

### 4. 新手最实用的判断标准

如果你准备把 MoD 加到已有模型里，先问自己四个问题：

1. 你的主要瓶颈是训练 FLOP，还是注意力显存？
2. 你能否接受训练态和采样态存在不同的路由实现？
3. 你的推理系统能否支持高效的 gather/scatter 和固定容量子批处理？
4. 你的任务是否允许一部分 token 在某些层完全不更新？

只要这四个问题里有两个以上答案偏否定，MoD 的落地成本就可能高于收益。

---

## 替代方案与适用边界

MoD 最适合的场景是：你接受“不是每个 token 在每一层都更新”，目标是压低 FLOP，同时尽量保留标准 Transformer 的总体结构和接口。

先把常见方案放在一起对比：

| 方案 | 计算分配对象 | 动态路径 | 可微性处理 | 部署复杂度 |
| --- | --- | --- | --- | --- |
| MoD | token 是否经过某层 | 按深度动态 | 硬 top-k，常配 router 梯度技巧/STE | 中等 |
| MoE | token 走哪个专家 | 按专家动态 | router + load balance 机制 | 较高 |
| 稀疏 Attention | token 看哪些 token | 按连接模式动态或半动态 | 通常无需 top-k 层跳过 | 中等 |
| FlashAttention | 不改计算语义，只改内核实现 | 无 | 无额外路由 | 低 |

### 1. MoD 和 MoE 的区别

这两个概念非常容易混在一起，但它们解决的不是同一个问题。

- MoE 问的是：这个 token 应该交给哪个专家网络处理？
- MoD 问的是：这个 token 在当前层要不要处理？

所以：

- MoE 改的是“宽度上的专家路径”
- MoD 改的是“深度上的层路径”

这意味着二者可以叠加。也就是先用 MoD 决定“哪些 token 值得花这一层的成本”，再用 MoE 决定“这些 token 交给哪个专家”。这种组合常被称为 MoDE。它的潜在收益是进一步压缩单次前向中真正参与计算的 token 数和激活参数数，但代价是系统复杂度更高，训练和部署都更难调。

### 2. MoD 和稀疏 Attention 的区别

MoD 节省的是“哪些 token 进入当前层的完整块计算”，而稀疏 Attention 节省的是“一个 token 在 attention 里看哪些别的 token”。两者关注点不同：

- MoD 更像“某些 token 在这一层暂时不加工”
- 稀疏 Attention 更像“每个 token 仍然加工，但只看局部或稀疏邻居”

如果你的主要问题是长上下文 attention 的 $O(S^2)$ 内存和时间，优先看的通常是 FlashAttention、块稀疏 attention 或滑窗 attention。因为这类方法不改变“每层每个 token 都更新”的语义，更容易和现有推理系统兼容。

### 3. MoD 的适用边界

下面这个判断表可以直接用于方案筛选：

| 场景 | 是否优先考虑 MoD | 原因 |
| --- | --- | --- |
| 长序列训练 FLOP 是主瓶颈 | 是 | 可以把算力集中到更重要 token |
| 注意力显存是主瓶颈 | 不一定 | 先看 FlashAttention 或稀疏 attention |
| 必须保证每个 token 每层都更新 | 否 | MoD 的基本前提就是允许跳层 |
| 只能接受训练和推理完全同构 | 谨慎 | MoD 的采样路由通常要额外设计 |
| 已有成熟的 MoE 基础设施 | 可以考虑叠加 | MoD 与 MoE 在原理上兼容 |

可以把 MoD 的适用边界概括成一句话：  
它适合“愿意用更复杂的路由逻辑，换取更低的层级计算成本”的场景；如果你更看重语义保守性、推理路径简单性或注意力内核优化，那么别的方案通常更直接。

---

## 参考资料

下面的参考资料按“先原始定义，再工程理解，再中文梳理”的顺序排。对初学者而言，最稳妥的阅读顺序是：先看论文摘要和图，再看一篇工程解读，最后回到公式。

| 来源 | 内容亮点 |
| --- | --- |
| [Raposo et al., Mixture-of-Depths: Dynamically allocating compute in transformer-based language models (arXiv 2404.02258)](https://arxiv.org/abs/2404.02258) | 原始论文，给出 MoD 定义、固定容量 top-k、训练与采样阶段设计、iso-FLOP 对比结果，是所有二手资料的起点 |
| [Graphcore Research Blog: Mixture-of-Depths](https://graphcore-research.github.io/mixture-of-depths/) | 对机制、容量控制、采样难点的讲解比较清楚，适合先建立整体图景 |
| [Andrei Alexandru, Understanding mixture-of-depths](https://inwaves.io/mixture-of-depths/) | 面向工程读者的解读，适合快速理解“为什么要按 token 分配层内算力” |
| [fearthedeer9: Mixture-of-Depths 笔记](https://fearthedeer9.github.io/2025/01/04/mixture-of-depths/) | 中文材料里对公式、容量比例、MoD 与 MoE 组合关系梳理得比较细，适合复习 |
| [Champaign Magazine: Mixture-of-Depths](https://champaignmagazine.com/2025/11/03/aikipedia-mixture-of-depths/) | 对“是否应把训练过程简单说成 STE”这件事有纠偏价值，适合检查表述是否过度简化 |

如果只打算读一篇，优先读原论文；如果你想先形成工程直觉，再回去啃论文，那么可以先看 Graphcore 的总结。判断一篇解读是否可靠的一个简单标准是：它有没有同时讲清楚三件事。

1. 固定容量 top-k 是为了什么
2. router 分数为什么要乘到被选中的分支上
3. 训练态全序列路由为什么不能直接照搬到自回归采样

如果这三点里缺了任意一点，那篇解读大概率只讲清了表面概念，没有讲清 MoD 真正难落地的部分。

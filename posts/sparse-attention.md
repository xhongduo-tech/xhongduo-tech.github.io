## 核心结论

Attention 稀疏化解决的是一个很具体的问题：标准自注意力在长度为 $n$ 的序列上需要构造 $n \times n$ 的相关性矩阵，因此计算量和显存占用都会随序列长度按平方增长。对长文本、长代码、长文档任务，这个成本很快会变得不可接受。

结论先固定下来：

1. 标准自注意力要求每个 token 与全部 $n$ 个 token 交互，所以复杂度是 $O(n^2)$。这里的 token 可以理解为模型处理文本时的最小离散单元，通常是词片段而不是完整单词。
2. 局部注意力只让每个 token 关注固定窗口内的邻居。若单侧窗口宽度为 $w$，每个 token 只与大约 $2w+1$ 个位置交互，总复杂度可记为 $O(nw)$。当 $w \ll n$ 时，成本接近线性。
3. Longformer 在局部窗口之外，再给少数关键位置分配全局注意力。全局 token 指可以与整条序列双向通信的位置，例如 `[CLS]`、问题区间、标题位置、章节锚点。其复杂度通常写成
   $$
   O(nw + |G|n)
   $$
   其中 $|G|$ 是全局 token 数量。
4. BigBird 在“局部 + 全局”之外，再给每个 token 添加少量随机连接。复杂度可写成
   $$
   O\bigl(n(w+g+r)\bigr)
   $$
   其中 $g$ 表示与全局相关的连接规模，$r$ 表示随机连接数。若 $w,g,r$ 都视为常数，则总复杂度仍是 $O(n)$。
5. 理论上的“更省”不等于工程上的“更快”。BigBird 的随机边能改善图连通性，但在真实 GPU 上，FlashAttention 这类密集注意力优化 kernel 往往已经非常高效。因此是否采用稀疏注意力，不能只看渐近复杂度，还要看硬件实现、kernel 质量、batch 大小和上下文长度。

先看一个最小数值例子。设序列长度 $n=4096$，单侧窗口 $w=64$，则：

| 方案 | 单层大致交互次数 | 量级说明 |
|---|---:|---|
| 全局注意力 | $4096^2 = 16{,}777{,}216$ | 每个位置看全部位置 |
| 局部窗口 | $4096 \times (2 \times 64 + 1) \approx 528{,}384$ | 每个位置只看邻近位置 |
| Longformer，额外 2 个全局 token | $4096 \times 129 + 2 \times 4096 \approx 536{,}576$ | 局部基础上保留少量全局通道 |

如果把窗口定义成“总宽度 $w$”而不是“单侧宽度 $w$”，公式会写成 $O(nw)$，两种写法只差常数，不影响结论。

这组数字说明了稀疏注意力的真正价值：不是让所有位置继续保留完整全局交互，而是只为关键位置保留全局通信能力，把大部分非必要连接删掉。

---

## 问题定义与边界

问题可以表述为：

给定一个长度为 $n$ 的序列，如何在尽量不丢失长程依赖建模能力的前提下，把注意力计算从 $O(n^2)$ 降到更可控的规模，并且让这种稀疏模式能够在真实硬件上高效执行。

这里需要先划清几个边界。

第一，稀疏注意力解决的是“长上下文代价过高”问题，不是“所有 Transformer 都应该稀疏化”问题。若输入长度只有 512 或 1K，成熟的密集实现往往已经足够好。真正让稀疏模式变得必要的，通常是 4K、8K、16K 甚至更长的上下文。

第二，稀疏化的本质不是简单减少边数，而是重新设计注意力图。注意力图可以理解为：哪些位置之间允许直接交换信息。边删掉之后，信息传播路径会变化。全局注意力下，任意两个 token 一层就能直接交互；局部窗口下，远端 token 必须通过多层逐步传递。

第三，稀疏注意力至少要回答三个设计问题：

| 参数 | 含义 | 核心设计问题 |
|---|---|---|
| $w$ | 局部窗口宽度 | 每个 token 至少要看多远的邻居 |
| $g=|G|$ | 全局 token 数量 | 哪些位置必须保留全局双向通信 |
| $r$ | 随机连接数 | 是否需要额外跨段跳跃来缩短路径 |

所以，稀疏注意力不是一个单一算法，而是一类“注意力图设计方法”。

为了让这个边界更具体，先看两个例子。

玩具例子：一篇 4000 token 的长文档分类任务。如果模型最终只需要输出一个类别，那么把 `[CLS]` 或文首摘要位设为全局 token，通常就足够。因为模型只需要一个全局汇总通道，而不是让所有位置彼此直连。

真实工程例子：长文问答任务中，问题在文首，答案证据分散在不同段落。若只用局部窗口，问题 token 需要经过多层传播才能影响远处证据段；若把问题 token、本章标题、章节编号设为全局 token，则问题约束可以更快传播到答案区域。

再看一个更贴近业务的场景。长合同审查里，模型需要判断“赔偿上限条款”和“违约责任条款”是否冲突。这两个片段可能相隔几千 token。若只用滑动窗口，模型必须依赖多层接力才能让两个片段发生关联；若把条款标题、编号、问题提示词设为全局 token，则模型能更快建立跨章节联系。

从理论访问数量看，很多稀疏方案都可以统一写成：
$$
\text{Cost} = O\bigl(n(w+g+r)\bigr)
$$
但这只描述“需要访问多少个位置”，没有描述这些访问在 GPU 上是否连续、是否适合分块、是否容易融合 kernel。后者直接决定工程表现，这一点会在后文展开。

---

## 核心机制与推导

先回顾标准自注意力。给定查询矩阵 $Q \in \mathbb{R}^{n \times d}$、键矩阵 $K \in \mathbb{R}^{n \times d}$、值矩阵 $V \in \mathbb{R}^{n \times d_v}$，标准自注意力写成：
$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中：

| 符号 | 含义 |
|---|---|
| $n$ | 序列长度 |
| $d$ | 单个注意力头的隐藏维度 |
| $QK^\top$ | 所有 query 与所有 key 的相关性分数矩阵 |
| softmax | 把分数归一化成注意力权重 |

问题在于 $QK^\top$ 的形状是 $n \times n$。这意味着：

1. 分数矩阵本身就有 $n^2$ 个元素。
2. 计算这些分数需要大规模矩阵乘。
3. 训练时还要考虑 softmax 中间值、mask、梯度缓存等额外开销。

### 1. 局部注意力

局部注意力的做法是：第 $i$ 个 token 不再看全部位置，而只看局部区间。例如，若单侧窗口宽度为 $w$，则第 $i$ 个位置只与
$$
[j \mid \max(0, i-w) \le j \le \min(n-1, i+w)]
$$
中的位置交互。

如果忽略边界位置，那么每个 token 大约只看 $2w+1$ 个位置，总成本约为：
$$
O(n(2w+1)) = O(nw)
$$

这里最容易让新手混淆的一点是：复杂度从二次变成线性，并不代表表达能力没有损失。局部窗口的代价更低，是因为它主动删掉了绝大多数远距离连接。

局部窗口的直接后果是远程信息不能一步到达。

假设位置 0 的信息要传到位置 $n-1$。若单层只能跨越宽度 $w$ 的局部邻域，那么经过 $L$ 层后，单个位置的感受野大致扩展为 $Lw$ 量级。要覆盖整个序列，至少需要满足：
$$
L \cdot w \gtrsim n
$$
因此：
$$
L = O(n/w)
$$

这个式子的含义很重要：纯局部窗口虽然省，但远距离依赖的传播速度很慢。窗口越小，需要的层数越多。

一个简单例子：

- 序列长度 $n=1024$
- 单侧窗口 $w=32$

则单层只能覆盖约 65 个位置。若想让序列开头的信息影响到序列末尾，理论上需要大约
$$
1024 / 32 = 32
$$
层级别的逐步传播。即使有残差连接和多头机制，这个传播路径依然明显比全局注意力更长。

### 2. Longformer：局部 + 全局

Longformer 的核心思路是：保留局部窗口作为主干，再为少数关键位置增加全局注意力。

设全局 token 集合为 $G$。那么：

1. 普通 token 关注自己的局部窗口，以及全局 token。
2. 全局 token 可以关注整条序列。
3. 整条序列也可以把信息发送给全局 token。

于是复杂度通常写成：
$$
O(nw + |G|n)
$$

这个式子可以拆开理解：

- $nw$ 对应普通 token 的局部窗口开销；
- $|G|n$ 对应全局 token 与整条序列的双向交互开销。

为什么这会显著改善长程传播？

因为全局 token 充当了“中转站”或“集线器”。任意普通 token 都可以在一层内把信息传给全局 token，再由全局 token 在下一层影响远处位置。于是原本需要多层接力的传播路径，被压缩成常数层级。

可以把它写成一个非常直观的路径对比：

| 结构 | 从位置 $i$ 到远处位置 $j$ 的典型路径 |
|---|---|
| 纯局部窗口 | $i \rightarrow i+w \rightarrow i+2w \rightarrow \cdots \rightarrow j$ |
| Longformer | $i \rightarrow g \rightarrow j$，其中 $g \in G$ |

玩具例子：假设一排 16 个人，每个人只能和左右 2 个人交谈。第 1 个人要把消息传给第 16 个人，需要多轮接力。现在加一个主持人，所有人都能直接联系主持人，主持人也能联系所有人，那么远程传递就被大幅缩短。

这类结构特别适合存在“天然锚点”的任务，例如：

| 任务 | 适合设置为全局 token 的位置 |
|---|---|
| 文档分类 | `[CLS]`、标题、章节标题 |
| 长文问答 | 问题区间、问题中的关键词、`[CLS]` |
| 信息抽取 | 实体锚点、字段名、分隔符 |
| 合同审查 | 条款编号、标题、问题提示词 |

### 3. BigBird：局部 + 全局 + 随机

BigBird 在 Longformer 基础上再添加随机连接。每个 token 除了局部邻居和全局通道外，还会额外连接到少量远处位置。

复杂度写成：
$$
O\bigl(n(w+g+r)\bigr)
$$

其中：

| 符号 | 含义 |
|---|---|
| $w$ | 局部窗口规模 |
| $g$ | 与全局相关的连接规模 |
| $r$ | 随机远程连接规模 |

如果把 $w,g,r$ 都视为常数，那么总复杂度仍近似线性。

BigBird 的理论动机可以用图论理解。把 token 当成图节点，把可见性关系当成边：

1. 只有局部窗口时，图接近一条局部连接的链，图直径较大。
2. 加入全局 token 后，部分路径可以通过“超级节点”快速缩短。
3. 再加入随机边后，图更接近小世界结构，即局部聚集但全局最短路径很短。

图直径指图中任意两点最短路径长度的最大值。图直径越小，远距离信息越容易传播。

直觉上，随机边的作用不是“让模型更随机”，而是“给图增加少量远程跳板”。少量随机桥梁就足以打破纯链式传播结构。

下面把三类结构放在一起比较：

| 结构 | 单层成本 | 远距传播路径 | 优点 | 缺点 |
|---|---|---|---|---|
| 全局注意力 | $O(n^2)$ | 1 跳 | 表达力直接，所有位置全连接 | 长序列代价高 |
| 局部窗口 | $O(nw)$ | 约 $O(n/w)$ 层 | 简单、便宜、接近线性 | 远程依赖传播慢 |
| Longformer | $O(nw+|G|n)$ | 经过全局 token 可到常数层 | 关键位置可快速汇聚 | 依赖全局 token 选取 |
| BigBird | $O(n(w+g+r))$ | 局部路径 + 全局桥 + 随机桥 | 连通性更强，理论更完整 | mask 和 kernel 更复杂 |

真实工程例子：长文问答中，问题在开头，证据散落在后文多个段落。Longformer 会把问题 token 设为全局，让所有段落直接感知问题约束；BigBird 再增加随机块连接，可以缩短“证据段与证据段之间”的通信路径，减少只靠局部滑动传播的限制。

---

## 代码实现

工程实现的第一步通常不是直接写 CUDA，而是先把注意力模式用 mask 表达清楚。mask 可以理解为一个二值矩阵：`1` 表示位置 $i$ 可以看位置 $j$，`0` 表示必须屏蔽。

下面给出一个可运行的 Python 玩具实现，目标不是高性能，而是把“局部窗口 + 全局 token + 随机连接”的规则完整表达出来，并且顺手演示如何把它用于一次真正的 masked attention 计算。

```python
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


def build_sparse_mask(
    n: int,
    window: int,
    global_indices: list[int] | None = None,
    random_links: int = 0,
    seed: int = 0,
    bidirectional_random: bool = False,
) -> np.ndarray:
    """
    构造一个形状为 [n, n] 的二值 mask。
    mask[i, j] = True 表示第 i 个 token 可以关注第 j 个 token。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")
    if random_links < 0:
        raise ValueError("random_links must be non-negative")

    global_indices = sorted(set(global_indices or []))
    for g in global_indices:
        if g < 0 or g >= n:
            raise ValueError(f"global index out of range: {g}")

    rng = random.Random(seed)
    mask = np.zeros((n, n), dtype=bool)

    # 1) 每个位置默认能看到自己
    np.fill_diagonal(mask, True)

    # 2) 局部滑动窗口
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        mask[i, left:right] = True

    # 3) 全局 token：对应行和列都设为可见
    for g in global_indices:
        mask[g, :] = True
        mask[:, g] = True

    # 4) 随机连接：为每一行补充少量原本不可见的远程边
    for i in range(n):
        blocked = np.where(~mask[i])[0].tolist()
        blocked = [j for j in blocked if j != i]
        if not blocked or random_links == 0:
            continue

        picked = rng.sample(blocked, k=min(random_links, len(blocked)))
        for j in picked:
            mask[i, j] = True
            if bidirectional_random:
                mask[j, i] = True

    return mask


def masked_self_attention(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    一个最小可运行的单头自注意力实现。
    x: [n, d]
    mask: [n, n]，True 表示可见
    返回:
        y: [n, d]
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [n, d]")
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must have shape [n, n]")
    if x.shape[0] != mask.shape[0]:
        raise ValueError("x and mask must share the same sequence length")

    d = x.shape[1]

    # 玩具实现里直接令 Q=K=V=x，便于验证逻辑
    q = x
    k = x
    v = x

    scores = (q @ k.T) / math.sqrt(d)

    # 不可见位置置为极小值
    masked_scores = np.where(mask, scores, -1e9)

    # 数值稳定版 softmax
    row_max = masked_scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_max)
    exp_scores = exp_scores * mask  # 保证被屏蔽位置严格为 0
    denom = exp_scores.sum(axis=1, keepdims=True)
    probs = exp_scores / np.clip(denom, 1e-12, None)

    return probs @ v


def count_edges(mask: np.ndarray) -> int:
    return int(mask.sum())


def shortest_path(mask: np.ndarray, src: int, dst: int) -> int | None:
    """
    在有向图上计算最短路径长度。
    若不可达，返回 None。
    """
    n = mask.shape[0]
    visited = [False] * n
    dist = [0] * n
    queue = deque([src])
    visited[src] = True

    while queue:
        u = queue.popleft()
        if u == dst:
            return dist[u]

        for v in np.where(mask[u])[0]:
            if not visited[v]:
                visited[v] = True
                dist[v] = dist[u] + 1
                queue.append(v)

    return None


def main() -> None:
    n = 16
    d = 8

    rng = np.random.default_rng(123)
    x = rng.normal(size=(n, d)).astype(np.float32)

    dense_mask = np.ones((n, n), dtype=bool)
    local_mask = build_sparse_mask(n=n, window=2, global_indices=[], random_links=0)
    longformer_mask = build_sparse_mask(n=n, window=2, global_indices=[0, 8], random_links=0)
    bigbird_like_mask = build_sparse_mask(
        n=n,
        window=2,
        global_indices=[0, 8],
        random_links=1,
        seed=42,
        bidirectional_random=True,
    )

    # 基本结构检查
    assert local_mask.shape == (n, n)
    assert np.all(np.diag(local_mask))
    assert np.all(longformer_mask[0, :]) and np.all(longformer_mask[:, 0])
    assert np.all(longformer_mask[8, :]) and np.all(longformer_mask[:, 8])

    # 运行一次真实注意力
    y_local = masked_self_attention(x, local_mask)
    y_longformer = masked_self_attention(x, longformer_mask)
    y_bigbird = masked_self_attention(x, bigbird_like_mask)

    assert y_local.shape == (n, d)
    assert y_longformer.shape == (n, d)
    assert y_bigbird.shape == (n, d)

    print("Dense edges      :", count_edges(dense_mask))
    print("Local edges      :", count_edges(local_mask))
    print("Longformer edges :", count_edges(longformer_mask))
    print("BigBird edges    :", count_edges(bigbird_like_mask))

    print("Shortest path 1 -> 15")
    print("Local      :", shortest_path(local_mask, 1, 15))
    print("Longformer :", shortest_path(longformer_mask, 1, 15))
    print("BigBird    :", shortest_path(bigbird_like_mask, 1, 15))

    print("Output sample (token 0, first 3 dims)")
    print("Local      :", np.round(y_local[0, :3], 4))
    print("Longformer :", np.round(y_longformer[0, :3], 4))
    print("BigBird    :", np.round(y_bigbird[0, :3], 4))


if __name__ == "__main__":
    main()
```

这段代码有几个值得注意的点。

第一，它真的可运行，不依赖深度学习框架，只需要 `numpy`：
```bash
python sparse_attention_demo.py
```

第二，它把“图结构”和“数值计算”拆开了：

| 部分 | 作用 |
|---|---|
| `build_sparse_mask` | 定义谁能看谁 |
| `masked_self_attention` | 在给定 mask 下执行一次注意力 |
| `shortest_path` | 观察不同连接模式对信息路径的影响 |

第三，这个实现故意保留了初学者能看懂的显式步骤，而不是追求速度。真实训练代码不会这样逐元素构造 mask，也不会直接用 NumPy 做大规模计算。

为了帮助理解，可以把运行结果关注在两类指标上：

| 指标 | 观察什么 | 含义 |
|---|---|---|
| `edges` | 边数比全连接少多少 | 理论访问量是否下降 |
| `shortest_path` | 从一个位置到远端位置的最短路径是否缩短 | 长程信息传播是否改善 |

如果把这个思路迁移到真实模型，常见流程通常是：

| 步骤 | 目的 |
|---|---|
| 构造局部窗口规则 | 保留邻近语义和顺序连续性 |
| 指定全局 token | 为关键位置提供全局汇聚与广播通道 |
| 可选添加随机或块级连接 | 改善图连通性，减少纯局部传播的层数压力 |
| 根据模式选择 kernel | 决定理论稀疏是否能转化为真实速度收益 |

伪代码可以写成：

```python
def forward(hidden_states, global_attention_mask=None):
    seq_len = hidden_states.shape[1]

    if use_dense_flash_attention and seq_len <= dense_threshold:
        return dense_flash_attention(hidden_states)

    sparse_mask = build_mask(
        seq_len=seq_len,
        window=attention_window,
        global_mask=global_attention_mask,
        random_blocks=num_random_blocks,
    )
    return sparse_attention(hidden_states, sparse_mask)
```

真实工程里，常见实现路线有三种：

| 实现模式 | 典型场景 | 优点 | 缺点 |
|---|---|---|---|
| PyTorch 原生张量操作 | 原型验证、功能调试 | 可读性高，修改快 | 大序列性能通常差 |
| Triton / TVM / 编译式 kernel | 中期优化 | 性能和可控性较平衡 | 开发门槛较高 |
| 手写 CUDA kernel | 大规模训练、高吞吐推理 | 峰值性能最好 | 维护和调试成本最高 |

以 Hugging Face 风格配置为例，Longformer / BigBird 一般会暴露这类参数：

```python
config.attention_window = 256
config.num_random_blocks = 3

global_attention_mask = [0] * seq_len
global_attention_mask[0] = 1  # [CLS]

for idx in question_token_indices:
    global_attention_mask[idx] = 1
```

真实工程里，一个常见经验是：长文问答任务不要只给 `[CLS]` 一个全局位。更稳妥的做法通常是把整段问题 token 都设为全局，因为 `[CLS]` 只负责汇总，不一定足够表达细粒度问题约束。

---

## 工程权衡与常见坑

稀疏注意力最容易被误解的地方是：理论复杂度下降，不代表端到端训练或推理一定更快。

先把主要权衡列出来：

| 问题 | 直接后果 | 常见规避方法 |
|---|---|---|
| 全局 token 选得太少 | 关键远距依赖传不过去 | QA 任务把问题 token 设为全局；分类任务至少保留 `[CLS]` 和标题 |
| 只有局部窗口，没有跳跃边 | 远程传播需要很多层 | 增大全局 token，或加入随机/块级连接 |
| 窗口过大 | 线性项常数迅速变大 | 从 128、256 这类窗口开始做消融 |
| 稀疏模式过于不规则 | GPU 访存离散，吞吐下降 | 使用块稀疏而不是逐点稀疏 |
| 随机连接模式不稳定 | 不同 seed 带来波动 | 固定随机模式，按块采样 |
| mask 构造本身太慢 | 前处理成为瓶颈 | 预生成模式，避免 Python 循环逐步构造 |
| 理论边数下降但 kernel 不成熟 | 实际速度不升反降 | 先做基准测试，再决定是否替换 dense kernel |

这里必须单独讲 FlashAttention。它不是稀疏注意力，而是密集注意力的高效实现。它的核心思想不是删边，而是减少 HBM 读写，把 softmax、缩放、分块矩阵乘等步骤融合到更紧凑的 tile 计算中。HBM 可以理解为 GPU 上容量大但访问代价高的显存。

这带来三个直接结论：

1. 在中等上下文长度上，FlashAttention 往往已经把密集注意力做得足够快。
2. 若稀疏注意力的实现质量一般，即使理论访问次数更少，也可能跑不过高度优化的密集 kernel。
3. 真正实用的系统常采用混合策略，而不是简单把所有层都改成稀疏注意力。

一个简单判断式可以写成：
$$
\text{Real Speed} \neq f(\text{Theoretical FLOPs}) \text{ only}
$$
更准确地说，真实速度还取决于：
$$
\text{Real Speed} \approx f(\text{FLOPs}, \text{HBM IO}, \text{memory layout}, \text{kernel fusion}, \text{parallelism})
$$

这也是为什么很多论文上的“复杂度更低”，在工程 benchmark 上未必自动兑现成“吞吐更高”。

再看一个常见坑：只给 `[CLS]` 设置全局注意力。

对长文分类，这通常还能工作，因为模型只需要把全文压成一个全局摘要。但对以下任务，这种做法经常不够：

| 任务 | 为什么只给 `[CLS]` 不够 |
|---|---|
| 长文 QA | 问题约束无法直接作用到答案区域 |
| 合同比对 | 多个条款之间需要并行跨段对齐 |
| 检索增强阅读 | 查询词需要直接连接多个证据段 |
| 表格理解 | 字段名、列头、单位等结构锚点需要全局传播 |

更稳的做法通常是把“语义骨架”做成全局图。例如：

1. 问题 token 全局可见。
2. 每个章节标题 token 全局可见。
3. 特殊分隔符、条款编号、字段名按需设为全局。
4. 大量正文 token 仍保持局部窗口。

这种做法的本质不是恢复全连接，而是只给少量“结构性强、语义密度高”的位置保留全局通道。

还要补一个常被忽略的问题：稀疏模式是否与预训练分布一致。若底层模型原本在密集注意力上预训练，而下游阶段突然换成强稀疏模式，模型可能需要重新适应。工程上常见做法是：

| 策略 | 作用 |
|---|---|
| 从预训练阶段就使用目标稀疏模式 | 减少分布偏移 |
| 先 dense 微调，再切换 sparse 微调 | 降低迁移震荡 |
| 混合层设计：前几层局部，后几层 dense 或 global-heavy | 在成本和表达力之间折中 |

---

## 替代方案与适用边界

稀疏注意力不是长上下文的唯一解。常见路线至少有五类：

| 方案 | 理论复杂度 | 连通性特点 | 硬件友好度 | 适用边界 |
|---|---|---|---|---|
| FlashAttention（密集） | 仍是 $O(n^2)$，但 IO 更优 | 完整全局交互 | 高 | 4K 左右到中等长度，且已有成熟实现时非常实用 |
| Longformer | $O(nw + |G|n)$ | 局部 + 指定全局 | 中 | 文档分类、长文 QA、存在明确锚点的任务 |
| BigBird | $O(n(w+g+r))$ | 局部 + 全局 + 随机 | 中偏低 | 更长序列、希望改善图连通性时 |
| Reformer | 常写作 $O(n \log n)$ 量级 | 通过 LSH 近似找相似位置 | 中偏低 | 可接受近似分桶误差的场景 |
| Linformer | $O(nk)$ | 用低秩投影近似全局注意力 | 中高 | 注意力矩阵低秩假设较合理时 |

这些方案的出发点并不相同。

Reformer 的重点不是窗口，而是 LSH 分桶。它假设相似 token 可以通过哈希近似聚在一起，然后只在桶内做注意力。因此它是在做“近邻检索式近似”。

Linformer 的重点也不是删边，而是低秩近似。它假设注意力矩阵可以投影到较低维空间，用更小的 $k$ 来替代完整长度维度，因此复杂度写成 $O(nk)$。

什么时候优先考虑哪种方案，可以用下面的经验边界：

1. 序列不算极长，且已有成熟 FlashAttention 实现时，先验证密集注意力是否已经满足吞吐和显存要求。
2. 序列较长，任务中存在明确锚点，例如问题区间、标题、章节编号，此时 Longformer 往往最直接。
3. 序列更长，担心纯局部传播和少量全局位仍然不够时，BigBird 通常更稳。
4. 如果团队对 kernel 维护成本非常敏感，优先选择已有成熟实现的路线，而不是只看论文复杂度。
5. 如果部署目标是通用 GPU，而不是专门为块稀疏优化的硬件，先做 benchmark，再决定是否上稀疏。

一个常见经验是：

- 在 4K 左右上下文，先验证密集 FlashAttention 是否已经够快。
- 在 8K、16K 甚至更长上下文，$n^2$ 成本开始明显压缩 batch size、训练吞吐和显存预算，这时再认真评估 Longformer 或 BigBird 往往更有意义。

还可以再给一个更工程化的选择表：

| 需求 | 更优先考虑 |
|---|---|
| 需要完整全局交互，长度中等 | FlashAttention |
| 文档有清晰结构锚点 | Longformer |
| 想在近线性成本下提高全局连通性 | BigBird |
| 愿意接受近似最近邻搜索 | Reformer |
| 认为注意力矩阵具有低秩结构 | Linformer |

---

## 参考资料

下面给出更完整的参考资料，并说明各自应该重点看什么。

| 标题 | 来源 | 重点 |
|---|---|---|
| Longformer: The Long-Document Transformer | Iz Beltagy, Matthew E. Peters, Arman Cohan, 2020 | 滑动窗口注意力、全局注意力、长文档任务设计 |
| Big Bird: Transformers for Longer Sequences | Manzil Zaheer et al., 2020 | 局部 + 全局 + 随机稀疏结构，图连通性与理论性质 |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Tri Dao et al., 2022 | 为什么密集注意力在工程上仍然可以非常快 |
| Self-Attention with Sparse Patterns | 稀疏注意力相关综述/教程材料 | 适合补齐不同稀疏模式的横向比较 |
| Hugging Face Longformer 文档与实现 | Transformers 文档/源码 | `attention_window`、`global_attention_mask` 的具体用法 |
| Hugging Face BigBird 文档与实现 | Transformers 文档/源码 | `num_random_blocks`、块稀疏实现细节 |
| Reformer: The Efficient Transformer | Nikita Kitaev et al., 2020 | LSH 注意力与可逆层设计 |
| Linformer: Self-Attention with Linear Complexity | Sinong Wang et al., 2020 | 低秩投影近似注意力 |

建议阅读顺序可以这样安排：

1. 先读 Longformer，建立“局部窗口 + 全局 token”的主框架。
2. 再读 BigBird，理解为什么随机连接可以改善图直径与传播路径。
3. 然后读 FlashAttention，补上“理论更省不代表工程更快”的实现视角。
4. 最后按需看 Reformer 和 Linformer，用来区分“稀疏化”“分桶近似”“低秩近似”三条不同路线。

如果只记住一条主线，可以记成：

- Longformer 解决的是“哪些位置必须保留全局通道”；
- BigBird 解决的是“如何在近线性成本下进一步增强全局连通性”；
- FlashAttention 解决的是“即使不删边，也能把密集注意力做得更高效”。

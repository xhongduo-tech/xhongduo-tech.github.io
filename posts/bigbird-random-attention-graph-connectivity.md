## 核心结论

BigBird 的关键不是“把注意力做稀疏”，而是把稀疏模式设计成一个仍然能快速连通全局的图。这里的图可以先理解为一张网络：每个 token 或 block 是一个点，谁能看见谁，就是一条边。

BigBird 每层的注意力由三部分组成：

1. 局部窗口 attention：每个位置只看附近的 $w$ 个位置或若干相邻 blocks，负责短程信息交换。
2. 全局 token attention：少量特殊位置与很多节点相连，负责汇聚和广播全局信息。
3. 随机 attention：每个 block 额外连 $r$ 条远距离随机边，负责缩短长程路径，避免图退化成近似链式结构。

论文里的核心理论结论可以准确表述为：在包含局部边、全局边和常数级随机边的 BigBird 稀疏模式下，模型仍然保留与 full attention 同级别的若干理论性质，包括 universal approximation 和 Turing completeness。直白地说，边数减少了，但在“能表示什么函数”这个层面没有本质降级。这里的结论依赖的是**特定结构组合**，不是“任意稀疏 attention 都成立”。

如果只用滑动窗口，那么序列首尾之间的信息传播层数大约是 $O(n/w)$；加入随机边后，图的最短路径长度会以高概率下降到接近对数级，常写成近似 $O(\log n)$。这里的“图直径”是图中任意两点最短路径的最大值，可以理解成“最坏情况下消息要跳几次才能传到最远处”。

| 方案 | 每个位置可见范围 | 图直径趋势 | 是否有随机边 | 理论保证 |
| --- | --- | --- | --- | --- |
| Full Attention | 全部 $n$ 个位置 | $O(1)$ | 否 | 有 |
| Longformer | 局部 + 少量全局 | 近似 $O(n/w)$ | 否 | 没有 BigBird 同级证明 |
| BigBird | 局部 + 全局 + 随机 | 高概率接近 $O(\log n)$ | 是 | 有 |

一个最小玩具例子：设 $n=1024$，按块视角把窗口等效成每次能向前传播 $w=16$ 个位置。纯局部窗口下，序列头部想把信息传到尾部，大约需要 $\lceil 1024/16 \rceil=64$ 层；如果每个位置再增加常数条随机远程连接，最短路径的量级会更接近 $\log_2 1024=10$。BigBird 理论真正利用的就是这个数量级差异。

---

## 问题定义与边界

要理解 BigBird，先要定义它到底解决什么问题。

标准 Transformer 的 full attention 会让每个 token 与所有 token 计算注意力，时间和显存代价都是 $O(n^2)$。当序列长度从 512 提高到 4096、8192 甚至更长时，这个成本会很快失控。因此很多工作都在做 sparse attention，也就是只保留一部分边，把复杂度压到线性或近线性。

但稀疏不是随便删边。删错了，模型虽然更省，信息却传不动，表达能力也会下降。最典型的反例就是纯局部窗口结构：它每层只能在邻域内扩散信息，跨远距离依赖必须靠很多层逐步传递。对长文档问答、长序列分类、基因组建模这类任务，这会直接成为效果瓶颈。

形式化一点，给长度为 $n$ 的序列定义一个注意力图 $G=(V,E)$，其中：

- 节点集 $V$ 对应序列中的 token 或 block。
- 有向边 $(i,j)\in E$ 表示位置 $i$ 可以 attend 到位置 $j$。
- 如果把双向可见关系合并看成无向边，图直径定义为
  $$
  \mathrm{diam}(G)=\max_{u,v\in V} d(u,v)
  $$
  其中 $d(u,v)$ 是从 $u$ 到 $v$ 的最短路径长度。

为什么直径重要？因为一层 attention 可以理解为沿着一跳邻接边做一次信息混合；堆叠 $L$ 层后，一个节点最多只能聚合 $L$ 跳范围内的信息。因此：

$$
L \ge d(u,v)
$$

是节点 $u$ 获取节点 $v$ 信息的一个基本下界。图直径越大，最坏情况下需要的层数越多。

如果注意力图接近一条链，那么它的直径随序列长度线性增长；如果图中存在少量但有效的远程 shortcut，直径就会显著降低。这就是这里要说的小世界效应：不是把所有边都补回来，而是用少量远程边打断“只能一站一站挪”的传播方式。

为了避免讨论范围失控，本文只覆盖下面这些内容：

| 讨论对象 | 本文是否覆盖 | 说明 |
| --- | --- | --- |
| BigBird 的图连通性与传播层数 | 是 | 重点内容 |
| BigBird 的 universal approximation / Turing completeness 结论 | 是 | 解释其依赖的结构条件 |
| 完整复现论文证明细节 | 否 | 只保留对工程理解必要的部分 |
| CUDA/TPU 内核级优化 | 否 | 只讲 block-sparse 设计思想 |
| 稀疏注意力所有变体 | 否 | 只与 Longformer、Full Attention 做必要对比 |

BigBird 的可见性掩码可以写成：

$$
M(i,j)=M_{\text{local}}(i,j)\ \lor\ M_{\text{global}}(i,j)\ \lor\ M_{\text{random}}(i,j)
$$

如果把它写成矩阵和的形式，也常见到：

$$
A(i,j)=A_{\text{local}}(i,j)+A_{\text{global}}(i,j)+A_{\text{random}}(i,j)
$$

这里的含义是“由三类边共同定义允许访问的位置”，不是把三类 attention 分数直接数值相加。真正计算时，先由它们形成 mask，再在允许访问的子图上做标准 attention。

---

## 核心机制与推导

BigBird 的核心机制可以分成两件事：

1. 它怎样把 full attention 改写成常数出度的稀疏图。
2. 这种稀疏图为什么仍然能快速把全局信息混合起来。

先看结构。BigBird 通常按 block 分析，而不是按单个 token 分析。原因是理论上更清楚，工程上也更接近真实实现。

设整条序列被切成 $m$ 个 blocks。对某个 query block，BigBird 只连接三类邻居：

- 左右相邻窗口 blocks。
- 少量全局 blocks。
- $r$ 个随机 blocks。

如果每个 block 看到的邻居数量是常数，那么单层计算量就从 $O(m^2)$ 降为 $O(m)$。把 block 大小视为常数时，这对应 token 视角的近线性复杂度。

设一个 query block 连接：

- 左右窗口块共约 $2w+1$ 个。
- $g$ 个全局块。
- $r$ 个随机块。

那么每个块的出度满足：

$$
\deg(i) \approx (2w+1)+g+r
$$

只要 $w,g,r$ 不随 $m$ 增长，这就是常数出度图，因此总边数是：

$$
|E| = O\big(m(w+g+r)\big)=O(m)
$$

### 1. 纯滑窗为什么慢

如果只有局部窗口，每层最多把信息向前推进约 $w$ 个 blocks。于是头尾之间的传播层数至少是：

$$
L_{\text{local}} \gtrsim \left\lceil \frac{m-1}{w} \right\rceil
$$

把 block 换回 token 视角，本质上还是线性增长。

玩具例子：

- 序列长度 $n=128$
- 每个 token 只能看左右各 3 个位置，可视为有效窗口跨度 $w=3$

那么首尾之间的最短路径大约是：

$$
\left\lceil \frac{127}{3} \right\rceil = 43
$$

这意味着即使每层都充分利用局部 attention，模型也要很多层才能把尾部信息稳定传播到首部。

### 2. 随机边为什么有效

随机边的作用不是“补一点远程信息”，而是改写整个图的拓扑结构。纯局部窗口的图近似是一条带宽很窄的链；加入随机边后，它更像一个小世界图或稀疏扩展图，局部团块之间会出现 shortcut。

论文里的论证不是说“任意一次随机采样都必然完美”，而是说当每个 block 保留常数级随机连接时，图会以高概率具有良好的连通性和扩展性。这里“扩展性”可以先理解为：

- 从任意一个小集合出发，
- 沿着边走一两步，
- 能接触到很多此前没见过的新节点，
- 而不是一直在局部小团里打转。

用一个常见的启发式估计：假设每一层传播后，前沿集合大致能扩展一个固定倍数 $\beta>1$，那么经过 $L$ 层后可触达节点数近似为：

$$
|R_L| \approx \beta^L
$$

要覆盖全部 $m$ 个 blocks，只需满足：

$$
\beta^L \gtrsim m
\quad\Longrightarrow\quad
L \gtrsim \log_\beta m
$$

这就是为什么随机边会把路径长度从线性量级压到对数量级。这个推导不是论文原封不动的正式证明，但它抓住了论文证明依赖的核心直觉：**随机远程边带来了快速扩张。**

### 3. global token 为什么还需要

随机边已经能降低直径，为什么还要 global token？因为两者分工不同。

- global token 提供稳定的中心节点，用来做信息汇聚和广播。
- random edge 提供概率性的跨区 shortcut，用来防止图退化成多个局部团块。

只用 global token，图确实能借助中心节点缩短部分路径，但信息流会高度依赖少数枢纽；只用 random edge，图虽可能有较低直径，但缺少稳定的全局读写通道。BigBird 选择的是混合结构：局部边负责邻域建模，全局边负责稳定汇聚，随机边负责打通远程路径。

一个很实用的理解方式是把三类边对应到三类职责：

| 组件 | 主要职责 | 失去后会发生什么 |
| --- | --- | --- |
| Local | 保留邻近顺序与局部模式 | 局部语义变差 |
| Global | 稳定汇聚与广播 | 全局读写依赖随机性，训练更不稳 |
| Random | 缩短最坏路径、增强扩展性 | 图更接近长链，层数需求变大 |

### 4. 理论性质为什么还能保住

BigBird 论文的重要结论不是“经验上效果还不错”，而是证明这种结构仍然足够强，能保住一些全注意力模型的重要理论性质。这里最容易误解的一点是：结论不是“只靠随机边就够了”，而是依赖下面这组结构条件。

1. 存在少量全局节点，能够承担类似共享内存或汇聚缓冲区的角色。
2. 存在局部窗口，保留邻域交互和顺序归纳偏置。
3. 存在常数级随机远程连接，使图在深度方向上快速混合，不会被局部链式结构卡死。

因此随机边不是工程装饰，而是理论结构的一部分。没有它，图的最坏路径太长；没有 global token，很多需要统一读写或聚合的证明构件又搭不起来。论文的 universal approximation 与 Turing completeness 结论依赖的是这套组合，而不是“任意稀疏 mask 都行”。

真实工程里，这个结构优势最容易体现在长文档多跳问答。以 HotpotQA 一类任务为例，模型往往需要：

1. 在一段文字里找到实体；
2. 跳到另一段文字里找到关系；
3. 再回到问题上下文做汇总。

纯局部窗口下，这条路径可能要跨很多层；BigBird 通过全局块和随机块把跨段传播路径压短，所以它在长上下文场景中比单纯窗口结构更有优势。

---

## 代码实现

工程实现里，BigBird 通常不是按单个 token 随机连边，而是按 block 做稀疏。原因很直接：硬件更擅长处理连续块矩阵，不擅长做大量零碎的逐元素 gather。

实现思路可以概括为：

1. 把序列切成固定大小的 blocks。
2. 对每个 query block，确定它能访问的 window blocks、global blocks 和 random blocks。
3. 把这些 key/value blocks gather 出来并拼接。
4. 在拼接后的局部 dense 张量上做普通矩阵乘法和 softmax。

也就是说，核心不是“写一个逐元素极稀疏的 attention kernel”，而是“用块级 gather 把稀疏问题转成小规模 dense matmul”。

下面给一个可运行的 Python 玩具实现。它不计算真正的 attention 分数，只构造 BigBird 风格的块级邻接图，并测量图直径。代码额外做了三件原文里缺失的事情：

- 明确保证图是无向可达的，直径一定可定义。
- 同时比较 `local only`、`local + global`、`local + global + random` 三种结构。
- 对多个随机种子取均值，避免单次采样碰巧过好或过坏。

```python
from __future__ import annotations

from collections import deque
import random
from statistics import mean


def build_bigbird_graph(
    num_blocks: int,
    window: int,
    global_block_ids: list[int],
    num_random: int,
    seed: int,
) -> dict[int, set[int]]:
    """
    构造一个无向 block 图。
    - num_blocks: block 总数
    - window: 每个 block 向左右各连接多少个邻居
    - global_block_ids: 充当 global blocks 的编号
    - num_random: 每个非全局 block 额外采样多少个随机邻居
    """
    assert num_blocks > 0
    assert window >= 0
    assert num_random >= 0
    assert all(0 <= g < num_blocks for g in global_block_ids)

    rng = random.Random(seed)
    graph = {i: set() for i in range(num_blocks)}
    global_set = set(global_block_ids)

    def add_edge(u: int, v: int) -> None:
        graph[u].add(v)
        graph[v].add(u)

    for i in range(num_blocks):
        add_edge(i, i)  # 自环只为实现简单，BFS 时无影响

        # 1) local window
        left = max(0, i - window)
        right = min(num_blocks - 1, i + window)
        for j in range(left, right + 1):
            add_edge(i, j)

        # 2) global blocks
        for g in global_set:
            add_edge(i, g)

        # 3) random blocks
        blocked = graph[i] | {i}
        candidates = [j for j in range(num_blocks) if j not in blocked]
        if candidates:
            k = min(num_random, len(candidates))
            for j in rng.sample(candidates, k=k):
                add_edge(i, j)

    return graph


def bfs_eccentricity(graph: dict[int, set[int]], start: int) -> int:
    q = deque([(start, 0)])
    seen = {start}
    farthest = 0

    while q:
        node, dist = q.popleft()
        farthest = max(farthest, dist)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))

    if len(seen) != len(graph):
        raise ValueError("graph is disconnected")
    return farthest


def graph_diameter(graph: dict[int, set[int]]) -> int:
    return max(bfs_eccentricity(graph, start=i) for i in graph)


def run_experiment() -> None:
    num_blocks = 128
    seeds = list(range(10))

    local_only = []
    local_global = []
    bigbird_like = []

    for seed in seeds:
        g1 = build_bigbird_graph(
            num_blocks=num_blocks,
            window=1,
            global_block_ids=[],
            num_random=0,
            seed=seed,
        )
        g2 = build_bigbird_graph(
            num_blocks=num_blocks,
            window=1,
            global_block_ids=[0, 1],
            num_random=0,
            seed=seed,
        )
        g3 = build_bigbird_graph(
            num_blocks=num_blocks,
            window=1,
            global_block_ids=[0, 1],
            num_random=3,
            seed=seed,
        )

        local_only.append(graph_diameter(g1))
        local_global.append(graph_diameter(g2))
        bigbird_like.append(graph_diameter(g3))

    print("avg diameter (local only):           ", mean(local_only))
    print("avg diameter (local + global):       ", mean(local_global))
    print("avg diameter (local + global + rnd): ", mean(bigbird_like))

    assert mean(local_only) > mean(local_global)
    assert mean(local_global) >= mean(bigbird_like)


if __name__ == "__main__":
    run_experiment()
```

在一组典型参数下，你通常会看到类似结果：

| 结构 | 平均图直径的大致量级 |
| --- | --- |
| `local only` | 60 左右 |
| `local + global` | 2 到 4 |
| `local + global + random` | 2 到 3 |

这里有一个需要讲清楚的细节：如果 global blocks 和所有节点双向全连接，那么直径本来就会非常小。那为什么 BigBird 论文还强调 random edge？

原因是论文关心的不只是“存在一条极短路径”，还关心更强的结构性质，包括稳定的信息扩散、避免图在去掉部分特殊节点后退化，以及支持理论证明所需的稀疏模式。global 节点给出的是中心化捷径，random 边给出的是更分散的扩展性，两者不是完全替代关系。

一个更接近真实工程的伪代码如下：

```python
for q_block in query_blocks:
    visible = []
    visible += local_neighbor_blocks(q_block, window_blocks=3)
    visible += all_global_blocks()
    visible += random_blocks(q_block, num_random_blocks=4, seed=fixed_seed)

    visible = unique_and_sort(visible)

    k_cat = gather_blocks(K, visible)
    v_cat = gather_blocks(V, visible)

    scores = q_block @ k_cat.T / sqrt(head_dim)
    probs = softmax(scores + build_mask(q_block, visible))
    out[q_block] = probs @ v_cat
```

这里最重要的工程点有三个：

1. `random_blocks(...)` 通常要固定 seed 或预先生成 mask，避免训练和推理看到的图结构不一致。
2. `gather_blocks(...)` 应按 block 连续取数，否则稀疏访问本身就会把性能吃掉。
3. `unique_and_sort(...)` 很重要。真实实现里如果同一个 block 同时属于 local、global、random 三类，必须去重，否则会重复 gather。

如果把复杂度写得更明确一点，设 block 数为 $m=n/b$，每个 query block 可见的总 block 数为 $k=(2w+1)+g+r$，则单层复杂度近似是：

$$
O(m \cdot k \cdot b^2 \cdot d)
$$

当 $b,w,g,r$ 都是常数时，可化成：

$$
O(n d)
$$

这就是 BigBird 稀疏设计的工程收益来源。

---

## 工程权衡与常见坑

BigBird 的真实价值主要体现在长上下文任务上，而不是短文本基准。因为只有序列足够长时，$O(n^2)$ 和近线性复杂度的差距才会真正转化成可训练、可部署的收益。

真实工程例子可以看长文档问答和长文本分类。比如 HotpotQA、Natural Questions、TriviaQA 这类任务，输入经常超过普通 BERT 可承受的长度。BigBird 的 block-sparse 设计允许模型在 4K 甚至更长上下文内保留可接受的显存与计算成本，同时通过随机块加速跨证据片段的信息流动。

常见权衡如下：

| 维度 | BigBird 的收益 | BigBird 的代价 |
| --- | --- | --- |
| 复杂度 | 近线性，适合长序列 | 实现明显更复杂 |
| 表达能力 | 有较强理论保证 | 证明依赖 local + global + random 的组合结构 |
| 硬件效率 | block-sparse 时较好 | 小序列下常不如 dense kernel |
| 稳定性 | 固定 mask 时较稳 | 采样策略不当会导致波动 |

常见坑主要有这些：

| 常见坑 | 现象 | 为什么会出问题 | 规避方法 |
| --- | --- | --- | --- |
| `seq_len` 不能整除 `block_size` | 边界块逻辑复杂，mask 分支变多 | 最后一个 block 不完整，gather 和 pad 都更麻烦 | 训练前统一 pad 到块边界 |
| 每层重新随机采样 | 训练和推理行为不一致，结果抖动 | 模型每层看到的是不同图，优化更难稳定 | 固定 seed 或离线预生成随机 mask |
| `num_random_blocks=0` | 退化成局部+全局结构 | 图的扩展性下降，理论条件也被破坏 | 长序列任务不要省掉随机边 |
| 短序列也强上稀疏 | 速度未提升，甚至更慢 | 稀疏 gather 的开销盖过了计算节省 | 小于 1K 时优先实测 dense |
| token 级稀疏访问过细 | 实际速度比 dense 更慢 | 内存访问离散，硬件不擅长 | 按 block gather，再做 dense matmul |
| global blocks 太少 | 汇聚能力不足 | 多跳任务里缺少稳定广播通道 | 至少保留少量固定 global 位置 |
| global blocks 太多 | 稀疏度被吃掉 | 中心节点太多会逼近 dense | 只保留任务必需的少量全局块 |

参数上也有经验边界：

| 场景 | 建议配置趋势 | 备注 |
| --- | --- | --- |
| 短文本分类（<512） | 直接 full attention | 简单且通常更快 |
| 中长文档分类（1K-4K） | 中等窗口 + 少量全局 + 少量随机 | 先保证块对齐和 kernel 可用 |
| 多跳 QA（4K+） | 保留 random blocks，global blocks 不能太少 | 需要跨段证据融合 |
| 基因组/长序列建模 | 较长序列、稳定随机图 | 更依赖远距离联系 |

一个容易误解的点是：BigBird 不保证“随机边越多越好”。随机边增加会改善连通性，但也会增加 gather 成本、显存占用和 mask 管理复杂度。工程目标不是把稀疏模式重新堆回 full attention，而是在有限的稀疏预算下保住图的全局可达性。

---

## 替代方案与适用边界

如果任务不长，或者并不依赖远距离多跳推理，BigBird 不一定是最优解。

Full attention 的优点是实现最直接、库支持最好、行为最稳定。短序列场景下，它通常反而更快，因为不需要 block 切分、mask 构造和稀疏 gather。

Longformer 这类“局部 + 少量全局”方案，比 BigBird 更简单，也更容易落地。但它缺少随机边，因此图结构更接近“长链 + 少数中心节点”。对很多任务它已经够用，但如果你关心最坏路径长度、全局扩展性，或者任务确实依赖大范围证据交互，BigBird 的结构更完整。

| 方案 | 适用场景 | 不适合场景 |
| --- | --- | --- |
| Full Attention | 短序列、实现优先、调试优先 | 超长序列 |
| Longformer | 长文本编码、实现复杂度要低 | 强依赖理论保证或复杂跨段传播 |
| BigBird | 超长上下文、多跳推理、长距离依赖明显 | 短序列、小模型、追求最简单实现 |

选择标准可以压缩成一句话：

- 小于 512 或 1K 的输入，优先 full attention。
- 需要长文本但工程复杂度要低，可以选 Longformer。
- 需要 4K 以上上下文，并且希望稀疏结构仍保留较强理论性质，选 BigBird。

所以 BigBird 的适用边界很明确：它不是“任何 Transformer 都该换上的默认结构”，而是“当序列足够长、局部传播已经成为瓶颈时，能同时兼顾复杂度、图连通性和理论表达能力的方案”。

---

## 参考资料

下面的参考资料不只是“列出标题”，而是按阅读用途拆开说明。

| 资料 | 核心贡献 | 建议阅读方式 |
| --- | --- | --- |
| Big Bird: Transformers for Longer Sequences, NeurIPS 2020 | 主论文。给出 block sparse 设计、实验结果和理论结论 | 先看 sparse pattern 图示和 complexity，再看理论部分的定理陈述 |
| Big Bird Supplemental Material | 补充定理证明、构造细节和附加实验 | 只在你需要追踪 universal approximation / Turing completeness 证明时再看 |
| Google Research Blog: Constructing Transformers for Longer Sequences with Sparse Attention Methods | 面向工程读者的直观解释，适合快速建立整体认识 | 先读这一篇，再回头看论文公式会更容易 |
| Michael Brenndoerfer 的 BigBird 讲解 | 对论文做了更口语化的二次整理 | 适合作为预热材料，不建议替代原论文 |

参考链接：

- Big Bird: Transformers for Longer Sequences, NeurIPS 2020  
  https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf
- Big Bird Supplemental Material  
  https://proceedings.neurips.cc/paper_files/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Supplemental.pdf
- Constructing Transformers for Longer Sequences with Sparse Attention Methods, Google Research Blog  
  https://research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/
- BigBird: Sparse Attention with Random Connections for Long Documents, Michael Brenndoerfer  
  https://mbrenndoerfer.com/writing/bigbird-sparse-attention-random-connections-long-documents

## 核心结论

Longformer 和 BigBird 都在解决同一个问题：把自注意力从 $O(n^2)$ 降到近似 $O(n)$，让模型能处理更长的文本。这里的“自注意力”可以先理解成“每个 token 决定要看哪些其他 token”。

两者的核心区别不在“是不是稀疏”，而在“稀疏图怎么连”：

- Longformer 使用**确定性稀疏模式**，也就是预先写死的连接规则：滑动窗口、可选扩张窗口、少量全局 token。
- BigBird 在局部窗口和全局 token 之外，再加入**随机连接**，也就是每个块额外连到一些随机远处块。

对零基础读者，可以先用一个玩具类比理解：Longformer 像多条平行的固定区间轨道，车只能按既定线路跑；BigBird 则在轨道之间额外架起临时快线，让远处信息更快相遇。

这带来三个直接结论：

| 维度 | Longformer | BigBird |
|---|---|---|
| 稀疏组成 | 滑动窗口 + 全局 token | 滑动窗口 + 全局 token + 随机连接 |
| 复杂度近似 | $n \cdot (2w+g)$ | $n \cdot (2w+g+r)$ |
| 图结构 | 确定、可解释 | 半确定、半随机 |
| 信息传播速度 | 主要靠层数逐步扩散 | 随机捷径缩短平均路径 |
| 工程实现 | 更简单 | 更复杂，需要采样与复现控制 |
| 理论性质 | 强工程可控性 | 更接近全注意力的理论保证 |

其中：

- $n$ 是序列长度，也就是 token 总数。
- $w$ 是窗口半径，也就是左右各看多少个邻居。
- $g$ 是全局 token 数，也就是允许看全局、也被全局看的特殊位置数。
- $r$ 是随机连接数，也就是每个位置或块额外增加的远程跳跃边数。

如果你的目标是高吞吐、稳定部署、可解释的注意力模式，Longformer 通常更合适。如果你的目标是尽量保留长程依赖，接受更复杂的实现控制，BigBird 更接近“稀疏版全注意力”。

---

## 问题定义与边界

问题可以精确定义为：

给定长度为 $n$ 的序列，如何在不做全连接注意力的前提下，让每个 token 只连接少量位置，但仍尽量保留跨段信息传播能力。

这里的“跨段信息”可以理解成：一句话里的词，能不能影响很远位置上的另一句话。对长文问答、合同分类、长文摘要，这一点很关键。

先定义统一记号。对第 $i$ 个 token，它能关注的位置集合记为 $\text{Attend}(i)$。则一个常见的稀疏注意力近似写法是：

$$
|\text{Attend}(i)| = (2w+1) + g + r
$$

含义是：

- $(2w+1)$：本地窗口，含自己、左边 $w$ 个、右边 $w$ 个。
- $g$：所有全局 token。
- $r$：额外随机连接。

在 Longformer 中，通常可视为：

$$
|\text{Attend}_{\text{Longformer}}(i)| \approx (2w+1) + g
$$

因为它没有 BigBird 那种专门的随机块连接项，所以可近似看作 $r=0$。

在 BigBird 中：

$$
|\text{Attend}_{\text{BigBird}}(i)| \approx (2w+1) + g + r
$$

它的关键新增量就是 $r$。

一个新手更容易懂的问答例子是：

- $n$：整篇文档被切成多少 token。
- $w$：每个 token 左右能直接看到多少距离。
- $g$：像 `[CLS]`、问题 token、标题 token 这种被赋予全局访问能力的位置数。
- $r$：额外随机打通的远程连接数。

边界也要说清楚。本文比较的是**稀疏模式本身**，不是完整模型所有细节，因此不展开：

- 不比较预训练语料差异。
- 不比较不同 hidden size、层数、参数量。
- 不比较不同 tokenizer 对结果的影响。
- 不把块稀疏实现细节和论文中的理论形式完全混为一谈，重点看连接图的性质与工程代价。

因此，本文的核心边界是：比较 Longformer 与 BigBird 的**连接规则、复杂度、信息传播路径、工程可部署性**。

---

## 核心机制与推导

先看最朴素的情况。全注意力下，每个 token 都看全部 $n$ 个位置，所以一层的连接数近似为：

$$
n \cdot n = O(n^2)
$$

这在长序列上会很快爆掉显存和算力。

### Longformer 的机制

Longformer 的核心是局部窗口注意力。直观上，每个 token 只看近邻。术语“滑动窗口”第一次出现时，可以理解成“一个固定大小的观察框，沿序列逐格移动”。

若不考虑边界效应，每个 token 关注 $(2w+1)+g$ 个位置，则总连接数近似为：

$$
E_{\text{Longformer}} \approx n \cdot ((2w+1)+g)
$$

因为 $w$ 和 $g$ 一般远小于 $n$，所以复杂度是：

$$
O(n(2w+g)) \approx O(n)
$$

但它的代价是，远距离信息传播必须靠多层逐步传递。若只有局部窗口，没有足够全局 token，那么一条信息从序列左端传到右端，所需步数近似正比于距离：

$$
\text{diameter} \propto O(n / w)
$$

这里的“图直径”可以先理解成“最远两点之间最短路径的长度上界”。如果图直径大，说明远处信息要经过很多跳才能传过去。

### BigBird 的机制

BigBird 在 Longformer 的基础上额外加入随机连接。术语“随机连接”第一次出现时，可以理解成“预先抽样的一批远程捷径边”。

于是总连接数变成：

$$
E_{\text{BigBird}} \approx n \cdot ((2w+1)+g+r)
$$

复杂度仍然是线性的：

$$
O(n(2w+g+r)) \approx O(n)
$$

因为通常 $w,g,r$ 都被设成常数或小常数级别。

BigBird 的关键不是把复杂度再降，而是让图的连通性更像“小世界网络”。直观上，局部窗口保证附近信息交换，随机边让远处节点以更短路径相连。论文层面的理论结论是，这种结构在一定条件下可以获得接近全注意力的表达能力，并把平均最短路径压到对数级别附近：

$$
\mathbb{E}[\text{shortest path}] \approx O(\log n)
$$

这不是说任意一次具体采样都严格等于 $\log n$，而是说随着随机远程边加入，图的平均传播距离显著缩短。

### 玩具例子：n=64

设：

- $n=64$
- $w=3$
- $g=2$
- $r=3$

则单个非全局 token 的连接数近似为：

- Longformer：$(2 \times 3 + 1) + 2 = 9$
- BigBird：$(2 \times 3 + 1) + 2 + 3 = 12$

如果把边界修正、全局 token 双向连接等细节折进总数，一个常见的示意性统计是：

| 模式 | 总关注位置数（示意） | 相对全注意力 |
|---|---:|---:|
| 全注意力 | 4096 | 100% |
| Longformer | 674 | 16.5% |
| BigBird | 860 | 21.0% |

这说明 BigBird 虽然比 Longformer 多出随机连接，但仍远低于全注意力。新增连接比例大约是：

$$
\frac{860 - 674}{674} \approx 27.6\%
$$

也就是随机捷径不是“免费”的，它要用额外连接成本来换更短路径。

### 信息传播图示

可以用文字图看差异：

```text
Longformer:
1 - 2 - 3 - 4 - 5 - 6 - 7 - 8
      |G|               |G|

BigBird:
1 - 2 - 3 - 4 - 5 - 6 - 7 - 8
 \        /         \      /
   random            random
      |G|               |G|
```

这里 `G` 表示全局 token。Longformer 主要靠局部边和全局点扩散；BigBird 多了远程跳边，所以从 1 到 8 不一定要逐层绕过去。

### 真实工程例子：长文法律分类

在法律文档分类或合同审查中，一份文档可能有几千到上万 token。比如：

- 开头定义术语；
- 中间多页出现义务条款；
- 末尾出现免责条款或生效条件。

如果模型只能看局部窗口，那么“定义”影响“免责”需要经过很多层中转。Longformer 可以通过把标题、问题 token、`[CLS]` 设成全局 token 缓解这个问题，但你得先知道谁重要。BigBird 则通过随机远程边，让“定义段”和“免责段”更可能在较短路径内产生交互，即使你没有手工标出所有关键位置。

这就是两者最本质的取舍：Longformer 依赖**人为指定的重要节点**，BigBird 依赖**图结构中的随机捷径**。

---

## 代码实现

工程上，Longformer 更像“先画好固定 mask”，BigBird 更像“固定模板上再加一层可复现的随机 mask”。

下面给一个可运行的 Python 玩具实现，用布尔矩阵构造两种注意力 mask。这里的“mask”可以理解成“一张黑白连接图，1 表示能看，0 表示不能看”。

```python
import random

def longformer_mask(n, w, global_tokens=None):
    if global_tokens is None:
        global_tokens = []
    gset = set(global_tokens)

    mask = [[0] * n for _ in range(n)]

    for i in range(n):
        # 局部窗口
        left = max(0, i - w)
        right = min(n, i + w + 1)
        for j in range(left, right):
            mask[i][j] = 1

        # 所有 token 都能看全局 token
        for j in gset:
            mask[i][j] = 1

        # 全局 token 反过来看所有位置
        if i in gset:
            for j in range(n):
                mask[i][j] = 1

    return mask

def bigbird_mask(n, w, global_tokens=None, r=0, seed=0):
    if global_tokens is None:
        global_tokens = []
    gset = set(global_tokens)
    rng = random.Random(seed)

    mask = longformer_mask(n, w, global_tokens)

    for i in range(n):
        if i in gset:
            continue

        # 候选随机连接：排除窗口内、自己、全局 token
        candidates = []
        for j in range(n):
            in_window = (max(0, i - w) <= j < min(n, i + w + 1))
            if j != i and j not in gset and not in_window:
                candidates.append(j)

        picks = rng.sample(candidates, k=min(r, len(candidates)))
        for j in picks:
            mask[i][j] = 1

    return mask

def count_edges(mask):
    return sum(sum(row) for row in mask)

# 玩具例子
n, w = 16, 2
global_tokens = [0, 8]

m1 = longformer_mask(n, w, global_tokens)
m2 = bigbird_mask(n, w, global_tokens, r=2, seed=42)

# 基本正确性检查
assert m1[0][15] == 1          # 全局 token 看全局
assert m1[7][0] == 1           # 普通 token 能看全局 token
assert m1[7][15] == 0          # Longformer 普通 token 不会凭空看远处
assert count_edges(m2) >= count_edges(m1)  # BigBird 连接数更多

# seed 固定时可复现
m2_again = bigbird_mask(n, w, global_tokens, r=2, seed=42)
assert m2 == m2_again

print("Longformer edges:", count_edges(m1))
print("BigBird edges:", count_edges(m2))
```

这段代码体现了两个实现重点。

第一，Longformer 的 mask 是纯确定性的。你给定 `n`、`w`、全局 token 列表，结果就固定。这对 CUDA kernel 或块稀疏 kernel 很友好，因为访问模式稳定，容易提前优化。

第二，BigBird 的随机连接必须**固定 seed**。否则训练一次和推理一次得到的稀疏图不同，会出现两个问题：

- 结果不可复现，调参很难定位。
- 注意力图每次变化，底层 kernel 很难稳定优化。

一个简化伪代码如下：

```python
seed = 1234
rng = Random(seed)

for each block i:
    local_neighbors = fixed_window(i, w)
    global_neighbors = global_blocks
    random_neighbors = sample_blocks(rng, i, r)
    attend_to = local_neighbors + global_neighbors + random_neighbors
```

真实工程里，BigBird 常按“块”采样而不是按单 token 采样，因为块稀疏更适合 GPU。这里可以把“块”先理解成“把连续 token 打包成小段统一处理”。

对新手来说，最直接的理解是：

- Longformer：直接给每个 token 一张固定黑白图。
- BigBird：先给固定黑白图，再额外打一些随机点，而且这些随机点必须固定下来。

---

## 工程权衡与常见坑

两者在论文和实验里都能处理长文档，但真正上线时，决定胜负的往往不是理论，而是系统复杂度。

### 核心权衡

| 维度 | Longformer | BigBird |
|---|---|---|
| 部署复杂度 | 低 | 中到高 |
| 可复现性 | 高 | 依赖 seed 与 mask 固定 |
| 调试难度 | 低 | 更高 |
| 对关键 token 选择依赖 | 高 | 中 |
| 长程依赖建模 | 中 | 更强 |
| 推理吞吐稳定性 | 更好 | 略差 |

高吞吐推理里，可以把 Longformer 看成“预先刻好的轨道”；BigBird 则像“轨道上额外搭的随机桥梁”，每次训练与部署都要确认桥梁图是否一致。

### Longformer 常见坑

1. 全局 token 选错。
如果你没把真正重要的位置设成全局，远程信息就只能靠多层慢慢传，关键条件可能到不了分类头。长文问答里，问题 token、标题 token、段落首 token 常是候选。

2. 窗口太小。
窗口过小会让局部语义还没整合好就被截断，尤其对跨句关系抽取不利。

3. 误以为“有全局 token 就够了”。
全局 token 能做汇聚，但不是所有任务都适合把所有远程依赖都压缩到少数节点中转。信息瓶颈可能出现在这里。

### BigBird 常见坑

1. 训练和推理的随机图不一致。
如果 seed 不固定，模型训练看到的是一套图，推理时又换一套，效果可能波动。

2. 随机边采样逻辑写错。
比如没有排除局部窗口或全局块，导致“随机边”其实重复覆盖已有边，白白增加复杂度却没有增加有效连通性。

3. 底层 kernel 不友好。
随机稀疏访问模式更难做高效访存，理论复杂度线性，不代表实际 wall-clock time 一定更快。

4. 额外连接数带来的算力开销。
如果随机连接带来约 20% 到 30% 的额外边数，真实吞吐会明显受影响，特别在 batch 大、序列长的场景。

### checklist

| 检查项 | Longformer | BigBird |
|---|---|---|
| 是否明确关键 token | 必须 | 建议 |
| 是否固定注意力图 | 天然固定 | 必须固定 seed |
| 是否验证 mask 可视化 | 建议 | 必须 |
| 是否做吞吐压测 | 必须 | 必须 |
| 是否评估长程依赖覆盖 | 必须 | 必须 |

真实工程例子是长文 RAG 后处理分类。假设你把检索回来的 20 段法律条文拼成一个 8k token 输入，做“是否存在违约责任”的判定：

- 如果你的规则明确知道“标题、问题、段首”最重要，Longformer 往往更稳，因为全局 token 可以手工指定。
- 如果你无法提前知道哪一段和哪一段会发生关键交互，BigBird 的随机远程连接更灵活，但你要承担更复杂的训练与部署控制。

---

## 替代方案与适用边界

Longformer 和 BigBird 不是唯一选择。它们只是“显式稀疏图”这一路线中的两种代表。

### 什么时候选 Longformer

如果任务强调最大吞吐率、稳定推理、明确关键 token，可优先选 Longformer。例如：

- 长文分类
- 结构化文档问答
- 关键位置可预定义的审查任务

一句话边界是：你能提前猜到“哪些位置最重要”，Longformer 往往就足够。

### 什么时候选 BigBird

如果任务更强调近似全注意力的长程交互，又能接受更复杂的 mask 控制，可优先考虑 BigBird。例如：

- 长文多段证据融合
- 文档中远距离实体关系建模
- 很难预定义关键 token 的场景

一句话边界是：你希望任意两句都有更高概率通过短路径发生交互，BigBird 更灵活。

### 与其他方案对比

| 方案 | 核心思想 | 优点 | 边界 |
|---|---|---|---|
| Longformer | 固定局部窗口 + 全局 token | 稳定、易部署 | 长程依赖依赖全局点设计 |
| BigBird | 局部 + 全局 + 随机连接 | 理论更强，路径更短 | 实现复杂、随机控制麻烦 |
| Reformer | LSH 注意力，按相似性分桶 | 降低复杂度 | 哈希误差影响稳定性 |
| Performer | 线性注意力核近似 | 真正线性、易扩展 | 是核近似，不是显式稀疏图 |
| 全注意力 + 截断 | 直接截短输入 | 最简单 | 丢失长文信息最严重 |

这里 Reformer 的“LSH”可以先理解成“把相似 token 哈希到同一桶里再算注意力”；Performer 的“核近似”可以先理解成“用数学变换避免显式两两配对”。

因此，替代方案不是谁绝对更强，而是谁更符合你的系统约束：

- 算子实现能力弱，优先 Longformer。
- 需要理论上更像全注意力，优先 BigBird。
- 更关心极致线性扩展，可看 Performer。
- 只做研究试验，不急于上线，可把 BigBird、Performer 一起纳入对比。

---

## 参考资料

1. Beltagy, Iz, Matthew E. Peters, and Arman Cohan. “Longformer: The Long-Document Transformer.” arXiv:2004.05150. https://arxiv.org/abs/2004.05150  
2. Zaheer, Manzil, et al. “Big Bird: Transformers for Longer Sequences.” NeurIPS 2020. https://arxiv.org/abs/2007.14062  
3. Martin Brenndoerfer, “BigBird Sparse Attention: Random Connections for Long Documents.” 对 BigBird 的随机连接、复杂度和直观解释做了清晰总结，也给出与 Longformer 的对照。https://mbrenndoerfer.com/writing/bigbird-sparse-attention-random-connections-long-documents?utm_source=openai  
4. LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. 用于评估长法律文档任务表现，常被拿来比较长文本模型。https://arxiv.org/abs/2110.00976  
5. Hugging Face Transformers 文档中关于 Longformer 与 BigBird 的模型说明，可用于核对工程接口与注意力模式实现。https://huggingface.co/docs/transformers/model_doc/longformer 以及 https://huggingface.co/docs/transformers/model_doc/big_bird  
6. BigBird 原论文中的理论部分给出的重点是：在加入全局、局部、随机边后，稀疏注意力图可以保持线性复杂度并获得更强的表达与连通性质。新手可以先抓住这一点，不必一开始就深入全部证明细节。

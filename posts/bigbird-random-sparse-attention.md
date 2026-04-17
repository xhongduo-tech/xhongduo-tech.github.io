## 核心结论

BigBird 的核心不是“把注意力做稀疏”这么简单，而是用三种边同时保留表达力：

1. 窗口注意力：每个 token 只看左右各 $w$ 个邻居。白话说，就是先保证“附近信息”能稳定交流。
2. 全局注意力：少量全局 token 与所有位置双向连接。白话说，就是放几个“中转站”。
3. 随机注意力：每个 token 再随机连到 $r$ 个远处位置。白话说，就是给长距离通信加“捷径”。

因此，每个普通 token 关注的位置数近似是：

$$
|\text{Attend}(i)| \approx 2w + g + r
$$

整层计算量变成：

$$
\text{Cost} = O\big(n(2w+g+r)\big)=O(n)
$$

只要 $w,g,r$ 不随序列长度 $n$ 增长，复杂度就是线性的。

更关键的是，BigBird 不是只换来速度。原论文证明，这种“窗口 + 全局 + 随机”的稀疏图仍保留了接近完整 Transformer 的理论表达力，包括通用近似能力；在特定构造下还保留图灵完备性。直观上看，一层 attention 就是一张图：窗口边负责本地连通，全局 token 负责星形中枢，随机边负责跨段捷径，组合后图的直径显著下降，远距离信息不必靠很多层慢慢爬过去。

一个简图可以这样理解：

```text
窗口边：   0-1-2-3-4-5-6-7-8-9
全局边：        [G] 连到所有位置
随机边：   1-----7, 2---------9, 4-----0
```

如果只有窗口边，信息传播像沿着链表走；加入全局边和随机边后，更像小世界网络，任意两点通常只需很少跳数就能互达。

---

## 问题定义与边界

BigBird 要解决的问题是：在把自注意力从 $O(n^2)$ 降到 $O(n)$ 时，怎样不把模型“削弱成只能看局部”的近似模型。

这里先给出边界。

| 方案 | 每个 token 关注数 | 单层复杂度 | 远距离通信 | 理论表达力 |
|---|---:|---:|---|---|
| 全注意力 | $n$ | $O(n^2)$ | 1 跳直达 | 最强、最直接 |
| 纯窗口注意力 | $2w+1$ | $O(nw)$ | 约 $O(n/w)$ 跳 | 明显受限 |
| Longformer | $2w+g$ | $O(n(2w+g))$ | 依赖全局 token | 工程有效，理论较弱 |
| BigBird | $2w+g+r$ | $O(n(2w+g+r))$ | 全局 + 随机捷径 | 有完整理论支撑 |

这里的“图直径”可以用一句白话解释：它表示最远两点之间最短还要走几步。直径越小，信息越容易在少数层里传到全局。

玩具例子：设 $n=64,\ w=3,\ g=2,\ r=3$。

- 每个普通 token 最多看 $2w+g+r=11$ 个位置。
- 全注意力总位置数是 $64\times 64=4096$。
- BigBird 的实际连边数大约是几百到一千量级，而不是 4096。
- 如果只有窗口，序列两端大约要 $\lceil 63/3 \rceil \approx 21$ 次“窗口跳”才能传到。
- 加入全局 token 和随机边后，最短路径通常会缩短到常数级或对数级。

所以 BigBird 的目标不是“让每个 token 少看一点”，而是“在看的位置很少时，仍保持整张图全局可达”。

需要特别说明一个准确性边界：很多二手材料会把 BigBird 直观概括成“当随机边足够多时，稀疏图像 expander graph 一样具有低直径”。这个方向是对的，但原论文里的严格结论是基于它的块稀疏构造、全局 token 和随机块的联合证明，不应机械简化成单一的“$r\cdot w\ge \Omega(n)$ 就自动成立”口号。

---

## 核心机制与推导

把一层 attention 看成有向图 $G=(V,E)$：

- 节点 $V$：序列里的 token。
- 边 $E$：若位置 $i$ 可以 attend 到位置 $j$，就有边 $(i,j)$。

### 1. 只有窗口时，图像一条“宽一点的链”

如果每个 token 只看左右 $w$ 个邻居，那么一次 attention 最多把信息推进 $w$ 个位置。两端 token 的最短路径大约是：

$$
d_{\text{local}} \approx \left\lceil \frac{n-1}{w} \right\rceil
$$

这意味着层数不够时，远距离依赖根本传不过去。

### 2. 全局 token 提供“星形中枢”

设有一个全局 token $G$，且所有 token 都能看它、它也能看所有 token，那么任意两点 $u,v$ 至少存在路径：

$$
u \to G \to v
$$

也就是 2 跳可达。白话说，全局 token 像路由器，把本来沿链条传播的信息改成“先上高速，再下高速”。

但只靠全局 token 还不够，原因有两个：

- 所有远距离信息都挤在少量中枢上，容易形成瓶颈。
- 只靠“窗口 + 全局”的确定性结构，原论文没有给出与 BigBird 同等级的通用近似证明。

### 3. 随机边把规则图改造成“小世界”

随机边的作用不是碰运气找重要词，而是从图论上降低平均路径长度。只要每个节点额外连少量远处节点，整张图就会出现很多跨区间捷径。于是大致会从：

$$
O(n/w)
$$

下降到接近：

$$
O(\log n)
$$

或者在带全局节点的情况下接近常数跳数的通信效果。

这也是 BigBird 最重要的理论点：随机边不是工程上的装饰，而是让稀疏图获得“足够强的全局混合能力”。

### 4. 为什么它还能近似全注意力

完整注意力本质上允许任意 token 直接聚合全局信息。BigBird 的证明思路可以粗化为两步：

1. 用全局 token 模拟“中心节点”。
2. 用窗口边和随机边保证普通节点与中心、不同区域之间的有效通信。

于是可以把许多“先汇聚，再变换，再广播”的全局计算，改写成稀疏图上的有限步消息传递。也就是说，BigBird 并不是逐项复制完整注意力矩阵，而是在函数表达能力上保留了足够强的近似能力。

---

## 代码实现

下面给一个可运行的玩具实现，只生成 BigBird 的 attention mask，不做真正的 softmax 计算。重点是三类边怎么拼起来，以及随机 seed 必须固定。

```python
import random

def build_bigbird_mask(n, w, global_tokens, r, seed=42):
    rng = random.Random(seed)
    global_set = set(global_tokens)
    mask = []

    for i in range(n):
        keys = set()

        # 1) 窗口注意力
        for j in range(max(0, i - w), min(n, i + w + 1)):
            keys.add(j)

        # 2) 全局注意力：所有 token 都看全局 token
        keys.update(global_set)

        # 3) 全局 token 自己看所有位置
        if i in global_set:
            keys.update(range(n))
        else:
            # 4) 随机注意力：排除已连接位置后再采样
            candidates = [
                j for j in range(n)
                if j not in keys
            ]
            sample_k = min(r, len(candidates))
            keys.update(rng.sample(candidates, sample_k))

        mask.append(sorted(keys))
    return mask

def count_edges(mask):
    return sum(len(row) for row in mask)

# 玩具例子
n, w, g, r = 16, 2, 2, 3
global_tokens = [0, 1]
mask1 = build_bigbird_mask(n, w, global_tokens, r, seed=7)
mask2 = build_bigbird_mask(n, w, global_tokens, r, seed=7)
mask3 = build_bigbird_mask(n, w, global_tokens, r, seed=8)

# 同 seed 必须一致，不同 seed 通常不同
assert mask1 == mask2
assert mask1 != mask3

# 普通 token 至少包含窗口和全局 token
for i in range(g, n):
    row = set(mask1[i])
    assert 0 in row and 1 in row
    assert i in row
    assert len(row) >= (2 * w + 1)

# 全局 token 看所有位置
for i in global_tokens:
    assert len(mask1[i]) == n

# 稀疏边数明显少于全注意力
assert count_edges(mask1) < n * n

print("total_edges =", count_edges(mask1))
print("row_5 =", mask1[5])
```

这段代码对应的逻辑就是：

```python
for i in range(n):
    window = range(i - w, i + w + 1)
    keys = window + global_tokens + random.sample(other_tokens, r)
```

其中：

- `global_tokens` 是全局中枢。
- `random.sample` 负责制造远距离捷径。
- `seed` 必须配置化，训练和推理都保持一致，否则 mask 分布会变，模型学到的注意力路径会失配。

真实工程例子：长文档问答。比如一个 QA 样本长度 4096 token，问题在开头，证据散落在多个段落中。常见做法是把问题 token 或 `[CLS]` 设为全局 token，让它持续汇聚全局信息；正文 token 使用窗口边捕获局部语义，再用随机边让“问题相关段落”和“答案段落”之间更快形成路径。这样能在长上下文下保留问答性能，而不是把文档切碎后做多段拼接。

---

## 工程权衡与常见坑

BigBird 的理论很好，但落地并不是“把注意力稀疏化”就结束了。

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 不设全局 token | QA、分类任务退化明显 | 缺少稳定的全局汇聚点 | 至少保留问题 token、`[CLS]` 或专门的 global slots |
| 随机 seed 不固定 | 训练正常，推理掉点 | 训练/推理 mask 分布不一致 | 把 seed 写入配置并持久化 |
| 层数太少 | 长距离依赖仍传不过去 | 稀疏图再好也需要足够层数传播 | 长上下文任务避免“超长输入 + 极浅网络” |
| 只保留窗口，去掉随机边 | 模型更像 Longformer，理论退化 | 图直径变大，远距离捷径消失 | 若追求 BigBird 理论性质，不要删随机边 |
| 全局 token 过多 | 复杂度回升、显存吃紧 | 全局边本质是稠密边 | 全局 token 数通常保持常数级 |
| 随机边采样含重复/落在窗口内 | 实际有效边数变少 | 实现细节错误 | 先去重，再从补集采样 |

一个真实工程坑：某长文档 QA 模型只有 6 层，并且为了实现简单，把随机边关掉，只保留窗口和问题 token 的全局连接。结果短样本上看不出问题，长样本上性能明显下降。原因不是“BigBird 理论失效”，而是实现已经不再是完整的 BigBird，而且 6 层对于最远依赖的传播仍然偏浅。很多失败案例本质上不是模型思想不行，而是把关键结构删掉了。

还要注意一个现实权衡：理论上 BigBird 是 $O(n)$，但真实速度取决于内核实现。若框架没有高效的块稀疏 kernel，实际 wall-clock 速度未必线性改善，甚至可能出现“理论省算力，工程没跑快”的情况。

---

## 替代方案与适用边界

BigBird 不是唯一的长序列方案，但它在“理论表达力 + 线性复杂度”这个组合上比较特殊。

| 方案 | 连接结构 | 复杂度 | 优势 | 局限 |
|---|---|---:|---|---|
| Longformer | 窗口 + 全局 | 线性 | 结构确定、实现较稳 | 缺少 BigBird 的随机捷径理论 |
| Reformer | LSH 哈希注意力 | 近线性 | 节省显存，适合近似相似性分组 | 依赖哈希质量，语义稳定性看任务 |
| BigBird | 窗口 + 全局 + 随机 | 线性 | 有通用近似与图灵完备相关证明 | 实现更复杂，随机 mask 需严格管理 |
| 全注意力 | 全连接 | 二次 | 最直接、最通用 | 长序列成本高 |

什么时候优先用 BigBird：

- 输入是几千到上万 token 的长文档。
- 任务既需要局部语义，也需要跨段推理。
- 你希望在稀疏注意力里保留较强的理论保证。
- 你能接受更复杂的稀疏 mask 和内核实现。

什么时候不一定选 BigBird：

- 序列不长，普通 Transformer 已经够用。
- 你所在框架对 Longformer 或 FlashAttention 支持更成熟。
- 任务对全局一致性极端敏感，且显存允许全注意力。
- 你无法保证训练与部署使用同一套随机稀疏规则。

一句话概括选择标准：如果长上下文是刚需，而你又不想把模型退化成“只能局部看”的稀疏近似，BigBird 是很强的折中；如果上下文不长，直接全注意力通常更简单。

---

## 参考资料

- BigBird 原论文：Zaheer et al., *Big Bird: Transformers for Longer Sequences*  
  作用：给出线性复杂度、通用近似能力、图灵完备性等正式论证。  
  URL: https://arxiv.org/abs/2007.14062

- AI千集《BigBird 稀疏注意力》  
  作用：中文材料，适合快速理解论文主张与核心结论。  
  URL: https://www.aiqianji.com/blog/article/4036

- Docsaid《BigBird》  
  作用：强调为什么随机、窗口、全局三者要同时存在，并从表达力角度解释贡献。  
  URL: https://docsaid.org/en/papers/transformers/bigbird/

- Michael Brenndoerfer, *BigBird: Sparse Attention with Random Connections for Long Documents*  
  作用：可视化展示稀疏图、复杂度、图直径变化，对初学者很友好。  
  URL: https://mbrenndoerfer.com/writing/bigbird-sparse-attention-random-connections-long-documents

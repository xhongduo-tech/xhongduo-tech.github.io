## 核心结论

图召回的邻域扩展，本质上是在图里做“从种子节点向外传播”，把原本只靠一跳行为能看到的候选，扩展到二跳、三跳甚至更高阶的共现邻居。这里的“共现”可以直白理解为：虽然用户没有直接点过某个物品，但很多和他相似的路径最终都会走到这个物品，于是它值得进入候选集。

在推荐系统里，图通常是用户-物品二部图，也就是用户节点只连物品节点、物品节点只连用户节点。也可以是物品同构图，意思是只保留“物品和物品之间相似”的边。无论具体图怎么建，核心目标都一样：把高阶关系压缩成一个可控、低延迟的候选扩展过程。

最常见的传播形式是 Personalized PageRank，简称 PPR，可以直白理解为“不断沿着图走一步，但随时允许回到起点”。它的标准形式是：

$$
\pi = \varepsilon e_u + (1-\varepsilon)P\pi
$$

其中，$\pi$ 是最终的节点得分分布，$e_u$ 是只在种子用户 $u$ 位置为 1 的向量，$\varepsilon$ 是重启概率，$P$ 是归一化后的转移矩阵。$\varepsilon$ 越大，传播越浅；越小，传播越深，但也越容易被热门节点带偏。

工程上，图召回不是为了直接完成最终排序，而是为了把“可能感兴趣”的候选集合先扩出来。后续通常还会接精排模型。离线复杂度大多和边数 $O(E)$ 相关，在线为了做到毫秒级响应，常见做法是预计算 embedding、预存 top-k 邻接表，或缓存随机游走片段而不是临时全图重算。

---

## 问题定义与边界

问题先定义清楚：图召回解决的是“候选集太窄”的问题，不是“最终排序绝对最优”的问题。它的职责是提高 Recall，也就是“别漏掉本来有可能感兴趣的内容”。

如果只做单跳召回，比如“用户看过什么，就找最相似的几个物品”，模型只能看到直接邻居。但很多有效信号其实来自多跳路径。例如：

- 用户 A 点过手机壳
- 很多也点手机壳的用户，后来又买了充电器
- 于是充电器虽然不是 A 的直接邻居，却可能是二跳上很强的候选

这就是邻域扩展的价值。

常见图构造方式如下：

| 图类型 | 节点 | 边含义 | 优点 | 适用边界 |
| --- | --- | --- | --- | --- |
| 用户-物品二部图 | 用户、物品 | 点击、购买、收藏等交互 | 直接承接协同过滤信号 | 推荐召回最常见 |
| 物品 kNN 图 | 物品 | embedding 相似、共现相似 | 在线传播简单，适合物品扩展 | 物品到物品召回 |
| 异构图 | 用户、物品、类目、品牌、作者等 | 多种关系边 | 能融入上下文与语义结构 | 复杂推荐、知识图谱检索 |

边界也要说明白。

第一，图召回适合“候选很多、关系稀疏但可连通”的场景，比如推荐、检索、广告。  
第二，它不适合把全部精力放在最终排序解释上，因为图传播的结果更多是候选相关性，不是最终转化概率。  
第三，图如果建得太稠密，传播会变成“热门节点扩散器”；图如果建得太稀疏，又无法形成有效二跳信号。

一个真实工程例子是：离线阶段先根据用户行为和双塔 embedding 建立用户-物品图；在线请求到来时，以当前用户为 seed，用 PPR 或近似随机游走收集前 100 个高分物品，再交给排序模型打分。假设 $\varepsilon=0.15$，那么传播会有足够的二跳、三跳能力，但不会无限跑向全图。

---

## 核心机制与推导

PPR 的关键不是“走得远”，而是“远近都有，但远处自动衰减”。这来自两个机制：

1. 重启  
重启就是每一步都有概率回到种子用户。白话说，它防止路径跑偏。

2. 归一化传播  
归一化就是出边多的节点，每条边分到的权重更小。白话说，热门节点不会把全部流量都吞掉，但它仍然可能因为连接很多而积累优势，所以工程上还要额外抑制热门偏置。

迭代公式通常写成：

$$
\pi^{(t+1)} = \varepsilon e_u + (1-\varepsilon)P\pi^{(t)}
$$

这里的含义很直接：

- $\varepsilon e_u$：每轮强行把一部分概率拉回用户自己
- $(1-\varepsilon)P\pi^{(t)}$：剩下的概率按图边扩散

### 玩具例子

考虑最小二部图：一个用户 $u$，两个物品 $i_1, i_2$。用户连接这两个物品，两个物品也都连回用户。用节点顺序 $[u, i_1, i_2]$，定义转移矩阵：

$$
P =
\begin{bmatrix}
0 & 1 & 1 \\
1/2 & 0 & 0 \\
1/2 & 0 & 0
\end{bmatrix}
$$

取种子向量：

$$
e_u = [1, 0, 0]^T
$$

令 $\varepsilon = 0.2$，初始 $\pi^{(0)} = e_u$。则第一次迭代：

$$
P\pi^{(0)} =
\begin{bmatrix}
0 \\
1/2 \\
1/2
\end{bmatrix}
$$

所以：

$$
\pi^{(1)} = 0.2
\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}
+ 0.8
\begin{bmatrix}
0 \\ 1/2 \\ 1/2
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\ 0.4 \\ 0.4
\end{bmatrix}
$$

这说明即使只走一轮，两个物品已经各拿到 0.4 的概率质量。继续迭代时，概率会从物品回流到用户，再扩散到别的物品，于是二跳、三跳共现被吸收进来。

### 为什么它能表达高阶共现

把公式展开，可以得到：

$$
\pi = \varepsilon \sum_{k=0}^{\infty}(1-\varepsilon)^k P^k e_u
$$

这里的 $P^k e_u$ 表示从种子出发走 $k$ 步后能到达的分布。也就是说，PPR 实际上把所有路径长度都加权求和了，只是路径越长，权重越小。于是：

- $k=1$ 对应一跳兴趣
- $k=2,3$ 对应共现与社区信号
- 更大的 $k$ 对应更远的结构相关性，但影响会衰减

这也是图召回比纯 itemCF 更稳的原因。itemCF 往往只看局部相似，PPR 则把多跳路径统一纳入一个概率框架。

---

## 代码实现

工程里很少在线直接解完整 PPR 线性方程，因为代价高。更常见的是两类方案：

- 离线做邻居聚合或 embedding 预计算，在线查表
- 预存随机游走片段，查询时拼接近似 PPR

先给一个可以运行的最小 Python 版本，展示 PPR 迭代。

```python
import math

def ppr_step(P, pi, seed, eps):
    n = len(pi)
    nxt = [eps * seed[i] for i in range(n)]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += P[i][j] * pi[j]
        nxt[i] += (1 - eps) * s
    return nxt

P = [
    [0.0, 1.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.0, 0.0],
]
seed = [1.0, 0.0, 0.0]
pi0 = [1.0, 0.0, 0.0]

pi1 = ppr_step(P, pi0, seed, eps=0.2)

assert all(abs(a - b) < 1e-9 for a, b in zip(pi1, [0.2, 0.4, 0.4]))
assert abs(sum(pi1) - 1.0) < 1e-9

def iterate_ppr(P, seed, eps=0.2, steps=20):
    pi = seed[:]
    for _ in range(steps):
        pi = ppr_step(P, pi, seed, eps)
    return pi

pi_final = iterate_ppr(P, seed, eps=0.2, steps=20)
assert abs(sum(pi_final) - 1.0) < 1e-9
print(pi_final)
```

如果要做在线低延迟，一种典型伪代码是“walk segment cache”，也就是缓存随机游走片段。这里的“segment”可以直白理解为：提前替很多节点走好一小段路，查询时直接拼起来，不要现场从头算。

```python
def extendWalk(seed, segments, restart_prob, max_steps):
    walk = [seed]
    cur = seed
    for _ in range(max_steps):
        if random() < restart_prob:
            cur = seed
        else:
            cur = segments[cur].next_hop()
        walk.append(cur)
    return walk

def queryTopK(seed, segments, k, restart_prob, max_steps):
    score = Counter()
    walk = extendWalk(seed, segments, restart_prob, max_steps)
    for node in walk:
        score[node] += 1
    return topk(score, k)

def updateSegments(u, w, segments):
    affected = find_related_segments(u, w)
    for seg in affected:
        seg.resample()
```

参数含义通常是：

| 参数 | 含义 | 调大后的效果 | 代价 |
| --- | --- | --- | --- |
| $\varepsilon$ | 重启概率 | 传播更浅，更贴近局部兴趣 | 远距离召回下降 |
| $R$ | 每节点缓存的 segment 数 | 估计更稳，在线 fetch 更少 | 存储增加 |
| top-k 邻接表大小 | 每个节点保留多少邻居 | 候选更丰富 | 图更稠密，噪声更多 |

### 真实工程例子

电商推荐里，一个常见链路是：

1. 离线用 30 天点击、加购、购买日志构建用户-物品二部图  
2. 对高频物品再补一个 item-item kNN 图，避免冷启动太弱  
3. 在线以用户最近活跃节点作为 seeds，做多 seed 邻域扩展  
4. 合并得到 200 到 500 个候选，交给精排模型

这样设计的原因是：用户-物品图擅长找协同关系，item-item 图擅长在用户行为少的时候补物品相似性。多图融合后，召回覆盖通常比单图更高。

---

## 工程权衡与常见坑

图召回真正难的部分不在公式，而在“怎么不让图把系统带偏”。

第一个坑是热门节点偏置。所谓热门节点，就是和几乎所有人都有边的节点，例如爆款商品、全站热门视频、超级活跃用户。因为它们连接太多路径，传播时很容易被高频访问，最后让候选变得同质化。常见抑制办法有：

- 出边归一化
- 对边权做时间衰减
- 降低热门节点的最大出边数
- 在训练或增强阶段做 edge dropout，随机丢一部分用户边，减弱度偏差

第二个坑是传播太深。深传播会让召回更广，但也更“虚”。也就是路径存在，不代表兴趣真的强。尤其在二部图里，四跳之后很容易进入泛相似区域，相关性明显下降。

第三个坑是动态图更新成本。新增边、删除边如果触发全图重算，系统会撑不住，所以工业系统通常只做局部更新，或者把图召回限制成近似方案。

下面这个表能概括主要权衡：

| 参数/策略 | 提高后会怎样 | 好处 | 风险 |
| --- | --- | --- | --- |
| $\varepsilon$ 增大 | 重启更频繁 | 更稳定，更贴近当前兴趣 | 多跳共现利用不足 |
| $R$ 增大 | 缓存 segment 更多 | 结果方差更小，在线更稳 | 存储和预处理成本增加 |
| top-k 边数增大 | 图更稠密 | 更容易连到长尾候选 | 热门噪声和误召回增加 |
| dropout ratio 增大 | 更多边被随机抑制 | 减轻度偏差 | 过大时会丢失真实强关系 |

实际调参时，一个常见经验是：

- 如果系统已经很热门化，先增加热门抑制，再考虑加深传播
- 如果系统已经很个性化但覆盖太低，先减小 $\varepsilon$ 或增加 seed 数
- 如果尾部召回不稳定，优先增加 $R$，不要只靠加深路径长度

还有一个容易被忽略的问题是“多 seed 冲突”。线上请求常常不止一个种子，比如最近点击 5 个商品。简单平均会让旧兴趣和新兴趣互相稀释。更稳的办法是对 seed 做时间衰减，最近行为权重大、旧行为权重小。

---

## 替代方案与适用边界

PPR 不是唯一方案，只是工程上很均衡的一类方案。替代方法主要有三类。

第一类是全图 GNN，例如 LightGCN 一类的邻居聚合。这里的“聚合”可以直白理解为：把邻居的表示向量做多层混合，得到新的节点表示。优点是能把多跳关系编码进 embedding，在线只需向量检索；缺点是训练和更新更重，对动态图不够灵活。

第二类是 HITS 或全局 PageRank。这类方法更强调全局中心性，可以直白理解为“谁在整张图里更权威”。它适合文档、网页、知识网络的全局排序，不太适合强个性化召回，因为它天然更偏全局热点而不是某个用户的局部兴趣。

第三类是基于规则或 ANN 的非图召回。比如双塔召回、倒排召回、规则召回。这些方法延迟低、链路简单，但对高阶共现的吸收通常不如图传播自然。

对比如下：

| 方案 | 延迟 | 更新频率 | 个性化程度 | 适用边界 |
| --- | --- | --- | --- | --- |
| PPR / walk cache | 低到中 | 可做增量更新 | 高 | 在线个性化召回 |
| 全图 GNN / LightGCN | 低（在线）高（离线训练） | 通常按批更新 | 高 | 大规模稳定场景 |
| HITS / 全局 PageRank | 低 | 可周期更新 | 低 | 全局权威排序 |
| 双塔 ANN | 很低 | 中 | 中到高 | 大规模向量召回 |

可以用一个直白类比帮助理解边界：

- HITS 或全局 PageRank，像“全图投票后，谁普遍重要”
- PPR，像“从我自己出发随机走路，看最后最容易走到谁”
- LightGCN，像“提前把邻居信息压进向量里，查询时直接做近邻搜索”

如果系统要求“强个性化 + 在线低延迟 + 图关系明显”，优先考虑 PPR 近似或图 embedding 预计算。  
如果系统要求“全局权威 + 解释稳定”，全局 PageRank/HITS 更合适。  
如果系统每天批量更新、在线必须极致快，LightGCN 或双塔 ANN 往往更实用。

---

## 参考资料

- EmergentMind, [Graph-Based Re-ranking](https://www.emergentmind.com/topics/graph-based-re-ranking?utm_source=openai)
- EmergentMind, [Graph-Based Re-Ranking Strategy](https://www.emergentmind.com/topics/graph-based-re-ranking-strategy?utm_source=openai)
- EmergentMind, [Personalized PageRank Iteration](https://www.emergentmind.com/topics/personalized-pagerank-iteration?utm_source=openai)
- Scientific Reports, [GR-MC: Multi-scale attention GNN + contrastive learning](https://www.nature.com/articles/s41598-025-17925-y?utm_source=openai)

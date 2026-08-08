---
title: 基于物品的协同过滤（ItemCF）：原理与实现
date: 2026-08-07
---

# 基于物品的协同过滤（ItemCF）：原理与实现

<div class="epigraph">
<p>基于物品的协同过滤算法，是目前业界应用最多的算法。</p>
<footer>—— 项亮《推荐系统实践》§2.4.2</footer>
</div>

<div class="article-byline">
<p>第四级 · 推荐系统 ｜ 项亮《推荐系统实践》§2.4.2 ｜ 2026-08-07</p>
</div>

## 为什么轮到 ItemCF

上一篇结尾留下了一个矛盾：UserCF 需要离线维护一张「用户 × 用户」相似度矩阵，而**用户数量庞大、且每天都在增长**。用行话说，用户的「保鲜期」短，基于用户的相似度很快过期。

于是工业界把目光从用户转向物品：**物品的数量远少于用户，物品的属性相对稳定**——一本书上架后，它的「邻居」不会三天两头变。基于物品的协同过滤（ItemCF，Item-based Collaborative Filtering）因此成为业界应用最广的邻域算法<span class="marginnote">亚马逊的「购买此商品的顾客也购买了……」（Customers who bought this item also bought…）就是 ItemCF 的招牌应用。</span>。

## 1 物以类聚：ItemCF 的两步

与 UserCF 几乎对称，ItemCF 也是两步：

- **第一步，算物品相似度**：对所有物品两两计算相似度 $w_{ij}$，得到「物品 × 物品」相似度矩阵。
- **第二步，用历史行为加权**：找出用户 $u$ 历史上喜欢的物品集合 $N(u)$，把「和 $N(u)$ 里物品相似、但 $u$ 还没看过」的物品按相似度加权求和，生成推荐列表。

**对称之美**：UserCF 是「用户像用户」，ItemCF 是「物品像物品」；公式结构完全一致，只是把「用户相似度」里的 $N(u)$（用户喜欢的物品集）换成了「物品相似度」里的 $N(i)$（喜欢物品的用户集）。

## 2 物品相似度：从「买了又买」出发

什么样的两个物品算「相似」？直觉：**经常被同一批用户购买的物品，是相似的**——「买了《机器学习实战》的人还买了《Python 数据分析》」，这两本书就是近邻。

定义 $N(i)$ 为喜欢过物品 $i$ 的用户集合。物品 $i$、$j$ 的相似度，最朴素的写法是「比例」：

$$
w_{ij} = \frac{|N(i) \cap N(j)|}{|N(i)|}
$$

分子是同时喜欢 $i$ 和 $j$ 的用户数，分母是喜欢 $i$ 的用户数。这个定义有个严重的偏差：**如果 $j$ 是热门物品，$|N(i) \cap N(j)|$ 会普遍偏大**——几乎所有物品都能和爆款搭上边，$w_{ij}$ 一律趋近 1，「相似度」失去区分度。这正是上一篇 IUF 想解决的「热门污染」的另一个马甲。

沿用与 UserCF 相同的手法，把分母换成几何平均，就得到余弦形式的物品相似度：

$$
w_{ij} = \frac{|N(i) \cap N(j)|}{\sqrt{|N(i)| \cdot |N(j)|}}
$$

**辨析｜易错点：** 这里 $N(\cdot)$ 的含义与 UserCF 正好「转置」——UserCF 里 $N(u)$ 是「用户喜欢的物品集」，ItemCF 里 $N(i)$ 是「喜欢物品的用户集」。同一个符号，站在用户这边是「往外看物品」，站在物品这边是「往里收用户」<span class="marginnote">把「用户 × 物品」行为矩阵画出来：UserCF 沿行看（用户的一行 = 他喜欢的物品），ItemCF 沿列看（物品的一列 = 喜欢它的用户）。两个算法就是同一张矩阵的两个方向。这种「同一数据两种读法」的思路，在第三级《线性代数》的转置与第三级《机器学习》的特征视角里会反复出现。</span>。

## 3 公式解析：物品相似度 w_ij

以余弦版为例，拆成四步：

- **第一步，取共现 $N(i) \cap N(j)$**：统计「同时喜欢 $i$ 和 $j$」的用户集合，其大小记作 $|N(i)\cap N(j)|$——这是两个物品「被同一批人选中」的直接证据。
- **第二步，分子是共现数**：共现越多，相似度越高。
- **第三步，分母 $\sqrt{|N(i)||N(j)|}$**：惩罚「单独来看都很热门」的物品——一个 1000 万人喜欢、一个 10 万人喜欢，即使交集 10 万，相似度也只有 $10^5/\sqrt{10^7\times 10^5}\approx 0.1$，不会因为绝对数量大而虚高。
- **第四步，量纲说明**：$w_{ij}$ 落在 $[0,1]$ 之间，$w_{ij}=1$ 当且仅当两者被完全相同的用户喜欢。

**辨析｜易错点：** $N(i)\cap N(j)=N(j)\cap N(i)$，交集天然对称，$w_{ij}=w_{ji}$——**如果你写出的相似度矩阵不对称，通常是代码里对用户/物品的遍历方向处理错了**。对称性是验证实现正确性的第一道关卡。

## 4 公式解析：推荐打分 p(u,j)

得到物品相似度矩阵后，用户 $u$ 对物品 $j$ 的推荐分是：

$$
p(u,j) = \sum_{i \in N(u) \cap S(j,K)} w_{ji} \cdot r_{ui}
$$

四步拆解：

- **第一步，读懂下标集合** $N(u) \cap S(j,K)$：$N(u)$ 是 $u$ 喜欢过的物品集，$S(j,K)$ 是与物品 $j$ 最相似的 $K$ 个物品。取交集意味着：**只看「用户喜欢过的、且与 $j$ 相似」的那些物品 $i$**。
- **第二步，权重 $w_{ji}$**：$i$ 与 $j$ 越相似，$i$ 对 $j$ 的推荐越有分量。
- **第三步，$r_{ui}$**：$u$ 对 $i$ 的真实反馈（隐式反馈为 1）。
- **第四步，求和**：累加所有「$u$ 喜欢过的近邻物品」对 $j$ 的加权贡献。

**与 UserCF 打分的对照**：UserCF 的 $p(u,i)$ 是「相似用户投票」，ItemCF 的 $p(u,j)$ 是「相似物品背书」。前者以人为中心，后者以物为中心。

## 5 跑通一个最小例子

沿用上一篇的用户：小明看过《Matrix》《Inception》，小红看过《Matrix》《Inception》《Titanic》，小刚看过《Inception》《Coco》。

先算物品相似度。《Inception》与《Matrix》同时被小明、小红喜欢，共现数为 2：

$$
w_{\text{Matrix, Inception}} = \frac{2}{\sqrt{2 \times 3}} = \frac{2}{\sqrt{6}} \approx 0.816
$$

（$|N(\text{Matrix})|=2$，$|N(\text{Inception})|=3$，交集为 $\{\text{小明},\text{小红}\}$。）

![ItemCF 物品相似度计算示意](/images/recommender-systems/itemcf-principle-and-implementation-1.svg)

现在给小明（$N=\{\text{Matrix}, \text{Inception}\}$）推荐他没看过的《Titanic》与《Coco》：

$$
p(\text{小明}, \text{Titanic}) = w_{\text{Titanic, Matrix}}\cdot 1 + w_{\text{Titanic, Inception}}\cdot 1 = \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{3}} \approx 1.284
$$

$$
p(\text{小明}, \text{Coco}) = w_{\text{Coco, Inception}}\cdot 1 = \frac{1}{\sqrt{3}} \approx 0.577
$$

《Titanic》排在《Coco》前面——因为它同时与小明喜欢的两个物品都相似，而《Coco》只与《Inception》一个相似。

## 6 代码实现

```python
from collections import defaultdict
import math

def item_similarity(user_items):
    """user_items: {user: set(items)}，返回物品-物品余弦相似度矩阵 W。"""
    item_users = defaultdict(set)                 # 物品 → 喜欢它的用户集合
    for u, items in user_items.items():
        for i in items:
            item_users[i].add(u)

    # 同一用户喜欢过的物品两两共现——ItemCF 不需要物品-用户倒排表
    C = defaultdict(lambda: defaultdict(int))
    for u, items in user_items.items():
        for i in items:
            for j in items:
                if i != j:
                    C[i][j] += 1

    W = defaultdict(dict)
    for i, related in C.items():
        for j, cij in related.items():
            W[i][j] = cij / math.sqrt(len(item_users[i]) * len(item_users[j]))
    return W

def recommend(user_items, W, u, K=10):
    """给用户 u 推荐：与 u 喜欢过的物品相似、且 u 未交互的物品。"""
    rank = defaultdict(float)
    for i in user_items[u]:
        for j, wij in W[i].items():
            if j not in user_items[u]:
                rank[j] += wij
    return sorted(rank.items(), key=lambda x: -x[1])[:K]

# 跑第 5 节的最小例子
user_items = {
    "小明": {"Matrix", "Inception"},
    "小红": {"Matrix", "Inception", "Titanic"},
    "小刚": {"Inception", "Coco"},
}
W = item_similarity(user_items)
print(recommend(user_items, W, "小明"))
# [('Titanic', 1.284), ('Coco', 0.577)]
```

注意：ItemCF 不需要物品-用户倒排表——它直接遍历每个用户的物品列表，**在同一用户的物品集合内部统计共现**。这正是「同时被一个人喜欢」的天然载体。

## 7 小结

- **ItemCF 两步**：算物品相似度 → 按用户历史物品加权推荐。
- **物品相似度** $\frac{|N(i)\cap N(j)|}{\sqrt{|N(i)||N(j)|}}$：经常被同一批用户购买的物品相似；分母几何平均抑制热门污染。
- **推荐打分** $p(u,j)=\sum_{i\in N(u)\cap S(j,K)} w_{ji}r_{ui}$：相似物品背书。
- 与 UserCF **对称**：同一张「用户 × 物品」矩阵的两个方向；物品稳定 → 相似度矩阵保质期长。
- ItemCF 也有两个隐藏问题：**跨类目相似度不可比**与**活跃用户污染**——下一篇专门处理。

在下一篇，我们将给 ItemCF 打上两个补丁：**按行归一化**让不同类目的相似度可比，**活跃用户惩罚**让「什么都买」的用户不再污染相似度——这是邻域方法家族的最后一块拼图。

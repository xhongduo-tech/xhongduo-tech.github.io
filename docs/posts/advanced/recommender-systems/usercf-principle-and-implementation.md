---
title: 基于用户的协同过滤（UserCF）：原理与实现
date: 2026-08-07
---

# 基于用户的协同过滤（UserCF）：原理与实现

<div class="epigraph">
<p>方以类聚，物以群分，吉凶生矣。</p>
<footer>—— 《周易 · 系辞上》</footer>
</div>

<div class="article-byline">
<p>第四级 · 推荐系统 ｜ 项亮《推荐系统实践》§2.4.1 ｜ 2026-08-07</p>
</div>

## 为什么从 UserCF 开始

上一讲我们学会了三种相似度的尺子：余弦相似度、皮尔逊相关系数与 Jaccard 系数<span class="marginnote">见本专题《相似度计算》一文。那把尺子将在这里第一次投入实战。</span>。但「相似度」是一个只定义、不使用的工具——谁跟谁相似？相似之后干什么？UserCF 回答的正是这两个问题。

**基于用户的协同过滤（User-CF，User-based Collaborative Filtering）**：利用「兴趣相似的用户」之间行为的互相预测，来给目标用户推荐物品。它是协同过滤（Collaborative Filtering）家族里最早成形的一支，也是理解后面一切推荐算法的地基<span class="marginnote">「协同过滤」一词最早见于 Tapestry 系统（Goldberg et al., 1992）；GroupLens 在 1994 年把它推广到新闻推荐。协同过滤的共同点是：不分析物品内容，只分析「用户 × 物品」行为矩阵。</span>。学完它，你再看 ItemCF、LFM、双塔，都能看到这棵树的影子。

## 1 人以群分：UserCF 的两步

UserCF 的思想朴素到可以用一句话概括：**和你口味相似的人喜欢的东西，你也很可能喜欢。**

它把推荐拆成两步：

**第一步，找相似用户**：把目标用户 $u$ 与其他所有用户两两比较，找出与 $u$ 最相似的 $K$ 个用户，记为 $S(u, K)$。
**第二步，聚合投票**：把这 $K$ 个用户喜欢过、而 $u$ 没有看过的物品挑出来，按「推荐程度」从高到低排序，生成推荐列表。

「人以群分」在这里第一次变成可计算的流程：**相似度是桥梁，投票是机制**。接下来我们分别把这两步量化。

## 2 用户相似度：把「口味一致」量化

设 $N(u)$ 表示用户 $u$ 有过正反馈行为的物品集合——买过、看过、点过赞都算正反馈<span class="marginnote">对隐式反馈（购买、浏览、播放）一般记为 $r=1$；对显式评分则用评分值。显式/隐式反馈的区别见本专题《数据来源》一文。</span>。

两个用户口味有多像，最自然的度量就是「共同喜欢的物品占比」。上一讲我们见过两种写法：Jaccard 系数与余弦相似度：

$$
w_{uv} = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|} \quad \text{（Jaccard）}
$$

$$
w_{uv} = \frac{|N(u) \cap N(v)|}{\sqrt{|N(u)| \cdot |N(v)|}} \quad \text{（余弦，本书采用）}
$$

$|N(u) \cap N(v)|$ 是两人共同喜欢的物品数，$|N(u)|$ 是 $u$ 喜欢的物品总数。**分母用几何平均 $\sqrt{|N(u)||N(v)|}$ 而非并集大小，是为了归一化**：两人交集相同，谁的「总兴趣面」更宽，谁的相似度就该被稀释。

**辨析｜易错点：** 很多同学把 Jaccard 的并集分母与余弦的几何平均分母混为一谈。两者差异在「用户总反馈数悬殊」时最明显：设 $N(u)=\{a,b\}$，$N(v)=\{a,b,c,d,e,f,g,h\}$，交集为 $2$，则 Jaccard $=2/8=0.25$，而余弦 $=2/\sqrt{2\times 8}=0.5$。余弦对「反馈多的用户」更宽容——这正是它成为工业默认的原因之一，但也埋下了「活跃用户」的问题，我们下一篇专门处理。

## 3 高效计算：物品-用户倒排表

用户相似度定义很干净，但实现有个陷阱：**用户两两组合是 $O(n^2)$ 的**，一万用户就是近五千万对，直接枚举不可行。

仔细观察 $w_{uv}$：只有「共同喜欢过至少一个物品」的用户对才有非零交集。于是我们只需要处理那些「被同一个物品连接」的用户对。做法是**倒排表（inverted index）**：

1. 建立「物品 → 用户集合」的倒排表；
2. 对每个物品，把它连接的所有用户两两计数 $C(u,v) \leftarrow C(u,v)+1$；
3. 最后用 $C(u,v)$ 作为交集大小，代入公式求相似度。

![UserCF 用户相似度计算示意](/images/recommender-systems/usercf-principle-and-implementation-1.svg)

这样，复杂度从「所有用户对」降为「所有共现用户对」——在实践中，共现对的规模远小于全量用户对<span class="marginnote">倒排表是信息检索与数据库的经典结构，第三级《数据结构》与《数据库》会系统讨论它的变体。</span>。

## 4 公式解析：推荐打分 p(u,i)

找到相似用户之后，怎么给「$u$ 没看过的物品 $i$」打分？UserCF 的推荐分数是：

$$
p(u, i) = \sum_{v \in S(u, K) \cap N(i)} w_{uv} \cdot r_{vi}
$$

分四步拆解：

- **第一步，读懂下标集合** $S(u,K) \cap N(i)$：$S(u,K)$ 是 $u$ 最相似的 $K$ 个邻居，$N(i)$ 是喜欢过物品 $i$ 的用户集合。两者取交集，意思是**只让「既是 $u$ 的邻居、又喜欢过 $i$」的用户来投票**——其他人对 $i$ 没有发言权。
- **第二步，权重 $w_{uv}$**：相似度越高的邻居，票越重。这保证了「品味接近的人」比「点头之交」更能影响推荐。
- **第三步，$r_{vi}$**：$v$ 对 $i$ 的真实反馈，隐式反馈下 $r_{vi}=1$，显式评分下取评分值。
- **第四步，求和**：把邻居们的加权票累加，得到 $u$ 对 $i$ 的兴趣强度。

**注意 $p(u,i)$ 的不对称性**：相似度矩阵满足 $w_{uv}=w_{vu}$，但打分 $p(u,i)$ 是「以 $u$ 为中心、只统计 $u$ 的邻居」，因此对不同用户并不对称——推荐本身就是高度个性化的。

## 5 跑通一个最小例子

用一张「用户 × 物品」矩阵把上面公式走一遍：

| 用户 | 看过《Matrix》 | 看过《Inception》 | 看过《Titanic》 | 看过《Coco》 |
| --- | --- | --- | --- | --- |
| 小明 | ✓ | ✓ |  |  |
| 小红 | ✓ | ✓ | ✓ |  |
| 小刚 |  | ✓ |  | ✓ |

设 $N(\text{小明})=\{\text{Matrix}, \text{Inception}\}$，$N(\text{小红})=\{\text{Matrix}, \text{Inception}, \text{Titanic}\}$，两人相似度：

$$
w_{\text{小明,小红}} = \frac{2}{\sqrt{2 \times 3}} = \frac{2}{\sqrt{6}} \approx 0.816
$$

对《Coco》：只有小刚看过，而小刚与小明交集仅 $\{\text{Inception}\}$，$w_{\text{小明,小刚}} = 1/\sqrt{2\times2}=0.5$。若取 $K=2$，小明的推荐列表是《Titanic》（来自小红，$0.816$ 分）与《Coco》（来自小刚，$0.5$ 分）——**《Titanic》排第一**，因为投它的小红与小明更相似。

## 6 代码实现：十分钟的 UserCF

把上面的逻辑写成 Python，核心不到 40 行：

```python
from collections import defaultdict
import math

def user_similarity(user_items):
    """user_items: {user: set(items)}。倒排表 + 共现计数算用户余弦相似度。"""
    item_users = defaultdict(set)                 # 倒排表：物品 → 用户集合
    for u, items in user_items.items():
        for i in items:
            item_users[i].add(u)

    C = defaultdict(lambda: defaultdict(int))
    for i, users in item_users.items():
        for u in users:
            for v in users:
                if u != v:
                    C[u][v] += 1                  # 共同喜欢的物品数

    W = defaultdict(dict)
    for u, related in C.items():
        for v, cuv in related.items():
            W[u][v] = cuv / math.sqrt(len(user_items[u]) * len(user_items[v]))
    return W

def recommend(user_items, W, u, K=2, N=10):
    """给用户 u 推荐：取 K 个最相似用户，加权聚合他们喜欢而 u 没看过的物品。"""
    rank = defaultdict(float)
    for v, wuv in sorted(W[u].items(), key=lambda x: -x[1])[:K]:
        for i in user_items[v]:
            if i not in user_items[u]:            # 跳过已交互物品
                rank[i] += wuv
    return sorted(rank.items(), key=lambda x: -x[1])[:N]

# 跑第 5 节的最小例子
user_items = {
    "小明": {"Matrix", "Inception"},
    "小红": {"Matrix", "Inception", "Titanic"},
    "小刚": {"Inception", "Coco"},
}
W = user_similarity(user_items)
print(recommend(user_items, W, "小明", K=2))
# [('Titanic', 0.816), ('Coco', 0.5)]
```

两个实现要点：**倒排表把 $O(n^2)$ 压成了共现对规模**；**推荐时跳过用户已交互的物品**，保证推荐的新颖性。

## 7 小结

- **UserCF 两步**：找 $K$ 个最相似用户 → 聚合他们的行为给物品打分。
- **用户相似度**用余弦 $\frac{|N(u)\cap N(v)|}{\sqrt{|N(u)||N(v)|}}$；**倒排表**是高效计算的核心手段。
- **推荐打分** $p(u,i)=\sum_{v\in S(u,K)\cap N(i)} w_{uv}r_{vi}$：相似度是权重，$r_{vi}$ 是反馈。
- 相似度对称、打分不对称；**活跃用户会制造虚假相似**，是下一讲的引子。

在下一篇，我们将直面 UserCF 最著名的病灶：**热门物品让所有人「变像」**，并给出项亮书中引用的经典解药——IUF（对热门物品的惩罚）。

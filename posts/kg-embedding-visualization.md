## 核心结论

知识图谱嵌入降维的目标，不是“让模型更准”，而是把原本几十到几百维的实体向量、关系向量压到人能直接观察的 2 维或 3 维空间，用来检查模型到底学到了什么结构。这里的“嵌入”是把实体或关系编码成向量，白话讲，就是把“节点的语义位置”写进一个数字坐标。

对知识图谱可视化，三种方法的角色非常清楚：

| 方法 | 保留的主要结构 | 优点 | 典型用途 | 主要限制 |
| --- | --- | --- | --- | --- |
| PCA | 全局方差结构 | 快、稳定、可重复 | 看整体分布、先降噪 | 只能表达线性结构 |
| t-SNE | 局部邻居结构 | 聚类视觉效果强 | 看实体是否局部成簇 | 全局距离不可靠，参数敏感 |
| UMAP | 局部邻居 + 部分全局形状 | 更快，通常更稳定 | 大规模嵌入可视化、持续投影 | 仍依赖参数和预处理 |

如果只记一个工程结论，就是：`PCA 先降到 30~50 维，再用 t-SNE 或 UMAP 投到 2D`。PCA 负责去掉高维噪声，t-SNE 负责检查局部语义簇，UMAP 负责在局部结构之外尽量保留一点整体形状。

真实工程里，这种可视化经常用于回答三个问题：

1. 同类实体是否聚在一起。
2. 不同关系模式是否形成可分离结构。
3. 哪些实体被模型学成了“离群点”，也就是看起来明显不合群的点。

在 SKG-4、SKG-5 一类知识图谱嵌入实验里，研究者就用 t-SNE 和 UMAP 观察嵌入投影。结果通常表现为：语义相近的实体形成紧凑簇，不同模板或扩展版本之间仍能保持可分离结构。这类图不能直接证明推理正确，但能直观暴露模型是否学到了像样的语义边界。

---

## 问题定义与边界

问题可以写成一句话：给定知识图谱中每个实体的高维向量 $x_i \in \mathbb{R}^d$，如何把它映射成二维点 $y_i \in \mathbb{R}^2$，让人能观察概念分布、聚类关系和潜在错误。

这里有两个边界必须先讲清楚。

第一，降维可视化主要用于解释和诊断，不直接参与训练。也就是说，模型训练时仍在高维空间优化，2D 图只是后验分析工具。简化数据流可以写成：

`知识图谱 -> 嵌入模型 -> 高维向量 -> PCA/t-SNE/UMAP -> 2D 可视化`

第二，降维图展示的是“投影后的几何关系”，不是原图谱逻辑本身。一个实体在图上离另一个实体更近，只能说明在当前投影里更相似，不能直接推出二者存在某条具体关系。

一个新手能立刻理解的玩具场景是：把“编程语言”“数据库”“框架”三类实体的嵌入投影到平面，用颜色区分类别。如果“Python”“Java”“Go”靠得很近，而“MySQL”“PostgreSQL”在另一团，说明模型至少把“类型相近”的实体放到了相似区域。如果某个数据库实体跑到了语言簇里，那通常意味着训练数据、负采样或关系约束出了问题。

真实工程里的对应场景更直接。比如你训练了一个医疗知识图谱嵌入模型，里面有“疾病”“药物”“症状”“检查项”四类实体。把中心节点和类型节点投影到 2D 后，颜色按类别标注。如果某些药物点大量混进疾病簇，说明模型可能把“治疗对象相关性”错误学成了“同类语义相似性”。这类问题只看 MRR、Hits@K 往往看不出来，但在投影图上会非常明显。

因此，降维的边界不是“做预测”，而是“暴露结构”。它最适合做三类工作：

1. 探索式分析：模型有没有形成预期聚类。
2. 误差排查：离群点、混簇点、异常桥接点在哪里。
3. 版本对比：同一数据集上，TransE、RotatE、ComplEx 的嵌入结构差异是什么。

---

## 核心机制与推导

先看 PCA。PCA 的全称是主成分分析，白话讲，就是在所有方向里找“数据变化最大”的方向，把数据沿这些方向展开。它是线性方法，本质上是在求协方差矩阵的特征向量。若样本矩阵中心化后为 $X$，那么 PCA 要找投影方向 $W$，使投影后的方差最大：

$$
W = \arg\max_{W^\top W = I} \mathrm{Tr}(W^\top X^\top X W)
$$

PCA 的优点是确定性强，重复运行结果一致，所以非常适合做第一步预降维。

再看 t-SNE。t-SNE 的核心思想是：高维里谁和谁是近邻，低维里尽量也保持近邻。这里的“近邻”不是直接保留原距离，而是先把距离变成概率。

对高维点 $x_i$，t-SNE 定义条件相似度：

$$
p_{j|i} = \frac{\exp\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \ne i} \exp\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)}
$$

白话讲，离 $x_i$ 越近的点，被当成“邻居”的概率越大。然后把它对称化得到 $p_{ij}$。在低维空间里，t-SNE 用 Student's t 分布定义相似度：

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \ne l}(1 + \|y_k - y_l\|^2)^{-1}}
$$

最后最小化两个分布之间的 KL 散度：

$$
\mathrm{KL}(P \| Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

这意味着：高维里本来相近的点，如果低维里被拉远，会被重罚；但高维里本来不近的点，在低维里偶然靠近，惩罚相对小。所以 t-SNE 特别擅长把局部簇“拉紧”。

UMAP 的机制和 t-SNE 相似，但建模对象更像“邻接图”。它先在高维空间构造局部 fuzzy graph。这里的 fuzzy graph 可以理解成“边存在的概率图”，不是非黑即白连边，而是每条边有一个强弱概率。单向边权通常写成：

$$
p_{j|i} = \exp\left(-\frac{d(x_i,x_j)-\rho_i}{\sigma_i}\right)
$$

再做对称化：

$$
p_{ij} = p_{j|i} + p_{i|j} - p_{j|i}p_{i|j}
$$

低维空间也构造一套边概率，然后最小化交叉熵，使高维邻接结构和低维邻接结构尽量一致。和 t-SNE 相比，UMAP 往往更快，也更容易保留一些全局轮廓。

可以用一个三点玩具例子理解。

设三个实体向量分别是：

- $A=(1,0,0,0)$
- $B=(0.9,0.1,0,0)$
- $C=(0,0,1,0)$

显然，$A$ 和 $B$ 在高维里更接近，$C$ 离它们更远。无论用 t-SNE 还是 UMAP，优化结果通常都会把 $A,B$ 放在相邻区域，把 $C$ 放得更远。这里重要的不是二维坐标的绝对值，而是“谁和谁保持邻接”。

对知识图谱嵌入，这个机制为什么有用？因为很多模型训练出来的语义结构，本来就体现在近邻关系上。比如在 TransE 这类模型里，同一类型实体、共享关系上下文的实体，经常会被推到相近区域。降维方法只是在尽量不破坏这些局部关系的前提下，把高维结构摊平给人看。

---

## 代码实现

下面给出一个可运行的最小 Python 例子。它不依赖外部绘图库，只演示“标准化 + PCA”的核心流程，并用一个玩具知识图谱嵌入验证同类点在前两主成分上更接近。`assert` 用来做最基本的正确性检查。

```python
import numpy as np

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std

def pca_reduce(X, n_components=2):
    X = standardize(X)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order[:n_components]]
    return X @ eigvecs

# 玩具知识图谱嵌入：
# 前三条更像“编程语言”类，后三条更像“数据库”类
X = np.array([
    [1.0, 1.2, 0.9, 0.0],
    [1.1, 1.0, 1.0, 0.1],
    [0.9, 1.1, 1.2, 0.0],
    [-1.0, -1.1, -0.9, 0.2],
    [-1.2, -0.8, -1.0, 0.1],
    [-0.9, -1.0, -1.1, 0.0],
], dtype=float)

Y = pca_reduce(X, n_components=2)

def dist(a, b):
    return np.linalg.norm(a - b)

# 同类之间应比跨类之间更近
same_class = dist(Y[0], Y[1]) + dist(Y[1], Y[2])
cross_class = dist(Y[0], Y[3]) + dist(Y[1], Y[4])

assert Y.shape == (6, 2)
assert same_class < cross_class

print("2D projection:")
print(np.round(Y, 3))
```

如果在真实工程里使用 `scikit-learn` 和 `umap-learn`，常见流程如下：

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None

def reduce_embeddings(X, labels):
    X_std = StandardScaler().fit_transform(X)

    # 先做 PCA 去噪，减少高维随机噪声
    X_pca50 = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X_std)

    # PCA 直接投 2D，查看整体形状
    X_pca2 = PCA(n_components=2, random_state=42).fit_transform(X_std)

    # 多组 t-SNE，比对参数稳定性
    tsne_5 = TSNE(n_components=2, perplexity=5, random_state=42, init="pca").fit_transform(X_pca50)
    tsne_30 = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(X_pca50)

    result = {
        "pca2": X_pca2,
        "tsne_5": tsne_5,
        "tsne_30": tsne_30,
    }

    if umap is not None:
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        ).fit_transform(X_pca50)
        result["umap"] = umap_2d

    # 最基本的运行检查
    assert result["pca2"].shape[1] == 2
    assert result["tsne_5"].shape[1] == 2
    return result
```

这个管线里有两个工程点最重要。

第一，先标准化。因为不同维度的数值尺度可能不一致，如果直接降维，大尺度维度会主导距离计算。

第二，多组参数同时跑。尤其是 t-SNE，至少应同时比较不同 `perplexity` 和不同随机种子。因为你看到的“簇”，有可能是参数制造出来的，不一定是数据真的有那么清楚的结构。

一个真实工程例子是：你有 20 万个知识图谱实体，嵌入维度 256，由 RotatE 训练得到。此时常见做法不是一次性把全量点都送进 t-SNE，因为太慢，而且结果不稳定。更稳妥的方案是：

1. 先按实体类型分层抽样。
2. 标准化后用 PCA 降到 50 维。
3. 跑 UMAP 看全局轮廓。
4. 对可疑类型子集再单独跑 t-SNE 看局部细节。
5. 把类别标签、关系频次、训练误差一起叠加到图上解释。

这样做，图不只是“好看”，而是能回到训练问题本身。

---

## 工程权衡与常见坑

最常见的误判，是把降维图当成原始空间的真实地图。实际上，2D 图只能看趋势，不能按像素级距离解释语义强弱。

下面这个表格可以直接当检查单使用：

| 坑 | 表现 | 验证方法 | 缓解方法 |
| --- | --- | --- | --- |
| t-SNE 伪簇 | 同一数据换参数后簇数量大变 | 多组 `perplexity`、多随机种子重复跑 | 只相信稳定重复出现的簇 |
| 全局距离误读 | 两个簇在图上远，不代表原空间就远 | 对照原始高维距离或余弦相似度 | 把 t-SNE 用于“局部邻居”，不用来解释全局位置 |
| 高维噪声污染 | 图像碎裂、边界发毛 | 先看 PCA 方差解释比例 | 先 PCA 到 30~50 维 |
| 类别不平衡 | 大类挤压小类，小类看起来像离群点 | 分层抽样、按类型单独作图 | 同时做全量图和子集图 |
| 过度解释离群点 | 个别点偏离就被认为是脏数据 | 回查原始三元组和实体频次 | 结合训练样本、度数和标签一起判断 |

t-SNE 的参数敏感性尤其需要强调。`perplexity` 可以粗略理解为“每个点期望参考多少邻居”。白话讲，它控制你看局部结构时的视野大小。视野太小，会把噪声也看成簇；视野太大，又可能把本来分开的局部结构揉平。工程上经常出现这样的情况：同一批嵌入，在 `perplexity=5` 时像 12 个簇，在 `perplexity=30` 时像 6 个簇，在 `perplexity=100` 时只剩 3 个大团。这个变化本身就说明，图中“簇数”并不是天然真相。

UMAP 往往更稳定，但也不是免调参。`n_neighbors` 控制局部邻域大小，`min_dist` 控制低维空间里点能靠多近。如果你想看类型内的细粒度团块，`n_neighbors` 不能太大；如果你想看整体板块分布，就不应把 `min_dist` 压得过低，否则所有点会被挤成紧块，失去层次。

还有一个很现实的坑：可视化只做一次。一次图是不够的，必须形成最小验证 protocol，也就是固定一套验证流程。一个可执行的版本是：

1. PCA 直接画 2D。
2. PCA50 -> t-SNE，至少跑 3 个 `perplexity`。
3. PCA50 -> UMAP，至少跑 2 组 `n_neighbors`。
4. 对每个结果统计类别纯度、近邻一致率，必要时加 ARI 这类聚类一致性指标。
5. 只有在多次结果都支持同一结构时，才把结论写进报告。

---

## 替代方案与适用边界

如果目标是快速判断“这批嵌入有没有明显可分结构”，PCA 仍然是最稳的基线。它快、可重复、解释简单，而且不会因为随机初始化导致图完全变样。缺点也很明确：只能表达线性结构，很多非线性语义簇会被压扁。

如果目标是看局部语义是否成团，t-SNE 仍然有价值，尤其适合小到中等规模、强调近邻质量的场景。但它不适合做服务化持续投影，因为新来一个点，通常要重新算整张图。

如果目标是大规模、非线性强、还希望未来新数据也能投到已有空间，UMAP 更合适。它通常比 t-SNE 更快，并且可以通过 `transform()` 把新样本投到已有低维空间，这在知识图谱在线分析系统里很实用。

下面给一个决策表：

| 问题场景 | 数据特性 | 推荐方案 |
| --- | --- | --- |
| 想快速看整体分布是否分层 | 样本中等，先做粗判断 | PCA |
| 想看局部近邻是否成语义簇 | 中小规模，重视局部结构 | PCA + t-SNE |
| 想兼顾局部结构和部分全局形状 | 中大规模，类别较多 | PCA + UMAP |
| 需要持续接收新实体并复用已有投影 | 服务化、增量分析 | UMAP 或参数化 UMAP |
| 嵌入高度非线性，想端到端学习投影 | 数据量大、允许训练附加模型 | Autoencoder / Parametric UMAP |

一个跨领域的真实例子是：对 10,000 个 2,048 维图像特征做可视化，常见流程也是先 PCA，再 UMAP。原因和知识图谱类似：原始高维空间噪声多，先用 PCA 抽掉冗余维度，再让 UMAP 在相对干净的空间里建邻接图，通常能得到比直接 t-SNE 更稳定、速度更好的结果。

所以“哪种方法最好”这个问题本身不成立。正确问法是：你要观察的是全局分布、局部邻居，还是未来的增量投影能力。目标不同，工具就不同。

---

## 参考资料

- ScienceNewsToday, *Dimensionality Reduction: PCA, t-SNE, UMAP Explained*  
  https://www.sciencenewstoday.org/dimensionality-reduction-pca-t-sne-umap-explained?utm_source=openai

- aplab, *Dimensionality Reduction: PCA, t-SNE, UMAP*  
  https://www.aplab.academy/en/courses/ml-advanced/lessons/dimensionality-reduction?utm_source=openai

- Sage Journals, *Experiments in Graph Structure and Knowledge Graph Embeddings*  
  https://journals.sagepub.com/doi/10.1177/29498732261420038?utm_source=openai

- MCP Analytics, *t-SNE vs PCA vs UMAP: Which Reveals True Clusters?*  
  https://mcpanalytics.ai/articles/t-sne-vs-pca-vs-umap-which-reveals-true-clusters?utm_source=openai

- Let’s Data Science, *UMAP Explained*  
  https://letsdatascience.com/blog/umap-explained-the-faster-smarter-alternative-to-t-sne?utm_source=openai

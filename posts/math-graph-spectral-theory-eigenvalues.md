## 核心结论

谱图理论用图 Laplacian 的特征值和特征向量刻画图结构，把“连通性、切分、平滑性”转成线性代数问题。

对无向带权图 $G=(V,E,W)$，最常用的未归一化 Laplacian 定义为：

$$
L=D-W
$$

其中 $W$ 是权重矩阵，$D$ 是度矩阵，$D_{ii}=\sum_j w_{ij}$。度矩阵就是把每个节点连出去的总边权放在对角线上的矩阵。

谱图理论里最重要的两个信号是：

$$
\lambda_1=0
$$

以及第二小特征值：

$$
\lambda_2=\min_{x\perp \mathbf{1},x\neq 0}\frac{x^\top Lx}{x^\top x}
$$

$\lambda_1=0$ 的重数等于图的连通分量数。连通分量是指图中彼此能走到的一组节点；如果一张图断成三块，那么零特征值就会出现三次。

$\lambda_2$ 常叫 algebraic connectivity，即代数连通度。它越小，通常说明图越容易被切开。新手版理解是：如果一张图里有两团点只靠一条很弱的边连接，那么这条边就是结构边界。谱分析会在 $\lambda_2$ 上体现出这个弱连接，数值越小，说明越容易沿这个位置切开。

玩具例子：四个点排成一条链，边权分别是 $1,0.1,1$。中间边权只有 $0.1$，所以图天然分成左边两个点和右边两个点。Laplacian 的第二特征向量通常会在这条弱边两侧取相反符号，从而给出 $\{1,2\}$ 和 $\{3,4\}$ 的切分。

---

## 问题定义与边界

本文只讨论无向带权图。无向表示 $w_{ij}=w_{ji}$，带权表示边有强弱，例如相似度、距离核函数、通信频率或共同点击次数。本文不讨论有向图的谱，因为有向图的 Laplacian 定义不唯一，特征值也可能变成复数，需要额外工具。

需要先区分三类矩阵，否则后面的结论会混用。

| 矩阵 | 定义 | 含义 | 适用场景 | 主要风险 |
|---|---|---|---|---|
| 邻接矩阵 $A$ 或 $W$ | $w_{ij}$ 表示边权 | 直接描述谁和谁相连 | 路径、局部连接、消息传递 | 谱结论不等同于 Laplacian |
| 未归一化 Laplacian $L$ | $L=D-W$ | 衡量图信号在边上的变化 | 度分布较均衡、中小图分析 | 容易受高度节点影响 |
| 对称归一化 Laplacian $L_{\mathrm{sym}}$ | $I-D^{-1/2}WD^{-1/2}$ | 把节点度数影响归一化 | 度差异大、谱聚类 | 对零度节点要单独处理 |

归一化 Laplacian 的定义是：

$$
L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}
$$

这里的归一化不是改变图的边，而是改变度量方式。节点度数差异很大时，直接看 $L$ 可能会被高度节点影响。真实工程例子是社交网络：少数超级节点连接很多人，如果直接使用未归一化 Laplacian，算法可能把“连接数很多”误判为“结构上更紧密”。这时更适合用 $L_{\mathrm{sym}}$ 或随机游走 Laplacian。

---

## 核心机制与推导

谱图理论的核心是二次型。二次型就是形如 $x^\top Lx$ 的标量表达式，它把一个向量和矩阵结合，输出一个数。对图 Laplacian，有：

$$
x^\top Lx=\sum_{(i,j)\in E}w_{ij}(x_i-x_j)^2
$$

$x$ 可以理解为图信号。图信号就是给每个节点分配一个数，例如网页重要性、用户活跃度、像素亮度或传感器温度。上式说明：$x^\top Lx$ 衡量的是相连节点之间数值变化的总量。

如果一条边权重大，那么 $(x_i-x_j)^2$ 会被放大，低能量向量就倾向于让这两个节点取相近值。如果一条边权很小，那么两端允许出现更大的差异。因此，小特征值对应的特征向量更平滑，也就是更倾向于在强连接内部保持相近值，在弱连接处发生变化。

把每个节点上的数值看成温度，是一个玩具例子。如果相连节点温度差很小，那么整张图上的温度图案就是平滑的。当图中存在一条很弱的桥边时，最容易出现“左边一片偏低、右边一片偏高”的模式。这个模式通常由第二小特征向量给出。

Laplacian 可以做谱分解：

$$
L=U\Lambda U^\top
$$

其中 $U$ 的列是特征向量，$\Lambda$ 的对角线是特征值。把图信号 $x$ 投影到这些特征向量上，得到图频率坐标：

$$
\hat x=U^\top x
$$

| 模式 | 对应特征值 | 图信号表现 | 结构含义 |
|---|---:|---|---|
| 低频模式 | 小特征值 | 相连节点取值接近 | 社区、连通块、全局趋势 |
| 高频模式 | 大特征值 | 相连节点取值剧烈变化 | 噪声、边界、局部异常 |
| 第二特征向量 | $\lambda_2$ | 常在弱连接两侧变号 | 二分切分线索 |

这也是谱聚类的基础：先用前几个小特征值对应的特征向量把节点嵌入到低维空间，再在这个空间里做普通聚类。

---

## 代码实现

实现谱图分析通常分三步：构图，计算 Laplacian，求少量最小特征值及其特征向量。工程里一般不会做全量特征分解，因为复杂度太高；通常用 `eigsh` 这类稀疏迭代方法求前几个特征对。

下面代码输入边列表，构造稀疏权重矩阵，计算未归一化 Laplacian，并用第二小特征向量做二分切分。

```python
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

def spectral_bipartition(n, edges, k=3):
    rows, cols, data = [], [], []

    for i, j, w in edges:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = diags(deg)
    L = D - W

    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    fiedler_vec = vecs[:, 1]
    labels = (fiedler_vec > 0).astype(int)
    return vals, fiedler_vec, labels

# 玩具图：0-1 强连接，1-2 弱连接，2-3 强连接
edges = [
    (0, 1, 1.0),
    (1, 2, 0.1),
    (2, 3, 1.0),
]

vals, fiedler_vec, labels = spectral_bipartition(4, edges)

assert abs(vals[0]) < 1e-8
assert vals[1] < 0.2
assert labels[0] == labels[1]
assert labels[2] == labels[3]
assert labels[1] != labels[2]

print("eigenvalues:", vals)
print("fiedler vector:", fiedler_vec)
print("labels:", labels)
```

这个例子中，第二小特征值会比较小，因为中间桥边权重只有 $0.1$。第二特征向量的正负号会把左侧两个点和右侧两个点分开。

真实工程例子是图像分割。可以把像素或超像素作为节点，把颜色相似、空间距离近的像素连边，然后计算 normalized cut。Shi 和 Malik 的方法把图像分割写成广义特征值问题，优点是能抓住全局边界，而不是只看局部颜色差。

---

## 工程权衡与常见坑

$\lambda_2$ 不是绝对连通强度。它和节点度数、边权尺度、图规模、是否归一化都有关系。同一张图的边权全部乘以 $10$，未归一化 Laplacian 的特征值也会按比例变化。因此不能只拿一个裸数值说“这张图很连通”。

另一个关键点是：特征向量的符号没有物理意义。如果 $v$ 是特征向量，那么 $-v$ 也是特征向量。真正有意义的是它张成的方向、节点之间的相对大小，以及多个特征向量构成的子空间。

聚类时还要看 eigengap。eigengap 是相邻特征值之间的间隙：

$$
\mathrm{eigengap}=\lambda_{k+1}-\lambda_k
$$

如果间隙很小，说明第 $k$ 个结构和第 $k+1$ 个结构区分不稳定，强行指定 $k$ 类可能不可靠。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 混淆 $A$ 和 $L$ | 把邻接矩阵谱结论误套到 Laplacian | 先明确分析对象 |
| 忽略归一化 | 高度节点主导结果 | 度差异大时使用 $L_{\mathrm{sym}}$ |
| 把特征向量正负号当绝对结论 | 不同运行结果看似相反 | 看相对符号和子空间 |
| eigengap 太小还强行聚类 | 聚类边界不稳定 | 检查 $\lambda_{k+1}-\lambda_k$ |
| 大图做全量分解 | 内存和时间不可接受 | 用 `eigsh`、Lanczos、LOBPCG |
| 零度节点未处理 | 归一化时除零 | 删除、合并或单独建模 |

真实工程中，推荐先检查图的基本统计：节点数、边数、连通分量数、度分布、边权范围。很多谱分析问题不是特征分解错了，而是图构造本身把噪声、孤立点或异常权重放大了。

---

## 替代方案与适用边界

谱方法适合全局结构清晰、希望得到软切分或低维嵌入的场景。软切分是指不只是给出硬标签，还能通过特征向量数值看出节点靠近哪个结构边界。它不一定适合局部噪声很大、图结构极不规则、或需要实时更新的任务。

| 方法 | 适用场景 | 优点 | 边界 |
|---|---|---|---|
| 谱聚类 | 全局结构清晰，中小规模图 | 理论解释强，能处理非凸结构 | 特征分解成本高 |
| Louvain / 社区发现 | 大规模社区结构发现 | 速度快，常用于网络分析 | 优化目标偏模块度 |
| 图神经网络 | 有节点特征和监督信号 | 表达能力强，适合预测任务 | 解释性弱，依赖训练数据 |
| 邻接矩阵启发式切分 | 快速近似、规则简单 | 实现简单，成本低 | 稳定性和理论保证弱 |
| 标签传播 | 少量标注扩散 | 在线性较好，容易实现 | 对图噪声敏感 |

图像分割里，谱方法能找到更符合全局边界的切分；但如果只是想快速做近似分类，图神经网络、标签传播或局部聚类可能更省算力。也就是说，谱方法不是最通用的方案，但常常是最有理论解释力的方案。

在图神经网络中，Laplacian 仍然很核心。早期图卷积可以从谱滤波角度推导，把图信号分解到 Laplacian 特征向量空间，再对不同频率做变换。后来的工程实现通常不会显式求完整特征分解，而是使用局部消息传递近似谱滤波。

---

## 参考资料

1. [Fiedler, 1973, Algebraic connectivity of graphs](https://eudml.org/doc/12723)  
用于理解 $\lambda_2$ 为什么能刻画图的代数连通性。

2. [von Luxburg, 2007, A Tutorial on Spectral Clustering](https://is.mpg.de/publications/4488)  
系统介绍谱聚类、归一化 Laplacian 和相关理论。

3. [Shi & Malik, 2000, Normalized Cuts and Image Segmentation](https://www.ri.cmu.edu/publications/normalized-cuts-and-image-segmentation/)  
展示 normalized cut 如何用于真实图像分割问题。

4. [Berkeley CS 252 Lecture Note, Spectral Graph Theory](https://people.eecs.berkeley.edu/~venkatg/teaching/15252-sp20/notes/Spectral-graph-theory.pdf)  
适合补充二次型、Rayleigh 商和图切分之间的关系。

5. [Kipf & Welling, 2016, Semi-Supervised Classification with Graph Convolutional Networks](https://doi.org/10.48550/arXiv.1609.02907)  
用于理解谱图理论和图神经网络之间的联系。

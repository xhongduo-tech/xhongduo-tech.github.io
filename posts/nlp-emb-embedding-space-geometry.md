## 核心结论

预训练模型输出的嵌入空间通常不是“各个方向都差不多均匀”的，而是明显**各向异性**。各向异性可以先白话理解为：向量更喜欢扎堆在少数几个方向上，而不是均匀铺开。结果就是，很多本来语义差异很大的句子，余弦相似度仍然会很高，常见现象是均值落在 0.6 到 0.9 之间，语义检索和 STS 这类依赖余弦排序的任务就会失去判别力。

一个直观的玩具例子是：如果所有句向量都挤在同一个锥体里，那么任意两句话看起来都“方向差不多”。这时余弦相似度已经不再主要反映语义，而是在反映“它们都落在那个锥体里”。白化可以把这种狭窄分布重新拉回接近球形，让不同方向重新有区分度。

对白化后的单位向量，余弦和欧氏距离可以直接联系起来：
$$
\cos(x,y)=\langle x,y\rangle = 1-\frac{1}{2}\|x-y\|_2^2
$$
这里假设 $x,y$ 都已经做了 $L_2$ 归一化。白话说，角度越小，欧氏距离越短；角度恢复可分性后，语义距离才更可信。

SimCSE 的价值不只是“做对比学习”，而是它同时优化两个目标：**alignment** 和 **uniformity**。alignment 可以白话理解为“语义相同或相近的样本要靠拢”；uniformity 可以理解为“整体分布不要挤在几个方向上”。白化先把空间尽量拉圆，SimCSE 再在这个更健康的空间里继续训练，通常会让句向量检索更稳定。

下面这个结果足够说明问题：

| 方法 | STS 平均 Spearman |
|---|---:|
| SimCSE-BERT-base | 76.25 |
| WhitenedCSE / 白化增强后 | 78.78 |

这不是说“白化必胜”，而是说：当任务本身就是拿余弦做语义比较时，先处理各向异性，往往能恢复本来被空间畸形掩盖掉的语义结构。

---

## 问题定义与边界

本文讨论的是**嵌入空间的几何分析**，更具体地说，是下面这个问题：

> 当我们把句子、词或代码片段编码成向量后，这些向量的空间形状是否适合用余弦相似度做语义比较？

这里的“嵌入”可以白话理解为：模型把文本压成一个高维数字数组，用它表示语义。很多系统默认认为“语义相近的文本，向量夹角也应该更小”。但这个前提只有在空间分布比较均匀时才成立。

各向异性带来的两个典型症状是：

1. **平均余弦偏高**  
   任意拿两条不相关句子，余弦也可能是 0.85。这说明“高分”已经不稀缺，排序自然失真。

2. **hubness 现象严重**  
   hubness 可以白话理解为：总有少数向量会变成“万人迷邻居”，无论查什么都容易被召回。它不是因为这些向量真的语义通用，而是因为它们恰好占据了空间里的主方向。

一个新手容易理解的例子是：  
“今天天气很好” 和 “数据库索引失效导致查询变慢” 本该很远，但如果向量都挤在同一锥体里，它们的余弦依然可能高达 0.85。模型不是觉得它们语义相同，而是它的空间几何已经不适合做细粒度区分。

下面用表格看这个问题：

| 对象 | 分布特征 | 常见后果 |
|---|---|---|
| 高频词相关向量 | 更集中在少数主方向 | 更容易抬高平均余弦 |
| 低频词相关向量 | 分布更分散 | 区分度相对更好 |
| 全体句向量 | 形成高密度锥体 | 近邻检索失真、hubness 增强 |

如果用文字画图，可以这样理解：

- 失真空间：大量向量从原点出发，密集指向一个狭窄锥体。
- 理想空间：向量在单位球面附近更均匀分散，不同方向都有足够覆盖。

本文的边界也要说清楚：

| 场景 | 是否适合直接讨论白化收益 |
|---|---|
| STS、语义检索、向量召回 | 适合 |
| 只看分类 logits 的分类任务 | 不一定 |
| 依赖非归一化距离的任务 | 不一定 |
| 端到端再训练的大模型系统 | 需要单独验证 |

原因很简单：本文核心假设是“最终比较方式主要是余弦相似度”。如果你的下游任务靠的是一个可训练分类头，白化不一定提升，甚至可能破坏原有的线性可分结构。

---

## 核心机制与推导

先看白化。设嵌入向量经过中心化后的协方差矩阵为：
$$
\Sigma = U\Lambda U^\top
$$
其中 $U$ 是特征向量矩阵，$\Lambda$ 是特征值对角矩阵。白化的目标是找到一个线性变换 $W$，让变换后的协方差接近单位阵：
$$
W\Sigma W^\top = I
$$

最常见的是 ZCA 白化：
$$
W_{\text{ZCA}} = U\Lambda^{-1/2}U^\top
$$

如果担心小特征值导致数值不稳定，会加入一个平滑项 $\epsilon$，得到 Soft-ZCA：
$$
W_{\text{soft}} = U(\Lambda+\epsilon I)^{-1/2}U^\top
$$

白话解释是：  
先把数据坐标轴旋转到“最主要变化方向”的坐标系里，再把每个方向按方差大小缩放，最后旋回原坐标系。原来被拉得很长的方向会缩短，原来被压得很扁的方向会拉开，于是整体更接近球形。

这件事为什么会改善余弦？因为在归一化条件下，余弦和欧氏距离等价，而欧氏距离是否有判别力，取决于不同方向的尺度是不是畸形。如果某几个方向的方差远高于其他方向，那么“角度”本身就被主方向劫持了。

再看另一条常见路线：去掉 top-\(k\) 主成分。  
主成分可以白话理解为：数据最主要的几个公共方向。很多时候这些方向承载的不是你想要的精细语义，而是词频、模板句式、标点习惯、训练语料偏置等全局噪声。去掉前几个主成分，本质上是在做一种更便宜的均匀化。

一个新手可理解的箭头例子是：

- 原始空间：很多箭头都朝东北方向偏一点。
- 去主成分：把“最明显的东北偏向”先减掉。
- 白化：不只是减掉一个方向，而是把所有方向都缩放到更平衡的尺度。

它们的数学效果可以对照看：

| 操作 | 数学形式 | 作用 |
|---|---|---|
| 中心化 | $x \leftarrow x-\mu$ | 去掉整体偏移 |
| 去 top-\(k\) 主成分 | $x \leftarrow x-\sum_{i=1}^k (u_i^\top x)u_i$ | 去除公共主方向 |
| ZCA 白化 | $x \leftarrow U\Lambda^{-1/2}U^\top x$ | 全方向方差拉平 |
| Soft-ZCA | $x \leftarrow U(\Lambda+\epsilon I)^{-1/2}U^\top x$ | 更稳定的白化 |
| 归一化 | $x \leftarrow x/\|x\|_2$ | 让余弦比较成立 |

SimCSE 的解释要放在几何框架里看。它常被写成“dropout 产生正样本，再做对比学习”，但几何上更重要的是下面两个趋势：

$$
\ell_{\text{align}} = \|h_i-h_j\|_2^2
$$

alignment 表示正样本对要靠近。比如同一句子两次 dropout 编码得到的两个向量，不应该漂得很远。

uniformity 常用“让所有样本不要塌缩到一起”的形式表达。不同论文写法不同，但核心思想一致：负样本之间要占据更均匀的空间。

因此可以把它理解成：

- 白化解决“空间底座是歪的”
- SimCSE 解决“在这个空间里怎么重新组织样本关系”

真实工程里，两者常常组合使用。比如在线语义检索服务中，先离线统计一批句向量的协方差做白化，再在推理时对所有向量应用同一个变换，最后做归一化与余弦排序。这样做不改变主模型结构，但会直接改变检索几何。

---

## 代码实现

下面先给一个可运行的玩具实现。它做了四件事：中心化、估计协方差、Soft-ZCA 白化、归一化。代码里用一个刻意“挤在单一方向”的小数据，展示白化后平均余弦会下降，说明空间更分散。

```python
import numpy as np

def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, eps, None)

def mean_cosine(x):
    x = l2_normalize(x)
    sim = x @ x.T
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return sim[mask].mean()

def soft_zca_whiten(x, eps=1e-3):
    # x: [n, d]
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu

    cov = (xc.T @ xc) / x.shape[0]
    U, S, _ = np.linalg.svd(cov, full_matrices=False)

    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    z = xc @ W.T
    return z, mu, W

# 人工构造一个强各向异性的玩具数据：大多数样本都沿第一维展开
X = np.array([
    [10.0, 0.2, 0.1],
    [9.5, -0.1, 0.0],
    [10.5, 0.1, -0.2],
    [9.8, 0.0, 0.2],
    [10.2, -0.2, -0.1],
], dtype=np.float64)

before = mean_cosine(X)
Z, mu, W = soft_zca_whiten(X, eps=1e-3)
after = mean_cosine(Z)

# 白化后协方差应更接近单位阵
cov_z = ((Z - Z.mean(axis=0)).T @ (Z - Z.mean(axis=0))) / Z.shape[0]

assert before > after
assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-8)
assert np.allclose(cov_z, np.eye(3), atol=5e-2)

print("mean cosine before:", round(before, 4))
print("mean cosine after :", round(after, 4))
```

这个例子里，第一维方差远大于其他维，导致所有样本几乎都朝同一个方向。白化之后，协方差被拉到接近单位阵，平均余弦会明显下降。

如果只想做更轻量的“去主成分”，可以这样写：

```python
import numpy as np

def remove_topk_components(x, k=1):
    x = x - x.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(x, full_matrices=False)
    comps = Vt[:k]              # [k, d]
    proj = x @ comps.T @ comps  # 投影到 top-k 主方向
    y = x - proj
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return y

X = np.array([
    [1.0, 0.1, 0.0],
    [0.9, -0.1, 0.1],
    [1.1, 0.0, -0.1],
], dtype=np.float64)

Y = remove_topk_components(X, k=1)
assert Y.shape == X.shape
assert np.allclose(np.linalg.norm(Y, axis=1), 1.0)
```

两段代码的区别是：

| 方法 | 处理方式 | 复杂度与效果 |
|---|---|---|
| 去 top-k 主成分 | 只削掉最强公共方向 | 更简单，更便宜 |
| Soft-ZCA 白化 | 对所有方向做缩放平衡 | 更系统，通常更稳 |

如果放进 SimCSE 流程，可以把逻辑理解成下面这个文字流程图：

`embedding -> projection(optional) -> whiten -> normalize -> cosine similarity`

更接近工程实现的伪代码如下：

```python
# 训练后或验证集统计得到 whiten 参数
# mu: [d], W: [d, d]

def encode_for_retrieval(texts, encoder, mu, W):
    h = encoder(texts)          # [batch, d]
    h = h - mu
    h = h @ W.T
    h = h / h.norm(dim=-1, keepdim=True)
    return h
```

真实工程例子：  
做 FAQ 检索时，离线先把 10 万条知识库句子编码，再用一批开发集句向量估计 $\mu$ 和 $W$。之后对 query 和 document 都使用同一组白化参数，再做 ANN 检索或余弦排序。这样常常不需要改模型权重，只改后处理，就能稳定提高 Top-K 召回和 STS 类指标。

---

## 工程权衡与常见坑

白化不是“通用增强器”，它更像一个针对几何失真的修正器。判断是否值得上，先看你的任务是不是严重依赖余弦空间。

最重要的权衡是：**uniformity 提升了，不代表 alignment 一定还在。**  
如果你把空间打得太圆，可能会把原本对分类有利的方向信息也一起抹平。特别是在带监督分类里，某些主方向恰恰编码了类别边界。

这就是为什么 alignment 仍然重要：
$$
\ell_{\text{align}} = \|h_i - h_j\|_2^2
$$
如果正样本对被白化后拉散过头，语义上该靠近的样本也会丢失局部结构。

下面这个对比更实用：

| 任务类型 | 白化常见效果 | 风险 |
|---|---|---|
| STS | 通常提升 | 参数估计不稳会波动 |
| 语义检索 | 常见提升 | 训练域和上线域不一致时收益下降 |
| 分类任务 | 不稳定，可能下降 | 主方向被破坏，线性边界变差 |
| 聚类任务 | 视数据而定 | 可能改变簇间相对结构 |

几个常见坑：

| 坑点 | 说明 | 规避方式 |
|---|---|---|
| 用太小样本估计协方差 | 白化矩阵不稳定 | 用足够大的开发集或训练子集 |
| 忘记中心化 | 白化前提不成立 | 先减均值再算协方差 |
| 直接用硬 ZCA | 小特征值导致爆炸 | 加 $\epsilon$ 做 Soft-ZCA |
| 查询和库向量处理不一致 | 排序空间不统一 | 两边使用同一个 $\mu, W$ |
| 分类场景盲目套用 | 可能损失边界信息 | 单独做离线验证 |

新手可以记住一个很实用的判断标准：  
如果你的系统最后一步就是“算余弦，然后按分数排序”，白化值得试；如果你的系统最后一步是“接一个分类头输出类别”，那就不要先假设它一定有益。

---

## 替代方案与适用边界

白化不是唯一方法。它的替代方案大致分三类：去中心化与归一化、PCA 主成分裁剪、训练阶段的对比式正则。

先看对比：

| 方法 | 操作步骤 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 仅归一化 | $L_2$ normalize | 最简单 | 不解决主方向塌缩 | 基础余弦检索 |
| 去 top-k 主成分 | 中心化 + PCA 裁剪 + 归一化 | 便宜、实现简单 | 只处理少数方向 | 检索后处理 |
| 白化 | 中心化 + 协方差分解 + 全方向缩放 | 最系统 | 需要稳定估计协方差 | STS、语义检索 |
| 对比学习正则 | 训练中优化 alignment/uniformity | 可从源头改善空间 | 需要训练成本 | 有训练预算的系统 |

新手可以把 PCA 裁剪理解成“剃掉最突出的几个方向”，把白化理解成“给所有方向重新校准尺度”。前者像局部修补，后者像整体校正。

一个简单对照代码如下：

```python
# PCA 去 top-k：减掉最强公共方向
y = x - x.mean(axis=0, keepdims=True)
# ... SVD 后减投影

# 白化：所有方向都缩放到更平衡
z = (x - mu) @ W.T
z = z / ||z||
```

什么时候优先选哪种方法？

1. 数据不多、想快速试验：先试去 top-k 主成分。  
2. 有稳定开发集、任务明确依赖余弦：优先试 Soft-ZCA 白化。  
3. 可以继续训练模型：把白化与对比学习结合，通常比纯后处理更稳。  
4. 目标是分类准确率而不是语义相似排序：不要默认采用白化。

真实工程里，常见做法不是二选一，而是分层使用：

- 轻量系统：`中心化 + 去 top-k + 归一化`
- 语义检索系统：`白化 + 归一化`
- 训练可控系统：`对比学习 + 白化后处理`

因此，“哪种最好”不是一个抽象理论问题，而是一个依赖任务目标、样本规模、上线约束的工程问题。

---

## 参考资料

1. Su, J. et al. “Whitening Sentence Representations for Better Semantics and Retrieval.”  
用途：解释句向量各向异性、白化为何能改善余弦语义比较。  
URL: https://www.aimodels.fyi/papers/arxiv/semantics-at-angle-when-cosine-similarity-works?utm_source=openai

2. Gao, T., Yao, X., Chen, D. “SimCSE: Simple Contrastive Learning of Sentence Embeddings.”  
用途：理解 SimCSE 中 alignment 与 uniformity 的训练目标，以及为什么对比学习能改善句向量几何。  
URL: https://arxiv.org/abs/2104.08821

3. “On Isotropy of Multimodal Embeddings.” MDPI.  
用途：说明 isotropy（各向同性）与 embedding 几何质量的关系，并讨论主成分与白化机制。  
URL: https://www.mdpi.com/2380676?utm_source=openai

4. “Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search.”  
用途：理解 Soft-ZCA 的数值稳定版本，以及工程上为何常加 $\epsilon$。  
URL: https://www.themoonlight.io/en/review/isotropy-matters-soft-zca-whitening-of-embeddings-for-semantic-code-search?utm_source=openai

5. “WhitenedCSE: Whitening-based Contrastive Learning for Sentence Embeddings.”  
用途：查看白化结合对比学习后在 STS 上的增益示例，如 76.25 到 78.78 的提升。  
URL: https://liner.com/review/whitenedcse-whiteningbased-contrastive-learning-sentence-embeddings?utm_source=openai

6. “Applied Sciences 14(19): 8887.” MDPI.  
用途：讨论白化可能带来的副作用，尤其是在分类与非检索场景中的性能权衡。  
URL: https://www.mdpi.com/2076-3417/14/19/8887?utm_source=openai

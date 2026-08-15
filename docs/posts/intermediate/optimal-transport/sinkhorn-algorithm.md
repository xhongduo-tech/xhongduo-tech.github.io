---
title: 熵正则化与 Sinkhorn 算法
date: 2026-08-07
---

# 熵正则化与 Sinkhorn 算法

<div class="epigraph">
<p>熵正则化让最优传输从一个昂贵的线性规划，变成一个光滑的、可用矩阵乘法逼近的问题。</p>
<footer>—— 马尔科 · 库图里（Marco Cuturi），《Sinkhorn Distances: Lightspeed Computation of Optimal Transport》（2013，意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优传输理论 ｜ Peyré & Cuturi《Computational Optimal Transport》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从熵正则化开始

前六篇建立的理论很美，但落到计算上有致命短板。离散 Kantorovich 问题

$$
\min_{P \in \Pi(a,b)} \sum_{i,j} C_{ij} P_{ij}
$$

是一个线性规划，变量数 $n^2$，通用线性规划求解器的复杂度在 $O(n^3 \log n)$ 量级——当 $n$ 是图像像素（百万级）或批量样本（十万级）时，完全不可行。2013 年，**马尔科 · 库图里（Marco Cuturi）**给问题加了一小勺"熵正则化"，把它变成一个**强凸的光滑问题**，并指出其最优解具有稀疏但极好算的"乘积形式"，催生了 **Sinkhorn 算法**——今天几乎所有计算最优传输的工具箱（POT、GeomLoss、OTT）都以它为核心。<span class="marginnote">Sinkhorn 这个名字来自 Richard Sinkhorn 1967 年关于"非负矩阵的对角缩放"的定理：给一个正矩阵，交替归一化行与列会收敛到"双随机"矩阵。库图里的贡献是看出这个古老算法恰好能解熵正则化最优传输。</span>

## 1 从线性规划到光滑近似

线性规划最优解落在多面体顶点上——传输多面体的顶点往往极端稀疏且支集复杂，求解器要处理大量的基变量交换，这就是慢的根源。熵正则化的想法：**给目标函数加一个负熵项，让解"变糊"（更均匀、更光滑），同时把强凸性送进来。**

熵正则化问题：

$$
\min_{P \in \Pi(a,b)} \; \sum_{i,j} C_{ij} P_{ij} + \varepsilon \sum_{i,j} P_{ij} \log P_{ij}
$$

参数 $\varepsilon > 0$ 是正则化强度。$\varepsilon \to 0$ 时恢复原问题；$\varepsilon \to \infty$ 时解趋于"最均匀的耦合" $a \otimes b$（独立乘积）。<span class="marginnote">负熵项 $\sum P_{ij}\log P_{ij}$ 越小意味着分布越"尖锐"。减去一个 $\varepsilon$ 倍的负熵，等价于鼓励 $P$ 接近均匀——这是"正则化"两个字的来历。$\varepsilon$ 在信息论里也叫"温度"：温度高，解越熵化、越便宜但越模糊。</span>

## 2 公式解析：最优解的乘积形式

把熵正则化问题写完整，并给出它的解的结构。记 $a, b$ 为两个离散分布（边际），$C$ 为代价矩阵，$\varepsilon > 0$。定义核矩阵

$$
K_{ij} = e^{-C_{ij}/\varepsilon}
$$

那么熵正则化问题有唯一最优解，且形如

$$
P_{ij} = u_i \, K_{ij} \, v_j
$$

其中 $u, v$ 是两个非负向量，满足边际条件 $\sum_j P_{ij} = a_i$、$\sum_i P_{ij} = b_j$。拆成三步理解：

- **第一步，读懂核矩阵 $K$**：$K_{ij}$ 是"代价越小、权重越大"的指数函数。$\varepsilon$ 大时 $K$ 接近全 1 矩阵（无差别），$\varepsilon$ 小时 $K$ 尖锐地集中在低代价通路上。它是把代价矩阵翻译成"相似度"的软化版本。
- **第二步，读懂乘积形式**：$P = \mathrm{diag}(u)\, K\, \mathrm{diag}(v)$——最优耦合是核矩阵的**行与列分别缩放**。这正是"对角缩放"结构，Sinkhorn 定理保证：反复调整 $u, v$ 总能同时满足两个边际。
- **第三步，读懂为什么可算**：边际条件化作两个**显式方程**

$$
u \odot (K v) = a, \qquad v \odot (K^{\mathsf{T}} u) = b
$$

其中 $\odot$ 是逐分量相乘。这两个方程可以交替求解（见下节），每一步只是矩阵–向量乘法，复杂度 $O(n^2)$，天然适合 GPU。

**辨析｜易错点：** 熵正则化解**不是**原问题的最优解，它有一个 $O(\varepsilon \log \ldots)$ 量级的偏差。$\varepsilon$ 取太大解太糊、取太小又回到慢的困境。实践中 $\varepsilon$ 通常取在代价量级的一个可分数上（如 $C$ 的 1%），并随应用调参。<span class="marginnote">"先求熵正则化解、再令 $\varepsilon\to0$ 外推"这一派叫 Sinkhorn 外推；库图里 2019 年的后续工作又把它与"对偶加速"结合。偏差可控是它进入工程的前提——我们在第 9 篇应用里会看到它在图像上的实际表现。</span>

## 3 Sinkhorn 算法：交替缩放的推导

Sinkhorn 算法就是把上节两个方程交替求解。从任意正向量 $v^{(0)} = \mathbf{1}$ 出发，反复迭代直到收敛：

$$
u^{(k+1)} = \frac{a}{K v^{(k)}}, \qquad v^{(k+1)} = \frac{b}{K^{\mathsf{T}} u^{(k+1)}}
$$

每一轮做两次矩阵–向量乘法与两次逐分量除法。为什么有效？把每一步展开看：

- **第一步，$u$ 更新**：固定 $v$，缩放 $u$ 使行和 $\sum_j u_i K_{ij} v_j$ 恰好等于 $a_i$。除法直接给出精确解，因为 $u_i$ 只出现在第 $i$ 行的每个元素里（线性）。
- **第二步，$v$ 更新**：固定新的 $u$，缩放 $v$ 使列和等于 $b_j$。同理。
- **第三步，迭代**：每轮固定一侧、校正另一侧，交替投影到两个线性约束上。这与**迭代比例拟合（iterative proportional fitting, IPF）**在数学上完全等价，单调收敛到唯一解。

收敛速度通常极快（几十轮内达到机器精度量级），且每轮都可以批量并行。<span class="marginnote">Sinkhorn 迭代还能解读为<strong>块坐标上升</strong>：它交替地在对偶变量 $\varphi_i$（行势）与 $\psi_j$（列势）上精确极大化。对偶视角见第 3 篇——Sinkhorn 只是把"最优性条件"变成"可执行的更新规则"。</span>

## 4 实现要点与 ε 的选择

工程实现时有三件事值得注意：

**数值稳定性**：$\varepsilon$ 很小时 $K_{ij} = e^{-C_{ij}/\varepsilon}$ 可能下溢到 0。解决方法是把 $u, v$ 吸收进"对数域"：定义 $\varphi_i = \varepsilon \log u_i$、$\psi_j = \varepsilon \log v_j$，迭代变成

$$
\varphi^{(k+1)} = -\varepsilon \log\Big( K v^{(k)} \Big), \qquad
\psi^{(k+1)} = -\varepsilon \log\Big( K^{\mathsf{T}} u^{(k+1)} \Big)
$$

并在每轮减去常数以防溢出。<span class="marginnote">对数域里 $K v$ 变成 log-sum-exp 运算，是经典的"softmax 型"稳定技巧，与深度学习里避免 softmax 上溢的做法完全相同。</span>

**ε 的选择**：ε 控制"平滑–保真"折中。常见做法是取 $\varepsilon$ 约为代价矩阵标准差的一小部分；多尺度方案（从大 ε 开始、逐层减半）能显著加速。

**边际不平衡**：真实数据里边际 $a, b$ 往往不平衡。可用**无界熵正则化**（只约束一边）或引入 "Kantorovich 松弛" 的平衡项处理，POT 库中均有实现。

## 5 一个微型 Sinkhorn 迭代

用最小的 $2 \times 2$ 例子亲手跑几轮，把"交替缩放"变成肌肉记忆。设

$$
a = \begin{pmatrix} 0.6 \\ 0.4 \end{pmatrix}, \quad
b = \begin{pmatrix} 0.4 \\ 0.6 \end{pmatrix}, \quad
C = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \varepsilon = 1
$$

核矩阵 $K_{ij} = e^{-C_{ij}/\varepsilon}$，即 $K = \begin{pmatrix} 1 & e^{-1} \\ e^{-1} & 1 \end{pmatrix} \approx \begin{pmatrix} 1 & 0.368 \\ 0.368 & 1 \end{pmatrix}$。从 $v^{(0)} = (1,1)$ 出发迭代：

| 轮次 | $u$ | $v$ |
| --- | --- | --- |
| 0 | — | $(1.000, 1.000)$ |
| 1 | $(0.439, 0.292)$ | $(0.732, 1.322)$ |
| 2 | $(0.492, 0.251)$ | $(0.684, 1.388)$ |
| 3 | $(0.487, 0.251)$ | $(0.684, 1.393)$ |

可以看到 $u, v$ 在几轮内就趋于稳定。取第 3 轮的缩放向量，得到耦合

$$
P = \mathrm{diag}(u)\, K\, \mathrm{diag}(v) \approx
\begin{pmatrix} 0.333 & 0.267 \\ 0.067 & 0.333 \end{pmatrix}
$$

验证边际：行和 $0.333+0.267 = 0.600$、$0.067+0.333 = 0.400$；列和 $0.333+0.067 = 0.400$、$0.267+0.333 = 0.600$。两条边际都精确匹配 $a, b$。<span class="marginnote">观察解的结构：对角线项大（$C_{ii}=0$ 便宜）、非对角线项小（$C_{ij}=1$ 贵）——熵正则化解在"便宜通路多给质量"与"整体尽量均匀"之间折中。$\varepsilon$ 越小，对角占比越高，越接近精确 LP 解。</span>

**辨析｜易错点：** 收敛后**不必再归一化**。当 $u,v$ 满足 $u \odot (Kv) = a$ 与 $v \odot (K^{\mathsf{T}}u) = b$ 时，$\mathrm{diag}(u)K\mathrm{diag}(v)$ 的行和列和已经自动等于 $a,b$；若中途额外做一次归一化反而会破坏已达平衡的边际。工程库（POT 的 `sinkhorn`）会检测残差 $\|u\odot(Kv) - a\|$ 来决定何时停。

## 6 小结

- 离散最优传输是线性规划，$O(n^3\log n)$ 量级，大规模不可行。
- **熵正则化**：$\min \langle C,P\rangle + \varepsilon \sum P_{ij}\log P_{ij}$，问题变强凸光滑。
- 最优解呈**乘积形式** $P = \mathrm{diag}(u) K \mathrm{diag}(v)$，$K_{ij}=e^{-C_{ij}/\varepsilon}$。
- **Sinkhorn 算法**：交替缩放 $u \leftarrow a/(Kv)$、$v \leftarrow b/(K^{\mathsf{T}}u)$，每轮 $O(n^2)$
---
title: 非交换空间的思想（算子代数、von Neumann 代数、Morita 等价）
date: 2026-08-17
---

# 非交换空间的思想

<div class="epigraph">
<p>我们观察到的并不是自然本身，而是自然对我们追问方式所作的回应。</p>
<footer>—— 维尔纳 · 海森堡（Werner Heisenberg）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.2; Connes《Noncommutative Geometry》Ch.I ｜ 2026-08-17</p>
</div>

## 为什么从空间的思想开始

上一节我们确立了 Gelfand 对偶：紧 Hausdorff 空间 $\cong$ 交换 C\*-代数。本节要回答一个更根本的问题：**当我们松开交换律时，那个「空间」还在吗？** 答案是——它不在，但取而代之的是一个更丰富的结构：**非交换空间（noncommutative space）**。

「非交换空间」这个词本身就是个悖论。空间由点组成，点与点之间的代数不交换意味着什么？答案藏在量子力学里：物理学家早就知道，在量子尺度下「轨迹」不存在，粒子没有确定的位置。海森堡的不确定性原理并不是实验误差，而是**世界的代数结构本身就是非交换的**。非交换几何的雄心，就是为这类「没有点的空间」建立完整的几何学。

本节是概念性的枢纽：它把前一篇的 C\*-代数（拓扑层面）与 von Neumann 代数（测度层面）连接起来，并引入 Morita 等价这个「两个代数什么时候是同一个非交换空间」的判据。这是后续所有内容的地基。

## 1 空间如何消失：非交换空间的定义

### 1.1 核心原则

**非交换空间（noncommutative space）**：一个非交换 C\*-代数 $A$ 被理解为一个「虚空间」上的连续函数代数，尽管这个空间没有点。<span class="marginnote">这是 Connes 在 1970 年代末提出的纲领：几何学的对象不是空间本身，而是其上的函数代数。Gelfand 对偶保证了在交换情形下两者完全等价；非交换情形下我们「把代数当作空间」来研究。</span>

为什么说没有点？因为「点」在代数语言里就是特征（乘性线性泛函）。而非交换代数上通常不存在非零特征：

**辨析｜易错点：** 并非所有非交换代数都没有特征。例如上三角矩阵代数 $T_n(\mathbb{C})$ 有特征（取对角元）。没有特征的是「足够非交换」的代数，如 $M_n(\mathbb{C})$、$\mathcal{B}(\mathcal{H})$、Cuntz 代数 $\mathcal{O}_n$。对这类代数，$\Delta(A) = \emptyset$，空间确实消失了。

### 1.2 空间消失的三个层面

Connes 把「空间」拆成三个互相独立的结构，分别对应三种算子代数：

| 空间结构 | 经典对应 | 非交换对应 | 代数类型 |
| --- | --- | --- | --- |
| 拓扑 | 开集、连续函数 | C\*-代数 $A$ | $C(X)$ |
| 测度 | 可测集、积分 | von Neumann 代数 $M$ | $L^\infty(X)$ |
| 度规 | 距离、度量 | 谱三元组 | $(A, H, D)$ |

这个「三件套」——拓扑、测度、度规——是非交换几何的总纲领。前两件在本节讲，第三件（谱三元组）在《谱三元组》一文展开。

## 2 von Neumann 代数：测度论的非交换化

### 2.1 从 C\* 代数到 von Neumann 代数

C\*-代数对应拓扑；von Neumann 代数对应测度论。它们的差别在于完备性：C\*-代数对范数完备，von Neumann 代数对更细的**弱算子拓扑（weak operator topology）**完备。

**von Neumann 代数**：$\mathcal{B}(\mathcal{H})$ 中在弱算子拓扑下闭、含单位元、且 $M'' = M$ 的 $*$-子代数。

### 2.2 双换位子定理

von Neumann 双换位子定理是这一理论的基石：

$$
M = (M')' \quad \text{即} \quad M'' = M
$$

其中 $M' = \{T \in \mathcal{B}(\mathcal{H}) \mid TS = ST \ \forall S \in M\}$ 是 $M$ 的**换位子（commutant）**。<span class="marginnote">这个定理 1929 年由 von Neumann 证明。它的惊人之处在于：一个纯代数的条件（等于自身双换位子）等价于一个拓扑条件（弱闭）。这一等价是「非交换测度论」的理论基础。</span>

**公式解析：双换位子定理** 该等式分三步理解：

- **第一步**，$M \subseteq M''$ 是显然的：$M$ 中的元素自然与 $M'$ 中所有元素交换。
- **第二步**，困难的方向 $M'' \subseteq M$：任取 $T \in M''$，需要对任意 $\xi \in \mathcal{H}$ 证明 $T\xi$ 可以用 $M$ 中的算子逼近。这要用到 $M'$ 中的投影算子（通过 Gram–Schmidt 投影定理构造）把 $\mathcal{H}$ 分解为 $M$-循环子空间。
- **第三步**，技术核心：对 $M$-循环向量 $\xi$（即 $\overline{M\xi} = \mathcal{H}$），定义算子 $\pi$ 使 $T\xi = \pi \xi$。由 $T \in M''$ 可推出 $\pi$ 与 $M'$ 交换，再由 $M' \supseteq M''$ 且 $\pi$ 与所有 $M'$ 交换得出 $\pi \in M$。

**结论：弱闭 $*$-子代数恰好是「双换位子闭」的 $*$-子代数。** 这让我们可以在不谈及拓扑的情况下纯代数地刻画 von Neumann 代数。

### 2.3 因子与 Murray–von Neumann 分类

**因子（factor）**：中心 $Z(M) = M \cap M'$ 只含数乘 $\mathbb{C}1$ 的 von Neumann 代数。Murray 与 von Neumann（1936–1943 年一系列论文）证明了因子按类型分为三类：

- **I 型**：$\mathcal{B}(\mathcal{H})$ 型，如 $M_n(\mathbb{C})$、$\mathcal{B}(\mathcal{H})$。
- **II 型**：有无穷维但「有限」的投影，如超有限 II$_1$ 因子（$R$，约当代数）。
- **III 型**：所有投影都无穷，对应「无穷温度」的物理系统，如 $L^\infty(\mathbb{R})$ 在群作用下的交叉积。

Connes 1973 年在 III 型因子分类上做出了突破性工作，提出模自同构（modular automorphism）与权流（flow of weights），这为后续非交换测度论和指标理论铺路。<span class="marginnote">III 型因子最初由 Murray 与 von Neumann 在 1936 年的论文中定义，但直到 1970 年代才被 Connes 彻底分类。Connes 因此获得 1982 年 Fields 奖。</span>

### 2.4 物理意义

量子统计力学里，物理系统的「状态」是 $M$ 上的正规态（normal state），「可观测量」是 $M$ 的自伴元素，「对称」是 $M$ 的自同构。III 型因子精确刻画了有限温度下的热平衡态（KMS 态）。这告诉我们：**非交换测度论不是抽象游戏，它就是量子统计力学的数学语言。**

## 3 Morita 等价：什么时候两个代数「是同一个空间」

两个不同的代数可能描述同一个非交换空间。例如 $M_n(\mathbb{C})$ 与 $\mathbb{C}$：一个矩阵代数看起来比复数复杂得多，但在非交换几何的意义上它们对应同一个「单点空间」。

### 3.1 代数 Morita 等价

**Morita 等价（Morita equivalence）**：两个环 $A, B$ 称为 Morita 等价，如果它们的左模范畴等价：$\mathbf{Mod}_A \simeq \mathbf{Mod}_B$。

**定理（Morita）**：$A$ 与 $B$ Morita 等价当且仅当存在双模 $P$（$A$-$B$ 双模）与 $Q$（$B$-$A$ 双模）使得 $P \otimes_B Q \cong A$、$Q \otimes_A P \cong B$（作为相应双模）。

**例子**：$M_n(\mathbb{C})$ 与 $\mathbb{C}$ Morita 等价。事实上，任意 $A$ 与其矩阵代数 $M_n(A)$ Morita 等价——矩阵代数只是「同一点的 $n$ 维纤维化」。这就是为什么矩阵代数被称为「把空间拷贝 $n$ 份」：它的模是 $A$ 的模的直和。

### 3.2 强 Morita 等价（Rieffel）

对 C\*-代数，需要分析版的 Morita 等价，称为**强 Morita 等价（strong Morita equivalence）**，由 Rieffel 在 1974 年引入：

**强 Morita 等价**：$A, B$ 强 Morita 等价，如果存在 $A$-$B$ 等价双模（equivalence bimodule）$\mathcal{E}$，它是一个 Hilbert $B$-模同时又是 $A$ 模，且满足伴随条件 $\langle a\xi, \eta\rangle_B = \langle \xi, a^*\eta\rangle_B$。

**关键性质**：强 Morita 等价的 C\*-代数有相同的 K-理论。这导出惊人结论——**K-理论是 Morita 不变的**，从而 K-理论不区分「同一点的不同纤维化」。

### 3.3 为什么需要 Morita 等价

**核心对比表**：Morita 等价 vs 同构：

| 概念 | 同构 | Morita 等价 |
| --- | --- | --- |
| 定义 | 存在 $*$-同构 | 模范畴等价 |
| 保持 | 所有代数性质 | K-理论、谱 |
| 区别 | 点的结构 | 点的结构 |
| 例子 | $M_2 \not\cong \mathbb{C}$ | $M_2 \sim \mathbb{C}$ |

**物理含义**：物理上两个 Morita 等价的代数描述的是同一个物理系统，只是观察的「坐标系」不同。这有点像两种语言描述同一个流形，坐标图不同但流形本身不变。

## 4 小结

- **非交换空间** = 没有点的空间，由一个非交换 C\*-代数或 von Neumann 代数表示；空间结构拆成拓扑、测度、度规三件套。
- **von Neumann 代数**是非交换测度论；双换位子定理 $M'' = M$ 使弱闭等价于代数闭。
- **因子分类**（Murray–von Neumann）：I、II、III 型；III 型因子刻画无穷温度量子系统，由 Connes 彻底分类（1973）。
- **Morita 等价**：两个代数同一个非交换空间的判据；矩阵代数与标量代数 Morita 等价；K-理论对其不变。
- **哲学主线**：非交换几何用代数重写几何，交换性只是其中一条性质，不是前提。

在下一节，我们将看到空间上的几何对象——**向量丛**——如何在非交换框架下变成代数上的模，这就是 Serre–Swan 定理与**非交换向量丛**。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.2; Connes《Noncommutative Geometry》Ch.I; 对 III 型因子分类与 KMS 态，见 Connes Ch.V。</span>
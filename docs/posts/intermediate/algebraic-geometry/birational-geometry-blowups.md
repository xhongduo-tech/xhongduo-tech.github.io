---
title: 双有理几何：blow-up 与有理映射
date: 2026-08-11
---

# 双有理几何：blow-up 与有理映射

<div class="epigraph">
<p>双有理几何把代数簇当作"由有理函数决定的形状"来研究——形状的大部分细节都是外衣，内里只有一个核心。</p>
<footer>—— 由奥斯卡 · 扎里斯基（Oscar Zariski）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从双有理几何继续

第 3 篇我们定义了双有理等价：$X \sim Y$ 当且仅当 $K(X) \cong K(Y)$——"在差一个稠密开集的意义下同构"。本专题最后三篇都围绕这个等价关系展开。本节的主角是**blow-up（爆炸 / 拉开）**：把簇上某点"展开"成一条例外除子，从而把奇点"拉直"。这是双有理几何最基础的手术刀，也是第 15 篇相交理论的核心工具。

为什么值得学？因为**几乎所有"修好奇点"的过程都归结为有限次 blow-up**（Hironaka 的解消定理），而"哪些性质在 blow-up 下不变"恰好是双有理几何的中心问题。blow-up 在几何上与"把过点的所有方向打开"一致——如本节的图所示：一点被展开成 $\mathbb{P}^{n-1}$，穿过该点的曲线被"分成"互不粘连的分支。这个操作与深度学习里的"注意力展开"在精神上同构：把"一点处的模糊"展开成"一个方向空间"。

## 1 射影空间的 blow-up：直觉与定义

**核心概念：$\mathbb{P}^n$ 在一点的 blow-up。** 设 $P \in \mathbb{P}^n$。取 $P$ 处的投影，即有理映射

$$\pi: \mathbb{P}^n \dashrightarrow \mathbb{P}^{n-1}, \qquad X \longmapsto \text{直线 } \overline{PX} \text{ 的方向}$$

它在 $P$ 处未定义（所有方向都从 $P$ 出发）。**Blow-up** 就是把 $\pi$ 的定义域"修补好"：令

$$\widetilde{\mathbb{P}^n} = \{ (X, \ell) \in \mathbb{P}^n \times \mathbb{P}^{n-1} \mid X \in \ell \}$$

（$X$ 在过 $P$ 的直线 $\ell$ 上），并装备自然射影 $\varepsilon: \widetilde{\mathbb{P}^n} \to \mathbb{P}^n$，$(X, \ell) \mapsto X$。<span class="marginnote">$\varepsilon$ 在 $P$ 处的纤维是 $\varepsilon^{-1}(P) = \{(P, \ell)\} \cong \mathbb{P}^{n-1}$——所有过 $P$ 的方向。这条纤维叫<strong>例外除子（exceptional divisor）</strong> $E$。除了 $P$ 外，$\varepsilon$ 在其他点是一一对应（保 $X \neq P$ 时唯一确定 $\ell$）。于是"$P$ 被换成一条 $\mathbb{P}^{n-1}$"。</span>

**重点：blow-up 是双有理态射。** $\varepsilon: \widetilde{\mathbb{P}^n} \to \mathbb{P}^n$ 诱导 $\varepsilon^*: k(\mathbb{P}^n) \to k(\widetilde{\mathbb{P}^n})$，因为 $P$ 外的稠密开集上 $\varepsilon$ 是一一对应，函数域同构——所以 blow-up 不改变函数域，是**双有理**的（第 3 篇：$K(X) \cong K(Y)$）。

## 2 曲线穿过奇点时发生了什么

blow-up 的核心价值在于**分离**。设 $C \subseteq \mathbb{A}^2$ 是过原点 $O$ 的曲线，在 $O$ 处奇异（如尖点 $y^2 = x^3$，或两分支相交的结点 $y^2 = x^2$）。blow-up 后：

**核心概念：正常像（proper transform）**：$C$ 的**正常像** $\widetilde{C} \subseteq \widetilde{\mathbb{A}^2}$ 是 $\varepsilon^{-1}(C \setminus \{O\})$ 的闭包。<span class="marginnote">正常像 = "把 $O$ 挖掉、拉开后取闭包"：原本"挤在一点"的多个分支，在 $\varepsilon$ 下被"按方向"分开。尖点 $y^2 = x^3$ 的 blow-up 后正常像是<strong>光滑</strong>的（接触 $E$ 于一点）；结点 $y^2 = x^2$ 的正常像有两个分支，各自光滑地交于 $E$ 的两个点。</span>

![blow-up 示意图](/images/algebraic-geometry/birational-geometry-blowups-1.svg)

**重点：blow-up 解消简单奇点。** 对代数闭域上的尖点与结点，一次 blow-up 就使正常像光滑。一般奇点需要有限次 blow-up 序列（**解消定理**，Hironaka，1964）：任意特征 0 的代数簇上的奇点，可通过有限次 blow-up（沿奇异子簇）完全解除。<span class="marginnote">Hironaka 解消是 20 世纪代数几何最深的定理之一，曾获 1970 菲尔兹奖。它的"有限次"保证了奇点可以系统地修好；而"在奇点子簇上 blow-up"正是把本节的一点点小手术推向任意维度的总纲。</span>

**辨析｜易错点：** blow-up 不是"把曲线切两半"，而是"改变空间"：不是 $C$ 被修改，而是**环境空间**被修改（$O$ 换成 $E$），$C$ 作为子簇在新的环境里"重新落位"。所以 $C$ 的拓扑（亏格）在 blow-up 下不变——亏格是双有理不变量（第 13 篇：R-H 公式的分支结构由正常像决定，总亏格守恒）。**"改空间而不改对象"**是理解 blow-up 的第一句话。

## 3 除子、线性系与 blow-up 的关系

blow-up 与第 9 篇的线性系理论紧密相连：**"给曲线一个有理映射到射影空间"与"在基点处做 blow-up"互为表里。** 设 $|D|$ 是有基点的线性系，$\varphi_{|D|}: X \dashrightarrow \mathbb{P}^n$ 是有理映射（在基点处未定义）。做 blow-up $\varepsilon: \widetilde{X} \to X$（沿基点集），则 $\varphi_{|D|}$ 提升为**态射** $\widetilde{\varphi}: \widetilde{X} \to \mathbb{P}^n$。<span class="marginnote">这一步是双有理几何最重要的操作之一："有理映射 → 取 blow-up → 变成态射"。几乎所有的有理映射（典范映射、到射影空间的嵌入）都先经过"blow-up 掉基点"的预处理，然后成为真正的态射。这正是第 7 篇赋值判别准则"延拓存在"的构造性版本。</span>

**核心概念：blow-up 下的除子与典范类。** 若 $\varepsilon: \widetilde{X} \to X$ 是沿光滑点 $P$ 的 blow-up（$\dim X = n$，例外除子 $E \cong \mathbb{P}^{n-1}$），则 <span class="marginnote">这条公式是"blow-up 的账本"：除子被分成"正常像"与"例外部分"，而典范类被修改一次（加 $(n-1)E$）。计算 blow-up 后一切不变量（相交数、Euler 特征）都以它为起点。</span>

$$K_{\widetilde{X}} = \varepsilon^* K_X + (n-1) E$$

对曲面（$n = 2$）：$K_{\widetilde{X}} = \varepsilon^* K_X + E$。<span class="marginnote">$\mathbb{P}^2$ 在一点的 blow-up 记作 $\mathbb{F}_1$（第一类 Hirzebruch 曲面），它是研究曲面相交理论（第 15 篇）的最小模型。$E^2 = -1$ 的"自交 -1 曲线"是整个负曲线理论的起点。</span>

## 4 双有理几何的中心问题

有了 blow-up，双有理几何的核心问题清晰浮现：

**核心问题：** 每个双有理等价类里，怎样的"代表"最"小"或最"简单"？给定一个簇，能否通过有限次 blow-up / 收缩（blow-down）把它化成某种"极小模型"？

**核心概念：极小模型（minimal model）**：对曲面，极小模型 = 不含 $(-1)$-曲线（$E^2 = -1$ 的光滑有理曲线）的模型。**Castelnuovo 收缩定理**保证：任意曲面通过有限次"收缩 $(-1)$-曲线"化为极小曲面，且极小曲面在同构意义下唯一（除了有理曲面与某些例外）。<span class="marginnote">曲面的极小模型分类是经典代数几何的桂冠：正则化后曲面分为有理曲面、直纹面（ruled）、K3 曲面、Enriques 曲面、阿贝尔曲面、一般型…… 由 Kodaira 维数分类。现代<strong>极小模型纲领</strong>（MMP，Mori 1980s）把同一思路推向三维与一般情形：用"$K_X$ 的符号"决定收缩的方向。</span>

**重点：Kodaira 维数作为粗分类。** 令 $\kappa(X)$ = "$\{|mK_X|\}$ 的维数增长速率"。则按 $\kappa = -\infty, 0, 1, n$ 把簇分为：有理型 / Calabi-Yau 型 / 纤维化型 / 一般型。这是双有理几何对"簇的丰富程度"最粗的、也是最重要的分类。<span class="marginnote">$\kappa = -\infty$（如 $\mathbb{P}^n$：$|mK|$ 空）对应"负弯曲、几何上简单"；$\kappa = n$（一般型：$|mK|$ 给出到射影空间的嵌入）对应"正弯曲、几何上复杂"；$\kappa = 0$（K3、椭圆曲线：$K \sim 0$）是"临界情形"。这个"曲率符号"的分类与微分几何的常曲率分类精神相通。</span>

## 5 公式解析：blow-up 的典范类

$$
K_{\widetilde{X}} = \varepsilon^* K_X + (n-1) E, \qquad E \cong \mathbb{P}^{n-1}
$$

分三步拆解：

- **第一步，$\varepsilon^* K_X$ 是"从底下拉上来的典范"**：$K_X$ 在 $P$ 处之外无变化地"搬"上 $\widetilde{X}$。因为 $\varepsilon$ 在 $P$ 外一一对应，拉回在 $P$ 外与原来一致。<span class="marginnote">"拉回"把底空间的典范类直接复制到总空间；但总空间多出了例外方向 $E$ 的"额外弯曲"，必须补上 $E$ 项。</span>
- **第二步，$(n-1)E$ 是"例外方向的修正"**：$E \cong \mathbb{P}^{n-1}$ 的典范是 $-(n) H$（$\mathbb{P}^{n-1}$ 的典范 $= -n \cdot$ 超平面），但 $E$ 作为 $\widetilde{X}$ 的子簇，其法丛的"自交"要按 $(n-1)E$ 修正。对曲面：$E^2 = -1$ 时 $K_{\widetilde{X}} = \varepsilon^*K_X + E$，系数 1 恰使 $E \cdot K_{\widetilde{X}} = -1$（用伴随公式核对）。
- **第三步，为什么这不改变函数域**：$K_{\widetilde{X}}$ 与 $K_X$ 只差"$\mathbb{P}^{n-1}$ 上的某条除子"——在 $P$ 外逐点一致。于是 $\deg$、亏格等双有理不变量不受影响，而"自交数"这类依赖嵌入结构的量被精确修正。**blow-up 是双有理的（不动函数域），但非双有理不变的具体量（自交、相交数）要按公式重算**——这正是第 15 篇的入口。

一句话直觉：**blow-up = 把一点展开成方向空间 $\mathbb{P}^{n-1}$，典范类因此多出一层"例外弯曲" $(n-1)E$**；它修好奇点、不改函数域，是一切双有理操作的最小单元。

## 6 小结

- **Blow-up** $\varepsilon: \widetilde{X} \to X$：把点 $P$ 换成例外除子 $E \cong \mathbb{P}^{n-1}$；在 $P$ 外一一对应，是**双有理**态射。
- **正常像**：挖掉奇点、拉开后取闭包；简单奇点（尖点、结点）经一次 blow-up 变光滑（Hironaka 解消：有限次即可）。
- **易错**：blow-up 改的是**环境空间**，不是对象本身；亏格等双有理不变量不变。
- **与线性系**：有理映射的基点由 blow-up 消除，$\varphi_{|D|}$ 提升为态射。
- **典范类**：$K_{\widetilde{X}} = \varepsilon^* K_X + (n-1)E$；曲面情形 $K = \varepsilon^*K + E$，$E^2 = -1$。
- **极小模型**：Castelnuovo 收缩 $(-1)$-曲线得极小曲面；Kodaira 维数做粗分类。

在下一节，我们进入曲面的几何：**曲面的相交理论与 Riemann-Roch**——定义曲线在曲面上的相交数，并用 $\chi(\mathcal{O}(D))$ 的公式计算线丛截面。

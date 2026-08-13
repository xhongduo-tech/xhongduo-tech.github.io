---
title: 交换子与约化
date: 2026-08-07
---

# 交换子与约化

<div class="epigraph">
<p>数学的本质在于它的自由。</p>
<footer>—— 格奥尔格 · 康托尔（Georg Cantor）</footer>
</div>

<div class="article-byline">
<p>第二级 · 算子代数 ｜ Kadison & Ringrose《Fundamentals of the Theory of Operator Algebras》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从约化开始

上一节我们把「因子」立为 von Neumann 代数的原子，但只回答了一半：**一般 von Neumann 代数如何拆成因子？** 这一节回答另一半——**约化理论（reduction theory）**。它的核心工具是**中心分解**与**直接积分**：把一般 von Neumann 代数「化整为零」，沿着中心分解成一片片的因子，就像把一个函数沿着变量展开成积分。

约化理论同时是「交换子」概念的大成：$\mathcal{M}''$（双交换子）、$\mathcal{M}'$（交换子）、中心 $\mathcal{Z}=\mathcal{M}\cap\mathcal{M}'$ 三者互相缠绕，而约化理论证明：**von Neumann 代数的全部结构，都可以从「它的交换子代数 + 分解」读出来**。量子力学的超选择规则、表示的分解、以及第 23 篇类型分类的推广，全靠这一套技术。

## 1 中心投影：代数的「开关」

**中心投影（central projection）**：$z\in\mathcal{Z}(\mathcal{M})$ 且 $z^2=z=z^*$（即 $z$ 是投影且在中心里）。中心投影是「与一切可交换的开关」：$z\mathcal{M}$ 与 $(1-z)\mathcal{M}$ 是两个独立的 von Neumann 代数块。

**引理（中心的格子）**：$\mathcal{Z}(\mathcal{M})$ 作为可交换 von Neumann 代数 ≅ $L^\infty(\Lambda,\mu)$（$\Lambda$ 是某测度空间），其中的投影对应 $L^\infty$ 的特征函数——**中心投影 = 测度空间上的可测子集**。<span class="marginnote">「中心 ≅ $L^\infty(\Lambda)$」是第 21 篇交换 von Neumann 代数结构定理的现成应用。于是中心投影的格子 = 测度空间 $\Lambda$ 的「可测点集」格子；「在 $\lambda$ 处的约化」对应「在可测子集上的积分」。测度论的全套直觉在这里接管。</span>

**例（量子超选择）**：$\mathcal{M}$ 的交换子 $\mathcal{M}'$ 非平凡时，中心投影把 $\mathcal{H}$ 切成「超选择扇区」：$z_\alpha\mathcal{H}$ 是互不混合的物理子空间，不同扇区之间的相干叠加物理上不可见。约化理论把这些扇区按测度积分起来。

## 2 直接积分：把 Hilbert 空间「积分」起来

**可测场**：一族 Hilbert 空间 $\{\mathcal{H}_\lambda\}_{\lambda\in\Lambda}$ 配上「可测截面」结构（$\lambda\mapsto\xi_\lambda\in\mathcal{H}_\lambda$ 的可测性由可数生成集定义）。

**直接积分（direct integral）**：

$$\mathcal{H} = \int_\Lambda^\oplus \mathcal{H}_\lambda\, d\mu(\lambda),$$

其内积为 $\langle\xi,\eta\rangle=\int_\Lambda\langle\xi_\lambda,\eta_\lambda\rangle\,d\mu(\lambda)$（模去测度零集）。$\mathcal{H}$ 的元素是「可测截面」$\xi=(\xi_\lambda)$。

**可分解算子**：$T=\int^\oplus T_\lambda d\mu(\lambda)$ 若 $T\xi=(T_\lambda\xi_\lambda)$，且 $\|T\|=\mathrm{ess}\sup_\lambda\|T_\lambda\|$。<span class="marginnote">直接积分把「连续族的 Hilbert 空间」缝合成一个 Hilbert 空间，与第 19 篇张量积互为表里：张量积是「离散直和的连续版」之外的乘法结构，直接积分是「直和的连续化」。$L^2(\Lambda\times X)=\int^\oplus L^2(X)d\mu$ 是最熟悉的例子。</span>

**例**：$\mathcal{H}=L^2([0,1]\times\{0,1\})$ 可写成 $\int^\oplus_{[0,1]}\mathbb{C}^2\,d\lambda$；对角算子 $\int^\oplus A_\lambda d\lambda$ 对应「每个点上放一个 $2\times2$ 矩阵」。

## 3 约化定理：拆成因子

**定理（约化 / 分解定理）**：设 $\mathcal{M}\subset B(\mathcal{H})$ 是可分 Hilbert 空间上的 von Neumann 代数，$\mathcal{Z}(\mathcal{M})=L^\infty(\Lambda,\mu)$。则存在直接积分分解

$$\mathcal{H}=\int_\Lambda^\oplus\mathcal{H}_\lambda\,d\mu(\lambda), \qquad \mathcal{M}=\int_\Lambda^\oplus\mathcal{M}_\lambda\,d\mu(\lambda),$$

其中**几乎每个 $\mathcal{M}_\lambda$ 都是因子**（在 $\mathcal{H}_\lambda$ 上），且中心投影 $z(E)$（$E\subset\Lambda$ 可测）恰是「限制到 $E$」的投影。每个 $\mathcal{M}_\lambda$ 的类型（第 23 篇）是 $\lambda$ 的可测函数，从而 $\mathcal{M}$ 被分解成「一片片因子」。<span class="marginnote">约化定理是「测度论 × 算子代数」的顶点：von Neumann 代数作为「算子场」沿中心积分，几乎处处都是因子。它把第 23 篇的因子分类扩展到一般 von Neumann 代数：先中心分解，再逐点分类。Kaplansky 密度定理保证分解与逼近可以互换。</span>

**推论（类型分解）**：$\mathcal{M}$ 有唯一的中心投影 $z_\mathrm{I},z_{\mathrm{II}_1},z_{\mathrm{II}_\infty},z_{\mathrm{III}}$（互不正交、和为 1），使 $z_\mathrm{I}\mathcal{M}$ 是 I 型、$z_{\mathrm{II}_1}\mathcal{M}$ 是 II$_1$ 型…… **每个 von Neumann 代数都是「各类型块的直积分」**。

**辨析｜易错点：**约化定理要求 $\mathcal{H}$ **可分**（或 $\Lambda$ 适当的可分性条件）；不可分情形需要用「连续选择」处理，技术性陡增。另一个易错点：分解中 $\mathcal{M}_\lambda$ 是因子是「几乎处处」的——在测度零集上可以例外，而「几乎处处」在算子代数里意味着「不影响任何结构」。

## 4 公式解析：$\mathcal{M}=\int^\oplus\mathcal{M}_\lambda\,d\mu(\lambda)$

$$
\mathcal{M} = \int_\Lambda^\oplus \mathcal{M}_\lambda\, d\mu(\lambda), \qquad \mathcal{Z}(\mathcal{M})=L^\infty(\Lambda,\mu)
$$

- **第一步，看左端**：$\mathcal{M}$ 是一个「大」von Neumann 代数，元素是算子 $T$。
- **第二步，看右端**：$\mathcal{M}_\lambda$ 是一族因子。$\mathcal{M}$ 的元素 $T=(T_\lambda)$ 是一个「可测算子场」——每个 $\lambda$ 处放一个算子 $T_\lambda\in\mathcal{M}_\lambda$。$\|T\|=\mathrm{ess}\sup\|T_\lambda\|$，所以 $T$ 的范数由「各点的最大范数」几乎处处决定。
- **第三步，看中心如何驱动分解**：$\mathcal{Z}(\mathcal{M})=L^\infty(\Lambda,\mu)$ 的元素是「标量场」$f=(f_\lambda)$，即「在每个 $\mathcal{M}_\lambda$ 里都只乘标量 $f(\lambda)$」的算子。中心投影 $z(E)$ 就是「只在 $E$ 上为 1 的特征函数场」——分解的「点集」正是中心自己提供的。
- **第四步，看意义**：公式说「一般 von Neumann 代数 = 因子的积分」。于是第 23 篇的因子分类立即升级为「所有 von Neumann 代数的分类骨架」：**类型、迹、正常态全部逐点分解**。分解的唯一性（模测度零）保证分类不依赖分解方式——这是「中心分解」优于「任意分解」的根本原因。

## 5 约化与交换子的现代应用

**应用 1（表示的分解）**：群 $G$ 的表示 $\pi$（第 16 篇）在 $\mathcal{M}=\pi(G)''$ 上做中心分解，得到「$\pi$ 分解成不可约表示的直积分」——**Plancherel 分解**就是约化理论的经典形态。$L^2(G)$ 上的正则表示沿 $\widehat G$ 积分，重现调和分析的整个架构。<span class="marginnote">约化理论给「连续族表示」提供测度论语言：非交换调和分析的 Plancherel 公式、以及量子场论里「超选择扇区的直积分」，都是「$\mathcal{M}=\int^\oplus\mathcal{M}_\lambda$」的特例。不可约表示对应「点」$\lambda$，直积分把这些点连续地「积分」起来。</span>

**应用 2（超选择规则）**：物理可观测量代数 $\mathcal{A}$ 的中心投影对应「守恒荷的扇区」；约化分解说明「不同扇区的观察完全独立」——数学上严格化「超选择」概念，是量子场论与量子信息交叉处的常客。

**应用 3（类型分解与物理）**：给定一个态 $\varphi$，它的 GNS 表示 $\pi_\varphi$ 的 von Neumann 代数 $\pi_\varphi(A)''$ 的类型分解回答「该态属于哪种量子相」——II$_1$ 对应「有限温度平衡态」（有 KMS 态与迹），III 对应「真空态」（无迹）。**类型成为相变的数学指纹**。

**辨析｜易错点：**「约化」与「分解」不自动「保表示」：同一个抽象 $W^*$-代数可以有多个不同的直接积分实现，但**中心分解是唯一的**（模测度零），类型与迹由中心决定。初学者常以为「一个 von Neumann 代数 = 一个因子」，忘了它可能是因子的积分；约化理论的正确定位是「先因子、后积分」。

## 6 例：中心分解的直观

把中心分解在具体例子里「看」出来，抽象的直积分就落地了。

**$L^\infty([0,1])$（交换 von Neumann 代数）**：中心 = 自己。分解 $\Lambda=[0,1]$，每个纤维 $\mathcal{M}_\lambda=\mathbb{C}$（一维因子，I 型）。「交换代数 = 一维因子的积分」。

**$L^\infty([0,1])\otimes M_2$**：中心 = $L^\infty([0,1])$（张量第一因子）。分解 $\Lambda=[0,1]$，纤维 $\mathcal{M}_\lambda=M_2$（I$_2$ 因子）。「矩阵块连续族」。

**$\bigoplus_n M_n$（可数直和，非直积分）**：中心 = 对角标量（$c_0$ 型）。分解是离散的：每个纤维 $M_n$。「直和 = 离散直积分」。

**$C^*_r(G)''$（群代数生成的 von Neumann 代数）**：中心分解给出 $G$ 的「Plancherel 分解」——不可约表示按测度积分。I 型群（如 $\mathbb{R}$）分解成 I 型因子。

**II$_1$ 因子（如 $\mathcal{R}$）**：中心 = $\mathbb{C}1$，分解「只有一个纤维」——因子本身就是「原子」。中心分解对因子是平凡的。

**一句话总结**：中心分解 = 「沿中心积分成因子」——交换代数积分成一维因子，矩阵块连续族积分成矩阵因子，因子本身是单纤维。

## 7 延伸：直接积分与 Plancherel

直接积分最漂亮的出场，是非交换调和分析的 Plancherel 公式。

**正则表示的分解**：$L^2(G)$ 上的正则表示 $\lambda$ 沿 $\widehat G$（不可约表示的空间）分解为 $\int^\oplus \pi\,d\mu(\pi)$。$L^2(G)=\int^\oplus \mathcal{H}_\pi\,d\mu(\pi)$。

**Plancherel 定理**：对可均的 I 型群，存在 Plancherel 测度 $\mu$ 使 $L^2(G)\cong\int^\oplus\mathcal{H}_\pi d\mu(\pi)$，且 $f\in L^1\cap L^2$ 有 $\|f\|_2^2=\int\|f(\pi)\|_{\mathrm{HS}}^2 d\mu(\pi)$——「Fourier 变换保范」。

**$G=\mathbb{R}$**：$\widehat{\mathbb{R}}=\mathbb{R}$，Plancherel 测度是 Lebesgue 测度——经典 Plancherel 公式 $\int|f|^2=\int|\widehat f|^2$。

**$G$ 交换**：$\widehat G$ 是 Pontryagin 对偶，直接积分退化为「一维表示积分」= 通常 Fourier 分析。

**为什么需要约化理论**：非交换群的不可约表示「连续地」出现，必须用直积分而不是直和。约化理论是「连续族的表示」的测度论语言。

**一句话总结**：Plancherel 公式 = 约化理论在群表示上的具体化——「$L^2(G)$ 分解成不可约表示的直积分」是调和分析的终极形态。

## 8 延伸：约化理论的边界

约化理论威力无穷，但它的边界值得看清。

**可分性假设**：约化定理要求 $\mathcal{H}$ 可分（或适当条件）。不可分情形（如 $B(\mathcal{H})$ 的某些大子代数）需要「连续选择」与更精细的测度论，技术难度陡增。

**类型的逐点分解**：$\mathcal{M}_\lambda$ 的类型几乎处处确定，但「几乎处处」意味着测度零集上可「失控」。类型函数 $f(\lambda)$ 的可测性是约化理论的核心技术。

**不可分表示的直积分**：并非每个表示都能「干净地」分解成不可约表示（III 型因子没有不可约的「原子」）。约化到因子已经是最佳：因子未必有不可约子表示。

**中心分解唯一**：中心分解在「模测度零」意义下唯一；一般分解（非中心）不唯一。分类时只用中心分解，正是为了唯一性。

**一句话总结**：约化理论在可分世界给出漂亮的「因子积分」；跨过可分性，测度论的精细技术登场，但「先因子、后积分」的策略不变。

## 9 小结

- **中心投影**：与一切可交换的开关；$\mathcal{Z}(\mathcal{M})\cong L^\infty(\Lambda,\mu)$，中心投影 = 可测子集。
- **直接积分** $\int^\oplus\mathcal{H}_\lambda d\mu(\lambda)$：连续族 Hilbert 空间的缝合；可分解算子逐点作用。
- **约化定理**：$\mathcal{M}=\int^\oplus\mathcal{M}_\lambda d\mu(\lambda)$，几乎每个 $\mathcal{M}_\lambda$ 是因子，分解沿中心唯一。
- **类型分解**：$z_\mathrm{I}+z_{\mathrm{II}_1}+z_{\mathrm{II}_\infty}+z_{\mathrm{III}}=1$
---
title: 特征值联合密度与 Vandermonde 行列式
date: 2026-08-07
---

# 特征值联合密度与 Vandermonde 行列式

<div class="epigraph">
<p>大多数行列为零，但行列式永远记得它的根。</p>
<footer>—— 佚名（Vandermonde 行列式在数学中的注脚）</footer>
</div>

<div class="article-byline">
<p>第四级 · 随机矩阵理论 ｜ Mehta, Random Matrices (3rd ed.) Ch. 3；AGZ §2.6 ｜ 2026-08-07</p>
</div>

## 为什么从特征值联合密度开始

高斯系综的密度 $p(H) \propto e^{-\frac{\beta}{2}\operatorname{tr}(H^2)}$ 是定义在「矩阵空间」上的，可物理观测对象是特征值。要从矩阵语言切换到特征值语言，就必须对特征向量做积分——而这一步的 Jacobian 因子，正是随机矩阵理论最著名的结构：**Vandermonde 行列式**。特征值联合密度因此获得一个优美而暴力的形式：平方势能 + 两两排斥。这一讲是这个专题的分水岭：前面讲的都是「矩阵」，从这一讲开始，一切都发生在「特征值」上，而特征值联合密度则是后续所有理论（行列式点过程、Tracy–Widom、可积系统）的公共起点。

## 1 换元：从矩阵元到特征值

实对称矩阵 $H$ 的对角化写作 $H = U \Lambda U^{-1}$，其中 $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_N)$ 是特征值对角阵，$U$ 是正交（酉/辛）矩阵，特征向量的集合。**做变量替换 $(H_{ij}) \mapsto (\lambda_i, U)$，要小心：这不是等距替换，度量会变形。** 对 $H$ 的每个矩阵元微元做体积元分析，得到

$$dH = J(\lambda) \, d\lambda \, dU$$

其中 $dH$ 是矩阵空间的标准体积元，$dU$ 是正交群上的 Haar 测度，而 Jacobian 为

$$J(\lambda) = \prod_{1 \le i \lt  j \le N} |\lambda_i - \lambda_j|^{\beta}$$

这里 $\beta$ 正是 Dyson 参数，而 $d\lambda = d\lambda_1 \cdots d\lambda_N$。<span class="marginnote">Jacobian 的推导靠「无穷小转动的位移」：若特征值 $\lambda_i$ 与 $\lambda_j$ 靠近，那么一个微小的特征向量扰动就会造成 $\mathcal{O}(|\lambda_i - \lambda_j|^{-1})$ 的矩阵元变化，因此矩阵体积元在特征值简并处被拉伸。$\beta$ 的幂次来自「每个特征对共享多少个实自由度方向」。</span>

**重点：当两个特征值相等时，Jacobian 为 0。** 这意味着特征值「重复」的概率测度为 0，即特征值几乎必然互不相同。这是随机矩阵理论的一条基本事实：**谱几乎必然无重数**，特征值排斥在密度层面就已经写死了。

## 2 Vandermonde 行列式登场

幂次的差乘积有一个人尽皆知的行列式表达——**Vandermonde 行列式**：

$$\Delta(\lambda) = \prod_{1 \le i \lt  j \le N} (\lambda_j - \lambda_i) = \det \bigl[ \lambda_i^{\, j-1} \bigr]_{i,j=1}^{N} = \begin{vmatrix} 1 & \lambda_1 & \cdots & \lambda_1^{N-1} \\ 1 & \lambda_2 & \cdots & \lambda_2^{N-1} \\ \vdots & \vdots & & \vdots \\ 1 & \lambda_N & \cdots & \lambda_N^{N-1} \end{vmatrix}$$

第二式是定义，第三式是「用元素幂构造的行列式」——两式相等的证明是经典技巧：把 $\prod(\lambda_j - \lambda_i)$ 展开，每一行提出 $\lambda_i$ 的最低次幂，就得到纯 Vandermonde。<span class="marginnote">Vandermonde 行列式是「最简交替多项式」：交换任意两列它变号，因此当 $\lambda_i = \lambda_j$ 时它必须为 0。它是多重积分理论里少有的「既优美又可显式计算」的对象，19 世纪的代数不变量理论早已把它研究透。</span>

因此 Jacobian 写成 $J(\lambda) = |\Delta(\lambda)|^\beta$，而特征值联合密度变为

$$p(\lambda_1, \dots, \lambda_N) = \frac{1}{Z'_{N,\beta}} \, |\Delta(\lambda)|^\beta \, e^{-\frac{\beta}{2}\sum_{i=1}^N \lambda_i^2}$$

对特征向量部分积分后，矩阵密度只剩特征值的函数。这个式子就是随机矩阵理论里**特征值联合密度**的规范形式，几乎所有精确计算都从它出发。

## 3 公式解析：密度拆成三块

$$p(\lambda_1, \dots, \lambda_N) = \underbrace{\frac{1}{Z'_{N,\beta}}}_{\text{归一化}} \cdot \underbrace{\prod_{i<j}|\lambda_i - \lambda_j|^\beta}_{\text{排斥项}} \cdot \underbrace{e^{-\frac{\beta}{2}\sum_i \lambda_i^2}}_{\text{势能项}}$$

- **势能项 $e^{-\frac{\beta}{2}\sum_i \lambda_i^2}$**：来自矩阵密度的平方范数。它把每个特征值拉向 0，强度由 $\beta$ 控制——这是「外部势」，像把 $N$ 个粒子约束在一维抛物线阱里。
- **排斥项 $\prod_{i<j}|\lambda_i - \lambda_j|^\beta$**：来自 Jacobian，是「粒子与粒子之间」的相互作用。$\lambda_i \to \lambda_j$ 时它趋于 0，所以两个特征值越靠近，联合密度越小——**排斥**。$\beta$ 越大，排斥越强。
- **归一化 $Z'_{N,\beta}$**：保证积分为 1。它的精确值用正交多项式（对高斯势是 Hermite 多项式）计算，见 Mehta 第 3 章。

把它读成统计力学的语言：**$N$ 个带电荷的一维粒子，在抛物线外势阱里互相排斥**。这正是「库仑气体」或「对数气体」模型：排斥项取对数就是 $\exp(\beta \sum_{i<j} \log|\lambda_i - \lambda_j|)$，恰是二维库仑相互作用的对数势。<span class="marginnote">对数气体的视角非常强大：随机矩阵的特征值分布只是「对数势外场下的带电粒子平衡态」的特例。第 6 篇 Marchenko–Pastur 律、第 7 篇自由概率，都能在这个框架下获得统一的直觉。</span>

**重点：密度只依赖特征值，不依赖特征向量。** 特征向量被积分掉了，且它们的分布是「独立的 Haar 均匀」。这意味着任何「只与特征值有关」的统计量，都完全由上式决定；而任何「与特征向量有关」的统计量则另有一套独立的理论（如第 12 篇里随机投影的高维统计应用）。

## 4 特例：$\beta = 2$ 的福星——行列式结构

GUE（$\beta = 2$）的联合密度有独一无二的简化：$|\Delta|^2 = \Delta^2$ 是行列式的**平方**，而 $\Delta^2$ 本身可写成行列式之积。对任意函数 $f$ 构造矩阵 $[f(\lambda_i - \lambda_j)]$，我们有经典恒等式

$$\Delta(\lambda)^2 \prod_i f(\lambda_i) = \det\bigl[ K_N(\lambda_i, \lambda_j) \bigr]$$

其中 $K_N(x, y)$ 是由正交多项式（Hermite 多项式）给出的**核**。换句话说，$\beta = 2$ 时，特征值联合密度是一个**行列式**——这直接引出下一讲的「行列式点过程」：$N$ 个点 $(\lambda_1, \dots, \lambda_N)$ 的联合密度由某个 $N \times N$ 行列式给出时，这个点过程的一切 $k$ 点关联函数都能写成更小的行列式。<span class="marginnote">为什么只有 $\beta = 2$ 有行列式结构？因为 $|\Delta|^\beta$ 只有在 $\beta = 2$ 时才是「行列式的绝对值平方」，而平方正好让「和式可写成行列式乘积」的代数成立。$\beta = 1, 4$ 需要更复杂的 Pfaffian 结构，难度陡增。</span>

因此 GUE 是「可精确计算」的模范生：从联合密度到两点关联函数，都能显式写出行列式公式。GOE、GSE 则需要 Pfaffian（反对称行列式）技巧——这正是为什么理论文献里 GUE 结果最多、最漂亮。

## 5 从密度到关联函数

有了联合密度，一切特征值统计量都是它的积分。定义 **$k$-点关联函数**

$$R_k(x_1, \dots, x_k) = \frac{N!}{(N-k)!} \int p(\lambda_1, \dots, \lambda_N) \, d\lambda_{k+1} \cdots d\lambda_N$$

它是「在 $x_1, \dots, x_k$ 附近各找到一个特征值」的密度，单位是「个」而非概率。$R_1(x)$ 就是特征值密度（对 GUE 收敛到半圆）；$R_2(x,y)$ 则刻画两特征值的相关性。对 $\beta = 2$ 有决定性结果：**核 $K_N$ 一旦算出来，$R_k$ 就是它的 $k$ 阶行列式**

$$R_k(x_1, \dots, x_k) = \det \bigl[ K_N(x_i, x_j) \bigr]_{i,j=1}^{k}$$

这是第 4 讲全部内容的核心入口。它能直接给出 $R_2$：$R_2(x,y) = K_N(x,x)K_N(y,y) - K_N(x,y)^2$，第二项正是「排斥」的量化——若 $x = y$，$R_2 = 0$。<span class="marginnote">对 $x=y$ 时 $R_2$ 为零的另一种解读：同一位置出现两个特征值的密度为 0，这从密度层面对应了上一讲的「间距统计在 $s \to 0$ 时以 $s^\beta$ 消失」。排斥无处不在：先于概率出现在密度里，再显形于关联函数中。</span>

## 6 术语速查表与联合密度算例

| 术语 | 记号 / 公式 | 一句话含义 |
| --- | --- | --- |
| 对角化 | $H = U\Lambda U^{-1}$ | 特征值 + 特征向量分解 |
| 体积元 | $dH = J(\lambda)\,d\lambda\,dU$ | 换元时度量变形 |
| Jacobian | $\prod_{i<j}|\lambda_i-\lambda_j|^\beta$ | 特征值简并处体积元被拉伸 |
| Vandermonde 行列式 | $\Delta(\lambda)=\det[\lambda_i^{j-1}]=\prod_{i<j}(\lambda_j-\lambda_i)$ | 差乘积的行列式表达 |
| 特征值联合密度 | $p(\lambda) \propto |\Delta|^\beta e^{-\frac{\beta}{2}\sum\lambda_i^2}$ | 抛物线外势 + 对数排斥 |
| 对数气体 | $\exp(\beta\sum_{i<j}\log|\lambda_i-\lambda_j|)$ | 排斥项取对数后的形式 |
| Haar 测度 | $dU$ | 正交/酉群上的均匀概率 |
| $k$-点关联函数 | $R_k(x_1,\dots,x_k)$ | 同时找到 $k$ 个特征值的密度 |
| 核 | $K_N(x,y)=\sum_{k<N}\varphi_k(x)\varphi_k(y)$ | GUE 关联函数的一切编码者 |
| Pfaffian | $\operatorname{Pf}(A)$，反对称矩阵的行列式平方根 | $\beta=1,4$ 的替代工具 |

**算例：GUE（$\beta=2$）$N=2$ 的联合密度**。$2\times2$ GUE 特征值 $\lambda_1,\lambda_2$ 的联合密度是 $p(\lambda_1,\lambda_2) = \frac{1}{Z_2}(\lambda_1-\lambda_2)^2 e^{-(\lambda_1^2+\lambda_2^2)}$。

- **Hermite 核**：$N=2$ 时 $K_2(x,y)=\varphi_0(x)\varphi_0(y)+\varphi_1(x)\varphi_1(y)$，其中 $\varphi_0(x)=\pi^{-1/4}e^{-x^2/2}$、$\varphi_1(x)=\sqrt2\,\pi^{-1/4}x\,e^{-x^2/2}$。
- **行列式**：$\det\begin{pmatrix}K_2(\lambda_1,\lambda_1)&K_2(\lambda_1,\lambda_2)\\K_2(\lambda_2,\lambda_1)&K_2(\lambda_2,\lambda_2)\end{pmatrix}$ 展开后，含 $\varphi_0^2\varphi_1^2$、$\varphi_1^2\varphi_0^2$ 与交叉项。
- **化简**：利用 $\int\varphi_0^2=1$、$\int\varphi_1^2=1$、$\int\varphi_0\varphi_1=0$ 的正交性，上式化为 $(\lambda_1-\lambda_2)^2 e^{-(\lambda_1^2+\lambda_2^2)}/Z_2$——**$|\Delta|^2$ 变成了行列式**。

**为什么 $\beta = 1, 4$ 需要 Pfaffian**：$|\Delta|^\beta$ 只有在 $\beta=2$ 时是「行列式的模方」，才能套用 Cauchy–Binet 把差乘积平方写成核的行列式。$\beta=1,4$ 时 $|\Delta|$ 或 $|\Delta|^4$ 不是行列式的平方，取而代之的是**反对称核的 Pfaffian**：$p(\lambda)\propto\operatorname{Pf}[\tilde K(\lambda_i,\lambda_j)]$。Pfaffian 是「反对称矩阵行列式的平方根」，它保留行列式的组合便利，却要处理额外的符号与约束——这正是文献里 GUE 结果最多、GOE/GSE 次之的原因。

**应用预告**：联合密度一旦写出行列式，$k$-点关联函数、谱边缘、间距分布全部有迹可循。第 4 篇把它抽象成行列式点过程，第 5 篇用它的边缘极限得到 Tracy–Widom——「联合密度 → 行列式 → 普适对象」这条链路，是本专题前五篇的暗线。

**关于归一化常数的一句说明**：$Z'_{N,\beta}$ 的显式值由 Hermite 多项式给出（$\beta=2$ 时为 $\prod_{k=0}^{N-1} k! \cdot (2\pi)^{N/2}$ 量级）。它看似是技术细节，却控制着一切「计数型」问题——如「某区间内特征值个数的期望」——所以不可省略。

**一句话收束**：Vandermonde 行列式是随机矩阵从「矩阵世界」进入「特征值世界」的换乘站：向左它是 Jacobian，向右它是行列式核，而两边的工具（组合、正交多项式、行列式）都由此汇入同一条轨道。

**辨析｜易错点：** $|\Delta(\lambda)|^\beta$ 的绝对值不能随手丢掉。$\beta = 2$ 时 $|\Delta|^2 = \Delta^2$ 恰好是行列式的平方，才能套用 Cauchy–Binet 把差乘积平方写成核的行列式；若把 $\beta = 1$ 或 $4$ 的密度直接当「行列式」处理，会得到错误的关联函数——这正是 Pfaffian 结构出场的根本原因。

## 7 小结

- **Jacobian**：矩阵元替换到特征值时的体积元变形因子是 $\prod_{i<j}|\lambda_i - \lambda_j|^\beta$，特征向量部分积分后消失。
- **Vandermonde 行列式**：$\Delta(\lambda) = \det[\lambda_i^{j-1}] = \prod_{i<j}(\lambda_j - \lambda_i)$，它把排斥项写成行列式。
- **联合密度**：$p(\lambda) \propto |\Delta(\lambda)|^\beta e^{-\frac{\beta}{2}\sum \lambda_i^2}$——抛物线外势 + 对数排斥。
- **对数气体图像**：特征值像带同号电荷的粒子，在一维抛物线阱中平衡。
- **$\beta = 2$ 特例**：联合密度是行列式，$k$-点关联函数 $R_k = \det[K(x_i, x_j)]$，通向行列式点过程。

在下一节，我们将把「联合密度是行列式」这个 $\beta=2$ 的特例抽象成一般理论：行列式点过程——它把排斥、可计算、普适三件事焊成一体。
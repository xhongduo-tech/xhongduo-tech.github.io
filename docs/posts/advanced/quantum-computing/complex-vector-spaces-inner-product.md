---
title: 复数域上的向量空间与内积
date: 2026-08-07
---

# 复数域上的向量空间与内积

<div class="epigraph">
<p>虚数是上帝精神的奇妙避难所，几乎是存在与不存在之间的两栖之物。</p>
<footer>—— 戈特弗里德 · 威廉 · 莱布尼茨（Gottfried Wilhelm Leibniz），论虚数</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从复向量空间开始

前六篇反复出现一句话：**量子态是向量**。$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ 是向量，$n$ 个量子比特的叠加 $\sum_x c_x|x\rangle$ 也是向量，量子门的本质是「矩阵乘向量」。但直到现在，我们都没认真回答一个基础问题：这些向量住在怎样的空间里？答案从第二级《线性代数》学过的「实数域上的向量空间」升级一步——**复数域上的向量空间（complex vector space）**。

这一篇是整个量子计算技术部分的**地基中的地基**。Dirac 记号、算符、测量、纠缠，全部建立在这套语言之上。把这一篇学透，后面所有公式都会「自己会读」。

## 1 从实数到复数：为什么必须是 ℂ

线性代数的课本通常从**实数域 $\mathbb{R}$ 上的向量空间**讲起。量子力学需要的是**复数域 $\mathbb{C}$ 上的向量空间**——把「标量」从实数换成复数。这不是口味问题，而是量子力学的结构要求。<span class="marginnote">复数第一次进入数学，是为了「造出负数的平方根」——见第一级《基础数学》的复数一节。在量子力学里，复数不再只是方程的解，而是承载「相位」这个物理量的天然容器：$i$ 的引入让「相位」有了代数操作对象。</span>

为什么实向量空间不够用？因为量子力学的状态不只是「幅度」决定，还有**相位（phase）**。考虑两个叠加态：

$$
|\psi_1\rangle = \frac{|0\rangle + |1\rangle}{\sqrt2}, \qquad
|\psi_2\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt2}
$$

对 $|0\rangle$、$|1\rangle$ 分别做测量，两个态的「测量概率」完全相同——$|0\rangle$ 都是 $\frac12$。但它们是**不同的物理状态**：一旦让它们与第三个态发生干涉，$i$ 引入的 $90°$ 相位差会带来可观测的干涉条纹差异。<span class="marginnote">这个「相位看得见、概率看不见」的现象，正是量子计算里「干涉」的数学根源：第零篇《量子计算为什么可能更快》里说「干涉让相位正确的分支相长、错误的相消」，翻译成代数就是：系数是复数，它们的相对相位决定叠加结果。经典概率论里只有非负实数，装不下相位，所以经典信息论描述不了量子干涉。</span>

> 辨析｜易错点： 量子态系数 $\alpha$ 的**整体相位（global phase）**不可观测——$|\psi\rangle$ 与 $e^{i\theta}|\psi\rangle$ 是同一个物理状态；但**相对相位（relative phase）**完全可观测——$|0\rangle+|1\rangle$ 与 $|0\rangle+i|1\rangle$ 是不同的物理状态。判断「相位有没有物理意义」，就看它是不是「整体乘上去的」。

## 2 向量空间的定义与复数的四则运算

**向量空间（vector space）**：一个集合 $V$，配有两种运算——加法 $V\times V\to V$ 与数乘 $\mathbb{C}\times V\to V$，满足八条公理（加法交换律、结合律、零元、负元；数乘结合律、单位元、对加法的分配律、对域加法的分配律）。<span class="marginnote">八条公理的完整陈述在第二级《线性代数》第一章。这里要强调的是：向量空间的「标量域」从 $\mathbb{R}$ 换成 $\mathbb{C}$ 后，八条公理一条都不用改——因为 $\mathbb{C}$ 本身就是一个域，满足域的全部条件。这正是「抽象代数」里「域」概念的威力：整套线性代数理论只依赖「标量来自一个域」，至于这个域是 $\mathbb{R}$、$\mathbb{C}$ 还是有理数域，理论照样成立。</span>

最基本的复向量空间是 **$n$ 维复坐标空间**：

$$
\mathbb{C}^n = \{(z_1, z_2, \dots, z_n) \mid z_i \in \mathbb{C}\}
$$

加法与数乘都是逐分量的。$n=1$ 时就是复平面本身。量子计算里最常出现的两个空间是 $\mathbb{C}^2$（单量子比特状态空间）与 $\mathbb{C}^{2^n}$（$n$ 量子比特状态空间）——后者维数随 $n$ 指数增长，这正是第零篇《量子计算的起源》里「状态空间装不下」的数学表述。

## 3 基、线性无关与维数

**线性无关（linear independence）**：一组向量 $\{|v_1\rangle,\dots,|v_m\rangle\}$，若 $\sum_i a_i|v_i\rangle = 0$ 当且仅当所有 $a_i=0$，则它们线性无关。<span class="marginnote">判断标准：任何一个向量都不能被其余向量线性表示。这保证了「表示唯一」——如果基向量线性相关，同一个向量会有无穷多种系数写法，一切计算都会失稳。</span>

**基（basis）**：一组既能**张成**整个空间、又**线性无关**的向量。空间的**维数（dimension）**就是基的个数。$\mathbb{C}^n$ 有标准基 $e_1=(1,0,\dots,0),\dots,e_n=(0,\dots,1)$。

量子计算里两个关键维数：

- 单量子比特：$\dim \mathbb{C}^2 = 2$，标准基是 $\{|0\rangle, |1\rangle\}$。
- $n$ 量子比特：$\dim \mathbb{C}^{2^n} = 2^n$，标准基是全部 $n$ 位二进制串 $\{|x\rangle : x\in\{0,1\}^n\}$。

**任何线性无关的向量组都可以扩充为一组基**，同一个空间有无穷多种基的选择。后面我们会反复「换基」——比如把计算基换成 $|+\rangle = (|0\rangle+|1\rangle)/\sqrt2$、$|-\rangle = (|0\rangle-|1\rangle)/\sqrt2$——换基不改变空间，只改变坐标，这是线性代数的基本观点。

## 4 内积：给向量一个「夹角」

有了向量空间还远远不够。量子计算需要**长度**（概率要归一化）与**正交性**（基要互不混淆），这两者都来自**内积（inner product）**。

**内积**：$\mathbb{C}^n$ 上的一个函数 $\langle \cdot, \cdot\rangle : V\times V \to \mathbb{C}$，满足三条公理：

1. **对第二个分量的线性性**：$\langle v, \alpha w_1 + \beta w_2\rangle = \alpha\langle v,w_1\rangle + \beta\langle v,w_2\rangle$；
2. **共轭对称性**：$\langle v,w\rangle = \langle w,v\rangle^*$；
3. **正定性**：$\langle v,v\rangle \ge 0$，且 $\langle v,v\rangle = 0 \iff v = 0$。

物理学约定下，内积**对第二个分量（ket）线性，对第一个分量（bra）共轭线性**——这个约定与 Dirac 记号严丝合缝，我们下一节就讲。

$\mathbb{C}^n$ 上的标准内积是：

$$
\langle v, w\rangle = \sum_{i=1}^{n} v_i^*\, w_i
$$

其中 $^*$ 表示复共轭。<span class="marginnote">为什么第一个分量要取共轭？看正定性：$\langle v,v\rangle = \sum_i v_i^* v_i = \sum_i |v_i|^2 \ge 0$。如果不取共轭，$\langle v,v\rangle = \sum_i v_i^2$ 会是一个复数，就谈不上「长度」了。共轭是让「自己跟自己内积」变成非负实数的关键一笔。</span>

由内积可以诱导出**范数（norm）**：

$$
\|v\| = \sqrt{\langle v, v\rangle} = \sqrt{\sum_i |v_i|^2}
$$

以及**正交性（orthogonality）**：$\langle v,w\rangle = 0$ 时称 $v$ 与 $w$ 正交。一组两两正交且范数都为 1 的基叫**标准正交基（orthonormal basis）**——$\{|0\rangle,|1\rangle\}$ 和 $\{|+\rangle,|-\rangle\}$ 都是。

## 5 公式解析：内积、柯西–施瓦茨不等式与归一化

**内积是量子概率的代数翻译机**：Born 规则「测量概率 = 振幅模方」的一切，都藏在 $\langle v,v\rangle = \sum_i|v_i|^2$ 这一条式子里。下面把最关键的一条不等式拆开。

**柯西–施瓦茨不等式（Cauchy–Schwarz inequality）**：

$$
|\langle v, w\rangle|^2 \;\le\; \langle v, v\rangle\,\langle w, w\rangle
$$

等号成立当且仅当 $v$ 与 $w$ 线性相关。它是内积空间里「两个向量夹角不能小于 0 度」的精确表达，也是量子信息里很多下界（比如第零篇提过的 Holevo 界、纠缠熵的性质）的源头。证明拆三步：

**第一步，正交化**：设 $v \neq 0$，取 $\lambda = \dfrac{\langle v,w\rangle}{\langle v,v\rangle}$，令

$$
w_\perp = w - \lambda v
$$

**第二步，验证正交与勾股**：直接算 $\langle v, w_\perp\rangle = \langle v,w\rangle - \lambda\langle v,v\rangle = 0$，所以 $v \perp w_\perp$。于是由内积的正定性：

$$
\langle w,w\rangle = \langle \lambda v + w_\perp,\ \lambda v + w_\perp\rangle
= |\lambda|^2\langle v,v\rangle + \langle w_\perp, w_\perp\rangle
\ge |\lambda|^2\langle v,v\rangle
$$

**第三步，代回 $\lambda$**：把 $|\lambda|^2 = \dfrac{|\langle v,w\rangle|^2}{\langle v,v\rangle^2}$ 代入，两边乘 $\langle v,v\rangle$，得到

$$
|\langle v,w\rangle|^2 \le \langle v,v\rangle\,\langle w,w\rangle
$$

证毕。<span class="marginnote">这一步「正交分解 + 勾股」是内积空间的核心技巧：任何一个向量 $w$ 都能拆成「沿着 $v$ 的分量」加「垂直于 $v$ 的分量」。这个几何直觉在后面学投影测量、以及 Grover 算法把状态「旋转」到目标子空间时，会反复出现。</span>

**把不等式用到量子态上**：量子态是归一化向量 $\|v\|=\|w\|=1$，于是柯西–施瓦茨退化为 $|\langle v,w\rangle| \le 1$——这正是「两个量子态的内积模长不超过 1」的由来，也是概率 $|\langle\phi|\psi\rangle|^2 \le 1$ 的代数保证。**量子力学的一切概率，都在内积空间这个「尺度」之内。**

## 6 小结

- **量子态是复数域上的向量**：$\mathbb{C}$ 承载相位，相对相位可观测、整体相位不可观测。
- **$n$ 量子比特的状态空间是 $\mathbb{C}^{2^n}$**，维数随 $n$ 指数增长；标准正交基是全部 $n$ 位二进制串。
- **内积**对第二个分量线性、第一个分量共轭线性，标准形式 $\langle v,w\rangle = \sum_i v_i^* w_i$；共轭保证正定性。
- **范数** $\|v\| = \sqrt{\langle v,v\rangle}$，**正交**由 $\langle v,w\rangle=0$ 定义；标准正交基是量子计算的默认坐标系。
- **柯西–施瓦茨不等式** $|\langle v,w\rangle|^2 \le \langle v,v\rangle\langle w,w\rangle$ 是内积空间一切「长度与夹角」性质的地基，量子态的归一化让 $|\langle v,w\rangle|\le1$。

在下一节，我们将给这套向量空间配上量子力学专用的记号——**Dirac 记号（bra-ket）**：右矢 $\langle v|$、左矢 $|v\rangle$ 与它们的内积、外积，让「向量运算」变成顺手的心算。

---
title: Lagrangian Floer 同调与障碍理论
date: 2026-08-07
---

# Lagrangian Floer 同调与障碍理论

<div class="epigraph">
<p>两个 Lagrangian 子流形之间的交点，是辛几何最敏感的信使；Floer 同调把它们数成同调。</p>
<footer>—— 深谷贤治（Kenji Fukaya）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从 Lagrangian Floer 同调开始

上一篇的 Floer 同调数哈密顿同胚的不动点。现在把不动点换成**两个 Lagrangian 子流形的交点**——这是辛几何里更基本的配对（温斯坦那句「量子力学是关于 Lagrangian 相交的理论」在这里兑现）。**Lagrangian Floer 同调**把「数交点」变成「算同调」，并揭示一个深刻的障碍现象：**一般情况下这个同调可能根本不存在**（障碍类非零），除非用「bounding cochain」修补——这就是障碍理论（Fukaya-Oh-Ohta-Ono）。修补的副产品是一个惊人的代数结构：**Fukaya 范畴**（$A_\infty$ 范畴），它把 Lagrangian 子流形变成「对象」、交点变成「态射」、全纯圆盘变成「运算」。这一篇讲定义、障碍、以及通向镜面对称的道路。<span class="marginnote">在课程地图上：这一篇是第4篇的心脏——它上承 Floer 同调，下启 Gromov-Witten 不变量与镜面对称。Fukaya 范畴是「辛几何的代数化」，也是下一篇量子上同调与末篇镜面对称的必备语言。</span>

## 1 交点问题与 Lagrangian 相交

设 $(M^{2n}, \omega)$ 辛流形，$L_0, L_1$ 是两个**横向相交**的 Lagrangian 子流形（在每一交点 $T_pL_0 \oplus T_pL_1 = T_pM$，即切空间横截）。<span class="marginnote">横截性保证交点<strong>离散</strong>。一般位置下，横向 Lagrangian 交点是有限多个——因为 $2n = n + n$ 维子空间横截相交维数为 0。交点个数是「几何数据」，但会随同伦/哈密顿形变而变；Floer 同调想找出<strong>同伦不变</strong>的交点计数。</span>

**Arnold-Givental 猜想**：对 Lagrangian $L$，在一般位置下

$$
\#(L \cap \phi(L)) \ge \sum_k \dim H^k(L)
$$

（$\phi$ 是哈密顿同胚）。这是「不动点猜想」的 Lagrangian 版本，同样需要 Floer 同调证明。

**目标**：定义 $HF(L_0, L_1)$，使其「基」由交点生成、同调类只依赖 $L_0, L_1$ 的哈密顿同痕类——于是「交点数的下界」来自「$HF$ 的贝蒂数」。

## 2 Floer 复形与全纯圆盘

**链复形**：

$$
CF(L_0, L_1) = \bigoplus_{p \in L_0 \cap L_1} \mathbb{Z}_2 \langle p \rangle
$$

生成元是交点，系数 $\mathbb{Z}_2$（回避定向技术）。**微分** $\partial$ 计数「带 Lagrangian 边界条件的全纯圆盘」：

$\partial(p) = \sum_q n(p,q) q$，其中 $n(p,q)$ 是满足下列条件的解 $u: D^2 \to M$ 的个数（模 2）：

- $\bar\partial_J u = 0$（$u$ 是 $J$-全纯）；
- $u(\text{上边界}) \subset L_0$，$u(\text{下边界}) \subset L_1$（Lagrangian 边界条件）；
- $u$ 的「角」在 $p$ 与 $q$（圆盘边界上的两个角点贴在两个交点上）。

**直观**：交点 $p$ 与 $q$ 之间「被一个全纯圆盘连接」，就贡献一条微分边。<span class="marginnote">这正是「伪全纯曲线」理论在 Lagrangian 配对的用法：曲线不是闭的（$S^2$），而是带边界（$D^2$），边界切在 Lagrangian 子流形上。这种「带 Lagrangian 边界的全纯圆盘」是 Fukaya 范畴的态射与 $A_\infty$ 运算的几何原料。</span>

**Maslov 指标**：交点 $p$ 的「Floer 指标」（类似 Morse 指标）由 Maslov 指标给出——沿全纯圆盘边界的 $T_pL$ 沿 $J$ 转动圈数。**指标差 1 的轨迹才贡献微分项**，这与 Morse/Floer 理论完全平行。

## 3 障碍：$\partial^2 \neq 0$ 的时刻

Floer 理论的关键是证明 $\partial^2 = 0$。对 Lagrangian 情形，$\partial^2(p)$ 计数「两个圆盘相接的边界配置」，它有两种贡献：

**可分解配置**：两个圆盘沿交点相接——这种按「模空间边界」应互相抵消（Floer 的 Gromov 紧致性论证）；
**球面/圆盘冒泡**：能量收缩成「带边界的小圆盘」或「闭球面」——**这种不自动抵消**。

当「带边界的全纯圆盘」出现（$L$ 不是单调的、或 Maslov 类不够大），$\partial^2(p)$ 可能非零，Floer 同调**不定义**。<span class="marginnote">对比上一篇（哈密顿 Floer）：那里没有边界条件，唯一的冒泡是球面冒泡，在单调/Calabi-Yau 情形可控。Lagrangian 情形多出「边界冒泡」——$L$ 上自带的全纯圆盘——这是障碍的真正来源。</span>

**障碍类（obstruction class）**：$\partial^2 \neq 0$ 对应一个非零的「障碍」$m_0 \in HF^*(L, L)$（单位圆盘计数）。**若 $m_0 = 0$（如 $L$ 是单连通、或单调且 Maslov 指标足够大），Floer 同调良定义。**

**例（$L = S^1 \subset \mathbb{R}^2$）**：$S^1$ 是 Lagrangian（$\mathbb{R}^2$ 中 1 维迷向闭曲线）。$\pi_2(M, L) \neq 0$：存在全纯圆盘 $D^2 \to \mathbb{C}$ 把边界映到 $S^1$（内部填满单位圆盘）。于是 $S^1$ 有非零障碍 $m_0$——**$HF(S^1, S^1)$ 在平凡意义上不定义**，需要 bounding cochain 修补。<span class="marginnote">$S^1 \subset \mathbb{C}$ 是障碍理论最干净的例：单位圆盘本身就是一个「连接 $S^1$ 到 $S^1$ 的 $J$-全纯圆盘」（边界切在 $S^1$ 上），所以 $m_0 = 1$（非零）。修补后 $HF$ 仍可定义——这就是 bounding cochain 的用武之地。</span>

## 4 公式解析：Floer 微分的配置计数

**核心公式（$\partial^2$ 的配置分解）：**

$$
\partial^2(p) = \sum_{\text{可分解}} (\text{相互抵消}) + \sum_{\text{冒泡}} m_0(L) \cdot (\text{相关项})
$$

拆解：

- **第一步，看 $\partial^2(p)$ 的几何**：$\partial^2(p)$ 计数「$p \to q \to r$ 两步」或「$p$ 到 $r$ 直接」的配置。Gromov 紧致性说「两步」配置的模空间边界与「直接」配置的模空间内部配对，给出链复形意义下的相消。
- **第二步，发现边界冒泡**：若模空间的紧化边界额外出现「一个独立的全纯圆盘 + 一个缩小的曲线」，这一项不在可分解配置里——它对应「$L$ 上自带的圆盘」。
- **第三步，障碍类出场**：边界冒泡项的系数正是 $m_0(L)$（单位圆盘计数）。**$m_0 \neq 0$ 时 $\partial^2 \neq 0$**，链复形失效。
- **第四步，修补（bounding cochain）**：选一个「边界链」$b$（偶次相交链）使「扭曲微分」$\partial_b = \partial + m_b$ 满足 $\partial_b^2 = 0$（其中 $m_b$ 由 $b$ 加权的圆盘计数）。若这样的 $b$ 存在，$L$ 是**可障碍消解（obstructed but bounding）**，$HF$ 仍可定义，只是复形带扭转。

**直觉总结：** 障碍理论说「Floer 同调不是自动存在的」——它的存在本身是一个几何条件（$m_0 = 0$ 或可被 $b$ 修补）。**这正是辛几何「局部平凡、整体刚性」主题的最深体现**：同调的结构由「整体圆盘计数」决定。

## 5 Fukaya 范畴与 $A_\infty$ 结构

把障碍理论升维，就得到 **Fukaya 范畴**：

- **对象**：$M$ 的 Lagrangian 子流形（带额外结构：grading、bounding cochain）；
- **态射**：$\mathrm{Hom}(L_0, L_1) = CF(L_0, L_1)$（交点链群）；
- **组合**：$m_k$ 运算（$k \ge 0$）由「带 $k+2$ 个 Lagrangian 边界的全纯圆盘」计数，满足 **$A_\infty$ 关系**：

$$
\sum_{i+j=k} (\pm) m_{k-j+1}(x_1, \dots, x_i, m_j(x_{i+1}, \dots), \dots) = 0
$$

这是「结合律的无穷维修正」——**Fukaya 范畴是 $A_\infty$ 范畴**，$m_1 = \partial$ 是微分，$m_2$ 是（非结合的）合成。<span class="marginnote">$A_\infty$ 结构（Stasheff）是「结合代数」的同伦推广：合成只结合到同伦（$m_3$）为止，$m_3$ 又满足更高阶关系。Fukaya 范畴的 $A_\infty$ 结构是镜面对称（末篇）的代数核心——它编码了 $M$ 的全部「带边界全纯曲线」信息。</span>

**分次与可乘性**：Lagrangian 需带 **grading**（提升 Maslov 类到 $\mathbb{Z}$）与相对 spin 结构，Fukaya 范畴才可分次、可加。

**用途**：
- **Split-generation**：某 Lagrangian 族是否「生成」Fukaya 范畴（对应镜面对称里的对象覆盖）。
- **位移能量下界**：$HF(L, \phi(L)) \neq 0$ 说明 $L$ 不能被「小能量」哈密顿同胚移开——给出位移能量的谱下界，与 Hofer 几何（上一篇）衔接。

**从同调到几何的反馈**：Fukaya 范畴不仅「记录」几何，还反过来**约束**几何。若两个 Lagrangian 在 Fukaya 范畴里同构（如 $L \cong L'$），则它们的位移能量、相交性质全部一致——**范畴等价是辛几何的「同伦等价」**。当代结果（Abouzaid、Auroux 等）用「Fukaya 范畴的生成元」证明位移能量下界与嵌入障碍，把「代数工具」变成「几何定理」——这正是第3篇 Hofer 几何与第4篇障碍理论合流的体现。

**例：圆盘的 Lagrangian 分类**。二维圆盘 $D^2$（带面积形式）里的 Lagrangian 是嵌入曲线（圆弧）。Fukaya 范畴在这里退化成「带边全纯圆盘的模空间」，而障碍理论给出「哪些圆弧可被 bounding cochain 修补」——**最简单的二维情形已经能看到障碍与修补的全部机制**，是进入一般理论的最佳练兵场。

**为什么系数选 $\mathbb{Z}_2$**：Floer 微分计数「模空间的模 2 个数」。用 $\mathbb{Z}_2$ 回避了「定向」问题——定向（以及相对 pin/spin 结构）是定义整数系数的前置条件，而 $\mathbb{Z}_2$ 永远可行。**「先用 $\mathbb{Z}_2$，再升级到 $\mathbb{Z}$」是 Floer 理论的标准路线**：障碍理论、定向理论都先在 $\mathbb{Z}_2$ 上建立，再在额外结构下强化。

## 6 小结

- **Lagrangian 交点 Floer 同调**：链复形由横向交点生成，微分数带边界全纯圆盘。
- **Arnold-Givental 猜想**：交点下界 = 贝蒂数和；由 $HF \cong H^*(L)$（良定义时）证明。
- **障碍类 $m_0$**：$L$ 上的全纯圆盘使 $\partial^2 \neq 0$；$S^1 \subset \mathbb{C}$ 是最简单反例。
- **Bounding cochain**：扭曲微分 $\partial_b$ 修补障碍；可修补的 $L$ 仍可定义 $HF$。
- **Fukaya 范畴**：对象 = Lagrangian，态射 = 交点，$m_k$ 运算来自带边全纯圆盘，满足 $A_\infty$
---
title: Gromov-Witten 不变量与量子上同调
date: 2026-08-07
---

# Gromov-Witten 不变量与量子上同调

<div class="epigraph">
<p>量子上同调是给上同调环装上曲线计数器——乘法不再是拓扑的，而是量子化的。</p>
<footer>—— 爱德华 · 威滕（Edward Witten, 1991）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第11-12章 ｜ 2026-08-07</p>
</div>

## 为什么从 Gromov-Witten 不变量开始

第3篇的伪全纯曲线理论证明了「曲线存在」，但还差一步：**数曲线**。Gromov-Witten（GW）不变量就是「数 $J$-全纯曲线」的精确答案——对固定的同调类、固定的曲线亏格、以及固定数量的「标记点约束」，曲线有多少条？这个「数」是辛不变量，也是代数几何里「枚举几何」的辛推广。而把 GW 不变量装进上同调环，就得到**量子上同调**：一个「量子变形」的乘法，使 $\mathbb{CP}^n$ 的上同调环带上「球面曲线计数」的修正项。这一篇讲 GW 不变量的定义（模空间、虚拟基本类）、量子上同调与例子（$\mathbb{CP}^n$），并预告它通向镜面对称（末篇）的桥梁作用。<span class="marginnote">在课程地图上：GW 不变量是「闭曲线计数」，与上一篇「带边界曲线计数」（Fukaya 范畴）互为镜像；量子上同调是「代数化」的最终形态，也是镜面对称与几何表示论（末两篇）的直接输入。</span>

## 1 枚举几何与 GW 不变量

**古典枚举问题**：过 $k$ 个给定点、在给定同调类里的曲线有几条？代数几何里，五次曲面上过 5 个点的有理曲线有 2875 条（1870 年 Chasles 类结果）——这类「数曲线」问题在代数几何里有悠久传统。GW 不变量把它们推广到**辛流形**。

**GW 不变量**：固定 $(M^{2n}, \omega)$、相容 $J$、同调类 $A \in H_2(M;\mathbb{Z})$、亏格 $g$、标记点数 $k$。考虑模空间

$$
\mathcal{M}_{g,k}(M, A) = \{ (u, z_1, \dots, z_k) : u \text{ $J$-全纯}, [u] = A \} / \text{同构}
$$

**求值映射（evaluation）**：$ev_i: \mathcal{M} \to M$，$(u, z) \mapsto u(z_i)$。对 $M$ 的上同调类 $\alpha_1, \dots, \alpha_k$（「约束」），GW 不变量定义为

$$
\mathrm{GW}_{g,k}^{A}(\alpha_1, \dots, \alpha_k) = \int_{[\mathcal{M}_{g,k}(M,A)]^{\mathrm{vir}}} ev_1^*\alpha_1 \wedge \cdots \wedge ev_k^*\alpha_k
$$

其中 $[\mathcal{M}]^{\mathrm{vir}}$ 是**虚拟基本类**（virtual fundamental class）——当模空间不光滑（有障碍）时，用障碍理论（李-提安/弗洛尔泛函分析）构造的「正确的同调类」，使积分有定义。<span class="marginnote">虚拟基本类是 GW 理论的技手术语：对「一般 $J$」模空间光滑，积分就是普通积分；但一般 $J$ 未必可得到，虚拟基本类用「局部障碍束」的正规锥构造，保证不变量定义良好且不依赖 $J$。这是 Witten 猜想与 Gromov 理论的严格化（1990 年代完成）。</span>

**GW 不变量的辛不变性**：GW 不变量不依赖 $J$（相容同伦下不变）、依赖 $(M, \omega)$ 的辛形变类——**它是辛不变量**，且在代数簇情形与代数几何里的 GW 不变量一致（虚拟基本类兼容）。

## 2 量子上同调环

**量子上同调（quantum cohomology）**：作为向量空间 $QH^*(M) = H^*(M) \otimes \Lambda$（$\Lambda$ 是 Novikov 环：形式幂级数 $\sum a_A q^A$，$q$ 的指数是 $A \in H_2$）。关键是**量子积（quantum product）**：对 $\alpha, \beta \in H^*(M)$，

$$
\alpha * \beta = \sum_{A \in H_2(M)} \mathrm{GW}_{0,3}^{A}(\alpha, \beta, \cdot) \, q^A
$$

更完整：$\alpha * \beta$ 是 $H^*(M)$ 中「由 GW 三点点数对偶」定义的元素：

$$
\langle \alpha * \beta, \gamma \rangle = \sum_A \mathrm{GW}_{0,3}^A(\alpha, \beta, \gamma) \, q^A
$$

**物理直觉**：普通上同调乘法「$\alpha \cup \beta$」只数拓扑相交；量子积额外数「$A$ 类全纯曲线穿过三个约束」的贡献，用 $q^A$ 加权——**曲线计数修正乘法**。<span class="marginnote">对「从极限到大模型」的读者：量子积就像「注意力」——两个类相乘时不仅看它们本身的拓扑位置，还看它们之间「有没有曲线联系」（权重 $q^A$）。普通乘法是 $q=0$ 的退化，量子乘法把「联系数」编码进代数结构。</span>

**定理（量子上同调是环）**：量子积 $*$ 是结合的、分次交换的，且 $\gamma = c_1(M)$ 的「量子维数」使 $QH^*(M)$ 成为分次环——**$QH^*(M)$ 是 $H^*(M)$ 的量子形变**。结合律对应 GW 不变量的「分裂/退化」关系（WDVV 方程）。

## 3 例：$\mathbb{CP}^n$ 的量子上同调

$\mathbb{CP}^n$：$H^*(\mathbb{CP}^n) = \mathbb{R}[x]/(x^{n+1})$，$x = c_1(\mathcal{O}(1))$（超平面类）。量子上同调是

$$
QH^*(\mathbb{CP}^n) = \mathbb{R}[x, q] / (x^{n+1} - q)
$$

**关键计算**：$\mathrm{GW}_{0,3}^A(x, x, x)$ 对 $A = [\mathbb{CP}^1]$（直线类）：过两个点恰好一条直线，且它与第三个超平面相交一次，故 $\mathrm{GW}_{0,3}^{[L]}(x, x, x) = 1$。于是

$$
x * x * x = x^3 + q = q
$$

（$x^3$ 是普通乘法项 $x^{n+1}$ 当 $n=2$，对 $n=2$：$x^3 = 0$ 平凡，$x*x*x = q \cdot 1$）。对一般 $n$：$x^{*n+1} = q$——**「$n+1$ 条曲线约束」量子化后不再为零，等于 $q$**。<span class="marginnote">这就是「量子变形」的精髓：普通乘法里 $x^{n+1} = 0$（超平面类自乘 $n+1$ 次为零），量子乘法里 $x^{*(n+1)} = q$（因为存在「过 $n+1$ 个一般位置点的直线」——$n$ 维 $\mathbb{CP}^n$ 上 $n+1$ 个点确定一条直线）。曲线计数把「平凡为零」变成「非零 = 曲线数」。</span>

**$S^2$ 与 $S^2\times S^2$**：$QH^*(S^2) = \mathbb{R}[x,q]/(x^2 - q)$（$x$ 是面积类）；$QH^*(S^2\times S^2)$ 由两个面积类 $A, B$ 生成，带关系 $A^2 = B^2 = 0$、$A * B = 1 + q$（$q$ 计数「对角线类」曲线）。

## 4 公式解析：量子积

**核心公式：**

$$
\langle \alpha * \beta, \gamma \rangle = \sum_{A \in H_2(M;\mathbb{Z})} \mathrm{GW}_{0,3}^A(\alpha, \beta, \gamma) \, q^A
$$

拆解：

- **第一步，配对**：$\langle \cdot, \cdot \rangle$ 是 $H^*(M)$ 与 $H_*(M)$ 的配对（或 Poincaré 对偶后的上同调配对）。左边 $\alpha*\beta$ 与 $\gamma$ 配对，把「量子积的结果」测出来。
- **第二步，求和**：右边对所有同调类 $A$ 求和，每项 = 「$A$ 类曲线穿过 $\alpha, \beta, \gamma$ 三个约束的次数」乘 $q^A$。$A = 0$ 时曲线是常值映射，$\mathrm{GW}_{0,3}^0(\alpha,\beta,\gamma) = \langle\alpha\cup\beta,\gamma\rangle$（普通乘法）——**$q^0$ 项还原普通乘法**。
- **第三步，$q^A$ 的权重**：$q^A = \exp(-\int_A \omega)$ 型（Novikov 环），让「大能量曲线」贡献指数级小。物理里 $q^A \sim e^{-A/\hbar}$ 是瞬子权重。
- **第四步，结合性来源**：$( \alpha*\beta)*\gamma = \alpha*(\beta*\gamma)$ 对应 GW 不变量的**退化分裂公式**（4 点 → 3 点的边界分解）——几何结合律。

**直觉总结：** 量子积 = 普通乘法 + 所有「曲线中间态」的加权修正。**上同调环由此从「纯拓扑」升级为「拓扑 + 曲线枚举」**，GW 不变量是其中的连接系数。

## 5 GW 与 Floer 的对话：Piunikhin-Salamon-Schwarz

GW 不变量与 Floer 同调不是孤立的——**PSS 同构**（Piunikhin-Salamon-Schwarz）说：对哈密顿 Floer 同调与量子上同调，

$$
HF^*(H) \cong QH^*(M)
$$

作为环同构（带量子积的环结构）。这统一了两条线：**哈密顿周期轨道的同调 = 曲线计数的量子环**。<span class="marginnote">PSS 同构是现代辛拓扑的基石之一：它让「动力系统不变量」（Floer 同调）与「几何不变量」（GW/量子上同调）互相翻译。Arnol'd 猜想的「$HF \cong H^*$」只是它不带量子修正的退化版本。</span>

**应用**：
- **Arnol'd 猜想的量子版**：用 $QH^*(M)$ 的谱给出不动点的**量子下界**。
- **嵌入障碍**：量子上同调的非平凡结构（如「量子 Steenrod 幂」）给出嵌入/填充障碍。
- **镜面对称**：$QH^*(X)$ 与对偶侧（B-模）的「量子环」对应，是末篇镜面对称的代数入口。

**辨析｜易错点：** GW 不变量**依赖同调类 $A$ 与亏格 $g$**——不同 $A$、不同 $g$ 是不同不变量；「GW 不变量」是整族。另外量子上同调的系数环（Novikov 环）里 $q$ 是形式变量，**不是**普朗克常数——$q^A$ 的指数是 $-\int_A\omega$，物理上对应 $e^{-2\pi A/\hbar}$ 但数学上 $q$ 是纯形式的。初学者把 $q$ 当普通变量会错——它是「同调类权重」。

## 6 小结

- **GW 不变量**：数亏格 $g$、类 $A$、$k$ 个约束下的 $J$-全纯曲线；用虚拟基本类定义，是辛不变量。
- **枚举几何推广**：代数几何里「数曲线」问题（五次曲面 2875 条直线）的辛版本。
- **量子上同调**：$QH^* = H^* \otimes \Lambda$，量子积 $*$ 由 GW 三点函数定义；$q^0$ 项还原普通乘法。
- **$\mathbb{CP}^n$**：$QH^*(\mathbb{CP}^n) = \mathbb{R}[x,q]/(x^{n+1} - q)$——量子修正是「曲线计数」。
- **PSS 同构**：$HF \cong QH$
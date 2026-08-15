---
title: moment map 与辛约化
date: 2026-08-07
---

# moment map 与辛约化

<div class="epigraph">
<p>对称性不是被『打破』，而是被『约化』：约化之后，留下的流形依旧辛。</p>
<footer>—— 维克多 · 金兹堡 精神续写（辛约化教学传统）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ Cannas 第10章；McDuff & Salamon 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从 moment map 开始

可积系统告诉我们：一组守恒量把相空间纤维化成环面。现在问一般问题：如果辛流形上有一个**李群作用**保持辛结构，能不能用「守恒量」把流形「除以对称」得到一个新的、更小的辛流形？答案藏在 **moment map（动量映射）** 里——它是「作用在辛几何中的生成元」，把李群元素映射成哈密顿函数。**辛约化（Marsden-Weinstein-Meyer reduction）** 则告诉你：守恒量的公共水平集模掉群作用，仍是一个辛流形。这是辛几何最有力的「降维手术」：它解释经典力学的角动量守恒如何减少自由度，也是后面环面作用、Delzant 分类、乃至量子化的核心机制。<span class="marginnote">「moment」一词借自力学：对旋转群作用，moment map 就是角动量。对平移群，它是动量。名字起得恰如其分——moment map 是「动量」的几何抽象。</span>

## 1 辛群作用与无穷小生成元

设紧李群 $G$ 光滑作用于 $(M, \omega)$。若每个 $g \in G$ 的作用 $g: M \to M$ 都是辛同胚，称 $G$ 的作用为**辛作用（symplectic action）**。

对李代数元素 $\xi \in \mathfrak{g}$，诱导**无穷小生成元（fundamental vector field）**

$$
\xi_M(p) := \left. \frac{d}{dt} \right|_{t=0} \exp(t\xi) \cdot p
$$

$\xi_M$ 是 $M$ 上的向量场，其流正是 $\exp(t\xi)$ 的作用。<span class="marginnote">直觉：$\xi$ 是「群里的无穷小方向」，$\xi_M$ 是它「推」到流形上的向量场。对旋转作用，$\xi = \text{绕某轴的角度增量}$ 对应 $\xi_M = $ 绕该轴的旋转向量场。</span>

**无穷小哈密顿作用**：对每个 $\xi$，$\xi_M$ 是哈密顿向量场（存在 $H_\xi$ 使 $\xi_M = X_{H_\xi}$）。线性性要求 $H_{\xi}$ 对 $\xi$ 依赖「相容」，引出 moment map。

## 2 moment map 的定义

**moment map（动量映射）**：设 $G$ 辛作用于 $(M, \omega)$。映射

$$
\mu: M \longrightarrow \mathfrak{g}^*
$$

称为 moment map，若对所有 $\xi \in \mathfrak{g}$，

$$
d\langle \mu, \xi \rangle = \iota_{\xi_M} \omega
$$

其中 $\langle \mu, \xi \rangle$ 是 $\mu$ 与 $\xi$ 的配对（$M$ 上的函数）。等价地，**$\xi_M$ 是关于函数 $\langle\mu,\xi\rangle$ 的哈密顿向量场**：$\xi_M = X_{\langle\mu,\xi\rangle}$。<span class="marginnote">比较：哈密顿向量场的定义是 $\iota_{X_H}\omega = dH$。moment map 把这条推广到整个李代数：对每个 $\xi$，$\iota_{\xi_M}\omega = d\langle\mu,\xi\rangle$。所以 moment map 是「无穷小生成元的哈密顿函数族」的打包。</span>

**哈密顿作用（Hamiltonian action）**：存在 moment map 的辛作用。$G$-等变性（$\mu(g\cdot p) = \mathrm{Ad}^*_g\mu(p)$）通常作为额外要求加入。

**例1（$S^1$ 在 $\mathbb{C}$ 上旋转）**：$M = \mathbb{C} = \mathbb{R}^2$，$\omega_0 = dx\wedge dy$，$S^1$ 以 $e^{i\theta} \cdot z = e^{i\theta}z$ 作用。李代数 $\mathfrak{g} = \mathbb{R}$，无穷小生成元 $\xi_M = \partial_\theta$（旋转场），moment map 是

$$
\mu(z) = \frac{|z|^2}{2} = \frac{x^2 + y^2}{2}
$$

因为 $\iota_{\partial_\theta}\omega_0 = \frac{1}{2}d(x^2+y^2)$。<span class="marginnote">$\mu(z) = |z|^2/2$ 是「到原点的距离平方的一半」——即旋转对称的守恒量（角动量）。这验证了 moment map 是「守恒量的生成器」。</span>

**例2（$T^n$ 在 $\mathbb{C}^n$ 上）**：$T^n = S^1 \times \cdots \times S^1$ 按 $(\theta_j) \cdot (z_1,\dots,z_n) = (e^{i\theta_1}z_1, \dots)$ 作用，moment map 是

$$
\mu(z_1, \dots, z_n) = \left( \frac{|z_1|^2}{2}, \dots, \frac{|z_n|^2}{2} \right) \in \mathbb{R}^n
$$

这是后面环面作用与 Delzant 分类的原始例子。

## 3 辛约化定理

**Marsden-Weinstein-Meyer 约化定理**：设 $G$ 紧李群**自由**且**哈密顿**作用于 $(M^{2n}, \omega)$，moment map $\mu$。若 $0 \in \mathfrak{g}^*$ 是 $\mu$ 的正则值，则

1. $Z = \mu^{-1}(0)$ 是 $M$ 的余维 $\dim G$ 光滑子流形；
2. $G$ 在 $Z$ 上的作用自由，商空间 $M_{\mathrm{red}} = Z / G = \mu^{-1}(0)/G$ 是光滑流形；
3. $M_{\mathrm{red}}$ 上有唯一辛形式 $\omega_{\mathrm{red}}$，满足 $\pi^*\omega_{\mathrm{red}} = i^*\omega$（$\pi: Z \to M_{\mathrm{red}}$ 是商映射，$i: Z \to M$ 是包含）；
4. $\dim M_{\mathrm{red}} = 2n - 2\dim G$。

**为什么这样降维？** 直觉：$Z$ 砍掉 $\dim G$ 个「动量约束」，商掉 $G$ 又砍掉 $\dim G$ 个「对称方向」。总降维 $2\dim G$，奇数个维度被成对消去，于是保持偶数维——这保证了约化流形还能装下辛形式。<span class="marginnote">约化流形的辛形式「投影」自原流形：$\omega_{\mathrm{red}}([p]) (d\pi(u), d\pi(v)) := \omega_p(u, v)$。良定义性需要验证「沿纤维方向 $\omega$ 为零」——这正是 $Z$ 上 $G$ 的作用是「各向同性」的：纤维切向量与 $Z$ 上任何向量辛配对为零。</span>

**关键验证（纤维是各向同性的）**：$Z = \mu^{-1}(0)$ 的切空间是 $\ker d\mu$。而 $d\langle\mu,\xi\rangle = \iota_{\xi_M}\omega$ 显示 $\xi_M(p)$ 与 $Z$ 相切（在正则值处），且 $\omega(\xi_M, v) = d\langle\mu,\xi\rangle(v) = 0$ 对 $v \in TZ$（因为 $\mu$ 在 $Z$ 上常数）。**所以纤维方向是迷向的，商有良定义辛形式。**

**例（$S^1$ 约化 $\mathbb{C}$）**：$\mu^{-1}(0) = \{0\}$，$M_{\mathrm{red}}$ 是单点，辛形式零。$\dim M_{\mathrm{red}} = 2 - 2 = 0$ ✓。这个平凡的约化是「把所有旋转对称除去后剩一个点」——对应角动量为零的系统只有一个平衡点。

**更一般的约化（Level $c$）**：约化到 $\mu^{-1}(c)/G_c$（$G_c$ 是稳定子群）也是辛流形，且约化到不同水平面的流形可以「辛形变」相连。**辛约化不是单个流形，而是一个族**。

## 4 公式解析：moment map 的核心等式

**核心公式：**

$$
d\langle \mu, \xi \rangle = \iota_{\xi_M} \omega
$$

四步拆解：

- **第一步，读两边类型**：左边是函数 $\langle\mu,\xi\rangle$ 的微分（1-形式）；右边是 2-形式 $\omega$ 与向量场 $\xi_M$ 的内乘（1-形式）。类型匹配——等式可以谈。
- **第二步，与哈密顿向量场联系**：$\iota_{\xi_M}\omega = dH_\xi$ 意味着 $\xi_M = X_{H_\xi}$，即 $\xi_M$ 是**哈密顿**向量场，哈密顿量 $H_\xi = \langle\mu,\xi\rangle$。所以 moment map 就是把「李代数方向」$\xi$ 对应到「哈密顿函数」$\langle\mu,\xi\rangle$——**群作用的方向就是某个哈密顿量的梯度方向**。
- **第三步，对 $\xi$ 线性**：$\xi \mapsto \langle\mu(p),\xi\rangle$ 对每个 $p$ 是 $\mathfrak{g}$ 上的线性泛函，所以 $\mu(p) \in \mathfrak{g}^*$。这正是「moment map 取值对偶李代数」的原因。
- **第四步，物理含义**：当 $G = SO(3)$ 作用于相空间，$\xi$ 是绕某轴的无穷小旋转，$\langle\mu,\xi\rangle$ 是该轴的角动量分量。**moment map 在 $\xi$ 方向的分量 = 沿 $\xi$ 的守恒量**。角动量守恒（$\mu$ 沿哈密顿流不变）是 Noether 定理的 moment map 表述。

**直觉总结：** moment map 是「把群作用编码成一族守恒函数」的机器。约化定理反其道而行：**用守恒函数族反过来构造更小的辛流形**。二者合起来是「对称性 ↔ 守恒量 ↔ 降维」三位一体。

## 5 与可积系统的联系：Lagrangian 纤维化

可积系统（上一篇）其实是环面作用的 moment map 纤维化：$F = (F_1,\dots,F_n)$ 是 $T^n$ 作用（由 $X_{F_i}$ 生成）的 moment map，其水平集 $M_c$ 是环面。**Liouville-Arnold 的环面正是 moment map 的纤维。**

约化把这些统一：可积系统是「$n$ 维环面作用的 moment map 纤维化」；约化到水平集 $M_c$ 是「把整个纤维化压缩成一个点」。**Delzant 分类（下一篇）正是给「环面作用的约化」这整类流形做拓扑分类。**

**辨析｜易错点：** 不是每个辛群作用都是哈密顿作用。若 $\iota_{\xi_M}\omega$ 是闭但非精确的 1-形式（即 $H^1(M)\neq 0$ 造成的障碍），则该方向没有全局哈密顿量，moment map 只能局部定义或根本不存在。典型例子：环面 $T^2$ 上的平移作用——moment map 存在但需要「选基点」，是所谓「多重 moment map」。**moment map 的存在性是整体问题**，这正是 flux 同态（上一篇）的余音。

**moment map 的物理与代数两面**：从哈密顿力学看，moment map 是「守恒量的打包」；从表示论看（末篇），它是「辛约化构造表示」的入口。**同一对象的两副面孔**——这是辛几何语言力量的标志：一个概念同时服务动力学、几何与代数，因为它们共享同一个辛结构。

## 6 小结

- **辛群作用与无穷小生成元** $\xi_M$：李代数方向推成的向量场。
- **moment map** $\mu: M \to \mathfrak{g}^*$：$d\langle\mu,\xi\rangle = \iota_{\xi_M}\omega$，把群作用编码成哈密顿函数族；例子 $S^1$ 旋转的 $\mu = |z|^2/2$。
- **辛约化（MWM）**：$\mu^{-1}(0)/G$ 是辛流形，维数 $2n - 2\dim G$；辛形式投影自原流形。
- **例**：$S^1$ 旋转 $\mathbb{C}$ 约化到单点；$T^n$ 在 $\mathbb{C}^n$ 上的 moment map 是「Delzant 分类」的原料。
- **与可积系统的统一**：Liouville-Arnold 的环面 = 环面作用的 moment map 纤维；约化把纤维压缩成点。
- **存在性整体性**：moment map 的存在依赖 $H^1$ 障碍（flux 余音），不是每个辛作用都哈密顿。

在下一节，我们将给「环面作用的约化」这整类流形做分类：**环面作用与 Delzant 分类**——辛环面流形完全由它的动量像（一个 Delzant 多胞形）决定。moment map 的像在这里从「守恒量」升级为流形的「指纹」。
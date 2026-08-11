---
title: Artin 互反律
date: 2026-08-11
---

# Artin 互反律

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert，Wir müssen wissen, wir werden wissen）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Artin 互反律开始

把局部类域论「沿每个素数拼装」，得到的是代数数论最富盛名的定理之一——**Artin 互反律**。它回答一个看起来吓人、实际极自然的问题：数域 $K$ 的阿贝尔扩张 $L/K$，它的 Galois 群究竟由什么数据完全决定？Artin 的答案是：由 $K$ 的「广义理想类群」（idèle 类群）的某个商群完全决定。这条定理一举统一了高斯二次互反律、Dirichlet 算术级数定理、Kronecker-Weber 定理等一串伟大成果，也被 Langlands 纲领视为一切现代互反律的原型。<span class="marginnote">「互反（reciprocity）」这个词源自高斯：$(\frac{p}{q})$ 与 $(\frac{q}{p})$ 的对称性。Artin 把这种「两个素数之间的对称」提升为「整个数域与其阿贝尔扩张之间的对称」——互反律从一条公式变成一门理论。</span>

## 1 二次互反律：最小的互反

高斯在 1796 年左右证明了**二次互反律**：对奇素数 $p \ne q$，

$$
\left(\frac{p}{q}\right)\left(\frac{q}{p}\right) = (-1)^{\frac{p-1}{2}\cdot\frac{q-1}{2}}
$$

勒让德符号 $(\frac{a}{p})$ 回答「$a$ 是不是模 $p$ 的平方」。<span class="marginnote">互反律的美在「非对称输入的对称结论」：$p$ 在 $q$ 一侧的平方性，由 $q$ 在 $p$ 一侧的平方性配平，只差一个符号。高斯自己给出了 6 个证明，称它是「算术的珍宝」——而它最终只是 Artin 互反律的一格投影。</span>

为什么二次互反律是「互反」？因为勒让德符号可以看成**分圆域里的 Artin 符号**。设 $\zeta_p$ 为本原 $p$ 次单位根，$K = \mathbb{Q}(\zeta_p)$。素理想 $\mathfrak{p} \mid q$ 的 **Frobenius 自同构** $\sigma_{\mathfrak{p}} \in \mathrm{Gal}(\mathbb{Q}(\zeta_p)/\mathbb{Q})$ 由下式决定

$$
\sigma_{\mathfrak{p}}(\zeta_p) = \zeta_p^{q}
$$

于是「$q$ 在 $p$ 一侧的平方性」$(\frac{q}{p})$ 恰好记录了这个 Frobenius 是否平凡——而 Artin 互反律把这个观察推广到一切阿贝尔扩张。

## 2 Artin 符号与 Frobenius

设 $L/K$ 是**有限阿贝尔**扩张，$\mathfrak{p}$ 是 $K$ 中不分歧的素理想，$\mathfrak{P} \mid \mathfrak{p}$。**Frobenius 自同构** $(\mathfrak{p}, L/K) \in \mathrm{Gal}(L/K)$ 是剩余类域上 $x \mapsto x^{\mathrm{N}(\mathfrak{p})}$ 的唯一提升：

$$
(\mathfrak{p}, L/K)(\alpha) \equiv \alpha^{\mathrm{N}(\mathfrak{p})} \pmod{\mathfrak{P}}, \qquad \forall \alpha \in \mathcal{O}_L
$$

由于 $L/K$ 阿贝尔，这个元素**不依赖** $\mathfrak{P}$ 的选择，只依赖 $\mathfrak{p}$。<span class="marginnote">阿贝尔性在此至关重要：Galois 理论中不同 $\mathfrak{P}$ 给出的 Frobenius 只差共轭（$D$ 的共轭类），而阿贝尔群里「共轭类 = 单点」。所以 Artin 符号 $(K,L/K)$ 对阿贝尔扩张是个<strong>良定的群元素</strong>——这正是 Chebotarev 定理（下一节）里「共轭类」在阿贝尔情形坍缩成单点的原因。</span>

**Artin 符号**：对无关于判别式（不整除 $\delta_{L/K}$）的整理想 $\mathfrak{m} = \prod \mathfrak{p}^{e}$，定义

$$
\left(\frac{L/K}{\mathfrak{m}}\right) = \prod_{\mathfrak{p}} \left(\frac{L/K}{\mathfrak{p}}\right)^{e}
$$

## 3 Artin 互反律的陈述

**定理（Artin 互反律）：** 设 $L/K$ 是**有限阿贝尔**扩张，$\mathfrak{m}$ 是被所有分歧素数整除（并含足量「容量」）的一个模。则 Artin 符号给出**满射**

$$
\left(\frac{L/K}{\cdot}\right): \mathrm{Cl}_{\mathfrak{m}}(K) \longrightarrow \mathrm{Gal}(L/K)
$$

其中 $\mathrm{Cl}_{\mathfrak{m}}(K)$ 是**广义理想类群**（与 $\mathfrak{m}$ 互素的理想模主理想、再模「与 1 在 $\mathfrak{m}$ 处同余」的子群）。<span class="marginnote">「互素于 $\mathfrak{m}$」让所有分歧素数从输入中剔除（它们没有 Frobenius）；「容量」让主理想的像可控。当 $\mathfrak{m} = 1$ 且 $L/K$ 是无分歧阿贝尔扩张时，Artin 映射恰是 $\mathrm{Cl}(K) \twoheadrightarrow \mathrm{Gal}(L/K)$——这就是 <strong>Hilbert 类域</strong>的初貌：它的 Galois 群同构于类群。</span>

**推论（Kronecker-Weber 定理）**：$\mathbb{Q}$ 的最大阿贝尔扩张就是**分圆扩张**：

$$
\mathbb{Q}^{\mathrm{ab}} = \bigcup_{n} \mathbb{Q}(\zeta_n)
$$

即**每个 $\mathbb{Q}$ 的阿贝尔扩张都含在某个分圆域里**。这是 Artin 互反律在 $K = \mathbb{Q}$ 时的完全落实，也是互反律最惊人的简化：**整数环的算术（全部分圆多项式）覆盖了有理数域的一切阿贝尔对称**。<span class="marginnote">把 Kronecker-Weber 推广到虚二次域 $K$：$\mathbb{Q}(\sqrt{-1})^{\mathrm{ab}}$ 由<strong>椭圆函数的复数乘法（complex multiplication）</strong>生成——这正是 Hilbert 第 12 问题：为每个数域显式构造它的最大阿贝尔扩张。至今只有 $\mathbb{Q}$ 与虚二次域（以及个别类型）被完整解决。</span>

## 4 公式解析：Frobenius 自同构 $( \mathfrak{p}, L/K )$

$$
(\mathfrak{p}, L/K)(\alpha) \equiv \alpha^{\mathrm{N}(\mathfrak{p})} \pmod{\mathfrak{P}}, \qquad \forall \alpha \in \mathcal{O}_L
$$

- **第一步，看剩余类域**：$\mathcal{O}_L/\mathfrak{P}$ 是 $\mathcal{O}_K/\mathfrak{p}$ 的有限扩域（次数 $f$）。有限域 $\mathbb{F}_q$ 的 Galois 群是循环群，由 $\mathrm{Fr}: x \mapsto x^{q}$（$q = \mathrm{N}(\mathfrak{p}) = p^f$）生成。
- **第二步，看唯一提升**：从剩余类域上的 $x \mapsto x^q$ 沿 Hensel 式的「从 $\mathfrak{P}$ 处不动性」唯一提升到 $\mathcal{O}_L$ 上的自同构——这正是分解群 $D_{\mathfrak{P}}$ 中在剩余类域上作用为 Frobenius 的那个元素。
- **第三步，为什么良定**：在阿贝尔扩张里，所有 $\mathfrak{P} \mid \mathfrak{p}$ 给出的提升在共轭意义下相同，而共轭类坍缩为单点——于是 $(\mathfrak{p}, L/K)$ 只依赖 $\mathfrak{p}$。Artin 符号因此可以线性扩展到所有整理想，得到互反律的满射。
- **第四步，信息量**：$(\mathfrak{p}, L/K) = 1$ 当且仅当 $\mathfrak{p}$ 在 $L$ 中**完全分裂**——所以 Artin 映射把「$\mathfrak{p}$ 在哪一层分裂」编码进 Galois 群的元素位置，这是 Chebotarev 密度定理的燃料。

## 5 idèle 形态：互反律的现代语言

把 Artin 互反律的各个局部版本「拼起来」，得到更精炼的 idèle 表述。定义**idèle 类群**

$$
C_K = \frac{\mathbb{A}_K^\times}{K^\times}, \qquad \mathbb{A}_K^\times = \prod_{\mathfrak{p}}' K_{\mathfrak{p}}^\times
$$

（限制直积：几乎所有分量是单位 $\mathcal{O}_{K_\mathfrak{p}}^\times$）。**全局 Artin 映射**给出连续满射

$$
\mathrm{rec}_K: C_K \longrightarrow \mathrm{Gal}(K^{\mathrm{ab}}/K)
$$

其核是 $\mathbb{R}$ 的连通分量（archimedean 情形的单位连通分支）。<span class="marginnote">idèle 类群是「广义理想类群」的极限形态——它把有限与无穷、每个素理想的所有局部数据打包成一个拓扑阿贝尔群，是类域论、自守形式与 Langlands 纲领共享的「控制台」。Dirichlet 单位定理、类数公式都在这个框架下获得统一陈述。</span>

**Langlands 纲领预告**：Artin 互反律说 $K^\times$-侧的对象（idèle 类群）的**阿贝尔**表示= Galois 侧的一维表示。Langlands 纲领把「一维」推广到「任意维」：**$K^\times$ 侧的 $n$ 维表示与 Galois 侧的 $n$ 维表示对应**。Taniyama-Shimura、Fermat 大定理证明里的「椭圆曲线模性」都是这个纲领的具体个案——一切现代数论的互反律都从 Artin 互反律出发。

**辨析｜易错点：** Artin 符号定义在「不分歧素理想」上，不能对分歧素数取 Frobenius；而互反律陈述里 $\mathfrak{m}$ 必须吃掉所有分歧。另外**阿贝尔性是「良定单点」的前提**——对非阿贝尔扩张，Frobenius 只能定义成共轭类，这正是下一节 Chebotarev 定理的表述形式。

## 6 实例：Artin 符号的计算

**例**：$L = \mathbb{Q}(\sqrt{5})$，$G = \mathrm{Gal}(L/\mathbb{Q}) = \{\mathrm{id}, \sigma\}$。$p \ne 5$ 不分歧时，Frobenius 由「$\sqrt5$ 被 $\mathrm{Fr}$ 送到 $\sqrt5$（分裂）还是 $-\sqrt5$（惯性）」决定，即

$$
\left(\frac{L/\mathbb{Q}}{p}\right) = \begin{cases} \mathrm{id}, & \left(\frac{5}{p}\right) = 1 \\ \sigma, & \left(\frac{5}{p}\right) = -1 \end{cases}
$$

验算 $p = 11$：$\left(\frac{5}{11}\right) = \left(\frac{11}{5}\right) = \left(\frac{1}{5}\right) = 1$，故 $11$ 分裂，Artin 符号平凡。$p = 3$：$\left(\frac{5}{3}\right) = -1$，$3$ 惯性，符号 $\sigma$。<span class="marginnote">二次域上「Artin 符号 = 勒让德符号」这一步看似平凡，却是整条互反律的试金石：一切关于 Artin 互反律的证明，都必须在这个最简情形恢复高斯二次互反律。</span>

**辨析｜易错点：** Artin 映射的核是「在 $L$ 中完全分裂的素数所生成的（广义）主理想子群」——**互反律的精确陈述是「$\mathrm{Cl}_{\mathfrak m}(K)$ 的一个子群被平凡映射」**，而单个素理想是否在核里由 Frobenius 是否平凡判定。别把「Artin 符号」（$G$ 中的元素）与「理想类」（$K$ 的算术对象）混为一谈。**$\mathfrak m$（conductor）的选择是记号的难点**：$\mathfrak m$ 必须含所有分歧素数，且使主理想的像平凡——conductor 越小，互反律越「经济」。

## 7 小结

- 二次互反律 = 分圆域上 Artin 符号的一格投影；$(\frac{q}{p})$ 记录 $\zeta_p \mapsto \zeta_p^q$ 的 Frobenius。
- **Frobenius** $(\mathfrak{p}, L/K)$：剩余类域上 $x \mapsto x^{\mathrm{N}(\mathfrak{p})}$ 的提升；阿贝尔时良定且只依赖 $\mathfrak{p}$。
- **Artin 互反律**：$\mathrm{Cl}_{\mathfrak{m}}(K) \to \mathrm{Gal}(L/K)$ 满射；Hilbert 类域给出 $\mathrm{Cl}(K) \cong \mathrm{Gal}(H/K)$。
- **Kronecker-Weber**：$\mathbb{Q}^{\mathrm{ab}} = \bigcup \mathbb{Q}(\zeta_n)$；Hilbert 第 12 问题（虚二次域的复数乘法）。
- **idèle 形态**：$\mathrm{rec}_K: C_K \to \mathrm{Gal}(K^{\mathrm{ab}}/K)$；Langlands 纲领把互反律推广到任意维表示。

在下一节，我们研究 Frobenius 元素在素数间的**分布**——**Chebotarev 密度定理**：Galois 群中每个共轭类被「恰好 $|C|/|G|$ 比例的素数」实现，这是统计数学的素数版。

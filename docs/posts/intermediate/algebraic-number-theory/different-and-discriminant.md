---
title: 差积与判别式
date: 2026-08-11
---

# 差积与判别式

<div class="epigraph">
<p>数学的本质在于它的自由。</p>
<footer>—— 格奥尔格 · 康托尔（Georg Cantor，Das Wesen der Mathematik liegt in ihrer Freiheit）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从差积与判别式开始

分歧理论告诉我们每个素理想 $\mathfrak{P}$ 在 $\mathfrak{p}$ 上有分歧指数 $e$，但「分歧到底有多强」是一个更精细的问题：$e = p$ 与 $e = p^2$ 都是分歧，它们的差别要用**高阶分歧群**来度量。本节把这种「强度」铸成一个标准的算术对象——**差积（different）**——再把它「压」到基域得到一个理想——**判别式（discriminant）**。这两个理想把分歧理论的全部信息编码进两个乘法对象，是整个类域论与代数几何交界的黏合剂。<span class="marginnote">「判别式」这个词你可能在初中听过（二次方程 $\Delta = b^2 - 4ac$ 判断根的情况）。这里的判别式是它的远亲：<strong>数域扩张的判别式「判别」哪些素数分歧</strong>——同族的直觉，格局完全不同。</span>

## 1 对偶模与余差积

回忆数域 $L/K$ 中的迹映射 $\mathrm{Tr} = \mathrm{Tr}_{L/K}: L \to K$。用迹定义一个「对偶」：

**余差积 / 逆差积（codifferent，inverse different）**：

$$
\mathcal{D}_{L/K}^{-1} = \{x \in L : \mathrm{Tr}(x\, \mathcal{O}_L) \subseteq \mathcal{O}_K\}
$$

它是包含 $\mathcal{O}_L$ 的分式理想，可看成「$\mathcal{O}_L$ 关于迹形式的对偶模」。<span class="marginnote">类比线性代数：有限维向量空间里，一个子格关于非退化双线性型（迹）的对偶格。$\mathcal{D}^{-1}$ 正是 $\mathcal{O}_L$ 的对偶格——这个观点让差积理论从「整除技巧」变成「格论事实」。</span>

**差积（different）**：$\mathcal{D}_{L/K} = \big(\mathcal{D}_{L/K}^{-1}\big)^{-1}$，它是 $\mathcal{O}_L$ 的理想。差积的素理想分解直接编码分歧：

$$
\mathcal{D}_{L/K} = \prod_{\mathfrak{P}} \mathfrak{P}^{d_{\mathfrak{P}}}, \qquad d_{\mathfrak{P}} \ge 0
$$

并且 $d_{\mathfrak{P}} = 0$ **当且仅当 $\mathfrak{P}$ 在 $\mathfrak{p}$ 上不分歧**——**差积精确标注了分歧发生的「位置」与「强度」**。

## 2 Galois 情形：差积指数 = 分歧群级数

当 $L/K$ 是 Galois 扩张时，指数 $d_{\mathfrak{P}}$ 有一个惊人的闭式——**Hilbert 差积公式**：

$$
d_{\mathfrak{P}} = \sum_{i = 0}^{\infty} \big(|G_i| - 1\big)
$$

其中 $G_i$ 是上一节的高阶分歧群滤过（$G_{-1} = G$，$G_0 = I$，$G_1$ 为野群）。<span class="marginnote">这条公式把「高阶分歧群的每一项长度」逐级相加：$|G_0| - 1 = e - 1$ 已给出「分歧指数 $e$」的贡献，而 $G_1, G_2, \dots$ 每一项再添「更深一层分歧」的力度。驯分歧时 $G_1 = \{1\}$，于是 $d_{\mathfrak{P}} = e - 1$——干净利落。</span>

**例（完全分歧的驯情形）**：$K = \mathbb{Q}$，$L = \mathbb{Q}(\sqrt{-1})$，素理想 $(1+i)$（在 $(2)$ 之上）。$e = 2$，$G_0 = G \cong \mathbb{Z}/2$，$G_1 = \{1\}$，故 $d = 2 - 1 = 1$，$\mathcal{D}_{L/\mathbb{Q}} = (1+i)$。判别式 $d_L = \mathrm{N}(1+i) = 2$，正是 $d_{\mathbb{Q}(i)} = -4$ 的绝对值除符号。

## 3 判别式理想与「分歧 ⟺ 整除」

**判别式理想（discriminant ideal）** 把差积「压」到基域：

$$
\delta_{L/K} = \mathrm{N}_{L/K}\big(\mathcal{D}_{L/K}\big) = \prod_{\mathfrak{p}} \mathfrak{p}^{\,\sum_{\mathfrak{P} \mid \mathfrak{p}} f_{\mathfrak{P}}\, d_{\mathfrak{P}}}
$$

对 $K = \mathbb{Q}$，判别式理想对应整数 $d_L = \mathrm{N}_{L/\mathbb{Q}}(\mathcal{D}_{L/\mathbb{Q}})$，即通常的数域判别式（差一个符号）。

**核心定理（分歧判别）：**

$$
\mathfrak{p} \text{ 在 } L/K \text{ 中分歧} \iff \mathfrak{p} \mid \delta_{L/K}
$$

**判别式「收集」了所有分歧素数**。有限多个素数才分歧（判别式只有有限素因子），于是：**有限扩张只在有限个素理想处分歧**——这个看似平淡的事实是类域论「无分歧扩张」理论的地基。<span class="marginnote">对 $\mathbb{Q}$ 的扩张，这给出一个惊人结论：<strong>分歧素数的集合可以任意指定吗？</strong>不能——由判别式有限性，分歧集合必须有限；而哪些有限集合能成为某个数域的「分歧集」，正是类域论的准入问题之一。</span>

**例**：$\mathbb{Q}(\sqrt{d})$（$d$ 平方自由）的判别式为

$$
d_L = \begin{cases} d, & d \equiv 1 \pmod 4 \\ 4d, & d \not\equiv 1 \pmod 4 \end{cases}
$$

故 $d = -5$ 时 $d_L = -20 = -2^2 \cdot 5$，分歧素数为 $2$ 与 $5$——回到第一节：$2$ 在 $\mathbb{Z}[\sqrt{-5}]$ 分歧（$(2) = \mathfrak{p}_2^2$），$5$ 也分歧，与公式吻合。

## 4 公式解析：$d_{\mathfrak{P}} = \sum_{i\ge0}(|G_i| - 1)$

$$
\boxed{\,d_{\mathfrak{P}} = \sum_{i = 0}^{\infty} \big(|G_i| - 1\big)\,}
$$

三步拆开这条「分歧强度计」：

- **第一步，看 $G_0 = I$ 的贡献**：$|G_0| - 1 = e - 1$，是「幂次损失」的粗糙度量：分歧指数 $e$ 意味着一份素理想被摊成 $e$ 份，损失 $e - 1$ 个「维数」。
- **第二步，看更高阶群**：$G_1, G_2, \dots$ 逐步「不动到更深的 $\mathfrak{P}^{i+1}$」。它们非平凡，正是「分歧在野性上还要再深」的标志。驯分歧时 $G_1 = \{1\}$，各 $|G_i| - 1 = 0$，公式退化为 $d_{\mathfrak{P}} = e - 1$。
- **第三步，为什么要「和」**：因为差积的素因子指数是各层「长度损失」的**累积**——每层滤过的每个非平凡元素都「吃掉」一个幂次。这个加法结构保证了差积的范（判别式）仍是理想层面的乘法不变量，从而把「分歧强度」整合成单一整数。

## 5 差积、判别式与类域论的交接

差积和判别式不只是记账工具，它们直接参与代数数论最深的推论：

- **分歧集与无分歧扩张**：判别式决定分歧素数的有限集；**无分歧（且处处分歧指数 $=1$）扩张**是类域论的主角——Hilbert 类域（最大无分歧阿贝尔扩张）的 Galois 群恰好同构于类群。
- **不同公式**：判别式还可用迹矩阵 $\det(\mathrm{Tr}(\omega_i\omega_j))$ 计算——判别式 = 整基的迹矩阵的行列式，这是它与「格体积」在 Minkowski 嵌入下的又一次重逢。
- **判别式与函数域**：在代数几何里，差积对应曲线的**微分模**，判别式对应**分歧除子**——同一个理论换了名字换了视角。<span class="marginnote">这就是为什么代数数论与代数几何（尤其是算术代数几何）能无缝对话：<strong>理想、差积、判别式</strong>在几何侧就是<strong>除子、微分、分歧除子</strong>。给后续《代数几何》与《椭圆曲线》专题留的接口，就在这条对应上。</span>

**辨析｜易错点：** 差积 $\mathcal{D}_{L/K}$ 是 **$L$ 里的理想**，判别式 $\delta_{L/K}$ 是 **$K$ 里的理想**，别混。$K = \mathbb{Q}$ 时判别式是个整数，但它仍由「$L$ 侧的分歧」决定——方向是 **$L$ 分歧 ⟹ $K$ 的判别式有因子**，不是反过来。另外「$d_{\mathfrak{P}} = e - 1$」只在**驯**分歧时成立，野分歧必须用全和式。

## 6 实例：差积与判别式的计算

**例 1（$K = \mathbb{Q}(\sqrt{-1})$）**：素理想 $(1+i) \mid (2)$，$e = 2$，驯分歧（$p = 2 \nmid e$），故 $d_{(1+i)} = e - 1 = 1$，差积

$$
\mathcal{D}_{K/\mathbb{Q}} = (1+i), \qquad d_K = \mathrm{N}(1+i) = 2
$$

而 $d_{\mathbb{Q}(i)} = -4$ 的绝对值正是 $2$——一致。

**例 2（$K = \mathbb{Q}(\sqrt{2})$）**：$(\sqrt{2})$ 上分歧、驯，$\mathcal{D} = (\sqrt{2})$，$|\mathrm{N}(\sqrt2)| = 2$，$|d_K| = 8$——再次核对。

**二次域判别式表**：

| $K$ | $d$ | $d_K$ | 分歧素数 |
| --- | --- | --- | --- |
| $\mathbb{Q}(\sqrt{-1})$ | $-1$ | $-4$ | $2$ |
| $\mathbb{Q}(\sqrt{-3})$ | $-3$ | $-3$ | $3$ |
| $\mathbb{Q}(\sqrt{-5})$ | $-5$ | $-20$ | $2, 5$ |
| $\mathbb{Q}(\sqrt{2})$ | $2$ | $8$ | $2$ |

**辨析｜易错点：** 差积 $\mathcal{D}_{L/K}$ 是 **$L$ 中的理想**，判别式 $\delta_{L/K} = \mathrm{N}(\mathcal{D}_{L/K})$ 是 **$K$ 中的理想**——「$\mathfrak{p}$ 分歧 $\iff \mathfrak{p} \mid \delta$」读作「分歧的迹落在判别式的素因子」。二次域判别式常带符号（$-4, -3, -20$），**符号本身与分歧无关**，判断只看素因子。

## 7 小结

- **余差积** $\mathcal{D}^{-1} = \{x : \mathrm{Tr}(x\mathcal{O}_L) \subseteq \mathcal{O}_K\}$：$\mathcal{O}_L$ 关于迹的对偶格；**差积** $\mathcal{D} = (\mathcal{D}^{-1})^{-1}$。
- **差积分解** $\mathcal{D} = \prod \mathfrak{P}^{d_{\mathfrak{P}}}$；$d_{\mathfrak{P}} = 0 \iff$ 不分歧。
- **Galois 时** $d_{\mathfrak{P}} = \sum_{i\ge0}(|G_i| - 1)$（Hilbert 公式），驯时退化 $= e - 1$。
- **判别式理想** $\delta_{L/K} = \mathrm{N}(\mathcal{D})$；**$\mathfrak{p}$ 分歧 $\iff \mathfrak{p} \mid \delta_{L/K}$**。
- 二次域判别式：$d \equiv 1 \pmod 4$ 时 $d$，否则 $4d$；分歧素数恰为判别式的素因子。

在下一节，我们进入代数数论的内核——**局部类域论**：把视线锁在单个素理想的完备化 $\mathbb{Q}_p$ 里，用范数群与 Artin 映射完全分类它的阿贝尔扩张。

---
title: Cohen-Macaulay 模与 Gorenstein 环
date: 2026-08-07
---

# Cohen–Macaulay 模与 Gorenstein 环

<div class="epigraph">
<p>Cohen–Macaulay 环是「所有系统参数都是正则序列」的环——维数在每一步都真正下降。</p>
<footer>—— Francis S. Macaulay 与 Irvin S. Cohen（引述其理论精神）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Matsumura Ch. 6 / Eisenbud Ch. 18 ｜ 2026-08-07</p>
</div>

## 为什么从 CM 开始

上一节我们认识了深度：$\operatorname{depth} M \leq \dim M$ 永远成立。什么时候取等号？这个条件定义了交换代数里「好脾气」的一族对象——**Cohen–Macaulay 模**（下称 CM）。正则环、多项式环、超曲面、完整交、代数簇的坐标环（在不太坏时）全都是 CM；而 **Gorenstein 环**更是 CM 里的自对偶贵族，它是「对偶化模等于自身」的环，也是同调对偶（Serre 对偶、局部对偶）的代数化身。<span class="marginnote">Macaulay（1862—1937）研究多项式理想系统的深度与参数；Cohen（1917—1955，此前已见其正规化引理）1946 年把这个性质推广到一般局部环，因而得名 Cohen–Macaulay。Gorenstein 则得名于 Daniel Gorenstein（1923—1992），他在 1952 年研究平面曲线的对偶性时提出这个概念——此后 Gorenstein 的名字在代数几何与表示论里随处可见。</span>

这一篇把「CM」与「Gorenstein」讲成一张递进的谱系：正则 ⊂ 完整交 ⊂ Gorenstein ⊂ CM，再落到判据与例子。

## 1 CM 模：深度达到上限

**Cohen–Macaulay 模**：有限生成 $R$-模 $M$ 称为 CM 模，若 $\operatorname{depth} M = \dim M$（$R$ 局部）。**CM 环**：$R$ 作为自身模是 CM。

**重点：CM = 存在「系统参数同时是正则序列」。** 对 $R$ 局部、$\dim R = d$，以下等价：
1. $R$ 是 CM 环（$\operatorname{depth} R = d$）；
2. 存在系统参数 $x_1, \dots, x_d$ 构成正则序列；
3. 每个系统参数都是正则序列（排列序后）；
4. $R[x]$、$R$ 的局部化、$R$ 的商环（商掉正则序列）都 CM。

标准例子：
**正则局部环是 CM**（$\mathfrak{m}$ 由正则序列生成——上一节的 Koszul 判据）。
- 超曲面 $k[x_1,\dots,x_n]/(f)$（$f$ 非零因子）：$\dim = n-1$、$\operatorname{depth} = n-1$，CM。
- $R = k[x,y]/(xy)$：CM（深度 = 维数 = 1）；$R = k[x,y]/(x^2, xy)$：深度 0、维数 1，**非** CM。<span class="marginnote">几何直觉：CM 簇的坐标环「处处维数被方程真正压低」，没有「多余维度的残余」。节点 $xy=0$ 是 CM（它真的是一维曲线），而 $k[x,y]/(x^2, xy)$ 的嵌入分支让深度跌到 0——「嵌入素理想」正是深度的杀手，回想第1篇《准素分解》的嵌入素理想。</span>

**辨析｜易错点：** CM 是**逐点**性质：$R$ CM ⇔ 所有 $R_{\mathfrak{p}}$ 都 CM。整环未必 CM（存在非 CM 的整环——高度 2 的正规环还好，但一般的整环可能失败）。**「整环」与「CM」是两个独立的好性质**，别混为一谈。

**核心对照表：谁是 CM？**

| 环 | $\dim$ | $\operatorname{depth}$ | CM? | 备注 |
| --- | --- | --- | --- | --- |
| 正则局部环 | $d$ | $d$ | 是 | 最光滑的一族 |
| $k[x_1,\dots,x_n]$ | $n$ | $n$ | 是 | 多项式环 |
| $k[x,y]/(xy)$ | 1 | 1 | 是 | 节点曲线 |
| $k[x,y]/(x^2, xy)$ | 1 | 0 | 否 | 嵌入分支 |
| $k[x,y,z]/(xy, xz)$ | 2 | 1 | 否 | 平面与线相交，深度不足 |
| 完整交 | $n-r$ | $n-r$ | 是 | 方程数 = 余维 |

末两行是「差一步」的典型：$k[x,y,z]/(xy,xz)$ 是平面 $x=0$ 与直线 $y=z=0$ 的并，维数 2 但深度只有 1——**CM 性在「分支相交」处最容易失效**，这正是几何学家偏爱 CM 簇的原因。

**辨析｜CM 与平坦：** CM 性是「忠实平坦不变」的：$R$ CM ⇔ 所有 $R_{\mathfrak{p}}$ CM ⇔ $R[x]$ CM。但它不随「任意子环」传递。几何上，CM 簇的平坦纤维仍 CM，这让「CM 是能放心做代数几何的性质」。

## 2 CM 的判据：正则序列与 unmixedness

CM 的另一个重要刻画是**无混合性（unmixedness）**：

**重点：CM 环中，正则序列生成的理想 $\mathfrak{a}$ 满足「$\operatorname{Ass}(R/\mathfrak{a})$ 全部是极小素理想」——没有嵌入素理想。** 反过来，一个环若对每个由正则序列生成的理想都无嵌入素理想，则它是 CM。

这条连接了《准素分解》与《深度》：CM ⇔ **「准素分解干净」**——没有嵌入素理想。这解释了为什么几何学家偏爱 CM 簇：分解唯一、无多余分支、维数处处正确。<span class="marginnote">「深度 = 维数」与「无嵌入素理想」在这条线互为表里：嵌入素理想恰是让 $\operatorname{depth}$ 跌落的元凶。CM 性把两边的直觉一次性理顺。</span>

**辨析｜易错点：** 「无嵌入素理想」只对「由正则序列生成的理想」成立，是 CM 的强性质。$k[x,y]/(x^2,xy)$ 不 CM，它的理想 $(\bar{x})$ 由单个元素生成，但 $\bar{x}$ 本身是零因子、不是正则序列——**无混合性根本没被触发**。判断「某理想是否由正则序列生成」，先确认生成元确实是正则序列，别只看个数。

## 3 Gorenstein 环：自对偶的 CM 环

Gorenstein 环是 CM 环里具有「自对偶性」的一族。现代定义用**规范模（canonical module）**：

**规范模（canonical module）**：CM 局部环 $R$（$\dim d$）的规范模 $\omega_R$ 是一个 CM 模，满足 $\omega_R$ 的深度 $= d$ 且 $\omega_R$ 在局部化下行为良好（$\omega_{R_{\mathfrak{p}}} = (\omega_R)_{\mathfrak{p}}$ 的合适意义）。

**Gorenstein 环**：CM 环 $R$，其规范模 $\omega_R \cong R$（同构意义下）。<span class="marginnote">等价的经典定义：$R$ 作为自身的模，<strong>内射维数有限</strong>——即「$R$ 的自内射分解只有有限多项」。正则环、完整交都是 Gorenstein；Gorenstein 环的局部化、完备化仍 Gorenstein。</span>

**递进谱系**（对局部环）：

$$\text{正则} \;\subsetneq\; \text{完整交} \;\subsetneq\; \text{Gorenstein} \;\subsetneq\; \text{CM}$$

其中「完整交（complete intersection）」指「由正则序列生成的理想的商」，它在几何上对应「余维 = 方程个数」的横截对象。<span class="marginnote">谱系的每一层都严格：$k[x,y]/(x^2,y^2)$ 是 Gorenstein 非完整交（二维余环）的例子稍复杂；最简单区分：维数 0 时 Gorenstein ⇔ socle 一维——$k[x]/(x^2)$ 是 Gorenstein（socle = $(\bar{x})$ 一维），而 $k[x,y]/(x^2, xy, y^2)$（socle 二维）是 CM 非 Gorenstein。</span>

**重点：维数 0 时，Gorenstein ⇔ socle 恰一维。** 这一条把 0 维 Gorenstein 环完全分类：$R$ Artin 局部，$\operatorname{socle}(R) = \{m \mid \mathfrak{m}m = 0\}$ 作为 $k$-向量空间维数恰为 1。这是判断小例子的最实用判据。

**0 维 Gorenstein 判据的算例：**

| 环 | socle | Gorenstein? |
| --- | --- | --- |
| $k$ | $k$（一维） | 是 |
| $k[x]/(x^2)$ | $k\cdot\bar{x}$（一维） | 是 |
| $k[x]/(x^3)$ | $k\cdot\bar{x}^2$（一维） | 是 |
| $k[x,y]/(x^2,xy,y^2)$ | $k\cdot\bar{x} \oplus k\cdot\bar{y}$（二维） | 否 |
| $k[x,y]/(x^2,y^2)$ | $k\cdot\bar{x}\bar{y}$（一维） | 是 |

socle 一维与否，是 0 维 Gorenstein 的**完全判据**——这几行全部可以手算，是检查理解的最快题组。

## 4 公式解析：规范模与 Gorenstein 的判定

对 CM 局部环 $(R, \mathfrak{m}, k)$、$\dim R = d$，Gorenstein 判据可以写成：

$$\omega_R \cong R \iff \operatorname{Ext}^i_R(k, R) = 0\ (i < d)\ \text{且}\ \operatorname{Ext}^d_R(k, R) \cong k.$$

- **第一步，左侧到右侧**：$\omega_R \cong R$ 时，规范模的深度 $d$ 由「$\operatorname{Ext}^i(k, \omega) = 0$（$i < d$）」体现（CM 的同调刻画，见《深度》），而 $\operatorname{Ext}^d_R(k, \omega_R) \cong k$ 是规范模的「对偶性公理」——把「$\omega$ 是 $k$ 的『最后一阶对偶』」写成了等式。
- **第二步，右侧到左侧**：$\operatorname{Ext}^d_R(k, R) \cong k$ 说明 $R$ 满足「socle 一维」的 $d$ 维推广（$\operatorname{Ext}^d(k, R)$ 的非零元对应 $R$ 的「顶部对偶」），配合 CM 条件推出 $R \cong \omega_R$。
- **第三步，记住结论**：**Gorenstein ⇔「顶部 Ext 恰好是 $k$」**。0 维时 $\operatorname{Ext}^0(k, R) = \operatorname{Hom}(k, R) = \operatorname{socle} R$，就回到「socle 一维」。<span class="marginnote">最后一篇《局部上同调》会看到 $\operatorname{Ext}^d_R(k, R)$ 与顶部局部上同调 $H^d_{\mathfrak{m}}(R)$ 的关系——Gorenstein 判据在那里还会再以「$H^d_{\mathfrak{m}}(R)$ 是 $k$ 的内射包」的面目登场。</span>

**辨析｜易错点：** 完整交 ⊂ Gorenstein 但反过来不成立。$k[x_1,x_2,x_3]/(x_1^2, x_2^2, x_3^2, x_1 x_2 x_3)$ 之类的例子说明「方程数超过余维」也能 Gorenstein。判断层级别只记「Gorenstein = 完整交」——**谱系是严格嵌套的**。

对 0 维情形把判据落到 $\operatorname{Ext}$：$R = k[x]/(x^2)$，$d = 0$，$\operatorname{Ext}^0(k, R) = \operatorname{Hom}(k, R) = \operatorname{socle} R = k \cdot \bar{x} \cong k$——「$\operatorname{Ext}^0 \cong k$」正是「socle 一维」的 $\operatorname{Ext}$ 写法。$R$ 是 0 维 Gorenstein，同时也是正则序列 $(x^2)$ 的商（完整交），两个层次在此重合。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| CM 模 | $\operatorname{depth} M = \dim M$ |
| 系统参数 | 张出 $\mathfrak{m}$-准素理想的 $d$ 个元素 |
| 无混合性 | 正则序列的商无嵌入素理想 |
| 规范模 $\omega_R$ | CM 环的「对偶化模」 |
| Gorenstein | $\omega_R \cong R$（或内射维数有限） |
| 完整交 | 正则序列生成理想的商 |

## 5 小结

- **CM 模**：$\operatorname{depth} = \dim$；等价于系统参数是正则序列；正则、超曲面都是 CM。
- **CM 的无混合性**：正则序列的商无嵌入素理想——「分解干净」。
- **规范模** $\omega_R$：CM 环的「对偶化模」，局部化下行为良好。
- **Gorenstein**：$\omega_R \cong R$（或内射维数有限）；谱系 正则 ⊂ 完整交 ⊂ Gorenstein ⊂ CM；0 维判据 = socle 一维。

在下一节，我们从「对偶化」退一步，回到最基础的成员资格检查：**相伴素与支集**——$\operatorname{Ass} M$ 与 $\operatorname{Supp} M$ 是模的「病历卡」与「领土图」。

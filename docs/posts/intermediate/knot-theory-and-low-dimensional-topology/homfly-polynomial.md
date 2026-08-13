---
title: HOMFLY 多项式
date: 2026-08-07
---

# HOMFLY 多项式

<div class="epigraph">
<p>六个作者独立找到同一个多项式，数学终于承认它配得上六个字母。</p>
<footer>—— 本文作者按</footer>
</div>

<div class="article-byline">
<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第16章 ｜ 2026-08-07</p>
</div>

## 为什么从「HOMFLY」开始

Jones 多项式「能区分镜像」的能力让全领域兴奋：那 Alexander 多项式与 Jones 多项式能否统一进一个更一般的 skein 不变量？1985 年，六个研究组（Freyd–Yetter、Hoste、Lickorish–Millett、Ocneanu、Przytycki–Traczyk）独立找到同一个答案——一个双变量的 Laurent 多项式 $P_L(\ell, m)$，把 Alexander 与 Jones 作为特例一次性包含。为纪念六组作者，它命名为 **HOMFLY 多项式**（各取首字母）。

HOMFLY 的深刻之处不在「更强的分辨力」，而在「通用性的数学结构」：它证明 Alexander 与 Jones 是同一条 skein 关系的两个「投影」，一切 skein 关系确定的不变量都是它的特例。<span class="marginnote">Przytycki–Traczyk 与 Traczyk 的证明尤其漂亮：他们从「定向 skein 关系」的公理出发，证明满足该关系的双变量多项式必为 HOMFLY（相差归一化）。这意味着「最一般定向 skein 不变量」恰好是 HOMFLY——没有更一般的同类对象，它在「定向 skein 谱系」中是终点。</span>

## 1 双变量 skein 关系

**HOMFLY 多项式** $P_L(\ell, m) \in \mathbb{Z}[\ell^{\pm 1}, m^{\pm 1}]$ 对定向链环定义，满足：

$$
\ell\, P_{L_+} + \ell^{-1} P_{L_-} + m\, P_{L_0} = 0,
$$

归一化 $P_{0_1} = 1$。其中 $L_+, L_-, L_0$ 是定向的三种局部状态。

与 Conway 关系 $\nabla_{L_+} - \nabla_{L_-} = z \nabla_{L_0}$ 比较，HOMFLY 把「系数」参数化为 $\ell, \ell^{-1}, m$ 三个一般系数。这个「参数化」正是它包罗万象的原因：

- 取 $\ell = 1$ 时，关系化为 $\nabla_{L_+} - \nabla_{L_-} = -m \nabla_{L_0}$——正是 Conway 关系（$z = -m$）。
- 取适当的 $\ell, m$ 组合，化为 Jones 关系。

**定理（HOMFLY 包含 Alexander 与 Jones）**：对任意定向链环 $L$，

$$
\Delta_L(t) = P_L(1, \, -(t^{1/2} - t^{-1/2})),
$$

$$
V_L(t) = P_L(it^{-1}, \, i(t^{-1/2} - t^{1/2})),
$$

（代入相应变量后归一化），即**替换变量即可从 HOMFLY 读出 Alexander 与 Jones**。

## 2 公式解析：为什么一个关系套住两个不变量

把 HOMFLY 关系重写为「解出 $P_{L_+}$」的形式：

$$
P_{L_+} = -\ell^{-2} P_{L_-} - \ell^{-1} m\, P_{L_0}.
$$

- **第一步，系数是「开关」**：每个交叉的展开系数 $(-\ell^{-2}, -\ell^{-1}m)$ 是两个变量的一次式。Alexander 与 Jones 各自是「把系数空间压到一维」的投影——就像三维向量投影到两个不同的坐标轴。
- **第二步，为什么投影不同**：Conway/Alexander 要求「镜像下 $z \to -z$」（对称规范），Jones 要求「$t \leftrightarrow t^{-1}$ 与镜像配套」。两条不同的对称性约束，在 HOMFLY 的二维系数空间里被统一处理——不再二选一。
- **第三步，多余信息**：HOMFLY 通常比 Alexander 与 Jones 的分辨力都强（信息更多），但也可能「打平」——存在 HOMFLY 相同的不同结。它不万能，但它是「定向 skein 世界的最大值」。

**易错点｜HOMFLY 的定义不唯一**：文献中 HOMFLY 有多种归一化约定（差 $(-1)^{\mu-1}$ 因子、变量 $\ell \leftrightarrow \ell^{-1}$ 等）。引用数值时必须说明采用哪种约定——Lickorish 的 $\ell P_+ + \ell^{-1}P_- + m P_0 = 0$ 是最常见版本，但并非唯一。

## 3 用 HOMFLY 算三叶结

以三叶结 $3_1$ 为例展示双变量计算。设 $T = 3_1$，$H$ 为 Hopf 链环，$U$ 为平凡结。两个独立的 skein 关系：

$$
\ell P_T + \ell^{-1} P_U + m P_H = 0, \qquad
\ell P_H + \ell^{-1} P_U + m P_U = 0.
$$

第二个式子解出 $P_H = -\ell^{-1} m - \ell^{-2}$（用 $P_U = 1$）。代入第一个式子：

$$
P_T = -\ell^{-2} - \ell^{-1} m\, P_H = -\ell^{-2} + \ell^{-1} m(\ell^{-1} m + \ell^{-2})
$$

$$
P_{3_1}(\ell, m) = -\ell^{-2} + \ell^{-2} m^2 + \ell^{-3} m.
$$

- **第一步，两个关系两条路**：三叶结的「展开树」只有两层（正交叉→Hopf→平凡），每一步都用同一条 skein 关系。
- **第二步，验证投影**：代入 $\ell = 1, m = -(t^{1/2} - t^{-1/2})$，得到 $P = -1 + (t - 2 + t^{-1}) - (t^{1/2} - t^{-1/2}) + \ldots$，经归一化还原为 $\Delta_{3_1} = t^2 - t + 1$ 的 Conway 形式。
- **第三步，双变量的价值**：$P_{3_1}$ 同时「记得」Alexander 与 Jones 两个投影——这就是「一个对象，两个视角」。

## 4 HOMFLY 的性质

- **归一化与分量数**：$P_{0_1} = 1$；平凡 $\mu$ 分量链环的 $P$ 与 $\mu$ 有关（含 $(-1)^{\mu-1}$ 因子）。
- **镜像**：$P_{K^*}(\ell, m) = P_K(\ell^{-1}, m)$——镜像把 $\ell$ 换 $\ell^{-1}$，$m$ 不变。
- **定向翻转**：整个定向翻转，$P$ 不变；翻转一个分量则 $\ell \to \ell^{-1}$。
- **连通和**：$P_{K_1 \# K_2} = P_{K_1} P_{K_2}$。
- **镜像判定**：$P_K(\ell, m) \neq P_K(\ell^{-1}, m)$ 则 $K$ 手性。三叶结手性、八字结两性，均由 HOMFLY 直接验证。<span class="marginnote">HOMFLY 的镜像规则「$\ell \leftrightarrow \ell^{-1}$」比 Jones 的「$t \leftrightarrow t^{-1}$」更细：它把「镜像」与「定向翻转」两个操作分开了。链环的「分量子定向翻转」不再是简单的变量倒置，而是与 $\ell$ 的翻转绑定——这使 HOMFLY 能分辨更精细的定向现象。</span>

**辨析｜HOMFLY 与 Jones 的镜像判定能力**：对**单结**，HOMFLY 判镜像与 Jones 判镜像等价（都是「多项式是否对称」）。但对**链环**，HOMFLY 的分量定向信息更完整——它能区分「改变一个分量定向」与「整体镜像」，这是 Jones 做不到的。

## 5 HOMFLY 在量子群中的位置

HOMFLY 不只是「通用 skein 不变量」。1990 年代，它被纳入**量子群不变量**框架（第3篇之四）：HOMFLY 对应量子群 $U_q(\mathfrak{sl}_N)$ 的伴随表示不变量，$N$ 是一个参数：

$N = 2$：$U_q(\mathfrak{sl}_2)$ 给出 Jones 多项式。
$N \to 1$（适当极限）：Alexander 多项式。
- 一般 $N$：得到 $A_N$ 型量子不变量，HOMFLY 是它们的「通用表示」。

于是 HOMFLY 的「两个投影」有了代数解释：**Alexander 与 Jones 是量子群表示论中两个不同维数表示的痕迹**。<span class="marginnote">这条「量子群 → 结不变量」的路线由 Reshetikhin–Turaev 系统化（第3篇之四）。HOMFLY 在其中扮演「$A_N$ 型通用对象」：一换 $N$，就得到一整族不变量。结多项式从此从「孤立技巧」升级为「表示论的天然产物」。</span>

### 为什么「双变量」比「单变量」强

HOMFLY 用两个变量 $\ell, m$，比单变量（Conway、Jones）多一个自由度——这正是它分辨力更强的来源：

- 单变量不变量只「记录一条曲线的信息」；双变量记录「一条曲线在两个方向上的投影」。
- HOMFLY 能区分某些 Jones 与 Alexander 都分不开的结——因为它在「两个变量」里同时保留了两类信息。
- 但 HOMFLY 也不是万能的（存在 HOMFLY 相同而不同结），它只是「定向 skein 谱系」的最强成员。

**「更多参数 = 更多信息」不是自动的**：参数多了，关系也可能退化。HOMFLY 的价值在于它的两个参数**独立携带信息**——Alexander 与 Jones 是它的两个「方向」，合起来信息更多。

### 判别「HOMFLY 相同 ≠ 同结」的例子

文献里常有「HOMFLY 无法区分某些结对」的反例（如同伴结对），提醒我们「不变量再强也有极限」。处理办法：换 Kauffman 多项式（无定向谱系）或换几何量（体积、签名）补上——「多谱系并用」是结分类的实践准则。

## 6 小结

- **HOMFLY 多项式** $P_L(\ell, m)$ 满足双变量 skein 关系 $\ell P_+ + \ell^{-1} P_- + m P_0 = 0$。
- 它是**最一般的定向 skein 不变量**：Alexander 与 Jones 都是它的投影（变量替换）。
- 三叶结 $P = -\ell^{-2} + \ell^{-2}m^2 + \ell^{-3}m$；镜像对应 $\ell \to \ell^{-1}$。
- HOMFLY 对链环的定向敏感度高于 Jones，是「分量子定向翻转」的精细探测器。
- 在量子群框架中，HOMFLY 是 $U_q(\mathfrak{sl}_N)$ 的通用不变量，$N = 2$
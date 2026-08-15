---
title: 与代数数论的联系（类群与调节子）
date: 2026-08-07
---

# 与代数数论的联系（类群与调节子）

<div class="epigraph">
<p>K 理论把类数公式从 $s = 1$ 推广到了所有整数 $s = n$。</p>
<footer>—— 斯蒂芬·利希滕鲍姆（Stephen Lichtenbaum）</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数 K 理论 ｜ Weibel《The K-book》§10 ｜ 2026-08-07</p>
</div>

## 为什么代数数论是 K 理论的试金石

代数数论与代数 K 理论的相遇，是 20 世纪数学最肥沃的交叉之一。数论的核心对象——**理想类群**与**单位群**——从 K 理论的视角看，不过是 $K_0$ 与 $K_1$ 的特殊情形；而**高阶 K 群**则把「类数公式」「调节子」这些古典珍宝，推广到所有整数 $s = n$。<span class="marginnote">这一节是「从极限到大模型」里「局部规律 → 全局结构」的极致样本：Dedekind zeta 函数 $\zeta_F(s)$ 的一个值（$s = 1$ 处的留数）编成类数公式；K 理论说，$s = n$ 处的值也各自对应一个「高阶类数」（$K_{2n-2}$ 的大小）与「高阶调节子」（$K_{2n-1}$ 的 Borel 调节子）。</span>

这条线也是第 1 篇埋下的伏笔的回响：Dedekind 整环上 $K_0 = \mathbb{Z} \oplus \mathrm{Pic}$，理想类群从那时就住在 $K_0$ 里。现在我们要把这条故事线推向高潮。

## 1 数域与整数环

设 $F$ 是**数域**（$\mathbb{Q}$ 的有限扩张），$\mathfrak{o}_F$ 是它的**整数环**（$F$ 里在整数上整的元素构成）。记 $r_1$ 为实嵌入个数、$r_2$ 为复嵌入对数，则 $[F : \mathbb{Q}] = r_1 + 2r_2$。$F = \mathbb{Q}$ 时 $r_1 = 1, r_2 = 0$，$\mathfrak{o}_F = \mathbb{Z}$；$F = \mathbb{Q}(\sqrt{-1})$ 时 $r_1 = 0, r_2 = 1$，$\mathfrak{o}_F = \mathbb{Z}[i]$。

$\mathfrak{o}_F$ 是 **Dedekind 整环**：每个理想唯一分解成素理想之积，但元素未必唯一分解——这个「理想与元素的差距」正是理想类群。K 理论从这里接棒，因为 Dedekind 整环上的一切都已被 $K_0, K_1$ 精确刻画。

## 2 K₀ 与理想类群

第 1 篇我们见过一个表格：Dedekind 整环的 $K_0 = \mathbb{Z} \oplus \mathrm{Pic}$。对 $\mathfrak{o}_F$，$\mathrm{Pic}(\mathfrak{o}_F)$ 正是**理想类群 $\mathrm{Cl}(F)$**，于是

$$
K_0\big(\mathfrak{o}_F\big) = \mathbb{Z}\ \oplus\ \mathrm{Cl}(F), \qquad
\widetilde K_0\big(\mathfrak{o}_F\big) = \mathrm{Cl}(F)
$$

**约化 K 群 $= $ 理想类群**——「秩为零却可能不自由的模」的障碍，与「理想不一定主理想」的障碍，是同一个东西。<span class="marginnote">用第 1 篇的语言：理想类群度量「$\mathfrak{o}_F$ 上有没有秩 0 的非平凡投射模」。类数 $h_F = |\mathrm{Cl}(F)|$ 是「唯一分解失败」的精确计数——$h_F = 1$ 当且仅当 $\mathfrak{o}_F$ 是 UFD。$F = \mathbb{Q}(\sqrt{-5})$ 时 $h_F = 2$，$2\cdot 3 = 6 = (1+\sqrt{-5})(1-\sqrt{-5})$ 的经典反例由此获得结构解释。</span>

**类数有限**是代数数论的定理（Minkowski）：$h_F \lt  \infty$。于是 $K_0(\mathfrak{o}_F) \otimes \mathbb{Q} = \mathbb{Q}$——类群只贡献挠，不贡献秩。

## 3 K₁、单位与 Dirichlet 定理

对整数环 $\mathfrak{o}_F$，$SK_1 = 0$，所以

$$
K_1\big(\mathfrak{o}_F\big) = \mathfrak{o}_F^\times
$$

**$K_1$ 就是单位群**。而单位群的形状由 **Dirichlet 单位定理**完全给出：

$$
\mathfrak{o}_F^\times \ \cong\ \mu_F\ \times\ \mathbb{Z}^{\,r_1 + r_2 - 1}
$$

其中 $\mu_F$ 是 $F$ 里的单位根群（有限循环）。$F = \mathbb{Q}$：$\mathbb{Z}^\times = \{\pm 1\} = \mathbb{Z}/2$，指数 $0$；$F = \mathbb{Q}(\sqrt{2})$：$\mathbb{Z}[\sqrt 2]^\times \cong \{\pm1\} \times \mathbb{Z}$，由 $1+\sqrt2$ 生成。<span class="marginnote">单位定理是「秩 $r_1+r_2-1$ 的自由阿贝尔群」——这个「$-1$」来自「对数嵌入的迹为零」约束。Borel 调节子（第 5 节）正是把 $K_1$ 的秩公式推广到 $K_{2n-1}$：秩从 $r_1+r_2-1$ 变成 $r_1+r_2$（$n$ 偶）或 $r_2$（$n$ 奇）。</span>

**对账**：$K_1 \otimes \mathbb{Q}$ 的秩 $= r_1 + r_2 - 1$，与 Dirichlet 的指数一致。**$K_1$ 把单位定理重新表述了一遍**——而它还有余地，因为 K 理论从不挑环。

## 4 公式解析：类数公式

古典数论里最著名的一条公式，是 **Dedekind zeta 函数 $\zeta_F(s)$ 在 $s=1$ 处的留数**：

$$
\operatorname*{Res}_{s=1} \zeta_F(s) = \frac{2^{\,r_1}\, (2\pi)^{\,r_2}\ h_F\ R_F}{w_F\ \sqrt{|d_F|}}
$$

**第一步，读每个符号**：$h_F = |\mathrm{Cl}(F)|$ 是**类数**（= $|\widetilde K_0(\mathfrak{o}_F)|$）；$R_F$ 是**（Dirichlet）调节子**——单位格 $\mathbb{Z}^{r_1+r_2-1}$ 嵌入 $\mathbb{R}^{r_1+r_2}$ 后张成格的体积；$w_F = |\mu_F|$ 是单位根个数；$d_F$ 是判别式。<span class="marginnote">$2^{r_1}(2\pi)^{r_2}/\sqrt{|d_F|}$ 是「几何因子」——由 $F$ 的 Minkowski 嵌入的体积决定。整条公式把「解析对象（$\zeta_F$ 的留数）」与「代数对象（$h_F, R_F, w_F$）」焊接在一起，是类数有限性、单位定理与判别式理论的统一出口。</span>

**第二步，看 $h_F$ 的 K 理论身份**：$h_F = |\widetilde K_0(\mathfrak{o}_F)|$——类数就是「约化 K 群的大小」。**类数公式的第一项，是 $K_0$ 的挠**。

**第三步，看 $R_F$ 的 K 理论身份**：$R_F$ 是「$K_1(\mathfrak{o}_F) \otimes \mathbb{Q}$（秩 $r_1+r_2-1$ 的格）嵌入实空间后的体积」——**调节子是 $K_1$ 的「格不变量」**。类数公式的第二项，是 $K_1$ 的「体积」。

**第四步，读出纲领**：类数公式 = **$K_0$ 的挠 × $K_1$ 的体积 = $\zeta_F$ 的留数**。K 理论的纲领立刻浮现：把 $K_0, K_1$ 换成 $K_{2n-2}, K_{2n-1}$，把 $s=1$ 换成 $s=n$，公式是否仍然成立？下一节给出答案。

## 5 高阶 K 群、Borel 调节子与 Lichtenbaum 猜想

**Borel 定理（1974）** 把单位定理推广到一切奇数阶：

$$
\operatorname{rank}_{\mathbb{Z}} K_{2n-1}\big(\mathfrak{o}_F\big) = \begin{cases} r_1 + r_2, & n \text{ 偶} \\ r_2, & n \text{ 奇} \end{cases}, \qquad
K_{2n}\big(\mathfrak{o}_F\big) \ \text{有限} \ (n \ge 1)
$$

且 Borel 构造了**高阶调节子** $R_n$（$K_{2n-1}(\mathfrak{o}_F)$ 的格嵌入体积），证明 $\zeta_F(n)$ 是 $R_n$ 的**有理数倍**（对 $n \ge 2$）：<span class="marginnote">$n=1$ 时 $R_1 = R_F$ 是 Dirichlet 调节子；$n$ 偶时秩升到 $r_1+r_2$、$n$ 奇时秩掉到 $r_2$——「$-1$ 被吃掉」的机理是 $K_1$ 里那个迹零约束在 $n \ge 2$ 时消失。$\zeta_F(n)$ 的<strong>无理部分</strong>全部由 $R_n$ 承载。</span>

$$
\zeta_F(n) \ \sim\ R_n \times \frac{\big|K_{2n-2}\big(\mathfrak{o}_F\big)\big|}{\big|K_{2n-1}\big(\mathfrak{o}_F\big)_{\mathrm{tor}}\big|} \qquad (n \ge 2)
$$

**逐项对账**：右侧第一因 $R_n$（$K_{2n-1}$ 的体积，Borel 已证），第二因是「高阶类数」$|K_{2n-2}(\mathfrak{o}_F)|$ 与「高阶单位群的挠」之比。这正是 **Lichtenbaum 猜想**（已被 Bloch–Kato、Voevodsky、Rost 在大量情形证明）：**$\zeta_F(n)$ 的有理部分是「$K_{2n-2}$ 的大小 / $K_{2n-1}$ 的挠」**——类数公式在 $s = n$ 处的完整推广。

**实例（$F = \mathbb{Q}$）**：$\zeta(n)$ 在偶数处是 $\pi^n$ 的有理倍数，在奇数处（$n \ge 3$）至今神秘（Apéry 只证明了 $\zeta(3)$ 无理）。K 理论给出另一半：

$$
K_3(\mathbb{Z}) = \mathbb{Z}/48, \qquad K_5(\mathbb{Z}) = \mathbb{Z}/240, \qquad K_7(\mathbb{Z}) = \mathbb{Z}/240
$$

这些奇数 K 群的挠，由 **Bernoulli 数**（$\zeta$ 在负奇数的值，经函数方程翻过来）控制：$|K_{4m-1}(\mathbb{Z})|$ 的素数因子恰是「$p \mid B_{2m}$ 分子」的那些 $p$。**$\zeta$ 的秘密藏在 K 群的大小里**。<span class="marginnote">这解释了开篇那句话：K 理论把「类数公式」从 $s=1$ 逐格推到所有整数 $s=n$。Lichtenbaum 猜想、Tamagawa 数猜想、主猜想（Iwasawa 理论）与 K 群在分圆塔里的行为，把「解析特殊值」与「代数 K 群」焊成一张大网——现代算术的许多分支都在织这张网。</span>

### 术语速查表：数论与 K 理论

| 记号 | 名称 | 含义 |
| --- | --- | --- |
| $\mathfrak{o}_F$ | 整数环 | $F$ 中整元；Dedekind 整环 |
| $r_1, r_2$ | 嵌入计数 | 实嵌入 / 复嵌入对数 |
| $\mathrm{Cl}(F)$ | 理想类群 | $=\widetilde K_0(\mathfrak{o}_F)$ |
| $h_F$ | 类数 | $|\mathrm{Cl}(F)|$，有限 |
| $\mu_F$ | 单位根群 | 有限循环 |
| $R_F$ | Dirichlet 调节子 | 单位格的体积 |
| $R_n$ | Borel 调节子 | $K_{2n-1}$ 格嵌入的体积 |
| $d_F$ | 判别式 | Minkowski 嵌入的体积因子 |

**辨析｜易错点：** $K_{2n-1}(\mathfrak{o}_F)$ 的秩公式里，「$n$ 偶时 $r_1+r_2$、$n$ 奇时 $r_2$」指的是 $n \ge 2$ 的情形；$n=1$ 是例外——$K_1 = \mathfrak{o}_F^\times$ 的秩是 $r_1+r_2-1$，那个「$-1$」是 $K_1$ 独有的迹零约束。把 $n=1$ 硬套进高阶公式是最常见的错。

## 6 小结

- **$K_0(\mathfrak{o}_F) = \mathbb{Z} \oplus \mathrm{Cl}(F)$**：约化 K 群就是理想类群，类数 $h_F = |\widetilde K_0|$。
- **$K_1(\mathfrak{o}_F) = \mathfrak{o}_F^\times$**：$K_1$ 就是单位群，Dirichlet 定理给出 $\mathfrak{o}_F^\times \cong \mu_F \times \mathbb{Z}^{r_1+r_2-1}$。
- **类数公式**：$\operatorname{Res}_{s=1}\zeta_F(s) = 2^{r_1}(2\pi)^{r_2} h_F R_F / (w_F \sqrt{|d_F|})$——$K_0$ 的挠 × $K_1$ 的体积。
- **Borel 定理**：$\operatorname{rank} K_{2n-1}(\mathfrak{o}_F) = r_1+r_2$（$n$ 偶）或 $r_2$（$n$ 奇）；$K_{2n}$ 有限。
- **Borel 调节子 $R_n$**：$\zeta_F(n)$ 的无理部分由 $R_n$ 承载；**Lichtenbaum 猜想**：$\zeta_F(n)$ 的有理部分 $= |K_{2n-2}|/|K_{2n-1}^{\mathrm{tor}}|$。
- **实例**：$K_3(\mathbb{Z}) = \mathbb{Z}/48$，$K_5 = K_7 = \mathbb{Z}/240$，挠由 Bernoulli 数控制。

在最后一节，我们把镜头拉回「拓扑 K 理论」——这门理论的另一半江山——并看到 Bott 周期性如何让 $K^0(X)$
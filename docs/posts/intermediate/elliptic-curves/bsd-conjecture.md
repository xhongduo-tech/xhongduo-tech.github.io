---
title: BSD 猜想
date: 2026-08-11
---

# BSD 猜想

<div class="epigraph">
<p>一个函数在一点的值，居然应该说出整条曲线上有理点的全部家底——这几乎太过美好，以至于它必须是真的。</p>
<footer>—— 布赖恩 · 伯奇（Bryan Birch）与彼得 · 斯温纳顿-戴尔（Peter Swinnerton-Dyer）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 椭圆曲线与模形式 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 BSD 猜想开始

七千年来，丢番图方程的核心问题是「有多少解」；而椭圆曲线把「解」组织成群 $E(\mathbb{Q})$（第 7 篇），其中有两个待定量：**秩 $r$**（自由部分维数）与 **Sha 群**（「有解但看不出来」的障碍）。BSD 猜想说：**这两个纯算术量，恰好被 L-函数 $L(E,s)$ 在 $s=1$ 处的行为读出**。<span class="marginnote">Birch 与 Swinnerton-Dyer 在 1960 年代用早期计算机（EDSAC）统计几百条曲线的秩与 $L$-函数的关系，发现了这条著名的经验律：$\prod_{p \leq X} \#E(\mathbb{F}_p)/p \sim C(\log X)^r$——「点个数的乘积像 $(\log X)^r$ 一样生长，指数正是秩」。它从「大量数据」里淘出的金子，至今是千禧年七大难题（Clay）之一。</span>本节主题：Hasse-Weil L-函数、弱 BSD（零点阶数 = 秩）、强 BSD（Taylor 系数公式）与它们的数值证据。

## 1 Hasse-Weil L-函数：把「所有素数」写成一个函数

### 定义

对 $E/\mathbb{Q}$ 导子 $N$，定义 **Hasse-Weil L-函数**

$$L(E, s) = \prod_{p} L_p(s)^{-1}, \qquad L_p(s) = \begin{cases} 1 - a_p p^{-s} + p^{1-2s} & p \nmid N \\[4pt] 1 - a_p p^{-s} & p \mid N \ (\text{分裂乘法}) \\[4pt] 1 + a_p p^{-s} & p \mid N \ (\text{非分裂乘法}) \\[4pt] 1 & p \mid N \ (\text{加法型}) \end{cases}$$

其中 $a_p = p + 1 - \#E(\mathbb{F}_p)$（第 5 篇）。坏素数处的局部因子来自约化类型（第 6 篇）：乘法型保留「一次因子」，加法型整个消失。<span class="marginnote">这个 L-函数把「曲线在每一个素数处的寿命」汇总成一个函数：好素数贡献「二次因子」，坏素数贡献「一次或零」。它的定义方式与第 12 篇模形式 L-函数完全同构（$k=2$），这正是模性定理（第 17 篇）「对表」的前提。</span>

### 解析延拓与中心点

**定理（Wiles-Breuil-Conrad-Diamond-Taylor，模性定理的推论）**：$L(E,s)$ 延拓为整函数，且满足

$$\Lambda(E, s) := \big(\tfrac{\sqrt{N}}{2\pi}\big)^s \Gamma(s)\, L(E,s) = \varepsilon\, \Lambda(E, 2-s)$$

**中心点是 $s = 1$**（对应模形式中心点 $s = k/2 = 1$）。BSD 猜想的全部赌注都押在「$s=1$」这一个点上。<span class="marginnote">在模性定理证明前，$L(E,s)$ 的解析延拓是不完备的（只知道在 $\mathrm{Re}(s) > 3/2$ 收敛）。是「模形式 → L-函数延拓」（Hecke，第 12 篇）把椭圆曲线 L-函数「洗白」成整函数——BSD 猜想的所有精确陈述都依赖这一洗白。没有模性，BSD 连「在 $s=1$ 求值」都无定义。</span>

## 2 弱 BSD：零点阶数 = 秩

### 陈述

**弱 BSD 猜想**：$L(E,s)$ 在 $s = 1$ 处的零点阶数等于 $E$ 的秩：

$$\mathrm{ord}_{s=1}\, L(E,s) = \mathrm{rank}\, E(\mathbb{Q})$$

**重点：这是一个「解析对象」与「算术对象」的等式**——左边是可微函数的零点阶数，右边是群的自由生成元个数。它们是两种完全不同的数学，而猜想断言它们恒等。<span class="marginnote">直觉来源：$\#E(\mathbb{F}_p) \approx p + 1$ 时 $a_p$ 很小，L-函数「靠近」$L(\mathbb{G}_m, s) = \zeta(s-1)^{-1}$？实际上正确的类比是「秩 = 乘积的生长指数」。Birch-Swinnerton-Dyer 的经验公式 $\prod_{p\leq X} \frac{\#E(\mathbb{F}_p)}{p} \sim (\log X)^r \cdot C$ 正是「欧拉积在 $s=1$ 的对数发散度 = 秩」的经验版。</span>

### 已知部分

- 若 $L(E,1) \neq 0$，则 $\mathrm{rank}\, E(\mathbb{Q}) = 0$（**Coates-Wiles、Kolyvagin 等**）：「L-函数在中心点非零 ⇒ 只有挠点」。这是「弱 BSD 的一半」。
- 「$L(E,1) = 0 \Rightarrow$ 秩 ≥ 1」即「零点存在 ⇒ 有点」——目前**未证明**（这是「Gross-Zagier + Kolyvagin」方法能处理秩 ≤ 1 的原因）。

**辨析｜易错点：** 「秩 0」不是「没有点」，而是「点全部是挠点」（$E(\mathbb{Q})$ 有限）。「$L(E,1) \neq 0$ 推出秩 0」是说「有无限多个点 ⇒ L-函数必须为零」——**逻辑方向是「点多 ⇒ 零」，不是「零 ⇒ 点多」**。后者（零点阶数 ≥ 1 ⇒ 秩 ≥ 1）是更难的，至今未完全证明。

## 3 强 BSD：首项系数的精细账目

### 陈述

若 $\mathrm{rank} = r$，则 $L(E,s)$ 在 $s=1$ 的 Laurent/Taylor 展开首项系数满足

$$\frac{L^{(r)}(E, 1)}{r!} = \frac{\Omega_{E} \cdot \operatorname{Reg}(E) \cdot \prod_p c_p \cdot \#\text{Sha}(E)}{|E(\mathbb{Q})_{\mathrm{tors}}|^2}$$

逐项解读：

- $\Omega_E$：**周期**——$E(\mathbb{C}) \cong \mathbb{C}/\Lambda$（第 15 篇）中「实格」的周长积分，$\Omega = \int_{E(\mathbb{R})} \frac{dx}{2y+ a_1 x + a_3}$。
- $\operatorname{Reg}(E)$：**Néron-Tate 规范高度矩阵的行列式**（第 7 篇的 $\widehat{h}$）——它度量「秩的方向」在高度几何里的体积。
- $c_p$：**Tamagawa 数**（第 6 篇）——坏素数处「约化分量数」。
- $\#\text{Sha}(E)$：**Tate-Shafarevich 群的大小**——「所有局部处处有解、但整体无解」的障碍。
- 分母：挠点个数的平方。

**重点：Sha 群是公式里唯一「看不见」的量**——其他各项都能计算，唯独 $\#\text{Sha}$ 要由「右边 = 左边」反推。<span class="marginnote">Tate-Shafarevich 群 $\text{Sha}(E)$ = 「被所有局部解包围但整体无解」的元素的集合。它的元素是「曲线 $E$ 的一个扭」（torsor），在每一个 $\mathbb{Q}_p$ 和 $\mathbb{R}$ 上都有点，但整体上没有。BSD 公式实际上「定义了」$\#\text{Sha}$——它把「看不见的障碍」变成了「可算的余数」。$\text{Sha}$ 的有限性本身就是未解决的大猜想。</span>

### 与「秩 = 零点阶数」的相容

当 $r = 0$：$L(E,1) = \Omega \cdot \frac{\prod c_p \cdot \#\text{Sha}}{|E_{\mathrm{tors}}|^2}$——一个「纯数」公式。当 $r \geq 1$：左边「从零开始生长」，右边 $\operatorname{Reg}$ 与 $r$ 一起「从零开始生长」——**两边同时变零，才有真正的等式**。

## 4 公式解析：$\prod_p c_p$ 如何「记账」

Tamagawa 数在强 BSD 公式里逐个素数累加，把它算清楚。

$$
\frac{L^{(r)}(E,1)}{r!} = \frac{\Omega \cdot \operatorname{Reg} \cdot \big(\prod_p c_p\big) \cdot \#\text{Sha}}{|E_{\mathrm{tors}}|^2}
$$

- **第一步，$\prod_p c_p$ 是「坏账清单」**：好素数处 $c_p = 1$（第 6 篇），只有坏素数贡献非平凡值。于是 $\prod_p c_p$ 是「曲线在每个坏素数处欠下的『有限项』」——它等于「Kodaira 类型的总账」。例如 $E: y^2 = x^3 - x$（$\Delta = 64$）的坏素数只有 $p = 2$ 一处，其余素数的 $c_p$ 全为 1——于是 $\prod_p c_p = c_2$ 只含一个非平凡因子。
- **第二步，分母 $|E_{\mathrm{tors}}|^2$ 的由来**：BSD 的「规范高度」与「配对」计算把「挠点」作为「分母」扫除——它修正「每个秩方向被挠点周期性重复」的冗余。对 $y^2 = x^3 - x$，$E(\mathbb{Q})_{\mathrm{tors}} \cong \mathbb{Z}/2 \times \mathbb{Z}/2$，分母 $= 4$。
- **第三步，$\operatorname{Reg}$ 的几何**：$\operatorname{Reg}$ 是「秩方向张成的平行体」在规范高度下的体积——**秩 0 时 $\operatorname{Reg} = 1$（空体积）**，秩 ≥ 1 时它「从 0 生长」并抵消左边的零点。
- **第四步，一段完整的核对**：对 $E: y^2 = x^3 - x$（秩 0，$E(\mathbb{Q})_{\mathrm{tors}} \cong \mathbb{Z}/2 \times \mathbb{Z}/2$，故 $|E_{\mathrm{tors}}| = 4$、分母为 $4^2 = 16$）：坏素数只有 2，其 Tamagawa 数 $c_2$ 由 Kodaira 类型读出（加法型的小整数）。公式断言 $L(E,1) = \Omega\cdot c_2\cdot\#\text{Sha}/16$。数值上把 $L(E,1)$ 与 $\Omega$ 都算到高位，代入 $c_2$ 后反解出 $\#\text{Sha} = 1$（此曲线 Sha 平凡）。**强 BSD 的每个非平凡项都能被「算」出来，唯独 $\#\text{Sha}$ 要由等式「定义」**。

## 5 BSD 的疆域与证据

- **秩 0 / 1 的已验证部分**：Gross-Zagier + Kolyvagin（1980s）证明「$\mathrm{ord}_{s=1} L \leq 1 \Rightarrow$ 秩 = 零点阶数，且 $\#\text{Sha}$ 有限」——秩 0、1 的情形「几乎」完全解决。
- **数值与大数据**：Birch-Swinnerton-Dyer 的统计、现代「计算机代数」（SageMath、Magma）对数万条曲线核对强 BSD——**零反例**。<span class="marginnote">计算验证的思路：用「模形式的数值方法」（第 12 篇 L-函数）把左边算到任意精度，用「约化 + 高度」把右边各项算出来，最后「解出 $\#\text{Sha}$ 是否为 1 或小整数」。已知最大秩的曲线（秩 ≥ 28）也满足「数值上的零点阶数 ≥ 28」——虽非证明，却是强烈的信号。</span>
- **尚未解决的核心**：$\#\text{Sha}$ 的有限性、秩 ≥ 2 的 BSD、以及「秩与零点阶数」的一般性——这是千禧年问题的全部含金量。
- **与主线的关系**：BSD 是「从极限到大模型」这条主线里「极限」的极致——**L-函数在 $s=1$ 的极限行为（零点阶数、Taylor 系数）编码了「无穷多个素数的点个数」的全体信息**。解析延拓让「无限」变得可微，BSD 让「可微」说出「有限」。

### 补充：如何数值「核对」强 BSD——一条秩 0 曲线的完整流程

强 BSD 在秩 0 时退化为一个纯数等式 $L(E,1) = \Omega\cdot\prod c_p\cdot\#\mathrm{Sha}/|E_{\mathrm{tors}}|^2$，数值核对分五步：

1. **定秩**：由数值 L-函数在 $s=1$ 处非零（或 Gross-Zagier 型工具）确定秩为 0。
2. **数挠点**：Nagell-Lutz 有限枚举 + Mazur 清单 → $|E_{\mathrm{tors}}|$（例如 $\mathbb{Z}/2\times\mathbb{Z}/2$ 时 $|E_{\mathrm{tors}}|=4$，平方为 16）。
3. **读坏账**：极小方程 → Kodaira 类型 → 每个坏素数的 Tamagawa 数，求积 $\prod c_p$（好素数全为 1）。
4. **算周期**：$\Omega = \int_{E(\mathbb{R})} dx/(2y)$ 数值积分（第 15 篇）。
5. **对表**：用模形式的 L-函数方法（第 12 篇）把 $L(E,1)$ 算到高精度，代入等式反解出 $\#\mathrm{Sha}$——理想结果是 1 或很小的完全平方数。

对大量秩 0 曲线（现代数据库如 LMFDB 已覆盖数十万条），第 5 步得到的 $\#\mathrm{Sha}$ 全部落在「完全平方数 × 小因子」的范围内——与「$\mathrm{Sha}$ 是有限群」的预期吻合。**这条流水线正是 BSD 数值证据的日常来源：五个可算的块，加上一个「由等式定义」的未知量。** 值得注意的是，第 2 步「数挠点」的机械化（Nagell-Lutz + Mazur）与第 3 步「读坏账」（Kodaira 分类）都来自本专题前几篇——BSD 的数值核对，是整本书工具箱的联合演习。

**最后提醒一点「方向感」**：弱 BSD（零点阶数 = 秩）说的是「从解析到算术」；强 BSD（首项系数公式）说的是「从解析的精细结构到算术的精细结构」。两者都未证明，但已知的秩 0/1 结果（Coates-Wiles、Gross-Zagier、Kolyvagin）已经把「BSD 的一半」坐实——**这个猜想离「完全」的距离，也正是这个领域最有价值的未知。**

## 6 小结

- **Hasse-Weil L-函数** $L(E,s) = \prod_p L_p(s)^{-1}$ 把每个素数的点个数汇总为一个整函数（模性定理保证延拓）。
- **弱 BSD**：$\mathrm{ord}_{s=1}L(E,s) = \mathrm{rank}\,E(\mathbb{Q})$——「零点阶数 = 秩」。
- **强 BSD**：首项系数 $= \Omega\cdot\operatorname{Reg}\cdot\prod c_p\cdot\#\text{Sha} / |E_{\mathrm{tors}}|^2$——每个因子都有几何或算术含义。
- 已知进展：**秩 0/1 情形几乎解决**（Coates-Wiles、Gross-Zagier、Kolyvagin）；一般情形是千禧年问题。
- $\#\text{Sha}$ 是「局部处处有解但整体无解」的障碍，其有限性是独立的大猜想。

在下一节，我们把全书的天平压实：**模性定理（Taniyama-Shimura-Weil）**——为什么「每条 $\mathbb{Q}$ 上的椭圆曲线都来自一个权 2 模形式」，而这条断言又如何一夜之间解决了费马大定理。

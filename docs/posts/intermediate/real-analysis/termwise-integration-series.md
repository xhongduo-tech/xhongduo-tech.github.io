---
title: 逐项积分与级数的积分
date: 2026-08-07
---

# 逐项积分与级数的积分

<div class="epigraph">
<p>级数与积分能否交换，是分析学的灵魂问题——Lebesgue 理论给出最清晰的答案图谱。</p>
<footer>—— 恩里科 · 波莱尔（Émile Borel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从逐项积分开始

Levi 定理的一则推论是「非负级数可逐项积分」，本节把它作为独立主题展开：**函数项级数 $\sum_kf_k$ 的积分何时等于各项积分之和？** 这是傅里叶级数、幂级数、概率母函数里最常用的运算——「逐项积分」。数学分析里每次逐项积分都要验证「一致收敛」，Lebesgue 理论把这套手续大幅简化。

本节给出三种「逐项积分」的合法条件：**非负**（Levi 直接许可）、**绝对可积的控制**（控制收敛的雏形）、以及**$\sigma$-有限积分的分片逐项**。它们构成一张完整的「何时可以逐项积分」的地图。<span class="marginnote">逐项积分在<strong>傅里叶级数理论</strong>中举足轻重：$\int\sum a_n\varphi_n=\sum a_n\int\varphi_n$ 的成立与否，决定级数表示能否积分。Lebesgue 的三大定理让「先验地」逐项积分成为常态，而非例外——这是实变函数论对调和分析的直接馈赠。</span>

## 1 非负级数的逐项积分

**定理（非负逐项积分）**：设 $\{f_k\}$ 是非负可测函数列。则

$$\int_E\left(\sum_{k=1}^{\infty}f_k\right)dm=\sum_{k=1}^{\infty}\int_Ef_k\,dm$$

**证明**：部分和 $S_N=\sum_{k=1}^Nf_k$ 单调递增（非负），$S_N\uparrow\sum_kf_k$。由 Levi 单调收敛：

$$\int\sum_kf_k=\int\lim_NS_N=\lim_N\int S_N=\lim_N\sum_{k=1}^N\int f_k=\sum_{k=1}^{\infty}\int f_k$$

**重点：这个定理是「无条件」的——不要求级数收敛，不要求各项可积，不要求一致收敛。** 非负性让「$\infty$」也合法。例：$f_k=\chi_{\{k\}}$ 在 $\mathbb{R}$ 上，$\sum_kf_k=\chi_{\mathbb{N}}$，两边都等于 $0$（可数集零测）；$f_k=\tfrac1{k^2}\chi_{\mathbb{R}}$，$\sum_kf_k$ 常等于 $\tfrac{\pi^2}{6}$，积分两边都等于 $\infty$（非常数函数在无穷区间上积分无穷）。

**例（实用）**：$\int_0^1\sum_{k=0}^\infty x^k dx=\int_0^1\tfrac1{1-x}dx=\infty$，而 $\sum_k\int_0^1x^kdx=\sum_k\tfrac1{k+1}=\infty$——两边都是无穷，非负逐项积分合法地给出「$\infty=\infty$」。

## 2 控制情形与一般情形的逐项积分

**定理（可积控制下逐项积分）**：设 $\{f_k\}$ 可测，$\sum_k\int|f_k|<\infty$。则

- $\sum_kf_k$ 几乎处处绝对收敛（收敛到 a.e. 有限的函数）；
- $\sum_kf_k\in L^1$，且逐项积分成立：$\int\sum_kf_k=\sum_k\int f_k$。

**证明**：$\int\sum_k|f_k|=\sum_k\int|f_k|<\infty$（非负逐项），故 $\sum_k|f_k|$ a.e. 有限，绝对收敛。$|\sum_kf_k|\le\sum_k|f_k|\in L^1$，被控制原则给可积性。积分交换由控制收敛（或对 $\sum_{k=1}^N$ 取极限，被 $\sum|f_k|\in L^1$ 控制）。

**辨析｜易错点：条件 $\sum_k\int|f_k|<\infty$ 是「绝对收敛级数」的积分版。** 它比「$\sum_kf_k$ 逐点收敛」强：要求「各项绝对值的积分和收敛」。例：$f_k=\tfrac{(-1)^k}{k}\chi_{[k,k+1]}$，$\sum_kf_k$ 在 $[1,\infty)$ 上逐点条件收敛（交错调和），但 $\sum_k\int|f_k|=\sum_k\tfrac1k=\infty$——**逐项积分不合法**（L 意义下 $\int\sum_kf_k$ 未定义，因为不绝对可积）。这再次呼应「Lebesgue 只认绝对」的纪律。<span class="marginnote">「$\sum_k\int|f_k|<\infty$」是「绝对收敛」在函数空间的正确推广：它保证「交换求和与积分」安全，因为一切都由 $\sum|f_k|\in L^1$ 这个「大伞」罩住。这个条件也是<strong>概率论中「$E[\sum X_n]=\sum E[X_n]$」的标准判据</strong>（Fubini/Tonelli 的级数版）。</span>

## 3 分片逐项积分

**定理（分片逐项积分）**：设 $E=\bigcup_kE_k$（不相交可测），$\{f_k\}$ 可测且 $f=\sum_kf_k\chi_{E_k}$（各块函数不同）。若各块「整体可积」条件满足，则 $\int_Ef=\sum_k\int_{E_k}f_k$。

这是「区域可数可加性」（第五篇）与「级数逐项积分」的合成：把区域拆成可数片，各片用不同函数，总积分是各片积分之和。它统一处理了「不同区域不同规则」的分片函数。

**应用（概率的期望可加）**：随机变量 $X=\sum_kX_k$（可数分解），若 $\sum_kE[|X_k|]<\infty$，则 $E[X]=\sum_kE[X_k]$——**期望的可数可加性**，概率论的核心计算工具。

## 4 公式解析：$\sum_k\int|f_k|<\infty$ 的「大伞」作用

控制情形逐项积分的证明靠「被 $\sum|f_k|$ 控制」：

$$\left|\sum_{k=1}^{N}f_k\right|\le\sum_{k=1}^{N}|f_k|\le\sum_{k=1}^{\infty}|f_k|=:g\in L^1$$

- **第一步，读「$g=\sum_k|f_k|\in L^1$」**：由非负逐项积分，$\int g=\sum_k\int|f_k|<\infty$。**$g$ 是一个可积的「大伞」**，罩住所有部分和。
- **第二步，读「控制收敛的适用」**：$S_N=\sum_{k=1}^Nf_k\to\sum_kf_k$ 逐点（a.e.），且 $|S_N|\le g\in L^1$。由 Lebesgue 控制收敛，$\int S_N\to\int\sum_kf_k$。**「可积大伞 + 逐点收敛」自动给出极限交换**——无需一致收敛。
- **第三步，读「与条件收敛的分界」**：若 $\sum_k\int|f_k|=\infty$（如 $f_k=\tfrac{(-1)^k}k\chi_{[k,k+1]}$），$g\notin L^1$，大伞失效，逐项积分非法。**「绝对收敛级数」是逐项积分的分水岭**。

**「大伞控制 + 控制收敛」**，是把「级数极限」变成「可交换积分」的标准通道——它把「一致收敛」的旧条件替换为「可积控制」的新条件，宽松得多。

## 5 小结

- **非负逐项**：$\int\sum_kf_k=\sum_k\int f_k$（无条件，允许无穷）——Levi 直接推论。
- **控制逐项**：$\sum_k\int|f_k|<\infty$ ⇒ 绝对收敛 + $L^1$ + 逐项积分。
- **分片逐项**：区域可数分解 + 各片积分相加（期望可加性）。
- **纪律**：条件收敛级数不合法（$\tfrac{(-1)^k}k\chi_{[k,k+1]}$ 反例）。
- **工具**：大伞 $g=\sum|f_k|\in L^1$ + 控制收敛，替代一致收敛。

在下一节，我们介绍三大定理的第二位：**Fatou 引理**——从「非单调」的困境中抢救出「≤」的智慧。

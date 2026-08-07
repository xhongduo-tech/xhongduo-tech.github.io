---
title: Lᵖ 范数与依范数收敛（p 次平均收敛）
date: 2026-08-07
---

# Lᵖ 范数与依范数收敛（p 次平均收敛）

<div class="epigraph">
<p>收敛不再发生在每个点上，而发生在「平均的能量」里——这是函数空间的呼吸方式。</p>
<footer>—— 弗里杰什 · 里斯（Frigyes Riesz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第七章 ｜ 2026-08-07</p>
</div>

## 为什么从 Lᵖ 范数与依范数收敛开始

$L^p$ 空间配上范数后，最自然的收敛概念是**依范数收敛**（$p$ 次平均收敛）：$\|f_k-f\|_p\to0$。它是「函数列在平均意义下趋近」的精确化——与逐点收敛、a.e. 收敛、依测度收敛并列，构成收敛模式的完整谱系。本节研究 $L^p$ 收敛的性质、与其他收敛的关系、以及它的度量结构。

理解 $L^p$ 收敛的关键是「**整体 vs 逐点**」的张力：$L^p$ 收敛不关心单点行为，只关心「能量」（$p$ 次幂积分）是否趋零。它比 a.e. 收敛「更整体」，比依测度收敛「更强」——三条收敛模式的关系图在本节展开。<span class="marginnote">$L^p$ 收敛在概率论中就是「$L^p$ 收敛」：$X_n\overset{L^p}{\to}X$ 即 $E|X_n-X|^p\to0$。<strong>$L^p$ 收敛蕴含依概率收敛</strong>（Markov 不等式），这是大数定律的强弱版本分界；而「$L^p$ 收敛但不 a.s. 收敛」的例子（移动冒泡）说明逐点与平均的分道扬镳。</span>

## 1 依范数收敛的定义

**定义（$p$ 次平均收敛）**：设 $\{f_k\}\subset L^p$，$f\in L^p$，$1\le p<\infty$。若

$$\|f_k-f\|_p=\left(\int_E|f_k-f|^p\,dm\right)^{1/p}\longrightarrow0\quad(k\to\infty)$$

则称 $\{f_k\}$ **$p$ 次平均收敛**到 $f$，记 $f_k\overset{L^p}{\to}f$。这是 $L^p$ 中由范数诱导的收敛。

**p=2 特例**：$\|f_k-f\|_2^2=\int|f_k-f|^2$——均方收敛，概率论与信号处理的常用收敛（$L^2$ 收敛）。

**例**：$f_k=\chi_{[0,1/k]}$（$k$ 倍的？）——取 $f_k=\sqrt{k}\chi_{[0,1/k]}$：$\|f_k\|_2^2=k\cdot\tfrac1k=1$ 不趋零，不 $L^2$ 收敛；但 $\|f_k\|_1=\sqrt{k}/k=1/\sqrt{k}\to0$，$L^1$ 收敛。**不同 $p$ 的收敛不同**。

**重点：$L^p$ 收敛是「能量趋零」，不是「逐点趋零」。** $\|f_k-f\|_p\to0$ 意味着「$|f_k-f|^p$ 的积分」趋零——函数值可以在单点任意大，只要「大值的区域」收缩到零测（且不贡献能量）。

## 2 $L^p$ 收敛的性质

**性质一（极限唯一 a.e.）**：$f_k\overset{L^p}{\to}f$ 且 $\to g$ ⇒ $f=g$ a.e.（范数的分离性）。

**性质二（线性保持）**：$f_k\overset{L^p}{\to}f$、$g_k\overset{L^p}{\to}g$ ⇒ $f_k+g_k\overset{L^p}{\to}f+g$、$cf_k\overset{L^p}{\to}cf$（Minkowski 与齐次性）。

**性质三（范数连续）**：$f_k\overset{L^p}{\to}f$ ⇒ $\|f_k\|_p\to\|f\|_p$（反三角不等式）。

**性质四（收敛子列）**：$f_k\overset{L^p}{\to}f$ 且 $m(E)<\infty$ 时，存在子列 a.e. 收敛到 $f$（由 $L^p\Rightarrow$ 依测度 + Riesz 定理）。**「$L^p$ 收敛 ⇒ 子列 a.e. 收敛」**——这是常用的「从平均到逐点」桥梁。

## 3 与依测度收敛、a.e. 收敛的关系

**定理（$L^p$ ⇒ 依测度）**：$f_k\overset{L^p}{\to}f$ ⇒ $f_k\overset{m}{\to}f$。证明用 Markov/Chebyshev：

$$m\left(\{|f_k-f|\ge\varepsilon\}\right)\le\frac{1}{\varepsilon^p}\int|f_k-f|^p\xrightarrow{k\to\infty}0$$

**关系图**（有限测度 $m(E)<\infty$ 时）：

$$L^p\ \text{收敛}\ \Rightarrow\ \text{依测度收敛}\ \Rightarrow\ \exists\ \text{a.e. 收敛子列}$$

（$L^p\Rightarrow$ 依测度用 Markov；依测度 ⇒ 子列 a.e. 用 Riesz 定理。）**$L^p$ 收敛是最强的，且总能「抽」出 a.e. 收敛子列。**

**反例（$L^p$ 收敛但 a.e. 不收敛）**：移动冒泡序列（$[0,1]$ 上依次点亮二分区间的指示函数）——$L^p$ 范数 $\to0$（每个函数的 $p$ 次积分 $\to0$），但处处不收敛。**「平均收敛」与「逐点收敛」彻底分家。**

**辨析｜易错点：$L^p$ 收敛不蕴含「整列 a.e. 收敛」，只蕴含「子列 a.e. 收敛」。** 移动冒泡整列处处不收敛，但存在 a.e. 收敛子列。**「子列」而非「整列」是 $L^p$ 收敛能承诺的全部逐点信息**——这与 Riesz 定理（依测度收敛的子列）一致。

## 4 公式解析：Markov 不等式桥

$L^p$ 收敛 ⇒ 依测度收敛的证明核心是 Markov 不等式：

$$m\left(\{|f_k-f|\ge\varepsilon\}\right)=\int\chi_{\{|f_k-f|\ge\varepsilon\}}\le\frac1{\varepsilon^p}\int|f_k-f|^p$$

- **第一步，读「指示函数的下界」**：在 $\{|f_k-f|\ge\varepsilon\}$ 上，$|f_k-f|^p\ge\varepsilon^p$，故 $\chi_{\{|f_k-f|\ge\varepsilon\}}\le\tfrac{|f_k-f|^p}{\varepsilon^p}$（逐点）。**「超阈值」被「幂函数」从上方控制**。
- **第二步，读「积分」**：两边积分，左边 $=m(\{\dots\})$（指示函数的积分是测度），右边 $=\tfrac1{\varepsilon^p}\int|f_k-f|^p=\tfrac{\|f_k-f\|_p^p}{\varepsilon^p}$。**「坏区测度 ≤ 能量/阈值$^p$」**——能量趋零，坏区测度趋零。
- **第三步，读「$p$ 的作用」**：阈值 $\varepsilon^p$ 在分母，能量 $\|f_k-f\|_p^p$ 在分子。**$p$ 次平均收敛直接给出「坏区」的 $p$ 次方控制**——$p$ 越大，控制越强（但要求也越高）。

**「Markov/Chebyshev 桥」**是「积分信息 → 测度信息」的标准转换——它在依测度收敛、大数定律、极大函数理论中反复出现。

## 5 小结

- **$p$ 次平均收敛**：$\|f_k-f\|_p\to0$——能量趋零的整体收敛。
- **性质**：极限唯一、线性、范数连续、子列 a.e. 收敛。
- **关系**：$L^p$ ⇒ 依测度（Markov）⇒ 子列 a.e.（Riesz）。
- **反例**：移动冒泡 $L^p$ 收敛但整列 a.e. 不收敛。
- **纪律**：「子列 a.e.」≠「整列 a.e.」——平均与逐点的张力本质。

在下一节，我们证明 $L^p$ 空间的**完备性**：Riesz–Fischer 定理——$L^p$ 是 Banach 空间。

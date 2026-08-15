---
title: 线性方程组的误差分析：条件数与病态方程组
date: 2026-08-07
---

# 线性方程组的误差分析：解有多可靠

<div class="epigraph">
<p>解方程组不是问题，知道解是否可信才是问题。</p>
<footer>—— 数值线性代数第一课</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§5.5 ｜ 2026-08-07</p>
</div>

## 为什么从线性方程组误差分析开始

前面的章节分别研究了「矩阵分解算法」与「范数」。本节把两者合起来回答一个工程核心问题：**$A\mathbf{x}=\mathbf{b}$ 的解有多可靠？** 误差有两个来源：输入数据 $\mathbf{b}$（或 $A$）带的扰动，以及算法舍入误差。两者都被**条件数**放大。理解条件数，是判断「要不要相信这个解」的唯一理性依据。<span class="marginnote">回顾：第一篇《病态问题与条件数》给出了条件数概念的一元函数与初步矩阵版本；本节在范数完备后把它做彻底——<strong>推导「解相对误差 ≤ cond(A) × 输入相对误差」的完整不等式，并处理扰动同时在 $A$ 与 $\mathbf{b}$ 上的情形</strong>。</span>

本节推导误差界、给出条件数的完整理论，并讨论「残差小但解错」的经典反直觉。

## 1 右端扰动：解对 b 的敏感度

设真解 $\mathbf{x}$ 满足 $A\mathbf{x}=\mathbf{b}$，扰动后 $A(\mathbf{x}+\Delta\mathbf{x})=\mathbf{b}+\Delta\mathbf{b}$。两式相减 $A\Delta\mathbf{x}=\Delta\mathbf{b}$，$\Delta\mathbf{x}=A^{-1}\Delta\mathbf{b}$。取范数：

$$
\lVert\Delta\mathbf{x}\rVert \le \lVert A^{-1}\rVert\,\lVert\Delta\mathbf{b}\rVert
$$

又 $\lVert\mathbf{b}\rVert\le\lVert A\rVert\lVert\mathbf{x}\rVert$，即 $\dfrac{1}{\lVert\mathbf{x}\rVert}\le\dfrac{\lVert A\rVert}{\lVert\mathbf{b}\rVert}$。两式相乘得**核心不等式**：

$$
\frac{\lVert\Delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} \le \mathrm{cond}(A)\,\frac{\lVert\Delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert}, \qquad \mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert
$$

**条件数把「输入的相对误差」放大成「解的相对误差」的上界。** $\mathrm{cond}(A)\ge1$，越大越病态。<span class="marginnote">这个推导我们在《条件数》篇已经走过一遍，这里在范数完备后重述——同样的不等式，现在每一步都有严格依据。<strong>重复推导不是浪费，是让「直觉」升级成「证明」</strong>。</span>

**一个可手算的例子**：$A=\begin{pmatrix}100&100\\100&100.01\end{pmatrix}$，$\lVert A\rVert_1=200.01$，$\lVert A^{-1}\rVert_1=20000$，故 $\mathrm{cond}_1(A)\approx4\times10^6$。若 $\mathbf{b}$ 有 $10^{-6}$ 的相对误差，解的相对误差上界约 $4$——**解可能完全翻车**。这个例子把「条件数大 ⇒ 解不可信」从口号变成可核算的数值：手算范数再相乘，就得到放大倍数。

## 2 更全面的扰动：A 和 b 一起动

实际中 $A$ 也带误差（系数是测量值或近似值）。设 $(A+\Delta A)(\mathbf{x}+\Delta\mathbf{x})=\mathbf{b}+\Delta\mathbf{b}$，展开并忽略高阶项，可得**更完整的不等式**：

$$
\frac{\lVert\Delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} \le \frac{\mathrm{cond}(A)}{1-\mathrm{cond}(A)\frac{\lVert\Delta A\rVert}{\lVert A\rVert}}\left(\frac{\lVert\Delta A\rVert}{\lVert A\rVert}+\frac{\lVert\Delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert}\right)
$$

当 $\lVert\Delta A\rVert$ 足够小（$\mathrm{cond}(A)\lVert\Delta A\rVert/\lVert A\rVert<1$）时，分母接近 1，不等式约化为

$$
\frac{\lVert\Delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} \lesssim \mathrm{cond}(A)\left(\frac{\lVert\Delta A\rVert}{\lVert A\rVert}+\frac{\lVert\Delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert}\right)
$$

**解读**：$A$ 的相对误差与 $\mathbf{b}$ 的相对误差**相加**后，被 $\mathrm{cond}(A)$ 统一放大。**条件数是「总输入误差」到「解误差」的总放大倍数**。<span class="marginnote">工程含义：若 $\mathrm{cond}(A)\approx10^{6}$，则输入的 $10^{-6}$ 相对误差会放大成解的 $O(1)$ 误差——<strong>解可能完全不可信</strong>。双精度下，$\mathrm{cond}(A)>10^{15}$ 的解「全是噪声」。</span>

一个工程换算：若想解的相对误差 $\le10^{-4}$，输入相对误差 $\le10^{-8}$（双精度下可能），而 $\mathrm{cond}(A)=10^4$，则放大约 $10^4$ 倍后刚好到 $10^{-4}$ 的边界——**精度预算必须倒着算**。

**数值实验（希尔伯特矩阵）**：$n=12$ 的希尔伯特矩阵 $\mathrm{cond}(H)\approx10^{16}$，双精度下解 $H\mathbf{x}=\mathbf{b}$ 的残差 $\lVert H\mathbf{x}-\mathbf{b}\rVert$ 可能很小，但解的相对误差接近 1——**残差小 ≠ 解准**（见下节）。

## 3 残差与真实误差：反直觉的事实

工程直觉说「残差小 = 解准」，数值分析戳破它：**残差 $\mathbf{r}=\mathbf{b}-A\hat{\mathbf{x}}$ 小，只能说明 $\hat{\mathbf{x}}$ 是「某个近似问题」的好解，不代表它接近真解**。两者通过条件数相连：

$$
\frac{\lVert\hat{\mathbf{x}}-\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} \le \mathrm{cond}(A)\,\frac{\lVert\mathbf{r}\rVert}{\lVert\mathbf{b}\rVert}
$$

若 $\mathrm{cond}(A)$ 巨大，即使残差小到 $10^{-16}$（机器精度），真实误差也可能高达 $10^{-16}\times10^{16}=O(1)$。

**例子**：$A=\begin{pmatrix}1&1\\1&1.0001\end{pmatrix}$，$\mathrm{cond}(A)\approx4\times10^4$。真解 $\mathbf{x}=(1,1)^\top$，$\mathbf{b}=(2,2.0001)$。若 $\hat{\mathbf{x}}=(0.5,1.5)^\top$，残差 $\lVert\mathbf{r}\rVert\approx0$（几乎精确满足方程组！），但 $\hat{\mathbf{x}}$ 与真解 $\mathbf{x}$ 的误差 $\lVert\Delta\mathbf{x}\rVert\approx0.7$——**残差几乎为零，解却错得离谱**。<span class="marginnote">这个例子是「病态矩阵的反直觉现场」：<strong>方程组几乎「被满足」了，但解完全不对</strong>。因为两条方程几乎平行，右端小小的比例差异对应解空间里大大的位移。判断可信度只能看条件数，不能看残差。</span>

## 4 公式解析：病态方程组的诊断流程

遇到一个线性方程组，如何系统判断「解靠不靠谱」？

**第一步，算条件数。** $\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert$。工程上用 1-或 ∞-范数（便宜），或 $\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert$。
**第二步，看数量级。** $\mathrm{cond}\sim10^1$：极良态；$10^2\sim10^3$：正常；$10^6$：病态；$10^{12+}$：双精度下几乎不可信。
**第三步，结合输入误差。** 解的相对误差上界 $\approx\mathrm{cond}(A)\times$（输入相对误差）。若输入数据本身只有 $10^{-3}$ 精度而 $\mathrm{cond}=10^6$，解最多有 $10^3$ 倍放大——**直接弃用或换问题表述**。
**第四步，对症下药。** 病态时可选：预条件（换基）、扩展精度（如改用 MPFR 任意精度算术或四精度 `float128`）、或者重新建模（换变量/归一化）。

**哪些矩阵天然病态？** 希尔伯特矩阵（元素 $1/(i+j-1)$）、接近奇异的矩阵、特征值跨多个数量级的矩阵、以及**拟合高次多项式**的设计矩阵——它们的共同特征：列之间几乎线性相关。<span class="marginnote">防病态的最佳实践在建模阶段：<strong>「换基不如换变量」</strong>——把范围大的变量归一化、把高次多项式换成正交基、把接近线性相关的特征删掉。条件数是「问题的体检报告」，建模时就要看。</span>

**条件数的几何直觉**：$A$ 把单位球映射成某个椭球，$\lVert A\rVert$ 是「最长半轴」，$\lVert A^{-1}\rVert$ 是「最短半轴倒数的上界」——条件数正是「椭球最拉伸 vs 最压扁的比值」。条件数大，意味着 $A$ 把某些方向**剧烈拉伸、另一些方向剧烈压扁**，解对这些「压扁方向」上的输入误差极度敏感。<span class="marginnote">这解释了为什么病态矩阵像「迷宫」：<strong>方向差异巨大的线性映射，逆映射必然放大微小扰动</strong>。几何视角让「条件数」不再是神秘数字，而是椭球的拉伸比。</span>

## 5 条件数、主元与精度的全景

| 环节 | 控制的误差 | 工具 |
| --- | --- | --- |
| 问题（病态） | 输入扰动放大 | 条件数 $\mathrm{cond}(A)$ |
| 算法（不稳定） | 舍入误差放大 | 列主元、正交化 |
| 实现（精度） | 浮点舍入 | 双精度、扩展精度 |

**辨析｜易错点：** 病态（问题）与不稳定（算法）再次划清——**列主元救的是「算法放大」，救不了「问题病态」**。一个条件数 $10^{12}$ 的方程组，无论用多稳定的算法，解的相对误差都被放大约 $10^{12}$ 倍。**看到条件数巨大，第一反应是「问题本身难」，不是「换算法」。**

## 6 术语速查表

| 术语 | 英文 | 一句话含义 |
| --- | --- | --- |
| 条件数 | condition number | $\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert$，误差放大倍数 |
| 病态方程组 | ill-conditioned system | 条件数大，解对输入误差极度敏感 |
| 残差 | residual | $\mathbf{r}=\mathbf{b}-A\hat{\mathbf{x}}$，衡量解多不满足方程 |
| 希尔伯特矩阵 | Hilbert matrix | $H_{ij}=1/(i+j-1)$，经典病态矩阵 |
| 预条件 | preconditioning | 换基/换表述以降低条件数 |

## 7 小结

- **核心误差界** $\dfrac{\lVert\Delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert}\le\mathrm{cond}(A)\dfrac{\lVert\Delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert}$：条件数是输入相对误差到解相对误差的放大倍数。
- 扰动同时在 $A$ 与 $\mathbf{b}$ 时：解误差 $\lesssim\mathrm{cond}(A)\bigl(\lVert\Delta A\rVert/\lVert A\rVert+\lVert\Delta\mathbf{b}\rVert/\lVert\mathbf{b}\rVert\bigr)$。
- **残差小 ≠ 解准**：$\lVert\hat{\mathbf{x}}-\mathbf{x}\rVert/\lVert\mathbf{x}\rVert\le\mathrm{cond}(A)\lVert\mathbf{r}\rVert/\lVert\mathbf{b}\rVert$——病态时残差可几乎为零而解错得离谱。
- 诊断流程：算条件数 → 看数量级 → 结合输入误差 → 预条件/换表述。
- 病态是问题、不稳定是算法——列主元救算法，救不了病态。
- 几何直觉：条件数是「单位球被 $A$ 拉伸成椭球的拉伸比」——方向差异越大越病态。
- 手算练习：$A=\begin{pmatrix}100&100\\100&100.01\end{pmatrix}$ 的 $\mathrm{cond}_1\approx4\times10^6$——输入 $10^{-6}$ 误差放大成解 $O(1)$ 误差。
- 一句话收束：**条件数是问题的体检报告，残差是解的安慰剂——信体检，别信安慰。**
- 手算条件数：1-范数下对矩阵逐列求和再相乘，上界立即可估。
- 病态不全是坏事：接近奇异的矩阵常对应物理「临界」状态，条件数是发现它的工具。
- 精度预算倒着算：输入误差 × 条件数 ≤ 期望解误差，先估条件数再定输入精度。
- 扰动在 $A$ 与 $\mathbf{b}$ 同时出现时，两者相对误差相加后统一放大。

在下一节，我们给出「病态但还想救」的一招：**迭代改善法（iterative refinement）**——用残差修正解，在有限精度下逼近高精度结果。

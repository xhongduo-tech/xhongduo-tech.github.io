---
title: 不确定性原理的算子表述
date: 2026-08-07
---

# 不确定性原理的算子表述

<div class="epigraph">
<p>两个不对易的可观测量，永远无法同时精确测量——这是内积几何的必然推论。</p>
<footer>—— 沃纳 · 海森堡（Werner Heisenberg），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§10.8 ｜ 2026-08-07</p>
</div>

## 为什么不确定性原理是「定理」而非「假说」

海森堡 1927 年提出的不确定性原理，听起来像物理直觉：**位置与动量无法同时精确测量**。可它其实是内积几何的严格推论——不需要任何物理假设，从「自伴算子 + Cauchy-Schwarz」就能证明。这正是泛函分析的力量：把一个深刻的物理原理，还原成一条数学定理。本节给出不确定性原理的算子表述与完整证明，并看它的几个变体（能量-时间、角动量分量）。<span class="marginnote">不确定性原理的数学核心是「不对易」：<strong>若 $[A,B] \\neq 0$，则 $A$ 与 $B$ 的标准差之积有正下界</strong>。它从「$[\\hat x, \\hat p] = i\\hbar I$」这一条对易关系出发，其余全是 Cauchy-Schwarz 的机械推导——「物理的深度原理」在数学里只有两行。</span>

## 1 方差与标准差

**定义（可观测量 $A$ 的方差）**：在态 $\psi$（$\|\psi\| = 1$）中，可观测量 $A$ 的期望是 $\langle A\rangle_\psi = \langle \psi, A\psi\rangle$，**方差**

$$
(\Delta A)_\psi^2 = \big\langle (A - \langle A\rangle I)^2\big\rangle_\psi = \|(A - \langle A\rangle I)\psi\|^2
$$

**标准差** $(\Delta A)_\psi = \|(A - \langle A\rangle I)\psi\|$。它度量「测量结果偏离期望的程度」。

**例**：$A = \hat x$，$\langle x\rangle = \int x|\psi|^2$，$(\Delta x)^2 = \int (x - \langle x\rangle)^2|\psi|^2$——位置的标准差就是「波包宽度」。<span class="marginnote">方差的算子定义 $\\|(A - \\langle A\\rangle I)\\psi\\|^2$ 与概率论的 $E[(X - EX)^2]$ 完全平行：<strong>期望是内积 $\\langle\\psi, A\\psi\\rangle$，方差是「去均值后的范数平方」</strong>。概率论与泛函分析在这再次合流（随机变量 = 自伴算子 + 态 = 概率测度）。</span>

## 2 不确定性原理

**定理（Robertson-Schrödinger 不确定性原理）**：设 $A, B$ 是自伴算子，$\psi$ 是态（$\|\psi\| = 1$）。则

$$
(\Delta A)_\psi\, (\Delta B)_\psi \ge \frac12 \big| \langle \psi, [A, B]\psi\rangle \big|
$$

**推论（Heisenberg）**：取 $A = \hat x$、$B = \hat p$，$[\hat x, \hat p] = i\hbar I$，得

$$
(\Delta x)(\Delta p) \ge \frac{\hbar}{2}
$$

**证明（用 Cauchy-Schwarz）**：

- **第一步，定义去均值算子**：$\tilde A = A - \langle A\rangle I$、$\tilde B = B - \langle B\rangle I$（自伴）。$\|\tilde A\psi\| = \Delta A$、$\|\tilde B\psi\| = \Delta B$。
- **第二步，Cauchy-Schwarz**：$\|\tilde A\psi\|\|\tilde B\psi\| \ge |\langle\tilde A\psi, \tilde B\psi\rangle|$。
- **第三步，交叉项分解**：$\langle\tilde A\psi, \tilde B\psi\rangle = \frac12\langle\psi, [\tilde A,\tilde B]\psi\rangle + \frac12\langle\{\tilde A,\tilde B\}\psi,\psi\rangle$（实部 + 虚部分解）。
- **第四步，取模**：$|\langle\tilde A\psi,\tilde B\psi\rangle| \ge \frac12|\langle\psi,[\tilde A,\tilde B]\psi\rangle|$（虚部 ≤ 模）。
- **第五步，代入**：$(\Delta A)(\Delta B) \ge \frac12|\langle[A,B]\psi,\psi\rangle|$（$[\tilde A,\tilde B] = [A,B]$）。<span class="marginnote">证明的巧妙在第三步：<strong>把 $\\langle\\tilde A\\psi, \\tilde B\\psi\\rangle$ 拆成「实部（反交换子）+ 虚部（交换子）」</strong>。对自伴算子，$\\langle\\tilde A\\psi,\\tilde B\\psi\\rangle$ 的虚部恰是对易子（乘以 $i$），实部是反交换子。取模时虚部给出下界——「不对易 ⟹ 不确定」由此而来。</span>

## 3 为什么「不对易 ⟹ 不确定」

不确定性原理的深层信息：**如果 $[A,B] \neq 0$，就不存在「$A$ 与 $B$ 都有确定值的态」**。

**定理（共同本征态）**：若 $\psi$ 同时是 $A$ 与 $B$ 的本征态（$A\psi = a\psi$、$B\psi = b\psi$），则 $(\Delta A) = (\Delta B) = 0$，由不确定性原理 $[A,B]\psi = 0$。**反过来：若 $[A,B] \neq 0$，则不存在同时确定 $A$ 与 $B$ 的态**。

**例**：$[\hat x, \hat p] = i\hbar I \neq 0$，所以位置与动量不能同时确定——海森堡原理的必然性。位置本征态（$\delta$ 函数）动量完全不确定，动量本征态（平面波）位置完全不确定。<span class="marginnote">「共同本征态不存在」是「不对易」的物理翻译：<strong>两个不对易的可观测量没有共同的本征基</strong>。这与「矩阵能否同时对角化」完全同构——$AB = BA$ 才能同时对角化。量子力学的不确定性，本质是「两个矩阵不对易」这一线性代数事实。</span>

## 4 公式解析：Cauchy-Schwarz 的下界

把证明中「取模」一步的细节写清：

$$
|\langle\tilde A\psi, \tilde B\psi\rangle|^2 = \Big|\frac12\langle\psi,[\tilde A,\tilde B]\psi\rangle\Big|^2 + \Big|\frac12\langle\psi,\{\tilde A,\tilde B\}\psi\rangle\Big|^2
$$

（实部与虚部的平方和——注意这里没有交叉项，因为内积取模。）

- **第一步（去均值算子自伴）**：$\tilde A^* = \tilde A$。于是 $\langle\tilde A\psi,\tilde B\psi\rangle = \langle\psi, \tilde A\tilde B\psi\rangle$。
- **第二步（分解）**：$\tilde A\tilde B = \frac12[\tilde A,\tilde B] + \frac12\{\tilde A,\tilde B\}$（交换子 + 反交换子）。
- **第三步（实虚分离）**：$[\tilde A,\tilde B]^* = [\tilde B,\tilde A] = -[\tilde A,\tilde B]$（反自伴 ⟹ 虚部）；$\{\tilde A,\tilde B\}^* = \{\tilde A,\tilde B\}$（自伴 ⟹ 实部）。
- **第四步（取模）**：$|\langle\tilde A\psi,\tilde B\psi\rangle| \ge \frac12|\langle[\tilde A,\tilde B]\psi,\psi\rangle|$——模不小于虚部。

**关键**：交换子是反自伴的（虚部），反交换子是自伴的（实部）。**不确定性下界来自交换子（虚部）——「不对易」直接给出不确定性的下界**。

## 5 例题精讲：不确定性原理的应用

**例题一：基态能量与不确定性**。

- 谐振子 $\hat H = \frac{\hat p^2}{2m} + \frac12 m\omega^2 \hat x^2$。
- 由 $(\Delta x)(\Delta p) \ge \hbar/2$，$E = \frac{(\Delta p)^2}{2m} + \frac12 m\omega^2(\Delta x)^2 \ge \frac{\hbar\omega}{2}$（用不等式优化）。
- 基态能量 $E_0 = \hbar\omega/2$——不确定性原理给出零点了能！<span class="marginnote">「零点能」$E_0 = \\hbar\\omega/2$ 是纯量子效应：<strong>位置与动量不能同时为零（否则都确定），于是能量有正下界</strong>。用不等式优化：$E \\ge \\frac{\\hbar^2}{8m(\\Delta x)^2} + \\frac12 m\\omega^2(\\Delta x)^2$，对 $\\Delta x$ 取极小即得 $\\hbar\\omega/2$。</span>

**例题二：能量-时间不确定性**。

- $[H, A] \neq 0$ 时，$(\Delta E)(\Delta A) \ge \frac12|\langle[H,A]\rangle|$。
- 若 $A$ 随演化「变化率」$|\frac{d\langle A\rangle}{dt}| = \frac1\hbar|\langle[H,A]\rangle|$，则 $(\Delta E)(\Delta t) \ge \frac{\hbar}{2}$。
- 「能量-时间不确定性」是 Robertson 版的推论（时间不是算子，但有有效表述）。

**例题三：角动量分量的不确定性**。

- $[\hat L_x, \hat L_y] = i\hbar\hat L_z$：$(\Delta L_x)(\Delta L_y) \ge \frac\hbar2|\langle L_z\rangle|$。
- 角动量两分量不能同时确定（除非 $L_z = 0$ 的本征态）。
- 这是「自旋不能同时知道两个方向」的数学表述。

**核心要点**：不确定性原理的三个应用——零点能、能量-时间、角动量——都是「$[A,B]$ 非零 ⟹ 标准积有界」的推论。

**辨析｜易错点：** 不确定性原理是「标准差」之积的下界，不是「测量的任何误差」。它说的是「统计散布」——大量重复测量的标准差之积 ≥ $\hbar/2$，不是「一次测量扰动」。不要把原理误解为「测量干扰论」。

## 6 常见误区与辨析

**误区一：把不确定性当「测量误差」**。

- 是标准差（统计散布）的下界，不是单次测量误差。
- 「测量干扰论」是错误解读。

**误区二：以为不确定性原理需要物理假设**。

- 纯数学：Cauchy-Schwarz + 对易关系。
- 从 $[\hat x,\hat p] = i\hbar I$ 机械推出。

**误区三：把「不能同时确定」当「不能同时测量」**。

- 数学内容：无共同本征态（$[A,B] \neq 0$）。
- 共同本征态不存在 = 不对易的物理翻译。

**核心要点：不确定性原理 = 内积几何 + 对易关系的推论**——是定理不是假说。


## 7 小结

- **方差**：$(\Delta A)^2 = \|(A - \langle A\rangle I)\psi\|^2$——标准差 = 去均值后的范数。
- **Robertson 不确定性**：$(\Delta A)(\Delta B) \ge \frac12|\langle[A,B]\rangle|$——Cauchy-Schwarz 的推论。
- **Heisenberg**：$(\Delta x)(\Delta p) \ge \hbar/2$——由 $[\hat x,\hat p] = i\hbar I$。
- **不对易 ⟹ 不确定**：$[A,B] \neq 0$ 时无共同本征态。
- **应用**：零点能、能量-时间、角动量分量。
- **定位**：不确定性原理是「对易关系 + 内积几何」的纯粹推论。

在下一节，我们完成第十章——**积分方程的 Fredholm 理论应用**，把紧算子理论带回积分方程的求解。

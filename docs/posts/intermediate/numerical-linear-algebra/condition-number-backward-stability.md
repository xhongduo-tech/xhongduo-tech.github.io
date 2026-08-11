---
title: 条件数与向后稳定性
date: 2026-08-11
---

# 条件数与向后稳定性

<div class="epigraph">
<p>错误地使用不充分的数据，其危害远小于完全不用数据。</p>
<footer>—— 查尔斯 · 巴贝奇（Charles Babbage）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 数值线性代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么误差要分成两层来看

任何一个数值计算都有误差。但「误差从哪来、怪谁」是个哲学问题：同一个计算结果，可能是**问题本身难**，也可能是**算法不行**。把这两者混为一谈，是数值线性代数里最常见的糊涂账。比如「高斯消去不稳定」和「矩阵本身病态」是两回事——前者换算法能救，后者换什么算法都救不了。

本节对标 Trefethen 与 Bau《Numerical Linear Algebra》第 12–14 讲，建立全书的方法论骨架：**问题的条件数（conditioning）**与**算法的稳定性（stability）**。这两个概念一旦清楚，之前所有算法（LU、QR、SVD）的优劣就都有了统一的判据，后面的特征值算法也会被放在同一把尺子下量。这是从「会算」走向「懂算」的关键一跃。

## 1 问题的条件数：输入扰动放大了多少

**核心概念：问题的条件数（condition number）** 度量「输入的小扰动，最多能把输出放大多少倍」。形式化地说，设问题 $f$ 在 $x$ 处可微，则相对条件数为

$$
\kappa(x) = \lim_{\delta \to 0} \sup_{\lVert \delta x \rVert \leq \delta} \frac{\lVert f(x + \delta x) - f(x) \rVert}{\lVert f(x) \rVert} \Big/ \frac{\lVert \delta x \rVert}{\lVert x \rVert}
$$

人话：**条件数 ≈ 输出相对误差 / 输入相对误差 的最大放大倍数**。$\kappa$ 接近 1 是好问题（良态，well-conditioned），$\kappa$ 巨大是坏问题（病态，ill-conditioned）。

举两个极端。求 $f(x) = \sqrt{x}$ 在 $x = 100$ 处的值，条件数约 $\kappa \approx \frac{|x f'(x)|}{|f(x)|} = \frac{1}{2}$——误差**被缩小**，这是良态。反之，求多项式 $f(x) = (x-1)^{20}$ 在 $x = 1$ 附近的根，输入值域上微小的函数扰动就能让根挪动几个数量级——这是典型的病态问题。<span class="marginnote">最著名的病态实例是 Wilkinson 多项式 $\prod_{i=1}^{20}(x-i)$：根是 $1,2,\dots,20$，看起来温和极了；但把 $x^{19}$ 的系数改掉 $10^{-10}$，就有几个根变成复数。2002 年 Wilkinson 去世前的演讲里，这仍是他最引以为傲的反例——它直接证明了「特征多项式 + 求根」路线在数值上不可行。</span>

**重要区分：条件数是问题的属性，跟算法无关。** 同一个病态问题，任何算法都只能给出同样被放大的误差。这与下一节的「算法稳定性」正好互补。

## 2 矩阵的条件数：Ax = b 的放大镜

对线性方程组 $Ax = b$，输入是 $(A, b)$，输出是 $x$。若只扰动 $b$，可推出**相对误差不等式**：

$$
\frac{\lVert \delta x \rVert}{\lVert x \rVert} \leq \kappa(A) \, \frac{\lVert \delta b \rVert}{\lVert b \rVert}, \qquad \kappa(A) = \lVert A \rVert \, \lVert A^{-1} \rVert
$$

**核心概念：矩阵条件数** $\kappa(A) = \lVert A \rVert \lVert A^{-1} \rVert$。在 2-范数下，结合上一讲 $\lVert A \rVert_2 = \sigma_1$、$\lVert A^{-1} \rVert_2 = 1/\sigma_{\min}$，得到

$$
\kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_{\min(m,n)}}
$$

**条件数是最大奇异值与最小奇异值的比值**：它度量「$A$ 把单位球拉得有多扁」。拉得越扁，越难以分辨方向，求解时输入的小误差越容易被放大成输出方向上的大误差。

**推导速览**：由 $b = Ax$ 得 $\delta x = A^{-1} \delta b$，于是 $\lVert \delta x \rVert \leq \lVert A^{-1} \rVert \lVert \delta b \rVert$；同时 $\lVert b \rVert \leq \lVert A \rVert \lVert x \rVert$。两式相除即得。**上界在「$\delta b$ 对准 $u_{\min}$ 方向、$x$ 对准 $v_{\min}$ 方向」时取到**——病态的本质是存在一个几乎落在零空间的方向。

### 量一量：亲手算几个条件数

理论讲再多不如手算一次。取对角矩阵 $D = \operatorname{diag}(2, 1)$：奇异值 $2, 1$，$\kappa_2 = 2$——良态，解方程几乎不放大误差。取 $D = \operatorname{diag}(10^8, 1)$：$\kappa_2 = 10^8$，沿长轴方向上的误差被放大 $10^8$ 倍，病态无疑。取旋转矩阵 $Q$（正交）：$\kappa_2(Q) = 1$——**正交矩阵是条件数的黄金标准，永远不放大误差**，这正是前几讲「数值上只用正交变换」的深层理由。取 Hilbert 矩阵 $H_{ij} = 1/(i+j-1)$ 的 $10$ 阶：$\kappa_2 \approx 1.6 \times 10^{13}$，逼近双精度极限——**看着温和、摸着病态**的标本。这些数字会反复出现在后面的误差分析里，建议与它们先混个脸熟。

**两个关于条件数的小性质，值得记牢**：

- **缩放不变**：$\kappa(cA) = \kappa(A)$（$c \neq 0$）。**「把矩阵整体乘个大数来改善条件数」是无效的**——条件数只关心比值 $\sigma_{\max}/\sigma_{\min}$，整体缩放约不掉。这是初学者最爱犯的「数学直觉型」错误。
- **行/列平衡才有效**：真正有用的是**对角相似变换** $D_1 A D_2$（给各行各列乘不同量级），让每行每列量级均匀。解病态方程前的「平衡（balancing）」预处理正基于此——它能把差到 $10^{20}$ 的量级差压回可处理范围。

**辨析｜易错点：**「缩放」是对整矩阵做一个常数乘法（无效），「平衡」是对各行各列做不同乘法（有效）。前者想「用一个数修好所有病」，后者是「让每个方向的量级匹配它的实际大小」——一字之差，天壤之别。

## 3 扰动分析：残差小 ≠ 误差小

算完 $Ax = b$，人们习惯看一眼**残差 $r = b - Ax_{\mathrm{computed}}$**。残差小，是不是就说明算得好？**不一定。** 关键定理（Trefethen 定理 12.1 的通俗版）：

$$
\frac{1}{\kappa(A)} \frac{\lVert r \rVert}{\lVert b \rVert} \leq \frac{\lVert x - x_{\mathrm{computed}} \rVert}{\lVert x \rVert} \leq \kappa(A) \frac{\lVert r \rVert}{\lVert b \rVert}
$$

**残差被条件数双向夹逼。** 若 $\kappa(A) = 1$，残差小等价于误差小，万事大吉；若 $\kappa(A) = 10^{10}$，残差 $10^{-16}$ 只能推出误差最多 $10^{-6}$——残差看着完美，答案已经错得离谱。<span class="marginnote">Hilbert 矩阵 $H_{ij} = 1/(i+j-1)$ 是经典反面教材：$H_{10}$ 的条件数已超 $10^{13}$，用它解方程组，哪怕数据本身精确到机器精度，结果也可以全错。它在最小二乘拟合里频繁出现，是「看着无害实则病态」的标本。</span>

**辨析｜易错点：**「残差小所以算得准」是初学数值线性代数最容易踩的坑。判定标准永远是**误差**（与真解比较），而不是残差（与方程比较）。残差只能告诉我们「这个解在原方程里显得有多合理」，无法告诉我们「它离真解有多近」——两者之间隔着 $\kappa(A)$。

## 4 向后稳定性：把锅甩给输入

条件数把问题的难易定死了。剩下的自由度在算法。**向后稳定性（backward stability）** 给出一个漂亮的判据：算法算出的解 $\hat{x}$ 不要求靠近真解，只要求**它恰好是某个「略微扰动过」的问题的精确解**。

**核心概念：算法是向后稳定的，若对每个输入 $x$，存在扰动 $\delta x$ 使得**

$$
\hat{f}(x) = f(x + \delta x), \qquad \frac{\lVert \delta x \rVert}{\lVert x \rVert} = O(\varepsilon_{\mathrm{machine}})
$$

**其中 $\varepsilon_{\mathrm{machine}}$ 是机器精度（双精度约 $2.2 \times 10^{-16}$）。** 直觉：算法诚实地回答了「你那个问题的精确解」，只是悄悄把你给的输入改动了 $10^{-16}$ 量级——这个量级的输入误差，本来就超出我们采集数据的精度范围，无可指摘。

**向后稳定性 + 条件数 = 前向误差上界**，这是全书最常用的一个不等式：

$$
\frac{\lVert \hat{x} - x \rVert}{\lVert x \rVert} \lesssim \kappa(\text{问题}) \times \varepsilon_{\mathrm{machine}}
$$

**前向误差 ≤ 条件数 × 向后误差。** 公式分清了责任：算法只保证向后误差是 $O(\varepsilon)$，剩下的放大全部由问题的条件数承担。任何一个声称「精度好」的算法，都可以翻译成这条不等式。

**哪些算法向后稳定？** 我们的老朋友大部分都合格：Householder QR 向后稳定；选主元的 Gauss 消去在绝大多数情形向后稳定；未选主元的 Gauss 消去**不是**（下一节会细算）。这正是「为什么选主元是生死攸关的事」的根源。

## 5 公式解析：κ(A) = ‖A‖ ‖A⁻¹‖

这是全节最核心的公式，拆成三步理解：

- **第一步，看 $\lVert A \rVert$**：$\lVert A \rVert_2 = \max_{x \neq 0} \lVert Ax \rVert / \lVert x \rVert = \sigma_1$，度量「$A$ 最多能把输入拉长多少倍」。它是放大率的上限。
- **第二步，看 $\lVert A^{-1} \rVert$**：同理 $= 1/\sigma_{\min}$，度量「解方程 $Ax=b$ 时，$b$ 的误差最多被 $A^{-1}$ 放大多少倍」。它是反向放大率的上限。
- **第三步，相乘**：正向拉长 $\times$ 反向放大，得到**往返放大率**。为什么乘积必然 $\geq 1$？因为 $\lVert A \rVert \lVert A^{-1} \rVert \geq \lVert AA^{-1} \rVert = \lVert I \rVert = 1$。**条件数不可能小于 1**——任何问题都不可能把误差缩小到「负」的程度。

最后提醒记号：$\kappa$ 依赖范数，$\kappa_1, \kappa_2, \kappa_\infty$ 不同但量级通常一致；本书默认 2-范数，故记为 $\kappa_2(A) = \sigma_1/\sigma_{\min}$。

## 6 小结

- **条件数属于问题**，$\kappa = \lim \sup \frac{\text{输出相对误差}}{\text{输入相对误差}}$；矩阵版 $\kappa(A) = \lVert A \rVert \lVert A^{-1} \rVert = \sigma_{\max}/\sigma_{\min} \geq 1$。
- 解 $Ax=b$：$\frac{\lVert \delta x \rVert}{\lVert x \rVert} \leq \kappa(A) \frac{\lVert \delta b \rVert}{\lVert b \rVert}$——病态矩阵放大输入误差。
- **残差小 ≠ 误差小**：两者隔着条件数 $\kappa(A)$ 的双向夹逼；判误差要看与真解的比较。
- **向后稳定性属于算法**：$\hat{f}(x) = f(x+\delta x)$ 且 $\lVert \delta x \rVert / \lVert x \rVert = O(\varepsilon)$——算法只对扰动后的输入精确。
- **前向误差 ≤ 条件数 × 向后误差**：责任一分为二，这是评判一切算法的总纲。

**两把尺子的分工**：条件数回答「这问题有多难」（问题属性，不可控），稳定性回答「这算法有多好」（算法属性，可控）；算法再稳也补不了病态的窟窿，问题再良也架不住算法自毁——**把责任分清楚，是误差分析的第一步，也是最关键的一步**。

在下一节，我们把条件数与稳定性这两把尺子带进特征值世界。特征值问题病态起来毫不逊色——而它的标准解法 **Schur 分解**，将告诉我们为什么「求特征多项式再求根」是数值上最大的禁忌之一。

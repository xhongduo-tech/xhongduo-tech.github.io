---
title: 纠缠的度量：并发度（concurrence）与纠缠熵
date: 2026-08-07
---

# 纠缠的度量：并发度（concurrence）与纠缠熵

<div class="epigraph">
<p>熵是一个系统「内在不确定性」的度量，而纠缠熵度量的是两半系统之间共享的这种不确定性。</p>
<footer>—— 尼尔斯 · 玻尔（Niels Bohr）与冯 · 诺伊曼（John von Neumann）的思想合流</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§11.3 ｜ 2026-08-07</p>
</div>

## 为什么从纠缠度量开始

前面我们有了「可分/纠缠」的二值判断，也有了 CHSH 的数值测试，但还缺一样东西：**一把尺子，能说出「这个态比那个态纠缠得多」。** 量子信息论需要给纠缠定量的原因很实际：隐形传态的保真度随纠缠量变、纠缠蒸馏的产量由纠缠量决定、NISQ 设备的质量也要用「能产生多少纠缠」来衡量。<span class="marginnote">纠缠度量（entanglement measure）要满足几条公理：可分态取 0、LOCC 操作下不增、纯态上可解析计算。本节聚焦两个最常用的量——<strong>纠缠熵</strong>（纯态）与<strong>并发度</strong>（两比特任意态）。</span>本节给纠缠配两把尺子：一把给纯态（von Neumann 熵），一把给混合态（concurrence）。

顺带交代「纠缠为什么要定量」的一条主线：第二篇《量子隐形传态》里「1 个 ebit 传 1 个量子比特」、以及《超密编码》里「1 个 ebit + 1 个经典比特 = 传 2 个量子比特」，说的正是纠缠的量。纠缠度量把这句「经验规律」变成可计算的数，也由此与量子通信的资源论接上轨。

纠缠度量必须先立三条件，缺一不可：

| 公理 | 含义 | 直觉 |
| --- | --- | --- |
| 可分态取 0 | $E(\text{可分}) = 0$ | 没纠缠就没资源 |
| LOCC 不增 | 局域操作不产生纠缠 | 免费操作不创造资源 |
| 纯态可算 | 纯态上有闭式公式 | 度量必须可计算 |

## 1 纯态的纠缠熵：$S(\rho_A)$

对纯态 $\lvert\psi\rangle_{AB}$，定义它的**纠缠熵**为任一部分的 von Neumann 熵：

$$
E(\lvert\psi\rangle) = S(\rho_A) = -\operatorname{tr}(\rho_A \log_2 \rho_A), \qquad \rho_A = \operatorname{tr}_B(\lvert\psi\rangle\langle\psi\rvert)
$$

关键事实（前面已见）：**对纯态 $S(\rho_A) = S(\rho_B)$**，所以选 A 还是选 B 无所谓。<span class="marginnote">直觉：整体是纯态意味着「量子系统没有内在随机性」，那熵从哪来？来自「丢掉 B 的那部分信息」。纠缠越多，丢掉的信息越多，熵越大。完全可分时 $S(\rho_A)=0$；最大纠缠时 $S(\rho_A) = \log_2 d$（$d$ 是单方维数）。</span>

它满足度量公理：可分态熵为 0，最大纠缠态取最大值，LOCC 下不增。且对纯态，纠缠熵有个**操作意义**：纠缠蒸馏率（可提取的 ebit 数/份）恰好等于它——理论上纠缠的量就等于「能蒸馏出的贝尔态个数」。

三种典型两比特态的纠缠量对照：

| 态 | 形态 | 纠缠熵 $S(\rho_A)$ | 并发度 $C$ |
| --- | --- | --- | --- |
| $\lvert00\rangle$ | 积态（可分） | 0 | 0 |
| $\tfrac12\lvert00\rangle\langle00\rvert+\tfrac12\lvert11\rangle\langle11\rvert$ | 可分混合 | 1 | 0 |
| $\lvert\Phi^+\rangle$ | 最大纠缠 | 1 | 1 |
| $\cos\theta\lvert00\rangle+\sin\theta\lvert11\rangle$ | 部分纠缠 | $0<S<1$ | $0<C<1$ |

表格里最有启发性的是第二行：经典相关的混合态，纠缠熵已经到 1，并发度却还是 0——**熵量到「随机性」，并发度量到「可利用的纠缠」**，两者分道扬镳正是混合态的核心困难。

## 2 公式解析：$\lvert\Phi^+\rangle$ 的纠缠熵

两比特最大纠缠态的纠缠熵应该取最大值 $\log_2 2 = 1$ ebit。验证一下：

$$
\rho_A = \operatorname{tr}_B(\lvert\Phi^+\rangle\langle\Phi^+\rvert) = \frac{1}{2}\lvert0\rangle\langle0\rvert + \frac{1}{2}\lvert1\rangle\langle1\rvert = \frac{I}{2}
$$

- **第一步，部分迹**：把 $\lvert\Phi^+\rangle$ 的密度算符 $\frac12(\lvert00\rangle+\lvert11\rangle)(\langle00\rvert+\langle11\rvert)$ 对 B 求和，交叉项消失，留下 $\frac12\lvert0\rangle\langle0\rvert + \frac12\lvert1\rangle\langle1\rvert$。
- **第二步，算熵**：$\rho_A$ 的本征值是 $\frac12,\frac12$，于是 $S(\rho_A) = -\frac12\log_2\frac12 - \frac12\log_2\frac12 = 1$。
- **第三步，读结果**：1 ebit——「一个贝尔态含一个单位纠缠」，与隐形传态里「传 1 个未知量子比特消耗 1 个 ebit」完全吻合。<span class="marginnote">这个「纠缠熵 = 可蒸馏 ebit 数」的等式在纯态上成立得干净利落；混合态上两者分道扬镳——混合态能蒸馏出的纠缠通常小于纠缠熵，因为「经典随机性」会稀释纠缠。</span>

## 3 混合态为什么需要新的度量

对混合态，纠缠熵 $S(\rho_A)$ 不再够用。反例就是前面见过的：$\rho = \frac12\lvert00\rangle\langle00\rvert + \frac12\lvert11\rangle\langle11\rvert$ 是可分的（经典相关），但 $S(\rho_A) = 1$——用熵量出来「像纠缠」，实际不是。<span class="marginnote">问题根源：混合态里的熵既包含「量子纠缠」又包含「经典随机性」，而经典随机性不构成可用的纠缠资源。把两者分离开，是混合态纠缠度量的全部困难。</span>于是需要**纠缠蒸馏熵（entanglement of formation）**等更精细的度量：它定义为「要制备该混合态，平均最少需要多少个贝尔态」。

**纠缠蒸馏熵（$E_F$）**：对两比特态，它有一个显式公式（Wootters 1998）：$E_F = h(C)$，其中 $h(x) = -\frac{1+\sqrt{1-x^2}}{2}\log_2\frac{1+\sqrt{1-x^2}}{2} - \frac{1-\sqrt{1-x^2}}{2}\log_2\frac{1-\sqrt{1-x^2}}{2}$ 是二进制熵函数，而 $C$ 是下面要讲的**并发度**。

## 4 并发度（concurrence）：两比特混合态的标准尺

**并发度（concurrence）** $C(\rho)$ 是 Wootters 给出的两比特纠缠度量，定义需要「自旋翻转」：

$$
\tilde{\rho} = (\sigma_y \otimes \sigma_y)\, \rho^*\, (\sigma_y \otimes \sigma_y), \qquad C(\rho) = \max\{0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4\}
$$

其中 $\lambda_i$ 是厄米矩阵 $\sqrt{\sqrt{\rho}\,\tilde{\rho}\,\sqrt{\rho}}$ 的本征值（降序排列）。<span class="marginnote">$\sigma_y$ 是 Pauli-Y 矩阵，$\tilde{\rho}$ 叫「自旋翻转态」（spin-flipped state）：对两比特系统的每个「时间反演」翻转做一次翻转。并发度把「纠缠量」压缩进一个 $[0,1]$ 的数：可分态为 0，贝尔态为 1。</span>

对纯态，并发度有简洁闭式：若 $\lvert\psi\rangle = \sum_{ij} a_{ij}\lvert ij\rangle$，则 $C = 2\lvert \det(a_{ij}) \rvert$（系数矩阵行列式）。<span class="marginnote">直觉：系数矩阵的「不可分解性」就是纠缠——若 $\lvert\psi\rangle$ 是积态，系数矩阵秩 1，行列式为 0，并发度 0；贝尔态的系数矩阵是 $\frac{1}{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix}$，行列式 $\frac12$，$C = 1$。

### 数值例：部分纠缠态的并发度

取 $\lvert\psi\rangle = \frac{\sqrt3}{2}\lvert00\rangle + \frac12\lvert11\rangle$（即 $\theta = 30^\circ$ 的情形）。系数矩阵是对角阵：

$$
A = \begin{pmatrix} \frac{\sqrt3}{2} & 0 \\ 0 & \frac12 \end{pmatrix}, \qquad \det A = \frac{\sqrt3}{4}, \qquad C = 2 \times \frac{\sqrt3}{4} = \frac{\sqrt3}{2} \approx 0.866
$$

对应的纠缠熵：$\rho_A$ 本征值为 $\frac34, \frac14$，$S(\rho_A) = -\frac34\log_2\frac34 - \frac14\log_2\frac14 \approx 0.811$ ebit。<span class="marginnote">注意 $C \neq S$：<strong>并发度与纠缠熵是两个不同的数，但纯态上互为单调</strong>，所以「谁大谁小」的排序一致。$C$ 取 $[0,1]$、$S$ 取 $[0,\log_2 d]$——量纲不同，别直接比较数值。</span></span>

## 5 公式解析：纯态并发度 $C = 2\lvert\det A\rvert$

以 $\lvert\Phi^+\rangle$ 验证并发度与纠缠熵一致：

$$
\lvert\Phi^+\rangle = \frac{1}{\sqrt2}\lvert00\rangle + \frac{1}{\sqrt2}\lvert11\rangle \Rightarrow A = \frac{1}{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix}
$$

- **第一步，排系数矩阵**：把 $\lvert\psi\rangle = \sum_{ij}a_{ij}\lvert ij\rangle$ 的系数排成 $2\times2$ 矩阵 $A$。$\lvert\Phi^+\rangle$ 的 $A = \frac{1}{\sqrt2}I$。
- **第二步，算行列式**：$\det A = \frac12$，取绝对值乘 2：$C = 2 \times \frac12 = 1$。
- **第三步，对照**：$C = 1$ 对应最大纠缠；由 $E_F = h(C)$，$h(1) = -\frac12\log_2\frac12 - \frac12\log_2\frac12 = 1$ ebit，与纠缠熵 $S(\rho_A)=1$ 对上。<span class="marginnote">这里浮现一条漂亮的对应：<strong>纯态上并发度、纠缠熵、蒸馏熵三个量互为单调函数</strong>，给出相同的纠缠排序；只有进入混合态，它们才各自分化。

三个度量各有适用面，总结成速查表：

| 度量 | 适用对象 | 取值范围 | 操作意义 |
| --- | --- | --- | --- |
| 纠缠熵 $S(\rho_A)$ | 纯态 | $[0, \log_2 d]$ | 可蒸馏 ebit 数（纯态） |
| 纠缠蒸馏熵 $E_F$ | 两比特混合态 | $[0, 1]$ | 制备该态所需贝尔态数 |
| 并发度 $C$ | 两比特任意态 | $[0, 1]$ | 纠缠含量（不是概率） |</span>

**辨析｜易错点：** 并发度的取值范围是 $[0,1]$，但**它不是概率**。一个 $C=0.5$ 的态并不意味着「有一半概率纠缠」；它只是说「这个态的纠缠含量相当于半个贝尔态」。把并发度当概率是初学者最常犯的错误。

## 6 小结

- **纠缠熵** $S(\rho_A)$：纯态的标准度量，等于任一部分的 von Neumann 熵；最大纠缠两比特态取 1 ebit。
- 纯态上 $S(\rho_A) = S(\rho_B)$，且纠缠熵 = 可蒸馏 ebit 数（操作意义）。
- 混合态熵会混入**经典随机性**，需用**纠缠蒸馏熵** $E_F = h(C)$ 或**并发度** $C$。
- **并发度** $C$：$[0,1]$ 间的纠缠含量，纯态闭式 $C = 2\lvert\det A\rvert$，贝尔态取 1。

在下一节，我们把镜头从「两比特」拉到「多比特」：三个以上粒子的纠缠长什么样？这就是 **GHZ 态与 W 态：多体纠缠的两种类型**。

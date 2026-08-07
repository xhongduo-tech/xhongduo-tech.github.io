---
title: Simon 算法
date: 2026-08-07
---

# Simon 算法

<div class="epigraph">
<p>Simon 的算法是通往 Shor 算法的第一块真正意义的里程碑。</p>
<footer>—— 尼尔森（Michael Nielsen）与庄（Isaac Chuang）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§1.4.2、§5.4.1 ｜ 2026-08-07</p>
</div>

## 为什么从 Simon 算法开始

Bernstein-Vazirani 把「隐藏结构」编码成内积，Simon 则把结构升级为**周期**——而「找周期」正是 Shor 大数分解算法的灵魂。<span class="marginnote">Simon 算法出自 D. Simon, "On the Power of Quantum Computation," <i>SIAM J. Comput.</i> 26 (1997) 1474。它是第一个证明「量子能在有承诺的查询问题上指数级优于随机经典」的算法，也是 Shor（1994）的直接前奏。</span>Simon 问题承诺一个函数 $f:\{0,1\}^n \to \{0,1\}^n$ 满足「$f(x)=f(y)$ 当且仅当 $x = y$ 或 $x = y \oplus s$」（$s$ 为隐藏周期），要求找出 $s$。经典算法需要 $\Theta(2^{n/2})$ 次查询，Simon 算法只需 $O(n)$ 次——**指数级加速**，而且它的分析（相位 + 纠缠 + 测量后处理）几乎原封不动地移植到 Shor 上。

## 1 Simon 问题的设定

**Simon 问题**：给定 $f:\{0,1\}^n \to \{0,1\}^n$ 及承诺：存在非零 $s$ 使 $f(x) = f(x \oplus s)$ 对所有 $x$ 成立，且除此之外 $f$ 是**二对一**（每两个输入对应一个输出）。求 $s$。<span class="marginnote">承诺的含义：$f$ 像一把「折叠扇」，把 $2^n$ 个输入两两折叠成 $2^{n-1}$ 个输出，折叠的「折痕」正是 $s$。若 $s = 0$ 则 $f$ 是双射（一对一的退化情形）。</span>

经典算法：随机采样 $f(x)$，靠「生日悖论」找碰撞。要找到一对 $x, y$ 使 $f(x)=f(y)$，期望需要约 $2^{n/2}$ 次查询（生日悖论），再对两次结果异或得 $s = x \oplus y$。下界论证表明这就是最优。

## 2 Simon 算法的量子线路

线路骨架与 BV 一致，但查询变成「输入与输出双寄存器」的翻转查询，且多了一个测量后处理阶段：

1. 制备 $\lvert0\rangle^{\otimes n}\lvert0\rangle^{\otimes n}$，对第一个寄存器作用 $H^{\otimes n}$。
2. 作用翻转查询 $O_f$：$\lvert x\rangle\lvert0\rangle \to \lvert x\rangle\lvert f(x)\rangle$。
3. **测量第二个寄存器**，记录输出 $f(x)$。
4. 对第一个寄存器作用 $H^{\otimes n}$，测量，得一个 $y$。
5. 重复步骤 1–4 约 $O(n)$ 次，收集一组线性方程，经典求解 $s$。

第 2 步后态是 $\frac{1}{\sqrt{2^n}}\sum_x \lvert x\rangle\lvert f(x)\rangle$——这是「输入与输出」的**纠缠态**：同一个 $f$ 值对应两个输入 $x$、$x\oplus s$。第 3 步测量第二个寄存器后，第一个寄存器坍缩成 $\frac{1}{\sqrt2}(\lvert x\rangle + \lvert x\oplus s\rangle)$（$x$ 是该输出的一个随机原像）。

## 3 公式解析：为什么测到的 $y$ 正交于 $s$

第 4 步的 $H^{\otimes n}$ 把「两原像叠加」变到一组满足正交条件的 $y$ 上。设第二个寄存器测得输出 $f(x_0)$，第一个寄存器是

$$
\lvert\psi\rangle = \frac{1}{\sqrt2}(\lvert x_0\rangle + \lvert x_0 \oplus s\rangle)
$$

$H^{\otimes n}$ 作用后：

$$
H^{\otimes n}\lvert\psi\rangle = \frac{1}{\sqrt{2^{n+1}}}\sum_y \left[(-1)^{x_0\cdot y} + (-1)^{(x_0\oplus s)\cdot y}\right]\lvert y\rangle
$$

- **第一步，逐项变换**：$H^{\otimes n}\lvert x_0\rangle = \frac{1}{\sqrt{2^n}}\sum_y (-1)^{x_0\cdot y}\lvert y\rangle$，对 $\lvert x_0 \oplus s\rangle$ 同理，合并到一个和式里。
- **第二步，提取公因式**：$(-1)^{(x_0\oplus s)\cdot y} = (-1)^{x_0\cdot y}(-1)^{s\cdot y}$，于是括号为 $(-1)^{x_0\cdot y}\big[1 + (-1)^{s\cdot y}\big]$。
- **第三步，干涉相消**：若 $s\cdot y = 1$，括号为 $1-1=0$，该项消失；若 $s\cdot y = 0$，括号为 2，振幅保留。所以**测到的 $y$ 必满足 $s\cdot y = 0$**。<span class="marginnote">这就是 Simon 的心脏：一次测量给出一个随机向量 $y$，它正交于隐藏周期 $s$（在 $\mathbb{F}_2$ 内积意义下）。每次测量得到一个独立的正交方程 $s\cdot y = 0$，$O(n)$ 次后方程组就能唯一解出 $s$。</span>

## 4 经典后处理：从正交方程解出 $s$

收集到的测量结果 $y_1, \dots, y_m$ 都满足 $s \cdot y_i = 0$。把它们组成矩阵，经典解齐次线性方程组：

$$
\begin{pmatrix} \text{— } y_1 \text{ —} \\ \text{— } y_2 \text{ —} \\ \vdots \\ \text{— } y_m \text{ —} \end{pmatrix} \begin{pmatrix} s_1 \\ \vdots \\ s_n \end{pmatrix} = 0 \pmod 2
$$

- **第一步，信息量**：每次测量均匀给出 $n-1$ 维子空间 $\{y : s\cdot y = 0\}$ 里的一个随机向量。随机向量的新信息量平均接近 1 个独立方程。
- **第二步，凑齐秩**：约 $m = n + O(1)$ 次测量后，$y_i$ 张满 $s$ 的正交补（秩 $n-1$），方程组的解空间就是 $\{0, s\}$。
- **第三步，排除平凡解**：若解得 $s = 0$（退化情形），需多测几次验证；非零 $s$ 即答案。<span class="marginnote">这里的后处理是经典的——量子部分只负责「批量生成正交向量」，解方程交给高斯消元。这个「量子生成数据 + 经典后处理」的两阶段模式，在 Shor 里原样重现。</span>

**辨析｜易错点：** Simon 算法是**概率性**的：测量结果 $y$ 随机，偶尔会重复、方程冗余，所以需要约 $O(n)$ 次而非恰好 $n$ 次，且失败概率指数小。这与 BV 的「一次成功」不同。另外，Simon 加速是**查询复杂度**的指数加速，但若把 oracle 内部 $f$ 的实现算进去，总时间未必比经典快——它和 Deutsch-Jozsa 一样是「结构化承诺问题」的理论证明，实用价值在于其思想。

## 5 从 Simon 到 Shor：算法的谱系

Simon 与 Shor 共享同一套骨架：**相位编码周期 → 测量提取 → 经典后处理**。对照如下：

| 要素 | Simon | Shor |
| --- | --- | --- |
| 隐藏结构 | 群 $\mathbb{Z}_2^n$ 里的周期 $s$ | 循环群 $\mathbb{Z}_N$ 里的周期 $r$ |
| 相位编码 | $(-1)^{s\cdot x}$（内积） | $e^{2\pi i jx / N}$（傅里叶相位） |
| 变换 | $H^{\otimes n}$（Hadamard） | 量子傅里叶变换 QFT |
| 测量输出 | 正交向量 $y$ | 与 $j/r$ 相关的数 |
| 后处理 | 解线性方程组 | 连分数展开求 $r$ |

Simon 用的是「$\mathbb{Z}_2$ 上的 Hadamard」（$2\times2$ 傅里叶），Shor 用的是「$\mathbb{Z}_N$ 上的 QFT」——**QFT 是 Hadamard 的推广**。<span class="marginnote">这条谱系极具教学价值：理解了 Simon 的「测量给出正交方程」，就理解了 Shor 的「测量给出周期的估计」——后者只是把前者从 $2$ 元域换成 $N$ 元循环群。下一节我们正式进入量子傅里叶变换，它是这条谱系的下一个节点。</span>

## 6 小结

- **Simon 问题**：承诺 $f(x)=f(x\oplus s)$ 且二对一，求隐藏周期 $s$；经典 $\Theta(2^{n/2})$ 次，量子 $O(n)$ 次（指数加速）。
- 线路 = **$H^{\otimes n}$ → 翻转查询 → 测第二寄存器 → $H^{\otimes n}$ → 测第一寄存器**。
- 核心：测量第一寄存器得到**随机正交向量** $y$（$s\cdot y = 0$），重复 $O(n)$ 次凑齐方程，经典解出 $s$。
- **概率性**算法，失败概率指数小；与 BV 的确定成功不同。
- **Simon → Shor**：Hadamard 是 QFT 的特例，周期提取的骨架一脉相承。

在下一节，我们正式把 Hadamard 推广成傅里叶——**量子傅里叶变换（QFT）及其实现**，这是 Shor 算法的数学发动机。

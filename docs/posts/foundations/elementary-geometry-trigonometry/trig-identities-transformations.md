---
title: 三角恒等变换（和差角/二倍角）
date: 2026-08-07
---

# 三角恒等变换（和差角/二倍角）

<div class="epigraph">
<p>三角恒等变换是三角学的炼金术：把看似不同的表达式，熔炼成同一条等式。</p>
<footer>—— 托勒密（Ptolemy，《天文学大成》）</footer>
</div>

<div class="article-byline">
<p>第一级 · 初等几何与三角 ｜ 人教A版 必修第一册 §5.5 ｜ 2026-08-07</p>
</div>

## 为什么从三角恒等变换开始

同角关系告诉我们同一个角的三函数如何互转；但现实问题经常遇到「**和角**」「**二倍角**」——$\sin(\alpha + \beta)$ 等于什么？$\cos 2\theta$ 能不能写成 $\sin\theta$ 的表达式？**和差角公式**与**二倍角公式**回答了这些问题。它们是三角变形的「引擎」：求值、化简、证明恒等式、推导正弦余弦定理，全都依赖它们。从「从极限到大模型」的主线看，这组公式是「加法如何穿过函数」的第一次系统研究，也是后续微积分里「和差化积、积化和差」的前奏。

## 1 两角和与差的公式

**两角和与差的余弦**：

$$
\cos(\alpha + \beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta
$$

$$
\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta
$$

**两角和与差的正弦**：

$$
\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta
$$

$$
\sin(\alpha - \beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta
$$

**两角和与差的正切**：

$$
\tan(\alpha + \beta) = \frac{\tan\alpha + \tan\beta}{1 - \tan\alpha\tan\beta}, \qquad
\tan(\alpha - \beta) = \frac{\tan\alpha - \tan\beta}{1 + \tan\alpha\tan\beta}
$$

<span class="marginnote">记忆技巧：$\sin$ 展开是「异名相乘、符号不变」（$\sin\cos + \cos\sin$），$\cos$ 展开是「同名相乘、符号相反」（$\cos\cos - \sin\sin$）。正切的分子分母都跟 $\tan$ 的和差有关，分母 $1 \mp \tan\alpha\tan\beta$ 的符号与分子相反。</span>

**重点：** 这些公式的「输入输出」都是**乘积**——两角和的正弦 = 交叉乘积的和。这是「和 → 积」的转换，后面还能反过来用（积化和差）。

## 2 公式解析：两角差的余弦公式证明

为什么 $\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$？用向量证明最简洁，拆三步：

**第一步，单位圆上的两个点**：角 $\alpha$、$\beta$ 的终边与单位圆分别交于 $A(\cos\alpha, \sin\alpha)$、$B(\cos\beta, \sin\beta)$。
**第二步，算两向量夹角的余弦**：$\vec{OA}$ 与 $\vec{OB}$ 的夹角是 $\alpha - \beta$，由向量内积公式 $\cos(\alpha - \beta) = \frac{\vec{OA}\cdot\vec{OB}}{|\vec{OA}||\vec{OB}|}$，而 $|\vec{OA}| = |\vec{OB}| = 1$，$\vec{OA}\cdot\vec{OB} = \cos\alpha\cos\beta + \sin\alpha\sin\beta$。
**第三步，直接读出**：$\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta$。

<span class="marginnote">这个证明展示了「向量内积」的威力：几何角度的差，变成两个单位向量的内积。到第二篇《空间向量与立体几何》你会看到同一思想在三维空间的版本。这也是「数形结合」最优雅的一次合作。</span>

由 $\cos(\alpha - \beta)$ 出发，把 $\beta$ 换成 $-\beta$ 或用诱导公式，就能推出其它三个公式——**两角差的余弦是「母公式」**，其余都是它的推论。

## 3 二倍角公式

在两角和公式中令 $\beta = \alpha$，得到**二倍角公式**：

$$
\sin 2\alpha = 2\sin\alpha\cos\alpha
$$

$$
\cos 2\alpha = \cos^2\alpha - \sin^2\alpha = 2\cos^2\alpha - 1 = 1 - 2\sin^2\alpha
$$

$$
\tan 2\alpha = \frac{2\tan\alpha}{1 - \tan^2\alpha}
$$

**重点：** 二倍角的 $\cos$ 有三种写法，各有用途：

- $\cos 2\alpha = \cos^2\alpha - \sin^2\alpha$：标准形式；
- $\cos 2\alpha = 2\cos^2\alpha - 1$：用于「升幂」（把 $\cos^2\alpha$ 用 $\cos 2\alpha$ 表示）；
- $\cos 2\alpha = 1 - 2\sin^2\alpha$：用于「升幂」（把 $\sin^2\alpha$ 用 $\cos 2\alpha$ 表示）。

反过来用，就是**降幂公式**：

$$
\cos^2\alpha = \frac{1 + \cos 2\alpha}{2}, \qquad \sin^2\alpha = \frac{1 - \cos 2\alpha}{2}
$$

<span class="marginnote">降幂公式在微积分里至关重要：$\int \sin^2 x\, dx$ 的积分靠它把「平方」化成「一次」，才能积出来。到第二级《高等数学》，你会反复用「降幂 → 积分」这个套路。今天埋下的种子，那时收获。</span>

## 4 公式解析：辅助角公式

辅助角公式（也叫「合一变形」）把正弦余弦的线性组合合并成一个正弦：

$$
a\sin x + b\cos x = \sqrt{a^2 + b^2} \cdot \sin(x + \varphi)
$$

其中 $\varphi$ 满足 $\cos\varphi = \frac{a}{\sqrt{a^2+b^2}}$、$\sin\varphi = \frac{b}{\sqrt{a^2+b^2}}$。对这条式子做三步拆解：

- **第一步，提出「模长」**：$a\sin x + b\cos x = \sqrt{a^2 + b^2}\left(\frac{a}{\sqrt{a^2+b^2}}\sin x + \frac{b}{\sqrt{a^2+b^2}}\cos x\right)$。
- **第二步，造出余弦与正弦**：令 $\cos\varphi = \frac{a}{\sqrt{a^2+b^2}}$、$\sin\varphi = \frac{b}{\sqrt{a^2+b^2}}$——因为这两个数的平方和等于 1，正好可以「扮演」一个角的余弦与正弦。
- **第三步，用两角和的正弦**：$\cos\varphi\sin x + \sin\varphi\cos x = \sin(x + \varphi)$，得证。

**重点：** 辅助角公式的价值是把「两个函数」变成「一个函数」——于是求最值、画图像、求单调区间全部回到熟悉的单一正弦波。<span class="marginnote">这是「降维」的经典：两维（$\sin$ 和 $\cos$ 两个正交方向）合并成一维（一个带相位的正弦）。物理里「同频率振动的合成」、信号里「同频信号叠加」用的正是它。</span>

## 5 恒等变换的应用：求值与证明

三角恒等变换的三类典型应用：

**给值求值**：已知 $\sin\alpha = \frac{3}{5}$、$\cos\beta = -\frac{5}{13}$ 等，求 $\sin(\alpha + \beta)$——先分别求出所需函数值（注意象限定符号），再套公式；
**给值求角**：已知函数值求角，先求角的范围再定角——范围决定取舍；
**证明恒等式**：从左到右、从繁到简、统一函数（都化成 $\sin$、$\cos$）、活用「1」的代换。

<span class="marginnote">证明恒等式的通用策略：<strong>从复杂的一边入手，化简到简单的一边</strong>；遇到 $\tan$ 就写成分式；需要统一角度就考虑二倍角/降幂。本质是「朝着目标单向变形」——这正是前面《几何证明的基本方法》里分析法的三角版本。</span>

三角恒等变换还有一个「和差化积」与「积化和差」的方向，虽然高中学得浅，但它是恒等式家族的「另一半」：把乘积变和差（积化和差）或把和差变乘积（和差化积）。例如：

$$
\sin\alpha\cos\beta = \frac{1}{2}\left[\sin(\alpha+\beta) + \sin(\alpha-\beta)\right]
$$

这类公式由「两角和差公式相加/相减」直接得到——把 $\sin(\alpha+\beta)$ 与 $\sin(\alpha-\beta)$ 相加，交叉项抵消，就得到 $\sin\alpha\cos\beta$。它们是「和差角公式」的逆用。

| 恒等式 | 由谁推出 | 用途 |
| --- | --- | --- |
| 二倍角 | 和差角令 $\beta = \alpha$ | 化简、求值 |
| 降幂公式 | 二倍角反解 | 积分、化简平方 |
| 辅助角公式 | 两角和正弦逆用 | 合一变形 |
| 积化和差 | 和差角加减 | 积分、信号处理 |

这张表揭示了恒等式家族的结构：**全部由「两角和差公式」派生**。所以记忆的核心只有四个和差角公式，其余都是它们的推论——「先记母公式，再推子公式」比「逐个死背」高效得多。

<span class="marginnote">「积化和差」在物理与工程里有直接应用：两个声波叠加（乘积）会生成「拍频」——$\sin A \sin B$ 展开成 $\cos(A-B) - \cos(A+B)$ 后，和频与差频分量清晰可见。这是「三角函数公式 = 频谱分析」的初等版本。</span>

恒等变换还有一个「证明恒等式」的常用策略值得强调——**「统一函数、统一角度」**：先看式子里的角度种类（$\alpha, 2\alpha$ 等），用二倍角把角度统一；再看函数种类（$\sin, \cos, \tan$），用同角关系统一成一种。两个「统一」做完，式子通常已化到最简。这个「先统一再化简」的策略，是三角化简的通用兵法。

最后，恒等变换与「计算技巧」的结合——「给值求值」题里常需要**构造目标角**：已知 $\sin\alpha$、$\cos\beta$，求 $\sin(\alpha - \beta)$，关键是把目标角 $\alpha - \beta$ 的展开式需要的 $\cos\alpha$、$\sin\beta$ 全部凑齐（用同角关系补全，注意象限定符号）。「凑齐目标角的展开项」是这类题的核心操作，它与「凑微分」「配方」一样，都是「主动补全结构」的思维。

## 6 小结

- **和差角公式**：$\sin(\alpha \pm \beta)$、$\cos(\alpha \pm \beta)$、$\tan(\alpha \pm \beta)$；$\cos(\alpha-\beta)$ 是「母公式」，向量内积证明最优雅。
- **二倍角公式**：$\sin 2\alpha$、$\cos 2\alpha$（三种写法）、$\tan 2\alpha$；**降幂公式** $\cos^2\alpha = \frac{1+\cos 2\alpha}{2}$ 是微积分的常用工具。
- **辅助角公式**：$a\sin x + b\cos x = \sqrt{a^2+b^2}\sin(x+\varphi)$，把两函数合并成一个正弦。
- 应用三类：给值求值（注意符号）、给值求角（注意范围）、证明恒等式（单向化简）。
- 恒等变换是「加法穿过函数」的研究，通向和差化积、积分与傅里叶分析。

在下一节，我们把三角学用于三角形——研究**正弦定理与余弦定理**，开启解三角形的篇章。

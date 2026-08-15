---
title: Deutsch 算法与 Deutsch-Jozsa 算法
date: 2026-08-07
---

# Deutsch 算法与 Deutsch-Jozsa 算法

<div class="epigraph">
<p>这类问题不能用经典手段高效求解，却能用量子手段求解——这正是量子计算的第一道曙光。</p>
<footer>—— 多伊奇（David Deutsch）与约扎（Richard Jozsa）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§1.4.1 ｜ 2026-08-07</p>
</div>

## 为什么从 Deutsch-Jozsa 算法开始

上一节我们把黑盒模型立了起来，现在用它打第一仗。**Deutsch 问题**（1985）是史上第一个量子算法解决的问题：判定一个单比特函数是常数还是平衡，经典需要 2 次查询，Deutsch 用 1 次；**Deutsch-Jozsa 问题**（1992）把它推广到 $n$ 比特：判定 $f:\{0,1\}^n\to\{0,1\}$ 是常数还是平衡（恰好一半输入输出 0、一半输出 1），经典最坏要 $2^{n-1}+1$ 次查询，而 Deutsch-Jozsa 算法**只用 1 次**——这是史上第一个「指数级查询加速」的证明。<span class="marginnote">Deutsch 发表于 D. Deutsch, "Quantum theory, the Church–Turing principle and the universal quantum computer," <i>Proc. R. Soc. Lond. A</i>` 400 (1985) 97；Deutsch &amp; Jozsa 发表于 <i>Proc. R. Soc. Lond. A</i>` 439 (1992) 553。问题的学术价值远大于实用价值，但它是理解量子算法「并行+干涉」引擎的最佳样例。本节逐行拆开这台引擎。

本节是第五篇《量子算法基础》的第一课。它把《量子查询复杂度》的黑盒语言落地成第一个真实线路——读完你会发现「黑盒 + 相位编码 + 干涉读出」就是后面所有算法（BV、Simon、Shor、Grover）的通用骨架。</span>

## 1 Deutsch 算法：单比特版本

问题：$f:\{0,1\}\to\{0,1\}$，判定是否「常数」（$f(0)=f(1)$）还是「平衡」（$f(0)\ne f(1)$）。线路如下：

1. 两个比特都制备到 $\lvert0\rangle$，各作用 $H$：$\lvert0\rangle\lvert0\rangle \to \lvert+\rangle\lvert+\rangle$。
2. 作用翻转查询 $O_f$（辅助比特设为 $\lvert1\rangle$ 再 $H$ 后为 $\lvert-\rangle$）。
3. 对第一个比特再作用 $H$，测量。

为什么用「翻转查询」$O_f\lvert x\rangle\lvert y\rangle = \lvert x\rangle\lvert y\oplus f(x)\rangle$ 而非普通查询？因为量子查询必须可逆（幺正），而 $y \oplus f(x)$ 把 $f$ 的值「编码」进相位（当辅助为 $\lvert-\rangle$ 时）而不破坏叠加——这是「把经典函数变成量子相位」的标准技巧，在 Simon、Shor 里反复出现。

核心计算：设辅助比特 $\lvert-\rangle$，相位查询下寄存器变为 $(-1)^{f(x)}\lvert x\rangle$。对 $\lvert+\rangle = \frac{1}{\sqrt2}(\lvert0\rangle+\lvert1\rangle)$ 作用后

$$
\frac{1}{\sqrt2}\left[(-1)^{f(0)}\lvert0\rangle + (-1)^{f(1)}\lvert1\rangle\right] = \pm\frac{\lvert0\rangle+\lvert1\rangle}{\sqrt2} \text{ 或 } \pm\frac{\lvert0\rangle-\lvert1\rangle}{\sqrt2}
$$

前者（$f(0)=f(1)$）再 $H$ 后是 $\pm\lvert0\rangle$，后者（$f(0)\ne f(1)$）再 $H$ 后是 $\pm\lvert1\rangle$。<span class="marginnote">判别：测量第一个比特，得 0 说明常数，得 1 说明平衡。经典需要问 $f(0)$ 和 $f(1)$ 两次，Deutsch 只用一次——但只有常数倍的加速。真正的指数加速在下面 $n$ 比特版本。

把 Deutsch 的单比特算一遍：$f(0)=0, f(1)=1$（平衡）。辅助 $\lvert-\rangle$ 下查询给 $\frac{1}{\sqrt2}(\lvert0\rangle - \lvert1\rangle)$，再 $H$ 后是 $\lvert1\rangle$——测量得 1，判平衡。$f \equiv 0$ 时给 $\lvert0\rangle$，测 0，判常数。一次查询、一次测量，两种答案精确对应。</span>

## 2 Deutsch-Jozsa 算法：$n$ 比特版本

问题：$f:\{0,1\}^n \to \{0,1\}$，承诺 $f$ 要么常数要么平衡，判定是哪一种。线路：

1. 前 $n$ 个比特制备到 $\lvert0\rangle^{\otimes n}$，全部作用 $H$；辅助比特制备到 $\lvert-\rangle$。
2. 作用翻转查询 $O_f$。
3. 前 $n$ 个比特全部再作用 $H$，逐一测量。

$H^{\otimes n}$ 把 $\lvert0\rangle^{\otimes n}$ 变成所有 $2^n$ 个计算基态的等幅叠加：

$$
H^{\otimes n}\lvert0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}}\sum_{x\in\{0,1\}^n}\lvert x\rangle
$$

查询后每个分量带上相位 $(-1)^{f(x)}$。**关键**：若 $f$ 常数，则所有相位相同，叠加不受破坏，第二次 $H$ 后回到 $\lvert0\rangle^{\otimes n}$，测量全 0；若 $f$ 平衡，则相位正负各半，干涉相消，测量结果必不全 0。**一次查询判定。**

### 数值例：$f(x) = x_1$（平衡函数）

取 $n = 2$、$f(x) = x_1$（只看第一个比特，四个输入里两个出 0、两个出 1，平衡）。查询后相位为 $(-1)^{x_1}$：$\lvert00\rangle, \lvert01\rangle$ 带 $+1$，$\lvert10\rangle, \lvert11\rangle$ 带 $-1$。第二次 $H^{\otimes2}$ 后，「全 0」振幅 $\frac{1}{4}[(+1)+(+1)+(-1)+(-1)] = 0$——精确相消，测量结果必不全 0。<span class="marginnote">把四种输入写出来手动算一次，你就能「看见」干涉相消：$x_1=0$ 的两项与 $x_1=1$ 的两项在「全 0」方向上的相位正好相反。这种手算对理解所有相位提取算法都极其有效。</span>

再验 $f \equiv 0$（常数）：查询后所有相位 $+1$，状态保持 $\lvert+\rangle^{\otimes n}$；第二次 $H^{\otimes n}$ 精确回到 $\lvert0\rangle^{\otimes n}$，测量必为全 0。两个例子对照，「全 0」的振幅从 1（常数）跳到 0（平衡）——判别完全由干涉相位决定，与函数的具体值无关。

经典为什么最坏要 $2^{n-1}+1$ 次？因为最坏情形下前 $2^{n-1}$ 次查询可能全部给出同一个答案（比如全 0），你无法判断是「常数 0」还是「恰好在另一半是 1 的平衡函数」——必须再多查一个才能区分。这个「确定性最坏情形」的分析，正是查询复杂度下界的标准推理。

## 3 公式解析：为什么「全 0」的概率足以判别

把测量概率算出来，是理解这台引擎的关键一步。第二次 $H^{\otimes n}$ 后，测量结果为某个 $y$ 的振幅是

$$
\tilde{f}(y) = \frac{1}{2^n}\sum_{x\in\{0,1\}^n} (-1)^{f(x) + x\cdot y}
$$

- **第一步，写振幅**：$H^{\otimes n}$ 把 $\lvert x\rangle$ 映到 $\frac{1}{\sqrt{2^n}}\sum_y (-1)^{x\cdot y}\lvert y\rangle$（$x\cdot y$ 是模 2 内积）。查询相位 $(-1)^{f(x)}$ 与 $H$ 相位合并，得上面的和式。
- **第二步，看 $y=0$**：当 $y=0$ 时 $x\cdot y = 0$，于是 $\tilde{f}(0) = \frac{1}{2^n}\sum_x (-1)^{f(x)}$。
- **第三步，两种情形**：$f$ 常数 ⇒ $(-1)^{f(x)}$ 全为 $+1$ 或全为 $-1$ ⇒ $\lvert\tilde{f}(0)\rvert = 1$ ⇒ 测量必为全 0。$f$ 平衡 ⇒ $\sum_x (-1)^{f(x)} = 0$（正负各半）⇒ $\tilde{f}(0) = 0$ ⇒ 测量全 0 的概率为 0。<span class="marginnote">这就是干涉的定量面目：<strong>平衡函数让「全 0」这个结果的振幅精确相消</strong>，而常数函数让它们相长。一次查询 + 两次 Hadamard 层，完成经典需要指数次查询的判断——加速全部来自「把 $2^n$ 个值的相位同时叠加、再由一次干涉统一读出」。

还有一个值得注意的点：Deutsch-Jozsa 的「一次查询」不需要知道 $f$ 的任何值——它把「函数性质」整个压进了相位分布，再让干涉读出「全 0 的振幅是否为 1」。这种「不读数值、只读性质」的范式，是量子算法与经典算法的根本分水岭。</span>

## 4 公式解析：$H^{\otimes n}$ 的作用与逆作用

Deutsch-Jozsa 只有两招：$H^{\otimes n}$ 开叠加、$H^{\otimes n}$ 收干涉。为什么同一门能既开又收？因为 $H^{\otimes n}$ 是**自逆**的（$H^2 = I$）。展开看它对单分量 $\lvert x\rangle$ 的作用：

$$
H^{\otimes n}\lvert x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y\in\{0,1\}^n} (-1)^{x\cdot y}\lvert y\rangle
$$

- **第一步，逐比特 $H$**：每个 $\lvert x_i\rangle$ 被 $H$ 作用为 $\frac{1}{\sqrt2}(\lvert0\rangle + (-1)^{x_i}\lvert1\rangle)$。
- **第二步，张量积展开**：$n$ 个这样因子相乘，$(-1)^{x\cdot y}$ 正是把 $x_i y_i$ 的相位逐位相乘拼起来。
- **第三步，自逆性**：因为 $H^{\otimes n}$ 的矩阵对称且 $H^2=I$，用两次就回到原状——这正是「先开后收」的数学基础。<span class="marginnote">这套「$H$ 开、$H$ 收」的模式是所有「黑盒相位提取」算法的骨架：Deutsch-Jozsa、Bernstein-Vazirani、Simon 全是它的变体。区别只在查询后插入的相位函数 $f$ 与第二次变换的选法。</span>

$H^{\otimes n}$ 的自逆性还带来一个工程便利：同一个「开叠加」的门可以直接复用为「收干涉」的门——线路不需要额外的「逆 QFT」类结构（区别于相位估计里 QFT 与其逆）。Deutsch-Jozsa 是整个课程里「对称最干净」的算法之一。

## 5 局限与意义

**辨析｜易错点：** Deutsch-Jozsa 的「指数加速」是**查询**意义上的，且依赖「平衡或常数」的承诺。若 $f$ 既不常数也不平衡，算法可能给出任何结果——它不是一个通用的「判断 $f$ 性质」算法，而是针对特定承诺问题的。另外，若比较的是**确定性**经典算法（$2^{n-1}+1$ 次），加速是指数的；若允许随机化经典算法，则只需 $O(1)$ 次查询（随机抽几个输入试）——**随机化经典几乎抹平了这个加速**。这个「对随机化经典不再指数」的事实，常被当作「查询模型里要小心定义加速」的教材案例。

不过它的历史意义无可替代：它第一次证明**量子计算能在某个问题上有结构性的查询优势**，让 Feynman 的猜想有了可验证的具体算法，也为 Simon、Shor 铺了路。

几个「相位提取」算法的谱系对照：

| 算法 | 函数承诺 | 查询次数 | 读出什么 |
| --- | --- | --- | --- |
| Deutsch | 单比特常数/平衡 | 1 | 常数 or 平衡 |
| Deutsch-Jozsa | $n$ 比特常数/平衡 | 1 | 常数 or 平衡 |
| Bernstein-Vazirani | 内积函数 | 1 | 隐藏比特串 $s$ |
| Simon | 周期函数 | $O(n)$ | 周期 $s$ |

## 6 小结

- **Deutsch 问题**：单比特常数/平衡判定，经典 2 次查询，量子 1 次。
- **Deutsch-Jozsa**：$n$ 比特版本，经典最坏 $2^{n-1}+1$ 次，量子 **1 次**（指数查询加速）。
- 引擎 = **$H$ 开叠加 → 查询编码相位 → $H$ 收干涉**；平衡时「全 0」振幅精确相消。
- **局限**：需要「常数或平衡」承诺；对随机化经典不再指数加速。
- **引擎三招**：$H^{\otimes n}$ 开叠加 → 查询编码相位 → $H^{\otimes n}$ 收干涉。
- **翻转查询**：$y \oplus f(x)$ 把函数值编码进相位，是「经典→量子相位」的标准技巧。
- **谱系**：DJ、BV、Simon、Shor 共享「$H$ 开、相位、$H$ 收」的引擎，区别只在函数与后处理。

在下一节，我们沿用同一台引擎，但让 $f$ 的结构更丰富——**Bernstein-Vazirani 算法**用一次查询读出整个隐藏比特串。

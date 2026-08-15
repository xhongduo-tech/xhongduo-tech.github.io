---
title: 依测度收敛：定义与性质
date: 2026-08-07
---

# 依测度收敛：定义与性质

<div class="epigraph">
<p>当收敛不再由「每个点」判定，而由「每个区域偏离多少」判定，我们进入了测度的语言。</p>
<footer>—— 弗里杰什 · 里斯（Frigyes Riesz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§4.7 ｜ 2026-08-07</p>
</div>

## 为什么从依测度收敛开始

a.e. 收敛是「逐点」的放宽，但它仍以「点」为判断单位。本节引入一种全新的收敛模式——**依测度收敛**：不看单点，而看「函数偏离超过阈值的集合的测度」。$f_k$ 依测度收敛到 $f$，指「$|f_k-f|\ge\varepsilon$ 的区域」测度趋于零。

依测度收敛的重要性有三重：它是**概率论中依概率收敛**的化身（大数定律的弱形式正是依概率收敛）；它是**度量 L^p 收敛与 a.e. 收敛的中介**（下节 Riesz 定理说依测度收敛蕴含 a.e. 收敛的子列）；它还是**积分极限理论**里最自然的收敛概念——控制收敛定理的结论正是依测度意义下的积分收敛。<span class="marginnote">依测度收敛之所以「自然」，在于它<strong>不依赖具体点</strong>：两个函数在零测集上怎么改都不影响依测度收敛。它是「a.e. 等价类」上的收敛，与 Lebesgue 理论的「函数=等价类」观念完全一致。概率论里「$X_n\overset{P}{\to}X$」几乎一字不差。</span>

## 1 依测度收敛的定义

**定义（依测度收敛）**：设 $\{f_k\}$、$f$ 是 $E$ 上的可测函数（有限 a.e.）。若对任意 $\varepsilon>0$，

$$\lim_{k\to\infty}m\left(\left\{x\in E:|f_k(x)-f(x)|\ge\varepsilon\right\}\right)=0$$

则称 $\{f_k\}$ 在 $E$ 上**依测度收敛（convergence in measure）**到 $f$，记作 $f_k\overset{m}{\longrightarrow}f$。

**重点：测度趋零的不是「误差本身」，而是「误差超阈值的区域」。** $m(\{|f_k-f|\ge\varepsilon\})$ 度量「坏区域面积」，而非误差大小。误差可以很大（$f_k-f$ 无界），只要「误差大的区域」越来越小就行。例如 $f_k=k\chi_{[0,1/k]}$：对任意 $\varepsilon>0$，$\{|f_k-0|\ge\varepsilon\}=[0,1/k]$，测度 $1/k\to0$，故 $f_k\overset{m}{\to}0$——**函数值在收缩区间上冲高，但区间面积归零**。

## 2 依测度收敛的基本性质

**性质一（极限唯一性 a.e.）**：若 $f_k\overset{m}{\to}f$ 且 $f_k\overset{m}{\to}g$，则 $f=g$ a.e.。证明用「三角分割」：$\{|f-g|\ge\varepsilon\}\subset\{|f-f_k|\ge\varepsilon/2\}\cup\{|f_k-g|\ge\varepsilon/2\}$，两边测度趋零，故 $m(\{|f-g|\ge\varepsilon\})=0$ 对每个 $\varepsilon$，即 $f=g$ a.e.。<span class="marginnote">「三角分割」是证明中处理「两个收敛的差距」的标准手法：中间插一个 $f_k$，把 $|f-g|$ 拆成两段。这个手法在分析学里是万能胶——只要收敛性对「区域测度」成立，不等式就在集合层面传递。</span>

**性质二（与 a.e. 收敛的关系：有限测度下）**：设 $m(E)<\infty$。若 $f_k\to f$ a.e.，则 $f_k\overset{m}{\to}f$。

证明：由 Egorov 定理，$f_k$ 近一致收敛：对任意 $\varepsilon,\delta>0$ 存在 $E_\delta$，$m(E\setminus E_\delta)<\delta$，$f_k\rightrightarrows f$ 于 $E_\delta$。取 $K$ 使 $k\ge K$ 时 $|f_k-f|<\varepsilon$ 于 $E_\delta$，则 $\{|f_k-f|\ge\varepsilon\}\subset E\setminus E_\delta$，测度 $<\delta$。令 $\delta\to0$。**a.e. 收敛（有限测度）⇒ 依测度收敛**。

**性质三（反方向不成立）**：依测度收敛不蕴含 a.e. 收敛。经典反例「移动冒泡」：$f_1=\chi_{[0,1/2]}$，$f_2=\chi_{[1/2,1]}$，$f_3=\chi_{[0,1/4]}$，$f_4=\chi_{[1/4,1/2]}$，$f_5=\chi_{[1/2,3/4]}$，$f_6=\chi_{[3/4,1]}$，……（把 $[0,1]$ 反复二分，依次点亮）。对任意 $\varepsilon>0$，$\{|f_k-0|\ge\varepsilon\}$ 是当前被点亮的区间，测度趋于 $0$，故 $f_k\overset{m}{\to}0$；但对每个 $x$，$f_k(x)$ 无限次取 $1$ 又无限次取 $0$，**处处不收敛**。<span class="marginnote">「移动冒泡」反例的精髓：坏区域不断搬迁（从一个二分区间跳到下一个），<strong>没有哪个点被固定地「长期弄坏」</strong>——于是测度趋零而逐点永不收敛。它说明「测度」看的是「总坏面积」，对「坏点是否固定」无感。</span>

**性质四（Cauchy 性）**：若对任意 $\varepsilon>0$，$m(\{|f_j-f_k|\ge\varepsilon\})\to0$（当 $j,k\to\infty$），称 $\{f_k\}$ 依测度 Cauchy。依测度收敛的列必依测度 Cauchy（三角分割），下节证明逆定理（Riesz）——所以依测度收敛空间是完备的。

## 3 与 L^p 收敛的关系

**定理（L^p 收敛 ⇒ 依测度收敛）**：若 $f_k\overset{L^p}{\to}f$（即 $\int|f_k-f|^p\to0$），则 $f_k\overset{m}{\to}f$。

证明用 Markov/Chebyshev 型不等式：

$$m\left(\{|f_k-f|\ge\varepsilon\}\right)\le\frac{1}{\varepsilon^p}\int_E|f_k-f|^p\,dm\ \xrightarrow{k\to\infty}0$$

（积分的定义与性质在第五篇，这里先借结果。）**L^p 收敛是依测度收敛的强化版**——不仅坏区域面积趋零，坏的程度（$p$ 次方积分）也趋零。

**辨析｜易错点：依测度收敛不蕴含 L^p 收敛。** $f_k=k\chi_{[0,1/k]}$ 依测度收敛到 $0$，但 $\int|f_k|^p=k^p\cdot\tfrac1k=k^{p-1}\to\infty$（$p>1$）——**函数在收缩区间上冲得过高，能量发散**。依测度收敛只控制「坏区面积」，不控制「坏区强度」，后者需要积分条件（可积控制、p 次可积）。

## 4 公式解析：三角分割不等式

依测度收敛理论的核心不等式是三角分割：

$$\left\{|f-g|\ge\varepsilon\right\}\subset\left\{|f-f_k|\ge\frac{\varepsilon}{2}\right\}\cup\left\{|f_k-g|\ge\frac{\varepsilon}{2}\right\}$$

- **第一步，读「集合包含」**：若 $|f(x)-g(x)|\ge\varepsilon$，则要么 $|f(x)-f_k(x)|\ge\varepsilon/2$，要么 $|f_k(x)-g(x)|\ge\varepsilon/2$（否则两项之和 $<\varepsilon$ 矛盾）。**「误差大」必可归因于某一段「半误差」**——三角不等式在集合层面的翻版。
- **第二步，读「测度次可加」**：$m(\{|f-g|\ge\varepsilon\})\le m(\{|f-f_k|\ge\varepsilon/2\})+m(\{|f_k-g|\ge\varepsilon/2\})$。**次可加性把「总坏面积」控制在「两段坏面积之和」**。
- **第三步，读「极限传递」**：若 $f_k\overset{m}{\to}f$ 与 $f_k\overset{m}{\to}g$，右边两项各趋零，故 $m(\{|f-g|\ge\varepsilon\})=0$ 对每个 $\varepsilon$——**极限唯一性**。若 $f_k$ 依测度 Cauchy，令 $g$ 是「假想极限」，同理得极限存在性。

**三角分割 + 次可加性** 是依测度收敛全部性质的标准推导机器——记住这两件套，性质一与 Riesz 定理（下节）都能顺流而下。

## 5 数值演练与收敛谱系

**算例一（收缩冲高：依测度收敛但非 L^p）**：$f_k=k\chi_{[0,1/k]}$。对任意 $\varepsilon>0$，$\{|f_k|\ge\varepsilon\}=[0,1/k]$（$k>\varepsilon$ 时），$m=1/k\to0$，依测度收敛到 0；但 $\int|f_k|^p=k^p\cdot\tfrac1k=k^{p-1}$，$p>1$ 时 $\to\infty$——依测度收敛不蕴含 L^p 收敛。若改取 $f_k=\tfrac1{\sqrt k}\chi_{[0,1/k]}$，$\int f_k^2=\tfrac1k\cdot\tfrac1k=k^{-2}\to0$，此时既依测度又 L² 收敛——「冲高」与「收窄」的权衡决定积分是否收敛。

**算例二（移动冒泡的测度跟踪）**：设 $I_k$ 为按长度 $\to0$ 扫过 $[0,1]$ 的二分区间列，$f_k=\chi_{I_k}$。$m\{|f_k|>\tfrac12\}=|I_k|\to0$，故 $f_k\overset{m}{\to}0$；但每个 $x$ 都被无穷多次点亮，$f_k(x)$ 处处不收敛。**把「测度收敛」与「逐点收敛」放在同一个函数列上对照，是理解两种收敛差异的最佳练习。**

**对照表：五种收敛的谱系**

| 收敛 | 判据 | 蕴含关系（有限测度） |
| --- | --- | --- |
| 一致收敛 | $\sup\|f_k-f\|\to0$ | 最强 |
| a.e. 收敛 | 逐点收敛除零测集 | 一致 ⇒ a.e. |
| 依测度收敛 | 坏区测度 $\to0$ | a.e. ⇒ 依测度（Egorov） |
| L^p 收敛 | $\int\|f_k-f\|^p\to0$ | L^p ⇒ 依测度（Markov） |
| 依概率收敛 | 同依测度 | 概率论别名 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| $f_k\overset{m}{\to}f$ | 依测度收敛 |
| 坏区 | $\{\|f_k-f\|\ge\varepsilon\}$ |
| Markov 不等式 | $m\{\|g\|\ge\varepsilon\}\le\varepsilon^{-p}\int\|g\|^p$ |
| 移动冒泡 | 坏区不断搬迁的反例族 |

**辨析｜易错点：依测度收敛的极限「按 a.e. 等价类」唯一，不是逐点唯一。** 若 $f=g$ a.e.，$f_k\overset{m}{\to}f$ 与 $f_k\overset{m}{\to}g$ 同时成立——极限是一族「a.e. 相等」的函数。因此在依测度语境下，谈论「极限函数」时应默认 a.e. 等价类。

### 极限唯一性：三角分割的完整演示

- **目标**：$f_k\overset{m}{\to}f$ 且 $f_k\overset{m}{\to}g$ ⇒ $f=g$ a.e.。
- **三角分割**：$\{\|f-g\|\ge\varepsilon\}\subset\{\|f-f_k\|\ge\tfrac\varepsilon2\}\cup\{\|f_k-g\|\ge\tfrac\varepsilon2\}$。
- **次可加**：坏区测度 $\le$ 两段之和 $\to0$。
- **逐 $\varepsilon$ 归零**：$m\{\|f-g\|\ge\varepsilon\}=0$ 对每个 $\varepsilon>0$。
- **并零测**：$\{f\neq g\}=\bigcup_n\{\|f-g\|\ge\tfrac1n\}$ 零测。

**延伸（概率论视角）**：$X_n\overset{P}{\to}X$ 即依概率收敛。大数定律（弱）恰是「样本均值依概率收敛到期望」；而「依概率收敛 ⇒ 存在几乎必然收敛子列」正是下节 Riesz 定理的概率论表述——从弱收敛中萃取强收敛。

**为什么「有限测度」关键**：a.e. 收敛 ⇒ 依测度收敛需要 Egorov，而 Egorov 假设 $m(E)<\infty$；若 $m(E)=\infty$（如 $E=\mathbb{R}$），$f_k=\chi_{[k,\infty)}$ 逐点收敛到 0 但不依测度收敛（$m\{f_k\ne0\}=\infty$）。**无限测度时两套收敛脱钩。**

**一道自测题**：设 $f_k$ 依测度收敛到 $f$，$g_k$ 依测度收敛到 $g$。证明 $f_k+g_k$ 依测度收敛到 $f+g$（用三角分割：$\{\|(f_k+g_k)-(f+g)\|\ge\varepsilon\}\subset\{\|f_k-f\|\ge\varepsilon/2\}\cup\{\|g_k-g\|\ge\varepsilon/2\}$）。

## 6 小结

- **依测度收敛**：$m(\{|f_k-f|\ge\varepsilon\})\to0$ 对每个 $\varepsilon$；只看「坏区面积」。
- **有限测度下**：a.e. 收敛 ⇒ 依测度收敛（Egorov 桥）。
- **反向不成立**：移动冒泡反例——坏区搬迁，测度归零而逐点永不收敛。
- **与 L^p 关系**：L^p ⇒ 依测度（Markov 不等式）；反向不成立（能量发散例）。
- **核心工具**：三角分割 + 次可加性，极限唯一性、Cauchy 性由此导出。
- **测度谱系**：一致 ⊂ a.e. ⊂ 依测度 ⊂ ？——依测度最弱之一，但抽子列可得 a.e.。
- **前提**：「a.e. ⇒ 依测度」需要 $m(E)<\infty$；无限测度时脱钩。

在下一节，我们证明 **Riesz 定理**：依测度收敛的函数列必有几乎处处收敛的子列——这是「从测度收敛中提炼逐点收敛」的惊人桥梁。

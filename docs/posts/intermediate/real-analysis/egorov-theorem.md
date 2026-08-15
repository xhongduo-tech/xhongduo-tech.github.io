---
title: 叶戈罗夫（Egorov）定理：a.e. 收敛与近一致收敛
date: 2026-08-07
---

# 叶戈罗夫（Egorov）定理：a.e. 收敛与近一致收敛

<div class="epigraph">
<p>在有限测度的舞台上，几乎处处收敛其实是一种伪装的「几乎一致收敛」——除去任意小的舞台角落，其余地方整齐划一。</p>
<footer>—— 德米特里 · 叶戈罗夫（Dmitri Egorov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§4.6 ｜ 2026-08-07</p>
</div>

## 为什么从 Egorov 定理开始

前节建立的 a.e. 收敛是分析学最常用的收敛模式，但它太弱：a.e. 收敛不保证极限函数的积分等于积分的极限（需要额外条件）。Egorov 定理是连接 a.e. 收敛与一致收敛的桥梁：**在有限测度集上，a.e. 收敛可以在「除去一个任意小测度集合」之后变成一致收敛**。

这条定理的重要性在于它把「a.e. 收敛」翻译成「几乎一致收敛」，而一致收敛拥有最强大的运算性质（保持连续、可交换极限与积分）。Egorov 定理与 Luzin 定理（后两节）一起构成「测度论的正则性三件套」：可测对象总能用「更规则」的对象在「测度误差任意小」的意义下逼近。<span class="marginnote">Egorov 定理的条件「$m(E)<+\infty$」是必要的：$f_k=\chi_{[k,k+1]}$ 在 $\mathbb{R}$ 上 $f_k\to0$ a.e.，但任何「除去小测度后一致收敛」都失败——因为坏点「均匀地撒在无穷远处」。<strong>有限测度是「测度论正则性」的地基</strong>，这是 Lebesgue 测度与抽象局部紧群测度共有的特征。</span>

## 1 近一致收敛

**定义（近一致收敛）**：设 $\{f_k\}$、$f$ 可测。若对任意 $\varepsilon>0$，存在可测集 $E_\varepsilon$，使

$$m(E\setminus E_\varepsilon)<\varepsilon,\qquad f_k\rightrightarrows f\ \text{于}\ E_\varepsilon$$

即：**在 $E_\varepsilon$ 上一致收敛，且被排除的坏集测度任意小**，则称 $\{f_k\}$ 在 $E$ 上**近一致收敛（almost uniformly）**到 $f$。

**重点：近一致收敛 ≠ a.e. 收敛的等价物，而是更强。** 近一致收敛 ⇒ a.e. 收敛（取 $E_k=E_{1/k}$，坏点集 $\bigcup_k(E\setminus E_{1/k})$ 零测）；但 a.e. 收敛不必然近一致收敛（无穷测度反例）。Egorov 定理说：**有限测度下两者等价**。在概率论中「近一致收敛」对应「几乎必然收敛的加强版」，是随机过程收敛分析的重要工具。

## 2 Egorov 定理

**定理（Egorov）**：设 $E\subset\mathbb{R}^n$ 可测，$m(E)<+\infty$，$\{f_k\}$ 是 $E$ 上的可测函数列，$f_k\to f$ a.e.（$f$ 有限 a.e.）。则 $\{f_k\}$ 在 $E$ 上**近一致收敛**到 $f$。

**证明**（核心是「用可数并控制坏点」）：对固定的 $\varepsilon>0$，目标是找 $E_\varepsilon\subset E$，$m(E\setminus E_\varepsilon)<\varepsilon$，使 $f_k\rightrightarrows f$ 于 $E_\varepsilon$。

- **第一步，定义「坏点集」**：对 $j$、$N$ 定义
$$E_j(N)=\bigcup_{k=N}^{\infty}\left\{x:|f_k(x)-f(x)|\ge\frac1j\right\}$$
$E_j(N)$ 是「从第 $N$ 项开始还有误差 $\ge1/j$ 的点」。由 $f_k\to f$ a.e.，对每个 $j$，
$$\bigcap_{N=1}^{\infty}E_j(N)\ \text{零测（坏点集）},\qquad m(E_j(N))\downarrow 0\ (N\to\infty)$$
- **第二步，用测度连续性压缩**：由递减连续性（$m(E)<\infty$），对每个 $j$ 存在 $N_j$ 使 $m(E_j(N_j))<\tfrac{\varepsilon}{2^j}$。
- **第三步，取排除集**：令 $E_\varepsilon=E\setminus\bigcup_{j=1}^{\infty}E_j(N_j)$。则 $m(E\setminus E_\varepsilon)\le\sum_jm(E_j(N_j))<\varepsilon$。
- **第四步，证一致收敛**：对任意 $\delta>0$，取 $j$ 使 $1/j<\delta$。在 $E_\varepsilon$ 上，$x\notin E_j(N_j)$，故对所有 $k\ge N_j$，$|f_k(x)-f(x)|<1/j<\delta$——**与 $x$ 无关的 $N_j$ 控制住一切点**，一致收敛。<span class="marginnote">证明的枢纽是「$m(E_j(N))\downarrow0$」——由 $m(E)<\infty$ 与测度的递减连续性（第三篇）保证。若 $m(E)=\infty$，这个单调下降到零不成立：$E_j(N)$ 可能测度恒为无穷。这再次印证「有限测度」是 Egorov 的心脏。</span>

## 3 Egorov 定理的意义与推论

**推论一（a.e. 收敛 ≈ 近一致收敛，有限测度下）**：结合定义中的「近一致 ⇒ a.e.」，在 $m(E)<\infty$ 时，**a.e. 收敛 ⇔ 近一致收敛**。

**推论二（连续函数的逼近）**：若 $f$ 是有限测度集 $E$ 上的可测函数，则对任意 $\varepsilon>0$，存在连续函数 $g$ 使 $m(\{f\neq g\})<\varepsilon$（这是 Luzin 定理的弱形式，下节证明强版本）。

**应用（积分交换的桥梁）**：Egorov 定理让「$\lim\int=\int\lim$」的证明有了抓手：先一致收敛（此时交换合法），再处理被排除的小测度集（其贡献可控制在任意小）。这是控制收敛定理证明的经典路径之一。<span class="marginnote">Egorov 定理与<strong>控制收敛定理</strong>的关系深刻：前者把「a.e. 收敛」升级为「近一致」，后者用「可积控制函数」接管小测度集上的误差。<strong>一个控制「形状」，一个控制「面积」</strong>，合起来就是第六篇最强大的极限交换工具。</span>

**辨析｜易错点：Egorov 给的「一致收敛」只在 $E_\varepsilon$ 上，$E_\varepsilon$ 的构造依赖 $\varepsilon$。** 不能指望「在整个 $E$ 上一致」——a.e. 收敛不保证一致收敛（如锯齿波逼近方波，逐点收敛但非一致）。Egorov 的智慧正在于：**允许排除任意小的坏集，换来「整齐」**。这是「牺牲一点测度，换取全局规则」的权衡艺术。

## 4 公式解析：$E_j(N)$ 的双指标收缩

Egorov 证明的灵魂是「双指标 $E_j(N)$」的收缩结构：

$$E_j(N)=\bigcup_{k=N}^{\infty}\left\{|f_k-f|\ge\frac1j\right\},\qquad \bigcap_{N=1}^{\infty}E_j(N)=\varnothing\ \text{a.e.}$$

$$m(E_j(N_j))<\frac{\varepsilon}{2^j},\qquad \bigcup_{j}E_j(N_j)\ \text{测度}<\varepsilon$$

- **第一步，读「$E_j(N)$ 的含义」**：固定误差阈值 $1/j$ 与起始下标 $N$，$E_j(N)$ 收集「尾部还有 $1/j$ 误差」的点。**它是「坏点」的分层账本**——误差越大、起始越晚，集合越小。
- **第二步，读「$N\to\infty$ 收缩为零」**：$\bigcap_NE_j(N)$ 是「对任意 $N$ 都坏」的点，恰是收敛失败点集，零测。由 $m(E)<\infty$，递减连续性给出 $m(E_j(N))\downarrow0$——**每个 $j$ 层的坏点可以压到任意小**。
- **第三步，读「$\varepsilon/2^j$ 分配」**：每层误差预算 $\varepsilon/2^j$，可数层求和仍 $<\varepsilon$。**这是测度论「$\varepsilon/2^k$」分配法的又一次登场**——把无穷多个「每层小误差」汇总为总误差受控。
- **第四步，读「统一 $N_j$ 控制」**：在排除集外，第 $j$ 层从 $N_j$ 起误差 $<1/j$。对任意 $\delta$ 取 $j$ 大，$N_j$ 成为「与 $x$ 无关」的起点——**一致收敛的本质是「存在统一起点」，Egorov 用排除集制造了这个起点**。

## 6 数值演练与术语速查

**算例一（锯齿波逼近方波的 Egorov 演示）**：$f_k$ 在 $[0,1]$ 上用斜率 $k$ 的锯齿逼近 $f=\chi_{[\tfrac12,1]}$。$f_k\to f$ a.e.，但逐点收敛非一致（在 $\tfrac12$ 附近斜率无限陡）。Egorov 保证：对任意 $\varepsilon$，排除一个测度 $<\varepsilon$ 的小邻域 $(\tfrac12-\delta,\tfrac12+\delta)$ 后，$f_k\rightrightarrows f$——**坏区只在跳跃点附近，且可任意缩小**。

**算例二（无穷测度的失败）**：$E=\mathbb{R}$，$f_k=\chi_{[k,k+1]}\to0$ a.e.，但任何「排除小测度后一致收敛」都失败：在任意「补集测度有限」的集合上，总有无穷多个 $k$ 使 $f_k=1$。**有限测度是 Egorov 成立的前提。**

**对照表：收敛模式的强度谱系**

| 模式 | 定义 | Egorov 之后 |
| --- | --- | --- |
| 一致收敛 | $\sup\|f_k-f\|\to0$ | 保持连续性、交换极限 |
| 近一致收敛 | 除小测度外一致 | 有限测度下 ⇔ a.e. |
| a.e. 收敛 | 逐点收敛除零测集 | 有限测度下可升级为近一致 |
| 依测度收敛 | 坏区测度 $\to0$ | 下一节 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| $f_k\rightrightarrows f$ | 一致收敛 |
| 近一致收敛 | 除小测度集外一致收敛 |
| $E_j(N)$ | 尾部误差 $\ge1/j$ 的坏点分层 |
| $\varepsilon/2^j$ 分配 | 可数层误差预算 |

**辨析｜易错点：近一致收敛中的「排除集」依赖 $\varepsilon$，不可让 $\varepsilon\to0$ 后取并。** $\bigcup_kE_{1/k}$ 的补集可能不是「一个测度为零的集合」——Egorov 只给「任意小测度例外」，不给「零测例外」；要零测例外需更强的条件（如 Luzin 点集版反例所示）。

### 三步记住 Egorov 证明

- **分层**：$E_j(N)=\bigcup_{k\ge N}\{\|f_k-f\|\ge1/j\}$，每层随 $N$ 收缩为零（$m(E)<\infty$）。
- **预算**：$m(E_j(N_j))<\varepsilon/2^j$，可数层求和仍 $<\varepsilon$。
- **统一**：排除 $\bigcup_jE_j(N_j)$ 后，第 $j$ 层从 $N_j$ 起误差 $<1/j$——「统一起点」即一致收敛。

**延伸（与控制收敛的连接）**：Egorov 证明中「先近一致、再处理小测度集」正是控制收敛定理的经典路径——近一致给「形状」，可积控制给「面积」，两者合成极限交换的完整论证。下一节的依测度收敛则是从另一维度（测度而非逐点）重新组织同一图景。

**为什么「近一致」比「a.e.」好用**：一致收敛保持连续性、可交换极限与积分；a.e. 收敛没有这些性质。Egorov 用「牺牲小测度」把 a.e. 收敛升级为可用的一致收敛——代价极小，收益巨大。

**从 Egorov 到 Luzin 的接力**：Egorov 管「函数列」的收敛，Luzin 管「单个函数」的连续性——两者都靠「排除小测度」换取强结构，构成正则性三件套（可测集逼近、Egorov、Luzin）的核心。

**一道收束练习**：设 $f_k\to f$ a.e. 于 $m(E)<\infty$。证明存在递增下标 $k_j$ 使 $m\{\sup_{k\ge k_j}|f_k-f|>2^{-j}\}<2^{-j}$（用 $E_j(N)$ 与 $\varepsilon/2^j$ 分配直接构造）——这预告了 Riesz 定理的抽子列技巧。

## 7 小结

- **近一致收敛**：除去任意小测度集后一致收敛；强于 a.e. 收敛。
- **Egorov 定理**：$m(E)<\infty$ 且 $f_k\to f$ a.e. ⇒ $f_k$ 近一致收敛。
- **证明枢纽**：双指标 $E_j(N)$ + 递减连续性 + $\varepsilon/2^j$ 分配。
- **推论**：有限测度下 a.e. ⇔ 近一致；可测函数被连续函数在测度误差下逼近（Luzin 弱形式）。
- **哲学**：「牺牲小测度，换取全局规则」——测度论正则性的核心权衡。
- **前提**：$m(E)<\infty$ 不可省——$f_k=\chi_{[k,k+1]}$ 于 $\mathbb{R}$ 即反例。
- **衔接**：Egorov（列收敛）+ Luzin（单函数连续）= 正则性双引擎。

在下一节，我们引入**依测度收敛**，并研究它与 a.e. 收敛的精确关系（Riesz 定理）——这是收敛模式的另一维度。

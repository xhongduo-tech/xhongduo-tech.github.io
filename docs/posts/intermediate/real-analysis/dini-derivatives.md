---
title: Dini 导数与单调函数导数的存在性
date: 2026-08-07
---

# Dini 导数与单调函数导数的存在性

<div class="epigraph">
<p>当导数不存在时，还有四个 Dini 导数在站岗——它们总能给出「导数缺席」的精确界限。</p>
<footer>—— 乌利塞 · 迪尼（Ulisse Dini）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第六章 ｜ 2026-08-07</p>
</div>

## 为什么从 Dini 导数开始

上一节提到 Lebesgue 定理的证明依赖「Dini 导数的一致化」。本节把 Dini 导数独立成主题：它是**导数不存在的精确替代品**——当 $\lim$ 不存在时，$\limsup$ 与 $\liminf$ 永远存在（允许无穷），于是「可微性」被分解为「四个 Dini 导数相等」这个可检查的条件。

Dini 导数的价值在于「存在性免费」：任何函数在任何点都有四个 Dini 导数（在扩充实数意义下）。这让「不可微点」的分析成为可能——单调函数不可微的点，恰好是「四个 Dini 导数不一致」的点。本节详细展开定义、性质、以及在单调函数可微性证明中的作用。<span class="marginnote">Dini 导数的思想与数列的上下极限一脉相承：<strong>「极限不存在」被「上下极限的区间」代替</strong>。分析学里凡是「某极限可能不存在」的场景，都靠 $\limsup/\liminf$ 兜底——从数列到函数，从 Dini 导数到泛函分析的弱极限，同一哲学反复出现。</span>

## 1 四个 Dini 导数的定义

**定义（Dini 导数）**：设 $f$ 在 $x$ 附近有定义。定义差商 $Q_x(h)=\tfrac{f(x+h)-f(x)}{h}$。四个 **Dini 导数**为：

$$\underline{D}^-f(x)=\liminf_{h\to0^-}\frac{f(x+h)-f(x)}{h},\qquad \overline{D}^-f(x)=\limsup_{h\to0^-}\frac{f(x+h)-f(x)}{h}$$

$$\underline{D}^+f(x)=\liminf_{h\to0^+}\frac{f(x+h)-f(x)}{h},\qquad \overline{D}^+f(x)=\limsup_{h\to0^+}\frac{f(x+h)-f(x)}{h}$$

其中 $h\to0^-$ 是左极限（$h<0$），$h\to0^+$ 是右极限。**$\overline{D}$ 与 $\underline{D}$ 分别代表「上、下」，$-$ 与 $+$ 代表「左、右」**。

**性质（对单调函数）**：$f$ 单调不减时，差商 $Q_x(h)\ge0$（$h>0$）或 $\le0$（$h<0$），四个 Dini 导数都不小于 $0$。且存在天然的偏序：$\underline{D}^-f\le\overline{D}^-f$、$\underline{D}^+f\le\overline{D}^+f$ 恒成立（$\liminf\le\limsup$）。

**重点：Dini 导数永远存在（允许 $\pm\infty$）。** 与导数不同，Dini 导数不需要「极限存在」——$\limsup$ 与 $\liminf$ 作为扩展实数总有值。**「可微」的判据变成「四者相等且有限」**，这是把不可分析的问题改造成可分析的问题。

**四个 Dini 导数速查表**：

| 记号 | 方向 | 上/下 | 含义 |
| --- | --- | --- | --- |
| $\underline{D}^-f$ | 左（$h<0$） | 下包络 | 左侧差商的 $\liminf$ |
| $\overline{D}^-f$ | 左（$h<0$） | 上包络 | 左侧差商的 $\limsup$ |
| $\underline{D}^+f$ | 右（$h>0$） | 下包络 | 右侧差商的 $\liminf$ |
| $\overline{D}^+f$ | 右（$h>0$） | 上包络 | 右侧差商的 $\limsup$ |

**对照记忆**：<strong>「方向（$-$/$+$）× 包络（$\underline{\phantom{D}}$/$\overline{\phantom{D}}$）」的四种组合全部存在</strong>——上标决定方向、包络记号决定上界下界，四者缺一不可地覆盖了「方向 × 包络」的乘积。

## 2 可微性的 Dini 判据

**定理（Dini 判据）**：$f$ 在 $x$ 处可微，当且仅当

$$\underline{D}^-f(x)=\overline{D}^-f(x)=\underline{D}^+f(x)=\overline{D}^+f(x)\in\mathbb{R}$$

即四个 Dini 导数都相等且为有限实数，公共值就是 $f'(x)$。

**证明**：可微 ⇔ 双侧差商极限存在 ⇔ 左极限与右极限分别存在且相等 ⇔ 各自的 $\limsup=\liminf$ 且左右相等。

**推论（单调函数的 Dini 不等式）**：$f$ 单调不减时，对几乎处处 $x$：

$$\underline{D}f(x)\le\overline{D}f(x)\ \text{且}\ \int_a^b\underline{D}f\le f(b)-f(a)\le\int_a^b\overline{D}f$$

（这个不等式组是 Lebesgue 定理证明的落点之一。）

**辨析｜易错点：Dini 导数不是「四个方向的导数」，而是「上包络与下包络 × 左右」。** 初学者常误以为 Dini 导数是「左上、右上、左下、右下」四个方向——不对。它是**同一方向（左右各一）上的上下极限**。四个值 = 2（左右）× 2（上下包络）。

## 3 Dini 导数在 Lebesgue 定理证明中的角色

Lebesgue 定理「单调函数 a.e. 可微」的证明，用 Dini 导数重写后变得清晰：

**证明骨架**（用 Vitali 覆盖，下一节细讲）：

- 记 $A=\{x:\overline{D}f(x)>\underline{D}f(x)\}$（上下 Dini 导数分离的点）。目标是 $m(A)=0$。
- 分解 $A=\bigcup_{p>q,\ p,q\in\mathbb{Q}}A_{pq}$，其中 $A_{pq}=\{x:\overline{D}f(x)>p>q>\underline{D}f(x)\}$。可数并，只需证每个 $A_{pq}$ 零测。
- 对 $A_{pq}$ 用 Vitali 覆盖引理，取一族「$f$ 增量 $\le q|I|$」的短区间覆盖 a.e. 部分；再在覆盖的区间内用「$\exists$ 更短区间使增量 $\ge p|I|$」反推。结合单调性，推出 $m(A_{pq})(p-q)\le0$——只能 $m(A_{pq})=0$。<span class="marginnote">「$p>q$ 而 $m(A_{pq})(p-q)\le0$」的矛盾来自：在 $A_{pq}$ 上，同一个 $f$ 的增量被「下 Dini ≤ q」与「上 Dini ≥ p」同时约束，而 $p>q$ 让这两个约束互斥。<strong>「上下包络分离」与「单调性」是矛盾的</strong>——这直观上就是 Lebesgue 定理的证明本质。</span>

**推论（有界变差的 Dini 分析）**：$f$ 有界变差时，由 Jordan 分解 $f=f_1-f_2$，四个 Dini 导数满足「$\underline{D}f\le f_1'-f_2'\le\overline{D}f$」型关系，同样 a.e. 可微。

## 4 公式解析：$\limsup$ 与 $\liminf$ 如何夹住「导数缺席」

把 Dini 导数与数列上下极限的对应关系写清：

$$\overline{D}f(x)=\limsup_{h\to0}\frac{f(x+h)-f(x)}{h}=\inf_{\delta>0}\sup_{0<|h|<\delta}\frac{f(x+h)-f(x)}{h}$$

- **第一步，读「$\limsup$ 的嵌套」**：$\limsup_{h\to0}Q(h)=\inf_{\delta>0}\sup_{0<|h|<\delta}Q(h)$——先对「$0<|h|<\delta$ 的邻域」取上确界，再让邻域收缩取极限。**「先局部上包络，再全局极限」**。
- **第二步，读「单调性对差商的限制」**：$f$ 单调不减时，$Q_x(h)\ge0$（$h>0$），故 $\underline{D}f\ge0$；同时 $f$ 单调给差商「上下界」：对 $h_2>h_1>0$，$Q_x(h_2)\le Q_x(h_1)\cdot$（关系），从而 $\overline{D}f$ 有限（若 $f$ 有界）。**单调性把 Dini 导数从「可能无穷」压到「有限」**——这是可微性论证的前提。
- **第三步，读「判据的实战意义」**：证明可微时，不需要算出导数，只需证「$\overline{D}\le\underline{D}$」（加上平凡的反向）。**「上下包络合拢」比「极限存在」更好验证**——因为 $\overline{D}$ 与 $\underline{D}$ 是固定的量，可以用不等式夹。

**「$\limsup/\liminf$ 嵌套 + 单调性压界 + 包络合拢」**，是 Dini 导数方法的完整三件套。

## 5 Dini 导数的计算实例

用一个具体函数把四个 Dini 导数算到底，消除抽象感。

**例**：$f(x)=|x|$ 在 $x=0$ 处。差商 $Q_0(h)=\tfrac{|h|}{h}$：$h>0$ 时 $=1$，$h<0$ 时 $=-1$。于是

$$\underline{D}^+f(0)=\overline{D}^+f(0)=1,\qquad \underline{D}^-f(0)=\overline{D}^-f(0)=-1$$

四个 Dini 导数存在但左右不等（$1\neq-1$）——**$f$ 在 $0$ 处不可微**（尖点），Dini 导数精确记录「左斜率 -1、右斜率 +1」。这里是「$h\to0^-$ 与 $h\to0^+$」的方向差别在起作用，上下包络恰好相等。

**例**：$f(x)=x\sin\tfrac1x$ 在 $x=0$（延拓 $f(0)=0$）。差商 $Q_0(h)=\sin\tfrac1h$ 在 $[-1,1]$ 间剧烈振荡，$\limsup=\liminf=1=-1$——不，$\limsup_{h\to0}\sin\tfrac1h=1$、$\liminf=-1$。故

$$\underline{D}f(0)=-1,\qquad \overline{D}f(0)=1$$

四个 Dini 导数中上下包络分离（$-1\neq1$）——**$f$ 在 $0$ 处不可微**，且 Dini 导数捕捉到「斜率在 $[-1,1]$ 间无处收敛」的事实。注意这里 $f$ 连续（$|x\sin\tfrac1x|\le|x|\to0$），**连续函数也可以有分离的 Dini 导数**——连续性不保证可微性，Dini 导数如实记录。

**例（单调函数的 Dini 不等式验证）**：$f(x)=x^2$ 在 $[0,1]$ 上单调，处处可微，$f'(x)=2x$，四个 Dini 导数处处相等 $=2x$。验证积分不等式：$\int_0^1\underline{D}f=\int_0^12x=1=f(1)-f(0)$，等号取到。<strong>光滑单调函数让不等式两边都取等</strong>；而「平台 + 陡升」的单调函数则会在 $\underline{D}$ 侧偏小、$\overline{D}$ 侧偏大的区间出现中间空隙——那里正是 Lebesgue 定理的证明要处理的点集。

**重点：两个例子分别展示 Dini 导数的两种「分离方式」**——$|x|$ 是「左右分离」（方向问题），$x\sin\tfrac1x$ 是「上下分离」（振荡问题）。可微要求两种分离都不发生。**Dini 导数把「不可微」细化为「哪一侧、哪个方向出了问题」**，这是它比「导数不存在」这一句话提供的信息多得多的地方。

## 6 小结

- **四个 Dini 导数**：左右 × 上下包络，永远存在（允许无穷）。
- **可微判据**：四者相等且有限 ⇔ 可微。
- **单调函数**：差商非负、Dini 导数有限、天然偏序。
- **Lebesgue 定理证明**：$A_{pq}$ 分解 + Vitali 覆盖 + 矛盾。
- **哲学**：Dini 导数是「导数缺席」的精确替身——极限不存在也总能测量。
- **速查**：四者 = 左右 × 上下包络；$|x|$ 是左右分离、$x\sin\tfrac1x$ 是上下分离。
- **单调性**：差商非负压界，Dini 导数有限，积分不等式 $\int\underline{D}f\le f(b)-f(a)\le\int\overline{D}f$ 成立。

在下一节，我们引入证明 Lebesgue 定理的引擎：**Vitali 覆盖引理**。

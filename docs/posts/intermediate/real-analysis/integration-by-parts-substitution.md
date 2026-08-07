---
title: 分部积分与换元积分在 L 积分中的推广
date: 2026-08-07
---

# 分部积分与换元积分在 L 积分中的推广

<div class="epigraph">
<p>分部与换元是微积分的两大引擎——在 Lebesgue 框架下，它们获得最宽松的适用条件。</p>
<footer>—— 亨利 · 勒贝格（Henri Lebesgue）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第六章 ｜ 2026-08-07</p>
</div>

## 为什么从分部积分与换元积分开始

FTC 给了「$\int F'=F$」这个框架，但实际计算还需要两个引擎：**分部积分**（把 $\int F'G$ 转为 $FG-\int FG'$）与**换元积分**（把 $\int$ 的变量替换为 $\varphi(t)$）。初等微积分中它们要求「导数连续」；Lebesgue 框架把它们推广到最自然的条件——**只需 AC**。

这两条推广是「从理论到计算」的桥梁：傅里叶分析中的分部积分（光滑化）、概率论中期望的换元（变量替换公式）、以及随机微积分中的 Itô 公式（分部积分的随机版）——全部建立在 AC 版的积分技巧上。<span class="marginnote">分部积分在<strong>Itô 积分</strong>中的对应物是 Itô 公式：$d(XY)=XdY+YdX+d\langle X,Y\rangle$。普通微积分的分部积分「$d(FG)=FdG+GdF$」对随机过程需要额外的二次变差修正。<strong>「AC 分部积分」是 Itô 公式在确定情形的极限</strong>——随机版正是这个公式的推广。</span>

## 1 分部积分

**定理（分部积分，AC 版）**：设 $F,G\in AC([a,b])$。则

$$\int_a^bF'(x)\,G(x)\,dx=F(b)G(b)-F(a)G(a)-\int_a^bF(x)\,G'(x)\,dx$$

**证明**：由乘积求导（a.e.）：$(FG)'=F'G+FG'$（$F,G$ a.e. 可微，链式在 a.e. 点成立）。$FG\in AC$（AC 函数的乘积仍 AC），由 FTC：

$$F(b)G(b)-F(a)G(a)=(FG)(b)-(FG)(a)=\int_a^b(FG)'=\int_a^bF'G+\int_a^bFG'$$

移项即得。<span class="marginnote">证明的全部要点是「<strong>$FG$ 仍是 AC</strong>」：由 $|(FG)(y)-(FG)(x)|\le|F(y)-F(x)||G|_\infty+|G(y)-G(x)||F|_\infty$ 与 AC 的线性组合，乘积保持 AC。FTC 于是对 $FG$ 生效——分部积分水到渠成。</span>

**条件说明**：只需 $F,G$ AC（不强求 $F',G'$ 连续或有界）。$F'G$ 与 $FG'$ 都在 $L^1$（由 $F',G'\in L^1$ 与 $F,G$ 有界），两个积分都有意义。

**例子**：$\int_0^1x\ln x\,dx$：取 $F=x^2/2$，$G=\ln x$（AC 于 $[0,1]$？$\ln x$ 在 $0$ 无界，需取 AC 的适当延拓或在 $(\varepsilon,1]$ 上分部再取极限——这是「广义分部」的用法）。

## 2 换元积分

**定理（换元积分，AC 版）**：设 $f\in L^1$，$\varphi\in AC([a,b])$ 且 $\varphi$ 单调（或至少 $\varphi'$ 保号 a.e.）。则

$$\int_{\varphi(a)}^{\varphi(b)}f(x)\,dx=\int_a^bf(\varphi(t))\,|\varphi'(t)|\,dt$$

（取适当符号时，$|\varphi'|$ 由 $\varphi'$ 的符号决定方向。）

**证明思路**：先对 $f=\chi_{(u,v)}$（区间指示）验证——两边都是 $|\varphi^{-1}(v)-\varphi^{-1}(u)|$ 型；再线性延拓到简单函数、单调收敛到非负 $f$，最后拆分到 $L^1$。每一步的合法性由 $\varphi$ AC（保持零测集）与 $f$ 可测性保证。<span class="marginnote">换元积分的实质是「<strong>积分对变量替换的测度变换</strong>」：$dx$ 在替换 $x=\varphi(t)$ 下变成 $|\varphi'(t)|dt$。对 AC 的 $\varphi$，$|\varphi'|$ 正是「长度伸缩率」——这是「长度元素 $dx$ 变换」的严格化，也是测度论中「推前测度」的概念。</span>

**条件说明**：$\varphi$ 只需 AC（不强求严格单调，单调即可）；$f$ 只需可积。$f(\varphi(t))$ 的可测性由 $\varphi$ AC 保证（可测函数的可测函数，$\varphi$ 不破坏可测性在 a.e. 意义下）。

## 3 与黎曼版的条件对比

| 运算 | 黎曼框架的条件 | Lebesgue（AC）框架的条件 |
| --- | --- | --- |
| 分部积分 | $F',G'$ 连续 | $F,G\in AC$ |
| 换元积分 | $\varphi'$ 连续、$f$ 连续 | $\varphi\in AC$、$f\in L^1$ |
| 核心保证 | 处处可微 + 连续 | a.e. 可微 + 无奇异增长 |

**重点：Lebesgue 版的宽大来自 AC 的「a.e. 可微 + 积分恢复」双保险。** 黎曼版要求「处处连续可微」（强且繁琐），Lebesgue 版只要「AC」（弱且本质）。**条件从「处处」放松到「几乎处处 + AC」**——这是 Lebesgue 积分对计算技巧的解放。

**辨析｜易错点：换元积分的 $\varphi$ 必须「保持零测集」（AC 的性质），否则 $f\circ\varphi$ 的可测性与积分公式可能崩溃。** 若 $\varphi$ 只是连续但非 AC（如康托尔函数），换元公式不成立——康托尔函数把零测的康托尔集映成 $[0,1]$，测度变换失真。**「AC 换元」的条件不可再弱**——这是康托尔函数又一次划界。

## 4 公式解析：分部积分证明的链式

把分部积分证明写成完整链条：

$$(FG)'=F'G+FG'\ \text{a.e.}\ \Longrightarrow\ \int_a^b(FG)'=\int_a^bF'G+\int_a^bFG'\ \Longrightarrow\ FG\big|_a^b=\int F'G+\int FG'$$

- **第一步，读「乘积求导 a.e.」**：$F,G$ a.e. 可微（Lebesgue 定理），乘积求导法则在「两者都可微」的点成立（a.e.）。**「a.e. 的链式法则」来自单调可微性的传递**。
- **第二步，读「积分线性拆分」**：$(FG)'=F'G+FG'$ 两边积分，右边拆分（线性性）为两个积分。**这里要求 $F'G,FG'$ 都可积**——由 $F,G$ 有界（AC ⇒ 有界）与 $F',G'\in L^1$ 保证。
- **第三步，读「FTC 收官」**：$\int_a^b(FG)'=(FG)(b)-(FG)(a)$——**FTC 对 $FG\in AC$ 成立**。移项得分部积分公式。**每个等号都有前面的定理背书**：Lebesgue 定理（可微）、积分线性（可积）、FTC 充要（AC）。

**「乘积求导 + 线性 + FTC」三件套**，是分部积分证明的完整结构——它依赖第七篇全部积累。

## 5 小结

- **分部积分**：$F,G\in AC$ ⇒ $\int F'G=FG\big|-\int FG'$。
- **换元积分**：$\varphi\in AC$、$f\in L^1$ ⇒ $\int f\circ\varphi\cdot|\varphi'|=\int f$。
- **条件对比**：黎曼要处处连续可微，Lebesgue 只要 AC——「a.e. + 无奇异」。
- **康托尔函数划界**：非 AC 的连续函数使换元失效。
- **应用**：傅里叶分析、概率换元、Itô 公式的确定版。

在下一节，我们介绍第七篇的收尾：**奇异函数与 Lebesgue 分解**——把 BV 函数分解为 AC 部分与奇异部分。

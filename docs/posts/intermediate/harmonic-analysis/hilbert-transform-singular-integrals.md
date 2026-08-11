---
title: Hilbert 变换与奇异积分算子
date: 2026-08-11
---

# Hilbert 变换与奇异积分算子

<div class="epigraph">
<p>Hilbert 变换是奇异积分理论的原型与全部：它的一个定理，就是一个学派的一百页。</p>
<footer>—— 依据卡尔德隆-齐格蒙德学派的工作（1948 年以降）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 调和分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Hilbert 变换开始

前四讲（第 2、3、4 篇）我们打磨了三件兵器：极大函数、Calderón–Zygmund 分解、Marcinkiewicz 插值。今天要用它们征服第一个「真正的」算子——**Hilbert 变换**。

它长得惊险：被积函数在 $y=x$ 处有极点，积分原则上发散。可一旦用「主值」把它驯服，它就摇身一变，成为整个奇异积分理论的原型：Fourier 乘子、解析函数边界值、$L^p$ 有界性、Hörmander 条件……我们今天将在一只算子上看到本专题前半程的全部思想第一次完整汇合。<span class="marginnote">Hilbert 变换在历史上接住了「调和函数的共轭」问题：给定圆盘里调和函数的边界实部，能不能恢复它的虚部？答案（所谓共轭函数 / conjugate function）正是 Hilbert 变换。1924 年 Kolmogorov 证明其弱 (1,1)，1928 年 Riesz 证明 $L^p$ 有界——今天我们用 C–Z 理论一行行重新走完这两条路。</span>

## 1 主值：给奇异积分一个名分

设 $f\in\mathcal{S}(\mathbb{R})$。**Hilbert 变换（Hilbert transform）** 定义为**主值积分（principal value）**：

$$
Hf(x)=\frac{1}{\pi}\,\mathrm{p.v.}\!\int_{\mathbb{R}}\frac{f(y)}{x-y}\,dy
=\frac{1}{\pi}\lim_{\varepsilon\to 0^{+}}\int_{|x-y|>\varepsilon}\frac{f(y)}{x-y}\,dy.
$$

**重点：** 去掉以 $x$ 为中心、半径 $\varepsilon$ 的小洞再积分，最后让洞缩掉。极限存在全靠两点：$f$ 光滑使洞附近分子可控，且核 $1/(x-y)$ 是**奇函数**（关于 $x$ 反号），对称的洞把「无穷大」的两半互相抵消。这就是「主值」的本质：**用奇性+对称性把发散裁成收敛。**

**辨析｜易错点：** 主值极限是「从两侧同时、对称地缩洞」。若只从一侧缩（比如 $\int_{x+\varepsilon}^{\infty}-\int_{-\infty}^{x-\varepsilon}$ 的顺序不对），极限可能不存在或取到不同的值。对称性不是可选项，是定义的组成部分。

再看核 $K(x)=1/\pi x$ 的性质，这是它被称为「奇异」的原因：

- **奇性**：$|K(x)|=1/(\pi|x|)$，在原点不可积，但在远处恰好可积；
- **零均值**：$\int_{\varepsilon<|x|<R}K(x)\,dx=0$——小洞里恰好「进出的量相等」。

## 2 公式解析：Fourier 乘子表示

Hilbert 变换最优雅的身份是**乘子算子**：

$$
\boxed{\;\widehat{Hf}(\xi)=-i\,\mathrm{sgn}(\xi)\,\widehat f(\xi)\;}
$$

逐项拆解：

- **第一步，左边到右边的推导**：$Hf=f*K$（卷积，主值意义），卷积的变换是乘积（第 5 讲翻译表），所以 $\widehat{Hf}=\widehat f\cdot\widehat K$。核的变换 $\widehat K(\xi)=-i\,\mathrm{sgn}(\xi)$ 是整个定理的核心计算，可通过对数奇异积分算出。
- **第二步，$-i\,\mathrm{sgn}(\xi)$ 在说什么**：每个频率 $\xi$ 被乘上纯虚数 $-i\cdot(\xi$ 的符号$)$。频率不变，**振幅不变（模长恒为 1），相位整体转 $90°$**——正频率转 $+90°$，负频率转 $-90°$，零频率归零。
- **第三步，为什么这解释了 $H$ 的一切好性质**：乘子有界（$|\cdot|\le1$）⟹ Plancherel 下 $H$ 在 $L^2$ 有界且范数 $\le1$（实际 $=1$）。乘子 $-i\,\mathrm{sgn}$ 平方是 $-1$，于是 $H^2=-I$（在 $L^2$ 上）。$H$ 是 $L^2$ 上的**等距 + 反对合（anti-involution）**，像一把把函数「旋转 $90°$」的几何刀子。<span class="marginnote">乘子观点是调和分析最锋利的翻译工具：<strong>任何在频域「逐点乘一个可测函数」的算子，叫乘子算子；而判断它是否 $L^2$ 有界，只需看乘子是否有界。</strong> 第 10 篇的 Riesz 变换与 Mikhlin 乘子定理，就是把这把刀磨到更大。</span>
- **第四步，$L^2$ 与 $L^p$ 之间那道缝**：乘子只能给 $L^2$。要 $1<p<\infty$，得走 C–Z 流水线——这正是本节下面要做的。

## 3 从 $L^2$ 到 $L^p$：Calderón–Zygmund 流水线

要把 $H$ 的有界性从 $L^2$ 推广到 $L^p$（$1<p<\infty$），教科书式路线分三步：

1. **核的尺寸条件（size condition）**：$|K(x)|\le C/|x|$。$1/\pi|x|$ 满足。
2. **核的光滑性条件（Hörmander condition）**：
$$
\int_{|x|>2|y|}\left|K(x-y)-K(x)\right|dx \le C,\qquad \forall y\ne0.
$$
   $1/\pi x$ 满足（差商估计得 $|y|/|x|^2$，积分出来有界）。这个条件让「均值零的坏块」在卷积里被吸收。
3. **Calderón–Zygmund 定理（$L^p$ 版）**：若 $T$ 是 $L^2$ 有界的奇异积分算子（核满足上两条），则 $T$ 是弱 $(1,1)$ 型，从而（Marcinkiewicz 插值）强 $(p,p)$，$1<p<\infty$。

对 $H$ 应用即得 **Riesz 定理（Riesz, 1928）**：

$$
\left\|Hf\right\|_{L^p} \le C_p\,\|f\|_{L^p}, \qquad 1<p<\infty.
$$

<span class="marginnote">回忆第 3 篇的预告：弱 (1,1) 的证明用 C–Z 分解把 $f=g+b$ 劈开，对 $g$ 用 $L^2$ 有界 + Chebyshev，对 $b$ 用核的 Hörmander 条件 + 坏块均值零，最后靠「坏立方体的扩张区域测度可控」收尾。今天它原样跑通了——这就是「通用流水线」的意义。</span>

**辨析｜易错点：** $H$ 在 $L^1$ 与 $L^\infty$ 上都**不**有界——两个端点全部失效。$H1=0$ 但不改 $L^1$ 问题（$H$ 把 $L^1$ 函数送回弱 $L^1$），而 $\|\cdot\|_\infty$ 界也被反例（对数型爆点）否决。**「$1<p<\infty$」不是记号习惯，而是定理的真实边界。**

## 4 解析函数视角：共轭与边界值

Hilbert 变换并非孤立玩具，它是复分析与前几讲之间的脐带。

设 $F=u+iv$ 在**上半平面 $\mathbb{H}$** 解析、在边界处有好的极限。则由 Cauchy–Riemann 方程，实部 $u$ 唯一决定虚部 $v$（差常数），且边界上的换算正是 Hilbert 变换：

$$
v(x)=\lim_{\varepsilon\to0}v(x+i\varepsilon)=H u(x),\qquad u(x)=\lim_{y\downarrow0}u(x+iy).
$$

也就是说：**$H$ 把调和函数在边界上的实部「翻译」成虚部**。<span class="marginnote">这个视角在 PDE 与工程里无处不在：电路里「解析信号」$z=u+iHu$（把实数信号补上 90° 相移的虚部）、地震资料分析里的 Hilbert 包络、以及信号处理里从时域振幅恢复瞬时相位，全是「共轭函数」的工程化身。</span>

由这个视角还能得到 Riesz 投影：

$$
P_{\pm}=\frac{I\pm iH}{2},
$$

它们是 $L^2(\mathbb{R})$ 上「正/负频率」的正交投影（乘子 $1_{\pm\xi>0}$ 的化身）——**Hilbert 变换把自己拆成两把「半平面滤子」**，这把拆分在 Littlewood–Paley 理论里长成了整片森林。

## 5 奇异积分的家族视图

Hilbert 变换是全家最小的成员。同一张核条件（size + Hörmander）还管着：

- **$n$ 维 Calderón–Zygmund 核**：$K(x)=a(x)/|x|^{n}$，$a$ 在球面上均值零且光滑——Riesz 变换（第 10 篇）的核就是这个家族。
- **分数次积分**：核 $1/|x|^{n-\alpha}$ 连奇性都没有（可积奇点），属另一族，见第 10 篇。
- **一般奇异积分算子**：满足 C–Z 条件且 $L^2$ 有界 ⟹ 弱 (1,1) ⟹ $L^p$，$1<p<\infty$——Hilbert 变换的证明一字不改地升级成家族公理。

**重点：** 学 Hilbert 变换的真正收益不是记住它的公式，而是记住它**示范的论证范式**——主值驯服奇性、核条件控制全局、分解+插值打穿 $L^p$。这范式将原封不动地统治第 10 篇，并最终在一般局部紧群上再次重现。

**给主值一个「可操作」的等价定义**：对 $\varepsilon>0$ 记 $H_\varepsilon f(x)=\frac1\pi\int_{|x-y|>\varepsilon}\frac{f(y)}{x-y}dy$，则 $H_\varepsilon$ 是普通（非奇异）积分、逐点有限；$Hf$ 正是 $\varepsilon\to0$ 时 $H_\varepsilon f$ 在 $L^p$（$1<p<\infty$）中的极限，由 Riesz 定理保证其存在。工程上常见的「Hilbert 滤波」就是取某个固定小 $\varepsilon$（或等效的窗函数）——主值在实践里从未真正「取到极限」，但这不妨碍理论把极限的良定义性、与 $L^p$ 有界性钉得严严实实。

**一处值得记下的数值观感**：$L^p$ 常数 $C_p$ 随 $p\to1$ 或 $p\to\infty$ 以 $\frac{1}{p-1}$、$\frac{1}{p}$ 量级发散——这正是 Marcinkiewicz 插值常数公式 $\frac{p}{p-p_1}+\frac{p}{p_2-p}$ 的数值回响。$C_p$ 在 $p=2$ 附近最小（等于 1），离端点越近越陡。调和分析里「常数炸在哪、以多快速度炸」，从来不只是技术细节：它直接决定 Littlewood–Paley 型结论与拟微分算子估计能否成立。

## 6 小结

- **主值积分**：对称缩洞 + 奇核，把发散的 $\int$ 裁成收敛；$K=1/\pi x$ 满足零均值与尺寸条件。
- **乘子公式** $\widehat{Hf}=-i\,\mathrm{sgn}(\xi)\widehat f$：$L^2$ 有界、范数 1、$H^2=-I$。
- **C–Z 流水线**：size + Hörmander + $L^2$ 有界 ⟹ 弱 (1,1) ⟹（Marcinkiewicz）$L^p$，$1<p<\infty$（Riesz 定理）。
- **端点失效**：$p=1$ 与 $p=\infty$ 都不行——边界是真实的。
- **复视角**：$H$ 是共轭函数（调和函数边界实部→虚部）；Riesz 投影 $P_\pm=(I\pm iH)/2$ 分频。

在下一节，我们把 Hilbert 变换这张单维地图铺成多维：**Riesz 变换、乘子理论与分数次积分**——那里，同一把刀切开 $\mathbb{R}^n$，并第一次遇见「非对角」的插值。

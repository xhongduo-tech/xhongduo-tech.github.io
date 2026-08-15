---
title: 函数 y = A sin(ωx + φ) 的图象
date: 2026-08-07
---

# 函数 y = A sin(ωx + φ) 的图象

<div class="epigraph">
<p>自然界这座伟大的书，是用数学的语言写成的。</p>
<footer>—— 伽利略 · 伽利莱（Galileo Galilei）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §5.6 ｜ 2026-08-07</p>
</div>

## 为什么从正弦型函数开始

纯粹的 $y=\sin x$ 是「标准件」，而自然界几乎没有标准件：弹簧振子会晃得高或矮（振幅不同）、快或慢（周期不同），摆动的起点可能歪向一侧（相位不同）。函数 $y=A\sin(\omega x+\varphi)$ 就是给标准正弦波装上三个旋钮——**振幅旋钮 $A$、频率旋钮 $\omega$、初相旋钮 $\varphi$**。学会读这三个参数，你就掌握了描述一切周期现象（声波、潮汐、电流、心跳）的统一语言。<span class="marginnote">「正弦型函数」这个词的「型」字很关键：它不是一个新的函数，而是正弦函数的放大、压缩与平移组合。把标准件改装成万用件，是数学建模中「先标准化、再参数化」的标准流程。</span> 这也是我们在「从极限到大模型」主线里第一次见到**用参数刻画一类函数**——后面指数函数、数列、乃至大模型里的注意力分数，都沿用「一组参数 + 一个母函数」的思路。

## 1 三个参数各管什么

设 $y=A\sin(\omega x+\varphi)$，其中 $A>0$，$\omega>0$，$\varphi$ 为常数。三个参数各司其职：

**$A$：振幅**。$|\sin(\omega x+\varphi)|\le 1$，所以 $|y|\le A$。$A$ 决定图象波峰波谷的「高度」，图象在直线 $y=A$ 与 $y=-A$ 之间摆动。<span class="marginnote">振幅对应物理中振动的「能量」：同一声调的琴弦，振幅越大声音越响。数学参数与物理量的这一一对应，正是 $y=A\sin(\omega x+\varphi)$ 成为物理建模标准件的原因。</span>
- **$\omega$：角频率**。它控制振动的「快慢」。函数 $f(x)=A\sin(\omega x+\varphi)$ 的最小正周期是 $T=\dfrac{2\pi}{\omega}$，每增加 $T$，相位 $\omega x+\varphi$ 恰好增加 $2\pi$，图象完整重复一次。
- **$\varphi$：初相**。当 $x=0$ 时，相位为 $\varphi$，它决定图象在「起点」处的样子——整体把 $\sin \omega x$ 的图象向左或向右平移。

**重点：把三者合起来读——$y=A\sin(\omega x+\varphi)$ 是 $\sin x$ 经过「横向伸缩（$\omega$）、纵向伸缩（$A$）、左右平移（$\varphi$）」三步改造得到的。** 这条总纲比任何单独的参数记忆都重要。

## 2 五点法作图与图象变换

### 五点法

画 $y=A\sin(\omega x+\varphi)$ 一个周期的图象，只需抓住**相位 $\omega x+\varphi$ 取五个关键值**的时刻：

$$
0,\quad \frac{\pi}{2},\quad \pi,\quad \frac{3\pi}{2},\quad 2\pi
$$

这五个时刻对应正弦函数的零点、波峰、波谷等关键点。对每个关键相位解出 $x=\dfrac{\text{相位}-\varphi}{\omega}$，得到五个点的横坐标，再配上对应纵坐标 $0,A,0,-A,0$，连线即成。<span class="marginnote">为什么选这五点？因为正弦函数在一个周期内的「形状骨架」就由这五个特殊点决定：两个端点零点、一个最高点、一个最低点、一个中间零点。中间点可以画错，这五点画对，波形大体不差。</span>

### 图象变换的两条路线

把 $y=\sin x$ 变到 $y=A\sin(\omega x+\varphi)$，可以「先平移后伸缩」或「先伸缩后平移」，两条路殊途同归：

**先平移后伸缩**：$y=\sin x \xrightarrow{\text{左移}\varphi} \sin(x+\varphi) \xrightarrow{\text{横缩}\frac{1}{\omega}} \sin(\omega x+\varphi) \xrightarrow{\text{纵伸}A} A\sin(\omega x+\varphi)$
**先伸缩后平移**：$y=\sin x \xrightarrow{\text{横缩}\frac{1}{\omega}} \sin\omega x \xrightarrow{\text{左移}\frac{\varphi}{\omega}} \sin\big(\omega(x+\frac{\varphi}{\omega})\big) \xrightarrow{\text{纵伸}A} A\sin(\omega x+\varphi)$

**辨析｜易错点：** 两条路线的平移量不同——先伸缩再平移时，平移量是 $\frac{\varphi}{\omega}$ 而不是 $\varphi$。原因在于：对 $y=\sin\omega x$ 来说，「向左平移 $\varphi$」会改变 $x$ 的相位 $\omega x \to \omega(x+\varphi)$，相位增量成了 $\omega\varphi$；要得到相位增量 $\varphi$，必须只平移 $\frac{\varphi}{\omega}$。**口诀：平移的对象永远是 $x$ 本身，横坐标的每次变换都直接作用在「括弧里的 $x$」上。**

## 3 公式解析：周期 $T=\frac{2\pi}{\omega}$

周期是正弦型函数最常用的指标，拆三步看它为什么成立：

**第一步，写出相位**：$f(x)=A\sin(\omega x+\varphi)$，相位 $p(x)=\omega x+\varphi$ 随 $x$ 线性增长。
**第二步，找重复条件**：正弦函数满足 $\sin(t+2\pi)=\sin t$。要让 $f$ 重复，只需相位增加 $2\pi$，即 $\omega(x+T)+\varphi-(\omega x+\varphi)=\omega T=2\pi$。
**第三步，解出 $T$**：$\omega T=2\pi \Rightarrow T=\frac{2\pi}{\omega}$。可见 $\omega$ 越大周期越短、振动越快——频率与周期成反比，与直觉完全吻合。

顺带可得：**频率** $f=\dfrac{1}{T}=\dfrac{\omega}{2\pi}$，表示单位时间内振动的次数。物理里交流电的「50 Hz」、声波的「440 Hz」，说的都是这个量。<span class="marginnote">注意 $\omega$ 与 $f$ 的区别：$\omega=2\pi f$，一个是「每秒走过的弧度」，一个是「每秒振动的次数」。数学教材习惯用 $\omega$，物理教材习惯用 $f$，换算系数 $2\pi$ 正是圆周角。</span>

## 4 由图象求解析式

给出一条正弦曲线的草图，要反求 $A,\omega,\varphi$，走三步：

1. **求 $A$**：量出最高点与最低点纵坐标之差的一半，即 $A=\dfrac{y_{\max}-y_{\min}}{2}$。
2. **求 $\omega$**：量出一个周期 $T$（相邻两个波峰或波谷的距离），代入 $\omega=\dfrac{2\pi}{T}$。
3. **求 $\varphi$**：取一个已知关键点（如最靠近原点的零点或波峰），把坐标代入 $y=A\sin(\omega x+\varphi)$ 解出 $\varphi$。

**重点：由图象求解析式的次序是「先 $A$，再 $\omega$，最后 $\varphi$」**——因为前两个参数只依赖图象的「几何尺寸」（高度与跨度），只有最后一个依赖「摆放位置」，逐层剥离是最稳的做法。<span class="marginnote">这个「先结构、后位置」的求解次序，和后面学数列、学二次函数待定系数法一脉相承：能先用几何特征确定的量，绝不放进方程里硬解。</span>

## 5 例题精讲：图象变换的顺序

图象变换题考「先平移还是先伸缩」。看一道题：由 $y=\sin x$ 的图象经过怎样的变换得到 $y=2\sin(3x-\frac\pi4)$？

**第一步，拆解三个参数**：振幅 $A=2$（纵伸 2 倍）、$\omega=3$（横缩 $\frac13$）、$\varphi=-\frac\pi4$。注意 $y=2\sin(3x-\frac\pi4)=2\sin\left(3(x-\frac\pi{12})\right)$——把相位写成「$3$ 乘括号」的形式。
**第二步，先伸缩后平移**：$y=\sin x \xrightarrow{\text{横缩}\frac13} \sin3x \xrightarrow{\text{右移}\frac\pi{12}} \sin\left(3(x-\frac\pi{12})\right) \xrightarrow{\text{纵伸}2} 2\sin(3x-\frac\pi4)$——右移量是 $\frac\pi{12}$，不是 $\frac\pi4$。
**第三步，先平移后伸缩**：$y=\sin x \xrightarrow{\text{右移}\frac\pi4} \sin(x-\frac\pi4) \xrightarrow{\text{横缩}\frac13} \sin(3x-\frac\pi4) \xrightarrow{\text{纵伸}2}$——右移量是 $\frac\pi4$。两条路线平移量不同，殊途同归。

<span class="marginnote">「先伸缩后平移」与「先平移后伸缩」的平移量不同：<strong>先伸缩再平移，平移量要除以 $\omega$（$\frac{\pi/4}{3}=\frac\pi{12}$）；先平移再伸缩，平移量就是 $\varphi$ 对应的 $\frac\pi4$</strong>。口诀「平移作用在 $x$ 本身」：$3(x-\frac\pi{12})$ 里 $x$ 右移 $\frac\pi{12}$。<strong>两种顺序都行，但必须记住「先伸缩后平移时平移量要按 $\omega$ 缩小」</strong>——这是图象变换最大的失分点。</span>

**辨析｜易错点：** 一是**先伸缩后平移时平移量没除 $\omega$**——把右移 $\frac\pi4$ 当答案，实际是 $\frac\pi{12}$；二是**纵伸与横缩的顺序**——纵伸不影响横坐标，可任意安排；三是**把「右移」写成「左移」**——$\sin(3x-\frac\pi4)$ 是「减」对应右移，别把符号记反。

## 6 对照表：图象变换的两条路线

把「先平移后伸缩」与「先伸缩后平移」两条路线的每一步并排列出，平移量差异一目了然（以 $y=A\sin(\omega x+\varphi)$ 为例）：

| 步骤 | 先平移后伸缩 | 先伸缩后平移 |
| --- | --- | --- |
| 第 1 步 | $\sin x \to \sin(x+\varphi)$：左移 $\varphi$ | $\sin x \to \sin\omega x$：横缩 $\frac1\omega$ |
| 第 2 步 | $\sin(x+\varphi)\to\sin(\omega x+\varphi)$：横缩 $\frac1\omega$ | $\sin\omega x\to\sin\big(\omega(x+\frac\varphi\omega)\big)$：左移 $\frac\varphi\omega$ |
| 第 3 步 | 纵伸 $A$ | 纵伸 $A$ |
| 平移量 | $\varphi$ | $\frac\varphi\omega$ |

**重点：两条路线平移量不同——先伸缩后平移时，平移量是 $\frac\varphi\omega$ 而不是 $\varphi$。** 因为伸缩已经改变了 $x$ 的尺度，平移必须以「缩后的尺度」来度量。

<span class="marginnote"><strong>记忆锚：平移永远是「作用在 $x$ 本身」上的操作</strong>。把目标式写成 $A\sin(\omega(x+\frac\varphi\omega))$ 的形式，$x$ 右边加了多少，就是右移多少。写不出这个「标准形式」时先别平移——先配方。这个「配方看平移」的习惯，能消灭九成的图象变换错误。</span>

**辨析｜易错点：** 一是**平移量与 $\omega$ 的关系**——先伸缩后平移的平移量必须除以 $\omega$；二是**符号方向**——$x-\frac\pi{12}$ 是右移、$x+\frac\pi{12}$ 是左移，别记反；三是**先平移后伸缩时 $\varphi$ 的直接性**——此时 $\varphi$ 就是相位里的平移量，直接使用。

## 7 术语速查表

| 术语 | 记号 / 公式 | 一句话释义 |
| --- | --- | --- |
| 振幅 | $A$ | 波峰波谷的高度 |
| 角频率 | $\omega$ | 控制振动快慢 |
| 周期 | $T=\frac{2\pi}{\omega}$ | 相位走完 $2\pi$ 所需 |
| 频率 | $f=\frac{1}{T}=\frac{\omega}{2\pi}$ | 单位时间振动次数 |
| 初相 | $\varphi$ | $x=0$ 时的相位 |
| 相位 | $\omega x+\varphi$ | 正弦的自变量 |
| 五点法 | 相位取 $0,\frac\pi2,\pi,\frac{3\pi}2,2\pi$ | 画一个周期的骨架 |
| 图象变换 | 平移 / 伸缩 | 参数改动的图象效果 |

**记忆线索**：三个参数三条线——$A$ 管高、$\omega$ 管快慢（$T=\frac{2\pi}{\omega}$）、$\varphi$ 管左右；「平移作用在 $x$ 本身」是变换的唯一口诀。由图象求解析式按「先 $A$、再 $\omega$、后 $\varphi$」的顺序剥离。

## 8 小结

- $y=A\sin(\omega x+\varphi)$ 的三个参数：**$A$ 管振幅、$\omega$ 管周期（$T=\frac{2\pi}{\omega}$）、$\varphi$ 管初相（平移）**。
- 图象是 $\sin x$ 经过横向伸缩、纵向伸缩、左右平移得到的；**平移与伸缩谁先谁后，平移量不同**，口诀是「平移作用在括弧里的 $x$ 上」。
- 五点法抓住相位取 $0,\frac{\pi}{2},\pi,\frac{3\pi}{2},2\pi$ 的五个时刻。
- 由图象求解析式按「$A \to \omega \to \varphi$」的顺序逐步确定。

在下一节，我们将把正弦型函数放到真实世界里：从简谐运动、交变电流到潮汐涨落，用 $y=A\sin(\omega x+\varphi)$ 建立模型、预测变化，这就是**三角函数的应用**。

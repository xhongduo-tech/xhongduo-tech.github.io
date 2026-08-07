---
title: 留数定理在定积分计算上的应用：∫R(x)e^{iax}dx 型与若尔当引理
date: 2026-08-08
---

# 留数定理在定积分计算上的应用：∫R(x)e^{iax}dx 型与若尔当引理

<div class="epigraph">
<p>振荡因子 $e^{iax}$ 改变了圆弧积分的命运——若尔当引理说：只需分母比分子高一次，圆弧就归零。</p>
<footer>—— 傅里叶型积分与若尔当引理</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》§5.9 ｜ 2026-08-08</p>
</div>

## 为什么 $e^{iax}$ 让事情变好

上一节要求「分母比分子高两次」。但当被积函数变成 $R(x)e^{iax}$（$a>0$）——傅里叶变换的常客——圆弧上的积分反而更容易归零：**$e^{iaz}$ 在上半平面指数式衰减**（因为 $z=x+iy$，$|e^{iaz}|=e^{-ay}$），它把圆弧上的被积函数「压」下去。若尔当引理给出了精确版本：只要分母比分子**高一次**，上半圆弧积分就趋于零。这条引理让一大类傅里叶型积分 $\int_{-\infty}^{\infty}R(x)e^{iax}dx$ 迎刃而解，也是第七章傅里叶变换的复变前奏。<span class="marginnote">$|e^{iaz}|=e^{-ay}$（$a>0,y>0$）是这套方法的命脉：$z$ 在上半平面越高（$y$ 越大），$e^{iaz}$ 越小。所以「上半平面 + $a>0$」是黄金组合；若 $a<0$，指数在上半平面反而暴涨，必须改用下半平面围道。</span>

## 1 若尔当引理

**核心概念：若尔当引理（Jordan's lemma）**：设 $R(z)$ 在 $|z|\ge R_0$、$\mathrm{Im}\,z\ge 0$ 上连续，且 $\lim_{z\to\infty}R(z)=0$ 在上半平面一致成立。则对 $a>0$，

$$\lim_{R\to\infty}\int_{C_R} R(z)e^{iaz}\,dz = 0$$

其中 $C_R$ 是上半圆弧 $|z|=R$，$\mathrm{Im}\,z\ge 0$。

**注意条件的「放宽」：** 这里只要求 $R(z)\to0$（即分母比分子**高至少一次**），而不是上一节的高两次。$e^{iaz}$ 的指数衰减「帮」圆弧积分归零。

**证明思想（估值不等式 + 正弦估计）：** 在 $C_R$ 上，$|R(z)|\le\varepsilon_R$（$\varepsilon_R\to0$），$|e^{iaz}|=e^{-aR\sin\theta}$。圆弧积分模：

$$\left|\int_{C_R}R(z)e^{iaz}dz\right|\le\varepsilon_R\int_0^{\pi} e^{-aR\sin\theta}R\,d\theta = 2\varepsilon_R R\int_0^{\pi/2}e^{-aR\sin\theta}d\theta$$

利用 $\sin\theta\ge\frac{2\theta}{\pi}$（对 $\theta\in[0,\pi/2]$）得 $\int_0^{\pi/2}e^{-aR\sin\theta}d\theta\le\int_0^{\pi/2}e^{-2aR\theta/\pi}d\theta=\frac{\pi}{2aR}(1-e^{-aR})$。代入：

$$\le 2\varepsilon_R R\cdot\frac{\pi}{2aR}=\frac{\pi\varepsilon_R}{a}\to 0$$

**关键杠杆：$\sin\theta\ge\frac{2\theta}{\pi}$ 这条「直线下界」把指数积分压成可估的 $\frac1R$ 项，抵消圆弧长 $R$。**<span class="marginnote">「$\sin\theta\ge2\theta/\pi$」是若尔当引理证明里最巧妙的一步——弦长估计把三角积分变成简单指数积分。这个技巧在实分析估计里也很常见（如证明傅里叶变换的衰减性质），值得记下。</span>

## 2 傅里叶型积分的计算公式

**核心公式：** 设 $R$ 是有理函数，$Q$ 无实零点，$\deg Q\ge\deg P+1$，$a>0$，则

$$\int_{-\infty}^{\infty}R(x)e^{iax}\,dx = 2\pi i\sum_{\mathrm{Im}\,z_k>0}\mathrm{Res}\left(R(z)e^{iaz}, z_k\right)$$

**取实部虚部可得：**

$$\int_{-\infty}^{\infty}R(x)\cos(ax)\,dx = \mathrm{Re}\left[2\pi i\sum\mathrm{Res}\right], \qquad \int_{-\infty}^{\infty}R(x)\sin(ax)\,dx = \mathrm{Im}\left[2\pi i\sum\mathrm{Res}\right]$$

**围道与上一节相同**（实轴线段 + 上半圆弧），区别只在圆弧归零用若尔当引理（高一次即可）而非衰减条件（高两次）。

**例：** 求 $\int_{-\infty}^{\infty}\frac{\cos x}{x^2+1}\,dx$。

- 把 $\cos x$ 写成 $\mathrm{Re}(e^{ix})$，考虑 $\int_{-\infty}^{\infty}\frac{e^{ix}}{x^2+1}dx$。
- $R(z)=\frac1{z^2+1}$，$\deg Q=2\ge0+1$ ✓，$a=1>0$。
- 上半平面奇点 $z=i$（简单极点），留数：
$$\mathrm{Res}\left(\frac{e^{iz}}{z^2+1},i\right)=\frac{e^{iz}}{2z}\Big|_{z=i}=\frac{e^{-1}}{2i}=-\frac{i}{2e}$$
- $\int_{-\infty}^{\infty}\frac{e^{ix}}{x^2+1}dx=2\pi i\cdot(-\frac{i}{2e})=\frac{\pi}{e}$。
- 取实部（注意 $\frac{\pi}{e}$ 已是实数，虚部为零）：$\int_{-\infty}^{\infty}\frac{\cos x}{x^2+1}dx=\frac{\pi}{e}$，$\int_{-\infty}^{\infty}\frac{\sin x}{x^2+1}dx=0$（被积函数为奇函数）。<span class="marginnote">经典结果 $\int_{-\infty}^{\infty}\frac{\cos x}{x^2+1}dx=\frac{\pi}{e}\approx1.156$。这类「余弦 + 有理函数」的积分在信号处理、概率论（特征函数）里频繁出现——第七章傅里叶变换的谱密度正是这类积分的舞台。</span>

## 3 公式解析：$\cos ax$ 型积分为什么「取实部」

很多学生困惑：为什么算 $\cos ax$ 积分要先用 $e^{iax}$，最后取实部？拆开看：

$$e^{iax} = \cos ax + i\sin ax$$

- **第一步，把实积分嵌入复积分。** $\int R(x)\cos(ax)dx$ 不是复积分，但 $\int R(x)e^{iax}dx=\int R(x)\cos(ax)dx+i\int R(x)\sin(ax)dx$，一个复积分同时装了余弦（实部）与正弦（虚部）。
- **第二步，算复积分。** 复积分可以用若尔当引理 + 留数定理，得到 $2\pi i\sum\mathrm{Res}$。
- **第三步，取实部。** 结果的实部就是 $\cos ax$ 积分，虚部就是 $\sin ax$ 积分。**一石二鸟：一个复积分给出两个实积分。**

**辨析｜易错点：取实部要在「最终结果」取，不是中途。** 留数 $\mathrm{Res}(R e^{iaz}, z_k)$ 本身是复数，取实部前必须先把整个 $2\pi i\sum\mathrm{Res}$ 算完。**中途取实部会丢信息，导致错误。** 也别忘了检查最终结果是否为实数（虚部应为 $\sin$ 奇函数的零贡献，可作验算）。<span class="marginnote">「嵌入复积分 + 取实虚部」是工程数学的标准套路：一个复量同时编码两个实量。第七章傅里叶变换的频谱、谱密度、卷积定理都建立在这个「复量编码实信息」的思想上。</span>

## 4 例二：$\int_{-\infty}^{\infty}\frac{\sin x}{x}\,dx=\pi$

这是傅里叶分析里最著名的积分（Dirichlet 积分），若尔当引理方法的看家案例。

- $\frac{\sin x}{x}$ 在实轴上处处可去（$x=0$ 处补定义为 $1$），全实轴可积（广义积分）。
- 考虑 $F(z)=\frac{e^{iz}}{z}$。它在 $z=0$ 有极点——**恰在实轴上**，围道不能穿过。处理：取小半圆绕 $z=0$（上或下），再用若尔当引理。

**结果：** $\int_{-\infty}^{\infty}\frac{\sin x}{x}\,dx=\pi$。**推导要点：** 取围道 = 实轴（绕过原点的小半圆）+ 上半圆弧。上半平面无奇点（$z=0$ 被绕过），留数贡献来自小半圆。小半圆（半径 $\varepsilon\to0$）上的积分恰好贡献 $\pi$（留数的一半）。

**重点：这个积分不能用「高一次 + 直接公式」——因为 $z=0$ 在实轴上，围道必须变形。** 它展示了「实轴奇点」场景的标准处理：小半圆绕行，贡献主值。<span class="marginnote">$\int_{-\infty}^{\infty}\frac{\sin x}{x}dx=\pi$ 在信号处理里意义非凡：它说明理想低通滤波器的冲激响应积分有限，也是香农采样定理的数学基石之一。第八章拉普拉斯变换与第七章傅里叶变换都会反复引用这个结果。</span>

**辨析｜易错点：$z=0$ 恰在实轴上的极点，围道不能穿过。** 处理原则：用小半圆绕过（半径 $\varepsilon\to0$），小半圆上的积分贡献「留数的一半」（方向相关），最终结果是主值积分。**「实轴极点绕行」是傅里叶型积分最精细的技术细节，先识别「奇点是否踩线」，再决定绕行方向。**

## 5 补充：傅里叶型积分的常见结果与验算

若尔当引理方法在工程里高频出现，把常用结果与验算手段整理成「查表 + 核对」的工作流。

**常用变换结果：**

| 积分 | 结果 | 条件 |
| --- | --- | --- |
| $\int_{-\infty}^{\infty}\frac{\cos ax}{x^2+b^2}dx$ | $\frac{\pi}{b}e^{-ab}$ | $a>0,b>0$ |
| $\int_{-\infty}^{\infty}\frac{\sin ax}{x}dx$ | $\pi$ | Dirichlet 积分 |
| $\int_{-\infty}^{\infty}\frac{\cos ax}{(x^2+b^2)^2}dx$ | $\frac{\pi}{2b^3}(1+ab)e^{-ab}$ | 二阶极点 |
| $\int_{-\infty}^{\infty}\frac{x\sin ax}{x^2+b^2}dx$ | $\pi e^{-ab}$ | 分子含 $x$ |

**例（查表应用）：** $\int_{-\infty}^{\infty}\frac{\cos 2x}{x^2+1}dx$：$a=2,b=1$，结果 $\frac\pi1 e^{-2}=\pi e^{-2}$。**验算**：数值积分 $0.425$，$\pi e^{-2}\approx0.425$ ✓。

**验算手段一：极限一致性。** $a\to0^+$ 时 $\int\frac{\cos ax}{x^2+b^2}dx\to\int\frac{dx}{x^2+b^2}=\frac\pi b$，与公式 $a=0$ 时 $\frac\pi b e^0=\frac\pi b$ 吻合——**$a\to0$ 时余弦退化常数，结果应退化到纯有理积分**。

**验算手段二：奇偶性。** $\int\frac{\sin ax}{x}dx$ 的被积函数为偶函数（$\frac{\sin ax}{x}$ 偶），结果非零合理；若被积函数为奇函数，结果应为零。

**综合例：** 计算 $\int_{-\infty}^{\infty}\frac{\cos x}{(x^2+1)(x^2+4)}dx$。$F(z)=\frac{e^{iz}}{(z^2+1)(z^2+4)}$，上半平面极点 $z=i$（一阶）、$z=2i$（一阶）。留数：

$$\mathrm{Res}(i)=\frac{e^{-1}}{(2i)(3)}=\frac{1}{6ie},\qquad \mathrm{Res}(2i)=\frac{e^{-2}}{(4i)(-3)}=-\frac{1}{12ie^2}$$

积分 $=2\pi i\left(\frac1{6ie}-\frac1{12ie^2}\right)=2\pi\left(\frac1{6e}-\frac1{12e^2}\right)=\frac{\pi}{3e}-\frac{\pi}{6e^2}$，取实部（结果已是实数）。

**重点：含多个极点的傅里叶型积分，逐极点算留数再求和**——与纯有理积分流程一致，只是被积函数多了 $e^{iaz}$。

**辨析｜易错点：$a<0$ 时不能沿用上半平面围道。** $e^{iaz}$ 在 $a<0$ 时上半平面增长、下半平面衰减，须改取下半平面围道（方向注意）。**先看 $a$ 的符号再选半平面。**

## 6 小结

- **若尔当引理**：$R(z)\to0$（高一次）+ $a>0$ ⟹ 上半圆弧积分归零。
- **核心公式**：$\int_{-\infty}^{\infty}R(x)e^{iax}dx=2\pi i\sum_{\mathrm{Im}z_k>0}\mathrm{Res}(R(z)e^{iaz},z_k)$。
- **余弦/正弦**：取复积分的实部/虚部，一个复积分给两个实积分。
- **经典结果**：$\int_{-\infty}^{\infty}\frac{\cos x}{x^2+1}dx=\frac{\pi}{e}$；$\int_{-\infty}^{\infty}\frac{\sin x}{x}dx=\pi$（Dirichlet 积分）。
- **实轴奇点**：用小半圆绕行，贡献主值；先判「奇点是否踩线」。

在下一节，第五章以两个深刻定理收尾：**对数留数与辐角原理、儒歇定理**。对数留数把「零点与极点」计数变成积分，辐角原理用辐角变化数零点，儒歇定理则成为判断多项式根的分布、乃至代数基本定理的利器。

---
title: 留数定理在定积分计算上的应用：∫R(sinθ, cosθ)dθ 型
date: 2026-08-08
---

# 留数定理在定积分计算上的应用：∫R(sinθ, cosθ)dθ 型

<div class="epigraph">
<p>把三角函数积分塞进单位圆，让留数定理去数奇点——实积分从此有了复平面的捷径。</p>
<footer>—— 留数法积分观</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》§5.7 ｜ 2026-08-08</p>
</div>

## 为什么三角积分难，而复积分易

形如 $\int_0^{2\pi}R(\sin\theta,\cos\theta)\,d\theta$（$R$ 为有理函数）的积分，在实分析里通常要动用万能公式 $t=\tan\frac\theta2$，结果往往是一坨繁琐的有理积分。复分析给出了一条截然不同的路：**令 $z=e^{i\theta}$，三角函数就变成 $z$ 的有理函数，积分区间 $[0,2\pi]$ 变成单位圆，积分自动成为围道积分。** 于是实积分 ⟶ 留数定理，只需要数单位圆内的奇点。本节是留数法三件套的第一件，也是「复变算实积分」的开胃菜。<span class="marginnote">核心代换 $z=e^{i\theta}$ 的几何意义：$z$ 沿单位圆走一圈，正好扫过 $\theta\in[0,2\pi]$。$\sin\theta$、$\cos\theta$ 用欧拉公式写成 $z$ 与 $1/z$，$d\theta$ 换成 $\frac{dz}{iz}$——整个积分被「翻译」成复平面语言。</span>

## 1 代换的标准化

**核心代换：** 令 $z=e^{i\theta}$（$\theta\in[0,2\pi]$），则单位圆 $|z|=1$ 走一圈。由欧拉公式：

$$\cos\theta = \frac{e^{i\theta}+e^{-i\theta}}{2} = \frac{z+z^{-1}}{2}, \qquad \sin\theta = \frac{e^{i\theta}-e^{-i\theta}}{2i} = \frac{z-z^{-1}}{2i}$$

又由 $z=e^{i\theta}$ 微分得 $dz=ie^{i\theta}d\theta=iz\,d\theta$，故

$$d\theta = \frac{dz}{iz}$$

代入原积分：

$$\int_0^{2\pi}R(\sin\theta,\cos\theta)\,d\theta = \oint_{|z|=1} R\left(\frac{z-z^{-1}}{2i},\frac{z+z^{-1}}{2}\right)\frac{dz}{iz}$$

**被积函数成为 $z$ 的有理函数**（分子分母都是多项式），于是留数定理直接适用：

$$\int_0^{2\pi}R(\sin\theta,\cos\theta)\,d\theta = 2\pi i\sum_{|z_k|<1}\mathrm{Res}(F(z), z_k)$$

其中 $F(z)$ 是代换后的被积函数，求和取单位圆内的奇点。<span class="marginnote">这个代换一举两得的本质：三角函数的有理函数 $R(\sin\theta,\cos\theta)$ 变成 $z$ 的有理函数，而「沿 $[0,2\pi]$ 积分」恰好对应「绕单位圆一圈」。实积分的困难（三角函数的有理化）被复积分的成熟工具（留数）取代。</span>

## 2 例一：$\int_0^{2\pi}\frac{d\theta}{a+\cos\theta}$（$a>1$）

用代换 $z=e^{i\theta}$，$\cos\theta=\frac{z+z^{-1}}2$，$d\theta=\frac{dz}{iz}$：

$$I=\oint_{|z|=1}\frac{1}{a+\frac{z+z^{-1}}2}\cdot\frac{dz}{iz}=\oint_{|z|=1}\frac{2}{2az+z^2+1}\cdot\frac{dz}{iz}=\frac{2}{i}\oint_{|z|=1}\frac{dz}{z^2+2az+1}$$

分母 $z^2+2az+1$ 的根：$z=-a\pm\sqrt{a^2-1}$。两个根都在实轴上，一个在单位圆内、一个在圆外：

- $z_1=-a+\sqrt{a^2-1}$：由于 $a>1$，$0<|z_1|<1$，**在圆内**。
- $z_2=-a-\sqrt{a^2-1}$：$|z_2|>1$，在圆外。

单位圆内只有一个**简单极点** $z_1$。其留数：

$$\mathrm{Res}\left(\frac{1}{z^2+2az+1},z_1\right)=\frac{1}{2z_1+2a}=\frac{1}{2\sqrt{a^2-1}}$$

于是

$$I=\frac{2}{i}\cdot 2\pi i\cdot\frac{1}{2\sqrt{a^2-1}}=\frac{2\pi}{\sqrt{a^2-1}}$$

**答案：$\int_0^{2\pi}\frac{d\theta}{a+\cos\theta}=\frac{2\pi}{\sqrt{a^2-1}}$（$a>1$）。** 这是经典结果，留数法三行得出；实分析里用万能公式要折腾一大页。<span class="marginnote">验证特例：$a=2$ 时积分 $=\frac{2\pi}{\sqrt3}\approx3.63$。数值积分 $\int_0^{2\pi}\frac{d\theta}{2+\cos\theta}$ 的近似值与之一致。用具体数值验证公式是「留数法结果可信」的最快检查。</span>

## 3 公式解析：$\int_0^{2\pi}R(\sin\theta,\cos\theta)d\theta$ 的完整流程

把方法拆成四步，建立可复用的流程：

- **第一步，代换。** $z=e^{i\theta}$，$\sin\theta=\frac{z-z^{-1}}{2i}$，$\cos\theta=\frac{z+z^{-1}}2$，$d\theta=\frac{dz}{iz}$。代入后化成 $\oint_{|z|=1}F(z)\,dz$，$F$ 为有理函数。
- **第二步，找单位圆内的奇点。** 解 $F$ 的分母零点，逐个判定 $|z_k|<1$ 与否。**注意分母可能有重因子（高阶极点），按阶数处理。**
- **第三步，算留数。** 每个圆内奇点用上一节的规则表算 $\mathrm{Res}(F,z_k)$。
- **第四步，求和乘 $2\pi i$。** 结果 $=2\pi i\sum_{|z_k|<1}\mathrm{Res}(F,z_k)$。

**辨析｜易错点：代换后分母出现 $z$ 的高次，别漏根。** $z^2+2az+1=0$ 有两根，其中 $z_1 z_2=1$（常数项为 $1$）——两根互为倒数，必一内一外。**这类「常数项为 1 的二次方程」的两根必一个在圆内一个在圆外**，是高频考点，直接断言即可。**另外注意 $z=0$ 也可能成为 $F$ 的奇点**（代换引入 $z^{-1}$ 项时），不要漏掉。

## 4 例二：$\int_0^{2\pi}\frac{d\theta}{1+\sin\theta\cos\theta}$

代换：$\sin\theta\cos\theta=\frac{z-z^{-1}}{2i}\cdot\frac{z+z^{-1}}2=\frac{z^2-z^{-2}}{4i}$。代入：

$$I=\oint_{|z|=1}\frac{1}{1+\frac{z^2-z^{-2}}{4i}}\cdot\frac{dz}{iz}=\oint_{|z|=1}\frac{4i z^2}{z^4+4iz^2-1}\cdot\frac{dz}{iz}=\oint_{|z|=1}\frac{4z^2}{z^4+4iz^2-1}dz$$

分母 $z^4+4iz^2-1$ 是 $z^2$ 的二次式。令 $w=z^2$，$w^2+4iw-1=0$，$w=-2i\pm\sqrt{-4+1}=-2i\pm i\sqrt3$，即 $w=i(\pm\sqrt3-2)$。两根中 $\sqrt3-2\approx-0.268$，故 $w_1=i(\sqrt3-2)$ 满足 $|w_1|=\sqrt3-2<1$；$w_2=i(-\sqrt3-2)$ 满足 $|w_2|=\sqrt3+2>1$。对应 $z$：$z^2=w_1$ 给出**两个**圆内根 $z=\pm\sqrt{w_1}$（各为简单极点）。

**重点：$z^2=w_1$ 的两个根都在单位圆内**，因为 $|z|=\sqrt{|w_1|}=\sqrt{\sqrt3-2}<1$。算两个根的留数（各为 $\frac{4z^2}{4z^3+8iz}$ 在根处的值），对称性使两者相等。完整计算得

$$I = 2\pi i\cdot 2\cdot \mathrm{Res} = \frac{2\pi}{\sqrt{1-\frac14}}\cdot(\cdots)$$

（精确结果略繁，重点是流程：**换元 $w=z^2$ 判根、注意一对根、逐点算留数。**）<span class="marginnote">例二展示了「换元判根」技巧：高次分母先令 $w=z^k$ 化成二次式，判出 $w$ 的内外，再还原 $k$ 个根。$z^k=w$ 的 $k$ 个根要么全在圆内（$|w|<1$）要么全在圆外——这大大简化了「哪些奇点在圈内」的判定。</span>

## 5 三角积分的三种变形

**变形一：被积函数只含 $\cos\theta$ 的偶函数。** 积分区间可折半：$\int_0^{2\pi}$ 换成 $2\int_0^\pi$ 或 $4\int_0^{\pi/2}$，有时能进一步简化。

**变形二：分母含 $\sin^2\theta,\cos^2\theta$。** 用 $\sin^2\theta=\frac{1-\cos2\theta}2$ 降次，再令 $z=e^{i\theta}$ 或直接 $z=e^{i2\theta}$。

**变形三：$R$ 在单位圆上有极点。** 若代换后 $F$ 的奇点恰好落在 $|z|=1$ 上，留数定理需谨慎——通常取柯西主值或改用小圆避让。**考试里先检查奇点是否「踩线」。**<span class="marginnote">「踩线奇点」是最容易忽略的失败模式。判断方法：解出分母零点后，不仅要看是否 $|z|<1$，还要确认是否 $|z|=1$。若恰在单位圆上，围道必须绕过（通常取半径 $1\pm\varepsilon$ 再取极限），公式的「圈内」集合随之改变。</span>

## 6 补充：三角积分留数法的系统练习

把「$\int_0^{2\pi}R(\sin\theta,\cos\theta)d\theta$」的完整流程用三道难度递进的题练透。

**例 1（基础）：** $\int_0^{2\pi}\frac{d\theta}{5-4\cos\theta}$。代换 $z=e^{i\theta}$，$\cos\theta=\frac{z+z^{-1}}2$：

$$I=\oint_{|z|=1}\frac{1}{5-2(z+z^{-1})}\cdot\frac{dz}{iz}=\oint_{|z|=1}\frac{1}{-2z^2+5z-2}\cdot\frac{dz}{i}=\frac1i\oint_{|z|=1}\frac{dz}{(2z-1)(z-2)}$$

极点 $z=\frac12$（圆内）、$z=2$（圆外）。留数：$\mathrm{Res}\left(\frac1{(2z-1)(z-2)},\frac12\right)=\frac1{2(\frac12-2)}=-\frac13$。$I=\frac1i\cdot2\pi i\cdot(-\frac13)=\frac{2\pi}3$。

**例 2（含 $\sin$）：** $\int_0^{2\pi}\frac{d\theta}{2+\sin\theta}$。代换后分母 $2+\frac{z-z^{-1}}{2i}$，化简为二次方程 $z^2+4iz-1=0$，根 $z=i(-2\pm\sqrt3)$。圆内根 $z=i(\sqrt3-2)$（模 $<1$）。算留数求和即得 $\frac{2\pi}{\sqrt3}$。**含 $\sin$ 时分母出现 $i$，判根要小心模长。**

**例 3（分母高次）：** $\int_0^{2\pi}\frac{d\theta}{(a+\cos\theta)^2}$。先算 $J(a)=\int\frac{d\theta}{a+\cos\theta}=\frac{2\pi}{\sqrt{a^2-1}}$（$a>1$），对参数 $a$ 求导：

$$\frac{dJ}{da}=\int_0^{2\pi}\frac{-\cos\theta\,d\theta}{(a+\cos\theta)^2}\ne I$$

**参数求导不一定直接给出目标——此法要另寻路。** 直接代换：$I=\oint\frac{4z}{(z^2+2az+1)^2}\cdot\frac{dz}{iz}$，圆内二阶极点（$z^2+2az+1$ 的一内根），用二阶留数公式计算。**「参数求导」只对特定形式的积分有效，别硬套。**

**重点：流程固化——代换、判根内外、算留数、乘 $2\pi i$。** 三类题（纯 $\cos$、含 $\sin$、高次分母）都是这四步，差异只在「判根」与「留数阶数」。

**辨析｜易错点：代换后 $z=0$ 可能是奇点。** 分母含 $z^{-1}$ 时，$z=0$ 处被积函数可能有极点——**别漏掉 $z=0$ 的留数。** 例 1 的分母 $-2z^2+5z-2$ 在 $z=0$ 处取值 $-2\ne0$，无奇点；但分母含 $z^{-2}$ 的题目要查 $z=0$。

## 7 小结

- **核心代换**：$z=e^{i\theta}$，$\sin\theta=\frac{z-z^{-1}}{2i}$，$\cos\theta=\frac{z+z^{-1}}2$，$d\theta=\frac{dz}{iz}$。
- **流程**：代换 ⟶ 找圆内奇点 ⟶ 算留数 ⟶ 乘 $2\pi i$。
- **经典结果**：$\int_0^{2\pi}\frac{d\theta}{a+\cos\theta}=\frac{2\pi}{\sqrt{a^2-1}}$（$a>1$）。
- **判根技巧**：常数项为 $1$ 的二次方程两根一内一外；$z^k=w$ 的根同进同出。
- **易错**：漏 $z=0$ 奇点、奇点踩线（$|z|=1$）、高阶极点按阶算。

在下一节，留数法进入无穷区间的战场：**$\int_{-\infty}^{\infty}R(x)\,dx$ 型无穷积分**。取上半平面的大半圆围道，让圆弧上的积分随半径趋于零——有理函数在实轴上的无穷积分，就这样被「压缩」进上半平面的几个留数。

---
title: 二重积分的变量变换：极坐标与一般变换
date: 2026-08-07
---

# 二重积分的变量变换：极坐标与一般变换

<div class="epigraph">
<p>换元在二重积分里不只是「换变量」，还要「换面积微元」——$dx\,dy$ 变成 $|J|\,du\,dv$，雅可比行列式决定局部伸缩。</p>
<footer>—— 卡尔·雅可比（Carl Gustav Jacob Jacobi），变量变换理论（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§21.5 ｜ 2026-08-07</p>
</div>

## 为什么换元要「多乘一个雅可比」

一元换元 $\int f(x)dx=\int f(\varphi(t))\varphi'(t)dt$ 有「换元因子 $\varphi'$」。二重积分换元同样要换「面积微元」：$dx\,dy$ 变换成 $|J|\,du\,dv$，其中 $J$ 是雅可比行列式。**「换元因子」在二维就是「雅可比行列式的绝对值」**——它度量「旧坐标面积与新坐标面积的比例」。

最常见的换元是**极坐标**：圆域、扇形、含 $x^2+y^2$ 的被积函数，在极坐标下上下限与被积函数都大幅简化。**「极坐标的雅可比 $=r$」是本节最常用的公式**——$dx\,dy=r\,dr\,d\theta$。它也是高斯积分 $\int e^{-x^2}dx=\sqrt\pi$ 的钥匙。<span class="marginnote">「$dx\,dy=r\,dr\,d\theta$」的直觉：极坐标的「小格子」不是矩形而是「小扇形」——半径 $r$、角度宽 $d\theta$、径向宽 $dr$，面积约 $r\,dr\,d\theta$（「半径 × 弧长」）。离原点越远（$r$ 越大），同样 $dr,d\theta$ 扫过的面积越大——<strong>$r$ 是「远离原点放大的因子」</strong>。这个「网格不均匀」的直觉，正是雅可比行列式「局部伸缩」的几何意义。</span>

## 1 二重积分的变量变换公式

**定理（变量变换）：设变换 $x=x(u,v),\ y=y(u,v)$ 把 $(u,v)$ 平面区域 $D'$ 一一映射到 $(x,y)$ 平面区域 $D$，$x,y$ 有连续偏导数，雅可比行列式 $\frac{\partial(x,y)}{\partial(u,v)}\ne0$ 在 $D'$ 内，$f$ 在 $D$ 上连续，则**

$$\iint_D f(x,y)\,dx\,dy=\iint_{D'}f(x(u,v),y(u,v))\,\left|\frac{\partial(x,y)}{\partial(u,v)}\right|\,du\,dv.$$

**公式解析：三步拆解**

**第一步，换点**。$f(x,y)$ 换成 $f(x(u,v),y(u,v))$——新坐标代入；

**第二步，换微元**。$dx\,dy=\left|\frac{\partial(x,y)}{\partial(u,v)}\right|du\,dv$——**面积微元乘雅可比行列式的绝对值**（伸缩因子）；

**第三步，换区域**。$D$ 换成 $D'$（变换前的区域）——积分在参数域进行。

**要点**：**换元公式 = 「$f$ 代入 + $d\sigma$ 乘 $|J|$ + 区域换 $D'$」**。雅可比行列式的绝对值 $|J|$ 是「新旧面积的比例」，必须加绝对值（面积非负）。

> **辨析｜易错点：**二重积分换元与一元的关键区别在**$|J|$ 因子**——一元换元是 $\varphi'(t)$（可正可负，不需绝对值），二维必须取 $|J|$。另一个易错点：**$J$ 的分子分母顺序**——$\frac{\partial(x,y)}{\partial(u,v)}$（新坐标对旧坐标），与 $\frac{\partial(u,v)}{\partial(x,y)}$ 互为倒数（§18.2 反函数组）。写反会导致因子弄错。还有：**换元必须「一一映射」**（或分段一一），否则多重覆盖会多算。

## 2 极坐标变换

**极坐标**：$x=r\cos\theta,\ y=r\sin\theta$。雅可比

$$\frac{\partial(x,y)}{\partial(r,\theta)}=\begin{vmatrix}\cos\theta&-r\sin\theta\\\sin\theta&r\cos\theta\end{vmatrix}=r.$$

**极坐标二重积分**：

$$\iint_D f(x,y)\,dx\,dy=\iint_{D'}f(r\cos\theta,r\sin\theta)\,r\,dr\,d\theta.$$

**示范**：$\displaystyle\iint_D\sqrt{x^2+y^2}\,dx\,dy$，$D$：$x^2+y^2\le R^2$（圆盘）。极坐标 $D'$：$0\le r\le R$、$0\le\theta\le2\pi$：

$$\iint_D\sqrt{x^2+y^2}dx\,dy=\int_0^{2\pi}\int_0^R r\cdot r\,dr\,d\theta=2\pi\cdot\frac{R^3}{3}=\frac{2\pi R^3}{3}.$$

**「$\sqrt{x^2+y^2}$ 在极坐标下变成 $r$，圆盘区域变成矩形 $[0,R]\times[0,2\pi]$」**——被积函数与区域同时简化，这就是极坐标的价值。

**公式解析：极坐标的三步**

**第一步，判断适用**。区域是圆/扇形/环形，或被积函数含 $x^2+y^2$、$\sqrt{x^2+y^2}$、$\frac{y}{x}$——极坐标顺手；

**第二步，代换**。$x=r\cos\theta,\ y=r\sin\theta$，$f$ 换新，$dx\,dy=r\,dr\,d\theta$；

**第三步，定区域**。$D'$ 通常是「$r$ 从 $r_1(\theta)$ 到 $r_2(\theta)$、$\theta$ 从 $\alpha$ 到 $\beta$」——**极坐标的累次积分：先 $r$ 后 $\theta$**（对固定 $\theta$ 沿径向扫）。

## 3 高斯积分与经典示范

**示范（高斯积分）**：$\displaystyle\int_0^\infty e^{-x^2}dx=\frac{\sqrt\pi}{2}$。设 $I=\int_0^\infty e^{-x^2}dx$，则

$$I^2=\left(\int_0^\infty e^{-x^2}dx\right)\left(\int_0^\infty e^{-y^2}dy\right)=\iint_{[0,\infty)^2}e^{-(x^2+y^2)}dx\,dy.$$

换极坐标：$[0,\infty)^2$ 变为 $0\le\theta\le\frac\pi2$、$0\le r<\infty$：

$$I^2=\int_0^{\pi/2}\int_0^\infty e^{-r^2}r\,dr\,d\theta=\int_0^{\pi/2}\left[-\frac12e^{-r^2}\right]_0^\infty d\theta=\int_0^{\pi/2}\frac12d\theta=\frac\pi4,$$

故 $I=\frac{\sqrt\pi}{2}$。**高斯积分靠「平方 → 二重积分 → 极坐标」三步求值**——这是分析学最著名的积分之一。<span class="marginnote">「$I^2$ 化二重积分、极坐标分离变量」是高斯积分的标准证明，它同时给出 $\Gamma(\frac12)=\sqrt\pi$（§19.4 已用）。高斯积分是概率论的基石：正态分布的归一化常数 $\frac1{\sqrt{2\pi}\sigma}$ 由它确定。机器学习里的高斯核、扩散模型的高斯噪声、热传导方程的基本解——全部建在「$\int e^{-x^2}=\sqrt\pi$」这个积分上。「从极限到大模型」主线上，高斯积分是概率与机器学习的共同常数。</span>

**示范二（含 $x^2+y^2$ 的分式）**：$\displaystyle\iint_D\frac{dx\,dy}{1+x^2+y^2}$，$D$：$x^2+y^2\le1$：

$$\int_0^{2\pi}\int_0^1\frac{r}{1+r^2}dr\,d\theta=2\pi\cdot\frac12[\ln(1+r^2)]_0^1=\pi\ln2.$$

**示范三（扇形区域）**：$\displaystyle\iint_Dxy\,dx\,dy$，$D$ 是第一象限单位圆。极坐标 $0\le r\le1,\ 0\le\theta\le\frac\pi2$：

$$\int_0^{\pi/2}\int_0^1r^2\cos\theta\sin\theta\cdot r\,dr\,d\theta=\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\cdot\int_0^1r^3dr=\frac12\cdot\frac14=\frac18.$$

**「极坐标让圆域积分从『三角边界』变成『矩形区域』」**——每次换极坐标都是这个「区域矩形化」的胜利。

## 4 一般变量变换

除极坐标外，其他变换各有适用：

**（一）广义极坐标**：$x=ar\cos\theta,\ y=br\sin\theta$，$|J|=abr$——**椭圆区域 $x^2/a^2+y^2/b^2\le1$ 变成圆 $r\le1$**。

**（二）旋转/平移**：$x=u\cos\alpha-v\sin\alpha$ 等——处理倾斜区域。

**（三）双曲换元**：$x=r\cosh t,\ y=r\sinh t$——处理双曲区域。

**示范（广义极坐标）**：$\displaystyle\iint_D\sqrt{1-\frac{x^2}{a^2}-\frac{y^2}{b^2}}dx\,dy$，$D$ 是椭圆 $\frac{x^2}{a^2}+\frac{y^2}{b^2}\le1$。换 $x=ar\cos\theta,\ y=br\sin\theta$，$|J|=abr$：

$$\int_0^{2\pi}\int_0^1\sqrt{1-r^2}\,abr\,dr\,d\theta=ab\cdot2\pi\cdot\frac13=\frac{2\pi ab}{3}.$$

**「椭圆 → 圆」的缩放由广义极坐标完成，雅可比 $abr$ 把椭圆面积因子带进来**。<span class="marginnote">广义极坐标是「椭圆换圆」的通用工具：把椭圆 $x^2/a^2+y^2/b^2\le1$ 通过 $x=ar\cos\theta$ 变成单位圆，雅可比 $abr$ 记住面积伸缩。这暗示了更深的原理——<strong>任何「可逆光滑变换」都能把复杂区域拉成简单区域，代价是乘一个 $|J|$</strong>。第二十二章的球坐标、圆柱坐标，以及数值积分里的「等参元」（有限元方法），全是「用换元把区域拉直」的思想。</span>

## 5 变量变换的选择

| 情形 | 变换 | 雅可比 | 效果 |
| --- | --- | --- | --- |
| 圆/扇形/环形 | 极坐标 | $r$ | 区域矩形化 |
| 椭圆 | 广义极坐标 | $abr$ | 椭圆 → 圆 |
| 含 $x^2+y^2$ | 极坐标 | $r$ | 被积函数简化 |
| 含 $\frac yx$ | 极坐标 | $r$ | 比值变 $\tan\theta$ |

**选择标准**：区域或被积函数的对称结构决定变换——**「看到圆想极坐标」是第一直觉**。

## 6 小结

- **变量变换公式**：$\iint_Df=\iint_{D'}f|J|\,du\,dv$——$f$ 代入 + 乘 $|J|$ + 换区域。
- **极坐标**：$x=r\cos\theta,y=r\sin\theta$，$|J|=r$，$dx\,dy=r\,dr\,d\theta$。
- **高斯积分**：$\int_0^\infty e^{-x^2}=\frac{\sqrt\pi}2$——平方化二重 + 极坐标分离。
- **广义极坐标**：$x=ar\cos\theta,y=br\sin\theta$，$|J|=abr$——椭圆换圆。
- **选择**：圆/扇形/含 $x^2+y^2$ 用极坐标；椭圆用广义极坐标。

在下一节，我们进入**三重积分**：直角坐标、柱坐标、球坐标计算。三个坐标系的雅可比因子 $1,\ r,\ r^2\sin\varphi$ 决定面积微元的伸缩。

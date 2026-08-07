---
title: 定积分的换元法与分部积分法
date: 2026-08-07
---

# 定积分的换元法与分部积分法

<div class="epigraph">
<p>带上下限的积分，换元时连上下限一起换——这一「同步更新」让定积分计算既更快，也更容易在细节上栽跟头。</p>
<footer>—— 欧拉（Leonhard Euler），《积分学原理》</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§9.6 ｜ 2026-08-07</p>
</div>

## 为什么定积分要「重学一遍」换元与分部

第八章的换元与分部针对**不定积分**——结果是函数族。到定积分，第八章的整套技术依然可用，但多了「**上下限**」这个成员：换元后变量从 $x$ 变 $t$，上下限也必须跟着从「$x$ 的界限」变成「$t$ 的界限」。这一「同步换限」让计算更直接（不必回代），也埋下了独特的错误源。

本节的公式可以看作「牛顿—莱布尼茨 + 不定积分技术」的组合拳：**换元/分部处理被积函数，同步换限处理端点，最后代入即得**。<span class="marginnote">定积分换元与不定积分的核心差别：不定积分换元后必须<strong>回代</strong>（把 $t$ 换成 $x$），定积分换元后<strong>不必回代</strong>——因为上下限已经同步换成 $t$ 的界限，直接在 $t$ 世界里代入即可。这个「少一步回代」是定积分换元的好处，但它要求「换限」必须精确——很多人回代忘不了，却常常忘记换限。</span>

## 1 定积分的换元法

**定理（定积分换元法）：设 $f$ 在 $[a,b]$ 上连续，$x=\varphi(t)$ 在 $[\alpha,\beta]$ 上单调（或 $\varphi$ 在 $[\alpha,\beta]$ 上可导且 $\varphi(\alpha)=a,\ \varphi(\beta)=b$，值域含于 $[a,b]$），则**

$$\int_a^b f(x)\,dx=\int_{\alpha}^{\beta}f(\varphi(t))\,\varphi'(t)\,dt.$$

**证明**（用原函数）：设 $F$ 是 $f$ 的原函数，则 $F(\varphi(t))$ 是 $f(\varphi(t))\varphi'(t)$ 的原函数（链式法则）。由牛顿—莱布尼茨：

$$\int_\alpha^\beta f(\varphi(t))\varphi'(t)dt=F(\varphi(\beta))-F(\varphi(\alpha))=F(b)-F(a)=\int_a^bf(x)dx.$$

∎

**公式解析：换元法的「换限三步骤」**

**第一步，选代换**：令 $x=\varphi(t)$，写出 $dx=\varphi'(t)dt$；
**第二步，同步换限**：$x=a\Rightarrow t=\alpha$（解 $\varphi(\alpha)=a$），$x=b\Rightarrow t=\beta$；
**第三步，全量替换**：被积函数、$dx$、上下限全部换成 $t$，积分 $\int_\alpha^\beta\cdots dt$，**不回代**，直接代入。

**示范**：$\displaystyle\int_0^4\frac{dx}{1+\sqrt x}$。令 $t=\sqrt x$（即 $x=t^2$），$dx=2t\,dt$。换限：$x=0\Rightarrow t=0$，$x=4\Rightarrow t=2$：

$$\int_0^4\frac{dx}{1+\sqrt x}=\int_0^2\frac{2t}{1+t}dt=2\int_0^2\left(1-\frac1{1+t}\right)dt=2\bigl[t-\ln(1+t)\bigr]_0^2=2(2-\ln3).$$

**不回代**，直接在 $t$ 里代入上下限 $0,2$。

> **辨析｜易错点：**定积分换元的两大陷阱。**一是忘换限**：$t=\sqrt x$ 后仍写 $0,4$ 作为上下限——那会把 $t$ 与 $x$ 的界限混在一起，结果必然错误。**二是换限方向**：换元 $x=a\cos t$ 时，$t$ 的范围要取使 $\sin t,\cos t$ 符号可判的单调区间（如 $\sqrt{a^2-x^2}$ 中 $t\in[0,\pi]$ 使 $\sin t\ge0$）。**「单调区间」保证换元可逆且符号清晰**，是三角代换换限的关键。

## 2 定积分的分部积分法

**定理（定积分分部积分）：设 $u,v$ 在 $[a,b]$ 上连续可导，则**

$$\int_a^b u(x)\,dv=\bigl[u(x)v(x)\bigr]_a^b-\int_a^b v(x)\,du.$$

即

$$\int_a^b u v'\,dx=\bigl[uv\bigr]_a^b-\int_a^b vu'\,dx.$$

**示范一**：$\displaystyle\int_0^1 x e^x\,dx$。取 $u=x$、$dv=e^xdx$：

$$\int_0^1 xe^xdx=\bigl[xe^x\bigr]_0^1-\int_0^1e^xdx=(e-0)-(e-1)=1.$$

**「代入端点」项 $[uv]_a^b$ 在分部积分里替代了「$+C$ 常数」**——定积分分部后无需再管常数，端点的差自动消掉任意常数。

**示范二**：$\displaystyle\int_0^{\pi/2}\sin^2x\,dx$。直接算原函数再代入即可，或利用对称性。这里展示分部：取 $u=\sin x$、$dv=\sin x\,dx$：

$$\int_0^{\pi/2}\sin^2xdx=\bigl[-\sin x\cos x\bigr]_0^{\pi/2}+\int_0^{\pi/2}\cos^2xdx=0+\int_0^{\pi/2}(1-\sin^2x)dx,$$

设 $I=\int_0^{\pi/2}\sin^2xdx$，则 $I=\frac\pi2-I$，故 $I=\frac\pi4$。**循环消去在定积分里同样管用**——且端点的边界项 $[uv]_0^{\pi/2}=0$ 让消去更干净。

> **辨析｜易错点：**定积分分部的易错点集中在边界项。**一是忘写 $[uv]_a^b$**——它来自乘积法则的积分，不可或缺；**二是代入端点时把 $u,v$ 算错**——端点处的值 $u(b)v(b)-u(a)v(a)$ 要代入完整乘积。另外，**定积分里「$C$」不再出现**——牛顿—莱布尼茨把常数自动消掉，写了 $C$ 反而是冗余（且容易引发「$C$ 从哪来」的混乱）。

## 3 公式解析：对称性的妙用

定积分换元法的一个「甜点」是**奇偶性简化**。若 $f$ 在 $[-a,a]$ 上连续：

$$\int_{-a}^{a}f(x)\,dx=\begin{cases}0,&f\ \text{为奇函数}\\2\int_0^{a}f(x)\,dx,&f\ \text{为偶函数}\end{cases}$$

**证明（以奇函数为例）**：换元 $t=-x$，$x=-t$，$dx=-dt$，上下限 $-a\to a$、$a\to-a$：

$$\int_{-a}^{a}f(x)dx=-\int_{a}^{-a}f(-t)dt=\int_{-a}^{a}f(-t)dt=-\int_{-a}^{a}f(t)dt,$$

最后一步用奇性 $f(-t)=-f(t)$。于是 $\int_{-a}^{a}f=\text{自身}$ 的相反数，故为 0。∎

**三步拆解**：
- **第一步，对称换元**：$t=-x$ 把 $[-a,a]$ 映到自身，方向翻转；
- **第二步，奇性翻译**：$f(-t)=-f(t)$ 让积分变成自身负值；
- **第三步，方程求解**：$I=-I\Rightarrow I=0$。

**示范**：$\displaystyle\int_{-\pi}^{\pi}\frac{x\sin x}{1+x^2}dx$——被积函数是偶函数 × 偶函数？$x\sin x$ 是偶（奇×奇=偶），除以偶 $1+x^2$ 仍偶，故 $\int_{-\pi}^{\pi}=2\int_0^\pi\cdots$。若被积函数是奇函数（如 $x^3\cos x$），整个区间积分直接为 0——**「奇函数在对称区间积分为零」是最高频的偷懒神器**。<span class="marginnote">奇偶性简化在物理与概率里屡见不鲜：正态分布密度关于均值对称，故 $\int_{-\infty}^{\infty}(x-\mu)f(x)dx=0$（期望即均值）；傅里叶级数里，奇函数只有正弦项、偶函数只有余弦项（第十五章）——对称性从积分一直传导到展开系数。</span>

## 4 定积分技术的联合应用

**示范一（换元 + 对称）**：$\displaystyle\int_0^{\pi}\frac{x\sin x}{1+\cos^2x}dx$。令 $t=\pi-x$（常用技巧「$x\to\pi-x$」）：

$$\int_0^{\pi}\frac{x\sin x}{1+\cos^2x}dx=\int_0^{\pi}\frac{(\pi-t)\sin t}{1+\cos^2t}dt=\pi\int_0^\pi\frac{\sin x}{1+\cos^2x}dx-\int_0^\pi\frac{x\sin x}{1+\cos^2x}dx.$$

设原积分为 $I$，则 $I=\pi\int_0^\pi\frac{\sin x}{1+\cos^2x}dx-I$，故

$$I=\frac\pi2\int_0^\pi\frac{\sin x}{1+\cos^2x}dx=\frac\pi2\bigl[-\arctan(\cos x)\bigr]_0^\pi=\frac\pi2\cdot\frac\pi2=\frac{\pi^2}{4}.$$

**「对称变换 $x\mapsto\pi-x$ 产生自身方程」**——这种「换元造方程」的技巧是定积分计算的进阶神器。

**示范二（换元处理周期）**：$\int_0^{2\pi}\frac{dx}{a+\cos x}$（$a>1$）——含三角的分式积分可用万能代换 $t=\tan\frac x2$，换限 $0\to0,2\pi\to$（$\tan$ 在 $\pi$ 处间断需分段）。**「万能代换在定积分里要小心区间分裂」**——周期函数的积分常需拆段处理，这是定积分独有的注意事项。

## 5 小结

- **定积分换元**：$x=\varphi(t)$ 时同步换限 $a\to\alpha,\ b\to\beta$，全量替换后不回代直接代入。
- **两大陷阱**：忘换限、三角代换的单调区间选择。
- **定积分分部**：$\int_a^b u\,dv=[uv]_a^b-\int_a^b v\,du$；边界项不可省，$C$ 不再出现。
- **对称性简化**：奇函数在 $[-a,a]$ 积分为 0，偶函数为 $2\int_0^a$；对称换元可造方程。
- **技术全景**：换元、分部、对称、换元造方程——定积分计算的完整工具箱。

在下一节，我们回到泰勒公式，用积分语言重新书写它的余项：**泰勒公式的积分型余项**。拉格朗日余项来自中值定理，积分型余项则来自微积分基本定理——两条路殊途同归，后者还天然给出了余项的「可控估计」。

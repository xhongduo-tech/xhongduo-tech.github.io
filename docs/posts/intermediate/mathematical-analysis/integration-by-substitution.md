---
title: 换元积分法：第一换元法（凑微分）与第二换元法
date: 2026-08-07
---

# 换元积分法：第一换元法（凑微分）与第二换元法

<div class="epigraph">
<p>积分学中最重要的艺术，就是找到一个好的变量替换——它把陌生的积分翻译成熟知的形状，如同翻译家把外文诗歌译回母语。</p>
<footer>—— 欧拉（Leonhard Euler），《积分学原理》（Institutiones Calculi Integralis）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§8.2 ｜ 2026-08-07</p>
</div>

## 为什么需要「换元」

直接积分法只能处理「能拆成基本型」的积分。但 $e^{\sin x}\cos x$、$\frac1{1+\sqrt x}$ 这类积分拆不出来——它们的结构是**复合函数**。复合函数的求导用链式法则，那逆运算呢？**链式法则的逆，就是换元积分法。**

换元法有两副面孔：**第一换元法（凑微分）**——从「积分号里藏着复合函数的导数」出发，把 $g'(x)dx$ 看成 $dg$，直接拼出 $d(\text{内层})$；**第二换元法**——主动引入新变量 $x=\varphi(t)$，把难看的根式变成友好的三角或代数式。<span class="marginnote">「换元」的本质是「视角切换」：同样的积分，换一个自变量来衡量，结构可能面目全非。第一换元法问「这堆东西是不是某个函数的导数的一部分」，第二换元法问「换什么变量能让根式消失」。这两个问题合起来，覆盖了几乎所有「能用手算」的积分——剩下的，只能交给数值方法或特殊函数。</span>

## 1 第一换元法（凑微分）

**定理（第一换元法）：设 $F$ 是 $f$ 的原函数，$u=\varphi(x)$ 可导，则**

$$\int f(\varphi(x))\,\varphi'(x)\,dx=\int f(u)\,du=F(u)+C=F(\varphi(x))+C.$$

**公式解析：为什么叫「凑微分」**

把 $\varphi'(x)dx$ 看成 $d\varphi(x)$（微分定义 $dy=f'(x)dx$，§5.5），则

$$\int f(\varphi(x))\,\underbrace{\varphi'(x)\,dx}_{d\varphi(x)}=\int f(u)\,du.$$

**三步拆解**：
- **第一步，认内层**：找「内层函数」$\varphi(x)$，看它的导数 $\varphi'(x)$ 是否**以因子形式**出现在被积函数里；
- **第二步，凑微分**：把 $\varphi'(x)dx$ 合并成 $du$，积分变成 $\int f(u)\,du$（基本型）；
- **第三步，代回**：对 $u$ 积分得 $F(u)$，再把 $u=\varphi(x)$ 代回。

**示范一**：$\displaystyle\int e^{\sin x}\cos x\,dx$。内层 $\varphi(x)=\sin x$，$\varphi'(x)=\cos x$ 恰好是因子：

$$\int e^{\sin x}\cos x\,dx=\int e^{\sin x}\,d(\sin x)=\int e^u\,du=e^u+C=e^{\sin x}+C.$$

**示范二**：$\displaystyle\int\frac{\ln x}{x}\,dx$。内层 $\varphi=\ln x$，$\varphi'=\frac1x$ 在分母上：

$$\int\ln x\cdot\frac1x\,dx=\int\ln x\,d(\ln x)=\frac{(\ln x)^2}{2}+C.$$

**示范三**：$\displaystyle\int\tan x\,dx=\int\frac{\sin x}{\cos x}dx$。内层 $\varphi=\cos x$，分子 $\sin x\,dx=-d(\cos x)$：

$$\int\tan x\,dx=-\int\frac{d(\cos x)}{\cos x}=-\ln|\cos x|+C.$$

这是「凑微分」里「**差一个负号/常数就要补上**」的经典示范。

> **辨析｜易错点：**第一换元法的关键是**内层导数必须「以因子形式」出现**（或差一个常数倍）。$\int e^{\sin x}dx$ 就不能直接凑——没有 $\cos x$ 因子，$\sin x$ 的导数不出现。若差常数倍：$\int x\cos x^2\,dx$ 中 $x\,dx=\frac12 d(x^2)$，凑完要乘 $\frac12$。**「缺因子则不能凑，差倍数则补倍数」**——这是第一换元法的全部要点。

## 2 第二换元法

**定理（第二换元法）：设 $x=\varphi(t)$ 严格单调、可导且 $\varphi'(t)\ne0$，$f(\varphi(t))\varphi'(t)$ 有原函数 $G(t)$，则**

$$\int f(x)\,dx=\int f(\varphi(t))\,\varphi'(t)\,dt=G(t)+C=G(\varphi^{-1}(x))+C.$$

**公式解析：为什么换元后要「回代」**

把 $x$ 换成 $\varphi(t)$，$dx=\varphi'(t)dt$，积分变成对 $t$ 的积分。算出 $G(t)$ 后，**必须**用反函数 $t=\varphi^{-1}(x)$ 换回 $x$——因为原积分是「关于 $x$ 的函数」。

**三步拆解**：
- **第一步，选代换**：挑 $\varphi$ 让根式/复杂结构消失（三角代换消根式、倒代换处理分母高次）；
- **第二步，全量替换**：$x=\varphi(t)$、$dx=\varphi'(t)dt$，被积函数与 $dx$ **全部换成 $t$**（不能留下 $x$）；
- **第三步，回代**：对 $t$ 积分，用反函数回到 $x$。

**示范（三角代换）**：$\displaystyle\int\sqrt{1-x^2}\,dx$。令 $x=\sin t$（$t\in[-\frac\pi2,\frac\pi2]$，此时 $\cos t\ge0$）：

$$dx=\cos t\,dt,\qquad \sqrt{1-x^2}=\sqrt{1-\sin^2t}=\cos t,$$

$$\int\sqrt{1-x^2}\,dx=\int\cos^2t\,dt=\int\frac{1+\cos2t}{2}\,dt=\frac t2+\frac{\sin2t}{4}+C.$$

回代：$t=\arcsin x$，$\sin2t=2\sin t\cos t=2x\sqrt{1-x^2}$，故

$$\int\sqrt{1-x^2}\,dx=\frac{\arcsin x}{2}+\frac{x\sqrt{1-x^2}}{2}+C.$$

**示范（根式代换）**：$\displaystyle\int\frac{dx}{1+\sqrt x}$。令 $t=\sqrt x$（$x=t^2$，$dx=2t\,dt$）：

$$\int\frac{dx}{1+\sqrt x}=\int\frac{2t}{1+t}dt=\int\left(2-\frac2{1+t}\right)dt=2t-2\ln|1+t|+C=2\sqrt x-2\ln(1+\sqrt x)+C.$$

## 3 公式解析：三角代换的三种形态

根式与三角的对应关系是第二换元法的核心记忆表：

| 根式 | 代换 | 化简结果 | 适用 |
| --- | --- | --- | --- |
| $\sqrt{a^2-x^2}$ | $x=a\sin t$ | $a\cos t$ | 圆的弧长、面积 |
| $\sqrt{a^2+x^2}$ | $x=a\tan t$ | $a\sec t$ | 双曲/反正切型 |
| $\sqrt{x^2-a^2}$ | $x=a\sec t$ | $a\tan t$ | 双曲线型 |

以 $\sqrt{a^2-x^2}$ 为例拆解：

- **第一步，选代换**：$x=a\sin t$（也可 $x=a\cos t$，习惯取 $\sin$），$t\in[-\frac\pi2,\frac\pi2]$ 保证 $\cos t\ge0$；
- **第二步，化简根式**：$\sqrt{a^2-a^2\sin^2t}=a\cos t$——根式变成**没有根号**的 $a\cos t$；
- **第三步，换 $dx$**：$dx=a\cos t\,dt$，积分里出现 $\cos^2t$，用倍角公式 $\cos^2t=\frac{1+\cos2t}{2}$ 处理。

**每做一次三角代换，都要画辅助直角三角形来回代**：$\sin t=x/a$，$t=\arcsin(x/a)$，其余三角函数用勾股关系写回 $x$。**「画三角形回代」是三角代换的收尾动作**，跳过它就会把 $t$ 的答案错当 $x$ 的答案。<span class="marginnote">三角代换本质是「把代数根式翻译成三角函数」，利用的是 $\sin^2+\cos^2=1$、$1+\tan^2=\sec^2$ 这些三角恒等式——根式在三角世界里「开方」是免费的。这套翻译在第二级《复变函数与积分变换》、物理学里的简谐运动、以及概率论的正态分布积分（$\int e^{-x^2}dx$ 的极坐标技巧）中反复出现。</span>

## 4 两种换元的对比

| | 第一换元法 | 第二换元法 |
| --- | --- | --- |
| 方向 | 内层 $u=\varphi(x)$ | 新变量 $x=\varphi(t)$ |
| 思路 | 找内层导数因子，凑成 $du$ | 主动换变量，消根式 |
| 典型 | $e^{\sin x}\cos x$、$\frac{\ln x}{x}$ | $\sqrt{1-x^2}$、$\frac1{1+\sqrt x}$ |
| 是否回代 | 是（$u$ 换成 $x$） | 是（$t$ 换成 $x$） |
| 别称 | 凑微分法、配元法 | 代换法、替换法 |

**两者其实可以互相转化**：第一换元法「设 $u=\varphi(x)$」如果反解 $x$ 可行，就等价于第二换元法。区别只在**先看到什么**：看到「复合 + 内层导数因子」用第一换元，看到「根式 + 无合适内层」用第二换元。**判断用哪种，是积分计算最重要的直觉**——通常先试第一换元（快），不行再考虑第二换元。

## 5 小结

- **第一换元法**：$\int f(\varphi(x))\varphi'(x)dx=\int f(u)du$；认内层、凑微分、代回三步骤。
- **凑微分要点**：内层导数必须「以因子形式」出现；差倍数要补倍数（$\int x\cos x^2\,dx=\frac12\sin x^2+C$）。
- **第二换元法**：$x=\varphi(t)$，$dx=\varphi'(t)dt$，全量替换后积分、回代。
- **三角代换三表**：$\sqrt{a^2-x^2}\to x=a\sin t$、$\sqrt{a^2+x^2}\to x=a\tan t$、$\sqrt{x^2-a^2}\to x=a\sec t$；画三角形回代。
- **选择直觉**：先试第一换元，根式难题用第二换元。

在下一节，我们学习积分计算的第二套高级武器：**分部积分法**。它来自乘积求导法则的逆运算，专门对付「两种不同类型函数相乘」的积分（如 $x\sin x$、$e^x\cos x$、$x\ln x$）——这类积分换元法常常束手无策。

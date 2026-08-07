---
title: 欧拉积分：Γ 函数与 B 函数及其关系
date: 2026-08-07
---

# 欧拉积分：Γ 函数与 B 函数及其关系

<div class="epigraph">
<p>阶乘只是 Γ 函数的一个特例——把「正整数相乘」延伸为「连续变量的积分」，欧拉打开了一扇通往概率、物理与数论的大门。</p>
<footer>—— 莱昂哈德·欧拉（Leonhard Euler），1729 年关于 Γ 函数的工作（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§19.4 ｜ 2026-08-07</p>
</div>

## 为什么 Γ 与 B 是「特殊函数之王」

含参量反常积分的理论武器（§19.2–19.3）在此结出最丰硕的果实：**欧拉积分**——两个以积分定义的函数，它们统治概率论、数理方程、统计学与物理学：

$$\Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}\,dx\qquad(\alpha>0),$$

$$B(p,q)=\int_0^1x^{p-1}(1-x)^{q-1}\,dx\qquad(p,q>0).$$

**Γ 函数是「连续化的阶乘」**（$\Gamma(n+1)=n!$），**B 函数是「Beta 分布」的归一化常数**。它们的关系、递推、特殊值，构成分析学最常用的特殊函数库。<span class="marginnote">Γ 函数几乎是「现代统计的第一函数」：Gamma 分布（等待时间）、卡方分布（$\chi^2=\Gamma(\frac n2,\frac12)$）、Beta 分布（先验）、t 分布、F 分布——第二级《概率论与数理统计》的连续分布几乎全用 Γ/B 函数定义。机器学习里的 Dirichlet 分布（主题模型、变分推断）也由 Γ 函数写成。<strong>今天学的两个积分，是统计学与机器学习「分布工厂」的母体</strong>。</span>

## 1 Γ 函数及其性质

**Γ 函数（Gamma function）**：

$$\Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}\,dx\qquad(\alpha>0).$$

**收敛性**：$x\to0$ 时 $x^{\alpha-1}$（瑕点，$\alpha>0$ 收敛）；$x\to\infty$ 时 $e^{-x}$ 衰减极快（无穷区间收敛）。**$\alpha>0$ 时积分收敛**。

**核心性质**：

$$\Gamma(\alpha+1)=\alpha\Gamma(\alpha)\qquad(\text{递推公式});$$

$$\Gamma(1)=1,\qquad\Gamma(n+1)=n!\qquad(n\ \text{正整数});$$

$$\Gamma\left(\frac12\right)=\sqrt\pi.$$

**公式解析：递推公式与 $\Gamma(\frac12)$**

**第一步，分部积分得递推**。$\Gamma(\alpha+1)=\int_0^\infty x^\alpha e^{-x}dx$，分部积分（$u=x^\alpha,\ dv=e^{-x}dx$）：

$$\Gamma(\alpha+1)=\left[-x^\alpha e^{-x}\right]_0^\infty+\alpha\int_0^\infty x^{\alpha-1}e^{-x}dx=0+\alpha\Gamma(\alpha).$$

**边界项为零**（$x^\alpha e^{-x}\to0$）——**递推 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$**（§8.3 分部积分的「长期客户」在此兑现）。

**第二步，从 $\Gamma(1)=1$ 推出阶乘**。$\Gamma(1)=\int_0^\infty e^{-x}dx=1$，递推：$\Gamma(2)=1\cdot1=1$、$\Gamma(3)=2\cdot1=2$、…、$\Gamma(n+1)=n!$——**Γ 是阶乘的连续延伸**。

**第三步，算 $\Gamma(\frac12)$**。用换元 $x=t^2$（$dx=2t\,dt$）：

$$\Gamma\left(\frac12\right)=\int_0^\infty x^{-1/2}e^{-x}dx=2\int_0^\infty e^{-t^2}dt=\sqrt\pi,$$

最后用高斯积分 $\int_0^\infty e^{-t^2}dt=\frac{\sqrt\pi}2$（§21.5 用极坐标证）。∎

**示范**：$\Gamma(\frac52)=\frac32\Gamma(\frac32)=\frac32\cdot\frac12\Gamma(\frac12)=\frac{3\sqrt\pi}{4}$——**递推把半整数 Γ 归约到 $\sqrt\pi$**。

## 2 B 函数及其性质

**B 函数（Beta function）**：

$$B(p,q)=\int_0^1x^{p-1}(1-x)^{q-1}\,dx\qquad(p,q>0).$$

**核心性质**：

$$B(p,q)=B(q,p)\qquad(\text{对称性});$$

$$B(p,q)=\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}\qquad(\text{与 Γ 的关系});$$

$$B(p,q)=\int_0^\infty\frac{t^{p-1}}{(1+t)^{p+q}}dt\qquad(\text{换元 } x=\frac{t}{1+t});$$

$$B(p,q)=2\int_0^{\pi/2}\sin^{2p-1}\theta\cos^{2q-1}\theta\,d\theta\qquad(\text{换元 }x=\sin^2\theta).$$

**公式解析：B 与 Γ 的关系为什么成立**

**第一步，写出乘积**。$\Gamma(p)\Gamma(q)=\int_0^\infty\int_0^\infty x^{p-1}y^{q-1}e^{-(x+y)}dx\,dy$——**两个积分的乘积**（§21 二重积分的预览，这里先用累次积分形式）；

**第二步，换元极坐标**。$x=r\cos^2\theta,\ y=r\sin^2\theta$（$r=x+y$，$\theta$ 比例参数），$dx\,dy$ 的雅可比给出 $r\,d\theta$ 因子（§19.2 反函数组的雅可比思想）。积分分离：

$$\Gamma(p)\Gamma(q)=\int_0^\infty r^{p+q-1}e^{-r}dr\cdot\int_0^{\pi/2}(\cos^2\theta)^{p-1}(\sin^2\theta)^{q-1}\cdot2\sin\theta\cos\theta\,d\theta;$$

**第三步，读出 B**。第一个因子 $=\Gamma(p+q)$；第二个因子 $=\int_0^1t^{p-1}(1-t)^{q-1}dt$（$t=\sin^2\theta$）$=B(p,q)$。故 $\Gamma(p)\Gamma(q)=\Gamma(p+q)B(p,q)$。∎

**要点**：**$B=\frac{\Gamma\Gamma}{\Gamma}$——三个 Γ 的「分数」**是 B 与 Γ 关系的标准形态。它让 B 函数的任何计算都归约为 Γ 函数的递推与特殊值。

## 3 B 函数的应用：Beta 分布与三角积分

**示范一（Beta 分布归一化）**：概率密度 $f(x)=\frac{x^{p-1}(1-x)^{q-1}}{B(p,q)}$ 在 $[0,1]$ 上积分为 1——**B 函数是 Beta 分布的归一化常数**。Beta 分布是贝叶斯统计里「伯努利参数 $p$ 的先验」，$B(p,q)$ 的分母保证密度积分为 1。<span class="marginnote">Beta 分布在贝叶斯推断里无处不在：抛硬币的成功率 $\theta$ 的先验取 $\text{Beta}(a,b)$，观察到 $n_1$ 次正面、$n_2$ 次反面后，后验是 $\text{Beta}(a+n_1,b+n_2)$——共轭先验让后验仍在 Beta 家族。第二级《概率论与数理统计》与第四级《机器学习》的贝叶斯方法，全靠 $B(p,q)$ 归一化。<strong>今天算的 $B(p,q)$，是贝叶斯统计的「分母」</strong>。</span>

**示范二（三角函数幂积分）**：$\int_0^{\pi/2}\sin^{2p-1}\theta\cos^{2q-1}\theta d\theta=\frac12B(p,q)=\frac{\Gamma(p)\Gamma(q)}{2\Gamma(p+q)}$。令 $p=q=\frac12$：

$$\int_0^{\pi/2}d\theta=\frac{\Gamma(\frac12)^2}{2\Gamma(1)}=\frac\pi2,$$

正确。令 $p=1,q=\frac12$：$\int_0^{\pi/2}\cos^{2q-1}\theta d\theta=\int_0^{\pi/2}\cos^0\theta d\theta=\frac\pi2=\frac{\Gamma(1)\Gamma(\frac12)}{2\Gamma(\frac32)}$——验证 $\Gamma(\frac32)=\frac{\sqrt\pi}2$。**三角函数幂积分全部归约为 Γ 函数**。

**示范三（无穷积分换元）**：$\int_0^\infty\frac{x^{p-1}}{(1+x)^{p+q}}dx=B(p,q)$——**把 $[0,\infty)$ 的积分用 $x=\frac t{1+t}$ 换成 $[0,1]$ 的 B 函数**。这个换元在概率（F 分布）、物理（散射截面）里常见。

## 4 欧拉积分与特殊函数的地位

欧拉积分是「以积分定义的特殊函数」的始祖，它们的地位：

| 应用领域 | 用到的函数 |
| --- | --- |
| 概率统计 | Gamma 分布、Beta 分布、$\chi^2$、t、F 分布 |
| 机器学习 | Dirichlet 分布（主题模型）、变分推断 |
| 数学物理 | 热传导、扩散方程的解、散射截面 |
| 数论 | 黎曼 ζ 函数（与 Γ 的关系 $\zeta(s)=\frac1{\Gamma(s)}\int_0^\infty\frac{x^{s-1}}{e^x-1}dx$） |
| 组合数学 | 阶乘的连续延伸、墙式积分 |

**「用积分定义函数」是处理「没有初等公式」的函数的通用策略**——§9.5 的误差函数 erf、这里的 Γ/B，都是这条路的产物。<span class="marginnote">Γ 函数在数论里的惊人连接：黎曼 ζ 函数 $\zeta(s)=\sum\frac1{n^s}$ 与 Γ 函数满足 $\zeta(s)=\frac1{\Gamma(s)}\int_0^\infty\frac{x^{s-1}}{e^x-1}dx$——这条「ζ-Γ 公式」把数论（素数分布）与分析（特殊函数）焊在一起，是黎曼猜想研究的基础工具。而「$\Gamma(\frac12)=\sqrt\pi$」这个「阶乘与圆周率的意外相遇」，正是「特殊函数」充满惊喜的证明。到第二级《数论》与《复变函数》，你会与这两个函数重逢。</span>

## 5 欧拉积分求值总表

| 积分 | 值 | 方法 |
| --- | --- | --- |
| $\Gamma(n+1)$ | $n!$ | 递推公式 |
| $\Gamma(\frac12)$ | $\sqrt\pi$ | 高斯积分 |
| $\Gamma(\frac{k+1}2)$ | 半整数值 | 递推 + $\sqrt\pi$ |
| $B(p,q)$ | $\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}$ | 极坐标乘积 |
| $\int_0^{\pi/2}\sin^{2p-1}\cos^{2q-1}$ | $\frac12B(p,q)$ | 换元 |
| $\int_0^\infty\frac{x^{p-1}}{(1+x)^{p+q}}$ | $B(p,q)$ | 换元 |

**一切欧拉积分求值 = 归约为 Γ 的递推 + $\Gamma(\frac12)=\sqrt\pi$**。

## 6 小结

- **Γ 函数**：$\Gamma(\alpha)=\int_0^\infty x^{\alpha-1}e^{-x}dx$；递推 $\Gamma(\alpha+1)=\alpha\Gamma(\alpha)$、$\Gamma(n+1)=n!$、$\Gamma(\frac12)=\sqrt\pi$。
- **B 函数**：$B(p,q)=\int_0^1x^{p-1}(1-x)^{q-1}dx$；对称、可换元成无穷积分、三角积分。
- **关系**：$B(p,q)=\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}$——极坐标乘积的产物。
- **应用**：Beta/Gamma 分布归一化、三角幂积分、贝叶斯共轭先验。
- **地位**：特殊函数之王，连接概率、数理方程、数论与物理。

在下一节，我们进入第二十章：**曲线积分**。沿曲线的积分——第一型（对弧长）与第二型（对坐标），它们是「定积分的曲线版」，也是通向格林公式、路径无关性的前奏。

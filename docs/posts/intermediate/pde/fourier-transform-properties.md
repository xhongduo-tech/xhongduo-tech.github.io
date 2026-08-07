---
title: 傅里叶变换的基本性质（平移、微分、卷积）
date: 2026-08-08
---

# 傅里叶变换的基本性质（平移、微分、卷积）

<div class="epigraph">
<p>时域的平移，是频域的相位；时域的微分，是频域的乘法。</p>
<footer>—— 傅里叶变换的代数性质</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第七章 ｜ 2026-08-08</p>
</div>

## 为什么从基本性质开始

傅里叶变换的价值不在定义本身，而在它把「时域的运算」翻译成「频域的代数」。平移变相位、微分变乘法、卷积变乘积——这三条是解 PDE 的全部弹药。这一节把傅里叶变换的基本性质系统整理出来，每一条都给出验证思路与物理/几何直觉。它们是下一节卷积定理、以及后续所有「变换解 PDE」应用的运算基础。

## 1 线性、平移与伸缩

**线性**：

$$
\mathcal{F}[af + bg] = a\,\hat f + b\,\hat g
$$

积分是线性算子，直接由定义得出。PDE 的线性叠加原理在频域里保持。

**平移（shift）**：$f(x-a)$ 的变换

$$
\mathcal{F}[f(x-a)](\omega) = e^{-ia\omega}\,\hat f(\omega)
$$

**证明**：$\int f(x-a)e^{-i\omega x}dx$，换元 $y = x-a$：$=\int f(y)e^{-i\omega(y+a)}dy = e^{-ia\omega}\hat f(\omega)$。

**物理**：时域平移不改变频谱的**幅值**（$|e^{-ia\omega}| = 1$），只改变**相位**——信号整体移动，各频率成分的含量不变，只是相位滞后了 $a\omega$。<span class="marginnote">这条性质是信号处理「线性相位」概念的来源：纯时移 = 相位随频率线性变化。它在滤波器设计中极端重要——理想的延迟滤波器就是 $e^{-ia\omega}$ 的相位响应。</span>

**伸缩（scaling）**：$f(ax)$（$a \ne 0$）的变换

$$
\mathcal{F}[f(ax)](\omega) = \frac{1}{|a|}\,\hat f\!\left(\frac{\omega}{a}\right)
$$

**证明**：换元 $y = ax$。$a > 0$ 时 $dx = dy/a$、积分限不变：$=\frac{1}{a}\hat f(\omega/a)$；$a < 0$ 时积分限翻转，绝对值吸收符号。

**物理**：时域压扁（$a > 1$）⇔ 频域展宽、幅度变小——与上一节「窄脉冲宽频谱」一致。**伸缩性质是「时频不确定性」的定量表达。**

## 2 微分与乘法

**微分性质（求导变乘法）**——傅里叶变换最锋利的性质：

$$
\mathcal{F}[f'(x)](\omega) = i\omega\,\hat f(\omega), \qquad \mathcal{F}[f^{(n)}(x)](\omega) = (i\omega)^n\,\hat f(\omega)
$$

**证明**（分部积分）：$\int f'(x)e^{-i\omega x}dx = \big[f e^{-i\omega x}\big]_{-\infty}^{\infty} + i\omega\int f e^{-i\omega x}dx$。边界项为零（$f$ 在无穷远处衰减），留下 $i\omega\hat f$。

**推论（对拉普拉斯算子）**：$f''$ 的变换是 $-\omega^2\hat f$——**二阶导变乘 $-\omega^2$**。这就是第五篇傅里叶变换法里 $u_{xx} \mapsto -\omega^2\hat u$ 的正式来源：拉普拉斯算子在频域是对角化的。

**对称性质（乘 $x$ 变求导）**：

$$
\mathcal{F}[x f(x)](\omega) = i\,\hat f\,'(\omega)
$$

对 $\omega$ 求导可与积分交换：$\frac{d}{d\omega}\int f e^{-i\omega x}dx = \int f(-ix)e^{-i\omega x}dx$，整理即得。<span class="marginnote">「微分 ⇄ 乘法」是一对完全对称的操作：时域求导 = 频域乘 $i\omega$，时域乘 $x$ = 频域求导。这个对偶（连同平移与调制的对偶）让傅里叶变换成为一种「对合式」运算——变换后再变换几乎回到自身（对偶性）。它是调幅（乘 $e^{i\omega_0x}$ 搬移频谱）这类物理操作的代数根源。</span>

## 3 调制与共轭

**调制（modulation）**：

$$
\mathcal{F}[e^{ia x}f(x)](\omega) = \hat f(\omega - a)
$$

**证明**：把 $e^{iax}$ 并进核：$\int f(x)e^{-i(\omega-a)x}dx = \hat f(\omega-a)$。**时域乘复指数 = 频域搬移**——这是调幅、混频的全部原理。

**共轭与奇偶**：

$$
\overline{\hat f(\omega)} = \hat f(-\omega) \quad（f \text{ 实值}）, \qquad \mathcal{F}[\bar f](\omega) = \overline{\hat f(-\omega)}
$$

实值函数 $f$ 的变换满足 $\hat f(-\omega) = \overline{\hat f(\omega)}$——**实函数的频谱关于原点共轭对称**：负频率的信息由正频率完全决定。<span class="marginnote">这条对称性让「只存一半频谱」成为可能：实信号的负频分量是正频分量的共轭镜像。工程中的解析信号、单边带调制都建立在这条性质上。对 PDE 求解的意义：解通常是实值，变换的对称性可用来简化计算、检查数值。</span>

## 4 公式解析：性质在 PDE 求解中的组合拳

看这些性质如何组合成「解 PDE 的流水线」。热传导方程 $u_t = u_{xx}$ 的柯西问题：

- **第一步，变换整个方程。** 对 $x$ 变换，用线性 + 微分性质：$\widehat{u_t} = \partial_t\hat u$，$\widehat{u_{xx}} = -\omega^2\hat u$。得 $\hat u_t = -\omega^2\hat u$。
- **第二步，解 ODE。** 这是关于 $t$ 的一阶线性方程，解为 $\hat u(\omega,t) = \hat\varphi(\omega)e^{-\omega^2 t}$（用了初值 $\hat u(\omega,0) = \hat\varphi(\omega)$）。
- **第三步，反变换。** 需要 $e^{-\omega^2t}\hat\varphi(\omega)$ 的逆变换。**卷积定理（下一节）**说：乘法变卷积，$u = \varphi * \mathcal{F}^{-1}[e^{-\omega^2t}]$。
- **第四步，认出热核。** $\mathcal{F}^{-1}[e^{-\omega^2t}]$ 是高斯，即热核——回到泊松公式。

**整个流程只用了「微分变乘法」与「乘法变卷积」两条性质。** 这就是傅里叶方法「机械化」的全部秘密：求导的次数变成 $\omega$ 的幂，指数算子 $e^{t\partial_{xx}}$ 变成乘法 $e^{-t\omega^2}$——**把 PDE 的「微分算子」变成频域的「函数」**，一切难算的运算都变成了代数。

## 5 性质速查表

把本节性质收拢成一张表，作为后续求解的「手册」：

| 时域运算 | 频域结果 | 名称 |
| --- | --- | --- |
| $af + bg$ | $a\hat f + b\hat g$ | 线性 |
| $f(x-a)$ | $e^{-ia\omega}\hat f$ | 平移 |
| $e^{iax}f(x)$ | $\hat f(\omega-a)$ | 调制 |
| $f(ax)$ | $\frac{1}{|a|}\hat f(\omega/a)$ | 伸缩 |
| $f^{(n)}(x)$ | $(i\omega)^n\hat f$ | 微分 |
| $x^n f(x)$ | $i^n\hat f^{(n)}(\omega)$ | 乘幂 |
| $f * g$ | $\hat f\,\hat g$ | 卷积（下节） |

**「会查表」比「会推导」更快，但「会推导」才知道表怎么用。** 每条性质都从一个换元或分部积分出发，花一次功夫想透，之后就是机械调用。

**辨析｜易错点：** 微分性质要求 $f$ 在无穷远处衰减（分部积分的边界项为零）。对「不够好」的函数（如阶梯函数），$\mathcal{F}[f']$ 不等于 $i\omega\hat f$ 的古典版本——要修正边界项。实际 PDE 求解中初值通常落在好函数类里，但遇到间断初值时要警觉：间断处的贡献会让公式多出边界项（这就是为什么间断初值的变换要小心处理）。

## 6 小结

- 平移 ⇔ 相位（$e^{-ia\omega}$），伸缩 ⇔ 频域展缩，调制 ⇔ 频域搬移。
- 微分 ⇔ 乘 $i\omega$，$f'' \leftrightarrow -\omega^2\hat f$——PDE 代数化的核心。
- 乘 $x$ ⇔ 频域求导，与微分性质形成对偶。
- 实函数频谱共轭对称：$\hat f(-\omega) = \overline{\hat f(\omega)}$。
- 性质组合 = 变换解 PDE 的完整流水线（微分变乘法、乘法变卷积）。

在下一节，我们专讲卷积定理——时域卷积与频域乘法的等价。

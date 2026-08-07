---
title: Stokes 定理
date: 2026-08-07
---

# Stokes 定理

<div class="epigraph">
<p>边界上的积分等于内部求导的积分——这是微积分最美丽的一句话，从一维到任意维都成立。</p>
<footer>—— 乔治 · 加布里埃尔 · 斯托克斯（George Gabriel Stokes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§7.7 ｜ 2026-08-07</p>
</div>

## 为什么从 Stokes 定理开始

我们攒齐了全部零件：微分形式（被积对象）、外微分（求导）、定向与积分（测量）。现在把三者合成微积分学最伟大的定理——**Stokes 定理（Stokes' theorem）**

$$
\int_M d\omega = \int_{\partial M} \omega
$$

**「内部的求导积分 = 边界的原函数积分」。** 它一条公式同时覆盖：牛顿-莱布尼茨公式（1 维）、Green 公式（2 维平面）、Gauss 散度定理（3 维）、经典 Stokes 旋度定理（3 维曲面）——四大微积分公式的全统一。

Stokes 定理是现代分析的巅峰：**它揭示了「微分」（$d$）与「边界」（$\partial$）的对偶关系**——一个区域内部的求导总量，完全由它的边界决定。这条定理统治着从电磁学到微分几何的一切。<span class="marginnote">Stokes 定理的历史充满「命名误会」：它其实是 Kelvin 与 Stokes 在书信中讨论时系统化的，而「Green 定理」「Gauss 定理」「Newton-Leibniz」各自独立发现。Cartan 的外微分语言把四大公式统一成一条——「$\int_M d\omega = \int_{\partial M}\omega$」。「统一」是 20 世纪数学的核心主题，Stokes 定理是它的模范代表。</span>

## 1 定理的陈述

**定理（Stokes 定理）**：设 $M$ 是 $n$ 维**定向流形**（带边界），$\partial M$ 带**诱导定向**，$\omega$ 是 $M$ 上的 $n-1$ 阶紧支集微分形式。则

$$
\int_M d\omega = \int_{\partial M} \omega
$$

**重点：等式两端都是坐标无关的几何量，且不需要任何坐标选择。** 左边对 $\omega$ 求外微分后在整个 $M$ 上积分，右边直接在边界 $\partial M$ 上积分 $\omega$ 本身。<span class="marginnote">「带边界流形」（manifold with boundary）是 $M$ 的推广：局部同胚于 $\mathbb{R}^{n-1}\times[0,\infty)$（半空间），边界 $\partial M$ 是 $n-1$ 维流形。单位圆盘、立方体、球体都有边界；闭流形（球面、环面）边界为空——Stokes 定理在无边界情形退化为 $\int_M d\omega = 0$（重要特例）。</span>

## 2 四大经典公式的统一

Stokes 定理是微积分四大定理的「父母」：

| 定理 | $M$ | $\omega$ | Stokes 展开 |
| --- | --- | --- | --- |
| 牛顿-莱布尼茨 | 区间 $[a,b]$ | 0-形式 $f$ | $\int_a^b f'\,dx = f(b) - f(a)$ |
| Green 定理 | 平面区域 | 1-形式 $P\,dx + Q\,dy$ | $\iint (Q_x - P_y)\,dx\,dy = \oint P\,dx + Q\,dy$ |
| Gauss 散度定理 | 三维体 $V$ | 2-形式 | $\iiint \nabla\cdot\mathbf{F}\,dV = \iint_{\partial V}\mathbf{F}\cdot d\mathbf{S}$ |
| 经典 Stokes | 曲面 $S$ | 1-形式 | $\iint_S (\nabla\times\mathbf{F})\cdot d\mathbf{S} = \oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}$ |

**重点：牛顿-莱布尼茨是 1 维 Stokes；Gauss 散度是「$\omega$ 为 2-形式」的 3 维 Stokes。** 四条「独立」的定理，其实是同一条定理在不同维数与阶数下的投影。<span class="marginnote">记忆「四大公式」的现代方式：它们全是「$\int d\omega = \int_\partial \omega$」的特例。一旦接受 Stokes 定理，微积分课本里需要单独记忆的四章内容就合并成一章。这是「统一」带来的认知解放——少记三章，理解加深十倍。</span>

## 3 公式解析：为什么证明只需「检查立方体」

Stokes 定理的证明出奇地简单，因为它可以「局部化」。思路拆开：

- **第一步，化归到半空间**：用单位分解把 $\omega$ 拆到坐标卡上——只需证明「$\omega$ 支集在一个半空间坐标卡内」的情形。
- **第二步，假设支集在内部**：若 $\omega$ 的支集在 $M$ 内部（不碰边界），则 $\int_{\partial M}\omega = 0$（边界上 $\omega = 0$）；而 $\int_M d\omega$ 也 $= 0$（对紧支形式，内部积分是「分部积分」的边界项，支集在内部使边界项消失）。**两边都为零，平凡成立。**
- **第三步，支集碰边界**：假设 $\omega$ 支集在边界附近、坐标下 $\omega = f(x^1,\dots,x^{n-1})\,dx^1\wedge\cdots\wedge dx^{n-1}$（不含 $dx^n$）。则
  $$
  d\omega = \frac{\partial f}{\partial x^n}\,dx^n\wedge dx^1\wedge\cdots\wedge dx^{n-1}
  $$
  积分为 $\int \frac{\partial f}{\partial x^n}\,dx^n\,dx^1\cdots dx^{n-1}$——对 $x^n$ 积分恰是 $f(x^1,\dots,x^{n-1}, \text{边界})$（牛顿-莱布尼茨），正是右边的 $\int_{\partial M}\omega$。**一边是「牛顿-莱布尼茨」，另一边是「边界积分」——等式成立。**

**重点：Stokes 定理的证明核心是「分部积分」——把「内部求导」变成「边界值」。** 牛顿-莱布尼茨是它的 1 维种子，$n$ 维只是「逐变量分部」的重复。<span class="marginnote">「$d$ 是 $-\partial$ 的对偶」这个观点在更抽象的理论里被精致化：Stokes 定理说明「外微分」是「边界算子」的伴随。这正是 de Rham 上同调与奇异同调「对偶」（Poincaré 对偶）的雏形——「微分形式的 $d$」与「胞腔的 $\partial$」是一对镜像。</span>

## 4 例：验证 Stokes 定理

**例：单位圆盘上的 Green 定理。** $M$ = 单位圆盘，$\omega = x\,dy$（1-形式）。则

$$
d\omega = dx\wedge dy
$$

左边 $\int_M d\omega = \iint_{\text{disk}} dx\,dy = \pi$（面积）。右边 $\int_{\partial M} x\,dy$：边界是单位圆 $(\cos\theta,\sin\theta)$，$x\,dy = \cos\theta\,d(\sin\theta) = \cos^2\theta\,d\theta$，积分 $\int_0^{2\pi}\cos^2\theta\,d\theta = \pi$。**左右相等——验证通过。**

**例：无边界流形。** $M$ 是闭曲面（如 $S^2$），$\partial M = \emptyset$，Stokes 给出 $\int_{S^2} d\omega = 0$ 对任何 1-形式 $\omega$。由 Gauss-Bonnet 的角度，$\iint K\,dA = 4\pi$ 且 $K\,dA = d(\cdots)$（局部）——**但整体上 $K\,dA$ 不是精确形式**（积分不为零），这正是「闭而不精确」的实例（de Rham 上同调）。

## 5 Stokes 定理的地位

Stokes 定理是微分形式理论的王冠：

- **物理定律**：Maxwell 方程的积分形式、Gauss 定律、Faraday 定律全是 Stokes 的应用。
- **de Rham 上同调**：Stokes 保证「$d$ 与 $\partial$ 对偶」，上同调因此良定义。
- **Gauss-Bonnet 的证明**：曲率形式积分 = 欧拉类积分（第五篇的现代证明路径）。
- **偏微分方程**：分部积分（弱形式、变分法）全是 Stokes 在算子上的翻译。<span class="marginnote">在变分法与有限元里，「分部积分」（把 $\int \nabla u\cdot\nabla v$ 换成边界项）就是 Stokes 定理——这是偏微分方程弱解理论的基石（第二级《偏微分方程》会展开）。而在机器学习里，「梯度下降 + 边界条件」也以 Stokes/散度定理为后台。「积分与边界」的对偶，从微积分课本一路统治到现代计算。</span>

**重点：Stokes 定理是「微分与边界」的精确对偶——一维的牛顿-莱布尼茨是它的种子，任意维的微积分是它的花。** 它是本专题从曲线论到微分形式的高潮之一。

### 例：三维中的「边界为空的流形」

重要的特例：无边界流形。设 $M$ 是闭曲面（如 $S^2$），$\partial M = \emptyset$。Stokes 定理给出

$$
\int_M d\omega = \int_{\emptyset}\omega = 0
$$

对任何 $k$-形式 $\omega$（$k = \dim M - 1$）。

**重点：无边界流形上，任何「精确形式」的积分为零。** 这个看似平凡的结果是 de Rham 上同调的核心：闭形式 $\eta$（$d\eta = 0$）可以积分不为零（如球面的面积形式 $\iint K\,dA = 4\pi$），但它不是精确形式（否则积分为零）——「闭而不精确」的区域就是上同调。「边界为空 ⟹ 精确形式积分零」这一条，是整个上同调理论的起点。

### Stokes 定理的「哲学读法」

Stokes 定理最深刻的哲学读法：**「边界是内部的镜像」——一个区域的积分行为完全由它的边界决定。** 这不是巧合，而是「微分」与「边界」互为对偶的必然：$d$ 与 $\partial$ 是「同一个结构的两个面」。

**重点：Stokes 定理揭示了「内部 ↔ 边界」的对偶——这是从微积分到拓扑的普适法则。** 在物理里，「通量 = 源的积分」是守恒律；在几何里，「曲率积分 = 边界转角」是 Gauss-Bonnet；在拓扑里，「上同调 = 同调的对偶」是 Poincaré 对偶。**「边界决定内部」这一句话，贯穿数学与物理两百年。**

### 从 Stokes 看「守恒」

物理里的守恒律全是 Stokes 的化身：**「内部的变化 = 边界的通量」。** 质量守恒 $\frac{d}{dt}\int_V\rho = -\int_{\partial V}\mathbf{J}\cdot d\mathbf{S}$（密度变化 = 流出量）、电荷守恒、能量守恒——全是「$\int_M d\omega = \int_{\partial M}\omega$」的物理翻译。

**「守恒 = 边界上的进出 = 内部的源」——Stokes 定理是守恒律的数学骨架。** 这解释了为什么「散度定理」是流体力学、电磁学的核心工具：「源」与「流」在 Stokes 里统一。「守恒律从微积分的基本定理长出来」——这是物理与数学最深刻的共生之一。

## 6 小结

- **Stokes 定理**：$\int_M d\omega = \int_{\partial M}\omega$——内部求导积分 = 边界原函数积分。
- 覆盖四大公式：牛顿-莱布尼茨、Green、Gauss 散度、经典 Stokes。
- 证明：单位分解 + 化归半空间 + 牛顿-莱布尼茨（分部积分）。
- 无边界流形：$\int_M d\omega = 0$（闭而不精确的反例）。
- 地位：物理定律、de Rham 上同调、Gauss-Bonnet 证明、PDE 弱形式的地基。

在下一节，我们用 Stokes 定理重看经典公式：**从 Stokes 定理看 Green 公式、Gauss 公式与 Gauss-Bonnet**——统一视角下的四大经典。

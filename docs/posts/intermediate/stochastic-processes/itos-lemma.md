---
title: 伊藤公式（Itô's Lemma）：单变量形式
date: 2026-08-07
---

# 伊藤公式（Itô's Lemma）：单变量形式

<div class="epigraph">
<p>微积分的链式法则在随机世界里多出一项——而那多余的一项，恰是波动本身的价格。</p>
<footer>—— 伊藤清（Kiyosi Itô）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§8.3 ｜ 2026-08-07</p>
</div>

## 随机版本的链式法则

普通微积分里，复合函数求导是链式法则：$dg(B(t)) = g'(B(t)) dB(t)$。但布朗运动的 $dB$ 不是「真正的微分」——它是 $\sqrt{dt}$ 量级的东西，平方不可忽略。于是 Itô 公式在链式法则后多出一项：

$$
dg\big(B(t)\big) = g'\big(B(t)\big)\, dB(t) + \frac{1}{2} g''\big(B(t)\big)\, dt.
$$

**这就是 Itô 公式（Itô's Lemma）**——随机微积分的核心定理，Black-Scholes 公式的心脏。它告诉我们：**在随机世界里，「泰勒展开」不能只到一阶，二阶项 $g''\,(dB)^2$ 因为 $(dB)^2 = dt$ 而存活**。普通微积分丢掉二阶项，随机微积分必须保留它——这就是「随机」与「确定」的全部差别。<span class="marginnote">一句话版：<strong>Itô 公式 = 普通链式法则 + 半个二阶导数项</strong>。那半个 $g''dt$ 就是波动率的「定价」——越凸的函数（$g''$ 大），受波动影响越大，多出来的项越大。</span>

本节目标：陈述并推导单变量 Itô 公式、给出 Itô 过程的推广形式、并用它算两个经典例子。

## 1 Itô 公式（布朗运动情形）

**Itô 公式（单变量）**：设 $g \in C^2$，则
$$
g\big(B(t)\big) = g(B(0)) + \int_0^t g'\big(B(s)\big)\, dB(s) + \frac12 \int_0^t g''\big(B(s)\big)\, ds.
$$
**微分形式**：
$$
dg(B) = g'(B)\, dB + \frac12 g''(B)\, dt.
$$

**推导直觉（泰勒展开）**：把 $g(B + \Delta B)$ 展开到二阶：
$$
g(B + \Delta B) = g(B) + g'(B)\Delta B + \frac12 g''(B)(\Delta B)^2 + \cdots
$$
取 $\Delta t$ 极小：$\Delta B \sim \sqrt{\Delta t}$ 量级，$(\Delta B)^2 \approx \Delta t$（二次变差），更高阶 $(\Delta B)^3 \sim (\Delta t)^{3/2}$ 可忽略。于是
$$
\Delta g \approx g'\Delta B + \frac12 g'' \Delta t.
$$
**把 $(dB)^2 = dt$ 直接代入泰勒展开，就是 Itô 公式。**<span class="marginnote">「$(dB)^2 = dt$ 代入泰勒」是 Itô 公式的全部秘密：<strong>普通泰勒到二阶时忽略 $(\Delta B)^2$（因为光滑情形它 $\sim (\Delta t)^2$），布朗世界它 $\sim \Delta t$ 不能丢</strong>。记住这个口诀，Itô 公式就是「带二阶项的泰勒」。</span>

## 2 Itô 过程的 Itô 公式

**Itô 过程（Itô process）**：形如
$$
dX(t) = \mu(t)\, dt + \sigma(t)\, dB(t),
$$
即 $X(t) = X(0) + \int_0^t \mu\, ds + \int_0^t \sigma\, dB$。$\mu$ 是漂移（$dt$ 项），$\sigma$ 是扩散（$dB$ 项）。

**Itô 公式（Itô 过程版）**：对 $g \in C^2$，
$$
dg\big(X(t)\big) = g'(X)\, dX + \frac12 g''(X)\, (dX)^2,
$$
其中 $(dX)^2 = \sigma^2 dt$（因为 $(dt)^2 = 0$、$dt\,dB = 0$、$(dB)^2 = dt$）。展开：
$$
dg(X) = \Big( g'(X)\mu + \frac12 g''(X)\sigma^2 \Big) dt + g'(X)\, \sigma\, dB.
$$
**漂移被「凸性修正」$\frac12 g''\sigma^2$ 增强，扩散被 $g'$ 缩放。** 这就是 SDE 变换的标准公式。<span class="marginnote">$(dX)^2 = \sigma^2 dt$ 的计算：<strong>$(dX)^2 = (\mu dt + \sigma dB)^2 = \sigma^2 (dB)^2 + 2\mu\sigma\, dt\,dB + \mu^2(dt)^2 = \sigma^2 dt$</strong>（交叉项与高阶项全归零）。「$dB$ 项平方成 $dt$，其余全消失」——这个口诀算 $(dX)^2$ 永远够用。</span>

## 3 公式解析：验证 Itô 公式 g(B) = B²

**目标：用 Itô 公式重新导出「$B(t)^2 = 2\int_0^t B\,dB + t$」，并与前面的代数推导对照。**

第一步，取 $g(x) = x^2$。则 $g'(x) = 2x$，$g''(x) = 2$。

第二步，代入 Itô 公式：
$$
d(B^2) = 2B\, dB + \frac12 \cdot 2\, dt = 2B\, dB + dt.
$$
第三步，积分形式：
$$
B(t)^2 = B(0)^2 + \int_0^t 2B\, dB + \int_0^t dt = 2\int_0^t B\,dB + t.
$$
第四步，对照前面的代数。这正是「$\int_0^1 B\,dB = (B(1)^2 - 1)/2$」的来源——**Itô 公式把「相差 1」从代数巧合升级为系统定理**。

**这个推导为什么重要**：$B^2$ 的例子把「$(dB)^2 = dt$」变成可操作的计算——**Itô 公式不是玄学，是「泰勒 + 二次变差」的必然**。同样的例子将直接通向随机微分方程的求解（第八篇末）。

## 4 应用：几何布朗运动的 Itô 视角

取 $S(t) = S_0 e^{(\mu - \sigma^2/2)t + \sigma B(t)}$（GBM）。设 $g(x) = S_0 e^x$，$X(t) = (\mu - \sigma^2/2)t + \sigma B(t)$，则 $dX = (\mu - \sigma^2/2)dt + \sigma dB$。

由 Itô 公式（$g'(x) = g''(x) = S_0 e^x$）：
$$
dS = S\Big[ (\mu - \frac{\sigma^2}{2}) dt + \sigma dB \Big] + \frac12 S \cdot \sigma^2 dt = S\big( \mu\, dt + \sigma\, dB \big).
$$
**结论**：GBM 满足随机微分方程
$$
\frac{dS}{S} = \mu\, dt + \sigma\, dB.
$$
**这个方程是 Black-Scholes 的出发点和第七节变体的「方程身份」**。注意那个 $-\sigma^2/2$ 在方程里消失了——它被 Itô 公式的 $\frac12 g''\sigma^2$ 修正「吃掉」了。<span class="marginnote">「$-\sigma^2/2$ 被 Itô 修正吃掉」是理解 GBM 的钥匙：<strong>定义里故意写 $(\mu - \sigma^2/2)$，正是为了让微分方程 $dS/S = \mu dt + \sigma dB$ 干净</strong>。Itô 公式在两个视角之间无缝切换——指数形式的定义与微分形式的方程，是同一个对象。</span>

## 5 Itô vs 普通微积分的对照

| 情形 | 链式法则 | 二阶项 |
| --- | --- | --- |
| 普通函数 $g(f(t))$ | $g' df$ | 忽略（$df^2 \sim dt^2$） |
| Itô $g(B(t))$ | $g' dB$ | **保留 $\frac12 g'' dt$**（$dB^2 = dt$） |

**唯一的差别就是二阶项是否保留。** 记住这个对照，Itô 公式永远不失手。<span class="marginnote">工程里的体现：<strong>金融风险管理的「凸性修正」（convexity）正是 $\frac12 g''\sigma^2$</strong>；随机梯度下降的「噪声诱导的漂移」也是 Itô 修正的离散版。「凸函数遇波动则上偏」是 Itô 公式放之四海的直觉。</span>

## 6 小结

- **Itô 公式**：$dg(B) = g'(B)dB + \frac12 g''(B)dt$——带 $(dB)^2 = dt$ 的泰勒。
- **Itô 过程** $dX = \mu dt + \sigma dB$ 的变换：$dg(X) = (g'\mu + \frac12 g''\sigma^2)dt + g'\sigma dB$。
- **$(dX)^2 = \sigma^2 dt$**：$dB$ 平方成 $dt$，其余归零。
- $B^2$ 例子：$B^2 = 2\int B\,dB + t$；GBM 例子：$dS/S = \mu dt + \sigma dB$。
- 与普通微积分的差别只有二阶项——凸性修正是 Itô 的灵魂。

**例（$g(B) = e^{\theta B}$，指数鞅的诞生）**：取 $g(x) = e^{\theta x}$，则 $g' = \theta e^{\theta x}$、$g'' = \theta^2 e^{\theta x}$。Itô 公式给出
$$
d(e^{\theta B}) = \theta e^{\theta B} dB + \frac{\theta^2}{2} e^{\theta B} dt.
$$
整理成积分形式：
$$
e^{\theta B(t) - \frac{\theta^2}{2}t} = 1 + \theta \int_0^t e^{\theta B(s) - \frac{\theta^2}{2}s} dB(s).
$$
左边正是第七节的**指数鞅** $\mathcal{E}_\theta(t)$——**Itô 公式直接「造出」指数鞅**：当 $g''$ 项被一个确定的 $dt$ 补偿（这里的 $-\theta^2 t/2$）时，$e^g$ 型过程就变成鞅。这个推导是 Girsanov 测度变换（第十篇）与期权定价的全部入口：鞅化的关键不是靠运气，而是靠 Itô 公式的修正项。

**再一个核对（$g(B) = B^3$）**：$g' = 3B^2$、$g'' = 6B$，Itô 公式给 $dB^3 = 3B^2 dB + 3B\,dt$——多出的 $3B\,dt$ 正是普通微积分 $dB^3 = 3B^2dB$ 没有的「随机修正」。阶数越高，修正越复杂；这也解释了为什么 Itô 公式对多项式、指数、对数都成立——它不是某个特殊函数的故事，而是随机微积分的完整法则，对一切 $C^2$ 函数生效。

用它还能算矩：$E[B(t)^3] = E\big[3\int_0^t B^2 dB + 3\int_0^t B\,ds\big] = 3\int_0^t E[B(s)]ds = 0$——Itô 积分期望归零、Fubini 换序，三阶矩恒为零，与 $N(0,t)$ 的奇阶矩为零吻合。这条「用 Itô 公式算矩」的路线，把随机微积分与分布计算直接打通。

在下一节，我们把 Itô 公式推广到多个变量：**多维伊藤公式与乘积法则**——两个随机过程的乘积怎么求导。

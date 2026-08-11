---
title: Itô 公式与 Lévy 特征
date: 2026-08-11
---

# Itô 公式与 Lévy 特征

<div class="epigraph">
<p>随机世界的链式法则少不了一个额外的项——那是布朗运动留下的二阶脚印。</p>
<footer>—— 伊藤清（Kiyosi Itô）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 随机分析（Itô 微积分） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Itô 公式开始

上一节我们定义了随机积分，并用手算出了唯一一条积分：$\int B\,dB = (B^2 - t)/2$。但它更像「运气好」。真正的随机分析需要一个能处理**一切光滑函数**的机器：给定一个扩散 $X$，如何求 $f(X_t)$ 的微分？这就是 **Itô 公式**——随机版本的链式法则。

普通链式法则 $df(x) = f'(x)\,dx$ 之所以在随机世界失效，是因为 $dx$ 的平方不再可以忽略。我们把「微积分 1」的直觉（二阶小量忽略不计）升级为「微积分 2」的纪律（二阶小量精确等于 $dt$），一切就通了。

同时这一篇还要回答一个更深刻的问题：**怎么才能「只凭二次变差」认出布朗运动？** 答案叫 **Lévy 特征定理**——它把布朗运动的定义从「有限维分布」切换成「轨道的二阶量」，这将是下一节 Girsanov 测度变换的跳板。<span class="marginnote">本篇对标 Protter《Stochastic Integration and Differential Equations》第二章与 Karatzas &amp; Shreve 第三章。<strong>Itô 公式是随机分析的计算引擎，Lévy 特征是它的判官。</strong></span>

## 1 一维 Itô 公式

**Itô 公式（Itô's formula）**：设 $B$ 是布朗运动，$f \in C^2$，则

$$f(B_t) = f(B_0) + \int_0^t f'(B_s)\,dB_s + \frac12 \int_0^t f''(B_s)\,ds.$$

微分写法更醒目：

$$df(B_t) = f'(B_t)\,dB_t + \frac12 f''(B_t)\,dt.$$

**重点：随机链式法则 = 普通链式法则 + 半倍二阶修正。** 这个修正项从哪里来？把 $f$ 做二阶泰勒展开：$df = f'\,dB + \tfrac12 f''\,(dB)^2 + \cdots$，而 $(dB)^2 = dt$（布朗运动二次变差的微观形态），于是多出 $+\tfrac12 f''\,dt$。<span class="marginnote">泰勒展开在高阶项上止步：$(dB)^3$、$(dB)\,dt$ 等都是 $o(dt)$，取极限后消失。<strong>随机微积分只留到二阶，因为布朗运动恰好「粗糙到二阶」。</strong></span>

**辨析｜易错点：** Itô 公式里的积分是 Itô 积分，**不是**普通积分。公式左边是「我们想要的量」，右边是「我们算得动的量」——Itô 公式从不声称 $df$ 本身有意义，它说的是两边积分相等。

## 2 公式解析：$\int B\,dB = \frac{B^2 - t}{2}$ 的完整推演

用 Itô 公式可以重算上一节的手工结果，这次不靠运气。取 $f(x) = x^2$，$f'(x) = 2x$，$f''(x) = 2$：

- **第一步，代入公式**：$d(B^2) = 2B\,dB + \tfrac12 \cdot 2\,dt = 2B\,dB + dt$。
- **第二步，改写**：$B\,dB = \tfrac12 d(B^2) - \tfrac12 dt$。
- **第三步，两边积分**：$\int_0^t B\,dB = \tfrac12 (B_t^2 - B_0^2) - \tfrac12 t = \tfrac12(B_t^2 - t)$。

**这条式子的全部意义在于：Itô 公式不是「新假设」，而是「二次变差的定价」。** 普通微积分免费送你的 $\int x\,dx = x^2/2$，在随机世界要花掉一笔 $t/2$ 的「二阶账」。金融里把这类修正叫做**凸性修正**——标的波动越剧烈，修正越明显。

## 3 半鞅版本与多维版本

Itô 公式不止服务于布朗运动。若 $X$ 是**半鞅**（局部鞅 + 有界变差项，见本专题《局部鞅与半鞅理论》），同样有

$$d f(X_t) = f'(X_t)\,dX_t + \frac12 f''(X_t)\,d\langle X\rangle_t.$$

对两个过程 $X, Y$，还有更一般的形式（乘积公式）

$$d(X_t Y_t) = X_t\,dY_t + Y_t\,dX_t + d\langle X, Y\rangle_t,$$

其中 $\langle X,Y\rangle$ 是交叉二次变差。**$XY$ 的随机微分比普通乘积公式多出一项交叉二次变差**——这是鞅与鞅相乘时「二阶纠缠」的体现。<span class="marginnote">若 $X = Y = B$，交叉项退化为 $d\langle B,B\rangle = dt$，乘积公式就给出 $d(B^2) = 2B\,dB + dt$——<strong>与上一节完全自洽</strong>。</span>

## 4 Lévy 特征定理：用二次变差当身份证

布朗运动的原始定义依赖有限维分布（正态增量），那是一条很「重」的路。Lévy 找到了一条轻得多的路：

**Lévy 特征定理（Lévy's characterization）**：设 $X$ 是连续局部鞅，$X_0 = 0$，且 $\langle X, X\rangle_t = t$，则 $X$ 是标准布朗运动。

**重点：一个连续局部鞅只要「二次变差等于 $t$」，就自动是布朗运动——不需要检查任何正态分布。** 为什么？可以用指数鞅快速看到线索：对连续局部鞅 $M$，$Z_t^\theta = \exp\big(\theta M_t - \tfrac{\theta^2}{2} t\big)$ 是局部鞅；若 $\langle M\rangle_t = t$，则由 Itô 公式可证 $Z^\theta$ 还是鞅，而 $Z^\theta$ 的期望恒为 1，这正是正态分布的特征函数 $\exp(\theta^2 t/2)$——于是增量分布被二次变差「顶」成了正态。<span class="marginnote">这是一条几乎反直觉的定理：你从不检查「增量高不高斯」，只检查「二阶量给不给力」。<strong>Lévy 把布朗运动的本质还原成了「二阶变差守恒」。</strong></span>

**辨析｜易错点：** 条件是「二次变差 $= t$」，不是「二次变差 $\to t$」。差之毫厘：等号意味着整个过程的时间参数被「校准」了，这对后面的测度变换至关重要。

## 5 Lévy 特征的第一个果实：识别「布朗运动的等价物」

Lévy 特征定理最有威力的用法，是**换一个角度看同一个对象**。举例：对常数 $\sigma > 0$，$X_t = \sigma B_t$ 不是标准布朗运动，因为 $\langle X,X\rangle_t = \sigma^2 t \ne t$。要让二次变差回到 $t$，要么让振幅回到 1，要么**重新定义时间**——这正是 Dambis–Dubins–Schwarz 定理的思路：任何连续局部鞅都能被自己的二次变差「重新参数化」成布朗运动。

这个「把任意连续局部鞅拉到布朗运动的标准时钟上」的视角，是下一节 Girsanov 测度变换（在**不改变二次变差**的前提下改变漂移）的理论基础。<span class="marginnote">对比记忆：Lévy 特征改「形状」不改「时钟」（二次变差固定为 $t$），Girsanov 改「漂移」不改「二阶量」。<strong>两条路都在追问：什么量在变换下不变？</strong></span>

## 6 多维 Itô 公式与一次实战

设 $X_t = (X^1_t, \dots, X^d_t)$ 是 $d$ 维半鞅，$f \in C^2(\mathbb{R}^d)$，则

$$df(X_t) = \sum_{i=1}^d \frac{\partial f}{\partial x_i}(X_t)\,dX^i_t + \frac12 \sum_{i,j=1}^d \frac{\partial^2 f}{\partial x_i \partial x_j}(X_t)\,d\langle X^i, X^j\rangle_t.$$

多维公式看起来吓人，但结构清晰：**一阶项照抄普通链式法则，二阶项则遍历所有（含交叉）二次变差**——这正是 Itô 公式「二阶账」的完整清单。

**实战：验证 $S_t = S_0 e^{(\mu - \sigma^2/2)t + \sigma B_t}$ 是几何布朗运动的解。** 设 $Z_t = (\mu - \sigma^2/2)t + \sigma B_t$，$f(x) = e^x$：

- 一阶项：$f'(Z)\,dZ = e^Z\big((\mu - \sigma^2/2)dt + \sigma dB\big)$；
- 二阶项：$\tfrac12 f''(Z)\,d\langle Z\rangle = \tfrac12 e^Z \sigma^2 dt$；
- 相加：$dS = e^Z\big[(\mu - \sigma^2/2)dt + \sigma dB + \tfrac12\sigma^2 dt\big] = S\big(\mu\,dt + \sigma\,dB\big)$。

**瞧，二阶项的 $\tfrac12\sigma^2$ 精确抵消了漂移里扣除的 $\tfrac12\sigma^2$——Itô 公式自己验证了「对数解」的合法性。** 这类「先设对数、再用公式检验」的套路，是求解扩散方程最常用的双手。

## 7 Stratonovich：另一笔二阶账

上一节提到 Stratonovich 积分保留链式法则。用 Itô 公式的语言说：对 Itô SDE $dX = b\,dt + \sigma\,dB$，对应的 Stratonovich 形式是 $dX = \tilde b\,dt + \sigma \circ dB$，其中

$$\tilde b(x) = b(x) - \frac12 \sigma'(x)\sigma(x).$$

**重点：同一过程，Itô 记法多一项「半倍导数修正」，Stratonovich 记法不收这笔账。** 两者描述的是同一个随机过程，只是记账科目不同——物理学家爱用 Stratonovich（因为不破坏链式法则），金融与鞅论爱用 Itô（因为保鞅性）。**记住：Itô 与 Stratonovich 的分歧不是「谁对」，而是「二阶账记在谁头上」。**

（想亲手再练一题：用 Itô 公式求 $d(\sin B_t)$ 与 $d(B_t^3)$，会分别长出 $-\tfrac12 \sin B_t\,dt$ 与 $3B_t\,dt$ 两笔二阶账——试试它们如何被「$+\tfrac12 f''\,dt$」记账。）

## 8 例：随机指数就是 Itô 公式的自证

设 $Z_t = e^{B_t - t/2}$（我们在《局部鞅与半鞅理论》还会再见它）。对它用 Itô 公式：

- 一阶项：$Z\,d(B - \tfrac12 t) = Z(dB - \tfrac12 dt)$；
- 二阶项：$\tfrac12 Z\,d\langle B\rangle = \tfrac12 Z\,dt$；
- 相加：$dZ = Z\,dB - \tfrac12 Z\,dt + \tfrac12 Z\,dt = Z\,dB$。

**漂移项与二阶项精确对消，$Z$ 只剩「纯噪声」——它是局部鞅（且是非鞅的经典例子）。** 这个 $-t/2$ 修正的几何意义：$B_t$ 的典型尺度是 $\sqrt t$，它的指数 $e^{B_t}$ 期望约为 $e^{t/2}$，扣掉 $t/2$ 恰好让期望归一。**Itô 公式在这里扮演的角色，是把「指数的凸性」翻译成「漂移的修正」**——随机指数是两者对账的活证据，也是下一节 Girsanov 密度的原型。<span class="marginnote">这也是一种「量纲感」训练：<strong>$dB$ 的量纲是 $\sqrt{dt}$，所以 $e^{B}$ 里必须配一个 $dt$ 量的修正，量纲才自洽</strong>——二阶账处处在给量纲把关。</span>

## 9 小结

- **Itô 公式** $df(X_t) = f'(X_t)\,dX_t + \tfrac12 f''(X_t)\,d\langle X\rangle_t$：随机链式法则 = 普通链式法则 + 二阶修正。
- 修正项来自 $(dB)^2 = dt$ 的二次变差定价；它使得 $\int B\,dB = (B^2 - t)/2$。
- 乘积公式 $d(XY) = X\,dY + Y\,dX + d\langle X,Y\rangle$ 多出一项交叉二次变差。
- **Lévy 特征定理**：连续局部鞅 $X_0 = 0$ 且 $\langle X,X\rangle_t = t$ 时，$X$ 自动是布朗运动。
- Lévy 特征把布朗运动的身份从「分布」切换为「二阶量」，是测度变换时代的通行证。

在下一节，我们将让 Itô 公式登场扮演它最核心的角色：从「已知过程的微分」到「未知过程的定义」——**Itô 随机微分方程**。

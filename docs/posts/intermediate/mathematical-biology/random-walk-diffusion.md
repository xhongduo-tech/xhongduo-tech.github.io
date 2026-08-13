---
title: 随机游走与扩散方程：个体运动的空间建模
date: 2026-08-07
---

# 随机游走与扩散方程：个体运动的空间建模

<div class="epigraph">
<p>醉汉找路灯下的钥匙，因为那里有光。</p>
<footer>—— 概率论经典笑话</footer>
</div>

<div class="article-byline">
<p>第二级 · 生物数学 ｜ Edelstein-Keshet《Mathematical Models in Biology》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从随机游走讲起

前面所有模型都假设种群「没有位置」——一个物种的数量是全局一个数。但真实生物生活在空间里：细菌在培养皿上蔓延、麋鹿在森林里扩散、肿瘤细胞向周围组织浸润。空间如何进入方程？答案藏在**随机游走（random walk）**里：无数个体的随机运动，在大尺度上涌现出确定性的**扩散方程**。这一节是第 4 篇的开门课——我们从「单个个体随机瞎走」出发，推导出「整个种群确定扩散」的偏微分方程。这个「从微观随机到宏观确定」的跃迁，是统计力学与生物数学共同的方法论核心。

## 1 一维随机游走：微观规则

考虑一维直线上的随机游走。设一个粒子从原点出发，每个时间步 $\Delta t$ 以概率 $1/2$ 向左、$1/2$ 向右移动距离 $\Delta x$。用 $P(x, t)$ 表示在时刻 $t$ 处于位置 $x$ 的概率。

每一步的规则可以写成递推：

$$P(x, t + \Delta t) = \frac{1}{2}P(x - \Delta x, t) + \frac{1}{2}P(x + \Delta x, t)$$

粒子要么从左边跳来，要么从右边跳来，各占一半。<span class="marginnote">这是「主方程（master equation）」的一维形式：新状态的概率 = 从各邻居流入概率之和。注意没有「停留」项——每步必须动，左右等概率。若左右概率不等（有偏向的随机游走），方程里就会多出一个「漂移」项，对应生物里的趋化/风漂流。</span>

**关键推导**：对上式做泰勒展开（$\Delta t$、$\Delta x$ 都很小）：

$$P + \Delta t\, \frac{\partial P}{\partial t} \approx \frac{1}{2}\left[P - \Delta x\frac{\partial P}{\partial x} + \frac{\Delta x^2}{2}\frac{\partial^2 P}{\partial x^2}\right] + \frac{1}{2}\left[P + \Delta x\frac{\partial P}{\partial x} + \frac{\Delta x^2}{2}\frac{\partial^2 P}{\partial x^2}\right]$$

一阶项抵消，二阶项保留，得**扩散方程**

$$\frac{\partial P}{\partial t} = D\,\frac{\partial^2 P}{\partial x^2}, \qquad D = \frac{(\Delta x)^2}{2\,\Delta t}$$

**扩散系数 $D$ 的量纲是 $[\text{长度}^2/\text{时间}]$**——不是速度，而是「平方距离除以时间」。

## 2 扩散方程的解读：均值不变，方差线性增长

扩散方程 $\partial P/\partial t = D\,\partial^2 P/\partial x^2$ 的解（初值 $\delta(x)$）是**高斯分布**：

$$P(x, t) = \frac{1}{\sqrt{4\pi D t}}\, e^{-x^2/(4Dt)}$$

这个解揭示扩散的三个核心特征：

- **均值不变**：$\langle x \rangle = 0$——随机游走没有净位移（无偏向时）。
- **方差线性增长**：$\langle x^2 \rangle = 2Dt$——扩散的范围随时间**线性**增长，标准差 $\propto \sqrt{t}$。
- **扩散是「二阶」过程**：$x \sim \sqrt{t}$ 而非 $x \sim t$。直线运动 $x \sim t$ 是「弹道式」的，随机游走只有「平方根式」的进展。<span class="marginnote">「$\sqrt{t}$ 定律」是整个扩散理论的基石：分子扩散一个距离 $L$ 需要的时间 $\sim L^2/D$。细菌扩散 1 mm 只需秒级，而扩散 1 m 要 $10^4$ 倍时间——<strong>扩散在短距离高效、长距离极慢</strong>。这解释了为什么生物体靠主动运输而非被动扩散来长距离输送物质，也解释了为什么扩散斑图总是「边缘扩散快、核心慢」。</span>

**辨析｜易错点：** $P(x,t)$ 在 $t \to 0$ 时趋于 $\delta$ 函数（全部概率集中一点），不是有限高度的尖峰。许多初学者把初值写成「常数在原点附近」，得到错误的解。用 $\delta$ 函数作初值是扩散方程的标准手法，它与「瞬时点释放」的物理情形一一对应。

## 3 从概率到种群密度：连续化

把「单个粒子的概率分布」升级为「大量个体的密度分布」。设种群密度 $u(x, t)$（单位长度的个体数），每只个体独立地作随机游走，则**宏观密度也满足同一个扩散方程**：

$$\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial x^2}$$

推导的关键一步：密度只是概率分布乘以总个体数，而扩散方程是**线性**的——线性叠加原理保证「多粒子 = 单粒子的叠加」。<span class="marginnote">扩散方程的线性是它易解的根本原因：整体解是单个点源解（Green 函数）的叠加。任何初始分布都可以写成 $\delta$ 函数的叠加，解就是「初始密度与高斯核的卷积」。这种「线性 + 叠加」结构在后面加进非线性反应项时会被打破——但那时我们仍有 Fisher-KPP 的深刻理论（下一节）。</span>

对二维三维，扩散方程推广为

$$\frac{\partial u}{\partial t} = D\,\nabla^2 u$$

其中 $\nabla^2 = \partial^2/\partial x^2 + \partial^2/\partial y^2$ 是**拉普拉斯算子**。<span class="marginnote">拉普拉斯算子的生物学解读：它度量「局部浓度相对于邻居的平均」的偏离。$\nabla^2 u > 0$ 意味着该点的密度低于周围平均，净流入补差；$\nabla^2 u \lt  0$ 则净流出。扩散天然地<strong>抹平不均匀</strong>——它是空间里最强大的「平均化」力量，而下一节的反应项则试图制造不均匀，两者对抗正是斑图形成的源泉。</span>

## 4 公式解析：从随机游走递推到扩散方程

完整走一遍「微观规则 → 宏观方程」的推导，这是本节最核心的公式链。

$$
P(x, t + \Delta t) = \frac{1}{2}P(x - \Delta x, t) + \frac{1}{2}P(x + \Delta x, t)
$$

- **第一步，泰勒展开左边**（对 $t$）：

$$
P(x, t+\Delta t) = P + \Delta t \frac{\partial P}{\partial t} + O(\Delta t^2)
$$

- **第二步，泰勒展开右边**（对 $x$，各到二阶）：

$$
P(x\pm\Delta x, t) = P \pm \Delta x \frac{\partial P}{\partial x} + \frac{\Delta x^2}{2}\frac{\partial^2 P}{\partial x^2} + O(\Delta x^3)
$$

- **第三步，代回并抵消**：左边代入，右边取平均：

$$
P + \Delta t\frac{\partial P}{\partial t} = P + \frac{\Delta x^2}{2}\frac{\partial^2 P}{\partial x^2}
$$

一阶项 $\pm \Delta x \partial P/\partial x$ 在左右两项相加时**互相抵消**——这正是无偏向随机游走的标志。

- **第四步，取极限**：令 $\Delta t \to 0$、$\Delta x \to 0$，保持 $\frac{(\Delta x)^2}{2\Delta t} = D$ 有限，得

$$
\frac{\partial P}{\partial t} = D\frac{\partial^2 P}{\partial x^2}
$$

<span class="marginnote">注意极限的取法：$\Delta x$ 与 $\Delta t$ 不能独立地任意趋零，必须保持 $\Delta x^2/\Delta t$ 收敛到固定常数 $D$。这是扩散极限的「缩放标度」——若 $\Delta x/\Delta t$ 固定（弹道极限），得到的是波动方程而非扩散方程。<strong>同一套微观规则，不同的缩放，不同的宏观方程</strong>——这个教训在多个尺度建模里反复出现。</span>

**这条推导链的要点**：一阶项抵消（无偏向）、二阶项保留（随机性）、$D = \Delta x^2/(2\Delta t)$ 定义扩散系数。它把「微观的随机瞎走」与「宏观的确定扩散」精确地连了起来——随机与确定之间没有鸿沟，只有尺度。

## 5 有偏向的随机游走：从扩散到平流

真实生物的运动很少完全无偏向——细菌向着食物游、鱼逆流而上、细胞被流体冲走。把随机游走加上**偏向**，扩散方程就会多出一项。

设粒子每步以概率 $p$ 向右、$q$ 向左（$p + q = 1$，$p \ne q$）。重复泰勒展开，一阶项不再抵消：

$$\frac{\partial u}{\partial t} + v\,\frac{\partial u}{\partial x} = D\,\frac{\partial^2 u}{\partial x^2}$$

其中

$$v = \lim_{\Delta t \to 0} \frac{(p - q)\Delta x}{\Delta t}, \qquad D = \lim_{\Delta t \to 0} \frac{(\Delta x)^2}{2\Delta t}$$

新出现的是**平流项** $v\,\partial u/\partial x$——$v$ 是**漂移速度（drift velocity）**，代表整体「被带着走」的净速度。当 $p = q$ 时 $v = 0$，回到纯扩散。<span class="marginnote">有偏向游走出现在几乎所有生物运动里：细菌的趋化（向信号梯度偏）、河流中浮游生物的下游输运、风媒孢子的顺风扩散。方程从「纯扩散」变成「平流-扩散方程」，行为也随之改变——漂移 $v$ 让整个分布以速度 $v$ 平移，同时扩散仍以 $\sqrt{Dt}$ 展宽。<strong>漂移主导长距离（$\sim vt$），扩散主导短距离（$\sim\sqrt{Dt}$）</strong>：长时间后，漂移把「位置不确定性」线性地拉开。</span>

**平流-扩散方程是第 4 篇所有空间模型的「母方程」**：第 21 篇的密度依赖扩散、第 22 篇的流行病波、第 23 篇的 Turing 斑图、第 24 篇的趋化——全部是「扩散 + 反应 + 平流」三件套的特定组合。趋化模型里那个「沿梯度定向运动」的项，正是一个与局部浓度梯度耦合的平流项（通量 $\chi u\nabla v$，见第 24 篇）。

**辨析｜易错点：** 漂移速度 $v$ 不是「单个粒子每秒走多远」，而是「概率分布的质心移动速度」。单个粒子一步仍只走 $\Delta x$，但分布的质心每步净移 $(p-q)\Delta x$。**分布行为 ≠ 个体行为**——这是从微观到宏观过渡时最容易犯的概念错误，也是理解一切「涌现」的第一课。

## 6 小结

- **随机游走**的递推 $P(x,t+\Delta t) = \tfrac12 P(x-\Delta x,t) + \tfrac12 P(x+\Delta x,t)$ 描述单粒子的微观运动。
- 泰勒展开后得到**扩散方程** $\partial u/\partial t = D\,\partial^2 u/\partial x^2$，$D = \Delta x^2/(2\Delta t)$。
- 高斯解三特征：均值不变、方差 $\langle x^2 \rangle = 2Dt$、标准差 $\propto \sqrt{t}$——扩散是平方根过程。
- 扩散方程**线性**，密度解 = 点源解的叠加；多维推广为 $u_t = D\nabla^2 u$
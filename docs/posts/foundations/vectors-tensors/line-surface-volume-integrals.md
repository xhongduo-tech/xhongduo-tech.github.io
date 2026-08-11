---
title: 线积分、面积分与体积分
date: 2026-08-11
---

# 线积分、面积分与体积分

<div class="epigraph">
<p>积分学教会我们把整体想成部分的叠加。</p>
<footer>—— 戈特弗里德 · 莱布尼茨（Gottfried Wilhelm Leibniz）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础科学 · 向量与张量初步 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从「沿着什么积分」开始

上一讲的梯度、散度、旋度全是「逐点」的局部信息。但物理量总要在**区域**上累计：做功是沿路径累计力与位移的点积，通量是穿过曲面的累计，质量是体积上的累计。<span class="marginnote">本讲对应 Boas《Mathematical Methods in the Physical Sciences》§6.3–§6.5 与 Arfken 第一章末尾：三种积分是后续 Gauss/Stokes 定理的构件。</span>

普通单变量积分 $\int f(x)\,dx$ 总是沿 $x$ 轴积。但多元世界里的「积分轴」可以是曲线、曲面或整个体——于是有了三种积分：**线积分、面积分、体积分**。它们的共同套路是同一个：**切成小片，近似，求和，取极限**。本讲把这条套路走三遍。

## 1 线积分：沿路径做累加

**线积分（line integral）** 是沿一条空间曲线 $C$ 的积分。若 $\mathbf F$ 是力场，质点沿 $C$ 移动时做的功是

$$
W = \int_C \mathbf F \cdot d\mathbf l = \int_a^b \mathbf F(\mathbf r(t))\cdot \mathbf r'(t)\, dt
$$

其中 $\mathbf r(t)$ 是曲线的参数化（$a\le t\le b$），$d\mathbf l = \mathbf r'(t)\,dt$ 是弧长微元向量。<span class="marginnote">直观：把路径切成无数小段 $d\mathbf l$，每一段上力近似不变，做功 $\mathbf F\cdot d\mathbf l$，再把所有小段加起来——这就是「切、近似、和、极限」四部曲。</span>

线积分对路径的依赖是后续物理的核心问题。若 $\mathbf F = \nabla\phi$ 是某标量场的梯度（保守力场），则

$$
\int_C \nabla\phi \cdot d\mathbf l = \phi(\mathbf r_b) - \phi(\mathbf r_a)
$$

**积分只取决于端点，与路径无关**——重力做功只取决于高度差，正是这个公式的日常体现。与之相对，摩擦力场不是梯度场，做功路径相关。

**辨析｜易错点：** 符号 $\oint_C$ 表示**闭合曲线**上的线积分（沿环路转一整圈）。在保守场中 $\oint \mathbf F\cdot d\mathbf l = 0$（回到起点，势能复原）。「积分值沿哪条路径走」——这是无数物理题的考点。

## 2 面积分：穿过曲面的通量

**面积分（surface integral）** 度量向量场穿过曲面的总量，即**通量（flux）**：

$$
\Phi = \int_S \mathbf F \cdot d\mathbf S = \iint_S \mathbf F\cdot\hat{\mathbf n}\, dS
$$

其中 $\hat{\mathbf n}$ 是曲面上每点的单位法向量，$dS$ 是面积微元，$d\mathbf S = \hat{\mathbf n}\,dS$ 是有向面积元。<span class="marginnote">法向量的取向约定：闭合曲面一律取「朝外」为正。这个约定在 Gauss 定理里是关键——把方向搞反，通量符号整体翻车。</span>

通量的直觉：站在曲面后，把 $\mathbf F$ 当作穿过它的水流速度。只有**垂直于曲面**的分量 $\mathbf F\cdot\hat{\mathbf n}$ 才「穿过」；平行于曲面的分量只是贴着面滑过，不产生通量。夹角越大、穿过越斜，有效通量越小。

## 3 体积分：在三维体上累加

**体积分（volume integral）** 最朴素：把标量密度 $\rho(\mathbf r)$ 在体积 $V$ 上累加：

$$
M = \int_V \rho(\mathbf r)\, dV
$$

直角坐标下直接化为三次积分 $dV = dx\,dy\,dz$。向量场也能做体积分——把每个分量分别积分：

$$
\int_V \mathbf F\, dV = \left(\int_V F_1 dV,\ \int_V F_2 dV,\ \int_V F_3 dV\right)
$$

体积分的难点全在**定限**：确定 $z$ 从哪到哪、$y$ 从哪到哪、$x$ 从哪到哪。球的体积用直角坐标极其痛苦，而换到球坐标只需一条公式——这正是《正交曲线坐标》一讲的内容。

## 4 换元公式：三种积分的统一后端

三种积分的「切、近似、和」都落在一个共同问题上：**切出来的微元到底有多大**。曲线微元、面积微元、体积微元分别由曲线、曲面、体积的 Jacobian 决定：

- 线：$d l = |\mathbf r'(t)|\, dt$；
- 面：$dS = \left|\dfrac{\partial\mathbf r}{\partial u} \times \dfrac{\partial\mathbf r}{\partial v}\right| du\, dv$；
- 体：$dV = |\det J|\, du\,dv\,dw$。

三个式子共享一个思想：**微元尺寸 = 参数变化对应的「拉伸倍数」**。<span class="marginnote">面积公式里的叉积模长，正是上一讲「叉积模长 = 平行四边形面积」的现场复用；体积公式里的 $|\det J|$ 是叉积思想向三维的推广。</span> 记住这一点，换个坐标系积分时就心里有底：多出的因子不是魔法，是微元被参数拉长/压缩的比例。

## 5 公式解析：做功积分里藏着一个向量函数

$$

\int_C \mathbf F \cdot d\mathbf l = \int_a^b \mathbf F(\mathbf r(t))\cdot \mathbf r'(t)\, dt

$$

这条公式把「几何路径上的积分」翻译成「参数区间上的普通积分」，逐项拆开：

- **$\mathbf r(t)$——参数化**：曲线 $C$ 被写成向量函数 $\mathbf r(t)$，$t$ 从 $a$ 走到 $b$。它是曲线的「时间」标签，一条曲线可以有无穷多种参数化。
- **$\mathbf r'(t)$——速度向量**：切向量，方向沿曲线。它与时间微元 $dt$ 相乘给出弧长微元 $d\mathbf l = \mathbf r'(t)\,dt$。
- **$\mathbf F(\mathbf r(t))$——场沿路径取值**：每走到参数 $t$，就取该点的力场值。注意要先代入路径，再与速度点积。
- **点积 $\mathbf F\cdot\mathbf r'$——只取切向贡献**：力与切向量点积，把「垂直于路径的力」自动清零。做负功、零功还是正功，全由这个点积的符号决定。

检验：若 $\mathbf F = \nabla\phi$，由链式法则 $\dfrac{d}{dt}\phi(\mathbf r(t)) = \nabla\phi\cdot\mathbf r'(t)$，积分立即变回 $\phi$ 在两端点的差——路径无关性在这一行里显形。

## 6 实例：一条闭合路径上的功

用具体的数把三讲的知识串起来。取力场

$$
\mathbf F = (-y,\ x,\ 0)
$$

这是一个绕原点打转的「涡旋」场。让质点沿圆周 $C$：$\mathbf r(t) = (R\cos t,\ R\sin t,\ 0)$，$0\le t\le 2\pi$，走一整圈做的功：

$$
W = \oint_C \mathbf F\cdot d\mathbf l = \int_0^{2\pi} \mathbf F(\mathbf r(t))\cdot \mathbf r'(t)\,dt
$$

代入 $\mathbf F(\mathbf r(t)) = (-R\sin t,\ R\cos t,\ 0)$，$\mathbf r'(t) = (-R\sin t,\ R\cos t,\ 0)$，点积得

$$
W = \int_0^{2\pi} R^2(\sin^2 t + \cos^2 t)\,dt = \int_0^{2\pi} R^2\,dt = 2\pi R^2
$$

结果非零——**沿闭合路径走一圈还做了功**，说明 $\mathbf F$ 不是保守场。核对：$\nabla\times\mathbf F = (0,0,2)$，Stokes 定理给出 $\oint\mathbf F\cdot d\mathbf l = \iint_S 2\,dS = 2\pi R^2$，两边完全吻合。<span class="marginnote">这个例子演示了下一讲的「预演」：闭合环路上的环量 = 所张曲面上的旋度通量。$\mathbf F$ 的涡旋结构（旋度处处为 2）在这里变成了可计算的数值。</span>

再对比保守场 $\mathbf F = \nabla(x^2+y^2) = (2x,2y)$：沿同一条圆周 $\oint \mathbf F\cdot d\mathbf l = 0$，因为回到起点势能复原。**同一几何路径，两种场，一非零一为零——做功的物理差异全在旋度是否为 0。** 这就是积分定理的价值：把「路径相关/无关」翻译成「旋度为零/非零」。

## 7 小结

- **线积分** $\int_C \mathbf F\cdot d\mathbf l$：沿路径累计，做功的数学模型；保守场（梯度场）下积分与路径无关。
- **面积分** $\iint_S \mathbf F\cdot d\mathbf S$：穿过曲面的通量，只有法向分量参与；闭合曲面法向约定朝外。
- **体积分** $\int_V \rho\, dV$：标量/向量场在体上的累加，难点在积分限。
- **换元统一**：线、面、体的微元 = 参数拉伸倍数（Jacobian），是三种积分的共同后端。

在下一节，我们把这三种积分与上一讲的梯度、散度、旋度焊接起来——Gauss 与 Stokes 两大定理登场，局部分析与整体累加在这里握手。

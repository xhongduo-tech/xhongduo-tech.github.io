---
title: 传染病空间传播与流行病行波
date: 2026-08-07
---

# 传染病空间传播与流行病行波

<div class="epigraph">
<p>瘟疫从不沿着直线行走，它在人与人之间跳跃，也在空间里爬行。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第二级 · 生物数学 ｜ Murray《Mathematical Biology I》第13章；Brauer & Castillo-Chavez 第9章 ｜ 2026-08-07</p>
</div>

## 为什么从流行病行波讲起

中世纪黑死病以每天约 2 公里的速度横穿欧洲；1918 流感沿铁路线跳跃传播；狂犬病在欧洲狐狸群中以每年几十公里的波阵面扩散。传染病不仅是时间过程，更是**空间过程**——病原体以行波形式在空间蔓延。这一节把 SIR 仓室模型与 Fisher-KPP 的行波理论结合，得到**空间 SIR 模型**，并导出流行病行波的波速公式。我们会看到，$R_0$ 依然统治着「能不能流行」，而波速则回答了「流行有多快」——两个问题合起来，就是空间流行病学的全部核心。

## 1 空间 SIR 模型：把扩散加进仓室

把 SIR 的每个仓室都赋予空间扩散。设 $S(x,t), I(x,t), R(x,t)$ 是位置 $x$、时刻 $t$ 的密度：

$$\frac{\partial S}{\partial t} = D_S\frac{\partial^2 S}{\partial x^2} - \beta S I$$

$$\frac{\partial I}{\partial t} = D_I\frac{\partial^2 I}{\partial x^2} + \beta S I - \gamma I$$

$$\frac{\partial R}{\partial t} = D_R\frac{\partial^2 R}{\partial x^2} + \gamma I$$

$D_S, D_I, D_R$ 分别是三仓室的扩散系数。<span class="marginnote">一个自然的简化假设是三个扩散系数相等 $D_S = D_I = D_R = D$（个体移动能力与疾病状态无关）。此时令总密度 $N = S + I + R$，三条方程相加得 $N_t = D N_{xx}$——总人口仍是纯扩散。许多分析在此假设下进行，而更精细的模型允许 $D_I$ 与 $D_S$ 不同（生病的人移动更少）。</span>

**传播项与扩散项并存**：$\beta SI$ 在局部制造感染，$D I_{xx}$ 把感染扩散到邻域。与 Fisher-KPP 的结构完全同构——只是「反应项」从单物种 Logistic 换成了双仓室的传染结构。

## 2 流行病行波与波速

在空间 SIR 模型里，疫情以**行波**形式传播：波前扫过之处，$I$ 从 0 升到峰值再回落，$S$ 从 $N$ 降到某个剩余值，$R$ 逐步积累。波速是核心问题：**疫情蔓延的空间速度是多少？**

对感染仓室，在波前（$S \approx N$，$I \to 0$）线性化：

$$\frac{\partial I}{\partial t} \approx D\frac{\partial^2 I}{\partial x^2} + (\beta N - \gamma) I$$

这是 Fisher-KPP 的线性化形式，增长率 $r = \beta N - \gamma$。于是**流行病行波的最小波速**

$$c_{\min} = 2\sqrt{D(\beta N - \gamma)} = 2\sqrt{D\gamma(R_0 - 1)}$$

其中用到 $R_0 = \beta N/\gamma$。<span class="marginnote">这条公式把两个域的结论连成一体：$R_0 - 1$ 是「超过阈值多少」的度量，$\gamma$ 是移除率，$D$ 是扩散系数。波速随 $R_0$ 增大而增大——<strong>疾病传染性越强，空间蔓延越快</strong>。注意 $R_0 = 1$ 时波速为 0——疾病原地打转、无法形成空间扩张；$R_0 \lt  1$ 时波速无实数，疫情局域消亡。<strong>$R_0 > 1$ 是空间扩散的前提</strong>，时间域的阈值在这里原样搬进空间域。</span>

**波速公式的直觉**：有效增长率 $r_{\text{eff}} = \beta N - \gamma = \gamma(R_0 - 1)$ 越高，波前「送料」越足；扩散系数 $D$ 越大，波前「开路」越快。两者按几何平均 $2\sqrt{\cdot}$ 组合——与 Fisher-KPP 的 $2\sqrt{rD}$ 完全同构。

## 3 案例：狂犬病与黑死病的波速测算

流行病行波理论最经典的验证是**狐狸狂犬病**。Murray 等用空间 SIR 建模狐狸种群中的狂犬病传播，得到与观测吻合的波速。

参数：狐狸扩散系数 $D \approx 60\ \text{km}^2/\text{year}$，去除率 $\gamma \approx 1/\text{year}$（潜伏期+发病期约 1 年），$R_0$ 估计约 2–3。代入波速公式：

$$c \approx 2\sqrt{D\gamma(R_0 - 1)} \approx 2\sqrt{60 \times 1 \times (2.5 - 1)} \approx 19\ \text{km}/\text{year}$$

与欧洲狐狸狂犬病实际观测的约 20–60 km/年量级吻合。<span class="marginnote">波速公式在这里的价值不是「精确预测」，而是「量级检验」——它把一个复杂的空间流行病压缩成三个可测参数（$D, \gamma, R_0$）的平方根组合，并能与野外观测交叉验证。类似的量级测算被用于禽流感的空间风险评估、新冠疫情初期的传播速度估计。<strong>行波理论提供的是「快还是慢、为什么」的机理判断，而不是具体日期的预报。</strong></span>

**黑死病（1347–1350）**的历史扩散速度约每天 1–2 km。若假设当时欧洲人口 $R_0 \approx 2$、感染期约 20 天（$\gamma \approx 0.05$），反推扩散系数

$$D \approx \frac{c^2}{4\gamma(R_0 - 1)} \approx \frac{(1.5\ \text{km/day})^2}{4 \times 0.05 \times 1} \approx 11\ \text{km}^2/\text{day}$$

这个数量级对应「人通过日常通勤与商贸在数公里尺度移动」——行波模型把历史传染病变成了可反推的物理参数。

## 4 公式解析：流行病波速的完整推导

从空间 SIR 到波速公式，走完整推导链。

$$
\frac{\partial I}{\partial t} = D\frac{\partial^2 I}{\partial x^2} + \beta S I - \gamma I
$$

- **第一步，波前线性化**：设行波 $I(x,t) = I_0 e^{-\lambda(x - ct)}$（前缘指数衰减），$S \approx N$。代入方程：

$$
\lambda c\, I = D\lambda^2 I + (\beta N - \gamma) I
$$

- **第二步，约去 $I$，得色散关系**：

$$
c = D\lambda + \frac{\beta N - \gamma}{\lambda}
$$

- **第三步，对 $\lambda$ 最小化**：$\frac{dc}{d\lambda} = D - \frac{\beta N - \gamma}{\lambda^2} = 0$，得 $\lambda^* = \sqrt{\frac{\beta N - \gamma}{D}}$，代回得

$$
c_{\min} = D\sqrt{\frac{\beta N - \gamma}{D}} + \frac{\beta N - \gamma}{\sqrt{(\beta N - \gamma)/D}} = 2\sqrt{D(\beta N - \gamma)}
$$

- **第四步，用 $R_0$ 重写**：$\beta N - \gamma = \gamma(R_0 - 1)$，故

$$
c_{\min} = 2\sqrt{D\gamma(R_0 - 1)}
$$

<span class="marginnote">与 Fisher-KPP 的对照值得铭记：Fisher 的 $c_{\min} = 2\sqrt{rD}$ 里的 $r$ 是单物种增长率；这里的「增长率」换成了<strong>净传播速率</strong> $\beta N - \gamma = \gamma(R_0 - 1)$。两个理论共享同一个数学骨架——<strong>波速 = 2 × √(扩散 × 净增长率)</strong>——只是「净增长率」的内容因系统而异。学完 Fisher-KPP 再学流行病波，你其实是在同一个公式上换了参数。</span>

**波速公式的管理含义**：降低 $D$（限制流动、封控、隔离）与压低 $R_0$（疫苗、口罩）对波速的影响是**乘性**的——两条腿都要动。只压 $R_0$ 不控流动，波速的平方根里仍留有 $D$ 的贡献。

## 5 与 Fisher-KPP 的对照：一张表看懂两个波速

流行病行波与生态行波共享同一个数学骨架，把它们并排比较，能同时加深对两者的理解。

| 项目 | Fisher-KPP 生态波 | 流行病行波 |
| --- | --- | --- |
| 方程 | $u_t = D u_{xx} + r u(1-u/K)$ | $I_t = D I_{xx} + (\beta S - \gamma) I$ |
| 前缘线性化 | $u_t \approx D u_{xx} + r u$ | $I_t \approx D I_{xx} + (\beta N - \gamma) I$ |
| 净增长率 | $r$ | $\beta N - \gamma = \gamma(R_0 - 1)$ |
| 最小波速 | $2\sqrt{rD}$ | $2\sqrt{D\gamma(R_0-1)}$ |
| 失稳条件 | $r > 0$ | $R_0 > 1$ |
| 波后状态 | 饱和密度 $K$ | $S$ 降到剩余值，$I \to 0$ |

<span class="marginnote">表格的最后一行的差别最值得品味：生态波「占领并填满」（波后密度 $K$），流行病波「扫过并退潮」（波后 $I \to 0$、留下免疫者）。同样是行波，<strong>一个留下「永久的占领」，一个留下「暂时的疤痕」</strong>——这是反应项结构不同（单物种 Logistic vs SIR 仓室）的直接后果。理解这个差别，你就不会再混淆两类模型。</span>

**反过来用**：流行病波速公式还可以「倒着读」——实测疫情的空间传播速度 $c$ 与时间动力学参数（$R_0$、$\gamma$），可以反推空间扩散系数 $D = c^2/[4\gamma(R_0-1)]$。这个反演让「人群流动对疫情传播的贡献」第一次变得可测量：$D$ 大说明长距离流动强，$D$ 接近背景值说明传播主要是局域接触。**行波理论由此成为流行病空间监测的标定工具**，而非仅仅是黑板上的漂亮公式。

**辨析｜易错点：** 波速公式假设「行波已稳定」且人群密度均匀。真实疫情常处于「加速期」或受地形、人口分布、交通网络调制——直接套公式会系统性偏差。与生态行波同理，**把公式当「量级基线 + 参数反演工具」，比当精确预报更可靠**。

## 6 小结

- **空间 SIR 模型**给每个仓室加扩散项：$S_t = D S_{xx} - \beta SI$，$I_t = D I_{xx} + \beta SI - \gamma I$，$R_t = D R_{xx} + \gamma I$。
- **流行病行波最小波速** $c_{\min} = 2\sqrt{D(\beta N - \gamma)} = 2\sqrt{D\gamma(R_0 - 1)}$。
- 波速与 $R_0$ 的关系：$R_0 > 1$ 才能形成空间扩张；$R_0 = 1$ 时波速为零；$R_0 \lt  1$ 局域消亡。
- 狂犬病狐狸波速量级验证：$D \approx 60$ km²/yr、$R_0 \approx 2.5$ 给出约 20 km/yr，与观测吻合。
- 波速公式与 Fisher-KPP 同构：**波速 = 2 × √(扩散 × 净增长率)**，管理上扩散与 $R_0$
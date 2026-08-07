---
title: 非线性方程与线性化
date: 2026-08-07
---

# 非线性方程与线性化

<div class="epigraph">
<p>局部地看，每条曲线都像它的切线；局部地看，每个系统都像它的线性近似。</p>
<footer>—— 化用于亨利 · 庞加莱（Henri Poincaré）的动力系统思想</footer>
</div>

<div class="article-byline">
<p>第二级 · 常微分方程 ｜ 丁同仁《常微分方程》 第二篇 第五章 §3 ｜ 2026-08-07</p>
</div>

## 为什么非线性必须「绕着看」

前两篇的线性世界是温室：解能叠加、能写公式。可真实系统几乎全是非线性的——单摆 $\ddot\theta + \sin\theta = 0$、捕食者-猎物模型、化学反应、人口逻辑斯谛。非线性没有叠加原理，也没有通解公式。<span class="marginnote">庞加莱在 19 世纪末开辟了「定性理论」：不求解，而是研究解的几何形状与长期命运。这是动力系统的起点，也直接通向混沌研究（20 世纪下半叶）。</span>

非线性问题的破局点是**线性化（linearization）**：在一个平衡点附近，用雅可比矩阵的线性系统「近似」非线性系统。就像用切线近似曲线——局部精确、全局有限。Hartman–Grobman 定理给这条思路背书。

## 1 平衡点与非线性系统

考虑平面自治系统

$$\begin{cases} x' = f(x, y) \\ y' = g(x, y) \end{cases} \qquad\text{或简写}\quad \boldsymbol{x}' = \boldsymbol{F}(\boldsymbol{x})$$

**平衡点（equilibrium / critical point）**：使 $\boldsymbol{F}(\boldsymbol{x}^*) = \boldsymbol{0}$ 的点。在平衡点处系统静止——$(x^*, y^*)$ 若作为初值，解永远停在那里。

**例子**：单摆方程 $\ddot\theta + \sin\theta = 0$ 化为方程组（设 $x = \theta,\ y = \dot\theta$）：

$$x' = y, \qquad y' = -\sin x$$

平衡点满足 $y = 0,\ \sin x = 0$，即 $(k\pi, 0)$。物理直觉：$x = 0$（最低点）与 $x = \pi$（倒立点）都是平衡——但前者稳定、后者不稳定。线性化将把这种直觉变成数学判断。<span class="marginnote">「平衡点」一词在工程里也叫「不动点」「驻点」。自治系统（右端不含时间）才有「点」的平衡；非自治系统随时间漂移，平衡概念要换成周期解等，见第 22 节《周期解》。</span>

## 2 线性化：雅可比矩阵

在平衡点 $\boldsymbol{x}^*$ 附近做泰勒展开，令扰动 $\boldsymbol{u} = \boldsymbol{x} - \boldsymbol{x}^*$，忽略二阶以上小量：

$$\boldsymbol{u}' \approx J\,\boldsymbol{u}, \qquad J = \begin{pmatrix} \dfrac{\partial f}{\partial x} & \dfrac{\partial f}{\partial y} \\[6pt] \dfrac{\partial g}{\partial x} & \dfrac{\partial g}{\partial y} \end{pmatrix}_{\boldsymbol{x} = \boldsymbol{x}^*}$$

$J$ 是**雅可比矩阵（Jacobian matrix）**，在平衡点处取值。$\boldsymbol{u}' = J\boldsymbol{u}$ 称为原系统的**线性化系统**。

**Hartman–Grobman 定理**：若平衡点 $\boldsymbol{x}^*$ 处的雅可比矩阵 $J$ **没有实部为零的特征值**（即平衡点是**双曲的，hyperbolic**），则非线性系统与线性化系统在 $\boldsymbol{x}^*$ 附近**拓扑等价**——存在同胚把非线性流的轨道映射为线性流的轨道。<span class="marginnote">「拓扑等价」的意思是：局部轨道图画法不变——稳定方向还是稳定、螺旋还是螺旋、鞍点还是鞍点。直觉：<strong>双曲平衡点像钉子一样「扎」住附近的流形结构，非线性小项扭不动它</strong>。</span>

**例子**：单摆在 $(0,0)$ 处线性化。$f = y, g = -\sin x$，$J = \begin{pmatrix} 0 & 1 \\ -\cos x & 0 \end{pmatrix}$，在 $(0,0)$ 得 $J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$，特征值 $\pm i$——中心。在 $(\pi, 0)$ 处 $J = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$，特征值 $\pm 1$——鞍点（不稳定）。这与物理直觉完全吻合：最低点绕圈振，倒立点一推就跑。

## 3 线性化失效的情形

Hartman–Grobman 要求「双曲」。当 $J$ 有**零实部特征值**（实部为零或纯虚）时，非线性项可能改变局部分类，线性化结论不再可靠。常见三种「非双曲」陷阱：

- **零特征值**：如 $x' = -x^3$。线性化 $u' = 0$ 说「中性稳定」，实际非线性使 $x \to 0$ 稳定。线性化完全误判。
- **纯虚特征值（中心）**：线性化给出中心（环绕），非线性可能把它「掰弯」成螺旋或产生极限环。范德波尔方程 $x'' + \mu(x^2-1)x' + x = 0$ 在原点线性化是中心，但真实系统有一个**稳定的极限环**。
- **高次项主导**：当线性部分「退化」（如齐次项全为零）时，需要中心流形或正规形理论。

**辨析｜易错点：** 「线性化」不是「答案」，只是「局部近似」。在双曲点它可靠（Hartman–Grobman），在非双曲点必须回到非线性分析。**看到特征值实部为零，先别下稳定/不稳定的结论**——这一步判断失误，后面全盘皆错。<span class="marginnote">范德波尔方程提醒我们：极限环（自激振荡）这类现象在纯线性系统里根本不存在，必须由非线性来创造。电子管振荡器、心脏搏动、神经脉冲都与它同源——第三级《自动控制原理》里它叫「自振」。</span>

## 4 公式解析：线性化 $\boldsymbol{u}' = J\boldsymbol{u}$

把非线性系统「切成」线性系统的完整过程，逐层拆：

- **第一步，泰勒展开**：$\boldsymbol{F}(\boldsymbol{x}) = \boldsymbol{F}(\boldsymbol{x}^*) + J(\boldsymbol{x}^*)\boldsymbol{u} + O(\|\boldsymbol{u}\|^2)$。平衡点处 $\boldsymbol{F}(\boldsymbol{x}^*) = \boldsymbol{0}$，于是 $\boldsymbol{u}' \approx J\boldsymbol{u}$。
- **第二步，$J$ 的每一项是什么**：$J_{ij} = \partial F_i/\partial x_j$ 表示「第 $j$ 个变量动一点，第 $i$ 个分量变多快」——一阶影响矩阵。常数项为零是平衡点的馈赠。
- **第三步，为什么小扰动下才成立**：丢掉的是 $O(\|\boldsymbol{u}\|^2)$。$\|\boldsymbol{u}\|$ 足够小时二阶项相对一阶项可忽略；离平衡点远了，近似失效。
- **第四步，特征值的意义**：$J$ 的特征值实部决定扰动方向的增长/衰减。Hartman–Grobman 说：只要实部都非零，这个「一阶判断」就不仅仅是近似，而是**拓扑上精确的**——这是线性化理论的定心丸。<span class="marginnote">这套「在平衡点切开泰勒展开、用雅可比特征值判断」的流程，正是工程里「工作点小信号分析」的数学骨架：放大器的工作点线性化、电力系统的小扰动稳定性、化学反应的平衡点附近行为，全都这一个动作。</span>

## 5 实例：捕食者-猎物模型

**洛特卡-沃尔泰拉模型（Lotka–Volterra）**

$$x' = \alpha x - \beta xy, \qquad y' = -\gamma y + \delta xy$$

描述猎物 $x$ 与捕食者 $y$ 的种群振荡。平衡点：$(0,0)$ 与 $(x^*, y^*) = (\gamma/\delta,\ \alpha/\beta)$。

在 $(x^*, y^*)$ 处线性化。雅可比

$$J = \begin{pmatrix} \alpha - \beta y & -\beta x \\ \delta y & -\gamma + \delta x \end{pmatrix}_{(x^*,y^*)} = \begin{pmatrix} 0 & -\beta\gamma/\delta \\ \delta\alpha/\beta & 0 \end{pmatrix}$$

特征方程 $\lambda^2 + \alpha\gamma = 0$，特征值 $\pm i\sqrt{\alpha\gamma}$——**纯虚根**！线性化给出中心（绕圈）。但注意：**这是非双曲点**，Hartman–Grobman 定理不适用，中心可能被非线性项改造成焦点。

实际该模型有**首次积分**（守恒量），轨道确实是闭合曲线族——种群循环振荡，扰动不会让轨道衰减或发散。这类「中心在非线性下保持中心」的情形，要凭守恒量或下一节的李雅普诺夫方法单独证明，不能靠线性化。<span class="marginnote">洛特卡-沃尔泰拉模型还揭示了「渔获增加反而可能减少捕食者」等反直觉结论——定性理论在这里不是学术游戏，而是生态决策的数学依据。</span>

**辨析｜易错点：** 千万别在纯虚根处下「稳定」或「不稳定」的结论——那是非双曲情形，线性化给不出确定性答案。正确姿势：找守恒量（首积分），或进入李雅普诺夫方法。

### 首次积分：非线性系统里的守恒量

洛特卡-沃尔泰拉模型除了线性化，还有一条更深刻的路径：找**首次积分（first integral）**。把两式相除消去时间：

$$\frac{dy}{dx} = \frac{-\gamma y + \delta xy}{\alpha x - \beta xy}$$

分离得 $\dfrac{\alpha - \beta y}{y}dy = \dfrac{-\gamma + \delta x}{x}dx$，积分得

$$\alpha\ln y - \beta y + \gamma\ln x - \delta x = \text{常数}$$

这就是系统的守恒量 $H(x,y)$。**轨道正是 $H = $ 常数的等高线**——闭合曲线族，周期振荡。这个首积分证明：即使线性化不可靠（纯虚根非双曲），系统仍是中心型，种群循环往复。<span class="marginnote">首次积分把「微分方程的求解」降维成「代数等高线的绘制」——不积分时间，只积分空间方向。力学里的能量守恒、角动量守恒都是首积分的物理名号。</span>

**对比三条路**：

| 方法 | 结论 | 适用范围 |
| --- | --- | --- |
| 线性化（Hartman–Grobman） | 局部拓扑 | 双曲平衡点 |
| 首次积分 | 全局轨道形状 | 存在守恒量的系统 |
| 李雅普诺夫函数 | 稳定性 | 找到合适 $V$ |

**辨析｜易错点：** 首次积分可遇不可求——大多数非线性系统没有守恒量。但一旦找到，它能一锤定音地给出全局结论，胜过线性化的局部近似。遇到「疑似守恒」的系统（如不含耗散项的力学系统），先试找首积分，常常事半功倍。

**再强调**：线性化是「局部放大镜」，不是「全局地图」。双曲平衡点处它精确，远离平衡点它失效。定性分析的正确姿势是：局部用线性化，全局用首积分/相图/数值。

**对照**：Hartman–Grobman 只保拓扑、不保距离——局部形状一样，具体轨道位置可以不同。别把「拓扑等价」误读成「数值相等」。

## 6 小结

- **非线性系统** $\boldsymbol{x}' = \boldsymbol{F}(\boldsymbol{x})$ 在**平衡点** $\boldsymbol{F}(\boldsymbol{x}^*)=\boldsymbol{0}$ 附近用**雅可比矩阵** $J$ 线性化：$\boldsymbol{u}' = J\boldsymbol{u}$。
- **Hartman–Grobman 定理**：双曲平衡点（$J$ 无零实部特征值）处非线性系统与线性化系统拓扑等价。
- 单摆：最低点（中心）稳定绕圈、倒立点（鞍点）不稳定——线性化重现物理直觉。
- **非双曲情形**（零实部特征值）线性化可能失效：零特征值、中心、退化情形都要回到非线性分析。
- 极限环等非线性特有现象不存在于线性世界，线性化看不到它们。

在下一节，我们把「平衡点附近的拓扑」画成图——**相平面与奇点分类**，系统化地给奇点命名与归类。

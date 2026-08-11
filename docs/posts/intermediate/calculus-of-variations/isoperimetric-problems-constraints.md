---
title: 等周问题与约束变分
date: 2026-08-11
---

# 等周问题与约束变分

<div class="epigraph">
<p>在所有给定周长的平面图形中，圆围出的面积最大。</p>
<footer>—— 数学史上最早的变分命题（传说源于狄多女王）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 变分法 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从等周问题开始

上一课的 Euler-Lagrange 方程处理的是**无约束**变分：只固定端点，自由竞争。但现实中的极值问题几乎总带着约束——绳长固定、周长固定、能量固定、总质量固定。<span class="marginnote">约束变分在应用里随处可见：固定表面积的肥皂泡要最大体积、固定长度的链条要最低重心、固定预算的组合要最高收益——「固定某积分」是约束的抽象形态。</span> 而「等周」（isoperimetric，字面义「等周长」）问题正是史上第一个变分问题：相传腓尼基公主狄多（Dido）用一张牛皮切成细条，围出海岸线上最大的土地，最终得到半圆。这一课把有限维的 Lagrange 乘子法推广到无穷维，得到处理积分约束的完整配方。它也是 van Brunt Ch. 4 的核心内容。

## 1 狄多的传说与等周问题

传说归传说，数学命题是精确的：

**等周问题（isoperimetric problem）**：在所有长度固定为 $L$ 的闭合平面曲线中，求围出面积最大的那一条。

答案是一条圆（半径为 $L/2\pi$）。这个事实可以浓缩成一条优美的**等周不等式（isoperimetric inequality）**：

$$
4\pi A \le L^2
$$

其中 $A$ 是闭合曲线围出的面积，$L$ 是周长；等号成立当且仅当曲线是圆。<span class="marginnote">等周不等式的证明有很多条路：变分法、傅里叶分析（用 Parseval 恒等式可给出最简证明）、以及凸几何里的对称化。它在数学内部纵横交错，堪称「一条命题，半部分析」。</span>

把问题翻译成变分语言：参数化曲线 $(x(t), y(t))$，围出面积

$$
A = \frac{1}{2}\oint (x\,y' - x'\,y)\, dt
$$

周长

$$
L = \oint \sqrt{x'(t)^2 + y'(t)^2}\, dt
$$

约束是 $L$ 固定。这与「极小化 $J = \int F\,dt$ 并让 $\int G\,dt$ 保持常数」是同一类问题。

## 2 带积分约束的变分问题

一般地，考虑**等周约束（isoperimetric constraint）**下的变分问题：

$$
\min \int_a^b F(x,y,y')\, dx, \qquad \text{约束：} \int_a^b G(x,y,y')\, dx = C
$$

外加端点条件 $y(a)=A,\ y(b)=B$。$C$ 是一个给定的常数。注意约束的形式是「被积函数 $G$ 的积分固定」，而不是「每一点上 $G$ 固定」——这是它与「点约束」（如 $y(a)=A$）的根本区别。<span class="marginnote">点约束直接划掉一个自由度；积分约束则是「总体预算」——沿整条曲线的贡献之和被锁定，但每一点上仍可自由调整。这正是「等周」的历史遗风：周长被锁定，形状仍可改变。</span>

处理它的工具是 Lagrange 乘子，与有限维优化一模一样：

**约束变分的关键定理**：若 $y$ 是上述带约束问题的极值曲线（且约束非退化，即存在扰动使 $\int\delta G\,dx \neq 0$），则存在常数 $\lambda$，使得 $y$ 是**无约束**问题

$$
\min \int_a^b \bigl[ F(x,y,y') + \lambda\, G(x,y,y') \bigr] dx
$$

的极值曲线。<span class="marginnote">直观上 $\lambda$ 是「约束的价格」：它把一条硬约束折算成对目标函数的线性惩罚，然后我们放它自由竞争。$\lambda$ 的数值不是预先知道，而是最后由「代入约束方程」反解出来。</span>

## 3 推导：为什么乘子法在无穷维也成立

把第一变分的记号用上。设 $y$ 是带约束问题的极值曲线，则对一切满足

$$
\int_a^b \bigl(G_y\,\eta + G_{y'}\,\eta'\bigr) dx = 0
$$

的扰动 $\eta$（即「保持约束的一阶变分」），必须有目标的一阶变分

$$
\int_a^b \bigl(F_y\,\eta + F_{y'}\,\eta'\bigr) dx = 0
$$

这两条「正交性」合在一起说明：泛函 $\delta J$ 与 $\delta C$ 在约束超平面上的作用成比例，即存在 $\lambda$ 使

$$
\int_a^b \bigl(F_y - \frac{d}{dx}F_{y'} + \lambda\bigl[G_y - \frac{d}{dx}G_{y'}\bigr]\bigr)\eta\, dx = 0 \qquad \forall \eta
$$

（这里照例对 $F_{y'}\eta'$、$G_{y'}\eta'$ 做了分部积分。）由基本引理得到

$$
F_y - \frac{d}{dx}F_{y'} + \lambda\Bigl(G_y - \frac{d}{dx}G_{y'}\Bigr) = 0
$$

这正是对修正被积函数 $F + \lambda G$ 写的 Euler-Lagrange 方程。<span class="marginnote">严格化需要处理「约束非退化」与端点项，van Brunt Ch. 4 给出了完整证明；这里略去的是技术细节，不是思想——与有限维 Lagrange 乘子法完全同构。</span>

## 4 公式解析：修正泛函 $F + \lambda G$

$$
\boxed{\; \tilde F(x,y,y',\lambda) = F(x,y,y') + \lambda\, G(x,y,y') \;}
$$

以及它对 $\tilde F$ 的 Euler-Lagrange 方程：

$$
\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} + \lambda\Bigl(\frac{\partial G}{\partial y} - \frac{d}{dx}\frac{\partial G}{\partial y'}\Bigr) = 0
$$

三步拆解：

- **第一步，$\lambda$ 是一个未知标量**：它不是新函数，只是一个待定常数。解 E-L 方程时把它当参数留着，得到的解 $y = y(x; \lambda)$ 一簇一簇地依赖 $\lambda$。
- **第二步，约束回来定 $\lambda$**：把 $y(x;\lambda)$ 代回约束 $\int G\,dx = C$，解出 $\lambda$ 的具体值。这正是「未知数个数 = 方程个数」的平衡：多了一个 $\lambda$，多了一条约束方程。
- **第三步，物理直觉**：力学里固定弦长求最大面积，$\lambda$ 对应「张力」；固定体积求最小表面积，$\lambda$ 对应「压强」。$\lambda$ 是约束对目标施加的「影子价格」。<span class="marginnote">「影子价格」的说法来自运筹学与经济学：一个约束的边际价值等于最优时乘子的值。这个想法在第三级《凸优化》与《强化学习》的拉格朗日对偶里会以更完整的面貌重现。</span>

## 5 解等周问题：圆是怎么长出来的

回到等周问题。取参数 $t$ 为弧长参数（则 $x'(t)^2 + y'(t)^2 = 1$），修正泛函取 $G = \sqrt{x'^2 + y'^2}$，目标 $F = \frac{1}{2}(x y' - x' y)$，于是

$$
\tilde F = \frac{1}{2}(x y' - x' y) + \lambda \sqrt{x'^2 + y'^2}
$$

对 $x$ 与 $y$ 分别写 Euler-Lagrange 方程，得到

$$
y' = \lambda\,\frac{d}{dt}\frac{x'}{\sqrt{x'^2+y'^2}}, \qquad
x' = -\lambda\,\frac{d}{dt}\frac{y'}{\sqrt{x'^2+y'^2}}
$$

把 $x'/\sqrt{x'^2+y'^2}$ 看成「单位切向量的 $x$ 分量」，两条方程合起来说明：**曲线上每一点的曲率都是同一常数 $1/|\lambda|$**。而曲率为常数的闭合平面曲线只能是圆。于是半径 $r = |\lambda| = L/2\pi$，面积 $A = L^2/4\pi$，等周不等式取等号。<span class="marginnote">「曲率恒定 ⇒ 圆」是微分几何的著名事实；而「曲率恒定的等周封闭曲线围出最大面积」也是肥皂泡问题（固定表面积的极小曲面）在一维的投影——见本专题《极小曲面问题与平均曲率》。</span>

狄多问题略有不同：海岸线充当「免费」直线边界，于是最优是半圆——推理过程完全一样，只是把圆沿直径切开。<span class="marginnote">狄多问题正是「给定长度的曲线 + 直线边界围最大面积」，最优是半圆；它在 19 世纪被 Weierstrass 严格证明前，已被众多数学家非形式地确认。整段历史也是变分法「先有猜想、后有证明」的缩影。</span>

## 6 小结

- **等周约束**是 $\int G\,dx = C$ 形式的整体约束，区别于逐点的边界条件。
- 解法：**Lagrange 乘子法**——极小化 $\int (F + \lambda G)\,dx$，把 $\lambda$ 当参数解 E-L，再回代约束定出 $\lambda$。
- 等周问题中修正泛函的 E-L 方程给出「曲率恒定」，从而推出最优曲线是**圆**。
- 几何结论浓缩为**等周不等式** $4\pi A \le L^2$，等号仅由圆取得。
- $\lambda$ 的物理解释是约束的「影子价格」（张力、压强），与有限维优化一脉相承。

在下一节，我们把变分法推进物理的核心地带：**Hamilton 原理与最小作用量原理**——在那里，真实运动的轨迹被重新定义为「作用量泛函的驻点」。

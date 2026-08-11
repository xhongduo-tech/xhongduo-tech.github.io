---
title: 单粒子轨道与漂移运动
date: 2026-08-11
---

# 单粒子轨道与漂移运动

<div class="epigraph">
<p>单个带电粒子在给定电磁场中的运动，是一切等离子体研究的起点——群体的复杂，正是由这枚螺线构筑的。</p>
<footer>—— 弗朗西斯 · 陈（Francis F. Chen, <em>Introduction to Plasma Physics and Controlled Fusion</em>）</footer>
</div>

<div class="article-byline">
<p>第四级 · 高阶专题 · 等离子体物理 ｜ 对标教材 Chen Ch.2 ｜ 2026-08-11</p>
</div>

## 为什么从单粒子开始

上一节我们站在群体视角，靠 Debye 屏蔽理解了等离子体的准电中性。但集体行为的舞台——磁场约束、波的传播、
聚变装置的几何——最终都是**一个粒子**画出来的。一个带电粒子在均匀磁场中画螺旋线，在非均匀场中则会缓慢「漂移」。托卡马克的约束方案、
磁镜的反射原理，全都藏在这几条漂移公式里。这一篇我们只做一件事：盯住一个粒子，看它怎么动。

## 1 回旋运动：无场之外的第一个解

设均匀磁场 $\mathbf{B} = B\hat{z}$，无电场，无碰撞。洛伦兹力 
$\mathbf{F} = q\mathbf{v}\times\mathbf{B}$恒垂直于速度，因此它**不做功**——粒子的动能不变，
只在垂直于 $\mathbf{B}$ 的平面内把直线拉成圆周。
<span class="marginnote">洛伦兹力不做功、磁场只拐弯不加速，这是磁约束的第一原理：约束靠的是「改方向」而不是「关笼子」。
</span>

这个圆周运动叫**回旋运动（gyromotion）**，两个基本量：

$$
\omega_c = \frac{|q|B}{m}, \qquad r_L = \frac{m v_\perp}{|q|B}
$$

$\omega_c$ 是**回旋频率（cyclotron frequency）**，$r_L$ 是**回旋半径（Larmor radius）
**。看公式：电子轻，$r_L$ 小、$\omega_c$ 高；离子重，反之。**回旋方向由电荷符号决定**——面向磁场方向看去，离子逆时针、
电子顺时针。
<span class="marginnote">在托卡马克等离子体里，电子回旋半径约零点几毫米，离子的约一厘米——一个「粒子回旋半径」尺度的物理（
如漂移波、湍流涡旋）正好落在这两种尺度之间。</span>整个运动是：沿 $\mathbf{B}$ 匀速直线 + 垂直 
$\mathbf{B}$ 匀速圆周 = **螺旋线**。我们把「圆心的位置」称为**导向中心（guiding center）**，
接下来所有漂移，都是这个导向中心在动。

## 2 三种漂移：E×B、梯度、曲率

磁场不可能是完美均匀的。一旦场有梯度或弯曲，导向中心就会在回旋之上叠加一个缓慢的横向运动，叫作**漂移（drift）**。

**E×B 漂移。** 电场与磁场同时存在时，导向中心以

$$
\mathbf{v}_E = \frac{\mathbf{E}\times\mathbf{B}}{B^2}
$$

漂移。这个公式最令人吃惊的地方是：**它不含 $q$，也不含 $m$**。电子和离子以完全相同速度朝同一方向漂移——两种电荷不分离、不产生电流。
直觉来自「电场加速，磁场拐弯」的交替：在回旋的半个周期电场沿速度方向推，半个周期逆着推，最终导向中心被「拖着」横向匀速移动，
漂移速度正好使平均电场力 $q\mathbf{E}$ 与平均磁力 $q\mathbf{v}_E\times\mathbf{B}$ 抵消。

![E×B 漂移：离子（青）与电子（红）回旋方向相反，但导向中心以同速同向漂移](/images/plasma-physics/single-particle-orbits-drifts-1.svg)

**梯度漂移。** 磁场 $B$ 有空间梯度 $\nabla B$ 时，回旋半径在强场侧变小、弱场侧变大，轨迹变成「滚动的圆」，导向中心以

$$
\mathbf{v}_{\nabla B} = \frac{m v_\perp^2}{2 q B^3}\,\mathbf{B}\times\nabla B
$$

漂移。注意这次 $q$ 在分母上——**离子与电子漂向相反方向**，造成电荷分离、产生电流。这正是磁约束装置里所有麻烦的源头。

**曲率漂移。** 磁力线弯曲（沿磁场走）时，粒子在弯曲轨迹上受到离心力 
$m v_\parallel^2 \mathbf{R}_c/R_c^2$，导向中心以

$$
\mathbf{v}_R = \frac{m v_\parallel^2}{q}\,\frac{\mathbf{R}_c\times\mathbf{B}}{R_c^2 B^2}
$$

漂移，同样与 $q$ 有关、与 $v_\parallel^2$ 成正比。
<span class="marginnote">把梯度漂移与曲率漂移合写：
$\mathbf{v}_D = \dfrac{m}{qB^2}\big(v_\parallel^2+\tfrac12 v_\perp^2\big)\dfrac{\mathbf{B}\times\nabla B}{B}$
——两条漂移同向叠加，是环形装置中粒子外漂移的全部来源。</span>

**辨析｜易错点：** 漂移不是「粒子被力推着横向走」，而是「回旋运动的不对称性被统计平均后的净效果」。E×B 漂移对两种粒子同向，
**不产生电荷分离**；梯度与曲率漂移对两种粒子反向，**产生电荷分离与电流**——这一区别决定了磁约束的命运。

## 3 磁矩守恒与磁镜：绝热不变量

磁场缓慢变化（在一个回旋周期内变化很小）时，粒子的**磁矩（magnetic moment）**

$$
\mu = \frac{m v_\perp^2}{2B}
$$

近似不变——这是第一个**绝热不变量（adiabatic invariant）**。
<span class="marginnote">严格证明来自作
用量不变量 $\oint p\,dq$ 在缓变参数下守恒：$\mu$ 正是回旋运动的横向作用量除以 $2\pi m$。</span>

磁矩守恒的推论非常漂亮：粒子向强磁场区运动时 $B$ 增大，为保持 $\mu$ 不变，$v_\perp$ 必须增大；动能守恒又要求 
$v_\parallel$ 减小——粒子「越走越慢」，最终 $v_\parallel = 0$ 被**反射**回头。这种两端强、
中间弱的磁场位形叫**磁镜（magnetic mirror）**，地球范艾伦辐射带就是天然的磁镜囚笼。
<span class="marginnote">被俘获的条件：粒子在弱场处与磁力线的夹角 $\theta_0$ 满足 
$\sin^2\theta_0 > B_{\min}/B_{\max}$，否则会从「磁镜的漏斗嘴」
漏出去——这个漏掉的相空间范围叫损失锥（loss cone），磁镜装置难以聚变的根本原因。</span>

## 4 时间变化的电场：极化漂移

电场缓慢随时间变化时，还会多出一种漂移：

$$
\mathbf{v}_p = \frac{m}{q B^2}\frac{d\mathbf{E}}{dt}
$$

它依赖 $q$——电子与离子漂向相反方向，于是**产生极化电荷与极化电流**。这套「极化」逻辑在波的传播（第 4、5 篇）中至关重要：
交变电场驱动的极化电流，正是电磁波在等离子体中折射率偏离真空的来源。

## 5 公式解析：E×B 漂移速度

$$
\mathbf{v}_E = \frac{\mathbf{E}\times\mathbf{B}}{B^2}
$$

三步拆解：

- **第一步，看清运算。** 叉积 $\mathbf{E}\times\mathbf{B}$ 给出一个垂直于 $\mathbf{E}$ 和 $\mathbf{B}$ 的方向；除以 $B^2$ 只是归一化，保证量纲是速度（$\mathrm{V/m}\times\mathrm{T}/\mathrm{T^2}=\mathrm{m/s}$）。
- **第二步，理解为何与 q、m 无关。** 在漂移参考系里变换到以 $\mathbf{v}_E$ 运动的坐标系，电场被伽利略变换吃掉一部分（$\mathbf{E}' = \mathbf{E} + \mathbf{v}\times\mathbf{B}$），正好取 $\mathbf{v}=\mathbf{v}_E$ 使 $\mathbf{E}'=0$。电荷与质量的作用都已经被这个「换参考系」抹平了。
- **第三步，记下方向口诀。** 「左手不是右手」：设 $\mathbf{B}$ 指向纸外、$\mathbf{E}$ 向上，则 $\mathbf{E}\times\mathbf{B}$ 指向**右**——所有粒子都朝右漂。<span class="marginnote">方向务必自己验算一次：$\hat{x}\times\hat{z} = -\hat{y}$，叉积的反交换律在这里是命根子。托卡马克等离子体电流产生的电场与环向磁场的 E×B 漂移方向，直接决定杂质输运的方向。</span>

同一套逻辑可以立刻推广：把「等效重力场」$\mathbf{g}$ 放进来，就是重力漂移 
$\mathbf{v}_g = \dfrac{m}{q}\dfrac{\mathbf{g}\times\mathbf{B}}{B^2}$——它同
样与 $q$ 有关，是后文 Rayleigh–Taylor 型不稳定性（第 7 篇）的微观源头。

## 6 小结

- 均匀磁场中的粒子做**回旋运动**：$\omega_c = |q|B/m$，$r_L = mv_\perp/|q|B$；磁力不做功、只改方向。
- **导向中心**在非均匀场中漂移：E×B 漂移 $\mathbf{v}_E = \mathbf{E}\times\mathbf{B}/B^2$ 与电荷无关、不产生电流。
- **梯度漂移**与**曲率漂移**依赖 $q$，两种粒子反向，是电荷分离与磁约束困难的根源。
- **磁矩守恒** $\mu = mv_\perp^2/2B$ 是第一个绝热不变量，构造出**磁镜**与范艾伦带。
- 漂移的物理本质是回旋不对称性的平均，不是外力直接推着走。

在下一节，我们把无数粒子的漂移与回旋平均成一种连续介质，写出等离子体自己的流体力学——**磁流体力学方程**。

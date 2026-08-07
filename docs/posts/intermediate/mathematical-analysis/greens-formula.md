---
title: 格林公式：平面区域与曲线积分的联系
date: 2026-08-07
---

# 格林公式：平面区域与曲线积分的联系

<div class="epigraph">
<p>区域内部的旋度总量，等于边界上的环量——格林公式把「面的积分」与「边的积分」连接起来，是高斯、斯托克斯公式的二维样板。</p>
<footer>—— 乔治·格林（George Green），1828 年《数学分析在电磁理论中的应用》（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§21.3 ｜ 2026-08-07</p>
</div>

## 为什么格林公式是「二重积分的分水岭」

累次积分解决了「怎么算二重积分」，但二重积分与曲线积分之间还缺一座桥：**面积分与线积分如何互化？** 格林公式回答：

$$\oint_{\partial D}P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\,dy.$$

**边界上的线积分 = 内部面积分。** 它是「牛顿—莱布尼茨公式」的二维版——一元里「函数端点差 = 内部导数积分」，二维里「边界环量 = 内部旋度积分」。格林公式、高斯公式（§22.3）、斯托克斯公式（§22.4）共同构成「场论三定理」，而格林公式是它们中最简单、最优雅的一个。<span class="marginnote">格林公式的精神与微积分基本定理一脉相承：<strong>「边界的量 = 内部的导数」</strong>。一元：$f(b)-f(a)=\int f'$（端点 = 内部）；二维：$\oint$（边界）= $\iint$（内部旋度）；三维：$\iint$（曲面）= $\iiint$（内部散度）。这一族「广义斯托克斯公式」是分析学最深刻的统一主题——「边界算子」与「导数算子」是对偶的。第二级《微分几何》里，这条统一用「外微分 + 斯托克斯定理」一句话写完。</span>

## 1 格林公式

**定理（格林公式 / Green's Theorem）：设 $D$ 是有界闭区域，边界 $\partial D$ 是分段光滑简单闭曲线（逆时针方向），$P(x,y),Q(x,y)$ 在 $D$ 上有连续偏导数，则**

$$\oint_{\partial D}P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\,dy.$$

**公式解析：三步拆解**

**第一步，拆成两半**。只需证 $\oint P\,dx=-\iint\frac{\partial P}{\partial y}$ 与 $\oint Q\,dy=\iint\frac{\partial Q}{\partial x}$，相加即得——**两个方向的「分部积分」**；

**第二步，先证 $P$ 半**。设 $D$ 是 $X$ 型区域（$a\le x\le b,\ \varphi_1(x)\le y\le\varphi_2(x)$）。累次积分：

$$-\iint_D\frac{\partial P}{\partial y}dx\,dy=-\int_a^b\left[\int_{\varphi_1}^{\varphi_2}\frac{\partial P}{\partial y}dy\right]dx=-\int_a^b\left[P(x,\varphi_2)-P(x,\varphi_1)\right]dx;$$

**第三步，与线积分对上**。$\oint P\,dx$ 沿边界（下边 $y=\varphi_1$ 正向、上边 $y=\varphi_2$ 反向）：

$$\oint_{\partial D}P\,dx=\int_a^bP(x,\varphi_1)dx-\int_a^bP(x,\varphi_2)dx=\int_a^b\left[P(x,\varphi_1)-P(x,\varphi_2)\right]dx,$$

与上一步相等。$Q$ 半同理（用 $Y$ 型区域）。∎

**要点**：**证明的灵魂是「化归 $X$/$Y$ 型区域 + 累次积分 + 边界线积分」**——二重积分（内部）与线积分（边界）在「竖切片」的视角下自然相等。复杂区域用「分割成标准区域」拼接（可加性）。

## 2 格林公式的「旋度」解读

**旋度（curl，二维）**：$Q_x-P_y=\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}$ 称为向量场 $(P,Q)$ 的（标量）旋度。格林公式：

$$\oint_{\partial D}\vec F\cdot d\vec r=\iint_D(\text{curl}\,\vec F)\,d\sigma.$$

**边界环量 = 内部旋度总量**——**场绕边界的净旋转，等于内部每点旋度的总和**。这就是「环量」的微观来源：每个小区域内部的旋转，在边界上「累加」成净环量。

> **辨析｜易错点：**格林公式的**方向约定**：$\partial D$ 取**逆时针**（正方向，区域在左）。若取顺时针，线积分反号。另一个易错点：**$P,Q$ 要有连续偏导数**（$D$ 上）——若边界或内部有奇点（$P,Q$ 在点 $P_0$ 无定义），格林公式直接失效，需「挖洞」处理（见下节路径无关性的奇点）。还有：**$D$ 可以是「带洞区域」**（如圆环），格林公式对「外边界逆时针、内边界顺时针」的复合边界仍成立。

## 3 格林公式的应用

**应用一：用线积分算面积**。取 $P=-y$、$Q=x$，则 $Q_x-P_y=1+1=2$，格林公式给出

$$\iint_D d\sigma=\frac12\oint_{\partial D}(-y\,dx+x\,dy).$$

**区域面积 = 边界线积分的一半**——「平面面积仪」（planimeter）的原理！$x=\cos t,\ y=\sin t$（单位圆）：$\frac12\oint(-y\,dx+x\,dy)=\frac12\int_0^{2\pi}(\sin^2t+\cos^2t)dt=\pi$。✓<span class="marginnote">「面积 = $\frac12\oint(x\,dy-y\,dx)$」是测量学与计算机图形学的实用工具：平面面积仪（planimeter）靠追踪边界直接读出面积，多边形面积（鞋带公式）也由它推导——计算机图形学（第三级）里计算任意多边形面积的标准公式 $A=\frac12|\sum(x_iy_{i+1}-x_{i+1}y_i)|$ 正是这条线积分公式的离散版。<strong>一个纯理论公式，成了工程测量的日常工具</strong>。</span>

**应用二：用面积分算线积分（简化计算）**。$\oint_C y^2dx+x^2dy$ 沿单位圆逆时针：格林公式给出

$$\iint_D\left(\frac{\partial x^2}{\partial x}-\frac{\partial y^2}{\partial y}\right)dx\,dy=\iint_D(2x-2y)dx\,dy=2\iint_Dx-2\iint_Dy=0$$

（圆盘关于原点对称，$\iint x=\iint y=0$）。**把难算的线积分化成好算的面积分**。

**应用三：计算环量**。$\oint_C(-y)dx+x\,dy$ 沿椭圆 $x=a\cos t,\ y=b\sin t$：格林公式给出 $\iint_D(1+1)=2\cdot\text{面积}=2\pi ab$——**向量场 $( -y,x)$ 的环量 = 2 × 包围面积**，这正是「涡旋场」的环量度量。

## 4 格林公式与路径无关（预告）

格林公式直接催生「曲线积分与路径无关」的判据（§21.4）：设 $D$ 单连通（无洞），$P,Q$ 有连续偏导数，则

**$\oint_{\partial D}P\,dx+Q\,dy=0$（任何闭曲线）$\iff Q_x-P_y=0$（在 $D$ 内）**——由格林公式，闭曲线积分 = 内部旋度积分；旋度恒零则环量恒零，反之亦然。**「旋度为零」是「积分与路径无关」的判据**——§21.4 的主角，将在下节展开。

## 5 格林公式的地位

格林公式是「场论三定理」之首，也是「广义斯托克斯公式」的二维版本：

| 定理 | 维度 | 内容 |
| --- | --- | --- |
| 微积分基本定理 | 1 | $f(b)-f(a)=\int_a^bf'$ |
| 格林公式 | 2 | $\oint Pdx+Qdy=\iint(Q_x-P_y)$ |
| 高斯公式 | 3 | $\iint\vec F\cdot d\vec S=\iiint\text{div}\,\vec F$（§22.3） |
| 斯托克斯公式 | 3（曲面） | $\oint\vec F\cdot d\vec r=\iint\text{curl}\,\vec F\cdot d\vec S$（§22.4） |

**「边界积分 = 内部导数积分」是统一主题**——格林是二维样板，高斯与斯托克斯是三维推广。<span class="marginnote">「广义斯托克斯公式」在第二级《微分几何》里以「$\int_\Omega d\omega=\int_{\partial\Omega}\omega$」一句话统摄格林、高斯、斯托克斯——这是微分形式理论最辉煌的成就：一个公式、所有维度。今天学的格林公式，是那条统一公式在二维的具体展开。而「面积 = $\frac12\oint(xdy-ydx)$」这类工程公式，正是广义斯托克斯的实用后代。</span>

## 6 小结

- **格林公式**：$\oint_{\partial D}Pdx+Qdy=\iint_D(Q_x-P_y)dxdy$——边界线积分 = 内部旋度积分。
- **证明**：化归 $X$/$Y$ 型区域 + 累次积分 + 边界线积分。
- **旋度解读**：$Q_x-P_y$ 是二维旋度；环量 = 旋度总量。
- **方向约定**：$\partial D$ 逆时针为正；带洞区域内外边界方向相反。
- **应用**：面积 = $\frac12\oint(-ydx+xdy)$（面积仪）、线积分简化、环量计算。

在下一节，我们研究格林公式的直接推论：**曲线积分与路径无关的条件**。旋度为零 ⇒ 积分与路径无关 ⇒ 存在势函数——「保守场」的完整刻画。

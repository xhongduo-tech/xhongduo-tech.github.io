---
title: 密切平面与副法向量
date: 2026-08-07
---

# 密切平面与副法向量

<div class="epigraph">
<p>几何学并非真实，但它是有用的。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）《科学与假设》</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§1.3 ｜ 2026-08-07</p>
</div>

## 为什么从密切平面开始

上一节我们用曲率 $\kappa$ 回答了一个问题：「曲线有多弯」。但「弯」是平面里的直觉——直线弯成圆，圆弯成椭圆，全都发生在同一个平面里。**空间曲线真正的难点不在「弯」，而在「不在同一个平面上」**：它一边弯，一边还可能伸出当前所在的平面，飘向第三维。要描述这种「伸出平面」的行为，就必须先问一个更基础的问题：**曲线在一点附近，到底贴近哪个平面？**

本节就回答这个问题。答案是**密切平面（osculating plane）**：曲线在切点处贴合得最好的那个平面，它是上一节密切圆由「圆」升维成「面」的产物。围绕它我们会长出两个新对象：**主法向量** $\mathbf{N}$（指向凹侧、已在上节惊鸿一瞥）与**副法向量** $\mathbf{B}$（$\mathbf{T} \times \mathbf{N}$）。三个向量两两垂直，从此曲线在每一点都有了自己的三维坐标系——这正是整座 **Frenet 标架**大厦的第一块砖，也是通往曲面论「每一点都长着一根法向量」的语言准备。这个「每点一副正交坐标」的想法，你在第二级《线性代数》里见过的都是它的**静态**版本——正交基、坐标分解；本节开始，它第一次以**动态**的面目出现：坐标系跟着曲线跑。

## 1 从一个圆到一个平面

回顾上一节的密切圆：曲线在 $P$ 点的曲率半径为 $\rho = 1/\kappa$，圆心在**主法向量**方向、距 $P$ 恰好 $\rho$ 处。<span class="marginnote">主法向量（principal normal）在弧长参数下定义为 $\mathbf{N}(s) = \mathbf{T}'(s)/\kappa(s)$，它总是指向曲线「弯过去」的那一侧，也就是曲率中心所在的一侧。</span>密切圆是一个**圆**，圆必然躺在某个**平面**里——这个平面，就是曲线在 $P$ 点最贴近的平面。

于是自然定义：

**密切平面（osculating plane）**：曲线 $\alpha$ 在参数 $t$ 处的密切平面，是由切向量 $\alpha'(t)$ 与 $\alpha''(t)$ 张成的平面，即

$$
\Pi(t) = \operatorname{span}\big\{\alpha'(t),\, \alpha''(t)\big\} = \big\{\,\alpha(t) + u\,\alpha'(t) + v\,\alpha''(t)\;\big|\;u,v\in\mathbb{R}\big\}
$$

在弧长参数下，$\alpha'(s) = \mathbf{T}$，$\alpha''(s) = \kappa\,\mathbf{N}$，所以**密切平面正是 $\mathbf{T}$ 与 $\mathbf{N}$ 张成的平面**。词源上 "osculating" 来自拉丁语 *osculare*（亲吻）——这个平面与曲线在 $P$ 点「亲吻」在一起，比任何别的平面都贴得更紧。

**重点：密切平面是曲线在一点附近的「最佳二维近似」，正如切线是「最佳一维近似」。** 切线只保证方向相同；密切平面保证曲线局部不再离开这个平面（偏差是三阶小量，见下）。把「直线、圆、平面」三层近似放一起，就得到一张完整的逼近阶梯。

| 近似层级 | 对象 | 用到的导数 | 贴合精度 |
| --- | --- | --- | --- |
| 一阶 | 切线 | $\alpha'$ | 偏差 $O(h^2)$ |
| 二阶 | 密切圆 | $\alpha',\alpha''$ | 偏差 $O(h^3)$ |
| 二阶 | 密切平面 | $\alpha',\alpha''$ | 偏差 $O(h^3)$（曲线暂时不离开平面） |

![密切平面与 Frenet 标架示意](/images/differential-geometry/osculating-plane-binormal-1.svg)

上图中，曲线在山谷底部 $P$ 处的密切平面用青色半透明平行四边形表示，$\mathbf{T}$、$\mathbf{N}$ 都躺在它里面，而副法向量 $\mathbf{B}$ 垂直地指向读者——**$\mathbf{B}$ 的方向，就是密切平面「朝向」的方向**。

## 2 副法向量：把平面「竖」起来

一个平面由它的**法向量**唯一决定。密切平面的法向量，就是我们的第二个主角：

**副法向量（binormal）**：

$$
\mathbf{B}(t) = \frac{\alpha'(t) \times \alpha''(t)}{\big\|\alpha'(t) \times \alpha''(t)\big\|}
$$

在弧长参数下它简化为 $\mathbf{B} = \mathbf{T} \times \mathbf{N}$。三个名字的来历值得说清：$\mathbf{T}$ 叫**切**（tangent），$\mathbf{N}$ 叫**主法**（principal normal，因为它指向弯曲的主方向），$\mathbf{B}$ 叫**副法**（binormal，字面是「第二个法向量」）——因为它也是法向量（垂直于切平面），但排在 $\mathbf{N}$ 之后。<span class="marginnote">有的教材把 $\mathbf{N}$ 与 $\mathbf{B}$ 的命名反过来强调「谁承载弯曲信息」：$\mathbf{N}$ 在密切平面内、管「弯」，$\mathbf{B}$ 垂直于密切平面、管「拧」。记住分工，后面理解挠率时会非常省力。</span>

副法向量的几何身份清晰：它垂直于 $\mathbf{T}$，也垂直于 $\mathbf{N}$，所以垂直于整个密切平面。换句话说，**$\mathbf{B}$ 的方向就是密切平面「朝向」的指针**。若密切平面在空间里转动，$\mathbf{B}$ 就会跟着转——这一句话是下一节挠率的全部伏笔。

## 3 三个平面，一个标架

有了 $\mathbf{T}$、$\mathbf{N}$、$\mathbf{B}$ 三个两两垂直的单位向量，曲线在每一点就有了一副完整的正交轴。它们两两张出三个互相垂直的平面：

- **密切平面**（$\mathbf{T},\mathbf{N}$ 张成）：曲线局部贴着它，密切圆躺在它里面。
- **法平面**（$\mathbf{N},\mathbf{B}$ 张成）：垂直于 $\mathbf{T}$，过 $P$ 点的所有「横截面」方向都在其中。
- **从切平面**（$\mathbf{T},\mathbf{B}$ 张成）：垂直于 $\mathbf{N}$，名字来自拉丁语 *rectificare*（拉直）。

它们像墙角的三个墙面，两两相交于一条轴：

$$
\text{法平面} \perp \text{切线},\qquad
\text{从切平面} \perp \text{主法线},\qquad
\text{密切平面} \perp \text{副法线}
$$

**$\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$ 构成单位右手正交标架**，因为 $\mathbf{B} = \mathbf{T}\times\mathbf{N}$ 保证了 $\mathbf{B}$ 的方向符合右手定则。曲线上每一点都背着这样一副小坐标系，它随 $s$ 而动——这就是「**活动标架**」的最初形态，下一节我们将给它一个正式的名字：**Frenet 标架**。

## 4 公式解析：副法向量的显式公式

对非弧长参数，$\mathbf{T}$、$\mathbf{N}$ 都有烦人的归一化分母，但 $\mathbf{B}$ 的公式干净得多，值得逐项拆解：

$$
\mathbf{B}(t) = \frac{\alpha'(t) \times \alpha''(t)}{\big\|\alpha'(t) \times \alpha''(t)\big\|}
$$

- **第一步，认分母**：$\|\alpha'(t) \times \alpha''(t)\|$。由上一节的曲率推导我们知道，这个叉积的模长 $= v^3\kappa$，其中 $v = \|\alpha'\|$。它出现在分母，作用是**归一化**——让 $\mathbf{B}$ 成为单位向量，不随参数化的快慢而缩放。
- **第二步，认分子**：$\alpha'(t) \times \alpha''(t)$。把 $\alpha'' = v'\mathbf{T} + v^2\kappa\,\mathbf{N}$ 代入（上一节已推出），$\alpha' = v\mathbf{T}$ 与切向部分 $v'\mathbf{T}$ 的叉积为零，只剩下

$$
\alpha' \times \alpha'' = v\mathbf{T} \times \big(v'\mathbf{T} + v^2\kappa\,\mathbf{N}\big) = v^3\kappa\,(\mathbf{T}\times\mathbf{N})
$$

  $\mathbf{T}\times\mathbf{N}$ 是单位向量，所以分子的**方向**恰恰是 $\mathbf{B}$ 的方向，只是长度多了 $v^3\kappa$。
- **第三步，合并**：归一化把 $v^3\kappa$ 这个长度因子消掉，剩下的正是 $\mathbf{B} = \mathbf{T}\times\mathbf{N}$。**这条公式的核心动作是叉积**：叉积同时做了「消去切向分量」（$\alpha'\times\alpha'$ 部分为 $0$）与「给出垂直于张成平面的方向」（$\mathbf{T}\times\mathbf{N}$）两件事——与曲率公式同源同构。<span class="marginnote">注意公式里没有一个平方根套着分子分母的嵌套：分母本就是叉积的模长，自带归一化，所以 $\mathbf{B}$ 的表达式是三个对象里最「轻」的。</span>

## 5 算例与辨析

**例 1（圆）**：圆躺在 $xy$ 平面内，$\mathbf{T}$、$\mathbf{N}$ 都在平面内，$\mathbf{B} = \mathbf{T}\times\mathbf{N}$ 恒为 $\pm\mathbf{z}$，是一个**常向量**。密切平面就是 $xy$ 平面本身——圆从生到死没有离开过自己的平面。

**例 2（圆柱螺旋线）**：$\alpha(t) = (a\cos t, a\sin t, bt)$。我们已经算过 $\alpha'\times\alpha'' = (ab\sin t,\, -ab\cos t,\, a^2)$，归一化得

$$
\mathbf{B}(t) = \frac{1}{\sqrt{a^2+b^2}}\left(b\sin t,\; -b\cos t,\; a\right)
$$

**$\mathbf{B}$ 随 $t$ 旋转**——这意味着密切平面在空间里不断倾斜。这正是螺旋线区别于圆的地方：它一边绕圈一边上升，逼近它的平面也只好跟着拧。<span class="marginnote">把螺旋线的 $\mathbf{B}$ 沿一整圈 $t\in[0,2\pi]$ 看一遍，它绕 $z$ 轴恰好转了一整圈。这个「转了几圈」的整数，在 DNA 超螺旋理论里就是链接数（linking number）的雏形。</span>

**辨析｜易错点 1：直线没有密切平面。** 直线的 $\alpha''=\mathbf{0}$，$\alpha'\times\alpha''=\mathbf{0}$，副法向量 $\mathbf{B}$ 没有定义。几何上，直线不弯、也不需要任何「贴合平面」——它本身就是直的。凡是曲率为零的孤立点（如拐点），该点的密切平面同样没有定义。

**辨析｜易错点 2：$\mathbf{B}$ 的符号依赖定向。** 若把曲线反向重参数化，切向量 $\mathbf{T}$ 变号，主法向量 $\mathbf{N}$ **不变**（它总指向曲率中心，与行走方向无关），于是副法向量 $\mathbf{B} = \mathbf{T}\times\mathbf{N}$ 跟着 $\mathbf{T}$ **变号**。反过来，若人为规定 $\mathbf{N}$ 取反（许多教材对平面曲线正是这么做的），$\mathbf{B}$ 也会反转。**副法向量的方向依赖定向选择，这是曲线论里符号约定的第一个「雷区」**——下一节我们会看到，正因为如此，曲率与挠率的公式里才会出现刻意的符号约定。

**辨析｜易错点 3：密切平面 ≠ 曲线所在平面。** 平面曲线（如圆、椭圆、抛物线）的密切平面恒为曲线所在平面；但空间曲线在每一点的密切平面一般**各不相同**。判断一条曲线是否真的「三维」，标准不是它有没有曲率，而是它的密切平面会不会转动——转动与否，由下一节的挠率来精确计量。

## 6 小结

- **密切平面** $\Pi = \operatorname{span}\{\alpha', \alpha''\} = \operatorname{span}\{\mathbf{T},\mathbf{N}\}$：曲线在一点附近贴合得最好的平面，是密切圆的「升维」。
- **副法向量** $\mathbf{B} = \dfrac{\alpha'\times\alpha''}{\|\alpha'\times\alpha''\|} = \mathbf{T}\times\mathbf{N}$：密切平面的单位法向量，管「平面朝哪」。
- 三个平面——**密切平面**（$\mathbf{T}\mathbf{N}$）、**法平面**（$\mathbf{N}\mathbf{B}$）、**从切平面**（$\mathbf{T}\mathbf{B}$）——两两垂直。
- $\{\mathbf{T},\mathbf{N},\mathbf{B}\}$ 构成单位右手正交标架，随弧长 $s$ 在曲线上滑动。
- **易错**：直线（$\kappa=0$ 处）无密切平面与副法向量；$\mathbf{B}$ 的符号依赖定向选择；空间曲线各点密切平面一般不同。

在下一节《Frenet 标架》中，我们将把这个「每点一副坐标系」的想法系统化：给 $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$ 一个正式的名字，研究它如何随 $s$ 运动，并预告它在整个微分几何中作为「活动标架」方法的种子地位。

---
title: Frenet 公式
date: 2026-08-07
---

# Frenet 公式

<div class="epigraph">
<p>在数学中，你并不是理解事物，你只是习惯了它们。</p>
<footer>—— 约翰 · 冯 · 诺依曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§1.5 ｜ 2026-08-07</p>
</div>

## 为什么从公式开始

前面的三节，我们像拼图一样凑齐了全部零件：切向量 $\mathbf{T}$、主法向量 $\mathbf{N}$、副法向量 $\mathbf{B}$ 构成的 Frenet 标架，度量「弯」的曲率 $\kappa$，度量「拧」的挠率 $\tau$。零件齐了，但还差最后一步：**把它们焊成一个会动的整体。** 这一步就是 **Frenet 公式**——三条方程，把标架随弧长 $s$ 的「运动」完整地写出来。

这三条方程的美妙之处在于：**一个单位正交标架随参数的变化，永远可以写成「它自身 × 一个反对称矩阵」的形式**，而 Frenet 公式就是这个一般原理在一维曲线上的具体化身。它一箭三雕：让 $\kappa$ 与 $\tau$ 的全部几何意义有了归宿，让我们能反过来从 $\kappa(s)$、$\tau(s)$ 重建整条曲线（曲线论基本定理），并且为后面曲面论与黎曼几何里更一般的「联络」埋下原型。可以说，Frenet 公式是曲线论的总账本——前几节的每一分耕耘，都在这里结账。

## 1 三条方程：标架的「运动方程」

**Frenet 公式（Frenet–Serret formulas）**：设 $\alpha(s)$ 是弧长参数化的正则曲线，$\kappa(s)>0$，$\tau(s)$ 是它的曲率与挠率，则 Frenet 标架满足

$$
\left\{\begin{aligned}
\mathbf{T}'(s) &= \phantom{-}\kappa(s)\,\mathbf{N}(s),\\
\mathbf{N}'(s) &= -\kappa(s)\,\mathbf{T}(s) + \tau(s)\,\mathbf{B}(s),\\
\mathbf{B}'(s) &= -\tau(s)\,\mathbf{N}(s)
\end{aligned}\right.
$$

每条方程都是一句几何告白：

- $\mathbf{T}' = \kappa\mathbf{N}$：**切向量只朝 $\mathbf{N}$ 方向弯**，弯的速率是 $\kappa$。这正是曲率的定义。
- $\mathbf{B}' = -\tau\mathbf{N}$：**副法向量只朝 $\mathbf{N}$ 方向倒**，倒的速率是 $\tau$。这正是挠率的定义。
- $\mathbf{N}' = -\kappa\mathbf{T} + \tau\mathbf{B}$：**主法向量同时被两个机制拉扯**——曲率把它拽向 $-\mathbf{T}$，挠率把它推向 $+\mathbf{B}$。

**重点：$\mathbf{T}$ 与 $\mathbf{B}$ 的方程里都只有 $\mathbf{N}$，而 $\mathbf{N}$ 的方程里同时出现 $\mathbf{T}$ 与 $\mathbf{B}$。** 这不奇怪——$\mathbf{N}$ 夹在另外两个之间，是标架「转动」的中枢。<span class="marginnote">直觉图像：$\mathbf{T}$ 像车厢头，$\mathbf{N}$ 像车厢地板，$\mathbf{B}$ 像车顶法线。地板被车头的转向（$\kappa$）拉回、被车顶的侧倾（$\tau$）推走——Frenet 公式就是这节车厢的运动学方程。</span>

## 2 公式解析：逐条推导 Frenet 公式

三条方程里，第一条与第三条几乎就是定义，唯一需要真正「推」的是第二条。完整走一遍：

- **第一步，写出 $\mathbf{N} = \mathbf{B}\times\mathbf{T}$**：由 $\mathbf{B} = \mathbf{T}\times\mathbf{N}$ 与右手系性质，等价地有 $\mathbf{N} = \mathbf{B}\times\mathbf{T}$。对 $s$ 求导，用叉积的乘积法则

$$
\mathbf{N}' = \mathbf{B}'\times\mathbf{T} + \mathbf{B}\times\mathbf{T}'
$$

- **第二步，代入第一、第三条方程**：把 $\mathbf{T}'=\kappa\mathbf{N}$、$\mathbf{B}'=-\tau\mathbf{N}$ 代入

$$
\mathbf{N}' = (-\tau\mathbf{N})\times\mathbf{T} + \mathbf{B}\times(\kappa\mathbf{N})
          = -\tau\,(\mathbf{N}\times\mathbf{T}) + \kappa\,(\mathbf{B}\times\mathbf{N})
$$

- **第三步，算两个叉积**：$\mathbf{N}\times\mathbf{T} = -(\mathbf{T}\times\mathbf{N}) = -\mathbf{B}$；再由向量恒等式 $(\mathbf{T}\times\mathbf{N})\times\mathbf{N} = -\mathbf{T}$，得 $\mathbf{B}\times\mathbf{N} = -\mathbf{T}$。

- **第四步，整理**：代入得

$$
\mathbf{N}' = -\tau(-\mathbf{B}) + \kappa(-\mathbf{T}) = -\kappa\mathbf{T} + \tau\mathbf{B}
$$

**第二条方程出炉。** 这条推导的全部秘密，是把「$\mathbf{N}$ 是另外两个的叉积」这一几何事实，转化为「$\mathbf{N}$ 的导数由另外两个的导数决定」这一微分事实——**标架的每一个分量都可以从其余分量算出，这正是正交系最可贵的性质。**<span class="marginnote">细心的读者会发现：三步中用到的全是前两节已经建立的事实（$\mathbf{B}=\mathbf{T}\times\mathbf{N}$、$\mathbf{T}'=\kappa\mathbf{N}$、$\mathbf{B}'=-\tau\mathbf{N}$）。Frenet 公式不是新的天外信息，而是旧信息的浓缩——这也是它可信的根源。</span>

## 3 矩阵形式：反对称的优雅

把三条方程并成矩阵乘法，会有惊人的对称：

$$
\begin{bmatrix}
\mathbf{T}'\\[2pt]
\mathbf{N}'\\[2pt]
\mathbf{B}'
\end{bmatrix}
=
\begin{bmatrix}
0 & \kappa & 0\\
-\kappa & 0 & \tau\\
0 & -\tau & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{T}\\[2pt]
\mathbf{N}\\[2pt]
\mathbf{B}
\end{bmatrix}
$$

中间这个矩阵记为 $A(s)$，它满足 $A^\mathsf{T} = -A$——**反对称**。这不是巧合，而是一条深刻原理在一维的显形：**任何一个单位正交标架场，它对参数的导数矩阵必然反对称**。理由很简单：$[\mathbf{T}\ \mathbf{N}\ \mathbf{B}]$ 作为列拼成的矩阵 $F$ 恒满足 $F^\mathsf{T}F = I$，求导得 $F'^{\mathsf{T}}F + F^{\mathsf{T}}F' = 0$，即 $A + A^{\mathsf{T}} = 0$。<span class="marginnote">你在第二级《线性代数》里见过反对称矩阵与叉积的联系：对任意向量 $\boldsymbol\omega$，映射 $\mathbf{v}\mapsto\boldsymbol\omega\times\mathbf{v}$ 就是一个反对称线性算子。Frenet 公式里的反对称矩阵正来自某个「角速度向量」的叉积作用。</span>

这个「角速度向量」确实存在，叫 **Darboux 向量**：

$$
\boldsymbol\omega(s) = \tau(s)\,\mathbf{T}(s) + \kappa(s)\,\mathbf{B}(s)
$$

可以逐条验证（留给读者）：$\mathbf{T}'=\boldsymbol\omega\times\mathbf{T}$、$\mathbf{N}'=\boldsymbol\omega\times\mathbf{N}$、$\mathbf{B}'=\boldsymbol\omega\times\mathbf{B}$。**整副标架绕 $\boldsymbol\omega$ 旋转**，旋转速率是 $\|\boldsymbol\omega\| = \sqrt{\kappa^2+\tau^2}$。曲线论至此给出一个极简洁的画面：Frenet 标架像一架被角速度 $\boldsymbol\omega$ 驱动的陀螺仪，而 $\kappa$、$\tau$ 恰好是这个角速度的两个分量。

## 4 从公式到基本定理：两个函数决定一条曲线

Frenet 公式最深刻的应用，是把「曲线」从几何对象变成**微分方程的解**。

给定两个连续函数 $\kappa(s)>0$、$\tau(s)$，以及初始时刻的标架与起点，微分方程组

$$
\mathbf{T}' = \kappa\mathbf{N},\quad
\mathbf{N}' = -\kappa\mathbf{T} + \tau\mathbf{B},\quad
\mathbf{B}' = -\tau\mathbf{N}
$$

在初值条件下有**唯一解**（常微分方程理论的基本定理），再积分 $\alpha'(s)=\mathbf{T}(s)$ 就得到整条曲线。于是：

**定理（曲线论基本定理，预告）**：给定 $\kappa(s)>0$、$\tau(s)$，存在唯一的空间曲线（相差一个刚体运动）以它们为曲率与挠率；两条曲线形状相同当且仅当它们的 $\kappa$、$\tau$ 逐点相同。

这就是为什么我们说**曲线被两个函数完全编码**。位置与姿态的刚体自由度（6 个）是「初始条件」的事，而形状本身只剩 $\kappa$ 与 $\tau$。<span class="marginnote">类比：一条平面曲线的形状被曲率函数 $\kappa(s)$ 一个函数决定；空间曲线需要两个函数，因为空间比平面多了一个「拧」的自由度。信息论地说，曲线把三维几何压缩成了两个一维信号。</span>

## 5 辨析：使用 Frenet 公式时的陷阱

**辨析｜易错点 1：Frenet 公式必须在弧长参数下使用。** 公式里的撇号是 $\dfrac{d}{ds}$。若手上只有一般参数 $t$，$\alpha''$ 里混着切向加速，直接套公式会错。正确姿势：要么先换成弧长参数，要么用链式法则把 $\dfrac{d}{dt}$ 与 $\dfrac{d}{ds}$ 的换算（差一个 $v = ds/dt$ 因子）显式写出来。

**辨析｜易错点 2：$\tau$ 的符号约定再次考验你。** 若换用 $\mathbf{B}'=+\tau\mathbf{N}$ 的约定，第二、第三条方程都会变号：$\mathbf{N}'=-\kappa\mathbf{T}-\tau\mathbf{B}$、$\mathbf{B}'=+\tau\mathbf{N}$。**三个公式必须配套使用**，混用两套约定是初学者最常见的「翻车」方式。

**辨析｜易错点 3：$\kappa=0$ 处整条公式崩塌。** 当 $\kappa=0$（如拐点），$\mathbf{N}$ 无定义，Frenet 标架与公式同时失效。对大多数正则曲线这只是一些孤立点，但严格处理时要在这些点分开讨论。

**辨析｜易错点 4：不要混淆「标架的方程」与「曲线的方程」。** Frenet 公式描述的是**标架**如何随 $s$ 运动；曲线本身的坐标由 $\alpha'(s)=\mathbf{T}(s)$ 再积分一次得到。标架微分方程组是一阶的，但曲线 $\alpha$ 的微分方程是三阶的（$\alpha'''$ 与 $\kappa,\tau$ 的关系），二者层级不同。

## 6 应用的远行：从 Frenet 到联络

Frenet 公式不是曲线论的终点，而是整个微分几何「局部标架方法」的起跑线：

**曲面论与联络**：曲面上取切平面标架，标架的导数写成自身乘矩阵——这个「标架的导数等于自身乘一个反对称矩阵」的模式，在曲面论里就是 **Gauss 公式**，在黎曼几何里就是 **Levi-Civita 联络**（第四篇与第八篇）。Frenet 公式可以看作一维的联络：它告诉我们「标架沿曲线怎么转」。
**机器人学与图形学**：样条路径的姿态插值、机械臂末端朝向的数值积分，都是把 Frenet 公式当作「运动方程」来数值求解；Darboux 向量正是路径的角速度。
**计算机图形学**：三维建模中沿曲线放样（sweep）、管道曲面（tube surface）的截面方向，几乎都基于 Frenet 标架。
**机器学习**：数据流形上「对齐不同点的局部坐标」——如流形学习中把邻域的局部基拼接起来——需要知道局部坐标沿路径如何旋转，这正是 Frenet/Darboux 思想在高维的推广。<span class="marginnote">这条线索在第四级《大模型原理》的「位置编码」与第三级《机器学习》的「流形假设」中都会以不同面目重现：给每个位置配一个会旋转的局部坐标系，是几何与序列建模共同的语言。</span>

## 7 小结

- **Frenet 公式**：$\mathbf{T}'=\kappa\mathbf{N}$，$\mathbf{N}'=-\kappa\mathbf{T}+\tau\mathbf{B}$，$\mathbf{B}'=-\tau\mathbf{N}$——标架随弧长 $s$ 的运动方程。
- **第二条方程是唯一需要推导的**：由 $\mathbf{N}=\mathbf{B}\times\mathbf{T}$ 求导并代入前两条，叉积化简即得。
- **矩阵形式**：$F' = A F$，$A$ 为反对称矩阵；存在 **Darboux 向量** $\boldsymbol\omega = \tau\mathbf{T}+\kappa\mathbf{B}$，标架绕它旋转，速率 $\sqrt{\kappa^2+\tau^2}$。
- **曲线论基本定理**：$\kappa(s)>0$、$\tau(s)$ 完全决定曲线形状（至多差一个刚体运动）——两个函数编码一条空间曲线。
- **易错**：公式只在弧长参数下成立；$\tau$ 的约定必须全篇统一；$\kappa=0$ 处失效；勿混淆标架方程与曲线方程。

在下一节《平面曲线的曲率与相对曲率》中，我们把三维的标架收回到二维平面里：平面曲线只有一个弯曲自由度，曲率有了**符号**（往左弯还是往右弯），Frenet 公式也随之退化成一个纯量方程——那是通向四顶点定理与等周不等式等整体结论的起点。

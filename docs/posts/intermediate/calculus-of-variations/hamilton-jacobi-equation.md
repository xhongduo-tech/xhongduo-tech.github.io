---
title: Hamilton-Jacobi 方程
date: 2026-08-11
---

# Hamilton-Jacobi 方程

<div class="epigraph">
<p>作用量作为初值位置的函数，满足一个一阶偏微分方程。</p>
<footer>—— 哈密顿与雅可比的发现</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 变分法 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Hamilton-Jacobi 方程开始

前面的变分问题都被归结为常微分方程：Euler-Lagrange 方程。但 Hamilton 与 Jacobi 发现，同样的问题还能被重新表述为一个**偏微分方程**——Hamilton-Jacobi 方程。这个翻转带来的不只是新方程，而是一种全新的世界观：把「选一条路径」换成「解一个场」，把极值曲线看作这个场里的「特征线」。<span class="marginnote">对照上一课《Hilbert 积分不变与场论方法》：场论里的斜率函数 $p(x,y)$ 恰好是 H-J 方程解的梯度。两条线索在这里接榫——H-J 方程正是场的「位势方程」。</span>

这一课依据 Evans Ch. 10。它是 Evans 教科书里唯一一章专门讲一阶非线性 PDE 的——因为 H-J 方程在最优控制、几何光学、波动前沿传播里无处不在，且它的现代解法（粘性解）是 Evans 本人的重要贡献。<span class="marginnote">对「从极限到大模型」的读者：H-J 方程是「Hamilton-Jacobi-Bellman（HJB）方程」的雏形——强化学习的「值函数满足贝尔曼方程」正是它在控制理论里的化身，见第三级《强化学习》。</span>

## 1 从拉格朗日到哈密顿：Legendre 变换

先把运动方程从「拉格朗日语言」切换到「哈密顿语言」。给定拉格朗日量 $L(q, \dot q)$，定义**哈密顿量（Hamiltonian）**

$$
H(q, p) = \max_{\dot q}\; \bigl[p\,\dot q - L(q, \dot q)\bigr]
$$

其中 $p$ 是「动量」坐标。这个极值运算正是 **Legendre-Fenchel 变换**（见本专题《凸分析与变分问题的对偶理论》）——$H$ 是 $L$ 关于 $\dot q$ 的共轭函数。<span class="marginnote">对 $L = \frac12 m\dot q^2 - V(q)$，求 $\max$ 得 $p = m\dot q$、$H = \frac{p^2}{2m} + V(q)$——总能量。哈密顿量守恒正是上一课《Noether 对称性定理》里时间平移对称的结果。</span>

拉格朗日方程等价地改写为**哈密顿正则方程**：

$$
\dot q = \frac{\partial H}{\partial p}, \qquad \dot p = -\frac{\partial H}{\partial q}
$$

两个一阶方程拼出二阶的 E-L 方程，把问题对称化、几何化。

## 2 作用量作为端点的函数

现在换一个视角：固定出发位置 $q_0$，对每个到达的时空点 $(t, x)$，考虑**最小作用量**——沿从 $(0, q_0)$ 到 $(t, x)$ 的极值曲线的作用量：

$$
S(t, x) = \int_0^t L\bigl(q(\tau), \dot q(\tau)\bigr)\, d\tau
$$

把它看成端点 $(t,x)$ 的函数，那么沿极值曲线有著名的两个公式：

$$
\frac{\partial S}{\partial x} = p, \qquad \frac{\partial S}{\partial t} = -H
$$

第一个：作用量对终点位置的偏导是终点的动量；第二个：对时间的偏导是负的哈密顿量。<span class="marginnote">这是 Hamilton 原理 + 变分学的「对端点求导」版本，也叫<strong>哈密顿-雅可比第一积分公式</strong>。$S$ 扮演「位势函数」，它的梯度就是动量场——与上一课 Mayer 场的位势 $S(x,y)$ 是同一个对象。</span>

## 3 Hamilton-Jacobi 方程

把 $p = S_x$ 代进 $\partial S/\partial t = -H$，得到**Hamilton-Jacobi 方程（含时）**：

$$
\boxed{\;S_t + H(x, S_x, t) = 0\;}
$$

对不显含时间的保守系统（$H$ 与 $t$ 无关），解常可分离为 $S = -E t + W(x)$，其中 $W$ 满足**定常 Hamilton-Jacobi 方程**

$$
H(x, \nabla W) = E
$$

这就是几何光学里的 **eikonal（程函）方程**。

**Hamilton-Jacobi 方程**：一阶非线性偏微分方程，解 $S$ 是「作用量场」，其梯度给出极值曲线的动量。<span class="marginnote">关键词是「场」：E-L 方程描述单条曲线，H-J 方程描述整片「曲线族」——这正是从常微分到偏微分、从一条到一族的那一步。</span>

## 4 公式解析：eikonal 方程 $|\nabla S|^2 = 1$

取自由粒子 $H = \frac{|p|^2}{2m}$，定常方程 $H(x,\nabla W) = E$ 化为

$$
|\nabla W|^2 = 2mE
$$

归一化后即**eikonal 方程** $|\nabla S|^2 = 1$。

三步拆解：

- **第一步，读几何**：$|\nabla S| = 1$ 说明 $S$ 的梯度是单位向量——$S$ 是「到某集合的符号距离」。它的**水平集** $S = \text{常数}$ 是等相位面（波前）。
- **第二步，读光学**：光沿垂直于波前的方向传播，即沿 $\nabla S$ 的方向——这些正交轨道就是**光线（rays）**。eikonal 方程把几何光学浓缩成「距离函数的梯度恒为单位」。
- **第三步，特征线与极值曲线重合**：H-J 方程的特征方程恰是哈密顿正则方程

$$
\dot x = H_p(x, p), \qquad \dot p = -H_x(x, p)
$$

沿特征线 $p = \nabla S$，特征线就是 E-L 方程的极值曲线——**变分法的路径在 H-J 方程里以特征线的身份复活**。<span class="marginnote">「特征线 = 极值曲线」是整章的关键换算：解 H-J 方程（一个 PDE）可以转化为追踪特征线（解 ODE）——而后者我们早已熟悉。这是从 Hamilton 原理到变分法的大回环。</span>

## 5 求解思路与粘性解

**完整积分（complete integral）**：H-J 方程的解往往不是一条光滑曲面，而是一族曲面。Jacobi 的方法：先求含参数 $n$ 的完整积分 $S(t, x; \alpha)$，再用包络（envelope）操作把参数消掉，得到真正满足初值条件的解。包络就是「一族曲面相切包出来的面」——波动前沿的几何意义。

但光滑解常常**不存在**：波前会交叉、形成冲击波（shock），$S$ 变为多值，古典解失效。<span class="marginnote">例：把 $S_0(x)$ 作为初值，短时间后特征线可能相交，$S$ 在交叉处变成「多值函数」。物理上这是激波、焦散（caustics）；数学上这意味着「没有处处光滑的解」——就像《Weierstrass 过分函数与角条件》里极值曲线会折断一样。</span>

**粘性解（viscosity solution）**（Crandall-Lions 1980s，Evans Ch. 10.1）给出现代答案：把 H-J 方程当作加了无穷小耗散 $-\varepsilon\Delta S_\varepsilon$ 的正则化问题的极限：

$$
S_\varepsilon^{\,t} + H(x, S_\varepsilon^x) = \varepsilon\, \Delta S_\varepsilon
$$

当 $\varepsilon \to 0^+$，$S_\varepsilon$ 收敛到唯一的粘性解——一个「处处下方式定义」的连续弱解。<span class="marginnote">「粘性」来自这个人工耗散项：像把一点粘稠度注入方程，选出「信息从正确方向传播」的那一支解。粘性解是 Evans 教科书的招牌内容，也是最优控制、图像分割（水平集方法）、以及金融数学里 HJB 方程的标准工具。</span>

对「从极限到大模型」的读者，粘性解与水平集方法正是现代计算前沿的骨架：eikonal 神经网络学习「符号距离场」、扩散模型的时间演化、最优传输的对数散度——H-J 方程和它的粘性解都在幕后。<span class="marginnote">扩散模型的前向/反向 SDE 与「沿特征线传播」同构；HJB 与强化学习的贝尔曼方程、最优控制的动态规划是同一数学家族。这条线会一路通到第三级。</span>

## 6 小结

- **哈密顿量** $H = \sup_{\dot q}[p\dot q - L]$ 是拉格朗日量的 **Legendre 共轭**；哈密顿正则方程重构运动方程。
- 作用量作为端点的函数满足 $S_x = p$、$S_t = -H$，从而推出 **Hamilton-Jacobi 方程** $S_t + H(x,S_x,t) = 0$。
- 定常情形给出 **eikonal 方程** $|\nabla S|^2 = 1$：水平集是波前，$\nabla S$ 是光线方向。
- H-J 方程的**特征线就是极值曲线**——ODE 与 PDE 两种语言在此会师。
- 光滑解可能不存在（冲击波），现代解法是**粘性解**：正则化 + 极限。

在下一节，我们把变分法推向二维与几何：**极小曲面问题与平均曲率**——面积泛函的极小者，正好是平均曲率为零的曲面。

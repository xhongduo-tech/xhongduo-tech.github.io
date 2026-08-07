---
title: 角动量算符与球谐函数
date: 2026-08-07
---

# 角动量算符与球谐函数

<div class="epigraph">
<p>角动量是量子力学里最优雅的结构——它由一组对易关系完全确定，不依赖任何具体实现。</p>
<footer>—— 尤金 · 维格纳（Eugene Wigner）</footer>
</div>

<div class="article-byline">
<p>第二级 · 量子力学 ｜ 曾谨言《量子力学》卷一 第5章 / Griffiths《量子力学概论》§4.3 ｜ 2026-08-07</p>
</div>

## 为什么从角动量开始

三维薛定谔方程里冒出了 $l$ 和 $m$，它们来自角动量算符的本征值问题。角动量是量子力学最重要的可观测量之一：它不但决定原子轨道与光谱，还通过「角动量耦合」连接原子、分子与原子核物理。本篇把角动量算符独立出来系统研究：先从经典定义出发，再建立它的**代数结构**（对易关系），最后把球谐函数作为它的共同本征函数完整呈现。<span class="marginnote">角动量的代数方法可以完全独立于波函数：只从对易关系 $[\hat{L}_i,\hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$ 出发，就能推出 $l$ 的半整数/整数取值——这使同一套理论同时适用于轨道角动量（整数）和自旋（半整数）。「代数决定一切」是这一节的方法论核心。</span>

## 1 轨道角动量算符

经典角动量 $\mathbf{L} = \mathbf{r} \times \mathbf{p}$。把位置与动量换成算符，得**轨道角动量算符**：

$$
\hat{\mathbf{L}} = \hat{\mathbf{r}} \times \hat{\mathbf{p}} = -i\hbar\,\mathbf{r}\times\nabla
$$

分量形式（球坐标下）：

$$
\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}, \qquad
\hat{L}_x = i\hbar\left(\sin\phi\frac{\partial}{\partial\theta} + \cot\theta\cos\phi\frac{\partial}{\partial\phi}\right), \qquad
\hat{L}_y = i\hbar\left(-\cos\phi\frac{\partial}{\partial\theta} + \cot\theta\sin\phi\frac{\partial}{\partial\phi}\right)
$$

其中 $\hat{L}_z = -i\hbar\frac{\partial}{\partial\phi}$ 特别简洁——它只对 $\phi$ 求导。<span class="marginnote">$\hat{L}_z = -i\hbar\partial_\phi$ 的简洁性来自球坐标里「绕 $z$ 轴旋转」只改变 $\phi$。它的本征函数 $e^{im\phi}$ 天然是平面波在角度方向的推广——这提示我们，角动量量子化本质上是「角度方向的周期性」带来的驻波条件，与一维势阱的 $k$ 量子化同构。</span>

**角动量算符是厄米的**：$\hat{\mathbf{L}}^\dagger = \hat{\mathbf{L}}$，所以它的本征值是实数，可对应物理测量。

## 2 角动量代数

三个分量之间满足**角动量代数（angular momentum algebra）**：

$$
[\hat{L}_x, \hat{L}_y] = i\hbar\hat{L}_z, \qquad
[\hat{L}_y, \hat{L}_z] = i\hbar\hat{L}_x, \qquad
[\hat{L}_z, \hat{L}_x] = i\hbar\hat{L}_y
$$

可统一写成 $[\hat{L}_i, \hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$（$\epsilon_{ijk}$ 是列维-奇维塔符号，循环求和）。<span class="marginnote">「角动量分量互不对易」的物理后果极其深刻：$L_x$、$L_y$、$L_z$ 不能同时精确测量。这与位置—动量的不确定度关系同源——旋转方向上的角动量分布也不可能三个轴都确定。唯一能与 $L_z$ 同时对易的组合是 $\hat{\mathbf{L}}^2$：$[\hat{L}^2, \hat{L}_z] = 0$。</span>核心事实是：

$$
[\hat{\mathbf{L}}^2, \hat{L}_z] = 0
$$

**总角动量平方与任一分量对易**——所以可以选 $(\hat{\mathbf{L}}^2, \hat{L}_z)$ 作为**共同本征函数的完备集**，用 $(l, m)$ 两个量子数标记。

## 3 阶梯算符与谱

引入**升降算符**（阶梯算符）：

$$
\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y
$$

它们的作用是把 $m$ 升/降一档：

$$
\hat{L}_\pm\,Y_l^m = \hbar\sqrt{l(l+1) - m(m\pm 1)}\,Y_l^{m\pm 1}
$$

系数 $\hbar\sqrt{l(l+1)-m(m\pm1)}$ 是**归一化因子**——它保证升降不破坏归一化。<span class="marginnote">这套「升降算符 + 谱的上下限」的逻辑与谐振子完全平行：$m$ 有上限 $l$ 和下限 $-l$（否则 $\hat{L}_+|m\rangle$ 越界），正定性 + 阶梯性联合锁死 $l$ 为整数、$m = -l,\dots,l$。角动量谱就这样从代数里「长」出来，无需解微分方程。</span>关键约束：$m$ 有界 ⟹ $l$ 为非负整数（轨道角动量），且 $|m| \le l$。由此：

- $\hat{\mathbf{L}}^2$ 的本征值：$l(l+1)\hbar^2$，$l = 0, 1, 2, \dots$
- $\hat{L}_z$ 的本征值：$m\hbar$，$m = -l, -l+1, \dots, l$（共 $2l+1$ 个）

## 4 公式解析：$\hat{L}_\pm$ 如何给出谱

把「从代数推出角动量谱」的核心一步拆开：

$$
\hat{L}_\pm = \hat{L}_x \pm i\hat{L}_y, \qquad \hat{L}_\pm Y_l^m = \hbar\sqrt{l(l+1)-m(m\pm1)}\,Y_l^{m\pm1}
$$

- **第一步，$\hat{L}_\pm$ 与 $\hat{L}_z$ 的对易**：$[\hat{L}_z, \hat{L}_\pm] = \pm\hbar\hat{L}_\pm$。这意味着若 $Y_l^m$ 是 $\hat{L}_z$ 的本征态（本征值 $m\hbar$），则 $\hat{L}_\pm Y_l^m$ 也是本征态，本征值 $(m\pm1)\hbar$——**升降算符改变 $m$，不改变 $l$**。
- **第二步，升降的规范**：$\hat{L}_+$ 把 $m$ 升到 $m+1$，$\hat{L}_-$ 降到 $m-1$。反复作用会产生 $m$ 的整条链。
- **第三步，链必须有限**：$m$ 若无限增大，$\langle Y_l^m|\hat{L}^2|Y_l^m\rangle = \langle Y_l^m|\hat{L}_z^2 + \frac12(\hat{L}_+\hat{L}_-+\hat{L}_-\hat{L}_+)|Y_l^m\rangle$ 将无法保持正定。正定性要求链在 $m=l$ 处截断：$\hat{L}_+Y_l^l = 0$。
- **第四步，归一化系数**：$\|\hat{L}_\pm Y_l^m\|^2 = \langle Y_l^m|\hat{L}_\mp\hat{L}_\pm|Y_l^m\rangle = \hbar^2[l(l+1)-m(m\pm1)]$，开方即得系数。

## 5 球谐函数与角动量

**球谐函数 $Y_l^m(\theta,\phi)$ 是 $\hat{\mathbf{L}}^2$ 与 $\hat{L}_z$ 的共同本征函数**：

$$
\hat{\mathbf{L}}^2 Y_l^m = l(l+1)\hbar^2\,Y_l^m, \qquad \hat{L}_z Y_l^m = m\hbar\,Y_l^m
$$

它们构成角度空间的一组完备正交基，任何角度函数都能展开：

$$
f(\theta, \phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} c_{lm}\,Y_l^m(\theta,\phi)
$$

这正是一维问题「展开到完备基」在球面上的版本。<span class="marginnote">球谐函数在物理里无处不在：原子轨道（s/p/d/f）、分子轨道、电磁多极辐射（偶极/四极）、宇宙微波背景辐射的各向异性图（CMB 的多极展开！），全用 $Y_l^m$ 表示。宇宙学里那张著名的「普朗克卫星微波图」的统计分析，就是把温度场按球谐函数展开，系数 $C_l$ 给出宇宙的角功率谱——一门课学完，居然直接接到宇宙学前沿。</span>化学里，$|Y_l^m|^2$ 直接画出电子云的角分布——s 轨道的球形、p 轨道的哑铃形、d 轨道的四叶形，全都是球谐函数的模样。

### 易错辨析

**辨析｜易错点：$\hat{\mathbf{L}}^2$ 的本征值是 $l(l+1)\hbar^2$，不是 $l^2\hbar^2$。** 角动量矢量长度 $|\mathbf{L}| = \hbar\sqrt{l(l+1)}$，严格大于 $l\hbar$。$l^2$ 的写法来自经典直觉，但量子角动量「永远不能完全对齐 $z$ 轴」正是这 $\sqrt{l(l+1)} > l$ 的体现——记成 $l^2$ 会丢掉垂直方向的残余分量。

**辨析｜易错点：$\hat{L}_\pm$ 改变 $m$，但不改变 $l$，也不改变能量。** $\hat{L}_\pm Y_l^m \propto Y_l^{m\pm1}$——它只在同一 $l$ 的 $2l+1$ 个 $m$ 态之间搬动，不跨 $l$。因此 $l$ 是「阶梯」的楼层号，$m$ 是同一楼层内的房间号。用升降算符升出「另一个 $l$」是不可能的。

**辨析｜易错点：$\hat{L}_\pm$ 的归一化系数里是 $\sqrt{l(l+1)-m(m\pm1)}$，不是 $\sqrt{l(l+1)-m^2}$ 之类的简化。** $m(m\pm1)$ 与 $m^2$ 差一个 $\pm m$。用错系数会让升降后的态既不正交也不归一。熟练记忆 $\hat{L}_\pm|lm\rangle = \hbar\sqrt{l(l+1)-m(m\pm1)}|l,m\pm1\rangle$ 的完整形式。

**辨析｜易错点：$m$ 有界（$|m|\le l$）是「代数自洽」的结果，不是外加假设。** 若 $m > l$，$\hat{L}_+|lm\rangle$ 的系数平方为负，正定性破坏。正是「链必须有端点」这个约束，联合对易关系唯一推出 $l$ 取整数、$m$ 从 $-l$ 到 $l$。把 $m$ 的取值范围当经验事实记，就丢了最漂亮的代数论证。

## 6 小结

- **轨道角动量算符** $\hat{\mathbf{L}} = \hat{\mathbf{r}}\times\hat{\mathbf{p}} = -i\hbar\,\mathbf{r}\times\nabla$，厄米、分量不对易：$[\hat{L}_i,\hat{L}_j] = i\hbar\epsilon_{ijk}\hat{L}_k$。
- $[\hat{\mathbf{L}}^2,\hat{L}_z] = 0$，选 $(\hat{\mathbf{L}}^2,\hat{L}_z)$ 为共同本征函数完备集，量子数 $(l,m)$。
- **升降算符** $\hat{L}_\pm = \hat{L}_x\pm i\hat{L}_y$ 使 $m$ 升降一档；谱的正定性与有限性联合推出 $l$ 整数、$m=-l,\dots,l$。
- 本征值：$\hat{\mathbf{L}}^2 \to l(l+1)\hbar^2$，$\hat{L}_z \to m\hbar$；每个 $l$ 有 $2l+1$ 个 $m$ 态。
- **球谐函数** $Y_l^m$ 是角空间完备正交基，在原子轨道、多极辐射、CMB 分析里普遍使用。

在下一节，我们把角动量代数应用到最经典的物理系统——**氢原子与类氢离子**，首次得到真实原子的能级。

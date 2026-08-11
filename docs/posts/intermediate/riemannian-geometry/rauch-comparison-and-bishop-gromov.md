---
title: Rauch 比较定理与体积比较：Bishop-Gromov
date: 2026-08-11
---

# Rauch 比较定理与体积比较：Bishop-Gromov

<div class="epigraph">
<p>比较定理把每个黎曼流形都放进一个陈列柜：摆好球面、欧氏空间和双曲空间当标尺，然后说——你的几何，长得像标尺里最温和的那一个。</p>
<footer>—— 彼得 · 彼得森（Peter Petersen）《黎曼几何》（Riemannian Geometry，3rd ed.，2016）（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 黎曼几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从比较定理开始

Bonnet-Myers 已经示范了「曲率下界 ⇒ 几何上界」的威力，但它只给出直径这种「粗糙」的控制。更精细的问题是：**给定曲率上界或下界，Jacobi 场多大？球的体积多大？测地线的聚散多快？** 答案由一个统一的方法给出——**比较几何（comparison geometry）**：把任意流形的曲率与常曲率空间（$S^n$、$\mathbb{R}^n$、$H^n$）的曲率比较，再用常曲率空间里可精确求解的 ODE 去**比较**控制目标流形的 ODE。

这个流派的两大支柱正是本节主题：

- **Rauch 比较定理**：控制**Jacobi 场的大小**（曲率上界/下界 ⟹ 相邻测地线的聚散更快/更慢）；
- **Bishop-Gromov 体积比较**：控制**测地球的体积**（Ricci 下界 ⟹ 体积不超过常曲率空间的体积）。

比较定理是「从极限到大模型」主线中最「可计算」的一环：体积增长的控制是概率测度集中、随机矩阵谱分析、以及深度网络宽度增长律背后的几何语言。<span class="marginnote">思维捷径：把比较定理想成「曲率就是二阶导数的符号」。$K \le \delta$ 意味着 Jacobi 场分量 $f'' + \delta f \le 0$，于是 $f$ 被「$\delta$ 空间里那个解」从上方控制。整个比较几何 = Sturm 比较定理的几何翻译。</span>

## 1 常曲率空间的 Jacobi 场：精确标尺

先在常曲率空间 $M_\delta$（$K\equiv\delta$）里把 Jacobi 场解出来。取单位测地线，$\delta$ 值决定解族：

$$
f'' + \delta\, f = 0
$$

$$
f(t) = \begin{cases}
\sin(\sqrt{\delta}\,t)/\sqrt{\delta}, & \delta > 0 \;(\text{球面}),\\
t, & \delta = 0 \;(\text{欧氏}),\\
\sinh(\sqrt{-\delta}\,t)/\sqrt{-\delta}, & \delta < 0 \;(\text{双曲}).
\end{cases}
$$

在球面上，$f$ 在 $t = \pi/\sqrt{\delta}$ 处回到零——出现共轭点；在欧氏空间线性增长；在双曲空间指数增长。**这三种「基准解」是全部比较定理的标尺。**<span class="marginnote">记忆口诀：三角函数「回来」、线性函数「匀速走」、双曲正弦「飞走」。曲率越负，测地线发散越猛——这正是负曲率流形「面积指数增长」的根源。</span>

## 2 Rauch 比较定理：控制相邻测地线的聚散

**定理（Rauch 比较定理）**：设 $\gamma_1, \gamma_2$ 是两条单位测地线，$J_1, J_2$ 是沿它们的 Jacobi 场（$J_i(0)=0$、$J_i'(0)$ 垂直于 $\gamma_i'$、$|J_1'(0)| = |J_2'(0)|$）。若对每个点、每个与切向量张成的二维方向都有

$$
K_{\gamma_1} \le \delta \le K_{\gamma_2}
$$

则在两者都未到达共轭点的范围内：

$$
\frac{|J_1(t)|}{|J_1'(0)|} \;\le\; \frac{f_\delta(t)}{f_\delta'(0)} \;\le\; \frac{|J_2(t)|}{|J_2'(0)|}
$$

其中 $f_\delta$ 是 $K\equiv\delta$ 空间的 Jacobi 场解。<span class="marginnote">读法：曲率越小（越负），Jacobi 场越长、相邻测地线散得越快；曲率越大（越正），Jacobi 场被压得越短、测地线聚拢越凶。把 $K_{\gamma_1}$ 与 $K_{\gamma_2}$ 换成常数 $\delta$ 上下夹住，就得到与常曲率空间的逐点比较。</span>

**推论（第一共轭点比较）**：$K \ge \delta > 0$ 的完备流形上，沿任意测地线的**第一个共轭点出现不晚于** $t = \pi/\sqrt{\delta}$——这正是 Bonnet-Myers 中「直径 $\le \pi/\sqrt{\kappa}$」的细粒度版本。

**辨析｜易错点：Rauch 比较的是**截面曲率**的上下界，且要求两端 Jacobi 场初值对齐。** 常见错误是把「Ricci 下界」误当成 Rauch 的前提（Ricci 下界对应的是 Bonnet-Myers 与体积比较）。Rauch 精细到每一个二维方向，信息最多，前提也最严格。

**辨析｜易错点：比较只在「无共轭点区间」内成立。** 一旦 $J_1$ 或 $J_2$ 碰到共轭点，两个 Jacobi 场不再「同域」，比较失去意义。这正是为什么比较定理常与「单射半径」「无共轭点」的讨论配套使用。

## 3 Bishop-Gromov 体积比较：Ricci 下界控制体积

**定理（Bishop-Gromov 体积比较）**：设 $(M,g)$ 完备，$\mathrm{Ric} \ge (n-1)\kappa$。记 $V_M(r) = \operatorname{vol}(B_p(r))$ 为 $p$ 处半径为 $r$ 的测地球体积，$V_\kappa(r)$ 为曲率 $\kappa$ 的常曲率空间里半径为 $r$ 的球体积。则比值 $V_M(r)/V_\kappa(r)$ 是 $r$ 的**单调递减**函数，即

$$
\frac{V_M(r_1)}{V_\kappa(r_1)} \;\ge\; \frac{V_M(r_2)}{V_\kappa(r_2)}, \qquad r_1 < r_2
$$

特别地，$V_M(r) \le V_\kappa(r)$（因为 $r \to 0$ 时两者比值为 1）。<span class="marginnote">直觉：$\mathrm{Ric}\ge(n-1)\kappa$ 意味着「体积被正曲率方向的收缩力挤压」，球长不大。等号只在 Ricci 恒等于下界（常曲率或局部对称极值）时达到——体积比较是「曲率下界 ⟹ 体积上界」的刚性陈述。</span>

推论（**Bishop 定理**）：$\mathrm{Ric} \ge (n-1)\kappa > 0$ 时，$V_M(r) \le V_\kappa(r)$ 且 $\lim_{r\to\pi/\sqrt{\kappa}} V_\kappa(r)$ 有限，重新给出「直径 $\le \pi/\sqrt{\kappa}$」。

**辨析｜易错点：Bishop-Gromov 用 Ricci 下界，Rauch 用截面曲率上下界。** 两者是不同粒度的工具：Ricci 只管「平均收缩」，适合体积这种标量观测；截面曲率管「每个方向」，适合 Jacobi 场这种向量观测。把 Ricci 下界用到 Rauch 上、把截面曲率用到体积上，都会得出错误结论。

**辨析｜易错点：单调性是「比值」，不是「体积本身」。** 不能断言 $V_M(r)$ 单调（球体积当然随 $r$ 增大）；单调的是**与标尺空间的比值**。这也解释了为什么 $H^n$（$\kappa=-1$，体积 $\sim e^{r}$）能「装下」比 $\mathbb{R}^n$ 多得多的体积——比值不再受压。

## 4 公式解析：Bishop-Gromov 的核心估计

$$
\frac{d}{dr}\left(\frac{V_M(r)}{V_\kappa(r)}\right) \le 0
$$

四步拆解：

- **第一步，$V_M(r)$ 的几何含义**：$V_M(r) = \int_{B_p(r)} \sqrt{\det g}\,dx$。用指数映射与极坐标，测地球的体积可写成「径向积分」：$V_M(r) = \int_{S^{n-1}}\int_0^r J_\theta(t)\,dt\,d\theta$，其中 $J_\theta$ 是沿方向 $\theta$ 的**体积 Jacobi 行列式**（$n-1$ 个正交 Jacobi 场的乘积）。

- **第二步，行列式满足的方程**：对正交 Jacobi 场组 $\{J_i\}$，令 $A = \log J_\theta$（行列式对数），Jacobi 方程结合 Jacobi 恒等式给出

$$
A'' + \frac{1}{n-1}\mathrm{Ric}(\gamma',\gamma') \le (A')^2
$$

Ricci 下界 $\ge (n-1)\kappa$ 把 $A$ 控制成「不高于常曲率解」。

- **第三步，为什么是比值单调**：把 $A \le A_\kappa$ 代入径向积分，$V_M(r)/V_\kappa(r)$ 是两个同初值积分之比；逐点被控制 ⟹ 比值随 $r$ 单调递减。**「逐点的曲率不等式」通过积分被放大成「整体的体积不等式」。**

- **第四步，刚性**：若比值在某处不再严格递减（出现等号），则所有不等式取等，$A \equiv A_\kappa$、$\mathrm{Ric}\equiv(n-1)\kappa$——流形局部等距于常曲率空间。**体积比较的等号情形是「刚性定理」的来源**，这与等周不等式、Gromov-Hausdorff 收敛的研究一脉相承。

## 5 比较几何的现代视野

- **Gromov 预紧**：由 Bishop-Gromov 的「体积比率单调」可推出，固定维数、Ricci 下界、直径上界的流形族是 Gromov-Hausdorff 预紧的——比较几何催生了「流形收敛」的理论，并通向正 Ricci 曲率的分类。
- **Heintze-Karcher、Günther** 等是 Rauch 族的各种变体，控制子流形焦点与测地线族的密度。
- **双曲体积定理**（Thurston）与「Mostow 刚性」都建立在体积比较的框架上。

**辨析｜易错点：比较定理给出的是「界」，不给出「精确几何」。** $K\le\delta$ 只保证 Jacobi 场不超过基准解，不保证流形等距于任何常曲率空间——除非出现等号（刚性）。**「界」与「刚性」是两回事**，把「有界」误读成「等距」是初学比较几何最常见的偏差。

## 6 小结

- **基准解**：常曲率空间的 Jacobi 场 $f_\delta$（三角 / 线性 / 双曲正弦），是比较几何的标尺。
- **Rauch 比较定理**：$K_{\gamma_1}\le\delta\le K_{\gamma_2}$ ⟹ Jacobi 场大小被 $f_\delta$ 夹住；第一个共轭点不晚于 $\pi/\sqrt{\delta}$。
- **Bishop-Gromov**：$\mathrm{Ric}\ge(n-1)\kappa$ ⟹ $V_M(r)/V_\kappa(r)$ 单调递减，$V_M(r)\le V_\kappa(r)$。
- 两种工具的粒度不同：**截面曲率**（Rauch）vs **Ricci**（体积、Bonnet-Myers）。
- 等号情形是刚性定理的温床；比较几何通向流形收敛理论（Gromov）。

在下一节，我们把「测地线 + 共轭点 + 指标形式」组装成黎曼几何里最优雅的有限维-无限维桥梁：**Morse 指标定理**——能量泛函在环路空间上的指标，恰好等于共轭点重数之和。

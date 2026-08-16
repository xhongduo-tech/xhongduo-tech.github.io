---
title: 庞特里亚金极大值原理（最小时间/燃料问题）
date: 2026-08-07
---

# 庞特里亚金极大值原理（最小时间/燃料问题）

<div class="epigraph">
<p>Everything should be made as simple as possible, but not simpler.（一切都应尽可能简单，但不能更简单。）</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein，常被引用的格言）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Kirk《Optimal Control Theory: An Introduction》Ch. 4 ｜ 2026-08-07</p>
</div>

## 为什么变分法还不够

上一节的变分法有个致命前提：控制 $u(t)$ 可以**自由光滑地变化**。但现实里控制几乎总是**受限**的：油门有最大开度、舵面有偏转限位、推进器有最大推力，记作 $u(t) \in \Omega$（约束集）。当最优解想让 $u$ 越界时，变分法的「内部取极值」条件失效——最优控制常常落在约束的**边界**上。<span class="marginnote">庞特里亚金（Lev Pontryagin，1908–1988）及其学派在 1950 年代给出了处理「控制有界」的完整理论。他本人 14 岁失明，却靠口头协作完成这一控制理论里程碑——<strong>极大值原理（maximum principle）</strong>。这套必要条件的优雅之处：<strong>把「对 $u$ 求极值」换成「对 $u$ 求最小化 Hamiltonian」</strong>，即使 $u$ 在约束集合上也能用。</span>

爱因斯坦的格言在这节的寓意特别贴切：受限最优控制的最优策略，常常「简单」到极致——油门不是全开就是全关，这就是**Bang-Bang 控制**。它不复杂，但绝不「更简单」。

## 1 Hamiltonian 与极大值原理的三件套

考虑最小化问题（控制受限 $u(t) \in \Omega$）：

$$
J = \phi(x(t_f)) + \int_{t_0}^{t_f} L(x, u, t)\,\mathrm{d}t, \qquad \dot{x} = f(x, u, t).
$$

定义**Hamiltonian 函数**

$$
H(x, u, \lambda, t) = L(x, u, t) + \lambda^T f(x, u, t).
$$

极大值原理给出的**必要条件**三件套：

1. **状态方程**：$\dot{x} = \frac{\partial H}{\partial \lambda} = f(x, u, t)$；
2. **协态方程**：$\dot{\lambda} = -\frac{\partial H}{\partial x}$；
3. **最优性条件（关键）**：对每个 $t$，最优控制 $u^*(t)$ 使 **$H$ 达到（关于 $u$ 的）最小值**：

$$
H(x^*, u^*, \lambda, t) \le H(x^*, u, \lambda, t), \qquad \forall\, u \in \Omega.
$$

注意第 3 条的表述是「**极小值**」却叫「极大值原理」——历史命名习惯（庞特里亚金原书考虑最大化），我们只需记住**本质是「Hamiltonian 关于 $u$ 取极值」**。<span class="marginnote">把第 3 条与变分法对比：变分法要求 $\partial H/\partial u = 0$（光滑极值），极大值原理要求「$H$ 在约束集 $\Omega$ 上取全局最小」（边界也可）。<strong>前者是后者的「内部解」特例</strong>——$u^*$ 落在 $\Omega$ 内部时两者一致，落在边界时极大值原理仍然成立。</span>

## 2 为什么会出现 Bang-Bang：线性控制

当 Hamiltonian 对控制 $u$ 是**线性的**时，极值原理会自然逼出开关型控制。设系统为

$$
\dot{x} = Ax + Bu, \qquad H = L + \lambda^T(Ax + Bu).
$$

若 $L$ 不含 $u$（如最小时间问题 $L = 1$），则

$$
H = \lambda^TAx + \lambda^TBu.
$$

关于 $u$ 的部分是 $\lambda^TBu$，而 $u \in [u_{\min}, u_{\max}]$。为使 $H$ 最小，只需把 $u$ 取到「与 $\lambda^TB$ 符号相反」的端点：

$$
u^*(t) =
\begin{cases}
u_{\max}, & \lambda^TB < 0, \\
u_{\min}, & \lambda^TB > 0.
\end{cases}
$$

**控制只取两个极端值，切换时刻由 $\lambda^TB$ 过零决定**——这就是 Bang-Bang 控制。函数 $\lambda^TB$ 称为**开关函数（switching function）**。<span class="marginnote">Bang-Bang 的直觉：最小时间问题里，<strong>慢就是最大的浪费</strong>，所以每时每刻都要用最大油门（或最大刹车），只在「该减速的时刻」切换。这正对应赛车手「要么全油门要么全刹车」的驾驶方式。</span>

## 3 最小时间控制：双积分器的完整求解

最经典的最小时间问题是**双积分器**（车在无摩擦轨道上，推力 $u$ 有界 $|u| \le 1$）：

$$
\ddot{y} = u, \qquad x_1 = y,\; x_2 = \dot{y},\;
\dot{x}_1 = x_2,\; \dot{x}_2 = u.
$$

目标：从 $(x_1, x_2)$ 到原点的最小时间。Hamiltonian $H = 1 + \lambda_1 x_2 + \lambda_2 u$，开关函数 $\lambda_2$。协态方程 $\dot\lambda_1 = 0$，$\dot\lambda_2 = -\lambda_1$，解出 $\lambda_2(t)$ 是**线性函数**——至多过零一次。于是控制**至多切换一次**，形式为「全开 → 全关」（或反过来）。

把 $u = +1$ 与 $u = -1$ 的相轨线族画出来：$u = +1$ 时轨线是抛物线族 $x_1 = \frac12 x_2^2 + c$（开口朝右），$u = -1$ 时 $x_1 = -\frac12 x_2^2 + c$。能直接到原点的两条特殊轨线构成**开关曲线（switching curve）**：

$$
\Gamma: \; x_1 = -\frac12 x_2|x_2|.
$$

**最优策略：若初始点在 $\Gamma$ 上方先 $u=-1$ 再 $u=+1$，在 $\Gamma$ 下方先 $u=+1$ 再 $u=-1$**，即在 $\Gamma$ 上切换一次。<span class="marginnote">双积分器最小时间问题几乎是每个最优控制课程的「必修例题」，因为<strong>它把抽象的极大值原理落到了一幅可画的相图</strong>。它的解——抛物线开关曲线——也是自动驾驶「最短时间换道」、伺服定位「最快归位」等应用的数学原型。</span>

## 4 最小燃料问题：Bang-Off-Bang

若成本改为最小燃料 $J = \int_{t_0}^{t_f} |u(t)|\,\mathrm{d}t$（$|u| \le 1$），Hamiltonian 变为 $H = |u| + \lambda_1 x_2 + \lambda_2 u$。关于 $u$ 的最小化给出三值逻辑：

$$
u^*(t) =
\begin{cases}
-1, & \lambda_2 > 1, \\
0, & |\lambda_2| < 1, \\
+1, & \lambda_2 < -1.
\end{cases}
$$

**中间出现了「滑行段」（coast，$u = 0$）**——燃料昂贵时，最优策略在「全开」「全关」「熄火滑行」之间切换，即 **Bang-Off-Bang 控制**。<span class="marginnote">为什么会有滑行段？因为燃料成本 $|u|$ 对 $u$ 在 0 处不可导、且「松油门不花钱」——<strong>只要开关函数幅值不足，最优就选择「不作为」</strong>。这个「代价足够大时最优解是休息」的结论，是经济学里「最优不做」的数学版本。</span>

**对比视角**：最小时间 ↔ 最小燃料，一个「总想全力冲刺」，一个「能省则省」。两者都是极大值原理在受限线性控制下的产物，只是成本函数不同导致开关函数阈值不同。第 4 篇《时间最优与最省燃料控制》已给过详细解法，本节强调其统一性：**都源于「Hamiltonian 关于 $u$ 在约束集上取极值」这一条**。

## 5 公式解析：开关函数为什么「至多切换一次」

把「双积分器最小时间至多切换一次」这个结论拆开：

$$
u^* = -\operatorname{sgn}(\lambda_2(t)), \qquad
\lambda_2(t) = \lambda_2(t_0) - \lambda_1(t_0)(t - t_0).
$$

- **第一步，写出开关函数**：$H = 1 + \lambda_1 x_2 + \lambda_2 u$，$u$ 前面的系数是 $\lambda_2$。由极值条件，$u^* = -\operatorname{sgn}(\lambda_2)$（使 $H$ 最小）。
- **第二步，解协态方程**：$\dot\lambda_1 = -\partial H/\partial x_1 = 0$，故 $\lambda_1 = \text{常数}$；$\dot\lambda_2 = -\partial H/\partial x_2 = -\lambda_1$，积分得 $\lambda_2(t)$ 是关于 $t$ 的**一次函数**（直线）。
- **第三步，数零点**：一次函数至多穿过零点一次，因此开关函数 $\lambda_2(t)$ 至多改变一次符号——控制 $u^*$ **至多切换一次**。
- **第四步，几何落地**：切换时刻就是轨线与开关曲线 $\Gamma$ 的交点；分段 $u = \pm 1$ 的抛物线拼接就是最优轨线。**从「一次函数的零点」到「相图的开关曲线」，整条推理严丝合缝。**

这条「数零点」的技巧具有普遍意义：**最优控制的开关次数，常常由协态方程解的结构决定**。协态是多项式的，切换次数就有上界——这也是为什么「先解协态、再看开关函数」是处理 Bang-Bang 问题的标准套路。<span class="marginnote">关于「奇异弧」有一个易错提醒：若开关函数在一段区间上恒为零，会出现「奇异控制」（如最小时间里的滑行段、最小能量里的中间弧）。<strong>那不是「不切换」，而是「切换函数恒零、$u$ 由高阶条件决定」</strong>——奇异最优控制的处理是另一门精细手艺。</span>

## 6 小结

- 变分法要求 $u$ 光滑可取极值；**控制有界时最优解常落在边界**，需要极大值原理。
- 极大值原理三件套：**状态方程 + 协态方程 + 「$H$ 关于 $u$ 在约束集上取极值」**。
- $H$ 关于 $u$ 线性时出现 **Bang-Bang**：控制只在两个极端间切换，由开关函数过零决定。
- **双积分器最小时间**问题给出抛物线开关曲线 $\Gamma: x_1 = -\frac12 x_2|x_2|$，至多切换一次。
- **最小燃料**问题把 $|u|$ 的成本引入，出现 Bang-Off-Bang 三段式（全开/熄火/全关）。
- 切换次数由**协态方程解的结构**决定；「先解协态、再看开关函数」是通用套路。

在下一节，极大值原理给的是「必要条件 + 两点边值问题」，求解麻烦；动态规划则用「递推」给出另一条更直观、可数值化的路径——**动态规划与 LQR/LQG（HJB 方程、卡尔曼滤波）**。

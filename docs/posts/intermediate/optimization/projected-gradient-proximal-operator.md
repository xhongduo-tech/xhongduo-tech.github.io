---
title: 投影梯度法与邻近算子
date: 2026-08-07
---

# 投影梯度法与邻近算子

<div class="epigraph">
<p>走一步，若出了界，就把自己拉回来。</p>
<footer>—— 类比自投影梯度法的「走与拉」机制</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优化理论 ｜ 《最优化方法》约束优化章、Boyd《Convex Optimization》§9、Parikh & Boyd《Proximal Algorithms》 ｜ 2026-08-07</p>
</div>

## 为什么「投影」与「邻近」是一对孪生工具

处理约束与不可微目标，两大基本工具都源于同一个操作：**在集合上投影（projection）**或更一般的**邻近算子（proximal operator）**。投影梯度法把「沿负梯度走一步，然后投影回可行域」——处理**约束**；邻近梯度法把「沿梯度走一步，然后做邻近映射」——处理**不可微项**。两者在数学上是近亲：**投影是邻近算子取指示函数时的特例**。<span class="marginnote">这套「先走步、再修正」的模板在「从极限到大模型」里无处不在：深度学习里「先梯度更新、再投影/裁剪到约束」的各类算法，稀疏优化的软阈值，图像处理的 TV 去噪——全是投影/邻近思想的变体。它也是 ADMM 闭式子问题的底层语言。</span>

## 1 投影梯度法

对带约束问题 $\min_{x \in \mathcal{C}} f(x)$（$\mathcal{C}$ 凸闭集），**投影梯度法（projected gradient method）**的迭代是

$$
x_{k+1} = \Pi_{\mathcal{C}}\big(x_k - t_k \nabla f(x_k)\big)
$$

其中 $\Pi_\mathcal{C}$ 是**欧氏投影**：$\Pi_\mathcal{C}(v) = \arg\min_{x \in \mathcal{C}}\|x - v\|_2^2$。<span class="marginnote">几何直觉：梯度步在无约束空间里「走」，若走出 $\mathcal{C}$ 就垂直拉回最近的点。投影后的点一定可行，且「走 + 拉」整体仍然朝向目标下降（在凸 $\mathcal{C}$ 与合适步长下）。常见闭式投影：盒子约束 $[l,u]$ 的投影是逐分量截断；球 $\|x\|\le r$ 的投影是缩放；半空间、多面体的投影通常无闭式（需二次规划）。</span>收敛性：与梯度下降同型——光滑强凸时线性，一般凸时 $O(1/k)$。**投影是「把不可行的梯度步拉回可行域」的最小修正**。

## 2 邻近算子

**邻近算子（proximal operator）**处理复合目标 $\min f(x) + g(x)$，其中 $f$ 光滑、$g$ 不可微。定义

$$
\mathrm{prox}_g(v) = \arg\min_x\ \frac12\|x - v\|_2^2 + g(x)
$$

它回答：「站在 $v$ 附近，如何最小化 $g$ 而不离 $v$ 太远？」——**$g$ 被「局部化」到一个以 $v$ 为中心的问题**。三个经典例子：

- $g = 0$：$\mathrm{prox}(v) = v$（恒等）。
- $g = \lambda\|\cdot\|_1$：软阈值 $\mathcal{S}_\lambda(v)$（上一节）。
- $g = I_\mathcal{C}$（指示函数：域内 0、域外 $+\infty$）：$\mathrm{prox}_{I_\mathcal{C}}(v) = \Pi_\mathcal{C}(v)$——**投影是邻近算子的特例**！<span class="marginnote">把约束写成指示函数 $I_\mathcal{C}$，邻近算子自动退化为投影。这个统一让「投影梯度」与「邻近梯度」是同一套理论的两种装扮：约束 = 指示函数的不可微项。邻近算子的性质（非扩张性、与次梯度的关系）让它有干净的收敛理论。</span>

## 3 邻近梯度法（ISTA）

**邻近梯度法（proximal gradient method）**处理

$$
\min_x\ f(x) + g(x), \qquad f \text{ 光滑}，\ g \text{ 闭凸（可不光滑）}
$$

迭代：

$$
x_{k+1} = \mathrm{prox}_{t_k g}\big(x_k - t_k \nabla f(x_k)\big)
$$

**先对光滑部分做梯度步，再用邻近算子处理非光滑部分**。经典实例是 ISTA（Iterative Shrinkage-Thresholding Algorithm）：$f = \frac12\|Ax-b\|^2$、$g = \lambda\|\cdot\|_1$ 时，迭代 = 梯度步 + 软阈值，逐分量闭式。<span class="marginnote">收敛性：$f$ 光滑（$L$-Lipschitz 梯度）+ $g$ 闭凸时，固定步长 $t = 1/L$ 给出 $O(1/k)$ 收敛；强凸时线性。<strong>FISTA</strong>（加速版）用动量把一般凸的收敛提到 $O(1/k^2)$——与 Nesterov 加速梯度同源。这套「光滑梯度 + 非光滑邻近」的二分法，是今天稀疏优化与正则化学习的标准武器。</span>

## 4 公式解析：软阈值从次梯度推出

把 $\ell_1$ 的邻近算子用**次梯度**推导一遍，展示「邻近 = 次梯度方程的解」。对 $g(z) = \lambda\|z\|_1$：

**第一步，写最优性条件**：$z^* = \mathrm{prox}_{\lambda\|\cdot\|_1}(v)$ 当且仅当 $0 \in z^* - v + \lambda\partial\|z^*\|_1$，即 $v - z^* \in \lambda\partial\|z^*\|_1$。
**第二步，逐分量写次梯度**：$\partial|z_i| = \begin{cases} \{1\}, & z_i > 0 \\ [-1,1], & z_i = 0 \\ \{-1\}, & z_i < 0 \end{cases}$。
**第三步，解三个分支**：$z_i > 0$ ⇒ $v - z_i = \lambda$ ⇒ $z_i = v - \lambda$（需 $v > \lambda$）；$z_i < 0$ ⇒ $z_i = v + \lambda$（需 $v < -\lambda$）；$z_i = 0$ ⇒ $v \in [-\lambda, \lambda]$。
**第四步，合并**：$z_i^* = \mathrm{sign}(v)\max\{|v| - \lambda, 0\}$——软阈值。<span class="marginnote">这条推导的关键动作是「用次梯度含 0 的条件解邻近方程」——它把「最小化一个含不可微项的函数」变成「解一个含集值映射的方程」。这套语言让非光滑优化不再靠直觉，而是有严格方程可解。记住：$\ell_1$ 次梯度在 0 处是区间 $[-1,1]$，这是「稀疏解」产生的机制——解被「钉」在 0 上。</span>

**要点：邻近算子是「次梯度方程的解」，软阈值是它在 $\ell_1$ 下的显式形态**——理解次梯度，就理解为什么稀疏解出现在 0 处。

## 5 投影/邻近方法速查

| 方法 | 迭代 | 处理对象 | 收敛 |
| --- | --- | --- | --- |
| 投影梯度 | $\Pi_\mathcal{C}(x - t\nabla f)$ | 约束 $\mathcal{C}$ | 线性 / $O(1/k)$ |
| 邻近梯度（ISTA） | $\mathrm{prox}_{tg}(x - t\nabla f)$ | 不可微 $g$ | 线性 / $O(1/k)$ |
| FISTA | 加动量 | 同上 | $O(1/k^2)$ |
| ADMM 子问题 | 邻近/投影交替 | 分块复合 | 渐进 |

**辨析｜易错点：**其一，投影/邻近梯度要求 $f$ **光滑**、$g$ 或 $\mathcal{C}$ **闭凸**——$f$ 不可微时要用双邻近或次梯度法；其二，投影算子对「非凸集」不唯一、无收敛保证；其三，步长 $t$ 受 $f$ 的 Lipschitz 常数约束（$t \le 1/L$ 保收敛），工程上常用回溯线搜索；其四，「投影后目标不降」可能发生（在非凸 $\mathcal{C}$），凸情形才是安全的。

## 6 小结

- **投影梯度法**：$x \leftarrow \Pi_\mathcal{C}(x - t\nabla f)$——走步 + 拉回可行域。
- **邻近算子**：$\mathrm{prox}_g(v) = \arg\min \frac12\|x-v\|^2 + g(x)$；投影是 $g = I_\mathcal{C}$ 的特例。
- **邻近梯度法（ISTA）**：光滑梯度步 + 非光滑邻近步；FISTA 加速到 $O(1/k^2)$。
- **软阈值**：$\ell_1$ 的邻近算子，从次梯度方程推出，产生稀疏解。
- **统一视角**：约束 = 指示函数，投影与邻近是同一理论的两张脸。

在下一节，我们把约束优化的算法链收尾——**序列二次规划（SQP）初步**。

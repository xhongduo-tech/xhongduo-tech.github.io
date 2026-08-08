---
title: 非线性方程组的牛顿法简介
date: 2026-08-07
---

# 非线性方程组的牛顿法：把一维的切线推广成多维的切平面

<div class="epigraph">
<p>多维世界里的牛顿，用雅可比矩阵画出整个切平面。</p>
<footer>—— 从一维到多维的牛顿跳跃</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§8.6 ｜ 2026-08-07</p>
</div>

## 为什么从非线性方程组开始

一维求根解决 $f(x)=0$；现实问题几乎都是**方程组**：电路工作点、化学反应平衡、机械系统静平衡、机器学习模型的固定点——全是多个非线性方程联立。**非线性方程组**

$$
\mathbf{F}(\mathbf{x}) = \mathbf{0}, \qquad \mathbf{F}:\mathbb{R}^n\to\mathbb{R}^n
$$

的求解是科学与工程的核心引擎。**多维牛顿法**把一维的「切线」推广为多维的「切平面」，用雅可比矩阵驱动迭代——它是所有非线性求解器的母体（包括隐式微分方程、优化内层）。<span class="marginnote">一维牛顿 $x_{k+1}=x_k-\dfrac{f(x_k)}{f'(x_k)}$ 的分子分母都是「数」；多维时分子变成「向量 $\mathbf{F}$」，分母变成「矩阵 $\mathbf{J}$」，相除变成「解线性方程组 $\mathbf{J}\mathbf{s}=-\mathbf{F}$」。<strong>「除以导数」在多维世界 = 「解雅可比线性系统」</strong>——这是从一维到多维最核心的翻译。</span>

本节给出多维牛顿法、雅可比矩阵、收敛性与实现要点。

## 1 多维牛顿法

**雅可比矩阵（Jacobian）**：

$$
\mathbf{J}(\mathbf{x}) = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \dfrac{\partial f_n}{\partial x_1} & \cdots & \dfrac{\partial f_n}{\partial x_n} \end{pmatrix}
$$

**多维牛顿法**：每步解线性方程组

$$
\mathbf{J}(\mathbf{x}^{(k)})\,\mathbf{s}^{(k)} = -\mathbf{F}(\mathbf{x}^{(k)})
$$

再更新 $\mathbf{x}^{(k+1)}=\mathbf{x}^{(k)}+\mathbf{s}^{(k)}$。**核心结构：每步 = 组装雅可比 + 解一个线性系统**。

**几何**：$\mathbf{F}$ 在 $\mathbf{x}^{(k)}$ 处的切平面（由 $\mathbf{J}$ 给出）与「零平面」的交点给出下一步——多维的「切线跳跃」。

**数值例子**：解
$$
\begin{cases}
x^2+y^2=4 \\
x^2-y^2=1
\end{cases}
\Rightarrow \mathbf{F}(x,y)=\begin{pmatrix}x^2+y^2-4\\x^2-y^2-1\end{pmatrix}, \quad \mathbf{J}=\begin{pmatrix}2x&2y\\2x&-2y\end{pmatrix}
$$

初值 $(2,1)$：$\mathbf{F}=(1,2)^\top$，$\mathbf{J}=\begin{pmatrix}4&2\\4&-2\end{pmatrix}$。解 $\mathbf{J}\mathbf{s}=-\mathbf{F}$ 得 $\mathbf{s}\approx(-0.375,-0.25)$，$\mathbf{x}^{(1)}\approx(1.625,0.75)$。迭代收敛到 $(\sqrt{2.5},\sqrt{1.5})\approx(1.581,1.225)$——**3~4 步二次收敛**。<span class="marginnote">观察每步：<strong>「解一次线性系统」是牛顿步的成本核心</strong>——$n$ 维问题每步 $O(n^3)$（若满稠密）。这就是多维牛顿「贵」的来源，也是大规模非线性求解要「拟牛顿」（近似雅可比）的原因。</span>

## 2 收敛性与条件

**定理（多维牛顿局部二次收敛）。** 设 $\mathbf{F}\in C^2$，$\mathbf{x}^*$ 是 $\mathbf{F}$ 的简单零点（$\mathbf{J}(\mathbf{x}^*)$ 可逆），初值足够靠近，则牛顿迭代二次收敛：

$$
\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^*\rVert \le C\,\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert^2
$$

**与一维完全同构**：简单零点（雅可比可逆）+ 初值够近 = 二次收敛；雅可比奇异或初值远 = 可能发散。

**公式解析：二次收敛从哪来。**

- **第一步，泰勒展开（向量版）。** $\mathbf{F}(\mathbf{x}^*)=\mathbf{F}(\mathbf{x}^{(k)})+\mathbf{J}(\mathbf{x}^{(k)})(\mathbf{x}^*-\mathbf{x}^{(k)})+O(\lVert\mathbf{e}\rVert^2)$。
- **第二步，代入牛顿步。** $\mathbf{x}^{(k+1)}=\mathbf{x}^{(k)}-\mathbf{J}^{-1}\mathbf{F}(\mathbf{x}^{(k)})$，相减得 $\mathbf{e}^{(k+1)}=\mathbf{e}^{(k)}-\mathbf{J}^{-1}\mathbf{F}(\mathbf{x}^{(k)})$。
- **第三步，合并。** 用泰勒展开代回：$\mathbf{e}^{(k+1)}=O(\lVert\mathbf{e}^{(k)}\rVert^2)$——**二次收敛**。

## 3 工程挑战：雅可比从哪来

多维牛顿法的工程难点不在公式，在**雅可比矩阵**：

| 来源 | 做法 | 代价 |
| --- | --- | --- |
| 解析 | 手推偏导公式 | 准确但繁琐 |
| **有限差分** | $\dfrac{\partial f_i}{\partial x_j}\approx\dfrac{f_i(x+h_j)-f_i(x)}{h}$ | 免推导但 $O(n)$ 次求值 |
| 自动微分 | AD 框架（如 JAX） | 现代首选 |
| **拟牛顿** | 用相邻迭代近似更新雅可比（Broyden） | 省去求导但降收敛阶 |

**有限差分的坑**：$n$ 维雅可比需要 $n$ 次额外求值（每列一次）——**$n$ 大时每步成本爆炸**。拟牛顿（Broyden 等）用「秩一更新」近似雅可比，每步只有 $O(n^2)$，但收敛阶降到超线性（约 1.618）——**在「算不动雅可比」时是标准解**。<span class="marginnote">工程权衡：<strong>「精确雅可比 + 二次收敛」 vs 「近似雅可比 + 超线性」</strong>——大 $n$ 时拟牛顿胜（每步便宜），小 $n$ 或雅可比好算时用精确牛顿。现代求解器（如 PETSc、SciPy 的 $n$）两者都提供，按问题规模选。</span>

## 4 实现框架

```python
import numpy as np

def newton_system(F, J, x0, tol=1e-10, max_iter=50):
    """多维牛顿法：每步解 J(x) s = -F(x)，再 x ← x + s。"""
    x = np.array(x0, dtype=float)
    for it in range(max_iter):
        Fx = F(x)
        s = np.linalg.solve(J(x), -Fx)
        x = x + s
        if np.linalg.norm(s, ord=np.inf) < tol and np.linalg.norm(Fx, ord=np.inf) < tol:
            return x, it + 1
    return x, max_iter

# 例：x²+y²=4, x²-y²=1 → (√2.5, √1.5)
F = lambda v: np.array([v[0]**2 + v[1]**2 - 4, v[0]**2 - v[1]**2 - 1])
J = lambda v: np.array([[2*v[0], 2*v[1]], [2*v[0], -2*v[1]]])
print(newton_system(F, J, [2.0, 1.0]))   # 3 步到 1e-10
```

**终止准则**：残差范数 $\lVert\mathbf{F}\rVert_\infty$ 或步长 $\lVert\mathbf{s}\rVert$。**双准则**（残差 + 步长）防「残差小但位置远」的重根病（与一维一致）。

## 5 多维牛顿与一维的对照 + 延伸

| 判据 | 一维牛顿 | 多维牛顿 |
| --- | --- | --- |
| 迭代 | $x_{k+1}=x_k-\dfrac{f}{f'}$ | $\mathbf{x}_{k+1}=\mathbf{x}_k-\mathbf{J}^{-1}\mathbf{F}$ |
| 「除法」 | 除以标量 | 解线性系统 |
| 收敛阶 | 二次 | 二次（简单根） |
| 重根 | 降为线性 | 降为线性（雅可比奇异） |
| 每步成本 | $O(1)$ | $O(n^3)$（解系统） |

**延伸**：多维牛顿是**优化算法**的心脏——求 $\nabla g=0$ 就是解非线性方程组（$\mathbf{F}=\nabla g$，$\mathbf{J}=\nabla^2g$ 海森阵）。**「牛顿法求解方程组」与「牛顿法做优化」是同一套数学**——第二级《最优化理论》与第三级《机器学习》里还会重逢。<span class="marginnote">一条主线贯穿：<strong>求根 → 求解方程组 → 优化 → 机器学习</strong>——全是「用局部线性化（切线/切平面/梯度）反复逼近不动点」。多维牛顿是这条主线的枢纽。</span>

## 6 小结

- **多维牛顿法**：$\mathbf{J}(\mathbf{x}^{(k)})\mathbf{s}^{(k)}=-\mathbf{F}(\mathbf{x}^{(k)})$，$\mathbf{x}^{(k+1)}=\mathbf{x}^{(k)}+\mathbf{s}^{(k)}$——每步「解线性系统」。
- **雅可比矩阵** $\mathbf{J}$：偏导矩阵，可由解析、有限差分、自动微分或拟牛顿获得。
- **二次收敛**（简单零点 + 初值够近）：$\lVert\mathbf{e}^{(k+1)}\rVert\le C\lVert\mathbf{e}^{(k)}\rVert^2$。
- 工程：雅可比来源决定成本；大 $n$ 用拟牛顿（超线性）、小 $n$ 用精确牛顿。
- 一维「除以导数」= 多维「解线性系统」；多维牛顿是优化与机器学习的内核。

至此，非线性方程求根的十一章写完了。下一章进入最后一个主题：**常微分方程数值解法**——从欧拉方法开始，认识 RK 与线性多步法，把微分方程变成可计算的离散世界。

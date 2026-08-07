---
title: 逐次超松弛迭代法（SOR method）与松弛因子的选择
date: 2026-08-07
---

# SOR 迭代法：给收敛踩一脚油门

<div class="epigraph">
<p>迭代太慢时，不要增加步数——而是调整步幅。</p>
<footer>—— 松弛方法的智慧</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§6.3 ｜ 2026-08-07</p>
</div>

## 为什么从 SOR 迭代法开始

高斯-赛德尔迭代有时收敛得「太稳、太慢」——每步只朝真解挪一小步。**逐次超松弛迭代法（Successive Over-Relaxation，SOR）** 在高斯-赛德尔基础上引入一个**松弛因子** $\omega$，把每次更新的步子拉长（$\omega>1$）或缩短（$\omega<1$）。选对 $\omega$，收敛速度可以成倍提升——这是 1950 年代数值线性代数的里程碑之一，也让大型 PDE 求解从「可等」变成「快」。<span class="marginnote">SOR 的松弛思想来自「外推」：高斯-赛德尔的更新 $\Delta x_i$ 太小，乘一个因子 $\omega$ 让它「超调」一点。<strong>$\omega=1$ 退化回高斯-赛德尔；$\omega>1$ 是超松弛（常用），$\omega<1$ 是亚松弛</strong>（处理不易收敛的系统）。</span>

本节给出 SOR 的分量与矩阵形式、收敛定理、以及最优 $\omega$ 的选择。

## 1 分量形式：超调一步

SOR 迭代分两步走（对每个 $i$）：先算高斯-赛德尔式的「临时值」，再**外推**：

$$
\tilde{x}_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i}a_{ij}x_j^{(k+1)} - \sum_{j>i}a_{ij}x_j^{(k)}\right)
$$

$$
x_i^{(k+1)} = x_i^{(k)} + \omega\left(\tilde{x}_i^{(k+1)} - x_i^{(k)}\right)
$$

合并成一个公式：

$$
x_i^{(k+1)} = (1-\omega)x_i^{(k)} + \frac{\omega}{a_{ii}}\left(b_i - \sum_{j<i}a_{ij}x_j^{(k+1)} - \sum_{j>i}a_{ij}x_j^{(k)}\right)
$$

**$\omega>1$ 时，更新量被放大——每步「超调」到可能超过真解，然后回弹**；振荡收敛通常比单调逼近快得多。<span class="marginnote">物理直觉：收敛慢时，误差沿某些方向「凝固」不动，超松弛刻意打乱这种凝固，<strong>用轻微过冲换更快的整体收敛</strong>。这和「动量」在优化中的角色异曲同工——SGD 加动量也是把步子拉长、冲出平坦区。</span>

**数值例子**（同前方程组 $10x_1-x_2-x_3=6$ 等，初值零，$\omega=1.1$）：

- $x_1^{(1)}=(1-1.1)\cdot0+\dfrac{1.1}{10}(6)=0.66$（比高斯-赛德尔的 0.6 更远）
- $x_2^{(1)}=0+\dfrac{1.1}{10}(8-0.66)=0.807$
- $x_3^{(1)}=0+\dfrac{1.1}{10}(8-0.66-0.807)=0.719$

比高斯-赛德尔的第一轮更「激进」，向真解 $(1,1,1)$ 扑得更远。

## 2 矩阵形式与分裂

SOR 的分裂为

$$
M_\omega = \frac{1}{\omega}D - L, \qquad N_\omega = \left(\frac{1}{\omega}-1\right)D + U
$$

迭代矩阵 $G_\omega=M_\omega^{-1}N_\omega$。$\omega=1$ 时回到高斯-赛德尔（$M=D-L$）。<span class="marginnote">记忆：<strong>SOR 把 $D$ 除以 $\omega$、再把「剩下的 $\omega$」塞进 $N$</strong>——$M$ 与 $N$ 的 $\omega$ 对称分布，保证 $\omega=1$ 时恒等回退。</span>

## 3 公式解析：收敛定理与最优松弛因子

**定理（Kahan）。** 对任意矩阵，SOR 迭代收敛的必要条件是 $0<\omega<2$。

**定理（Ostrowski-Reich）。** 若 $A$ 对称正定，则 $0<\omega<2$ 也是充分条件——**SPD 矩阵上 $0<\omega<2$ 保证 SOR 收敛**。

**最优松弛因子**：当 $A$ 是对称正定且**具有一致性性质**（consistently ordered，如很多 PDE 离散矩阵）时，存在解析公式：

$$
\omega_{\mathrm{opt}} = \frac{2}{1+\sqrt{1-\rho(G_J)^2}}
$$

其中 $\rho(G_J)$ 是雅可比迭代矩阵的谱半径。

**公式解析：最优 ω 从哪来。**

- **第一步，谱半径关系。** 对一致性有序的 SPD 矩阵，SOR 迭代矩阵谱半径满足 $\rho(G_\omega)=\dfrac{\omega-1}{\rho(G_J)^2}$ 等关系（当 $\omega\le\omega_{\mathrm{opt}}$），在 $\omega_{\mathrm{opt}}$ 处谱半径最小。
- **第二步，几何理解。** $\rho(G_J)$ 越接近 1（收敛慢），$\omega_{\mathrm{opt}}$ 越接近 2（超调越狠）——**雅可比越慢，就越该猛踩油门**。
- **第三步，收敛加速倍数。** 高斯-赛德尔（$\omega=1$）每步压缩 $\rho(G_J)^2$；最优 SOR 每步压缩 $\approx\dfrac{\omega_{\mathrm{opt}}-1}{\rho(G_J)}$。$\rho(G_J)=0.99$ 时，高斯-赛德尔要约 460 步到 $10^{-4}$，最优 SOR 只要约 47 步——**近 10 倍加速**。

**工程选择 $\omega$ 的务实做法**：不知道 $\rho(G_J)$ 时，先试 $\omega=1$（高斯-赛德尔），若收敛慢，逐步增大 $\omega$（1.1、1.2、…）观察；或直接扫描 $\omega\in(1,2)$ 找最快收敛点。<span class="marginnote">$\omega$ 选太小（<1）是亚松弛，用于本来就不稳的系统；$\omega>2$ 必发散（Kahan 定理）。<strong>工程经验：多数 PDE 离散问题 $\omega\in[1.2,1.8]$ 表现良好</strong>，精确最优值靠扫描或解析公式。</span>

## 4 数值实验：松弛因子的威力

解一维泊松方程离散出的三对角系统（$n=100$，SPD），比较三种迭代达到 $10^{-6}$ 误差所需步数：

| 方法 | 谱半径 $\rho$ | 迭代步数（到 $10^{-6}$） |
| --- | --- | --- |
| 雅可比 | $0.9990$ | 约 14000 |
| 高斯-赛德尔 | $0.9980$ | 约 6900 |
| **SOR（$\omega_{\mathrm{opt}}\approx1.94$）** | $0.937$ | **约 250** |

**SOR 比高斯-赛德尔快近 30 倍**——这就是松弛因子的价值。对 $n=10^6$ 的 PDE，这个差距意味着「几秒 vs 几个小时」。<span class="marginnote">谱半径从 0.998 降到 0.937，看起来只降了 6%，但迭代步数从 6900 降到 250——<strong>接近 1 的谱半径是「收敛慢」的真正敌人，把它拉离 1 一点点，就是数量级的加速</strong>。这也是预条件方法（现代共轭梯度）的全部动机。</span>

```python
def sor(A, b, omega, x0=None, tol=1e-10, max_iter=10000):
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
            x[i] = (1 - omega) * x_old[i] + omega * s
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k + 1
    return x, max_iter
```

## 5 SOR 与其他迭代法的定位

| 判据 | 雅可比 | 高斯-赛德尔 | SOR |
| --- | --- | --- | --- |
| 分裂 $M$ | $D$ | $D-L$ | $\tfrac1\omega D-L$ |
| 参数 | 无 | 无 | $\omega$ |
| 收敛（SPD） | 是 | 是 | $0<\omega<2$ |
| 典型谱半径 | $\rho_J$ | $\rho_J^2$ | $\omega_{\mathrm{opt}}$ 时约 $\dfrac{\omega-1}{\rho_J}$ |
| 加速 | — | 快 1 倍 | 快 10~30 倍 |

**辨析｜易错点：** SOR 的 $\omega$ 不是「越大越好」——超过 $\omega_{\mathrm{opt}}$ 后谱半径反而上升，$\omega\ge2$ 直接发散。**调参不是单调的，最优 $\omega$ 在中间某处**。<span class="marginnote">现代视角：SOR 的「松弛 + 超调」思想是预条件与 Krylov 子空间方法的先驱。如今大规模问题更多用<strong>共轭梯度（CG）或 GMRES + 预条件</strong>，但 SOR 仍是非对称/小规模系统的经典工具，也是理解「松弛为什么有用」的最佳入门。</span>

## 6 小结

- **SOR 迭代**：高斯-赛德尔 + 松弛因子 $\omega$，更新公式 $x_i^{(k+1)}=(1-\omega)x_i^{(k)}+\omega\tilde{x}_i^{(k+1)}$。
- 矩阵分裂 $M_\omega=\tfrac1\omega D-L$，$\omega=1$ 退化回高斯-赛德尔。
- **收敛定理**：SPD 且 $0<\omega<2$ ⇒ 收敛（Kahan 必要性 + Ostrowski-Reich 充分性）。
- **最优因子** $\omega_{\mathrm{opt}}=\dfrac{2}{1+\sqrt{1-\rho(G_J)^2}}$（一致性有序 SPD 矩阵），把谱半径压到最小。
- $\omega$ 不是越大越好：超 $\omega_{\mathrm{opt}}$ 谱半径回升，$\omega\ge2$ 必发散；工程上先扫 $\omega$ 再定。

在下一节，我们把「为什么收敛」钉到理论上限：**迭代法收敛的基本定理：谱半径条件**——$\rho(G)<1$ 的充要性、充分条件的谱系，以及谱半径的估计。

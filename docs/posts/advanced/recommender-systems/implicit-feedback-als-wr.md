---
title: 隐式反馈的矩阵分解：加权交替最小二乘（ALS-WR）
date: 2026-08-07
---

# 隐式反馈的矩阵分解：加权交替最小二乘（ALS-WR）

<div class="epigraph">
<p>行动胜于言辞。</p>
<footer>—— 英语谚语（Actions speak louder than words）</footer>
</div>

<div class="article-byline">
<p>第四级 · 推荐系统 ｜ 项亮《推荐系统实践》第2章 §2.5 隐语义模型（延伸） ｜ 2026-08-07</p>
</div>

## 为什么从隐式反馈开始

BiasSVD 把显式评分玩到了精致，可真实世界的数据大多**没有评分**：用户不会给每部片打分，但系统里记录着「看过没看过、看了几次、停留多久」。这类数据叫**隐式反馈（implicit feedback）**——它不直接说「喜欢」，只说「做了」。网购、视频、音乐、新闻的场景几乎全是隐式数据。

隐式反馈有三个特性，让矩阵分解必须改头换面：**没有负例**（用户没看某片 ≠ 不喜欢，可能只是没遇见）；**有置信度**（看了 20 次比看了 1 次更能说明问题）；**规模巨大**（行为日志比评分多几个数量级）。2008 年 Hu、Koren 与 Volinsky 提出的 **ALS-WR（Alternating Least Squares with Weighted Regularization，加权正则交替最小二乘）** 同时回应了这三点，并因适合分布式并行，成为 Spark MLlib 的默认推荐算法。<span class="marginnote">「没看 = 不喜欢」是最危险的假设。我们在《协同过滤的优缺点》里强调过缺失里有信息；ALS-WR 的聪明之处是：<strong>把「未交互」当成置信度最低的负例，而不是当成正例或干脆丢弃</strong>。</span>

## 1 隐式反馈：把行为翻译成两个量

对每个用户 $u$ 与物品 $i$，ALS-WR 定义两个量，把原始行为计数 $r_{ui}$（看了几次、点了几次）翻译成可学习的信号：

- **偏好（preference）** $p_{ui}$：$r_{ui} > 0$ 时取 1，否则取 0。它是「想不想看」的代理——发生过交互，说明用户至少对这个物品有真实兴趣。
- **置信度（confidence）** $c_{ui} = 1 + \alpha r_{ui}$：行为越频繁，置信度越高。$\alpha$ 是放大系数，控制「多一次观看」对置信度的加成。看 20 次的物品其 $c_{ui}$ 远大于只看 1 次的。

**辨析｜易错点：** $p_{ui}$ 不是评分。它是二值的「偏好指示」，而 $c_{ui}$ 才是携带强度的量。若把「观看次数」直接当评分喂进 Funk-SVD，会被少数狂热行为主导；ALS-WR 用「偏好 + 置信度」把「是否有兴趣」和「兴趣有多强」分开建模，模型更稳。

## 2 置信度加权的损失函数

模型仍是用户向量 $u_u$ 与物品向量 $v_i$ 的内积，预测 $\hat x_{ui} = u_u^{\mathsf{T}} v_i$，但损失函数改写成**加权平方误差**：

$$
\min_{U, V} \; \sum_{u, i} c_{ui} \left( p_{ui} - u_u^{\mathsf{T}} v_i \right)^2
+ \lambda \left( \sum_u \|u_u\|^2 + \sum_i \|v_i\|^2 \right)
$$

求和跑遍**所有用户—物品对**，包括没交互的那些：

- 对**有交互**的 $(u,i)$：$p_{ui} = 1$，$c_{ui} = 1 + \alpha r_{ui}$ 较大，模型必须认真把预测推到接近 1。
- 对**没交互**的 $(u,i)$：$p_{ui} = 0$，$c_{ui} = 1$，模型把预测往 0 拉，但拉的力量最弱。

**重点：** 未交互的格子**没有被丢弃**，而是以「置信度 1 的负例」身份参与训练。这让模型学到「这个用户大概率不想看这个物品」，而不是对没见过的物品毫无意见。<span class="marginnote">置信度 1 对应「证据不足」而非「确定不喜欢」。你可以把 $c=1$ 想成「这条样本的权重」，权重小的样本对参数影响也小——这正是后面第五篇《双塔召回中的负采样》里难负样本思想的雏形。</span>

## 3 交替最小二乘：把非凸问题切成无数个凸问题

直接对 $U$、$V$ 一起求导是非凸的，梯度下降容易陷入次优。ALS 的杀手锏是**坐标下降的升级版**：

1. **固定 $V$**：此时损失对每个 $u_u$ 是独立的**二次函数**，可对每个用户单独求闭式解（加权岭回归）。
2. **固定 $U$**：同理，对每个 $v_i$ 单独求闭式解。
3. 交替重复直到收敛。

**辨析｜易错点：** 千万不要以为 ALS 是在整体上对 $U$ 求导一次。真正可行的是「固定一边、对另一边求最小二乘」——因为固定 $V$ 后，损失函数对 $U$ 是凸的，有解析解。两边交替，就是沿坐标轴轮流精确最小化。

## 4 公式解析：用户的加权岭回归闭式解

固定 $V$ 后，看单个用户 $u$ 的子问题。定义 $C_u$ 为对角矩阵，对角线是 $c_{ui}$（$i = 1,\dots,m$）；$p(u) \in \{0,1\}^m$ 是用户 $u$ 对所有物品的偏好向量。对 $u_u$ 求导并令梯度为零：

$$
\left( V^{\mathsf{T}} C_u V + \lambda I \right) u_u = V^{\mathsf{T}} C_u\, p(u)
$$

于是得到闭式解：

$$
u_u = \left( V^{\mathsf{T}} C_u V + \lambda I \right)^{-1} V^{\mathsf{T}} C_u\, p(u)
$$

四步拆解：

- **第一步，看 $V^{\mathsf{T}} C_u V$**：这是**加权的 $V$ 的自协方差矩阵**，$k \times k$ 阶。权重 $c_{ui}$ 表示「物品 $i$ 对确定 $u_u$ 有多大发言权」。
- **第二步，看 $+\lambda I$**：正则项让矩阵**可逆**，同时把 $u_u$ 的模长压住。没有它，$V^{\mathsf{T}} C_u V$ 可能奇异，解不稳定——这正是「正则 = 可逆 + 收缩」的第二次现身（第一次在 BiasSVD 的偏置）。
- **第三步，看右侧 $V^{\mathsf{T}} C_u\, p(u)$**：把用户 $u$ 的偏好（含权重）投影到物品空间，再回转用户空间。它回答「这个用户到底偏好了哪些方向的物品」。
- **第四步，解线性方程组**：$k$ 一般只有 20~200，求逆或 Cholesky 分解都极快——**每个用户是独立的 $k \times k$ 问题**。

对物品侧完全对称：$v_i = \left( U^{\mathsf{T}} C_i U + \lambda I \right)^{-1} U^{\mathsf{T}} C_i\, p(i)$。

## 5 稀疏加速与并行：为什么它是「工程之选」

直接算 $V^{\mathsf{T}} C_u V$ 要对所有 $m$ 个物品求和，$m$ 可达千万级。关键观察是：$C_u$ 与单位阵只在对角线上不同，且只有用户交互过的物品那一格 $c_{ui} \neq 1$。于是：

$$
V^{\mathsf{T}} C_u V = V^{\mathsf{T}} V + V^{\mathsf{T}} (C_u - I) V
$$

$V^{\mathsf{T}} V$ 是全局共享的，可离线算一次；$(C_u - I)$ 只在用户交互过的 $N_u$ 个物品上非零，所以第二项只需 $O(k^2 N_u)$。整体每个用户的代价降到 $O(k^2 N_u + k^3)$。

**重点：** 这个分解让 ALS 的**每个用户、每个物品互相独立**，天然适合 MapReduce / Spark 的并行：先并行算所有 $u_u$，再并行算所有 $v_i$，一轮两遍。相比 SGD 逐样本串行推进，ALS 在隐式数据上收敛更稳，还**不需要调学习率**。<span class="marginnote">这也是 Spark MLlib 与许多离线批处理框架默认推荐 ALS 而非 SGD 的原因：<strong>无学习率、可并行、可复现</strong>。而在线增量场景 SGD 仍占优，我们在第十二篇《实时推荐》会再回到这条 trade-off。</span>

把一步交替写成代码：

```python
import numpy as np
from scipy.sparse import csr_matrix

def als_step(Pref, C, U, V, lam, alpha=40.0):
    """一轮：固定 V 更新 U，再固定 U 更新 V。Pref 是 (n_users, n_items) 稀疏矩阵。"""
    n_u, n_i = Pref.shape
    k = V.shape[1]
    VtV = V.T @ V                    # 全局共享项，离线算一次
    U_new = np.zeros_like(U)
    for u in range(n_u):
        rows = Pref[u].indices            # 用户 u 交互过的物品下标
        if len(rows) == 0:
            U_new[u] = 0.0
            continue
        Cu_minus_I = (alpha * Pref[u].data + 1.0) - 1.0   # c-1 非零项
        Cu = alpha * Pref[u].data + 1.0                    # 置信度
        # 稀疏技巧：VtV + sum_j Cu_j * outer(v_j, v_j)
        M = VtV.copy()
        for idx, j in enumerate(rows):
            M += Cu_minus_I[idx] * np.outer(V[j], V[j])
        rhs = (Cu * Pref[u].data) @ V[rows]                # V^T C_u p(u)
        U_new[u] = np.linalg.solve(M + lam * np.eye(k), rhs)
    return U_new
```

循环体是一个 $k \times k$ 的线性方程组求解，配合上述稀疏技巧，即可处理千万级物品。

## 6 小结

- **隐式反馈**没有评分，只有行为计数；ALS-WR 把它翻译成**偏好 $p_{ui}$（二值）与置信度 $c_{ui} = 1 + \alpha r_{ui}$（强度）**。
- **加权损失** $\sum_{u,i} c_{ui}(p_{ui} - u_u^{\mathsf{T}} v_i)^2 + \lambda(\cdot)$：未交互的格子以「置信度 1 的负例」参与，不被丢弃。
- **ALS** 固定一边、对另一边求最小二乘，把非凸问题切成无数个凸子问题；每个用户/物品是独立的 $k \times k$ 岭回归。
- **闭式解** $u_u = (V^{\mathsf{T}} C_u V + \lambda I)^{-1} V^{\mathsf{T}} C_u p(u)$，依赖正则保证可逆与收缩。
- **稀疏技巧** $V^{\mathsf{T}} C_u V = V^{\mathsf{T}} V + V^{\mathsf{T}}(C_u - I)V$ 把单用户代价压到 $O(k^2 N_u + k^3)$，且天然并行——这是它在 Spark 中胜出的工程原因。

在下一节，我们把显式与隐式**两套信号同时装进一个模型**：SVD++ 用「用户评过哪些物品」这个免费信号给每个用户叠加一套隐式因子，成为 Netflix 夺冠方案的基石。

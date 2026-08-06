---
title: 李群 SO(3)/SE(3)、李代数与指数映射：旋量（Twist）与螺旋运动
date: 2026-08-07
---

# 李群 SO(3)/SE(3)、李代数与指数映射：旋量（Twist）与螺旋运动

<div class="epigraph">
<p>对称性，是人类世世代代借以理解、创造秩序、美感与完美的一个观念。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl），《对称》（Symmetry, 1952）</footer>
</div>

<div class="article-byline">
<p>第四级 · 具身智能 ｜ Craig《机器人学导论》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从李群李代数开始

前两节我们学会了**静态**地描述姿态：旋转矩阵、欧拉角、四元数。但机器人从不停下来——机械臂的每个关节都在转，无人机的姿态每毫秒都在变。于是真正要紧的问题是**动态**的：刚体正在**以多快的角速度转**？把时间带进姿态描述，就引出了李群与李代数。

这个视角的威力超乎想象。**旋转矩阵对时间的导数，永远是反对称矩阵**——这个「反对称」的对象就是李代数 $\mathfrak{so}(3)$；反过来，一个**常数角速度作用一段时间**，得到的旋转矩阵恰好是「矩阵指数」——这就是指数映射。更妙的是，把旋转和平移的速度拼成一个 6 维对象，就是**运动旋量（twist）**，它的指数映射直接给出机器人学里的**螺旋运动**：一边绕轴转、一边沿轴平移。

这不仅是理论装饰。现代机器人学里正运动学的**指数积公式（POE）**、雅可比矩阵、奇异性分析、全身控制的微分运动学，全都建立在李群的语言上；SLAM 后端的状态估计、惯性导航的姿态传播，也在用「小速度」累积成「大姿态」。上一节结尾那句话——四元数生活在球面 $S^3$ 上——正是 $SO(3)$ 作为光滑流形的一个投影。

## 1 从导数到李代数：旋转的「瞬时速度」

设 $R(t)$ 是一族随时间变化的旋转矩阵，$R(0) = I$。旋转矩阵恒满足 $R R^T = I$，两边对 $t$ 求导：

$$
\dot R R^T + R \dot R^T = 0
$$

移项得 $\dot R R^T = -(\dot R R^T)^T$——**$\dot R R^T$ 是反对称矩阵**。反对称矩阵一定可以写成某个向量的叉乘矩阵：

$$
\dot R R^T = [\omega]_\times =
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
$$

**$\omega = (\omega_x, \omega_y, \omega_z)$ 就是刚体的瞬时角速度**（body 系表达时用 $R^T \dot R$）。于是旋转的「速度」天然是一个 3 维反对称矩阵。

**核心概念：李代数（Lie algebra）**：全体 3×3 反对称矩阵构成的集合，记作 $\mathfrak{so}(3)$：

$$
\mathfrak{so}(3) = \left\{ [\omega]_\times \in \mathbb{R}^{3\times3} \;\middle|\; [\omega]_\times^T = -[\omega]_\times \right\}
$$

它是 $SO(3)$ 在单位元处的「切空间」：$R(t)$ 从单位元出发，它的速度方向就躺在 $\mathfrak{so}(3)$ 里。<span class="marginnote">把 $SO(3)$ 想象成一张嵌在 9 维空间里的光滑「曲面」，$I$ 处的切平面正是 $\mathfrak{so}(3)$；而沿切平面里一个固定方向「匀速直行」一小段时间，再投影回曲面上，就是指数映射。李群与李代数，本质上是把「一个姿态」与「这个姿态的变化率」打包成了两个互补的对象。</span>

**重点：切空间里是速度（线性、可加），流形上是姿态（乘法、不可交换）。** 这一句是全部李群直觉的浓缩。

## 2 公式解析：指数映射 exp([ω]θ) 与罗德里格斯公式

给定一个**常数角速度** $[\omega]_\times$ 作用时间 $\theta$（把角速度大小吸收进 $\theta$，设 $\|\omega\| = 1$，$\theta$ 即转角），微分方程 $\dot R = [\omega]_\times R$、$R(0) = I$ 的解是矩阵指数：

$$
R(\theta) = e^{[\omega]_\times \theta} = \sum_{k=0}^{\infty} \frac{\theta^k}{k!} [\omega]_\times^k
$$

这个无穷级数能不能算出闭式？分三步。

**第一步，算反对称矩阵的幂。** 对单位向量 $\omega$，关键恒等式是 $[\omega]_\times^2 = \omega\omega^T - I$ 与 $[\omega]_\times^3 = -[\omega]_\times$。验证后者：$[\omega]_\times^3 = [\omega]_\times([\omega]_\times^2) = [\omega]_\times(\omega\omega^T - I) = (\omega \times \omega)\omega^T - [\omega]_\times = -[\omega]_\times$（因为 $\omega \times \omega = 0$）。于是所有幂都能归约到 $I$、$[\omega]_\times$、$[\omega]_\times^2$ 三种。

**第二步，把级数按奇偶项拆开。**

$$
e^{[\omega]_\times \theta}
= I + \theta[\omega]_\times + \frac{\theta^2}{2!}[\omega]_\times^2 + \frac{\theta^3}{3!}[\omega]_\times^3 + \frac{\theta^4}{4!}[\omega]_\times^4 + \cdots
$$

用 $[\omega]^3 = -[\omega]$ 与 $[\omega]^4 = -[\omega]^2$ 代入，奇数项都含 $[\omega]$，偶数项都含 $[\omega]^2$：

$$
e^{[\omega]_\times \theta}
= I + \left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)[\omega]_\times
+ \left(\frac{\theta^2}{2!} - \frac{\theta^4}{4!} + \frac{\theta^6}{6!} - \cdots\right)[\omega]_\times^2
$$

**第三步，认出三角级数。** 括号里分别是 $\sin\theta$ 与 $1 - \cos\theta$：

$$
e^{[\omega]_\times \theta} = I + \sin\theta\, [\omega]_\times + (1 - \cos\theta)\, [\omega]_\times^2
$$

**这就是罗德里格斯公式——上一节我们用几何拆出来的旋转矩阵，这一节从矩阵指数里重新长了出来。** 两个方向殊途同归，说明「绕单位轴转 $\theta$」与「以单位角速度转 $\theta$ 秒」是同一件事。<span class="marginnote">当 $\omega$ 不是单位向量时，把 $\theta$ 吸收进 $\omega$：$e^{[\omega]_\times} = I + \frac{\sin\|\omega\|}{\|\omega\|}[\omega]_\times + \frac{1-\cos\|\omega\|}{\|\omega\|^2}[\omega]_\times^2$，其中 $\|\omega\|$ 就是转角。这套「先取对数反解轴角、再指数映射回矩阵」的操作，是 SLAM 后端、李群优化的日常。</span>

## 3 SE(3) 与运动旋量：旋转与平移的统一速度

把姿态升级到位姿。齐次变换矩阵构成**特殊欧氏群**：

$$
SE(3) = \left\{ T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix} \;\middle|\; R \in SO(3),\ p \in \mathbb{R}^3 \right\}
$$

对 $T(t)$ 求导，$T^{-1} \dot T$（body 系）或 $\dot T T^{-1}$（空间系）落在 $\mathfrak{se}(3)$ 上，形如：

$$
[\mathcal{V}] =
\begin{bmatrix}
[\omega]_\times & v \\
0 & 0
\end{bmatrix}
\in \mathbb{R}^{4\times4}
$$

**核心概念：运动旋量（twist）**：把角速度 $\omega$ 与线速度 $v$ 拼成一个 6 维矢量，是 $\mathfrak{se}(3)$ 的坐标表示：

$$
\mathcal{V} = (v_x, v_y, v_z, \omega_x, \omega_y, \omega_z) \in \mathbb{R}^6
$$

$v$ 的几何含义需要斟酌：$v$ 是**刚体上恰好与坐标系原点重合的那个点的速度**（空间 twist）——一个「想象的点」的速度，不是某个真实质点的速度。<span class="marginnote">旋量（screw）的概念可追溯到 19 世纪：罗伯特 · 鲍尔（Robert Ball）1876 年《螺旋理论》（The Theory of Screws）把刚体运动的瞬时状态描述为「绕一根轴转动 + 沿同轴平移」的组合。李群语言把它压缩成一个 6 维向量与一个指数映射——数学史与机器人学在这里相遇。</span>

任何运动旋量都可以几何化：若 $\omega \neq 0$，存在一根**螺旋轴（screw axis）**，刚体一边绕它转、一边沿它平移。轴的方向是 $\hat{\omega}$，轴上一点（旋量矩）满足

$$
\omega \times q = v \quad\Longleftrightarrow\quad q = \frac{\omega \times v}{\|\omega\|^2}
$$

**螺距（pitch）**定义为

$$
h = \frac{\omega \cdot v}{\|\omega\|^2}
$$

它度量「每转一弧度，沿轴前进多少」——$h = 0$ 是纯旋转，$\omega = 0$ 是纯平移。

![exp(ξ̂θ)：绕螺旋轴转动 θ 同时沿轴平移 hθ 的螺旋运动](/images/embodied-ai/lie-groups-twists-1.svg)

## 4 公式解析：旋量的指数映射与螺旋运动

现在回答这一节的核心问题：**常速度作用 $\theta$ 秒，位姿变成什么？** 即 $T(\theta) = e^{[\mathcal{V}]\theta}$。

设螺旋轴取单位角速度 $\|\omega\| = 1$，记 $[\mathcal{V}] = \begin{bmatrix} [\omega]_\times & v \\ 0 & 0 \end{bmatrix}$。展开矩阵指数需要先算幂。**第一步，算 $[\mathcal{V}]$ 的幂。**

$$
[\mathcal{V}]^2 =
\begin{bmatrix} [\omega]_\times^2 & [\omega]_\times v \\ 0 & 0 \end{bmatrix},
\qquad
[\mathcal{V}]^3 =
\begin{bmatrix} [\omega]_\times^3 & [\omega]_\times^2 v \\ 0 & 0 \end{bmatrix}
$$

规律：左上角块是 $[\omega]$ 的幂（上一节已算过），右上角块依次是 $[\omega]^{k-1} v$，左下角恒为 0。

**第二步，求和得到左上角（旋转部分）。** 与上一节完全相同，指数级数给出：

$$
e^{[\mathcal{V}]\theta} \text{ 的左上角 } = e^{[\omega]_\times \theta}
= I + \sin\theta\,[\omega]_\times + (1 - \cos\theta)\,[\omega]_\times^2
$$

**第三步，求和得到右上角（平移部分）。** 右上角块是

$$
\sum_{k=1}^{\infty} \frac{\theta^k}{k!} [\omega]_\times^{k-1} v
= \Big(I\theta + \frac{\theta^2}{2!}[\omega]_\times + \frac{\theta^3}{3!}[\omega]_\times^2 + \cdots\Big) v
$$

利用 $[\omega]^3 = -[\omega]$ 把高阶项归约，括号里的矩阵级数收敛为

$$
G(\theta) = I\theta + (1 - \cos\theta)\,[\omega]_\times + (\theta - \sin\theta)\,[\omega]_\times^2
$$

于是**运动旋量的指数映射**为：

$$
e^{[\mathcal{V}]\theta} =
\begin{bmatrix}
e^{[\omega]_\times\theta} & G(\theta)\, v \\
0 & 1
\end{bmatrix},
\qquad
\|\omega\| = 1
$$

当 $\omega = 0$（纯平移）时退化为 $e^{[\mathcal{V}]\theta} = \begin{bmatrix} I & v\theta \\ 0 & 1 \end{bmatrix}$。

**重点：$G(\theta)v$ 的每一项都有几何含义。** 把它与螺旋轴结合：$v = \omega \times q + h\omega$（轴矩分解），代入后可以验证 $G(\theta)v$ 恰好等于「绕螺旋轴转动 $\theta$ + 沿轴平移 $h\theta$」。**指数映射把常数旋量变成了真正的螺旋运动**——这就是图里那条螺旋线的来历。<span class="marginnote">验证一个特例：纯旋转（$h=0$）时螺旋轴过原点，$v = \omega \times 0 = 0$，$G(\theta)v = 0$，于是 $e^{[\mathcal{V}]\theta} = \begin{bmatrix} R & 0 \\ 0 & 1 \end{bmatrix}$，退化回 $SO(3)$ 的指数映射。SE(3) 的指数映射是 SO(3) 的推广——这正是「统一」二字的数学含义。</span>

**辨析｜易错点：** 矩阵指数不可「逐元素取指数」。$e^{[V]\theta}$ 绝不等价于把 6 个分量分别取指数再拼回去——它必须作为**矩阵级数**整体求和。数值实现直接用 `scipy.linalg.expm` 或上述闭式，而不是对每个元素用 `np.exp`。

## 5 从旋量到正运动学：指数积（POE）的预告

旋量带来一套全新的正运动学写法。机械臂的每个关节 $i$ 对应一个运动旋量（关节轴的单位旋量）$\mathcal{S}_i$；当关节转角为 $\theta_i$ 时，相对位姿是 $e^{[\mathcal{S}_i]\theta_i}$。**整条机械臂的正运动学是**

$$
T(\theta) = e^{[\mathcal{S}_1]\theta_1}\, e^{[\mathcal{S}_2]\theta_2} \cdots e^{[\mathcal{S}_n]\theta_n}\, T_{M}
$$

其中 $T_M$ 是机械臂在零位时末端相对基座的位姿。这称为**指数积公式（product of exponentials, POE）**，是 Lynch & Park《Modern Robotics》的正运动学主线。

它比 D-H 连乘更现代的地方在于：每个关节旋量直接对应**世界坐标系里的一条螺旋轴**，不需要在每根杆上费心挑选坐标系，写代码、做优化都更顺。我们会在下一篇 D-H 约定、以及正运动学一节里把两种写法对照起来。<span class="marginnote">旋量还统治着微分运动学：雅可比矩阵的每一列恰是关节轴对应的运动旋量。求速度不用再做一次正运动学，直接查表取旋量即可——这就是李群视角在控制里的实用性。</span>

用代码验证旋量指数映射的闭式与数值结果一致：

```python
import numpy as np
from scipy.linalg import expm

def wedge(v, w):
    """se(3) 矩阵：v 为线速度、w 为角速度（单位向量）"""
    V = np.zeros((4, 4))
    V[:3, :3] = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
    V[:3, 3] = v
    return V

def exp_twist_closed(v, w, theta):
    """单位角速度 w 的闭式指数映射"""
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, G @ v
    return T

w = np.array([0.0, 0.0, 1.0])          # 绕 Z 轴的螺旋
v = np.array([0.1, 0.0, 0.5])          # 线速度：含沿轴的 0.5 → 螺距 0.5
theta = np.pi / 2

T1 = expm(wedge(v, w) * theta)          # 数值矩阵指数
T2 = exp_twist_closed(v, w, theta)      # 闭式公式
print("闭式 vs 数值一致:", np.allclose(T1, T2, atol=1e-12))   # True
print("旋转部分(前两列):\n", np.round(T2[:3, :2], 3))
print("平移部分:", np.round(T2[:3, 3], 3))
```

## 6 小结

- **李代数 $\mathfrak{so}(3)$**：全体反对称矩阵，是 $SO(3)$ 在单位元处的切空间；$\dot R R^T = [\omega]_\times$ 给出瞬时角速度。
- **指数映射**：常数角速度作用 $\theta$ 后，$e^{[\omega]_\times\theta} = I + \sin\theta\,[\omega]_\times + (1-\cos\theta)[\omega]_\times^2$——从李代数回到李群的「积分」。
- **运动旋量（twist）**：$\mathcal{V} = (v, \omega) \in \mathbb{R}^6$，是 $\mathfrak{se}(3)$ 的坐标表示，统一描述旋转与平移速度。
- **螺旋运动**：$e^{[\mathcal{V}]\theta} = \begin{bmatrix} e^{[\omega]\theta} & G(\theta)v \\ 0 & 1 \end{bmatrix}$，$G(\theta) = I\theta + (1-\cos\theta)[\omega]_\times + (\theta-\sin\theta)[\omega]_\times^2$；螺距 $h = \omega\cdot v / \|\omega\|^2$。
- **指数积公式（POE）**：正运动学 $T(\theta) = \prod_i e^{[\mathcal{S}_i]\theta_i}\, T_M$，是现代机器人学的正运动学主线。

在下一节，我们回到 Craig 的经典路线：为机械臂的每一根杆件分配坐标系，用四个参数（$a, \alpha, d, \theta$）把「相邻两杆的相对位姿」标准化成一张表——**D-H 约定**。那是工业机器人运动学的第一块敲门砖。

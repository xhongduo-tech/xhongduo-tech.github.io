---
title: 曲面的结构方程：Gauss 公式与 Weingarten 公式
date: 2026-08-07
---

# 曲面的结构方程：Gauss 公式与 Weingarten 公式

<div class="epigraph">
<p>曲面局部几何的全部信息，浓缩在两个公式里：一个描述基向量如何随位置变，一个描述法向量如何随位置变。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§4.6 ｜ 2026-08-07</p>
</div>

## 为什么从结构方程开始

曲面论的「基本对象」是坐标基向量 $\mathbf{x}_u, \mathbf{x}_v$ 与法向 $\mathbf{N}$。要研究曲面的局部几何，就要知道这些向量**如何随位置变化**——它们的导数。这些导数被两大公式统一起来，称为**结构方程（structure equations）**：

- **Gauss 公式**：描述 $\mathbf{x}_u, \mathbf{x}_v$（切基）的导数如何分解。
- **Weingarten 公式**：描述法向 $\mathbf{N}$ 的导数如何分解。

**这两个公式是曲面论的「运动方程」**：它们告诉我们「标架 $\{\mathbf{x}_u,\mathbf{x}_v,\mathbf{N}\}$ 走到哪、怎么转」。而把「混合偏导可交换」这套相容性条件用上去，就能推出 **Gauss-Codazzi 方程**——那是下一节、也是整个曲面论基本定理的核心。<span class="marginnote">结构方程的现代形式由 Cartan（嘉当）用「活动标架法」重新表述：把标架的无穷小变化写成「结构方程」。这里我们走经典路线（Gauss 公式 + Weingarten 公式），它们与 Cartan 版本完全等价——只是语言不同。两种语言都值得认得。</span>

## 1 Gauss 公式

**Gauss 公式（Gauss formula）**：坐标基向量 $\mathbf{x}_u,\mathbf{x}_v$ 的二阶偏导分解为切向 + 法向：

$$
\begin{aligned}
\mathbf{x}_{uu} &= \Gamma^1_{11}\,\mathbf{x}_u + \Gamma^2_{11}\,\mathbf{x}_v + L\,\mathbf{N},\\
\mathbf{x}_{uv} &= \Gamma^1_{12}\,\mathbf{x}_u + \Gamma^2_{12}\,\mathbf{x}_v + M\,\mathbf{N},\\
\mathbf{x}_{vv} &= \Gamma^1_{22}\,\mathbf{x}_u + \Gamma^2_{22}\,\mathbf{x}_v + N\,\mathbf{N}
\end{aligned}
$$

（用对称性 $\mathbf{x}_{uv} = \mathbf{x}_{vu}$。）

**重点：Gauss 公式把二阶导拆成「切向部分（$\Gamma$）+ 法向部分（$L,M,N$）」。** 切向部分由 Christoffel 记号给出（内蕴），法向部分正是第二基本形式的系数（外蕴）。**一个公式同时装进了内蕴与外蕴——曲面局部几何的完整结构。**<span class="marginnote">对比：上一节的 Christoffel 记号是「二阶导的切向分量」，第二基本形式是「二阶导的法向分量」——Gauss 公式把它们合写成一个完整的分解。$\Gamma$ 管「坐标系弯折」，$L,M,N$ 管「曲面弯曲」，二者在 Gauss 公式里并肩而立。</span>

## 2 Weingarten 公式

**Weingarten 公式（Weingarten formula）**：法向 $\mathbf{N}$ 的偏导分解为切向（没有法向分量，因为 $\mathbf{N}\cdot\mathbf{N}=1$ 保证 $\mathbf{N}_u, \mathbf{N}_v \perp \mathbf{N}$）：

$$
\begin{aligned}
\mathbf{N}_u &= -\frac{GL - FM}{EG - F^2}\,\mathbf{x}_u + \frac{EM - FL}{EG - F^2}\,\mathbf{x}_v,\\[4pt]
\mathbf{N}_v &= -\frac{GM - FN}{EG - F^2}\,\mathbf{x}_u + \frac{EN - FM}{EG - F^2}\,\mathbf{x}_v
\end{aligned}
$$

（或写成紧凑形式 $\mathbf{N}_i = -\sum_j S_i^{\,j}\,\mathbf{x}_j$，其中 $S$ 是形状算子。）

**重点：Weingarten 公式说「法向的变化完全由切向表达」——法向变化的系数矩阵正是形状算子。** 这印证了第三篇的结论：$dN$（法向变化）与 $II$（弯曲）是同一件事的两个视角。<span class="marginnote">记忆：Weingarten 公式里的系数正是形状算子矩阵 $[S] = \mathcal{I}^{-1}\mathcal{II}$ 的负号。所以 Weingarten 公式本质上就是「形状算子的坐标实现」：$\mathbf{N}_i = -S(\mathbf{x}_i)$。这条公式也是「法向如何被切向场决定」的机制。</span>

## 3 公式解析：结构方程的统一视角

把 Gauss 公式与 Weingarten 公式放到一起，可以统一成一个「标架运动方程」。令 $3\times3$ 矩阵

$$
F = \big(\mathbf{x}_u\ \ \mathbf{x}_v\ \ \mathbf{N}\big)
$$

（三列是标架向量），则所有一阶、二阶导数都被一个「结构矩阵」编码：

- **第一阶**：$F_u, F_v$ 关于 $F$ 的展开系数，含 $\Gamma$（切向）与 $L,M,N$（法向）——正是 Gauss 公式与 Weingarten 公式。
- **第二阶（混合偏导可交换）**：$(\mathbf{x}_{uu})_v = (\mathbf{x}_{uv})_u$ 等——推出相容性条件。

**重点：结构方程是「标架如何运动」的完整描述。** 它扮演的角色，正如曲线论里 Frenet 公式之于 Frenet 标架：**给定标架的「转速」，标架的整个演化由 ODE 决定。**<span class="marginnote">对比曲线论：Frenet 公式 $\mathbf{T}' = \kappa\mathbf{N}$ 等描述 Frenet 标架怎么转（用曲率挠率）；曲面的结构方程描述 $\{\mathbf{x}_u,\mathbf{x}_v,\mathbf{N}\}$ 怎么动（用 $\Gamma$ 和 $L,M,N$）。「用局部量编码标架运动」是同一套方法论的两次使用——活动标架法的精髓。</span>

## 4 从结构方程到相容性：混合偏导可交换

结构方程本身只是「分解」，要让它们**一致**，必须满足「混合偏导可交换」：$\mathbf{x}_{uuv} = \mathbf{x}_{uvu}$、$\mathbf{N}_{uv} = \mathbf{N}_{vu}$ 等。把 Gauss 公式代入这些恒等式，就得到**相容性条件（compatibility conditions）**：

**Gauss 方程（Gauss equation）**：由 $\mathbf{x}_{uuv} = \mathbf{x}_{uvu}$ 推出——它把 $K$ 与 $\Gamma$、$L,M,N$ 联系起来，**保证 $K$ 内蕴**。
**Codazzi-Mainardi 方程（Codazzi-Mainardi equations）**：由 $\mathbf{N}_{uv} = \mathbf{N}_{vu}$ 推出——它是关于 $L,M,N$ 的偏微分方程，刻画「$II$ 沿曲面的变化」必须与 $I$ 相容。

这两组方程合称 **Gauss-Codazzi 方程**——下一节的正式主题。它们是曲面论基本定理的「可积性条件」：**不是任意 $E,F,G,L,M,N$ 都能拼出一张曲面，必须满足 Gauss-Codazzi。**<span class="marginnote">直观：给定六个函数 $E,F,G,L,M,N$，它们要像「真曲面」的系数，必须满足 Gauss-Codazzi——否则「拼」出来的对象会自相矛盾（混合偏导对不上）。这正如给定函数 $f$ 要求 $f_{xy}=f_{yx}$：不是随便给都能做「势函数」。曲面论基本定理说：Gauss-Codazzi 就是充分必要条件。</span>

## 5 结构方程的三种角色

结构方程在整个曲面论里身兼数职：

**曲率的内蕴证明**：Gauss 方程把 $K$ 用 $E,F,G,\Gamma$ 表达——这是 Gauss 绝妙定理的严格证明路径。
**曲面论基本定理**：Gauss-Codazzi 是可积性条件，保证「给定 $I,II$ 存在曲面」。
**标架方法**：结构方程是 Cartan 活动标架法的原型，在现代微分几何里反复出现。

**重点：结构方程是曲面论的「宪法」**——它约束着六个系数 $E,F,G,L,M,N$ 必须如何协调，才能描述一张真实存在的曲面。<span class="marginnote">在黎曼几何（第八篇），Gauss 方程升级为「曲率张量的第一 Bianchi 恒等式」与「第二 Bianchi 恒等式」，Codazzi 方程成为曲率张量导数与联络的关系。结构方程的思想从二维曲面一路长成黎曼几何的骨架。</span>

### 例：圆柱面的结构方程

用圆柱面（$\mathbf{x}(u,v) = (\cos u, \sin u, v)$）具体写出结构方程，感受每个符号的落点。

- **基向量**：$\mathbf{x}_u = (-\sin u, \cos u, 0)$、$\mathbf{x}_v = (0,0,1)$、$\mathbf{N} = (\cos u, \sin u, 0)$。
- **Gauss 公式**：$\mathbf{x}_{uu} = (-\cos u, -\sin u, 0) = -1\cdot\mathbf{x}_u + 0\cdot\mathbf{x}_v + 1\cdot\mathbf{N}$（取外法向约定）——切向部分 $\Gamma^1_{11} = 0$（坐标基本身「直」），法向部分 $L = 1$（弯曲在法向）。
- **Weingarten 公式**：$\mathbf{N}_u = (-\sin u, \cos u, 0) = -1\cdot\mathbf{x}_u$——法向沿 $u$ 方向变化，系数正是形状算子。

**重点：圆柱面的结构方程极简——$\Gamma$ 全部为零（坐标基是直的）、只有 $L$ 非零（弯曲只在法向）。** 这对照「$\Gamma$ 管坐标系弯折、$L,M,N$ 管曲面弯曲」的分工：圆柱面的坐标是「直的」（$\Gamma=0$），弯曲全在法向（$L\neq0$）。结构方程把「坐标弯」与「曲面弯」分得清清楚楚。

### 结构方程与 Cartan 活动标架

结构方程的现代升级版是 Cartan 的**活动标架法**：不固定坐标系，让标架 $\{\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3\}$ 随点移动，用「结构方程」（标架的无穷小变化）研究几何。Gauss 公式与 Weingarten 公式正是「坐标标架的活动标架方程」。

**重点：结构方程是「活动标架法」的坐标版本——标架怎么动，全部几何就怎么长。** 从 Gauss-Weingarten（坐标标架）到 Cartan（任意标架），同一套「标架运动方程」思想贯穿。活动标架法在现代微分几何（纤维丛、规范场论）里仍是核心工具——结构方程是它的源头。

### 结构方程的信息量

Gauss 公式（3 条）+ Weingarten 公式（2 条）共 5 条方程，编码了「标架 $\{\mathbf{x}_u,\mathbf{x}_v,\mathbf{N}\}$ 的全部一阶运动信息」——$\Gamma$（切向）与 $L,M,N$（法向）。**五个自由度（$\Gamma$ 3 个 + $II$ 3 个，减对称性）恰好匹配「标架怎么动」。**

**结构方程是「曲面局部几何的完整运动方程」——给定它们，标架的演化被唯一确定。** 这与曲线论 Frenet 公式的地位完全相同：一条曲线被 $\kappa,\tau$ 驱动，一张曲面被 $\Gamma,L,M,N$ 驱动。「结构方程 = 曲面的 Frenet 公式」是记忆它的最佳类比。

## 6 小结

- **Gauss 公式**：$\mathbf{x}_{ij} = \Gamma^k_{ij}\mathbf{x}_k + (L,M,N)\,\mathbf{N}$——二阶导 = 切向（$\Gamma$）+ 法向（$II$）。
- **Weingarten 公式**：$\mathbf{N}_i = -S(\mathbf{x}_i)$——法向变化由形状算子给出，系数矩阵 $= -[S]$。
- 两公式统一为「标架运动方程」，对应曲线论的 Frenet 公式。
- **相容性**：混合偏导可交换 ⟹ Gauss 方程 + Codazzi-Mainardi 方程。
- 结构方程是曲率内蕴证明、曲面基本定理、活动标架法的共同基础。

在下一节，我们深入研究相容性条件本身：**Gauss-Codazzi 方程**——它们如何约束六个系数，以及如何导出高斯曲率的内蕴公式。

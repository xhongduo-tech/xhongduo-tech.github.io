---
title: 等参数单元与数值积分
date: 2026-08-07
---

# 等参数单元与数值积分

<div class="epigraph">
<p>给我一个支点，我可以撬动地球；给我一套形函数，我可以扭曲整个网格。</p>
<footer>—— 有限元工程师的行话（改编自阿基米德）</footer>
</div>

<div class="article-byline">
<p>第六级 · 计算力学与有限元方法 ｜ Zienkiewicz《The Finite Element Method》第8–9章 ｜ 2026-08-07</p>
</div>

## 为什么从等参数单元开始

上一节我们止步于一个悬而未决的问题：双线性四边形单元怎么处理不规则的任意四边形？直接给一个扭曲的四节点单元写插值公式，坐标混在一起，形函数根本没法分离。**等参数映射（isoparametric mapping）** 用一招化解了全部麻烦：让「插值位移的那套形函数」同时去插值坐标，把任意四边形「拉回」一个标准正方形里计算。形函数因此成为单元最核心的资产——形状和位移都由它说了算。代价是单元刚度积分变得复杂，不再能闭式求出，必须交给**数值积分**处理。这一节讲清楚这两件事，你就拿到了现代有限元软件的最后一个拼图。

## 1 等参数映射：一个形函数，两副面孔

以四节点四边形为例。标准单元定义在自然坐标 $(\xi, \eta) \in [-1,1]^2$ 上，形函数 $N_i = \frac{1}{4}(1+\xi\xi_i)(1+\eta\eta_i)$。

**等参数的思想**：用同一组形函数分别插值几何与位移——

$$
x = \sum_{i=1}^{4} N_i(\xi,\eta)\, x_i, \qquad y = \sum_{i=1}^{4} N_i(\xi,\eta)\, y_i
$$

$$
u = \sum_{i=1}^{4} N_i(\xi,\eta)\, u_i, \qquad v = \sum_{i=1}^{4} N_i(\xi,\eta)\, v_i
$$

「等参数」的「等」字，指的就是几何插值函数与位移插值函数**完全相等**。<span class="marginnote">如果几何用高一阶形函数、位移用低一阶，叫「超参数（superparametric）」；反过来位移更高阶，叫「亚参数（subparametric）」——梁单元常用亚参数。等参数是兼顾精度与实现难度的黄金选择。</span>

**关键结论：等参数映射让单元能表达任意四边形、甚至曲边（更高阶时）。** 我们不再需要为每种形状重写插值——标准单元 + 节点坐标，就完全确定了实际单元的形状。

## 2 雅可比矩阵：局部伸缩的账本

形函数 $N_i$ 是 $(\xi,\eta)$ 的函数，而应变需要对物理坐标 $(x,y)$ 求导。两者之间需要一座桥：**雅可比矩阵（Jacobian matrix）**

$$
\boldsymbol{J} = \begin{bmatrix} \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\ \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta} \end{bmatrix} = \begin{bmatrix} \sum_i \frac{\partial N_i}{\partial \xi} x_i & \sum_i \frac{\partial N_i}{\partial \xi} y_i \\ \sum_i \frac{\partial N_i}{\partial \eta} x_i & \sum_i \frac{\partial N_i}{\partial \eta} y_i \end{bmatrix}
$$

偏导链式法则告诉我们物理导数与自然导数之间满足：

$$
\begin{bmatrix} \frac{\partial N_i}{\partial x} \\ \frac{\partial N_i}{\partial y} \end{bmatrix} = \boldsymbol{J}^{-1} \begin{bmatrix} \frac{\partial N_i}{\partial \xi} \\ \frac{\partial N_i}{\partial \eta} \end{bmatrix}
$$

而积分换元给出面积关系：$d\Omega = \det(\boldsymbol{J})\, d\xi\, d\eta$。<span class="marginnote">$\det\boldsymbol{J}$ 是「局部面积缩放因子」：标准单元里 $d\xi\,d\eta$ 的小块，映射到物理空间后被放大 $\det\boldsymbol{J}$ 倍。如果单元畸变严重，$\det\boldsymbol{J}$ 会在某处变负——意味着网格「翻转」，单元自相重叠，这是最严重的网格错误之一。</span>

**辨析｜易错点：** 四边形每个角的内角都不能接近或超过 $180^\circ$，否则 $\det\boldsymbol{J}$ 趋近零甚至变号。判断网格质量最常用的指标——**雅可比行列式**，说的就是它。商用软件里检查 `Jacobian Ratio`，标准就是看 $\det\boldsymbol{J}$ 的最小值与最大值之比。

## 3 数值积分：高斯积分

等参数单元的应变矩阵 $\boldsymbol{B}$ 含 $\boldsymbol{J}^{-1}$，不再恒为常数，单元刚度只能数值积分：

$$
\boldsymbol{k}^e = \int_{-1}^{1}\int_{-1}^{1} \boldsymbol{B}^{\mathsf{T}}(\xi,\eta)\, \boldsymbol{D}\, \boldsymbol{B}(\xi,\eta) \, \det\boldsymbol{J} \, d\xi\, d\eta
$$

**高斯积分（Gauss quadrature）** 是标准选择。一维 $n$ 点高斯公式：

$$
\int_{-1}^{1} f(\xi)\, d\xi \approx \sum_{i=1}^{n} w_i f(\xi_i)
$$

高斯积分最迷人的性质是：**用 $n$ 个点可以精确积分 $2n-1$ 次多项式**——点取在最优化位置（Gauss 点），而不是等间距。二维张量积形式：$\int\int f \, d\xi d\eta \approx \sum_i \sum_j w_i w_j f(\xi_i, \eta_j)$。

| 单元类型 | 应变变化 | 常用积分方案 | 备注 |
| --- | --- | --- | --- |
| CST 三角形 | 常数 | 1 点（重心） | 精确，且最省 |
| Q4 四边形 | 线性 | 2×2 点 | 精确积分 |
| Q8/Q9 高阶 | 二次/三次 | 3×3 点 | 精确积分 |
| Q4 减缩积分 | 线性 | 1×1 点 | 低阶、易沙漏，但抗剪切锁死 |

## 4 公式解析：Q4 单元刚度的高斯积分流程

以四节点四边形为例，一步步走完数值积分。

**第一步，构造形函数及其自然坐标导数**。$N_i = \frac{1}{4}(1+\xi\xi_i)(1+\eta\eta_i)$，对 $(\xi,\eta)$ 求导得 $\partial N_i/\partial \xi = \xi_i(1+\eta\eta_i)/4$，$\partial N_i/\partial \eta = \eta_i(1+\xi\xi_i)/4$。

**第二步，组雅可比并求逆**。用节点坐标 $(x_i, y_i)$ 代入上一节的公式得 $\boldsymbol{J}(\xi,\eta)$，数值求逆得到 $\boldsymbol{J}^{-1}$ 与 $\det\boldsymbol{J}$。

**第三步，在 Gauss 点上组装并求和**。对每个 Gauss 点 $(\xi_g, \eta_g)$：

- 用 $\boldsymbol{J}^{-1}$ 把 $\partial N_i/\partial\xi,\partial N_i/\partial\eta$ 转到物理导数，填入 $\boldsymbol{B}$ 矩阵；
- 累加被积函数：$\boldsymbol{k}^e \mathrel{+}= w_g \, \boldsymbol{B}^{\mathsf{T}} \boldsymbol{D} \boldsymbol{B} \, \det\boldsymbol{J}$。

Q4 的 $\boldsymbol{B}$ 是 $(\xi,\eta)$ 的线性函数，$\boldsymbol{B}^{\mathsf{T}}\boldsymbol{D}\boldsymbol{B}$ 是二次多项式，乘 $\det\boldsymbol{J}$ 后仍不高于二次，所以 **2×2 高斯积分给出精确结果**。<span class="marginnote">「用 2×2 高斯点积分 Q4 刚度」是手算、编程、校核时都要记住的常识。选错积分阶数不会「报错」，只会静默地给出错误结果——减缩积分（1×1）会出现沙漏模式，这是有限元里最隐蔽的数值病态之一。</span>

## 5 减缩积分与沙漏

出于成本考虑，有些单元故意用低于精确所需的积分阶数——**减缩积分（reduced integration）**。它让单元变「软」，能缓解**剪切锁死（shear locking）**（低阶单元在纯弯曲问题里过分刚硬的现象），但也引入了**零能量模式（hourglassing，沙漏）**：单元可以按某种模式变形而不产生任何应变能，网格像揉纸一样扭曲。

**关键结论：减缩积分是一把双刃剑。** 用得好，它大幅提升弯曲问题的精度；用得不好，沙漏模式会让结果彻底失真。商用软件（如 Abaqus 的 C3D8R）默认开减缩积分并配沙漏控制，正是这个权衡的工程化产物。<span class="marginnote">「剪切锁死」与「沙漏」是低阶单元的两大宿敌，也是一对矛盾：完全积分会锁死，减缩积分会沙漏。高阶单元（Q8、C3D20）同时缓解两者，代价是自由度暴涨。理解这对矛盾，才算真正懂了低阶单元的脾气。</span>

## 6 积分阶次速查与常见疑问

**常见疑问：为什么叫「等参数」**——因为几何与位移共用同一组形函数。若几何插值阶次更高，则单元可以表达曲边而位移仍是低阶，这类「超参数」单元能大幅提升几何逼近能力，代价是位移精度受限于较低阶的形函数——工程里以「等参数」最均衡。

**一个数值实验的预期**：把同一块悬臂板分别用 Q4 全积分（2×2）、Q4 减缩积分（1×1）与 Q8 单元计算端部位移：Q8 最准，Q4 全积分略刚，Q4 减缩积分在小网格时可能偏软甚至沙漏。这个实验能让「锁死、沙漏、阶次」三个概念同时落地。

**积分阶次选择速查**：

| 单元 | $\boldsymbol{B}$ 的次数 | 被积函数最高次 | 精确积分 | 常用方案 |
| --- | --- | --- | --- | --- |
| CST T3 | 0 | 0 | 1 点 | 1 点 |
| Q4 | 1 | 2 | 2×2 | 2×2 |
| Q8 | 2 | 4 | 3×3 | 3×3 |
| Q9 | 2 | 4 | 3×3 | 3×3 |

- 高阶单元对网格畸变更敏感：同一网格下 Q8 比 Q4 更容易因畸变而精度骤降。
- 减缩积分的沙漏模式在动力分析中尤须警惕——沙漏位移模式会以零刚度高频振荡，污染时程结果。

## 7 小结

- **等参数映射**：同一组形函数同时插值几何与位移，任意四边形统一映射到标准单元。
- **雅可比矩阵** $\boldsymbol{J}$：连接自然坐标与物理坐标，$d\Omega = \det\boldsymbol{J}\, d\xi\,d\eta$；$\det\boldsymbol{J} \le 0$ 表示网格翻转。
- **高斯积分**：$n$ 点精确积分 $2n-1$
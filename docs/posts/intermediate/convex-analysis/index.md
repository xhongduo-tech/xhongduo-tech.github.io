---
pageClass: plain-doc
---

# 凸分析

凸分析以凸集与凸函数为核心对象，系统研究分离定理、次梯度、共轭函数与对偶性，
是最优化理论、变分分析与博弈论共同的分析基础。
它把「凸性」这一几何结构转译为可用的分析工具，
是通向凸优化算法、非光滑分析与广义导数（次梯度、次微分）理论的必经之路。

## 对标教材

- R. Tyrrell Rockafellar, "Convex Analysis" (Princeton University Press)
- Stephen Boyd & Lieven Vandenberghe, "Convex Optimization" (Cambridge University Press)

## 主题规划

<ProgressGrid cat="intermediate/convex-analysis" />

### 第1篇 凸集与分离定理

- [x] [凸集与凸组合：定义、基本性质与凸包](./convex-set-convex-combination)
- [x] [凸集的保运算：交、仿射变换、线性和与投影](./convex-set-operations)
- [x] [相对内部、仿射包与凸集的维数](./relative-interior-affine-hull)
- [x] [分离定理：超平面分离凸集与严格分离](./separation-theorem)
- [x] [支撑函数与支撑超平面](./support-function-supporting-hyperplane)
- [x] [凸锥与对偶锥：极与对偶运算](./convex-cone-dual-cone)
- [x] [极值点、极值方向与 Carathéodory 定理](./extreme-points-carathéodory)

### 第2篇 凸函数

- [x] [凸函数：定义、上图与判定条件](./convex-function-definition)
- [x] [凸函数的一阶与二阶刻画](./first-second-order-characterization)
- [x] [凸函数的运算：和、复合、逐点最大与上确界](./convex-function-operations)
- [x] [闭凸函数与凸函数的闭包](./closed-convex-functions)
- [x] [凸函数的连续性：局部 Lipschitz 与有界性](./continuity-lipschitz)
- [x] [水平集、拟凸函数与凸性的推广](./level-sets-quasiconvex)

### 第3篇 次梯度与微分理论

- [x] [方向导数与次梯度：定义与存在性](./directional-derivative-subgradient)
- [x] [次微分的运算规则：和、复合与逐点极大](./subdifferential-calculus)
- [x] [极小化问题的次梯度最优性条件](./subgradient-optimality)
- [x] [凸函数的可微性：梯度的单调性与连续性](./differentiability-monotonicity)
- [x] [次梯度方法：非光滑凸优化的数值基础](./subgradient-method)

### 第4篇 共轭函数与对偶性

- [x] [共轭函数：定义与 Fenchel 不等式](./conjugate-function-fenchel)
- [x] [双共轭与凸函数的闭包刻画](./biconjugate-closure)
- [x] [Lagrange 对偶：凸优化对偶问题的构造](./lagrange-duality)
- [x] [弱对偶、强对偶与 Slater 条件](./weak-strong-duality-slater)
- [x] [鞍点刻画与对偶间隙的几何解释](./saddle-point-duality-gap)
- [x] [KKT 最优性条件：Karush–Kuhn–Tucker](./kkt-conditions)
- [x] [Fenchel 对偶定理](./fenchel-duality-theorem)

### 第5篇 多面体与凸优化应用

- [x] [多面体与多面体凸函数](./polyhedra-polyhedral-convex-functions)
- [x] [Farkas 引理与线性不等式组的可解性](./farkas-lemma)
- [x] [线性规划及其对偶理论](./linear-programming-duality)
- [x] [二次规划与锥规划（SOCP / SDP）](./qp-conic-programming)
- [x] [凸优化的典型应用：最小二乘、回归与支持向量机](./convex-optimization-applications)

### 第6篇

- [x] [凸集（凸组合、凸包、凸锥）](./convex-sets-hull-cones)
- [x] [凸函数（定义、Jensen 不等式、下半连续）](./convex-functions-jensen-lsc)
- [x] [分离定理（超平面分离、支撑超平面）](./separation-supporting-hyperplanes)
- [x] [共轭函数与对偶（Fenchel 共轭）](./fenchel-conjugate-duality)
- [x] [次微分（次梯度、Moreau-Rockafellar 理论）](./subdifferential-moreau-rockafellar)
- [x] [最优性条件（KKT 条件的凸分析形式）](./optimality-conditions-kkt-convex)
- [x] [极值表示与凸几何（极点、Minkowski 定理）](./extreme-points-minkowski)
- [x] [应用（对偶算法、机器学习中的凸方法）](./duality-algorithms-machine-learning)

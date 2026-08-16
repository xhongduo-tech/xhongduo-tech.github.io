---
pageClass: plain-doc
---

# 最优化理论

对标《最优化方法》与 Boyd《Convex Optimization》入门部分，从凸分析基础出发，系统覆盖线性规划、无约束与约束优化的经典算法，并延伸到整数规划、多目标、随机与全局优化的入门内容。学完这一分类，即具备阅读优化领域文献与建模求解实际问题的能力。

## 主题规划

<ProgressGrid cat="intermediate/optimization" />


### 第一篇 凸集与凸函数

- [x] [仿射集与凸集：定义与基本例子](./affine-and-convex-sets)
- [x] [凸组合、凸包与凸锥](./convex-hull-and-cone)
- [x] [超平面、半空间与多面体](./hyperplanes-halfspaces-polyhedra)
- [x] [保凸运算：仿射变换、透视函数与线性分式函数](./convexity-preserving-operations)
- [x] [分离超平面定理与支撑超平面定理](./separating-and-supporting-hyperplanes)
- [x] [凸函数的定义与等价刻画](./convex-function-definition)
- [x] [凸函数的一阶条件与二阶条件](./convex-function-conditions)
- [x] [上镜图（epigraph）与下水平集](./epigraph-sublevel-sets)
- [x] [Jensen 不等式及其推论](./jensen-inequality)
- [x] [常见凸函数与凹函数的例子](./common-convex-functions-examples)
- [x] [保凸运算：非负加权和、最大值、复合函数规则](./convex-operations-sums-max-composition)
- [x] [共轭函数（conjugate function）](./conjugate-function)
- [x] [拟凸函数（quasiconvex function）初步](./quasiconvex-functions)

### 第二篇 凸优化问题

- [x] [优化问题的标准形式：目标、约束与最优值](./optimization-problem-standard-form)
- [x] [凸优化问题的定义与局部最优即全局最优](./convex-optimization-problem)
- [x] [等价变换：消去等式约束、引入松弛变量、上镜图形式](./equivalent-reformulations)
- [x] [线性规划（LP）的标准形式与常见建模](./linear-programming-standard-form)
- [x] [二次规划（QP）与二次约束二次规划（QCQP）](./quadratic-programming-qcqp)
- [x] [二阶锥规划（SOCP）初步](./second-order-cone-programming)
- [x] [半定规划（SDP）初步](./semidefinite-programming)
- [x] [几何规划（GP）初步](./geometric-programming)
- [x] [最小二乘、正则化与鲁棒优化建模实例](./least-squares-regularization-robust)

### 第三篇 对偶理论与最优性条件

- [x] [拉格朗日函数与拉格朗日对偶函数](./lagrangian-dual-function)
- [x] [对偶函数的凹性与最优值下界](./dual-function-concavity-lower-bound)
- [x] [拉格朗日对偶问题](./lagrangian-dual-problem)
- [x] [弱对偶与强对偶](./weak-strong-duality)
- [x] [Slater 约束品性（constraint qualification）](./slater-constraint-qualification)
- [x] [互补松弛条件](./complementary-slackness)
- [x] [KKT 条件：必要性与充分性](./kkt-conditions)
- [x] [对偶视角下的灵敏度分析](./sensitivity-analysis-duality)
- [x] [Farkas 引理与择一定理](./farkas-lemma)

### 第四篇 线性规划

- [x] [线性规划的几何直观：可行域、顶点与最优解](./lp-geometric-intuition)
- [x] [基、基本可行解与顶点的一一对应](./basic-feasible-solutions-vertices)
- [x] [单纯形法的基本思想：换基迭代](./simplex-method-idea)
- [x] [单纯形表的构造与转轴（pivot）运算](./simplex-tableau-pivot)
- [x] [退化、循环与 Bland 规则](./degeneracy-cycling-bland)
- [x] [两阶段法与大 M 法：寻找初始基本可行解](./two-phase-big-m)
- [x] [线性规划的对偶问题与对偶单纯形法](./lp-duality-dual-simplex)
- [x] [影子价格与灵敏度分析](./shadow-price-sensitivity)
- [x] [内点法概述：中心路径与原始-对偶算法](./interior-point-methods-overview)
- [x] [椭球法与线性规划的多项式可解性](./ellipsoid-method)

### 第五篇 无约束优化

- [x] [无约束问题的一阶与二阶最优性条件](./unconstrained-optimality-conditions)
- [x] [下降方向与迭代算法的一般框架](./descent-methods-framework)
- [x] [线搜索：精确线搜索与 Armijo 回溯准则](./line-search-armijo)
- [x] [Wolfe 条件与强 Wolfe 条件](./wolfe-conditions)
- [x] [梯度下降法及其收敛性分析](./gradient-descent)
- [x] [最速下降与预处理思想](./steepest-descent-preconditioning)
- [x] [牛顿法：推导、局部二次收敛与阻尼牛顿法](./newtons-method)
- [x] [拟牛顿法：割线条件与 DFP 公式](./quasi-newton-dfp)
- [x] [BFGS 公式与有限内存 L-BFGS](./bfgs-lbfgs)
- [x] [共轭梯度法：线性共轭梯度](./conjugate-gradient-linear)
- [x] [非线性共轭梯度：FR 与 PRP 公式](./nonlinear-conjugate-gradient)
- [x] [信赖域方法与 Levenberg–Marquardt 算法](./trust-region-levenberg-marquardt)

### 第六篇 约束优化

- [x] [等式约束问题的拉格朗日乘子法](./lagrange-multipliers-equality)
- [x] [不等式约束与 KKT 条件的几何解释](./inequality-constraints-kkt-geometry)
- [x] [外罚函数法](./exterior-penalty-methods)
- [x] [内点障碍法（log barrier method）](./interior-point-barrier-method)
- [x] [增广拉格朗日方法与乘子法](./augmented-lagrangian)
- [x] [交替方向乘子法（ADMM）初步](./admm)
- [x] [投影梯度法与邻近算子](./projected-gradient-proximal-operator)
- [x] [序列二次规划（SQP）初步](./sequential-quadratic-programming)

### 第七篇 二次规划

- [x] [二次规划的标准形式与凸性判别](./qp-standard-form-convexity)
- [x] [等式约束二次规划的解析解](./equality-constrained-qp)
- [x] [积极集法（active-set method）](./active-set-method)
- [x] [二次规划的内点法](./qp-interior-point)
- [x] [二次规划的典型应用：SVM 与投资组合优化](./qp-applications-svm-portfolio)

### 第八篇 整数规划与组合优化初步

- [x] [整数规划建模：0-1 变量与逻辑约束](./integer-programming-modeling)
- [x] [线性松弛与整数间隙（integrality gap）](./linear-relaxation-integrality-gap)
- [x] [分支定界法（branch and bound）](./branch-and-bound)
- [x] [割平面法与 Gomory 割](./cutting-plane-gomory)
- [x] [背包问题与指派问题](./knapsack-assignment)
- [x] [图上的经典问题：最短路、最大流与最小割](./shortest-path-max-flow-min-cut)
- [x] [匹配问题与匈牙利算法](./matching-hungarian-algorithm)

### 第九篇 多目标优化

- [x] [Pareto 最优与有效前沿](./pareto-optimal-efficient-frontier)
- [x] [加权和方法及其局限](./weighted-sum-method)
- [x] [ε-约束法](./epsilon-constraint-method)
- [x] [目标规划（goal programming）初步](./goal-programming)

### 第十篇 随机优化

- [x] [随机逼近与 Robbins–Monro 算法](./robbins-monro-stochastic-approximation)
- [x] [随机梯度下降（SGD）：从批量梯度到随机采样](./stochastic-gradient-descent)
- [x] [SGD 的步长规则与收敛性分析](./sgd-step-size-convergence)
- [x] [小批量、动量与方差缩减（SVRG）思想](./mini-batch-momentum-svrg)
- [x] [期望约束与随机规划初步](./stochastic-programming-expectation-constraints)

### 第十一篇 全局优化与启发式算法

- [x] [全局优化的困难性与 NP-hard 背景](./global-optimization-np-hard)
- [x] [模拟退火（simulated annealing）：Metropolis 准则与退火计划](./simulated-annealing)
- [x] [遗传算法（genetic algorithm）：编码、选择、交叉与变异](./genetic-algorithm)
- [x] [禁忌搜索与局部搜索策略](./tabu-search-local-search)
- [x] [粒子群优化（PSO）初步](./particle-swarm-optimization)
- [x] [启发式算法的评价：收敛保证与实验比较](./heuristic-evaluation)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [ ] 优化问题与最优性条件（无约束一阶二阶条件）
- [ ] 线搜索方法（最速下降、步长策略）
- [ ] 牛顿法与拟牛顿法（BFGS、收敛性）
- [ ] 信赖域方法（模型函数、狗腿法）
- [ ] 最小二乘问题（线性/非线性、GN/LM 算法）
- [ ] 线性规划（单纯形法、对偶理论）
- [ ] 凸优化问题类（QP、SOCP、SDP）
- [ ] 约束优化（KKT 条件、罚函数、增广拉格朗日）
- [ ] 内点法（原始对偶内点法）
- [ ] 大规模优化与应用（一阶方法、ADMM、机器学习应用）

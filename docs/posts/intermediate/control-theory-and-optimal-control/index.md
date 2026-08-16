---
pageClass: plain-doc
---

# 控制论与最优控制

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。本专题覆盖运筹学与控制论的核心理论：从状态空间建模、能控能观性、Lyapunov 稳定性，到状态反馈与最优控制的完整链条。

## 对标教材

- Eduardo D. Sontag, "Mathematical Control Theory: Deterministic Finite Dimensional Systems" (2nd ed., Springer, 1998)
- Katsuhiko Ogata, "Modern Control Engineering" (5th ed., Prentice Hall, 2010)
- Donald E. Kirk, "Optimal Control Theory: An Introduction" (Dover, 2004)

## 主题规划

<ProgressGrid cat="intermediate/control-theory-and-optimal-control" />

### 第1篇 系统建模与状态空间描述

- [x] [控制系统的数学描述：微分方程与传递函数](./control-systems-mathematical-description)
- [x] [状态空间模型与系统框图](./state-space-model-block-diagrams)
- [x] [状态方程的求解：矩阵指数与状态转移矩阵](./state-equation-solution-matrix-exponential)
- [x] [从传递函数到状态空间：实现理论](./transfer-function-to-state-space-realization)
- [x] [非线性系统与平衡点的线性化](./nonlinear-systems-linearization)

### 第2篇 能控性与能观性

- [x] [能控性：定义、条件与可控 Gramian](./controllability)
- [x] [能观性：定义、条件与可观 Gramian](./observability)
- [x] [能控性与能观性的对偶原理](./duality-controllability-observability)
- [x] [Kalman 规范分解与不可简约实现](./kalman-canonical-decomposition)
- [x] [最小实现与传递函数的关系](./minimal-realization)

### 第3篇 稳定性与反馈设计

- [x] [Lyapunov 稳定性理论：直接法与判据](./lyapunov-stability)
- [x] [线性系统的 Lyapunov 方程与稳定性判定](./lyapunov-equation-linear-systems)
- [x] [瞬态响应与稳态误差分析](./transient-response-steady-state-error)
- [x] [状态反馈与极点配置](./state-feedback-pole-placement)
- [x] [状态观测器：全维与降维设计](./state-observer)
- [x] [分离原理与闭环系统综合](./separation-principle)

### 第4篇 最优控制理论

- [x] [最优控制问题的一般提法](./optimal-control-problem-formulation)
- [x] [变分法与最优控制：Euler-Lagrange 方程](./calculus-of-variations-euler-lagrange)
- [x] [变分法求解最优控制：横截条件与边界](./transversality-conditions)
- [x] [Pontryagin 极大值原理](./pontryagin-maximum-principle)
- [x] [线性二次型调节器（LQR）与 Riccati 方程](./lqr-riccati)
- [x] [动态规划与 Bellman 方程](./dynamic-programming-bellman)
- [x] [时间最优与最省燃料控制](./time-optimal-fuel-optimal)

### 第5篇

- [ ] 控制系统建模（微分方程、传递函数）
- [ ] 时域分析（阶跃响应、性能指标）
- [ ] 稳定性分析（Routh 判据、Nyquist 判据）
- [ ] 根轨迹与频域设计（Bode 图、校正）
- [ ] 状态空间方法（状态方程、实现）
- [ ] 能控性与能观性（Kalman 判据）
- [ ] 状态反馈与观测器（极点配置、Luenberger 观测器）
- [ ] 李雅普诺夫稳定性（直接法、拉萨尔原理）
- [ ] 变分法与最优控制（Euler-Lagrange 方程）
- [ ] 庞特里亚金极大值原理（最小时间/燃料问题）
- [ ] 动态规划与 LQR/LQG（HJB 方程、卡尔曼滤波）
- [ ] 现代控制前沿（鲁棒控制 H∞、模型预测控制）

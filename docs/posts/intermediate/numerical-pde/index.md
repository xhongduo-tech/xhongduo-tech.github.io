---
pageClass: plain-doc
---

# 偏微分方程数值解

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Randall J. LeVeque, "Finite Difference Methods for Ordinary and Partial Differential Equations" (SIAM, 2007)
- John C. Strikwerda, "Finite Difference Schemes and Partial Differential Equations" (2nd ed., 2004)
- Claes Johnson, "Numerical Solution of Partial Differential Equations by the Finite Element Method" (1987)

## 主题规划

<ProgressGrid cat="intermediate/numerical-pde" />

### 第1篇

- [x] [有限差分法基础 (LeVeque Ch. 1-3)](./finite-difference-basics)
- [x] [椭圆方程与迭代求解 (LeVeque Ch. 4)](./elliptic-iterative-solvers)
- [x] [一维抛物方程的稳定性分析 (LeVeque Ch. 9)](./parabolic-stability-analysis)
- [x] [双曲方程与 CFL 条件 (LeVeque Ch. 10)](./hyperbolic-cfl-condition)
- [x] [Lax-Richtmyer 等价定理 (Strikwerda Ch. 2)](./lax-richtmyer-equivalence)
- [x] [有限元变分形式 (C. Johnson Ch. 1-2)](./finite-element-variational)
- [x] [Sobolev 空间与误差估计 (C. Johnson Ch. 4-5)](./sobolev-error-estimates)
- [x] [多维问题与边界条件处理 (LeVeque Ch. 13)](./multidimensional-boundary-conditions)

### 第2篇

- [x] [非线性守恒律与激波捕捉：Riemann 问题与 Godunov 型格式 (LeVeque Ch. 11-12)](./riemann-problem-godunov)
- [x] [有限体积法 (LeVeque Part V)](./finite-volume-methods)
- [x] [抛物方程分裂方法：ADI 与算子分裂 (LeVeque Ch. 6-8)](./adi-operator-splitting)

---
pageClass: plain-doc
---

# 数值分析

对标李庆扬《数值分析》，覆盖数值算法的基本理论与九大核心主题，逐节成文。

## 主题规划

<ProgressGrid cat="intermediate/numerical-analysis" />


### 误差与数值算法稳定性

- [x] [数值分析的研究对象与特点](./what-is-numerical-analysis)
- [x] [误差的来源与分类：模型误差、观测误差、截断误差、舍入误差](./error-sources-classification)
- [x] [绝对误差、相对误差与有效数字](./absolute-relative-error)
- [x] [函数运算的误差估计](./error-propagation)
- [x] [病态问题与条件数（condition number）](./condition-number)
- [x] [算法的数值稳定性](./numerical-stability)
- [x] [避免误差危害的若干原则：避免相近数相减、避免大数吃小数、减少运算次数](./avoid-error-hazards)
- [x] [秦九韶算法（Horner 算法）与多项式求值](./qin-jiushao-horner)

### 插值法

- [x] [插值问题的提法与多项式插值的存在唯一性](./interpolation-existence-uniqueness)
- [x] [拉格朗日插值多项式（Lagrange interpolation）](./lagrange-interpolation)
- [x] [插值余项与误差估计](./interpolation-remainder)
- [x] [均差（差商）与牛顿插值多项式（Newton interpolation）](./newton-interpolation-divided-difference)
- [x] [差分与等距节点插值公式：牛顿前插与后插公式](./finite-difference-newton-formulas)
- [x] [埃尔米特插值（Hermite interpolation）](./hermite-interpolation)
- [x] [分段低次插值：分段线性插值与分段三次埃尔米特插值](./piecewise-low-order-interpolation)
- [x] [龙格现象（Runge phenomenon）与高次插值的局限](./runge-phenomenon)
- [x] [三次样条插值（cubic spline）：三弯矩方程](./cubic-spline-moment)
- [x] [三次样条插值：三转角方程与边界条件](./cubic-spline-slope)

### 函数逼近与曲线拟合

- [x] [函数逼近的基本概念：范数与内积空间](./function-approximation-norm-inner-product)
- [x] [正交多项式：勒让德多项式（Legendre polynomials）](./legendre-polynomials)
- [x] [切比雪夫多项式（Chebyshev polynomials）及其性质](./chebyshev-polynomials)
- [x] [其他常用正交多项式：拉盖尔（Laguerre）与埃尔米特（Hermite）多项式](./laguerre-hermite-polynomials)
- [x] [最佳一致逼近多项式](./best-uniform-approximation)
- [x] [最佳平方逼近](./best-square-approximation)
- [x] [切比雪夫插值与近似最佳逼近](./chebyshev-interpolation)
- [x] [曲线拟合的最小二乘法（least squares method）](./least-squares-fitting)
- [x] [用正交多项式作最小二乘拟合](./orthogonal-least-squares)
- [x] [矛盾方程组与线性最小二乘问题](./linear-least-squares-normal-equations)

### 数值积分与数值微分

- [x] [数值求积的基本思想与代数精度](./quadrature-algebraic-precision)
- [x] [牛顿-柯特斯公式（Newton-Cotes formulas）](./newton-cotes-formulas)
- [x] [梯形公式与辛普森公式（Simpson's rule）的余项](./trapezoid-simpson-remainder)
- [x] [复化求积公式：复化梯形与复化辛普森](./composite-quadrature)
- [x] [复化求积的误差估计与逐次分半法](./quadrature-error-halving)
- [x] [龙贝格求积算法（Romberg integration）：理查森外推（Richardson extrapolation）](./romberg-integration-richardson)
- [x] [高斯求积公式（Gaussian quadrature）及其构造](./gaussian-quadrature)
- [x] [高斯-勒让德求积公式](./gauss-legendre-quadrature)
- [x] [高斯-切比雪夫求积公式与带权高斯公式](./gauss-chebyshev-quadrature)
- [x] [数值微分：插值型求导公式与三点公式](./numerical-differentiation)

### 线性方程组的直接解法

- [x] [高斯消去法（Gaussian elimination）](./gaussian-elimination)
- [x] [高斯消去法的计算量与矩阵三角分解的关系](./gaussian-elimination-lu)
- [x] [列主元消去法与全主元消去法](./partial-pivoting)
- [x] [高斯-若尔当消去法与矩阵求逆](./gauss-jordan-inverse)
- [x] [矩阵的 LU 分解：杜利特尔分解（Doolittle）与克劳特分解（Crout）](./lu-doolittle-crout)
- [x] [三对角方程组的追赶法（Thomas algorithm）](./thomas-algorithm)
- [x] [对称正定矩阵的平方根法（Cholesky 分解）](./cholesky-decomposition)
- [x] [改进的平方根法（LDLᵀ 分解）](./ldl-decomposition)
- [x] [向量范数与矩阵范数](./vector-matrix-norms)
- [x] [线性方程组的误差分析：条件数与病态方程组](./linear-system-condition-number)
- [x] [迭代改善法（iterative refinement）](./iterative-refinement)

### 线性方程组的迭代解法

- [x] [迭代法的基本思想与分裂格式](./iterative-splitting)
- [x] [雅可比迭代法（Jacobi method）](./jacobi-method)
- [x] [高斯-赛德尔迭代法（Gauss-Seidel method）](./gauss-seidel-method)
- [x] [逐次超松弛迭代法（SOR method）与松弛因子的选择](./sor-method)
- [x] [迭代法收敛的基本定理：谱半径条件](./iteration-convergence-spectral-radius)
- [x] [严格对角占优与不可约对角占优矩阵的收敛性](./diagonally-dominant-convergence)
- [x] [对称正定矩阵上 SOR 的收敛性](./spd-sor-convergence)
- [x] [迭代法的误差估计与终止准则](./iteration-error-termination)

### 矩阵特征值计算

- [x] [特征值的性质与估计：格什戈林圆盘定理（Gershgorin circle theorem）](./gershgorin-circle-theorem)
- [x] [幂法（power method）求主特征值与主特征向量](./power-method)
- [x] [幂法的加速：原点平移法与瑞利商加速（Rayleigh quotient）](./power-method-acceleration)
- [x] [反幂法（inverse power method）](./inverse-power-method)
- [x] [豪斯霍尔德变换（Householder transformation）与约化矩阵为三对角形](./householder-reduction)
- [x] [矩阵的 QR 分解](./qr-decomposition)
- [x] [基本 QR 算法及其收敛性](./basic-qr-algorithm)
- [x] [带原点位移的 QR 算法与海森伯格矩阵（Hessenberg matrix）](./shifted-qr-algorithm)
- [x] [雅可比方法求实对称矩阵全部特征值](./jacobi-eigenvalue-method)

### 非线性方程求根

- [x] [方程求根的基本步骤与根的搜索](./root-finding-basics)
- [x] [二分法（bisection method）及其收敛性](./bisection-method)
- [x] [不动点迭代法（fixed-point iteration）](./fixed-point-iteration)
- [x] [不动点迭代的收敛性定理与局部收敛](./fixed-point-convergence)
- [x] [迭代收敛阶的概念与判定](./convergence-order)
- [x] [迭代加速：埃特肯方法（Aitken's Δ² method）与斯特芬森迭代（Steffensen's method）](./aitken-steffensen)
- [x] [牛顿迭代法（Newton's method）及其局部收敛性](./newton-method)
- [x] [重根情形的牛顿法修正](./newton-multiple-roots)
- [x] [弦截法（secant method）](./secant-method)
- [x] [抛物线法（Müller's method）](./mullers-method)
- [x] [非线性方程组的牛顿法简介](./newton-nonlinear-systems)

### 常微分方程数值解法

- [x] [初值问题数值解法的基本概念：单步法与多步法](./ode-numerical-basics)
- [x] [欧拉方法（Euler's method）及其局部截断误差](./euler-method)
- [x] [后退欧拉方法与梯形方法](./implicit-euler-trapezoid)
- [x] [改进的欧拉方法与预测-校正格式](./improved-euler)
- [x] [龙格-库塔方法（Runge-Kutta methods）的基本思想](./runge-kutta-basics)
- [x] [二阶与三阶龙格-库塔方法](./rk2-rk3)
- [x] [经典四阶龙格-库塔方法](./rk4)
- [x] [变步长龙格-库塔方法与误差控制](./adaptive-rk)
- [x] [单步法的收敛性与稳定性](./single-step-stability)
- [x] [线性多步法（linear multistep methods）：阿当姆斯显式与隐式格式（Adams methods）](./adams-methods)
- [x] [阿当姆斯预测-校正系统](./adams-predictor-corrector)
- [x] [一阶方程组与高阶方程的数值解法](./systems-higher-order-ode)
- [x] [刚性问题（stiff problems）与绝对稳定域](./stiff-problems)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [x] [误差分析与浮点运算（舍入误差、条件数、稳定性）](./draft-f0b36a39d1)
- [x] [非线性方程求根（二分法、牛顿法、收敛阶）](./draft-5d33874f1b)
- [x] [线性方程组直接法（LU 分解、选主元）](./draft-4148adf73baa.md)
- [x] [迭代法（Jacobi、Gauss-Seidel、收敛性）](./draft-ab10ee242007.md)
- [x] [矩阵特征值计算（幂法、QR 算法）](./intermediate-numerical-analysis-qr-c39a66df.md)
- [x] [插值（拉格朗日、牛顿、样条插值）](./intermediate-numerical-analysis-38088d94.md)
- [x] [函数逼近（最佳一致逼近、最小二乘）](./intermediate-numerical-analysis-7db25b98.md)
- [x] [数值积分与微分（Newton-Cotes、高斯求积）](./intermediate-numerical-analysis-newton-cotes-0f3b1ce0.md)
- [x] [常微分方程数值解（Euler、Runge-Kutta、刚性问题）](./intermediate-numerical-analysis-eulerrunge-kutta-9466842b.md)
- [x] [偏微分方程数值方法（有限差分、有限元初步）](./intermediate-numerical-analysis-2b581f28.md)
- [x] [快速算法（FFT、多 grid 方法简介）](./intermediate-numerical-analysis-fft-grid-353903ad.md)
- [x] [数值优化与软件实践（MATLAB/Python 实现）](./intermediate-numerical-analysis-matlab-python-40ecb13e.md)

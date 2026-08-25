---
pageClass: plain-doc
---

# 偏微分方程

偏微分方程是描述自然界连续介质运动与场的数学语言，本分类对标《数学物理方程》（谷超豪、姜礼尚等）的章节体系，系统梳理三大经典方程（波动、热传导、拉普拉斯）与傅里叶方法、特殊函数、广义函数及弱解理论的全部入门内容。

## 主题规划

<ProgressGrid cat="intermediate/pde" />


### 第一篇 偏微分方程的基本概念

- [x] [偏微分方程的定义与实例（数学物理中的三类典型方程）](./what-is-pde)
- [x] [方程的阶、线性与非线性、齐次与非齐次](./order-linearity-homogeneous)
- [x] [解、通解与特解的概念](./solutions-general-particular)
- [x] [定解条件：初始条件与边界条件](./initial-boundary-conditions)
- [x] [三类边界条件（Dirichlet、Neumann、Robin）](./boundary-condition-types)
- [x] [定解问题的提法与适定性（存在性、唯一性、稳定性）](./well-posedness)
- [x] [叠加原理](./superposition-principle)

### 第二篇 二阶线性方程的分类与标准化

- [x] [两个自变量的二阶线性方程的一般形式与特征方程](./general-form-characteristics)
- [x] [双曲型、抛物型、椭圆型的判别](./classification-hyperbolic-parabolic-elliptic)
- [x] [双曲型方程的标准形（弦振动方程）](./canonical-form-hyperbolic)
- [x] [抛物型方程的标准形（热传导方程）](./canonical-form-parabolic)
- [x] [椭圆型方程的标准形（拉普拉斯方程）](./canonical-form-elliptic)
- [x] [常系数方程的化简与混合型方程举例（Tricomi 方程）](./constant-coefficient-tricomi)
- [x] [多个自变量的二阶线性方程的分类](./classification-multivariable)

### 第三篇 一阶偏微分方程

- [x] [一阶线性齐次方程与常微分方程组首次积分](./first-order-linear-homogeneous-first-integrals)
- [x] [特征线法（Method of Characteristics）解柯西问题](./method-of-characteristics)
- [x] [一阶拟线性方程的几何理论](./quasilinear-geometric-theory)
- [x] [一阶非线性方程与 Charpit 方法](./charpit-method)
- [x] [完全积分、奇异积分与包络](./complete-singular-integrals-envelope)
- [x] [Hamilton–Jacobi 方程初步](./hamilton-jacobi-equation)

### 第四篇 波动方程

- [x] [弦振动方程的导出与物理背景](./wave-equation-derivation)
- [x] [一维波动方程的达朗贝尔（d'Alembert）公式](./d-alembert-formula)
- [x] [传播波、依赖区间、决定区域与影响区域](./wave-propagation-domains)
- [x] [半无界弦问题与延拓法（奇延拓与偶延拓）](./semi-infinite-string-extension)
- [x] [有界弦的初边值问题：分离变量法](./bounded-string-separation-variables)
- [x] [非齐次方程的齐次化原理（Duhamel 原理）](./duhamel-principle)
- [x] [非齐次边界条件的处理](./nonhomogeneous-boundary)
- [x] [高维波动方程的球面平均法（Poisson 公式）](./spherical-mean-poisson-formula)
- [x] [降维法与二维波动方程](./descent-method-2d-wave)
- [x] [Huygens 原理与波的弥散](./huygens-principle-dispersion)
- [x] [能量不等式与解的唯一性](./energy-inequality-uniqueness)
- [x] [能量方法与解的稳定性](./energy-method-stability)

### 第五篇 热传导方程

- [x] [热传导方程的导出（傅里叶热传导定律）](./heat-equation-derivation)
- [x] [有界杆的初边值问题：分离变量法](./heat-bounded-rod-separation)
- [x] [傅里叶级数解的收敛性](./heat-fourier-series-convergence)
- [x] [圆形区域上的热传导问题](./heat-circular-domain)
- [x] [热传导方程柯西问题的泊松（Poisson）公式](./heat-cauchy-poisson-formula)
- [x] [傅里叶变换法解柯西问题](./heat-fourier-transform-cauchy)
- [x] [极值原理（最大值原理）](./maximum-principle-heat)
- [x] [由极值原理证明初边值问题解的唯一性与稳定性](./maximum-principle-uniqueness)
- [x] [柯西问题解的唯一性与稳定性](./heat-cauchy-uniqueness)
- [x] [解的渐近性态（t → ∞ 的衰减）](./heat-asymptotic-decay)

### 第六篇 拉普拉斯方程（位势方程）

- [x] [拉普拉斯方程与泊松方程的物理背景（引力场、静电场）](./laplace-poisson-physical-background)
- [x] [调和函数的定义与基本例子](./harmonic-functions-basics)
- [x] [格林（Green）公式及其推论](./green-formula)
- [x] [基本解（三维 r⁻¹ 与二维 ln r）](./fundamental-solution)
- [x] [调和函数的积分表达式](./harmonic-integral-representation)
- [x] [调和函数的平均值定理](./mean-value-theorem-harmonic)
- [x] [极值原理与解的唯一性](./maximum-principle-laplace)
- [x] [调和函数的解析性与可去奇点定理](./analyticity-removable-singularity)
- [x] [Harnack 不等式与刘维尔（Liouville）定理](./harnack-liouville)
- [x] [静电源像法（镜像法）](./method-of-images)
- [x] [球的格林函数与泊松公式](./green-function-ball-poisson)
- [x] [半空间、二维圆域等特殊区域的格林函数](./green-function-special-domains)
- [x] [牛曼（Neumann）内问题有解的相容性条件](./neumann-compatibility)
- [x] [用试探法解特殊区域上的边值问题](./trial-method-boundary-problems)

### 第七篇 傅里叶变换方法

- [x] [傅里叶级数回顾：正交函数系与展开定理](./fourier-series-review)
- [x] [傅里叶积分的导出](./fourier-integral-derivation)
- [x] [傅里叶变换与逆变换的定义](./fourier-transform-definition)
- [x] [傅里叶变换的基本性质（平移、微分、卷积）](./fourier-transform-properties)
- [x] [卷积定理](./convolution-theorem)
- [x] [傅里叶变换解热传导方程柯西问题](./fourier-transform-heat-cauchy)
- [x] [傅里叶变换解半无界区域问题（正弦与余弦变换）](./sine-cosine-transform-half-plane)
- [x] [拉普拉斯变换及其在定解问题中的应用](./laplace-transform-pde)
- [x] [傅里叶方法解高维热传导与波动方程](./fourier-method-higher-dim)

### 第八篇 特殊函数

- [x] [柱坐标系下方程的分离变量与贝塞尔（Bessel）方程的导出](./bessel-equation-derivation)
- [x] [贝塞尔方程的级数解与贝塞尔函数](./bessel-series-solution)
- [x] [贝塞尔函数的递推公式](./bessel-recurrence)
- [x] [贝塞尔函数的零点与渐近性质](./bessel-zeros-asymptotics)
- [x] [傅里叶–贝塞尔级数展开](./fourier-bessel-series)
- [x] [贝塞尔函数应用举例（圆柱体的热传导与圆膜振动）](./bessel-applications)
- [x] [球坐标系下方程的分离变量与勒让德（Legendre）方程的导出](./legendre-equation-derivation)
- [x] [勒让德多项式及其性质](./legendre-polynomials)
- [x] [勒让德多项式的正交性与展开定理](./legendre-orthogonality-expansion)
- [x] [连带的勒让德多项式与球函数](./associated-legendre-spherical-harmonics)
- [x] [勒让德多项式应用举例（球形区域上的位势问题）](./legendre-applications)

### 第九篇 广义函数与基本解初步

- [x] [集中量与 δ 函数的物理引入](./dirac-delta-physical-introduction)
- [x] [基本函数空间与广义函数的定义](./test-functions-distributions)
- [x] [广义函数的极限、导数与乘子运算](./distribution-operations)
- [x] [广义函数的卷积与傅里叶变换](./distribution-convolution-fourier)
- [x] [索伯列夫（Sobolev）空间初步](./sobolev-spaces)
- [x] [基本解的概念与三类典型方程的基本解](./fundamental-solutions-typical)

### 第十篇 变分方法与弱解初步

- [x] [变分问题与欧拉（Euler）方程](./calculus-of-variations-euler)
- [x] [边值问题与变分问题的等价性](./boundary-variational-equivalence)
- [x] [弱解（广义解）的定义](./weak-solutions)
- [x] [里兹（Ritz）方法](./ritz-method)
- [x] [伽辽金（Galerkin）方法](./galerkin-method)
- [x] [有限元方法初步](./finite-element-method)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [x] [方程分类与一阶方程（特征线法）](./draft-d9a0a0a648)
- [x] [波动方程（d'Alembert 公式、能量方法）](./draft-4ac3954d9e)
- [x] [热传导方程（基本解、极值原理）](./draft-bbd26a19ea9a.md)
- [x] [拉普拉斯方程（调和函数、平均值性质）](./draft-87688415c5d7.md)
- [x] [格林函数（Poisson 公式、镜像法）](./intermediate-pde-poisson-41b571b5.md)
- [x] [分离变量法与傅里叶级数（混合问题求解）](./intermediate-pde-ebf46dee.md)
- [x] [傅里叶变换方法（广义函数初步）](./intermediate-pde-40cec0f7.md)
- [x] [Sobolev 空间（弱导数、嵌入定理）](./intermediate-pde-sobolev-c65bebfe.md)
- [x] [二阶椭圆方程弱解理论（Lax-Milgram、变分方法）](./intermediate-pde-lax-milgram-0f4b4a86.md)
- [x] [正则性理论初步（椭圆正则性、Schauder 估计简介）](./intermediate-pde-schauder-68676332.md)
- [x] [抛物与双曲方程的弱解（Galerkin 方法）](./intermediate-pde-galerkin-8658b4d7.md)
- [x] [非线性方程初步（守恒律、激波、半线性方程）](./intermediate-pde-32653094.md)

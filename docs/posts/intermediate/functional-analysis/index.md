---
pageClass: plain-doc
---

# 泛函分析

泛函分析是研究无穷维空间及其上线性算子的学科，是现代分析数学的核心框架。本篇对标《泛函分析》（程其襄、张恭庆）的章节体系，覆盖从度量空间到谱理论初步的全部入门内容。

## 主题规划

<ProgressGrid cat="intermediate/functional-analysis" />


### 第一章 度量空间（Metric Space）

- [x] [度量空间的定义与基本例子](./metric-space)
- [x] [度量空间中的极限、稠密集与可分空间](./limit-dense-separable)
- [x] [连续映射](./continuous-mapping)
- [x] [柯西点列与完备度量空间](./cauchy-complete-metric)
- [x] [度量空间的完备化](./completion-metric-space)
- [x] [压缩映像原理（Banach 不动点定理）](./banach-fixed-point)
- [x] [压缩映像原理在微分方程与积分方程中的应用](./contraction-application-ode-ie)
- [x] [列紧集与紧集](./compact-set-sequential-compactness)
- [x] [全有界性与 Arzela-Ascoli 定理](./totally-bounded-arzela-ascoli)
- [x] [拓扑空间与度量空间的关系](./topology-metric-space)

### 第二章 赋范线性空间与 Banach 空间

- [x] [线性空间与范数的定义](./linear-space-norm-definition)
- [x] [经典例子：C[a,b]、l^p、L^p 空间](./classical-examples-c-lp)
- [x] [Banach 空间的定义与判别](./banach-space-definition)
- [x] [范数的等价性](./equivalent-norms)
- [x] [有限维赋范空间的刻画（Riesz 引理）](./finite-dim-normed-riesz-lemma)
- [x] [商空间与商范数](./quotient-space-quotient-norm)
- [x] [线性算子的基本概念与例子](./linear-operator-basics)

### 第三章 有界线性算子与连续线性泛函

- [x] [有界线性算子的定义与算子范数](./bounded-linear-operator-operator-norm)
- [x] [有界性与连续性的等价性](./boundedness-continuity-equivalence)
- [x] [算子空间 B(X,Y) 及其完备性](./operator-space-completeness)
- [x] [有界线性算子的代数性质：乘法、逆算子](./operator-algebra-inverse)
- [x] [连续线性泛函与共轭空间](./continuous-linear-functional-dual-space)
- [x] [具体空间上连续线性泛函的表示](./dual-space-representations)
- [x] [有限秩算子与近似](./finite-rank-operator-approximation)
- [x] [矩阵表示与无穷维算子的坐标化观点](./matrix-representation-coordinate-view)

### 第四章 内积空间与 Hilbert 空间

- [x] [内积空间的定义与柯西-施瓦茨不等式](./inner-product-cauchy-schwarz)
- [x] [由内积诱导的范数与 Hilbert 空间](./inner-product-norm-hilbert-space)
- [x] [平行四边形公式与内积空间的范数刻画](./parallelogram-law-norm-characterization)
- [x] [正交、正交系与格拉姆-施密特正交化](./orthogonality-gram-schmidt)
- [x] [正交分解定理](./orthogonal-decomposition-theorem)
- [x] [投影算子与最佳逼近](./projection-operator-best-approximation)
- [x] [规范正交系：贝塞尔不等式与帕塞瓦尔等式](./orthonormal-bessel-parseval)
- [x] [完全规范正交系与傅里叶级数的抽象观点](./complete-orthonormal-fourier-abstract)
- [x] [可分 Hilbert 空间与 l^2 的同构](./separable-hilbert-l2-isomorphism)
- [x] [Riesz 表示定理及其应用](./riesz-representation-theorem)
- [x] [伴随算子的定义与性质](./adjoint-operator)

### 第五章 Banach 空间的三大基本定理

- [x] [纲定理与 Baire 纲定理](./baire-category-theorem)
- [x] [一致有界原理（共鸣定理）](./uniform-boundedness-principle)
- [x] [一致有界原理的应用：傅里叶级数发散问题](./uniform-boundedness-fourier-divergence)
- [x] [开映射定理](./open-mapping-theorem)
- [x] [逆算子定理](./inverse-operator-theorem)
- [x] [闭图像定理](./closed-graph-theorem)
- [x] [闭算子及其例子](./closed-operator-examples)

### 第六章 Hahn-Banach 定理及其推论

- [x] [线性空间上的 Hahn-Banach 延拓定理](./hahn-banach-extension-algebraic)
- [x] [复线性空间情形的延拓](./hahn-banach-complex-case)
- [x] [赋范空间上有界线性泛函的保范延拓](./hahn-banach-norm-preserving)
- [x] [Hahn-Banach 定理的几何形式：凸集分离定理](./hahn-banach-geometric-separation)
- [x] [推论：保范泛函的存在性与点的分离](./norm-functional-existence-point-separation)
- [x] [推论：稠密性的泛函判别法](./density-functional-criterion)

### 第七章 共轭空间与弱收敛

- [x] [二次共轭空间与典范嵌入](./double-dual-canonical-embedding)
- [x] [自反空间及其例子](./reflexive-spaces-examples)
- [x] [弱收敛与弱* 收敛的定义](./weak-weakstar-convergence)
- [x] [弱收敛与强收敛的关系](./weak-strong-convergence-relation)
- [x] [具体空间中弱收敛的刻画](./weak-convergence-characterization)
- [x] [弱列紧性与 Alaoglu 定理](./weak-sequential-compactness-alaoglu)
- [x] [伴随算子与对偶算子](./adjoint-dual-operator)

### 第八章 紧算子

- [x] [紧算子的定义与基本性质](./compact-operator-basics)
- [x] [紧算子的例子：积分算子、有限秩算子](./compact-operator-examples)
- [x] [紧算子的极限与紧算子空间的闭性](./compact-operator-closure)
- [x] [紧算子的伴随算子](./compact-operator-adjoint)
- [x] [紧算子方程的可解性理论（Fredholm 二择一）](./fredholm-alternative)

### 第九章 谱理论初步

- [x] [线性算子的谱、正则点与预解式](./spectrum-resolvent-basics)
- [x] [谱的分类：点谱、连续谱与剩余谱](./spectrum-classification)
- [x] [有界线性算子谱的基本性质：非空性与紧性](./spectrum-basic-properties)
- [x] [谱半径公式（Gelfand 定理）](./spectral-radius-formula)
- [x] [紧算子的谱理论（Riesz-Schauder 理论）](./compact-operator-spectrum)
- [x] [自伴算子的谱](./self-adjoint-operator-spectrum)
- [x] [投影算子与自伴算子的谱分解初步](./spectral-decomposition-preliminary)

### 第十章 泛函分析的应用

- [x] [逼近论：最佳逼近元的存在性与唯一性](./best-approximation-existence-uniqueness)
- [x] [内积空间中的正交多项式与逼近](./orthogonal-polynomials-approximation)
- [x] [变分法初步：泛函的极值与欧拉-拉格朗日方程](./calculus-of-variations-euler-lagrange)
- [x] [变分原理与里茨方法](./variational-principles-ritz-method)
- [x] [微分方程边值问题的变分形式](./variational-form-bvp)
- [x] [量子力学的数学框架：Hilbert 空间与态](./quantum-mechanics-hilbert-space)
- [x] [量子力学中的算子：可观测量与自伴算子](./quantum-observables-self-adjoint)
- [x] [不确定性原理的算子表述](./uncertainty-principle-operator)
- [x] [积分方程的 Fredholm 理论应用](./fredholm-theory-integral-equations)

---

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [x] [赋范线性空间与巴拿赫空间（完备性、有限维特征）](./normed-linear-space-banach)
- [x] [有界线性算子（算子范数、有界线性泛函）](./bounded-linear-operators)
- [x] [三大基本定理（一致有界原理、开映射与闭图像定理）](./three-fundamental-theorems)
- [x] [Hahn-Banach 定理及其应用（延拓、分离）](./hahn-banach-theorem)
- [x] [对偶空间与弱拓扑（自反性、弱* 拓扑）](./dual-space-weak-topology)
- [x] [希尔伯特空间（正交投影、Riesz 表示定理）](./hilbert-space-orthogonal-projection)
- [x] [希尔伯特空间上的算子（伴随、自伴算子）](./operators-on-hilbert-space)
- [x] [紧算子与谱理论（Fredholm 二择一、谱分解）](./compact-operators-spectral-theory)
- [x] [无界算子初步（闭算子、对称与自伴）](./unbounded-operators)
- [x] [应用选讲（分布与 Sobolev 空间、量子力学数学基础）](./applications-distributions-quantum)

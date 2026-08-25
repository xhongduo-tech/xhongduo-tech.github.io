---
pageClass: plain-doc
---

# 线性代数

理解高维空间的工具，深度学习的几何语言。本分类对标同济大学《线性代数》全部章节，并以 Gilbert Strang《Introduction to Linear Algebra》补充四个基本子空间、最小二乘、SVD 与数据科学应用等专题。

## 主题规划

<ProgressGrid cat="intermediate/linear-algebra" />


### 第一篇 行列式

- [x] [二阶与三阶行列式](./determinants)
- [x] [全排列、逆序数与对换](./permutations-inversions)
- [x] [n 阶行列式的定义](./n-order-determinant)
- [x] [行列式的性质](./determinant-properties)
- [x] [行列式按行（列）展开：余子式与代数余子式](./cofactor-expansion)
- [x] [范德蒙德行列式](./vandermonde-determinant)
- [x] [克拉默法则](./cramers-rule)

### 第二篇 矩阵及其运算

- [x] [矩阵的概念与常见特殊矩阵](./matrix-concept)
- [x] [矩阵的线性运算与乘法](./matrix-operations)
- [x] [矩阵的转置与方阵的行列式](./matrix-transpose)
- [x] [逆矩阵的概念与性质](./inverse-matrix)
- [x] [伴随矩阵与求逆公式](./adjugate-matrix)
- [x] [矩阵的分块法](./block-matrix)

### 第三篇 矩阵的初等变换与线性方程组

- [x] [矩阵的初等变换与行阶梯形、行最简形](./elementary-transformations)
- [x] [初等矩阵及其与初等变换的关系](./elementary-matrices)
- [x] [用初等变换求逆矩阵](./inverse-via-elementary)
- [x] [矩阵的秩及其性质](./matrix-rank)
- [x] [线性方程组的解：高斯消元与解的判定定理](./gaussian-elimination)

### 第四篇 向量组的线性相关性

- [x] [n 维向量与向量组](./n-dimensional-vectors)
- [x] [向量组的线性组合与线性表示](./linear-combination)
- [x] [线性相关与线性无关的判定](./linear-independence)
- [x] [极大线性无关组与向量组的秩](./maximal-independent-group)
- [x] [向量空间、基与维数](./vector-space-basis-dimension)
- [x] [齐次线性方程组解的结构与基础解系](./homogeneous-solution-structure)
- [x] [非齐次线性方程组解的结构](./nonhomogeneous-solution-structure)

### 第五篇 相似矩阵及二次型

- [x] [向量的内积、长度与夹角](./inner-product-norm-angle)
- [x] [正交向量组与正交矩阵](./orthogonal-vectors-matrices)
- [x] [施密特（Gram-Schmidt）正交化](./gram-schmidt)
- [x] [方阵的特征值与特征向量](./eigenvalues-eigenvectors)
- [x] [相似矩阵与相似对角化](./similar-matrices-diagonalization)
- [x] [实对称矩阵的对角化](./symmetric-matrix-diagonalization)
- [x] [二次型及其矩阵表示](./quadratic-form)
- [x] [用正交变换化二次型为标准形](./orthogonal-transformation-standard-form)
- [x] [用配方法化二次型为标准形](./completing-square)
- [x] [惯性定理与规范形](./inertia-theorem)
- [x] [正定二次型与正定矩阵](./positive-definite)

### 第六篇 线性空间与线性变换

- [x] [线性空间的定义与基本性质](./linear-space)
- [x] [线性空间的维数、基与坐标](./dimension-basis-coordinate)
- [x] [基变换与坐标变换](./change-of-basis)
- [x] [线性子空间](./linear-subspace)
- [x] [线性变换及其运算](./linear-transformations)
- [x] [线性变换的矩阵表示](./matrix-representation)

### 第七篇 向量空间与四个基本子空间（Strang 专题）

- [x] [列空间与零空间](./column-null-space)
- [x] [秩-零化度定理](./rank-nullity-theorem)
- [x] [四个基本子空间及其维数关系](./four-fundamental-subspaces)
- [x] [子空间的正交补](./orthogonal-complement)
- [x] [线性方程组解的几何图景](./solution-geometry)

### 第八篇 正交性、投影与最小二乘

- [x] [正交投影与投影矩阵](./orthogonal-projection)
- [x] [正交基下的 Gram-Schmidt 再认识](./gram-schmidt-orthonormal)
- [x] [QR 分解及其应用](./qr-decomposition)
- [x] [最小二乘问题与正规方程](./least-squares)
- [x] [最小二乘的几何解释与直线拟合](./least-squares-geometry)

### 第九篇 矩阵分解与数值线性代数

- [x] [LU 分解：消元的矩阵形式](./lu-decomposition)
- [x] [Cholesky 分解与正定矩阵的判定](./cholesky-decomposition)
- [x] [谱定理：对称矩阵的特征分解](./spectral-theorem)
- [x] [Jordan 标准形简介](./jordan-form)
- [x] [矩阵范数与条件数](./matrix-norm-condition-number)
- [x] [线性方程组的迭代解法与收敛性](./iterative-methods)

### 第十篇 奇异值分解

- [x] [奇异值分解的定义与几何意义](./svd-definition)
- [x] [SVD 与四个基本子空间](./svd-subspaces)
- [x] [伪逆（Moore-Penrose 逆）及其性质](./pseudo-inverse)
- [x] [截断 SVD 与最佳低秩近似](./truncated-svd)
- [x] [用 SVD 求解最小二乘问题](./svd-least-squares)

### 第十一篇 线性代数在数据科学中的应用

- [x] [数据矩阵、均值与协方差矩阵](./data-matrix-covariance)
- [x] [主成分分析（PCA）](./pca)
- [x] [最小二乘回归与正规方程实践](./least-squares-regression)
- [x] [图像压缩中的 SVD 低秩近似](./image-compression-svd)
- [x] [推荐系统中的矩阵分解](./matrix-factorization-recommender)
- [x] [马尔可夫链与稳态向量](./markov-chain-steady-state)
- [x] [PageRank 算法与特征向量](./pagerank)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [x] [行列式（定义、性质、克拉默法则）](./draft-cbbb0604a7)
- [x] [矩阵及其运算（秩、逆矩阵、分块矩阵）](./draft-b20b37c9ad62.md)
- [x] [线性方程组（高斯消元、解的结构）](./intermediate-linear-algebra-c09dd1b4.md)
- [x] [向量空间与线性相关性（基、维数、坐标）](./intermediate-linear-algebra-9bf892eb.md)
- [x] [线性映射与矩阵表示（核与像、相似）](./intermediate-linear-algebra-a339f5a5.md)
- [x] [特征值与特征向量（对角化、Cayley-Hamilton）](./intermediate-linear-algebra-cayley-hamilton-13e6fd88.md)
- [x] [二次型（标准形、正定性）](./intermediate-linear-algebra-74c5e086.md)
- [x] [内积空间与正交性（Gram-Schmidt、正交投影）](./intermediate-linear-algebra-gram-schmidt-bcb857b5.md)
- [x] [标准形理论（若尔当标准形、谱定理）](./intermediate-linear-algebra-a2ffb923.md)
- [x] [应用与计算（SVD、最小二乘、数值线性代数初步）](./intermediate-linear-algebra-svd-2d4006aa.md)

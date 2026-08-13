---
pageClass: plain-doc
---

# 近似算法

近似算法研究在 NP 困难组合优化问题上，如何在多项式时间内求出接近最优的解，并刻画问题的不可近似性边界。它是算法设计、线性规划与计算复杂性理论的交汇点，也是大规模实际优化问题（调度、网络设计、选址、割与度量等）的基石。

## 对标教材

- Vijay V. Vazirani, "Approximation Algorithms" (Springer, 2001)
- David P. Williamson & David B. Shmoys, "The Design of Approximation Algorithms" (Cambridge University Press, 2011)

## 主题规划

<ProgressGrid cat="cs/approximation-algorithms" />

### 第1篇

- [x] [近似算法导论：近似比与问题建模](./introduction)
- [x] [贪心算法与局部搜索](./greedy-local-search)
- [x] [数据舍入与动态规划](./rounding-dynamic-programming)
- [x] [线性规划的确定性舍入](./deterministic-rounding-lp)
- [x] [LP 对偶、对偶拟合与集合覆盖](./lp-duality-set-cover)
- [x] [背包与装箱问题](./knapsack-bin-packing)
- [x] [最小完工时间调度与无关并行机](./makespan-scheduling)

### 第2篇

- [x] [原始对偶方法](./primal-dual-method)
- [x] [集合覆盖的原始对偶 Schema 与局部比率法](./primal-dual-set-cover-local-ratio)
- [x] [Steiner 树与旅行商问题](./steiner-tree-tsp)
- [x] [Steiner 森林与 Steiner 网络设计](./steiner-forest-network)
- [x] [设施选址问题](./facility-location)
- [x] [k-中位数问题](./k-median)
- [x] [多向割与多割问题](./multicut-multiway-cut)
- [x] [最稀疏割与谱方法](./sparsest-cut-spectral)

### 第3篇

- [x] [随机舍入](./randomized-rounding)
- [x] [MAX-SAT 的随机化舍入与随机采样](./max-sat-randomized)
- [x] [半定规划松弛](./semidefinite-programming)
- [x] [Goemans–Williamson 最大割算法](./goemans-williamson)
- [x] [割与度量（Cuts and Metrics）](./cuts-metrics)
- [x] [度量的进一步应用与嵌入技术](./metric-embedding)

### 第4篇

- [x] [近似保持归约与 APX 复杂性类](./approximation-preserving-reduction-apx)
- [x] [PCP 定理与 Max-3SAT 的不可近似性](./pcp-theorem-max-3sat)
- [x] [标签覆盖问题与硬度归约](./label-cover-hardness)
- [x] [不可近似性证明技术](./inapproximability-techniques)
- [x] [开放问题与前沿方向](./open-problems)

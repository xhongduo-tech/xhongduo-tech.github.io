---
pageClass: plain-doc
---

# 随机算法与概率方法

以随机性为算法设计工具，通过 Las Vegas/Monte Carlo 两类算法与概率方法解决确定性算法难以高效处理的问题，是现代算法理论的重要主干。掌握集中不等式与概率论证，能在随机性中把握确定性的边界，为高级算法、计算复杂性、密码学与机器学习奠定数学基础。

## 对标教材

- Rajeev Motwani & Prabhakar Raghavan, "Randomized Algorithms" (Cambridge University Press)
- Michael Mitzenmacher & Eli Upfal, "Probability and Computing: Randomization and Probabilistic Techniques in Algorithms and Data Analysis" (Cambridge University Press, 2nd ed.)
- Noga Alon & Joel H. Spencer, "The Probabilistic Method" (Wiley, 4th ed.)

## 主题规划

<ProgressGrid cat="cs/randomized-and-probabilistic-algorithms" />

### 第1篇

- [x] [随机变量与概率空间基础](./random-variables-and-probability-spaces)
- [x] [期望、方差与矩](./expectation-variance-moments)
- [x] [马尔可夫不等式与切比雪夫不等式](./markov-chebyshev-inequalities)
- [x] [切尔诺夫界](./chernoff-bounds)
- [x] [霍夫丁不等式与尾概率](./hoeffding-inequality-tails)

### 第2篇

- [x] [Las Vegas 与 Monte Carlo 算法](./las-vegas-monte-carlo)
- [x] [随机化快速排序与期望时间](./randomized-quicksort)
- [x] [随机化选择与中位数查找](./randomized-selection-median)
- [x] [随机主方法与分治](./randomized-master-method)
- [x] [随机化平摊分析](./randomized-amortized-analysis)

### 第3篇

- [x] [随机二叉搜索树与 Treap](./random-bst-treap)
- [x] [跳表与随机平衡](./skip-lists)
- [x] [生日悖论与哈希碰撞](./birthday-paradox-hash-collisions)
- [x] [通用哈希与两两独立](./universal-hashing-pairwise-independent)
- [x] [掷球入箱与负载均衡](./balls-and-bins-load-balancing)
- [x] [布隆过滤器与近似成员查询](./bloom-filters)

### 第4篇

- [x] [期望方法](./expectation-method)
- [x] [线性期望与图着色](./linear-expectation-graph-coloring)
- [x] [变化与构造方法](./alteration-construction-method)
- [x] [第二矩方法与随机图](./second-moment-method-random-graphs)
- [x] [洛瓦斯局部引理 LLL](./lovasz-local-lemma)
- [x] [大偏差与六标准差方法](./large-deviations-six-deviation)

### 第5篇

- [x] [马尔可夫链与平稳分布](./markov-chains-stationary-distribution)
- [x] [马尔可夫链的收敛与混合时间](./markov-chain-mixing-time)
- [x] [随机游走与图性质](./random-walks-graphs)
- [x] [马尔可夫链蒙特卡洛 MCMC](./mcmc)
- [x] [鞅与尾界](./martingales-tail-bounds)
- [x] [去随机化与条件期望方法](./derandomization-conditional-expectation)
- [x] [随机复杂性类与伪随机发生器](./randomized-complexity-prg)

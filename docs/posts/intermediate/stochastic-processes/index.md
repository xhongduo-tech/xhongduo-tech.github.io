---
pageClass: plain-doc
---

# 随机过程

随机过程研究随时间演化的随机现象，是概率论向动态系统的延伸。本分类对标张波《应用随机过程》与 Ross《Stochastic Processes》的章节体系，从泊松过程、马尔可夫链一路写到鞅、布朗运动与随机积分。

## 主题规划

<ProgressGrid cat="intermediate/stochastic-processes" />


### 第一篇 预备知识

- [x] [随机过程的基本概念：样本轨道与有限维分布族](./basic-concepts)
- [x] [随机过程的分类：离散/连续时间与离散/连续状态](./process-classification)
- [x] [条件概率与条件期望的回顾](./conditional-expectation-review)
- [x] [条件期望的严格定义：关于 σ-代数的期望](./conditional-expectation-sigma-algebra)
- [x] [条件期望的性质：塔性质、取己知量、独立性](./conditional-expectation-properties)
- [x] [全期望公式与条件方差公式](./total-expectation-conditional-variance)
- [x] [矩母函数（Moment Generating Function）与特征函数](./mgf-characteristic-function)
- [x] [常用分布的矩母函数与独立和的卷积](./mgf-convolution)
- [x] [收敛性概念：几乎必然收敛、依概率收敛、依分布收敛、均方收敛](./convergence-concepts)

### 第二篇 泊松过程

- [x] [计数过程与独立平稳增量](./counting-process-independent-increments)
- [x] [泊松过程的三种等价定义](./poisson-process-equivalent-definitions)
- [x] [到达间隔的指数分布与无记忆性](./poisson-arrival-intervals-memoryless)
- [x] [到达时刻的条件分布：均匀分布的顺序统计量](./poisson-arrival-times-order-statistics)
- [x] [泊松过程的叠加与分解（稀疏定理）](./poisson-superposition-decomposition)
- [x] [非齐次泊松过程：强度函数与累积强度](./nonhomogeneous-poisson-process)
- [x] [复合泊松过程：定义、均值与方差](./compound-poisson-process)
- [x] [条件泊松过程与混合泊松模型](./conditional-poisson-process)
- [x] [泊松过程的模拟与参数估计](./poisson-simulation-estimation)

### 第三篇 更新过程

- [x] [更新过程的定义：间隔独立同分布的计数过程](./renewal-process-definition)
- [x] [N(t) 的分布与更新函数 m(t)](./renewal-function-mt)
- [x] [更新方程与更新函数的渐近性质](./renewal-equation-asymptotics)
- [x] [初等更新定理（Elementary Renewal Theorem）](./elementary-renewal-theorem)
- [x] [关键更新定理（Key Renewal Theorem）与 Blackwell 定理](./key-renewal-theorem-blackwell)
- [x] [更新报酬（Renewal Reward）过程与长期平均成本](./renewal-reward-process)
- [x] [年龄与剩余寿命：平衡更新过程](./age-residual-life-balanced-renewal)
- [x] [交替更新过程及其在可靠性中的应用](./alternating-renewal-process-reliability)
- [x] [延迟更新过程与再生过程](./delayed-renewal-regenerative)

### 第四篇 离散时间马尔可夫链

- [x] [马尔可夫性与马尔可夫链的定义](./markov-chain-definition)
- [x] [一步转移概率矩阵与 Chapman-Kolmogorov 方程](./transition-matrix-chapman-kolmogorov)
- [x] [n 步转移概率的计算与矩阵幂](./n-step-transition-matrix-power)
- [x] [状态分类：互通、闭集与不可约性](./state-classification-communication-closed-irreducible)
- [x] [周期性：周期状态的判定与等价类](./periodicity)
- [x] [常返与暂留：首次到达概率与期望回访次数](./recurrence-transience)
- [x] [常返性的判别：常返等价于 ∑pⁿᵢᵢ 发散](./recurrence-criterion-summation)
- [x] [正常返与零常返：平均回访时间](./positive-null-recurrence)
- [x] [首达概率与首达时间的计算方法](./hitting-times-probabilities)
- [x] [不变分布（平稳分布）的存在性与唯一性](./invariant-distribution-existence-uniqueness)
- [x] [极限定理：不可约非周期正常返链的遍历定理](./limit-theorem-ergodic)
- [x] [可逆马尔可夫链与细致平衡条件](./reversible-markov-chain-detailed-balance)
- [x] [分支过程：灭绝概率与矩的计算](./branching-process)
- [x] [马尔可夫链的应用：随机游动与 PageRank](./random-walk-pagerank-applications)

### 第五篇 连续时间马尔可夫链

- [x] [连续时间马尔可夫链的定义与转移概率函数](./continuous-time-markov-chain-definition)
- [x] [停留时间的指数分布与嵌入链](./sojourn-time-exponential-embedded-chain)
- [x] [转移速率矩阵（Q 矩阵 / 无穷小生成元）](./q-matrix-infinitesimal-generator)
- [x] [Kolmogorov 向后方程与向前方程](./kolmogorov-forward-backward-equations)
- [x] [平稳分布与长期行为：连续时间情形](./stationary-distribution-ctmc)
- [x] [纯生过程与 Yule 过程](./pure-birth-yule-process)
- [x] [生灭过程：定义与平稳分布](./birth-death-process)
- [x] [排队论初步：M/M/1 排队模型](./mm1-queue)
- [x] [M/M/s 与 M/M/∞ 排队系统](./mms-queue)
- [x] [Little 公式与排队系统的性能指标](./little-formula)

### 第六篇 鞅

- [x] [鞅的定义：离散时间鞅与例子](./martingale-definition)
- [x] [上鞅、下鞅与鞅的等价刻画](./submartingale-supermartingale)
- [x] [停时（Stopping Time）的定义与性质](./stopping-time)
- [x] [可选停时定理（Optional Stopping Theorem）](./optional-stopping-theorem)
- [x] [停时定理的应用：赌徒破产问题](./gamblers-ruin-optional-stopping)
- [x] [Wald 等式与随机游动的首达时刻](./wald-equation-first-passage)
- [x] [鞅收敛定理（Martingale Convergence Theorem）](./martingale-convergence-theorem)
- [x] [Doob 不等式与极大不等式](./doob-inequality-maximal-inequality)
- [x] [鞅差序列与 Azuma 不等式](./martingale-difference-azuma)
- [x] [鞅在算法分析中的应用](./martingales-in-algorithm-analysis)

### 第七篇 布朗运动

- [x] [布朗运动的历史背景：从花粉运动到维纳过程](./brownian-motion-history)
- [x] [布朗运动的定义：独立平稳正态增量](./brownian-motion-definition)
- [x] [布朗运动轨道的连续性](./brownian-motion-continuity)
- [x] [布朗运动轨道的不可微性与二次变差](./brownian-motion-nondifferentiability-quadratic-variation)
- [x] [布朗运动的马尔可夫性与鞅性](./brownian-motion-markov-martingale)
- [x] [首中时的分布与反射原理（Reflection Principle）](./hitting-time-reflection-principle)
- [x] [最大值的分布与反正弦律](./maximum-distribution-arcsine-law)
- [x] [布朗桥与高斯过程视角](./brownian-bridge-gaussian-process)
- [x] [布朗运动的变体：带漂移布朗运动与几何布朗运动](./drifted-brownian-geometric-brownian)

### 第八篇 随机积分初步

- [x] [为什么黎曼-斯蒂尔切斯积分不够用](./riemann-stieltjes-inadequacy)
- [x] [伊藤积分（Itô Integral）的构造：从简单过程开始](./ito-integral-construction)
- [x] [伊藤积分的性质：等距性与鞅性](./ito-integral-properties)
- [x] [伊藤公式（Itô's Lemma）：单变量形式](./itos-lemma)
- [x] [多维伊藤公式与乘积法则](./multidimensional-ito-product-rule)
- [x] [随机微分方程（SDE）的解与存在唯一性](./sde-existence-uniqueness)
- [x] [常见 SDE 的求解：几何布朗运动与 OU 过程](./solving-sdes-gbm-ou)

### 第九篇 平稳过程

- [x] [严平稳与宽平稳（弱平稳）的定义](./strict-weak-stationarity)
- [x] [自相关函数（ACF）的性质与估计](./acf-properties-estimation)
- [x] [遍历性（Ergodicity）：时间平均与集合平均](./ergodicity)
- [x] [功率谱密度与 Wiener-Khinchin 定理](./power-spectral-density-wiener-khinchin)
- [x] [平稳过程的线性变换与滤波](./linear-transform-filtering)
- [x] [白噪声、滑动平均与自回归过程](./white-noise-ma-ar)
- [x] [平稳过程的谱分解初步](./spectral-decomposition)

### 第十篇 应用

- [x] [金融中的随机模型：从布朗运动到 Black-Scholes](./black-scholes-financial-models)
- [x] [风险中性定价与鞅测度初步](./risk-neutral-pricing-martingale-measure)
- [x] [随机利率模型：Vasicek 与 CIR 模型](./vasicek-cir-interest-rate)
- [x] [保险中的随机过程：Cramér-Lundberg 破产模型](./cramer-lundberg-ruin-theory)
- [x] [库存管理与排队网络中的随机模型](./inventory-queueing-networks)
- [x] [马尔可夫决策过程（MDP）：状态、动作与回报](./mdp)
- [x] [强化学习的随机过程视角：贝尔曼方程与马尔可夫性](./rl-bellman-markov)
- [x] [MCMC：用马尔可夫链做采样的基本原理](./mcmc)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

### 第1篇

- [ ] 预备知识（条件期望、矩母函数）
- [ ] 泊松过程（定义、复合与非齐次推广）
- [ ] 更新理论（更新方程、极限定理）
- [ ] 离散时间马尔可夫链（转移矩阵、状态分类）
- [ ] 马尔可夫链极限理论（平稳分布、遍历性）
- [ ] 连续时间马尔可夫链（生灭过程、Kolmogorov 方程）
- [ ] 鞅（停时定理、鞅收敛）
- [ ] 布朗运动（性质、首达时）
- [ ] 随机积分与扩散初步（伊藤公式简介）
- [ ] 应用选讲（排队论、金融模型、MCMC）

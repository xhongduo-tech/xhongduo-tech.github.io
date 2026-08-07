---
pageClass: plain-doc
---

# 机器学习

学完机器学习 = 写完周志华《机器学习》（西瓜书）全部 16 章对应的博文，外加 GBDT/XGBoost/LightGBM、特征工程、调参与 AutoML 三个工程专题。

## 主题规划

<ProgressGrid cat="advanced/machine-learning" />


### 第1章 绪论
- [x] [基本术语](./basic-terminology)
- [x] [假设空间](./hypothesis-space)
- [x] [归纳偏好](./inductive-bias)

### 第2章 模型评估与选择
- [x] [经验误差与过拟合](./empirical-error-overfitting)
- [x] [评估方法（留出法、交叉验证法、自助法）](./evaluation-methods)
- [x] [性能度量（错误率与精度、查准率与查全率、ROC 与 AUC、代价敏感错误率）](./performance-measures)
- [x] [比较检验（假设检验、交叉验证 t 检验、McNemar 检验、Friedman 检验与 Nemenyi 后续检验）](./comparative-testing)
- [x] [偏差与方差](./bias-variance)

### 第3章 线性模型
- [x] [基本形式](./linear-model-basics)
- [x] [线性回归](./linear-regression)
- [x] [对数几率回归](./logistic-regression)
- [x] [线性判别分析](./linear-discriminant-analysis)
- [x] [多分类学习](./multi-class-learning)
- [x] [类别不平衡问题](./class-imbalance)

### 第4章 决策树
- [x] [基本流程](./decision-tree-basics)
- [x] [划分选择（信息增益、增益率、基尼指数）](./decision-tree-splitting)
- [x] [剪枝处理（预剪枝与后剪枝）](./decision-tree-pruning)
- [x] [连续值与缺失值处理](./decision-tree-continuous-missing)
- [x] [多变量决策树](./multivariate-decision-tree)

### 第5章 神经网络
- [x] [神经元模型](./neuron-model)
- [x] [感知机与多层网络](./perceptron-multi-layer)
- [x] [误差逆传播算法（BP）](./backpropagation)
- [x] [全局最小与局部极小](./global-minima-local-minima)
- [x] [其他常见神经网络（RBF 网络、ART 网络、SOM 网络、级联相关网络、Elman 网络、Boltzmann 机）](./other-neural-networks)
- [x] [深度学习](./deep-learning)

### 第6章 支持向量机
- [x] [间隔与支持向量](./svm-margin-support-vector)
- [x] [对偶问题](./svm-dual-problem)
- [x] [核函数](./svm-kernel-function)
- [x] [软间隔与正则化](./svm-soft-margin)
- [x] [支持向量回归](./svr)
- [x] [核方法](./kernel-methods)

### 第7章 贝叶斯分类器
- [x] [贝叶斯决策论](./bayesian-decision-theory)
- [x] [极大似然估计](./maximum-likelihood-estimation)
- [x] [朴素贝叶斯分类器](./naive-bayes)
- [x] [半朴素贝叶斯分类器](./semi-naive-bayes)
- [x] [贝叶斯网](./bayesian-network)
- [x] [EM 算法](./em-algorithm)

### 第8章 集成学习
- [x] [个体与集成](./ensemble-individuals)
- [x] [Boosting](./boosting)
- [x] [Bagging 与随机森林](./bagging-random-forest)
- [x] [结合策略（平均法、投票法、学习法）](./combining-strategies)
- [x] [多样性（误差-分歧分解、多样性度量、多样性增强）](./ensemble-diversity)

### 第9章 聚类
- [x] [聚类任务](./clustering-task)
- [x] [性能度量](./clustering-performance-measures)
- [x] [距离计算](./distance-calculation)
- [x] [原型聚类（k 均值算法、学习向量量化、高斯混合聚类）](./prototype-clustering)
- [x] [密度聚类（DBSCAN）](./density-clustering)
- [x] [层次聚类（AGNES）](./hierarchical-clustering)

### 第10章 降维与度量学习
- [x] [k 近邻学习](./k-nearest-neighbors)
- [x] [低维嵌入（多维缩放 MDS）](./multidimensional-scaling)
- [x] [主成分分析（PCA）](./pca)
- [x] [核化线性降维（KPCA）](./kpca)
- [x] [流形学习（等度量映射 Isomap、局部线性嵌入 LLE）](./manifold-learning)
- [x] [度量学习](./metric-learning)

### 第11章 特征选择与稀疏学习
- [x] [子集搜索与评价](./feature-subset-selection)
- [x] [过滤式选择（Relief）](./filter-relief)
- [x] [包裹式选择（LVW）](./wrapper-lvw)
- [x] [嵌入式选择与 L1 正则化](./embedded-l1)
- [x] [稀疏表示与字典学习](./sparse-representation-dictionary)
- [x] [压缩感知](./compressed-sensing)

### 第12章 计算学习理论
- [x] [基础知识](./computational-learning-basics)
- [x] [PAC 学习](./pac-learning)
- [x] [有限假设空间](./finite-hypothesis-space)
- [x] [VC 维](./vc-dimension)
- [x] [Rademacher 复杂度](./rademacher-complexity)
- [x] [稳定性](./stability)

### 第13章 半监督学习
- [x] [未标记样本](./unlabeled-data)
- [x] [生成式方法](./generative-semi-supervised)
- [x] [半监督 SVM](./semi-supervised-svm)
- [x] [图半监督学习](./graph-semi-supervised)
- [x] [基于分歧的方法](./disagreement-based)
- [x] [半监督聚类](./semi-supervised-clustering)

### 第14章 概率图模型
- [x] [隐马尔可夫模型](./hidden-markov-model)
- [x] [马尔可夫随机场](./markov-random-field)
- [x] [条件随机场](./conditional-random-field)
- [x] [学习与推断](./pgm-learning-inference)
- [x] [近似推断（MCMC、变分推断）](./approximate-inference)
- [x] [话题模型（隐狄利克雷分配 LDA）](./latent-dirichlet-allocation)

### 第15章 规则学习
- [x] [基本概念](./rule-learning-basics)
- [x] [序贯覆盖](./sequential-covering)
- [x] [剪枝优化](./rule-pruning)
- [x] [一阶规则学习（FOIL）](./foil)
- [x] [归纳逻辑程序设计](./inductive-logic-programming)

### 第16章 强化学习
- [x] [任务与奖赏](./rl-task-reward)
- [x] [K 摇臂赌博机](./k-armed-bandit)
- [x] [有模型学习](./model-based-rl)
- [x] [免模型学习](./model-free-rl)
- [x] [值函数近似](./value-function-approximation)
- [x] [模仿学习](./imitation-learning)

### 工程专题一 GBDT / XGBoost / LightGBM
- [x] [GBDT 原理与实现](./gbdt)
- [x] [XGBoost 原理与工程优化](./xgboost)
- [x] [LightGBM 原理与工程优化](./lightgbm)
- [x] [三大框架对比与实战选型](./gbdt-xgboost-lightgbm-comparison)

### 工程专题二 特征工程
- [x] [数据清洗与预处理](./data-cleaning)
- [x] [特征变换与编码](./feature-transform-encoding)
- [x] [特征构建与特征交叉](./feature-construction-crossing)
- [x] [特征工程实战案例](./feature-engineering-practice)

### 工程专题三 调参与 AutoML
- [x] [超参数搜索策略（网格搜索、随机搜索）](./hyperparameter-search)
- [x] [贝叶斯优化](./bayesian-optimization)
- [x] [AutoML 框架（Auto-sklearn、AutoGluon 等）](./automl-frameworks)
- [x] [神经架构搜索（NAS）简介](./neural-architecture-search)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

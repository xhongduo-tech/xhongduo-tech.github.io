---
pageClass: plain-doc
---

# Machine Learning

Finishing this course means writing a blog post for each of the 16 chapters of Zhou Zhihua's *Machine Learning* (the "Watermelon Book"), plus three engineering topics: GBDT/XGBoost/LightGBM, feature engineering, and hyperparameter tuning with AutoML.

## Topic Overview

<ProgressGrid cat="advanced/machine-learning" />


### Chapter 1 · Introduction
- [ ] Basic terminology
- [ ] Hypothesis space
- [ ] Inductive bias

### Chapter 2 · Model Evaluation and Selection
- [ ] Empirical error and overfitting
- [ ] Evaluation methods (hold-out, cross-validation, bootstrap)
- [ ] Performance measures (error rate and accuracy, precision and recall, ROC and AUC, cost-sensitive error rate)
- [ ] Comparative tests (hypothesis testing, paired t-test over cross-validation folds, McNemar's test, Friedman test and Nemenyi post-hoc test)
- [ ] Bias and variance

### Chapter 3 · Linear Models
- [ ] Basic form
- [ ] Linear regression
- [ ] Logistic regression
- [ ] Linear discriminant analysis
- [ ] Multi-class learning
- [ ] Class imbalance

### Chapter 4 · Decision Trees
- [ ] Basic process
- [ ] Split selection (information gain, gain ratio, Gini index)
- [ ] Pruning (pre-pruning and post-pruning)
- [ ] Handling continuous values and missing values
- [ ] Multivariate decision trees

### Chapter 5 · Neural Networks
- [ ] Neuron model
- [ ] Perceptron and multi-layer networks
- [ ] Backpropagation algorithm (BP)
- [ ] Global minimum and local minima
- [ ] Other common neural networks (RBF networks, ART networks, SOM networks, cascade-correlation networks, Elman networks, Boltzmann machines)
- [ ] Deep learning

### Chapter 6 · Support Vector Machines
- [ ] Margins and support vectors
- [ ] Dual problem
- [ ] Kernel functions
- [ ] Soft margins and regularization
- [ ] Support vector regression
- [ ] Kernel methods

### Chapter 7 · Bayesian Classifiers
- [ ] Bayesian decision theory
- [ ] Maximum likelihood estimation
- [ ] Naive Bayes classifiers
- [ ] Semi-naive Bayes classifiers
- [ ] Bayesian networks
- [ ] EM algorithm

### Chapter 8 · Ensemble Learning
- [ ] Individuals and ensembles
- [ ] Boosting
- [ ] Bagging and random forests
- [ ] Combining strategies (averaging, voting, stacking)
- [ ] Diversity (error-ambiguity decomposition, diversity measures, diversity enhancement)

### Chapter 9 · Clustering
- [ ] The clustering task
- [ ] Performance measures
- [ ] Distance computation
- [ ] Prototype clustering (k-means, learning vector quantization, Gaussian mixture clustering)
- [ ] Density clustering (DBSCAN)
- [ ] Hierarchical clustering (AGNES)

### Chapter 10 · Dimensionality Reduction and Metric Learning
- [ ] k-Nearest Neighbor learning
- [ ] Low-dimensional embedding (multi-dimensional scaling, MDS)
- [ ] Principal component analysis (PCA)
- [ ] Kernelized linear dimensionality reduction (KPCA)
- [ ] Manifold learning (Isomap, locally linear embedding, LLE)
- [ ] Metric learning

### Chapter 11 · Feature Selection and Sparse Learning
- [ ] Subset search and evaluation
- [ ] Filter-based selection (Relief)
- [ ] Wrapper-based selection (LVW)
- [ ] Embedded selection and L1 regularization
- [ ] Sparse representation and dictionary learning
- [ ] Compressed sensing

### Chapter 12 · Computational Learning Theory
- [ ] Fundamentals
- [ ] PAC learning
- [ ] Finite hypothesis spaces
- [ ] VC dimension
- [ ] Rademacher complexity
- [ ] Stability

### Chapter 13 · Semi-Supervised Learning
- [ ] Unlabeled samples
- [ ] Generative methods
- [ ] Semi-supervised SVM
- [ ] Graph-based semi-supervised learning
- [ ] Disagreement-based methods
- [ ] Semi-supervised clustering

### Chapter 14 · Probabilistic Graphical Models
- [ ] Hidden Markov models
- [ ] Markov random fields
- [ ] Conditional random fields
- [ ] Learning and inference
- [ ] Approximate inference (MCMC, variational inference)
- [ ] Topic models (latent Dirichlet allocation, LDA)

### Chapter 15 · Rule Learning
- [ ] Basic concepts
- [ ] Sequential covering
- [ ] Pruning and optimization
- [ ] First-order rule learning (FOIL)
- [ ] Inductive logic programming

### Chapter 16 · Reinforcement Learning
- [ ] Tasks and rewards
- [ ] K-armed bandits
- [ ] Model-based learning
- [ ] Model-free learning
- [ ] Value function approximation
- [ ] Imitation learning

### Engineering Topic 1 · GBDT / XGBoost / LightGBM
- [ ] GBDT principles and implementation
- [ ] XGBoost principles and engineering optimizations
- [ ] LightGBM principles and engineering optimizations
- [ ] Comparison and practical selection of the three frameworks

### Engineering Topic 2 · Feature Engineering
- [ ] Data cleaning and preprocessing
- [ ] Feature transformation and encoding
- [ ] Feature construction and feature crossing
- [ ] Feature engineering in practice

### Engineering Topic 3 · Hyperparameter Tuning and AutoML
- [ ] Hyperparameter search strategies (grid search, random search)
- [ ] Bayesian optimization
- [ ] AutoML frameworks (Auto-sklearn, AutoGluon, etc.)
- [ ] Introduction to neural architecture search (NAS)

> When a post is finished: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.

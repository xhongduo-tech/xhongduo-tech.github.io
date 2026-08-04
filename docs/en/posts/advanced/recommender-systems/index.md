---
pageClass: plain-doc
---

# Recommender Systems

Built around the book *Recommender Systems Practice* (Xiang Liang) as its backbone, and drawing on classic papers and engineering practice from the modern industrial recommendation ecosystem, this series progresses from fundamental algorithms all the way to large-scale online services and LLM-based recommendation.

## Topic Plan

<ProgressGrid cat="advanced/recommender-systems" />


### Part 1 · Recommender Systems Overview and Evaluation

- [ ] What is a recommender system: information overload and the long-tail problem
- [ ] Data sources for recommender systems: explicit feedback and implicit feedback
- [ ] Experimental methods: offline experiments, user surveys, and online experiments
- [ ] Offline evaluation metrics: precision, recall, RMSE, and MAE
- [ ] Ranking metrics: MAP, NDCG, and AUC
- [ ] Beyond accuracy: diversity, novelty, serendipity, and coverage
- [ ] A/B testing: traffic split, layered experiments, and metric significance

### Part 2 · Collaborative Filtering

- [ ] Basic ideas of collaborative filtering and the neighborhood-method framework
- [ ] Similarity computation: cosine similarity, Pearson correlation coefficient, and Jaccard
- [ ] User-based collaborative filtering (UserCF): principles and implementation
- [ ] Improving UserCF similarity: penalizing popular items (IUF idea)
- [ ] Item-based collaborative filtering (ItemCF): principles and implementation
- [ ] ItemCF normalization and active-user penalization
- [ ] Comparing UserCF and ItemCF, and their applicable scenarios
- [ ] Pros and cons of collaborative filtering: sparsity, cold start, and interpretability

### Part 3 · Latent Factor Models and Matrix Factorization

- [ ] Basic idea of latent factor models (LFM): latent factors and interest categorization
- [ ] SVD-based matrix factorization: Funk-SVD and gradient-descent solving
- [ ] Matrix factorization with biases (BiasSVD)
- [ ] Matrix factorization for implicit feedback: weighted alternating least squares (ALS-WR)
- [ ] SVD++: fusing explicit and implicit feedback
- [ ] Limitations of matrix factorization: weak generalization and underutilized features

### Part 4 · Content-Based Recommendation

- [ ] Basic framework of content-based recommendation: item profiles and user profiles
- [ ] Representing text content: TF-IDF and the bag-of-words model
- [ ] From TF-IDF to word vectors: Word2Vec and item embeddings
- [ ] Modeling user interests with content features and similarity matching
- [ ] Hybrid strategies combining content-based recommendation and collaborative filtering

### Part 5 · Recall Strategies

- [ ] Overall recommender architecture: recall, coarse ranking, fine ranking, and reranking
- [ ] Design principles of multi-channel recall: complementarity and fusion strategies
- [ ] Vector recall: structure and training objective of the two-tower model
- [ ] Negative sampling in two-tower recall: in-batch negative sampling and hard negatives
- [ ] Multi-interest recall: MIND and ComiRec
- [ ] Graph recall: graph embeddings (DeepWalk, Node2Vec, EGES) and PinSage
- [ ] Evaluating the recall stage: recall rate, hit rate, and pipeline-consistency analysis

### Part 6 · The Evolution of Ranking Models

- [ ] Formalizing the ranking problem: CTR prediction and pointwise modeling
- [ ] Logistic regression (LR): interpretability and the art of feature engineering
- [ ] GBDT+LR: automating feature crosses (the Facebook approach)
- [ ] FM (factorization machines): second-order feature interactions on sparse data
- [ ] FFM: introducing field-aware factorization
- [ ] Wide & Deep: combining memorization and generalization
- [ ] DeepFM: replacing the manual features on the Wide side with FM
- [ ] DCN and xDeepFM: explicit high-order feature crossing
- [ ] DIN: attention mechanisms for modeling user interests
- [ ] DIEN: interest evolution networks and sequence modeling

### Part 7 · Sequential Recommendation

- [ ] Defining the sequential recommendation problem: from static interests to dynamic behavior sequences
- [ ] Markov chains and FPMC: first-order sequential models fused with matrix factorization
- [ ] GRU4Rec: RNN-based session recommendation
- [ ] SASRec: self-attention-based sequential recommendation
- [ ] BERT4Rec: bidirectional sequence modeling and cloze-style training
- [ ] Long-sequence modeling: two-stage interest retrieval in SIM

### Part 8 · Multi-Objective and Multi-Task Learning

- [ ] The multi-objective ranking problem: balancing clicks, dwell time, likes, and consumption
- [ ] Multi-objective fusion: weighted summation, multiplicative formulas, and evolutionary learning
- [ ] MMOE: multi-gate mixture-of-experts models
- [ ] PLE: progressive layered extraction and mitigating task conflicts
- [ ] ESMM: full-space modeling to address sample selection bias in CVR prediction

### Part 9 · Cold Start

- [ ] Classifying cold-start problems: user cold start, item cold start, and system cold start
- [ ] Leveraging user registration information and demographic features
- [ ] New-user cold start: choosing suitable items to collect feedback
- [ ] New-item cold start: exploiting content features and item attributes
- [ ] Meta-learning for cold start: the MeLU and MAML approaches

### Part 10 · Exploration and Exploitation

- [ ] The exploration vs. exploitation problem
- [ ] The multi-armed bandit problem and regret analysis
- [ ] ε-Greedy and naive exploration strategies
- [ ] UCB: the upper confidence bound algorithm
- [ ] Thompson Sampling: exploration from a Bayesian perspective
- [ ] LinUCB and contextual bandits

### Part 11 · Reranking and Diversity

- [ ] The role of the reranking stage: from pointwise optimal to listwise optimal
- [ ] Defining and measuring diversity: ILAD, ILD, and category interleaving
- [ ] MMR: maximum marginal relevance reranking
- [ ] DPP: diversity reranking based on determinantal point processes
- [ ] Sequence-aware reranking models: PRM and generative reranking

### Part 12 · Recommender System Engineering

- [ ] Industrial recommender architecture overview: data flow and serving pipeline
- [ ] Feature platforms: offline features, online features, and feature consistency
- [ ] Sample construction: label delay, sample backfill, and attribution windows
- [ ] Model training frameworks: Parameter Server and sparse-embedding training
- [ ] Online inference serving: low latency, high concurrency, and model compression
- [ ] Vector retrieval systems: ANN algorithms (HNSW, IVF) and Faiss/Milvus
- [ ] Real-time recommendation: streaming features and online learning

### Part 13 · Search, Recommendation, and Advertising

- [ ] Similarities and differences between search and recommendation: proactive intent vs. passive discovery
- [ ] Computational advertising basics: eCPM, CTR, and CVR prediction
- [ ] What makes ad ranking special: bidding, budget, and billing mechanisms
- [ ] A unified ranking framework across search, recommendation, and advertising

### Part 14 · LLMs and Recommender Systems

- [ ] The evolution of recommendation paradigms: from ID-based to semantic recommendation
- [ ] LLMs as feature extractors: text-semantic enhancement of item representations
- [ ] LLMs doing recommendation directly: prompt-based recommendation and generative recommendation
- [ ] Generative retrieval: semantic IDs and TIGER
- [ ] Combining LLMs with the traditional recommendation pipeline: recall enhancement, ranking features, and explanation generation

> After finishing a post: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.

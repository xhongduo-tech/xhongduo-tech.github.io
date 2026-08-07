---
pageClass: plain-doc
---

# 推荐系统

以《推荐系统实践》（项亮）为骨架，结合现代工业界推荐体系的经典论文与工程实践，从基础算法一路写到大规模在线服务与 LLM 推荐。

## 主题规划

<ProgressGrid cat="advanced/recommender-systems" />


### 第一篇 推荐系统概述与评估

- [x] [什么是推荐系统：信息过载与长尾问题](./what-is-recommender-system)
- [x] [推荐系统的数据来源：显式反馈与隐式反馈](./data-sources-explicit-implicit-feedback)
- [x] [推荐系统实验方法：离线实验、用户调查与在线实验](./experiment-methods-offline-user-online)
- [x] [离线评估指标：准确率、召回率、RMSE 与 MAE](./offline-evaluation-metrics)
- [x] [排序指标：MAP、NDCG 与 AUC](./ranking-metrics-map-ndcg-auc)
- [x] [超越准确性：多样性、新颖性、惊喜度与覆盖率](./beyond-accuracy-diversity-novelty-serendipity-coverage)
- [x] [A/B 测试：分流、分层实验与指标显著性](./ab-testing-traffic-splitting-stratified-experiments)

### 第二篇 协同过滤

- [x] [协同过滤的基本思想与邻域方法框架](./collaborative-filtering-basics-neighborhood-methods)
- [x] [相似度计算：余弦相似度、皮尔逊相关系数与 Jaccard](./similarity-cosine-pearson-jaccard)
- [x] [基于用户的协同过滤（UserCF）：原理与实现](./usercf-principle-and-implementation)
- [x] [UserCF 的相似度改进：对热门物品的惩罚（IUF 思想）](./usercf-similarity-improvement-iuf)
- [x] [基于物品的协同过滤（ItemCF）：原理与实现](./itemcf-principle-and-implementation)
- [x] [ItemCF 的归一化与活跃用户惩罚](./itemcf-normalization-active-user-penalty)
- [x] [UserCF 与 ItemCF 的对比与适用场景](./usercf-vs-itemcf-comparison-scenarios)
- [x] [协同过滤的优缺点：稀疏性、冷启动与可解释性](./collaborative-filtering-pros-cons-sparsity-cold-start)

### 第三篇 隐语义模型与矩阵分解

- [x] [隐语义模型（LFM）的基本思想：隐因子与兴趣分类](./latent-factor-model-lfm-basics)
- [x] [基于 SVD 的矩阵分解：Funk-SVD 与梯度下降求解](./funk-svd-matrix-factorization)
- [x] [带偏置项的矩阵分解（BiasSVD）](./biassvd-matrix-factorization)
- [x] [隐式反馈的矩阵分解：加权交替最小二乘（ALS-WR）](./implicit-feedback-als-wr)
- [x] [SVD++：融合显式与隐式反馈](./svd-plus-plus)
- [x] [矩阵分解的局限：泛化能力弱与特征利用不足](./matrix-factorization-limitations)

### 第四篇 基于内容的推荐

- [x] [基于内容推荐的基本框架：物品画像与用户画像](./content-based-filtering-framework-item-user-profiles)
- [x] [文本内容的表示：TF-IDF 与词袋模型](./text-representation-tfidf-bag-of-words)
- [x] [从 TF-IDF 到词向量：Word2Vec 与物品 Embedding](./word2vec-item-embedding)
- [x] [内容特征下的用户兴趣建模与相似度匹配](./content-based-user-profiling-similarity-matching)
- [x] [基于内容的推荐与协同过滤的混合策略](./hybrid-recommendation-content-collaborative)

### 第五篇 召回策略

- [x] [推荐系统整体架构：召回、粗排、精排与重排](./recsys-overall-architecture-recall-ranking-reranking)
- [x] [多路召回的设计原则：互补性与融合策略](./multi-channel-recall-design-principles)
- [x] [向量召回：双塔模型的结构与训练目标](./two-tower-model-vector-recall)
- [x] [双塔召回中的负采样：batch 内负采样与难负样本](./two-tower-negative-sampling-hard)
- [x] [多兴趣召回：MIND 与 ComiRec](./mind-comirec-multi-interest-recall)
- [x] [图召回：Graph Embedding（DeepWalk、Node2Vec、EGES）与 PinSage](./graph-embedding-recall-deepwalk-node2vec-pinsage)
- [x] [召回层的评估：召回率、hit rate 与链路一致性分析](./recall-layer-evaluation-hit-rate)

### 第六篇 排序模型演进

- [x] [排序问题的形式化：CTR 预估与 pointwise 建模](./ctr-prediction-pointwise-formulation)
- [x] [逻辑回归（LR）：可解释性与特征工程的艺术](./logistic-regression-ctr-feature-engineering)
- [x] [GBDT+LR：特征交叉的自动化（Facebook 方案）](./gbdt-lr-automatic-feature-cross)
- [x] [FM（因子分解机）：稀疏数据下的二阶特征交互](./fm-factorization-machines)
- [x] [FFM：引入 field 感知的因子分解](./ffm-field-aware-factorization-machines)
- [x] [Wide & Deep：记忆能力与泛化能力的结合](./wide-and-deep)
- [x] [DeepFM：用 FM 替代 Wide 侧的手工特征](./deepfm)
- [x] [DCN 与 xDeepFM：显式高阶特征交叉](./dcn-xdeepfm-explicit-feature-crossing)
- [x] [DIN：注意力机制建模用户兴趣](./din-deep-interest-network)
- [x] [DIEN：兴趣演化网络与序列建模](./dien-deep-interest-evolution)

### 第七篇 序列推荐

- [x] [序列推荐问题定义：从静态兴趣到动态行为序列](./sequential-recommendation-problem-definition)
- [x] [马尔可夫链与 FPMC：融合矩阵分解的一阶序列模型](./fpmc-markov-chain-sequential)
- [x] [GRU4Rec：基于 RNN 的会话推荐](./gru4rec-session-recommendation)
- [x] [SASRec：基于自注意力的序列推荐](./sasrec-self-attention)
- [x] [BERT4Rec：双向序列建模与完形填空式训练](./bert4rec-bidirectional)
- [x] [长序列建模：SIM 的两阶段兴趣检索](./sim-long-sequence-interest-retrieval)

### 第八篇 多目标与多任务学习

- [x] [多目标排序问题：点击、时长、点赞与消费的权衡](./multi-objective-ranking-click-duration)
- [x] [多目标融合：加权求和、乘法公式与进化学习](./multi-objective-fusion-weighting)
- [x] [MMOE：多门控混合专家模型](./mmoe-multi-gate-mixture-of-experts)
- [x] [PLE：渐进分层抽取与任务间冲突缓解](./ple-progressive-layered-extraction)
- [x] [ESMM：全空间建模解决 CVR 预估的样本选择偏差](./esmm-entire-space-multi-task)

### 第九篇 冷启动

- [x] [冷启动问题分类：用户冷启动、物品冷启动与系统冷启动](./cold-start-classification)
- [x] [利用用户注册信息与人口统计学特征](./cold-start-user-registration-demographics)
- [x] [新用户冷启动：选择合适的物品收集反馈](./cold-start-new-user-feedback-collection)
- [x] [新物品冷启动：内容特征与物品属性利用](./cold-start-new-item-content-features)
- [x] [元学习在冷启动中的应用：MeLU 与 MAML 思路](./meta-learning-cold-start-melu-maml)

### 第十篇 探索与利用

- [x] [探索与利用（Exploration & Exploitation）问题](./exploration-exploitation-problem)
- [x] [多臂老虎机（Bandit）问题与遗憾（Regret）分析](./multi-armed-bandit-regret)
- [x] [ε-Greedy 与朴素探索策略](./epsilon-greedy-exploration)
- [x] [UCB：置信上界算法](./ucb-confidence-bound)
- [x] [Thompson Sampling：贝叶斯视角的探索](./thompson-sampling-bayesian-exploration)
- [x] [LinUCB 与上下文老虎机（Contextual Bandit）](./linucb-contextual-bandit)

### 第十一篇 重排与多样性

- [x] [重排层的定位：从单点最优到列表最优](./reranking-layer-list-wise)
- [x] [多样性的定义与度量：ILAD、ILD 与类目打散](./diversity-metrics-ild-category)
- [x] [MMR：最大边际相关性重排](./mmr-maximal-marginal-relevance)
- [x] [DPP：基于行列式点过程的多样性重排](./dpp-determinantal-point-process)
- [x] [序列感知的重排模型：PRM 与生成式重排](./prm-generative-reranking)

### 第十二篇 推荐系统工程

- [x] [工业推荐系统架构总览：数据流与服务链路](./industrial-recsys-architecture)
- [x] [特征平台：离线特征、在线特征与特征一致性](./feature-platform-offline-online-consistency)
- [x] [样本构建：Label 延迟、样本回灌与归因窗口](./sample-construction-label-delay)
- [x] [模型训练框架：Parameter Server 与 Embedding 稀疏参数训练](./parameter-server-embedding-training)
- [x] [在线推理服务：低延迟、高并发与模型压缩](./online-inference-low-latency)
- [x] [向量检索系统：ANN 算法（HNSW、IVF）与 Faiss/Milvus](./ann-vector-search-hnsw-faiss)
- [x] [实时推荐：流式特征与在线学习](./realtime-recommendation-streaming)

### 第十三篇 搜索、推荐与广告

- [x] [搜索与推荐的异同：主动意图与被动发现](./search-vs-recommendation)
- [x] [计算广告基础：eCPM、CTR 与 CVR 预估](./computational-advertising-ecpm)
- [x] [广告排序的特殊性：出价、预算与计费机制](./ad-ranking-bidding-budget)
- [x] [搜推广场景下的统一排序框架](./unified-ranking-search-recommendation-ads)

### 第十四篇 LLM 与推荐系统

- [x] [推荐范式的演进：ID 推荐到语义推荐](./recommender-paradigm-evolution-id-to-semantic)
- [x] [LLM 作为特征提取器：文本语义增强物品表示](./llm-as-feature-extractor)
- [x] [LLM 直接做推荐：Prompt 化推荐与生成式推荐](./llm-prompt-based-recommendation)
- [x] [生成式检索：语义 ID 与 TIGER](./generative-retrieval-semantic-id-tiger)
- [x] [LLM 与传统推荐链路结合：召回增强、排序特征与解释生成](./llm-traditional-recsys-integration)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

---
pageClass: plain-doc
---

# 信息检索

对标《Introduction to Information Retrieval》（Manning）与现代搜索体系，从倒排索引与经典检索模型一路写到语义检索、学习排序与 RAG，覆盖一个工业级搜索系统所需的全部理论与实践。

## 主题规划

<ProgressGrid cat="advanced/information-retrieval" />


### 第一篇 信息检索概述与评测

- [x] [什么是信息检索：从结构化查询到非结构化文本检索](./what-is-information-retrieval)
- [x] [检索系统的基本任务：相关性与用户需求](./retrieval-system-tasks-relevance)
- [x] [倒排索引的一个例子：莎士比亚文集检索](./shakespeare-inverted-index-example)
- [x] [检索评测方法：测试集、查询集与相关性判定](./retrieval-evaluation-methodology)
- [x] [查准率与查全率（Precision / Recall）及其权衡](./precision-recall-tradeoff)
- [x] [F 值（F-Measure）与插值查准率](./f-measure-interpolated-precision)
- [x] [平均查准率（MAP）与 R-Precision](./map-and-r-precision)
- [x] [排序评测指标：NDCG、MRR 与 ERR](./ranking-metrics-ndcg-mrr-err)
- [x] [用户满意度评测：点击率、停留时间与线上 A/B 测试](./user-satisfaction-evaluation)

### 第二篇 布尔检索与倒排索引

- [x] [布尔检索模型与布尔查询处理](./boolean-retrieval-model)
- [x] [倒排索引的结构：词典与倒排表](./inverted-index-structure)
- [x] [倒排索引的构建：内存式构建流程](./in-memory-index-construction)
- [x] [基于块排序的索引构建（BSBI）](./bsbi-index-construction)
- [x] [基于哈希的内存单次扫描构建（SPIMI）](./spimi-in-memory-single-pass)
- [x] [分布式索引构建：MapReduce 方案](./distributed-indexing-mapreduce)
- [x] [动态索引：辅助索引与即时合并策略](./dynamic-indexing-merge)
- [x] [索引压缩的意义与词典压缩技术](./dictionary-compression)
- [x] [倒排表压缩：可变字节编码与 γ 编码](./postings-compression-vb-gamma)
- [x] [跳表（Skip List）与跳表指针的设计](./skip-list-postings)

### 第三篇 词典与容错检索

- [x] [词典的数据结构：哈希表与搜索树](./dictionary-data-structures)
- [x] [通配符查询：轮排索引与 k-gram 索引](./wildcard-queries-permutation-kgram)
- [x] [拼写矫正：编辑距离与加权编辑距离](./edit-distance-spelling-correction)
- [x] [基于 k-gram 重叠度的拼写矫正](./kgram-overlap-spelling-correction)
- [x] [发音矫正：Soundex 算法](./soundex-phonetic-correction)
- [x] [中文检索的特殊性：分词与单字索引](./chinese-retrieval-tokenization)

### 第四篇 词项权重与向量空间模型

- [x] [Jaccard 系数与词袋模型的局限](./jaccard-bag-of-words)
- [x] [词项频率（TF）与对数词频权重](./term-frequency-log-weighting)
- [x] [逆文档频率（IDF）及其概率解释](./inverse-document-frequency)
- [x] [TF-IDF 权重机制](./tf-idf-weighting)
- [x] [向量空间模型与余弦相似度](./vector-space-model-cosine)
- [x] [文档长度归一化](./document-length-normalization)
- [x] [SMART 权重体系与 lnc.ltc 方案](./smart-weighting-lnc-ltc)

### 第五篇 概率检索与 BM25

- [x] [概率排序原理（PRP）](./probability-ranking-principle)
- [x] [二元独立模型（BIM）](./binary-independence-model)
- [x] [Okapi BM25 排序函数](./okapi-bm25)
- [x] [BM25 的参数调节：k1、b 与文档长度归一化](./bm25-parameter-tuning)
- [x] [BM25F 与多字段加权检索](./bm25f-multifield-weighting)

### 第六篇 语言模型检索

- [x] [查询似然模型（Query Likelihood Model）](./query-likelihood-model)
- [x] [数据平滑：Jelinek-Mercer 平滑](./jelinek-mercer-smoothing)
- [x] [Dirichlet 先验平滑](./dirichlet-prior-smoothing)
- [x] [KL 散度检索模型](./kl-divergence-retrieval)
- [x] [相关性模型 RM3 与伪相关反馈](./relevance-model-rm3)

### 第七篇 学习排序（Learning to Rank）

- [x] [学习排序的问题定义与特征工程](./learning-to-rank-problem)
- [x] [Pointwise 方法：回归与分类视角](./pointwise-learning-to-rank)
- [x] [Pairwise 方法：RankNet 与 RankSVM](./pairwise-ranknet-ranksvm)
- [x] [Listwise 方法：ListNet 与 LambdaRank](./listwise-listnet-lambdarank)
- [x] [LambdaMART：梯度提升树与 λ 梯度](./lambdamart)
- [x] [排序模型的评测与训练数据构建：人工标注与点击日志](./ltr-evaluation-training-data)
- [x] [点击偏差与无偏学习排序（IPS、位置偏差模型）](./click-bias-unbiased-ltr)

### 第八篇 语义检索与稠密向量

- [x] [稀疏检索的语义鸿沟问题](./sparse-retrieval-semantic-gap)
- [x] [词向量与句向量表示基础](./word-sentence-embeddings)
- [x] [双塔模型（Dual Encoder / Bi-Encoder）与对比学习](./dual-encoder-contrastive-learning)
- [x] [双塔模型的负采样与难负例挖掘](./hard-negative-mining)
- [x] [ANN 索引综述：精确搜索的代价](./ann-index-survey)
- [x] [IVF 索引：聚类倒排与乘积量化（PQ）](./ivf-product-quantization)
- [x] [HNSW：分层可导航小世界图](./hnsw-hierarchical-navigable-small-world)
- [x] [稀疏与稠密混合检索：融合策略与 RRF](./hybrid-retrieval-rrf)
- [x] [重排序（Rerank）：交叉编码器（Cross-Encoder）](./reranking-cross-encoder)

### 第九篇 查询理解与改写

- [x] [查询意图分类](./query-intent-classification)
- [x] [查询纠错与查询分词](./query-correction-segmentation)
- [x] [查询扩展：同义词、词库与全局分析](./query-expansion)
- [x] [查询改写：生成式改写的兴起](./generative-query-rewriting)
- [x] [相关反馈：Rocchio 算法与伪相关反馈](./rocchio-relevance-feedback)
- [x] [查询性能预测](./query-performance-prediction)

### 第十篇 搜索系统架构

- [x] [搜索引擎的整体架构：离线索引与在线查询](./search-engine-architecture)
- [x] [索引构建流水线：文档处理、分词与索引更新](./indexing-pipeline)
- [x] [查询处理：两级架构（召回 + 排序）](./two-stage-retrieval-ranking)
- [x] [分布式检索：文档划分与词项划分](./distributed-retrieval-partitioning)
- [x] [索引分片、副本与负载均衡](./index-sharding-replication)
- [x] [缓存机制：结果缓存与倒排表缓存](./search-caching)
- [x] [级联排序：从粗排到精排的效率设计](./cascade-ranking)
- [x] [搜索系统的性能指标：延迟、吞吐与索引新鲜度](./search-system-performance-metrics)

### 第十一篇 个性化搜索

- [x] [个性化搜索的问题定义](./personalized-search-problem)
- [x] [用户画像：长期兴趣与短期兴趣建模](./user-profiling)
- [x] [基于上下文的查询理解与会话内个性化](./contextual-personalization)
- [x] [个性化重排序方法](./personalized-reranking)
- [x] [个性化的评测难题与隐私问题](./personalization-evaluation-privacy)

### 第十二篇 多模态检索

- [x] [多模态检索概述：图文检索与视频检索](./multimodal-retrieval-overview)
- [x] [跨模态表示学习：CLIP 及其训练机制](./clip-cross-modal-representation)
- [x] [图像检索：以图搜图的特征与索引](./image-retrieval)
- [x] [视频检索：时序特征与片段定位](./video-retrieval)
- [x] [多模态混合检索系统的工程实践](./multimodal-hybrid-engineering)

### 第十三篇 搜索与 RAG

- [x] [从搜索引擎到生成式问答：RAG 的诞生](./from-search-to-rag)
- [x] [RAG 的基本架构：检索器 + 生成器](./rag-architecture)
- [x] [RAG 中的文档切分（Chunking）策略](./rag-chunking)
- [x] [检索质量对生成质量的影响：召回率与上下文窗口](./retrieval-quality-generation)
- [x] [混合检索在 RAG 中的应用](./hybrid-retrieval-rag)
- [x] [RAG 的评测：忠实度、相关性与答案正确性](./rag-evaluation)
- [x] [Agentic RAG 与多轮检索规划](./agentic-rag)

### 第十四篇 对话式搜索

- [x] [对话式搜索概述：从单轮查询到多轮交互](./conversational-search-overview)
- [x] [对话上下文建模与查询重写](./conversational-context-query-rewriting)
- [x] [对话式检索（Conversational Dense Retrieval）](./conversational-dense-retrieval)
- [x] [主动澄清：向用户提问澄清需求](./active-clarification)
- [x] [对话式搜索的评测方法](./conversational-search-evaluation)
- [x] [大模型时代的搜索形态：AI 搜索产品剖析](./ai-search-products)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。

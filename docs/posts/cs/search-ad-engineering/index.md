---
pageClass: plain-doc
---

# 搜索引擎与计算广告工程（索引/排序/竞价）

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Manning, Raghavan & Schütze, "Introduction to Information Retrieval" (2008)
- Liu, "Learning to Rank for Information Retrieval" (2011)
- Google Ads / 计算广告 (刘鹏《计算广告》体系, 2019)

## 主题规划

<ProgressGrid cat="cs/search-ad-engineering" />

### 第1篇

- [x] [搜索引擎总览（爬取→索引→检索→排序的链路）](./search-engine-overview)
- [x] [倒排索引（词典压缩、跳表、索引构建与更新）](./inverted-index)
- [x] [检索模型（布尔/向量空间/BM25 的概率框架）](./retrieval-models)
- [x] [查询理解与改写（分词/纠错/意图识别）](./query-understanding-and-rewriting)
- [x] [学习排序 LTR（Pointwise/Pairwise/Listwise、LambdaMART）](./learning-to-rank)
- [x] [语义检索（稠密向量召回、ANN 索引 HNSW/IVF）](./semantic-search-and-ann)
- [x] [搜索评测（NDCG/MAP、AB 测试与 Interleaving）](./search-evaluation)
- [x] [计算广告模式（合约广告/竞价广告/程序化交易）](./online-advertising-models)

### 第2篇

- [x] [广告拍卖机制（GSP/VCG、广义二价的经济学）](./ad-auction-mechanisms)
- [x] [CTR 预估（LR→FM→深度模型的特征工程演进）](./ctr-prediction)
- [x] [广告系统工程（流量分配、预算平滑、反作弊）](./ad-system-engineering)
- [x] [生成式搜索（RAG 重塑搜索、答案引擎的商业化挑战）](./generative-search-and-rag)

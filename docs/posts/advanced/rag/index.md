---
pageClass: plain-doc
---

# RAG 与检索增强

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- Vladimir Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020)
- Akari Asai et al., "Bridging the Generalization Gap in Retrieval-Augmented Generation" (2024)

## 主题规划

<ProgressGrid cat="advanced/rag" />

### 第1篇

- [x] [RAG 框架与架构 (Lewis et al., RAG 2020)](./rag-framework-architecture)
- [x] [向量检索与 ANN (Karpukhin et al., DPR 2020)](./vector-search-ann)
- [x] [稠密检索 DPR (Karpukhin et al., DPR 2020)](./dense-passage-retrieval)
- [x] [重排序 Cross-Encoder (Karpukhin et al., DPR 2020)](./reranking-cross-encoder)
- [x] [RAG-Sequence 与 RAG-Token (Lewis et al., RAG 2020)](./rag-sequence-rag-token)
- [x] [检索增强多跳推理 (Asai et al., 2024)](./multi-hop-retrieval)
- [x] [检索增强自反 Self-RAG (Asai et al., Self-RAG 2023)](./self-rag)
- [x] [RAG 评估与基准 (Asai et al., 2024)](./rag-evaluation-benchmarks)

### 第2篇

- [x] [文档分块策略（固定/语义/递归分块） (Gao et al., RAG Survey 2023)](./chunking-strategies)
- [x] [嵌入模型选择与评估 (Reimers & Gurevych, SBERT 2019)](./embedding-model-selection)
- [x] [向量数据库（FAISS/Milvus/Pinecone） (Johnson et al., FAISS 2017)](./vector-databases)
- [x] [生成融合与 RAG-Fusion (Raudaschl, RAG-Fusion 2023)](./rag-fusion)
- [x] [查询重写与扩展 (Ma et al., 2023)](./chunking-strategies)
- [x] [混合检索（BM25+稠密混合） (Robertson & Zaragoza, BM25 2009)](./hybrid-search-bm25)

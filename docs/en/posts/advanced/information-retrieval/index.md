---
pageClass: plain-doc
---

# Information Retrieval

Framed around *Introduction to Information Retrieval* (Manning) and modern search systems, this track runs from inverted indexes and classic retrieval models all the way to semantic retrieval, learning to rank, and RAG, covering the full set of theory and practice needed to build an industrial-grade search system.

## Topic Plan

<ProgressGrid cat="advanced/information-retrieval" />


### Part 1 Overview of Information Retrieval and Evaluation

- [ ] What information retrieval is: from structured queries to unstructured text retrieval
- [ ] The basic tasks of a retrieval system: relevance and user needs
- [ ] An example of an inverted index: searching the Shakespeare corpus
- [ ] Retrieval evaluation methods: test collections, query sets, and relevance judgments
- [ ] Precision / Recall and the precision-recall trade-off
- [ ] F-Measure and interpolated precision
- [ ] Mean Average Precision (MAP) and R-Precision
- [ ] Ranking metrics: NDCG, MRR, and ERR
- [ ] User satisfaction evaluation: click-through rate, dwell time, and online A/B testing

### Part 2 Boolean Retrieval and the Inverted Index

- [ ] The Boolean retrieval model and Boolean query processing
- [ ] Structure of the inverted index: the dictionary and postings
- [ ] Constructing the inverted index: an in-memory construction pipeline
- [ ] Block sort-based index construction (BSBI)
- [ ] Single-pass in-memory indexing (SPIMI)
- [ ] Distributed index construction: a MapReduce approach
- [ ] Dynamic indexing: auxiliary indexes and immediate merge strategies
- [ ] Why index compression matters, and dictionary compression techniques
- [ ] Postings compression: variable byte encoding and γ codes
- [ ] Skip lists and the design of skip pointers

### Part 3 The Dictionary and Tolerant Retrieval

- [ ] Data structures for the dictionary: hash tables and search trees
- [ ] Wildcard queries: permuterm indexes and k-gram indexes
- [ ] Spelling correction: edit distance and weighted edit distance
- [ ] Spelling correction based on k-gram overlap
- [ ] Phonetic correction: the Soundex algorithm
- [ ] The special case of Chinese retrieval: segmentation and character-based indexing

### Part 4 Term Weighting and the Vector Space Model

- [ ] The Jaccard coefficient and the limits of the bag-of-words model
- [ ] Term frequency (TF) and log-frequency weighting
- [ ] Inverse document frequency (IDF) and its probabilistic interpretation
- [ ] The TF-IDF weighting scheme
- [ ] The vector space model and cosine similarity
- [ ] Document length normalization
- [ ] The SMART weighting system and the lnc.ltc scheme

### Part 5 Probabilistic Retrieval and BM25

- [ ] The Probability Ranking Principle (PRP)
- [ ] The Binary Independence Model (BIM)
- [ ] The Okapi BM25 ranking function
- [ ] Tuning BM25's parameters: k1, b, and document length normalization
- [ ] BM25F and multi-field weighted retrieval

### Part 6 Language Model Retrieval

- [ ] The Query Likelihood Model
- [ ] Data smoothing: Jelinek-Mercer smoothing
- [ ] Dirichlet prior smoothing
- [ ] The KL-divergence retrieval model
- [ ] Relevance models RM3 and pseudo-relevance feedback

### Part 7 Learning to Rank

- [ ] Problem definition and feature engineering for learning to rank
- [ ] Pointwise methods: the regression and classification views
- [ ] Pairwise methods: RankNet and RankSVM
- [ ] Listwise methods: ListNet and LambdaRank
- [ ] LambdaMART: gradient boosted trees and λ gradients
- [ ] Evaluating ranking models and building training data: human labels and click logs
- [ ] Click bias and unbiased learning to rank (IPS, position bias models)

### Part 8 Semantic Retrieval and Dense Vectors

- [ ] The semantic gap problem of sparse retrieval
- [ ] Foundations of word and sentence embeddings
- [ ] Dual encoders (Bi-Encoders) and contrastive learning
- [ ] Negative sampling and hard negative mining for dual encoders
- [ ] A survey of ANN indexes: the cost of exact search
- [ ] IVF indexes: cluster-based inversion and product quantization (PQ)
- [ ] HNSW: hierarchical navigable small world graphs
- [ ] Hybrid sparse-dense retrieval: fusion strategies and RRF
- [ ] Reranking with cross-encoders

### Part 9 Query Understanding and Rewriting

- [ ] Query intent classification
- [ ] Query correction and query segmentation
- [ ] Query expansion: synonyms, thesauri, and global analysis
- [ ] Query rewriting: the rise of generative rewriting
- [ ] Relevance feedback: the Rocchio algorithm and pseudo-relevance feedback
- [ ] Query performance prediction

### Part 10 Search System Architecture

- [ ] Overall architecture of a search engine: offline indexing and online querying
- [ ] The indexing pipeline: document processing, tokenization, and index updates
- [ ] Query processing: a two-stage architecture (recall + ranking)
- [ ] Distributed retrieval: document partitioning and term partitioning
- [ ] Index sharding, replicas, and load balancing
- [ ] Caching: result caches and postings caches
- [ ] Cascaded ranking: efficiency design from coarse to fine ranking
- [ ] Performance metrics for search systems: latency, throughput, and index freshness

### Part 11 Personalized Search

- [ ] Problem definition for personalized search
- [ ] User profiles: modeling long-term and short-term interests
- [ ] Context-aware query understanding and in-session personalization
- [ ] Personalized reranking methods
- [ ] The evaluation challenges and privacy concerns of personalization

### Part 12 Multimodal Retrieval

- [ ] Overview of multimodal retrieval: image-text retrieval and video retrieval
- [ ] Cross-modal representation learning: CLIP and its training mechanism
- [ ] Image retrieval: features and indexes for reverse image search
- [ ] Video retrieval: temporal features and segment localization
- [ ] Engineering practices for hybrid multimodal retrieval systems

### Part 13 Search and RAG

- [ ] From search engines to generative QA: the birth of RAG
- [ ] The basic RAG architecture: retriever + generator
- [ ] Document chunking strategies in RAG
- [ ] How retrieval quality affects generation quality: recall and context windows
- [ ] Applying hybrid retrieval in RAG
- [ ] Evaluating RAG: faithfulness, relevance, and answer correctness
- [ ] Agentic RAG and multi-hop retrieval planning

### Part 14 Conversational Search

- [ ] Overview of conversational search: from single-turn queries to multi-turn interaction
- [ ] Conversational context modeling and query rewriting
- [ ] Conversational Dense Retrieval
- [ ] Active clarification: asking users clarifying questions
- [ ] Evaluation methods for conversational search
- [ ] The search paradigm in the LLM era: dissecting AI search products

> After writing: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.

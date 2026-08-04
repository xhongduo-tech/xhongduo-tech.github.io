---
pageClass: plain-doc
---

# Natural Language Processing

This category follows the classic framework of *Natural Language Processing* (Zong Chengqing) and Jurafsky & Martin's *Speech and Language Processing*, covering the full range of NLP as it developed before (and in parallel with) large language models: from text preprocessing and language models to syntactic and semantic analysis, machine translation, information extraction, dialogue, summarization, and other core tasks.

## Topic Planning

<ProgressGrid cat="advanced/nlp" />


### Part 1 · NLP Overview

- [ ] What is natural language processing: definition, tasks, and research scope
- [ ] A brief history of NLP: rule-based, statistical, and neural network methods
- [ ] Ambiguity in natural language and the fundamental difficulties of NLP
- [ ] The taxonomy of NLP tasks and typical applications

### Part 2 · Linguistics and Corpus Foundations

- [ ] Linguistics basics: phonetics, morphology, syntax, semantics, and pragmatics
- [ ] Types of corpora and their construction: raw corpora and annotated corpora
- [ ] Common Chinese and English corpora (Penn Treebank, CTB, People's Daily corpus, etc.)
- [ ] Corpus statistics basics: word frequency, Zipf's Law, and the sparsity problem
- [ ] Regular expressions and text matching

### Part 3 · Text Preprocessing

- [ ] The Chinese word segmentation problem: ambiguous segmentation and out-of-vocabulary word recognition
- [ ] Dictionary-based segmentation: maximum matching and shortest-path segmentation
- [ ] Statistics-based segmentation: character-based (character labeling) segmentation
- [ ] Part-of-speech tagging (POS Tagging): tag sets and tagging methods
- [ ] Named entity normalization, stop words, and Stemming & Lemmatization
- [ ] Text representation basics: bag of words (BoW) and TF-IDF

### Part 4 · n-gram Language Models

- [ ] Language model overview: probabilistic definition and the chain rule
- [ ] n-gram models: the Markov assumption and parameter estimation
- [ ] Data smoothing: additive smoothing and Good-Turing estimation
- [ ] Data smoothing: interpolation and Kneser-Ney smoothing
- [ ] Perplexity and the evaluation of language models
- [ ] Neural network language models (NNLM) and recurrent neural network language models (RNNLM)

### Part 5 · Word Embeddings and Distributed Representations

- [ ] The distributional hypothesis and distributed semantic representations
- [ ] Word2Vec: CBOW and Skip-gram models
- [ ] Word2Vec training techniques: negative sampling and hierarchical softmax
- [ ] GloVe: word vectors based on global word co-occurrence statistics
- [ ] FastText: subword information and representations of out-of-vocabulary words
- [ ] Evaluating word vectors: word similarity and analogy tasks
- [ ] Context-dependent word representations: CoVe and ELMo

### Part 6 · Sequence Labeling

- [ ] The sequence labeling problem and labeling schemes (BIO/BIOES)
- [ ] Hidden Markov Models (HMM): model definition and the three basic problems
- [ ] HMM learning and decoding: the forward algorithm and the Viterbi algorithm
- [ ] Maximum entropy Markov models (MEMM) and the label bias problem
- [ ] Conditional random fields (CRF): model definition and feature functions
- [ ] CRF parameter estimation and inference
- [ ] BiLSTM-CRF: combining neural networks with structured prediction

### Part 7 · Text Classification and Sentiment Analysis

- [ ] The text classification problem and feature engineering
- [ ] Naive Bayes text classifiers
- [ ] Logistic regression and support vector machine (SVM) classifiers
- [ ] CNN-based text classification (TextCNN)
- [ ] RNN- and attention-based text classification
- [ ] Sentiment analysis: sentiment lexicon methods and document-level sentiment classification
- [ ] Aspect-Based Sentiment Analysis

### Part 8 · Syntactic Parsing

- [ ] Syntactic parsing overview: phrase structure grammar and dependency grammar
- [ ] Context-free grammars (CFG) and probabilistic context-free grammars (PCFG)
- [ ] Constituency parsing: the CYK algorithm and PCFG-based statistical parsing
- [ ] Dependency parsing: transition-based methods (Arc-Standard / Arc-Eager)
- [ ] Dependency parsing: graph-based methods and neural dependency parsers

### Part 9 · Semantic Analysis

- [ ] Word Sense Disambiguation: supervised and dictionary-based methods
- [ ] Semantic Role Labeling
- [ ] FrameNet and PropBank semantic resources
- [ ] Semantic compositionality and distributional compositional semantics
- [ ] Textual entailment and natural language inference (NLI)
- [ ] Semantic Parsing and SQL generation

### Part 10 · Machine Translation

- [ ] Machine translation overview: history and evaluation methods (BLEU)
- [ ] Statistical machine translation: word-based and phrase-based models
- [ ] Statistical machine translation: IBM models and word alignment
- [ ] Neural machine translation: the encoder-decoder framework (Seq2Seq)
- [ ] The application of attention mechanisms in machine translation
- [ ] Transformer: self-attention and multi-head attention
- [ ] Handling out-of-vocabulary words and subword segmentation (BPE) in machine translation

### Part 11 · Information Extraction

- [ ] Information extraction overview: task taxonomy and evaluation campaigns (MUC/ACE)
- [ ] Named entity recognition (NER): sequence labeling methods and nested entity recognition
- [ ] Entity recognition and entity linking
- [ ] Relation extraction: pattern-based, supervised, and distant supervision methods
- [ ] Event extraction: event detection and argument extraction
- [ ] Coreference resolution
- [ ] Open-domain information extraction (Open IE)

### Part 12 · Knowledge Graphs and NLP

- [ ] Knowledge graph overview: representation, construction, and applications
- [ ] NLP techniques in knowledge graph construction: entity and relation acquisition
- [ ] Knowledge representation learning: TransE and its extensions
- [ ] Knowledge graph completion and knowledge reasoning
- [ ] Applications of knowledge graphs in NLP tasks

### Part 13 · Question Answering Systems

- [ ] Question answering overview: history and task types
- [ ] Retrieval-based QA and knowledge base question answering (KBQA)
- [ ] Machine reading comprehension (MRC): datasets and classic models
- [ ] Machine reading comprehension: BiDAF, QANet, and pretraining-based methods
- [ ] Open-domain QA: the retrieve-read two-stage framework

### Part 14 · Dialogue Systems

- [ ] Dialogue system overview: task-oriented dialogue and open-domain chitchat
- [ ] Task-oriented dialogue systems: dialogue state tracking and dialogue policy
- [ ] Natural language understanding (NLU): intent detection and slot filling
- [ ] Retrieval-based and generation-based response generation
- [ ] Dialogue system evaluation and end-to-end dialogue systems

### Part 15 · Text Summarization

- [ ] Automatic summarization overview: extractive and abstractive methods
- [ ] Extractive summarization: TextRank, LexRank, and supervised learning methods
- [ ] Abstractive summarization: Seq2Seq models and the pointer-generator network
- [ ] Summarization evaluation: ROUGE and human evaluation
- [ ] Multi-document summarization and query-focused summarization

### Part 16 · Text Generation

- [ ] Natural language generation (NLG) overview and task types
- [ ] Decoding strategies for text generation: greedy search, beam search, and sampling
- [ ] Data-to-text generation
- [ ] Text style transfer and controllable text generation
- [ ] Evaluating text generation and the exposure bias problem

### Part 17 · Discourse Analysis

- [ ] Discourse analysis overview: coherence and cohesion
- [ ] Rhetorical Structure Theory (RST) and discourse structure analysis
- [ ] Discourse relation recognition and the PDTB corpus
- [ ] Discourse coherence modeling and topic segmentation

### Part 18 · Cross-Lingual and Low-Resource NLP

- [ ] Cross-lingual NLP overview: language differences and the transfer problem
- [ ] Cross-lingual word embeddings and multilingual word representations
- [ ] Low-resource NLP: transfer learning, data augmentation, and semi-supervised methods
- [ ] Multilingual models (mBERT, XLM) and zero-shot cross-lingual transfer

### Part 19 · NLP Evaluation

- [ ] NLP evaluation methods: precision, recall, F1, and significance testing
- [ ] Benchmark datasets and leaderboards (GLUE, CLUE)
- [ ] Automatic and human evaluation of generation tasks
- [ ] Evaluating dataset bias, robustness, and generalization

> When finished writing: create `xxx.md` in this directory, then change the corresponding entry above to `- [x] [Title](./xxx)`.

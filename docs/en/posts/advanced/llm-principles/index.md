---
pageClass: plain-doc
---

# LLM Principles

A systematic breakdown of the complete knowledge tree of large language models: from development history, Tokenizers, architectural details, and pretraining, to long context, multimodality, reasoning, RAG, agents, hallucination, safety, and evaluation.

## Topic Roadmap

<ProgressGrid cat="advanced/llm-principles" />


### Part 1: Development History and Scaling Laws

- [ ] From statistical language models to neural language models: N-gram, Word2Vec, and RNNLM
- [ ] The establishment of Transformer and the pretraining paradigm: GPT-1 and BERT
- [ ] GPT-2 and GPT-3: scale as capability, and the emergence of in-context learning
- [ ] T5 and the unified Text-to-Text paradigm
- [ ] InstructGPT and ChatGPT: instruction tuning and alignment with human feedback
- [ ] After GPT-4 and the open-source ecosystem: the LLaMA lineage and the community boom
- [ ] Kaplan scaling laws: power-law relationships between loss and model size, data, and compute
- [ ] Chinchilla scaling laws: parameter-data allocation under compute-optimal conditions
- [ ] Emergent abilities: the debate over phenomenology, mechanistic explanations, and "metric artifacts"

### Part 2: Tokenizer and Tokenization

- [ ] The fundamental problems of tokenization: trade-offs among character, word, and subword granularity
- [ ] BPE (Byte Pair Encoding): the training algorithm and the encoding process
- [ ] WordPiece: how its objective function differs from BPE
- [ ] Unigram language model tokenization: subword segmentation from a probabilistic perspective
- [ ] SentencePiece and byte-level BPE: handling space-less languages and the end of OOV
- [ ] Vocabulary design: vocabulary size, multilingual coverage, and the Chinese "tokenization tax"
- [ ] Side effects of tokenization: digit splitting, code, spelling, and security issues
- [ ] Hands-on practice: training a Tokenizer from scratch with BPE

### Part 3: Layer-by-Layer Anatomy of the GPT Architecture

- [ ] Decoder-only architecture overview: the complete data flow from tokens to logits
- [ ] The word embedding layer: lookup tables, initialization, and weight tying
- [ ] Scaled dot-product attention: the math derivation of Q/K/V and Softmax
- [ ] Causal masking and multi-head attention: parallel "multiple perspectives"
- [ ] The feed-forward network: the position-wise two-layer MLP and parameter share analysis
- [ ] The placement of residual connections and normalization: the Pre-LN vs. Post-LN debate
- [ ] The output head, Softmax, and decoding sampling: Temperature, Top-k, and Top-p
- [ ] KV Cache: the acceleration core of autoregressive inference and its memory cost
- [ ] Estimating parameter counts and FLOPs: working through GPT-2 by hand
- [ ] Hands-on implementation: writing a mini-GPT from scratch in PyTorch

### Part 4: Comparing Mainstream Open-Source Architectures

- [ ] Anatomy of the LLaMA architecture: the "trio" of RMSNorm, RoPE, and SwiGLU
- [ ] Qwen architecture evolution: design changes from Qwen1 to Qwen3
- [ ] DeepSeek architecture evolution: MLA, DeepSeekMoE, and the system-level innovations of V3
- [ ] Mistral, Gemma, and other architectures: sliding-window attention and the spectrum of design choices
- [ ] Understanding a model from its config.json: the architectural choices behind hyperparameters

### Part 5: Normalization and Activation Functions

- [ ] LayerNorm vs. BatchNorm: why NLP chose the former
- [ ] RMSNorm: the simplification and speedup of dropping mean centering
- [ ] Pre-LN vs. Post-LN: the trade-off between training stability and final performance
- [ ] The evolution of activation functions: from ReLU and GELU to SwiGLU
- [ ] Advanced techniques: QK-Norm, Sandwich-LN, and soft capping of attention logits

### Part 6: Positional Encoding

- [ ] Why positional encoding is needed: the permutation invariance of self-attention
- [ ] Sinusoidal absolute positional encoding: construction and properties
- [ ] Relative positional encoding: T5 Bias and the idea of relative position representations
- [ ] The math behind RoPE: rotation matrices, the complex number perspective, and long-range decay
- [ ] Reading RoPE code closely: a line-by-line analysis of the HuggingFace implementation
- [ ] ALiBi: linear attention bias and decoupling training from inference length
- [ ] Positional interpolation and extrapolation: PI, NTK-aware, and YaRN

### Part 7: Attention Variants and Efficient Attention

- [ ] The bottleneck of standard MHA: analyzing KV Cache memory usage
- [ ] MQA (Multi-Query Attention): extreme compression by sharing K/V
- [ ] GQA (Grouped-Query Attention): a compromise between quality and efficiency
- [ ] MLA (Multi-head Latent Attention): DeepSeek's low-rank joint compression
- [ ] Sliding-window attention: local receptive fields and layer-by-layer information propagation
- [ ] Sparse attention: Sparse Transformer, Longformer, and BigBird
- [ ] FlashAttention: IO-aware block-based exact attention
- [ ] Linear attention and state space models: can Mamba replace attention?

### Part 8: Mixture of Experts (MoE)

- [ ] The core idea of MoE: sparse activation and conditional computation
- [ ] Gating and routing: implementing the Top-k router
- [ ] Load balancing: auxiliary losses and the expert collapse problem
- [ ] Switch Transformer: Top-1 routing and expert capacity factors
- [ ] DeepSeekMoE: fine-grained expert splitting and shared experts
- [ ] The engineering challenges of MoE: communication overhead, memory, and inference deployment

### Part 9: Pretraining Data Engineering

- [ ] A panorama of data sources: web, books, code, papers, and encyclopedias
- [ ] Data cleaning pipelines: deduplication, filtering, and personal information handling
- [ ] Quality filtering: heuristic rules, classifiers, and perplexity filtering
- [ ] Data mixing ratios: experimental methods for domain mixing proportions
- [ ] Curriculum learning and the annealing phase: data scheduling in late-stage training
- [ ] Synthetic data: Self-Instruct, textbook-style data, and their limits

### Part 10: Pretraining Objectives and Training Techniques

- [ ] Causal language modeling (CLM): the merits and demerits of next-token prediction
- [ ] MLM and Prefix-LM: the revival of understanding-oriented objectives
- [ ] UL2 and mixture-of-denoisers objectives: unifying different pretraining paradigms
- [ ] FIM (Fill-in-the-Middle): bidirectional context for code models
- [ ] Learning rate scheduling: Warmup, Cosine decay, and WSD
- [ ] Training stability: loss spikes, gradient explosion, and countermeasures
- [ ] μP (Maximal Update Parameterization): transferring hyperparameters across scales
- [ ] Mixed-precision training: FP16, BF16, and FP8
- [ ] Distributed training: data/tensor/pipeline parallelism and ZeRO

### Part 11: Long Context

- [ ] The value and bottleneck of long context: quadratic complexity and the failure of positional extrapolation
- [ ] Extending the context window: interpolation, extrapolation, and continued pretraining
- [ ] Efficient long-text attention: the trade-offs of sparsification and linearization
- [ ] Constructing long-text data: long-document filtering and synthetic tasks
- [ ] KV Cache compression: quantization, eviction, and selective retention
- [ ] Long-context evaluation: Needle-in-a-Haystack (NIAH) and RULER

### Part 12: Multimodal Large Models

- [ ] The overall multimodal paradigm: the three-stage encoder-alignment-LLM pipeline
- [ ] Visual encoders: ViT and SigLIP
- [ ] Vision-language alignment: CLIP's contrastive learning objective
- [ ] VLM architectures (I): LLaVA-style projectors and visual instruction tuning
- [ ] VLM architectures (II): Qwen-VL and dynamic resolution handling
- [ ] The audio modality: the Whisper encoder and end-to-end speech language models
- [ ] Video understanding: frame sampling, spatiotemporal compression, and long-video modeling
- [ ] Natively multimodal: unified understanding and generation via next-token prediction

### Part 13: Reasoning Ability

- [ ] Chain-of-Thought (CoT) prompting: getting the model to "think step by step"
- [ ] Self-Consistency: majority-vote path aggregation
- [ ] Tree of Thoughts (ToT) and search: exploring the reasoning space
- [ ] Process Reward Models (PRM) and Outcome Reward Models (ORM)
- [ ] Scaling test-time compute: trading more inference for higher accuracy
- [ ] The OpenAI o1 paradigm: RL-driven long chains of thought
- [ ] DeepSeek-R1: GRPO, pure-RL emergence, and the "Aha moment"
- [ ] Distilling reasoning ability: transferring long chains of thought to smaller models

### Part 14: Retrieval-Augmented Generation (RAG)

- [ ] The overall RAG framework: why retrieval mitigates hallucination and knowledge staleness
- [ ] Document chunking strategies: fixed windows, semantic splitting, and structure-aware chunking
- [ ] Text vectorization: embedding models and contrastive learning training
- [ ] Indexing and vector databases: HNSW, IVF, and quantization
- [ ] Retrieval strategies: dense retrieval, sparse retrieval (BM25), and hybrid retrieval
- [ ] Reranking: fine-grained ranking with cross-encoders
- [ ] Advanced RAG: query rewriting, HyDE, and iterative retrieval

### Part 15: Agents

- [ ] An overview of the agent paradigm: the ReAct "reason-act" loop
- [ ] Tool calling: the protocol and implementation of Function Calling
- [ ] Planning and reflection: Plan-and-Execute and Reflexion
- [ ] Memory mechanisms: short-term context and long-term external memory
- [ ] MCP and the tool ecosystem: a standardized model-tool interface
- [ ] Multi-agent systems: role division, debate, and collaboration
- [ ] Agent failure modes and evaluation: dimensions beyond success rate

### Part 16: Hallucination and Safety

- [ ] Types and causes of hallucination: factual/faithfulness hallucination from the perspectives of data, objectives, and decoding
- [ ] Detecting and mitigating hallucination: retrieval grounding, uncertainty estimation, and decoding constraints
- [ ] Jailbreak attacks and prompt injection: a panorama of the attack surface
- [ ] Safety alignment: safety rewards in SFT and RLHF, and Constitutional AI
- [ ] Red teaming: methodology and automated red teaming
- [ ] Privacy and bias: training data leakage, fairness, and value alignment

### Part 17: Evaluation

- [ ] An overview of evaluation systems: classifying capability dimensions and evaluation methods
- [ ] Knowledge and discipline evaluation: MMLU, C-Eval, and GPQA
- [ ] Math and code evaluation: GSM8K, MATH, HumanEval, and SWE-bench
- [ ] Instruction following and dialogue evaluation: MT-Bench, AlpacaEval, Arena, and the reliability of LLM-as-a-Judge
- [ ] Data contamination: benchmark overfitting and evaluation distortion

> After writing: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.

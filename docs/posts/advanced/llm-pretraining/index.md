---
pageClass: plain-doc
---

# 大模型预训练

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Ian Goodfellow et al., "Deep Learning" (2016)
- Alec Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Hugo Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)

## 主题规划

<ProgressGrid cat="advanced/llm-pretraining" />

### 第1篇

- [x] [Transformer 架构 (Vaswani et al., 2017)](./transformer-architecture)
- [x] [自回归预训练目标 (Radford et al., GPT-2 2019 §2)](./autoregressive-pretraining-objective)
- [x] [Scaling Laws (Kaplan et al., 2020; Hoffmann et al., Chinchilla 2022)](./scaling-laws)
- [x] [预训练数据构造 (Touvron et al., LLaMA §2.1)](./pretraining-data-construction)
- [x] [位置编码 RoPE (Touvron et al., LLaMA §2.2)](./rope-positional-encoding)
- [x] [RMSNorm 与 SwiGLU (Touvron et al., LLaMA §2.2)](./rmsnorm-and-swiglu)
- [x] [高效训练与并行 (Touvron et al., LLaMA §2.4)](./efficient-training-parallelism)
- [x] [训练稳定性与损失尖峰（loss spike） (Chowdhery et al., PaLM 2022)](./training-stability-loss-spike)

### 第2篇

- [x] [评估基准与 zero-shot (Radford et al., GPT-2 2019 §3)](./evaluation-benchmarks-zero-shot)
- [x] [分词算法（BPE/SentencePiece/Unigram） (Sennrich et al., BPE 2016)](./tokenization-bpe)
- [x] [模型初始化与正则化（Xavier/He init） (Goodfellow §8.4)](./initialization-regularization)
- [x] [优化器选择与学习率调度（AdamW/Cosine） (Goodfellow §8.5)](./optimizer-lr-scheduling)
- [x] [检查点保存与断点续训 (Megatron-LM §4)](./checkpoint-resume-training)

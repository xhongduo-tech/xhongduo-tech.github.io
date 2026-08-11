---
pageClass: plain-doc
---

# 长上下文与注意力优化

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Ian Goodfellow et al., "Deep Learning" (2016)
- Tri Dao & Daniel Y. Fu et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)
- Iz Beltagy et al., "Longformer: The Long-Document Transformer" (2020)

## 主题规划

<ProgressGrid cat="advanced/long-context" />

### 第1篇

- [x] [注意力计算复杂度 (Vaswani et al., 2017 §3.2)](./attention-complexity)
- [x] [稀疏注意力 Longformer (Beltagy et al., Longformer 2020)](./longformer-sparse-attention)
- [x] [线性注意力机制 (Katharopoulos et al., Linear Transformers 2020)](./linear-attention)
- [x] [FlashAttention IO 优化 (Dao et al., FlashAttention 2022)](./flashattention-io-optimization)
- [x] [分块注意力 Blockwise (Dao et al., FlashAttention 2022)](./blockwise-attention)
- [x] [滑动窗口注意力 (Beltagy et al., Longformer 2020)](./sliding-window-attention)
- [x] [位置编码外推（ALiBi/RoPE/YaRN） (Press et al., ALiBi 2022; Su et al., RoPE 2021; Peng et al., YaRN 2023)](./attention-complexity)
- [x] [GQA/MQA 与 KV Cache 优化 (Ainslie et al., GQA 2023; Shazeer, MQA 2019)](./blockwise-attention)

### 第2篇

- [x] [长上下文评估 LongBench (Bai et al., LongBench 2023)](./longbench-evaluation)

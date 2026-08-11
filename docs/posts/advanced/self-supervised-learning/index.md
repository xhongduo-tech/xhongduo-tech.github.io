---
pageClass: plain-doc
---

# 自监督学习

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Ian Goodfellow et al., "Deep Learning" (2016)
- Liu et al., "Self-Supervised Learning: Generative or Contrastive" (IEEE TNNLS 2023)
- Liu Jing & Tian, "Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey" (IEEE TPAMI 2021)

## 主题规划

<ProgressGrid cat="advanced/self-supervised-learning" />

### 第1篇

- [x] [自监督预训练任务 (Liu et al., TNNLS 2023 §2)](./self-supervised-pretraining-tasks)
- [x] [对比学习框架 SimCLR/MoCo (Liu et al., TNNLS 2023 §3)](./contrastive-learning-frameworks-simclr-moco)
- [x] [对比损失与 InfoNCE (Liu et al., TNNLS 2023 §3.2)](./contrastive-loss-infonce)
- [x] [掩码图像建模 MAE (Liu et al., TNNLS 2023 §4)](./masked-autoencoder-mae)
- [x] [BYOL 与负样本免方法 (Jing & Tian, TPAMI 2021 §5)](./byol-negative-free-methods)
- [x] [自蒸馏机制 BYOL/SimSiam 原理细化](./self-distillation-byol-simsiam)
- [x] [生成式自监督主线（GPT 自回归/掩码语言建模）](./generative-self-supervised-mainline-gpt-mlm)
- [x] [语言自监督 MLM (Devlin et al., BERT 2018)](./masked-language-modeling-bert)

### 第2篇

- [x] [视觉自监督评估协议](./visual-self-supervised-evaluation-protocols)
- [x] [多模态自监督 CLIP (Liu et al., TNNLS 2023 §6)](./multimodal-self-supervised-clip)

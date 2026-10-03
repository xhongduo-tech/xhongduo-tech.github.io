---
title: Transfusion
date: 2026-09-08
section: llm
---

# Transfusion

<div class="epigraph">
<p>不必为图像另开一套网。同一 Transformer 对文本做下一词交叉熵，对图像潜变量做扩散损失，一次前向里两种目标。</p>
<footer>—— Zhou 等 Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model</footer>
</div>

[上一课](/llm/unified-understanding-generation)用双塔或混合离散目标缝理解与生成。缺口是连续图像与离散文本在同层注意力里共存：文本是词表，图是 [潜空间 VAE](/llm/latent-vae) 的 $z$。Transfusion 把扩散训练塞进语言模型前向。后课蒸馏默认：图像支路仍可能要多步采样。

## 问题

纯离散统一（Chameleon）付 VQ 税。纯连续理解（LLaVA）没有像素生成头。若图像用独立扩散模型，权重不共享，谈图与造图的表征会分叉。Transfusion 的序列是文本 token 与噪声潜补丁交错；注意力按模态用因果（文本）或双向（图像块）掩码。损失是 $\mathcal{L}_{\mathrm{CE}}+\lambda\mathcal{L}_{\mathrm{diff}}$。

<span class="marginnote">直觉类比：把输入序列想成一份图文混排文档。文字从左到右逐字读（因果掩码）；插图部分允许来回看（双向掩码），而且这块「画布」一开始是涂了噪点的，模型一边读文字一边把它涂清楚。</span>

这与 [CFG](/llm/cfg-image) 兼容：图像位置走扩散采样，文本仍自回归。

<span class="marginnote">掩码设计是正确性来源：图像段内部可双向，但对文本因果——图像不看未来的文本，而图像之后的文本看得到（加噪的）图像，图生文必需。实现必须按论文设置，否则扩散训练时学到的可见集与推理对不上。</span>

## 方法

VAE 编码图像到潜补丁，加噪后当连续嵌入送入 Transformer（线性映到模型维）。文本位置出 logits；图像位置出噪声预测。训练混合纯文本、文生图、图生文。推理：文生图从噪声 $z_T$ 多步更新图像槽，同时可读文本前缀。

<span class="marginnote">术语翻译：扩散训练就是「先弄脏再考还原」——把干净图像潜变量 $z_0$ 加随机噪声得到 $z_t$，让模型看着 $z_t$ 猜加进来的噪声是什么。猜得准，反着一步步减噪声，就能从纯噪声还原出一张图。</span>

```mermaid
flowchart TD
  TXT["文本 token"] --> TR["同一 Transformer"]
  ZT["噪声潜补丁"] --> TR
  TR --> CE["下一词 CE"]
  TR --> DIFF["扩散损失"]
```

## 机制

```mermaid
flowchart TD
  MIX["一个 batch：文本段 + 图像段交错"] --> MASK{"按模态选掩码"}
  MASK -->|"文本：只看过去"| CE["下一词交叉熵"]
  MASK -->|"图像段内：双向"| DIFF["扩散噪声预测"]
  CE --> SUM["总损失 = CE + λ·扩散"]
  DIFF --> SUM
  SUM --> LAMBDA{"λ 拨向哪边"}
  LAMBDA -->|"过大"| B1["语言能力被噪声梯度带偏"]
  LAMBDA -->|"过小"| B2["文生图不收敛"]
```

共享层让文本条件直接活在每层键值里，不必另接交叉注意力适配器。扩散损失给连续槽稠密梯度，CE 给离散槽。$\lambda$ 决定谁主导：过大则语言模型被图像噪声带偏，过小则文生图不收敛。这是统一训练的主旋钮，不是架构点缀。

## 边界

多步扩散仍贵。下一课把步数压下去，不改「两种损失可以共存」。

## 小结

- Transfusion 在同一 Transformer 里混合 CE 与扩散。
- 图像走连续潜补丁，文本走词表。
- 注意力掩码必须按模态分因果/双向。
- 出处：Zhou 等 Transfusion。

---
title: LLaMA 架构剖析：RMSNorm、RoPE 与 SwiGLU 的"三件套"
date: 2026-08-07
---

# LLaMA 架构剖析：RMSNorm、RoPE 与 SwiGLU 的"三件套"

<div class="epigraph">
<p>一个架构的三个选择，定义了开源大模型的谱系。</p>
<footer>—— 架构分析谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ LLaMA 技术报告（Touvron et al., 2023） ｜ 2026-08-07</p>
</div>

## 为什么 LLaMA 是「开源分水岭」

LLaMA（Meta, 2023）发布后，开源大模型进入新纪元——几乎所有后续开源模型（Mistral、Qwen、Gemma、Yi、Baichuan……）都是「LLaMA 架构的变体」。它没有发明全新结构，而是把三个已有的组件——**RMSNorm、RoPE、SwiGLU**——组合成一套「黄金三件套」，并证明了这个组合在规模化下的效果。**理解 LLaMA 的架构选择，就是理解整个开源大模型谱系的 DNA**。<span class="marginnote">LLaMA 的技术报告只有 15 页，但影响深远。它的架构不是「创新」而是「集大成」：RMSNorm 来自 Zhang & Sennrich 2019、RoPE 来自 Su et al. 2021、SwiGLU 来自 Shazeer 2020——LLaMA 的贡献在于「把它们用对、用在大规模上」。这提示了一个规律：<strong>大模型架构的成功常常不是「发明」，而是「正确地组合」</strong>。</span>

## 1 三件套之一：RMSNorm（归一化）

LLaMA 用 **RMSNorm** 替代 LayerNorm（我们在第五篇讲过它的细节）。

- 去掉均值平移，只做「除均方根」的缩放。
- 参数只剩 $\gamma$（无 $\beta$）。
- 计算更轻、训练更稳。

**为什么 LLaMA 选它**：RMSNorm 在**省算力**与**稳训练**上双赢——对大模型「每一层都归一化」的成本，RMSNorm 的轻量是实打实的收益。而且 LLaMA 在 7B→65B 放大时发现 Pre-LN + RMSNorm 的组合「稳得不需要特殊初始化」。

**位置**：Pre-LN（归一化在残差分支内）。

## 2 三件套之二：RoPE（位置编码）

LLaMA 用 **RoPE**（旋转位置编码）替代「可学习位置嵌入」。

- 把位置信息通过「旋转 Q/K」注入——相对位置语义、无额外参数。
- `rope_theta=10000`（频率基数），支持 `rope_scaling`（长度扩展）。

**为什么 LLaMA 选它**：RoPE 的**外推/插值能力**让 LLaMA 能「训练短、用长」——配合 `rope_scaling` 可以把 4k 训练模型扩展到 32k 推理。同时它不需要位置嵌入参数，节省了 `block_size × d` 的参数量。

## 3 三件套之三：SwiGLU（激活函数）

LLaMA 的 FFN 用 **SwiGLU** 替代「GELU + 单分支」：

$$
\text{FFN}_{\text{SwiGLU}}(x) = (xW_1 \odot \text{swish}(xW_g)) W_2
$$

**关键细节——中间维度补偿**：SwiGLU 有三个权重矩阵（内容、门、输出），参数比普通 FFN 多 50%。LLaMA 因此把中间维度从「标准 4d」降到「$\frac{2}{3} \cdot 4d$」：

$$
d_{\text{ff}}^{\text{LLaMA}} = \frac{2}{3} \times 4 \times d_{\text{model}} = \frac{8}{3} d_{\text{model}}
$$

- LLaMA-7B：$d=4096$，$d_{\text{ff}} = \frac{8}{3} \times 4096 \approx 11008$。
- **参数总量与「GELU + 4d」持平**，但 SwiGLU 的门控表达更强——**同等参数，效果更好**。<span class="marginnote">这个「$\frac{2}{3}$ 补偿」是 LLaMA 架构里最容易被忽略的细节：它让 SwiGLU 的「参数变多」被「维度缩小」抵消，从而公平地对比「SwiGLU vs GELU」——实验结论是 SwiGLU 胜出。读 LLaMA 系模型的 config.json 时，`intermediate_size=11008`（而非 16384）就是这个补偿的直接证据。</span>

## 4 公式解析：LLaMA 一层的完整计算

把三件套组装进一个 LLaMA 解码层：

$$
x' = x + \text{Attn}\left(\text{RMSNorm}(x)\right), \qquad
x'' = x' + \text{SwiGLU-FFN}\left(\text{RMSNorm}(x')\right)
$$

其中注意力内部：

$$
q = \text{Rotate}\left(xW^Q, pos\right), \quad k = \text{Rotate}\left(xW^K, pos\right), \quad
\text{Attn} = \text{softmax}\left(\frac{q k^{\top}}{\sqrt{d_k}}\right) v
$$

对这套式子做四步拆解：

- **第一步，读懂 Pre-LN 顺序**：`RMSNorm(x)` 在注意力/FFN **之前**，残差 `x + ...` 在之后——**Pre-LN**。归一化保持输入稳定，残差保持梯度畅通。
- **第二步，读懂 RoPE 的注入点**：Q/K 投影后、注意力分数前，做**旋转**——位置信息在「匹配」时生效，不进入「内容」。
- **第三步，读懂 SwiGLU 的三投影**：`xW^Q` 与 `xW^g` 逐元素相乘（门控）再投影——**比 GELU 多一个「门」分支**，表达更强。
- **第四步，读出整体**：整个 LLaMA 层 = **归一化（RMSNorm）+ 关系（RoPE 注意力）+ 内容（SwiGLU FFN）** 的 Pre-LN 循环——三件套各司其职。

**辨析｜易错点：** LLaMA 的 **`head_dim`** 可能不等于 `n_embd / n_head`——在 LLaMA 3.2 及以后（及某些变体）里 `head_dim` 被独立指定（如 128），与「n_embd/n_head」解耦。读 config 时别假设「head_dim = hidden_size / num_attention_heads」——**现代实现允许它们不同**。

## 5 LLaMA 的其他架构细节

- **GQA**：LLaMA-2 70B 起用 GQA（8 个 KV 头），大幅压缩 KV Cache。
- **无偏置**：attention 的 QKV 投影与 FFN 均无 bias（`bias=False`）——省参数、省计算，且与 RMSNorm 配合良好。
- **输入输出头不共享**：`tie_word_embeddings=false`——LLaMA 不共享嵌入与输出头（与 GPT-2 不同），参数量更大但「输出侧」有独立表示。
- **上下文长度**：LLaMA-1 是 2048，LLaMA-2 是 4096，LLaMA-3 是 8192——逐步扩展。
- **tokenizer**：SentencePiece + BPE（byte-level），词表 32000。

## 6 小结

- LLaMA 的「三件套」：**RMSNorm（归一化）+ RoPE（位置）+ SwiGLU（激活）**——都不是 LLaMA 发明，但被它「用对」。
- **SwiGLU 的 $\frac{2}{3}$ 补偿**：`intermediate_size = 8/3 · d`，让「门控」不吃参数预算。
- 架构形态：**Pre-LN + RoPE 每层注入 + 无偏置 + GQA**。
- LLaMA 是开源谱系的「基准 DNA」——Mistral、Qwen 等都是它的变体。
- 读 config.json 要留意：`head_dim` 可能独立、`intermediate_size` 是补偿后的值。

在下一节，我们顺着谱系看中国模型的代表——**Qwen 架构演进**：从 Qwen1 到 Qwen3 的设计变化。

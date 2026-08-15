---
title: MQA（多查询注意力）：共享 K/V 的极致压缩
date: 2026-08-07
---

# MQA（多查询注意力）：共享 K/V 的极致压缩

<div class="epigraph">
<p>让许多人共用一盏灯，比每人各点一盏省得多。</p>
<footer>—— 推理工程谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Shazeer 2019《Fast Transformer Decoding》 ｜ 2026-08-07</p>
</div>

## 为什么"多个 query 头共享一份 K/V"能成立

上一节我们算出 KV Cache 随头数 $H$ 线性增长。MQA（Multi-Query Attention）的想法非常大胆：**把所有 query 头共享同一份 K/V**——$H$ 个 query 头，只用 1 个 K 头与 1 个 V 头。KV Cache 瞬间缩小 $H$ 倍（典型 32 倍），而质量损失在当年（2019）的实验中几乎可以忽略。<span class="marginnote">MQA 出自 Shazeer（Transformer 作者之一）2019 年的论文《Fast Transformer Decoding: One Head Is Being a Few Heads》——标题已经说破：对「解码」来说，一个 K/V 头就够了。它的动机纯工程：加速自回归解码，因为解码是「每步只读一次 K/V」，KV 越小、内存带宽压力越小。</span>

## 1 结构：H 个 Q，1 个 K，1 个 V

标准 MHA：$H$ 个 query 头、$H$ 个 key 头、$H$ 个 value 头，各头独立。

MQA：**只有 Q 保持多头，K/V 各只剩 1 个头**。

$$
\text{MQA}(Q, K, V): \qquad Q \in \mathbb{R}^{H \times d_k}, \quad K, V \in \mathbb{R}^{1 \times d_k}
$$

前向时，那唯一的 K/V 头被**广播（broadcast）**到所有 $H$ 个 query 头上：

- 第 $h$ 个 query 头与**同一个 K** 算分数、与**同一个 V** 加权。
- 各头的内容不同（Q 不同），但「被查询的历史」（K/V）相同。

**参数量变化**：K/V 投影从 $2H d_k \cdot d$ 降到 $2 \cdot d_k \cdot d$——**K/V 的权重少了一个大维度**。但更关键的是 KV Cache：每 token 的 KV 从 $2 \cdot H \cdot d_k$ 降到 $2 \cdot d_k$，缩小 $H$ 倍。<span class="marginnote">可以这样理解：MQA 假设「不同头需要查询的『历史内容』本质相同，只是『查询的角度』不同」。既然要查的历史一样，就共享同一份 K/V，省下重复存储。这个假设在多数任务上近似成立——于是 $H$ 倍的显存与带宽被省下来。</span>

## 2 为什么它加速：解码是"内存带宽受限"的

解码（自回归生成）的每一步，核心操作是：读 KV Cache（全部历史），和当前 Q 做注意力。**KV Cache 越大，每步从显存搬进 GPU 的数据越多**——而计算量却很小（每步只算一个 token 的注意力）。

这就形成了**内存带宽受限（memory-bound）**：瓶颈不是「算不过来」，而是「数据搬不完」。MQA 把 KV Cache 缩小 $H$ 倍，每步搬运的数据量直接除以 $H$——**解码速度成倍提升**。

量化对比（LLaMA-7B 级，$H=32$，$d_k=128$，$L=4096$）：

| 方案 | 每 token KV 元素 | KV Cache（FP16） | 相对解码带宽 |
| --- | --- | --- | --- |
| MHA | $2 \times 32 \times 128$ | ~2.1 GB | 1×（基线） |
| MQA | $2 \times 1 \times 128$ | ~66 MB | ~32× 更省 |

从 2.1 GB 降到 66 MB——**KV Cache 缩小 32 倍**，内存带宽压力大幅缓解。<span class="marginnote">这正是「计算 vs 内存」的经典分野：训练是计算密集（矩阵乘大），解码是带宽密集（搬 KV）。MQA 精准打击解码的瓶颈——带宽。这也是为什么「推理优化」与「架构压缩」总是围绕 KV 打转。</span>

## 3 质量损失：为什么"几乎没有"

直觉上共享 K/V 应该损失信息——不同头看「不同关系」，怎么会共享同样的历史？MQA 的实验结论是：**在多数任务上，质量损失很小甚至可忽略**。原因有二：

**Q 才是「多视角」的主力**：头的多样性主要来自 query 投影（$W^Q$ 不同），每个头「问不同的问题」。K/V 是「被问的内容」，内容的多样性远没有问题的多样性重要。
**K/V 共享是「低秩化」的极限**：把多个 K/V 头压缩成一个，等价于把 K/V 空间「降维」到 $d_k$——而 $d_k$ 本身已经够大，信息损失有限。<span class="marginnote">不过 MQA 也非全无损失：在需要「多视角检索」的任务（复杂推理、长上下文综合）上，单一 K/V 头可能成为瓶颈。这正是 GQA 出现的原因——GQA 是「折中」：共享不是 32→1，而是 32→8，保留一点多视角，又省下大部分缓存。下一节细讲。</span>

## 4 公式解析：MQA 的计算与参数

设 $W^Q \in \mathbb{R}^{d \times H d_k}$、$W^K, W^V \in \mathbb{R}^{d \times d_k}$，MQA 前向：

$$
\text{MQA}(x) = \text{Concat}_{h=1}^{H} \left[ \text{softmax}\left(\frac{(xW^Q)_h \cdot (xW^K)^{\top}}{\sqrt{d_k}}\right) (xW^V) \right] W^O
$$

对这条式子做四步拆解：

- **第一步，读懂 $(xW^Q)_h$**：第 $h$ 个 query 头，形状 $(xW^Q)_h$。
- **第二步，读懂共享的 $(xW^K)$ 与 $(xW^V)$**：它们不按下标 $h$ 区分——**所有头用的是同一个 K 和 V**。这是与 MHA 公式唯一的差别（MHA 里是 $(xW^K)_h$）。
- **第三步，读出参数量**：$W^K, W^V$ 从「每头一个」变成「全局一个」：$2Hd_k d \to 2d_k d$，K/V 投影参数缩小 $H$ 倍。
- **第四步，读出缓存量**：KV Cache 每 token 从 $2Hd_k$ 降到 $2d_k$——**这就是 MQA 的全部收益来源**：参数省得有限，缓存省得巨大。

**辨析｜易错点：** MQA 不是「把注意力头数减成 1」。**Q 还是 $H$ 个多头**，只是 K/V 变成单头。注意力计算仍是 $H$ 个头的并行打分（用同一份 K/V），不是「单头注意力」。区别 Q 的多与 K/V 的多——这是最容易混的点。

## 5 MQA 的落点与演进

- **FAST Transformer Decoding**：MQA 在 PaLM、多语言模型（如 Flan-PaLM）中被采用。
- **工程集成**：HuggingFace 的 `num_key_value_heads` 或 `n_kv_heads`（LLaMA 系配置）设为 1 即 MQA。
- **演进**：MQA 太激进 → **GQA**（折中）成为 LLaMA-2/3 的标配 → **MLA**（DeepSeek）用低秩压缩做更精细的折中。MQA 是这条「KV 压缩光谱」的端点之一。

一张「KV 压缩光谱」：

| 方案 | K/V 头数 | KV 相对大小 | 质量 | 代表 |
| --- | --- | --- | --- | --- |
| MHA | $H$ | 1× | 基线 | 原版 Transformer |
| GQA | $g$（1<g<H） | $g/H$× | 近基线 | LLaMA-2/3 |
| MQA | 1 | $1/H$× | 略降 | PaLM |
| MLA | 低秩潜在 | $\approx 1/(H\cdot d)$× | 近基线 | DeepSeek-V2 |

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| MQA | multi-query attention | 多 Q 头共享单 K/V 头 |
| 广播 | broadcast | 单 K/V 分发到所有 Q 头 |
| 内存带宽受限 | memory-bound | 解码瓶颈在搬数据 |
| KV Cache | key-value cache | 推理时缓存的 K/V |
| 多视角检索 | multi-view retrieval | 不同头查询不同信息 |
| n_kv_heads | num key-value heads | HF 配置中 KV 头数 |

## 7 数值算例：解码每步的搬运量

设 $H=32$、$d_k=128$、32 层、FP16，比较 decode 阶段每步从显存搬出的 KV：

- **MHA**：每 token 每层 $2 \times 32 \times 128 \times 2$ 字节 = 16 KB；32 层 → 每步 **512 KB**。
- **MQA**：每 token 每层 $2 \times 128 \times 2$ 字节 = 512 B；32 层 → 每步 **16 KB**。

**读这张表**：decode 每步搬运量差 32 倍。解码是「每步都搬一次全部历史」，这个 32 倍直接换算成「吞吐」或「延迟」——**MQA 不是减少计算，而是减少「每步要搬的数据」，精准击中 decode 的瓶颈**。这也是「KV 压缩」在推理优化里如此重要的原因。

**辨析｜易错点：** MQA 省的是「带宽」，不省「FLOPs」——注意力分数的计算量（Q 与 K 的点积）不变，因为 Q 仍是 $H$ 个头。**「带宽」与「算力」是两种瓶颈，MQA 解决前者**。训练阶段（算力密集）用 MQA 收益有限，这也是为什么它主要被用于「解码优化」的场景。

## 8 训练与推理：MQA 的收益在哪里

| 阶段 | 瓶颈 | MQA 的收益 |
| --- | --- | --- |
| 训练 | 计算密集（大矩阵乘） | 小 |
| 推理 prefill | 计算 + 访存 | 中等 |
| 推理 decode | 带宽密集 | 大（主战场） |
| 长上下文推理 | KV 显存 + 带宽 | 大 |

**结论**：MQA（及其后继 GQA/MLA）本质是「面向推理的架构优化」。训练阶段它们只是「少些 K/V 参数」；推理阶段才真正兑现「KV 显存 + 带宽」的双重收益。**评估任何 KV 压缩方案，都要放在「推理场景」里看价值**。

## 9 小结

- MQA：**H 个 Q 头，共享 1 个 K/V 头**——KV Cache 缩小 $H$ 倍。
- 解码是**内存带宽受限**的，缩小 KV 直接加速解码。
- 质量损失小：头多样性主要来自 Q，K/V 内容共享损失有限。
- 与 MHA 的唯一公式差别：K/V 不再按下标 $h$ 区分。
- MQA 是 KV 压缩光谱的「极致端点」，GQA、MLA 都是它的折中后继。

在下一节，我们看那个更实用的折中——**GQA（分组查询注意力）**：把 $H$ 个头分进 $g$ 组，每组共享一份 K/V。

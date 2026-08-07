---
title: Qwen 架构演进：从 Qwen1 到 Qwen3 的设计变化
date: 2026-08-07
---

# Qwen 架构演进：从 Qwen1 到 Qwen3 的设计变化

<div class="epigraph">
<p>看一个模型的演进，就是看一个团队对「最优解」的反复修正。</p>
<footer>—— 架构分析谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ Qwen 技术报告（1/1.5/2/2.5/3） ｜ 2026-08-07</p>
</div>

## 为什么 Qwen 是「LLaMA 之上的中国演化」

Qwen（通义千问，阿里）是开源大模型里与 LLaMA 谱系并行的重要分支。它的演化史展示了「LLaMA 三件套」如何被本土化改进：**更大的词表服务中文、更强的门控与注意力、更长的上下文、更细的分层蒸馏**。从 Qwen1 到 Qwen3，每一代都在「工程细节」上做加法——读 Qwen 的演进，能看清「开源架构的下一步该往哪改」。<span class="marginnote">Qwen 系列的技术报告非常「工程化」：它们不像 LLaMA 那样「只公布结论」，而是详细记录「改了什么、为什么改、效果如何」。这使它成为学习「架构迭代方法论」的绝佳样本——<strong>每一代 Qwen 都是对上一代「已知短板」的系统修复</strong>。</span>

## 1 Qwen1：中文友好的「LLaMA 加强版」

Qwen1（2023）的架构基本是 LLaMA 三件套，但做了两个关键的本土化：

**① 更大的词表（151,936）**：为了中文，词表从 LLaMA 的 32k 扩到 151k。更大的词表意味着**中文高频词能整词入表**，中文 token 效率显著提升——这是「分词税」的直接缓解（第二篇讲过）。

**② 更长的上下文（初始 2048）**：Qwen1 的上下文不长，但它的 tokenizer 在「中文效率」上的优势从一开始就确立了。

**架构要点**（Qwen1-7B）：

- 层数 32、宽度 4096、32 头、SwiGLU（`intermediate_size=22016`）。
- **QKV bias**：与 LLaMA 不同，Qwen 保留了 bias。
- **RMSNorm + RoPE + SwiGLU**：三件套不变。

## 2 Qwen1.5：GQA 与「标准化」

Qwen1.5（2024）是一次「架构现代化」：

- **引入 GQA**：从 MHA 改为分组查询注意力（`num_key_value_heads=8`）——KV Cache 压缩，推理加速。
- **统一配置**：提供 0.5B→72B 的完整谱系，参数配置系统化。
- **上下文提升**：支持 32k。

**意义**：Qwen1.5 把「LLaMA-2 时代的 GQA」学到手，同时保持了「大词表 + bias」的自身特色——**它是 Qwen 从「追随」走向「自成体系」的转折点**。

## 3 Qwen2：MoE 与「超大上下文」

Qwen2（2024）是架构创新的高峰：

- **Qwen2-57B-A14B**：**MoE 版本**——总参数 57B，激活 14B（4 个专家路由 + 共享专家，细粒度切分，DeepSeekMoE 风格）。
- **更长上下文**：Qwen2 系列支持 32k–128k。
- **更强的注意力细节**：**QK-Norm** 引入——在 Q/K 上做 RMSNorm，稳定注意力分数（第五篇的进阶技巧）。
- **RoPE 配置灵活**：`rope_theta`、`rope_scaling` 成为标准配置项，支持长度扩展。

**关键设计**（Qwen2-7B）：

- `num_attention_heads=28`、`num_key_value_heads=4`（GQA-4）。
- **`head_dim` 独立为 128**——不再等于 `hidden/n_heads`。
- SwiGLU + RMSNorm + Pre-LN + QK-Norm——「LLaMA 三件套 + QK-Norm」的完整形态。

## 4 Qwen2.5：深挖「细粒度」与「稀疏」

Qwen2.5（2024）延续 Qwen2 并强化：

- **更大规模的 MoE**：Qwen2.5-Max（闭源）与开源 MoE 变体把「细粒度专家」用得更极致。
- **强化推理与编码**：架构细节上继续打磨（激活函数、归一化位置）。
- **注意力 logits 软裁剪**：部分模型引入 logit capping（第五篇讲的软裁剪）——限制 logits 幅度，稳定长上下文生成。

## 5 Qwen3：混合推理与「原生长上下文」

Qwen3（2025）带来两个结构性变化：

**① 混合推理模式（thinking + non-thinking）**：Qwen3 模型显式支持「思考模式」（长思维链，CoT）与「非思考模式」（直接回答）切换——这是对 o1 范式（第十三篇）的架构级响应，通过特殊的思考 token 与训练数据实现，而非纯 prompt 技巧。

**② 更长的原生上下文**：Qwen3 旗舰支持 128k+ 原生上下文，配合 YaRN 插值可到 256k。

**架构演进总结表**：

| 版本 | 关键变化 | 上下文 | 代表作 |
| --- | --- | --- | --- |
| Qwen1 | 大词表 151k、中文友好 | 2k | Qwen-7B |
| Qwen1.5 | GQA、统一谱系 | 32k | Qwen1.5-72B |
| Qwen2 | MoE、QK-Norm、超长上下文 | 32k-128k | Qwen2-57B-A14B |
| Qwen2.5 | 细粒度 MoE、logit capping | 32k-128k | Qwen2.5-72B |
| Qwen3 | 混合推理、原生长上下文 | 128k+ | Qwen3-235B-A22B |

## 6 公式解析：Qwen 的「head_dim 解耦」

Qwen2 起的一个关键配置变化——`head_dim` 不再由「宽度 ÷ 头数」自动决定。设 `hidden_size=3584`、`num_attention_heads=28`，若按旧规则 `head_dim = 3584/28 = 128`；但 Qwen 直接把 `head_dim=128` **写死在 config**，且允许它独立变化。

对这条式子做三步拆解：

- **第一步，读懂旧规则**：传统上 `head_dim = hidden_size / n_heads`——宽度被头数整除。这约束了「宽度与头数」的耦合。
- **第二步，读懂解耦**：Qwen 把 `head_dim` 独立出来——**「每头多大」与「有几个头」可以分别设计**。这让「更小的 hidden + 更大的 head_dim」成为可能（如 hidden=896、n_heads=8、head_dim=128 的混合）。
- **第三步，读出收益**：解耦让「计算效率」与「表达容量」独立调优——`head_dim` 控制「每头的容量」，`n_heads` 控制「并行视角数」。**现代模型（Qwen、LLaMA-3.2）都开始显式指定 head_dim**——读 config 时别再用除法猜测。

**辨析｜易错点：** Qwen 与 LLaMA 的 `intermediate_size` 都不是「4×hidden」——它们被 SwiGLU 的 2/3 补偿调整过。Qwen2-7B 的 `intermediate_size=18944`（≈ 2.7×hidden），这是「8/3 × d」的变体。**读任何现代模型的 config，都不能假设「FFN 维度 = 4×hidden」**。

## 7 小结

- Qwen 是 LLaMA 谱系之上的**中国化演化**：大词表（151k）服务中文。
- 演进主线：**Qwen1（词表）→ 1.5（GQA）→ Qwen2（MoE + QK-Norm）→ 2.5（细粒度）→ Qwen3（混合推理）**。
- 关键创新：**head_dim 解耦、QK-Norm、MoE 细粒度、思考模式**。
- 工程方法论：**每代都是对已知短板的系统修复**——「工程细节」是 Qwen 的护城河。
- 读 config 的教训：**别假设 head_dim 与 FFN 维度的「默认关系」**。

在下一节，我们看 Qwen 之外的另一条技术路线——**DeepSeek 架构演进**：MLA、DeepSeekMoE 与 V3 的系统级创新。

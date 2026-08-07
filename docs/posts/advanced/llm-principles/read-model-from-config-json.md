---
title: 从 config.json 读懂一个模型：超参背后的架构选择
date: 2026-08-07
---

# 从 config.json 读懂一个模型：超参背后的架构选择

<div class="epigraph">
<p>一张 JSON，就是模型架构的「完整族谱」。</p>
<footer>—— 架构分析谚语（化用）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ HuggingFace 模型配置规范 ｜ 2026-08-07</p>
</div>

## 为什么 config.json 值得逐字段读

每个 HuggingFace 模型都有一个 `config.json`——几十个字段定义了模型的全部架构。**会读 config.json，就能在几秒内「看出」一个模型的架构设计**：它有多大、用什么位置编码、有没有 MoE、KV 压不压缩。这是「架构分析」的核心技能，也是「模型选型」的第一手资料。<span class="marginnote">config.json 是「模型与代码之间的契约」：`modeling_xxx.py` 里的每个分支都由 config 字段驱动。读 config 不是「读参数表」，而是「读设计决策」——<strong>每个字段都对应一个我们在前几篇讲过的架构选择</strong>。</span>

## 1 一张真实的 config.json（Qwen2-7B 精简版）

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "vocab_size": 152064,
  "max_position_embeddings": 131072,
  "rope_theta": 1000000.0,
  "rope_scaling": {"type": "yarn", "factor": 4.0},
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": false,
  "use_cache": true,
  "attention_dropout": 0.0,
  "initializer_range": 0.02
}
```

## 2 核心字段逐项拆解

**结构类**：

- `hidden_size: 3584`：宽度 $d$。注意**不是**「round 数」（如 4096）——Qwen 用非标准宽度。这直接影响「参数与 FLOPs」的估算。
- `num_hidden_layers: 28`：层数。
- `intermediate_size: 18944`：FFN 中间维度。$18944 / 3584 \approx 5.28$——不是 4！这是 SwiGLU 的「8/3·d 变体」调整后的值（前面讲过「别假设 4×d」）。
- `vocab_size: 152064`：词表大小——Qwen 的大词表（中文友好）。

**注意力类**：

- `num_attention_heads: 28`、`num_key_value_heads: 4`：**GQA-4**（28 个 Q 头共享 4 组 KV）——KV Cache 缩到 MHA 的 1/7。
- `head_dim: 128`：**显式指定**——不等于 `hidden_size / num_attention_heads`（$3584/28 = 128$，这里恰好相等，但很多模型不相等）。
- `attention_dropout: 0.0`：注意力 dropout 关闭——大模型常用 0（省算力，靠数据正则）。<span class="marginnote">读 GQA 的字段：`num_key_value_heads < num_attention_heads` 说明用了 GQA；`= 1` 是 MQA；`= num_attention_heads` 是 MHA。Qwen2-7B 的 `4 < 28` 一眼看出「GQA-4」。这是读 config 判断「KV 压不压缩」的最快方法。</span>

**位置与归一化类**：

- `rope_theta: 1000000`：RoPE 频率基数。Qwen 用 100 万（LLaMA 是 1 万）——**更大的 theta 降低频率，是「长上下文」的设计**。
- `rope_scaling: {"type": "yarn", "factor": 4.0}`：**YaRN 插值、因子 4**——把训练长度扩展 4 倍。
- `rms_norm_eps: 1e-6`：RMSNorm 的 epsilon——防除零的小常数，不同模型取值略异（LLaMA 是 1e-5）。

**激活与共享类**：

- `hidden_act: "silu"`：激活是 SiLU——**SwiGLU 的门激活**（内容分支用线性、门分支用 SiLU）。如果这里显示 `"gelu"`，说明用 GeGLU。
- `tie_word_embeddings: false`：**输入输出头不共享**——LLaMA/Qwen 系「不共享」；Gemma/GPT-2「共享」。

## 3 公式解析：从 config 反推参数量

拿到 config 就能「反推」参数量。对 Qwen2-7B：

$$
N = \underbrace{28 \cdot \left(4 \cdot 3584^2 + 3 \cdot 3584 \cdot 18944\right)}_{\text{层：注意力+FFN}} + \underbrace{152064 \cdot 3584}_{\text{嵌入}}
$$

对这条式子做三步拆解：

- **第一步，读懂注意力项**：$4d^2 = 4 \times 3584^2$——QKV 投影 $3d^2$ + 输出投影 $d^2$。
- **第二步，读懂 FFN 项**：SwiGLU 是「三个投影」：内容 $d \times d_{\text{ff}}$ + 门 $d \times d_{\text{ff}}$ + 输出 $d_{\text{ff}} \times d$ = $3 \cdot d \cdot d_{\text{ff}} = 3 \times 3584 \times 18944$。
- **第三步，读出结果**：$28 \times (51.4M + 203.6M) + 545M \approx 7.7B$——约 7B 量级。**手算反推与官方参数量级吻合**，验证了 config 的「可读性」。

**注意**：还要减去「共享部分」（这里不共享所以不用减），加上「norm/attention bias」等小项——精确值以 `num_parameters` 字段为准，但量级估算靠这些公式就够。

**辨析｜易错点：** `intermediate_size` 与 `hidden_size` 的关系**不固定**——不同模型用不同系数（4、8/3、甚至自定义）。判断 FFN 类型要靠 `hidden_act`（silu→SwiGLU，gelu→GeGLU）而不是猜「中间维度 = 4×宽度」。**读 config 要看「字段组合」而非「单个字段」**。

## 4 一份「读 config 清单」

拿到任何模型，按这个顺序「快读」：

1. **`architectures` / `model_type`**：是什么模型家族（LLaMA、Qwen、Mistral…）。
2. **`hidden_size` × `num_hidden_layers`**：模型「多大」（参数的量级）。
3. **`num_attention_heads` vs `num_key_value_heads`**：MHA / GQA / MQA。
4. **`head_dim`**：是否显式指定（现代模型的趋势）。
5. **`rope_theta` + `rope_scaling`**：位置编码配置 + 长度扩展方式。
6. **`intermediate_size` + `hidden_act`**：FFN 形态（SwiGLU/GeGLU）。
7. **`tie_word_embeddings`**：嵌入是否共享。
8. **`vocab_size`**：词表大小（多语言/中文）。
9. **`max_position_embeddings`**：原生上下文长度。

**「看一眼就能说的结论」示例**：`num_key_value_heads=4 < 28` + `head_dim=128` + `rope_scaling=yarn` + `intermediate_size=18944` → 「GQA-4 + head_dim 解耦 + YaRN 长上下文 + SwiGLU」——**架构画像 5 秒成型**。

## 5 config 之外的「配套文件」

- **`tokenizer_config.json`**：tokenizer 类型（BPE/Unigram）、特殊 token、`rope_scaling` 的配套。
- **`generation_config.json`**：默认解码参数（温度、top-p、max_new_tokens）。
- **`model.safetensors.index.json`**：权重分片信息（多少片、各片多大）。
- **`tokenizer.json`**：词表 + 合并规则（BPE 的完整形态）。

这些文件合起来才是「完整模型规格」——config 是「架构」，tokenizer_config 是「表示」，generation_config 是「行为」。

## 6 小结

- **config.json 是「架构契约」**：每个字段对应一个设计决策。
- 核心字段：**hidden/layers（大小）、KV 头数（GQA）、head_dim（解耦）、rope（位置）、intermediate+act（FFN）、tie（共享）**。
- 反推参数量：$N = L(4d^2 + 3d \cdot d_{\text{ff}}) + Vd$——**量级一眼看穿**。
- 读法要「看组合」：`silu` → SwiGLU、`num_kv < num_q` → GQA、`yarn` → 插值。
- 配套文件：tokenizer_config（表示）、generation_config（行为）。

第四篇《主流开源架构对比》全部完成。下一篇进入**归一化与激活函数**——等等，第五篇我们已经写完了（LayerNorm、RMSNorm、Pre-LN、激活、进阶技巧）。让我们继续下一篇未完成的——实际已全部完成。进入**第六篇 位置编码**，也已全部完成。接下来按规划进入**第十一篇 长上下文**。

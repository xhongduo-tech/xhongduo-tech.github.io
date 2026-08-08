---
title: 离散语音单元（Speech Token）与语音语言模型
date: 2026-08-07
---

# 离散语音单元（Speech Token）与语音语言模型

<div class="epigraph">
<p>让声音和文字说同一种语言，机器才能真正地听与说。</p>
<footer>—— 语音语言模型的统一理想</footer>
</div>

<div class="article-byline">
<p>第四级 · 语音技术 ｜ 《Spoken Language Processing》第18章 自监督语音表征 ｜ 2026-08-07</p>
</div>

## 为什么从 Speech Token 开始

AudioLM 证明「音频 = token 序列」能生成，但它的 token 是「音频专用的」，与文本 token 还是两套系统。**语音语言模型（Speech Language Model）** 走得更远：**把语音 token 与文本 token 放进同一个序列、同一个 Transformer**——模型既可以「听」（音频 token → 语义），也可以「说」（语义 → 音频 token），还可以「想」（在文本 token 层面推理）。这就是 SpeechGPT、VALL-E、Qwen-Audio、GPT-4o 语音背后的统一范式。

**离散语音单元（Speech Token）** 是这个范式的「货币」：一段语音被编解码器切成 token 流，语言模型像处理词一样处理它们。理解 Speech Token 与语音 LM 的架构，就理解了「为什么语音大模型能端到端听说」——以及它的技术代价（序列长、声学细节 vs 语义、多任务平衡）。

## 1 Speech Token：语音的「离散字母表」

**Speech Token**：把连续音频映射成离散编号序列的单元，由神经编解码器（SoundStream/EnCodec）或聚类（HuBERT）产出。按「信息层次」分两类，几乎所有语音 LM 都用它们的组合：

- **语义 token（semantic / content token）**：低帧率（约 25 Hz）、管内容——近似音素/词。来自 HuBERT 聚类或编解码器浅层。**序列短，适合做「思考」**。
- **声学 token（acoustic / unit token）**：高帧率（约 50–100 Hz）、管音色细节。来自编解码器 RVQ 深层。**承载「渲染」，序列长**。

**核心概念：Speech Token 的价值 = 让音频「进入语言模型的序列」。** 文本 token 编码语义，Speech Token 编码「语义 + 音色 + 韵律」——一旦两者可互换地躺在同一序列里，模型就能做「听、说、想」的任意组合。<span class="marginnote">Speech Token 与「词」的关键差别：<strong>词是离散语义单位（可计数、可组合），Speech Token 是声学单位（带音色、依赖说话人）</strong>。同一句话由不同人说的 token 不同——所以语音 LM 需要「说话人信息」或「把音色独立出来」。这也是「语音 token + 说话人嵌入」常同时出现的原因。</span>

## 2 语音语言模型的架构：单流与双流

把 Speech Token 与文本 Token 组织进一个模型，有两种主流架构：

**单流（unified / single-stream）**：文本 token 与语音 token **在同一序列**里，模型用一套参数处理。任务用**特殊 token** 区分：

$$
[\text`{ <speech> }`]\; \text{音频 token} \; [\text`{ <text> }`]\; \text{文本} \; [\text`{ <answer> }`]\; \text{音频 token}
$$

训练时把「识别、合成、对话」都表达成「序列补全」——**一个模型，一个损失（下一个 token 预测），全部任务**。SpeechGPT 是早期代表：输入音频 → 识别成文本 → 用 LLM 思考 → 合成音频回答，整条链在同一个模型里。

**双流（dual-stream / two-tower）**：文本与语音**各走一条流**，中间用**桥接**对齐。语音编码器把音频转成「语音侧表征」，与文本侧通过交叉注意力/投影相连。优点是文本语义不被声学噪声污染；缺点是结构复杂，跨模态对齐难。

**公式解析：任务 token 的统一表达。**

$$
\mathcal{L} = \sum_{l} -\log P\big( \text{token}_l \mid \text{token}_{\\lt l}, \text{任务 token} \big)
$$

- **第一步，看统一序列**：序列里既有 `<transcribe>` / `<synthesize>`——告诉模型「现在干什么」。
- **单流 vs 双流**：统一序列 vs 文本/语音分离 + 桥接——「简单」与「纯净」的权衡。
- **三大挑战**：序列长度（声学 token 太密）、音色内容纠缠、多任务平衡。
- **语义思考 + 声学渲染**：短序列管「说什么」、长序列管「怎么发声」——分而治之。
- **离散 vs 连续**：离散 token 稳定兼容语言模型，连续表征信息无损——路线之争。
- **级联 vs 端到端**：语音 LM ≠ ASR+LLM+TTS 拼接——一个模型 vs 三个模型。

**延伸思考**：Speech Token 让「语音」进入了语言模型的「母语」——但这有一个深层代价：**离散化丢失了连续声学信息**（音色细节、微韵律）。GPT-4o 语音用「连续表征」部分绕开这个损失，说明「离散 vs 连续」不只是工程选择，而是「语言的抽象 vs 声音的实在」的哲学张力。语音大模型的成熟，或许在于找到「既抽象又保真」的中间表示——这是 Speech Token 研究的下一个十年。

在下一节，我们将看语音 LM 的产品化巅峰：**GPT-4o 语音模式与端到端语音对话架构**——「听懂、思考、说出」如何在一个模型里实时完成。

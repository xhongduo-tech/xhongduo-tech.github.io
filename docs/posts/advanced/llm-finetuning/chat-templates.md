---
title: 对话模板：ChatML、Llama 系列模板与特殊 token 的设计
date: 2026-08-07
---

# 对话模板：ChatML、Llama 系列模板与特殊 token 的设计

<div class="epigraph">
<p>形式不是内容的容器，形式就是内容的一部分。</p>
<footer>—— 引意自马歇尔 · 麦克卢汉（Marshall McLuhan）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第二章 ｜ 2026-08-07</p>
</div>

## 为什么从对话模板开始

上一节的多轮数据，最终要变成一串 token 才能喂给模型。这串 token 长什么样、角色怎么区分、特殊标记放哪里——这套「把对话翻译成 token 序列」的规则，就叫**对话模板（chat template）**。

模板看似只是「加几个尖括号」，实则牵一发动全身：**训练时用什么模板，推理时就必须用同一个模板**。若训练时用 ChatML、推理时却用 Llama 模板，模型会当场「失忆」——它认不出角色标记，分不清哪句是用户哪句是助手，输出立刻崩坏。本节的三个主题——**特殊 token、三大模板家族、模板工程**——是所有微调工作里最容易翻车也最好修的一环。<span class="marginnote">为什么角色标记必须是「特殊 token」而不是普通词？因为普通词会出现在正文里，无法被唯一识别；特殊 token 在词表里独占 id、几乎不会与正文混淆。角色标记的本质，是给模型一个「稳定的、可学习的对话语法符号」。</span>

## 1 特殊 token：模板的原料

模板由三类 token 拼成：**文本 token**（对话内容本身）与**特殊 token（special token）**。特殊 token 有三类角色：

**边界 token**：标记对话/序列的起止，如 `<s>`、（Llama 2 的 BOS/EOS）、`<|begin_of_text|>`（Llama 3）。
**角色 token**：区分说话人，如 `<|im_start|>` 与 `<|im_end|>`（ChatML）、`[INST]`（Llama 2 的指令标记）。
**轮次终止 token**：标记一段话结束，如 `<|im_end|>`（ChatML）、`<|eot_id|>`（Llama 3）、<end_of_turn>`（Gemma）。

设计特殊 token 有几条硬规矩：

1. **必须加入词表并独占 id**：新 token 要扩进 tokenizer 词表，获得唯一 id。若漏加，分词器会把它们拆成一串普通字符，模型学到的「角色标记」语义全部落空。
2. **id 稳定性**：训练、推理、部署全链路必须使用同一套 tokenizer 与词表。任何一处不一致（比如推理时换了不同版本的 tokenizer），id 对不上，输出直接乱。
3. **避免与正文冲突**：特殊 token 文本要选得足够「生僻」，如 `<|im_start|>`，避免用户正文里恰好出现同样的字符串被误判。<span class="marginnote">一个经典的坑：有人用普通文本 `[USER]` 做角色标记，结果用户在正文里写「请解释 `[USER]` 标签」，模型便分不清这到底是标记还是内容。成熟的模板都选用带竖线、双角括号的「不可能自然出现」的组合。</span>

## 2 三大模板家族

市面上的对话模板虽多，主流的就三个家族，学会它们就能读懂几乎所有模型。

### ChatML（OpenAI 系）

ChatML 是 OpenAI 为 GPT 系列设计的模板，也是开源社区用得最广的格式。基本结构：

```
<|im_start|>system
你是助手，请用中文回答。<|im_end|>
<|im_start|>user
请介绍一下你自己。<|im_end|>
<|im_start|>assistant
你好，我是助手。<|im_end|>
```

特点：**每个角色用 `<|im_start|>` 开头、`<|im_end|>` 收尾**，结构规整，加轮次只是重复同一模式。ChatML 的优雅在于「system / user / assistant」三类角色一视同仁地套同一套语法，机器解析极其简单。

### Llama 2 模板

Llama 2 用方括号风格，且把 system 消息直接拼进第一轮用户指令里：

```
<s>[INST] <<SYS>>
你是助手，请用中文回答。
<</SYS>>

请介绍一下你自己。 [/INST] 你好，我是助手。</s>
```

特点：**指令用 `[INST]` 与 `[/INST]` 包裹**，句子间以 `<<SYS>>`、`<</SYS>>` 分隔。没有独立的 system 标记——系统提示词被塞进 `<<SYS>>` 块里。多轮对话时，前一轮的「回答 + 」会原样拼在下一轮的 `[INST]` 前面。

### Llama 3 模板

Llama 3 把 ChatML 的优雅与 Llama 家族的习惯合并，改用「header + eot」结构：

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是助手，请用中文回答。<|eot_id|>
<|start_header_id|>user<|end_header_id|>

请介绍一下你自己。<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

你好，我是助手。<|eot_id|>
```

特点：**`<|start_header_id|>` 声明说话人，`<|eot_id|>` 结束本轮**，`<|begin_of_text|>` 作全局起点。它本质上是 ChatML 的变体，但每个 token 的命名更显式。

### 三者的对照

| 家族 | 角色标记 | 轮次终止 | 系统提示词 | 代表模型 |
| --- | --- | --- | --- | --- |
| ChatML | `<\|im_start\|>` | `<\|im_end\|>` | 独立 system 角色 | GPT、Qwen、DeepSeek |
| Llama 2 | `[INST]` |  | `<SYS>` 块 | Llama 2 |
| Llama 3 | `<\|start_header_id\|>` | `<\|eot_id\|>` | 独立 system 角色 | Llama 3、Mistral-NeMo |

## 3 公式解析：模板是一个确定性序列化函数

不要被各种括号吓到——**模板本质上是一个确定性函数**：输入一段对话，输出一串 token。把它写成数学形式，一切模板都只是同一个公式的特例：

$$
\mathrm{Seq}(C) = \big[\,\mathrm{BOS}\,\big] + \sum_{i=1}^{n} \Big[\;\mathsf{hs}(r_i) + T_i + \mathsf{he}(r_i)\;\Big]
$$

逐项拆解：

- $C = \{(r_1, T_1), \dots, (r_n, T_n)\}$：一段 $n$ 轮对话，每轮由**角色 $r_i$**（system / user / assistant）与**文本 $T_i$** 组成；
- $\mathrm{BOS}$：序列起始 token（可有可无，依模型而定）；
- $\mathsf{hs}(r_i)$：**header start**——根据角色 $r_i$ 生成的开头标记。ChatML 是 `<|im_start|>`，Llama 3 是 `<|start_header_id|>`…`<|end_header_id|>`；
- $T_i$：该轮文本内容（原始内容，不再做任何转义）；
- $\mathsf{he}(r_i)$：**header end / 轮次终止**——该轮的收尾标记，ChatML 是 `<|im_end|>`，Llama 2 是 ；
- $+$ 号：字符串拼接，最终得到一整串 token。

**直觉**：模板设计的工作，就是给 $\mathsf{hs}(\cdot)$ 与 $\mathsf{he}(\cdot)$ 选一套「不会撞车」的特殊 token。无论模板多复杂——多轮、嵌套 system、工具调用——它都是这个「开头标记 + 正文 + 收尾标记」循环的堆叠。理解了这一点，阅读任何模型的 template 文件都不再神秘。<span class="marginnote">这也解释了「模板错位的后果」：推理时若用了另一套 $\mathsf{hs}/\mathsf{he}$，模型见到的 token 序列与训练时完全不同——相当于把一个人放进了陌生的语法环境，角色语义全部丢失。<strong>训练模板与推理模板必须字节级一致</strong>。</span>

## 4 模板的工程细节与常见错误

理论清楚了，工程上还有几个高频翻车点。

**错误一：训练与推理模板不一致。** 最常见的翻车。训练脚本用 `tokenizer.apply_chat_template` 序列化，推理时却手写了另一套拼接逻辑。症状是模型「变笨」——不遵循指令、角色混乱、疯狂重复。修复很简单：**推理一律调用同一个 `apply_chat_template`**，不要手拼字符串。

**错误二：新增特殊 token 后没重训/没同步 embedding。** 给模型追加 `<|im_start|>` 等 token 时，embedding 层会长出一行随机初始化的向量。若随即就用预训练权重推理，这些新 token 的 embedding 是随机值，模型对它们一无所知。**新 token 必须经过训练（哪怕只是微调几步）才能被模型真正认识。**

**错误三：把特殊 token 写死进正文。** 有人图省事，把 `<|im_start|>` 直接写进普通文本当字面量。这会让分词器把它们拆成若干个子词，不仅浪费 token，还让角色标记「名存实亡」。正确做法是用 tokenizer 的 `add_special_tokens` 机制。

**错误四：EOS 与轮次终止混淆。** Llama 3 的 `<|eot_id|>` 是「轮次结束」，`<|end_of_text|>` 才是「整段结束」；生成时若把 eot 当 EOS 触发停止，多轮对话会在第一轮就截断。<span class="marginnote">生成侧还常配 `pad_token_id` 与 `eos_token_id` 的设定：若模板有独立的 pad token（如 `<pad>`），生成参数里的 pad 与 eos 必须显式区分，否则停止条件会误触发。</span>

## 5 小结

- **对话模板是把对话序列化成 token 序列的确定性函数**：「开头标记 + 正文 + 收尾标记」循环堆叠，任何模板都是同一公式的特例。
- 特殊 token 分三类：**边界**（BOS/EOS）、**角色**（区分说话人）、**轮次终止**；必须独占词表 id、全链路一致。
- 三大模板家族：**ChatML**（`<|im_start|>`/`<|im_end|>`，结构最规整）、**Llama 2**（`[INST]`/`[/INST]`）、**Llama 3**（header + `<|eot_id|>`）。
- 最致命的坑是**训练与推理模板不一致**——务必统一走 `apply_chat_template`，不要手拼字符串。
- 新特殊 token 需要**训练才能生效**；EOS 与轮次终止 token 不可混用，生成参数里的 pad/eos 要显式区分。

在下一节，我们钻回损失函数内部，把「只对回答算损失」这句话变成可落地的代码：**loss mask——只对回答部分计算损失的实现细节与常见错误**。

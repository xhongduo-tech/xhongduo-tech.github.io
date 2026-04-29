## 核心结论

先给结论：Decoder-only 与 Encoder-Decoder 的差别，不是“一个能生成，另一个不能生成”，而是“输入和输出是否共用同一条生成链”。

| 架构 | 建模方式 | 适合的任务 | 核心特点 |
|---|---|---|---|
| Decoder-only | $p(z)=\prod_i p(z_i\mid z_{<i})$ | 聊天、补全、代码生成 | 输入输出共用同一生成链 |
| Encoder-Decoder | $p(y\mid x)=\prod_t p(y_t\mid x,y_{<t})$ | 翻译、摘要、改写 | 输入与输出显式分离 |

“生成链”可以先用白话理解成：模型按 token 一个一个往后预测时，信息是沿着哪条序列流动的。Decoder-only 把提示词、上下文、目标答案都放进同一条序列里，像“先看题，再顺着往下写”。Encoder-Decoder 则先把输入读完形成内部表示，再单独生成输出，像“先理解题目，再开始作答”。

因此，架构选择本质上是在四件事之间做取舍：

1. 是否需要统一接口。也就是所有任务都能用“给前缀，然后续写”这一个调用方式完成。
2. 是否需要更直接的条件建模。也就是输出每一步都显式依赖完整输入，而不是依赖“输入已经被拼进前缀”。
3. 训练目标是否天然兼容。也就是预训练、微调、推理能否共用相近的数据组织方式。
4. 部署成本是否可控。也就是显存、延迟、缓存复用、服务接口复杂度是否符合工程现实。

如果任务像“继续写下去”，例如聊天回复、代码补全、长文本续写，Decoder-only 往往更自然。如果任务像“把一个输入稳定映射成另一个输出”，例如机器翻译、新闻摘要、文本改写，Encoder-Decoder 往往更顺手。

这不是“谁更先进”的问题，而是任务形态与工程约束是否匹配的问题。

---

## 问题定义与边界

先把问题写清楚，否则“文本生成”这个词很容易把不同任务混成一类。

设输入序列为 $x=(x_1,\dots,x_n)$，输出序列为 $y=(y_1,\dots,y_m)$。在 Decoder-only 里，常见做法是把两者拼成一个联合序列 $z=[x;y]$。这里“联合序列”就是把输入和输出首尾相接，交给同一个自回归模型处理。

| 符号 | 含义 |
|---|---|
| $x=(x_1,\dots,x_n)$ | 输入序列 |
| $y=(y_1,\dots,y_m)$ | 输出序列 |
| $z=[x;y]$ | Decoder-only 中拼接后的联合序列 |

看两个最常见任务：

1. 翻译任务中，$x$ 是英文句子，$y$ 是中文句子。输入和输出边界很硬，任务就是条件映射。
2. 聊天任务中，系统提示、历史对话、用户问题、工具结果、模型回复都可能被组织成一个长前缀。这里虽然也有“输入”和“输出”，但工程上常被统一成“给定前缀后继续生成”。

所以本文讨论的不是“模型是否会输出文字”，而是：当任务被形式化为输入 $x$ 到输出 $y$ 时，两类架构怎样处理依赖关系、怎样组织训练目标，以及它们分别适合什么任务。

本文边界也要明确：

| 本文回答的问题 | 不展开的问题 |
|---|---|
| 两类架构的目标函数差异 | 参数规模大小谁更强 |
| 注意力掩码如何不同 | RLHF、对齐训练细节 |
| 典型任务如何适配 | 数据清洗与配比策略 |
| 工程实现有哪些关键坑 | 具体厂商模型效果排名 |

这条边界很重要。因为实际模型能力不仅受架构影响，还受数据规模、优化器、训练时长、词表设计、推理策略影响。把所有差异都归结为“架构决定一切”，结论通常会失真。

一个对初学者很有用的判断标准是：先别问“哪类模型更强”，先问“这个任务到底是前缀续写问题，还是条件映射问题”。前一个问题对应使用方式，后一个问题对应任务结构。架构选择主要看后者。

---

## 核心机制与推导

### 1. Decoder-only：把输入和输出放进同一条链

“自回归”先用白话解释：每次只预测下一个 token，并且只能看已经出现过的内容。  
Decoder-only 的训练核心就是语言建模：

$$
p(z)=\prod_{i=1}^{n+m} p(z_i\mid z_{<i})
$$

如果把输入 $x$ 和输出 $y$ 拼起来，得到 $z=[x;y]$，那么做任务微调时，通常只对输出段计算损失：

$$
L_{\text{dec}}=-\sum_{i=n+1}^{n+m}\log p(z_i\mid z_{<i})
$$

这条式子的含义很直接：虽然模型在读整条序列，但优化目标只关心“给定输入前缀后，能不能把输出生成对”。

它的注意力掩码是因果掩码：

$$
M_{ij}=1[j\le i]
$$

“因果掩码”可以先理解成一个可见性规则：第 $i$ 个位置只能看自己和前面的 token，不能偷看未来。

### 2. Encoder-Decoder：先编码输入，再条件生成输出

Encoder-Decoder 把问题写成条件概率：

$$
p(y\mid x)=\prod_{t=1}^{m} p(y_t\mid x,y_{<t})
$$

对应损失函数是：

$$
L_{\text{seq2seq}}=-\sum_{t=1}^{m}\log p(y_t\mid x,y_{<t})
$$

这里的关键不是公式长得不一样，而是依赖结构不同。Decoder 在生成第 $t$ 个 token 时，不只看此前已生成的 $y_{<t}$，还会通过 cross-attention 看完整输入 $x$ 的编码结果。

“cross-attention”可以先理解成：decoder 在写输出时，能随时去查输入的整段表示，而不是只能依靠把输入塞进同一条前缀后留下的隐式记忆。

### 3. 注意力权限差异

| 架构 | self-attention | cross-attention | 可见范围 |
|---|---|---|---|
| Decoder-only | 因果掩码 | 无 | 只能看前文 |
| Encoder-Decoder | decoder 内部是因果掩码 | 有 | decoder 可看完整输入 |

这个表是两类架构任务适配差异的根源。

Decoder-only 的 decoder 只能沿前缀向右看，所以它擅长“前缀驱动的续写”。Encoder-Decoder 的 decoder 则每一步都可以重新查询输入表示，因此在“输出强依赖输入全文”的任务上更直接。

### 4. 玩具例子：两种视角看同一个翻译任务

设输入 $x=$“I eat”，目标输出 $y=$“我 吃”。

如果模型给出：

- $p(\text{"我"}\mid x)=0.6$
- $p(\text{"吃"}\mid x,\text{"我"})=0.9$

那么整句条件概率是：

$$
p(y\mid x)=0.6\times 0.9=0.54
$$

这就是自回归分解：一句话的概率等于每一步条件概率的连乘。

两种架构如何处理它？

1. Decoder-only：把序列写成 `[I, eat, 我, 吃]`，训练时模型依次预测后续 token，但只对“我、吃”这两个位置计损失。
2. Encoder-Decoder：encoder 先读入 `[I, eat]`，decoder 再按顺序生成“我、吃”，每一步都能访问整段输入表示。

这个玩具例子看起来结果相似，但工程语义不同。前者是在同一条文本链上做续写，后者是在明确条件 $x$ 下生成 $y$。

### 5. 训练时与推理时

两类架构在训练时通常都用 teacher forcing。这个术语的白话解释是：训练阶段把真实历史 token 喂给模型，让模型学习预测下一个 token，而不是让它自己滚动生成。

但推理时两者都要逐 token 生成，因此都会面临重复、截断、长度控制等问题。差别在于：

- Decoder-only 推理时的上下文就是整条前缀。
- Encoder-Decoder 推理时的上下文分成两部分：固定输入表示，加上逐步扩展的输出前缀。

因此，选择架构时不能只看“是否都能生成”，而要看“生成时依赖输入的方式是否适合你的任务结构”。

---

## 代码实现

工程上最大的差异，不在模型名字，而在数据组织方式和 loss mask。

### 1. Decoder-only 的 seq2seq 微调

做条件生成时，输入和答案通常拼在一起，例如：

`prompt = "翻译成中文：I eat\n答案："`  
`answer = "我吃"`

模型真正看到的是 `[prompt + answer]`。但如果你直接让所有 token 都参与 loss，模型会同时学习“复述 prompt”与“生成 answer”，目标就错了。正确做法是把 prompt 部分 mask 掉。

```python
def build_labels(input_ids, prompt_len):
    labels = input_ids[:]
    for i in range(prompt_len):
        labels[i] = -100
    return labels

input_ids = [101, 102, 103, 201, 202]   # 前3个是 prompt，后2个是 answer
labels = build_labels(input_ids, prompt_len=3)

assert labels == [-100, -100, -100, 201, 202]
assert sum(1 for x in labels if x != -100) == 2
```

这里 `-100` 是很多训练框架默认使用的 ignore index，也就是“这个位置不参与损失计算”。

Hugging Face 风格的伪代码通常写成下面这样：

```python
# input: [prompt + answer]
# loss only on answer tokens
labels = input_ids.clone()
labels[:prompt_len] = -100
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
```

### 2. Encoder-Decoder 的实现方式

Encoder-Decoder 更接近经典的 source-target 组织方式：输入和标签天然分开。

```python
outputs = model(
    input_ids=source_ids,
    attention_mask=source_mask,
    labels=target_ids
)
loss = outputs.loss
```

它的好处是语义清晰：`input_ids` 就是输入 $x$，`labels` 就是目标 $y$。不需要手工把 prompt mask 掉，也不需要在联合序列里计算边界位置。

### 3. 一个可运行的玩具实现

下面用纯 Python 模拟“只对输出段计算损失”的思想。这个例子不依赖深度学习框架，但能反映训练目标差异。

```python
import math

def neg_log_likelihood(probs):
    loss = 0.0
    for p in probs:
        assert 0.0 < p <= 1.0
        loss += -math.log(p)
    return loss

# 玩具例子：x = "I eat", y = "我 吃"
# 只统计输出 token 的条件概率
output_token_probs = [0.6, 0.9]
loss = neg_log_likelihood(output_token_probs)

joint_prob = 1.0
for p in output_token_probs:
    joint_prob *= p

assert round(joint_prob, 2) == 0.54
assert round(loss, 4) == round(-math.log(0.6) - math.log(0.9), 4)

print("joint_prob =", joint_prob)
print("loss =", round(loss, 4))
```

这个代码体现的就是：

- Decoder-only 微调时，虽然模型也“看见”输入 token，但损失只统计输出 token。
- Encoder-Decoder 虽然组织方式不同，但本质上也只对目标序列 $y$ 的预测质量负责。

### 4. 实现对照表

| 项目 | Decoder-only | Encoder-Decoder |
|---|---|---|
| 输入格式 | 拼接成一个序列 | 源句和目标句分开 |
| loss 计算 | 只算输出段 | 只算目标序列 |
| 推理方式 | 续写前缀 | 编码后逐步解码 |
| 常见配置 | `labels=-100` mask prompt | `decoder_start_token_id`、`pad_token_id` |

### 5. 真实工程例子：新闻摘要 API

假设你要做一个“新闻摘要 API”：

- 输入：一篇 3000 字新闻正文
- 输出：80 字摘要
- 任务边界稳定，不需要聊天、多轮历史、工具调用

这个任务通常更适合 Encoder-Decoder。原因不是它“更高级”，而是它更贴近任务结构：

1. 输入长、输出短，条件映射关系明确。
2. decoder 每一步都能 cross-attend 到完整输入表示。
3. 服务接口清晰，就是 `source -> summary`。

相反，如果你要做的是“统一问答、摘要、改写、翻译、闲聊”的一个通用文本入口，Decoder-only 往往更合适，因为所有任务都能包装成同一种 prompt 续写问题。

---

## 工程权衡与常见坑

### 1. 核心权衡

| 维度 | Decoder-only | Encoder-Decoder |
|---|---|---|
| 接口统一性 | 强 | 中等 |
| 条件建模显式性 | 较弱 | 强 |
| 多任务混合训练 | 方便 | 需要更明确任务格式 |
| 任务边界清晰场景 | 可做，但常需 prompt 设计 | 更自然 |
| 部署接口复杂度 | 低 | 略高 |

这里“接口统一性”指的是：同一个模型是否能把聊天、补全、翻译、摘要都转成“给前缀然后继续写”。Decoder-only 在这点上优势明显，所以它很适合大一统产品接口。

但统一接口不等于所有任务都同样高效。若输入输出边界稳定，而且输出每一步都高度依赖输入全文，Encoder-Decoder 的结构会更直接。

### 2. 常见坑与规避

| 常见坑 | 影响 | 规避方式 |
|---|---|---|
| 把 Decoder-only 当成“只能生成” | 误判模型能力 | 认识到它也能做 QA、翻译、摘要 |
| 把 Encoder-Decoder 当成“只能翻译” | 误判适用范围 | 认识到它也能做分类、问答、摘要 |
| Decoder-only 微调时不 mask prompt | loss 目标错误 | 对 prompt 部分设 `-100` |
| Encoder-Decoder 配置不一致 | 训练/推理不稳定 | 对齐 `decoder_start_token_id` 和 `pad_token_id` |

### 3. 真实工程里更容易踩的坑

第一，输入截断。  
很多初学者把架构选择理解成理论问题，忽略了上下文长度限制。长文摘要时，如果输入被粗暴截断，那么无论选哪种架构，效果都会掉。区别只是：Encoder-Decoder 更像“读完输入再输出”，所以你更容易意识到输入完整性的重要性；Decoder-only 因为一切都是前缀，初学者反而容易忽视上下文预算。

第二，标签对齐错误。  
例如 Decoder-only 训练时，answer 段的起始位置错了一位，loss 还能正常下降，但学到的是错位目标。这个 bug 很隐蔽，因为程序不一定报错。

第三，padding 与 attention mask 混淆。  
“padding”是为了把不同长度样本补成同样长度的占位 token；“attention mask”是告诉模型哪些位置有效、哪些位置无效。两者相关但不是一回事。Encoder-Decoder 里这个问题更常见，因为 encoder 和 decoder 可能各自有 mask。

第四，推理重复。  
两类架构推理时都可能出现重复生成、句子打转、过早结束。很多人误以为这是架构本身的问题，实际上常常与 decoding 策略、长度惩罚、训练数据分布有关。

### 4. 一个实用判断

如果你在做产品，不妨先问四个问题：

1. 任务是不是稳定的“输入变输出”映射？
2. 输出每一步是否强依赖输入全文？
3. 是否希望所有任务共用一个前缀接口？
4. 部署时更在意统一服务栈，还是更在意任务专用效率？

这四个问题比“社区最近流行哪种架构”更接近真实工程决策。

---

## 替代方案与适用边界

先说结论：很多任务两类架构都能做，但“能做”不等于“最合适”。

| 任务类型 | 更适合的架构 | 原因 |
|---|---|---|
| 聊天 / 指令跟随 | Decoder-only | 前缀统一，扩展自然 |
| 代码补全 | Decoder-only | 本质是续写 |
| 机器翻译 | Encoder-Decoder | 输入输出边界清晰 |
| 新闻摘要 | Encoder-Decoder | 输入长、输出短，条件建模直接 |
| 问答式生成 | 两者都可 | 取决于数据组织与部署需求 |
| 分类 / 结构化输出 | 两者都可 | 一个偏条件映射，一个偏统一接口 |

### 1. 什么情况下优先选 Decoder-only

如果任务像“继续写下去”，优先考虑 Decoder-only。典型情况包括：

- 对话系统：系统提示、历史消息、用户提问天然就是一个前缀。
- 代码补全：已有代码上下文就是前缀，目标就是继续预测下一个 token。
- 通用文本助手：翻译、摘要、改写、问答可以统一包装成指令模板。

它的边界在于：当输入输出结构非常稳定，而且输入很长、输出较短时，虽然也能做，但往往需要更精细的 prompt 设计和 token 预算控制。

### 2. 什么情况下优先选 Encoder-Decoder

如果任务像“把输入变成另一个输出”，优先考虑 Encoder-Decoder。典型情况包括：

- 机器翻译
- 新闻摘要
- 标题生成
- 文本改写
- 信息抽取后转成结构化文本

它的边界在于：若你希望一个模型同时覆盖大量不同交互模式，Encoder-Decoder 的统一性通常不如 Decoder-only，自定义任务模板、训练流程和推理接口可能更复杂。

### 3. 替代思路

在真实系统中，还常见三种替代思路：

1. 用 Decoder-only 统一所有任务，靠 prompt 和数据格式解决任务差异。
2. 用 Encoder-Decoder 做核心条件生成任务，外围再配规则系统或分类器。
3. 按任务拆模，例如聊天用 Decoder-only，摘要翻译用 Encoder-Decoder。

第三种做法并不“落后”。如果你的业务接口稳定、流量结构清楚、延迟预算严格，拆分架构有时比追求“大一统”更实用。

最终选择标准可以压缩成一句话：  
如果任务核心是“统一前缀续写”，偏向 Decoder-only；如果任务核心是“稳定条件映射”，偏向 Encoder-Decoder。

---

## 参考资料

| 类型 | 资料 |
|---|---|
| 基础论文 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| Decoder-only | [Better Language Models and Their Implications](https://openai.com/index/better-language-models/) |
| Seq2Seq 统一建模 | [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) |
| 预训练 Seq2Seq | [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) |
| 工程实践 | [Hugging Face: Causal language modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling) |
| 工程实践 | [Hugging Face: EncoderDecoderModel](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder) |

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Better Language Models and Their Implications](https://openai.com/index/better-language-models/)
3. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
4. [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
5. [Hugging Face: Causal language modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling)
6. [Hugging Face: EncoderDecoderModel](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder)

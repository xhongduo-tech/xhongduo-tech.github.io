## 核心结论

指令数据的格式设计，核心不是“把对话写成 JSON”这一层外观，而是两件更底层的事：第一，明确模型看到的角色边界和轮次边界；第二，明确哪些 token 参与监督，哪些 token 只作为上下文存在。

“角色边界”就是让模型知道这段文字是谁说的。常见做法是在文本里插入专门的控制 token，例如 `<start_of_turn>` 和 `<end_of_turn>`，再配合 `user`、`assistant`、`model`、`system` 这类角色标记。这样，多轮对话在 tokenizer 看来不是一团普通文本，而是带结构的序列。

“监督”指训练时哪些位置会计算 loss，也就是交叉熵误差。指令微调里通常只让 assistant 的回答部分参与 loss，把 system 和 user 的内容全部 mask 掉。白话说：模型要学“怎么回答”，不是学“怎么复述题目”。

一个最小例子如下：

`<start_of_turn>user: 你好<end_of_turn><start_of_turn>assistant: 我是助手<end_of_turn>`

如果把 user 段的 mask 设为 0，把 assistant 段的 mask 设为 1，那么梯度只会从“我是助手”这段回答回传。这个设计通常比“整段都算 loss”更稳定，也更接近指令微调的目标。

| token片段 | role | 是否参与loss |
|---|---|---|
| `<start_of_turn>` | 控制符 | 0 |
| `user:` | 用户角色 | 0 |
| `你好` | 用户内容 | 0 |
| `<end_of_turn>` | 控制符 | 0 |
| `<start_of_turn>` | 控制符 | 0 或 1，取决于实现是否把回答起始控制符也纳入监督 |
| `assistant:` | 助手角色 | 通常 0 |
| `我是助手` | 助手内容 | 1 |
| `<end_of_turn>` | 控制符 | 通常 0 或 1，需训练与推理保持一致 |

---

## 问题定义与边界

这里先把问题说窄。本文讨论的是 SFT，也就是监督微调。监督微调可以理解为“给模型看标准问答样本，让它模仿这种回答方式”的训练阶段。本文不讨论 RLHF、DPO、推理时采样策略，也不讨论底层 tokenizer 训练过程。

一个训练样本，本质上是一个按角色拼接好的 token 序列：

1. 前面是若干轮 system/user/assistant 历史。
2. 最后一段是本轮 assistant 的目标回答。
3. loss 只对被标记为“应该学会生成”的 token 生效。

因此，输入不是“问题”，而是“完整上下文”；输出也不是“独立标签”，而是“同一序列里需要预测的后半段”。如果没有角色和轮次边界，模型只能把整串文本当作普通语言建模语料，无法稳定区分“指令”和“回答”。

边界还包括两类约束。

第一类是 tokenizer 边界。tokenizer 可以理解为“把字符串切成模型内部最小单位的工具”。如果你的模板里有 `<start_of_turn>`、`<end_of_turn>`、`<|assistant|>` 之类的控制符，就必须确认它们在训练和推理两侧的写法完全一致。少一个空格、换一个冒号、首轮和中间轮模板不同，都会让边界定位失败。

第二类是模板一致性。训练时如果用：

`<start_of_turn>user: 文本<end_of_turn><start_of_turn>assistant: 回答<end_of_turn>`

推理时却改成：

`<s>user : 文本</s>assistant :`

那你虽然“语义上看起来差不多”，但对 tokenizer 和 collator 来说，已经是另一种格式。这里的 collator 可以理解为“把单条样本整理成 batch，并顺手生成 labels 与 mask 的组件”。模板一旦变，mask 规则往往也要跟着改。

可以把训练流水线理解成下面这条链路：

`posts.json / jsonl / parquet 原始样本 -> 按角色拼接轮次 -> tokenizer 切分 -> 生成 loss mask -> 计算 loss`

这条链路里，真正决定模型学到什么的，不是 JSON 外壳，而是“拼接模板”和“loss mask”。

---

## 核心机制与推导

把 token 序列记作 $y_1, y_2, \dots, y_n$，再定义一个二值掩码 $m_i \in \{0, 1\}$。其中：

- $m_i = 1$ 表示第 $i$ 个 token 参与 loss。
- $m_i = 0$ 表示第 $i$ 个 token 只提供上下文，不参与 loss。

那么常见的目标函数可以写成：

$$
L = - \sum_{i=1}^{n} m_i \log p(y_i \mid y_{<i})
$$

如果实现里要做平均，通常还会再除以 $\sum_i m_i$，而不是除以总长度 $n$。原因很直接：如果序列里 70% 都是 prompt，除以总长度会把真实梯度稀释掉。

这里最容易混淆的一点是：被 mask 的 token 不是“对模型没用”。它们仍然进入上下文。也就是说，user 的问题、system 的约束、多轮历史，都会影响 assistant 回答 token 的条件概率 $p(y_i \mid y_{<i})$；只是这些 prompt token 自己不产生监督误差。

玩具例子最容易看明白。假设一条样本共有 15 个 token：

- 前 10 个 token 是 prompt，全部 $m=0$
- 后 5 个 token 是 assistant 回答，全部 $m=1$

那么这条样本的梯度来源占比就是：

- prompt：0%
- answer：100%

这就是“只教模型如何回答”的精确数学含义。

多轮场景下，mask 不是只有一段连续的 1，而可能是分散的。例如：

- system 说明：全 0
- 第 1 轮 user：全 0
- 第 1 轮 assistant：全 1
- 第 2 轮 user：全 0
- 第 2 轮 assistant：全 1

也就是说，active token 可能在序列中多次出现，中间夹着很多 0。只要模板明确，mask 依然能精确对齐。

这里再给一个简化的轮次例子：

`<start_of_turn>user: 2+2=?<end_of_turn><start_of_turn>assistant: 4<end_of_turn><start_of_turn>user: 再加1呢？<end_of_turn><start_of_turn>assistant: 5<end_of_turn>`

如果没有 `<end_of_turn>` 这种结束符，模型在训练中更难稳定学到“上一轮回答到哪里结束、下一轮提问从哪里开始”。多轮越长，串轮风险越高。所谓“串轮”，就是模型把上一轮回答和下一轮问题混成一段连续文本，进而在推理时出现角色错乱或接话位置错误。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实 tokenizer，但足够说明“先拼模板，再生成 mask”的基本方法。

```python
from typing import List, Dict

START = "<start_of_turn>"
END = "<end_of_turn>"

def encode_turn(role: str, content: str) -> List[str]:
    return [START, f"{role}:", *content.split(), END]

def build_sample(messages: List[Dict[str, str]]):
    tokens = []
    mask = []

    for msg in messages:
        turn_tokens = encode_turn(msg["role"], msg["content"])
        tokens.extend(turn_tokens)

        if msg["role"] == "assistant":
            # 控制符和角色名通常不参与loss，只让回答正文参与
            turn_mask = [0, 0] + [1] * (len(turn_tokens) - 3) + [0]
        else:
            turn_mask = [0] * len(turn_tokens)

        mask.extend(turn_mask)

    return tokens, mask

messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "我是 助手"},
    {"role": "user", "content": "1 加 1 等于 几"},
    {"role": "assistant", "content": "等于 2"},
]

tokens, mask = build_sample(messages)

assert len(tokens) == len(mask)
assert sum(mask) == 4  # “我是 助手”等于2，共4个正文token参与loss
assert mask[tokens.index("user:")] == 0
assert "assistant:" in tokens

active_tokens = [t for t, m in zip(tokens, mask) if m == 1]
assert active_tokens == ["我是", "助手", "等于", "2"]

print(tokens)
print(mask)
print(active_tokens)
```

如果换成真实工程代码，逻辑也是同一套，只是多了真实 tokenizer 和 batch 处理。最常见的两种实现方式是：

1. 先把整段对话按模板拼成字符串，再用 tokenizer 编码，然后依据 assistant 起始位置构造 `labels` 或 `loss_mask`。
2. 使用训练框架提供的 `train_on_inputs: false` 或“completion-only collator”，自动把 prompt 部分置为忽略标签。

下面是更接近工程现场的伪代码：

```python
def format_dialogue(user_text: str, assistant_text: str) -> str:
    prompt = f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
    reply = f"<start_of_turn>assistant\n{assistant_text}<end_of_turn>\n"
    return prompt + reply

def build_labels(tokenizer, user_text: str, assistant_text: str):
    full_text = format_dialogue(user_text, assistant_text)
    prompt_text = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>assistant\n"

    full_ids = tokenizer(full_text)["input_ids"]
    prompt_ids = tokenizer(prompt_text)["input_ids"]

    labels = [-100] * len(full_ids)
    for i in range(len(prompt_ids), len(full_ids)):
        labels[i] = full_ids[i]

    return full_ids, labels
```

这里的 `-100` 是很多深度学习框架里“忽略该位置 loss”的常见约定。真实工程例子通常是：你有一个客服机器人训练集，每条样本含 system 规则、用户提问、历史轮次和标准回复。训练时要保留全部上下文，但只对客服标准回复回传梯度。这样模型会学会在相同上下文下给出接近标准的答复，而不是学会把用户问题原样背出来。

---

## 工程权衡与常见坑

真正容易出问题的地方，不是“有没有模板”，而是“模板是否稳定到 token 级”。

最常见的坑是边界字符串不一致。比如训练时写的是 `<start_of_turn>user:`，后来某个数据脚本改成了 `<start_of_turn>user :`。人眼看只是多一个空格，但 completion-only collator 往往靠精确匹配模板定位回答起点。一旦定位失败，就可能把 prompt 也算进 loss，结果模型开始学习复述问题。

| 差异类型 | 例子 | 直接后果 |
|---|---|---|
| 空格不同 | `user:` vs `user :` | boundary 搜索失败 |
| 角色名不同 | `assistant` vs `model` | mask 错位 |
| 起始符不同 | `<s>` vs `<start_of_turn>` | tokenizer 切分完全变化 |
| 首轮模板不同 | 首轮有 BOS，中间轮没有 | 第一次匹配位置异常 |
| 结束符缺失 | 少了 `<end_of_turn>` | 多轮串轮 |
| 训练推理不一致 | 训练是 chat template，推理是纯前缀 | 可用率显著波动 |

这里要特别强调一个工程事实：同一个底座模型，只改模板，不改权重，线上可用率也可能明显波动。原因不是模型“心情不好”，而是它在训练时学到的是某种固定格式下的条件分布。你换了提示边界，相当于换了输入分布。

第二个坑是“字符长度代替 token 长度”。很多人会写出这样的逻辑：`response_start = len(prompt_text)`。这在中文、BPE、SentencePiece 甚至带特殊 token 的场景里都不可靠。正确做法是分别 tokenize `prompt_text` 和 `full_text`，然后用 token 数对齐边界。

第三个坑是多轮样本没有显式轮次结构。比如直接写：

`用户：... 助手：... 用户：... 助手：...`

这种前缀模板在少量一轮问答里可以工作，但多轮拉长后，模型更容易把“用户：”和上一轮回答尾部粘在一起，尤其当 tokenizer 没把这些前缀当成稳定控制符时，边界就更脆弱。

一个实用排查清单是：

- 训练模板和推理模板是否字节级一致
- 特殊 token 是否已加入 tokenizer 或 chat template
- 首轮和后续轮是否使用同一套边界规则
- loss 是否只在 assistant 正文上平均
- 多轮样本是否包含显式结束符
- 数据清洗后是否引入了多余空格、换行、全角标点

---

## 替代方案与适用边界

最稳妥的默认方案，是“显式控制 token + 明确角色模板 + assistant-only loss mask”。它的优点是可解释、可调试、适合多轮，对大多数 instruction tuning 任务都成立。

但它不是唯一方案。常见替代方案是简化前缀模板，例如：

`user: 解释一下哈希表\nassistant:`

这种格式的优点是简单、兼容老数据、容易人工阅读；缺点是强依赖 tokenizer 对这些前缀的切分是否稳定，也更容易在多轮里发生边界歧义。

| 方案 | 适用场景 | 优点 | 风险 |
|---|---|---|---|
| 控制token + mask | 多轮 SFT、生产训练 | 边界清晰、可控性高 | 模板维护成本更高 |
| 简化 prefix | 单轮或短对话、旧项目迁移 | 实现简单 | 边界脆弱、串轮概率更高 |
| 全量loss | 继续预训练、领域续训 | 实现最直接 | 不适合 instruction tuning 目标 |
| 仅最后一轮assistant算loss | 多轮客服、工具调用 | 更贴近线上目标 | 历史 assistant 轮可能学不到 |

这里还要划清一个边界：在 RLHF、偏好优化、评估打分阶段，数据格式和监督信号会换一套机制。比如偏好学习更关心“哪条回答更好”，而不是把整段 assistant token 作为逐位置监督目标。因此，“assistant-only mask”是 SFT 中的核心设计，不必机械推广到所有训练阶段。

真实工程上，一个常见做法是：

- 预训练或继续预训练：全量 token 参与 loss。
- 指令微调：只监督 assistant 回答。
- 偏好优化：基于回答对进行奖励或排序学习。

所以，格式设计必须跟训练目标绑定，而不是跟“数据是不是聊天样子”绑定。

---

## 参考资料

- Google AI for Developers, Gemma prompt structure / formatting 文档，说明了 `<start_of_turn>`、`<end_of_turn>`、`user`、`model` 这类控制 token 的作用，以及训练与推理模板需要保持一致：https://ai.google.dev/gemma/docs/core/prompt-structure
- Michael Brenndoerfer, *Instruction Tuning Training: Data Mixing & Loss Masking*，解释了为什么 prompt token 应被 mask，为什么平均 loss 应按有效 token 数而不是总长度归一化：https://mbrenndoerfer.com/writing/instruction-tuning-training-data-mixing-loss-masking
- Hugging Face TRL Issue #1184，展示了 instruction template 首次出现和后续出现 token 化不同，导致 `DataCollatorForCompletionOnlyLM` 边界定位失败的具体案例：https://github.com/huggingface/trl/issues/1184
- 如果要继续查实际实现，可以继续检索关键词：`Gemma prompt structure`、`completion only collator`、`loss masking assistant only`、`TRL DataCollatorForCompletionOnlyLM`

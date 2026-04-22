## 核心结论

Token Dropout 是一种输入腐化训练方法：训练时按概率随机屏蔽一部分 token，通常做法是把这些 token 的 embedding 置零，或替换成 `[MASK]`、sentinel token 这类特殊占位符，然后让模型从上下文中恢复原始信息。

“输入腐化”指的是故意破坏模型看到的输入，而不是修改真实标签。它的目的不是让数据变脏，而是让模型不能只记住局部表面模式，必须学习上下文依赖。

设第 $i$ 个 token 的 embedding 是 $e_i$，随机变量 $m_i$ 表示这个 token 是否保留：

$$
m_i \sim Bernoulli(1-r)
$$

污染后的 embedding 可以写成：

$$
\tilde e_i = m_i e_i + (1-m_i)v
$$

其中 $r$ 是 dropout rate，$v$ 可以是零向量、`[MASK]` 的 embedding，或 T5 中的 `<extra_id_0>` 这类 sentinel embedding。

新手版理解：原句是“我 爱 深度 学习 模型”，训练时随机挖掉“深度 学习”，输入变成“我 爱 `<extra_id_0>` 模型”，模型需要根据上下文恢复被挖掉的内容。

Token Dropout 的核心价值不在于替代预训练目标，而在于作为一种输入扰动手段，提高模型对缺失、噪声和断裂文本的鲁棒性。

| 方法 | 作用位置 | 输入是否被破坏 | 训练目标 | 典型模型 |
|---|---:|---:|---|---|
| Token Dropout | 输入 token 或 embedding | 是 | 取决于模型目标 | BERT、T5、BART、GPT 变体 |
| 普通 Dropout | 神经网络激活 | 否 | 原训练目标不变 | 各类神经网络 |
| BERT MLM | 输入 token | 是 | 只预测 masked 位置 | BERT |
| T5 去噪 | 输入 span | 是 | decoder 重建被污染 span | T5 |
| BART 去噪 | 输入序列 | 是 | decoder 重建原序列 | BART |

---

## 问题定义与边界

Token Dropout 描述的是“怎么破坏输入”，而 BERT MLM、T5 去噪、BART 去噪描述的是“用什么目标训练模型”。这两个层次不能混在一起。

MLM 是 Masked Language Modeling，白话说就是“遮住一些词，只让模型预测这些被遮住的词”。Denoising Seq2Seq 是去噪序列到序列训练，白话说就是“给模型一段被破坏的文本，让 decoder 生成需要恢复的目标文本”。

可以把问题写成：

$$
x \rightarrow \tilde x
$$

其中 $x$ 是原始序列，$\tilde x$ 是被污染后的输入。Token Dropout 关注的是从 $x$ 到 $\tilde x$ 的污染过程；训练目标则决定模型如何从 $\tilde x$ 学回 $x$ 的信息。

同一句话在不同方法下的训练信号不同。原句：

```text
我 爱 深度 学习 模型
```

如果遮掉“深度 学习”：

| 方法 | 污染输入 | loss 计算位置 | 是否使用 decoder | 重建内容 |
|---|---|---|---:|---|
| Token Dropout | 我 爱 `<MASK>` `<MASK>` 模型 | 由具体目标决定 | 不一定 | 不固定 |
| BERT MLM | 我 爱 `[MASK]` `[MASK]` 模型 | 只在“深度”“学习”位置 | 否 | 单 token |
| T5 去噪 | 我 爱 `<extra_id_0>` 模型 | decoder 目标序列全部位置 | 是 | 被污染 span |
| BART 去噪 | 我 爱 `[MASK]` 模型 | decoder 目标序列全部位置 | 是 | 原始序列或被恢复序列 |

这里的边界很重要：

Token Dropout = 输入层噪声注入。它回答“训练输入怎么被随机破坏”。

MLM = 预测被遮住 token。它回答“encoder 输出后，哪些位置参与分类 loss”。

Denoising Seq2Seq = 重建被污染序列。它回答“encoder 读污染输入，decoder 生成什么目标”。

文献和工程代码里，Token Dropout 的命名不完全统一。有时它指 token 级随机 mask，有时指连续 span mask，有时指 embedding zeroing，有时指 sentinel replacement。因此读论文或看代码时，不要只看名字，要看三个字段：输入如何改、label 如何构造、loss 在哪里算。

---

## 核心机制与推导

Token Dropout 有效的原因是：它让模型在训练时反复看到不完整输入，从而学习更稳定的上下文表示。

“上下文表示”指的是一个 token 的向量不只包含它自己的信息，还包含周围 token 对它的约束。例如“我 爱 ___ 学习 模型”中，空缺处更可能是“深度”，而不是一个随机无关词。

推导可以分三步。

第一步，定义噪声变量：

$$
m_i \sim Bernoulli(1-r)
$$

当 $m_i=1$ 时，第 $i$ 个 token 保留；当 $m_i=0$ 时，第 $i$ 个 token 被替换或置零。

第二步，构造污染输入：

$$
\tilde e_i = m_i e_i + (1-m_i)v
$$

这里 $v$ 的选择决定污染格式。如果 $v=0$，就是 embedding zeroing；如果 $v$ 是 `[MASK]` embedding，就是 mask replacement；如果 $v$ 是 `<extra_id_k>`，就是 T5 常用的 sentinel replacement。

第三步，用训练目标提供学习信号。

BERT MLM 的 loss 是：

$$
L = - \sum_{i \in M} \log p(x_i \mid \tilde x)
$$

其中 $M$ 是被 mask 的位置集合。注意，BERT MLM 只在被遮住的位置计算 loss，不是所有位置都参与预测。

Seq2Seq 去噪的 loss 是：

$$
L = - \sum_{t=1}^{|y|} \log p(y_t \mid \tilde x, y_{<t})
$$

其中 $y$ 是 decoder 端目标序列。T5 通常让 $y$ 只包含被污染的 span 加 sentinel；BART 更常见的是让 decoder 重建原始序列。

玩具例子：

```text
原句: [我, 爱, 深度, 学习, 模型]
r = 0.4
被选中 span: [深度, 学习]
污染输入: [我, 爱, <extra_id_0>, 模型]
T5 目标: [<extra_id_0>, 深度, 学习, </s>]
```

这件事的机制版本是：模型 encoder 只能看到“我 爱 `<extra_id_0>` 模型”，decoder 必须根据上下文生成“深度 学习”。如果训练集中反复出现类似模式，模型会学到“深度 学习”与“模型”之间的语言关联，而不是只记住某个固定位置的词。

| 机制 | 做法 | 优点 | 风险 |
|---|---|---|---|
| token-level mask | 单个 token 随机替换 | 实现简单 | 容易破坏碎片化，语义不连续 |
| span-level mask | 连续片段整体替换 | 更接近真实缺失文本 | label 构造更复杂 |
| embedding zeroing | 直接把 embedding 置零 | 不增加词表符号 | checkpoint 未必见过零向量输入 |
| sentinel replacement | 用 `<extra_id_k>` 占位 | 适合 T5 span corruption | 格式必须和预训练一致 |

从优化角度看，Token Dropout 相当于在输入空间做数据增强。模型不只学习 $x$ 上的预测，还学习 $\tilde x$ 上的预测。这样得到的表示通常更平滑：输入少量缺失或噪声时，输出不应该剧烈变化。

---

## 代码实现

实现 Token Dropout 时，必须区分“改输入”和“改标签”。改输入是把 `input_ids` 里的部分 token 替换掉；改标签是决定哪些位置参与 loss，或者 decoder 端应该生成什么。

下面是一个最小可运行的 Python 例子，演示 token 级 dropout、label 对齐和 `ignore_index` 处理。这里不用深度学习框架，只保留训练数据构造逻辑。

```python
import random

PAD = 0
MASK = 103
IGNORE_INDEX = -100

def token_dropout_for_mlm(input_ids, dropout_rate=0.15, seed=7):
    random.seed(seed)

    corrupted = list(input_ids)
    labels = [IGNORE_INDEX] * len(input_ids)

    for i, token_id in enumerate(input_ids):
        if token_id == PAD:
            continue

        if random.random() < dropout_rate:
            corrupted[i] = MASK
            labels[i] = token_id

    attention_mask = [0 if token_id == PAD else 1 for token_id in input_ids]
    return {
        "input_ids": corrupted,
        "attention_mask": attention_mask,
        "labels": labels,
    }

example = [11, 22, 33, 44, 55, PAD]
batch = token_dropout_for_mlm(example, dropout_rate=0.4, seed=1)

assert len(batch["input_ids"]) == len(example)
assert len(batch["attention_mask"]) == len(example)
assert len(batch["labels"]) == len(example)
assert batch["attention_mask"][-1] == 0
assert batch["labels"][-1] == IGNORE_INDEX
assert any(x == MASK for x in batch["input_ids"])

print(batch)
```

这段代码对应 BERT 类 MLM 的简化版本：被 mask 的位置保留原始 token 作为 label，未被 mask 的位置设为 `-100`，表示不参与 loss。PyTorch 的 `CrossEntropyLoss(ignore_index=-100)` 常用这个约定。

对于 T5/BART 类 seq2seq 模型，逻辑会不同。输入侧需要构造污染后的 `input_ids`，decoder 侧需要构造目标 `labels`。例如：

```text
原始输入: 我 爱 深度 学习 模型
encoder input_ids: 我 爱 <extra_id_0> 模型
decoder labels: <extra_id_0> 深度 学习 </s>
```

新手可理解版流程是：

```text
1. 随机采样要屏蔽的 token 或 span
2. 构造污染后的 encoder 输入
3. 构造 decoder 目标序列
4. 把 padding 位置的 label 改成 ignore_index
5. 送入模型计算 loss
```

训练实现时至少检查这些字段：

| 字段 | 含义 | 常见错误 |
|---|---|---|
| `input_ids` | 模型实际看到的输入 | 只改了 label，忘了污染输入 |
| `attention_mask` | 哪些位置是有效 token | padding 位置被当成有效输入 |
| `labels` | 参与 loss 的目标 | MLM 与 seq2seq label 格式混用 |
| `pad_token_id` | padding token 编号 | 和 tokenizer 配置不一致 |
| `ignore_index` | loss 忽略位置 | padding 位置参与了 loss |

真实工程例子：在医学问答语料上做领域继续预训练。医学文本常有缩写、错别字、缺词和口语截断，例如“二甲双胍早晚？”、“血压 150 要不要加药”。如果继续预训练时加入 $r \approx 0.15$ 的 token/span corruption，模型会更频繁地学习“缺失词附近的上下文约束”，下游问答或分类任务通常更不容易被轻微输入缺失影响。

---

## 工程权衡与常见坑

Token Dropout 最重要的超参数是 $r$。经验上可以先从：

$$
r \approx 0.15
$$

开始，再扫 `0.1 / 0.2`。$r$ 过低，输入几乎没有变化，正则化作用弱；$r$ 过高，输入信息损失太多，任务会变成猜谜，训练 loss 可能明显变差。

| 风险 | 表现 | 规避方式 |
|---|---|---|
| `r` 过高 | loss 上升，模型生成泛化变差 | 从 `0.15` 起步，扫 `0.1 / 0.2` |
| span 太短 | 只学到局部 token 替换 | 对 T5/BART 尝试 span corruption |
| span 太长 | 输入信息不足，目标过难 | 限制平均 span 长度 |
| label 和 padding 未对齐 | padding 参与 loss，训练不稳定 | padding label 设为 `ignore_index` |
| sentinel 格式错误 | 模型无法理解污染格式 | 使用 checkpoint 预训练时见过的格式 |
| 与 checkpoint 预训练格式不一致 | 微调退化或收敛慢 | 查 tokenizer 和模型文档 |

第一个常见坑是把 Token Dropout 和普通 Dropout 混淆。普通 Dropout 是在网络中间层随机置零激活，用来减少神经元之间的共适应；Token Dropout 是在输入层破坏 token 或 embedding，用来增强模型对输入缺失的鲁棒性。

第二个常见坑是把输入 corruption 和 loss mask 混为一谈。BERT MLM 中，“输入被 mask”与“只在 masked 位置算 loss”经常同时出现，但它们是两件事。你可以污染输入但让所有位置参与某种训练目标，也可以只让部分位置参与 loss。

第三个常见坑是忽略 checkpoint 的预训练格式。T5 常用 `<extra_id_0>` 这类 sentinel token 表示被污染 span。如果把它改成普通 `[MASK]`，模型不一定知道这个符号是什么意思。BART、T5、BERT 的输入污染格式不是随便互换的。

工程检查清单：

```text
1. labels 的 padding 是否设为 ignore_index
2. decoder 输入是否由 target 正确右移得到
3. input_ids 是否真的被污染
4. attention_mask 是否仍然匹配污染后的输入长度
5. tokenizer 是否包含所用的 mask 或 sentinel token
6. 当前 checkpoint 是否在预训练时见过这种污染格式
```

对于短文本场景，还要控制污染强度。比如用户搜索 query 只有 5 个 token，如果 $r=0.4$，可能一次遮掉两个关键 token，语义已经不可恢复。对于长文档，适当 span corruption 更合理，因为真实噪声往往是连续片段缺失，而不是均匀随机缺一个字。

---

## 替代方案与适用边界

Token Dropout 适合做鲁棒性增强和领域继续预训练，不适合被当作所有语言模型的默认预训练方式。

对于 GPT 类纯自回归模型，标准训练目标是 causal LM，白话说就是“只看左边上下文，预测下一个 token”。它的目标可以写成：

$$
p(x) = \prod_t p(x_t \mid x_{<t})
$$

这和去噪重建不同。去噪模型看到的是被污染后的整体输入；GPT 标准训练时不应该看到未来 token。因此，在 GPT 上使用 Token Dropout 通常只是额外正则或数据增强，不是原生训练目标。

| 方案 | 适用模型 | 训练目标 | 典型用途 |
|---|---|---|---|
| Token Dropout | 多种架构 | 破坏输入，目标依具体任务而定 | 鲁棒性增强、继续预训练 |
| BERT MLM | encoder-only | 预测 masked token | 表示学习、分类、检索 |
| T5 去噪 | encoder-decoder | 重建被污染 spans | 文本到文本任务 |
| BART 去噪 | encoder-decoder | 从污染输入重建原文 | 生成、摘要、翻译 |
| GPT causal LM | decoder-only | 预测下一个 token | 自回归生成 |

什么时候值得用 Token Dropout：

| 场景 | 是否适合 | 原因 |
|---|---:|---|
| 医学、法律、客服等小规模领域语料 | 适合 | 术语密集，输入噪声常见 |
| 搜索 query、短文本分类 | 谨慎使用 | 文本短，遮掉关键 token 后语义损失大 |
| T5/BART 继续预训练 | 适合 | 原生支持去噪式目标 |
| 标准 GPT 大规模生成预训练 | 不作为默认方案 | 目标与自回归生成不一致 |
| 已有强格式依赖的 checkpoint 微调 | 谨慎使用 | 污染格式不一致会破坏分布 |

一个真实工程例子是客服机器人。用户输入经常不完整，例如“退款 昨天 到账没”、“会员 发票 开不了”。如果模型在继续预训练时见过随机缺词或 span 缺失，它更容易从剩余上下文判断意图。但如果目标是训练一个通用聊天 GPT，默认仍应使用 causal LM；Token Dropout 只能作为附加实验，而不是主训练范式。

适用边界可以压缩成一句话：如果目标是提升输入缺失和噪声条件下的表示质量，Token Dropout 值得用；如果目标是标准自回归生成，就不要把它当默认预训练方式。

---

## 参考资料

| 资料 | 对应概念 | 建议阅读顺序 |
|---|---|---:|
| BERT | token 级 MLM | 1 |
| T5 | sentinel + span corruption | 2 |
| BART | 去噪 seq2seq | 3 |
| Hugging Face T5 文档 | 工程实现参考 | 4 |

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)
3. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://aclanthology.org/2020.acl-main.703/)
4. [Hugging Face T5 文档](https://huggingface.co/docs/transformers/model_doc/t5)

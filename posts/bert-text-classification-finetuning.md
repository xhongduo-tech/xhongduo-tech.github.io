## 核心结论

BERT 文本分类微调，就是把输入文本送入 BERT，取最终层的 `[CLS]` 表示，再经过一个很小的分类头输出类别概率。

任务输出流程可以写成：

```text
文本 -> BERT -> [CLS] -> 分类头 -> 类别
```

`[CLS]` 是一个特殊标记，白话解释是：它不对应原文里的某个词，而是放在句首，让模型在训练中学会把整段文本的信息汇总到这个位置。分类任务通常不需要对每个 token 都输出标签，只需要拿 `[CLS]` 的最终表示作为句级表示。

示意图：

```text
[CLS]  退款  未到账  ，  已经  等了  3天  [SEP]
  |      |      |        |    |      |     |     |
 h0     h1     h2       h3   h4     h5    h6    h7

分类使用 h0，也就是 [CLS] 的最终隐状态
```

新手版理解：把“这个商品很好”交给 BERT，BERT 先根据上下文生成整句话的表示，再用 `[CLS]` 这个总括向量判断它是“积极”还是“消极”。

工程版理解：客服工单输入“退款未到账，已经等了 3 天”，BERT 输出 `[CLS]` 向量，分类头最终预测为“退款”类。

微调不是从零训练一个分类器，而是在预训练语义表示上做任务适配。BERT 已经通过大规模语料学习了词、短语、句子关系等通用表示；微调阶段只用较小的标注数据，把这些通用表示调整到情感分析、主题分类、自然语言推断、客服工单分流等具体任务上。

---

## 问题定义与边界

文本分类的输入是文本，输出是离散标签。离散标签是预先定义好的类别，例如“正面/负面”“退款/物流/账户异常/其他”。BERT 文本分类微调解决的是句级或句对级判断问题，不是给每个词打标签的序列标注问题。

输入输出可以定义为：

$$
g(X) \rightarrow y
$$

其中 $X$ 是输入文本或文本对，$y$ 是标签 id。标签 id 是把人类可读标签映射成整数，例如：

```text
{"退款": 0, "物流": 1, "账户异常": 2, "其他": 3}
```

常见任务形式如下：

| 任务形式 | 输入 | 输出 | BERT 输入格式 | 例子 |
|---|---|---|---|---|
| 单句分类 | 一段文本 | 一个类别 | `[CLS] A [SEP]` | “这个商品很好” -> 正面 |
| 句对分类 | 两段文本 | 一个类别或关系 | `[CLS] A [SEP] B [SEP]` | “问题描述 + 用户补充说明” -> 工单类别 |
| 多标签分类 | 一段或两段文本 | 多个可同时成立的标签 | `[CLS] A [SEP]` 或 `[CLS] A [SEP] B [SEP]` | 一条投诉同时属于“退款”和“物流” |

单句任务的工程输入：

```text
[CLS] 这个商品很好 [SEP]
```

句对任务的工程输入：

```text
[CLS] 问题描述 [SEP] 用户补充信息 [SEP]
```

句对分类中的 `token_type_ids` 用来告诉模型哪些 token 属于 A，哪些 token 属于 B。白话解释：它像一列段落编号，第一段通常是 0，第二段通常是 1。

BERT 微调适合以下场景：

| 适合 | 原因 |
|---|---|
| 情感分析 | 标签固定，输入通常较短 |
| 主题分类 | 类别体系明确，文本可截断到合理长度 |
| 自然语言推断 | 句对关系判断是 BERT 的典型用法 |
| 工单分流 | 输入描述短，标签体系稳定 |

不适合或需要改造的场景包括：

| 不适合直接使用 | 原因 |
|---|---|
| 超长文档分类 | 普通 BERT 通常有最大长度限制，常见为 512 tokens |
| 标签经常变化 | 分类头依赖固定 `num_labels` 和固定标签映射 |
| 多标签任务直接用 softmax | softmax 假设类别互斥，多标签应使用 sigmoid |
| 序列标注任务 | 需要对每个 token 输出标签，不能只取 `[CLS]` |

---

## 核心机制与推导

BERT 把输入序列编码成上下文相关表示。上下文相关表示的意思是：同一个词在不同句子里会得到不同向量，因为模型会根据周围词重新计算它的含义。

例如“这家店发货太慢了”里，“慢了”不是孤立的速度描述，而是对“发货”的负面评价。BERT 通过多层 Transformer 编码器，让“慢了”的表示融合“这家店”“发货”等上下文信息，最后 `[CLS]` 位置得到整句表示。

单句输入记为：

$$
X = [x_0=[CLS], x_1, ..., x_n]
$$

句对输入记为：

$$
X = [CLS]\ A\ [SEP]\ B\ [SEP]
$$

BERT 编码后得到：

$$
H = f_\theta(X) = [h_0, h_1, ..., h_n]
$$

其中 $h_0$ 是 `[CLS]` 的最终隐状态。隐状态是神经网络中间层输出的向量，白话解释是：模型把 token 转成一串数字，这串数字保存了它在当前上下文里的语义信息。

原始 BERT 分类路径通常先使用 pooler：

$$
u = \tanh(W_p h_0 + b_p)
$$

pooler 是把 `[CLS]` 的最终隐状态再过一层线性变换和 `tanh` 激活得到的句级向量。然后分类头继续计算：

$$
u' = Dropout(u)
$$

$$
z = W_c u' + b_c
$$

$$
p = softmax(z)
$$

$$
L = -\log p_y
$$

`Dropout` 是训练时随机丢弃部分神经元输出的正则化方法，白话解释是：它强迫模型不要过度依赖某几个特征。`logits` 是进入 softmax 前的原始分数，白话解释是：它还不是概率，但分数越大，模型越倾向于该类别。`softmax` 会把多个分数转成总和为 1 的概率分布。交叉熵损失用于惩罚模型给真值类别的概率太低。

| 符号 | 含义 |
|---|---|
| $h_0$ | `[CLS]` 的最终隐状态 |
| $u$ | pooler 输出的句级向量 |
| $z$ | 分类头输出的 logits |
| $p$ | softmax 后的类别概率 |
| $L$ | 交叉熵损失 |

完整流程图：

```text
输入 token
  -> BERT 编码
  -> [CLS] 最终隐状态 h0
  -> pooler: tanh(Wp h0 + bp)
  -> dropout
  -> linear
  -> softmax
  -> loss
```

数值例子：分类头输出

```text
logits = [1.2, 0.8]
```

softmax 后近似为：

```text
softmax(logits) ≈ [0.60, 0.40]
```

如果真值是第 2 类，则：

$$
L = -\ln(0.40) \approx 0.916
$$

这个例子说明，模型给真值类别的概率越高，损失越小；给真值类别的概率越低，损失越大。

---

## 代码实现

最小训练流程可以写成：

```text
数据集 -> tokenizer -> dataloader -> model -> loss -> optimizer
```

新手版伪代码：

```text
1. 读取文本和标签
2. 用 tokenizer 编码成 input_ids、attention_mask、token_type_ids
3. 把输入送入 BERT
4. 取 [CLS] 表示
5. 接分类头
6. 计算交叉熵并反向传播
```

三个核心输入字段如下：

| 字段 | 含义 | 作用 |
|---|---|---|
| `input_ids` | token 对应的词表 id | 告诉模型输入了哪些 token |
| `attention_mask` | 真实 token 为 1，padding 为 0 | 告诉模型哪些位置有效 |
| `token_type_ids` | A 段为 0，B 段为 1 | 句对任务中区分两段文本 |

参数表：

| 参数 | 常见值 | 含义 |
|---|---:|---|
| `num_labels` | 2、4、10 | 标签类别数 |
| `max_length` | 128、256、512 | 最大 token 长度 |
| `batch_size` | 8、16、32 | 每步训练样本数 |
| `lr` | 2e-5、3e-5、5e-5 | BERT 微调学习率 |

下面是一个可运行的玩具例子，不依赖外部模型下载，用 PyTorch 模拟 `[CLS] -> Dropout -> Linear -> CrossEntropyLoss` 的分类头路径。它不等价于完整 BERT，但能验证分类头和损失计算机制。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

label2id = {"negative": 0, "positive": 1}
num_labels = len(label2id)
hidden_size = 4

# 假设这是 BERT 输出的 [CLS] 向量，batch_size=2
cls_hidden = torch.tensor([
    [0.2, 0.1, 0.7, -0.3],
    [-0.4, 0.9, 0.2, 0.5],
], dtype=torch.float32)

labels = torch.tensor([
    label2id["positive"],
    label2id["negative"],
], dtype=torch.long)

classifier = nn.Sequential(
    nn.Dropout(p=0.0),
    nn.Linear(hidden_size, num_labels),
)

logits = classifier(cls_hidden)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

probs = torch.softmax(logits, dim=-1)
preds = torch.argmax(probs, dim=-1)

assert logits.shape == (2, num_labels)
assert probs.shape == (2, num_labels)
assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)
assert loss.item() > 0
assert preds.shape == labels.shape

print("logits:", logits)
print("probs:", probs)
print("loss:", loss.item())
```

Hugging Face 风格的工程代码通常更短，因为 `BertForSequenceClassification` 已经封装了 BERT、pooler、dropout、linear 和 loss：

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
)

label2id = {"退款": 0, "物流": 1, "账户异常": 2, "其他": 3}

texts = ["退款未到账，已经等了 3 天"]
labels = torch.tensor([label2id["退款"]])

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=labels,
)

loss = outputs.loss
logits = outputs.logits
pred = torch.argmax(logits, dim=-1)

assert logits.shape[-1] == 4
```

单句任务一般传 `input_ids` 和 `attention_mask` 就能工作；句对任务应传入两段文本，并保留 `token_type_ids`：

```python
batch = tokenizer(
    ["退款未到账"],
    ["已经等了 3 天，客服说还要核实"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch.get("token_type_ids"),
)
```

真实工程例子：客服工单自动分流。输入“退款未到账，已经等了 3 天”，标签体系是“退款、物流、账户异常、其他”。训练时先固定 `label2id`，再用 tokenizer 生成模型输入，最后用 `num_labels=4` 初始化分类模型。推理时输出的 id 必须再通过 `id2label` 转回业务标签。

---

## 工程权衡与常见坑

只训练分类头能跑，因为 BERT 的预训练表示已经有通用语义能力。但它通常不如全量微调稳定，原因是底层表示没有真正适配当前任务。对于“积极/消极”这种简单任务，只训分类头可能足够；对于行业文本、专业术语、标签边界细的任务，通常需要全量微调或至少解冻顶部几层。

训练策略对比：

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 只训分类头 | 冻结 BERT，只更新 Linear | 快，显存低 | 表示无法适配任务 |
| 全量微调 | 更新 BERT 和分类头 | 效果通常更好 | 显存高，需控制学习率 |
| 分层学习率 | 底层小学习率，顶层大学习率 | 更稳，适合精调 | 实现复杂度更高 |

常见坑如下：

| 常见坑 | 现象 | 原因 | 修正 |
|---|---|---|---|
| 把分类任务当 token 级任务 | 结果不稳定 | 取了最后一个 token 或每个 token | 取 `[CLS]` 或 pooler 输出 |
| 句对任务忘记 `token_type_ids` | A/B 关系学不好 | 模型难区分两段文本 | 使用 `[CLS] A [SEP] B [SEP]` 并传段落 id |
| padding 后 mask 错误 | 长度越长效果越差 | padding 被当作正文 | padding 位置 `attention_mask=0` |
| `num_labels` 不匹配 | loss 报错或预测错位 | 类别数和标签映射不一致 | 固定 `label2id`，同步 `num_labels` |
| 标签 id 每次重排 | 离线评估和线上推理不一致 | 训练与推理映射不同 | 保存并复用同一份映射 |
| 长文本直接截断 | 关键信息丢失 | 超过 `max_length` 的部分被切掉 | 分段编码或换长文本模型 |

`attention_mask` 的作用可以写成：

```text
真实 token: attention_mask = 1
padding:   attention_mask = 0
```

它告诉自注意力层哪些位置应该参与计算。自注意力是 Transformer 中让每个 token 读取其他 token 信息的机制，白话解释是：每个词会根据句子里其他词重新更新自己的表示。如果 padding 没有被 mask 掉，模型会把补齐位置也当成有效内容。

`num_labels` 必须和标签映射一致：

```text
len(label2id) == num_labels
```

如果 `label2id = {"退款": 0, "物流": 1, "账户异常": 2, "其他": 3}`，那么 `num_labels` 必须是 4。训练、评估、推理都必须使用同一套映射。

小数据集上还要注意过拟合。过拟合是模型在训练集表现很好，但在新数据上表现差。常见修正包括降低学习率、增加 dropout、早停、使用验证集选择最佳 checkpoint、合并过细标签、补充更均衡的数据。

类不平衡也会影响结果。例如 90% 样本都是“其他”，模型只预测“其他”也能得到高准确率，但业务上没有价值。此时应查看每类 precision、recall、F1，而不是只看 accuracy。

---

## 替代方案与适用边界

BERT 不是唯一选择。模型选择取决于任务规模、延迟要求、文本长度、部署环境和标注数据质量。

| 模型 | 特点 | 适用场景 | 边界 |
|---|---|---|---|
| BERT | 经典双向编码器，分类微调稳定 | 短文本分类、句对分类 | 推理不算轻，长文本受限 |
| RoBERTa | 改进预训练策略，通常效果更强 | 有更高精度需求的分类任务 | 资源开销通常不低 |
| DistilBERT | 蒸馏版 BERT，参数更少 | 低延迟线上服务 | 精度可能低于大模型 |
| TinyBERT | 更小的蒸馏模型 | 移动端、资源受限部署 | 复杂语义任务可能不足 |
| Longformer / BigBird | 支持更长上下文 | 长文档分类、报告分类 | 架构和部署复杂度更高 |

新手版判断：如果只是做一个简单情感分类，BERT 微调通常足够；如果要在超低延迟设备上部署，应该考虑 DistilBERT、TinyBERT 或传统轻量模型。

工程版判断：

| 任务类型 | 推荐做法 |
|---|---|
| 单标签多分类 | `BertForSequenceClassification` + softmax + cross entropy |
| 多标签分类 | 分类头输出多个 logits，使用 `BCEWithLogitsLoss` |
| 句对分类 | 使用 `[CLS] A [SEP] B [SEP]` 和 `token_type_ids` |
| 序列标注 | 使用 token 级输出，例如 `BertForTokenClassification` |

多标签任务不能直接套普通 softmax 交叉熵。因为 softmax 假设类别互斥，而多标签允许多个类别同时为真。多标签通常对每个类别独立计算 sigmoid：

$$
p_i = \sigma(z_i)
$$

然后使用 `BCEWithLogitsLoss`。例如一条客服投诉可以同时属于“退款”和“物流”，这时不能强迫模型只选一个类别。

适用边界清单：

| 场景 | 是否适合 BERT 微调 |
|---|---|
| 文本长度在 512 tokens 内 | 适合 |
| 标签体系固定 | 适合 |
| 有几百到几万条标注数据 | 通常适合 |
| 需要解释每个词的标签 | 不适合普通分类头，应用序列标注 |
| 文档很长且关键信息分散 | 需要分段、检索或长文本模型 |
| 延迟预算极低 | 考虑蒸馏模型、量化或传统模型 |
| 类别频繁新增 | 需要重新训练分类头或换检索/生成式方案 |

传统方法也有价值。TF-IDF + 线性分类器在小规模、强关键词、低延迟任务中仍然有效。TF-IDF 是一种根据词频和逆文档频率表示文本的方法，白话解释是：某个词在当前文本里常见、但在全部文本里不常见，它就更能代表当前文本。它不如 BERT 擅长上下文语义，但训练快、部署简单、可解释性强。

---

## 参考资料

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)：BERT 原始论文页面，用于理解预训练、句对输入和下游任务微调思想。
2. [Google Research BERT GitHub Repository](https://github.com/google-research/bert)：Google 官方 BERT 仓库，包含 fine-tuning 说明和原始实现入口。
3. [Google BERT modeling.py](https://raw.githubusercontent.com/google-research/bert/master/modeling.py)：原始模型实现，可对照 pooler 中 `[CLS] -> dense -> tanh` 的计算路径。
4. [Google BERT run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py)：原始分类微调脚本，可查看输入样本、`segment_ids`、`input_mask` 和分类头训练流程。
5. [Hugging Face BERT Documentation](https://huggingface.co/docs/transformers/en/model_doc/bert)：Hugging Face BERT 文档，用于落地 `BertForSequenceClassification`、输入字段和工程调用方式。

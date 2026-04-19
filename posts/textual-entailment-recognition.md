## 核心结论

文本蕴含识别，也叫自然语言推理（Natural Language Inference，NLI）或文本蕴含识别（Recognizing Textual Entailment，RTE），是在给定前提 `p` 和假设 `h` 时，判断 `p` 与 `h` 的关系属于 `entailment`、`contradiction` 还是 `neutral`。

前提可以理解为“已经告诉你的事实”，假设可以理解为“你要验证的说法”。如果事实能推出说法，就是蕴含；如果事实直接推翻说法，就是矛盾；如果事实既不能证明也不能反驳，就是中立。

| 标签 | 含义 | 判断标准 |
|---|---|---|
| `entailment` | 蕴含 | `p` 足以推出 `h` |
| `contradiction` | 矛盾 | `p` 与 `h` 不能同时成立 |
| `neutral` | 中立 | 既不能推出，也不能反驳 |

玩具例子：

| 前提 `p` | 假设 `h` | 标签 |
|---|---|---|
| 小明养了一只猫 | 小明养了动物 | `entailment` |
| 小明没有参加会议 | 小明参加了会议 | `contradiction` |
| 小明参加了会议 | 小明做了会议纪要 | `neutral` |

真实工程例子：在客服或 RAG 系统里，证据句是“订单已取消”，候选回答是“该订单仍然有效”，这两句话不能同时成立，应判为 `contradiction`。这个结果可以用于拦截不被证据支持的回答。

NLI 的核心价值不是判断两句话像不像，而是判断“前提是否足以支持假设成立”。这使它常用于事实校验、检索增强生成、客服质检、答案一致性检查等场景。

---

## 问题定义与边界

NLI 的形式化定义是：输入一对文本 $(p, h)$，输出标签 $y$：

$$
f(p, h) \rightarrow y,\quad y \in \{\text{entailment}, \text{contradiction}, \text{neutral}\}
$$

其中 `p` 是 premise，表示前提；`h` 是 hypothesis，表示假设。模型要学习的是两段文本之间的逻辑关系，而不是单段文本的主题。

这个任务本质上是监督学习中的句对分类问题。监督学习是指训练数据里已经给出了输入和正确答案，模型通过这些样本学习映射关系。句对分类是指每个样本包含两段文本，分类结果由两段文本共同决定。

| 任务 | 问什么 | 和 NLI 的区别 |
|---|---|---|
| 语义相似度 | 两句是否意思接近 | 不判断真假关系 |
| 问答 | 问题的答案是什么 | NLI 不生成答案 |
| 文本分类 | 这段话属于什么主题 | NLI 要比较两句之间关系 |
| 信息检索 | 哪些文本相关 | 相关不等于能推出 |

例子：`p = "小明没有参加会议"`，`h = "小明参加了会议"`。这两句话都在讲“会议”，主题很接近，但关系是直接矛盾。NLI 不是看它们是不是都谈同一件事，而是看第一句话能不能证明第二句话，或者能不能推翻第二句话。

SNLI 和 MultiNLI 是两个经典数据集。数据集是模型训练和评估使用的样本集合。SNLI 主要来自图片描述场景，句子较短；MultiNLI 覆盖更多文本领域，例如小说、电话转写、政府文本等，因此更接近跨领域测试。它们都把问题建模成三分类句对任务。

一个重要边界是：前提和假设的顺序不能随意交换。`p = "狗在草地上奔跑"`，`h = "动物在草地上奔跑"` 是蕴含；但反过来，`p = "动物在草地上奔跑"`，`h = "狗在草地上奔跑"` 通常只能判为中立，因为“动物”不一定是“狗”。

---

## 核心机制与推导

主流做法是把前提和假设拼接后送入编码器，再用句对表示做三分类。编码器是把文本转换成向量表示的模型，例如 BERT。向量表示是数字数组，模型用它承载词语、句子和上下文信息。

BERT 类模型通常使用如下输入格式：

$$
x = [CLS]\ p\ [SEP]\ h\ [SEP]
$$

`[CLS]` 是分类标记，常被用来汇总整段输入的信息；`[SEP]` 是分隔标记，用来区分前提和假设。编码器输出为：

$$
H = E(x)
$$

其中 $E$ 是编码器，$H$ 是每个 token 的上下文向量。取第一个位置的向量：

$$
z = H_0
$$

然后通过线性分类层得到三类分数：

$$
s = Wz + b
$$

最后用 softmax 把分数转成概率：

$$
\hat y = \arg\max \operatorname{softmax}(s)
$$

softmax 是把多个实数分数转换成概率分布的函数，分数越大，对应概率通常越高。

数值例子：设标签顺序为 `[蕴含, 矛盾, 中立]`，模型输出 logits：

$$
s = [2.0,\ 0.5,\ -1.0]
$$

softmax 后约为：

$$
[0.79,\ 0.18,\ 0.04]
$$

最大值对应第一类，所以预测为 `entailment`。

只用 `[CLS]` 分类可以得到很强的基线，但它不直接展示两句话之间哪些词在互相对应。为了更清楚地建模对齐关系，可以使用跨句子注意力。注意力机制是一种加权汇总方法，用来让模型决定当前 token 应该重点参考哪些其他 token。

设前提中的 token 向量为 $u_i$，假设中的 token 向量为 $v_j$。两者的匹配分数可以写成：

$$
e_{ij} = u_i^T W v_j
$$

再对假设侧所有 token 做归一化：

$$
\alpha_{ij} = \operatorname{softmax}_j(e_{ij})
$$

最后得到前提 token $u_i$ 对齐后的假设信息：

$$
\tilde u_i = \sum_j \alpha_{ij} v_j
$$

这表示：对前提里的每个词，模型显式计算它应该关注假设里的哪些词。比如在“订单已取消”和“订单仍然有效”中，“取消”和“有效”之间的冲突比“订单”和“订单”的重叠更关键。跨句子注意力能更直接地表达这种局部冲突。

从输入到输出的流程可以概括为：

```text
前提 p + 假设 h
        ↓
拼接为 [CLS] p [SEP] h [SEP]
        ↓
编码器生成上下文表示 H
        ↓
取 [CLS] 表示 z 或加入跨句子注意力
        ↓
线性层输出三类 logits
        ↓
softmax 得到概率
        ↓
选择概率最高的标签
```

---

## 代码实现

实现 NLI 时，最重要的是三件事：标签映射固定、前提和假设顺序固定、训练和推理使用同一套处理逻辑。

下面是一个不依赖深度学习框架的最小可运行 Python 例子，用 softmax 演示三分类预测过程。它不能替代真实 BERT 训练，但能准确说明 logits、概率和标签映射之间的关系。

```python
import math

label2id = {"entailment": 0, "contradiction": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]

def predict_label(logits):
    probs = softmax(logits)
    pred_id = max(range(len(probs)), key=lambda i: probs[i])
    return id2label[pred_id], probs

logits = [2.0, 0.5, -1.0]
label, probs = predict_label(logits)

assert label == "entailment"
assert len(probs) == 3
assert abs(sum(probs) - 1.0) < 1e-9
assert probs[0] > probs[1] > probs[2]

print(label, [round(p, 2) for p in probs])
```

真实工程中通常会使用 Transformers 库加载 BERT 类模型。训练和推理骨架如下：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label2id = {"entailment": 0, "contradiction": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    label2id=label2id,
    id2label=id2label,
)

premise = ["The order has been cancelled."]
hypothesis = ["The order is still active."]

batch = tokenizer(
    premise,
    hypothesis,
    truncation=True,
    padding=True,
    return_tensors="pt",
)

logits = model(**batch).logits
pred = logits.argmax(dim=-1)
print(id2label[pred.item()])
```

这里的 `tokenizer(premise, hypothesis, ...)` 会按模型要求把两段文本拼成句对输入。`truncation=True` 表示过长文本会被截断；`padding=True` 表示把同一批样本补齐到相同长度。`logits` 是模型输出的未归一化分数，训练时通常接交叉熵损失。交叉熵损失是分类任务常用的目标函数，用来惩罚模型把概率分给错误标签。

一个样本表可以长这样：

| `premise` | `hypothesis` | `label` | `label_id` |
|---|---|---:|---:|
| The order has been cancelled. | The order is still active. | contradiction | 1 |
| A dog is running. | An animal is moving. | entailment | 0 |
| A man is cooking. | A woman is reading. | neutral | 2 |

如果把训练数据看成一个索引表，那么每一行都必须能稳定找到前提、假设和标签。类似静态博客里的 `posts.json` 管理文章元数据，NLI 数据表也需要明确字段含义，否则模型训练时很容易把列读错。

---

## 工程权衡与常见坑

NLI 的难点不只在模型结构，也在数据和流程。很多模型在 SNLI 上准确率很高，但换到 MultiNLI 或真实业务文本后明显下降，原因通常是训练集分布和业务分布不同。数据分布是指样本来源、语言风格、句子长度、标签比例等统计特征。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| `label` 顺序写错 | 训练和评估对不上 | 固定 `label2id/id2label` |
| `premise/hypothesis` 位置颠倒 | 结果不稳定或错误 | 全流程保持输入顺序一致 |
| 过度依赖词重叠 | 看似高分，泛化差 | 用跨域数据验证 |
| `max_length` 太小 | 证据被截断 | 按任务调长或做分段 |
| 冻结 `[CLS]` 不微调 | 表达能力不足 | 优先端到端微调 |
| 只看整体准确率 | 少数类问题被掩盖 | 分标签看 precision、recall、F1 |

最常见的捷径是词面重叠。词面重叠是指两句话共享很多相同词。模型可能学会“重复词越多越像蕴含”，但这会误判很多矛盾样本。例如“订单已取消”和“订单未取消”只差一个否定词，重叠度很高，但标签相反。

是否需要跨句子注意力，取决于任务复杂度。短句、标签清晰、训练数据充足时，BERT cross-encoder 的 `[CLS]` 分类通常足够。cross-encoder 是把两段文本一起送入模型的结构，它能在编码阶段直接交换两句信息。若任务需要解释证据、定位冲突词、处理否定和数量变化，显式对齐机制更有价值。

在 RAG 系统里，单独用 NLI 分类器并不稳。更合理的流程是：先用检索系统找出高召回证据，再用 NLI 判断生成答案是否被证据支持。高召回是指尽量把可能相关的证据找全，即使混入一些噪声也可以接受。这样做比直接让分类器在全库里判断事实关系更可靠。

---

## 替代方案与适用边界

NLI 不是唯一方案。不同方法在精度、速度、解释性上有不同取舍。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Decomposable Attention | 结构清晰，解释性强 | 表达能力较弱 | 基础研究、教学 |
| ESIM | 对齐能力较好 | 训练和调参更复杂 | 中等规模任务 |
| BERT Cross-Encoder | 精度高 | 推理慢 | RAG 校验、事实验证 |
| Bi-Encoder / 向量检索 | 快速 | 关系建模弱 | 候选召回、粗筛 |

Decomposable Attention 是一种早期经典 NLI 模型，核心思想是先对齐、再比较、最后聚合。ESIM 使用 BiLSTM 和注意力对齐，能更细地建模句子内部和句子之间的关系。BiLSTM 是双向长短期记忆网络，会从左右两个方向读取序列。BERT cross-encoder 通常精度更强，因为前提和假设在 Transformer 层内可以充分交互。Transformer 是一种基于自注意力的神经网络结构，适合处理上下文依赖。

Bi-Encoder 是另一类常见方案，它把前提和假设分别编码成向量，再计算相似度。它速度快，适合大规模检索，但不擅长细粒度逻辑判断。比如“订单已取消”和“订单未取消”向量可能很接近，但 NLI 标签应是矛盾。

适合使用 NLI 的场景包括：事实一致性判断、证据支持判定、自动质检、摘要一致性检查、客服答案拦截。不适合的场景包括：开放式生成、复杂多跳推理、需要外部常识但证据不足的判断。

从句子级 NLI 扩展到段落级、文档级 NLI 时，边界会明显变化。长文本会带来证据定位、截断、跨段引用和多跳推理问题。此时通常不能简单把整篇文档塞进模型，而要先检索相关片段，再对片段和假设做 NLI 判断。

---

## 参考资料

| 类型 | 资料 | 用途 |
|---|---|---|
| 数据集 | SNLI | 经典三分类句对任务 |
| 数据集 | MultiNLI | 跨领域泛化评估 |
| 模型 | BERT | 强基线与主流实现 |
| 模型 | A Decomposable Attention Model for NLI | 经典对齐机制 |
| 实现 | Google Research BERT repo | 代码参考 |

1. [SNLI: A Large Annotated Corpus for Learning Natural Language Inference](https://aclanthology.org/D15-1075/)
2. [MultiNLI: A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://aclanthology.org/N18-1101/)
3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)
4. [Google Research BERT Repository](https://github.com/google-research/bert)
5. [A Decomposable Attention Model for Natural Language Inference](https://research.google/pubs/a-decomposable-attention-model-for-natural-language-inference/)

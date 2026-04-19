## 核心结论

多语言预训练模型的本质，是在多种语言的语料上联合训练同一个模型，让不同语言共享一套参数和一个表示空间。表示空间是模型内部向量所在的空间，意思相近的文本会被映射到相近的向量区域。

它的核心价值不是“模型会很多语言”，而是跨语言迁移：把一种语言上学到的任务能力，迁移到另一种语言上。最典型的场景是零样本跨语言迁移。零样本是指目标语言没有标注训练数据，模型仍然直接在目标语言上推理。

玩具例子：只用英文情感分类数据训练模型，例如 `I like this product` 标为正面、`This is terrible` 标为负面。测试时直接输入西语句子 `Me gusta este producto`，同一个模型仍然输出正面。这里没有先把西语翻译成英文，而是模型直接读取西语文本，并在共享表示空间里复用英文分类能力。

真实工程例子：跨语言工单分类系统只有英文标注数据，线上却收到西语、葡语、德语工单。工程上可以用 `mBERT` 或 `XLM-R` 在英文工单上微调，再把西语工单 `Necesito cambiar mi contraseña` 直接送入同一个模型，输出“修改密码”这类意图标签。

| 模型 | 是否使用平行语料 | 是否使用 MLM | 是否使用 TLM | 主要特点 |
|---|---:|---:|---:|---|
| `mBERT` | 否 | 是 | 否 | 在 104 种语言上联合训练，可做零样本迁移 |
| `XLM` | 可使用 | 是 | 是 | 引入翻译语言建模，用平行句增强对齐 |
| `XLM-R` | 否 | 是 | 否 | 使用更大规模多语言数据，低资源语言通常更强 |

`MLM` 是 masked language modeling，即遮住部分 token，让模型根据上下文预测被遮住内容。`TLM` 是 translation language modeling，即把平行句拼接后遮盖预测，让模型同时利用源语言和译文上下文。

一句话定义：多语言预训练模型是指在多语言语料上联合训练同一个编码器，让不同语言共享表示空间，从而把一种语言学到的任务能力迁移到另一种语言。

---

## 问题定义与边界

多语言预训练模型不是机器翻译系统。机器翻译的目标是生成另一种语言的文本；多语言预训练的目标是学习跨语言可复用的表示。编码器是把输入文本转换为向量表示的模型模块，例如 BERT 或 XLM-R 的主体网络。

需要区分三组概念。

| 概念 | 目标 | 训练数据 | 典型输出 |
|---|---|---|---|
| 多语言预训练 | 学共享表示 | 多语言单语语料，或少量平行语料 | 编码表示、分类、检索 |
| 机器翻译 | 跨语言生成 | 平行语料 | 目标语言文本 |
| 跨语言表示学习 | 表示对齐 | 单语或平行语料 | 共享向量空间 |

联合预训练是指把多种语言一起放进同一个训练过程，共享 tokenizer 和模型参数。显式对齐是指训练数据或训练目标直接告诉模型“这两个不同语言句子表达同一意思”。`mBERT` 和 `XLM-R` 主要依赖联合预训练；`XLM` 可以额外使用平行句对，通过 `TLM` 加强显式对齐。

新手版例子：如果有英文、中文、德语三种单语语料，把它们混在一起训练一个编码器，这叫多语言预训练。如果再额外把英文句子和法语翻译句子成对喂给模型，让模型同时看到两句之间的对应关系，这就是加入显式跨语言对齐信号。

反例：只有机器翻译系统，不等于多语言预训练模型。一个英译法模型的主要输出是法语文本，它可以有很强的翻译能力，但它的核心任务不是为分类、检索、问答等下游任务提供共享表示。

统一符号如下：

| 符号 | 含义 |
|---|---|
| $f_\theta$ | 共享编码器，$\theta$ 是模型参数 |
| $x$ | 单语输入序列 |
| $y$ | 与 $x$ 对应的翻译句 |
| $z=[x;y]$ | 把平行句拼接后的输入 |
| $h=f_\theta(x)$ | 输入文本对应的表示向量 |

多语言预训练讨论的是：如何让 $f_\theta$ 对不同语言输入都产生可复用的表示，而不是如何把 $x$ 翻译成 $y$。

---

## 核心机制与推导

核心训练目标是 `MLM`。给定一个输入序列 $x$，随机遮住其中一部分位置 $M$，模型要根据剩余上下文预测被遮住的 token。token 是模型处理文本的基本单位，可以是单词、子词或字符片段。

$$
L_{MLM}=-\sum_{i\in M}\log p_\theta(x_i \mid x_{\setminus M})
$$

其中，$M$ 表示被遮住的位置集合，$x_i$ 表示第 $i$ 个位置的真实 token，$x_{\setminus M}$ 表示遮盖后的上下文，$p_\theta$ 表示模型给正确 token 分配的概率。损失越小，说明模型越能根据上下文恢复原词。

玩具例子：英文句子 `The cat sits on the mat.` 中遮住 `cat`，输入变成 `The [MASK] sits on the mat.`。模型需要预测 `[MASK]` 位置最可能是 `cat`。如果模型给 `cat` 的概率是 $0.70$，这一项损失是 $-\log(0.70)\approx0.357$。

`XLM` 引入了 `TLM`。它把平行句拼接在一起，例如英文句子和法语翻译句一起输入模型，再对拼接后的序列做遮盖预测。

$$
L_{TLM}=-\sum_{i\in M}\log p_\theta(z_i \mid z_{\setminus M}),\quad z=[x;y]
$$

这里 $x$ 是源语言句子，$y$ 是目标语言翻译句，$z$ 是拼接后的序列。直观上，模型预测英文被遮住词时，不仅能看英文上下文，也能看法语翻译句中的语义线索。如果加入平行法语句后，正确词概率从 $0.70$ 提升到 $0.90$，损失从 $0.357$ 降到 $-\log(0.90)\approx0.105$，说明翻译句提供了有用的对齐信息。

机制流程可以写成：

```text
多语言文本
  -> 共享 tokenizer
  -> 共享 encoder f_theta
  -> MLM 预测被遮住 token
  -> 可选：TLM 使用平行句增强跨语言对齐
  -> 下游任务头：分类、检索、问答
```

为什么它能迁移，关键在三点。

第一，共享参数。英文、西语、中文都通过同一个 $f_\theta$，模型不能为每种语言完全单独记一套规则，只能在有限容量中学习可复用结构。

第二，共享或部分共享 tokenizer。tokenizer 是把原始文本切成 token 的组件。多语言模型通常使用同一套子词词表处理多种语言，使相似字符、共享词根、数字、标点、专有名词等能进入同一套输入系统。

第三，下游任务头只学习一次。假设分类头在英文表示上学到了一个决策边界：靠近区域 A 的向量是“正面”，靠近区域 B 的向量是“负面”。如果西语句子的表示也落入同一空间中相近区域，英文训练出的边界就可能直接复用。

零样本迁移可以写成：

$$
\theta^\*=\arg\min_\theta L_{task}^{src}(\theta)
$$

训练时只优化源语言任务损失 $L_{task}^{src}$，测试时直接把同一组参数 $\theta^\*$ 用到目标语言输入上，不再使用目标语言标注微调。

---

## 代码实现

实际工程中，通常不从零训练多语言预训练模型，而是加载已经训练好的 `mBERT` 或 `XLM-R`，再针对业务任务微调。微调是指在预训练模型基础上，用少量任务标注数据继续训练。

下面代码是可运行的最小玩具例子，用一个固定词表模拟“共享 tokenizer + 共享 encoder + 分类头”的结构。它不依赖外部模型，重点是展示跨语言共享表示如何让英文规则迁移到西语输入。

```python
import math

# 玩具版共享词表：英文和西语表达相近语义时，映射到相近特征。
LEXICON = {
    "like": (1.0, 0.0),
    "love": (1.0, 0.0),
    "gusta": (1.0, 0.0),
    "encanta": (1.0, 0.0),
    "bad": (0.0, 1.0),
    "terrible": (0.0, 1.0),
    "malo": (0.0, 1.0),
    "terrible_es": (0.0, 1.0),
}

def encode(text):
    tokens = text.lower().replace("terrible", "terrible terrible_es").split()
    pos, neg = 0.0, 0.0
    for token in tokens:
        if token in LEXICON:
            p, n = LEXICON[token]
            pos += p
            neg += n
    return (pos, neg)

def classify(text):
    pos, neg = encode(text)
    score = pos - neg
    prob_positive = 1 / (1 + math.exp(-score))
    return "positive" if prob_positive >= 0.5 else "negative"

# 英文规则：like/love 偏正面，bad/terrible 偏负面。
assert classify("I like this product") == "positive"
assert classify("This is bad") == "negative"

# 零样本跨语言：没有写西语训练流程，但共享表示让西语词进入同一特征空间。
assert classify("me gusta este producto") == "positive"
assert classify("producto malo") == "negative"
```

真实工程里会使用 `transformers` 加载模型。下面是结构示意，训练代码通常还需要 `Trainer`、数据集、评价指标和保存逻辑。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

K = 4
model_name = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=K,
)

text = "Necesito cambiar mi contraseña"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
logits = model(**inputs).logits
predicted_label_id = logits.argmax(dim=-1).item()
```

这段代码对应的真实工程例子是跨语言工单分类。训练阶段只用英文工单和标签，例如“reset password”“refund request”“shipping issue”。推理阶段直接输入西语或葡语文本，不需要目标语言标签。关键点是：同一个 tokenizer 处理多语言文本，同一个 encoder 输出共享表示，同一个分类头输出业务类别。

| 步骤 | 输入 | 输出 |
|---|---|---|
| tokenizer | 任意语言文本 | 子词 token |
| encoder | token 序列 | 共享表示向量 |
| 分类头 | 表示向量 | 意图标签 |
| 评测 | 分语言测试集 | 每种语言的准确率、召回率、F1 |

工程上不能只看整体准确率。至少要按语言拆分指标，因为一个多语言模型可能在英文、西语上很好，在低资源语言上明显变差。

---

## 工程权衡与常见坑

多语言效果不是只看“模型是否支持多语言”，而是由数据量、覆盖面、模型容量、语言相似度、任务分布共同决定。可以粗略理解为：

$$
效果 \approx g(\text{数据量}, \text{tokenizer覆盖}, \text{模型容量}, \text{语言距离}, \text{任务一致性})
$$

语言距离是指两种语言在词形、语序、文字系统、语法结构上的差异。英文到西语通常比英文到阿拉伯语、泰语这类差异更大的语言更容易迁移。

| 常见坑 | 原因 | 规避方式 |
|---|---|---|
| 误以为必须有平行语料 | 把 `XLM` 和 `mBERT` 混淆 | 先确认模型训练目标 |
| 语言越多越好 | 容量被稀释 | 看数据配比与模型容量 |
| 词表覆盖不足 | 子词切分过碎 | 检查 tokenizer 和脚本覆盖 |
| 论文高分直接可上线 | 数据分布不同 | 做真实线上回放测试 |
| 认为已完全对齐 | 远距语言仍有偏差 | 做分语言评测 |

新手版例子：论文里模型在标准西语测试集上表现不错，但线上西语工单可能是口语化文本，夹杂英文缩写、错别字和产品内部代号。标准测试集上的高分不能直接等于线上稳定效果。

另一个例子：中文和英文共享同一个模型时，如果 tokenizer 对中文切分很差，把常见词切成大量低频碎片，模型会更难学到稳定表示。模型参数再多，也会被输入质量限制。

真实工程中还要关注长文本。很多预训练编码器有最大长度限制，例如 512 token。工单、合同、客服对话超过长度后会被截断，截断位置如果刚好包含关键信息，分类结果会变差。解决方法包括滑窗切分、摘要后分类、层级编码，或者改用支持长上下文的模型。

还要注意标签体系是否跨语言一致。英文里的 `refund`、西语里的退款请求、葡语里的取消订单，在业务系统中可能被标成不同标签。如果标注标准本身不一致，模型迁移能力再强也会学到混乱边界。

---

## 替代方案与适用边界

多语言预训练不是唯一方案。它适合多语言分类、检索、相似度匹配、轻量问答等任务，但不保证在所有语言和所有业务域都最优。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 多语言预训练直接迁移 | 简洁、无翻译成本 | 受语言差异影响 | 多语言分类、检索 |
| 翻译后再推理 | 统一到高资源语言 | 翻译误差引入噪声 | 目标语言极低资源 |
| 单语言模型 | 简单稳定 | 无跨语言能力 | 只服务单一语言 |
| 任务特定多语微调 | 效果通常更高 | 需要更多标注 | 业务场景固定 |

新手版例子：你只有少量阿拉伯语数据，但英文标注很充足。第一步可以试 `XLM-R` 零样本迁移；如果指标不稳定，再考虑“阿拉伯语先翻译成英文，再用英文模型分类”。这不是退步，而是根据数据和任务性质选择更稳的路径。

真实工程例子：全球客服系统中，英文和西语流量大，阿拉伯语和印地语流量小。可以先用 `XLM-R` 建一个统一分类器，快速覆盖全部语言。上线验证后，如果发现某些低资源语言召回率不足，可以为这些语言增加少量标注做多语微调，或接入翻译后推理作为兜底链路。

当任务强依赖目标语言细粒度表达时，直接零样本迁移风险更高。例如法律文本、医疗文本、方言评论、讽刺检测、文化语境强的内容审核。这些任务不仅需要“语义大致相同”，还需要捕捉细微措辞差异。此时应优先做目标语言评测，必要时收集目标语言标注数据。

选择路径可以按这个顺序判断：

| 问题 | 倾向方案 |
|---|---|
| 只有一种语言 | 单语言模型 |
| 多语言任务相同，目标语言无标注 | 多语言预训练零样本 |
| 低资源语言零样本不稳 | 翻译后推理或少量标注微调 |
| 业务长期固定且流量充足 | 任务特定多语微调 |
| 输出必须是另一种语言文本 | 机器翻译或生成式多语言模型 |

结论是：多语言预训练提供的是强基线，不是最终答案。它让跨语言迁移从“必须为每种语言单独建模”变成“先用一个共享模型覆盖，再按风险补强”。

---

## 参考资料

1. [How multilingual is Multilingual BERT?](https://research.google/pubs/how-multilingual-is-multilingual-bert/)
2. [bert-base-multilingual-cased 模型卡](https://huggingface.co/google-bert/bert-base-multilingual-cased)
3. [Cross-lingual Language Model Pretraining](https://papers.nips.cc/paper_files/paper/2019/hash/c04c19c2c2474dbf5f7ac4372c5b9af1-Abstract.html)
4. [Cross-lingual pretraining sets new state of the art for natural language understanding](https://engineering.fb.com/2019/02/04/ai-research/cross-lingual-pretraining/)
5. [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/)

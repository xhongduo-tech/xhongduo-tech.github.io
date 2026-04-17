## 核心结论

BERT 的核心价值，不是“会写文章”，而是“会理解上下文”。这里的上下文，指一个词左边和右边同时出现的信息。BERT 采用 **Encoder-only Transformer**，也就是只保留 Transformer 编码器部分，让序列中的每个 token 都能同时关注两侧 token，因此学到的是**双向上下文表示**。

它的预训练主要依赖两个任务：

| 组件 | 作用 | 输入处理 | 输出类型 | 典型用途 |
|---|---|---|---|---|
| MLM | 让模型恢复被遮住的词 | 随机采样 15% token，并按 80/10/10 替换 | token 级表示 | NER、QA、序列标注 |
| NSP | 判断两句是否前后相连 | 句对输入，50% 正样本，50% 随机负样本 | `[CLS]` 句向量 | 句对分类、匹配 |
| `[CLS]` | 句子级聚合位 | 放在序列开头 | 单个向量 | 分类、排序、检索 |
| 普通 token | 词级语义单元 | 保留原词序 | 每个位置一个向量 | 实体识别、抽取 |

玩具例子最直观：

原句是“我 爱 吃 苹果”，如果把“吃”遮住，输入变成“我 爱 `[MASK]` 苹果”，BERT 会结合“爱”和“苹果”两个方向的信息，预测这个位置最可能是“吃”。这说明它学到的不是孤立词表，而是“一个词在完整上下文里的含义”。

因此，BERT 非常适合**判别任务**，也就是“给定输入，做判断”。例如情感分类、命名实体识别、阅读理解、句对匹配。它不适合直接做左到右文本生成，因为双向注意力会在训练时看到未来词，破坏自回归生成所需的因果约束。

---

## 问题定义与边界

BERT 解决的问题，可以概括为一句话：**如何让模型在预训练阶段就学到能感知左右上下文的表示，并把这种表示迁移到下游任务。**

这里的“表示”，可以理解成模型内部对一个词或一句话的数字化理解结果。传统词向量常常给“苹果”一个固定向量，但 BERT 会根据上下文区分：

- “我买了苹果”中的“苹果”更接近水果
- “我买了苹果电脑”中的“苹果”更接近公司品牌

这就是双向上下文表示的价值。

但边界也要说清楚。BERT 不是通用生成模型，它不能像 GPT 那样自然地一个词接一个词往后写。原因是 BERT 的 self-attention 是全可见的，每个位置都能看到两侧 token，而自回归生成要求位置 $t$ 只能看见 $1 \dots t-1$。

下面这张表可以快速判断任务是否适合 BERT：

| 任务类型 | 是否适合 BERT | 备注 |
|---|---|---|
| 文本分类 | 适合 | 直接取 `[CLS]` 向量接线性层 |
| 句对匹配 | 适合 | 如相似度、自然语言推断 |
| 命名实体识别 | 适合 | 使用每个 token 的输出向量 |
| 阅读理解抽取 | 适合 | 预测答案起止位置 |
| 文本续写 | 不适合 | 不能 autoregressive 生成 |
| 对话生成 | 不适合 | 需要 decoder 或 encoder-decoder |
| 翻译 | 一般不优先 | 更适合 T5、BART 一类结构 |

真实工程例子：做电商评论情感分类时，输入可能是“物流很快，但是包装破损”，常见做法是在序列前加入 `[CLS]`，把最终的 `[CLS]` 表示送入一个线性层加 softmax，输出“正面/负面/中性”概率。这个流程本质上不是让 BERT 生成文字，而是让它基于整句语义做判别。

---

## 核心机制与推导

BERT 的预训练包含两个目标：**MLM** 和 **NSP**。

### 1. MLM：掩码语言模型

MLM 的意思是“把部分词遮住，再让模型猜回来”。“掩码”这个词的白话解释，就是在输入里故意挖空，让模型补空。

设原始序列为 $x_1, x_2, \dots, x_n$，从中采样一部分位置组成集合 $M$，通常约占 15%。模型只在这些被采样的位置上计算预测损失：

$$
L_{\text{MLM}} = \sum_{i \in M} -\log P(x_i \mid x_{1..n \setminus M})
$$

这表示：对每个被挑中的位置 $i$，模型需要根据其余上下文，给出原词 $x_i$ 的概率，概率越高，损失越小。

关键不只是“遮住 15%”，而是**怎么遮**。BERT 使用经典的 80/10/10 策略：

| 被选中的 token | 处理方式 | 比例 | 目的 |
|---|---|---|---|
| 替换成 `[MASK]` | 显式挖空 | 80% | 让模型学会补全 |
| 替换成随机词 | 制造噪声 | 10% | 防止过度依赖 `[MASK]` |
| 保持不变 | 看似未改动 | 10% | 减轻训练与推理分布差异 |

玩具例子：

原句：`我 爱 吃 苹果`

如果“吃”被采样到，那么三种可能输入是：

- 80%：`我 爱 [MASK] 苹果`
- 10%：`我 爱 咸 苹果`
- 10%：`我 爱 吃 苹果`

无论输入变成哪一种，监督信号都还是“这个位置原词应当是‘吃’”。这一步非常重要，因为下游任务和真实推理时通常不会出现 `[MASK]`，如果训练中 100% 依赖 `[MASK]`，模型就会学到一种不符合部署环境的习惯。

### 2. NSP：下一句预测

NSP 的意思是“给模型两句话，让它判断第二句是不是第一句的真实后续”。“句对”这个词的白话解释，就是把两句话拼成一个训练样本。

典型构造方式：

- 50% 概率使用真实相邻句子，标签是 `IsNext`
- 50% 概率随机抽一句不相关句子，标签是 `NotNext`

输入格式通常是：

`[CLS] 句子A [SEP] 句子B [SEP]`

然后取 `[CLS]` 的最终表示，送入一个二分类层。

用公式表示，若句对标签为 $y \in \{0,1\}$，预测概率为 $\hat y$，则：

$$
L_{\text{NSP}} = - \big(y \log \hat y + (1-y)\log(1-\hat y)\big)
$$

最终总损失通常写成：

$$
L = L_{\text{MLM}} + L_{\text{NSP}}
$$

### 3. 双向表示为什么有用

Transformer 的 self-attention 会让每个 token 聚合其他位置的信息。BERT 不使用因果 mask，所以位置 $i$ 可以看到左边和右边所有 token。于是“银行”在“河岸边的银行”和“去银行开户”里会得到不同向量。

这类表示对判别任务直接有价值：

- 分类任务依赖整句语义，取 `[CLS]`
- NER 依赖词与前后词关系，取各 token 向量
- QA 依赖问题与段落的跨位置对齐，预测答案 span

但也正因如此，它不能直接按生成方式工作，因为生成时第 5 个词不能提前看到第 6 个词。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，演示 MLM 数据构造与一个最小版句子分类头。这里不实现完整 Transformer，只展示 BERT 训练管线中最关键的两步：**80/10/10 掩码替换** 和 **`[CLS]` 分类头**。

```python
import random
import math

MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"

def build_mlm_example(tokens, vocab, mask_prob=0.15, seed=7):
    rng = random.Random(seed)
    input_tokens = tokens[:]
    labels = [None] * len(tokens)
    candidate_positions = list(range(len(tokens)))

    for i in candidate_positions:
        if rng.random() < mask_prob:
            original = tokens[i]
            labels[i] = original

            p = rng.random()
            if p < 0.8:
                input_tokens[i] = MASK_TOKEN
            elif p < 0.9:
                # 换成随机词，但尽量不要刚好等于原词
                choices = [w for w in vocab if w != original]
                input_tokens[i] = rng.choice(choices)
            else:
                # 保持原样
                input_tokens[i] = original

    return input_tokens, labels

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def cls_classifier(cls_hidden, weight, bias):
    # 线性层: logits = cls_hidden @ weight + bias
    logits = []
    for j in range(len(bias)):
        score = sum(cls_hidden[i] * weight[i][j] for i in range(len(cls_hidden))) + bias[j]
        logits.append(score)
    return softmax(logits)

# 玩具例子
tokens = ["我", "爱", "吃", "苹果"]
vocab = ["我", "爱", "吃", "苹果", "咸", "香蕉", "跑"]

masked_tokens, labels = build_mlm_example(tokens, vocab, mask_prob=1.0, seed=1)
assert len(masked_tokens) == len(tokens)
assert len(labels) == len(tokens)
assert all(label in vocab for label in labels if label is not None)

# 句子级分类头示意: [CLS] 向量 -> 二分类
cls_hidden = [0.2, -0.1, 0.7]
weight = [
    [0.5, -0.3],
    [0.1, 0.2],
    [0.4, 0.6],
]
bias = [0.0, 0.1]

probs = cls_classifier(cls_hidden, weight, bias)
assert len(probs) == 2
assert abs(sum(probs) - 1.0) < 1e-9

print("masked_tokens =", masked_tokens)
print("labels =", labels)
print("class_probs =", probs)
```

如果把它映射回真实 BERT 训练流程，可以理解为：

1. 先把原始文本切成 token。
2. 随机选 15% 位置构造 MLM 样本。
3. 经过嵌入层、位置编码和多层 self-attention。
4. 被掩码位置走 MLM 预测头。
5. `[CLS]` 位置走句子级分类头，如 NSP 或下游分类任务。

伪代码可以写成：

```python
tokens = [CLS] + sent_a + [SEP] + sent_b + [SEP]
hidden = bert_encoder(tokens)

mlm_logits = mlm_head(hidden[masked_positions])
cls_logits = cls_head(hidden[0])  # hidden[0] 对应 [CLS]

loss = mlm_loss(mlm_logits, mlm_labels) + nsp_loss(cls_logits, nsp_label)
```

真实工程例子：做中文客服意图分类时，输入“我要修改收货地址”，推理流程通常是：

- 分词并加 `[CLS]`
- 送入预训练 BERT
- 取最后一层 `[CLS]` 向量
- 经过线性层输出“退款/改地址/催发货”等类别概率

这类任务往往样本不算大，但 BERT 预训练已经学到大量语言模式，因此微调速度通常比从零训练快很多。

---

## 工程权衡与常见坑

BERT 的设计很经典，但工程落地时有几个点必须清楚。

| 常见坑 | 现象 | 原因 | 缓解方法 |
|---|---|---|---|
| 误把 BERT 当生成模型 | 续写效果差 | 结构不是自回归 | 改用 GPT、T5、BART |
| 过度依赖 NSP | 指标不升反降 | NSP 信号常较弱 | 直接去掉，或改用更强句间目标 |
| `[MASK]` 训练分布偏差 | 预训练好，下游泛化一般 | 推理时没有 `[MASK]` | 保留 10% 随机词和 10% 不变 |
| `[CLS]` 直接拿来用效果不稳 | 句向量质量波动 | 不同任务对 pooling 敏感 | 比较 `[CLS]`、mean pooling、task head |
| 长文本截断 | 关键信息丢失 | BERT 长度上限固定 | 滑窗、分块、长文本变体 |

### 1. NSP 不一定有用

早期 BERT 把 NSP 作为标准配置，但后续很多工作发现，NSP 并不是效果提升的稳定来源。有些情况下，它甚至会带来额外噪声。原因很简单：随机拼句得到的负样本太容易，模型学到的可能只是主题差异，而不是深层句间关系。

因此像 RoBERTa 这类后续模型，直接去掉了 NSP，只保留更强的数据规模和 MLM 训练，效果反而更好。这说明工程里不要把“原论文用了”理解成“永远必须保留”。

### 2. 预训练和推理分布不完全一致

MLM 有一个天然问题：预训练时看到 `[MASK]`，真实任务中通常看不到。这会造成输入分布错位。80/10/10 策略就是一种折中方案，不是为了好看，而是为了减少这个偏差。

### 3. `[CLS]` 不是魔法按钮

很多初学者会以为 `[CLS]` 一定代表“最佳句向量”。更准确的说法是：它是**为句子级任务预留的聚合位置**，是否足够好，要看预训练目标和具体任务。有些检索或相似度任务里，平均池化所有 token 反而更稳定。

### 4. 小数据集容易“会背不会泛化”

BERT 参数量大，在样本很少时容易过拟合。工程上常见做法包括：

- 冻结部分底层参数
- 使用更小学习率
- 减少训练轮数
- 选用 DistilBERT 这类轻量版本

---

## 替代方案与适用边界

如果任务目标不是“理解”，而是“生成”，就不该继续强行用 BERT。结构选型应当服从任务约束。

| 模型 | 结构类型 | 是否擅长生成 | 典型场景 |
|---|---|---|---|
| BERT | Encoder-only | 否 | 分类、NER、抽取式 QA |
| RoBERTa | Encoder-only | 否 | 更强的判别任务基线 |
| DistilBERT | Encoder-only | 否 | 轻量部署、低延迟 |
| ALBERT | Encoder-only | 否 | 参数共享、节省显存 |
| GPT | Decoder-only | 是 | 续写、对话、代码生成 |
| T5 | Encoder-decoder | 是 | 翻译、摘要、生成式问答 |
| BART | Encoder-decoder | 是 | 文本重写、摘要、纠错 |

这里给一个对比思路。

还是句子“我 爱 吃 苹果”。

- 用 BERT 时，任务通常是“判断这句话情感是否正面”或“识别‘苹果’是不是实体”。
- 用 T5 时，任务可以写成“翻译: 我 爱 吃 苹果”，然后输出英文句子。
- 用 GPT 时，输入可能是“我 爱”，模型继续生成“吃 苹果”。

也就是说：

- **BERT** 适合输入到标签
- **GPT** 适合输入到后续文本
- **T5/BART** 适合输入到目标文本序列

如果你做的是信息抽取、文本分类、实体识别，BERT 仍然是一个清晰且高效的起点。如果你要做开放式问答、长文本续写、对话生成，那么应直接换成生成架构，而不是在 BERT 上补丁式扩展。

---

## 参考资料

- Devlin 等，BERT 原始论文《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》。核心内容：提出 encoder-only 预训练框架，使用 MLM 与 NSP 学习双向表示。
- Aalto 教学文档。核心内容：解释 BERT 的双向上下文、`[CLS]` 用法、MLM 与 NSP 的训练流程。
- Michael Brenndoerfer 的 BERT 预训练说明。核心内容：细化 15% token 采样及 80/10/10 替换策略，并说明 cross-entropy 如何恢复原词。
- BERT-pytorch 相关实现文档。核心内容：展示 MLM/NSP 的数据构造与训练头实现方式。
- RoBERTa 相关资料。核心内容：说明去掉 NSP、扩大训练数据与训练强度后，判别任务性能通常更强。
- Azure 模型目录中的 BERT 说明。核心内容：总结 BERT 在分类、NER、QA 等判别任务中的常见用法。

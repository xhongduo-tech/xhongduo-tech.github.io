## 核心结论

ELMo（Embeddings from Language Models）是一种上下文相关词表示方法：它不再给每个词分配一个固定向量，而是根据词所在的完整句子生成当前语境下的向量。

Word2Vec、GloVe 这类静态词向量的假设是：一个词对应一个稳定表示。这个假设在很多任务中有效，但遇到一词多义时会失效。`Apple released a new chip` 中的 `Apple` 更可能是公司，`eat an apple` 中的 `apple` 是水果。词本身没有变，变的是词在句子里的身份，所以表示也应该改变。

ELMo 的核心做法是：先用字符级模型得到词的底层表示，再用双向 LSTM 语言模型读取上下文，最后把多层表示按任务需要加权求和。对句子中第 $k$ 个词，ELMo 不是输出一个全局固定向量，而是输出：

$$
\mathrm{ELMo}_k^{(task)} = \gamma^{(task)} \sum_{j=0}^{L} s_j^{(task)} \mathbf{h}_{k,j}
$$

其中 $\mathbf{h}_{k,j}$ 是第 $j$ 层对第 $k$ 个词的表示，$s_j$ 是任务学到的层权重，$\gamma$ 是缩放系数。

| 维度 | Word2Vec | ELMo |
|---|---|---|
| 表示类型 | 静态词向量 | 上下文相关词向量 |
| 同一个词 | 总是同一个向量 | 不同句子中向量不同 |
| 上下文信息 | 训练后不再变化 | 编码时动态引入 |
| 典型问题 | 难处理一词多义 | 推理成本更高 |
| 工程角色 | 词表特征 | 上下文特征生成器 |

ELMo 的重要性不只在于某个模型结构，而在于它把 NLP 表示学习从“查表得到词向量”推进到“根据上下文生成词向量”。这个范式后来被 BERT 等预训练语言模型进一步放大。

---

## 问题定义与边界

ELMo 解决的问题是：静态 embedding 无法根据上下文改变词表示。embedding 是把离散符号映射成连续向量的技术，白话说就是把词变成模型能计算的一串数字。静态 embedding 只给每个词一个向量，因此 `bank` 在 `river bank` 和 `bank loan` 中只能共用同一个表示。

ELMo 的边界也要说清楚。它不是今天意义上的生成式大模型，不负责直接生成长文本；它也不是完全替代下游模型的端到端系统。更准确的定位是：ELMo 是一个预训练上下文特征生成器，下游任务仍然需要分类器、序列标注器、问答模型等任务结构。

几个术语先定义：

| 术语 | 白话解释 |
|---|---|
| `token` | 模型处理的最小文本单元，可以是词、子词或符号；在 ELMo 论文语境中通常按词处理 |
| `contextualized representation` | 上下文化表示，即同一个词会根据前后文得到不同向量 |
| `biLM` | bidirectional Language Model，双向语言模型，同时利用左侧上下文和右侧上下文预测词 |
| OOV | out-of-vocabulary，词表外词，即训练词表中没有出现过的词 |

| 静态 embedding 的局限 | ELMo 的解决方式 | ELMo 不解决的事情 |
|---|---|---|
| 一词多义只能共享一个向量 | 用双向语言模型根据上下文生成表示 | 不直接替代全部任务模型 |
| 对词形变化不敏感 | 用字符卷积建模词形 | 不保证理解所有长距离推理 |
| 不能在推理时读取整句语境 | 编码句子时读取左右上下文 | 不像生成式模型那样直接生成答案 |
| 任务差异无法改变词向量层级 | 用 scalar mix 学习层权重 | 不自动消除数据偏差 |

玩具例子：`bank` 在 `sit near the river bank` 中靠近 `river`，ELMo 会把它表示成“河岸”语义；在 `apply for a bank loan` 中靠近 `loan`，ELMo 会把它表示成“银行”语义。新手可以把它理解成：ELMo 先读完句子，再决定这个词在当前句子里扮演什么角色。

真实工程例子：命名实体识别（NER）要判断词是不是公司名、地名、人名等。`Apple` 在科技新闻中常是组织名，在食谱文本中常是普通名词。如果只用固定词向量，模型必须靠后续网络自己修正歧义；如果输入前已经接入 ELMo，送入 CRF 或 BiLSTM-CRF 前的词向量已经带有上下文信息，低资源场景下通常更稳。

---

## 核心机制与推导

ELMo 的输入和输出可以分成六步：

```text
字符输入
  ↓
字符卷积 CNN
  ↓
词的底层表示 h_{k,0}
  ↓
多层双向 LSTM
  ↓
各层上下文化表示 h_{k,1}, ..., h_{k,L}
  ↓
scalar mix 加权求和
  ↓
下游任务模型
```

字符卷积是对词内部字符进行卷积建模的模块，白话说就是让模型看到 `playing`、`played`、`plays` 这些词形之间的关系，而不只是把它们当作完全无关的词。LSTM 是一种循环神经网络结构，适合按顺序读取文本；双向 LSTM 则同时从左到右和从右到左读句子。

对第 $k$ 个词、第 $j$ 层，ELMo 把前向语言模型和后向语言模型的隐状态拼接起来：

$$
\mathbf{h}_{k,j} = [\overrightarrow{\mathbf{h}}_{k,j}; \overleftarrow{\mathbf{h}}_{k,j}]
$$

这里 $\overrightarrow{\mathbf{h}}_{k,j}$ 来自左到右的 LSTM，主要包含词左侧上下文；$\overleftarrow{\mathbf{h}}_{k,j}$ 来自右到左的 LSTM，主要包含词右侧上下文。拼接后的 $\mathbf{h}_{k,j}$ 同时拥有左右信息。

ELMo 的预训练目标是双向语言模型目标：

$$
\mathcal{L}=\sum_{k=1}^{N}\left(\log p(t_k\mid t_{<k})+\log p(t_k\mid t_{>k})\right)
$$

意思是：对每个位置 $k$ 的词 $t_k$，前向模型根据左边词 $t_{<k}$ 预测它，后向模型根据右边词 $t_{>k}$ 预测它。模型为了预测准确，必须学习语法、词义、搭配关系和上下文线索。

下游任务不直接拿最后一层，而是学习一个层级加权和：

$$
\mathrm{ELMo}_k^{(task)} = \gamma^{(task)} \sum_{j=0}^{L} s_j^{(task)} \mathbf{h}_{k,j}
$$

其中：

$$
s_j^{(task)}=\frac{e^{a_j}}{\sum_{i=0}^{L}e^{a_i}}
$$

$s_j$ 是 softmax 后的权重，所有层权重加起来为 1。$\gamma$ 是整体缩放系数，用来调整 ELMo 表示进入任务模型时的强度。

最小数值推导如下。假设某个词有 3 层表示：

$$
\mathbf{h}_{k,0}=[1,0],\quad \mathbf{h}_{k,1}=[0,2],\quad \mathbf{h}_{k,2}=[3,1]
$$

任务学到的权重是：

$$
s=[0.2,0.3,0.5],\quad \gamma=1
$$

则最终表示为：

$$
\mathrm{ELMo}_k=0.2[1,0]+0.3[0,2]+0.5[3,1]=[1.7,1.1]
$$

这个例子说明两件事。第一，最终向量不是某一层的输出，而是多层信息的组合。第二，不同任务可以学习不同的 $s_j$，比如句法任务可能更依赖低层，语义任务可能更依赖高层。新手可以理解为：模型不是只看“最后答案”，而是从字形、句法、语义等不同深度的信息里按比例取用。

---

## 代码实现

工程上使用 ELMo 通常分三步：加载预训练模型、提取句子的上下文表示、把表示接入下游任务模型。常见做法是冻结 ELMo 主体参数，只训练任务头和 scalar mix，这样计算成本更可控，也能减少小数据集上的过拟合。

代码结构可以抽象成：

```text
load_elmo()
encode_sentence()
concat_or_mix_layers()
task_head()
```

下面是一个可运行的 Python 玩具实现，不依赖真实 ELMo 权重，只演示 scalar mix 的核心计算。真实框架中的 ELMo 会把 `layers` 换成字符卷积和双向 LSTM 输出。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def scalar_mix(layers, logits, gamma=1.0):
    """
    layers: 多层表示，例如 [[1,0], [0,2], [3,1]]
    logits: 可学习层权重的原始分数
    gamma: 整体缩放系数
    """
    weights = softmax(logits)
    dim = len(layers[0])
    mixed = [0.0] * dim

    for weight, vector in zip(weights, layers):
        for i, value in enumerate(vector):
            mixed[i] += weight * value

    return [gamma * x for x in mixed], weights

layers = [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]

# 直接使用接近 [0.2, 0.3, 0.5] 的 softmax logits
target_weights = [0.2, 0.3, 0.5]
logits = [math.log(x) for x in target_weights]

mixed, weights = scalar_mix(layers, logits, gamma=1.0)

assert all(abs(a - b) < 1e-9 for a, b in zip(weights, target_weights))
assert all(abs(a - b) < 1e-9 for a, b in zip(mixed, [1.7, 1.1]))

print(mixed)
```

接入下游任务时，可以把 ELMo 当作“特征生成器”。以 NER 为例：

```python
def load_elmo():
    return "pretrained_elmo"

def encode_sentence(elmo, tokens):
    # 真实实现会返回每个 token 的多层上下文化向量
    return {
        "Apple": [[0.2, 0.1], [0.8, 0.4], [1.2, 0.7]],
        "chip": [[0.1, 0.3], [0.4, 0.8], [0.5, 1.1]],
    }

def concat_or_mix_layers(layer_vectors):
    logits = [0.0, 0.5, 1.0]
    mixed, _ = scalar_mix(layer_vectors, logits, gamma=1.0)
    return mixed

def task_head(token_vectors):
    # 真实任务中这里可以是 BiLSTM、MLP、CRF 等
    return ["B-ORG", "O"]

tokens = ["Apple", "chip"]
elmo = load_elmo()
all_layers = encode_sentence(elmo, tokens)
features = [concat_or_mix_layers(all_layers[token]) for token in tokens]
predictions = task_head(features)

assert predictions == ["B-ORG", "O"]
```

| 参数部分 | 训练时是否更新 | 常见原因 |
|---|---:|---|
| 字符卷积层 | 通常冻结 | 保留预训练词形知识，减少训练成本 |
| 双向 LSTM 主体 | 通常冻结 | 参数量大，小数据集微调容易过拟合 |
| scalar mix 权重 | 通常训练 | 让任务决定用哪几层 |
| $\gamma$ 缩放系数 | 通常训练 | 调整 ELMo 特征强度 |
| 任务头 | 必须训练 | 学习具体任务标签或预测目标 |

真实工程例子中，一个 NER 系统可以先用 ELMo 编码 `Apple released a new chip`，此时 `Apple` 的表示已经包含 `released`、`chip` 等上下文线索；再把这个表示送入序列标注模型，输出 `B-ORG`。在 `I ate an apple` 中，`apple` 的上下文不同，ELMo 输出也不同，任务头更容易判断它不是组织名。

---

## 工程权衡与常见坑

ELMo 的最大价值在于上下文歧义强、标注数据有限、词形变化明显的任务。它比静态词向量更懂句子，但代价是计算更重、推理更慢、部署更复杂。

| 取舍项 | 使用 ELMo 的收益 | 使用 ELMo 的代价 |
|---|---|---|
| 准确率 | 一词多义、NER、问答等任务可能提升明显 | 对简单任务提升可能有限 |
| 速度 | 表示质量高 | 双向 LSTM 推理慢于查表 |
| 显存 | 可复用预训练知识 | 多层隐状态占用更多内存 |
| 可迁移性 | 小数据任务更容易受益 | 领域差异大时仍可能需要适配 |
| 实现复杂度 | 可作为特征模块接入 | 依赖预训练权重和额外编码流程 |

常见坑如下：

| 常见坑 | 问题 | 规避方式 |
|---|---|---|
| 把 ELMo 当静态 embedding | 预先给每个词存一个固定向量会丢掉核心价值 | 每次按句子编码，保留上下文差异 |
| 只取顶层表示 | 顶层不一定对所有任务最好 | 使用 scalar mix 让任务学习层权重 |
| 忽略字符卷积 | OOV 和词形变化能力下降 | 保留字符级输入和预训练字符层 |
| 不做任务自适应加权 | 不同任务被迫使用同一层组合 | 训练 $s_j$ 和 $\gamma$ |
| 在小数据上全量微调 | 容易过拟合，训练成本高 | 先冻结 ELMo，只训练任务头 |

“层越高越好”是一个常见误解。低层表示通常更接近词形和句法，高层表示通常更接近语义和上下文抽象。不同任务需要的信息不同，所以不能机械地只拿最后一层。

真实工程中，NER 对大小写、词形、局部上下文很敏感，字符层和低层信息很有价值；问答任务更依赖语义匹配，高层信息可能更重要。ELMo 的 scalar mix 正是为这种差异设计的。

部署时还要考虑延迟。如果一个线上检索系统只需要极低延迟的粗召回，静态向量查表可能更合适。如果一个离线文档理解任务需要处理大量歧义实体，ELMo 或更现代的上下文化模型更有意义。选型不应只看模型是否先进，而要看任务是否真的需要上下文敏感表示。

---

## 替代方案与适用边界

ELMo 是上下文化表示的经典方案，但不是所有任务的默认最优解。它处在静态词向量和 Transformer 预训练模型之间：比 Word2Vec/GloVe 更能处理上下文，比 BERT 更早、更轻一些，但表达能力和并行效率不如现代 Transformer。

| 方法 | 表示方式 | 优点 | 代价 | 适用场景 |
|---|---|---|---|---|
| Word2Vec/GloVe | 静态词向量 | 快、简单、易部署 | 无法区分一词多义 | 低延迟检索、传统特征工程、资源极受限任务 |
| ELMo | 双向 LSTM 上下文化表示 | 能处理上下文歧义，字符层缓解 OOV | 推理比静态向量慢 | NER、问答、低资源序列标注 |
| BERT | 双向 Transformer 表示 | 表达能力强，预训练范式更成熟 | 参数更大，部署成本更高 | 文本分类、语义匹配、阅读理解、复杂语义任务 |
| 领域专用模型 | 在领域语料上预训练或微调 | 更贴近业务文本 | 需要数据和训练成本 | 医疗、法律、金融等专业领域 |

选择决策可以按下面的顺序判断：

```text
是否需要区分同词多义？
  ├─ 否：优先考虑 Word2Vec / GloVe / 轻量特征
  └─ 是
      ↓
是否能接受更高推理延迟？
  ├─ 否：考虑静态向量 + 上下文特征工程
  └─ 是
      ↓
是否已有可用 BERT 类模型和部署资源？
  ├─ 是：优先评估 BERT 或领域 Transformer
  └─ 否：ELMo 是可解释、经典、相对直接的上下文化方案
```

玩具例子中，`Apple`、`bank` 这类词展示了 ELMo 的必要性。真实工程中，客户服务工单、医学实体识别、新闻事件抽取都可能出现大量上下文歧义。比如“放电”在电池文本和医学文本中含义不同，固定词向量很难单独处理这种差异。

边界总结：ELMo 是上下文化表示的经典方案，适合理解预训练语言模型如何从“词表查表”走向“按上下文编码”，但它不是所有任务的默认最优解。

---

## 参考资料

| 资料名称 | 能解决的问题 | 推荐阅读顺序 |
|---|---|---:|
| ELMo 原始论文 | 理解模型目标、结构和实验结论 | 1 |
| AllenNLP `Elmo` 模块文档 | 理解工程接口和模块输入输出 | 2 |
| AllenNLP `elmo` 命令文档 | 理解如何抽取 ELMo 表示 | 3 |
| ELMo-BiDAF 项目页 | 理解 ELMo 如何接入问答系统 | 4 |

1. [Deep Contextualized Word Representations](https://a11y2.apps.allenai.org/paper?id=3febb2bed8865945e7fddc99efd791887bb7e14f)
2. [AllenNLP Elmo Module](https://docs.allennlp.org/main/api/modules/elmo/)
3. [AllenNLP elmo Command](https://docs.allennlp.org/v0.9.0/api/allennlp.commands.elmo.html)
4. [ELMo-BiDAF Project Page](https://gallery.allennlp.org/project/bidaf_elmo)

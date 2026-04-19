## 核心结论

命名实体识别，简称 NER，是从一段文本中找出人名、组织名、地名、产品名等实体，并判断它们类别的序列标注任务。NER 的特征工程，本质是把“词长什么样、前后是什么、像不像地名/人名/机构名”这类人工可解释信号，转成模型能直接使用的输入。

传统 NER 不让模型直接“读懂”整句 `Apple 收购 Beats`，而是先告诉模型：`Apple` 首字母大写、出现在句首、可能命中组织名词典、后面跟着动词 `收购`；再由 CRF 这类序列标注模型综合判断它更像 `ORG`，也就是组织名。

在传统 NER 中，特征工程不是附属步骤，而是决定模型上限的核心部分。尤其在标注数据少、领域词多、实体长尾严重的场景中，词形特征、词典特征、上下文窗口、词性标签、标签转移特征仍然有明确价值。即使用 BiLSTM-CRF 或 Transformer，词典特征也可以作为额外先验增强模型。

| 特征类型 | 作用 | 典型用途 |
|---|---|---|
| 词形特征 | 描述 token 自身长什么样 | 大小写、数字、连字符、前后缀 |
| 上下文特征 | 描述 token 前后出现了什么词 | 判断 `Apple` 是公司还是水果 |
| 词典特征 | 判断 token 是否命中外部词表 | 地名表、机构表、药品名表 |
| 词性特征 | 描述 token 在句法中的粗粒度功能 | 名词更可能是实体，动词通常不是实体 |
| 标签转移特征 | 描述相邻标签是否合法或常见 | `B-ORG` 后面可接 `I-ORG`，不应乱接 `I-PER` |

---

## 问题定义与边界

NER 的目标是对输入序列中的每个 token 打标签。token 是文本被切分后的基本单位，可以是英文单词、中文词、中文字符或子词。标签通常采用 BIO 或 BIOES 方案：`B-ORG` 表示组织名开头，`I-ORG` 表示组织名内部，`O` 表示非实体。

以句子 `Apple 收购 Beats` 为例，NER 关注的是：

| token | 可能标签 | 含义 |
|---|---|---|
| Apple | `B-ORG` | 一个组织名实体的开始 |
| 收购 | `O` | 非实体 |
| Beats | `B-ORG` | 另一个组织名实体的开始 |

这里的任务不是判断“这句话讲了收购事件”，也不是把 `Apple` 链接到某个知识库页面，而是识别实体边界和实体类型。

| 任务类型 | 输入 | 输出 | 与 NER 的边界 |
|---|---|---|---|
| 文本分类 | 整篇文本或句子 | 一个或多个类别 | 判断整体类别，不标注每个 token |
| 实体识别 | token 序列 | 实体边界与类别 | NER 本身 |
| 实体链接 | 已识别实体或文本片段 | 知识库 ID | 解决 `Apple` 指公司还是水果的问题 |
| 关系抽取 | 实体对与上下文 | 实体之间的关系 | 判断 `Apple` 与 `Beats` 是收购关系 |
| 事件抽取 | 文本和实体 | 事件类型、触发词、论元 | 关注事件结构，不只是实体 |

本文讨论的特征工程，主要指传统序列标注模型中的人工特征设计，包括 CRF、最大熵马尔可夫模型、感知机序列标注器，以及 BiLSTM-CRF 中外接词典特征的做法。它不等同于端到端神经网络的表示学习。表示学习是模型从数据中自动学向量特征；特征工程是人把领域知识显式编码成输入。两者不是互斥关系。

---

## 核心机制与推导

CRF，全称 Conditional Random Field，中文常译为条件随机场。线性链 CRF 是最常见的 NER 模型之一，它不独立判断每个 token 的标签，而是为整个标签序列打分，然后选择全局分数最高的一条标签路径。

线性链 CRF 常写成：

$$
P(y|x)=\frac{1}{Z(x)}\exp\Big(\sum_{i=1}^{n}\sum_k \lambda_k f_k(y_{i-1},y_i,x,i)\Big)
$$

这条公式可以分三层理解。

第一层，`x` 是输入句子，例如 `Apple 收购 Beats`；`y` 是标签序列，例如 `[B-ORG, O, B-ORG]`。模型要求的是在给定输入 $x$ 时，某个标签序列 $y$ 的概率。

第二层，$f_k(y_{i-1},y_i,x,i)$ 是特征函数。特征函数是一种判断器：如果某个条件成立就返回 1，否则返回 0，或者返回一个数值。例如“当前位置词首字母大写且当前标签是 `B-ORG`”就是一个特征函数。

第三层，$\lambda_k$ 是特征权重，表示模型学到的某个特征有多重要。$Z(x)$ 是归一化项，用来让所有可能标签序列的概率加起来等于 1。

玩具例子：看 `Apple 收购 Beats` 中的 `Apple`。假设模型对 `ORG` 的激活特征权重之和是：

$$
0.6_{\text{首字母大写}} + 0.2_{\text{后缀特征}} + 1.0_{\text{组织词典命中}} + 0.4_{\text{句首到ORG转移}} = 2.2
$$

而对 `PER` 的权重之和是：

$$
0.1_{\text{首字母大写}} + 0.0_{\text{其他特征}} = 0.1
$$

在未归一化分数上，`ORG` 明显高于 `PER`，模型就更倾向把 `Apple` 标成组织名。注意，如果只看“首字母大写”，`Apple` 可能是公司，也可能是普通专有名词；加入上下文和词典后，判断才更稳。

| 特征类型 | 含义 | 例子 | 对标签的影响 |
|---|---|---|---|
| 当前词特征 | 当前 token 的原始形式 | `word=Apple` | 记住高频实体词 |
| 词形特征 | token 的形态模式 | `is_capitalized=True` | 提高英文专名概率 |
| 前后缀特征 | token 的开头或结尾片段 | `suffix=Inc`、`prefix=Mc` | 辅助判断机构名、人名 |
| 上下文特征 | 前后窗口内的词 | `next_word=收购` | `收购` 前后常见组织名 |
| 词典特征 | 是否命中外部 gazetteer | `gazetteer_org=True` | 强化领域实体先验 |
| 词性特征 | token 的语法类别 | `POS=NNP` | 专有名词更可能是实体 |
| 转移特征 | 前一标签到当前标签的关系 | `B-ORG -> I-ORG` | 避免非法 BIO 序列 |

gazetteer 是实体词典或名称表，通常包含地名、机构名、药品名、公司名等。特征模板的工程价值在于组合局部信号，例如：

```text
prev_word + current_word_shape + current_POS + gazetteer_hit
```

这表示模型不是单看一个属性，而是学习“前文是什么、当前词长什么样、词性是什么、是否命中词典”共同出现时，对标签有什么影响。

---

## 代码实现

实现 NER 特征工程通常分三步：先给每个 token 抽取特征，再把这些特征组织成模板，最后交给 CRF 或 BiLSTM-CRF 训练和解码。

伪代码如下：

```text
sentences = load_labeled_data()
for sentence in sentences:
    for token_index in sentence:
        features[token_index] = extract_features(sentence, token_index)
model = CRF()
model.fit(features, labels)
predicted_labels = model.predict(new_sentence_features)
```

新手版理解：给每个词做一张“属性卡”。例如 `Apple` 的属性卡可以包含 `is_capitalized=True`、`suffix=ple`、`gazetteer_org=True`、`prev_word=<BOS>`、`next_word=收购`。模型不是凭空判断标签，而是读这些属性卡。

下面是一个最小可运行的 Python 特征提取函数：

```python
def word_shape(token: str) -> str:
    if token.isdigit():
        return "NUMBER"
    if token.isupper():
        return "ALL_UPPER"
    if token[:1].isupper():
        return "INIT_UPPER"
    if "-" in token:
        return "HAS_HYPHEN"
    return "LOWER_OR_OTHER"


def extract_token_features(tokens, i, org_gazetteer=None, loc_gazetteer=None):
    org_gazetteer = org_gazetteer or set()
    loc_gazetteer = loc_gazetteer or set()

    token = tokens[i]
    prev_token = tokens[i - 1] if i > 0 else "<BOS>"
    next_token = tokens[i + 1] if i + 1 < len(tokens) else "<EOS>"

    return {
        "word.lower": token.lower(),
        "shape": word_shape(token),
        "prefix2": token[:2],
        "suffix3": token[-3:],
        "is_capitalized": token[:1].isupper(),
        "prev_word.lower": prev_token.lower(),
        "next_word.lower": next_token.lower(),
        "gazetteer_org": token in org_gazetteer,
        "gazetteer_loc": token in loc_gazetteer,
        "bias": 1.0,
    }


tokens = ["Apple", "acquired", "Beats"]
features = extract_token_features(
    tokens,
    0,
    org_gazetteer={"Apple", "Beats"},
    loc_gazetteer={"Paris"},
)

assert features["word.lower"] == "apple"
assert features["shape"] == "INIT_UPPER"
assert features["suffix3"] == "ple"
assert features["prev_word.lower"] == "<bos>"
assert features["next_word.lower"] == "acquired"
assert features["gazetteer_org"] is True
assert features["gazetteer_loc"] is False
```

| 原子特征 | 组合模板 | 含义 |
|---|---|---|
| `word.lower` | `current_word + current_label` | 当前词对当前标签的影响 |
| `shape` | `shape + current_label` | 词形对实体类别的影响 |
| `prev_word.lower` | `prev_word + current_word + label` | 上下文搭配对标签的影响 |
| `suffix3` | `suffix3 + label` | 后缀对类别的提示 |
| `gazetteer_org` | `gazetteer_org + label` | 命中组织词典时是否倾向 `ORG` |
| `prev_label` | `prev_label + current_label` | 标签转移是否合法或常见 |

在 Stanford NER 这类传统工具中，配置项本质上也是特征模板开关。`gazette` 用于加入词典命中特征，`wordShape` 用于加入词形模式，`disjunctive` 用于加入较宽上下文窗口特征，额外列输入可以把 POS、chunk、词典类别等信息喂给模型。工程实现会比上面的 Python 函数复杂，但核心思路相同：把 token 周围的可解释信号编码成模型可学习的特征。

真实工程例子：医疗 NER 中需要识别“阿司匹林”“CT”“内分泌科”“2 型糖尿病”。如果只有几百条标注样本，模型很难从数据中完整学到所有药品、检查项目和科室名。这时可以维护药品词典、检查项目词典、科室词典，把 `gazetteer_drug=True`、`gazetteer_department=True` 作为特征输入 CRF 或 BiLSTM-CRF。模型仍然根据上下文决策，但词典提供了稳定先验。

---

## 工程权衡与常见坑

特征不是越多越好。每增加一个模板，模型参数空间都会扩大。如果训练数据少，很多组合特征只出现一两次，模型容易记住偶然噪声，而不是学到可泛化规律。这种现象叫稀疏，意思是大部分特征在训练集中很少出现，统计证据不足。

词典特征也不是越大越好。低质量词典会把普通词误收进实体表，导致模型学歪。例如地名词典里混入大量普通名词，模型可能把正常名词误判为 `LOC`。更稳的做法是使用高精度词表、给词典命中设置软特征，而不是直接把命中结果当最终标签。

| 常见坑 | 典型表现 | 原因 | 规避方法 |
|---|---|---|---|
| 词典噪声大 | 普通词被标成地名或机构名 | 词表召回高但精度低 | 清洗词表，区分强词典与弱词典，加入上下文判断 |
| 特征过多导致过拟合 | 训练集效果高，测试集下降 | 模板组合太细，数据支撑不足 | 做消融实验，控制窗口大小，正则化 |
| 只看词形会误判 | 首字母大写词都被当实体 | 词形信号不等于语义类别 | 结合上下文、POS、词典和转移特征 |
| BIO 标注不一致 | 出现 `O -> I-ORG` 等非法序列 | 数据清洗和转换不统一 | 训练前校验标签，解码时加入转移约束 |
| 神经网络排斥词典特征 | 专业实体漏识别严重 | 误以为端到端不需要先验 | 将词典命中作为额外 embedding 或 CRF 特征 |
| 上下文窗口过宽 | 模型变慢且噪声变多 | 远距离词未必相关 | 从 `[-2,+2]` 开始实验，按验证集调整 |
| 中英文分词不稳定 | 实体边界被切碎或合并 | tokenization 与标注粒度不一致 | 统一分词方案，必要时使用字符级标注 |

一个常见错误是把“词典命中”当成规则结果。例如 `Amazon` 命中组织名词典，但在 “Amazon rainforest” 中可能是地理区域的一部分。更合理的方式是让词典命中成为特征，而不是最终答案。CRF 会同时考虑 `rainforest`、POS、前后标签等信号。

另一个错误是忽视标签转移。BIO 标注中，`I-ORG` 前面通常应该是 `B-ORG` 或 `I-ORG`，不应直接跟在 `B-PER` 后面。CRF 的优势之一就是把这种相邻标签关系纳入全局解码，而不是每个 token 独立分类。

---

## 替代方案与适用边界

在标注数据较多、上下文复杂、实体表达变化丰富的场景中，BiLSTM-CRF、CNN-CRF、Transformer 可以替代以手工特征为主的方案。BiLSTM 是双向长短期记忆网络，可以同时看左侧和右侧上下文；Transformer 是基于注意力机制的模型，可以建模更灵活的上下文关系。

但这不意味着特征工程过时。在低资源、专业领域、长尾实体密集的场景中，词典、规则、软 gazetteer 仍然有稳定价值。soft gazetteer 指把词典命中变成可学习的软信号，而不是硬规则。它可以告诉模型“这个词像药品名”，但最终仍由模型结合上下文判断。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 传统 CRF + 特征工程 | 可解释、数据需求较低、可控性强 | 依赖人工模板，迁移成本高 | 小数据、规则清晰、上线要求稳 |
| BiLSTM-CRF | 自动学习上下文表示，CRF 保留转移约束 | 需要更多数据和训练成本 | 中等数据量的通用 NER |
| BiLSTM-CRF + 词典特征 | 兼顾表示学习与领域先验 | 词典维护有成本 | 医疗、法律、政务等专业领域 |
| 纯 Transformer | 上下文建模能力强，迁移效果好 | 标注少时可能不稳，成本较高 | 数据较多或可用预训练模型的场景 |
| soft gazetteer | 低资源下补充长尾实体知识 | 实现复杂度高于硬匹配 | 多语言、低资源、领域实体密集场景 |

医疗文本是典型例子。“阿司匹林、CT、内分泌科、肝功能、二甲双胍”这些词在通用语料中不一定高频。如果只有少量标注样本，纯神经网络可能漏掉很多专业实体。加入药品词典、检查项目词典、科室词典后，模型可以更快学到领域边界。

最终选择可以按一个简单原则判断：如果数据少、实体表稳定、错误成本高，优先保留特征工程和词典；如果数据多、语言变化复杂、可用预训练模型强，可以减少手工模板；如果是低资源专业场景，通常选择神经网络加词典特征，而不是二选一。

---

## 参考资料

读文献不要只看标题，要知道它回答的是哪一个问题：CRF 为什么适合序列标注，NER 里哪些特征最有效，工程实现如何配置，低资源时词典怎么用。

| 文献 | 核心贡献 | 对应章节 | 适合读者 |
|---|---|---|---|
| Lafferty et al. 2001 | 提出 CRF 序列建模框架 | 核心机制与推导 | 想理解公式的读者 |
| Ratinov & Roth 2009 | 系统讨论 NER 特征设计与误区 | 工程权衡与常见坑 | 想做传统 NER 的读者 |
| Stanford NERFeatureFactory | 展示工程中特征模板如何配置 | 代码实现 | 想落地工具的读者 |
| Stanford CoreNLP NER | 说明 Stanford NER 的使用方式 | 代码实现 | 想快速实验的读者 |
| Chiu & Nichols 2016 | 将 BiLSTM-CNN 用于 NER | 替代方案与适用边界 | 想理解神经 NER 的读者 |
| Rijhwani et al. 2020 | 讨论低资源 NER 中 soft gazetteer 的价值 | 替代方案与适用边界 | 做低资源或多语言 NER 的读者 |

1. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/entities/publication/c9aea099-b5c8-4fdd-901c-15b6f889e4a7)
2. [Design Challenges and Misconceptions in Named Entity Recognition](https://aclanthology.org/W09-1119/)
3. [Stanford CoreNLP NERFeatureFactory Documentation](https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ie/NERFeatureFactory.html)
4. [Stanford CoreNLP Named Entity Recognition](https://stanfordnlp.github.io/CoreNLP/ner.html)
5. [Named Entity Recognition with Bidirectional LSTM-CNNs](https://aclanthology.org/Q16-1026/)
6. [Soft Gazetteers for Low-Resource Named Entity Recognition](https://aclanthology.org/2020.acl-main.722/)

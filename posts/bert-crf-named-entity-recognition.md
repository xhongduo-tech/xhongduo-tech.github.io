## 核心结论

BERT-CRF 是一种序列标注式命名实体识别模型：用 BERT 生成每个词元的上下文表示，再用 CRF 在整句范围内解码出全局最优的实体标签序列。

命名实体识别（Named Entity Recognition, NER）是从文本中找出有明确边界和类别的片段，例如人名、机构名、药品名、剂量、地点。BERT-CRF 的核心分工很清楚：

```text
输入文本
  |
分词器 Tokenizer
  |
BERT 编码器：为每个 token 生成上下文向量
  |
线性层：把向量转换成每个标签的发射分数
  |
CRF 解码层：结合标签转移约束，选择整句最优标签串
  |
BIO 标签序列
  |
实体片段
```

BERT 负责表示，CRF 负责解码。它的优势不只是“每个 token 分类更准”，而是“整句标签串更合法、更稳定”。

玩具例子：句子“张三 来了”不是把“张三”和“来了”完全分开判断。模型先看整句语境，再预测最合理的标签序列。对于按字切分的输入，“张 / 三 / 来 / 了”可能对应：

| token | 标签 | 含义 |
|---|---:|---|
| 张 | `B-PER` | 人名开始 |
| 三 | `I-PER` | 人名内部 |
| 来 | `O` | 非实体 |
| 了 | `O` | 非实体 |

`BIO` 是一种实体边界标注方式：`B-X` 表示某类实体的开始，`I-X` 表示某类实体的内部，`O` 表示不属于任何实体。CRF 的价值是减少 `O, I-PER` 这类边界不完整的输出。

真实工程例子：医疗病历抽取中，需要从“患者昨日服用阿莫西林 500mg 后皮疹缓解”里识别 `药品=阿莫西林`、`剂量=500mg`、`症状=皮疹`。这类任务对边界很敏感，错一个字就可能改变实体含义。BERT 提供语义理解能力，CRF 提供标签序列约束，所以 BERT-CRF 在工业 NER 中很常见。

---

## 问题定义与边界

NER 的输入是一段 token 序列，输出是同长度的标签序列，再由标签序列还原实体片段。它不是普通文本分类。文本分类只需要给整句话打一个标签，例如“这句话是否包含药品”；NER 要回答“药品从哪里开始，到哪里结束，类别是什么”。

| 项目 | 内容 |
|---|---|
| 输入 | token 序列，例如 `我 / 吃 / 了 / 阿 / 莫 / 西 / 林 / 500mg` |
| 输出 | 标签序列，例如 `O / O / O / B-DRUG / I-DRUG / I-DRUG / I-DRUG / B-DOSE` |
| 目标 | 找到实体边界和实体类别 |
| 常用标签体系 | `BIO`、`BIOES` |
| 典型实体 | 人名、机构、地点、药品、剂量、疾病、检查项 |

`BIOES` 是 `BIO` 的扩展：`S-X` 表示单 token 实体，`E-X` 表示实体结束。它比 `BIO` 更精细，但标签数量更多。

| 标签体系 | 标签含义 | 示例 |
|---|---|---|
| `BIO` | Begin、Inside、Outside | `B-PER I-PER O` |
| `BIOES` | Begin、Inside、Outside、End、Single | `B-PER E-PER O` 或 `S-PER O` |

新手例子：在“我昨天吃了阿莫西林 500mg”中，模型不仅要判断“阿莫西林”是药品，还要判断它的边界是“阿/莫/西/林”，不能只抽出“莫西林”；同时“500mg”是剂量，也不能和药品合并成一个实体。

边界问题还会出现在子词切分中。BERT 常用 WordPiece 或类似分词方法。WordPiece 是一种把稀有词拆成更小子词的分词方法，例如英文中 `unaffordable` 可能被拆成 `un`, `##afford`, `##able`。中文场景也可能因为词表或领域词导致一个实体被拆成多个 token。此时模型不能把每个子词都当成独立实体重新学习，否则训练信号会被重复放大。

需要区分三个概念：

| 任务 | 输出 | 是否关心边界 | 示例 |
|---|---|---:|---|
| 单标签分类 | 一个类别 | 否 | 判断句子是否包含药品 |
| 序列标注 | 每个 token 一个标签 | 是 | 标出“阿莫西林”和“500mg” |
| 结构化实体抽取 | 实体、属性、关系 | 是 | 药品-剂量-频次关联抽取 |

BERT-CRF 主要解决标准平铺实体的序列标注问题。平铺实体是指实体之间不互相嵌套，例如“北京大学”和“医学部”分别出现。它不直接解决所有 NER 难题：数据稀缺、超长文本截断、跨句实体、嵌套实体、多实体关系归因，都需要额外设计。

---

## 核心机制与推导

设输入序列为 $x=(x_1,\dots,x_n)$，标签序列为 $y=(y_1,\dots,y_n)$。BERT 对每个 token 输出上下文向量 $h_i \in R^d$。上下文向量是“带有整句语境信息的数字表示”，不是只看当前 token 本身。

线性层把 $h_i$ 映射为每个标签的发射分数：

$$
s_i(k)=W_k h_i+b_k
$$

发射分数表示“第 $i$ 个 token 被标成标签 $k$ 的局部证据强度”。如果 `张` 的 `B-PER` 发射分数高，说明 BERT 认为它像人名开头。

CRF（Conditional Random Field，条件随机场）是一种对整个标签序列建模的概率模型。它不只看每个位置的局部分数，还学习相邻标签之间的转移分数。例如 `B-PER -> I-PER` 通常合理，`O -> I-PER` 通常不合理。

序列总分为：

$$
S(x,y)=\sum_i s_i(y_i)+\sum_i T_{y_{i-1},y_i}
$$

其中 $T_{a,b}$ 是从标签 $a$ 转移到标签 $b$ 的分数，$y_0$ 可以看作 `START` 标签。

CRF 把所有可能标签序列都纳入归一化：

$$
p(y|x)=\frac{\exp(S(x,y))}{\sum_{y'}\exp(S(x,y'))}
$$

训练时最大化真实标签序列 $y^*$ 的概率，等价于最小化负对数似然：

$$
L=-\log p(y^*|x)
$$

推理时选择分数最高的标签序列：

$$
\hat{y}=\arg\max_y S(x,y)
$$

这个解码通常用 Viterbi 算法。Viterbi 是一种动态规划算法，用来在大量可能路径中高效找到最高分路径。

最小数值例子：句子“张三 来了”，为简单起见只看两个 token：`张三` 和 `来了`，标签集为 `{B-PER, I-PER, O}`。

| 序列 | 发射分数 | 转移分数 | 总分 |
|---|---:|---:|---:|
| `B-PER, I-PER` | `2.0 + 1.7` | `1.0` | `4.7` |
| `B-PER, O` | `2.0 + 0.2` | `0.0` | `2.2` |
| `O, O` | `0.5 + 0.2` | `0.0` | `0.7` |

如果只看局部分数，第一个 token 会选 `B-PER`，第二个 token 可能在 `I-PER` 和 `O` 间摇摆。CRF 会把“前一个标签是 `B-PER`”这个信息纳入判断，使 `B-PER -> I-PER` 这种连续人名结构更容易被选中。

这就是 BERT-CRF 的核心：BERT 提供每个位置的强语义证据，CRF 用标签转移关系修正局部贪心错误，得到更稳定的全局结果。

---

## 代码实现

实现 BERT-CRF 通常分为五步：

```text
tokens
  -> tokenizer 分词并对齐标签
  -> BERT(tokens) 得到 hidden_states
  -> linear(hidden_states) 得到 emissions
  -> CRF(emissions, tags, mask) 计算 loss
  -> CRF.decode(emissions, mask) 得到 pred_tags
```

工程难点主要不在模型层，而在标签对齐、padding mask 和解码结果还原。`mask` 是一个布尔数组，用来告诉模型哪些位置是真实 token，哪些位置是 padding 或不参与训练的位置。

标签对齐示例：

| 原词 | 原标签 | 子词 | 训练标签 |
|---|---|---|---|
| 我 | `O` | 我 | `O` |
| 吃了 | `O` | 吃 | `O` |
| 吃了 | `O` | ##了 | `-100` |
| 阿莫西林 | `B-DRUG` | 阿 | `B-DRUG` |
| 阿莫西林 | `B-DRUG` | ##莫 | `-100` |
| 阿莫西林 | `B-DRUG` | ##西 | `-100` |
| 阿莫西林 | `B-DRUG` | ##林 | `-100` |

`-100` 是 PyTorch 里常见的忽略标签值，表示这个位置不参与 loss。CRF 实现不一定直接支持 `-100`，更通用的做法是用 `mask` 把这些位置排除。

下面是一个可运行的 Python 玩具实现，不依赖 BERT，只演示 CRF/Viterbi 解码如何利用发射分数和转移分数选择整句最高分标签序列：

```python
from math import inf

labels = ["B-PER", "I-PER", "O"]
idx = {name: i for i, name in enumerate(labels)}

# 两个 token: 张三 / 来了
emissions = [
    [2.0, 0.1, 0.5],  # 张三: B-PER, I-PER, O
    [0.1, 1.7, 0.2],  # 来了: B-PER, I-PER, O
]

# transition[from][to]
transition = [[0.0 for _ in labels] for _ in labels]
transition[idx["B-PER"]][idx["I-PER"]] = 1.0
transition[idx["O"]][idx["I-PER"]] = -2.0  # 不鼓励 O -> I-PER

def viterbi_decode(emissions, transition):
    n = len(emissions)
    m = len(emissions[0])

    dp = [[-inf] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]

    for tag in range(m):
        dp[0][tag] = emissions[0][tag]

    for i in range(1, n):
        for cur in range(m):
            best_score = -inf
            best_prev = -1
            for prev in range(m):
                score = dp[i - 1][prev] + transition[prev][cur] + emissions[i][cur]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i][cur] = best_score
            back[i][cur] = best_prev

    last = max(range(m), key=lambda tag: dp[-1][tag])
    path = [last]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return [labels[i] for i in path], dp[-1][last]

path, score = viterbi_decode(emissions, transition)
assert path == ["B-PER", "I-PER"]
assert abs(score - 4.7) < 1e-9
print(path, score)
```

简化 PyTorch 结构通常如下：

```python
class BertCrfForNer:
    def __init__(self, bert, hidden_size, num_labels, crf):
        self.bert = bert
        self.classifier = Linear(hidden_size, num_labels)
        self.crf = crf

    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(hidden_states)

        mask = attention_mask.bool() & label_mask.bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss

        pred_tags = self.crf.decode(emissions, mask=mask)
        return pred_tags
```

真实工程中，`attention_mask` 通常排除 padding，`label_mask` 排除不参与训练的子词位置。训练和推理必须使用一致的 mask，否则 loss 看到的位置和 decode 输出的位置不一致，会导致评估结果失真。

---

## 工程权衡与常见坑

BERT-CRF 的效果很大程度取决于数据处理，而不是只取决于模型名字。尤其是标签对齐、padding mask、学习率、实体边界规范，会直接影响最终 F1。

常见坑如下：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 子词全部打同一标签 | 长词训练权重被重复放大 | 只标首个子词，其余子词 mask 或设为 `-100` |
| padding 位置参与 CRF | 模型把空白位置当有效标签学习 | loss 和 decode 都传入 mask |
| `O -> I-X` 不受约束 | 输出非法实体片段 | 在 CRF 转移中学习或限制非法转移 |
| 标签体系不统一 | 训练和评估口径不一致 | 统一 `BIO` 或 `BIOES`，并写清转换规则 |
| 长文本直接截断 | 实体被切断，召回下降 | 使用滑窗、分段、重叠窗口 |
| 小数据全量微调过猛 | 过拟合，验证集波动大 | 小学习率、早停、冻结部分层 |

新手例子：`WordPiece` 把“阿莫西林”切成多个子词时，如果每个子词都标成 `B-DRUG`，模型会学到一个错误模式：同一个药品内部每个位置都像实体开头。更合理的做法是第一个子词标 `B-DRUG`，后续子词要么标 `I-DRUG` 并参与训练，要么全部 mask。工业中更常用“只训练首个子词”，因为它和原词级标注更一致。

真实工程例子：医疗病历里常有大量 padding，因为 batch 内句子长度不同。如果 CRF 没有 mask padding 位置，它会把 padding 后面的无意义位置也纳入序列分数。这样训练 loss 看似下降，但模型实际学到的是“句尾后面还会出现某些标签”，推理时边界会变差。

超参数建议：

| 参数 | 常见范围 | 说明 |
|---|---:|---|
| BERT 学习率 | `1e-5` 到 `5e-5` | 过大容易破坏预训练表示 |
| CRF/分类头学习率 | `1e-4` 到 `1e-3` | 可高于 BERT |
| batch size | `8` 到 `32` | 显存不足可用梯度累积 |
| max length | `128` 到 `512` | 长文本需滑窗 |
| 是否冻结 BERT | 小数据可冻结底层 | 降低过拟合和显存压力 |
| 早停 | 观察验证集 F1 | 不只看训练 loss |

训练与推理的 mask 逻辑应保持一致：

| 位置类型 | attention mask | label mask | 是否算 loss | 是否参与 decode |
|---|---:|---:|---:|---:|
| 真实首子词 | 1 | 1 | 是 | 是 |
| 后续子词 | 1 | 0 | 否 | 通常否 |
| `[CLS]` / `[SEP]` | 1 | 0 | 否 | 否 |
| padding | 0 | 0 | 否 | 否 |

这里有一个工程权衡：CRF 能让标签序列更稳定，但会增加解码复杂度和实现细节。如果任务数据充足、实体边界简单，BERT-Softmax 可能已经够用。若实体类别多、边界敏感、小样本明显，CRF 通常更值得加入。

---

## 替代方案与适用边界

BERT-CRF 不是唯一方案。选型应看任务是否需要精确边界、是否有嵌套实体、是否要求复杂结构化输出。

| 方法 | 实现复杂度 | 解码约束 | 适合数据规模 | 是否支持嵌套实体 | 适用场景 |
|---|---:|---:|---:|---:|---|
| BERT-Softmax | 低 | 弱 | 中到大 | 否 | 边界简单、快速基线 |
| BERT-CRF | 中 | 强 | 小到中 | 否 | 标准 BIO NER、边界敏感 |
| Span-based | 中高 | 中 | 中到大 | 可支持 | 枚举片段并分类 |
| Pointer | 高 | 中高 | 中到大 | 可设计支持 | 抽取起止位置 |
| 生成式抽取 | 高 | 依赖提示和约束 | 中到大 | 可表达 | 开放 schema、复杂抽取 |

Span-based 方法是“先枚举候选文本片段，再判断片段类别”的方法。它不强制每个 token 只有一个标签，因此更适合嵌套实体。例如“北京大学医学部”中，“北京大学”可以是机构，“北京大学医学部”也可以是机构下属单位。标准 BIO 序列一个 token 只能有一个标签，不能自然表达这种嵌套关系。

Pointer 方法是预测实体开始位置和结束位置的方法，常见于阅读理解式抽取。生成式抽取则让模型直接生成结构化结果，例如 JSON 或三元组，但需要更强的约束和后处理。

如何选型：

| 任务条件 | 建议方案 |
|---|---|
| 只判断句子里有没有某类实体 | 文本分类器 |
| 需要抽出实体边界，实体不嵌套 | BERT-Softmax 或 BERT-CRF |
| 数据较少，边界错误代价高 | BERT-CRF |
| 有大量嵌套实体 | Span-based 或 Pointer |
| 需要抽取实体关系、属性归属 | 级联抽取或结构化生成 |
| 超长文档跨段抽取 | 滑窗 NER + 合并策略，或文档级模型 |

新手对比可以这样理解：如果任务只是“这句话里有没有药品”，分类器就够了；如果任务是“把药品名准确切出来”，BERT-CRF 更合适；如果任务是“抽出药品、剂量、频次，并判断它们属于同一次用药”，只靠 BERT-CRF 不够，还需要关系抽取或结构化抽取。

BERT-CRF 的边界是清楚的：它非常适合标准 BIO 序列标注，但不适合直接表达复杂层级结构。把所有复杂抽取都塞进 BIO 标签，会导致标签爆炸、训练样本稀疏、错误难以解释。工程上应先判断任务结构，再决定模型形式。

---

## 参考资料

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)  
   BERT 基础论文，适合理解预训练语言模型如何产生上下文表示。

2. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/handle/20.500.14332/6188)  
   CRF 基础论文，适合理解序列标注中的全局归一化和标签转移建模。

3. [Hugging Face Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)  
   官方实践文档，适合理解 NER 任务的数据格式、分词对齐和训练流程。

4. [Portuguese Named Entity Recognition using BERT-CRF](https://arxiv.org/abs/1909.10649)  
   BERT-CRF 应用论文，适合观察 BERT 编码器与 CRF 解码层在具体 NER 任务中的组合方式。

5. [bond005/bert_ner](https://github.com/bond005/bert_ner)  
   BERT + CRF 的开源实现参考，适合对照工程代码理解训练、解码和评估流程。

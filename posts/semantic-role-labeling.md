## 核心结论

语义角色标注（Semantic Role Labeling，SRL）解决的是“谓词周围的参与者分别扮演什么语义角色”这个问题。谓词是句子里表示动作、状态或事件核心的词，通常是动词；论元是围绕谓词出现的参与者、对象、时间、地点等成分。

SRL 的核心不是找主语、宾语这类句法关系，而是给动作参与者贴语义标签。一个最小定义可以写成：

$$
\text{SRL}(x,p)=\text{谓词}+\text{论元}+\text{角色标签}
$$

其中 $x$ 是句子，$p$ 是谓词位置，输出是一组带角色标签的文本片段。

玩具例子：

句子：`小王在北京昨天买了苹果`  
谓词：`买了`  
SRL 要回答的是：谁买了什么、在哪里、什么时候。

| 词语 | 角色 | 说明 |
|---|---|---|
| 小王 | ARG0 | 做出“买”这个动作的人 |
| 北京 | LOC | 动作发生的地点 |
| 昨天 | TMP | 动作发生的时间 |
| 苹果 | ARG1 | 被购买的对象 |

可以写成：

```text
小王/ARG0 北京/LOC 昨天/TMP 苹果/ARG1
```

新手可以先把 SRL 理解为：把一句话拆成“动作和参与者”的表格。这个结构化结果可以直接用于事件抽取、信息检索、问答系统和业务规则处理。例如在工单系统里，把“用户昨天在 App 里取消了订单”转成“用户 / 取消 / 订单 / 昨天 / App”，后续就能做统计、检索和自动流转。

---

## 问题定义与边界

语义角色标注的标准输入通常是一个句子和一个谓词位置，输出是该谓词对应的论元及其角色标签。设句子为：

$$
x=(w_1,\dots,w_n)
$$

其中 $w_i$ 表示第 $i$ 个词或 token，谓词位置为 $p$，每个位置的标签为：

$$
y_i \in \mathcal{L}\cup\{O\}
$$

$\mathcal{L}$ 是角色标签集合，例如 `ARG0`、`ARG1`、`LOC`、`TMP`；`O` 表示该位置不属于当前谓词的论元。

SRL 依赖上下文，不是只看词面。同一个动词在不同句子里，参与者可能不同。

例子：

```text
他把门打开了。
门打开了。
```

第一句里，`打开` 有显式施事和受事：`他` 是做动作的人，`门` 是被打开的对象。第二句里，`门` 更像状态变化的主体，句子没有显式说明是谁打开了门。新手可以理解为：同一个动词不保证有同一套参与者，SRL 要根据当前句子判断角色。

SRL 也不等于词性标注、依存句法分析或命名实体识别。

| 任务 | 关注目标 | 输出例子 | 与 SRL 的差异 |
|---|---|---|---|
| SRL | 谁对谁做了什么，何时何地发生 | `小王/ARG0 苹果/ARG1` | 关注谓词和参与者的语义关系 |
| 依存句法 | 词和词之间的句法依赖 | 主谓、动宾、定中 | 关注语法结构，不直接给语义角色 |
| NER | 人名、地名、机构名等实体 | `北京/地点` | 只识别实体类型，不说明它在事件中的作用 |
| 事件抽取 | 事件类型和事件槽位 | `购买事件: 买方、商品、时间` | 通常比 SRL 更贴近业务 schema |

真实工程例子：

在新闻系统里，句子“某公司昨日在上海收购了一家芯片企业”可以通过 SRL 得到“某公司是收购方、芯片企业是被收购对象、昨日是时间、上海是地点”。如果业务目标是构建企业并购数据库，这种结构比单纯识别公司名更有用。

---

## 核心机制与推导

传统 SRL 通常分两步：先识别谓词，再围绕谓词识别论元并分类。两阶段方法可以写成：

$$
p^*=\arg\max_p s_{\text{pred}}(p\mid x)
$$

$$
y^*=\arg\max_y \sum_i s_{\text{role}}(y_i\mid x,p^*)
$$

第一步从句子里找最可能的谓词位置 $p^*$，第二步在给定谓词的条件下，为每个 token 预测角色标签。

端到端神经网络会把这两个步骤合到一个模型里。端到端是指模型从原始输入直接预测最终输出，中间不需要人工拆成多个独立系统。常见做法是用 BIO 标签体系统一表示论元边界和角色类型。

BIO 是一种序列标注格式：`B-` 表示一个片段开始，`I-` 表示片段内部，`O` 表示不属于任何片段。

玩具例子：

```text
句子：小王 在 北京 昨天 买了 苹果
谓词：买了
BIO：小王/B-ARG0 在/O 北京/B-LOC 昨天/B-TMP 买了/V 苹果/B-ARG1
```

| 标签 | 含义 | 例子 |
|---|---|---|
| B-ARG0 | ARG0 片段开始 | `小王/B-ARG0` |
| I-ARG0 | ARG0 片段内部 | `北京大学/I-ARG0` 中的后续 token |
| B-ARG1 | ARG1 片段开始 | `苹果/B-ARG1` |
| B-LOC | 地点片段开始 | `北京/B-LOC` |
| B-TMP | 时间片段开始 | `昨天/B-TMP` |
| O | 不属于论元 | `在/O` |
| V | 当前谓词 | `买了/V` |

其中 `ARG0` 通常表示施事，也就是做事的人或实体；`ARG1` 通常表示受事，也就是动作影响的对象；`LOC` 是地点；`TMP` 是时间。不同语料和谓词框架对 `ARG0`、`ARG1` 的定义可能有细节差异，工程中必须以数据集标注规范为准。

如果使用 CRF（条件随机场）做解码，模型不仅看每个 token 的局部分数，还会看相邻标签之间是否合法。CRF 是一种序列模型，常用于让输出标签序列满足连续性约束。目标函数可以写成：

$$
y^*=\arg\max_y \left[\sum_i s_i(y_i\mid x,p)+\sum_{i=2}^{n}T(y_{i-1},y_i)\right]
$$

其中 $s_i$ 是第 $i$ 个位置取某个标签的分数，$T$ 是从前一个标签转移到当前标签的分数。这样可以降低非法序列的概率，例如 `I-ARG1` 不应该在没有 `B-ARG1` 的情况下突然出现。

---

## 代码实现

代码实现的重点不是训练一个完整大模型，而是把 SRL 的数据流拆清楚：输入句子、确定谓词、预测标签、解码 BIO 序列、合并连续片段。

一个极简流程是：

1. 读取句子并分词。
2. 指定或预测谓词位置。
3. 对每个 token 计算角色分数。
4. 用 CRF 或贪心方式解码得到 BIO 标签。
5. 合并连续的 `B-` / `I-` 片段，输出结构化论元。

解码目标可以简写为：

$$
score(y)=\sum_i s_i(y_i)+\sum_i T(y_{i-1},y_i)
$$

下面代码不是训练代码，而是一个可运行的最小示例：它模拟模型输出的 BIO 标签，并实现 BIO 合并与非法 `I-` 修正。

```python
def normalize_bio(tags):
    """把非法 I-X 开头或跨类型 I-X 修正成 B-X。"""
    normalized = []
    prev_type = None

    for tag in tags:
        if tag == "O" or tag == "V":
            normalized.append(tag)
            prev_type = None
            continue

        prefix, role = tag.split("-", 1)
        if prefix == "B":
            normalized.append(tag)
            prev_type = role
        elif prefix == "I":
            if prev_type == role:
                normalized.append(tag)
            else:
                normalized.append(f"B-{role}")
            prev_type = role
        else:
            raise ValueError(f"unknown tag: {tag}")

    return normalized


def bio_to_spans(tokens, tags):
    tags = normalize_bio(tags)
    spans = []
    current = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O" or tag == "V":
            if current:
                spans.append(current)
                current = None
            continue

        prefix, role = tag.split("-", 1)
        if prefix == "B":
            if current:
                spans.append(current)
            current = {"role": role, "start": i, "end": i + 1, "text": [token]}
        elif prefix == "I" and current and current["role"] == role:
            current["end"] = i + 1
            current["text"].append(token)

    if current:
        spans.append(current)

    for span in spans:
        span["text"] = "".join(span["text"])

    return spans


tokens = ["小王", "在", "北京", "昨天", "买了", "苹果"]
predicate_index = 4
decoded_tags = ["B-ARG0", "O", "B-LOC", "B-TMP", "V", "B-ARG1"]

spans = bio_to_spans(tokens, decoded_tags)

assert predicate_index == tokens.index("买了")
assert spans == [
    {"role": "ARG0", "start": 0, "end": 1, "text": "小王"},
    {"role": "LOC", "start": 2, "end": 3, "text": "北京"},
    {"role": "TMP", "start": 3, "end": 4, "text": "昨天"},
    {"role": "ARG1", "start": 5, "end": 6, "text": "苹果"},
]

illegal_tags = ["I-ARG0", "O", "I-LOC"]
assert normalize_bio(illegal_tags) == ["B-ARG0", "O", "B-LOC"]
```

关键变量可以这样理解：

| 变量 | 含义 | 形状或例子 |
|---|---|---|
| `tokens` | 分词后的输入 | `["小王", "在", "北京", ...]` |
| `predicate_index` | 谓词位置 | `4` |
| `tag_scores` | 每个 token 对每个标签的分数 | `[seq_len, num_tags]` |
| `decoded_tags` | 解码后的 BIO 标签 | `["B-ARG0", "O", ...]` |
| `spans` | 合并后的论元片段 | `{"role": "ARG1", "text": "苹果"}` |

真实工程里，`tag_scores` 通常来自 BERT、RoBERTa 或其他 Transformer 编码器。Transformer 是一种基于注意力机制的神经网络结构，适合建模长距离上下文。谓词位置可以通过额外特征编码，例如给谓词 token 加特殊标记，或输入一个 predicate indicator 向量。

---

## 工程权衡与常见坑

SRL 的第一个工程问题是评测口径。`gold predicate` 和 `predicted predicate` 是两套不同难度的设置。`gold predicate` 表示评测时已经给定正确谓词位置，模型只需要预测论元；`predicted predicate` 表示模型还要自己找谓词。后者更接近真实场景，通常更难，分数不能和前者直接横比。

常用指标是精确率、召回率和 F1：

$$
F1=\frac{2PR}{P+R}
$$

其中 $P$ 是精确率，$R$ 是召回率。SRL 的 F1 通常基于“论元边界和角色标签是否同时正确”计算，所以边界错了也会扣分。

常见坑如下：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 谓词歧义 | 同一个词在不同语境下角色不同 | 引入上下文编码，不只用词典 |
| 分词错误 | 论元边界错位 | 保持训练和推理 tokenization 一致 |
| BIO 断裂 | 输出非法片段，如 `I-ARG1` 开头 | 解码阶段增加合法性约束 |
| 评测口径不一致 | 分数看起来提升但不可比 | 明确 `gold predicate` 或 `predicted predicate` |
| 句法错误传播 | 依存句法错导致 SRL 错 | 对句法特征做消融实验 |
| 多词论元 | 漏掉修饰词或边界不完整 | 用 span-level 评测分析错误 |

真实工程例子：

句子：`患者昨晚在家服用阿莫西林后过敏`。  
目标结构可能是：

| 片段 | 角色 | 说明 |
|---|---|---|
| 患者 | ARG0 | 服用动作的执行者 |
| 昨晚 | TMP | 时间 |
| 在家 | LOC | 地点 |
| 阿莫西林 | ARG1 | 被服用的药物 |

如果分词或 BIO 边界把“在家”拆错，`LOC` 就可能丢失；如果把“昨晚在家”整体标成 `TMP`，地点信息会被污染。新手需要记住：SRL 不只是“认出词”，还要“把词块边界圈对”。

另一个坑是把“高 F1”直接写成“达到人类水平”。这类表述必须谨慎。模型在 CoNLL-2009 等基准上的分数很高，BERT-SRL 这类强基线也显著提升了结果，但数据集、标注规范、谓词是否给定、语言类型、评测脚本都会影响结论。工程报告里更稳妥的写法是：在某个公开数据集和某种评测设置下达到接近人工标注一致性的表现，而不是泛化为所有文本都接近人类理解。

---

## 替代方案与适用边界

SRL 适合需要把自然语言转成结构化事件的场景。如果任务只需要识别实体或关键词，直接用 NER、关键词抽取或规则可能更划算。

结构化输出可以概括为：

$$
\text{结构化输出}=\text{角色标签}+\text{片段边界}+\text{事件槽位}
$$

不同方案的适用边界如下：

| 方法 | 适合场景 | 不适合场景 |
|---|---|---|
| SRL | 需要抽取动作和参与者关系 | 只需要实体列表或关键词 |
| NER | 识别人名、地名、产品名、时间 | 需要知道实体在事件中的作用 |
| 依存句法 | 分析句法结构、主谓宾关系 | 直接输出业务事件槽位 |
| 事件抽取 | 已有明确事件类型和槽位 | 事件类型开放、schema 不稳定 |
| 规则抽取 | 格式固定、成本敏感、可解释性强 | 表达变化多、召回要求高 |

客服工单例子：

如果目标只是识别“产品名”和“时间”，NER 可能足够。例如从“用户昨天反馈 iPhone 充电异常”里抽出 `昨天/TIME` 和 `iPhone/PRODUCT`。但如果目标是抽取“谁在什么时候对哪个产品做了什么操作”，SRL 更合适：

| 槽位 | 结果 |
|---|---|
| 施事 | 用户 |
| 时间 | 昨天 |
| 动作 | 反馈 |
| 对象 | iPhone 充电异常 |

句法增强模型和无句法模型也有取舍。句法增强模型会显式使用依存树、句法路径等信息，优点是结构约束强，缺点是解析器错误会向下游传播。无句法模型主要依赖上下文编码器，优点是系统更简单、鲁棒性更好，缺点是对长距离结构关系的约束不一定稳定。

工程选择可以按问题复杂度来定：规则能解决且维护成本低，就不需要上 SRL；NER 能满足业务目标，就不用引入谓词-论元建模；只有当系统必须理解“动作-参与者”结构时，SRL 才是合适工具。

---

## 参考资料

| 论文 | 贡献 | 适合阅读目的 |
|---|---|---|
| Palmer, Titov, Wu 2013 | 梳理 SRL 任务、术语和研究背景 | 入门定义 |
| Hajič et al. 2009 | CoNLL-2009 Shared Task 数据与评测设置 | 理解基准 |
| Cai et al. 2018 | 端到端 SRL 与统一建模方法 | 理解模型结构 |
| Shi & Lin 2019 | BERT 在 SRL 上的强基线 | 理解预训练模型效果 |
| Jindal et al. 2023 | 分析 SRL 评测与误差传播 | 理解评测风险 |

1. [Semantic Role Labeling](https://aclanthology.org/N13-4004/)
2. [The CoNLL-2009 Shared Task: Syntactic and Semantic Dependencies in Multiple Languages](https://aclanthology.org/W09-1201/)
3. [A Full End-to-End Semantic Role Labeler, Syntactic-Agnostic Over Syntax-Aware?](https://aclanthology.org/C18-1233/)
4. [Simple BERT Models for Relation Extraction and Semantic Role Labeling](https://arxiv.org/abs/1904.05255)
5. [PriMeSRL-Eval: A Practical Evaluation Framework for Semantic Role Labeling Systems](https://aclanthology.org/2023.findings-eacl.134/)

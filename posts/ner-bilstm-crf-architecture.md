## 核心结论

命名实体识别（Named Entity Recognition，NER，白话说就是“从文本里找出有名字的关键片段并判断它是什么类型”）本质上是一个**带边界约束的结构化预测问题**：输入是词序列 $x_{1:n}$，输出不是一个单点标签，而是一组满足规则的实体片段及其类别。

传统做法通常把 NER 写成序列标注：给每个词贴上 `B/I/O` 这类位置标签，再配合线性链 CRF（Conditional Random Field，条件随机场，白话说就是“按整条标签路径一起打分，而不是每个位置各自打分”）做全局解码。这样做的优势是路径一致性强，能显式避免很多非法标签组合。核心形式是：

$$
P(y|x)=\frac{1}{Z(x)}\exp\left(\sum_j \lambda_j h_j(y,x)\right)
$$

其中 $Z(x)$ 是归一化项，作用是把所有可能标签路径的分数统一纳入比较，而不是只看某个位置的局部最大值。

近年的 span-based NER（区间式 NER，白话说就是“直接枚举文本中的候选片段，再判断这个片段是不是实体以及属于什么类”）则把问题从“逐词贴标签”改成“逐区间分类”。例如句子 `the wall street journal launched ...`，BIO 做法会标成：

| token | the | wall | street | journal | launched |
|---|---:|---:|---:|---:|---:|
| tag | B-ORG | I-ORG | I-ORG | I-ORG | O |

而 span-based 会直接判断区间 $[1,4]$ 是 `ORG`，其余区间为非实体。这个改写提升了对嵌套实体、长实体和边界信息的直接表达能力，但计算量通常更高。

如果实体**不嵌套、边界规则稳定、标注体系清晰**，BIO+CRF 往往更稳、更省数据；如果实体**嵌套频繁、重叠明显、上下文语义复杂**，span-based 通常更合适。

---

## 问题定义与边界

NER 的形式化定义可以写成：给定输入序列 $x_{1:n}$，在其中找出若干个片段 $(s,e,t)$，其中 $s,e$ 分别是起止位置，$t \in \mathcal{T}$ 是实体类型，例如 `PER`（人名）、`LOC`（地点）、`ORG`（组织）、`MISC`（其他）。目标是在预设标签集 $\mathcal{T}$ 上，最大化实体识别与分类的准确性。

对初学者，可以把它先理解成三步：

1. 分词：把句子切成模型处理的基本单元。
2. 编码：把每个词变成向量表示。
3. 标注：输出词级标签或区间级标签。

序列标注中最常见的是 BIO 体系。BIO 的含义如下：

| 标签 | 含义 | 白话解释 |
|---|---|---|
| `B-X` | Begin of entity type X | 一个实体的开头 |
| `I-X` | Inside of entity type X | 同一个实体的中间或后续 |
| `O` | Outside | 不属于任何实体 |

例如句子 `Apple opened a store in Shanghai`，可以标成：

| token | Apple | opened | a | store | in | Shanghai |
|---|---|---|---|---|---|---|
| BIO | B-ORG | O | O | O | O | B-LOC |

这一定义有几个重要边界：

| 维度 | 说明 | 工程影响 |
|---|---|---|
| 标签集合固定 | 训练前先定义实体类型 | 新类型通常不能零成本扩展 |
| 是否允许嵌套 | 一个实体内部是否还能包含实体 | 决定 BIO 是否足够 |
| 是否允许重叠 | 不同实体区间是否共享词 | 线性链模型通常难处理 |
| 上下文窗口 | 模型一次看到多长的上下文 | 长文本可能截断关键信息 |
| 域迁移能力 | 新闻、医疗、电商等领域是否共用规则 | 领域差异会显著影响效果 |

这里最容易混淆的一点是：**NER 不是单纯的分类问题，而是“边界检测 + 类型判定”的联合问题**。如果边界错了，即使类别判断正确，结果仍然是错的。

玩具例子可以看这句：

`the wall street journal`

如果标为：

- `the -> B-ORG`
- `wall -> I-ORG`
- `street -> I-ORG`
- `journal -> I-ORG`

那么模型认为整个连续片段是组织名。若把 `journal` 错标成 `O`，实体边界立刻断裂，结果从一个正确实体变成一个残缺片段。这说明 NER 的错误常常不是“分错类”，而是“边界被破坏”。

---

## 核心机制与推导

### 1. BIO + 线性链 CRF：为什么要按整条路径打分

线性链 CRF 的思想是：给整条标签路径 $y_{1:n}$ 打一个总分，再在所有合法路径里选分数最高的那条。常见打分形式可写为：

$$
\text{score}(x,y)=\sum_{i=1}^n \phi_i(y_i, x)+\sum_{i=1}^n \psi_i(y_{i-1},y_i,x)
$$

其中：

- $\phi_i$ 是发射分数，表示“第 $i$ 个词像不像标签 $y_i$”。
- $\psi_i$ 是转移分数，表示“标签 $y_{i-1}$ 接到 $y_i$ 是否合理”。

于是条件概率是：

$$
P(y|x)=\frac{\exp(\text{score}(x,y))}{Z(x)}
$$

归一化项：

$$
Z(x)=\sum_{y' \in \mathcal{Y}(x)} \exp(\text{score}(x,y'))
$$

这里 $\mathcal{Y}(x)$ 是输入 $x$ 上所有可能的标签路径集合。

$Z(x)$ 的意义很关键。它让模型不是只把正确标签在某个位置打高分，而是要求**正确整条路径相对于所有候选路径都更优**。这会抑制一种典型错误：局部看似合理、全局却非法。

举一个简化数值例子。句子有两个词，候选标签只有 `B-ORG, I-ORG, O`。假设局部分类器给出：

| 位置 | `B-ORG` | `I-ORG` | `O` |
|---|---:|---:|---:|
| 1 | 2.0 | 1.8 | 0.2 |
| 2 | 0.1 | 1.9 | 1.7 |

如果只看局部最大，第 1 个位置会选 `B-ORG`，第 2 个位置会选 `I-ORG`，这还合理。但若另一个句子第 1 个位置局部最大是 `O`，第 2 个位置局部最大是 `I-ORG`，逐点解码就会输出非法组合 `O, I-ORG`。CRF 会通过转移项对 `O -> I-ORG` 给很低分，迫使模型在全局上改选 `O, O` 或 `B-ORG, I-ORG` 这类更一致的路径。

这就是“全局归一化保障一致性”的具体含义：不是保证绝对不出错，而是把“路径是否合法”纳入目标函数，而不是留到后处理阶段修补。

### 2. span-based：为什么它更容易处理嵌套

span-based NER 直接考虑区间 $(s,e)$。做法通常是先编码出上下文表示 $h_1,\dots,h_n$，然后对每个候选区间构造表示：

$$
g_{s,e} = [h_s \oplus h_e \oplus h_{inside}]
$$

其中 $\oplus$ 是向量拼接，$h_{inside}$ 可以是区间内部词向量的平均、池化或注意力聚合。再通过分类器输出：

$$
P(t \mid s,e,x)=\text{softmax}(W g_{s,e}+b)
$$

这意味着模型直接回答：“从第 $s$ 个词到第 $e$ 个词，这一段是不是 `ORG`、`PER` 或 `O`？” 它不再强制把输出组织成一条单链标签序列，因此天然更适合下面这类情况：

- 嵌套实体：`[Bank of China [Shanghai Branch]]`
- 重叠候选：一个长实体里包含一个短实体
- 长距离边界依赖：实体的开始和结束都要综合判断

可以把两类方法想成两种信息流：

| 方法 | 决策单位 | 主要依赖 | 强项 | 弱项 |
|---|---|---|---|---|
| BIO+CRF | 单词位置 | 相邻标签转移 | 路径一致性强 | 难处理嵌套重叠 |
| span-based | 候选区间 | 起止点与区间语义 | 边界表达直接 | 候选数量多 |

从表达能力上看，BIO 链式模型把信息压缩到“当前位置标签 + 相邻标签转移”上；span-based 则把一个区间整体作为对象建模，因此对“实体是一个整体片段”这个事实表达得更直接。

不过这不代表 span-based 一定更强。它的代价是候选数近似为：

$$
\frac{n(n+1)}{2}=O(n^2)
$$

如果不加长度上限，句子长度翻倍，候选区间数量接近四倍。工程上这会直接影响显存、训练速度和负样本比例。

---

## 代码实现

下面用一个最小可运行例子说明两件事：

1. 如何从 BIO 标签中解析实体区间。
2. 如何做一个最简单的 span 枚举与筛选。

```python
from typing import List, Tuple

def decode_bio(tokens: List[str], tags: List[str]) -> List[Tuple[str, int, int, str]]:
    assert len(tokens) == len(tags)
    entities = []
    start = None
    ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                entities.append((" ".join(tokens[start:i]), start, i - 1, ent_type))
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("-", 1)
        assert prefix in {"B", "I"}

        if prefix == "B":
            if start is not None:
                entities.append((" ".join(tokens[start:i]), start, i - 1, ent_type))
            start = i
            ent_type = cur_type
        else:  # I
            illegal = (start is None) or (ent_type != cur_type)
            if illegal:
                # 常见修复策略：把非法 I-X 当成 B-X
                if start is not None:
                    entities.append((" ".join(tokens[start:i]), start, i - 1, ent_type))
                start = i
                ent_type = cur_type

    if start is not None:
        entities.append((" ".join(tokens[start:]), start, len(tokens) - 1, ent_type))

    return entities

def enumerate_spans(tokens: List[str], max_len: int = 4) -> List[Tuple[int, int, str]]:
    spans = []
    n = len(tokens)
    for s in range(n):
        for e in range(s, min(n, s + max_len)):
            text = " ".join(tokens[s:e+1])
            spans.append((s, e, text))
    return spans

tokens = ["the", "wall", "street", "journal", "launched"]
tags = ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O"]

ents = decode_bio(tokens, tags)
assert ents == [("the wall street journal", 0, 3, "ORG")]

spans = enumerate_spans(tokens, max_len=4)
assert (0, 3, "the wall street journal") in spans
assert len(spans) > 0

print(ents)
print(spans[:5])
```

这个例子没有训练模型，但已经体现了两种表示方式的差异：

- BIO 解码输出依赖一串逐词标签。
- span 方法先枚举区间，再单独判断区间类别。

如果写成 BiLSTM+CRF 的典型流水线，伪代码是：

```python
# x: token ids
emb = embedding(x)
h = bi_lstm(emb)                  # 上下文编码
emission = linear(h)              # 每个位置对每个标签的打分
best_path = crf_decode(emission)  # Viterbi 解码得到最优标签序列
entities = decode_bio(tokens, best_path)
```

span-based 的伪代码则更像：

```python
emb = embedding(x)
h = encoder(emb)

candidates = []
for s in range(n):
    for e in range(s, min(n, s + max_span_len)):
        span_repr = concat(h[s], h[e], pool(h[s:e+1]))
        label = mlp_classifier(span_repr)
        if label != "O":
            candidates.append((s, e, label))
```

真实工程例子可以看电商评论抽取。假设评论是：

`这款华为 Mate 60 Pro 拍照很好，但电池一般`

目标不是只做情感分类，而是抽出：

- `华为` -> `BRAND`
- `Mate 60 Pro` -> `MODEL`

后续系统可以把这些实体送入库存预警、竞品分析、售后聚类等模块。如果实体边界错了，比如把 `Mate 60` 和 `Pro` 切开，后面的商品归并就会失败。此时 NER 不再是“文本美化任务”，而是直接影响下游数据库关联与业务指标。

---

## 工程权衡与常见坑

最常见的工程问题不是“模型太弱”，而是“问题定义和约束没有处理干净”。

| 问题 | 原因 | 缓解方式 |
|---|---|---|
| 非法 BIO 组合 | 局部预测独立，出现 `O -> I-ORG` | 用 CRF 或解码约束 |
| 长实体边界不稳 | 上下文不足，尾词容易掉 | 提升编码器上下文能力 |
| 嵌套实体识别差 | 单链标签无法表达重叠结构 | 改用 span-based 或层次式方法 |
| span 数量过多 | 候选区间是 $O(n^2)$ | 限制最大长度、先做粗筛 |
| 负样本极多 | 大多数 span 都不是实体 | 采样、阈值、两阶段筛选 |
| 跨域效果差 | 新闻和电商的实体分布不同 | 做领域继续训练或重标注 |

### 1. CRF 不是“自动修复一切”

CRF 能约束路径，但它依然依赖上游表示。如果编码器没学到足够上下文，CRF 的转移约束有时会把噪声放大。例如一个品牌和普通名词表面形式接近，局部表示已经错了，CRF 可能只是把这个错误在相邻位置“连贯化”，并不会凭空纠正语义错误。

### 2. Masked-CRF 的作用

Masked-CRF 可以显式禁止某些非法转移，例如不允许：

- `O -> I-X`
- `B-PER -> I-ORG`
- `I-LOC -> I-ORG`

伪代码示意如下：

```python
valid = {
    ("START", "O"), ("START", "B-ORG"),
    ("O", "O"), ("O", "B-ORG"),
    ("B-ORG", "I-ORG"), ("B-ORG", "O"), ("B-ORG", "B-ORG"),
    ("I-ORG", "I-ORG"), ("I-ORG", "O"), ("I-ORG", "B-ORG"),
}

def transition_score(prev_tag, cur_tag):
    if (prev_tag, cur_tag) not in valid:
        return -10**9  # 近似负无穷，表示禁止
    return 0.0
```

这类规则的作用不是提高表达能力，而是减少明显不合法的搜索空间。

### 3. span 模型的计算膨胀

span-based 最大的问题是候选太多。句长 128 时，候选区间大约有 8256 个；如果每个区间都要分类，显存和时间成本都会迅速增加。常用做法有：

- 设置最大实体长度，例如不超过 8 或 10 个词。
- 先做起点/终点粗筛，只保留高置信位置。
- 两阶段结构：先检测候选边界，再做精分类。

### 4. 标注规范比模型结构更重要

NER 标注的一致性要求很高。比如英文组织名中的冠词 `the` 是否计入实体，中文型号里的空格、连字符、括号是否算边界，团队如果不统一，模型会学到互相冲突的信号。很多线上误差，根因其实在标注规范而不是网络结构。

---

## 替代方案与适用边界

NER 并不只有 BIO+CRF 和 span-based 两条路线，还常见：

- 纯 token 分类：每个词直接 softmax，不加 CRF。
- 指针网络/边界预测：分别预测实体起点和终点。
- 生成式方法：把实体抽取改写成文本生成任务。
- MRC 式抽取：把每种实体类型改成一个问答问题。

但如果只讨论主流、可解释、工程上可落地的方案，通常还是在 BIO+CRF 与 span-based 之间权衡。

| 方案 | 适用条件 | 优点 | 局限 |
|---|---|---|---|
| BIO + Softmax | 数据简单，想快速上线 | 实现最简单 | 非法标签难控制 |
| BIO + CRF | 实体不嵌套，边界规则稳定 | 路径一致性好，成熟稳健 | 难表达重叠与嵌套 |
| Span-based | 嵌套多，边界复杂 | 区间表达直接 | 候选多，训练成本高 |
| 边界预测 | 边界比类别更关键 | 对起止点建模明确 | 匹配策略复杂 |
| 生成式抽取 | 任务统一、多任务场景 | 框架统一 | 控制性和稳定性较弱 |

可以用一个简化选择流程理解：

1. 如果实体基本不嵌套，先选 `BIO + CRF`。
2. 如果实体经常重叠或嵌套，优先考虑 `span-based`。
3. 如果数据量很小、上线周期短，先用 `token classifier + 规则后处理` 做基线。
4. 如果下游任务强依赖边界精确匹配，优先选择边界显式建模的方法。

对初级工程师，最重要的判断标准不是“哪篇论文指标更高”，而是你的数据是否满足该方法的假设。

- 当实体嵌套少、标注成本低、业务要求稳定时，BIO+CRF 往往性价比最高。
- 当文档长、实体语义丰富、短实体嵌套长实体频繁出现时，span-based 更能发挥优势。
- 当线上延迟敏感、资源有限时，简单链式方法通常更容易部署。

因此，NER 的核心不是背公式，而是先明确：**你在解决的是“词标签一致性问题”，还是“区间表达能力问题”**。前者偏向 CRF，后者偏向 span。

---

## 参考资料

1. Wikipedia, Named-entity recognition：用于基本定义与任务范围说明。https://en.wikipedia.org/wiki/Named-entity_recognition  
2. MindSpore Sequence Labeling Tutorial：用于 BIO 标注与序列标注流程示意。https://www.mindspore.cn/tutorials/en/r2.7.1/nlp/sequence_labeling.html  
3. ScienceDirect, Conditional Random Field 主题页：用于 CRF 条件概率与全局归一化机制说明。https://www.sciencedirect.com/topics/computer-science/conditional-random-field  
4. Sensors 2022, span-based 与序列标注方法对比：用于误差分析与适用边界讨论。https://www.mdpi.com/1424-8220/22/8/2852  
5. TechTarget, Named Entity Recognition 定义与应用：用于面向工程读者的业务背景补充。https://www.techtarget.com/whatis/definition/named-entity-recognition-NER  
6. Springer / AI Review 关于 span-based NER 的综述：用于区间式建模优缺点补充。https://link.springer.com/article/10.1007/s10462-025-11321-8  
7. ScienceDirect 电商评论实体抽取相关研究：用于品牌/型号真实工程场景示例。https://www.sciencedirect.com/science/article/pii/S1110866525000969

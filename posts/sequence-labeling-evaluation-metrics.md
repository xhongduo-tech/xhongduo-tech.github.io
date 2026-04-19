## 核心结论

序列标注评估分成两层：词级评估和实体级评估。词级评估把每个 token 当成一个分类样本，判断该位置的标签是否预测正确；实体级评估先把标签序列还原成实体片段，再判断实体的边界和类型是否同时正确。

在命名实体识别，也就是 NER 中，主指标通常应使用实体级 F1，而不是词级准确率。NER 的目标不是“让每个位置的标签看起来接近”，而是抽出完整实体。例如真实实体是 `原发性高血压`，模型只预测出 `高血压`，词级可能只错少量 token，但实体级严格评估会判为错误，因为实体 span 不一致。

精确率、召回率、F1 的通用形式是：

$$
P = \frac{TP}{TP + FP}, \quad
R = \frac{TP}{TP + FN}, \quad
F1 = \frac{2PR}{P + R}
$$

| 维度 | 词级评估 | 实体级评估 |
|---|---|---|
| 统计对象 | 单个 token 的标签 | 一个完整实体 span |
| 命中条件 | 当前 token 标签正确 | 起止边界和实体类型都正确 |
| 适用场景 | 调试标签分布、定位局部错误 | 汇报 NER 主结果 |
| 常见误区 | 把 token accuracy 当成模型抽实体能力 | 忽略 strict / partial 的差异 |
| 对 `O` 的处理 | 可能计入 accuracy，但通常不计入实体 P/R/F1 | `O` 不构成实体 |

---

## 问题定义与边界

序列标注，是指给输入序列中的每个 token 分配一个标签。token 是模型处理的最小文本单元，可以是字、词或子词。NER 是序列标注的典型任务，目标是从文本中找出人名、地点、疾病、药品等实体。

评估前必须先定义两个问题：评估对象是什么，什么算预测正确。否则两个模型即使在同一数据集上测试，结果也可能不可比。

实体级严格评估中，通常把 gold 实体集合记为 $G$，把预测实体集合记为 $P$。每个实体可表示为：

$$
(start, end, type)
$$

其中 `start` 是起始位置，`end` 是结束位置，`type` 是实体类型。严格匹配下，命中集合大小为：

$$
C = |G \cap P|
$$

只有 `start`、`end`、`type` 全部一致，实体才进入交集。

常见标签方案包括 BIO 和 BIOES。BIO 中，`B-X` 表示类型为 `X` 的实体开始，`I-X` 表示实体内部，`O` 表示非实体。BIOES 在此基础上增加 `E-X` 和 `S-X`，分别表示实体结束和单 token 实体。

| 边界条件 | 词级评估 | 实体级评估 |
|---|---|---|
| 是否依赖标签方案 | 依赖，但可逐位置比较 | 强依赖，需要先解析 span |
| 是否统计 `O` | accuracy 常会统计，P/R/F1 通常不统计 | 不统计，`O` 不是实体 |
| 是否要求边界正确 | 不直接要求 | strict 模式必须正确 |
| 是否要求类型正确 | 当前 token 类型正确即可 | 整个实体类型必须正确 |
| 是否可跨句 | 不应跨句 | 不应跨句，空行分句很重要 |

玩具例子：文本为 `张三 来自 北京`。gold 标签是 `B-PER O B-LOC`，预测标签是 `B-PER O O`。词级看有两个位置正确；实体级看，`张三/PER` 命中，`北京/LOC` 漏掉。

---

## 核心机制与推导

词级评估本质是逐位置分类。假设只统计实体标签，不把 `O` 当作正类，则：

$$
P_{tok} = \frac{TP_{tok}}{TP_{tok} + FP_{tok}}, \quad
R_{tok} = \frac{TP_{tok}}{TP_{tok} + FN_{tok}}
$$

$$
F1_{tok} = \frac{2P_{tok}R_{tok}}{P_{tok} + R_{tok}}
$$

实体级评估本质是集合匹配。先把标签序列恢复成 span，再比较 gold span 和 pred span：

$$
P_{ent} = \frac{C}{|P|}, \quad
R_{ent} = \frac{C}{|G|}
$$

$$
F1_{ent} = \frac{2P_{ent}R_{ent}}{P_{ent} + R_{ent}}
$$

看一个最小例子：

```text
Token:  John Smith went Paris today
Gold:   B-PER I-PER O    B-LOC O
Pred:   B-PER I-PER O    O     B-LOC
```

gold 实体有两个：`[1,2] PER` 和 `[4,4] LOC`。pred 实体也有两个：`[1,2] PER` 和 `[5,5] LOC`。其中只有 `John Smith` 命中，`Paris` 被预测到相邻位置，边界错了，所以实体级只命中 1 个。

词级可能只错 2 个位置；实体级会认为 LOC 实体完全没抽对。这就是实体级指标更严格的原因。

| 情况 | span 是否正确 | type 是否正确 | strict 实体级结果 |
|---|---:|---:|---|
| 二者都对 | 是 | 是 | 命中 |
| 边界正确但类型错 | 是 | 否 | 错误 |
| 类型正确但边界错 | 否 | 是 | 错误 |
| 二者都错 | 否 | 否 | 错误 |

真实工程例子：医疗 NER 接在关系抽取前面，用来抽 `疾病名` 和 `药品名`。如果 gold 是 `原发性高血压/疾病`，pred 是 `高血压/疾病`，模型确实抓到了核心疾病词，但少了修饰边界。对于下游“药品治疗疾病”的关系抽取，这可能改变实体对齐、检索召回和医学含义，因此 strict 实体级评估会判错。

---

## 代码实现

实现实体级评估时，不应直接计算 token accuracy。正确流程是：输入标签序列，解析实体区间，比较 gold 和 pred，计算 P/R/F1。

```python
from typing import List, Tuple, Set

Span = Tuple[int, int, str]

def parse_spans(tags: List[str]) -> Set[Span]:
    spans = set()
    start = None
    ent_type = None

    def close(end: int):
        if start is not None and ent_type is not None:
            spans.add((start, end, ent_type))

    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            close(i - 1)
            start = None
            ent_type = None
            continue

        prefix, typ = tag.split("-", 1)

        if prefix == "B" or ent_type != typ:
            close(i - 1)
            start = i
            ent_type = typ
        elif prefix == "I":
            continue
        else:
            raise ValueError(f"Unsupported tag prefix: {prefix}")

    return spans

def compute_f1(gold_tags: List[str], pred_tags: List[str]):
    gold = parse_spans(gold_tags)
    pred = parse_spans(pred_tags)
    correct = len(gold & pred)

    precision = correct / len(pred) if pred else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1, gold, pred

gold = ["B-PER", "I-PER", "O", "B-LOC", "O"]
pred = ["B-PER", "I-PER", "O", "O", "B-LOC"]

p, r, f1, gold_spans, pred_spans = compute_f1(gold, pred)

assert gold_spans == {(0, 1, "PER"), (3, 3, "LOC")}
assert pred_spans == {(0, 1, "PER"), (4, 4, "LOC")}
assert (p, r, f1) == (0.5, 0.5, 0.5)
```

| 输入格式 | 示例 | 注意点 |
|---|---|---|
| 一句一个标签列表 | `["B-PER", "I-PER", "O"]` | 适合代码内部计算 |
| CoNLL 多列格式 | `token gold pred` | 常用于脚本评估 |
| 空行分句 | 句子之间保留空行 | 防止跨句实体误连 |
| BIO 标签 | `B-X/I-X/O` | 最常见 |
| BIOES 标签 | `B-X/I-X/E-X/S-X/O` | 需要专门解析 |

如果使用 `seqeval`，需要固定 `mode`、`scheme` 等配置。不同默认行为可能处理非法标签的方式不同，例如 `I-PER` 出现在句首时，有的实现会宽松地当作实体开始，有的实现会严格报错或不计入合法实体。

---

## 工程权衡与常见坑

词级指标容易虚高，尤其在 `O` 很多的任务中。假设一个数据集 95% 的 token 都是 `O`，模型全部预测为 `O`，token accuracy 可能达到 95%，但它一个实体也没抽出来，实体级 F1 是 0。

评估脚本比很多新手想象得更重要。模型没有变，只要标签方案、句子分隔或 strict 规则变化，分数就可能变化。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 把 token accuracy 当主指标 | 高估模型能力 | 主结果报实体级 F1 |
| BIO / BIOES 混用 | span 解析错误 | 评估前统一标签方案 |
| strict / partial 混淆 | 结果不可横向比较 | 报告中写清匹配规则 |
| 空行丢失 | 跨句实体误连 | 保留 CoNLL 分句格式 |
| `O` 过多 | 词级 accuracy 虚高 | 分开报告 `O` 与实体指标 |
| 非法转移未检查 | 解析结果不稳定 | 评估前校验标签合法性 |

评估前可以做几类检查：

```python
def validate_bio(tags):
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        prefix, typ = tag.split("-", 1)
        assert prefix in {"B", "I"}
        if prefix == "I":
            prev = tags[i - 1] if i > 0 else "O"
            assert prev.endswith("-" + typ), f"Illegal I tag at {i}: {tag}"

validate_bio(["B-DISEASE", "I-DISEASE", "O"])
```

CoNLL 风格主结果通常使用 micro 统计。micro 统计，是先把所有类型、所有句子的 TP、FP、FN 加总，再计算 P/R/F1。它适合报告整体实体抽取能力。macro 统计，是先分别算每个类别的指标，再取平均，更容易反映小类别表现，但会让低频类别对总分影响更大。

---

## 替代方案与适用边界

严格匹配不是唯一标准。部分任务更关心“有没有大致找到相关文本”，这时可以使用 partial、overlap 或 token-level 指标。但这些指标不能和 strict F1 直接比较。

一种常见 relaxed 指标是按重叠比例打分：

$$
score = \frac{|span_{gold} \cap span_{pred}|}{|span_{gold} \cup span_{pred}|}
$$

这个公式衡量两个 span 的重叠程度，但它不是 CoNLL 标准实体 F1。它适合业务分析，不适合直接替代标准 NER benchmark 结果。

| 评估方式 | 命中定义 | 适用场景 | 风险 |
|---|---|---|---|
| strict | span 和 type 完全一致 | 标准 NER、论文对比、CoNLL 风格评估 | 对边界错误零容忍 |
| partial | span 部分重叠且类型可匹配 | 医学文本分析、标注边界模糊任务 | 分数偏高，定义不统一 |
| overlap | 只要有重叠就算部分正确 | 检索、召回型系统 | 可能奖励过长预测 |
| token-level | token 标签逐位正确 | 调试模型、分析错误位置 | 不能代表实体抽取质量 |

医疗场景中，`高血压` 和 `原发性高血压` 的差异可能影响诊断语义和关系抽取，因此 strict 边界很重要。粗粒度检索场景中，用户只需要找到包含相关疾病词的文档，partial match 可能更符合业务目标。

结论是：能横向比较的通常是同一数据集、同一标签方案、同一匹配规则下的结果。strict F1 可以和 strict F1 比，partial F1 只能和定义相同的 partial F1 比。评估指标不是装饰项，而是任务目标的形式化定义。

---

## 参考资料

1. [CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](https://aclanthology.org/W03-0419/)
2. [Output Example conlleval](https://www.clips.uantwerpen.be/conll2000/chunking/output.html)
3. [Various criteria in the evaluation of biomedical named entity recognition](https://link.springer.com/article/10.1186/1471-2105-7-92)
4. [seqeval](https://github.com/chakki-works/seqeval)

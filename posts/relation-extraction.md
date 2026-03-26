## 核心结论

关系抽取的目标，是把句子里已经识别出的两个实体，判定为某一种预定义语义关系。这里的“关系”就是实体之间的语义连接，比如“位于”“雇佣”“配偶”“药物-不良反应”。它最终输出的是三元组，例如 `(John Doe, Paris, LIVES_IN)`，供知识图谱、搜索问答、推荐和风控系统使用。

对初学者来说，可以把它先理解成一个“带约束的分类问题”：先圈出实体对，再从有限标签集合里选最合适的一类。输入不是整篇文章的开放式理解，而是某个实体对及其上下文；输出也不是自由文本，而是固定标签集合中的一个标签。

当前工程上最常见的主线仍然是“监督学习 + 深度编码器”。深度编码器就是把句子变成向量的模型，常见实现是 BERT 或它的变体；分类头再把这个向量映射到关系类别。远程监督是另一条重要路线，它用知识库里的现成三元组自动给文本打标签，能低成本制造大量训练数据，但标签噪声很大，不能直接当成干净真值使用。

一个最小玩具例子是句子 `John Doe lives in Paris`。如果实体识别阶段已经给出 `John Doe` 和 `Paris`，那么关系抽取阶段只需要判断它们在这个上下文中最可能是哪一种关系。此时 `LIVES_IN` 的得分应高于 `WORKS_AT`、`BORN_IN` 等其他候选标签，于是输出 `(John Doe, Paris, LIVES_IN)`。

监督学习和远程监督的工程差异可以先看下面这张表：

| 方案 | 标签成本 | 标签质量 | 可扩展性 | 主要风险 | 典型用途 |
| --- | --- | --- | --- | --- | --- |
| 人工监督 | 高 | 高 | 中 | 数据少、长尾类别难覆盖 | 高精度垂直场景 |
| 远程监督 | 低 | 中到低 | 高 | 错标、漏标、语义漂移 | 大规模预训练或弱监督启动 |
| 规则/模板 | 中 | 高于粗糙远程监督 | 低 | 迁移差、维护成本高 | 小规模高一致性领域 |

---

## 问题定义与边界

关系抽取通常建立在“实体已经知道”这一前提上。这里的“实体”就是文本中有明确指代的对象，比如人名、公司名、地点名、药品名。最常见的输入可以写成：

$$
(e_1, e_2, ctx)
$$

其中 $e_1$ 和 $e_2$ 是两个实体，$ctx$ 是它们所在的上下文，通常是一句话或一个局部片段。输出是关系集合 $R$ 中的一个标签：

$$
r \in R
$$

如果写成完整映射，就是：

$$
g:(e_1,e_2,ctx)\rightarrow r
$$

对白话一点的理解是：先固定两个实体，再问“这两个实体在这句话里是什么关系”。

例如句子 `John Doe lives in Paris`，先圈出 `John Doe` 和 `Paris`，再从 `LIVES_IN`、`WORKS_AT`、`BORN_IN` 这类有限列表里选一个最匹配的标签。这就是最基础的关系抽取。

边界要讲清楚，否则初学者很容易把它和其他任务混在一起：

| 任务 | 解决什么问题 | 输出形式 |
| --- | --- | --- |
| 实体识别 | 找出“谁是实体” | 实体 span 与类型 |
| 关系抽取 | 判断两个实体之间的关系 | 关系标签或三元组 |
| 事件抽取 | 找出事件及其参与角色 | 事件触发词、角色、参数 |
| 开放信息抽取 | 不依赖固定标签，直接抽关系短语 | 开放三元组 |

关系抽取通常有几个常见限制。

第一，很多数据集默认一个实体对在一个上下文里只标一个主关系。如果同一句里存在多关系，工程上常拆成多条训练样本分别处理。第二，关系集合通常是封闭的，也就是训练前先定义好。第三，实体类型会限制可选关系空间，例如 `person-location` 常考虑 `LIVES_IN`、`BORN_IN`，而 `drug-adverse_event` 根本不应进入 `MARRIED_TO` 这样的标签集合。这种“类型约束”能显著减少误判。

因此，一个更实用的形式化定义是：

$$
r^*=\arg\max_{r\in R(e_1,e_2)} P(r\mid e_1,e_2,ctx)
$$

其中 $R(e_1,e_2)$ 表示在实体类型约束下允许的候选关系集合。它不是所有标签，而是经过先验过滤后的子集。

---

## 核心机制与推导

关系抽取的核心机制可以拆成两步：先编码，再分类。

第一步是编码。编码的意思是把“两个实体以及它们的上下文”变成一个数值向量。记为：

$$
\mathbf{x}=f(e_1,e_2,ctx)
$$

这里的 $f$ 就是编码器。早期常见的是人工特征、CNN、RNN；现在主流是 Transformer，最常见代表就是 BERT。BERT 可以把句子中每个 token 的上下文语义编码成向量，因此它既能看到实体本身，也能看到它们周围的触发词，比如 `lives in`、`works for`、`caused by`。

如果一句话经过 BERT 后得到最后一层隐表示 $H=[h_1,\dots,h_n]$，而实体 $e_1,e_2$ 分别对应 token 区间 $S_1,S_2$，那么一种很常见的实体表示方法是对 span 做均值池化：

$$
v_1=\frac{1}{|S_1|}\sum_{i\in S_1} h_i,\quad
v_2=\frac{1}{|S_2|}\sum_{i\in S_2} h_i
$$

再把句级表示和实体表示拼接：

$$
\mathbf{x}=[h_{\text{[CLS]}};v_1;v_2]
$$

这里 `[CLS]` 可以理解为“整句摘要向量”，它大致汇总了全句语义。拼接后，模型同时看到“整句是什么意思”和“两个目标实体分别是什么”。

第二步是分类。把上面的向量送入一个线性层，再经过 softmax：

$$
p(r\mid e_1,e_2,ctx)=\text{Softmax}(W\mathbf{x}+b)
$$

最终预测关系为：

$$
r^*=\arg\max_{r\in R}\text{Softmax}(W\mathbf{x}+b)
$$

这套推导本质上就是“表示学习 + 多分类”。所谓“表示学习”，就是让模型自己学出一个好用的特征向量 $\mathbf{x}$，而不是人工写一堆句法规则。

玩具例子可以手工走一遍。

句子：`John Doe lives in Paris`

实体对：`(John Doe, Paris)`

候选标签：`[LIVES_IN, WORKS_AT, BORN_IN, NO_RELATION]`

如果编码器学到了 `lives in` 强烈提示居住关系，那么分类器可能给出：

| 标签 | 得分概率 |
| --- | --- |
| LIVES_IN | 0.91 |
| WORKS_AT | 0.05 |
| BORN_IN | 0.02 |
| NO_RELATION | 0.02 |

于是输出 `LIVES_IN`。

真实工程例子更有代表性。比如临床文本里有一句：`Rash was likely caused by amoxicillin.` 实体是 `Rash` 和 `amoxicillin`。系统需要判断这是 `ADE_OF`，也就是“不良反应属于该药物”，还是仅仅共现无关。这里触发词 `caused by` 很关键，实体类型 `adverse_event-drug` 也会强约束候选关系空间。医疗关系抽取之所以重要，是因为它能直接支持临床知识图谱、药物警戒和病历结构化。

---

## 代码实现

实际工程通常分三层：样本构造、编码器、分类头。

样本构造阶段，每条样本至少要有：
1. 原句文本
2. 两个实体的起止位置
3. 关系标签

如果是远程监督，还要先拿知识库中的三元组 $(e_1,e_2,r)$ 去对齐文本。对齐规则最经典的一条是假设：如果知识库里存在某关系，且一句话同时提到了这两个实体，那么这句话就可能表达该关系。注意这里的“可能”非常关键，因为这正是远程监督噪声的来源。

下面先给一个可运行的 Python 玩具实现。它不是 BERT，只是用关键词打分模拟“关系分类器”的数据流，重点是帮助理解输入、输出和断言测试。

```python
from math import exp

LABELS = ["LIVES_IN", "WORKS_AT", "BORN_IN", "NO_RELATION"]

def softmax(scores):
    exps = [exp(s) for s in scores]
    s = sum(exps)
    return [v / s for v in exps]

def extract_relation(sentence, e1, e2):
    text = sentence.lower()
    scores = {
        "LIVES_IN": 0.0,
        "WORKS_AT": 0.0,
        "BORN_IN": 0.0,
        "NO_RELATION": 0.0,
    }

    if "lives in" in text or "settled in" in text:
        scores["LIVES_IN"] += 3.0
    if "works at" in text or "employed by" in text:
        scores["WORKS_AT"] += 3.0
    if "born in" in text:
        scores["BORN_IN"] += 3.0

    if e1.lower() in text and e2.lower() in text:
        for k in ["LIVES_IN", "WORKS_AT", "BORN_IN"]:
            scores[k] += 0.5
    else:
        scores["NO_RELATION"] += 2.0

    ordered_scores = [scores[label] for label in LABELS]
    probs = softmax(ordered_scores)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return LABELS[best_idx], dict(zip(LABELS, probs))

label, probs = extract_relation("John Doe lives in Paris.", "John Doe", "Paris")
assert label == "LIVES_IN"
assert probs["LIVES_IN"] > probs["WORKS_AT"]

label2, probs2 = extract_relation("John Doe works at OpenAI.", "John Doe", "OpenAI")
assert label2 == "WORKS_AT"
assert abs(sum(probs2.values()) - 1.0) < 1e-9

print(label, probs)
print(label2, probs2)
```

真正的深度学习实现会把“关键词规则”换成“预训练编码器 + 线性分类层”。下面是一个接近真实工程的 PyTorch 伪代码，展示 BERT 到关系分类的主数据流：

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BertRelationClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden * 3, num_labels)

    def span_mean(self, sequence_output, spans):
        reps = []
        for b, (start, end) in enumerate(spans):
            reps.append(sequence_output[b, start:end].mean(dim=0))
        return torch.stack(reps, dim=0)

    def forward(self, input_ids, attention_mask, e1_spans, e2_spans):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state                 # [B, T, H]
        cls = seq[:, 0, :]                             # [B, H]
        e1 = self.span_mean(seq, e1_spans)             # [B, H]
        e2 = self.span_mean(seq, e2_spans)             # [B, H]
        x = torch.cat([cls, e1, e2], dim=-1)           # [B, 3H]
        logits = self.classifier(x)                    # [B, C]
        return logits

# 训练时通常配合 CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()
```

如果要做一个能上线的最小系统，流程通常是：

| 步骤 | 输入 | 输出 |
| --- | --- | --- |
| 实体识别 | 原始文本 | 实体 span 与类型 |
| 构造实体对 | 实体列表 | 候选实体对 |
| 编码与分类 | 候选实体对 + 上下文 | 关系标签 |
| 生成三元组 | 标签结果 | `(head, relation, tail)` |

初学者容易忽略的一点是，关系抽取并不只是在“句子编码”上做文章，实体标记方式本身也很重要。常见做法包括：
- 在输入文本中插入实体边界标记
- 用实体类型 embedding
- 拼接实体 span 平均向量
- 对实体之间的最短依赖路径单独建模

这些设计都会影响 $\mathbf{x}$ 的质量。

---

## 工程权衡与常见坑

工程上最常见的误区，不是模型太弱，而是任务定义太松。

第一个坑是候选实体对爆炸。如果一句话有 $n$ 个实体，朴素做法会生成 $O(n^2)$ 个实体对。实体一多，负样本会急剧增多，模型会学成“几乎都预测无关系”。对策通常是先用实体类型过滤、窗口限制、句法启发或轻量二分类做候选剪枝。

第二个坑是标签定义不稳。比如 `works at`、`founded`、`invested in` 是否都算 `ORG_AFFILIATION`，如果标注规范不统一，模型再大也只会学到混乱边界。关系抽取很依赖标注协议，尤其是方向性和细粒度层级。

第三个坑是规则法可解释，但迁移差。规则法就是人工写模板，例如“`<人名> lives in <地点>` 判为 `LIVES_IN`”。它在小规模、格式稳定的数据上很有效，但一旦文本改写成 `Paris is where John Doe settled`，原规则就容易失效。规则维护成本会随着领域扩张快速上升。

第四个坑是远程监督噪声。远程监督的典型假设是：知识库里有 `(John Doe, Paris, LIVES_IN)`，那么所有同时出现 `John Doe` 和 `Paris` 的句子都可标成 `LIVES_IN`。这显然会错。比如：

- `John Doe visited Paris last summer.`  
- `John Doe left Paris in 2018.`

这两句都出现同一实体对，但并不表达“居住于”。这类错标会把模型带偏，使它把“visited”也学成 `LIVES_IN` 的证据。

常见坑和对策可以压缩成下表：

| 常见坑 | 具体表现 | 对策 |
| --- | --- | --- |
| 规则难迁移 | 换领域后命中率骤降 | 自动模板、统计特征、逐步过渡到预训练模型 |
| 负样本过多 | 模型几乎都预测 `NO_RELATION` | 候选剪枝、重采样、类别权重 |
| 远程监督噪声 | 共现即正例，误标严重 | 多实例学习、句子注意力、噪声感知 loss |
| 实体边界错误 | 上游实体识别错误向下游传播 | 联合建模或加入置信度过滤 |
| 关系方向错 | `(A,B,r)` 与 `(B,A,r)` 混淆 | 显式建模 head/tail 与 direction |

这里的“多实例学习”可以用一句白话解释：不要把某一条句子当成绝对真值，而是把“同一实体对对应的多句文本”看成一个包，只要求包里至少有一句真的表达该关系。这样可以缓和远程监督的错标问题。句子注意力也是类似思想，它让模型自动给真正表达关系的句子更高权重。

真实工程例子里，临床文本经常出现缩写、模板句、跨句关系和省略表达。比如药物与不良反应不一定出现在同一句，`caused by` 这类显式触发词也不总在场。这时纯句级分类会遇到边界，往往要升级到文档级关系抽取，或者配合规则后处理。

---

## 替代方案与适用边界

如果数据很少、领域非常稳定，规则/模板法仍然值得先做。原因很简单：它上线快、错误模式可解释、便于和业务专家对齐。例如客服质检场景里，如果只关心 `品牌-型号`、`故障-部件` 两三类关系，且文本模式比较固定，规则法可能是最省成本的起点。

如果你有一批高质量标注数据，希望在较复杂文本上追求更好的召回与泛化，监督学习更合适。它的前提是标签体系明确、实体识别质量尚可，并且关系类型不会频繁变更。

如果你几乎没有人工标注，但有知识库和海量文本，远程监督是合理起点。它的价值不是直接得到最终模型，而是先用低成本拿到一个足够大的预训练数据池，再配合噪声抑制、少量人工校正或主动学习逐步提纯。

还有一类替代路线是联合抽取。联合抽取就是把“实体识别”和“关系抽取”一起学，而不是先识别实体再抽关系。它适合上游实体边界不稳定、级联误差很大的场景，但实现更复杂、训练也更难。

下面给出一个简化选择表：

| 方案 | 适用场景 | 数据需求 | 优点 | 边界 |
| --- | --- | --- | --- | --- |
| 规则/模板 | 文本模式稳定、专家规则明确 | 很少标注 | 上线快、可解释 | 迁移差、维护重 |
| 监督学习 | 有标注语料、追求泛化 | 中到大量人工标注 | 精度高、可持续优化 | 标注成本高 |
| 远程监督 | 有知识库、缺人工标注 | KB + 大文本语料 | 扩展快、覆盖广 | 噪声大、需要后续治理 |
| 联合抽取 | 实体边界不稳定 | 标注更复杂 | 减少级联误差 | 实现与调参成本高 |

对零基础到初级工程师，一个务实路线通常是：
1. 先把任务压成“实体对分类”
2. 用少量规则做基线，确认标签定义
3. 有标注就上 BERT 分类器
4. 数据不够再引入远程监督，但把噪声治理当成必做项，而不是可选优化

---

## 参考资料

1. [relex 文档：What is semantic relation extraction?](https://relex-docs.readthedocs.io/en/latest/philosophy.html)  
   作用：给出关系抽取的基本定义、实体对编码方式，以及“实体识别 -> 实体配对 -> 关系分类”的模块化流程。  
   重点：适合理解任务边界和最小例子，如 `John Doe` 与 `Paris` 的 `LIVES_IN` 关系。

2. [BMC Bioinformatics 2021：Biomedical relation extraction via knowledge-enhanced reading comprehension](https://link.springer.com/article/10.1186/s12859-021-04534-5)  
   作用：展示如何把关系抽取形式化为可学习的监督任务，并讨论远程监督构造的数据集与知识增强方法。  
   重点：说明实体对候选构造、上下文建模与大规模自动标注数据的价值和局限。

3. [AI 2024：Improving Distantly Supervised Relation Extraction with Multi-Level Noise Reduction](https://www.mdpi.com/2673-2688/5/3/84)  
   作用：专门讨论远程监督关系抽取中的噪声问题。  
   重点：把噪声拆成词级和句级，并用多实例学习、噪声抑制思路改进远程监督训练。

4. [PMC：Relation Extraction from Clinical Narratives Using Pre-trained Language Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC7153059/)  
   作用：给出临床关系抽取的真实工程场景，对比 CNN-RNN、JOINT 与 BERT 类方法。  
   重点：说明预训练语言模型在医疗关系抽取中的优势，也能帮助理解“真实工程例子”为什么不能只靠关键词规则。

5. [PMC 综述：Relation extraction: advancements through deep learning and entity-related features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256580/)  
   作用：汇总 ACE05 等数据集上的模型表现。  
   重点：文中总结了 Yang 等工作中 CNN+BERT 在 ACE05 Chinese 上约 80.30 F1 的基线结果，可作为“BERT-CNN 是高效基线”的补充背景。

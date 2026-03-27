## 核心结论

关系抽取的目标，是把一句话里已经识别出的实体对，映射成一个固定关系标签。实体就是文本里可指代真实对象的片段，比如“Tim Cook”“Apple”；关系就是两个实体之间的语义连接，比如“雇佣”“位于”“配偶”。如果句子是“Tim Cook 领导 Apple”，先拿到实体对 `<Tim Cook, Apple>`，再由模型输出关系分布，例如“雇佣”概率 0.92、“创始人”概率 0.05、“无关系”概率 0.03，那么工程上通常直接判为“雇佣”。

简化流程可以写成：

`实体识别 -> 构造实体对与上下文 -> 句子编码 -> 分类器 -> 语义标签`

这里“句子编码”指把自然语言转成向量，也就是一串数字表示；“分类器”指根据这些数字判断它更像哪一种关系。对零基础读者，可以把它理解成“句子先变成数字，再让模型判断哪条关系最亮”。

关系抽取不是只有一种做法。模板方法依赖规则和关键词，优点是好解释；监督学习依赖人工标注数据，优点是精度高；远程监督用知识库自动打标，优点是便宜可扩展，但噪声大；BERT-CNN 则是工程里很常见的基线，把 BERT 的上下文表示和 CNN 的局部模式提取结合起来，在成本、效果、实现复杂度之间比较平衡。

概率决策本质上是在做“最大后验选择”。如果模型输出的是
$$
p(y=\text{雇佣}\mid s)=0.92
$$
就表示在当前参数下，句子 $s$ 被判为“雇佣”的置信度最高。线上系统一般会再配一个阈值，比如大于 0.8 才写入知识图谱，否则进入人工审核或回退为“无关系”。

---

## 问题定义与边界

关系抽取首先不是实体识别。实体识别负责找出“人名、公司名、地点名”这些片段；关系抽取是在实体已经给定后，判断这些实体之间是什么关系。因此它的标准输入不是一整篇文章，而更像三元组：

`输入 = 句子 + 实体1位置 + 实体2位置`

输出则是一个离散标签集合中的某一个元素，例如：

| 关系名 | 白话解释 | 典型句子 |
|---|---|---|
| 位于 | 某个实体在某个地点 | “Google 在旧金山设有总部” |
| 雇佣 | 某个人与某组织存在任职关系 | “Tim Cook 领导 Apple” |
| 配偶 | 两个人有婚姻关系 | “A 与 B 于 2012 年结婚” |
| 创始人 | 某人创建某组织 | “马斯克创办了 SpaceX” |
| 无关系 | 句子里没有明确表达目标关系 | “Google 今天发布了新模型，旧金山天气晴朗” |

新手最容易混淆的点有三个。

第一，输入必须带实体边界。比如“Google 在旧金山”这句话，如果系统不知道要看的是 `<Google, 旧金山>`，就无法确定关系抽取的目标对象。  
第二，输出必须来自预定义集合。模型不是自由生成一句解释，而是在“位于/雇佣/配偶/无关系”等标签中选一个。  
第三，不是出现两个实体就一定有关系。句子里可能只是共现，也就是两个实体同时出现，但没有语义连接。

玩具例子可以看下面两句：

1. “Google 在旧金山设有大型办公室。”
2. “Google 发布了新模型，旧金山今天有开发者大会。”

两句都包含 `<Google, 旧金山>`，但第一句表达“位于”，第二句更适合标成“无关系”或“其他”。这就是关系抽取的边界：它关心的是句子中是否**明确表达**了关系，而不是实体有没有同时出现。

如果把任务再形式化一点，可以写成：

给定句子 $s$ 和实体对 $(e_1,e_2)$，预测
$$
y \in \mathcal{R}
$$
其中 $\mathcal{R}$ 是关系集合，通常包含一个负类标签，比如“无关系”或 “Other”。在很多公开数据集里，负类占比不低，因此模型不仅要学会识别正关系，还要学会拒绝误报。

---

## 核心机制与推导

最传统的做法是模板方法。模板就是人工写规则，例如：

- `X 领导 Y` -> 雇佣
- `X 位于 Y` -> 位于
- `X 的妻子是 Y` -> 配偶

它的好处是可解释，坏处也明显：语言表达非常多样，“领导”“担任 CEO”“出任首席执行官”“执掌”都可能表达接近关系，模板很难覆盖完。

监督学习的思路是把句子表示成向量，再做多分类。设句子及实体标记后的输入为 $s$，编码器输出向量 $h(s)$，那么最常见的预测形式是：
$$
y=f(s)=\mathrm{softmax}(W\cdot h(s)+b)
$$
这里：

- $h(s)$ 是句子的向量表示，可以来自 BERT、CNN、LSTM 等编码器
- $W,b$ 是分类层参数
- `softmax` 的作用是把原始分数转成各关系的概率分布

如果真实标签是 one-hot 向量 $t$，预测概率是 $\hat y$，训练时常用交叉熵损失：
$$
\mathcal{L}=-\sum_{k=1}^{|\mathcal{R}|} t_k \log \hat y_k
$$
白话解释就是：正确标签的概率越低，惩罚越大。

以“Tim Cook 领导 Apple”为例，工程里常先把实体做显式标记，例如：

`[E1] Tim Cook [/E1] 领导 [E2] Apple [/E2]`

这样编码器更容易知道“谁和谁之间的关系”才是关注重点。之后一般有两条信息流：

1. BERT 输出每个 token 的上下文向量  
2. CNN 在这些向量上滑动卷积核，提取局部 n-gram 模式

可以把 BERT 理解成“读懂整句上下文”，把 CNN 理解成“抓住局部短语模式”。例如两个卷积核可能分别对下面模式响应更强：

- 卷积核 A：关注“领导 Apple”
- 卷积核 B：关注“Tim Cook 领导”

再经过最大池化，保留每个卷积核最强的激活，最后拼接成句向量送入 softmax 分类器。如果“雇佣”维度得分最高，且概率 0.92，就输出“雇佣”。

一个简化图示如下：

`带实体标记的句子 -> BERT 编码 -> token 向量序列 -> CNN 卷积/池化 -> 句向量 -> softmax -> 关系标签`

远程监督则是另一条路线。它不是先人工标注句子，而是拿知识库里的三元组自动对齐语料。假设知识库里有 `(Tim Cook, 雇佣, Apple)`，那么所有同时包含 “Tim Cook” 和 “Apple” 的句子，都可能被自动标成“雇佣”。这极大降低了标注成本，但也引入核心噪声：并不是每个共现句子都真的表达该关系。

真实工程里通常会把“句子级关系抽取”和“包级关系抽取”区分开。所谓“包”，就是同一实体对的多条句子集合。多实例学习会假设：一个包里只要有一条句子真正表达该关系，整个包就可以视为正例。这比“每句都强行打正标签”更稳健。

---

## 代码实现

下面先给一个可运行的玩具版本。它不是 BERT，也不是生产模型，而是用关键词分数模拟“句子编码 -> softmax -> 标签选择”的最小闭环。作用是帮助理解数据流。

```python
import math

LABELS = ["雇佣", "位于", "配偶", "无关系"]

KEYWORDS = {
    "雇佣": ["领导", "任职", "担任", "CEO", "加入"],
    "位于": ["位于", "在", "总部", "坐落"],
    "配偶": ["妻子", "丈夫", "结婚", "配偶"],
}

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def extract_relation(sentence, entity1, entity2):
    # 最小版特征：关键词命中计数
    scores = {"雇佣": 0.0, "位于": 0.0, "配偶": 0.0, "无关系": 0.5}

    if entity1 not in sentence or entity2 not in sentence:
        raise ValueError("实体不在句子中")

    for label, words in KEYWORDS.items():
        for w in words:
            if w in sentence:
                scores[label] += 1.0

    logits = [scores[label] for label in LABELS]
    probs = softmax(logits)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])

    return {
        "label": LABELS[best_idx],
        "prob": probs[best_idx],
        "distribution": dict(zip(LABELS, probs)),
    }

toy = extract_relation("Tim Cook 领导 Apple", "Tim Cook", "Apple")
assert toy["label"] == "雇佣"
assert toy["prob"] > 0.4

toy2 = extract_relation("Google 在旧金山设有总部", "Google", "旧金山")
assert toy2["label"] == "位于"

toy3 = extract_relation("Google 发布了新模型，旧金山今天有大会", "Google", "旧金山")
assert "无关系" in toy3["distribution"]
print(toy, toy2, toy3)
```

上面这段代码对应的是最朴素的分类框架：

`text -> feature -> logits -> softmax -> label`

真实实现会把“feature”从关键词计数替换成神经网络表示。一个常见的 BERT-CNN 伪代码如下：

```python
# sentence_ids: [batch, seq_len]
# e1_mask/e2_mask: 实体位置掩码，用来取实体表示
# bert_hidden: [batch, seq_len, hidden]

bert_hidden = bert(sentence_ids, attention_mask)

# CNN 提取局部模式
cnn_feat = Conv1D(filters=128, kernel_size=3)(bert_hidden)
cnn_feat = relu(cnn_feat)
cnn_feat = global_max_pool(cnn_feat)

# 取实体向量，强调目标实体对
e1_vec = masked_average(bert_hidden, e1_mask)
e2_vec = masked_average(bert_hidden, e2_mask)

# 融合句子特征与实体特征
h = concat([cnn_feat, e1_vec, e2_vec])

logits = Dense(num_relations)(h)
prob = softmax(logits)
loss = cross_entropy(prob, gold_label)
```

关键变量解释：

- `bert_hidden`：BERT 输出的上下文表示，意思是每个词都带上了整句语义
- `cnn_feat`：卷积后的局部模式特征，适合抓“担任 CEO”“位于 北京”这类短语
- `e1_vec/e2_vec`：两个实体自身的表示，避免模型只看句子整体、忽略目标实体
- `logits`：每个关系的原始分数
- `prob`：归一化后的概率分布

真实工程例子是知识图谱构建流水线。比如你有一批公司公告、新闻稿、百科文本，前面已经跑过实体识别，得到“人物、公司、地点”实体；关系抽取模块就接在后面，把 `<人物, 公司>` 判断成“任职”，把 `<公司, 地点>` 判断成“总部地点”，再写回图数据库。这里 BERT-CNN 的价值不是“最前沿”，而是“训练成本和线上延迟通常还可控”。

---

## 工程权衡与常见坑

关系抽取的难点不在“能不能训练一个分类器”，而在“标签、样本、噪声、上线目标是否匹配”。

先看三类主流方法的取舍：

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 模板/规则 | 可解释、冷启动快、无需训练 | 覆盖差、维护成本高、迁移困难 | 关系类型少、表述稳定、强合规场景 |
| 监督学习 | 精度高、边界清晰、便于评估 | 标注贵、长尾关系难学 | 重点关系有限、可投入标注资源 |
| 远程监督 | 低成本扩数据、适合大规模语料 | 标签噪声大、假阳性多 | 有知识库可对齐、需要快速扩容 |

最常见的坑是把“共现”当“关系表达”。知识库里有 `(Tim Cook, 雇佣, Apple)`，如果你把所有包含这两个实体的句子都打成“雇佣”，就会出现误标，比如：

“Tim Cook 在采访中评价了 Apple 新财报。”

这句话实体共现了，但未直接表达雇佣关系。对新手可以直接记一句：自动标注一定会带来假阳性。

第二个坑是“无关系”类别建得太弱。实际语料里负样本很多，如果训练时只关注正类，线上误报会很严重。很多系统最终不是败在“分不清雇佣和创始人”，而是败在“什么都想抽出来”。

第三个坑是长尾关系。所谓长尾，就是样本特别少的关系类别，比如“母公司”“品牌代言”“子行政区”。BERT-CNN 这类基线在头部关系上通常够用，但面对少样本关系时容易偏向频次高的类别。

第四个坑是句子太长、触发词太远。比如法律文本、医学论文、招股说明书里，两个实体可能隔很多从句。CNN 擅长抓局部模式，但不一定擅长复杂依赖；这时引入依存句法图、GNN 或基于最短依赖路径的特征，往往更有帮助。

第五个坑是训练目标和业务目标不一致。论文里常看宏平均 F1，但线上系统更关心“高置信写库的准确率”“人工审核命中率”“某几类核心关系的召回”。如果 KPI 不一致，模型选型就会跑偏。

---

## 替代方案与适用边界

BERT-CNN 是实用基线，但不是唯一选择。选择模型时，最重要的不是“谁更先进”，而是“数据量、噪声水平、推理成本、句法复杂度”是否匹配。

| 模型 | 关键特征 | 适配场景 |
|---|---|---|
| 模板/规则 | 人工定义触发词和模式 | 关系固定、可解释性优先 |
| CNN/PCNN | 强调局部短语与实体区间特征 | 中小数据、延迟敏感 |
| BERT + Linear | 实现简单、基线稳 | 数据质量较好、快速上线 |
| BERT-CNN | 上下文语义 + 局部模式融合 | 通用关系分类、工程基线 |
| BERT + Attention/PCNN | 对远程监督噪声更稳 | 实体对多句聚合、包级预测 |
| BERT + GNN/GCN | 融合依存句法或图结构 | 长距离依赖明显、句法信息重要 |
| 大语言模型提示法 | 少样本迁移灵活 | 标注极少、允许较高推理成本 |

一个简化决策树可以写成：

`数据很少 -> 先规则或小模型`
`数据中等且标签干净 -> BERT 或 BERT-CNN`
`数据很多但噪声大 -> 远程监督 + 多实例注意力`
`句法依赖强、长距离关系多 -> BERT + GNN/GCN`

真实工程例子可以看语义与句法融合路线。`Scientific Reports` 的一篇关系抽取工作将 BERT 表示与图卷积网络结合，在 DuIE2.0 和 SemEval-2010 Task 8 上验证，通过依存句法图传播信息，并用图池化和噪声抑制模块减少无关节点干扰。这个方向说明一个事实：当句子结构复杂时，只靠平面序列编码未必足够，句法图能补充“谁依赖谁”的结构信息。

但它也有明显边界。GNN 方案要依赖句法分析器，中文和跨领域文本里句法树本身可能带噪；模型也更重、更难部署。如果你的业务语料是短新闻标题或电商描述，BERT-CNN 可能已经更划算。如果你的语料是长句科研文献、法律合同，句法增强才更值得投入。

---

## 参考资料

1. [SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals](https://aclanthology.org/S10-1006/) ，ACL Anthology。要点：关系分类经典数据集说明，常见设置为 8000 条训练、2717 条测试，包含 `Other` 负类。  
2. [Distant Supervision for Relation Extraction without Labeled Data](https://nlp.stanford.edu/pubs/mintz09.pdf) ，Stanford NLP。要点：远程监督开创性工作，用知识库三元组自动对齐语料，显著降低人工标注成本，但默认假设会引入噪声。  
3. [Relation classification via BERT with piecewise convolution and focal loss](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0257092) ，PLOS One。要点：展示 BERT 与分段卷积结合的关系分类思路，在 SemEval-2010 Task 8 上验证了 BERT + CNN/PCNN 路线的实用性。  
4. [关系抽取之远程监督方法总结](https://blog.csdn.net/qq_27668313/article/details/112369730) ，CSDN。要点：对远程监督、PCNN、句子级注意力进行了中文综述，适合作为入门梳理材料。  
5. [Relationship extraction between entities with long distance dependencies and noise based on semantic and syntactic features](https://www.nature.com/articles/s41598-025-00915-5) ，Scientific Reports。要点：结合 BERT、图卷积、图池化与噪声抑制模块，在 DuIE2.0 和 SemEval-2010 Task 8 上验证语义与句法融合方案。  
6. [Large-scale Exploration of Neural Relation Classification Architectures](https://aclanthology.org/D18-1250.pdf) ，EMNLP 2018。要点：系统比较多种神经关系分类架构，提醒工程上不要只看单一数据集成绩，应结合数据特征与负类分布做选型。

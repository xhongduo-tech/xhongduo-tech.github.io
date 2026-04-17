## 核心结论

LLM + KG 的“预训练增强”指的是：在模型还没进入下游任务之前，就把知识图谱里的实体、关系、邻接结构，和自然语言语料一起送进 Transformer 训练。知识图谱（KG，Knowledge Graph）可以先理解成“用三元组存事实的结构化数据库”，例如 `(Harry Potter, 母亲, Lily Potter)`。这样做的目的，不是单纯给模型多喂一些文本，而是让模型在表示层就同时学习两种信息：

1. 文本里的上下文共现关系。
2. 图谱里的显式事实关系。

结论可以压缩成一句话：如果任务明显依赖事实、实体关系、专业术语约束，那么把 KG 注入预训练阶段，通常比只在微调阶段外挂知识更稳定，尤其在知识问答、领域分类、实体相关推理中更容易得到持续收益。

这类方法的代表路线有三条：

| 方法 | 核心思路 | 注入位置 | 主要收益 | 主要代价 |
| --- | --- | --- | --- | --- |
| ERNIE 系列 | 将实体级知识与语言目标联合训练 | 预训练目标与表示层 | 提升知识型理解、少样本泛化 | 训练设计更复杂 |
| K-BERT | 把三元组沿句子实体位置展开，加入 soft-position 和 visible matrix | 输入层 | 改造相对轻，适合已有 BERT 系统 | 噪声控制要求高 |
| CoLAKE | 把文本 token 图和 KG 子图合成 word-knowledge graph 联合预训练 | 编码层与目标层 | 文本表示与图谱表示同步学习 | 图构建、负采样、算力成本高 |

对新手最直观的理解是：以前 BERT 主要靠“词和词经常一起出现”来学知识；知识增强预训练则额外告诉模型，“这些词背后对应的是同一个实体，而且实体之间有确定关系”。这会让模型在回答事实型问题时，不只是“像见过”，而是“在表示里显式对齐过”。

ERNIE 3.0 一类工作常把多个训练目标联合起来，写成：

$$
L_{\text{total}}=L_{\text{MLM}}+L_{\text{DLM}}+L_{\text{SR}}+L_{\text{SD}}+L_{\text{UKTP}}
$$

这里的联合损失本质上是在做一件事：让模型同时学会补全文本、预测顺序、判断句段关系、以及对齐“文本里提到的实体”和“图谱里存在的事实”。

---

## 问题定义与边界

问题先说清楚：为什么不能只靠大规模文本预训练？

因为语言模型从文本中学到的大量知识，本质上是统计相关性。统计相关性可以理解成“哪些词经常一起出现”。它很强，但不总是可靠。比如“某药物治疗某疾病”“某人物和某公司的关系”，在语料里可能出现频率不高，或者表达方式很多变，模型就容易记混、忘记、或者只学到模糊模式。

知识图谱注入要解决的是另一个层面的问题：把事实写成结构，让模型在预训练阶段就接触“谁和谁是什么关系”。于是每个 token 的表示不只来自句子上下文，还来自相关实体和关系的邻域。

边界也必须明确。它更适合以下任务：

- 事实问答、实体分类、专业领域文本理解
- 金融、法律、医疗等实体约束强的任务
- 低资源场景下需要借助已有知识库补信息的任务

它不一定适合以下场景：

- 强开放生成、创意写作
- 知识变化极快但图谱更新跟不上的场景
- 图谱噪声很高、实体链接错误很多的场景

玩具例子先看一句简单话：

“Harry Potter 是 Lily Potter 的儿子。”

如果只做普通语言建模，模型看到的是词序列；如果做知识增强预训练，系统先通过实体链接把“Harry Potter”“Lily Potter”识别成图谱实体，再把 `(Harry Potter, 母亲, Lily Potter)` 或其等价关系注入模型。这样模型并不是只学“儿子”和“母亲”这两个词常一起出现，而是在表示层接触到“这两个名字之间有可计算的关系”。

真实工程例子可以看金融评级文本：

“这家银行获得 AAA 评级。”

对人来说这是自然句子，对机器来说如果没有领域知识，它未必知道“银行”“AAA”“评级”之间的结构关系。K-BERT 的思路是把类似 `(银行, 评级, AAA)` 的三元组沿着句子中“银行”这个实体展开插入，再用 soft-position 和 visible matrix 约束其影响范围。soft-position 可以白话理解成“知识插进来了，但不完全打乱原句位置”；visible matrix 可以理解成“谁能看到谁”的注意力白名单，避免无关知识污染整句语义。

下面这个对比能帮助理解边界：

| 模型 | 是否使用 KG | KG 是否上下文化 | 是否联合 MLM 预训练 | 典型问题 |
| --- | --- | --- | --- | --- |
| 原生 BERT | 否 | 否 | 是 | 事实依赖强时容易只学到表面共现 |
| KEPLER 类方法 | 是 | 较弱，常偏静态实体表示 | 是，但文本与实体对齐方式较受限 | 文本空间与知识空间可能分裂 |
| CoLAKE | 是 | 是，实体/关系在上下文中编码 | 是，且联合图谱目标 | 训练成本高，图构建复杂 |

所以核心边界不是“有没有知识”，而是“知识是否在预训练时参与了上下文化表示学习”。

---

## 核心机制与推导

三条路线分别看。

ERNIE 的关键点是“知识感知目标”。它不是只做 Masked Language Modeling，也就是遮住部分词再预测，而是把实体级目标、句段级目标和知识对齐目标一起训练。MLM 是“猜被遮住的词”；DLM 是“按顺序生成后续内容”；SR 和 SD 用于学习句段顺序与距离；UKTP 则负责把文本和知识三元组对齐。

其中 UKTP 可以粗略理解成：给模型一段文本和一条候选知识关系，模型要判断它们是否匹配。论文里常通过头实体和尾实体标记，例如 `[HD]`、`[TL]` 这类位置标记，让模型知道“这段文本里哪个位置对应头实体，哪个位置对应尾实体”，再结合关系分类完成对齐。

从损失函数上看：

$$
L_{\text{total}}=L_{\text{MLM}}+L_{\text{DLM}}+L_{\text{SR}}+L_{\text{SD}}+L_{\text{UKTP}}
$$

这五项的作用可以直接拆开：

| 损失项 | 作用 | 直白解释 |
| --- | --- | --- |
| $L_{\text{MLM}}$ | 恢复被遮住的 token | 让模型理解局部上下文 |
| $L_{\text{DLM}}$ | 自回归生成 | 让模型理解顺序与长程依赖 |
| $L_{\text{SR}}$ | 句段重排序 | 让模型识别结构是否合理 |
| $L_{\text{SD}}$ | 句段距离预测 | 让模型理解段落间相对关系 |
| $L_{\text{UKTP}}$ | 文本与知识对齐 | 让模型把句子里的实体和图谱关系对应起来 |

K-BERT 的关键点则不是重做整个目标体系，而是在输入层做“知识注入”。它沿着句子中的实体展开知识树，再把树线性化送入 Transformer。问题在于：一旦把很多知识 token 直接插进去，原句位置会被打乱，注意力也会扩散到无关部分。因此它设计了两个控制器：

1. soft-position  
把知识 token 挂在原实体附近，但不完全按真实线性位置重排，减少位置编码失真。

2. visible matrix  
限制注意力连接。一个知识 token 不应该让整句所有词都关注到它，只应让相关实体邻域看到它。

CoLAKE 走得更远。它把文本序列看成 word graph，把 KG 三元组看成 knowledge graph，再合成 word-knowledge graph。这里“图”不是画图，而是“节点 + 边”的数据结构。文本 token 是节点，实体和关系也是节点，边表示它们的连接关系。Transformer 编码时，实际上在同时处理词级上下文和知识结构。

玩具例子可以写成：

句子：`Harry Potter inherited from Lily Potter.`  
图谱：`(Harry Potter, child_of, Lily Potter)`

如果训练时同时把文本中的 `Harry` 遮住，把图谱中的 `Lily Potter` 也遮住，那么模型必须联合两种线索来恢复。这样学到的表示不再只是“Harry 常和 Potter 一起出现”，而是“这个句子事件和这个图谱事实是同一组实体关系”。

这也是 CoLAKE 比“静态实体向量外挂”更强的地方：实体、关系、文本 token 都在同一个上下文化编码过程中更新，而不是先各自训练再硬拼起来。

---

## 代码实现

工程上可以把流程拆成 5 步：

1. 对文本分词。
2. 做实体链接，把文本片段映射到 KG 实体。
3. 为每个实体查询邻居三元组。
4. 把文本 token 与实体/关系节点线性化为模型输入。
5. 训练时同时计算文本目标和知识目标。

下面给一个可运行的简化版 Python 玩具实现。它不是完整 Transformer，而是演示“句子 + 三元组展开 + visible mask”的核心思路。

```python
from dataclasses import dataclass

@dataclass
class Triple:
    head: str
    relation: str
    tail: str

def inject_kg(tokens, entity_spans, triples_by_entity):
    """
    tokens: 原句 token 列表
    entity_spans: {实体名: 在 tokens 中的索引}
    triples_by_entity: {实体名: [Triple, ...]}
    """
    output = []
    soft_positions = []
    owners = []  # 记录每个注入 token 属于哪个原实体，便于构建 visible matrix

    for i, tok in enumerate(tokens):
        output.append(tok)
        soft_positions.append(i)
        owners.append(i)

        if tok in entity_spans and tok in triples_by_entity:
            for triple in triples_by_entity[tok]:
                # 简化: 在实体后面插入 relation 和 tail
                output.append(f"<rel:{triple.relation}>")
                soft_positions.append(i)   # soft-position: 挂靠在实体原位置
                owners.append(i)

                output.append(f"<ent:{triple.tail}>")
                soft_positions.append(i)
                owners.append(i)

    n = len(output)
    visible = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            same_owner = owners[i] == owners[j]
            both_text = i < len(tokens) and j < len(tokens)
            # 文本 token 全互见；知识 token 只在同一实体局部可见
            if both_text or same_owner:
                visible[i][j] = 1

    return output, soft_positions, visible

tokens = ["Harry", "is", "Lily", "son"]
entity_spans = {"Harry": 0, "Lily": 2}
triples = {
    "Harry": [Triple("Harry", "mother", "Lily")],
    "Lily": [Triple("Lily", "child", "Harry")]
}

seq, pos, vis = inject_kg(tokens, entity_spans, triples)

assert seq[0] == "Harry"
assert "<rel:mother>" in seq
assert "<ent:Lily>" in seq
assert len(seq) == len(pos) == len(vis)
assert vis[0][1] == 1   # 原句 token 互相可见
assert vis[0][2] == 1   # 原句 token 互相可见
print(seq)
```

这段代码对应的工程含义是：

- `soft_positions` 表示知识 token 绑定在原实体附近，而不是重新占据新的绝对句法位置。
- `visible` 是可见矩阵，控制注意力范围，避免每个知识 token 都影响整句。
- `owners` 是一个简化技巧，用来表示“这段知识属于哪个实体”。

如果继续往预训练方向扩展，伪代码大致是：

```python
for sentence in corpus:
    words = tokenize(sentence)
    entities = entity_link(words)
    wk_graph = build_wk_graph(words, entities, kg)
    input_ids, soft_pos, visible = linearize(wk_graph)

    output = transformer(
        input_ids=input_ids,
        soft_positions=soft_pos,
        visible_matrix=visible
    )

    loss = (
        mlm_loss(output, masked_tokens=True)
        + uktp_loss(output, graph=wk_graph)
        + relation_cls_loss(output)
    )
```

真实工程例子可以设成法律合同审核。假设句子里出现“甲方”“担保责任”“连带责任”，知识图谱里存有合同法术语和责任关系。普通 BERT 可能主要依赖相邻词分布，而知识增强模型可以在预训练时就接触“主体-责任类型-法律后果”的结构。这样在条款分类、风险点抽取、相似案例检索中，模型更容易把术语映射到稳定关系，而不是只记表述模板。

CoLAKE 一类实现还会引入负采样。负采样可以白话理解成“给模型一些错误候选，让它学会分辨真假关系”。例如每个实体采 200 个负例，就是说除了正确尾实体外，再随机或按难度选 200 个错误尾实体，让模型不要只会背正样本。批量规模如 2048，则反映这类训练通常依赖较大算力和较强吞吐。

---

## 工程权衡与常见坑

第一个权衡是知识覆盖 vs. 噪声控制。

KG 注入不是越多越好。你给每个实体扩 1 条三元组，和扩 50 条三元组，效果完全不同。覆盖高意味着模型看到更多事实；但噪声也会急剧上升，尤其当实体链接不准、图谱边质量不齐时，模型会被错误关系拉偏。

一个常见坏例子是把无关事实全量插入。比如句子讨论“重力异常数据处理”，结果给“gravity”错误链接到了流行文化条目，再把无关三元组插入输入。模型可能会把整个句子的表示拉向错误领域。K-BERT 的 visible matrix 之所以重要，就是要让这些知识只在相关实体附近“亮起来”，而不是全局广播。

第二个权衡是模型效果 vs. 训练成本。

CoLAKE 这类联合预训练方案通常比原生 BERT 昂贵得多。原因包括：

- 训练前要做实体链接和图检索
- 输入更长，注意力开销更高
- 需要负采样与额外分类头
- 数据流水线更复杂，难以纯文本并行化

可以把几个关键成本记住：

| 维度 | 轻量做法 | 重量做法 | 影响 |
| --- | --- | --- | --- |
| 知识接入 | 微调或推理时外挂 | 预训练联合注入 | 重量做法效果更稳，但成本高 |
| 图谱展开 | 1-hop 少量邻居 | 多跳、全子图 | 多跳更全，但噪声显著增加 |
| 负采样 | 少量随机负例 | 大规模难负例，如每实体 200 个 | 区分能力更强，但训练更慢 |
| batch | 小 batch | 大 batch，如 2048 | 稳定性更好，但显存和吞吐要求高 |

另一个坑是“表示分裂”。表示分裂可以理解成“文本 embedding 说一种语言，实体 embedding 说另一种语言”。如果 KG 向量和文本向量来自完全不同的训练过程，再在后面简单拼接，模型很容易学出两套不协调的空间。表现就是：文本判断像一个系统，知识判断像另一个系统，最后融合层压力过大。CoLAKE、ERNIE 这类方法的优势正在于联合训练，把它们尽量压到同一个表示空间里。

还要注意知识时效性。预训练阶段注入的 KG 一旦过时，模型学到的是“固化事实”。这对百科类稳定知识没问题，但对金融主体状态、法规版本、药物禁忌更新就必须谨慎。此时往往需要把预训练增强和检索增强配合使用，而不是只依赖静态知识注入。

---

## 替代方案与适用边界

如果不做知识增强预训练，还有三类常见替代方案。

第一类是 vanilla BERT，也就是普通预训练语言模型。它的优势是简单、成熟、便宜。若任务主要靠语气、主题、局部语义，比如通用情感分类、短文本粗分类，常常已经够用。

第二类是 KEPLER 一类“文本 + 实体表示联合学习”方案。它比纯 BERT 更重视实体，但很多时候实体表示仍偏静态，或者上下文化不如 CoLAKE 充分。它适合“需要知识，但不一定要重构整个预训练图结构”的中间路线。

第三类是推理时外挂知识，例如 RAG、检索式 KG 查询、prompt-based grounding。它的特点是不用改预训练本体，而是在推理时把相关事实查出来再喂给模型。优点是更新快、部署灵活；缺点是系统链路更长，召回不稳定时效果会波动。

可以简化成下面的比较：

| 方案 | 训练阶段改动 | 是否强依赖 KG | 部署复杂度 | 适合场景 |
| --- | --- | --- | --- | --- |
| vanilla BERT | 低 | 否 | 低 | 通用分类、知识依赖低 |
| KEPLER 类 | 中 | 中 | 中 | 需要实体表示但预算有限 |
| 知识增强预训练 | 高 | 高 | 中到高 | 法律、金融、医疗等强事实任务 |
| 推理时检索 KG | 低 | 高 | 高 | 知识更新频繁、不能重训模型 |

这里给一个真实工程判断准则。

如果你只是做商品评论情感分类，先上 base BERT 往往最划算，因为任务主信号在情绪词和语义模式，不在稳定实体关系。  
如果你做的是合同风险审查、临床术语归一、产业链主体关系抽取，那么知识增强预训练更有价值，因为错误常来自“事实结构没学稳”，而不是“句子没看懂”。

对新手可以这样记：  
普通 BERT 更像“会读很多文档的人”；  
知识增强预训练更像“读文档时还同步查过结构化资料并做了标注的人”。  
前者擅长一般理解，后者更擅长带事实约束的理解。

所以适用边界很明确：

- 任务知识密度高、实体约束强、术语稳定，优先考虑预训练增强。
- 知识变动快、预算有限、系统重训练困难，优先考虑推理时检索或微调外挂知识。
- 如果数据量足、任务简单，不要为了“高级方法”强行引入 KG，因为维护图谱、链接实体、清洗关系本身就是不小的工程成本。

---

## 参考资料

- CoLAKE: Contextualized Language and Knowledge Embedding. 重点看 word-knowledge graph、联合 MLM 与知识节点训练设计。  
  https://www.researchgate.net/publication/348345684_CoLAKE_Contextualized_Language_and_Knowledge_Embedding

- K-BERT: Enabling Language Representation with Knowledge Graph. 重点看 knowledge layer、soft-position、visible matrix。  
  https://www.researchgate.net/publication/335880304_K-BERT_Enabling_Language_Representation_with_Knowledge_Graph

- ERNIE 3.0 Titan 相关论文与综述。重点看多任务联合损失、知识对齐目标 UKTP，以及知识驱动任务上的性能提升。  
  https://www.emergentmind.com/topics/ernie-3-0-titan

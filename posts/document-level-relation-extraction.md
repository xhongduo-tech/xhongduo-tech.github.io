## 核心结论

文档级关系抽取（Document-level Relation Extraction, DocRE，白话说就是“不是只看一句话，而是看整篇文章来判断两个实体有没有关系”）解决的是跨句关系识别问题。它不再假设关系证据一定出现在单句中，而是允许多个句子共同支持一个三元组 $(head, relation, tail)$。

这件事的本质变化有三点：

1. 预测对象从“句子里的两个 mention”变成“文档里的两个实体”。
2. 证据来源从“局部词序列”变成“全文上下文 + 共指链 + 句间结构”。
3. 建模方式从“单句分类”变成“篇章级信息聚合与推理”。

对初学者最重要的结论是：如果一条关系需要把两句甚至三句拼起来才能看懂，那么句子级关系抽取天然不够，必须引入文档级建模。常见做法是构建文档图，把 mention、实体、句子甚至 token 当成节点，把依存关系、共指链接、句间邻接等当成边，再用 GNN 或 Transformer 聚合信息。

玩具例子：

- 句子 1：`阿司匹林可降低炎症反应。`
- 句子 2：`这种药物常用于类风湿关节炎治疗。`

这里“这种药物”指代“阿司匹林”。如果只看第二句，并不知道“这种药物”是谁；如果只看第一句，也看不到“类风湿关节炎”。只有把两句连起来，才能推断出“阿司匹林 - 可能用于治疗 - 类风湿关节炎”。

文档图中的典型消息传递可写成：

$$
h_v^{(l+1)}=\sigma\!\left(\sum_{(u,r)\in\mathcal{N}(v)}\frac{W_r h_u^{(l)}}{|\mathcal{N}_r(v)|}\right)
$$

其中 $v$ 是当前节点，$u$ 是邻居节点，$r$ 是边类型，$\mathcal{N}(v)$ 是邻居集合。白话说，就是“当前节点从不同类型的邻居那里接收信息，再更新自己的表示”。

---

## 问题定义与边界

文档级关系抽取的输入通常不是一条句子，而是一个完整文档及其实体标注信息。输出则是实体对之间的关系，很多数据集还要求给出支持该关系的证据句集合。

可以把任务形式化为：

- 输入：文档 $D=\{s_1,s_2,\dots,s_n\}$，实体集合 $E=\{e_1,e_2,\dots,e_m\}$，每个实体可能对应多个 mention。
- 输出：若干三元组 $(e_i,r,e_j)$，其中 $r$ 来自关系标签集合 $\mathcal{R}$，也可能是 `NONE`，表示无关系。

下面这张表先把输入输出说清楚：

| 元素 | 含义 | 典型来源 | 输出中是否直接使用 |
|---|---|---|---|
| 文档 | 多个句子组成的全文 | 原始文本 | 是 |
| mention | 实体在文本中的一次出现 | NER 或人工标注 | 间接使用 |
| 实体 | 多个 mention 的聚合对象 | mention 聚类/共指 | 是 |
| 共指链 | 指向同一实体的 mention 集合 | 共指消解 | 是 |
| 句间结构 | 句子顺序、段落邻接 | 文档结构 | 是 |
| 关系标签 | 如“位于”“治疗”“创始人” | 标注体系 | 是 |
| 证据句 | 支撑关系成立的句子索引 | 数据标注或模型解释 | 可选 |

边界也要说清楚，否则任务会被说得过大：

1. 这里讨论的是“只依赖当前文档”的抽取，不假设外部知识库参与推理。
2. 输入通常默认已经有实体 mention，或者至少能先做 NER；如果实体识别本身错误，关系抽取会直接受污染。
3. 共指消解（coreference resolution，白话说就是“判断‘他’‘该公司’‘这种药物’到底指谁”）经常是隐含前提，但它本身也是一个有误差的子任务。
4. 文档级不等于无限长上下文。实际工程里一旦篇幅过长，编码成本和显存成本会迅速上升。

用 DocRED 风格的最小例子说明边界更直接。假设有两句文本，实体 mention 分布如下：

- `句子 0`：`Marie Curie discovered radium.`
- `句子 1`：`The scientist later won a Nobel Prize.`

如果实体集合里有 `Marie Curie`、`radium`、`Nobel Prize`，那么“the scientist”需要和 `Marie Curie` 共指对齐。此时某些关系可能由句子 0 支持，另一些关系需要句子 0 和句子 1 联合支持。DocRED 的 `evidence` 字段正是用来标识支撑关系的句子编号。

---

## 核心机制与推导

文档级关系抽取的核心不是“分类器更大”，而是“先把分散证据组织起来，再做分类”。常见机制可以拆成四步。

第一步是构建节点。

常见节点类型有三类：

- mention 节点：实体在文本中的一次具体出现。
- entity 节点：同一实体的多个 mention 聚合后的抽象节点。
- sentence 节点：句子级上下文载体。

有的模型还加入 token 节点或段落节点，但对初学者先理解上述三类就够了。

第二步是构建边。

边的作用是告诉模型“哪些信息应该直接传播”。常见边类型如下：

| 边类型 | 连接对象 | 作用 |
|---|---|---|
| 句内依存边 | token/mention 到 token/mention | 建模句法关联 |
| 共指边 | mention 到 mention | 把“它”“该药物”等接回真实实体 |
| mention-entity 边 | mention 到实体 | 聚合同一实体的多次出现 |
| 句间邻接边 | 相邻句子或跨句 mention | 传递篇章上下文 |
| 实体-句子边 | 实体到其出现句 | 保留证据位置 |

第三步是表示传播。

如果使用 GNN，核心思想是“节点不断吸收邻居信息”。前面的公式再看一遍：

$$
h_v^{(l+1)}=\sigma\!\left(\sum_{(u,r)\in\mathcal{N}(v)}\frac{W_r h_u^{(l)}}{|\mathcal{N}_r(v)|}\right)
$$

它表达的是分类型消息传递：

- $h_u^{(l)}$：邻居节点在第 $l$ 层的表示。
- $W_r$：不同边类型用不同参数变换。
- $|\mathcal{N}_r(v)|$：按边类型归一化，避免某类邻居数量过多时把其他信息淹没。
- $\sigma$：激活函数，比如 ReLU。

白话解释：如果一个 mention 节点通过共指边连到另一个 mention，又通过句子边连到上下文句子，那么经过几轮传播后，它的表示里就不再只有本句局部词义，还带上了跨句证据。

第四步是关系分类。

最终我们要预测的是实体对 $(e_i,e_j)$ 的关系，而不是单个节点。所以通常会构造实体对表示，例如：

$$
z_{ij}=[h_{e_i};h_{e_j};h_{e_i}\odot h_{e_j};|h_{e_i}-h_{e_j}|]
$$

然后把 $z_{ij}$ 输入分类器得到各关系的分数。这里 $\odot$ 表示逐元素乘法，白话说就是“让模型显式看到两个实体之间的交互模式”。

再给一个新手版图示理解：

- mention 节点：`阿司匹林`、`这种药物`、`类风湿关节炎`
- entity 节点：`阿司匹林实体`、`类风湿关节炎实体`
- sentence 节点：`句1`、`句2`

连边方式：

- `阿司匹林` -> `阿司匹林实体`
- `这种药物` -> `阿司匹林实体`
- `类风湿关节炎` -> `类风湿关节炎实体`
- `阿司匹林` <-> `这种药物` 通过共指或实体聚合间接连通
- `句1` <-> `句2` 通过句间邻接相连

这样传播几层后，`阿司匹林实体` 就能感知到第二句中的疾病信息，`类风湿关节炎实体` 也能感知到第一句中的药物信息，最后分类器再决定是否存在“治疗”关系。

真实工程例子是医学知识库构建。很多论文摘要里，化学物质在第一句出现，疾病在后文出现，作用机制在中间句补充。如果只做句子级抽取，会漏掉大量跨句关系；如果做文档图推理，则能把化学物质、靶点、疾病、试验结论连接起来，用于知识库更新或医学问答检索。

---

## 代码实现

下面给一个可运行的玩具实现。它不是训练版 DocRE 模型，而是一个“把文档、mention、共指、句间边转成图，再基于聚合结果做简单关系判断”的最小示例。重点是让结构跑通。

```python
from collections import defaultdict

def build_graph(doc_sentences, mentions, coref_groups):
    graph = defaultdict(list)
    nodes = {}

    # sentence nodes
    for i, sent in enumerate(doc_sentences):
        sid = f"sent:{i}"
        nodes[sid] = {"type": "sentence", "text": sent}

    # mention nodes
    for i, m in enumerate(mentions):
        mid = f"mention:{i}"
        nodes[mid] = {
            "type": "mention",
            "text": m["text"],
            "sent_id": m["sent_id"],
            "entity": m["entity"],
        }
        sid = f"sent:{m['sent_id']}"
        graph[mid].append(("in_sentence", sid))
        graph[sid].append(("has_mention", mid))

    # entity nodes
    entity_names = sorted({m["entity"] for m in mentions})
    for e in entity_names:
        eid = f"entity:{e}"
        nodes[eid] = {"type": "entity", "text": e}

    # mention -> entity
    for i, m in enumerate(mentions):
        mid = f"mention:{i}"
        eid = f"entity:{m['entity']}"
        graph[mid].append(("belongs_to", eid))
        graph[eid].append(("has_mention", mid))

    # sentence adjacency
    for i in range(len(doc_sentences) - 1):
        a, b = f"sent:{i}", f"sent:{i+1}"
        graph[a].append(("next_sentence", b))
        graph[b].append(("prev_sentence", a))

    # coreference links
    for group in coref_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = f"mention:{group[i]}", f"mention:{group[j]}"
                graph[a].append(("coref", b))
                graph[b].append(("coref", a))

    return nodes, graph

def aggregate_entity_context(nodes, graph, entity_name):
    eid = f"entity:{entity_name}"
    visited = set([eid])
    frontier = [eid]
    texts = []

    # toy 2-hop propagation
    for _ in range(2):
        new_frontier = []
        for node in frontier:
            for edge_type, nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    new_frontier.append(nb)
                    texts.append((edge_type, nodes[nb]["text"]))
        frontier = new_frontier
    return texts

def predict_treat_relation(nodes, graph, drug, disease):
    drug_ctx = " ".join(text for _, text in aggregate_entity_context(nodes, graph, drug))
    disease_ctx = " ".join(text for _, text in aggregate_entity_context(nodes, graph, disease))
    joined = (drug_ctx + " " + disease_ctx).lower()

    trigger_words = ["treat", "therapy", "used for", "治疗", "用于"]
    return any(t in joined for t in trigger_words)

doc = [
    "Aspirin reduces inflammation.",
    "This drug is used for rheumatoid arthritis treatment."
]

mentions = [
    {"text": "Aspirin", "sent_id": 0, "entity": "Aspirin"},
    {"text": "This drug", "sent_id": 1, "entity": "Aspirin"},
    {"text": "rheumatoid arthritis", "sent_id": 1, "entity": "RheumatoidArthritis"},
]

# mention 0 and mention 1 are coreferent
coref_groups = [[0, 1]]

nodes, graph = build_graph(doc, mentions, coref_groups)

assert "entity:Aspirin" in nodes
assert predict_treat_relation(nodes, graph, "Aspirin", "RheumatoidArthritis") is True
```

这个玩具代码做了四件事：

1. 把句子、mention、实体都建成节点。
2. 把 mention 和实体、mention 和句子、句子和句子连边。
3. 用 `coref_groups` 把同指 mention 接起来。
4. 做一个两跳聚合，再用简单触发词判断关系。

真实工程实现会更复杂，通常是下面这个流程：

```python
for sentence in doc:
    mentions += extract_mentions(sentence)

corefs = resolve_coreference(doc, mentions)
graph = build_graph(mentions, corefs, sentence_edges, dependency_edges)

token_repr = encoder(doc)                  # Transformer 编码全文或窗口
node_repr = gather_node_repr(token_repr)   # 抽取 mention/entity/sentence 表示
node_repr = gnn(graph, node_repr)          # 图上传播
pair_repr = build_pair_features(node_repr) # 实体对表示
relation_scores = classifier(pair_repr)    # 多分类或多标签
```

如果把它画成数据流，就是：

`文档 -> 实体/mention抽取 -> 共指消解 -> 文档图构建 -> 编码器 -> GNN/Transformer聚合 -> 实体对分类器`

工程上常见的组合是：

- 底层编码器用 Transformer 提供局部上下文表示。
- 上层图网络处理跨句、跨 mention、跨实体传播。
- 最后做实体对分类，并可同时预测证据句。

这种“序列编码 + 图推理”的组合比纯图或纯序列更常见，因为前者擅长建模局部语义，后者擅长建模显式结构。

---

## 工程权衡与常见坑

文档级关系抽取在论文里看起来很顺，但工程实现有几个坑几乎一定会踩。

第一个坑是类别失衡。大多数实体对其实没有关系，`NONE` 样本通常远多于正样本。假设一个文档里有 100 个实体，那么实体对数量接近 $100 \times 99$，但真正存在标注关系的只占很小一部分。模型如果直接按原始分布训练，最容易学到的策略就是“全预测 NONE”，训练集准确率可能还不低，但召回会非常差。

第二个坑是长文档成本。Transformer 的自注意力复杂度近似是 $O(n^2)$，这里的 $n$ 是 token 数。文档一长，显存和时延都会明显上升。很多系统不得不做截断、滑动窗口，或者先做句级编码再做文档级聚合。

第三个坑是共指误差传播。共指消解一旦错，把两个不相干 mention 合并到一个实体上，后续图传播就会把错误信息大范围扩散。这个问题比单句模型更严重，因为图结构会把局部错误变成全局污染。

第四个坑是证据分散但标签稀疏。现实数据常常只标关系，不一定完整标注所有证据句。结果是模型明明需要多跳推理，但训练监督却不够细，导致可解释性差，也难以调试。

下面把坑和规避策略压缩成表：

| 常见坑 | 具体表现 | 常用规避策略 |
|---|---|---|
| `NONE` 失衡 | 模型倾向全预测无关系 | 重采样、类别重加权、focal loss、pair pruning |
| 长文档过长 | 显存爆炸、训练慢 | sliding window、local encoder、句级压缩 |
| 共指链错误 | 错误实体聚合，图被污染 | 高质量共指前处理、置信度过滤、软链接 |
| 候选对过多 | 实体对组合爆炸 | 基于实体类型或距离做候选剪枝 |
| 证据分散 | 关系成立但难定位证据 | 联合预测证据句、多跳推理模块 |
| 数据域迁移 | 新闻有效，医学失效 | 领域继续预训练、领域词表、数据增强 |

新手很容易忽略一个事实：DocRE 不只是“模型更复杂”，更是“错误来源更多”。句子级任务主要担心局部语义理解，文档级任务还要额外处理实体归一、篇章结构、长程依赖、候选对爆炸。

真实工程例子是企业知识图谱构建。假设处理上市公司公告，实体包括公司名、高管名、子公司名、金额、项目名。关系证据可能分散在“标题 + 正文第一段 + 风险提示”三处。如果不做候选剪枝，实体对数量会非常大；如果共指把“本公司”和“子公司”混了，关系就会被系统性污染。此时通常要先做规则约束，例如只允许某些实体类型组合进入关系分类器，再叠加图模型。

---

## 替代方案与适用边界

不是所有关系抽取任务都该上文档级模型。建模能力越强，通常意味着标注、训练、部署成本越高，所以要按问题选方案。

下面直接比较三类思路：

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 句子级关系抽取 | 证据主要在单句内，文档短 | 实现简单、训练便宜、推理快 | 跨句关系召回差 |
| 文档级序列模型 | 证据跨句，但结构不复杂 | 端到端方便，少做图工程 | 长文成本高，显式结构弱 |
| 图式文档级推理 | 证据分散、共指明显、需多跳推理 | 结构表达强，适合复杂篇章关系 | 图构建复杂，依赖前处理质量 |

如果你的数据满足下面条件，句子级方法通常更合适：

- 关系证据大多集中在一个句子。
- 实体 mention 很少跨句指代。
- 数据量足够大，可以靠简单模型堆出效果。
- 系统对时延和资源敏感。

如果你的数据更接近下面情况，就该考虑文档级：

- 同一实体在文中反复被不同别名提及。
- 关系证据分散在多句。
- 需要给出证据链或提高召回。
- 领域文本长且结构化明显，如医学、金融、法律。

再具体一点：

- 对短新闻、商品评论、简短问答，句子级模型 often enough。
- 对 DocRED 这类跨句标注明显的数据，文档级模型是基本盘。
- 对生物医学文献、临床摘要、专利文档，图式推理通常比纯序列方法更稳，因为实体多、共指多、关系证据更碎。

有些替代路线也值得知道：

1. 基于检索的两阶段方法  
先检索疑似相关句，再做关系分类。优点是便宜；缺点是检索漏掉关键句就无法补救。

2. 基于大模型的生成式抽取  
直接让模型从整篇文档生成三元组。优点是上手快；缺点是可控性、稳定性、评测一致性通常不如专用判别模型。

3. 基于规则和模板的弱监督系统  
适合高精度、低召回的垂直场景，比如企业公告或特定医学模板文本。但迁移性差，维护成本高。

所以，文档级关系抽取并不是“替代所有方法”，而是在“关系证据天然跨句”这个边界内最合适。

---

## 参考资料

- DocRED 数据集与论文：文档级关系抽取的代表性基准，重点价值是把“跨句证据 + 实体级预测 + 证据句标注”放到了同一个任务定义里。
- DocRED 数据样例：适合理解 `vertexSet`、`labels`、`evidence` 这些字段怎样把 mention、实体和证据句联系起来。
- Cross-sentence reasoning graph 相关论文：核心贡献是把跨句推理显式做成图传播，而不是只靠序列编码器隐式学习。
- 生物医学文档级关系抽取论文：展示了化学物质、疾病、基因等实体在长文中如何通过依存图、GCN 或异构图建模。
- 强化 cross-evidence reasoning 的 PMC 方向工作：重点在“一个关系往往由多处证据联合支持”，适合理解为什么简单池化会丢失关键信息。
- GRACR 等图推理方法：适合进一步看“如何把全局文档结构、局部句法结构、实体交互结构合并到统一图中”。
- 文档级关系抽取综述或主题概览：适合建立全局图景，快速了解从句子级到文档级、从序列模型到图模型的演化脉络。

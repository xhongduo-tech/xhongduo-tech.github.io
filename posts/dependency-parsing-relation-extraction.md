## 核心结论

基于依存句法的关系抽取，本质上是在回答一个更窄但更可控的问题：给定一句话和两个实体，真正决定它们关系的信号，往往集中在两者的最短依存路径（Shortest Dependency Path, SDP，白话说就是“句法树里连接这两个实体的最短语法链”）附近，而不是分散在整句所有词上。

这个假设为什么有用？因为关系抽取最常见的问题不是“信息太少”，而是“噪声太多”。一句长句里会出现时间、地点、插入语、限定词、并列结构，它们对实体关系判断通常没有直接帮助。SDP 的作用就是把这些外围词先压掉，只保留更接近触发关系的语法骨架。

玩具例子最直接：

“阿司匹林缓解头痛。”

如果实体是“阿司匹林”和“头痛”，那么关键链通常就是：

`阿司匹林 -> 缓解 <- 头痛`

模型只要抓住“缓解”这个连接动作，就很容易判断这是“治疗/缓解”关系，而不需要把整句所有 token 都同等看待。

下面这张表可以先把核心差异看清楚：

| 方案 | 输入范围 | 噪声量 | 长句表现 | 主要风险 |
|---|---|---:|---|---|
| 整句序列模型 | 全句所有 token | 高 | 容易被修饰语干扰 | 关系触发词被淹没 |
| SDP 模型 | 实体间最短依存路径 | 低 | 通常更稳 | 依赖句法分析质量 |
| 路径+整句融合 | 路径骨架 + 全句上下文 | 中 | 往往最均衡 | 实现更复杂 |

结论可以压缩成一句话：依存句法关系抽取的价值不在于“理解整句全部语义”，而在于“用句法结构做降噪”，把模型注意力集中到最可能产生关系的那条路径上。

---

## 问题定义与边界

关系抽取是一个监督分类任务。输入是句子 $s$、主语实体 $e_s$、宾语实体 $e_o$，输出是一个关系标签 $r$。形式化写法可以记为：

$$
f(s, e_s, e_o) \rightarrow r
$$

这里的“关系”通常是预先定义好的离散类别，例如：

- 药物-疾病：`治疗`
- 药物-症状：`缓解`
- 公司-产品：`发布`
- 人物-组织：`任职于`

它不是开放式生成任务。系统通常不会自由写一段解释，而是从标签集合里选一个最可能的类别。

真实工程例子可以看电子病历：

“医生在门诊记录中建议患者使用阿司匹林以缓解头痛。”

目标不是找出句子所有重要信息，而是判断实体“阿司匹林”和“头痛”之间是否存在“治疗/缓解”关系。即使中间夹了“医生”“门诊记录”“建议患者使用”这些成分，SDP 仍可能把真正的关系骨架抽出来。

这个方法的边界也必须说清楚。它擅长的是“句法可见的关系”，也就是关系线索能在单句结构里被看见；它不擅长的是必须依赖常识、跨句推理或隐含背景知识的关系。

| 场景 | 是否适合 SDP | 原因 |
|---|---|---|
| 单句内显式关系 | 适合 | 关系触发词通常就在路径附近 |
| 长句、插入语多 | 适合 | 句法路径能有效降噪 |
| 语序较灵活文本 | 较适合 | 句法结构比线性位置更稳 |
| 跨句关系抽取 | 不适合 | 依存树通常只覆盖单句 |
| 强依赖常识的隐含关系 | 不适合 | 句法本身不提供世界知识 |
| OCR 错字、口语转写文本 | 风险高 | 上游句法分析不稳定 |

所以问题边界可以概括为两条：

1. 这是“句内、给定实体对、关系分类”的方法。
2. 它依赖 parser（句法分析器，白话说就是“把句子分析成语法树的程序”）是否把句子结构分析对。

---

## 核心机制与推导

先把句子做依存分析，得到一棵依存树：

$$
T = (V, E)
$$

其中 $V$ 是词节点集合，$E$ 是依存边集合。依存边表示“哪个词在语法上依附于哪个词”，例如主谓、动宾、定中等。

接着，对实体对 $(e_s, e_o)$ 找最短依存路径：

$$
P(e_s, e_o) = v_1, v_2, \dots, v_m
$$

这里的 $v_i$ 是路径上的节点。所谓最短，不是按词距算，而是按图上的边数算。因为依存树本质上是图结构，所以这一步可以理解成图最短路径搜索。

还是用玩具例子：

“阿司匹林缓解头痛”

如果“缓解”是中心谓词，那么依存结构常可简化为：

- `阿司匹林 -> 缓解`
- `头痛 -> 缓解`

于是最短路径就是：

`阿司匹林 -> 缓解 <- 头痛`

路径长度只有 2 条边。这个路径已经把关系抽取得很干净了：动作词“缓解”就是最核心的触发点。

路径上的每个节点并不只用词向量表示，还会拼接更多结构特征。常见写法是：

$$
x_i = [emb(w_i); emb(dep_i); emb(dir_i); emb(pos_i); emb(type_i)]
$$

各部分含义如下：

| 特征 | 含义 | 白话解释 |
|---|---|---|
| $emb(w_i)$ | 词嵌入 | 这个词本身是什么 |
| $emb(dep_i)$ | 依存类型嵌入 | 这个词和父节点是什么语法关系 |
| $emb(dir_i)$ | 方向嵌入 | 路径是向上走还是向下走 |
| $emb(pos_i)$ | 词性嵌入 | 这个词是动词、名词还是其他 |
| $emb(type_i)$ | 实体类型嵌入 | 它是不是药物、疾病、组织等 |

然后用一个编码器把整条路径编码成向量：

$$
h = Enc(x_1, x_2, \dots, x_m)
$$

这里的 `Enc` 可以有多种实现：

- `CNN`：卷积神经网络，适合抓局部模式
- `LSTM`：长短期记忆网络，适合顺序建模
- `Tree-LSTM`：树结构 LSTM，适合直接编码树
- `GNN`：图神经网络，适合更一般的图传播

不管编码器怎么换，最后都要做分类：

$$
p(r \mid s, e_s, e_o) = softmax(Wh + b)
$$

训练时常用交叉熵损失：

$$
L = -\log p(y \mid s, e_s, e_o)
$$

其中 $y$ 是真实标签。

把流程压成一条链就是：

`句子 -> 依存分析 -> SDP 提取 -> 特征构造 -> 编码器 -> 分类器`

这里最重要的推导逻辑不是数学本身，而是信息筛选逻辑：

1. 原始句子包含关系信号和大量噪声。
2. 依存树把线性句子改写成语法图。
3. 最短依存路径保留最可能连接两实体的骨架。
4. 编码器只对骨架建模，学习成本更低，干扰更少。

这也是它在长句中的优势来源。线性距离很远的两个实体，在依存树里未必远；相反，线性上靠得近的词，语法上可能并不直接相关。

---

## 代码实现

工程上不要把所有步骤揉进一个函数。最低限度应该拆成四步：

1. `parse_dependency(sentence)`：句法分析
2. `extract_sdp(tree, e1, e2)`：找最短依存路径
3. `build_features(path)`：构造路径特征
4. `classify_relation(features)`：输出关系类别

下面给一个可运行的极简 Python 玩具实现。它不依赖外部 parser，而是直接手写一棵小依存树，用来演示“如何从树里提取 SDP 并做一个最小分类”。

```python
from collections import deque
import math

def build_undirected_graph(edges):
    graph = {}
    for child, parent, dep in edges:
        graph.setdefault(child, []).append((parent, dep, "up"))
        graph.setdefault(parent, []).append((child, dep, "down"))
    return graph

def shortest_dependency_path(edges, source, target):
    graph = build_undirected_graph(edges)
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for nxt, _, _ in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None

def classify_relation_by_path(path):
    # 一个玩具规则：路径中出现“缓解”或“治疗”则判为 treat
    if any(token in {"缓解", "治疗"} for token in path):
        return "treat"
    return "none"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 手写依存边：(child, parent, dep)
# “阿司匹林 缓解 头痛”
edges = [
    ("阿司匹林", "缓解", "nsubj"),
    ("头痛", "缓解", "dobj"),
]

path = shortest_dependency_path(edges, "阿司匹林", "头痛")
label = classify_relation_by_path(path)

# 一个最小数值打分例子
h = 1.1
logit = 2 * h - 1
prob = sigmoid(logit)

assert path == ["阿司匹林", "缓解", "头痛"]
assert label == "treat"
assert round(prob, 2) == 0.77

print("path:", path)
print("label:", label)
print("prob:", round(prob, 2))
```

这段代码说明了三个关键点：

- 依存树可转成图来找路径。
- 实体关系判断可以只依赖路径上的词。
- 分类器本质上只是把路径表示映射到标签空间。

真实工程里，流程会更完整。以医疗文本为例：

“患者因长期偏头痛，在门诊被建议服用阿司匹林进行缓解，后续症状减轻。”

如果要抽“阿司匹林-偏头痛”的关系，常见实现会这样分层：

| 模块 | 输入 | 输出 | 工程职责 |
|---|---|---|---|
| 实体识别 | 原始句子 | 实体边界与类型 | 找出药物、疾病等实体 |
| 依存解析 | 分词结果 | 依存树 | 提供句法结构 |
| SDP 提取 | 树 + 实体对 | 路径节点与边 | 做结构降噪 |
| 特征编码 | 路径词、边、方向 | 向量表示 | 学习关系模式 |
| 分类器 | 向量表示 | 关系标签 | 输出最终类别 |

如果模型化一点，伪代码可以写成：

```python
def relation_extract(sentence, entity1, entity2):
    tree = parse_dependency(sentence)
    path = extract_sdp(tree, entity1, entity2)
    features = build_features(path)
    label = classify_relation(features)
    return label
```

这里最值得强调的不是“代码多复杂”，而是“数据结构是否清楚”。至少要保存：

- 原句
- 实体文本与起止位置
- 实体类型
- 依存边
- SDP 节点序列
- 边方向
- 依存类型
- 关系标签

否则一旦效果差，你根本无法定位是实体边界错、分词错、parser 错，还是分类器错。

---

## 工程权衡与常见坑

SDP 方法最容易被误解成“只要找出最短路径，后面自然会准”。实际不是。它的最大风险在上游，而不是分类器本身。

第一类坑是分词和实体边界错误。中文没有天然空格，分词一旦把“高血压患者”切坏，后续依存树的节点就已经不对，SDP 会从错误节点出发，整条路径都会偏。

第二类坑是 parser 偏差。parser 偏差就是“句法分析器对某个领域的句式理解有系统性误差”。新闻语料上训练的分析器，拿去跑病历、专利、客服对话，常常会失真。这个误差会直接传递给关系抽取。

第三类坑是只看 SDP 可能丢信息。最短路径是强降噪，但降噪过度就会丢上下文。比如否定词、时态、条件约束，有时不在路径内，却会改变关系含义。

看一个真实工程例子：

“患者在既往有胃病史的情况下服用阿司匹林后出现头痛加重。”

如果要判断“阿司匹林-头痛”是什么关系，难点在于这里不是“治疗头痛”，而更接近“不良反应/诱发加重”。但如果 parser 把“头痛加重”的结构挂错，SDP 可能只看到了“服用”和“头痛”，没抓住“加重”这个关键触发词，结果就会误判。

下面这张表是工程中最常见的坑点：

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 分词错误 | 实体节点错位，路径错误 | 领域词典、实体先验、联合建模 |
| parser 偏差 | 句法骨架失真 | 领域适配 parser，人工抽样评估 |
| 只看 SDP | 路径外触发词丢失 | 融合整句序列特征 |
| 并列结构复杂 | 最短路径不一定含真正触发词 | 增强路径，加入局部子树 |
| 跨分句关系 | 单句树无法覆盖 | 上升到文档级建模 |
| 低置信度解析 | 错误结构强行进入分类器 | 增加回退策略 |

工程上常用的补救思路有四个：

- 用领域数据微调 parser，而不是直接用通用模型。
- 把 SDP 特征和整句 Transformer 表示拼接，而不是二选一。
- 给实体做显式标记，避免模型搞不清路径两端是谁。
- 对低置信度样本回退到整句模型，避免“错误句法结构高权重输入”。

所以，SDP 的工程价值更像一个“高精度结构特征”，而不是一个能单独包打天下的总方案。

---

## 替代方案与适用边界

SDP 不是唯一解，也不应该被写成“默认最优解”。选型要看文本形态、解析质量和任务边界。

常见替代路线有四类：

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| 纯序列模型 | 实现简单，不依赖 parser | 长句噪声大，结构感弱 | 解析器不可用、文本较短 |
| SDP 模型 | 降噪强，结构解释性好 | 强依赖句法质量 | 单句显式关系、长句 |
| 整树/图模型 | 结构信息更完整 | 训练与实现更复杂 | 需要更多句法上下文 |
| 路径+序列融合 | 兼顾结构与上下文 | 成本更高 | 工程落地最常见 |

还可以按条件做更实用的选型：

| 条件 | 推荐方案 |
|---|---|
| 句子短、关系触发词明显 | 纯序列模型即可 |
| 句子长、修饰语多 | 优先考虑 SDP 或路径融合 |
| parser 在该领域质量高 | SDP 价值明显上升 |
| 文本噪声大、错别字多 | 少依赖句法，多依赖序列与预训练模型 |
| 需要跨句关系 | 用文档级模型，不要只靠 SDP |

社交媒体就是一个典型反例。口语、省略、错拼、断句混乱都很常见。此时依存树本身就不稳定。如果“树”搭歪了，只看树上的最短路径，错误反而会被进一步放大。这个场景里，整句 Transformer、指令微调模型，或者多特征融合方案通常比纯 SDP 更稳。

因此，SDP 更适合被理解成一个“结构偏置”。所谓结构偏置，就是“模型被人为引导，更关注某类结构信息”。当句法结构可信时，这个偏置能显著提升稳定性；当句法结构不可信时，它反而可能成为误差放大器。

---

## 参考资料

1. [A Shortest Path Dependency Kernel for Relation Extraction](https://aclanthology.org/H05-1091/)
2. [Classifying relations via long short term memory networks along shortest dependency paths](https://experts.illinois.edu/en/publications/classifying-relations-via-long-short-term-memory-networks-along-s/)
3. [Relation classification via modeling augmented dependency paths](https://experts.illinois.edu/en/publications/relation-classification-via-modeling-augmented-dependency-paths/)
4. [Integrating shortest dependency path and sentence sequence into a deep learning framework for relation extraction in clinical text](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0736-9)
5. [On the form of parsed sentences for relation extraction](https://www.sciencedirect.com/science/article/abs/pii/S0950705122005883)

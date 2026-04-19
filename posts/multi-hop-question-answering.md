## 核心结论

多跳问答推理是指：答案不能从单个句子直接读出，必须跨越两个或更多证据步骤，把中间事实串起来，才能得到最终答案。

它的核心不是“模型能不能猜出答案”，而是“模型能不能给出一条可验证的证据链”。证据链是指从问题出发，经过若干中间事实，最终导出答案的一组句子或文档片段。对多跳问答来说，答案正确但证据错误，不能算真正完成推理。

玩具例子：

问题：`A 的配偶出生在哪个城市？`

可用证据：

- 句子 1：`A married B.`
- 句子 2：`B was born in Shanghai.`

这里不能直接从 `A married B` 得到 `Shanghai`。第一步只能得到中间事实：`A` 的配偶是 `B`。第二步再查 `B` 的出生地，得到 `Shanghai`。完整路径是：

`A -> B -> Shanghai`

对新手来说，这就是“先找 A，再用 A 找 B，最后得到答案”。

单跳问答与多跳问答的差别如下：

| 对比项 | 单跳问答 | 多跳问答 |
|---|---|---|
| 输入 | 一个问题和一段或多段文本 | 一个问题和多个候选文档或句子 |
| 证据数量 | 通常 1 条证据就够 | 至少 2 条证据共同成立 |
| 是否需要中间事实 | 不需要或很弱 | 必须需要 |
| 是否可解释 | 可以解释，但不是核心要求 | 必须能输出推理路径或支持句 |
| 典型问题 | `B 出生在哪个城市？` | `A 的配偶出生在哪个城市？` |

HotpotQA 这类数据集把“答案”和“支持句子”一起作为监督信号。支持句子是指直接参与推理的证据句。它的设计目的不是只奖励模型答对，而是检查模型是否真的找到了正确路径。

---

## 问题定义与边界

多跳问答的形式可以写成：给定问题 $q$ 和文档集合 $D$，模型要输出答案 $a$，同时找到支持证据集合 $S$。如果答案必须依赖两个或更多证据组合才能推出，那么这个任务就是多跳问答。

更严格地说，模型不应只从单句或单段直接抽取答案，而应依赖多个事实之间的连接。例如：

- 单跳问题：`B 出生在哪座城市？`
- 多跳问题：`A 的配偶出生在哪座城市？`

第一类问题只要找到 `B was born in Shanghai.` 就能回答。第二类问题需要先识别 `A married B.`，再用 `B` 作为中间实体继续查找出生地。

不同问答任务的边界如下：

| 类型 | 白话解释 | 典型输入 | 推理要求 | 输出重点 |
|---|---|---|---|---|
| 单跳问答 | 在一条证据里找答案 | 问题 + 单段文本 | 低 | 答案文本 |
| 多跳问答 | 多条证据串起来回答 | 问题 + 多文档 | 高 | 答案 + 支持证据 |
| 开放域问答 | 先从大语料里找相关文档，再回答 | 问题 + 大规模语料库 | 取决于问题 | 检索质量 + 答案 |
| 多轮对话问答 | 当前问题依赖历史对话 | 多轮上下文 + 当前问题 | 主要依赖上下文消解 | 当前轮答案 |

这里容易混淆两个概念：开放域问答不一定是多跳问答。开放域问答强调文档不知道在哪里，需要先检索；多跳问答强调答案需要多个证据组合。一个问题既可以是开放域的，也可以是多跳的。例如在全公司知识库中问“某报错应该找哪个团队”，系统既要检索相关手册，又要跨越“故障码 -> 部件 -> 团队”的链路。

另一个边界是模板匹配。模板匹配是指系统看起来在推理，实际只是靠固定句式或关键词命中答案。例如问题总是长成“X 的配偶出生在哪”，语料里总是紧挨着出现配偶和出生地，那么模型可能没有学会推理，只是学会了模式。这种系统在分布变化后很容易失败。

---

## 核心机制与推导

多跳问答可以拆成四步：召回证据、构建图、图上传播、预测支持句和最终答案。

机制图如下：

`问题 -> 候选证据 -> 图传播 -> 支持句 -> 答案`

第一步是召回证据。召回是指从大量文档中找出可能相关的句子或段落。多跳任务对召回很敏感，因为只要某一跳证据没被找回来，后续推理就会断链。

第二步是构建图。图是由节点和边组成的数据结构。节点可以是句子、段落、实体或文档；边表示它们之间存在关系，例如共享实体、共指关系、超链接、同一文档相邻句子。对多跳问答来说，图的作用是把“可能需要一起使用的证据”连接起来。

第三步是图上传播。图神经网络是一类在图结构上更新节点表示的模型。白话说，它让一个节点吸收邻居节点的信息，从而把跨句、跨文档的线索合并起来。

设问题为 $q$，文档集合为 $D$，证据图为 $G=(V,E)$，节点 $v$ 对应句子 $s_v$。先把问题和句子编码成节点初始表示：

$$
h_v^{(0)} = Enc(q, s_v)
$$

其中 $Enc$ 是编码器，可以是 TF-IDF 特征、BiLSTM、BERT 或其他文本编码模型。$h_v^{(0)}$ 是节点在第 0 层的向量表示。

然后做消息传递。消息传递是指节点从相邻节点接收信息并更新自身表示：

$$
h_v^{(l+1)} =
\sigma\left(
W_{self}h_v^{(l)}
+
\sum_{u \in N(v)} \alpha_{uv} W_m h_u^{(l)}
\right)
$$

其中 $N(v)$ 表示节点 $v$ 的邻居，$\alpha_{uv}$ 是从邻居 $u$ 传到 $v$ 的权重，$W_{self}$ 和 $W_m$ 是可学习参数，$\sigma$ 是非线性函数。直观上，如果句子 1 提到 `A married B`，句子 2 提到 `B was born in Shanghai`，共享实体 `B` 会让两条证据在图上互相影响。

接着预测支持句：

$$
p_{sup}(v) = softmax(w_{sup}^{T}h_v^{(L)})
$$

这里 $p_{sup}(v)$ 表示节点 $v$ 是支持句的概率。最后把上下文表示 $h_{ctx}$ 输入答案预测层：

$$
p(ans \mid q,D) = softmax(W_{ans}h_{ctx})
$$

如果使用链式推理，也可以把中间步骤写成显式变量 $z$：

$$
p(a,z \mid q,D)=\prod_t p(z_t \mid q,D,z_{<t}) \cdot p(a \mid q,D,z)
$$

其中 $z_t$ 是第 $t$ 个中间推理步骤。思维链提示让大模型生成这些中间步骤，但要注意：思维链文本不等于真实证据链。真实证据链必须能回到原始文档中被验证。

多跳主流程可以写成：

```python
# 伪代码：多跳问答主流程
q = input_question()
candidates = retrieve_sentences(q, docs)           # 召回候选句
graph = build_evidence_graph(candidates)           # 构建证据图
node_repr = encode_question_sentence(q, graph)      # 初始化节点表示

for _ in range(L):
    node_repr = message_passing(graph, node_repr)   # 图上传播

support_scores = predict_support(node_repr)         # 支持句打分
answer = decode_answer(q, graph, node_repr)         # 输出答案
return answer, top_support_sentences(support_scores)
```

玩具数值例子中，句子 1 的置信度是 $0.9$，句子 2 的置信度是 $0.8$。如果把两跳路径分数粗略看成乘积，那么：

$$
score(A \rightarrow B \rightarrow Shanghai)=0.9 \times 0.8 = 0.72
$$

这个分数不是严格概率建模，只是说明路径质量会同时受每一跳证据影响。

---

## 代码实现

实现多跳问答时，不要一开始就追求复杂模型。先把“检索、证据打分、支持句输出、答案生成”这条流水线跑通。工程上最重要的是保留中间结果，包括候选句、句子分数、边关系、支持句预测和最终答案。这样系统出错时才能定位是召回错、建图错、推理错，还是答案解码错。

最小实现可以分成 5 个模块：

| 模块 | 输入 | 输出 | 核心逻辑 | 常见失败点 |
|---|---|---|---|---|
| 文本切分 | 原始文档 | 句子列表 | 按句号、换行或解析器切分 | 切分过粗或过碎 |
| 候选召回 | 问题 + 句子 | 候选句 | 关键词、实体、向量检索 | 关键证据没召回 |
| 图构建 | 候选句 | 证据图 | 按共享实体或相邻关系连边 | 图过密或漏边 |
| 图推理 | 证据图 | 节点分数 | 沿边聚合证据分数 | 假链得分过高 |
| 答案解码 | 支持句 | 答案 | 从最终证据中抽取或生成 | 答案和证据不一致 |

下面是一段可运行的 Python 玩具实现。它不是神经网络，只用字符串规则模拟“检索 -> 建图 -> 推理 -> 输出答案和支持句”的过程，目的是让数据结构和流程清楚。

```python
import re
from collections import defaultdict

docs = [
    "A married B.",
    "B was born in Shanghai.",
    "C was born in Beijing.",
    "A works at ExampleCorp."
]

question = "Where was A's spouse born?"

def tokenize(text):
    return set(re.findall(r"[A-Za-z]+", text.lower()))

def retrieve_sentences(q, docs):
    q_tokens = tokenize(q)
    scored = []
    for i, sent in enumerate(docs):
        score = len(q_tokens & tokenize(sent))
        # 对玩具例子补一个简单规则：spouse 与 married 相关
        if "spouse" in q.lower() and "married" in sent.lower():
            score += 2
        scored.append({"id": i, "text": sent, "score": score})
    return [x for x in scored if x["score"] > 0]

def extract_entities(sent):
    return set(re.findall(r"\b[A-Z][A-Za-z]*\b", sent))

def build_evidence_graph(candidates, docs):
    nodes = {c["id"]: c for c in candidates}

    # 多跳场景下，第一跳可能引入新实体，因此把共享实体相关句也补进图里
    known_entities = set()
    for c in candidates:
        known_entities |= extract_entities(c["text"])

    for i, sent in enumerate(docs):
        if i not in nodes and (extract_entities(sent) & known_entities):
            nodes[i] = {"id": i, "text": sent, "score": 1}

    edges = defaultdict(set)
    ids = list(nodes)
    for i in ids:
        for j in ids:
            if i >= j:
                continue
            if extract_entities(nodes[i]["text"]) & extract_entities(nodes[j]["text"]):
                edges[i].add(j)
                edges[j].add(i)
    return nodes, edges

def reason(nodes, edges):
    node_scores = {i: nodes[i]["score"] for i in nodes}
    for i in nodes:
        for j in edges[i]:
            node_scores[j] += 0.5 * nodes[i]["score"]
    return node_scores

def decode_answer(nodes, node_scores):
    support_ids = sorted(node_scores, key=node_scores.get, reverse=True)[:2]
    support_texts = [nodes[i]["text"] for i in support_ids]

    for text in support_texts:
        m = re.search(r"born in ([A-Za-z]+)", text)
        if m:
            return m.group(1), support_texts
    return None, support_texts

candidates = retrieve_sentences(question, docs)
nodes, edges = build_evidence_graph(candidates, docs)
node_scores = reason(nodes, edges)
answer, supports = decode_answer(nodes, node_scores)

assert answer == "Shanghai"
assert "A married B." in supports
assert "B was born in Shanghai." in supports

print(answer)
print(supports)
```

真实工程例子可以放在企业知识库问答中。用户问：“某型号设备出现 E42 报错后，应该联系哪个团队处理？”系统需要先在设备手册中找到 `E42 -> power module`，再在组织责任表中找到 `power module -> hardware platform team`。最终回答不能只说“联系硬件平台团队”，还应输出两条证据：故障码对应的部件，以及部件对应的责任团队。

---

## 工程权衡与常见坑

多跳问答最大的风险通常不是“模型不会推理”，而是“证据没召回到”或“图边连错了”。前者会断链，后者会制造假链。

如果问题是 `A 的配偶出生在哪个城市？`，检索阶段只找到了 `A married B.`，没有找到 `B was born in Shanghai.`，后面的模型再强也无法可靠回答。如果图构建阶段把 `B was born in Shanghai.` 错连到 `C married D.`，模型可能沿着错误路径得到看似合理的答案。

常见问题如下：

| 问题类型 | 风险 | 规避手段 |
|---|---|---|
| 召回不足 | 关键证据缺失，推理链断掉 | 多轮检索、实体扩展、召回 Top-K 加大 |
| 图过密 | 无关句子互相传播，假链得分升高 | 限制边类型、加入边权、过滤低置信边 |
| 证据漂移 | 从正确实体跳到相似但错误的实体 | 实体消歧、共指解析、路径一致性检查 |
| 事后编造 | 模型生成合理解释，但证据不存在 | 强制引用原文句子，答案与支持句双验证 |

评估时只看 EM/F1 不够。EM 是 exact match，意思是预测答案与标准答案完全一致；F1 衡量预测词和标准词的重叠程度。它们主要评价答案文本，不足以评价推理过程。

例如模型答对了 `Shanghai`，但支持句不是 `B was born in Shanghai.`，而是某条无关句子里的 `Shanghai`，那它可能只是猜对或被语料偏差误导。对多跳问答来说，还要看：

| 指标 | 评价对象 | 意义 |
|---|---|---|
| Answer EM/F1 | 最终答案 | 答案是否对 |
| Supporting Fact F1 | 支持句 | 证据是否对 |
| Joint F1 | 答案 + 证据 | 两者是否同时正确 |
| Path Quality | 推理路径 | 中间步骤是否连贯 |
| Faithfulness | 忠实性 | 答案是否能由证据推出 |

工程上应把“答案生成”和“证据选择”分开记录。日志里至少要有候选证据、边关系、支持句分数、最终答案和答案来源。否则上线后很难解释错误。

---

## 替代方案与适用边界

不是所有问答系统都需要多跳推理。当任务只需要简单事实抽取时，单跳检索加抽取式阅读理解通常更便宜、更稳定。抽取式阅读理解是指答案直接来自原文片段，模型只需要定位答案跨度。

当问题链路很长、证据跨多个文档、或业务要求强审计性时，多跳方案更合适，因为它能显式输出中间证据。

方案选择表如下：

| 方案 | 适用条件 | 优点 | 局限 |
|---|---|---|---|
| 单跳检索 + 抽取 | 答案在单段文本中直接出现 | 简单、稳定、成本低 | 不能处理复杂证据链 |
| 多跳图推理 | 需要跨实体、跨文档组合证据 | 可解释，能输出路径 | 建图和标注成本高 |
| 链式思维提示 | 大模型需要展示中间步骤 | 实现快，适合原型验证 | 推理文本可能不忠实于证据 |
| 检索增强生成 | 需要结合外部知识生成答案 | 覆盖面广，交互自然 | 容易生成证据外内容 |

企业知识库中可以这样选择：如果用户问“这个设备的额定功率是多少”，通常单跳检索就够，因为答案可能直接出现在规格表里。如果用户问“这个报错该找谁”，则更适合多跳链路：`故障码 -> 部件 -> 团队`。这时系统不仅要回答团队名称，还要展示故障码对应哪个部件、该部件归哪个团队负责。

链式思维提示适合做快速原型。例如让大模型按步骤分析问题，先找实体关系，再找目标属性。但在严肃工程场景中，链式思维不能替代证据链。正确做法是让模型的每一步都绑定可检索、可引用的原文证据。

检索增强生成也常用于问答系统。它先检索相关文档，再让生成模型回答。若只要求自然语言回答，RAG 已经够用；若要求“每一步为什么成立”，就需要增加支持句预测、路径验证或图推理模块。

---

## 参考资料

1. [HotpotQA 官方主页](https://hotpotqa.github.io/)
2. [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/)
3. [Hierarchical Graph Network for Multi-hop Question Answering](https://aclanthology.org/2020.emnlp-main.710/)
4. [Question Answering by Reasoning Across Documents with Graph Convolutional Networks](https://aclanthology.org/N19-1240/)
5. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

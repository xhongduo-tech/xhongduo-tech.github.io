## 核心结论

联合抽取的定义是：在一个端到端模型里，同时预测实体边界和关系三元组，而不是先做实体识别再做关系分类。白话说，模型不是先把“人名、公司名”找出来再二次判断关系，而是一次性决定“谁”和“谁之间是什么关系”。

传统 Pipeline 方案通常拆成两步：命名实体识别（NER，先找实体）和关系抽取（RE，再判断关系）。它的主要问题不是“不能用”，而是错误会级联。第一步漏了实体，第二步就没有输入；第一步把边界切错，第二步就会在错误对象上继续推理。这个现象可以直接写成：

$$
P(\text{triple correct}) \approx P(\text{entity correct}) \times P(\text{relation correct}\mid \text{entity correct})
$$

因此，只要实体阶段出错，后续三元组正确率就会被连带压低。

联合抽取的价值在于把“实体是什么”和“实体之间有什么关系”放到同一上下文里建模。这样做有两个直接收益：

1. 降低误差传播。
2. 更容易处理重叠关系和共享实体。

一个玩具例子就足够说明差别。句子：

“李雷在百度工作，也负责搜索团队。”

如果 Pipeline 的 NER 漏掉“搜索团队”，那“负责”关系永远不可能被抽出来。联合模型则会同时看“李雷”“百度”“搜索团队”以及它们之间的连线，漏报概率更低。

下表先看总体差别：

| 模型 | 错误传播 | 重叠支持 | 主要优点 | 主要短板 |
| --- | --- | --- | --- | --- |
| Pipeline | 强，NER 错误直接传给 RE | 通常较弱 | 易调试、模块清晰 | 召回受前置阶段限制 |
| CasRel | 弱于 Pipeline | 支持共享 subject 的重叠关系 | 解码直观，适合三元组生成 | 逐 subject 解码，阈值敏感 |
| TPLinker | 基本避免级联 | 强 | 实体和关系统一标注，并行预测 | $O(n^2)$ 开销明显 |
| SPN | 基本避免级联 | 强 | 直接预测无序三元组集合 | 训练和匹配实现更复杂 |

在公开基准上，联合建模的收益不是概念上的。以 NYT 数据集为例，TPLinker 报告的 F1 可达 91.3% 左右，相比常见 Pipeline 基线高约 4.2 个点。如果拿 CasRel 89.6 和 TPLinker 91.9 的一组常见对比数字来写，增量就是：

$$
\Delta F1 = 91.9 - 89.6 = 2.3
$$

这 2.3 个点不是“看起来不大”的装饰指标。对于每 100 个应抽取的三元组，往往就意味着少漏掉约 2 到 3 个结果。

---

## 问题定义与边界

关系抽取任务的标准输出通常写成三元组集合：

$$
T = \{(s, r, o)\}
$$

其中 $s$ 是 subject，主语实体；$o$ 是 object，宾语实体；$r$ 是 relation，关系类型。对白话解释就是：从一句话里抽出“谁-和-谁-是什么关系”。

但工程上真正要预测的不是抽象符号，而是文本中的 span。span 指“从第几个 token 到第几个 token 的连续片段”。所以更完整的映射应写成：

$$
s \leftrightarrow (i_s, j_s), \quad o \leftrightarrow (i_o, j_o)
$$

也就是每个实体都要对应一句话里的起止位置。于是任务变成：

1. 找出所有实体 span。
2. 判断哪些 span 对之间存在关系。
3. 给每对关系分配类型。

这也是联合抽取和普通分类问题的本质区别：输出不是一个标签，而是一组结构化对象。

新手最容易忽略的边界有两个。

第一，重叠实体。比如“北京大学研究员”里，“北京大学”是机构，“研究员”是职位，整段还可能被某些任务定义成岗位描述。不同任务的标注边界并不总是互斥。

第二，共享实体关系。比如：

“李雷在百度工作，主管搜索，也参与知识图谱平台建设。”

这里“李雷”同时和多个宾语、多个关系相连，形成一个 subject 对多个 triple 的结构。联合模型天然更适合这种图状输出。

可以用表格把输入输出结构看清楚：

| 方案 | 输入 | 中间结果 | 输出 |
| --- | --- | --- | --- |
| Pipeline 两阶段 | 原始句子 | 先实体列表，再实体对候选集 | 三元组集合 |
| 联合模型 | 原始句子 | 编码器共享语义表示 | 实体 span + 三元组同时预测 |

继续看一个更具体的句子：

“李雷在百度工作，主管搜索。”

合理输出至少可以包含：

- 实体：“李雷”“百度”“搜索”
- 关系：$(李雷,\ 就职于,\ 百度)$
- 关系：$(李雷,\ 管理/负责,\ 搜索)$

如果你先做 NER，再把实体对喂给 RE，那么任何一个实体边界错位都可能使候选实体对缺失。联合模型则倾向于在同一个句子表示里同时学习“李雷”是人、“百度”是组织、“就职于”连接两者。

因此，这类方法的适用边界很明确：当任务输出是结构化关系集合，而不是单一标签时，联合建模通常比串联模块更自然。

---

## 核心机制与推导

联合抽取主流路线可以分成三类：级联指针、token-pair linking、集合预测。对应代表方法就是 CasRel、TPLinker、SPN。

先看 CasRel。它的核心思想不是“先抽实体再抽关系”，而是“把关系看成 subject 到 object 的映射函数”。白话说，模型先定位主语，再针对每种关系去找它对应的宾语。

设编码器输出 token 表示为 $H=\{h_1,\dots,h_n\}$。CasRel 常见做法是先用两个二分类器预测 subject 的起点和终点：

$$
p_i^{subj,start} = \sigma(W_s^{start} h_i + b_s^{start})
$$

$$
p_i^{subj,end} = \sigma(W_s^{end} h_i + b_s^{end})
$$

subject loss 可写成二元交叉熵之和：

$$
\mathcal{L}_{subj} = \mathcal{L}_{BCE}(y^{subj,start}, p^{subj,start}) + \mathcal{L}_{BCE}(y^{subj,end}, p^{subj,end})
$$

拿到某个 subject 后，再把 subject 信息融合回上下文，对每种关系 $r$ 预测 object 的起止位置：

$$
p_{i,r}^{obj,start} = \sigma(W_{r}^{start}\tilde{h}_i + b_{r}^{start})
$$

$$
p_{i,r}^{obj,end} = \sigma(W_{r}^{end}\tilde{h}_i + b_{r}^{end})
$$

于是关系部分损失是：

$$
\mathcal{L}_{rel} = \sum_{r \in R}
\left[
\mathcal{L}_{BCE}(y_r^{obj,start}, p_r^{obj,start}) +
\mathcal{L}_{BCE}(y_r^{obj,end}, p_r^{obj,end})
\right]
$$

总损失通常写成：

$$
\mathcal{L} = \mathcal{L}_{subj} + \mathcal{L}_{rel}
$$

CasRel 的优点是结构直观，尤其适合“一个 subject 连多个 object”的场景。它的问题也同样明显：解码是级联的，subject 阶段一旦漏掉，后续相关 triple 全部消失。

再看 TPLinker。它把问题改写成 token pair linking，也就是“任意两个 token 之间是否存在某种连接”。白话说，不再先找实体再连关系，而是把句子上方所有 token 两两组合，直接给这些 token 对打标签。

对长度为 $n$ 的句子，只考虑上三角区域 $(i,j), i \le j$，形成所谓 handshake 空间。对每个关系类型 $r$，定义标签：

$$
H_{i,j,r} \in \{0,1,\dots,K\}
$$

这里的 $H_{i,j,r}$ 可以表示实体边界连接、head-to-head 关系连接、tail-to-tail 关系连接等不同链接类型。你可以把它理解成一个“握手矩阵”：如果两个位置需要在某种结构上相连，就给它打对应标签。

它的好处是并行。重叠三元组不需要像序列解码那样排顺序，只要多个 token pair 同时被激活即可。这就是 TPLinker 处理 overlapping triples 更稳定的根本原因。

一个简化玩具例子如下。句子 token 序列为：

`[李雷, 在, 百度, 工作, 主管, 搜索]`

如果“李雷”和“百度”之间存在“就职于”，那么主语头、宾语头，主语尾、宾语尾等 token 对会在对应关系槽位上被置为正例。再加上实体起止链接，最终可以从矩阵中反推出完整三元组。

SPN 的思路又不同。SPN 是 set prediction，也就是集合预测。白话说，模型不再一个个顺序生成三元组，而是一次性输出若干“候选槽位”，然后通过二分图匹配把预测集合和真值集合对应起来。

核心机制是 bipartite matching。设模型输出 $m$ 个预测三元组槽位 $\hat{Y}=\{\hat{y}_1,\dots,\hat{y}_m\}$，真值集合为 $Y=\{y_1,\dots,y_k\}$，则训练时寻找一个最优匹配 $\pi$：

$$
\pi^* = \arg\min_{\pi} \sum_{i=1}^{k} \mathcal{C}(y_i, \hat{y}_{\pi(i)})
$$

其中 $\mathcal{C}$ 是匹配代价，通常综合关系类别、实体边界等误差。这样做的意义是：输出集合不依赖顺序，因此天然适合多三元组、重叠三元组场景。

如果把三类方法放在一条线上看：

- CasRel：先定主语，再找宾语，结构最接近“级联抽取”。
- TPLinker：把实体和关系统一成 token-pair 标注，偏并行。
- SPN：把输出当集合，训练时靠匹配保证一一对应，最接近“检测式”建模。

---

## 代码实现

先给一个可运行的玩具实现。它不是真正的神经网络，只是把 TPLinker 的“握手空间”思想压缩成一个可验证的数据结构，帮助理解标签构造。

```python
from typing import List, Tuple, Dict

Triple = Tuple[Tuple[int, int], str, Tuple[int, int]]

def handshake_index(n: int):
    pairs = []
    pos2idx = {}
    idx = 0
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j))
            pos2idx[(i, j)] = idx
            idx += 1
    return pairs, pos2idx

def build_tplinker_labels(tokens: List[str], triples: List[Triple], relations: List[str]) -> Dict[str, List[int]]:
    n = len(tokens)
    _, pos2idx = handshake_index(n)
    size = n * (n + 1) // 2

    labels = {}
    for r in relations:
        labels[f"ent::{r}"] = [0] * size
        labels[f"rel_head::{r}"] = [0] * size
        labels[f"rel_tail::{r}"] = [0] * size

    for (s_start, s_end), rel, (o_start, o_end) in triples:
        labels[f"ent::{rel}"][pos2idx[(s_start, s_end)]] = 1
        labels[f"ent::{rel}"][pos2idx[(o_start, o_end)]] = 1

        h_pair = (min(s_start, o_start), max(s_start, o_start))
        t_pair = (min(s_end, o_end), max(s_end, o_end))
        labels[f"rel_head::{rel}"][pos2idx[h_pair]] = 1
        labels[f"rel_tail::{rel}"][pos2idx[t_pair]] = 1

    return labels

tokens = ["李雷", "在", "百度", "工作", "主管", "搜索"]
triples = [((0, 0), "works_for", (2, 2)), ((0, 0), "manages", (5, 5))]
relations = ["works_for", "manages"]

labels = build_tplinker_labels(tokens, triples, relations)

assert sum(labels["rel_head::works_for"]) == 1
assert sum(labels["rel_tail::works_for"]) == 1
assert sum(labels["rel_head::manages"]) == 1
assert sum(labels["rel_tail::manages"]) == 1
assert len(labels["ent::works_for"]) == len(tokens) * (len(tokens) + 1) // 2
```

这个例子说明三件事：

1. handshake 空间大小是 $\frac{n(n+1)}{2}$，不是完整 $n^2$，因为只取上三角。
2. 每个关系都需要一套链接标签。
3. 解码时只要从这些标签反推 head/tail 和实体边界，就能还原三元组。

如果写成 CasRel 风格的伪代码，结构大致如下：

```python
h = encoder(tokens)
subj_start = sigmoid(W1 @ h)
subj_end = sigmoid(W2 @ h)

subjects = decode_spans(subj_start, subj_end, threshold=0.5)

triples = []
for subj in subjects:
    h_cond = condition_on_subject(h, subj)
    for r in relations:
        obj_start = sigmoid(W_obj_start[r] @ h_cond)
        obj_end = sigmoid(W_obj_end[r] @ h_cond)
        objects = decode_spans(obj_start, obj_end, threshold=0.5)
        for obj in objects:
            triples.append((subj, r, obj))
```

如果写成 TPLinker 风格，则更接近：

```python
h = encoder(tokens)
for i in range(n):
    for j in range(i, n):
        pair_repr = handshake(h[i], h[j])
        for r in relations:
            predict_link(pair_repr, r)
```

训练阶段的损失结构可以归纳为：

| 方法 | 编码器后接 head | 训练损失 |
| --- | --- | --- |
| CasRel | subject start/end + relation-specific object pointer | $\mathcal{L}_{subj} + \mathcal{L}_{obj}$ |
| TPLinker | handshake token-pair classifier | handshake BCE / multi-label loss |
| SPN | 非自回归 triple decoder | matching loss + 分类/边界损失 |

真实工程例子比玩具句子更能暴露问题。比如轨道维护日志：

“X3 道岔绝缘接头松动，导致 5G 轨道电路红光带异常，工区已更换接头并复测。”

这里可能同时存在：

- 实体：`X3道岔`、`绝缘接头`、`5G轨道电路`、`红光带异常`
- 关系：部件归属、故障导致、处理措施、复测对象

若用 Pipeline，先做实体识别时很容易把“红光带异常”切成“红光带”和“异常”，再导致后续关系候选集错乱。TPLinker 这类联合标注方式在这类短句压缩、多关系并列的场景里通常更稳，因为它不是依赖一个独立的实体结果作为唯一入口。

---

## 工程权衡与常见坑

联合抽取不是“效果更高所以无脑替换”。它的代价主要来自表示空间、解码复杂度和调参成本。

先看复杂度。TPLinker 的核心问题是 token-pair 数量随长度平方增长。如果句长为 $n$，关系数为 $R$，则相关显存和计算量近似满足：

$$
\text{Memory} \propto n^2 \times R
$$

即使只取上三角，量级也仍然是平方级。句长从 128 增加到 256，token pair 数量会接近变成 4 倍。这就是为什么长文本场景里 TPLinker 经常需要：

- 限制 `max_length`
- 滑动窗口切分
- 只保留局部候选
- 用稀疏标签或负采样压缩训练

CasRel 的主要问题则不是平方复杂度，而是阈值联动。subject start、subject end、object start、object end 往往都有阈值，稍微调高一点，召回会明显下降。这个现象可以粗略写成：

$$
Recall \approx P(p_{subj}>\tau_s) \times P(p_{obj}>\tau_o \mid subj)
$$

其中 $\tau_s,\tau_o$ 分别是 subject 和 object 阈值。只要阈值过高，尤其在长尾实体上，漏判会被成倍放大。

常见工程坑可以直接列出来：

| 方法 | 长文本 | 训练速度 | 调优成本 | 重叠关系覆盖 |
| --- | --- | --- | --- | --- |
| Pipeline | 相对友好 | 较快 | 低到中 | 一般 |
| CasRel | 中等 | 中等 | 中到高 | 好 |
| TPLinker | 敏感，需控长 | 较慢 | 中 | 很好 |
| SPN | 中等 | 中到慢 | 高 | 很好 |

新手最容易踩的几个具体坑：

1. 标注不一致。  
实体边界如果在训练集中前后不统一，联合模型不会替你“纠错”，反而会同时污染实体和关系学习。

2. 负样本过多。  
TPLinker 的 handshake 空间大多数位置都是否定标签，若不做重加权或采样，模型会倾向于“全预测为无关系”。

3. 关系方向混乱。  
`works_for` 和 `employs` 在语义上接近，但主宾方向不同。如果标签设计没把方向编码清楚，解码后会出现反向三元组。

4. 长句跨窗口丢失。  
把长文本切块时，跨块关系会直接消失。比如主语在窗口 1，宾语在窗口 2，这条关系在局部模型里不可见。

5. 评测口径不统一。  
有的论文按严格 span 匹配，有的对实体边界更宽松；有的按三元组完全匹配算 F1。没有统一评测口径时，数字不能直接横比。

---

## 替代方案与适用边界

如果任务很小、关系种类很少、标注也很规范，Pipeline 仍然是合理选择。白话说，不是所有项目都值得为了多几个点的 F1 把系统复杂度翻倍。

比如一个小型产品知识库，只需要抽取：

- 产品名
- 品牌
- 价格
- “属于品牌”“售价为”两种关系

这种场景下，先用现成 NER 找实体，再写一个轻量关系分类器，通常已经够用。因为实体形态固定、关系很少、重叠关系也少，Pipeline 的短板不会被放大。

Hybrid 则是很多工程系统更现实的形态：主干用联合模型，后处理再做规则校验或阈值过滤。比如：

- 联合模型负责生成候选三元组
- 若实体置信度低于阈值则丢弃
- 若关系违反 schema 约束则回退
- 若宾语类型与关系定义不一致则重打分

这样做的原因很现实：联合模型提升召回，但线上系统往往更怕脏数据，因此还需要后验约束来保精度。

三类方案可以这样比较：

| 方案 | 优点 | 适用场景 | 典型瓶颈 |
| --- | --- | --- | --- |
| Pipeline | 模块清楚、开发快、便于定位问题 | 小数据、少关系、低重叠场景 | error cascade，召回受 NER 限制 |
| Joint | 统一建模、重叠关系支持强、整体 F1 通常更高 | 知识图谱构建、复杂日志、新闻事实抽取 | 实现复杂，训练开销高 |
| Hybrid | 兼顾召回与精度，容易接入业务规则 | 线上生产系统 | 系统链路更长，维护成本高 |

可以用一个极简流程图理解 error cascade：

`Pipeline: 文本 -> NER -> 候选实体对 -> RE -> 三元组`  
`Joint: 文本 -> 共享编码 -> 实体与关系联合解码 -> 三元组`

前者的问题在于第二步必须依赖第一步结果；后者则在同一表示空间里直接优化最终结构输出。

所以选择标准并不神秘：

- 数据简单、工程时间紧：优先 Pipeline。
- 重叠关系多、共享实体多、追求整体抽取质量：优先 Joint。
- 线上精度要求高、允许增加规则层：考虑 Hybrid。

---

## 参考资料

1. CasRel: Relational Triple Extraction via Cascade Binary Tagging  
核心贡献：把关系建模成从 subject 到 object 的函数映射，用级联指针解决重叠三元组问题；也系统说明了 Pipeline 的误差传播问题。  
链接：https://arxiv.org/abs/1909.03227

2. TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking  
核心贡献：提出 token pair linking，把实体和关系统一到握手标注空间中，单阶段并行预测；在 NYT 等数据集上取得强结果。  
链接：https://arxiv.org/abs/2010.13415

3. Set Prediction Networks for End-to-End Entity Relation Extraction  
核心贡献：把关系抽取改写为无序集合预测，用 bipartite matching 训练，避免顺序解码对重叠三元组的不稳定性。  
链接：https://aclanthology.org/2021.emnlp-main.94/

4. 轨道维护日志场景中的联合抽取研究综述/实验论文  
核心贡献：说明在工程日志、设备维护文本这类结构松散、关系重叠明显的领域，TPLinker 一类模型相比传统 Pipeline 更稳定。  
链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11598563/

5. 联合抽取方法的工程化分析论文  
核心贡献：总结 CasRel、TPLinker、SPN 等方法在长文本、重叠关系、显存复杂度上的差异，为模型选型提供经验依据。  
链接：https://journals.sagepub.com/doi/full/10.1177/14604582241274762

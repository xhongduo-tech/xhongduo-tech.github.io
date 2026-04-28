## 核心结论

远程监督的关键问题不是“没有标签”，而是“标签太吵”。远程监督的白话解释是：把知识库里的关系自动投到文本上，省掉人工标注，但会把很多并不表达该关系的句子也标成正例。最常见的错误假设是：只要知识库里存在三元组 $(h,r,t)$，所有同时提到 $h,t$ 的句子都表达 $r$。这在真实语料里通常不成立。

多实例学习，白话解释是“把多个句子打包后整体判断”，是远程监督降噪的基本修正。它不再要求每句都正确，而只要求同一实体对形成的 `bag` 中，至少有一句真在表达该关系：
$$
B_i=\{x_{i1},x_{i2},...,x_{in}\}, \quad y_i=1 \iff \exists j,\ y_{ij}=1
$$
这条假设把“句子级硬标签”改成了“bag 级弱标签”，是后续 selective attention、prototype、contrastive learning、curriculum learning 的共同起点。

可以把远程监督噪声处理分成三层：

| 层级 | 解决的问题 | 常见方法 | 适合场景 |
|---|---|---|---|
| `sentence` 层 | 哪一句更像证据句 | attention、selective attention、hard/soft selection | 同一实体对有多句共现 |
| `bag` 层 | 多句如何合成一个训练样本 | MIL、bag encoder、relation-aware aggregation | 句子数不均、证据强弱不同 |
| `sample` 层 | 哪些样本整体不可信 | sample reweighting、prototype、contrastive、curriculum | 噪声极重、类别混淆明显 |

玩具例子最容易说明问题。知识库里有 `("Bill Gates", "Microsoft", founder)`，文本中有三句：

1. Bill Gates founded Microsoft in 1975.
2. Bill Gates donated to programs once supported by Microsoft employees.
3. Bill Gates discussed Microsoft’s AI strategy in an interview.

远程监督会把三句都标成 `founder`。但真正表达关系的只有第 1 句。降噪不是把数据全删掉，而是让模型在训练时“更相信第 1 句，少相信第 2、3 句”。

---

## 问题定义与边界

远程监督，白话解释是“用知识库自动给文本打标签”，本质是弱监督，不是干净标注。它的噪声主要来自“实体共现不等于关系成立”。如果文本里同时出现两个实体，只能说明它们被提到了，不能直接推出当前句子在表达知识库里的那条关系。

但并不是所有错误都叫“远程监督噪声”。工程上至少要分清四类现象：

| 现象 | 表现形式 | 是否属于噪声处理问题 | 典型应对方式 |
|---|---|---|---|
| 共现误标 | 句子提到实体对，但没表达目标关系 | 是 | MIL、attention、样本降权 |
| 跨句证据 | 关系分散在多句甚至整段中 | 不完全是 | 文档级建模、跨句推理 |
| KB 缺失 | 文本表达了关系，但知识库没有这条边 | 不是传统正例降噪 | PU learning、开放世界假设 |
| 长尾关系稀疏 | 某些关系样本极少且表达弱 | 部分相关 | 重采样、prototype、few-shot 技术 |

这里的边界很重要。比如一句话里有 `Apple` 和 `Steve Jobs`，但句子是在讲公司历史、访谈、产品发布会、股权事件，不能都算 `founder_of`。这是典型噪声。  
但如果一篇文章前一句介绍 `Steve Jobs co-founded Apple`，后一句只说 `Apple later expanded globally`，第二句本身不表达关系，这不是误标模型能单独解决的句法问题，而是证据范围定义的问题。

真实工程例子常见于新闻知识图谱。企业舆情系统想抽取“高管任职”“公司收购”“合作签约”三类关系。知识库里有 `("Satya Nadella", "Microsoft", CEO)`，抓回来的新闻里同时出现这两个实体的句子很多，有些在谈财报，有些在谈云产品，有些在谈历史访谈。若把这些句子都当 `CEO` 正例，模型会学到“只要出现这两词就像 CEO”，最后高频实体共现会压过真正的关系表达。

---

## 核心机制与推导

多实例学习的第一步是“按实体对分组”。同一实体对 $(h,t)$ 的所有句子组成一个 `bag`。设第 $i$ 个 `bag` 为 $B_i=\{x_{i1},...,x_{in}\}$，其中 $x_{ij}$ 是第 $j$ 句的表示。远程监督只提供 bag 标签 $y_i$，不提供每句的真值 $y_{ij}$。

selective attention，白话解释是“让模型自己给每句打证据分数”，核心是 relation-aware 的加权聚合：
$$
e_{ij}=x_{ij}^TAr
$$
$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
$$
$$
s_i=\sum_j \alpha_{ij}x_{ij}
$$
其中 $r$ 是关系向量，$A$ 是可学习矩阵，$e_{ij}$ 表示“第 $j$ 句与关系 $r$ 的匹配程度”，$\alpha_{ij}$ 是 softmax 后的权重，$s_i$ 是 bag 表示。直观上，模型不是在证明某句一定正确，而是在训练时把梯度更多分给“更像证据”的句子。

玩具例子可以直接算。一个 `bag` 有 3 句，分数是 $e=[2,1,0]$，则：
$$
\alpha \approx [0.67,0.24,0.09]
$$
若句表示压缩成标量 $x=[0.9,0.2,0.1]$，则：
$$
s \approx 0.67\times0.9 + 0.24\times0.2 + 0.09\times0.1 \approx 0.66
$$
这说明第 1 句主导了 bag 判断，第 3 句即使是噪声，也只留下很小影响。

但 sentence attention 只解决“bag 内谁更可信”，还没有解决“整个 bag 也可能不可信”。于是会进一步引入 sample-level 可靠度。常见做法有三类。

第一类是 prototype。prototype，白话解释是“某个关系的典型表示中心”。如果一个样本离本关系原型很远，说明它可能被错误对齐。原型可写成某类高置信样本的均值：
$$
p_r=\frac{1}{|S_r|}\sum_{z\in S_r} z
$$
这里 $S_r$ 通常不是该类全部样本，而是高置信子集，否则原型本身会被噪声拖偏。

第二类是 contrastive learning。对比学习，白话解释是“拉近同类表示，拉远异类表示”。如果再把可靠度权重 $w$ 引入损失，就能降低噪声对表示空间的污染：
$$
L = -\log \frac{\sum_{p\in P} w_p \exp(\mathrm{sim}(z,z_p)/T)}
{\sum_{p\in P} w_p \exp(\mathrm{sim}(z,z_p)/T)+\sum_{n\in N} w_n \exp(\mathrm{sim}(z,z_n)/T)}
$$
其中 $P$ 是正样本集合，$N$ 是负样本集合，$T$ 是温度参数。若某个正样本疑似噪声，就让 $w_p$ 变小；若某个负样本本身也不可靠，也可以降低 $w_n$。

第三类是 curriculum learning。课程学习，白话解释是“先学简单样本，再学困难样本”。它不是直接删掉噪声，而是控制训练顺序。训练早期只让高置信样本进入主干网络，后期再逐步放宽。这样做的原因很现实：模型在初期表征还没稳定时，最容易被大噪声带偏。

把这几步连起来，就是一条完整链路：

1. 句子打分：估计哪句像证据。
2. bag 聚合：把多句变成一个 relation-aware 表示。
3. 样本加权：估计整个样本是否值得强学、弱学或延后学。

这三步的目标一致，都是在估计可信度，只是作用层次不同。

---

## 代码实现

实现上建议把逻辑拆成 `bag 构造`、`证据聚合`、`损失加权` 三段。不要把远程监督、注意力、原型更新、课程学习全塞进一个 `forward`，否则后续调试几乎不可做。

下面给一个可运行的最小 Python 例子。它不依赖深度学习框架，只演示“按句打分 -> softmax 聚合 -> 样本权重”的基本机制。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def bag_attention(sentence_scores, sentence_values):
    """
    sentence_scores: 每句和目标关系的匹配分数 e_ij
    sentence_values: 每句的表示，这里用标量做玩具例子
    """
    assert len(sentence_scores) == len(sentence_values)
    alpha = softmax(sentence_scores)
    bag_value = sum(a * v for a, v in zip(alpha, sentence_values))
    return alpha, bag_value

def weighted_cross_entropy(prob, target, sample_weight=1.0, eps=1e-12):
    """
    prob: 预测为正类的概率
    target: 0 或 1
    sample_weight: 样本可靠度
    """
    prob = min(max(prob, eps), 1 - eps)
    loss = -(target * math.log(prob) + (1 - target) * math.log(1 - prob))
    return sample_weight * loss

# 玩具 bag：3 句，其中第一句最像证据
scores = [2.0, 1.0, 0.0]
values = [0.9, 0.2, 0.1]

alpha, bag_value = bag_attention(scores, values)

# 将 bag_value 简化映射成一个“正类概率”
prob = min(max(bag_value, 0.0), 1.0)

# 高置信 bag
high_conf_loss = weighted_cross_entropy(prob, target=1, sample_weight=1.0)

# 低置信 bag：即便标签同样是正类，也降低其训练影响
low_conf_loss = weighted_cross_entropy(prob, target=1, sample_weight=0.2)

assert len(alpha) == 3
assert round(sum(alpha), 6) == 1.0
assert alpha[0] > alpha[1] > alpha[2]
assert 0.6 < bag_value < 0.7
assert low_conf_loss < high_conf_loss

print("attention:", [round(x, 3) for x in alpha])
print("bag_value:", round(bag_value, 3))
print("high_conf_loss:", round(high_conf_loss, 4))
print("low_conf_loss:", round(low_conf_loss, 4))
```

在真实工程里，代码结构通常是：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `Dataset` | 句子级 DS 对齐结果 | `(entity_pair, sentences, label)` | 按实体对组装 bag |
| `collate_fn` | 多个 bag | padding 后的 batch | 处理不同 bag 长度 |
| `SentenceEncoder` | 单句 token | 句向量 $x_{ij}$ | 编码语义 |
| `AttentionAggregator` | 一组句向量 + relation query | $\alpha_{ij}, s_i$ | 聚合证据 |
| `LossFn` | bag logits + sample weights | 标量 loss | 降低噪声影响 |

真实工程例子里，新闻关系抽取常见流程是：`KB 对齐 -> bag 构造 -> BERT/PCNN 编码 -> selective attention -> 关系分类 -> prototype 或 curriculum 重加权 -> 人工复核低置信结果`。如果一开始就同时打开 attention、prototype、contrastive、curriculum，往往很难判断收益来自哪一层。更稳的顺序通常是：先做纯 bag attention 基线，再逐步加样本级降噪。

---

## 工程权衡与常见坑

最常见的误解是把 attention 当解释。attention 更接近“软选择器”，不是因果证明。一个句子权重高，表示模型当前更依赖它更新参数，不代表这句就是人类可接受的唯一证据。若要做可解释抽取，仍然需要额外的证据标注、规则对齐或人工审核。

第二个核心权衡是“降噪强度”和“召回损失”。噪声压得太狠，模型会变保守，只敢预测头部关系和强模板句。长尾关系、弱表达句、跨风格句子都会掉召回。工程里最危险的不是噪声没压住，而是“看起来精度升了，实际业务覆盖率塌了”。

常见坑可以直接列出来：

| 坑 | 为什么会错 | 怎么规避 |
|---|---|---|
| `bag` 标签直接当句标签 | 把弱标签误当真标签，噪声被放大 | 坚持 bag-level 训练或只用高置信句伪标注 |
| 把 attention 当解释 | 权重高不等于语义必然正确 | 用人工抽检、证据标注或 counterfactual 验证 |
| prototype 用全噪声初始化 | 原型会从一开始就偏 | 只用高置信样本初始化，或用 EMA 平滑更新 |
| curriculum 过度筛样本 | 头部关系更强，长尾更容易被丢 | 设最小保留率，并监控类间分布 |
| 随机切分导致泄漏 | 同一实体对或相近文档进入训练和测试 | 按实体对、文档或时间切分 |

还有两个工程细节经常被忽略。

第一，远程监督里 `NA/no_relation` 往往占大头。若负类采样策略不稳，模型会学成“大多数都不是关系”，精度表面上不差，但业务价值很低。  
第二，评估不能只看整体 P/R/F1。至少还要看长尾关系召回、高置信样本比例变化、不同 bag 大小上的性能，以及人工抽检中的证据句质量。

一个实用检查清单是：

1. 每轮训练后，高置信样本比例是否单调上升过快。过快通常意味着模型过拟合头部模板。
2. 不同关系的保留率是否失衡。若长尾类几乎都被降权，后面很难补回来。
3. 同一实体对是否跨数据集泄漏。远程监督任务里这比普通文本分类更常见。

---

## 替代方案与适用边界

不同降噪方法不是互斥关系，而是处理同一问题的不同切面。attention 偏句子选择，prototype 偏类中心校正，对比学习偏表示空间，curriculum 偏训练顺序。选型时不要问“哪篇论文最强”，而要问“我的噪声主要发生在哪一层”。

| 方法 | 主要作用 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| bag attention | 在 bag 内找证据句 | 简单、稳定、易做基线 | 只能部分处理 bag 级噪声 | 证据句较明确、模板较稳定 |
| prototype | 用类中心校正样本 | 对类别混淆有效 | 原型初始化敏感 | 类间边界相对清晰 |
| contrastive learning | 优化表示空间结构 | 对迁移和低资源有效 | 构造正负对更复杂 | 预训练或半监督场景 |
| curriculum learning | 控制学习顺序 | 能稳定重噪声训练 | 超参敏感、容易伤召回 | 噪声极重、训练不稳定 |
| 人工复核 / 半监督校正 | 直接修正高风险样本 | 精度上限高 | 成本高 | 高价值业务、闭环系统 |

一个简单选型规则通常够用：

- 噪声一般，证据句明确：先上 `bag attention`。
- 噪声较重，关系容易混淆：加 `prototype` 或 `contrastive learning`。
- 噪声极重，早期训练很不稳：加 `curriculum learning`，必要时引入人工复核。
- 如果关系经常跨句表达，或者知识库严重不完备：单纯句子级降噪不会根治，需要改任务定义或增加文档级建模。

再给一个真实工程判断。新闻里的“创始人”“总部位于”通常句式稳定，bag attention 往往已经够用。医疗文本或法律文本里的关系表达更隐晦，实体共现和关系表达的间隔更长，单靠 attention 经常不稳，这时 prototype 或对比学习更有价值。若业务还是高风险领域，比如医疗知识图谱补全，最后一步通常都要把低置信输出送人工复核，而不是完全信任自动降噪。

---

## 参考资料

1. [Distant supervision for relation extraction without labeled data](https://aclanthology.org/P09-1113/)
2. [Neural Relation Extraction with Selective Attention over Instances](https://aclanthology.org/P16-1200/)
3. [Distant Supervision for Relation Extraction with an Incomplete Knowledge Base](https://aclanthology.org/N13-1095/)
4. [Relation Extraction with Weighted Contrastive Pre-training on Distant Supervision](https://aclanthology.org/2023.findings-eacl.195/)
5. [Curriculum learning for distant supervision relation extraction](https://www.sciencedirect.com/science/article/pii/S1570826820300093)
6. [Distantly Supervised Relation Extraction Based on Residual Attention and Self Learning](https://link.springer.com/article/10.1007/s11063-024-11497-0)

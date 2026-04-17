## 核心结论

知识图谱补全（Knowledge Graph Completion，白话说就是“把知识库里漏掉的事实补上”）主要做两类事：

1. 链接预测（Link Prediction）：给定 $(h, r, ?)$ 或 $(?, r, t)$，从候选实体里找出最可能的缺失实体。
2. 三元组分类（Triple Classification）：给定完整三元组 $(h, r, t)$，判断它更像真事实还是假事实。

这两类任务的共同核心不是“生成一句自然语言”，而是对候选三元组打分。模型通常先把实体和关系映射成向量，再用评分函数计算
$$
score=f_r(h,t)
$$
分数越高，表示这个三元组越可信。链接预测看“谁排第一”，三元组分类看“是否超过阈值”。

对初学者，最重要的判断标准有三项：

| 指标 | 作用 | 直观含义 |
| --- | --- | --- |
| MRR | 看正确答案平均排得多靠前 | 第一名给 1，第二名给 1/2，越靠前越好 |
| Hits@K | 看正确答案是否进前 K | 例如 Hits@10 就是“有没有进前 10” |
| AUC-ROC | 看真假三元组整体可分性 | 越接近 1，说明正负样本越容易分开 |

玩具例子很直接。已知 `(小明, 喜欢, ?)`，候选实体是 `{小红, 小李, 苹果}`。模型分别打分为 `0.91, 0.42, 0.03`，那么预测结果就是 `小红`。这就是链接预测的最小形式：评分，排序，取最优。

真实工程里，任务通常不是“猜一个名字”，而是补全一个业务知识库。比如制造企业的工业知识图谱中，已有 `(缝合车间, 使用, 工艺A)`、`(缝合车间, 负责, 足球外壳加工)`，但缺少 `(缝合车间, 使用, ?)` 的某个设备或材料事实。补全模型会在候选设备、材料、工艺里排序，把最可能的缺失项顶到前面，供系统检索、规则引擎或人工审核使用。

工程上要特别记住两点。第一，评估必须使用 filtered setting，也就是把“其他已知真实答案”从候选里滤掉，否则指标会失真。第二，FB15k-237 是最常见的标准基准之一，它专门去掉了明显的反向关系泄露；公开结果里，reasoning 风格任务常见最佳 MRR 大约在 `0.36` 量级，不同模型家族和评测协议会有波动，不能把不同论文的数字直接横向硬比。

---

## 问题定义与边界

知识图谱（Knowledge Graph，白话说就是“把事实写成节点和边的图”）里的基本单位是三元组：
$$
(h,r,t)
$$
其中 $h$ 是头实体，$r$ 是关系，$t$ 是尾实体。比如 `(乔布斯, 创立, 苹果)`。

链接预测的问题定义是：当三元组缺一端时，利用已有图结构和训练数据补全缺失实体。形式上分两种：

- 尾实体预测：$(h,r,?)$
- 头实体预测：$(?,r,t)$

三元组分类的问题定义是：输入一个完整三元组，输出真或假，或者输出一个概率分数再与阈值比较。

这两个任务相似，但目标不同。链接预测强调“排序”，三元组分类强调“判别”。

| 任务 | 输入 | 输出 | 候选范围 | 主要评价方式 |
| --- | --- | --- | --- | --- |
| 链接预测 | `(h,r,?)` 或 `(?,r,t)` | 缺失实体的排名列表 | 通常是实体全集或受限候选集 | MRR、Hits@K |
| 三元组分类 | `(h,r,t)` | 真/假或概率 | 单个三元组 | AUC-ROC、Accuracy、F1 |

边界一：闭世界与开世界。

闭世界（Closed World，白话说就是“没见到就先当不存在”）通常意味着评估时只在固定实体集里选答案，很多 benchmark 都是这个设定。开世界（Open World，白话说就是“图外也可能有新实体或新事实”）更接近真实业务，但更难，因为候选范围不再稳定，很多标准嵌入方法在这里会明显吃力。

边界二：正负样本怎么来。

知识图谱里一般只有正样本，也就是已知真事实。负样本通常靠“扰动”构造，例如把 `(乔布斯, 创立, 苹果)` 改成 `(乔布斯, 创立, 微软)`。这叫负采样（negative sampling，白话说就是“人工制造假样本”）。但这里有一个关键风险：你造出来的“假样本”可能其实是真事实，只是训练集没收录。

| 样本类型 | 典型来源 | 风险 |
| --- | --- | --- |
| 正样本 | 训练集、验证集、测试集中的已知三元组 | 可能有标注偏差 |
| 负样本 | 替换头实体或尾实体得到的扰动三元组 | 伪负样本，即“看起来假，其实可能真” |

一个新手容易混淆的点是：三元组分类不是简单把链接预测的 top1 结果当答案。它需要阈值。比如 `(苹果, 创立, 乔布斯)` 和 `(乔布斯, 创立, 苹果)` 在语义上完全不同，关系方向错了就应当判假。也就是说，分类任务不仅要会排序，还要能给出可分的分数区间。

---

## 核心机制与推导

主流方法是嵌入模型（embedding-based model，白话说就是“先把实体和关系编码成稠密向量，再算匹配程度”）。核心流程可以压缩成四步：

1. 为每个实体学习向量 $\mathbf{e}$，为每个关系学习向量或矩阵 $\mathbf{r}$。
2. 定义评分函数 $f_r(h,t)$。
3. 用正样本和负样本训练，让真三元组得分更高。
4. 推理时遍历候选实体，排序后得到答案。

最经典的评分函数之一是 TransE。它把关系理解为“向量平移”，也就是：
$$
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
$$
常见打分形式为：
$$
f_r(h,t)=-\|\mathbf{h}+\mathbf{r}-\mathbf{t}\|
$$
距离越小，分数越高，三元组越可信。

白话解释：如果“创立”这个关系真能把“乔布斯”推到“苹果”附近，那么 `(乔布斯, 创立, 苹果)` 的距离就会小；而 `(乔布斯, 创立, 微软)` 往往距离更大。

再看一个玩具例子。候选实体只有 `{苹果, 微软}`，对 `(乔布斯, 创立, ?)` 打分：

| 候选 | 分数 |
| --- | --- |
| 苹果 | 0.90 |
| 微软 | 0.20 |

排序后苹果的 `rank=1`，则这个样本对 MRR 的贡献是：
$$
\frac{1}{rank}=\frac{1}{1}=1
$$
如果另一个样本正确答案排第 4，它对 MRR 的贡献是 $1/4=0.25$。所以 MRR 定义为：
$$
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
$$

Hits@K 则更简单：
$$
Hits@K=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}(rank_i \le K)
$$
只要正确答案进前 $K$ 就记 1，否则记 0。它不关心第 1 和第 3 的细差，但很适合业务里“前几条候选是否可用”的判断。

三元组分类常用 AUC-ROC。AUC-ROC（白话说就是“把不同阈值下的真假区分能力积分成一个数”）本质上衡量：随机抽一个正样本和一个负样本，模型给正样本更高分的概率有多大。AUC 越接近 1，分数排序越稳定；AUC 接近 0.5，说明几乎和乱猜一样。

评估时最容易出错的是 filtered setting。设测试问题是 `(乔布斯, 创立, ?)`，而图里真实答案不只 `苹果` 一个，另一个真实尾实体也存在。如果你把所有实体都直接拿来排，另一个真实实体会挤占排名，导致正确答案看起来被“错排”了。于是标准做法是：把除当前目标外、所有在训练/验证/测试中已知为真的候选三元组先移除，再算 rank。

伪代码可以写成：

```text
for query in test_triples:
    candidates = all_entities
    scores = []
    for e in candidates:
        triple = (h, r, e)
        if triple is true and e != gold_tail:
            continue   # filtered
        scores.append(score(triple))
    rank = position_of(gold_tail in sorted(scores, desc=True))
```

这一步不是细节，而是 benchmark 能否对齐论文的前提。尤其在 FB15k-237 这类数据集上，如果漏了 filter，MRR 和 Hits@K 会被系统性扭曲。

---

## 代码实现

下面给一个可运行的最小 Python 示例。它不依赖深度学习框架，只演示三个核心动作：打分、filtered 排序、计算 MRR/Hits/AUC。这里用的是简化版 TransE 风格打分。

```python
import math

# 实体和关系向量
entity_vec = {
    "jobs": [0.2, 0.8],
    "apple": [1.1, 1.0],
    "microsoft": [0.0, 0.1],
    "pixar": [1.0, 0.9],
}
relation_vec = {
    "founded": [0.9, 0.2],
}

# 已知真实三元组全集，用于 filtered evaluation
true_triples = {
    ("jobs", "founded", "apple"),
    ("jobs", "founded", "pixar"),
}

def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def score(h, r, t):
    hv = entity_vec[h]
    rv = relation_vec[r]
    tv = entity_vec[t]
    translated = [x + y for x, y in zip(hv, rv)]
    # 距离越小越可信，因此取负号变成“分数越大越好”
    return -l2_distance(translated, tv)

def filtered_rank(h, r, gold_t):
    candidates = []
    for t in entity_vec:
        triple = (h, r, t)
        if triple in true_triples and t != gold_t:
            continue
        candidates.append((t, score(h, r, t)))
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    for idx, (t, _) in enumerate(ranked, start=1):
        if t == gold_t:
            return idx, ranked
    raise ValueError("gold entity not found")

def mrr(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks)

def hits_at_k(ranks, k):
    return sum(1 for r in ranks if r <= k) / len(ranks)

def auc_roc(pos_scores, neg_scores):
    total = 0
    good = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            total += 1
            if ps > ns:
                good += 1
            elif ps == ns:
                good += 0.5
    return good / total

rank, ranked_list = filtered_rank("jobs", "founded", "apple")
assert rank == 1
assert ranked_list[0][0] == "apple"

ranks = [1, 2, 4]
assert abs(mrr(ranks) - (1 + 0.5 + 0.25) / 3) < 1e-9
assert hits_at_k(ranks, 1) == 1 / 3
assert hits_at_k(ranks, 3) == 2 / 3

pos_scores = [score("jobs", "founded", "apple"), score("jobs", "founded", "pixar")]
neg_scores = [score("jobs", "founded", "microsoft")]
auc = auc_roc(pos_scores, neg_scores)
assert 0.0 <= auc <= 1.0
assert auc == 1.0
```

这段代码对应的思路很标准：

- `score(h, r, t)` 负责计算三元组可信度。
- `filtered_rank` 在打分前应用 filtered 规则。
- `mrr` 和 `hits_at_k` 用排名列表更新指标。
- `auc_roc` 用正负分数比较实现一个最小版本的 AUC。

真实工程不会遍历 Python 字典，而会用矩阵批量计算。比如把所有候选尾实体向量一次性取出，与同一个 $(h,r)$ 批量做分数计算。这样才能在十万到百万实体规模上把评估时间压下来。

真实工程例子可以设成供应链知识图谱。假设已有三元组：

- `(供应商A, 提供, 皮革)`
- `(工艺B, 需要, 皮革)`
- `(缝合车间, 使用, 工艺B)`

现在缺失 `(缝合车间, 使用, ?)` 的尾实体候选。系统会先生成所有候选工艺或设备，再用模型批量算分，排序后把最可信的前几项给业务侧。若任务是三元组分类，则对 `(缝合车间, 使用, 工艺B)` 直接输出概率，供规则系统决定是否自动入库或进入人工审核。

---

## 工程权衡与常见坑

知识图谱补全真正难的地方，不是公式，而是评估协议、负样本和系统规模。

最常见的坑如下：

| 常见坑 | 表现 | 后果 | 规避方式 |
| --- | --- | --- | --- |
| 漏掉 filtered evaluation | 直接对全体候选排序 | MRR/Hits 与论文不可比 | 过滤掉其他已知真实三元组 |
| inverse leakage | 训练集和测试集存在明显反向泄露 | 模型看起来很强，实际只是记住反向边 | 采用 FB15k-237 等去泄露数据集 |
| 伪负样本过多 | 替换出的“负样本”其实可能为真 | 训练目标被污染 | 用 typed negative sampling 或业务约束过滤 |
| 只看 Hits@10 | 指标看起来不错 | 但 top1 很差，线上不可用 | 联合看 MRR、Hits@1、Hits@10 |
| 阈值直接复用 | 三元组分类直接拿 0.5 当阈值 | 不同关系的分数分布不一致 | 在验证集上按关系或全局调阈值 |
| embedding 维度过大 | 显存或内存爆炸 | 训练和评估都变慢 | 分块打分、混合精度、采样候选 |

FB15k-237 常被用来教学和论文复现，原因不是它“最真实”，而是它至少处理了原始 FB15k 中明显的 inverse relation leakage。白话说，原始数据里可能出现“测试集问 `(A, parent_of, ?)`，训练集却已经有 `(B, child_of, A)`”，模型根本不用推理，只要记住反向关系就够了。FB15k-237 去掉了大量这种捷径，所以更适合比较方法本身的能力。

如果做线上系统，还要额外考虑两类权衡。

第一类是召回与精度。链接预测常常先做大召回，再由规则或分类器重排。原因是业务要的不只是“一个最优答案”，而是“前几条候选足够可审”。这时 Hits@K 比单纯 Accuracy 更有意义。

第二类是表达能力与成本。TransE 很快、易实现，但对一对多、多对多、对称关系表达有限。ComplEx 这类模型能更好表达非对称关系，但参数和实现复杂度更高。GNN 类方法利用局部结构更强，但训练成本和图采样成本也更高。

一个实用 checklist：

| 检查项 | 是否必须 |
| --- | --- |
| 训练、验证、测试是否严格分离 | 必须 |
| 评估是否使用 filtered setting | 必须 |
| 负采样是否考虑实体类型约束 | 强烈建议 |
| 是否同时报告 MRR、Hits@1、Hits@10 | 强烈建议 |
| 三元组分类阈值是否在验证集调优 | 强烈建议 |
| 是否检查数据集中反向泄露和重复边 | 必须 |

---

## 替代方案与适用边界

嵌入模型不是唯一方案。按工程视角，至少有三类路线。

| 路线 | 输入要求 | 优点 | 缺点 | 典型场景 |
| --- | --- | --- | --- | --- |
| Embedding-based | 三元组即可 | 实现简单，推理快，适合大规模候选排序 | 可解释性弱，对复杂逻辑有限 | 通用补全、召回排序 |
| GNN-based | 需要图邻接结构 | 能聚合邻居信息，结构利用更强 | 训练复杂，图大时开销高 | 稠密图、关系传播明显的场景 |
| Rule-based | 需要可挖掘规则或专家知识 | 可解释性强，便于审计 | 覆盖率有限，维护成本高 | 金融、医疗、工业规则系统 |

GNN（图神经网络，白话说就是“让节点把邻居的信息汇总后再做判断”）适合图结构密、局部模式强的任务。比如一个设备节点周围连着工艺、车间、原料、故障码，邻居本身就能提供很强线索。这时 R-GCN、CompGCN 一类方法往往比纯打分模型更有优势。

规则推理则适合高可解释场景。例如有明确规则：
- 若 `(设备X, 安装于, 车间Y)` 且 `(车间Y, 执行, 工艺Z)`，则 `(设备X, 服务于, 工艺Z)` 可能成立。

这种方法对审计友好，但覆盖率依赖规则质量，不适合完全靠规则硬撑整个补全系统。

三元组分类还有一条常见替代路线：把嵌入模型当特征抽取器，再接一个二分类器。比如把 `score`、路径特征、实体类型一致性、规则命中数一起送进逻辑回归或梯度提升树。这样做的好处是线上阈值和业务规则更容易控制，缺点是系统链路更长。

适用边界可以这样判断：

- 如果你只有结构化三元组，想先做一个能跑、能评估、能上线候选召回的版本，先用 embedding-based。
- 如果图局部结构很强，且可以承担更高训练成本，考虑 GNN。
- 如果行业合规要求高，必须解释“为什么补出了这条边”，就要引入规则推理或规则与模型的混合方案。
- 如果数据是开世界，实体不断新增，仅靠静态实体嵌入往往不够，需要结合文本编码、实体对齐或检索增强方法。

---

## 参考资料

- Knowledge Graph Completion 综述：介绍链接预测、三元组分类、嵌入模型与开放世界问题。  
  https://academic.oup.com/bib/article/doi/10.1093/bib/bbae161/7644136

- 开放世界知识图谱补全综述：适合理解闭世界与开世界设定差异。  
  https://www.sciencedirect.com/science/article/pii/S002002552101207X

- FB15k-237 数据卡：说明该数据集用于缓解 inverse relation leakage。  
  https://huggingface.co/datasets/KGraph/FB15k-237

- PyG 的 FB15k-237 文档：给出常见实体数、关系数和三元组规模，便于快速核对实验配置。  
  https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.FB15k_237.html

- 评估指标与排名任务说明：包含 MRR、Hits@K 等定义与实验语境。  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10264130/

- 工业知识图谱补全案例：展示补全在制造业场景中的实际用途。  
  https://www.researchgate.net/publication/381775761_Link_Prediction_in_Industrial_Knowledge_Graphs_A_Case_Study_on_Football_Manufacturing

- TorchDrug 基准页：可用来快速查看常见方法在 FB15k-237 上的公开结果量级。  
  https://torchdrug.ai/docs/benchmark/reasoning.html

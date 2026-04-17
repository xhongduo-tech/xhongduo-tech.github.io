## 核心结论

通用 Embedding 模型适合“跨领域大致相似”，不擅长“领域内细粒度区分”。Embedding 可以理解为“把一句话压缩成一个向量，方便按距离做检索”。在医疗、法律、代码这类垂直领域里，通用模型常把“术语相近但结论不同”的文本放得过近，结果是召回不稳定，尤其容易漏掉真正相关的文档。

更有效的做法不是单纯继续堆训练数据，而是做领域对比学习微调。对比学习可以理解为“让查询更靠近正确答案、远离干扰项”的训练方式。核心抓手有两个：

| 抓手 | 作用 | 做错会怎样 |
| --- | --- | --- |
| 高质量正样本 | 教模型什么叫“真的相关” | 正例太松，模型学不到领域边界 |
| hard negative | 教模型什么叫“看起来像，但其实不对” | 负例太弱，模型只会区分简单样本；负例太强且含假负例，会把正确语义推远 |

NV-Retriever 一类实践表明，domain-specific contrastive fine-tuning 配合 positive-aware hard negative mining，能显著提升领域检索效果。你给出的目标值“Recall@10 从 0.68 提升到 0.89”可以理解为一个典型垂直场景下的工程量级：不是所有任务都能到这个数，但“通过高质量正负样本把召回拉高一个明显台阶”是可复现的。

还要注意一个反直觉点：hard negative 不是越难越好。CIKM 2022 的结论很关键，强检索器挖出来的 top-K 结果里往往混入大量假负例。假负例可以理解为“标注里没写相关，但实际上相关”的样本。直接拿它们当负例训练，模型会被错误惩罚，结果可能出现“召回更强，排序反而更差”的退化。

---

## 问题定义与边界

这里的问题不是“Embedding 模型太弱”，而是“训练目标和领域检索目标不一致”。

通用 Embedding 常见训练目标是跨任务泛化，所以它学到的是广义语义邻近关系。例如“糖尿病并发症”和“糖尿病筛查建议”会被判定为比较近，因为都属于糖尿病语义簇。但医疗检索真正要回答的问题可能是：“早期糖尿病筛查指南中 HbA1c 的阈值是什么？”这时模型需要区分的是指南、诊断、并发症、随访、药物禁忌之间的专业边界，而不是只知道它们都和糖尿病有关。

问题边界主要有四个：

| 边界 | 说明 |
| --- | --- |
| 目标是检索，不是生成 | 微调的是向量空间，目标指标通常是 Recall@K、NDCG@K，而不是 BLEU 或 ROUGE |
| 重点在领域内区分 | 医疗、法律、代码等垂直域比开放域更依赖细粒度语义 |
| 标注不完全是常态 | 检索数据常有 pooling bias，即“只标注了被某些召回器捞出来的一小部分文档” |
| 负样本质量决定上限 | 随机负例太容易，强检索器负例又可能是假负例，必须做筛选 |

所谓 pooling bias，可以直白理解为“数据集只在一个有限候选池里做人工标注，所以池外可能还有没标出来的正确答案”。这会直接影响微调。CIKM 2022 指出，用更强的检索器构造 top-ranked negatives，不一定提升训练效果；以 MS MARCO 为例，RepBERT 的 Recall@1k 比 BM25 高约 15%，但用它采出的负例训练 BERT-base ranker，MRR@10 反而比 BM25 采样低约 9%。这说明问题不在“强检索器没用”，而在“把强检索器返回的未标注文档直接当负例”这件事本身是错的。

玩具例子可以先看一个最小场景：

- Query: “高血压复发机制”
- 正例: “长期血管重塑导致血压反复升高的病理机制”
- 候选负例 A: “高血压患者的饮食建议”
- 候选负例 B: “继发性高血压的鉴别诊断”
- 候选负例 C: “高血压复发的流行病学随访结论”

如果只做随机负采样，模型很容易把无关文本分开，却学不会 A、B、C 之间的细差别。如果直接把最相似的 C 当负例，又可能误伤，因为 C 很可能对某些标注标准来说本来就算相关。真正有效的训练，是在“足够难”和“不是假负例”之间找到平衡。

---

## 核心机制与推导

核心训练目标通常是 InfoNCE。它可以理解为“在一堆候选里，让正例的相似度最大”。设查询向量为 $q_i$，正例为 $p_i^+$，hard negatives 为 $p_{i,1}^-, \dots, p_{i,K}^-$，batch 内其他样本也作为负例，则单样本损失可以写成：

$$
\ell_i=-\log
\frac{\exp(\mathrm{sim}(q_i,p_i^+)/\tau)}
{\exp(\mathrm{sim}(q_i,p_i^+)/\tau)+
\sum_{j=1}^{K}\exp(\mathrm{sim}(q_i,p_{i,j}^-)/\tau)+
\sum_{p^- \in \text{InBatch}}\exp(\mathrm{sim}(q_i,p^-)/\tau)}
$$

其中 $\mathrm{sim}$ 常用余弦相似度，$\tau$ 是温度参数。温度可以理解为“放大或压缩相似度差异的旋钮”。$\tau$ 越小，模型越在意很小的分数差别；$\tau$ 太小则会让训练变得尖锐、不稳定。

为什么 hard negative 有用？因为随机负例大多离 query 很远，它们在分母里的贡献很小，梯度也小。模型很快就学会“区分简单不相关内容”，却学不会“区分专业近义项”。hard negative 会让分母里出现和正例很接近的样本，从而产生更强的训练信号。

但 hard negative 需要过滤。NV-Retriever 的 positive-aware mining 思路，就是用正例分数做锚点，只保留“难，但没难到像正例”的负样本。一个常见形式是 TopK-PercPos：

$$
\mathrm{sim}(q, n) \le \alpha \cdot \mathrm{sim}(q, p^+)
$$

其中 $\alpha$ 常设为 $0.95$ 左右。白话解释是：如果某个候选负例和 query 的相似度已经接近正例的 95% 甚至更高，那它大概率不是一个安全负例，先不要拿来训练。

继续看前面的玩具例子。假设教师模型给出：

| 样本 | 与 query 的相似度 |
| --- | --- |
| 正例 | 0.84 |
| 负例 A | 0.53 |
| 负例 B | 0.71 |
| 负例 C | 0.82 |

当 $\alpha=0.95$ 时，阈值是 $0.84 \times 0.95 = 0.798$。这时 A、B 可以保留，C 要丢掉，因为它已经太接近正例，极可能是假负例。这样做的好处是，模型仍然被迫区分专业近义内容，但不会被错误监督拖偏。

两阶段训练的逻辑也很直接：

| 阶段 | 目标 | 数据特点 | hard negatives |
| --- | --- | --- | --- |
| Stage 1 | 先把检索边界拉清楚 | 以检索对为主 | 少量，通常 1 个左右 |
| Stage 2 | 再扩大泛化范围 | 加入多任务正例和更多难例 | 更多，通常 5 个左右 |

Stage 1 更像“先把领域主干语义学稳”，Stage 2 更像“再补多任务覆盖，避免过拟合成死板词表匹配”。

真实工程例子是医疗指南检索。Query 是“早期糖尿病筛查指南”。正例来自指南正文段落，候选负例可能来自“糖尿病并发症综述”“妊娠期糖尿病风险因素”“HbA1c 检验原理”。这些文本都高度相关，但并不都回答筛查指南问题。通过 InfoNCE 加上 positive-aware hard negative，模型学到的不是“见到 diabetes 就靠近”，而是“指南、筛查、阈值、适用人群”这个更窄的专业邻域。

---

## 代码实现

下面给一个可运行的最小 Python 例子，演示两件事：

1. 用正例分数过滤 hard negatives。
2. 计算一个简化版 InfoNCE 损失。

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(a):
    return math.sqrt(sum(x * x for x in a))

def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def select_hard_negatives(q, pos, candidates, margin_pct=0.95, topk=2):
    pos_score = cosine(q, pos)
    threshold = pos_score * margin_pct
    scored = []
    for name, vec in candidates:
        s = cosine(q, vec)
        if s <= threshold:
            scored.append((name, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return pos_score, threshold, scored[:topk]

def info_nce_loss(q, pos, negatives, tau=0.1):
    pos_score = cosine(q, pos) / tau
    denom = math.exp(pos_score)
    for _, neg in negatives:
        denom += math.exp(cosine(q, neg) / tau)
    return -math.log(math.exp(pos_score) / denom)

# toy vectors
q = [1.0, 0.9, 0.0]
pos = [0.95, 0.85, 0.05]

candidates = [
    ("diet_advice", [0.8, 0.2, 0.1]),
    ("secondary_hypertension", [0.85, 0.55, 0.1]),
    ("relapse_followup", [0.96, 0.82, 0.02]),  # too close, likely false negative
]

pos_score, threshold, selected = select_hard_negatives(q, pos, candidates, margin_pct=0.95, topk=2)

selected_names = [name for name, _ in selected]
assert "relapse_followup" not in selected_names
assert len(selected) == 2

negatives = [(name, vec) for name, vec in candidates if name in selected_names]
loss = info_nce_loss(q, pos, negatives, tau=0.2)

assert loss > 0
assert threshold < pos_score

print("pos_score:", round(pos_score, 4))
print("threshold:", round(threshold, 4))
print("selected:", selected_names)
print("loss:", round(loss, 4))
```

这个例子只是教学版，但工程逻辑和真实系统一致：先用教师模型或已有检索器给候选打分，再做“接近正例但不越界”的负样本筛选，然后送入对比学习。

真实工程实现通常分三步：

```bash
python compute_negatives.py --policy topk_percpos --margin_pct 0.95
python train_stage1.py --epochs 12 --hard_negatives 1 --batch 32
python train_stage2.py --epochs 12 --hard_negatives 5 --batch 8
```

这类流程里，`compute_negatives.py` 负责离线构造 hard negatives，`train_stage1.py` 和 `train_stage2.py` 负责分阶段微调。LoRA 可以理解为“只训练少量低秩适配参数，而不是更新整个大模型”。它的意义是把显存和训练成本压下来，适合中小团队在已有基础模型上做领域适配。

部署时，一般把最终 embedding 模型接到向量索引库，比如 FAISS 的 IVF-PQ 或 HNSW。FAISS 可以理解为“给大量向量做近邻检索的基础设施”。线上链路通常是：

1. 用户 query 编码成向量
2. 在向量库召回 top-K 段落
3. 可选地再经过 reranker 重排
4. 把结果送给 RAG 生成模块

如果你的场景是代码库检索，一个真实工程例子是“排查 Python 数据管道中的空值传播”。通用 Embedding 容易把“空值处理”“类型注解”“Pandas merge”混成同一堆文本；做了领域微调后，模型会更稳定地区分“缺失值传播路径”“数据清洗策略”“连接键为空导致的 join 异常”这些具体语义。

---

## 工程权衡与常见坑

第一类坑是把 hardest negative 直接当最好负例。很多团队会默认“相似度越高越有训练价值”，这不对。难例过强时，最常见的问题不是训练变慢，而是把未标注正例误当负例，直接污染监督信号。

第二类坑是只用随机负例。这样模型训练会很稳定，但学不到细粒度边界。表面现象通常是训练 loss 很好看，线上检索却仍然抓不到真正专业答案。

第三类坑是只看平均指标，不拆 query 类型。领域检索里，按 query 类型分桶几乎是必须的。指南类、定义类、诊断类、代码报错类的收益往往不同。如果只看总体 Recall@10，可能掩盖某一类关键 query 的退化。

一个实用检查表如下：

| 参数/现象 | 建议 | 风险信号 |
| --- | --- | --- |
| `margin_pct` | 从 0.95 附近起试 | 太高会混入假负例，太低会丢掉有价值难例 |
| easy:hard 比例 | 保留 easy negative 混合训练 | 全是 hard negative 时容易震荡 |
| hard negative 上限 | 先控制在较小 top-K | 一次塞太多，分母噪声过大 |
| batch 内负例 | 默认开启 | batch 太小会削弱 in-batch 效果 |
| 评估指标 | Recall@10 + NDCG@10 + 线上成功率 | 只看单一指标容易误判 |
| 数据清洗 | 去重、切块一致、标签一致 | chunk 策略变化会让对比结果失真 |

第四类坑是训练集和部署切块方式不一致。Chunk 可以理解为“把长文切成可检索片段”。如果训练时用 512 token 段落，线上却用 128 token 句块，向量空间会错位，召回收益会明显缩水。

第五类坑是忽视教师模型质量。positive-aware mining 依赖教师分数。如果教师模型本身不懂目标领域，筛出来的 hard negatives 质量也有限。医疗、法律、代码场景里，教师模型最好至少经过同领域预适配，或者先用现有检索日志做初筛。

---

## 替代方案与适用边界

如果拿不到高质量教师模型，或者标注数据的 pooling bias 很重，可以考虑 CET。CET 可以理解为“同时学一个相关性模型和一个被选中概率模型”，用来估计哪些样本只是因为数据构造过程没被标出来，而不是真的不相关。它不是替代对比学习，而是在负样本不可信时，先做偏差校正。

从建模角度看，CET 试图联合估计：

- $\rho(q,d)$：文档对 query 的真实相关性
- $\lambda(q,d)$：文档被选进标注池的概率

当训练集天然带偏时，直接把未标注样本视为负例，相当于默认 $\lambda(q,d)$ 不影响监督，这个假设通常不成立。CET 用联合学习去修正这个问题，适合“标注稀缺且检索池偏差明显”的环境。

替代路线可以按资源条件选：

| 场景 | 更合适的方案 |
| --- | --- |
| 有领域问答对，有教师模型 | 对比学习微调 + positive-aware hard negatives |
| 只有少量标注，没有强教师 | 先用 BM25 或随机负例做保守版本，再迭代 |
| 标注池偏差严重 | CET 或其他 debias 方法先纠偏 |
| 算力有限 | LoRA 微调，不全量更新 |
| 线上延迟极严 | 先保留通用 embedding，离线蒸馏后再灰度 |

法律场景是一个典型边界例子。假设团队没有教师 embedding，也没有稳定的 case-law 标注集。如果这时直接从强检索器 top-K 里取 hardest negatives，很可能把“不同判例但同一法条适用”的文本误当负例。更稳妥的做法是先做 CET 或保守采样，等有了更可靠的相关性估计，再上 positive-aware mining。

所以结论不是“所有领域都必须做两阶段微调”，而是：只要你的检索失败主要来自“专业近义项区分不清”，Embedding 微调就有价值；只要你的负样本存在较高假负例风险，就必须先解决采样偏差，再谈 hard negative 强化。

---

## 参考资料

- Moreira et al., *NV-Retriever: Improving text embedding models with effective hard-negative mining*, arXiv:2407.15831. https://arxiv.org/abs/2407.15831
- NVIDIA, *nvidia/NV-Retriever-v1 model card*. https://huggingface.co/nvidia/NV-Retriever-v1
- Cai et al., *Hard Negatives or False Negatives: Correcting Pooling Bias in Training Neural Ranking Models*, CIKM 2022. https://doi.org/10.1145/3511808.3557343
- Jiafeng Guo 团队公开 PDF：*Hard Negatives or False Negatives*. https://jiafengguo.github.io/2022/CIKM2022-Hard%20Negatives%20or%20False%20Negatives.pdf
- Meta 工程经验综述：*Embedding-based retrieval in Facebook Search*. https://readmedium.com/embedding-based-retrieval-in-facebook-search-ed801accabc8

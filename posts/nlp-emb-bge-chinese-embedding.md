## 核心结论

BGE，白话讲就是“把一句中文压成一个向量，并让语义接近的文本在向量空间里靠近”的模型系列。它的关键不是只做一次预训练，而是分成两步：先用 RetroMAE 预训练学语言结构，再用对比学习把“查询”和“答案段落”对齐到同一检索空间。

对中文场景，`bge-large-zh` 的价值很明确。它是 1024 维向量模型，官方模型卡写明训练流程是 `RetroMAE pre-train + contrastive fine-tune`，并且在 C-MTEB 中文榜上长期强于同代通用模型。以官方公开分数看，`bge-large-zh` 的 Retrieval 分数是 71.53，`multilingual-e5-large` 是 63.66，差距接近 8 分；若看 v1.5 版本，`bge-large-zh-v1.5` Retrieval 为 70.46。

BGE-M3 则不是“更大的单向量模型”，而是“三路检索器合体”。它同时输出 dense、sparse、multi-vector 三种表示。白话讲，dense 负责“语义像不像”，sparse 负责“关键词中不中”，ColBERT 风格多向量负责“局部片段是否精确匹配”。这让它更适合 RAG，因为真实查询既有同义改写，也有术语硬匹配。

一个可操作的判断是：

| 模型 | 维度 | 公开能力定位 | 适合什么 |
|---|---:|---|---|
| BGE-large-zh | 1024 | 中文 dense 检索强 | 中文 FAQ、知识库、语义搜索 |
| BGE-large-zh-v1.5 | 1024 | 相似度分布更稳，不用 instruction 也更能用 | 直接上线中文检索 |
| BGE-M3 | 1024 | dense+sparse+multi-vector 混合检索 | RAG、多术语场景、长文档检索 |

---

## 问题定义与边界

BGE 要解决的问题不是“文本分类”，而是“中文语义检索对齐”。所谓对齐，白话讲就是：用户输入很短的问句，库里存的是很长的段落，但模型仍然要把它们映射到同一片向量区域。

这里有两个边界必须先说清：

第一，RetroMAE 预训练本身不等于可用检索模型。预训练目标是重建文本，也就是“从被遮住的句子中恢复原内容”。这能提升编码器对中文结构、词序、上下文的理解，但它不会自动学会“用户问句应该靠近哪段答案”。所以官方文档明确说明：预训练后的模型不能直接拿来算相似度，必须再做对比学习微调。

第二，中文短查询通常需要 instruction。instruction，白话讲就是“给查询加一句任务提示”，例如“为这个句子生成表示以用于检索相关文章：”。原因很简单：短查询信息量太少，模型需要知道当前任务是“拿去查段落”，不是做普通语义编码。BGE v1 时代这一点更明显，v1.5 缓解了问题，但在短 query 对长 passage 的检索里，显式 instruction 仍常常有效。

可以把整个流程压成一张表：

| 阶段 | 输入 | 学到什么 | 产出 |
|---|---|---|---|
| RetroMAE 预训练 | 中文通用语料 | 语言结构、上下文恢复能力 | 更强编码器 |
| 对比学习微调 | `(query, positive, negative)` 三元组 | 查询和答案段落的相对距离 | 可检索 embedding |
| 部署检索 | query + 索引库 | 召回与排序 | RAG 候选文档 |

玩具例子很容易理解。假设库里有两段文本：

- 段落 A：“Python 中列表和元组的区别是可变性不同。”
- 段落 B：“MySQL 事务的隔离级别包括读已提交和可重复读。”

用户查询是“python tuple 能修改吗”。

如果只有预训练，模型可能只学到“Python、列表、元组”这些词的语言共现；但做完对比学习后，它会进一步学到这个查询应当靠近段落 A，而远离段落 B。也就是把“短问句”和“长解释”压到同一个向量簇。

真实工程例子是企业内部知识库。用户问“报销单附件大小限制”，文档里真正的句子可能是“员工费用报销系统单次上传附件总大小不得超过 20MB”。字面不一样，但检索系统仍要把它们对齐。这正是 BGE 这类 embedding 模型存在的原因。

---

## 核心机制与推导

BGE 的核心机制可以拆成两层。

第一层是 InfoNCE 对比损失。它约束查询 $q$ 靠近正样本 $p^+$，远离负样本 $p$：

$$
\mathcal{L}=-\log\frac{e^{s(q,p^+)/\tau}}{\sum_{p}e^{s(q,p)/\tau}}
$$

这里：

- $s(q,p)$ 是相似度，通常可看成点积或余弦相似度。
- $\tau$ 是温度参数，白话讲就是“拉开或压缩分数差距的旋钮”。
- 分母里包含很多负样本，既有显式负样本，也有 batch 内负样本。

它的直觉并不复杂：同一批里，正确段落分数要最高，而且要明显高于其他段落。这样训练久了，向量空间就不再只是“语言相近”，而是“检索任务上相关”。

第二层是 BGE-M3 的三路打分。它并行计算：

- $s_{dense}$：dense 语义分，负责同义改写、语义近似。
- $s_{lex}$：lexical 稀疏分，负责关键词硬匹配，效果类似神经化 BM25。
- $s_{mul}$：multi-vector 分，负责更细粒度的 token 交互，风格接近 ColBERT。

最终混合分数是：

$$
s_{rank}=w_1 s_{dense}+w_2 s_{lex}+w_3 s_{mul}
$$

这可以理解成“三个评分器投票”。dense 说“语义上像”，sparse 说“术语命中”，multi-vector 说“局部片段对得上”。最后按权重合并。工程上常见做法不是全库直接跑三路，而是先用 dense 或 sparse 粗召回，再对 top-k 用 multi-vector 或 reranker 精排，因为后两者更贵。

从检索角色看，三路 head 的分工如下：

| Head | 表示形式 | 擅长问题 | 代价 |
|---|---|---|---|
| Dense | 单个稠密向量 | 同义词、改写、语义泛化 | 低 |
| Sparse | 词项权重 | 专有名词、版本号、错误码 | 低到中 |
| Multi-vector | 多个 token 向量 | 长文局部精确匹配 | 高 |

---

## 代码实现

下面先给一个最小可运行的玩具代码，只演示两个核心思想：InfoNCE 和混合打分。它不是训练完整 BGE，而是把公式落成可执行代码。

```python
import math

def infonce(pos_score, neg_scores, temp=0.05):
    logits = [pos_score / temp] + [s / temp for s in neg_scores]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    prob = exps[0] / sum(exps)
    return -math.log(prob)

def hybrid_score(dense_score, sparse_score, colbert_score, w1=0.5, w2=0.2, w3=0.3):
    return w1 * dense_score + w2 * sparse_score + w3 * colbert_score

# 玩具例子：一个 query，对两个候选段落
q_to_pos = 0.82
q_to_neg = [0.21, 0.18, 0.05]

loss = infonce(q_to_pos, q_to_neg, temp=0.05)
assert loss < 1e-3  # 正样本显著高于负样本时，loss 应很小

# 混合检索得分
score_a = hybrid_score(0.86, 0.30, 0.72)  # 真相关文档
score_b = hybrid_score(0.70, 0.10, 0.20)  # 干扰文档

assert score_a > score_b
print(round(loss, 6), round(score_a, 4), round(score_b, 4))
```

如果把它对应到 BGE 的真实训练流程，可以写成下面这个新手版伪代码：

```python
# 训练端
encoder = RetroMAEEncoder()
encoder.pretrain(corpus="中文通用语料")

for query, pos, neg in triples:
    q_emb = encoder.encode_query(query, instruction="为这个句子生成表示以用于检索相关文章：")
    p_emb = encoder.encode_passage(pos)
    n_emb = encoder.encode_passage(neg)
    loss = infonce(sim(q_emb, p_emb), [sim(q_emb, n_emb)], temp=0.01)
    loss.backward()
```

部署端如果是 BGE-M3，可以理解成：

```python
# 检索端
dense_candidates = dense_search(query, topk=100)
sparse_candidates = sparse_search(query, topk=100)
candidates = merge(dense_candidates, sparse_candidates)

for doc in candidates:
    dense_score = calc_dense(query, doc)
    sparse_score = calc_sparse(query, doc)
    colbert_score = calc_colbert(query, doc)
    doc.score = 0.5 * dense_score + 0.2 * sparse_score + 0.3 * colbert_score

results = sorted(candidates, key=lambda x: x.score, reverse=True)[:10]
```

真实工程例子可以放到 Milvus。常见做法是：

1. 用 `bge-large-zh` 或 `bge-m3` 编码文档。
2. 把 dense 向量写入向量库。
3. 如果是 `bge-m3`，同时存 sparse 表示。
4. 查询时先做 dense/sparse 召回。
5. 对 top-k 结果再做 hybrid ranking，必要时加 reranker。

公开案例里，Milvus 文档展示了标准检索 `Pass@5 = 80.92%`，混合检索后到 `84.69%`。这不是“BGE-M3 对所有业务固定提升 5 个点”，而是说明在包含关键词约束的 RAG 场景里，混合召回经常能带来约 5% 左右的相对提升。

---

## 工程权衡与常见坑

最常见的坑是把“预训练模型”误当成“可直接检索模型”。这是错误的。RetroMAE 学的是重建，不是 query-passage 对齐。直接拿去搜，常见现象是相似度分数挤在高区间，排序区分度不够。

第二个坑是把绝对相似度阈值当真。BGE 官方模型卡专门提醒过，早期版本存在相似度分布偏高的问题，分数大于 0.5 不代表真的相似。检索看重的是排序，不是某个固定阈值。

第三个坑是 short query 不加 instruction。尤其是“两个词的短问题”去搜“几百字段落”时，不加 instruction 往往掉召回。v1.5 改善了这个问题，但不是说 instruction 永远没意义。

第四个坑是 BGE-M3 全流程直接上三路精排。dense、sparse、ColBERT 同时跑当然更强，但成本高，尤其 multi-vector 和 reranker 都会放大延迟。资源有限时，正确顺序是先粗排再精排。

风险对照可以压成一张表：

| 问题 | 现象 | 处理办法 |
|---|---|---|
| 只做 RetroMAE 预训练 | 能编码，不能稳定检索 | 再做 contrastive fine-tune |
| 相似度过于集中 | 分数普遍偏高，阈值失真 | 用 v1.5，按业务数据重设阈值 |
| 短查询召回差 | query 太短，语义不完整 | 给 query 加 instruction |
| 混合检索太慢 | GPU/CPU 压力大，延迟升高 | dense/sparse 粗排，ColBERT 或 reranker 只跑 top-k |
| 长文档切分后丢上下文 | 片段命中但答案不完整 | 优化 chunk 策略，必要时加 contextual retrieval |

---

## 替代方案与适用边界

如果你的目标只是“中文知识库搜索，成本尽量低”，`bge-large-zh` 或 `bge-large-zh-v1.5` 已经够用。它的好处是部署简单，只维护一套 dense 向量索引，推理路径最短。

如果你的场景有很多硬关键词，比如报错码、药品名、接口名、法律条文编号，BGE-M3 更合适。因为这类查询不只是“语义像”，还要求“词面必须中”。dense 往往会把“相近概念”也召回，而 sparse 能把术语锚住。

如果资源再紧一点，可以只做 dense + lexical 两路，不上 ColBERT 风格多向量。这样大部分收益已经能拿到，代价明显低于完整三路。

可以直接比较：

| 方案 | 性能特点 | 成本 | 适用边界 |
|---|---|---|---|
| 纯 dense | 语义泛化强，术语场景一般 | 低 | FAQ、通用中文检索 |
| BGE-large-zh / v1.5 | 中文 dense 表现强，接入简单 | 低到中 | 中文主场景，先求稳上线 |
| BGE-M3 dense+sparse | 兼顾语义和关键词 | 中 | RAG、术语密集库、混合召回 |
| BGE-M3 完整 hybrid | 召回和精度更强 | 高 | 高价值检索、长文与复杂查询 |

还有一个边界要注意：BGE-large-zh 是中文专长模型，不应默认拿去做跨语种统一空间检索；BGE-M3 虽支持 100+ 语言，但多语种统一效果仍应以你的业务数据验证，不能只看公开榜单。

---

## 参考资料

| 资料 | URL | 内容定位 |
|---|---|---|
| BAAI `bge-large-zh` README | https://huggingface.co/BAAI/bge-large-zh/blob/1b543b301eb63dd32914b56d939db2a972df15d5/README.md | 官方训练流程，说明 `RetroMAE + contrastive learning`，并给出中文模型用法 |
| BAAI `bge-large-zh` 模型页 | https://huggingface.co/BAAI/bge-large-zh | 官方 C-MTEB 分数，含 `bge-large-zh` 与 `multilingual-e5-large` 对比 |
| BGE-M3 官方文档 | https://bge-model.com/bge/bge_m3.html | 解释 dense、sparse、multi-vector 三路表示与混合打分公式 |
| NVIDIA BGE-M3 文档 | https://docs.api.nvidia.com/nim/reference/baai-bge-m3 | 给出 `hybrid retrieval + re-ranking` 的部署建议与模型规格 |
| 智源 MTP 数据说明 | https://hub.baai.ac.cn/view/30780 | 公开说明 BGE 训练数据来自 3 亿中英文文本对，其中中文约 1 亿 |
| 智源 BGE 发布说明 | https://hub.baai.ac.cn/view/28429 | BGE 中文能力、C-MTEB 表现与整体定位 |
| Milvus BGE-M3 文档 | https://blog.milvus.io/docs/v2.5.x/embed-with-bgm-m3.md | BGE-M3 在 Milvus 中的接入方式 |
| Milvus 上下文检索教程 | https://milvus.io/docs/pt/contextual_retrieval_with_milvus.md | 公开示例中标准检索与混合检索的 `Pass@5` 对比 |
| RetroMAE 论文 | https://aclanthology.org/2022.emnlp-main.35/ | 解释为什么“重建式预训练”能改善检索编码器 |

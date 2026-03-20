## 核心结论

动态稀疏注意力的核心不是“把注意力做稀”，而是“先预测哪些位置值得算，再只算这些位置”。这里的“动态”指稀疏模式会随输入变化，不再像 Longformer 那样提前写死窗口或全局 token 规则。

固定稀疏模式的问题是：它默认所有样本都遵守同一套关注结构。但真实输入里，代码、长文档、对话、图文混合序列的注意力分布差异很大。动态方法因此引入一个轻量评分网络，先为每个 query 估计 key 的重要性，再保留 top-k。top-k 的白话意思是“只留分数最高的前 k 个候选”。

一类典型路线是低秩近似。低秩的白话意思是“先把高维向量压到更小的维度里做粗筛”，常见形式是
$$
Q' = QW_q,\quad K' = KW_k,\quad A'=\frac{Q'K'^\top}{\sqrt r},\quad r\ll d
$$
先用 $A'$ 找到重要位置，再回到原始维度上的 $K,V$ 做精确 attention。这样把“全量比较”拆成“便宜粗筛 + 昂贵精算”两步。

因此，动态稀疏注意力解决的是“如何让稀疏模式跟着输入走”，而低秩投影、可微 top-k、LSH 哈希只是不同实现手段。它们的共同目标都是避免标准 attention 的 $O(N^2)$ 全连接代价。

---

## 问题定义与边界

标准自注意力对长度为 $N$ 的序列，需要计算 $N\times N$ 个 query-key 相关性，复杂度通常记为 $O(N^2)$。FlashAttention 优化了访存和 kernel 调度，但并没有改变“每对 token 仍要比较一次”这个事实。所以当上下文拉到 32k、64k 甚至更长时，二次复杂度仍然是主要瓶颈。

动态稀疏注意力要解决两个问题：

1. 不能先完整算出 $QK^\top$ 再做 top-k，否则筛选本身已经把钱花完了。
2. 训练时不能只靠硬 top-k，因为硬筛选不可微，不利于端到端优化。不可微的白话意思是“梯度过不去，评分网络学不会该选谁”。

下面这张表先把边界说明白：

| 路线 | 估算阶段 | 精确阶段 | 典型复杂度 | 是否输入自适应 | 训练友好性 |
| --- | --- | --- | --- | --- | --- |
| 全 attention | 无 | 全量计算 | $O(N^2)$ | 是 | 最简单 |
| 先全量打分再 top-k | 全量 $QK^\top$ | 稀疏精算 | 仍接近 $O(N^2)$ | 是 | 可以 |
| 低秩估算 + top-k | $Q'K'^\top$ | 仅对选中位置精算 | 常写作 $O(Nr)+O(Nk^2)$ 或按实现记作近线性加稀疏项 | 是 | 可以 |
| Reformer LSH | 哈希分桶 | 桶内 attention | $O(N\log N)$ | 部分是 | 可以 |

玩具例子最容易看清楚。假设只有 4 个 token，对某个 query 的粗评分是 `[0.2, 1.3, 0.7, 0.1]`，取 top-2 后只保留第 2、第 3 个 key。原来要看 4 个 key，现在只看 2 个。若 4 个 query 都这样做，全量 16 次比较可以降到约 8 次。这不是理论魔术，而是明确地减少了“真正进入 softmax 和 value 聚合的候选数”。

边界也很明确：如果任务非常依赖密集、平滑、全局的弱关联，$k$ 过小会丢信息；如果估算网络本身不够准，稀疏会把重要 key 错删；如果硬件不擅长稀疏 gather，理论节省未必变成真实加速。

---

## 核心机制与推导

动态稀疏注意力通常分成两阶段。

第一阶段是粗筛。以 LoRA-Sparse 为代表，先把 $Q,K$ 投影到低秩空间：
$$
Q'=QW_q,\quad K'=KW_k,\quad W_q,W_k\in\mathbb{R}^{d\times r},\quad r\ll d
$$
然后计算近似分数：
$$
S'=\frac{Q'K'^\top}{\sqrt r}
$$
再对每一行取 top-k，得到候选索引集合 $\mathcal{I}_i$。这一阶段的目的不是给出最终注意力，而是尽量便宜地回答“哪些 key 值得进入下一轮”。

第二阶段是精算。对第 $i$ 个 query，只从原始 $K,V$ 中取出被选中的位置：
$$
K_i^\star = \mathrm{gather}(K,\mathcal{I}_i),\quad V_i^\star = \mathrm{gather}(V,\mathcal{I}_i)
$$
再计算真正的 scaled dot-product attention：
$$
\alpha_i=\mathrm{softmax}\left(\frac{q_i(K_i^\star)^\top}{\sqrt d}\right),\quad
o_i=\alpha_iV_i^\star
$$

如果希望 top-k 也可训练，就要把“离散选中”改造成“可微近似选中”。SPARSEK 的思路可以理解为：不直接输出 0/1 的硬 mask，而是输出一个受约束的软 mask
$$
M=\mathrm{SPARSEK}(S',k)
$$
其中 $M$ 的元素落在 $[0,1]$，并控制总保留量接近 $k$。白话解释是：训练时先让模型“软选”，这样梯度还能传；推理时再把它收紧成真正的 top-k。

这里的推导重点不是某个符号，而是流水线：

1. 用便宜特征估分。
2. 用 top-k 或可微 top-k 缩小候选集。
3. 只在小候选集上做原始高精度 attention。

这比固定稀疏模式强的地方在于，模式不是人工定义的，而是由输入和评分网络共同决定。比如同样是一个逗号，出现在法律文本里可能只关注局部上下文，出现在多轮对话里可能要回看很远的说话人状态。

真实工程例子是 NSA。它没有只做“单一路径 top-k”，而是把长上下文拆成三条并行分支：compression、selection、sliding window。compression 负责粗粒度全局感知，selection 负责动态 top-k 选重要块，sliding window 负责局部精度。ACL 2025 的结果显示，在 64k 序列上，它相对 Full Attention 的 decoding、forward、backward 分别达到约 11.6x、9.0x、6.0x 速度提升，同时任务效果不掉。这说明动态稀疏真正可用时，往往不是单个算法点子，而是“选择机制 + 层级结构 + 硬件对齐”一起成立。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它模拟“低秩打分 + top-k 选择 + 精确 attention”的完整数据流，不依赖第三方库。

```python
import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def transpose(m):
    return list(map(list, zip(*m)))

def matmul(a, b):
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def topk_indices(xs, k):
    return [i for i, _ in sorted(enumerate(xs), key=lambda x: x[1], reverse=True)[:k]]

def gather(rows, indices):
    return [rows[i] for i in indices]

def sparse_attention(Q, K, V, Wq, Wk, k):
    # 1) 低秩粗评分
    Q_low = matmul(Q, Wq)
    K_low = matmul(K, Wk)
    scale_low = math.sqrt(len(Q_low[0]))
    approx_scores = [
        [dot(q, kk) / scale_low for kk in K_low]
        for q in Q_low
    ]

    # 2) 每个 query 选 top-k
    selected = [topk_indices(row, k) for row in approx_scores]

    # 3) 在原始维度上做精确 attention
    outputs = []
    for qi, q in enumerate(Q):
        idx = selected[qi]
        K_sel = gather(K, idx)
        V_sel = gather(V, idx)

        scale = math.sqrt(len(q))
        exact_scores = [dot(q, kk) / scale for kk in K_sel]
        weights = softmax(exact_scores)

        out = [0.0 for _ in range(len(V[0]))]
        for w, v in zip(weights, V_sel):
            for j in range(len(out)):
                out[j] += w * v[j]
        outputs.append(out)

    return approx_scores, selected, outputs

Q = [[1.0, 0.0], [0.0, 1.0]]
K = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.9]]
V = [[10.0], [8.0], [5.0], [4.0]]

# 把 2 维压到 1 维，模拟低秩估算
Wq = [[1.0], [1.0]]
Wk = [[1.0], [1.0]]

approx_scores, selected, outputs = sparse_attention(Q, K, V, Wq, Wk, k=2)

assert len(selected) == 2
assert all(len(x) == 2 for x in selected)
assert selected[0] == [1, 3] or selected[0] == [3, 1]
assert selected[1] == [3, 2] or selected[1] == [2, 3]
assert len(outputs) == 2
assert outputs[0][0] > outputs[1][0]
```

这段代码故意写得简单，目的是把顺序讲清楚：

1. `Wq/Wk` 先把 `Q/K` 压成低维表示。
2. `approx_scores` 只负责筛人，不负责最终输出。
3. `selected` 保存每个 query 的候选 key 下标。
4. 真正输出 `outputs` 时，仍然回到原始 `K/V` 计算精确权重。

如果换成 PyTorch，核心接口通常还是这四步，只是会把 `topk`、`gather`、`softmax` 和 batch/head 维度并行化。

---

## 工程权衡与常见坑

动态稀疏注意力最常见的误区是：只盯理论复杂度，不看估算阶段和硬件代价。

| 开销位置 | 主要问题 | 典型坑 |
| --- | --- | --- |
| 估算阶段 | 粗筛本身也可能很贵 | 仍然算近似全矩阵，筛选比精算还慢 |
| 精确阶段 | 稀疏 gather 和重排不连续 | kernel 启动多、访存碎片化 |
| 硬件配合 | GPU/NPU 喜欢规则访问 | 稀疏模式太动态，无法充分利用 Tensor Core |

一个典型反例来自移动 SoC 场景。研究指出，若动态稀疏实现仍需在 CPU/GPU 上为全部 token 做重要性估算，那么当稀疏率很高时，估算阶段会吞掉超过 60% 的注意力开销。结果是：你明明删掉了 80% token，端到端加速却很有限。这说明“筛选器的成本”必须比“被筛掉的计算”更便宜，否则稀疏没有意义。

另一个坑是 $k$ 的设置。$k$ 太小，召回不足，重要信息直接丢失；$k$ 太大，精度回来了，但加速没了。工程上常把 $k$ 做成分层、分头或分阶段可调，而不是全模型一个固定值。

再一个坑是训练和推理不一致。训练用软 mask，推理用硬 top-k，如果两者差距太大，会出现训练看起来稳定、上线后退化的情况。所以不少方法会在训练后期逐步加硬，或者额外做蒸馏、排序模仿损失，让估算分数的排序更贴近真实 attention 排序。

---

## 替代方案与适用边界

动态 top-k 不是唯一答案。它适合“注意力位置因输入而变，但最终输出仍要求高精度”的任务，比如开放域问答、长文档摘要、代码补全、多模态对齐。

Reformer 走的是另一条路。LSH 的白话意思是“把相似向量尽量哈希到同一个桶里”，这样只在桶内做 attention，把复杂度降到 $O(N\log N)$。它适合相似性结构较稳定、允许近似分桶误差的场景。但如果哈希把本该相遇的 token 分到不同桶，召回会受影响。

再对比一次：

| 方法 | 核心思想 | 典型复杂度 | 优点 | 局限 |
| --- | --- | --- | --- | --- |
| 动态 top-k | 评分后选最重要位置 | 近线性估算 + 稀疏精算 | 输入自适应，精度控制直接 | 依赖评分器质量 |
| LoRA-Sparse | 低秩空间先粗筛，再原维精算 | 常写作 $O(Nr)+O(Nk^2)$ | 估算成本低，易接预训练模型 | 低秩近似可能漏排关键项 |
| Reformer LSH | 哈希分桶后桶内 attention | $O(N\log N)$ | 适合超长序列 | 哈希误分桶会损失召回 |

一个直观判断标准是：

如果你能提前假设“相关 token 大概率彼此相似，可被哈希聚类”，Reformer 一类方案值得考虑；如果你无法提前知道注意力结构，且希望模型按样本实时决定关注点，动态 top-k 更稳妥；如果你的主要瓶颈在“粗筛太贵”，低秩近似通常是第一步。

---

## 参考资料

- SPARSEK Attention / *Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers*：可微 top-k 与评分网络的动态稀疏路线。<https://arxiv.org/abs/2406.16747>
- LoRA-Sparse / *Low-Rank Approximation for Sparse Attention in Multi-Modal LLMs*：低秩投影先估分，再对 top-k 做精确 attention。<https://openaccess.thecvf.com/content/CVPR2024/html/Song_Low-Rank_Approximation_for_Sparse_Attention_in_Multi-Modal_LLMs_CVPR_2024_paper.html>
- NSA / *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*：分层动态稀疏与硬件对齐实现。<https://aclanthology.org/2025.acl-long.1126/>
- Reformer / *Reformer: The Efficient Transformer*：LSH attention 将复杂度降到 $O(N\log N)$。<https://arxiv.org/abs/2001.04451>
- *Dynamic Sparse Attention on Mobile SoCs*：移动端场景下估算阶段可能成为主要瓶颈。<https://arxiv.org/abs/2508.16703>

## 核心结论

Multi-Head MoE 可以理解为“把 MoE 的稀疏选择机制，从 FFN 层推进到 Attention 层”。MoE 指混合专家模型，本质是先做路由，再只激活少量子模块。这里的“专家”不再是 FFN 专家网络，而是不同的 attention head。attention head 指注意力头，也就是多头注意力里一组独立的 $Q,K,V$ 投影和注意力计算单元。

传统多头注意力里，每个 token 都要经过全部 $H$ 个头；Multi-Head MoE 里，每个 token 只激活其中 $k$ 个头，通常 $k \ll H$。因此，核心收益非常直接：

$$
\text{compute\_ratio}=\frac{k}{H}
$$

这个比例表示“每个 token 实际使用的头数，占总头数的多少”。如果总共有 16 个头，但每个 token 只激活 4 个头，那么 Attention 主体计算大致只做了原来的 $25\%$。

关键点不只是“少算了”，而是“少算的同时保留了多头参数独立性”。也就是说，模型仍然可以拥有很多头，让不同头学到不同功能，例如语法依赖、实体对齐、位置模式、局部短语关系；只是每个 token 不需要把所有头都跑一遍。这样做的目标不是删头，而是让头变成“按需执行”。

一个适合新手理解的玩具例子是翻译任务中的某个 token。假设一个英文句子里出现专有名词，router 会把更擅长实体识别、跨语言对齐的头选出来，而不激活那些更擅长标点、局部语法的头。结果是：参数容量还在，但执行成本下降了。

---

## 问题定义与边界

问题先定义清楚：为什么要把 MoE 放进 Attention，而不是继续只在 FFN 上做稀疏化？

原因是 Attention 本身很贵，尤其在长上下文下更明显。Transformer 指一种基于自注意力的序列建模结构。对于长度为 $T$ 的序列、总头数为 $H$、每头维度为 $d$ 的多头注意力，若忽略常数项，计算量可以粗略写成和 $H$ 成正比。头数越多，表达能力通常越强，但计算和显存也同步增加。

Multi-Head MoE 的边界是：它减少的是“实际执行的 head 数量”，不是把 head 总数物理删掉。模型参数中仍保留全部头，router 只是决定某个 token 当前该调用哪些头。因此它适合下面这种需求：

| 模式 | 每 token 激活 head | 近似复杂度 |
|---|---:|---|
| Dense MHA | 16 | $O(16 \cdot d \cdot T)$ |
| Multi-Head MoE | 4 | $O(4 \cdot d \cdot T)+O(16 \cdot T)$ |

表里多出来的 $O(H \cdot T)$ 是路由开销，也就是对每个 token 给每个头打分，再做 top-k 选择。这个开销通常是线性的，而真正重的注意力主体计算被压到了 $k$ 个头上。

数值上看更直观。设：

- 序列长度 $T=512$
- 总头数 $H=16$
- 激活头数 $k=4$

那么每个 token 只访问 4 个头，占全部头的 $25\%$。如果路由器足够轻量，理论上 Attention 的主要执行量可接近缩减到原来的四分之一。这里要注意“接近”两个字，因为真实系统里还受 kernel 启动、访存、mask 构造、top-k 实现方式影响，未必线性等于 4 倍提速。

边界也要说清楚：

1. 它不是免费午餐。router 自己要训练，还可能不稳定。
2. 它不保证 wall-clock 时间一定按 $k/H$ 等比例下降。wall-clock 指真实运行时间，受硬件和实现细节影响。
3. 它更适合“头很多、上下文长、希望保持参数容量”的场景；如果本来头就很少，或者实现环境不支持高效稀疏执行，收益可能不明显。

---

## 核心机制与推导

核心机制分三步：打分、选头、只执行被选中的头。

设输入 token 表示为 $x_t$，router 是一个小网络，通常是一层线性层或更轻的打分模块。对于每个 token $t$，router 计算它对每个 head 的得分：

$$
s_{t,h} = \text{Router}(x_t)_h,\quad h=1,\dots,H
$$

然后在 head 维度上取 top-k，得到被激活的头集合 $\mathcal{A}_t$。只有这些头会实际参与注意力计算，其余头的输出视为 0，或直接不执行。

如果再加一个归一化权重，可以写成：

$$
g_{t,h}=
\begin{cases}
\frac{\exp(s_{t,h})}{\sum\limits_{j\in \mathcal{A}_t}\exp(s_{t,j})}, & h \in \mathcal{A}_t \\
0, & h \notin \mathcal{A}_t
\end{cases}
$$

其中 $g_{t,h}$ 是门控权重，也就是该 token 对某个头分配了多大权重。这样每个 token 的输出可以写成：

$$
y_t = \sum_{h=1}^{H} g_{t,h}\,\text{Attn}_h(x)_t
$$

因为大多数 $g_{t,h}=0$，所以求和实际上只发生在 top-k 个头上。这和经典 MoE 的思想完全一致：先路由，再稀疏执行。

复杂度可以写成：

$$
\text{Complexity}=O(k \cdot d \cdot T)+O(H \cdot T)
$$

其中：

- 第一项是被选中头的注意力主体计算；
- 第二项是路由分数、softmax 或 top-k 的成本。

与 dense 版的

$$
O(H \cdot d \cdot T)
$$

相比，只要 $k \ll H$，而且 $d$ 不太小，那么节省就主要来自第一项从 $H$ 变成了 $k$。

这里有个很重要的认识：Multi-Head MoE 不是把“多头多样性”破坏掉，而是把“多样性”与“执行量”拆开。头仍然可以专门化，只是不同 token 看到的头集合不同。举个玩具例子：

一句话是“Apple released a new chip in 2025”。  
对于 token “Apple”，router 可能激活“实体头”“跨词依赖头”“大小写模式头”；  
对于 token “released”，router 可能更倾向“动词关系头”“时态模式头”；  
对于 token “2025”，router 可能偏向“数字模式头”“时间表达头”。

这比“所有 token 都跑所有头”更接近条件计算的直觉：不同 token 的结构需求不同，不必强行共享同一组计算路径。

真实工程例子则是长上下文语言建模。假设团队想把模型上下文扩到 128K 甚至更长，同时还想增加头数来容纳更丰富的模式。如果坚持 dense MHA，头数翻倍意味着 Attention 成本和 KV cache 压力同步上升。KV cache 指推理时缓存历史 token 的 key/value 张量，用来避免重复计算。Multi-Head MoE 的思路是保留更多头的参数容量，但让每个 token 只调用其中一部分，从而把额外预算用于 specialization，而不是把所有头都无差别跑满。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不追求高性能，只展示路由、top-k、稀疏 head 聚合这三个关键动作。为了便于运行，代码用 `numpy` 实现，且把单个 token 的“head 输出”简化为线性变换。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def topk_mask(scores, k):
    # scores: [T, H]
    T, H = scores.shape
    idx = np.argsort(scores, axis=1)[:, -k:]
    mask = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        mask[t, idx[t]] = 1.0
    return mask

def multi_head_moe(x, router_w, head_w, k_heads):
    """
    x: [T, D]
    router_w: [D, H]
    head_w: [H, D, D]
    return:
      y: [T, D]
      gate: [T, H]
      mask: [T, H]
    """
    scores = x @ router_w                 # [T, H]
    mask = topk_mask(scores, k_heads)     # [T, H]

    gated_scores = np.where(mask > 0, scores, -1e9)
    gate = softmax(gated_scores, axis=1)  # only top-k heads get non-zero weight

    T, D = x.shape
    H = head_w.shape[0]
    y = np.zeros((T, D), dtype=np.float32)

    for h in range(H):
        head_out = x @ head_w[h]          # toy "head output"
        y += gate[:, h:h+1] * head_out

    return y, gate, mask

# toy example
np.random.seed(0)
T, D, H, K = 3, 4, 6, 2
x = np.array([
    [2.0, 0.0, 0.0, 0.0],   # token A
    [0.0, 2.0, 0.0, 0.0],   # token B
    [0.0, 0.0, 2.0, 0.0],   # token C
], dtype=np.float32)

router_w = np.array([
    [3, 0, 0, 2, 0, 0],
    [0, 3, 0, 0, 2, 0],
    [0, 0, 3, 0, 0, 2],
    [0, 0, 0, 0, 0, 1],
], dtype=np.float32)

head_w = np.stack([np.eye(D, dtype=np.float32) * (h + 1) for h in range(H)], axis=0)

y, gate, mask = multi_head_moe(x, router_w, head_w, K)

# each token activates exactly K heads
assert np.all(mask.sum(axis=1) == K)

# gate is a valid probability distribution on selected heads
assert np.allclose(gate.sum(axis=1), 1.0)

# inactive heads have zero gate
assert np.all(gate[mask == 0] < 1e-6)

print("mask=\n", mask)
print("gate=\n", np.round(gate, 4))
print("output=\n", np.round(y, 4))
```

这段代码说明了最核心的接口：

1. `scores = x @ router_w`：给每个 token 的每个 head 打分。
2. `topk_mask`：只保留前 $k$ 个 head。
3. `softmax`：把被选中 head 的分数归一化成门控权重。
4. 仅对这些 head 的输出加权求和。

如果放进真实 Transformer，结构上通常是：

```python
def sparse_multi_head_attention(x):
    scores = router(x)                 # [B, T, H]
    topk_idx = topk(scores, k)         # select active heads per token
    gate = sparse_softmax(scores, topk_idx)

    outputs = []
    for h in range(H):
        if head_h_is_selected_for_any_token:
            q_h, k_h, v_h = proj_h(x)
            o_h = scaled_dot_product_attention(q_h, k_h, v_h)
        else:
            o_h = 0
        outputs.append(gate_h * o_h)

    return merge(outputs)
```

工程上真正难的不是数学，而是实现细节：

- 如果只是先算出全部头，再把不需要的头乘成 0，那并没有省掉真正的计算。
- 真正有效的实现必须在 kernel 层或批处理组织层面，让未激活 head 根本不执行，或者只对被选中的 token-head 对进行打包计算。
- 在训练中，还要处理 padding、causal mask、KV cache、mixed precision 和并行通信。

一个真实工程例子是长上下文预训练。假设模型原本有 16 个头，现在想扩到 32 个头来提高 specialization，但单卡显存和训练吞吐不够。Dense 做法会让 Attention 路径整体变重；Multi-Head MoE 做法则是保留 32 个头的参数，但每个 token 只选 6 到 8 个头。这样模型仍有更大的头空间可分工，同时把真正执行的计算控制在接近原预算附近。

---

## 工程权衡与常见坑

最常见的问题是 router collapse。collapse 指路由塌缩，也就是 router 总是偏向少数几个头，导致其余头几乎永远不被用到。这样模型表面上有很多头，实际上只有少数头在工作，既浪费参数，也破坏 specialization。

一个简单观察指标是平均激活分布。设某批数据里 head $h$ 被激活的平均概率为 $p_h$，如果所有头都均匀工作，那么 $p_h$ 应该接近 $1/H$。可以用熵来度量是否过于集中：

$$
\mathcal{H}(p) = -\sum_{h=1}^{H} p_h \log p_h
$$

熵越低，说明路由越集中；熵越高，说明使用越均匀。也可以用变异系数 CoV 衡量各头负载是否失衡。

常见问题与规避手段如下：

| 问题 | 现象 | 规避手段 |
|---|---|---|
| 专家倾斜 | 少数头长期过热 | 加 load balancing loss，增大 `top-k` |
| Router collapse | 大部分头几乎不被选中 | 对平均路由分布加 KL/entropy 正则 |
| 头失活 | 某些头训练后长期无梯度 | head-specific dropout、随机温度 |
| 路由抖动 | 相邻 step 选头剧烈变化 | 温度退火、router 低维化、EMA 统计 |
| 实际不提速 | 理论稀疏但 kernel 仍 dense | 做真正的稀疏执行或 token-head 打包 |

这里的 load balancing loss 可以理解为“别让 router 把所有学生都派给同一个老师”。例如，让一个 batch 内所有 head 的平均使用率尽量接近均匀分布：

$$
\mathcal{L}_{balance} = \mathrm{KL}(p \,\|\, u)
$$

其中 $u$ 是均匀分布。KL 指 Kullback-Leibler 散度，用来衡量两个分布差异。白话说，就是惩罚“实际路由分布”和“理想均匀分布”差太远。

新手容易踩的第二个坑是把“头选择”和“头表达能力”混为一谈。若 router 太强、主干太弱，模型可能学会“靠路由投机”，而不是让各头真正专门化。实践上通常会：

- 把 router 设计得比主干轻，避免路由器过拟合。
- 使用 top-k > 1，而不是 top-1，降低离散选择过硬带来的不稳定。
- 给 head 增加独立正则，让不同头有动力学出差异。

第三个坑是硬件不友好。GPU 往往更擅长规则、密集的大矩阵运算，不擅长碎片化稀疏执行。于是可能出现“理论 FLOPs 降了，但真实吞吐没升”。所以在工程里，Multi-Head MoE 是否值得做，不只看论文公式，还要看你的训练框架、内核支持、序列长度、batch 组织方式。

---

## 替代方案与适用边界

如果目标只是让模型更稳定、更容易训练，Dense MHA 依然是最简单的基线。Dense MHA 指标准多头注意力，即所有 token 跑所有头。它没有路由不稳定问题，调参成本最低。

如果目标是“想减少 active head，但不想做太硬的离散选择”，可以考虑 MoH。MoH 指 Mixture-of-Head，一类更偏软选择的方案，通常会保留一些 shared head，再对其他头做加权组合。它的优点是训练往往比硬 top-k 更平滑，适合从已有模型微调切入。

如果目标是长上下文和 KV cache 压力，而不执着于“头维度稀疏”，可以看 MoSA。MoSA 指 Mixture of Sparse Attention，它更关注 token 选择或稀疏注意力模式，而不是单纯做 head 选择。对于超长文本，token 维度的稀疏化往往更直接影响缓存和注意力图大小。

对比如下：

| 方案 | Sparsity 位置 | 优点 | 适用场景 |
|---|---|---|---|
| Dense MHA | 无 | 简单、稳定、实现成熟 | 小模型、短上下文、先做基线 |
| MoH | head 选择或软组合 | 训练更平滑，可保留 shared head | 视觉模型、LLM 微调 |
| MoSA | token 选择 | 更关注长上下文与 KV cache | 超长文本、缓存受限 |
| Multi-Head MoE | head 选择 | 保留头多样性，路由开销相对小 | 长上下文、头数多、希望做条件执行 |

适用边界可以总结为三句话：

1. 如果你最关心训练稳定性，先用 Dense MHA。
2. 如果你最关心“保留很多头的参数容量，但不想全部执行”，Multi-Head MoE 很合适。
3. 如果瓶颈主要在超长上下文的 token 规模，而不是 head 数量，MoSA 一类方案可能更直接。

一个具体决策例子：

- 小团队做 7B 级语言模型微调，序列长度只有 4K，首选 Dense MHA 或 MoH，因为实现成本低。
- 团队做长上下文预训练，希望把头数从 16 扩到 32 甚至 64，但预算有限，此时 Multi-Head MoE 更有吸引力。
- 团队做 1M token 级上下文探索，KV cache 已成主瓶颈，则应优先研究 token 稀疏注意力，而不是只在 head 上做选择。

---

## 参考资料

- Mixture of Attention Heads, EMNLP 2022: https://aclanthology.org/2022.emnlp-main.278/
- Mixture of Sparse Attention, arXiv 2025 综述页: https://www.emergentmind.com/papers/2505.00315
- EmergentMind, Routing Collapse 主题综述: https://www.emergentmind.com/topics/routing-collapse
- EmergentMind, Mixture-of-Experts MoE Attention 主题综述: https://www.emergentmind.com/topics/mixture-of-experts-moe-attention
- Awesome-Efficient-MoE 工程资料集合: https://github.com/pprp/Awesome-Efficient-MoE

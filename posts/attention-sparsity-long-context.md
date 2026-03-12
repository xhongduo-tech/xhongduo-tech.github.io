## 核心结论

长上下文里的注意力不是“平均看全部”，而是“只盯住极少数位置”。注意力，白话说，就是每个 token 在决定自己输出时会参考哪些别的 token。对超长序列来说，绝大多数参考关系的权重接近零，因此真实有效的交互通常只有 $k \ll T$ 个，而不是全部 $T$ 个。

标准自注意力写作：

$$
Attention(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

如果按稠密方式计算，复杂度是 $O(T^2)$。但当每个 query 只真正依赖少数 token 时，可以把计算改成 $O(k\cdot T)$。这不是纯工程近似，而是有理论支撑的剪枝思路：若被截断的位置注意力都不超过 $\epsilon$，则常见误差界可写成

$$
\|\Delta o\|_\infty \le (n-k)\epsilon \|V\|_\infty
$$

它的含义很直接：被你丢掉的项越多，单项残余权重越大，误差就越大；反过来，只要尾部权重足够小，稀疏化就是可控的。

玩具例子：设 $T=1024$，每一行只保留最大的 $k=16$ 个注意力，其余 $1008$ 个位置都小于 $\epsilon=10^{-5}$，再假设 $\|V\|_\infty\approx 1$，那么输出误差上界约为

$$
(1024-16)\times 10^{-5}\approx 0.01008
$$

这说明 98.4% 的位置对结果几乎没有贡献。长上下文提速的核心，不是“把模型变小”，而是“承认大多数连接本来就没用”。

---

## 问题定义与边界

问题的核心不是“注意力能不能算”，而是“在百万级上下文下还能不能以可接受的成本算”。上下文，白话说，就是模型当前能同时看到的历史内容。序列长度 $T$ 增长后，注意力矩阵大小是 $T\times T$，计算、显存、KV cache 和内存带宽都会迅速膨胀。

但不能简单把远处 token 全删掉。原因是很多模型会出现 attention sink。attention sink，白话说，就是少数固定位置会吸走大量注意力，像“全局锚点”。它有时能稳定推理，有时也会制造假象：模型看起来一直“记得历史”，其实只是反复看那几个锚点。

因此，长上下文稀疏化要解决的是两个约束：

| 稀疏模式 | 作用 | 解决的问题 | 适用场景 |
|---|---|---|---|
| local window | 只看最近邻域 | 保留局部语义连续性 | 流式生成、代码补全 |
| global token / sink | 保留少数全局锚点 | 让远距离信息有汇聚点 | 长对话、长文摘要 |
| dilated pattern | 间隔采样更远位置 | 用少量连接扩大感受野 | 超长文档扫描 |
| random / sampled | 随机补充远处位置 | 防止固定模式漏掉关键信息 | 平坦注意力分布 |
| dynamic top-k | 按当前 query 动态选点 | 避免静态规则断开依赖 | 推理加速、检索增强 |

一个常见边界是流式对话。假设系统只保留最近 64 个 token，如果没有全局通道，模型会逐渐失去更早的信息；如果只保留早期 sink，又会丢掉最近上下文。更稳妥的做法是“最近窗口 + 少数全局 token + 动态补充”的混合结构，也就是 local+global 或带 gating 的稀疏注意力。

---

## 核心机制与推导

机制可以概括成一句话：先确保“必看”的局部和全局位置，再让每个 query 自己补选少数真正重要的远处 token。

这里有三个层次。

第一层是局部窗口。窗口，白话说，就是固定保留离当前 token 最近的若干位置，比如最近 64 个。它负责语言的连续性，因为很多依赖本来就是局部的。

第二层是全局锚点。全局锚点，白话说，就是无论当前 query 在哪里，都允许它访问的一小组特殊位置，比如开头 token、段落标记或历史 sink。它负责跨段连接。

第三层是动态 top-k。top-k，白话说，就是当前 query 根据分数从全体候选里临时挑出最值得看的 $k$ 个位置。这样不会被死板模板限制。

若对某一行注意力分数 $s_i$ 先做掩码，再只对保留项做 softmax，可写成：

$$
\tilde{s}_i=
\begin{cases}
s_i, & i\in \mathcal{M}(q) \\
-\infty, & i\notin \mathcal{M}(q)
\end{cases}
$$

$$
o=\mathrm{softmax}(\tilde{s})V
$$

其中 $\mathcal{M}(q)$ 可以由三部分并集构成：

$$
\mathcal{M}(q)=\text{Window}(q)\cup \text{Global}\cup \text{TopK}(q)
$$

这就是“先看近处，再看固定锚点，再补几个高分远点”的形式化表达。

玩具例子可以更直观。假设当前 query 是一句话末尾的“它”，最近窗口里包含“缓存”“显存”“推理”这些局部词；全局锚点里有文档开头的“Transformer 长上下文”；动态 top-k 又从很远处选中了“attention sink”“KV cache”。那么这个 query 不需要和全部 1024 个 token 交互，只要和这十几个位置交互，就足以恢复语义。

这里的关键不是“绝对精确找到真值”，而是“让被丢弃的尾部贡献足够小”。这也是误差界 $\|(n-k)\epsilon \|V\|_\infty\|$ 有意义的原因：它把“直觉上大部分位置不重要”变成了“可以估算的近似误差”。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现。它不是高性能 CUDA 内核，而是把 sliding window、sink token 和 dynamic top-k 组合成一条清晰的逻辑链。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def sparse_attention_single(q, K, V, window=4, sink_indices=None, top_k=2):
    T = len(K)
    assert T == len(V)
    assert 1 <= top_k <= T

    if sink_indices is None:
        sink_indices = [0]

    # 最近窗口
    local_start = max(0, T - window)
    local_idx = set(range(local_start, T))

    # 固定全局锚点
    sink_idx = set(i for i in sink_indices if 0 <= i < T)

    # 对全部位置打分
    scores = [dot(q, k) / math.sqrt(len(q)) for k in K]

    # 动态 top-k，从非必选位置里补点
    must_keep = local_idx | sink_idx
    ranked = sorted(range(T), key=lambda i: scores[i], reverse=True)
    dynamic_idx = []
    for i in ranked:
        if i not in must_keep:
            dynamic_idx.append(i)
        if len(dynamic_idx) >= top_k:
            break

    keep = must_keep | set(dynamic_idx)
    masked_scores = [scores[i] if i in keep else -1e9 for i in range(T)]
    weights = softmax(masked_scores)

    out_dim = len(V[0])
    output = [0.0] * out_dim
    for w, v in zip(weights, V):
        for j in range(out_dim):
            output[j] += w * v[j]

    return output, weights, keep

# 玩具例子：第 4 个位置是当前 query 更关心的远处 token
K = [
    [5.0, 0.0],   # sink
    [0.1, 0.1],
    [0.2, 0.0],
    [0.0, 0.2],
    [4.8, 0.1],   # important remote token
    [0.1, 0.3],
]
V = [
    [10.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 2.0],
    [8.0, 1.0],
    [0.0, 3.0],
]
q = [5.0, 0.0]

output, weights, keep = sparse_attention_single(q, K, V, window=2, sink_indices=[0], top_k=1)

assert 0 in keep          # sink 被保留
assert 4 in keep          # 动态 top-k 抓到关键远处 token
assert len(output) == 2
assert abs(sum(weights) - 1.0) < 1e-9
```

这段代码体现了一个新人也能把握的实现原则：先保留最近若干 token，再强制保留少数全局 token，然后根据当前 query 的分数补几个高价值位置。其余分数在 softmax 前设成极小值，相当于不参与计算。

真实工程例子里，这个逻辑会被进一步做成高效 kernel、块稀疏布局或选择性加载。Gemini 1.5 被广泛讨论的价值就在这里：它不是只靠更大显存硬算长上下文，而是把稀疏路由、长窗口和条件激活结合起来，让百万到千万 token 的处理变得可落地。即便具体实现细节未完全公开，方向很明确：只有少数连接和少数参数在每次推理中真正活跃。

---

## 工程权衡与常见坑

稀疏注意力的主要风险不是“速度不够快”，而是“连接断了但你没发现”。

第一类坑是 attention sink 过强。模型可能把大量 query 都压到少数全局 token 上，造成表面稳定、实际信息流单一。解决方法通常是前几层保留更强的 global 通道，或者给 sink 加门控，避免它垄断所有注意力。

第二类坑是静态稀疏模式过死。比如永远只看固定 block 或固定步长采样，会在长距离依赖出现时直接失联。解决方法是让每个 query 有动态补点能力，也就是 adaptive routing 或 gated sparse attention。

第三类坑是“理论稀疏”不等于“硬件更快”。如果稀疏模式不规则，GPU 上的访存和 kernel launch 开销可能抵消一部分收益。所以很多系统更偏好 block sparse。block，白话说，就是按小块而不是单个 token 组织稀疏结构，这样更适合硬件执行。

工程上常见设计原则如下：

| 设计原则 | 目的 | 典型做法 |
|---|---|---|
| union coverage | 防止 token 永远失联 | local + global + dynamic 并集 |
| block granularity | 提升硬件效率 | 用块级掩码替代逐点掩码 |
| hybrid patterns | 同时覆盖近距和远距 | window + sink + top-k |
| adaptive tuning | 按层或按头调整稀疏度 | 不同层设置不同 $k,\epsilon$ |
| selective loading | 降低内存和带宽压力 | 只加载候选 KV 块 |

真实工程例子是长文检索和超长对话。Gemini 1.5 Pro/Flash 的公开讨论常把焦点放在 needle-in-a-haystack 能力上，即在海量上下文中找到极少数关键片段。needle-in-a-haystack，白话说，就是“草堆里找针”式检索。要做到这一点，系统不能只保留局部窗口，也不能把所有远处内容平均处理，而必须借助稀疏路由、选择性激活和长程候选筛选来平衡成本与召回。

---

## 替代方案与适用边界

Top-k 稀疏不是唯一答案。它的优势是直观，问题是当注意力分布不尖锐时，硬截断可能不稳。尖锐，白话说，就是少数位置分数特别高；平坦则表示很多位置都差不多重要。

下面给出三类常见方案对比：

| 方案 | 是否有统计/理论保证 | 对尖峰分布适应性 | 对平坦分布适应性 | 适合场景 |
|---|---|---|---|---|
| Top-k | 常有误差上界分析 | 强 | 中 | 通用推理加速 |
| Sampling | 可做概率保证 | 中 | 强 | 需要覆盖长尾信息 |
| vAttention | 强调可验证近似保证 | 强 | 强 | 对稳定质量要求高的推理 |

vAttention 可以理解成“确定性保留关键位置，再随机补一些样本”的混合方法。对初学者来说，它像“高亮词必须看，最近词必须看，剩下再随机翻几页防漏”。这种方法的意义在于：即使注意力分布没有明显 top-k 峰值，仍然能给出 $(\epsilon,\delta)$ 风格的误差控制。

另一类替代思路不是直接稀疏化注意力，而是先做 token selection 或压缩。也就是先减少参与注意力的 token 数，再做后续计算。这更适合 retrieval-heavy 或 reasoning-heavy 场景，因为很多原始 token 在进入深层前就可以被压缩成少量代表。

适用边界也要说清楚。如果任务高度依赖精细对齐，比如逐 token 的精确复制、复杂代码编辑中的远距符号绑定，过于激进的稀疏可能伤质量。这种情况下更合理的策略通常是温和稀疏、分层稀疏，或者只在部分层启用稀疏。

---

## 参考资料

- Scalable Sparse Attention: [https://www.emergentmind.com/topics/scalable-sparse-attention](https://www.emergentmind.com/topics/scalable-sparse-attention)
- 清华综述《Efficient Inference》相关 PDF: [https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/1c678c23-69df-405b-992d-130fc6d5a4f5.pdf](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/1c678c23-69df-405b-992d-130fc6d5a4f5.pdf)
- Gemini 1.5 相关综述: [https://www.emergentmind.com/topics/gemini-1-5](https://www.emergentmind.com/topics/gemini-1-5)
- vAttention 相关综述: [https://www.emergentmind.com/topics/vattention](https://www.emergentmind.com/topics/vattention)
- Token Sparse Attention 相关综述: [https://www.emergentmind.com/topics/token-sparse-attention](https://www.emergentmind.com/topics/token-sparse-attention)

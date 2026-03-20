## 核心结论

MoE，Mixture of Experts，中文通常叫“专家混合模型”，可以理解为“同一层里放很多个子网络，但每次只激活其中少数几个”。真正影响它效果和部署成本的关键，不只是“有多少专家”，而是“路由器按什么粒度选专家”。

Token 级路由的定义是：每个 token，也就是每个位置上的向量表示，都独立决定自己要去哪些专家。它的优势是细粒度，因为一句话里不同 token 的语义、难度、语言现象本来就不一样；它的代价是调度极不规则，专家负载波动大，训练和推理都要处理复杂的分发与聚合。

序列级路由的定义是：先把整句或整段压缩成一个统一表示，再由这个统一表示选择一组专家，整条序列共享同一组专家。它的优势是吞吐更稳定、通信更可控、硬件利用率更高；它的代价是丢掉 token 间差异，容易把“局部难点”平均掉。

如果只看今天主流大模型实践，Token 级路由仍然是主流，因为语言建模天然需要细粒度条件化能力。但如果系统目标是高吞吐翻译、多任务在线部署、低延迟服务，序列级或任务级路由常常更有工程价值。更现实的结论是：纯 Token 级和纯序列级都不是终点，层级路由往往更合理，即先用任务或句级信息缩小专家范围，再在小范围内做 token 级细分。

---

## 问题定义与边界

这篇文章讨论的不是“MoE 是否比稠密模型更强”，而是更具体的问题：路由决策到底应该发生在 token 级，还是序列级，或者任务级。

先定义输入。设一条序列长度为 $T$，每个 token 的隐藏状态是 $h_t \in \mathbb{R}^d$，共有 $E$ 个专家。所谓“路由”，就是给出一个规则，把输入分配给一个或多个专家处理。

Token 级路由直接对每个 $h_t$ 打分：
$$
s_t = \mathrm{softmax}(W_r h_t), \quad s_t \in \mathbb{R}^E
$$
这里的 softmax 可以理解为“把原始分数变成总和为 1 的选择概率”。

序列级路由先做聚合：
$$
h_{\text{seq}} = \mathrm{Pool}(\{h_t\}_{t=1}^{T}), \quad
s_{\text{seq}} = \mathrm{softmax}(W_r h_{\text{seq}})
$$
这里的 Pool 是池化，也就是把多个 token 表示压缩成一个统一向量，常见做法有均值池化、CLS 向量、任务 ID 嵌入等。

这两类方法的边界很清楚：

| 维度 | Token 级路由 | 序列/任务级路由 |
|---|---|---|
| 决策颗粒度 | 每个 token 单独决策 | 整句或整个任务统一决策 |
| 特化能力 | 强 | 弱一些 |
| 负载波动 | 高 | 低 |
| 通信成本 | 高 | 低 |
| 吞吐稳定性 | 差一些 | 更好 |
| 适合场景 | 长上下文、复杂语言现象 | 高吞吐、多任务服务 |

问题的核心不是“谁绝对更好”，而是“你是否接受不均匀激活”。如果系统能承受复杂调度，Token 级能换来更强的表达能力；如果系统优先考虑稳定吞吐，序列级更容易落地。

这里还要明确一个常被忽视的边界：训练和推理不是同一个环境。训练时常用 load balancing loss，也就是“负载均衡损失”，目的很简单，别让所有 token 都挤到少数专家上；还会设置 capacity factor，也就是“专家容量系数”，限制每个专家一次最多接收多少 token。可到了推理阶段，batch 组成、请求长度、并发模式都变了，原本训练出的均衡性不一定还能成立，因此会出现专家坍塌、前缀偏置、局部拥堵等问题。

玩具例子可以帮助理解。把 token 想成学生，把专家想成老师。Token 级路由是每个学生都单独选老师，数学题去数学老师，英语题去英语老师；序列级路由则是整班先选一组老师，全班都按这组老师安排课程。前者更精细，后者更省组织成本。

---

## 核心机制与推导

标准 MoE 层通常由“路由器 + 多个 FFN 专家”组成。FFN，Feed-Forward Network，指 Transformer 里常见的前馈子网络。路由器先决定 token 去哪个专家，再把专家输出按权重合并。

Token 级 top-k 路由的基本形式是：
$$
s_t = \mathrm{softmax}(W_r h_t)
$$
$$
y_t = \sum_{i \in T_k(s_t)} s_{t,i} E_i(h_t)
$$
其中 $T_k(s_t)$ 表示分数最高的 $k$ 个专家，$E_i$ 是第 $i$ 个专家。top-k 的意思是“不让所有专家都算，只算最有希望的几个”。

为什么这种机制能产生专家特化？因为不同 token 的表示 $h_t$ 不同，路由器参数 $W_r$ 会把这种差异映射成不同专家偏好。训练久了以后，一些专家会更常处理数字、一些更常处理代码、一些更常处理长程依赖，这就是“特化”。

但这也带来第一个推导结果：激活集合随 token 改变，系统负载必然不规则。设一批 token 共 $N$ 个，专家数为 $E$，若路由接近均匀，则每个专家期望负载约为 $\frac{kN}{E}$；但只要输入分布偏斜，方差就会上升，某些专家会成为热点。热点专家需要排队，冷门专家则空闲，这就是 MoE 的典型硬件矛盾。

序列级路由的核心变化只有一步：把 $h_t$ 换成 $h_{\text{seq}}$。也就是：
$$
s_{\text{seq}} = \mathrm{softmax}(W_r h_{\text{seq}})
$$
然后对整条序列复用同一个专家集合：
$$
y_t = \sum_{i \in T_k(s_{\text{seq}})} s_{\text{seq},i} E_i(h_t)
$$
注意，这里每个 token 的输入还是自己的 $h_t$，只是“去哪几个专家”这个决定不再单独变化。

这一步直接带来两个后果。

第一，通信模式稳定了。因为整条序列共享相同专家集合，系统可以提前规划分发路径，all-to-all 交换更容易批量化。

第二，条件分辨率下降了。假设一句话里既有普通停用词，也有稀有实体、数学符号、代码片段。Token 级路由可以把这些不同部分送去不同专家；序列级路由会用一个平均后的全局表示做决定，低频但关键的局部模式容易被淹没。

看一个最小数值例子。设有 4 个专家，序列长度为 3，top-1 路由。

三个 token 的路由分数分别是：

- token 1: $[0.7, 0.2, 0.1, 0.0]$
- token 2: $[0.1, 0.8, 0.1, 0.0]$
- token 3: $[0.1, 0.2, 0.6, 0.1]$

Token 级路由会让三个 token 分别去专家 1、2、3。说明这句话内部三处内容性质不同。

如果先做均值池化，再得到统一分数：
$$
s_{\text{seq}} = [0.3, 0.4, 0.2, 0.1]
$$
那 top-1 就是专家 2，整句话三个 token 都交给专家 2。系统看起来更整齐，但 token 1 与 token 3 的个体差异被压平了。

真实工程中，这种差异会被放大。比如机器翻译场景里，一句输入可能同时包含通用词、专有名词、数字格式和语言特定结构。纯序列级路由对“这句话主要属于哪个语言或哪个域”判断很有效，但对句内局部结构不够敏感。因此很多系统会把任务级路由当第一层筛选，而不是最后决策。

---

## 代码实现

下面用一个可运行的 Python 玩具实现展示两种路由的差别。代码不依赖深度学习框架，只保留核心逻辑：打分、top-1 选择、负载统计。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def argmax(xs):
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i

def mean_pool(vectors):
    n = len(vectors)
    d = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(d)]

def matvec(W, x):
    return [sum(wij * xj for wij, xj in zip(row, x)) for row in W]

def token_level_route(tokens, router_W):
    probs = [softmax(matvec(router_W, h)) for h in tokens]
    experts = [argmax(p) for p in probs]
    return probs, experts

def sequence_level_route(tokens, router_W):
    h_seq = mean_pool(tokens)
    seq_probs = softmax(matvec(router_W, h_seq))
    expert = argmax(seq_probs)
    experts = [expert for _ in tokens]
    return seq_probs, experts

def load_balance_loss(experts, num_experts):
    # 一个极简版本：统计负载与理想均匀分布的平方误差
    counts = [0] * num_experts
    for e in experts:
        counts[e] += 1
    total = len(experts)
    target = total / num_experts
    return sum((c - target) ** 2 for c in counts) / num_experts

router_W = [
    [3.0, 0.0],   # expert 0
    [0.0, 3.0],   # expert 1
    [-2.0, -2.0], # expert 2
    [1.0, 1.0],   # expert 3
]

tokens = [
    [2.0, 0.1],   # 更像 expert 0
    [0.1, 2.0],   # 更像 expert 1
    [1.2, 1.1],   # 可能更像 expert 3
]

token_probs, token_experts = token_level_route(tokens, router_W)
seq_probs, seq_experts = sequence_level_route(tokens, router_W)

assert token_experts == [0, 1, 3]
assert len(set(seq_experts)) == 1
assert load_balance_loss(token_experts, 4) >= 0
assert load_balance_loss(seq_experts, 4) >= 0

print("token experts:", token_experts)
print("sequence experts:", seq_experts)
print("token load loss:", round(load_balance_loss(token_experts, 4), 4))
print("sequence load loss:", round(load_balance_loss(seq_experts, 4), 4))
```

这个例子能直接运行，并会看到一个典型现象：Token 级路由把三个 token 分到不同专家；序列级路由给整条序列分配同一个专家。

如果换成真实训练框架，前向过程通常长这样：

1. 路由器输出 logits，形状一般是 `[B, T, E]`。
2. 对每个 token 做 softmax 和 top-k。
3. 按专家重排 token，执行 dispatch。
4. 每个专家分别处理自己拿到的 token。
5. 再按原位置 gather 回来。
6. 计算主任务 loss，同时叠加负载均衡 loss。

序列级或任务级路由更像一个前置分支：

1. 先从 CLS、均值池化结果、任务 ID 中构造 $h_{\text{seq}}$。
2. 选出一组专家。
3. 后续 token 直接复用这组专家，或者仅在这组专家内部再做细粒度 top-k。

真实工程例子是多语机器翻译。任务级路由会先根据语言对、领域标签或句级表示选出候选专家组。比如“德语法律文本”先映射到法律域和德语相关专家组，再让 token 级路由在这组专家里细分。这样做的好处是把“全局任务差异”和“局部 token 差异”拆开处理，既减少全量专家搜索成本，又保留部分细粒度能力。

---

## 工程权衡与常见坑

工程上最难的不是写出路由公式，而是让它在真实集群上稳定、快、可控。

第一个坑是专家坍塌。所谓专家坍塌，就是大多数 token 长期只去少数几个专家，其他专家几乎没训练到。出现后，模型虽然名义上有很多专家，实际有效容量却很低。常见缓解方式是加 auxiliary load balancing loss，也就是辅助负载均衡损失；必要时再加 router z-loss、输入噪声、温度调整等手段。

第二个坑是容量溢出。top-k 选出来并不代表专家一定接得下，如果一个 batch 里大量 token 同时命中同一专家，就要丢 token、延迟处理，或走备选专家。capacity factor 的作用就是给专家预留额外容量，但它不是越大越好，越大意味着显存和通信开销也更高。

第三个坑是训练推理不一致。训练时 batch 通常大且混合，负载看起来较平均；线上推理时，单请求、短序列、特定前缀模板会让路由分布变形。比如聊天系统中大量请求以相似系统提示开头，前几个 token 会被反复送到同一专家，形成 prefix bias，也就是“前缀偏置”。

第四个坑是序列级路由过粗。它常常能提高吞吐，但会损伤长句内部的异质性处理能力。最容易受影响的是低频 token、代码片段、专名、数字格式，以及句中突然切换的语言结构。这类内容在全局池化后权重偏低，却对最终输出很关键。

第五个坑是只看理论 FLOPs，不看实际吞吐。Token 级 MoE 的理论稀疏计算量可能很低，但如果 all-to-all 通信、专家重排、padding、容量保护占掉大量时间，最终延迟不一定优于更简单的序列级方案。部署上必须盯真实的 tokens/s、p95 延迟、显存峰值，而不是只看纸面参数量。

一个实用的检查表如下：

| 问题 | 典型表现 | 常见处理 |
|---|---|---|
| 专家坍塌 | 少数专家极热，其他专家闲置 | 加 load balancing loss，调 router 温度 |
| 容量溢出 | token 被截断或回退 | 调整 capacity factor，改 batch 组织 |
| 通信过重 | GPU 利用率低，all-to-all 成瓶颈 | 限制候选专家范围，做层级路由 |
| 粗粒度失真 | 低频 token 表现差 | 序列级上再叠一层 token 级细化 |
| 训练推理漂移 | 线上分布明显偏移 | 用更贴近线上流量的验证集压测 |

真实工程里，THOR-MoE 这类层级方案值得注意。它的思路不是在所有专家上对每个 token 做完整竞争，而是先用任务级信号，例如语言或领域标签，筛出一个较小专家子集，再在该子集内做上下文相关的细粒度路由。这样能显著减少搜索空间和通信成本，同时避免纯任务级路由丢失句内差异。对于多任务、多语言、高并发推理服务，这种折中通常比纯 Token 级更容易落地。

---

## 替代方案与适用边界

如果把路由策略看成一个谱系，而不是两个二元选项，会更容易做设计决策。

第一类是纯 Token 级路由。适合追求最大表达能力、句内异质性很强、长上下文信息丰富的场景，比如通用语言建模、复杂代码建模。限制是调度最复杂，对硬件和系统优化要求最高。

第二类是纯序列级或任务级路由。适合任务边界清晰、吞吐优先、任务特征强于句内差异的场景，比如多任务分类、领域翻译、固定业务流量。限制是细粒度不足，对混合内容和局部难点不敏感。

第三类是层级路由。先做粗筛，再做细分。比如先用任务 ID、语言标签、CLS 表示选候选专家组，再让 token 级路由在候选组里 top-k。这类方案通常是工程上最均衡的设计。

第四类是弹性路由。所谓弹性，就是运行时可根据预算调整激活规模，例如把 top-2 改为 top-1，或者按设备负载动态缩减专家数。它适合延迟预算波动大、离线在线混合部署的系统，但实现更复杂，且需要处理训练与推理激活模式不一致的问题。

下面给出一个对照表：

| 方案 | 何时用 | 优势 | 限制 |
|---|---|---|---|
| 纯 Token 级 | 通用大模型、复杂语言现象 | 细粒度最强，专家特化充分 | 通信重，负载不稳 |
| 纯序列级 | 高吞吐服务、任务边界清晰 | 路由稳定，部署简单 | 丢失句内差异 |
| 纯任务级 | 明确任务 ID、多任务平台 | 易缓存，易规划资源 | 对同任务内细节不敏感 |
| 层级路由 | 多语言、多域、高并发 | 兼顾效率与精度 | 设计和调参复杂 |
| 弹性路由 | 预算动态变化 | 可按延迟或成本缩放 | 训练推理一致性更难 |

一个简单判断原则是：

如果你面对的是“同一句话内部差异很大”的问题，优先考虑 Token 级或层级路由。

如果你面对的是“请求量大、延迟硬约束强、任务类别清楚”的问题，优先考虑序列级、任务级或层级路由。

如果你已经知道自己最终要上线服务，通常不要直接从“全量 Token 级 + 全专家搜索”开始，因为这会把很多系统成本推迟到后期爆发。更合理的路径是先定义业务粒度，再决定是否需要细粒度 token 路由作为第二层。

---

## 参考资料

- EmergentMind, Token-Based Mixture-of-Experts Models
- Zuo et al., Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference, Findings of EMNLP 2021
- Li et al., THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation, ACL 2025
- Efficient Implementation of Large Language Models (LLMs), Medium technical article on MoE engineering details
- EmergentMind, Training-Inference Discrepancies in MoE Models

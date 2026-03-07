## 核心结论

混合专家模型（Mixture of Experts, MoE，白话讲就是“很多个子网络里每次只调用少数几个”）的核心价值，不是单纯把参数堆大，而是在**每个 token 的计算预算基本不变**的前提下，把模型总参数量显著做大。

标准做法是：把 Transformer 层里的前馈网络 FFN（Feed-Forward Network，白话讲就是“每个 token 单独经过的一段两层 MLP”）替换成 $N$ 个专家 FFN。对输入 token 表示 $x$，先由门控网络（gating network，白话讲就是“负责决定找谁来处理的路由器”）计算：

$$
G(x)=\operatorname{TopK}(\operatorname{softmax}(xW_g))
$$

其中 $W_g$ 是路由参数，softmax 给出每个专家的概率，TopK 只保留前 $k$ 个专家。于是，虽然总参数量近似从一个 FFN 变成 $N$ 个 FFN，但单个 token 实际只运行 $k$ 个专家，所以单 token FLOP 近似仍是 $k$ 个 FFN 的量级，而不是 $N$ 个 FFN 的量级。

新手版玩具例子可以这样理解：原来只有一个“全科医生”处理所有问题；MoE 则是有很多“专科医生”，每次只找最相关的 2 个医生会诊，其他医生待命。这样医院总知识储备更大，但单次问诊并没有把全部医生都叫来。

| 对比项 | 稠密 FFN | MoE FFN |
|---|---:|---:|
| 每层 FFN 个数 | 1 | $N$ |
| 每个 token 激活专家数 | 1 | $k$，通常 $k=2$ 或更大 |
| 总参数量 | $\#P_{\text{FFN}}$ | 约 $N \cdot \#P_{\text{FFN}}$ |
| 单 token 计算量 | 1 个 FFN | 约 $k$ 个 FFN |
| 是否依赖路由 | 否 | 是 |

所以 MoE 的一句话结论是：**参数量可以按专家数横向扩展，但每个 token 的实际计算只按激活专家数增长。**

---

## 问题定义与边界

MoE 要解决的问题是：**怎么把模型的表达能力继续做大，而不让训练和推理 FLOP 线性爆炸。**

如果使用普通稠密 Transformer，参数量每增加一倍，通常计算量、显存压力、训练成本也要跟着显著增长。MoE 试图把“总参数量”和“单次激活计算量”拆开，让模型拥有更多备用参数，但不要求每次都把它们全部算一遍。

设单个 FFN 参数量为 $\#P_{\text{FFN}}$，专家数为 $N$，则总参数量近似是：

$$
\#P \approx N \cdot \#P_{\text{FFN}}
$$

但若每次只激活 top-$k$ 个专家，则单 token 的 FFN 计算量近似为：

$$
F_{\text{MoE}} \approx k \cdot F_{\text{FFN}}
$$

这就是 MoE 能成立的基本边界：**参数量看总库存，FLOP 看实际调用数。**

再看适用边界。MoE 不是“任何场景都更好”，它更适合下面这类任务：

| 任务类型 | MoE 适用性 | 原因 |
|---|---|---|
| 大规模语言建模 | 高 | token 多、数据复杂，专家可学到不同模式 |
| 多任务/多领域模型 | 高 | 不同专家可分工处理不同分布 |
| 长序列理解/生成 | 中到高 | 参数扩展收益大，但通信成本需控制 |
| 小模型部署 | 低 | 路由与通信开销可能盖过收益 |
| 极低延迟在线服务 | 中 | 理论 FLOP 低，但工程延迟未必低 |
| 单卡小批量实验 | 低到中 | 专家并行收益不明显，调试成本高 |

新手版理解：把一本很厚的百科全书交给一个人读完，速度慢；把百科拆给很多专家，各自熟悉一部分知识，再按问题只找相关专家回答，知识库更大，但一次回答并不需要全员上线。问题在于，如果这些专家分散在很多设备上，来回调度也要花时间。

因此，MoE 的边界不只是数学问题，也是系统问题。它在“大模型、大批量、多设备并行”的条件下最有价值；在“小模型、低延迟、单机推理”里不一定划算。

---

## 核心机制与推导

MoE 的核心由三部分组成：**路由、专家计算、负载均衡。**

### 1. 路由机制

给定 token 表示 $x$，门控网络先输出每个专家的打分：

$$
p=\operatorname{softmax}(xW_g)
$$

这里 $p_i$ 表示“专家 $i$ 适合处理当前 token 的概率”。然后执行 top-$k$ 选择，只保留最大的 $k$ 个概率，其余置零。若被选中的专家集合是 $\mathcal{K}(x)$，则输出可写成：

$$
y=\sum_{i\in \mathcal{K}(x)} \tilde{p}_i \cdot E_i(x)
$$

其中 $E_i(\cdot)$ 是第 $i$ 个专家，$\tilde{p}_i$ 是在被选中专家内重新归一化后的权重。

softmax 的作用是让路由分数可微，白话讲就是“路由器不是硬编码写死，而是能通过梯度慢慢学会派单”。

### 2. 玩具例子：4 个专家、top-2

设某个 token 对 4 个专家的 softmax 概率是：

$$
[0.45, 0.30, 0.15, 0.10]
$$

top-2 后只保留前两个专家，即专家 1 和专家 2。重新归一化得到：

$$
\tilde{p}=
\left[
\frac{0.45}{0.45+0.30},
\frac{0.30}{0.45+0.30}
\right]
=
[0.6, 0.4]
$$

于是最终输出是：

$$
y=0.6 \cdot E_1(x) + 0.4 \cdot E_2(x)
$$

这个例子说明两件事：

1. 模型虽然有 4 个专家库存，但这个 token 实际只算了 2 个专家。
2. 如果每个专家结构和一个普通 FFN 一样，那么这个 token 的 FFN 计算量就约等于“双 FFN”，而不是“四 FFN”。

### 3. 负载均衡

如果只做 top-$k$ 路由，常见问题是“专家崩塌”（expert collapse，白话讲就是“大家都挤到少数几个专家，其他专家闲着”）。这样会让少数专家过载、过拟合，其他专家学不到东西。

常见辅助损失写成：

$$
L_{\text{bal}}=\alpha \sum_i f_i \cdot P_i
$$

其中：

- $f_i$：专家 $i$ 的实际使用率，也就是分到它的 token 占比
- $P_i$：专家 $i$ 的平均路由概率
- $\alpha$：负载均衡损失权重

直觉上，如果所有专家都平均使用，则 $f_i$ 和 $P_i$ 都更接近均匀分布，训练会更稳定。

| 场景 | 专家1 | 专家2 | 专家3 | 专家4 | 结果 |
|---|---:|---:|---:|---:|---|
| 均衡路由的 frequency $f_i$ | 0.26 | 0.24 | 0.25 | 0.25 | 训练稳定 |
| 崩塌路由的 frequency $f_i$ | 0.82 | 0.10 | 0.05 | 0.03 | 少数专家过载 |
| 均衡路由的 probability $P_i$ | 0.25 | 0.25 | 0.25 | 0.25 | router 正常 |
| 崩塌路由的 probability $P_i$ | 0.70 | 0.15 | 0.10 | 0.05 | router 偏置严重 |

### 4. 容量控制

光有负载均衡还不够。因为一个 batch 中，哪怕平均分布正常，也可能某一时刻有太多 token 同时被送到同一专家。于是工程上还要设置容量上限（capacity，白话讲就是“一个专家本轮最多接多少 token”）。

若 batch token 数为 $T$，专家数为 $N$，capacity factor 为 $c$，则单专家容量常写成：

$$
\text{capacity}=\left\lceil c \cdot \frac{kT}{N} \right\rceil
$$

若某专家超出容量，常见策略有三种：

1. 重新路由到下一个候选专家
2. 丢弃该 token 的该路由分支
3. 送到共享专家或残差路径

这一步决定了 MoE 在工程上是否真正可训练。

### 5. 真实工程例子：DeepSeek-V3

真实工程中，MoE 的重点不是“会不会路由”，而是“能不能在超大规模下稳住”。以公开资料中的 DeepSeek-V3 为例，其设计包含大量路由专家，每个 token 只激活一小部分专家，因而总参数达到数百 B，但单次激活计算量远低于总参数量。常见解读是：**总参数很大，但每次真正参与计算的活跃参数规模接近一个 30B 到 40B 级别的稠密模型。**

这正是 MoE 的工程意义：把大模型的“知识库存”做大，同时把“每次结账的计算账单”控制在一个较低水平。

---

## 代码实现

下面给一个可运行的简化版 Python 示例，演示 top-$k$ 路由、容量限制、专家输出合并和一个最小的负载均衡指标。这里不依赖深度学习框架，只用标准库，目的是看清机制。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def renorm(weights):
    s = sum(weights)
    return [w / s for w in weights]

def expert_fn(expert_id, x):
    # 玩具专家：不同专家做不同线性变换
    return (expert_id + 1) * x + expert_id

def moe_forward(tokens, gate_logits, top_k=2, capacity_factor=1.0):
    num_tokens = len(tokens)
    num_experts = len(gate_logits[0])

    capacity = math.ceil(capacity_factor * (top_k * num_tokens / num_experts))
    assigned_count = [0] * num_experts
    routed_tokens = [[] for _ in range(num_experts)]
    outputs = []

    for token_idx, x in enumerate(tokens):
        probs = softmax(gate_logits[token_idx])
        candidates = topk_indices(probs, top_k)

        chosen = []
        chosen_probs = []

        # 容量控制：优先按 top-k 分配，满了就跳过
        for expert_id in candidates:
            if assigned_count[expert_id] < capacity:
                assigned_count[expert_id] += 1
                routed_tokens[expert_id].append(token_idx)
                chosen.append(expert_id)
                chosen_probs.append(probs[expert_id])

        # 如果 top-k 都满了，退化为残差直通
        if not chosen:
            outputs.append(x)
            continue

        chosen_probs = renorm(chosen_probs)
        y = 0.0
        for expert_id, w in zip(chosen, chosen_probs):
            y += w * expert_fn(expert_id, x)
        outputs.append(y)

    # 统计实际使用率 f_i
    total_assignments = sum(assigned_count)
    freq = [c / total_assignments if total_assignments else 0.0 for c in assigned_count]

    # 统计平均路由概率 P_i
    mean_prob = [0.0] * num_experts
    for row in gate_logits:
        probs = softmax(row)
        for i, p in enumerate(probs):
            mean_prob[i] += p
    mean_prob = [p / num_tokens for p in mean_prob]

    aux_loss = sum(f * p for f, p in zip(freq, mean_prob))
    return outputs, assigned_count, freq, mean_prob, aux_loss, capacity

tokens = [1.0, 2.0, 3.0, 4.0]
gate_logits = [
    [3.0, 2.0, 0.0, -1.0],
    [2.5, 2.4, 0.1, -0.5],
    [0.2, 0.1, 2.8, 2.7],
    [0.0, -0.2, 3.2, 3.1],
]

outputs, assigned_count, freq, mean_prob, aux_loss, capacity = moe_forward(
    tokens, gate_logits, top_k=2, capacity_factor=1.0
)

assert len(outputs) == 4
assert capacity == 2
assert sum(assigned_count) <= 8
assert abs(sum(mean_prob) - 1.0) < 1e-9
assert aux_loss > 0.0

print("outputs:", outputs)
print("assigned_count:", assigned_count)
print("freq:", freq)
print("mean_prob:", mean_prob)
print("aux_loss:", aux_loss)
```

这段代码省略了反向传播、分布式通信和张量并行，但已经覆盖了 MoE 的最基本流程：

1. 输入 token 表示
2. 计算 gate logits
3. softmax 变成概率
4. top-$k$ 选专家
5. 检查 capacity
6. 调用专家
7. 按权重合并输出
8. 统计负载均衡指标

对应的伪码可以进一步压缩成：

```python
probs = softmax(x @ W_g)
ids = topk(probs, k)
ids = drop_over_capacity(ids)
y = sum(norm(probs[ids])[j] * expert[ids[j]](x) for j in range(len(ids)))
```

关键配置通常有这些：

| 配置项 | 作用 | 常见取值/倾向 |
|---|---|---|
| `top_k` | 每个 token 激活几个专家 | 2 常见；更大更稳定但更贵 |
| `capacity_factor` | 单专家容量冗余倍数 | 1.0 到 1.25 常见 |
| `load_loss_weight` | 负载均衡辅助损失权重 | 小权重，避免主任务被压制 |
| `num_experts` | 专家总数 | 随模型规模增长 |
| `shared_expert` | 共享专家/公共路径 | 常用于稳住基础能力 |
| `router_noise` | 路由噪声 | 用于避免早期塌缩 |

真实工程例子里，PyTorch 或 Megatron 风格实现还会加入 `all-to-all` 通信，即把属于某个专家的 token 从不同 GPU 收集到同一设备，再统一计算，这一步往往比公式本身更影响性能。

---

## 工程权衡与常见坑

MoE 理论上“参数大但 FLOP 不大”，但工程里最难的不是这个口号，而是如何避免失控。

### 1. 专家崩塌

表现是少数专家收到绝大多数 token，其他专家几乎不工作。现象包括：

- 训练 loss 不稳定
- 某些专家梯度极大，其余专家梯度接近零
- 路由频率长期偏斜

新手版理解：团队里总把任务派给两个人，其余人一直闲着。久而久之，忙的人越来越忙，闲的人越来越不会做事。

缓解方法通常包括：

- 加辅助负载均衡损失
- 路由加噪声，强制早期探索
- 使用 shared expert 保底
- 使用 bias-based 或 aux-loss-free 的平衡策略

### 2. 容量溢出

当某个专家瞬时接单过多，就会超过 capacity。若处理不好，会直接丢信息，或者让通信堵塞。

典型逻辑是：

```python
if assigned_count[expert_id] >= capacity:
    reroute_to_next_expert()
```

如果没有备份路径，overflow token 的表示会被系统性削弱，尤其在 batch 较小时更明显。

### 3. 通信瓶颈

在多卡环境中，token 不一定和它要访问的专家在同一张卡上。于是需要设备间交换 token。理论上 FLOP 省了，但如果网络互联慢，端到端延迟还是会高。

这就是为什么 MoE 常常“算力账便宜，系统账未必便宜”。

| 常见坑 | 表现 | 定位指标 | 暂缓策略 |
|---|---|---|---|
| 专家崩塌 | 少数专家超忙，其他专家闲置 | expert frequency、router entropy | aux loss、router noise、共享专家 |
| 容量溢出 | token 被丢弃或回退过多 | overflow rate、drop rate | 增大 capacity factor、改进 reroute |
| 通信瓶颈 | GPU 利用率低但延迟高 | all-to-all 时间占比 | 优化 expert 并行拓扑、减少跨卡路由 |

辅助 loss 的计算在实现上通常是批次级统计：

$$
L_{\text{bal}}=\alpha \sum_i f_i \cdot P_i
$$

其中 $f_i$ 来自实际派单结果，$P_i$ 来自 router 的概率均值。二者同时看，才能区分“router 本来就偏”还是“capacity 截断后才偏”。

还有一个常见误解：有人以为“MoE 的 FLOP 和稠密模型完全一样”。这只在“拿一个专家的 FFN 当基准”时近似成立。若你把 $k$ 从 1 提到 2 或 8，单 token 计算当然也会按激活专家数增长。准确说法是：**MoE 的计算增长看 $k$，不是看总专家数 $N$。**

---

## 替代方案与适用边界

MoE 不是唯一的稀疏化路线。它解决的是“参数扩展与计算脱钩”的问题，但别的方案也能减少 FLOP，只是路径不同。

### 1. 与稠密模型对比

稠密 Transformer 的优点是结构简单、行为稳定、延迟更可控。对于中小模型、低延迟场景、单机部署，稠密模型常常更务实。它的问题是参数扩展时计算也跟着涨，很难做到“库存大但每次少算”。

### 2. 与其他稀疏机制对比

还有一些方法通过稀疏注意力、线性注意力、token 剪枝、head 剪枝、空间稀疏等方式节省计算。它们主要减少“参与运算的 token、head 或位置”，而 MoE 主要减少“参与运算的参数子模块”。

可用下面的表对比：

| 方案 | 参数扩展能力 | FLOP 控制方式 | 实现难度 | 主要瓶颈 |
|---|---|---|---|---|
| Dense Transformer | 低到中 | 无条件计算，全部执行 | 低 | 参数和 FLOP 同涨 |
| MoE | 高 | 每 token 只激活 $k$ 个专家 | 高 | 路由、容量、通信 |
| Token Sparsity | 中 | 丢弃不重要 token | 中到高 | 信息丢失风险 |
| Head Sparsity | 中 | 只保留部分注意力头 | 中 | 表达受限 |
| Spatial/Window Sparsity | 中 | 只算局部区域 | 中 | 结构依赖强 |

MoE 的数学特征可以概括为：

$$
\#P_{\text{MoE}} \approx N \cdot \#P_{\text{FFN}}
$$

但单次 FFN 计算更接近：

$$
F_{\text{limited}} \approx k \cdot F_{\text{FFN}}
$$

这和 Dense 模型的本质区别是：Dense 模型里“有多少参数就基本都要算”，MoE 里“有多少参数”和“每次算多少”是两件事。

新手版对比可以这样记：

- 稠密模型：所有工人一起上，流程稳定，但人一多成本就直线上升。
- MoE：养很多专家，但每次只叫少数人来做，成本受控，但调度系统必须足够强。
- 其他稀疏方法：不是多养人，而是减少任务范围，例如只做局部、只看重点。

所以适用边界很明确：

- 如果你要的是最大参数容量，且有成熟多机并行能力，MoE 很合适。
- 如果你要的是低延迟、低复杂度、易部署，Dense 往往更稳。
- 如果瓶颈在注意力而不是 FFN，优先考虑注意力稀疏化，而不是盲目上 MoE。

---

## 参考资料

- Wikipedia: Mixture of Experts – 用于快速建立 MoE 基本定义、路由公式与历史背景，可查 `TopK(softmax(xW_g))` 的直观解释。
- NVIDIA Megatron Core MoE 文档 – 重点看负载均衡、capacity、并行策略与工程实现细节，适合查训练级配置项。
- Neptune: Mixture of Experts LLMs – 偏工程综述，适合看专家崩塌、通信开销和系统层面的常见问题。
- DeepSeek-V3 介绍与技术报告 – 适合查“总参数量很大但激活参数较小”的真实案例，以及专家数量、激活专家数等设计思路。
- DeepSeek 相关源码解析文章 – 适合对照理解共享专家、路由专家、负载均衡策略在代码里如何落地。

推荐顺序阅读：

1. 先看 Wikipedia，建立“专家 + 路由 + top-k”的基本图景。
2. 再看 Megatron 文档，理解 capacity、aux loss、并行通信这些工程关键字。
3. 最后看 DeepSeek-V3 资料，理解真实大模型里“为什么能做到 671B 总参数但每次只激活一部分”。

如果你最关心“37B 激活量级是怎么来的”，优先看 DeepSeek-V3 的官方介绍或技术报告；如果你最关心“为什么专家会塌缩”，优先看 Megatron 文档和工程综述。

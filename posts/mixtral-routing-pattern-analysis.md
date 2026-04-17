## 核心结论

Mixtral 8x7B 的路由模式，不是“所有层都在做同一件事”。它更接近一种分层分工的 MoE（Mixture of Experts，意思是“一层里有很多个前馈网络，但每个 token 每次只激活少数几个”）结构：浅层更容易按词性、位置、局部句法模式分流，深层更容易按主题、实体类型、任务语义分流。对 Mixtral 这类 top-2 路由模型，观察每层行为时，最有信息量的不是单个 token 进了哪个专家，而是三类统计量一起看：

- 专家利用率方差：这一层的 token 是否长期集中到少数专家。方差越大，偏载越明显。
- 路由熵：路由概率分布有多分散。熵越高，说明前几名专家分数更接近。
- 跨层一致性：相邻层或相邻 token 是否反复命中相似专家组合。一致性高说明模式稳定，但也可能形成热点专家。

一个可操作的结论是：早层和末层通常更保守，中层更值得扩展候选集。因为 Mixtral 固定 top-2 时，早层和末层经常呈现更高的 top-2 mass，也就是前两个专家已经吃掉大部分概率质量；中层的分布更平、更接近高熵状态，此时允许 `c=3` 或 `c=4` 的候选池，往往能改善负载均衡，而不明显伤害质量。

如果用更直白的话解释：浅层更像“表面模式分拣器”，看词形、标点、位置和短距离搭配；深层更像“任务语义分拣器”，看这段内容在讲谁、讲什么、要完成什么任务。LASER 这类推理策略并不是推翻 top-2，而是在“中层分数没那么集中”时，先多看一两个备选专家，再从里面挑两位真正执行。

| 层段 | 典型分布形状 | top-2 mass / dominance 倾向 | 路由熵倾向 | 更直白的解释 | 推荐策略 |
|---|---|---:|---:|---|---|
| 浅层 | 偏集中，受语法/位置影响大 | 高 | 低到中 | 多在做表面模式分流 | 保守 top-2 |
| 中层 | 更平滑，多个专家接近 | 中 | 中到高 | 多个专家都像合理候选 | 候选池扩到 3/4 |
| 深层 | 再次集中，但偏语义决策 | 较高 | 中 | 更接近最终语义选择 | 以 top-2 为主，谨慎扩展 |

如果要做可视化，最有价值的不是只画某一层的专家命中次数，而是并列画三张图：early / middle / final 的 entropy、top-2 mass、expert variance 对比图。这样才能同时看见“是否尖锐”“是否偏载”“是否稳定”。

---

## 问题定义与边界

问题不是“Mixtral 某个专家学了什么”，而是“不同层为什么需要不同程度的专家多样性”。这里的边界要先说清：

1. 讨论对象是 Mixtral-8x7B 这类每层 8 个专家、每个 token 只选 2 个的稀疏 MoE。
2. 讨论重点是路由行为，不是训练语料成分，也不是具体 benchmark 排名。
3. 讨论目标是解释“层内分布形状”和“推理期负载策略”的关系，而不是声称某一层只学语法、另一层只学语义。

先定义几个最常用的量。

设某一层某个 token 的路由概率为：

$$
p = (p_1,p_2,\dots,p_8), \qquad \sum_{i=1}^{8} p_i = 1
$$

其中 $p_i$ 表示该 token 被第 $i$ 个专家处理的概率。路由熵定义为：

$$
H(p)=-\sum_{i=1}^{8} p_i \log p_i
$$

这里的直观含义很简单：

- 若概率非常集中在少数专家上，熵较低。
- 若 8 个专家概率比较接近，熵较高。

对 8 个专家来说，理论最大值出现在完全均匀分布：

$$
p_i=\frac{1}{8}
$$

因此：

$$
H_{\max}=\log 8 \approx 2.08
$$

再定义 top-2 mass，也可当作一个 dominance 指标：

$$
D = p_{(1)} + p_{(2)}
$$

其中 $p_{(1)}, p_{(2)}$ 是最大的两个概率。$D$ 越高，说明前两个专家已经基本决定结果，额外候选的价值越小。

为了让这两个量更容易理解，可以把常见情况列成表：

| 路由分布特征 | 熵 $H$ | top-2 mass $D$ | 说明 |
|---|---:|---:|---|
| 一个专家特别强，第二名也明显领先 | 低 | 高 | top-2 已很明确 |
| 前 3 到 4 名比较接近 | 中到高 | 中 | 适合给更多候选空间 |
| 8 个专家都差不多 | 高 | 低 | 最像“高熵层” |

再看两个玩具例子。

例子 1：

$$
p=[0.52,0.24,0.08,0.05,0.04,0.03,0.02,0.02]
$$

则：

$$
D=0.52+0.24=0.76
$$

这说明前两个专家已经占了 76% 的概率质量。若此时还要强行扩到 4 个候选，通常收益有限。

例子 2：

$$
p=[0.19,0.17,0.15,0.13,0.11,0.10,0.08,0.07]
$$

则：

$$
D=0.19+0.17=0.36
$$

这说明路由更平，多个专家都像合理选择。此时扩大候选池，更可能帮助负载调度。

为了方便新手判断，可以记一个经验对照表：

| 观测现象 | 更可能意味着什么 |
|---|---|
| 熵低，top-2 mass 高 | 路由很自信，扩展收益小 |
| 熵高，top-2 mass 低 | 路由更犹豫，扩展收益可能更大 |
| 熵高，但专家利用率方差也高 | 并不一定负载健康，可能只是少数专家在某些 token 类型上反复占优 |
| 跨层一致性很高 | 路由很稳定，但也要防热点 |

从 LASER 论文给出的 Mixtral-8x7B GSM8K 阈值看，early / middle / final 层的 $\varepsilon_{high}$ 约为 `0.72 / 0.75 / 0.80`。它不是“永远正确”的常数，而是一类代表性现象：末层往往需要更高 dominance 才不扩展，因为末层错选专家更容易直接影响最终输出。

---

## 核心机制与推导

路由器（gate，意思是“决定把 token 送给谁的打分器”）先读取当前 token 的隐藏状态 $x \in \mathbb{R}^{d}$，再投影成 8 个专家分数：

$$
z = xW_g \in \mathbb{R}^{8}
$$

其中 $W_g \in \mathbb{R}^{d \times 8}$ 是路由器参数。然后对 $z$ 做 softmax，得到概率分布：

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{8}\exp(z_j)}
$$

之后只保留前两名专家，并重新归一化。可写成：

$$
G(x)=\mathrm{Norm}(\mathrm{Top2}(\mathrm{Softmax}(xW_g)))
$$

其中：

- `Softmax`：把原始分数变成概率。
- `Top2`：只保留最大的两个专家。
- `Norm`：把这两个权重重新缩放到和为 1。

假设 top-2 专家索引是 $a,b$，那么最终的门控权重是：

$$
\tilde p_a=\frac{p_a}{p_a+p_b},\qquad
\tilde p_b=\frac{p_b}{p_a+p_b}
$$

其他专家权重为 0。于是该层输出为：

$$
y=\tilde p_a \cdot E_a(x) + \tilde p_b \cdot E_b(x)
$$

其中 $E_a,E_b$ 是对应专家的前馈网络。在 Mixtral 里，专家内部常见实现是 SwiGLU，可以先把它理解成“每个专家自己的非线性处理器”。

如果再写得更完整一点，专家内部可表示为：

$$
E_i(x)=W_{2,i}\big(\mathrm{SiLU}(W_{1,i}x)\odot (W_{3,i}x)\big)
$$

这里：

- $W_{1,i},W_{2,i},W_{3,i}$ 是第 $i$ 个专家的参数。
- $\odot$ 是逐元素乘法。
- `SiLU` 是常见激活函数。

为什么会出现“浅层语法、深层语义”的层次差异？关键不在于专家先天带标签，而在于输入表征在逐层变化。

可以把表征变化理解成下面三段：

| 层段 | 隐藏状态更接近什么 | 路由更容易依据什么 |
|---|---|---|
| 浅层 | token 表面形式 | 词形、标点、位置、词性、局部搭配 |
| 中层 | 组合后的局部语义 | 片段结构、短句角色、局部主题 |
| 深层 | 更抽象的任务表征 | 实体类型、主题、指令意图、答案组织 |

也就是说，同一个路由函数在不同层处理的是不同性质的输入，因此它做出的专家分配也会不同。浅层更靠近“输入长什么样”，深层更靠近“输入想表达什么、要完成什么”。

一个具体例子是长上下文问答。假设输入中前半段是法规条文，后半段是提问：

- 浅层可能更容易按数字格式、章节编号、专有名词、标点结构分流。
- 中层开始把“定义句”“条件句”“例外条款”这些局部结构区分开。
- 深层更可能按“法规解释”“事实匹配”“问题求解”这样的语义角色分流。

因此，中层高熵并不表示“中层不重要”，反而常说明这一层正在整理多种可能的表示路径。多个专家都像合理候选，所以它最适合做弹性调度。

LASER 的关键观察是：并不是每层都该机械执行固定 top-2。论文把路由分布分成三类：

| 路由形状 | 直观解释 | 工程动作 |
|---|---|---|
| single-head | 一个专家明显压倒性领先 | 几乎不要扩展 |
| plateau | 前几名很接近 | 可以适度调度负载 |
| smooth | 多个专家都差不多 | 最适合扩展候选 |

把它翻译成更直接的工程判断：

- 若前两名已经明显强于其他专家，继续加候选只会增加调度空间，不一定增加正确性。
- 若第 2、3、4 名非常接近，那么“谁来处理”未必是严格唯一的，此时扩展候选池更有意义。
- 若分布过于平滑，说明路由器并没有强偏好，此时可以优先考虑负载均衡，但也要警惕过度扩展带来的错选风险。

所以，中层高熵层的价值，不是“让更多专家都算一遍”，而是“先允许更多候选，再在候选里选最合适的两个”。

---

## 代码实现

下面先给一个可运行的玩具实现，复现 top-2 路由、熵计算、专家利用率方差，以及一个简化版 LASER 风格候选池决策。代码只依赖 Python 标准库，直接运行即可。

```python
import math
from collections import Counter

NUM_EXPERTS = 8

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def top2_normalize(probs):
    idx = topk_indices(probs, 2)
    mass = probs[idx[0]] + probs[idx[1]]
    out = [0.0] * len(probs)
    out[idx[0]] = probs[idx[0]] / mass
    out[idx[1]] = probs[idx[1]] / mass
    return out, idx, mass

def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)

def utilization_variance(assignments, num_experts=NUM_EXPERTS):
    counts = [0] * num_experts
    for idx in assignments:
        counts[idx] += 1
    mean = sum(counts) / num_experts
    return sum((c - mean) ** 2 for c in counts) / num_experts

def candidate_pool_size(probs, eps_high, high_entropy=1.85, mid_entropy=1.55):
    sorted_probs = sorted(probs, reverse=True)
    top2_mass = sorted_probs[0] + sorted_probs[1]
    h = entropy(probs)

    if top2_mass >= eps_high:
        return 2
    if h >= high_entropy:
        return 4
    if h >= mid_entropy:
        return 3
    return 2

def route_one_token(logits, eps_high):
    probs = softmax(logits)
    gate, top2_idx, top2_mass = top2_normalize(probs)
    pool = candidate_pool_size(probs, eps_high=eps_high)
    cand_idx = topk_indices(probs, pool)
    return {
        "logits": logits,
        "probs": probs,
        "entropy": entropy(probs),
        "top2_idx": top2_idx,
        "top2_mass": top2_mass,
        "candidate_pool": cand_idx,
        "candidate_pool_size": pool,
        "normalized_top2_gate": gate,
    }

def route_batch(batch_logits, eps_high):
    results = [route_one_token(logits, eps_high) for logits in batch_logits]
    top1_assignments = [max(range(NUM_EXPERTS), key=lambda i: r["probs"][i]) for r in results]
    top2_pairs = [tuple(r["top2_idx"]) for r in results]

    avg_entropy = sum(r["entropy"] for r in results) / len(results)
    avg_top2_mass = sum(r["top2_mass"] for r in results) / len(results)
    var_top1 = utilization_variance(top1_assignments, num_experts=NUM_EXPERTS)
    pair_counts = Counter(top2_pairs)

    return results, {
        "avg_entropy": avg_entropy,
        "avg_top2_mass": avg_top2_mass,
        "top1_utilization_variance": var_top1,
        "top2_pair_counts": pair_counts,
    }

def pretty(xs, ndigits=4):
    return [round(x, ndigits) for x in xs]

if __name__ == "__main__":
    # 例子 1：dominance 高，更适合固定 top-2
    early_like = [
        [3.2, 2.4, 0.8, 0.3, 0.1, -0.2, -0.4, -0.5],
        [2.8, 2.1, 0.7, 0.5, 0.2, -0.1, -0.2, -0.6],
        [3.0, 2.3, 0.9, 0.4, 0.0, -0.3, -0.4, -0.7],
    ]

    # 例子 2：更平滑，更像中层高熵分布
    middle_like = [
        [1.2, 1.1, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65],
        [1.1, 1.0, 0.98, 0.94, 0.86, 0.8, 0.72, 0.67],
        [1.15, 1.08, 1.02, 0.96, 0.84, 0.79, 0.73, 0.69],
    ]

    early_results, early_stats = route_batch(early_like, eps_high=0.72)
    middle_results, middle_stats = route_batch(middle_like, eps_high=0.75)

    print("=== Early-like layer ===")
    for i, r in enumerate(early_results):
        print(f"token {i}:")
        print("  probs =", pretty(r["probs"]))
        print("  entropy =", round(r["entropy"], 4))
        print("  top2_idx =", r["top2_idx"])
        print("  top2_mass =", round(r["top2_mass"], 4))
        print("  candidate_pool =", r["candidate_pool"])

    print("summary:", {
        "avg_entropy": round(early_stats["avg_entropy"], 4),
        "avg_top2_mass": round(early_stats["avg_top2_mass"], 4),
        "top1_utilization_variance": round(early_stats["top1_utilization_variance"], 4),
    })

    print("\n=== Middle-like layer ===")
    for i, r in enumerate(middle_results):
        print(f"token {i}:")
        print("  probs =", pretty(r["probs"]))
        print("  entropy =", round(r["entropy"], 4))
        print("  top2_idx =", r["top2_idx"])
        print("  top2_mass =", round(r["top2_mass"], 4))
        print("  candidate_pool =", r["candidate_pool"])

    print("summary:", {
        "avg_entropy": round(middle_stats["avg_entropy"], 4),
        "avg_top2_mass": round(middle_stats["avg_top2_mass"], 4),
        "top1_utilization_variance": round(middle_stats["top1_utilization_variance"], 4),
    })
```

这段代码里每个函数分别对应一个明确问题：

| 函数 | 作用 | 新手应关注什么 |
|---|---|---|
| `softmax` | 把 logits 变成概率 | 概率总和为 1 |
| `top2_normalize` | 只保留前两名并归一化 | Mixtral 真正执行的是这一步后的权重 |
| `entropy` | 计算路由分布是否分散 | 熵高不等于一定更好，只表示更平 |
| `candidate_pool_size` | 按分布形状决定候选池大小 | 这就是简化版动态路由策略 |
| `utilization_variance` | 观察专家负载是否偏 | 工程上不能只看准确率 |

若运行这段代码，你会看到两个稳定现象：

1. `early_like` 的 `top2_mass` 更高，`candidate_pool` 往往保持在 2。
2. `middle_like` 的熵更高，`candidate_pool` 更容易扩到 3 或 4。

这正对应前文的核心判断。

如果把它对应到真实模型，伪代码可以写成：

| 条件 | 含义 | 动作 |
|---|---|---|
| `top2_mass >= ε_high` | 前两名已足够主导 | 直接 top-2 |
| `top2_mass < ε_high` 且熵高 | 分数更平均 | 扩大候选池到 4 |
| `top2_mass < ε_high` 且熵中等 | 有少量可替代专家 | 候选池到 3 |

如果要接近真实工程实现，流程通常是：

1. 算 gate logits。
2. softmax 得到 8 个专家概率。
3. 看该层统计是否允许扩展候选池。
4. 在候选池里结合当前负载挑出 2 个专家。
5. 对两个专家输出做归一化加权求和。

这里要强调一个容易混淆的点：真实系统里“候选池扩到 4”不等于“4 个专家都执行”。通常仍然只执行 2 个，只是选 2 个时不再死板地只看原始 top-2，而会把负载约束一起考虑进去。

---

## 工程权衡与常见坑

最大权衡是：负载均衡和路由质量不是同一个目标。若只看均衡，最平均的方案最好；但只看均衡会破坏“本来就该由这两个专家处理”的语义选择，准确率会掉。

Mixtral 上更稳妥的经验是：

| 熵范围 | top-2 mass 倾向 | 候选池建议 | 监控重点 |
|---|---:|---:|---|
| 低 | 高 | 2 | 质量优先 |
| 中 | 中 | 3 | 观察 Iagg 与尾部延迟 |
| 高 | 低 | 4 | 防止过度扩展导致选错专家 |

这里的 Iagg 可以理解成“聚合后的不均衡程度”，越高说明某些专家或某些 GPU 更容易成为慢点。它不是在问“平均负载高不高”，而是在问“最忙的那部分是否明显更忙”。

工程上至少要同时监控下面几类量：

| 指标 | 作用 | 异常时意味着什么 |
|---|---|---|
| 每层 entropy | 看分布是否过尖或过平 | 过尖可能塌缩，过平可能路由无判别力 |
| 每层 top-2 mass | 看前两名是否足够主导 | 很低时扩展更有意义 |
| 每层 utilization variance | 看专家负载是否偏载 | 很高时要警惕热点 |
| 相邻 token 专家重复率 | 看时间局部性 | 过高可能造成连续热点 |
| 相邻层 top-2 重合率 | 看跨层一致性 | 持续偏高可能说明路径过于固定 |

一个常见坑是把“中层高熵”误解成“中层无语义”。实际恰好相反。高熵只说明多个专家都像合理候选，不说明这一层不重要。它反而说明该层在做更细的表示整理，因此多个专家都有参与价值。

另一个常见坑是把“扩展候选池”误解成“让更多专家一起算”。这两件事不同：

- 扩展候选池：是增加可选范围。
- 增加激活专家数：是增加真实计算量。

LASER 一类方法主要做前者，不是简单把 top-2 改成固定 top-4。

再一个常见坑是忽略时间局部性。Hugging Face 的路由分析提到，在第 15 和 31 层，连续 token 命中同一组专家的概率明显高于随机分布。时间局部性就是“相邻 token 喜欢走相似路径”。这会带来两面性：

- 好处：路由更稳定，通信模式更平滑。
- 风险：热点专家持续过载，尾延迟被放大。

因此跨层一致性不能只拿来解释机制，也要拿来做监控。实践里可以跟踪：

- 每层 expert utilization variance
- 每层 entropy 与 top-2 mass
- 相邻 token 的专家重复率
- 相邻层的 top-2 重合率，即简化版 M2 观察曲线

如果中层已经扩到 `c=4`，但重复率仍长期偏高，说明问题不一定是候选太少，而可能是输入分布、batch 结构、专家映射方式或跨设备布局本身导致了热点。

再补一个更贴近服务端的例子。假设某一批请求里有大量表格抽取任务：

- 如果这些请求在中层都偏向同 2 个专家，那么即使 top-2 很“正确”，系统层面仍可能出现两个 GPU 长期拥堵。
- 若该层熵本来较高，允许从前 4 名里再选 2 个，就可能把部分 token 导到相近但更空闲的专家上。
- 如果这样做后质量基本不变、尾延迟下降，那么这层就是适合做动态调度的层。

这正是“模型机制”和“系统目标”之间的连接点。

---

## 替代方案与适用边界

为什么 Mixtral 选择 top-2，而不是 top-1 或 top-4？

| 方案 | 优点 | 缺点 | 更适合的场景 |
|---|---|---|---|
| top-1 | 最省算力、最省通信 | 抖动大，鲁棒性差 | 极端追求吞吐 |
| top-2 | 质量和效率平衡最好 | 仍可能偏载 | 通用推理与训练 |
| top-4 | 负载调度空间更大 | 计算和通信明显上升 | 高熵中层的实验性策略 |
| auxiliary load loss | 训练期可抑制偏载 | 可能扭曲主任务目标 | 训练阶段 |
| expert dropout | 增强鲁棒性 | 不一定改善真实服务负载 | 训练正则化 |
| 动态调度如 LASER | 推理期可按层适配 | 需要额外统计与阈值校准 | 在线服务 |

对 Mixtral 这种 coarse-grained MoE 来说，top-2 的意义在于“给每个 token 两条专家路径”，既保留冗余，也不把通信成本推太高。top-1 更便宜，但一旦路由器判断失误，恢复空间更小；top-4 更灵活，但会明显增加计算与调度成本。top-2 落在这两者之间，是一个工程上更稳的平衡点。

LASER 本质上不是改成固定 top-4，而是在 top-2 和更大候选空间之间做动态折中。它依赖的是“不同层的分布形状不同”，而不是“更多专家总是更好”。

适用边界也要说清：

- 如果模型层间分布非常稳定，动态扩展收益会变小。
- 如果服务瓶颈不在专家负载，而在 KV cache、prefill、网络通信或调度框架本身，优化路由未必是首要矛盾。
- 如果任务极度依赖末层少数专家，过度扩展候选池可能直接伤害答案质量。
- 如果 batch 很小，负载均衡空间本来就有限，复杂调度可能没有明显收益。
- 如果你拿不到 per-layer 路由统计，只能盲调阈值，那么动态策略容易失控。

可以把几种方案的适用判断再压缩成一张表：

| 你的目标 | 更优先考虑的方案 |
|---|---|
| 吞吐优先，允许少量质量损失 | 更激进的 top-1 / 简化路由 |
| 质量与效率平衡 | 固定 top-2 |
| 在线服务遇到明显热点 | 分层动态调度，如 LASER |
| 训练时专家塌缩严重 | auxiliary loss、router 稳定化 |
| 问题主要是系统通信瓶颈 | 先查并行策略和设备映射，不要只盯路由 |

因此，这篇文章的结论边界是：它解释的是“为什么不同层适合不同程度的候选扩展”，不是在宣称“任何 Mixtral 部署都应该改动态路由”。

---

## 参考资料

| 资料 | 核心贡献 | 建议阅读方式 |
|---|---|---|
| LASER 路由分析论文（OpenReview, ICLR 2026 under review） | 提供 per-layer `top-k mass`、entropy、M2、Iagg，以及 Mixtral-8x7B 在 GSM8K 上 early / middle / final 的阈值示例 `0.72 / 0.75 / 0.80` | 先看它如何定义层级分布，再看阈值和负载策略，不要只抄结论 |
| Emergent Mind 的 Mixtral-8x7B 主题页 | 总结 Mixtral 的 top-2 路由、层级 specialization、语义驱动路由和注意力-专家相关性 | 适合作为概念整理材料，用来补齐“浅层到深层分工变化”的直观图景 |
| Emergent Mind 的 Mixtral family 架构页 | 给出 Mixtral 路由公式、专家聚合公式和机制性定义 | 适合对照本文公式，明确 gate、top-2 和专家加权输出分别是什么 |
| Hugging Face 关于 MoE 负载均衡演化的综述 | 讨论时间局部性、连续 token 重复命中专家、深层热点问题，以及工程监控视角 | 适合在理解机制之后阅读，用来建立“模型统计如何映射到服务指标”的意识 |

进一步阅读时，顺序建议如下：

1. 先看 Mixtral family 架构定义，搞清楚 top-2 路由到底在算什么。
2. 再看 Mixtral-8x7B 主题页，理解“分层 specialization”为什么成立。
3. 然后读 LASER，重点看 per-layer entropy、top-k mass、Iagg、M2 这些指标怎样连接到动态调度。
4. 最后看 Hugging Face 的工程综述，把“连续 token 热点”“跨层一致性”“尾延迟”这些现象放进系统视角里理解。

原始链接如下：

1. LASER 路由分析论文（OpenReview, ICLR 2026 under review）：提供 per-layer `top-k mass`、entropy、M2、Iagg，以及 Mixtral-8x7B 在 GSM8K 上 early/middle/final 的阈值示例 `0.72/0.75/0.80`。  
   https://openreview.net/pdf/d8241f7aea6825a2c63da849cc73a980faa2d6dc.pdf

2. Emergent Mind 的 Mixtral-8x7B 主题页：总结 Mixtral 的 top-2 路由、层级 specialization、语义驱动路由和注意力-专家相关性，适合理解“浅层到深层的分工变化”。  
   https://www.emergentmind.com/topics/mixtral-8x7b

3. Emergent Mind 的 Mixtral family 架构页：给出 Mixtral 路由公式与专家聚合公式，适合作为机制定义参考。  
   https://www.emergentmind.com/topics/mixtral-family-transformer-architecture

4. Hugging Face 关于 MoE 负载均衡演化的综述：讨论时间局部性、连续 token 重复命中专家、以及深层路由热点问题，对工程监控很有价值。  
   https://huggingface.co/blog/NormalUhr/moe-balance

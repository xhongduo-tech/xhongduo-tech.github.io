## 核心结论

Sparse Upcycling 的定义很直接：把已经预训练好的 dense 模型中的部分 FFN 层，也就是前馈网络层，改造成 MoE 层，也就是“多个专家里只激活少数几个”的稀疏结构，然后继续训练，而不是从零训练一个全新的 MoE。

它解决的核心问题不是“能不能做出 MoE”，而是“能不能把 dense 里已经学到的语义知识保留下来，同时引入专家分工”。结论是可以，但前提很严格：只复制 FFN 权重还不够，必须同时处理路由器初始化、专家同质化和训练后期收敛速度。

从论文结果看，这条路线的价值在于计算预算受限时的能力提升。相比继续做 dense 预训练，Sparse Upcycling 能在类似训练预算下增加模型容量，并在质量上取得明显收益；代价是推理延迟通常更高，尤其在大模型上会出现大约 40% 的额外延迟。Drop-Upcycling 进一步说明，单纯复制专家会让专家过于相似，加入“部分重初始化”后，长期收敛会明显改善，甚至能让一个激活参数量约 5.9B 的 MoE 达到 13B dense 的效果，同时只用后者约四分之一的 FLOPs。

下表可以先把三种路线放在一起看：

| Metric | Dense CPT | Sparse Upcycling | Drop-Upcycling |
| --- | --- | --- | --- |
| Training FLOPs | Baseline | 在相近预算下增大容量 | 约为同等 scratch MoE 的较小一部分 |
| Inference latency | 1x | 通常更慢，较大模型可到约 1.4x | 基本继承 MoE 推理成本 |
| Long-term convergence | 标准 | 容易因专家同质而变慢 | 通过部分重置恢复分化能力 |
| 知识继承 | 连续 | 强，直接继承 dense 权重 | 强，同时增加专家差异 |

玩具例子可以这样理解：一个 100M 参数的语言模型，原来每层只有一个 FFN。现在把前 4 层 FFN 改成 4 个专家的 MoE，每个专家先复制原 FFN 权重，再加入一个新的 router。这样模型刚开始并没有失去原来的语言能力，但已经有了“未来可以分工”的结构。

真实工程例子是 NVIDIA 的 Llama 3-8B upcycling：把 dense 的 FFN 复制成 8 个专家，采用 Top-2 路由，也就是每个 token 只送到分数最高的两个专家，再对 router 做约 100 步 warm-up，最后以远小于从零训练的代价完成 upcycling，并获得约 2% 的 MMLU 提升。这说明它不是纸上机制，而是能落到大模型训练流水线里的工程方案。

---

## 问题定义与边界

这项技术讨论的对象很明确：已经有一个可用的 dense checkpoint，希望在不丢掉原始能力的前提下，把它升级成 MoE。

这里的边界也很明确。

第一，通常替换的是 FFN，不是整个 Transformer。注意力层、词嵌入层、输出头大多保持不变。原因很简单：FFN 占据了大量参数，也是最适合拆成专家的模块；如果把注意力也一起改掉，迁移稳定性会急剧变差。

第二，Sparse Upcycling 不是“免费增益”。它继承了 dense 的知识，但也继承了 dense 的局限。复制出来的多个专家在初始时几乎一模一样，router 很容易把大量 token 挤到少数专家上，这种现象叫 routing collapse，也就是“路由塌缩”，白话说就是系统名义上有很多专家，实际只用了几个。

第三，它的收益主要发生在训练阶段的“省成本”和模型容量的“变大”，不是部署阶段的“更快”。MoE 每次虽然只激活少量专家，但涉及路由、跨设备通信、padding 和 capacity 控制，真实延迟并不会像纸面 FLOPs 那样线性下降。

可以把边界整理成一个表：

| 边界 | 含义 |
| --- | --- |
| 继承 vs 重设 | 全复制能保知识，但专家太像，往往需要部分重初始化 |
| Routing 稳定性 | router 需要 warm-up、负载均衡损失等约束 |
| 推理成本 | 部署时常见额外延迟，吞吐不一定更高 |
| 适用阶段 | 更适合继续预训练或大规模微调，不是所有小任务都值得 |

一个常见误解是：只要把 FFN 复制成多个专家，模型自然会学会分工。这个判断不成立。因为如果所有专家的初始参数完全相同，那么对同一批 token，它们算出的表示也高度相似，router 很难形成稳定偏好，训练信号也很难把它们拉开。

---

## 核心机制与推导

Sparse Upcycling 的基本机制可以拆成三步。

第一步，复制。给定 dense FFN 的参数 $W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}$，把它们复制到 $E$ 个专家中。这样每个专家一开始都具备原模型的表达能力。

第二步，加 router。router 是一个小网络，用来决定每个 token 该去哪个专家。它本质上输出一个分数向量，再取 Top-k。Top-k 的白话意思是“只让分数最高的几个专家处理这个 token”。

第三步，打破同质化。只复制会让专家长期相似，所以 Drop-Upcycling 提出部分重初始化。它不是把整层全重置，而是只随机替换一部分维度，让专家处在“半继承、半新生”的状态。公式写作：

$$
\widetilde W_{\text{type}} = I_{\mathcal S}\odot R_{\text{type}} + (1-I_{\mathcal S})\odot W_{\text{type}}, \quad
R_{\text{type}} \sim \mathcal N(\mu_{\text{type}},\sigma_{\text{type}}^2)
$$

这里 $I_{\mathcal S}$ 是掩码，也就是“哪些位置要重置”的 0/1 选择器；$\odot$ 是逐元素相乘。若某个位置被掩码选中，就用随机采样的新值替代；否则保留 dense 权重。若重置比例是 $r$，那么会有大约 $r\cdot d_f$ 个维度被替换，其中 $d_f$ 是 FFN 的中间维度。

为什么它有效，可以从训练动力学，也就是训练过程中参数如何演化，来理解。

如果所有专家完全相同，那么初始时不同专家的输出几乎相同，router 收到的梯度差异很小，专家分工形成得慢。部分重初始化后，专家在表示空间一开始就有轻微偏移，router 更容易在不同 token 上形成区分，随后梯度会进一步放大这种差异，最终形成 specialization，也就是“专家专门化”。

玩具例子如下。设一个 dense FFN 的中间维度 $d_f=8$，专家数 $E=4$，重置比例 $r=0.5$。那么每个专家保留 4 个维度的原权重，再随机替换 4 个维度。这样做的效果不是让专家“忘掉一半知识”，而是让每个专家在保留大部分功能轮廓的同时，拥有不同的优化起点。

再进一步看 router warm-up。warm-up 的意思是“先做一小段稳定启动”。如果 router 一开始完全随机，而专家又几乎一样，那么前几百步的 token 分配会非常噪声化。论文中的做法是用少量 seed 数据先训练 router，或者先让 dense 跑一小段再切换成 MoE。目标不是让 router 立刻最优，而是让它摆脱“纯随机入口状态”。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，模拟“复制 FFN -> 部分重初始化 -> 简单路由”的最小流程。它不依赖深度学习框架，但足够说明机制。

```python
import random
from copy import deepcopy

def clone_experts(dense_ffn, num_experts):
    return [deepcopy(dense_ffn) for _ in range(num_experts)]

def partial_reinit(weights, ratio, mu=0.0, sigma=0.02, seed=0):
    rng = random.Random(seed)
    new_weights = weights[:]
    k = int(len(weights) * ratio)
    idx = list(range(len(weights)))
    rng.shuffle(idx)
    chosen = set(idx[:k])
    for i in range(len(new_weights)):
        if i in chosen:
            new_weights[i] = rng.gauss(mu, sigma)
    return new_weights

def router_score(token_value, expert_weights):
    # 用一个极简分数函数模拟 router 对 expert 的偏好
    return sum(token_value * w for w in expert_weights)

def top_k_route(token_value, experts, k=2):
    scores = [(i, router_score(token_value, e)) for i, e in enumerate(experts)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:k]]

dense_ffn = [0.10, -0.20, 0.30, 0.40, -0.10, 0.05, 0.12, -0.07]
experts = clone_experts(dense_ffn, num_experts=4)

# 对每个专家做不同 seed 的部分重初始化
experts = [
    partial_reinit(e, ratio=0.5, seed=i)
    for i, e in enumerate(experts)
]

routes = top_k_route(token_value=1.0, experts=experts, k=2)

assert len(experts) == 4
assert all(len(e) == len(dense_ffn) for e in experts)
assert experts[0] != experts[1]  # 经过部分重置后，不同专家不应完全相同
assert len(routes) == 2
assert all(0 <= r < 4 for r in routes)

print("selected experts:", routes)
```

这段代码对应真实流程中的三个动作：

1. `clone_experts` 对应复制 dense FFN 到多个专家。
2. `partial_reinit` 对应 Drop-Upcycling 的部分重初始化。
3. `top_k_route` 对应 router 选择少数专家。

如果把它翻译成真实训练流水线，伪代码大致如下：

```python
for each FFN_layer in dense_checkpoint:
    experts = [copy(FFN_layer) for _ in range(num_experts)]
    for i, W in enumerate(experts):
        mask = sample_mask(r, W.shape, seed=i)
        W[mask] = sample_from_original_distribution(W, mask)

router = Router(init="random")
warmup(router, seed_loader, steps=100)
train_moe(model_with_experts, router, full_dataset, aux_loss="load_balance")
```

真实工程例子可以继续用 Llama 3-8B。把原始 dense FFN 复制成 8 个专家、用 Top-2 路由、capacity 设置为 4。capacity 可以理解为“每个专家最多接多少 token 的容量限制”。若没有它，热门专家会被塞爆；有了它，就能控制批次内的负载上限。然后对 router 做约 100 步 warm-up，再进入完整训练。这个流程的关键不是“复制”本身，而是复制后立刻让 router 和专家差异一起工作起来。

---

## 工程权衡与常见坑

Sparse Upcycling 最大的工程矛盾是：训练看起来更划算，部署不一定更划算。

第一类坑是专家过于相似。最典型症状是某几个专家负载极高，其余专家几乎闲置。监控指标通常是 expert load，也就是每个专家处理 token 的比例，以及 router entropy，也就是路由分布的离散程度。若负载长时间偏斜，说明 collapse 已经发生。

第二类坑是短期好、长期慢。很多 upcycling 实验在开始阶段会迅速超过 dense baseline，因为它立刻拥有更多参数容量；但如果专家没有真正分化，后面训练会越来越慢。Drop-Upcycling 的贡献就在这里，它通过部分重置制造差异，让长期收敛重新变快。

第三类坑是推理延迟。MoE 理论上是稀疏激活，但真实服务系统里还要处理 expert parallel、all-to-all 通信和不均匀负载。结果就是 FLOPs 下降，不等于 wall-clock latency，也就是实际响应时间下降。

可以把常见风险和规避措施列成表：

| 风险 | 规避措施 |
| --- | --- |
| Router collapse | 部分重初始化 + load balancing aux loss |
| 专家长期同质 | 每个专家使用不同 seed，控制 reinit ratio |
| 训练后期变慢 | router warm-up，持续监控 expert load |
| 推理延迟增加 | 先测吞吐与延迟，再决定 MoE 层数和 Top-k |
| 容量溢出 | 设置合理 capacity，避免热门专家拥塞 |

一个很实际的工程判断标准是：如果你的任务只是小规模领域微调，数据量不大、部署延迟又敏感，那么 upcycling 可能不值得。相反，如果你已经有一个大 dense 基座模型，计划继续喂大量 token，希望在有限训练预算下提升容量和质量，那么它就很有吸引力。

---

## 替代方案与适用边界

Sparse Upcycling 不是唯一方案，它只是“从已有 dense checkpoint 出发做 MoE”的一类代表。

CLIP-UP 说明这种思路不只适用于语言模型，也能迁移到视觉-语言模型。它保留 CLIP 原有的跨模态语义，再把部分 FFN 变成稀疏专家结构，在某些检索指标上优于 dense，同时降低推理 FLOPs。这说明 upcycling 的核心价值不是某个特定架构，而是“把已有语义表示迁移到更高容量的稀疏结构”。

UpIT 更适合 instruction tuning 场景，也就是“让模型更会遵循指令”的微调阶段。它的思路是从中间 checkpoint 或指令调优阶段出发，用少量 seed 数据让 router 先学会区分任务类型，比如问答、写作、总结，再继续做稀疏训练。

SIMoE 更进一步，把“专家”理解为不同稀疏子网络，而不是完整复制的 FFN 专家，灵活度更高，但实现与训练都更复杂。

对比表如下：

| 方法 | 适用范围 | 关键差异 |
| --- | --- | --- |
| Sparse Upcycling | 继续预训练、dense 转 MoE | 直接复制 FFN，快速继承能力 |
| Drop-Upcycling | 长周期稀疏训练 | 用部分重初始化解决专家同质化 |
| CLIP-UP | 多模态 CLIP | 证明 upcycling 不局限于纯语言模型 |
| UpIT | Instruction tuning | 从指令阶段 checkpoint 出发做专家化 |
| SIMoE | 自动专家发现 | 专家是稀疏子集，结构更灵活 |

适用边界可以一句话概括：如果你最在意的是“训练预算内把 dense 升级成更大容量的模型”，Sparse Upcycling 很合适；如果你最在意的是“线上低延迟推理”，那它往往不是首选。

---

## 参考资料

| Title | Source |
| --- | --- |
| Sparse Upcycling: Inference Inefficient Finetuning (ENLSP-IV 2024) | https://proceedings.mlr.press/v262/doubov24a.html |
| Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization (ICLR 2025) | https://proceedings.iclr.cc/paper_files/paper/2025/file/d24b7366d714b09a977946ef0d9bf3ad-Paper-Conference.pdf |
| Llama 3 Meets MoE: Efficient Upcycling | https://chatpaper.com/chatpaper/paper/90245 |
| Drop-Upcycling overview and partial re-init details | https://www.emergentmind.com/topics/drop-upcycling |
| CLIP-UP: A Sparse Upcycling Recipe | https://aclanthology.org/2025.findings-emnlp.1156/ |
| UpIT / Automatic Expert Discovery | https://www.emergentmind.com/articles/2410.01610 |

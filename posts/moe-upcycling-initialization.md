## 核心结论

Upcycling 初始化的核心价值，是把已经训练过的 Dense 模型继续利用，而不是把它丢掉后重新训练一个 MoE。MoE 是 Mixture of Experts，中文通常译作“专家混合”。它的含义不是“整个网络都变复杂”，而是“把原来一层里的单个 FFN 扩成多个 FFN 专家，但每个 token 在一次前向里只激活其中少数几个”。这样做的结果是：模型总参数可以明显增大，但单次计算量不必按总参数线性增长。

Upcycling 的具体做法，是把 Dense 模型里的 FFN 复制成多个专家的初始权重；新增的 router，也就是“决定每个 token 该送到哪些专家”的小网络，再单独随机初始化。这里的 FFN 指 Transformer 中负责逐 token 非线性映射的前馈网络，常见形式是两层线性层加激活函数。router 不负责“加工 token”，只负责“分配 token”。

它的工程结论很明确：如果手里已经有一个训练好的 Dense checkpoint，那么先复制 FFN，再训练 router 和专家分工，通常比“继续把 Dense 往下训”更划算，也比“从头训练等价 MoE”便宜得多。公开文献里常见的结论是：额外训练只需原计划约 40% 到 60% 的预算，就能超过继续训练 Dense；如果和从头训练等价 MoE 相比，额外训练 FLOPs 经常还能再省 40% 到 70%。这些数字不是普适常数，但它们足够说明一个事实：Dense checkpoint 是资产，不是历史包袱。

一个具体背景例子是 T5-Large 一类模型。设某层 FFN 的中间维度为 $d_f=4096$，把这层改成 $E=4$ 个专家，每个专家都直接复制原 Dense FFN 的权重，router 用 $\mathcal N(0,0.02)$ 初始化，训练时只激活 Top-2 专家，并加入负载均衡损失。这样做的关键，不是“专家一开始就会分工”，而是“先保住原模型能力，再用少量训练让专家慢慢分化”。对新手来说，这里最重要的认识是：Upcycling 的优势来自初始化质量，而不是来自初始化时的专家差异。

| 方案 | 是否复用 Dense 权重 | 额外 FLOPs | 早期收敛 | 风险点 |
|---|---:|---:|---|---|
| Dense 继续训练 | 是 | 1.0x | 稳定 | 上限受 Dense 结构限制 |
| Upcycling 到 MoE | 是 | 0.4x 到 0.6x | 通常更快 | 专家同质化、router 崩塌 |
| 从头训练 MoE | 否 | 1.0x 甚至更高 | 慢 | 冷启动成本高 |

再压缩成一句工程判断：当目标是“在不重做整轮预训练的前提下，把已有模型扩成更大容量的稀疏模型”时，Upcycling 往往是最优先考虑的路径。

---

## 问题定义与边界

问题定义可以写成一句话：在不丢失已有 Dense 能力的前提下，用最少新增计算把模型改造成 MoE，并让专家真正学会分工。

这句话里有三个关键词，必须先解释清楚。

第一，“不丢失已有 Dense 能力”不是指一个 step 都不退化，而是指初始化后模型函数仍然接近原模型，训练前期不会像从头训练那样完全冷启动。第二，“最少新增计算”强调的是增量成本，而不是总成本。Dense 已经训练过的那部分预算，不该在比较中被忽略。第三，“专家真正学会分工”意味着不同专家长期接收不同类型 token，并因此形成不同参数更新轨迹，而不是表面上变成了 MoE，实际上所有 token 仍主要走同一两个专家。

这里有几个边界不能忽略。

第一，Upcycling 依赖已训练 checkpoint。没有 Dense 模型，就不存在“复制已有知识”这一步。没有 checkpoint 时讨论的是“从头训练 MoE”，不是 Upcycling。

第二，通常不会把所有层都改成 MoE，而是优先改部分 FFN 层，因为 FFN 是参数和计算都很集中的位置。对多数 Transformer，attention 决定信息交互路径，FFN 决定逐 token 的大部分参数容量，所以把 FFN 稀疏化最划算。

第三，专家数量不能无约束地增大。专家越多，理论容量越大，但通信、路由、负载均衡、显存布局和 token dispatch 都会更复杂。工程上常见的起点是 4、8、16 个专家，而不是上来就把每层扩成几十上百个专家。

第四，Upcycling 默认回答的是“继续预训练或大规模领域适配”的问题，不是“小数据下游任务微调”的问题。如果你的目标只是让模型适配某个窄任务，那么 LoRA、Adapter 或全参微调通常更直接。

最常见的问题是专家同质化。因为所有专家一开始都是同一份 FFN 的拷贝，所以如果 router 也没有足够扰动，各专家会收到近似相同的梯度，最后容易出现“某一个专家吸走大部分 token”的崩塌现象。这个现象通常叫 router collapse，意思是“路由器没有学到分工，而是把流量挤到少数专家”。

一个玩具例子可以说明这个问题。假设有一个 4 维输入、隐藏层宽度为 4 的小 MLP，现在把它复制成 2 个专家。若两个专家参数完全相同，router 初始输出也非常接近，那么同一个 token 在两个专家上的得分几乎一样，反向传播后两个专家更新方向也几乎一样。结果不是“形成分工”，而是“继续保持相同”。这就是为什么 Upcycling 不是“复制完就结束”，而是“复制后必须主动打破对称性”。

负载均衡通常用辅助损失约束：

$$
\mathcal L_{\rm aux}=\alpha N\sum_i f_i P_i
$$

其中 $N$ 是专家数，$f_i$ 是专家 $i$ 实际接收 token 的比例，$P_i$ 是 router 对专家 $i$ 的平均概率。直观理解是：如果某些专家拿走了太多 token，这个损失就会增大，训练会被推回更均衡的分配。对新手来说，可以把它理解成“给路由器加一个额外规则，不允许它长期只偏爱少数专家”。

为了避免边界混淆，下面再给一张表。

| 维度 | Upcycling 默认设定 | 不在本文重点覆盖的情况 |
|---|---|---|
| 起点 | 已训练 Dense checkpoint | 完全随机初始化 |
| 改造对象 | 主要是 FFN 层 | 大改 attention 主干 |
| 训练目标 | 继续预训练、领域继续训练 | 小样本下游任务微调 |
| 关键挑战 | 专家分化、路由均衡 | 全新架构搜索 |
| 主要收益 | 节省增量训练预算 | 追求绝对最长期上限 |

---

## 核心机制与推导

Upcycling 的机制分成两部分：保留旧知识，打破新结构的对称性。

先从标准 Dense FFN 写起。对输入 token 表示 $x\in\mathbb R^{d}$，一层 FFN 常写成：

$$
\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2
$$

其中 $W_1\in\mathbb R^{d_f\times d}$，$W_2\in\mathbb R^{d\times d_f}$，$d_f$ 是 FFN 的中间维度，$\phi$ 是激活函数，比如 ReLU、GELU 或 SwiGLU 里的非线性部分。Dense 情况下，每个 token 都经过同一个 FFN。

MoE 版本把这个“单个 FFN”换成“多个专家 FFN”。设专家数为 $E$，router 对 token $x$ 产生长度为 $E$ 的打分向量 $z(x)$。经过 softmax 后得到路由概率：

$$
p_i(x)=\frac{\exp(z_i(x))}{\sum_{j=1}^{E}\exp(z_j(x))}
$$

如果采用 Top-$k$ 路由，那么只保留概率最大的 $k$ 个专家。记集合为 $\mathcal T_k(x)$，则该 token 的 MoE 输出可写成：

$$
\mathrm{MoE}(x)=\sum_{i\in \mathcal T_k(x)} \hat p_i(x)\,\mathrm{FFN}_i(x),
\qquad
\hat p_i(x)=\frac{p_i(x)}{\sum_{j\in \mathcal T_k(x)}p_j(x)}
$$

这条式子非常关键。它说明 MoE 不是“随机挑一个专家”，而是“先打分，再选前 $k$ 个，再把这些专家的输出按权重加权求和”。

### 1. 复制 FFN 以保留旧知识

设原来的 Dense FFN 权重为 $W_1,W_2,b_1,b_2$，现在有 $E$ 个专家，那么最基本的 Upcycling 初始化就是：

$$
W_1^{(1)}=\cdots=W_1^{(E)}=W_1,\qquad
W_2^{(1)}=\cdots=W_2^{(E)}=W_2
$$

偏置同理复制。这样做的直接结果是：如果 router 在初始化时对多个专家给出近似相同的权重，那么 MoE 层在函数上会非常接近原来的 Dense 层。也就是说，模型不会因为结构从 Dense 改成 MoE 而立刻失去原能力。

对新手来说，可以用一个最小数值例子理解这点。假设原 FFN 对某个 token 输出 3.6，现在复制成 4 个完全相同的专家。若 Top-2 选中了专家 1 和专家 3，权重分别是 0.7 和 0.3，那么输出仍是：

$$
0.7\times 3.6+0.3\times 3.6=3.6
$$

这就是“先保能力”的数学原因。

### 2. 为什么复制后不会永远相同

复制 FFN 只能解决“起点不要太差”，不能解决“专家如何产生分工”。专家后续能分化，靠的是不同专家接收不同 token，于是收到不同梯度。对专家 $i$，一次参数更新可近似写成：

$$
W_i^{(t+1)}=W_i^{(t)}-\eta\sum_{x\in \mathcal B_i^{(t)}} \hat p_i(x)\nabla_{W_i}\mathcal L(x)
$$

其中 $\eta$ 是学习率，$\mathcal B_i^{(t)}$ 表示第 $t$ 步中被送到专家 $i$ 的 token 集合。只要不同专家对应的集合不同，它们的梯度和更新方向就会不同。专家分工不是手工写死的，而是这种“不同 token 子集导致不同梯度轨迹”的结果。

### 3. router 归一化为什么有用

Top-$k$ 路由很容易过早变尖锐。一个常见做法是先对 router logit 做标准化，再做 softmax：

$$
\tilde z=\lambda\frac{z-\mu}{\sigma+\varepsilon}, \quad g=\mathrm{softmax}(\tilde z)
$$

其中 $\mu,\sigma$ 是当前 token 对所有专家分数的均值和标准差，$\lambda$ 是缩放系数，$\varepsilon$ 是防止除零的小常数。它的作用是让不同 token 的路由分数处在更稳定的尺度上，避免某些 token 因 logit 整体偏大而过早形成极端路由。

可以把它理解成：先把“专家打分”拉到可比较的标准尺子上，再做竞争。没有这一步时，router 很容易出现“数值上很自信，但分工上并不健康”的情况。

### 4. Drop-Upcycling 如何打破完全对称

只复制 FFN 有一个明显问题：所有专家完全相同，分化信号出现得慢。Drop-Upcycling 的做法是只对部分参数重新采样：

$$
\widetilde W^{(i)}=I_{\mathcal S}\odot R+(1-I_{\mathcal S})\odot W
$$

其中 $I_{\mathcal S}$ 是二值掩码，表示哪些坐标被替换，$R$ 是随机初始化的新参数，$W$ 是 Dense 权重。直观上，它等于“保留大部分旧知识，只在少量位置制造专家之间的差异”。

一个简单例子：某个专家权重矩阵有 8 列，随机挑 2 列重采样，其余 6 列保留原权重。这样改动很小，但已经足够让不同专家对同一输入产生略有差异的输出，router 也因此更容易学到“哪些 token 更适合送到哪个专家”。

### 5. 负载均衡辅助损失为什么必要

没有辅助损失时，Top-$k$ 路由很容易出现“富者愈富”。某个专家早期多拿几个 token，它就会拿到更多梯度；更多梯度又会让它更容易在下一轮被选中。于是流量越来越集中，最后形成 router collapse。

辅助损失：

$$
\mathcal L_{\rm aux}=\alpha N\sum_i f_iP_i
$$

这里有两个量要区分。

$f_i$ 是硬分配后的负载比例，表示“第 $i$ 个专家实际上吃到了多少 token”。

$P_i$ 是软概率的平均值，表示“router 在概率上平均有多偏爱第 $i$ 个专家”。

如果某个专家既在概率上被偏爱，又在实际分配中吃掉大量 token，那么 $f_iP_i$ 就会变大，整体损失随之上升。训练因此被推向更均衡的解。

### 6. 一个完整 toy 例子

设 4 个专家，对某个 token 的 router 原始打分为：

$$
z=[1.2,\,0.9,\,-0.1,\,0.0]
$$

先做标准化，假设得到：

$$
\tilde z=[1.15,\,0.73,\,-1.02,\,-0.86]
$$

再 softmax，可得近似概率：

$$
p=[0.48,\,0.32,\,0.06,\,0.14]
$$

若取 Top-2，则保留专家 1 和专家 2，归一化后权重为：

$$
\hat p=[0.60,\,0.40]
$$

如果这两个专家输出分别是 5.0 和 4.0，那么最终输出就是：

$$
0.60\times 5.0 + 0.40\times 4.0 = 4.6
$$

初始化早期，若两个专家本来就是同一个 Dense FFN 的拷贝，那么它们可能都输出 4.7，最终结果仍接近原模型。继续训练后，如果代码 token 更常流向专家 1、数学 token 更常流向专家 2，那么二者参数就会逐渐拉开。

把前面的机制压缩成一张表：

| 机制 | 公式/做法 | 解决的问题 | 调参要点 |
|---|---|---|---|
| FFN 复制 | $W^{(1)}=\cdots=W^{(E)}=W$ | 保住 Dense 已有能力 | 全量复制最稳 |
| Router 归一化 | $\tilde z=\lambda(z-\mu)/(\sigma+\varepsilon)$ | 稳定路由尺度 | $\lambda$ 过大易尖锐 |
| Drop-Upcycling | 部分参数重采样 | 打破专家完全相同 | 重采样比例宜小到中等 |
| Aux Loss | $\mathcal L_{\rm aux}=\alpha N\sum_i f_iP_i$ | 防止流量集中 | $\alpha$ 过高会伤主任务 |

---

## 代码实现

实现上最重要的，不是把 MoE 写复杂，而是把初始化、路由、监控三个环节写清楚。很多工程框架已经支持从 Dense checkpoint 复制到专家，例如 NVIDIA Megatron Core 文档中的 `--moe-use-upcycling`。它的本质逻辑并不复杂：加载 Dense FFN，复制到每个专家，router 单独随机初始化，然后在训练中计算 Top-$k$ 路由和辅助损失。

下面给出一个可运行的纯 Python 玩具实现。它不依赖 `torch`、`numpy` 或其他三方库，直接用标准库演示五件事：

1. 复制 Dense FFN 到多个专家。
2. 对部分权重做 Drop-Upcycling 扰动。
3. 计算归一化 router logits。
4. 执行 Top-$k$ 路由和专家前向。
5. 统计辅助损失和专家利用率。

```python
import math
import random
from typing import List, Tuple


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def normalize_logits(logits: List[float], scale: float = 1.0, eps: float = 1e-6) -> List[float]:
    mu = sum(logits) / len(logits)
    var = sum((x - mu) ** 2 for x in logits) / len(logits)
    std = math.sqrt(var + eps)
    return [scale * (x - mu) / std for x in logits]


def topk_gate(probs: List[float], k: int) -> Tuple[List[float], List[int]]:
    if k <= 0 or k > len(probs):
        raise ValueError("invalid k")
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    masked = [probs[i] if i in idx else 0.0 for i in range(len(probs))]
    s = sum(masked)
    gated = [x / s if s > 0 else 0.0 for x in masked]
    return gated, idx


class DenseFFN:
    def __init__(self, w1: List[List[float]], b1: List[float], w2: List[List[float]], b2: List[float]):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x: List[float]) -> List[float]:
        h = []
        for row, bias in zip(self.w1, self.b1):
            h.append(relu(sum(a * b for a, b in zip(row, x)) + bias))

        y = []
        for row, bias in zip(self.w2, self.b2):
            y.append(sum(a * b for a, b in zip(row, h)) + bias)
        return y


def clone_matrix(mat: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in mat]


def clone_vector(vec: List[float]) -> List[float]:
    return vec[:]


def drop_upcycling_copy(
    dense_ffn: DenseFFN,
    num_experts: int,
    drop_ratio: float = 0.1,
    noise_std: float = 0.02,
) -> List[DenseFFN]:
    experts = []
    for _ in range(num_experts):
        w1 = clone_matrix(dense_ffn.w1)
        b1 = clone_vector(dense_ffn.b1)
        w2 = clone_matrix(dense_ffn.w2)
        b2 = clone_vector(dense_ffn.b2)

        # 对 w1 和 w2 的少量位置重新采样，制造专家差异
        for mat in (w1, w2):
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if random.random() < drop_ratio:
                        mat[i][j] = random.gauss(0.0, noise_std)

        experts.append(DenseFFN(w1, b1, w2, b2))
    return experts


class Router:
    def __init__(self, input_dim: int, num_experts: int, init_std: float = 0.02):
        self.weight = [
            [random.gauss(0.0, init_std) for _ in range(input_dim)]
            for _ in range(num_experts)
        ]
        self.bias = [0.0 for _ in range(num_experts)]

    def logits(self, x: List[float]) -> List[float]:
        out = []
        for row, bias in zip(self.weight, self.bias):
            out.append(sum(a * b for a, b in zip(row, x)) + bias)
        return out


def moe_forward(
    x: List[float],
    experts: List[DenseFFN],
    router: Router,
    top_k: int,
    logit_scale: float = 1.5,
) -> Tuple[List[float], List[float], List[int], List[List[float]]]:
    raw_logits = router.logits(x)
    norm_logits = normalize_logits(raw_logits, scale=logit_scale)
    probs = softmax(norm_logits)
    gated, chosen = topk_gate(probs, top_k)

    expert_outputs = [expert.forward(x) for expert in experts]

    out_dim = len(expert_outputs[0])
    y = [0.0] * out_dim
    for expert_id, gate in enumerate(gated):
        if gate == 0.0:
            continue
        for i in range(out_dim):
            y[i] += gate * expert_outputs[expert_id][i]

    return y, probs, chosen, expert_outputs


def aux_loss(assignments: List[List[int]], router_probs: List[List[float]], alpha: float = 0.01) -> float:
    num_tokens = len(assignments)
    num_experts = len(router_probs[0])

    f = [0.0] * num_experts
    P = [0.0] * num_experts

    for chosen in assignments:
        for i in chosen:
            f[i] += 1.0 / (num_tokens * len(chosen))

    for probs in router_probs:
        for i, p in enumerate(probs):
            P[i] += p / num_tokens

    return alpha * num_experts * sum(fi * pi for fi, pi in zip(f, P))


def utilization(assignments: List[List[int]], num_experts: int) -> List[int]:
    counts = [0] * num_experts
    for chosen in assignments:
        for i in chosen:
            counts[i] += 1
    return counts


def main() -> None:
    random.seed(7)

    dense = DenseFFN(
        w1=[
            [0.20, -0.10, 0.30, 0.05],
            [0.40, 0.10, -0.20, 0.30],
            [-0.30, 0.20, 0.10, 0.40],
        ],
        b1=[0.01, -0.02, 0.03],
        w2=[
            [0.50, -0.20, 0.10],
            [-0.10, 0.30, 0.20],
        ],
        b2=[0.00, 0.05],
    )

    experts = drop_upcycling_copy(
        dense_ffn=dense,
        num_experts=4,
        drop_ratio=0.15,
        noise_std=0.02,
    )
    router = Router(input_dim=4, num_experts=4, init_std=0.02)

    tokens = [
        [0.8, 0.1, -0.2, 0.5],
        [0.2, 0.6, 0.3, -0.1],
        [-0.4, 0.9, 0.1, 0.2],
        [0.7, -0.3, 0.4, 0.6],
    ]

    router_probs = []
    assignments = []

    for token_id, x in enumerate(tokens):
        y, probs, chosen, expert_outputs = moe_forward(
            x=x,
            experts=experts,
            router=router,
            top_k=2,
            logit_scale=1.5,
        )
        router_probs.append(probs)
        assignments.append(chosen)

        print(f"token {token_id}")
        print("  probs   =", [round(p, 4) for p in probs])
        print("  chosen  =", chosen)
        print("  output  =", [round(v, 4) for v in y])

        # 检查每个专家都能正常前向
        assert len(expert_outputs) == 4
        assert len(y) == 2
        assert abs(sum(topk_gate(probs, 2)[0]) - 1.0) < 1e-6

    loss = aux_loss(assignments, router_probs, alpha=0.01)
    counts = utilization(assignments, num_experts=4)

    print("aux_loss =", round(loss, 6))
    print("usage    =", counts)

    assert loss >= 0.0
    assert sum(counts) == len(tokens) * 2
    assert any(experts[0].w1[i][j] != dense.w1[i][j] for i in range(len(dense.w1)) for j in range(len(dense.w1[0])))


if __name__ == "__main__":
    main()
```

这段代码的目的不是复现真实大模型训练，而是把 Upcycling 的初始化和路由逻辑拆到最小可验证单元。你可以直接运行它，观察三个现象。

第一，所有专家都从同一个 Dense FFN 开始，但由于少量重采样，不再完全相同。第二，router 是单独随机初始化的，不继承 Dense 参数。第三，最终输出只来自 Top-2 专家，而不是所有专家。

如果换成真实框架，伪代码通常就是下面这样：

```python
for expert in experts:
    expert.w1.copy_(dense_ffn.w1)
    expert.b1.copy_(dense_ffn.b1)
    expert.w2.copy_(dense_ffn.w2)
    expert.b2.copy_(dense_ffn.b2)

    if drop_upcycling:
        mask1 = torch.rand_like(expert.w1) < drop_ratio
        mask2 = torch.rand_like(expert.w2) < drop_ratio
        expert.w1[mask1] = torch.randn_like(expert.w1[mask1]) * noise_std
        expert.w2[mask2] = torch.randn_like(expert.w2[mask2]) * noise_std

router.weight.normal_(mean=0.0, std=0.02)
router.bias.zero_()
```

真实工程例子可以参考 “Llama 3 Meets MoE: Efficient Upcycling” 和 Megatron Core 的 Upcycling 文档。它们展示的流程都类似：载入已有 Dense checkpoint，把若干 FFN 层替换为专家层，再用较小额外预算继续训练。这个场景下，重点监控的不是单个 batch 的 loss，而是专家利用率、每个 batch 的 token 分布、router 概率熵，以及 aux loss 是否进入“看起来很小但其实已经失效”的区间。

| 参数 | 常见取值 | 作用 | 过大或过小的风险 |
|---|---|---|---|
| `num_experts` | 4, 8, 16 | 控制总容量 | 过大时通信和负载均衡变差 |
| `top_k` | 1 或 2 | 控制每个 token 激活几个专家 | `1` 更省算力但更易崩塌 |
| `router_init_std` | 0.01 到 0.02 | router 初始扰动强度 | 过小难分化，过大不稳定 |
| `drop_ratio` | 0.01 到 0.1 | Drop-Upcycling 比例 | 过低差异不足，过高破坏旧知识 |
| `aux_weight` | 0.001 到 0.01 | 负载均衡损失权重 | 过高会拖慢主任务 |

---

## 工程权衡与常见坑

第一类坑是专家完全一致，导致 router collapse。现象通常是训练前几百步里，几乎所有 token 都打到专家 0 或专家 1。这个时候不要先怀疑数据，先检查是否所有专家都只是“纯复制”，没有任何扰动；再检查 router 初始化是否过小，导致所有 token 的分数差异都被压扁。

第二类坑是扰动太大。Upcycling 的目标是保住旧知识后再分工，不是把旧 FFN 直接打散。如果把大量参数重新初始化，模型会退化成“半个从头训练”，这会抵消 Upcycling 的优势。对新手来说，一个简单判断是：如果改完结构后验证集指标明显大幅回退，而且几千步内都恢复不了，通常不是“专家正在分化”，而是“初始化已经破坏了原模型函数”。

第三类坑是只看总 loss，不看专家利用率。MoE 训练是否健康，必须同时看每个专家被分配的 token 比例、router 概率熵、Top-$k$ 后的流量分布。如果总 loss 在下降，但 80% 的 token 长期只进两个专家，那么实际上没有学到有效的专家分工，模型只是把更多参数堆在了纸面上。

第四类坑是 router 过尖锐。常见表现是 softmax 后某个专家概率长期接近 1，其余专家接近 0。这样做看上去“决策很果断”，但从训练角度看，它会让大量专家拿不到梯度。解决方法一般包括：降低 router 温度、开启 logit 归一化、加入噪声探索、适当提高 aux loss 权重。

第五类坑是容量与路由设置不匹配。比如专家数较多，但 capacity factor 太小，结果很多 token 到了热门专家后被丢弃或强制重路由。此时模型并不是“训练不动”，而是“路由规则和容量约束打架”。

一个典型新手场景是：训练初期几乎只激活专家 0，aux loss 很快下降，但验证集没有提升。这往往不是“已经收敛”，而是 router 进入单专家路径。处理办法通常是三步：给每个专家少量列加 $\mathcal N(0,0.01)$ 级别噪声；适度提高 aux loss 权重；检查 router 归一化是否开启，以及 Top-1 是否应该先换成 Top-2。

下面这张表适合直接作为排障索引。

| 常见坑 | 典型现象 | 更可能的原因 | 规避策略 | 监控指标 |
|---|---|---|---|---|
| Router 崩塌 | 少数专家吃掉大多数 token | 纯复制且 router 扰动太弱 | 加噪声、Drop-Upcycling、增大 aux | 专家负载直方图 |
| 负载过度集中 | Top-$k$ 长期固定 | logit 过尖、温度不合适 | 归一化、调温度、加探索噪声 | 平均路由熵 |
| 扰动过强 | 初始性能明显回退 | 重采样比例过高 | 降低 `drop_ratio` 和噪声标准差 | 冷启动验证集指标 |
| Aux 权重过高 | 主任务变慢 | 路由均衡目标压过主损失 | 减小 $\alpha$ | 主 loss 与 aux loss 比值 |
| 容量设置不当 | token 被丢弃或重路由过多 | `capacity_factor` 过小 | 增大容量或减少专家竞争 | token drop rate |

工程上还有一个常被忽略的权衡：Upcycling 改善的是“增量训练效率”，不是“任何部署场景下都更便宜”。如果服务端是严格的低延迟推理环境，额外的路由开销、专家 dispatch 和并行通信也要算进去。因此 Upcycling 值不值得做，最终仍要看训练预算、推理约束和模型用途三者是否匹配。

---

## 替代方案与适用边界

Upcycling 不是所有场景都最优，它只是在“已有 Dense checkpoint，且希望低成本扩容”时很强。

如果没有现成 checkpoint，直接从头训练 MoE 更自然，因为你不需要背负 Dense 结构留下的初始化约束。如果预算非常充足，而且目标是更大规模的专家体系，从头训练也更容易把结构、并行方式和路由机制一次性设计到位。它的代价是前期更慢，冷启动更重，训练稳定性通常也更难处理。

如果目标只是低成本做任务适配，而不是把基础模型扩成稀疏架构，那么 LoRA 往往更合适。LoRA 是低秩适配，意思是“冻结原模型大部分参数，只训练少量低秩增量矩阵来改模型行为”。它对算力和显存更友好，但它解决的是“便宜地改行为”，不是“把 Dense 结构扩成更大容量的 MoE”。

还有一类替代路线是参数高效的稀疏改造方法，比如在指令调优阶段做稀疏化，或通过参数合并、adapter 化 MoE 来降低改造成本。它们适合“想做 MoE 化，但不愿承担完整继续预训练”的情形。不过这类方案通常更偏任务适配，不如标准 Upcycling 那样强调“继续利用 Dense 预训练沉没成本”。

一个简单决策规则是：

| 条件 | 更合适的方案 | 原因 |
|---|---|---|
| 已有高质量 Dense checkpoint，预算只能中等增加 | Upcycling | 能把已有预训练成本直接转成 MoE 起点 |
| 没有 checkpoint，准备训练新基座模型 | 从头训练 MoE | 无需受 Dense 结构约束 |
| 只做下游任务微调，算力和显存都紧 | LoRA | 目标是便宜适配，不是结构扩容 |
| 想做稀疏化但训练预算很有限 | 参数高效稀疏改造 | 牺牲部分上限，换更低改造成本 |

再从工程视角压缩一次：

| 方案 | 依赖已有 checkpoint | 额外算力 | 适合场景 | 局限 |
|---|---:|---:|---|---|
| Upcycling | 是 | 中等 | 已有 Dense，想快速扩成 MoE | 需要处理路由和分化稳定性 |
| 从头训练 MoE | 否 | 高 | 新模型预训练，预算充足 | 冷启动贵，早期收敛慢 |
| LoRA | 是 | 低 | 下游微调，资源受限 | 不提供 MoE 结构容量扩展 |
| 参数高效稀疏改造 | 通常是 | 低到中 | 指令调优、轻量改造 | 更偏任务适配，通用性有限 |

结论可以写得很直接：如果你的任务本质上是“继续做基座训练，但希望在相近单次计算下获得更高容量”，优先看 Upcycling；如果你的任务本质上是“便宜微调”，优先看 LoRA；如果你的任务本质上是“设计下一代大规模稀疏基座”，再考虑从头训练 MoE。

---

## 参考资料

| 标题 | 作者/平台 | 重点内容 | 访问指引 |
|---|---|---|---|
| [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055) | Komatsuzaki 等，ICLR 2023 | Upcycling 的原始定义、Dense 到 MoE 的复制初始化、T5 与 ViT 的额外预算对比 | 先读摘要、实验表和初始化方法 |
| [Mixture of Experts: Upcycling](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html#upcycling) | NVIDIA Megatron Core 文档 | 工程实现层面的 `--moe-use-upcycling`、granular upcycling、与并行配置的关系 | 适合落地前确认实现接口 |
| [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) | Zoph 等，Google Research | 负载均衡、路由稳定性、MoE 训练常见不稳定来源 | 适合补稳定训练背景 |
| [Llama 3 Meets MoE: Efficient Upcycling](https://arxiv.org/abs/2412.09952) | Vavre 等，2024 | 用 Llama 3-8B 做 8-expert Top-2 Upcycling 的工程案例，强调低额外计算预算 | 适合看现代 LLM 场景下的落地方式 |
| [Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization](https://arxiv.org/abs/2502.19261) | Nakamura 等，2025 | 为什么“纯复制专家”后期会变慢，以及局部重初始化如何改善专家分化 | 适合深入看专家同质化问题 |
| [Parameter-Efficient Sparsity Crafting from Dense to Mixture-of-Experts for Instruction Tuning on General Tasks](https://aclanthology.org/2024.emnlp-main.43/) | Wu 等，EMNLP 2024 | 参数高效地把 Dense 改造成稀疏模型，偏指令调优场景 | 适合和标准 Upcycling 做边界比较 |

这些资料的共同价值在于：第一类解释为什么 Upcycling 在训练计算上划算；第二类给出打破专家同质化和稳定路由的机制；第三类提供工程落地案例，说明它不是只在论文里成立的技巧。

如果只打算读三份，推荐顺序是：

1. `Sparse Upcycling`：先建立问题定义和主结论。
2. `Megatron Core 文档`：再看真实框架如何落地。
3. `Drop-Upcycling` 或 `Llama 3 Meets MoE`：最后补“为什么纯复制不够”或“现代大模型里如何做”。

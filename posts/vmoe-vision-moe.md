## 核心结论

V-MoE，Vision Mixture of Experts，意思是把视觉 Transformer 里最重的前馈子层拆成多个专家网络，但每个 patch token 在一次前向中只激活少数专家。它不是重写整套 ViT，而是在保留自注意力主干的前提下，把部分 MLP 层替换成稀疏 MoE 层。

结论可以压缩成三点：

1. 在每个 token 只激活少数专家的前提下，V-MoE 能显著增大参数规模，而单次推理 FLOP 不需要按参数量同步增长。
2. 这种条件计算把“模型容量”和“实际计算”部分解耦。参数库存可以很大，但一次前向只调用其中很小一部分。
3. 推理阶段再配合 Batch Prioritized Routing，简称 BPR，即“高置信 token 优先占用专家容量”，可以继续压缩 FLOP，并把精度下降控制在较小范围内。

先看一个最小对比表：

| 方法 | 参数规模 | 单次 FLOP | 每个 patch 的处理方式 | 扩展性 |
| --- | --- | --- | --- | --- |
| 标准 ViT | 随层宽和层数上升 | 所有参数都参与 | 所有 patch 走同一套 MLP | 参数一变大，计算同步变贵 |
| 稀疏 MoE | 可显著增大 | 主要由 top-k 专家决定 | patch 先路由，再只走少数专家 | 参数和计算可部分解耦 |
| V-MoE | 在 ViT 中局部替换 MLP 为 MoE | 接近同级 ViT 或略高 | patch 级路由，专家级处理 | 更适合超大视觉预训练 |

再看一个玩具例子。假设某个 MoE 层有 8 个专家，每个 patch 只选 top-2。那一次前向不是 8 个专家全部计算，而是每个 patch 只进入 2 个专家，再把两个专家的输出按路由权重加权。参数规模像“有 8 家专科诊室”，但每个病人只会挂其中 2 个号。

真实结果上，论文中的 V-MoE 最多使用 24 个 MoE 层、32 个专家，总参数接近 15B；在 ImageNet 微调后达到 90.35% top-1 准确率。更重要的是，在部分推理设置下，它可以用显著低于同级 dense ViT 的 FLOP，取得接近甚至更好的效果。这个结果说明，V-MoE 的价值不在“省参数”，而在“把更多参数装进模型，但不要求每次都全部激活”。

为了避免概念混淆，再补一张判断表：

| 问题 | V-MoE 的答案 |
| --- | --- |
| 是不是整张图像选一个专家？ | 不是，通常是 patch/token 级路由 |
| 是不是注意力层也变成专家？ | 不是，主要替换 MLP 子层 |
| 是不是所有专家都会参与一次前向？ | 不是，每个 token 只选 top-k 个专家 |
| 是不是参数越大，推理一定越慢？ | 不一定，关键取决于 `k`、容量和通信开销 |

---

## 问题定义与边界

先定义问题。ViT，Vision Transformer，意思是“把图像切成 patch 序列，再像处理词序列一样处理它们”。标准 ViT 的每一层通常由两部分组成：

1. 多头自注意力，负责 token 间信息交互。
2. MLP，负责对每个 token 做位置独立的非线性变换。

在大模型里，参数的大头通常来自 MLP。原因很直接。若隐藏维度为 $D$，MLP 扩展倍数为 $r$，则一个标准前馈层大致有两次线性映射：

$$
D \rightarrow rD \rightarrow D
$$

其参数量主项约为

$$
2rD^2
$$

而注意力里的投影虽然也重，但通常不会像多层大宽度 MLP 那样迅速膨胀。因此，当 ViT 继续做大时，最自然的想法不是先稀疏化注意力，而是先处理 MLP。

V-MoE 要解决的问题可以写成三句：

- 能否在不显著增加单次计算的前提下，让视觉模型拥有更大的参数容量？
- 能否让不同 patch 交给更合适的专家处理，而不是统一走同一套 MLP？
- 能否在推理阶段继续按 token 重要性调节算力，得到更平滑的精度和 FLOP 曲线？

它的边界也要写清楚：

| 维度 | V-MoE 的边界 |
| --- | --- |
| 任务类型 | 主要讨论大规模图像分类与视觉预训练 |
| 路由粒度 | patch/token 级，不是整图像级 |
| 稀疏位置 | 主要替换 ViT 中的 MLP 层，不改自注意力主干 |
| 训练前提 | 更适合大数据、大模型、多设备训练 |
| 推理前提 | 若使用 BPR，需要允许 token 排序和容量截断 |
| 收益来源 | 主要来自容量扩展，不是单层算子本身更高效 |

标准 ViT 和 V-MoE 的一层可以对照理解：

| 模块 | 标准 ViT | V-MoE |
| --- | --- | --- |
| Attention 子层 | 所有 token 共用同一组参数 | 通常保持不变 |
| MLP 子层 | 所有 token 共用同一组参数 | token 先路由，再进入少数专家 MLP |
| 参数共享方式 | 全局共享 | 专家局部共享 |
| 每个 token 的激活参数 | 全部 MLP 参数 | top-k 专家参数 |

整个流程可以写成一句“文字图”：

`patch embedding -> attention -> router 打分 -> softmax -> top-k 选专家 -> 专家 MLP -> 加权聚合 -> 残差回主干`

这里有两个经常被忽略的边界。

第一，V-MoE 优化的是 token 级信息流，而不是样本级分治。它不是“这张猫图走专家 A，那张狗图走专家 B”，而是“同一张图里的天空 patch、边缘 patch、主体 patch，可能在某一层走不同专家”。

第二，专家容量不是无限的。每个专家都有固定 buffer capacity，意思是“本轮最多接收多少 token”。当某个专家超载时，多余 token 不会进入该专家计算分支，通常只能走残差路径或被丢弃对应专家分配。于是，V-MoE 的收益成立在三个条件之上：

1. 路由器能学到有意义的分工。
2. 容量设置不会让大量 token 过早溢出。
3. 负载足够均衡，不会长期只有少数专家在工作。

对新手来说，这一节可以记成一句话：**V-MoE 不是让视觉模型“更稀奇”，而是让 ViT 中最重的 MLP 层从“所有 token 共用一套参数”变成“不同 token 选择不同专家”。**

---

## 核心机制与推导

V-MoE 的一个 MoE 层可以形式化写成：

$$
y(x)=\sum_{i=1}^{M} g_i(x)\, h_i(x)
$$

其中：

- $x$ 是某个 token 的输入表示。
- $M$ 是专家数。
- $h_i(x)$ 是第 $i$ 个专家的输出，本质是一套独立参数的 MLP。
- $g_i(x)$ 是路由权重，表示这个 token 应该分给专家 $i$ 多大比重。

如果所有 $g_i(x)$ 都非零，那仍然接近 dense 计算。V-MoE 的关键在于稀疏路由。一个常见流程是：

1. 路由器输出 logits：$z = Wx + \epsilon$
2. 对全部专家做 softmax：$p=\text{softmax}(z)$
3. 只保留 top-k 专家：$S(x)=\text{TopK}(p, k)$
4. 将未入选专家权重置零，并对入选权重做归一化或直接保留原 softmax 权重

于是输出变成：

$$
y(x)=\sum_{i\in S(x)} g_i(x)\, h_i(x), \quad |S(x)|=k
$$

### 为什么是 `softmax -> top-k`

这一步很关键。V-MoE 论文附录专门比较过路由顺序。若先做 `top-k`，再在入选专家上做 softmax，那么未入选专家在这一轮的梯度几乎处处为零，训练容易过早固化。先对全体专家做 softmax，再做 top-k，能保留更平滑的概率结构，训练更稳定。

可以把两种顺序并排看：

| 顺序 | 训练效果 | 问题 |
| --- | --- | --- |
| `top-k -> softmax` | 往往更难训练 | 未入选专家梯度极弱 |
| `softmax -> top-k` | 更稳定 | 实现稍复杂，但更可用 |

### 一个完整前向例子

假设某层有 4 个专家，某个 token 的路由 logits 为：

$$
z = [2.4,\ 0.7,\ 1.9,\ -0.3]
$$

先做 softmax，得到概率近似：

$$
p \approx [0.53,\ 0.10,\ 0.32,\ 0.05]
$$

若 `k=2`，则选中专家 0 和专家 2。若对入选权重再归一化，则新的组合权重大致为：

$$
g \approx [0.62,\ 0,\ 0.38,\ 0]
$$

这表示该 token 的前馈变换由两个专家共同完成：

$$
y(x)=0.62\,h_0(x)+0.38\,h_2(x)
$$

这和标准 ViT 的区别非常直接：

- 标准 ViT：所有 token 都走同一个 MLP。
- V-MoE：不同 token 可能走完全不同的专家组合。

### 为什么参数扩张不等于计算同步扩张

设：

- batch 中总 token 数为 $N$
- 专家数为 $M$
- 每个 token 激活 $k$
- 单个专家 MLP 的计算成本为 $F_{\text{mlp}}$

若做 dense MoE，让每个 token 都经过全部专家，则计算主项近似是：

$$
N \cdot M \cdot F_{\text{mlp}}
$$

而稀疏 top-k 路由下，计算主项近似是：

$$
\text{FLOPs} \approx N \cdot k \cdot F_{\text{mlp}} + F_{\text{router}} + F_{\text{dispatch}}
$$

当 $k \ll M$ 时，参数量随 $M$ 线性变大，但计算主项主要只随 $k$ 增长。于是得到 V-MoE 最重要的性质：

$$
\text{参数容量} \uparrow \quad \not\Rightarrow \quad \text{单次 FLOPs 同比例} \uparrow
$$

更直观地说：

| 量 | 主要受什么控制 |
| --- | --- |
| 总参数量 | 专家数 $M$、每个专家宽度 |
| 单 token 计算量 | 激活专家数 $k$ |
| 推理吞吐 | $k$、容量、通信与实现质量 |
| 精度上限 | 数据规模、模型容量、路由质量 |

### 负载平衡为什么是必要条件

如果不加约束，路由器很容易把大部分 token 都送给少数几个“热门专家”。这会导致两类退化：

1. 表达退化：大部分专家几乎不用，等于白白堆参数。
2. 工程退化：少数设备过载，其他设备空闲，吞吐显著下降。

V-MoE 使用辅助损失维持负载平衡。论文里常用两类统计量。

第一类是 importance，表示一个专家在一批样本中获得的总概率质量：

$$
I_i=\sum_{x\in \mathcal{B}} p_i(x)
$$

其均衡损失可写为变异系数平方：

$$
\mathcal{L}_{\text{importance}}=\left(\frac{\sigma(I)}{\mu(I)}\right)^2
$$

第二类是 load，表示一个专家实际被分配到的 token 数量或近似数量：

$$
L_i=\sum_{x\in \mathcal{B}} \mathbf{1}[i \in S(x)]
$$

同样定义均衡损失：

$$
\mathcal{L}_{\text{load}}=\left(\frac{\sigma(L)}{\mu(L)}\right)^2
$$

最终辅助项通常写成：

$$
\mathcal{L}_{\text{aux}}=\frac{1}{2}\left(\mathcal{L}_{\text{importance}}+\mathcal{L}_{\text{load}}\right)
$$

这两个量不能互相替代。原因是：

- `importance` 约束的是“权重总和是否均衡”。
- `load` 约束的是“实际接单量是否均衡”。

一个专家可能拿到不少概率质量，但始终卡在 top-k 边缘，实际很少被选中。因此只看 importance 不够。

### 容量公式的含义

每个专家的容量常写成：

$$
\text{capacity} = \frac{B \cdot T \cdot k}{M}\cdot C
$$

其中：

- $B$ 是 batch size。
- $T$ 是每张图的 token 数。
- $k$ 是每个 token 选择的专家数。
- $M$ 是专家总数。
- $C$ 是 capacity ratio，也叫 capacity factor。

这个公式很好理解。若所有 token 完全平均地路由到所有专家，那么平均每个专家应收到的 token 数就是：

$$
\frac{B \cdot T \cdot k}{M}
$$

但真实路由一定有波动，所以再乘一个松弛因子 $C$。例如平均应收 12 个 token，若容量设成 16，则：

$$
C=\frac{16}{12}=\frac{4}{3}
$$

这表示允许大约 33% 的超载空间。

容量太小会发生什么，可以直接看表：

| `C` 设置 | 现象 | 影响 |
| --- | --- | --- |
| `C < 1` | 容量非常紧 | 更省 FLOP，但溢出更多 |
| `C = 1` | 仅够平均负载 | 对路由波动敏感 |
| `C > 1` | 有余量 | 训练更稳，但 padding 和空槽更多 |

### BPR 为什么能进一步省计算

BPR，Batch Prioritized Routing，解决的是“容量紧张时，谁先占坑”。标准做法里，token 往往按原始顺序或局部顺序进入专家 buffer；这样即使某些 token 路由很不确定，也可能先把容量占满。

BPR 会先为每个 token 定义优先级，常见方式是取最大路由权重：

$$
s(x)=\max_i g_i(x)
$$

然后按 $s(x)$ 从大到小排序，让高置信 token 先进入容量有限的专家 buffer。于是，低置信 token 更可能在容量紧张时被跳过。

它的核心含义不是“更聪明的模型结构”，而是“更精细的算力分配策略”。

| 路由方式 | 容量紧张时谁先进入专家 | 后果 |
| --- | --- | --- |
| 普通 routing | 接近原始顺序 | 低价值 token 可能抢占容量 |
| BPR | 高置信 token 优先 | 更容易保住精度，减少无效计算 |

因此，BPR 实际上提供了一条推理期可调曲线：

$$
\text{更低容量} \Rightarrow \text{更低 FLOPs} \Rightarrow \text{可能更低精度}
$$

但因为先保留高价值 token，这条精度下降曲线通常更平缓。

---

## 代码实现

下面给一个可以直接运行的纯 Python 玩具实现。它不依赖深度学习框架，重点演示四件事：

1. 路由器如何做 `softmax -> top-k`
2. 专家容量如何限制接单
3. BPR 如何按优先级重排 token
4. 专家输出如何按路由权重加权聚合

```python
import math
from typing import List, Tuple


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def topk_indices(xs: List[float], k: int) -> List[int]:
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:k]


def renorm_selected(probs: List[float], selected: List[int]) -> List[float]:
    total = sum(probs[i] for i in selected)
    out = [0.0] * len(probs)
    for i in selected:
        out[i] = probs[i] / total if total > 0 else 0.0
    return out


def relu(vec: List[float]) -> List[float]:
    return [max(0.0, x) for x in vec]


def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [sum(w * x for w, x in zip(row, vec)) for row in mat]


class ExpertMLP:
    def __init__(self, w1: List[List[float]], w2: List[List[float]]):
        self.w1 = w1
        self.w2 = w2

    def __call__(self, x: List[float]) -> List[float]:
        hidden = relu(matvec(self.w1, x))
        return matvec(self.w2, hidden)


def route_token(token: List[float], router_w: List[List[float]], k: int):
    logits = matvec(router_w, token)
    probs = softmax(logits)
    selected = topk_indices(probs, k)
    weights = renorm_selected(probs, selected)
    priority = max(weights)
    return logits, probs, selected, weights, priority


def bpr_assign(tokens, router_w, k, capacity):
    routed = []
    for token_id, token in enumerate(tokens):
        logits, probs, selected, weights, priority = route_token(token, router_w, k)
        routed.append(
            {
                "token_id": token_id,
                "token": token,
                "logits": logits,
                "probs": probs,
                "selected": selected,
                "weights": weights,
                "priority": priority,
            }
        )

    order = sorted(
        range(len(routed)),
        key=lambda i: routed[i]["priority"],
        reverse=True,
    )

    num_experts = len(router_w)
    expert_load = [0] * num_experts
    assigned = [[] for _ in tokens]

    for idx in order:
        item = routed[idx]
        for expert_id in item["selected"]:
            if expert_load[expert_id] < capacity:
                expert_load[expert_id] += 1
                assigned[item["token_id"]].append(
                    (expert_id, item["weights"][expert_id])
                )

    return routed, order, expert_load, assigned


def combine_outputs(tokens, experts, assigned):
    outputs = []
    for token_id, token in enumerate(tokens):
        y = [0.0] * len(token)
        for expert_id, weight in assigned[token_id]:
            out = experts[expert_id](token)
            y = [a + weight * b for a, b in zip(y, out)]
        outputs.append(y)
    return outputs


if __name__ == "__main__":
    # 4 个 token，每个 token 维度 3
    tokens = [
        [1.2, 0.1, 0.0],
        [0.0, 1.0, 0.2],
        [0.2, 0.1, 1.5],
        [0.8, 0.7, 0.6],
    ]

    # router_w: [num_experts, hidden_dim]
    router_w = [
        [2.0, 0.2, 0.1],   # expert 0 偏好第一维强的 token
        [0.1, 2.0, 0.2],   # expert 1 偏好第二维强的 token
        [0.2, 0.1, 2.2],   # expert 2 偏好第三维强的 token
    ]

    experts = [
        ExpertMLP(
            w1=[[1.0, 0.0, 0.2], [0.1, 0.8, 0.0]],
            w2=[[0.7, 0.2], [0.1, 0.9], [0.2, 0.1]],
        ),
        ExpertMLP(
            w1=[[0.1, 1.1, 0.0], [0.0, 0.5, 0.6]],
            w2=[[0.6, 0.1], [0.2, 0.8], [0.3, 0.2]],
        ),
        ExpertMLP(
            w1=[[0.2, 0.1, 1.0], [0.5, 0.0, 0.7]],
            w2=[[0.8, 0.2], [0.1, 0.3], [0.2, 0.9]],
        ),
    ]

    k = 2
    capacity = 2

    routed, order, expert_load, assigned = bpr_assign(
        tokens=tokens,
        router_w=router_w,
        k=k,
        capacity=capacity,
    )
    outputs = combine_outputs(tokens, experts, assigned)

    # 基本正确性检查
    assert all(len(x) <= k for x in assigned)
    assert all(load <= capacity for load in expert_load)

    print("BPR order:", order)
    print("expert_load:", expert_load)
    print()

    for item in routed:
        tid = item["token_id"]
        print(f"token {tid}")
        print("  logits   =", [round(x, 4) for x in item["logits"]])
        print("  probs    =", [round(x, 4) for x in item["probs"]])
        print("  selected =", item["selected"])
        print("  priority =", round(item["priority"], 4))
        print("  assigned =", [(e, round(w, 4)) for e, w in assigned[tid]])
        print("  output   =", [round(x, 4) for x in outputs[tid]])
        print()
```

这段代码为什么是“可运行”的，可以逐项核对：

| 检查项 | 状态 |
| --- | --- |
| 依赖外部框架 | 不需要 |
| `softmax` 是否数值稳定 | 是，先减最大值 |
| 是否真的做了 `softmax -> top-k` | 是 |
| 是否实现容量约束 | 是 |
| 是否实现 BPR 排序 | 是，按最大路由权重排序 |
| 是否真的做了专家前向与加权聚合 | 是 |

如果想把它映射到真实深度学习代码，可以把每一步对应起来：

| 玩具代码 | 工程实现中的对应模块 |
| --- | --- |
| `router_w` | 路由器线性层 |
| `route_token` | router forward + top-k |
| `bpr_assign` | dispatch 前的排序与容量控制 |
| `ExpertMLP` | 每个专家的 FFN |
| `combine_outputs` | combine/scatter back |

再给一个更接近工程实现的伪代码：

```python
# x: [num_tokens, hidden_dim]
# router: Linear(hidden_dim -> num_experts)
# experts[e]: MLP(hidden_dim -> hidden_dim)

logits = router(x)                         # [N, M]
probs = softmax(logits, dim=-1)            # [N, M]
topk_val, topk_idx = topk(probs, k=2)      # [N, 2], [N, 2]

# 可选：对 top-k 权重再归一化
topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)

# BPR：按 max(topk_val) 排序，让高置信 token 先占容量
priority = topk_val.max(dim=-1).values
order = argsort(priority, descending=True)

# dispatch：按 expert 收集 token，并裁剪到容量
expert_batches = dispatch_with_capacity(x, topk_idx, topk_val, order, capacity)

# local expert compute
expert_outputs = []
for e in range(num_experts):
    expert_outputs.append(experts[e](expert_batches[e].tokens))

# combine：把专家输出 scatter 回原 token 位置
y = combine(expert_outputs, expert_batches)
```

张量形状再补充一遍：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[N, D]` | N 个 token 的表示 |
| `logits` | `[N, M]` | 每个 token 对 M 个专家的打分 |
| `topk_idx` | `[N, k]` | 每个 token 选中的专家编号 |
| `topk_val` | `[N, k]` | 每个 token 对应的 top-k 权重 |
| `expert_in[e]` | `[N_e, D]` | 分配给专家 `e` 的 token 子批次 |
| `expert_out[e]` | `[N_e, D]` | 专家 `e` 的输出 |
| `y` | `[N, D]` | 聚合后的 token 表示 |

这一节真正要记住的不是代码细节，而是一个事实：**路由公式往往不难，难的是 dispatch、capacity 和跨设备通信。**

---

## 工程权衡与常见坑

V-MoE 真正容易出问题的地方，不是公式本身，而是“训练可用性”和“系统可运行性”同时失效。下面这张表把最常见的坑集中列出来：

| 常见坑 | 现象 | 为什么会出问题 | 规避方式 |
| --- | --- | --- | --- |
| 先 `top-k` 再 `softmax` | 路由学不动，尤其在 `k>1` 时更明显 | 未入选专家梯度几乎处处为 0 | 用 `softmax -> top-k` |
| 不加负载平衡损失 | 少数专家爆满，其他专家长期闲置 | 路由器会依赖少数强专家 | 加 `importance` 和 `load` 辅助损失 |
| 容量比 `C` 太小 | 大量 token 被溢出 | 路由一有波动就超载 | 训练期一般取 `C > 1` |
| 只看理论 FLOP | 理论便宜，实际延迟不降反升 | dispatch/combine 和 All-to-All 很贵 | 同时看通信、padding 和设备拓扑 |
| BPR 不排序 | 低置信 token 先占满容量 | 有价值 token 反而进不去专家 | 推理期按优先级排序 |
| MoE 层放得太多 | 吞吐下降明显 | 每层都要路由和通信 | 先试隔层放置或只放后几层 |
| 小数据集硬上大 MoE | 收益不稳定甚至退化 | 大容量模型更依赖大规模预训练 | 先用 dense ViT 打底 |
| 专家数过多但 batch 太小 | 负载极不均衡 | 平均每个专家拿到的 token 太少 | 增大 batch 或减少专家数 |

下面把几个新手最容易误判的点展开。

### 1. 为什么“importance 均衡”还不够

很多人初看公式会觉得：既然各专家总权重分布均匀，问题不就解决了？其实不对。看一个反例。

假设很多 token 的概率分布都接近：

$$
[0.49,\ 0.48,\ 0.03]
$$

若 `k=1`，则专家 0 几乎总被选中，专家 1 虽然概率质量也很大，但几乎接不到单。于是：

- importance 看起来不算太差
- load 却高度失衡

所以必须同时约束“权重总量”和“实际分配次数”。

### 2. 为什么容量不是越大越好

直觉上看，容量大一点好像更安全。确实，容量大能减少 token 溢出，但它会带来三个代价：

1. 更多 padding 和空槽位。
2. 更高内存占用。
3. 更高通信成本。

所以容量不是越大越好，而是一个精度、吞吐和实现复杂度之间的折中量。

### 3. 为什么理论 FLOP 低，不代表真实延迟低

V-MoE 常见的误解是“每个 token 只算 2 个专家，所以一定比 dense 更快”。这只在忽略通信时才近似成立。真实系统里还有三类额外成本：

| 额外成本 | 来源 |
| --- | --- |
| 路由成本 | router 线性层、top-k、排序 |
| dispatch/combine 成本 | token 重排、索引、scatter/gather |
| 通信成本 | 跨设备 All-to-All、同步等待 |

当专家分布在不同设备上时，通信往往是大头。因此，V-MoE 更准确的说法不是“天然更快”，而是“给定相近 FLOP 预算下，允许更大参数容量”。

### 4. 为什么 BPR 更适合推理而不是随意放进训练

BPR 的核心是“容量有限时先保留高置信 token”。这很适合推理，因为推理的目标是用更少算力保住效果。但训练期若过早、过强地裁剪 token，可能带来两个风险：

1. 早期路由器还不稳定，排序依据本身不可靠。
2. 被裁掉的 token 无法参与专家学习，可能影响收敛。

因此，训练通常更强调稳定的容量和负载均衡；而 BPR 更适合在推理期拿来做精度和算力之间的可调开关。

---

## 替代方案与适用边界

V-MoE 不是所有视觉任务的默认最优解。是否值得使用，主要看三件事：

1. 你是否真的需要更大的模型容量。
2. 你是否有足够的训练数据和分布式资源。
3. 你是否能接受更复杂的训练和推理系统。

先看场景判断表：

| 场景 | 推荐方案 | 原因 | 额外约束 |
| --- | --- | --- | --- |
| 超大规模预训练，追求更高容量上限 | V-MoE | 容量和 FLOP 可部分解耦 | 需要稳定分布式训练和通信 |
| 中小数据集，训练预算有限 | 标准 ViT 或 ConvNet | 稀疏路由收益未必覆盖复杂度 | 实现更简单，调参更稳 |
| 推理必须稳定且不允许 token 排序 | dense ViT 或少量 MoE 层 | 无法充分利用 BPR | 延迟更可控 |
| 想要更简单的 MoE | Switch 风格 `k=1` | 实现和调试更简单 | 表达能力通常弱于 `k=2` |
| 想在已有 ViT 上低风险试验 | Hybrid dense+MoE | 只替换少数 MLP 层 | 需要重新选择插入层位 |

再和相近路线做一个横向比较：

| 方法 | 路由粒度 | 每 token 激活专家数 | 优势 | 代价 |
| --- | --- | --- | --- | --- |
| dense ViT | 无 | 全参数激活 | 结构简单，延迟稳定 | 参数扩张昂贵 |
| Switch Transformer 风格 | token 级 | `k=1` | 更简单，通信稍少 | 表达灵活性较弱 |
| V-MoE | token/patch 级 | 常见为 `k=2` | 容量大，效果上限高 | 路由、combine、通信更复杂 |
| 图像级专家选择 | 样本级 | 通常少数专家 | 控制逻辑更简单 | 不够细粒度，无法针对 patch 分工 |

一个实用判断标准是：

- 如果你有超大数据、足够多设备、目标是尽可能拉高视觉预训练上限，V-MoE 值得考虑。
- 如果你更关心训练稳定、实现简单、部署延迟稳定，dense ViT 往往是更稳的工程选项。
- 如果你只是想验证“专家化”是否有收益，先在后几层替换少量 MLP，比全层替换更可控。

最后补一个实际决策清单：

| 决策问题 | 推荐判断方式 |
| --- | --- |
| 要不要用 V-MoE | 先看数据和训练资源够不够大 |
| 专家放哪些层 | 先从后部层或隔层放置开始 |
| `k` 取多少 | 先试 `k=2`，再按通信和稳定性回调 |
| 容量怎么定 | 训练先保守设 `C>1`，推理再压缩 |
| 是否用 BPR | 若推理要做算力可调，优先考虑 |
| 是否值得部署 | 不只看 FLOP，也要测真实延迟和吞吐 |

这一节的核心结论是：**V-MoE 适合追求高容量上限的视觉预训练，不适合把“理论稀疏”误当成“任何场景都更快”。**

---

## 参考资料

1. Carlos Riquelme, Joan Puigcerver, Basil Mustafa, et al. *Scaling Vision with Sparse Mixture of Experts*. NeurIPS 2021. https://arxiv.org/abs/2106.05974
2. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. https://arxiv.org/abs/2010.11929
3. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. https://arxiv.org/abs/1701.06538
4. William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 2022. https://arxiv.org/abs/2101.03961
5. V-MoE 论文第 2 节：V-MoE 结构定义、路由方式、专家容量设置。https://arxiv.org/abs/2106.05974
6. V-MoE 论文第 4 节：Batch Prioritized Routing 与低容量推理分析。https://arxiv.org/abs/2106.05974
7. V-MoE 论文附录 A：`softmax -> top-k` 路由顺序、importance loss 与 load loss 的定义和比较。https://arxiv.org/abs/2106.05974
8. V-MoE 论文附录 C：Vanilla Routing 与 Batch Prioritized Routing 的算法细节。https://arxiv.org/abs/2106.05974

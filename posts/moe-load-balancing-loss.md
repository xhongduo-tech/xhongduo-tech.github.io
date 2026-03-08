## 核心结论

MoE（Mixture of Experts，专家混合模型，意思是“很多子网络里只激活少数几个”）里的负载均衡损失，核心作用不是提升单步精度，而是防止路由器把大多数 token 都塞给少数专家，导致其余专家几乎不训练。常见形式是：

$$
L_{\text{aux}}=\alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i
$$

其中，$f_i$ 是专家 $i$ 实际接收的 token 比例，$P_i$ 是路由器给专家 $i$ 的平均概率，$N$ 是专家数，$\alpha$ 是调节强度的系数。

这个式子有三个工程含义。

第一，它约束“实际负载”和“路由意图”同时不要过度集中。只看实际分配不够，因为 top-k 之后的硬路由本身不可导；只看 softmax 概率也不够，因为真正执行时可能仍然挤到少数专家。把 $f_i$ 和 $P_i$ 乘起来，本质上是在惩罚“高概率且高占用”的专家被反复放大。

第二，$\alpha$ 决定约束强度。工程上常从 `1e-2` 到 `1e-1` 起步，再根据主任务 loss 和负载统计微调。太小，负载均衡约束几乎不起作用，容易出现 expert collapse（专家塌缩，意思是“只有少数专家在工作”）；太大，路由器会被逼到近似均匀分配，专家失去特化能力。

第三，$N$ 不是装饰项。它让不同专家数量下的损失尺度可比。否则从 4 个专家换到 64 个专家时，辅助损失的量级会自动缩小，调参失去稳定参考。

对新手最重要的理解是：负载均衡损失不是要求“所有专家永远平均工作”，而是要求“不要让路由早期的随机偏差滚雪球”。没有这项约束时，某个专家只要在早期偶然更常被选中，就会拿到更多梯度，进而更容易在后续继续被选中，形成正反馈。加入 $L_{\text{aux}}$ 后，优化器会看到“这个专家已经太热门了”，从而抑制它继续独占流量。

可以把它理解成一种训练期稳定器。主任务 loss 决定“这个 token 应该由谁处理更合适”，而负载均衡损失决定“不要让这种偏好在训练早期失控到只剩少数专家还有训练信号”。前者负责能力，后者负责生态。

---

## 问题定义与边界

MoE 的目标不是让所有专家平均工作，而是让“该分工时分工，该均衡时均衡”。这句话很重要，因为很多初学者会把负载均衡误解成“平均主义”。

先定义问题。路由器会对每个 token 计算一个 gating score（门控分数，意思是“这个 token 更该去哪个专家”），再从多个专家中选 top-1 或 top-k。训练初期，路由器参数很不稳定，少量随机偏差就可能被梯度迅速放大，导致某些专家拿到绝大多数 token。结果通常有两个：

1. 热门专家过载，容量不够时会丢 token 或触发溢出。
2. 冷门专家几乎收不到训练信号，逐渐“死亡”。

如果把 token 数记为 $T$、专家数记为 $N$，理想情况下每个专家平均应接收大约 $\frac{T}{N}$ 个 token；若使用 top-k 路由，则平均派发次数接近 $\frac{kT}{N}$。问题在于，真实路由分布通常不是平均值附近的小波动，而是会快速出现长尾或尖峰分布。也就是：

$$
\exists i,\quad f_i \gg \frac{1}{N}
$$

一旦这种集中化持续多个 step，训练就不再是“少数专家更擅长某些模式”，而是“少数专家垄断了训练机会”。

问题边界也要说清楚。负载均衡损失不是为了把所有 batch 都压成严格均匀，而是把失衡控制在可接受范围内。因为专家存在的意义本来就是特化。如果每个输入都被平均送到所有专家附近，那 MoE 就退化成“多个相似子网络的平均分工”，模型容量虽然还在，条件计算的优势却弱了。

下面这张表概括了“无辅助约束”和“有辅助约束”的差异：

| 场景 | 专家负载分布 | 专家特化能力 | 训练稳定性 | 常见后果 |
|---|---|---|---|---|
| 无 auxiliary loss | 常快速失衡，少数专家吃掉大部分 token | 表面上更强，但常是伪特化 | 差 | 专家死亡、容量浪费、训练震荡 |
| 有适中 auxiliary loss | 保持可控失衡，不追求绝对平均 | 能保留真正有用的分工 | 较稳 | 大多数专家持续获得训练信号 |
| auxiliary loss 过强 | 过度接近均匀 | 特化被抹平 | 表面稳定，实际退化 | 路由器失去区分输入能力 |

一个训练初期的直观例子是：8 个专家、一个 batch 有 1024 个 token。没有辅助约束时，几轮更新后就可能出现 70% 以上 token 挤到 1 到 2 个专家，其余专家接近空闲。此时主任务 loss 可能还在下降，但模型实际上已经放弃了大部分专家容量。

所以边界很明确：

- 它解决的是“失衡过头”的问题。
- 它不负责创造特化，只负责给特化留下健康生长空间。
- 它不能替代容量限制、token dropping、路由噪声、router z-loss 等其他机制。
- 它主要作用于训练阶段；推理阶段通常不单独计算这项辅助损失。

对新手来说，最容易混淆的是“均衡”和“有用”。均衡只说明训练机会分配更健康，不说明专家已经学到互补功能；真正的特化仍然来自主任务目标和数据分布。

---

## 核心机制与推导

先把两个量说清楚。

$f_i$ 是专家 $i$ 的实际负载比例：

$$
f_i = \frac{\text{batch 中送到专家 } i \text{ 的 token 数}}{\text{batch 总 token 数}}
$$

它回答的是“最终有多少 token 真正去了这个专家”。

若使用 top-1 路由，则 $\sum_i f_i = 1$；若使用 top-k 路由并把一个 token 发送给多个专家，常见做法有两种：

1. 仍按“每个 token 只计一次主专家”统计 $f_i$。
2. 按总派发次数归一化，即让 $\sum_i f_i = k$ 或重新缩放到 1。

工程里必须先统一定义，否则不同代码库的监控数值不能直接比较。

$P_i$ 是专家 $i$ 的平均路由概率：

$$
P_i = \frac{1}{T}\sum_{t=1}^{T} p_{t,i}
$$

其中 $T$ 是 batch 中 token 总数，$p_{t,i}$ 是第 $t$ 个 token 对专家 $i$ 的 softmax 概率。它回答的是“路由器平均有多想把 token 发给这个专家”。

为什么用 $\sum_i f_i P_i$？

因为它同时结合了硬分配和软概率。若某个专家既经常被实际选中，又长期拿到较高概率，那么 $f_i P_i$ 会变大，总损失升高，梯度就会推动路由概率往更平衡的方向调整。

先看均衡状态。若有 $N$ 个专家，且完全均衡，则：

$$
f_i = \frac{1}{N}, \quad P_i = \frac{1}{N}
$$

代入得：

$$
\sum_{i=1}^{N} f_i P_i
= \sum_{i=1}^{N} \frac{1}{N}\cdot\frac{1}{N}
= N \cdot \frac{1}{N^2}
= \frac{1}{N}
$$

再乘上 $N$：

$$
N\sum_i f_i P_i = 1
$$

于是：

$$
L_{\text{aux}} = \alpha
$$

这说明一个很实用的性质：在理想均衡状态下，辅助损失的基线大约就是 $\alpha$。这让调参很直观。你把 $\alpha$ 设成 `0.01`，意味着“最理想时，这项损失大致贡献 0.01 的量级”。

再看极端失衡状态。若所有 token 都被送到专家 1，且路由器也几乎总给它最高概率，则有近似：

$$
f_1 \approx 1,\quad P_1 \approx 1,\quad f_{i>1}\approx 0,\quad P_{i>1}\approx 0
$$

于是：

$$
N\sum_i f_iP_i \approx N
$$

从而：

$$
L_{\text{aux}} \approx \alpha N
$$

也就是说，在极端塌缩下，这项损失相对均衡基线会放大约 $N$ 倍。专家数越多，塌缩的惩罚越明显，这也是为什么大规模 MoE 更依赖这项正则。

玩具例子如下。

设 $N=4$，$\alpha=0.01$。

均衡时：

$$
f=[0.25,0.25,0.25,0.25],\quad P=[0.25,0.25,0.25,0.25]
$$

则：

$$
L_{\text{aux}}=0.01 \cdot 4 \cdot (4\times 0.0625)=0.01
$$

失衡时：

$$
f=[0.5,0.2,0.2,0.1],\quad P=[0.4,0.25,0.2,0.15]
$$

则：

$$
\sum_i f_iP_i = 0.5\times0.4 + 0.2\times0.25 + 0.2\times0.2 + 0.1\times0.15 = 0.305
$$

所以：

$$
L_{\text{aux}}=0.01 \times 4 \times 0.305 = 0.0122
$$

从 `0.01` 上升到 `0.0122`，单步看差异不大，但这项损失会在每个 batch 上持续施加方向一致的梯度。长时间累积后，它会明显改变路由分布。

再补一个极端例子。若：

$$
f=[1,0,0,0],\quad P=[1,0,0,0]
$$

则：

$$
L_{\text{aux}} = 0.01 \times 4 \times 1 = 0.04
$$

这比理想均衡状态下的 `0.01` 高了 4 倍，已经足以在训练中持续施加显著修正。

这里还要强调一个细节：$f_i$ 常来自离散 top-k 选择，本身不可导；真正传递梯度的主要是 $P_i$。所以这个损失的设计不是“直接让硬路由反向传播”，而是“通过软概率去修正下一轮的硬路由结果”。这也是为什么它有效，但又不会像强制重写分配规则那样生硬。

如果把 loss 对某个 $P_i$ 的偏导写出来，会更直观：

$$
\frac{\partial L_{\text{aux}}}{\partial P_i} = \alpha N f_i
$$

它的含义很简单：哪个专家当前实际负载 $f_i$ 更高，优化器就会更强地压低这个专家未来的平均概率。虽然这不是完整梯度链路，但它足够解释“热门专家为什么会被抑制”。

一个真实工程例子是 Switch Transformer。它使用 top-1 routing，也就是每个 token 只去一个专家。top-1 的优点是计算省，但缺点是更容易塌缩，因为一旦选中就更集中。此时辅助负载均衡损失就很关键：它不是锦上添花，而是让 top-1 路由在大规模训练下还能工作的稳定器之一。

---

## 代码实现

工程实现里，通常需要三步：

1. 计算路由概率 `router_probs`，形状一般是 `[tokens, num_experts]`。
2. 根据 top-1 或 top-k 结果统计每个专家收到的 token 数，得到 `f`。
3. 对 `router_probs` 在 token 维度求均值，得到 `P`，然后计算辅助损失。

下面先给一个不依赖深度学习框架的可运行 Python 示例。它演示了 top-1 场景下的公式、监控和数值检查。

```python
import math
from typing import List, Tuple


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def argmax(xs: List[float]) -> int:
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs[1:], start=1):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def compute_aux_loss(
    router_logits: List[List[float]],
    alpha: float = 0.01,
) -> Tuple[float, List[float], List[float], List[int]]:
    """
    router_logits: shape [num_tokens, num_experts]
    returns: (aux_loss, f, P, chosen_experts)
    """
    num_tokens = len(router_logits)
    if num_tokens == 0:
        raise ValueError("router_logits must be non-empty")

    num_experts = len(router_logits[0])
    if num_experts == 0:
        raise ValueError("num_experts must be positive")

    for row in router_logits:
        if len(row) != num_experts:
            raise ValueError("all rows must have the same num_experts")

    router_probs = [softmax(row) for row in router_logits]

    counts = [0] * num_experts
    chosen_experts = []
    for probs in router_probs:
        chosen = argmax(probs)
        chosen_experts.append(chosen)
        counts[chosen] += 1

    f = [c / num_tokens for c in counts]
    P = [
        sum(probs[i] for probs in router_probs) / num_tokens
        for i in range(num_experts)
    ]

    aux = alpha * num_experts * sum(fi * pi for fi, pi in zip(f, P))
    return aux, f, P, chosen_experts


def format_vector(xs: List[float], digits: int = 4) -> List[float]:
    return [round(x, digits) for x in xs]


balanced_logits = [
    [2.0, 0.2, 0.1, 0.0],  # -> expert 0
    [0.1, 2.0, 0.2, 0.0],  # -> expert 1
    [0.0, 0.1, 2.0, 0.2],  # -> expert 2
    [0.0, 0.2, 0.1, 2.0],  # -> expert 3
    [2.1, 0.1, 0.0, 0.0],  # -> expert 0
    [0.0, 2.1, 0.1, 0.0],  # -> expert 1
    [0.0, 0.1, 2.1, 0.0],  # -> expert 2
    [0.1, 0.0, 0.2, 2.1],  # -> expert 3
]

collapsed_logits = [
    [3.0, 0.2, 0.1, 0.0],
    [3.1, 0.1, 0.0, 0.0],
    [2.9, 0.3, 0.2, 0.1],
    [3.2, 0.1, 0.0, 0.0],
    [3.0, 0.2, 0.1, 0.1],
    [3.1, 0.1, 0.1, 0.0],
    [3.2, 0.1, 0.0, 0.0],
    [2.8, 0.4, 0.2, 0.1],
]

aux_balanced, f_balanced, p_balanced, chosen_balanced = compute_aux_loss(
    balanced_logits, alpha=0.01
)
aux_collapsed, f_collapsed, p_collapsed, chosen_collapsed = compute_aux_loss(
    collapsed_logits, alpha=0.01
)

assert len(f_balanced) == 4
assert abs(sum(f_balanced) - 1.0) < 1e-9
assert abs(sum(p_balanced) - 1.0) < 1e-9
assert aux_collapsed > aux_balanced

print("balanced chosen:", chosen_balanced)
print("balanced aux:", round(aux_balanced, 6))
print("balanced f:", format_vector(f_balanced))
print("balanced P:", format_vector(p_balanced))

print("collapsed chosen:", chosen_collapsed)
print("collapsed aux:", round(aux_collapsed, 6))
print("collapsed f:", format_vector(f_collapsed))
print("collapsed P:", format_vector(p_collapsed))
```

这段代码会得到两个很直观的结果：

- `balanced` 场景里，4 个专家的 `f` 接近均匀。
- `collapsed` 场景里，绝大多数 token 都会落到 expert 0，`aux_loss` 会显著更高。

如果需要支持 top-k，可以把“每个 token 只选一个专家”的逻辑改成“选前 k 个专家并累计计数”。下面给一个仍然可运行的 top-k 版本，重点是把统计定义写清楚。

```python
import math
from typing import List, Tuple


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def top_k_indices(xs: List[float], k: int) -> List[int]:
    pairs = list(enumerate(xs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in pairs[:k]]


def compute_aux_loss_topk(
    router_logits: List[List[float]],
    alpha: float = 0.01,
    top_k: int = 2,
) -> Tuple[float, List[float], List[float], List[List[int]]]:
    """
    这里将 f 定义为“按总派发次数归一化后的负载比例”。
    因此 sum(f) = 1，即使一个 token 会被派到多个专家。
    """
    num_tokens = len(router_logits)
    if num_tokens == 0:
        raise ValueError("router_logits must be non-empty")

    num_experts = len(router_logits[0])
    if not (1 <= top_k <= num_experts):
        raise ValueError("top_k must be in [1, num_experts]")

    router_probs = [softmax(row) for row in router_logits]

    counts = [0] * num_experts
    dispatches = []
    for probs in router_probs:
        chosen = top_k_indices(probs, top_k)
        dispatches.append(chosen)
        for idx in chosen:
            counts[idx] += 1

    total_dispatches = num_tokens * top_k
    f = [c / total_dispatches for c in counts]
    P = [
        sum(probs[i] for probs in router_probs) / num_tokens
        for i in range(num_experts)
    ]

    aux = alpha * num_experts * sum(fi * pi for fi, pi in zip(f, P))
    return aux, f, P, dispatches


router_logits = [
    [2.5, 1.9, 0.3, 0.1],
    [2.4, 1.8, 0.2, 0.2],
    [2.6, 1.7, 0.1, 0.3],
    [2.7, 1.6, 0.2, 0.1],
]

aux, f, P, dispatches = compute_aux_loss_topk(router_logits, alpha=0.01, top_k=2)
assert abs(sum(f) - 1.0) < 1e-9
assert abs(sum(P) - 1.0) < 1e-9

print("dispatches:", dispatches)
print("aux:", round(aux, 6))
print("f:", [round(x, 4) for x in f])
print("P:", [round(x, 4) for x in P])
```

在 PyTorch 里，逻辑通常更直接。下面给一个可直接放进训练循环的 top-1 版本：

```python
import torch


def moe_aux_loss(
    router_logits: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    router_logits: [T, N]
    returns:
        aux_loss: []
        router_probs: [T, N]
        f: [N]
        P: [N]
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [T, N]")

    T, N = router_logits.shape
    if T == 0 or N == 0:
        raise ValueError("router_logits must be non-empty")

    router_probs = torch.softmax(router_logits, dim=-1)     # [T, N]
    expert_indices = torch.argmax(router_probs, dim=-1)     # [T]

    counts = torch.bincount(expert_indices, minlength=N).to(router_probs.dtype)
    f = counts / T
    P = router_probs.mean(dim=0)

    aux_loss = alpha * N * torch.sum(f * P)
    return aux_loss, router_probs, f, P


if __name__ == "__main__":
    torch.manual_seed(0)

    router_logits = torch.tensor(
        [
            [2.0, 0.1, 0.1, 0.0],
            [1.9, 0.2, 0.1, 0.0],
            [0.1, 2.0, 0.1, 0.0],
            [0.1, 0.2, 2.0, 0.0],
            [2.1, 0.1, 0.1, 0.0],
            [2.2, 0.1, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    aux_loss, router_probs, f, P = moe_aux_loss(router_logits, alpha=0.05)

    print("aux_loss =", round(aux_loss.item(), 6))
    print("f =", [round(x, 4) for x in f.tolist()])
    print("P =", [round(x, 4) for x in P.tolist()])

    total_loss = aux_loss
    total_loss.backward = lambda: None  # 这里只演示接口形状，不做真实训练
```

实现时建议记录这些张量和指标：

| 名称 | 形状 | 含义 | 是否参与梯度 |
|---|---|---|---|
| `router_logits` | `[T, N]` | 路由器原始输出 | 是 |
| `router_probs` | `[T, N]` | softmax 后概率 | 是 |
| `expert_indices` | `[T]` | top-1 或 top-k 选择结果 | 通常否 |
| `counts` | `[N]` | 每个专家接收 token 数 | 否 |
| `f` | `[N]` | 实际负载比例 | 通常否 |
| `P` | `[N]` | 平均路由概率 | 是 |
| `aux_loss` | `[]` | 负载均衡损失 | 是 |

如果训练规模较大，$\alpha$ 不一定从第一步就用固定值。更稳妥的做法是 warm-up（预热，意思是“前期先弱一点，再逐渐加大”）。例如前 5% 训练步从 `0` 线性升到 `0.05`，避免模型还没学会基本任务时就被强行要求均衡。

一个常见调度写法是：

$$
\alpha_t = \alpha_{\max} \cdot \min\left(1,\frac{t}{t_{\text{warmup}}}\right)
$$

如果后期发现专家已经形成稳定分工，也可以反过来做衰减，让主任务在后半程拥有更高权重。

---

## 工程权衡与常见坑

这部分决定你是否能把公式真正用稳。

第一类坑是 $\alpha=0$ 或太小。表面上看，模型仍然能训练，loss 也可能下降，但专家利用率会迅速恶化。你会发现总参数很多，真正有梯度更新的却只有少数几个专家。这叫“有效容量缩水”。

第二类坑是 $\alpha$ 太大。比如直接设到 `1.0` 甚至更高。此时路由器的主要优化目标不再是“把合适输入送给合适专家”，而是“谁都别太突出”。结果是每个专家负载很漂亮，任务指标却上不去，因为模型失去了条件计算的判别力。

第三类坑是只看辅助损失下降，不看主任务 loss。`L_aux` 下降只能说明分配更平均，不等于任务做得更好。真正要监控的是比例关系：

$$
r = \frac{L_{\text{aux}}}{L_{\text{task}}}
$$

如果 $r$ 持续升高，或者 `L_aux` 很快压到很低但 `L_task` 停滞，通常说明均衡约束过强。

第四类坑是统计口径混乱。比如训练日志里的 `f_i` 是按 top-1 主专家统计，而分析脚本里的 `f_i` 是按 top-2 总派发次数统计，这会直接导致“看起来同一套实验，负载曲线却对不上”。

第五类坑是把负载均衡问题全部归咎于 $\alpha$。实际训练中，下面几个变量也经常起决定性作用：

| 变量 | 作用路径 | 典型风险 |
|---|---|---|
| `capacity_factor` | 决定单专家可承载 token 上限 | 太小会频繁 overflow |
| `top_k` | 决定每个 token 派给几个专家 | 太小易塌缩，太大增通信 |
| `router_lr` | 决定路由器更新速度 | 太大易震荡 |
| batch size | 决定负载统计稳定性 | 太小噪声大 |
| router noise | 增加早期探索 | 过强会扰乱收敛 |

下面是一个更实用的“问题-症状-调参”表：

| 问题 | 典型症状 | 常见原因 | 调整方向 |
|---|---|---|---|
| 专家塌缩 | 少数专家负载接近 80% 以上 | $\alpha$ 太小，batch 太小，top-1 太激进 | 提高 $\alpha`，增加 batch，加入路由噪声 |
| 路由过均匀 | 每个专家负载都很整齐，但任务 loss 不降 | $\alpha$ 太大 | 下调 $\alpha`，后期衰减辅助项 |
| 部分专家长期为 0 | 某些专家一直收不到 token | 初始化偏置或早期正反馈过强 | 预热 $\alpha`，加入随机探索 |
| 训练震荡 | 负载分布和任务 loss 都剧烈波动 | 路由器学习率过高 | 单独降低 router lr |
| capacity overflow | 热门专家反复溢出、丢 token | 负载集中但容量设置太小 | 增大 capacity factor，同时保留 balance loss |

真实工程里，建议至少监控以下指标：

| 指标 | 公式或定义 | 作用 |
|---|---|---|
| 负载最大最小比 | `max(load) / max(min(load), eps)` | 快速判断是否严重失衡 |
| 负载标准差 | `std(f)` | 看整体离散程度 |
| 路由熵 | $-\frac{1}{T}\sum_t\sum_i p_{t,i}\log p_{t,i}$ | 看 softmax 是否过尖 |
| 溢出率 | `dropped_tokens / total_tokens` | 判断容量是否不足 |
| 辅助项占比 | `aux_loss / task_loss` | 判断辅助损失是否越权 |

一个经验判断是：

- 若 `max/min` 长期大于 `5`，通常负载已经明显失衡。
- 若单个专家长期占用超过 `50%`，通常不是“自然特化”，而是路由偏置失控。
- 若 `aux/task` 高到和主任务同量级，说明辅助项可能已经过强。
- 若训练前期失衡严重、后期逐渐收敛，可考虑先高后低或先低后高的 $\alpha$ 调度，而不是只用常数。

对新手最重要的一点是：不要只问“$\alpha$ 该设多少”，要问“它相对主任务 loss 占多大权重，是否真的改变了路由统计”。因为不同任务、不同 token 数、不同专家数，绝对值都可能变化，但相对权重才真正决定优化方向。

可以把调参过程压缩成一个最小流程：

1. 先固定 `top_k`、`capacity_factor` 和 router 学习率。
2. 用较小 $\alpha$ 跑短实验，观察 `f_i` 分布和 overflow。
3. 若明显塌缩，再把 $\alpha$ 逐步提高一档，而不是直接放大 10 倍。
4. 若负载已均衡但主任务指标变差，优先怀疑 $\alpha$ 过大，而不是继续堆更多正则。

---

## 替代方案与适用边界

辅助负载均衡损失不是唯一办法，但它通常是最直接、最便宜的一种。

常见替代方案有四类。

第一类是 entropy regularization（熵正则，意思是“鼓励路由概率不要过于尖锐”）。它直接约束 softmax 分布的尖锐程度，常见形式是最大化门控分布熵：

$$
L_{\text{ent}} = -\beta \cdot \frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{N} p_{t,i}\log p_{t,i}
$$

优点是连续可导、实现简单；缺点是它只管概率形状，不直接管真实负载，可能出现“概率看着平，但硬路由还是挤”的情况。

第二类是 expert dropout（专家 dropout，意思是“训练时随机屏蔽部分专家或连接”）。它能打断热门专家的正反馈，让冷门专家也获得机会。缺点是引入了额外随机性，若本来 batch 就小，训练可能更不稳。

第三类是容量控制和 token dropping。它不是正则项，而是硬约束。每个专家最多接收固定容量，超出的 token 被丢弃或重路由。它能防止单个专家爆掉，但不能从根上解决“为什么总是它最热门”的问题，所以通常要和辅助损失一起用。

第四类是加噪声路由或随机探索。例如在 router logits 上加入高斯噪声、Gumbel 噪声，或在早期提高温度。这类方法解决的是“探索不足”问题，而不是直接最优化负载指标。

对比如下：

| 方法 | 主要约束对象 | 何时够用 | 何时不足 |
|---|---|---|---|
| Auxiliary loss | 实际负载 + 平均概率 | 大多数通用 MoE 训练 | 路由震荡很大时可能还需别的稳定器 |
| Entropy regularization | 概率分布尖锐度 | 想抑制过尖 softmax 时 | 不能直接保证真实负载均衡 |
| Expert dropout | 热门专家的垄断趋势 | 早期探索不足时 | 可能增加训练方差 |
| Capacity control | 单专家容量上限 | 防止局部爆炸 | 不解决长期塌缩根因 |
| Router noise | 早期探索能力 | 初始化偏置较强时 | 后期若过强会拖慢收敛 |

适用边界也要分场景看。

小规模 MoE，比如 4 到 8 个专家、batch 也不大时，天然负载偏差没那么极端，辅助损失可以设得较弱。因为这时强行均衡的副作用，往往比塌缩风险更明显。

大规模 MoE，比如几十到上百专家、大 batch、top-1 routing 时，辅助损失几乎是必需项。专家越多，随机偏差被放大的空间越大；batch 越大，热门专家一旦形成优势，就会在统计上更加稳定地压制其他专家。

如果用 top-2 或更大的 top-k，负载会比 top-1 稍稳，但通信和显存开销也会增加。因此它不是“替代均衡损失”，而是把问题从“极端塌缩”改成“较温和的偏斜”。

实践上常用的组合策略是：

- 基础组合：`auxiliary loss + capacity control`
- 容易塌缩时：再加 `router noise` 或小幅 `expert dropout`
- 已经过于平均时：减小 `alpha`，甚至后期衰减到更低
- 如果大量 token 被 drop：先检查 `capacity_factor`，不要只盯着 $\alpha$

所以结论不是“有没有更好的替代方案”，而是“你的塌缩发生在概率层、硬路由层，还是容量层”。不同层次的问题，需要不同层次的约束。

再压缩成一句话就是：

- 若问题是“路由意图太尖”，优先看熵或温度。
- 若问题是“真实分配太偏”，优先看 auxiliary loss。
- 若问题是“热门专家装不下”，优先看 capacity。
- 若问题是“训练一开始就锁死少数专家”，优先看噪声、初始化和 warm-up。

---

## 参考资料

1. **Noam Shazeer et al., _Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer_ (2017)**  
   核心贡献：早期大规模稀疏 MoE 的代表论文，系统展示了门控、容量、专家负载不均衡和辅助损失之间的关系。  
   建议阅读方式：先看问题设定和训练难点，再看 load-balancing 相关讨论，不必一开始就陷入全部并行细节。

2. **Dmitry Lepikhin et al., _GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding_ (2020)**  
   核心贡献：把稀疏专家层和大规模分布式训练结合起来，说明为什么负载均衡不仅影响模型学习，也影响设备利用率和通信开销。  
   建议阅读方式：重点看 MoE 层、top-2 routing 和 capacity 设计，理解“均衡”为什么是系统问题而不只是优化问题。

3. **William Fedus, Barret Zoph, Noam Shazeer, _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity_ (2021)**  
   核心贡献：把路由简化为 top-1，使 Switch Transformer 成为理解现代 MoE 的经典入口，也让辅助负载均衡损失的重要性更明显。  
   建议阅读方式：优先看 top-1 routing、auxiliary loss、capacity factor 和 token dropping 的实验说明。

4. **Yanqi Zhou et al., _Mixture-of-Experts with Expert Choice Routing_ (2022)**  
   核心贡献：从另一个角度说明“谁拥有选择权”会如何影响负载均衡。虽然它讨论的是 expert-choice routing，但非常适合对比理解为什么 token-choice 往往需要额外的 balance loss。  
   建议阅读方式：结合本文一起看“负载均衡是由结构保证，还是由损失鼓励”。

5. **DeepSpeed-MoE 或 Megatron-LM 的 MoE 实现文档与源码**  
   核心贡献：展示工程里如何统计 `router_probs`、`expert_indices`、overflow、aux loss 和 capacity。  
   建议阅读方式：不要只看公式，重点看训练日志和监控项的命名，理解论文公式如何落成实际代码。

6. **开源博客或技术笔记中对 Switch/GShard 辅助损失的拆解文章**  
   核心贡献：通常会把 $\alpha$ 的影响、top-k 统计口径、路由塌缩现象用更接近工程调试的方式写清楚。  
   建议阅读方式：可作为论文阅读后的补充，但遇到公式实现不一致时，应以论文和源码为准。

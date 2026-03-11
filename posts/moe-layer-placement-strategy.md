## 核心结论

MoE，Mixture of Experts，直译是“专家混合”。在 Transformer 里，它通常表示：**同一层准备多个前馈网络（experts），但每个 token 只激活其中少数几个**。因此，MoE 层放在哪些 Transformer 层里，不是实现细节，而是直接决定容量、计算、通信和训练稳定性的主开关。

结论先写清楚：

1. 全层 MoE，指几乎每个 FFN 都替换成 MoE，参数容量最大，适合大规模预训练；代价是每层都要做路由，通信更频繁，训练和部署更难。
2. 交替层 MoE，指每隔 1 层、2 层或 4 层插入一次 MoE，其余层保持普通 dense FFN，容量增长没那么激进，但更容易守住 FLOPs 预算和稳定性。
3. “最优放置”没有统一答案，它取决于训练目标。如果目标是同等计算下尽量增大参数容量，全层更激进；如果目标是固定吞吐、固定硬件下稳定训练，交替层通常更稳。
4. Switch Transformer 的路线偏“控制计算和通信成本”，代表性做法是只在部分 FFN 层引入稀疏专家；Mixtral 的路线偏“把稀疏容量沿深度铺满”，在每层都使用 MoE FFN；ST-MoE 的经验表明，中等密度，例如每 2 层或每 4 层插一次，在不少场景里是更平衡的工程点。
5. MoE 层越密，路由次数越多，专家负载越难均衡，token drop，指 token 因专家容量不足而被丢弃、残差直通，或回退到备用路径，越容易出现；这时 Capacity Factor、负载均衡损失、z-loss 和 batch 规模，往往比“是不是 MoE”本身更关键。

一个新手能直接抓住的玩具例子：

假设有 24 层 Transformer，每层原本都有一个 dense FFN。
- 全层 MoE：24 层都换成 MoE，每个 token 在每层选 2 个专家。
- 交替层 MoE：只在第 2、4、6、...、24 层换成 MoE，其余 12 层仍是 dense FFN。
- 每 4 层 MoE：只在第 4、8、12、16、20、24 层换成 MoE。

这三种方案的差别，不在“有没有专家”，而在“每个 token 一路上要被路由多少次”。

| 策略 | 参数容量 | 计算/FLOPs | 路由频率 | 通信压力 | 训练稳定性 |
|---|---:|---:|---:|---:|---:|
| 每层都是 MoE | 最高 | 单层可控，但全局调度最重 | 最高 | 最高 | 最难调 |
| 每隔一层 MoE | 中高 | 更容易贴住预算 | 中等 | 中等 | 较稳 |
| 每 4 层一次 MoE | 中等 | 最容易控制 | 低 | 低 | 通常最稳 |

再把这个表翻成一句工程话：

- 你想把“总参数”尽量做大，就增加 MoE 层密度。
- 你想把“实际吞吐”尽量守住，就降低 MoE 层密度。
- 你想减少调参风险，就优先从隔层或每 4 层一次开始，而不是直接全层。

---

## 问题定义与边界

这篇讨论的问题不是“MoE 好不好”，而是更具体的一个架构问题：

**在 Transformer 的哪些 FFN 层上，把 dense FFN 替换成 MoE FFN？**

FFN，Feed-Forward Network，就是每层注意力后面的前馈子层，典型形式是两层线性变换加激活函数。大多数 MoE 模型把“替换位置”放在 FFN，而不是 self-attention，原因很直接：

| 原因 | 解释 |
|---|---|
| FFN 参数占比高 | 在标准 Transformer 中，FFN 往往比注意力部分占更多参数 |
| 易于专家化 | 不同 token 学不同“变换模板”，天然适合专家分工 |
| 实现相对独立 | 替换 FFN 不必重写整套注意力机制 |
| 稀疏收益明确 | 总参数随专家数增长，但单 token 计算只跟 Top-K 激活专家有关 |

这个问题的边界要先明确，否则“全层更强”或“交替更稳”都容易说错。

| 维度 | 需要问清的问题 | 对 MoE 位置选择的影响 |
|---|---|---|
| 训练目标 | 是大规模预训练，还是小数据微调 | 预训练更能吃下高密度 MoE；微调更容易过拟合 |
| 预算 | 限制的是参数、FLOPs、显存，还是跨卡通信 | 若通信是瓶颈，全层 MoE 可能不划算 |
| 数据规模 | token 数是否足够大 | 数据少时，专家更容易学偏，负载也更不稳 |
| 路由方式 | Top-1 还是 Top-2 | Top-2 表达更强，但通信和容量管理更复杂 |
| 容量策略 | 是否允许 token drop，CF 取多大 | 直接影响专家溢出频率 |
| 硬件拓扑 | 专家是否跨设备放置 | 层数越多，路由同步越频繁 |
| 推理目标 | 是追求低延迟还是高吞吐 | 全层 MoE 推理延迟更容易被路由与通信放大 |

可以用一个简单预算例子说明边界：

假设你的训练预算只能接受每步约 1.5B FLOPs。  
- 如果把 24 层全部换成 Top-2 MoE，你可能把参数容量做大很多，但每层都有路由和跨卡 dispatch，实际吞吐未必能守住预算。  
- 如果改成每 4 层一次 MoE，总共只有 6 次路由，通信和调度明显下降，更容易在同样硬件上稳定跑通。  
- 代价是专家容量只分布在 6 层，而不是 24 层，表示空间的释放没那么彻底。

还要补一个常被忽略的边界：**“计算预算相同”不等于“训练成本相同”**。  
MoE 常见的收益是“理论 FLOPs 接近 dense，但总参数更大”。问题在于，真实系统里还有两类额外成本：

| 成本类型 | dense FFN | MoE FFN |
|---|---|---|
| 乘加计算 | 稳定 | 只算被选中的专家，通常可控 |
| 路由计算 | 无 | 每个 MoE 层都要做 |
| token 重排 | 无 | 需要按专家分桶、再聚合 |
| 跨卡通信 | 较少 | 专家跨卡时显著增加 |
| padding 浪费 | 低 | 容量预留越大，padding 越多 |

可以把层级结构画成一个最简框图：

```text
全层 MoE:
[L1 MoE] -> [L2 MoE] -> [L3 MoE] -> ... -> [L24 MoE]

交替层 MoE:
[L1 Dense] -> [L2 MoE] -> [L3 Dense] -> [L4 MoE] -> ...

每4层一次 MoE:
[L1 Dense] -> [L2 Dense] -> [L3 Dense] -> [L4 MoE] -> ...
```

这里的核心不是“替换几层”，而是**稀疏容量沿深度方向如何分布**。深度上的分布决定模型是在“每一步都做专家选择”，还是“只在关键层做专家选择”。

对新手来说，可以把三种方案理解成三种“参数释放节奏”：

| 放置方式 | 深度上的感觉 | 适合理解成什么 |
|---|---|---|
| 全层 MoE | 每走一步都重新分流 | 每层都让 token 选专门处理器 |
| 隔层 MoE | 稀疏层和稳定层交替 | 一层做分流，一层做整合 |
| 每 4 层一次 | 少量关键层做专家化 | 只在阶段节点上做能力扩容 |

---

## 核心机制与推导

路由器，router，就是一个小网络，用来决定某个 token 应该去哪些专家。它先输出专家打分，再做 Top-K 选择。

设第 $l$ 层有 $E$ 个专家，输入 token 表示为 $x$，路由器输出 logits：

$$
z^{(l)}(x) \in \mathbb{R}^{E}
$$

如果使用 Top-K 路由，只保留得分最高的 $K$ 个专家索引集合 $\mathcal{T}_K(x)$。常见写法可以表示为：

$$
g_i^{(l)}(x)=
\begin{cases}
\frac{\exp(z_i^{(l)}(x))}{\sum_{j \in \mathcal{T}_K(x)} \exp(z_j^{(l)}(x))}, & i \in \mathcal{T}_K(x) \\
0, & \text{otherwise}
\end{cases}
$$

其中 $g_i^{(l)}(x)$ 是第 $i$ 个专家的门控权重。MoE 层输出为：

$$
\text{MoE}^{(l)}(x)=\sum_{i=1}^{E} g_i^{(l)}(x)\,\text{Expert}_i^{(l)}(x)
$$

白话解释是：先选专家，再把选中的专家输出按权重加起来。

### 玩具例子

某个 token 在第 4 层路由器上得到 4 个专家分数：

$$
z=[5.4,\ 4.9,\ 1.2,\ -0.5]
$$

如果用 Top-2，只保留前两个专家。对应 softmax 权重大约是：

$$
g \approx [0.62,\ 0.38,\ 0,\ 0]
$$

于是该层输出就是：

$$
y = 0.62 \cdot \text{Expert}_1(x) + 0.38 \cdot \text{Expert}_2(x)
$$

如果第 4 层不是 MoE，而是普通 dense FFN，那么就没有“选专家”这一步，直接计算一个固定 FFN：

$$
y=\text{FFN}(x)
$$

两者最大的区别是：dense FFN 对所有 token 都走同一套参数，MoE 允许不同 token 走不同子网络。

### 为什么“放置位置”会影响效果

假设总层数为 $L$，其中只有 $M$ 层是 MoE，那么每个 token 的总路由次数就是 $M$。这会带来三个直接后果：

1. 总专家容量近似与 $M$ 成正比。
2. 路由开销和跨设备通信近似与 $M$ 成正比。
3. token drop 风险会随着路由层数增加而累积。

如果每个 MoE 层有 $E$ 个专家、每个 token 选 $K$ 个专家，那么单层的激活计算量近似与 $K$ 成正比，而不是与 $E$ 成正比。这就是 MoE 的核心收益：**总参数按 $E$ 增长，单 token 计算主要按 $K$ 增长**。

把它写成更明确的近似关系：

- dense FFN 单层参数量：$P_{\text{ffn}}$
- 一个 MoE 层含 $E$ 个专家时，总参数量近似：$E \cdot P_{\text{ffn}} + P_{\text{router}}$
- 单 token 单层激活参数量近似：$K \cdot P_{\text{ffn}}$

因此当 $E \gg K$ 时，模型可以拥有很多参数，但每次只实际计算少数专家。

如果全网有 $M$ 个 MoE 层，则总参数量可近似写成：

$$
P_{\text{total}} \approx (L-M)\,P_{\text{ffn}} + M\,(E\cdot P_{\text{ffn}} + P_{\text{router}})
$$

而单 token 的 FFN 激活参数量近似为：

$$
P_{\text{active}} \approx (L-M)\,P_{\text{ffn}} + M\,(K\cdot P_{\text{ffn}})
$$

这两个式子一起说明一个关键点：  
**增加 MoE 层数，放大得最快的是“总容量”，不是“单 token 计算”**。  
但增加得同样快的，还有路由次数和系统复杂度。

容量不是无限可用的。每个专家能接收的 token 数量通常有上限。设 batch 内 token 数为 $T$，则单专家容量常写成：

$$
C = \left\lceil \text{CF} \cdot \frac{K T}{E} \right\rceil
$$

其中 CF 是 Capacity Factor，容量因子，可以理解为“给专家留多少额外余量”。

- CF 太小：专家容易爆满，token 被丢弃、跳过，或者退回备用逻辑。
- CF 太大：显存和 padding 浪费上升，吞吐下降。
- $E$ 太大但 batch 太小：平均到每个专家的 token 不够，专家训练信号变稀。
- $K$ 从 1 变 2：表达力更强，但每层 dispatch 和 combine 更复杂。

放置位置会改变这个式子的工程含义。因为当 $M$ 很大时，整个网络中发生容量竞争的次数也更多。即使单层 drop rate 不高，多层叠加后也可能影响表示稳定性。

如果第 $l$ 层的 token drop rate 是 $d_l$，那么一个 token 穿过所有 MoE 层后，至少在某一层遇到 drop 的概率近似为：

$$
1-\prod_{l=1}^{M}(1-d_l)
$$

当各层 drop rate 近似相同，记为 $d$，则有：

$$
1-(1-d)^M
$$

这个式子很简单，但很重要。举个数值例子：

| 单层 drop rate | MoE 层数 $M$ | 至少一次遇到 drop 的近似概率 |
|---|---:|---:|
| 1% | 4 | 3.9% |
| 1% | 12 | 11.4% |
| 1% | 24 | 21.4% |
| 2% | 24 | 38.4% |

这就是为什么“每层都 MoE”经常不是只多一点点难度，而是调参复杂度明显跃迁。

一个流程图可以把机制看清楚：

```text
token x
  |
  v
router logits z
  |
  v
Top-K 选择专家
  |
  v
dispatch 到 K 个专家
  |
  v
专家各自计算 FFN
  |
  v
按门控权重加权求和
  |
  v
输出到下一层
```

再把“层位置选择”加进流程里：

```text
进入第 l 层
  |
  +-- 如果 l 不是 MoE 层 --> 直接走 dense FFN
  |
  +-- 如果 l 是 MoE 层 ----> router -> Top-K -> experts -> combine
```

### 真实工程例子

Switch Transformer 的工程重点是：**让稀疏专家带来的容量增益，不把训练吞吐拖垮**。它的代表性特征包括：
- 使用更简单的路由策略，代表性配置是 Top-1。
- 重点处理训练不稳定、通信成本和低精度训练问题。
- 核心目标是“在接近固定计算预算下提升规模与质量”。

Mixtral 的工程重点不同。它在每层都使用 Top-2 MoE，让深层表示的每一步都可专家化。这样做的结果是：
- 参数容量很大。
- 每个 token 仍只激活少量专家。
- 深度方向上每一层都发生路由，因此对并行实现、token 重排和专家负载都更敏感。

ST-MoE 的价值在于，它不是简单追求“更多专家”或“更多 MoE 层”，而是系统讨论：
- 如何让 sparse model 训练更稳定。
- 如何让稀疏模型在迁移和微调时更可控。
- 为什么一些看起来“更大”的 sparse 配置，实际不一定比更克制的配置更好。

这就是“全层 vs 交替层”的本质差异：不是谁更先进，而是谁把稀疏预算分布得更密。

---

## 代码实现

实现层位置选择时，最直接的办法不是复制两套模型，而是把“哪些层是 MoE”做成配置。这样同一份 Transformer 代码可以支持全层、隔层、每 4 层一次等策略。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示四件事：

1. 如何指定哪些层是 MoE  
2. 如何做 Top-K 路由  
3. 如何做专家容量统计  
4. 不同放置策略下，前向路径和路由次数如何变化

```python
from math import exp
from typing import Callable, Dict, List, Tuple


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    s = sum(exps)
    return [v / s for v in exps]


def topk_indices(logits: List[float], k: int) -> List[int]:
    assert 1 <= k <= len(logits)
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]


def topk_gating(logits: List[float], k: int) -> List[float]:
    idx = topk_indices(logits, k)
    probs = softmax([logits[i] for i in idx])

    gates = [0.0] * len(logits)
    for i, p in zip(idx, probs):
        gates[i] = p
    return gates


def dense_ffn(x: float) -> float:
    # 玩具 dense FFN
    return 1.8 * x + 0.7


def make_expert(scale: float, bias: float) -> Callable[[float], float]:
    def expert(x: float) -> float:
        return scale * x + bias
    return expert


class Router:
    def __init__(self, expert_count: int):
        self.expert_count = expert_count

    def logits(self, x: float, layer_idx: int) -> List[float]:
        # 用确定性公式生成 logits，便于复现，不依赖随机数
        base = x + layer_idx * 0.3
        return [
            1.2 * base + 0.1,
            1.0 * base + 0.4,
            0.7 * base - 0.2,
            0.4 * base + 0.8,
        ][:self.expert_count]


class MoELayer:
    def __init__(self, experts: List[Callable[[float], float]], k: int = 2):
        self.experts = experts
        self.k = k

    def forward(
        self,
        x: float,
        logits: List[float],
        capacity: int
    ) -> Tuple[float, Dict[str, object]]:
        gates = topk_gating(logits, self.k)
        selected = [i for i, g in enumerate(gates) if g > 0.0]

        # 单 token 玩具例子里，每个被选专家都会收到 1 个 token
        overloaded = len(selected) > capacity
        dropped = max(0, len(selected) - capacity)

        y = 0.0
        accepted = 0
        for expert_idx in selected:
            if accepted >= capacity:
                break
            y += gates[expert_idx] * self.experts[expert_idx](x)
            accepted += 1

        # 若容量不足，未被处理的 token 份额直接保留残差输入，模拟一种常见回退思想
        if dropped > 0:
            dropped_mass = sum(gates[i] for i in selected[accepted:])
            y += dropped_mass * x

        info = {
            "selected_experts": selected,
            "gates": gates,
            "dropped_expert_slots": dropped,
            "overloaded": overloaded,
        }
        return y, info


def build_moe_layers(total_layers: int, moe_interval: int) -> set:
    assert total_layers > 0
    assert moe_interval > 0
    # 例：interval=2 -> 第 2,4,6... 层为 MoE，内部下标为 1,3,5...
    return set(range(moe_interval - 1, total_layers, moe_interval))


def capacity_per_expert(
    tokens_per_batch: int,
    top_k: int,
    expert_count: int,
    capacity_factor: float
) -> int:
    expected = top_k * tokens_per_batch / expert_count
    return max(1, int(expected * capacity_factor + 0.999999))


def forward_network(
    x: float,
    total_layers: int,
    moe_interval: int,
    top_k: int = 2,
    expert_count: int = 4,
    tokens_per_batch: int = 8,
    capacity_factor: float = 1.25
) -> Dict[str, object]:
    moe_layers = build_moe_layers(total_layers, moe_interval)
    experts = [
        make_expert(1.0, 0.0),
        make_expert(1.4, 0.2),
        make_expert(0.8, 0.5),
        make_expert(1.8, -0.3),
    ][:expert_count]
    router = Router(expert_count)
    moe = MoELayer(experts, k=top_k)

    cap = capacity_per_expert(
        tokens_per_batch=tokens_per_batch,
        top_k=top_k,
        expert_count=expert_count,
        capacity_factor=capacity_factor,
    )

    route_count = 0
    layer_logs = []

    for layer_idx in range(total_layers):
        is_moe = layer_idx in moe_layers
        if is_moe:
            logits = router.logits(x, layer_idx)
            x, info = moe.forward(x, logits, capacity=cap)
            route_count += 1
            layer_logs.append(
                {
                    "layer": layer_idx + 1,
                    "type": "moe",
                    "selected_experts": info["selected_experts"],
                    "gates": [round(v, 4) for v in info["gates"]],
                    "overloaded": info["overloaded"],
                }
            )
        else:
            x = dense_ffn(x)
            layer_logs.append({"layer": layer_idx + 1, "type": "dense"})

    return {
        "output": x,
        "moe_layers": sorted(i + 1 for i in moe_layers),
        "route_count": route_count,
        "capacity_per_expert": cap,
        "logs": layer_logs,
    }


def main() -> None:
    # 位置选择测试
    assert build_moe_layers(24, 1) == set(range(24))
    assert build_moe_layers(24, 2) == set(range(1, 24, 2))
    assert build_moe_layers(24, 4) == {3, 7, 11, 15, 19, 23}

    # Top-K 权重测试
    gates = topk_gating([5.4, 4.9, 1.2, -0.5], 2)
    assert abs(sum(gates) - 1.0) < 1e-9
    assert gates[0] > gates[1] > 0.0
    assert gates[2] == 0.0 and gates[3] == 0.0

    # 三种策略对比
    full = forward_network(1.0, total_layers=8, moe_interval=1)
    alt = forward_network(1.0, total_layers=8, moe_interval=2)
    sparse = forward_network(1.0, total_layers=8, moe_interval=4)

    assert full["route_count"] == 8
    assert alt["route_count"] == 4
    assert sparse["route_count"] == 2

    print("full route_count =", full["route_count"], "output =", round(full["output"], 4))
    print("alt  route_count =", alt["route_count"], "output =", round(alt["output"], 4))
    print("4th  route_count =", sparse["route_count"], "output =", round(sparse["output"], 4))

    # 打印其中一种策略的层日志
    print("alt moe layers =", alt["moe_layers"])
    for row in alt["logs"]:
        print(row)


if __name__ == "__main__":
    main()
```

这段代码有几个对新手重要的点：

| 代码段 | 它在说明什么 |
|---|---|
| `build_moe_layers` | “层位置选择”本质上就是一个下标集合 |
| `topk_gating` | 路由器先选专家，再对入选专家重新归一化 |
| `capacity_per_expert` | Capacity Factor 不是抽象概念，它直接决定单专家可接收多少 token |
| `route_count` | 全层 vs 交替层，最直接的差别就是整个前向要路由多少次 |
| `layer_logs` | 训练中需要观察每层路由情况，而不是只看总 loss |

如果运行这段代码，你会看到：
- `moe_interval=1` 时，每层都路由。
- `moe_interval=2` 时，只在偶数层路由。
- `moe_interval=4` 时，只在第 4、8、... 层路由。

在真实实现中，位置选择通常会写成类似下面的结构：

```python
moe_layers = set(range(moe_interval - 1, total_layers, moe_interval))

for idx, layer in enumerate(transformer_layers):
    x = layer.self_attn(x)
    if idx in moe_layers:
        x = moe_ffn_layers[idx](x)
    else:
        x = dense_ffn_layers[idx](x)
```

如果想进一步接近真实训练代码，一般还会把配置显式写开：

```python
class ModelConfig:
    total_layers = 24
    moe_interval = 2
    num_experts = 8
    top_k = 2
    capacity_factor = 1.25
    use_load_balance_loss = True
    use_router_z_loss = True
```

这里 `moe_interval` 控制 MoE 密度：
- `1` 表示每层都是 MoE
- `2` 表示每隔一层
- `4` 表示每 4 层一次

真实工程里还应额外追踪三类指标：

1. `router_z_loss`  
路由器正则，用来限制 logits 过大，减轻路由极端化。
2. `load_balance_loss`  
负载均衡损失，用来防止少数专家过热、多数专家闲置。
3. `token_drop_rate`  
token 丢弃率，用来观察 Capacity Factor 是否太小，或 batch 是否不足以支撑当前专家数。

再补一个经常遗漏的监控表：

| 指标 | 为什么要看 | 异常信号 |
|---|---|---|
| 每层专家利用率 | 看是否均匀分流 | 少数专家长期占绝大多数 token |
| 每层 drop rate | 看容量是否溢出 | 某些层持续偏高 |
| router logits 范围 | 看门控是否过尖 | 极大值持续增大 |
| 不同层吞吐时间 | 看是不是某几层通信卡住 | 个别 MoE 层耗时异常高 |
| 微调前后专家分布 | 看是否过拟合 | 微调后专家塌缩更明显 |

---

## 工程权衡与常见坑

MoE 位置选择不是纯理论问题，最后都会落到工程权衡。

### 1. 全层 MoE 不一定“白赚容量”

全层 MoE 的优点很明显：每层都能做专家化，容量释放最充分。问题在于，路由、dispatch、all-to-all 通信也发生在每层。模型论文里常说“激活参数少”，但工程里真正卡住的往往不是乘加次数，而是跨设备搬运 token 的代价。

更具体地说，全层 MoE 会同时放大四件事：
- 每层路由器计算
- token 到专家的重排
- 专家结果回收后的聚合
- 多层累计的不稳定性

所以“理论 FLOPs 接近 dense”不能直接推导出“工程吞吐接近 dense”。

### 2. 小数据微调时，高密度 MoE 更容易过拟合

过拟合就是模型把训练集模式记得太死，换数据就掉性能。专家网络比 dense FFN 参数更多、选择更灵活，所以在小任务上更容易学出“很专门但不泛化”的路径。

对新手来说，可以这样理解：
- dense FFN 像所有 token 共用一套改写规则。
- MoE 像给 token 提供多套改写规则。
- 数据很多时，多套规则有助于分工。
- 数据很少时，多套规则容易把训练样本记得过细。

因此小数据任务上，“先用隔层 MoE，再考虑全层”通常比“一开始就全层”稳得多。

### 3. token drop 不是越低越好

很多人第一次调 MoE，会把“token drop 一定要接近 0”当作目标。这不总是对。把 Capacity Factor 拉得很大确实能减少 drop，但也会带来 padding 浪费、显存增长和吞吐下降。实际工程里，适度的 drop 可能比过高容量更划算。

可以把这个权衡写成简表：

| 调整方向 | 好处 | 代价 |
|---|---|---|
| 提高 CF | 降低 drop | 更占显存，吞吐可能下降 |
| 减少专家数 $E$ | 每个专家更“吃饱” | 总容量下降 |
| 降低 Top-K | 路由更简单 | 表达力可能下降 |
| 减少 MoE 层数 $M$ | 累计不稳定性下降 | 稀疏容量释放变少 |

### 4. 路由不稳定常常不是“专家太少”，而是“层太密”

如果每层都路由，任何一层的小抖动都会沿深度传播。表现通常是：
- 某些专家长期热点
- 某些专家几乎不被选中
- loss 波动明显
- 不同 batch 的 token 分配差异很大

这时与其继续加专家数，不如先降低 MoE 密度，或者增强正则。

### 5. batch 太小会让“专家数很多”失去意义

这是一个很典型的新手坑。假设你有 64 个专家，但一个 batch 只有很少 token，那么平均到每个专家上的样本数可能很低。结果是：
- 有些专家几乎得不到更新
- 负载均衡损失变得很难调
- 指标表面上在收敛，但专家分工实际上没建立起来

所以专家数 $E$、batch token 数 $T$、Top-K、CF 和 MoE 层数 $M$ 是联动关系，不能分开看。

下面给出一个常见坑表：

| 常见问题 | 现象 | 常见原因 | 对策 |
|---|---|---|---|
| 过拟合 | 训练好、验证差 | 小数据下 MoE 容量过强 | 降低 MoE 密度、增大 dropout、提高正则 |
| token 丢失 | 路由后部分 token 未处理完整 | Capacity Factor 太小、专家过热 | 调 CF、调 batch、加负载均衡损失 |
| 路由塌缩 | 少数专家吃掉大多数 token | router logits 过尖、层太密 | 加 z-loss、温度控制、降低 MoE 密度 |
| 吞吐低 | 理论 FLOPs 不高但实际很慢 | all-to-all 通信重、token 重排成本高 | 减少 MoE 层数、优化专家并行布局 |
| 微调不稳 | 小任务波动大 | 高密度 MoE 在小样本下不稳 | 冻结部分层、只微调部分专家、改用更稀的放置 |
| 专家学不动 | 很多专家长期闲置 | batch 太小、专家太多 | 降低专家数、增大 batch、减少层密度 |

一个真实工程例子是：假设你做多语言预训练，模型很大，训练卡数也多。此时每隔一层放一次 MoE，往往比每层都放更容易达到“吞吐可接受、质量明显提高”的平衡点。反过来，如果你做的是一个中小规模分类微调任务，把原本 dense 模型直接换成全层 MoE，收益可能很小，调参成本却明显上升。

可以把常见选择压成一句经验法则：

- 预训练先担心“容量不够”。
- 微调先担心“稳定性不够”。
- 部署先担心“通信和延迟不够好”。

---

## 替代方案与适用边界

可以把选择逻辑压缩成一句话：

**预训练优先看容量释放，微调优先看稳定与成本。**

### 什么时候选全层 MoE

适用边界：
- 大规模预训练
- 数据量大
- 有成熟的专家并行实现
- 追求最大参数容量
- 可以接受更复杂的路由和部署

Mixtral 这类方案就是典型代表。每层都放 MoE，相当于让“专家选择”贯穿整个深度路径。每个 token 只激活少量专家，所以单 token 计算没有按总参数线性增长，但总模型容量非常大。

更直白地说，全层 MoE 适合下面这类目标：

| 目标 | 是否适合全层 MoE |
|---|---|
| 尽量把总参数做大 | 适合 |
| 已有强并行与通信优化 | 适合 |
| 小规模任务快速落地 | 通常不适合 |
| 希望最少调参成本 | 不适合 |

### 什么时候选交替层 MoE

适用边界：
- FLOPs 或吞吐预算明确受限
- 硬件通信能力一般
- 训练稳定性优先
- 希望在 dense 基线之上稳步增益

Switch Transformer 的代表性思路就是：不必每层都稀疏化，只要在足够多的层上放专家，就能拿到明显收益，同时把实现复杂度控制住。

交替层的一个重要优点是：  
dense 层会在深度中起到“稳定器”作用。它不能完全消除路由噪声，但能减少“每层都重新分流”带来的连锁放大。

### 什么时候选更稀疏的放置，例如每 4 层一次

适用边界：
- 小到中等规模数据
- 需要降低路由不稳定
- 训练资源有限
- 正在从 dense 模型向 MoE 平滑迁移

ST-MoE 给出的一个重要工程启示是：**MoE 密度不是越高越好，层间稀疏放置可以成为正则化的一部分。**

每 4 层一次的意义，不只是“更省”，也是“更稳”：
- 路由次数明显减少
- 累计 drop 风险下降
- 调参空间缩小
- 更容易定位问题究竟来自 MoE 本身还是其他训练设置

### 替代方案

如果你的目标并不是“以稀疏方式扩大参数量”，还可以考虑下面几类替代思路：

| 方案 | 核心思路 | 适用边界 |
|---|---|---|
| 纯 dense 扩宽/加深 | 直接扩大 FFN 或层数 | 硬件简单、实现稳定 |
| Top-1 MoE | 每个 token 只选 1 个专家 | 通信更低，但表达力可能下降 |
| Top-2 MoE | 每个 token 选 2 个专家 | 常见折中，表达与成本更平衡 |
| 动态 K 路由 | 不同 token 选不同数量专家 | 复杂度更高，适合研究型探索 |
| 共享专家 + 私有专家 | 部分专家跨层共享 | 想降低参数冗余时可试 |
| 只在高层用 MoE | 浅层保持稳定，深层专家化 | 微调和任务适配更常见 |
| 仅专家微调 | 预训练后只更新专家或门控 | 适合参数高效迁移 |

可以用一个简化决策图来理解：

```text
数据很大、做预训练、追求容量
  -> 倾向每层 MoE

预算固定、吞吐敏感、实现要稳
  -> 倾向每隔一层 MoE

数据较小、先求稳定、调参资源有限
  -> 倾向每4层一次 MoE 或直接 dense
```

也可以把它写成一个更工程化的决策表：

| 约束条件 | 优先方案 |
|---|---|
| 通信不是主要瓶颈，目标是最大容量 | 全层 MoE |
| 通信有限，目标是稳定收益 | 隔层 MoE |
| batch 小、数据少、先求可控 | 每 4 层一次 MoE |
| 团队缺少 MoE 训练经验 | 从更稀的放置开始 |
| 任务是小规模微调 | 先比较 dense、隔层 MoE 和只高层 MoE |

所以“每层 MoE”和“交替层 MoE”不是谁替代谁，而是在不同约束下做不同最优。

---

## 参考资料

1. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**，Fedus, Zoph, Shazeer，2021。  
核心结论：MoE 的关键价值在于用稀疏激活扩大参数规模，同时尽量控制计算与通信；论文重点讨论了训练不稳定、路由简化和系统效率问题。  
链接：[https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)

2. **ST-MoE: Designing Stable and Transferable Sparse Expert Models**，Zoph et al.，2022。  
核心结论：稀疏专家模型的效果不仅取决于专家数，还取决于稳定训练、迁移能力、正则项和容量管理；“更大更稀疏”不自动等于“更好用”。  
链接：[https://arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906)

3. **Mixtral of Experts**，Jiang et al.，2024。  
核心结论：每层使用 Top-2 MoE，可以把稀疏容量沿深度铺开；每个 token 虽只激活少量专家，但整个模型总参数和可访问参数显著提升。  
链接：[https://arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)

4. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**，Lepikhin et al.，2020。  
核心结论：MoE 不只是模型结构问题，也是并行切分和跨设备调度问题；专家层一旦跨卡，系统设计会直接影响收益。  
链接：[https://arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668)

5. **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts**，Du et al.，2021。  
核心结论：在大规模预训练下，MoE 可以在较低激活计算下取得强表现，但前提是路由、容量和并行策略足够稳。  
链接：[https://arxiv.org/abs/2112.06905](https://arxiv.org/abs/2112.06905)

6. 如果只想抓住本文最核心的参考方向，可以按这个顺序读：  
Switch Transformer 看“为什么需要简化路由”；  
ST-MoE 看“为什么稳定性和迁移能力不是自动得到的”；  
Mixtral 看“为什么全层 Top-2 MoE 能把容量沿深度拉满”。

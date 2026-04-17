## 核心结论

MoE upcycling 的定义可以直接写成一句话：把已经预训练好的 dense 模型中的 FFN 层扩成多个专家，再新增一个随机初始化的路由器，继续预训练，而不是从零训练一个稀疏模型。

这里先把三个核心名词钉死。

- FFN：Transformer 每层里对单个 token 独立做非线性变换的前馈网络，通常写成两层线性层加激活函数。
- 专家：同一层里并行存在的多个 FFN 副本。
- 路由器：对每个 token 计算“该送去哪些专家”的打分网络。

它之所以有效，不是因为“稀疏天然更强”，而是因为它复用了 dense 预训练已经付出的沉没成本。dense 模型已经学到的通用表示能力，会被完整带入每个专家；新增训练主要做两件事：

1. 让路由器学会把不同 token 分给不同专家。
2. 让一开始完全相同的专家在训练中逐步分化。

因此，upcycling 的真正价值不是“免费变强”，而是“把已有 checkpoint 变成更高参数容量、但单次计算仍可控的模型”。在很多公开结果里，这种改造在中短期预算下都很划算：追加训练 token 不需要太多，就能比 dense 继续训练更快拿到收益，也通常比从零训练同规模 MoE 更早进入可用区间。

Sparse Upcycling 论文中的代表性结论可以压缩成下面这张表。表里“额外预算”指相对于 dense 基座继续追加的训练预算，而不是相对于从零训练全模型的总预算。

| 模型 | upcycling 设置 | 额外训练预算 | 观察到的结果 |
|---|---|---:|---|
| T5-Base | dense FFN 扩成 MoE，继续预训练 | 约 55% | 在下游任务上优于 dense continuation |
| T5-Large | dense FFN 扩成 MoE，继续预训练 | 约 46% | SuperGLUE 相比 dense continuation 提升约 1.5 到 2 分 |
| T5-XL | 同类设置 | 中等额外预算 | 延续同样趋势，upcycled MoE 更快进入高性能区间 |
| ViT-B/16, ViT-L | 视觉 MLP 扩成专家 | 低于 dense 继续扩训所需成本 | 达到同等提升所需的额外训练更少 |

但这个结论有边界。upcycling 不是“永远最优”，它更像“中短期增量改造最划算”的方案。如果你后续还愿意投入非常长的训练预算，那么从零训练的 scratch MoE 可能会追上，甚至反超 upcycled MoE。OLMoE 的附录给了一个很具体的参考：在一个已经训练过 2T token 的 dense 检查点上做 upcycle 后，scratch MoE 大约在再训练 500B token 左右追平，并在约 600B token 左右开始反超。

所以，工程上更实用的结论不是“upcycling 最强”，而是下面这张对比表：

| 方案 | 初始化方式 | 额外 token/算力 | 短期收益 | 长期上限 |
|---|---|---:|---|---|
| Dense 继续训练 | 保持原 FFN，不改结构 | 中 | 稳，但容量增长有限 | 受 dense 容量约束 |
| MoE Upcycling | 复制 dense FFN 到多个专家，路由器随机初始化 | 中低到中 | 常常最快超过 dense | 长跑可能被 scratch MoE 反超 |
| Scratch MoE | 专家和路由器全部随机初始化 | 高 | 前期慢 | 长预算下上限更高 |

---

## 问题定义与边界

问题可以表述成一句话：已经有一个训练好的 dense Transformer，能不能不浪费它已有的预训练结果，直接把它改造成 MoE，并在有限增量预算下拿到更高效果。

这个问题表面看很简单，真正落地时有几个边界必须先说清，否则“upcycling 成功”这句话没有可比性。

第一，upcycling 不是重写整个模型。常见做法是只替换部分 FFN 或 MLP 层，不动 attention、embedding、layer norm、输出头等主干模块。原因很直接：如果你连主干都一起大改，那么收益到底来自“利用旧 checkpoint”，还是来自“重新设计了一个新模型”，就分不清了。

第二，upcycling 必须以已有 dense checkpoint 为前提。没有 checkpoint，就不存在“升级已有能力”的问题，那只是 scratch MoE。checkpoint 可以理解为某一训练阶段保存下来的参数快照，它是 upcycling 的资产来源。

第三，最好还能访问原始数据或至少同分布数据。因为 upcycling 后新增了路由器，专家也要重新分化。如果数据分布同时发生明显变化，那么你测到的就不只是 upcycling 的效果，还叠加了迁移学习或领域适配的因素。这样做当然也可以，但那是在回答另一个问题。

第四，要提前固定实验条件。下面这些变量都不能训练到一半再随意解释，否则结论不具备可复现性。

| 类别 | 需要固定的内容 | 为什么重要 |
|---|---|---|
| 结构 | 替换哪些层、替换多少层 | 决定稀疏容量加在哪里 |
| 稀疏配置 | 专家数 $E$、每个 token 激活的专家数 $K$ | 直接决定计算量与容量 |
| 路由形式 | top-K、expert choice、是否归一化 | 影响稳定性、训练效率和推理一致性 |
| 训练恢复 | 是否继承优化器状态、学习率日程怎么接续 | 决定 upcycle 后是否会抖动 |
| 范围 | 只改编码器、只改解码器，还是全改 | 影响收益位置和工程复杂度 |

对新手来说，一个玩具例子最容易理解。

假设原来某一层只有 1 个 FFN。现在你把它复制成 4 个专家。对 token “猫”“代码”“积分”“SQL” 来说，在初始化时它们无论走哪个专家，结果都几乎一样，因为 4 个专家的参数完全相同。继续训练后，路由器开始学习：编程相关 token 更常送去某几个专家，数学 token 更常送去另一些专家。因为被访问的数据不同，每个专家收到的梯度也不同，于是参数慢慢分开，开始“偏科”。

这里的“偏科”不是人工规定“专家 1 负责代码，专家 2 负责数学”，而是训练自然长出来的分工。MoE 的关键能力，正来自这种数据驱动的自动分工。

再看一个真实工程版本。以 T5-Large 为例，做法不是重新训练整个模型，而是在已经训练好的 dense T5-Large 上，把部分 FFN 层换成 MoE 层，专家从原 FFN 复制而来，路由器随机初始化，然后在原始预训练语料上继续训练。Sparse Upcycling 报告的结果是：在约 46% 的额外预算下，这种做法在 SuperGLUE 上优于 dense continuation。这个结果的工程含义很明确：已有 dense 资产不是废料，可以被直接转化为稀疏大容量模型的起点。

把输入、固定条件和输出再压缩成一张表，更容易形成工程视角：

| 类别 | 内容 |
|---|---|
| 输入 | dense checkpoint、继续预训练语料、原模型结构定义 |
| 固定条件 | 被替换层位置、专家数 $E$、top-K 中的 $K$、路由器随机初始化方式 |
| 可选条件 | 是否恢复优化器状态、是否做路由归一化、是否只 upcycle 部分层 |
| 输出 | 一个可继续训练、可微调、推理时稀疏激活的 MoE 模型 |

---

## 核心机制与推导

MoE 的关键不是“总参数更多”，而是“每个 token 只激活少数参数”。所谓稀疏，可以直接理解为：模型总容量很大，但每次前向传播只动其中一小部分。

先写出标准形式。

设一个 batch 里有 $n$ 个 token，一层有 $E$ 个专家。对于第 $i$ 个 token 的隐状态 $x_i$，路由器会输出长度为 $E$ 的一组分数：
$$
g_i = [g_{i1}, g_{i2}, \dots, g_{iE}]
$$

如果把它经过 softmax，可以得到路由概率：
$$
r_{ij} = \frac{\exp(g_{ij})}{\sum_{m=1}^{E}\exp(g_{im})}
$$

于是对每个 token，都有
$$
r_{ij} \ge 0,\quad \sum_{j=1}^{E} r_{ij}=1
$$

这就是“第 $i$ 个 token 送到第 $j$ 个专家的权重”。

但标准 MoE 不会让 token 经过所有专家，否则计算量就回到了 dense。通常采用 top-K 路由：每个 token 只保留概率最高的 $K$ 个专家。记这组专家索引为 $S_i$，那么该 token 的输出写成

$$
x'_i = \sum_{j\in S_i}\frac{r_{ij}}{\sum_{k\in S_i}r_{ik}} e_j(x_i)
$$

其中：

- $e_j(\cdot)$ 表示第 $j$ 个专家，也就是第 $j$ 个 FFN。
- 分母的作用是重新归一化，因为只保留了 top-K 那几个专家。
- 如果 $K=1$，那就是每个 token 只走 1 个专家。
- 如果 $K=2$，那就是常见的 top-2 路由。

这时单层的总参数量大约从一个 FFN 变成 $E$ 个 FFN，但每个 token 的实际计算只与 $K$ 个专家相关。只要 $K \ll E$，参数容量和单次计算就被分离了。

### 为什么 dense 复制到多个专家后还能工作

upcycling 的初始化可以拆成三步。

第一步，复制 dense FFN 到所有专家。设原始 dense FFN 的参数为 $W_{\text{dense}}$，则初始化时：

$$
W_1^{(0)} = W_2^{(0)} = \cdots = W_E^{(0)} = W_{\text{dense}}
$$

如果 FFN 还有偏置或第二层权重，同样完整复制。此时每个专家都是原 dense FFN 的一份拷贝。

第二步，新增随机路由器。路由器在 dense 模型里没有一一对应的旧模块，因此通常随机初始化。一个简化写法是：

$$
g_i = x_i W_r + b_r
$$

其中 $W_r, b_r$ 是新引入的路由器参数。它们没有历史知识，需要在继续训练中自己学会分工。

第三步，专家分化。初始化时专家相同，但训练后它们会收到不同梯度。原因有三个最常见来源：

- 不同 token 被路由到不同专家。
- mini-batch 采样和 dropout 引入随机性。
- 优化器更新路径随时间分叉。

于是，尽管起点完全一致，专家很快会朝不同方向漂移，形成专长。

这个过程可以用梯度更新写得更直白一点。对专家 $e_j$ 来说，如果第 $t$ 步只有某些 token 被送到它，那么它的更新近似只由这些 token 决定：

$$
W_j^{(t+1)} = W_j^{(t)} - \eta \sum_{i: j\in S_i}\alpha_{ij}\nabla_{W_j}\mathcal{L}_i
$$

其中：

- $\eta$ 是学习率。
- $\mathcal{L}_i$ 是第 $i$ 个 token 对应的损失贡献。
- $\alpha_{ij}$ 可以理解为路由权重或其近似。

只要不同专家接收的 token 子集不同，它们的更新方向就不同，于是参数自然分叉。

### 一个可直接算清楚的 toy 例子

假设某个 token 的路由分数是：

$$
[0.1, 0.6, 0.2, 0.1]
$$

取 top-2，则只保留专家 2 和专家 3。归一化后权重变成：

$$
\frac{0.6}{0.6+0.2}=0.75,\quad \frac{0.2}{0.6+0.2}=0.25
$$

如果这两个专家对同一个输入的输出分别是 8 和 4，那么最终聚合输出为：

$$
0.75 \times 8 + 0.25 \times 4 = 7
$$

其余两个专家这次完全不参与计算。

更重要的是看初始化时会发生什么。假设 4 个专家都是从同一个 dense FFN 复制来的，那么一开始这 4 个专家对同一输入都会输出相同值。也就是说，上面的 8 和 4 在初始化那一刻其实会是同一个数，比如都是 7，于是聚合结果依然是 7。这正是 upcycling 能平滑起步的原因：模型函数不会因为“多复制了几个专家”而瞬间坏掉。

### 为什么“复制相同专家”不是白做

新手最容易困惑的一点是：既然一开始所有专家都一样，那复制出来有什么意义？

意义在于“起点质量”，而不在于“起点差异”。

- scratch MoE 的专家从随机参数起步，既没有通用表示能力，也没有稳定函数形状。
- upcycled MoE 的专家从 dense 已学好的函数起步，至少一开始就具备可用能力。

所以复制的作用不是立刻制造多样性，而是确保一开始模型已经站在一个好位置上。之后的多样性由路由和训练产生。Sparse Upcycling 里也观察到：如果专家不复制，而是随机初始化，恢复速度会明显更慢。

### 还需要考虑负载均衡

只说 top-K 还不够，真实 MoE 训练里还必须考虑负载均衡，否则所有 token 可能挤到少数几个专家，其他专家几乎学不到东西。

设第 $j$ 个专家在一个 batch 中接收到的 token 数量为 $c_j$，理想状态下应尽量接近平均值 $nK/E$。很多 MoE 实现会加入一个辅助损失，让路由既考虑任务损失，也考虑专家利用率更均匀。不同论文写法不完全一致，但直觉是一样的：

- 如果某些专家被严重过载，训练和推理都会不稳定。
- 如果某些专家长期空闲，它们虽然占了参数，却没有真正学到能力。

所以，MoE upcycling 的完整机制不是“复制专家 + 加个 router”这么简单，还包括“让 router 会分工，而且分得别太偏”。

---

## 代码实现

实现上最核心的是区分两类参数路径：

1. 原模型已有参数：尽量复制。
2. 新增路由器参数：重新初始化。

下面先给一个可直接运行的最小 Python 例子。它不依赖深度学习框架，只模拟“复制专家 + top-K 路由 + 聚合输出 + 训练后专家分化”的逻辑。保存为 `moe_upcycling_demo.py` 后直接执行 `python moe_upcycling_demo.py` 即可运行。

```python
from dataclasses import dataclass
from typing import List
import random


@dataclass
class DenseFFN:
    w: float
    b: float

    def __call__(self, x: float) -> float:
        # 用最简单的线性层代替真实 FFN，方便演示
        return self.w * x + self.b


@dataclass
class Expert:
    w: float
    b: float

    def __call__(self, x: float) -> float:
        return self.w * x + self.b

    def step(self, grad_w: float, grad_b: float, lr: float) -> None:
        self.w -= lr * grad_w
        self.b -= lr * grad_b


def upcycle_dense_to_experts(dense: DenseFFN, num_experts: int) -> List[Expert]:
    return [Expert(w=dense.w, b=dense.b) for _ in range(num_experts)]


def topk_indices(scores: List[float], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(scores):
        raise ValueError("k cannot exceed number of experts")
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def normalize_selected(scores: List[float], chosen: List[int]) -> List[float]:
    denom = sum(scores[i] for i in chosen)
    if denom <= 0:
        raise ValueError("selected router scores must sum to a positive number")
    return [scores[i] / denom for i in chosen]


def moe_forward(x: float, experts: List[Expert], router_scores: List[float], k: int) -> float:
    chosen = topk_indices(router_scores, k)
    weights = normalize_selected(router_scores, chosen)
    return sum(weight * experts[idx](x) for idx, weight in zip(chosen, weights))


def train_one_token(
    x: float,
    target: float,
    experts: List[Expert],
    router_scores: List[float],
    k: int,
    lr: float,
) -> float:
    chosen = topk_indices(router_scores, k)
    weights = normalize_selected(router_scores, chosen)

    expert_outputs = [experts[idx](x) for idx in chosen]
    y = sum(weight * out for weight, out in zip(weights, expert_outputs))
    loss = 0.5 * (y - target) ** 2

    # 只更新被选中的专家，模拟“不同 token 只让部分专家收到梯度”
    dloss_dy = y - target
    for idx, weight in zip(chosen, weights):
        grad_w = dloss_dy * weight * x
        grad_b = dloss_dy * weight
        experts[idx].step(grad_w=grad_w, grad_b=grad_b, lr=lr)

    return loss


def main() -> None:
    random.seed(0)

    dense = DenseFFN(w=2.0, b=1.0)
    experts = upcycle_dense_to_experts(dense, num_experts=4)

    print("=== Step 1: upcycling 后，所有专家与 dense 完全一致 ===")
    for i, expert in enumerate(experts):
        print(f"expert_{i}: y(3.0) = {expert(3.0):.4f}")
    print(f"dense: y(3.0) = {dense(3.0):.4f}")

    scores = [0.1, 0.6, 0.2, 0.1]
    y_init = moe_forward(3.0, experts, scores, k=2)
    print(f"\n初始化时，MoE 输出 = {y_init:.4f}")
    assert abs(y_init - dense(3.0)) < 1e-9

    print("\n=== Step 2: 用不同 token 模拟训练，让专家分化 ===")
    samples = [
        # x, target, router_scores
        (1.0, 4.0, [0.7, 0.2, 0.1, 0.0]),  # 更偏向 expert 0
        (2.0, 7.0, [0.6, 0.3, 0.1, 0.0]),
        (3.0, 1.0, [0.1, 0.7, 0.2, 0.0]),  # 更偏向 expert 1
        (4.0, 0.0, [0.1, 0.8, 0.1, 0.0]),
    ]

    for step in range(20):
        x, target, router_scores = random.choice(samples)
        loss = train_one_token(
            x=x,
            target=target,
            experts=experts,
            router_scores=router_scores,
            k=2,
            lr=0.05,
        )
        if step in {0, 4, 9, 19}:
            print(f"step={step + 1:02d}, loss={loss:.4f}")

    print("\n=== Step 3: 训练后专家参数开始不同 ===")
    for i, expert in enumerate(experts):
        print(f"expert_{i}: w={expert.w:.4f}, b={expert.b:.4f}")

    print("\n=== Step 4: 同一个输入，不同路由会得到不同输出 ===")
    x = 3.0
    route_a = [0.8, 0.1, 0.1, 0.0]
    route_b = [0.1, 0.8, 0.1, 0.0]
    y_a = moe_forward(x, experts, route_a, k=2)
    y_b = moe_forward(x, experts, route_b, k=2)

    print(f"route_a output = {y_a:.4f}")
    print(f"route_b output = {y_b:.4f}")


if __name__ == "__main__":
    main()
```

这个例子展示了三个关键事实。

第一，upcycling 刚完成时，MoE 层的函数值和原 dense 层几乎一致，因为专家是复制来的。  
第二，训练时不同 token 只更新被选中的专家，因此专家会自然分化。  
第三，分化之后，相同输入在不同路由下会得到不同输出，这就是专家分工真正开始发挥作用的时候。

如果想把这个 toy 代码映射回真实 Transformer，可以建立下面这组对应关系：

| toy 代码元素 | 真实模型里的含义 |
|---|---|
| `DenseFFN` | Transformer block 中的 FFN/MLP |
| `Expert` | MoE 层中的单个专家 FFN |
| `router_scores` | 路由器对每个 token 输出的 logits 或概率 |
| `topk_indices` | top-K 路由选择 |
| `train_one_token` | 简化后的反向传播，只保留“被选中的专家收到梯度”这个核心机制 |

更接近工程的伪代码如下：

```python
# pseudo-code: replace a dense FFN block with MoE
dense_ffn = model.layers[i].ffn
moe = MoELayer(num_experts=8, top_k=2)

for expert in moe.experts:
    expert.load_state_dict(dense_ffn.state_dict())   # 复制 dense FFN 参数

moe.router.reset_parameters()                        # 路由器随机初始化
model.layers[i].ffn = moe                            # 模型手术替换

# 优化器策略
# 1. 原有参数可选恢复 optimizer state
# 2. 新路由器参数重新建状态
optimizer = build_optimizer(model.parameters())
warmup_router_steps = 1000
```

真实工程里还要补上几个通常不能省的细节，否则“代码能跑”和“训练稳定”是两回事。

| 工程点 | 常见做法 | 目的 |
|---|---|---|
| 容量限制 | 给每个专家设置 capacity factor | 防止单个专家过载 |
| 辅助损失 | load balancing loss / router z-loss | 减少路由塌缩 |
| 学习率 | router 单独 warm-up，主干沿用原日程 | 降低 upcycle 初期抖动 |
| 分布式 | expert parallel / all-to-all 通信 | 支撑大规模专家训练 |
| 推理 | 固定 top-K、控制 dispatch 开销 | 避免理论 FLOPs 下降但时延上升 |

以 T5 路线为例，可以把实现流程概括为：

1. 加载 dense checkpoint。
2. 按预设规则把一部分 FFN 层替换成 MoE 层。
3. 每层放入若干专家，专家参数从原 FFN 拷贝。
4. 路由器参数随机初始化。
5. 恢复或重建优化器状态，并为路由器设置短 warm-up。
6. 在原始或同分布语料上继续预训练。
7. 监控主任务损失、负载均衡指标、专家利用率和墙钟吞吐。

这套实现里最重要的不是代码量，而是参数迁移边界。你必须明确：

- 哪些参数是复制来的。
- 哪些参数是新增的。
- 哪些优化器状态可以继承。
- 哪些状态必须重置。
- 哪些训练超参要沿用，哪些必须重新调。

很多“论文复现失败”，并不是模型思想本身有问题，而是这些边界没有处理清楚。

---

## 工程权衡与常见坑

upcycling 的核心工程权衡可以写成一句话：你想要更大的总参数容量，但不想为每个 token 都付出 dense 大模型的全额计算。MoE 解决了这个问题，但引入了新的系统复杂度。

最常见的误区，是把“复制专家”误解成“所有新参数都应该继承”。这是错的。正确拆分是：

- 专家参数尽量复制。
- 路由器参数随机初始化。
- 路由相关的优化器状态重新建立。

原因不复杂。专家对应的是 dense 里本来就存在的 FFN，能继承；路由器对应的是 dense 里根本没有的功能模块，无法继承。你如果强行让路由器“继承旧行为”，本质上是在假装 dense 模型里早就存在明确分工，但 dense 并没有这个结构，因此通常会妨碍专家分化。

第二个坑是忽略负载均衡。很多人初看 MoE，只盯着 top-K，却忘了如果所有 token 都涌向同一个专家，那么其余专家几乎不学习，训练吞吐也会下降。MoE 实践里，下面几个指标通常要一起看：

| 指标 | 含义 | 异常表现 |
|---|---|---|
| expert load | 每个专家接收到多少 token | 少数专家过载，多数专家空闲 |
| importance | 路由概率质量分布 | 某几个专家长期占据绝大多数权重 |
| dropped tokens | 超过容量限制后被丢弃或回退的 token | 说明容量设置过紧或路由过偏 |
| router entropy | 路由分布是否过于尖锐 | 太尖容易塌缩，太平又难形成分工 |

第三个坑是把视觉任务里的技巧直接照搬到语言任务。Sparse Upcycling 的经验之一是：某些在视觉侧有帮助的路由权重归一化技巧，迁移到语言任务未必成立，甚至可能伤性能。原因不是“论文矛盾”，而是两类任务的 token 结构、解码方式和序列相关性不同。工程上更稳妥的做法是：跨模态技巧默认不继承，先做小规模 ablation 再上大训练。

第四个坑是盲目给路由器或专家加噪声。直觉上，噪声似乎能帮助专家更快分化，但论文和复现实践里都没看到稳定收益。原因很简单：专家分化需要的是“有信息的梯度差异”，而不是“纯随机扰动”。噪声过大，会直接破坏 dense checkpoint 带来的平滑起点；噪声过小，又基本没有作用。

第五个坑是只看理论 FLOPs，不看墙钟时间。MoE 的纸面计算量经常很好看，但真实系统里还要付出 token 分发、跨设备 all-to-all、批内负载不均和缓存失配等成本。所以“FLOPs 下降”不等于“推理更快”。公开分析里，一些 upcycled 稀疏模型即使训练阶段有效，推理阶段仍可能出现 30% 到 40% 的变慢。对于线上系统，这不是小问题，而是直接决定方案能否部署。

下面把高频坑压缩成表：

| 操作/坑 | 结果 | 规避方式 |
|---|---|---|
| 把专家随机初始化，不复制 dense FFN | 前期恢复很慢，loss 抖动大 | 专家直接复制原 FFN |
| 试图让路由器“继承旧行为” | 专家难以分化，路由僵化 | 路由器随机初始化并单独 warm-up |
| 忽略负载均衡 | 少数专家过载，多数专家学不到东西 | 加辅助损失并监控 expert load |
| 语言模型里强做路由归一化 | 可能拖累下游表现 | 语言任务默认先不做，单独做 ablation |
| 盲目加 Gaussian noise | 常常无明显收益，过大还伤性能 | 不把噪声当默认配置 |
| 训练预算过长仍坚持 upcycling | 后期可能被 scratch MoE 超过 | 中短期预算优先 upcycling，长跑重新评估 |
| 忽略推理通信成本 | 线上时延不达预期 | 提前做专家并行和 dispatch profiling |

如果把这些坑总结成一个最实用的判断标准，那就是：先问自己要解决的是“训练性价比问题”，还是“长期极限性能问题”。

- 如果你只打算再训练原预算的 10% 到 60%，upcycling 往往很有吸引力。
- 如果你本来就准备继续砸一个完整预训练周期，scratch MoE 的长期上限可能更值得追。
- 如果线上部署时延非常敏感，那么即使训练指标好看，也必须提前验证通信和推理路径。

OLMoE 给出的 500B 到 600B token 量级比较之所以重要，不是因为这个数值可以机械复用，而是它提醒你：upcycling 的优势主要发生在“已有资产 + 有限追加预算”这个区间，一旦预算足够长，起点优势会逐渐被 scratch MoE 的长期可塑性消化掉。

---

## 替代方案与适用边界

vanilla upcycling 的优点是简单、稳、好复现，但它不是唯一做法。它的直接问题也很明显：所有专家都从同一个 dense FFN 复制而来，初始多样性很低。对于短期恢复，这通常是好事；对于长期分化，它也可能成为限制。

### 1. Drop-Upcycling：部分继承，部分重置

Drop-Upcycling 想解决的核心问题是：能不能在保留 dense 能力的同时，给专家更大的初始差异。它的做法不是把专家完整复制成一模一样，而是保留一部分 dense 权重，另一部分重新初始化。

可以写成：

$$
W_e^{(0)} = M \odot W_{\text{dense}} + (1-M)\odot W_{\text{rand}}
$$

其中：

- $M$ 是一个与权重同形状的 0/1 掩码。
- $W_{\text{dense}}$ 是 dense 模型的原始参数。
- $W_{\text{rand}}$ 是随机初始化参数。
- $\odot$ 表示逐元素乘法。

如果用 re-init ratio $r$ 表示“被重置的参数比例”，那么：

- $r$ 小：更接近 vanilla upcycling，稳定，但多样性低。
- $r$ 大：初始多样性更强，但训练早期更不稳定。

因此，它本质上是在“保留旧能力”和“释放专家分化空间”之间做折中。

### 2. UpIT：用中间 checkpoint 提供更自然的专家差异

还有一类思路不是从一个 dense checkpoint 复制出多个完全相同的专家，而是从不同训练阶段、不同 instruction tuning 阶段或不同任务路径的 checkpoint 中挑选专家初始化来源。UpIT 就属于这类方向。

它解决的问题是：如果专家一开始就有一些真实差异，而不是纯粹靠后续路由打散，那么专家可能更快学出稳定分工。白话解释是：不是把一个厨师复制 8 份，而是把几位已经在不同菜系上练过一段时间的厨师拉进同一个厨房。

这种方法的前提也更苛刻：你必须已经有多组中间 checkpoint 或多阶段调优产物。否则它在工程上不如 vanilla upcycling 直接。

### 3. DeRS：把专家写成“基座 + 增量”

另一类方法不直接保存每个专家的完整参数，而是把专家表示成：

$$
W_e = W_{\text{base}} + \Delta_e
$$

其中：

- $W_{\text{base}}$ 是共享的基座权重。
- $\Delta_e$ 是每个专家各自的增量参数。

这种方法的目标不是单纯提高训练效果，而是降低参数存储、显存和通信成本。对于部署更敏感的场景，这类结构往往比“每个专家都存一整份 FFN”更有吸引力。

它的代价是实现更复杂，训练和推理框架都要支持“共享基座 + 专家增量”的参数组织方式。

### 4. 方案对比

把这几类方法放在一起，差异会更清楚：

| 方法 | 初始化方式 | 额外训练/算力需求 | 优势 | 适用边界 |
|---|---|---:|---|---|
| Vanilla Upcycling | 全量复制 dense FFN，路由器随机初始化 | 中 | 最简单，复现成本低，短期恢复快 | 中短期增量训练 |
| Drop-Upcycling | 部分复制，部分重置 | 中到中高 | 专家多样性更强，长跑更有潜力 | 愿意多调参，预算更长 |
| UpIT | 用不同中间 checkpoint 生成专家 | 中 | 专家初始差异更自然 | 已有 instruction 或多阶段 checkpoints |
| DeRS | 基座权重 + 专家 delta | 训练中，部署低 | 存储和带宽更省 | 部署成本敏感、系统侧约束强 |

所以适用边界可以压缩成一句话：如果你已经有一个质量不错的 dense 模型，预算有限，目标是在较短追加训练里换到更高容量模型，那么 vanilla upcycling 通常是第一选择；如果你更在意长期专家多样性和最终上限，可以继续评估 Drop-Upcycling 或 scratch MoE；如果部署成本是主矛盾，那么 DeRS 这类“共享基座 + 稀疏增量”的方向更值得关注。

再把决策写得更工程化一点：

| 你的约束 | 更优先考虑的方案 |
|---|---|
| 已有强 dense checkpoint，想尽快拿增量收益 | Vanilla Upcycling |
| 预算比 upcycling 更长，愿意为更高上限调参 | Drop-Upcycling 或 Scratch MoE |
| 已有多阶段 instruction / task checkpoints | UpIT |
| 显存、带宽、存储是主瓶颈 | DeRS 一类参数高效方案 |

---

## 参考资料

- Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*. 论文系统提出 sparse upcycling，并给出 T5 和 ViT 上的代表性结果。  
  https://ar5iv.org/html/2212.05055

- OLMoE, Appendix B.1.2, sparse upcycling comparison. 用更长追加训练预算比较 upcycled MoE 与 scratch MoE，说明 upcycling 的优势主要集中在中短期预算区间。  
  https://proceedings.iclr.cc/paper_files/paper/2025/file/9b224ace8963c9385ad5e2b5c9039b97-Paper-Conference.pdf

- Doubov et al., *Sparse Upcycling: Inference Inefficient Finetuning*. 讨论 upcycled sparse 模型在推理效率上的问题，提醒“训练省算力”不等于“推理一定更快”。  
  https://raw.githubusercontent.com/mlresearch/v262/main/assets/doubov24a/doubov24a.pdf

- Drop-Upcycling 相关工作，*Training Sparse Mixture of Experts with Partial Re-initialization*. 讨论通过部分重置参数来提高专家初始多样性。  
  https://www.emergentmind.com/papers/2502.19261

- Emergent Mind, *MoE Upcycling: Transforming Dense Models*. 适合快速浏览概念、术语和相关变体，但应以原论文结论为准。  
  https://www.emergentmind.com/topics/moe-upcycling-technique

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. 经典 MoE 路由与负载均衡背景，理解 upcycling 前最好先掌握这里的基本机制。  
  https://arxiv.org/abs/1701.06538

- Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. 了解 top-1/top-K 路由、容量限制和负载均衡实现时很有帮助。  
  https://arxiv.org/abs/2101.03961

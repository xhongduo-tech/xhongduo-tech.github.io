## 核心结论

MoE，Mixture of Experts，中文常译为“专家混合模型”，意思是模型里有很多子网络，但每个 token 只激活其中少数几个。Dense 模型，中文可叫“稠密模型”，意思是每个 token 都走完整套参数。两者比较时，最重要的不是“总参数量”，而是“在相同激活计算量下，loss 能降到多低”。

Unified Scaling Law 给出的核心表达式是：

$$
\log L(N,E)=a\log N+b\log \hat E+c\log N\log \hat E+d
$$

其中：

- $L$ 是训练后的 loss
- $N$ 是基座参数规模
- $E$ 是专家数
- $\hat E$ 是经过饱和修正后的有效专家数，意思是“不是所有增加的专家都按 1:1 继续产生收益”

这个式子的结论很直接：在固定 activate FLOPs，也就是固定“每个 token 实际参与计算的浮点运算”时，增加专家数通常能继续降低 loss，因此 MoE 往往比 dense 更会“花同样的钱做更大的模型”。

更实用的近似是：MoE 的有效参数收益大致只按 $E^{0.3}$ 增长。白话说，专家数翻倍，收益不是翻倍，而是只增加一小截，但这“一小截”在大模型区间依然很值钱。

一个适合初学者的玩具例子是：假设一个 10M 的基座模型扩成 32 个专家，按经验换算，有效参数规模可以接近 48M dense 的效果，但每个 token 只激活少数专家，所以计算量没有按 4.8 倍增长。这就是 MoE 的核心吸引力：总参数变大很多，活跃计算只增加一部分。

---

## 问题定义与边界

这类比较必须先把问题说清楚，否则很容易把“总参数量”“活跃参数量”“训练 FLOPs”“推理 FLOPs”混成一件事。

本文讨论的对象是：

| 维度 | 含义 | 这里如何比较 |
|---|---|---|
| 总参数量 | 模型存储了多少参数 | MoE 通常远大于 dense |
| 活跃参数量 | 单个 token 真正经过多少参数 | MoE 接近较小 dense |
| activate FLOPs | 单个 token 实际计算成本 | 用它做公平对比 |
| loss | 训练后误差 | 谁更低谁更优 |

因此，本文的“MoE 优于 dense”不是说 MoE 在所有意义上都优于 dense，而是说：

1. 在相同激活计算预算下，MoE 往往能取得更低 loss。
2. 或者在相同目标 loss 下，MoE 往往需要更少计算。
3. 但这种收益不是无限的，因为专家数存在饱和区间。

这里的饱和区间来自 $\hat E$ 的定义。原始专家数 $E$ 并不会一直完整生效，超过某个范围后，额外专家对 loss 的边际收益会明显下降。论文里进一步给出 cutoff，也就是“专家扩张开始失效的模型规模阈值”。示例数值包括：

- S-BASE 路由下，$N_{\text{cutoff}} \approx 937B$
- HASH 路由下，$N_{\text{cutoff}} \approx 83B$

这两个数字不是“所有系统都必须照抄”的硬阈值，而是一个方向性信号：收益递减不是工程偶然，而是 scaling law 本身就预测得到的现象。

可以把 isoFLOP，等计算量曲线，理解成一条“同样算力预算下谁的 loss 更低”的曲线。新手版理解是：你有固定油量，dense 选择一辆发动机更大的车，MoE 选择更多备选发动机但每次只点火其中几台。前者结构简单，后者在大规模下更省“有效油耗”，但油箱加满以后再继续加专家，收益会越来越不明显。

---

## 核心机制与推导

先看最关键的结构。传统 dense scaling law 只关心参数规模 $N$ 与数据规模、计算规模之间的关系。Unified Scaling Law 则把 routed model，也就是带路由的稀疏模型，单独加了一维：专家数 $E$。

可把它理解成两层修正：

1. 先承认“总参数更多”确实有价值。
2. 但再承认“额外参数是否真的被高效利用”要看路由是否让这些参数对 token 分工。

因此，论文把 loss 近似写成关于 $\log N$ 与 $\log \hat E$ 的双变量函数：

$$
\log L(N,E)=a\log N+b\log \hat E+c\log N\log \hat E+d
$$

其中交叉项 $c\log N\log \hat E$ 的意义是：专家数的收益不是常数，而会随基座模型规模一起变化。白话说，小模型加专家和大模型加专家，不是同一种效果。

进一步地，$\hat E$ 不是直接等于 $E$。论文中引入饱和变换，可以抽象理解为：

$$
\hat E = f(E; \alpha,\beta,\dots)
$$

其中 $f$ 在小 $E$ 区间近似随 $E$ 增长，在大 $E$ 区间逐渐变平。也就是：

$$
\frac{\partial \hat E}{\partial E} > 0,\qquad
\frac{\partial^2 \hat E}{\partial E^2} < 0
$$

这两个不等式的直观意义是：

- 专家数增加，收益仍然增加
- 但每多一个专家，新增收益比前一个更小

把拟合参数代入后，可以得到一个很常见的经验近似：有效参数增益大致满足

$$
N_{\text{eff}} \propto N\cdot E^{0.3}
$$

这里 $N_{\text{eff}}$ 不是模型真实存储参数，而是“如果换成 dense，需要多大参数规模才能接近这个 loss”。这就是 effective parameter count，简称 EPC，有时可理解为“等效稠密参数量”。

下面给出一个简化玩具表格。假设基座模型是 10M：

| 专家数 $E$ | $E^{0.3}$ 近似倍率 | 等效 dense 参数 |
|---|---:|---:|
| 1 | 1.00 | 10M |
| 2 | 1.23 | 12.3M |
| 4 | 1.52 | 15.2M |
| 8 | 1.87 | 18.7M |
| 16 | 2.30 | 23.0M |
| 32 | 2.83 到更高拟合区间 | 约 28M 到 48M |

为什么最后一行是区间而不是单值？因为不同论文、不同路由方式、不同基座规模下，32 专家时的换算不是同一条严格常数曲线。工程博客里更稳妥的说法是：$E^{0.3}$ 给出“次线性增长”的总体规律，而具体 EPC 数值还会受路由策略、专家大小、数据分布影响。研究和实践中常见的“10M + 32 专家约等效 48M dense”可以视为一个代表性案例，而不是所有 MoE 都必须达到的固定倍率。

这也解释了一个容易误解的点：MoE 的收益并不是“免费参数”。它其实是在做参数利用率重分配。dense 模型让每个 token 都走相同大网络，MoE 则让不同 token 走不同专家，因此把总参数池变大，却不把每次前向的计算同步放大。

真实工程例子是 Mixtral 8x7B。这里“8x7B”指 8 个专家，每个专家基于 7B 级别的 Mistral 架构；每个 token 只选 top-2 专家，也就是只走 2 个专家分支。结果是：

- 总参数约 47B
- 活跃参数约 13B
- 推理成本更接近 13B dense，而不是 47B dense

这不是数学游戏，而是生产系统里真正重要的指标：你为部署付出的延迟和吞吐压力，通常更接近活跃参数，而不是总参数。

---

## 代码实现

下面给一个可运行的 Python 玩具脚本，用来模拟“给定基座参数 $N$ 与专家数 $E$，估算等效 dense 参数和简化 loss”。这里不是复现论文原始拟合，而是把文中的直觉写成最小可运行版本，便于验证趋势。

```python
import math

def effective_experts(E, saturation=64.0):
    # 一个简单的饱和函数：E 越大，收益越趋缓
    assert E >= 1
    return saturation * (1 - math.exp(-E / saturation))

def effective_params(N, E, alpha=0.30, saturation=64.0):
    # N: 基座参数量
    # E: 专家数
    e_hat = effective_experts(E, saturation=saturation)
    return N * (e_hat ** alpha)

def toy_loss(N, E, a=-0.08, b=-0.03, c=-0.01, d=2.0):
    # 仅用于演示 log-law 结构，不代表论文原始系数
    e_hat = effective_experts(E)
    logL = a * math.log(N) + b * math.log(e_hat) + c * math.log(N) * math.log(e_hat) + d
    return math.exp(logL)

base_N = 10_000_000
dense = effective_params(base_N, 1)
moe_8 = effective_params(base_N, 8)
moe_32 = effective_params(base_N, 32)

assert dense > 0
assert moe_8 > dense
assert moe_32 > moe_8
assert toy_loss(base_N, 32) < toy_loss(base_N, 8) < toy_loss(base_N, 1)

for E in [1, 2, 4, 8, 16, 32]:
    print(f"E={E:>2d} | effective_params={effective_params(base_N, E)/1e6:6.2f}M | toy_loss={toy_loss(base_N, E):.4f}")
```

这段代码体现了三件事：

1. `effective_experts` 把原始专家数 $E$ 压成饱和后的 $\hat E$。
2. `effective_params` 用 $N \cdot \hat E^{0.3}$ 近似等效 dense 参数。
3. `toy_loss` 用一个简化版 log-law 展示“$E$ 增加时 loss 下降，但下降速度逐渐变慢”。

如果在训练脚本里做实验，最实用的不是直接相信公式，而是把不同 `E` 的 isoFLOP 结果打成表。伪代码可以写成：

```python
def evaluate_isoflop(models, flops_budget):
    rows = []
    for model in models:
        # 假设这里已经约束了每个模型的 active FLOPs 接近同一预算
        val_loss = run_validation(model)
        rows.append({
            "name": model.name,
            "total_params": model.total_params,
            "active_params": model.active_params,
            "experts": model.num_experts,
            "val_loss": val_loss,
        })
    return sorted(rows, key=lambda x: x["val_loss"])
```

一个新手能直接拿去理解的结果表大概长这样：

| 模型 | 总参数 | 活跃参数 | 专家数 | isoFLOP 下验证 loss |
|---|---:|---:|---:|---:|
| Dense-13B | 13B | 13B | 1 | 1.85 |
| MoE-8E | 47B | 13B | 8 | 1.78 |
| MoE-16E | 89B | 13B | 16 | 1.75 |
| MoE-32E | 173B | 13B | 32 | 1.74 |

这张表想表达的不是“数字一定如此”，而是同一条规律：在活跃计算差不多时，增加专家数通常能继续降 loss，但从 16 到 32 的收益已经明显变小，这就是收益递减。

---

## 工程权衡与常见坑

MoE 的第一大坑是把“总参数更大”误解成“白拿收益”。实际系统里，路由、通信、负载均衡都会额外收费。

先看一个工程对比表：

| 架构 | 总参数 | 活跃参数 | 单 token 计算 | 主要额外成本 |
|---|---:|---:|---:|---|
| Dense-13B | 13B | 13B | 高且固定 | 结构简单 |
| Mixtral 8x7B | 47B | 约13B | 接近 13B dense | 路由、跨卡通信、负载不均 |
| 更大专家数 MoE | 更高 | 可控制 | 不一定线性上升 | 专家空转、溢出、显存抖动 |

最常见的问题有五类。

第一，负载不均衡。路由器，router，就是决定 token 去哪个专家的模块。如果很多 token 同时挤进少数专家，会出现有的专家过载、有的专家闲置，最终吞吐下降，loss 也可能变差。工程上通常加 auxiliary loss，辅助损失，用来鼓励更平均的专家利用率，但这个项不能过强，否则模型为了“看起来平均”而牺牲真正的任务学习。

第二，capacity factor 设置不当。它的意思是“每个专家允许接收的 token 容量上限倍率”。太小会丢 token 或频繁回退，太大会增加 padding 和显存浪费。实践里常配合 top-k 路由一起调，比如 top-1 或 top-2，再测不同 batch size 下的吞吐。

第三，跨设备通信成本。MoE 非常依赖专家并行，专家如果散在多张卡上，token dispatch，也就是 token 分发，与 combine，也就是结果回收，会带来大量 all-to-all 通信。理论上 FLOPs 没涨太多，实际上 wall-clock time，也就是实际训练时间，可能涨得很明显。

第四，误判饱和区。很多团队看到 8 专家有效，就自然想上 32、64、128。但 scaling law 告诉你，$\hat E$ 会饱和。也就是说，专家数不是越多越好，必须通过等算力实验验证自己是否已经进入边际收益很低的区间。

第五，部署侧忽略热路径。线上推理不只关心平均成本，还关心 P99 延迟，也就是最慢 1% 请求的尾延迟。MoE 的路由随机性、批次组成变化、不同专家热点，都可能让尾延迟比 dense 更难控制。

一个常见的路由正则项可以抽象写成：

$$
L_{\text{total}} = L_{\text{task}} + \lambda L_{\text{balance}}
$$

其中：

- $L_{\text{task}}$ 是原任务 loss
- $L_{\text{balance}}$ 是负载均衡项
- $\lambda$ 是权重

$\lambda$ 太小，专家失衡；太大，主任务退化。这个参数通常没有通用最优值，只能依赖具体训练曲线和集群 profile 去调。

---

## 替代方案与适用边界

MoE 不是 dense 的简单替代，而是一个在特定区间更划算的架构选择。

先给一个决策矩阵：

| 方案 | 什么时候更合适 | 优点 | 限制 |
|---|---|---|---|
| Dense | 小到中等模型、部署要求简单 | 稳定、易训、易部署 | 同算力下参数池受限 |
| MoE | 大模型、训练批量大、集群通信能力强 | 同激活算力下可获得更低 loss | 系统复杂，收益会饱和 |
| 其他稀疏方法 | 想降成本但不想引入完整路由系统 | 实现更简单 | 参数利用率通常不如优质 MoE |

对初级工程师最有价值的判断规则是：

1. 如果模型规模不大、资源有限、推理系统要求极稳，先选 dense。
2. 如果已经进入大模型区间，且瓶颈是“同样 FLOPs 下效果不够”，再考虑 MoE。
3. 如果集群网络差、跨卡通信贵，MoE 的理论收益可能会被系统开销吃掉。
4. 如果实验显示从 8 专家到 16 专家提升明显，但从 16 到 32 几乎不动，说明你可能已接近本任务的饱和区。

还要注意，论文里的 cutoff 数值，例如 S-BASE 约 937B、HASH 约 83B，是在特定设定下得到的拟合结果。它们的价值在于提醒你“存在上限”，而不是让你把它们当成本项目的绝对边界。

因此，更稳妥的工程结论是：MoE 的优势主要成立于“大规模、等激活计算对比、路由开销可控”的场景。离开这三个条件，dense 往往仍是更好的默认选择。

---

## 参考资料

1. Clark et al., *Unified Scaling Laws for Routed Language Models*，给出把参数规模 $N$ 与专家数 $E$ 统一写进 loss 的 scaling law，是本文公式与 cutoff 讨论的直接来源。  
2. Cerebras, *MoE Fundamentals / Why MoE*，用 isoFLOP 视角解释为什么 MoE 能在相同激活计算预算下优于 dense，并强调工程上的负载均衡与通信成本。  
3. Ryan Lee, *Mixtral 8x7B* 相关文章，用 Mixtral 8x7B 说明“总参数很大但活跃参数较小”的工业落地形态。  
4. Dawson Chen, 关于 MoE scaling law 的解读文章，提供了更适合直观理解的 effective parameter count 示例，包括 10M 基座加 32 专家的换算讨论。

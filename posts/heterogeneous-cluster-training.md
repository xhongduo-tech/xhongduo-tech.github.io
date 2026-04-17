## 核心结论

异构集群训练的核心不是“把层平均分给每张卡”，而是“让每个流水线阶段的完成时间尽量接近”。异构集群，指 GPU 算力、显存、互联带宽不一致的训练集群；混合 A100/H100、跨 NVLink/PCIe/Ethernet 的部署都属于这一类。对这类集群继续使用均匀并行策略，通常会触发木桶效应，也就是最慢阶段决定整轮训练速度。

AMP（Adaptive Model Parallelism，中文可理解为“自适应模型并行”）做的事，是在 TP、PP、DP、micro-batch 这几个维度上联合搜索。TP 是张量并行，白话说是一层内部拆给多卡一起算；PP 是流水线并行，白话说是把模型层切成多个连续阶段；DP 是数据并行，白话说是多份模型副本各自处理不同样本；micro-batch 是把一个大 batch 再切成更小的训练片段，方便流水线连续流动。AMP 的目标不是追求某个维度最大，而是最小化最长阶段时间。

一个最小玩具例子就能看出差异。假设有 2 张 A100 和 2 张 H100，H100 计算更快、卡间链路也更好。如果把 6 层模型机械地平均分成 4 份，慢卡又碰上通信重的层，最长阶段可能达到 2.72s；AMP 会把重层和重通信阶段更多放到快卡和高带宽链路，让慢卡只承担较轻阶段，整轮迭代可降到 1.28s。这个例子说明：在异构集群里，“平均分工作量”不等于“平均分时间”。

下面的表先给出一个直观对比：

| 策略 | GPU/阶段分配 | 阶段时间（s） | 最长阶段（s） | 估算迭代时间（s） | 吞吐提升 |
|---|---|---:|---:|---:|---:|
| 均匀层数 | A100: L1-L2, A100: L3, H100: L4-L5, H100: L6 | 2.72 / 2.10 / 1.46 / 1.31 | 2.72 | 2.72 | 1.00x |
| AMP | H100: 重层+重通信, H100: 重层, A100: 轻层, A100: 轻层 | 1.24 / 1.28 / 1.19 / 1.21 | 1.28 | 1.28 | 2.13x |

这也是 AMP 在论文和工程实践里有效的原因：它不假设“设备相同、层相同、链路相同”，而是把差异显式建模，然后根据代价模型选策略。实际系统中常见的收益区间是 1.1x 到 2.7x；若和简单均匀策略相比，20% 到 30% 的吞吐增益并不罕见，在更严重异构场景下还会更高。

---

## 问题定义与边界

问题可以表述为：已知模型各层的计算耗时、激活大小、参数大小，以及集群中每张 GPU 的计算速率和各链路带宽，如何选择 TP/PP/DP 组合、层切分点、设备放置和 micro-batch 数，使每轮训练时间最小，同时不破坏可部署性。

这里的“可部署性”有三个边界。

第一，策略必须落在现有训练框架能实现的范围内。比如你可以调 PP 切分和 micro-batch，但不能假设框架支持任意细粒度跨层重排，除非你真的实现了对应调度器。

第二，优化目标不是单卡最快，而是整轮吞吐最大。吞吐，白话说是单位时间内能完成多少训练样本。某一张快卡闲着，和某一张慢卡拖慢整轮，后者通常更致命。

第三，代价模型必须同时考虑计算与通信。通信，白话说就是卡之间交换激活、梯度、参数的成本。如果只看 FLOPs，不看链路，你会把通信密集层误放到低带宽节点，结果比“少算一点”更糟。

下面这个表把异构来源和影响列清楚：

| 异构维度 | 典型表现 | 直接影响 | 如果忽略会发生什么 |
|---|---|---|---|
| GPU 计算速率 | H100 快于 A100，A100 快于 V100/T4 | 层前向/反向时间不同 | 慢卡成为最长阶段 |
| 互联带宽 | NVLink > PCIe > 以太网 | 激活/梯度传输时间不同 | 通信密集阶段堵在慢链路 |
| 模型层负载 | Attention、MLP、Embedding 开销不同 | 各阶段计算量不均 | “按层数均分”仍然失衡 |
| 显存容量 | 不同卡可承载不同层数或 batch | 限制切分与 micro-batch | 搜出来的最优策略不可落地 |

新手版例子可以这样理解。假设两张卡之间是 NVLink，另外两张卡之间只有 PCIe。如果把需要频繁传大激活张量的层切在 PCIe 边界，哪怕两边 GPU 算力都不差，通信也会把流水线拖慢。也就是说，异构不只是“卡快慢不同”，还包括“卡之间连接快慢不同”。

---

## 核心机制与推导

AMP 的核心是一个代价模型。代价模型，白话说就是“先不真跑，把一次训练会花多久用公式估出来”。对于流水线并行，常见估计式为：

$$
T_{\text{pp}}=(gas-1)\cdot \max_{1\le i\le pp} t_i + \sum_{i=1}^{pp-1} e_i + \sum_{i=1}^{pp} t_i
$$

其中：

- $pp$ 是流水线阶段数。
- $gas$ 是内部累积的 micro-batch 数，近似理解为一轮里有多少小批次排队流经流水线。
- $t_i$ 是第 $i$ 个阶段的计算时间，包含该阶段所有层的前向与反向。
- $e_i$ 是阶段 $i$ 到阶段 $i+1$ 的通信时间，通常与激活大小、链路带宽、协议开销有关。

这个式子里最重要的是第一项：

$$
(gas-1)\cdot \max_i t_i
$$

它表示流水线稳定运行后，额外 micro-batch 会被最慢阶段“卡住”。这就是为什么慢阶段不是“慢一点点”，而是会在多个 micro-batch 上重复放大。若某阶段是 1.8s，其他阶段都在 1.1s 左右，那么每多一个 micro-batch，整轮时间都会继续被 1.8s 主导。

对异构集群，还要把 $t_i$ 和 $e_i$ 展开：

$$
t_i \approx \sum_{l \in S_i}\frac{c_l}{r(g_i)} + \Delta^{tp}_i + \Delta^{dp}_i
$$

$$
e_i \approx \frac{a_i}{b(g_i,g_{i+1})} + \delta_i
$$

含义是：

- $S_i$ 是分给第 $i$ 个阶段的层集合。
- $c_l$ 是第 $l$ 层的基础计算量。
- $r(g_i)$ 是设备 $g_i$ 的有效计算速率。
- $\Delta^{tp}_i$、$\Delta^{dp}_i$ 是 TP 和 DP 带来的额外同步开销。
- $a_i$ 是阶段间传输的激活大小。
- $b(g_i,g_{i+1})$ 是相邻阶段设备间的有效带宽。
- $\delta_i$ 是启动延迟、协议额外开销等非理想项。

于是问题就变成：怎样选择层集合 $S_i$、设备 $g_i$、TP/DP 规模和 $gas$，使 $T_{\text{pp}}$ 最小。

下面给一个玩具例子。假设 4 个阶段，$gas=4$：

| 阶段 | 设备 | 层集合 | 计算时间 $t_i$ | 通信时间 $e_i$ | 估算总阶段压力 |
|---|---|---|---:|---:|---:|
| 1 | H100 + NVLink | L1-L2 | 1.10 | 0.08 | 1.18 |
| 2 | H100 + NVLink | L3 | 1.28 | 0.06 | 1.34 |
| 3 | A100 + PCIe | L4-L5 | 1.19 | 0.17 | 1.36 |
| 4 | A100 | L6 | 1.21 | 0.00 | 1.21 |

最坏阶段接近 1.28 到 1.36s，比较均衡；如果把 L3-L4 这种激活更大的边界切到 PCIe 上，$e_2$ 可能直接上升到 0.35s，整轮时间马上恶化。AMP 的搜索逻辑本质上就是不断试探这些组合，优先压低最长阶段。

真实工程例子更典型。一个 70B 级模型部署在 8 张 GPU 上，其中 4 张 H100 在同一节点，4 张 A100 在另一节点，节点内 NVLink、节点间 100GbE。若把 attention-heavy 的层平均切开，跨节点的激活同步会频繁打到以太网；AMP 通常会把通信密集阶段尽量收敛到节点内，把跨节点边界放在激活较小或切分较浅的位置，再结合 DP/TP 规模重排，避免链路成为瓶颈。

---

## 代码实现

实现上不需要一开始就写完整搜索器。先做一个可运行的简化版 cost model，再逐步加入 TP、DP、显存约束和异步调度，通常更稳。

最小实现需要的输入输出如下：

| 类型 | 具体内容 |
|---|---|
| 输入 | `layer_times`：每层基础耗时 |
| 输入 | `activation_sizes`：阶段边界激活大小 |
| 输入 | `device_speed`：各 GPU 相对计算速率 |
| 输入 | `bandwidths`：设备间有效带宽 |
| 输入 | `gas`：micro-batch 数 |
| 输入 | `pp`：阶段数 |
| 输出 | `assignments`：每阶段分到哪些层、落在哪张卡 |
| 输出 | `estimated_latency`：估算总迭代时间 |
| 输出 | `bottleneck_stage`：当前最慢阶段 |

下面是一个简化版 Python 示例。它没有覆盖完整 TP/DP 联合搜索，但已经体现了 AMP 的核心思路：不是均分层数，而是枚举切分和设备放置，最小化最长阶段时间。

```python
from itertools import permutations

def estimate_pipeline_time(stage_times, edge_times, gas):
    assert len(edge_times) == len(stage_times) - 1
    return (gas - 1) * max(stage_times) + sum(edge_times) + sum(stage_times)

def build_stage_time(layers, speed):
    # layers 是该阶段包含的层基础耗时，speed 越大表示卡越快
    return sum(layers) / speed

def build_edge_time(act_size, bandwidth):
    # 简化通信模型：时间 = 数据量 / 带宽
    return act_size / bandwidth

def split_layers(layer_costs, cuts):
    parts = []
    start = 0
    for end in cuts:
        parts.append(layer_costs[start:end])
        start = end
    parts.append(layer_costs[start:])
    return parts

def search_best_strategy(layer_costs, act_sizes, devices, gas=4):
    # devices: [{"name": "H100", "speed": 1.4}, ...]
    n_layers = len(layer_costs)
    best = None

    # 这里固定 4 个阶段，枚举 3 个切点
    for c1 in range(1, n_layers - 2):
        for c2 in range(c1 + 1, n_layers - 1):
            for c3 in range(c2 + 1, n_layers):
                stages = split_layers(layer_costs, [c1, c2, c3])

                for placement in permutations(devices, 4):
                    stage_times = [
                        build_stage_time(stages[i], placement[i]["speed"])
                        for i in range(4)
                    ]
                    edge_times = []
                    for i in range(3):
                        bw = placement[i]["links"][placement[i + 1]["name"]]
                        edge_times.append(build_edge_time(act_sizes[i], bw))

                    total = estimate_pipeline_time(stage_times, edge_times, gas)
                    bottleneck = max(stage_times)

                    candidate = {
                        "cuts": (c1, c2, c3),
                        "placement": [d["name"] for d in placement],
                        "stage_times": stage_times,
                        "edge_times": edge_times,
                        "total": total,
                        "bottleneck": bottleneck,
                    }

                    if best is None or candidate["total"] < best["total"]:
                        best = candidate
    return best

devices = [
    {"name": "A100_0", "speed": 1.0, "links": {"A100_1": 64, "H100_0": 16, "H100_1": 16}},
    {"name": "A100_1", "speed": 1.0, "links": {"A100_0": 64, "H100_0": 16, "H100_1": 16}},
    {"name": "H100_0", "speed": 1.4, "links": {"H100_1": 128, "A100_0": 16, "A100_1": 16}},
    {"name": "H100_1", "speed": 1.4, "links": {"H100_0": 128, "A100_0": 16, "A100_1": 16}},
]

layer_costs = [1.2, 1.0, 1.6, 1.5, 0.8, 0.7]
act_sizes = [4.0, 6.0, 2.0]  # GB, 简化表示
best = search_best_strategy(layer_costs, act_sizes, devices, gas=4)

assert best is not None
assert len(best["stage_times"]) == 4
assert best["total"] >= best["bottleneck"]
print(best)
```

这段代码的意义有三点。

第一，它把“设备速度”和“链路带宽”同时纳入估计，而不是只看层数。  
第二，它用 `max(stage_times)` 代表瓶颈阶段，这和前面的公式一致。  
第三，它为后续扩展留了位置，比如把 `speed` 换成 profile 得到的前向/反向时间，把 `links` 换成真实拓扑矩阵，再把 TP/DP 的同步项加进去。

若要进一步贴近真实系统，调度器还会加入异步流水线。异步流水线，白话说就是不要求所有阶段严格同一步前进，而是让快卡先继续吃后续 micro-batch。常见补丁包括：

- `1F1B`：一次前向、一次反向交替推进，减少显存压力。
- `token-throttling`：限制某些阶段进入过多 micro-batch，避免队列爆炸。
- `weight stashing`：保存旧版本权重，确保异步反向仍能拿到对应前向时的权重快照。

---

## 工程权衡与常见坑

AMP 不是“搜一下就完事”，它解决的是性能最优化问题，但工程上还要面对收敛、稳定性和实现复杂度。

最常见的坑是忽略带宽差。很多团队会认真 profile 每层算多久，却把所有卡间通信都当成一个常数。这在同机 NVLink 环境下问题不大，但一旦跨节点或混用 PCIe，就会严重失真。结果通常是模型切分看似平衡，线上一跑却发现某个边界总在等激活。

第二个坑是低估 pipeline bubble。bubble，白话说是流水线里因为上下游不同步而产生的空转时间。$gas$ 越大，理想情况下设备利用率越高；但如果某阶段明显更慢，更多 micro-batch 只会把等待放大。也就是说，增加 micro-batch 不是无条件收益，它依赖阶段平衡。

第三个坑是异步带来的权重陈旧。权重陈旧，白话说是反向传播用到的权重版本已经不是前向时那一版。这样会带来梯度偏差，严重时让收敛变慢或不稳定。工程上通常要加补偿机制。

下面把常见问题和规避手段汇总：

| 坑点 | 现象 | 原因 | 常见规避手段 |
|---|---|---|---|
| 忽略带宽差 | 理论最优，实测很慢 | 通信边界被切到慢链路 | 在 cost model 中显式加入链路惩罚 |
| 只按层数均分 | 部分阶段长期排队 | 层计算量与激活大小不均 | 用 profile 后的层耗时重新切分 |
| gas 盲目增大 | 显存升高，吞吐反降 | 最慢阶段重复放大 bubble | 联合搜索 gas 与 PP 切分 |
| 异步训练不稳 | loss 抖动、收敛变慢 | 权重陈旧和梯度延迟 | weight stashing、delay LR、版本控制 |
| 只优化计算不优化内存 | 搜出来策略跑不起来 | 显存约束未纳入搜索 | 把 activation checkpoint、KV/cache、optimizer state 一并建模 |

新手可以记住一个简单判断：如果某阶段比其他阶段慢两倍，那么 gas 越大，流水线里“堆车”的现象越明显。这时不要先加 batch，而应该先重切分层，或者把通信更重的边界移到高带宽链路上。

真实工程里还有一个常见取舍：为了减少跨节点通信，你可能会把更多重层堆到同一节点，结果显存更紧。这时往往要配合 activation checkpoint。它的白话意思是“中间结果不全存，反向时再算一遍”，用额外算力换显存空间。AMP 搜索若不把这类开关纳入约束，最优解会停留在纸面上。

---

## 替代方案与适用边界

AMP 不是唯一方案。它适合“设备异构 + 带宽异构 + 层异构”同时存在的场景，也就是最复杂的那类生产集群。但如果问题结构更简单，别的方案可能更便宜。

PipePar 可以理解为“更聚焦流水线切分”的方法。它强调按 GPU 型号和链路带宽做动态规划，把层切得更均衡。如果你的主要问题是 PP 切分不合理，而 TP/DP 维度基本固定，PipePar 这种专注切分的方案更轻量。

纯异步流水线方案则适合另一类场景：设备基本同构，但通信延迟高、同步损耗大。这时主要矛盾不在“谁更慢”，而在“大家都在等”。1F1B、queue-based pipeline、temporally disaggregated schedule 之类方法，往往就足够明显改善利用率。

下面给出一个选择表：

| 方案 | 适合场景 | 优点 | 局限 |
|---|---|---|---|
| AMP | GPU、链路、层负载都异构 | 能联合搜索 TP/PP/DP/放置 | 实现复杂，依赖 profile 质量 |
| PipePar | 主要是流水线切分失衡 | 动态规划切分清晰、工程改动较小 | 对 TP/DP 联动考虑较弱 |
| Async 1F1B | 设备较同构，但同步等待重 | 调度成熟、显存压力可控 | 不能自动解决异构失衡 |
| Queue/异步变体 | 快卡多、通信长尾明显 | 能让快设备持续回填任务 | 需要处理 staleness 和调度复杂度 |
| 手工 heuristic | 小规模集群、快速试验 | 落地快 | 难以跨场景复用，常被局部最优困住 |

新手版判断可以很直接：

- 如果 GPU 基本一样，但网络延迟高，先看 async 1F1B。
- 如果 GPU 混用，且 NVLink/PCIe/以太网同时存在，优先上 AMP 或 AMP + PipePar。
- 如果模型不大、卡数不多，先用 heuristics 和 profile 做粗平衡，避免一开始投入过高实现成本。
- 如果模型足够大、训练成本高，AMP 的搜索开销通常值得，因为它换来的是长期吞吐收益。

AMP 的适用边界也要说清楚。它依赖 profile 数据，因此对动态 shape、样本长度剧烈波动、MoE 门控高度不稳定等场景，估计误差会放大。此时更适合把静态搜索和在线自适应调度结合，而不是只靠离线一次性搜索。

---

## 参考资料

1. Amp: Automatically Finding Model Parallel Strategies. NeurIPS 2022. 讲 3D 并行策略搜索、代价模型和异构集群实验，是本文主参考。
2. PipePar: Enabling Fast DNN Pipeline Parallel Training in Heterogeneous GPU Clusters. 讲按 GPU 型号和链路带宽做流水线层切分的动态规划。
3. Asynchronous Pipeline Parallelism. Emergent Mind, 2026. 汇总 1F1B、queue 异步流水线及实测吞吐改进案例。
4. Megatron-LM 相关并行实践资料。用于理解 TP、PP、DP 在工程框架中的落地方式与通信路径。
5. GPipe / PipeDream 系列论文。用于理解 pipeline bubble、micro-batch 和 weight stashing 的基础机制。

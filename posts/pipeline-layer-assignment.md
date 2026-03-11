## 核心结论

PP（Pipeline Parallelism，流水线并行）的“层分配”不是把模型层数平均拆到多张卡上，而是把**最慢 stage 的总耗时压到最低**。训练吞吐在 steady state 下近似由最慢 stage 决定，因此“每段 8 层”这种规则只在各层、各附加算子都近似同构时才成立；一旦首段包含 embedding、末段包含 LM Head 与 loss，均分层数通常就会失效。

对 Transformer 来说，中间 block 的计算量往往接近，但首尾 stage 常带有结构性额外负担：

- 首段常负责 token embedding、position embedding、输入 cast 或部分预处理。
- 末段常负责最终 norm、LM Head、cross entropy loss，若词表较大，这部分开销会很显著。
- 某些实现还会把额外通信、激活搬运或特殊 kernel 放在首尾，使两端进一步变重。

因此，很多训练系统里的经验规则都会写成“首尾少分一层”或“首尾少分若干层”。Megatron Bridge 提供针对 embedding 和 loss 的 pipeline split 选项，背后反映的是同一个事实：**首尾 stage 天生更重，层数上必须做补偿，否则整条流水线会被两端拖慢**。

进一步说，如果用 profiling 拿到每层或每个可切分单元的前后向耗时 $t_i$，那么层分配就可以写成一个“最小化最大段负载”的优化问题。此时分层不再只是经验调参，而是一个可以计算、可以复现、可以回放验证的工程优化流程。实际结果往往呈现近似 V 型分布：首尾更少，中间更多。它本身就能改善 stage imbalance；再配合 V schedule、Zero Bubble 一类更擅长消化等待时间的调度方式，吞吐提升常见在 5% 到 10%，在 TP/PP 组合更合适、通信可重叠空间更大的场景下还会更高。

---

## 问题定义与边界

先把讨论对象限定清楚。这里讲的是**训练场景下**的 PP 层分配策略，重点是“同一轮迭代中，不同 pipeline stage 该拿哪些层”。不展开讨论以下问题：

- 不单独分析算子融合、kernel 选择、FlashAttention 等单算子优化。
- 不单独分析激活重计算、checkpoint 策略对显存和时间的联动影响。
- 不讨论数据加载、数据增强、I/O 抖动。
- 不讨论纯推理场景，因为推理的目标函数和训练不同，瓶颈位置也不同。

本文只关心一个目标：**让 steady-state 下最慢的 stage 尽量变快**。

先统一术语。

| 术语 | 含义 | 这里的作用 |
|---|---|---|
| `stage` | 一段由某张卡或某组设备负责执行的流水线分段 | 是实际运行中的负载单位 |
| `physical stage` | 物理上的 PP 分段，对应真实设备分工 | 决定模型如何跨设备切开 |
| `virtual stage` | 在 VPP（Virtual Pipeline Parallelism）下，把一个 physical stage 再细分后的调度单元 | 用于更细粒度调度和负载均衡 |
| `bubble` | 因依赖、等待、排空排满造成的空闲时间 | 直接决定流水线利用率 |
| `steady state` | 流水线填满后开始稳定前后向交替执行的阶段 | 吞吐评估的核心阶段 |

如果模型共有 $L$ 个可切分单元，记每个单元的时间为 $t_1,t_2,\dots,t_L$。这里的“单元”可以是：

- 一层 Transformer block；
- embedding、LM Head、loss 这种额外算子；
- 或者更细的 profile bucket，只要系统允许这样切分。

若目标是把前 $l$ 个单元切成 $m$ 段，并让最慢段尽可能短，可写成经典的线性划分动态规划：

$$
F(l,m)=\min_{j<l}\max\left(F(j,m-1), \sum_{i=j+1}^{l} t_i\right)
$$

它表达的含义很直接：

1. 枚举最后一段从哪里开始，即枚举切点 $j$。
2. 最后一段的负载是 $\sum_{i=j+1}^{l} t_i$。
3. 前面 $m-1$ 段的最优最慢负载是 $F(j,m-1)$。
4. 这次切法的整体瓶颈，是这两者中更大的那个。
5. 在所有切点里选最小值。

这一定义有两个重要边界：

- 它优化的是**最大段和**，不是总和。总和在层固定时通常不变，真正影响吞吐的是最大值。
- 它首先解决的是**静态分配问题**，不是调度问题。也就是说，它回答“谁拿哪些层”，不直接回答“这些层如何在时间线上交错运行”。

### 为什么“均分层数”会失败

看一个最小例子。假设 8 个可切分单元，PP=4，每个普通层耗时近似为 1。再假设：

- 首段有 embedding，额外代价记为 1；
- 末段有 LM Head，额外代价记为 1。

若机械地按“每段两层”切分：

| 分配方式 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | 最慢 stage |
|---|---:|---:|---:|---:|---:|
| 均匀分层 | 2 + Embedding(1) = 3 | 2 | 2 | 2 + LM Head(1) = 3 | 3 |
| 首尾少一层 | 1 + Embedding(1) = 2 | 2 | 3 | 1 + LM Head(1) = 2 | 3 |
| 近似 V 型 | 1 + Embedding(1) = 2 | 2 | 2 | 2 + LM Head(1) = 3 | 3 |

这个表故意保留了一个容易误解的地方：三种切法的“最慢 stage”看起来都可能是 3，因此很多新手会觉得“那分层有什么意义”。问题在于，**流水线墙钟时间不是只看一个静态表格**。真实 step time 还受 warmup、steady state、flush 的长度影响；首尾如果更慢，它们在填充和排空阶段会反复拉长关键路径。也就是说，同样是“最慢值=3”，不同位置的慢、不同段数分布、不同调度方式，最终 step time 可能不一样。

再看一个更贴近公式的例子。设 8 个单元、4 个 virtual stage，耗时为：

$$
[1,1,1,1,1,1,1,2]
$$

若硬切成四段，每段两个单元，则负载为：

$$
[2,2,2,3]
$$

最后一段是瓶颈。更合理的切法会尝试把那个“2”尽量单独放，避免它再和太多普通层绑定。最优解不一定让所有段完全相等，但会让**全局最大值尽可能小**。

整个工程流程可以抽象成下面四步：

| 步骤 | 输入 | 输出 | 目的 |
|---|---|---|---|
| 1 | profiler 记录每层或每个算子时间 | $t_i$ 数组 | 把“哪一段更重”量化 |
| 2 | $t_i$、PP、VPP 配置 | 最优切分边界 | 最小化最慢段 |
| 3 | 切分边界、系统约束 | 实际 stage 布局 | 映射到训练配置 |
| 4 | 训练日志、吞吐、step time | 新一轮验证结果 | 判断优化是否真正生效 |

### 本文默认的优化对象

本文后续默认优化对象是：

$$
M = PP \times V
$$

个 virtual stage 的最大负载。其中：

- $PP$ 是 physical pipeline stage 数；
- $V$ 是每个 physical stage 内的 virtual stage 数；
- 若不开 VPP，则 $V=1$。

因此，优化目标更准确地写成：

$$
\min \max_{1 \le k \le M} \text{load}_k
$$

这里的 $\text{load}_k$ 是第 $k$ 个 virtual stage 的总耗时。这个形式之所以重要，是因为它把“经验规则”转换成了一个明确目标：**不是平均层数，而是最小化最大段负载**。

---

## 核心机制与推导

### 1. 为什么层分配会直接影响吞吐

PP 不是简单地把总计算切成几份再相加。它的运行时间由三部分共同构成：

$$
T_{\text{step}} \approx T_{\text{warmup}} + T_{\text{steady}} + T_{\text{flush}}
$$

其中 steady state 的吞吐近似受最慢 stage 控制，而 warmup 与 flush 会把首尾的额外负担放大。因此，首尾偏重的问题不是局部问题，而是整条时间线的问题。

如果把第 $k$ 个 stage 的时间记为 $s_k$，在一个简化模型里，steady-state 的单个节拍近似为：

$$
T_{\text{tick}} \approx \max_k s_k
$$

于是总 step time 会近似随该最大值上升。这个式子虽然粗糙，但足够说明为什么工程里总盯着“最慢 stage”看：**因为它就是上界里最敏感的量**。

### 2. 动态规划在这里到底做了什么

再看一次 DP 公式：

$$
F(l,m)=\min_{j<l}\max\left(F(j,m-1), \sum_{i=j+1}^{l} t_i\right)
$$

它本质上是在做“线性数组分段”，区别只在于这里的数组元素不是一般数值，而是 layer time。递推时要回答的问题是：

- 第 $m$ 段拿哪些层？
- 这些层一旦拿定，是否让该段太重？
- 如果该段不重，前面 $m-1$ 段会不会更重？
- 哪个切点能让全局最慢段最小？

可以把它理解成一条非常具体的工程规则：

> 切分的目标不是让每一段“看起来差不多”，而是让“最坏那一段尽量没那么坏”。

### 3. 为什么真实结果常呈 V 型

先看没有额外算子的理想情况。若每个 block 都完全同构，则最优分布通常接近均匀。但 Transformer 训练里经常出现下面这些额外开销：

| 位置 | 常见额外开销 | 结果 |
|---|---|---|
| 首段 | token embedding、position embedding、输入规整 | 首段偏重 |
| 中间 | 普通 Transformer block | 近似同构 |
| 末段 | final norm、LM Head、loss、大词表映射 | 末段偏重 |

这会把负载曲线从“平的”变成“两头高、中间低”。如果仍按层数均分，首尾必然偏慢。最自然的补偿方式就是：

- 首段少给一点层；
- 尾段少给一点层；
- 中间段多拿一些普通层。

从段数上看，结果就常呈近似 V 型，例如：

$$
[3,4,4,5,5,4,4,3]
$$

它不是一个必须精确命中的数学模式，而是负载补偿后的常见形状。

### 4. 调度为什么会和分层联动

只调层分配还不够，因为 pipeline 里还有 bubble。1F1B（one-forward-one-backward）是最常见的稳态调度，但它默认 stage 之间大致均衡。如果首尾更慢，或者 TP 通信很重，那么即便分层已经较好，bubble 仍可能暴露。

一些文献把传统调度中的 bubble 近似写成：

$$
(p-1)(T_F + T_{AR} + T_B + T_W)
$$

而更积极的 V 型调度或 Zero Bubble 思路会尝试把其中部分等待隐藏到其他计算后面，使有效 bubble 更接近：

$$
(p-1)(T_F + T_{AR} + T_B - T_W)
$$

这里：

- $p$ 是 pipeline stage 数；
- $T_F$ 是前向时间；
- $T_B$ 是后向时间；
- $T_W$ 是权重梯度相关计算时间；
- $T_{AR}$ 是通信时间，例如 AllReduce。

这组式子不要求死记。需要记住的只有一点：**更细粒度的调度可以把原本纯等待的时间段，与别的计算重叠起来**。于是，分层与调度就形成联动：

- 分层决定每段有多重；
- 调度决定这些段的等待能不能被吞掉。

### 5. 一个更直观的时间线理解

下面用简化表格看差异：

| 调度 | 时间线特征 | 对不均衡 stage 的容忍度 | 常见效果 |
|---|---|---|---|
| 1F1B-I | 前后向块较粗，等待暴露较多 | 低到中 | 实现简单，稳妥 |
| ZB-V | 主动压缩 bubble | 中到高 | 更适合等待显著的场景 |
| V schedule | 更细地交错 TP/PP 计算与通信 | 中到高 | 常能进一步提吞吐 |

一个玩具时序例子。假设 TP=4、PP=4，每层近似满足：

$$
T_F=T_B=4,\quad T_W=1,\quad T_{AR}=1
$$

传统 1F1B 下，某些时刻会出现“前向做完了，反向还没轮到，通信也不能完全被掩盖”的片段。这些片段就是 bubble 的来源。改成 V schedule 后，前后向块被拆得更细，一部分原本裸露的等待可以被其他分块覆盖，所以即使总计算量没变，墙钟时间也会下降。

### 6. 新手最容易混淆的三个点

| 容易混淆的问题 | 正确理解 |
|---|---|
| “层数一样是不是就公平？” | 不是。公平的是时间，不是层数。 |
| “单层最慢是不是就一定是瓶颈？” | 不一定。瓶颈看 stage 总和。 |
| “DP 求出最优切分是不是就结束了？” | 不是。还要和实际调度、通信、micro-batch 配合验证。 |

真实工程里，像 Optimus 一类工作会把问题推广到更多 virtual stage，并用真实 profiling 数据做切分，再配合 bubble exploitation 的调度器。在这种框架下，优化对象已经不是“平均分层”这种静态直觉，而是“**用可观测的时间数据，去最小化端到端训练时间**”。

---

## 代码实现

工程实现可以分成三步：

1. 采集每层或每个可切分单元的前向、后向平均耗时，形成 `times`
2. 用 DP 求给定 virtual stages 数下的最优切分
3. 把切分结果映射回训练配置，并结合 embedding / LM Head 的额外补偿做验证

下面给出一个可以直接运行的 Python 脚本。它解决的是“线性划分最小化最大段和”问题，正好对应 PP 层分配的核心抽象。代码包含：

- 前缀和；
- 动态规划；
- 回溯切分边界；
- 每段负载统计；
- 一个最小示例和输出说明。

```python
from math import inf
from typing import List, Sequence, Tuple


Bounds = List[Tuple[int, int]]


def balance_layers(times: Sequence[float], virtual_stages: int) -> Tuple[float, Bounds]:
    """
    把一维负载数组切成 virtual_stages 段，最小化最大段和。

    参数:
        times: 每层或每个可切分单元的耗时，要求为正数。
        virtual_stages: 目标分段数，必须满足 1 <= virtual_stages <= len(times)

    返回:
        best_cost: 最优切分下的最大段负载
        bounds: 每段边界列表，元素为 (l, r)，表示该段覆盖 times[l:r]
    """
    n = len(times)
    if n == 0:
        raise ValueError("times must not be empty")
    if not (1 <= virtual_stages <= n):
        raise ValueError("virtual_stages must satisfy 1 <= virtual_stages <= len(times)")
    if any(t <= 0 for t in times):
        raise ValueError("all times must be positive")

    # prefix[i] = sum(times[:i])
    prefix = [0.0] * (n + 1)
    for i, t in enumerate(times, start=1):
        prefix[i] = prefix[i - 1] + float(t)

    # dp[m][l]: 前 l 个元素切成 m 段时，最优的“最大段和”
    dp = [[inf] * (n + 1) for _ in range(virtual_stages + 1)]
    cut = [[-1] * (n + 1) for _ in range(virtual_stages + 1)]

    dp[0][0] = 0.0

    for m in range(1, virtual_stages + 1):
        # 至少要有 m 个元素，才能切成 m 段，每段至少一个元素
        for l in range(m, n + 1):
            best_cost = inf
            best_j = -1

            # 最后一段为 [j, l)，前面 j 个元素切成 m-1 段
            for j in range(m - 1, l):
                last_seg = prefix[l] - prefix[j]
                candidate = max(dp[m - 1][j], last_seg)
                if candidate < best_cost:
                    best_cost = candidate
                    best_j = j

            dp[m][l] = best_cost
            cut[m][l] = best_j

    bounds: Bounds = []
    l = n
    m = virtual_stages
    while m > 0:
        j = cut[m][l]
        bounds.append((j, l))
        l = j
        m -= 1

    bounds.reverse()
    return dp[virtual_stages][n], bounds


def segment_sums(times: Sequence[float], bounds: Bounds) -> List[float]:
    return [sum(times[l:r]) for l, r in bounds]


def pretty_print_solution(times: Sequence[float], bounds: Bounds) -> None:
    loads = segment_sums(times, bounds)
    print("times =", list(times))
    for idx, ((l, r), load) in enumerate(zip(bounds, loads)):
        print(f"stage {idx}: layers[{l}:{r}] -> {list(times[l:r])}, load={load}")
    print("max stage load =", max(loads))
    print("total load     =", sum(loads))


if __name__ == "__main__":
    # 玩具例子：最后一个单元更重
    times = [1, 1, 1, 1, 1, 1, 1, 2]
    best, bounds = balance_layers(times, virtual_stages=4)
    loads = segment_sums(times, bounds)

    assert abs(best - max(loads)) < 1e-9
    assert abs(sum(loads) - sum(times)) < 1e-9
    assert len(bounds) == 4

    print("best max stage time:", best)
    print("bounds:", bounds)
    print("loads:", loads)
    print()
    pretty_print_solution(times, bounds)
```

这段代码可以直接运行。对输入：

```python
times = [1, 1, 1, 1, 1, 1, 1, 2]
virtual_stages = 4
```

一种常见输出会是：

```text
best max stage time: 3.0
bounds: [(0, 2), (2, 4), (4, 6), (6, 8)]
loads: [2, 2, 2, 3]

times = [1, 1, 1, 1, 1, 1, 1, 2]
stage 0: layers[0:2] -> [1, 1], load=2
stage 1: layers[2:4] -> [1, 1], load=2
stage 2: layers[4:6] -> [1, 1], load=2
stage 3: layers[6:8] -> [1, 2], load=3
max stage load = 3
total load     = 9
```

### 代码解决了什么，没解决什么

这段脚本已经解决了最核心的问题：给定一组测得的负载，它能找出**理论上最优的静态切分边界**。但它没有直接处理以下工程细节：

| 未直接建模的问题 | 影响 | 常见处理方式 |
|---|---|---|
| Embedding / LM Head 是单独算子 | 首尾额外偏重 | 把它们显式加入 `times` |
| 通信时间不均匀 | 某些 stage 实际更慢 | 把通信时间并入 layer bucket |
| 某些系统不允许任意切分 | 结果无法直接落地 | 在候选边界上加约束 |
| 调度策略不同 | 同一切法收益不同 | 固定 schedule 后重新测 |

### 更接近真实训练的建模方式

真实训练中，通常不会只给 `32` 个 block 各一个时间，而会把首尾额外算子单独记出来。比如 32 层模型，PP=4，VPP=2，总共 8 个 virtual stage。一次 profiling 后可能拿到下面这样的近似时间：

| 单元 | 时间(ms) |
|---|---:|
| Embedding | 5.0 |
| Block 1 | 8.0 |
| Block 2 | 8.1 |
| ... | ... |
| Block 32 | 8.0 |
| LM Head + Loss | 6.2 |

此时更合理的建模方式是把它展开成：

$$
[\text{Emb}, \text{B1}, \text{B2}, \dots, \text{B32}, \text{Head}]
$$

再做切分，而不是只拿 32 个 block 做 DP。这样得到的结果更接近真实负载，因为首尾开销被显式纳入了优化目标。

### 一个近似真实的示例

假设：

- 32 层 Transformer；
- PP=4，VPP=2，因此共有 8 个 virtual stage；
- 普通 block 约 8 ms；
- embedding 约 5 ms；
- LM Head + loss 约 6 ms。

若机械均分，可能得到每个 virtual stage 4 个 block，但首尾实际负载会变成：

- 首段：`5 + 4*8 = 37 ms`
- 中段：`4*8 = 32 ms`
- 末段：`4*8 + 6 = 38 ms`

显然首尾偏慢。更合理的切法往往接近：

$$
[3,4,4,5,5,4,4,3]
$$

如果把 embedding 和 LM Head 视为额外固定项，则近似负载可能变成：

| Virtual Stage | Block 数 | 额外项 | 估算总时间(ms) |
|---|---:|---:|---:|
| 0 | 3 | Embedding=5 | 29 |
| 1 | 4 | 0 | 32 |
| 2 | 4 | 0 | 32 |
| 3 | 5 | 0 | 40 |
| 4 | 5 | 0 | 40 |
| 5 | 4 | 0 | 32 |
| 6 | 4 | 0 | 32 |
| 7 | 3 | LM Head+Loss=6 | 30 |

这个表又会引出一个常见疑问：为什么中间反而更重？答案是，**V 型不是固定形状，而是补偿结果**。如果 embedding/head 很重，首尾就应更少；如果首尾额外项中等，而中间通信、同步或某些 block 更慢，中间也可能成为瓶颈。最终仍然是看 profile 数据，而不是追求某个形状本身。

### 在 Megatron 体系里的实际落地方式

实际流程通常是：

1. 打开 profiling，记录每层平均前向、后向和必要通信时间。
2. 用脚本计算候选边界。
3. 若系统支持，对 embedding / loss 使用单独 split 或补偿选项。
4. 固定 micro-batch、global batch、TP/PP/VPP 后重新测 step time。
5. 比较吞吐、MFU、各 stage idle time，而不是只比较单层时间。

也就是说，分层脚本给出的是**候选切法**，最终判断标准仍然是训练日志里的端到端性能。

---

## 工程权衡与常见坑

### 1. 把“层数均分”误当成“负载均衡”

这是最常见的错误。层数均分只是一个很粗的近似，成立前提是：

- 每层结构几乎完全一致；
- 没有重型 embedding；
- 没有大词表 LM Head；
- 没有明显的通信不对称；
- 没有额外首尾算子。

这些前提在真实 LLM 训练中经常不成立。特别是词表大、序列长、激活大时，首尾偏重很常见。

下面是一份更典型的 profiler 观察表：

| 层/算子 | 平均耗时(ms) | 备注 | 建议归属 |
|---|---:|---|---|
| Embedding | 5.2 | 首段独有 | 作为首段补偿项 |
| Block 1 | 8.0 | 普通层 | 按 DP 分 |
| Block 2 | 7.9 | 普通层 | 按 DP 分 |
| Block 3 | 8.1 | 普通层 | 按 DP 分 |
| ... | ... | ... | ... |
| Block 32 | 8.1 | 普通层 | 按 DP 分 |
| LM Head + Loss | 6.0 | 末段独有 | 作为末段补偿项 |

### 2. 只看单层时间，不看 stage 总和

训练被拖慢的不是“最慢的单层”，而是“总耗时最长的 stage”。下面这种日志对排查最有用：

```text
unit=embedding      time_ms=5.2   assigned_stage=0
unit=block_01       time_ms=8.0   assigned_stage=0
unit=block_02       time_ms=7.9   assigned_stage=1
unit=block_03       time_ms=8.1   assigned_stage=1
unit=block_30       time_ms=8.0   assigned_stage=6
unit=block_31       time_ms=8.1   assigned_stage=7
unit=block_32       time_ms=8.0   assigned_stage=7
unit=lm_head_loss   time_ms=6.0   assigned_stage=7
```

正确的分析方式是把它按 stage 聚合，例如：

| Stage | 组成 | 总时间(ms) |
|---|---|---:|
| 0 | embedding + block_01 | 13.2 |
| 1 | block_02 + block_03 | 16.0 |
| ... | ... | ... |
| 7 | block_31 + block_32 + lm_head_loss | 22.1 |

真正的瓶颈是“总时间 22.1 ms 的那个 stage”，不是 `lm_head_loss=6.0 ms` 这个局部数字。

### 3. VPP 开得过大

VPP 可以减少 bubble，但不会免费。它带来的代价包括：

- 更高的切换频率；
- 更频繁的激活传递；
- 更高的调度复杂度；
- 可能更差的 kernel 粒度。

如果在 8 GPU、PP=4 的环境里，把 VPP 一路拉到很大，而 micro-batch 又不足，结果常常不是更快，而是更慢。原因很简单：每个分块太小，算不满，通信频次反而更高。

可以把这个权衡写成一句更准确的话：

> VPP 提高的是调度灵活性，不是无条件提升吞吐；当分块过细时，通信和切换成本会反过来吃掉收益。

### 4. 只看 FLOP，不看真实时间

FLOP 只描述理论计算量，不等于真实运行时间。真实训练里的时间还受到以下因素影响：

| 因素 | 例子 | 后果 |
|---|---|---|
| kernel 启动开销 | 小矩阵或小 batch | 低 FLOP 但不一定快 |
| memory 带宽 | embedding 查表、缓存 miss | 同 FLOP 不同耗时 |
| 通信阻塞 | TP AllReduce、P2P 传输 | 理论均衡但实际失衡 |
| 实现细节 | fused kernel、layout 差异 | 不同框架时间差异大 |

因此，初始分配可以用 FLOP 粗估，最终定版必须看 profiler。

### 5. 忽略 warmup 和 flush

很多新手只看 steady-state 的平均值，却忽略 warmup 和 flush。问题在于：

- warmup 时前几个 stage 更早开始承担负担；
- flush 时末几个 stage 会更晚结束；
- 若首尾本来就慢，这两个阶段会放大它们的影响。

因此，即使 steady-state 的差距看起来不大，整轮 step time 仍可能被两端拖慢。

### 6. 术语堆砌，但没有实际判断顺序

实际排查时，建议按下面顺序判断：

| 先看什么 | 为什么 |
|---|---|
| 每个 stage 总时间 | 先判断是否真的不均衡 |
| 各 stage idle time | 判断是“算得慢”还是“等得久” |
| TP/PP 通信时间 | 判断瓶颈是通信还是计算 |
| VPP 变化前后吞吐 | 判断细粒度调度是否有效 |
| step time 而非单层时间 | 确认优化是否真的落到端到端 |

如果不按这个顺序，很容易出现“调了很多参数，但不知道到底是哪一项起作用”的情况。

---

## 替代方案与适用边界

层分配不是唯一优化手段。更准确地说：

- 层分配解决“每段多重”；
- 调度策略解决“这些段如何交错执行”；
- 通信优化解决“等待能否被隐藏”。

三者经常要一起判断。

| 方案 | 核心思路 | 适用信号 | 局限 |
|---|---|---|---|
| 1F1B-I | 标准稳态前后向交替 | stage 已较均衡，先求稳 | bubble 相对明显 |
| ZB-V | 更激进地削减空泡 | 通信开销大、等待显著 | 调度更复杂 |
| V schedule | 更细地重叠 TP/PP 计算与通信 | TP 与 PP 都不小 | 需要更细粒度切分 |
| DP 重分层 | 直接重做 stage 负载分配 | 首尾明显更慢 | 本身不解决调度问题 |

### 怎么判断先动哪一个

下面这个表更适合工程判断：

| 观察到的现象 | 更可能的根因 | 优先动作 |
|---|---|---|
| stage 时间差很小，但 idle time 大 | bubble/调度问题 | 先看 ZB-V 或 V schedule |
| stage 0 和最后一段明显更慢 | 首尾结构性额外开销 | 先做首尾减层或 DP |
| TP 通信时间暴涨 | TP 侧通信主导 | 优先换调度或减轻通信 |
| VPP 增大后吞吐下降 | 分块过细，通信反噬 | 回退 VPP，重新测 |
| 只在某个大词表任务上末段变慢 | LM Head/loss 偏重 | 单独补偿末段 |

### 两个典型场景

场景一：`16 GPUs, TP=8, PP=2`

- PP 只有 2 段，pipeline 深度不大；
- TP 通信往往更重；
- 此时继续增加 VPP 未必有效；
- 通常先看调度或 TP 通信重叠。

场景二：`16 GPUs, TP=2, PP=4`

- PP 更深；
- 首尾负载不均更容易暴露；
- 若 profiling 显示两端更慢，通常先做 DP 重分层更直接。

### 适用边界

层分配优化也有边界，不是任何场景都值得做复杂 DP。

| 场景 | 是否值得精细分层 | 原因 |
|---|---|---|
| 模型很小、PP 很浅 | 一般不值得 | 收益可能低于调参成本 |
| 模型很大、PP 较深 | 值得 | 最慢 stage 会明显拖慢吞吐 |
| 首尾额外算子明显 | 值得 | 结构性不均衡已存在 |
| profiling 数据不稳定 | 暂缓 | 输入噪声大，DP 结果不可靠 |
| 系统切分限制很强 | 视情况而定 | 理论最优可能无法落地 |

可以把选择原则压缩成一句话：

> 先判断瓶颈属于“负载不均”还是“等待暴露”，再决定优先改分层还是改调度；如果两者同时存在，就把分层和调度联动起来看。

---

## 参考资料

下列资料适合分两类阅读：一类看工程接口和系统实现，另一类看调度模型与优化思路。

| 资料 | 类型 | 适合解决的问题 |
|---|---|---|
| NVIDIA Megatron Bridge Performance Guide | 工程文档 | 如何在 Megatron 体系里配置和调优 PP/VPP |
| Optimus: Accelerating Large-Scale Multimodal LLM Training by Bubble Exploitation | 论文 | 如何把 profiling、虚拟阶段切分和 bubble 利用结合起来 |
| Synergistic Tensor and Pipeline Parallelism | 论文 | 如何理解 TP/PP 联动及调度收益 |
| Pipeline and Tensor Parallelism Strategies for Training LLMs on Limited VRAM | 综述/经验文档 | 如何从资源受限角度理解 TP/PP 组合 |

建议的阅读顺序：

1. 先读 Megatron Bridge 的性能指南，理解实际系统里有哪些可调接口。
2. 再读关于 V schedule、bubble exploitation 的论文，看“为什么调度会改变等待结构”。
3. 最后回到自己的 profiler 日志，对照论文里的术语重新解释本地瓶颈。

参考链接：

- NVIDIA Megatron Bridge Performance Guide: https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html
- Optimus: Accelerating Large-Scale Multimodal LLM Training by Bubble Exploitation: https://www.usenix.org/system/files/atc25-feng.pdf
- Synergistic Tensor and Pipeline Parallelism: https://openreview.net/pdf?id=eIojV2epgX
- Pipeline and Tensor Parallelism Strategies for Training LLMs on Limited VRAM: https://www.researchgate.net/publication/398601065_Pipeline_and_Tensor_Parallelism_Strategies_for_Training_LLMs_on_Limited_VRAM

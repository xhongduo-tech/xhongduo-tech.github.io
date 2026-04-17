## 核心结论

Switch Transformer 的核心改动只有一条：把传统 MoE（Mixture of Experts，专家混合，指很多前馈网络里只激活少数几个）的 Top-2 路由改成 Top-1 路由。也就是每个 token 只送到一个 expert，而不是同时送到两个 expert。

这件事的直接结果是三点：

1. 计算更省。原来一个 token 要经过两个 expert，现在只经过一个，expert 侧前向与反向计算大约减半。
2. 通信更简单。token 不再复制两份跨设备发送，all-to-all 通信压力明显下降。
3. 训练更稳。每个 token 只有一条专家路径，梯度归属更清晰，负载控制也更容易做成静态张量。

可以把 router（路由器，指决定 token 去哪个 expert 的小网络）理解为“给题目找最擅长老师”的分发器。Top-2 是一题发给两位老师同时答，Top-1 是只选一位最合适的老师。后一种方案看起来更激进，但在大规模预训练里反而更有效，因为它减少了重复计算和重复传输。

| 维度 | Top-2 路由 | Top-1 路由 |
|---|---|---|
| expert 计算 | 每个 token 走 2 个 expert | 每个 token 走 1 个 expert |
| 通信开销 | token 复制两份，通信更重 | token 只发一次，通信更轻 |
| 梯度路径 | 两条路径混合，归因更复杂 | 单一路径，梯度更清晰 |
| 样本效率 | 单个 expert 看到的有效样本更分散 | 单个 expert 的样本更集中 |
| 工程复杂度 | dispatch/combine 更复杂 | 更容易实现和调试 |

---

## 问题定义与边界

先定义几个基础术语。

| 符号/术语 | 含义 |
|---|---|
| $x$ | 一个 token 的隐藏向量 |
| $W_r$ | router 的参数矩阵 |
| $E$ | expert 总数 |
| $n$ | 当前批次 token 总数 |
| $C$ | capacity factor，容量系数，表示给每个 expert 预留多少缓冲 |
| capacity | 单个 expert 在当前批次最多接收多少 token |
| overflow | 超过容量后无法进入 expert 的 token |
| drop/skip | overflow token 不进入 expert，而是走残差路径 |

这里的边界很重要。Switch Transformer 不是“让所有 token 一定进 expert”，而是“在固定容量内尽量分配，超出的 token 直接跳过 expert，但保留残差”。这样做的原因不是数学更优，而是工程上必须给每个 expert 固定张量大小，才能高效并行。

玩具例子可以直接看出差别。

原始 Top-2 MoE 流程：

```text
token A -> router -> expert 3, expert 7
token A 被复制 2 份
发送 2 次，接收 2 次，最后再合并
```

Switch Top-1 流程：

```text
token A -> router -> expert 3
token A 只发送 1 次
发送 1 次，接收 1 次，无需双路合并
```

如果一个 batch 有 512 个 token，那么 Top-2 最多会产生 1024 次 expert 分发记录，Top-1 只有 512 次。这里减少的不只是算力，还包括设备间同步、排序、scatter/gather（按索引分发/收集张量）等一整套数据搬运。

下面用简化流程图对比：

```text
原始 MoE (Top-2)
x -> router打分 -> 选2个expert -> 复制token -> 分发到2处 -> expert计算 -> 加权合并 -> residual

Switch Transformer (Top-1)
x -> router打分 -> 选1个expert -> 分发到1处 -> capacity检查
   -> 若未满: expert计算 -> residual相加
   -> 若已满: 跳过expert，直接走residual
```

因此，Switch 的问题定义不是“如何找到最优两个 expert”，而是“如何在单专家路由下，用容量限制、跳过机制和负载均衡保证训练不崩”。

---

## 核心机制与推导

Switch 的路由公式很直接。对每个 token 表示 $x$，router 先计算各个 expert 的概率：

$$
p(x)=\mathrm{softmax}(W_r x)
$$

然后选概率最大的 expert：

$$
e^\*=\arg\max_i p_i(x)
$$

这一步就是 Top-1 的核心。一个 token 最多只激活一个 expert。

接下来是容量约束。若当前 batch 有 $n$ 个 token，专家数为 $E$，则每个 expert 的理论平均负载是 $\frac{n}{E}$。但实际分布不可能完全均匀，所以需要容量缓冲：

$$
\mathrm{Capacity}=\left\lfloor \frac{n}{E}\times C \right\rfloor
$$

其中 $C$ 是 capacity factor，常见取值在 $1.0 \sim 1.5$。它的意义可以理解成“给热门 expert 预留多少额外座位”。

最小数值例子：

- batch token 数 $n=512$
- expert 数 $E=64$
- 理论平均负载 $512/64=8$

如果取 $C=1.25$，则：

$$
\mathrm{Capacity}=\lfloor 8\times1.25\rfloor=10
$$

这表示每个 expert 最多接收 10 个 token。若某个 expert 被分到 12 个 token，那么前 10 个进入 expert，后 2 个 overflow，直接跳过专家计算，只保留残差路径输出。

这里的“残差保底”很关键。白话说，就是即使没抢到 expert 的处理名额，这个 token 也不会在网络里消失，而是走标准 Transformer 残差通道继续向后传。这样做会损失一层专家变换，但不会破坏整条计算图。

不同 $C$ 的影响可以表格化：

| $C$ | capacity（$n=512,E=64$） | overflow 风险 | 代价 |
|---|---:|---|---|
| 1.0 | 8 | 高 | 几乎无空槽，最省内存 |
| 1.25 | 10 | 中 | 较平衡，常见默认区间 |
| 1.5 | 12 | 低 | 空槽更多，通信和显存浪费上升 |

为什么 Top-1 反而可能优于 Top-2？关键不在“表达能力更强”，而在“大规模训练的系统效率更高”。Top-2 的确给了每个 token 两次专家机会，但也引入了两份通信、两份缓存、两路梯度和额外合并步骤。当模型扩展到数百亿甚至万亿参数时，系统瓶颈常常先出现在通信和吞吐上，而不是单层表达能力上。Top-1 让每一步更便宜，于是可以在同等训练预算下跑更多 step、看到更多样本，最终整体效果更好。

为了防止所有 token 都挤向少数 expert，训练时通常还要加负载均衡损失。它的直觉目标不是提升主任务准确率，而是惩罚路由分布过度偏斜，避免 router collapse（路由塌缩，指大多数 token 总选同几个 expert）。

---

## 代码实现

下面给一个可运行的简化版 Python 例子。它没有用深度学习框架，只演示四件事：router 打分、Top-1 选择、capacity 检查、overflow 跳过。

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]

def matvec(W, x):
    return [sum(w * xi for w, xi in zip(row, x)) for row in W]

def switch_route(tokens, W_r, capacity_factor):
    n = len(tokens)
    E = len(W_r)
    capacity = int((n / E) * capacity_factor)

    assignments = []
    expert_load = [0] * E
    outputs = []
    dropped = 0

    for x in tokens:
        probs = softmax(matvec(W_r, x))
        expert = max(range(E), key=lambda i: probs[i])

        residual = x[:]  # 残差保底
        if expert_load[expert] < capacity:
            expert_load[expert] += 1
            # 用一个简单变换代替真正 expert FFN
            expert_out = [v * 2 for v in x]
            y = [a + b for a, b in zip(residual, expert_out)]
            assignments.append(expert)
        else:
            y = residual
            assignments.append(None)
            dropped += 1

        outputs.append(y)

    return {
        "capacity": capacity,
        "expert_load": expert_load,
        "assignments": assignments,
        "outputs": outputs,
        "dropped": dropped,
    }

tokens = [
    [3.0, 0.0],
    [2.5, 0.1],
    [2.0, 0.2],
    [0.0, 3.0],
]

# 2个expert：expert 0 偏好第一维大，expert 1 偏好第二维大
W_r = [
    [1.0, 0.0],
    [0.0, 1.0],
]

result = switch_route(tokens, W_r, capacity_factor=1.0)

assert result["capacity"] == 2
assert result["expert_load"] == [2, 1]
assert result["dropped"] == 1
assert result["assignments"][2] is None
assert result["outputs"][2] == tokens[2]
assert result["outputs"][0] == [9.0, 0.0]
```

这段代码故意保留了最关键的工程结构：

1. `softmax(matvec(W_r, x))` 计算 router 概率。
2. `argmax` 只选一个 expert。
3. `expert_load[expert] < capacity` 做容量检查。
4. overflow 时直接返回 `residual`，不做 expert 计算。

真实工程实现通常还会多出几步：

- 对 token 按 expert 排序，形成连续内存块，减少 scatter/gather 开销。
- 给每个 expert 预先分配固定大小的张量槽位，静态 shape 更适合并行硬件。
- 计算辅助负载均衡 loss，并和主任务 loss 一起反传。
- 屏蔽 padding token，避免无意义 token 占掉容量。
- 对空 slot 做最小化处理，避免 $C$ 太高导致大量无效计算。

真实工程例子是 Google 在 Switch Transformer 中展示的大规模预训练结果：在相同每步 FLOPs 约束下，Switch-Base 的训练速度显著快于同级稠密基线，且 Top-1 路由使模型更容易扩展到超大参数规模。这里真正起作用的不是某个花哨算子，而是“单 token 单 expert”的系统简化。

---

## 工程权衡与常见坑

Switch 的代价不是没有，而是从“多专家重复计算”转成了“更依赖路由质量”。

最常见的问题如下：

| 问题 | 现象 | 解决办法 |
|---|---|---|
| router collapse | 少数 expert 长期过载，其他 expert 几乎空闲 | 加负载均衡 loss，必要时对 router logits 做正则 |
| overflow/drop 过高 | 太多 token 跳过 expert，层层累积后效果下降 | 提高 $C$，优化路由分布，监控 drop ratio |
| empty slot | $C$ 太高，大量预留槽位没人用 | 降低 $C$，减少 padding 式浪费 |
| 通信抖动 | expert 分布极不均，某些设备成为热点 | expert 并行分片要和路由监控一起设计 |

可以看两个对比配置。

配置 A：

- $C=1.0$
- 没有负载均衡 loss
- 训练早期某几个 expert 很快成为热门
- overflow 比例达到 10%

结果是虽然单步成本低，但很多 token 在多层里持续跳过 expert，专家层名义上存在，实际利用率很差。

配置 B：

- $C=1.2$
- 加负载均衡 loss
- 定期监控每层 expert load 直方图
- overflow 比例降到约 2%

这个配置通常更接近可用状态。原因很简单：适度增加容量，比频繁丢 token 更划算；但再往上加到 1.5 甚至更高，又会出现大量空槽。

一个容易被初学者忽略的点是：overflow token “没有报错”不等于“没有损失”。它只是通过残差继续传播，不会让网络断掉，但如果比例高，等价于很多 token 在这层没用上 MoE。训练时应该把 overflow 当监控指标，而不是当成正常现象忽略掉。

---

## 替代方案与适用边界

Switch 的 Top-1 不是对所有场景都最优，它只是对“大规模、通信敏感”的训练条件更合适。

| 方案 | 适用边界与缺点 |
|---|---|
| Top-1 路由 | 适合 expert 数多、设备多、通信昂贵的场景；缺点是更依赖负载均衡，路由塌缩更敏感 |
| Top-2 路由 | 适合 expert 数较少、资源较充足、希望更强容错的场景；缺点是计算和通信成本更高 |
| Top-1 + expert pruning | 适合进一步压缩推理或训练成本；缺点是需要额外统计和剪枝策略 |
| 稠密 FFN | 适合模型较小、部署简单优先的场景；缺点是参数扩展成本高 |
| Top-1 + 低精度训练 | 适合追求更大规模吞吐；缺点是数值稳定性和硬件支持要额外验证 |

什么时候继续用 Top-2？一个典型场景是 expert 数不多，例如 4 到 8 个，并且训练集群通信压力不大。这时 Top-2 的“多给一次专家机会”可能更稳，调参也更宽松。

什么时候优先选 Top-1？当 expert 数很多、采用 TPU/GPU 集群做专家并行、all-to-all 已经接近瓶颈时，Top-1 的收益最明显。它不是理论上绝对最好，而是在系统预算固定时，通常能把更多算力花在“多训练一些样本”上，而不是花在“同一个 token 多跑一次 expert”上。

---

## 参考资料

1. Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. 核心论文，定义 Top-1 路由、capacity 和负载均衡机制。
2. Switch Transformer Top-1 routing 详解（mbrenndoerfer, 2025）。适合解释“为什么只选一个 expert 反而更高效”，也对大规模训练直觉有帮助。
3. Deep Paper 对 Switch Transformer 的拆解（deep-paper, 2025）。适合查容量公式、overflow 示例和静态 shape 背后的实现约束。
4. Top-1 Routing and Expert Pruning 工程分析（mtechresearch, 2025）。适合理解 router collapse、drop ratio、capacity 调参等工程问题。
5. Shazeer 等人关于 Sparsely-Gated MoE 的早期工作。用于对比传统 Top-k MoE 与 Switch 的差别，理解为什么通信会成为主要瓶颈。

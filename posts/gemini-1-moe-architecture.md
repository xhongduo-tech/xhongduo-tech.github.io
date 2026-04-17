## 核心结论

公开资料能确认的结论有两层。

第一层，Gemini 系列已经把稀疏 MoE 引入到大模型扩展路径里。MoE，Mixture of Experts，中文常译“混合专家”，意思是模型里有很多组并行的前馈网络，但每个 token 只调用其中少数几组，而不是把全部参数都跑一遍。这样做的直接收益是：总参数容量可以继续增大，但单次推理的计算量不必按总参数线性增长。

第二层，关于“Gemini 1.0 的专家数量、每层多少专家、Top-K 具体取值”这类底层超参数，Google 公开材料并没有完整披露。更完整公开的是 Gemini 1.5 Pro 技术报告，它明确说明自己建立在 Gemini 1.0 的研究与基础设施之上，并采用了 MoE Transformer。因此，讨论“Gemini 1.0 的 MoE 架构”时，准确表述应该是：可以确定 Gemini 路线采用了稀疏专家思路，但很多内部实现细节只能结合公开论文和通用 MoE 机制来推导，不能把未公开数字当成已知事实。

如果把推理过程写成公式，单个 token 在一层 MoE 中的输出通常可写为：

$$
y = g_1 E_{k_1}(x) + g_2 E_{k_2}(x)
$$

其中 $E_{k_i}$ 是被选中的专家网络，$g_i$ 是路由权重，满足 $g_1 + g_2 = 1$。这句话的工程含义很直接：总共可能有很多专家，但当前 token 只让其中两个真正工作。

下表把“密集 Transformer”和“稀疏 MoE Transformer”的核心差异压缩到最重要的维度。

| 维度 | 密集 Transformer | 稀疏 MoE Transformer |
|---|---|---|
| 参数使用方式 | 每个 token 激活全部层参数 | 每个 token 只激活少数专家 |
| 单 token FLOPs | 随总参数一起升高 | 主要随激活专家数升高 |
| 扩容手段 | 直接把层做宽、做深 | 增加专家总数，但保持 Top-K 很小 |
| 训练难点 | 主要是稳定性与显存 | 还要解决路由、负载均衡、分布式调度 |
| 适合场景 | 结构简单、部署稳定 | 追求更大容量且希望控制计算成本 |

玩具例子可以这样理解：假设一层里有 8 个专家，普通密集模型相当于 8 个老师每次都要一起批改同一张卷子；MoE 则是先看题目内容，再只叫最合适的 2 个老师来批改。老师总人数没变少，但每次实际工作的老师更少，所以算力省下来了。

真实工程例子是 Gemini 1.5 Pro 处理超长上下文，例如完整 JAX 代码库或长篇小说时，模型不需要在每个位置都让全部专家参与，而是让与当前 token 更相关的少数专家前向计算。这就是“参数很多，但激活很少”的核心收益。

---

## 问题定义与边界

这里要解决的问题不是“怎么把模型做得更大”，而是“怎么在计算预算有限时，把有效容量做得更大”。容量可以粗略理解为模型能容纳多少种模式和知识类型。对多模态和长上下文任务，单纯把 dense Transformer 一路加宽加深，会同时推高训练 FLOPs、推理延迟和显存占用，成本增长过快。

MoE 给出的答案叫条件计算。条件计算的白话解释是：不是所有参数都必须参与每次计算，而是根据当前输入内容，有条件地启用一部分参数。这样同一层可以拥有大量专家，但每个 token 只走少数几条路径。

这类架构的边界也必须说清楚。

第一，公开边界。Gemini 公开资料证明了该路线采用 MoE，但没有完整公开 1.0 的每层专家数、容量因子、具体 dispatcher 设计，因此本文能讲清“机制”，不能把未公开数字写成“官方已确认”。

第二，任务边界。MoE 的价值在长上下文、多任务、多模态场景最明显，因为这类场景需要更高模型容量，但又不希望每个 token 都付出 dense 模型的完整代价。像 746k token 级别的上下文，或者跨代码、图像、文本的统一建模，都是 MoE 更容易体现优势的地方。

第三，工程边界。MoE 并不是“白送参数”。它把问题从“算力不够”转成了“路由和调度很难”。专家选错、负载不均、某些专家过载、跨设备通信过重，都会直接吞掉理论收益。

下表给出密集模型和 MoE 模型在问题边界上的对比。

| 对比项 | 密集模型 | MoE 模型 |
|---|---|---|
| 总参数增长 | 与每次计算成本强耦合 | 可解耦，总参数增大但激活成本受 Top-K 控制 |
| 长上下文扩展 | 成本高，层层都全量计算 | 更适合配合条件计算做容量扩展 |
| 多模态知识分化 | 共享参数容易互相干扰 | 不同专家可偏向不同模式 |
| 分布式部署 | 相对直接 | 需要 token 到专家的调度网络 |
| 实现复杂度 | 低 | 高 |

玩具例子可以把它看成一场百人讲座。密集模型像始终只有一位总讲师，从头讲到尾；MoE 像有很多专业讲师，每遇到一个主题片段，只安排最合适的两位上台。这样总师资更强，但每一段内容并不是所有人都上场。

真实工程例子是处理大型代码库问答。用户提问某个 JAX 函数行为时，理想模型不应把所有知识子空间都同等激活，而应让与 API 语义、代码模式、依赖关系更相关的专家优先工作。MoE 不是减少输入长度，而是在固定输入长度上减少“每步真正运行的参数量”。

---

## 核心机制与推导

MoE 的核心组件只有两个：专家和路由器。专家通常就是若干个并行的 FFN，FFN 是前馈网络，可以理解成 Transformer 层里负责非线性变换的那一块。路由器则负责决定“当前 token 该送给谁”。

设当前 token 的隐藏状态为 $x \in \mathbb{R}^d$，专家总数为 $E$。路由器先做一次线性映射：

$$
h = W_r x
$$

这里 $h \in \mathbb{R}^E$，每个分量 $h_i$ 都对应第 $i$ 个专家的打分，也叫 routing logits。logit 可以理解为“还没归一化的偏好分数”。

如果采用 Top-2 路由，就从全部 $E$ 个专家里选出分数最高的两个，记作 $k_1, k_2$。然后只对这两个分数做 softmax 归一化：

$$
g_i = \frac{\exp(h_{k_i})}{\exp(h_{k_1}) + \exp(h_{k_2})}
$$

最终输出为：

$$
y = g_1 E_{k_1}(x) + g_2 E_{k_2}(x)
$$

这三个式子已经概括了 MoE 最重要的推导链路：先打分，再截断选择，再加权混合。

下面给一个最小数值例子。假设有 $E=8$ 个专家，Top-K 里 $K=2$。某个 token 的路由分数如下：

| 专家编号 | logit |
|---|---:|
| 0 | 2.1 |
| 1 | -0.3 |
| 2 | 1.0 |
| 3 | -1.2 |
| 4 | 0.2 |
| 5 | -0.4 |
| 6 | 0.5 |
| 7 | -0.9 |

此时 Top-2 选中专家 0 和 2。两者的归一化权重大致为：

$$
g_1 = \frac{e^{2.1}}{e^{2.1}+e^{1.0}} \approx 0.75,\quad
g_2 = \frac{e^{1.0}}{e^{2.1}+e^{1.0}} \approx 0.25
$$

这意味着当前 token 只激活了 2 个专家，占总专家数的 $2/8 = 25\%$。如果每个专家的 FFN 规模相同，那么这一层的主要计算也只落在这 25% 的专家路径上。

但只会“选专家”还不够。MoE 最大的训练风险叫 Expert Collapse，中文可叫“专家塌陷”，意思是路由器总把大多数 token 发给少数几个专家，其他专家长期接不到活，最后整层虽然参数很多，实际只用了少量容量。为了解决这个问题，常见做法是加入负载均衡损失。

一种公开讨论较多的形式是 Importance 的变异系数惩罚：

$$
\mathcal{L}_{\text{importance}} = \mathrm{CV}(\mathrm{Importance})^2
$$

其中 $\mathrm{Importance}$ 是每个专家接收到的加权流量统计，$\mathrm{CV}$ 是变异系数，定义为标准差除以均值。白话解释是：如果每个专家收到的 token 量差不多，CV 就小；如果某些专家极忙、某些专家闲置，CV 就大。训练时把这个损失加到总损失里，就是在惩罚“使用分布太偏”。

可以把这个平衡过程理解成排班系统。路由器不仅要选“最合适”的老师，还要避免总是把全部课都排给同一个老师。否则那位老师会爆满，其他老师被浪费，系统吞吐也会下降。

真实工程里，这个问题会进一步扩展到分布式环境。因为专家常常被分散放到不同设备上，token 必须先被 dispatch，dispatch 可以理解为“把 token 按路由结果送到对应设备或缓冲区”。如果某个专家突然收到过多 token，就会超过容量上限，产生 overflow。于是工程实现通常还要引入 capacity factor，也就是专家容量放大系数，用于决定每个专家最多接收多少 token。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，覆盖四个关键步骤：计算 logits、选择 Top-K、归一化权重、计算辅助平衡损失。代码不是完整训练框架，但足够对应上面公式。

```python
import math

def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [v / s for v in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def route_token(router_logits, k=2):
    idx = topk_indices(router_logits, k)
    weights = softmax([router_logits[i] for i in idx])
    return idx, weights

def combine_experts(expert_outputs, indices, weights):
    assert len(indices) == len(weights)
    out = [0.0 for _ in expert_outputs[0]]
    for i, w in zip(indices, weights):
        for j, v in enumerate(expert_outputs[i]):
            out[j] += w * v
    return out

def importance_cv_squared(batch_routes, num_experts):
    importance = [0.0] * num_experts
    for indices, weights in batch_routes:
        for i, w in zip(indices, weights):
            importance[i] += w

    mean = sum(importance) / num_experts
    assert mean > 0.0
    variance = sum((x - mean) ** 2 for x in importance) / num_experts
    std = math.sqrt(variance)
    cv = std / mean
    return cv ** 2, importance

# 玩具例子：8 个专家，Top-2 路由
router_logits = [2.1, -0.3, 1.0, -1.2, 0.2, -0.4, 0.5, -0.9]
indices, weights = route_token(router_logits, k=2)

assert indices == [0, 2]
assert abs(sum(weights) - 1.0) < 1e-9
assert 0.74 < weights[0] < 0.76
assert 0.24 < weights[1] < 0.26

# 假设每个专家都输出二维向量
expert_outputs = [
    [10.0, 2.0],  # expert 0
    [0.0, 0.0],
    [2.0, 6.0],   # expert 2
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
]

y = combine_experts(expert_outputs, indices, weights)
assert len(y) == 2
assert y[0] > y[1]  # 因为 expert 0 权重大

# 一个小 batch 的平衡损失
batch_routes = [
    route_token([2.1, -0.3, 1.0, -1.2, 0.2, -0.4, 0.5, -0.9], 2),
    route_token([1.9,  0.2, 0.8, -0.5, 0.1, -0.2, 0.4, -1.0], 2),
    route_token([-0.2, 2.3, 0.7, -0.1, 0.1, -0.5, 0.3, -0.8], 2),
]
cv2, importance = importance_cv_squared(batch_routes, 8)

assert cv2 >= 0.0
assert len(importance) == 8
```

如果把这段逻辑映射回真实系统，通常会拆成三个函数：

| 模块 | 作用 | 真实工程关注点 |
|---|---|---|
| `route_tokens` | 计算 logits 并选 Top-K | 数值稳定性、并行化、梯度处理 |
| `dispatch` | 按专家编号重排 token | 跨设备通信、容量限制、overflow |
| `combine` | 聚合专家输出 | 保持 token 原顺序、残差连接 |

真实工程例子可以想成 TPU 集群上的一层 MoE。一个 batch 进来后，路由器先给每个 token 选专家；然后 dispatcher 把属于同一专家的 token 聚到一起，送到对应设备上的 FFN；专家计算完成后，再把结果按原 token 顺序拼回去。这一步如果做不好，理论上的 FLOPs 节省会被通信成本抵消。

还有两个实现细节不能忽略。

第一，capacity factor。设某层 batch 内共有 $T$ 个 token，专家数为 $E$，Top-K 为 $K$，则平均每个专家期望接收大约 $TK/E$ 个路由项。工程里通常给每个专家分配：

$$
\text{capacity} = \left\lceil \text{capacity\_factor} \times \frac{TK}{E} \right\rceil
$$

如果某专家实际收到的 token 超过这个上限，超出的部分就要被 drop 或走旁路。旁路通常指 residual bypass，也就是让这些 token 暂时跳过该专家层，直接走残差路径，避免整个 batch 卡死。

第二，Top-K 是离散选择，离散操作对梯度不友好。训练时常见做法包括 softmax 近似、straight-through estimator，或者加入带噪声的 routing 机制，让路由学习不至于过早僵死。

---

## 工程权衡与常见坑

MoE 最大的优点是“同等激活成本下给更多总参数”，最大的问题则是“这些参数不一定被均匀且有效地用起来”。所以工程权衡不是单一维度，而是三组同时存在的拉扯：容量和稳定性、理论 FLOPs 和真实吞吐、专家分工和通信成本。

最常见的三个坑如下表所示。

| 问题 | 现象 | 原因 | 常见规避措施 |
|---|---|---|---|
| Expert Collapse | 少数专家吃掉大多数 token | 路由器过度偏向局部最优 | 平衡损失、router bias、带噪声路由 |
| Token Overflow | 某专家容量爆掉，部分 token 被丢弃 | 负载不均或 capacity 太小 | 提高 `capacity_factor`、drop 策略、residual bypass |
| Top-K 梯度震荡 | 训练早期不稳定，路由频繁跳变 | 离散选择不平滑 | softmax 近似、straight-through、温度控制 |

先看 Expert Collapse。新手最容易误解的一点是：MoE 不是只要多放几个专家，模型容量就自动变大。如果路由器总把 token 发给同一两个专家，其他专家几乎不更新，那些参数等于摆设。平衡损失的作用不是“让大家绝对平均”，而是防止极端倾斜，让每个专家至少有机会学到不同模式。

再看 Token Overflow。假设某层有 4 个专家、Top-2 路由、一个 batch 中共有 1024 个 token 路由项。如果理论均匀分配下每个专家该接收 512 个路由项，但某个专家因为更受欢迎收到了 700 个，而 capacity 只给到 640，那么多出的 60 个就必须处理。处理策略通常有三种：直接 drop、降级到次优专家、或者走残差旁路。不同策略对质量和吞吐的影响不同，没有免费方案。

真实工程例子里，这个问题很典型。假设你在多机环境中部署一层 MoE，专家按设备分片。某轮输入里大量 token 都与代码语义相关，结果“代码型专家”集体过热，而“通用语义专家”负载很低。这时候即使总算力看起来充足，系统瓶颈也会集中在少数设备上，吞吐下降、尾延迟上升。

还有一个容易被忽视的权衡：Top-K 选得越小，FLOPs 越省，但路由出错的代价越大。比如从 Top-2 改成 Top-1，计算更便宜，但一旦最优专家选错，模型就没有第二个专家兜底。Top-2 往往是在效果和成本之间较常见的折中。

可以把这一点写成很简单的经验判断：

$$
\text{激活成本} \propto K \times \text{单专家成本}
$$

在专家数固定时，$K$ 越大，激活成本越高；$K$ 越小，对路由质量要求越高。MoE 的工程优化，很多时候就是围绕这个比例关系做妥协。

---

## 替代方案与适用边界

MoE 不是唯一的扩容办法，它只是“让总参数变大但不想让每次推理都变贵”这条路线上的代表方案。实际工程里至少有四类常见替代思路。

| 方案 | 参数容量 | 单次 FLOPs | 调度复杂度 | 适合场景 | 主要问题 |
|---|---|---|---|---|---|
| 密集 Transformer | 中到高 | 高 | 低 | 中短上下文、稳定部署 | 成本随参数一起涨 |
| Top-K MoE | 很高 | 中 | 高 | 超大模型、长上下文、多任务 | 路由与通信复杂 |
| 固定专家子集 | 高 | 中 | 中 | 任务边界清晰的场景 | 灵活性差，泛化较弱 |
| Mixture-of-Adapters | 中 | 低到中 | 中 | 已有大模型增量扩展 | 容量上限受底座约束 |

先说密集 Transformer。它的优点是简单、稳定、成熟，训练和推理链路都更直观。如果你的上下文长度不大、资源不算极端紧张、团队没有处理复杂并行调度的能力，dense 往往更稳。对 1k token 左右的普通问答服务，引入完整 MoE 基础设施很可能不划算。

固定专家子集是另一种折中做法。它的意思是专家划分更像人工指定，而不是对每个 token 都做动态 Top-K 路由。好处是调度简单一些，坏处是灵活性下降，模型没法像动态路由那样细粒度匹配输入。

Mixture-of-Adapters 则更适合“底座模型已经固定”的场景。adapter 可以理解为挂在主干模型旁边的小模块，通过少量新增参数适配新任务。它的优势是便宜，缺点是容量扩展幅度通常不如完整 MoE。

玩具例子可以继续沿用“老师系统”。密集模型是每节课都由同一位全能老师完整讲完；Top-K MoE 是每段内容都从老师库里动态挑两位；固定专家子集像按课程先分班，一门课固定几位老师；adapter 则像在原老师身边配若干助教。不同方案没有绝对好坏，只是针对不同成本结构。

真实工程里，MoE 更适合这几类条件同时成立的任务：

- 你需要非常大的总参数容量。
- 你希望把单 token 激活成本控制在 dense 大模型之下。
- 你有足够的并行硬件和调度工程能力。
- 任务包含长上下文、多模态或明显异质的知识分布。

相反，下面这些情况通常不值得优先上 MoE：

- 数据量不大，专家很难学出稳定分工。
- 部署环境并行能力弱，通信成本会盖过理论收益。
- 模型主要面对短上下文、单一任务。
- 团队无法接受复杂的训练与监控系统。

因此，把 Gemini 路线理解为“为了大容量与长上下文效率而走向稀疏专家”是合理的；但把 MoE 理解为“任何场景都优于 dense”则不准确。

---

## 参考资料

| 来源 | 年份 | 可验证内容 |
|---|---:|---|
| Gemini 1.5 Pro Technical Report | 2024 | 说明 Gemini 1.5 Pro 建立在 Gemini 1.0 的研究和基础设施上，采用 MoE Transformer，并给出长上下文与效率对比 |
| Top-K Routing: Expert Selection in Mixture of Experts Models | 2025 | 系统解释 Top-K 路由、路由公式、专家选择与工程实现注意事项 |
| Auxiliary Balancing Loss: Preventing Expert Collapse in MoE | 2025 | 解释平衡损失、CV 形式的直觉和避免专家塌陷的方法 |

1. Gemini Team, “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” 2024。可验证点：Gemini 1.5 Pro 采用 MoE Transformer，且建立在 Gemini 1.0 的研究与基础设施之上。
2. Michael Brenndoerfer, “Top-K Routing: Expert Selection in Mixture of Experts Models,” 2025。可验证点：$h=W_rx$、Top-K 选择、softmax 归一化与加权求和的标准写法。
3. Michael Brenndoerfer, “Auxiliary Balancing Loss: Preventing Expert Collapse in MoE,” 2025。可验证点：使用 Importance 的变异系数约束负载均衡，避免专家塌陷。

如果只保留一条判断标准，应该是这一句：Gemini 公开路线已经证明稀疏 MoE 是其扩展大模型能力的重要手段，但 Gemini 1.0 的很多超参数没有完整公开，因此讨论时应区分“已公开事实”和“基于公开 MoE 机制的合理推导”。

## 核心结论

共享专家机制可以理解为：在 MoE（Mixture of Experts，专家混合）层里，先放一组“所有 token 都必须经过”的通用前馈网络，再让路由系统按需选择少量专业专家。这里的 token 指模型处理的最小文本单元，比如一个字、词或子词。它的核心价值不是“让更多专家一起算”，而是把通用知识和专门知识拆开学。

这件事解决的是传统 Sparse MoE（稀疏专家混合）里的一个长期问题：多个路由专家往往会重复学习语法、常识、格式对齐、常见短语组合这类基础能力，导致参数被浪费。共享专家固定激活后，基础表示先被统一抽出来，路由专家就能把容量更多用于数学、代码、长上下文推理、工具调用模式、知识检索风格等更细的能力分工。

DeepSeek 系列的关键设计点在于：共享专家始终参与计算，但不会无限增加总算力预算。工程上更常见的做法是“加共享专家，就相应减少路由激活预算”，把每个 token 的总 FLOPs（浮点运算量）近似控制住。因此，共享专家不是免费午餐，它本质上是在固定预算内重新分配容量，把一部分“可选容量”改成“稳定容量”。

下面这个配置能概括 DeepSeek-V3 一层 MoE 的直觉结构：

| 组成 | 数量 | 说明 |
|---|---:|---|
| 共享专家 | 1 | 每个 token 必走，负责通用知识 |
| 路由专家 | 256 | 按 gate 选择，负责专业模式 |
| 每 token 激活专家 | 9 | 1 个共享 + 8 个 Top-K 路由 |

这个表的含义要读清楚：模型里一共不是只有 9 个专家，而是有 257 个专家；只是对任意一个 token 来说，真正参加计算的是其中 9 个。稀疏性的来源就在这里。

如果只看一句话结论：共享专家的作用是给 MoE 加一个“通用知识锚点”，减少专家间的重复学习，提高参数效率，但共享专家数量不能无约束增加，通常 1 到 3 个更实用。

---

## 问题定义与边界

先定义问题。传统 MoE 的目标是：总参数很多，但每个 token 只激活少量专家，这样既保留大模型容量，又控制单次计算成本。问题在于，路由专家虽然被设计成“分工合作”，但在训练初期和中期，它们经常会重复学到相似的基础模式，比如标点、语法骨架、常见事实表达、简单句法依赖、模板化的回答结构。

这里的“知识冗余”可以直接理解为：本来只需要学一次的基础能力，被很多专家各自又学了一遍。  
这会带来三个后果：

| 现象 | 直接后果 | 长期影响 |
|---|---|---|
| 多个专家都学语法和常见表达 | 参数重复占用 | 专家专业化程度下降 |
| 路由专家承担过多共性任务 | gate 难以形成清晰分工 | 路由信号变得模糊 |
| 通用知识分散在不同专家中 | 相同 token 模式走不同路径 | 训练效率和可解释性变差 |

共享专家就是为了解决这件事。它给所有 token 提供统一入口，把“谁都需要”的那部分表示先抽取出来。这样剩下的路由专家不需要反复打基础，更容易形成真正的专业化分工。

但这个设计有边界，不是任何场景都适合：

| 问题 | 边界/影响 |
|---|---|
| 知识冗余 | 没有共享专家时，多个路由专家会重复学习通用语义，参数利用率下降 |
| 激活预算 | 每个 token 可激活的专家数有限，共享专家增加后通常必须压缩别的部分 |
| 路由价值 | 如果任务本身高度单一，通用知识占比不高，共享专家收益可能有限 |
| 通信成本 | 多机多卡训练下，专家越多，分发与聚合越复杂 |
| 负载均衡 | 即使有共享专家，路由专家仍可能出现热点和闲置 |

一个玩具例子能说明这个边界。

假设有 4 个路由专家，分别处理“数学、代码、百科、对话”。如果没有共享专家，那么“基本语法”和“常见词法模式”这四个专家都得学一遍。现在加入 1 个共享专家，专门负责这些基础模式，那么数学专家就更可能把参数花在公式变形上，代码专家更可能花在 API 调用结构上，百科专家更可能花在事实组织上，对话专家更可能花在语气和多轮上下文衔接上。

但如果你把共享专家加到 4 个，同时还保持原来 4 个路由专家都激活，那么总计算就显著增加，稀疏化带来的效率优势会被吃掉。MoE 的前提是“总参数大，但每次只算少数参数”；如果固定激活部分膨胀过快，MoE 就会逐步退化成“很多参数真的都要算”。

所以问题定义不是“共享专家好不好”，而是：

> 在固定 token 计算预算下，共享专家是否能比同等数量的路由专家更有效地承载通用知识。

这个提法比“共享专家能不能提升效果”更准确，因为它把讨论放回了模型设计最核心的约束：预算不变时，容量该怎么分。

---

## 核心机制与推导

MoE 层可以看成两条并行通道。

第一条是共享通道：所有 token 都进入共享专家。  
第二条是路由通道：token 经过 gate（门控网络，负责给专家打分的模块），只选出 Top-K 个路由专家参与计算。

设输入 token 表示为 $u_t \in \mathbb{R}^d$，共享专家数量为 $N_s$，路由专家数量为 $N_r$。第 $i$ 个共享专家记作 $\mathrm{FFN}^{(s)}_i$，第 $i$ 个路由专家记作 $\mathrm{FFN}^{(r)}_i$。则一层输出可以写成：

$$
h'_t = u_t + \sum_{i=1}^{N_s} \mathrm{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t}\,\mathrm{FFN}^{(r)}_i(u_t)
$$

这条式子分三部分看：

| 项 | 含义 | 是否每个 token 都执行 |
|---|---|---|
| $u_t$ | 残差项，保留原始输入 | 是 |
| $\sum_{i=1}^{N_s}\mathrm{FFN}^{(s)}_i(u_t)$ | 共享专家输出，抽取通用表示 | 是 |
| $\sum_{i=1}^{N_r} g_{i,t}\mathrm{FFN}^{(r)}_i(u_t)$ | 路由专家输出，处理专业模式 | 否，只对被选中的专家有效 |

接下来定义 gate。设第 $i$ 个路由专家的路由向量为 $e_i$，那么 token 与专家的亲和分数可以写成：

$$
a_{i,t} = u_t^\top e_i
$$

再通过 Softmax 归一化为分配概率：

$$
s_{i,t} = \frac{\exp(a_{i,t})}{\sum_{j=1}^{N_r}\exp(a_{j,t})}
$$

然后只保留 Top-K 个得分最高的路由专家：

$$
g_{i,t} =
\begin{cases}
s_{i,t}, & i \in \mathrm{TopK}(\{s_{j,t}\}_{j=1}^{N_r}, K_r) \\
0, & \text{otherwise}
\end{cases}
$$

这里要注意一个容易混淆的点：

1. `Softmax` 的作用是把“打分”变成“可比较的权重”。
2. `Top-K` 的作用是把大部分专家直接裁掉，不参与这次计算。
3. 共享专家不受 `Top-K` 约束，它们不参加这个筛选过程。

因此，路由专家的本质是“候选集合里选少数”，共享专家的本质是“固定保底通道”。

如果进一步写得更贴近工程实现，很多系统会在打分时引入附加项，例如容量约束、负载均衡偏置或噪声项。一个更一般的写法是：

$$
a_{i,t} = u_t^\top e_i + b_i + \epsilon_{i,t}
$$

其中：

| 记号 | 含义 |
|---|---|
| $u_t^\top e_i$ | token 与专家的基础匹配度 |
| $b_i$ | 专家级偏置，可用于缓解负载倾斜 |
| $\epsilon_{i,t}$ | 训练时加入的扰动或噪声，增加探索性 |

这也是为什么论文和工程代码里的 gate 往往比“线性层 + Softmax”更复杂。公式上是一个打分器，工程上它还承担了负载控制器的角色。

这套机制的关键不是公式本身，而是容量分工：

| 部分 | 学习目标 | 是否固定激活 |
|---|---|---|
| 共享专家 | 语法、常识、基础表达结构、常见语义模板 | 是 |
| 路由专家 | 领域模式、任务特化、复杂组合能力 | 否 |
| Gate | 决定某个 token 更该找谁处理 | 否 |

从预算角度看，如果单个专家 FFN 成本近似相同，那么每个 token 的主要计算近似满足：

$$
\text{Cost}_{\text{token}} \propto N_s + K_r
$$

更细一点，如果单个专家前馈网络的计算成本记为 $C_{\text{ffn}}$，则：

$$
\text{Cost}_{\text{token}} \approx (N_s + K_r)\,C_{\text{ffn}} + C_{\text{router}}
$$

通常 $C_{\text{router}}$ 相比多个 FFN 的成本更小，所以设计上最敏感的量仍然是 $N_s + K_r$。  
这说明共享专家数量从 1 增到 4 时，若想保持总成本近似不变，就需要把 $K_r$ 相应减小。比如原来是 $N_s=1, K_r=8$，总激活数约为 9；如果改成 $N_s=3$，为了维持近似预算，$K_r$ 往往要降到 6 左右。

这就是共享专家设计里最重要的推导：

> 它不是单纯“多加一层保险”，而是在固定预算下，把一部分稀疏路由容量改成稳定的通用容量。

真实工程例子是 DeepSeek-V3。公开技术报告和官方仓库中给出的描述通常可以概括为：每层保留 1 个共享专家和 256 个路由专家，每个 token 实际激活 1 个共享专家和 8 个路由专家。这个配置反映的工程判断很明确：1 个共享专家已经足够承担大量基础表示，继续增加共享专家的边际收益，不一定高于保留下来的路由选择空间。

---

## 代码实现

下面先给一个最小可运行的 Python 示例。它不依赖深度学习框架，只用标准库模拟“共享专家固定参与、路由专家 Top-K 参与”的聚合逻辑。这里不追求真实训练，只演示数据流。

```python
import math
from typing import List, Tuple


Vector = List[float]


def softmax(xs: Vector) -> Vector:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def topk_indices(scores: Vector, k: int) -> List[int]:
    if not 1 <= k <= len(scores):
        raise ValueError(f"k must be in [1, {len(scores)}], got {k}")
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def vector_add(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vector size mismatch")
    return [x + y for x, y in zip(a, b)]


def vector_scale(a: Vector, scale: float) -> Vector:
    return [scale * x for x in a]


def shared_expert(x: Vector, bias: Vector) -> Vector:
    # 共享专家：统一做一个偏移，模拟通用表示提取
    return vector_add(x, bias)


def routed_expert(x: Vector, matrix: List[Vector]) -> Vector:
    # 路由专家：做一个最小线性变换 y = W x
    rows = len(matrix)
    cols = len(matrix[0])
    if len(x) != cols:
        raise ValueError("input size does not match expert weight")
    return [
        sum(matrix[r][c] * x[c] for c in range(cols))
        for r in range(rows)
    ]


def moe_layer(
    token: Vector,
    shared_biases: List[Vector],
    routed_weights: List[List[Vector]],
    router_logits: Vector,
    k: int,
) -> Tuple[Vector, Vector, List[int]]:
    if not shared_biases:
        raise ValueError("at least one shared expert is required")
    if len(routed_weights) != len(router_logits):
        raise ValueError("router logits must match routed expert count")

    dim = len(token)

    shared_out = [0.0] * dim
    for bias in shared_biases:
        out = shared_expert(token, bias)
        shared_out = vector_add(shared_out, out)

    probs = softmax(router_logits)
    selected = topk_indices(probs, k)

    routed_out = [0.0] * dim
    for i in selected:
        out = routed_expert(token, routed_weights[i])
        routed_out = vector_add(routed_out, vector_scale(out, probs[i]))

    output = vector_add(token, vector_add(shared_out, routed_out))
    return output, probs, selected


def main() -> None:
    token = [1.0, 2.0]

    # 1 个共享专家
    shared_biases = [
        [0.1, -0.2],
    ]

    # 4 个路由专家，每个专家是一个 2x2 线性层
    routed_weights = [
        [[1.0, 0.0], [0.0, 1.0]],    # 偏保留原表示
        [[0.5, 0.5], [0.2, 1.2]],    # 偏平滑组合
        [[2.0, 0.0], [0.0, 1.5]],    # 偏放大某些维度
        [[-1.0, 0.0], [0.0, 0.5]],   # 偏反向/抑制
    ]

    router_logits = [0.2, 1.1, 2.0, -0.5]

    output, probs, selected = moe_layer(
        token=token,
        shared_biases=shared_biases,
        routed_weights=routed_weights,
        router_logits=router_logits,
        k=2,
    )

    assert len(output) == 2
    assert len(probs) == 4
    assert abs(sum(probs) - 1.0) < 1e-9
    assert len(selected) == 2
    assert 2 in selected  # 第 3 个专家得分最高，必须进入 Top-K

    print("router probs:", [round(p, 4) for p in probs])
    print("selected experts:", selected)
    print("output:", [round(v, 4) for v in output])


if __name__ == "__main__":
    main()
```

这段代码可以直接运行。你会看到三个关键结果：

1. `router probs` 给出 4 个路由专家的归一化概率。
2. `selected experts` 只保留 Top-2。
3. `output` 是残差、共享专家输出、路由专家输出三者相加后的结果。

这个例子里的流程就是典型的 `shared -> router -> aggregate`：

1. 共享专家先对 token 做通用变换。
2. Router 对所有路由专家打分。
3. 只保留 Top-K。
4. 用 gate 分数加权各个路由专家输出。
5. 最后和残差一起相加。

如果写成更接近工程代码的伪代码，核心结构如下：

```python
shared_out = sum(shared_ffn(token) for shared_ffn in shared_experts)

scores = router(token)                    # 所有路由专家打分
probs = softmax(scores)
topk_idx = select_topk(probs, k=8)

routed_out = 0
for i in topk_idx:
    routed_out += probs[i] * routed_experts[i](token)

output = token + shared_out + routed_out
```

真实工程里会比这个复杂很多，至少还要处理下面这些问题：

| 实现点 | 工程含义 | 为什么不能省 |
|---|---|---|
| token dispatch | 把不同 token 分发到不同专家执行 | 否则专家并行无法落地 |
| expert parallel | 不同 GPU 或节点分别承载专家 | 单卡放不下大规模专家集合 |
| capacity control | 限制单个专家在一个 batch 中接收过多 token | 防止热点专家溢出 |
| load balance | 避免少数专家被选中过多 | 防止大量专家闲置 |
| combine/scatter | 把专家输出按原 token 顺序聚合回来 | 否则后续层无法对齐 |
| padding / packing | 对变长 token 批次做整理 | 提高吞吐，减少空算 |

对初级工程师来说，最重要的是先记住一条：共享专家不经过 Top-K 选择，它是无条件执行的；Top-K 只控制路由专家。  
只要这条没混淆，MoE 里大部分实现细节都能顺着数据流看懂。

---

## 工程权衡与常见坑

共享专家机制的收益和代价都很直接，所以工程上最怕“概念上知道有用，预算上没算清楚”。

第一类坑是共享专家过多。  
共享专家每增加一个，所有 token 都会额外多走一个 FFN。这是固定成本，不像路由专家那样只在被选中时才花钱。如果共享专家从 1 增到 4，而 Top-K 还维持 8，那么总激活从 9 变成 12，单 token 计算会明显上升。对超大模型来说，这不是小改动，而是会直接反映在训练吞吐、推理延迟和显存占用上。

第二类坑是误把共享专家当成“万能专家”。  
共享专家适合承载通用表示，不适合承担大量强任务分化能力。因为它对所有 token 都生效，目标函数会把它推向“对多数样本都有帮助”的方向，这天然更像共性抽取器，而不是高度专业化模块。如果让共享专家承担太多“特化任务”，它反而会学得不够尖锐。

第三类坑是负载均衡被忽略。  
即使有共享专家，路由系统仍可能偏爱少数路由专家，导致热点专家过载、其余专家闲置。负载均衡的意思是尽量让专家使用更均匀，否则总参数虽然大，真正发挥作用的参数却少。

第四类坑是把“效果更稳”误解成“并行更简单”。  
共享专家确实可能让训练 loss 更平滑，因为通用模式有稳定承载路径；但 token dispatch、all-to-all 通信、专家容量限制、跨机回收这些工程问题并不会自动消失。

下面是常见权衡：

| 风险 | 描述 | 缓解 |
|---|---|---|
| 共享专家过多 | 每 token 固定 FLOPs 上升，压缩路由预算 | 共享数优先控制在 1 到 3 |
| 路由专家退化 | 通用信息被共享专家吸走后，路由专家学不到稳定分工 | 调整 gate 温度、专家宽度和训练配比 |
| 负载倾斜 | 少数专家被频繁选中，出现热点 | 使用负载均衡策略、偏置调节或容量限制 |
| 通信放大 | 多机环境下 token 分发与回收更复杂 | 限制跨节点路由范围，优化专家部署 |
| 容量浪费 | 专家很多但长期不被选中 | 监控 expert hit rate，做路由诊断 |

一个更贴近真实训练的例子是：假设你有 64 张 GPU，训练一个大规模中文通用模型。加入共享专家后，loss 可能更稳，因为通用模式有固定承载路径；但如果路由偏斜没有控制住，就会出现某些专家所在 GPU 长时间更忙，导致整体吞吐下降。也就是说，共享专家解决的是“学什么”的问题，不自动解决“怎么高效并行”的问题。

关于“共享专家数量 1 到 4 的影响”，可以概括成下面这张经验表：

| 共享专家数 | 潜在收益 | 主要代价 | 常见判断 |
|---|---|---|---|
| 1 | 最低固定成本，先建立通用锚点 | 通用容量有限 | 最常见、最稳妥 |
| 2 | 更强的通用表示承载 | 需要压缩部分 Top-K | 可尝试 |
| 3 | 进一步减少通用冗余 | 路由空间开始被明显挤占 | 需严格算预算 |
| 4 | 通用容量更大 | 固定 FLOPs 增长明显 | 除非有明确证据，否则偏激进 |

还可以再补一条经验判断：

| 观察到的现象 | 更可能的问题 | 更常见的动作 |
|---|---|---|
| 多个路由专家学到很像的模式 | 通用知识没有集中承载 | 增加 1 个共享专家或增强共享容量 |
| 专家选择过于集中 | 路由塌缩或负载失衡 | 调整 gate、平衡策略或容量约束 |
| 共享专家输出占比过高 | 共享专家过强，路由空间被压缩 | 降低共享规模或提高路由预算 |
| 训练很稳但专业化不明显 | 共性抽取得多，差异化不足 | 检查 expert specialization 指标 |

工程上真正要追的不是“共享专家越多越好”，而是一个更朴素的目标：  
在不抬高单 token 成本的前提下，让共享专家只承载共性，让路由专家尽可能承载差异。

---

## 替代方案与适用边界

共享专家不是唯一选择，它只是 DeepSeek 一类架构给出的答案。

第一类替代方案是纯 Sparse MoE。  
也就是没有共享专家，所有专家都走路由。它的优点是结构更纯粹、固定成本更低；缺点是通用知识更容易在多个专家间重复学习。若任务本身很专一，比如高比例代码补全、固定格式信息抽取，这种重复未必严重，纯 Sparse MoE 仍然可能更划算。

第二类替代方案是层级路由，例如先粗分再细分的层级 MoE。  
这类方法更强调“先决定大类，再决定小类”，目标是让专家选择更稳定、更可控。它解决的是“怎么把 token 更有秩序地送到专家”，而不是“是否给所有 token 一个固定通道”。

第三类替代方案是回到稠密 FFN。  
也就是不用 MoE，所有 token 都经过同一个大前馈层。它最简单，训练最稳定，调参空间也更小；但扩展到超大参数规模时，单位计算带来的容量增长通常不如稀疏结构划算。

第四类替代方案是不加共享专家，但用更强的路由和负载控制。  
例如更严格的 capacity 约束、更细粒度专家划分、专家选择偏置、专家级统计反馈。这一类方案承认“冗余存在”，但选择从路由器而不是结构上处理。

对比可以简化为：

| 方案 | 适用边界 | 额外成本 |
|---|---|---|
| 共享专家 + 路由专家 | 通用知识很多，希望降低专家冗余 | 需要重新平衡共享数与 Top-K |
| 纯 Sparse MoE | 任务更专、希望保留最大稀疏性 | 需额外处理通用知识重复 |
| 层级路由 MoE | 专家选择不稳定、易 collapse | gate 结构更复杂 |
| 稠密 FFN | 规模较小或强调稳定性 | 推大模型时计算成本高 |
| 强路由控制但无共享专家 | 不想增加固定成本，又想改善分工 | 路由器设计与调参复杂 |

因此，共享专家的适用边界很明确：

1. 通用知识占比较高的通用大模型，收益通常更明显。
2. 模型很大、专家很多时，减少冗余更有价值。
3. 如果任务高度专门化，共享专家未必值得。
4. 如果系统通信已经很紧张，再加复杂专家结构要谨慎评估吞吐。

也可以换一种更直观的判断方式：

| 场景 | 是否适合共享专家 | 原因 |
|---|---|---|
| 通用聊天模型 | 较适合 | 通用表达、常识、对齐模式占比高 |
| 多领域混合模型 | 较适合 | 共性多，特化也多，适合做通用/专业拆分 |
| 高比例代码模型 | 视数据而定 | 如果模式高度集中，共享收益未必大 |
| 垂直信息抽取模型 | 往往一般 | 输入格式固定，共性和特化边界不明显 |
| 强实时推理场景 | 需谨慎 | 固定激活部分会直接影响延迟 |

一句话概括适用边界：  
共享专家适合“共性很重、又需要很多专业分工”的模型；不适合“共性很少、算力预算又极紧”的模型。

---

## 参考资料

下面列的一手资料足够支撑本文的核心结论。涉及 DeepSeek-V3 的“1 个共享专家 + 256 个路由专家 + 每 token 激活 8 个路由专家”配置，主要来自官方仓库与技术报告；共享专家用于承载 common knowledge、缓解 routed experts 冗余，主要来自 DeepSeekMoE 论文；MoE 的负载均衡、路由稳定性和稀疏计算背景，主要参考 GShard 和 Switch Transformer。

| 来源 | 内容聚焦 |
|---|---|
| [DeepSeekMoE 论文（arXiv:2401.06066）](https://arxiv.org/abs/2401.06066) | 提出 shared experts isolation，直接讨论 common knowledge 与 routed experts 冗余 |
| [DeepSeek-MoE 官方仓库](https://github.com/deepseek-ai/DeepSeek-MoE) | 模型说明、论文链接、DeepSeekMoE 的公开实现背景 |
| [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) | DeepSeek-V3 的 MoE 配置、激活参数规模、工程侧设计说明 |
| [DeepSeek-V3 Technical Report（arXiv:2412.19437）](https://arxiv.org/abs/2412.19437) | V3 中 DeepSeekMoE 的落地、auxiliary-loss-free load balancing、系统优化 |
| [GShard（arXiv:2006.16668）](https://arxiv.org/abs/2006.16668) | 大规模 MoE 的经典路由框架、Top-K 专家选择与分布式训练背景 |
| [Switch Transformer（JMLR 2022）](https://jmlr.org/papers/v23/21-0998.html) | 稀疏 MoE 的训练稳定性、负载均衡、计算效率问题的经典参考 |
| [Hugging Face 关于 MoE 的综述文章](https://huggingface.co/blog/moe) | 对新手更友好的 MoE 直觉解释，适合补充背景概念 |

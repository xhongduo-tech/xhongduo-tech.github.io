## 核心结论

条件计算的核心不是“把模型做小”，而是“让同一个模型对不同输入走不同计算路径”。这里的“条件”就是输入触发的决策规则。更准确地说，模型会先根据当前样本、当前 token 或当前上下文估计“这一步值不值得继续投入算力”，再决定激活多少层、多少专家、多少注意力通路。

从机制上看，条件计算通常沿三个轴展开：动态深度、动态宽度、动态注意力。

- 动态深度：按输入决定跑多少层，等价于“这一题要不要提前结束”。
- 动态宽度：按输入决定激活多少通道、多少专家或多少子模块，等价于“同一层里不是所有零件都要同时开机”。
- 动态注意力：按上下文决定哪些 token、哪些头、哪些模态值得被关注，等价于“把注意力预算花在更相关的位置”。

这类方法真正优化的是“平均每个样本的计算量”，而不是“最坏情况下的计算量”。因此，易样本可以走轻路径，难样本仍走全路径。以 ACL 2025 的 D3 为例，论文报告在不重训模型的前提下，对 Llama 系列做 token 级动态深度控制，在 GSM8K 和 BBH 上实现平均约 `1.5x` 的推理加速，同时性能下降控制在 `1%` 以内；分模型设置下，速度提升区间约为 `1.07x` 到 `1.94x`。这说明动态计算不是简单的“拿精度换速度”，而是在尽量不碰关键信息流的前提下，削掉冗余计算。

还可以把它写成一个更工程化的目标：

$$
\min \ \mathbb{E}_{x\sim \mathcal{D}}[\mathrm{Cost}(x)]
\quad
\text{s.t.}
\quad
\mathbb{E}_{x\sim \mathcal{D}}[\Delta \mathrm{Quality}(x)] \le \varepsilon
$$

其中，$\mathrm{Cost}(x)$ 是样本 $x$ 的推理开销，$\Delta \mathrm{Quality}(x)$ 是相对全量推理的质量损失，$\varepsilon$ 是你能接受的退化上界。条件计算的工作，本质上就是在这个约束优化问题里设计一个靠谱的门控策略。

| 轴 | 典型机制 | 激活条件 | 主要节省什么 | 主要风险 |
| --- | --- | --- | --- | --- |
| 动态深度 | 早退、跳层、D3 | 置信度、位置、门控分数 | Decoder block FLOPs | 状态不完整、KV cache 对齐 |
| 动态宽度 | 通道裁剪、MoE top-k | 门控网络输出 | FFN/专家计算 | 专家失衡、路由抖动 |
| 动态注意力 | 稀疏注意力、条件路由注意力 | 上下文相关 gate/mask | 注意力矩阵与部分头计算 | 梯度难训、硬路由不稳定 |

一个新手容易忽略的点是：这三种方法并不是竞争关系。实际系统里经常是“先保守地做动态深度，再叠加宽度路由，最后在注意力侧做细粒度预算分配”。因为深度最容易解释，宽度次之，注意力路由的实现和训练往往最难。

---

## 问题定义与边界

问题可以定义得很直接：在保证输出质量基本不变的前提下，是否能让模型对简单输入少算一点、对复杂输入多算一点。传统静态网络的问题在于，不管输入是一张清晰的猫图还是一张模糊的夜间图，网络都走同样的全路径。这意味着大量样本实际上“被过度计算”了。

这里要先划清边界。条件计算并不总是安全，尤其在自回归生成任务里。自回归生成的意思是：当前 token 的预测依赖前面所有 token 的状态。如果前几个 token 只跑了浅层，后面的 token 却需要深层状态，那系统必须补齐缺失状态，否则会出现“同一句历史，在不同层深上不一致”的问题。

这个问题在分类任务里通常不严重，因为分类是“一次性输出”。模型只要在某一层已经把类别分开，就可以在那里结束。而在生成任务里，历史 token 不是只服务一次，它们还要持续参与后续 token 的注意力计算，所以“前面少算的账，后面可能还要补”。

一个适合新手理解的玩具例子是图片分类。假设前 3 层已经把“清晰猫图”和“清晰狗图”分开了，那么清晰猫图可以在第 3 层直接输出；而模糊图像还需要继续走到第 8 层甚至第 12 层。这类任务的输出是一次性分类，早退通常比较自然。

再看语言生成。假设第 5 个 token 只走了 20 层，第 6 个 token 却需要 28 层，那么第 6 个 token 在第 21 到 28 层会访问前面 token 的 key/value 状态。如果第 5 个 token 在这些层没有真实状态，系统就要回答三个问题：

1. 缺失状态是否允许直接复制。
2. 复制发生在什么时候，是立刻补，还是按需补。
3. KV cache 缺失时，是复制旧值、重算一遍，还是禁止访问。

D3 论文把这类问题统一到 handling missing state。翻成工程语言，就是“跳过的层以后要不要补账，以及怎么补”。

为了避免概念混淆，可以把几个常见术语先对齐：

| 术语 | 严格含义 | 新手可理解说法 |
| --- | --- | --- |
| Early Exit | 在中间层直接输出，不再继续后续层 | 提前交卷 |
| Layer Skipping | 当前 token 跳过部分层，但最终仍从正常输出头生成 | 中途绕开几层 |
| Missing State | 某个 token 在某些层没有真实隐藏状态或 KV | 历史账本有空页 |
| KV Cache | 解码阶段缓存历史 token 的 key/value | 注意力的历史中间结果 |
| Gate / Router | 决定是否激活某条路径的函数 | 路口红绿灯 |

不同任务对条件计算的容忍度差异很大。下面这张表可以作为快速判断：

| 输入类型 | 推荐机制 | 原因 | 不适合场景 |
| --- | --- | --- | --- |
| 短文本分类 | 早退、动态宽度 | 输出稳定，浅层已够区分 | 极端长尾类别 |
| 图像分类 | 早退、跳层 | 易样本冗余大 | 细粒度识别 |
| 长文本生成 | 位置感知深度衰减、软门控注意力 | 可做 token 级预算分配 | 粗暴硬早退 |
| 多模态生成 | 条件路由注意力 + 局部 MoE | 不同模态相关性不均衡 | 无同步机制的硬路由 |
| 在线边缘设备 | 早退、轻量门控 | 时延收益直接 | 门控本身过重 |

判断边界时还有一个实用标准：如果错误一旦发生会沿时间传播，条件计算就要更保守；如果错误只影响当前样本的一次性输出，条件计算可以更激进。

---

## 核心机制与推导

动态深度最典型的一个形式，是按 token 位置衰减计算量。D3 使用的位置感知深度衰减可以写成：

$$
L_{\text{use}}(i)=\left\lfloor L\cdot \alpha^i \right\rfloor,\quad 0<\alpha<1
$$

其中，$L$ 是总层数，$i$ 是当前生成位置，$\alpha$ 是衰减系数，$L_{\text{use}}(i)$ 是第 $i$ 个 token 实际使用的层数。直观解释是：越靠后的 token，通常拥有越多上下文，模型对它的困惑度可能更低，因此不必总跑满全部层。

这个公式有几个直接性质。

1. 当 $i$ 增大时，$\alpha^i$ 单调变小，所以使用层数单调下降。
2. 当 $\alpha$ 越接近 1，衰减越慢，策略越保守。
3. 当 $\alpha$ 越小，衰减越快，算力节省更多，但质量风险也更高。
4. 若实现中加入最小层数约束 `L_min`，则真实使用层数更接近：
   $$
   L_{\text{real}}(i)=\max(L_{\min}, \lfloor L\alpha^i \rfloor)
   $$

如果总层数是 $L=48$，$\alpha=0.94$，第 10 个 token 的理论保留层数约为：

$$
48\times 0.94^{10}\approx 25.85
$$

若实现里取 `floor`，就是 25 层；若取 `ceil`，则是 26 层。这个细节不能省，因为它会直接影响线上预算。论文公式、伪代码和真实实现如果在取整方式上不一致，最终速度和质量都可能对不上。

进一步看平均计算量。若一段输出长度为 $T$，并假设每层成本近似相同，则平均深度比例近似为：

$$
r_{\text{depth}}
=
\frac{1}{TL}
\sum_{i=0}^{T-1} L_{\text{use}}(i)
$$

那么总计算量可粗略近似为：

$$
\mathrm{Cost}_{\text{dynamic}}
\approx
r_{\text{depth}}\cdot \mathrm{Cost}_{\text{full}}
$$

这是一个很有用的工程估算式。它不能精确预测 wall-clock latency，但足够帮助你在实验前判断“这个策略大概能省多少 FLOPs”。

再看动态宽度。以 MoE 为例，假设某层有 $E$ 个专家，每次只激活 top-$k$ 个专家，则这一层的专家计算比例可近似写成：

$$
r_{\text{width}} \approx \frac{k}{E}
$$

例如 $E=16, k=2$，理论上只激活 `12.5%` 的专家路径。但这只是算子级开销，不等于端到端时延节省，因为还要考虑路由、通信、padding 和负载均衡。

动态注意力的核心则是：不是先把所有注意力都算完，再事后裁掉，而是在算之前先决定“哪些通路值得被计算”。常见门控形式可以写成：

$$
b_t=\sigma\left(w_b^\top \mathrm{LReLU}(W_b z_t+b_b)\right)
$$

这里：

- $z_t$ 是当前 token 的上下文特征；
- $W_b,b_b,w_b$ 是门控网络参数；
- $\mathrm{LReLU}$ 是 LeakyReLU；
- $\sigma$ 是 sigmoid，把输出压到 $[0,1]$；
- $b_t$ 可以理解为“打开更昂贵注意力路径的概率分数”。

若 $b_t > \tau$，就走完整注意力；否则只走局部或稀疏注意力。把它写成决策规则就是：

$$
\text{route}(t)=
\begin{cases}
\text{full attention}, & b_t>\tau \\
\text{sparse/local attention}, & b_t\le \tau
\end{cases}
$$

因此，三轴回答的是三个不同问题：

- 深度回答“要跑多少层”。
- 宽度回答“每层开多少部件”。
- 注意力回答“信息该流向哪里”。

如果要给新手一个统一心智模型，可以把模型看成一栋 48 层办公楼。

- 动态深度：决定这次访客要上到几层。
- 动态宽度：决定每层开几个办公室。
- 动态注意力：决定这些办公室里谁真的参加讨论。

条件计算的本质，就是把“所有人全程到场”改成“按需出勤”。

再补一层现实约束。D3 并不是简单地“从尾部一路砍层”，而是区分了 `core layers` 和 `flex layers`。从论文描述看，头部和尾部层更像关键层，负责基础特征建立和输出对齐；中间层冗余更大，更适合作为可跳过的 `flex layers`。这也是为什么很多动态深度方法在真实系统里更偏向“优先跳中间层”，而不是粗暴裁掉最前或最后一段。

---

## 代码实现

下面给一个最小可运行的 Python 版本，只演示“先算深度预算，再根据 gate 选择注意力路径，并显式处理缺失状态”的顺序。它不是完整 Transformer，但调度逻辑是完整可运行的。

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def layers_to_use(total_layers: int, alpha: float, position: int, min_layers: int = 1) -> int:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if position < 0:
        raise ValueError("position must be non-negative")
    if not (1 <= min_layers <= total_layers):
        raise ValueError("min_layers must be in [1, total_layers]")

    used = math.floor(total_layers * (alpha ** position))
    return max(min_layers, used)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def routed_attention_mode(feature_score: float, threshold: float = 0.5) -> str:
    gate = sigmoid(feature_score)
    return "full_attention" if gate > threshold else "local_attention_only"


@dataclass
class TokenState:
    used_layers: int
    hidden_states: Dict[int, str]
    missing_layers: List[int]
    attention_mode: str


def decode_one_token(
    total_layers: int,
    alpha: float,
    position: int,
    feature_score: float,
    min_layers: int = 1,
) -> TokenState:
    used_layers = layers_to_use(total_layers, alpha, position, min_layers=min_layers)

    hidden_states: Dict[int, str] = {}
    missing_layers: List[int] = []

    for layer_id in range(total_layers):
        if layer_id < used_layers:
            hidden_states[layer_id] = f"h(pos={position}, layer={layer_id})"
        else:
            missing_layers.append(layer_id)

    attention_mode = routed_attention_mode(feature_score)
    return TokenState(
        used_layers=used_layers,
        hidden_states=hidden_states,
        missing_layers=missing_layers,
        attention_mode=attention_mode,
    )


def fill_missing_state_by_copy(
    previous_state: TokenState,
    current_state: TokenState,
) -> List[Tuple[int, str]]:
    repaired: List[Tuple[int, str]] = []

    for layer_id in range(current_state.used_layers):
        if layer_id not in previous_state.hidden_states:
            copied_value = f"copied_from_lower_budget(layer={layer_id})"
            repaired.append((layer_id, copied_value))

    return repaired


def conditional_decode_step(
    total_layers: int,
    alpha: float,
    position: int,
    feature_score: float,
    previous_state: TokenState | None = None,
) -> Dict[str, object]:
    current_state = decode_one_token(
        total_layers=total_layers,
        alpha=alpha,
        position=position,
        feature_score=feature_score,
        min_layers=4,
    )

    repaired = []
    if previous_state is not None and previous_state.used_layers < current_state.used_layers:
        repaired = fill_missing_state_by_copy(previous_state, current_state)

    return {
        "position": position,
        "used_layers": current_state.used_layers,
        "attention_mode": current_state.attention_mode,
        "missing_layers": current_state.missing_layers,
        "repaired_from_history": repaired,
    }


if __name__ == "__main__":
    easy_t0 = decode_one_token(total_layers=48, alpha=0.94, position=10, feature_score=-1.0, min_layers=4)
    hard_t1 = conditional_decode_step(
        total_layers=48,
        alpha=0.94,
        position=2,
        feature_score=2.0,
        previous_state=easy_t0,
    )

    assert easy_t0.used_layers == 25
    assert easy_t0.attention_mode == "local_attention_only"
    assert hard_t1["attention_mode"] == "full_attention"
    assert hard_t1["used_layers"] > easy_t0.used_layers
    assert len(hard_t1["repaired_from_history"]) > 0

    print("easy token layers:", easy_t0.used_layers)
    print("hard token decision:", hard_t1)
```

这段代码有三个关键点。

1. `layers_to_use` 明确了位置衰减和最小层数约束。
2. `routed_attention_mode` 把注意力门控单独写成决策函数，便于以后替换成真实 MLP。
3. `fill_missing_state_by_copy` 显式暴露了“补账”动作，让新手能看到：动态深度不是只决定跳过哪些层，还必须定义历史状态如何被后续 token 使用。

如果把这个思路放进 Llama 类解码器，伪代码大致如下：

```python
L_use = floor(L * alpha ** position)
L_use = max(min_layers, L_use)

for layer_id in range(L):
    if layer_id in core_layers:
        x, kv_cache[layer_id] = transformer_block(x, kv_cache[layer_id])
    elif flex_layer_should_run(layer_id, L_use):
        x, kv_cache[layer_id] = transformer_block(x, kv_cache[layer_id])
    else:
        mark_missing_state(token_id=position, layer_id=layer_id)

gate = sigmoid(gate_mlp(x))
if gate > threshold:
    x = routed_attention(x, mode="full")
else:
    x = routed_attention(x, mode="sparse")
```

真实工程里通常还会补上四个约束：

| 约束 | 作用 | 省略后会怎样 |
| --- | --- | --- |
| `min_layers` | 防止后部 token 被削得过浅 | 质量突然崩 |
| `core_layers` | 保留头尾关键层 | 语义对齐和输出稳定性下降 |
| `mark_missing_state` | 记录哪些层没有真实状态 | 后续 token 无法安全读 cache |
| `budget logging` | 统计每个 token 实际预算 | 离线分析和回归排障困难 |

因此，所谓“代码可运行”和“机制可落地”不是一回事。前者只要求逻辑能跑通，后者要求你把预算、缓存、补账、监控四件事一起做完整。

---

## 工程权衡与常见坑

第一个大坑是 KV cache 缺层。KV cache 是解码时缓存历史 key/value 的结构，可以理解成“前面 token 已经算好的注意力中间结果”。如果不同 token 走的层数不同，那么某些层对某些历史 token 根本没有缓存。后续 token 一旦要访问这些层，就会遇到两难：要么重算，要么复制，要么绕过访问。三种方案都不是免费午餐。

- 复制最快，但误差可能传播。
- 重算最稳，但会吞掉原本省下的算力。
- 绕过访问实现复杂，而且容易和现有 kernel 不兼容。

D3 论文明确讨论了 missing state 和 copy operation 对生成质量的影响，尤其指出早期 token 的错误更敏感。原因不复杂：前面几步一旦错了，后面很多 token 都会反复依赖这份错误历史。

第二个大坑是“门控本身比省下来的计算还贵”。这在小模型、短序列、低 batch 场景里尤其常见。比如你想省 `20%` 的 FFN 计算，却引入了额外的 MLP、离散采样、负载均衡损失、scatter/gather 和复杂调度，最后 FLOPs 看起来下降了，但 wall-clock latency 没降，甚至更慢。

这里必须区分四个指标：

| 指标 | 含义 | 常见误判 |
| --- | --- | --- |
| FLOPs | 理论计算量 | 下降了不代表延迟也降 |
| Latency | 单次请求耗时 | 会受 kernel、通信、同步影响 |
| Throughput | 单位时间处理样本数 | 与 batch size 强相关 |
| Memory | 显存与缓存占用 | 跳层不一定减少 KV 占用 |

第三个大坑是硬路由训练不稳定。硬路由指 top-k、argmax、Bernoulli 这类离散决策，意思是“只开这几个，其余全部关闭”。问题在于离散操作不可导，梯度不能自然回传，因此通常需要 Straight-Through Estimator、Gumbel-Softmax 或策略梯度一类近似技巧。若处理不好，会出现专家塌缩，也就是所有 token 长期挤进少数专家，其他专家几乎不工作。

第四个大坑是质量退化不均匀。动态计算常常在平均指标上看起来很漂亮，但一拆任务就会发现：长链推理、长上下文、多跳依赖、代码生成、工具调用这类样本掉点更严重。因为这些任务的难点不是某个 token 的局部置信度，而是跨段落、跨步骤、跨模块的一致性。只看当前位置打分，常常低估真正难度。

第五个大坑是“批处理友好性”不足。很多动态方法单样本上能跑，但一到 batch 推理就退化。原因在于不同样本、不同 token 走不同路径后，GPU 更难保持大矩阵连续计算，分支越多，硬件越不友好。理论省算和实际吞吐之间，经常隔着一整层系统实现问题。

下面这张表可以把主要权衡放在一起看：

| 权衡点 | 好处 | 风险 | 常见缓解方法 |
| --- | --- | --- | --- |
| 更激进的深度衰减 | 更高加速比 | 长链推理掉点 | 提高 $\alpha$，保留更多 core layers |
| 更稀疏的宽度路由 | FFN/MoE 开销下降 | 专家失衡 | 加负载均衡损失、最小激活约束 |
| 更硬的注意力门控 | 稀疏度更高 | 梯度不稳定 | 用软门控预热，再切硬路由 |
| 更复杂的缺失状态补齐 | 上下文一致性更好 | 实现成本高 | 统一 copy/recompute 策略并做回归测试 |

工程上一个可执行原则是：先做“保守动态”，再做“激进动态”。

- 第一阶段：只做动态深度，且设置最小层数和核心层保护。
- 第二阶段：在 FFN 或 MoE 上加动态宽度。
- 第三阶段：最后再碰条件注意力和更细粒度的 token 路由。

原因很简单。深度调度最容易与现有 Transformer 结构兼容，而宽度与注意力路由更容易碰到 kernel、并行、训练稳定性和 batch 碎片化问题。

如果要把这段话再压缩成一句实操建议，那就是：先证明“省算能转化成真实延迟收益”，再追求更稀疏的漂亮图表。

---

## 替代方案与适用边界

如果目标是移动端分类器，最直接的方案通常不是 D3，而是 confidence-based early exit。它的逻辑简单：如果中间层分类头已经足够有把握，就直接输出。这种方法的优点是实现成本低、行为容易解释、调试路径短，尤其适合短文本分类、图像分类、检索打分等“一次性决策”任务。

如果目标是大模型生成，confidence-based early exit 往往不够稳。因为生成不是只看当前 token 对不对，还要求历史状态在后续步骤持续可用。此时更适合位置感知深度衰减，或者“浅层保底 + 条件注意力稀疏”的组合。这样做的逻辑是：先用深度轴控制总预算，再用注意力轴把有限预算花到更相关的上下文上。

如果你的模型本来就是 MoE，那么动态宽度几乎已经天然存在。每个 token 本来就只激活部分专家，此时进一步要考虑的不是“能不能稀疏”，而是“稀疏能否带来真实收益”。因为 MoE 的理论优势是单次激活少，但实际系统还要承担路由、跨设备通信和负载均衡成本。

几个常见方案的适用边界可以放在一起比较：

| 方法 | 最适合任务 | 优点 | 主要边界 |
| --- | --- | --- | --- |
| Early Exit | 分类、检索打分 | 简单、易部署 | 生成任务状态一致性差 |
| Skip Layer | 生成、深网络推理 | 不改输出头 | 缺失状态处理麻烦 |
| MoE / 动态宽度 | 大模型、专家分工明显任务 | 参数大但单次激活少 | 负载均衡与通信开销 |
| 条件路由注意力 | 长上下文、多模态 | 直接减少无关注意力 | 路由训练复杂 |
| D3 类深度衰减 | 自回归生成 | 无需重训、兼容现有模型 | 需要仔细处理 KV cache |

还可以从“是否重训”和“是否改模型结构”两个维度看：

| 方案 | 是否常需重训 | 是否改模型结构 | 部署门槛 |
| --- | --- | --- | --- |
| Early Exit | 常需要 | 通常要加中间分类头 | 低到中 |
| D3 类训练后解码策略 | 不需要 | 通常不改主体结构 | 低 |
| MoE | 需要 | 需要 | 高 |
| 条件路由注意力 | 需要 | 需要 | 高 |

因此，适用边界可以总结为一句话：若任务输出一次性、局部可判定，优先 early exit；若任务是逐 token 生成、依赖长历史，优先保守的动态深度；若模型已模块化或专家化，动态宽度与条件注意力更值得投入。

最后再强调一个容易被忽略的结论：条件计算不是“任何任务都该上”的通用加速按钮。它最适合的是“样本难度差异大、模型冗余明显、系统允许不均匀预算分配”的场景。若任务本身几乎每个样本都同样困难，或者硬件环境非常不擅长分支执行，那么静态优化、量化、蒸馏、KV cache 工程优化，往往比条件计算更直接。

---

## 参考资料

1. Fan, Siqi; Fang, Xuezhi; Xing, Xingrun; Han, Peng; Shang, Shuo; Wang, Yequan. 2025. *Position-Aware Depth Decay Decoding (D3): Boosting Large Language Model Inference Efficiency*. Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.154/
2. D3 论文 PDF. https://aclanthology.org/2025.findings-acl.154.pdf
3. Zhang, Zihan; Shi, Linze. 2023. *Dynamic inference techniques for deep neural networks*. Applied and Computational Engineering, 2:281-293. https://www.ewadirect.com/proceedings/ace/article/view/2699
4. 该文 PDF. https://www.ewadirect.com/proceedings/ace/article/view/2699/pdf
5. Schuster, Tal; Fisch, Adam; Gupta, Jai Prakash; Dehghani, Mostafa; Bahri, Dara; Tran, Vinh Q.; Tay, Yi; Metzler, Donald. 2022. *Confident Adaptive Language Modeling*. NeurIPS 2022. https://people.csail.mit.edu/tals/publication/calm/
6. Google Research 页面：*Confident Adaptive Language Modeling*. https://research.google/pubs/confident-adaptive-language-modeling/
7. Shazeer, Noam et al. 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. https://research.google/pubs/pub45929
8. Fedus, William; Zoph, Barret; Shazeer, Noam. 2022. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 23(120). https://jmlr.org/papers/v23/21-0998.html
9. Emergent Mind: *Condition-Routed Attention*. 作为机制综述可帮助快速定位相关门控公式与代表性论文。 https://www.emergentmind.com/topics/condition-routed-attention

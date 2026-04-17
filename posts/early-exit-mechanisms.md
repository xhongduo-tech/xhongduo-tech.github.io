## 核心结论

早期退出机制（early exit）是一种动态计算方法。所谓动态计算，指的不是“把模型整体做小”，而是“让模型针对不同输入，实际执行不同深度的计算”。在 Transformer 分类模型里，它的典型实现方式是：在若干中间层后面额外挂上分类头，模型每前进一层，就先做一次当前层预测；如果这个预测已经足够确定，就直接返回结果，不再继续向后计算。

它要解决的问题很具体：并不是每个样本都值得跑完整个模型。很多文本在浅层就已经包含足够明确的信号。例如，“我要退款”“物流很快，包装很好”“帮我开电子发票”这类输入，语义边界通常很清楚；相反，“价格能接受，但更新后卡顿更明显”“功能不错，不过售后回复太慢”这类输入往往同时包含正负信息，需要更深层表示才能稳定判断。早期退出利用的正是这种“样本难度不均匀”的事实。

因此，它带来的收益不是改变最坏情况，而是降低平均情况成本。对于难样本，模型仍然可以一路计算到最后一层，保留完整模型的表达能力；对于易样本，模型在前几层就结束，节省计算量和时延。这个特性决定了它更像一条可调的速度-精度曲线，而不是一个固定压缩点。

可以把完整模型看成一段楼梯。固定减层方法是把楼梯整体截短，例如从 12 层改成 6 层，那么所有样本都只能走 6 层；早期退出则保留整段楼梯，但在第 2 层、第 4 层、第 8 层设置“中途出口”，样本到了某一层如果已经足够确定，就直接离开。这也是它和 DistilBERT 一类固定减层模型的本质区别。

| 方法 | 路径长度是否固定 | 是否按样本决策 | 速度弹性 | 精度上限 |
|---|---|---:|---:|---:|
| 固定减层（如 DistilBERT） | 是 | 否 | 低 | 受限于裁剪后深度 |
| 早期退出（如 DeeBERT、FastBERT） | 否 | 是 | 高 | 保留完整模型上限 |

再补一句面向初学者最重要的判断标准：如果一个方法让所有输入都执行同样多的层数，它通常不是早期退出；如果一个方法会根据当前样本的状态决定“现在停”还是“继续算”，它才属于动态推理范畴中的早期退出。

---

## 问题定义与边界

问题可以形式化地写成下面这样。给定一个有 $L$ 层的 Transformer 编码器，在若干层甚至每一层后面增加一个额外出口。每个出口都包含一个小分类器，它读取当前层隐藏表示并给出预测分布。模型从第 1 层开始前向传播，每经过一层，就判断一次“是否已经可以退出”；一旦满足退出条件，就终止后续计算并返回当前预测。

为了方便理解，可以把每个出口记为第 $l$ 层的一个函数：
$$
p_l = g_l(h_l)
$$
其中：

- $h_l$ 表示第 $l$ 层的隐藏状态
- $g_l(\cdot)$ 表示第 $l$ 层后接的中间分类头
- $p_l$ 表示该层输出的类别概率分布

最常见的退出条件有两类。

1. 最大类别概率足够高：
$$
\max_i p_{l,i} \ge \tau
$$

这里 $\tau$ 是置信度阈值。如果某一类的概率已经高到超过阈值，就认为当前层已经“足够确定”。

2. 预测分布的熵足够低：
$$
H(p_l) = -\sum_i p_{l,i}\log p_{l,i} \le S
$$

这里 $S$ 是熵阈值。熵可以理解为“预测分布有多分散”。如果一个样本的概率质量高度集中在某个类别上，熵就低；如果多个类别概率都接近，熵就高。对新手来说，最大概率和熵都在衡量“不确定性”，只是角度不同：前者盯着最可能的那一类，后者看整个分布是否分散。

如果是二分类任务，二者的关系尤其直观。假设输出为 $p$ 和 $1-p$，那么：
$$
H(p) = -p\log p -(1-p)\log(1-p)
$$
当 $p$ 接近 $0.5$ 时，熵较高；当 $p$ 接近 $0$ 或 $1$ 时，熵较低。这说明“高置信度”与“低熵”在很多场景里是一致的，但不完全等价。

边界也要说清楚。不是所有“少算一点”的方法都叫早期退出。

| 方法 | 是否动态 | 是否逐样本不同 | 典型做法 |
|---|---:|---:|---|
| 早期退出 | 是 | 是 | 每层判断是否提前输出 |
| 蒸馏减层 | 否 | 否 | 训练一个更浅的固定模型 |
| 剪枝/量化 | 否 | 否 | 减少参数或数值精度 |
| 跳层/路由 | 可能是 | 可能是 | 对部分层或子网络做条件选择 |

初学者最容易混淆的有两组概念。

第一组是“固定减层”和“早期退出”。DistilBERT 不是早期退出。它是预先训练一个更浅的模型，例如把 12 层教师模型蒸馏成 6 层学生模型。上线之后，所有请求都固定执行这 6 层，没有逐样本动态停止的过程。因此它提供的是固定收益，而不是连续可调的速度-精度权衡。

第二组是“跳层”与“早期退出”。跳层方法可能让模型跳过某些层，但样本最终仍然到达最后输出层；早期退出则是直接提前终止整个推理过程。这两者都属于动态推理，但控制目标不同。跳层关注“少走某些步骤”，早退关注“是否现在就结束”。

看一个玩具例子更直观。假设有一个 4 层文本分类模型，输入是“这个手机壳很好看，物流也快”。第 1 层可能只捕捉到“手机壳”“物流”这些词，第 2 层已经把“很好看”“也快”整合为明显正向信号，此时若得到：
$$
p_2 = [0.97, 0.03]
$$
并设置 $\tau = 0.95$，模型就可以在第 2 层退出。

另一个输入是“价格不错但续航一般，系统更新后发热更严重”。这条文本同时出现了正面与负面线索，而且后半句更影响整体判断。假设第 2 层输出为：
$$
p_2 = [0.71, 0.29]
$$
此时最大概率还不够高，就不能退出，需要继续进入更深层。这个例子反映了早期退出的核心假设：浅层能解决简单样本，但复杂样本必须保留更深表示。

---

## 核心机制与推导

机制可以拆成四步：

1. 输入经过第 $l$ 层 Transformer，得到隐藏状态 $h_l$。
2. 中间分类头读取 $h_l$，输出 logits，再通过 softmax 得到概率分布 $p_l$。
3. 根据 $\max_i p_{l,i}$、$H(p_l)$ 或其他判据判断当前层是否可以退出。
4. 若满足退出条件，则直接返回当前预测；若不满足，则继续计算第 $l+1$ 层。

把这个过程写成统一形式，可以表示为：
$$
\hat{y}=
\begin{cases}
\arg\max p_1, & \text{if } \phi(p_1)=1 \\
\arg\max p_2, & \text{if } \phi(p_1)=0,\ \phi(p_2)=1 \\
\vdots \\
\arg\max p_L, & \text{otherwise}
\end{cases}
$$

其中 $\phi(p_l)$ 是一个退出函数。它输出 1 表示“在第 $l$ 层退出”，输出 0 表示“继续向后计算”。

这套机制真正改变的不是单次最坏成本，而是期望成本。设 $C_l$ 表示执行到第 $l$ 层时的累计 FLOPs、累计时间或累计能耗，那么平均推理成本为：
$$
\mathbb{E}[C] = \sum_{l=1}^{L} P(\text{exit at } l)\cdot C_l
$$

这个公式有两个关键信息。

第一，只有“会退出”还不够，必须“尽量早退出”才有明显收益。如果大量样本都在最后一层才退出，那么虽然系统具备早退逻辑，但平均成本几乎没有下降。

第二，收益由退出分布决定，而不是由某一个样本决定。一个简单样本第 2 层退出可以获得很高的单样本加速，但系统整体收益要看整个数据集上有多少样本在第 2 层、第 3 层、第 4 层退出。

举一个最小数值例子。假设一个 4 层模型的累计 FLOPs 如下：

| 层数 | 累计 FLOPs |
|---|---:|
| Layer 1 | 5 |
| Layer 2 | 10 |
| Layer 3 | 15 |
| Layer 4 | 20 |

若某个样本在 Layer 2 就满足 $\max_i p_i=0.97 \ge 0.95$，那么它的实际成本是 10，而完整推理成本是 20，因此单样本理论加速比为：
$$
\frac{20}{10}=2\times
$$

再看平均成本。假设 100 个样本中：

- 20% 在第 2 层退出
- 50% 在第 3 层退出
- 30% 在第 4 层退出

那么有：
$$
\mathbb{E}[C] = 0.2\times 10 + 0.5\times 15 + 0.3\times 20 = 15.5
$$

相对总是跑满 20 FLOPs 的基线，平均加速比约为：
$$
\frac{20}{15.5}\approx 1.29\times
$$

这个结果很重要，因为它说明早期退出不是“自动就快很多”。如果浅层出口的质量不够高，或者阈值过保守，模型仍然会让大部分样本走到最后，那么实现复杂度增加了，但收益并不理想。

还可以把退出分布写成累计形式。若定义：
$$
q_l=P(\text{not exit before } l)
$$
即样本走到第 $l$ 层时仍未退出的概率，那么平均成本也可写成近似的层累加形式：
$$
\mathbb{E}[C] \approx \sum_{l=1}^{L} q_l \cdot \Delta C_l
$$
其中 $\Delta C_l$ 是第 $l$ 层的边际计算成本。这个写法更贴近工程直觉：只有走到某一层的样本，才需要支付这一层的额外成本。

从训练角度看，中间分类头必须“足够能用”。如果只在推理时临时给每层加一个分类器，通常会遇到两个问题：

- 中间层表示还不够判别，浅层头部准确率低
- 浅层 softmax 虽然数值高，但并不可靠，容易过早退出

FastBERT 一类方法采用自蒸馏（self-distillation）来解决这一点。所谓自蒸馏，就是让深层输出充当教师，指导中间层分类头学习接近最终层的预测分布。若把最终层教师分布记为 $p_T$，第 $l$ 层学生分布记为 $p_l$，则常见目标之一是最小化 KL 散度：
$$
D_{\mathrm{KL}}(p_T \parallel p_l)
=
\sum_i p_T(i)\log \frac{p_T(i)}{p_l(i)}
$$

这比只用硬标签训练更稳定，因为它不仅告诉浅层“答案是哪一类”，还告诉它“其他类相对有多像正确答案”。对中间层出口来说，这种“软监督”通常比单纯交叉熵更有用。

---

## 代码实现

下面给出一个可运行的简化 Python 实现。它不依赖深度学习框架，只模拟“逐层预测并按阈值退出”的控制流，便于先理解逻辑，再迁移到真实模型。

```python
from math import log
from typing import Iterable, List, Optional, Sequence, Tuple


def entropy(probs: Sequence[float]) -> float:
    if not probs:
        raise ValueError("probs must not be empty")
    total = sum(probs)
    if total <= 0:
        raise ValueError("sum(probs) must be positive")
    normalized = [p / total for p in probs]
    return -sum(p * log(p) for p in normalized if p > 0)


def validate_probs(probs: Sequence[float]) -> None:
    if not probs:
        raise ValueError("each layer output must contain at least one class probability")
    if any(p < 0 for p in probs):
        raise ValueError("probabilities must be non-negative")
    total = sum(probs)
    if total <= 0:
        raise ValueError("sum of probabilities must be positive")


def early_exit_predict(
    layer_outputs: Sequence[Sequence[float]],
    tau: Optional[float] = 0.95,
    entropy_threshold: Optional[float] = None,
) -> Tuple[int, List[float], str]:
    """
    根据逐层概率输出决定在哪一层退出。

    参数:
        layer_outputs: 每层的类别概率分布，例如 [[0.6, 0.4], [0.97, 0.03]]
        tau: 最大类别概率阈值；设为 None 表示不使用该条件
        entropy_threshold: 熵阈值；设为 None 表示不使用该条件

    返回:
        (exit_layer, probs, reason)
        exit_layer: 退出层号，从 1 开始计数
        probs: 退出层对应的概率分布（已归一化）
        reason: 退出原因，取值为 "max_prob"、"entropy"、"fallback_last_layer"
    """
    if not layer_outputs:
        raise ValueError("layer_outputs must not be empty")
    if tau is None and entropy_threshold is None:
        raise ValueError("at least one exit criterion must be enabled")

    for idx, raw_probs in enumerate(layer_outputs, start=1):
        validate_probs(raw_probs)
        total = sum(raw_probs)
        probs = [p / total for p in raw_probs]

        max_prob = max(probs)
        ent = entropy(probs)

        if tau is not None and max_prob >= tau:
            return idx, probs, "max_prob"

        if entropy_threshold is not None and ent <= entropy_threshold:
            return idx, probs, "entropy"

    last = layer_outputs[-1]
    total = sum(last)
    probs = [p / total for p in last]
    return len(layer_outputs), probs, "fallback_last_layer"


def average_cost(exit_layers: Iterable[int], cumulative_costs: Sequence[float]) -> float:
    """
    根据退出层分布计算平均累计成本。
    cumulative_costs[i] 表示第 i+1 层的累计成本。
    """
    exit_layers = list(exit_layers)
    if not exit_layers:
        raise ValueError("exit_layers must not be empty")
    if not cumulative_costs:
        raise ValueError("cumulative_costs must not be empty")

    max_layer = len(cumulative_costs)
    total_cost = 0.0
    for layer in exit_layers:
        if layer < 1 or layer > max_layer:
            raise ValueError(f"exit layer {layer} out of range 1..{max_layer}")
        total_cost += cumulative_costs[layer - 1]
    return total_cost / len(exit_layers)


def demo() -> None:
    easy_sample = [
        [0.60, 0.40],
        [0.97, 0.03],
        [0.98, 0.02],
        [0.99, 0.01],
    ]
    exit_layer, probs, reason = early_exit_predict(easy_sample, tau=0.95)
    assert exit_layer == 2
    assert round(max(probs), 2) == 0.97
    assert reason == "max_prob"

    hard_sample = [
        [0.55, 0.45],
        [0.62, 0.38],
        [0.71, 0.29],
        [0.96, 0.04],
    ]
    exit_layer, probs, reason = early_exit_predict(hard_sample, tau=0.95)
    assert exit_layer == 4
    assert round(max(probs), 2) == 0.96
    assert reason == "max_prob"

    entropy_sample = [
        [0.52, 0.48],
        [0.88, 0.12],
        [0.90, 0.10],
    ]
    exit_layer, probs, reason = early_exit_predict(
        entropy_sample,
        tau=None,
        entropy_threshold=0.40,
    )
    assert exit_layer == 2
    assert reason == "entropy"

    cumulative_costs = [5, 10, 15, 20]
    avg = average_cost([2, 3, 3, 4, 2], cumulative_costs)
    assert abs(avg - 12.0) < 1e-9

    print("all tests passed")


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行。它比常见示意代码多做了几件对新手重要的事：

- 对输入做合法性检查，避免空列表、负概率等无意义输入
- 显式支持两类退出条件：最大概率阈值、熵阈值
- 返回退出原因，便于调试“为什么停在这一层”
- 单独提供 `average_cost`，把“单样本退出”与“总体平均收益”区分开

如果你运行这段代码，结果应该是：

```text
all tests passed
```

再看真实模型中的伪代码。以 BERT 类分类模型为例，控制流通常如下：

```python
hidden = embedding_output(input_ids, attention_mask)

for layer_idx, (layer, exit_head) in enumerate(zip(transformer_layers, exit_heads), start=1):
    hidden = layer(hidden, attention_mask=attention_mask)

    cls_repr = hidden[:, 0]                 # 例如取 [CLS] 位置表示
    logits = exit_head(cls_repr)
    probs = softmax(logits, dim=-1)

    if probs.max(dim=-1).values.item() >= tau:
        return {
            "logits": logits,
            "exit_layer": layer_idx,
            "reason": "max_prob",
        }

final_logits = final_classifier(hidden[:, 0])
return {
    "logits": final_logits,
    "exit_layer": len(transformer_layers),
    "reason": "final_layer",
}
```

但真实工程里不能只写这几行控制逻辑，还要考虑三个问题。

第一，中间头怎么训练。通常不是简单复制最终分类头，而是每个出口单独接一个轻量分类器，并在训练时联合优化，或者用最终层结果蒸馏中间头。

第二，阈值怎么来。阈值一般不是拍脑袋设定，而是在验证集上调出来的。常见做法是扫一组 $\tau$ 值，分别测量准确率、平均退出层、平均时延，再选一个满足业务要求的点。

第三，日志怎么打。线上部署时，至少要记录每层退出占比、不同类别的退出分布、误判样本的退出层，以及阈值变更前后的速度与精度变化。否则很难判断早退逻辑是“真的省了算力”，还是“只是让错误更早发生”。

看一个更贴近业务的例子。假设客服意图分类系统需要识别“退款”“发票”“物流”“售后”等类别。很多请求像“我要退货”“开电子发票”“帮我查物流”都很短、模式也固定，这些输入往往在前 2 到 3 层就能给出可靠预测；而像“上次你们说能补发，但现在订单状态没更新，我该找谁处理”这类长文本，意图边界更复杂，通常要走到更深层。早期退出适合这种场景，因为简单样本占比高，且业务更关心平均时延而不是每个请求都固定执行同样路径。

---

## 工程权衡与常见坑

早期退出真正难的地方不是“把出口挂上去”，而是“怎样让退出决策可信且有收益”。下面几个坑在论文和工程里都很常见。

第一个坑是阈值敏感。$\tau$ 过低，很多样本会过早退出，速度看起来很好，但精度会明显下降；$\tau$ 过高，又会导致大部分样本都跑到最后，收益接近没有。阈值本质上控制了系统的风险偏好。

| 阈值 $\tau$ | 平均退出层 | 平均成本占满层比例 | 精度变化趋势 |
|---|---:|---:|---|
| 0.80 | 2.1 | 55% | 下降风险高 |
| 0.90 | 2.8 | 70% | 常见折中点 |
| 0.95 | 3.4 | 85% | 更稳但加速变弱 |
| 0.99 | 3.9 | 98% | 接近无退出 |

这个表的重点不是具体数值，而是趋势：早退机制没有单一“最优阈值”，只有对某个业务目标更合适的阈值。例如，客服预分类允许少量误差但要求低延迟，阈值可以略低；医疗、风控、法务等高风险任务就不能这么做。

第二个坑是置信度未校准。softmax 输出高，不等于模型真的可靠。很多分类模型存在过度自信问题，也就是它给出 $0.95$ 的概率，但真实正确率未必接近 95%。如果直接把未校准置信度拿来做退出条件，就可能在错误但自信的样本上提前截断。

这一点可以用校准误差来描述。设模型的置信度为 $\hat{p}$，真实准确率为 $\text{acc}(\hat{p})$，则理想情况是：
$$
\text{acc}(\hat{p}) \approx \hat{p}
$$
如果偏差很大，说明退出条件不可信。工程上常见的补救方法包括：

- 温度缩放（temperature scaling）
- 在验证集上单独校准中间层输出
- 不同类别使用不同阈值
- 对高风险类别禁用早退
- 结合熵、margin、能量函数等多种指标，而不是只看最大概率

第三个坑是浅层头部训练不足。很多中间层天然更偏向词法和局部模式，对复杂语义的判别能力弱。如果训练时没有专门约束中间头，它们往往出现两种情况：

- 准确率低，导致根本不敢早退
- 表面上置信度高，但排序不稳定，导致早退后误判

这正是 FastBERT 等方法强调自蒸馏的原因。把最终层作为教师，能让浅层出口尽量学习到“完整模型会怎么判断”，而不是只靠中间层原生表示硬分类。

第四个坑是线上分布偏移。训练集上的“简单样本”不一定等于线上真实流量中的“简单样本”。例如训练时大多是短句，线上突然出现大量长文本、混合意图、跨句依赖样本，中间层可能会给出虚高置信度。常见应对方式有：

- 增加困难样本训练
- 使用线上回流数据重做校准
- 对超长文本直接关闭早退
- 对低频类别或高损失类别设置更严格阈值
- 在退出前增加一致性检查，例如比较相邻两层预测是否稳定

第五个坑是理论 FLOPs 收益不等于真实延迟收益。很多论文先报告“平均计算量下降多少”，但上线后真实加速未必同步。原因在于实际系统还受以下因素影响：

| 因素 | 为什么会影响真实收益 |
|---|---|
| 框架调度开销 | 频繁分支判断和多出口逻辑会增加控制开销 |
| 批处理不规则 | 同一批样本退出层不同，会破坏并行 |
| GPU 利用率 | 对大 batch GPU 推理，动态分支可能降低吞吐 |
| 内存访问 | 多个出口头和中间张量保留会增加带宽压力 |
| 服务链路 | 网络、序列化、后处理可能掩盖部分模型加速收益 |

因此，早期退出通常在以下场景更容易体现价值：

- CPU 推理
- 小 batch 或单请求在线服务
- 边缘设备、移动端
- 延迟敏感而非吞吐绝对优先的服务

而在大 batch GPU 推理中，理论 FLOPs 降低并不一定转化为同等比例的吞吐提升。因为 GPU 更擅长规则、连续、统一的计算图，不规则的样本级分支容易稀释收益。

第六个坑是指标设计不完整。只看准确率和平均时延是不够的，至少还应同时观察：

- 每层退出比例
- 各类别的退出深度分布
- 不同文本长度上的退出行为
- 错误样本的平均退出层
- P95/P99 延迟，而不只是平均延迟

如果只看平均值，很容易误以为系统“整体更快了”，但实际上某些关键类别被过早截断，或者尾延迟并没有改善。

---

## 替代方案与适用边界

早期退出不是动态推理的唯一解，也不是所有任务的最优解。它适合的是“样本难度差异明显，且业务希望把这种差异转化为计算收益”的场景。

如果业务要求非常稳定，例如每个请求都必须在固定时延预算内完成，而且部署环境希望尽量简单，那么 DistilBERT 这类固定减层模型通常更直接。它的优点是结构规整、工程路径成熟、性能收益稳定；缺点是对所有样本一刀切，不能利用“简单样本本来就不需要完整深度”这一事实。

如果资源主要瓶颈是显存、模型大小或带宽，而不是逐样本平均计算量，那么剪枝、量化、低秩分解等压缩方法通常更合适。它们改变的是模型整体资源占用，不依赖动态控制逻辑，也更容易在统一推理框架里部署。

如果任务允许更复杂的动态决策，还可以使用跳层、条件路由、MoE 类门控等方法。它们不是单纯决定“现在停不停止”，而是决定“接下来走哪条路径”或“哪些模块需要激活”。这类方法灵活性更高，但工程复杂度也更高。

| 方法 | 是否 sample-wise | 代表工作 | 最佳部署场景 |
|---|---:|---|---|
| DistilBERT | 否 | DistilBERT | 固定延迟、简化部署 |
| 早期退出 | 是 | DeeBERT、FastBERT | 延迟敏感且样本难度差异大 |
| 剪枝/量化 | 否 | 多类压缩方法 | 资源受限、统一优化 |
| 条件路由/跳层 | 是 | SkipNet、LayerDrop 类思路及后续动态网络工作 | 更复杂的动态推理系统 |

适用边界也必须明确。

第一，如果数据集本身极难，几乎所有样本都需要深层语义才能区分，那么早期退出的空间很小。此时多数样本都会跑到最后一层，动态逻辑几乎只增加复杂度，不产生实质收益。

第二，如果任务不是分类，而是生成式解码、多轮工具调用、严格排序或多步推理，那么“什么时候退出”不再是一个简单的 softmax 阈值问题。生成任务中的每一步解码都可能改变后续分布，不能简单套用分类任务里的早退规则。

第三，如果业务强依赖稳定批处理吞吐，而不是单请求低延迟，那么早退未必优于固定压缩。因为样本在不同层退出会打乱批次一致性，降低硬件利用率。

第四，如果模型置信度长期不稳定，或者任务错误代价极高，那么即便理论上能早退，工程上也可能不应启用。动态推理的前提不是“能算出概率”，而是“这个概率足够可信，足以驱动控制决策”。

可以把这些方案的差异总结为一句话：固定减层是在模型结构上做统一裁剪，给出离散的几个运行点；早期退出则是在完整模型内部增加多个动态出口，给出连续可调的速度-精度曲线。前者更简单，后者更灵活，但也更依赖校准、训练和系统实现质量。

---

## 参考资料

- DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference | ACL 2020  
  论文链接：https://aclanthology.org/2020.acl-main.204/  
  解决的问题：给 BERT 增加逐层退出头，并使用熵作为退出判据，系统展示了 Transformer 分类任务中“按样本动态停止”的可行性。  
  值得关注的点：它把“早退”从 CNN 领域迁移到 BERT，并明确展示了速度与精度之间的可调关系。

- FastBERT: a Self-distilling BERT with Adaptive Inference Time | ACL 2020  
  论文链接：https://aclanthology.org/2020.acl-main.537/  
  解决的问题：通过自蒸馏训练中间分类器，使每个出口都具备较强判别能力，并支持按阈值进行自适应推理。  
  值得关注的点：不是只讨论“怎么退”，而是重点解决“中间出口怎么训得可用”。

- DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter | 2019  
  论文链接：https://arxiv.org/abs/1910.01108  
  解决的问题：作为固定减层代表，通过知识蒸馏训练更浅模型，在精度损失可控前提下降低推理成本。  
  值得关注的点：它不是早期退出方法，但非常适合作为对照边界，用来理解“固定压缩”和“动态推理”的区别。

- BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks | 2016  
  论文链接：https://arxiv.org/abs/1709.01686  
  解决的问题：在深度网络中引入分支出口，让简单样本在浅层分类，属于早期退出思想的经典起点之一。  
  值得关注的点：虽然不是 Transformer 论文，但很多后续早退工作都继承了“多出口 + 阈值控制”的基本框架。

- SkipNet: Learning Dynamic Routing in Convolutional Networks | ECCV 2018  
  论文链接：https://arxiv.org/abs/1711.09485  
  解决的问题：通过学习式门控决定样本跳过哪些层，属于与早退相邻的动态推理路线。  
  值得关注的点：它帮助区分“提前停止”和“条件跳层”这两类不同的动态计算机制。

- SmartBERT 及后续动态推理工作 | 2023 及后续  
  解决的问题：围绕中间层表示不足、早退误判、训练不稳定等问题，引入更强一致性约束、改进校准与动态控制策略。  
  阅读建议：把这类工作看成对 DeeBERT/FastBERT 路线的补强，重点关注它们如何改善中间头质量与退出可靠性，而不是只比较最终准确率。

这些工作共同构成本文的基础脉络：

- DeeBERT 说明“怎么把早退装进 BERT”
- FastBERT 说明“怎么把中间出口训好”
- DistilBERT 提供“固定减层”的对照边界
- BranchyNet 和 SkipNet 帮助理解更广义的动态推理背景

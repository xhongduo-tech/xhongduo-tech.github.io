## 核心结论

专家坍缩，指的是在稀疏 MoE（Mixture of Experts，专家混合模型）训练中，少数专家反复接收大部分 token，其他专家几乎拿不到训练样本，最终更新趋近于停滞。结果是：模型名义上有很多专家，实际只有少数专家在持续工作，参数容量、显存占用和通信开销都没有换来对应收益。

这不是一次性的“分配不均”，而是一个会自我强化的动态过程。硬路由配合 top-k 选择，会把训练初期极小的随机优势不断放大。某个专家只要在早期略占上风，就会收到更多 token；收到更多 token，就会拿到更多梯度；拿到更多梯度，又会更容易在后续路由中继续胜出。

一个玩具例子可以直接说明问题。假设系统配置了 16 个专家，设计目标是让 token 大致分散；但训练早期很快演化成前 4 个专家分别处理 47%、22%、18%、8%，剩余 12 个专家合计不到 5%。这时模型虽然“挂着”16 个专家的配置，实际可用容量已经接近一个只有 4 个有效分支的小模型。

负载平衡损失常写为：

$$
\mathcal{L}_{\mathrm{load}} = E \sum_{i=1}^{E} f_i G_i
$$

其中：

- $E$ 是专家数。
- $f_i$ 是专家 $i$ 实际收到的 token 比例。
- $G_i$ 是 gate（门控网络）对专家 $i$ 给出的平均路由概率。

理想情况下，各专家负载接近均匀，此时该项接近 1；如果少数专家长期垄断，$f_i$ 与 $G_i$ 会同时向头部专家集中，这个值会偏离理想区间，训练就需要借助该辅助项把路由重新拉回更平衡的状态。

| 状态 | 活跃专家数 | 前 4 专家负载占比 | 其余专家是否持续更新 |
|---|---:|---:|---|
| 未坍缩 | 12-16 | 35%-50% | 大多能更新 |
| 已坍缩 | 3-4 | 80%-95% | 大多停滞 |

从工程视角看，专家坍缩直接对应成本浪费。假设一个 16 专家的 MoE 中有 4 个专家处理了 87% 的 token，那么系统仍然要承担 16 专家的参数存储、路由、跨卡通信和调度成本，但模型效果往往只接近一个远小于理论容量的系统，泛化能力也通常更差。

---

## 问题定义与边界

本文讨论的“专家坍缩”有明确边界：它主要出现在以下条件同时成立时：

| 条件 | 含义 | 为什么相关 |
|---|---|---|
| 稀疏 MoE | 每个 token 只走少数几个专家 | 会产生明显竞争 |
| 硬 top-k 路由 | 未入选专家几乎没有前向和反向贡献 | 梯度分配不连续 |
| 训练阶段 | 专家参数仍在更新 | 早期偏差会被放大 |
| 分布式部署 | 专家通常跨设备分布 | 负载失衡还会放大系统代价 |

这里的“稀疏”不是“专家很多”本身，而是“每个 token 只使用少数几个专家”。如果所有专家都参与计算，那是稠密混合，不属于本文重点。

更形式化地说，给定 token 表示 $x$，门控网络输出 logits（未归一化分数）$z \in \mathbb{R}^E$，softmax 后得到路由概率：

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^E \exp(z_j)}
$$

如果采用 top-k 硬选择，那么只有概率最高的 $k$ 个专家真正参与计算，其余专家对这个 token 的前向与反向近似消失。于是梯度不是连续地分给所有专家，而是集中流向少数被选中的专家。

对新手来说，可以把它理解成一种“筛选再训练”的机制：

1. gate 先给每个专家打分。
2. 系统只保留分数最高的 $k$ 个专家。
3. 只有这 $k$ 个专家真的看到这个 token。
4. 没被选中的专家，这一步几乎学不到任何东西。

训练前几千步里，如果专家 A 和 B 因初始化略占优势，经常挤进 top-k；专家 C 到 P 则几乎见不到 token。结果是 A、B 更新更快，下一轮也更容易继续入选。这个过程不是偶然波动，而是竞争机制本身带来的偏置累积。

下表给出哪些设置更容易触发坍缩：

| 设置 | 是否易坍缩 | 原因 |
|---|---|---|
| 稀疏 top-1 路由 | 很高 | 每个 token 只给 1 个专家，梯度最集中 |
| 稀疏 top-2 路由 | 高 | 比 top-1 略好，但仍存在强竞争 |
| 硬路由 + 低温度 softmax | 很高 | 概率分布更尖锐，赢家更稳定 |
| 软路由 | 低 | 所有专家都能收到一部分梯度 |
| 训练期加噪声/抖动 | 降低 | 能打破训练早期的确定性偏差 |
| 有负载平衡正则 | 降低 | 显式惩罚过度集中 |

这里还需要区分几个容易混淆的概念。

| 问题 | 含义 | 是否等于专家坍缩 |
|---|---|---|
| 负载不均衡 | 专家接收 token 数量不同 | 不一定 |
| 专家坍缩 | 少数专家长期垄断且其他专家停更 | 是 |
| 通信热点 | 某些设备的流量过高 | 可能是结果，不是定义本身 |

因此，短时间的不均衡未必危险。真正危险的是长期、稳定、会持续自增强的不均衡。

下面是一段可直接运行的最小代码，展示 top-k 路由如何统计 $f_i$ 与 $G_i$。它只依赖 Python 标准库。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def compute_load_stats(logits_batch, k):
    T = len(logits_batch)
    E = len(logits_batch[0])

    probs_batch = [softmax(row) for row in logits_batch]
    chosen = [topk_indices(probs, k) for probs in probs_batch]

    expert_counts = [0] * E
    for picked in chosen:
        for e in picked:
            expert_counts[e] += 1

    f = [c / (T * k) for c in expert_counts]
    G = [sum(probs[i] for probs in probs_batch) / T for i in range(E)]
    load_loss = E * sum(fi * gi for fi, gi in zip(f, G))
    return chosen, expert_counts, f, G, load_loss

if __name__ == "__main__":
    logits_batch = [
        [3.2, 2.9, 1.0, 0.8],
        [3.0, 2.8, 1.1, 0.7],
        [3.3, 3.1, 0.9, 0.6],
        [3.1, 2.7, 1.0, 0.8],
    ]
    chosen, counts, f, G, load_loss = compute_load_stats(logits_batch, k=2)

    print("chosen =", chosen)
    print("counts =", counts)
    print("f =", [round(x, 4) for x in f])
    print("G =", [round(x, 4) for x in G])
    print("load_loss =", round(load_loss, 4))

    assert sum(counts) == len(logits_batch) * 2
    assert abs(sum(f) - 1.0) < 1e-9
    assert abs(sum(G) - 1.0) < 1e-9
```

这段代码里的两个量含义很明确：

- `f` 是“实际被选中的次数比例”。
- `G` 是“gate 在概率层面偏向谁”。

前者看结果，后者看偏好。两者一起看，才能判断系统是在短时波动，还是已经进入长期偏斜。

---

## 核心机制与推导

专家坍缩的核心不是“某些专家更强”这句话本身，而是“被选中”会进一步制造变强的条件。这个机制可以拆成四步：

1. 门控网络对每个 token 计算所有专家的分数。
2. top-k 只保留少数高分专家。
3. 被选中的专家收到前向样本与反向梯度。
4. 参数更新后，这些专家对相似 token 的表现更好，于是下一轮更容易再被选中。

这构成一个标准的正反馈循环：

$$
\text{更多 token}
\rightarrow
\text{更大梯度}
\rightarrow
\text{更快变好}
\rightarrow
\text{更容易被选中}
$$

如果把“梯度”理解成“模型根据误差修正参数的信号”，那么问题就很直观了：未被路由到的专家几乎收不到这个信号，也就几乎没有学习机会。

继续用 16 专家的玩具例子。假设某一批次结束后，负载分布如下：

| 专家组 | token 占比 | 梯度量级（示意） | 状态 |
|---|---:|---:|---|
| 专家 1 | 47% | 1.00 | 高活跃 |
| 专家 2 | 22% | 0.52 | 高活跃 |
| 专家 3 | 18% | 0.40 | 高活跃 |
| 专家 4 | 8% | 0.19 | 中活跃 |
| 专家 5-16 | 5% 合计 | 0.01-0.03 | 近休眠 |

这里的“梯度量级”不是严格定值，只是说明趋势：收到的 token 越多，累计梯度通常越大，参数变化也越明显。于是专家 1-4 持续增强，专家 5-16 因长期样本不足，很难重新进入竞争区间。

为什么负载平衡损失能缓解这个问题？关键是它同时约束两个量：

1. $f_i$：专家 $i$ 实际处理了多少 token。
2. $G_i$：路由器整体有多偏爱专家 $i$。

形式化地，实际负载比例可以写成：

$$
f_i = \frac{1}{Tk}\sum_{t=1}^{T}\sum_{r=1}^{k}\mathbf{1}\{e_{t,r}=i\}
$$

其中：

- $T$ 是 token 数。
- $k$ 是每个 token 选中的专家数。
- $\mathbf{1}\{\cdot\}$ 是指示函数，条件为真时取 1，否则取 0。
- $e_{t,r}$ 表示第 $t$ 个 token 的第 $r$ 个路由位置选中了哪个专家。

平均门控概率写成：

$$
G_i = \frac{1}{T}\sum_{t=1}^{T} p_{t,i}
$$

如果所有专家都接近均匀，那么有：

$$
f_i \approx \frac{1}{E}, \quad G_i \approx \frac{1}{E}
$$

代入负载平衡项：

$$
\mathcal{L}_{\mathrm{load}}
=
E \sum_{i=1}^E f_i G_i
\approx
E \sum_{i=1}^E \frac{1}{E}\frac{1}{E}
=
E \cdot \frac{1}{E}
=
1
$$

这说明该辅助项不是在追求“绝对相等”，而是在惩罚“实际分配”和“路由偏好”同时集中到头部专家的情况。换句话说，只有少数专家既被高概率偏爱、又实际吃掉大部分 token 时，惩罚才会显著上升。

进一步看，top-k 的不可微选择还会带来一个训练动力学问题。设 gate 输出 logits 为 $z$，softmax 概率为 $p$。在纯 soft routing 下，每个专家的梯度都与 $p_i$ 相关，因此即使权重很小，也仍可能得到少量更新；但在硬 top-k 路由中，未进入 top-k 的专家在该 token 上近似满足：

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} \approx 0
\quad \text{if } i \notin \mathrm{TopK}(p)
$$

其中 $\theta_i$ 表示专家 $i$ 的参数。这个近似式不要求精确推导，也足够说明问题本质：在硬路由下，“没被选中”往往就等价于“这一步没法学”。

真实工程里，这种问题常常比玩具例子更严重。比如一个 1.6B 级 MoE，16 个专家分布在多张卡上，4 个专家吃掉 87% 的 token。此时除了模型容量浪费，还会出现两类额外代价：

| 代价 | 具体表现 |
|---|---|
| 统计效率下降 | 大量专家长期学不到东西，参数只占内存不产出能力 |
| 系统效率下降 | 热门专家所在设备更拥堵，all-to-all 通信更容易抖动 |

因此，专家坍缩本质上既是优化问题，也是系统问题。只从损失函数看它，会低估部署成本；只从系统负载看它，又会忽略其训练根源。

---

## 代码实现

工程里最稳妥的做法不是“等坍缩发生后再补救”，而是在路由层早期就加入抑制机制。常见手段有三类：

| 手段 | 作用点 | 目标 |
|---|---|---|
| 负载平衡损失 | 损失函数 | 约束长期偏斜 |
| logits 加噪声 | 路由打分 | 打破早期确定性优势 |
| 训练期 routing dropout | top-k 选择后 | 强制系统保留探索 |

下面给出一个可运行的 Python 玩具实现。它不依赖深度学习框架，只模拟门控、top-k、统计 $f_i$、计算 load loss，并演示噪声与 dropout 如何改变分配结果。

```python
import math
import random

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def route_tokens(logits_batch, k=2, noise_std=0.0, drop_prob=0.0, seed=0):
    rng = random.Random(seed)
    T = len(logits_batch)
    E = len(logits_batch[0])

    probs_batch = []
    chosen = []
    counts = [0] * E

    for logits in logits_batch:
        noisy_logits = [x + rng.gauss(0.0, noise_std) for x in logits]
        probs = softmax(noisy_logits)
        probs_batch.append(probs)

        ranked = sorted(range(E), key=lambda i: probs[i], reverse=True)

        picked = []
        for idx in ranked:
            if len(picked) == k:
                break
            if rng.random() < drop_prob:
                continue
            picked.append(idx)

        if len(picked) < k:
            for idx in ranked:
                if idx not in picked:
                    picked.append(idx)
                if len(picked) == k:
                    break

        chosen.append(picked)
        for e in picked:
            counts[e] += 1

    f = [c / (T * k) for c in counts]
    G = [sum(probs[i] for probs in probs_batch) / T for i in range(E)]
    load_loss = E * sum(fi * gi for fi, gi in zip(f, G))
    return {
        "chosen": chosen,
        "counts": counts,
        "f": f,
        "G": G,
        "load_loss": load_loss,
    }

if __name__ == "__main__":
    logits_batch = [
        [3.2, 3.0, 1.1, 0.9],
        [3.1, 2.9, 1.0, 0.8],
        [3.3, 3.1, 1.2, 1.0],
        [3.0, 2.8, 1.0, 0.7],
        [3.4, 3.2, 1.1, 0.9],
        [3.2, 3.1, 1.0, 0.8],
    ]

    base = route_tokens(logits_batch, k=2, noise_std=0.0, drop_prob=0.0, seed=42)
    regularized = route_tokens(logits_batch, k=2, noise_std=0.8, drop_prob=0.2, seed=42)

    print("baseline counts =", base["counts"], "load_loss =", round(base["load_loss"], 4))
    print("regularized counts =", regularized["counts"], "load_loss =", round(regularized["load_loss"], 4))

    assert sum(base["counts"]) == len(logits_batch) * 2
    assert sum(regularized["counts"]) == len(logits_batch) * 2
    assert abs(sum(base["f"]) - 1.0) < 1e-9
    assert abs(sum(regularized["f"]) - 1.0) < 1e-9
```

这段代码有三个值得明确指出的地方。

第一，`f` 是从“最终选中了谁”统计出来的，表示实际负载。  
第二，`G` 是 softmax 概率的 batch 平均，表示门控偏好。  
第三，噪声和 dropout 并不是直接修正专家参数，而是先打破过早固化的选择，让冷门专家有机会重新拿到训练样本。

如果换成深度学习框架，路由层的 forward 流程通常可以概括为：

```python
def moe_router_forward(x, gate, k, noise_std, drop_prob, alpha, training):
    logits = gate(x)  # [T, E]

    if training and noise_std > 0:
        logits = logits + randn_like(logits) * noise_std

    probs = softmax(logits, dim=-1)  # [T, E]
    topk_val, topk_idx = topk(probs, k=k, dim=-1)  # [T, k]

    if training and drop_prob > 0:
        keep_mask = (rand_like(topk_val) > drop_prob).float()
        topk_val = topk_val * keep_mask

    dispatch = build_dispatch_matrix(topk_idx, topk_val)
    expert_outputs = run_experts(dispatch, x)

    f = compute_actual_fraction(topk_idx, num_experts=probs.shape[-1], k=k)
    G = probs.mean(dim=0)
    load_loss = probs.shape[-1] * (f * G).sum()

    total_loss = task_loss(expert_outputs) + alpha * load_loss
    return expert_outputs, total_loss
```

上面这段是工程结构示意，不是可直接运行代码。真正落地时，下面几个点必须处理清楚：

| 环节 | 必须注意的问题 |
|---|---|
| `topk_val` dropout | 被置零后是否重新归一化 |
| `dispatch` 构造 | token 到专家的索引映射是否和梯度路径一致 |
| capacity 限制 | 某个专家满载后，多余 token 如何处理 |
| 辅助损失系数 `alpha` | 过小无效，过大可能压制主任务 |
| 训练 / 推理分支 | 推理阶段通常关闭噪声与 dropout |

不同配置对重分配的典型影响可概括如下：

| 配置 | 对负载再分配的作用 | 常见副作用 |
|---|---|---|
| `noise_std` 增大 | 提升冷门专家被选中的概率 | 过大时训练抖动明显 |
| routing dropout | 降低头部专家的连续垄断能力 | 过强时损伤主任务收敛 |
| router 慢学习率 | 防止 gate 过早定型 | 收敛速度更慢 |
| 辅助 load loss | 直接约束偏斜 | 系数过大时影响任务目标 |

对新手来说，可以用一句话概括这些手段的共同逻辑：不要让路由器在训练很早的时候就“认死理”。

---

## 工程权衡与常见坑

工程里最常见的误区，是把“专家分工”误解成“负载不均衡很正常，所以不用管”。这只说对了一半。专家分工确实合理，但分工不等于长期只有少数专家在工作。前者是 specialization，表示不同专家学会处理不同模式；后者是 collapse，表示大量专家长期失去训练机会。

第一个常见坑是刚性 top-k。它的优点是推理高效，缺点是训练早期过于残酷。某个专家初始化只要略好一点，后续就更容易连胜，其他专家几乎拿不到翻盘机会。这个问题在 top-1 中比 top-2 更严重，因为 top-1 完全不给第二名留下参与空间。

第二个坑是不给路由器随机性。没有噪声、没有 dropout、温度又很低时，路由几乎是确定的。确定性本身不是错误，但在训练前期，确定性会把尚未成熟的偏好迅速冻结成长期结构。

第三个坑是初始化与优化器设置不当。专家参数初始尺度略有差异，或者 router 学习率明显快于专家网络，都会让门控更早形成“明星专家”结构。之后即便主任务 loss 继续下降，专家层内部也可能已经坍缩。

下面这个表格可以直接作为排障清单：

| 坑 | 现象 | 对策 |
|---|---|---|
| 刚性 top-k | 前几个专家快速垄断 | 训练期改 top-2、加辅助 loss |
| 无噪声 | 路由很早固定 | logits jitter 或高斯噪声 |
| 无 dropout | 头部专家持续吃满 | routing dropout |
| 初始化差异大 | 训练早期就失衡 | 专家并行初始化、统一尺度 |
| router 学习率过大 | gate 比专家更快定型 | router 慢学习率 |
| 只看总 loss | 主任务看似正常，专家已休眠 | 监控 expert_counts、路由熵、负载方差 |

这里建议至少监控四类指标：

| 指标 | 含义 | 异常信号 |
|---|---|---|
| `expert_counts` | 每个专家处理的 token 数 | 头部长期过高，尾部接近 0 |
| load loss | 负载偏斜程度 | 长期偏离理想区间 |
| 路由熵 | gate 输出分布是否过尖 | 熵持续过低 |
| 活跃专家数 | 真正参与更新的专家数量 | 明显低于总专家数 |

路由熵可写为：

$$
H(p_t) = -\sum_{i=1}^{E} p_{t,i}\log p_{t,i}
$$

如果平均熵过低，通常说明 gate 对少数专家的偏好已经非常尖锐；如果同时负载也高度集中，就要警惕系统正在进入坍缩区间。

routing dropout 可以理解成“训练时故意让一部分本来会入选的专家临时失效”，从而迫使 token 分流。下面是一段可运行的示意代码：

```python
def apply_routing_dropout(topk_scores, drop_prob, training, random_values):
    if not training:
        return topk_scores[:]

    masked = []
    for score, rv in zip(topk_scores, random_values):
        masked.append(0.0 if rv < drop_prob else score)
    return masked

if __name__ == "__main__":
    scores = [0.82, 0.71]
    masked = apply_routing_dropout(
        scores,
        drop_prob=0.5,
        training=True,
        random_values=[0.3, 0.8],
    )
    print(masked)
    assert masked == [0.0, 0.71]
```

它的关键点不在于“永久压制高分专家”，而在于“训练期间保留探索机会”。推理阶段通常恢复确定性路由，否则延迟、吞吐和结果稳定性都会受影响。

一个实用的工程判断标准是：

| 现象 | 更像正常分工 | 更像专家坍缩 |
|---|---|---|
| 负载有差异，但尾部专家仍持续更新 | 是 | 否 |
| 头部专家明显更忙，但活跃专家数稳定 | 是 | 否 |
| 多数专家长期接近零样本 | 否 | 是 |
| 总 loss 正常下降，但路由熵持续降低且尾部停更 | 否 | 是 |

所以，真正需要警惕的不是“每个专家不一样”，而是“系统不再给大多数专家学习机会”。

---

## 替代方案与适用边界

如果目标是彻底减轻专家坍缩，而不是坚持极致推理效率，可以考虑从“硬选择”转向“软选择”。软路由不会让未入选专家完全断梯度，因此天然更稳定。

最基本的形式是：

$$
y = \sum_{i=1}^{E} p_i \cdot \mathrm{Expert}_i(x)
$$

其中 $p_i$ 由 softmax 给出。还可以引入温度参数 $\tau$：

$$
p_i = \frac{\exp(z_i/\tau)}{\sum_{j=1}^{E}\exp(z_j/\tau)}
$$

温度的直观含义如下：

| 温度 $\tau$ | 概率分布 | 路由行为 |
|---|---|---|
| 较大 | 更平滑 | 更多专家参与 |
| 适中 | 有区分但不过尖 | 折中 |
| 较小 | 更尖锐 | 行为接近硬路由 |

对应代码也很简单，下面这段可以直接运行：

```python
import math

def soft_routing(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    probs = [e / s for e in exps]

    assert abs(sum(probs) - 1.0) < 1e-9
    return probs

if __name__ == "__main__":
    logits = [2.0, 1.0, 0.5]
    for tau in [2.0, 1.0, 0.5]:
        p = soft_routing(logits, temperature=tau)
        print("tau =", tau, "probs =", [round(v, 4) for v in p])
```

但软路由的代价也很明确：参与计算的专家更多，意味着更多 FLOPs、更多显存访问和更多通信。它通常更适合训练稳定性优先的场景，不一定适合对延迟极敏感的线上推理。

常见替代路线可以这样比较：

| 路线 | collapse 风险 | 训练成本 | 推理效率 | 适用边界 |
|---|---|---:|---:|---|
| 硬 top-1 sparse | 很高 | 低 | 很高 | 极致效率优先 |
| 硬 top-2 sparse | 高 | 中 | 高 | 大多数稀疏 MoE |
| 硬路由 + load loss + noise | 中 | 中 | 高 | 工程上最常见 |
| 软路由 | 低 | 高 | 低 | 稳定性优先 |
| 熵正则化 | 低到中 | 中 | 中 | 希望增加探索 |
| 训练软、推理硬 | 较低 | 中到高 | 高 | 实际工程常用折中 |

“训练软、推理硬”是一个很实用的折中方案。它的思路是：

1. 训练时允许路由更平滑或带随机性，让更多专家获得样本。
2. 推理时恢复 deterministic routing，只选最有把握的少数专家。
3. 这样既保留训练期稳定性，也尽量保留线上效率。

如果把它说得更直接一点，替代方案之间真正的差别不是“谁更先进”，而是“你愿意把成本花在训练稳定性，还是花在推理效率”。

下面这个表格适合在方案选型时直接使用：

| 你的主要约束 | 更合适的方向 |
|---|---|
| 线上延迟最重要 | 硬路由，但必须加强早期稳定措施 |
| 训练稳定性最重要 | 软路由或更强正则 |
| 希望兼顾两者 | 训练软、推理硬 |
| 当前已经出现坍缩 | 先加强噪声、dropout、load loss 与监控 |
| 多卡通信已经成为瓶颈 | 不只看算法，还要看专家放置和热点分布 |

所以替代方案的核心不是“哪种理论最好”，而是“你的系统更怕什么”。如果更怕坍缩，就提高探索与正则；如果更怕延迟，就坚持稀疏硬路由，但必须用训练期机制抑制路由退化。

---

## 参考资料

| 来源 | 侧重点 | 对应章节 |
|---|---|---|
| Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | 稀疏 MoE、门控路由、负载均衡辅助项的经典起点 | 核心结论、问题定义、核心机制 |
| Fedus et al., *Switch Transformers* | top-1 路由、负载平衡、capacity 与工程稳定性 | 问题定义、代码实现、工程权衡 |
| Lepikhin et al., *GShard* | 大规模分布式 MoE、跨设备路由与系统代价 | 核心机制、工程权衡 |
| Zhou et al., *Mixture-of-Experts with Expert Choice Routing* 或同类工作 | 从路由策略角度讨论如何降低头部垄断 | 替代方案与适用边界 |
| Dropout regularization in hierarchical mixture of experts | dropout 如何打破早期偏置、改善专家参与度 | 代码实现、工程权衡 |
| EmergentMind 上的 MoE / Sparse MoE 综述条目 | 适合快速回顾 $f_i$、$G_i$、负载平衡与 collapse 定义 | 核心结论、核心机制 |
| 工程博客与实现笔记，如 DeepSpeed-MoE、Megatron-LM 相关文章 | 初始化、router 学习率、capacity overflow、监控指标 | 代码实现、工程权衡 |

这些资料可以按“解决什么问题”来读：

- 如果想先弄清“专家坍缩到底是什么”，优先看稀疏 MoE 基础论文和综述材料。
- 如果想回答“为什么少数专家会越学越强”，重点看 top-k 路由、capacity 和负载平衡项的推导。
- 如果想解决“训练中具体该怎么压制坍缩”，重点看 Switch、GShard 以及工程实现笔记。
- 如果想理解“为什么这不仅是模型问题，也是系统问题”，要同时看分布式路由和通信热点分析。

对新手更实用的阅读顺序通常是：

1. 先看定义与总图，理解 MoE 的训练流程。
2. 再看负载平衡损失和 top-k 机制，搞清楚为什么会形成正反馈。
3. 最后看工程实现与替代方案，理解噪声、dropout、capacity、慢学习率分别在修什么问题。

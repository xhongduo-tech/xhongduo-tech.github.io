## 核心结论

混合专家模型，MoE，指的是“每个 token 只激活少数几个子网络”的稀疏架构。它的训练难点不在“专家够不够强”，而在“路由能不能持续把 token 分散出去”。

如果 router，路由器，也就是“决定 token 该送给哪个专家的小网络”，长期把大多数 token 发给少数专家，就会出现专家崩溃。专家崩溃的直白解释是：少数专家越来越忙、越来越会做事，其他专家越来越闲、越来越学不到东西。结果是算力虽然看起来是稀疏的，训练行为却越来越像密集模型，MoE 最值钱的并行和容量优势都会打折。

MoE 训练里通常有三类负载均衡手段：

| 手段 | 类型 | 直接作用点 | 典型副作用 |
| --- | --- | --- | --- |
| Auxiliary loss | 软约束 | 通过梯度惩罚负载不均 | 权重大时会干扰主任务 |
| Capacity factor | 硬约束 | 限制每个专家每个 batch 最多接收多少 token | 过小会大量 drop token |
| Bias-based routing | 前向约束 | 直接修改路由分数，压热点专家、抬冷门专家 | 需要稳定的更新策略 |

对初学者最重要的结论有两个。

第一，均衡路由是 MoE 的命脉。没有负载均衡，专家数再多也可能白加。

第二，aux loss、capacity、bias 不是互斥关系，而是不同层面的控制器。aux loss 用梯度“慢慢纠偏”，capacity 用硬上限“立刻截流”，bias routing 用前向分数“直接改道”。工程上往往需要联合使用。

一个玩具例子可以直接说明问题。假设有 4 位主厨代表 4 个专家，100 份订单里 80 份都分给 A，B/C/D 各拿几份。A 会排队爆满，后面的订单只能退单或延迟；B/C/D 长期接不到单，也学不到菜式。餐厅表面上雇了 4 位主厨，实际只在靠 1 位运转。MoE 的专家崩溃就是这个结构性问题。

---

## 问题定义与边界

先把问题说准。MoE 里的负载均衡问题，不是“所有专家必须完全一样忙”，而是“不能长期出现明显偏载，尤其不能让少数专家持续吸走大部分 token”。

常见的 top-k routing 会先给每个专家打分，再选出分数最高的 $k$ 个专家。这里有一个很强的自增强效应：某个专家一旦早期分数略高，就会收到更多 token；收到更多 token 后，它得到更多梯度更新；更新后表现更好，又更容易被选中。这会形成正反馈。

设一层里共有 $T$ 个 token，$N$ 个专家。记：

- $f_i$：第 $i$ 个专家实际接收到的 token 比例，也就是“真实负载”
- $P_i$：router 对第 $i$ 个专家分配的平均概率，也就是“主观偏好”

常见的辅助负载均衡损失可以写成：

$$
L_{\text{bal}} = \lambda N \sum_{i=1}^{N} f_i P_i
$$

其中 $\lambda$ 是权重，意思是“主任务损失之外，再给不均衡分配一个惩罚”。直观上，如果某些专家既经常被选中，又总被给高概率，那么 $f_i$ 和 $P_i$ 会同时偏大，损失就会上升。

这类问题有几个明确边界。

| 边界项 | 含义 | 太小/太弱会怎样 | 太大/太强会怎样 |
| --- | --- | --- | --- |
| Aux loss 权重 $\lambda$ | 负载均衡惩罚强度 | 无法纠正偏载 | 压制主任务，逼 router 选“次优专家” |
| Capacity factor $\alpha$ | 专家容量放大系数 | token drop 变多 | 显存和通信压力增大 |
| Drop rate | 被丢弃 token 占比 | 表达能力受损 | 降低说明容量更充分，但成本更高 |
| Bias update rate | bias 调整速度 | 纠偏太慢 | 路由抖动，训练不稳 |

“邮包分派”是另一个玩具例子。4 个快递员邮箱容量相同，系统却把大多数包裹发给 A。A 的邮箱塞满后，多余包裹只能退回；B/C/D 长期收不到包裹，也没有机会积累经验。MoE 里被 drop 的 token 就像这些被退回的包裹，它们要么被跳过，要么走降级路径，总之没有按原计划参与该专家的学习。

因此，问题边界可以概括为一句话：MoE 不是只关心“选对专家”，还必须关心“专家是否被公平地训练到”。

---

## 核心机制与推导

MoE 的负载均衡最好从容量开始理解。设每层当前 batch 中共有 $T$ 个 token，要路由给 $N$ 个专家。理想平均负载是 $T/N$。为了给随机波动留余量，工程上会引入 Expert Capacity Factor，专家容量系数，也就是“允许专家比平均值多吃多少 token”：

$$
C = \frac{T}{N} \times \alpha
$$

这里 $C$ 是单个专家允许接收的最大 token 数，$\alpha$ 通常略大于 1。

例如 $T=128$、$N=4$、$\alpha=1.25$，则：

$$
C = \frac{128}{4}\times1.25 = 40
$$

这意味着每个专家最多接 40 个 token。假设路由结果是 A 收到 80 个，B/C/D 各收到 16 个，那么 A 中只有前 40 个会被真正处理，剩下 40 个会被 drop 或改道。此时 drop rate，可理解为“超容量而没按原计划执行的比例”，可写成：

$$
r = \frac{1}{T}\sum_{i=1}^{N}\max(0, T f_i - C)
$$

代入上面的例子，只有 A 超载，超了 $80-40=40$ 个 token，因此：

$$
r = \frac{40}{128}=31.25\%
$$

这已经是很高的比例。高 drop rate 的问题不只是“浪费了一些 token”，而是训练信号被系统性截断。A 只学到了 40 个 token，B/C/D 也没有获得足够丰富的样本，整个路由分布还在朝偏载方向固化。

所以需要软约束来纠偏。除了前面的 $L_{\text{bal}}=\lambda N\sum_i f_iP_i$，也常见用方差类惩罚项：

$$
L_{\text{var}} = \beta \sum_{i=1}^{N}(l_i - \bar l)^2
$$

其中 $l_i$ 是第 $i$ 个专家的真实负载，$\bar l$ 是平均负载。它的意思很直接：谁偏离平均值太远，就罚谁。

但软约束有一个天然缺点。它必须通过反向传播起作用，所以它在“优化目标”层面和主任务竞争。若 $\lambda$ 或 $\beta$ 太大，router 可能为了让负载更均匀，主动把 token 送给并不最合适的专家。

这就是为什么近年的一些工作转向 bias-based routing。bias，也就是“给每个专家附加一个可调常数”，在 softmax 前直接改写路由分数。若原始分数为 $s_i$，偏置为 $b_i$，则实际分数变成：

$$
\tilde s_i = s_i + b_i
$$

当某个专家近期过载时，就减小它的 $b_i$；当某个专家长期欠载时，就增大它的 $b_i$。于是 softmax 前就已经在做“冷热调节”，而不是把压力留到反向传播再修正。

一个简化的 bias 更新原则可以写成：

$$
b_i \leftarrow b_i + \eta (\bar l - l_i)
$$

其中 $\eta$ 是更新步长。若 $l_i > \bar l$，说明第 $i$ 个专家过热，$\bar l-l_i<0$，则 $b_i$ 下降；若 $l_i < \bar l$，说明该专家偏冷，则 $b_i$ 上升。这种方法的核心优势是：它不直接向主损失里再塞一个额外梯度项，而是在前向路由时就把偏载压回去。

完整链条就清楚了：

1. 路由器给专家打分。
2. top-k 选出候选专家。
3. capacity 判断是否超载。
4. aux loss 在反向传播阶段鼓励均衡。
5. bias routing 在前向阶段直接压制热点专家。
6. 日志系统持续记录每层负载、活跃专家数、drop rate 和 bias 变化。

真实工程例子里，这套链条非常关键。以 DeepSeek 一类大规模 MoE 训练为例，问题不是“有没有专家可用”，而是“如何在大规模并行下既保持高吞吐，又不让少数专家吃掉大部分 token”。如果只靠 aux loss，可能会和主任务梯度互相牵制；如果只靠 capacity，热点专家会频繁截流，drop rate 升高。因此更偏前向控制的 bias-based balancing 会更有吸引力，因为它把“负载均衡”从损失设计问题，部分转成了路由控制问题。

---

## 代码实现

下面用一个可以直接运行的 Python 玩具实现，把路由、capacity、drop 和 bias 更新串起来。这个实现不是训练框架代码，但足够说明控制点应该放在哪里。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def top1_route(scores, bias):
    routed = []
    probs = []
    for token_scores in scores:
        adjusted = [s + b for s, b in zip(token_scores, bias)]
        p = softmax(adjusted)
        expert_id = max(range(len(p)), key=lambda i: p[i])
        routed.append(expert_id)
        probs.append(p)
    return routed, probs

def apply_capacity(routes, num_experts, capacity):
    loads = [0] * num_experts
    accepted = []
    dropped = 0
    for expert_id in routes:
        if loads[expert_id] < capacity:
            loads[expert_id] += 1
            accepted.append(expert_id)
        else:
            accepted.append(None)
            dropped += 1
    return loads, accepted, dropped

def update_bias(bias, loads, eta=0.1):
    avg = sum(loads) / len(loads)
    new_bias = []
    for b, l in zip(bias, loads):
        new_bias.append(b + eta * (avg - l))
    return new_bias

# 8 个 token，4 个专家
scores = [
    [4.0, 1.0, 1.0, 1.0],
    [4.2, 1.1, 1.0, 1.0],
    [4.1, 1.0, 1.0, 1.0],
    [4.3, 1.2, 1.0, 1.0],
    [4.0, 1.0, 1.1, 1.0],
    [4.4, 1.0, 1.0, 1.0],
    [4.1, 1.0, 1.0, 1.1],
    [4.2, 1.0, 1.0, 1.0],
]

bias = [0.0, 0.0, 0.0, 0.0]
routes, probs = top1_route(scores, bias)
loads, accepted, dropped = apply_capacity(routes, num_experts=4, capacity=2)

assert sum(loads) + dropped == len(scores)
assert dropped > 0
assert loads[0] == 2  # 热点专家被容量截断

new_bias = update_bias(bias, loads, eta=0.2)

# 专家 0 过载，bias 应下降；其他专家偏冷，bias 应上升
assert new_bias[0] < bias[0]
assert any(new_bias[i] > bias[i] for i in [1, 2, 3])

print("routes =", routes)
print("loads =", loads)
print("dropped =", dropped)
print("new_bias =", [round(x, 3) for x in new_bias])
```

这段代码里有四个关键 hook 点。

第一，`top1_route` 对应 router 前向。真实系统常用 top-2 而不是 top-1，但逻辑一样，都是“先打分，再归一化，再选专家”。

第二，`apply_capacity` 对应 dispatch 之前的容量裁剪。dispatch 指“把 token 按路由结果发送到各专家执行”的过程。真实训练里常见做法包括直接 drop、溢出重路由、或者 padding 到固定容量后再并行计算。

第三，`update_bias` 对应无辅助损失负载均衡。它不改主损失，只改下一批次的路由倾向。

第四，`assert` 明确验证了两个性质：超载会产生 drop，热点专家的 bias 会被压低。

如果改成更接近真实工程的伪代码，大体流程是：

```python
scores = gate(x)                  # [tokens, num_experts]
probs, ids = topk_softmax(scores, k=2)
dispatch_map = build_dispatch(ids, probs)
loads = count_tokens_per_expert(dispatch_map)
clip_by_capacity(dispatch_map, C) # 超过 C 的部分 drop 或 reroute
y = experts_forward(dispatch_map)
aux = load_balance_loss(loads, probs)
bias = update_expert_bias(bias, loads)
loss = task_loss(y, target) + aux
```

这里最容易被初学者忽略的是：capacity 检查不是训练末尾的统计动作，而是前向路径上的强控制点。它直接决定哪些 token 能进入专家、哪些 token 被截断，所以必须和 router 同级看待。

---

## 工程权衡与常见坑

MoE 的很多失败实验，不是理论不对，而是参数组合不合理。

最常见的坑是 capacity factor 太小。$\alpha$ 太小时，drop rate 会迅速上升。drop rate 一旦长期超过 5% 到 10%，通常就已经值得排查，因为这说明大量 token 没有按设计走完专家路径。尤其在长序列、top-k>1、batch 波动大的情况下，$\alpha$ 的安全区间往往不能设得太激进。

第二个坑是 aux loss 权重过大。负载均衡本来是为了让更多专家参与训练，但如果它压过主任务，router 就会为了“看起来均匀”而牺牲“语义上更合适的专家选择”。结果是专家确实都忙起来了，但模型质量反而下降。

第三个坑是只看平均值，不看分层日志。MoE 的问题常常是局部的、分层的。有些层负载很均匀，有些层已经严重塌缩。如果只看全局平均活跃专家数，很容易掩盖掉单层热点专家的持续超载。

建议至少监控这些指标：

| 监控项 | 观察目的 | 异常信号 |
| --- | --- | --- |
| Per-layer expert load | 看每层是否有热点专家 | 某专家长期远高于平均 |
| Drop rate | 看 capacity 是否过紧 | 持续高于 5% 到 10% |
| Active experts | 看是否发生崩溃 | 活跃专家数逐步下降 |
| Router entropy | 看路由分布是否过尖锐 | 熵快速降低 |
| Bias trajectory | 看 bias 是否稳定收敛 | 某些专家 bias 持续单边漂移 |

真实工程里，一个很典型的排障场景是：训练前期 loss 正常下降，但几万 step 后某几层的活跃专家数开始下降，随后 drop rate 抬升，最终验证集指标恶化。这个时候如果只调学习率，往往没有用。真正该查的是该层 router 的负载分布、capacity 设置和 aux/bias 的纠偏强度。

另一个常见坑是把“专门化”和“均衡”理解成完全对立。其实合理的目标不是每个专家做同样的事，而是每个专家都能在自己擅长的区域持续接到足够样本。均衡是为了保证训练机会，专门化是为了提高表达能力。工程上要避免的是“少数专家垄断样本”，不是“专家之间存在差异”。

---

## 替代方案与适用边界

如果不希望 auxiliary loss 直接干扰主损失，bias-based balancing 是很自然的替代方案。它的核心思想可以概括成一句话：谁最近太忙，就给谁降分；谁最近太闲，就给谁加分。

一个简化步骤列表如下：

1. 统计当前 batch 或滑动窗口内每个专家的 load。
2. 计算每个专家相对平均负载的偏差。
3. 用偏差更新对应专家的 bias。
4. 在下一次 softmax 前把 bias 加到原始 router score 上。
5. 重复迭代，直到负载分布稳定。

它适合两类场景。第一类是超大规模训练，主任务梯度已经很复杂，不希望再额外引入强 aux 信号。第二类是已经观察到 aux loss 和主任务存在明显“拔河”，即负载更均匀了，但主指标下降。

假设某个专家连续两个 batch 都几乎没被选中，那么它的 bias 会被逐步抬高。下一个 batch 到来时，即便它的原始分数不占优，叠加 bias 后也更有机会进入 top-k。这个机制的白话解释就是：谁训练得少，就人为多给一点出场机会。

除此之外，还有两类替代路线。

| 方案 | 核心思路 | 优点 | 代价 |
| --- | --- | --- | --- |
| Bias-based balancing | 直接调整路由分数 | 不额外污染主损失梯度 | 需要稳定更新规则 |
| Expert Choice | 由专家反向挑 token | 负载更易控制 | dispatch 更复杂 |
| Dropless MoE | 不丢 token，只改并行策略 | 避免表达损失 | 显存和调度成本更高 |

Expert Choice，专家反选，是“不是 token 找专家，而是专家选 token”。它更容易把每个专家的容量控制在目标范围内，但实现和通信都会更复杂。

Dropless MoE，零丢弃 MoE，目标是尽量不 drop token，而是通过更复杂的调度和并行策略把溢出问题消化掉。它适合对训练稳定性要求高、不愿接受 token 丢失的系统，但实现成本明显高于简单的 capacity clipping。

因此适用边界很明确：

- 如果系统简单、训练规模中等，`aux loss + 合理 capacity` 往往已经够用。
- 如果发现 aux loss 明显干扰主任务，可以考虑 `bias-based balancing`。
- 如果 drop rate 难以接受，且有更强的并行调度能力，可以考虑 `dropless` 或更复杂的 expert choice 路线。
- 如果专家数很多、层数很深，必须做 per-layer 监控，否则很难发现局部崩溃。

---

## 参考资料

- AgentFlow: Mixture of Experts，介绍 MoE 路由、负载均衡和简化伪代码。<https://www.agentflow.academy/blog/mixture-of-experts>
- Aman.ai: Mixture-of-Experts primer，介绍 capacity factor、辅助损失和工程权衡。<https://aman.ai/primers/ai/mixture-of-experts/>
- Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts，介绍基于 bias 的无辅助损失均衡。<https://www.emergentmind.com/papers/2408.15664>
- DeepSeek-V3 相关解读，讨论 bias-based balancing 的工程思路。<https://uplatz.com/blog/the-deepseek-v3-mixture-of-experts-revolution-architectural-breakdown-scaling-dynamics-and-computational-efficiency/>
- NVIDIA Megatron Core / NeMo MoE 文档，介绍 capacity、token drop 与日志配置。<https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html>

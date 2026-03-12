## 核心结论

DeepSeek-V3 的“无辅助损失负载均衡”可以概括成一句话：**不要再用一个额外的 loss 去“劝”路由器均衡，而是在 TopK 选专家之前，直接给每个专家加一个可调偏置 $b_i$，让空闲专家更容易被选中、过载专家更难被选中。**

这里的“辅助损失”是指主训练目标之外，额外加上的平衡项；它的作用是让专家分工更均匀，但副作用是会把主任务梯度一起拉偏。DeepSeek-V3 的做法把两件事拆开：

1. 主任务只优化语言建模本身。
2. 负载均衡只通过偏置更新来做，不把梯度塞回主损失。

这件事的重要性不在“数学更漂亮”，而在于它解决了一个长期工程矛盾：**MoE 要想跑得省算力，就必须路由均衡；但传统均衡方法又常常伤害模型性能。**

| 问题 | 传统方案 | 典型副作用 | V3 方案优势 |
| --- | --- | --- | --- |
| 专家负载不均 | 加 auxiliary loss | 主 loss 与平衡 loss 梯度冲突 | 负载控制与主任务解耦 |
| 某些专家过载 | 提高平衡 loss 权重 $\alpha$ | perplexity 变差，专长被冲淡 | 直接调节 TopK 选择概率 |
| 某些专家长期闲置 | 降低 $\alpha$ 保性能 | 均衡失效，出现“死专家” | 按真实负载动态修正 $b_i$ |

玩具例子可以这样理解：每个专家门口都放一个砝码。砝码越重，这个专家在“候选排序”里越占便宜。某个专家这一步接了太多 token，下步就把它的砝码减一点；某个专家太闲，就加一点。最后真正送进 FFN 的加权，仍然看原始亲和度，不看砝码本身。

---

## 问题定义与边界

MoE，Mixture of Experts，白话说就是“很多前馈网络并排摆着，但每个 token 只激活其中少数几个”。这样做的目标是：**总参数很多，但每次计算只动一小部分参数。**

问题也随之出现。TopK 路由会把每个 token 送给分数最高的 $K$ 个专家。如果某几个专家总被选中，就会出现三类后果：

1. 某些 GPU 持续过载，吞吐下降。
2. 某些专家几乎不接活，训练不到位。
3. 路由器学到“偏科”策略，模型专长变窄。

传统做法是在主损失外再加一个平衡项，例如：
$$
\mathcal{L} = \mathcal{L}_{main} + \alpha \mathcal{L}_{bal}
$$
其中 $\alpha$ 是平衡项权重。问题在于，$\alpha$ 本质上是在问一个很难的问题：**你愿意牺牲多少主任务性能来换更均匀的专家负载？**

“新手示例”可以把它想成两根拉绳：

- 主任务 loss 希望“哪个专家更擅长，就多用哪个专家”。
- 平衡 loss 希望“别老用它，给别的专家也分点活”。

当 $\alpha$ 太小，平衡绳子拉不动，负载还是偏。
当 $\alpha$ 太大，主任务被拉偏，路由器不再优先选择真正合适的专家。

这也是 DeepSeek-V3 要避开辅助 loss 的核心边界：**它不是说辅助 loss 一定错误，而是说在大规模、跨节点、专家很多的场景下，辅助 loss 的副作用越来越难接受。**

| 问题 | 边界条件 | 为何要避开辅助 loss |
| --- | --- | --- |
| 专家分配极不均衡 | 专家数多、跨设备通信重 | 需要强平衡，但强 $\alpha$ 伤主任务 |
| 专家长期不激活 | 稀疏路由、长训练周期 | “死专家”会让有效容量下降 |
| 系统吞吐不稳定 | 多机多卡、All-to-All 昂贵 | 负载波动会直接放大系统抖动 |
| 想保留专家专长 | 不同专家需要自然分工 | 辅助梯度容易过度平均化 |

所以，本文讨论的边界不是“所有 MoE 都必须这样做”，而是：**当你既要大规模训练稳定性，又要尽量不牺牲主模型性能时，无辅助损失方案更值得优先考虑。**

---

## 核心机制与推导

先看标准路由分数。对 token 表示 $u_t$ 和专家向量 $e_i$，V3 使用的亲和度可写成：
$$
s_{i,t} = \sigma(u_t^\top e_i)
$$

这里的“亲和度”可以白话理解为“这个 token 和第 $i$ 个专家有多匹配”。

传统 TopK 是直接在 $\{s_{i,t}\}$ 上挑前 $K$ 个。V3 改成先加偏置：
$$
\tilde{s}_{i,t} = s_{i,t} + b_i
$$

然后只在“选谁进 TopK”这一步使用 $\tilde{s}_{i,t}$。但一旦专家选出来，真正参与混合加权的仍然是原始分数 $s_{i,t}$，不是 $\tilde{s}_{i,t}$。可以写成：
$$
g'_{i,t}=
\begin{cases}
s_{i,t}, & \tilde{s}_{i,t}\in \mathrm{TopK}(\{\tilde{s}_{j,t}\},K) \\
0, & \text{otherwise}
\end{cases}
$$

如果需要归一化，再对被选中的 $g'_{i,t}$ 做 normalize。关键点有两个：

1. **偏置只参与“选谁”**。
2. **偏置不参与“权重多大”**。

这就是“解耦”。因为 $b_i$ 不进入主 loss 的反向传播，所以不会出现“为了均衡而改坏语言建模梯度”的问题。

接着看偏置如何更新。设某一步里，第 $i$ 个专家的实际负载占比为 $f_i$，目标占比为 $P_i$。在均匀目标下：
$$
P_i=\frac{1}{N}
$$
其中 $N$ 是专家数。

最简单的更新规则是符号更新：
$$
b_i \leftarrow
\begin{cases}
b_i - \gamma, & f_i > P_i \\
b_i + \gamma, & f_i < P_i
\end{cases}
$$

$\gamma$ 是更新步长，白话说就是“每次调砝码调多大”。

### 玩具例子

假设只有两个专家，目标负载都是 50%。

| 专家 | 当前负载 $f_i$ | 目标负载 $P_i$ | 偏置更新 |
| --- | --- | --- | --- |
| Expert 1 | 0.6 | 0.5 | $b_1 \leftarrow b_1-\gamma$ |
| Expert 2 | 0.4 | 0.5 | $b_2 \leftarrow b_2+\gamma$ |

如果下一批 token 原始亲和度很接近，比如：

- token A: $s_1=0.52,\ s_2=0.50$
- 当前偏置：$b_1=-0.03,\ b_2=+0.03$

那么用于 TopK 决策的分数变成：

- $\tilde{s}_1=0.49$
- $\tilde{s}_2=0.53$

此时 token A 会被送去 Expert 2。注意，**这不是把 Expert 2 真的变得“更擅长”了，而是给了它更多上场机会。**

### 为什么这比辅助 loss 更直接

辅助 loss 常见形式会惩罚负载偏差，但它的作用路径是“先改梯度，再希望梯度改路由”。而偏置法的路径是“观察负载，再直接改 TopK 排名”。后者更像控制系统，前者更像间接优化。

如果把负载误差记作：
$$
\Delta_i = f_i - P_i
$$
那么偏置法本质上做的是一个离散反馈控制：当 $\Delta_i>0$，降低该专家被选中的先验；当 $\Delta_i<0$，提高它被选中的先验。

这类方法的直觉很像温控器：

- 温度高了，立刻降功率。
- 温度低了，立刻升功率。

而不是再训练一个网络去“学会以后少开一点空调”。

---

## 代码实现

工程上最重要的一点是：**“用于选择”的分数和“用于加权”的分数必须分开。** 这是很多复现失败的根源。

下面给一个可运行的 Python 玩具实现。它不是完整训练代码，但足够验证机制。

```python
import math

def route_with_bias(affinity, bias, k):
    scored = [(i, affinity[i] + bias[i]) for i in range(len(affinity))]
    topk = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    chosen = [i for i, _ in topk]

    raw = [affinity[i] for i in chosen]
    total = sum(raw)
    weights = [x / total for x in raw]
    return chosen, weights

def update_bias(bias, loads, target, gamma):
    new_bias = bias[:]
    for i, load in enumerate(loads):
        if load > target[i]:
            new_bias[i] -= gamma
        elif load < target[i]:
            new_bias[i] += gamma
    return new_bias

# 两个专家，选 1 个
affinity = [0.52, 0.50]
bias = [0.0, 0.0]

chosen, weights = route_with_bias(affinity, bias, k=1)
assert chosen == [0]
assert abs(weights[0] - 1.0) < 1e-9

# 第一个专家过载，第二个欠载，更新偏置
loads = [0.6, 0.4]
target = [0.5, 0.5]
bias = update_bias(bias, loads, target, gamma=0.03)

chosen2, weights2 = route_with_bias(affinity, bias, k=1)
assert bias == [-0.03, 0.03]
assert chosen2 == [1]
assert abs(weights2[0] - 1.0) < 1e-9

# 注意：被选中后，加权仍然用原始 affinity，而不是 affinity + bias
assert affinity[1] == 0.50
```

上面这段代码体现了 V3 思路的两个核心约束：

1. `affinity + bias` 只用于 `topk` 排序。
2. `weights` 只由原始 `affinity` 归一化得到。

一个更接近训练循环的伪代码如下：

```python
for batch in dataloader:
    affinity = sigmoid(token_repr @ expert_repr.T)   # [T, E]

    route_score = affinity + bias                    # 只用于选路由
    topk_index = topk(route_score, k=K)

    gate = normalize(mask_select(affinity, topk_index))  # 只用原始 affinity
    output = moe_combine(gate, experts)

    loss = lm_loss(output, target)
    loss.backward()
    optimizer.step()

    expert_load = count_tokens(topk_index) / (T * K)
    bias = bias_update(bias, expert_load, target_load, gamma)
```

变量作用可以用一张表记住：

| 变量 | 含义 | 作用 |
| --- | --- | --- |
| $s$ | 原始亲和度 | 表示 token 与专家匹配程度 |
| $b$ | 专家偏置 | 只影响 TopK 选择 |
| $\gamma$ | 偏置更新速率 | 控制调节快慢 |
| $P$ | 目标负载 | 常见是均匀分布 |
| $f$ | 实际负载 | 由本批次或 EWMA 统计得到 |

### 真实工程例子

在 DeepSeek-V3 公开技术报告描述的训练配置里，模型是 **671B 总参数、每 token 激活 37B 参数、256 个 routed experts、训练总成本 2.788M H800 GPU hours，且整个训练过程中没有发生不可恢复的 loss spike，也没有回滚**。这类规模下，负载不均已经不是“某个专家学得差一点”，而是会直接影响：

- All-to-All 通信峰值
- 单卡尾延迟
- 节点间负载漂移
- token 是否需要丢弃或降级

因此偏置法的价值不仅是论文里的一点分数提升，而是**它更适合被接到一个大规模分布式训练系统上长期稳定跑。**

---

## 工程权衡与常见坑

无辅助损失不等于“没有调参”。它只是把难题从“怎么平衡两个 loss”改成了“怎么稳定做反馈控制”。

### 常见坑 1：把 bias 也拿去做最终 gate

这是最常见错误。这样会改变专家输出的混合权重，等于又把负载均衡干预塞回主路径。短期看可能更均衡，长期 often 会伤害专长分化。

### 常见坑 2：$\gamma$ 太大，系统来回振荡

如果步长太大，某个专家这一步过载就被猛砍，下一步又变成欠载，再下一步又猛加，最终形成抖动。工程里通常会配合 EWMA，Exponential Weighted Moving Average，白话说就是“用滑动平均平滑最近几步的负载”，避免被单个 batch 噪声误导。

### 常见坑 3：统计口径不一致

有的实现统计“被 TopK 选中的次数”，有的统计“实际 dispatch 到专家的 token 数”，还有的按 sequence 聚合。口径一变，$f_i$ 的含义就变了，$\gamma$ 也要重调。

### 常见坑 4：残留 auxiliary loss

有些复现会保留一个不小的 auxiliary loss，再叠加 bias 更新。这样做的结果常常不是“两边都好”，而是两套控制器互相抢方向。大 $\alpha$ 时尤其明显：路由分数会被人为压平，perplexity 容易上升。

| 问题 | 建议处理 |
| --- | --- |
| 专家负载周期性振荡 | 减小 $\gamma$，对 $f_i$ 做 EWMA |
| 长尾专家始终激活不足 | 检查 bias 是否只用于 TopK，必要时放慢衰减 |
| 训练后期 perplexity 反弹 | 排查是否把 bias 参与了 gate 权重或残留较大 auxiliary loss |
| 多机负载看似均衡但单机仍爆 | 分层统计：层内专家负载、节点负载、GPU 负载分开看 |

如果把公开工程信息抽成要点，DeepSeek-V3 这类方案的系统含义主要有三点：

- `256` 个 routed experts / 层，说明路由空间非常大，负载控制必须稳定。
- 每 token 激活 `8` 个 routed experts，说明局部失衡会被迅速放大到通信层。
- 公开报告强调训练稳定、无回滚，说明这种控制方式至少在超大规模上是可操作的。

---

## 替代方案与适用边界

无辅助损失不是唯一答案，但它非常适合“大模型、强并行、强稳定性需求”的场景。

先给一个简化流程图：

```text
token -> 计算 affinity s
      -> 加 bias 得到 s + b
      -> TopK 选专家
      -> 用原始 s 做归一化加权
      -> 统计实际负载 f
      -> 每步或每若干步更新 b
```

### 偏置法 vs 辅助 loss

| 方案 | 适用场景 | 优势 | 限制 |
| --- | --- | --- | --- |
| 动态偏置 `b_i` | 大规模 MoE、跨节点训练 | 不引入干扰梯度，控制路径直接 | 需要稳定的负载统计与同步 |
| auxiliary loss | 小规模 MoE、实现简单优先 | 容易并入现有训练图 | 需调 $\alpha$，会与主任务竞争 |
| 混合策略 | 极端负载波动场景 | 可兼顾局部与全局控制 | 系统更复杂，调参成本高 |

### 玩具边界例子

如果硬件限制导致你不能每步都同步 bias，那么可以每隔若干步批量更新一次：

- 每步只累计专家负载。
- 每 `M` 步统一计算一次 $f_i$。
- 再按 $b_i \leftarrow b_i \pm \gamma$ 更新。

这样响应会变慢，但仍然保留“主损失与负载均衡解耦”的优点。

### 什么时候仍然可以用辅助 loss

如果你的模型很小、专家数不多、训练只在单机内完成，辅助 loss 仍然可用。因为这时通信压力和负载失衡的系统成本都没那么高，你更在意实现简单。但前提是：

1. 监控 perplexity 是否回退。
2. 监控是否出现专家长期闲置。
3. 把 $\alpha$ 当成核心超参，而不是默认值。

更直接地说：

- 小规模实验：auxiliary loss 可以先用。
- 大规模生产训练：优先考虑 bias 控制。
- 需要极致稳定性：bias + 平滑统计通常更合适。

---

## 参考资料

1. **DeepSeek-AI, DeepSeek-V3 Technical Report**  
   原始技术报告，给出 V3 的总体架构、671B/37B 规模、训练稳定性，以及“auxiliary-loss-free strategy for load balancing”的官方表述。  
   链接：https://arxiv.org/abs/2412.19437

2. **deepseek-ai/DeepSeek-V3 GitHub Repository**  
   官方仓库 README 汇总了模型规模、训练成本、评测结果和论文入口，适合快速确认公开配置与实验结论。  
   链接：https://github.com/deepseek-ai/DeepSeek-V3

3. **Lean Wang et al., Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts**  
   这篇论文专门讨论“无辅助损失负载均衡”本身，核心是专家偏置、动态反馈更新，以及相对 auxiliary loss 的性能与负载优势。  
   链接：https://arxiv.org/abs/2408.15664

4. **deepseek-ai/DeepEP**  
   官方开源的 MoE 通信库，说明了 V3 相关路由和 expert parallel 通信优化在系统层的落地背景。  
   链接：https://github.com/deepseek-ai/DeepEP

5. **deepseek-ai/EPLB**  
   官方开源的专家并行负载均衡实现，展示了大规模专家复制与放置策略，帮助理解“路由均衡”和“设备均衡”不是同一层问题。  
   链接：https://github.com/deepseek-ai/EPLB

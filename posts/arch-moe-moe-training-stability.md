## 核心结论

MoE，Mixture of Experts，直译是“专家混合”，本质上是在很多子网络里按输入内容只激活少数几个，从而用更低计算量换更大模型容量。MoE 训练不稳定，核心通常不在专家本身，而在“路由器”这个负责分配 token 到专家的小网络。

训练失稳主要来自三个来源：

1. 路由器梯度稀疏甚至消失。路由器先输出 logits，再经过 softmax 变成概率，最后用 top-k 只选前几个专家。top-k 可以理解成“只保留得票最高的几个候选人”，它本质上依赖 argmax 排序，严格来说不可微，反向传播时无法自然地给所有专家分配梯度。结果是被选中的专家继续变强，没被选中的专家长期没有更新机会。

2. 专家初始化差异会放大早期偏置。初始化就是训练开始前参数的随机起点。哪怕专家结构完全相同，只要初始权重略有差异，路由器就可能在前几步里偏爱某一个专家，随后 top-k 机制把这种偏爱放大，形成“赢者通吃”。

3. 路由器 logits 会自然漂移到过大的数值范围。logits 就是 softmax 前的原始分数。它们一旦绝对值越来越大，softmax 就会变得极端，接近 one-hot 分布，进而带来 loss spike、梯度爆炸、数值溢出，严重时出现 NaN。ST-MoE 提出的 router z-loss 用来专门压制这个问题：

$$
\mathcal{L}_z=\frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_j e^{z_{i,j}}\right)^2
$$

这个项直接惩罚 log-sum-exp 过大。直观上看，logits 越极端，惩罚越强，因此会形成一个“往回拉”的恢复力。

一个适合新手的比喻是“选举”。如果某位候选人一开始偶然多拿了几票，选举委员会之后每轮都只给得票最高的人上台发言，那其他候选人永远没有机会证明自己。top-k 路由就有这个问题。z-loss 相当于对“得票差距过大”额外收费，让路由器不要太快把概率压到极端，从而保留探索空间。

工程上，load balancing loss 解决“负载均不均”，z-loss 解决“logits 会不会失控”，STE 或 dense backprop 解决“梯度能不能穿过 top-k”。三者不是替代关系，而是分工互补。

---

## 问题定义与边界

先明确本文讨论的“稳定性”边界。这里不讨论分布式通信瓶颈、不讨论专家并行的带宽调度，也不讨论下游任务泛化差异。本文只讨论训练阶段由路由器引起的三类失稳：

| 故障模式 | 触发条件 | 典型症状 | 规避措施 |
|------|------|------|------|
| Expert collapse | top-k 只给少数专家梯度 | 部分专家长期不更新，负载集中 | STE、dense backprop、初始化控制 |
| Logit overflow | logits 持续变大 | loss spike、梯度爆炸、NaN | z-loss、logit 监控、较小初始化 |
| Early routing bias | 专家初始参数略有差异 | 某个专家早期占用率异常高 | 相同初始化 + 小扰动、warmup 噪声 |
| Fake balance | 只看 aux loss，不看 logits | 负载看似均匀但训练仍不稳 | 同时监控 z-loss 与 expert load |

MoE 的标准流程是：给定 token 表示 $h$，路由器计算

$$
z = W_r h
$$

其中 $W_r$ 是路由权重矩阵。然后对 $z$ 做 softmax：

$$
p_j=\frac{e^{z_j}}{\sum_t e^{z_t}}
$$

最后取 top-k 个专家参与前向。这里的问题在于，softmax 是可微的，但 top-k 选择不是平滑操作。也就是说，概率分布本来还能传递连续梯度，一旦进入 top-k，梯度路径就被切成稀疏离散的选择。

玩具例子可以直接看三位专家。假设某个 token 的初始 logits 是：

$$
[2.0,\ 0.5,\ 0.3]
$$

如果使用 top-1，第一位专家被选中。此时只有它真的参与前向和主要梯度更新。若下一步参数更新把 logits 拉成：

$$
[20,\ 5,\ 3]
$$

softmax 几乎就等于 $[1,0,0]$。这时第二、第三个专家不仅没被选中，还会进一步失去被选中的可能，collapse 就发生了。

要注意，常见的 load balancing loss 只是在批次统计上鼓励“专家使用率更平均”，它并不直接约束 logits 数值大小。因此它能改善“长期分配偏斜”，但不能单独解决“某一次 logit 爆炸把训练打崩”的问题。这就是为什么很多实现里 auxiliary loss 明明开着，训练还是会突然 spike。

---

## 核心机制与推导

先看路由器的数学结构。对第 $i$ 个 token，路由器输出第 $j$ 个专家的分数 $z_{i,j}$，再得到概率：

$$
p_{i,j}=\text{softmax}(z_{i,j})
$$

定义 top-k mask 为 $m_{i,j}\in\{0,1\}$，只有排名前 $k$ 的专家 mask 为 1。最终参与前向的权重通常是：

$$
\tilde{p}_{i,j}=\frac{m_{i,j}p_{i,j}}{\sum_t m_{i,t}p_{i,t}}
$$

问题出在 $m_{i,j}$ 的生成过程。排序和截断是离散操作，严格求导时不可微。因此实际训练里通常要么依赖近似，要么接受非常稀疏的梯度。

### 1. 路由器梯度为什么会“断”

straight-through estimator，简称 STE，可以理解成“前向按离散规则走，反向假装它是连续的”。常见近似写法是：

$$
\frac{\partial m_{i,j}}{\partial z_{i,j}} \approx 1
$$

它不是真实导数，而是人为指定一个近似梯度，让反向传播不要彻底断掉。这样做的目标不是数学上精确，而是工程上可训练。

对白话解释：前向时你仍然只让 top-k 专家工作，计算量不变；反向时你告诉优化器，“不要把没选中的专家完全当空气”。这能缓解赢者通吃。

### 2. 初始化差异为什么会被放大

假设两个专家结构相同，但专家 A 初始输出方差略大，路由器在最早几个 batch 上更容易给它高分。由于 top-k 只训练被选中的专家，A 会更快适配数据，分数继续升高，形成正反馈闭环：

1. 初始随机差异
2. 路由偏向某专家
3. 该专家获得更多梯度
4. 能力继续增强
5. 路由更偏向它

因此很多实践会让专家使用相同初始化策略，再加极小随机扰动。目标不是让专家完全一样，而是避免“谁先赢纯靠随机”。

### 3. z-loss 为什么能抑制极端 softmax

定义

$$
\text{LSE}(z)=\log\sum_j e^{z_j}
$$

那么 router z-loss 就是

$$
\mathcal{L}_z=\frac{1}{B}\sum_{i=1}^{B}\text{LSE}(z_i)^2
$$

LSE，log-sum-exp，可以理解成“对一组 logits 的平滑最大值”。当某一维非常大时，LSE 会接近那个最大值。于是如果整体 logits 规模持续变大，$\mathcal{L}_z$ 会近似按平方增长。

更关键的是它的梯度：

$$
\frac{\partial \mathcal{L}_z}{\partial z_{i,j}}
=
\frac{2}{B}\,\text{LSE}(z_i)\cdot \text{softmax}(z_{i,j})
$$

这说明两件事：

1. 当 LSE 很大时，梯度会成比例变强。
2. 梯度分布本身按 softmax 分配，对较大 logit 的约束更明显。

所以 z-loss 不是在“要求平均分配专家”，而是在“要求不要把分数做得过于极端”。这是数值稳定约束，不是负载均衡约束。

继续用玩具例子。若 logits 为 $[2.0, 0.5, 0.3]$，则

$$
\text{LSE}\approx \log(e^2+e^{0.5}+e^{0.3}) \approx 2.34
$$

平方后约为 $5.48$。若 logits 变成 $[20,5,3]$，则

$$
\text{LSE}\approx 20
$$

平方后约为 $400$。这意味着当路由器开始冲向极端区间时，z-loss 会快速变强，把参数往可控区域拉回。

总损失一般写成：

$$
\mathcal{L}_{total}=\mathcal{L}_{task}+\alpha_{aux}\mathcal{L}_{aux}+\alpha_z\mathcal{L}_z
$$

其中 $\mathcal{L}_{task}$ 是主任务损失，$\mathcal{L}_{aux}$ 是负载均衡类辅助项，$\mathcal{L}_z$ 是路由器数值稳定项。三者分别回答三个不同问题：学得对不对、用得均不均、分数会不会炸。

真实工程例子是大规模 sparse Transformer 预训练。GShard 及其后续 ST-MoE 经验显示，在 top-1 或 top-2 路由下，超大模型常出现训练曲线尖峰；加入 router z-loss 后，loss spike 频率明显下降，训练能在同一套超参数下持续跑过更长 token 数。这类结果说明 z-loss 的主要价值不是提升最终上限，而是让训练过程先别崩。

---

## 代码实现

下面给一个可以直接运行的 Python 玩具实现，用来演示 z-loss 如何随 logits 增大而快速变强。这里不用 PyTorch，直接用标准库完成，便于理解数学本身。

```python
import math

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def softmax(xs):
    lse = logsumexp(xs)
    return [math.exp(x - lse) for x in xs]

def z_loss(batch_logits):
    values = [logsumexp(row) ** 2 for row in batch_logits]
    return sum(values) / len(values)

small = [[2.0, 0.5, 0.3]]
large = [[20.0, 5.0, 3.0]]

small_probs = softmax(small[0])
large_probs = softmax(large[0])

assert abs(sum(small_probs) - 1.0) < 1e-9
assert abs(sum(large_probs) - 1.0) < 1e-9
assert z_loss(large) > z_loss(small) * 50

print("small_probs =", small_probs)
print("large_probs =", large_probs)
print("small_z_loss =", z_loss(small))
print("large_z_loss =", z_loss(large))
```

如果你把这段代码跑起来，会看到第二组 logits 的 softmax 几乎压成 one-hot，而 z-loss 暴涨很多倍。这正是它在训练中起作用的基础。

在真实实现里，通常直接写成 PyTorch 版本，并在 top-k mask 之前计算 z-loss，原因是此时梯度还能覆盖全部专家 logits，而不是只覆盖被选中的少数几个：

```python
import torch

def compute_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    # router_logits: [batch, num_experts]
    log_sum_exp = torch.logsumexp(router_logits, dim=-1)
    return torch.mean(log_sum_exp ** 2)

def moe_router_loss(task_loss, router_logits, aux_loss, z_loss_coeff=1e-3, aux_coeff=1e-2):
    z = compute_z_loss(router_logits)
    total = task_loss + aux_coeff * aux_loss + z_loss_coeff * z
    return total, z
```

几个实现细节要明确：

1. z-loss 要在 top-k 截断前算。否则未激活专家拿不到这一项的梯度。
2. 路由器权重初始化要小。常见思路是让 $W_r$ 的初始标准差较小，比如 `std=0.01` 量级，避免训练刚开始 logits 就过大。
3. 专家参数最好采用一致初始化策略，再叠加很小扰动，避免一开始就出现明显“亮专家”。
4. 监控不能只看总 loss。至少要同时看 `expert load`、`router logits` 的范围、`z-loss` 曲线。

一个真实工程中的简化流程如下：

1. 前向得到 `router_logits`
2. 计算 softmax 概率
3. 取 top-k 专家并重归一化
4. 执行稀疏专家前向
5. 计算 task loss + load balancing loss + z-loss
6. 统一反向传播

这类实现改动很小，但稳定性收益通常很高，因此 z-loss 在工程上是低成本高收益的手段。

---

## 工程权衡与常见坑

MoE 稳定训练不是“加一个 z-loss 就结束”。真正的问题在于多个机制会互相影响。

| 坑 | 监控指标 | 调整建议 |
|----|----------|----------|
| z-loss 过低 | logit 最大值持续上升，loss 间歇尖峰 | 提高 $\alpha_z$，或对 router 做更长 warmup |
| z-loss 过高 | 专家分工不明显，路由分布过平 | 降低 $\alpha_z$，避免抑制 specialization |
| 只开 aux loss | 负载均匀但仍 spike | 补 z-loss，单独监控 LSE 或 max logit |
| 专家初始化偏差 | 某个 expert load 长期 > 80% | 重设初始化，增加早期噪声 |
| 只用硬 top-k | 非激活专家长期零梯度 | 用 STE 或 dense backprop 近似 |
| 容量设置过小 | token overflow/drop 增多 | 提高 capacity factor，避免路由拥塞 |

### 常见坑 1：把负载均衡误当成稳定性本身

很多初学者看到专家使用率更均匀，就以为模型稳定了。其实不是。负载均衡只说明“大家都被分到了一些 token”，不说明“logits 没有爆炸”。一个模型完全可能一边保持大致均匀的专家负载，一边在少数 batch 上出现极端 logits，最终触发 loss spike。

### 常见坑 2：z-loss 系数不是越大越好

$\alpha_z$ 太小，约束形同虚设；太大，则路由器不敢拉开差异，专家很难形成专门化分工。经验上它通常是一个较小系数，常在 $10^{-4}$ 到 $10^{-2}$ 区间试验，但真正该看的是监控曲线，而不是死记某个数字。

### 常见坑 3：早期几百步最危险

训练前期路由器和专家都还没有形成稳定分工，这时最容易出现随机偏置被放大。工程上常见处理包括：

1. 更小的 router 初始化
2. router 学习率 warmup
3. 轻微噪声注入
4. 统一专家初始化

真实工程例子：某团队训练 8 个专家、top-1 路由的 MoE。前几十步监控发现 Expert A 占到 90% 以上，其他专家几乎空闲，随后总 loss 出现尖峰并伴随 NaN。排查后发现他们只有 load balancing loss，没有 z-loss，且 router 初始化偏大。补上 $\alpha_z=10^{-3}$、缩小 router 初始化标准差后，expert load 明显回到可控范围，loss spike 不再频繁出现。这个例子说明，collapse 和数值失稳往往是一起出现的，不应拆开看。

---

## 替代方案与适用边界

z-loss 不是唯一方案，也不是任何情况下都足够。

| 方法 | 优点 | 代价 | 适用场景 |
|------|------|------|----------|
| top-k + z-loss | 改动小，直接抑制 logit 漂移 | 仍需调 $\alpha_z$，梯度仍偏稀疏 | 中小规模 MoE、已有稀疏实现 |
| top-k + STE | 保持稀疏前向，同时缓解梯度断裂 | 近似梯度有偏 | 希望低改造成本缓解 collapse |
| dense backprop / Default MoE | 非激活专家也能收到反馈，稳定性更强 | 有额外计算与实现复杂度 | 超大规模预训练 |
| 仅 load balancing loss | 易实现，直接改善负载不均 | 不能控制 logit 爆炸 | 作为基础配置，不应单独依赖 |
| 噪声路由 / warmup | 对早期偏置有效 | 需要额外调参 | 训练初期容易 collapse 的模型 |

可以这样理解两条路线：

1. z-loss 管的是“入口坡度”。也就是 softmax 前的分数不要太夸张。
2. dense backprop 管的是“反馈覆盖面”。也就是即使没被选中，也别完全听不到训练信号。

如果用一个更直观的道路比喻，z-loss 是控制每条道路入口的坡度和限速，避免一条路突然把所有车都吸走；Default MoE 则是在没有车经过的道路上也保留维护反馈，不让它长期失修。

适用边界也要说清楚。若模型规模不大、专家数不多、训练预算有限，`top-k + load balancing + z-loss` 往往已经够用。若进入超大规模预训练，专家数很多、并且发现未激活专家长期学不到东西，仅靠 z-loss 可能不够，需要引入更密集的梯度近似，比如 STE 或 dense backprop。

因此更准确的结论不是“z-loss 最好”，而是：

- 你要解决数值极端和 loss spike，优先加 z-loss。
- 你要解决未激活专家长期无梯度，优先考虑 STE 或 dense backprop。
- 你要解决专家使用率不均，继续保留 load balancing loss。
- 你要减少早期随机偏置，控制初始化和 warmup。

这四类手段分别处理不同故障，不应混为一个开关。

---

## 参考资料

- Fedus, Zoph, Shazeer 等，*ST-MoE: Designing Stable and Transferable Sparse Expert Models*, arXiv, 2022.
- Michael Brenndoerfer, *Router Z-Loss: Numerical Stability for MoE Training*.
- Panda 等，*Dense Backpropagation Improves Training for Sparse Mixture-of-Experts*.
- GShard 与后续稀疏 Transformer 路由稳定性相关论文与工程报告。
- Mixture of Experts 相关工程实践资料，涉及初始化、辅助损失、容量因子与路由监控。

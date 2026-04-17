## 核心结论

ST-MoE 对 MoE 训练不稳定的判断很直接：真正容易把训练拖垮的，不只是“专家分配不均”，而是 router logits 持续变大，最后触发数值问题。logits 就是 softmax 前的原始打分，数值越大，路由器越“自信”。自信本身不是问题，问题是这种自信会无限膨胀。

Router Z-Loss 的做法也很直接：不去强行改 top-k 路由规则，而是在训练目标里额外惩罚 logits 的整体尺度。核心公式是

$$
L_z=\frac{1}{B}\sum_{i=1}^B\left(\log\sum_{j=1}^N e^{x_{i,j}}\right)^2
$$

总损失写成

$$
L_{tot}=L_{CE}+c_B L_B+c_z L_z
$$

其中 $L_{CE}$ 是任务主损失，$L_B$ 是负载均衡损失，$L_z$ 是 z-loss，$c_z$ 是它的权重。ST-MoE 的结论是：这类“只约束绝对值、不破坏相对排序”的软约束，可以显著降低训练崩溃率，同时不明显伤害模型质量。

一个最重要的直觉是：z-loss 不要求所有专家得分接近，它只要求“别把所有分数一起推到极大值”。所以它保留了“谁更适合当前 token”的区分能力，只是阻止 router 变成数值上危险的极端分类器。

---

## 问题定义与边界

MoE 的路由器本质上是一个分类器。给定某个 token 表示，router 为每个专家输出一个 logit，再通过 softmax 变成概率，然后选 top-k 专家执行计算。这里的边界要说清楚：z-loss 处理的是“路由分数的数值尺度失控”，不是所有 MoE 问题的万能解。

为什么 logits 爆炸危险？因为 softmax 要计算指数：

$$
p_j=\frac{e^{x_j}}{\sum_{k=1}^N e^{x_k}}
$$

当某个 $x_j$ 很大时，$e^{x_j}$ 可能在低精度或不稳定实现下直接溢出。对初学者可以把 overflow 理解为“数大到当前数据类型装不下了”。以 float32 为例，$\exp(88)$ 左右就已经接近上界；如果某个 logit 超过这个量级，数值计算就可能出现 `inf`，随后 softmax、loss、梯度一路污染成 `NaN`。

常见故障与根因可以对应起来：

| 现象 | 直接表现 | 更深一层的数值根因 | z-loss 是否直接针对 |
|---|---|---|---|
| loss spike | loss 突然尖峰 | logit 过大导致 softmax 极端化，梯度剧烈波动 | 是 |
| gradient explosion | 梯度范数突然放大 | 概率过度尖锐，链式求导后局部梯度异常集中 | 是 |
| expert collapse | 少数专家长期被选中 | router 过度自信，top-k 基本固化 | 部分针对 |
| poor generalization | 验证集效果恶化 | 路由过早僵化，专家无法形成健康分工 | 间接改善 |

这里还要区分两个问题。第一，专家负载不均衡，是“流量分配问题”；第二，router logits 爆炸，是“数值稳定性问题”。两者经常一起出现，但不是同一个因果层级。负载均衡损失 $L_B$ 主要解决前者，z-loss 主要解决后者。

玩具例子最能说明这个区别。看两组 logits：

- $[2, 0.5, 0.3, 0.1]$
- $[20, 5, 3, 1]$

它们的 top-2 排序都一样，都是第 1 个和第 2 个专家更可能被选中。也就是说，“该选谁”的相对判断没有根本变化。但第二组的绝对值大得多，$\log\sum e^{x_j}$ 也会大很多，z-loss 就会把第二组惩罚得更重。这个机制不是在改路由决策，而是在压制危险的数值尺度。

---

## 核心机制与推导

理解 z-loss，关键是理解 log-sum-exp。log-sum-exp 可以看作“平滑最大值”，意思是它和最大 logit 很接近，但又比直接取最大值更平滑、可导：

$$
\mathrm{LSE}(x)=\log\sum_{j=1}^N e^{x_j}
$$

当某个分量远大于其他分量时，有近似关系：

$$
\mathrm{LSE}(x)\approx \max_j x_j
$$

所以 z-loss 本质上是在惩罚“最大 logit 的平滑版本”。它为什么平方？因为平方会对大值给出更强惩罚，小幅偏大时影响温和，特别大时拉回力度明显。这很像一个二次弹簧：偏得越远，拉回越强。

对单个样本 $i$，记

$$
z_i=\log\sum_{j=1}^N e^{x_{i,j}}
$$

则单样本 z-loss 是 $z_i^2$。对某个 logit 的梯度为

$$
\frac{\partial z_i^2}{\partial x_{i,j}}
=2 z_i \frac{\partial z_i}{\partial x_{i,j}}
=2 z_i \cdot \frac{e^{x_{i,j}}}{\sum_k e^{x_{i,k}}}
=2 z_i \cdot \mathrm{softmax}(x_i)_j
$$

这条式子有三个直接结论。

第一，梯度方向平滑。它没有硬阈值，不会像“超过某值就截断”那样突然改变优化形态，所以训练更稳。

第二，梯度重点落在高概率专家上。因为 $\mathrm{softmax}(x_i)_j$ 越大，说明该专家当前得分越高、越可能是导致过度自信的来源，于是它收到的回拉也越强。

第三，z-loss 不破坏排序。假设所有 logits 同时加上一个常数 $\Delta$，softmax 概率并不会变化，因为分子分母都会乘上 $e^\Delta$。但 $\mathrm{LSE}(x)$ 会大约增加 $\Delta$，z-loss 会立刻变大。因此 z-loss 正好补上了 softmax 对“整体平移不敏感”的缺口。

这也是它比直接给概率加正则更合适的原因。概率已经归一化，只看概率看不出“整体尺度是否正在危险上升”；而 LSE 正好能看到这一点。

从训练动力学角度，可以把总目标理解为三股力的合成：

- $L_{CE}$ 要求模型把任务做好。
- $L_B$ 要求不同专家都能收到足够数据，避免系统吞吐和容量失衡。
- $L_z$ 要求 router 别把分数推到数值危险区。

三者一起优化时，router 学到的是“既能分辨 token 属于哪个专家，又不会因为过度自信把训练炸掉”的路由器。

真实工程例子是大规模 T5/Transformer 风格的 MoE 训练。模型一旦进入长时间训练、混合精度、上百亿参数、多个 seed 同时跑的场景，单次数值事故不是“偶尔多一步损失变大”，而是整个作业直接废掉。ST-MoE 的贡献不只是提出一个新公式，而是把问题定位到 router logits 的尺度控制，并给出一个足够轻量、可并入现有训练目标的解法。

---

## 代码实现

先看一个最小实现。下面代码不依赖深度学习框架，只用 Python 标准库复现 z-loss 的核心数学，并验证“尺度放大后，z-loss 会显著增大”。

```python
import math

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def softmax(xs):
    lse = logsumexp(xs)
    return [math.exp(x - lse) for x in xs]

def z_loss(batch_logits):
    # batch_logits: [batch, num_experts]
    vals = []
    for row in batch_logits:
        lse = logsumexp(row)
        vals.append(lse ** 2)
    return sum(vals) / len(vals)

small = [2.0, 0.5, 0.3, 0.1]
large = [20.0, 5.0, 3.0, 1.0]

small_probs = softmax(small)
large_probs = softmax(large)

# 两组 logits 的 top-1 / top-2 排序一致
assert sorted(range(len(small_probs)), key=lambda i: small_probs[i], reverse=True)[:2] == [0, 1]
assert sorted(range(len(large_probs)), key=lambda i: large_probs[i], reverse=True)[:2] == [0, 1]

small_z = z_loss([small])
large_z = z_loss([large])

# 尺度变大后，z-loss 显著增大
assert large_z > small_z * 20

print("small_probs =", small_probs)
print("large_probs =", large_probs)
print("small_z =", small_z)
print("large_z =", large_z)
```

这个版本已经说明原理了。工程里如果用 PyTorch，核心函数通常就是下面这样：

```python
import torch

def compute_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    # router_logits: [num_tokens, num_experts]
    lse = torch.logsumexp(router_logits, dim=-1)
    return torch.mean(lse ** 2)
```

如果把它放到 router 中，逻辑通常分三步：

1. 先产出 `router_logits`
2. 在 softmax 之前计算 `z_loss`
3. 再做 softmax、top-k、dispatch

伪代码如下：

```python
class RouterWithZLoss(torch.nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, hidden_states):
        router_logits = self.linear(hidden_states)          # [tokens, experts]
        z_loss = compute_z_loss(router_logits)

        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)

        return {
            "router_logits": router_logits,
            "router_probs": router_probs,
            "topk_probs": topk_probs,
            "topk_idx": topk_idx,
            "z_loss": z_loss,
        }
```

在完整 MoE 层里，还会把负载均衡损失一起加上：

```python
def total_loss(task_loss, load_balance_loss, z_loss, c_b=0.01, c_z=0.001):
    return task_loss + c_b * load_balance_loss + c_z * z_loss
```

这里有一个初学者容易忽略的点：z-loss 应该对 `router_logits` 计算，而不是对 softmax 后的概率计算。因为它要约束的是“指数化之前的尺度”，这才是数值风险源头。

---

## 工程权衡与常见坑

z-loss 的优点是代价低，但不代表可以“加上就不用管”。最核心的工程权衡是系数 $c_z$。

| 场景 | 常见 `c_z` 范围 | 太小的风险 | 太大的风险 |
|---|---|---|---|
| 小规模实验 | $10^{-4}$ 到 $10^{-3}$ | 看不到稳定性改善 | 路由过软，专家分工不明显 |
| 中大规模训练 | $10^{-3}$ 左右 | logits 仍持续上升 | token 被过度平均分配 |
| 极端大模型或低精度 | $10^{-3}$ 到 $10^{-2}$ | 长训练后仍可能炸 | 主任务收敛变慢 |

经验上可以从 `1e-4` 或 `1e-3` 起步，然后观察三类日志：

- `router_lse_mean` 或 `logsumexp_mean`
- `router_max_logit`
- `task_loss / load_balance_loss / z_loss`

如果 `LSE` 长期持续抬升，或者训练中后期开始出现 loss 震荡，就说明 z-loss 可能太弱。如果 routing entropy 明显下降为近乎 0，或者所有专家利用率越来越平均但效果没提升，就说明 z-loss 可能太强了。entropy 就是概率分布的不确定性，高 entropy 表示更平均，低 entropy 表示更尖锐。

常见坑主要有四个。

第一，只监控专家负载，不监控 logits。这样会漏掉真正的数值先兆。很多实验在“专家分配看着还行”时，router logits 已经开始向危险区漂移。

第二，把 z-loss 当成 load-balance loss 的替代品。它不是。z-loss 负责稳住数值，load-balance 负责稳住流量，两者最好同时有。

第三，在 bf16、fp16 场景下以为“框架的稳定 softmax 已经足够”。稳定 softmax 能减少局部溢出，但不能阻止 logits 长期漂移。z-loss 解决的是训练动力学问题，不只是一个算子实现问题。

第四，top-k 之后再算正则。这样信息已经丢了。z-loss 需要看到所有专家 logits，才能感知整个分布的尺度。

真实工程里，Megatron Core 已经把 `z_loss_func` 作为标准工具暴露出来，这说明它不是论文里的“附加技巧”，而是生产 MoE 训练流水线中的常规稳定化组件。它通常和容量限制、aux load balancing、mixed precision 策略一起配置，而不是孤立存在。

---

## 替代方案与适用边界

z-loss 很有效，但不是唯一方案。把替代手段放在一起看，边界会更清楚。

| 手段 | 主要解决什么 | 优点 | 代价或边界 |
|---|---|---|---|
| update clipping | 控制参数更新幅度 | 实现简单，短期能止血 | 可能明显伤害模型质量 |
| gradient clipping | 控制梯度范数 | 通用稳定化手段 | 不直接约束 router logits 漂移 |
| 更高精度格式 | 减少数值舍入和溢出 | 对底层数值更友好 | 显存和吞吐成本更高 |
| load-balance loss | 防止专家流量极不均 | 改善专家利用率 | 不能直接解决 logits 爆炸 |
| z-loss | 直接约束 router logits 尺度 | 稳定性强，通常质量损失小 | 系数不当会让路由过软 |

ST-MoE 的一个关键实验是：单纯依赖 update clipping 虽然也可能提升稳定性，但模型质量下降更明显。原因不复杂。clipping 是对整个更新做硬限制，等于把很多本来有效的优化步也一起压扁了；z-loss 则是更局部地作用在 router 的危险维度上，所以通常更“对症”。

什么时候 z-loss 不是首选？有三个边界。

第一，你的模型不是 MoE，或者路由器本身不是 softmax/top-k 结构，那就不一定适用。

第二，如果训练早期根本不是 router 出问题，而是数据脏、学习率爆炸、全局梯度异常，那么先解决全局优化问题，比加 z-loss 更优先。

第三，如果你的目标是让专家分配更均匀，而不是解决数值崩溃，那么负载均衡、capacity factor、token dropping 这些手段更直接。

可以把决策顺序记成一句话：先确认是不是 router logits 漂移，再决定是否引入 z-loss；如果是，就优先用 z-loss，而不是先上激进 clipping。

---

## 参考资料

- ST-MoE 论文：<https://arxiv.org/pdf/2202.08906.pdf>
- Router Z-Loss 讲解文章：<https://mbrenndoerfer.com/writing/router-z-loss-moe-training-stability>
- NVIDIA Megatron Core `z_loss_func` 文档：<https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/apidocs/core/core.transformer.moe.moe_utils.html>

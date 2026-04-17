## 核心结论

Switch Transformer 的混合精度策略不是“全模型尽量低精度”，而是“只在最敏感的路由环节保留高精度”。结论可以压缩成两条：

1. 路由器 `router` 的 `softmax` 和 `Top-K/Top-1` 必须在 `float32` 下执行。这里的路由器，白话说，就是“决定每个 token 送去哪个专家网络”的分发模块。  
2. 专家内部的 FFN，白话说，就是“真正做大部分非线性计算的前馈网络”，仍然可以用 `bfloat16`，从而保住吞吐率和显存效率。

Switch Transformer 的经验结果表明，若把 router 也放进低精度，训练很容易发散；而采用 selective precision，只给 router 的关键计算升到 `float32`，稳定性会显著改善，同时速度接近纯 `bfloat16` 方案。

下面这张表先给出结论层面的对比：

| 方案 | Router precision | FFN precision | 吞吐率 | 稳定性 |
|---|---|---:|---:|---|
| 全局 `bfloat16` | `bfloat16` | `bfloat16` | 高 | 差，容易发散 |
| Selective Precision | `float32` 仅用于 softmax/Top-K | `bfloat16` | 接近高 | 好，接近全 `float32` |
| 全局 `float32` | `float32` | `float32` | 低 | 最稳，但代价大 |

对初学者可以这样理解：Switch Transformer 在“选专家”之前，先把 router logits 临时提升到 `float32`，像把敏感测量仪表换成高精度；选完专家后，再回到 `bfloat16` 去做大计算。高精度只花在最值得的地方。

---

## 问题定义与边界

问题的核心不是 MoE 一定难训，而是 Sparse MoE 的路由计算对数值精度异常敏感。Sparse MoE，白话说，就是“很多专家都存在，但每个 token 只激活其中很少几个”的模型结构。Switch Transformer 更激进，它通常只做 `Top-1` 路由，即每个 token 只选一个专家。

这会带来一个放大器效应：

$$
p_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

这里 $z_i$ 是 router 给第 $i$ 个专家的打分，$p_i$ 是送到该专家的概率。`softmax` 本质上是“把打分变成概率分布”的函数。问题在于它依赖指数函数 $\exp(z)$，而指数函数对输入尺度和精度都很敏感。

如果用 `bfloat16` 直接算 router：

- 大的正 logits 会让某个专家概率接近 1
- 小的负 logits 可能被舍入到几乎没有区分度
- `Top-1` 会把这种微小误差放大成离散决策，直接改变“进哪个专家”

对新手可以这样理解：`softmax` 像一台“比分机器”。当两个选手分差已经很大，而机器本身精度又不够时，它不会给你一个细腻的比分，而是直接把输家记成 0。MoE 又恰好要基于这个比分去做离散路由，所以误差会被立刻放大。

这里要明确边界：

- 不是所有混合精度训练都会出问题，问题集中在 router 的概率计算和离散选择上。
- 也不是说专家 FFN 必须 `float32`，真正敏感的是 `softmax`、`Top-K/Top-1` 以及与之直接相关的 logits 统计量。
- 仅靠负载均衡 loss 不能替代高精度 softmax，因为负载均衡处理的是“分配倾向”，不是底层数值稳定性。

---

## 核心机制与推导

Selective Precision 的机制可以拆成两步。

第一步是只把 router 的关键链路切到 `float32`：

$$
z = xW_r
$$

其中 $x$ 是 token 表示，$W_r$ 是 router 权重。随后在 `float32` 下计算：

$$
p = \text{softmax}(z)
$$

再做 `Top-K` 或 `Top-1` 选择：

$$
k^\*=\arg\max_i p_i
$$

dispatch/combine 之后，再把中间结果转回 `bfloat16`，继续送入专家 FFN。这样做的逻辑是：最怕精度丢失的是“选专家”，不是“专家内部乘加”。

第二步是加入 router z-loss。它的定义是：

$$
\mathcal{L}_z=\frac{1}{B}\sum_{i=1}^{B}\left(\log \sum_{j=1}^{N}\exp(z_{i,j})\right)^2
$$

这里：

- $B$ 是 batch 中 token 数
- $N$ 是专家数
- $\log\sum\exp$ 即 `logsumexp`，白话说，就是“比直接求和更稳定地统计指数规模”的写法

z-loss 的作用不是直接让路由更均匀，而是抑制 logits 无限膨胀。因为 softmax 在 logits 绝对值越来越大时会快速饱和，梯度会变得非常小，甚至数值溢出。z-loss 相当于给 router 一个“别把分数打得越来越离谱”的约束。

玩具例子可以直接看二维 logits：

$$
z=[12,-12]
$$

理论上：

$$
\exp(12)\gg \exp(-12)
$$

于是 softmax 接近：

$$
[0.9999999999,\ 0.0000000000]
$$

如果低精度把第二项直接压到 0，`Top-1` 就永远选第一个专家。此时第二个专家几乎收不到梯度，路由会越来越尖锐。

加入 z-loss 后，`logsumexp(z)` 很大，平方惩罚也大，梯度会把整体 logits 往回拉。它不是强迫两个概率相等，而是防止 router 打分无界增长。结果通常是：概率仍然有主次，但不会过早塌成“一个专家吃掉全部 token”。

真实工程例子是大规模 Switch 预训练。论文报告中，纯 `bfloat16` router 会出现发散，而 selective precision 在 32 个 TPUv3 上能保持接近纯低精度的吞吐率，同时训练稳定。这说明主要不稳定源不是专家 FFN，而是 router 的低精度 softmax 链路。

可以从梯度再看一层。softmax 的导数满足：

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
$$

当某个 $p_i$ 过早接近 1，其他概率接近 0 时，大多数梯度都会接近 0。低精度会更早把这种“饱和区”触发出来。`float32` 的价值不是让模型更聪明，而是让它别过早进入不可恢复的饱和区。

---

## 代码实现

实现原则很简单：router logits 先升精度，再做 softmax 和 Top-K，z-loss 直接基于 logits 计算，专家 FFN 保持低精度。

下面是一个可运行的 Python 玩具实现，展示三个点：

1. softmax 应基于较高精度 logits 计算  
2. z-loss 要在 `Top-K` 之前、直接对 logits 计算  
3. 最终路由决策可以再送回低精度执行专家计算

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def top1(probs):
    return max(range(len(probs)), key=lambda i: probs[i])

def router_step(logits, z_loss_coef=1e-2):
    # 实际工程里这里通常是 cast 到 float32 后再算
    probs = softmax([float(x) for x in logits])
    expert_id = top1(probs)
    z = logsumexp([float(x) for x in logits])
    z_loss = z_loss_coef * (z ** 2)
    return probs, expert_id, z_loss

# 玩具例子：一个明显尖锐的路由分布
logits = [12.0, -12.0]
probs, expert_id, z_loss = router_step(logits)

assert expert_id == 0
assert probs[0] > 0.999999
assert probs[1] < 1e-9
assert z_loss > 1.0

# 一个更温和的例子：概率还没完全塌缩
logits2 = [2.0, 1.5, 0.1]
probs2, expert_id2, z_loss2 = router_step(logits2)

assert len(probs2) == 3
assert abs(sum(probs2) - 1.0) < 1e-9
assert expert_id2 == 0
assert z_loss2 > 0.0
```

对应到训练框架中的伪代码，一般是：

```python
# router: sensitive path
logits_fp32 = cast(router_input @ router_w, "float32")
router_probs = softmax(logits_fp32, axis=-1)
expert_index = top_k(router_probs, k=1)

# z-loss computed before discrete dispatch
z_loss = z_loss_coef * mean(square(logsumexp(logits_fp32, axis=-1)))

# dispatch/combine can be converted back for expert compute
dispatch_weight = cast(router_probs, "bfloat16")
expert_out = expert_ffn(dispatch_weight, expert_inputs_bf16)

total_loss = task_loss + aux_load_balance_loss + z_loss
```

这里有两个落点必须明确：

- `cast(float32)` 的位置要在 `softmax` 之前，而不是之后。
- z-loss 作用对象是 logits，不是已经离散化后的 expert index。

真实工程里，监控指标至少包括：

| 指标 | 含义 | 作用 |
|---|---|---|
| `router_logsumexp_mean` | router logits 的指数尺度 | 观察是否持续膨胀 |
| `expert_token_fraction` | 每个专家接收 token 占比 | 观察是否出现专家饿死 |
| `router_entropy` | 路由分布熵，白话说就是“分布有多尖” | 判断是否过早塌缩 |
| `z_loss` | 数值稳定正则项 | 判断约束是否有效 |

---

## 工程权衡与常见坑

Selective Precision 的工程价值在于“只修最脆弱的一段”。它不是理论上最纯粹的方案，但在吞吐率、显存和稳定性之间很均衡。

常见错误配置如下：

| 错误配置 | 直接后果 | 缓解方式 |
|---|---|---|
| Router 全程 `bfloat16` | softmax 饱和、Top-1 错判、训练发散 | `softmax/Top-K` 前强制转 `float32` |
| `softmax` 后又在 `Top-K` 前转回 `bfloat16` | 路由排序被舍入误差污染 | `Top-K` 也保留在 `float32` |
| `z-loss coef = 0` | logits 膨胀，`logsumexp` 持续变大 | 启用 `1e-2` 或 `2e-2` 量级系数并监控 |
| 只加 z-loss，不做高精度 softmax | 分布略缓，但数值问题仍在 | z-loss 只能补充，不能替代 `float32` router |
| 只做高精度 router，不做负载均衡 | 专家可能严重偏载 | 与 load-balancing loss 联合使用 |

有两个坑最容易被忽略。

第一，`z-loss` 和负载均衡 loss 解决的不是同一个问题。负载均衡 loss 处理“别让某些专家太忙，某些专家没活干”；z-loss 处理“别让 logits 绝对值越来越离谱，导致 softmax 数值坏掉”。前者偏分配，后者偏数值。缺其中一个，都可能出问题。

第二，`Top-1` 比 `Top-2` 更敏感。因为 `Top-1` 是单路由，一次误判就直接改了 token 去向；`Top-2` 至少还能保留次优专家的信息。但 Switch Transformer 选择 `Top-1` 是为了更高效率，所以它更依赖 selective precision 和 z-loss 来兜底。

经验上，若你看到下面现象，优先排查 router 数值链路：

- loss 突然 NaN
- 某个专家长期接近 0 token
- router 熵快速跌到极低
- `logsumexp` 均值持续上升
- 同样超参下，`float32` 稳定而 `bfloat16` 立刻发散

---

## 替代方案与适用边界

如果只看稳定性，全 `float32` routing 当然更直接，但它的代价也最明确：更高的内存带宽压力、更慢的训练速度、更差的吞吐率。它更适合 debug、小模型复现实验，或者你正在排查“到底是不是数值问题”。

如果只加正则，不改精度，则通常不够。比如加 entropy regularization 或 KL regularization，本质上是在鼓励分布别太尖，但它们建立在“softmax 已经被正确计算”的前提上。若底层 `softmax` 自己就因为低精度而失真，正则项也只是对错误概率继续修补，不能从根上解决。

可以用一个对比表收尾：

| 方案 | 稳定性 | 性能 | 适用场景 | 边界 |
|---|---|---:|---|---|
| Selective Precision | 高 | 高 | 大规模训练默认选择 | 需单独实现 router 高精度链路 |
| 全 `float32` routing | 很高 | 中到低 | Debug、验证数值问题 | 成本较高，不适合长期大规模训练 |
| 仅 `z-loss` + 全局 `bfloat16` | 低到中 | 高 | 只做快速试验 | 不能替代高精度 softmax |
| 高精度 router + entropy/KL 正则 | 高 | 中高 | 对尖锐分布额外约束 | 仍必须保留 `float32` softmax |

对初学者可以这样记：KL 或 entropy 正则是在“压平概率分布”，而 selective precision 是在“确保你算出来的概率本身是可信的”。前者是调形状，后者是保真度。顺序不能反。

---

## 参考资料

1. Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR, 2022. 重点看第 2.4 节与表 2，给出 selective precision 的稳定性与吞吐率对比。  
2. Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. 文中系统讨论了 router z-loss 的定义、动机与训练稳定性作用。  
3. Michael Brenndoerfer. *Router Z-Loss: Numerical Stability for MoE Training*. 对 z-loss 的数值意义、`logsumexp` 监控方式和工程落点有较清晰的实践解释。  
4. Switch Transformer 相关实现笔记与公开复现项目。适合对照查看 `router cast -> softmax -> topk -> z-loss -> dispatch` 的实际代码路径。

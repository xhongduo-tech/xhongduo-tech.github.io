## 核心结论

ST-MoE 的核心贡献，不是重新发明一种新的 MoE 结构，而是系统回答了一个更实际的问题：为什么大规模 MoE 经常不是“效果差”，而是“根本训不完”。论文把这个问题拆成三个相互关联的失稳源头，然后分别处理：

| 稳定性维度 | Baseline MoE 常见状态 | ST-MoE 的处理 |
|---|---|---|
| router logits 幅度 | 往往持续增大，缺少显式约束 | 加入 `z-loss`，约束 logit 整体尺度 |
| 专家负载 | 少数专家拥塞，其他专家空闲 | 保留并调优 `balance loss`，抑制挤兑 |
| 数值精度 | router 路径常直接跟随混合精度 | router 与 `softmax/logsumexp/exp` 强制 `float32` |
| 训练现象 | loss 跳变、overflow、NaN、训练中断 | 曲线更平滑，训练可持续完成 |
| 实验结论 | 大模型训练存在明显失败率 | 失败率显著下降，迁移与下游效果更稳定 |

MoE，Mixture of Experts，中文通常叫“专家混合模型”。它的基本做法不是让每个 token 都经过完整前馈网络，而是由一个 router 决定“这个 token 该送给哪些专家处理”。这样可以把总参数量做得很大，但单步计算量只增长一部分，而不是按参数量线性增长。

新手可以先抓住一句话：ST-MoE 并没有改变“token 只走少数专家”这件事，它做的是给路由系统加上三道保险。

1. `z-loss` 防止 router logits 越变越大。
2. `balance loss` 防止所有 token 挤向少数专家。
3. `float32` 防止指数运算在低精度下失真或溢出。

如果把 MoE 训练类比成“分流系统”，那么 baseline 常见问题不是“路分得不够聪明”，而是“分流器越来越极端，最终把流量全部压到少数出口，系统直接堵塞甚至崩溃”。ST-MoE 的价值就在这里：先让系统稳定，再讨论模型效果。

三种稳定化手段可以压缩成一句话：

`z-loss` 控制 router 输出幅度，`balance loss` 约束专家负载分布，`float32` 保护最脆弱的数值路径。

---

## 问题定义与边界

ST-MoE 讨论的问题很具体：在大规模稀疏 MoE 训练中，router 会为每个 token 输出一组 logits，也就是 softmax 之前的原始分数。logits 一旦整体持续抬高，softmax 会迅速饱和，路由分布变得极端，随后引出专家拥塞、token 丢弃、梯度偏斜、overflow 和 NaN，最终训练中断。

softmax 定义为：

$$
p_i=\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

其中 $x_i$ 是第 $i$ 个专家的 logit，$p_i$ 是该专家被选中的概率。

当某个 $x_k$ 远大于其他 $x_j$ 时，会出现：

$$
p_k \approx 1,\quad p_{j\neq k}\approx 0
$$

这看起来像“router 更有信心了”，但从训练角度看并不是好事。因为 router 一旦长期过度自信，系统会同时出现三类问题：

| 问题类型 | 直接现象 | 后续连锁反应 |
|---|---|---|
| 概率饱和 | 选中的专家概率接近 1，其他接近 0 | 路由探索空间收缩，梯度学习变差 |
| 负载失衡 | 少数专家收到大量 token | 容量溢出、token 被丢弃、吞吐不稳 |
| 数值不稳定 | `exp` 和 `softmax` 在大 logit 下敏感 | 低精度下更容易 overflow / NaN |

可以看一个更直观的两专家例子。设 logits 从 `[4, 1]` 变成 `[8, 1]`，再变成 `[20, 1]`：

| logits | softmax 概率 |
|---|---|
| `[4, 1]` | `[0.9526, 0.0474]` |
| `[8, 1]` | `[0.9991, 0.0009]` |
| `[20, 1]` | `[0.999999994, 0.000000006]` |

这不是“只是更偏向专家 1”，而是“几乎所有 token 都会被推给专家 1”。如果这是 top-2 或 top-k 路由中的长期趋势，那么某些专家会变成热点，其他专家几乎学不到东西，训练逐渐退化成一套不健康的稀疏分配。

这里还要区分“相对差距变大”和“整体抬高”两个概念。ST-MoE 特别关心后者。因为即使 logits 的相对差距没变，只要整体抬高，`exp` 和 `logsumexp` 的数值风险也会上升。例如：

$$
[2,-1] \rightarrow [8,5]
$$

两者差值都为 3，但第二组 logits 的绝对尺度更大，softmax 更接近饱和，数值也更危险。ST-MoE 的 `z-loss` 主要就是为这个问题设计的。

论文讨论的边界也很清楚，它不是泛泛地说“所有 MoE 都不稳定”，而是针对大规模、稀疏、top-2 routing 训练区间给出稳定方案。

| 边界项 | 典型设定 |
|---|---|
| 路由方式 | `top-2 routing` |
| 训练目标 | 在较高参数规模下保持可控 FLOPs |
| 容量设置 | 训练时常见 capacity factor 约 `1.25` |
| 模型规模 | T5-XL 到更大规模的稀疏模型 |
| 问题焦点 | 训练可完成性，而非只看最终分数 |

top-2 routing 可以简单理解为：每个 token 不是把所有专家都算一遍，而是只选择得分最高的两个专家。这种方式节省计算，但也把训练稳定性高度集中到了 router 上。router 一旦出问题，后面所有稀疏激活路径都会一起出问题。

整个失稳链条可以写成：

```text
Baseline MoE training
  -> router logits 持续变大
  -> softmax 饱和
  -> 少数专家拥塞，部分 token 被截断或丢弃
  -> overflow / NaN / loss spike
  -> training abort
```

而 ST-MoE 的对应链条是：

```text
ST-MoE training
  -> z-loss 抑制 logits 绝对尺度
  -> balance loss 约束专家负载
  -> router path 使用 float32
  -> softmax / logsumexp 更稳定
  -> loss 曲线平稳
  -> training completes
```

这也是为什么 ST-MoE 首先是一个训练稳定性方案，其次才是一个模型效果方案。模型训不完，下游任务分数没有讨论意义。

---

## 核心机制与推导

ST-MoE 的核心机制可以概括成一个总损失：

$$
L_{\text{total}} = L_{\text{task}} + c_B L_B + c_z L_z
$$

其中：

- $L_{\text{task}}$ 是主任务损失，通常是语言建模或 seq2seq 的交叉熵。
- $L_B$ 是负载平衡损失，用来限制专家使用过度倾斜。
- $L_z$ 是 router z-loss，用来抑制 logits 整体尺度持续膨胀。
- $c_B$ 和 $c_z$ 是两个辅助损失系数。

### 1. z-loss：约束 logits 的绝对尺度

ST-MoE 使用的 z-loss 定义为：

$$
L_z(x)=\frac{1}{B}\sum_{i=1}^{B}\left(\log \sum_{j=1}^{N}\exp(x_j^{(i)})\right)^2
$$

其中：

- $B$ 是 batch 中参与路由的 token 数。
- $N$ 是专家数。
- $x_j^{(i)}$ 是第 $i$ 个 token 对第 $j$ 个专家的 logit。
- $\log \sum \exp(\cdot)$ 即 `logsumexp`，是 softmax 分母的稳定对数形式。

为什么这个定义能约束 logits 的规模？因为当 logits 整体上移时，`logsumexp` 也会随之增加。

设某个 token 的 logits 向量整体加上常数 $\Delta$，即：

$$
x'_j = x_j + \Delta
$$

则有：

$$
\log\sum_j e^{x'_j}
= \log\sum_j e^{x_j+\Delta}
= \log\left(e^\Delta \sum_j e^{x_j}\right)
= \Delta + \log\sum_j e^{x_j}
$$

所以 `logsumexp` 会随着整体平移线性增长，平方后的 z-loss 会更快增大。也就是说，z-loss 惩罚的不是“某个值偶尔偏大”，而是“整组 logits 在整体抬高”。

这点很关键，因为 router 的失稳常常不是单点异常，而是整体尺度漂移。

看一个最小例子。若单个 token、两个专家，logits 为 `[4, 1]`：

$$
\log(e^4+e^1)\approx 4.049
$$

所以：

$$
L_z \approx 4.049^2 \approx 16.39
$$

如果改成 `[8, 5]`，虽然两个值之差仍然是 3，但：

$$
\log(e^8+e^5)\approx 8.049
$$

于是：

$$
L_z \approx 8.049^2 \approx 64.79
$$

这说明 z-loss 对“整体抬高”非常敏感，这正是它区别于简单 clipping 的地方。

### 2. balance loss：约束专家使用分布

仅有 z-loss 还不够，因为 logits 不爆并不等于负载一定均衡。router 可能在一个较稳定的数值区间内，依然长期偏向少数专家。

因此需要 balance loss。它的目的不是让每个专家永远完全平均，而是避免专家利用率塌缩到极少数节点。

在 MoE 文献里，负载平衡通常围绕两个量展开：

- `importance`：某个专家从概率上“被偏好”的程度。
- `load`：某个专家实际接收到多少 token。

直觉上可以这样理解：

| 指标 | 含义 | 如果失衡会怎样 |
|---|---|---|
| importance | router 在概率层面偏爱谁 | 某些专家长期高权重 |
| load | 实际有多少 token 被分给谁 | 某些专家直接拥塞或被闲置 |

不同实现的 balance loss 公式不完全相同，但目标一致：降低专家使用分布的方差，避免极端偏置。

### 3. 为什么两者必须同时存在

这两个辅助项各自解决的不是同一个问题。

| 只使用某项 | 能解决什么 | 解决不了什么 |
|---|---|---|
| 只用 `z-loss` | 控制 logits 尺度，缓和 softmax 饱和 | 不能保证专家负载均衡 |
| 只用 `balance loss` | 防止专家过度拥塞 | 不能阻止 logits 数值持续膨胀 |
| 两者同时使用 | 同时约束数值稳定性和负载分配 | 才接近完整稳定方案 |

这可以从反向传播路径看得更清楚：

```text
router logits
  -> softmax / top-k
  -> token 分配到专家
  -> 主任务损失回传
  -> z-loss 约束 logits 绝对尺度
  -> balance loss 约束专家分配形状
  -> router 参数被拉回稳定区
```

### 4. 为什么 `float32` 是必要条件，不是细节

ST-MoE 的另一个关键发现是：即使模型主体在 `bf16/fp16` 下训练得还可以，router 路径也不应该简单跟随降精度。

原因是 router 路径包含最脆弱的操作：

- `softmax`
- `exp`
- `logsumexp`
- top-k 前的排序比较
- load / importance 的统计累计

这些步骤对数值误差非常敏感。尤其当 logits 已经偏大时，低精度下的小误差会被指数运算放大。

可以把这个问题写成一个简单表：

| 操作 | 风险 |
|---|---|
| `exp(x)` | 当 $x$ 偏大时容易溢出或失真 |
| `softmax` | 大 logit 下分布极端，低精度更容易退化 |
| `logsumexp` | 本来是稳定写法，但前提是足够精度 |
| 负载累计 | 低精度累计误差会影响 balance 统计 |

因此 ST-MoE 的策略不是“全模型一律高精度”，而是“只把最脆弱的局部路径固定为 `float32`”。这是一种典型工程折中：用少量额外成本换显著稳定性收益。

### 5. 系数调度为什么重要

`c_z` 和 `c_B` 不是越大越好。辅助损失过弱，稳定性不够；过强，又会压制主任务学习。

论文中的经验不是“一上来就把正则拉满”，而是扫描与调度。对新手来说，最实用的理解方式是：

- 把 `c_z` 看成“router 软刹车”的力度。
- 把 `c_B` 看成“交通疏导”的力度。
- 两者都应该先从小值开始，再根据监控结果调整。

常见经验值可以写成：

| 系数 | 常见起点 | 作用 |
|---|---|---|
| `c_z` | `1e-4` 到 `1e-3` | 抑制 logits 漂移 |
| `c_B` | `1e-2` 左右 | 控制专家分布失衡 |

这些值不是通用常数。数据分布、batch size、专家数、capacity factor、是否启用 token dropping，都会改变最优区间。论文给的是经验边界，不是免调参模板。

---

## 代码实现

工程上最重要的原则有两条：

1. `z-loss` 应该在 router forward 之后直接计算，而不是等训练发散后再“补救”。
2. router 相关的关键数值路径要显式切到 `float32`，不要依赖框架默认行为。

下面先给出一个可以直接运行的纯 Python 示例。它不依赖深度学习框架，但完整展示了三个核心概念：

- 稳定 softmax
- `z-loss`
- 一个可理解的 balance loss

```python
import math
from typing import List, Tuple


def stable_softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def logsumexp(logits: List[float]) -> float:
    m = max(logits)
    return m + math.log(sum(math.exp(x - m) for x in logits))


def z_loss(batch_logits: List[List[float]]) -> float:
    if not batch_logits:
        raise ValueError("batch_logits must not be empty")
    vals = [logsumexp(logits) ** 2 for logits in batch_logits]
    return sum(vals) / len(vals)


def topk_indices(xs: List[float], k: int) -> List[int]:
    if k <= 0 or k > len(xs):
        raise ValueError("invalid k")
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:k]


def importance_vector(router_probs: List[List[float]]) -> List[float]:
    num_tokens = len(router_probs)
    num_experts = len(router_probs[0])
    importance = [0.0] * num_experts
    for probs in router_probs:
        for i, p in enumerate(probs):
            importance[i] += p
    return [x / num_tokens for x in importance]


def load_vector(router_probs: List[List[float]], top_k: int = 2) -> List[float]:
    num_tokens = len(router_probs)
    num_experts = len(router_probs[0])
    load = [0.0] * num_experts
    for probs in router_probs:
        for idx in topk_indices(probs, top_k):
            load[idx] += 1.0
    return [x / (num_tokens * top_k) for x in load]


def l2_uniform_penalty(xs: List[float]) -> float:
    n = len(xs)
    target = 1.0 / n
    return sum((x - target) ** 2 for x in xs) / n


def balance_loss(router_probs: List[List[float]], top_k: int = 2) -> Tuple[float, List[float], List[float]]:
    imp = importance_vector(router_probs)
    load = load_vector(router_probs, top_k=top_k)
    loss = 0.5 * l2_uniform_penalty(imp) + 0.5 * l2_uniform_penalty(load)
    return loss, imp, load


def total_loss(
    ce_loss: float,
    batch_logits: List[List[float]],
    c_z: float = 1e-3,
    c_b: float = 1e-2,
    top_k: int = 2,
) -> Tuple[float, float, float]:
    router_probs = [stable_softmax(logits) for logits in batch_logits]
    z = z_loss(batch_logits)
    b, _, _ = balance_loss(router_probs, top_k=top_k)
    total = ce_loss + c_z * z + c_b * b
    return total, z, b


def main() -> None:
    batch_logits = [
        [4.0, 1.0, 0.0, -1.0],
        [3.5, 0.5, 0.0, -0.5],
        [2.0, 2.0, 1.0, 0.0],
        [5.0, 1.0, -1.0, -2.0],
    ]

    router_probs = [stable_softmax(x) for x in batch_logits]
    total, z, b = total_loss(ce_loss=2.3, batch_logits=batch_logits, c_z=1e-3, c_b=1e-2, top_k=2)

    print("router_probs:")
    for probs in router_probs:
        print([round(p, 6) for p in probs])

    print("z_loss =", round(z, 6))
    print("balance_loss =", round(b, 6))
    print("total_loss =", round(total, 6))

    # 基本正确性检查
    assert z_loss([[4.0, 1.0]]) > 16.0
    assert stable_softmax([20.0, 1.0])[0] > 0.99999999
    assert total > 2.3


if __name__ == "__main__":
    main()
```

这个示例和真实训练之间有两个差别，需要明确说明。

| 示例代码 | 真实训练 |
|---|---|
| 用 Python 列表演示 | 用张量和并行计算 |
| balance loss 用简化的均匀惩罚 | 实际实现通常与具体路由策略绑定 |
| 不做反向传播 | 实际要纳入自动求导图 |
| 没有容量截断 | 真实系统要处理 capacity 和 token dropping |

如果你在 PyTorch 中实现，关键不是代码长短，而是 dtype 位置必须正确。下面这个版本更接近真实训练：

```python
import torch
import torch.nn.functional as F


def compute_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    # router_logits: [num_tokens, num_experts]
    logits_f32 = router_logits.float()
    return torch.mean(torch.logsumexp(logits_f32, dim=-1) ** 2)


def compute_balance_loss(router_probs: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    # router_probs: [num_tokens, num_experts]
    num_tokens, num_experts = router_probs.shape

    importance = router_probs.mean(dim=0)

    topk_idx = torch.topk(router_probs, k=top_k, dim=-1).indices
    one_hot = F.one_hot(topk_idx, num_classes=num_experts).float()
    load = one_hot.sum(dim=(0, 1)) / (num_tokens * top_k)

    target = torch.full((num_experts,), 1.0 / num_experts, device=router_probs.device)
    return 0.5 * ((importance - target) ** 2).mean() + 0.5 * ((load - target) ** 2).mean()


def moe_aux_losses(router_logits: torch.Tensor, c_z: float = 1e-3, c_b: float = 1e-2, top_k: int = 2):
    logits_f32 = router_logits.float()
    router_probs = F.softmax(logits_f32, dim=-1)

    z = compute_z_loss(logits_f32)
    b = compute_balance_loss(router_probs, top_k=top_k)
    aux = c_z * z + c_b * b
    return aux, z, b
```

在真正的训练代码里，结构通常是下面这样：

```python
hidden_states = backbone_outputs

router_logits = router(hidden_states).float()          # 关键：router 路径先转 float32
router_probs = torch.softmax(router_logits, dim=-1)    # 关键：softmax 在 float32 上做
topk_vals, topk_idx = torch.topk(router_probs, k=2, dim=-1)

expert_outputs = dispatch_to_experts(hidden_states, topk_idx, topk_vals)
main_loss = compute_task_loss(expert_outputs, labels)

z = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
b = compute_balance_loss(router_probs, top_k=2)

loss = main_loss + c_z * z + c_B * b
loss.backward()
```

哪些路径必须优先保持 `float32`，可以明确列出来：

| 模块 | 是否建议 `float32` | 原因 |
|---|---|---|
| router logits | 是 | softmax 稳定性的源头 |
| `exp` / `softmax` / `logsumexp` | 是 | 数值最敏感 |
| top-k 前的概率计算 | 是 | 排序结果受数值误差影响大 |
| load / importance 统计 | 是 | 累计误差会扭曲 balance loss |
| 主干 Attention / FFN | 可混合精度 | 一般不是最先失稳的位置 |

对于新手，一个实用判断标准是：如果某段代码直接决定“token 去哪个专家”，或者直接包含指数运算，就默认优先检查它的 dtype。

---

## 工程权衡与常见坑

ST-MoE 不是“加一个辅助项就行”的技巧，它更像一组必须配套落地的工程约束。最常见的坑如下。

| 坑 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| `c_z` 过大 | 主任务 loss 降不下去 | router 被过度压制，分流不灵活 | 从小值起步，逐步增加 |
| `c_z` 过小 | logits 继续膨胀 | z-loss 形同虚设 | 监控 `z-loss` 与 logit 均值/方差 |
| `c_B` 过大 | 专家看似均衡，但性能下降 | router 被迫平均化，失去选择能力 | 让 balance 成为约束，不要成为主导目标 |
| `c_B` 过小 | 热点专家持续拥塞 | 负载约束不足 | 监控各专家 token 占比 |
| router 使用 `fp16/bf16` | loss spike、overflow、NaN | 指数相关运算对低精度敏感 | router path 显式转 `float32` |
| 只看总 loss | 崩溃前没有明显预警 | 局部异常被主损失掩盖 | 单独记录 `z-loss`、load、overflow |
| 只做 clipping | 局部看似稳定，长期仍发散 | 只能裁极值，不能压整体漂移 | clipping 只能做补充，不能替代 z-loss |

### 1. `c_z` 不是越大越稳

一个常见误区是：既然 z-loss 有用，那就直接把 `c_z` 设得很大。这样做经常会让 router 进入另一种坏状态，即“过度保守”。表面上 logits 很小，训练也不炸，但模型不会有效区分应该送往哪个专家，最终主任务性能下降。

这可以理解成另一个极端：不是路由器过度自信，而是路由器几乎不敢做明确决策。

### 2. balance loss 也可能伤害性能

balance loss 的目标是防止塌缩，不是追求机械平均。真实数据本来就可能具有不均匀模式，如果强行要求所有专家完全均匀，router 会失去根据样本内容做差异化分配的能力。

所以在工程实践里，更合理的目标通常是：

- 避免极端失衡；
- 接受适度不均衡；
- 不让辅助项压过主任务。

### 3. clipping 为什么不能替代 z-loss

clipping 的做法是：超过阈值就截断，例如把 logits 限制在 `[-10, 10]`。这个方法能缓解极端值，但处理不了“整体漂移”问题。

例如这两组 logits：

$$
[2,-1],\quad [8,5]
$$

它们的相对差距相同，但第二组的整体尺度更大。若 clipping 阈值较高，第二组可能根本不会被裁；即使被裁，router 也可能长期贴着上限运行。softmax 饱和和数值脆弱性仍然存在。

而 z-loss 会持续对整体尺度施压，因此更适合作为主稳定化方案。

### 4. 监控要看什么

在 MoE 训练里，单看总 loss 往往不够。因为很多问题最开始只会出现在 router 局部路径上，总 loss 可能晚几千步才表现出异常。

建议至少记录以下指标：

| 指标 | 监控目的 |
|---|---|
| `z-loss` 的 step 曲线和滑动平均 | 判断 logits 是否整体漂移 |
| 每个专家的 token 占比 | 判断是否出现热点专家 |
| 最大专家占比 / 最小专家占比 | 快速观察负载失衡 |
| token drop rate | 判断容量是否频繁溢出 |
| router overflow / NaN 计数 | 提前发现数值问题 |
| router logits 的均值与方差 | 判断路由分数是否失控 |

如果只能保留最少监控，我建议优先保留这三个：

1. `z-loss`
2. 每个 expert 的 load 分布
3. overflow / NaN 计数

### 5. 一个典型故障排查顺序

当 MoE 训练出现 loss spike 或 NaN 时，排查顺序可以按下面走：

1. 先检查 router 路径是不是确实在 `float32` 上运行。
2. 再看 `z-loss` 是否长期上升或突然飙升。
3. 再看专家负载是否集中到少数 expert。
4. 最后再回看全局学习率、梯度裁剪和混合精度配置。

这个顺序的意义是：ST-MoE 指出的主要故障源头就在 router 路径，先查最可能的地方，效率最高。

---

## 替代方案与适用边界

ST-MoE 不是唯一可行路线，但它针对的是“大规模稀疏路由为什么经常训崩”这个核心问题。把它与常见替代方案放在一起看，更容易理解它的定位。

| 方案 | 能解决什么 | 局限 | 适用边界 |
|---|---|---|---|
| ST-MoE：`z-loss + balance + float32 router` | 同时处理数值失稳和负载失衡 | 需要额外监控和调参 | 大规模 top-k 稀疏训练 |
| 仅 clipping logits | 缓解极端异常值 | 不能控制整体尺度漂移 | 小实验、补丁式修复 |
| 仅 gradient scaling | 改善部分混合精度训练稳定性 | 主要作用于反向缩放，不解决路由塌缩 | 通用混合精度训练 |
| 仅加 balance loss | 缓解专家拥塞 | 数值问题仍可能保留 | 中小规模、问题较轻时 |
| router 全程低精度 | 节省一点资源 | 大模型下风险高 | 小模型、验证性实验 |

对新手，一个很好记的比较方式是：

- clipping 像“超速抓拍”，只在太离谱时干预；
- z-loss 像“限速带”，从机制上降低整体失控概率；
- balance loss 像“车道分流”，避免所有车都挤到同一条路；
- `float32` 像“把最脆弱的桥梁加固”，防止关键环节先塌。

### 适用边界要明确

ST-MoE 并不是所有场景都同样值得上整套配置。

| 场景 | 是否强烈建议使用 ST-MoE 式稳定化 |
|---|---|
| 专家数少、模型较小、训练步数短 | 收益可能有限 |
| 中等规模稀疏模型，已出现专家失衡 | 建议至少引入 balance + dtype 保护 |
| 大规模 top-2 / top-k MoE | 基本建议作为默认起点 |
| 无法单独控制 router dtype 的框架 | 收益会打折，需特别警惕 |
| 无监控能力、无法调辅助系数 | 不建议机械照抄超参 |

还要强调一点：ST-MoE 讨论的是“最低必要稳定化措施”，不是“唯一正确写法”。如果你的训练规模还没到高风险区间，或者你的路由方案与论文差别很大，那么整套配置未必都要照搬。但一旦你进入大规模稀疏训练区间，它提供的是一条经过验证的起点，而不是拍脑袋试错。

---

## 参考资料

下面把核心资料按“能解决什么问题”重新整理。

| 资料 | 关键贡献 | 适用章节 |
|---|---|---|
| ST-MoE: Designing Stable and Transferable Sparse Expert Models (arXiv 2022) | 提出稳定训练问题定义、router z-loss、训练失败率分析与大模型实验 | 全文主线 |
| ST-MoE 论文 Appendix B | 给出辅助损失系数、router 精度设置、稳定训练实现细节 | 代码实现、工程权衡 |
| Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | 提供稀疏路由、专家负载平衡、capacity 等基础背景 | 问题定义、balance loss 背景 |
| GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | 早期大规模 MoE 路由与分布式稀疏训练背景 | 问题边界、工程理解 |

建议的阅读顺序如下：

1. 先读 ST-MoE 主文，抓住它到底在解决什么不稳定问题。
2. 再读 Appendix B，看超参、dtype 和训练细节。
3. 最后回看 Switch Transformer 或 GShard，对比 ST-MoE 到底新增了哪些“稳定化约束”。

阅读时建议重点关注三个问题：

1. `z-loss` 为什么盯的是 `logsumexp`，而不是直接对 logits 做 clipping。
2. 为什么论文反复强调 router 路径使用 `float32`。
3. 为什么“训练失败率下降”和“最终下游性能提升”必须分开评价。

常用链接如下：

- ST-MoE 论文：https://arxiv.org/pdf/2202.08906.pdf
- Switch Transformer 论文：https://arxiv.org/abs/2101.03961
- GShard 论文：https://arxiv.org/abs/2006.16668

如果只看一篇，优先看 ST-MoE 主文和附录；如果想真正理解它为什么必要，必须把它放回前代 MoE 训练问题的上下文里看。否则很容易误以为它只是“又加了一个正则项”，而忽略了它真正解决的是“大规模稀疏训练经常无法可靠完成”的系统性问题。

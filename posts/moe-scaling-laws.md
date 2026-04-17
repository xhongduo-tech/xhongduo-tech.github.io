## 核心结论

MoE，Mixture of Experts，直译是“专家混合”。它的做法不是把每个 token 都送进同一个大前馈层，而是先让一个路由器做选择，再把这个 token 分给少数几个专家子网络处理。最常见的情形是 top-1 或 top-2 路由，也就是“每个 token 只激活 1 个或 2 个专家”。

这会带来一个和 dense 模型明显不同的事实：**总参数量的增长，不再和单步计算量同步增长。**  
dense 模型里，参数越大，通常每一步也要真实计算更多参数；MoE 里，总参数可以很大，但单个 token 实际只经过其中一小部分，所以“参数增长”和“计算增长”被部分解耦。

Clark 等人把 routed language model 的经验规律写成下面这个统一形式：

$$
\log L(N,E)=a\log N+b\log E+c\log N\log E+d
$$

其中：

- $N$ 表示 dense model size，可以理解为“单个 token 实际会经过的那部分参数规模”
- $E$ 表示 expert count，即专家数量
- $L$ 表示验证损失
- 交互项 $c\log N\log E$ 用来描述 $N$ 和 $E$ 不是彼此独立的扩展轴

这条式子最重要的工程含义不是“专家越多越好”，而是：

**MoE 的收益通常存在一个清晰的甜点区间。专家数从较小规模扩到中等规模时，收益往往很明显；继续往上堆，loss 可能还会下降，但边际改善会越来越小。**

新手可以先用一个非常直白的想法理解这件事。

假设你只有 8 个专家，那么不同类型的 token 很容易被迫共享同一套前馈模式；当你增加到 32 或 64 个专家后，路由器能把“代码 token、数学 token、普通叙述 token、长尾词形”分给更合适的专家，效果通常会比较明显；但当专家已经很多时，新增加的专家接住的往往是更细碎、更少见的模式，这时继续翻倍专家数，收益自然会变缓。

Switch Transformer 的结果给了一个很典型的工程参照：在和 T5-Base 接近的每 token FLOPs 约束下，稀疏模型可以把更多参数暴露给训练，因此训练效率可能显著提高。它说明的不是“无限堆专家”，而是“**在固定计算预算下，用稀疏激活换取更大的有效参数池**”。

下表可以先把经验趋势记住：

| 专家数 $E$ | 常见现象 | 对 loss 的边际改善 |
|---|---|---|
| 8 | 稀疏化刚开始起作用 | 明显 |
| 16 | 专家分工开始出现 | 明显 |
| 32 | 路由更容易形成专业化 | 中等偏高 |
| 64 | 常见甜点区间 | 中等 |
| 128 | 通常仍有收益，但斜率变缓 | 中等偏低 |
| 256 | 更依赖数据量、路由质量和系统实现 | 偏低 |

如果只记一句话，可以记这句：

**MoE 的 scaling law 研究的核心，不是“总参数能做多大”，而是“在相同训练计算下，怎样把更多参数有效地变成真实收益”。**

---

## 问题定义与边界

这篇文章讨论的不是“MoE 能不能比 dense 更强”，而是更具体的问题：

**在给定训练计算预算下，MoE 应该怎样同时扩展 $N$ 和 $E$，才能继续保持好的 scaling，而不是过早进入收益递减区。**

这句话里有三个边界，必须先讲清楚。

第一，**数据边界**。  
scaling law 的讨论默认处在“数据足够多”或者至少“还没被数据瓶颈卡死”的区域。否则会出现一个常见误区：你以为是“专家加多了没用”，其实是“数据已经不够支撑更多专家去学出稳定分工”。因为专家越多，每个专家分到的有效样本往往越碎，数据不足时收益会更早饱和。

第二，**专家有效范围边界**。  
Clark 的结果不是说“128 个专家之后就没有收益”，而是说继续增加专家时，收益斜率通常会明显变小。所以工程上常把 64 到 128 视为优先检查区间，而不是一开始就直接跳到 256 或 512。原因很简单：在这个区间里，稀疏化收益通常还比较容易兑现，但通信、调度和负载不均问题还没恶化到最难控制的程度。

第三，**路由稳定性边界**。  
MoE 不是只看参数规模和 FLOPs。它还受到路由系统的约束，最典型的两个量是：

- `capacity factor`：每个专家最多预留多少 token 容量
- `auxiliary loss`：为负载均衡加入的辅助损失，避免 token 全挤到少数热门专家

如果这两个量没有调好，那么“专家更多”只会让系统更容易出现热点专家、token overflow、训练不稳定，而不会稳定地带来更好 loss。

可以先把问题空间压缩成下面这张表：

| 条件 | 状态 | 结果 |
|---|---|---|
| 数据量充足 | 是 | 扩专家更容易转化为真实收益 |
| 数据量充足 | 否 | 每个专家见到的数据更碎，收益提前饱和 |
| 专家数 | 8 到 64 | 通常是收益最明显的区间 |
| 专家数 | 64 到 128 | 往往仍有效，但更依赖路由质量 |
| 专家数 | 256 以上 | 更容易进入递减区，需要更强工程控制 |
| `capacity factor` + `aux loss` | 合理 | token 分布更稳，MoE 优势更容易兑现 |
| `capacity factor` + `aux loss` | 不合理 | token drop、热点专家、训练波动更明显 |

对初学者，一个最实用的理解方式是：

- dense scaling 关心的是“模型整体变大，loss 会怎样变”
- MoE scaling 额外多问了一层：“这些新增参数到底有多少能被路由系统稳定利用”

所以真实工程里，MoE 的优势并不来自“无限扩专家”，而来自“**在同样计算下，把稀疏参数做大，同时把路由副作用控制住**”。

---

## 核心机制与推导

先看 Clark 的公式本身：

$$
\log L(N,E)=a\log N+b\log E+c\log N\log E+d
$$

如果这里的 $\log$ 取自然对数，那么可以把它直接改写成：

$$
L(N,E)=e^d\cdot N^a\cdot E^{\,b+c\log N}
$$

这一步很重要，因为它把“专家数量怎么起作用”写得更直观了：

- $N^a$ 表示 dense 规模扩展的主效应
- $E^{b+c\log N}$ 表示专家数量的效应
- 而且这个效应不是常数，它会随着 $N$ 变化

这正是 MoE 和普通 dense scaling 不同的地方。  
在普通 dense 模型里，你通常只问“把模型做大，loss 按什么幂律下降”；在这里，你还要问“**同样是加专家，这个收益会不会随着已有模型规模变化而变化**”。

### 1. 为什么交互项代表“收益递减”

对 $\log E$ 求偏导，可以得到：

$$
\frac{\partial \log L}{\partial \log E}=b+c\log N
$$

因为我们关心的是“增加专家后，loss 会下降多少”，所以更方便看它的相反数：

$$
-\frac{\partial \log L}{\partial \log E}=-(b+c\log N)
$$

如果经验拟合满足：

- $b<0$，表示在小到中等规模区域，增加专家通常会降低 loss
- $c>0$，表示随着 $N$ 变大，$b+c\log N$ 会逐渐变得“没那么负”

那么就意味着：

**对更大的基础模型来说，再去翻倍专家数，收益虽然仍可能存在，但单位对数尺度下的改善会越来越小。**

白话解释如下。

- 当基础模型还不大时，加专家相当于快速增加“可供路由选择的模式库”，收益很直接
- 当基础模型已经很强时，它本身就能表达很多模式，这时再加更多专家，新增专家更多是在拟合长尾细节，边际收益会缩小

这就是“递减区”的来源。

### 2. 一个具体数值例子

为了让这个式子更直观，下面给一个教学级数字例子。设：

$$
a=-0.08,\quad b=-0.03,\quad c=0.004,\quad d=1.2
$$

再设 $N=1.3\times10^9$。因为 $\log N$ 很大，$b+c\log N$ 仍可能是负值，但绝对值不大。于是：

- 从 $E=8$ 增到 $E=64$，loss 往往能明显下降
- 从 $E=64$ 增到 $E=256$，loss 可能还降，但不会像前一段那样显著

这和论文里常见的趋势图是同一个意思：  
**前期扩专家很有用，后期继续扩专家，收益更像“磨损式下降”，而不是“线性赚收益”。**

### 3. Effective Parameter Count 的含义

Clark 还定义了 Effective Parameter Count，可以翻成“有效参数规模”。它想回答的问题是：

**一个 routed model，如果只从 loss 表现看，相当于多大的 dense 模型？**

先把一个等效 dense 模型写成：

$$
\log L_{dense}(\bar N)=a\log \bar N+d
$$

再让 routed model 和这个等效 dense model 的 loss 对齐：

$$
a\log \bar N+d=a\log N+b\log E+c\log N\log E+d
$$

整理后得到：

$$
\log \bar N=\log N+\frac{b+c\log N}{a}\log E
$$

于是：

$$
\bar N(N,E)=N\cdot E^{\frac{b+c\log N}{a}}
$$

这个式子最容易被误读成“专家数翻倍，有效参数也差不多翻倍”。这其实不对。原因就在指数项：

$$
\frac{b+c\log N}{a}
$$

它不是常数，甚至会随着 $N$ 改变。因此：

**MoE 里的“等效 dense 提升”不是线性换算，而是一个受交互项控制的非线性映射。**

这也是为什么“总参数数百亿甚至上千亿”并不自动等于“训练收益也同比上去”。

### 4. 为什么还要看粒度 $G$

只讨论专家数 $E$ 仍然不够。Fine-grained MoE 的工作进一步提醒：  
**专家的大小，也是一条独立扩展轴。**

常见定义是：

$$
G=\frac{d_{ff}}{d_{expert}}
$$

其中：

- $d_{ff}$ 是原始 dense FFN 的中间层宽度
- $d_{expert}$ 是单个专家的宽度
- $G$ 越大，表示专家切得越细、越窄

另一个常见量是总扩张率：

$$
R=\frac{N_{MoE}}{N_{ff}}
$$

它表示 MoE 相对于原始 dense FFN，在总参数上扩了多少。

这两个量为什么重要？因为“专家数量”只回答了“分成多少份”，但没有回答“每一份有多大”。  
下面这个对比最容易理解：

| 情况 | 直观后果 |
|---|---|
| 专家少且很宽 | 更像少数几个大块专家，稀疏化灵活性有限 |
| 专家多且适中 | 往往更容易形成稳定分工 |
| 专家极多且很窄 | 通信、调度、容量浪费可能变重 |

所以 MoE 的 scaling 不是单变量问题，而是三变量耦合问题：

- $N$ 决定“单个 token 实际走过的有效容量”
- $E$ 决定“总共能提供多少专家槽位”
- $G$ 决定“专家是粗切还是细切”

可以把它们画成一个简图：

```text
总训练预算
   |
   +--> 选择 N：每个 token 实际经过的参数规模
   |
   +--> 选择 E：系统里有多少个专家
   |
   +--> 选择 G：每个专家切得多细
            |
            +--> 影响负载均衡、通信开销、容量浪费和收益拐点
```

因此，真正的工程问题不是“把 $E$ 拉到多大”，而是：

**在固定预算下，怎样联合选择 $N$、$E$、$G$，让路由系统能稳定地把稀疏参数转成真实收益。**

---

## 代码实现

下面给一个可以直接运行的教学版 top-1 MoE 示例。它不依赖深度学习框架，只保留四个关键部分：

- Clark 形式的教学版 loss 计算
- 有效参数规模估计
- top-1 路由
- capacity 限制与辅助损失

代码不是生产级训练器，但逻辑和真实系统是一一对应的。

```python
import math
from typing import List, Dict, Optional


def clark_log_loss(
    N: float,
    E: int,
    a: float = -0.08,
    b: float = -0.12,
    c: float = 0.004,
    d: float = 1.2,
) -> float:
    """
    教学版统一 scaling 公式，使用自然对数。

    ln L(N, E) = a ln N + b ln E + c ln N ln E + d

    参数选择满足：
    - a < 0: N 增大时，loss 倾向下降
    - b < 0: E 增大时，loss 倾向下降
    - c > 0: 随着 N 变大，扩专家的收益逐渐变缓
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if E <= 0:
        raise ValueError("E must be positive")

    lnN = math.log(N)
    lnE = math.log(E)
    return a * lnN + b * lnE + c * lnN * lnE + d


def clark_loss(
    N: float,
    E: int,
    a: float = -0.08,
    b: float = -0.12,
    c: float = 0.004,
    d: float = 1.2,
) -> float:
    """返回正数 loss。"""
    return math.exp(clark_log_loss(N, E, a=a, b=b, c=c, d=d))


def expert_gain_slope(
    N: float,
    b: float = -0.12,
    c: float = 0.004,
) -> float:
    """
    返回 -∂ log L / ∂ log E，即“沿 log E 方向增加专家的局部收益”。
    数值越大，说明继续增加专家越值得。
    """
    return -(b + c * math.log(N))


def effective_parameter_count(
    N: float,
    E: int,
    a: float = -0.08,
    b: float = -0.12,
    c: float = 0.004,
) -> float:
    """
    等效 dense 参数规模：
        N_bar = N * E^((b + c ln N) / a)
    """
    exponent = (b + c * math.log(N)) / a
    return N * (E ** exponent)


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def route_tokens_top1(
    token_logits: List[List[float]],
    num_experts: int,
    capacity_factor: float = 1.0,
    alpha: float = 0.01,
) -> Dict[str, object]:
    """
    token_logits:
        shape = [num_tokens, num_experts]
        每个 token 对每个 expert 的路由打分

    返回：
    - loads: 每个专家实际接收的 token 数
    - dropped: 因容量限制被丢弃的 token 数
    - assignments: 每个 token 被分到哪个专家，若丢弃则为 None
    - router_probs_mean: 路由器在每个专家上的平均概率
    - aux_loss: Switch 风格的简化负载均衡损失
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if not token_logits:
        raise ValueError("token_logits must not be empty")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")

    num_tokens = len(token_logits)

    for row in token_logits:
        if len(row) != num_experts:
            raise ValueError("Each logit row must have length == num_experts")

    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)

    loads = [0] * num_experts
    router_prob_sum = [0.0] * num_experts
    assignments: List[Optional[int]] = []
    dropped = 0

    for logits in token_logits:
        probs = softmax(logits)

        for i, p in enumerate(probs):
            router_prob_sum[i] += p

        top1 = max(range(num_experts), key=lambda i: probs[i])

        if loads[top1] < capacity:
            loads[top1] += 1
            assignments.append(top1)
        else:
            assignments.append(None)
            dropped += 1

    token_fraction = [load / num_tokens for load in loads]
    router_probs_mean = [p / num_tokens for p in router_prob_sum]

    # 简化版 Switch 辅助损失：
    # alpha * E * sum(f_i * P_i)
    aux_loss = alpha * num_experts * sum(
        fi * pi for fi, pi in zip(token_fraction, router_probs_mean)
    )

    return {
        "capacity": capacity,
        "loads": loads,
        "dropped": dropped,
        "assignments": assignments,
        "token_fraction": token_fraction,
        "router_probs_mean": router_probs_mean,
        "aux_loss": aux_loss,
    }


def main() -> None:
    N = 1.3e9

    loss_e8 = clark_loss(N=N, E=8)
    loss_e64 = clark_loss(N=N, E=64)
    loss_e256 = clark_loss(N=N, E=256)

    assert loss_e64 < loss_e8
    assert loss_e256 < loss_e64
    assert (loss_e8 - loss_e64) > (loss_e64 - loss_e256)  # 收益递减

    token_logits = [
        [5.0, 1.0, 0.5, 0.2],
        [4.8, 1.1, 0.4, 0.3],
        [4.9, 0.9, 0.6, 0.2],
        [5.1, 0.8, 0.3, 0.1],
        [5.2, 0.7, 0.4, 0.2],
        [5.0, 0.9, 0.5, 0.1],
        [4.7, 1.2, 0.6, 0.2],
        [4.9, 1.0, 0.5, 0.3],
    ]

    routing = route_tokens_top1(
        token_logits=token_logits,
        num_experts=4,
        capacity_factor=1.0,
        alpha=0.01,
    )

    assert routing["capacity"] == 2
    assert routing["dropped"] > 0
    assert sum(routing["loads"]) + routing["dropped"] == len(token_logits)

    print("loss(E=8)   =", round(loss_e8, 6))
    print("loss(E=64)  =", round(loss_e64, 6))
    print("loss(E=256) =", round(loss_e256, 6))
    print("expert gain slope =", round(expert_gain_slope(N), 6))
    print("effective params @ E=64 =", int(effective_parameter_count(N, 64)))
    print("loads =", routing["loads"])
    print("dropped =", routing["dropped"])
    print("aux_loss =", round(routing["aux_loss"], 6))


if __name__ == "__main__":
    main()
```

如果运行这段代码，会看到两个最关键的现象。

第一，`E` 从 8 增到 64，再到 256，loss 会继续下降，但下降幅度不是线性的，后面那一段改善更小。  
第二，虽然总共有 4 个专家，但因为所有 token 都明显偏向第 0 个专家，所以在 `capacity_factor=1.0` 时会发生 overflow，部分 token 被丢弃。

这正对应了真实训练里的两类核心问题：

1. **scaling 问题**：专家数扩展的收益是否已经进入递减区  
2. **systems 问题**：路由是否稳定，容量是否足够，负载是否均衡

为了帮助新手把代码和概念对应起来，可以对照下表看：

| 代码对象 | 对应概念 | 作用 |
|---|---|---|
| `clark_loss` | 经验 scaling law | 观察 $N$、$E$ 对 loss 的影响 |
| `expert_gain_slope` | 扩专家的局部收益 | 判断继续加专家值不值得 |
| `effective_parameter_count` | 有效参数规模 | 把 routed model 映射成等效 dense 规模 |
| `route_tokens_top1` | top-1 路由 | 模拟每个 token 只选 1 个专家 |
| `capacity` | 容量上限 | 控制每个专家最多接收多少 token |
| `aux_loss` | 负载均衡损失 | 避免热点专家长期过载 |

真实工程里的训练循环通常还会额外监控这些指标：

- 每个专家的 token 占比
- overflow 比例，也就是 token drop 比例
- router entropy，路由熵，用来观察分配是否过度集中
- 各专家激活是否长期塌缩
- expert load 的时间序列，而不只是单步快照

下面这张表可以直接当作调参入口：

| 变量 | 作用 | 初始建议 |
|---|---|---|
| $E$ | 决定总专家数 | 先试 32、64、128 |
| $G$ | 决定专家粒度 | 不要默认固定不变 |
| `capacity factor` | 决定每个专家的缓冲容量 | 先保证 drop 不高，再谈更大稀疏度 |
| `auxiliary loss` 系数 | 决定均衡力度 | 太小会热点，太大会干扰主目标 |
| router temperature | 影响路由分布尖锐程度 | 过尖锐更容易热点 |
| batch/token 数 | 决定每步可分配样本量 | 太小会让负载统计更不稳定 |

---

## 工程权衡与常见坑

MoE 最常见的错误，不是“不会实现路由器”，而是“把专家数当成唯一扩展轴”。

### 1. token drop 过多

容量公式通常近似为：

$$
\text{capacity}\approx
\frac{\text{tokens per batch}}{E}\times \text{capacity factor}
$$

它表达的是一个很朴素的事实：  
专家越多，平均每个专家分到的理论 token 数越少；但如果路由不均匀，热门专家依然会过载。

一旦某个专家先满载，后来的 token 就会：

- 被直接丢弃
- 或者被发送到备选专家
- 或者触发更复杂的回退路径

无论是哪一种，都会削弱训练信号的质量。于是你会看到一个很反直觉的现象：

**总参数明明更大了，但有效学习反而变差。**

### 2. 热点专家

所谓热点专家，就是少数专家长期很忙，而很多专家长期几乎不工作。  
这在训练初期尤其常见，因为路由器很容易把还不稳定的偏好迅速放大。

Switch Transformer 的辅助损失可以写成简化形式：

$$
\text{loss}_{aux}=\alpha \cdot E \sum_{i=1}^{E} f_i P_i
$$

其中：

- $f_i$ 是实际分配到专家 $i$ 的 token 占比
- $P_i$ 是路由器在专家 $i$ 上分配的平均概率
- $\alpha$ 是辅助损失系数

这个量的目标不是“让所有专家绝对一样忙”，而是避免分配极度失衡。因为一旦热点专家形成，系统会同时出现两类问题：

- 训练层面：部分专家过拟合、部分专家学不到东西
- 系统层面：通信压力集中、overflow 增加、吞吐恶化

### 3. 粒度失调

只把 $E$ 从 128 拉到 512，但不重新思考单个专家的宽度，经常会得到一个“参数看起来更大，但系统和优化都更差”的模型。

这是因为专家太细时，问题不再只是建模能力，而会变成：

- dispatch 开销更高
- all-to-all 通信更重
- 单个专家吞吐更差
- 容量浪费更明显
- 路由噪声更容易压过真实收益

所以 Fine-grained MoE 的价值不在于“专家越细越先进”，而在于提醒你：

**专家个数和专家宽度必须联动设计。**

### 4. 把训练加速误读成推理加速

MoE 的优势首先是训练侧优势。  
原因是训练时你可以在接近固定每 token FLOPs 的条件下，让更多参数参与整体学习。

但推理期是否同样受益，要看：

- 硬件是否对稀疏 dispatch 友好
- batch 形态是否足够大
- expert 并行与通信是否高效
- 在线服务是否允许这种延迟波动

因此，“训练更省计算”不等于“推理更低延迟”。  
很多线上系统最后会收敛到一个折中方案：专家数量不追求极大，而是控制在一个硬件能稳定支撑的范围内。

下面这张表适合直接拿来排障：

| 常见坑 | 现象 | 常见原因 | 规避动作 |
|---|---|---|---|
| token drop 过多 | loss 不降或波动大 | `capacity factor` 太小，或路由过偏 | 提高容量，增强均衡约束 |
| 热点专家 | 少数专家 load 极高 | router 早期塌缩，分布过尖锐 | 增大 `aux loss`，调温度，检查初始化 |
| 专家越多越慢 | wall-clock 不升反降 | 通信与调度开销超过收益 | 先验证 64/128，再决定是否到 256+ |
| 稀疏参数大但收益小 | 验证集改善有限 | 数据不足或粒度不合理 | 同时调数据量、$E$、$G$ |
| 推理延迟偏高 | 服务抖动大 | sparse dispatch 不友好 | 减少专家数，适当增宽 expert |
| 负载波动很大 | 每步 load 分布不稳定 | batch 太小或路由器太敏感 | 增大 batch，平滑路由分布 |

一个很典型的错误场景是：  
有人把 $E$ 从 128 提到 512，但 `capacity_factor` 不变，路由器也没有重新调，结果热门专家 overflow 更严重，all-to-all 更重，最终训练速度和稳定性一起变差。

这时正确动作通常不是“继续堆专家”，而是：

1. 回退到 64 或 128  
2. 检查负载均衡指标  
3. 重新评估粒度 $G$  
4. 再决定是否值得继续扩专家

---

## 替代方案与适用边界

如果硬件、框架或者服务形态并不适合稀疏 dispatch，MoE 不是唯一方案。

第一种替代方案是 **dense scaling**。  
也就是继续使用普通 Transformer，只扩展宽度、层数或上下文长度。它的优点是实现简单、硬件友好、推理行为稳定；缺点是参数和 FLOPs 强绑定，扩一点通常就要真实多算很多。

第二种是 **coarse-grained MoE**。  
也就是专家数量相对较少、每个专家较宽的方案。这类方法保留了一部分稀疏优势，同时把系统复杂度控制在比较可接受的范围内，适合“训练收益想要一些，但工程风险不能太高”的场景。

第三种是 **fine-grained MoE**。  
它允许更灵活地控制专家粒度 $G$。优点是理论上更有机会逼近 compute-optimal 区域；缺点是系统实现复杂度更高，对通信、并行策略和路由稳定性要求更高。

下表适合做方案对比：

| 方案 | FLOPs 特征 | 延迟特征 | scaling 收益 | 适用边界 |
|---|---|---|---|---|
| dense | 参数与 FLOPs 强绑定 | 最稳定 | 可靠但昂贵 | 小中规模训练，在线推理优先 |
| coarse-grained MoE | 训练 FLOPs 可控 | 取决于通信实现 | 通常明显 | 训练加速优先，工程复杂度可接受 |
| fine-grained MoE | 计算更可塑 | 系统要求最高 | 理论最优空间更大 | 大规模训练，愿意做系统调优 |

可以把选择规则压缩成两条。

第一条，如果目标是**最小工程风险**，优先 dense 或者专家数较少的 coarse-grained MoE。  
第二条，如果目标是**在大预算下追求更强的 compute-optimal scaling**，就必须把 $E$ 和 $G$ 一起调，而不是只调 $E$。

对初学者，一个实用起点是先记住两个量：

$$
R=\frac{N_{MoE}}{N_{ff}},\qquad G=\frac{d_{ff}}{d_{expert}}
$$

然后按下面顺序调参：

1. 先固定数据量和训练预算  
2. 再试 $E\in\{32,64,128\}$  
3. 观察 load balance、drop rate 和 wall-clock  
4. 最后再调 $G$，看更细专家是否真的带来净收益

如果硬件对稀疏计算不友好，可以退到“更少专家 + 更宽 expert”的折中方案。  
这样做的逻辑不是保守，而是务实：在很多系统里，这种折中比“极端增大专家数”更容易把理论收益落到真实吞吐和真实 loss 上。

---

## 参考资料

- Aidan Clark, Diego de las Casas, Aurelia Guy, et al. *Unified Scaling Laws for Routed Language Models*. ICML 2022. <https://proceedings.mlr.press/v162/clark22a.html>
- William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 2022. <https://jmlr.org/papers/v23/21-0998.html>
- Jan Ludziejewski, Maksym Andriushchenko, Artem Chernodub, et al. *Scaling Laws for Fine-Grained Mixture of Experts*. ICML 2024. <https://proceedings.mlr.press/v235/ludziejewski24a.html>
- Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. <https://research.google/pubs/pub45929/>
- Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR 2021. <https://openreview.net/forum?id=qrwe7XHTmYb>

## 核心结论

Expert Capacity 是 MoE 中每个专家在一次前向计算里最多接收多少个 token 的上限。MoE 是“混合专家模型”，即在很多专家子网络里只激活少数几个，让模型总参数量很大，但单次计算量仍然可控。这个上限通常写成：

$$
C=\left\lfloor \text{CF}\times\frac{T}{N}\right\rfloor
$$

其中，$T$ 是当前分组内的 token 总数，$N$ 是专家数，CF 是 capacity factor，中文常译为“容量因子”。直观理解上，$\frac{T}{N}$ 是理想均匀分配时每个专家应接收的平均 token 数，而 CF 表示在这个平均值之上预留多少缓冲。

结论先说清楚：

1. 当某个专家实际收到的 token 数 $T_i$ 超过容量 $C$ 时，多出来的 token 不再进入该专家计算，而是沿残差连接直接跳过这一层。
2. CF 调大，token drop 会减少，但计算、激活、通信、缓存占用会近似线性上升。
3. CF 调小，吞吐更高、成本更低，但路由稍微不均衡就会出现明显的信息丢失。
4. 在真实工程里，容量问题通常不是“公式算错”，而是“路由分布不均”。路由器把太多 token 发给少数专家时，即使平均容量足够，也会局部溢出。
5. ST-MoE 的经验很直接：训练阶段常用较小 CF 追求效率，例如 1.25；推理阶段可把 CF 提高到 2.0，减少 drop，换更稳定的质量。

可以把专家理解为一排有固定槽位的处理单元，token 是待处理任务，路由器负责把任务分发给不同专家。CF 越大，每个专家可接收的槽位越多；一旦槽位被占满，后续 token 就不能再进入该专家，只能走残差主干继续向后传播。这里的关键不是“token 被删除了”，而是“该 token 在这一层失去了专家变换”。

---

## 问题定义与边界

先把术语压实，否则后面容易混淆。

- token：模型处理的最小离散单位，可以近似理解为“切分后的词片段”。
- expert：MoE 层中的子网络，通常是若干个并行 FFN。
- router：路由器，负责为每个 token 计算各专家分数，并选择 Top-k 专家。
- capacity：每个专家在当前分组中最多可处理多少 token。
- overflow：被某个专家选中，但因该专家已满而无法进入计算的那部分 token。

常用变量如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $T$ | 组内 token 总数 | 当前这小批数据一共有多少 token |
| $N$ | 专家数量 | 当前 MoE 层里有多少个专家 |
| CF | 容量因子 | 在平均负载上预留多少缓冲 |
| $C$ | 单专家容量 | 每个专家最多接收多少 token |
| $T_i$ | 第 $i$ 个专家实际收到的 token 数 | 第 $i$ 个专家被分配了多少任务 |
| $O_i$ | overflow | 第 $i$ 个专家超出容量的数量 |
| $r$ | drop rate | 被容量机制回退的 token 占比 |

核心边界有四个：

1. 容量是按组计算的，不是按整个训练集计算的。不同 batch、不同 micro-batch、不同并行切分下，$T$ 都可能变化。
2. 路由是离散分配，不是连续平均分配。即使理论均值是 $\frac{T}{N}$，实际的 $T_i$ 也会波动。
3. 在因果语言模型中，token 顺序有语义意义，因此 drop 不是纯噪声，而是某些位置少经过了一次专家变换。
4. 容量机制只决定“token 能否进入被选中的专家”，不改变残差主干仍然存在这一事实。

溢出的定义很直接：

$$
O_i=\max(0, T_i-C)
$$

总 overflow 则是：

$$
O_{\text{total}}=\sum_{i=1}^{N} O_i
$$

如果进一步定义丢弃率：

$$
r=\frac{O_{\text{total}}}{T}
=\frac{1}{T}\sum_{i=1}^{N}\max(0,T_i-C)
$$

这个式子说明，drop rate 不是只由 CF 决定，也不是只由专家数决定，而是由“容量大小”和“路由偏斜程度”共同决定。

看一个最小例子。设当前组内有 $T=8$ 个 token、$N=4$ 个专家、CF=1.25，则：

$$
C=\left\lfloor 1.25\times\frac{8}{4}\right\rfloor
=\left\lfloor 2.5\right\rfloor
=2
$$

也就是说，每个专家本轮最多接收 2 个 token。若某个专家被分到 3 个 token，则有 1 个 token 会溢出，无法进入该专家。

这里有一个新手常见误区：公式里算出的是浮点数，但实现里一定要落成离散槽位，因此最终容量几乎总是整数。不同实现可能用 `floor`、`ceil` 或额外加最小保底，但本质都一样，核心目标都是把连续容量变成离散的可分配槽位。

---

## 核心机制与推导

MoE 层中的典型流程可以写成：

`token 输入 -> 路由器打分 -> 选择 Top-k 专家 -> 按优先级尝试入槽 -> 能入槽则执行专家 FFN -> 不能入槽则走残差`

如果只看 Top-1 路由，流程最容易理解：每个 token 只选择一个专家，然后检查该专家是否还有剩余容量。

容量公式之所以写成

$$
C=\left\lfloor \text{CF}\times\frac{T}{N}\right\rfloor
$$

原因是：

1. $\frac{T}{N}$ 表示理想均匀负载。
2. CF 提供对路由偏斜的容忍空间。
3. 整数化后的 $C$ 才能对应实际内存槽位、通信桶和专家输入张量的尺寸。

如果所有专家都完全均匀，那么 $T_i \approx \frac{T}{N}$，此时 CF 只要略高于 1 就足够。但真实模型中，路由器常常会偏爱某些专家，形成热点。于是即使平均容量够，局部仍可能溢出。

继续用上面的玩具例子。设 8 个 token 的路由结果如下：

| 专家 | 收到 token 数 $T_i$ | 容量 $C$ | overflow $O_i$ |
|---|---:|---:|---:|
| E0 | 3 | 2 | 1 |
| E1 | 2 | 2 | 0 |
| E2 | 2 | 2 | 0 |
| E3 | 1 | 2 | 0 |

则总 drop rate 为：

$$
r=\frac{1}{8}=12.5\%
$$

如果把 CF 从 1.25 调到 2.0，则：

$$
C=\left\lfloor 2.0\times\frac{8}{4}\right\rfloor=4
$$

这时每个专家最多可接收 4 个 token，上表中的 E0 也不会溢出，于是：

$$
r=0
$$

这正是工程上常见的现象：CF 提高后，质量更稳，但资源开销会同步上涨。

还可以把这个过程写成更一般的判断逻辑。对任意一个被分到专家 $i$ 的 token：

$$
\text{dispatch}(x)=
\begin{cases}
\text{Expert}_i(x), & \text{if } \text{load}_i < C \\
x, & \text{otherwise}
\end{cases}
$$

这里的 $x$ 表示输入 token 的隐藏状态。第二种情况里的 $x$ 并不是“什么都没做”，而是“这一层专家变换被跳过，直接保留原表示继续向后流动”。从数值图上看路径不断，从功能图上看该 token 少经历了一次专门处理。

如果是 Top-2 路由，还要再多注意一点：一个 token 可能被两个专家同时选中，而容量限制会分别作用在两个专家上。此时“drop”既可能是两个专家都满，也可能是主专家满、次专家未满，具体取决于实现如何调度和回退。这个细节解释了为什么不同论文都在讲 capacity，但代码行为并不总是完全相同。

真实工程里的经验也由此而来。ST-MoE 在训练阶段常采用较小 CF，比如 1.25，优先保证吞吐、显存和通信成本；到了推理阶段，常把 CF 提高到 2.0，以减少本该进入专家却被挡在门外的 token。这不是理论变化，而是优化目标变化：训练偏效率，推理偏质量。

---

## 代码实现

下面给出一个可直接运行的 Python 玩具实现。它显式模拟了四件事：

1. 容量计算。
2. token 按专家分发。
3. 专家满载时回退到残差。
4. 输出可检查的统计信息。

```python
from math import floor
from typing import List, Dict, Any


def moe_dispatch(assignments: List[int], num_experts: int, cf: float) -> Dict[str, Any]:
    """
    assignments[j] 表示第 j 个 token 被路由到的专家 id。
    这里用 Top-1 路由做最小实现，便于把容量与 overflow 逻辑看清楚。
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if cf < 0:
        raise ValueError("cf must be non-negative")

    group_tokens = len(assignments)
    capacity = floor(cf * group_tokens / num_experts)

    expert_load = [0] * num_experts
    processed = [[] for _ in range(num_experts)]
    residual = []
    overflow = [0] * num_experts

    for token_id, expert_id in enumerate(assignments):
        if not 0 <= expert_id < num_experts:
            raise ValueError(f"invalid expert id {expert_id} for token {token_id}")

        if expert_load[expert_id] < capacity:
            processed[expert_id].append(token_id)
            expert_load[expert_id] += 1
        else:
            residual.append(token_id)
            overflow[expert_id] += 1

    drop_rate = len(residual) / group_tokens if group_tokens else 0.0

    return {
        "group_tokens": group_tokens,
        "capacity": capacity,
        "expert_load": expert_load,
        "processed": processed,
        "residual": residual,
        "overflow": overflow,
        "drop_rate": drop_rate,
    }


def pretty_print(result: Dict[str, Any]) -> None:
    print(f"group_tokens = {result['group_tokens']}")
    print(f"capacity     = {result['capacity']}")
    print(f"expert_load  = {result['expert_load']}")
    print(f"processed    = {result['processed']}")
    print(f"residual     = {result['residual']}")
    print(f"overflow     = {result['overflow']}")
    print(f"drop_rate    = {result['drop_rate']:.4f}")


if __name__ == "__main__":
    # 玩具例子：T=8, N=4, CF=1.25
    result = moe_dispatch(
        assignments=[0, 0, 0, 1, 1, 2, 2, 3],
        num_experts=4,
        cf=1.25,
    )

    assert result["capacity"] == 2
    assert result["overflow"][0] == 1
    assert result["residual"] == [2]
    assert abs(result["drop_rate"] - 0.125) < 1e-9

    pretty_print(result)
```

运行后可得到如下结果：

```text
group_tokens = 8
capacity     = 2
expert_load  = [2, 2, 2, 1]
processed    = [[0, 1], [3, 4], [5, 6], [7]]
residual     = [2]
overflow     = [1, 0, 0, 0]
drop_rate    = 0.1250
```

这段代码里最值得对照理解的是下面几行：

| 代码 | 对应概念 | 说明 |
|---|---|---|
| `capacity = floor(cf * group_tokens / num_experts)` | 容量公式 | 把理论容量落成整数槽位 |
| `expert_load[expert_id] < capacity` | 是否还能入槽 | 专家未满则允许进入 |
| `processed[expert_id].append(token_id)` | 真正进入专家 | token 会执行专家计算 |
| `residual.append(token_id)` | 残差回退 | token 跳过当前专家层 |
| `overflow[expert_id] += 1` | 溢出统计 | 记录热点专家 |
| `drop_rate = len(residual) / group_tokens` | 丢弃率 | 调参时最常看的指标之一 |

如果想进一步贴近真实实现，通常还会增加三类逻辑：

1. 先按路由分数排序，高分 token 优先占用容量。
2. Top-k 路由时，一个 token 可能尝试多个专家。
3. 统计维度更细，不只看全局平均，还要看“按层、按 batch、按专家”的分布。

下面给一个更贴近工程诊断的表，帮助理解“该看什么指标”：

| 诊断指标 | 代表问题 | 如果异常，通常意味着什么 |
|---|---|---|
| `drop_rate` | 总体回退比例 | CF 过低或路由严重不均 |
| `max(expert_load)` | 最热点专家负载 | 有专家被集中打爆 |
| `overflow[i]` | 单专家溢出量 | 路由器偏爱某些专家 |
| `std(expert_load)` | 负载离散程度 | 均衡性差，平均值失真 |
| 分层统计 | 哪一层最脆弱 | 某些层路由器更不稳定 |

对新手来说，最重要的理解是：这段代码不是在模拟“删除 token”，而是在模拟“token 失去一次本应发生的专家加工”。这正是为什么 drop 往往会影响模型质量。

---

## 工程权衡与常见坑

CF 的本质是“用资源换更少的信息损失”。下面这张表是最实用的直觉总结：

| CF 设置 | drop 影响 | 资源开销 | 推荐场景 |
|---|---|---|---|
| `<1` | 很容易明显上升 | 最省计算与通信 | 极端吞吐实验，通常需配动态策略 |
| `1.0` | 对路由均衡很敏感 | 基线成本 | 路由较稳时可尝试 |
| `1.25` | 常见折中点 | 略高于基线 | 训练常用 |
| `2.0` | drop 显著减少 | 成本明显上升 | 推理或高质量评估 |

为什么成本会随着 CF 提高而上升？可以拆成三部分看：

| 成本项 | 为什么会变高 |
|---|---|
| 计算 | 更多 token 真正进入专家 FFN |
| 激活/缓存 | 需要为更多专家输入与输出保留张量空间 |
| 通信 | all-to-all 里被交换的 token 更多 |

常见坑主要有五个。

第一，只看平均负载，不看尾部专家。  
平均每个专家收到 20 个 token，不代表没有专家收到 40 个。容量溢出永远发生在最拥挤的专家上，而不是发生在平均数上。

第二，把 drop 当成无害近似。  
残差确保了主干路径不断，但不意味着功能等价。对于本来应该接受某类专家特化处理的 token，跳过专家层就是信息损失。

第三，训练和推理使用同一 CF。  
训练更在意吞吐与成本，推理更在意质量与稳定性，两者目标不同，容量设置没必要完全一致。

第四，忽略通信成本。  
MoE 的瓶颈不只在 FFN 计算，还常在 token 分发与聚合。CF 提高后，通信流量通常也会上升。

第五，低 CF 下忽略顺序偏差。  
如果实现是“谁先到谁先占位”，那么序列前面的 token 更容易进专家，后面的 token 更容易被回退。这种偏差在因果模型里尤其危险，因为它会和位置顺序耦合，形成系统性失真。

为了更直观，可以看下面这个例子：

| token 顺序 | 路由到 E0 | E0 是否还有容量 | 结果 |
|---|---|---|---|
| token 0 | 是 | 有 | 进入专家 |
| token 1 | 是 | 有 | 进入专家 |
| token 2 | 是 | 无 | 被回退 |
| token 3 | 是 | 无 | 被回退 |

如果 E0 的容量只有 2，那么后来的 token 即使分数更高，也可能因为“来晚了”而失去专家处理机会。这就是为什么一些论文会引入优先级路由。

ST-MoE 附录里提出的 BPR，Batch Prioritized Routing，就是为了解决这个问题。它的核心思想不是按原始顺序填满专家，而是按某种优先级分配有限容量，例如优先让路由分数更高的 token 先占位。这样做的收益不是“凭空增大容量”，而是在相同容量下减少低价值分配。

从工程视角看，BPR 的意义非常明确：

1. 它不能消除容量上限。
2. 它能改善有限容量的使用顺序。
3. 它尤其适合低 CF 场景，因为这时每个槽位都更贵。

---

## 替代方案与适用边界

除了静态 CF，还可以考虑其他策略。常见方案如下：

| 方案 | 核心思想 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| 静态 CF | 所有专家固定容量 | 实现简单，延迟稳定 | 容易受路由偏斜影响 | 大多数基础实现 |
| BPR | 按优先级分配容量 | 低 CF 下 drop 更可控 | 需要排序与额外逻辑 | 训练效率敏感场景 |
| Token Retry | 溢出后尝试次优专家 | 减少直接丢弃 | 增加路由与通信延迟 | 延迟不敏感系统 |

如果再抽象一层，这三种方法分别是在回答三个不同问题：

| 方法 | 它在解决什么问题 |
|---|---|
| 静态 CF | 给每个专家固定多少槽位 |
| BPR | 槽位先给谁 |
| Token Retry | 主专家满了以后怎么办 |

因此它们并不是完全互斥的。一个系统可以同时使用“静态容量 + BPR”，也可以在此基础上再增加“次优专家重试”。

动态容量则是更进一步的思路。可以写成：

$$
C_i = C_{\text{base}} + \Delta_i
$$

其中，$C_{\text{base}}$ 是基准容量，$\Delta_i$ 表示根据当前 batch 的拥塞状态、路由分数分布或历史负载，对第 $i$ 个专家做的动态修正。

这类方法的优点是能更好处理局部热点，但也有明确代价：

| 维度 | 动态容量的影响 |
|---|---|
| 质量 | 热点专家更不容易被打爆 |
| 延迟 | 更难预测，尾延迟可能波动 |
| 实现 | 调度逻辑更复杂 |
| 调优 | 需要更多监控和更多超参数 |

因此，动态容量不是“白送收益”，而是把问题从“固定容量够不够”转成“如何在专家之间重新分配有限容量”。

实际选择时，可以用下面这组判断：

1. 如果优先级是实现简单、线上延迟稳定，静态 CF 仍然是首选。
2. 如果必须把 CF 压得很低，又担心顺序偏差导致无意义 drop，优先考虑 BPR。
3. 如果系统调度能力强、延迟预算宽松，才考虑 Token Retry 或更复杂的动态容量策略。

从决策逻辑上说，真正该先回答的问题不是“CF 该设多少”，而是：

- 你更怕质量下降，还是更怕成本上升？
- 你更怕峰值延迟，还是更怕平均吞吐下降？
- 你当前的问题是容量太小，还是路由太偏？

这三个问题决定了你应该调容量、调路由，还是改调度策略。很多容量问题最终并不是靠“把 CF 调大”解决的，而是靠“让 token 分得更均匀”解决的。

---

## 参考资料

| 来源 | 贡献 | 关键章节 |
|---|---|---|
| ST-MoE: Designing Stable and Transferable Sparse Expert Models, Zoph et al. | 给出容量公式、训练与推理中的 CF 经验、BPR 设计 | 容量定义、Appendix D |
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, Fedus et al. | 给出早期大规模稀疏路由实现、capacity 与 token dropping 的经典背景 | MoE 路由与 capacity 相关章节 |
| Sparse Mixture of Experts at Scale, Next Electronics | 用工程视角总结动态容量、通信与部署权衡 | 动态容量与系统代价综述 |

1. Zoph 等，《ST-MoE: Designing Stable and Transferable Sparse Expert Models》  
   链接：https://ar5iv.org/pdf/2202.08906  
   价值：容量公式 $C=\text{CF}\times\frac{T}{N}$、训练/推理 CF 经验、BPR 设计来源。

2. Fedus 等，《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》  
   链接：https://arxiv.org/abs/2101.03961  
   价值：理解大规模稀疏专家模型里的容量限制、token dropping 和路由实现背景。

3. Next Electronics，《Sparse Mixture of Experts at Scale》  
   链接：https://www.next.gr/ai/large-language-models/sparse-mixture-of-experts-at-scale  
   价值：从系统实现角度解释容量、通信和部署成本的关系。

可以直接提炼成三条实践准则：

| Do / Don't | 建议 |
|---|---|
| Do | 先监控每层、每专家的 overflow，而不是只看全局平均 drop |
| Do | 训练优先效率时可从 CF=1.25 起步，推理优先质量时考虑更高 CF |
| Don't | 把残差回退当作“没有损失”的免费兜底 |
| Don't | 在低 CF 场景忽略顺序偏差与路由优先级问题 |

## 核心结论

Hash Layer 是一种用于 MoE（Mixture of Experts，专家混合）的固定路由机制。它不训练一个 router（路由器，用来决定 token 去哪个 expert 的模块），而是把 token 的离散标识直接映射到 expert。最简单的形式是：

$$
e_t = h(x_t) \bmod E
$$

其中：

- $x_t$：第 $t$ 个 token 的 ID
- $h(\cdot)$：哈希函数或等价的固定映射表
- $E$：expert 数量
- $e_t$：该 token 被送往的 expert 编号

它和学习型 router 的区别，不在于“是否使用多个 expert”，而在于“路由规则是否通过梯度学习得到”。Hash Layer 的路由规则在训练前就确定，训练过程中不更新。因此它的直接结果很明确：

| 维度 | Hash Layer | 学习型 Router |
|---|---|---|
| 路由参数 | 0 | 有额外参数 |
| 训练稳定性 | 高 | 常需调 load balancing loss |
| 语义自适应 | 弱 | 强 |

如果再配合 Balanced assignment（平衡分配，指先根据 token 频率做离线分桶，把高频 token 尽量均匀分到不同 expert），Hash Layer 往往能同时得到两类收益：

1. 路由计算几乎没有额外开销。
2. expert 负载分布可控，不容易出现某个 expert 被高频 token 挤爆。

可以把它先理解成一个仓储分拣问题。若每种货物类型都固定走某条传送带，那么分拣时几乎不需要实时决策，系统最简单、延迟最低；但如果销量最高的几类货物恰好都走同一条带，吞吐就会崩掉。因此真正关键的不是“固定映射”这四个字，而是“固定映射之前先做频率平衡”。

结论还要补完整：Hash Layer 的主要弱点，不是它一定比学习型 router 精度差，而是它无法根据上下文语义动态改路由。一个 token 不管出现在什么句子里，原则上都去同一个 expert。只要任务非常依赖语义级动态分流，例如同形异义严重、领域变化快、上下文决定处理方式，那么学习型 router 的上限通常更高。

---

## 问题定义与边界

MoE 的核心问题不是“有没有很多 expert”，而是“一个 batch 里的大量 token 应该怎么分发到 expert，才能既快又稳”。这里的“稳”至少包含两层含义：

1. 不要让某些 expert 长期吃不到 token，导致参数几乎学不到东西。
2. 不要让某个 expert 突然吞下大多数 token，导致容量溢出、token 被丢弃、all-to-all 通信抖动，甚至让整层的吞吐恶化。

学习型 router 的典型做法，是根据 hidden state（隐藏状态，指 token 经过前面层计算后的向量表示）计算每个 expert 的分数，再从中选 top-k。形式上常写成：

$$
p(e \mid h_t) = \mathrm{softmax}(W h_t)
$$

然后取概率最高的一个或几个 expert。它的优点是能看语义，因为 $h_t$ 本身包含上下文信息；缺点是训练时容易出现路由偏置，所以通常还要额外加 auxiliary load loss（负载均衡辅助损失）或其他稳定项，防止某些 expert 被过度偏好。

Hash Layer 直接绕开这一类问题。它不看 hidden state，而只看 token ID，或者由 token ID 派生出的离散特征。这一设计立刻决定了它的边界：

| 项目 | 支持 | 不支持或较弱 |
|---|---|---|
| 输入依据 | token ID、预先离散化特征 | 基于上下文语义的动态路由 |
| 可调参数 | expert 数、哈希策略、预平衡表 | 通过梯度学习路由规则 |
| 负载均衡方式 | 预统计频率后固定分桶 | 训练时在线自适应修正 |
| 适合场景 | 高频词分布稳定的大规模训练 | 强语义歧义、域漂移明显的任务 |

为什么频率统计是前提，而不是“可选优化”？因为自然语言 token 分布通常接近 Zipf 分布。Zipf 分布可以简单理解为：

- 极少数 token 非常常见
- 大量 token 很少出现
- 高频 token 的流量占比，远高于“种类数占比”

这意味着如果你直接使用一个没有预平衡的朴素哈希，例如：

```python
expert = token_id % num_experts
```

那么虽然它是确定性的，但并不保证负载均匀。多个超高频 token 只要碰巧落到同一个 expert，那个 expert 就会长期过载。也就是说：

$$
\text{固定路由} \neq \text{均衡路由}
$$

更准确地说，Hash Layer 解决的是“高吞吐、低开销、低不稳定性”的路由问题；它不解决“按上下文做最优语义分发”的问题。只要任务需求偏向前者，它就很有吸引力；如果需求偏向后者，它就天然受限。

---

## 核心机制与推导

最基础的 Hash Layer 可以写成：

$$
e_t = \mathrm{Hash}(x_t)
$$

更严格一点，是把 token $x_t$ 通过预定义查表或确定性函数映射到 expert 集合中的一个元素：

$$
e_t \in \{0,1,\dots,E-1\}
$$

然后该 token 的隐藏向量 $h_t$ 只经过对应 expert 的 FFN（前馈网络）：

$$
y_t = \mathrm{FFN}_{e_t}(h_t)
$$

整个过程可以拆成两步：

1. 用离散 token 标识决定 expert。
2. 用该 expert 的参数处理隐藏向量。

这里最容易混淆的一点是：决定 expert 的不是 $h_t$，而是 token ID；但真正被 FFN 处理的输入仍然是 $h_t$。因此 Hash Layer 不是“只处理 token ID”，而是“只用 token ID 做路由决策”。

如果写成更完整的函数形式，可以表示为：

$$
g: \mathcal{V} \to \{0,\dots,E-1\}, \quad e_t = g(x_t)
$$

其中 $\mathcal{V}$ 是词表。于是对任意两个位置 $t_1, t_2$，只要它们的 token ID 相同，就有：

$$
x_{t_1} = x_{t_2} \Rightarrow e_{t_1} = e_{t_2}
$$

这正是 Hash Layer 的强约束，也是它的核心限制。相同 token 在不同上下文下不会改路由。

### 玩具例子：A/B/C/D 的平衡分配

假设有 2 个 expert、4 个 token，它们在训练语料中的统计频率如下：

| token | 频率 | 分配后的 expert |
|---|---:|---|
| A | 100 | Expert 0 |
| B | 80 | Expert 1 |
| C | 40 | Expert 1 |
| D | 20 | Expert 0 |

一个简单且实用的 Balanced assignment 流程是：

1. 先按频率从高到低排序。
2. 维护每个 expert 当前已分配的总负载。
3. 每次把当前 token 分给“当前总负载最轻”的 expert。

按这个规则，过程如下：

1. A=100，先给 Expert 0，负载变成 `[100, 0]`
2. B=80，给 Expert 1，负载变成 `[100, 80]`
3. C=40，给更轻的 Expert 1，负载变成 `[100, 120]`
4. D=20，给更轻的 Expert 0，负载变成 `[120, 120]`

最终达到近似完美均衡。对应伪代码是：

```text
sort tokens by frequency descending
loads = [0] * num_experts
for token in sorted_tokens:
    e = argmin(loads)
    assign[token] = e
    loads[e] += freq[token]
```

这个过程本质上是一个离线装桶问题。它不是在每个训练 step 动态学习，而是在训练前先把大部分负载风险处理掉。你可以把它理解为：

- 在线学习型 router：每次发货时临时决定走哪条路
- 离线 Balanced assignment：先把货类和线路规划好，运行时只执行

前者更灵活，后者更稳定。

如果进一步形式化，设词表中 token 的频率为 $f(x)$，映射函数为 $g(x)$，那么 expert $e$ 的期望负载近似为：

$$
L_e = \sum_{x \in \mathcal{V}} f(x)\,\mathbf{1}[g(x)=e]
$$

Balanced assignment 的目标，不是让“token 种类数”均匀，而是让这些 $L_e$ 尽量接近。因为系统真正承受的是流量，而不是词表条目数。

### 为什么这能稳定

Hash Layer 稳定的根本原因，不是哈希本身神奇，而是它把“路由器训练”这个不稳定来源整个拿掉了。

学习型 router 中，路由决策依赖参数 $W$，而 $W$ 会不断更新：

$$
p(e \mid h_t) = \mathrm{softmax}(W h_t)
$$

这意味着：

- 路由行为会随着训练变化
- expert 接收的数据分布会随着训练变化
- 数据分布变化又会反过来影响 expert 的梯度
- 整个系统存在明显的耦合反馈

Hash Layer 中没有这套反馈回路。路由表一旦固定，训练全过程中 expert 流量分布基本稳定。于是会带来几项直接收益：

- 没有 router 前向计算开销
- 没有 router 反向传播
- 不需要额外的 load balancing loss
- expert 流量更容易提前评估和容量规划
- 训练前期不容易出现“某个 expert 学不到东西”的极端情况

当然，这种稳定是通过牺牲动态自适应换来的。假设 token “bank” 在两个上下文中分别表示“银行”和“河岸”，学习型 router 可以依据 hidden state 把它送去不同 expert；Hash Layer 原则上做不到，因为它只看到同一个 token ID。

### MultiHash 的作用

单一哈希的最大问题，是表达能力过于刚性。一个 token 一次只绑定一个 expert，这种离散划分很容易过死。MultiHash 的思路，就是不要把整层参数当成“一个整体 expert”，而是拆成多个 slice（切片、子块），每个子块由不同哈希选择。

可写成类似形式：

$$
\mathrm{FFN}_{MH}(h) = [B_{k_1}(v), B_{k_2}(v), \dots, B_{k_N}(v)]
$$

其中：

- $v$ 是中间表示
- $B_{k_i}$ 是第 $i$ 个被选中的参数块
- $k_i$ 由不同的哈希规则决定
- 方括号表示拼接或组合这些块的输出

直观理解：

- 单 Hash：整件货只进一个仓
- MultiHash：同一件货拆成几个箱子，分别走不同固定通道

这样做仍然不需要训练型 router，但会比“整个 token 永远只去一个 expert”更灵活。它缓解了单次固定分配过硬的问题，也让参数利用率更高。不过要注意，MultiHash 依然不是语义路由。它只是把固定路由从“一次选一个大块”变成“一次选多个小块”。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例。它分成三个部分：

1. 根据 token 频率构建平衡映射表
2. 在运行时按映射表把 token 分到 expert
3. 对比“预平衡映射”和“朴素取模哈希”的负载差异

代码不依赖深度学习框架，只演示 Hash Layer 的核心逻辑。

```python
from collections import Counter
from typing import Dict, List, Callable, Tuple


def build_balanced_hash(freqs: Dict[int, int], num_experts: int) -> Tuple[Dict[int, int], List[int]]:
    """
    根据 token 频率做离线平衡分配。
    返回：
      - table[token_id] = expert_id
      - loads[expert_id] = 该 expert 被分配到的总频率
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    items = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
    loads = [0 for _ in range(num_experts)]
    table = {}

    for token_id, count in items:
        expert_id = min(range(num_experts), key=lambda i: loads[i])
        table[token_id] = expert_id
        loads[expert_id] += count

    return table, loads


def build_mod_hash(vocab: List[int], num_experts: int) -> Dict[int, int]:
    """
    朴素哈希：直接 token_id % num_experts
    """
    return {token_id: token_id % num_experts for token_id in vocab}


def route_tokens(
    tokens: List[int],
    assignment_table: Dict[int, int],
    experts: List[Callable[[float], float]],
    token_embeds: List[float],
) -> Tuple[List[float], List[int]]:
    """
    在线阶段：只查表，不学习。
    """
    if len(tokens) != len(token_embeds):
        raise ValueError("tokens and token_embeds must have the same length")

    outputs = []
    expert_ids = []

    for tok, emb in zip(tokens, token_embeds):
        if tok not in assignment_table:
            raise KeyError(f"token {tok} is not in assignment_table")
        expert_id = assignment_table[tok]
        out = experts[expert_id](emb)
        expert_ids.append(expert_id)
        outputs.append(out)

    return outputs, expert_ids


def estimate_runtime_load(tokens: List[int], assignment_table: Dict[int, int], num_experts: int) -> List[int]:
    """
    估计一段输入序列在运行时落到各 expert 的 token 数。
    """
    loads = [0 for _ in range(num_experts)]
    for tok in tokens:
        loads[assignment_table[tok]] += 1
    return loads


def main() -> None:
    # 假设词表中的 6 个 token 的离线频率统计如下
    freqs = {
        0: 1000,  # 高频 token
        1: 800,
        2: 400,
        3: 200,
        4: 50,
        5: 30,
    }
    num_experts = 2
    vocab = sorted(freqs.keys())

    balanced_table, balanced_offline_loads = build_balanced_hash(freqs, num_experts)
    mod_table = build_mod_hash(vocab, num_experts)

    print("Balanced assignment table:", balanced_table)
    print("Balanced offline loads:", balanced_offline_loads)
    print("Modulo-hash table:", mod_table)

    # 两个玩具 expert：输入 embedding，输出一个可检查的数值
    experts = [
        lambda x: x + 10.0,
        lambda x: x + 20.0,
    ]

    # 一段待路由的 token 序列，以及与之对应的 embedding
    tokens = [0, 1, 0, 2, 3, 1, 4, 5, 0, 2]
    token_embeds = [0.1, 0.2, 0.3, 1.0, 1.5, 0.4, 2.0, 2.5, 0.6, 1.2]

    outputs, expert_ids = route_tokens(tokens, balanced_table, experts, token_embeds)

    print("Tokens:", tokens)
    print("Expert IDs:", expert_ids)
    print("Outputs:", outputs)

    # 验证：同一个 token 的 expert 分配在整个运行期保持不变
    for tok in set(tokens):
        routed = balanced_table[tok]
        assert all(balanced_table[t] == routed for t in [tok])

    # 对比运行时负载
    balanced_runtime_loads = estimate_runtime_load(tokens, balanced_table, num_experts)
    mod_runtime_loads = estimate_runtime_load(tokens, mod_table, num_experts)

    print("Balanced runtime loads:", balanced_runtime_loads)
    print("Modulo-hash runtime loads:", mod_runtime_loads)

    # 输出结果做一个最基本的正确性检查
    for emb, eid, out in zip(token_embeds, expert_ids, outputs):
        expected = emb + (10.0 if eid == 0 else 20.0)
        assert abs(out - expected) < 1e-9

    print("All checks passed.")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行。一个典型输出会体现两件事：

1. `build_balanced_hash` 在离线阶段把高频 token 分散到不同 expert。
2. `route_tokens` 在在线阶段只查表，完全没有学习型 router 的打分过程。

如果把流程抽象成更接近真实 MoE 推理的伪代码，通常就是：

```python
def hash_layer(tokens, token_embeds, balanced_table, experts):
    expert_ids = [balanced_table[tok] for tok in tokens]
    outputs = [experts[eid](emb) for eid, emb in zip(expert_ids, token_embeds)]
    return outputs
```

真实系统里，外层仍然会有 dispatch、all-to-all、capacity 限制、combine 等环节。Hash Layer 改变的只有一件事：`expert_id` 如何产生。它没有改变 MoE 并行的整体工程结构。

### 真实工程例子

以 Reddit 这类大规模文本训练为例，token 频率分布通常高度长尾。若 expert 数很多，例如 `1x64` 或 `1x128`，学习型 router 在训练前期很容易出现两个工程问题：

1. 某些 expert 很少被选中，利用率偏低。
2. 某些 expert 因为早期偏置接收大量 token，造成拥塞。

Hash Layer 在这种场景下的优势很直接：

- token 到 expert 的映射固定，可提前检查每个 expert 的理论负载
- 通信路径更稳定，更容易做系统层规划
- 没有 router 的前向和反向计算
- expert 数量增大时，系统行为通常更可预测

这也是为什么相关工作里，Hash Layer 在大 expert 数设置下常表现出更稳的训练曲线，且在某些数据集上 perplexity 不输、甚至优于学习型 router。它赢的不是“更聪明”，而是“少了一个会抖动的子系统”。

---

## 工程权衡与常见坑

Hash Layer 的工程价值很高，但前提非常明确。最常见的误用，是把“固定路由”误当成“天然均衡”或“天然更便宜”。

### 常见坑一：直接随机哈希，没做频率平衡

如果语料满足 Zipf 分布，而你直接写：

```python
expert = token_id % num_experts
```

那么多个高频 token 可能碰巧落在同一 expert。结果不是“近似均匀”，而是热点 expert 被打爆。

可以用一个极端化例子看差异：

| 方案 | Expert 0 | Expert 1 | Expert 2 | Expert 3 |
|---|---:|---:|---:|---:|
| Single Hash，未预平衡 | 90% | 4% | 3% | 3% |
| Balanced Hash | 26% | 25% | 24% | 25% |

这里真正重要的是流量份额，而不是“每个 expert 分到多少个 token ID”。高频 token 即使只有几个，也足以主导整体负载。

### 常见坑二：把 Hash Layer 当成“更便宜的 Switch”

Hash Layer 和 Switch Transformer 不是简单的“贵版”和“便宜版”关系。两者的目标函数不同：

- Switch 更偏向语义自适应
- Hash Layer 更偏向工程稳定性和低开销

如果任务里同一个 token 在不同上下文中的处理差异很大，例如代码补全中的同名标识符、跨领域术语、检索增强生成中的实体消歧，那么固定映射可能过于僵硬。

### 常见坑三：MultiHash 参数复制不一致

MultiHash 通常会把一个大 FFN 分成多个参数块，再让不同哈希去选这些块。这里很容易出错的地方有：

1. 切片顺序和拼接顺序不一致，导致维度对不上。
2. 名义上多个 hash 应该选不同块，但实现里错误共享了参数。
3. 初始化或 checkpoint 恢复时，块映射关系没有同步保存。

这类 bug 往往不会在第一步就报错，而是在训练一段时间后表现为效果异常、梯度奇怪或加载失败。

### 常见坑四：沿用学习型 router 的调参思路

Hash Layer 没有 load balancing loss，也没有 router z-loss 这类损失项。继续照搬学习型 MoE 的调参思路，通常抓不住重点。Hash Layer 真正需要关注的是：

- 词频统计窗口如何定义
- 冷启动时频率估计是否可靠
- 新词、低频词、未登录词如何回退
- 词表更新后映射表是否要重算
- expert capacity 是否应按静态负载来配置

如果用公式表示，系统负载不均衡风险更接近：

$$
\max_e L_e - \min_e L_e
$$

而不是学习型 router 里的概率熵、路由分布偏置等问题。

### 实际权衡

| 维度 | Hash Layer 优势 | Hash Layer 代价 |
|---|---|---|
| 训练复杂度 | 少一个 router 子系统 | 需要离线词频统计 |
| 运行开销 | 路由几乎零成本 | 需要维护查表逻辑 |
| 负载均衡 | 可提前做强约束 | 依赖分布稳定性 |
| 表达能力 | 规则简单、稳定 | 不理解上下文语义 |

因此，Hash Layer 不是“全局更强”的 MoE 方案，而是“在吞吐、稳定性、实现复杂度优先时更合适”的方案。

---

## 替代方案与适用边界

Hash Layer 最直接的对照对象，是 Switch Transformer、BASE Layers 这类学习型路由方法。

| 模型/路线 | Route Type | 优点 | 限制 |
|---|---|---|---|
| Hash Layer | 固定哈希 | 零路由参数、稳定、快 | 无法按语义动态调整 |
| Switch Transformer | 学习型 top-1 | 实现相对简单，语义自适应 | 需调负载损失，可能不稳 |
| BASE Layers | 学习型平衡分配 | 更强调均衡 | 系统实现更复杂 |
| MultiHash | 多重固定哈希 | 比单哈希更灵活 | 仍不是真正语义路由 |

如果只看论文中的实验摘要，Hash Layer 在较大 expert 数配置下往往显示出不错的稳定性收益。例如在 Reddit 任务上，有如下对比：

| 配置 | Hash Layer PPL | Switch PPL |
|---|---:|---:|
| 1x64 | 23.16 | 23.65 |
| 1x128 | 22.89 | 23.52 |

PPL（perplexity，困惑度）越低越好。这里的结果说明，在这些设定下，固定路由并没有天然吃亏，反而可能因为训练更稳定而得到更好的最终效果。

不过这个结论必须带边界一起理解。更准确的适用判断可以概括为：

1. 如果 token 频率分布长期稳定，Hash Layer 很合适。
2. 如果任务优先追求吞吐、训练稳定性和系统简洁度，Hash Layer 很合适。
3. 如果任务需要依据上下文把同一个 token 路由到不同 expert，学习型 router 更合适。
4. 如果领域变化快、词表经常变化、语义漂移明显，Hash Layer 会更脆弱。
5. 如果希望在固定路由和更高表达能力之间取中间路线，可以考虑 MultiHash 这类折中设计。

一句话概括边界：Hash Layer 适合“分布稳定、工程吞吐优先”的系统，不适合“路由策略本身需要持续学习”的系统。

---

## 参考资料

1. Roller, S. et al. *Hash Layers for Large Sparse Models*. NeurIPS 2021. 作用：给出 Hash Layer、Balanced assignment、MultiHash 的核心定义、结构形式与实验结果。
2. 论文对应的 OpenReview 页面。作用：补充审稿讨论与引用链，便于追踪 Hash Layer 与其他稀疏路由方法的关系。
3. Switch Transformer 论文。作用：作为学习型 top-1 路由的直接对照，帮助理解“语义自适应”与“训练稳定性”之间的取舍。
4. BASE Layers 相关论文。作用：用于对照“平衡分配”路线，说明 Hash Layer 并不是唯一重视负载均衡的 MoE 方法。
5. 论文中的 Reddit 实验部分。作用：展示在 `1x64`、`1x128` 等大 expert 配置下，Hash Layer 在稳定性和 perplexity 上可以优于学习型路由。
6. Zipf 分布相关统计背景资料。作用：解释为什么自然语言中少数高频 token 会主导负载，也说明为什么“先做频率平衡”是 Hash Layer 的前提而不是附加优化。

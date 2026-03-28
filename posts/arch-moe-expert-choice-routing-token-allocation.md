## 核心结论

Expert Choice Routing，简称 ECR，是一种把路由决策从“token 选专家”改成“专家选 token”的 MoE 路由方法。白话说，传统做法是每个 token 自己挑前 $k$ 个专家；ECR 则是每个专家先看完整个批次，再挑自己最想处理的若干 token。这个视角切换的直接结果是：每个专家的容量更容易被填满，负载均衡更自然，token 被分配到的专家数也可以不是固定值。

这件事重要的原因不在于“路由方向很新”，而在于它解决了传统 token-centric top-$k$ 的两个老问题。第一，热门 token 往往会同时冲向少数高分专家，导致这些专家超载，其他专家空闲。第二，容量不足时必须丢 token，训练会出现一部分样本根本没经过专家层。ECR 把容量约束放在专家侧后，能够直接保证每个专家最多处理固定数量的 token，从机制上减少长尾负载。

NeurIPS 2022 的论文给出的核心经验结论很明确：在相同计算预算下，Expert Choice 比 Switch Transformer 的 top-1 和 GShard 的 top-2 收敛更快，文中报告在 8B/64E 设定下训练收敛时间超过 2 倍提升；在 GLUE/SuperGLUE 任务上，同等激活计算下也更强。Apple 2025 的 EC-DIT 则把这一思路用到 Diffusion Transformer，按图像区域复杂度分配不同专家数，在 97B 参数级别仍能维持质量与速度平衡。这说明 ECR 不是只适合论文中的语言模型，它更适合“整批 token 一起看、再统一调度”的大规模训练系统。

下表先给出一个工程上最有用的判断：

| 方法 | 路由视角 | 每 token 专家数 | 专家容量保障 | 负载均衡方式 | 典型问题 |
|---|---|---:|---|---|---|
| Switch Top-1 | token-centric | 固定 1 | 弱 | 辅助 loss + 截断 | 热门 token 抢同一专家 |
| GShard Top-2 | token-centric | 固定 2 | 弱 | 辅助 loss + 截断 | 通信更大，仍会过载 |
| Expert Choice | expert-centric | 可变 | 强 | 由专家容量直接约束 | 排序和实现更复杂 |

玩具理解可以先用“三张桌子分披萨”来记。桌子就是专家，披萨片就是 token。传统方法像是每片披萨自己跑去找最喜欢的桌子，结果好几片都挤到同一桌。ECR 则是每张桌子先从所有披萨里挑自己最想吃的两片，这样每张桌子都能吃满，也不容易浪费容量。

---

## 问题定义与边界

Mixture of Experts，简称 MoE，是“很多子网络里每次只激活少数几个”的结构。白话说，模型总参数很多，但每个 token 实际只走一小部分网络，因此可以在不按比例增加计算量的前提下扩大模型容量。路由器负责决定“哪个 token 送到哪个专家”。

问题的核心不是“能不能把 token 送出去”，而是“送得是否均衡、是否稳定、是否浪费”。传统 token-choice 路由通常对每个 token 计算所有专家分数，然后取 top-$k$。这看起来合理，但有一个结构性问题：每个 token 独立决策，没人负责全局容量。于是很多 token 会共同偏好少数专家，造成局部拥塞。容量满了以后，系统只能丢弃部分 token，或者把它们回退到残差路径。这样一来，训练信号被破坏，专家专门化也会变差。

ECR 要解决的，就是这个“局部最优导致全局过载”的问题。它把硬约束写成：每个专家最多接收 $C_j$ 个 token，其中 $C_j$ 是专家 $j$ 的桶大小；所有 token 总负载不能超过总容量；最好每个 token 至少被一个专家处理。论文里常用
$$
k = \frac{n \cdot c}{e}
$$
其中 $n$ 是本批 token 总数，$e$ 是专家数，$c$ 是 capacity factor，也叫容量因子。白话说，$c$ 控制“平均每个 token 能占用多少专家预算”。如果 $c=1$，总容量大致够每个 token 进一个专家；如果 $c=2$，总容量允许平均每个 token 被两个专家处理。

一个最小边界例子：

- 4 个 token，2 个专家
- 每个专家容量 2
- 总容量是 4

如果每个专家各自从 4 个 token 里选分数最高的 2 个，那么两个专家都能满载。但这不自动保证“每个 token 都至少被选一次”。如果两个专家都偏爱同样的两个 token，那么另外两个 token 可能无人处理。所以 ECR 的“专家满载”是天然保证，“token 全覆盖”则依赖容量因子、分数分布，或者额外约束求解。

这一点和传统 top-$k$ 的差异可以直接列出来：

| 维度 | 传统 top-$k$ | Expert Choice |
|---|---|---|
| 开放 token 数 | 每个 token 独立决策 | 专家基于全批 token 决策 |
| 容量保障 | 先选后截断，容易丢 token | 先定容量再选 token |
| 每 token 专家数 | 固定 | 可变 |
| 专家是否一定满载 | 不一定 | 通常可以直接满载 |
| 是否天然避免热门拥塞 | 否 | 更强 |

所以 ECR 的边界也很清楚。它最适合批处理训练、encoder、vision 或 diffusion 一类“能同时看到整段序列或整张图 patch”的场景；如果是严格流式、强因果、一步一 token 的低延迟推理，它的全局排序优势会被“看不到未来 token”削弱。

---

## 核心机制与推导

先定义亲和度。亲和度就是“某个 token 和某个专家有多匹配”的分数，白话说就是专家对这个 token 的兴趣强弱。给定 token 表示 $x_i$ 和路由矩阵 $W_g$，可以写成
$$
g_{ij} = (x_i W_g)_j
$$
或者像原论文那样先做 softmax，得到一个 $n \times e$ 的分数矩阵 $S$。无论写法是 logits 还是归一化后的分数，本质都一样：先得到“token 对所有专家”的打分矩阵。

传统 top-$k$ 是对每一行做 top-$k$，也就是“一个 token 选若干专家”。ECR 则对每一列做 top-$C_j$，也就是“一个专家选若干 token”。形式化写法是
$$
J_j = \operatorname{arg\,top}_{C_j}\{g_{ij}\}_{i=1}^n
$$
这里 $J_j$ 是专家 $j$ 最终接收的 token 集合。因为每列都只取前 $C_j$ 个，所以专家负载被硬性限制住了。

看一个 3 个专家、每个容量 2 的玩具例子。假设分数矩阵是：

$$
G=
\begin{bmatrix}
0.9 & 0.2 & 0.1\\
0.8 & 0.7 & 0.3\\
0.1 & 0.6 & 0.4\\
0.2 & 0.3 & 0.95\\
0.4 & 0.1 & 0.8
\end{bmatrix}
$$

行是 token $t_1\sim t_5$，列是专家 $e_1,e_2,e_3$。按列选前 2 个：

- $e_1$ 选 $t_1,t_2$
- $e_2$ 选 $t_2,t_3$
- $e_3$ 选 $t_4,t_5$

这时三个专家都满载。更关键的是，$t_2$ 被两个专家同时选中，说明它是“复杂 token”或者“高价值 token”，值得更多计算；而其他 token 只被一个专家选中。ECR 就是在机制上允许这种不均匀计算分配。

如果希望进一步控制“每个 token 最多被几个专家选中”，可以引入额外约束。论文给出的是熵正则化线性规划。把分配矩阵记为 $A \in \mathbb{R}^{e \times n}$，其中 $A_{ij}$ 表示专家 $i$ 对 token $j$ 的选择强度。目标可写为
$$
\max_A \langle S^\top, A \rangle + \lambda H(A)
$$
约束是
$$
\sum_j A_{ij} = k,\quad \sum_i A_{ij} \le b,\quad 0 \le A_{ij} \le 1
$$
其中：

- $\langle S^\top, A \rangle$ 是总匹配分数
- $H(A)$ 是熵项，白话说是让解更平滑、更容易迭代求
- $\sum_j A_{ij}=k$ 表示每个专家正好选 $k$ 个 token
- $\sum_i A_{ij}\le b$ 表示每个 token 最多被 $b$ 个专家选中

很多入门资料会把这个约束写成“每行和为 1”。那是另一种归一化写法，用于表达“每个 token 最终只保留一个总权重单位”。工程上要抓住的不是符号差异，而是约束意图：专家容量受限，token 覆盖或重复分配也受限。

可以把 $A$ 想成一个打勾矩阵。列是 token，行是专家。每个专家这一行最多填满自己的几个格子；每个 token 这一列最多被若干专家勾上。这样就从“局部各选各的”变成了“全局有容量约束的匹配问题”。

---

## 代码实现

实现 ECR 可以拆成四步：算分、专家侧取 top-$C$、分发 token、聚合输出。下面给一个可运行的 Python 玩具实现。它不涉及 GPU 通信，但足够说明 expert-centric 选择、容量控制和 token 覆盖检查。

```python
import numpy as np

def expert_choice_routing(scores, capacity):
    """
    scores: shape [num_tokens, num_experts]
    capacity: int, every expert selects top-capacity tokens
    returns:
        assignments: list of (expert_id, token_id)
        token_to_experts: dict[token_id] -> list[expert_id]
    """
    num_tokens, num_experts = scores.shape
    assignments = []
    token_to_experts = {i: [] for i in range(num_tokens)}

    for expert_id in range(num_experts):
        col = scores[:, expert_id]
        top_tokens = np.argsort(col)[-capacity:][::-1]
        for token_id in top_tokens:
            assignments.append((expert_id, int(token_id)))
            token_to_experts[int(token_id)].append(expert_id)

    return assignments, token_to_experts

scores = np.array([
    [0.90, 0.20, 0.10],
    [0.80, 0.70, 0.30],
    [0.10, 0.60, 0.40],
    [0.20, 0.30, 0.95],
    [0.40, 0.10, 0.80],
], dtype=float)

assignments, token_to_experts = expert_choice_routing(scores, capacity=2)

# 每个专家都正好选择 2 个 token
expert_counts = {}
for expert_id, token_id in assignments:
    expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
assert expert_counts == {0: 2, 1: 2, 2: 2}

# token 1 会被两个专家选中，说明 ECR 允许可变专家数
assert len(token_to_experts[1]) == 2

# 这个玩具例子里每个 token 至少被一个专家覆盖
assert all(len(v) >= 1 for v in token_to_experts.values())
```

这个例子里最关键的不是 `argsort`，而是“按列取 top-$C$”这个方向。如果你把它改成按行取 top-$k$，就回到了 token-choice。

实际系统里的伪代码通常更像这样：

```python
# score -> select -> dispatch -> aggregate
scores = router(x)                  # [tokens, experts]
for expert in experts:
    top_tokens = topk(scores[:, expert], C)
    dispatch_buffer[expert] = x[top_tokens]

expert_outputs = all_to_all_and_run_ffn(dispatch_buffer)
y = unshuffle_and_weight_sum(expert_outputs)
```

这里有两个工程关键词。

第一，`shuffle/unshuffle`。白话说，就是先把属于同一专家的 token 重新排到一起，送到对应设备或对应 FFN，再把输出按原 token 顺序拼回去。如果这个映射关系错了，模型不会报错，但输出语义会直接错位。

第二，`all-to-all` 通信。白话说，就是多个设备之间互相交换各自需要处理的 token。MoE 性能很大一部分不是算力瓶颈，而是通信瓶颈。ECR 因为专家先看全批 token，通常比 token-choice 更依赖高效排序和跨设备搬运。

真实工程例子可以看 Apple 的 EC-DIT。Diffusion Transformer 会把图像切成 patch token，再与文本条件一起处理。ECR 在这里的好处是，模型可以把更多专家预算给复杂区域，比如人物细节、文字、边缘结构，而不是让所有 patch 一律走固定数量的专家。换句话说，ECR 把“计算量跟输入复杂度走”这件事做成了路由机制本身，而不是靠外部规则硬调。

---

## 工程权衡与常见坑

ECR 的理论优势很直接，但工程代价也很直接。最主要的额外成本不是 FFN 本身，而是专家侧排序和分发通信。token-choice 只需要每个 token 本地取 top-$k$；ECR 需要每个专家面对全批 token 取 top-$C$。当 batch 很大、专家很多、设备很多时，这一步会把 latency 和实现复杂度同时推高。

下表可以作为落地时的检查清单：

| 维度 | 收益 | 风险 | 常见缓解措施 |
|---|---|---|---|
| 排序开销 | 更稳定的容量控制 | expert-side global top-$C$ 成本高 | 用高效 topk/radix sort，减少不必要精度 |
| 通信开销 | 更精确的 dispatch | all-to-all 放大尾延迟 | 按专家分桶、做通信与计算重叠 |
| LP/迭代成本 | 更强 token 覆盖与专家数约束 | Dykstra 等迭代增加时延 | 只在训练启用，推理关闭或简化 |
| capacity factor 调整 | 控制平均激活专家数 | 太小会漏 token，太大浪费算力 | 先从 $f_c=1$ 或 $2$ 做网格搜索 |
| token coverage | 提高训练稳定性 | 高分 token 重复被选，低分 token 无覆盖 | 增大容量、加上 token 上限约束 |

一个很常见的坑是把 $f_c=1$ 当成“天然合理”。实际上 $f_c=1$ 只表示总容量大致等于 token 数，不表示分配一定稳定。若 batch 变小，分数噪声会更大，专家端排序更容易反复挑中相似 token，长尾专家会越来越同质化。结果不是“系统崩了”，而是专家学习到的模式越来越窄，最终表现成收敛变慢或者泛化下降。论文里的结果也显示，capacity factor 从 2 降到 1 会退化，但通常仍优于简单 top-1。

另一个坑是输出聚合。因为 token 可能被多个专家处理，你必须明确采用什么聚合规则：加权和、平均、门控加权，还是保留最强一路。只要 `shuffle -> expert FFN -> unshuffle -> aggregate` 这条链条里索引有一个地方错位，训练损失照样会下降，但你得到的是一套错误的路由-输出对应关系。这类 bug 很难靠单元测试之外的手工观察发现。

还要注意一个现实问题：ECR 的优势主要体现在训练和大批处理上。如果你的目标是在线低延迟服务，排序和通信尾延迟可能会吃掉精度收益。MoE 从来不是“参数更大一定更值”，而是“路由、通信、容量和场景一起看”。

---

## 替代方案与适用边界

如果业务目标是最低延迟，传统 token-choice 仍然有很强的现实价值。Switch Transformer 的 top-1 和 GShard 的 top-2 都更容易实现，因为每个 token 只需本地选几个专家，通信模式更简单，也更适合自回归生成。代价是负载均衡只能通过辅助 loss 和容量截断间接维持，热门 token 抢占少数专家的问题不会真正消失。

如果业务目标是大规模训练效率、专家利用率、异构计算分配，ECR 更合适。它尤其适合以下场景：

- encoder 或 bidirectional 训练，能同时看到整个序列
- vision / diffusion，能同时看到整批 patch token
- 训练阶段可接受更复杂的排序与通信
- 希望不同 token 获得不同数量的专家计算

还存在一类混合方案，比如加入 null expert，或者做 hybrid routing。null expert 可以理解成“什么都不做但占一个可选路由位的专家”，白话说是给路由器一个显式的“这类 token 不值得额外计算”的出口。hybrid routing 则是部分层用 token-choice 保持低延迟，部分层用 ECR 做强负载控制。它们不如纯 ECR 干净，但在流式推理里更现实。

对比可以总结为：

| 方案 | token 可见性要求 | 延迟友好度 | load balance 保障 | 实施复杂度 | 适用边界 |
|---|---|---|---|---|---|
| ECR | 需要整批或大窗口可见 | 中到低 | 强 | 高 | 大规模训练、DiT、Vision |
| Token-choice | 只需当前 token 局部信息 | 高 | 中到弱 | 低 | 自回归推理、低延迟服务 |
| 混合/Null-Expert | 可按系统设计折中 | 中 | 中 | 中到高 | 训练推理一体化系统 |

真实工程上，Apple EC-DIT 就是 ECR 更适合的代表。图像生成里的 patch 难度差异很大，天空背景和人脸细节显然不该消耗同样计算。ECR 可以给复杂 patch 更多专家，而简单 patch 少给甚至不给，这和 Diffusion Transformer 的异构计算需求天然匹配。相反，在自回归文本生成里，每一步只能看到当前上下文状态，系统往往更看重稳定 latency、KV cache 配合和实现可控性，这时 token-choice 往往更稳妥。

结论可以压缩成一句话：ECR 不是“更高级的 top-$k$”，而是“把容量约束放到专家侧”的另一套路由哲学。你在训练里追求负载均衡和异构算力分配，它通常更好；你在推理里追求低延迟和简单实现，传统 token-choice 往往更划算。

---

## 参考资料

- Zhou et al., NeurIPS 2022, [Mixture-of-Experts with Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)
- Zhou et al., NeurIPS 2022 PDF, [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf)
- Apple Machine Learning Research, April 2025, [EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing](https://machinelearning.apple.com/research/ec-dit)
- Emergent Mind, [Mixture of Experts with Expert Choice Routing](https://www.emergentmind.com/topics/mixture-of-experts-with-expert-choice-routing)

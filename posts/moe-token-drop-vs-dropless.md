## 核心结论

MoE，Mixture of Experts，指“很多专家子网络里每次只激活少数几个”的稀疏模型。它的核心收益不是让每一步都更便宜，而是让**参数规模**可以远大于**单 token 的实际计算量**。同样是 8 个、16 个甚至更多 expert，每个 token 通常只经过其中 1 个或 2 个 expert，因此总参数量可以继续扩张，而每次前向的 FLOPs 不按 expert 总数线性增长。

这里真正把 MoE 分成两条工程路线的，不是“有没有 expert”，而是“某个 expert 装不下时，token 怎么处理”。

传统 token-drop MoE 用容量因子限制每个 expert 一次最多接收多少 token。常见写法是：

$$
C=\frac{T}{N}\alpha
$$

其中：

| 符号 | 含义 |
|---|---|
| $T$ | 当前 batch 或当前微批次中的 token 总数 |
| $N$ | expert 数量 |
| $\alpha$ | capacity factor，容量因子，用来给平均容量留缓冲 |

如果 expert $i$ 实际收到的 token 比例是 $f_i$，那么超过容量的部分可以近似写成：

$$
r\approx\sum_i \max\left(0, f_i-\frac{\alpha}{N}\right)
$$

这里的 $r$ 可以理解为整体 drop rate 的近似上界。它表达的不是某种神秘现象，而是一个直接事实：**当路由不均衡超过容量缓冲时，就会有 token 没法进入原本选中的 expert。**

传统 token-drop 路线里，这些超出的 token 往往有三种处理方式：

| 处理方式 | 含义 | 影响 |
|---|---|---|
| 直接丢弃 | 不再进入 expert 计算 | 该 token 没有用到这层 expert 参数 |
| 只走残差 | 保留主干网络信息流 | 模型还能前传，但失去该次专家变换 |
| reroute 到备选 expert | 少见于最早期实现 | 复杂度上升，但能减少浪费 |

Dropless MoE 的目标正好相反：**尽量不让 token 因容量问题失去 expert 计算。** MegaBlocks 这类方法把 MoE 计算改写成 block-sparse，也就是“只对真实存在数据的块做稀疏矩阵运算”；Expert Choice 则反过来让 expert 选 token，用固定桶大小在选择阶段就控制负载。实现路径不同，但共同目标一致：**every token has a destination，每个 token 都应当有归宿。**

下面先给出结论层面的对比：

| 方案 | drop 机制 | latency 影响 | 长序列表现 |
|---|---|---|---|
| 传统 token-drop MoE | expert 超容量后丢弃或仅走残差 | 最忙 expert 容易形成尾部等待，延迟波动更大 | 末尾关键 token 可能反复被跳过，质量更脆弱 |
| Dropless MoE | 通过重排、块稀疏或 expert choice 避免 token 丢失 | 每层 token 数和调度更可控，延迟通常更稳定 | 长上下文更稳，因为不会因 overflow 直接漏掉 token |

可以用“邮箱”做一个有限度的比喻。每个 expert 像一个有容量上限的邮箱。传统做法是邮箱满了就退信；dropless 做法是改成重新分拣、按块打包，或者让邮箱主动挑信，保证信最终进入某个可处理的邮箱。这个比喻只帮助理解，不替代正式定义，因为真实系统里还要考虑 router 分数、通信代价、并行拓扑和 kernel 形状。

---

## 问题定义与边界

先把问题定义清楚。Router，也叫门控网络，可以理解成“给 token 分配 expert 的调度器”。在传统 top-k routing 里，router 先对每个 token 输出一组 expert 分数，再选择前 $k$ 个 expert。常见配置是 top-1 或 top-2。到这里为止，一切都还只是“偏好”；真正的问题出在下一步：**偏好要落到有限容量的硬件缓冲和离散 dispatch 上。**

所以 MoE 的关键矛盾不是“平均负载是否均衡”，而是：

$$
\text{期望均衡} \neq \text{单步不溢出}
$$

即使长期统计上每个 expert 都差不多忙，某一个 step、某一层、某一段局部上下文，仍可能出现热门 expert 瞬间拥塞。

设有 4 个 expert、1024 个 token、$\alpha=1.25$，则单 expert 容量是：

$$
C=\frac{1024}{4}\times 1.25=320
$$

如果某个 expert 实际收到 400 个 token，那么溢出量就是：

$$
O_i=\max(0,400-320)=80
$$

这 80 个 token 在传统 token-drop 路线里，通常不会排队等下一轮，而是直接被裁掉专家路径，或者只保留残差支路继续前传。这样做的收益很明确：

| 好处 | 原因 |
|---|---|
| 实现简单 | 固定容量 buffer 更容易写 dispatcher 和 kernel |
| 吞吐稳定 | 不会因为少数 expert 爆满而拖垮整层 |
| 内存可控 | 每个 expert 的激活上界是明确的 |

代价也同样直接：

| 代价 | 本质 |
|---|---|
| 参数没被充分利用 | 被 drop 的 token 没真正走进所选 expert |
| 长序列质量更脆弱 | 关键 token 若反复命中热点 expert，损失会累积 |
| 行为更难解释 | 模型不是“不会做”，而是“这次没让这批 token 用上专家” |

边界条件可以压缩成下面这张表：

| 条件 | 会不会触发 drop | 原因 |
|---|---|---|
| $f_i \le \alpha/N$ 对所有 expert 成立 | 通常不会 | 每个 expert 都没超过容量 |
| 少数 expert 的 $f_i$ 明显高于平均 | 会 | 热点 expert 溢出 |
| 长序列中局部 token 模式高度相似 | 更容易 | router 对相邻 token 做出集中选择 |
| dropless 的 block-sparse dispatch | 不以 drop 解决 overflow | 通过重排和稀疏算子吸收动态负载 |
| Expert Choice 的固定桶选择 | 通常不靠 token drop 保持容量 | expert 先按桶大小选择 token |

这里还要区分两个常被混在一起的问题：**训练时的 token drop** 和 **推理时的容量控制**。

| 场景 | 主要目标 | 关注点 |
|---|---|---|
| 训练 | 稳定训练、控制显存和吞吐 | router 是否学到更均衡的分配 |
| 推理 | 控制延迟，尤其是 p95/p99 | 最忙 expert 是否形成 straggler |

训练里，token drop 往往是“为了适应固定 buffer 的一种实现折中”；推理里，容量控制更多是“为了不让某个 expert 拖慢整层完成时间”。两者都和容量相关，但优化目标并不相同。

一个新手更容易理解的工程例子是长文档问答。假设输入 64K 上下文，前半段是背景，后半段某几行出现关键合同条款或错误码。如果这一段 token 在多个 layer、多个 step 中反复命中热门 expert，而容量又偏小，那么这些 token 被 drop 的概率会随层数和位置累积。模型通常不会表现成“整题完全答错”，而是会出现下面这种更隐蔽的退化：

| 现象 | 实际原因 |
|---|---|
| 引用不完整 | 关键 token 没充分经过 expert 变换 |
| 摘要漏掉一条重要条件 | 后段信号被稀疏路径多次跳过 |
| 检索判断变钝 | 热门 expert 上的关键信号利用率下降 |

这也是为什么“平均 drop rate 很低”并不总能保证长序列质量稳定。你还要看**哪些 token 被 drop**，以及它们是否集中出现在任务最关键的位置。

---

## 核心机制与推导

传统 token-drop MoE 的链路可以压缩成 4 步：

1. Router 为每个 token 计算各 expert 分数。
2. 根据 top-1 或 top-2 规则分配 expert。
3. 对每个 expert 应用容量上限 $C$。
4. 超过容量的 token 被丢弃、跳过，或退回残差支路。

写成一条最简链路就是：

`Gate -> Top-k assignment -> Capacity filter -> Expert compute / Drop`

如果定义 expert $i$ 实际接收的 token 数为 $T_i$，那么它的溢出量是：

$$
O_i=\max(0, T_i-C)
$$

总溢出量是：

$$
O=\sum_{i=1}^{N} O_i
$$

总 drop rate 为：

$$
r=\frac{O}{T}
$$

再把

$$
T_i = T f_i
$$

代回去，得到：

$$
r=\frac{1}{T}\sum_{i=1}^{N}\max(0, Tf_i-C)
$$

又因为

$$
C=\frac{T}{N}\alpha
$$

所以：

$$
r=\frac{1}{T}\sum_{i=1}^{N}\max\left(0, Tf_i-\frac{T}{N}\alpha\right)
$$

进一步化简为：

$$
r\approx \sum_i \max\left(0, f_i-\frac{\alpha}{N}\right)
$$

这个式子有三个非常实用的含义：

| 结论 | 解释 |
|---|---|
| drop 不是随机噪声 | 它直接来自路由不均衡和容量不足 |
| $\alpha$ 越小，drop 越容易出现 | 容量缓冲更薄，热点更难吸收 |
| expert 越多，不代表越不容易 drop | 如果 router 仍严重偏向少数 expert，热点照样出现 |

这也解释了为什么 load balancing loss 很重要。它的作用不是让每个 step 都绝对均匀，而是让 router 在长期统计上不要过度偏向少数 expert。常见写法可以抽象成：

$$
\mathcal{L}=\mathcal{L}_{task}+\lambda \mathcal{L}_{balance}
$$

其中：

| 项 | 含义 |
|---|---|
| $\mathcal{L}_{task}$ | 主任务损失，例如语言建模损失 |
| $\mathcal{L}_{balance}$ | 负载均衡损失，鼓励 expert 使用更均匀 |
| $\lambda$ | 平衡系数，决定“均衡”在总目标中的权重 |

这里要提醒一个常见误解：**负载均衡损失不是让 router 变得“公平”，而是让系统更少出现极端拥塞。**  
如果 $\lambda$ 太小，router 容易一边倒；如果 $\lambda$ 太大，router 又可能为均衡而牺牲任务相关性，把本该去某个 expert 的 token 也强行摊平。

Dropless MoE 的核心不是“把容量概念删除”，而是把“超过容量以后怎么办”换掉。它的链路更接近：

`Gate -> Structured dispatch -> Sparse/block compute -> All tokens return`

这条链路里最关键的变化是：**overflow 不再用 drop 作为默认收尾方式，而是通过更结构化的分发和计算来吸收。**

### 1. MegaBlocks 这一路

MegaBlocks 的关键点不是“更聪明的 router”，而是“更适合不规则 token 分配的计算表示”。传统实现里，如果某个 expert 收到的 token 数不是固定值，就容易遇到两难：

| 做法 | 问题 |
|---|---|
| padding 成固定大矩阵 | 浪费算力和显存 |
| 超出部分直接 drop | 伤害模型质量和参数利用率 |

MegaBlocks 改写后的思路是：把 token 到 expert 的不规则映射，组织成 block-sparse 矩阵乘法。这样系统只计算真实存在数据的块，不需要把所有 expert 输入补成同样大，也不必因为固定 buffer 不够而直接丢 token。

### 2. Expert Choice 这一路

Expert Choice 的关键点是把“谁做选择”反过来。传统 routing 是 token 选 expert，Expert Choice 则让 expert 在候选 token 里选自己要处理的那一桶。这样每个 expert 的容量天然更容易前置为固定桶大小，系统在选择阶段就更容易维持上限，而不是等选完后再裁掉。

它更适合新手理解的一点在于：  
**token-choice 是“每个 token 去抢自己最喜欢的 expert”；expert-choice 是“每个 expert 从候选里挑自己要收的 token”。**  
前者容易在热门 expert 处拥堵，后者更容易控制每个 expert 的装载上限。

下面用机制步骤对比更直观：

| 机制步骤 | 传统 token-drop | Dropless |
|---|---|---|
| 路由主体 | token 选 expert | token 选 expert 或 expert 选 token |
| 容量约束位置 | 分配后裁剪 | 分配时结构化满足 |
| overflow 处理 | 丢弃、跳过或退回残差 | 重排、补位、块稀疏计算 |
| 对质量的主要风险 | 热门 expert 上的 token 没有专家计算 | 路由早期不均可能造成内存尖峰 |
| 对系统的主要风险 | 尾部延迟波动、drop 难解释 | 需要更复杂 kernel 和调度实现 |

再给一个小例子，把公式和直觉对应起来。

假设有 8 个 expert、总共 800 个 token、$\alpha=1.0$，那么：

$$
C=\frac{800}{8}\times1.0=100
$$

如果某一步实际分配是：

| expert | token 数 |
|---|---|
| 0 | 160 |
| 1 | 120 |
| 2 | 95 |
| 3 | 90 |
| 4 | 85 |
| 5 | 85 |
| 6 | 85 |
| 7 | 80 |

那么 overflow 总量是：

$$
O=(160-100)+(120-100)=80
$$

所以：

$$
r=\frac{80}{800}=10\%
$$

10% 的 token 没进入它们原本命中的 expert。这个比例不算小，尤其如果这些 token 正好来自同一段密集相关的上下文，质量退化就可能明显。如果换成 dropless 路线，系统要做的不是“承认这 80 个 token 没法处理”，而是想办法把这 80 个 token 通过结构化 dispatch 或重排纳入可计算形式。

---

## 代码实现

下面先给一个**最小可运行**的 Python 玩具实现。它不依赖任何深度学习框架，只演示三件事：

1. 传统 token-drop 如何裁掉 overflow  
2. dropless 如何把 overflow 重新分配到有空位的 expert  
3. 如何计算每个 expert 的负载和整体 drop rate

```python
from collections import defaultdict, deque

def per_expert_load(buckets, num_experts):
    return {expert_id: len(buckets.get(expert_id, [])) for expert_id in range(num_experts)}

def token_drop_dispatch(assignments, num_experts, capacity):
    """
    assignments: [(token_id, expert_id), ...]
    return:
      buckets: {expert_id: [token_id, ...]}
      dropped: [token_id, ...]
    """
    buckets = defaultdict(list)
    dropped = []

    for token_id, expert_id in assignments:
        if len(buckets[expert_id]) < capacity:
            buckets[expert_id].append(token_id)
        else:
            dropped.append(token_id)

    return buckets, dropped

def dropless_dispatch(assignments, num_experts, capacity):
    """
    简化版 dropless:
    先按原始 expert 放入；满了之后进入 overflow 队列；
    再把 overflow 依次放入还有空位的 expert。
    注意：真实系统会参考 router 分数、通信拓扑和块结构，
    这里仅演示“所有 token 最终都有归宿”的思想。
    """
    buckets = defaultdict(list)
    overflow = deque()

    for token_id, expert_id in assignments:
        if len(buckets[expert_id]) < capacity:
            buckets[expert_id].append(token_id)
        else:
            overflow.append(token_id)

    for expert_id in range(num_experts):
        while overflow and len(buckets[expert_id]) < capacity:
            buckets[expert_id].append(overflow.popleft())

    if overflow:
        raise RuntimeError("total capacity is insufficient for dropless dispatch")

    return buckets

def drop_rate(total_tokens, dropped_tokens):
    return len(dropped_tokens) / total_tokens if total_tokens > 0 else 0.0

def print_result(title, buckets, num_experts, dropped=None):
    print(f"=== {title} ===")
    load = per_expert_load(buckets, num_experts)
    for expert_id in range(num_experts):
        tokens = buckets.get(expert_id, [])
        print(f"expert {expert_id}: load={len(tokens):2d}, tokens={tokens}")
    if dropped is not None:
        print(f"dropped tokens: {dropped}")
    print()

def main():
    # 12 个 token，其中大量 token 被分到 expert 0，制造热点
    assignments = (
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
        + [(8, 1), (9, 2), (10, 3), (11, 3)]
    )

    num_experts = 4
    capacity = 3
    total_tokens = len(assignments)

    drop_buckets, dropped = token_drop_dispatch(assignments, num_experts, capacity)
    print_result("token-drop", drop_buckets, num_experts, dropped)
    print(f"drop rate = {drop_rate(total_tokens, dropped):.2%}")
    print()

    full_buckets = dropless_dispatch(assignments, num_experts, capacity)
    print_result("dropless", full_buckets, num_experts)
    print(f"all tokens dispatched = {sum(len(v) for v in full_buckets.values()) == total_tokens}")

if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出会类似这样：

```text
=== token-drop ===
expert 0: load= 3, tokens=[0, 1, 2]
expert 1: load= 1, tokens=[8]
expert 2: load= 1, tokens=[9]
expert 3: load= 2, tokens=[10, 11]
dropped tokens: [3, 4, 5, 6, 7]

drop rate = 41.67%

=== dropless ===
expert 0: load= 3, tokens=[0, 1, 2]
expert 1: load= 3, tokens=[8, 3, 4]
expert 2: load= 3, tokens=[9, 5, 6]
expert 3: load= 3, tokens=[10, 11, 7]

all tokens dispatched = True
```

这个例子能说明两个最重要的事实：

| 现象 | 含义 |
|---|---|
| token-drop 版本里有 5 个 token 被裁掉 | 热点 expert 的溢出会直接变成 drop |
| dropless 版本里所有 token 都被放进了某个 expert | 代价是系统必须支持更复杂的重排与调度 |

但要明确，这只是**教学级玩具实现**。真实系统至少还要处理：

| 真实问题 | 玩具代码里有没有覆盖 |
|---|---|
| top-2 gating 和 combine 权重 | 没有 |
| 不同 expert 的并行执行 | 没有 |
| all-to-all 通信 | 没有 |
| block packing / sparse kernel | 没有 |
| 按 router score 决定 reroute 优先级 | 没有 |
| 训练中的梯度传播 | 没有 |

下面给两段更接近工程实现的伪代码。

传统 token-drop routing：

```text
# token-drop MoE
scores = router(hidden_states)              # [num_tokens, num_experts]
topk_experts = select_topk(scores, k=1 or 2)
dispatch_map = build_dispatch(topk_experts)

for expert in experts:
    tokens = dispatch_map[expert]

    if len(tokens) > capacity:
        kept = tokens[:capacity]
        overflow = tokens[capacity:]

        expert_out_kept = expert_forward(kept)
        fallback_out = residual_forward(overflow)
    else:
        expert_out_kept = expert_forward(tokens)

outputs = combine_outputs(expert_out_kept, fallback_out)
metrics = {
    "drop_rate": dropped_tokens / total_tokens,
    "max_tokens_per_expert": max_load,
    "load_balance_loss": aux_loss,
}
```

Dropless 版本则更接近：

```text
# dropless MoE
scores = router(hidden_states)
routing = structured_assign(scores)            # 不把 overflow 留到最后裁剪
packed_blocks = pack_tokens_into_blocks(routing)
sparse_out = block_sparse_expert_forward(packed_blocks)
outputs = unpermute_and_combine(sparse_out)

metrics = {
    "max_tokens_per_expert": max_load,
    "block_utilization": non_empty_blocks / total_blocks,
    "load_balance_loss": aux_loss,
    "layer_latency_p95": p95_latency,
}
```

工程上真正要监控的，不是“代码能不能跑”，而是系统是否稳定地跑。下面这些指标通常比单次 loss 更重要：

| 指标 | 传统 token-drop | Dropless |
|---|---|---|
| `drop_rate` | 必须监控 | 应接近 0 |
| `max_tokens_per_expert` | 必须监控 | 必须监控 |
| `load_balance_loss` | 必须监控 | 必须监控 |
| `expert_oom_count` | 较少见 | 初期更需要盯 |
| `layer_latency_p95/p99` | 很关键 | 更关键，因为目标之一就是稳定尾延迟 |
| `block_utilization` | 通常没有 | 很关键 |
| `all_to_all_time` | 关键 | 更关键 |

如果你是新手，可以先把监控理解成一句话：  
**不是只看“loss 有没有下降”，而是要看“token 有没有被合理地送进 expert，并且系统有没有因局部拥塞失控”。**

---

## 工程权衡与常见坑

第一类坑是 $\alpha$ 设得过小。$\alpha$ 小，单 expert buffer 小，显存和吞吐看起来更省，但热点一出现就频繁 drop。短序列里，退化可能不明显，因为很多样本本来就不依赖极少数关键 token；长上下文里，它会变成“关键 token 反复拿不到专家计算”。这类问题难查的原因在于：dense baseline 往往没问题，短样本评测也可能没问题，只有在长输入和特定负载分布下才暴露。

第二类坑是把 load balancing loss 当成万能药。它可以缓解长期偏斜，但不能消除单步尖峰。训练早期 router 还没稳定时，某些 expert 可能突然吸进大量 token。传统路线里，这表现为高 drop；dropless 路线里，则可能表现成显存峰值、通信拥塞甚至局部 OOM。

第三类坑发生在多卡推理。MoE 常用 expert parallel，不同 expert 分布在不同设备上。一层 MoE 的结束时间，往往取决于最忙的那个 expert。这就是 straggler effect，拖尾效应。它的本质不是“平均负载高”，而是“最慢那一路决定大家一起结束的时间”。

可以写成非常粗糙但直观的近似：

$$
\text{layer latency} \approx \max_i \text{latency}(expert_i)
$$

如果只优化平均负载，而不控制最大负载，那么 p50 可能很好，p99 仍然很差。Dropless 并不自动等于低延迟，但因为 dispatch 结构更清晰、每层 token 的处理去向更可控，延迟通常更容易预测。

下面把常见坑和规避策略压缩成表格：

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| low $\alpha$ -> key token drop | 长序列质量下降，摘要遗漏，检索答非所问 | 提高 $\alpha$，联动监控 `drop_rate` 与质量指标 |
| 路由极不均衡 | 少数 expert 过热，其余 expert 闲置 | 使用 auxiliary/load balancing loss，必要时加 router warm-up |
| dropless 早期 OOM | 训练前几百步显存尖峰 | warm-up、router 抖动、限制初期 top-k 或 block 大小 |
| chunk leak | 分块策略破坏因果或让部分 token 长期吃亏 | 因果场景做严格顺序约束，必要时 shuffle 与重排 |
| 只看平均延迟 | p50 正常但 p99 很差 | 按 layer 和 expert 监控 tail latency |
| 忽略残差补偿效果 | 误以为 drop 等于完全消失 | 明确区分“未经过 expert”与“模型不可用”，单独做 ablation |

上表里有两个点值得展开。

### 1. 为什么“drop”不等于“模型直接坏掉”

在很多实现中，被 drop 的 token 不是从网络中彻底消失，而是仍然保留残差支路、注意力支路或主干表示。因此 drop 的真实含义更接近：

> 这个 token 没有得到这层 expert 的专门变换，而不是这个 token 从模型里消失了。

所以你在实验里看到的退化往往是“细节变差、关键证据引用率下降、少量长尾样本失分”，而不是“整体完全崩溃”。这也是为什么做 ablation 时，不能只问“有没有 drop”，而要问：

| 更重要的问题 | 原因 |
|---|---|
| drop 发生在哪些 layer | 不同层的语义作用不同 |
| drop 集中在什么位置 | 长文后段被 drop 代价通常更大 |
| drop 的 token 是不是任务关键 token | 关键 token 少但影响大 |
| 残差是否足够补偿 | 不同模型补偿能力不同 |

### 2. 为什么 dropless 也会有自己的坑

很多新手会把 dropless 理解成“既然不丢 token，就一定更好”。这不准确。dropless 解决的是“token 不该因为固定容量而直接失去专家计算”，但它换来的系统成本包括：

| 成本 | 解释 |
|---|---|
| 更复杂的 dispatcher | 需要更复杂的路由后重排 |
| 更依赖稀疏 kernel | 没有高质量 sparse kernel，收益可能落空 |
| 对显存峰值更敏感 | 所有 token 都要被实际承接 |
| 对通信质量更敏感 | 多卡 all-to-all 更容易成为瓶颈 |

一个实用 checklist：

- 监控每层 `drop_rate`
- 监控最忙 expert 的 token 数
- 训练前期单独看 expert 负载直方图
- 长上下文任务单独评测，不要只看短样本
- 推理场景看 p95/p99 latency，不只看平均值
- dropless 路线提前验证 block-sparse kernel 与显存峰值
- 区分“路由不均”与“kernel 太慢”两类问题
- 做少量长尾样本的人工误差分析，而不只看平均 benchmark

真实工程例子是多卡部署 Mixtral 一类模型做长文档检索推理。业务真正要的是“单条请求延迟可预测”，而不是“某些批次平均吞吐很好看”。这时最忙 expert 的负载往往直接决定用户感知延迟。传统 token-drop 方案可以通过裁掉 overflow token 把最差情况压住，但代价是文档后半段的重要 token 更可能失去专家处理。Dropless 方案如果配合 block-sparse dispatch，把每层 token 编排得更规则，通常能同时改善质量一致性和尾延迟稳定性，但前提是内核、通信和显存预算都跟得上。

---

## 替代方案与适用边界

传统 token-drop MoE 不是“落后方案”，而是“在工程复杂度、吞吐和质量之间做出的明确折中”。如果目标是先把大规模稀疏训练系统搭起来，top-1 或 top-2 routing 加固定 capacity 仍然是实现门槛最低、调试路径最成熟的一条路线。尤其在训练场景，只要 drop rate 被压到足够低，它依然很有实用价值。

Dropless 也不是“天然更优”。它更适合这些前提同时成立的场景：

| 前提 | 为什么更适合 dropless |
|---|---|
| 长上下文很重要 | 关键 token 不能频繁失去专家计算 |
| 在线推理对 p99 敏感 | 最忙 expert 的拖尾代价高 |
| 可以接受更复杂的系统栈 | 需要 dispatcher、稀疏 kernel、监控配套 |
| 希望专家参数利用率更高 | 不希望 token 因 overflow 直接跳过专家 |

先给一个场景对比：

| 维度 | Token-drop MoE | Dropless MoE |
|---|---|---|
| 训练实现复杂度 | 低 | 中到高 |
| 推理延迟可预测性 | 一般 | 较好 |
| 长上下文适配 | 较弱 | 较强 |
| 对稀疏 kernel 依赖 | 低 | 高 |
| 对通信调度要求 | 中 | 高 |
| 质量上限 | 受 drop 影响 | 通常更充分利用参数容量 |
| 硬件要求 | 更宽松 | 更依赖高质量 sparse/dispatch 支持 |

如果任务是“先拿到稳定可复现的稀疏训练曲线”，传统 token-drop 路线通常更容易落地。因为它的系统行为更简单，出现问题时也更容易定位：要么是 router 失衡，要么是 capacity 太小，要么是通信或 kernel 本身。  
如果任务是“长文档检索推理”“多轮 agent 读超长日志”“超长上下文代码分析”，dropless 通常更值得优先评估，因为这类任务往往对少量关键 token 极其敏感。

这里也可以把选择逻辑压成一个更具体的决策树：

1. 任务是否主要受长上下文质量影响？  
   是：优先考虑 dropless。

2. 推理是否有严格 p99 latency 约束？  
   是：优先评估 dropless，或至少评估容量感知 reroute。

3. 当前系统是否缺少 block-sparse / 高效 dispatcher 支持？  
   是：先用 token-drop，把 `drop_rate` 压低，再决定是否升级。

4. 训练早期是否频繁 OOM 或路由极不稳定？  
   是：先从简单 token-drop 路线起步，再逐步过渡到 dropless。

5. 业务是否真的在意“每个 token 都进入专家”而不是“平均吞吐更高”？  
   如果前者更重要，dropless 的工程成本通常值得；如果后者更重要，token-drop 可能更实用。

可以把这一节的结论归纳成一句更工程化的话：  
**token-drop 是“牺牲一部分 token 的专家覆盖，换更简单的系统边界”；dropless 是“提高 token 覆盖和延迟稳定性，但把复杂度转移到 dispatcher、kernel 和通信栈”。**

---

## 参考资料

| 引用 | 主题 | 贡献 |
|---|---|---|
| Aman.ai Mixture-of-Experts Primer | 定义与数学 | 梳理 capacity factor、token drop、drop rate 近似表达 |
| MegaBlocks: Efficient Sparse Training with Mixture-of-Experts | dropless 机制 | 说明 block-sparse dispatch 如何避免因固定容量而丢 token |
| Mixture-of-Experts with Expert Choice Routing | expert choice 机制 | 说明 expert 选 token 如何用固定桶大小改善负载均衡 |
| NVIDIA Megatron Core MoE Docs | 工程实现 | 给出 dropless MoE、dispatcher、训练配置和常见系统优化项 |
| Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts | 推理延迟 | 说明多 expert 并行下最忙 expert 主导延迟，以及容量感知 drop/reroute 的推理价值 |

- Aman.ai，支持本文中的 token drop 定义、容量公式 $C=\frac{T}{N}\alpha$、以及基于 $f_i$ 的 drop rate 近似分析。链接：https://aman.ai/primers/ai/mixture-of-experts/
- Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia，MegaBlocks: Efficient Sparse Training with Mixture-of-Experts，支持本文中的 block-sparse dropless 结构、避免 padding 与 token dropping 的核心机制。链接：https://arxiv.org/abs/2211.15841
- Yanqi Zhou 等，Mixture-of-Experts with Expert Choice Routing，支持本文中“expert 选 token、固定桶大小、改善收敛与负载均衡”的部分。链接：https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html
- NVIDIA Megatron Core MoE 文档，支持本文中的 dropless MoE 工程实现、token dispatcher、负载均衡损失与训练配置要点。链接：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
- Shwai He 等，Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts，支持本文中的 straggler effect、容量感知 token drop / reroute 与推理延迟分析。链接可检索 arXiv 标题：`Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts`

进一步阅读时，建议按下面顺序看：

| 阅读顺序 | 为什么 |
|---|---|
| 先看 Aman.ai Primer | 先把 capacity、routing、drop rate 这些基本符号看明白 |
| 再看 Expert Choice | 理解“token 选 expert”和“expert 选 token”的结构差异 |
| 再看 MegaBlocks | 理解 dropless 不是口号，而是计算表示和 kernel 设计变化 |
| 最后看 Megatron Core 与推理论文 | 把训练机制和在线系统问题连起来 |

如果你只想记住本文最后一句判断，可以记成：

$$
\text{MoE 的关键分叉} \;=\; \text{overflow token 如何处理}
$$

传统 token-drop 路线把 overflow 当作容量边界后的裁剪问题；dropless 路线把它当作 dispatch 与计算表示问题。前者实现更简单，后者通常在长上下文质量和尾延迟稳定性上更有优势。

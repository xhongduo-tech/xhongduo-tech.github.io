## 核心结论

MoE，Mixture of Experts，直译是“专家混合模型”。它的核心做法是：每个 token 不经过全部前馈子网络，而是只激活其中少数几个专家。这样可以在总参数量很大的前提下，把每次实际参与计算的参数量控制在较低水平。

但在推理阶段，MoE 的主要瓶颈通常不是“算力不够”，而是“专家权重加载过于频繁”。如果专家不能全部常驻在当前计算设备的高带宽内存中，那么系统就需要不断把不同专家的权重搬到可执行位置。这个搬运过程一旦打碎、重复、不可预测，延迟就会上升。

对单层推理，一个足够实用的近似延迟模型可以写成：

$$
L \approx b \cdot T + a \cdot Bk
$$

其中：

| 符号 | 含义 | 直观解释 |
|---|---|---|
| $L$ | 单层推理延迟 | 这一层最终花了多久 |
| $b$ | 单次专家权重加载延迟 | 把一个专家权重搬到可计算位置的代价 |
| $T$ | 本批次实际访问到的专家数 | 这一批 token 一共触发了多少个不同专家 |
| $a$ | 单个 token-expert 计算延迟 | 一个 token 在一个专家上完成前馈的计算代价 |
| $B$ | batch 大小 | 这一批有多少 token |
| $k$ | top-k 专家数 | 每个 token 会被送到多少个专家 |

这个公式对应一个很重要的工程判断：

$$
\text{必要计算} \approx a \cdot Bk,\qquad \text{可压缩搬运} \approx b \cdot T
$$

也就是说，$a \cdot Bk$ 是模型语义决定的必要工作量，而 $b \cdot T$ 往往包含大量可以通过调度、缓存、预取和重组来减少的等待时间。因此，在带宽受限、专家不能全量常驻的场景里，优先优化 $T$，通常比单纯优化算子本身更直接。

可以用一个玩具例子理解。假设一个 batch 有 16 个 token，每个 token 选 2 个专家，总共会形成 $16 \times 2 = 32$ 次 token-expert 计算，这部分工作量没有变。如果这些访问分散到了 20 个不同专家，系统就可能要搬运 20 次专家权重；如果通过更好的路由和调度，最终只访问 8 个专家，那么计算次数仍然是 32 次，但权重加载次数明显下降，延迟通常也会同步下降。

下表给出直观对比：

| 路由方式 | 平均访问专家数 $T$ | 计算量 $Bk$ | 单层延迟趋势 |
|---|---:|---:|---:|
| 普通随机路由 | 高 | 不变 | 常由 $b \cdot T$ 主导，延迟高 |
| 热点感知路由 | 中 | 不变 | 加载次数减少，延迟下降 |
| 重组 + 预取 + 缓存 | 低 | 不变 | 等待时间进一步压缩 |

一个公开数值例子是：在 $k=8, N=128, B=16$ 的设定下，Batch Prioritized Routing，简称 BPR，可把平均访问专家数从 48.8 降到 25.1，单层延迟从 175.7μs 降到 106.8μs，约提升 39%。这个结果说明，减少 batch 内的离散专家集合，往往能带来比“再挤一点算子效率”更直接的收益。

---

## 问题定义与边界

MoE 推理的标准流程可以拆成四步：

1. router 对每个 token 输出一组专家分数。
2. 系统从中选出 top-k 专家。
3. token 被分发到对应专家。
4. 每个专家执行自己的前馈计算，再把结果合并回主干。

如果只看数学表达，这个过程很简单；但一旦进入工程实现，问题就会出在“不同 token 访问了不同专家”。当 batch 内 token 的路由分布比较发散时，系统需要处理两类成本：

1. token 的分发成本。
2. 专家权重的加载成本。

其中第二类在大规模推理里经常更难处理。原因很简单：专家很多，显存有限，热点不断变化，权重不能总是留在最近的内存层级里。于是实际推理时间里会出现大量不连续、不可预测的等待。

对新手来说，可以先把问题理解成“取货和加工分离”的系统。

- router 相当于分单员，决定每张工单该送到哪些专家。
- 专家相当于不同的加工站。
- 权重加载相当于把加工站需要的物料搬到工作台。
- 真正做前馈计算，相当于加工本身。

MoE 的难点不是“加工站太慢”，而是“工单被派得到处都是，物料要不断来回搬”。

分析这个问题时，常用变量如下：

| 变量 | 含义 | 典型作用 |
|---|---|---|
| $B$ | batch 大小 | batch 越大，token 越有机会共享专家 |
| $k$ | 每个 token 的 top-k 专家数 | $k$ 越大，计算量和访问面都增大 |
| $T$ | 本批次实际访问专家数 | 决定专家加载次数，是核心延迟变量 |
| $b$ | 单次专家加载延迟 | 受内存层级、缓存命中、带宽影响 |
| $a$ | 单个 token-expert 的计算延迟 | 受 kernel、量化、并行映射影响 |
| $N$ | 专家总数 | $N$ 越大，访问离散化风险越高 |
| $C$ | 可驻留专家容量 | 可长期留在快内存里的专家数量 |
| $H$ | 缓存命中率 | 命中越高，有效 $b$ 越低 |

进一步说，$b$ 往往不是常数，而是一个“有效加载代价”。如果命中本地缓存，它可能很小；如果需要跨设备或从更慢的存储层拉取，它就会变大。因此工程上更实用的写法常常是：

$$
b_{\text{eff}} = H \cdot b_{\text{hit}} + (1-H)\cdot b_{\text{miss}}
$$

于是总延迟近似变成：

$$
L \approx b_{\text{eff}} \cdot T + a \cdot Bk
$$

这个写法更贴近真实系统，因为它把“是否命中缓存”也纳入了延迟估计。

边界需要说清楚。

第一，本文讨论的是推理，不是训练。训练还要处理反向传播、梯度同步、辅助负载均衡损失、优化器状态等额外问题，目标函数和约束完全不同。

第二，本文主要针对“专家数多、权重不能全量常驻、带宽或显存受限”的场景。如果模型规模较小，或者热点专家能稳定留在显存中，那么 $b \cdot T$ 不再主导，很多复杂调度策略的收益会明显下降。

第三，本文重点讨论“在不改变模型主语义的前提下”优化推理链路。也就是说，我们尽量不修改专家参数本身，而是优化这些环节：

- 哪些专家优先加载
- token 如何按专家重组
- 哪些热点专家应该保留在缓存中
- 路由在多大程度上允许向批次复用率倾斜

第四，本文关注的是单层和系统级延迟，不只看 FLOPs。因为 MoE 的真实问题常常不是算术量，而是“离散访存 + 碎片化执行 + 跨设备通信”。

下表总结了几类常见场景：

| 场景 | 主要瓶颈 | 是否优先优化 $T$ |
|---|---|---|
| 小模型、专家少、可常驻显存 | 计算核效率 | 否，优先级较低 |
| 大模型、专家多、热点明显 | 权重加载 + 缓存命中 | 是，优先级很高 |
| 跨卡专家并行、路由分布随机 | all-to-all 通信 + 加载 | 是，但需联动通信优化 |
| 小 batch、强随机在线请求 | dispatch 碎片 + 尾延迟 | 是，但收益受限 |

---

## 核心机制与推导

从近似公式

$$
L \approx b \cdot T + a \cdot Bk
$$

出发，可以直接得到两个结论。

第一，$a \cdot Bk$ 基本不可避免。因为每个 token 最终都要经过 top-k 个专家，这部分计算是模型结构本身要求的。你可以通过更好的 kernel、量化、算子融合、GroupGEMM 或张量并行映射降低 $a$，但不能把这部分工作量接近零。

第二，$b \cdot T$ 有明显的可优化空间。因为不同 token 之间原本就可能共享专家，只要让访问更集中、热点更稳定、预取更提前，$T$ 或有效的 $b$ 都能下降。

因此，MoE 推理优化的主线通常落在三件事上：

1. 预测下一段时间会用到哪些专家。
2. 把这些专家提前搬到更近的内存层级。
3. 把属于同一专家的 token 聚起来连续执行。

把这个逻辑写得更明确一点，就是：

$$
\text{优化目标} \neq \min(Bk), \qquad \text{优化目标} \approx \min(T) \text{ 或 } \min(b_{\text{eff}})
$$

这也是为什么很多 MoE 推理系统的收益主要来自调度，而不是来自更激进的数学压缩。

### 为什么降低 $T$ 的收益通常更直接

对 $T$ 求偏导：

$$
\frac{\partial L}{\partial T} = b
$$

这表示在其他变量固定时，每减少 1 个实际访问专家，延迟就近似减少一个固定的加载代价 $b$。如果当前系统的加载路径很慢，例如需要跨设备拉取权重，那么这个收益会非常直接。

再看计算项：

$$
\frac{\partial L}{\partial a} = Bk
$$

这说明降低单位计算代价 $a$ 当然也有收益，但它通常需要更重的底层工程，例如：

- 更高效的 GEMM kernel
- 更合适的量化格式
- 更好的算子融合
- 更细致的并行切分

这些优化是必要的，但工程门槛更高，而且它们不会改变“必须执行 $Bk$ 次 token-expert 计算”这个事实。

从趋势上看，在 $B,k,a,b$ 固定时，$L$ 与 $T$ 近似线性相关：

```text
L
^
|                                  *
|                            *
|                      *
|                *
|          *
|    *
+----------------------------------------> T
     4    8    12   16   20   24   28
```

如果 $b$ 很大，这条线会更陡。工程上这意味着：在带宽受限场景里，减少专家访问数通常能立刻带来可见收益。

### 玩具例子

设：

- $B = 4$
- $k = 2$
- $a = 1$
- $b = 10$

那么必要计算项为：

$$
a \cdot Bk = 1 \cdot 4 \cdot 2 = 8
$$

如果当前 batch 实际访问了 8 个专家，则总延迟近似为：

$$
L \approx 10 \cdot 8 + 8 = 88
$$

如果通过更集中的路由和重组，只访问 3 个专家，则：

$$
L \approx 10 \cdot 3 + 8 = 38
$$

延迟下降比例为：

$$
\frac{88 - 38}{88} \approx 56.8\%
$$

这里最关键的一点是：计算量完全没有变。减少的是“专家权重被反复搬运”的等待。这正是 MoE 推理优化和普通 dense 模型优化的最大区别之一。

为了避免新手只记住公式，可以把这个例子再翻成一句工程话：

- 你没有减少业务工作量。
- 你只是让原本分散在 8 个专家上的请求，集中到 3 个专家上执行。
- 于是系统少做了很多搬运和切换。

### BPR 的核心思想

BPR，Batch Prioritized Routing，可以理解为“让路由不仅考虑单个 token 的局部偏好，也考虑整个 batch 的专家复用率”。

普通 top-k 路由只关心当前 token 自己最喜欢哪些专家，而 BPR 额外关心一个问题：

$$
\text{如果把很多 token 都分到彼此重叠的专家集合里，整体会不会更快？}
$$

如果答案是“会”，那么系统就可以在可接受的精度约束下，对热点专家施加轻微偏置，使 batch 内更多 token 共享同一组专家。

可以把这个思路写成一个更完整的伪公式：

$$
\text{AdjustedScore}(t,e) = \text{RouterScore}(t,e) + \alpha \cdot \text{Hotness}(e) + \beta \cdot \text{BatchReuse}(e)
$$

其中：

- $\text{RouterScore}(t,e)$ 是 token $t$ 对专家 $e$ 的原始路由分数。
- $\text{Hotness}(e)$ 表示该专家在最近窗口内的热度。
- $\text{BatchReuse}(e)$ 表示该专家在当前 batch 中已经被多少 token 选中过。
- $\alpha,\beta$ 控制“精度偏好”和“复用偏好”的权衡。

伪代码如下：

```python
def bpr_route(logits, k, hotness, reuse_bias=0.1, hot_bias=0.2):
    num_tokens = len(logits)
    num_experts = len(logits[0])
    selected = []
    batch_count = [0] * num_experts

    for token_id in range(num_tokens):
        adjusted = []
        for expert_id in range(num_experts):
            score = logits[token_id][expert_id]
            score += hot_bias * hotness.get(expert_id, 0.0)
            score += reuse_bias * batch_count[expert_id]
            adjusted.append((score, expert_id))

        adjusted.sort(reverse=True)
        topk = [expert_id for _, expert_id in adjusted[:k]]
        selected.append(topk)

        for expert_id in topk:
            batch_count[expert_id] += 1

    return selected
```

这个过程没有改变“每个 token 仍然选 $k$ 个专家”的事实，但它试图让 token 们选出的专家集合更加重叠。也正因为如此，BPR 的收益常常直接表现为 $T$ 下降。

### 真实工程例子

在真实系统中，这种思想通常不会独立存在，而是和以下机制一起工作：

| 机制 | 作用 |
|---|---|
| GroupGEMM | 把同类小矩阵乘法打包，提高 GPU 利用率 |
| 权重预取 | 在真正需要前提前拉专家参数 |
| 热点缓存 | 把高频专家保留在更近的内存层级 |
| token 重组 | 把同专家 token 连续分派，减少碎片化执行 |
| 限流与回退 | 避免热点专家过载，必要时回退到更保守路径 |

系统级链路通常是这样一条流水线：

```text
router 输出 top-k
    ->
统计当前批次热点专家
    ->
预取高优先级专家权重
    ->
token 按专家重组
    ->
按专家分组执行 GroupGEMM
    ->
保留高频专家到缓存
    ->
更新下一个窗口的热度统计
```

这说明 MoE 推理优化本质上是“路由、访存、调度、kernel”联动，而不是单点技术。

---

## 代码实现

工程实现里，最关键的数据结构通常不是“token 原顺序数组”，而是“按专家聚合后的 bucket”。bucket 可以理解为“同一专家对应的一组 token 临时容器”。一旦有了 bucket，后续的预取、排序、分组执行和 GroupGEMM 才有可能做得高效。

下面给出一个可以直接运行的 Python 玩具实现，演示四件事：

1. 为每个 token 选 top-k 专家。
2. 统计当前 batch 的活跃专家集合。
3. 根据历史热度和当前负载生成预取优先级。
4. 按专家重组 token，模拟 grouped dispatch。

```python
from collections import defaultdict
from typing import Dict, List, Tuple


def topk_indices(scores: List[float], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(scores):
        raise ValueError("k cannot exceed num_experts")
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [expert_id for expert_id, _ in ranked[:k]]


def route_and_bucket(
    router_scores: List[List[float]],
    k: int,
    hotness: Dict[int, float],
    prefetch_limit: int = 2,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Tuple[Dict[int, List[int]], List[int], List[Tuple[int, List[int]]]]:
    """
    router_scores: shape = [num_tokens, num_experts]
    hotness: 历史热度
    alpha: 历史热度权重
    beta: 当前负载权重
    """
    if not router_scores:
        return {}, [], []

    num_experts = len(router_scores[0])
    for row in router_scores:
        if len(row) != num_experts:
            raise ValueError("all rows in router_scores must have the same length")

    expert_buckets: Dict[int, List[int]] = defaultdict(list)

    # 1. 每个 token 选 top-k 专家
    for token_id, scores in enumerate(router_scores):
        experts = topk_indices(scores, k)
        for expert_id in experts:
            expert_buckets[expert_id].append(token_id)

    active_experts = sorted(expert_buckets.keys())

    # 2. 生成预取优先级
    priorities = []
    for expert_id in active_experts:
        current_load = len(expert_buckets[expert_id])
        priority = alpha * hotness.get(expert_id, 0.0) + beta * current_load
        priorities.append((priority, expert_id))

    priorities.sort(reverse=True)
    prefetch_queue = [expert_id for _, expert_id in priorities[:prefetch_limit]]

    # 3. 模拟 grouped dispatch 的顺序
    dispatch_plan = sorted(
        ((expert_id, token_ids) for expert_id, token_ids in expert_buckets.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )

    return dict(expert_buckets), prefetch_queue, dispatch_plan


def simulate_latency(
    expert_buckets: Dict[int, List[int]],
    b: float,
    a: float,
) -> float:
    T = len(expert_buckets)
    Bk = sum(len(token_ids) for token_ids in expert_buckets.values())
    return b * T + a * Bk


def main() -> None:
    router_scores = [
        [0.90, 0.10, 0.80, 0.20, 0.05],
        [0.85, 0.30, 0.70, 0.10, 0.20],
        [0.20, 0.95, 0.10, 0.60, 0.50],
        [0.88, 0.05, 0.75, 0.20, 0.15],
        [0.10, 0.92, 0.30, 0.40, 0.80],
    ]
    hotness = {0: 10.0, 1: 6.0, 2: 8.0, 3: 2.0, 4: 5.0}

    buckets, prefetch_queue, dispatch_plan = route_and_bucket(
        router_scores=router_scores,
        k=2,
        hotness=hotness,
        prefetch_limit=3,
    )

    latency = simulate_latency(buckets, b=10.0, a=1.0)

    print("Expert buckets:")
    for expert_id, token_ids in sorted(buckets.items()):
        print(f"  expert {expert_id}: tokens {token_ids}")

    print("\nPrefetch queue:", prefetch_queue)

    print("\nDispatch plan:")
    for expert_id, token_ids in dispatch_plan:
        print(f"  dispatch expert {expert_id} with {len(token_ids)} tokens")

    print(f"\nEstimated latency: {latency:.1f}")

    assert len(buckets) > 0
    assert len(prefetch_queue) <= 3
    assert all(expert_id in buckets for expert_id in prefetch_queue)
    assert latency > 0


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出三类结果：

1. 每个专家最终分到哪些 token。
2. 哪些专家被优先预取。
3. 一个最简单的延迟估计。

如果想进一步理解数据结构，下表可以直接对照代码读：

| 字段 | 功能 | 为什么重要 |
|---|---|---|
| `router_scores` | router 给出的专家分数矩阵 | 决定每个 token 的候选专家 |
| `expert_buckets` | 按专家聚合 token | 后续 grouped dispatch 的基础 |
| `active_experts` | 当前 batch 实际访问的专家 | 直接决定 $T$ |
| `prefetch_queue` | 预取专家队列 | 决定哪些权重先搬 |
| `dispatch_plan` | 专家执行顺序 | 影响执行连续性和碎片度 |
| `hotness` | 历史热度统计 | 用于近似预测下一步访问模式 |

如果用更接近前端或服务端推理框架的伪 TypeScript 表示，结构通常如下：

```ts
type Token = { id: number; hidden: number[] };
type RouterChoice = { expertId: number; score: number };
type ExpertBuckets = Record<number, Token[]>;

function prefetch(expertId: number): void {
  // 异步触发权重加载
}

function dispatchGroupedTokens(expertId: number, tokens: Token[]): void {
  // 这里通常会调用 grouped GEMM / fused MLP kernel
}

function routeTokens(tokens: Token[], k: number) {
  const expertBuckets: ExpertBuckets = {};
  const selectedCount: Record<number, number> = {};

  for (const token of tokens) {
    const topk: RouterChoice[] = routerTopK(token.hidden, k);

    for (const item of topk) {
      if (!expertBuckets[item.expertId]) {
        expertBuckets[item.expertId] = [];
        selectedCount[item.expertId] = 0;
      }
      expertBuckets[item.expertId].push(token);
      selectedCount[item.expertId] += 1;
    }
  }

  const activeExperts = Object.keys(expertBuckets).map(Number);

  const sortedExperts = activeExperts.sort((lhs, rhs) => {
    return selectedCount[rhs] - selectedCount[lhs];
  });

  for (const expertId of sortedExperts.slice(0, 4)) {
    prefetch(expertId);
  }

  for (const expertId of sortedExperts) {
    dispatchGroupedTokens(expertId, expertBuckets[expertId]);
  }

  return { expertBuckets, activeExperts };
}
```

这个流程的关键点是：不要在“遍历 token”的同时立刻执行专家计算。因为那会造成三个问题：

1. token 级别 dispatch 太碎。
2. 同一专家可能被多次重复切入。
3. 底层 kernel 很难形成连续的大批次执行。

更合理的顺序通常是：

$$
\text{route} \rightarrow \text{bucket} \rightarrow \text{prefetch} \rightarrow \text{grouped dispatch}
$$

真实工程里，这一层还要和 GroupGEMM 协同。因为你即使在上层把 token 分好了，如果底层仍然以很多很小的微批次执行，那么 GPU 利用率仍然会上不去。

---

## 工程权衡与常见坑

MoE 推理优化不是“把访问专家数压到最低”这么简单。因为当你试图降低 $T$ 时，常常会同时影响通信、负载均衡、缓存占用和尾延迟。一个看起来合理的优化，可能会把瓶颈从“加载”转移到“通信”或“热点排队”。

第一类常见问题是 all-to-all 通信过重。  
如果专家并行切得很细，而 token 到专家的映射又非常随机，那么每轮路由后都可能触发大量跨卡数据交换。此时即使你降低了本地 $T$，系统仍然可能被通信拖慢。

第二类常见问题是负载不均衡。  
如果某几个专家变得特别热门，那么它们虽然提升了复用率，却也可能成为排队热点。平均延迟看起来下降了，P99 反而会上升。

第三类常见问题是预取过量。  
预取的目的，是把“未来大概率要用的专家”提前搬进来；但如果预测器不准，或者一次性预取太多冷门专家，就会造成缓存污染，甚至占满本来应该留给热点专家的带宽和空间。

第四类常见问题是 bucket 过碎。  
即使访问专家数减少了，如果每个专家只拿到很少 token，那么 GroupGEMM 和批量前馈的收益仍然有限。

第五类常见问题是只看平均值。  
MoE 服务真正影响用户体验的，常常是尾延迟，而不是均值。一个方案如果把平均延迟降了 10%，却让 P99 恶化 30%，通常不能算成功。

下表汇总常见坑与规避策略：

| 坑 | 典型现象 | 根因 | 规避策略 |
|---|---|---|---|
| all-to-all 过重 | 跨卡时间高于本地计算 | 路由过散、专家分布跨设备 | 优先本地聚合，减少碎片化跨卡分发 |
| 负载不均衡 | 少数专家排队严重 | 复用过于集中 | 设置 per-expert token cap 或回退策略 |
| 预取过量 | 缓存污染、带宽被错误占用 | 预测过于激进 | 限制预取窗口和预取数量 |
| bucket 太碎 | kernel 利用率低 | 单专家 token 太少 | 合并微批次，延后 dispatch |
| 只看平均延迟 | 线上抖动大 | 忽略尾部风险 | 同时监控 P95、P99、miss rate |

自适应预取，Adaptive Prefetch，常可以抽象成一个优先级函数：

$$
Priority(e) = \alpha \cdot Hotness(e) + \beta \cdot CurrentLoad(e) - \gamma \cdot MissCost(e)
$$

其中：

- $Hotness(e)$ 表示专家 $e$ 在历史窗口中的使用热度。
- $CurrentLoad(e)$ 表示当前 batch 里已经落到专家 $e$ 的 token 数量。
- $MissCost(e)$ 表示如果该专家不命中缓存，后续补载有多贵。
- $\alpha,\beta,\gamma$ 是需要通过实验调节的权重。

如果再细一点，可以把 miss 成本展开成内存层级的期望值：

$$
MissCost(e) \approx p_{\text{remote}} \cdot c_{\text{remote}} + p_{\text{host}} \cdot c_{\text{host}} + p_{\text{disk}} \cdot c_{\text{disk}}
$$

这说明预取不是“看谁热门就搬谁”，而是“谁热门、谁当前会被用、谁一旦 miss 代价更高，就更应该先搬”。

从新手视角，可以把整个工程权衡理解为一条原则：

- 不要只减少专家数。
- 要减少“无效搬运、碎片化执行、热点排队、错误预取”这几类总损耗。

一个更接近线上系统的评估表通常至少要包含：

| 指标 | 含义 | 为什么要看 |
|---|---|---|
| Mean Latency | 平均延迟 | 反映总体效率 |
| P95 / P99 | 高分位延迟 | 反映稳定性和尾部体验 |
| Active Experts $T$ | 活跃专家数 | 衡量路由集中度 |
| Cache Hit Rate | 缓存命中率 | 反映预取与缓存策略有效性 |
| All-to-All Time | 跨卡通信时间 | 判断瓶颈是否转移 |
| Expert Imbalance | 负载不均衡程度 | 识别热点专家风险 |

---

## 替代方案与适用边界

并不是所有 MoE 推理都适合同一套优化。是否值得做 BPR、热点缓存、Selective Loading 或 Hybrid MoE，取决于四个因素：

1. 专家访问分布是否稳定。
2. 显存是否足以容纳热点专家。
3. 远端带宽是否是主要瓶颈。
4. batch 是否大到足以形成专家复用。

常见方案可以概括为：

| 方案 | 核心思路 | 优势 | 局限 | 何时用 |
|---|---|---|---|---|
| BPR | 让 batch 内 token 更集中访问较少专家 | 直接降低 $T$ | 可能影响路由最优性 | 路由可轻微调整、目标是降延迟 |
| ExpertFlow | 用自适应预取和缓存跟踪热点专家 | 降低 miss 率和有效 $b$ | 依赖访问局部性 | 热点明显、时序可预测 |
| Selective Loading | 只按需加载部分专家 | 节省显存、适合大专家数 | miss 代价管理复杂 | 显存紧张、专家很多 |
| Hybrid MoE | 在部分场景回退到更稳定路径 | 时延更可控 | 可能增加计算量 | 小 batch、强随机请求 |

可以把这些方案分成两类。

第一类是“优先降低 $T$”的方案。  
代表是 BPR。它试图让更多 token 共享更少专家集合，从源头减少活跃专家数。

第二类是“即使 $T$ 不变，也尽量降低有效 $b$”的方案。  
代表是 ExpertFlow 和 Selective Loading。它们更强调：既然专家总会被访问，那就尽量把高概率访问的专家提前搬好，或者只搬真正需要的那部分。

还有一种保守但实用的思路是 Hybrid MoE。  
如果线上流量很随机、batch 很小、热点几乎不可预测，那么复杂的预取器和缓存系统未必值得。这时混合一部分 dense 路径，虽然算力开销可能略高，但时延往往更稳定。

一个简化决策流程如下：

```text
提速目标
  |
  +-- 访问分布可预测？ -- 是 --> 优先考虑 ExpertFlow / 热点缓存
  |                         |
  |                         +-- 同时希望直接减少 T --> 叠加 BPR
  |
  +-- 否 --> 显存是否紧张？ -- 是 --> Selective Loading
  |                         |
  |                         +-- 否 --> 重点做本地重组与通信优化
  |
  +-- batch 很小且强随机？ -- 是 --> Hybrid MoE / 保守调度
```

适用边界也要明确。

如果模型较小、专家总数不多、且大部分热点专家都能稳定驻留在显存中，那么复杂的预测和缓存系统可能不值得。因为此时额外调度开销本身就可能吞掉收益。

如果业务是高并发长序列服务，相邻 step 之间的专家访问有明显局部性，那么预取和缓存往往非常有价值。因为这类业务天然给了系统“预测下一步热点”的机会。

如果业务是强交互、小 batch、随机性很强的在线服务，最应优先做的是：

- 降低 all-to-all 碎片
- 提高本地 bucket 连续度
- 控制热点专家排队

而不是一开始就构建复杂的长窗口预测器。

换句话说，MoE 推理优化并没有单一最优解。真正有效的方案通常取决于一句话：

$$
\text{先判断瓶颈在哪，再决定是降 } T \text{，还是降 } b_{\text{eff}}
$$

---

## 参考资料

| 来源 | 内容摘要 | 对应本文位置 |
|---|---|---|
| NVIDIA: *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems* | 讨论大规模 MoE 系统中的专家并行、分组执行、GroupGEMM、跨设备调度与吞吐优化 | “核心机制与推导”“代码实现” |
| Emergent Mind: *Batch Prioritized Routing (BPR)* | 给出以 $L \approx b \cdot T + a \cdot Bk$ 为核心的延迟建模，并说明通过降低活跃专家数提升推理效率 | “核心结论”“核心机制与推导” |
| Emergent Mind: *ExpertFlow* | 重点介绍自适应预取、缓存命中与专家热度跟踪，适合理解如何降低有效加载代价 $b$ | “工程权衡与常见坑”“替代方案与适用边界” |
| AWS Neuron MoE Architecture Deep Dive | 解释按需加载、推理调度、内存层级管理与部署侧实现细节 | “问题定义与边界”“替代方案与适用边界” |
| Emergent Mind: *Expert Parallel (EP)* | 讨论专家并行下的 all-to-all、负载均衡、跨设备路由和系统瓶颈 | “工程权衡与常见坑” |

如果把这些资料和本文主线对应起来，可以得到一个更清晰的阅读顺序：

1. 先理解延迟模型：为什么 $T$ 很关键。
2. 再理解系统实现：为什么要 bucket、prefetch、grouped dispatch。
3. 最后看工程边界：什么时候该优先降 $T$，什么时候该优先降 miss 成本。

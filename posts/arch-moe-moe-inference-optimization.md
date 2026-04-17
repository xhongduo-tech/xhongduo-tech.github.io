## 核心结论

MoE，Mixture of Experts，中文常译为“专家混合模型”，指把一层前馈网络拆成多个专家子网络，只让每个 token 激活其中少数几个。MoE 的推理优化，核心不是单纯减少乘加计算，而是解决“权重很多但每次只用一小部分”带来的显存、带宽和调度问题。

MoE 推理优化可以压缩成一句话：把冷专家从 GPU 挪走，把可能马上要用到的专家提前搬回来，把不常用专家压缩存储，把多步小操作合并成更大的连续操作，最终让 token 的“路由 -> 专家计算 -> 输出合并”链路尽量不断流。

最关键的机制是稀疏路由。对输入表示 $x$，路由器根据 $xW_g$ 选出得分最高的两个专家，再把两个专家的输出按权重混合：

$$
y=\sum_{i=0}^{N-1}\text{Softmax}(\text{Top2}(xW_g))_i \cdot \text{SwiGLU}_i(x)
$$

这里的 `Top2` 表示只保留得分最高的两个专家；`SwiGLU` 是一种前馈激活结构，可以理解为“带门控的前馈网络”；$W_g$ 是路由器参数。公式的重点不在数学形式本身，而在它直接说明了一个事实：每次只有两条专家支路真正参与计算，但所有专家参数都必须随时可访问。

玩具例子可以这样看：假设一层有 8 个专家，每个 token 只会走 2 个专家。计算时你只需要这 2 个专家的矩阵乘法，但你不能提前知道下一 token 会选哪 2 个，因此 8 个专家的权重都得“候命”。这就是 MoE 在推理时经常“内存受限，而不是计算受限”的根源。

真实工程里，Mixtral 8x7B 常被拿来说明这个问题。它总参数量约 47B，但单 token 每步真正激活的参数大约只有 13B，剩余约 34B 虽然这一步没算到，仍然要放在可访问层级里。实测在 A100 上，`batch=1` 时原始推理常只有约 10 tok/s；做完专家卸载、预测预取、核融合、并行切分后，才能提升到 35 到 50 tok/s 这个量级。

---

## 问题定义与边界

MoE 推理优化讨论的是“推理阶段如何降低延迟、提高吞吐、减少显存压力”，不是训练阶段如何稳定路由，也不是如何提升模型上限精度。边界必须先说清，否则优化目标会混在一起。

第一条边界：关注的是小 batch，尤其是 `batch=1` 或很小的 decode 场景。decode 指自回归生成时一次只生成一个或少量 token 的阶段。这个阶段算力很难被大矩阵乘法吃满，因此 GPU 更容易空转。

第二条边界：MoE 的瓶颈通常不是 FLOPs，而是“参数搬运”。FLOPs 指浮点运算量。因为每次只激活很少专家，乘加数量确实降了，但路由带来的随机访问让权重装载、显存驻留、PCIe/NVLink 传输都变成主要成本。

第三条边界：prefill 和 decode 不能混着看。prefill 指把整段输入上下文一次性编码进去，这时序列长、并行度高；decode 指之后逐 token 生成，这时序列短、batch 小、最难优化。很多方案会把这两个阶段拆开设计，因为同一套缓存和调度策略往往无法同时最优。

下面用 Mixtral 8x7B 做一个典型数字对照：

| 指标 | 数值量级 | 含义 |
|---|---:|---|
| 总参数 | 47B | 模型全部参数规模 |
| 单步激活参数 | 约13B | 每个 token 实际经过的参数 |
| 非激活但需可访问参数 | 约34B | 本步没用到，但下一步可能随机命中 |
| 每层专家数 | 8 | 一层中可选的 FFN 专家数量 |
| 每 token 激活专家数 | 2 | Top-2 路由的典型设置 |
| 是否适合卸载 | 冷专家适合 | 高频专家不适合频繁搬运 |

这个表说明一个核心矛盾：激活参数小，不等于驻留参数小。对新手来说，最容易误解的是“只算 2 个专家，那剩下 6 个专家就不用管了”。实际上不对。因为路由是动态的，下一 token 可能立刻选到另外 2 个专家，所以剩余专家要么在 GPU，要么在 CPU/NVMe，而且要能在延迟预算内搬回来。

再用一个极简边界例子说明。假设显存只够放 20% 的专家，而路由又可能命中任意专家，那么你只能做三件事：

| 类别 | 作用 | 风险 |
|---|---|---|
| 热专家常驻 GPU | 保证高频命中低延迟 | 占显存 |
| 冷专家卸载到 CPU/NVMe | 节省显存 | 命中时搬运延迟高 |
| 预测并预取 | 提前隐藏搬运成本 | 预测错会浪费带宽 |

因此，MoE 推理优化的问题定义不是“如何让 MoE 计算更少”，而是“在随机路由、有限显存、严格延迟目标下，如何让专家访问尽量像连续流水线”。

---

## 核心机制与推导

MoE 层可以拆成三个动作：路由、专家计算、结果合并。

路由器先计算 $xW_g$。这里 $x$ 是当前 token 的隐藏状态，$W_g$ 是一个较小的线性层。得到分数后，取 Top-2 专家，再做 softmax 归一化。归一化的作用是把两个专家的得分转成加权系数，保证权重和为 1。

如果一个 token 选中了专家 1 和专家 5，且 softmax 后权重分别是 0.7 和 0.3，那么输出就是：

$$
y = 0.7 \cdot \text{SwiGLU}_1(x) + 0.3 \cdot \text{SwiGLU}_5(x)
$$

白话解释：这个 token 先问“哪两个专家最懂我”，再按信任度把两位专家的答案加权求和。

玩具例子如下。假设一个输入 token 是“矩阵乘法复杂度”，8 个专家分别偏向不同主题。路由器给出的原始分数可能是：

| 专家 | 主题 | 路由分数 |
|---|---|---:|
| E0 | 语法 | 0.2 |
| E1 | 线性代数 | 2.8 |
| E2 | 操作系统 | 0.1 |
| E3 | 网络 | -0.4 |
| E4 | 编译器 | 0.6 |
| E5 | 算法复杂度 | 2.1 |
| E6 | 安全 | -0.3 |
| E7 | 数据库 | 0.0 |

Top-2 是 E1 和 E5。softmax 后大致可得到 0.67 和 0.33。于是只有 E1、E5 的 FFN 被真正调用。这里“稀疏”并不是把模型删掉，而是“运行时只走少数路径”。

但这会带来两个直接后果。

第一，单 token 的工作量过小。传统 dense FFN 会让 GPU 跑一个较大的矩阵乘法；而 MoE 在 `batch=1` 时，只对少量 token、少数专家做小矩阵乘法，很容易让 CUDA core 闲置。CUDA core 可以理解为 GPU 里做数值计算的大量并行执行单元。任务太碎，它们就吃不满。

第二，访问模式变随机。不同 token 可能命中不同专家，导致 gather、GEMM、scatter 这些步骤频繁切换。gather 指把属于同一专家的 token 收拢；GEMM 指矩阵乘法；scatter 指把结果再分发回原位置。步骤一多，中间 buffer 增加，带宽浪费也增加。

因此主要优化路径可以按顺序理解为：

| 优化动作 | 解决的问题 | 直观效果 |
|---|---|---|
| Expert offloading | GPU 放不下全部专家 | 冷专家下放到 CPU/NVMe |
| Expert prediction | 搬运来不及 | 提前 1 到 2 步预取 |
| Expert compression | 带宽和显存仍紧张 | 冷专家用 INT4，热专家保 FP16 |
| FlashMoE | 操作过碎 | gather/GEMM/scatter 融成单 kernel |
| Tensor Parallel for experts | 单专家仍太大 | 一个专家切到多卡并行算 |

这里最值得推导的是“为什么预测能成立”。MoE 路由虽然动态，但并非完全随机。相邻层、相邻 token、相近语义片段常有局部稳定性。也就是说，前一步命中的专家集合，对后一步具有统计上的预测价值。于是可以训练一个很轻量的预测器，在当前 token 还没算完整层时，就提前猜测下一层甚至下两层要用哪些专家。

真实工程例子是 DuoServe-MoE 一类方案：prefill 阶段利用较强并行性做双流预取，decode 阶段用轻量专家预测器提早 1 到 2 层加载专家，让 GPU 常驻的只保留热点部分，其余参数在 CPU 甚至 NVMe。这样本质上是把“必需全驻留”改成“高概率命中部分驻留 + 其余按需异步补货”。

如果把整条链路写成文字流程，就是：

`当前层路由 -> 立刻发起下一层可能专家的异步拷贝 -> 当前层已命中专家执行 -> 当前层输出回主干 -> 下一层所需专家大概率已到位`

这条流水线成立时，GPU 等的不是数据，数据等的才是计算。

---

## 代码实现

下面先给一个可以运行的玩具实现。它不追求真实性能，只演示 Top-2 路由、加权输出和“可预测预取”的基本逻辑。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def top2_indices(scores):
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:2]
    return [pairs[0][0], pairs[1][0]]

def swiglu_like(x, scale, gate):
    # 玩具版 SwiGLU：只保留“线性 * sigmoid门控”的结构直觉
    sig = 1.0 / (1.0 + math.exp(-gate * x))
    return (scale * x) * sig

def moe_forward(x, gate_scores, expert_params):
    idx = top2_indices(gate_scores)
    top_scores = [gate_scores[i] for i in idx]
    weights = softmax(top_scores)

    y = 0.0
    for w, i in zip(weights, idx):
        scale, gate = expert_params[i]
        y += w * swiglu_like(x, scale, gate)
    return y, idx, weights

experts = {
    0: (0.5, 0.2),
    1: (1.4, 1.0),
    2: (0.3, 0.1),
    3: (0.7, 0.8),
    4: (1.1, 0.4),
    5: (1.2, 1.2),
    6: (0.2, 0.3),
    7: (0.6, 0.5),
}

y, idx, weights = moe_forward(
    x=2.0,
    gate_scores=[0.2, 2.8, 0.1, -0.4, 0.6, 2.1, -0.3, 0.0],
    expert_params=experts,
)

assert idx == [1, 5]
assert abs(sum(weights) - 1.0) < 1e-9
assert y > 0

# 一个极简“预测器”：假设下一步仍大概率命中当前最热专家之一
def predict_next_experts(current_idx):
    return set(current_idx)

prefetch = predict_next_experts(idx)
assert prefetch == {1, 5}
```

这个例子说明两件事：

1. Top-2 路由后只有两个专家真的参与计算。
2. 即便预测器很粗糙，只要能把“下一步大概率会用到谁”提前猜出来，就可以立刻发异步拷贝请求。

再看工程里更接近真实系统的伪代码。这里把专家缓存、异步预取和量化切换放在一起：

```python
class ExpertStore:
    def __init__(self, gpu_fp16, gpu_int4, cpu_pool):
        self.gpu_fp16 = gpu_fp16
        self.gpu_int4 = gpu_int4
        self.cpu_pool = cpu_pool

    def has_gpu(self, expert_id, dtype):
        pool = self.gpu_fp16 if dtype == "fp16" else self.gpu_int4
        return expert_id in pool

    def async_copy_to_gpu(self, expert_id, dtype):
        # 实际工程里这里会是 cudaMemcpyAsync / DMA / NVLink copy
        pool = self.gpu_fp16 if dtype == "fp16" else self.gpu_int4
        pool.add(expert_id)

def choose_dtype(expert_id, heat):
    # 热度高的专家保留高精度，冷专家压缩
    return "fp16" if heat[expert_id] >= 0.7 else "int4"

def step_decode(layer_id, token_state, predictor, heat, store):
    predicted = predictor.predict(layer_id + 1, token_state, topk=4)

    for expert_id in predicted:
        dtype = choose_dtype(expert_id, heat)
        if not store.has_gpu(expert_id, dtype):
            store.async_copy_to_gpu(expert_id, dtype)

    routed = predictor.route_current(layer_id, token_state)  # 当前层真实路由结果

    outputs = []
    for expert_id, weight in routed:
        dtype = choose_dtype(expert_id, heat)
        if not store.has_gpu(expert_id, dtype):
            # 预测失败时的兜底同步加载
            store.async_copy_to_gpu(expert_id, dtype)
        outputs.append((expert_id, weight, dtype))

    return outputs
```

如果把配置也写清楚，工程系统往往类似下面这种结构：

```json
{
  "prefill": {
    "prefetch_layers_ahead": 2,
    "cache_policy": "hot_experts_fp16"
  },
  "decode": {
    "predictor": "light_router_head",
    "fallback": "sync_load_on_miss"
  },
  "quantization": {
    "hot": "fp16",
    "cold": "int4",
    "strategy": "adaptive"
  },
  "memory": {
    "gpu_hot_pool_ratio": 0.15,
    "cpu_cold_pool": true,
    "nvme_spill": true
  }
}
```

实现时通常还要区分两个池子。一个池子存热点 FP16 专家，另一个池子存冷门 INT4 专家。双池的意义是避免单一缓存里不断替换不同精度格式，造成碎片和频繁重排。

还要强调一点：真正高性能实现不会只写在 Python 里。Python 更适合表达调度逻辑，而核心 kernel 要落到 CUDA、Triton 或框架的 fused operator。尤其是 FlashMoE 这类方案，重点就是把 `gather -> GEMM -> scatter` 融为单 kernel，减少中间 buffer 写回和重复访存。

---

## 工程权衡与常见坑

最常见的错误是只看“激活参数少”，忽略“参数访问乱”。结果是算子本身不慢，但 GPU 在等数据。

第一个坑是静态全量量化。量化指用更低位数表示权重，比如 INT4。把所有冷专家永久压成 INT4 看起来省显存，但一旦某个原本不热的专家在某段请求里突然频繁命中，就会变成“高频低精度”，质量可能明显下滑。DynaExq 一类思路的价值就在这里：不是所有专家都一刀切，而是按热度动态决定 FP16 还是 INT4。

| 方案 | 精度 | 延迟 | 显存 | 风险 |
|---|---|---|---|---|
| 静态统一 INT4 | 较低 | 低到中 | 最省 | 热专家精度崩 |
| 热度感知 FP16/INT4 | 较稳 | 中 | 较省 | 需要维护热度统计 |
| 双池管理 + 异步切换 | 稳定 | 低 | 中 | 实现复杂度高 |

第二个坑是预测器命中率被高估。预测器不是魔法。若预测错误率很高，就会出现两种坏结果：一是白搬数据，浪费 PCIe/NVLink 带宽；二是真正命中的专家还得同步加载，延迟反而更差。因此预测器必须足够轻，否则路由预测本身会吃掉节省出来的时间。

第三个坑是内存碎片。GPU 缓存里如果频繁插入、淘汰不同大小、不同精度格式的专家块，会让可用空间越来越不连续。最后不是“显存不够”，而是“连续大块不够”。工程上常见做法是固定槽位、分池、预留对齐空间，避免像通用堆分配那样碎片化。

第四个坑是 decode 和 prefill 用同一调度策略。prefill 并行度高，更适合大批量聚合、整层预取；decode 延迟敏感，更适合细粒度预测、短窗口预取。两阶段如果不拆开，通常两边都做不好。

第五个坑是只优化专家，不看通信。多卡下做专家并行或张量并行时，如果 All-to-All、All-Gather、Reduce-Scatter 的通信开销太大，单卡上的 kernel 再快也被抵消。尤其低 batch 下，通信启动开销会非常突出。

一个新手能立即理解的反例是：把所有专家都压成 INT4，确实可以在纸面上把显存降很多；但如果某个“原本冷门”的专家突然在金融、代码、长上下文场景里连续命中，输出质量可能立刻抖动。这不是量化技术本身失效，而是“热度是动态的，你却用了静态策略”。

---

## 替代方案与适用边界

不同方案优化的不是同一层问题，因此不能简单说谁“最好”。

| 方案 | 主要思路 | 适用场景 | 依赖/代价 |
|---|---|---|---|
| Expert Offloading | 冷专家下放 CPU/NVMe | 显存最紧张 | 依赖高效异步搬运 |
| Expert Prediction / PreScope | 提前预测下一步专家并预取 | decode 延迟敏感 | 需要额外预测器和调度器 |
| Adaptive Quantization / DynaExq | 热专家高精度，冷专家低精度 | 同时受显存与带宽限制 | 需要热度统计与双池管理 |
| FlashMoE | 融合 gather/GEMM/scatter | 内存带宽瓶颈明显 | 需要复杂 kernel 实现 |
| Tensor Parallel for experts | 单专家权重跨卡切分 | 单专家仍过大或想提高吞吐 | 通信开销显著 |

FlashMoE 适合的问题是“算子太碎”。它像把原本分散执行的几个动作卷成一个连续大操作，减少中间结果落盘和重复访存。它对内存带宽关键、kernel 启动开销显著的场景很有效，但代价是实现复杂，对底层算子栈要求高。

PreScope、Pre-Attention Expert Prediction 这类方法更像“调度器增强”。它们不直接改变专家结构，而是尽可能在路由真正发生前就判断下一步资源需求。适合 decode 延迟敏感、请求模式较稳定的场景。如果请求内容高度跳变，预测收益会下降。

Expert Offloading 是最直接也最常用的路线，因为很多时候问题首先就是“放不下”。但它单独使用通常不够，因为只要命中冷专家就会触发搬运停顿，所以往往必须和预测预取绑定使用。

Tensor Parallel for experts 适合“大专家仍然太大”的情况。张量并行指把一个大矩阵按列或按行拆到多张卡上共同计算。它能提升单专家计算效率，但会引入通信。如果 batch 很小、网络互联一般，收益可能不稳定。

因此适用边界可以概括为：

1. 显存优先不够时，先做卸载和压缩。
2. 延迟优先时，重点做预测预取和流水线调度。
3. 带宽优先受限时，重点做核融合。
4. 单专家太大时，再考虑张量并行或专家并行混合切分。

---

## 参考资料

1. FlashMoE：核心贡献是把 MoE 中的 gather、GEMM、scatter 等步骤做融合，并通过持续 kernel 减少中间 buffer 和访存开销，适合理解“为什么 MoE 经常卡在带宽而不是算力”。
2. PreScope / Pre-Attention Expert Prediction：核心贡献是把专家预测和预取调度前移，说明“路由虽然动态，但存在可利用的短期稳定性”。
3. DuoServe-MoE：核心贡献是把 prefill 和 decode 分阶段优化，并在 decode 中使用轻量预测器提前 1 到 2 层预加载专家，展示了真实系统里“异步预取 + 分层缓存”的做法。
4. DynaExq：核心贡献是热度感知量化与双池管理，重点回答“为什么不能把所有专家都静态压成 INT4”。
5. Mixtral 8x7B 相关技术解读：常用于提供 MoE 推理的典型数字背景，例如 47B 总参数、单步激活约 13B，以及小 batch 下 GPU 利用率低的问题。
6. TensorRT-LLM 中的 MoE 推理优化实践：适合理解专家并行、张量并行和低 batch 推理的组合策略，尤其是工业部署时的多卡切分思路。

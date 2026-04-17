## 核心结论

流水线并行的负载均衡，本质上是在做一件事：让每个 stage 的总耗时尽量接近。这里的 stage 是“流水线中的一个执行段”，白话说就是一组 GPU 负责的一段连续模型计算。只要有一个 stage 明显更慢，整条流水线的吞吐就会被它锁死。

对 Transformer 来说，很多人会先按“层数平均”去切，比如 40 层、4 个 stage，就直接切成 `10/10/10/10`。这对“只有标准 Transformer 块”的理想模型近似成立，因为各层 FLOPs 大体接近。FLOPs 是“浮点运算次数”，白话说就是一段计算大概要做多少乘加操作。但真实 GPT 类模型首尾并不只有 Transformer 块：最前面通常还有 Embedding，最后面还有 LM Head 或 loss projection。这两部分会额外带来计算、显存和通信压力，因此首尾 stage 往往比中间 stage 更重。

所以，负载均衡不能只看层数，必须看每个 stage 的总延迟：

$$
\text{stage\_latency} = \frac{\text{compute\_FLOPs}}{\text{GPU\_FLOPS}} + \text{comm\_time}
$$

整条流水线的吞吐近似由最慢 stage 决定：

$$
\text{Throughput} \approx \frac{1}{\max(\text{stage\_latency})}
$$

结论直接落地成一个操作原则：如果 Embedding 和 LM Head 额外吃掉了相当于若干层 Transformer 的负载，那首尾 stage 就应该少分一些 Transformer 层。40 层 GPT、PP=4 的经典例子里，默认 `10/10/10/10` 往往不如 `5/15/15/5`。Megatron-LM 用 `--decoder-first-pipeline-num-layers` 和 `--decoder-last-pipeline-num-layers` 做这件事，Megatron-Bridge 则提供 embedding/loss-aware split 自动补偿首尾负载。

---

## 问题定义与边界

先把问题说清楚。流水线并行是“把一个大模型按层切成几段，让不同 GPU 组像工厂流水线一样并行处理不同 micro-batch”。micro-batch 是“一个大 batch 被拆出来的一小份”，白话说就是为了让不同 stage 同时开工而切出来的小批次。

这个问题讨论的是“stage 内部负载是否均衡”，不是在讨论数据并行、张量并行或激活重计算本身。也不是所有模型都要做首尾补偿。只有当以下条件成立时，这个问题才值得重点处理：

| 条件 | 含义 | 对负载均衡的影响 |
|---|---|---|
| 模型是 GPT/Decoder-only 类结构 | 结构首尾有 Embedding 和输出投影 | 首尾 stage 天生不对称 |
| 词表较大 | 输出投影矩阵大，loss 计算重 | LM Head 可能显著增重 |
| PP stage 数较多 | 每段可分到的层数变少 | 首尾额外负载更容易放大 |
| 通信不是绝对主导 | 计算仍是主要瓶颈 | 调层数才有明显收益 |

一个新手常见误区是：既然每层 Transformer 差不多，那只要层数平均就等于负载平均。这个结论只在“每个 stage 只有 Transformer 块、且通信也近似一致”的情况下才成立。实际训练中，第一个 stage 通常还要做 token embedding lookup，最后一个 stage 还要做词表投影和 loss。lookup 可以理解为“按 token id 去大表里取向量”，白话说就是查字典拿词向量。LM Head 可以理解为“把隐藏状态投影回词表空间”，白话说就是把模型内部表示重新变成每个词的分数。

玩具例子可以这样看。假设单层 Transformer 负载记为 1，Embedding 额外负载约等于 5 层，LM Head 额外负载也约等于 5 层。40 层、4 stage 若切成 `10/10/10/10`，那四个 stage 的有效负载就接近：

- Stage 0: `Embedding + 10 = 15`
- Stage 1: `10`
- Stage 2: `10`
- Stage 3: `10 + LM Head = 15`

表面上层数平均，实际负载是 `15/10/10/15`。这意味着 Stage 0 和 Stage 3 会长期拖慢流水线。

可以用一个简化图示理解：

| Stage | 功能块 | 近似负载 |
|---|---|---|
| 0 | Embedding + 前若干层 Transformer | 偏重 |
| 1 | 中间若干层 Transformer | 常规 |
| 2 | 中间若干层 Transformer | 常规 |
| 3 | 后若干层 Transformer + LM Head | 偏重 |

因此，本文讨论的边界很明确：目标不是“层数平均”，而是“耗时平均”；对象不是抽象 Transformer，而是包含首尾附加算子的真实训练图。

---

## 核心机制与推导

为什么最慢 stage 决定吞吐？因为流水线并行并不是所有 stage 同时完成同一个 micro-batch，而是以错位方式工作。某个 stage 处理慢了，后续 stage 就会等待输入，前序 stage 也会因为缓冲占满而不能无限前推，最终形成全局节拍被最慢段控制。

把单个 stage 的时间拆开看，至少有两部分：

$$
\text{stage\_latency}_i = \text{compute\_time}_i + \text{comm\_time}_i
$$

进一步写成更工程化的形式：

$$
\text{stage\_latency}_i = \frac{\text{FLOPs}_i}{\text{Effective GPU Throughput}_i} + \text{P2P Comm}_i
$$

这里的 Effective GPU Throughput 是“有效 GPU 吞吐率”，白话说就是显卡在真实训练里真正跑出来的速度，不是理论峰值。P2P Comm 是“stage 间点对点通信时间”，白话说就是上一段把激活传给下一段需要花的时间。

如果每个 stage 都很均衡，那么：

$$
\max(\text{stage\_latency}_i)
$$

会下降，流水线填满以后，单位时间能完成的 micro-batch 就更多。于是有：

$$
\text{Throughput} \approx \frac{1}{\max_i(\text{stage\_latency}_i)}
$$

现在看 40 层 GPT、PP=4 的核心例子。设：

- 单层 Transformer 负载 = 1
- Embedding 额外负载 = 5
- LM Head 额外负载 = 5

那么默认切法 `10/10/10/10` 与调整后 `5/15/15/5` 的比较如下：

| 方案 | Stage 0 | Stage 1 | Stage 2 | Stage 3 | 最大负载 |
|---|---:|---:|---:|---:|---:|
| 默认 `10/10/10/10` | `5+10=15` | `10` | `10` | `10+5=15` | `15` |
| 调整 `5/15/15/5` | `5+5=10` | `15` | `15` | `5+5=10` | `15` |

只看这组数，会发现最大值似乎还是 `15`。为什么实践里它仍然常常更好？关键在两个点。

第一，中间 stage 只有 Transformer 块，执行更规则，算子形态更统一，容易接近稳定高利用率；首尾 stage 带 Embedding 或 LM Head 时，不只是 FLOPs 多，还可能有更差的显存访问局部性、额外 kernel、loss 相关同步。这意味着“等价 5 层”的说法只是粗略近似，真实训练里首尾额外负载往往不只体现在 FLOPs 上。

第二，真实系统看的不是静态层数，而是 profile 出来的 stage latency。profile 就是“测量程序在各阶段花了多少时间”，白话说就是先跑一轮，把每段耗时量出来再调。很多时候 Stage 0/3 比“多 5 层”还重，于是给首尾减层数能把真实延迟拉平。

更贴近工程的推导方式是：把首尾附加开销记成补偿项 $\Delta_{emb}$ 和 $\Delta_{lm}$。那么分层目标不再是让每个 stage 的层数相等，而是让：

$$
L_0 + \Delta_{emb} \approx L_1 \approx L_2 \approx L_3 + \Delta_{lm}
$$

其中 $L_i$ 是第 $i$ 个 stage 的 Transformer 层数。若 $\Delta_{emb}\approx\Delta_{lm}\approx 5$，而中间 stage 可承受约 15 层，那么一个自然解就是：

$$
L_0=5,\quad L_1=15,\quad L_2=15,\quad L_3=5
$$

这就是为什么 Megatron-LM 会提供“首尾单独指定层数”的参数，而不是强制均分。

真实工程例子更能说明问题。假设一个 32 卡训练任务，PP=8、TP=4。团队一开始按层数平均切分，profiling 结果显示首 stage 延迟是中间 stage 的 `1.5x`，尾 stage 是 `1.4x`。他们把 embedding-aware split 打开后，再手动把首 stage 的两层挪到下一个 stage，把尾 stage 的两层挪到前一个 stage，最终各 stage 延迟收敛到 `1.00x ~ 1.05x` 区间。这个变化不一定让单卡更快，但会显著减少整条流水线的空等时间。

---

## 代码实现

Megatron-LM 的手动方式比较直接：显式指定首尾 stage 的 Transformer 层数。

```bash
python pretrain_gpt.py \
  --pipeline-model-parallel-size 4 \
  --num-layers 40 \
  --decoder-first-pipeline-num-layers 5 \
  --decoder-last-pipeline-num-layers 5
```

参数含义可以直接对应到划分逻辑：

- `--pipeline-model-parallel-size 4`：把模型切成 4 个流水线 stage。
- `--num-layers 40`：总共有 40 个 Transformer 层。
- `--decoder-first-pipeline-num-layers 5`：第一个 stage 只放 5 个 Transformer 层，Embedding 仍和它共置。
- `--decoder-last-pipeline-num-layers 5`：最后一个 stage 只放 5 个 Transformer 层，LM Head 或 loss 相关计算仍和它共置。

如果使用 Megatron-Bridge，推荐先开自动补偿开关，再看 profiling 结果决定是否继续手调。配置思想如下：

```python
def split_layers(total_layers, stages, embed_cost, lm_head_cost):
    """
    一个简化的玩具分配器：
    - 每层 Transformer 成本记为 1
    - embedding / lm_head 的额外成本用“等价层数”表示
    """
    assert stages >= 2
    assert total_layers > 0
    assert embed_cost >= 0
    assert lm_head_cost >= 0

    total_equivalent = total_layers + embed_cost + lm_head_cost
    target = total_equivalent / stages

    first_layers = max(1, round(target - embed_cost))
    last_layers = max(1, round(target - lm_head_cost))
    middle_stages = stages - 2
    remaining = total_layers - first_layers - last_layers
    assert remaining >= middle_stages

    base_middle = remaining // middle_stages if middle_stages > 0 else 0
    extra = remaining % middle_stages if middle_stages > 0 else 0

    result = [first_layers]
    for i in range(middle_stages):
        result.append(base_middle + (1 if i < extra else 0))
    result.append(last_layers)

    assert sum(result) == total_layers
    return result

layers = split_layers(total_layers=40, stages=4, embed_cost=5, lm_head_cost=5)
assert layers == [5, 15, 15, 5]
print(layers)
```

这个代码块不是 Megatron 的真实实现，而是一个可运行的“玩具例子”：先把 Embedding 和 LM Head 视为若干“等价层”，再按目标负载反推首尾应该分多少层。它的价值不是替代官方调度器，而是帮助理解“为什么首尾层数应该更少”。

如果想把这个思路进一步变成 profile 驱动的脚本，可以这样建模：

```python
def estimate_stage_latency(layer_counts, embed_cost=0.0, lm_head_cost=0.0, comm=0.2):
    assert len(layer_counts) >= 2
    latencies = []
    for i, layers in enumerate(layer_counts):
        compute = float(layers)
        if i == 0:
            compute += embed_cost
        if i == len(layer_counts) - 1:
            compute += lm_head_cost
        latencies.append(compute + comm)
    return latencies

default_plan = [10, 10, 10, 10]
balanced_plan = [5, 15, 15, 5]

default_latency = estimate_stage_latency(default_plan, embed_cost=5, lm_head_cost=5)
balanced_latency = estimate_stage_latency(balanced_plan, embed_cost=5, lm_head_cost=5)

assert max(default_latency) == 15.2
assert max(balanced_latency) == 15.2
print(default_latency, balanced_latency)
```

这段代码故意展示了一个事实：如果你只用“理想化 FLOPs 等价模型”，`10/10/10/10` 和 `5/15/15/5` 可能得到相同最大值。它提醒一个关键工程原则：最终不能只看纸面 FLOPs，必须看真实 profiling。Megatron-Bridge 的自动 split 机制就是把“首尾有额外负载”这个事实显式纳入切分逻辑，而不是只看平均层数。

---

## 工程权衡与常见坑

最常见的坑，是把“每层计算量近似相同”误解成“任何 stage 只要层数相同就一定均衡”。真实训练里至少有四种因素会破坏这个假设：

| 因素 | 现象 | 后果 |
|---|---|---|
| Embedding 额外开销 | 首 stage 更慢 | 前半段 GPU 排队 |
| LM Head / loss 开销 | 尾 stage 更慢 | 反向末端堆积 |
| 通信不对称 | 某些 stage 传输更重 | latency 不只由 FLOPs 决定 |
| kernel 利用率差异 | 首尾算子形态更碎 | 等价 FLOPs 不能直接等价成等时长 |

因此，正确流程应该是：

1. 先做 stage-level profiling。
2. 看每个 stage 的前向、反向、通信时间。
3. 判断瓶颈是否稳定落在首尾。
4. 再调整 `first/last_pipeline_num_layers` 或启用自动 split。
5. 重新 profile，而不是一次调参后凭感觉结束。

一个真实工程场景可以写得更具体。某团队在 32 卡上训练一个中型 GPT，起初直接按平均层数切分。profile 发现：

| 阶段 | 调整前 latency(ms) | 调整后 latency(ms) |
|---|---:|---:|
| Stage 0 | 15.0 | 10.5 |
| Stage 1 | 10.1 | 10.2 |
| Stage 2 | 9.9 | 10.0 |
| Stage 3 | 10.0 | 10.1 |
| Stage 4 | 10.1 | 10.2 |
| Stage 5 | 10.0 | 10.1 |
| Stage 6 | 10.2 | 10.3 |
| Stage 7 | 14.5 | 10.6 |

调整方法不是“平均地每个 stage 都动一点”，而是只处理瓶颈最明显的首尾：把首 stage 的两层移到 Stage 1，把尾 stage 的两层移到 Stage 6，同时开启 embedding/loss-aware split。结果是最大 stage latency 从 `15.0ms` 降到 `10.6ms`，吞吐提升就来自这个最大值的下降。

还有几个实操层面的坑需要单独记住：

- 不要只看前向时间，反向和 loss 计算常常让尾 stage 更重。
- 不要只看理论 FLOPs，显存带宽和 kernel 碎片化会让首尾 stage 更差。
- 不要一次改太多参数，否则很难判断到底是哪项调整生效。
- 如果使用 Megatron-Bridge，先确认 `account_for_embedding_in_pipeline_split=true`。
- 同时确认 `account_for_loss_in_pipeline_split=true`，否则尾段仍可能偏重。
- profile 时固定 micro-batch size 和 sequence length，否则结果不可比。
- 如果已经是通信主导，继续调层数可能收益很小，甚至没有收益。

---

## 替代方案与适用边界

不是所有团队都适合手动调 `first/last_pipeline_num_layers`。实践里大致有三类方案：

| 方案 | 控制粒度 | 实现难度 | 适用场景 |
|---|---|---|---|
| 自动 embedding/loss-aware split | 中等 | 低 | 使用 Megatron-Bridge，追求快速稳定落地 |
| 手动指定首尾层数 | 高 | 中 | 已知模型首尾偏重，且团队能做 profiling |
| 自定义 pipeline layout / schedule | 很高 | 高 | 老版本框架、异构硬件、特殊结构模型 |

对零基础到初级工程师来说，推荐顺序很明确：先自动，再手动，最后才是自定义调度。自动方案的优点是成本低、容易维护，缺点是控制不够细。手动首尾调层的优点是直接有效，缺点是要自己建立 profiling 闭环。自定义 layout 的优点是可覆盖复杂情况，缺点是调度、通信、调试成本都明显上升。

一个适合新手理解的替代方案例子是：某老版本训练系统没有 Megatron-Bridge，也没有 embedding-aware split。团队先用 profiler 估出每层 Transformer 平均耗时，再把 Embedding 视为“约等价 3 层”，LM Head 视为“约等价 3 层”，于是手工做成：

- Stage 0 = Embedding + 3 层
- 中间 stage = 常规 9 到 10 层
- 最后一个 stage = 3 层 + LM Head

这类方案适用于框架能力不足但允许自定义 layout 的情况。

但也有明确不建议调整层数的场景：

- stage 数已经很少，例如只切成 2 段，首尾补偿空间有限。
- 序列很长、激活很大，通信已经成为主要瓶颈。
- 硬件异构严重，某些 GPU 本身更慢，此时仅调层数不足以解决问题。
- 模型结构不是标准 GPT，首尾额外负载未必集中在 Embedding/LM Head。

所以，流水线并行负载均衡并不是一个“固定答案”，而是一种测量后再分配的工程方法。它成立的前提是：stage latency 可以被可靠测量，且瓶颈主要来自首尾额外计算，而不是别的系统性问题。

---

## 参考资料

1. **NVIDIA Megatron-Bridge Performance Guide**  
   链接：`docs.nvidia.com/nemo/megatron-bridge/nightly/performance-guide.html`  
   重点：说明自动切分时可以显式考虑 Embedding 和 loss 的额外开销，给出 embedding-aware、loss-aware split 的调优方向。

2. **Megatron-LM Issue #1303**  
   链接：`github.com/NVIDIA/Megatron-LM/issues/1303`  
   重点：展示首尾 stage 层数的配置钩子，说明可以通过 `--decoder-first-pipeline-num-layers` 和 `--decoder-last-pipeline-num-layers` 做非均匀切分。

3. **Pipeline and Tensor Parallelism Strategies for Training LLMs on Limited VRAM**  
   链接：`researchgate.net/publication/398601065_Pipeline_and_Tensor_Parallelism_Strategies_for_Training_LLMs_on_Limited_VRAM`  
   重点：从更一般的并行训练视角说明了负载均衡、profiling 和 stage 划分之间的关系，适合把公式和系统行为连起来理解。

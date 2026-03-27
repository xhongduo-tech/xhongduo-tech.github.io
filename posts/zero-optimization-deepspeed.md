## 核心结论

ZeRO 的核心作用不是“让单张卡算得更快”，而是“让单张卡少存重复状态”。状态指训练时必须常驻或频繁访问的数据，主要包括参数、梯度、优化器状态。优化器状态就是 Adam 这类优化器保存的动量、二阶矩和主权重副本；它们不是模型结构，但往往比模型本身更占显存。

传统数据并行会在每张 GPU 上复制一整份参数、一整份梯度、一整份优化器状态。这样做实现简单，但内存浪费严重。ZeRO 的做法是把这些状态按数据并行进程分片，每张卡只保留自己负责的那一份，需要时再通过通信临时拼起来。因此，ZeRO 本质上是“训练状态管理的分布式重排”。

对初学者可以直接记成一句话：4 张 GPU 训练时，不一定要 4 张都背完整个模型家当，也可以每张只背自己那 1/4，缺的部分靠通信补齐。

ZeRO 分三层：

| 阶段 | 分片对象 | 每卡常驻内存变化 | 额外通信特征 | 典型收益 |
|---|---|---:|---|---|
| ZeRO-1 | 优化器状态 | 先降优化器内存 | 更新前后需要同步参数结果 | 适合先解决 Adam 状态过大 |
| ZeRO-2 | 优化器状态 + 梯度 | 再降梯度内存 | 反向中大量 `reduce-scatter` | 训练大模型常用平衡点 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 参数也不全量常驻 | 前向/反向按需 `all-gather` | 内存节省最大 |

如果用混合精度训练，通常前向和反向用 FP16 或 BF16，优化器更新仍保留 FP32 主权重或部分 FP32 状态。混合精度就是“计算用低精度省内存和带宽，更新用高精度保稳定”。ZeRO 和混合精度并不冲突，二者通常一起使用。

---

## 问题定义与边界

问题先说清楚：ZeRO 解决的是数据并行训练中的内存冗余，不是所有分布式训练问题的总解。

设模型参数量为 $P$。如果参数用 16 位存储，那么参数本体大约占 $2P$ 字节；梯度若也是 16 位，再占 $2P$；Adam 优化器常见还要有 FP32 主权重、一阶矩、二阶矩，合起来常接近 $12P$ 字节量级。于是单卡训练状态总量可近似写成：

$$
M_{\text{dp}} \approx S_{\text{param}} + S_{\text{grad}} + S_{\text{opt}} + S_{\text{act}} + S_{\text{buffer}}
$$

这里：

- 参数 `param`：模型权重本身。
- 梯度 `grad`：反向传播算出来、用于更新参数的导数。
- 优化器状态 `opt`：优化器为了下一步更新而保存的历史统计量。
- 激活 `act`：前向过程中为反向保留的中间结果。
- 缓冲 `buffer`：通信桶、临时拼接张量、碎片化等额外开销。

ZeRO 主要压缩的是前三项，而不是激活内存。因此它的边界很明确：

| 项目 | ZeRO 是否直接解决 | 说明 |
|---|---|---|
| 参数/梯度/优化器状态冗余 | 是 | 核心目标 |
| 激活过大 | 否，通常需配合激活重计算 | 两者可叠加 |
| 单层算子算不下 | 否 | 可能需要张量并行或算子切分 |
| GPU 算力不足 | 否 | ZeRO 重心是内存，不是 FLOPs 提升 |

在 $N$ 个数据并行进程下，ZeRO 对模型状态内存的理想估计可写成：

$$
M_{\text{zero}} \approx \frac{S_{\text{opt}} + S_{\text{grad}} + S_{\text{param}}}{N} + S_{\text{act}} + S_{\text{buffer}}
$$

这不是严格等式，因为真实系统里还有桶缓冲、预取、对齐、碎片和峰值时刻的临时占用，但它足够说明趋势：模型状态占用会随着并行进程数近似按 $1/N$ 缩小。

玩具例子可以这样看。假设一个模型训练时总状态需要 200GB，机器有 4 张 80GB GPU。传统数据并行要求每张卡都放 200GB，显然放不下。ZeRO 若把状态均分成 4 份，则每张卡理论上只需约 50GB 再加一些缓冲，就有可能落进硬件容量内。

所以，ZeRO 的前提是：你已经在做数据并行，瓶颈主要是“每卡状态复制太多”，而不是“单层本身放不下”。

---

## 核心机制与推导

ZeRO 的三个阶段可以理解为“从最贵的状态开始逐步去重”。

先给一个状态拆解表。下面用每个参数对应的典型字节数做直观说明，假设前向/反向用 FP16，优化器是 Adam，且保留 FP32 主权重：

| 状态 | 单参数典型占用 | ZeRO-1 后 | ZeRO-2 后 | ZeRO-3 后 |
|---|---:|---:|---:|---:|
| FP16 参数 | 2B | 2B | 2B | $2/N$ B |
| FP16 梯度 | 2B | 2B | $2/N$ B | $2/N$ B |
| FP32 主权重 | 4B | $4/N$ B | $4/N$ B | $4/N$ B |
| Adam 一阶矩 | 4B | $4/N$ B | $4/N$ B | $4/N$ B |
| Adam 二阶矩 | 4B | $4/N$ B | $4/N$ B | $4/N$ B |

### ZeRO-1：先分优化器状态

优化器状态通常最肥，尤其是 Adam。ZeRO-1 把 FP32 主权重和动量统计量分散到不同 rank。rank 就是分布式训练里的进程编号，可以理解为“第几张卡负责的那份任务”。

这样做后，每张卡不再保留全量优化器状态，只保留自己分到的分片。参数和梯度仍然是全量副本，所以前向和反向逻辑改动较小，通信模式也比较接近普通数据并行。

结论是：

$$
M_{\text{ZeRO-1}} \approx S_{\text{param}} + S_{\text{grad}} + \frac{S_{\text{opt}}}{N}
$$

### ZeRO-2：再分梯度

ZeRO-2 在 ZeRO-1 基础上继续把梯度也分片。关键技术是 `reduce-scatter`。它的含义是“先把多个进程上的梯度做规约，再把规约结果按分片散到各进程”。与常规 `all-reduce` 相比，每张卡最后不需要拿到全量梯度，只拿到自己负责更新参数所需的那一段。

这一步很重要，因为反向传播结束后，梯度原本是全量驻留的，模型越大越容易炸显存。ZeRO-2 通过 bucket，也就是“把大张量拆成若干通信桶”的方式，边反向边归约边释放，降低峰值占用。

可以用伪代码看流程：

```text
for bucket in backward_generated_grad_buckets:
    local_grad = bucket.grad
    shard_grad = reduce_scatter(local_grad across data_parallel_ranks)
    store_only_owned_shard(shard_grad)
    free_full_grad_buffer(bucket)
for owned_param_shard in local_optimizer_partition:
    optimizer_step(owned_param_shard, owned_grad_shard, owned_opt_state)
broadcast_or_allgather_updated_params_if_needed()
```

这里的重点不是语法，而是顺序：梯度一产生就尽快规约并切分，然后释放全量梯度缓冲，避免它长期停留在显存里。

于是有：

$$
M_{\text{ZeRO-2}} \approx S_{\text{param}} + \frac{S_{\text{grad}}}{N} + \frac{S_{\text{opt}}}{N}
$$

### ZeRO-3：连参数也分

ZeRO-3 最激进。它把参数本身也分片。于是每张卡平时不再持有全量模型参数，只持有自己那一片。某一层要做前向时，系统才把该层需要的参数 `all-gather` 到本地；该层算完，再把不需要的完整副本释放，继续只保留分片。

`all-gather` 的含义是“各进程把自己的一片发出来，大家都拼成完整结果”。这正是 ZeRO-3 的核心通信。

一个简化的前向逻辑如下：

```text
for layer in model.layers:
    full_param = all_gather(param_shards_for_layer)
    output = layer_forward(input, full_param)
    release(full_param_if_safe)
```

反向类似，只是会在计算梯度和参数梯度时再次触发相关参数的收集或复用。

因此：

$$
M_{\text{ZeRO-3}} \approx \frac{S_{\text{param}} + S_{\text{grad}} + S_{\text{opt}}}{N} + S_{\text{buffer}}
$$

为什么这里仍有明显的 `buffer`？因为参数虽然理论上被分片，但前向和反向运行时要临时拼完整层权重，所以峰值时刻会出现额外缓冲。ZeRO-3 的工程难点就在这里：理论最省，实际最依赖通信调度和桶大小。

### 混合精度为什么能和 ZeRO 叠加

混合精度的典型套路是：

- 前向计算用 FP16/BF16，减小参数读写和激活带宽。
- 梯度多以低精度通信，减小通信量。
- 优化器更新保留 FP32 主权重和统计量，维持数值稳定。

所以 ZeRO 并没有改变“更新规则”，只是改变“这些状态放在哪张卡上”。换句话说，混合精度解决的是“每个状态用多大精度存”，ZeRO 解决的是“这些状态由谁来存”。

真实工程例子是：用 DeepSpeed 在多张 GPU 上训练 10B 级 GPT 类模型时，常见组合就是 BF16/FP16 + ZeRO-2 或 ZeRO-3；如果显存仍紧张，再加 CPU offload，把优化器状态甚至参数分片挪到主机内存。

---

## 代码实现

下面先给一个可运行的 Python 玩具例子，用数字模拟 ZeRO-1/2/3 的状态内存变化。这里不依赖 DeepSpeed，只是帮助建立量化直觉。

```python
def estimate_memory_per_gpu(params_billion, dp_world_size, stage):
    """
    粗略估算每张 GPU 的模型状态内存（GB）
    假设：
    - 参数: FP16, 2 bytes
    - 梯度: FP16, 2 bytes
    - Adam 状态: FP32 master + m + v = 12 bytes
    """
    P = params_billion * 1_000_000_000
    s_param = P * 2 / 1e9
    s_grad = P * 2 / 1e9
    s_opt = P * 12 / 1e9
    n = dp_world_size

    if stage == 0:
        return s_param + s_grad + s_opt
    if stage == 1:
        return s_param + s_grad + s_opt / n
    if stage == 2:
        return s_param + s_grad / n + s_opt / n
    if stage == 3:
        return (s_param + s_grad + s_opt) / n
    raise ValueError("stage must be 0, 1, 2, or 3")


# 玩具例子：10B 参数，8 卡数据并行
m0 = estimate_memory_per_gpu(10, 8, 0)
m1 = estimate_memory_per_gpu(10, 8, 1)
m2 = estimate_memory_per_gpu(10, 8, 2)
m3 = estimate_memory_per_gpu(10, 8, 3)

assert m0 > m1 > m2 > m3
assert round(m3, 2) == round(m0 / 8, 2)

print("DP:", round(m0, 2), "GB per GPU")
print("ZeRO-1:", round(m1, 2), "GB per GPU")
print("ZeRO-2:", round(m2, 2), "GB per GPU")
print("ZeRO-3:", round(m3, 2), "GB per GPU")
```

如果把这个例子代入，会看到 ZeRO-3 的理想状态内存接近普通数据并行的 $1/8$。这只是状态内存，不包含激活和缓冲，所以真实显存不会完全等于这个值。

下面看 DeepSpeed 的典型配置。这个 JSON 片段展示 ZeRO-3 加参数与优化器 offload：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000,
    "allgather_bucket_size": 200000000,
    "prefetch_bucket_size": 50000000,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.0001,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  }
}
```

几个关键字段必须读懂：

| 字段 | 作用 | 直白解释 |
|---|---|---|
| `stage: 3` | 开启 ZeRO-3 | 参数、梯度、优化器状态都分片 |
| `contiguous_gradients` | 梯度连续存储 | 把分散梯度拷到连续缓冲，减少碎片 |
| `overlap_comm` | 通信计算重叠 | 尽量让通信藏在算子执行空档里 |
| `reduce_bucket_size` | 归约桶大小 | 桶越大，通信次数越少，但峰值缓冲越大 |
| `allgather_bucket_size` | 参数聚合桶大小 | 影响 ZeRO-3 参数收集的频率和峰值 |
| `prefetch_bucket_size` | 预取大小 | 提前把后续层参数拉过来，减少等待 |
| `offload_optimizer.device: "cpu"` | 优化器卸载到 CPU | 用主机内存换 GPU 显存 |
| `offload_param.device: "cpu"` | 参数分片可卸载 | GPU 放不下时进一步扩容 |

真实工程例子可以设成这样：单机 8 卡训练 30B 级模型，若显存主要被优化器状态和参数占满，先用 BF16 + ZeRO-2；若仍然放不下，再切到 ZeRO-3；如果 ZeRO-3 还是有峰值爆显存，就把 `offload_optimizer` 开到 CPU，必要时再加 `offload_param`。这条路线的本质是从“纯 GPU 分片”逐步过渡到“GPU + CPU 分层存储”。

---

## 工程权衡与常见坑

ZeRO 不是白拿内存。它用通信、同步复杂度和更精细的调度去换显存。

第一类权衡是桶大小。`reduce_bucket_size` 和 `allgather_bucket_size` 直接影响系统行为：

| 配置项 | 调大后的影响 | 调小后的影响 | 常见坑 |
|---|---|---|---|
| `reduce_bucket_size` | 通信次数少，但缓冲更大 | 通信更频繁 | 小到一定程度会被通信启动开销拖慢 |
| `allgather_bucket_size` | 参数聚合批次大，吞吐可能更高 | 等待次数增加 | 太大可能导致前向前卡住 |
| `prefetch_bucket_size` | 预取更积极 | 预取更保守 | 太激进会提前吃掉显存 |
| `overlap_comm` | 能隐藏部分通信 | 实现简单但等待更显性 | 某些模型结构下重叠效果有限 |
| `offload_optimizer` | GPU 显存明显下降 | 无 | CPU 带宽不足会拖慢 step |
| `offload_param` | 进一步扩容 | 无 | PCIe/NVMe 成为瓶颈时速度下降明显 |

一个典型新手坑是只盯着“平均显存”，不看“峰值显存”。ZeRO-3 理论上最省，但实际在层参数 `all-gather`、预取和梯度缓冲叠加时，某些时刻峰值可能高于直觉。如果桶太大，前向开始前需要先收一大坨参数，就会看到 GPU 在那一刻突然冲高并卡住。

第二类坑是通信链路。ZeRO-2/3 对网络带宽和延迟更敏感。NVLink、PCIe、跨节点 InfiniBand 的差异会直接决定“省下来的显存值不值得”。同一配置在单机 8 卡和多机训练上，表现可能差很多。

第三类坑是梯度连续性和内存碎片。`contiguous_gradients` 的目的不是神秘优化，而是让梯度按连续区域组织，便于 bucket 化和释放。如果关闭它，理论逻辑仍成立，但显存碎片可能让你在看起来“还有剩余显存”时仍然 OOM。

第四类坑是 offload 误区。ZeRO-Offload 的意思不是“白送显存还不降速”，而是“拿更慢的 CPU 或 NVMe 内存，换取更大的可训练模型”。主机内存带宽和 NUMA 拓扑配置不好时，offload 可能让吞吐掉得很明显。尤其是 `offload_param`，因为参数在前向和反向中被频繁访问，比只卸载优化器状态更容易被带宽卡住。

第五类坑是把 ZeRO 当成激活内存优化。不是。长序列、大 batch 或超深网络下，激活经常才是头号显存来源。这时要配合 activation checkpointing，也就是“前向少存中间结果，反向时重算一部分”。

可以把权衡总结成一句更直白的话：ZeRO-3 最像“显存最省但调度最复杂”的方案；ZeRO-2 往往是“省得很多、代价又没那么重”的中间点。

---

## 替代方案与适用边界

ZeRO 不是唯一的大模型训练路径。它更像“数据并行的内存强化版”。

先和几类常见方案做对比：

| 方案 | 改动对象 | 主要解决什么 | 代码侵入性 | 更适合什么场景 |
|---|---|---|---|---|
| ZeRO | 训练状态管理 | 参数/梯度/优化器状态冗余 | 低到中 | 完整模型训练、想少改模型代码 |
| 张量并行/模型并行 | 模型内部张量或层 | 单层算不下、单卡算子过大 | 中到高 | 超大层、超大 hidden size |
| 流水线并行 | 模型按层切阶段 | 单卡放不下整网、跨设备分层执行 | 中到高 | 深模型、可容忍 pipeline bubble |
| LoRA/Adapter | 只训练少量增量参数 | 降低可训练参数量 | 低 | 微调而非全量预训练 |
| FSDP | 参数和梯度的全分片 | 类似 ZeRO-3 思路 | 中 | PyTorch 原生生态更紧密的场景 |

ZeRO 和流水线并行的区别，初学者可以这么理解：ZeRO 改的是“训练时谁来保存状态”，流水线并行改的是“模型本体怎么切开分给不同设备算”。前者更像仓库管理，后者更像生产线切段。

ZeRO 和 LoRA 也不是互斥关系。LoRA 的核心是“少训练一部分参数”，属于参数高效微调；ZeRO 的核心是“即使全量训练，也别在每张卡上重复存一份状态”。如果你只做 7B 模型的低成本微调，LoRA 往往比 ZeRO 更直接；如果你做全量预训练或大规模全参数微调，ZeRO 更对症。

适用边界可以总结为：

- 如果模型完整参数在单卡都放不下，但每一层本身还能算，优先考虑 ZeRO。
- 如果单层矩阵乘法本身就超出单卡能力，ZeRO 不够，需要张量并行。
- 如果你只训练极少量适配参数，LoRA 比 ZeRO 更省事。
- 如果你想尽量靠近 PyTorch 原生分片生态，FSDP 是重要替代方案；它与 ZeRO-3 在思想上接近，但具体实现和调优接口不同。

因此，ZeRO 的最佳位置不是“取代所有分布式方案”，而是“在尽量少改模型代码的前提下，把数据并行的可训练上限往上推”。

---

## 参考资料

- DeepSpeed 官方文档
  - ZeRO 文档
  - ZeRO-3 配置说明
  - ZeRO Offload 相关配置项说明

- DeepSpeed 官方教程
  - ZeRO 教程
  - ZeRO-Offload 教程
  - Large Model Training with DeepSpeed 教程

- DeepSpeed 官方博客与论文线索
  - ZeRO-Offload 博客
  - ZeRO-3 Offload / ZeRO-Infinity 相关文章
  - ZeRO 系列论文说明页

- 社区整理资料
  - DeepWiki 对 ZeRO Stage 1/2/3 的实现汇总
  - DeepWiki 对内存估算函数和配置项的整理

这些资料分别覆盖机制说明、配置字段、内存估算和工程案例。想继续深入，优先读 DeepSpeed 文档中的 ZeRO 章节和官方教程示例，因为它们最接近实际配置与运行方式。

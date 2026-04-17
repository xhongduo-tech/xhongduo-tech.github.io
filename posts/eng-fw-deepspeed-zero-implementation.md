## 核心结论

ZeRO（Zero Redundancy Optimizer，零冗余优化器，意思是把原本每张卡都重复保存的训练状态拆开存）本质上不是改变神经网络的数学计算，而是改变**训练状态在多张 GPU 上的存放方式**。传统数据并行里，每张 GPU 都有完整参数、完整梯度、完整优化器状态；ZeRO 则把这些状态按数据并行进程切片，让每张卡只保存自己负责的那一份，从而把显存需求从“每卡保存全量”降到“每卡只保存约 $1/N$ 的分片”。

ZeRO 分三步扩展。Stage 1 只分区优化器状态；Stage 2 再分区梯度；Stage 3 连参数也分区。阶段越高，显存节省越大，但通信和调度越复杂。它成立的关键原因是：**训练时并不是所有状态都必须长期完整驻留在每张卡上**，很多数据只在某个时刻被某个模块短暂使用，因此可以通过 `reduce-scatter`、`all-gather`、prefetch 和 offload 在“需要时取来，用完再释放”。

对初级工程师最重要的判断标准只有两个。第一，模型是否已经被单卡显存卡死；第二，通信链路是否足够支撑更多的分片与聚合。如果前者是主要矛盾，ZeRO 往往有效；如果后者更糟，Stage 3 甚至可能让吞吐下降。

| Stage | 分区对象 |
|-------|-----------|
| 1 | Optimizer state（例如 Adam 的一阶矩、二阶矩） |
| 2 | Optimizer state + Gradient |
| 3 | Optimizer state + Gradient + Parameter |

---

## 问题定义与边界

先定义问题。数据并行（Data Parallel，意思是每张卡跑同一份模型，只处理不同批次样本）在工程上最简单，但它有一个明显缺陷：**状态复制冗余**。如果一个模型参数占 $S_{\text{param}}$，梯度占 $S_{\text{grad}}$，优化器状态占 $S_{\text{opt}}$，那么传统数据并行中每张 GPU 都要承担

$$
M_{\text{DP}} \approx S_{\text{param}} + S_{\text{grad}} + S_{\text{opt}}
$$

这不是总集群内存，而是**每卡都要付一次**。模型一大，单卡先 OOM，而不是总 GPU 数不够。

ZeRO 的目标不是减少总训练信息量，而是减少**单卡必须长期持有的信息量**。如果数据并行度是 $N$，理想化情况下，ZeRO Stage 3 的每卡显存需求近似写成：

$$
M_{\text{per\ GPU}} \approx \frac{S_{\text{opt}} + S_{\text{grad}} + S_{\text{param}}}{N} + M_{\text{comm}}
$$

其中 $M_{\text{comm}}$ 是通信缓冲，意思是为了做聚合、广播、预取而额外占用的临时内存。这个公式说明了 ZeRO 的收益和边界：

1. 当 $N$ 增大时，主状态内存按约 $1/N$ 缩小。
2. 当通信缓冲、网络延迟、CPU/NVMe 带宽成为瓶颈时，节省不会线性兑现。
3. ZeRO 只解决“状态复制”问题，不解决算力不足、激活内存过大、算子低效等其他瓶颈。

一个玩具例子最容易理解。假设 4 张 GPU 训练一个模型，参数 8GB，梯度 2GB，Adam 优化器状态 4GB。传统数据并行时，每张卡都要保存这三部分，总共 14GB。  
如果切到 ZeRO Stage 1，每张卡只保留自己负责的 1/4 优化器状态，即 1GB，但仍保留完整参数和完整梯度，因此每卡大约是 $8+2+1=11$GB。  
切到 Stage 2，每张卡连梯度也只保留 1/4，于是是 $8+0.5+1=9.5$GB。  
切到 Stage 3，参数也分区，理想上每卡是 $(8+2+4)/4=3.5$GB，再加通信缓冲。这个量级变化解释了为什么 Stage 3 能把“根本放不下”的模型变成“能训练，但通信重”。

真实工程里，优化器状态往往比新手直觉更大。以 Adam 为例，参数本体通常不是全部，额外还要维护一阶矩和二阶矩，混合精度下还可能有 FP32 master weight，因此优化器状态常常是显存大头。官方教程里，8 卡场景下仅 Adam 状态如果占 18GB，那么 ZeRO Stage 1 分区后每卡只需约 2.25GB，这类场景就是 Stage 1 的直接适用边界：**模型参数还放得下，但优化器状态放不下**。

---

## 核心机制与推导

ZeRO 的核心不是“把内存缩小了”，而是“把什么时候需要哪些状态”这件事精细化了。

Stage 1 最简单。优化器状态被按参数分区，每个 rank 只维护自己分片对应参数的一阶矩、二阶矩等状态。反向传播后，相关梯度需要先被汇总到负责该分区的 rank，再由该 rank 执行 optimizer step。这意味着更新职责是分布式的，不再是每张卡各自更新完整模型副本。

Stage 2 在此基础上继续分区梯度。梯度分区通常依赖 `reduce-scatter`。`reduce-scatter` 可以理解为“一边求和，一边切分结果分给不同 rank”。与先全量 `all-reduce` 再手工切片相比，它更贴合 ZeRO 的目标，因为最终每张卡本来也只需要自己负责的梯度分片。于是，优化器状态和梯度都变成按 rank 管理，显存进一步下降。

一个简单推导如下。设优化器状态为 4GB，梯度为 2GB，$N=4$。  
Stage 2 中，每张卡只保留：

$$
\frac{4+2}{4} = 1.5\text{GB}
$$

的 optimizer+gradient 主体数据，再加完整参数和通信缓冲。对比传统数据并行中每卡要保存完整 6GB，这里已经把这两类状态缩到了 1/4。

Stage 3 难点最高，因为它连参数本体也不再常驻完整副本。参数分区后，前向传播到某个模块时，当前 rank 若缺少这个模块所需参数，就通过 `all-gather` 临时把完整参数拉齐；该模块计算结束后，如果参数短期内不会复用，就释放回分片状态。反向传播也采用类似策略，因此参数从“永久驻留”变成“按模块生命周期临时组装”。

这里有一个很重要的调度概念：参数状态机。Stage 3 中常见的状态可以简化理解为三种：

| 状态 | 含义 | 工程动作 |
|------|------|----------|
| `NOT_AVAILABLE` | 当前 rank 没有完整参数可用 | 需要触发 gather |
| `INFLIGHT` | 参数正在通信途中 | 等待通信句柄完成 |
| `AVAILABLE` | 参数已经可供当前模块使用 | 执行前向或反向计算 |

这个状态机的意义是：ZeRO Stage 3 不是粗暴地“缺了就同步”，而是通过 hook 和 coordinator 提前知道下一个模块会用到什么参数，于是可以做 prefetch。prefetch（预取，意思是在真正用到之前先把数据拉过来）把一部分通信隐藏到计算阶段里，从而减少空等时间。

因此，Stage 3 的显存近似式可以写成：

$$
M_{\text{stage3}} \approx \frac{S_{\text{opt}}+S_{\text{grad}}+S_{\text{param}}}{N} + M_{\text{comm}}
$$

其中 $M_{\text{comm}}$ 的大小与 `reduce_bucket_size`、`allgather_bucket_size`、`stage3_prefetch_bucket_size` 等设置直接相关。它们本质上都在控制一句话：**每次通信搬多大一块数据，以及要不要提前搬**。

真实工程例子能看清这套机制的价值。微软公开过 Turing-NLG 17B 的训练案例，DeepSpeed 通过 ZeRO-OS/ZeRO-Offload 让原本需要更大 GPU 规模的训练，在更少 GPU 上变得可行。这里的关键不只是“分片”，而是“分片 + CPU/存储 offload + 动态预取”组合在一起，把热数据留在 GPU，把冷数据挪到更便宜的层级。

---

## 代码实现

从代码结构看，Stage 1/2 和 Stage 3 是两套复杂度明显不同的实现路径。

Stage 1/2 的主体逻辑在 `deepspeed/runtime/zero/stage_1_and_2.py`。核心职责包括：

1. 给参数分配分区归属。
2. 在反向传播过程中收集梯度。
3. 按 bucket 触发 `reduce-scatter` 或相关通信。
4. 在 `step()` 中只更新当前 rank 负责的那部分参数状态。
5. 在必要时再同步参数视图。

这里的 bucket（桶，意思是把很多小 tensor 拼成较大块统一通信）非常关键。GPU 通信对“很多很小的消息”极不友好，因此 ZeRO 不会为每个参数单独通信，而是把梯度和参数片段攒成较大块再传。

Stage 3 的主逻辑在 `deepspeed/runtime/zero/stage3.py`，并依赖 `partition_parameters.py`、`partitioned_param_coordinator.py` 等模块。和 Stage 1/2 相比，新增的难点主要有三个：

1. 参数本体也要分区管理。
2. 前向、反向都要按模块粒度做 gather/release。
3. offload 到 CPU/NVMe 时，要额外处理设备间搬运和预取顺序。

下面先给一个最常见的配置示例。它不是源码本体，但能直接反映 ZeRO 的工程接口：

```json
{
  "train_batch_size": 32,
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

`contiguous_gradients` 的意思是把梯度尽量整理成连续内存，减少碎片和通信整理成本。`offload_optimizer` 表示优化器状态不全留在 GPU，而是下沉到 CPU。对很多“显存紧、PCIe 还行”的单机场景，这是性价比很高的一步。

下面用一个可运行的 Python 玩具程序模拟 ZeRO 不同阶段下的每卡主状态内存。这个程序不是 DeepSpeed 源码，而是把它的核心内存模型抽象出来：

```python
def zero_memory_per_gpu(param_gb, grad_gb, opt_gb, world_size, stage, comm_buffer_gb=0.0):
    assert world_size > 0
    assert stage in (0, 1, 2, 3)

    if stage == 0:  # 普通数据并行
        return param_gb + grad_gb + opt_gb + comm_buffer_gb
    if stage == 1:
        return param_gb + grad_gb + opt_gb / world_size + comm_buffer_gb
    if stage == 2:
        return param_gb + grad_gb / world_size + opt_gb / world_size + comm_buffer_gb
    return (param_gb + grad_gb + opt_gb) / world_size + comm_buffer_gb


param_gb = 8
grad_gb = 2
opt_gb = 4
world_size = 4

stage0 = zero_memory_per_gpu(param_gb, grad_gb, opt_gb, world_size, 0)
stage1 = zero_memory_per_gpu(param_gb, grad_gb, opt_gb, world_size, 1)
stage2 = zero_memory_per_gpu(param_gb, grad_gb, opt_gb, world_size, 2)
stage3 = zero_memory_per_gpu(param_gb, grad_gb, opt_gb, world_size, 3, comm_buffer_gb=0.5)

assert stage0 == 14
assert stage1 == 11
assert stage2 == 9.5
assert stage3 == 4.0
assert stage3 < stage2 < stage1 < stage0
```

这段代码的意义在于把 ZeRO 说成一句可验证的话：**Stage 越高，每卡常驻主状态越少，但通信缓冲不会消失。**

如果从源码行为理解，一次训练 step 可以粗略想成下面的顺序：

| 阶段 | 核心动作 |
|------|----------|
| Forward | Stage 3 在模块执行前 all-gather 所需参数 |
| Backward | 梯度产生后按 bucket 聚合，Stage 2/3 做 reduce-scatter |
| Optimizer Step | 每个 rank 更新自己负责的 optimizer partition |
| Post Step | 必要时释放完整参数，仅保留分片 |

这也是为什么 Stage 3 往往和 `GatheredParameters`、参数 hook、prefetch 调度一起出现。参数不是“有没有”，而是“当前时刻是不是完整可用”。

---

## 工程权衡与常见坑

ZeRO 的第一类权衡是**显存与通信互换**。你把常驻数据切掉了，就一定会在某些时刻多出 gather 或 scatter。节省显存不是免费午餐，而是把成本转移到了网络、PCIe、CPU 内存和调度复杂度上。

最典型的坑来自 bucket 参数。`reduce_bucket_size` 和 `allgather_bucket_size` 控制每次通信搬多少元素。太小，通信次数暴增，延迟主导；太大，缓冲本身就可能把显存顶满。对小显存卡尤其危险。

| 参数 | 作用 | 风险 |
|------|------|------|
| `reduce_bucket_size` | 每次 `reduce-scatter` 的数据块大小 | 太小频繁通信，太大占用缓冲 |
| `allgather_bucket_size` | 每次 `all-gather` 的数据块大小 | 太小调度开销高，太大导致峰值显存升高 |
| `overlap_comm` | 通信与计算重叠 | 会放大缓冲占用，低显存卡容易 OOM |
| `stage3_prefetch_bucket_size` | Stage 3 预取粒度 | 太小拉取过碎，太大预取过度 |

一个常见真实坑是 `overlap_comm`。它的目标是把通信塞进计算空隙里，但这通常要多准备几份缓冲区。对 8GB 级 GPU，如果直接照搬大模型默认配置，单是通信缓冲就可能吃掉数 GB，结果不是更快，而是直接 OOM。很多教程里给出的超大 bucket 值，默认假设的是 A100 这类高显存设备，不适合消费级卡。

第二类坑是**Stage 3 + offload 的链路不匹配**。offload（卸载，意思是把 GPU 放不下或暂时不用的数据挪到 CPU 或 NVMe）并不自动意味着高效。如果是 NVMe -> CPU -> GPU 的三级搬运，而 `stage3_prefetch_bucket_size`、`stage3_max_live_parameters`、`stage3_max_reuse_distance` 之类参数没调好，就会出现下面的问题：

1. 数据总是“刚要用时才开始搬”，GPU 空等。
2. 预取太激进，把 host 内存挤爆或制造无效传输。
3. 参数复用距离估计不合理，刚释放又重新拉取。

第三类坑是**把 ZeRO 当成万能扩展方案**。它不解决激活值占用过大时的瓶颈。比如长序列 Transformer，激活内存和 attention 临时张量可能比参数状态更大，这时只上 ZeRO 收益有限，还要配合 activation checkpointing、序列并行或 FlashAttention 一类优化。

第四类坑是**多节点环境中的不一致配置**。ZeRO 的分区假设所有 rank 对参数布局、分组、offload 目标的理解完全一致。如果不同节点上某些参数没被相同方式包装，或者 world size、通信组、checkpoint 恢复逻辑不一致，错误会非常隐蔽，经常表现为训练挂死、loss 异常或恢复失败，而不是直接报一个清晰异常。

---

## 替代方案与适用边界

最直接的替代方案是普通数据并行。它的优点是实现简单、调试容易、吞吐稳定，适合小模型或高显存卡场景。但一旦

$$
S_{\text{param}} + S_{\text{grad}} + S_{\text{opt}}
$$

超过单卡可承受范围，它就没有继续扩展的空间。

另一类替代方案是张量并行（Tensor Parallel，意思是把单层算子本身拆到多卡计算）和流水线并行（Pipeline Parallel，意思是把不同层分给不同卡顺序执行）。它们解决的是“单个模型算子或层堆叠放不下/算不过来”的问题，而 ZeRO 解决的是“数据并行复制太浪费”的问题。二者不是互斥关系，真实大模型训练里常常组合使用。

| 方案 | 显存占用特点 | 适用边界 |
|------|--------------|----------|
| 数据并行 | 每卡保存完整参数、梯度、优化器状态 | 小模型、追求简单稳定 |
| ZeRO Stage 2 | 分区优化器状态和梯度 | 参数还能放下，但状态冗余太大 |
| ZeRO Stage 3 | 参数也分区 | 单卡放不下完整模型，但网络尚可 |
| ZeRO-Offload / Infinity | 在 Stage 3 基础上继续下沉到 CPU/NVMe | GPU 显存极紧，但主机内存/存储带宽较强 |
| 张量并行 / 流水线并行 | 拆分模型计算图本身 | 单层太大或总算力扩展要求更高 |

对初级工程师，一个实用判断可以写成：

1. 如果模型本体能放下，只是 Adam 状态太大，先试 ZeRO Stage 1。
2. 如果参数能放下，但梯度和优化器一起顶爆显存，试 Stage 2。
3. 如果连完整参数都放不下，再进入 Stage 3。
4. 如果 Stage 3 仍不够，再考虑 offload。
5. 如果瓶颈已不是状态，而是激活或单层算子规模，ZeRO 不是主解法。

真实工程例子说明它的上界。DeepSpeed 在 Turing-NLG 17B 训练中结合 ZeRO-OS，把原先更依赖大规模 GPU 堆叠的方案压缩到更少卡数上完成，同时维持较高吞吐和更大 batch。这类案例说明 ZeRO 的价值不是学术上“更优雅”，而是工程上把“不可能训练”变成“可以训练且成本下降”。但同样要看到，它依赖高质量通信、预取策略和内存层次协同，不是简单改一项配置就自然生效。

---

## 参考资料

- ZeRO 官方文档：阶段定义、参数分区、Stage 3 生命周期与配置  
  https://deepspeed.readthedocs.io/en/stable/zero3.html
- DeepSpeed ZeRO 教程：Stage 1/2/3 配置示例与使用说明  
  https://www.deepspeed.ai/tutorials/zero/
- DeepWiki: ZeRO Optimizer 与 Stage 3 结构说明  
  https://deepwiki.com/deepspeedai/DeepSpeed/3.1-zero-optimizer  
  https://deepwiki.com/deepspeedai/DeepSpeed/3.1.1-zero-stage-3
- Hugging Face Transformers 文档中的 DeepSpeed 配置经验，包含 bucket 与 `overlap_comm` 风险说明  
  https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/deepspeed
- Microsoft Research: ZeRO/ZeRO-Offload 在超大模型训练中的工程案例  
  https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/

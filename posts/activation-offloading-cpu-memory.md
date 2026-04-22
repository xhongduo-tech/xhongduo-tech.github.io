## 核心结论

Activation Offloading 是将训练中需要在反向传播复用的激活值暂存到 CPU，以减少 GPU 峰值显存。激活值是神经网络前向传播产生的中间张量，反向传播计算梯度时需要再次使用它们。

核心结论可以写成：

$$
M'_{gpu\_peak} \approx M_{gpu\_peak} - A_{offload}
$$

其中 $M_{gpu\_peak}$ 是原始 GPU 峰值显存，$A_{offload}$ 是被卸载到 CPU 的激活值大小。

它解决的不是“模型参数太大放不下”，而是“训练中间结果太多导致峰值显存爆掉”。在大模型训练里，参数、梯度、优化器状态和激活值都会占内存；Activation Offloading 只处理激活值。

一个新手版理解是：GPU 是桌面，CPU 是抽屉。训练时先把暂时不用的中间结果放进抽屉，反向传播要用时再拿回桌面。桌面不会堆满，但来回取东西会花时间。这个比喻只能帮助理解存放位置变化，真正的工程问题是 PCIe 传输带宽和计算重叠。

常见方案对比如下：

| 方案 | 主要动作 | GPU 显存收益 | 主要代价 |
|---|---|---:|---|
| 不 offload | 激活值全留在 GPU | 无 | 显存峰值高 |
| 只 checkpoint | 少保存激活，反向时重算 | 中等 | 增加计算 |
| checkpoint + offload | 少保存，再把保存项放 CPU | 高 | 增加传输与调度复杂度 |
| FSDP CPUOffload | 参数或梯度放 CPU | 不针对激活 | 通信和传输开销高 |

工程上，Activation Offloading 通常不是第一步。更常见的顺序是：先使用 activation checkpointing 降低激活保存量，再对剩余必须保存的激活做 offload。

---

## 问题定义与边界

本文讨论的是激活值 offload，不是参数 offload，也不是梯度 offload。参数是模型的权重，梯度是反向传播得到的更新方向，优化器状态是 Adam 等优化器保存的动量和方差等辅助变量。它们和激活值都占内存，但生命周期不同。

训练显存峰值可以近似写成：

$$
M_{gpu\_peak} \approx M_{params} + M_{grads} + M_{opt} + A_{keep}
$$

其中 $A_{keep}$ 是前向传播后保留在 GPU、等待反向传播使用的激活值。Activation Offloading 只针对这一项。

边界区分如下：

| 术语 | 作用对象 | 是否等于 Activation Offloading |
|---|---|---|
| Activation Offloading | 激活值 | 是 |
| CPUOffload | 参数 / 梯度 | 否 |
| Activation Checkpointing | 少保存激活值，反向时重算 | 否，但常配合 |
| ZeRO Offload | 参数 / 优化器状态 / 梯度 | 否 |

它适用于显存瓶颈明显、CPU 内存充足、PCIe 或 NVLink 传输还能接受的训练任务。典型场景包括单机多卡长序列训练、microbatch 受限的 LLM 微调、Transformer block 激活峰值很高但参数本身还能放进 GPU 的任务。

玩具例子：一个 4 层 MLP 训练时，每层前向都会产生一个中间张量。如果 batch 稍大，这些中间张量加起来可能比参数还大。此时把第 2 层和第 3 层的激活暂存到 CPU，可以降低 GPU 峰值。

真实工程例子：长上下文 LLM SFT 中，模型参数通过 FSDP 分片后可以放进多张 GPU，但序列长度从 4K 提到 16K 后，attention 和 MLP 的激活峰值迅速上升。此时单纯分片参数不能解决问题，需要对 Transformer block 做 activation checkpointing，再将部分 checkpoint 激活卸载到 CPU。

---

## 核心机制与推导

机制分为六步：

| 阶段 | 动作 |
|---|---|
| forward | 执行前向传播 |
| save minimal tensors | checkpoint 只保存必要张量 |
| offload to CPU | 将保存的激活转移到 CPU |
| prefetch before backward | 反向传播前提前取回 |
| restore to GPU | 激活回到 GPU |
| backward | 使用激活计算梯度 |

Activation Checkpointing 是少保存。它不保存某些中间激活，而是在反向传播时重新执行一段前向计算。Activation Offloading 是换位置。它保存激活，但把保存位置从 GPU 换到 CPU。两者结合后，逻辑是先减少保存量，再移动保存项。

传输成本是关键。设某段激活大小为 $A$，有效传输带宽为 $B$，单向传输时间为：

$$
T_{tx} \approx \frac{A}{B}
$$

真正的额外训练时间不是 $T_{tx}$ 本身，而是没有被计算隐藏的部分：

$$
T_{overhead} \approx \max(0, T_{tx} - T_{compute\_slack})
$$

$T_{compute\_slack}$ 是其他计算提供的可重叠时间，例如后续层的前向计算、其他 microbatch 的计算、或反向传播中尚未依赖该激活的计算窗口。

判断表如下：

| 条件 | 结果 |
|---|---|
| $T_{compute\_slack} \ge T_{tx}$ | 传输大部分可隐藏 |
| $T_{compute\_slack} < T_{tx}$ | 训练步会变慢 |
| checkpoint 过少 | offload 的对象太多，传输压力大 |
| checkpoint 过多 | 重算开销过高 |

数值例子：某段 checkpoint 后仍需保存的激活是 512 MB，PCIe 有效带宽按 32 GB/s 估算。单向传输时间约为：

$$
T_{tx} \approx \frac{0.5}{32} = 0.0156s = 15.6ms
$$

如果前向卸载和反向取回各一次，总传输约 31.2 ms。若这 31.2 ms 被其他计算隐藏，训练速度影响较小；若不能隐藏，单步时间就会明显增加。

所以 Activation Offloading 的本质不是“免费省显存”，而是用 CPU 内存和数据传输换 GPU 显存。

---

## 代码实现

实际工程里，不应在训练循环中手工给每个中间张量写 `to("cpu")` 和 `to("cuda")`。这样容易遗漏张量，也难以和分布式训练、自动混合精度、随机数状态管理配合。更稳妥的方式是包装模块，让框架控制保存位置和恢复时机。

伪代码如下：

```python
# 伪代码：先 checkpoint，再 activation offload
block = offload_wrapper(
    checkpoint_wrapper(transformer_block),
    offload_to_cpu=True
)
```

通常包装顺序是先 checkpoint，后 offload。原因是 checkpoint 先决定“哪些激活需要保存”，offload 再决定“这些保存项放在哪里”。

简化流程：

```python
# 伪代码：对每个 Transformer block 统一包装
for i, layer in enumerate(model.layers):
    layer = checkpoint_wrapper(layer)
    layer = offload_wrapper(layer, offload_to_cpu=True)
    model.layers[i] = layer
```

下面是一个可运行的 Python 玩具例子，用来计算 offload 的显存收益和传输开销。它不依赖 GPU，只验证核心公式。

```python
def estimate_activation_offload(
    gpu_peak_gb: float,
    activation_offload_gb: float,
    bandwidth_gb_s: float,
    compute_slack_ms: float,
    round_trip: bool = True,
):
    new_peak = gpu_peak_gb - activation_offload_gb

    one_way_tx_ms = activation_offload_gb / bandwidth_gb_s * 1000
    tx_ms = one_way_tx_ms * (2 if round_trip else 1)

    overhead_ms = max(0.0, tx_ms - compute_slack_ms)

    return {
        "new_peak_gb": new_peak,
        "tx_ms": tx_ms,
        "overhead_ms": overhead_ms,
    }


case = estimate_activation_offload(
    gpu_peak_gb=23.0,
    activation_offload_gb=0.5,
    bandwidth_gb_s=32.0,
    compute_slack_ms=20.0,
)

assert abs(case["new_peak_gb"] - 22.5) < 1e-9
assert 31.0 < case["tx_ms"] < 32.0
assert 11.0 < case["overhead_ms"] < 12.0
```

在 DeepSpeed 中，相关配置常见于 activation checkpointing，例如 `partition_activations` 和 `checkpoint_in_cpu`。在 PyTorch/FSDP 生态中，常见组合是 `checkpoint_wrapper`、`apply_activation_checkpointing` 和 `offload_wrapper`。具体 API 会随版本变化，工程落地时要以官方文档和源码为准。

---

## 工程权衡与常见坑

Activation Offloading 的收益来自“显存换时间”。它减少 GPU 峰值显存，但增加 CPU 内存占用和 GPU-CPU 传输。它是否划算，取决于 PCIe 带宽、microbatch 数量、计算重叠窗口、CPU 内存余量和 offload 范围。

常见坑如下：

| 坑 | 结果 | 规避方式 |
|---|---|---|
| 只 offload 不 checkpoint | 保存项太多，传输压力大 | 先做 checkpoint |
| microbatch 太少 | 传输无法隐藏 | 调整 batch、pipeline 或并行策略 |
| CPU 内存不足 | 系统内存打满，甚至触发 swap | 控制 offload 范围 |
| 把 CPUOffload 当成激活 offload | 优化对象错误 | 明确区分参数、梯度、激活 |
| wrapper 顺序不对 | 不生效或收益不稳定 | 统一封装 block |
| root module 未包装 | 部分激活仍留在 GPU | 检查 wrapper 覆盖范围 |
| 随机层未处理 | 反向重算不一致 | 保证 checkpoint 的随机数状态一致 |

一个直接判断准则是：当 $A_{offload}$ 很大，且 $T_{compute\_slack}$ 足够覆盖 $T_{tx}$，offload 更划算；当 $T_{tx}$ 明显大于可隐藏计算时，offload 会拖慢训练。

真实工程里还要注意 pinned memory。Pinned memory 是不能被操作系统随意换页的 CPU 内存，GPU DMA 传输时更高效。它可以提高传输性能，但多卡同时使用时会快速消耗系统内存。长序列训练中，如果每张卡都把大量激活 pin 到 CPU，机器可能不是 GPU OOM，而是 CPU 内存先被打满。

还要区分 PCIe 和 NVLink。PCIe 4.0 x16 理论带宽约 32 GB/s，实际有效带宽还会受实现影响。NVLink 带宽更高，但不是所有 GPU-CPU 路径都能走 NVLink。不要只看硬件宣传值，要用训练日志和 profiler 观察实际传输是否成为瓶颈。

---

## 替代方案与适用边界

Activation Offloading 不是所有显存问题的首选。它更像最后一层显存优化手段，适用于需要保住模型结构、序列长度或 batch 设定，但被激活峰值卡住的训练任务。

替代方案对比如下：

| 方案 | 主要解决对象 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| Activation Checkpointing | 激活值 | 简单，显存收益稳定 | 增加重算 | 激活峰值高 |
| Activation Offloading | 激活值 | 进一步降低 GPU 峰值 | PCIe 开销 | 有足够计算重叠 |
| FSDP / ZeRO | 参数、梯度、优化器状态 | 降低长期内存占用 | 系统复杂 | 参数或状态占用大 |
| 减少 seq len / batch | 所有张量 | 简单直接 | 改变训练目标 | 能接受降规模 |
| 混合精度训练 | 多数张量 | 通常收益明显 | 数值稳定性要检查 | 模型支持低精度 |

选择顺序可以直接写成：

| 现象 | 优先方案 |
|---|---|
| 参数本身放不下 | FSDP / ZeRO |
| 优化器状态太大 | ZeRO / optimizer offload |
| 激活峰值高 | Activation Checkpointing |
| checkpoint 后仍不够 | Checkpointing + Activation Offloading |
| 计算太少，传输无法隐藏 | 调整 batch、seq len 或并行策略 |

新手容易误判的问题是：看到 OOM 就立刻 offload 激活。正确做法是先判断显存大头。如果显存主要被参数、梯度和优化器状态占用，激活 offload 的收益有限；如果显存峰值发生在长序列前向或反向阶段，且参数已经通过 FSDP 或 ZeRO 处理过，Activation Offloading 才更可能有效。

真实工程中，长上下文 LLM 微调是典型适用场景。模型参数可以通过 FSDP 分片，优化器状态可以用 ZeRO 降低，但序列长度增加会让激活按层数、batch、hidden size 和序列长度快速增长。此时保留训练目标不变的前提下，checkpoint + offload 是合理选择。

---

## 参考资料

1. [PyTorch torch.utils.checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)
2. [PyTorch checkpoint_wrapper.py 源码](https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py)
3. [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html)
4. [DeepSpeed Activation Checkpointing 文档](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)
5. [DeepSpeed 配置 JSON 文档](https://www.deepspeed.ai/docs/config-json/)
6. [AWS SageMaker Activation Offloading 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-pytorch-activation-offloading.html)

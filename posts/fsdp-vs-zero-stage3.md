## 核心结论

FSDP（Fully Sharded Data Parallel）与 ZeRO Stage 3 在目标上基本等价：都通过分片参数、梯度、优化器状态，降低单卡显存占用，让原本单卡放不下的大模型可以在多卡上训练。若只看“把训练状态切开并按 rank 分摊”这件事，FSDP 可以理解为 PyTorch 原生生态中的 ZeRO-3 路线。

两者的共同点先说清楚。传统 DDP 会让每张 GPU 都持有一整份参数、一整份梯度，以及本地优化器状态；FSDP 与 ZeRO-3 则把这三类状态都切成分片，常驻在各自 rank 上。真正计算某一层时，再临时把该层完整权重聚合出来，算完释放。反向传播时同理，梯度再被重新切散。于是，“模型状态常驻内存”从完整副本变成了分片副本。

两者的关键差异不在“是否分片”，而在“由谁来管理分片、能否进一步 offload、工程接口长什么样”。FSDP 由 PyTorch 官方维护，和 `torch.distributed`、`torch.compile`、原生模块封装方式更一致；DeepSpeed ZeRO-3 是独立训练引擎，配置面更宽，尤其在 CPU/NVMe offload 上能力更强。

对初学者来说，最重要的不是背定义，而是先做判断：

| 结论 | 含义 |
| --- | --- |
| 纯 GPU、多卡训练、希望尽量少引入额外系统 | 优先看 FSDP |
| 需要 CPU/NVMe offload 才能把模型跑起来 | 优先看 DeepSpeed ZeRO-3 |
| 不做 offload 时，两者吞吐通常接近 | 常见差异在几个百分点内，更像工程细节差而不是代际差 |
| 要和 `torch.compile` 协同 | FSDP 一般更顺手，但通常要配 `use_orig_params=True` |

“性能接近”这句话需要正确理解。很多对比里，在 4×A100、混合精度、不开 offload 的前提下，FSDP 和 ZeRO-3 的吞吐往往很接近，常见差距只有 1% 到 5%。这不表示“FSDP 永远更快”或“ZeRO-3 永远更快”，而是表示当系统瓶颈主要仍是 GPU 计算与 NCCL 通信时，两者底层通信模式足够相似，性能上限天然就不会差太远。真正拉开差距的，往往是 wrap 粒度、bucket 设置、通信重叠、网络拓扑、激活检查点策略，而不是名字本身。

再看工程量级的例子。以 13B 模型在 4×A100 上微调为例，不使用全分片时，参数、梯度、优化器状态很容易把 80GB 显存迅速吃满。实践中，FSDP 全分片混合精度常能把显存压到每卡几十 GB；ZeRO-3 无 offload 也在同一量级；若启用 CPU offload，显存还能继续大幅下降，但吞吐通常会明显下降。这个事实对应的工程结论很直接：FSDP 与 ZeRO-3 不是“新旧替代关系”，而是“纯 GPU 场景高度接近，offload 场景由 DeepSpeed 提供更强扩展”。

---

## 问题定义与边界

先把问题定义清楚。训练显存并不只由“模型参数”构成，还至少包括三大块常驻状态：

| 状态 | 白话解释 | 典型量级 |
| --- | --- | --- |
| Parameters | 模型权重本身 | $\Psi$ |
| Gradients | 反向传播为每个参数得到的梯度 | $\Psi$ |
| Optimizer States | 优化器内部缓存，例如 Adam 的一阶、二阶动量 | $k\Psi$ |

这里的 $\Psi$ 表示“按某种精度统计的全部参数大小”。例如 13B 参数、BF16 存储时，每个参数约 2 字节，那么仅参数本体就接近 `13e9 × 2 bytes ≈ 26GB`。如果梯度也按相近精度持有，那么梯度再来一份；如果优化器是 Adam，还会维护两组主要状态，通常又是非常可观的一笔内存。工程估算时，很多人会把 Adam 相关状态视为“至少与参数同量级，通常明显更大”。

如果使用普通 DDP，每张卡都会保存一整份这些内容。于是单卡显存占用可粗略写成：

$$
M_{\text{DDP}} \approx \Psi + \Psi + k\Psi = (2+k)\Psi
$$

若考虑混合精度训练里常见的“低精度参数参与计算，高精度主权重用于优化器更新”，估算还会更高。也就是说，DDP 的问题不是通信太多，而是“每张卡都在为同一套模型状态重复买单”。

ZeRO-3 与 FSDP 解决的是这个问题。它们的理想化目标是：把参数、梯度、优化器状态都切成 $N_d$ 份，让每张设备只长期保存其中一份。于是单卡常驻状态近似变成：

$$
M_{\text{shard}} \approx \frac{\Psi + \Psi + k\Psi}{N_d}
$$

如果把混合精度下的额外主权重也纳入简单工程估算，常见写法会扩展为：

$$
M_{\text{mixed}} \approx \frac{2\Psi + 2\Psi + k\Psi}{N_d}
$$

这个公式不是严格定律，而是够用的近似。它表达的核心只有一句话：在理想分片下，参数相关常驻状态大体会随着设备数近似线性下降。

为什么说 13B 模型在 4×A100 上很容易需要这类方案？做一个足够粗但足够有用的估算。13B 参数若按 BF16 存储，参数本体约 26GB；梯度再加一份，约 26GB；Adam 的一阶、二阶状态若按 FP32 维护，单看这两块就又可能来到百 GB 量级的一大部分。即便忽略激活值、临时通信 buffer、CUDA allocator 碎片等额外开销，单卡完整保存这些状态也已经逼近或超过 80GB。此时常见选择只有三类：

| 方案 | 本质 |
| --- | --- |
| 降模型规模 | 少放参数，直接回避问题 |
| 上模型并行/流水并行 | 把计算图拆开，不让一张卡承担全部计算 |
| 上 ZeRO-3 / FSDP | 不拆模型结构，先拆训练状态 |

本文只讨论第三类问题：在单机多卡或多机多卡训练里，参数、梯度、优化器状态如何分布与回收。这里不展开张量并行、流水并行，也不讨论推理时 KV Cache 的显存问题。

比较边界也必须先限定。只有在“不启用 CPU/NVMe offload”时，FSDP 与 ZeRO-3 的吞吐对比才相对公平。因为一旦开启 offload，瓶颈就不再主要是 GPU 算力和 NCCL 通信，而会转向 PCIe、主存带宽、页缓存行为甚至 NVMe 读写能力。此时 DeepSpeed 的优势会体现在“能跑更大模型”，但不能再简单和纯 GPU 方案按吞吐直接横比。

另一个边界是对象生命周期。FSDP/ZeRO-3 都会改变参数的组织与可见方式，所以训练代码的构造顺序不是细节，而是前置条件。最容易出错的例子就是先创建 optimizer，再做 FSDP 封装：

```python
# 错误：optimizer 先拿到了未分片参数
optimizer = Adam(model.parameters(), lr=1e-4)
model = FSDP(model)

# 正确：先 wrap，再创建 optimizer
model = FSDP(model)
optimizer = Adam(model.parameters(), lr=1e-4)
```

初学者可以把它理解为：优化器必须绑定“最终参与训练的那套参数视图”，否则它管理的对象和真正更新的对象就不是同一批张量。

---

## 核心机制与推导

先从一个最小玩具例子理解“全分片”到底发生了什么。假设模型只有 8 个参数，训练用 4 张 GPU：

| GPU | 本地长期保存的参数分片 |
| --- | --- |
| GPU0 | `[p0, p1]` |
| GPU1 | `[p2, p3]` |
| GPU2 | `[p4, p5]` |
| GPU3 | `[p6, p7]` |

这时候，每张卡都只常驻自己负责的那一小段参数，而不是完整 `[p0, ..., p7]`。问题来了：如果某一层前向计算需要完整参数，怎么做？

答案就是“按需聚合，算完释放”：

1. 进入该层前向前，执行 all-gather，把分散在 4 张卡上的参数碎片临时拼成完整权重。
2. 该层前向完成后，完整权重不再长期保留，只保留本地分片。
3. 进入该层反向时，若需要完整权重，再次 all-gather。
4. 梯度算完后，通过 reduce-scatter 把完整梯度重新切回各卡。

把这个过程写成生命周期表，会更容易和代码对应：

| 阶段 | 每卡长期保留什么 | 每卡临时获得什么 | 主要通信 |
| --- | --- | --- | --- |
| 初始化后 | 参数分片 | 无 | 无 |
| 某层前向前 | 参数分片 | 该层完整参数 | all-gather |
| 某层前向后 | 参数分片 | 释放完整参数 | 无 |
| 某层反向前/中 | 参数分片 | 该层完整参数 | all-gather |
| 梯度回收 | 梯度分片 | 完整梯度短暂存在或逻辑上完成归并 | reduce-scatter |
| `optimizer.step()` | 本地参数分片与本地优化器状态 | 无 | 通常无大规模参数同步 |

这就是 ZeRO-3 与 FSDP 的共同工作方式。它们真正长期存着的是“碎片”，而不是“完整版”。

接下来再看为什么这样能省显存。设全模型参数量为 $\Psi$，设备数为 $N_d$。在 DDP 里，每卡都保留一整份参数、梯度、优化器状态，所以每卡显存和模型规模是线性绑定的；而在全分片里，每卡常驻量大约只和 $\Psi / N_d$ 绑定。因此，当卡数翻倍时，参数相关常驻内存会近似减半。这也是为什么 ZeRO-3/FSDP 的收益在大模型上最明显，因为参数相关状态本来就是最重的那部分。

但这里有一个新成本：通信。前向前要 all-gather，反向时通常还要再来一次 all-gather，梯度完成后还要 reduce-scatter。若某层参数量为 $\Psi_l$，那么粗略地看，该层单步额外通信量大约是若干个 $\Psi_l$ 级别。把各层累加后，全模型每步会有数量可观的通信流量。

于是问题从“能不能省显存”变成“省下来的显存是否值得额外通信”。答案通常取决于通信与计算能否重叠。如果通信完全串行地插在计算之间，那么训练会明显变慢；如果能利用 prefetch 和 bucket，把一部分通信隐藏在另一层计算期间，那么墙钟时间更接近：

$$
T_{\text{step}} \approx \max(T_{\text{compute}}, T_{\text{comm}})
$$

而不是：

$$
T_{\text{step}} \approx T_{\text{compute}} + T_{\text{comm}}
$$

这就是 backward prefetch 的意义。它不是减少通信总量，而是尽量把“下一段需要传的数据”提前发起，让 GPU 算的时候，网络也在忙。

到这里就能解释一个常见现象：为什么很多实验里 FSDP 和 ZeRO-3 的性能很接近？因为两者面对的是同一个数学约束。

| 影响项 | 为什么会影响结果 |
| --- | --- |
| 参数组织方式 | 张量是大块连续还是很多碎片，会影响带宽利用率 |
| bucket 划分 | 决定一次通信打包多少数据，影响启动开销 |
| prefetch 策略 | 决定通信与计算能否重叠 |
| mixed precision | 更低精度会减少存储与通信压力 |
| wrap 粒度 | 太细通信次数多，太粗又会拉高峰值显存 |
| 网络拓扑 | NVLink、PCIe、IB 的带宽和时延差异会被实现细节放大 |

两者的核心差异也可以放在这里理解。FSDP 常见实现方式是把一个模块内部的多个参数拼成连续的 `FlatParameter`，再以此为单位进行分片和通信。对新手来说，可以把 `FlatParameter` 理解为“为了方便通信和管理，把一堆零散小张量先拼成一个大张量”。这样做的直接收益有两个：

| 目的 | 原因 |
| --- | --- |
| 降低通信碎片化 | 一次传大块数据，通常比传很多小块更有效率 |
| 降低状态管理成本 | bucket、元数据、分片映射都更简单 |

ZeRO-3 也会做类似的 bucket 化与状态管理，但它暴露给用户的是另一套训练引擎接口。换句话说，二者“底层物理问题”相同，“工程组织形式”不同。

再把结论落回真实训练。13B 模型在 4×A100 上使用全分片时，峰值显存通常不再是“完整模型状态 + 激活”，而更接近“本地分片 + 当前正在计算层的临时完整参数 + 激活 + 通信 buffer”。这就是为什么很多本来“完全放不下”的模型，在合理 wrap、混合精度、激活检查点搭配下，能够被塞进单机多卡里训练起来。

---

## 代码实现

先给一个可以直接运行的 Python 玩具脚本。它不依赖 GPU，不依赖 PyTorch 分布式，只模拟三件事：

1. 参数如何被分片。
2. 某层计算前如何 all-gather 拿回完整参数。
3. 梯度如何 reduce-scatter 回各自 shard。

```python
from math import isclose


def shard_params(params, world_size):
    if len(params) % world_size != 0:
        raise ValueError("len(params) must be divisible by world_size")
    shard_size = len(params) // world_size
    return [params[i * shard_size:(i + 1) * shard_size] for i in range(world_size)]


def all_gather(shards):
    full = []
    for shard in shards:
        full.extend(shard)
    return full


def reduce_scatter(full_grads, world_size):
    if len(full_grads) % world_size != 0:
        raise ValueError("len(full_grads) must be divisible by world_size")
    shard_size = len(full_grads) // world_size
    return [full_grads[i * shard_size:(i + 1) * shard_size] for i in range(world_size)]


def sgd_step(param_shards, grad_shards, lr):
    updated = []
    for p_shard, g_shard in zip(param_shards, grad_shards):
        updated.append([p - lr * g for p, g in zip(p_shard, g_shard)])
    return updated


def main():
    # 假设完整参数在逻辑上有 8 个元素，2 张“卡”各负责一半
    params = [1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0, 5.0]
    world_size = 2

    param_shards = shard_params(params, world_size)
    assert param_shards == [[1.0, 2.0, 3.0, 4.0], [8.0, 7.0, 6.0, 5.0]]

    # 计算前临时聚合完整参数
    full_params = all_gather(param_shards)
    assert full_params == params

    # 用一个最简单的“损失”：loss = sum(p^2)
    # 那么每个参数的梯度就是 dloss/dp = 2p
    full_grads = [2.0 * p for p in full_params]

    # 梯度重新切散回各 rank
    grad_shards = reduce_scatter(full_grads, world_size)
    assert grad_shards == [[2.0, 4.0, 6.0, 8.0], [16.0, 14.0, 12.0, 10.0]]

    # 每个 rank 只更新自己本地 shard
    lr = 0.1
    updated_shards = sgd_step(param_shards, grad_shards, lr)

    # 为了检查整体结果，再逻辑上拼回完整参数
    updated_full = all_gather(updated_shards)
    expected = [0.8, 1.6, 2.4, 3.2, 6.4, 5.6, 4.8, 4.0]

    assert all(isclose(a, b) for a, b in zip(updated_full, expected))
    print("toy fsdp/zero3 flow ok")


if __name__ == "__main__":
    main()
```

这个玩具脚本虽然非常简化，但它已经把全分片的核心状态流转表达出来了：

| 步骤 | 在真实系统里对应什么 |
| --- | --- |
| `shard_params` | 初始化后，各 rank 只保留自己负责的参数分片 |
| `all_gather` | 某层计算前，临时恢复该层完整参数 |
| `reduce_scatter` | 梯度完成后，把完整梯度重新切回本地分片 |
| `sgd_step` | 每个 rank 只更新自己管理的参数与优化器状态 |

真正的 PyTorch FSDP 代码要复杂得多，因为它要处理 CUDA、进程组、mixed precision、参数视图、自动求导图以及通信时机。下面给一个最小可运行示意，适合理解正确顺序。这个脚本要求你已经用 `torchrun` 启动多进程，并且机器上有可用 GPU。

```python
import os
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy


def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 1024),
    )


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup()
    device = torch.device("cuda", local_rank)

    model = build_model().to(device)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        device_id=device,
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # 若需要 torch.compile，通常在 wrap 与 optimizer 之后再处理
    model = torch.compile(model)

    for _ in range(3):
        x = torch.randn(8, 1024, device=device, dtype=torch.bfloat16)
        out = model(x)
        loss = out.float().pow(2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:
            print(f"loss={loss.item():.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

典型启动方式如下：

```bash
torchrun --nproc_per_node=4 train_fsdp.py
```

这段代码里，初学者最该盯住的是下面三项：

| 配置 | 作用 |
| --- | --- |
| `ShardingStrategy.FULL_SHARD` | 参数、梯度、优化器状态都分片，对应 ZeRO-3 语义 |
| `use_orig_params=True` | 让参数以更接近原始模块的方式暴露，通常更利于 `torch.compile` 与某些上层工具 |
| `BackwardPrefetch.BACKWARD_PRE` | 尝试在反向阶段提前拉取下一层所需参数，以增加通信与计算重叠 |

再看 DeepSpeed ZeRO-3。它的思路不是“直接给模型套一个 wrapper 就结束”，而是通过训练引擎和配置文件接管更多训练行为。一个最小 JSON 配置大致是这样：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "none"
    },
    "offload_optimizer": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

启动方式通常类似：

```bash
deepspeed --num_gpus=4 train_ds.py --deepspeed ds_config.json
```

如果把两者差异压缩成一句话，方便新手记忆，可以这样理解：

| 系统 | 更像什么 |
| --- | --- |
| FSDP | 在 PyTorch 原生分布式里，把模型包成“自动分片模块” |
| DeepSpeed ZeRO-3 | 把训练过程交给一个更完整的外部训练引擎 |

---

## 工程权衡与常见坑

实战里最常见的问题不是“概念听不懂”，而是“代码看起来没错，结果训练不稳定、速度奇怪、显存比预期高”。这些问题大多来自对象生命周期和粒度选择。

先看最常见的坑：

| 坑 | 现象 | 原因 | 规避措施 |
| --- | --- | --- | --- |
| 先建 optimizer，再做 FSDP wrap | 参数更新异常、状态错位、甚至直接报错 | optimizer 绑定的是旧参数引用 | 必须先 wrap，再建 optimizer |
| `torch.compile` 未配 `use_orig_params=True` | 编译失败、图捕获异常、参数视图问题 | 编译器看到的参数结构不符合预期 | 需要 compile 时通常打开 `use_orig_params=True` |
| wrap 粒度过细 | 通信次数暴增，吞吐下降 | all-gather / reduce-scatter 太碎 | 常以 transformer block 为基本粒度 |
| wrap 粒度过粗 | 峰值显存升高，重叠空间变差 | 一次聚合的权重块太大 | 不要把整个超大模型只包一层 |
| 开 CPU offload 还期待纯 GPU 吞吐 | 训练明显变慢 | 瓶颈转移到 PCIe / CPU 内存 / NVMe | 把 offload 当作“换显存”，不是“提速度” |
| 激活检查点与 mixed precision 配置混乱 | 速度不升反降，甚至 OOM | 重算与通信叠加不当 | 组合测试，不要凭直觉堆开关 |

优化器时机是最值得单独强调的一个点。错误顺序如下：

```python
model = build_model().cuda()
optimizer = AdamW(model.parameters(), lr=1e-4)  # 错误
model = FSDP(model, use_orig_params=True)
```

正确顺序如下：

```python
model = build_model().cuda()
model = FSDP(model, use_orig_params=True)
optimizer = AdamW(model.parameters(), lr=1e-4)
model = torch.compile(model)
```

为什么这件事这么敏感？因为 FSDP 在封装后，参数组织方式可能已经变化了。你可以把它理解成“模型的骨架被重新排布过一次”。如果 optimizer 在这之前就拿到了旧引用，它维护的动量、权重衰减状态和真正训练时的参数对象就可能不再一一对应。

第二个高频误判是“显存更低就代表整体更优”。这在 offload 场景尤其容易出错。举一个典型判断表：

| 指标 | 不开 offload | 开 CPU offload |
| --- | --- | --- |
| 可训练模型上限 | 较低 | 更高 |
| 单步吞吐 | 更快 | 更慢 |
| 系统瓶颈 | GPU 计算 + GPU 间通信 | 还叠加 CPU 内存、PCIe、磁盘链路 |
| 调参与排障复杂度 | 中等 | 更高 |

因此，团队做方案选择时，不能只问“能不能跑起来”，还要问“每天能推进多少 token、多少 step”。如果一个方案把显存从 44GB/GPU 压到 21GB/GPU，但吞吐接近腰斩，那么它更像是“容量换速度”的交易，而不是无代价优化。

第三个新手常踩的坑是把 FSDP/ZeRO-3 当成万能显存开关。它们主要解决的是模型状态，也就是参数、梯度、优化器状态怎么存；但训练显存里还有激活值。序列很长、batch 很大、注意力层很深时，激活值同样可能成为主瓶颈。此时只上全分片还不够，往往还要配合：

| 技术 | 解决哪类内存问题 |
| --- | --- |
| Activation Checkpointing | 用重算换激活显存 |
| Sequence / Context 并行 | 把长序列相关开销摊开 |
| Tensor Parallel | 把单层算子切分到多卡 |
| Pipeline Parallel | 把模型层按阶段切开 |
| Offload | 把一部分状态挪到 CPU / NVMe |

还有一个工程现实也要说明：当你同时追求 `torch.compile`、复杂分片、offload、多机多卡稳定性时，验证成本会上升。FSDP 因为在 PyTorch 主生态中，很多原生工具链配合得更直接；DeepSpeed 则因为功能面更宽，能做更多事，但意味着配置、排障、版本兼容的复杂度也更高。这个结论不是“谁更好”，而是“你要更多控制，就要接受更多系统复杂性”。

---

## 替代方案与适用边界

真正的工程选择不是“FSDP 对 DeepSpeed 二选一”，而是“在当前约束下，哪种方案用最小复杂度解决问题”。

先给一个直观判断表：

| 场景 | 推荐方案 | 原因 |
| --- | --- | --- |
| 纯 GPU、多卡训练、尽量少依赖额外系统 | FSDP | PyTorch 原生集成更自然 |
| 需要和 `torch.compile` 深度协同 | FSDP 优先 | 参数视图与编译路径通常更顺 |
| GPU 显存仍明显不够，需要 CPU/NVMe 换空间 | DeepSpeed ZeRO-3 | offload 功能更成熟 |
| 需要更细粒度训练引擎控制 | DeepSpeed | 配置面更宽，训练系统能力更强 |
| 团队已经大量使用 Hugging Face + PyTorch 原生分布式 | FSDP | 迁移与维护成本更低 |
| 单机仍放不下，还要叠加张量并行/流水并行 | 看整体栈，不只看名字 | 关键是并行方案的组合方式 |

如果只给新手一套最短决策规则，可以直接用下面三条：

1. 模型在纯 GPU 上通过全分片已经能跑起来，先用 FSDP。
2. 模型即便全分片仍放不下，或者 micro-batch 被压得太小，效率很差，再考虑 DeepSpeed 的 CPU/NVMe offload。
3. 如果从一开始就明确必须依赖 offload，直接围绕 DeepSpeed 设计训练栈，不要先把 FSDP 堆到极限再返工。

还要强调一个边界：FSDP 与 ZeRO-3 都不解决“所有训练瓶颈”。它们主要解决的是“模型状态如何存储和同步”，而不是“所有显存都自动消失”。当瓶颈转到激活值、注意力中间张量、超长序列、数据加载、磁盘吞吐时，单靠全分片并不够。

把它们放回完整系统设计里，更容易看清职责：

| 组件 | 主要解决什么 |
| --- | --- |
| FSDP / ZeRO-3 | 参数、梯度、优化器状态的分片 |
| Activation Checkpointing | 激活值重算换显存 |
| Tensor Parallel | 单层内部算子切分 |
| Pipeline Parallel | 层级切分与跨阶段流水 |
| Offload | 用 CPU/NVMe 换 GPU 显存 |

因此，若你的路线是“纯 GPU、多卡、想用 PyTorch 原生分布式和编译生态”，FSDP 往往是更低摩擦的选择；若你的路线是“必须把更大模型硬塞进现有显存，并接受更复杂配置和更低吞吐”，DeepSpeed ZeRO-3 往往更合适。两者不是互斥技术，而是针对不同约束条件的优先答案。

---

## 参考资料

1. PyTorch 官方 FSDP 文档  
   重要性：定义了 `FULL_SHARD`、`MixedPrecision`、`BackwardPrefetch`、`use_orig_params`、状态字典保存方式等核心语义，是理解 FSDP 的第一手资料。  
   来源：https://docs.pytorch.org/docs/stable/fsdp.html

2. PyTorch Distributed 教程与 FSDP 相关说明  
   重要性：补充了自动 wrap、参数视图、checkpoint 保存、与优化器构造顺序相关的工程细节。  
   来源：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

3. DeepSpeed ZeRO 官方文档  
   重要性：给出 ZeRO Stage 1/2/3 的分层定义，以及参数分片、优化器分片、梯度分片和 offload 能力的官方说明。  
   来源：https://www.deepspeed.ai/tutorials/zero/

4. DeepSpeed ZeRO-Offload 文档  
   重要性：offload 是 DeepSpeed 相比纯 FSDP 路线最重要的扩展能力之一，直接决定“显存不够时还能不能继续往上堆模型”。  
   来源：https://www.deepspeed.ai/tutorials/zero-offload/

5. Hugging Face Accelerate 关于 FSDP 与 DeepSpeed 的实践资料  
   重要性：适合理解两者在真实训练脚本中的接入差异，以及为何在无 offload 场景下二者性能常常接近。  
   来源：https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed

6. Hugging Face Ultra-Scale / Nanotron 相关材料  
   重要性：提供大模型训练时内存与通信量的近似推导，帮助理解为何全分片能把参数相关常驻内存按设备数近似摊薄。  
   来源：https://nanotron-ultrascale-playbook.static.hf.space/

7. PyTorch FSDP 论文与设计说明  
   重要性：有助于从实现角度理解 `FlatParameter`、按模块聚合、通信重叠与状态管理方式。  
   来源：PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

8. DeepSpeed ZeRO 论文  
   重要性：定义了 ZeRO 系列方法的基本思想，尤其适合理解 Stage 1、2、3 的演进关系。  
   来源：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

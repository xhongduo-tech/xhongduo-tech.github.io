## 核心结论

ZeRO-R 解决的不是参数、梯度、优化器状态这类“模型状态”内存，而是 **residual memory**。residual memory 可以直译为“剩余内存项”，指 ZeRO-DP 已经完成模型状态分片之后，训练过程中仍然必须支付的三类开销：激活、临时 buffer、内存碎片。

它的核心思路可以压缩成一张表：

| 残留内存 | ZeRO-R 策略 | 直接效果 |
| --- | --- | --- |
| 激活 | 激活分片，反向时按需 AllGather，可选 CPU offload | 每张卡只保留自己负责的那一片，不再长期保留完整副本 |
| 临时 buffer | 固定大小常量 buffer，按 chunk 循环复用 | 避免 fused buffer 随模型规模线性膨胀 |
| 碎片 | 预分配连续内存区，必要时做整理 | 减少 allocator 失败和“账面有空闲却 OOM” |

如果把总激活记为 $A$，激活分片组大小记为 $N$，固定 buffer 预算记为 $C_B$，碎片管理预留区记为 $F$，那么单卡近似内存可以写成：

$$
M_{\text{per-card}} \approx \frac{A}{N} + C_B + F
$$

如果进一步叠加激活重计算，设重计算后需要长期保留的激活比例为 $\alpha \in (0,1]$，那么更常用的估算式是：

$$
M_{\text{per-card}} \approx \frac{\alpha A}{N} + C_B + F
$$

这个公式表达的是同一件事：ZeRO-R 不是让激活消失，而是把原本“整块复制”的激活改成“按卡分摊”；把原本容易失控的临时 buffer 改成定额预算；把不可预期的碎片风险改成工程上可管理的成本。

一个常见例子是 GPT-2 1.5B。假设序列长度为 1024、global batch size 为 32，训练过程中的激活开销可达到约 60GB。若激活分片组大小为 8，那么单卡只需要长期保留大约：

$$
60 / 8 = 7.5 \text{GB}
$$

若再叠加 checkpointing，假设 $\alpha = 0.5$，则单卡长期保留的激活进一步变成：

$$
0.5 \times 60 / 8 = 3.75 \text{GB}
$$

这时 32GB V100 不再先被激活压垮，剩余空间还能留给参数分片、梯度通信、工作区和运行时开销。ZeRO-R 的价值就在这里：它补齐了 ZeRO-DP 只处理“模型状态冗余”，却没有处理“训练过程残余内存”的缺口。

---

## 问题定义与边界

先把问题边界讲清楚。很多初学者一提显存优化，只想到“把参数切开”。这只解决了一部分问题，而且往往不是训练中最难的一部分。

训练显存通常可以分成两大类：

| 类别 | 包含内容 | 生命周期特征 | ZeRO-DP 是否直接解决 |
| --- | --- | --- | --- |
| 模型状态 | 参数、梯度、优化器状态 | 贯穿整个训练 step，生命周期长 | 是 |
| residual memory | 激活、临时 buffer、碎片 | 随前向/反向动态变化，生命周期不规则 | 否 |

这里三个术语需要单独说清楚。

| 术语 | 定义 | 为什么会占显存 |
| --- | --- | --- |
| 激活（activation） | 前向传播中间层输出，反向时需要用它计算梯度 | 只要某层的反向还没执行，这层前向结果就不能随便丢 |
| 临时 buffer | 算子运行时申请的工作区，例如规约、拼接、归一化、融合 kernel 的缓存 | 很多高性能实现要先申请一块连续内存再做计算 |
| 内存碎片 | 空闲显存总量足够，但被切成很多不连续的小块 | 大块连续申请失败时，程序仍会 OOM |

因此 ZeRO-R 的边界很明确：

1. 它不是 ZeRO-DP 的替代，而是补充。
2. 它主要压缩 residual memory，而不是直接减少参数本身。
3. 它最适合“模型状态已经分片，但激活或运行时内存仍然 OOM”的场景。

一个玩具例子更容易看清边界。

假设你有 4 张 GPU，每张 16GB。参数、梯度、优化器状态经 ZeRO 分片后，每张卡只占 6GB，看起来还剩 10GB。问题是某些层在训练时需要 12GB 激活，结果仍然 OOM。这里失败的原因不是模型状态，而是 residual memory。

如果不用 ZeRO-R，总账是：

$$
6 + 12 = 18 \text{GB} > 16 \text{GB}
$$

如果把这 12GB 激活按 4 卡分片，则每张卡长期保存的激活变成：

$$
12 / 4 = 3 \text{GB}
$$

这时单卡近似变成：

$$
6 + 3 + C_B + F
$$

只要 $C_B + F < 7$，系统就能跑起来。也就是说，ZeRO-DP 解决的是“固定库存不要每层楼都摆一份”，ZeRO-R 解决的是“施工过程中的半成品、临时工具和过道占用也必须纳入预算”。

为了避免把问题混在一起，下面给一个初学者常犯的判断错误：

| 现象 | 常见误判 | 正确判断 |
| --- | --- | --- |
| 参数分片后仍 OOM | 说明参数还不够小 | 可能是激活或临时 buffer 才是峰值来源 |
| batch size 降低后能跑 | 说明参数问题解决了 | 更可能是激活随着 batch 下降而减小 |
| 监控显示还有空闲显存 | 说明不该 OOM | 可能是碎片导致无法分配连续内存 |

所以，ZeRO-R 讨论的是训练过程中“剩余但关键”的内存问题，而不是所有显存问题的总称。

---

## 核心机制与推导

ZeRO-R 通常拆成三个机制：Partitioned Activation Checkpointing、Constant-size Buffers、Memory Defragmentation。三者分别对应激活、临时 buffer、碎片这三类 residual memory。

### 1. 激活分片

先从最重要的激活说起。

**Activation checkpointing** 的基本思想是：不是所有中间结果都要一直保存，可以只保留少量检查点，其余在反向时重算。ZeRO-R 在这个基础上再走一步，不仅减少“保留多少层”，还减少“每层保留多少数据”。

设总激活大小为 $A_{\text{total}}$，分片组大小为 $N$，则每张卡长期保留的激活近似为：

$$
A_{\text{per-rank}} = \frac{A_{\text{total}}}{N}
$$

如果再叠加 checkpointing，设保留比例为 $\alpha$，则：

$$
A_{\text{per-rank}} = \frac{\alpha A_{\text{total}}}{N}
$$

反向传播经过某层时，再通过 AllGather 把各卡上的分片拼回该层所需的完整激活。  
AllGather 的含义可以直接理解为：每张卡拿出自己那一份，通信结束后，每张卡都拿到完整拼图。

为什么这个机制成立？因为反向传播是逐层回溯的，并不需要在同一时刻保留所有层的完整激活。对某一层来说，只要它开始反向之前能恢复出本层所需的完整激活即可。于是训练过程变成：

1. 前向时保存分片后的 checkpoint。
2. 反向走到某层前，按需恢复完整激活。
3. 完成本层反向后，释放恢复出来的完整副本。

这个生命周期很关键，因为 ZeRO-R 节省的不是“总激活量”，而是“同一时刻必须常驻的激活量”。

下面用一个 4 层网络举例：

| 策略 | 长期保存的内容 | 单层激活保存成本 |
| --- | --- | --- |
| 无优化 | 所有层完整激活 | $A$ |
| 仅 checkpointing | 少数检查点层完整激活 | $\alpha A$ |
| 仅 ZeRO-R 分片 | 所有保留层的分片激活 | $A/N$ |
| 二者叠加 | 少数检查点层的分片激活 | $\alpha A / N$ |

因此，checkpointing 和激活分片是互补关系，而不是替代关系。

### 2. 与重计算正交

“正交”这个词容易把新手劝退，实际含义很简单：两种方法解决的是不同维度的问题，可以叠加使用。

| 方法 | 解决的问题 | 代价 |
| --- | --- | --- |
| 激活重计算 | 减少需要长期保存的层数 | 增加额外前向计算 |
| 激活分片 | 减少每层长期保存的体积 | 增加通信与调度复杂度 |

用公式写得更清楚一些。设原始激活为 $A$：

1. 只做 checkpointing：变成 $\alpha A$
2. 只做分片：变成 $A/N$
3. 两者叠加：变成 $\alpha A / N$

因此：

$$
A \rightarrow \alpha A \rightarrow \frac{\alpha A}{N}
$$

这就是为什么在长序列 Transformer 训练里，ZeRO-R 很少单独出现，而是经常和 activation checkpointing 一起使用。前者把“宽度”切开，后者把“层数”压缩，两者叠加才可能把显存打到可运行区间。

### 3. 常量 buffer

第二类问题是临时 buffer。

很多训练系统为了提升吞吐，会做梯度融合、通信融合、算子融合。这些优化往往需要一块额外的大 buffer。问题在于，如果这个 buffer 的大小跟模型规模走，模型越大，它就越可能从“工具”变成“内存大户”。

ZeRO-R 的策略是：不用“整模型大小”的融合区，而使用固定大小 chunk 的工作区，循环复用。

假设某次规约要处理 8GB 数据，有两种方式：

| 方式 | buffer 大小 | 峰值显存特征 |
| --- | --- | --- |
| 一次性融合 | 接近 8GB | 随待处理数据线性膨胀 |
| 固定 chunk 复用 | 例如固定 256MB 或 512MB | 峰值被 buffer 上限锁死 |

如果固定 chunk 为 $C_B$，总数据量为 $T$，则处理轮数大约为：

$$
\left\lceil \frac{T}{C_B} \right\rceil
$$

这样做的收益不是减少总搬运量，而是把峰值内存从“不可预测”变成“可预测”。对大模型训练来说，这一点比单次算子最优吞吐更重要，因为 OOM 是硬失败。

初学者容易误解为“buffer 越大越高效”。这只在单个算子的局部视角下可能成立。从整机训练稳定性看，更重要的是给 buffer 一个硬上限，否则模型变大后，buffer 自己就会抢占原本要留给激活和状态分片的空间。

### 4. 碎片管理

第三类问题是内存碎片。

碎片的本质不是“总量不够”，而是“空闲不连续”。GPU allocator 长时间处理不同大小、不同生命周期的张量后，容易出现下面这种情况：

| 指标 | 现象 |
| --- | --- |
| 总空闲显存 | 还有几 GB |
| 最大连续空闲块 | 可能只有几百 MB |
| 新申请需求 | 某个 fused kernel 需要 1GB 连续空间 |
| 结果 | 分配失败，直接 OOM |

ZeRO-R 的思路是预分配连续区域给长生命周期对象，并在必要时做整理，把仍然存活的重要张量搬入连续区。这样做有一次额外拷贝的成本，但可以显著降低 allocator 因碎片导致的失败概率。

一个简单的判断式是：

$$
\text{OOM risk} \not\propto \text{free memory only}
$$

更准确的说法是，OOM 风险同时受到两件事影响：

$$
\text{OOM risk} = f(\text{free bytes}, \text{largest contiguous block})
$$

监控如果只告诉你“还有多少空闲显存”，但不告诉你“最大连续可分配块多大”，你就无法判断自己是不是已经处于碎片高风险区。

### 5. 合并后的内存估算

把三类开销放到一起，可以得到一个更完整的近似式：

$$
M_{\text{per-card}} \approx M_{\text{model-state}} + \frac{\alpha A}{N} + C_B + F + \epsilon
$$

其中：

- $M_{\text{model-state}}$ 是参数、梯度、优化器状态经 ZeRO 分片后的单卡开销
- $\frac{\alpha A}{N}$ 是保留激活的单卡份额
- $C_B$ 是常量 buffer 上限
- $F$ 是为碎片管理保留的预算
- $\epsilon$ 是框架运行时、CUDA context、内核工作区等杂项开销

这个式子非常适合做工程上的第一轮预算，因为它能把问题拆成可以单独观测和调整的几项。

---

## 代码实现

下面先给一个最小可运行的 Python 示例。它不依赖 GPU，只模拟“激活分片、按需恢复、固定 buffer 预算、简单碎片估算”的核心流程。直接用 `python3 demo.py` 就能运行。

```python
from math import ceil
from dataclasses import dataclass
from typing import List


def shard_activation(values: List[int], world_size: int) -> List[List[int]]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    chunk = ceil(len(values) / world_size)
    shards = []
    for i in range(world_size):
        start = i * chunk
        end = min(start + chunk, len(values))
        shards.append(values[start:end])
    return shards


def allgather_shards(shards: List[List[int]]) -> List[int]:
    merged = []
    for shard in shards:
        merged.extend(shard)
    return merged


def estimate_zero_r_memory(
    total_activation_gb: float,
    world_size: int,
    constant_buffer_gb: float,
    frag_gb: float,
    checkpoint_ratio: float = 1.0,
) -> float:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 < checkpoint_ratio <= 1:
        raise ValueError("checkpoint_ratio must be in (0, 1]")

    activation_per_rank = total_activation_gb * checkpoint_ratio / world_size
    return activation_per_rank + constant_buffer_gb + frag_gb


@dataclass
class SimpleAllocator:
    total_gb: float
    largest_contiguous_free_gb: float

    def can_allocate(self, need_gb: float) -> bool:
        return need_gb <= self.largest_contiguous_free_gb

    def reserve_contiguous(self, reserve_gb: float) -> None:
        if reserve_gb > self.largest_contiguous_free_gb:
            raise MemoryError("cannot reserve a contiguous block of that size")
        self.largest_contiguous_free_gb -= reserve_gb


def iter_chunks(values: List[int], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def process_with_constant_buffer(values: List[int], chunk_size: int) -> int:
    peak_buffer_items = 0
    total = 0
    for chunk in iter_chunks(values, chunk_size):
        peak_buffer_items = max(peak_buffer_items, len(chunk))
        total += sum(chunk)
    return peak_buffer_items, total


def main():
    # 1. 激活分片与恢复
    act = list(range(10))
    shards = shard_activation(act, world_size=4)

    assert shards == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9],
    ]

    restored = allgather_shards(shards)
    assert restored == act

    # 2. 单卡内存估算
    per_card = estimate_zero_r_memory(
        total_activation_gb=60.0,
        world_size=8,
        constant_buffer_gb=2.0,
        frag_gb=1.0,
        checkpoint_ratio=1.0,
    )
    assert abs(per_card - 10.5) < 1e-9

    per_card_with_ckpt = estimate_zero_r_memory(
        total_activation_gb=60.0,
        world_size=8,
        constant_buffer_gb=2.0,
        frag_gb=1.0,
        checkpoint_ratio=0.5,
    )
    assert abs(per_card_with_ckpt - 6.75) < 1e-9

    # 3. 固定大小 buffer 循环复用
    peak_items, total_sum = process_with_constant_buffer(list(range(100)), chunk_size=16)
    assert peak_items == 16
    assert total_sum == sum(range(100))

    # 4. 一个简单的“碎片导致分配失败”例子
    allocator = SimpleAllocator(total_gb=8.0, largest_contiguous_free_gb=0.8)
    assert allocator.can_allocate(0.5) is True
    assert allocator.can_allocate(1.0) is False

    print("shards:", shards)
    print("restored:", restored)
    print("per_card:", per_card, "GB")
    print("per_card_with_ckpt:", per_card_with_ckpt, "GB")
    print("peak_buffer_items:", peak_items)
    print("total_sum:", total_sum)
    print("can_allocate_1GB:", allocator.can_allocate(1.0))


if __name__ == "__main__":
    main()
```

这段代码分别对应 ZeRO-R 的三个核心点：

| 代码部分 | 对应机制 | 说明 |
| --- | --- | --- |
| `shard_activation` / `allgather_shards` | 激活分片与按需恢复 | 前向保存分片，反向再恢复完整视图 |
| `process_with_constant_buffer` | 常量 buffer | 每次只处理固定 chunk，峰值由 chunk 上限决定 |
| `SimpleAllocator` | 碎片影响 | 即使总量足够，只要连续块不够，仍然可能分配失败 |

如果把它映射到真实训练框架，训练循环的伪代码大致如下：

```python
for layer in model.layers:
    x = layer.forward(x)

    if should_checkpoint(layer):
        # 只保存当前 rank 对应的激活分片
        act_shard = shard_for_rank(x, rank, group_size)
        save_checkpoint(layer.id, act_shard)

for layer in reversed(model.layers):
    act_shard = load_checkpoint(layer.id)

    if act_shard.is_offloaded:
        act_shard = prefetch_from_cpu(act_shard)

    # 反向到这一层时再恢复完整激活
    full_act = allgather_activation(act_shard, group)

    grad = layer.backward(full_act, grad_out)

    release(full_act)
```

如果再加上固定大小 buffer，逻辑更接近下面这样：

```python
for bucket in split_into_fixed_chunks(gradients, chunk_bytes=256 * 1024 * 1024):
    pack_into_buffer(bucket)
    reduce_or_copy(bucket)
    clear_buffer()
```

真实工程里通常还会加两类控制：

1. 通信与计算 overlap，也就是通信和计算重叠，尽量避免 GPU 等待 AllGather。
2. checkpoint 的预取窗口与释放窗口，也就是在真正用到前稍早拉取，在用完后立即释放。

一个更接近 DeepSpeed 实战的组合路径通常是：

1. 先用 ZeRO Stage 2 或 Stage 3 处理模型状态分片。
2. 再对 activation checkpoint 做 partition。
3. batch 很小时，尤其 batch size 很接近 1 时，可考虑把 checkpoint 激活 offload 到 CPU。
4. 将梯度规约、融合通信、部分归一化工作区约束到固定大小 buffer。
5. 在长时间训练中通过预留连续区域或 allocator 策略降低碎片风险。

这里 batch size 很小的场景值得单独强调。因为这时吞吐的主要瓶颈往往不再是算力利用率，而是“显存能不能装下”。CPU offload 虽然会拉低吞吐，但可能是把任务从“根本跑不起来”变成“可以接受地跑”的关键。

---

## 工程权衡与常见坑

ZeRO-R 不是免费优化。它节省显存的方式，本质上是在计算、通信、调度和实现复杂度之间重新分配成本。

| 工程挑战 | 影响 | 常见缓解方式 |
| --- | --- | --- |
| 激活分片后的 AllGather | 反向阶段通信增加 | 分层预取、通信计算重叠、合理设置 checkpoint 粒度 |
| CPU offload | Host-Device 传输带来额外延迟 | 双缓冲、异步传输、CPU 绑核、NUMA 感知 |
| buffer 配置过大 | 临时工作区反客为主，占掉显存 | 使用固定 chunk，不让 buffer 随模型线性增长 |
| 不处理碎片 | 账面显存足够但仍 OOM | 预分配连续区、减少长短生命周期对象混放 |

### 常见坑 1：以为分片后通信可以忽略

不能这样理解。ZeRO-R 降的是显存，不是通信量。  
反向经过某层之前，你仍然需要把那一层的激活拼回来。也就是说，激活分片通常会引入额外的 AllGather 或类似恢复通信。

一个简单判断式是：

$$
\text{显存下降} \neq \text{通信下降}
$$

如果你的互联带宽较弱，例如只有普通 PCIe，没有更高带宽的 GPU 互联，那么激活分片的性能代价会更明显。论文和工程实践里常说“大模型计算足够重，通信可以被隐藏一部分”，这句话成立的前提是：

1. 网络带宽不能太差。
2. kernel 粒度不能太碎。
3. checkpoint 与预取调度要合理。
4. 反向图中的同步点不能过多。

只要其中一项失效，通信就可能直接暴露在 step time 上。

### 常见坑 2：CPU offload 被总线拖垮

CPU offload 的意思是：显存放不下的那部分激活 checkpoint 或状态，暂时放到主机内存，等需要时再搬回来。

这个策略的价值很明确，但代价也很明确。若 Host-Device 传输带宽不足，吞吐会下降，尤其在 PCIe 机器上更明显。经验上，吞吐下降不是异常，而是这个策略的正常副作用。

工程上通常需要同时做三件事：

1. 双缓冲：当前 step 在算时，下一批需要的数据已经开始搬运。
2. 异步预取：不要等到真正需要该层激活时才启动 CPU 到 GPU 的拷贝。
3. 绑定 CPU 核心：降低线程调度抖动，避免数据准备线程抢不到核。

可以把这件事理解成一个取舍：

| 目标 | 更优方案 |
| --- | --- |
| 首先跑起来 | 允许 offload，接受一部分吞吐下降 |
| 首先追求吞吐 | 尽量减少 offload，优先留在 GPU 内部解决 |

### 常见坑 3：buffer 预算没有硬上限

如果团队习惯用“尽可能大的融合桶”提高单次规约效率，那么 ZeRO-R 的常量 buffer 思路就会失效。

ZeRO-R 对 buffer 的要求不是“理论最优”，而是“峰值受控”。这意味着你需要接受一个事实：最优吞吐配置未必是最优稳定性配置。尤其在超大模型训练里，只要一次 buffer 峰值把系统推过显存上限，训练就不是慢一点，而是直接失败。

通常更稳妥的配置方法是：

1. 先给 buffer 一个硬上限，例如 256MB 或 512MB。
2. 以这个上限观察吞吐与显存峰值。
3. 若显存有余量，再逐步增加，而不是一开始就按模型规模线性放大。

### 常见坑 4：只看总显存，不看碎片

很多监控面板只展示 `allocated` 和 `reserved`，但不展示“最大连续可分配块”。这会导致一个常见错觉：日志显示还有几 GB 空闲，为什么某个 attention workspace 还会申请失败？

原因通常不是“统计错了”，而是碎片已经很严重。  
这类问题如果只靠继续减 batch，常常只能缓解，不能根治。真正有效的方向通常是：

1. 让长生命周期对象进入更稳定的连续区域。
2. 减少不同生命周期对象在同一区域频繁混用。
3. 限制那些会突然申请大块空间的融合行为。
4. 在框架允许的情况下使用更稳定的 allocator 策略。

### 常见坑 5：把 ZeRO-R 当成单独开关

实际工程里，ZeRO-R 很少是“打开就自动最佳”的功能。它通常依赖多个开关一起协同：

| 组件 | 作用 |
| --- | --- |
| checkpointing 粒度 | 决定 $\alpha$，影响保留激活规模 |
| 激活分片组大小 | 决定 $N$，影响单卡分摊比例 |
| offload 策略 | 决定能否把 GPU 压力转移到 CPU |
| buffer 大小 | 决定临时工作区峰值 |
| overlap 调度 | 决定额外通信是否暴露成性能损失 |

因此调参时不能只盯着一个开关，而要把它们看成一组联动参数。

---

## 替代方案与适用边界

ZeRO-R 的最佳使用场景是：模型状态已经被处理，但 residual memory 仍然是瓶颈。它不是所有训练任务的默认答案。

| 场景 | 更合适的方法 | 原因 |
| --- | --- | --- |
| 优化器状态占大头，激活还可接受 | ZeRO-Offload Stage 2/3 | 先解决模型状态更直接 |
| 激活非常大，尤其长序列训练 | ZeRO-R + activation checkpointing | 激活是主瓶颈，叠加收益明显 |
| 单卡或小集群极限扩展 | ZeRO-Infinity / NVMe offload | 还需要把状态进一步下沉到更慢层级 |
| 模型较小，主要追求吞吐 | 不一定启用 ZeRO-R | 通信和调度开销可能得不偿失 |

可以用一张判断表快速决定是否需要它：

| 观察到的现象 | 更可能的问题 | 优先动作 |
| --- | --- | --- |
| 参数、优化器状态占满显存 | 模型状态瓶颈 | 先上 ZeRO-DP / ZeRO-Offload |
| 前向或反向中间层爆显存 | 激活瓶颈 | 看 ZeRO-R + checkpointing |
| 日志显示有空闲但仍 OOM | 碎片或大块 workspace 失败 | 看碎片管理与 buffer 上限 |
| 能跑但吞吐很低 | 通信或 offload 代价过高 | 优化 overlap、预取、互联、chunk 配置 |

一个实用的判断流程是：

1. 先区分 OOM 发生在什么位置。
   如果 OOM 多发生在初始化或 optimizer step，先怀疑模型状态。
   如果 OOM 多发生在前向/反向中间层，先怀疑激活与临时 buffer。

2. 再看 batch size 是否对 OOM 非常敏感。
   如果 batch 稍微一增就爆，通常说明激活占比较大。

3. 再看碎片迹象。
   如果训练跑一段时间后才开始随机 OOM，而不是一开始就爆，碎片的嫌疑会更大。

4. 最后再判断性能目标。
   如果当前目标是“先让训练跑起来”，ZeRO-R 很合适。
   如果当前目标是“在已经跑得动的基础上最大化吞吐”，它不一定是第一选择。

再强调一次边界：ZeRO-R 解决的是 residual memory，而不是所有系统瓶颈。它最强的价值是把原本放不下的训练变成可运行，然后再通过 overlap、预取、分桶、拓扑优化等工程手段把吞吐逐步拉回去。

---

## 参考资料

下面给出更具体的参考资料方向，便于继续往下读。重点不是“把名字记住”，而是知道每份材料分别回答什么问题。

| 资料 | 应重点关注的内容 | 适合解决的问题 |
| --- | --- | --- |
| ZeRO 论文与后续材料 | residual memory、Partitioned Activation Checkpointing、Constant-size Buffers、Memory Defragmentation | ZeRO-R 到底解决什么，不解决什么 |
| DeepSpeed 官方 ZeRO / ZeRO-Offload 文档 | Stage 2/3 的职责边界、CPU offload、训练配置项 | 如何把模型状态优化与 residual memory 优化结合 |
| Activation Checkpointing 相关文章 | checkpoint 粒度、重计算代价、与分片的叠加关系 | 如何理解 $\alpha$ 的来源 |
| 分布式训练工程博客 | overlap、双缓冲、预取、bucket 配置 | 为什么“能省显存”不等于“不会拖慢训练” |
| CUDA/PyTorch 内存管理文档 | allocator、reserved/allocated、碎片现象 | 为什么账面有空闲显存仍可能 OOM |

如果要按阅读顺序安排，建议这样看：

1. 先读 ZeRO 论文中 residual memory 相关部分，明确 ZeRO-R 的问题定义。
2. 再读 activation checkpointing 的基本机制，理解它和激活分片的关系。
3. 然后看 DeepSpeed 的工程文档，理解这些概念如何落到配置和运行时实现上。
4. 最后看 allocator 与碎片相关材料，补上“为什么有空闲仍会 OOM”这一块。

为了便于检索，下面列出建议直接关注的资料名称：

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- DeepSpeed 官方文档中 ZeRO、ZeRO-Offload、Activation Checkpointing 相关页面
- PyTorch CUDA memory management 与 activation checkpointing 文档
- 与 overlap、double buffering、host-device prefetching 相关的分布式训练工程实践文章

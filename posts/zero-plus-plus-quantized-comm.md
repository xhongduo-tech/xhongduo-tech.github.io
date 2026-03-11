## 核心结论

ZeRO++ 是在 ZeRO-3 上做的一组**通信路径优化**，不改模型结构，不改优化器更新逻辑，也不把训练主状态整体改成低比特。它改的是另一件事：**参数和梯度在网络里传输时，如何传得更短、更少、更多留在节点内**。

它包含三个彼此配合的组件：

- `qwZ`：quantized weights for ZeRO。把前向阶段参数收集时要发送的权重临时量化后再传，核心收益是**跨节点参数 all-gather 体积变小**。
- `hpZ`：hierarchical partitioning for ZeRO。重新组织分区和进程组，让一部分原本必须跨节点完成的参数收集改为**节点内完成**。
- `qgZ`：quantized gradients for ZeRO。把梯度同步从传统 reduce-scatter / all-reduce，改成**分层、量化的 all-to-all 流程**，先利用节点内高带宽，再处理较少的跨节点流量。

对新手最重要的结论只有一个：

> ZeRO++ 不是“把训练整体改成低精度”，而是“把最贵的跨节点通信改成更短、更少、更多留在节点内”。

先看一个最简化的玩具例子。设传统 ZeRO-3 的一次迭代中，有三次主要跨节点通信，每次通信量都近似记为 $M$，则：

| 阶段 | 传统 ZeRO-3 | ZeRO++ 后 | 变化原因 |
|---|---:|---:|---|
| forward 参数 all-gather | $1M$ | $0.5M$ | `qwZ` 把传输中的权重量化为 INT8 |
| backward 参数 all-gather | $1M$ | $0$ | `hpZ` 把这部分跨节点通信改成节点内 |
| 梯度 reduce-scatter / averaging | $1M$ | $0.25M$ | `qgZ` 做分层量化同步 |
| 总跨节点通信 | $3M$ | $0.75M$ | 三个优化叠加 |

因此总跨节点通信量变成：

$$
Comm_{total} = 0.5M + 0 + 0.25M = 0.75M
$$

也就是把跨节点通信从 $3M$ 压到 $0.75M$，降到原来的四分之一。论文给出的实验结论是：**训练质量与原始 ZeRO 基本持平**，而在**低带宽、多节点、小 batch** 这些通信更容易成为瓶颈的场景里，吞吐可以明显提升。

---

## 问题定义与边界

先把 ZeRO-3 的问题说清楚。

ZeRO-3 的核心价值是**分片存储**。意思是：参数、梯度、优化器状态不再让每张 GPU 都保存全量副本，而是沿 data parallel 维度切开，每张卡只保留自己那一份。这样显存压力显著下降，所以可以训练更大的模型。

但代价也很明确：**你省掉了“静态冗余存储”，就要在运行时补回“动态数据聚合”**。也就是说：

- 前向计算某一层之前，要先把这一层权重临时收集齐。
- 反向传播某一层时，也可能要为对应计算收集权重。
- 梯度计算完以后，还要把各卡的梯度片段同步、归并，再交给优化器更新。

如果把一层需要参与 collective 的数据大小记作 $M$，那么传统 ZeRO-3 的单次 collective 通信量可以粗略近似为：

$$
Communication\ per\ collective \approx M
$$

于是，一次迭代里如果以下三个阶段都要跨节点完成：

- forward 的参数 all-gather
- backward 的参数 all-gather
- 梯度 averaging / reduce-scatter

那么总跨节点通信量就可以粗略写成：

$$
Comm_{ZeRO3} \approx M + M + M = 3M
$$

这里的 $3M$ 不是严格的逐字节精确公式，而是用于理解瓶颈位置的**工程近似模型**。它想表达的是：**三次大通信都压在慢链路上时，训练会被网络拖住**。

为什么这件事在多节点训练里尤其严重？因为硬件链路本来就分层：

| 链路类型 | 典型设备 | 特点 |
|---|---|---|
| 节点内 | NVLink / NVSwitch / PCIe | 带宽高、延迟低 |
| 节点间 | InfiniBand / RoCE / Ethernet | 带宽低于节点内，且更容易拥塞 |

因此，“传 1GB 数据”不是一个抽象成本，而是要看**通过哪条链路传**。在节点内传 1GB 和在节点间传 1GB，时间可能差出数倍。

这也是 ZeRO++ 的出发点：**真正昂贵的不是通信这个动作本身，而是跨节点通信**。

再把边界说清楚。ZeRO++ 只改通信表达和通信路径，不改训练语义本身：

| 项目 | ZeRO++ 是否改变 | 说明 |
|---|---|---|
| 本地 shard 的存储精度 | 不改变 | 仍保持训练主精度，如 FP16 / BF16 |
| optimizer 更新逻辑 | 不改变 | Adam、AdamW 等更新规则不变 |
| 模型结构 | 不改变 | 不改层数、注意力结构、参数形状 |
| 通信中的权重/梯度表示 | 改变 | 可临时量化为 INT8 或更低位宽 |
| 节点内/节点间通信路径 | 改变 | 优先把重通信留在节点内 |
| 收敛目标 | 不改变 | 仍以原始训练目标为准 |

所以 ZeRO++ 的目标不是“继续压缩显存到极限”，而是尽量把跨节点通信压到：

$$
Comm_{target} \le 0.75M
$$

更准确地说，是把最贵的跨节点流量拆成三类处理：

1. 能量化的，先量化再传。
2. 能留在节点内的，尽量不要跨节点。
3. 不得不跨节点的，尽量在节点内先归并、再少量发送。

可以用一个极简图看它优化的位置：

```text
传统 ZeRO-3
forward:   inter-node all-gather
backward:  inter-node all-gather
grad sync: inter-node reduce-scatter

ZeRO++
forward:   INT8 inter-node all-gather
backward:  intra-node all-gather
grad sync: hierarchical quantized all-to-all
```

一个真实感更强的工程场景是：4 台机器、每台 8 卡，共 32 张 GPU 训练 GPT 类模型。每台机器内部有 NVSwitch，节点内通信很快；机器之间走 InfiniBand 或 RoCE，节点间通信慢得多。此时，传统 ZeRO-3 会在每层都频繁触发跨节点 collectives，GPU 常常等网络；而 ZeRO++ 会把这些 collectives 重新组织，优先消耗节点内带宽，再减少跨节点字节数。

对新手来说，可以把它理解成一句话：

> ZeRO-3 解决“存不下”，ZeRO++ 解决“传不动”。

---

## 核心机制与推导

### 1. qwZ：前向参数收集减半

`qwZ` 的目标是优化**参数 all-gather**，尤其是前向阶段的跨节点权重收集。

传统 ZeRO-3 里，某一层开始计算前，每个 rank 需要把其他 rank 上的参数片段收集起来，拼成当前层所需的完整权重。如果这些参数片段原本是 FP16，那么通信时每个数占 16 bit。

`qwZ` 的做法是：**参数存储仍保持原精度，但在发送前临时量化，在接收后再反量化**。最常见的例子是 FP16 到 INT8，于是通信体积近似减半：

$$
Comm_{forward} \approx \frac{8}{16} M = 0.5M
$$

这里的“0.5”是位宽比例带来的第一层近似。工程里还会有少量额外开销，比如：

- scale / zero-point 等量化元数据
- block 对齐带来的 padding
- quantize / dequantize kernel 的启动与访存开销

所以实际不是机械地“正好 2 倍加速”，但在理解机制时，把它看成**体积近似减半**是合理的。

问题在于：量化不能乱做。若整张 tensor 只用一个全局 scale，离群值会把刻度拉得很粗，小值区间的有效分辨率就会变差。举个直觉例子：

- 某个权重块里大多数值在 `[-0.2, 0.2]`
- 另一个块里有离群值达到 `12.0`

如果两块共享一个 scale，那么 `0.1` 这种小值可能被量化得非常粗糙，甚至多个不同的小值映射到同一个整数。

因此论文采用的是 **block-wise quantization**。做法是把张量切成多个小块，每块单独计算缩放因子：

$$
s_b = \frac{\max_{x \in block_b}|x|}{Q_{max}}
$$

其中：

- $block_b$ 表示第 $b$ 个块
- $Q_{max}$ 表示目标整数位宽的最大值，INT8 对称量化时通常可取 127

于是块内每个元素按下面的方式量化：

$$
q_i = \text{round}\left(\frac{x_i}{s_b}\right)
$$

解量化则是：

$$
\hat{x}_i = q_i \cdot s_b
$$

这种做法的意义不是“数学上更高级”，而是很务实：**每个块内部数值范围更集中，量化误差更可控**。

下面这个对比可以帮助理解：

| 量化方式 | scale 数量 | 优点 | 缺点 |
|---|---:|---|---|
| 全局量化 | 1 | 元数据最少，实现最简单 | 容易被离群值拖坏 |
| 按通道量化 | 按通道数 | 比全局更稳 | 张量重排和实现更复杂 |
| block-wise 量化 | 按 block 数 | 误差和实现成本平衡较好 | 有额外 scale 开销 |

所以 `qwZ` 的核心不是“INT8 很神奇”，而是：

1. 只在通信链路上量化，不改本地主状态。
2. 用 block-wise 控制误差。
3. 把最贵的跨节点参数传输体积直接砍半。

---

### 2. hpZ：把 backward 参数收集留在节点内

`hpZ` 是 ZeRO++ 中最容易被误读的组件，因为它的重点不是“量化”，而是**分层分区**。

先明确一个常见误区：很多人看到 ZeRO++，会以为三个组件都在做“压缩”。其实不是。`hpZ` 做的事情更接近于：**重新安排参数分片和进程组，让某些原本需要跨节点完成的收集，变成在节点内完成**。

为什么这会发生？

因为传统 ZeRO-3 的分区通常只沿全局 data parallel 组展开。如果这个组横跨多台机器，那么参数片段天然散布在不同节点上。到了 backward 某个时刻，如果计算需要某层完整权重，就不得不跨节点把这些片段再次拉回来。

`hpZ` 引入的是一种**层次化 partitioning** 思路。可以把全局 data parallel 组拆成两层：

- 一级：节点内 secondary group
- 二级：跨节点的全局 group

然后把参数分布方式重新安排，使得 backward 更常访问到的那部分参数收集，优先能在节点内 secondary group 里完成。

于是，在理想化近似下，backward 这次最贵的跨节点参数 all-gather 可以视为消失：

$$
Comm_{backward} = 0
$$

注意这里的 0 只表示：

> **跨节点通信量近似为 0**

而不是“完全没有通信”。真实情况是：

- 节点内仍然会有 all-gather 或类似收集动作
- 只是这部分走的是 NVLink / NVSwitch 等高带宽链路
- 相比节点间带宽，代价小得多，所以在跨节点成本模型里可近似忽略

可以把 `hpZ` 看成一个“路由重排器”：

| 传统做法 | hpZ 后 |
|---|---|
| backward 时直接向全局 DP 组拉参数 | 优先向节点内 secondary group 拉参数 |
| 数据分片对节点边界不敏感 | 数据分片显式利用节点边界 |
| 参数收集常常触发跨节点流量 | 参数收集更多停留在节点内 |

一个 32 卡例子更直观。假设 4 个节点、每节点 8 卡：

- 全局 rank 为 `0..31`
- 若 `hpZ` 的 secondary group 设置为每节点 8 卡
- 则节点内组应形如 `[0..7]`、`[8..15]`、`[16..23]`、`[24..31]`

这样做的意义是：**让每个节点先成为一个高带宽“小团体”**。很多 backward 期望聚齐的参数，可以先在这个小团体内部完成，避免直接去慢链路上找全局所有 rank。

所以 `hpZ` 的本质是：

$$
\text{inter-node gather} \rightarrow \text{intra-node gather}
$$

这个转换不会改变训练数学语义，但会显著改变性能轮廓，因为它把“慢路径上的大流量”替换成了“快路径上的大流量”。

---

### 3. qgZ：把梯度 reduce-scatter 改成分层量化 all-to-all

`qgZ` 是三个组件里最复杂，也最值得单独拆开的部分。

很多初学者看到“量化梯度”，会自然联想到下面这种做法：

1. 把梯度直接压成 INT8
2. 做一次 all-reduce
3. 再解压回来

但 `qgZ` 不是这样。它不是“在原流程前面套一个压缩壳”，而是**连梯度同步的 collective 结构都改了**。

传统 ZeRO-3 里的梯度同步更接近 reduce-scatter / all-reduce 思路。而 `qgZ` 改成的是一个**分层的、带量化的 all-to-all 管线**。可以分成四步理解：

1. 先把本地梯度 bucket 切成多个 slice。
2. 按目标 rank 重新排列这些 slice。
3. 在节点内先交换、先归并。
4. 只有真正需要跨节点的那部分，再量化后发出去。

用更形式化的表达，可以把梯度同步成本拆成：

$$
Comm_{grad} = Comm_{intra} + Comm_{inter}
$$

ZeRO++ 的关键是让：

- 大头尽量落到 $Comm_{intra}$
- 而 $Comm_{inter}$ 只保留较小的必要部分，并且在发送前做量化

所以跨节点部分可被进一步压缩到近似：

$$
Comm_{grad} \approx 0.25M
$$

这里的 `0.25M` 也不是一条普适物理定律，而是论文在典型机制叠加后的一个理解型近似：  
**先节点内归并减少跨节点数据量，再对跨节点部分量化，最终把跨节点梯度通信压到更小量级**。

论文里，这一步不是单靠一个“量化函数”完成的，而是要配合三类工程机制：

| 机制 | 作用 | 如果缺失会怎样 |
|---|---|---|
| tensor slice reordering | 保证每个分片被送到正确目标 rank | 数据路由错位，梯度会归并到错误位置 |
| pipelined intra/inter all-to-all | 节点内和节点间通信并行推进 | GPU 空转变多，链路利用率下降 |
| fused quant/dequant/reduce kernel | 把量化、反量化、归并融合 | kernel 启动开销和显存搬运开销上升 |

这三件事里，最容易被忽略的是 **slice reorder**。因为 all-to-all 和 all-reduce 不同，前者要求发送端明确知道“哪一片发给谁”。如果 bucket 被切成若干 slice，但没有先按目标 rank 重排，那么接收端即使收到了数据，也可能不知道它在最终梯度 shard 里的正确位置，训练结果会直接错误。

可以把 `qgZ` 的直觉概括成一句话：

> 先在节点内把“该合并的尽量合并”，再把剩下必须过慢链路的部分压缩后发出。

下面给一个新手更容易读懂的伪代码：

```text
for each gradient bucket:
    split bucket into slices
    reorder slices by destination rank
    quantize each slice block-wise

    intra-node exchange first
    dequantize and locally reduce arrived chunks

    for chunks that still need remote transfer:
        re-quantize them
        launch inter-node all-to-all

    receive remote chunks
    dequantize them
    reduce into final gradient shard
```

如果把这段逻辑展开，它近似对应下面的时序：

```text
bucket 切分
    ↓
按目标 rank 重排
    ↓
节点内 all-to-all
    ↓
节点内归并
    ↓
对剩余远程部分量化
    ↓
跨节点 all-to-all
    ↓
最终反量化 + reduce
```

所以 `qgZ` 不是“量化一下梯度”这么简单，而是：

- 通信对象变了：从简单平均变成目的地驱动的数据交换
- 通信顺序变了：先节点内，再节点间
- 实现方式变了：要做重排、流水、融合 kernel

最终三部分合起来，总跨节点通信就是：

$$
Comm_{total} = Comm_{forward} + Comm_{backward} + Comm_{grad}
$$

代入前面的近似模型：

$$
Comm_{total} = 0.5M + 0 + 0.25M = 0.75M
$$

把三个组件再并排看一遍：

| 组件 | 主要优化对象 | 主要手段 | 对跨节点通信的影响 |
|---|---|---|---|
| qwZ | 参数 all-gather | 传输时 block-wise INT8 量化 | 体积近似减半 |
| hpZ | backward 参数收集 | 分层分区、节点内优先 | 某些跨节点 gather 直接消失 |
| qgZ | 梯度同步 | 分层 all-to-all + 量化 + 流水 | 进一步压缩剩余跨节点流量 |

到这里可以把 ZeRO++ 的逻辑总结成一个更工程化的式子：

$$
\text{ZeRO++} = \text{量化传输} + \text{层次化分区} + \text{层次化梯度路由}
$$

---

## 代码实现

下面给一个**可直接运行**的 Python 玩具实现。它不依赖 DeepSpeed，也不依赖 GPU，目标只有两个：

1. 演示通信量如何从 $3M$ 变成 $0.75M$。
2. 演示为什么 block-wise 量化通常比全局量化更稳。

这段代码只用标准库，复制到本地执行即可。

```python
from math import isclose


def comm_zero(m: float) -> dict:
    """Toy communication model for ZeRO-3."""
    return {
        "forward": 1.0 * m,
        "backward": 1.0 * m,
        "grad": 1.0 * m,
        "total": 3.0 * m,
    }


def comm_zeropp(m: float) -> dict:
    """Toy communication model for ZeRO++."""
    return {
        "forward": 0.5 * m,   # qwZ
        "backward": 0.0 * m, # hpZ: inter-node traffic is removed
        "grad": 0.25 * m,    # qgZ
        "total": 0.75 * m,
    }


def _calc_scale(block, qmax=127):
    max_abs = max(abs(x) for x in block) if block else 0.0
    if max_abs == 0.0:
        return 1.0
    return max_abs / qmax


def quantize_global(values, qmax=127):
    """One scale for the whole tensor."""
    scale = _calc_scale(values, qmax=qmax)
    q_values = [max(-qmax, min(qmax, int(round(x / scale)))) for x in values]
    return q_values, scale


def dequantize_global(q_values, scale):
    return [x * scale for x in q_values]


def quantize_block(values, block_size=4, qmax=127):
    """Block-wise symmetric quantization."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    q_values = []
    scales = []

    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        scale = _calc_scale(block, qmax=qmax)
        q_block = [max(-qmax, min(qmax, int(round(x / scale)))) for x in block]
        q_values.extend(q_block)
        scales.append(scale)

    return q_values, scales


def dequantize_block(q_values, scales, block_size=4):
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    restored = []
    for block_idx, scale in enumerate(scales):
        start = block_idx * block_size
        block = q_values[start:start + block_size]
        restored.extend([x * scale for x in block])
    return restored


def mse(a, b):
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def main():
    # 1) Toy communication example
    m = 1.0
    z = comm_zero(m)
    zp = comm_zeropp(m)

    assert isclose(z["total"], 3.0)
    assert isclose(zp["total"], 0.75)
    assert isclose(z["total"] / zp["total"], 4.0)

    # 2) Toy quantization example:
    # First block has tiny values, second block has large values.
    values = [0.10, -0.20, 0.15, -0.18, 12.0, -10.0, 9.5, -11.5]

    q_global, global_scale = quantize_global(values)
    restored_global = dequantize_global(q_global, global_scale)

    q_block, scales = quantize_block(values, block_size=4)
    restored_block = dequantize_block(q_block, scales, block_size=4)

    global_err = mse(values, restored_global)
    block_err = mse(values, restored_block)

    # Block-wise quantization should usually be more accurate on mixed-scale tensors.
    assert block_err < global_err

    print("== Communication model ==")
    print(f"ZeRO total inter-node comm:    {z['total']:.2f} M")
    print(f"ZeRO++ total inter-node comm:  {zp['total']:.2f} M")
    print(f"Compression ratio:             {z['total'] / zp['total']:.2f}x")

    print("\n== Quantization model ==")
    print(f"Global quantization MSE:       {global_err:.8f}")
    print(f"Block-wise quantization MSE:   {block_err:.8f}")
    print(f"Block-wise better:             {block_err < global_err}")

    print("\nOriginal values:")
    print(values)

    print("\nRestored with global scale:")
    print([round(x, 4) for x in restored_global])

    print("\nRestored with block-wise scale:")
    print([round(x, 4) for x in restored_block])


if __name__ == "__main__":
    main()
```

这段程序为什么是“可运行”的？

- 不依赖第三方库。
- `main()` 入口完整。
- 全局量化和 block-wise 量化都给了完整的量化与反量化逻辑。
- 用 `assert` 验证了通信比例和误差关系。
- 可以直接通过终端执行：`python3 demo_zeropp.py`

如果运行后看到 `Block-wise better: True`，它说明的是：

- 在混合量级张量上
- 全局单一 scale 更容易被大值主导
- block-wise 通常能保留更多小值细节

这就是 `qwZ` 采用 block-wise quantization 的核心原因。

为了让误差来源更明确，还可以把全局量化与 block-wise 量化的差异写成表格：

| 方法 | scale 粒度 | 对小值是否友好 | 对离群值是否敏感 |
|---|---|---|---|
| 全局量化 | 整张 tensor 一个 scale | 差 | 很敏感 |
| block-wise 量化 | 每个 block 一个 scale | 更好 | 敏感度更低 |

上面是玩具实现。真实工程里，这些操作不会在 Python 层完成，而会放到 CUDA kernel 或通信库集成路径里。对应伪代码大致如下：

```python
# qwZ: before forward all-gather
for param_shard in local_param_shards:
    q_weight, scale = quantize_block_cuda(param_shard, bits=8, block_size=128)
    gathered_q = dist.all_gather(q_weight, group=dp_group)
    full_weight = dequantize_block_cuda(gathered_q, scale)
    output = matmul(input, full_weight)

# qgZ: hierarchical gradient synchronization
for grad_bucket in grad_buckets:
    local_q, local_scale = quantize_block_cuda(grad_bucket, bits=8, block_size=128)
    intra_chunks = dist.all_to_all(local_q, group=intra_node_group)
    fused_local = fused_dequant_reduce_cuda(intra_chunks, local_scale)

    remote_q, remote_scale = quantize_block_cuda(
        fused_local.remote_part(), bits=8, block_size=128
    )
    remote_chunks = dist.all_to_all(remote_q, group=inter_node_group)
    final_grad = fused_dequant_reduce_cuda(remote_chunks, remote_scale)
```

用户侧最容易看到的不是这些内部 kernel，而是 DeepSpeed 配置项。可以把组件和配置的关系理解成：

| 组件 | 常见配置项 | 作用 |
|---|---|---|
| qwZ | `zero_quantized_weights` | 打开量化权重通信 |
| hpZ | `zero_hpz_partition_size` | 指定二级分区大小，通常等于每节点 GPU 数 |
| qgZ | `zero_quantized_gradients` | 打开量化梯度通信 |
| 通信重叠 | `overlap_comm` | 尽量把计算与通信流水化 |
| 连续梯度 | `contiguous_gradients` | 降低碎片和额外拷贝 |

一个更接近真实部署的例子是：4 节点、每节点 8 卡、训练 30B 级模型。此时：

- 若 `zero_hpz_partition_size=8`
- 它表达的意思是“把每个节点内部 8 张卡作为一个高带宽组”
- backward 参数收集优先在这个组内完成
- 剩余必须跨节点的部分，再交给 `qgZ`

所以，站在工程实现角度，ZeRO++ 不是一个独立“神奇算子”，而是一组彼此耦合的通信改造：

1. 权重收集时压缩。
2. 参数分区时分层。
3. 梯度同步时改路由。

---

## 工程权衡与常见坑

ZeRO++ 不是“白拿 4 倍通信优化”。它用以下成本换收益：

- 更多通信路径控制逻辑
- 量化 / 反量化带来的额外 kernel
- 更严格的 chunk 排序与同步约束
- 某些配置下的额外内存占用

先看最常见的坑：

| 坑 | 后果 | 检查项 |
|---|---|---|
| 跳过 block-wise scale | `qwZ` 误差大，前向不稳 | 检查 scale 是否按 block 计算 |
| 把通信量化误解成本地训练精度量化 | loss 抖动或收敛异常 | 确认 local shard 和 optimizer 仍是 FP16/BF16 |
| `qgZ` 的 slice reorder 错位 | 梯度落错 rank，训练直接坏掉 | 校验每个 slice 的源 rank、目标 rank、offset |
| intra/inter pipeline 顺序错误 | 吞吐下降，严重时死锁 | 检查 chunk 编号和通信事件依赖 |
| `hpZ` group 大小配错 | 本应节点内的流量跑到节点间 | 检查 secondary group 是否按节点切分 |
| 忽略额外副本开销 | 显存预算不准 | 评估 `hpZ` 引入的额外参数视图或缓存 |
| bucket 太小 | kernel 与通信启动开销过高 | 调整 bucket size，避免过碎 |
| bucket 太大 | overlap 不充分，尾部阻塞明显 | 看 profile 中通信尾巴是否过长 |

有两个原则必须单独强调。

第一，**通信量化不等于训练主状态量化**。这件事可以写成：

$$
Local\ shard,\ optimizer\ state \in \{FP16,\ BF16,\ FP32\}
$$

而通信链路上可能是：

$$
In\text{-}flight\ representation \in \{INT8,\ \text{low-bit}\}
$$

两者不一样。前者是“本地如何存、如何算、如何更新”，后者是“网络上传输时如何编码”。如果把这两件事混为一谈，就会误以为 ZeRO++ 等价于低比特训练，这是不对的。

第二，`hpZ` 省的是**跨节点带宽**，不是免费。因为要实现层次化调度，系统可能需要额外维护更适合节点内访问的参数布局、缓存或副本视图。所以在显存已经非常紧的场景里，收益要和成本一起看。可以把这种权衡写成：

$$
Net\ Benefit = Saved\ InterNode\ Comm - Extra\ Memory\&Kernel\ Overhead
$$

也就是说，真正值得不值得开，不是只看通信量，而是看**整体迭代时间**有没有下降。

怎么验证 `hpZ` 是否真的只在节点内完成了你期望的那部分收集？有三个简单办法：

| 验证方式 | 看什么 | 若异常说明什么 |
|---|---|---|
| 看进程组划分 | secondary group 是否按节点切分 | 组划分可能错了 |
| 看链路利用率 | backward 参数收集是否主要走 NVLink/NVSwitch | 若仍占满 IB，说明没留在节点内 |
| 看 profiler 时间线 | backward all-gather 是否仍大量卡在 inter-node collective | 说明 `hpZ` 没生效或配置错误 |

一个最朴素的 rank 分组例子如下。假设每节点 8 卡，那么理想 secondary group 应像这样：

```text
node 0: ranks [0,1,2,3,4,5,6,7]
node 1: ranks [8,9,10,11,12,13,14,15]
node 2: ranks [16,17,18,19,20,21,22,23]
node 3: ranks [24,25,26,27,28,29,30,31]
```

如果你看到的是交叉节点混编，例如 `[0,8,16,24,...]` 这种形式，那么 `hpZ` 就很可能失去了“把通信留在节点内”的意义。

还有一个容易忽略的点：**低带宽场景不等于所有问题都来自网络**。有时看起来是 ZeRO-3 慢，实际上瓶颈可能是：

- bucket 切得太碎
- overlap 没开
- kernel fusion 不充分
- CPU 侧调度或 dataloader 成了瓶颈

因此，上 ZeRO++ 之前，先确认当前瓶颈确实是跨节点 collective，而不是别的环节。

---

## 替代方案与适用边界

不是所有集群都必须上 ZeRO++。它最适合的是：**模型大、节点多、跨节点慢、batch 又不够大到能完全掩盖通信** 的场景。

先把几种常见方案放在一起看：

| 方案 | 适合场景 | 通信特点 | 误差/复杂度 |
|---|---|---|---|
| 传统 ZeRO-3 | 高带宽、大 batch | 不做量化，collective 路径更直接 | 误差最小，实现简单 |
| ZeRO++ | 低带宽、小 batch、多节点 | 跨节点通信显著压缩 | 量化、分层调度更复杂 |
| 梯度压缩 / 1-bit 类方案 | 极端通信瓶颈 | 梯度可压得更狠 | 对收敛和调参更敏感 |
| 只做 FP16/BF16 混合精度 | 显存和算力受限，但网络尚可 | 主要优化显存和算力，不专治跨节点通信 | 工程最成熟 |
| FSDP + 通信优化 | 使用 PyTorch 原生分片体系 | 取决于具体通信插件与拓扑 | 生态成熟，但策略需单独评估 |

可以用一个经验式判断是否值得上 ZeRO++：

$$
\text{当 } \frac{T_{comm}}{T_{compute}} \ll 1,\ \text{传统 ZeRO-3 通常已足够}
$$

意思是：如果通信时间本来就远小于计算时间，那么你继续优化通信，收益自然有限。反而可能因为量化、重排、流水化带来的额外复杂度，让系统更难调。

反过来，如果：

$$
\frac{T_{comm}}{T_{compute}} \gtrsim 1
$$

也就是通信时间已经接近或超过计算时间，那么 ZeRO++ 往往更值得考虑。因为此时训练速度被网络卡住，优化网络路径的回报更直接。

可以把是否适合 ZeRO++ 的信号总结为：

| 信号 | 是否偏向 ZeRO++ | 原因 |
|---|---|---|
| 100Gbps 级节点间互联 | 是 | 跨节点带宽更容易成为瓶颈 |
| 400Gbps 以上高带宽互联 | 未必 | 通信已较快，额外复杂度可能不划算 |
| 每卡 batch 很小 | 是 | 计算难以掩盖通信 |
| 每卡 batch 很大 | 未必 | 通信可被计算隐藏 |
| 模型参数量很大 | 是 | 参数收集和梯度同步更重 |
| 单节点训练 | 否 | 跨节点优化价值很低 |

对新手来说，可以先记一个朴素决策表：

- 高带宽、大 batch：先用传统 ZeRO-3。
- 低带宽、小 batch：优先评估 ZeRO++。
- 数值稳定性非常敏感：先只开 `qwZ`，确认误差可接受，再逐步加 `hpZ`、`qgZ`。
- 显存已经非常紧：开 `hpZ` 前先核算额外副本或缓存成本。
- 拓扑复杂、节点配置不整齐：先确认进程组和 rank 映射，再谈优化。

再用“100Gbps 链路”做一个直观对比：

| 路径顺序 | ZeRO-3 | ZeRO++ |
|---|---|---|
| 前向参数收集 | 直接跨节点 all-gather | 先量化，再跨节点 all-gather |
| 反向参数收集 | 直接跨节点 all-gather | 改为节点内 all-gather |
| 梯度同步 | reduce-scatter / all-reduce | 节点内优先，再量化跨节点 |

所以 ZeRO++ 的本质不是“额外塞一个压缩器”，而是：

$$
\text{Precision-aware transport} + \text{hierarchical routing}
$$

也就是把**量化传输**和**层次化路由**一起做。

最后要强调一个适用边界：ZeRO++ 解决的是**分布式训练中的通信瓶颈**。如果你的主要问题是以下这些，它不是第一选择：

- 单卡或单节点显存不够
- 前向算子本身算力不足
- 数据加载跟不上
- 模型结构导致算子效率很差

换句话说，ZeRO++ 是网络侧优化，不是万能训练加速器。

---

## 参考资料

下面按“原文、工程落地、综述”三类给出资料。原文负责机制和实验，工程文档负责配置方式，综述适合快速建立全局图景。

| 名称 | URL | 用途 |
|---|---|---|
| ZeRO++ ICLR 2024 论文 PDF | https://proceedings.iclr.cc/paper_files/paper/2024/file/d8ca28a32c05cd3b9b0940e43720f31b-Paper-Conference.pdf | 最核心资料，覆盖 `qwZ`、`hpZ`、`qgZ` 的机制、实验和收敛结果 |
| OpenReview: ZeRO++ | https://openreview.net/forum?id=BV1aISGPym | 论文页面，便于快速查看摘要、会议信息和讨论入口 |
| Microsoft Research ZeRO++ 页面 | https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-large-model-training/ | 论文入口与摘要，适合先确认核心结论 |
| DeepSpeed ZeRO++ 教程 | https://www.deepspeed.ai/tutorials/zeropp/ | 工程配置入口，查看开关和使用方式 |
| Microsoft Research 博客：4X less communication | https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/ | 工程视角解释吞吐提升场景与 RLHF 示例 |
| DeepSpeed ZeRO 文档 | https://www.deepspeed.ai/tutorials/zero/ | 补足 ZeRO / ZeRO-2 / ZeRO-3 基础，适合新手先补前置知识 |
| Emergent Mind: ZeRO++ | https://www.emergentmind.com/papers/2306.10209 | 快速综述，适合建立概念图后再回读原文 |

阅读顺序建议如下：

| 阅读目标 | 先看什么 | 重点看什么 |
|---|---|---|
| 搞清机制 | ICLR 论文 | 三个组件分别作用在哪三类 collective 上 |
| 弄懂 ZeRO-3 前置知识 | DeepSpeed ZeRO 文档 | 参数、梯度、优化器状态如何分片 |
| 想落地配置 | DeepSpeed ZeRO++ 教程 | `zero_quantized_weights`、`zero_hpz_partition_size`、`zero_quantized_gradients` |
| 想判断是否值得上 | Microsoft 博客 | 100Gbps / 400Gbps、多模型规模、RLHF 场景吞吐差异 |
| 想快速回顾全局 | OpenReview 或 Emergent Mind | 摘要、关键词和问题背景 |

如果你是第一次读这类论文，推荐用下面的顺序，而不是从公式细节硬啃：

1. 先看 ZeRO-3 到底为什么需要频繁参数收集。
2. 再看 ZeRO++ 把哪三次通信分别改了什么。
3. 最后回到论文图表，理解它为什么在低带宽和小 batch 下收益更大。

这样读，概念会更稳，不容易把“通信量化”和“训练主精度量化”混为一谈。

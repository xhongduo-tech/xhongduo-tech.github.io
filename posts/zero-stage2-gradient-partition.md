## 核心结论

ZeRO Stage 2 的核心变化只有一句话：在 ZeRO Stage 1 已经把优化器状态分片的基础上，再把梯度也按数据并行组切分，让每张卡只保留“自己负责更新的那一片参数”对应的梯度。

先把术语定义清楚。

数据并行：每张卡执行同一份模型，只是输入 mini-batch 不同。  
参数：模型要学习的权重。  
梯度：损失函数对参数的导数，决定参数该如何更新。  
分片：把一整份张量按规则切成多块，不同 rank 持有不同块。  
Reduce-Scatter：先对同名张量做规约求和，再把规约结果按块分发给不同 rank。  
All-Gather：各 rank 拿出自己持有的分片，再拼回完整张量。

标准 DDP 的特点是：前向时每张卡有完整参数，反向结束后每张卡也有完整梯度。这样做实现简单，但完整梯度会在每张卡上重复一遍。ZeRO Stage 2 去掉的就是这部分重复。某一层梯度一旦 ready，就不再走“每个人都拿到完整求和结果”的路径，而是直接做 Reduce-Scatter，把求和后的梯度按 owner rank 分片保存。非 owner rank 对这层梯度不再长期驻留。

因此，在论文和官方材料常见的近似记账口径下，单卡模型状态可以从：

$$
4\Psi + 12\Psi/N
$$

进一步降到：

$$
2\Psi + (2+12)\Psi/N = 2\Psi + 14\Psi/N
$$

其中：

- $\Psi$ 表示模型参数以半精度存储时的一份参数内存
- $N$ 表示数据并行组大小
- 这里的系数是工程上常见的 Adam 混合精度近似口径，用来比较不同 Stage 的相对变化，不是每个框架逐字节都完全一致的账本

当 $N$ 足够大时，$14\Psi/N$ 这一项会变小，所以常把 Stage 2 简写成“单卡模型状态约为 $2\Psi$ 量级”。这个说法的含义是：完整参数副本仍然保留，但梯度和优化器状态的大头已经被分摊了。

通信上，Stage 2 的重点不是“把通信做没”，而是“用与 DDP 同阶的通信，换更低的显存驻留”。标准 DDP 常把梯度同步写成一次 AllReduce，通信量近似记作：

$$
\mathrm{AllReduce}(\Psi) \approx 2\Psi
$$

而 Stage 2 改成：

$$
\mathrm{ReduceScatter}(\Psi) + \mathrm{AllGather}(\Psi) \approx 2\Psi
$$

所以核心收益是显存，不是通信字节数数量级的下降。

用一个最小例子理解就够了。假设总梯度长度为 8，4 张卡做数据并行。DDP 会让 4 张卡最终都拿到长度 8 的完整求和梯度；Stage 2 则让每张卡只留下其中 2 个元素的 owner shard。这样本轮优化器更新时，每张卡只处理自己负责的梯度和优化器状态。

---

## 问题定义与边界

Stage 2 解决的问题很具体：在 GPU 数量已经固定、又不想引入张量并行或流水并行的前提下，如何继续降低单卡显存占用，让更大的模型能在现有机器上训练。

先看训练里主要有哪些状态。

| 状态 | 作用 | 是否在每步都参与训练 | Stage 2 是否分片 |
|---|---|---|---|
| 参数 | 前向和反向计算都要读 | 是 | 否 |
| 梯度 | 反向产生，优化器更新时使用 | 是 | 是 |
| 优化器状态 | 例如 Adam 的一阶矩、二阶矩 | 是 | 是 |

如果以混合精度 Adam 为背景，常见近似记账如下：

| 方案 | 参数副本 | 梯度副本 | 优化器状态 | 单卡模型状态近似 | 通信特征 |
|---|---:|---:|---:|---:|---|
| Stage 0 / 标准 DDP | 全量 | 全量 | 全量 | $4\Psi + 12\Psi$ | AllReduce，约 $2\Psi$ |
| ZeRO Stage 1 | 全量 | 全量 | $1/N$ | $4\Psi + 12\Psi/N$ | 仍以梯度同步为主 |
| ZeRO Stage 2 | 全量 | $1/N$ | $1/N$ | $2\Psi + 14\Psi/N$ | Reduce-Scatter + All-Gather，约 $2\Psi$ |

这里必须把边界说清楚，否则很容易把 Stage 2 和 Stage 3 混在一起。

第一，Stage 2 不是参数分片。  
前向时每张卡仍需要完整参数视图，所以参数副本依旧是全量。

第二，Stage 2 的主要收益发生在反向阶段。  
因为梯度是在 backward 中生成并被尽早分片回收的，峰值显存通常也出现在这个阶段附近。

第三，Stage 2 不改变模型图的算子切分方式。  
它仍属于数据并行体系中的内存优化，而不是把一个线性层拆到多张卡上执行。

第四，Stage 2 的上限很明确。  
如果“完整参数副本本身都放不下”，那 Stage 2 不足以解决问题，必须走 Stage 3、张量并行、流水并行，或者结合参数/优化器 offload。

举一个具体数字。假设是 8 卡、1B 参数、Adam、混合精度训练。标准 DDP 下，每卡都保留完整参数、完整梯度、完整优化器状态，模型状态会非常重。切到 Stage 2 后，完整参数仍在，但梯度和 Adam 状态只留 $1/8$。这时显存下降的来源不是“模型变小”，而是“重复副本减少”。

对新手来说，可以用一句话判断适用性：

$$
\text{Stage 2 适合：完整参数还能放下，但完整梯度和完整优化器副本放不下。}
$$

---

## 核心机制与推导

先看机制，再看公式。

Stage 2 的执行逻辑可以压缩成四步：

1. 前向时，每张卡读取完整参数，行为和 DDP 基本一致。
2. 反向传播到某一层时，本地先得到这层梯度。
3. 这份梯度不再做传统 AllReduce，而是进入 Reduce-Scatter：先跨卡求和，再按 owner 规则把结果切片。
4. owner rank 保留自己那片梯度用于更新；其他 rank 释放非本地 shard。

这就是 Stage 2 的本质：梯度不再以完整副本的形式长期驻留在每张卡上。

把内存变化写成一个统一账本更清楚：

| 阶段 | 参数 | 梯度 | 优化器 | 合计 |
|---|---:|---:|---:|---:|
| Stage 0 | $2\Psi$ | $2\Psi$ | $12\Psi$ | $4\Psi + 12\Psi$ |
| Stage 1 | $2\Psi$ | $2\Psi$ | $12\Psi/N$ | $4\Psi + 12\Psi/N$ |
| Stage 2 | $2\Psi$ | $2\Psi/N$ | $12\Psi/N$ | $2\Psi + 14\Psi/N$ |

这里要注意两点。

第一，这是一种“比较不同方案相对变化”的近似表达。  
不同框架的实现细节、buffer、bucket、master weight、梯度精度、通信缓存，都会让真实显存比这个公式更复杂。

第二，Stage 2 相比 Stage 1，唯一新增的关键变化就是：

$$
\text{梯度：从 } 2\Psi \text{ 变成 } 2\Psi/N
$$

所以记忆时抓住这个主线即可：  
Stage 1 去掉优化器状态的全量重复；Stage 2 再去掉梯度的全量重复。

为什么通信量没有突然暴涨？因为 AllReduce 可以拆开看。对一个总大小为 $\Psi$ 的梯度向量，有一个常见理解方式：

$$
\mathrm{AllReduce} = \mathrm{ReduceScatter} + \mathrm{AllGather}
$$

因此，Stage 2 并不是凭空引入一类更重的同步，而是把“每个人都拿完整结果”的同步过程，改写成“先只拿 owner shard，必要时再拼回完整结果”的过程。总通信量仍与 DDP 同阶，只是中间张量的驻留形式发生了变化。

下面用长度为 8、4 张卡的玩具例子把这个过程写完整。假设 4 个 rank 的 owner 映射如下：

| rank | 负责的梯度位置 |
|---|---|
| rank0 | 0, 1 |
| rank1 | 2, 3 |
| rank2 | 4, 5 |
| rank3 | 6, 7 |

每张卡本地反向后，都先得到一份长度为 8 的本地梯度：

$$
g^{(0)}, g^{(1)}, g^{(2)}, g^{(3)} \in \mathbb{R}^8
$$

规约后的总梯度为：

$$
g = \sum_{r=0}^{3} g^{(r)}
$$

如果是 DDP，那么所有 rank 最终都会持有完整的 $g$。  
如果是 Stage 2，那么 Reduce-Scatter 结束后：

$$
\begin{aligned}
\text{rank0} &\leftarrow g[0:2] \\
\text{rank1} &\leftarrow g[2:4] \\
\text{rank2} &\leftarrow g[4:6] \\
\text{rank3} &\leftarrow g[6:8]
\end{aligned}
$$

于是每张卡只需要保存自己负责更新的那一段。若下一阶段需要完整参数视图，再通过 All-Gather 把更新后的参数分片拼回完整参数。

从训练生命周期看，可以把 Stage 2 理解成三层状态管理：

| 层 | 是否全量可见 | 为什么 |
|---|---|---|
| 参数 | 是 | 前向和局部反向都要读完整参数 |
| 梯度 | 否 | 只需要 owner rank 长期保留 |
| 优化器状态 | 否 | 只和 owner 参数分片绑定 |

因此，Stage 2 的重点不是“参数不见了”，而是“梯度不再有 N 份完整副本”。

---

## 代码实现

下面先给一个真正可运行的 Python 玩具实现。它不依赖 DeepSpeed，也不依赖多进程通信库，但完整模拟了 Stage 2 最关键的数据流：本地完整梯度、规约求和、按 owner 分片、必要时重组完整视图。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ZeroStage2Result:
    reduced_full_grad: List[float]
    grad_shards: List[List[float]]
    shard_owners: List[range]


def reduce_sum(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        raise ValueError("vectors must not be empty")

    total_len = len(vectors[0])
    if any(len(vec) != total_len for vec in vectors):
        raise ValueError("all vectors must have the same length")

    reduced = [0.0] * total_len
    for vec in vectors:
        for i, value in enumerate(vec):
            reduced[i] += float(value)
    return reduced


def shard_ranges(total_len: int, world_size: int) -> List[range]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if total_len % world_size != 0:
        raise ValueError("total_len must be divisible by world_size")

    shard_len = total_len // world_size
    return [range(rank * shard_len, (rank + 1) * shard_len) for rank in range(world_size)]


def reduce_scatter_sum(local_grads: Sequence[Sequence[float]], world_size: int) -> ZeroStage2Result:
    if len(local_grads) != world_size:
        raise ValueError("number of local gradients must equal world_size")

    full_grad = reduce_sum(local_grads)
    owners = shard_ranges(len(full_grad), world_size)
    shards = [[full_grad[i] for i in owner] for owner in owners]
    return ZeroStage2Result(
        reduced_full_grad=full_grad,
        grad_shards=shards,
        shard_owners=owners,
    )


def all_gather(shards: Sequence[Sequence[float]]) -> List[float]:
    full = []
    for shard in shards:
        full.extend(float(x) for x in shard)
    return full


def demo() -> None:
    # 4 个 rank，本地反向后都得到长度为 8 的梯度
    local_grads = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [2, 0, 2, 0, 2, 0, 2, 0],
    ]

    result = reduce_scatter_sum(local_grads, world_size=4)

    assert result.reduced_full_grad == [4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 10.0]
    assert result.grad_shards == [
        [4.0, 4.0],
        [6.0, 6.0],
        [8.0, 8.0],
        [10.0, 10.0],
    ]

    rebuilt = all_gather(result.grad_shards)
    assert rebuilt == result.reduced_full_grad

    print("Full reduced gradient:", result.reduced_full_grad)
    for rank, (owner, shard) in enumerate(zip(result.shard_owners, result.grad_shards)):
        print(f"Rank {rank} owns positions [{owner.start}, {owner.stop}) -> {shard}")
    print("All-gather check passed.")


if __name__ == "__main__":
    demo()
```

这段代码可以直接运行，输出会明确告诉你两件事：

1. 全部 rank 的本地梯度先被规约成一份完整梯度。
2. 这份完整梯度随后被切成 4 片，每个 rank 只保留自己的 shard。

对应到真实框架，Stage 2 的行为不是“反向时从来没有完整梯度出现过”，而是“梯度一旦 ready，就尽快进入分片规约和回收流程，不再让完整梯度长期驻留”。

为了把这个过程再看得更细一点，可以把同一例子写成表格。

| 阶段 | rank0 | rank1 | rank2 | rank3 |
|---|---|---|---|---|
| 本地反向后 | 长度 8 本地梯度 | 长度 8 本地梯度 | 长度 8 本地梯度 | 长度 8 本地梯度 |
| 规约后完整梯度 | \multicolumn{4}{c}{`[4, 4, 6, 6, 8, 8, 10, 10]`} |
| Reduce-Scatter 后 | `[4, 4]` | `[6, 6]` | `[8, 8]` | `[10, 10]` |
| 优化器更新时保留 | 本地 shard | 本地 shard | 本地 shard | 本地 shard |

如果换成 DeepSpeed，最小配置通常长这样：

```json
{
  "train_batch_size": 64,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  }
}
```

这些配置项各自的作用需要分开理解：

| 配置项 | 作用 | 为什么重要 |
|---|---|---|
| `stage: 2` | 开启梯度分片与优化器状态分片 | 决定是否进入 ZeRO-2 路径 |
| `reduce_scatter: true` | 用 reduce-scatter 替代纯 all-reduce 路径 | 决定梯度同步方式 |
| `allgather_partitions: true` | 在需要完整参数视图时聚合参数分片 | 决定参数重建路径 |
| `contiguous_gradients: true` | 把梯度整理到连续 buffer | 减少碎片和回退路径 |
| `overlap_comm: true` | 尝试让通信与 backward 重叠 | 改善 step time |

把它写成更接近训练框架的伪代码如下：

```text
for layer in reversed(model.layers):
    local_grad = backward(layer)

    # bucket ready 后，立刻参与跨 rank 规约
    grad_shard = reduce_scatter(local_grad)

    if current_rank is owner(layer.param_shard):
        keep(grad_shard)
    else:
        free_non_owner_grad()

optimizer.step(local_param_shards, local_grad_shards)

if next_forward_requires_full_params:
    full_params = all_gather(updated_param_shards)
```

这里最容易被忽略的细节有两个。

第一，真正决定峰值显存的，不只是“有没有分片”，还包括“分片发生得够不够早”。  
如果梯度已经算出来了，但还在 bucket 或临时 buffer 里滞留，峰值显存仍可能很高。

第二，真正决定吞吐的，不只是“总通信量”，还包括“bucket 粒度、通信时机、计算重叠、网络延迟”。  
所以 Stage 2 的理论公式给的是方向，不是最终性能保证。

---

## 工程权衡与常见坑

Stage 2 的优点很明确，但工程里最常见的问题也很集中。

第一类问题是：配置写成了 Stage 2，不代表运行时一定走到了理想路径。  
例如配置里开了 `reduce_scatter: true`，但日志里仍然能看到某些 bucket 用的是 `all_reduce`。这通常意味着实现存在回退路径，或者某些张量没有进入标准的 ZeRO hook。

第二类问题是：理论显存下降了，但峰值显存没有同步下降。  
原因通常不是公式错，而是时机不对。典型情况包括：

- 梯度 bucket 太小，通信和释放过于碎片化
- 梯度没有在 layer ready 后及时归并并释放
- contiguous buffer 没有生效，导致额外复制和碎片
- activation、temporary buffer、通信 buffer 才是真正峰值来源

第三类问题是：总通信字节数同阶，但 step time 变慢。  
这也不矛盾。因为训练性能不仅由字节数决定，还受下面这些因素影响：

| 因素 | 影响 |
|---|---|
| bucket 太小 | collective 次数变多，启动开销变大 |
| 网络延迟高 | 小消息代价被放大 |
| overlap 做得差 | 计算等通信，通信也等计算 |
| 参数分组异常 | 破坏原本的 bucket 合并策略 |
| 自定义模块或特殊参数 | 可能绕过标准 ZeRO 流程 |

一个更实用的判断方式是把“理论是否正确”和“实现是否跑对”拆开看。

理论层面，Stage 2 的判断标准很简单：  
梯度应该从“每卡完整副本”变成“每卡 owner shard”。

实现层面，真正要检查的是：

1. backward 期间是否出现 `reduce_scatter`
2. 非 owner 梯度是否在 bucket 完成后及时释放
3. 优化器是否只在本地 shard 上更新
4. 下一次前向前，参数是否按预期完成 gather

因此，排障时不要只看显存，还要同时看通信日志和时间线。一个实用检查清单如下：

1. 打开框架通信日志，确认 collective 类型到底是 `reduce_scatter`、`all_reduce` 还是混合。
2. 结合 profiler 看 backward 时间线，确认 bucket ready 到通信完成之间有没有长时间等待。
3. 对照显存曲线看峰值位置，如果峰值出现在 backward 后半段，通常意味着梯度释放不及时。
4. 调整 `reduce_bucket_size`、`allgather_bucket_size`，观察 collective 粒度和 step time 是否改善。
5. 检查是否存在未被 ZeRO 管理的参数组、自定义模块、稀疏参数或特殊通信路径。
6. 把网卡吞吐、GPU 利用率、step time、峰值显存放在一起看，不要只盯一个指标。

下面是一段“看日志识别是否退化”的示意：

```text
[Rank 3] bucket=17 op=all_reduce bytes=67MB
[Rank 3] bucket=18 op=reduce_scatter bytes=64MB
[Rank 3] bucket=18 release_non_owner_grad=true
[Rank 3] optimizer_step shard_params_only=true
```

如果你预期大部分梯度桶都走 Stage 2 的分片规约，那么第一行就是需要继续追踪的信号。它不一定意味着配置错误，但至少说明当前 bucket 没有按理想路径执行。

还要补充一个经常被忽略的边界：  
Stage 2 只处理模型状态，不处理激活值。如果模型的主要峰值来自 activation 而不是 optimizer/gradient，那么只开 ZeRO-2 收益可能有限，此时要结合 activation checkpointing 一起看。

---

## 替代方案与适用边界

是否该用 Stage 2，取决于瓶颈到底在哪里。

如果模型只有 100M 参数量级，标准 DDP 往往已经够用。这时系统瓶颈更常见于数据加载、kernel 利用率、batch size 设计、实验管理复杂度，而不是模型状态本身的显存冗余。为这类任务引入 ZeRO-2，收益可能不够大。

如果模型进入 1B、7B 甚至更高，但你暂时不想引入张量并行或流水并行，那么 Stage 2 往往是最先应该考虑的一步。原因是它在不改变模型切分方式的前提下，直接消掉了“完整梯度副本”和“完整优化器状态副本”的重复。

下面把几种常见方案并排看：

| 方案 | 参数驻留 | 梯度驻留 | 优化器状态 | 通信 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|---|---|
| DDP | 每卡全量 | 每卡全量 | 每卡全量 | AllReduce | 低 | 模型较小，优先求稳定 |
| ZeRO Stage 1 | 每卡全量 | 每卡全量 | 分片 | 近似 DDP | 低到中 | 优化器状态占用大 |
| ZeRO Stage 2 | 每卡全量 | 分片 | 分片 | RS + AG，约 $2\Psi$ | 中 | 参数还能放下，但梯度/优化器太大 |
| ZeRO Stage 3 | 分片 | 分片 | 分片 | 更多动态 gather/scatter | 高 | 完整参数也放不下 |
| ZeRO + Offload | 视 stage 而定 | 视 stage 而定 | 可下放到 CPU/NVMe | 更复杂 | 中到高 | GPU 显存极紧，但能接受更多数据搬运 |
| 张量并行 / 流水并行 | 模型结构被切分 | 按切分策略变化 | 按切分策略变化 | 更复杂 | 高 | 超大模型，单靠数据并行内存优化不够 |

对新手最有用的不是记住所有方案，而是记住下面三个判断。

第一，如果主要压力来自 Adam 状态，先看 Stage 1。  
第二，如果 Stage 1 后还是放不下，而且 backward 峰值高，Stage 2 通常正对问题。  
第三，如果完整参数副本本身都放不下，直接看 Stage 3 或模型并行，不要继续指望 Stage 2。

可以把这个判断压缩成一个简表：

| 现象 | 更可能有效的方案 |
|---|---|
| 优化器状态很大 | Stage 1 |
| backward 阶段梯度峰值很高 | Stage 2 |
| 前向开始前就放不下完整参数 | Stage 3 / 张量并行 / 流水并行 |
| 模型状态不是主峰值，激活才是 | activation checkpointing |
| GPU 显存不够但 CPU/NVMe 有空间 | ZeRO Offload |

所以，Stage 2 的适用边界可以概括成一句话：它适合“完整参数还能放下，但完整梯度和完整优化器副本放不下”的训练任务。

---

## 参考资料

| 来源 | 重点贡献 | 适合补充什么 |
|---|---|---|
| [ZeRO 论文（Microsoft Research）](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/) | 给出 Stage 1/2/3 的内存记账、通信分析和整体设计 | 理论来源、公式理解 |
| [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) | 说明 `stage`、`reduce_scatter`、`allgather_partitions`、`contiguous_gradients` 等配置项 | 配置方法、运行行为 |
| [PyTorch Distributed 文档](https://docs.pytorch.org/docs/stable/distributed.html) | 定义 `reduce_scatter`、`all_gather` 等 collective 的语义 | 通信原语、接口理解 |
| [Microsoft Research 博文：ZeRO & DeepSpeed](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) | 用图示解释 Stage 1/2/3 的内存节省与通信量关系 | 工程直觉、整体收益对比 |
| [DeepSpeed issue #7059](https://github.com/deepspeedai/DeepSpeed/issues/7059) | 展示配置与实际 collective 行为可能不完全一致的案例 | 排障、验证路径 |

阅读顺序建议如下。

第一步先看 ZeRO 论文，建立“为什么 Stage 2 能降显存而通信仍同阶”的主框架。  
第二步看 DeepSpeed 配置文档，把论文里的抽象机制映射到实际开关。  
第三步看 PyTorch distributed 文档，确认 `reduce_scatter` 和 `all_gather` 的通信语义。  
第四步再看工程博文和 issue，理解理论在真实系统里会如何受 bucket、buffer、回退路径、日志口径影响。

## 核心结论

FSDP，完整名称是 FullyShardedDataParallel，直译是“完全分片的数据并行”。白话说，它不是让每张卡都完整保存一份模型，而是把模型参数拆开，分给不同 GPU 分别保管，需要计算时再临时拼起来。

它可以理解为 PyTorch 对 ZeRO-3 思路的原生实现。ZeRO 的意思是“零冗余优化器”，目标是把训练里最占显存的三类对象都去重：参数、梯度、优化器状态。FSDP 对应的是最激进的一档，也就是 ZeRO-3：这三类数据都按 rank 分片存储。rank 可以理解为分布式训练里的一个参与者，通常对应一张 GPU。

核心收益很直接：

| 方案 | 参数存储 | 梯度存储 | 优化器状态存储 | 主要通信 |
| --- | --- | --- | --- | --- |
| DDP | 每卡一整份 | 每卡一整份 | 每卡一整份 | AllReduce |
| ZeRO-1 | 每卡一整份 | 每卡一整份 | 分片 | AllReduce |
| ZeRO-2 | 每卡一整份 | 分片 | 分片 | ReduceScatter / AllGather |
| ZeRO-3 / FSDP | 分片 | 分片 | 分片 | AllGather / ReduceScatter |

对新手最重要的直觉是这一句：DDP 是“每卡都背整本书”，FSDP 是“每卡只背几页，用到时把整本借来，看完再还回去”。

因此，在 world size 为 $N$ 时，如果忽略临时缓存和激活值，一个非常粗的近似是：

$$
\text{local\_memory} \approx \frac{\text{total\_parameters}}{N}
$$

这也是为什么 FSDP 常被用于 70B 级别模型训练。模型越大，完整复制越不现实，分片带来的显存下降越关键。

---

## 问题定义与边界

FSDP 解决的是“大模型训练时，单卡显存不够”的问题，而不是“训练一定更快”的问题。

先把边界讲清楚。

训练一个模型时，显存里主要有四类东西：

| 对象 | 白话解释 | 是否由 FSDP 直接分片 |
| --- | --- | --- |
| 参数 | 模型真正学到的权重 | 是 |
| 梯度 | 反向传播算出来的更新方向 | 是 |
| 优化器状态 | 例如 Adam 的一阶、二阶动量 | 是 |
| 激活值 | 前向中间结果，反向要用 | 不是核心目标 |

FSDP 的主要价值是压缩前三类。它不直接解决激活值爆炸问题，所以真实训练里常常还要和 activation checkpointing 一起用。activation checkpointing 可以理解为“少存中间结果，反向时重新算一遍”，用计算换显存。

再看适用边界。

1. 必须是多卡场景。单卡没有“分给别人保管”的对象，FSDP没有意义。
2. 通信带宽要足够。因为参数不是一直在本地，前向和反向会频繁通信。
3. 模型越大越值。小模型通常没必要承担额外复杂度。
4. 所有 rank 必须严格参与同一套通信步骤。否则很容易卡死在 collective operation，也就是集体通信操作上。

一个简单数量级例子：假设有一个 70B 参数模型，使用 BF16 存参数，每个参数约 2 字节。只算参数本身就约 140GB。若用 4 卡 DDP，每张卡都要尝试保存 140GB，显然不可能。若用 ZeRO-3/FSDP，理想情况下每张卡只保留约四分之一，也就是约 35GB 的参数切片，再叠加梯度、优化器状态和临时通信开销，才有机会落到可训练区间。

所以 FSDP 的问题定义可以压缩成一句话：在多卡环境下，通过把参数、梯度、优化器状态都切开，换取显存可承受的大模型训练能力。

---

## 核心机制与推导

FSDP 的经典实现可以从 FlatParameter 讲起。FlatParameter 是“把多个原始参数拼成一个一维大张量”。白话说，本来模型里有很多零碎小张量，FSDP 先把它们接成一根长条，再统一切片和通信。

设模型里一组参数为：

$$
P = \text{concat}(p_1, p_2, \dots, p_k)
$$

如果 world size 是 $N$，则把它切成 $N$ 份：

$$
P \rightarrow (P_0, P_1, \dots, P_{N-1})
$$

每个 rank 只长期保存自己那一片。例如 rank $i$ 只持有 $P_i$。

前向传播前，需要把完整参数临时恢复出来，因为线性层、注意力层计算时通常需要完整权重。这个恢复过程就是 AllGather。AllGather 可以理解为“每个人把自己的那片发出来，最后每个人都拿到完整集合”。

$$
P^{full} = \text{AllGather}(P_i)
$$

前向算完后，非本地的完整参数不能一直占显存，否则就失去分片意义，因此会尽快释放：

$$
\text{Free}(P^{full})
$$

反向传播时，各 rank 基于本轮恢复出的参数参与梯度计算。等梯度算完，不是像 DDP 那样直接做 AllReduce 后每卡保留完整梯度，而是做 ReduceScatter。ReduceScatter 可以理解为“先把所有人的梯度相加，再把结果切片发回各自对应 rank”。

$$
\nabla P_i = \text{ReduceScatter}(\nabla P^{full})
$$

这一步很关键。它意味着每个 rank 最终只保留自己那一片梯度，于是优化器状态也只需要维护本地片段对应的数据。整个训练就形成了“只在计算窗口短暂恢复完整参数，其他时候都保持分片”的内存模式。

### 玩具例子

假设 `world_size = 2`，某组参数展平后得到：

$$
P = [0,1,2,3,4,5,6,7]
$$

切片后：

- rank 0 持有 $P_0 = [0,1,2,3]$
- rank 1 持有 $P_1 = [4,5,6,7]$

前向前，两张卡执行 AllGather，都临时得到：

$$
P^{full} = [0,1,2,3,4,5,6,7]
$$

这时两边都能正常做前向和反向。等梯度算出来，设完整梯度为：

$$
\nabla P^{full} = [g_0,g_1,g_2,g_3,g_4,g_5,g_6,g_7]
$$

ReduceScatter 后：

- rank 0 留下 $[g_0,g_1,g_2,g_3]$
- rank 1 留下 $[g_4,g_5,g_6,g_7]$

所以从长期占用看，每张卡只需要保存 4 个元素，而不是 8 个元素。

下面用一个可运行的 Python 小程序模拟这个过程：

```python
def chunk_tensor(tensor, world_size):
    assert len(tensor) % world_size == 0
    chunk_size = len(tensor) // world_size
    return [tensor[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

def all_gather(shards):
    full = []
    for shard in shards:
        full.extend(shard)
    return [full[:] for _ in shards]

def reduce_scatter(full_grads_per_rank, world_size):
    assert len(full_grads_per_rank) == world_size
    total_len = len(full_grads_per_rank[0])
    for grads in full_grads_per_rank:
        assert len(grads) == total_len
    reduced = [sum(values) for values in zip(*full_grads_per_rank)]
    return chunk_tensor(reduced, world_size)

# toy example
full_param = list(range(8))
shards = chunk_tensor(full_param, world_size=2)
assert shards == [[0, 1, 2, 3], [4, 5, 6, 7]]

gathered = all_gather(shards)
assert gathered[0] == full_param
assert gathered[1] == full_param

rank0_grad = [1] * 8
rank1_grad = [2] * 8
scattered = reduce_scatter([rank0_grad, rank1_grad], world_size=2)

assert scattered == [[3, 3, 3, 3], [3, 3, 3, 3]]
print("FSDP toy flow works")
```

这段代码没有真的调用 GPU 通信，但逻辑和 FSDP 的核心通信顺序是一致的。

### 为什么它等价于 ZeRO-3

ZeRO-3 的关键定义就是：参数、梯度、优化器状态都不再全量复制，而是全局分片。FSDP 的核心执行过程恰好满足这一定义，所以通常说“FSDP 是 PyTorch 原生的 ZeRO-3 实现”。

### FSDP2 的变化

FSDP2 的设计重点是不再过度依赖 FlatParameter，而转向 per-parameter 分片，也就是“尽量保留原始参数边界，每个参数单独看待”。这通常基于 DTensor。DTensor 可以理解为“带有分布式布局信息的张量”，它知道自己是按哪个维度切开的。

这带来两个变化：

1. 参数仍然分片，但原始参数结构更清晰。
2. 更容易表达混合分片策略，例如有的层按参数分片，有的层只做复制。

本质上它没有推翻 FSDP 的核心思想，仍然是“按需聚齐，算完释放，梯度再分回去”，只是内部组织方式更灵活。

---

## 代码实现

真实工程里，最常见的使用方式不是手工操作 AllGather 和 ReduceScatter，而是交给 PyTorch 的 FSDP API。

先看一个简化版思路。假设我们有一个 Transformer，每层都比较重，那么通常会先对每个 block 做分片，再对根模块做一次包裹：

```python
import torch
import torch.nn as nn

# FSDP1 style
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.fc2(torch.relu(self.fc1(x))) + x)

class MyTransformer(nn.Module):
    def __init__(self, dim=256, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

def build_fsdp_model():
    model = MyTransformer()
    fsdp_model = FSDP(model, use_orig_params=True)
    x = torch.randn(2, 16, 256)
    y = fsdp_model(x)
    assert y.shape == (2, 16, 10)
    return fsdp_model
```

如果使用 FSDP2 风格，则常见写法是对层调用 `fully_shard(layer)`，最后对根模块再调用 `fully_shard(model)`。其含义是：每一层内部形成自己的分片单元，根模块再负责整体编排。

一个真实工程例子是训练多层 Transformer。假设模型有几十层注意力块，如果把整个模型作为一个巨大分片单元，通信和内存峰值可能都不理想。更实用的做法是“按层包裹”：

```python
# pseudo code for FSDP2 style
from torch.distributed.fsdp import fully_shard

model = MyTransformer()
for layer in model.layers:
    fully_shard(layer)

fully_shard(model)
```

这种层级式分片在大模型里很常见，因为它更容易控制每层参数在前向时的聚齐和释放时机。

`use_orig_params` 是一个必须知道的配置项。它控制优化器看到的是“原始参数”还是“展平后的 FlatParameter”。

| 配置 | optimizer 看到什么 | 适合什么场景 | 代价 |
| --- | --- | --- | --- |
| `use_orig_params=False` | FlatParameter | 追求更直接的分片实现 | per-parameter 超参不方便 |
| `use_orig_params=True` | 原始参数视图 | 需要按层、按参数组设置学习率/权重衰减 | 管理更复杂，兼容性要确认 |

如果你需要给 embedding、norm、head 设置不同学习率，或者想对某些参数关闭 weight decay，那么 `use_orig_params=True` 往往更合适。否则优化器只看到一根根大 flat tensor，很难精细控制。

---

## 工程权衡与常见坑

FSDP 不是“打开就一定赚”的开关，它本质上是在显存、通信、复杂度三者之间做交换。

先看常见坑：

| 坑点 | 影响 | 规避措施 |
| --- | --- | --- |
| FlatParameter 分组过碎 | 小 tensor 太多，AllGather 次数暴涨，通信开销变重 | 按层或按块分组，避免把大量微小参数单独分片 |
| `use_orig_params=False` 却想做细粒度超参配置 | optimizer 只能看到 flat tensor，参数组难以维护 | 需要 per-parameter 超参时启用 `use_orig_params=True` |
| 分片维度不合适或不能整除 | padding 增多，通信效率下降 | 优先沿 dim-0 分片，提前检查参数形状与 world size |

### 1. 小张量过多不是好事

很多新手会以为“切得越细越省显存”。这只对静态存储成立，不对整体吞吐成立。因为每次分片单元在前向前都要 AllGather，一旦模型里有大量小层，比如很多 LayerNorm、小 MLP、adapter 碎片模块，通信调用次数会明显增加。

白话说，卡不是被“总通信量”拖慢，而是被“通信过于频繁”拖慢。

实际工程里，通常按 Transformer block 作为一个较稳定的 FSDP unit，而不是把每个小层都单独切成一个通信单元。

### 2. 显存下降不等于峰值一定安全

FSDP 降的是长期持有成本，但前向和反向窗口里仍然会暂时恢复完整参数。如果 auto wrap 策略不合理，某几个大层可能在同一时间段聚齐，导致峰值显存仍然过高。

所以实际调参时要看两种显存：

- steady-state memory，也就是长期占用
- peak memory，也就是峰值占用

FSDP 主要优化前者，但你最终是否 OOM，很多时候取决于后者。

### 3. 激活值仍然可能是主瓶颈

很多人把 FSDP 打开后发现还是爆显存，原因通常不是参数没分够，而是激活值占用过大。尤其是长序列训练，attention 的中间结果会迅速变大。

这类场景一般要组合使用：

- FSDP
- activation checkpointing
- mixed precision
- 合理 batch size / sequence length

### 4. 优化器和 checkpoint 管理更复杂

FSDP 下保存和加载 checkpoint 不能完全按单卡思维处理。因为你手里未必有完整参数。真实工程里通常要区分：

- full state dict：导出完整模型，便于推理或迁移
- sharded state dict：按分片保存，训练恢复更高效

如果恢复流程和保存格式不匹配，很容易出现加载慢、内存峰值高甚至直接失败的问题。

---

## 替代方案与适用边界

FSDP 不是唯一方案，更不是所有模型的默认最优解。

先给一个实用判断表：

| 方案 | 存储节省 | 通信压力 | 实现复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| DDP | 低 | 低到中 | 低 | 小到中型模型，先求稳定训练 |
| ZeRO-1 | 中 | 中 | 中 | 优化器状态占用明显时 |
| ZeRO-2 | 高 | 中到高 | 中到高 | 梯度和优化器状态都很重时 |
| ZeRO-3 / FSDP | 最高 | 高 | 高 | 超大模型，参数复制无法接受时 |

再看具体边界。

### 小模型场景

如果你训练的是 1B 以内模型，卡数也不多，DDP 往往已经够用。原因不是 FSDP 不能用，而是收益未必覆盖复杂度成本。你会额外处理：

- 分片通信调试
- checkpoint 格式
- 参数自动包裹策略
- 优化器参数组兼容性

如果单卡或多卡 DDP 已经能稳跑，优先保持简单通常是更好的工程决策。

### 大模型场景

如果你训练的是 30B、70B 甚至更大模型，完整复制参数在显存上根本不可行，这时 ZeRO-3/FSDP 往往不是“优化选项”，而是“能否开训的前提条件”。

一个真实工程判断方式可以很直接：

- 如果 DDP 下模型、梯度、优化器状态三者之和远超单卡显存，优先考虑 FSDP。
- 如果 DDP 能跑，只是略紧张，先尝试 mixed precision、梯度累积、checkpointing，再决定是否引入 FSDP。

### 与张量并行、流水线并行的关系

FSDP 解决的是“复制太多”的问题。张量并行解决的是“单层算子本身太大”的问题。流水线并行解决的是“层太多，想分阶段放到不同设备”的问题。

所以它们不是互斥关系，而是不同维度的切分方式。超大模型训练里，经常是多种并行手段组合使用。只是对零基础到初级工程师来说，第一步先把 FSDP 理解成“ZeRO-3 式参数分片”就够了。

---

## 参考资料

1. PyTorch FSDP API 文档：`FullyShardedDataParallel`  
   https://docs.pytorch.org/docs/stable/fsdp.html

2. PyTorch 官方 FSDP 教程：包含 `fully_shard(model)` 等示例  
   https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

3. Meta / Facebook Engineering 对 FSDP 的工程介绍  
   https://engineering.fb.com/2021/07/15/open-source/fsdp/

4. PyTorch dev-discuss 中关于 FSDP2 设计重构的讨论  
   https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019

5. FSDP 与 FlatParameter、AllGather、ReduceScatter 的中文数值推演示例  
   https://www.cnblogs.com/fariver/p/18916961

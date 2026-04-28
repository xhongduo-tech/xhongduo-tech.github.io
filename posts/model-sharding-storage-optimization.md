## 核心结论

模型分片存储的目标，不是单纯把一个大文件切成很多小文件，而是把模型权重或检查点组织成“索引文件 + 多个 shard 文件”的结构。这样做的直接收益有三类：

1. 降低单文件读写瓶颈。单个超大文件在网络盘、对象存储、慢盘环境下，容易卡在一次长时间顺序读取上。
2. 支持并行加载。多个 worker 可以同时读取不同 shard，缩短冷启动和恢复时间。
3. 改善失败恢复与迁移。某个 shard 传输失败时，只需要重传该 shard，不必重传整个 checkpoint。

但分片不是越细越好。设总权重大小为 $W$，单分片大小上限为 $s$，则分片数为：

$$
n = \lceil W / s \rceil
$$

如果每个 shard 都要付出一次固定开销，例如打开文件、解析头部、定位偏移，记为 $L$；持续带宽记为 $B$；并行 worker 数记为 $p$，则可用一个简化模型理解加载时间：

$$
T_{serial} \approx n \cdot L + W / B
$$

$$
T_{parallel} \approx \lceil n / p \rceil \cdot L + W / B
$$

这里 $W / B$ 是“纯搬数据”的时间下界，通常很难消除；真正能通过分片策略优化的，是固定开销 $L$ 和有效并行度 $p$。

一个直观例子：16 GiB 权重拆成 4 个 4 GiB shard，通常比拆成 16 个 1 GiB shard 更均衡。前者允许并行读取，同时文件数还不算多；后者虽然更细，但元数据、文件打开次数、随机 IO 和尾延迟都更重。

| 分片策略 | 优点 | 风险 |
|---|---|---|
| 更粗 | 文件少，元数据少，顺序读更友好 | 并行度低，失败重试成本高 |
| 更细 | 并行度高，迁移更灵活 | 文件数多，随机 IO 和管理成本高 |

---

## 问题定义与边界

先把术语说清楚。

“权重”是模型学到的参数，白话说，就是神经网络真正拿来计算的数字。“检查点”是模型在某个时刻的可恢复快照，通常不只含权重，还可能包含优化器状态、随机数状态、训练步数等。“shard”是分片文件，也就是拆出来的每个实际文件。“索引文件”或 manifest，是一张映射表，告诉程序每个张量存在哪个 shard、位于什么偏移、长度是多少。“断点恢复”是训练中断后从最近一次保存状态继续跑。“重分片”是加载时按新的设备数或布局重新组织 checkpoint，而不是照搬原始分片布局。

下面这张表可以先建立词汇边界：

| 术语 | 白话解释 | 作用 |
|---|---|---|
| 权重 | 模型参数本身 | 决定推理结果 |
| 检查点 | 可恢复的训练/部署状态 | 用于恢复与迁移 |
| shard | 拆出来的单个文件 | 降低单文件瓶颈 |
| 索引文件 | 参数到 shard 的目录表 | 快速定位张量 |
| 断点恢复 | 从中断处继续 | 缩短故障恢复时间 |
| 重分片 | 按新布局重新映射分片 | 适配扩缩容和换卡 |

本文讨论的是“模型权重/检查点如何存储与加载”。这和训练并行不是一回事。参数并行、张量并行、流水并行，是把计算任务分到多设备上；模型分片存储，是把文件和数据组织方式做优化。两者经常同时出现，但解决的是不同问题。

对新手最有帮助的直觉是：一个 16 GiB 的大文件，改成 4 个 4 GiB 文件，再加一份索引，程序就知道某个参数在哪个文件里。这和“把书拆成四册，再额外做目录页”很像。目录不是内容本身，但没有目录，程序只能把整个文件从头扫到尾。

---

## 核心机制与推导

加载流程的本质是三步：先查索引，再定位 shard，最后按偏移读取 tensor。这里的 tensor 是“张量”，白话说，就是带形状的多维数组，模型的权重本质上就是很多张量。

一个玩具例子如下。假设索引里有一条记录：

- `layer.0.attn.q_proj.weight -> shard-02, offset=1048576, length=8388608`

程序启动时不会把所有 shard 全部读进来再慢慢找，而是：

| 步骤 | 输入 | 输出 | 主要瓶颈 |
|---|---|---|---|
| 读取索引 | index/manifest | 参数到文件位置映射 | 小文件解析 |
| 定位 shard | 参数名 | shard 文件名、偏移、长度 | 元数据查询 |
| 读取数据 | shard + offset + length | 张量字节流 | 磁盘或网络带宽 |
| 反序列化/装载 | 字节流 | 内存中的 tensor | CPU 与内存拷贝 |

这就是“先看目录，再去对应抽屉取东西”，而不是“把所有抽屉都拉开再找”。

为什么分片能加速？因为单个大文件常把所有工作压在一条 IO 路径上。分片后，多个 worker 可以同时读取不同文件，理论上把固定开销和等待时间分摊掉。上面的公式里，$n \cdot L$ 表示“每个文件都要付一次的成本”。只要 $n$ 增大，哪怕总数据量 $W$ 不变，这一项也会上升。

继续用数值例子。假设：

- $W = 16 \text{ GiB}$
- $L = 0.2 \text{ s}$
- $B = 2 \text{ GiB/s}$

当 $s = 4 \text{ GiB}$ 时：

- $n = \lceil 16 / 4 \rceil = 4$
- $T_{serial} \approx 4 \times 0.2 + 16 / 2 = 8.8 \text{ s}$

当 $s = 1 \text{ GiB}$ 时：

- $n = 16$
- $T_{serial} \approx 16 \times 0.2 + 8 = 11.2 \text{ s}$

总数据量没有变，但文件数多了，固定成本被放大了。

如果有并行度 $p = 4$，粗略估计：

- 4 个 4 GiB shard：$\lceil 4/4 \rceil \cdot 0.2 + 8 = 8.2 \text{ s}$
- 16 个 1 GiB shard：$\lceil 16/4 \rceil \cdot 0.2 + 8 = 8.8 \text{ s}$

这里已经能看到一个常见误区：分片变细后，并行加载确实可能更好，但收益会被固定开销吃掉。尤其在对象存储、NFS、Ceph 这类网络存储上，`open`、认证、元数据查询、随机读放大都可能让 $L$ 明显增加。

真实工程例子更能说明问题。70B 级模型在线推理冷启动时，模型常放在网络盘或对象存储里。此时真正决定服务恢复时间的，不只是总大小，还包括：

- 是否能让多个 rank 并行拉取不同 shard
- 是否支持只读自己负责的张量
- 节点数变化后能否在加载时重分片

如果 shard 过碎，系统可能花大量时间在列目录、打开大量对象、建立网络连接和处理尾延迟上。结果是：理论并行度提高了，实际恢复时间反而更长。

---

## 代码实现

实现层面可以拆成四步：保存时切 shard、生成索引、加载时按索引读取、需要时重分片。不同库的区别，主要就在这四步谁负责、支持到什么粒度。

先给一个最小可运行的 Python 玩具实现。它不依赖深度学习框架，只模拟“按大小切分”和“估算加载时间”的逻辑。

```python
import math

def shard_plan(total_gib: float, shard_gib: float):
    assert total_gib > 0
    assert shard_gib > 0
    n = math.ceil(total_gib / shard_gib)
    sizes = [shard_gib] * n
    sizes[-1] = total_gib - shard_gib * (n - 1)
    return n, sizes

def estimate_times(total_gib: float, shard_gib: float, latency_s: float, bandwidth_gib_s: float, workers: int):
    n, _ = shard_plan(total_gib, shard_gib)
    t_serial = n * latency_s + total_gib / bandwidth_gib_s
    t_parallel = math.ceil(n / workers) * latency_s + total_gib / bandwidth_gib_s
    return n, round(t_serial, 2), round(t_parallel, 2)

n1, t1s, t1p = estimate_times(16, 4, 0.2, 2, 4)
n2, t2s, t2p = estimate_times(16, 1, 0.2, 2, 4)

assert n1 == 4
assert n2 == 16
assert t1s == 8.8
assert t2s == 11.2
assert t1p < t2p

print((n1, t1s, t1p))
print((n2, t2s, t2p))
```

这个例子表达了两个核心事实：

1. 分片数由 $n = \lceil W / s \rceil$ 决定。
2. 在同样数据量下，过细分片会抬高固定成本。

如果把它映射到真实库，可以理解成下面这套伪代码流程：

```python
# 保存
for tensor_name, tensor_bytes in state_dict.items():
    place_tensor_into_current_shard()
    if current_shard_size > max_shard_size:
        flush_shard()

write_index_file(tensor_name -> {shard, offset, length})

# 加载
index = read_index_file()
for needed_tensor in required_tensors:
    shard, offset, length = index[needed_tensor]
    bytes_ = read_range(shard, offset, length)
    tensor = decode(bytes_)

# 重分片（分布式场景）
for target_rank in world_size:
    remap_needed_tensors_to_rank(target_rank)
```

不同方案的能力边界可以这样看：

| 方案 | 主要能力 | 适用场景 |
|---|---|---|
| Transformers 分片 checkpoint | 按 `max_shard_size` 切分并生成索引 | 单机或常规多机模型保存/加载 |
| `safetensors` | 更安全的权重格式，可按张量或切片读取 | 推理权重分发、避免 pickle 风险 |
| PyTorch DCP | 分布式保存、并行加载、加载时重分片 | 大规模训练恢复、弹性扩缩容 |

这里要特别区分两个层次：

- Transformers 的分片更偏“把 checkpoint 拆开并记录索引”。
- `safetensors` 更偏“安全格式 + 可高效定位读取”。
- PyTorch Distributed Checkpoint 更偏“面向分布式恢复和重分片”。

如果你只是做一个中小模型的本地部署，`safetensors` 单文件或少量分片就可能够用。如果你要处理多机训练恢复或 GPU 数变化，单纯切文件还不够，通常需要 DCP 这类支持 load-time resharding 的方案。

---

## 工程权衡与常见坑

工程里最常见的坑，不是“不会分片”，而是“以为分片天然更快”。

第一类坑是分片过细。文件数一多，inode 消耗、manifest 体积、`open/close` 次数、随机读取和尾延迟都会抬升。对象存储环境下，这种问题更明显，因为每个对象访问背后可能都伴随额外网络往返和认证逻辑。

第二类坑是大 tensor 本身不可拆。如果某个 embedding 或投影矩阵特别大，它可能单独就逼近 shard 上限。此时即使你把 `max_shard_size` 设得很小，也未必能得到理想的均匀分布。

第三类坑是继续使用 pickle `.bin`。pickle 是 Python 对象序列化机制，白话说，它更像“把内存对象原样打包”，不是专门为安全高效权重读取设计的格式。它有安全风险，也不适合高效做精细读取。

第四类坑是没有重分片能力。比如训练时是 8 卡保存，恢复时改成 16 卡，如果 checkpoint 只能按原布局硬读，恢复时间会明显变长。

下面这张表更适合工程排障：

| 症状 | 常见原因 | 处理方式 |
|---|---|---|
| 分片很多但启动更慢 | 分片过细，固定开销过大 | 提大 shard 上限，减少文件数 |
| 某些 shard 特别大 | 单个大 tensor 主导布局 | 先检查最大张量尺寸，接受不完全均匀 |
| 网络盘读取抖动大 | 随机 IO 和尾延迟放大 | 优先减少碎片化，增加顺序读比例 |
| 加载流程复杂且有风险 | 继续用 pickle `.bin` | 优先迁移到 `safetensors` |
| 扩缩容后恢复很慢 | 无法加载时重分片 | 采用支持 resharding 的 DCP |

真实工程里，一个典型场景是在线推理服务重启。70B 模型放在网络存储上，如果拆成几十甚至上百个小 shard，理论上每个 worker 都有事可做，但实际启动时会被大量小请求拖慢。此时更合理的策略往往是：先选一个中等 shard 粒度，再结合预热、缓存、并发读取和节点本地缓存一起优化。

经验上，分片设计不是独立参数，而是和以下条件共同决定：

- 底层存储是本地 NVMe、网络盘还是对象存储
- 单机并发数和 rank 数
- checkpoint 恢复是否频繁
- 是否存在弹性扩缩容
- 最大 tensor 是否极端不均匀

---

## 替代方案与适用边界

模型分片存储不是唯一方案，也不能替代训练并行方案。不同方案解决的问题不同，选错层次会导致系统复杂但收益不大。

| 方案 | 优点 | 缺点 | 适用场景 | 不适用场景 |
|---|---|---|---|---|
| 单文件 checkpoint | 最简单，易管理 | 单文件瓶颈明显，失败重传成本高 | 小模型、本地快速实验 | 大模型冷启动、多机恢复 |
| 单文件 `safetensors` | 安全，读取效率较好 | 并行加载空间有限 | 中小模型推理部署 | 大规模分布式恢复 |
| `safetensors` 分片 | 安全，支持按 shard/张量定位 | 文件数上升，需要权衡粒度 | 大模型推理、网络存储环境 | 极小模型或无需并行加载 |
| PyTorch DCP | 支持分布式保存与加载时重分片 | 方案更复杂，接入成本更高 | 多机训练恢复、弹性扩缩容 | 简单单机部署 |
| ZeRO | 降低训练显存占用 | 主要解决训练内存问题，不等于存储分片 | 超大模型训练 | 只想优化推理冷启动 |

ZeRO 是训练内存优化方案，核心是把优化器状态、梯度和参数在多个设备之间分摊，不是“磁盘上的 checkpoint 应该如何切文件”的直接替代。DCP 则更接近存储与恢复路径上的系统能力。两者可能一起用，但不要把它们看成同一层方案。

可以用一个简单决策原则：

- 模型不大、本地盘够快、恢复频率低：单文件 `safetensors` 往往足够。
- 模型较大、推理冷启动敏感：优先考虑分片存储，控制 shard 粒度。
- 多机训练、节点数经常变化：优先考虑支持 load-time resharding 的 DCP。
- 训练显存本身扛不住：再考虑 ZeRO、MiCS 这类训练并行优化。

结论不是“所有模型都该分片”，而是“当单文件成为瓶颈时，分片是更合理的组织方式；当恢复布局经常变化时，仅有分片还不够，还需要重分片能力”。

---

## 参考资料

| 来源 | 可支撑的结论 | 建议引用位置 | 备注 |
|---|---|---|---|
| Transformers 文档 | `max_shard_size`、分片保存与加载接口 | 代码实现、核心机制 | 适合说明传统分片 checkpoint |
| `safetensors` 文档 | 安全格式、按张量/切片读取 | 核心机制、工程权衡 | 适合说明替代 pickle 的理由 |
| PyTorch DCP 文档 | 分布式 checkpoint 与重分片 | 代码实现、替代方案 | 适合说明弹性恢复能力 |
| DeepSpeed ZeRO 文档 | 训练内存优化边界 | 替代方案 | 用于区分训练并行与存储分片 |
| ZeRO 论文 | 机制背景与设计目标 | 替代方案 | 更偏理论背景 |

1. [Transformers Model API: sharded checkpoints and `max_shard_size`](https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/model)
2. [Hugging Face safetensors Documentation](https://huggingface.co/docs/safetensors/index)
3. [PyTorch Distributed Checkpoint Documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
4. [DeepSpeed ZeRO-3 Documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
5. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)

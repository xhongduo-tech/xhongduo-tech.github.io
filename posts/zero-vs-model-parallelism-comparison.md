## 核心结论

ZeRO 与模型并行解决的是同一个根问题：单卡放不下大模型。区别在于它们拆分对象不同、通信时机不同、对网络的要求也不同。

先给结论：

| 方案 | 主要拆什么 | 开发改造量 | 通信位置 | 更适合的硬件环境 | 典型优势 | 典型短板 |
| --- | --- | --- | --- | --- | --- | --- |
| 数据并行（DP） | 不拆模型，只复制多份 | 低 | 反向阶段集中同步梯度 | 跨机可用 | 实现最简单 | 参数、梯度、优化器状态全部冗余 |
| ZeRO-1/2 | 拆优化器状态、梯度 | 低到中 | 仍以 DP 方式同步 | 跨机可用 | 显存下降明显 | 参数仍可能是瓶颈 |
| ZeRO-3 | 连参数也拆分 | 低到中 | 前向/反向按需 `AllGather`，反向后 `ReduceScatter` | 跨机尤其合适 | 不改模型结构也能显著扩展模型规模 | 通信次数增多 |
| Tensor Parallel，TP | 把单层算子切到多卡 | 高 | 几乎每层都通信 | 机内 NVLink/NVSwitch | 单层超大矩阵可直接拆 | 跨机性能很容易崩 |
| Pipeline Parallel，PP | 按层分段 | 中到高 | 相邻 stage 传激活值 | 机内外都可，但要管流水线气泡 | 能继续扩模型深度 | 调度复杂，吞吐受微批大小影响 |

ZeRO-3 最重要的认识是：它在数学效果上接近“把模型切开来放”，但对使用者来说仍然像数据并行。白话讲，数学等价指最终每张卡只负责自己那一片参数状态；像数据并行指训练脚本不用手动把每一层改写成分布式版本。它把“内存分片”这件事从模型代码中抽出来，交给运行时系统处理。

因此，工程上常见的最优组合不是“ZeRO 和模型并行二选一”，而是“机内 TP，跨机 ZeRO-3”。机内网络快，适合高频细粒度通信；跨机网络慢，适合只做 `AllGather` 和 `ReduceScatter` 这类更规则的集合通信。

---

## 问题定义与边界

先把问题说清楚。训练一个参数量为 $P$ 的模型时，显存通常至少要装下三类东西：

$$
\text{memory} \approx \text{params} + \text{grads} + \text{optimizer states} + \text{activations}
$$

其中：

- 参数：模型权重，也就是网络真正学到的数值。
- 梯度：反向传播算出来、用于更新参数的量。
- 优化器状态：例如 Adam 会额外保存一阶矩和二阶矩，可以理解为“帮助更新更稳定的历史统计量”。

如果用标准数据并行，$N$ 张卡上的每一张都要保存完整参数、完整梯度、完整优化器状态。模型越大，冗余越严重。7B、13B、70B 这类模型一上来，问题通常不是算不动，而是装不下。

这里要区分两个边界：

| 维度 | ZeRO 更关注 | 模型并行更关注 |
| --- | --- | --- |
| 核心目标 | 去掉状态冗余 | 拆开单层或分层计算 |
| 是否需要重写模型结构 | 通常不需要 | 通常需要或依赖特定框架 |
| 跨机适应性 | 较好 | TP 跨机通常较差 |
| 对高速互联依赖 | 相对较低 | 很高，尤其 TP |
| 主要瓶颈 | 集合通信次数和带宽 | 每层通信延迟和带宽 |

玩具例子先看 4 张 GPU。

假设模型总参数是 400MB，梯度 400MB，优化器状态 800MB。  
如果是普通 DP，每张卡都要放：

$$
400 + 400 + 800 = 1600\text{MB}
$$

如果是 ZeRO-3，4 张卡把这三类状态都均分，每张卡静态只保留约：

$$
\frac{1600}{4} = 400\text{MB}
$$

这就是它最直观的价值：不是把单层算子拆碎，而是把“本来每卡都重复保存的东西”去掉。

真实工程例子更能说明边界。训练 70B 模型时，若集群是 8 个节点、每节点 8 张 GPU，常见做法不是单独依赖某一种并行，而是：

- 节点内做 TP=8，利用 NVLink/NVSwitch。
- 节点间做 PP 或 ZeRO-DP。
- 若还需要进一步省显存，则用 ZeRO-3 继续拆参数状态。

原因很简单：TP 在每层都要通信，跨节点走 100GbE 或普通 RoCE 时，延迟和带宽都不够；ZeRO-3 的通信虽然更多，但模式规则，跨机更容易跑稳。

---

## 核心机制与推导

ZeRO 分三阶段理解最清楚：

| Stage | 分片对象 | 显存收益 | 通信变化 |
| --- | --- | --- | --- |
| ZeRO-1 | 优化器状态 | 先解决 Adam 状态太大 | 接近 DP |
| ZeRO-2 | 优化器状态 + 梯度 | 进一步下降 | 常见实现把梯度同步做成 `ReduceScatter` |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 收益最大 | 前后向都要按需取参数 |

ZeRO-3 的关键动作只有两个：

- `AllGather`：把多卡上分散的参数片段临时拼成当前层需要的完整参数。
- `ReduceScatter`：把各卡产生的梯度先归约，再按分片规则分回各卡。

白话讲，`AllGather` 像“把大家手里的拼图临时拼完整再计算”，`ReduceScatter` 像“把大家算出来的结果先合并，再按责任区发回去”。

为什么说它“操作像模型并行，调度像数据并行”？  
因为从存储视角看，每张卡只持有一部分参数，这和模型分片很像；但从执行组织看，所有卡仍然跑同样的前向和反向，只是在层执行前临时拿到完整权重，这又是数据并行式的调度。

通信量可以粗略写成：

$$
T_{\text{comm}} \approx n_{\text{transmissions}} \times n_{\text{bytes}} \times \frac{\text{model\_size}}{\text{bandwidth}}
$$

其中：

- $n_{\text{transmissions}}$：每步发生几次大规模传输。
- $n_{\text{bytes}}$：每个参数占多少字节，例如 bf16 是 2 字节。
- $\text{model\_size}$：参与通信的参数规模。
- $\text{bandwidth}$：实际可用带宽，不是理论峰值。

对比直觉如下：

- DDP：反向阶段主要是一轮梯度 `AllReduce`。
- ZeRO-3：前向前 `AllGather` 参数，反向前或反向中再次 `AllGather`，反向后 `ReduceScatter` 梯度。

所以它常被概括为通信量约为 DDP 的 1.5 倍左右。这个数字不是“永远固定”，而是一个常见估算，用来帮助判断趋势：显存大幅下降，通信代价上升。

再做一个玩具例子。  
设模型有 10B 参数，采用 bf16，每个参数 2 字节，则完整参数量约为：

$$
10^ {10} \times 2 = 20\text{GB}
$$

粗略估算：

- DDP 一次主要梯度同步，量级约 40GB。
- ZeRO-3 三次主要集合通信，量级约 60GB。

代价确实更高，但换来的结果是：参数、梯度、优化器状态都不再在每卡完整复制。对超大模型，这通常是可训练与不可训练的分界线，而不是“小优化”。

---

## 代码实现

先给一个可以运行的 Python 玩具实现。它不依赖 GPU，也不依赖 DeepSpeed，只用数组模拟“参数分片 + AllGather + ReduceScatter”的逻辑，目的是把机制看明白。

```python
from math import ceil

def shard_tensor(tensor, world_size):
    shard_size = ceil(len(tensor) / world_size)
    shards = []
    for i in range(world_size):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(tensor))
        shard = tensor[start:end]
        if len(shard) < shard_size:
            shard = shard + [0.0] * (shard_size - len(shard))
        shards.append(shard)
    return shards

def all_gather(shards, original_len):
    full = []
    for shard in shards:
        full.extend(shard)
    return full[:original_len]

def reduce_scatter(full_grads, world_size):
    # 模拟先求和后切分。这里假设每卡产生的梯度相同，直接按切分返回。
    shard_size = ceil(len(full_grads) / world_size)
    out = []
    for i in range(world_size):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(full_grads))
        shard = full_grads[start:end]
        if len(shard) < shard_size:
            shard = shard + [0.0] * (shard_size - len(shard))
        out.append(shard)
    return out

# 4 卡分片一个 10 维参数
params = [float(i) for i in range(10)]
world_size = 4

param_shards = shard_tensor(params, world_size)
restored = all_gather(param_shards, original_len=len(params))
assert restored == params

# 假设反向得到完整梯度
grads = [1.0] * 10
grad_shards = reduce_scatter(grads, world_size)

# 每卡只保留自己负责的梯度分片
assert len(grad_shards) == 4
assert sum(sum(s) for s in grad_shards) == 10.0

print("ZeRO-3 toy flow works.")
```

上面这段代码表达的是 ZeRO-3 的核心运行时思想：

1. 初始化时，每卡只保存参数分片。
2. 计算某层前，临时 `AllGather` 完整参数。
3. 反向结束后，用 `ReduceScatter` 只留下本卡负责的梯度分片。
4. 优化器也只更新本卡拥有的那部分状态。

真实工程里不会手写这些集合通信，而是直接让框架接管。DeepSpeed 的入口通常像这样：

```python
import deepspeed
import torch
import torch.nn as nn

class MyHugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

with deepspeed.zero.Init(
    remote_device="cpu",
    enabled=True,
    dtype=torch.float16,
):
    model = MyHugeModel()
```

这段代码的意义不是“只是换个初始化器”，而是告诉运行时：模型创建时就按 ZeRO-3 的规则管理参数，不要先在每卡生成一整份再去切。对超大模型，这一点非常关键，因为很多模型甚至在“构造阶段”就会 OOM。

真实工程例子通常会进一步配合这些配置思路：

- `stage: 3`：开启 ZeRO-3。
- `overlap_comm: true`：让通信和计算尽量重叠。
- `contiguous_gradients: true`：降低梯度碎片化带来的额外开销。
- `allgather_partitions`、`reduce_scatter`：控制分片同步方式。
- 参数预取或分层抓取策略：减少瞬时峰值显存。

---

## 工程权衡与常见坑

第一个权衡是网络域。  
TP 的通信发生在层内，频率高、消息细碎、对延迟极度敏感，所以通常只放在机内高速互联域。ZeRO-3 的集合通信虽然数据量不小，但模式规则、框架支持成熟，更适合跨节点。

第二个权衡是“能不能装下”和“跑得够不够快”不是一回事。  
ZeRO-3 往往先解决前者，再努力优化后者。如果网络很慢，模型虽然能训起来，但吞吐可能明显下降。这时要做的不是简单关掉 ZeRO，而是看：

| 常见问题 | 表现 | 原因 | 处理思路 |
| --- | --- | --- | --- |
| 跨机 TP 过大 | GPU 利用率低，通信等待长 | 每层都在慢网络同步 | 把 TP 缩回单机内 |
| ZeRO-3 吞吐偏低 | step time 被通信拉长 | `AllGather`/`ReduceScatter` 无法充分重叠 | 开启 overlap、预取、层次化分片 |
| 初始化就 OOM | 模型还没训练就爆显存 | 先完整构模再分片 | 用 `zero.Init` 在构造时直接分片 |
| 小模型反而变慢 | 显存省了但速度没收益 | 通信管理开销超过收益 | 小模型优先 DDP 或 ZeRO-1/2 |
| 参数抓取过细 | CPU 端或 host 端开销升高 | 模块粒度太碎 | 调整模块粒度阈值 |

一个常见误区是“ZeRO-3 能省显存，所以任何场景都该开到 Stage 3”。这不成立。  
如果模型本来就能舒服地装进单卡，或者训练主要瓶颈是网络而不是显存，那么 ZeRO-3 额外引入的参数抓取和集合通信可能不划算。ZeRO-1 或 ZeRO-2 往往更平衡。

另一个常见坑是把“数学等价”理解成“性能等价”。  
ZeRO-3 在结果上可以达到类似模型分片的效果，但性能不自动等价。TP 利用的是机内超高速互联，可以把单层大矩阵乘法直接拆开并行；ZeRO-3 利用的是状态分片和按需恢复。二者解决的问题相交，但不是同一种执行路径。

---

## 替代方案与适用边界

实践里常见的不是单一方案，而是组合方案。可以按下面的经验判断：

| 场景 | 更合适的方案 |
| --- | --- |
| 模型能放下，但想提吞吐 | 先 DDP，再考虑少量 TP |
| 模型放不下，且主要是状态冗余太大 | 先 ZeRO-2/3 |
| 单层矩阵太大，单卡算子都放不下 | 必须引入 TP |
| 模型层数很多，单机放不下整网 | 引入 PP |
| 多节点大模型训练 | 机内 TP，跨机 ZeRO-3/DP，必要时再叠加 PP |

3D 并行就是把这些能力叠加起来：

- TP 解决单层拆分。
- PP 解决分层部署。
- DP 或 ZeRO 解决副本同步与状态分片。

真实工程例子可以写成一个典型配比：  
70B 模型，256 张 A100，常见思路是 TP=8、PP=8、DP=4，并让 DP 维度使用 ZeRO-3。这样单节点内靠 NVLink 处理高频层内通信，节点间只承担更规则的参数同步与流水线数据流。

替代方案方面，还有 ZeRO++、分层分片、FSDP 这类路线。它们和 ZeRO-3 的关系不是“推翻”，而是继续优化跨节点通信。例如保留机内副本、只在跨机时进一步压缩或层次化同步，可以在网络受限时明显改善吞吐。

适用边界可以浓缩成一句话：

- 如果问题是“状态冗余太大”，优先 ZeRO。
- 如果问题是“单层算子本身太大”，必须 TP。
- 如果问题是“整个网络太深太长”，需要 PP。
- 如果问题三者同时存在，就做 3D 并行，并把 TP 尽量限制在高速互联域。

---

## 参考资料

- DeepSpeed ZeRO-3 官方文档：https://deepspeed.readthedocs.io/en/stable/zero3.html
- DeepSpeed ZeRO 教程与 `zero.Init` 用法：https://www.deepspeed.ai/tutorials/zero/
- JAX Scaling Book 关于 ZeRO / AllGather / ReduceScatter 的解释：https://jax-ml.github.io/scaling-book/training/
- Saforem2 关于模型并行与通信时间估算：https://saforem2.github.io/ml-engineering/qmd/training/model-parallelism/index.html
- SystemOverflow 关于 DP / TP / PP / ZeRO 的工程边界分析：https://www.systemoverflow.com/learn/ml-nlp-systems/nlp-scalability/how-do-you-choose-between-data-parallelism-tensor-parallelism-and-pipeline-parallelism
- Megatron-LM 相关并行训练论文与实践资料：https://cs.stanford.edu/~deepakn/assets/papers/megatron-sc21.pdf
- 分布式训练综述与 ZeRO++ 讨论：https://www.preprints.org/manuscript/202512.2207/v1
- ZeRO 通信量公式相关论文索引：https://par.nsf.gov/servlets/purl/10540132

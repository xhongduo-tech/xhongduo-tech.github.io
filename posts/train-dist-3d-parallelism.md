## 核心结论

3D 并行指的是把一次训练同时沿三条轴切开：数据并行（Data Parallelism, DP，白话说是“多份相同模型各自处理不同样本”）、张量并行（Tensor Parallelism, TP，白话说是“把一个很大的层拆到多张卡一起算”）、流水线并行（Pipeline Parallelism, PP，白话说是“把不同层分给不同卡，像装配线一样接力”）。它成立的原因很直接：大模型训练同时受三类瓶颈约束，参数放不下、层太深导致激活爆炸、吞吐又不够，单一并行策略只能解决其中一部分。

核心公式先记住两条：

$$
N = D \times T \times P
$$

这里 $N$ 是总设备数，$D,T,P$ 分别是 DP、TP、PP 的并行度。

$$
M_{3D} \approx \frac{F}{T \cdot P} + A \cdot \frac{B}{D}
$$

这是一个便于理解的近似写法：参数内存主要受 TP 与 PP 共同分摊，激活内存主要和每卡承担的微批有关。更完整的记号常写成 $M_{3D}=P_1P_3F + AP_2B$，本质表达的是同一件事：参数、激活、微批在三条轴上的分摊方式不同。

为什么 3D 并行常说“最好接近平衡”，也就是 $D \approx T \approx P$？因为通信和内存都不是单调只看一轴。TP 太大，层内同步过于频繁；PP 太大，流水线气泡（pipeline bubble，白话说是“有些卡在等，没活干”）变严重；DP 太大，梯度同步压力线性上升。三轴平衡的目标不是数学上的对称好看，而是避免把系统推到单一瓶颈上。

一个最小玩具例子：8 张 GPU 可以组织成 DP=2、TP=2、PP=2。结果是：
- 模型被切成 2 个流水线阶段，每张卡只放一半层。
- 每个大层再按 2 路 TP 拆开，每张卡只算一半张量。
- 整个系统再做 2 份副本，各自吃不同数据，吞吐翻倍。

这时单卡不再承担完整模型，但整体吞吐仍能靠 DP 扩大。

真实工程里，175B 这类模型常采用类似 TP=8、PP=8、DP=8 的结构，把 512 张卡组织成规则网格，再配合 ZeRO 进一步压缩优化器状态和梯度内存，才可能在 80GB A100 这一级别的显存上跑通。

---

## 问题定义与边界

3D 并行要解决的问题不是“如何多用几张卡”，而是“当单卡显存、单节点带宽、单策略扩展性同时不够时，怎么把训练继续推进”。

单独看三种常见策略：

| 并行轴 | 并行对象 | 主要收益 | 主要通信模式 | 单独使用的核心限制 |
| --- | --- | --- | --- | --- |
| DP | mini-batch | 提高吞吐 | AllReduce | 模型仍需完整放入每张卡 |
| TP | 权重张量/矩阵乘 | 拆宽层，缓解单层过大 | AllReduce / AllGather | 高频通信极重，强依赖高速互连 |
| PP | 网络层/阶段 | 拆深度，缓解层数和激活压力 | 点对点传激活 | 需要足够微批，否则有气泡 |

所以边界很清楚：
- 如果模型本身能放进单卡，且主要问题只是吞吐，DP 或 FSDP 往往更简单。
- 如果问题是单层太宽，比如超大 attention 或 MLP，必须引入 TP。
- 如果问题是层数深、激活大、重计算后仍放不下，就需要 PP。
- 当“宽、深、吞吐”三者一起成为约束时，才真正进入 3D 并行的适用区间。

新手可以用“三种交流频率”理解边界：
- DP 像“每天对账一次”，可以跨更远的网络。
- PP 像“每小时交接一次工单”，最好在相邻节点间。
- TP 像“每分钟一起抬同一块钢板”，只能放在极快的链路上。

这就是为什么拓扑感知映射（topology-aware mapping，白话说是“让通信最频繁的组靠得最近”）是 3D 并行成败的前提。TP 组通常要尽量落在同一台 NVSwitch 机器内；PP 可以跨节点但要控制激活传输；DP 最能容忍慢链路，因为它的同步频次相对低。

---

## 核心机制与推导

3D 并行的核心不是“把设备分三组”这么简单，而是把三类资源分别交给三条轴处理：

- TP 负责切参数宽度，解决“大层放不下”。
- PP 负责切网络深度，解决“整网太深、激活太大”。
- DP 负责复制训练副本，解决“吞吐不足”。

先看内存。设模型参数总量为 $F$，单样本激活规模为 $A$，全局 batch 为 $B$。直观上：

$$
\text{单卡参数内存} \propto \frac{F}{T \cdot P}
$$

因为 TP 把层内参数横切，PP 把层按阶段纵切。再看激活：

$$
\text{单卡激活内存} \propto A \cdot \frac{B}{D}
$$

因为 DP 把样本分给不同副本，每张卡只看到局部微批。合起来得到前面的近似：

$$
M_{3D} \approx \frac{F}{T \cdot P} + A \cdot \frac{B}{D}
$$

这说明三条轴在“减什么”上并不一样。TP、PP 更偏向减参数与层深压力，DP 更偏向分担样本和吞吐。

再看通信。可以粗略拆成三项：

$$
C_{\text{total}} = C_{\text{TP}} + C_{\text{PP}} + C_{\text{DP}}
$$

其中常见量级可以写成：

$$
C_{\text{TP}} = \mathcal{O}(P_3 P_1^2 F)
$$

$$
C_{\text{DP}} = \mathcal{O}(P_1 P_3 F)
$$

$$
C_{\text{PP}} = \mathcal{O}(A P_2 B)
$$

这些记号的重点不是死背符号，而是理解来源：
- TP 通信来自层内切分后必须重新拼结果或同步梯度，所以频率高。
- DP 通信来自副本间梯度归并，所以规模随参数量增长。
- PP 通信来自相邻 stage 之间传激活和反向梯度，所以和激活大小、微批数直接相关。

为什么常说 AM-GM 平衡会把最优点推向 $D \approx T \approx P$？因为如果总设备数 $N$ 固定，只把某一轴拉得很大，其他两轴太小，就会让某一种通信或内存项单独爆掉。均衡分摊能让最大瓶颈项下降。这不是严格总成立的闭式定理，而是工程上很稳定的启发式：先找近似平衡点，再按拓扑和模型结构微调。

一个玩具例子：512 张卡，设 TP=8、PP=8、DP=8。
- 每张卡只负责约 $1/8$ 的层内张量。
- 每张卡只负责约 $1/8$ 的网络深度。
- 每个 DP 副本只处理全局 batch 的 $1/8$。

这时 175B 模型不要求单卡容纳全部参数，也不要求单卡缓存整条网络的激活。再结合 ZeRO 把优化器状态进一步分片，系统才进入“可以训练”的区域。

真实工程例子可以看 OPT-175B 一类训练系统：它们并不是单靠更多 GPU 粗暴堆起来，而是把高频 TP 通信锁在节点内，把 PP 用于节点间分层传递，再用 DP 扩大有效 batch 和吞吐。能跑起来，靠的是三类通信被放到了不同成本的链路上。

---

## 代码实现

下面给一个“可运行但抽象化”的 Python 示例。它不真的发起 NCCL 通信，但把 3D 并行最关键的逻辑表达出来：先给每个 rank 计算 DP/TP/PP 坐标，再据此形成不同通信组。

```python
from collections import defaultdict
from math import prod

def build_3d_groups(world_size: int, dp: int, tp: int, pp: int):
    assert world_size == dp * tp * pp
    rank_to_coord = {}
    dp_groups = defaultdict(list)
    tp_groups = defaultdict(list)
    pp_groups = defaultdict(list)

    for rank in range(world_size):
        dp_idx = rank // (tp * pp)
        rem = rank % (tp * pp)
        pp_idx = rem // tp
        tp_idx = rem % tp

        coord = (dp_idx, tp_idx, pp_idx)
        rank_to_coord[rank] = coord

        # 同一 TP/PP 位置，不同 DP 副本，组成 DP 组
        dp_groups[(tp_idx, pp_idx)].append(rank)
        # 同一 DP/PP 位置，不同 TP 切片，组成 TP 组
        tp_groups[(dp_idx, pp_idx)].append(rank)
        # 同一 DP/TP 位置，不同 PP 阶段，组成 PP 组
        pp_groups[(dp_idx, tp_idx)].append(rank)

    return rank_to_coord, list(dp_groups.values()), list(tp_groups.values()), list(pp_groups.values())


def estimate_per_gpu_memory(total_params_gb: float, activation_gb_per_sample: float,
                            global_batch: int, dp: int, tp: int, pp: int):
    assert dp > 0 and tp > 0 and pp > 0
    param_mem = total_params_gb / (tp * pp)
    act_mem = activation_gb_per_sample * (global_batch / dp)
    return param_mem + act_mem


rank_to_coord, dp_groups, tp_groups, pp_groups = build_3d_groups(
    world_size=8, dp=2, tp=2, pp=2
)

assert rank_to_coord[0] == (0, 0, 0)
assert rank_to_coord[7] == (1, 1, 1)
assert len(dp_groups) == 4
assert len(tp_groups) == 4
assert len(pp_groups) == 4
assert all(len(g) == 2 for g in dp_groups + tp_groups + pp_groups)

mem = estimate_per_gpu_memory(
    total_params_gb=80.0,
    activation_gb_per_sample=0.5,
    global_batch=8,
    dp=2,
    tp=2,
    pp=2,
)

assert mem == 22.0
print("3D group construction and memory estimate look valid.")
```

这段代码对应的直觉是：

1. 初始化时先把全局 rank 映射到三维坐标 `(dp_idx, tp_idx, pp_idx)`。
2. 训练时：
   - TP 组内做层内切分计算和同步。
   - PP 相邻 stage 之间发激活与反向梯度。
   - DP 组在反向结束后做梯度聚合，或者交给 ZeRO/FSDP 变体做分片同步。

工业实现里，Megatron-LM、DeepSpeed 之类框架做的事更复杂，但骨架就是这个顺序。伪代码可以写成：

```python
for rank in cluster:
    tp_group = assign_tensor_parallel_group(rank)
    pp_stage = assign_pipeline_stage(rank)
    dp_group = assign_data_parallel_group(rank)

for micro_batch in loader:
    hidden = forward_with_tp_and_pp(micro_batch, tp_group, pp_stage)
    grads = backward_with_tp_and_pp(hidden, tp_group, pp_stage)
    sync_or_shard_gradients(grads, dp_group)
```

真实工程例子：在 512 卡、TP=8、PP=8、DP=8 的配置中，通常会让 8 张同节点 GPU 组成一个 TP 域；多个节点串成 PP；复制出多个 DP 副本。这样设计不是美观问题，而是因为 TP collectives 高频且延迟敏感，必须优先放在 NVLink/NVSwitch 范围内。

---

## 工程权衡与常见坑

3D 并行难的不是原理，而是三条轴一旦同时存在，错误配置会互相放大。

| 维度 | 常见坑 | 直接后果 | 规避策略 |
| --- | --- | --- | --- |
| TP | 把 TP 组跨节点部署 | 高频 AllReduce 走慢链路，利用率骤降 | TP 尽量限制在单节点或单 NVSwitch 域 |
| PP | stage 切分不均或微批太少 | pipeline bubble 明显，部分 GPU 长时间等待 | 做层级均衡、增大微批、必要时激活重计算 |
| DP | 仅扩大 DP 不做状态分片 | 梯度和优化器同步开销爆炸 | 配合 ZeRO、梯度压缩、1-bit Adam |
| 拓扑映射 | 按 rank 顺序随意分组 | 通信热点错位，链路拥塞 | 先看物理拓扑，再映射逻辑坐标 |
| 批大小设计 | 全局 batch、micro-batch、grad accumulation 混淆 | 吞吐和收敛都异常 | 明确三者关系并统一日志口径 |

最典型的坑是把 TP 组跨节点。因为 TP 的通信发生在几乎每一层、每个微批上，它不是“偶尔同步一次”，而是训练主路径的一部分。如果 TP 走 InfiniBand，而不是节点内高速互连，那么单次层内同步的几百微秒会被放大到几毫秒，随后 PP 的下一阶段也开始等待，最后形成串联阻塞。表面看是“TP 慢了”，实际是整条流水线都被拖住。

第二个坑是 PP 的 stage 不均衡。比如把 embedding 和几个超大 attention block 切到同一 stage，而其他 stage 负载较轻，那么最重的 stage 决定整体节拍，其他卡即使空着也不能跳过等待。PP 不是“层数均分”就够了，而是要按实际 FLOPs、激活和重计算成本做均衡。

第三个坑是误以为 DP 永远最便宜。对小模型这是对的，但模型参数一旦很大，DP 的梯度和优化器状态同步会变成主导开销。这就是 ZeRO 重要的原因。ZeRO 的本质是“把原本在每个 DP 副本上重复保存的优化器状态、梯度、参数分片存放”，减少冗余内存和通信峰值。1-bit Adam 则进一步压缩通信量，在网络成为瓶颈时尤其有效。

---

## 替代方案与适用边界

3D 并行不是唯一答案，也不是越复杂越先进。应当按瓶颈选策略，而不是按术语选策略。

| 方案 | 适用场景 | 优点 | 代价 |
| --- | --- | --- | --- |
| 1D：仅 DP | 模型能放进单卡，目标是提吞吐 | 最简单，工程成熟 | 无法解决超大模型显存问题 |
| 2D：DP+TP 或 DP+PP | 宽层过大或深度过深，但还没到全维瓶颈 | 比 3D 简单，收益明显 | 某一侧仍可能成为瓶颈 |
| 3D：DP+TP+PP | 百亿到千亿级训练，宽度、深度、吞吐同时受限 | 资源利用更全面 | 分组、拓扑、调度复杂 |
| 4D：3D+序列/空间并行 | 长序列或高分辨率输入成为主瓶颈 | 继续拆激活和注意力成本 | 实现复杂度再上升 |
| 3D+EP（MoE） | 稀疏专家模型 | 扩容效率高 | 路由、负载均衡更难 |

有些场景下，不对称方案比标准 3D 更合适。比如异构集群里有 80GB 和 40GB GPU 混用，就不适合强行做完全对称的 $D=T=P$。这时类似 AutoHet 的思路更合理：让 TP 固定在几张同构、高速互连的卡上，PP 和 DP 再按剩余资源灵活展开。目标不是对称，而是让每条通信走在合适的链路上。

另一个常见替代是放弃 TP，转向 FSDP + PP。它的好处是模型代码改动更小，很多层不需要显式张量切分；缺点是如果单层本身已经大到放不下，再强的参数分片也解决不了算子级别的峰值显存。这就是它的边界。

判断是否该上 3D，可以问三个问题：
1. 单层是否已经大到单卡无法承受？如果是，需要 TP。
2. 整网深度和激活是否仍放不下？如果是，需要 PP。
3. 即使能放下，吞吐是否仍远低于目标？如果是，需要 DP。

三个答案都接近“是”，那就进入 3D 并行的典型适用区。

---

## 参考资料

- EmergentMind, “3D Parallelism in Distributed Systems”: https://www.emergentmind.com/topics/3d-parallelism
- System Overflow, “3D Parallelism and Topology Aware Mapping in Production”: https://www.systemoverflow.com/learn/ml-training-infrastructure/distributed-training/3d-parallelism-and-topology-aware-mapping-in-production
- Microsoft Research Blog, “DeepSpeed: Extreme-scale model training for everyone”: https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/

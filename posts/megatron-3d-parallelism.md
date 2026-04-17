## 核心结论

Megatron-LM 的 3D 并行，本质上是把同一个训练任务沿三条彼此独立的轴拆开：

- TP，Tensor Parallelism，张量并行：把单层内部的参数和计算拆到多张 GPU 上
- PP，Pipeline Parallelism，流水线并行：把模型的不同层段拆到不同设备组上
- DP，Data Parallelism，数据并行：复制多份模型副本，让不同副本处理不同样本

三种并行度可以直接相乘，形成总资源规模：

$$
\text{Total GPUs} = TP \times PP \times DP
$$

如果再叠加上下文并行、专家并行等额外维度，公式会继续乘下去，但 3D 并行是最基本的骨架。

Megatron-LM 的关键不只是“同时用了三种并行”，而是“把三种并行映射到合适的硬件链路上”：

- TP 通信最频繁，优先放在单节点 NVLink 或 NVSwitch 域内
- PP 主要跨阶段传递激活值，频率低于 TP，可以跨节点走 InfiniBand
- DP 负责梯度同步和副本扩展，通常覆盖剩余的全局设备维度

对初学者最重要的一点是：`TP × PP × DP` 不只是一个计数公式，它同时决定了通信路径、显存压力和吞吐上限。并行度选错，训练会“能跑但很慢”；映射对了，才可能接近线性扩展。

---

## 问题定义与边界

训练大语言模型时，瓶颈通常同时来自三件事：

| 问题 | 白话解释 | 单靠哪种办法不够 |
|---|---|---|
| 参数装不下 | 单卡显存不足，完整模型放不进一张卡 | 只用 DP 不行，因为每张卡都要保留完整模型副本 |
| 单步计算太慢 | 一张卡无法在合理时间内完成一次前向和反向 | 只增大 batch 不够，因为单层矩阵乘本身已经很大 |
| GPU 间通信太重 | 多卡协作时，传数据的时间超过算数据的时间 | 并行度虽然更大，但错误映射会让通信盖过计算 |

这里先明确边界：

- 讨论对象是 Transformer 类大模型训练，重点是 Megatron-LM / Megatron-Core 的训练并行
- 默认硬件拓扑是“节点内高速互联，节点间较慢互联”，例如 DGX A100 内部 NVLink / NVSwitch，节点间 200Gb/s InfiniBand
- 不展开 ZeRO/FSDP 的完整实现细节，只在“替代方案”部分做边界比较
- 不讨论推理部署，只讨论训练阶段

为什么不能只靠 DP？

因为 DP 的基本做法是：每个副本都保留一整份模型参数，然后把不同样本分给不同副本计算，最后在反向传播后同步梯度。它实现简单，但有两个硬伤：

1. 单卡必须容纳完整模型
2. 梯度同步会在更大范围的网络上发生，模型越大，同步负担越重

可以把 DP 理解成“把同一台工厂复制多份”，而不是“把一台工厂拆开协作”。如果单台工厂本来就装不下，复制再多份也没有意义。

为什么不能只靠 TP？

因为 TP 解决的是“单层太大”的问题。它会把一个线性层的权重矩阵拆到多张卡上，让每张卡只算一部分。例如一个大矩阵乘法：

$$
Y = XW
$$

如果把权重矩阵 $W$ 按列切成 $W_1, W_2, \dots, W_t$，那么每张卡只需要计算：

$$
Y_i = XW_i
$$

最后再把各卡结果拼接或归约成完整输出。问题在于，这类层内切分几乎每层都会发生同步，因此通信频率非常高。只要 TP 跨节点，就容易撞上带宽瓶颈。

为什么不能只靠 PP？

因为 PP 解决的是“整网太深”的问题。它把模型按层段切开，比如把 80 层 Transformer 切成 4 段，每段各自放在不同设备组。但如果某一段内部仍然太大，单卡还是放不下。此外，PP 还有气泡问题。所谓气泡，就是流水线没有被完全填满时，前后阶段会出现空闲。

一个非常简化的示意如下：

| 时刻 | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 4 |
|---|---|---|---|---|
| 1 | 微批 1 | 空闲 | 空闲 | 空闲 |
| 2 | 微批 2 | 微批 1 | 空闲 | 空闲 |
| 3 | 微批 3 | 微批 2 | 微批 1 | 空闲 |
| 4 | 微批 4 | 微批 3 | 微批 2 | 微批 1 |

开始填充和最后排空时，总会有部分阶段空闲，这部分空闲就是流水线气泡。

所以，3D 并行解决的不是单一问题，而是在三个问题同时成立时做联合优化：

- TP 解决层内参数和算子过大
- PP 解决整网深度过大
- DP 解决全局吞吐扩展

一个常见误区是把“并行度更大”误认为“训练一定更快”。实际不是。并行度增加，首先换来的是“能训练更大的模型”；只有当通信域和硬件拓扑匹配时，速度才会提升。

---

## 核心机制与推导

先看最基本的资源关系。设总卡数为 `world_size`，那么：

$$
DP = \frac{\text{world\_size}}{TP \times PP}
$$

更一般地，如果还引入上下文并行 `CP`，则：

$$
DP = \frac{\text{world\_size}}{TP \times PP \times CP}
$$

这个公式的工程含义是：TP 和 PP 不是“先随便选，再看效果”，而是先占用世界规模，剩下的设备数才会形成 DP 副本。

### 三个维度到底切了什么

| 并行方式 | 切分对象 | 直接收益 | 主要通信 | 典型部署位置 |
|---|---|---|---|---|
| TP | 单层内部参数、矩阵乘结果 | 降低单卡参数量与单层计算压力 | all-reduce / all-gather / reduce-scatter | 节点内 NVLink / NVSwitch |
| PP | 模型深度上的层段 | 降低单阶段需要承载的模型深度 | 阶段边界激活值发送与接收 | 可跨节点 InfiniBand |
| DP | 完整模型副本 | 扩展样本吞吐量 | 梯度归约、参数同步 | 覆盖全局剩余设备 |

这里的“正交”很重要。对初学者来说，可以把它理解为“切分的方向不同，因此可以组合”：

- TP 切的是层内结构
- PP 切的是层间深度
- DP 切的是样本批次

因为三者切的不是同一个对象，所以它们通常可以同时存在。

### TP 的最小直觉：一层是怎么拆开的

以线性层为例，输入张量是 $X \in \mathbb{R}^{b \times h}$，权重矩阵是 $W \in \mathbb{R}^{h \times 4h}$。如果使用列并行，把输出维度切成两半：

$$
W = [W_1, W_2]
$$

那么两张卡分别计算：

$$
Y_1 = XW_1,\quad Y_2 = XW_2
$$

最后得到：

$$
Y = [Y_1, Y_2]
$$

如果是行并行，则会把输入侧维度切开，局部结果往往需要一次归约：

$$
Y = X_1W_1 + X_2W_2
$$

这就是为什么 TP 常见通信原语是 all-gather 和 all-reduce。它不是额外附带的步骤，而是张量切分本身就要求结果重组。

### 玩具例子：32 张卡如何切

假设你有 4 台 DGX A100，每台 8 卡，一共 32 GPU。

现在设：

- `TP = 8`
- `PP = 4`

则：

$$
DP = \frac{32}{8 \times 4} = 1
$$

这表示：

- 每个张量并行组正好占满一台 DGX 的 8 卡
- 模型被切成 4 个流水线阶段，跨 4 个节点串起来
- 没有额外的数据并行副本

这个映射的含义非常直接：

- TP 的高频通信完全留在单节点高速互联内
- PP 只在阶段边界上跨节点传激活值
- 没有 DP 副本，因此不需要额外的大范围梯度同步组

对初学者，这个例子能建立第一个直觉：  
`TP` 应优先贴合“节点内高速域”，`PP` 用来把多个节点串起来，`DP` 则由剩余资源自然形成。

### 真实工程例子：3072 GPU 的 Selene 配置

NVIDIA 在 Selene 上训练超大 GPT 时，典型思路是把 3072 张 A100 按下述方式组织：

- `TP = 8`
- `PP = 8`
- `DP = 48`

因为：

$$
3072 = 8 \times 8 \times 48
$$

这个配置为何合理？

1. `TP=8` 与单 DGX A100 的 8 卡高速互联天然对齐
2. `PP=8` 让超深模型能在深度方向上拆成多个阶段
3. 剩余维度形成 `DP=48`，用于扩大全局吞吐

它的重点不在“8 和 48 这几个数字本身”，而在通信层级的安排：

- 层内最频繁的 TP 通信不出节点
- PP 只在阶段边界跨节点交换激活
- DP 的同步落在副本维度，可以与分布式优化器和通信重叠策略一起调优

### 为什么 scatter/gather 有意义

Megatron-LM 的关键优化之一是：跨流水线边界时，不一定直接发送完整激活，而是尽量让跨节点部分更小，把重组留给节点内高速互联。

设某阶段输出激活总大小为 $A$，TP 大小为 $t$。如果先按 TP 维度切分，那么每张卡跨节点只需发送：

$$
\frac{A}{t}
$$

所有卡总共仍然传输了与原张量同阶的数据量，但跨慢链路的单次消息被缩小了。接收端再在本地 TP 组内做重组，例如 all-gather。

可写成：

$$
A \rightarrow \left(\frac{A}{t}, \frac{A}{t}, \dots, \frac{A}{t}\right)
$$

这类优化的价值不是“完全消灭通信”，而是“把贵的通信放少，把便宜的通信放多”。工程里真正重要的通常不是通信次数绝对最小，而是高频通信是否落在最快的链路上。

### 3D 并行的通信分层可以怎么理解

下面这个表可以帮助建立层级直觉：

| 维度 | 通信频率 | 通信数据的典型内容 | 最怕什么 |
|---|---|---|---|
| TP | 高 | 分片结果、归约结果、拼接结果 | 跨节点慢链路 |
| PP | 中 | 激活值、反向梯度 | 阶段切分失衡 |
| DP | 低到中 | 梯度、优化器相关状态 | 副本数过大导致全局同步变重 |

一个简单的判断原则是：

- 高频通信优先绑定最快链路
- 中频通信可以接受跨节点，但要尽量减少不必要的大张量传输
- 低频但范围大的通信，需要通过 overlap、sharding、bucket 化等方法降低可见开销

---

## 代码实现

实际使用 Megatron-Core 时，核心工作不是“打开一个并行开关”，而是把并行维度显式写进启动配置，并确保它们与硬件规模完全匹配。下面先给一个可运行的 Python 小工具，用来验证 3D 并行配置是否自洽，并把 GPU 分组结果打印出来。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ParallelConfig:
    world_size: int
    tp: int
    pp: int
    cp: int = 1

    @property
    def dp(self) -> int:
        denom = self.tp * self.pp * self.cp
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if min(self.tp, self.pp, self.cp) <= 0:
            raise ValueError("tp, pp, cp must be > 0")
        if self.world_size % denom != 0:
            raise ValueError(
                f"world_size={self.world_size} cannot be divided by "
                f"tp*pp*cp={denom}"
            )
        return self.world_size // denom


def build_tp_groups(cfg: ParallelConfig) -> List[List[int]]:
    groups = []
    for start in range(0, cfg.world_size, cfg.tp):
        groups.append(list(range(start, start + cfg.tp)))
    return groups


def build_pp_groups(cfg: ParallelConfig) -> List[List[int]]:
    # 这里给出一种便于理解的分组方式：
    # 先按 TP 把 rank 连续分块，再把每个阶段串起来。
    dp = cfg.dp
    stage_span = cfg.tp
    replica_span = cfg.tp * cfg.pp

    groups = []
    for replica_id in range(dp):
        replica_base = replica_id * replica_span
        for tp_lane in range(cfg.tp):
            group = []
            for stage_id in range(cfg.pp):
                rank = replica_base + stage_id * stage_span + tp_lane
                group.append(rank)
            groups.append(group)
    return groups


def build_dp_groups(cfg: ParallelConfig) -> List[List[int]]:
    # 相同 TP 位置、相同 PP 阶段，但属于不同副本的 rank 组成一个 DP 组。
    groups = []
    dp = cfg.dp
    replica_span = cfg.tp * cfg.pp

    for stage_id in range(cfg.pp):
        for tp_lane in range(cfg.tp):
            group = []
            for replica_id in range(dp):
                rank = replica_id * replica_span + stage_id * cfg.tp + tp_lane
                group.append(rank)
            groups.append(group)
    return groups


def main() -> None:
    examples = [
        ParallelConfig(world_size=32, tp=8, pp=4),
        ParallelConfig(world_size=3072, tp=8, pp=8),
        ParallelConfig(world_size=64, tp=4, pp=2),
    ]

    for cfg in examples:
        print("=" * 72)
        print(cfg)
        print(f"derived dp = {cfg.dp}")

    demo = ParallelConfig(world_size=16, tp=4, pp=2)
    print("=" * 72)
    print("demo config:", demo)
    print("tp groups:", build_tp_groups(demo))
    print("pp groups:", build_pp_groups(demo))
    print("dp groups:", build_dp_groups(demo))


if __name__ == "__main__":
    main()
```

这段代码有三个作用：

1. 验证并行维度是否满足整除约束
2. 明确 `dp = world_size / (tp * pp * cp)` 是硬约束，不是建议值
3. 让初学者看到“rank 到并行组”的具体映射，而不是只记住抽象公式

如果运行 `ParallelConfig(world_size=16, tp=4, pp=2)`，可以得到如下直觉：

- rank `0,1,2,3` 是一个 TP 组
- rank `4,5,6,7` 是下一个流水线阶段对应的 TP 组
- rank `0,4`、`1,5`、`2,6`、`3,7` 可以看成一组 PP 对位关系
- 再往后 `8-15` 是第二个 DP 副本

下面是一个简化版启动命令：

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 1 \
  --micro-batch-size 4 \
  --global-batch-size 512 \
  --seq-length 8192 \
  --num-layers 48 \
  --hidden-size 6144 \
  --num-attention-heads 48 \
  --bf16 \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --recompute-activations
```

如果总卡数是 64，那么这里对应：

$$
DP = \frac{64}{4 \times 2 \times 1} = 8
$$

也就是：

- 4 张卡组成一个 TP 组
- 2 个阶段组成一条 PP 链
- 剩余设备形成 8 个 DP 副本

这里顺便解释几个初学者容易混淆的参数：

| 参数 | 含义 | 为什么重要 |
|---|---|---|
| `--tensor-model-parallel-size` | TP 大小 | 决定层内切分的组大小 |
| `--pipeline-model-parallel-size` | PP 大小 | 决定模型被切成多少段 |
| `--micro-batch-size` | 单次送入流水线的微批大小 | 直接影响 PP 气泡和显存占用 |
| `--global-batch-size` | 全局 batch | 决定优化器看到的总样本量 |
| `--recompute-activations` | 激活重计算 | 用额外算力换更低显存 |

全局 batch 和各并行维度之间，通常还满足：

$$
\text{Global Batch} =
\text{Micro Batch} \times
\text{Data Parallel Size} \times
\text{Gradient Accumulation Steps}
$$

这条公式很重要，因为很多人只算了 `TP × PP × DP = world_size`，却忘了 batch 约束。结果是并行度虽然合法，但训练配置在吞吐、稳定性或显存上并不成立。

### 配置时应先算什么

建议按下面顺序推配置：

| 步骤 | 先回答的问题 | 常见依据 |
|---|---|---|
| 1 | `TP` 取多少 | 单节点 GPU 数、NVLink / NVSwitch 域大小 |
| 2 | `PP` 取多少 | 模型层数、每阶段能否装进显存 |
| 3 | `DP` 是多少 | 用总卡数除以前两者乘积 |
| 4 | micro-batch 是否足够 | 流水线能否被填满、气泡是否可接受 |
| 5 | 是否加 `CP/SP` | 长序列导致激活过大时再考虑 |

### 一个真实工程判断过程

假设你要训练一个较大的 GPT，集群是 8 台机器，每台 8 卡，总计 64 GPU。

可以按这个顺序想：

1. 节点内是 8 卡高速互联，优先考虑 `TP=8` 或 `TP=4`
2. 如果模型很深，单阶段放不下，可以考虑 `PP=4`
3. 若取 `TP=4, PP=4`，则：

$$
DP = \frac{64}{4 \times 4} = 4
$$

4. 这意味着一个模型副本占 16 张卡，总共复制 4 份

这个配置是否好，不由公式本身决定，而由两个更具体的问题决定：

- TP 组是否完全落在高速互联域内
- PP 切分后，每个阶段的计算量、显存占用、激活传输量是否大致平衡

一个更实用的判断表如下：

| 观察现象 | 更可能要调整什么 |
|---|---|
| 单层通信很重，GPU 经常等同步 | 优先检查 TP 是否跨慢链路 |
| 某些 stage 长时间忙，另一些 stage 空闲 | 优先重切 PP 分段 |
| 每步都能跑，但全局吞吐始终上不去 | 检查 DP 是否过大，或 batch / overlap 设置不合理 |
| 参数装得下，但激活爆显存 | 考虑增加 PP、开启重计算，或引入 CP / SP |

---

## 工程权衡与常见坑

3D 并行最常见的错误，不是公式算错，而是拓扑映射错。

### 常见坑一：TP 跨了慢链路

TP 通信是三者中最高频的。如果一个 TP 组被拆到两个节点，或者被拆到两个较慢的互联域上，那么几乎每层都要跨慢链路做归约，性能会明显下降。

| 问题场景 | 性能后果 | 原因 | 缓解手段 |
|---|---|---|---|
| TP 组跨节点 | 每层 all-reduce 走 IB，吞吐显著下滑 | TP 通信在每层反复出现 | 让 TP 优先占满节点内 NVLink / NVSwitch 域 |
| TP 映射不连续 | 局部通信绕远路 | rank 到物理设备的映射不贴拓扑 | 显式检查 rank mapping 和设备拓扑 |
| TP 过大 | 通信占比迅速升高 | 切分更细，重组更频繁 | 不要把 TP 盲目开到最大 |

初学者可以记一个非常实用的原则：  
如果只能保证一件事，先保证 TP 不跨慢链路。

### 常见坑二：只看卡数，不看网络层级

很多人能写出“数学上合法”的配置，但它并不“网络上合理”。

例如一个配置：

- `TP = 4`
- `PP = 4`
- `DP = 16`

从乘积上看完全成立。但如果这 4 卡 TP 组并没有稳定落在同一个高速域内，那么训练虽然能启动，实际表现可能是：

- 显存够
- 进程组也创建成功
- 但 MFU 很低
- GPU 利用率不理想
- NCCL 时间占比偏高

MFU 可以理解为“模型 FLOPs 利用率”，即理论算力中，真正被有效用于模型训练的比例。MFU 低不一定说明计算核慢，很多时候说明通信、等待、负载不均衡吞掉了时间。

### 常见坑三：把 PP 当成免费扩展

PP 的问题不是“不能用”，而是“不是白送的”。

PP 增大后，模型更容易放下，但同时：

- 流水线气泡可能变大
- 激活跨阶段传输次数会增加
- 阶段切分不均衡时，慢阶段会拖累全链路

通常有一个经验规律：

- PP 越大，越依赖足够多的 micro-batch 去填平气泡
- 如果模型其实没有深到必须切很多段，盲目增大 PP 反而得不偿失

可以用一个简化直觉理解气泡损失。设流水线阶段数为 $p$，微批数为 $m$，则 1F1B 调度下的气泡比例会随着 `m` 增大而下降，常见近似直觉是：

$$
\text{Bubble Fraction} \approx \frac{p - 1}{m + p - 1}
$$

这不是所有实现下都精确成立，但足够帮助初学者理解一件事：  
微批太少时，PP 放大后并不会自动带来更高利用率。

### 常见坑四：忽略激活和优化器状态

初学者经常只盯参数量，但训练显存至少还包括：

- 参数
- 激活值
- 梯度
- 优化器状态

很多优化器会额外维护一阶、二阶状态，因此仅参数本体远远不是全部显存占用。一个常见的误判是：

- 参数切开后看起来能放下
- 实际一跑长序列就 OOM
- 原因不是参数，而是激活值暴涨

下面这个表更接近真实训练时的显存视角：

| 显存组成 | 何时最突出 | 常见缓解手段 |
|---|---|---|
| 参数 | 模型本体很大 | TP、PP、参数分片 |
| 激活 | 序列长、层数深、micro-batch 大 | 激活重计算、PP、CP、SP |
| 梯度 | 反向传播时 | mixed precision、梯度分桶 |
| 优化器状态 | Adam 类优化器尤其明显 | distributed optimizer、ZeRO/FSDP |

所以即使 TP 把参数切小了，也不代表显存一定够。很多场景下，真正逼你继续引入 PP、SP、CP 或激活重计算的，不是参数，而是激活内存。

### 常见坑五：把 world size 整除理解成“唯一约束”

`world_size % (tp * pp * cp) == 0` 只是最低约束，不是充分条件。

下面两个配置都可能整除，但表现完全不同：

| 配置 | 数学上是否合法 | 工程上是否合理 |
|---|---|---|
| `world=64, tp=8, pp=2, dp=4` | 合法 | 若 8 卡正好是一整个高速域，通常合理 |
| `world=64, tp=8, pp=2, dp=4` 但 TP 组跨两个节点 | 仍然合法 | 通常不合理，因为高频 TP 通信跨慢链路 |
| `world=64, tp=2, pp=8, dp=4` | 合法 | 若模型没有深到需要 8 段，可能气泡太大 |

也就是说，整除关系只能回答“能不能构造进程组”，不能回答“跑得好不好”。

---

## 替代方案与适用边界

3D 并行不是唯一方案，它适合的是“模型特别大、集群网络分层明显、训练框架允许深度定制”的场景。

### 方案一：FSDP / ZeRO 类方案

这类方案的核心思想不是把模型按层内或层间切开，而是把参数、梯度、优化器状态做分片。所谓分片，可以理解为“不再让每张卡都长期持有完整状态”。

优点：

- 对许多模型代码改造较少
- 框架通用性更强
- 在中等规模集群上，工程落地往往更快

缺点：

- 通信仍然可能很重
- 当模型极大、拓扑分层明显时，不一定比精细设计的 TP/PP 更高效
- 某些场景下，通信与参数重组会对长序列训练造成额外压力

可以用一个简化表格对比：

| 方案 | 主要切分对象 | 优势 | 局限 |
|---|---|---|---|
| 3D 并行 | 层内、层间、样本 | 更容易贴合硬件拓扑 | 配置复杂，依赖模型和系统协同 |
| FSDP / ZeRO | 参数、梯度、优化器状态 | 通用性较高，改造成本相对低 | 通信模式不一定最适合超大规模拓扑 |
| 纯 DP | 样本 | 实现最简单 | 单卡必须装得下完整模型 |

### 方案二：DP-last 映射

默认思路通常是先考虑 TP，再由剩余资源形成 DP；而在更复杂的网络拓扑中，工程上常强调 `DP-last`，即先保证带宽敏感的维度放进高速域，最后再让 DP 吃掉剩余设备。

核心思想是：

- 先保 TP，因为它最怕慢链路
- 再保 PP 或 CP，因为它们也可能受拓扑影响
- DP 放在最后，因为它相对更容易接受跨更大范围的同步

| 方案 | 优先保障的通信 | 适用场景 | 风险 |
|---|---|---|---|
| 默认 TP/PP/DP | 传统 DGX + IB 集群 | 拓扑规则、节点对称 | 若映射粗糙，TP 可能跨慢链路 |
| DP-last | 先保住 TP/CP/PP 的高速域 | NVL72、GB200 等更复杂互联 | 规划复杂度更高 |
| 加 CP/SP | 主要处理长序列激活 | 长上下文训练 | 额外通信与实现复杂度上升 |

### 方案三：加入 Context Parallelism / Sequence Parallelism

当序列长度很长时，问题不再只是参数放不下，而是激活值增长得太快。此时可以加入 CP 或 SP，把序列维度也切开。

它们的价值在于：

- 不必推翻 TP/PP 的主框架
- 能更直接地降低长序列训练的激活内存压力
- 对超长上下文模型尤其重要

适用边界也很明确：

- 如果瓶颈是层内参数太大，先看 TP
- 如果瓶颈是模型深度太大，先看 PP
- 如果瓶颈是序列太长，优先考虑 CP/SP
- 如果瓶颈是优化器状态和全局显存压力，优先比较 ZeRO/FSDP

因此，3D 并行不是“唯一正确答案”，而是当前大模型训练里最经典、最容易与硬件拓扑对齐的一套主框架。

可以把选择路径简化成一张表：

| 主要瓶颈 | 优先考虑 |
|---|---|
| 单层太大，矩阵乘放不下 | TP |
| 模型总层数太深，整网装不下 | PP |
| 吞吐不够，需要更多样本并行 | DP |
| 上下文太长，激活爆炸 | CP / SP |
| 参数、梯度、优化器状态总量过大 | ZeRO / FSDP |

---

## 参考资料

- [NVIDIA Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/parallelism-guide.html)
- [Scaling Language Model Training to a Trillion Parameters Using Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
- [NVIDIA DGX A100 User Guide](https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html)
- [Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/nightly/performance-guide.html)
- [Megatron Bridge Latest Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)
- [Megatron-LM GitHub Repository](https://github.com/NVIDIA/Megatron-LM)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

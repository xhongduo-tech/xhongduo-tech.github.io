## 核心结论

ZeRO，中文常译为“零冗余优化器”，本质是在**数据并行**维度上把原来每张卡都完整复制的一套训练状态拆开保存。数据并行的白话解释是：多张 GPU 跑同一份模型、处理不同批次数据，然后同步梯度。

如果使用 mixed-precision Adam，传统数据并行里每张卡通常都要保存三类核心状态：

| 状态 | 白话解释 | 每参数字节数 |
|---|---:|---:|
| 参数 $P$ | 真正参与前向和反向计算的模型权重 | 2 Bytes（FP16） |
| 梯度 $G$ | 损失函数对参数的变化方向 | 2 Bytes（FP16） |
| 优化器状态 $OS$ | Adam 为了更新更稳定额外维护的统计量 | 12 Bytes（FP32 主参数 + $m$ + $v$） |

设 $\Psi$ 表示“模型参数按 FP16 存储时的总字节数”，$N_d$ 表示数据并行卡数，则传统数据并行每卡状态内存近似为：

$$
M_{\text{DP}} = 2\Psi + 2\Psi + 12\Psi = 16\Psi
$$

ZeRO 的三个阶段按“先切优化器，再切梯度，最后切参数”的顺序推进：

| 方案 | 每卡保存什么 | 每卡状态内存 |
|---|---|---|
| Stage 0 / 普通 DP | 参数、梯度、优化器状态都完整复制 | $16\Psi$ |
| ZeRO-1 | 仅优化器状态切成 $1/N_d$ | $4\Psi + 12\Psi/N_d$ |
| ZeRO-2 | 梯度和优化器状态都切成 $1/N_d$ | $2\Psi + 14\Psi/N_d$ |
| ZeRO-3 | 参数、梯度、优化器状态都切成 $1/N_d$ | $16\Psi/N_d$ |

当 $N_d=8$ 时：

- ZeRO-1：每卡从 $16\Psi$ 降到 $5.5\Psi$
- ZeRO-2：每卡从 $16\Psi$ 降到 $3.75\Psi$
- ZeRO-3：每卡从 $16\Psi$ 降到 $2\Psi$

这就是 ZeRO 的核心价值：**把显存瓶颈从“模型状态完整复制”改成“通信调度是否跟得上”**。7B 级模型在 8 卡上，ZeRO-3 可以把每卡模型状态显存从约 112GB 压到约 14GB，量级上接近 $1/8$。

---

## 问题定义与边界

问题不是“模型算不动”，而是“模型状态装不下”。

很多初学者第一次接触分布式训练，会以为多加几张卡就能自然训练更大的模型。这个结论只对**计算量**成立，不对**显存占用**成立。原因很直接：普通数据并行并没有把模型状态分摊到多卡上，而是每张卡都复制一整套。

### 为什么普通数据并行浪费显存

假设有 $N_d$ 张卡，每张卡都做下面几件事：

1. 保留完整参数，完成前向和反向。
2. 产生完整梯度。
3. 用完整优化器状态更新参数。
4. 与其他卡同步梯度。

所以虽然总算力随着 GPU 数增加了，但**单卡显存并没有因为卡变多而下降**。这就是传统 DP 的边界：吞吐提高了，但单卡能容纳的模型规模几乎没变。

### 本文讨论的边界

本文只讨论以下设定：

| 维度 | 范围 |
|---|---|
| 并行方式 | 数据并行及其 ZeRO 变体 |
| 优化器 | 以 Adam / AdamW 为代表 |
| 精度 | mixed precision，参数与梯度常见为 FP16/BF16，优化器状态常为 FP32 |
| 目标 | 降低模型状态显存占用 |
| 不覆盖 | 张量并行、流水线并行、激活检查点细节、推理 KV Cache 优化 |

也就是说，ZeRO 解决的是“**数据并行下的冗余存储**”，不是所有分布式训练问题的总解。

### 玩具例子

考虑一个只有 4 个参数的小模型，参数向量为：

$$
w = [w_0, w_1, w_2, w_3]
$$

如果有 2 张卡做普通数据并行，那么卡 0 和卡 1 都各自保存：

- 全部参数 $[w_0,w_1,w_2,w_3]$
- 全部梯度 $[g_0,g_1,g_2,g_3]$
- 全部 Adam 状态 $m,v$

这在逻辑上简单，但存储上完全重复。

如果用 ZeRO-3，那么可以改成：

- 卡 0 只“常驻”保存 $[w_0,w_1]$
- 卡 1 只“常驻”保存 $[w_2,w_3]$

当某一层计算需要完整参数时，临时通信拼起来；算完后再释放。这就是 ZeRO-3 的直觉模型。

---

## 核心机制与推导

ZeRO 的设计不是一步到位，而是按状态类型逐层去冗余。

### Stage 1：只切优化器状态

优化器状态是最先该切的，因为 Adam 的状态最重。白话讲，Adam 不只是保存参数本身，还要保存“历史梯度均值”和“历史梯度平方均值”，用于更稳定地更新。

普通 DP：

$$
M_0 = 2\Psi + 2\Psi + 12\Psi = 16\Psi
$$

ZeRO-1 只切优化器状态：

$$
M_1 = 2\Psi + 2\Psi + \frac{12\Psi}{N_d}
= 4\Psi + \frac{12\Psi}{N_d}
$$

当 $N_d$ 很大时，优化器状态部分被明显摊薄，但参数和梯度仍然是全量复制。

### Stage 2：再切梯度

梯度是反向传播得到的更新方向。白话讲，梯度告诉优化器“参数该往哪个方向改”。

ZeRO-2 把梯度也切到各卡上：

$$
M_2 = 2\Psi + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d}
= 2\Psi + \frac{14\Psi}{N_d}
$$

这一步之后，参数仍完整复制，但梯度和优化器状态不再完整复制，所以显存继续下降。

### Stage 3：最后切参数

参数是模型本体。ZeRO-3 连参数也只保留各卡自己的分片：

$$
M_3 = \frac{2\Psi}{N_d} + \frac{2\Psi}{N_d} + \frac{12\Psi}{N_d}
= \frac{16\Psi}{N_d}
$$

这是最彻底的去冗余方案，也是节省显存最多的一步。

### 为什么 ZeRO-3 会增加通信

Stage 1 和 Stage 2 中，参数仍完整驻留在每张卡，前向和反向都能直接算，所以主要通信仍围绕梯度同步展开。

到了 Stage 3，参数也被切碎了。某层在前向前必须先拿到“这一层当前需要的完整参数”，常见做法是：

1. 前向前对当前层参数做 AllGather。
2. 完成该层前向。
3. 反向时再次确保参数可用。
4. 梯度产生后做 ReduceScatter，把梯度切回各卡所属分片。
5. 只由拥有该分片的 rank 更新对应优化器状态和参数。

通信模式因此从“全量梯度 All-Reduce”为主，转成“参数 AllGather + 梯度 ReduceScatter”为主。常见经验结论是：**ZeRO-3 的通信带宽成本约为普通 DP 的 1.5 倍量级**。不是所有场景都严格等于 1.5，但工程上可以把它当成一个有用的估算。

### 一层参数在 ZeRO-3 中如何流动

下面用一个层级时间线说明：

| 时刻 | 动作 | 显存状态变化 |
|---|---|---|
| 前向开始前 | AllGather 当前层参数 | 当前层参数临时变完整 |
| 前向计算中 | 只保留当前层活跃数据 | 其他层参数可不常驻 |
| 反向开始 | 需要该层参数参与梯度计算 | 再次保证该层参数就绪 |
| 梯度产出后 | ReduceScatter 梯度 | 每卡只保留自己那份梯度 |
| 更新完成 | 释放临时完整参数 | 回到分片常驻状态 |

这里最关键的思想是：**参数不是永远完整存在，而是在需要时短暂“拼起来”**。这就是 ZeRO-3 显存能大幅下降的根本原因。

### 7B 模型的量级估算

7B 参数模型，若按 FP16 参数估算：

$$
\Psi = 7\times 10^9 \times 2 \text{ Bytes} \approx 14 \text{ GB}
$$

普通 DP 每卡模型状态：

$$
16\Psi \approx 16 \times 14 = 224 \text{ GB}
$$

很多工程材料会给出约 104GB 或 112GB 一类数字，差异来自是否把“参数量”记作元素数、是否把某些状态按半精度或实际实现方式折算。本文采用题目给定口径：**7B 在 8 卡 ZeRO-3 下可近似从约 112GB/卡降到约 14GB/卡**。工程上应把这视为“量级估算”，而不是字节级精算。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实 GPU 通信库，而是用列表切片模拟 ZeRO 各阶段的每卡状态内存。目的不是复现 DeepSpeed，而是验证公式。

```python
from dataclasses import dataclass

@dataclass
class ZeroMemory:
    psi_gb: float   # FP16 参数字节总量，以 GB 计
    dp: int         # 数据并行卡数

    def stage0(self) -> float:
        # 参数 2Psi + 梯度 2Psi + 优化器状态 12Psi
        return 16 * self.psi_gb

    def stage1(self) -> float:
        return 4 * self.psi_gb + 12 * self.psi_gb / self.dp

    def stage2(self) -> float:
        return 2 * self.psi_gb + 14 * self.psi_gb / self.dp

    def stage3(self) -> float:
        return 16 * self.psi_gb / self.dp


m = ZeroMemory(psi_gb=14.0, dp=8)

s0 = m.stage0()
s1 = m.stage1()
s2 = m.stage2()
s3 = m.stage3()

# 单调下降
assert s0 > s1 > s2 > s3

# 8 卡时 Stage 3 等于 2Psi
assert abs(s3 - 28.0) < 1e-9

# 与公式一致
assert abs(s1 - (4 * 14 + 12 * 14 / 8)) < 1e-9
assert abs(s2 - (2 * 14 + 14 * 14 / 8)) < 1e-9

print({
    "stage0_gb_per_gpu": s0,
    "stage1_gb_per_gpu": s1,
    "stage2_gb_per_gpu": s2,
    "stage3_gb_per_gpu": s3,
})
```

如果按题目给定的“7B 模型每卡约 112GB 到 14GB”口径，可以把 $\Psi$ 理解为另一种实现/统计下的折算单位；公式结构不变，变化的是底层字节口径。要点是：**ZeRO 关注的是相对缩放关系，不是某一份实现里的固定常数。**

下面再给一个更接近真实训练流程的伪代码，展示 ZeRO-3 的关键挂钩点：

```python
class Zero3Param:
    def __init__(self, local_shard, owner_rank):
        self.local_shard = local_shard
        self.owner_rank = owner_rank
        self.full_param_cache = None

    def all_gather_full_param(self, group):
        # 通信后得到当前层完整参数
        self.full_param_cache = group.all_gather(self.local_shard)
        return self.full_param_cache

    def free_full_param(self):
        self.full_param_cache = None

    def reduce_scatter_grad(self, full_grad, group):
        # 梯度回收为各 rank 自己拥有的分片
        return group.reduce_scatter(full_grad)


def forward_layer(layer, x, group):
    weight = layer.weight.all_gather_full_param(group)
    y = layer.compute_forward(x, weight)
    layer.weight.free_full_param()
    return y


def backward_layer(layer, dy, group):
    weight = layer.weight.all_gather_full_param(group)
    full_grad = layer.compute_grad(dy, weight)
    local_grad = layer.weight.reduce_scatter_grad(full_grad, group)
    layer.optimizer.update_local_shard(layer.weight.local_shard, local_grad)
    layer.weight.free_full_param()
    return local_grad
```

这段伪代码对应三个工程事实：

| 环节 | Stage 2 | Stage 3 |
|---|---|---|
| 参数是否常驻完整副本 | 是 | 否 |
| 前向前是否需要 AllGather 参数 | 通常不需要 | 需要 |
| 反向后是否 ReduceScatter 梯度 | 需要 | 需要 |
| 优化器状态是否按 rank 分片 | 是 | 是 |

### 真实工程例子

Hugging Face Alignment Handbook 的多套 7B 级全参数微调 recipe，常见做法就是 `8 x A100 80GB + DeepSpeed ZeRO-3`。这类配置说明了一点：**ZeRO-3 不是论文里的理论技巧，而是单节点多卡训练 7B 以上模型的主力工程方案之一**。

如果不用 ZeRO-3，很多全参数训练任务会先死在模型状态显存上；用了 ZeRO-3，状态显存降下来了，剩余显存才能留给：

- 激活值
- attention 临时缓冲
- CUDA kernel workspace
- 数据加载与通信缓存

---

## 工程权衡与常见坑

ZeRO 从来不是“白拿显存”。它拿显存换通信，拿简单实现换更复杂的调度。

### 主要权衡

| 维度 | 普通 DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 显存节省 | 低 | 中 | 高 | 最高 |
| 通信压力 | 基线 | 略增 | 中等 | 最高 |
| 实现复杂度 | 低 | 低到中 | 中 | 高 |
| 对高速互联依赖 | 低 | 低 | 中 | 高 |

### 常见坑 1：网络带宽不够

ZeRO-3 最常见的失败模式不是 OOM，而是吞吐掉得太厉害。因为层级 AllGather/ReduceScatter 很频繁，如果机器之间的互联带宽或延迟差，GPU 会经常等通信。

判断标准很实际：

- 单机 NVLink / 高速 PCIe 环境，ZeRO-3 通常更可行。
- 多机但网络一般，ZeRO-2 往往更稳。
- 如果通信不能与计算重叠，ZeRO-3 的理论显存优势会被训练速度惩罚抵消。

### 常见坑 2：只看模型状态，不看激活显存

很多人看到“ZeRO-3 把 7B 压到十几 GB”就以为 24GB 显卡也能轻松全参训练。这个结论不完整。训练显存不只有模型状态，还有激活。序列长度、batch size、注意力实现方式都会显著影响激活显存。

所以更准确的判断应是：

$$
\text{总显存} = \text{模型状态} + \text{激活} + \text{通信缓冲} + \text{框架额外开销}
$$

ZeRO 只直接优化了“模型状态”这一项。

### 常见坑 3：参数切分不均匀

真实模型不是所有层都等大。若切分粒度设计不好，某些 rank 可能总是拿到更大的 shard，导致：

- 显存峰值不均衡
- 通信耗时不均衡
- 某些 rank 成为慢节点，其他卡空等

这类问题通常需要依赖框架的 bucket、prefetch、contiguous memory 配置，以及更合理的层分组策略。

### 常见坑 4：重叠做得不够

“通信重叠”是指把通信藏在计算后面，不让 GPU 干等。白话讲，就是在算当前层时，提前把下一层需要的数据拉过来。

如果没有 overlap 和 prefetch，ZeRO-3 的理论节省会变成一连串同步栅栏。工程上通常需要关注：

- bucket 大小是否合理
- 是否开启 overlap communication
- 是否支持参数预取
- 是否使用更高效的 collectives 实现

### 常见坑 5：检查点与恢复更复杂

ZeRO-3 下，参数和优化器状态是分片存的。保存 checkpoint 时，不再是“每卡一份完整模型”，而是“多卡合起来才是完整状态”。这会影响：

- checkpoint 格式
- 单卡加载流程
- 推理导出流程
- 故障恢复复杂度

所以训练能跑通，不等于部署与恢复链路已经打通。

---

## 替代方案与适用边界

ZeRO 不是越高阶段越好，而是要看你的瓶颈在哪里。

### 什么时候停在 ZeRO-1 或 ZeRO-2

如果模型不算特别大，但 Adam 状态已经成为主要负担，ZeRO-1 往往就够用。它改动小，收益直接。

如果显存还差一点，但网络条件一般，ZeRO-2 常常是比较稳的平衡点。因为它已经切掉了梯度和优化器状态，但没有把参数变成按层动态收集，所以通信压力明显低于 ZeRO-3。

### 什么时候用 ZeRO-3

以下场景更适合 ZeRO-3：

| 条件 | 是否适合 ZeRO-3 |
|---|---|
| 单节点多卡，互联快 | 适合 |
| 模型状态显存是核心瓶颈 | 适合 |
| 需要全参数训练 7B+ 模型 | 很适合 |
| 机器间网络慢、延迟高 | 谨慎 |
| 模型本来就能轻松装下 | 不一定值得 |

### ZeRO 与其他方案的关系

| 方案 | 解决什么问题 | 典型适用边界 |
|---|---|---|
| 普通 DP | 提升吞吐 | 模型能放下，追求简单 |
| ZeRO-1/2/3 | 去掉数据并行冗余状态 | 显存先卡住 |
| 张量并行 | 把单层计算拆到多卡 | 单层太大，单卡算不下 |
| 流水线并行 | 把不同层放到不同卡 | 模型层数深、可切 stage |
| LoRA/PEFT | 少训练一部分参数 | 不追求全参数更新 |

这里要区分两类问题：

1. **模型状态放不下**：优先想 ZeRO。
2. **单层算子本身放不下或算不动**：优先想张量并行。
3. **想省资源但接受能力上限**：优先想 PEFT。

一个实用决策可以写成：

- 小模型或显存充裕：普通 DP。
- 中等模型，显存紧张但网络一般：ZeRO-1 或 ZeRO-2。
- 7B 以上全参训练，且有高速互联：ZeRO-3。
- ZeRO-3 仍不够，或带宽压力过大：再结合 offload、量化通信或其他并行策略。

---

## 参考资料

- Rajbhandari, Samyam, et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC20, 2020. 主要支持 ZeRO 三阶段设计与理论内存分析。https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap379.html
- DeepSpeed Team. *ZeRO-3 Offload: Democratizing Billion-Scale Model Training*. 2021. 主要支持 ZeRO-3 的工程通信代价、offload 与训练实践。https://www.deepspeed.ai/2021/03/07/zero3-offload.html
- Emergent Mind. *Zero Redundancy Optimizer (ZeRO)*. 主要支持分阶段公式、通信模式与记忆框架。https://www.emergentmind.com/topics/zero-redundancy-optimizer-zero
- mbrenndoerfer. *ZeRO Optimization Stages: Optimizer, Gradient, and Parameter Partitioning*. 主要支持 7B/8 卡的直观数值例子。https://mbrenndoerfer.com/writing/zero-optimization-stages-optimizer-gradient-parameter-partitioning
- Oboe Learn. *Zero Redundancy Optimizer Distributed GPU Training for LLMs*. 主要支持面向初学者的直观定义。https://oboe.com/learn/distributed-gpu-training-for-llms-3w49kt/zero-redundancy-optimizer-distributed-gpu-training-for-llms-3
- Hugging Face Alignment Handbook / DeepWiki. *DeepSpeed ZeRO-3*. 主要支持 7B 级模型在 `8 x A100 80GB` 上的工程落地示例。https://deepwiki.com/huggingface/alignment-handbook/5.1.1-deepspeed-zero-3

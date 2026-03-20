## 核心结论

Megatron-LM 的核心不是“某一种并行”，而是把训练任务拆成三个正交维度同时做：

| 并行方式 | 白话解释 | 主要切分对象 | 主要解决的问题 | 常见部署位置 |
|---|---|---|---|---|
| DP（Data Parallelism，数据并行） | 每张卡跑同一个模型，但喂不同样本 | 批次 batch | 提升总吞吐 | 整个集群 |
| TP（Tensor Parallelism，张量并行） | 一层里的大矩阵拆到多张卡一起算 | 参数矩阵/单层计算 | 单层太大，单卡放不下 | 单机内 |
| PP（Pipeline Parallelism，流水线并行） | 不同层段放到不同卡，像工厂流水线一样传递激活 | 模型深度/层 | 整个模型太深，单机放不下 | 常跨机 |

Megatron-LM 的工程经验可以压缩成一句公式：

$$
\text{total\_gpus} = \text{DP} \times \text{TP} \times \text{PP} \times \text{CP}
$$

如果不使用上下文并行 CP（Context Parallelism，按序列长度切分），就退化成：

$$
\text{total\_gpus} = \text{DP} \times \text{TP} \times \text{PP}
$$

真正重要的不是把公式背下来，而是理解“谁放在哪”。TP 需要最频繁的层内通信，所以应尽量限制在 NVLink/NVSwitch 连接的单机内；PP 可以跨节点，因为它传的是层间激活，延迟容忍度更高；DP 最后覆盖在 TP+PP 之上，把整个模型副本横向复制，换吞吐。

Megatron-Turing NLG 530B 的公开配置就是这个思想的工业实现：单个模型副本跨 280 张 A100，采用 8-way TP 和 35-way PP，再在此基础上继续用 DP 扩到数千卡规模。这个配置本身已经说明三件事：TP 不会无限做大，PP 会跨节点拉长模型，DP 才是最终扩容的“外层壳”。

---

## 问题定义与边界

问题定义很直接：当单卡显存装不下模型、单机吞吐又不够时，怎样把模型和训练任务映射到多卡、多机上，同时保证通信成本没有把收益吃掉。

这里有三个边界要先说清：

| 维度 | 约束来源 | 判断条件 | 超界后的典型后果 |
|---|---|---|---|
| TP | 节点内带宽 | 通常不应超过单机 GPU 数 | all-reduce/all-gather 变慢，算得越分越亏 |
| PP | 层数与调度 | 阶段数不能明显大于可均匀切分的层段数 | 阶段负载不均，流水线泡泡变大 |
| DP | 跨副本同步带宽 | 梯度同步时间不能压过计算时间 | 扩卡不再线性，吞吐提升变差 |

Megatron Bridge 文档给出的数据并行大小计算是：

$$
\text{data\_parallel\_size}=
\frac{\text{world\_size}}
{\text{tensor\_model\_parallel\_size}\times
\text{pipeline\_model\_parallel\_size}\times
\text{context\_parallel\_size}}
$$

这条公式的含义很朴素：先把世界规模 `world_size` 按 TP、PP、CP 切掉，剩下的才是 DP。也就是说，DP 不是先拍脑袋定出来的，而是“总卡数减去模型并行需求后的余量”。

玩具例子：8 张卡，如果设 `TP=2`、`PP=2`、`CP=1`，那么

$$
DP = 8 / (2 \times 2 \times 1) = 2
$$

这表示整个系统里存在 2 份完整的“TP×PP 模型副本”。如果你硬写成 `DP=3`，那总需求就变成 $2\times2\times3=12$ 张卡，初始化阶段就会失败，因为世界大小根本对不上。

更贴近配置的例子：32 张卡，设 `TP=2`、`PP=4`、`CP=2`，则

$$
DP = 32 / (2 \times 4 \times 2)=2
$$

这个结果的工程含义是：每个“2 路 TP + 4 段 PP + 2 路 CP”的模型布局，再复制 2 份做 DP。新手最容易混淆的点就在这里：DP 不是和 TP、PP 并列摆放，而是包在它们外面。

---

## 核心机制与推导

先分开看三个维度。

TP 的本质，是把一层里的大矩阵乘法拆到多卡上。白话说，一张卡算不完一个超大线性层，就让多张卡各算一部分，然后再把结果拼起来。它适合处理“单层太宽”的问题，比如隐藏维度非常大时的 MLP 和注意力投影。

PP 的本质，是把模型按层切成多段。白话说，GPU0 放前几层，GPU1 放中间几层，GPU2 放后几层，激活像接力棒一样一段段传下去。它适合处理“整体太深或太大”的问题。

DP 的本质，是复制整个模型副本，然后每个副本处理不同样本，最后同步梯度。白话说，这不是省显存，而是加吞吐。

为什么要把 TP 放单机、PP 放跨机？因为通信对象不一样。

| 维度 | 传什么 | 频率 | 对带宽/延迟的敏感性 |
|---|---|---|---|
| TP | 层内部分结果、梯度 | 很高 | 对高带宽极度敏感 |
| PP | 激活、反向梯度 | 中等 | 对延迟敏感，但比 TP 容忍度高 |
| DP | 梯度/参数同步 | 迭代级 | 对总带宽敏感 |

所以 Megatron-LM 的经典策略是：节点内做 TP，节点间串 PP，最外层再做 DP。

再看流水线效率。PP 最大的问题不是“能不能切”，而是“切完有没有空转”。NVIDIA 在 1F1B（One-Forward-One-Backward，前向和反向交替推进）调度里给出过两个关键公式。若一批次里有 $m$ 个微批次（microbatches），流水线阶段数为 $p$，单个微批的前向和反向时间分别是 $t_f,t_b$，那么流水线泡泡时间为：

$$
t_{pb}=(p-1)\cdot(t_f+t_b)
$$

泡泡占比近似是：

$$
\frac{t_{pb}}{t_{id}}=\frac{p-1}{m}
$$

其中 $t_{id}=m\cdot(t_f+t_b)$ 是理想无空转时间。这个式子非常重要，因为它直接说明：微批次数 $m$ 越大，相同 PP 阶段数下泡泡占比越小。

玩具例子：4 个流水线阶段、4 个微批。时间线可以写成下面这样：

| 时间片 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| 1 | F1 |  |  |  |
| 2 | F2 | F1 |  |  |
| 3 | F3 | F2 | F1 |  |
| 4 | F4 | F3 | F2 | F1 |
| 5 | B1 | F4 | F3 | F2 |
| 6 | B2 | B1 | F4 | F3 |
| 7 | B3 | B2 | B1 | F4 |
| 8 | B4 | B3 | B2 | B1 |

`F1` 表示第 1 个微批前向，`B1` 表示第 1 个微批反向。这个表不是 Megatron 的完整实现细节，但足够解释 1F1B 的关键思想：暖机后尽量让每个阶段都同时有活干。

当物理流水线阶段数固定时，还可以引入虚拟流水线阶段 `v`。白话说，一张物理卡上再切出多个更小的“逻辑层段”，交错执行，以缩短等待。NVIDIA 给出的简化结论是：

$$
\text{bubble time}=\frac{(p-1)\cdot(t_f+t_b)}{v}
$$

也就是虚拟阶段数越大，泡泡越小；但代价是通信次数更多，调度也更复杂。这正是为什么“虚拟阶段不是默认越大越好”。

真实工程例子：530B 训练中，8 路 TP 基本对应单机 8 张 A100 的高速互联范围，35 路 PP 则把 105 层模型切到 35 个阶段上，每阶段平均承担约 3 层，随后再把整个 280 卡模型副本复制做 DP。这个布局不是数学上唯一可行，而是通信代价和模型结构共同决定的工程折中。

---

## 代码实现

下面先用一个可运行的 Python 小程序，把并行度是否合法讲清楚。它不依赖 Megatron，但逻辑和框架初始化时的检查是一致的。

```python
def calc_dp(world_size, tp=1, pp=1, cp=1):
    denom = tp * pp * cp
    assert denom > 0
    assert world_size % denom == 0, (
        f"world_size={world_size} 不能被 tp*pp*cp={denom} 整除"
    )
    return world_size // denom


# 玩具例子：8 卡，TP=2，PP=2，CP=1，则 DP=2
assert calc_dp(8, tp=2, pp=2, cp=1) == 2

# 文档例子：32 卡，TP=2，PP=4，CP=2，则 DP=2
assert calc_dp(32, tp=2, pp=4, cp=2) == 2

# 530B 风格的单模型副本：8-way TP * 35-way PP = 280 GPUs
assert 8 * 35 == 280

# 非法配置：12 不能被 2*4*2=16 整除
try:
    calc_dp(12, tp=2, pp=4, cp=2)
    assert False, "这里应该报错"
except AssertionError:
    pass

print("parallel layout checks passed")
```

在 Megatron Bridge 里，配置入口大致如下：

```python
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.config import ConfigContainer, OptimizerConfig

model_config = GPTModelProvider(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,  # 给 1F1B 交错调度用
    context_parallel_size=2,
    # 其他模型参数略
)

optimizer_config = OptimizerConfig(
    optimizer="adam",
    use_distributed_optimizer=True,
)

config = ConfigContainer(
    model=model_config,
    optimizer=optimizer_config,
)
```

这里最关键的不是 API 名字，而是配置含义：

| 配置项 | 作用 | 典型思路 |
|---|---|---|
| `tensor_model_parallel_size` | 每层切几份 | 尽量限制在单机内 |
| `pipeline_model_parallel_size` | 模型切成几段 | 超单机容量时增加 |
| `virtual_pipeline_model_parallel_size` | 每个物理阶段再切几段 | 用于降低泡泡 |
| `context_parallel_size` | 按序列长度再切 | 长上下文时开启 |

`DP` 不需要你直接手算后再写死；框架根据 `world_size / (TP×PP×CP)` 自动推导。这一点很重要，因为很多初始化挂起问题，本质上就是 rank 分组和你以为的不一致。

---

## 工程权衡与常见坑

第一类坑，是把 TP 开得过大。TP 的通信最重，尤其在注意力和 MLP 的张量切分上会反复做集合通信。如果把 TP 扩到跨节点，显存压力是降了，但通信很容易把收益吃掉。所以经验上先问一句：模型是不是“单层太宽”，以及这些卡之间是不是 NVLink/NVSwitch 直连。

第二类坑，是 PP 阶段切得不均匀。比如 36 层模型硬切成 7 段，就很容易出现有的阶段 5 层、有的阶段 6 层，还可能把 embedding、loss head 这些特殊层压到边缘 rank，导致首尾阶段比中间阶段更慢。PP 不是能切就行，而是要尽量让每段计算量接近。

第三类坑，是微批配置不对。1F1B 想高效，微批数必须足够大；而交错流水线调度还有“微批数要与流水线并行度匹配”的约束。NVIDIA 的公开说明里给过明确例子：4 个设备时，微批数应是 4 的整数倍。否则泡泡大、调度乱，轻则利用率差，重则直接出错。

第四类坑，是把多节点 PP 的故障误判成网络问题。Megatron-LM 的一个公开 issue 就展示了这种情况：单节点 DP、TP、PP 都正常，多机 DP 也正常，但一上多节点 PP 就报错；日志里同时出现 `pipeline-model-parallel size: 2`、`Number of virtual stages per pipeline stage: None`，以及“non-interleaved schedule does not support overlapping p2p communication”的警告。这类故障通常说明问题不只在连通性，还可能出在 PP 调度、虚拟阶段设置、rank 映射、批次切分这些更上层的配置。

可以直接用下面这个检查表排查：

| 检查项 | 为什么重要 | 快速判断方法 |
|---|---|---|
| TP 是否限制在单机内 | TP 对带宽最敏感 | `TP <= 每节点GPU数` |
| `world_size` 是否能被 `TP×PP×CP` 整除 | 决定 DP 组能否正确生成 | 启动前手算一遍 |
| 微批数是否足够大且与 PP 匹配 | 决定泡泡和 1F1B 调度合法性 | 检查 `global_batch / (dp * micro_batch)` |
| 是否需要虚拟阶段 | 降低 PP 泡泡 | 观察各阶段空转比例 |
| 多节点 PP 日志里的 rank 分组是否符合预期 | 防止组映射错误 | 核对启动日志中的 `tp/pp/dp size` |

一句话概括：TP 主要怕带宽不够，PP 主要怕调度不好，DP 主要怕同步拖慢。混合并行难，不是因为概念多，而是因为这三个瓶颈来自完全不同的层次。

---

## 替代方案与适用边界

不是所有模型都值得上“DP+TP+PP”三维混合并行。选择应从问题规模倒推，而不是从框架功能正推。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| DP-only | 模型能放进单卡或单机 | 最简单，最稳 | 显存上限硬约束明显 |
| DP+TP | 模型单层很宽，但深度尚可 | 解决单层显存问题 | 对节点内带宽要求高 |
| DP+PP | 模型层很多，易按深度切 | 可跨机扩展 | 泡泡和负载均衡麻烦 |
| DP+TP+PP | 千亿级以上大模型 | 扩展性最强 | 配置、调度、排障最复杂 |

如果模型还能装进单机，优先考虑 DP-only，加上分布式优化器或分片式数据并行，通常更省事。白话说，能不切模型就别切模型，因为模型一旦被切，通信和调度复杂度就会直接抬升一个量级。

如果模型主要问题是“层太宽”，比如大隐藏维度把线性层撑爆，那么优先加 TP。因为这时候 PP 只能把层段分开，不能解决单层矩阵本身太大。

如果模型主要问题是“总层数太深、整体参数太大”，单机装不下，即使单层还能勉强算，那么再加 PP。PP 的价值不是替代 TP，而是接手“整条网络放不下”的部分。

真实工程里，一个常见决策流程是：

1. 先用 DP-only 跑基线。
2. 单卡放不下时，加 TP。
3. 单机还放不下时，加 PP。
4. 序列太长时，再考虑 CP。

这也是 Megatron Core 文档给出的推荐顺序。它背后的逻辑很朴素：先用最简单的并行方式解决问题，只有在当前维度失效时，才增加下一维。

---

## 参考资料

- NVIDIA Megatron Core, Parallelism Strategies Guide: https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/parallelism-guide.html
- NVIDIA Megatron Bridge, Parallelisms Guide: https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html
- NVIDIA Technical Blog, Scaling Language Model Training to a Trillion Parameters Using Megatron: https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
- Microsoft Research, Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B: https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/
- NVIDIA Megatron-LM Issue #1525, Multiple Node PP errors: https://github.com/NVIDIA/Megatron-LM/issues/1525

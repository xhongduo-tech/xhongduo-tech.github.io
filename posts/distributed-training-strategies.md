## 核心结论

分布式训练的核心不是“把 GPU 数量堆上去”，而是先回答一个更基本的问题：当前瓶颈到底是**显存**、**单步计算时间**，还是**跨卡通信**。数据并行、模型并行、流水线并行，分别对应三种不同切法。

- 数据并行（Data Parallel, DP）：每张卡保存一份完整模型，处理不同样本，最后同步梯度。白话说，就是“模型不拆，数据拆开跑”。
- 模型并行通常指张量并行（Tensor Parallel, TP）：把一个层内部的参数和计算切到多张卡。白话说，就是“一个大层太大，一张卡装不下，就横着切”。
- 流水线并行（Pipeline Parallel, PP）：把模型按深度切成多个阶段，不同卡负责不同层段。白话说，就是“前几层给第一组卡，后几层给第二组卡，让微批次像流水线一样往后传”。

三者不是互斥关系，而是可组合关系。常用资源关系可以写成：

$$
G = D \times P \times S
$$

其中 $G$ 是总 GPU 数，$D$ 是数据并行组大小，$P$ 是张量并行大小，$S$ 是流水线阶段数。

对初学者，最实用的判断顺序是：

| 问题 | 优先策略 | 原因 |
| --- | --- | --- |
| 单卡能放下模型，但训练太慢 | 数据并行 | 最简单，吞吐提升最直接 |
| 单卡放不下某些大层 | 模型并行 | 先解决“装不下” |
| 模型层数很深，整体放不下 | 流水线并行 | 按深度切分更自然 |
| 既放不下又太慢 | 混合并行 | DP + TP 或 DP + PP |
| 超大模型训练 | 3D 并行 | 同时解决显存、吞吐、扩展性 |

一个直接结论是：**数据并行是默认起点，模型并行和流水线并行是突破单卡显存边界的主要手段，真正的大模型训练通常使用混合并行。**

---

## 问题定义与边界

分布式训练要解决的不是一个问题，而是三个问题：

1. 模型参数、激活值、优化器状态太大，单卡显存放不下。
2. 单卡算得太慢，训练周期过长。
3. GPU 变多后，通信和调度开始吞掉收益。

这里先明确边界。

### 1. 数据并行解决的是“样本维度扩展”

每张卡上都是同一个模型副本，只是各自拿到不同的小批次数据。前向和反向各算各的，最后做一次梯度同步。它的优点是实现简单、框架支持最好，但它不减少单卡模型占用。也就是说，**模型必须先能装进一张卡**，否则纯数据并行没有意义。

### 2. 模型并行解决的是“单层或单模块太大”

例如一个巨大的线性层，权重矩阵过大，单卡放不下，或者单卡 GEMM 计算太重。这时把层切成多份，让多张卡协同完成一次前向和反向。它降低单卡参数和部分激活压力，但代价是层内通信显著增加，尤其依赖高速互联。

### 3. 流水线并行解决的是“模型深度太大”

如果模型天然是按层堆叠的，比如 Transformer block 重复很多层，那么按深度切成多个阶段最自然。每个阶段只保存自己负责的层。问题在于，不同阶段不能总是同时忙碌，会出现空档，这个空档叫 **pipeline bubble**。白话说，就是“流水线还没灌满时，有些卡会等活”。

### 4. 资源边界与组合边界

对于混合并行，一个常见的资源关系是：

$$
G = D \times P \times S
$$

这不是一个理论推导式，而是部署时的分组约束。比如有 16 张卡，如果设置：

- $D = 2$
- $P = 4$
- $S = 2$

那么总卡数就是：

$$
G = 2 \times 4 \times 2 = 16
$$

这意味着：
- 每个流水线阶段内，有 4 张卡做张量并行；
- 同样的整套模型结构复制 2 份做数据并行；
- 总体形成一个 3D 并行结构。

### 5. 一个最小边界例子

假设有 4 张 GPU，要训练一个中等规模 Transformer：

- 方案 A：4 路数据并行。前提是模型单卡能放下。
- 方案 B：2 路流水线并行 + 2 路数据并行，即 $D=2, S=2, P=1$。
- 方案 C：2 路张量并行 + 2 路数据并行，即 $D=2, P=2, S=1$。

这三个方案都用 4 张卡，但适用边界不同：

| 方案 | 能否降低单卡模型显存 | 主要通信 | 适用前提 |
| --- | --- | --- | --- |
| 纯 DP | 否 | 梯度 AllReduce | 单卡能装下完整模型 |
| DP + TP | 能 | 层内分片通信 + 梯度同步 | 层足够大，卡间互联快 |
| DP + PP | 能 | 阶段间激活传输 + 梯度同步 | 模型按层堆叠明显 |

所以，问题定义不能只写成“多卡训练怎么更快”，而应该写成：**在给定显存、带宽和模型结构下，应该沿哪个维度切分计算。**

---

## 核心机制与推导

这一节只讨论最核心的训练过程，不展开框架细节。

### 1. 数据并行的机制

设一份模型参数为 $\theta$，一个全局 batch 被拆到 $D$ 张卡，每张卡拿到一个局部 batch。第 $i$ 张卡算出的局部梯度记为 $g_i$。同步后的全局梯度通常是：

$$
g = \frac{1}{D}\sum_{i=1}^{D} g_i
$$

这一步通常用 AllReduce 实现。白话说，就是“每张卡先各算各的，再把梯度求平均，保证参数更新一致”。

如果局部 batch size 是 $b$，梯度累积步数是 $m$，数据并行大小是 $D$，那么全局 batch size 为：

$$
B_{\text{global}} = b \times m \times D
$$

这条式子很重要，因为很多“训练不稳定”其实不是并行本身的问题，而是你把全局 batch 放大后，学习率、warmup、梯度裁剪没有一起调整。

### 2. 张量并行的机制

以一个线性层为例：

$$
Y = XW
$$

若权重矩阵 $W \in \mathbb{R}^{h \times 4h}$ 太大，可以沿列切成两块：

$$
W = [W_1, W_2]
$$

于是：

$$
Y = [XW_1, XW_2]
$$

这表示两张卡可以并行计算输出的一部分，然后再拼接或在后续阶段做归约。白话说，就是“同一个层里的矩阵乘法，被拆给多张卡一起算”。

张量并行的本质收益是：
- 单卡参数更少；
- 单卡矩阵计算规模更小；
- 能突破单卡显存边界。

但代价是：
- 每层前向或反向都可能需要通信；
- 通信频率比数据并行高得多；
- 强依赖 NVLink、NVSwitch 等高速互联。

### 3. 流水线并行的机制

假设模型有 8 层，可以切成 2 个阶段：

- Stage 0：第 1 到 4 层
- Stage 1：第 5 到 8 层

如果一次只送入 1 个 batch，那么 Stage 1 开始工作时，Stage 0 可能已经空了；反向开始时，又会出现等待。为减少空档，通常把一个 batch 再切成多个 **micro-batch**。白话说，micro-batch 就是“为了填满流水线，把一个大批次拆成更小的连续小包”。

设流水线阶段数为 $S$，micro-batch 数为 $M$。经验上，$M$ 越大，bubble 占比越小。一个常见近似是：

$$
\text{bubble ratio} \approx \frac{S-1}{M+S-1}
$$

它不是所有调度下都严格成立，但足够说明方向：**阶段数固定时，micro-batch 越多，空转越少。**

### 4. 玩具例子：4 卡、2 阶段、2 路数据并行

设总共有 4 张卡，配置为：

- $D = 2$
- $S = 2$
- $P = 1$

所以：

$$
G = 2 \times 2 \times 1 = 4
$$

可以分成两组数据并行副本：

- 副本 A：GPU0 负责 Stage 0，GPU1 负责 Stage 1
- 副本 B：GPU2 负责 Stage 0，GPU3 负责 Stage 1

再把每个全局 batch 切成 8 个 micro-batch。训练过程是：

1. GPU0 和 GPU2 先处理 micro-batch 1。
2. 它们把激活传给 GPU1 和 GPU3。
3. 同时 GPU0 和 GPU2 继续处理 micro-batch 2。
4. 流水线逐步灌满后，各阶段交错执行前向和反向。
5. 一个 batch 完成后，A 和 B 两个数据并行副本之间做梯度 AllReduce。

这个例子说明了三件事：
- 流水线并行解决“层深切分”；
- 数据并行解决“吞吐扩展”；
- micro-batch 用来减少流水线空档。

### 5. 真实工程例子：Megatron-LM + DeepSpeed 训练大模型

真实大模型训练很少只用单一策略。以 GPT 类模型为例，常见做法是：

- 节点内用 Tensor Parallel，因为节点内带宽高；
- 跨节点再加 Pipeline Parallel，按层切深度；
- 最外层叠加 Data Parallel，提高吞吐；
- 再叠加 ZeRO 对优化器状态和梯度做分片，进一步省显存。

为什么这么组合？因为大模型的显存构成不只参数，还有激活值、梯度、优化器状态。只做 TP 或 PP，往往还不够；只做 DP，则根本装不下。工程上最终追求的是：**让通信尽量发生在快链路上，让最重的同步不要跨最慢的网络。**

---

## 代码实现

下面先给一个不依赖 GPU 的玩具代码，模拟三种切分方式在逻辑上的区别。它不是深度学习框架代码，但能帮助建立正确直觉。

```python
from dataclasses import dataclass

@dataclass
class ParallelPlan:
    data_parallel: int
    tensor_parallel: int
    pipeline_stages: int

    @property
    def world_size(self) -> int:
        return self.data_parallel * self.tensor_parallel * self.pipeline_stages

def global_batch_size(micro_batch_size: int, micro_batches: int, data_parallel: int) -> int:
    return micro_batch_size * micro_batches * data_parallel

def dp_memory_per_gpu(model_params_mb: int) -> int:
    # 数据并行每张卡都有完整模型
    return model_params_mb

def tp_memory_per_gpu(model_params_mb: int, tensor_parallel: int) -> int:
    assert tensor_parallel > 0
    # 简化：假设参数平均切分
    return model_params_mb // tensor_parallel

def pp_memory_per_gpu(model_params_mb: int, pipeline_stages: int) -> int:
    assert pipeline_stages > 0
    # 简化：假设层按阶段平均切分
    return model_params_mb // pipeline_stages

plan = ParallelPlan(data_parallel=2, tensor_parallel=1, pipeline_stages=2)
assert plan.world_size == 4

# 全局 batch = micro batch * micro batches * DP
assert global_batch_size(4, 8, 2) == 64

# 一个 16000 MB 的模型
model_params_mb = 16000

# 纯 DP 不降模型显存
assert dp_memory_per_gpu(model_params_mb) == 16000

# 2 路 TP 后，每卡参数约减半
assert tp_memory_per_gpu(model_params_mb, 2) == 8000

# 2 段 PP 后，每卡负责半个模型
assert pp_memory_per_gpu(model_params_mb, 2) == 8000

print("All assertions passed.")
```

这个代码表达的是逻辑，不是严格显存公式。真实训练中还要考虑：
- 激活值
- 梯度
- Adam 等优化器状态
- 混合精度后的主权重与缩放状态
- 检查点重计算（activation checkpointing）

### 1. 新手版伪代码：数据并行

```python
for batch in loader:
    local_loss = model(batch_on_this_rank)
    local_loss.backward()
    all_reduce_gradients(model.parameters())
    optimizer.step()
    optimizer.zero_grad()
```

关键点只有一个：**每张卡算的是不同数据，但更新前必须同步梯度。**

### 2. 新手版伪代码：流水线并行

```python
for step in range(num_steps):
    loss = engine.train_batch(data_iter=train_iter)
```

如果用 DeepSpeed 的 `PipelineModule`，你通常不会手写复杂的前向/反向交错逻辑。框架会根据：
- `num_stages`
- `micro_batches`
- `partition_method`

自动调度阶段之间的激活传递和反向传播。

### 3. 典型框架配置思路

Megatron-LM 中常见参数是：

- `--tensor-model-parallel-size`
- `--pipeline-model-parallel-size`

例如：

```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
```

如果总卡数是 16，那么剩余维度通常可用于数据并行：

$$
D = \frac{16}{4 \times 2} = 2
$$

于是形成 `TP=4, PP=2, DP=2` 的 3D 并行。

### 4. 分区方法的工程意义

| 分区方法 | 含义 | 适用场景 |
| --- | --- | --- |
| `uniform` | 按层数平均切分 | 每层计算量接近时 |
| `parameters` | 按参数量平衡切分 | 某些层特别大时 |
| `type` | 按层类型做规则切分 | 模型结构有明显模块边界时 |

不要把“层数平均”误以为“负载平均”。在 Transformer 里，不同层的参数量、激活尺寸、注意力开销可能并不完全一致。流水线切分不均，会直接放大 bubble。

---

## 工程权衡与常见坑

分布式训练的难点从来不是“能不能跑起来”，而是“能不能跑得值”。

### 1. 数据并行的常见坑

| 坑 | 现象 | 解决方式 |
| --- | --- | --- |
| 全局 batch 变太大 | 收敛变差，loss 曲线异常 | 联动调整学习率、warmup、梯度裁剪 |
| 通信隐藏失败 | GPU 利用率不高 | 梯度 bucket、通信计算重叠 |
| 单卡显存仍不足 | 一开始就 OOM | 引入 TP、PP 或 ZeRO |

很多人以为“DP 最简单所以最稳”，其实不完全对。DP 的逻辑简单，但当 batch 被放大后，优化超参数不重调，训练照样会坏。

### 2. 张量并行的常见坑

最大问题是通信。层内分片意味着每一层都可能需要：
- all-gather
- reduce-scatter
- all-reduce

如果 TP 组跨节点，性能会明显下滑。所以一个工程原则是：**优先把同一个 TP 组放在同一节点内。**

如果你有 8 卡一机和多机集群，通常更合理的做法是：
- 先在单机内做 TP；
- 再跨机做 DP 或 PP。

### 3. 流水线并行的常见坑

最典型的是 bubble。比如有 4 个阶段，却只给 2 个 micro-batch，那么流水线根本灌不满，很多卡会等。

还有两个常见问题：

| 坑 | 原因 | 处理方式 |
| --- | --- | --- |
| micro-batch 太少 | 流水线空转严重 | 增加梯度累积步数，提高 micro-batch 数 |
| 阶段切分不均 | 某一阶段成为瓶颈 | 调整 partition method，按参数量或计算量重切 |
| 激活传输开销高 | 阶段间数据太大 | 优化阶段边界，减少跨阶段激活大小 |

### 4. 混合并行的常见坑

混合并行最大的问题不是单个策略本身，而是**组网和并行组划分**。

例如 32 张卡：
- 若 TP 组跨节点，层内通信会很贵；
- 若 PP 阶段边界切在激活特别大的位置，阶段间带宽会爆；
- 若 DP 组和 ZeRO 分组冲突，优化器同步会复杂化。

所以实际部署时，必须把“数学上可分”和“网络拓扑上合理”分开看。

### 5. 一个真实工程判断

假设你训练一个 70B 级别的 Transformer：

- 单卡显存绝对不够；
- 单纯 PP 虽然能切深度，但单层 attention/MLP 仍可能过大；
- 单纯 TP 可以切大层，但层数很多时激活和整体参数依然吃紧；
- 纯 DP 根本不成立。

此时典型解法是：
- TP 解决单层太大；
- PP 解决整体模型太深太大；
- DP 解决吞吐；
- ZeRO 或优化器分片进一步压缩状态显存；
- activation checkpointing 减少激活显存，以重算换显存。

这就是为什么大模型训练文档里总在讲“3D 并行”而不是单一并行。

---

## 替代方案与适用边界

三种主策略已经覆盖大部分基础分布式训练，但并不是全部。

### 1. Context Parallelism

Context Parallelism, CP，中文可理解为“上下文并行”，是把长序列在序列维度切开。白话说，就是“一条特别长的输入，不再完整放在一张卡上处理”。

它适合：
- 超长上下文训练
- 序列长度远超普通 LLM 场景
- attention 激活显存成为主瓶颈

如果你的问题主要是“序列太长”，而不是“参数太多”，CP 往往比盲目加 TP 更对症。

### 2. Expert Parallelism

Expert Parallelism, EP，通常出现在 MoE（Mixture of Experts，专家混合）模型中。白话说，就是“不同专家子网络分散到不同卡，只激活其中一部分”。

它适合：
- MoE 模型
- 参数量极大但每次只用部分参数
- 希望扩大参数规模但不线性增加每 token 计算量

### 3. 决策矩阵

| 场景 | 优先选择 | 不优先选择 | 原因 |
| --- | --- | --- | --- |
| 单卡能装下，只想提吞吐 | DP | TP/PP | 额外复杂度不值 |
| 某些层单卡放不下 | TP | 纯 DP | 需要层内切分 |
| 模型层数深，整体放不下 | PP | 纯 DP | 需要按深度切 |
| 长序列导致 attention 爆显存 | CP | 只加 DP | 问题在序列维度 |
| MoE 训练 | EP + DP/TP/PP | 纯 DP | 专家天然可分布 |
| 超大模型稳定训练 | TP + PP + DP | 单一策略 | 单一策略通常不够 |

### 4. 一个实用的优先级建议

对零基础到初级工程师，最实用的顺序不是“把所有并行都学完”，而是：

1. 先理解 DP，因为它决定全局 batch 和梯度同步。
2. 再理解 TP，因为它直接解决“单层装不下”。
3. 再理解 PP，因为它解决“模型整体装不下”和深度扩展。
4. 最后再看 ZeRO、CP、EP，因为这些通常建立在前面概念之上。

如果是实际工程排障，可以用这条顺序：

1. 单卡 OOM 吗？先看 TP、PP、激活重计算、混合精度。
2. 单卡不 OOM，但训练太慢吗？先看 DP。
3. 多卡加上去不提速吗？查通信拓扑和并行组划分。
4. 流水线空转严重吗？增大 micro-batch，重做阶段均衡。
5. 优化器状态太大吗？引入 ZeRO 或状态分片。

结论可以压缩成一句话：**并行策略不是越多越好，而是要让切分维度和真实瓶颈对齐。**

---

## 参考资料

- Megatron Core Parallelism Strategies Guide: https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html
- DeepSpeed Pipeline Parallelism Tutorial: https://www.deepspeed.ai/tutorials/pipeline/
- DeepSpeed Pipeline Parallelism 机制说明: https://deepwiki.com/deepspeedai/DeepSpeed/3.2-pipeline-parallelism
- Microsoft Research, DeepSpeed Extreme-scale Model Training for Everyone: https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/

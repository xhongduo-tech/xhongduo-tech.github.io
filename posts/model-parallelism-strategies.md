## 核心结论

模型并行的目标只有一个：当单张 GPU 放不下模型或单卡吞吐太低时，把同一个模型拆到多张设备上训练。这里最常见的两种拆法是张量并行和流水线并行。

张量并行（Tensor Parallelism，TP）可以直白理解为“同一层内部拆开算”。一层里的大矩阵不再完整放在一张卡上，而是切成几片分给多张卡，每张卡只做自己那一片的矩阵乘法，最后再通过通信把结果拼回去或求和。它直接降低单卡参数和部分激活占用，但代价是层内通信变多，尤其依赖高速互联。

流水线并行（Pipeline Parallelism，PP）可以直白理解为“不同层分段算”。模型前半段放在前几张卡，后半段放在后几张卡，输入被拆成多个微批次（micro-batch，白话就是把一个大 batch 再切成更小的流水块）后，在不同 stage 之间依次传递。它擅长把深层模型铺到多卡上，并通过前后向重叠提升利用率，但会遇到 bubble，也就是部分 GPU 暂时无活可干的空转区间。

对大模型训练来说，单独使用 TP 或 PP 往往不够。工业界更常见的是混合并行：TP 负责层内切分，PP 负责跨层切分，再配合数据并行或序列并行扩展到更多设备。Megatron-LM 的主流路线就是这个组合。

| 维度 | 张量并行 | 流水线并行 |
| --- | --- | --- |
| 划分对象 | 单层内的权重和计算 | 不同层或层组 |
| 典型通信 | all-reduce / all-gather / reduce-scatter | 激活传输、梯度回传 |
| 直接收益 | 降低单卡参数显存 | 把更深模型铺到多卡 |
| 主要代价 | 层内频繁同步 | bubble 与负载不均 |
| 更依赖什么 | 高带宽、低延迟互联 | 合理 stage 划分与微批设置 |

---

## 问题定义与边界

先把问题说清楚。假设有一个线性层：

$$
Y = XA
$$

其中 $X$ 是输入激活，$A$ 是权重矩阵，$Y$ 是输出。模型越大，$A$ 越大；序列越长、batch 越大，$X$ 和中间激活也越大。单卡训练遇到的硬边界通常有三类：

1. 参数放不下。
2. 激活放不下。
3. 即使放得下，单卡训练速度也慢到不可接受。

模型并行主要解决前两类问题，顺带改善第三类，但它不是免费午餐。你用多卡换来更大模型，同时也引入通信、调度和实现复杂度。

一个容易混淆的点是：张量并行和流水线并行解决的不是同一个层面的瓶颈。

- 张量并行解决“单层太宽”。比如一个 Transformer 的 MLP 隐层维度特别大，单层矩阵已经很难完整塞进单卡。
- 流水线并行解决“整体太深”。比如几十层、上百层堆起来后，总参数和激活跨层累积太大。

玩具例子可以先看 2 张卡训练 4 层模型：

- 如果用 TP，4 层仍然都在两张卡上同时存在，但每一层的矩阵都切成两片，两张卡一起算同一层。
- 如果用 PP，第 1-2 层放卡 0，第 3-4 层放卡 1；卡 0 算完前两层，把中间激活传给卡 1，卡 1 再继续。

这两种方式的边界条件也不同。TP 通常要求每层的 hidden size、attention head 数等能整除并行数，否则切分困难。PP 则要求 stage 的计算量尽量均衡，否则慢的 stage 会拖垮全局吞吐。

---

## 核心机制与推导

### 1. 张量并行的基本推导

最常见的是 1D 张量并行，也就是列并行和行并行。

#### 列并行

把权重按列切开：

$$
A = [A_1, A_2, \dots, A_p]
$$

则前向可以写成：

$$
Y = XA = [XA_1, XA_2, \dots, XA_p]
$$

也就是每张卡只持有一个子矩阵 $A_i$，独立计算本地输出 $Y_i = XA_i$。如果下一层也能继续消费分片后的输出，可以先不聚合；如果需要完整输出，就做 all-gather。

白话解释：原来一张卡算一个大矩阵乘法，现在变成多张卡各算一部分列，最后把列方向结果拼起来。

#### 行并行

把权重按行切开：

$$
A = 
\begin{bmatrix}
B_1 \\
B_2 \\
\vdots \\
B_p
\end{bmatrix}
$$

设输入也按列对应分片为 $X=[X_1, X_2, \dots, X_p]$，则：

$$
Y = XA = \sum_{i=1}^{p} X_i B_i
$$

每张卡会得到一个局部部分和，最后通过 all-reduce 或 reduce-scatter 合成完整输出。

白话解释：这次不是拼接，而是“每张卡先算一部分贡献，最后把贡献相加”。

Transformer 中常见的做法是：第一层线性用列并行，第二层线性用行并行。这样中间张量可以自然接起来，通信次数比较可控。

### 2. 2D 张量并行

当 1D 并行继续放大后，单个方向的通信压力也会变大。2D 张量并行会把设备排成 $q \times q$ 网格，同时从输入和权重两个方向切块。它本质上接近 SUMMA 这一类并行矩阵乘法思路：每一步广播部分块、累加局部结果，做 $q$ 轮后得到完整乘积。

直观上看，2D 并行不是简单“横切”或“竖切”，而是把大矩阵切成棋盘格。好处是单卡保存的数据更少，通信也能在两个维度更均匀地摊开；代价是实现更复杂，对通信拓扑要求更高。

### 3. 流水线并行的基本机制

设模型被切成 $K$ 个 stage，每个 stage 放在一张或一组 GPU 上。一个 batch 再被切成 $M$ 个微批次。前向时，stage 0 先处理微批 1，处理完立即发给 stage 1，同时自己开始处理微批 2；这样不同 stage 就像工厂流水线一样并行工作。

如果只看理想状态，吞吐会接近“每个时刻每个 stage 都有活干”。但实际会有两个空转区：

- 开始时，后面的 stage 还没拿到数据。
- 结束时，前面的 stage 已经没新数据可处理。

这两段空转就是 bubble。通常 bubble 比例会和 $\frac{K-1}{M}$ 这一量级相关，含义很直接：stage 越多、微批越少，空转越严重。

### 4. 一个完整的玩具例子

假设有 2 张卡，4 层 Transformer block，输入维度 8，隐藏维度 8。

- TP 方案：每层的 $8 \times 8$ 权重切成两块，每张卡持有 $8 \times 4$ 或 $4 \times 8$ 的一部分。
- PP 方案：卡 0 放第 1-2 层，卡 1 放第 3-4 层。

如果只用 TP，那么每一层都需要两张卡一起参与，优点是每层显存下降；如果只用 PP，那么每层无需层内同步，但卡 0 和卡 1 之间要频繁传中间激活。

因此可以得到一个非常重要的判断准则：

- 模型“宽”到单层放不下，优先考虑 TP。
- 模型“深”到整体堆不下，优先考虑 PP。
- 两者都严重，就做混合并行。

### 5. 真实工程例子

Megatron-LM 训练 GPT 类模型时，常把一个 Transformer block 内的注意力投影和 MLP 做张量并行，同时把若干个 block 组成一个 pipeline stage。这样做的核心原因不是“概念上好看”，而是工程上最稳：

- TP 把单层超大矩阵拆掉，保证层能算。
- PP 把几十层甚至上百层铺到多卡，保证整模能放下。
- 再叠加数据并行，把全局 batch 扩起来。

GShard 更进一步，把自动分片和专家并行结合起来，适合超大规模模型，但复杂度明显更高，不是初学者第一选择。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，用 NumPy 模拟“列并行 + 行并行”的数学正确性。它不是分布式训练代码，但足够解释 TP 的核心计算关系。

```python
import numpy as np

def column_parallel_linear(X, A, parts=2):
    shards = np.split(A, parts, axis=1)  # 按列切
    local_outputs = [X @ shard for shard in shards]
    Y = np.concatenate(local_outputs, axis=1)  # 模拟 all-gather
    return Y, local_outputs

def row_parallel_linear(X, B, parts=2):
    x_shards = np.split(X, parts, axis=1)      # 输入按列切
    b_shards = np.split(B, parts, axis=0)      # 权重按行切
    partials = [x_part @ b_part for x_part, b_part in zip(x_shards, b_shards)]
    Y = sum(partials)                          # 模拟 all-reduce
    return Y, partials

# 玩具输入
X = np.array([[1., 2., 3., 4.],
              [5., 6., 7., 8.]])

A = np.array([[1., 0., 2., 1.],
              [0., 1., 1., 0.],
              [1., 1., 0., 2.],
              [2., 0., 1., 1.]])

# 列并行结果应等于普通矩阵乘法
Y_full = X @ A
Y_tp, local_cols = column_parallel_linear(X, A, parts=2)

assert np.allclose(Y_full, Y_tp)
assert local_cols[0].shape == (2, 2)
assert local_cols[1].shape == (2, 2)

# 再接一个行并行层
B = np.array([[1., 2.],
              [0., 1.],
              [2., 0.],
              [1., 1.]])

Z_full = Y_full @ B
Z_tp, partial_rows = row_parallel_linear(Y_tp, B, parts=2)

assert np.allclose(Z_full, Z_tp)
assert partial_rows[0].shape == (2, 2)
assert partial_rows[1].shape == (2, 2)

print("column parallel ok")
print("row parallel ok")
print("Z =", Z_tp)
```

上面这个例子对应的就是：

$$
Y = X[A_1, A_2] = [XA_1, XA_2]
$$

再经过下一层：

$$
Z = [Y_1, Y_2]
\begin{bmatrix}
B_1 \\
B_2
\end{bmatrix}
= Y_1B_1 + Y_2B_2
$$

如果把它映射到真实工程，分布式版本会把 `concatenate` 换成 all-gather，把 `sum` 换成 all-reduce 或 reduce-scatter。

再给一个接近工程结构的伪代码，说明 TP 和 PP 如何组合：

```python
for micro_batch in micro_batches:
    hidden = micro_batch
    for layer in local_pipeline_stage:
        hidden_local = tensor_parallel_linear(layer, hidden)
        hidden = collective_sync(hidden_local)   # all-gather / all-reduce
    if not is_last_stage:
        send_to_next_stage(hidden)
```

这个流程的关键不是语法，而是顺序：

1. 当前 stage 内部每一层先做 TP 计算。
2. 需要同步时做集体通信。
3. 一个 stage 算完后，把激活传给下一个 stage。
4. 反向时再按相反方向回传梯度。

---

## 工程权衡与常见坑

### 1. TP 不是“显存减半”这么简单

很多初学者会把 2 卡 TP 理解成“每卡工作量和显存都减半”。这不准确。参数通常会明显下降，但激活、优化器状态、临时缓冲区和通信缓存不一定同比例下降。尤其在长序列训练里，真正吃显存的往往不只是一份权重。

### 2. PP 最大的问题是 bubble

如果 4 个 stage 只配 4 个微批次，流水线很难被填满。前几拍只有前面的 stage 在工作，后几拍只有后面的 stage 在收尾。微批次数变多能减少 bubble，但又会带来更小的单批粒度，可能影响 kernel 效率，还会增加调度复杂度。

### 3. stage 切分不能只按层数平均

真实工程里，不同层耗时可能差很多。注意力层、MLP 层、embedding、最后的 lm head，计算和显存占用都可能不同。把 24 层简单切成“每卡 6 层”，并不一定平衡。更合理的做法是按 profile 数据切分，也就是先测每层耗时和显存，再决定 stage 边界。

### 4. 通信拓扑决定上限

TP 的通信一般更频繁，所以更适合同一节点内部、NVLink 或高带宽互联。PP 对点到点传激活更敏感，跨节点也能做，但 stage 边界要尽量减少大张量传输。一个常见工程原则是：把 TP 组尽量放在同机，把 PP 组沿机器扩展。

### 5. 混合并行的维度要能整除

比如 hidden size、attention heads、num layers、global batch size、micro-batch size，都可能受到 TP 或 PP 的整除约束。很多训练配置报错，不是算法错，而是并行度和模型结构根本对不上。

| 常见问题 | 本质原因 | 常见后果 |
| --- | --- | --- |
| TP 开大后反而变慢 | 通信超过计算收益 | GPU 等待同步 |
| PP 利用率低 | 微批太少或 stage 不均衡 | bubble 明显 |
| 显存仍超限 | 只看参数，忽略激活和优化器状态 | 训练启动失败 |
| 混合并行难复现 | 并行组划分复杂 | 调试成本高 |

真实工程例子是 Megatron-LM 训练大 GPT 时，往往不会只调一个参数，而是同时调整 tensor model parallel size、pipeline model parallel size、micro batch size 和 global batch size。任何一个维度变化，都会影响吞吐、显存和通信模式。

---

## 替代方案与适用边界

如果目标是“先把模型跑起来”，不一定一开始就上 TP + PP。

第一种替代方案是纯数据并行或 ZeRO/FSDP 这类参数分片方案。它们更像“每张卡都保留训练逻辑，但把参数、梯度或优化器状态拆开存”。对很多中等规模模型，这比模型并行更容易接入，调试也更简单。

第二种替代方案是专家并行（Expert Parallelism，白话就是只激活部分子网络）。MoE 模型会在一次前向里只让部分专家工作，这样参数总量很大，但每次参与计算的参数没有那么大。GShard 就属于这一路线，适合超大规模场景，但系统复杂度高，路由、负载均衡和通信都更难处理。

第三种替代方案是自动分片。它把“怎么切图、怎么放设备”的决定尽量交给编译器或运行时。优点是减少人工规则，缺点是可解释性和可控性通常更差，排障难度更高。

可以用下面这张表做选择：

| 方案 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 纯 TP | 单层很宽、单机多卡高速互联 | 直接解决层内放不下 | 通信频繁 |
| 纯 PP | 模型很深、可按层切分 | 容易理解 stage 边界 | bubble 明显 |
| TP + PP | 大模型主流训练 | 同时解决宽和深 | 系统复杂度高 |
| ZeRO/FSDP | 中大型模型通用训练 | 接入成本相对低 | 极大单层仍可能受限 |
| GShard / MoE | 超大规模、稀疏激活模型 | 参数规模扩展强 | 路由与系统复杂 |

一个实用判断是：如果你还在 1 台机器内训练，先考虑 TP 或 FSDP；如果已经跨多机且模型层数很多，再认真评估 PP；如果目标是百亿级以上并且训练系统成熟，再考虑 TP + PP + 数据并行的混合设计；如果追求更极端的参数规模，再看 MoE 或自动分片体系。

---

## 参考资料

- Megatron-LM / Hugging Face Accelerate 文档：说明张量并行、流水线并行及混合并行在 GPT/BERT/T5 中的实践  
  https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm
- NVIDIA Megatron Core Parallelism Guide：总结 TP、PP、数据并行、序列并行等组合方式  
  https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html
- Colossal-AI 1D Tensor Parallel 文档：介绍列并行、行并行的线性层拆分  
  https://colossalai.org/docs/features/1D_tensor_parallel
- Colossal-AI 2D Tensor Parallel 文档：介绍 2D 张量并行与 SUMMA 思路  
  https://colossalai.org/docs/features/2D_tensor_parallel/
- PipeFill 论文页面：讨论如何利用 pipeline bubble 提高 GPU 利用率  
  https://www.amazon.science/publications/pipefill-using-gpus-during-bubbles-in-pipeline-parallel-llm-training
- PipePar 论文页面：讨论异构集群中的 pipeline 划分与负载均衡  
  https://www.sciencedirect.com/science/article/pii/S0925231223007841
- GShard 论文页面：介绍自动分片与大规模 MoE 训练  
  https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/

## 核心结论

Megatron-LM 的张量并行，指的是把同一层里的大矩阵拆到多张 GPU 上共同完成。白话说：不是“每张卡都放一整层”，而是“每张卡只放这一层的一块”。

它为什么成立，核心在于线性层和注意力投影本身就是矩阵乘法，矩阵天然可以按列或按行切分。设线性层为

$$
Y = XA,\quad A \in \mathbb{R}^{d_{in}\times d_{out}}
$$

当张量并行度为 $p$ 时，只要被切的维度能被 $p$ 整除，就可以把 $A$ 分成 $p$ 份，每张卡各算一部分，再通过 `all-gather`、`all-reduce` 或 `reduce-scatter` 把结果拼回或汇总。

最直观的玩具例子是：把一个 $4096\times16384$ 的全连接按输出维切成 4 份。每张卡只保留一个 $4096\times4096$ 子矩阵，并计算自己的局部输出 $Y_i\in\mathbb{R}^{batch\times4096}$。从参数占用看，每张卡只保留原层参数的四分之一；从计算看，每张卡只做原来四分之一的 GEMM。

但要注意，Megatron-LM 真正高效的地方，不是“每层算完都立刻拼成完整张量”，而是尽量让切片状态在相邻算子之间继续传递。这样可以少做一次通信。新手可以先把它理解成“算完再拼回去”，工程上再进一步理解成“能不拼就不拼，直到下一个算子真的需要完整布局”。

| 方案 | 每卡参数占用 | 每卡激活占用 | 通信需求 | 适用场景 |
| --- | --- | --- | --- | --- |
| 单卡复制 | 1 倍 | 1 倍 | 无层内通信 | 模型仍放得下单卡 |
| 数据并行 | 1 倍 | 约 1 倍 | 主要在梯度同步 | 模型放得下，想提吞吐 |
| 张量并行 TP=4 | 约 1/4 | 某些层可降到约 1/4 | 几乎每层有通信 | 层内矩阵太大，单卡放不下 |

简化流程可以写成：

`输入分发/复制 -> 局部矩阵乘 -> 跨卡聚合或保持分片 -> 下一并行算子`

---

## 问题定义与边界

张量并行解决的问题很具体：Transformer 里单层过宽，导致 QKV 投影、输出投影、MLP 两层全连接的参数和激活单卡装不下，或者虽然装得下，但 batch 稍大就 OOM。

以一个 decoder block 为例，最占内存的通常不是 LayerNorm 这种小层，而是：

- 注意力里的 `Linear(h, 3h)` 和 `Linear(h, h)`
- MLP 里的 `Linear(h, 4h)` 和 `Linear(4h, h)`

这里的 $h$ 是 hidden size，白话说就是每个 token 的主特征宽度。$h$ 越大，单层矩阵越大，TP 的价值越高。

边界同样很明确：

1. 张量并行是“层内切分”，不是“层间切分”。
2. 它通常要和 Data Parallel、Pipeline Parallel、Activation Checkpoint 一起用。
3. 被切分的维度必须满足可整除条件，否则要改并行度、改模型宽度，或做 padding。
4. TP 的扩展上限受 hidden size 限制，不是卡越多越好。

有效切分条件可以写成：

$$
d_{out}\bmod p = 0 \quad \text{或} \quad d_{in}\bmod p = 0
$$

例如 $d_{model}=12288$，若 `tp_size=8`，每份是 $1536$，可切；若你还想切成 16 份，每份是 $768$，数学上仍可切，但工程上是否值得，要看通信是否已经压过计算。若 hidden size 是 5000，`tp_size=8` 就不能整齐切分，因为 $5000/8=625$ 不是问题，实际上是整数可切；但注意很多真实模块切的是 `num_heads`、`ffn_hidden_size=4h` 或 fused 权重，约束不只一个维度。更典型的失败例子是 `num_heads=40` 配 `tp_size=8`，每卡 5 个头可以；配 `tp_size=6` 就会出问题。

| hidden size | TP 大小 | 每卡宽度 | 是否可整齐切分 |
| --- | --- | --- | --- |
| 4096 | 8 | 512 | 是 |
| 12288 | 8 | 1536 | 是 |
| 12288 | 7 | 1755.43 | 否 |
| 20480 | 8 | 2560 | 是 |

边界条件 checklist：

- `hidden_size % tp_size == 0`
- `num_attention_heads % tp_size == 0`
- 若使用 GQA/MQA，还要检查 `num_query_groups % tp_size == 0`
- 集群内互连足够快，至少节点内有 NVLink，跨节点最好有 RDMA/InfiniBand
- 与 PP、DP 的乘积能刚好覆盖总卡数

---

## 核心机制与推导

Megatron-LM 最经典的是一对线性层交替使用：

- `ColumnParallelLinear`：按输出维切
- `RowParallelLinear`：按输入维切

### 1. ColumnParallelLinear

设

$$
Y = XA,\quad A=[A_1,A_2,\dots,A_p]
$$

其中每个 $A_i\in\mathbb{R}^{d_{in}\times d_{out}/p}$。第 $i$ 张卡计算：

$$
Y_i = XA_i
$$

于是完整输出满足：

$$
Y=[Y_1,Y_2,\dots,Y_p]
$$

这就是“按列切权重，按列得到输出碎片”。

玩具例子：$X\in\mathbb{R}^{2\times4096}$，$A\in\mathbb{R}^{4096\times16384}$，TP=4。  
每张卡各算一个 $2\times4096$ 的局部输出。对新手来说，可以把下一步理解成一次 `all-gather`，把四个局部结果拼回 $2\times16384$。

### 2. RowParallelLinear

紧接着第二层若是

$$
Z = YB,\quad 
B=
\begin{bmatrix}
B_1 \\
B_2 \\
\vdots \\
B_p
\end{bmatrix}
$$

其中每个 $B_i\in\mathbb{R}^{d_{out}/p\times d'_{out}}$，那么第 $i$ 张卡只需要拿自己那块 $Y_i$ 去算：

$$
Z_i = Y_iB_i
$$

最终

$$
Z = \sum_{i=1}^{p} Z_i
$$

这个求和在工程上通常通过 `all-reduce` 完成；若后续还想继续保持切片，也可以用 `reduce-scatter`，即“边求和边分发”。

这里要特别澄清一个常见误解：RowParallelLinear 并不是“总要先 all-gather 输入”。如果它前面接的正好是 ColumnParallelLinear，输入本来就已经按最后一维分片了，此时每张卡直接消费自己的 $Y_i$ 即可，反而不应该先 gather 成完整张量，否则白白多一次通信。

矩阵分块关系可以记成：

`X(复制或一致可见) -> [A1 A2 A3 A4] -> [Y1 Y2 Y3 Y4] -> [[B1],[B2],[B3],[B4]] -> sum(Zi)`

这也是 Megatron 在 MLP 中常见的结构：

1. 第一层 `h -> 4h` 用列并行
2. 激活函数逐卡本地执行
3. 第二层 `4h -> h` 用行并行
4. 输出处做归约

注意力里的 QKV 也类似。把 `Q,K,V` 投影按头维切开，本质上也是“让每张卡负责一部分注意力头”。头，白话说，就是注意力里并行存在的多个子空间。每张卡算自己的头，最后再在输出投影处汇合。

反向传播时，通信会沿相反方向出现。可以粗略理解为：

- `all-gather` 主要用于“把分片输出凑完整”
- `all-reduce` / `reduce-scatter` 主要用于“把各卡局部贡献合并成一致梯度或一致输出”

---

## 代码实现

Megatron-LM 自己有 `ColumnParallelLinear` 和 `RowParallelLinear`。如果用 PyTorch 官方 TP API，概念上对应的是 `ColwiseParallel` 和 `RowwiseParallel`。前者按输出维分片，后者按输入维分片。

下面先给一个可运行的 Python 玩具实现，只验证数学等价性，不依赖分布式环境：

```python
import numpy as np

def column_parallel_linear(x, a, tp):
    # a: [d_in, d_out]
    shards = np.split(a, tp, axis=1)
    local_outputs = [x @ shard for shard in shards]
    y = np.concatenate(local_outputs, axis=1)
    return y, local_outputs

def row_parallel_linear(y_shards, b):
    # b: [d_in, d_out], split on input dim
    b_shards = np.split(b, len(y_shards), axis=0)
    partials = [y_i @ b_i for y_i, b_i in zip(y_shards, b_shards)]
    z = sum(partials)
    return z

batch, d_in, d_mid, d_out = 2, 4, 8, 3
tp = 2

x = np.arange(batch * d_in).reshape(batch, d_in).astype(np.float32)
a = np.arange(d_in * d_mid).reshape(d_in, d_mid).astype(np.float32) / 10
b = np.arange(d_mid * d_out).reshape(d_mid, d_out).astype(np.float32) / 10

# 第一层：列并行
y_full = x @ a
y_tp, y_shards = column_parallel_linear(x, a, tp)
assert np.allclose(y_full, y_tp)

# 第二层：行并行
z_full = y_full @ b
z_tp = row_parallel_linear(y_shards, b)
assert np.allclose(z_full, z_tp)

print("tensor parallel toy example passed")
```

训练侧的伪代码可以写成：

```python
# 伪代码：Megatron 风格 MLP
def mlp_forward(x):
    # ColumnParallelLinear: [h, 4h] 按输出维切
    y_shard = local_matmul(x, w1_shard, b1_shard)

    # GeLU/SiLU 等逐元素算子可本地执行
    y_shard = gelu(y_shard)

    # RowParallelLinear: [4h, h] 按输入维切
    z_partial = local_matmul(y_shard, w2_shard)

    # 汇总各卡局部贡献
    z = all_reduce(z_partial)

    return z
```

若用 PyTorch 官方接口，关注点主要是这几个参数：

| API/参数 | 含义 | 什么时候关心 |
| --- | --- | --- |
| `parallelize_module` | 按计划改写模块并挂上 TP 布局 | 把普通 `nn.Module` 变成 TP 模块时 |
| `ColwiseParallel()` | 按输出维切分线性层 | 对应 Megatron 的列并行 |
| `RowwiseParallel()` | 按输入维切分线性层 | 对应 Megatron 的行并行 |
| `use_local_output` | 返回本地张量还是 DTensor | 调试布局或继续接下游分片算子时 |
| `output_layouts` | 指定输出是分片还是复制 | 避免不必要的 gather |

真实工程例子：训练 530B 级别模型时，单独靠 TP 不够，通常是 `TP + PP + DP` 混合。比如 MT-NLG 530B 使用过 8 路张量并行与 35 路流水并行的组合。原因很简单：TP 解决“层太宽”，PP 解决“层太深”，DP 解决“吞吐不够”。

---

## 工程权衡与常见坑

张量并行不是免费午餐。它在显存上省出来的部分，往往会在通信上付回去。

| 常见坑 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 网络慢 | GPU 利用率低，通信时间长 | 每层都要集体通信 | 节点内优先 NVLink，跨节点优先 RDMA/IB |
| 维度不可整除 | 直接 shape error | hidden/head/group 与 TP 不匹配 | 先算整除关系，再定模型宽度和 TP |
| TP 过大 | 速度反而下降 | 每卡算得太少，带宽成瓶颈 | 先做 profiler，再决定 TP 上限 |
| 错误地强制 gather | 显存升高、通信增多 | 本可保持分片却提前拼回 | 让相邻并行层直接消费 shard |
| 只配 TP 不配 PP/DP | 卡数用不满 | TP 受 hidden size 限制 | 与 PP、DP 组合铺满集群 |

一个新手常见问题是：“如果 TP=8 且 `d_model=4096`，不是每卡只有 512 维，切得越细越省吗？”  
答案是不一定。512 维虽然还能算，但 GEMM 规模太小后，单卡计算效率下降，通信相对占比上升。TP 能省显存，但不能无限线性扩展吞吐。

在 A100 DGX 这类机器上，节点内有 NVLink，TP 通常更友好；一旦 TP 组跨节点，通信代价明显变大，这也是很多大训练会优先把 TP 约束在单节点内，把跨节点的扩展留给 PP 或 DP。

---

## 替代方案与适用边界

如果模型参数还没超过单卡容量，最简单的通常不是 TP，而是数据并行。数据并行，白话说，就是每张卡都放同一个模型，只分不同样本去算，最后同步梯度。

如果模型是“层数很多但单层不太宽”，流水并行往往比 TP 更合适。因为 PP 只在 stage 边界传激活，不需要在几乎每层做集体通信。

| 方案 | 显存收益 | 吞吐特点 | 主要瓶颈 | 适用边界 |
| --- | --- | --- | --- | --- |
| 纯数据并行 | 低 | 简单直接 | 模型复制占显存 | 模型能放进单卡 |
| 纯张量并行 | 中到高 | 层内扩展强 | 高频通信 | 单层超宽 |
| 纯流水并行 | 中 | 深层模型友好 | pipeline bubble | 层很多、可分段 |
| 混合 TP+PP+DP | 最高 | 大模型主流方案 | 配置复杂 | 百亿到千亿级训练 |

适配边界 checklist：

- 小模型、单卡能放下：优先 DP
- 单层矩阵过宽：引入 TP
- 层数很多、单层没那么宽：引入 PP
- 序列很长、激活吃紧：再考虑 SP 或 CP
- 显存还是不够：加 Activation Checkpoint
- 网络一般：谨慎增大 TP，优先单节点内组 TP

一个直观判断法是：

- “模型复制太贵”是 DP 的问题
- “某一层根本塞不进单卡”是 TP 的问题
- “整条网络太长，一个 rank 放不下”是 PP 的问题

因此，像 530B 这种规模，常见做法一定是混合并行；而一个 7B、13B 量级、hidden size 不夸张的模型，在现代高显存卡上很多时候先用 FSDP 或 DP 就够了，不必一开始就上 TP。

---

## 参考资料

| 资料 | 主旨 | 优点 | 适用场景 |
| --- | --- | --- | --- |
| [PyTorch《Large Scale Transformer model training with Tensor Parallel (TP)》, 2024，更新至 2025-07-18](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) | 讲 TP 的基本思想、DeviceMesh 与官方教程实践 | API 视角清晰，适合先建立概念 | 想先学 PyTorch TP 用法 |
| [PyTorch《torch.distributed.tensor.parallel》文档, 2025，更新至 2025-09-09](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel) | 说明 `parallelize_module`、`ColwiseParallel`、`RowwiseParallel` | 适合查参数和布局细节 | 写代码时查 API |
| [RLinf《5D Parallelism Configuration》, 2025/2026 在线文档](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html) | 讲 TP、DP、PP、SP、CP 的组合关系 | 从系统视角看边界和配置 | 做真实集群配置 |
| [Colossal-AI《1D Tensor Parallelism》, 在线文档](https://colossalai.org/docs/features/1D_tensor_parallel/) | 用线性层推导 Column/Row 并行机制 | 数学表达直接，适合入门推导 | 想把公式和通信对应起来 |
| [NVIDIA Technical Blog《Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B》, 2021](https://developer.nvidia.com/blog/?p=38456) | 给出 530B 训练时的并行组合案例 | 有真实超大规模工程背景 | 理解为什么必须混合并行 |
| [Shoeybi 等《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》, 2019](https://arxiv.org/abs/1909.08053) | Megatron-LM 张量并行原始论文 | 是机制来源 | 想追根到最初设计 |

阅读顺序建议：

1. 先看 PyTorch TP 教程，建立“列切分/行切分”的基本图景。  
2. 再看 Colossal-AI 的线性层推导，把公式和通信一一对应。  
3. 然后看 RLinf 的 5D 文档，理解 TP 在完整训练系统里的位置。  
4. 最后看 NVIDIA 530B 案例，理解为什么单一并行策略不够。

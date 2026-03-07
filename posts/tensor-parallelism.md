## 核心结论

张量并行的定义是：把一个很大的权重矩阵按某个维度切成多份，让多张 GPU 分别保存和计算自己的那一份，再通过集体通信把结果汇总。白话讲，就是“一张卡放不下的大矩阵，拆给多张卡一起算”。

Megatron-LM 风格的张量并行有一个很重要的结论：Transformer 一层里，大型线性层如果按“前一层列切分，后一层行切分”的配对方式组织，整层通常只需要两次全局归约 `AllReduce`。一次发生在 Attention 输出回到完整隐藏状态时，一次发生在 FFN 第二层输出回到完整隐藏状态时。

这带来两个直接收益：

1. 显存占用按张量并行度 $TP$ 近似缩小为原来的 $1/TP$。
2. 每层通信量主要是激活大小，近似为 $2\times B\times S\times d$，其中 $B$ 是 batch size，$S$ 是序列长度，$d$ 是隐藏维度。它不随模型总参数线性增长。

可以先看一个对比：

| 方案 | 每张 GPU 保存的权重 | 每层主要通信 | 适合场景 |
| --- | --- | --- | --- |
| 单卡全参数 | 全部参数 | 无跨卡通信 | 模型能完整放进单卡 |
| 张量并行 | 约 $1/TP$ 参数 | 2 次 `AllReduce` | 单卡放不下，但层内矩阵可切分 |

直观图式可以这样理解：把一个大矩阵看成几条“竖条”或“横条”。列切分时，每张卡负责几条竖条；行切分时，每张卡负责几条横条。前面的局部结果先各自算，到了需要恢复完整隐藏表示的位置，再做一次全局归约。

---

## 问题定义与边界

问题很直接：大模型的线性层权重太大，单张 GPU 的显存放不下。所谓显存，就是 GPU 上用于存参数、激活和优化器状态的内存。

以一个 8B 参数量级模型为例，如果某些核心投影层接近 $d\times d$ 或 $d\times 4d$ 规模，那么仅参数加优化器状态就可能超过单卡承载上限。此时即使 batch size 降到很小，也仍然无法训练，因为问题不只是激活，而是“模型本体”放不下。

张量并行解决的是“单层矩阵太大”的问题。它的边界也很明确：

| 问题或场景 | 张量并行是否适合 | 说明 |
| --- | --- | --- |
| 单卡放不下单层大矩阵 | 适合 | 核心目标就是切分权重 |
| 模型能放下，但想提高吞吐 | 视情况而定 | 通信可能抵消收益 |
| 序列极长、激活远大于参数 | 不一定优先 | 可能先考虑 sequence parallel 或重算 |
| 跨很多节点训练 | 谨慎 | 通信拓扑会明显影响效率 |

在 Transformer 里，一个典型例子是 QKV 投影矩阵：

$$
W_{qkv}\in \mathbb{R}^{d\times 3d}
$$

如果做列切分，切成 $TP$ 份，那么第 $i$ 张卡保存：

$$
W_{qkv}^{(i)}\in \mathbb{R}^{d\times \frac{3d}{TP}}
$$

这样每张卡只保存原来约 $1/TP$ 的权重。输入激活 $X\in\mathbb{R}^{(B\cdot S)\times d}$ 会在每张卡上与本地权重相乘，得到局部 QKV 结果。

这里要注意一个边界：张量并行并不是把所有张量都永久切碎。它切的是大型权重以及与之对应的局部中间表示，而层与层之间需要维持逻辑一致的隐藏状态，因此必须在关键位置做通信恢复。

玩具例子可以用一个很小的矩阵看清楚。假设输入是 $X\in\mathbb{R}^{2\times 4}$，权重是 $W\in\mathbb{R}^{4\times 8}$。如果用两张卡列切分，那么每张卡只拿到 $4\times 4$ 的一块。两张卡各自算出 4 维输出，拼起来就是完整 8 维输出。这个过程不改变数学结果，只改变存储位置和计算归属。

---

## 核心机制与推导

先看 Attention。Megatron-LM 常见做法是：

1. Q/K/V 投影做列并行。
2. 多头 Attention 的每个头或一组头在本地完成计算。
3. Attention 输出投影做行并行。
4. 输出处做一次 `AllReduce`，恢复完整隐藏状态。

“列并行”的意思是按输出维度切权重；“行并行”的意思是按输入维度切权重。

设输入为：

$$
X\in\mathbb{R}^{(B\cdot S)\times d}
$$

QKV 投影做列切分后，第 $i$ 张卡计算：

$$
Y_{qkv}^{(i)} = XW_{qkv}^{(i)}
$$

因为切的是输出列，所以每张卡直接得到自己负责的那部分头。注意“头”就是 self-attention 里并行处理不同子空间的计算单元，白话讲就是把一个大表示拆成多块分别算注意力。

接着本地完成注意力：

$$
\text{Attn}^{(i)} = \text{softmax}\left(\frac{Q^{(i)}{K^{(i)}}^T}{\sqrt{d_h}}\right)V^{(i)}
$$

其中 $d_h$ 是单头维度。因为每张卡持有的是完整头的局部集合，所以这一段通常不需要额外通信。

然后进入输出投影。输出矩阵 $W_o$ 做行切分，每张卡拿到：

$$
W_o^{(i)}\in\mathbb{R}^{\frac{d}{TP}\times d}
$$

每张卡把本地注意力输出乘上本地行块，得到对最终隐藏状态的一个“部分和”：

$$
Z^{(i)} = \text{Attn}^{(i)} W_o^{(i)}
$$

由于最终输出是这些部分和的总和，因此要做：

$$
Z = \sum_{i=1}^{TP} Z^{(i)}
$$

这个求和在工程上就是一次 `AllReduce`。

FFN 也是同样思路，只是配对位置不同：

1. 第一层 $W_1:d\rightarrow 4d$ 做列并行。
2. 激活函数如 GELU 在本地独立计算。
3. 第二层 $W_2:4d\rightarrow d$ 做行并行。
4. 输出处再做一次 `AllReduce`。

写成公式是：

$$
H^{(i)} = \text{GELU}(XW_1^{(i)})
$$

$$
O^{(i)} = H^{(i)}W_2^{(i)}
$$

$$
O = \sum_{i=1}^{TP} O^{(i)}
$$

因此，一层 Transformer 的主干里，Attention 一次归约，FFN 一次归约，总共两次。

为什么通信量是 $2\times B\times S\times d$？因为每次需要恢复的都是完整隐藏表示，其形状近似就是 $B\times S\times d$。一层里发生两次，所以总量是：

$$
\text{Comm per layer} \approx 2\times B\times S\times d
$$

这里故意不把参数规模写进去，因为通信发生在激活汇总，不是把全部权重互相发送。

用题目给的数值例子：

- $B=2$
- $S=128$
- $d=2048$
- $TP=2$

则单次归约涉及标量数：

$$
2\times 128\times 2048 = 524{,}288
$$

两次归约总计：

$$
2\times 524{,}288 = 1{,}048{,}576
$$

如果用 `float32`，每个标量 4 字节，总通信量约 4 MB。这个值和“模型是 8B、30B 还是 70B”没有直接线性关系；只要层的隐藏维度和输入激活形状不变，通信规模就是这个量级。

这一点是张量并行能扩展到更宽模型的关键。模型参数增长会增加本地计算和本地存储，但层内主通信不按总参数数目一起爆炸。

真实工程例子是训练 8B 语言模型时，把 8 张 GPU 组成一个 tensor model parallel group。Attention 的 QKV、FFN 的第一层分别按列切，输出投影和 FFN 第二层按行切。这样每张卡只持有八分之一的相关权重，层末用 NCCL 在 NVLink 互联上做两次归约，既能放下模型，也能把通信固定在可控范围。

---

## 代码实现

下面先给一个可运行的玩具实现。它不依赖多 GPU，而是在单机上用 `numpy` 模拟“列切分 + 行切分 + 求和归约”，验证结果与完整矩阵乘法一致。

```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def split_columns(W, tp):
    return np.array_split(W, tp, axis=1)

def split_rows(W, tp):
    return np.array_split(W, tp, axis=0)

def tensor_parallel_ffn(x, W1, W2, tp):
    # W1: column parallel, W2: row parallel
    W1_parts = split_columns(W1, tp)
    hidden_parts = [gelu(x @ part) for part in W1_parts]

    W2_parts = split_rows(W2, tp)
    output_parts = [h @ w for h, w in zip(hidden_parts, W2_parts)]

    # allreduce(sum): gather partial sums into full output
    return sum(output_parts)

def dense_ffn(x, W1, W2):
    return gelu(x @ W1) @ W2

rng = np.random.default_rng(0)
B, S, d, ffn = 2, 3, 4, 8
x = rng.normal(size=(B * S, d))
W1 = rng.normal(size=(d, ffn))
W2 = rng.normal(size=(ffn, d))

y_dense = dense_ffn(x, W1, W2)
y_tp = tensor_parallel_ffn(x, W1, W2, tp=2)

assert np.allclose(y_dense, y_tp, atol=1e-6)
print("tensor parallel toy example passed")
```

这个例子说明一件事：只要切分方式正确，张量并行改变的是执行方式，不改变数学结果。

再看更接近工程实现的伪代码。Attention 部分：

```python
# x: [B, S, d]
# tp_rank: current GPU rank
# tp_world: tensor parallel size

W_qkv_local = split_column(W_qkv, tp_rank, tp_world)   # [d, 3d/TP]
qkv_local = x @ W_qkv_local                            # local heads only

q_local, k_local, v_local = split_qkv(qkv_local)
attn_local = attention(q_local, k_local, v_local)      # local multi-head attention

W_o_local = split_row(W_o, tp_rank, tp_world)          # [d/TP, d]
out_partial = attn_local @ W_o_local                   # partial sum to full hidden
out = allreduce_sum(out_partial)                       # merge hidden states
```

FFN 部分：

```python
W1_local = split_column(W1, tp_rank, tp_world)         # [d, 4d/TP]
hidden_local = gelu(x @ W1_local)                      # local expansion

W2_local = split_row(W2, tp_rank, tp_world)           # [4d/TP, d]
ffn_partial = hidden_local @ W2_local                  # partial sum
ffn_out = allreduce_sum(ffn_partial)                   # merge hidden states
```

可以把触发通信的位置单独列出来：

| 步骤 | 计算方式 | 是否通信 |
| --- | --- | --- |
| QKV 列切分投影 | 每卡独立 | 否 |
| 本地 Attention | 每卡独立 | 否 |
| 输出投影行切分 | 每卡独立先算部分和 | 是，`AllReduce` |
| FFN 第一层列切分 | 每卡独立 | 否 |
| GELU | 每卡独立 | 否 |
| FFN 第二层行切分 | 每卡独立先算部分和 | 是，`AllReduce` |

真实工程里，`allreduce_sum` 一般由 NCCL 实现。NCCL 是 NVIDIA 的集体通信库，白话讲就是“专门帮多张 GPU 高效交换和归约数据的底层组件”。在 PyTorch 中通常表现为：

```python
import torch.distributed as dist

dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_group)
```

如果 GPU 在同一节点、并且通过 NVLink 互联，`AllReduce` 往往更快。NVLink 可以理解为 GPU 之间的高速直连通道；如果退化到 PCIe 或跨节点网络，张量并行的收益会更依赖拓扑设计。

---

## 工程权衡与常见坑

张量并行不是“切开就结束”，真正难的是保证切分规则、通信位置和硬件拓扑一致。最常见的坑如下：

| 坑 | 后果 | 规避措施 |
| --- | --- | --- |
| 列切分和行切分顺序不一致 | 结果错位或多一次归约 | 固定 `column -> row` 配对 |
| 把本地局部结果误当成完整隐藏状态 | 下游维度对不上，数值错误 | 明确哪些张量是 partial sum |
| 激活检查点重算过于粗暴 | 重算时重复触发通信 | 使用 selective recompute |
| TP 组跨节点过大 | 通信延迟上升 | 尽量把 TP 组放在单节点 NVLink 域内 |
| TP 度数过大 | 通信占比超过计算收益 | 结合 profile 选择 2/4/8 等合理值 |

第一个坑最关键。假设某一层的 QKV 是列切分，但下一层错误地按不兼容的方式继续消费局部结果，没有在正确位置做行并行归约，那么要么会出现维度错配，要么只能额外补一次 `AllGather` 或 `AllReduce`。这不仅让通信翻倍，还可能把“本地头”和“全局隐藏状态”的语义搞混。

第二个坑出现在激活检查点。激活检查点的意思是前向时不保存全部中间结果，反向时再重算，以节省显存。问题在于，如果你对张量并行层做了朴素重算，那么每次重算都可能再次触发那两次 `AllReduce`。这样显存是省了，但通信和延迟会明显上升。实践里通常会配合 sequence parallel 的 selective recompute，只重算必要部分，减少重复通信。

第三个坑是拓扑。拓扑就是硬件连接关系。张量并行强依赖低延迟互联，因此通常优先在同一台机器内部开 TP 组，让归约尽量走 NVLink；把跨节点的扩展更多交给数据并行或流水线并行。如果把一个 8 卡 TP 组硬拆到两个节点，中间只靠较慢网络连接，那么层内两次归约就会直接拖慢每一步训练。

---

## 替代方案与适用边界

张量并行不是唯一方案，它和数据并行、流水线并行解决的是不同瓶颈。

数据并行的定义是：每张 GPU 都保存一整份模型，只切 batch。白话讲，就是“模型人人都有，样本分开算”。它适合模型本身能放进单卡，但想扩大吞吐的场景。

流水线并行的定义是：把不同层放到不同 GPU 上，让样本像流水线一样流过多个阶段。白话讲，就是“不同卡负责不同楼层”。

三者对比如下：

| 并行方式 | 切分对象 | 主要通信 | 显存收益 | 适用边界 |
| --- | --- | --- | --- | --- |
| 数据并行 | batch | 梯度同步 | 对参数几乎无帮助 | 模型能完整复制 |
| 张量并行 | 层内矩阵 | 层内激活归约 | 直接降低单卡参数占用 | 单层很宽、互联较快 |
| 流水线并行 | 网络层 | stage 边界传激活 | 每卡只存部分层 | 层数深、可容忍流水气泡 |

张量并行最适合的情况是：模型宽度很大，单层矩阵成为显存瓶颈，而且 GPU 之间通信比较快。它不太适合孤立使用在“超长序列、跨节点很多、通信很慢”的环境里。

实际大模型训练常常是混合并行。比如 Megatron Bridge 训练 8B 模型时，可能同时使用：

- tensor parallel：切单层大矩阵
- sequence parallel：切序列维度，降低激活压力
- pipeline parallel：切层
- data parallel：扩展总吞吐

这里的重点是，张量并行仍然保持自己的内部规律：Attention 和 FFN 各做一次归约，总体还是每层两次 `AllReduce`。其他并行方式是在更高层次上继续分担显存或吞吐压力，而不是推翻这套机制。

所以可以把适用边界总结成一句话：如果问题是“单层太大”，先看张量并行；如果问题是“整网太深”，看流水线并行；如果问题是“模型放得下但训练太慢”，看数据并行；如果问题三者同时存在，就做混合并行。

---

## 参考资料

| 来源 | 内容 | 用途 |
| --- | --- | --- |
| GPU Learning 第 11 章 | 多 GPU 与 Transformer 张量并行概览 | 用于解释列并行、行并行和两次归约的整体图景 |
| 《A Comprehensive Survey on Distributed Deep Learning Training》 | Tensor Parallelism 综述 | 用于补充张量并行的定义与边界 |
| NuMPItron / Lweitkamp 博客 | 张量并行通信分析与数值例子 | 用于说明通信量只与 $B,S,d$ 相关 |
| Megatron Bridge Performance Guide | tensor/sequence/pipeline 混合并行实践 | 用于工程场景与拓扑权衡 |

- GPU Learning 第 11 章：适合先建立“为什么列切分后能本地算、为什么行切分后要归约”的基本图像。
- 分布式训练综述论文：适合补充张量并行在并行策略谱系中的位置，避免把它和数据并行混为一谈。
- NuMPItron 张量并行分析：适合看通信量推导，尤其是“每层两次归约、通信不随总参数线性增长”这一点。
- Megatron Bridge 性能指南：适合看真实工程配置，特别是张量并行如何与 sequence parallel、pipeline parallel 组合使用。

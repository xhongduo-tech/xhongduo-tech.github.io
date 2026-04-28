## 核心结论

vLLM 张量并行，简称 TP（Tensor Parallelism，白话讲就是“把同一层拆给多张卡一起算”），解决的第一问题不是“绝对更快”，而是“单卡放不下模型，或者单卡显存压力过高”。它的核心动作是把同一层的权重切到 $p$ 张 GPU 上，每张卡只保留约 $1/p$ 的那一片参数，然后在前向阶段通过 `all-gather` 或 `all-reduce` 把局部结果汇总成完整结果。

4 张卡做 TP，不是 4 个人各自完整做一道题，而是每个人只做自己负责的那一部分，最后把结果拼起来。这样做的直接收益是单卡显存下降，近似可以看成：

$$
\text{单卡权重显存} \approx \frac{\text{原始权重显存}}{p}
$$

但代价也很直接：每一层都会引入至少一次 collective communication（集合通信，白话讲就是“多张卡一起同步数据”）。当序列短、batch 小时，计算量下降得很快，而通信固定开销下降没那么明显，所以跨卡延迟会更早变成瓶颈。

下面这张表先给出总览：

| 项目 | TP 的表现 | 直接收益 | 直接代价 |
|---|---|---|---|
| 权重存放 | 同一层切到多卡 | 单卡参数约降到 $1/p$ | 需要管理分片 |
| 前向计算 | 每卡只算局部 GEMM | 能支撑更大模型 | 每层都要通信 |
| 通信方式 | `all-gather` / `all-reduce` | 保证结果正确 | 短序列下通信更显眼 |
| 性能趋势 | 长序列、大 batch 更友好 | 计算更容易摊薄通信 | 在线 chat 常受限于延迟 |

结论先落地成一句工程判断：TP 先回答“模型能不能放下”，再回答“吞吐是不是还可接受”。这两个问题不能混在一起看。

---

## 问题定义与边界

TP 解决的问题可以严格表述成一句话：单卡无法完整承载模型参数，或者单卡推理时参数、KV cache、临时激活共同占用显存，导致部署失败或余量太小。这里的“承载”不是只看模型文件大小，而是看推理运行时总显存。

对初学者最容易混淆的一点是：TP 不是“多卡版数据并行”。数据并行（Data Parallel，白话讲就是“每张卡都放一份完整模型，各自处理不同请求”）适合副本扩展；TP 是把同一份模型本体拆开。前者解决吞吐扩展，后者优先解决装载问题。

一个常见真实工程例子是部署 13B 或 14B 级模型。假设你在单机 4 卡环境里运行推理，单卡可能因为参数加缓存而放不下，这时把 `tensor_parallel_size=4` 往往是最低可用方案。注意“最低可用”不等于“最快方案”。它只是说明：不拆不行，拆了才能跑。

适用边界可以先用表格看清：

| 问题类型 | 是否适合 TP | 说明 |
|---|---|---|
| 单卡显存不足 | 适合 | TP 的首要目标就是把模型切开 |
| 单机多卡且有 NVLink | 适合 | 卡间互联强，通信损失相对可控 |
| 在线 chat，`batch=1`，短输出 | 谨慎 | 每层通信频繁，延迟容易先暴露 |
| 单机多卡但无 NVLink | 谨慎甚至不优先 | 官方文档明确建议优先考虑 PP |
| 多机部署且网络强 | 可用 | 需要 InfiniBand、GPUDirect RDMA 等支撑 |
| 多机部署但网络弱 | 不优先 | 网络可能比算力更早成为瓶颈 |

这里还要明确一个边界：TP 并不减少整机总参数量，它只是把参数分摊到不同卡上。整机总显存消耗不会神奇消失，只是从“一张卡爆掉”变成“多张卡共同承担”。

如果把问题说得更工程化一些，TP 最适合这样的场景：模型比单卡大，但还没大到必须跨很多机器；机器内部互联较强；业务允许一定通信开销；你愿意用更多 GPU 换取“模型能放下”。

---

## 核心机制与推导

TP 的基础来自两类线性层切分方式：`ColumnParallelLinear` 和 `RowParallelLinear`。线性层可以理解成“矩阵乘法层”，也就是把输入向量乘上一个大权重矩阵得到输出。TP 本质上就是切这个大矩阵。

### 1. 按列切：`ColumnParallelLinear`

设输入为：

$$
X \in \mathbb{R}^{BL \times H}
$$

这里 $B$ 是 batch size，白话讲就是“一次并行处理多少条请求”；$L$ 是序列长度；$H$ 是隐藏维度，白话讲就是“每个 token 的特征宽度”。

如果权重矩阵：

$$
W \in \mathbb{R}^{H \times H}
$$

按列切到 $p$ 张卡上，可以写成：

$$
W = [W_1, W_2, ..., W_p], \quad W_i \in \mathbb{R}^{H \times H/p}
$$

每张卡只算自己的局部输出：

$$
Y_i = XW_i
$$

如果后续需要完整输出向量，就把各卡的 $Y_i$ 拼起来：

$$
Y = \text{all-gather}(Y_1, Y_2, ..., Y_p)
$$

白话解释：每张卡只负责输出向量的一段，最后把这些段拼回完整向量。

### 2. 按行切：`RowParallelLinear`

另一种方式是把权重按行切：

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix},
\quad
W_i \in \mathbb{R}^{H/p \times H}
$$

这时每张卡会拿到输入的一部分，分别算出局部贡献，最后再做求和：

$$
Y = \sum_{i=1}^{p} X_i W_i
$$

工程里通常对应一次 `all-reduce`。`all-reduce` 可以理解成“每张卡都拿出自己的部分和，再把总和同步回每张卡”。

### 3. 为什么通信不可避免

很多新手第一次看 TP 会问：既然每卡都只算一部分，为什么还要通信？原因是神经网络层与层之间要求张量形状和数值都正确对接。局部结果通常不够下一层直接使用，必须恢复成正确的逻辑含义。

因此，TP 的一个近似公式可以写成：

$$
\text{每层通信量} \approx O(BLH)
$$

它不是说通信一定和计算一样重，而是说通信规模跟激活张量大小强相关。于是就得到一个很关键的推论：

- 当 $B$ 和 $L$ 大时，计算量也大，通信更容易被摊薄。
- 当 $B=1, L=1$ 时，单步 decode 的计算很小，通信固定开销就变得很刺眼。

### 4. 玩具例子

取一个最小但有代表性的例子：

- 隐藏维度 $H=4096$
- TP 大小 $p=4$
- 精度 `bf16`，每个元素 2 字节

一个 $4096 \times 4096$ 的线性层权重大小约为：

$$
4096 \times 4096 \times 2 \text{ B} \approx 32 \text{ MiB}
$$

如果做 4 路 TP，每卡只存约：

$$
32 / 4 = 8 \text{ MiB}
$$

这就是 TP 最直接的显存收益。

再看 decode 阶段。假设 `B=1, L=1`，也就是只为一个请求生成一个新 token。这时一层需要处理的激活大约是一个长度为 4096 的向量，规模约：

$$
4096 \times 2 \text{ B} = 8192 \text{ B} = 8 \text{ KiB}
$$

表面看 8 KiB 很小，很多人会误以为“小数据通信肯定便宜”。问题在于：GPU collective 的成本不只由字节数决定，还包括发起、同步、等待、拓扑路径这些固定部分。于是就会出现一个反直觉现象：算得少，传得也少，但通信未必便宜。短序列推理慢，常常不是因为传了很多，而是因为“每层都得等一次”。

### 5. 真实工程例子

假设单机 4 张 A100 40GB 部署一个 13B/14B 级模型。

- 如果模型单卡放不下，`tensor_parallel_size=4` 是常见起点。
- 在 prefill 阶段，也就是“把整段输入提示词一次性编码”的阶段，$L$ 很长，矩阵乘法工作量足够大，TP 能得到一定并行收益。
- 在在线 chat 的 decode 阶段，经常是 `batch=1`、每次只生成一个 token，这时层层 collective 的延迟容易盖过 GEMM，吞吐不一定理想。

这就是为什么同一个部署配置，在长提示词压测里看起来还行，到了真实在线对话却未必好看。不是模型变了，而是计算与通信的占比变了。

---

## 代码实现

vLLM 的 TP 不是停留在概念图上的“理论并行”，而是落实到并行配置、权重分片和通信封装上的工程实现。官方文档明确说明，其实现包含 Megatron-LM 风格的张量并行算法；也就是说，理解 Megatron 的 `ColumnParallelLinear` / `RowParallelLinear`，就能抓住 vLLM TP 的骨架。

先看一个简化版 `ColumnParallelLinear` 伪代码：

```python
def column_parallel_linear(x, w_shard, gather_output, tp_group):
    # x: [B*L, H]
    # w_shard: [H, H/p]
    local_y = x @ w_shard
    if gather_output:
        full_y = all_gather_concat(local_y, group=tp_group, dim=-1)
        return full_y
    return local_y
```

再看 `RowParallelLinear`：

```python
def row_parallel_linear(x_shard, w_shard, tp_group):
    # x_shard: [B*L, H/p]
    # w_shard: [H/p, H]
    partial_y = x_shard @ w_shard
    full_y = all_reduce_sum(partial_y, group=tp_group)
    return full_y
```

上面这两段代码故意写得很短，因为要传达的是“局部 GEMM + 必要通信”这个结构。实际工程代码还会处理 bias、layout、dtype、kernel 选择、分布式组初始化等问题。

下面给一个可运行的 Python 玩具实现。它不依赖 GPU，也不依赖 NCCL，只是用 NumPy 模拟“切分后再拼回去”的数学正确性。

```python
import numpy as np

def column_parallel_linear(x, w, p):
    # 按列切 W: [H, O] -> p 个 [H, O/p]
    shards = np.split(w, p, axis=1)
    partials = [x @ shard for shard in shards]
    y = np.concatenate(partials, axis=1)
    return y

def row_parallel_linear(x, w, p):
    # 按行切 W: [H, O] -> p 个 [H/p, O]
    x_shards = np.split(x, p, axis=1)
    w_shards = np.split(w, p, axis=0)
    partials = [xs @ ws for xs, ws in zip(x_shards, w_shards)]
    y = sum(partials)
    return y

# 玩具数据
rng = np.random.default_rng(0)
B, L, H, O, p = 2, 3, 8, 8, 4
x = rng.normal(size=(B * L, H)).astype(np.float32)
w = rng.normal(size=(H, O)).astype(np.float32)

# 基准结果
y_ref = x @ w

# TP 模拟结果
y_col = column_parallel_linear(x, w, p)
y_row = row_parallel_linear(x, w, p)

assert np.allclose(y_ref, y_col, atol=1e-5)
assert np.allclose(y_ref, y_row, atol=1e-5)

# 显存量级估算：bf16 每元素 2 字节
bytes_total = H * O * 2
bytes_per_rank = bytes_total // p
assert bytes_per_rank * p == bytes_total

print("column parallel ok")
print("row parallel ok")
print("total bytes:", bytes_total)
print("bytes per rank:", bytes_per_rank)
```

这段代码证明两件事：

1. 线性层切分后，数学上仍然可以恢复与完整矩阵乘法一致的结果。
2. 权重存储确实按分片缩小到原来的约 $1/p$。

再看工程入口的对应关系：

| 组件 | 作用 | 读源码时要看什么 |
|---|---|---|
| `vllm/config/parallel.py` | 并行配置入口 | TP、PP 等尺寸参数如何定义 |
| `vllm/distributed/device_communicators/pynccl_wrapper.py` | 通信封装 | 为什么要自定义 NCCL wrapper |
| Megatron-LM `layers.py` | 参考实现 | `ColumnParallelLinear` 的切分与 gather 逻辑 |

`pynccl_wrapper.py` 还解释了一个很容易误判的问题：vLLM 做了自定义 NCCL wrapper，主要原因是要让 NCCL 与 CUDA graph 配合，而直接用 `torch.distributed.all_reduce` 可能触发一些不适合 graph capture 的 CUDA API。于是，很多“看起来像模型算错了”的报错，根因可能是通信、驱动、NCCL 版本或拓扑，不是 GEMM 本身。

---

## 工程权衡与常见坑

TP 的工程现实很简单：不是越大越好。`tensor_parallel_size` 增大时，单卡显存确实更轻松，但通信频率和同步成本也会更难隐藏。尤其在在线推理里，模型一层一层地生成 token，collective 是持续触发的，不是一次性成本。

最常见的误区是把“能跑起来”误认为“已经合理”。比如一个在线 chat 服务，典型特征是：

- `batch=1`
- 输出很短
- 每次只 decode 一个 token
- 服务更关心首 token 延迟和单请求尾延迟

在这种负载下，TP 可能只是“刚好能跑”，但吞吐和延迟都未必理想。如果把 TP 从 2 扩到 4，显存更稳了，但每层通信也更多，最终用户感受到的可能不是变快，而是变慢。

机器拓扑也决定上限。NVLink 可以理解成“GPU 之间更快的内部高速路”；InfiniBand 则是多机间更强的网络链路。如果这些条件不足，先怀疑通信链路，通常比先怀疑模型实现更靠谱。

常见坑可以直接列出来：

| 常见坑 | 现象 | 原因 | 规避手段 |
|---|---|---|---|
| TP 设得过大 | 吞吐下降、延迟上升 | 通信盖过计算 | 先选刚好能放下模型的最小 TP |
| 单机无 NVLink 还硬上大 TP | 多卡比单卡还难看 | 卡间互联弱 | 优先考虑 PP 或量化 |
| 多机网络弱 | 跨机 TP 抖动大 | TCP/低速网络不适合频繁 collective | 使用 InfiniBand、GPUDirect RDMA |
| 通信报错就怀疑模型层 | 排错方向错误 | NCCL/驱动/拓扑更常见 | 先查 NCCL 日志与拓扑 |
| CUDA graph 场景异常 | 复现不稳定 | 通信调用与 capture 兼容性问题 | 注意 vLLM 的自定义 NCCL wrapper 设计 |

还有一个容易被忽略的点：TP 降的是参数分摊压力，但在线推理的显存大头不只有参数，还有 KV cache。也就是说，TP 能帮你把模型“放下”，却不一定解决长上下文或高并发下的全部显存问题。把 TP 当成万能钥匙，是另一个常见误判。

---

## 替代方案与适用边界

如果把“如何让模型在多卡上跑起来”看成一个方案选择题，TP 只是其中一个选项。最常拿来比较的是 PP，也就是 pipeline parallel（流水线并行，白话讲就是“按层切模型，而不是按层内矩阵切模型”）。

两者差异可以先看表：

| 方案 | 显存占用 | 通信成本 | 适用场景 | 部署复杂度 |
|---|---|---|---|---|
| TP | 单卡参数约降到 $1/p$ | 每层都可能有 collective | 模型单卡放不下，且互联较强 | 中等 |
| PP | 每卡只放一部分层 | 主要在层边界传激活 | 无 NVLink 或需要按层切分时更稳 | 中等到较高 |
| 单卡 + 量化 | 从根上减小参数 | 几乎无跨卡通信 | 模型能靠量化放进单卡 | 低到中等 |

TP 和 PP 的根本区别是通信频率与位置不同。

- TP：同一层内部拆开，通信更细粒度，几乎层层都要参与。
- PP：模型按层段切开，通信更粗粒度，主要发生在阶段边界。

这就解释了为什么在单机 4 卡但没有 NVLink 的情况下，官方会建议优先考虑 PP，而不是硬上大 TP。因为弱互联环境下，频繁的小步同步通常比少量的大步传递更吃亏。

再看一个判断框架：

1. 如果模型通过量化后能稳定放进单卡，优先试单卡 + 量化。
2. 如果量化后仍放不下，但单机互联强，优先试最小可用 TP。
3. 如果单机互联弱，或者 GPU 数量与模型切分不整齐，优先看 PP。
4. 如果必须跨机，TP 是否可行首先取决于网络，而不是取决于 GPU 总数。

一句话收束这一节：TP 更像“先把同一层拆开”，PP 更像“把整条网络分段”，量化则是“直接把模型压小”。它们解决的是同一大问题下的不同子问题。

---

## 参考资料

1. [vLLM 官方文档：Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
2. [vLLM 源码：pynccl_wrapper.py](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py)
3. [vLLM 源码：config/parallel.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py)
4. [Megatron-LM 源码：tensor_parallel/layers.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py)

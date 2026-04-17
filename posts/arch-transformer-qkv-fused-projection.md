## 核心结论

QKV 投影合并优化的本质，是把原本三次线性变换

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

改写成一次更大的线性变换

$$
[Q,K,V] = XW_{QKV},\quad W_{QKV}\in \mathbb{R}^{D\times 3D}
$$

其中“线性变换”就是把输入向量乘上一个权重矩阵，得到新的特征表示。数学结果没有变化，变化的是执行方式。

它快的原因不是总 FLOPs 突然减少了，而是 GPU 的“内存带宽”压力下降了。内存带宽可以理解为显存每秒能搬多少数据。原始实现要把同一个输入 `X` 读三次、发三个 kernel、写三次中间结果；合并后通常只需读一次输入、发一次大 kernel，再在寄存器或共享内存里完成后续拆分。对注意力块这类常见的内存敏感路径，工程实践里常能看到约 30% 到 50% 的延迟下降；更一般的融合研究中，面向线性变换类模式的 speedup 常落在约 1.5x 到 3.13x 区间。

给零基础读者一个玩具理解：原来像“拨三次电话”，分别联系 Q、K、V 三个投影；合并后是“一次拨号接三条线”，输入 `X` 只搬运一次，后续结果直接在片上缓存里拆开。

---

## 问题定义与边界

先定义问题。设输入张量

$$
X \in \mathbb{R}^{B\times L\times D}
$$

其中 `B` 是 batch size，表示一次并行处理多少条样本；`L` 是序列长度，表示每条样本里有多少个 token；`D` 是隐藏维度，表示每个 token 的特征宽度。

标准自注意力前半段通常要做三次投影：

- `Q = XW_Q`
- `K = XW_K`
- `V = XW_V`

如果每个权重都是 `D x D`，那么这三步的总计算量和合并后基本等价，但执行路径不同。问题不在“算不算得完”，而在“数据搬得值不值”。GPU 上很多 Transformer 层并不是纯算力受限，而是内存访问、kernel 启动、张量重排这些开销先成为瓶颈。

看一个具体尺寸。假设 `B=1, L=4, D=512`：

- 未合并：3 次 `4 x 512` 乘 `512 x 512`
- 已合并：1 次 `4 x 512` 乘 `512 x 1536`

从代数上看，右边只是把三个权重横向拼起来了，FLOPs 同阶；但输入 `X` 在未合并时被重复读三遍，在合并时通常只读一遍。对长序列和大 batch，这个差异会直接放大。

这里要明确边界。QKV 合并优化主要解决的是“投影阶段”的访存效率问题，不等于整个 attention 都自动最优。后面的 `QK^T`、`softmax`、`AV` 仍然可能是瓶颈，所以工程里常见做法是继续把 `bias`、`reshape`、`transpose`、甚至整个 attention 都做更深层融合。

---

## 核心机制与推导

把三个矩阵拼接起来：

$$
W_{QKV} = [W_Q \; W_K \; W_V] \in \mathbb{R}^{D\times 3D}
$$

则有

$$
XW_{QKV} = [XW_Q,\; XW_K,\; XW_V] = [Q,K,V]
$$

这说明合并不会改变数学定义，只是把三次矩阵乘法变成一次更宽的矩阵乘法。

在多头注意力里，后续还要把最后一维拆成 `3 x num_heads x head_dim`。若

$$
D = h \cdot d_h
$$

其中 `h` 是 head 数，`d_h` 是每个 head 的维度，那么标准流程可写成：

$$
QKV = XW_{QKV} \in \mathbb{R}^{B\times L\times 3D}
$$

$$
QKV \rightarrow \text{reshape}(B,L,3,h,d_h)
$$

$$
QKV \rightarrow \text{transpose}(B,3,h,L,d_h)
$$

然后沿第 2 维切开，得到 `Q, K, V`。

“reshape”可以理解为只改张量的视图解释方式，不改变数值；“transpose”就是调整维度顺序，让后续 kernel 更容易按 head 并行。

玩具例子最直观。令

$$
x=[1,2]
$$

$$
W_{QKV}=
\begin{bmatrix}
1&0&1&0&1&0\\
0&1&0&1&0&1
\end{bmatrix}
$$

则

$$
xW_{QKV}=[1,2,1,2,1,2]
$$

把它按长度 2 一组切开，就是：

- `Q=[1,2]`
- `K=[1,2]`
- `V=[1,2]`

这和分别做三次投影完全一致，只是输入 `x` 只经过了一次矩阵乘法。

为什么常说它把内存访问从“看起来像三份输出”降回到一份输入主导的规模？因为真正昂贵的是反复从 HBM 读取同一个输入块、反复写回中间结果。输出仍然是 `3D`，这无法凭空消失；减少的是中间读写和 kernel 边界。进一步做 fused attention 时，`QK^T`、`softmax`、`PV` 也可以被放进更少的 kernel 中，于是全局内存 round-trip 继续下降，这就是 FlashAttention 一类 IO-aware 实现能继续提速的原因。

下表可以把“减少了什么”说清楚：

| 路径 | 未融合 | 融合后 | 直接收益 |
|---|---|---|---|
| 输入 `X` 读取 | 读 3 次 | 常见实现读 1 次 | 降低 HBM 压力 |
| kernel 启动 | 3 次线性层 + 若干后处理 | 1 次大投影 + 更少后处理 | 降低启动开销 |
| 中间张量写回 | Q/K/V 各自产生独立中间结果 | 可在寄存器/共享内存内拆分 | 减少 global memory 往返 |

再看公开性能结论时要注意口径。系统性融合研究报告过约 `1.5x-3.13x` 的 speedup；注意力专门优化资料常报告注意力块延迟下降约 `30%-50%`；有效带宽提升通常在 `20%-30%` 量级更常见。它们不是同一实验设置，但方向一致：减少中间访存，通常就能换来更高吞吐。

---

## 代码实现

下面先给一个可运行的 Python 版本，只验证“合并投影”和“三次独立投影”数值一致。这里不用 PyTorch，只用 `numpy`，便于理解。

```python
import numpy as np

def separate_qkv(x, w_q, w_k, w_v):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    return q, k, v

def fused_qkv(x, w_qkv):
    d = x.shape[-1]
    qkv = x @ w_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    return q, k, v

# 玩具例子
x = np.array([[1.0, 2.0]])
w_q = np.array([[1.0, 0.0], [0.0, 1.0]])
w_k = np.array([[2.0, 0.0], [0.0, 2.0]])
w_v = np.array([[3.0, 0.0], [0.0, 3.0]])

w_qkv = np.concatenate([w_q, w_k, w_v], axis=1)

q1, k1, v1 = separate_qkv(x, w_q, w_k, w_v)
q2, k2, v2 = fused_qkv(x, w_qkv)

assert np.allclose(q1, q2)
assert np.allclose(k1, k2)
assert np.allclose(v1, v2)

# 真实形状例子：B=2, L=4, D=8, h=2
B, L, D, H = 2, 4, 8, 2
Dh = D // H

x = np.random.randn(B, L, D)
w_qkv = np.random.randn(D, 3 * D)

qkv = x @ w_qkv                      # (B, L, 3D)
qkv = qkv.reshape(B, L, 3, H, Dh)    # (B, L, 3, H, Dh)
qkv = np.transpose(qkv, (0, 2, 3, 1, 4))  # (B, 3, H, L, Dh)

q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

assert q.shape == (B, H, L, Dh)
assert k.shape == (B, H, L, Dh)
assert v.shape == (B, H, L, Dh)
```

如果换成 PyTorch，核心逻辑通常就是：

```python
qkv = linear(x, W_qkv, bias)          # (B, L, 3D)
qkv = qkv.view(B, L, 3, H, Dh)
qkv = qkv.permute(0, 2, 3, 1, 4)
q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
attn = softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
out = attn @ v
```

真实工程例子是在 TensorRT、FasterTransformer、FlashAttention、FlashInfer 这类推理库里，`linear + bias + reshape + transpose` 往往不会按 Python 语义逐步落地为很多独立算子，而是被编译器或手写 CUDA kernel 融合。这样中间张量不必每一步都写回显存，而是尽量留在寄存器或共享内存里。

---

## 工程权衡与常见坑

QKV 合并不是“永远更快”，它有明确的硬件约束。

第一类坑是寄存器和共享内存压力。“寄存器”可以理解为离计算单元最近、最快但很小的存储；“共享内存”是一个线程块内部共享的片上缓存。融合越多，单个 kernel 要持有的中间状态越多，可能导致 occupancy 下降。occupancy 指一个 SM 上同时驻留多少个线程块，太低会让硬件吃不满。

举例说，某个 tile 配置如果想同时缓存更大的 Q/K/V 子块，理论上减少了 HBM 往返；但如果因此超过共享内存上限，或者寄存器溢出到 local memory，性能会明显反噬。很多 attention kernel 的调优，核心不是“融合越多越好”，而是“在片上资源够用的前提下尽量融合”。

第二类坑发生在量化场景。量化就是把高精度数压成低比特表示，例如 W4A16 表示权重 4 bit、激活 16 bit。低比特权重不能直接拿来高精度乘法，通常要配合 `scale`，有时还要处理 `zero-point`。如果 QKV 合并后每个小块都频繁解量化、重复加载缩放参数，省下的带宽很可能又被补回去了。

一个常见规避思路是 block-wise quantization，也就是按小块存 scale；再结合 Split-K 或 partial fusion，让每个 tile 先在局部完成更多累加，再统一写回，减少反复解量化和原子操作。

下表总结常见问题：

| 坑 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| register / shared memory 过载 | 融合后反而更慢 | 片上资源不够，occupancy 下降或 spill | 调小 tile，减少一次保留的中间状态 |
| W4A16 反复 scale | 理论带宽收益不明显 | 解量化开销吞掉优化收益 | 用 per-block scale，局部累加后再写回 |
| 误把 `reshape` 当复制 | 以为拆头一定多一次显存拷贝 | 视图变换和真实 materialize 混淆 | 保证张量连续性和 kernel 支持的布局 |
| 只合并 QKV 不看整体链路 | 优化后收益有限 | 真瓶颈可能在后续 attention 或 KV cache | 联合看投影、attention kernel、KV cache |

---

## 替代方案与适用边界

QKV 合并解决的是“三次投影能否变成一次投影”。但如果你的主要问题已经不是投影，而是解码阶段的 KV cache 读写，那么替代方案可能更有效。

最常见的是 MQA，Multi-Query Attention。它的白话解释是：每个 query head 仍然独立，但多个 head 共享同一组 K/V。这样做的最大收益不在训练前向，而在推理解码阶段，因为 KV cache 会显著变小。

如果模型有 `8` 个 query heads，而 K/V 只保留 `1` 组，那么缓存体积理论上能按 head 数大幅下降。对长上下文、在线解码、批量服务，这常比只做 QKV 合并更影响延迟。但代价是表达自由度下降，质量可能略受影响，所以后来又出现了 GQA，Grouped-Query Attention，作为 MQA 和标准 MHA 之间的折中。

可按下面的表理解：

| 方案 | latency | memory | accuracy 风险 | 适用边界 |
|---|---|---|---|---|
| 融合 QKV | 低，前向投影更快 | 中，主要省中间访存 | 很低 | 训练和推理都常用，几乎是默认优化 |
| MQA / GQA | 解码更低 | KV cache 显著更省 | 中，取决于模型与任务 | 长上下文推理、在线服务 |
| 标准三分离 | 通常最慢 | 最大 | 最稳 | 调试、教学、某些需要高度可解释拆分的实现 |

所以适用边界可以总结成一句话：

- 如果你在做标准 Transformer 实现，QKV 合并几乎总是值得先做。
- 如果你在做大模型推理解码，真正决定成本的往往是 KV cache，此时应同时考虑 GQA/MQA。
- 如果你在做低比特量化部署，必须把“融合收益”和“解量化成本”一起评估，不能只看代数形式。

---

## 参考资料

- Aman Tiwari, “Model Acceleration”, Aman.ai，关于 transformer kernel fusion、QKV 融合与注意力块延迟收益的综述。https://aman.ai/primers/ai/model-acceleration/
- Michael Brenndoerfer, “Multi-Head Attention: Parallel Attention for Richer Representations”，关于 `W_qkv` 合并投影与多头 reshape 的公式和实现说明。https://mbrenndoerfer.com/writing/multi-head-attention-transformers
- MDPI Electronics, “Analyzing the Impact of Kernel Fusion on GPU Tensor Operation Performance: A Systematic Performance Study”，关于 kernel fusion 的 speedup 与有效带宽数据。https://www.mdpi.com/2079-9292/15/5/1034
- Emergent Mind, “Fused Attention Kernel”，关于 fused attention 在内存流量、吞吐和端到端性能上的总结。https://www.emergentmind.com/topics/fused-attention-kernel
- FlashInfer 文档与项目页，关于 attention kernel、tile 选择、共享内存配置与推理实现。https://docs.flashinfer.ai/api/attention.html
- Fireworks AI, “Multi-Query Attention is All You Need”，关于 MQA 对 KV cache、延迟和吞吐的影响。https://fireworks.ai/blog/multi-query-attention-is-all-you-need
- NVIDIA Technical Blog, “Model Quantization: Concepts, Methods, and Why It Matters”，关于量化中的 scale、zero-point、per-block 量化与工程权衡。https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/

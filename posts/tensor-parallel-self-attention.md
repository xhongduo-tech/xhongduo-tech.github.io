## 核心结论

自注意力层的 Tensor Parallel，本质是把“按头独立”的计算拆到多个 GPU 上。Tensor Parallel 指把同一层的参数和计算横向切到多张卡，而不是每张卡都放完整一层。对标准多头注意力，最自然的切法是：

1. 将 $W_Q,W_K,W_V$ 按列切分，每个 GPU 只负责一部分输出维度，也就是一部分 attention head。
2. 每个 GPU 在本地完成自己的 $Q_i,K_i,V_i$ 投影、$Q_iK_i^\top$、softmax 和乘 $V_i$，中间不需要跨卡通信。
3. 将输出投影矩阵 $W_O$ 按行切分，每个 GPU 先算自己的部分，再做一次 AllReduce 求和，得到完整输出。

这件事为什么成立？因为多头注意力里的不同 head 在输出投影之前彼此独立。白话说，96 个 head 可以先看成 96 份互不干扰的小任务，最后再汇总。

一个最常见的工程配置是 $h=96,t=8$，其中 $h$ 是 head 数，$t$ 是 Tensor Parallel 组大小，也就是一起做张量并行的 GPU 数。此时每张卡处理 $96/8=12$ 个 head。以 GPT-3 175B 常见配置 $d=12288$ 为例，每张卡本地完成 12 个 head 的注意力，最终只在输出投影后对形状约为 $(B,S,d)$ 的部分和做一次聚合。通信频率不随 head 数增加而增加，核心通信量近似与隐藏维度 $d$ 线性相关。

| 阶段 | 权重切分方式 | 每个 GPU 做什么 | 是否需要通信 |
|---|---|---|---|
| Q/K/V 投影 | 按列切分 | 生成本地 head 的 $Q_i,K_i,V_i$ | 否 |
| 注意力计算 | 本地 head 独立 | 计算 $Q_iK_i^\top$、softmax、乘 $V_i$ | 否 |
| 输出投影 | $W_O$ 按行切分 | 计算本地部分输出 | 否 |
| 最终汇总 | 对局部结果求和 | AllReduce 得到完整输出 | 是，一次 |

---

## 问题定义与边界

问题可以定义为：给定输入张量 $X\in\mathbb{R}^{B\times S\times d}$，如何把一层自注意力拆到 $t$ 个 GPU 上，使每张卡只保存部分参数、执行部分计算，同时保持输出与单卡实现等价。

这里几个符号先固定：

- $B$：batch size，批大小，表示一次处理多少条样本。
- $S$：sequence length，序列长度，表示每条样本有多少个 token。
- $d$：hidden size，隐藏维度，表示每个 token 的特征长度。
- $h$：attention head 数，表示多头注意力被拆成多少个子头。
- $d_h=d/h$：每个 head 的维度。
- $t$：Tensor Parallel 组大小，也就是参与这一层切分的 GPU 数。
- $h_{\text{local}}=h/t$：每张卡本地负责的 head 数。

标准前提是：

$$
h \bmod t = 0,\qquad d \bmod h = 0
$$

很多实现里还要求输出维度和 FFN 中间维度也能被 $t$ 整除，否则切分会出现不均匀。

玩具例子先看一个最小场景。假设：

- $d=16$
- $h=4$
- $t=2$

那么每个 head 维度 $d_h=4$，每张卡拿到 $h_{\text{local}}=2$ 个 head。GPU0 负责 head0 和 head1，GPU1 负责 head2 和 head3。两张卡都看到同一个输入 $X$，但只持有自己那一部分 QKV 权重。

边界也很明确：

| 条件 | 含义 | 不满足时会怎样 |
|---|---|---|
| $t \le h$ | GPU 数不能超过 head 数 | 没有足够的 head 可分 |
| $h \bmod t=0$ | head 能平均分给 GPU | 负载不均，常需额外 padding 或特殊实现 |
| 高带宽互联 | 层间频繁通信要快 | 跨节点带宽低时 TP 收益会被通信吃掉 |
| 同步执行 | 各卡同时参与这一层 | 调度复杂度上升 |

例如 $h=96,t=8$ 时，每卡 12 个 head，分配很整齐。若 $h=100,t=8$，则每卡平均是 12.5 个 head，这就不是自然切分。理论上可以做不均匀分配，但工程上通常避免，因为会带来 kernel 形状不一致、负载不均衡和更复杂的通信调度。

---

## 核心机制与推导

标准自注意力可以写成：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

多头注意力的完整输出可写成：

$$
O=\text{Concat}(O_1,O_2,\dots,O_h)W_O
$$

其中每个 head 的输出是：

$$
O_j=\text{softmax}\left(\frac{Q_jK_j^\top}{\sqrt{d_h}}\right)V_j
$$

关键观察是：在乘 $W_O$ 之前，各个 head 的注意力完全独立。因此可以先把 head 分片，再把每片分给不同 GPU。

### 1. QKV 为什么按列切

设 $W_Q\in\mathbb{R}^{d\times d}$。把它按输出维度切成 $t$ 块：

$$
W_Q=[W_{Q,1},W_{Q,2},\dots,W_{Q,t}]
$$

其中每块大小约为 $\mathbb{R}^{d\times d/t}$。$W_K,W_V$ 同理。

于是第 $i$ 张卡只保存：

$$
W_{Q,i},W_{K,i},W_{V,i}
$$

并在本地计算：

$$
Q_i=XW_{Q,i},\quad K_i=XW_{K,i},\quad V_i=XW_{V,i}
$$

因为这一步每张卡都用完整输入 $X$ 去乘自己那一块权重，所以不需要从别的卡拿中间结果。

### 2. 注意力为什么能完全本地算

如果第 $i$ 张卡负责的是一组完整的 head，那么它本地的 $Q_i,K_i,V_i$ 已经包含了这些 head 所需的全部特征。于是它可以直接算：

$$
O_i=\text{Attention}(Q_i,K_i,V_i)
$$

这里 $O_i$ 是本地那一组 head 的输出。由于 head 之间在这一阶段没有相互依赖，所以不需要 AllGather，也不需要跨卡做 softmax。

可以把流程画成一个简化的 ASCII 图：

```text
输入 X
  |
  |---- GPU0: X * WQ0, WK0, WV0 -> 本地 12 个 head -> O0
  |---- GPU1: X * WQ1, WK1, WV1 -> 本地 12 个 head -> O1
  |---- GPU2: X * WQ2, WK2, WV2 -> 本地 12 个 head -> O2
  ...
  |---- GPU7: X * WQ7, WK7, WV7 -> 本地 12 个 head -> O7
```

### 3. 输出投影为什么按行切

把所有本地 head 拼起来后的整体输出记为 $\tilde O\in\mathbb{R}^{B\times S\times d}$。输出投影是：

$$
Y=\tilde O W_O
$$

此时将 $W_O$ 按输入维度切分，也就是按行切：

$$
W_O=
\begin{bmatrix}
W_{O,1}\\
W_{O,2}\\
\vdots\\
W_{O,t}
\end{bmatrix}
$$

如果第 $i$ 张卡持有 $\tilde O_i$ 和 $W_{O,i}$，就能先算局部部分和：

$$
Y_i=\tilde O_i W_{O,i}
$$

最终完整输出是：

$$
Y=\sum_{i=1}^{t} Y_i
$$

这就是为什么最后只需要一次 AllReduce。AllReduce 是“每张卡把自己的部分和发给所有卡，并做求和”的集体通信操作。白话说，每个人先把自己的计算结果写出来，最后统一求总和。

### 4. 玩具例子

设 $h=4,t=2$，每张卡 2 个 head。

- GPU0：负责 head0, head1，得到 $O^{(0)}=\text{Concat}(O_0,O_1)$
- GPU1：负责 head2, head3，得到 $O^{(1)}=\text{Concat}(O_2,O_3)$

再将 $W_O$ 切成两块：

$$
W_O=
\begin{bmatrix}
W_{O,0}\\
W_{O,1}
\end{bmatrix}
$$

则：

$$
Y=O^{(0)}W_{O,0}+O^{(1)}W_{O,1}
$$

GPU0 和 GPU1 各算一项，最后求和即可。这和单卡先拼 4 个 head 再乘完整 $W_O$ 完全等价。

### 5. 真实工程例子

GPT-3 175B 的典型配置里，隐藏维度 $d=12288$、head 数 $h=96$。若 Tensor Parallel 取 $t=8$，则：

$$
h_{\text{local}} = 96/8 = 12,\qquad d_{\text{local}} = 12288/8 = 1536
$$

每张卡负责 12 个 head，也就是 1536 维的 QKV 输出块。Megatron-LM 的经典做法正是把 Tensor Parallel 放在单节点内部，利用 8 张 A100 之间的 NVLink/NVSwitch 高带宽互联，把频繁的层内通信压低；跨节点则更多依赖 Pipeline Parallel。这个组合不是“TP 无通信”，而是“TP 的通信被限制在必要且较规整的阶段”。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现，验证“按 head 分片后本地计算，再在输出投影后求和”和“单机一次性计算”结果一致。它不依赖分布式环境，但数学结构与 Tensor Parallel 相同。

```python
import math
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention(q, k, v):
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    probs = softmax(scores, axis=-1)
    return probs @ v

def mha_full(X, Wq, Wk, Wv, Wo, h):
    B, S, d = X.shape
    dh = d // h

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    Q = Q.reshape(B, S, h, dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, h, dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, h, dh).transpose(0, 2, 1, 3)

    O = attention(Q, K, V)                       # (B, h, S, dh)
    O = O.transpose(0, 2, 1, 3).reshape(B, S, d)
    return O @ Wo

def mha_tensor_parallel_sim(X, Wq, Wk, Wv, Wo, h, t):
    B, S, d = X.shape
    dh = d // h
    heads_per_rank = h // t
    cols_per_rank = d // t

    # QKV 按列切；Wo 按行切
    Wq_shards = np.split(Wq, t, axis=1)
    Wk_shards = np.split(Wk, t, axis=1)
    Wv_shards = np.split(Wv, t, axis=1)
    Wo_shards = np.split(Wo, t, axis=0)

    partial_outputs = []
    for rank in range(t):
        q_local = X @ Wq_shards[rank]            # (B, S, d/t)
        k_local = X @ Wk_shards[rank]
        v_local = X @ Wv_shards[rank]

        q_local = q_local.reshape(B, S, heads_per_rank, dh).transpose(0, 2, 1, 3)
        k_local = k_local.reshape(B, S, heads_per_rank, dh).transpose(0, 2, 1, 3)
        v_local = v_local.reshape(B, S, heads_per_rank, dh).transpose(0, 2, 1, 3)

        o_local = attention(q_local, k_local, v_local)
        o_local = o_local.transpose(0, 2, 1, 3).reshape(B, S, cols_per_rank)

        # 本地输出投影部分和
        y_local = o_local @ Wo_shards[rank]      # (B, S, d)
        partial_outputs.append(y_local)

    # 等价于 all_reduce(sum)
    return sum(partial_outputs)

# 一个可运行的玩具例子
rng = np.random.default_rng(0)
B, S, d, h, t = 2, 3, 8, 4, 2
X = rng.normal(size=(B, S, d))
Wq = rng.normal(size=(d, d))
Wk = rng.normal(size=(d, d))
Wv = rng.normal(size=(d, d))
Wo = rng.normal(size=(d, d))

y_full = mha_full(X, Wq, Wk, Wv, Wo, h)
y_tp = mha_tensor_parallel_sim(X, Wq, Wk, Wv, Wo, h, t)

assert np.allclose(y_full, y_tp, atol=1e-8)
print("tensor parallel simulation matches full attention")
```

如果换成 PyTorch 分布式，核心结构通常是下面这样：

```python
import torch
import torch.distributed as dist

def tp_attention_forward(x, wq_local, wk_local, wv_local, wo_local, num_local_heads):
    # x: [B, S, d]
    q = x @ wq_local
    k = x @ wk_local
    v = x @ wv_local

    B, S, local_dim = q.shape
    head_dim = local_dim // num_local_heads

    q = q.view(B, S, num_local_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_local_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_local_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
    probs = torch.softmax(scores, dim=-1)
    o_local = torch.matmul(probs, v)

    o_local = o_local.transpose(1, 2).contiguous().view(B, S, local_dim)

    # Wo 按行切分，所以每张卡先得到一个 [B, S, d] 的部分和
    y_partial = o_local @ wo_local

    # 所有 rank 求和，得到完整输出
    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
    return y_partial
```

这段代码里最重要的不是 API，而是两个约束：

- `wq_local/wk_local/wv_local` 只覆盖本卡负责的 head。
- `wo_local` 对应输出投影的行分片，因此 `all_reduce` 的对象是局部部分和，而不是 QKV 中间结果。

---

## 工程权衡与常见坑

Tensor Parallel 的优点是明显的：显存下降、单层参数被切开、QKV 的主计算天然能并行。但它不是免费午餐，主要成本来自通信和硬件拓扑。

一个常用近似是：输出投影后的聚合通信量与输出张量大小成正比，因此近似随 $B\times S\times d$ 线性增长。若只看隐藏维度的影响，可以简写为：

$$
\text{Comm} \propto d
$$

这意味着模型越宽，AllReduce 越贵；互联越慢，TP 越不划算。

真实工程例子可以看 Megatron-LM 在 SC21 论文中的结果。论文展示了 GPT-3 175B 级别配置：96 层、96 个 attention head、hidden size 为 12288，并采用 $(t,p)=(8,8)$，也就是 8-way tensor parallel 加 8-way pipeline parallel。在基于 A100、节点内 NVLink/NVSwitch 的环境下，其弱扩展表中 145.6B 规模模型达到 47% 峰值吞吐水平。这个数字不表示“TP 完全没有损耗”，而是说明在高速互联下，TP 的额外通信仍可被工程优化部分掩盖。

| 常见坑 | 触发条件 | 后果 | 缓解方案 |
|---|---|---|---|
| head 数不能整除 TP | $h \bmod t \ne 0$ | 负载不均、kernel 形状复杂 | 优先调整 $h$ 或 $t$ |
| TP 大于 head 数 | $t > h$ | 无法按完整 head 分配 | 降低 TP，改用 PP/DP |
| 跨节点做 TP | 节点间带宽低 | 每层通信拖慢训练 | TP 尽量限制在单节点 |
| 误把输出投影也按列切 | 切分方式错误 | 无法只靠一次求和恢复输出 | 使用行并行输出层 |
| 忽略 reshape/head 排布 | 张量布局不一致 | 数值正确但性能差 | 固定 head-major 或 seq-major 布局 |
| 小 batch 下通信难隐藏 | 计算时间过短 | AllReduce 占比上升 | 增大 microbatch 或结合流水并行 |

新手最容易混淆的一点是：“QKV 阶段不通信”不等于“整层不通信”。正确说法是：QKV 和 head 内 attention 可以本地完成，输出投影后仍然要做聚合。也正因为聚合发生在每一层，TP 更依赖高带宽互联，而不是普通以太网。

---

## 替代方案与适用边界

当 Tensor Parallel 不适合时，通常会退回到 Data Parallel 或 Pipeline Parallel，或者把几种策略组合使用。

- Data Parallel，数据并行，意思是每张卡放完整模型，但处理不同样本，最后同步梯度。
- Pipeline Parallel，流水并行，意思是把不同层放到不同 GPU，上游层算完再交给下游层。
- Tensor Parallel，张量并行，意思是同一层内部再横向切开。

它们的区别可以直接对比：

| 方案 | 切分对象 | 主要通信 | 优点 | 适用边界 |
|---|---|---|---|---|
| TP | 同一层内部张量 | 几乎每层都有 | 单卡放不下单层时有效 | 需要高带宽，且 $t$ 受 head 数限制 |
| DP | 不同样本 | 每步同步梯度 | 最简单，吞吐稳定 | 单卡必须能放下完整模型副本 |
| PP | 不同层 | stage 边界传激活 | 跨节点更友好 | 有流水线气泡，调度更复杂 |

什么时候 TP 不该继续加大？核心边界有两个：

1. head 不够分。若 $h=16$，你却想做 $t=32$，那就已经没有足够的完整 head 分给每张卡。
2. 网络不够快。若 GPU 分布在多节点、节点间带宽明显低于节点内互联，那么层层 AllReduce 会迅速放大延迟。

因此，真实系统通常采用“节点内 TP，节点间 PP，再外层 DP”的组合。白话说，层内细切只在关系最紧密、连线最快的一小组 GPU 内进行；出了这个小组，就改用更粗粒度的切法。

还有一种常见情况是 head 数虽然够，但模型并不是宽而是深。比如某些中等宽度、超深层结构中，单层参数并不大，真正的问题是总层数多、激活长。此时 Pipeline Parallel 往往比一味增加 TP 更合适。

---

## 参考资料

下面这几份资料适合按“概览 → 数学机制 → 工程数据”顺序读：

| 资料 | 核心内容 | 适合阶段 |
|---|---|---|
| [Hugging Face: Tensor Parallelism (TP) in Transformers: 5 Minutes to Understand](https://huggingface.co/blog/qgallouedec/tp) | 用最直接的图和矩阵切分解释 QKV 列并行、输出层行并行 | 入门理解 |
| [Scaling Thoughts: Tensor Parallelism: How Large Models Fit Across GPUs](https://scalingthoughts.com/blog/tensor-parallelism-fundamentals/) | 从矩阵乘法分片、通信代价和设备网格角度解释 TP | 补数学和系统直觉 |
| [Megatron-LM SC21 论文](https://cs.stanford.edu/~deepakn/assets/papers/megatron-sc21.pdf) | 给出大模型训练中的 TP+PP 组合、吞吐和通信优化数据 | 工程落地与性能分析 |

推荐阅读路径：

1. 先读 Hugging Face 文章，建立“QKV 按列切、输出层按行切”的基本图景。
2. 再读 Scaling Thoughts，把这个图景和矩阵乘法分解、通信成本联系起来。
3. 最后看 Megatron-LM 论文，理解为什么工业级训练通常把 TP 限制在单节点内，并与 Pipeline Parallel 组合。

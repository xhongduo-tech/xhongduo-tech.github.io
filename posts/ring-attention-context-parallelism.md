## 核心结论

Ring Attention 是一种把长序列按块切到多张设备上，并通过环形通信轮流传递 $K/V$ 块、在块级在线 softmax 下计算精确 attention 的分布式方法。这里的“精确”意思是：最终结果和一次性对整条序列做全量 softmax attention 一致，不是用近似规则换显存。

它解决的核心问题不是“attention 算得不够快”，而是“长上下文训练时，单卡先被 attention 激活和中间矩阵撑爆显存”。普通全量 attention 在序列长度 $N$ 很大时，需要面对接近 $N \times N$ 的分数矩阵；Ring Attention 的思路是让每张卡只负责自己那一段 query，再让 key/value 块像接力棒一样在设备之间转一圈。每到一轮，本地 query 只和当前收到的 KV 块做一次块级计算，并把统计量累积起来，最后得到与全量计算相同的输出。

下面这张表先把它和常见方案分开：

| 方案 | 是否精确 | 是否全局注意力 | 单卡是否要见到全序列 KV | 是否依赖额外通信 | 主要目标 |
|---|---|---:|---:|---:|---|
| 普通全量 Attention | 是 | 是 | 是 | 否 | 标准实现 |
| 近似/稀疏 Attention | 否或部分 | 通常否 | 否 | 视实现而定 | 降低计算或显存 |
| Ring Attention | 是 | 是 | 否 | 是 | 扩展超长上下文训练 |

一句话概括：Ring Attention = 分块 + 环形通信 + 在线 softmax + 精确 attention。

它的边界也要说清楚：它主要解决序列维度上的 attention 显存问题，不自动解决模型参数、梯度、优化器状态的显存占用。这些问题通常还要靠 FSDP、ZeRO 或张量并行处理。

---

## 问题定义与边界

先定义问题。这里的“超长上下文训练”指模型在训练或微调阶段，要直接处理十万、几十万，甚至更长的 token 序列。对初学者来说，可以把“上下文”理解成模型一次前向计算中能同时看到的输入长度。

标准 self-attention 的核心打分是：

$$
A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)
$$

其中 $Q, K, V$ 分别是 query、key、value；$d$ 是隐藏维度；$M$ 是 mask。问题在于，如果序列长度是 $N$，那么 $QK^\top$ 的形状就是 $N \times N$。哪怕 kernel 再高效，训练时也会被中间激活、归一化状态和反向图拖垮。

一个真实工程例子是：做长代码库继续预训练，或者做百万 token 级视频-语言模型训练时，即使参数已经用 FSDP 或张量并行拆到了多卡，attention 这部分仍可能成为新的瓶颈。原因很简单，参数能分，序列也必须分，否则 attention 激活会先爆。

常见瓶颈可以拆成下面几类：

| 瓶颈来源 | 具体内容 | Ring Attention 是否直接处理 |
|---|---|---|
| attention 激活 | 分数矩阵、softmax 中间量、输出累积 | 是 |
| KV 临时块 | 当前轮次参与计算的 K/V 子块 | 部分处理 |
| 通信带宽 | 设备之间传递 K/V 块的时间 | 不能消除，只能权衡 |
| 参数/梯度/optimizer state | 模型本身与训练状态显存 | 否 |

几个术语先定义清楚：

| 术语 | 白话解释 |
|---|---|
| query chunk | 一小段 query，某张卡当前负责的输入片段 |
| key/value chunk | 一小段 key/value，会在设备间轮流传递 |
| blockwise attention | 按块算 attention，而不是一次算完整矩阵 |
| online softmax | 一边读入分块一边稳定更新 softmax 统计量，不必存完整分数矩阵 |
| ring communication | 环形通信，设备按固定顺序把数据传给下一个设备 |

它能解决什么、不能解决什么，也要明确：

| 项目 | 说明 |
|---|---|
| 能解决 | 单卡装不下超长序列的全局精确 attention |
| 能缓解 | 长序列 attention 的激活显存压力 |
| 不能解决 | 参数太大、优化器状态太大、网络带宽不足 |
| 不等价于 | 稀疏注意力、局部窗口注意力、低秩近似 |

所以，Ring Attention 不是“替代所有并行方式”的总方案，而是训练系统里专门负责“序列维切分”的一层。

---

## 核心机制与推导

设总序列长度为 $N$，设备数为 $D$，每张卡持有长度为 $B=N/D$ 的一个序列块。第 $i$ 张卡本地持有：

$$
Q_i, K_i, V_i \in \mathbb{R}^{B \times d}
$$

第 $t$ 轮时，它接收到第

$$
j=(i-t)\bmod D
$$

张卡的 $K_j, V_j$。也就是说，query 固定在本地，KV 块在环上转圈。

对于当前块，局部分数矩阵为：

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + M_{ij}
$$

这里 $M_{ij}$ 是这一块对应的 mask，比如 causal mask。因为 softmax 不能简单把每块结果直接平均，所以 Ring Attention 要维护三类累积状态：

- $m$：当前每一行见过的最大 logit
- $l$：当前每一行 softmax 分母的稳定累积值
- $O$：当前每一行加权 value 的未归一化累积值

块更新规则是：

$$
m' = \max(m, \text{rowmax}(S_{ij}))
$$

$$
l' = l \cdot \exp(m - m') + \text{rowsum}(\exp(S_{ij} - m'))
$$

$$
O' = O \cdot \exp(m - m') + \exp(S_{ij} - m')V_j
$$

全部块处理完以后，输出是：

$$
Y_i = \frac{O}{l}
$$

这里的关键点是：每次只看一个分块，但通过 $m, l, O$ 的更新，最终得到的结果与全量 softmax 完全一致。在线 softmax 的本质是“把分块 softmax 重新缩放到同一个数值基准下再累积”。

变量表如下：

| 符号 | 含义 |
|---|---|
| $N$ | 总序列长度 |
| $D$ | 设备数 |
| $B$ | 每设备持有的块长度 |
| $d$ | attention head 维度 |
| $i$ | 当前设备编号 |
| $j$ | 当前轮次对应的 KV 来源设备编号 |
| $t$ | 环形轮次 |
| $M_{ij}$ | 当前块的 mask |
| $m$ | 行最大值统计量 |
| $l$ | softmax 分母累积量 |
| $O$ | 未归一化输出累积量 |

玩具例子可以直接看出它为什么成立。假设只有 1 个 query，4 个 key/value，分到 2 张卡上，得分是 $[1,2 \mid 3,4]$，对应的 value 是 $[10,20 \mid 30,40]$。

第一轮处理本地块 $[1,2]$：

- $m = 2$
- $l = e^{-1}+1 \approx 1.3679$
- $O = 10e^{-1}+20 \approx 23.6788$

第二轮收到远端块 $[3,4]$ 后更新：

- $m' = 4$
- $l' = (e^{-1}+1)e^{-2} + (e^{-1}+1) \approx 1.5530$
- $O' = 23.6788e^{-2} + (30e^{-1}+40) \approx 54.2418$

最终输出：

$$
y = \frac{O'}{l'} \approx 34.92
$$

这个结果和直接对 $[1,2,3,4]$ 做一次全量 softmax 后再乘 $V$ 的结果一致。

流程可以写成伪代码：

```text
初始化 m=-inf, l=0, O=0
本地持有 Q_i, K_i, V_i
current_k, current_v = K_i, V_i

重复 D 轮:
  S = block_attention_score(Q_i, current_k, mask)
  用 online softmax 更新 m, l, O
  把 current_k, current_v 发送给下一张卡
  从上一张卡接收新的 current_k, current_v

输出 Y_i = O / l
```

---

## 代码实现

工程实现通常分两层。第一层管分块、通信、轮次调度；第二层管块级 attention kernel 和 online softmax 更新。对于初学者，可以把它理解成“外层是物流系统，内层是数学计算”。

前向主循环的接口通常类似：

- 输入：`q, k, v, attn_mask, chunk_size`
- 输出：`y`
- 隐式状态：设备编号、总设备数、当前轮次、通信 buffer

下面给一个可运行的 Python 玩具实现。它不是分布式代码，但完整模拟了“按块累积 online softmax”和“全量 attention”结果相等。

```python
import math

def full_attention(scores, values):
    exps = [math.exp(s) for s in scores]
    z = sum(exps)
    return sum(e * v for e, v in zip(exps, values)) / z

def ring_online_attention(scores, values, block_size):
    m = float("-inf")
    l = 0.0
    O = 0.0

    for start in range(0, len(scores), block_size):
        block_scores = scores[start:start + block_size]
        block_values = values[start:start + block_size]

        local_max = max(block_scores)
        new_m = max(m, local_max)

        scaled_old_l = 0.0 if m == float("-inf") else l * math.exp(m - new_m)
        scaled_old_O = 0.0 if m == float("-inf") else O * math.exp(m - new_m)

        exp_block = [math.exp(s - new_m) for s in block_scores]
        block_l = sum(exp_block)
        block_O = sum(e * v for e, v in zip(exp_block, block_values))

        l = scaled_old_l + block_l
        O = scaled_old_O + block_O
        m = new_m

    return O / l

scores = [1.0, 2.0, 3.0, 4.0]
values = [10.0, 20.0, 30.0, 40.0]

y_full = full_attention(scores, values)
y_ring = ring_online_attention(scores, values, block_size=2)

assert abs(y_full - y_ring) < 1e-9
print(round(y_full, 6), round(y_ring, 6))
```

实现结构可以再拆一下：

| 模块 | 职责 |
|---|---|
| chunking | 把序列按 query chunk 和 KV chunk 切开 |
| collective communication | 按 ring 顺序发送和接收 KV 块 |
| stable softmax update | 维护 $m,l,O$，避免数值溢出 |
| mask handling | 保证 causal mask、padding mask 在分块后仍正确 |
| backward replay | 反向按同样块顺序回放或重算 |

一个真实工程例子是：训练 100K+ token 的代码模型时，外层通常会把参数交给 FSDP，attention 则交给 Ring Attention。每张卡固定持有自己的 query chunk，KV 通过 `send/recv` 或 collective primitive 轮转。反向时必须使用与前向相同的块顺序和同样的归一化逻辑，否则梯度会和前向数值不一致。

---

## 工程权衡与常见坑

Ring Attention 的价值来自“显存可扩展”，代价来自“通信变复杂”。因此它不是默认最优解，而是用通信换单卡序列容量。

最常见的问题是 chunk 太小。块越小，每轮传输的数据越少，但轮次不会变少，通信启动开销会变得很重。结果是 GPU 计算时间不足以覆盖通信等待，总吞吐反而下降。一个典型失败配置是：`query_chunk_size` 和 `key_chunk_size` 都设得很小，导致每张卡频繁收发 KV，但每次矩阵乘又太短，整机看起来像在“忙着搬数据”。

常见坑如下：

| 坑 | 后果 |
|---|---|
| chunk 太小 | 通信盖过计算，训练变慢 |
| mask 不一致 | 结果错误，尤其是 causal 场景 |
| padding 不统一 | 不同卡看到的有效长度不同，输出错位 |
| backward 顺序错误 | 梯度不一致或数值异常 |
| 低带宽网络 | Ring 轮转等待时间过长 |
| 只做 attention 并行 | 参数或优化器状态仍然爆内存 |

对应规避建议：

| 问题 | 规避建议 |
|---|---|
| chunk 太小 | 让每轮 GEMM 足够大，优先测吞吐而不是只看显存 |
| mask 不一致 | 在块级别统一生成 mask 规则，尤其检查跨块 causal 边界 |
| padding 不统一 | 先做统一 padding，再做序列切分 |
| backward 顺序错误 | 复用前向相同的块顺序和状态更新逻辑 |
| 低带宽网络 | 优先 NVLink、高速 ICI 或同机箱高带宽互联 |
| 只做 attention 并行 | 与 FSDP、ZeRO、张量并行联合设计 |

一个经验规则是：优先在 NVLink 或同等级高速互联环境下使用，并让 chunk 大小与硬件带宽匹配。Ring Attention 的性能高度依赖“计算是否足够大，从而把通信隐藏在后面”。

---

## 替代方案与适用边界

如果你的上下文还没有长到极端，Ring Attention 往往不是第一选择。很多场景下，FlashAttention、activation checkpointing，甚至更简单的局部窗口注意力，工程代价更低。

可以先看对比：

| 方案 | 是否精确 | 是否全局 | 是否需要额外通信 | 是否适合超长上下文 |
|---|---|---:|---:|---:|
| FlashAttention | 是 | 是 | 否 | 中等，受单卡显存限制 |
| Recompute/Checkpointing | 是 | 是 | 否 | 中等，省激活但不拆序列 |
| Sparse/Window Attention | 否或部分 | 否或部分 | 否 | 适合局部依赖很强的任务 |
| Ring Attention | 是 | 是 | 是 | 是，尤其适合 100K+ |

再看选型边界：

| 场景 | 更适合的方案 |
|---|---|
| 8K 到 32K，上下文不算极长 | FlashAttention 往往足够 |
| 显存吃紧，但还没到多卡切序列 | Checkpointing 更简单 |
| 任务只需要局部依赖 | Sparse/Window 更便宜 |
| 必须保留精确全局 attention，且上下文极长 | Ring Attention 更合适 |

一个真实工程判断是：短到中等上下文训练时，FlashAttention 已经能很好控制显存和吞吐；但当上下文继续拉长到单卡根本放不下完整 attention 激活，并且任务又确实需要全局精确依赖，比如超长代码仓、长法律文档、多模态长视频序列，这时 Ring Attention 才真正有优势。

结论可以压缩成一句话：Ring Attention 不是默认最优解，而是“精确全局长上下文”场景下的专用方案。

---

## 参考资料

1. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) 用于理解 Ring Attention 的方法定义、分布式设定和超长上下文实验目标。  
2. [haoliuhl/ringattention](https://github.com/haoliuhl/ringattention) 用于查看官方实现思路，包括块级调度、通信组织和训练接口。  
3. [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370) 用于理解 blockwise parallelism 的相关方法谱系，以及为什么分块 attention 可以做到精确计算。  
4. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) 用于理解在线 softmax 的数值稳定更新公式，是 Ring Attention 累积推导的数学基础。

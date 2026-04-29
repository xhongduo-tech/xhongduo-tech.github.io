## 核心结论

FlashDecoding 可以先记成一句话：**它是 decode 阶段 attention 的 Split-K 并行实现**。Split-K 的意思是“把原本沿着同一维度串行处理的工作拆成多个分块并行做，再把中间结果规约回来”。在这里，被切分的是长序列上的 `K/V cache`，也就是历史 token 对应的 key 和 value 缓存。

它解决的不是“注意力公式不够先进”，而是**生成阶段的 GPU 并行度不够**。生成一个新 token 时，当前 query 通常只有 1 条，而历史 `K/V` 可能已经有 32k、64k 甚至 128k 个 token。传统 decode kernel 里，很多计算单元拿不到足够的独立任务；FlashDecoding 的做法是把这条很长的 `K/V` 序列按长度切成多个 split，让多个 thread block 或多个 SM 同时读不同分块，最后再把结果按 softmax 规则合并。

新手可以先抓住一个最重要的判断：**FlashDecoding 不是为了减少计算量，而是为了把同样的计算更充分地并行化**。因此它通常在“长上下文、小 batch、一次只生成 1 个 token”的场景收益最大，而不是所有推理场景都更快。

| 方案 | 并行粒度 | 主要瓶颈 | 典型收益点 | 适合场景 |
|---|---|---|---|---|
| 传统 decode attention | 多按 batch、head 分工 | 小 batch 时并行度不足，读 KV 成本高 | 实现简单 | 短上下文、普通推理 |
| FlashDecoding | 在 batch/head 之外，再按 KV 长度切 split | 需要额外规约，但能提高 SM 利用率 | 长序列 decode 吞吐提升 | `batch=1~4`、上下文很长 |
| 纯算力优化但不改并行粒度 | 仍可能受内存带宽限制 | 算得快但数据搬运还是慢 | 中等 | 算力受限而非带宽受限的场景 |

---

## 问题定义与边界

这里讨论的是 Transformer 推理里的 **decode 阶段**，不是 **prefill 阶段**。prefill 可以白话理解为“把已有整段输入一次性读完并建立上下文”；decode 可以理解为“每次只补 1 个新 token，再接着生成下一个”。两者都在算 attention，但硬件行为很不一样。

几个术语先定清：

| 术语 | 定义 | 白话解释 |
|---|---|---|
| prefill | 对整段已有输入做并行计算 | 一次处理很多 token |
| decode | 每步只为新 token 计算输出 | 一次通常只补 1 个 token |
| KV cache | 历史 token 的 key/value 缓存 | 之前算过的注意力记忆，后面反复读取 |
| attention | 用 query 对所有 key 打分，再加权求 value | 看当前 token 应该关注哪些历史位置 |
| Split-K | 沿同一归约维度切块并行，再做规约 | 把一大段工作拆给多人做，最后汇总 |

问题的核心不是 attention 公式本身，而是硬件层面的两个事实：

1. decode 时当前 `Q` 很短，甚至只有一个 query。
2. 历史 `KV cache` 很长，且必须频繁从显存读取。

所以 decode 常见瓶颈不是“乘加不够快”，而是**显存带宽和并行度不足**。更具体地说，算一个新 token 时，单头 attention 的打分是：

$$
s_i = \frac{q \cdot k_i}{\sqrt{d}}
$$

其中 $q$ 是当前 query，$k_i$ 是第 $i$ 个历史 key，$d$ 是 head dimension。这个公式很简单，但如果历史长度是 64k，就意味着你要把 64k 个 key 和 64k 个 value 从缓存里读出来。算术强度不高，搬数据很重，因此很容易变成带宽受限。

一个典型工程边界是：

| 维度 | 常见范围 |
|---|---|
| batch size | 1 到 4 |
| context length | 32k 到 128k |
| head dim | 64 或 128 |
| 场景 | 在线聊天、代码补全、长文档问答 |

这类场景里，单步 decode 很难像 prefill 那样把 GPU 吃满。FlashDecoding 的目标正是这类“最顽固”的瓶颈：**长 KV、短 Q、小 batch**。它不是训练优化，也不是所有 attention 场景的统一解法。

---

## 核心机制与推导

先看标准 attention。对单个 query，设所有 score 为 $s_i$，所有 value 为 $v_i$，输出是：

$$
y = \sum_i \text{softmax}(s)_i \, v_i
$$

展开写就是：

$$
y = \frac{\sum_i e^{s_i} v_i}{\sum_i e^{s_i}}
$$

为了数值稳定，工程实现里通常不会直接算 $e^{s_i}$，而是减去最大值 $m=\max_i s_i$：

$$
y = \frac{\sum_i e^{s_i-m} v_i}{\sum_i e^{s_i-m}}
$$

这一步很重要。数值稳定的白话解释是：避免指数过大溢出，或者指数过小直接下溢成 0。

FlashDecoding 的关键观察是：**分母和分子都可以先在局部算，再做稳定规约**。假设把整条历史序列切成多个 split，第 $r$ 个 split 内部有若干 token。对每个 split，先算：

$$
m_r = \max_j s_{r,j}
$$

$$
l_r = \sum_j e^{s_{r,j} - m_r}
$$

$$
o_r = \sum_j e^{s_{r,j} - m_r} v_{r,j}
$$

其中：

- $m_r$ 是该分块内的最大 score
- $l_r$ 是该分块局部 softmax 的未归一化分母
- $o_r$ 是该分块局部 softmax 的未归一化分子向量

然后对所有 split 做一次全局规约：

$$
m = \max_r m_r
$$

$$
L = \sum_r e^{m_r - m} l_r
$$

$$
O = \sum_r e^{m_r - m} o_r
$$

最后输出：

$$
y = \frac{O}{L}
$$

这就是 FlashDecoding 的核心。重点不是“每块各算各的 softmax 再拼起来”，那样是错的；重点是“每块先保留稳定 softmax 所需的中间量，再做全局校正和规约”。

下面给一个玩具例子。假设只有 4 个 token，分成 2 个 split。

- split 1: `s=[2,1]`, `v=[1,2]`
- split 2: `s=[0,-1]`, `v=[3,4]`

先算 split 1：

- $m_1=2$
- $l_1 = 1 + e^{-1} \approx 1.3679$
- $o_1 = 1 \times 1 + e^{-1} \times 2 \approx 1.7358$

再算 split 2：

- $m_2=0$
- $l_2 = 1 + e^{-1} \approx 1.3679$
- $o_2 = 1 \times 3 + e^{-1} \times 4 \approx 4.4715$

全局合并：

- $m = \max(2,0)=2$
- $L = 1 \times l_1 + e^{-2} \times l_2 \approx 1.5524$
- $O = 1 \times o_1 + e^{-2} \times o_2 \approx 2.3409$
- $y = O/L \approx 1.5073$

如果直接对 4 个 score 做完整 softmax，结果也是同一个值。这说明**按 KV 分块并不会改变数学结果，只要规约方法正确**。

| 标准整体 softmax | 分块 softmax 对应量 |
|---|---|
| 全局最大值 $m$ | 各块最大值 $m_r$ 的最大者 |
| 全局分母 $\sum_i e^{s_i-m}$ | $\sum_r e^{m_r-m} l_r$ |
| 全局分子 $\sum_i e^{s_i-m}v_i$ | $\sum_r e^{m_r-m} o_r$ |
| 最终输出 $y$ | $O/L$ |

真实工程里，这个推导带来的价值是：**每个 split 都可以独立读取一段 KV cache，局部计算结束后只上传少量中间统计量，再做一次 reduction**。这样并行粒度就从“只有 batch/head”扩展到了“batch/head + sequence split”。

---

## 代码实现

实现时通常分三层：

1. 数据切分：把长 `KV cache` 按序列长度切成多个 chunk。
2. 局部计算：每个 chunk 独立算局部 `m_r / l_r / o_r`。
3. 全局归约：把所有 chunk 的中间量合并成最终输出。

下面是一个可运行的 Python 玩具实现。它不是高性能 kernel，但它完整展示了 FlashDecoding 的数值逻辑，并用 `assert` 验证和标准 softmax 一致。

```python
import math

def direct_attention(scores, values):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = sum(exps)
    numer = sum(w * v for w, v in zip(exps, values))
    return numer / denom

def split_flashdecode(scores, values, split_size):
    partials = []

    for start in range(0, len(scores), split_size):
        s_chunk = scores[start:start + split_size]
        v_chunk = values[start:start + split_size]

        m_r = max(s_chunk)
        exps = [math.exp(s - m_r) for s in s_chunk]
        l_r = sum(exps)
        o_r = sum(w * v for w, v in zip(exps, v_chunk))
        partials.append((m_r, l_r, o_r))

    m = max(m_r for m_r, _, _ in partials)
    L = sum(math.exp(m_r - m) * l_r for m_r, l_r, _ in partials)
    O = sum(math.exp(m_r - m) * o_r for m_r, _, o_r in partials)
    return O / L

# 玩具例子
scores = [2.0, 1.0, 0.0, -1.0]
values = [1.0, 2.0, 3.0, 4.0]

y_direct = direct_attention(scores, values)
y_split = split_flashdecode(scores, values, split_size=2)

assert abs(y_direct - y_split) < 1e-9
print(y_direct, y_split)

# 再测一个更一般的例子
scores2 = [0.2, -0.5, 1.3, 2.1, -1.2, 0.7]
values2 = [3.0, 1.0, -2.0, 4.0, 0.5, 2.5]

y_direct2 = direct_attention(scores2, values2)
y_split2 = split_flashdecode(scores2, values2, split_size=3)

assert abs(y_direct2 - y_split2) < 1e-9
print(y_direct2, y_split2)
```

如果把它映射到 CUDA 或 Triton 之类的 GPU 实现，结构通常类似这样：

```text
for each split r in parallel:
    load K_r, V_r
    compute scores s_r = q @ K_r^T / sqrt(d)
    compute local max m_r
    compute local sum l_r = sum(exp(s_r - m_r))
    compute local output o_r = sum(exp(s_r - m_r) * V_r)

m = max_r(m_r)
L = sum_r(exp(m_r - m) * l_r)
O = sum_r(exp(m_r - m) * o_r)
y = O / L
```

一个常见工程组织方式如下：

| 模块 | 职责 | 常见落点 |
|---|---|---|
| Q 读取 | 当前 token 的 query 很短，常驻寄存器更划算 | register |
| K/V chunk 读取 | 每个 block 负责一个或多个 split | global memory 到 shared memory |
| 局部归约 | 求 `m_r`、`l_r`、`o_r` | warp-level 或 block-level reduction |
| 全局归约 | 合并多个 split 的中间量 | 第二阶段 kernel 或 block 间规约 |

真实工程例子可以看在线聊天系统。假设一个 70B 级别模型服务长文档问答，配置大致是：

- `batch=2`
- 上下文长度 64k
- 每步只生成 1 个 token
- 每个 head 的 `d=128`

这时每一步 decode 都要扫过很长的 KV cache。传统实现中，单次工作量不够把所有 SM 吃满；FlashDecoding 则把 64k token 拆成多个 split，比如每个 split 处理 2k 或 4k token，让更多 SM 并发读取和计算。最终单步延迟未必总是线性下降，但吞吐和 GPU 利用率通常会更好，尤其是在小 batch 服务里更明显。

---

## 工程权衡与常见坑

FlashDecoding 不是“开了就更快”的按钮。它的收益来自并行度提升，但代价也很明确：**更多 kernel 协调、更多 reduction、更多中间状态处理**。因此必须讲边界。

先给一个粗判断表：

| 场景 | 是否适合 Split-K | 可能收益 | 主要风险 |
|---|---|---|---|
| 长上下文，`batch=1~4` | 很适合 | decode 吞吐改善明显 | 需要调 split 数 |
| 中等上下文，batch 较大 | 视情况而定 | 收益不稳定 | 额外规约开销可能抵消收益 |
| 很短上下文 | 通常不适合 | 很小 | 启动和合并成本更高 |
| prefill 阶段 | 不是主要目标 | 有别的更优优化路径 | 并行模式不同 |

几个常见坑最值得注意。

第一，**短上下文强行 split**。如果 KV 只有几百个 token，把它切成很多块，等于把一小段工作拆给太多人，协调成本会比实际工作还大。这种情况通常不如直接走普通 decode kernel。

第二，**split 过多导致 reduction 开销变重**。split 数增加后，局部并行度确实提高，但每个 split 都要产出中间量，最后还要做一次稳定规约。并行度和规约成本之间存在最优点，不是越多越好。

第三，**错误地拼接局部 softmax 结果**。这是最危险的逻辑错误。局部 softmax 的归一化基准不同，不能直接拼。必须保留每个 split 的 $m_r$、$l_r$、$o_r$，再做全局 `max + log-sum-exp` 风格的规约。

第四，**KV layout 和 head 映射错误**。工程上经常会遇到 `GQA/MQA`。GQA 是 grouped-query attention，白话解释是“多个 query head 共享较少的 KV head”；MQA 是 multi-query attention，可以理解为“很多 query 共用一组 KV”。这些变体会改变 KV 的存储布局和 head 对齐方式。如果 stride、索引映射、causal mask 任一处出错，数值可能不会立刻 NaN，但结果会悄悄偏掉。

第五，**只看理论，不做基准测试**。FlashDecoding 很依赖具体模型、显卡架构、KV 布局和服务形态。工程里通常需要对不同 `split size`、不同 `num_splits` 做 benchmark，再按上下文长度动态选择是否启用。

一个实用判断规则是：

- 上下文很长，且 decode 明显带宽受限时，优先考虑 FlashDecoding。
- 上下文不长，或 batch 已经足够大时，先测，不要默认它更优。
- `num_splits` 依赖硬件和模型，应该靠 benchmark 选，而不是拍脑袋固定。

---

## 替代方案与适用边界

FlashDecoding 只是在 decode 阶段优化 attention 的一种方法，不是所有推理优化的总称。把它和别的方案分清，工程决策才不会混乱。

| 方案 | 优化对象 | 主要收益 | 适合场景 | 局限性 |
|---|---|---|---|---|
| FlashDecoding | 长上下文 decode attention | 提高小 batch、长 KV 场景并行度 | 在线聊天、代码补全、长文档问答 | 依赖长 KV；需要规约 |
| FlashAttention | 更广义的高效 attention kernel | 改善 attention 的 IO 效率 | 训练、prefill、部分 decode | 不专门针对 decode 并行度不足 |
| PagedAttention | KV cache 的分页管理与访存组织 | 提升服务端缓存管理效率 | 多请求并发服务 | 解决的是缓存组织，不等于 split 规约 |
| Speculative Decoding | 通过草稿模型减少主模型步数 | 降低每个最终 token 的平均成本 | 对延迟敏感且有辅助模型时 | 系统复杂度更高 |
| KV Cache 压缩/量化 | 减少缓存占用和带宽压力 | 显存省、带宽省 | 超长上下文服务 | 可能影响精度 |

可以这样理解边界关系：

- **FlashAttention** 更像“把 attention 这件事本身做得更高效”。
- **FlashDecoding** 更像“在 decode 这个特殊阶段，把 KV 这条长归约轴拆开并行”。
- **PagedAttention** 更偏向“KV cache 在服务系统里怎么存、怎么取”。
- **Speculative decoding** 解决的是“少走几步生成路径”，不是同一步 attention 怎么更快。

因此，什么时候优先考虑 FlashDecoding？

- 在线服务、小 batch。
- 上下文很长。
- profile 后发现 decode 受 KV 读取和并行度限制。

什么时候先考虑别的方案？

- 如果瓶颈在 prefill，不是 decode，优先看 FlashAttention 类优化。
- 如果瓶颈在多请求缓存管理，优先看 PagedAttention。
- 如果瓶颈在每个 token 都要走完整大模型，且系统允许引入辅助模型，优先评估 speculative decoding。
- 如果显存压力特别大，KV 量化和压缩可能比单纯改并行策略更直接。

结论可以压成一句：**FlashDecoding 最适合“长 KV、小 batch、decode 受带宽限制”的服务形态；一旦场景不满足这个前提，它就不一定是优先级最高的方案。**

---

## 参考资料

1. [PyTorch Blog: Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)
2. [Stanford CRFM: FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
3. [FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)
4. [CUTLASS 文档：Efficient GEMM in CUDA 中的 Split-K / Parallelized Reductions](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
5. [FlashInfer 官方博客：Introduce FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)

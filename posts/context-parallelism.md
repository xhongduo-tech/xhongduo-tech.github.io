## 核心结论

Context Parallelism，中文可叫“上下文并行”，意思是把原本沿着时间顺序排开的长序列切到多台设备上，让每台设备只保存自己那一段 token 的激活和缓存，再通过环形注意力完成跨设备的信息交换。它解决的不是“模型参数放不下”，而是“上下文太长，单卡扛不住”。

它的关键价值有三点：

1. Transformer 主体结构不需要改写，注意力语义仍然等价于原始因果注意力。
2. 显存压力从“单卡保存整段上下文”变成“每卡保存一段上下文”，因此可以把上下文从 128K 推到 1M 甚至更高。
3. 在多机多卡场景下，只要通信能被计算覆盖，扩展效率可以接近线性。

一个直观结论是：长上下文的主要瓶颈往往不是参数，而是序列长度带来的激活和 KV 缓存。Context Parallelism 本质上是在序列维度上做分布式切分，再用 Ring Attention，也就是“环形传递注意力所需张量”的方法，把全局上下文重新拼回来。

下面这张表先给出一个工程直觉：

| 场景 | 上下文长度 | 设备数 | 核心做法 | 结果特征 |
|---|---:|---:|---|---|
| 常规单卡注意力 | 128K | 1 | 全量 Q/K/V 都在单卡 | 实现简单，但很快撞显存墙 |
| Context Parallel | 128K | 8 或更多 | 序列切片 + Ring Attention | 单卡内存显著下降 |
| Context Parallel | 1M | 128 | pass-KV / pass-Q 动态切换 | 可扩展到百万级 token |
| Meta 报告工况 | 1M prefill | 128 张 H100 | 多节点上下文并行 | 约 77 秒，约 93% 并行效率 |

“prefill”是推理里把整段提示词一次性编码进 KV 缓存的阶段；长上下文主要就是在这个阶段最重。

一个玩具例子：把 1M 个 token 想成一条 100 万格的长纸带，128 张 GPU 各拿其中一小段。每张卡本地只处理自己手里的格子，但会把当前需要的 K/V 或 Q 沿着环传给邻居，传几轮后，每一段都等价地“见过”全局上下文。于是每张卡不需要保存全部历史，系统整体却保留了全局注意力。

---

## 问题定义与边界

问题先说清楚：为什么长上下文会让普通注意力失效？

设序列长度为 $L$，隐藏维度为 $d$，头数为 $H$。普通自注意力至少会遇到三类成本：

| 成本类型 | 主要随什么增长 | 为什么会成为瓶颈 |
|---|---|---|
| 注意力计算 | 近似随 $L^2$ 增长 | 每个 token 都要看很多历史 token |
| 激活保存 | 随 $L$ 增长，但层数多时非常大 | 训练时反向传播要保留中间结果 |
| KV 缓存 | 随 $L$ 增长 | 推理时历史越长，缓存越大 |

“KV 缓存”就是每层历史 token 的 Key 和 Value 表示，它让后续 token 不必重复计算历史，但代价是上下文越长，缓存越大。

如果只做张量并行或流水并行，参数可以分摊，层也可以分摊，但整条序列通常还是完整地待在每张参与注意力的卡上。于是当 $L$ 很大时，显存还是炸。Context Parallelism 的边界就在这里：它专门处理“序列维度太长”的问题。

再看一个新手版本的空间直觉。假设有 1 米长的序列，要交给 4 张 GPU：

| 方法 | 每张 GPU 持有的数据 | 是否能看到全局上下文 | 风险 |
|---|---|---|---|
| 不切分 | 整整 1 米 | 能 | 单卡显存爆 |
| 只切分不通信 | 25 厘米 | 不能 | 注意力语义错误 |
| Context Parallel | 25 厘米 + 环形交换 | 能 | 需要额外通信设计 |

所以它不是“把长序列切开就结束”，而是“切开后还要保证注意力结果与全局等价”。这就是 Ring Attention 存在的原因。

Context Parallelism 也有明确边界：

1. 它主要解决长序列问题，不直接减少模型参数量。
2. 它在多机上是否划算，取决于带宽 `BW`、单卡计算能力 `C`、序列长度和 KV 规模。
3. 如果上下文本身不长，比如 8K、16K，增加跨卡通信可能得不偿失。
4. 如果只有单卡，它根本不是第一选择，因为没有设备可切分。

真实工程例子可以这样理解：某团队训练一个长文档问答模型，目标上下文是 512K。参数通过 FSDP 或张量并行已经能放下，但训练一跑就 OOM。原因不是参数，而是注意力层里按层累积的激活和长序列 KV。此时把参数切得再细也没用，必须把序列本身切掉，这才是 Context Parallel 的切入点。

---

## 核心机制与推导

核心机制可以分成两步：先切分，再交换。

第一步，按序列维度切分。假设总长度是 $L$，设备数是 $R$，则每张卡大约处理：

$$
L_{\text{local}} \approx \frac{L}{R}
$$

这一步直接降低单卡激活和缓存占用。

第二步，做 Ring Attention。所谓“环”，就是 rank 0 把张量发给 rank 1，rank 1 发给 rank 2，依次下去，最后一张再发回 rank 0。每轮交换后，本地查询就能和新的远端 K/V 或 Q 配对计算一部分注意力，累计多轮后得到完整结果。

这里会出现两个常见策略：

| 策略 | 传什么 | 适合什么场景 |
|---|---|---|
| pass-KV | 传 Key/Value | 新 token 相对多，或 KV 传输更划算 |
| pass-Q | 传 Query | 新 token 较少、历史缓存很长时更划算 |

为什么会有这个切换？因为 Q 和 KV 的字节数不一样。论文给出的一个判断条件是：

$$
\frac{T}{T+P} \le \frac{2N_{KV}}{N_H}
$$

这里：

- $T$ 是本轮新 token 数。
- $P$ 是已缓存历史 token 数。
- $N_{KV}$ 是 KV heads 数。
- $N_H$ 是总 attention heads 数。

白话解释：左边表示“新 token 在总上下文里占多大比例”，右边表示“KV 头和总头数的相对比例”。如果左边更小，说明传 Q 比传整组 KV 更省带宽，此时优先 pass-Q。

看一个玩具数值例子。设：

- $N_{KV}=8$
- $N_H=128$
- $T=6400$
- $P=122400$

则：

$$
\frac{T}{T+P}=\frac{6400}{128800}\approx0.0497
$$

而：

$$
\frac{2N_{KV}}{N_H}=\frac{16}{128}=0.125
$$

因为 $0.0497 < 0.125$，所以 pass-Q 更合适。直观上说，新增 token 相对很少，传查询比传一大堆历史 KV 更省通信。

但“省带宽”还不够，工程上更重要的是：通信能不能藏进计算里。论文还给出了两个常见阈值，用于判断是否能把 SendRecv 和注意力计算重叠：

$$
T \ge \frac{N \cdot C \cdot N_{KV} \cdot e}{2 \cdot N_H \cdot BW}
$$

$$
T + P \ge \frac{N \cdot e \cdot C}{4 \cdot BW}
$$

这里：

- $N$ 是设备数。
- $C$ 可以理解为单位时间的计算吞吐能力。
- $e$ 是元素字节数，比如 FP16 常近似为 2 字节。
- $BW$ 是通信带宽。

白话解释：如果序列足够长，单轮注意力计算耗时就足够大，通信就有机会被“盖住”；如果序列太短，大家还没算多久就得等网络，Context Parallel 的收益会迅速下降。

这也是为什么它特别适合百万级上下文。上下文越长，每轮本地计算越重，越有机会覆盖环形通信。

再把等价性说严谨一点。因果注意力要求位置 $i$ 只能看到位置 $j \le i$ 的 token。Context Parallel 并没有改变这个数学定义，它只是把完整求和拆成多轮局部求和。例如对某个查询 $q_i$：

$$
\text{Attn}(q_i)=\sum_{r=1}^{R}\text{partial\_attn}_r(q_i, K_r, V_r)
$$

每个 rank 只贡献自己那段序列的注意力分量，最后通过数值稳定的 merge 过程合并。只要掩码和 merge 过程正确，结果就与“单设备一次性看完整序列”一致。

真实工程例子：在 128 卡、16 节点的集群上做 1M token prefill，如果每张卡都保留完整上下文，几乎不可能；如果按 128 份切片，每张卡只保留约 1/128 的局部段，再通过 pass-KV/pass-Q 切换，就能把问题从“单卡存不下”变成“多卡协作能否高效完成”，这是一个可以通过网络拓扑和调度优化的问题。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不依赖 GPU，也不真的做分布式通信，只模拟“把序列切成几段，再按段累积因果注意力”的思路，用来验证结果与普通注意力一致。

```python
import math
import numpy as np

def causal_attention(q, k, v):
    # q, k, v: [T, D]
    scores = q @ k.T / math.sqrt(q.shape[1])
    mask = np.triu(np.ones((q.shape[0], k.shape[0])), k=1)
    scores = np.where(mask == 1, -1e9, scores)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v

def context_parallel_sim(q, k, v, world_size):
    # 沿序列维度切分 K/V，模拟 ring 上逐段累积
    T = q.shape[0]
    shard = T // world_size
    outputs = []

    for i in range(T):
        qi = q[i:i+1]
        partial_scores = []
        partial_values = []

        for r in range(world_size):
            start = r * shard
            end = (r + 1) * shard if r < world_size - 1 else T

            # 因果约束：只能看到当前位置以前
            local_end = min(end, i + 1)
            if start >= local_end:
                continue

            kr = k[start:local_end]
            vr = v[start:local_end]
            sr = qi @ kr.T / math.sqrt(q.shape[1])

            partial_scores.append(sr)
            partial_values.append(vr)

        scores = np.concatenate(partial_scores, axis=1)
        values = np.concatenate(partial_values, axis=0)

        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        out = probs @ values
        outputs.append(out)

    return np.concatenate(outputs, axis=0)

np.random.seed(0)
T, D = 8, 4
q = np.random.randn(T, D)
k = np.random.randn(T, D)
v = np.random.randn(T, D)

full = causal_attention(q, k, v)
cp = context_parallel_sim(q, k, v, world_size=4)

assert np.allclose(full, cp, atol=1e-6)
print("context parallel toy simulation matches full causal attention")
```

这段代码的意义不是性能，而是验证“分片后逐段合并”可以保持数学等价。

如果进入真实 PyTorch 工程，思路会变成：

1. 先把 Q/K/V 按序列维度切到设备 mesh 上。
2. 用 `context_parallel(...)` 声明哪些张量是上下文并行相关张量。
3. 在上下文并行作用域中调用 `scaled_dot_product_attention`。
4. 用 `context_parallel_unshard` 或等效步骤恢复全局输出。

一个简化示意如下：

```python
import torch
import torch.nn.functional as F

def attention_block(q, k, v, cp_ctx=None):
    if cp_ctx is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    with cp_ctx:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out

# 伪代码，表达配置重点而非完整可直接启动的分布式脚本
# cp_ctx = context_parallel(
#     device_mesh=mesh,
#     buffers=(q, k, v, rotary_freqs),
#     buffer_seq_dims=(2, 2, 2, 0),
# )
# out = attention_block(q, k, v, cp_ctx=cp_ctx)
# out = context_parallel_unshard(out, ...)
```

这里的 `buffers` 是要一并纳入上下文并行语义的张量集合，`buffer_seq_dims` 表示每个张量的“序列维”在哪个轴上。比如 Q/K/V 常见形状可能是 `[B, H, T, D]`，那序列维就是 `2`。

术语解释一下，“device mesh”就是设备网格，也就是把多张 GPU 组织成一个可命名、可切分的逻辑拓扑。

真实工程里最容易忽略的是那些“不是 Q/K/V，但依赖序列位置”的张量，比如 rotary embedding 的频率缓存 `freq_cis`。它们如果没有一起切分和同步，表面看注意力代码没错，实际上不同 rank 对同一个位置会编码成不同相位，结果直接错。

---

## 工程权衡与常见坑

Context Parallel 不是“开关一开就提速”。它是一套内存与通信的交换。

先看典型坑：

| 常见坑 | 现象 | 对策 |
|---|---|---|
| 漏掉依赖序列维度的 buffer | 输出抖动、对齐错误、精度异常 | 把 rotary 等位置相关张量也纳入 `buffers` |
| 只看理论带宽，不做实测 | 线上收益远低于预期 | 实测 `BW`、计算吞吐和 overlap 比例 |
| 负载切分不均 | 某些 rank 显存先爆或变慢 | 做 load-balanced sharding |
| 盲目使用 pass-Q | merge 成本或 All2All 成本抵消收益 | 按公式和 profiling 动态切换 |
| 上下文不够长也强行上 CP | 通信开销大于收益 | 对短上下文回退单卡或普通并行 |

“load-balanced sharding”就是负载均衡切分，意思是别让某些设备分到特别重的序列段，否则系统会被最慢的卡拖住。

一个典型新手坑是 rotary buffer。故事可以这样理解：4 张卡各自处理一段文本，Q/K/V 都切好了，但位置编码缓存没同步。结果 rank 0 认为自己手里的第 1000 个 token 是全局第 1000 位，rank 1 却把自己的局部第 0 位也当成“第 0 位”编码。这样跨设备做注意力时，就像每个人拿着不同坐标系在说话，最后高频位置关系全部错位。

另一个常见误区是把 pass-Q 当成固定最优。不是。它只是在某些 $T/(T+P)$ 比例下更省。真实集群里还要考虑：

| 维度 | 问题 |
|---|---|
| 网络拓扑 | 跨节点带宽可能远小于节点内 NVLink |
| merge 开销 | 分轮合并注意力结果本身也要算和传 |
| KV 头配置 | GQA/MQA 会改变 Q 和 KV 的字节比例 |
| 推理阶段特征 | decode 与 prefill 的最佳策略常不同 |

真实工程例子：如果你在单节点 8 卡上验证通过，直接搬到 16 节点 128 卡，可能性能突然掉很多。原因不是算法失效，而是跨节点链路比节点内慢得多，原本能覆盖的通信现在盖不住了。此时通常要重新调切换阈值，甚至让系统在不同长度段选择不同 pass 策略。

---

## 替代方案与适用边界

Context Parallel 不是唯一方案，它只是“跨设备长上下文扩展”的强方案。和它容易混淆的还有两类。

| 方案 | 核心思想 | 适合场景 | 不适合场景 |
|---|---|---|---|
| Context Parallel | 切序列到多卡并交换注意力张量 | 超长上下文、多机多卡 | 单卡短上下文 |
| FlexAttention | 用更灵活的注意力 kernel 做局部优化 | 中短上下文、单机推理优化 | 百万级跨节点扩展 |
| MInference | 用动态稀疏模式减少长上下文推理代价 | 单卡长 prompt 推理 | 大规模跨卡全局等价注意力 |

FlexAttention 可以理解为“把注意力算子做得更聪明”，MInference 可以理解为“利用稀疏性，让很多其实不重要的注意力不必完整算”。它们都很有价值，但目标不同。

如果你只有一张 A100，想把 128K prompt 的 prefill 做快一点，优先看稀疏注意力、FlashAttention、MInference 这类方案更现实。因为你没有跨卡，自然也不存在上下文并行。

但一旦需求变成“我要在多节点上稳定处理 1M token，而且希望保持接近完整注意力语义”，Context Parallel 往往就从“可选优化”变成“核心基础设施”。

可以用一个简单判断表来做选型：

| 需求 | 更合适的方向 |
|---|---|
| 单卡，想更快 | FlashAttention / FlexAttention / 稀疏推理 |
| 单卡，想更长 | 稀疏注意力、压缩 KV、MInference |
| 多卡，参数放不下 | 张量并行、FSDP、流水并行 |
| 多卡，上下文放不下 | Context Parallel |
| 多卡，参数和上下文都很重 | 参数并行 + Context Parallel 组合 |

所以最重要的适用边界只有一句话：当瓶颈来自“序列太长”而不是“参数太大”时，Context Parallel 才是对症方案。

---

## 参考资料

- Meta, *Context Parallelism for Scalable Million-Token Inference*
- PyTorch 官方教程，*Introduction to Context Parallel*
- Exxact/Meta 工程文章，*How LLMs Reach 1 Million Token Context Windows*
- PyTorch Blog，*FlexAttention for Inference*
- Microsoft Research，*MInference: Million-Tokens Prompt Inference for Long-context LLMs*

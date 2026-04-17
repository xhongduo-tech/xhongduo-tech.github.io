## 核心结论

Self-Attention 的 Tensor Parallel，简称 TP，指把一个注意力层内部的参数和计算拆到多张 GPU 上并行执行。对多头注意力来说，最自然的切分维度是 head 维度，也就是“每个注意力头各算各的”。因此，总 query head 数 $H$ 可以平均分给 $P$ 张卡，每张卡负责自己的 $H_p=H/P$ 个 query head。

这件事成立的根本原因是：在进入输出投影之前，不同 head 的注意力计算彼此独立。于是每张卡都可以本地完成三件事：

1. 计算自己那部分 Q、K、V 投影。
2. 只对自己负责的 head 做 attention。
3. 生成本地的 attention 输出分块。

最后再把各卡的输出投影结果汇总，得到与单卡等价的层输出。

一个最直观的玩具例子是：把 32 个 attention head 看成 32 条并行通道，TP=4 时，每张卡各负责 8 条。每张卡本地完成这 8 条通道的 QKV 和注意力，再通过输出层通信把结果拼成完整 hidden state。标准多头注意力里，这种切分非常自然，扩展性通常也最好。

但 GQA 和 MQA 会改变这个结论的细节。GQA，Grouped Query Attention，指多个 query head 共享一组 K/V；MQA，Multi-Query Attention，指所有 query head 共享同一组 K/V。它们能显著减少 KV cache，但也带来一个新约束：KV head 数 $G$ 可能小于 TP 度 $P$。一旦 $G<P$，就会出现“有些卡没有本地 K/V 可算”的问题，这时必须广播或复制 K/V，否则本地 attention 无法成立。

可以把整个流程压缩成一句话：

$$
\text{按 head 切 Q/K/V} \rightarrow \text{各卡本地 attention} \rightarrow \text{输出投影并通信汇总}
$$

对初级工程师来说，最重要的判断标准只有两个：

1. 标准 MHA 下，按 head 做 TP 通常是首选。
2. GQA/MQA 下，要先检查 $G$ 和 $P$ 的关系，再决定是否需要广播 K/V。

再补一句更工程化的判断：

| 场景 | 本地 attention 能否直接成立 | 额外代价 |
|---|---|---|
| MHA，且 $H \bmod P = 0$ | 能 | 主要是输出投影后的汇总通信 |
| GQA，且 $G \ge P$ 且 $G \bmod P = 0$ | 通常能 | 仍需输出通信，但 K/V 不必额外复制 |
| GQA，$G < P$ 或 $G \bmod P \ne 0$ | 不能完全本地成立 | 需要广播、复制或重排 K/V |
| MQA，$G=1$ 且 $P>1$ | 不能 | 几乎一定要复制或广播单组 K/V |

这张表背后的逻辑很简单：TP 希望“每张卡都有完整的本地原料”；MHA 的原料最充足，GQA 次之，MQA 最紧张。

---

## 问题定义与边界

先定义本文涉及的几个量：

| 参数 | 说明 | 典型限制 |
|---|---|---|
| $D$ | hidden size，指每个 token 的特征宽度 | 通常需能被 $H$ 整除 |
| $H$ | query head 总数，指总注意力头数 | 通常要求 $H \bmod P = 0$ |
| $G$ | KV head 数或 query group 数 | GQA/MQA 时常有 $G \le H$ |
| $P$ | Tensor Parallel 度，指参与该层切分的 GPU 数 | 受通信拓扑和实现限制 |
| $d_h$ | 单个 head 宽度 | $d_h = D/H$ |
| $H_p$ | 每卡 query head 数 | $H_p = H/P$ |
| $G_p$ | 每卡 KV head 数 | 理想情况 $G_p = G/P$ |

本文讨论的是“注意力层内部的张量并行”，不是 Data Parallel，也不是 Pipeline Parallel。Data Parallel，数据并行，指每张卡都放完整模型，只分不同样本；Pipeline Parallel，流水线并行，指不同层放在不同卡上；而这里的 TP 是“同一层拆给多张卡一起算”。

边界要说清楚，否则很多实现细节会混在一起：

1. 讨论对象是自注意力层，不含 MLP、Embedding、MoE。
2. 关注训练和推理都成立的切分规律，但示例更偏训练框架实现。
3. 默认 hidden size 和 head 数已经合法对齐。
4. 默认序列切分、上下文并行等其他并行方式暂不展开。
5. 输出通信以 AllReduce 或等价的 ReduceScatter + AllGather 表达，不区分底层库细节。

核心公式先摆出来：

$$
H_p=\frac{H}{P}, \qquad G_p=\frac{G}{P}
$$

但这里有一个容易忽略的条件：上式只有在 $H$、$G$ 都能被 $P$ 整除时才是“均匀切分”。标准 MHA 往往满足，GQA/MQA 则未必满足。

再把注意力头和 KV 组的关系写完整：

$$
d_h=\frac{D}{H}, \qquad
r=\frac{H}{G}
$$

其中 $r$ 表示“每个 KV head 服务多少个 query head”。于是：

- MHA 时，$G=H$，所以 $r=1$。
- GQA 时，$1 < r < H$。
- MQA 时，$G=1$，所以 $r=H$。

这个比例 $r$ 很关键，因为它决定了一个 K/V 要被多少个 query head 共享。新手最容易混淆的是：`query head 多` 不等于 `KV head 也多`。在 GQA/MQA 里，两者就是刻意不相等的。

举一个边界很清楚的例子：

- 若 $H=32, P=4$，则每卡有 $H_p=8$ 个 query head。
- 若是标准 MHA，通常 $G=H=32$，于是每卡也有 8 个 KV head。
- 若是 GQA，$G=8$，则每卡只有 2 个 KV head。
- 若是 MQA，$G=1$，而 $P=4$，此时不可能均匀切成每卡 $0.25$ 个 KV head，必须复制或广播。

可以用一句白话理解：TP 希望每张卡都拿到“自己那份原料”后独立做菜；但在 GQA/MQA 中，K/V 这份原料可能比厨师还少，所以只能额外分发。

这也是本文的真正问题定义：

> 在多头注意力中，如何按 head 维度做 TP 切分，并在 GQA/MQA 场景下处理 KV head 数不足带来的通信与实现问题。

为了避免后文读起来抽象，可以先固定一个统一记号：

| 维度符号 | 含义 | 例子 |
|---|---|---|
| $B$ | batch size | 一次并行处理 4 个样本，则 $B=4$ |
| $S$ | sequence length | 每个样本 2048 个 token，则 $S=2048$ |
| $D$ | hidden size | 比如 4096 |
| $H$ | query head 数 | 比如 32 |
| $G$ | KV head 数 | MHA 时 32，GQA 时可能是 8，MQA 时是 1 |

于是输入张量通常写成：

$$
X \in \mathbb{R}^{B \times S \times D}
$$

后文所有公式，都是围绕这个输入做线性投影、按 head 重排、做局部 attention，再做输出汇总。

---

## 核心机制与推导

先从标准多头注意力开始。输入张量记为 $X \in \mathbb{R}^{B \times S \times D}$，其中 $B$ 是 batch size，$S$ 是序列长度。单头宽度是：

$$
d_h=\frac{D}{H}
$$

Q、K、V 三个线性层本质上都是把最后一维从 $D$ 投到若干个 head 上。若按 head 切分，则每张卡只保留自己负责的投影权重，因此本地投影尺寸可以写成：

$$
W_Q^{(p)} \in \mathbb{R}^{D \times (H_p d_h)}
$$

标准 MHA 下，K 和 V 也一样：

$$
W_K^{(p)}, W_V^{(p)} \in \mathbb{R}^{D \times (H_p d_h)}
$$

于是每张卡会得到本地的：

$$
Q^{(p)}, K^{(p)}, V^{(p)} \in \mathbb{R}^{B \times S \times H_p \times d_h}
$$

为了更直观看出“为什么可以本地算”，可以把单个 head 的注意力写成：

$$
\text{head}_i(X)=\text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}\right)V_i
$$

多头拼接后再经过输出投影：

$$
\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
$$

这里有两个阶段：

1. `head_i` 内部计算。
2. 所有 head 拼起来后乘输出矩阵 $W_O$。

TP 能成立，靠的是第 1 阶段彼此独立。也就是在进入 $W_O$ 之前，head 之间没有数学上的相互依赖。

注意力计算发生在本地：

$$
\text{Attn}^{(p)}(Q,K,V)=\text{softmax}\left(\frac{Q^{(p)} {K^{(p)}}^T}{\sqrt{d_h}}\right)V^{(p)}
$$

因为不同 head 本来就互不依赖，所以这里不需要先跨卡通信。也就是说，TP 真正省下来的，是“先把 head 独立算完再通信”，而不是“所有东西都得同步”。

现在看输出投影。各卡得到的 attention 输出可先 reshape 成：

$$
O^{(p)} \in \mathbb{R}^{B \times S \times (H_p d_h)} = \mathbb{R}^{B \times S \times D/P}
$$

这里 $D/P$ 就是每张卡本地持有的通道宽度。输出线性层会把这些局部结果映射回完整 hidden size。工程实现里常见两种写法：

| 写法 | 含义 | 前向常见通信 |
|---|---|---|
| Column Parallel Linear | 输入相同，输出列切分到多卡 | 后续通常需要聚合或让下一层继续并行消费 |
| Row Parallel Linear | 输入按列切开，输出按和相加 | 常见为 AllReduce 或 ReduceScatter |

不管底层细节如何，核心事实只有一个：每张卡只算输出的一部分，最终必须通信得到与单卡一致的结果。

把前向过程压缩成一组更完整的式子：

$$
Q^{(p)} = X W_Q^{(p)}, \qquad
K^{(p)} = X W_K^{(p)}, \qquad
V^{(p)} = X W_V^{(p)}
$$

$$
A^{(p)} = \text{softmax}\left(\frac{Q^{(p)}{K^{(p)}}^\top}{\sqrt{d_h}}\right)
$$

$$
O^{(p)} = A^{(p)}V^{(p)}
$$

$$
Y^{(p)} = O^{(p)} W_O^{(p)}
$$

$$
Y = \sum_{p=1}^{P} Y^{(p)}
$$

最后这一步的求和，就是很多实现里对应的 AllReduce 语义。

一个常见的数值例子：

- hidden size $D=4096$
- query head 数 $H=32$
- KV head 数 $G=8$
- TP 度 $P=4$

于是：

$$
H_p = 32/4 = 8
$$

$$
G_p = 8/4 = 2
$$

$$
D/P = 4096/4 = 1024
$$

这意味着每张卡本地负责：

- 8 个 query head
- 2 个 KV head
- 1024 宽度的本地输出分块

这正是 GQA 的典型模式。因为一个 KV head 会服务多个 query head，所以 query 和 KV 的 head 数不再相同，但只要 $G \ge P$ 且能整除，切分仍然很自然。

下面给出一个简化图示，帮助把机制串起来：

| 阶段 | 每张卡做什么 | 输入是否完整 | 是否需要跨卡通信 |
|---|---|---|---|
| Q 投影 | 生成自己的 $H_p$ 个 query head | 是，同一份 $X$ | 否 |
| K/V 投影 | 生成自己的 $G_p$ 个 KV head | 是，同一份 $X$ | 否，前提是本地有 KV |
| Attention | 对本地 query head 做打分和加权 | 需要本地可用的 K/V | 否 |
| 输出投影 | 计算本地输出分块 | 只拿本地 attention 输出 | 通常需要 |
| 层输出汇总 | 得到完整 hidden state | 各卡局部和 | 是 |

现在看 GQA/MQA 的关键变化。设 query head 数为 $H$，KV head 数为 $G$，则 K/V 投影矩阵变成：

$$
W_K^{(p)}, W_V^{(p)} \in \mathbb{R}^{D \times (G_p d_h)}
$$

只要 $G \ge P$ 且整除，就仍然能做到每卡本地有 K/V。问题出在 $G<P$ 时。例如：

- $H=32$
- $G=2$
- $P=4$

此时 query 还能切成每卡 8 个 head，但 K/V 只有两组，不够 4 张卡平均分。结果是至少有两张卡拿不到本地 K/V。没有 K/V，注意力分数和加权和都没法算，所以必须在 attention 前进行 K/V 广播或复制缓存。

从映射角度，可以把 GQA 写成：

$$
g(i)=\left\lfloor \frac{i}{r} \right\rfloor, \qquad r=\frac{H}{G}
$$

其中 $i$ 是 query head 编号，$g(i)$ 表示它对应的 KV group。举例：

- 若 $H=32, G=8$，则 $r=4$，每 4 个 query head 共用 1 组 K/V。
- 若 $H=32, G=1$，则所有 query head 都映射到同一组 K/V。

这就是 GQA/MQA 与标准 MHA 在 TP 上的本质差异：

| 变体 | Query head | KV head | 每个 KV 服务的 query head 数 | TP 友好度 |
|---|---:|---:|---:|---|
| MHA | $H$ | $H$ | 1 | 最高 |
| GQA | $H$ | $G<H$ | $H/G$ | 中等，取决于 $G$ 与 $P$ |
| MQA | $H$ | $1$ | $H$ | 最差，通常必须复制 KV |

从推导角度看，TP 的收益来源于“独立 head 足够多”；而 GQA/MQA 的收益来源于“KV 更少、cache 更省”。这两者并不总是同向变化。KV 越少，显存和带宽越省；但 KV 太少，又会让 TP 的均匀切分变差，通信反而增加。

真实工程里，这就是为什么同样是 attention 优化，MHA 的 TP 扩展性通常最稳定，而 GQA/MQA 更依赖具体配置。

---

## 代码实现

下面先给一个可直接运行的玩具实现，只演示三件事：

1. 按 head 维度切分 Q 和 K/V。
2. 检查某个 GQA/MQA 配置是否能均匀支撑 TP。
3. 给出 query head 到 KV group 的映射关系。

代码只依赖 Python 标准库，复制后可直接运行。

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AttentionTPPlan:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    tp_size: int
    head_dim: int
    q_heads_per_rank: int
    kv_heads_per_rank: Optional[int]
    output_width_per_rank: int
    need_kv_broadcast: bool
    q_head_to_kv_head: List[int]


def build_attention_tp_plan(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    tp_size: int,
) -> AttentionTPPlan:
    if hidden_size <= 0 or num_heads <= 0 or num_kv_heads <= 0 or tp_size <= 0:
        raise ValueError("all arguments must be positive integers")

    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

    if num_heads % tp_size != 0:
        raise ValueError("num_heads must be divisible by tp_size")

    head_dim = hidden_size // num_heads
    q_heads_per_rank = num_heads // tp_size
    output_width_per_rank = hidden_size // tp_size

    kv_even = (num_kv_heads % tp_size == 0)
    kv_heads_per_rank = num_kv_heads // tp_size if kv_even else None
    need_kv_broadcast = (num_kv_heads < tp_size) or (not kv_even)

    if num_heads % num_kv_heads != 0:
        raise ValueError("for simple GQA mapping, num_heads must be divisible by num_kv_heads")

    queries_per_kv = num_heads // num_kv_heads
    q_head_to_kv_head = [q_head // queries_per_kv for q_head in range(num_heads)]

    return AttentionTPPlan(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tp_size=tp_size,
        head_dim=head_dim,
        q_heads_per_rank=q_heads_per_rank,
        kv_heads_per_rank=kv_heads_per_rank,
        output_width_per_rank=output_width_per_rank,
        need_kv_broadcast=need_kv_broadcast,
        q_head_to_kv_head=q_head_to_kv_head,
    )


def print_plan(plan: AttentionTPPlan) -> None:
    print("hidden_size =", plan.hidden_size)
    print("num_heads =", plan.num_heads)
    print("num_kv_heads =", plan.num_kv_heads)
    print("tp_size =", plan.tp_size)
    print("head_dim =", plan.head_dim)
    print("q_heads_per_rank =", plan.q_heads_per_rank)
    print("kv_heads_per_rank =", plan.kv_heads_per_rank)
    print("output_width_per_rank =", plan.output_width_per_rank)
    print("need_kv_broadcast =", plan.need_kv_broadcast)
    print("q_head_to_kv_head =", plan.q_head_to_kv_head)


if __name__ == "__main__":
    # 例 1：标准 GQA，可均匀切分
    plan1 = build_attention_tp_plan(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        tp_size=4,
    )
    assert plan1.head_dim == 128
    assert plan1.q_heads_per_rank == 8
    assert plan1.kv_heads_per_rank == 2
    assert plan1.output_width_per_rank == 1024
    assert plan1.need_kv_broadcast is False
    assert plan1.q_head_to_kv_head[:8] == [0, 0, 0, 0, 1, 1, 1, 1]
    print("=== plan1 ===")
    print_plan(plan1)

    # 例 2：KV 头不足，必须广播或复制
    plan2 = build_attention_tp_plan(
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=2,
        tp_size=4,
    )
    assert plan2.q_heads_per_rank == 8
    assert plan2.kv_heads_per_rank is None
    assert plan2.need_kv_broadcast is True
    print("\n=== plan2 ===")
    print_plan(plan2)
```

运行结果里最值得关注的是这两个字段：

- `kv_heads_per_rank`
- `need_kv_broadcast`

如果 `kv_heads_per_rank is None` 或 `need_kv_broadcast is True`，说明这个配置无法把 K/V 均匀地分到每张卡，本地 attention 不能完全独立成立。

下面再给一个更接近真实计算图的最小示意。为了保证能运行，这里仍然只用 Python，不依赖 GPU 框架，但保留了真实实现里的关键步骤。

```python
import math


def scaled_dot_product_attention(q, k, v):
    """
    q: [num_q, head_dim]
    k: [num_kv, head_dim]
    v: [num_kv, head_dim]
    返回: [num_q, head_dim]
    这里只做极简示意，不处理 mask、batch、sequence。
    """
    head_dim = len(q[0])
    out = []

    for q_vec in q:
        scores = []
        for k_vec in k:
            score = sum(a * b for a, b in zip(q_vec, k_vec)) / math.sqrt(head_dim)
            scores.append(score)

        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        probs = [x / total for x in exps]

        out_vec = [0.0] * head_dim
        for prob, v_vec in zip(probs, v):
            for i in range(head_dim):
                out_vec[i] += prob * v_vec[i]
        out.append(out_vec)

    return out


def demo_local_attention():
    # 两个 query head，本地只有一个共享的 KV head，模拟 GQA/MQA 风格
    q_local = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    k_local = [
        [1.0, 1.0],
    ]
    v_local = [
        [10.0, 20.0],
    ]

    out = scaled_dot_product_attention(q_local, k_local, v_local)
    print("local attention output =", out)


if __name__ == "__main__":
    demo_local_attention()
```

这段代码虽然是最小版本，但它说明了一个关键事实：只要本地能拿到需要的 K/V，attention 本体就可以独立算完。真正麻烦的不是 softmax 本身，而是“本地是否有足够的 K/V”。

再看更接近真实框架的伪代码。这里用注释标明每一步为什么能本地完成。

```python
# x: [batch, seq, hidden]
# rank: 当前 GPU 的 TP rank
# tp_size: TP 度
# all_reduce: 跨 TP 组做求和汇总

def self_attention_tp_forward(
    x,
    q_proj,
    k_proj,
    v_proj,
    out_proj,
    num_heads,
    num_kv_heads,
    tp_size,
    rank,
):
    head_dim = x.shape[-1] // num_heads
    q_heads_per_rank = num_heads // tp_size

    # 1. 本地 Q 投影，只生成本 rank 负责的 query heads
    q_local = q_proj(x)   # [B, S, q_heads_per_rank * head_dim]
    q_local = q_local.reshape(x.shape[0], x.shape[1], q_heads_per_rank, head_dim)

    # 2. 本地或广播后的 K/V
    if num_kv_heads % tp_size == 0 and num_kv_heads >= tp_size:
        kv_heads_per_rank = num_kv_heads // tp_size
        k_local = k_proj(x).reshape(x.shape[0], x.shape[1], kv_heads_per_rank, head_dim)
        v_local = v_proj(x).reshape(x.shape[0], x.shape[1], kv_heads_per_rank, head_dim)
    else:
        # GQA/MQA 下 KV 头不足时，先复制或广播 K/V
        k_local, v_local = broadcast_or_replicate_kv(x, k_proj, v_proj)

    # 3. 本地 attention
    # 若是 GQA，一个 kv head 会映射给多个 query heads
    attn_local = grouped_scaled_dot_product_attention(q_local, k_local, v_local)

    # 4. 输出投影的本地分块
    # 这里 out_proj 只持有完整权重的一部分
    out_partial = out_proj(attn_local.reshape(x.shape[0], x.shape[1], -1))

    # 5. 跨 rank 汇总，得到与单卡一致的最终输出
    out = all_reduce(out_partial)
    return out
```

这里最容易理解错的是第 5 步。很多初学者以为“前面 attention 都是本地独立的，那后面是不是直接拼起来就行”。不一定。因为输出投影层本身也被切分了，每张卡通常只算了结果的一部分，所以必须通过 AllReduce 或等价通信拿到完整层输出。漏掉这一步，下一层看到的 hidden state 就是不完整的。

给一个真实工程风格的例子。假设某个训练配置是：

- `hidden_size=4096`
- `num_attention_heads=32`
- `num_query_groups=8`
- `tensor_model_parallel_size=8`

这里 `num_query_groups=8` 可以理解为 KV 组数 $G=8$。于是：

- 每卡 query head 数 $H_p = 32/8 = 4$
- 每卡 KV head 数 $G_p = 8/8 = 1$

这是一种很干净的 GQA 配置。每张卡都有 1 组本地 K/V，不需要额外广播；而 query head 仍然能均匀切开，attention 计算也能本地完成。这类配置在大模型训练中很常见，因为它同时兼顾了 TP 扩展性和 KV cache 压缩。

如果把同一个例子改成 `num_query_groups=4, TP=8`，结论就会变：

$$
G=4 < P=8
$$

此时不是数学上不能做 attention，而是工程上不能把 K/V 自然均匀地分配给每张卡。系统必须额外引入 K/V 的广播、复制或者特殊调度，这正是很多实现复杂度上升的来源。

---

## 工程权衡与常见坑

TP 在 attention 里并不是“只要多卡就一定快”。它的收益和代价非常明确：收益来自单卡参数与计算减少，代价来自跨卡通信和切分约束。

先看常见问题表：

| 问题 | 成因 | 典型现象 | 对策 |
|---|---|---|---|
| 某些卡没有 K/V | $G<P$ 或 $G$ 不能整除 $P$ | attention 前就要额外搬运张量 | 广播或复制 K/V；设计配置时尽量保证 $G \ge P$ |
| 输出不一致 | 输出投影后漏掉 AllReduce | shape 正常，但各 rank 数值不同 | 检查 TP 通信路径是否完整 |
| 性能没提升反而下降 | 小模型、小 batch、通信开销过高 | GPU 利用率低，NCCL 时间占比高 | 降低 TP 度，改用 Data Parallel |
| GQA/MQA 质量下降 | KV 共享过强，表达能力下降 | 同算力下效果退化 | 配合 uptraining 或重新调参 |
| shape 对不上 | query heads 和 KV groups 的映射关系写错 | reshape 正常但索引错位 | 明确 head 到 group 的索引规则 |
| 推理 cache 异常 | 训练时分片和推理时 cache 布局不一致 | decode 阶段输出错乱 | 明确 KV cache 的 rank 布局和复制策略 |

第一个坑最重要。很多人看到 GQA 会先想到“KV 更少，所以一定更省、更快”，这只对单卡或 cache 视角成立。到了 TP 视角，KV 太少反而可能让切分失衡。

举一个真实配置判断：

- `num_attention_heads=32`
- `num_query_groups=8`
- `TP=8`

这是较优配置。因为每卡正好 4 个 query head、1 个 KV group，本地 attention 完整成立。

但若改成：

- `num_attention_heads=32`
- `num_query_groups=4`
- `TP=8`

就会出现问题。8 张卡只对应 4 组 KV，意味着一半卡本地没有 K/V，必须额外广播。这样虽然逻辑上仍然能跑，但通信量和实现复杂度都会上升，TP 效率通常会变差。

第二个坑是输出投影的通信顺序。很多实现里，attention 主体没有跨卡依赖，于是调试时容易误以为“看到每卡 attention 输出 shape 正常就结束了”。实际上真正的层输出必须经过汇总。可以把它理解成：

1. 每张卡算出自己的局部贡献。
2. 所有贡献求和或拼接成完整输出。
3. 下一层读取完整结果。

如果第二步缺失，错误往往不会立刻报 shape 异常，而是表现为训练发散、loss 异常、不同 rank 数值不一致。这类问题排查起来比显式报错更难。

第三个坑是“把 TP 当成默认最优”。对小模型，比如参数量不到 1B、只在 2 张卡上训练的场景，TP 可能并不划算。原因很简单：模型本身放得下，通信却是实打实新增的。此时用 Data Parallel 往往更直接，因为它只需要同步梯度，不用把每一层内部都拆开。

第四个坑是 GQA/MQA 的精度代价。GQA/MQA 的目标是减少 KV 头数，也就是减少缓存和带宽，但这通常意味着表达能力有所压缩。工程上常见做法不是“直接替换就完事”，而是结合已有权重继续训练一段时间，也就是 uptraining。对初学者来说，可以把它理解为“结构变了，模型需要再适应一次”。

第五个坑是把“均匀切分”和“逻辑可运行”混为一谈。二者不是一回事：

| 结论 | 含义 |
|---|---|
| 能均匀切分 | 每张卡负载接近，通信最少，通常最好实现 |
| 不能均匀切分但仍可运行 | 需要复制、广播或不均衡分工 |
| 不能直接运行 | 通常是实现没有补齐额外的 K/V 分发逻辑 |

实际排查时，可以按下面顺序检查：

1. `num_heads % tp_size == 0` 是否成立。
2. `num_kv_heads % tp_size == 0` 是否成立。
3. `num_kv_heads >= tp_size` 是否成立。
4. 输出投影后是否有完整的 TP 汇总。
5. 推理阶段 KV cache 的布局是否和训练阶段一致。

这五步比直接看一堆 tensor shape 更有效，因为它们对应的是 TP Attention 最常见的错误源。

---

## 替代方案与适用边界

把 attention 按 head 做 TP，不是唯一方案，只是在大模型里非常常见。真正选型时，通常要和 Data Parallel、Pipeline Parallel 一起看。

| 方案 | 怎么切 | 优点 | 缺点 | 更适合什么场景 |
|---|---|---|---|---|
| Head 维 TP | 把同一层的 head 分到多卡 | 单层显存压力下降，MHA 非常自然 | 需要层内通信，GQA/MQA 受 $G$ 约束 | 大模型、单层放不下、带宽较好 |
| Data Parallel | 每卡一份完整模型，分不同样本 | 实现最简单，生态最成熟 | 模型太大时放不下，参数冗余 | 小中模型、卡数不多 |
| Pipeline Parallel | 不同层放不同卡 | 可支撑更深更大的模型 | 有流水线气泡，调度复杂 | 超大模型、层数很多 |

如果面向零基础读者，可以这样记：

- TP 是把“一个层内部”拆开，多卡共同算一个样本。
- Data Parallel 是把“同一个模型副本”复制多份，多卡各算不同样本。
- Pipeline Parallel 是把“模型纵向切段”，前几层在前面的卡，后几层在后面的卡。

一个非常实际的经验边界是：

1. 模型不大、卡数少时，优先考虑 Data Parallel。
2. 单卡放不下模型，或单层矩阵已经很大时，TP 价值明显上升。
3. 模型特别大时，往往是 TP、DP、PP 组合使用，而不是单独一种。

举一个小模型例子：如果模型不到 1B，只用 2 张卡，且单卡显存足够，那么 Data Parallel 通常更省事。因为这时 TP 带来的层内通信可能比减少的计算还贵。对新手来说，这可以理解成：本来一个人就能做完，硬拆成两个人协作，沟通成本未必值得。

再看大模型例子：如果模型已经到十几亿甚至更大，单个 attention 层的 QKV 和输出投影都很宽，单卡显存和算力都会吃紧，这时按 head 做 TP 是非常自然的切法。特别是在标准 MHA 或者 $G$ 足够大的 GQA 中，TP 的收益通常比较稳定。

因此，TP 的适用边界不是“多卡就上”，而是：

$$
\text{模型规模收益} - \text{通信成本} > 0
$$

而在 GQA/MQA 场景里，还要再加一个条件：

$$
G \text{ 是否足以支撑 } P \text{ 的本地切分}
$$

如果这个条件不满足，就要提前把 KV 广播成本也算进去。

把这件事说得更具体一点，可以得到一个工程判断表：

| 配置特征 | 更可能适合的方案 |
|---|---|
| 模型小、batch 大、单卡放得下 | Data Parallel |
| 单层线性层很宽，显存成为主瓶颈 | Tensor Parallel |
| 模型层数很多，单机放不下完整网络 | Pipeline Parallel |
| 模型超大，且多机多卡训练 | TP + DP + PP 组合 |

对 attention 这一层来说，head 维 TP 之所以常见，不是因为它“唯一正确”，而是因为它正好顺着多头注意力的数学结构切。只要理解这一点，很多实现细节就不会看起来像“框架魔法”。

---

## 参考资料

| 来源 | 说明 | 链接摘要 |
|---|---|---|
| Megatron Core 文档 | NVIDIA 官方文档，涵盖注意力模块、张量并行相关 API 与实现语义，是理解 TP Attention 工程落地的基础资料 | https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.attention.html |
| Megatron Bridge Attention Optimizations | NVIDIA 官方文档，给出 MHA、GQA、MQA 的配置方式与使用背景，适合理解 `num_query_groups` 的工程含义 | https://docs.nvidia.com/nemo/megatron-bridge/latest/training/attention-optimizations.html |
| GQA 论文《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》 | 说明为什么要用更少的 KV head，以及 GQA 相对 MHA/MQA 的折中位置 | https://arxiv.org/abs/2305.13245 |
| MQA 论文《Fast Transformer Decoding: One Write-Head is All You Need》 | 解释 MQA 的动机，即通过极少的 KV 头降低解码阶段的缓存和带宽开销 | https://arxiv.org/abs/1911.02150 |
| Megatron Core Context Parallel 文档 | 虽然重点是 CP，但文档图示清楚展示了 attention 周围 TP/CP 通信边界，适合区分“层内切分”和“序列切分” | https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html |

参考资料的使用方式也可以给初学者一个建议：

1. 先看 GQA/MQA 论文，理解为什么 KV head 会变少。
2. 再看 NVIDIA 的官方文档，理解这些结构在真实训练框架里如何配置。
3. 最后回到代码，把 `num_attention_heads`、`num_query_groups`、`tensor_model_parallel_size` 三个参数放在一起看。

这样读，通常比一开始直接啃源码更容易建立整体图景。

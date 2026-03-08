## 核心结论

Reformer 的 LSH Attention，中文可理解为“基于局部敏感哈希的稀疏注意力”，核心思想是：**先把相似的 token 分到同一个桶里，再只在桶内做注意力计算**。这里的 token 可以先简单理解为“序列里的一个位置对应的一条向量表示”。

标准自注意力要计算任意两个位置之间的相关性，时间复杂度是 $O(N^2)$；当序列长度 $N$ 很大时，计算量和显存占用都会迅速失控。Reformer 把这件事拆成两步：

1. 用 LSH 把相似向量映射到同桶。
2. 只对同桶或邻近桶做注意力。

这样总开销近似变成 $O(N\log N)$。这里的 $\log N$ 主要来自排序，因为 token 在得到桶 ID 后通常要按桶排序，才能把同桶元素排到一起批量计算。

一个直接的数量级对比：

| 序列长度 $N$ | 标准 Attention 交互数 | 若桶大小为 64，LSH 桶内交互近似 |
|---|---:|---:|
| 1,024 | $\approx 1.0$M | $\approx 65$K |
| 4,096 | $\approx 16.8$M | $\approx 262$K |
| 16,384 | $\approx 268.4$M | $\approx 1.05$M |

上表不是精确实现成本，而是帮助理解量级变化。若桶大小固定为 $B$，桶数大约是 $N/B$，那么桶内两两交互总量约为：

$$
\frac{N}{B}\cdot B^2 = N\cdot B
$$

当 $B \ll N$ 时，它远小于标准注意力的 $N^2$。以 $N=16384$、$B=64$ 为例，标准自注意力要考虑约 $2.68\times 10^8$ 对交互；只做桶内交互时，数量级约为 $1.05\times 10^6$。

Reformer 还配合了**可逆残差网络**。残差网络可以理解为“层与层之间保留一条直通路径”的结构；可逆残差则进一步要求这一层的输入能由输出反推回来。这样训练时不必保存每一层激活，激活内存可从 $O(N\cdot L)$ 降到 $O(N)$，其中 $L$ 是层数。

一句话概括：**Reformer 不是在重新定义注意力，而是在用“近似筛选 + 排序分块 + 可逆层”把长序列注意力压到可训练的成本区间。**

---

## 问题定义与边界

问题先说清楚：Reformer 要解决的不是“让注意力更准确”，而是**让超长序列上的注意力仍然算得动、训得起**。

标准 Transformer 的自注意力写成：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

其中 $Q,K,V$ 分别是查询、键、值矩阵。对白话解释：

| 符号 | 直观含义 | 张量形状 |
|---|---|---|
| $Q$ | 当前位置“想找什么” | $N\times d$ |
| $K$ | 每个位置“提供什么匹配信号” | $N\times d$ |
| $V$ | 真正被聚合的信息 | $N\times d_v$ |

困难在于 $QK^\top$ 是一个 $N\times N$ 矩阵。只要序列变长，计算和中间激活都按平方增长。

如果按浮点数个数粗略估算，仅注意力分数矩阵就有：

$$
N^2 \text{ 个元素}
$$

当 $N=16384$ 时，

$$
N^2 = 268{,}435{,}456
$$

这还没算 softmax、中间缓存、梯度、多个头、多层网络的叠加成本。

下面把标准 Attention 和 Reformer 放在同一张表里看：

| 方案 | 时间复杂度 | 额外排序 | 桶大小参数 | 训练激活内存 |
|---|---|---|---|---|
| 标准 Attention | $O(N^2)$ | 无 | 无 | 通常较高 |
| Reformer LSH Attention | 近似 $O(N\log N)$ | 需要 | 需要设置 | 配合可逆层可降到 $O(N)$ |

边界也要说清楚：

1. Reformer 不是精确 attention，而是近似 attention。
2. 它假设“真正重要的交互，大概率发生在相似 token 之间”，因此可以先做近邻筛选。
3. 如果任务非常依赖全局精确两两交互，这种近似可能带来质量损失。
4. 如果序列根本不长，比如几百个 token，标准 attention 往往更简单、更稳、更容易调试。

一个玩具例子：句子里有 8 个 token，其中 “cat” 和 “kitten” 的向量方向接近，LSH 可能把它们分到同桶，于是它们可以互相计算注意力；而 “cat” 和一个无关的标点符号落在不同桶，就不会发生交互。这个近似的前提是：我们愿意接受“先筛候选，再精算”。

一个更贴近工程的例子：在超长文档建模里，文档可能有 16k、32k 甚至 64k 个 token。若直接做标准 attention，模型瓶颈往往不是参数量，而是注意力矩阵和中间激活；而 Reformer 先用哈希找候选，再在局部做精确注意力，才让训练进入可落地区间。

再看一个高维场景：在单细胞 RNA 等任务里，一个样本可能包含上万维特征，文献里常见 $L>17000$ 的设置。若把这些高维位置当作序列处理，标准注意力的两两交互成本极高；而 Reformer 的近邻筛选机制更容易把训练压回可接受的显存范围。

---

## 核心机制与推导

Reformer 的 LSH 机制不是对原始 token 做字符串哈希，而是对**向量表示做随机投影后再分桶**。随机投影可以理解为：**拿若干个随机方向去“观察”向量，看它更接近哪一侧。**

论文里常见写法是：

$$
h(x)=\arg\max [xR;\,-xR]
$$

其中：

| 符号 | 含义 |
|---|---|
| $x\in\mathbb{R}^d$ | 某个 token 的表示向量 |
| $R\in\mathbb{R}^{d\times b}$ | 随机旋转或随机投影矩阵 |
| $[xR;\,-xR]$ | 把正投影和负投影拼接 |
| $h(x)$ | 桶 ID，也就是该 token 被分到哪个桶 |

为什么这样做有意义？因为如果两个向量方向接近，那么它们在随机投影后的相对排序也更可能相近，于是更容易进入同桶。这正是“局部敏感哈希”的含义：**相似对象更容易碰撞到同一个哈希桶**。

从机制上看，完整流程可以概括成四步：

1. 对每个 token 向量做随机投影，得到桶 ID。
2. 按桶 ID 排序，把同桶 token 排到连续位置。
3. 以固定块大小切分，在每个块内做普通 attention。
4. 再把结果按原始顺序还原回去。

如果把排序后的序列写成 $\pi(1),\pi(2),\dots,\pi(N)$，其中 $\pi$ 是由桶 ID 导出的排列，那么注意力不再在全部位置对上计算，而只在局部块中计算：

$$
\mathrm{Attn}_{\text{local}}(Q_{\pi},K_{\pi},V_{\pi})
$$

最后再通过逆排列 $\pi^{-1}$ 把输出映射回原序列顺序。

### 为什么复杂度会下降

假设总长度是 $N$，每个块大小是 $B$，那么：

- 排序代价约为 $O(N\log N)$
- 每个块内 attention 代价为 $O(B^2)$
- 一共有大约 $N/B$ 个块

于是块内总计算量近似为：

$$
\frac{N}{B}\cdot O(B^2)=O(NB)
$$

因此总复杂度常被写成：

$$
O(N\log N + NB)
$$

当 $B$ 固定且远小于 $N$ 时，通常可近似记成 $O(N\log N)$ 或“接近线性到 $N\log N$ 之间”。

### 多轮哈希为什么必要

一个新手容易忽略的问题是：**单轮哈希并不可靠。**

即使两个 token 很相似，也不保证某一轮一定落入同桶。随机投影只保证“更有可能”，不是“必然发生”。因此 Reformer 通常使用多轮哈希，也就是对同一批 token 用多个不同随机矩阵反复分桶。

可以把每一轮 hash 理解为“从一个新的角度重新看这批向量”。某两点在第一轮没碰上，并不意味着第二轮也碰不上。

若单轮中一对相似 token 没落入同桶的概率是 $p_{\text{miss}}$，做 $n_{\text{hash}}$ 轮后全部错过的概率近似变成：

$$
P_{\text{miss,all}}\approx p_{\text{miss}}^{\,n_{\text{hash}}}
$$

这不是覆盖所有实现细节的精确公式，但足够表达工程直觉：**多轮重哈希会指数式降低“完全错过”的概率**。

### 一个更完整的玩具例子

假设有 8 个 token，桶大小设为 2，hash 轮数是 2。

| token | 第 1 轮桶 ID | 第 2 轮桶 ID |
|---|---:|---:|
| t0 | 1 | 3 |
| t1 | 1 | 0 |
| t2 | 0 | 3 |
| t3 | 2 | 2 |
| t4 | 2 | 1 |
| t5 | 3 | 1 |
| t6 | 0 | 0 |
| t7 | 3 | 2 |

如果 t0 和 t2 在第 1 轮没分到一起，但第 2 轮都落入桶 3，它们仍然有机会在某一轮相互看到。多轮不是让每一轮更准，而是让总体召回更高。

### 为什么还要访问相邻块

排序后，同桶 token 会尽量聚集，但工程实现通常还会按固定块大小继续切块。问题在于：**桶边界附近的 token 可能被切到相邻块里。**

例如，某桶里有 130 个 token，而块大小是 64，那么这个桶会被切成 64、64、2 三段。若只看单块，最后那 2 个 token 只能看到极少数同桶伙伴。于是实践里经常采用：

- 当前块 + 前一块
- 当前块 + 左右邻块

这类“邻块可见”策略，本质是在修复分块带来的边界截断问题。

### 一个小结

Reformer 的 LSH 注意力可以拆成三层逻辑：

| 层次 | 作用 | 对应操作 |
|---|---|---|
| 候选召回 | 先找“可能相关”的 token | 随机投影 + 哈希分桶 |
| 批量计算 | 把候选排成连续内存区域 | 排序 + 分块 |
| 近似修补 | 降低漏判和边界损失 | 多轮哈希 + 邻块访问 |

这三层都不能省。只有分桶没有排序，算子难以批量化；只有单轮哈希没有邻块访问，召回容易不稳；只有局部 attention 没有可逆层，训练显存仍然会高。

---

## 代码实现

下面先给一个**最小可运行**的 Python 示例。它不追求高性能，只演示 Reformer 的关键步骤：

1. 随机投影得到桶 ID
2. 按桶排序
3. 在桶内做真实的 scaled dot-product attention
4. 允许访问前一块，缓解块边界问题
5. 把结果还原回原顺序
6. 多轮 hash 后做平均

代码只依赖 `numpy`，复制后可直接运行。

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def lsh_bucket_ids(x, n_buckets, seed=0):
    """
    x: [N, D]
    返回:
        bucket_ids: [N]
    """
    if n_buckets % 2 != 0:
        raise ValueError("n_buckets must be even because we concatenate [+proj, -proj].")

    rng = np.random.default_rng(seed)
    n_tokens, dim = x.shape

    # 随机投影矩阵: [D, n_buckets // 2]
    rotations = rng.standard_normal((dim, n_buckets // 2), dtype=np.float32)

    # 投影后得到 [N, n_buckets // 2]
    projected = x @ rotations

    # 拼接正负投影，得到 [N, n_buckets]
    logits = np.concatenate([projected, -projected], axis=-1)

    # 取最大位置作为桶 ID
    bucket_ids = np.argmax(logits, axis=-1)
    assert bucket_ids.shape == (n_tokens,)
    return bucket_ids


def attention_block(q, k, v):
    """
    标准缩放点积注意力
    q: [M, D]
    k: [T, D]
    v: [T, D]
    返回:
        out: [M, D]
    """
    scale = np.sqrt(q.shape[-1]).astype(np.float32)
    scores = (q @ k.T) / scale
    probs = softmax(scores, axis=-1)
    out = probs @ v
    return out


def reformer_lsh_round(x, bucket_size=4, n_buckets=4, seed=0):
    """
    单轮 LSH attention 演示实现
    x: [N, D]
    返回:
        y: [N, D]
        bucket_ids: [N]
        order: [N]
    """
    n_tokens, dim = x.shape
    if n_tokens % bucket_size != 0:
        raise ValueError("For this toy demo, N must be divisible by bucket_size.")

    # 1. 哈希分桶
    bucket_ids = lsh_bucket_ids(x, n_buckets=n_buckets, seed=seed)

    # 2. 稳定排序，把同桶 token 尽量排到一起
    order = np.argsort(bucket_ids, kind="stable")
    x_sorted = x[order]
    bucket_sorted = bucket_ids[order]

    # 3. 分块
    n_blocks = n_tokens // bucket_size
    x_blocks = x_sorted.reshape(n_blocks, bucket_size, dim)
    b_blocks = bucket_sorted.reshape(n_blocks, bucket_size)

    y_blocks = np.zeros_like(x_blocks)

    # 4. 每个块看自己 + 前一块
    for i in range(n_blocks):
        q = x_blocks[i]  # [B, D]

        if i == 0:
            k = x_blocks[i]
            v = x_blocks[i]
        else:
            k = np.concatenate([x_blocks[i - 1], x_blocks[i]], axis=0)  # [2B, D]
            v = k

        y_blocks[i] = attention_block(q, k, v)

    # 5. 合并并恢复原顺序
    y_sorted = y_blocks.reshape(n_tokens, dim)
    y = np.zeros_like(y_sorted)
    y[order] = y_sorted
    return y, bucket_ids, order


def reformer_lsh_attention(x, bucket_size=4, n_buckets=4, n_hashes=4):
    """
    多轮 LSH attention
    将每一轮结果求平均
    """
    round_outputs = []
    round_bucket_ids = []

    for seed in range(n_hashes):
        y, bucket_ids, _ = reformer_lsh_round(
            x,
            bucket_size=bucket_size,
            n_buckets=n_buckets,
            seed=seed,
        )
        round_outputs.append(y)
        round_bucket_ids.append(bucket_ids)

    y = np.mean(round_outputs, axis=0)
    return y, round_bucket_ids


def main():
    # 8 个 token, 每个 token 4 维
    x = np.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.1],
        [0.0, 0.1, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
        [1.0, 1.1, 0.0, 0.0],
        [0.1, 0.0, 1.0, 1.1],
        [0.8, 0.9, 0.1, 0.0],
        [0.0, 0.2, 0.8, 1.0],
    ], dtype=np.float32)

    y, round_bucket_ids = reformer_lsh_attention(
        x,
        bucket_size=4,
        n_buckets=4,
        n_hashes=4,
    )

    print("input shape :", x.shape)
    print("output shape:", y.shape)
    print("round bucket ids:")
    for i, ids in enumerate(round_bucket_ids):
        print(f"  round {i}: {ids.tolist()}")

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    print("ok")


if __name__ == "__main__":
    main()
```

运行后你会看到每一轮的桶 ID 不同，这正是“多轮重哈希”的效果。只要输出形状与输入一致、数值有限，就说明这个玩具流程跑通了。

### 这段代码和真实 Reformer 还有哪些差距

上面的实现是教学版，不是生产版。它和真实 Reformer 的差别主要在这里：

| 教学实现 | 真实实现 |
|---|---|
| 单头 attention | 多头 attention |
| 只用一个输入 `x` 同时充当 $Q,K,V$ | 会有独立的线性投影层 |
| 块内直接算 attention | 需要更细致的 masking、去重、因果约束 |
| 简单平均多轮输出 | 真实实现会处理不同轮的打分与聚合细节 |
| 只看当前块和前一块 | 邻块规则可能更复杂 |

但对新手来说，先把这 6 个步骤吃透，比直接阅读完整框架代码更有效：

1. 向量哈希
2. 稳定排序
3. 分块
4. 块内 attention
5. 逆置换还原
6. 多轮平均

### 如果写成更接近工程实现的伪代码

如果把 $Q,K,V$ 显式分开，逻辑可以写成：

```python
def lsh_attention(q, k, v, bucket_size, n_hashes):
    outputs = []

    for round_id in range(n_hashes):
        bucket_ids = hash_vectors(q, seed=round_id)
        order = stable_sort(bucket_ids)

        q_sorted = q[order]
        k_sorted = k[order]
        v_sorted = v[order]

        q_blocks = split_into_blocks(q_sorted, bucket_size)
        k_blocks = split_into_blocks(k_sorted, bucket_size)
        v_blocks = split_into_blocks(v_sorted, bucket_size)

        round_out = []
        for block_id in range(len(q_blocks)):
            q_cur = q_blocks[block_id]
            k_ctx, v_ctx = attach_neighbor_blocks(k_blocks, v_blocks, block_id)

            scores = q_cur @ k_ctx.T / sqrt(q.shape[-1])
            probs = softmax(scores, axis=-1)
            round_out.append(probs @ v_ctx)

        round_out = merge_blocks(round_out)
        outputs.append(invert_permutation(round_out, order))

    return average(outputs)
```

新手要重点看懂三件事：

1. `hash_vectors` 负责“粗筛候选”。
2. `stable_sort` 负责把候选排在一起，便于批量矩阵计算。
3. `invert_permutation` 负责把结果打回原位置，否则序列顺序会乱。

### 可逆残差为什么能省内存

可逆残差的最小形式写成：

$$
y_1=x_1+F(x_2),\qquad y_2=x_2+G(y_1)
$$

则反推时：

$$
x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2)
$$

普通残差层反向传播时，通常要保留中间激活；可逆层则可以在反向时从输出重建输入，所以不必保存每层完整激活。

这部分的意义不是“前向更快”，而是“训练更省显存”。因此 Reformer 的两个核心收益要分开看：

| 组件 | 主要解决的问题 |
|---|---|
| LSH Attention | 降低超长序列下的注意力计算成本 |
| 可逆残差 | 降低深层网络训练时的激活内存 |

一个真实工程例子：做 16k 到 64k 长度的文档建模时，标准 attention 的瓶颈常常不是参数量，而是中间激活和注意力矩阵。Reformer 的工程价值在这里最明显，因为它同时压缩了计算和激活存储。

---

## 工程权衡与常见坑

Reformer 的好处明确，但坑也很集中，主要都来自“近似”和“可逆”两件事。

| 问题 | 现象 | 原因 | 常见补救 |
|---|---|---|---|
| 哈希漏判 | 本该相关的 token 没相互注意 | 相似向量落入不同桶 | 增加 `n_hashes`，允许邻桶访问 |
| 桶分布不均 | 某些桶过满，某些桶几乎空 | 数据分布偏斜，随机投影不均 | 调整桶大小，监控桶直方图，多轮 hash |
| 复杂度回升 | 实际速度没快多少 | 桶太大、邻桶太多、轮数太多 | 控制 `bucket_size` 与 `n_hashes` |
| 训练不稳定 | loss 抖动或效果退化 | 近似注意力误差累积 | 从短序列预热，逐步拉长序列 |
| 可逆层出错 | 反向重构失败 | 模块不是严格可逆 | 避免在可逆路径中混入不可逆操作 |

先看哈希漏判。即使两个 token 真正相似，也不保证某一轮一定分到同桶。多轮 hash 的意义就是降低漏掉它们的概率。若单轮漏判概率为 $p$，则多轮后全漏的概率近似为：

$$
p^{n_{\text{hash}}}
$$

所以 `n_hashes=8` 这类设置，本质是用额外常数成本换召回率。

再看桶大小。`bucket_size` 太小，会让真正相关的点被切散；太大，又会让每个桶内 attention 变重，接近回到 $O(N^2)$。实践里至少要同时看两项指标：

1. 每个桶的平均长度。
2. 每个桶长度的方差。

如果平均长度合适但方差很大，说明哈希分布不均；这时不能只看“平均桶长正常”就放心。

一个典型新手错误是：把桶大小设成 64，hash 轮只设 1，然后发现某些桶塞进了 100 多个 token，而另一些几乎空。这时不是简单把桶再放大，而应优先检查：

1. 哈希轮数是否过少。
2. 排序后分块策略是否切碎了桶。
3. 是否允许邻块 attend。
4. 输入表示是否已经坍缩，导致很多 token 太相似。

### 为什么理论复杂度好看，实际速度却未必总赢

Reformer 的理论复杂度通常优于标准 attention，但实际 wall-clock time，也就是实际耗时，不一定总赢。原因有三类：

| 开销来源 | 说明 |
|---|---|
| 排序开销 | 哈希后必须重排 token，GPU 对排序通常不如矩阵乘法友好 |
| 内存访问不规则 | 分桶、切块、逆置换都会带来额外的数据搬运 |
| 并行效率下降 | 稀疏或不规则块操作，往往不如大矩阵乘法吃满硬件 |

因此当序列长度不够长时，标准 attention 反而可能更快。实践中常见的分界思路不是“复杂度更低就必胜”，而是看下面三件事：

1. 序列长度是否已经进入标准 attention 的显存或耗时瓶颈区。
2. 目标硬件是否擅长排序和不规则张量操作。
3. 当前框架对稀疏注意力的内核支持是否成熟。

### 可逆残差的坑更隐蔽

可逆残差要求某一层的输入能从输出恢复，所以不是所有常见模块都能直接塞进去。下面这些操作若处理不当，就会破坏可逆性：

| 操作类型 | 风险 |
|---|---|
| 随机丢弃信息的算子 | 输出不足以唯一确定输入 |
| 改变维度的算子 | 反推时信息丢失 |
| 依赖外部缓存状态的模块 | 反向重建时状态不一致 |
| 数值不稳定的变换 | 重构误差在深层累积 |

结果是：前向能跑，反向重构时却还原不回去，调试难度高于普通残差网络。

一个常见判断标准是：**如果你无法写出“如何从输出反推出输入”，那它就不该直接放进可逆主路径。**

### 一个实用的调参顺序

新手上手 Reformer 时，不要同时乱改所有超参数。更稳的顺序通常是：

1. 先固定较短序列，验证训练闭环能跑通。
2. 再调 `bucket_size`，观察桶长度直方图。
3. 再增加 `n_hashes`，观察召回和耗时。
4. 最后再把序列长度拉高，评估显存与吞吐。

这个顺序的好处是：每一步都只新增一类复杂度，不会把问题混在一起。

---

## 替代方案与适用边界

Reformer 不是唯一的长序列方案。它的特点是“**哈希筛候选 + 可逆省内存**”。如果问题不适合哈希近邻，还可以考虑其他路线。

| 方案 | 主要思想 | 时间复杂度 | 是否依赖哈希 | 是否强调可逆 | 实现难度 |
|---|---|---|---|---|---|
| Reformer | LSH 分桶后做局部 attention | 近似 $O(N\log N)$ | 是 | 是 | 较高 |
| Performer | 随机特征近似 softmax 核 | 近似线性 | 否 | 否 | 中等 |
| Linformer | 低秩投影压缩序列维度 | 近似线性 | 否 | 否 | 中等 |
| Sparse Transformer | 人工设计稀疏连接模式 | 依模式而定 | 否 | 否 | 中等到较高 |
| 标准 Transformer | 全连接 attention | $O(N^2)$ | 否 | 否 | 最低 |

可以把它们理解成不同的“筛子”：

1. Reformer 的筛子是 hash 桶，按向量相似性先分组。
2. Performer 的筛子是随机特征映射，把 softmax attention 改写成线性形式。
3. Linformer 的筛子是低秩假设，认为长序列注意力矩阵可以被压缩。
4. Sparse Transformer 的筛子是人为设计的连接图，先规定谁能看谁。

这些方法解决的是同一个大问题：**标准 attention 的平方复杂度太贵。**  
但它们对“哪些交互值得保留”的假设不同。

### Reformer 更适合哪些场景

| 场景特征 | 为什么适合 |
|---|---|
| 序列极长 | 标准 attention 的 $N^2$ 已成为主瓶颈 |
| 语义近邻结构明显 | 相似 token 更可能真的需要相互作用 |
| 显存紧张 | 可逆层能显著降低激活内存 |
| 团队接受更高实现复杂度 | 可以换取更长上下文窗口 |

具体说，Reformer 更适合这些场景：

1. 序列极长，$N$ 已经让标准 attention 明显不可接受。
2. 任务允许近似注意力，不要求精确全连接交互。
3. 显存压力大，且愿意接受更复杂的训练与调试流程。
4. 数据里“相似 token 应优先交互”的结构比较强。

### Reformer 不太适合哪些场景

| 场景特征 | 为什么不适合 |
|---|---|
| 序列较短 | 排序和重排的额外开销可能得不偿失 |
| 强依赖全局精确交互 | 哈希漏判代价高 |
| 团队更看重稳定和简单 | Reformer 调试成本明显更高 |
| 部署环境不友好 | 不规则算子和排序未必有最佳硬件支持 |

不太适合这些场景：

1. 序列较短，二次复杂度并不是瓶颈。
2. 团队更在意实现简单、调试方便和推理稳定。
3. 任务对全局精确依赖很强，哈希漏判代价高。
4. 部署环境不喜欢排序、重排这类不规则算子。

### 如何做选择

如果把问题压缩成一句判断标准，可以这样看：

- 你的核心矛盾是“长序列算不动、显存不够”，Reformer 值得考虑。
- 你的核心矛盾是“实现要简单、训练要稳、部署要省心”，优先看更简单的替代方案。

一句话概括选择标准：**如果你的核心矛盾是长序列算不动，Reformer 值得考虑；如果你的核心矛盾是训练稳定性和实现复杂度，优先看更简单的替代方案。**

---

## 参考资料

下面给出按用途划分的参考资料，避免只列名字不说明用途。

| 资料 | 类型 | 适合解决的问题 |
|---|---|---|
| *Reformer: The Efficient Transformer* | 原论文 | 查核心定义、复杂度分析、可逆残差与 LSH Attention 原始表述 |
| 官方或社区实现代码 | 实现参考 | 看桶 ID、排序、逆置换、邻块访问的具体写法 |
| 论文解读文章 | 二手综述 | 先建立直观图景，再回头读原文公式 |
| 长序列建模对比文章 | 横向比较 | 判断 Reformer 和 Performer、Linformer 的差别 |
| 高维生物数据或长文档应用论文 | 应用案例 | 评估它在真实任务中的收益与代价 |

建议阅读顺序如下：

1. 先看原论文摘要、模型图和复杂度表，明确它到底解决什么问题。
2. 再看 LSH attention 的公式定义，重点盯住 `hash -> sort -> chunk -> attend -> unsort` 这条链。
3. 然后看一份简化实现，确认桶 ID、排序和逆置换怎么落地。
4. 最后再看可逆残差部分，因为它和稀疏注意力解决的是不同瓶颈。

如果你是第一次接触这类模型，读参考资料时建议优先盯住下面几个关键词：

| 关键词 | 为什么重要 |
|---|---|
| `bucket_size` | 决定局部 attention 的候选规模 |
| `n_hashes` | 决定漏判概率和额外常数开销 |
| `stable sort` | 决定同桶 token 能否被连续批处理 |
| `neighbor chunks` | 决定边界 token 会不会被切散 |
| `reversible residual` | 决定训练时的激活内存策略 |

最终要记住的是：Reformer 的创新点不是某个单独公式，而是把几件事连成了一条完整工程链路：

1. 用 LSH 保留“更可能重要”的交互。
2. 用排序和分块把稀疏计算变成可批处理操作。
3. 用多轮 hash 和邻块访问修补近似误差。
4. 用可逆残差压低训练内存。

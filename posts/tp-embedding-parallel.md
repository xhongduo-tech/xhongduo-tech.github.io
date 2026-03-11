## 核心结论

Tensor Parallel，简称 TP，指的是“同一层参数拆到多张 GPU 上同时计算”。对 Embedding 层，工程上最常见的切分方式不是沿隐藏维 $H$ 切，而是沿词表维 $V$ 切。也就是把整张词表按行切成 $TP$ 份，每张卡只保留自己负责的那一段词向量。

如果原始 Embedding 权重形状为：

$$
W_{embed}\in \mathbb{R}^{V\times H}
$$

那么第 $r$ 张卡保存的本地权重大致是：

$$
W_{embed}^{(r)}\in \mathbb{R}^{\frac{V}{TP}\times H}
$$

单卡参数量就从：

$$
V\times H
$$

下降到：

$$
\frac{V}{TP}\times H
$$

大词表场景下，这个收益很直接。例如词表 $V=128K$、隐藏维 $H=8192$、数据类型是 BF16，完整 Embedding 约 2 GB；若 $TP=8$，单卡只保留约 256 MB 的 Embedding 权重。

Embedding 的前向恢复方式也很固定：

1. 每张卡只对“自己负责词表区间内”的 token 返回真实词向量。
2. 对区间外 token，返回全 0 向量。
3. 对所有卡的局部输出做一次按位求和的 AllReduce。

于是得到的最终输出与单卡完整查表完全一致：

$$
\text{EmbeddingOut}
=
\text{AllReduce}\big(\text{LocalLookupWithZeroMask}(token\_ids)\big)
$$

这件事成立的原因不是“求和碰巧有效”，而是每个 token 在整个 TP 组里只会命中唯一一张卡。最终求和时，只有那一张卡贡献真实向量，其余卡贡献 0，所以结果就是正确的词向量本身。

最小例子先看一遍。设词表大小 $V=8$，$TP=2$，GPU0 负责词 id `[0,4)`，GPU1 负责 `[4,8)`，输入 token id 是 `[0,4,3]`。则局部查表结果如下：

| GPU | 负责词区间 | token 0 | token 4 | token 3 |
| --- | --- | --- | --- | --- |
| GPU0 | `[0,4)` | `emb_0` | `0` | `emb_3` |
| GPU1 | `[4,8)` | `0` | `emb_4` | `0` |

AllReduce 求和后得到：

$$
[emb_0,\ emb_4,\ emb_3]
$$

与普通 Embedding 查表完全等价。

真实工程里，这种做法主要解决两类问题：

| 问题 | 为什么会出现 |
| --- | --- |
| Embedding 和 LM Head 太占显存 | 大词表下，$V\times H$ 本身就很大 |
| 输入 Embedding 与输出 LM Head 要共享权重 | GPT 类模型常要求同一份词表权重同时用于输入查表和输出投影 |

因此，Vocab Parallel Embedding 不是一个“可有可无的小优化”，而是大词表 GPT/LLaMA 类训练系统里的标准设计。

---

## 问题定义与边界

Embedding 的定义很简单：把离散的 token id 映射为连续向量，本质上是一个按行索引的查表矩阵。若词表大小是 $V$，隐藏维是 $H$，数据类型字节数是 $d$，则完整 Embedding 的参数内存近似为：

$$
\text{mem}_{embed}=V\times H\times d
$$

例如：

$$
V=128000,\quad H=8192,\quad d=2
$$

则：

$$
128000\times 8192\times 2\approx 2.10\times 10^9\text{ bytes}\approx 2.1\text{ GB}
$$

这还只是输入 Embedding 一份权重。如果模型采用“输入 Embedding 与输出 LM Head 权重共享”，那么训练系统必须保证：

1. 存储上仍然只保留一份逻辑权重。
2. 前向和反向都与未切分实现数值等价。
3. 切分方式对上层 Transformer block 尽量透明。

本文讨论的边界集中在下表这一类问题：

| 项目 | 约束或目标 |
| --- | --- |
| 词表大小 | 常见为 50K、100K、128K，甚至更大 |
| TP 组大小 | 需要把词表切到多张卡，通常希望尽量均匀 |
| 权重共享 | 输入 Embedding 与输出 LM Head 共享同一份词表权重 |
| 数值等价 | 前向输出、梯度、loss 需与普通 Embedding 一致 |
| 通信条件 | 需要支持 AllReduce、AllGather、ReduceScatter 等集体通信 |
| 适用模型 | GPT、LLaMA 这类自回归语言模型最常见 |

这里要先分清两个概念。

第一，参数切分。它回答的是“谁存哪部分权重”。  
第二，激活恢复。它回答的是“后续层最终要看到什么形状的数据”。

Vocab Parallel Embedding 的设计是：

| 对象 | 处理方式 |
| --- | --- |
| 参数 | 按词表维切分 |
| 输出激活 | 恢复为标准的 $B\times S\times H$ |

其中：

- $B$ 是 batch size，表示一次并行处理多少条样本。
- $S$ 是 sequence length，表示每条样本有多少个 token。
- $H$ 是 hidden size，表示每个 token 向量的维度。

也就是说，Embedding 内部虽然已经分布式切分，但对后续 Transformer block 来说，它仍然像普通单卡 Embedding 一样工作，输出仍是标准隐藏表示。

再看一个更贴近真实工程的数值例子。假设：

$$
V=131072,\quad H=8192,\quad TP=8
$$

则每张卡仅保存：

$$
\frac{131072}{8}=16384
$$

行词向量，本地权重形状为：

$$
[16384,\ 8192]
$$

对上层模块而言，拿到的仍然是：

$$
[B,\ S,\ 8192]
$$

而不是某种“被切开的 embedding 激活”。这正是“内部切分，对外透明”的含义。

这个设计还有一个容易忽略的边界：它主要降低的是参数显存，不直接降低最终 Embedding 输出的大小。也就是说，若你的主要瓶颈是长序列激活而不是大词表权重，那么只做 Vocab Parallel 还不够，往往还需要 Sequence Parallel、Activation Checkpointing 或 Context Parallel 这类机制。

---

## 核心机制与推导

### 1. 按词表连续区间切分

设词表总大小为 $V$，TP 组大小为 $t$。最简单的均匀切分方式是连续区间切分。第 $r$ 张卡负责：

$$
[start_r,\ end_r)
$$

其中：

$$
start_r=r\times \frac{V}{t},\qquad end_r=(r+1)\times \frac{V}{t}
$$

若 $V$ 不能被 $t$ 整除，工程上常见有两种处理方式：

| 处理方式 | 做法 | 代价 |
| --- | --- | --- |
| 词表补齐 | 把 $V$ pad 到 $t$ 的整数倍 | 会引入少量空行 |
| 不等长切分 | 让最后几张卡多或少一部分行 | 实现和检查更复杂 |

为了让后续初始化、前向、权重加载逻辑更简单，很多系统会优先选择“补齐后再均分”。

每张卡的本地参数量近似为：

$$
\text{params\_per\_card}\approx \frac{V}{t}\times H\times d
$$

这说明 Embedding 参数显存会随着 TP 近似线性下降。

### 2. 本地查表加 zero mask

关键步骤是：每张卡只处理自己负责区间内的 token。

设输入 token 张量为：

$$
I\in \mathbb{N}^{B\times S}
$$

对第 $r$ 张卡，定义布尔 mask：

$$
M_r = \mathbf{1}[start_r \le I < end_r]
$$

这里的 $\mathbf{1}[\cdot]$ 表示指示函数，条件成立时取 1，否则取 0。白话讲，它就是“这个 token 是否归当前 rank 负责”的标记。

把全局 token id 映射到本地下标：

$$
I_r^{local}=I-start_r
$$

但只有 mask 为 1 的位置才允许真正参与查表。于是第 $r$ 张卡的局部输出是：

$$
Y_r = \text{Embedding}(I_r^{local}, W_{embed}^{(r)}) \odot M_r
$$

其中 $\odot$ 表示逐元素乘法。因为 $M_r$ 的形状是 $[B,S]$，实际实现时会扩成 $[B,S,1]$ 再与 embedding 输出相乘。

这一步的目的非常具体：保证非本卡 token 在当前卡上的贡献严格为 0。

### 3. AllReduce 恢复完整输出

每张卡得到的 $Y_r$ 形状都相同，都是：

$$
Y_r\in \mathbb{R}^{B\times S\times H}
$$

最终输出通过一次按位求和恢复：

$$
Y=\sum_{r=0}^{t-1} Y_r
$$

工程实现上就是：

$$
Y=\text{AllReduce}_{sum}(Y_r)
$$

为什么求和一定正确？设某个 token id 为 $u$，且它只落在第 $k$ 张卡负责区间内，则：

$$
M_k(u)=1,\qquad M_r(u)=0\ \text{for }r\neq k
$$

因此：

$$
Y_k(u)=W_{embed}[u],\qquad Y_r(u)=0\ \text{for }r\neq k
$$

所以：

$$
\sum_{r=0}^{t-1}Y_r(u)=W_{embed}[u]
$$

得到的就是单卡 Embedding 对 token $u$ 的标准输出。

### 4. 一个完整的最小例子

设：

$$
V=8,\quad H=3,\quad t=2
$$

完整权重为：

$$
W=
\begin{bmatrix}
emb_0\\
emb_1\\
emb_2\\
emb_3\\
emb_4\\
emb_5\\
emb_6\\
emb_7
\end{bmatrix}
$$

GPU0 负责词表行 `[0,4)`，GPU1 负责 `[4,8)`。输入为：

$$
I=[0,4,3]
$$

则：

| token id | 所属 rank | GPU0 输出 | GPU1 输出 | 求和结果 |
| --- | --- | --- | --- | --- |
| 0 | GPU0 | `emb_0` | `0` | `emb_0` |
| 4 | GPU1 | `0` | `emb_4` | `emb_4` |
| 3 | GPU0 | `emb_3` | `0` | `emb_3` |

这个例子看起来简单，但已经包含了真实实现的全部本质：

1. 局部命中。
2. 非命中置零。
3. 全局求和恢复。

### 5. 通信量看什么

Embedding 前向的主要通信对象不是词表权重，而是恢复后的隐藏表示。所以前向通信量的主项与：

$$
B\times S\times H
$$

成正比，而不是与 $V$ 成正比。

可以粗略写成：

$$
\text{comm}_{embed}\propto B\times S\times H
$$

这点很重要，因为它解释了一个常见误区：词表再大，也不会让 Embedding 前向的 AllReduce 张量直接变成 $B\times S\times V$；真正恢复的是 token 的隐藏向量，而不是完整词表 logits。

### 6. 与共享 LM Head 的关系

如果输入 Embedding 与输出层 LM Head 共享权重，则同一份词表矩阵还要用于把隐藏状态投影回词表分数。此时第 $r$ 张卡只持有本地那一段：

$$
W_{lm}^{(r)}\in \mathbb{R}^{\frac{V}{t}\times H}
$$

给定隐藏状态：

$$
X\in \mathbb{R}^{B\times S\times H}
$$

本地 logits 为：

$$
L_r = X{W_{lm}^{(r)}}^T
\in \mathbb{R}^{B\times S\times \frac{V}{t}}
$$

这一步天然也是 vocab parallel 的。问题在于：如果你立刻把所有 $L_r$ 做 AllGather，恢复成完整：

$$
L\in \mathbb{R}^{B\times S\times V}
$$

那么通信压力会随着 $V$ 增长得非常快。于是工程上更常见的做法不是“先 gather 全量 logits，再算交叉熵”，而是直接使用 vocab-parallel cross entropy，在分片 logits 上完成 loss 计算，只交换 softmax 归一化所需的统计量。

因此：

| 部分 | 主要瓶颈 |
| --- | --- |
| Embedding 前向 | 参数存储 + $B\times S\times H$ 激活归约 |
| LM Head + Loss | $B\times S\times V$ 方向的输出通信压力 |

很多新手在这里会把两个问题混在一起。更准确的说法是：

- Vocab Parallel Embedding 先解决“词表权重太大，存不下”。
- Vocab Parallel Cross Entropy 再解决“输出词表太大，传不动”。

这两个模块通常需要配套理解。

---

## 代码实现

下面先给一个可运行的 `numpy` 实现，只模拟最核心的三件事：

1. 按词表切分权重。
2. 非本卡 token 用 zero mask 置 0。
3. 通过“求和”模拟 AllReduce 恢复完整输出。

这段代码可以直接运行，用来验证数值等价。

```python
import numpy as np


def split_vocab_ranges(vocab_size, tp_size):
    assert vocab_size % tp_size == 0, "toy example requires divisible vocab"
    shard = vocab_size // tp_size
    return [(rank * shard, (rank + 1) * shard) for rank in range(tp_size)]


def vocab_parallel_embedding(token_ids, full_weight, tp_size):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    vocab_size, hidden_size = full_weight.shape
    ranges = split_vocab_ranges(vocab_size, tp_size)

    partial_outputs = []
    for start, end in ranges:
        local_weight = full_weight[start:end]  # [V/TP, H]

        # mask: 当前 rank 是否负责该 token
        mask = (token_ids >= start) & (token_ids < end)

        # 非本地 token 的下标先写成 0，避免越界；真正是否生效由后面的 mask 控制
        local_ids = np.where(mask, token_ids - start, 0)

        # 查表
        local_out = local_weight[local_ids]  # [N, H]

        # 非本地 token 强制清零
        local_out = local_out * mask[:, None]
        partial_outputs.append(local_out)

    # 模拟 AllReduce(sum)
    return np.sum(partial_outputs, axis=0)


def dense_embedding(token_ids, full_weight):
    return full_weight[np.asarray(token_ids, dtype=np.int64)]


if __name__ == "__main__":
    # V=8, H=3
    weight = np.array(
        [
            [1, 0, 0],  # emb_0
            [0, 1, 0],  # emb_1
            [0, 0, 1],  # emb_2
            [1, 1, 0],  # emb_3
            [1, 0, 1],  # emb_4
            [0, 1, 1],  # emb_5
            [1, 1, 1],  # emb_6
            [2, 0, 0],  # emb_7
        ],
        dtype=np.float32,
    )

    token_ids = [0, 4, 3, 7, 1]

    y_tp = vocab_parallel_embedding(token_ids, weight, tp_size=2)
    y_ref = dense_embedding(token_ids, weight)

    print("TP output:")
    print(y_tp)
    print()
    print("Reference output:")
    print(y_ref)

    assert np.allclose(y_tp, y_ref)
    print("\ncheck passed")
```

如果你第一次接触这段逻辑，可以手动对照其中一个 token 看：

| 步骤 | token=4 时发生什么 |
| --- | --- |
| GPU0 检查区间 `[0,4)` | 不命中，mask=0，输出被清零 |
| GPU1 检查区间 `[4,8)` | 命中，返回 `emb_4` |
| AllReduce 求和 | `0 + emb_4 = emb_4` |

这就是整个机制最核心的一次完整路径。

下面再给一个更接近 PyTorch 工程实现的前向伪代码：

```python
# token_ids: [B, S]
# local_weight: [V/TP, H]
# vocab_start, vocab_end: 当前 rank 负责的词表区间

mask = (token_ids >= vocab_start) & (token_ids < vocab_end)   # [B, S]

# 非本地 token 先写成 0，避免越界
local_ids = token_ids - vocab_start
local_ids = torch.where(mask, local_ids, torch.zeros_like(local_ids))

# 本地查表
local_out = F.embedding(local_ids, local_weight)              # [B, S, H]

# 把非本地 token 的结果清零
local_out = local_out * mask.unsqueeze(-1)

# 对所有 TP rank 做求和归约，恢复完整 embedding 输出
dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
output = local_out
```

这里最容易写错的是两处：

| 容易写错的地方 | 为什么会错 |
| --- | --- |
| 只修正 `local_ids`，不乘回 `mask` | 会把本地第 0 行 embedding 泄漏到不属于本卡的 token 上 |
| `start/end` 与初始化切分规则不一致 | 某些 token 会永久命中错误 rank |

再看共享权重的 LM Head。本地只持有：

$$
W_{lm}^{(r)}\in\mathbb{R}^{\frac{V}{TP}\times H}
$$

本地 logits 计算通常写成：

```python
# hidden: [B, S, H]
# local_weight: [V/TP, H]

local_logits = torch.matmul(hidden, local_weight.t())   # [B, S, V/TP]
```

如果只是为了做训练 loss，更合理的做法不是：

```python
full_logits = all_gather(local_logits, dim=-1)
loss = cross_entropy(full_logits, labels)
```

而是直接使用 vocab-parallel cross entropy。它的核心思想是：

1. 每张卡只保留本地那一段 logits。
2. 通过少量归约拿到全局 `max` 和 `sum(exp(logits))`。
3. 只对目标 token 所在 rank 取出正确类分数。
4. 组合出与单卡 softmax-cross-entropy 完全一致的 loss。

写成公式，单个位置的交叉熵是：

$$
\ell = -z_y + \log\sum_{j=1}^{V} e^{z_j}
$$

其中：

- $z_y$ 是目标词的 logit。
- $\sum_{j=1}^{V} e^{z_j}$ 是完整词表上的 softmax 分母。

在 vocab parallel 下：

$$
\sum_{j=1}^{V} e^{z_j}
=
\sum_{r=0}^{t-1}\sum_{j\in \mathcal{V}_r} e^{z_j}
$$

每张卡只负责自己本地词表子集 $\mathcal{V}_r$ 的那一部分，再通过归约拿到全局结果。这就是为什么很多框架会把 `VocabParallelEmbedding` 和 `VocabParallelCrossEntropy` 作为一套配套组件来实现。

---

## 工程权衡与常见坑

最常见的坑不是“通信太慢”，而是“结果悄悄错了”。因为通信慢通常能通过 profiling 看到，而 zero mask 或边界映射写错时，程序仍然会正常跑，只是 loss 异常、收敛变慢，或者模型效果明显变差。

下面这个表是工程里最常见的一组问题：

| 问题 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 忽略 zero mask | 输出数值不对，但形状完全正确 | 非本卡 token 没有被清零 | 查表后强制乘 `mask.unsqueeze(-1)` |
| 越界 token 临时映射为 0 后忘记再清零 | 某些位置像是随机混入 `embedding[0]` | 非本地 token 被错误查成第 0 行 | 始终把“索引修正”和“结果清零”视为一对操作 |
| 词表切分边界不一致 | 少量 token 恒错，且常集中在边界附近 | 初始化、前向、checkpoint 加载用的切分函数不同 | 所有地方共用同一个 `vocab_range_from_global_vocab_size` 函数 |
| 词表不能整除 TP | 某些 rank 参数更多，逻辑变复杂 | 分片长度不一致 | 优先 pad 词表到整数倍 |
| 共享权重布局不一致 | 输入 Embedding 正常，LM Head 输出异常 | Embedding 和 LM Head 的切分顺序不同 | 两者严格共享同一份本地分片权重 |
| 训练时直接 AllGather 完整 logits | 带宽和显存突然升高 | 生成了 $B\times S\times V$ 大张量 | 用 vocab-parallel cross entropy |
| 推理时频繁 gather 全量 logits | 大词表下延迟明显上升 | 每步都恢复完整词表分数 | 只在必要时 gather，或只取 top-k 所需片段 |

新手最容易忽略的是“输出激活没有缩小”这一点。切分 Embedding 后，单卡参数显存确实下降了，但前向输出仍然是标准：

$$
[B,S,H]
$$

所以如果你的训练瓶颈来自超长序列激活，而不是词表权重，那么只做 Vocab Parallel 不会直接解决全部显存问题。

再看通信侧的权衡。Embedding 前向的 AllReduce 张量大小主要与 $B\times S\times H$ 有关，通常还算可控；真正容易膨胀的是输出侧 logits。举个数量级例子：

$$
B=8,\quad S=4096,\quad V=128K
$$

完整 logits 元素数是：

$$
8\times 4096\times 128000 \approx 4.19\times 10^9
$$

即使采用 BF16，每个元素 2 字节，中间张量也接近：

$$
8.39\text{ GB}
$$

这还只是“一个时间点上 materialize 出来的完整 logits”，不包含反向中的额外开销。也正因为如此，很多系统里真正要优先优化的往往不是 Embedding 查表本身，而是“LM Head 后面怎么计算 loss”。

还有一个实践建议：验证这类模块时，不要只看“代码能跑”，而要做最小数值对齐测试。最少应覆盖：

| 测试项 | 目的 |
| --- | --- |
| 单个 token 命中不同 rank | 验证区间判断正确 |
| 边界 token，如 `start`、`end-1` | 验证切分端点没有 off-by-one |
| 多个重复 token | 验证重复查表不会被污染 |
| 与 dense Embedding 输出逐元素对齐 | 验证前向完全等价 |
| 共享 LM Head 与 dense logits/loss 对齐 | 验证输出侧实现无误 |

这类测试很便宜，但能提前挡掉大量隐蔽 bug。

---

## 替代方案与适用边界

Vocab Parallel 不是唯一方案，但它是大词表语言模型中最自然、最直接的方案。原因很简单：Embedding 本身就是“按词表行索引”的矩阵，沿词表维切分最符合访问模式。

可以把常见方案放在一起比较：

| 方案 | 做法 | 适用条件 | 局限 |
| --- | --- | --- | --- |
| Vocab Parallel Embedding | 沿词表维切分 Embedding/LM Head | 大词表、TP 训练、需权重共享 | 需要额外通信，并要处理分布式 loss |
| Embedding Replication | 每张卡完整复制一份 Embedding | 词表较小、显存充足 | 参数冗余，扩展性差 |
| Hybrid Parallel | TP + PP + DP 组合 | 大模型整体训练 | 系统复杂度高，调优成本高 |
| Adaptive / Hierarchical Vocab | 词表分层或按频次做特殊处理 | 特定任务或特定词表分布 | 通用 LLM 中不一定划算 |
| Hash / Compressed Embedding | 用哈希或压缩表示减少参数量 | 推荐系统、特定工业场景 | 不适合通用 LLM 的标准词表建模 |
| Expert Parallel | 把 FFN 等部分做专家并行 | MoE 模型 | 不能直接替代 Embedding 切分 |

Parameter Parallel 可以理解为“广义地拆参数”；Expert Parallel 可以理解为“让不同 token 只走部分专家”。这些方法对 Transformer 其他部分很重要，但对 Embedding 层本身，最顺手、最稳定的仍是按 vocab 维切。

什么时候不值得做？

| 场景 | 更合理的选择 |
| --- | --- |
| 词表只有 32K 左右，隐藏维也不大 | 直接复制完整 Embedding，更简单 |
| 模型显存瓶颈明显在 attention 激活 | 先看 sequence/context parallel 或 checkpointing |
| 推理部署更关心极简实现 | 先用复制方案，确认吞吐和显存是否真的不够 |

什么时候基本必须认真考虑？

1. 词表很大，例如 100K 到 200K。
2. Embedding 与 LM Head 共享权重。
3. 训练系统本来就已经启用 TP。
4. 输出层通信和显存已经成为真实瓶颈。

满足这些条件时，Vocab Parallel 通常不是“锦上添花”，而是大模型训练系统的标准配置。

---

## 参考资料

| 资料 | 作用 | 关键点 |
| --- | --- | --- |
| [Megatron Core 文档：`VocabParallelEmbedding`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.layers.html) | 官方接口说明 | 明确说明 Embedding 按 vocabulary dimension 并行，并支持 `reduce_scatter_embeddings` |
| [Megatron Core 文档：`tensor_parallel` 包](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/tensor_parallel.html) | 官方并行组件说明 | 包含 `VocabParallelCrossEntropy`，说明输出侧通常要与分片 loss 配套处理 |
| [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM) | 代码实现入口 | 可对照实际训练系统里 Embedding、LM Head、并行 loss 的组织方式 |
| [Megatron-LM 论文（SC 2021）](https://www.microsoft.com/en-us/research/publication/efficient-large-scale-language-model-training-on-gpu-clusters/) | 背景与系统设计 | 给出张量并行与大词表训练的整体工程背景 |

推荐的学习顺序是：

1. 先看本文里的最小例子，确认“为什么 zero mask + AllReduce 可以恢复正确 Embedding”。
2. 再看 Megatron Core 的 `VocabParallelEmbedding` 接口，理解工程层面的模块边界。
3. 最后看 `VocabParallelCrossEntropy`，把 Embedding 侧和 LM Head/loss 侧放到同一个训练系统里理解。

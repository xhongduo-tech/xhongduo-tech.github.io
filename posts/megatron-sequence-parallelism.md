## 核心结论

Megatron-LM 的序列并行（Sequence Parallelism，白话解释：把一整段序列拆给多张卡分别保存和处理）解决的是张量并行里一个很具体的问题：**LayerNorm 和 Dropout 这类非张量并行层的激活在每张 TP 卡上都完整复制了一份，内存浪费严重**。

它的做法不是改写注意力或 MLP 的数学定义，而是只在合适的位置改变张量的分布方式：

1. 在 LayerNorm、Dropout 这类逐 token 独立的层上，按序列维度切分。
2. 在进入 Attention、MLP 这类需要完整序列或张量并行配合的层之前，用 AllGather 把序列拼回去。
3. 在这些层结束后，再用 Reduce-Scatter 把输出切回每张卡自己的序列分片。

因此，非 TP 层的激活从“每张卡都保存完整一份”变成“每张卡只保存 $1/t$”，其中 $t$ 是张量并行度。论文给出的激活存储量可写为：

$$
\text{Activation Memory} = \frac{sbh(34 + 5a/h)}{t}
$$

其中 $s$ 是序列长度，$b$ 是微批大小，$h$ 是隐藏维度，$a$ 是注意力头数，$t$ 是张量并行度。这个式子最重要的信息只有一个：**激活内存随 $t$ 线性分摊**。

玩具例子可以直接算。设 $s=1024,\ b=1,\ h=4096,\ a=16,\ t=4$，传统情况下激活近似为：

$$
1024 \times 1 \times 4096 \times \left(34 + \frac{5 \times 16}{4096}\right) \approx 1.42 \times 10^8
$$

启用序列并行后，每张卡约只需：

$$
\frac{1.42 \times 10^8}{4} \approx 3.55 \times 10^7
$$

也就是原来的四分之一。

真实工程例子更能说明价值。MLSys 2023 论文报告，在 530B GPT-3 风格模型、2240 张 A100 的训练中，序列并行配合选择性重算把激活内存降低了约 5 倍，并把 MFU 提升到 54.2%，相对全量重算快约 29%。这说明它不是“省内存但拖垮速度”的技巧，而是在大模型场景里能真正换来吞吐的工程优化。

---

## 问题定义与边界

先定义问题。张量并行（Tensor Parallelism，白话解释：把一个大矩阵拆到多张卡上共同算）主要切的是线性层参数和对应计算，例如 Attention 里的投影矩阵、MLP 里的前馈矩阵。但 Transformer 里并不是所有层都天然适合这样切。

LayerNorm 的特点是：它对每个 token 的隐藏维度做归一化，不依赖别的 token。Dropout 的特点是：它对每个元素独立随机置零，也不依赖别的 token。这两类层本身不需要完整的跨卡矩阵乘法，因此在传统 TP 实现里，往往直接让每张卡都保留完整的 $[B,S,H]$ 激活副本。这样做计算简单，但代价是：

- 激活重复存了 $t$ 份。
- Dropout 掩码也重复存了 $t$ 份。
- TP 度越大，这部分冗余越明显。

问题边界也要说清。序列并行**不是**把所有层都按序列切开。它只适合那些沿序列维度天然可分、且不同 token 间没有依赖的计算。Attention 和 MLP 这类层仍然要在进入计算前恢复成它们需要的分布状态。

下面这个表可以直接看出传统 TP 与 SP+TP 的差别：

| 项目 | 传统 TP | SP + TP |
|---|---|---|
| LayerNorm/Dropout 激活 | 每张卡保存完整 $sbh$ | 每张卡保存 $sbh/t$ |
| Dropout 掩码 | 每张卡一整份 | 每张卡只保留本地分片 |
| 进入 Attention/MLP 前 | 已是完整序列副本 | 需要 AllGather 恢复 |
| Attention/MLP 后 | 直接继续 | 需要 Reduce-Scatter 切回 |
| 主要收益 | 降低参数/算力分摊压力 | 降低非 TP 层激活冗余 |
| 适用条件 | 任意 TP | 需要 `tensor_model_parallel_size > 1` |

一个最小直观例子：TP=4，序列长度 $S=8$。传统做法里，4 张卡都保存 8 个 token 的 LayerNorm 输入和输出；序列并行里，4 张卡各自只保留 2 个 token。因为 LayerNorm 和 Dropout 对 token 之间没有依赖，所以每张卡只算自己的 2 个 token 就够了。

它的边界条件也很明确：

- 必须和张量并行一起使用，单卡或 TP=1 时没有意义。
- 必须复用 TP 通信组，否则新增通信会抵消收益。
- 主要优化的是**激活内存**，不是参数内存。
- 更适合长序列、大隐藏维度、高 TP 度场景。

---

## 核心机制与推导

可以把一个 Transformer block 的数据流写成文字版流程图：

`LayerNorm/Dropout -> AllGather -> Attention/MLP -> Reduce-Scatter -> 下一个逐序列可分层`

这里的关键不是“加了两个通信”，而是“**把原先无脑复制的激活，改成按需恢复和按需切分**”。

### 1. 为什么 LayerNorm 可以按序列切

设输入张量形状为 $[B,S,H]$。LayerNorm 对单个 token 的公式是：

$$
\text{LN}(x_i)=\gamma \cdot \frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}+\beta
$$

其中 $x_i \in \mathbb{R}^H$ 表示第 $i$ 个 token 的隐藏向量。注意 $\mu_i,\sigma_i^2$ 都只由这个 token 自己的 $H$ 维决定，不依赖别的 token。所以沿序列维度切成 $[B,S/t,H]$ 后，每张卡单独算本地 token，结果完全正确。

Dropout 也类似。它本质是对每个元素独立采样一个 0/1 掩码：

$$
y = \frac{m \odot x}{1-p}
$$

这里 $m$ 是伯努利随机变量矩阵。这个操作同样不要求看到整段序列，因此也可以在本地分片上独立完成。

### 2. 为什么 Attention/MLP 前要 AllGather

注意力（Attention，白话解释：每个 token 会看其他 token）不同。即使在张量并行里，注意力通常也需要完整序列长度上的上下文关系。如果当前每张卡只有 $S/t$ 个 token，就不能直接做完整的自注意力。因此进入这些层前，需要把分片重新拼成全序列表示。

于是，前向流程变成：

1. 输入先以序列分片形式存在，每张卡是 $[B,S/t,H]$。
2. 本地执行 LayerNorm、Dropout。
3. 执行 AllGather，得到逻辑上的完整序列 $[B,S,H]$。
4. 进入 Attention 或 MLP 的张量并行计算。
5. 输出后执行 Reduce-Scatter，再回到 $[B,S/t,H]$。

Reduce-Scatter（白话解释：先做规约，再把结果分发成分片）之所以关键，是因为它能把原来一次 AllReduce 的“求和+广播”拆成两步里更有用的一步。对序列并行来说，最终目标不是每张卡都拿到完整结果，而是**每张卡只拿回自己负责的那段序列**。

### 3. 为什么内存是 $1/t$

推导的核心很简单。假设原本非 TP 层需要保存的激活规模是：

$$
M_{\text{base}} = sbh\left(34+\frac{5a}{h}\right)
$$

在传统 TP 里，这部分激活往往仍然在每张卡保留完整一份，所以单卡看到的仍接近 $M_{\text{base}}$。而序列并行把这部分张量按序列维切成 $t$ 份，每张卡只保留：

$$
M_{\text{SP}} = \frac{M_{\text{base}}}{t}
= \frac{sbh(34+5a/h)}{t}
$$

这里要注意，**不是整个 Transformer 所有内存都严格变成 $1/t$**。参数、优化器状态、KV cache、通信缓存等并不都受这个式子控制。这个公式描述的是论文关注的那部分激活存储，尤其是 LayerNorm/Dropout 等非 TP 层的冗余。

### 4. 一个完整玩具例子

假设：

- batch $B=1$
- sequence $S=8$
- hidden $H=4$
- TP 度 $t=2$

输入序列有 8 个 token。序列并行后：

- GPU0 负责 token 0 到 3
- GPU1 负责 token 4 到 7

此时：

- LayerNorm：两张卡各自对自己 4 个 token 做归一化。
- Dropout：两张卡各自生成本地掩码。
- 进入 Attention 前：AllGather，逻辑上恢复 8 个 token。
- Attention 算完后：Reduce-Scatter，重新拆回 4+4。

所以，**逐 token 的层本地算，跨 token 的层先拼后算**。这就是序列并行最核心的机制。

---

## 代码实现

在 Megatron-LM 里，序列并行通常通过两个条件开启：

- `--tensor-model-parallel-size > 1`
- `--sequence-parallel`

这说明它不是独立并行策略，而是建立在 TP 之上的补充优化。

下面先给一个可运行的 Python 玩具代码，模拟“传统 TP 的激活副本”和“序列并行后的分片激活”之间的数量关系：

```python
def activation_elements(s, b, h, a, t=1):
    return s * b * h * (34 + 5 * a / h)

def sp_activation_elements_per_rank(s, b, h, a, t):
    return activation_elements(s, b, h, a, t=1) / t

# 玩具参数
s, b, h, a, t = 1024, 1, 4096, 16, 4

base = activation_elements(s, b, h, a)
sp_per_rank = sp_activation_elements_per_rank(s, b, h, a, t)

# 序列并行后每张卡应为原来的 1/t
assert abs(sp_per_rank * t - base) < 1e-6
assert int(base) == 142688256
assert int(sp_per_rank) == 35672064

print("base:", int(base))
print("sp_per_rank:", int(sp_per_rank))
```

这个例子只验证一个事实：如果被切的是序列维度，那么单卡持有量就是原来的 $1/t$。

再看更接近 Megatron-LM 的半伪代码。下面不是源码逐行拷贝，而是结构化表达它在做什么：

```python
def transformer_block(x_local, tp_group, sequence_parallel=True):
    # x_local: [B, S/t, H]
    x_local = layernorm(x_local)
    x_local = dropout(x_local)

    if sequence_parallel:
        # 恢复完整序列，供 Attention/MLP 使用
        x_full = all_gather_along_sequence(x_local, group=tp_group)   # [B, S, H]
    else:
        x_full = x_local

    # 进入需要完整序列或 TP 配合的层
    y_full = attention_and_mlp_with_tp(x_full, group=tp_group)

    if sequence_parallel:
        # 输出切回本地序列分片
        y_local = reduce_scatter_along_sequence(y_full, group=tp_group)  # [B, S/t, H]
    else:
        y_local = y_full

    return y_local
```

这段代码的重点只有三件事：

1. **非 TP 层在本地分片上执行。**
2. **TP 层前后插入 Gather/Scatter。**
3. **通信组复用已有 TP group，不需要额外建一套全新进程组。**

真实工程例子里，Megatron-Core 的一些并行线性层会根据 `sequence_parallel=True` 改变前后向的通信路径。前向可能在进入算子前做 AllGather，反向则利用 Reduce-Scatter 回收梯度或激活分布。这样做的目的不是让代码“更抽象”，而是把张量在不同阶段放在最省内存、又不破坏数学正确性的分布状态上。

如果你是从零开始理解，可以记住一个工程口诀：

| 阶段 | 张量形状倾向 | 为什么 |
|---|---|---|
| LayerNorm/Dropout | `[B,S/t,H]` | 每个 token 独立，可本地算 |
| Attention/MLP 前 | `[B,S,H]` | 需要完整序列或完整逻辑输入 |
| Attention/MLP 后 | `[B,S/t,H]` | 为下一段逐序列可分计算省内存 |

---

## 工程权衡与常见坑

序列并行省的是内存，但引入的是额外通信。因此工程上真正的判断标准不是“能不能开”，而是“**开了以后通信能不能被隐藏**”。

下面先看常见坑：

| 坑 | 现象 | 规避方式 |
|---|---|---|
| 重复通信 | 每层额外插入独立 AllGather/Reduce-Scatter，时延明显上升 | 复用 TP 通信组，尽量与原通信路径融合 |
| 顺序错配 | 在错误位置 Gather 或 Scatter，导致形状对不上或数值错误 | 严格限定在 LayerNorm/Dropout 与 TP 层边界切换 |
| 梯度同步理解错误 | 以为 LayerNorm 梯度还需要传统 AllReduce | 按 Megatron 的 SP 路径处理，避免重复规约 |
| 小模型硬开 SP | 通信比计算还重，吞吐下降 | 只在长序列、大隐藏维、高 TP 时启用 |
| 调试困难 | 前向张量形状在不同层来回变化，不易定位 bug | 打印每层逻辑 shape 与分布状态，而不是只看本地 shape |

最常见的误区是：看到 AllGather 和 Reduce-Scatter 就以为一定更慢。这个判断不完整。真正的问题是，**原系统的瓶颈在哪里**。

如果瓶颈是激活内存，结果通常是：

- 你不得不开更激进的重算。
- 微批大小被压得很小。
- 训练无法在目标并行度下稳定运行。

这时，序列并行虽然增加了 collective 通信，但它释放了显存，允许你减少重算或提高有效批量，整体反而更快。MLSys 2023 里的 530B 实验就是这个逻辑：不是通信免费了，而是激活内存从主瓶颈退下去了。

一个真实工程判断可以这样做：

- 如果你的模型已经因为激活爆显存而严重依赖 full recompute，优先评估 SP。
- 如果你的集群是高带宽互联，例如 NVLink 或高质量 InfiniBand，SP 更容易把通信隐藏在流水中。
- 如果你是 PCIe 小集群、TP 度也不高、序列长度还短，SP 可能收益有限。

因此，序列并行不是“默认全开”的万能开关，而是一个在**大模型、高 TP、高激活压力**下效果很强的专用优化。

---

## 替代方案与适用边界

如果不使用序列并行，最常见的替代方案是激活重算（Activation Recomputation，白话解释：前向不存中间结果，反向时再算一遍）。它的优点是实现简单、适用范围广，缺点是增加额外计算。

可以把常见方案放到一张表里比较：

| 方案 | 典型场景 | 内存节省 | 计算开销 | 通信开销 |
|---|---|---|---|---|
| Sequence Parallelism | 长序列、大模型、高 TP、高带宽互联 | 高，非 TP 激活约降到 $1/t$ | 低到中 | 中 |
| Selective Recomputation | 某些层激活大，但不想全量重算 | 中 | 中 | 低 |
| Full Recomputation | 显存极度紧张，先保证能跑 | 高 | 高，常见可到 30% 以上 | 低 |

适用边界可以概括成三条。

第一，**短序列或小模型未必值得开**。  
如果 $S$ 小、$H$ 小，LayerNorm/Dropout 这部分激活本来就不大，序列并行的收益不显著。

第二，**低带宽互联环境要谨慎**。  
在没有 NVLink、网络也一般的小集群上，AllGather/Reduce-Scatter 可能直接落在关键路径上。此时“TP + 选择性重算”往往比“TP + SP”更保守。

第三，**SP 不是上下文并行的替代品**。  
上下文并行（Context Parallelism，白话解释：把超长上下文切到多卡上联合处理）解决的是更长序列下的全局上下文容量问题；序列并行解决的是 TP 下非 TP 层的激活冗余问题。两者都和“序列有关”，但不是同一层面的优化。

可以用一个简单判断准则：

- 你的问题是“LayerNorm/Dropout 激活太冗余，TP 越大越浪费”时，用 SP。
- 你的问题是“反向显存不够，但通信预算紧张”时，优先 selective recompute。
- 你的问题是“无论如何都跑不下，只能先省显存”时，考虑 full recompute。
- 你的问题是“单层上下文太长，注意力本身放不下”时，看 context parallel 或更强的序列切分方案。

---

## 参考资料

- NVIDIA, *Reducing Activation Recomputation in Large Transformer Models*, MLSys 2023  
  https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf

- MLSys 2023 Abstract 页面  
  https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html

- DeepWiki, *Context and Sequence Parallelism*  
  https://deepwiki.com/NVIDIA/Megatron-LM/4.5-context-and-sequence-parallelism

- CSDN 技术复盘，*LayerNorm 与 Dropout 阶段的序列并行*  
  https://ascendai.csdn.net/68a18e87080e555a88da30c4.html

## 核心结论

长序列训练里的“序列并行”，本质上是把 `S` 个 token 按序列维切到 `N` 张 GPU 上，让每张卡只保存本地 token 块，再通过跨卡传递 `K,V` 完成全局 attention。

如果 32K 上下文放不进一张卡，就把 token 分到 8 张卡。每张卡只算自己的 `Q/K/V`，但 attention 仍要看到全局 `K/V`，所以必须通信。

总览图：

```text
S tokens
   |
   v
split by sequence dimension
   |
   v
GPU0: X_0 -> local Q_0/K_0/V_0
GPU1: X_1 -> local Q_1/K_1/V_1
...
GPU(N-1): X_(N-1) -> local Q/K/V
   |
   v
KV communication across GPUs
   |
   v
global attention output for each local token block
```

定义公式：

$$
X = [X_0; X_1; \dots; X_{N-1}], \quad X_r \in \mathbb{R}^{S/N \times d_{model}}
$$

其中 `X_r` 是第 `r` 张卡上的 token 块。`S` 是序列长度，`N` 是 GPU 数量，`d` 通常表示 attention head 的维度。

它解决的是“单卡装不下长上下文”的显存问题，不是单纯加速技巧；真正目标是把长序列训练从“单卡瓶颈”改成“多卡协同”。

| 问题 | 序列并行解决吗 | 说明 |
|---|---:|---|
| 单卡放不下 32K、128K 上下文激活 | 是 | token 被切到多卡，本地显存压力下降 |
| attention 需要看到全局上下文 | 是，但需要通信 | 本地 `Q` 要和全局 `K,V` 计算 |
| 所有训练都会更快 | 不一定 | 通信可能抵消收益 |
| 模型参数太大放不下 | 不直接解决 | 这是 tensor parallelism 或 ZeRO/FSDP 的职责 |
| 短序列训练吞吐优化 | 不一定适合 | 短序列上通信收益可能不够 |

---

## 问题定义与边界

长序列训练的问题不是“模型算不算得动”，而是“attention 的全局依赖会把显存和通信同时拉爆”。自注意力机制，白话说就是每个 token 都要根据其他 token 的信息重新计算自己的表示。序列越长，需要保存和计算的中间量越多。

标准 attention 的主要计算是：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt d}+M\right)V
$$

其中 `M` 是 mask。对因果语言模型来说，`M` 用来禁止当前位置看未来 token。

新手版解释：把一篇长文分给多个人分别看，每个人先处理自己那一段，但回答问题时必须把所有人的关键信息汇总起来。序列并行做的就是这个“分段看、统一答”。

需要区分 `TP`、`SP`、`CP` 三件事：

| 名称 | 全称 | 切分对象 | 是否处理全局 attention | 主要目标 |
|---|---|---|---:|---|
| `TP` | Tensor Parallelism | 模型权重或矩阵乘法维度 | 间接处理 | 让大模型参数和计算分布到多卡 |
| `SP` | Sequence Parallelism | 部分激活，如 LayerNorm、Dropout 的序列维 | 否 | 降低非 attention 层的激活显存 |
| `CP` | Context Parallelism | 上下文序列维度 | 是 | 支持长上下文 attention |
| Ring Attention | 环形 attention 通信范式 | 序列块与 `K,V` 块 | 是 | 通过环形轮转完成分布式 attention |
| DistFlashAttention | 分布式 FlashAttention | attention 分块与通信调度 | 是 | 提升长序列训练吞吐和负载均衡 |

术语统一如下：

| 符号 | 含义 |
|---|---|
| `S` | 总序列长度 |
| `N` | GPU 数量 |
| `d` | attention head 维度 |
| `X_r` | 第 `r` 张卡上的 token 块 |
| `Q_r` | 第 `r` 张卡根据 `X_r` 算出的 query |
| `K_r` | 第 `r` 张卡根据 `X_r` 算出的 key |
| `V_r` | 第 `r` 张卡根据 `X_r` 算出的 value |

适用边界很重要。短序列，比如 2K、4K tokens，不一定值得上 `CP`，因为通信开销可能超过显存收益。超长序列，比如 32K、128K、512K tokens，`CP`、Ring Attention、DistFlashAttention 更合适。只切激活的 `SP` 不能替代全局 attention 通信，因为它没有让本地 `Q` 看见所有 `K,V`。

---

## 核心机制与推导

序列并行 attention 的核心是两段式流程：先本地算投影，再跨卡算 attention。

本地投影：

$$
Q_r = X_r W_Q,\quad K_r = X_r W_K,\quad V_r = X_r W_V
$$

`W_Q,W_K,W_V` 是线性投影权重。白话说，它们把 token 表示转换成三类向量：`Q` 表示“我要找什么信息”，`K` 表示“我有什么索引”，`V` 表示“我真正提供什么内容”。

全局 attention：

$$
O_r = softmax\left(\frac{Q_r K^T}{\sqrt d}+M\right)V
$$

其中：

$$
K=[K_0;K_1;\dots;K_{N-1}],\quad V=[V_0;V_1;\dots;V_{N-1}]
$$

注意 `Q_r` 是本地的，但 `K,V` 必须是全局的。这就是为什么只切序列还不够，必须设计通信。

流程图：

```text
local X_r
   |
   v
local projection: Q_r, K_r, V_r
   |
   v
communicate K,V blocks
   |
   v
compute scores block by block
   |
   v
online softmax accumulation
   |
   v
local output O_r
```

Ring Attention 的关键思想是：不要一次性收集所有 `K,V`，而是让每张卡把自己的 `K,V` 沿环形拓扑传给下一张卡。每张卡边收到别人的 `K,V`，边用自己的 `Q_r` 做一块 attention，并用在线 softmax 维护稳定的累积结果。

在线 softmax，白话说就是不把所有分数一次性放进内存，而是分块更新最大值、归一化分母和输出。核心更新思路是：

$$
m = max(m, s)
$$

$$
l = \sum exp(s-m)
$$

$$
o = \frac{\sum exp(s-m)V}{l}
$$

真实实现会使用更完整的重缩放公式，以保证前一块和后一块的统计量在同一个最大值基准下合并。

玩具例子：两卡、`d=1`、单个 query `q=1`。

GPU0 有：

$$
(k,v)=(1,1)
$$

GPU1 有：

$$
(k,v)=(3,5)
$$

全局得分：

$$
s=[qk_0,qk_1]=[1,3]
$$

softmax 权重约为：

$$
[0.1192,0.8808]
$$

输出：

$$
o=0.1192 \times 1 + 0.8808 \times 5 \approx 4.5232
$$

Ring 视角下，GPU0 先看本地块，得到局部统计量；再收到 GPU1 的 `K,V`，更新统计量。最终输出与一次性全量 attention 完全一致。

可运行 Python 代码如下：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def full_attention(q, keys, values):
    scores = [q * k for k in keys]
    weights = softmax(scores)
    return sum(w * v for w, v in zip(weights, values))

q = 1.0
keys = [1.0, 3.0]
values = [1.0, 5.0]

out = full_attention(q, keys, values)
assert abs(out - 4.5231883119) < 1e-6

# 分块视角：每次只处理一个 KV block，但结果必须等价于全量 attention。
blocks = [(1.0, 1.0), (3.0, 5.0)]
scores = []
vals = []
for k, v in blocks:
    scores.append(q * k)
    vals.append(v)

ring_like_out = sum(w * v for w, v in zip(softmax(scores), vals))
assert abs(ring_like_out - out) < 1e-12
print(round(ring_like_out, 4))
```

Ring Attention 和 Megatron CP 的区别主要在通信组织方式：前者是环形轮转 `K,V`，后者更偏向 `KV all-gather / reduce-scatter`。`all-gather` 是把各卡的数据收集到每张卡；`reduce-scatter` 是先聚合结果再按切分维度分发回各卡。两者目标一致：让每张卡最终得到正确的全局 attention 输出。

---

## 代码实现

实现层面要把 attention 层和非 attention 层分开处理。attention 里走跨卡通信，MLP、Norm、残差连接等尽量保持本地执行。这样做的原因很直接：不是所有层都有全局 token 依赖，没必要让每个操作都跨卡通信。

伪代码：

```python
# 伪代码：序列并行 attention
X_r = local_token_block()

Q_r = X_r @ W_Q
K_r = X_r @ W_K
V_r = X_r @ W_V

state = init_online_softmax_state()

for peer_block in ring_or_allgather(K_r, V_r):
    scores = Q_r @ peer_block.K.T / sqrt(d) + mask
    state = update_online_softmax(state, scores, peer_block.V)

O_r = finalize_online_softmax(state)

# backward:
# attention 部分做 reduce-scatter / gradient merge
# 其他层保持 local backward
```

通信原语需要准确理解：

| 原语 | 白话解释 | 在序列并行里的作用 |
|---|---|---|
| `all-gather` | 每张卡把自己的数据发出来，并拿到所有卡的数据 | 收集全局 `K,V` 或相关块 |
| `reduce-scatter` | 先把多卡结果聚合，再按切分规则分回各卡 | 反向传播时合并梯度并切回本地 |
| ring rotation | 每张卡只和相邻卡交换数据，数据绕一圈 | 避免一次性收集所有 `K,V` |
| overlap | 通信和计算重叠 | 收下一块 `K,V` 的同时计算当前块 |

forward 与 backward 对照：

| 阶段 | attention 部分 | 非 attention 部分 | 易错点 |
|---|---|---|---|
| forward | 本地 `Q` 对全局 `K,V` 做分块 attention | 本地执行 MLP、Norm、残差 | mask、通信顺序、在线 softmax |
| backward | 需要合并 `Q,K,V` 相关梯度 | 本地反向传播 | 梯度归并维度、reduce-scatter 位置 |
| checkpoint | 可重算 attention 中间块 | 可按普通激活 checkpoint 处理 | 重算量可能随序列长度放大 |

哪些层本地算，哪些层跨卡算：

| 模块 | 是否跨卡 | 原因 |
|---|---:|---|
| `Q/K/V` 线性投影 | 通常本地算 | 输入 `X_r` 已经在本卡 |
| attention score `QK^T` | 需要跨卡 | 本地 `Q_r` 要看全局 `K` |
| attention output `softmax(...)V` | 需要跨卡 | 输出依赖全局 `V` |
| LayerNorm | 通常本地算 | 对 token 局部特征归一化 |
| MLP | 通常本地算，或配合 TP | 不需要全局 token 交互 |
| residual add | 本地算 | 只依赖同一 token 位置 |

真实工程例子：做 32K 上下文预训练时，常见做法是 `TP + CP`。`TP` 负责把大矩阵乘法切到多卡，`CP` 负责把长上下文切到多卡。这样既降低参数和计算压力，也降低长序列 attention 的单卡显存压力。如果继续扩到 128K 或 512K，上层还要考虑 DistFlashAttention 这类更强的负载均衡和通信重叠实现。

真正容易出错的是通信顺序、数值稳定性和反向传播的张量归并方式，而不是 forward 的单次矩阵乘法。

---

## 工程权衡与常见坑

序列并行的性能不只看计算量，还看通信开销、负载均衡和重算成本。块切得不合适会让通信吞掉收益。

新手可理解版：如果每次只传很小一块 `K,V`，卡虽然省显存了，但通信次数会暴涨，可能比直接算还慢。要让每次通信足够“值”，同时把计算和通信重叠起来。

通信-计算重叠示意图：

```text
time ->
GPU compute:   [compute block 0] [compute block 1] [compute block 2]
GPU comm:          [send/recv 1]     [send/recv 2]     [send/recv 3]
                      ^ overlap          ^ overlap          ^ overlap
```

常见坑：

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 把 `SP` 当成 `CP` | 长上下文 attention 仍然装不下 | 明确 `SP` 只切部分激活，长序列要用 `CP` 或 ring 类方案 |
| naive ring 的负三角负载不均 | causal LM 中部分卡计算少、部分卡计算多 | 使用更好的分块、条纹化分配或 DistFlashAttention 的负载均衡 |
| 块太小 | 通信次数过多，吞吐下降 | 增大 block size，并做通信计算重叠 |
| checkpoint 方案不当 | 反向重算成本爆炸 | 做 rematerialization-aware 设计 |
| mask 处理错误 | 模型看到未来 token，训练目标被破坏 | 对每个远端块正确生成 causal mask |
| 只测 forward 不测 backward | 实际训练时梯度通信崩溃 | 用端到端训练 step 验证显存、吞吐和 loss |

`rematerialization` 是重算中间激活的技术，白话说就是 forward 时少存一点，backward 时再算回来。长序列训练要做 rematerialization-aware 设计，因为 attention 的重算成本会随着上下文长度快速变大。

负载不均在 causal attention 里尤其明显。因果 mask 使第 `i` 个 token 只能看前面的 token，不能看后面的 token。如果按普通方块切分，右上角的未来区域全部被 mask 掉，计算量呈现“三角形”分布。naive ring 可能让某些 GPU 拿到很多有效块，另一些 GPU 拿到很多无效块，最终整体速度由最慢的卡决定。

工程上要同时看三个指标：

| 指标 | 关注点 |
|---|---|
| 显存 | 单卡是否能放下当前 batch、序列和模型 |
| 吞吐 | tokens/s 是否真的提升 |
| 扩展效率 | GPU 数增加后收益是否接近线性 |

序列并行不是打开一个开关就结束。它要求 attention kernel、通信拓扑、mask、checkpoint、梯度归并一起设计。

---

## 替代方案与适用边界

不同方案没有绝对优劣，关键看上下文长度、GPU 数量、网络带宽和目标吞吐。

场景化选择：如果你在做 32K 上下文预训练，且已经在 Megatron 体系里，优先看 `TP + CP`。如果你追求更长上下文和更强吞吐，可以看 DistFlashAttention。如果你想理解算法本质，先学 Ring Attention。

| 方案 | 核心思路 | 优点 | 局限 | 适用场景 |
|---|---|---|---|---|
| Ring Attention | `K,V` 沿环形拓扑轮转，本地 `Q` 分块累积 attention | 思路清晰，适合理解分布式 attention | 朴素实现可能有负载不均和通信调度问题 | 学习算法机制，构建自定义长上下文训练 |
| Megatron CP | 按 context 维切序列，在 attention 中做跨卡 `K,V` 通信 | 和 Megatron 训练栈集成好 | 依赖 Megatron 体系和并行配置 | 已使用 Megatron 的 8K+ 长序列训练 |
| DistFlashAttention | 分布式 FlashAttention，加负载均衡和通信优化 | 更关注吞吐、显存和工程效率 | 实现复杂，依赖特定 kernel 与调度 | 32K 到更长上下文的大规模训练 |
| 只用 FlashAttention | 单卡或单进程内做 IO-aware attention | 显著降低 attention 显存 | 不解决跨卡全局上下文切分 | 单卡能放下序列，但需要省显存和提速 |

为什么不是只靠 FlashAttention？FlashAttention 是 IO-aware attention，白话说就是减少 GPU 高带宽内存读写，把 attention 做得更省显存、更快。它很重要，但它主要优化单设备或局部 attention kernel。当序列长到单卡连局部激活和 `K,V` 都放不下时，就需要 `CP`、Ring Attention 或 DistFlashAttention 这类跨设备方案。

一个简单判断规则：

| 条件 | 推荐方向 |
|---|---|
| 4K 以下，上下文不构成瓶颈 | 普通 FlashAttention 可能足够 |
| 8K 到 32K，已有 Megatron 训练栈 | 优先评估 Megatron CP |
| 32K 以上，并且吞吐要求高 | 评估 DistFlashAttention 或更强分布式 attention |
| 想从原理理解长序列并行 | 先学习 Ring Attention |
| 模型参数本身放不下 | 先解决 TP、PP、FSDP 或 ZeRO，再叠加 CP |

最终选择不是“哪个名字更先进”，而是看瓶颈在哪里。如果瓶颈是参数显存，优先模型并行或参数分片。如果瓶颈是长上下文 attention 激活，优先序列维切分。如果瓶颈是通信，优先优化 block size、拓扑、overlap 和负载均衡。

---

## 参考资料

先看 Ring Attention 论文理解分块推导，再看 Megatron Core 文档理解工程实现，最后看 DistFlashAttention 的仓库理解负载均衡和吞吐优化。本文推导统一使用 `S` 表示序列长度，`N` 表示 GPU 数，`d` 表示 head 维度，`X_r/Q_r/K_r/V_r` 表示第 `r` 张卡上的本地块。

理论文献：

1. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
2. [DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training](https://openreview.net/forum?id=pUEDkZyPDl)
3. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

官方文档：

4. [Megatron Core: Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
5. [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/parallelism-guide.html)

官方代码：

6. [Ring Attention 官方代码仓库](https://github.com/Selimonder/ring-attention)
7. [DistFlashAttention 官方代码仓库](https://github.com/RulinShao/LightSeq)
8. [Megatron-LM 官方代码仓库](https://github.com/NVIDIA/Megatron-LM)

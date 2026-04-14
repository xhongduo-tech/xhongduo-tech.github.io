## 核心结论

Sequence Parallelism，简称 SP，指的是把一段输入里的 token 按序列维度分到多张 GPU 上，每张卡只保存并计算自己那一段的激活。白话说，就是“不让每张卡都背完整篇文章，而是每张卡只背其中几段”。

它解决的核心问题不是“模型参数太大”，而是“长序列时中间激活太占内存”。当序列长度从 2K 增长到 32K、128K 甚至更长时，单卡往往先被激活内存压垮，而不是先被参数压垮。

SP 通常和 Tensor Parallelism，简称 TP，一起使用。TP 是“把一层的权重切到多卡”，SP 是“把一批 token 切到多卡”。两者叠加后，单卡同时少拿一部分参数视图和一部分序列激活，长上下文训练才有现实可行性。

一个常用配置检查公式是：

$$
dp\_replicate\_size \times dp\_shard\_size \times sp\_size = \text{总进程数}
$$

它表达的是：数据并行副本数、数据分片数、序列并行组大小三者相乘，必须正好覆盖全部 GPU。

玩具例子：输入长度是 1024，`sp_size=4`。原来 4 张卡都要各自拿 1024 个 token 的激活；现在改成每张卡只拿 256 个 token。后续某些层前后再通过 `all-gather` 或 `reduce-scatter` 交换结果，把“局部片段”恢复成“全局视图”。

| 对比项 | 原始做法 | 启用 SP 后 |
|---|---|---|
| 每卡持有的序列激活 | 全部 token | 仅本卡负责的 token 分片 |
| 主要收益 | 实现简单 | 单卡激活内存显著下降 |
| 额外代价 | 通信少 | 需要 `all-gather` / `reduce-scatter` / 某些实现中的 `all-to-all` |
| 与 TP 的关系 | 可单独使用 TP | SP 通常依赖 `tensor_parallel > 1` |
| 适用场景 | 中短序列 | 长序列、超长上下文训练 |

---

## 问题定义与边界

先明确问题。Attention，注意力机制，白话说就是“每个 token 去看别的 token，有选择地聚合信息”。如果直接按朴素形式实现，注意力分数矩阵大小接近 $L \times L$，其中 $L$ 是序列长度。所以序列越长，内存压力增长越快。

例如，`seq_len=4096` 时，单头注意力矩阵就有：

$$
4096^2 = 16{,}777{,}216
$$

个元素。多头、多层、反向传播再叠加后，激活内存很容易成为瓶颈。

这里要区分三个概念：

| 概念 | 解决什么问题 | 白话解释 |
|---|---|---|
| Data Parallelism | 吞吐和全局 batch | 多张卡各自跑不同样本 |
| Tensor Parallelism | 单层参数太大 | 一层权重拆给多张卡 |
| Sequence Parallelism | 长序列激活太大 | 一段输入拆给多张卡 |

SP 的边界也必须说清楚。

第一，它通常不是“单独开关”。在很多工程实现里，SP 只有在 `tensor_model_parallel_size > 1` 时才真正生效。NVIDIA NeMo 文档对这一点写得很明确：`sequence_parallel=True` 只有在 `tensor_model_parallel_size` 大于 1 时有效。

第二，不同框架下“SP”这个词有两个常见语境。

1. Megatron/NeMo 语境：把某些层的激活按序列维度切开，典型通信模式是 `reduce-scatter -> 局部算子 -> all-gather`。
2. Ulysses/DeepSpeed/Accelerate 语境：为了让 attention 在更长序列上可扩展，会同时涉及序列分片和 head 维度重排，底层常见 `all-to-all` 通信。

因此，初学者最稳妥的理解方式不是死记某个算子名，而是抓住共同抽象：序列维度被切分，本地只算局部片段，必要时通过集体通信恢复完整上下文。

第三，约束条件不能忽略。对于 Ulysses 一类实现，注意力头数通常要能被参与该副本的 GPU 数整除。可以写成：

$$
sp\_size \ge 2,\quad \text{并且 attention\_heads} \bmod sp\_size = 0
$$

更准确地说，这个整除约束在很多实现里是“参与单个 SP 副本的 GPU 数必须能和 head 切分兼容”。

真实工程例子：如果一个 32-head 的模型在 4 卡上做 Ulysses SP，那么每张卡可以稳定接到 8 个 head；但如果你把 `sp_size` 设成 3，就会在 head 分配或重排阶段直接出错，因为 32 不能整除 3。

---

## 核心机制与推导

先看最朴素的抽象。设输入隐藏状态是：

$$
H \in \mathbb{R}^{L \times d}
$$

其中 $L$ 是序列长度，$d$ 是 hidden size。若 `sp_size = s`，则把序列均匀切成 $s$ 段，每张卡只保留：

$$
H_i \in \mathbb{R}^{\frac{L}{s} \times d}
$$

这一步可以抽象记成：

$$
H_{sp} = \text{ReduceScatter}(H)
$$

这里的 `ReduceScatter` 可以理解为“先做需要的聚合，再把结果分发成每卡一段”。白话说，它不是简单切片，而是“切片前把该归并的信息先归并掉”。

当某些逐 token 算子只依赖本 token，不依赖全序列时，例如 LayerNorm，层归一化，白话说就是“对每个 token 自己的通道做标准化”，就可以直接在本地分片上算：

$$
\widetilde{H}_i = \text{LayerNorm}(H_i)
$$

MLP，多层感知机，白话说就是“对每个 token 独立做前馈变换”，也可以本地计算：

$$
Y_i = \text{MLP}(\widetilde{H}_i)
$$

如果后续模块需要完整序列，再通过：

$$
Y = \text{AllGather}(Y_i)
$$

恢复为全序列视图：

$$
Y \in \mathbb{R}^{L \times d}
$$

这就是很多 SP 资料里反复出现的链路：

$$
\text{ReduceScatter} \rightarrow \text{LayerNorm} \rightarrow \text{MLP} \rightarrow \text{AllGather}
$$

### 玩具例子：`seq_len=1024, sp_size=4`

输入 `hidden_states` 的形状是 `[1024, hidden]`。切分后，每张卡各拿 `[256, hidden]`。

1. 通信后，本卡只保留第 0 到 255、256 到 511、512 到 767、768 到 1023 中的一段。
2. LayerNorm 在本地做，因为它只看单个 token 的 hidden 维。
3. MLP 在本地做，因为它也是逐 token 前馈。
4. 如果下一步需要完整序列布局，就把四段 `all-gather` 回 `[1024, hidden]`。

残差连接也要注意形状一致。设残差输入是 $R$，主分支输出是 $Y$。如果两者都已经按序列切成同样分片，那么每张卡先局部相加：

$$
Z_i = R_i + Y_i
$$

最后再决定是否 `all-gather`。顺序的关键不是“先 gather 再加”，而是“确保相加双方的分片边界一致”。

下面这段 Python 不依赖分布式库，但能模拟这个过程：

```python
import numpy as np

def split_sequence(x, sp_size):
    assert x.shape[0] % sp_size == 0
    chunk = x.shape[0] // sp_size
    return [x[i * chunk:(i + 1) * chunk].copy() for i in range(sp_size)]

def all_gather(chunks):
    return np.concatenate(chunks, axis=0)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def mlp(x, w1, w2):
    hidden = np.maximum(0.0, x @ w1)  # ReLU
    return hidden @ w2

# toy example: seq_len=8, hidden=4, sp_size=2
x = np.arange(32, dtype=np.float32).reshape(8, 4)
residual = x.copy()
sp_size = 2

chunks = split_sequence(x, sp_size)
res_chunks = split_sequence(residual, sp_size)

w1 = np.ones((4, 6), dtype=np.float32) * 0.1
w2 = np.ones((6, 4), dtype=np.float32) * 0.2

out_chunks = []
for local_x, local_res in zip(chunks, res_chunks):
    y = layer_norm(local_x)
    y = mlp(y, w1, w2)
    z = y + local_res
    out_chunks.append(z)

full = all_gather(out_chunks)

assert full.shape == (8, 4)
assert np.allclose(full[:4], out_chunks[0])
assert np.allclose(full[4:], out_chunks[1])
```

这段代码对应的就是“先按序列分片，本地做 LayerNorm 和 MLP，本地加 residual，最后再拼回完整序列”。

需要再强调一次：对 attention 本体，实际工程实现往往比这个玩具例子复杂。Ulysses 一类实现会在“序列切分”和“head 切分”之间做额外重排，以保证每张卡仍能完成自己那部分注意力计算。这也是为什么官方文档会强调 Deepspeed 后端、通信拓扑、head 整除等约束。

---

## 代码实现

如果你从 Hugging Face Accelerate 或 DeepSpeed 角度落地，核心不是手写通信算子，而是把并行配置对齐。

下面是一个贴近官方接口含义的配置示意：

```python
from accelerate import Accelerator
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    dp_shard_size=1,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length=4096,
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",
    ),
)

accelerator = Accelerator(
    parallelism_config=parallelism_config,
)
```

如果写成更接近工程讨论的伪配置，也可以这样理解：

```python
ds_like_config = {
    "tensor_parallel": {
        "tp_world_size": 4,
        "sequence_parallel": True,
        "sp_size": 4,
    },
    "attention": {
        "impl": "flash_attention_2",
    },
}
```

参数含义最好单独列清楚：

| 参数 | 含义 | 常见检查 |
|---|---|---|
| `tensor_model_parallel_size` / `tp_world_size` | TP 组大小 | 必须大于 1，很多 SP 实现才生效 |
| `sp_size` | 一组里有多少张卡共同切序列 | 一般要求能和 head 数兼容 |
| `sequence_parallel=True` | 打开 SP 逻辑 | 不开就不会走相应通信路径 |
| `dp_replicate_size` | 数据并行副本数 | 与总 GPU 数一起核对 |
| `dp_shard_size` | 数据并行分片数 | 与 `sp_size` 相乘后要匹配进程数 |
| `sp_seq_length_is_variable` | 是否允许不同 batch 序列长度变化 | 动态长序列训练常设为 `True` |
| `sp_attn_implementation` | attention 后端实现 | 常见为 `sdpa` 或 `flash_attention_*` |

真实工程例子：4 卡训练一个 32-head 模型，想把长序列从 8K 拉到 32K。你可以设 `tp=4, sp_size=4, dp_shard_size=1`。这样 4 张卡共同服务一个 batch 的一段长序列，每张卡只保留约四分之一序列激活。如果再叠加 FlashAttention 和 activation checkpoint，往往能把“原本直接 OOM”的配置压到可训练区间。

这里还有一个经常被忽略的检查式：

$$
dp\_replicate\_size \times dp\_shard\_size \times sp\_size = \text{GPU 总数}
$$

比如你有 8 张卡，若 `sp_size=4`，同时想做两路数据并行，那么可以取：

$$
1 \times 2 \times 4 = 8
$$

这表示两组样本流，每组内部再用 4 卡做 SP。

---

## 工程权衡与常见坑

SP 的收益来自“单卡少存激活”，代价来自“卡间多通信”。所以它不是白送性能，而是用通信换内存。

最关键的权衡是下面这张表：

| 问题 | 影响 | 规避方法 |
|---|---|---|
| `tensor_model_parallel_size <= 1` 还想开 SP | 配置不生效或收益极小 | 先确认后端是否要求 TP>1 |
| `attention_heads % sp_size != 0` | head 分配失败，直接报错 | 让 head 数能整除参与 SP 的卡数 |
| 序列太短 | 通信开销大于内存收益 | 中短序列优先考虑 FlashAttention |
| 网络延迟高 | `all-gather` / `all-to-all` 成为瓶颈 | 尽量在同机高速互联内做 SP |
| 忘记核对总进程公式 | 并行组划分错误 | 检查 `dp × dp_shard × sp = GPU数` |
| 以为 SP 能替代一切优化 | 仍然 OOM 或吞吐差 | 与 checkpoint、FlashAttention 叠加 |

一个典型坑是：4 卡环境里把 `sp_size=3`。如果模型是 32 个 attention heads，那么不管你从逻辑上怎么解释，“32 个头分给 3 张卡”都无法均匀切分，很多实现会在初始化或第一次 forward 时抛错。

另一个典型坑是：序列只有 512 或 1024，却强行开启 SP。此时 attention 本身并不大，反而是通信固定成本占主导。vllm-ascend 文档给出的经验值是，Ascend 上 `num_tokens < 1000` 时，SP 甚至可能带来负收益，因此提供了 `sp_min_token_num` 这样的门槛。

还有一个理解误区：很多人看到“按序列切分”，就以为每张卡从头到尾只看自己的 token，不再需要和别人通信。这是不对的。对于 transformer，某些步骤是逐 token 独立的，可以本地算；但注意力、残差布局恢复、某些归并阶段仍然需要集体通信。SP 不是“去掉通信”，而是“改变通信发生的位置和粒度”。

---

## 替代方案与适用边界

SP 不是唯一办法。它最适合的情况是：序列非常长，单卡主要卡在激活内存，而你又已经具备多卡和较好的互联。

常见替代方案如下：

| 方案 | 主要解决点 | 通信代价 | 实现复杂度 | 更适合什么情况 |
|---|---|---|---|---|
| FlashAttention | 降低 attention 内存与访存开销 | 低 | 中 | 单卡或少卡，中长序列 |
| Activation Checkpoint | 以重算换内存 | 低 | 低 | 先做基础降内存 |
| Tensor Parallelism | 切参数与部分激活 | 中 | 中 | 模型层本身太大 |
| Sequence Parallelism | 切序列激活 | 中到高 | 高 | 长序列、TP 已开启 |
| Context Parallelism | 更广义地切整个上下文计算 | 中到高 | 高 | 更长序列，特定后端 |

可以用一句话概括它们的分工：

1. FlashAttention 优先优化“attention 这一步怎么更省”。
2. Checkpoint 优先优化“中间值少存一点，回头再算”。
3. SP 优先优化“别让每张卡都背完整序列”。

如果只是 `seq_len <= 2048`，而且只有 1 到 2 张卡，通常先上 FlashAttention 和 activation checkpoint 就够了。此时 SP 的通信和工程复杂度往往不划算。

如果是 `seq_len > 2048`，尤其是 8K、32K、128K 甚至更长，并且已经在做 TP，那么 SP 才会明显进入“值得考虑”的区间。可以把经验边界粗写成：

$$
seq\_len \text{ 足够长} \land tensor\_parallel > 1 \land 网络延迟可控
$$

此时优先级通常是：先确认 FlashAttention 可用，再评估 checkpoint，最后在长序列场景下叠加 SP。

一个简单对比配置如下：

```python
# 只用 FlashAttention
config_a = {
    "tensor_parallel": 1,
    "sequence_parallel": False,
    "attn_impl": "flash_attention_2",
}

# TP + SP
config_b = {
    "tensor_parallel": 4,
    "sequence_parallel": True,
    "sp_size": 4,
    "attn_impl": "flash_attention_2",
}
```

前者更适合“先把单卡做到极限”；后者更适合“长序列已经超过单卡极限，只能靠多卡切序列继续往上推”。

---

## 参考资料

| 文档 | 重点介绍 |
|---|---|
| Hugging Face Accelerate Sequence Parallelism Concept Guide | SP 的动机、`sp_size` 配置、与 DeepSpeed/Ulysses 的关系、`dp_replicate_size × dp_shard_size × sp_size = num_processes` 的配置约束 |
| vllm-ascend Sequence Parallelism | Ascend 场景下的 SP 开关、`sp_min_token_num` 经验值、通信开销在小 token 数下可能反超收益 |
| NVIDIA NeMo Parallelisms 文档 | SP 在 Megatron/NeMo 体系中的位置，以及“只有 `tensor_model_parallel_size > 1` 时才有效”的边界 |

- Hugging Face Accelerate: https://huggingface.co/docs/accelerate/en/concept_guides/sequence_parallelism
- vllm-ascend: https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/sequence_parallelism.html
- NVIDIA NeMo: https://docs.nvidia.com/nemo-framework/user-guide/25.04/nemotoolkit/features/parallelisms.html

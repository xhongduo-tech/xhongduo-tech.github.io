## 核心结论

LLaMA 可以看成一种被大规模验证过的 decoder-only Transformer 基线。decoder-only 的白话解释是：模型只做“看前文再预测下一个 token”，不包含单独的编码器。它的稳定组合基本是三件事：pre-RMSNorm、SwiGLU、RoPE。

pre-RMSNorm 的意思是“每个子层前先做归一化”，RMSNorm 是“只按均方根缩放，不减去均值的归一化”。SwiGLU 是“带门控的前馈网络”，门控就是让一部分通道决定另一部分通道该放大还是压小。RoPE 是“把位置信息写进向量旋转角度里的相对位置编码”。

从 LLaMA-1 到 LLaMA-2，再到 LLaMA-3，主干没有大改，变化主要集中在三个工程点：

| 版本 | 主干组件 | 词表 | 上下文长度 | 注意力细节 |
| --- | --- | --- | --- | --- |
| LLaMA-1 7B | pre-RMSNorm + SwiGLU + RoPE | 32K BPE | 2048 | 标准 MHA |
| LLaMA-2 7B/13B | 主干不变 | 32K BPE | 4096 | 小模型多为 MHA，34B/70B 使用 GQA |
| LLaMA-3 8B/70B | 主干仍不变 | 128K tiktoken BPE | 8192 | 全尺寸使用 GQA，KV 组数统一为 8 |

这套组合能成为事实标准，不是因为它最“新”，而是因为它在训练稳定性、推理成本、长上下文扩展和实现复杂度之间取得了很强的平衡。对工程侧最重要的一点是：GQA 加上 RoPE，使上下文变长时，KV cache 的增长仍然可控。

公式上，LLaMA 的两个核心部件可以先记住：

$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}\cdot g
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

一个标准 LLaMA block 的直观流程是：输入 token 表示先做 RMSNorm，再分别进入注意力子层和 SwiGLU 前馈子层，每个子层输出都加回残差，输入输出维度保持不变，因此可以重复堆叠 32 层、40 层甚至更多层。

玩具例子可以这样理解：假设一个 token 的隐藏向量维度是 4，先做 RMSNorm 只是把整体尺度拉回稳定区间；再做 attention 让它从前文里取信息；再做 SwiGLU 让它按任务需要重新组合特征。整个过程中，向量长度不变，所以 block 可以一层一层堆起来。

---

## 问题定义与边界

讨论 LLaMA 的“架构细节”，核心不是列出所有超参数，而是回答三个问题：

1. 它和“标准 Transformer”相比，到底换了哪些关键部件。
2. 这些部件为什么会一起出现，而不是单独存在。
3. 哪些是主干设计，哪些只是版本迭代中的工程调整。

边界先划清楚。本文只讨论 LLaMA 系列的主干架构，不展开训练数据配比、指令微调、RLHF、量化和推理服务框架。也不把所有长上下文技巧都归到 LLaMA 身上，例如 ALiBi、状态空间模型、线性注意力都属于替代路线，不是 LLaMA 主线。

LLaMA 与早期“标准 Transformer”最关键的区别如下：

| 维度 | 标准 Transformer 常见设定 | LLaMA 设定 |
| --- | --- | --- |
| 总体结构 | encoder-decoder 或 decoder-only 都常见 | 纯 decoder-only |
| 归一化位置 | post-norm 或 pre-norm | pre-norm |
| 归一化类型 | LayerNorm | RMSNorm |
| FFN | 两层 MLP 或 GELU FFN | SwiGLU FFN |
| 位置处理 | 绝对位置 embedding 常见 | 无绝对位置 embedding，使用 RoPE |
| 注意力头 | 多数是 MHA | 后续版本大量使用 GQA |
| 部署目标 | 训练表达能力优先 | 兼顾训练稳定与推理成本 |

这里有两个容易混淆的边界。

第一，LLaMA 不是“完全新架构”，而是在 Transformer 框架内做了一组高度实用的部件选择。也就是说，注意力公式没变，残差连接没变，因果 mask 也没变，变的是归一化、前馈层、位置编码和 KV 共享方式。

第二，LLaMA-2 和 LLaMA-3 的升级，大部分不是推翻 LLaMA-1，而是在原基线上做实用扩展。比如词表变大、上下文变长、GQA 覆盖范围扩大。这意味着：如果你已经理解一个 LLaMA block，后面版本的大部分变化都能沿着同一条线理解。

---

## 核心机制与推导

### 1. 为什么是 pre-RMSNorm

RMSNorm 只关心向量整体能量，不像 LayerNorm 那样还要减均值。白话说，LayerNorm 会先把“中心位置”对齐再缩放，RMSNorm 只负责把“幅度大小”调稳。这样做的好处是计算更简单，也常被证明足够稳定。

pre-norm 的意义是：先归一化，再进入注意力或 FFN。这样残差主干更像一条稳定高速路，梯度在深层网络中更容易传递。对大模型来说，这类稳定性比单层表达力的小幅提升更重要。

### 2. 为什么是 SwiGLU

SwiGLU 不是普通两层 FFN，而是三组投影矩阵。最简形式可以写成：

$$
\mathrm{FFN}(x)=W_2\big(\mathrm{SiLU}(W_1x)\odot W_3x\big)
$$

其中 $\odot$ 是逐元素乘法。逐元素乘法的白话解释是：两个同形状向量按位置一一相乘。这里 $\mathrm{SiLU}(W_1x)$ 像一个门，决定 $W_3x$ 哪些维度该通过、哪些该抑制。

LLaMA 里 FFN 中间层宽度通常不是传统的 $4d$，而是接近 $\frac{8}{3}d$ 的实用整数近似。原因是 SwiGLU 已经有门控分支，参数效率和表达能力的平衡点与普通 ReLU/GELU FFN 不同。

### 3. 为什么是 RoPE

RoPE 是旋转位置编码。白话说，它不额外给 token 加一个位置向量，而是直接把 query 和 key 的每一对二维分量按位置做旋转。这样，相对距离会自然体现在内积里。

形式上，可以把某个二维子向量看成：

$$
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\cos \theta_m & -\sin \theta_m \\
\sin \theta_m & \cos \theta_m
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

其中位置 $m$ 的角频率通常写成：

$$
\theta_{m,i}=m \cdot \omega_i,\quad \omega_i=\text{base}^{-2i/d}
$$

这套设计的关键点不是“给每个位置一个编号”，而是“让不同位置的相对偏移变成可学习的角度差”。因此它比绝对位置 embedding 更适合做长度外推和长上下文扩展。

玩具例子：假设两个 token 的某个二维分量分别是 $(1,0)$ 和 $(1,0)$。位置 3 和位置 5 经过 RoPE 后会被旋转成不同角度。此时它们做内积时，结果不再只看原向量相似度，还会带上位置差的信息。这就是“相对位置被写进注意力分数”的直观来源。

### 4. 为什么 GQA 会显著省推理显存

MHA 是 Multi-Head Attention，多头注意力，意思是每个 query 头都有自己独立的一组 key 和 value 头。GQA 是 Grouped Query Attention，分组查询注意力，意思是多个 query 头共享一组 KV 头。

设隐藏维度 $d=4096$，query 头数 $H_q=32$，每头维度 $d_h=128$。如果是标准 MHA，则每个 token 需要缓存：

$$
H_q \times d_h \times 2 = 32 \times 128 \times 2 = 8192
$$

这里乘 2 是因为要同时存 key 和 value。

如果改成 GQA，令 KV 头数 $H_{kv}=8$，则每个 token 的 KV cache 变成：

$$
H_{kv} \times d_h \times 2 = 8 \times 128 \times 2 = 2048
$$

于是缓存开销直接降为原来的：

$$
\frac{2048}{8192}=25\%
$$

这也是 LLaMA-3 8B/70B 在 8K 上下文推理时很关键的工程收益。上下文长度为 8192 时，单层每个样本的 KV 元素数量从 $8192\times8192$ 降到 $8192\times2048$。真实显存还要乘层数、batch、数据类型字节数，但比例关系不变。

真实工程例子：如果你在一台 8 卡服务器上部署 70B 模型，延长上下文后最先爆掉的通常不是参数本体，而是 KV cache。GQA 不能减少参数前向计算的全部成本，但能明显降低长对话场景中的缓存占用，因此对客服、企业知识助手、多轮 RAG 这类任务特别有价值。

---

## 代码实现

下面给一个最小可运行的 Python 版本，展示 RMSNorm、简化的 GQA 形状映射和 KV cache 计算。它不是完整训练代码，但足够把 LLaMA block 的骨架看清楚。

```python
import math
import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    # x: [batch, seq, dim]
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def repeat_kv(x, num_query_groups):
    # x: [batch, seq, h_kv, head_dim]
    # 把每个 KV 头复制给多个 query 头组
    return np.repeat(x, repeats=num_query_groups, axis=2)


def gqa_shapes(batch, seq, dim, h_q, h_kv):
    assert dim % h_q == 0
    assert h_q % h_kv == 0

    head_dim = dim // h_q
    group_size = h_q // h_kv

    x = np.random.randn(batch, seq, dim).astype(np.float32)
    weight = np.ones((dim,), dtype=np.float32)

    x_norm = rmsnorm(x, weight)
    assert x_norm.shape == (batch, seq, dim)

    # 这里只模拟投影后的形状，不展开真实线性层
    q = np.random.randn(batch, seq, h_q, head_dim).astype(np.float32)
    k = np.random.randn(batch, seq, h_kv, head_dim).astype(np.float32)
    v = np.random.randn(batch, seq, h_kv, head_dim).astype(np.float32)

    k_expanded = repeat_kv(k, group_size)
    v_expanded = repeat_kv(v, group_size)

    assert q.shape == k_expanded.shape == v_expanded.shape
    return {
        "head_dim": head_dim,
        "group_size": group_size,
        "q_shape": q.shape,
        "k_cache_per_token": h_kv * head_dim,
        "v_cache_per_token": h_kv * head_dim,
    }


def kv_cache_units_per_token(h_kv, head_dim):
    return h_kv * head_dim * 2


info = gqa_shapes(batch=2, seq=16, dim=4096, h_q=32, h_kv=8)
assert info["head_dim"] == 128
assert info["group_size"] == 4

mha_units = kv_cache_units_per_token(h_kv=32, head_dim=128)
gqa_units = kv_cache_units_per_token(h_kv=8, head_dim=128)

assert mha_units == 8192
assert gqa_units == 2048
assert gqa_units / mha_units == 0.25
print(info)
```

如果把它翻译回真实 LLaMA block，结构大致是：

1. 输入 `x` 先做 `RMSNorm`
2. 投影成 `q, k, v`
3. 对 `q, k` 施加 RoPE
4. 用因果 mask 做 attention
5. 输出加回残差
6. 再做一次 `RMSNorm`
7. 进入 SwiGLU FFN
8. 再加回残差

伪代码形状如下：

```python
# Query head count = H_q, KV head count = H_kv
assert H_q % H_kv == 0

x1 = rmsnorm(x)
q = linear_q(x1).reshape(batch, seq, H_q, head_dim)
k = linear_k(x1).reshape(batch, seq, H_kv, head_dim)
v = linear_v(x1).reshape(batch, seq, H_kv, head_dim)

q, k = apply_rope(q, k, positions)
y = gqa_attention(q, k, v, causal=True)
x = x + proj_o(y)

x2 = rmsnorm(x)
ff = down_proj(silu(gate_proj(x2)) * up_proj(x2))
x = x + ff
```

这里最容易忽略的一点是：GQA 不是修改 softmax 公式，而是修改 KV 的头组织方式。也就是说，attention 仍然是标准 softmax attention，只是多个 query 头共享同一组 key/value。

---

## 工程权衡与常见坑

LLaMA 架构“看起来简单”，但工程上有几个坑反复出现。

| 问题 | 现象 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| RoPE 直接拉长上下文 | 长文本后半段质量明显下降 | 频率设定与训练分布不匹配 | 同步调整 RoPE base、位置缩放和继续训练数据 |
| GQA 头数不整除 | kernel 报 shape mismatch | `H_q % H_kv != 0` | 设计时保证整除，常见如 32 对 8 |
| FlashAttention 不支持当前配置 | 推理或训练直接回退到慢路径 | 内核只支持部分 head 布局或 dtype | 核对框架版本、dtype、causal/GQA 支持 |
| 词表升级后迁移微调失败 | 嵌入层尺寸不兼容 | tokenizer 改了，embedding/output head 也要变 | 明确区分 32K 与 128K 词表模型 |
| 误以为 GQA 一定“更强” | 训练指标不升反降 | GQA 首先是推理效率优化，不是无条件提升表达力 | 按模型规模和部署目标决定是否启用 |

一个容易误解的点是“上下文扩展”。把 `max_position_embeddings` 改大，不等于模型真正学会更长上下文。因为 RoPE 的角频率分布、训练样本长度、mask 策略和继续训练过程都在共同决定效果。只改配置、不做再训练，往往只能得到“能跑更长”，而不是“能正确利用更长”。

再看一个真实工程例子。假设你部署一个企业问答助手，单轮上下文需要 8192 token，用户并发几十路。若仍用标准 32 头 KV 缓存，显存和带宽压力会迅速放大；改成 8 组 GQA 后，缓存占用可降到 25%，这通常直接决定你是能在单机完成服务，还是必须额外切分到更多 GPU。这个收益对推理系统比对训练系统更直接。

一个简短的检查清单：

| 检查项 | 建议 |
| --- | --- |
| 归一化 | 确认是 pre-RMSNorm，不要混成 post-norm |
| FFN | 确认是 SwiGLU 三投影，而不是普通两层 MLP |
| 位置编码 | 确认没有绝对位置 embedding，RoPE 只作用于 Q/K |
| GQA | 确认 `H_q` 是 `H_kv` 的整数倍 |
| 长上下文 | 不只改配置，还要评估 RoPE 扩展和继续训练 |
| tokenizer | 32K 与 128K 词表不要混用权重 |

---

## 替代方案与适用边界

LLaMA 不是所有任务的默认最优解，它更像“面向通用生成任务的高性价比基线”。

| 方案 | 适合任务 | 优点 | 不足 |
| --- | --- | --- | --- |
| LLaMA 式 decoder-only | 对话、补全、代码生成、通用助手 | 实现成熟，推理生态强，GQA 降 KV 成本 | 对纯编码任务不如双向模型自然 |
| GPT-3-like decoder-only | 通用生成 | 路线接近，迁移容易 | 具体实现不一定含 GQA/RoPE 这些效率优化 |
| T5/UL2 encoder-decoder | 翻译、摘要、结构化输入到输出 | 输入输出分工清晰 | 自回归部署链路更复杂 |
| 标准 MHA + RoPE | 中小模型或训练优先 | 实现最直接 | 长上下文推理显存压力更大 |
| 更激进的长上下文方案 | 128K 以上、检索增强、超长文档 | 上限更高 | 训练和内核实现复杂度更高 |

为什么 LLaMA 仍坚持 decoder-only？因为大语言模型主流应用是“给定前文，继续生成”。客服、助手、代码补全、文档问答都天然符合因果生成接口。对这类任务，decoder-only 的训练目标和推理形式一致，工程链路最短。

什么时候不该直接套 LLaMA 设定？

1. 你的任务本质是编码-解码，例如机器翻译、信息抽取到结构化输出，这时 T5 类模型可能更自然。
2. 你的目标上下文远超 8K/32K，且必须稳定利用远距离信息，这时只保留标准 RoPE 设定往往不够，需要额外的位置缩放、继续训练，甚至换成长上下文专用方案。
3. 你的部署完全不受显存限制，而训练表达力更优先，这时是否使用 GQA 要通过实验决定，不能把“更省”误当成“更强”。

因此，一个实用判断可以写成：如果你做的是通用生成任务，需要成熟实现、可控成本和稳定推理，LLaMA 栈通常是默认起点；如果你做的是强结构输入输出或极长上下文任务，就要先检查任务边界，再决定是否沿用这套主干。

---

## 参考资料

1. [LLaMA 核心组件解析](https://mbrenndoerfer.com/writing/llama-components-rmsnorm-swiglu-rope?utm_source=openai)：重点解释 pre-RMSNorm、SwiGLU、RoPE 的配置含义，适合建立组件级理解。
2. [LLaMA 2 / LLaMA 3 家族演进](https://www.emergentmind.com/topics/llama-2-and-llama-3-families?utm_source=openai)：重点看词表、上下文长度和 GQA 覆盖范围的版本变化，适合做代际对比。
3. [LLaMA 架构综述与公式说明](https://kikaben.com/llama-2023-02/?utm_source=openai)：重点看 RMSNorm、attention、SWinGLU 等核心公式，适合补基础。
4. [Grouped Query Attention 说明](https://parasdahal.com/notes//Grouped%2BQuery%2BAttention%2B%28GQA%29?utm_source=openai)：重点看 GQA 的 KV cache 推导和资源占用变化，适合理解推理优化。
5. [GQA 工程复盘](https://tildalice.io/gqa-grouped-query-attention-review/?utm_source=openai)：重点看头数整除、实现约束和常见 kernel 问题，适合工程排错。
6. [Dell PowerEdge XE9680 上运行 LLaMA 3 的案例](https://infohub.delltechnologies.com/es-es/p/running-meta-llama-3-models-on-dell-poweredge-xe9680/?utm_source=openai)：重点看真实部署场景下的显存与吞吐考虑，适合把架构选择和硬件成本联系起来。
7. [LLaMA 3 8B 与长上下文设定综述](https://www.emergentmind.com/topics/llama-3-8b?utm_source=openai)：重点看 RoPE 频率、上下文扩展和稳定性问题，适合理解“能跑更长”与“学会更长”的差别。

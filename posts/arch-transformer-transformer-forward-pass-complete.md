## 核心结论

Decoder-only Transformer 的完整前向传播，可以压缩成一条固定路径：

| 步骤 | 操作 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 1 | Token Embedding | token IDs | $x_0 \in \mathbb{R}^{T \times d}$ |
| 2 | 每层注意力子层 | $x_l$ | $x_l + \text{Attn}(\text{RMSNorm}(x_l))$ |
| 3 | 每层 FFN 子层 | 上一步结果 | $x_{l+1} = x'_l + \text{FFN}(\text{RMSNorm}(x'_l))$ |
| 4 | 最终归一化 | $x_L$ | $\hat{x}_L$ |
| 5 | LM Head | $\hat{x}_L$ | logits |

这里的 `logits` 就是“每个词表位置的未归一化分数”，它还不是概率，但经过 softmax 后就能变成“下一个 token 的概率分布”。

最关键的结构不是“注意力”三个字本身，而是下面这个组合：

$$
x_{l+1}=x_l+\text{Sublayer}(x_l)
$$

这叫残差连接，白话解释是“子层只负责修正原始信号，不负责从零重造整个表示”。它让深层网络中的信息和梯度都能直接穿过去。

对现代大模型推理来说，真正重要的工程组合通常是：

1. Pre-RMSNorm：先归一化，再进子层，保证输入幅值稳定。
2. RoPE：把位置信息直接编码进 $Q,K$ 的旋转关系里。
3. KV Cache：把历史 token 的 Key/Value 缓存起来，避免每次生成都重算整段上下文。

玩具例子可以用 4 个 token 直观看。假设输入序列是：

`["我", "喜欢", "学", "习"]`

它会经历下面的变化：

1. token IDs 先查表得到 4 个长度为 $d$ 的 embedding 向量。
2. 第 1 层注意力里，每个 token 都算出自己的 $Q,K,V$。
3. 因果掩码生效后，第 4 个 token 只能看前 4 个位置，第 2 个 token 只能看前 2 个位置。
4. 注意力输出加回原输入，得到“保留原语义 + 引入上下文”的表示。
5. 再经过 SwiGLU FFN，对每个位置单独做非线性变换，再残差加回。
6. 所有层做完后，最后一个位置的向量进入 LM Head，输出下一个 token 的 logits。

这条链路里，残差的含义非常具体：每一步都不是“替换旧表示”，而是“在旧表示上叠加一个修正项”。

---

## 问题定义与边界

本文只讨论 Decoder-only Transformer 的前向传播，也就是：

输入一句已经分词并编码成 token IDs 的文本，模型如何一步步算出“下一个 token 应该是什么”的分数。

边界要先说清楚：

| 术语 | 含义 | 本文是否展开 |
| --- | --- | --- |
| Tokenizer | 把文本切成 token 并映射成整数 ID 的工具 | 只作为输入前提 |
| Embedding | 把离散 ID 变成连续向量的查表层 | 展开 |
| Pre-RMSNorm | 先归一化再进子层的规范化方式 | 展开 |
| RoPE | Rotary Positional Embedding，旋转位置编码，用旋转方式注入位置信息 | 展开 |
| KV Cache | 推理时缓存历史 Key/Value 的机制 | 展开 |
| 反向传播 | 训练时计算梯度并更新参数 | 不展开 |
| 多机并行 | 张量并行、流水并行等部署技术 | 不展开 |

如果用新手版一句话描述整个问题，就是：

“输入一句话后，Transformer 如何从 token IDs 开始，经过 Embedding、Attention、FFN、RMSNorm，最终得到下一个词的 logits？”

可以把整体看成一张简化流程图：

`输入 token IDs -> Embedding -> 多层[Pre-RMSNorm -> Attention -> 残差 -> Pre-RMSNorm -> FFN -> 残差] -> 最终 RMSNorm -> LM Head -> logits`

这里有两个容易混淆的点。

第一，前向传播的输出不是“一个词”，而是一整个词表上的分数向量。采样、贪心解码、top-k、top-p 这些都发生在前向传播之后。

第二，训练和推理虽然都走同一条前向链路，但推理会引入 KV Cache，因为生成阶段每次只新增一个 token，缓存能显著减少重复计算。本文重点放在这条“完整推理链路”上，不讨论训练时的反向细节。

---

## 核心机制与推导

先看单层的核心公式。设输入为 $x \in \mathbb{R}^{T \times d}$，其中 $T$ 是序列长度，$d$ 是模型宽度。

### 1. Pre-RMSNorm

RMSNorm 的白话解释是“只根据向量整体能量缩放，不减均值”。公式为：

$$
\text{RMSNorm}(x)=\gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
$$

它和 LayerNorm 的差别在于：RMSNorm 不做中心化，只做按均方根缩放，因此更轻，也常见于 LLaMA 一类结构。

Pre-RMSNorm 表示先算 $\text{RMSNorm}(x)$，再把结果送进注意力或 FFN。它的价值是让每个子层都看到尺度稳定的输入。对白话一点说，深层网络里每层拿到的“信号振幅”比较可控，不容易一层层放大失控。

### 2. Self-Attention

注意力先把输入投影成三组向量：

$$
Q=xW_Q,\quad K=xW_K,\quad V=xW_V
$$

其中 Query 可以理解为“我想找什么”，Key 可以理解为“我这里有什么标签”，Value 可以理解为“真正要被取出的内容”。

然后计算分数矩阵：

$$
S=\frac{QK^T}{\sqrt{d_k}}+\text{causal\_mask}
$$

其中 $d_k$ 是每个 head 的维度，$\sqrt{d_k}$ 是缩放因子，避免点积值随着维度升高而过大。

再做 softmax：

$$
A=\text{softmax}(S)
$$

最后聚合 Value：

$$
\text{Attention}(x)=A V W_O
$$

其中 $W_O$ 是输出投影矩阵，把多头结果重新映射回模型维度。

因果掩码 `causal mask` 的作用是禁止当前位置看未来位置。对第 $t$ 个 token，只允许看 $1...t$，不允许看 $t+1...T$。否则生成时会偷看答案。

### 3. RoPE

RoPE 的白话解释是“把位置信息编码成向量平面上的旋转角度”。它不是把一个位置向量直接加到 embedding 上，而是对 $Q,K$ 的偶数维和奇数维成对旋转。

对二维子向量 $(q_{2i}, q_{2i+1})$，位置 $p$ 上的旋转可写成：

$$
\text{RoPE}(q,p)=
\begin{bmatrix}
\cos \theta_{p,i} & -\sin \theta_{p,i}\\
\sin \theta_{p,i} & \cos \theta_{p,i}
\end{bmatrix}
\begin{bmatrix}
q_{2i}\\
q_{2i+1}
\end{bmatrix}
$$

$K$ 也做同样旋转。这样一来，$QK^T$ 的结果会自然携带相对位置信息。重点不是“第 7 个 token 的绝对编号”，而是“两个 token 相距多远”。

用一个玩具例子看形状更直观。假设：

- 序列长度 $T=4$
- 模型维度 $d=128$
- 头数 $H=2$
- 每头维度 $d_{head}=64$

则：

- 输入 $x$ 形状：$(4,128)$
- $Q,K,V$ 投影后形状：$(4,128)$
- reshape 成多头后：$(2,4,64)$
- 每个 head 的注意力分数矩阵：$(4,4)$

第 3 个 token 的某个 head 在打分时，只能对位置 1、2、3 打分，位置 4 会被 mask 成负无穷，softmax 后权重为 0。

### 4. SwiGLU FFN

FFN 是“逐位置的前馈网络”，白话解释是“每个 token 单独过一个更宽的非线性变换层，不做位置间交互”。

SwiGLU 常见写法是：

$$
\text{SwiGLU}(x)=\left(\text{SiLU}(xW_g)\odot xW_u\right)W_d
$$

其中：

- $W_g$ 是 gate 投影
- $W_u$ 是 up 投影
- $W_d$ 是 down 投影
- $\odot$ 是逐元素乘法

SiLU 是一种平滑激活函数：

$$
\text{SiLU}(z)=z\cdot \sigma(z)
$$

可以把 SwiGLU 理解成“先扩维，再让一个门控分支控制哪些维度该通过，最后再压回模型维度”。它比老式 ReLU FFN 表达能力更强，也是现代大模型常见选择。

### 5. Pre-Norm 与 Post-Norm 对比

| 方案 | 子层形式 | 优点 | 风险 |
| --- | --- | --- | --- |
| Post-Norm | $x+\text{Sublayer}(x)$ 后再 Norm | 早期 Transformer 常见 | 深层训练更不稳定 |
| Pre-Norm | $x+\text{Sublayer}(\text{Norm}(x))$ | 梯度路径更直接，深层更稳 | 输出尺度依赖残差设计 |
| Pre-RMSNorm | Pre-Norm 的轻量变体 | 计算更省，现代 LLM 常用 | 仍需合理初始化与缩放 |

现代 Decoder-only 模型大多选 Pre-RMSNorm，不是因为它更“新”，而是它在几十层甚至上百层时更稳定。

---

## 代码实现

下面用一个精简但可运行的 Python 例子，演示“Embedding -> causal self-attention -> residual -> RMSNorm -> logits”的最小前向链路。它不是工业实现，但足够说明数据怎样流动。

```python
import numpy as np

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def causal_mask(seq_len):
    mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
    return np.triu(mask, k=1)

def attention(x, Wq, Wk, Wv, Wo):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)
    scores = scores + causal_mask(x.shape[0])
    probs = softmax(scores, axis=-1)
    out = probs @ V
    return out @ Wo, probs

# toy config
V = 8      # vocab size
d = 4      # model dim
T = 4      # sequence length

token_ids = np.array([1, 3, 2, 5])
embed = np.array([
    [0.0, 0.1, 0.2, 0.3],
    [0.2, 0.0, 0.1, 0.4],
    [0.3, 0.1, 0.0, 0.2],
    [0.4, 0.2, 0.1, 0.0],
    [0.1, 0.3, 0.2, 0.0],
    [0.0, 0.2, 0.4, 0.1],
    [0.3, 0.0, 0.2, 0.1],
    [0.2, 0.4, 0.0, 0.1],
], dtype=np.float32)

x = embed[token_ids]
norm_w = np.ones((d,), dtype=np.float32)

Wq = np.eye(d, dtype=np.float32)
Wk = np.eye(d, dtype=np.float32)
Wv = np.eye(d, dtype=np.float32)
Wo = np.eye(d, dtype=np.float32)
lm_head = embed.T  # weight tying

x_norm = rms_norm(x, norm_w)
attn_out, probs = attention(x_norm, Wq, Wk, Wv, Wo)
x = x + attn_out
x = rms_norm(x, norm_w)
logits = x @ lm_head

assert x.shape == (T, d)
assert logits.shape == (T, V)
assert np.allclose(probs[0, 1:], 0.0)  # 第一个 token 不能看未来
assert np.argmax(logits[-1]) >= 0

print("last token logits:", logits[-1])
```

如果把完整 Decoder block 写成伪代码，大致是：

```python
x = embedding[token_ids]

for layer in layers:
    h = rms_norm(x)
    attn_out, cache = attention_with_rope_and_kv_cache(h, cache, layer)
    x = x + attn_out

    h = rms_norm(x)
    ffn_out = swiglu_ffn(h)
    x = x + ffn_out

x = rms_norm(x)
logits = x @ lm_head
```

真实工程里，推理阶段最重要的差别在 KV Cache。它的基本思路是：

```python
def decode_one_token(x_new, cache, layer):
    h = rms_norm(x_new)

    q = project_q(h)
    k = project_k(h)
    v = project_v(h)

    q, k = apply_rope(q, k, current_pos=cache["len"])

    cache["key"][layer] = concat(cache["key"][layer], k, axis=1)
    cache["value"][layer] = concat(cache["value"][layer], v, axis=1)

    all_k = cache["key"][layer]
    all_v = cache["value"][layer]

    attn_out = attend(q, all_k, all_v, causal=True)
    return attn_out, cache
```

KV Cache 的结构可以抽象成：

| 字段 | 含义 | 常见维度 |
| --- | --- | --- |
| `cache["key"][l]` | 第 `l` 层所有历史 token 的 Key | $(B, H_{kv}, T, d_{head})$ |
| `cache["value"][l]` | 第 `l` 层所有历史 token 的 Value | $(B, H_{kv}, T, d_{head})$ |
| `cache["len"]` | 当前已缓存 token 数 | 标量 |

它的关键不是“只算新 Q”，而是“新 token 只新增一组 K/V，旧 token 的 K/V 不再重算”。这样每一步生成的计算量才会从“重算整段上下文”下降到“对新 token 做一次投影，再和历史 cache 做注意力”。

---

## 工程权衡与常见坑

真实工程例子里，最大的瓶颈往往不是参数本身，而是 KV Cache。

KV Cache 的单层单 token 显存公式可写成：

$$
\text{bytes per token}=2 \times H_{kv} \times d_{head} \times \text{bytes}
$$

乘上层数 $L$ 和上下文长度 $T$ 后：

$$
\text{KV Cache Size}=2 \times L \times H_{kv} \times d_{head} \times \text{bytes} \times T
$$

其中前面的 2 表示要存 K 和 V 两份。

对 LLaMA-2-7B 这类 FP16 模型，常见经验值是每个 token 的 KV Cache 大约 0.5 MB，所以：

- 1K token 约 512 MB
- 2K token 约 1 GB

这就是很多人第一次做长上下文推理时会踩的坑：模型权重能放进显存，不代表推理时也能跑得动，因为上下文越长，缓存线性增长。

下面给一个更接近部署的量级表：

| 模型/设置 | 上下文长度 | KV Cache 量级 |
| --- | --- | --- |
| LLaMA-2-7B FP16 | 1K | 约 0.5 GB |
| LLaMA-2-7B FP16 | 2K | 约 1 GB |
| Llama 3.1 8B | 32K | 约 4 GB |
| Llama 3.1 8B | 128K | 约 16 GB |

这类数据说明一个现实问题：长上下文推理时，显存压力常常先被 KV Cache 吃掉，而不是先被参数吃掉。

再看 FLOP 分布，Decoder-only 模型的一层计算通常近似是：

| 组件 | FLOP 占比 |
| --- | --- |
| FFN / SwiGLU | 约 66% 到 67% |
| QKV 与输出投影 | 约 25% |
| Attention score 与加权求和 | 约 8% |

所以很多初学者会直觉认为“注意力最贵”，这不完全对。对大多数常见配置，FFN 才是前向 FLOP 大头，注意力更容易成为“长上下文下的内存瓶颈”。

常见坑主要有四类。

| 坑 | 表现 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 忽略 causal mask | 模型训练或推理异常乐观 | 偷看未来 token | 分数矩阵加上上三角负无穷 |
| 去掉 Pre-RMSNorm | 深层训练不稳 | 子层输入幅值漂移 | 保留 pre-norm 结构 |
| 不做 KV Cache | 生成速度极慢 | 每步重复计算历史 K/V | 推理阶段必须缓存 |
| 多 beam/batch 时 OOM | 显存突然翻倍 | cache 随 batch、beam、T 同时增长 | 控制 batch/beam，启用 GQA 或量化 |

还有一个常被忽视的点：KV Cache 是“线性增长”，但实际部署里它会被 `batch size x beam size x layer count x context length` 同时放大。所以单样本能跑，不代表批量服务也能跑。

---

## 替代方案与适用边界

如果目标是高效推理，现代主流选择通常是 Pre-RMSNorm + RoPE + GQA，而不是早期标准 Transformer 配置。

先看对比：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| Post-Norm + 经典 MHA | 结构直观，论文和教材多 | 深层稳定性较弱，长上下文推理成本高 | 教学、实验复现 |
| Pre-Norm + MHA | 训练更稳 | KV 头数仍多，缓存大 | 中等规模训练 |
| Pre-RMSNorm + RoPE + GQA | 推理友好，长上下文更省显存 | 实现细节更多 | 现代在线推理 |
| Pre-RMSNorm + RoPE + KV 量化 | 显存占用更低 | 可能损失精度，需校准 | 超长上下文服务 |

GQA 是 Grouped Query Attention，白话解释是“很多 Query 头共享较少的 KV 头”。也就是 $H_q > H_{kv}$。这样 Key/Value 头数下降，KV Cache 也跟着下降。

如果原本是标准多头注意力：

$$
H_{kv}=H_q
$$

改成 GQA 后：

$$
H_{kv}=\frac{H_q}{g}
$$

其中 $g$ 是分组倍数。于是缓存开销近似减少到原来的 $1/g$。

例如：

| 设置 | Query 头数 | KV 头数 | KV 显存趋势 |
| --- | --- | --- | --- |
| 标准 MHA | 32 | 32 | 基线 |
| GQA-4 | 32 | 8 | 约降到 1/4 |
| GQA-8 | 32 | 4 | 约降到 1/8 |

KV 量化则是另一条路线。思路不是减少头数，而是减少每个缓存元素占用的字节数，比如从 FP16 压到 INT8 或更低。简化理解是：

1. 对每层 K/V 做分块量化。
2. 存储低比特整数值和缩放因子。
3. 注意力计算前再反量化，或在低精度下直接计算。

它的优点是显存立刻下降，缺点是实现复杂，且过度量化会损伤长上下文质量。

因此边界可以总结成：

- 如果你在做教学、论文复现、小模型实验，标准 Transformer 结构更容易理解。
- 如果你在做真实推理服务，优先考虑 Pre-RMSNorm + RoPE + KV Cache。
- 如果你在做超长上下文服务，GQA 和 KV 量化通常不是“优化项”，而是“能不能跑起来”的必要项。

---

## 参考资料

1. LLaMA 组件拆解，主要涵盖 RMSNorm、Pre-Norm、RoPE、SwiGLU 的结构细节，适合对应本文“核心机制与推导”章节：mbrenndoerfer, *LLaMA Components: RMSNorm, SwiGLU, RoPE*  
2. Transformer FLOP 分布分析，主要用于支撑“FFN 约 67%，QKV 投影约 25%，score 矩阵约 8%”这一工程量级判断，对应本文“工程权衡与常见坑”章节：artfintel, *Where Do LLMs Spend Their FLOPs?*  
3. KV Cache 内存公式、LLaMA-2-7B 的每 token 显存估算与上下文长度示例，主要对应本文“代码实现”与“工程权衡与常见坑”章节：SystemOverflow, *What Is KV Cache and Why Does It Dominate Memory in LLM Inference?*  
4. 长上下文推理中的 KV Cache 优化、GQA 和 KV 量化的工程视角，主要对应本文“替代方案与适用边界”章节：InsiderLLM, *KV Cache Optimization Guide*  
5. 如果需要回到原始注意力定义与经典结构，可补充阅读 Vaswani et al., *Attention Is All You Need*，主要覆盖注意力基本公式与标准 Transformer 框架。

## 核心结论

Transformer 的参数量可以先抓住一个近似主公式：

$$
P \approx 12Ld^2 + Vd
$$

其中，$L$ 是层数，$d$ 是模型维度，$V$ 是词表大小。白话说法是：模型每一层主要由一堆 $d \times d$ 的矩阵组成，而词表里的每个词还要存一份长度为 $d$ 的向量。

这个公式为什么常用？因为在标准 Decoder-only Transformer 里，绝大多数参数都集中在两类地方：

| 模块 | 主要形状 | 参数量级 |
|---|---:|---:|
| Token Embedding | $V \times d$ | $Vd$ |
| Self-Attention | 4 个 $d \times d$ 投影 | $4d^2$ |
| FFN/MLP | $d \to 4d \to d$ | $8d^2$ |
| LayerNorm | 两组长度为 $d$ 的缩放/偏移 | 约 $2d$ |

所以单层主干约为：

$$
4d^2 + 8d^2 = 12d^2
$$

堆叠 $L$ 层后，就是 $12Ld^2$。再加上词嵌入 $Vd$，就得到最常见的估算式。对大模型来说，真正决定规模的通常不是词表，而是层数和宽度，尤其是宽度 $d$，因为它按平方增长。

---

## 问题定义与边界

这里讨论的是“模型权重参数量”，不是训练显存，也不是优化器状态。参数量指模型里需要学习的数字个数，也就是所有可训练矩阵和向量里元素的总数。

本文默认边界如下：

| 统计对象 | 是否计入 | 说明 |
|---|---|---|
| Token Embedding | 计入 | 每个词一个 $d$ 维向量 |
| Attention 的 Q/K/V/O 投影 | 计入 | 四个线性层 |
| FFN 两层全连接 | 计入 | 通常是 $d \to 4d \to d$ |
| LayerNorm | 计入 | 但量级通常很小 |
| Bias | 可选 | 对总量影响通常较小 |
| 位置编码 | 视实现而定 | RoPE 这类通常几乎不加参数 |
| 优化器状态 | 不计入 | 这是训练资源，不是模型参数 |
| KV Cache | 不计入 | 这是推理时缓存，不是参数 |

几个核心术语先说清楚。

“词表大小” $V$：就是模型能直接查到多少个 token。可以把它理解成一本词典里有多少张“词卡片”。

“模型维度” $d$：就是每个 token 内部表示的长度。可以把它理解成每张词卡片有多少个数字。

“权重共享”是指输入 embedding 和输出 unembedding 共用同一套矩阵。白话说法是：查词典和从隐藏状态映回词表，用的是同一本词典，而不是两本。

对新手来说，一个最实用的观察是：Transformer 可以看成 $L$ 个重复堆叠的函数块。每个函数块里有 Attention、FFN、LayerNorm；模型最前面再加一个大词表矩阵。参数量估算，本质上就是把这些块逐项相加。

---

## 核心机制与推导

先看一层 Transformer。

### 1. Attention 为什么是 $4d^2$

标准多头注意力虽然分了很多头，但如果总模型维度还是 $d$，那么 Q、K、V、O 四个投影矩阵的总规模并不会因为“头多”而凭空变成更高阶，通常仍然可以写成：

$$
W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}
$$

所以这一部分参数量是：

$$
4d^2
$$

很多初学者容易漏掉 $W_O$，只算 Q/K/V 三个矩阵，于是少算了四分之一。

### 2. FFN 为什么是 $8d^2$

标准 FFN 是两层线性变换：

$$
d \to 4d \to d
$$

因此参数量为：

$$
d \times 4d + 4d \times d = 8d^2
$$

这里的“扩张 4 倍”意思是：先把表示维度拉宽，再压回去，让模型有更强的非线性表达能力。白话说法是：先把信息摊开处理，再收回来。

### 3. LayerNorm 为什么通常可忽略

每个 LayerNorm 只有长度为 $d$ 的缩放和偏移参数，约为 $2d$。即使每层有两个 LayerNorm，总量也只是约 $4d$，相比 $12d^2$ 小很多。因此做一阶估算时通常忽略不计。

### 4. 更完整的写法

JAX Scaling Book 常见写法可以写成：

$$
P = L(3DF + 4DNH + D) + 2DV
$$

这里：

- $D$ 是模型维度，也就是本文的 $d$
- $F$ 是 FFN 中间层宽度，标准设定常取 $4D$
- $N \times H$ 表示头数乘单头宽度，通常约等于 $D$

代入标准设定 $F=4D$、$NH=D$，得到：

$$
P \approx L(3D\cdot 4D + 4D\cdot D + D) + 2DV
$$

即：

$$
P \approx L(12D^2 + 4D^2 + D) + 2DV
$$

不同资料对分组方式略有差异，但化简后主导项仍是：

$$
P \approx 12LD^2 + 2DV
$$

如果输入 embedding 与输出 unembedding 共享，就把两个词表矩阵合并成一个，于是常用简化式就是：

$$
P \approx 12Ld^2 + Vd
$$

### 5. 玩具例子

取一个小模型：

- $L=2$
- $d=128$
- $V=5000$

层内主干参数：

$$
12Ld^2 = 12 \times 2 \times 128^2 = 393{,}216
$$

Embedding 参数：

$$
Vd = 5000 \times 128 = 640{,}000
$$

总参数约为：

$$
393{,}216 + 640{,}000 = 1{,}033{,}216
$$

这里 embedding 占比约为：

$$
\frac{640000}{1033216} \approx 62\%
$$

这个例子说明：在小模型里，词表矩阵可能比层内主干还大，所以不能机械认为“Transformer 参数全在 Attention 和 MLP”。

### 6. 手算一个常见配置

取 BERT-base 量级的形状来做直观估算：

- $L=12$
- $d=768$
- $V=50{,}000$

单层主干：

$$
12 \times 768^2 = 7{,}077{,}888 \approx 7.1M
$$

12 层主干：

$$
12 \times 7.1M \approx 85M
$$

Embedding：

$$
50{,}000 \times 768 = 38.4M
$$

总参数约为：

$$
85M + 38.4M = 123.4M
$$

这已经非常接近很多实际基础模型的量级。

---

## 代码实现

下面给一个可运行的 Python 脚本。它按模块拆解参数，支持是否共享 embedding、是否统计 LayerNorm 与 bias。

```python
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    num_layers: int
    d_model: int
    vocab_size: int
    ffn_multiplier: int = 4
    shared_embed: bool = True
    include_layernorm: bool = True
    include_bias: bool = False
    ln_per_layer: int = 2  # pre-LN/post-LN 常见为每层两个 LN

def count_transformer_params(cfg: TransformerConfig):
    L = cfg.num_layers
    d = cfg.d_model
    V = cfg.vocab_size
    f = cfg.ffn_multiplier * d

    # Attention: Q, K, V, O 四个 d x d 投影
    attention_per_layer = 4 * d * d

    # FFN: d -> f -> d
    ffn_per_layer = d * f + f * d

    # LayerNorm: gamma + beta
    layernorm_per_layer = cfg.ln_per_layer * 2 * d if cfg.include_layernorm else 0

    # Bias: Attention 4 个偏置 + FFN 2 个偏置
    bias_per_layer = (4 * d + f + d) if cfg.include_bias else 0

    per_layer = attention_per_layer + ffn_per_layer + layernorm_per_layer + bias_per_layer
    backbone = L * per_layer

    embedding = V * d
    unembedding = 0 if cfg.shared_embed else V * d

    total = backbone + embedding + unembedding

    return {
        "attention_per_layer": attention_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "layernorm_per_layer": layernorm_per_layer,
        "bias_per_layer": bias_per_layer,
        "per_layer_total": per_layer,
        "backbone_total": backbone,
        "embedding": embedding,
        "unembedding": unembedding,
        "total": total,
    }

# 玩具例子
toy = TransformerConfig(num_layers=2, d_model=128, vocab_size=5000)
toy_stats = count_transformer_params(toy)
assert toy_stats["backbone_total"] == 393216
assert toy_stats["embedding"] == 640000
assert toy_stats["total"] == 1033728  # 含 LayerNorm，略大于简化公式

# 接近 12Ld^2 + Vd 的估算
bert_like = TransformerConfig(
    num_layers=12,
    d_model=768,
    vocab_size=50000,
    include_layernorm=False,
    include_bias=False,
)
bert_stats = count_transformer_params(bert_like)
approx = 12 * 12 * 768 * 768 + 50000 * 768
assert bert_stats["total"] == approx

print("toy:", toy_stats)
print("bert_like:", bert_stats)
```

这段代码有两个要点。

第一，它把“近似公式”和“逐项统计”分开了。简化公式适合手算，逐项统计适合工程核对。

第二，它故意保留了 `include_layernorm` 和 `shared_embed` 这样的开关，因为真实模型实现经常就在这些地方和教科书公式不完全一致。

真实工程里，一个常见做法是先用近似式快速估量规模，再用框架实际打印出的参数总量做校验。如果两者差距超过几个百分点，通常就是漏算了 unembedding、LayerNorm、bias，或者 FFN 宽度并不是标准的 $4d$。

---

## 工程权衡与常见坑

参数量公式最大的价值，不是背下来，而是帮助你判断“扩参到底在放大什么”。

| 改动方向 | 参数变化 | 直观含义 |
|---|---|---|
| 增加层数 $L$ | 线性增长 | 模型更深 |
| 增加宽度 $d$ | 近似平方增长 | 模型更宽，最贵 |
| 增加词表 $V$ | 线性增长 | 词典更大 |

这意味着：如果把 $d$ 从 4096 提到 8192，主干参数不是翻一倍，而是接近翻四倍。工程上这通常比加同等比例的层数更昂贵。

常见坑主要有四个。

| 漏算项 | 典型错误 | 后果 | 规避方法 |
|---|---|---|---|
| O 投影 | 只算 Q/K/V | Attention 少算 25% | 固定按 Q/K/V/O 四项拆 |
| Unembedding | 忘记输出词表矩阵 | 总量偏低 | 先确认是否权重共享 |
| LayerNorm/bias | 全部忽略 | 大模型下累计误差 | 用脚本做精确统计 |
| FFN 宽度 | 默认都按 4d | 遇到 SwiGLU 等会失真 | 查具体实现的中间维度 |

真实工程例子可以看 GPT-3 175B。公开资料中常见配置近似为：

- $L=96$
- $d=12288$
- 词表约 5 万级

先看主干项：

$$
12Ld^2 \approx 12 \times 96 \times 12288^2
$$

这个量级已经是上千亿。再看 embedding：

$$
Vd \approx 50{,}000 \times 12{,}288 \approx 6.1 \times 10^8
$$

如果不共享输入输出词表，则约再乘 2，来到十亿量级以上。无论按哪种实现口径，结论都一样：175B 这种规模里，主干的 Attention 和 FFN 才是绝对主体，embedding 只是边缘项。也正因为这样，大模型讨论“扩参”时，核心关注点几乎总在层数、宽度和 FFN 结构，而不是先去扩大词表。

---

## 替代方案与适用边界

最常用的简化式是：

$$
P \approx 12Ld^2 + Vd
$$

它适用于：

- 标准 Transformer block
- FFN 中间层约为 $4d$
- 输入输出 embedding 共享
- 忽略 LayerNorm 与 bias

如果不共享输入输出 embedding，更合适的写法是：

$$
P \approx 12Ld^2 + 2Vd
$$

如果你需要更精细一点，可以写成：

$$
P \approx L(4d^2 + 2df + \text{LN/bias}) + \text{embedding terms}
$$

这样一来，遇到非标准 FFN，比如 $f \neq 4d$，就能直接替换。

不同规模下，主导项也不同：

| 配置 | 主干 $12Ld^2$ | Embedding $Vd$ | 谁占主导 |
|---|---:|---:|---|
| $L=2,d=128,V=5000$ | 393,216 | 640,000 | Embedding |
| $L=12,d=768,V=50k$ | 84,934,656 | 38,400,000 | 主干 |
| $L=96,d=12288,V=50k$ | 约 174B | 约 0.61B | 主干极强 |

所以适用边界很明确。

小模型阶段，尤其是 $L \le 4, d \le 256$ 时，embedding 可能占大头。这个阶段如果词表特别大，压缩词表、共享 embedding、做子词切分优化，都可能有效。

大模型阶段，embedding 的线性项会迅速被 $12Ld^2$ 压过去。这时真正影响参数规模和训练成本的，是宽度、层数，以及 FFN 是否采用更大的中间层或门控结构。

---

## 参考资料

1. JAX Scaling Book, “Transformers” 与 “How To Scale Your Model”  
   https://jax-ml.github.io/scaling-book/transformers/  
   用于完整公式 $L(3DF+4DNH+D)+2DV$ 的推导与近似化。

2. AndoLogs, “Estimating Transformer Model Properties”  
   https://blog.ando.ai/posts/ai-transformer-sizes/  
   用于分模块说明 Embedding、Attention、MLP、LayerNorm 的参数贡献。

3. NVIDIA Developer Blog, “OpenAI Presents GPT-3, a 175 Billion Parameters Language Model”  
   https://developer.nvidia.com/blog/openai-presents-gpt-3-a-175-billion-parameters-language-model/  
   用于 GPT-3 175B 的工程量级案例与参数分布背景。

4. KDnuggets, “Why Depth Is Useful in Self-Attention”  
   https://www.kdnuggets.com/2020/07/depth-useful-self-attention.html  
   用于辅助理解层数与宽度对 Transformer 表达能力和规模的影响。

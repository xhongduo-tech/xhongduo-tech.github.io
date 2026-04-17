## 核心结论

计算 Transformer FLOP 时，先统一口径。FLOP 是“浮点运算次数”，这里按工程里最常见的规则处理：一次矩阵乘加记作 2 次运算。

本文只看单层 Decoder Block 的前向传播，并使用下面的符号：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $B$ | batch size | 一次并行处理多少条样本 |
| $N$ | sequence length | 一条样本里有多少个 token |
| $H$ | attention heads | 注意力被切成多少个头并行算 |
| $d_h$ | head dimension | 每个头内部的向量宽度 |
| $D=H\cdot d_h$ | model dimension | token 表示的总宽度 |

在这个定义下，单层 Decoder 的前向主项是：

$$
F_{\text{attn}} = 8BND^2 + 4BN^2D
$$

$$
F_{\text{ffn}} = 16BND^2
$$

所以总 FLOP 为：

$$
F_{\text{layer}} = 24BND^2 + 4BN^2D
$$

这比“直接背一个 $O(N^2D)$”更有用，因为它把两个来源分开了：

1. $BND^2$ 来自线性投影和 FFN，和序列长度线性相关。
2. $BN^2D$ 来自注意力矩阵，和序列长度平方相关，这才是自注意力里 $O(N^2)$ 的根源。

一个常见误区是把“单层约 $24BND^2$”误当成完整答案。严格说，$24BND^2$ 只是把注意力里的二次项 $4BN^2D$ 先忽略后的主项近似；当上下文变长时，这个二次项不能省。

---

## 问题定义与边界

本文讨论的是“标准 Decoder-only Transformer 单层前向 FLOP”，例如 GPT、LLaMA 这一类结构的一个 block。注意边界很重要，因为不同论文、博客、框架的口径经常不一样。

本文统计：

- Q、K、V 三个线性投影
- 注意力分数 $QK^\top$
- 注意力权重乘 $V$
- 输出投影
- FFN 的两次线性层

本文不统计：

- embedding 查表
- LayerNorm
- bias 加法
- dropout
- softmax、mask、RoPE 等逐元素操作
- 残差连接
- 采样阶段的 logits softmax

原因很简单：如果目标是估算主要算力消耗，矩阵乘法是绝对主项，其他部分通常不改变数量级。

注意力的定义是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

这里 $QK^\top$ 的输出形状是 $[B,H,N,N]$。这句话可以直接翻译成白话：序列里的每个 token 都要和其余 token 两两打分，所以会形成一个 $N\times N$ 的矩阵，这就是二次复杂度出现的位置。

玩具例子可以先看最小版。假设一句话只有 4 个 token，那么注意力分数矩阵就是 $4\times 4$。如果长度翻倍到 8，矩阵就变成 $8\times 8$，元素个数从 16 变成 64，不是翻倍，而是变成 4 倍。

推理时还要分成两个阶段：

| 阶段 | 发生什么 | 复杂度特点 |
| --- | --- | --- |
| Prefill | 把整段提示词一次性送进模型 | 注意力包含完整 $N\times N$ 计算 |
| Decode | 每次只生成 1 个新 token | 若有 KV Cache，新增 token 只和历史缓存交互，注意力增量变成线性 |

KV Cache 的白话解释是：把历史 token 的 K 和 V 先存起来，后面生成新 token 时直接复用，不要每一步把旧 token 的 K、V 再算一遍。

---

## 核心机制与推导

下面按组件逐项推导。先记住一个基础规则：矩阵乘法 $(m\times k)\cdot(k\times n)$ 的 FLOP 近似是 $2mkn$。

### 1. QKV 投影

输入 $X$ 形状为 $[B,N,D]$，权重矩阵 $W_Q,W_K,W_V$ 都是 $[D,D]$。

单个投影 FLOP：

$$
2BND^2
$$

三个投影总和：

$$
F_{QKV}=6BND^2
$$

白话理解：每个 token 都要被映射三次，分别变成 query、key、value。

### 2. Attention Score

按多头展开后，$Q$ 和 $K$ 的形状是 $[B,H,N,d_h]$。  
每个头里要做：

$$
[N,d_h]\cdot[d_h,N]\rightarrow[N,N]
$$

每个头的 FLOP 是 $2N^2d_h$，全部 batch 和头加起来：

$$
F_{\text{score}} = 2BHN^2d_h = 2BN^2D
$$

这一步就是“每个 token 和所有 token 做点积”。

### 3. Attention Weights 乘 V

注意力权重形状为 $[B,H,N,N]$，$V$ 形状为 $[B,H,N,d_h]$。  
矩阵乘法是：

$$
[N,N]\cdot[N,d_h]\rightarrow[N,d_h]
$$

总 FLOP：

$$
F_{\text{AV}} = 2BHN^2d_h = 2BN^2D
$$

### 4. 输出投影

多头拼回去以后再乘一个 $[D,D]$ 的输出矩阵：

$$
F_{\text{out}} = 2BND^2
$$

### 5. FFN

标准 Transformer 的 FFN 通常先把维度从 $D$ 扩到 $4D$，再降回 $D$。  
两次线性层分别是：

$$
2BND\cdot 4D = 8BND^2
$$

$$
2BN(4D)\cdot D = 8BND^2
$$

所以：

$$
F_{\text{ffn}} = 16BND^2
$$

### 6. 合并结果

把注意力四项加起来：

$$
F_{\text{attn}} = 6BND^2 + 2BN^2D + 2BN^2D + 2BND^2
= 8BND^2 + 4BN^2D
$$

再加 FFN：

$$
F_{\text{layer}} = 24BND^2 + 4BN^2D
$$

可以整理成一张表：

| 组件 | FLOP | 主要增长项 | 备注 |
| --- | --- | --- | --- |
| QKV 投影 | $6BND^2$ | $D^2$ | 三个线性层 |
| Attention Score | $2BN^2D$ | $N^2$ | $QK^\top$ |
| 权重乘 $V$ | $2BN^2D$ | $N^2$ | 上下文聚合 |
| 输出投影 | $2BND^2$ | $D^2$ | 多头拼回后线性层 |
| FFN | $16BND^2$ | $D^2$ | 4 倍扩展的两层 MLP |
| 总计 | $24BND^2+4BN^2D$ | 两者都有 | 长上下文时二次项更显著 |

玩具例子：设 $B=1,N=4,D=8$。  
则单层 FLOP 为：

$$
24\times1\times4\times8^2 + 4\times1\times4^2\times8 = 6144 + 512 = 6656
$$

这个例子很小，但能看清结构：线性层和 FFN 提供主干，注意力里的 $N^2$ 项在序列变长时会迅速放大。

真实工程例子：设一个常见配置 $B=1,N=4096,D=4096$。  
单层前向约为：

$$
24\cdot 1\cdot 4096\cdot 4096^2 + 4\cdot 1\cdot 4096^2\cdot 4096
\approx 1.92\times 10^{12}
$$

如果模型有 32 层，仅 prefill 的整网前向就约为：

$$
32\times 1.92\times 10^{12} \approx 6.15\times10^{13}
$$

这就是为什么长提示词的首 token 延迟通常很高。

---

## 代码实现

下面给一个可运行的 Python 版本。它把 prefill 和 decode with KV cache 分开算，并带 `assert` 做最基本的自检。

```python
from dataclasses import dataclass

@dataclass
class BlockFlops:
    qkv: int
    score: int
    attn_v: int
    out_proj: int
    ffn: int

    @property
    def total(self) -> int:
        return self.qkv + self.score + self.attn_v + self.out_proj + self.ffn


def decoder_block_flops(batch: int, seq_len: int, d_model: int, ffn_mult: int = 4) -> BlockFlops:
    # Q, K, V 三个投影
    qkv = 3 * 2 * batch * seq_len * d_model * d_model

    # QK^T
    score = 2 * batch * seq_len * seq_len * d_model

    # attention_weights @ V
    attn_v = 2 * batch * seq_len * seq_len * d_model

    # 输出投影
    out_proj = 2 * batch * seq_len * d_model * d_model

    # FFN: D -> 4D -> D
    hidden = ffn_mult * d_model
    ffn = 2 * batch * seq_len * d_model * hidden + 2 * batch * seq_len * hidden * d_model

    return BlockFlops(qkv, score, attn_v, out_proj, ffn)


def decode_with_kv_cache_flops(batch: int, prefix_len: int, new_tokens: int, d_model: int, ffn_mult: int = 4) -> int:
    # 每一步只处理 1 个新 token，但它要与 prefix_len + i 个历史 token 交互
    total = 0
    hidden = ffn_mult * d_model

    for i in range(new_tokens):
        past = prefix_len + i

        qkv = 3 * 2 * batch * 1 * d_model * d_model
        score = 2 * batch * 1 * past * d_model
        attn_v = 2 * batch * 1 * past * d_model
        out_proj = 2 * batch * 1 * d_model * d_model
        ffn = 2 * batch * 1 * d_model * hidden + 2 * batch * 1 * hidden * d_model

        total += qkv + score + attn_v + out_proj + ffn

    return total


# 玩具例子
toy = decoder_block_flops(batch=1, seq_len=4, d_model=8)
assert toy.total == 6656

# 一个真实一点的例子
prefill = decoder_block_flops(batch=1, seq_len=1024, d_model=4096).total
decode = decode_with_kv_cache_flops(batch=1, prefix_len=1024, new_tokens=100, d_model=4096)

assert prefill > 0
assert decode > 0
assert decode < prefill * 100  # 有 cache 时，100 步 decode 不会退化成每步重算完整 1024x1024 注意力

print(f"prefill FLOPs: {prefill:.3e}")
print(f"decode FLOPs for 100 new tokens: {decode:.3e}")
print(f"total: {prefill + decode:.3e}")
```

这段代码有两个用途：

1. 用来检查公式有没有写错。
2. 用来做容量估算，比如“上下文从 4K 提到 8K，首 token 延迟大约涨多少”。

如果把 `prefix_len=1024, new_tokens=100` 代进去，直觉上也很好理解：

- prefill 是一次性算完整的 $1024\times1024$ 注意力。
- decode 是 100 次增量计算，第 1 步大约看 1024 个历史 token，第 100 步大约看 1123 个历史 token。
- 所以 decode 总体更像“线性累加”，不是每一步都重跑一个完整平方矩阵。

---

## 工程权衡与常见坑

KV Cache 确实把解码阶段的注意力增量从“重复二次重算”压成了线性读取，但它不是免费午餐。

| 问题 | 影响 | 规避方式 |
| --- | --- | --- |
| 没开 KV Cache | 每步重复算历史 K/V，延迟明显升高 | 推理默认开启 `use_cache=True` |
| Cache 太大 | 占显存，压缩 batch 或上下文长度 | 用量化、offload、分页缓存 |
| Cache miss / eviction | 被驱逐的历史片段需要重算 | 设计足够大的缓存池，提高命中率 |
| 过度量化 | 省内存但可能伤精度或伤延迟 | 长上下文再量化，短上下文优先原生缓存 |
| Static cache 设太大 | 可编译但会浪费 masked token 计算 | 场景稳定时再用，别无脑开最大值 |

这里最容易误解的一点是：KV Cache 降低的是“随历史长度增长的注意力重算部分”，不是把整层成本都变成 $O(N)$。  
对单个新 token 而言，带 cache 的单层 FLOP 是：

$$
F_{\text{decode-step}} = 24BD^2 + 4BND
$$

其中：

- $24BD^2$ 是 QKV、输出投影、FFN 的固定项
- $4BND$ 是当前 query 和历史 K/V 的交互项

所以更准确的说法是：KV Cache 把解码阶段里“原本会重复出现的 $N^2$ 注意力部分”降成了“每步对历史长度线性增长的 $N$ 项”。

真实工程例子：聊天模型生成 1000 个 token。  
如果不开 cache，第 800 步时你还在重算前 799 个 token 的 K/V；如果开 cache，第 800 步只需要算当前 token 的 Q/K/V，然后让当前 Q 去读前 799 个 token 已缓存的 K/V。两者延迟曲线完全不是一个量级。

---

## 替代方案与适用边界

KV Cache 不是只有“开”或“关”两种状态，工程上常见的是几种缓存策略。

| 策略 | 显存占用 | 延迟 | `torch.compile` 友好性 | 适用场景 |
| --- | --- | --- | --- | --- |
| Dynamic Cache | 中 | 通常较好 | 否 | 默认选择，通用场景 |
| Static Cache | 高 | 稳定，适合编译优化 | 是 | 请求长度分布比较稳定 |
| Quantized Cache | 低 | 可能略有额外开销 | 否 | 显存紧张、长上下文 |
| Offloaded Cache | 很低 GPU 显存 | 通常更慢 | 依实现而定 | 小显存 GPU，先保不 OOM |

还可以进一步配合结构改造：

- MQA/GQA：让多个 query 头共享更少的 K/V 头，直接减少 KV Cache 体积。
- Sliding Window Attention：只保留最近一段上下文，主动截断 $N$。
- KV Cache 量化：例如 FP8、4-bit，把缓存带宽和容量压下去。
- Prefix cache / prompt cache：把公共前缀预填充一次，多请求复用。

这些方法的边界也要说清：

1. 如果上下文很短，量化 cache 可能不划算，因为省下的显存不多，反而多了量化和反量化成本。
2. 如果请求长度差异很大，Static Cache 容易造成浪费，因为短请求也得背着大缓存槽位。
3. 如果模型已经用了滑窗注意力，cache 长度不会无限增长，FLOP 曲线也会和全注意力不同。
4. 如果你在做训练而不是推理，KV Cache 通常不该打开，因为训练需要完整反向图。

---

## 参考资料

1. daiwk，《Transformer的FLOPS和访存带宽》  
   https://www.daiwk.net/1.1.pre_llm

2. Hugging Face Transformers 文档，《Caching》  
   https://huggingface.co/docs/transformers/cache_explanation

3. Hugging Face Transformers 文档，《Cache strategies》  
   https://huggingface.co/docs/transformers/kv_cache

4. NVIDIA Technical Blog，《Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache》，2025-12-08  
   https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/

5. Stanford NLP / Christopher Manning 相关讲义，《Large Language Models in 2025: How》  
   https://www-nlp.stanford.edu/~manning/talks/WWK-Understanding-and-Intelligence-2025.pdf

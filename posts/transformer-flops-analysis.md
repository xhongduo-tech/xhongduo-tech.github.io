## 核心结论

单层 Transformer 的前向计算，主 FLOP 可以拆成四部分：QKV 投影、注意力核心、输出投影、FFN。这里的 FLOP 指浮点运算次数，也就是乘法和加法的近似总量。

在不细分 softmax、LayerNorm、偏置、残差这些常数项时，常见估算式是：

$$
\text{FLOPs}_{\text{layer}} \approx 24nd^2 + 4n^2d
$$

其中：

- $n$ 是序列长度，也就是一段输入里 token 的个数
- $d$ 是模型维度，也就是每个 token 表示向量的宽度

这个公式说明了两件事：

1. QKV 投影、输出投影、FFN 都是 $O(nd^2)$，它们随序列长度线性增长，但会被 $d^2$ 放大。
2. 注意力核心是 $O(n^2d)$，它会随序列长度平方增长，所以长上下文时会迅速变贵。

因此，短序列场景下，FFN 往往是计算主力；长序列场景下，注意力核心的 $4n^2d$ 会越来越显著，最终压过前面的线性项。

一个直接判断条件是比较 $n$ 和 $d$ 的量级。若 $n \ll d$，则 $nd^2$ 项更大；若 $n$ 增长到接近甚至超过 $d$，则 $n^2d$ 项开始主导。

---

## 问题定义与边界

本文讨论的是**单层 Transformer 的一次前向传播**。前向传播就是“输入一批 token，算出下一层表示”的过程，不包含反向传播和参数更新。

边界刻意收窄到最核心的矩阵乘法，因为矩阵乘法通常决定大头计算量。我们只分析：

| 组件 | 输入到输出 | FLOP 近似式 | 复杂度类别 |
|---|---|---:|---|
| QKV 投影 | $(n,d)\to(n,3d)$ | $3\times 2nd^2$ | $O(nd^2)$ |
| 注意力核心 | $QK^\top,\ \text{Attn}\cdot V$ | $4n^2d$ | $O(n^2d)$ |
| 输出投影 | $(n,d)\to(n,d)$ | $2nd^2$ | $O(nd^2)$ |
| FFN | $(n,d)\to(n,4d)\to(n,d)$ | $16nd^2$ | $O(nd^2)$ |

这里默认：

- 多头注意力按总维度 $d$ 合并估算，不单独展开每个 head
- FFN 扩展倍数取 4，也就是隐藏层宽度是 $4d$
- 一个乘加记作约 2 FLOP，这是深度学习里常见的估算法
- softmax、激活函数、LayerNorm、残差连接忽略不计，因为通常不是主项

玩具例子可以先想成这样：一层 Transformer 不神秘，本质就是“先把输入线性变成 Q、K、V，再做两次大矩阵乘法，最后再过一段更宽的两层全连接”。

---

## 核心机制与推导

先看 QKV 投影。输入张量形状是 $(n,d)$，每个 token 都要乘三个 $d\times d$ 的权重矩阵，分别得到 Query、Key、Value。Query 是“拿什么去问”，Key 是“我有哪些特征可被匹配”，Value 是“真正被聚合的内容”。

单个线性层的 FLOP 近似是：

$$
2nd^2
$$

因为输出里有 $n\times d$ 个位置，每个位置都要做长度为 $d$ 的点积。Q、K、V 三个投影合起来：

$$
\text{FLOPs}_{QKV}=3\times 2nd^2 = 6nd^2
$$

再看注意力核心。先算分数矩阵：

$$
QK^\top \in \mathbb{R}^{n\times n}
$$

这是把每个 token 和所有 token 做一次匹配，得到“谁该关注谁”的分数。这里的 FLOP 是：

$$
2n^2d
$$

然后把注意力权重乘上 $V$，得到输出：

$$
\text{Attn}(Q,K,V)V
$$

这又是一次 $n\times n$ 和 $n\times d$ 的乘法，FLOP 也是：

$$
2n^2d
$$

所以注意力核心一共：

$$
\text{FLOPs}_{\text{attn-core}} = 4n^2d
$$

输出投影再来一次 $(n,d)\times(d,d)$：

$$
\text{FLOPs}_{\text{out}} = 2nd^2
$$

FFN 是两层线性层。第一层把 $d$ 扩到 $4d$，第二层再缩回 $d$。FFN 可以理解成“对每个 token 单独做一次更强的特征变换”，它不混 token，只混通道。

第一层：

$$
2n\cdot d\cdot 4d = 8nd^2
$$

第二层：

$$
2n\cdot 4d\cdot d = 8nd^2
$$

合起来：

$$
\text{FLOPs}_{FFN} = 16nd^2
$$

于是总 FLOP：

$$
\text{FLOPs}_{\text{layer}} = 6nd^2 + 4n^2d + 2nd^2 + 16nd^2
= 24nd^2 + 4n^2d
$$

玩具例子：取 $n=1,d=2$。这时注意力核心的平方项几乎看不出来，因为只有一个 token，根本没有“和别的 token 两两比较”的成本。代入后：

$$
24nd^2 + 4n^2d = 24\times1\times4 + 4\times1\times2 = 104
$$

这个例子说明：当序列极短时，主要在做线性投影和 FFN，而不是在做“全局两两匹配”。

再看更接近真实的数值：$n=128,d=1024$。

- QKV 投影：$6nd^2 \approx 8.05\times10^8$
- 注意力核心：$4n^2d \approx 6.71\times10^7$
- 输出投影：$2nd^2 \approx 2.68\times10^8$
- FFN：$16nd^2 \approx 2.15\times10^9$

这时 FFN 明显最大。也就是说，很多人一看到 Transformer 就只盯着 $n^2$，在短上下文下会误判主耗时位置。

---

## 代码实现

下面给一个可运行的 Python 函数，直接计算各部分 FLOP 和占比。

```python
def transformer_layer_flops(n: int, d: int, ffn_ratio: int = 4):
    assert n > 0 and d > 0 and ffn_ratio > 0

    qkv = 3 * 2 * n * d * d
    attn_core = 2 * n * n * d + 2 * n * n * d
    out_proj = 2 * n * d * d
    ffn = 2 * n * d * (ffn_ratio * d) + 2 * n * (ffn_ratio * d) * d

    total = qkv + attn_core + out_proj + ffn

    parts = {
        "qkv": qkv,
        "attn_core": attn_core,
        "out_proj": out_proj,
        "ffn": ffn,
        "total": total,
        "ratio": {
            "qkv": qkv / total,
            "attn_core": attn_core / total,
            "out_proj": out_proj / total,
            "ffn": ffn / total,
        },
    }
    return parts


# 玩具例子
toy = transformer_layer_flops(n=1, d=2)
assert toy["total"] == 104

# 真实工程里常见的短上下文例子
case_128 = transformer_layer_flops(n=128, d=1024)
assert case_128["ffn"] > case_128["attn_core"]

# 长上下文例子
case_32k = transformer_layer_flops(n=32768, d=4096)
assert case_32k["attn_core"] > case_32k["ffn"]

print(case_128["total"])
print(case_32k["ratio"])
```

如果传入 `n=128, d=1024`，会得到一个非常典型的结论：FFN 最大，注意力核心反而不大。若传入 `n=32768, d=4096`，注意力核心会明显超过 FFN。

用表格看两个场景更直观：

| 场景 | $n$ | $d$ | 主导项 |
|---|---:|---:|---|
| 玩具例子 | 1 | 2 | 线性项 |
| 短上下文小批推理 | 128 | 1024 | FFN |
| 长上下文检索/对话 | 32768 | 4096 | 注意力核心 |

真实工程例子：长文问答、代码仓库检索、32K/64K 上下文聊天模型，都会让 $4n^2d$ 快速膨胀。此时用户感知到的是“同样模型参数，窗口一拉长，延迟和显存突然失控”。

顺便补一个训练侧的粗估经验。若把模型总非嵌入参数记为 $N$，训练 token 数记为 $D$，常见经验是：

$$
C_{\text{train}} \approx 6ND
$$

它不是逐层精确计数，而是训练总计算量的工程近似，适合先估预算，再看是否需要细化到层级 FLOP。

---

## 工程权衡与常见坑

第一类权衡是短序列和长序列的优化方向不同。

| 场景 | 更该优先优化什么 | 原因 |
|---|---|---|
| $n \ll d$ | FFN、投影层 | 主项是 $nd^2$ |
| $n$ 很长 | 注意力实现 | 主项转向 $n^2d$ |
| 两者都大 | 两边都要动 | 线性项和平方项都贵 |

常见坑有四个。

第一，只盯着 $n^2d$。这会漏掉 QKV、输出投影、FFN。结果是在 128、256、512 token 这类常见上下文里，错误地把优化重点全放在 attention 上。

第二，把 QKV 投影当成“顺手做一下的小开销”。其实它和输出投影一样，本质都是大矩阵乘法，量级是 $nd^2$，并不轻。

第三，忽略 FFN 扩展倍数。FFN 通常用 $4d$，一改成更宽，FLOP 会直接线性抬高。很多结构修改看似只是在“加一点中间层宽度”，实际是在放大主耗时项。

第四，把训练 FLOP 和推理单层 FLOP 混为一谈。前者常用 $6ND$ 做总量粗估，后者是逐层逐模块拆解，两者用途不同。

一个直观对比：

- 72 token 的分类或短问答任务，注意力平方项很小，FFN 更值得先看。
- 32K token 的长文对话，平方项会暴涨，FlashAttention 之类的优化立刻变成刚需。

---

## 替代方案与适用边界

当注意力项成为瓶颈时，目标不是“让 Transformer 失去注意力”，而是“降低标准全连接注意力的代价”。

| 方案 | 主要压低哪一项 | 适用场景 |
|---|---|---|
| FlashAttention | 降显存访存开销，改进 attention 实现效率 | 长上下文训练与推理 |
| 局部/滑动窗口注意力 | 把全局 $n^2$ 改成局部依赖 | 文档、时序、局部相关强 |
| 稀疏注意力 | 减少两两比较次数 | 超长序列 |
| 降低 FFN ratio | 压低 $16nd^2$ | 短序列、小模型 |
| 降低模型宽度 $d$ | 同时压低 $nd^2$ 与 $n^2d$ | 预算受限的小模型 |

适用边界也要说清。

FlashAttention 并没有把理论复杂度从 $O(n^2d)$ 变成线性，它主要是通过更好的内存访问和分块计算，减少中间矩阵落地，让同样的 attention 更快、更省显存。所以它非常适合“理论结构不变，但工程实现要提速”的场景。

局部注意力、稀疏注意力则会改变注意力图本身。代价是模型不再看完整的全局两两关系，收益是复杂度下降。这类方法更像“换算法”，不是“换实现”。

如果你的模型主要跑 512 token 以下的小任务，先砍 FFN ratio、减小 $d$，通常比折腾复杂注意力变体更直接。反过来，如果要做 32K 甚至 128K 上下文，attention 才是必须正面处理的那一项。

---

## 参考资料

1. Michael Brenndoerfer, *Attention Complexity: Quadratic Scaling, Memory Limits & Efficient Alternatives*  
   用途：自注意力 $O(n^2d)$、组件级 FLOP 拆解与长上下文瓶颈说明。  
   链接：https://mbrenndoerfer.com/writing/attention-complexity-quadratic-scaling-memory-efficient-transformers

2. Adam Casson, *Transformer FLOPs*  
   用途：训练 FLOP 的工程估算、`6N` 每 token 与 `6ND` 总训练计算量的解释。  
   链接：https://www.adamcasson.com/posts/transformer-flops  
   PDF：https://www.adamcasson.com/transformer-flops.pdf

3. Jared Kaplan et al., *Scaling Laws for Neural Language Models*, 2020  
   用途：训练计算量与参数、数据规模关系的原始来源。  
   链接：https://arxiv.org/abs/2001.08361  
   OpenAI 页面：https://openai.com/index/scaling-laws-for-neural-language-models/

4. Technical University Berlin thesis / lecture materials on Transformer forward-pass FLOPs  
   用途：前向传播中 QKV、attention、FFN 的矩阵乘法展开视角。  
   链接：https://d-wetzel.de/documents/thesis.pdf

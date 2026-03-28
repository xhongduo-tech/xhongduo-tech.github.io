## 核心结论

因果掩码（causal mask）是 Transformer 做自回归生成时的基本约束。自回归的意思是“当前位置只能用过去，不能偷看未来”。它的直接实现方式很简单：先算注意力分数矩阵 $S=\frac{QK^\top}{\sqrt{d}}$，再把上三角位置加上 $-\infty$，这样 softmax 之后这些位置的概率就变成 0。

数学上可写成：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
$$

其中掩码矩阵 $M$ 满足：

$$
M_{ij}=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

结论有三个层次。

| 层次 | 做法 | 显存代价 | 适用场景 |
|---|---|---:|---|
| 朴素实现 | 显式构造 $(n,n)$ 掩码矩阵 | $O(n^2)$ | 教学、小序列、调试 |
| 内核融合实现 | 在 FlashAttention kernel 内按块处理掩码 | 近似 $O(n)$ 额外显存 | 长上下文训练/推理 |
| 结构化稀疏实现 | 因果约束再叠加局部窗口、块跳过 | 依实现而定，通常优于全矩阵 | Mistral 类长上下文模型 |

玩具例子先看长度为 4 的序列。掩码前后效果如下：

| 查询位置 | softmax 前可见键 | 被屏蔽键 | softmax 后非零权重位置 |
|---|---|---|---|
| 0 | 0 | 1,2,3 | 0 |
| 1 | 0,1 | 2,3 | 0,1 |
| 2 | 0,1,2 | 3 | 0,1,2 |
| 3 | 0,1,2,3 | 无 | 0,1,2,3 |

所以，第三个 token 只能看到前三个键，未来 token 的注意力权重严格为 0。这不是“模型学出来的习惯”，而是计算图层面直接禁止。

---

## 问题定义与边界

问题定义很具体：在注意力机制里，如何阻止位置 $i$ 访问位置 $j>i$ 的信息，同时又不让实现成本在长序列下失控。

这里有两个容易混淆的边界。

第一，因果掩码主要服务于自回归任务，比如 next-token prediction。它不适合双向编码器。双向编码器的意思是“一个位置允许同时看左边和右边”，像 BERT 这类模型如果套上因果掩码，任务定义就变了。

第二，三角掩码只能保证“不能看未来”，但不能天然保证“不同样本之间互相隔离”。这在 packed sequence 中尤其关键。packed sequence 的意思是“把多个不同长度样本拼进一个更紧凑的 batch，减少 padding 浪费”。

例如把两个句子拼成一行：

- 句子 A：`[A1, A2, A3]`
- 句子 B：`[B1, B2]`

拼接后得到：`[A1, A2, A3, B1, B2]`

如果只加普通因果掩码，那么 `B1` 可以看到 `A1,A2,A3`，因为这些位置都在它左边。这在数学上满足“因果”，但在数据语义上已经泄露了跨样本信息。因此 packed sequence 需要 block mask 或 attention bias 额外标记“跨样本不可见”。

边界判断可以压缩成下面这个表：

| 场景 | 是否需要因果掩码 | 还需额外约束吗 | 说明 |
|---|---|---|---|
| GPT 训练 | 需要 | 常常不需要 | 单样本自回归，标准场景 |
| GPT 单 token 解码 | 通常可省略显式 mask | 不需要 | KV cache 已天然限制只能看历史 |
| Encoder 双向注意力 | 不需要 | 不适用 | 目标就是双向建模 |
| Packed sequence 训练 | 需要 | 需要 block mask/bias | 仅三角掩码会跨样本泄露 |
| Sliding window attention | 需要 | 需要窗口约束 | 同时限制未来和过远历史 |

真实工程里，边界比公式更重要。很多“模型看到了不该看的内容”不是因为 softmax 写错，而是因为样本布局、cache 形状、位置偏移或 packed 策略处理错了。

---

## 核心机制与推导

先看基本推导。注意力分数矩阵是：

$$
S = \frac{QK^\top}{\sqrt{d}}
$$

其中 $Q$ 是查询，表示“当前位置想找什么”；$K$ 是键，表示“每个位置能提供什么匹配信号”；$V$ 是值，表示“真正要聚合的内容”。对初学者可以这样理解：$QK^\top$ 算的是“我该关注谁”，softmax 后再拿这些权重去加权 $V$。

加入因果掩码后：

$$
P_{ij}=\frac{\exp(S_{ij}+M_{ij})}{\sum_k \exp(S_{ik}+M_{ik})}
$$

因为 $\exp(-\infty)=0$，所以所有未来位置直接从概率分布里消失。注意这里不是“概率很小”，而是严格为 0。

再往前一步，为什么朴素实现贵？因为它通常会显式生成一个 $(n,n)$ 的矩阵。若序列长度是 $n=32768$，元素个数约为：

$$
32768^2 \approx 1.07\times 10^9
$$

如果按 FP32 存，每个元素 4 字节，仅一个全尺寸矩阵就接近 4GB。即使布尔掩码更省，也仍然是 $O(n^2)$ 存储和访存。这还没算注意力分数、中间 softmax 统计量和多头展开。

FlashAttention 的关键改动不是“换了一个更快的 softmax”，而是把“算分数、做掩码、求 softmax、乘 V”融合到同一个按块执行的 kernel 里。按块的意思是：不是一次把整张 $n \times n$ 的大表摊开，而是把 Q、K、V 切成小 tile，逐块搬到片上存储中计算。

对因果掩码来说，块级视角特别有价值：

1. 如果一个块完全在下三角区域，那么这个块内全部可见，不需要逐元素判断。
2. 如果一个块完全在上三角区域，那么这个块整体跳过，连算都不用算。
3. 只有穿过对角线的块，才需要做细粒度 mask。

可以用一个简化流程表示：

```text
遍历 Q-block
  遍历 K-block
    如果 K-block 完全在未来区域: skip
    如果 K-block 完全在历史区域: 正常算分数
    如果 K-block 穿过对角线: 对无效位置填 -inf
  对当前 Q-block 做在线 softmax
  输出当前块结果
```

这里“在线 softmax”指的是不必先把整行所有分数都存下来再统一 softmax，而是边读块边维护当前最大值和归一化和。这样中间激活不需要完整落到显存，访存压力明显下降。

再看 sliding window。若窗口大小为 $w$，则第 $i$ 个查询只允许访问区间 $[i-w+1, i]$。掩码变成：

$$
M_{ij}=
\begin{cases}
0, & i-w+1 \le j \le i \\
-\infty, & \text{otherwise}
\end{cases}
$$

这其实是“因果约束 + 局部窗口约束”的叠加。Mistral 的思路就是把可见区域从完整下三角，收缩成“贴着右下角的一条带状区域”。它仍然因果，但不是全历史可见，而是局部历史可见。

---

## 代码实现

先给一个最小可运行的 Python 例子，演示因果掩码在数值上如何把未来位置压成 0。这里不依赖深度学习框架，只用标准库完成 softmax。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def causal_mask_scores(scores):
    # scores: [seq_len, seq_len]
    seq_len = len(scores)
    masked = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if j > i:
                row.append(float("-inf"))
            else:
                row.append(scores[i][j])
        masked.append(row)
    return masked

scores = [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 3.0, 2.0, 0.0],
    [2.0, 1.0, 4.0, 8.0],
    [0.0, 1.0, 2.0, 3.0],
]

masked = causal_mask_scores(scores)
probs = [softmax(row) for row in masked]

# 第 0 行只能看自己
assert abs(probs[0][0] - 1.0) < 1e-9
assert probs[0][1] == 0.0
assert probs[0][2] == 0.0
assert probs[0][3] == 0.0

# 第 2 行不能看未来位置 3
assert probs[2][3] == 0.0

# 每一行仍然是合法概率分布
for row in probs:
    assert abs(sum(row) - 1.0) < 1e-9

print(probs)
```

这个例子对应前面的玩具场景：长度 4，第三个 token 只能关注前 3 个位置。

如果用常见深度学习框架，朴素实现大致是这样：

```python
# 伪代码
scores = (Q @ K.transpose(-2, -1)) / sqrt(d_head)
mask = tril(ones(seq_len, seq_len))          # 下三角为 1
scores = scores.masked_fill(mask == 0, -inf)
probs = softmax(scores, dim=-1)
out = probs @ V
```

这个版本好理解，但问题也很明确：`mask` 和 `scores` 都是全尺寸矩阵。

更接近真实工程的接口，往往不会直接传一个完整 `mask`，而是传“掩码类型”或“结构参数”。例如 FlashAttention 风格接口可写成：

```python
# 伪代码
out = flash_attention(
    q,                  # [B, Tq, H, Dh]
    k,                  # [B, Tk, H, Dh]
    v,                  # [B, Tk, H, Dh]
    causal=True,        # 启用因果约束
    window_left=None,   # 若为整数，表示局部窗口左边界
    window_right=0,     # 因果场景通常右边界为 0，禁止看未来
    attn_bias=None,     # packed sequence 时用于跨样本隔离
    kv_cache=None       # 推理阶段可复用历史 K/V
)
```

参数含义可以概括为：

| 参数 | 含义 | 典型用途 |
|---|---|---|
| `causal` | 是否启用因果约束 | 自回归训练/推理 |
| `window_left` | 最远能看到多早的历史 | sliding window |
| `window_right` | 是否允许向未来看 | 因果时一般为 0 |
| `attn_bias` | 额外加到分数上的偏置 | packed sequence、块隔离 |
| `kv_cache` | 缓存历史键值 | 单 token 解码 |

一些实现还会用 4-vector 编码掩码边界，例如 `(upper, lower, left, right)`。它的思想不是存整张矩阵，而是存“这个样本的可见区域边界”。如果可见区域能用少量边界描述，就能把掩码表达从二维压缩成一维或分块结构。

真实工程例子看 LLM 推理。预填充阶段（prefill）要一次处理整段 prompt，比如 8K token，这时必须施加因果约束，因为同一批次中每个位置都在一起算。到了单 token 解码阶段，当前步只有一个新查询 token，K/V 来自缓存的历史。此时从数据结构上就只有“历史 K/V + 当前 Q”，未来 token 尚未进入 cache，所以通常可以不再显式构造三角掩码。这也是很多推理框架把 prefill 和 decode 分成两套 kernel 的原因。

---

## 工程权衡与常见坑

因果掩码的难点不在定义，而在实现细节与资源约束。

| 问题 | 表现 | 应对方案 |
|---|---|---|
| 全矩阵掩码太大 | 长序列显存爆炸，速度也慢 | 用 FlashAttention 类 kernel，避免实化掩码 |
| packed sequence 泄露 | 一个样本看见另一个样本尾部 | 用 attention bias / block mask，而不是仅三角掩码 |
| sliding window 偏移错 | 可见区错一位，生成质量异常 | 明确窗口对齐规则，统一左闭右闭定义 |
| decode 仍构造全 mask | 推理多做无用工作 | 单 token + KV cache 阶段省略显式 mask |
| `-inf` 数值处理不当 | 半精度下出现 NaN 或错误概率 | 用框架推荐的 masked softmax 或 kernel 内融合逻辑 |
| 块边界处理错误 | 对角附近注意力错乱 | 分清“整块可见”“整块跳过”“对角细分”三类路径 |

这里重点讲三个坑。

第一个坑是把“理论上正确”当成“工程上可接受”。朴素三角掩码的逻辑完全正确，但序列一长就不现实。对 32K 上下文，光是相关中间张量就可能压垮显存，更不用说多头、多层、反向传播。

第二个坑是 packed sequence。很多人以为“只要因果，就不会串样本”，这是错的。因果只限制时间方向，不限制样本边界。只拼接不加 block mask，本质上等于偷偷把多个训练样本合并成一个长样本，标签虽然还能算，但训练语义已经变了。

第三个坑是把 sliding window 当成因果掩码的替代品。准确说，sliding window 不是替代，而是叠加。它解决的是“历史太长，没必要全看”的问题；因果解决的是“不能看未来”的问题。两者约束维度不同。

工程里常见的一条经验是：先问清楚你要优化的是哪一层成本。

- 如果瓶颈是显存，优先避免全尺寸 score 和 mask 落地。
- 如果瓶颈是带宽，优先做 kernel 融合和块跳过。
- 如果瓶颈是 batch 利用率，优先解决 packed sequence 与变长调度。
- 如果瓶颈是长上下文质量，优先评估 sliding window 是否伤害召回范围。

---

## 替代方案与适用边界

因果掩码不是唯一手段，但它是最基础、最稳妥的一种。替代方案通常是在“表达方式”或“计算路径”上不同，而不是放弃因果约束本身。

先看几种常见方案对比：

| 方案 | 核心思路 | 优点 | 局限 |
|---|---|---|---|
| 朴素因果掩码 | 直接构造上三角 $-\infty$ | 简单直观 | $O(n^2)$ 显存/访存 |
| Sliding window + causal | 仅保留 $[i-w+1,i]$ | 长上下文更省 | 看不到更早历史 |
| Incremental cache | 每步只算当前查询对历史 KV | 解码高效 | 只适合增量推理阶段 |
| Attention bias / block mask | 用偏置表达跨样本或结构约束 | 适合 packed 与复杂边界 | 实现更复杂 |
| FlashMask/块级掩码 | 结构化编码 + kernel 跳过空块 | 长序列效率高 | 依赖专门内核与接口 |

Mistral 的 sliding-window attention 是很好的真实工程例子。假设窗口大小为 $w$，那么第 $i$ 个位置不是看完整历史 $[0,i]$，而是只看最近的 $[i-w+1, i]$。如果超出窗口的历史被认为边际价值不高，这种方式能显著降低注意力成本。它本质上是把下三角可见区裁成一条斜向带状区域。

再看 incremental cache。它在推理阶段非常重要，但边界很明确：它并不替代训练时的因果约束，也不替代 prefill 阶段对整段 prompt 的掩码处理。它只是利用“未来 token 还不存在”这一事实，让解码阶段天然满足因果性。

如果任务里存在变长拼接、检索块拼接、多段上下文隔离，attention bias 往往比简单 mask 更通用。因为 bias 可以表达“同样在左边，但不允许看”的关系，比如跨样本、跨段落、跨文档隔离。这类需求用普通三角矩阵并不好表达。

因此适用边界可以总结成一句话：简单单样本训练用标准因果掩码，长上下文高性能实现用 FlashAttention 类动态掩码，局部建模用 sliding window，变长拼接和跨样本隔离用 attention bias 或 block mask，单 token 解码则优先依赖 KV cache。

---

## 参考资料

| 来源 | 核心内容 | 用途 |
|---|---|---|
| FlashAttention 相关技术文章 | 在线 softmax、块级 K-loop、减少显存访存 | 解释为何可在 kernel 内处理因果掩码 |
| NVIDIA 关于 FlashAttention 调优文章 | tile 划分、kernel 访存与吞吐优化 | 说明块级跳过与硬件友好实现 |
| FlashMask / FlashMaskedAttention 资料 | 用结构化边界或向量压缩 mask 表达 | 说明如何避免存全矩阵掩码 |
| Mistral sliding-window 相关文章 | 局部窗口与因果约束叠加 | 说明带状可见区的工程实现 |
| Transformer 基础教程与从零实现文章 | 标准 attention 公式与三角掩码示意 | 适合作为入门直观材料 |

1. FlashAttention 系列论文与工程解读：重点看在线 softmax、块级计算、为何无需完整落地 $QK^\top$。
2. NVIDIA 关于 FlashAttention kernel/tile 调优的技术博客：重点看对角块、完整块、跳过块如何在 CUDA 实现中分流。
3. FlashMask/FlashMaskedAttention 相关资料：重点看如何用边界向量或块结构表达复杂掩码。
4. Mistral sliding-window attention 的工程说明：重点看局部窗口与因果约束的组合。
5. Transformer 教程类材料：重点看标准因果掩码数值例子，适合理解 softmax 前后变化。

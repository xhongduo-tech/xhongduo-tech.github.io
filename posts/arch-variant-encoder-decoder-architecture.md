## 核心结论

Encoder-Decoder 是一种把“理解输入”和“生成输出”拆开的模型架构。Encoder 先把整个输入序列编码成上下文表示，Decoder 再按从左到右的顺序生成目标序列，并在每一步通过 cross-attention 读取 Encoder 的输出。

这里的 cross-attention 可以先用一句白话理解：Decoder 写下一个词之前，会回头查看“输入里哪些位置最相关”。它和普通 self-attention 的区别在于，查询向量 $Q$ 来自 Decoder 当前状态，而键值向量 $K,V$ 来自 Encoder 输出，而不是都来自同一段序列。

这类架构最适合典型 Seq2Seq 任务。Seq2Seq 的白话解释是“输入一段序列，输出另一段序列”，比如机器翻译、摘要、问答、代码修复说明、代码 diff 生成。它的优势不是“万能更强”，而是“分工更清晰”：

| 组件 | 主要职责 | 信息范围 | 典型收益 |
|---|---|---|---|
| Encoder | 全量理解输入 | 双向看到整个 source | 更擅长压缩和提炼输入语义 |
| Decoder | 逐步生成输出 | 只看已生成 target + Encoder 输出 | 输出更可控，适合短输出任务 |
| Cross-Attention | 桥接输入与输出 | 用 target 查询 source | 生成时按需读取输入证据 |

如果任务是“长输入、短输出”，Encoder-Decoder 通常比纯 Decoder-only 更有结构优势。原因很直接：输入只需要被深度编码一次，Decoder 负责短序列生成，不必让整个模型在每个生成步都重新处理完整输入语义。

但它也不是没有代价。Decoder 每层都多了一个 cross-attention 子层，带来额外参数、额外显存和额外延迟；推理时还要缓存 Encoder 输出，内存开销和输入长度成正比。所以在通用大模型时代，很多翻译、摘要任务正在被 Decoder-only 模型接管；但在短文摘要 API、专用生成接口、代码 diff 这类“输入理解很重、输出很短、格式很稳”的场景里，Encoder-Decoder 仍然很有工程价值。

---

## 问题定义与边界

要理解 Encoder-Decoder，先要定义它解决的问题：输入和输出不是同一段文本的续写，而是“基于输入生成另一段结果”。这和纯续写任务不同。

例如：

- 翻译：输入英文句子，输出法文句子
- 摘要：输入一篇长文，输出几句摘要
- 问答：输入上下文和问题，输出答案
- 代码 diff：输入旧代码与修改意图，输出补丁或变更说明

这些任务的共同点是：source 和 target 语义相关，但 token 序列不同步。模型既要完整理解 source，又要按目标格式生成 target。

一个简单边界判断是看输入输出长度关系：

| 场景 | 输入长度 | 输出长度 | Encoder-Decoder 适配度 |
|---|---:|---:|---|
| 长文摘要 | 长 | 短 | 高 |
| 文档问答 | 长 | 短 | 高 |
| 机器翻译 | 中长 | 中长 | 高 |
| 故事续写 | 短 | 长 | 低到中 |
| 开放式聊天 | 变动大 | 变动大 | 通常不优先 |

为什么“长输入、短输出”特别适合它？因为 cross-attention 的核心计算规模大致是：

$$
O(m \times n)
$$

其中：

- $n$ 是输入长度，也就是 Encoder 输出位置数
- $m$ 是输出长度，也就是 Decoder 当前生成的目标位置数

如果 source 长度是 64，target 长度是 16，那么单层 cross-attention 需要处理 $16 \times 64 = 1024$ 个位置相似度。这个量不算小，但有一个关键收益：Encoder 侧的表示只算一次，后续生成 16 步都复用。

可以把缓存路径写成一个极简图：

$$
\text{source} \xrightarrow{\text{Encoder}} H_{\text{enc}} \xrightarrow{\text{cache}} \text{all decode steps}
$$

对应的生成路径是：

$$
y_{<t} \xrightarrow{\text{Decoder self-attn}} s_t \xrightarrow{\text{cross-attn over }H_{\text{enc}}} y_t
$$

这说明它的边界也很明确：

1. 如果输入很短、输出很长，Encoder 的一次性编码优势会减弱。
2. 如果任务本质就是“接着写”，那单独拆出 Encoder 往往没有必要。
3. 如果输入长到几万 token，Encoder 输出缓存本身就可能变成显存瓶颈。

一个玩具例子可以直观看到这个边界。

输入是一封 6 段英文邮件，输出是 2 句摘要。此时最合理的做法是：Encoder 先完整读完邮件，形成一份上下文记忆；Decoder 在写第 1 句摘要和第 2 句摘要时，分别去这份记忆里取相关部分。这正是“先理解，再生成”。

如果把同一任务硬改成 Decoder-only，也能做，但通常会变成“把整封邮件和已经生成的摘要片段都塞进统一上下文，再继续预测下一个 token”。这在统一架构上更简单，但在任务结构上不如 Encoder-Decoder 清晰。

---

## 核心机制与推导

Transformer 里的 attention 可以先用一句白话记住：每个位置都会问一句“我现在应该关注谁”。Cross-attention 只是把“问的人”和“被问的人”放在了两段不同序列上。

设：

- Encoder 输出为 $X_{\text{enc}} \in \mathbb{R}^{n \times d}$
- Decoder 当前层输入为 $X_{\text{dec}} \in \mathbb{R}^{m \times d}$

那么 cross-attention 的三组投影是：

$$
Q = X_{\text{dec}}W^Q,\quad
K = X_{\text{enc}}W^K,\quad
V = X_{\text{enc}}W^V
$$

其中：

- Query，查询向量，白话就是“我现在想找什么信息”
- Key，键向量，白话就是“每个输入位置提供什么索引标签”
- Value，值向量，白话就是“真正被读取出来的内容”

然后进入标准的 scaled dot-product attention：

$$
\text{CrossAttention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里的逻辑顺序不能弄反：

1. 用 $QK^\top$ 计算 Decoder 每个位置对 Encoder 各位置的相关性分数
2. 除以 $\sqrt{d_k}$ 做缩放，避免数值过大
3. 经过 softmax 变成权重分布，每一行和为 1
4. 用这些权重对 $V$ 加权求和，得到“从输入中读出的结果”

### 玩具例子：从英文到法文的局部对齐

假设 source 是：

`The quick brown fox`

target 当前已经生成到：

`Le renard`

可以把它看成两个 Decoder 位置分别在问：

- `Le` 应该主要看 source 的哪些词
- `renard` 应该主要看 source 的哪些词

一个简化的 attention 权重矩阵可能是：

| Decoder token | The | quick | brown | fox |
|---|---:|---:|---:|---:|
| Le | 0.70 | 0.10 | 0.10 | 0.10 |
| renard | 0.05 | 0.05 | 0.10 | 0.80 |

每一行和为 1。含义很直接：

- 生成 `Le` 时，模型主要参考 `The`
- 生成 `renard` 时，模型主要参考 `fox`

真实模型里不会这么“翻译词典式”整齐，因为 attention 学到的是语义相关性，不一定是一一对齐；但这个玩具例子足够说明 cross-attention 的作用：Decoder 每个位置都在对输入做一次有条件检索。

### 多头机制为什么必要

多头注意力可以白话理解为“让模型同时从多个角度看输入”。一个 head 可能更关注位置对齐，另一个 head 更关注语义类别，还有的 head 更关注长距离依赖。

单头形式是：

$$
\text{head}_i
=
\text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
$$

多头聚合后是：

$$
\text{MultiHead}(Q,K,V)
=
\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

这也解释了为什么 Decoder 的每层 cross-attention 会额外带来较多参数。除了 $W^Q, W^K, W^V$ 之外，还有输出投影 $W^O$。在隐藏维度为 $d$ 的近似设定下，这部分额外参数量通常可以粗略看成约 $4d^2$ 量级。

### 为什么它在长输入短输出任务上更自然

因为 Encoder 负责对 source 做深度双向建模。双向的白话解释是“一个位置可以同时看左边和右边上下文”。所以输入语义在进入 Decoder 前已经被压缩成更稳定的表示。

而 Decoder 仍然保持自回归约束，即只能看见已经生成的 target 前缀。这保证了训练和推理的一致性。于是整个结构形成明确分工：

- Encoder：负责读懂
- Decoder：负责写出
- Cross-attention：负责把“读懂的内容”按需喂给“正在写的步骤”

这就是它和 Decoder-only 最大的结构差异。后者把“读”和“写”压到同一堆层里统一完成，前者则显式拆分。

---

## 代码实现

下面用一个最小可运行的 Python 例子实现单头 cross-attention，不依赖深度学习框架，只用 `numpy` 展示形状和计算顺序。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def cross_attention(decoder_state, encoder_output, Wq, Wk, Wv):
    """
    decoder_state: (m, d)
    encoder_output: (n, d)
    Wq, Wk, Wv: (d, dk)
    """
    Q = decoder_state @ Wq      # (m, dk)
    K = encoder_output @ Wk     # (n, dk)
    V = encoder_output @ Wv     # (n, dk)

    scores = (Q @ K.T) / np.sqrt(K.shape[-1])   # (m, n)
    weights = softmax(scores, axis=-1)          # (m, n)
    output = weights @ V                        # (m, dk)
    return output, weights

# 一个玩具例子：4 个 source token，2 个 target token
np.random.seed(0)
m, n, d, dk = 2, 4, 6, 4

decoder_state = np.random.randn(m, d)
encoder_output = np.random.randn(n, d)

Wq = np.random.randn(d, dk)
Wk = np.random.randn(d, dk)
Wv = np.random.randn(d, dk)

output, weights = cross_attention(decoder_state, encoder_output, Wq, Wk, Wv)

assert output.shape == (m, dk)
assert weights.shape == (m, n)
assert np.allclose(weights.sum(axis=1), np.ones(m))

print("output shape:", output.shape)
print("weights shape:", weights.shape)
print("attention row sums:", weights.sum(axis=1))
print("weights:\n", weights)
```

这段代码对应的关键计算顺序就是：

```python
Q = decoder_state @ Wq
K = encoder_output @ Wk
V = encoder_output @ Wv
scores = (Q @ K.T) / np.sqrt(dk)
weights = softmax(scores, axis=-1)
output = weights @ V
```

新手要特别看懂三个维度：

- Decoder 长度是 $m$
- Encoder 长度是 $n$
- 隐藏或头维度是 $d$ 或 $d_k$

因此：

- $Q$ 的形状是 $(m, d_k)$
- $K, V$ 的形状是 $(n, d_k)$
- 权重矩阵的形状是 $(m, n)$

这正好体现“每个输出位置都去查询所有输入位置”。

### 缓存为什么重要

真实推理时，Encoder 输出不会在每一步重新计算，而是先缓存下来。伪代码可以写成：

```python
class CachedCrossAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.cached_K = None
        self.cached_V = None

    def build_encoder_cache(self, encoder_output):
        self.cached_K = encoder_output @ self.Wk
        self.cached_V = encoder_output @ self.Wv

    def step(self, decoder_state_t):
        Q = decoder_state_t @ self.Wq
        scores = (Q @ self.cached_K.T) / np.sqrt(self.cached_K.shape[-1])
        weights = softmax(scores, axis=-1)
        return weights @ self.cached_V
```

这个缓存机制的意义是：

- Encoder 侧 K/V 只算一次
- 每次生成新 token，只需要算新的 $Q$
- 之后直接和缓存的 Encoder K/V 做 attention

### 真实工程例子：短文摘要 API

假设你要做一个企业内部摘要服务：

- 输入：一篇约 1000 token 的故障报告
- 输出：3 句摘要，总长度 60 token 左右

这就是 Encoder-Decoder 的典型舒适区：

1. Encoder 把 1000 token 的报告完整编码
2. Decoder 只生成 60 token 左右
3. 每一步通过 cross-attention 从 1000 个输入位置中读取相关证据

如果换成 Decoder-only，也能做，但通常要把“原文 + 已生成摘要”一直放进统一上下文，随着生成步数增加，上下文组织和推理路径会更重。对于这种“输入理解很重、输出很短”的服务，Encoder-Decoder 往往更容易做成稳定接口。

---

## 工程权衡与常见坑

工程上不要只看“架构是否优雅”，要看成本结构。

### 1. 显存和缓存不是免费的

Encoder-Decoder 的一个直接代价是必须保存 Encoder 输出，常见做法还会缓存其投影后的 K/V。输入越长，这部分缓存越大。

可以把成本和收益并排看：

| 维度 | 收益 | 成本 | 常见应对 |
|---|---|---|---|
| Encoder 一次编码 | 输入只深度处理一次 | 首次编码有固定开销 | 控制 encoder 层数 |
| Cross-attention | 输出可按需读取输入证据 | 每层多一次 attention 计算 | 减少 decoder 层数或宽度 |
| Encoder KV 缓存 | 后续解码可复用 | 内存随输入长度线性增长 | offload、压缩、截断 |
| 结构分工 | 长输入短输出任务更稳 | 架构更复杂，参数更多 | 只在明确任务上采用 |

一个常见误区是以为“Decoder 每步只算一个 token，所以一定很快”。这句话只对一半。确实，Decoder 是自回归逐步生成；但它每一层都要对 Encoder 的所有输入位置做 cross-attention，所以输入很长时，延迟并不会低到可以忽略。

### 2. 参数量通常高于同层数 Decoder-only

Decoder-only 的每层主要有 self-attention 和 FFN。Encoder-Decoder 则至少包含：

- 一整套 Encoder 层
- 一整套 Decoder 层
- Decoder 每层额外一个 cross-attention 子层

因此在相近层数和宽度下，它的总参数通常更大。粗略地说，Decoder 每层多出来的 cross-attention 大致是额外 $4d^2$ 量级参数，而整套 Encoder 也是完整堆叠，这使得总模型尺寸和推理图都更重。

### 3. 长输入时缓存会突然成为瓶颈

看一个简单量级估算。

如果输入长度从 1000 token 增加到 1500 token，而你缓存的是每层 Encoder 输出或其 K/V，那么缓存规模通常也会近似增长 1.5 倍。对于多层模型，这种增长很快会顶到显存上限。

这就是为什么真实部署中常见这些策略：

- 截断输入，只保留最相关段落
- 对 Encoder 输出做压缩或池化
- 把部分 KV offload 到 CPU
- 降低层数或隐藏维度
- 对长文先做检索，再送入 Encoder

### 4. 训练和推理的 mask 容易写错

新手实现里最常见的 bug 不是公式本身，而是 mask。

要区分两种 mask：

- Decoder self-attention 的 causal mask：阻止看到未来 token
- Cross-attention 的 padding mask：阻止关注输入中的 padding 位置

如果把这两种 mask 混淆，模型可能仍能跑通，但效果会明显异常。典型现象是：

- 翻译重复
- 摘要遗漏关键句
- 对 padding 位置产生高 attention
- 训练 loss 下降慢或不稳定

### 5. 不要把 attention 权重直接当成“解释性证明”

很多文章会展示 attention 热图，但要谨慎。attention 权重说明“模型在这一层这一头做了怎样的加权”，不等于“模型最终为什么这么预测”的完整因果解释。它适合做局部分析，不适合直接当作可解释性的充分证据。

---

## 替代方案与适用边界

今天做新系统时，通常不会默认选择 Encoder-Decoder，而是先问：任务是否真的需要“理解输入”和“生成输出”分开建模。

下面给出一个简表：

| 架构 | 输入处理方式 | 输出处理方式 | 适合任务 | 主要短板 |
|---|---|---|---|---|
| Encoder-Decoder | Encoder 双向编码 source | Decoder 自回归生成 target，并 cross-att 输入 | 翻译、摘要、问答、代码 diff | 参数和缓存更重 |
| Decoder-only | 输入和输出放在统一上下文 | 直接续写式生成 | 对话、创作、统一大模型服务 | 长输入短输出任务不一定最省 |
| Prefix-LM | 前缀部分可双向，后缀部分自回归 | 在同一模型中做混合掩码 | 某些条件生成、多任务折中 | 训练与实现更复杂 |

### 什么时候优先选 Encoder-Decoder

1. 输入远长于输出。
2. 任务是标准 Seq2Seq，不是开放式续写。
3. 你希望输入理解和输出生成职责清晰分离。
4. 你需要更稳定的格式化输出，比如摘要、问答、翻译、代码修复说明。

### 什么时候不优先选它

1. 任务本质是自由生成、对话或续写。
2. 你希望统一一套 Decoder-only 基础设施服务所有任务。
3. 输入长度极长，Encoder 缓存代价过高。
4. 你更关注通用能力和生态，而不是某个专用任务的结构最优。

### 一个对比例子

如果任务提示是：

`Write a story continuation from "Once upon a time"`，

这本质是续写。Decoder-only 很自然，因为输入只是前缀，输出就是同一语境下的延续。

如果任务提示是：

“阅读一份 8 页事故复盘，输出 5 条行动项摘要”，

这就更像先理解一大块 source，再生成压缩后的 target。此时 Encoder-Decoder 的结构更贴合任务本身。

因此，结论不是“Encoder-Decoder 过时”或“Decoder-only 全面替代”，而是：

- 通用大模型平台层面，Decoder-only 更统一
- 专用 Seq2Seq 服务层面，Encoder-Decoder 仍然是有效方案

---

## 参考资料

- IBM, “What is an encoder-decoder model?”  
  https://www.ibm.com/think/topics/encoder-decoder-model
- Michael Brenndoerfer, “Cross-Attention in Encoder-Decoder Transformers”  
  https://mbrenndoerfer.com/writing/cross-attention-encoder-decoder-transformers
- Michael Brenndoerfer, “Encoder-Decoder Architecture & Cross-Attention Transformers”  
  https://mbrenndoerfer.com/writing/encoder-decoder-architecture-cross-attention-transformers
- IntlPull, “LLM Translation Quality Benchmark 2026”  
  https://intlpull.com/blog/llm-translation-quality-benchmark-2026
- DeepPaper, “T5 架构回顾”  
  https://deep-paper.org/en/papers/2025-10/t5/
- Harold Benoit, “Encoder-decoder models”  
  https://haroldbenoit.com/notes/ml/llms/architecture/encoder-decoder-models
- NVIDIA, “How to Reduce KV Cache Bottlenecks”  
  https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/

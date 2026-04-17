## 核心结论

Bahdanau 注意力解决的是早期编码器-解码器模型里的固定上下文瓶颈。固定上下文的意思是：无论目标句子生成到第几个词，解码器都只能反复使用同一个压缩向量 $c$。这要求编码器把整句源文本的信息一次性压进一个向量里。句子一长，词序、修饰关系、局部细节就更容易丢失。

Bahdanau 的做法是把这个固定向量换成动态上下文。动态上下文的意思是：每个解码步 $t$ 都重新计算一次“当前最该关注哪些源位置”，于是得到该时刻专属的上下文向量
$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$
其中 $h_i$ 是编码器在第 $i$ 个源位置的隐藏状态，$\alpha_{t,i}$ 是当前解码步对这个位置分配的权重。

这件事的价值有两个。第一，模型不需要把所有信息硬塞进一个向量，而是可以“边生成边回看”源句。第二，权重矩阵 $\alpha$ 可以可视化成热图，帮助我们观察“目标词在生成时主要看了源句哪里”。

先看最核心的对比。

| 方案 | 上下文形式 | 是否每步重算 | 长句表现 | 可解释性 |
| --- | --- | --- | --- | --- |
| 固定上下文 Seq2Seq | 单一 $c$ | 否 | 容易退化 | 弱 |
| Bahdanau Attention | 动态 $c_t$ | 是 | 更稳定 | 可画对齐热图 |

这个差异可以写成更直白的公式对比：

| 模型 | 上下文公式 | 含义 |
| --- | --- | --- |
| 固定上下文 | $c=\text{Encoder}(x_{1:T_x})$ | 整个解码过程只用一个向量 |
| Bahdanau 注意力 | $c_t=\sum_i \alpha_{t,i}h_i$ | 每一步都重新汇总源端信息 |

玩具例子：把法语 `Le chat dort` 翻译成英语 `The cat sleeps`。当解码器准备输出 `cat` 时，注意力权重通常会让 `chat` 对应的位置拿到最大概率，而不是平均看整句。这就是“软对齐”。软对齐的意思是：不是硬性规定一个目标词只能对应一个源词，而是给出一组概率权重。某个位置权重大，表示它更相关；权重小，表示它仍可能有贡献，只是贡献较弱。

如果把这个例子写成一个注意力分布，可以是：

| 当前要生成的目标词 | 源词 `Le` | 源词 `chat` | 源词 `dort` |
| --- | --- | --- | --- |
| `The` | 0.80 | 0.15 | 0.05 |
| `cat` | 0.10 | 0.82 | 0.08 |
| `sleeps` | 0.05 | 0.10 | 0.85 |

这张表表达的不是“严格词典映射”，而是“当前解码步从哪些源位置取信息，取多少”。

---

## 问题定义与边界

问题定义可以写得更具体：给定编码器输出序列 $h_1,\dots,h_{T_x}$，在解码第 $t$ 步时，根据前一时刻解码状态 $s_{t-1}$ 和所有源端状态 $h_i$ 计算一组对齐分数 $e_{t,i}$，再把它们归一化为权重 $\alpha_{t,i}$，最后合成为当前上下文 $c_t$。这个 $c_t$ 不再是全句唯一表示，而是“当前生成任务需要的信息摘要”。

如果写成完整链条，就是：
$$
(s_{t-1}, h_1,\dots,h_{T_x})
\rightarrow e_{t,1:T_x}
\rightarrow \alpha_{t,1:T_x}
\rightarrow c_t
$$

这里有三个问题要分清：

| 问题 | 回答 |
| --- | --- |
| 注意力在算什么 | 当前解码步应该关注哪些源位置 |
| 输出是什么 | 一组权重 $\alpha_{t,i}$ 和一个上下文向量 $c_t$ |
| 它替代了什么 | 固定上下文 Seq2Seq 里的单一向量 $c$ |

边界也要说清楚。Bahdanau 注意力不是“免费增强”，它的计算要在每个目标步扫描所有源位置，所以总复杂度是
$$
O(T_x\times T_y)
$$
其中 $T_x$ 是源序列长度，$T_y$ 是目标序列长度。

如果把隐藏维度和注意力内部维度也算进去，单步打分成本可以近似写成
$$
O\big(T_x(d_a d_s + d_a d_h)\big)
$$
其中：

| 符号 | 含义 |
| --- | --- |
| $d_s$ | 解码器状态维度 |
| $d_h$ | 编码器状态维度 |
| $d_a$ | 注意力内部隐层维度 |
| $T_x$ | 源序列长度 |

也就是说，序列越长，或者隐藏维度越大，计算开销就越明显。

真实工程例子：把英文 `He loves apples` 翻译成法语。生成 `Il` 时，模型主要看 `He`；生成 `aime` 时，模型主要看 `loves`；生成 `les pommes` 时，又需要重新关注 `apples`。如果仍然只依赖一个固定上下文，那么“主语信息”“动词信息”“宾语信息”会混在一起，解码器很难在不同时间步提取不同重点。所以这里“每步都重算对齐”不是额外装饰，而是任务本身的需要。

下面这张表说明它在哪些情况下更有价值，哪些情况下更容易变成开销。

| 条件 | Bahdanau 注意力的收益 | 可能的问题 |
| --- | --- | --- |
| 句子较短 | 可学习对齐，效果稳定 | 额外计算未必显著值得 |
| 句子较长 | 明显优于固定上下文 | 扫描全部源位置更慢 |
| 需要可视化对齐 | 很适合 | 热图容易被误读 |
| RNN 解码器 | 兼容性好 | 并行性弱于 Transformer |

初学者还常混淆两个边界：

| 容易混淆的点 | 实际情况 |
| --- | --- |
| “注意力会不会替代解码器” | 不会。注意力只提供额外上下文，解码器仍负责状态更新和生成 |
| “有了注意力是不是就不丢信息” | 不是。它缓解瓶颈，但不能保证长序列信息永不丢失 |
| “权重最高的位置是不是唯一答案” | 不是。注意力是软分布，不是硬选择 |

---

## 核心机制与推导

Bahdanau 注意力的核心是“加性打分”。加性打分的意思是：先把解码器状态和编码器状态分别做线性变换，再相加后过一个非线性函数，最后投影成一个标量分数。公式是
$$
e_{t,i}=v^\top\tanh(W_1s_{t-1}+W_2h_i)
$$

如果把维度写出来，更容易看清每一步在做什么：
$$
s_{t-1}\in\mathbb{R}^{d_s},\quad
h_i\in\mathbb{R}^{d_h},\quad
W_1\in\mathbb{R}^{d_a\times d_s},\quad
W_2\in\mathbb{R}^{d_a\times d_h},\quad
v\in\mathbb{R}^{d_a}
$$
于是
$$
W_1s_{t-1}\in\mathbb{R}^{d_a},\quad
W_2h_i\in\mathbb{R}^{d_a}
$$
两者可以相加，再经过 $\tanh$，最后被 $v^\top$ 压成一个标量分数。

各项含义如下：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $s_{t-1}$ | 上一步解码器状态 | 当前准备生成下一个词时，解码器脑子里已有的信息 |
| $h_i$ | 编码器第 $i$ 个状态 | 源句第 $i$ 个位置的表示 |
| $W_1,W_2$ | 可学习矩阵 | 把两边投影到同一空间里，方便比较 |
| $v$ | 可学习向量 | 把隐藏表示压成一个分数 |
| $e_{t,i}$ | 对齐分数 | 当前步对第 $i$ 个源位置有多感兴趣 |

这里“加性”的名字来自结构本身：先把两边投影后相加，再经过非线性，而不是直接做点积。它和 dot-product attention 的差别，不在“有没有权重”，而在“分数是怎么计算出来的”。

有了分数后，用 softmax 做归一化：
$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_{j=1}^{T_x}\exp(e_{t,j})}
$$

softmax 的作用是把任意实数分数变成概率分布，也就是：
$$
\alpha_{t,i}\ge 0,\qquad \sum_{i=1}^{T_x}\alpha_{t,i}=1
$$

接着得到上下文向量：
$$
c_t=\sum_{i=1}^{T_x}\alpha_{t,i}h_i
$$

这个流程可以简写成：

| 步骤 | 操作 | 结果 |
| --- | --- | --- |
| 1 | 用 $s_{t-1}$ 和每个 $h_i$ 计算 $e_{t,i}$ | 每个源位置一个原始分数 |
| 2 | 对所有 $e_{t,i}$ 做 softmax，得到 $\alpha_{t,i}$ | 一个概率分布 |
| 3 | 用 $\alpha_{t,i}$ 对 $h_i$ 加权求和 | 当前步上下文 $c_t$ |
| 4 | 把 $c_t$ 与解码器状态、上一输出嵌入一起送入下一步生成 | 参与当前词预测 |

可以把它理解成一条简化流水线：

`投影 -> tanh -> 打分 -> softmax -> 加权求和`

如果想进一步理解“为什么它能训练起来”，关键是整个链条都是可导的。也就是说，损失函数对最终输出的梯度，可以一路传回到：
$$
\text{loss} \rightarrow y_t \rightarrow s_t \rightarrow c_t \rightarrow \alpha_{t,i} \rightarrow e_{t,i} \rightarrow (W_1,W_2,v)
$$
这意味着模型不仅能学“怎么生成词”，还能学“该看哪里”。

看一个最小数值例子。假设某个解码步对三个源位置的分数是：
$$
e=[-0.36,\,0.11,\,-0.40]
$$
softmax 后约为：
$$
\alpha\approx[0.28,\,0.45,\,0.27]
$$

softmax 的计算过程可以显式写成：
$$
\alpha_1=\frac{e^{-0.36}}{e^{-0.36}+e^{0.11}+e^{-0.40}},\quad
\alpha_2=\frac{e^{0.11}}{e^{-0.36}+e^{0.11}+e^{-0.40}},\quad
\alpha_3=\frac{e^{-0.40}}{e^{-0.36}+e^{0.11}+e^{-0.40}}
$$

如果三个源向量分别是
$$
h_1=[1,0],\quad h_2=[0,2],\quad h_3=[1,1]
$$
那么
$$
c_t=0.28[1,0]+0.45[0,2]+0.27[1,1]=[0.55,1.17]
$$

这里第二个位置的权重最大，所以它对当前上下文贡献最大。若把第二个位置理解成源词 `love`，就表示“当前生成这个词时，模型主要在看 `love``”。

再把贡献拆开看一次，更容易直观理解：

| 源位置 | 权重 $\alpha_i$ | 源向量 $h_i$ | 加权后贡献 |
| --- | --- | --- | --- |
| 1 | 0.28 | $[1,0]$ | $[0.28,0.00]$ |
| 2 | 0.45 | $[0,2]$ | $[0.00,0.90]$ |
| 3 | 0.27 | $[1,1]$ | $[0.27,0.27]$ |
| 合计 | 1.00 | - | $[0.55,1.17]$ |

这个机制的重要点不只是“能看回去”，而是误差可以沿着注意力权重回传到相关源位置。也就是说，如果某一步翻译错了，训练过程可以直接调整“应该看哪里”和“看到了之后怎样用”。

---

## 代码实现

下面给一个最小可运行的 Python 实现，只展示单个解码步如何计算 Bahdanau 注意力。代码只依赖标准库，可以直接运行。输入输出维度写在注释里，方便初学者对照公式。

```python
import math


def softmax(xs):
    """Stable softmax for a 1D list."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def matvec(mat, vec):
    """
    mat: [out_dim, in_dim]
    vec: [in_dim]
    return: [out_dim]
    """
    if not mat:
        raise ValueError("mat must not be empty")
    if len(mat[0]) != len(vec):
        raise ValueError(f"shape mismatch: {len(mat[0])} != {len(vec)}")
    return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]


def vec_add(a, b):
    if len(a) != len(b):
        raise ValueError(f"shape mismatch: {len(a)} != {len(b)}")
    return [x + y for x, y in zip(a, b)]


def tanh_vec(v):
    return [math.tanh(x) for x in v]


def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"shape mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def weighted_sum(weights, vectors):
    """
    weights: [Tx]
    vectors: [Tx, dim]
    return: [dim]
    """
    if len(weights) != len(vectors):
        raise ValueError("weights and vectors must have same length")
    if not vectors:
        raise ValueError("vectors must not be empty")

    out_dim = len(vectors[0])
    out = [0.0] * out_dim
    for w, vec in zip(weights, vectors):
        if len(vec) != out_dim:
            raise ValueError("all vectors must have same dim")
        for i, x in enumerate(vec):
            out[i] += w * x
    return out


def bahdanau_step(s_prev, encoder_states, W1, W2, v):
    """
    Compute Bahdanau attention for one decoder step.

    s_prev: [dec_dim]
    encoder_states: [Tx, enc_dim]
    W1: [attn_dim, dec_dim]
    W2: [attn_dim, enc_dim]
    v: [attn_dim]

    returns:
        scores: [Tx]
        alphas: [Tx]
        context: [enc_dim]
    """
    if not encoder_states:
        raise ValueError("encoder_states must not be empty")

    scores = []
    projected_decoder = matvec(W1, s_prev)  # [attn_dim]

    for h_i in encoder_states:
        projected_encoder = matvec(W2, h_i)  # [attn_dim]
        energy_hidden = tanh_vec(vec_add(projected_decoder, projected_encoder))
        e_ti = dot(v, energy_hidden)         # scalar
        scores.append(e_ti)

    alphas = softmax(scores)
    context = weighted_sum(alphas, encoder_states)
    return scores, alphas, context


def pretty(xs, digits=6):
    return [round(x, digits) for x in xs]


if __name__ == "__main__":
    # 一个玩具例子
    # dec_dim = 2, enc_dim = 2, attn_dim = 2, Tx = 3
    s_prev = [0.2, -0.1]

    encoder_states = [
        [1.0, 0.0],
        [0.0, 2.0],
        [1.0, 1.0],
    ]

    W1 = [
        [0.5, -0.3],
        [0.1, 0.4],
    ]

    W2 = [
        [0.2, 0.1],
        [-0.4, 0.3],
    ]

    v = [0.7, -0.2]

    scores, alphas, context = bahdanau_step(
        s_prev=s_prev,
        encoder_states=encoder_states,
        W1=W1,
        W2=W2,
        v=v,
    )

    # 基本正确性检查
    assert len(scores) == 3
    assert len(alphas) == 3
    assert abs(sum(alphas) - 1.0) < 1e-9
    assert all(a >= 0.0 for a in alphas)
    assert len(context) == 2

    print("scores =", pretty(scores))
    print("alphas =", pretty(alphas))
    print("context =", pretty(context))
```

按上面的参数，程序会输出一组合法的分数、权重和上下文。输出的具体数值由参数决定，但应满足三个条件：

| 检查项 | 预期 |
| --- | --- |
| `len(alphas)` | 等于源序列长度 $T_x$ |
| `sum(alphas)` | 约等于 1 |
| `context` 维度 | 等于编码器状态维度 `enc_dim` |

如果想把中间过程也打印出来，便于教学，可以加一版“逐位置展示”：

```python
scores, alphas, context = bahdanau_step(s_prev, encoder_states, W1, W2, v)

for i, (h_i, score, alpha) in enumerate(zip(encoder_states, scores, alphas), start=1):
    print(f"source_pos={i}, h_i={h_i}, score={score:.6f}, alpha={alpha:.6f}")

print("context =", pretty(context))
```

这段输出能帮助新手把“源位置”“分数”“归一化权重”“最终上下文”一一对应起来。

如果把它放回完整的 RNN 解码器，一般会按下面的伪代码组织：

```python
# 输入:
# encoder_states: [Tx, enc_dim]
# s_prev: [dec_dim]
# y_prev_emb: [emb_dim]
#
# 输出:
# y_t_logits: [vocab_size]
# s_t: [dec_dim]

scores, alphas, c_t = bahdanau_step(s_prev, encoder_states, W1, W2, v)

decoder_input = concat(y_prev_emb, c_t)
s_t = rnn_cell(decoder_input, s_prev)

readout = concat(s_t, c_t)
y_t_logits = linear(readout)
```

这里的关键不是代码量，而是接口关系：

| 模块 | 职责 |
| --- | --- |
| `attention` | 从源端提取“这一步要看的信息” |
| `rnn_cell` | 结合历史状态和当前输入更新解码器状态 |
| `linear` | 把状态映射到词表 logits |

很多实现还会把注意力放在状态更新之后，或者把 $c_t$ 同时用于状态更新和输出层。这些都是实现细节变化，但主线不变：先根据当前解码状态估计关注点，再把相关源信息汇总给解码器。

真实工程里通常不会用 Python for-loop 逐元素计算，而是把所有 $h_i$ 堆成矩阵，直接做批量线性变换和批量 softmax。原因很简单：数学上是一样的，但矩阵化实现更快，更适合 GPU。

---

## 工程权衡与常见坑

Bahdanau 注意力在工程上最常见的第一个问题是慢。因为每个目标步都要遍历全部源位置，所以句子一长、batch 一大、hidden size 一高，延迟就会上升。训练时还能依赖矩阵并行，推理时尤其容易暴露问题。

第二个问题是热图误读。热图好看，但它展示的是概率分布，不是硬匹配。某个目标词同时关注两个源词，并不表示模型出错；反过来，某个源词权重最高，也不等于它是唯一决定因素。

下面是常见坑和规避方式。

| 常见坑 | 现象 | 原因 | 规避手段 |
| --- | --- | --- | --- |
| 计算量高 | 长句推理慢 | 每步扫描全部 $T_x$ | 批量矩阵化、限制最大长度、局部窗口 |
| 热图全平均 | 所有位置颜色接近 | 分数区分度不够 | 调整模型容量、检查初始化、减小过强正则 |
| 热图过尖 | 几乎只盯一个位置 | softmax 过度饱和 | 检查学习率、温度缩放、梯度稳定性 |
| padding 干扰 | 模型关注到补齐位 | mask 没做好 | 对 padding 位置加负无穷 mask |
| 误把注意力当解释 | 结论过度绝对化 | $\alpha$ 只是一种中间分布 | 把它当调试信号，不当因果证明 |

padding mask 是实现里最容易漏的点之一。假设一条样本真实长度是 3，但为了组成 batch 被补到长度 5，那么最后两个位置不能参与 softmax。常见做法是对无效位置加一个极小值：

$$
e'_{t,i}=
\begin{cases}
e_{t,i}, & i \text{ 是有效位置} \\
-\infty, & i \text{ 是 padding}
\end{cases}
$$

然后再计算
$$
\alpha_{t,i}=\text{softmax}(e'_{t,i})
$$

这样 padding 位置的权重就会变成 0。

真实工程例子：在英译法系统里，如果你把 `Le chat dort -> The cat sleeps` 的注意力热图画出来，理想情况是生成 `The` 时主要看 `Le`，生成 `cat` 时主要看 `chat`，生成 `sleeps` 时主要看 `dort`。如果热图几乎整行都是均匀颜色，说明模型没有学到有效对齐；如果整列异常集中在某一个源位置，可能表示注意力塌缩，模型在“机械盯住一个词”。

还可以用下面这张表快速诊断：

| 热图现象 | 可能原因 | 先检查什么 |
| --- | --- | --- |
| 整体很平 | 打分差异太小 | 参数初始化、学习率、是否训练充分 |
| 某一列长期很亮 | 注意力塌缩 | 梯度是否爆炸、softmax 是否过饱和 |
| padding 列有颜色 | mask 失效 | 长度张量和 mask 广播是否正确 |
| 训练热图正常，推理热图变乱 | 暴露偏差 | teacher forcing 与自回归分布差异 |

还有一个初学者常漏掉的点：训练和推理的行为差异。训练时通常使用 teacher forcing，也就是把真实上一个目标词喂给解码器；推理时则喂模型自己上一步生成的词。注意力机制本身没变，但输入分布变了，所以推理时热图可能比训练时更散。

这一点可以概括成：

| 阶段 | 解码器输入 | 结果 |
| --- | --- | --- |
| 训练 | 真实上一个目标词 | 状态轨迹更稳定 |
| 推理 | 模型自己上一步输出 | 误差会逐步累积 |

所以不要因为“训练时热图很好看”就直接假设推理质量一定同样稳定。

---

## 替代方案与适用边界

Bahdanau 注意力不是唯一选择。它属于 additive attention，也就是加性注意力。它和 dot-product attention 的差别在于：前者先做投影和非线性，再打分；后者通常直接做向量点积，或做缩放点积。

两者可以这样比较：

| 方案 | 打分方式 | 计算效率 | 适合模型 | 可解释性 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| Bahdanau / Additive | $v^\top\tanh(W_1s + W_2h)$ | 较低 | RNN Seq2Seq | 较强 | 经典机器翻译、对齐可视化 |
| Dot-Product | $q^\top k$ | 较高 | Transformer | 中等 | 大规模并行建模 |
| Scaled Dot-Product | $\frac{q^\top k}{\sqrt{d}}$ | 高 | Transformer | 中等 | 现代主流注意力 |
| Local Attention | 只看局部窗口 | 更高 | 长序列模型 | 较弱 | 长文本、低延迟推理 |

如果把它们的核心区别压缩成一句话：

| 类型 | 分数来自什么 |
| --- | --- |
| Additive Attention | 先变换再相加，再过非线性 |
| Dot-Product Attention | 直接比较两个向量的方向一致性 |
| Scaled Dot-Product | 在点积基础上做尺度修正，避免维度大时数值过大 |

什么时候更适合选 Bahdanau？

1. 你用的是 RNN 编码器-解码器。
2. 你需要明确观察源词和目标词的对齐关系。
3. 输入长度中等，$O(T_x\times T_y)$ 仍可接受。
4. 任务更强调逐步生成和对齐，而不是大规模并行吞吐。

什么时候不该优先选它？

1. 你需要高并行、高吞吐。
2. 你处理的是很长序列。
3. 你的系统已经是标准 Transformer 栈。
4. 你更关心推理速度而不是对齐可视化。

可以用一个简单判断：如果你的问题是“RNN 翻译模型为什么能在不同时间步关注不同源词”，Bahdanau 是标准答案；如果你的问题是“现代大模型为什么能并行高效处理长上下文”，那重点通常已经转向缩放点积注意力及其变体。

还有一个实践层面的边界：Bahdanau 注意力今天更多出现在教学、论文复现、经典 NMT 系统分析里，而不是现代大规模生成模型的主干结构里。这不是因为它“无效”，而是因为工业重点已经转向更强的并行性和更长上下文处理能力。

最后给一个简洁选择表：

| 你的目标 | 更合适的方案 |
| --- | --- |
| 理解早期 NMT 如何做对齐 | Bahdanau 注意力 |
| 在 RNN 上补足固定上下文缺陷 | Bahdanau 注意力 |
| 构建现代高吞吐大模型 | Scaled Dot-Product Attention |
| 长序列低延迟推理 | Local / Sparse / Windowed Attention |

---

## 参考资料

1. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473, 2014; ICLR 2015.  
2. Kyunghyun Cho et al. *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. arXiv:1406.1078, 2014.  
3. Minh-Thang Luong, Hieu Pham, Christopher D. Manning. *Effective Approaches to Attention-based Neural Machine Translation*. arXiv:1508.04025, 2015.  
4. Ian Goodfellow, Yoshua Bengio, Aaron Courville. *Deep Learning*. MIT Press, 2016. Chapter on sequence modeling and attention-related context.  
5. Dive into Deep Learning. *Attention Mechanisms and Seq2Seq*. 适合把编码器-解码器、softmax 权重和上下文向量连起来看。  
6. Stanford CS224n 相关课程资料。适合补“编码器状态、解码器状态、对齐分数”这些术语的上下文。

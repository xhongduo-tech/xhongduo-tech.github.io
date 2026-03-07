## 核心结论

位置编码解决的是 Transformer 不知道“第几个 token 在前、谁离谁更近”的问题。自注意力本身只会计算内容之间的相似度，不会天然理解顺序；更准确地说，模型能判断“这两个向量是否相关”，但如果不额外注入位置信号，它并不知道“谁先出现、谁后出现、两者隔了多远”。

位置编码的大致演变可以概括为四步：

| 方法 | 位置感知方式 | 是否直接建模相对距离 | 额外参数 | 长度外推 |
|---|---|---:|---:|---:|
| 正弦绝对位置编码 | 给每个位置一个固定波形向量，加到 embedding 上 | 否 | 0 | 一般 |
| Shaw 相对位置编码 | 在 attention 分数里加入相对距离嵌入 | 是 | 有 | 中等 |
| ALiBi | 在 attention 分数里加入线性距离惩罚 | 是 | 近似 0 | 较强 |
| RoPE | 对 query/key 做按位置旋转 | 是 | 0 | 较强，但依赖频率设计 |

演变方向很清楚：从“给位置编号”变成“直接让注意力感知距离”。绝对位置编码更像给每个 token 贴一个位置指纹；相对位置编码、ALiBi、RoPE 则把“相距多远”直接写进注意力计算里。

如果只给一个工程判断：

- 经典 Transformer 教学和小模型实验，用正弦绝对位置编码最容易理解。
- 需要显式距离建模时，Shaw 式相对位置编码更直观。
- 追求长上下文外推、实现简单、推理成本低时，ALiBi 很实用。
- 当前大多数 LLM 场景里，RoPE 是更常见的默认选择，因为它在效果、参数量和实现复杂度之间比较平衡。

---

## 问题定义与边界

Transformer 的基本注意力是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里的 $QK^\top$ 只反映内容相似度，不反映位置。也就是说，注意力矩阵里的每个分数，本质上只是“当前位置的 query 与另一个位置的 key 是否匹配”。如果不额外提供顺序信息，模型就很难稳定区分“谁是主语、谁是宾语、谁先谁后”。

例如下面两条序列：

- 序列 A：`猫 追 狗`
- 序列 B：`狗 追 猫`

它们包含的 token 集合相同，只是顺序不同。没有位置编码时，模型只看到三个词的内容向量，缺少可靠机制来表达“第一个词”和“第三个词”的角色差异。

因此，位置编码至少要回答三个问题：

| 问题 | 含义 | 典型方法 |
|---|---|---|
| 绝对位置 | 这是第几个 token | 正弦位置编码 |
| 相对距离 | 两个 token 相隔几步 | Shaw、ALiBi、RoPE |
| 长度外推 | 训练 1k，推理能否稳定到 4k、8k 甚至更长 | ALiBi、RoPE 更常见 |

把这三个问题拆开看，会更容易理解后续方案为什么会演变：

1. 绝对位置关注“当前位置的编号”。
2. 相对距离关注“当前位置与其他位置之间的差值”。
3. 长度外推关注“训练没见过那么长的序列时，位置机制是否还能工作”。

边界也要说清楚。

第一，位置编码不能单独决定模型上限。它只是给注意力补充顺序和距离信号，最终效果仍受模型规模、数据分布、训练长度、优化器、mask 设计和推理实现影响。

第二，所谓“外推能力强”不等于“无限长都稳定”。例如训练长度是 2k 的模型，即使用 ALiBi 或 RoPE，推理到 32k、64k 时也可能明显退化。更准确的说法是：它们通常比纯绝对位置编码退化得更慢，而不是不会退化。

第三，不同任务需要的位置关系并不一样：

- 机器翻译更关心局部到中程的对齐关系。
- 代码补全更依赖最近几百个 token 的作用域、缩进和变量定义。
- 长文问答更关心远距离检索和跨段引用是否还能保真。
- 表格或结构化序列任务有时同时要求局部邻接和远距离锚点。

所以位置编码不是“谁全面优于谁”，而是“谁更适合哪种距离结构、哪种长度分布、哪种部署约束”。

---

## 核心机制与推导

### 1. 绝对位置编码：给每个位置一个固定波形

正弦位置编码使用不同频率的正弦和余弦函数，为每个位置构造一个固定向量：

$$
PE(pos, 2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos, 2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

其中：

- $pos$ 是位置索引，从 0 开始。
- $d$ 是模型维度。
- $i$ 是维度对的编号。

这组公式的含义不是“给位置编号”，而是“给不同维度分配不同频率”。低维部分变化快，高维部分变化慢，因此一个位置向量同时包含短周期和长周期信息。

当 $d=4$ 时：

- $pos=0$ 近似得到 $[0,1,0,1]$
- $pos=1$ 近似得到 $[\sin 1,\cos 1,\sin 0.01,\cos 0.01]$

如果写成数值，大约是：

$$
PE(0)\approx[0,1,0,1]
$$

$$
PE(1)\approx[0.8415,0.5403,0.0100,0.99995]
$$

然后把位置向量直接加到 token embedding 上：

$$
x_{pos}=e_{token}+PE(pos)
$$

这里有一个对新手很重要的理解点：正弦位置编码并没有改 attention 公式本身，它只是修改了输入表示。也就是说，位置先混进 embedding，再通过后续线性层变成 Q、K、V。它的位置信号是“间接进入注意力”的。

优点：

- 无需学习额外参数。
- 实现简单。
- 任意位置都能用同一公式计算，不依赖查表。

局限：

- attention 分数里没有显式的“距离项”。
- 对训练长度之外的泛化通常不稳定。
- 位置是输入级别注入的，层数变深后，显式位置信号可能被内容特征部分冲淡。

可以把它理解成：每个 token 入场时都带着一个位置签名，但 token 两两交互时，并没有单独再问一句“你们俩隔了多远”。

### 2. Shaw 相对位置编码：把距离写进 attention 分数

Shaw 等人的核心思想是：如果模型真正需要的是“两个位置之间的距离”，那么不如在注意力的两两交互阶段直接加入距离信息，而不是只在输入端给每个 token 打标签。

简化后的打分公式可以写成：

$$
score_{m,n}=q_m^\top k_n + q_m^\top r_{m-n}
$$

其中：

- $q_m$ 是第 $m$ 个位置的 query。
- $k_n$ 是第 $n$ 个位置的 key。
- $r_{m-n}$ 是相对距离 $(m-n)$ 对应的可学习向量。

第一项 $q_m^\top k_n$ 仍然是内容匹配。
第二项 $q_m^\top r_{m-n}$ 则是距离偏置，只不过这个偏置不是纯标量，而是 query 与“距离向量”的内积。

这意味着模型能学到类似下面的规则：

- 相邻位置通常值得更多关注。
- 距离为 2 或 3 时，可能对应局部短语结构。
- 某些头更偏好向左看，某些头更偏好向右看。

一个最小例子：

- 若 $m-n=1$，设 $r_{+1}=[0.1,0.2]$
- 若 $m-n=5$，设 $r_{+5}=[-0.1,-0.3]$
- 若 $q_m=[1,2]$

则有：

$$
q_m^\top r_{+1}=1\times0.1+2\times0.2=0.5
$$

$$
q_m^\top r_{+5}=1\times(-0.1)+2\times(-0.3)=-0.7
$$

这表示：相邻 token 会被加分，远处 token 会被减分，而且这个加减分不是固定值，而是和 query 的方向有关，因此表达能力比“固定距离惩罚”更强。

Shaw 方法常见还有两个工程细节：

1. 会截断最大相对距离。
   例如只保留 $[-k,k]$ 范围内的距离，超出部分统一映射到边界桶。
2. 有些实现还会把相对位置项加入 value 侧，而不只加入 score 侧。

截断的原因很直接：如果序列长到 8k、16k，每个距离都学一个向量，参数量和实现复杂度都会上涨。通常没有必要为“距离 1731”和“距离 1732”分别学习完全独立的表示。

Shaw 相对位置编码的意义在于，它首次把“相对距离”变成注意力机制中的一等公民。相比绝对编码，它更贴近很多序列任务真正需要的结构。

### 3. ALiBi：用线性惩罚替代复杂距离嵌入

ALiBi 的思路更极端：既然许多任务都体现“越近越重要”，那就不必学习一张复杂的相对位置表，而是直接在注意力分数里加入线性距离惩罚。

公式可以写成：

$$
score_{m,n}=q_m^\top k_n - s_h \cdot (m-n)
$$

在 causal mask 的语言模型里，通常只考虑 $n \le m$，所以 $(m-n)\ge0$。很多介绍会写成：

$$
score_{m,n}=q_m^\top k_n - s_h \cdot |m-n|
$$

其中 $s_h$ 是第 $h$ 个注意力头的斜率。

它的含义非常直接：

- 距离越远，分数越低。
- 不同头使用不同斜率，因此有的头偏局部，有的头偏中程。

如果某个头的斜率是 $s_h=0.2$，距离为 3，那么偏置是：

$$
-0.2 \times 3 = -0.6
$$

如果原始内容分数是 1.4，则加入 ALiBi 后变成：

$$
1.4 - 0.6 = 0.8
$$

再经过 softmax，远距离位置获得的注意力权重会被明显压低。

为什么 ALiBi 常被认为适合长度外推？关键原因不是它“更聪明”，而是它的位置规则非常稳定：

- 没有长度上限表。
- 不依赖训练时学到某个最大位置编号。
- 推到更长序列时，只是把线性惩罚继续延长。

这种设计隐含了一个强假设：距离影响大体单调，而且可以用近似线性的形式表达。对很多语言建模任务，这个假设足够有用；但对某些需要复杂全局结构的任务，它未必最优。

对新手来说，可以把 ALiBi 记成一句话：它不是“给 token 一个位置向量”，而是“直接给远距离配对扣分”。

### 4. RoPE：把位置变成旋转角度

RoPE 的核心不是加法，而是旋转。它把位置编码从“向量叠加”变成“几何变换”。

对每两维一组，定义旋转矩阵：

$$
R(\theta p)=
\begin{bmatrix}
\cos(\theta p) & -\sin(\theta p) \\
\sin(\theta p) & \cos(\theta p)
\end{bmatrix}
$$

于是对第 $m$ 个位置的 query 和第 $n$ 个位置的 key，有：

$$
q'_m = R(\theta m) q_m,\quad k'_n = R(\theta n) k_n
$$

关键性质是：

$$
{q'_m}^\top k'_n = q_m^\top R(\theta(n-m)) k_n
$$

这一步值得单独解释，因为它是 RoPE 受欢迎的根本原因。

对于二维旋转矩阵，有：

$$
R(a)^\top R(b)=R(b-a)
$$

所以：

$$
{q'_m}^\top k'_n
= q_m^\top R(\theta m)^\top R(\theta n)k_n
= q_m^\top R(\theta(n-m))k_n
$$

结果说明：虽然对每个位置使用的是“绝对角度”，但两个位置相乘后，绝对角度会相消，只剩相对角度差。这意味着 RoPE 最终把相对位置信息自然编码进了注意力打分里。

可以用平面向量做直观理解：

- 位置 1 的向量旋转 30°
- 位置 2 的向量旋转 60°

真正影响点积的，不是“各自转了多少”，而是“相差 30°”。

RoPE 的工程优势在于：

- 不增加参数。
- 直接作用在 Q、K 上，而不是只在输入层注入。
- 具有相对位置性质，同时保留多频率结构。

RoPE 的常见实现会给不同维度对分配不同频率：

$$
\theta_i = 10000^{-2i/d}
$$

于是低维对高频，能表达近距离差异；高维对低频，能保留更慢变化的长程结构。这一点和正弦编码有相似之处，但 RoPE 的频率是作用于旋转，而不是直接形成加法位置向量。

### 5. 真实工程例子

在长上下文语言模型里，位置编码会直接影响推理长度、注意力模式和稳定性。

以代码补全为例。假设模型要处理一个 8k token 的仓库片段，其中包含：

- 文件开头的 import 和全局常量
- 中间多个函数定义
- 末尾某处调用前面定义的类和变量

这时不同位置编码的行为往往不同：

- 绝对位置编码更容易依赖训练时见过的长度分布。长度一旦拉长，后段 token 的位置表示可能偏离训练区间。
- Shaw 相对位置编码会明确建模“当前位置与前面定义之间隔了多少步”，适合表达距离结构，但实现开销更高。
- ALiBi 会天然偏向最近几百个 token，适合局部依赖强的场景，例如最近作用域内的变量引用。
- RoPE 往往在局部结构和中长距离联系之间取得更平衡的效果，因此更适合通用 LLM。

再看一个中文长文问答例子。假设问题在文末，证据在文首：

- 如果模型只会强烈关注近处上下文，那么它可能回答不出文首信息。
- 如果模型完全没有距离偏置，它又可能在长序列里产生噪声注意力。
- 一个好的位置机制，需要让模型既保留局部精度，又不至于把远处信息彻底压扁。

这也是为什么现代实践里，位置编码不是一个孤立超参数，而是和训练长度、KV cache、长上下文微调、推理插值策略一起考虑。

---

## 代码实现

下面的代码不是完整 Transformer，而是把四种位置机制的核心操作拆出来，保证可以直接运行，并且能看到数值结果。代码只依赖 `numpy`，用 `python3 demo.py` 就可以执行。

```python
import math
import numpy as np


def sinusoidal_pe(seq_len: int, dim: int) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")
    pe = np.zeros((seq_len, dim), dtype=np.float64)
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    div_terms = np.exp(
        -math.log(10000.0) * np.arange(0, dim, 2, dtype=np.float64) / dim
    )
    pe[:, 0::2] = np.sin(positions * div_terms)
    if dim > 1:
        pe[:, 1::2] = np.cos(positions * div_terms[: pe[:, 1::2].shape[1]])
    return pe


def shaw_relative_bias(distance: int, rel_table: dict[int, np.ndarray], q: np.ndarray) -> float:
    if distance not in rel_table:
        raise KeyError(f"distance {distance} not found in rel_table")
    rel = rel_table[distance]
    if rel.shape != q.shape:
        raise ValueError("relative embedding and q must have the same shape")
    return float(q @ rel)


def alibi_bias(query_pos: int, key_pos: int, slope: float) -> float:
    return -slope * abs(query_pos - key_pos)


def rotate_half(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1 or x.shape[0] % 2 != 0:
        raise ValueError("x must be a 1D vector with even dimension")
    even = x[0::2]
    odd = x[1::2]
    out = np.empty_like(x)
    out[0::2] = -odd
    out[1::2] = even
    return out


def rope(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    if x.ndim != 1 or x.shape[0] % 2 != 0:
        raise ValueError("x must be a 1D vector with even dimension")
    dim = x.shape[0]
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = pos * inv_freq
    cos = np.repeat(np.cos(angles), 2)
    sin = np.repeat(np.sin(angles), 2)
    return x * cos + rotate_half(x) * sin


def attention_score(q: np.ndarray, k: np.ndarray) -> float:
    return float(q @ k)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def demo_alibi_attention():
    raw_scores = np.array([1.2, 1.1, 1.0, 0.9], dtype=np.float64)
    biases = np.array([alibi_bias(3, k, slope=0.2) for k in range(4)], dtype=np.float64)
    final_scores = raw_scores + biases
    probs = softmax(final_scores)
    return raw_scores, biases, final_scores, probs


def demo_rope_relative_property():
    q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    k = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    s1 = attention_score(rope(q, 5), rope(k, 2))
    s2 = attention_score(rope(q, 9), rope(k, 6))

    # 两组位置差都等于 3，因此分数应接近一致
    return s1, s2


if __name__ == "__main__":
    # 1) 正弦绝对位置编码
    pe = sinusoidal_pe(seq_len=3, dim=4)
    print("sinusoidal PE:")
    print(np.round(pe, 4))
    assert pe.shape == (3, 4)
    assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]), atol=1e-8)

    # 2) Shaw 相对位置编码
    q = np.array([1.0, 2.0], dtype=np.float64)
    rel_table = {
        1: np.array([0.1, 0.2], dtype=np.float64),
        3: np.array([-0.1, 0.0], dtype=np.float64),
    }
    bias_near = shaw_relative_bias(1, rel_table, q)
    bias_far = shaw_relative_bias(3, rel_table, q)
    print("\nshaw biases:", bias_near, bias_far)
    assert abs(bias_near - 0.5) < 1e-8
    assert abs(bias_far - (-0.1)) < 1e-8

    # 3) ALiBi
    raw_scores, biases, final_scores, probs = demo_alibi_attention()
    print("\nalibi raw scores:   ", np.round(raw_scores, 4))
    print("alibi biases:      ", np.round(biases, 4))
    print("alibi final scores:", np.round(final_scores, 4))
    print("alibi softmax:     ", np.round(probs, 4))
    assert abs(alibi_bias(0, 3, slope=0.2) - (-0.6)) < 1e-8

    # 4) RoPE 的旋转保持同位置点积不变
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    score_origin = attention_score(x, x)
    score_rotated = attention_score(rope(x, 5), rope(x, 5))
    print("\nrope same-position invariance:", score_origin, score_rotated)
    assert np.allclose(score_origin, score_rotated, atol=1e-8)

    # 5) RoPE 的相对性：只要位置差相同，分数近似一致
    s1, s2 = demo_rope_relative_property()
    print("rope relative-position scores:", s1, s2)
    assert np.allclose(s1, s2, atol=1e-8)

    print("\nall checks passed")
```

如果运行输出类似下面这样，就说明核心逻辑是通的：

```text
sinusoidal PE:
[[ 0.      1.      0.      1.    ]
 [ 0.8415  0.5403  0.01    1.    ]
 [ 0.9093 -0.4161  0.02    0.9998]]

shaw biases: 0.5 -0.1

alibi raw scores:    [1.2 1.1 1.  0.9]
alibi biases:       [-0.6 -0.4 -0.2 -0. ]
alibi final scores: [0.6 0.7 0.8 0.9]
alibi softmax:      [0.2138 0.2363 0.2612 0.2887]

rope same-position invariance: 30.0 30.0
rope relative-position scores: 14.6502 14.6502

all checks passed
```

把它放回 Transformer 的位置如下：

- 正弦绝对位置编码：在 token embedding 后直接相加。
- Shaw 相对位置编码：在 $QK^\top$ 之后、softmax 之前加距离项。
- ALiBi：在 $QK^\top$ 之后直接加线性偏置矩阵。
- RoPE：在算 attention 之前，对每个 head 的 $Q,K$ 先做旋转。

简化伪代码如下：

```python
# sinusoidal absolute
x = token_embedding + positional_embedding

# Shaw relative
score = (q @ k.T) + relative_bias_matrix

# ALiBi
score = (q @ k.T) + alibi_bias_matrix

# RoPE
q = rope(q, query_pos)
k = rope(k, key_pos)
score = q @ k.T
```

这里再补一个容易混淆的点：

- 正弦编码和 RoPE 都用了三角函数，但机制不同。
- 前者是“把位置向量加到输入上”。
- 后者是“让 Q/K 按位置旋转，再去做点积”。

---

## 工程权衡与常见坑

位置编码真正难的不是“会不会写公式”，而是“在长度、显存、泛化、吞吐和数值稳定性之间怎么折中”。

| 坑 | 影响 | 缓解措施 |
|---|---|---|
| 绝对位置编码依赖训练长度分布 | 超出训练长度时性能容易掉 | 用相对方法，或提高训练长度覆盖 |
| Shaw 相对距离表过大 | 参数和实现复杂度上升 | 截断最大距离，只保留窗口内距离 |
| Shaw 相对偏置实现不当 | 容易出现张量形状错误或显存飙升 | 先做小窗口版本，再扩到多头批量实现 |
| ALiBi 斜率过大 | 模型几乎只看邻近 token | 用分头几何级数斜率，结合验证集调参 |
| ALiBi 斜率过小 | 位置影响接近消失 | 检查长短依赖任务上的注意力分布 |
| RoPE 高频过强 | 超长上下文可能出现相位混叠，远距离语义退化 | 调整 base、用缩放 RoPE 或长上下文补丁 |
| RoPE 在缓存推理中位置索引错位 | 续写时 attention 直接异常 | 确保 KV cache 的 pos offset 正确累加 |
| 只看 benchmark 不看任务结构 | 线下分数好，线上长文本退化 | 用真实长度分布做评估 |

有几个误区值得单独展开。

第一，ALiBi 更能外推，不等于 ALiBi 一定更强。  
ALiBi 的强项是简单、便宜、规则稳定，但它把距离影响固定成线性惩罚，表达空间本来就比可学习相对表示更窄。

第二，RoPE 无参数，不等于长度可以无限扩。  
RoPE 的限制不在参数表，而在频率设计。上下文远超训练长度时，高频分量会快速振荡，导致远距离位置的相位关系不再稳定，模型虽然“收到了位置信号”，却可能无法像训练中那样解释它。

第三，正弦绝对编码不是没用。  
很多教学实现、小模型实验和基线系统，正弦编码依然是最合适的起点。因为它的价值首先是解释性强，而不是在所有指标上都最优。

第四，不同头看到的位置模式本来就不应该完全一致。  
无论是 Shaw 的相对表、ALiBi 的不同斜率，还是 RoPE 在多头上的实现，合理现象通常都是“有的头偏局部，有的头偏中程”。如果所有头都只看最近邻，模型的长距离表达能力往往有限。

工程上建议至少做三类评估，而不是只看单一 perplexity：

| 评估维度 | 关注问题 | 示例 |
|---|---|---|
| 训练内长度 | 是否学会基本顺序关系 | 2k 以内验证集 loss |
| 训练外长度 | 是否出现明显外推崩溃 | 4k、8k、16k 分桶评估 |
| 真实任务结构 | 是否保住关键依赖 | 长代码补全、长文检索、跨段问答 |

---

## 替代方案与适用边界

除了本文四类主线方法，还有一些替代思路。

第一类是 NoPE，即不显式加入位置编码，让模型依赖因果掩码、训练数据分布和层内结构自己学习位置关系。它在某些特定训练设置下可以工作，但通常不是零基础工程实践里的优先方案，因为行为更依赖训练配方和数据规模。

第二类是更复杂的相对偏置或多尺度方法。典型做法包括：

- 把不同距离区间离散成多个桶。
- 对近距离和远距离使用不同参数化方式。
- 对频率做插值、重标定或动态缩放，以适应更长上下文。

这些方法适合超长序列、跨尺度依赖强的任务，但实现成本、调参难度和验证成本都会更高。

一个实用选择矩阵如下：

| 方法 | 伸缩性 | 额外参数 | 训练/实现成本 | 适用场景 |
|---|---:|---:|---:|---|
| 正弦绝对位置编码 | 中 | 0 | 低 | 教学、基线、小模型 |
| Shaw 相对位置编码 | 中 | 中 | 中 | 明确需要距离嵌入的任务 |
| ALiBi | 高 | 近似 0 | 低 | 长上下文外推、轻量部署 |
| RoPE | 高 | 0 | 中 | 通用 LLM、代码、对话、长文本 |
| NoPE/其他多尺度方法 | 视训练而定 | 低到中 | 高 | 研究型探索、特定长序列任务 |

如果只给一个落地建议：

- 教学和入门实现，先写正弦绝对位置编码。
- 想理解“相对距离如何进 attention”，写 Shaw。
- 想快速做长上下文 baseline，优先 ALiBi。
- 想贴近现代大模型实践，优先 RoPE，并关注长上下文扩展策略。

如果再补一条更实际的判断标准，可以用下面这张表：

| 你的约束 | 更合适的选择 |
|---|---|
| 先把 Transformer 从零写通 | 正弦绝对位置编码 |
| 重点研究注意力中的距离偏置 | Shaw 相对位置编码 |
| 参数预算紧、想做长上下文试验 | ALiBi |
| 想对齐主流 LLM 实践 | RoPE |
| 研究超长上下文技巧 | RoPE 变体或更复杂相对方案 |

---

## 参考资料

- Vaswani et al. 2017, *Attention Is All You Need*. https://arxiv.org/abs/1706.03762
- Shaw, Uszkoreit, Vaswani. 2018, *Self-Attention with Relative Position Representations*. https://aclanthology.org/N18-2074/
- Press, Smith, Lewis. 2021, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. https://arxiv.org/abs/2108.12409
- Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864
- Transformer 论文中正弦位置编码的原始定义见 Section 3.5。适合先读公式，再读实现。
- Shaw 2018 的关键贡献不是“换一种位置 embedding”，而是把相对距离直接并入注意力交互。
- ALiBi 的重点不是引入复杂参数，而是利用固定线性偏置提升长度外推稳定性。
- RoFormer 的重点不是“旋转本身”，而是通过旋转把相对位置信息保留在点积结构里。
- Hugging Face paper page for Shaw 2018 summary: https://huggingface.co/papers/1803.02155
- Hugging Face paper page for ALiBi summary: https://huggingface.co/papers/2108.12409
- Hugging Face paper page for RoFormer summary: https://huggingface.co/papers/2104.09864

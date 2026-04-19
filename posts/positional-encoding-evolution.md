## 核心结论

位置编码 = 顺序信息注入机制。

Transformer 的自注意力机制可以计算 token 之间的相关性，但它天然不知道 token 的先后顺序。只看词集合时，“猫追狗”和“狗追猫”包含同样的三个字，但语义完全不同。模型必须知道“猫”在“追”前面还是“狗”在“追”前面。

位置编码的演进路线可以概括为：

| 类型 | 核心做法 | 代表方法 | 位置信息进入位置 |
|---|---|---|---|
| 绝对位置编码 | 给每个位置一个向量 | sinusoidal PE、learned PE | 输入层 |
| 相对/旋转/偏置式编码 | 让注意力分数直接感知距离 | RoPE、ALiBi | 注意力计算层 |

早期方法把位置向量 $p_i$ 加到词向量 $x_i$ 上，让模型从输入开始就看到“第几个 token”。后来的方法把位置信息写进注意力分数里，让模型在判断 token 关系时同时考虑相对距离。

---

## 问题定义与边界

自注意力是一种让序列中每个 token 关注其他 token 的计算机制。它关心“谁和谁相关”，但如果不额外加入位置，它不会自动理解“谁在前、谁在后”。

一个玩具例子：

| 序列 | 词集合 | 顺序 | 含义 |
|---|---|---|---|
| 猫 追 狗 | 猫、追、狗 | 猫在前，狗在后 | 猫是动作发起者 |
| 狗 追 猫 | 猫、追、狗 | 狗在前，猫在后 | 狗是动作发起者 |

如果模型只看到 token 内容，而没有位置信息，这两句话会非常接近。位置编码就是告诉模型：每个 token 在队列里的编号是什么。

统一记号如下：

| 符号 | 含义 |
|---|---|
| $x_i$ | 第 $i$ 个 token 的词向量 |
| $p_i$ | 第 $i$ 个位置的位置向量 |
| $h_i$ | 融合位置后的输入表示 |
| $q_i$ | 第 $i$ 个 token 的 query，表示“我要找什么” |
| $k_j$ | 第 $j$ 个 token 的 key，表示“我能被怎样匹配” |
| $d$ | 向量维度 |
| $d_k$ | attention 中 key/query 的维度 |

位置编码解决的是顺序建模问题，不等于解决所有长文本问题。

| 问题 | 位置编码是否直接解决 | 说明 |
|---|---:|---|
| token 先后顺序 | 是 | 告诉模型每个 token 的位置或相对距离 |
| 语义理解 | 否 | 语义主要来自训练数据、模型结构和参数 |
| 长文本记忆 | 否 | 还受上下文窗口、训练分布、注意力实现影响 |
| 长度外推稳定性 | 不保证 | 训练长度内有效，不代表推理更长时稳定 |
| 推理能力 | 否 | 位置编码只提供顺序信号 |

真实工程里常见边界是：模型训练时最大长度是 2048 token，推理时直接塞入 8192 token。即使位置编码理论上可以生成更长位置，也不代表模型一定能稳定理解长距离依赖。

---

## 核心机制与推导

最直接的方法是绝对位置编码。绝对位置的意思是：位置 $0$、位置 $1$、位置 $2$ 各自有自己的表示。输入表示为：

$$
h_i = x_i + p_i
$$

这像给每个词贴一个“第几个”的标签。词向量表示 token 内容，位置向量表示 token 所在位置，两者相加后送进 Transformer。

正弦位置编码使用固定公式，不需要训练：

$$
p(pos, 2k) = \sin(pos / 10000^{2k/d})
$$

$$
p(pos, 2k+1) = \cos(pos / 10000^{2k/d})
$$

其中 $pos$ 是位置编号，$k$ 是维度对编号。不同维度使用不同频率的正弦和余弦函数，因此一个位置会被表示成多种尺度上的周期信号。白话说，就是用一组快慢不同的波形给每个位置做标记。

可学习位置编码更直接：

$$
p_i = P[i], \quad P \in \mathbb{R}^{L_{max} \times d}
$$

$P$ 是一个可训练表，$L_{max}$ 是最大位置数。第 $i$ 个位置就查表拿出第 $i$ 行。它的优点是简单，缺点是位置表有硬上限。

RoPE，即 Rotary Position Embedding，中文常译为旋转位置编码。它不把位置向量加到输入上，而是对 attention 里的 $q_i$ 和 $k_j$ 做旋转：

$$
q'_i = R_i q_i
$$

$$
k'_j = R_j k_j
$$

$$
score(i,j) = (q'_i)^T k'_j = q_i^T R_{j-i} k_j
$$

$R_i$ 是由位置 $i$ 决定的旋转矩阵。关键点在最后一项：注意力分数可以写成和 $j-i$ 有关的形式。也就是说，RoPE 让模型在计算相似度时自然感知相对距离。

一个 2 维玩具例子：令 $q=[1,0]$，$k=[1,0]$。如果两个位置差为 1，旋转后的点积近似为 $\cos(1)\approx 0.5403$；如果位置差为 2，点积近似为 $\cos(2)\approx -0.4161$。距离变化会直接改变相似度。

ALiBi，即 Attention with Linear Biases，意思是在 attention 分数上加入线性偏置。它的形式是：

$$
score_h(i,j) = \frac{q_i^T k_j}{\sqrt{d_k}} + b_h(i,j)
$$

$$
b_h(i,j) = -m_h(i-j), \quad j \le i
$$

这里 $h$ 表示注意力头，$m_h$ 是每个头的固定斜率。因果语言模型里，当前位置 $i$ 只能看当前位置之前的 $j$。距离越远，偏置越负，attention 分数越低。白话说，ALiBi 是直接给远距离 token 扣分。

| 方法 | 位置类型 | 核心机制 | 优点 | 主要限制 |
|---|---|---|---|---|
| sinusoidal PE | 绝对位置 | 固定三角函数 | 无需训练参数 | 外推不一定稳定 |
| learned PE | 绝对位置 | 训练位置表 | 简单有效 | 有最大长度硬上限 |
| RoPE | 相对位置 | 旋转 Q/K | 适合相对距离建模 | 对 base、缩放等配置敏感 |
| ALiBi | 相对距离 | attention score 加线性偏置 | 外推能力强，参数少 | 主要验证于因果语言模型 |

---

## 代码实现

实现上常见三类：查表加法、旋转 Q/K、添加 attention bias。

learned PE 的核心就是：

```python
# absolute PE
h = x + pos_emb[position_ids]
```

RoPE 的核心就是：

```python
# RoPE
q, k = apply_rope(q, k, position_ids)
attn = q @ k.transpose(-1, -2)
```

ALiBi 的核心就是：

```python
# ALiBi
attn = (q @ k.transpose(-1, -2)) / sqrt(dk) + alibi_bias
```

下面是一段可运行的 Python 代码，演示正弦位置编码、简化 RoPE 和 ALiBi bias：

```python
import math
import numpy as np

def sinusoidal_position(pos, d):
    assert d % 2 == 0
    p = np.zeros(d)
    for k in range(d // 2):
        denom = 10000 ** (2 * k / d)
        p[2 * k] = math.sin(pos / denom)
        p[2 * k + 1] = math.cos(pos / denom)
    return p

def rotate_2d(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return r @ np.array(v)

def rope_score_2d(q, k, pos_q, pos_k):
    q_rot = rotate_2d(q, pos_q)
    k_rot = rotate_2d(k, pos_k)
    return float(q_rot @ k_rot)

def alibi_bias(i, j, slope):
    assert j <= i
    return -slope * (i - j)

p0 = sinusoidal_position(0, 4)
p1 = sinusoidal_position(1, 4)

assert np.allclose(p0, [0.0, 1.0, 0.0, 1.0])
assert np.allclose(p1, [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)])

q = [1.0, 0.0]
k = [1.0, 0.0]

score_distance_1 = rope_score_2d(q, k, 1, 0)
score_distance_2 = rope_score_2d(q, k, 2, 0)

assert abs(score_distance_1 - math.cos(1)) < 1e-9
assert abs(score_distance_2 - math.cos(2)) < 1e-9
assert alibi_bias(i=5, j=3, slope=0.25) == -0.5

print("position encoding checks passed")
```

真实工程例子：做长文档问答或代码审查时，输入可能从几千 token 增长到几万 token。如果模型使用 learned PE，并且 `max_position_embeddings=2048`，那么第 2049 个位置没有对应的训练位置表。即使框架允许扩展数组，新增位置也没有经过充分训练。使用 RoPE 或 ALiBi 的模型通常更适合这类长上下文生成任务，但仍然要验证实际长度外推效果。

实现检查清单：

| 检查项 | 为什么重要 |
|---|---|
| `position_ids` 是否连续 | 错位会让模型误判 token 顺序 |
| `pad token` 是否被正确排除 | padding 不应参与有效注意力 |
| `max_position_embeddings` 是否足够 | learned PE 超出上限会失败或退化 |
| `RoPE base/theta` 是否与训练一致 | 配置不同会改变旋转频率 |
| `attention mask` 是否与 bias 正确叠加 | mask 和 ALiBi 都会改 attention score |
| 推理长度是否经过验证 | 理论支持不等于工程稳定 |

---

## 工程权衡与常见坑

learned absolute PE 简单直接，适合固定长度附近的任务。它的问题是长度上限硬。如果模型只准备了 2048 个位置，就像只准备了 2048 个座位；超过以后没有自然对应的位置向量。

sinusoidal PE 可以按公式生成任意位置，但这不等于模型在更长长度上一定表现好。模型训练时没见过的距离分布，推理时仍然可能不稳定。

RoPE 更适合长上下文和相对位置建模，但它不是“换个函数名”就完成了。不同模型可能使用不同的 `theta`、缩放策略、插值策略和上下文扩展方法。训练和推理配置不一致，会直接改变 attention 分数。

ALiBi 的实现很轻量，不需要位置表，也不需要旋转 Q/K。它直接在 attention score 上加入距离惩罚，因此对长度外推友好。但它最经典的验证场景是因果语言模型。如果用于双向编码器、检索编码器或特殊 attention 结构，需要重新确认实现和任务是否匹配。

| 坑点 | 常见后果 | 规避方式 |
|---|---|---|
| learned PE 长度上限不足 | 超长输入报错或效果突降 | 预留足够 `max_position_embeddings` |
| sinusoidal PE 被误认为万能外推 | 长序列效果不稳定 | 做不同长度验证 |
| RoPE 参数不一致 | 推理分数分布变化 | 训练和推理对齐 `base/theta`、缩放配置 |
| ALiBi 任务不匹配 | bias 方向或 mask 语义错误 | 确认是否为因果 LM 及对应实现 |
| `position_ids` 与 padding 错位 | token 使用错误位置 | 检查首 token、pad offset 和 batch 对齐 |
| attention mask 与 bias 顺序错误 | 被 mask 的位置仍影响分数 | 明确 mask、bias、softmax 的组合顺序 |

一个高频问题是左 padding。生成模型批量推理时，为了对齐长度，短句可能在左侧补 pad。如果 `position_ids` 仍然从 pad 开始计数，真实第一个 token 的位置就会被偏移。位置偏移不会报语法错误，但会造成模型表现异常。

---

## 替代方案与适用边界

没有一种位置编码对所有模型都最优。选择方法取决于任务类型、上下文长度、训练预算和实现复杂度。

短文本分类中，输入长度稳定，句子通常不长，absolute PE 往往够用。比如判断一句用户评论是正向还是负向，模型只需要处理几十到几百 token，learned PE 或 sinusoidal PE 都可以稳定工作。

长文档问答、代码审查、长上下文对话中，模型需要在很长距离上建立联系。比如一个函数定义在文件开头，调用在几千行之后。此时更应该优先考虑 RoPE 或 ALiBi，因为它们把相对距离直接纳入 attention 计算。

| 方法 | 适用场景 | 不适合场景 |
|---|---|---|
| sinusoidal PE | 基础研究、教学实现、固定长度附近任务 | 对长文本外推要求很高的任务 |
| learned PE | 训练和推理长度稳定、追求实现简单 | 推理长度经常超过训练长度 |
| RoPE | 长上下文生成、代码模型、相对位置敏感任务 | 无法保证训练推理配置一致的系统 |
| ALiBi | 因果语言建模、强调长度外推 | 需要复杂双向位置关系的任务 |

选择建议可以压缩成下面这张表：

| 需求 | 优先选择 |
|---|---|
| 短文本任务 | learned PE 或 sinusoidal PE |
| 只追求实现简单 | learned PE |
| 需要固定公式、少参数 | sinusoidal PE |
| 长上下文生成 | RoPE |
| 强调推理长度外推 | ALiBi |
| 已有预训练模型 | 跟随原模型的位置编码，不随意替换 |

工程上最保守的原则是：预训练模型用了什么位置编码，微调和推理就保持一致。位置编码会影响 attention 分数分布，随意替换通常不是无成本改动。

---

## 参考资料

1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
2. [BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert)
3. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/papers/2104.09864)
4. [RoFormer model documentation](https://huggingface.co/docs/transformers/model_doc/roformer)
5. [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://github.com/ofirpress/attention_with_linear_biases)

阅读顺序建议：先看 Transformer 原论文理解正弦位置编码，再看 BERT 文档理解 absolute position embeddings 的工程约束，最后看 RoPE 和 ALiBi 的改进思路。

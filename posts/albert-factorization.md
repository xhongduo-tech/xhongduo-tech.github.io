## 核心结论

ALBERT 的 factorized embedding，中文常叫“分解式词嵌入参数化”，核心不是把整个模型统一缩小，而是把词表嵌入层从一个大矩阵 $V \times H$，拆成两个更小的矩阵 $V \times E$ 和 $E \times H$，其中 $V$ 是词表大小，$H$ 是 Transformer 隐藏维度，$E$ 是词嵌入维度，并且通常满足 $E \ll H$。

白话说，普通做法是“每个词直接存一个 $H$ 维向量”；ALBERT 改成“每个词先存一个更短的 $E$ 维向量，再投影到 $H$ 维去参加后续计算”。这样节省的不是注意力层和前馈层参数，而是最前面的查表矩阵参数。

它的价值在于两个点：

| 方案 | 参数形式 | 主要收益 | 主要不变部分 |
|---|---:|---|---|
| 普通 embedding | $V \times H$ | 实现简单 | 词表一大就膨胀 |
| ALBERT factorization | $V \times E + E \times H$ | 大幅降低 embedding 参数 | Encoder 计算规模基本不变 |

玩具例子：原来每个词直接对应一个 `768` 维向量；现在先查一个 `128` 维向量，再通过线性映射变成 `768` 维。前者像“词典里直接放完整版向量”，后者像“词典里只放压缩版，再统一转成模型工作维度”。

如果你的目标是“保留较大的 hidden size，但不让词表参数爆炸”，这就是一个直接有效的结构改动。

---

## 问题定义与边界

先定义问题。BERT 类模型里，embedding 层通常是一个大小为 $V \times H$ 的矩阵。这里的“embedding 层”就是把离散 token id 映射成连续向量的查表层。只要词表 $V$ 很大，这部分参数就会迅速变成显存和存储瓶颈。

符号含义如下：

| 符号 | 名称 | 白话解释 |
|---|---|---|
| $V$ | vocab size | 词表里一共有多少个 token |
| $E$ | embedding size | 词刚查出来时的低维表示长度 |
| $H$ | hidden size | Transformer 主干网络工作的表示长度 |

为什么这会成为问题？因为 $V$ 往往不是几百几千，而是几万到几十万。尤其在多语言预训练、搜索召回、代码模型、子词切分较细的场景里，词表很容易大到让 embedding 参数先占掉模型预算。

真实工程例子：一个多语言文本分类模型，假设词表做到 `120000`，hidden size 设为 `1024`。仅普通 embedding 就有 `122,880,000` 个参数。此时模型还没开始做上下文建模，只是“把 token 变成向量”这一步，就已经很重了。

边界也要说清楚。ALBERT 的 factorization 解决的是 embedding 参数膨胀，不是所有计算都变小。

| 部分 | 会不会明显变小 | 原因 |
|---|---|---|
| 词表 embedding 参数 | 会 | 从 $V \times H$ 变成 $V \times E + E \times H$ |
| Attention 参数 | 基本不会 | 仍然按 hidden size $H$ 工作 |
| FFN 参数 | 基本不会 | 仍然由 $H$ 和中间层维度决定 |
| 序列计算量 | 基本不会 | token 间交互仍发生在 $H$ 维空间 |

所以它适合回答的问题是：“词表太大，embedding 太贵，怎么办？”  
它不直接回答的问题是：“Transformer 编码器太慢，怎么办？”

---

## 核心机制与推导

普通 embedding 可以写成：输入 one-hot 向量 $x \in \mathbb{R}^{V}$，直接映射到 hidden space：

$$
h = xW,\quad W \in \mathbb{R}^{V \times H}
$$

因为 $x$ 是 one-hot，这本质上就是从矩阵 $W$ 里取出第 `token_id` 行。

ALBERT 把这一步拆成两段：

$$
e = xW_1,\quad W_1 \in \mathbb{R}^{V \times E}
$$

$$
h = eW_2,\quad W_2 \in \mathbb{R}^{E \times H}
$$

这里的 $e$ 是低维词表示，$h$ 才是送入 Transformer 的隐藏表示。白话说，词表只负责存一个短向量，进入主干网络之前再扩成模型需要的宽度。

参数量对比很直接：

$$
\text{普通 embedding 参数} = V \times H
$$

$$
\text{ALBERT embedding 参数} = V \times E + E \times H
$$

节省条件来自：

$$
V \times E + E \times H \ll V \times H
$$

当 $E \ll H$，并且 $V$ 足够大时，上式通常成立。尤其是大词表场景，节省非常明显。

看一个固定数值例子：

| 配置 | 计算式 | 参数量 |
|---|---:|---:|
| 普通 embedding | $30000 \times 768$ | 23,040,000 |
| ALBERT factorization | $30000 \times 128 + 128 \times 768$ | 3,938,304 |

参数减少量：

$$
23,040,000 - 3,938,304 = 19,101,696
$$

压缩比例约为：

$$
1 - \frac{3,938,304}{23,040,000} \approx 82.9\%
$$

这个结果说明，ALBERT 并不是靠缩小 hidden size 来省参数，而是把“和词表强绑定的那一大块”先压下来。这样 hidden size 仍可以保持在 `768`、`1024` 甚至更高，把更多容量留给上下文建模。

从表示学习角度看，这种设计隐含一个假设：词的“静态词义存储”不需要和主干网络的“动态上下文表示”使用完全相同的维度。前者可以更紧凑，后者需要更宽来支持注意力和层间变换。这就是 factorization 成立的结构前提。

---

## 代码实现

实现上最关键的一点，是把 `embedding_size` 和 `hidden_size` 显式分开。很多人沿用 BERT 习惯，默认认为词向量维度就等于隐藏维度，这正是 ALBERT 要打破的默认设置。

先看 Hugging Face 风格的配置示例：

```python
from transformers import AlbertConfig, AlbertModel

config = AlbertConfig(
    vocab_size=30000,
    embedding_size=128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)

model = AlbertModel(config)
```

核心流程其实很简单：

```python
e = embedding_lookup(token_ids)   # [batch, seq, E]
h = projection(e)                 # [batch, seq, H]
```

下面给一个可运行的最小 Python 例子，用纯列表模拟参数量计算和维度检查：

```python
def vanilla_embedding_params(vocab_size: int, hidden_size: int) -> int:
    return vocab_size * hidden_size

def albert_factorized_params(vocab_size: int, embedding_size: int, hidden_size: int) -> int:
    return vocab_size * embedding_size + embedding_size * hidden_size

def project_shape(batch: int, seq_len: int, embedding_size: int, hidden_size: int):
    e_shape = (batch, seq_len, embedding_size)
    h_shape = (batch, seq_len, hidden_size)
    return e_shape, h_shape

V, E, H = 30000, 128, 768

vanilla = vanilla_embedding_params(V, H)
factorized = albert_factorized_params(V, E, H)

assert vanilla == 23040000
assert factorized == 3938304
assert factorized < vanilla

e_shape, h_shape = project_shape(batch=2, seq_len=16, embedding_size=E, hidden_size=H)
assert e_shape == (2, 16, 128)
assert h_shape == (2, 16, 768)

reduction_ratio = 1 - factorized / vanilla
assert round(reduction_ratio, 3) == 0.829

print("vanilla:", vanilla)
print("factorized:", factorized)
print("reduction_ratio:", round(reduction_ratio, 4))
print("embedding shape:", e_shape)
print("hidden shape:", h_shape)
```

如果用 PyTorch 思路写，结构通常就是一个 `Embedding(V, E)` 加一个 `Linear(E, H)`。注意这不是“多加了一层 Transformer”，只是 embedding 之后的一个投影层。

配置时常见关注点如下：

| 配置项 | 含义 | 常见取值 | 配错后果 |
|---|---|---|---|
| `vocab_size` | 词表大小 | `30k`、`50k`、`120k` | 和 tokenizer 不一致会越界或语义错位 |
| `embedding_size` | 低维词表示长度 | `64`、`128`、`256` | 太小会损失词义信息 |
| `hidden_size` | 主干网络表示维度 | `768`、`1024` | 决定后续 attention/FFN 规模 |
| `num_attention_heads` | 注意力头数 | `12`、`16` | 通常要求能整除 `hidden_size` |

玩具例子可以这样理解：token `"cat"` 先查出一个 `128` 维向量，这一步只表示“这个 token 自身的大致词义”；然后统一映射成 `768` 维，才交给后面的自注意力层，让它结合上下文判断是动物、变量名，还是代码中的字符串。

---

## 工程权衡与常见坑

factorized embedding 的收益明确，但工程上不能把它理解成“白捡参数”。

第一个权衡是信息压缩。`embedding_size` 太小，词的初始表示会过度压缩。这里的“过度压缩”指的是很多原本需要区分的词，在进入主干网络之前就已经丢掉了细粒度差异。对于细粒度语义匹配、术语密集型检索、实体分类等任务，过小的 $E$ 容易伤效果。

第二个权衡是它不减主干算力。你会发现参数变少了，但训练速度和推理延迟未必按同样比例下降，因为 attention 和 FFN 仍然在 $H$ 维运行。

常见坑可以直接列成表：

| 坑点 | 现象 | 原因 | 规避方法 |
|---|---|---|---|
| `embedding_size` 设得过小 | 验证集效果下降 | 词信息在进入 encoder 前已损失 | 从 `128/256` 起试，不要直接压到 `32` |
| 只增大 `hidden_size` | 总参数又变大 | 主干层参数按 $H$ 增长 | 明确区分“embedding 优化”和“encoder 扩容” |
| 以为能显著降 FLOPs | 速度提升不明显 | 主计算仍在 attention/FFN | 把它看作参数优化，不是全面算力优化 |
| tokenizer 和 `vocab_size` 不一致 | 训练报错或索引越界 | 词表大小配置错误 | 固定 tokenizer 版本并校验 id 范围 |
| 复用 BERT 默认配置 | 实际上 `E=H` | 没真正启用 factorization | 显式检查 `embedding_size != hidden_size` |

再看一个工程对比：

| 做法 | embedding 参数 | encoder 参数 | 适合场景 |
|---|---|---|---|
| 只改小 `hidden_size` | 会下降 | 也会下降 | 需要整体降算力 |
| 保持大 `hidden_size`，改小 `embedding_size` | 明显下降 | 基本不变 | 词表大但仍想保留建模能力 |
| 同时改小 `hidden_size` 和 `embedding_size` | 都会下降 | 明显下降 | 极限轻量化部署 |

真实工程例子：做跨语言检索预训练时，词表可能覆盖多个语种和大量子词。此时如果直接用 `V×H`，embedding 先吃掉大量显存。把 `embedding_size` 设成 `128` 或 `256`，同时保留 `hidden_size=768/1024`，往往能更合理地分配参数预算。你省下的是“静态词典存储成本”，保住的是“上下文理解容量”。

训练前建议检查这几项：

| 检查项 | 为什么必须确认 |
|---|---|
| `embedding_size < hidden_size` 是否真的成立 | 否则没有 factorization 收益 |
| tokenizer 与 `vocab_size` 是否匹配 | 避免索引错位 |
| `hidden_size % num_attention_heads == 0` | 保证多头注意力维度合法 |
| 基线模型是否已记录 | 否则无法判断压缩是否值得 |

---

## 替代方案与适用边界

如果瓶颈确实来自大词表，ALBERT factorization 很合适；如果主要瓶颈来自 encoder 太深、长序列太贵、推理延迟太高，它就只能解决一部分。

常见替代方案如下：

| 方法 | 主要目标 | 优势 | 代价或限制 |
|---|---|---|---|
| ALBERT factorization | 降低 embedding 参数 | 对大词表很有效 | 不直接降低主干计算 |
| 参数共享 | 降低层间重复参数 | 对深层模型有效 | 表达能力可能受限 |
| 蒸馏 | 用小模型逼近大模型 | 部署友好 | 训练流程更复杂 |
| 量化 | 降低存储和推理开销 | 部署收益直接 | 可能带来精度损失 |
| 裁剪词表 | 直接缩小 $V$ | 简单有效 | 可能伤覆盖率和 OOV 表现 |
| 低秩分解 | 压缩大矩阵 | 理论统一 | 需额外验证结构适配性 |

适用边界也可以明确列出：

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 大词表预训练 | 很适合 | embedding 参数占比高 |
| 多语言任务 | 很适合 | 词表通常更大 |
| 低显存部署 | 较适合 | 可明显减模型参数量 |
| 长文本建模 | 作用有限 | 主要瓶颈常在注意力计算 |
| 极致低延迟推理 | 作用有限 | 主干 FLOPs 未根本改变 |

所以决策标准可以压缩成一句话：  
如果你卡在“词表查表矩阵太大”，优先考虑 factorized embedding；如果你卡在“Transformer 主干太重”，应优先看层数、hidden size、蒸馏、量化或稀疏注意力等方案。

---

## 参考资料

1. ALBERT 原论文：提出 factorized embedding parameterization，并给出参数分析与设计动机。<https://arxiv.org/abs/1909.11942>
2. Google Research ALBERT 官方实现：包含训练脚本、配置和模型说明。<https://github.com/google-research/albert>
3. Hugging Face ALBERT 文档：明确提供 `embedding_size` 与 `hidden_size` 的独立配置。<https://huggingface.co/docs/transformers/model_doc/albert>

## 核心结论

Embedding 层本质上不是“会理解词义的复杂模块”，而是一张形状为 $[V, d]$ 的参数表。这里的 $V$ 是词表大小，指一共有多少个 token；$d$ 是 embedding 维度，指每个 token 对应多少个浮点数。前向计算最核心的一步就是查表：

$$
h_{in} = E[t]
$$

其中 $E \in \mathbb{R}^{V \times d}$，$t$ 是 token id，$E[t]$ 表示取出第 $t$ 行向量。

语言模型里另一个关键事实是：输出端也常常使用同一张表。若最后一层隐藏状态是 $h_{out} \in \mathbb{R}^d$，那么输出 logits 可以写成：

$$
z = E h_{out}, \quad z_i = e_i \cdot h_{out}
$$

这里 $e_i$ 是 embedding 矩阵第 $i$ 行，$\cdot$ 表示向量内积，也就是两个向量逐项相乘再求和。白话讲，模型会拿“当前语义向量”去和词表里每个 token 的向量做相似度打分，分数越高，越可能被预测出来。

这就是 weight tying，也叫权重绑定：输入 embedding 和输出 LM Head 共用同一份参数。它既节省参数，也让输入空间和输出空间保持一致。对 GPT 类语言模型，这是标准做法之一。

一个直接的数字例子：

| 方案 | 参数公式 | 当 $V=32000,d=768$ 时 |
| --- | --- | --- |
| Untied（不共享） | $2 \times V \times d$ | $49{,}152{,}000 \approx 49.2M$ |
| Tied（共享） | $V \times d$ | $24{,}576{,}000 \approx 24.6M$ |

这意味着只做权重绑定，就能省掉约 2460 万个参数。对白盒理解来说，可以把它看成：原本输入端有一张词表表格，输出端还要再建一张同尺寸表格；共享后，第二张表直接复用第一张。

初始化上，embedding 常见做法是正态分布或截断正态，小标准差如 `std=0.02` 是 Transformer 预训练中的主流工程选择。Xavier/He 更适合分析全连接层在层间传播时的方差稳定性，而 embedding 是离散索引查表，不是“每次都把整层输入乘过去”的稠密映射，因此通常不是首选。

---

## 问题定义与边界

Embedding 层要解决的问题很具体：把离散 token id 变成连续向量，让后续注意力层和 MLP 能处理。离散，意思是 token 只是编号；连续，意思是模型里真正流动的是浮点向量。

最小定义如下：

- 输入：整数张量，比如 `[12, 98, 7, 0, 0]`
- 参数：embedding 矩阵 $E \in \mathbb{R}^{V \times d}$
- 输出：对应的向量序列，形状从 `[T]` 变成 `[T, d]`

参数量非常简单：

$$
\text{Params}_{embed} = V \times d
$$

如果再加一个独立的输出层 `Linear(d, V)`，且不带 bias，那么输出层参数量也是 $V \times d$。所以语言模型里 embedding 和 LM Head 往往是最“方方正正”的一大块参数。

对零基础读者，可以把它想成一张 Excel 表：

- 每一行是一个 token
- 行号就是 token id
- 每一列是这个 token 某个维度上的数值
- 输入 token 时，不做复杂计算，直接按行号把整行拿出来

`padding_idx` 是边界条件里最容易被忽略的一项。padding 指补齐位，也就是为了让不同长度样本能拼成一个 batch 而加上的占位 token。`padding_idx` 的含义是：这一行存在，但训练时不更新梯度。白话讲，它是“冻结行”。

工程上，embedding 还要同时和三部分边界对齐：

| 边界项 | 需要对齐什么 | 常见处理 |
| --- | --- | --- |
| 输入端 | token id 必须落在 `[0, V-1]` | 分词器与词表严格一致 |
| 输出端 | logits 维度必须覆盖词表 | `Linear(d, V)` 或 tied `E` |
| 并行端 | 词表尺寸要适配 TP 切分 | 把 $V$ pad 到合适倍数 |

这里的 TP 是 Tensor Parallel，张量并行，指把一个大矩阵按列或按行拆到多张 GPU 上。若词表维度要在多卡间均分，那么 $V$ 最好能被并行数整除；很多实现还会再 pad 到 64 或 128 的倍数，方便 kernel 对齐和内存布局。

---

## 核心机制与推导

先看输入端。对于一个 token id $t$，embedding 层输出：

$$
h_{in} = E[t]
$$

如果序列是 $x = [t_1, t_2, \dots, t_T]$，那么输出就是按位置逐个查表，得到：

$$
H = 
\begin{bmatrix}
E[t_1] \\
E[t_2] \\
\vdots \\
E[t_T]
\end{bmatrix}
\in \mathbb{R}^{T \times d}
$$

这一步没有“加权求和”，也没有“激活函数”，本质就是索引。

再看输出端。Transformer 主干处理完后，某个位置得到隐藏状态 $h_{out} \in \mathbb{R}^d$。若输出层权重矩阵记为 $W_{out} \in \mathbb{R}^{V \times d}$，那么：

$$
z = W_{out} h_{out}
$$

如果采用 weight tying，就令：

$$
W_{out} = E
$$

于是：

$$
z = E h_{out}
$$

对单个 token 的分数：

$$
z_i = e_i \cdot h_{out}
$$

这句话非常重要。它说明预测某个 token 的过程，其实是在比较“当前上下文压缩出来的语义向量”与“词表中每个 token 向量”的相似程度。内积大，说明更匹配；softmax 之后概率更高。

用一个玩具例子说明。假设词表只有 4 个 token：`["猫", "狗", "跑", "<pad>"]`，embedding 维度只有 2。模型最后得到一个状态 $h_{out}=[0.9, 0.1]$。如果词表中“猫”的向量是 $[1.0, 0.0]$，“狗”的向量是 $[0.8, 0.2]$，“跑”的向量是 $[-0.2, 1.1]$，那么：

- “猫”分数：$1.0 \times 0.9 + 0.0 \times 0.1 = 0.9$
- “狗”分数：$0.8 \times 0.9 + 0.2 \times 0.1 = 0.74$
- “跑”分数：$-0.2 \times 0.9 + 1.1 \times 0.1 = -0.07$

于是模型更偏向输出“猫”。这个例子虽然极小，但已经完整体现了 embedding 既做输入表示、又做输出匹配的逻辑。

文字图示可以写成：

查表 `token_id -> E[token_id]` → Transformer 隐藏状态 `h_out` → 与词表每一行做内积 `e_i · h_out` → 得到 logits → softmax

Untied 与 Tied 的参数流区别如下：

| 项目 | Untied | Tied |
| --- | --- | --- |
| 输入 embedding | 独立矩阵 $E_{in}$ | 共享矩阵 $E$ |
| 输出 LM Head | 独立矩阵 $W_{out}$ | 直接复用 $E$ |
| 参数量 | $2Vd$ | $Vd$ |
| 语义空间 | 输入输出可能漂移 | 输入输出天然对齐 |

真实工程例子里，GPT-2 一类模型经常把 token embedding 和 LM Head 直接绑定。这样做的收益不只是“省参数”，还包括语义空间一致性。输入时“apple”对应的向量与输出时“预测 apple 的模板向量”是同一行参数，训练信号会汇聚到同一个位置上。

---

## 代码实现

先给一个可运行的玩具实现，不依赖任何第三方库，只展示查表、tied logits 和参数量计算的核心逻辑。

```python
import math

def embedding_lookup(table, token_ids):
    return [table[i] for i in token_ids]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def tied_logits(table, h_out):
    return [dot(row, h_out) for row in table]

def param_count(vocab_size, dim, tied=True):
    return vocab_size * dim if tied else 2 * vocab_size * dim

# 4 个 token，2 维 embedding
E = [
    [1.0, 0.0],   # token 0
    [0.8, 0.2],   # token 1
    [-0.2, 1.1],  # token 2
    [0.0, 0.0],   # pad token 3
]

tokens = [0, 2, 3]
x = embedding_lookup(E, tokens)
assert x[0] == [1.0, 0.0]
assert x[2] == [0.0, 0.0]

h_out = [0.9, 0.1]
logits = tied_logits(E, h_out)

# token 0 的得分最高
best_token = max(range(len(logits)), key=lambda i: logits[i])
assert best_token == 0

# 参数量检查
V, d = 32000, 768
assert param_count(V, d, tied=False) == 49152000
assert param_count(V, d, tied=True) == 24576000

def pad_vocab_size(vocab_size, multiple):
    return math.ceil(vocab_size / multiple) * multiple

assert pad_vocab_size(50000, 128) == 50048
assert pad_vocab_size(50257, 128) == 50304
```

上面最后两个 `assert` 也顺手验证了词表补齐。`50257 -> 50304` 是 GPT-2 家族里非常常见的工程写法。

实际项目里通常直接用 PyTorch：

```python
import torch
import torch.nn as nn

vocab_size, d_model, pad_id = 50304, 768, 0

embed = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
    padding_idx=pad_id,
)

lm_head = nn.Linear(d_model, vocab_size, bias=False)
lm_head.weight = embed.weight  # weight tying

# 常见初始化：小标准差正态
nn.init.normal_(embed.weight, mean=0.0, std=0.02)

# 如果使用 padding_idx，通常会显式把 pad 行设为 0
with torch.no_grad():
    embed.weight[pad_id].zero_()
```

这里有四个实现点需要分清：

1. `nn.Embedding(V, d)` 的参数形状就是 `[V, d]`。
2. `padding_idx=pad_id` 会让该行不参与梯度更新。
3. `lm_head.weight = embed.weight` 不是“拷贝数值”，而是让两者指向同一份权重。
4. 初始化常见用小标准差正态，如 `std=0.02`；这和 GPT 类模型的整体初始化习惯一致。

为什么很多实现不用 Xavier/He 初始化 embedding？核心原因不是“embedding 完全没有梯度问题”，而是 Xavier/He 的设计目标是控制层与层之间线性变换后的方差传播；embedding 是离散查表，前向没有把输入向量乘过整个矩阵这一过程。它当然会接收反向梯度，但初始化关注点更多是早期训练时的表示尺度和 logits 稳定性，所以工程上更常见固定小标准差正态或截断正态。

---

## 工程权衡与常见坑

Embedding 层看起来简单，但在训练系统里经常踩坑，尤其是 `padding_idx`、weight tying 和词表对齐。

最常见的问题如下：

| 坑 | 影响 | 解决方式 |
| --- | --- | --- |
| 忘记设置 `padding_idx` | pad token 也被更新，污染表示 | `nn.Embedding(..., padding_idx=pad_id)` |
| 绑定权重后又手动复制 | 变成两份参数，失去 tying 效果 | 直接共享同一 `Parameter` |
| 词表不能整除 TP | 多卡切分不均，可能引入数值偏差甚至训练异常 | 先 pad vocab 再切分 |
| pad 行初始化后未清零 | 即使不更新，pad 向量也可能不是期望值 | 显式 `weight[pad_id].zero_()` |
| 输出层加 bias 但未考虑绑定 | 参数统计和实现理解混乱 | 大模型通常 `bias=False` |

先看 `padding_idx`。PyTorch 的语义是：指定该索引后，这一行不会对梯度有贡献，新建层时该行默认是全零。白话讲，这一行存在是为了占位，不是为了表达语义。如果你忘了设它，模型会把 pad 也当正常 token 训练。序列越短、pad 越多，这种噪声越明显。

再看词表对齐。真实工程里，多卡训练常常要求词表大小能被 TP 大小整除。假设 `tp_size=4`，原始词表是 `50000`。如果每卡要均分输出 logits 维度，那么最简单的要求是词表能均匀拆开；很多实现还会进一步 pad 到 64 或 128 的倍数。比如 GPT-2 原始 BPE 词表是 `50257`，很多实现会补到 `50304`，因为：

$$
50304 = 128 \times 393
$$

这让张量并行和 kernel 实现都更顺手。

这里要纠正一个常见误解：并不是“任何不能整除 TP 的词表都一定无法训练”，而是很多并行实现为了简化分片和 kernel，会强依赖或强偏好这种对齐；不对齐时，轻则实现复杂，重则数值行为和单卡不一致。工程上最稳妥的选择就是补齐。

真实工程例子：在 GPT-2 124M 复现中，常见配置是 `vocab_size=50304, d_model=768`，输入 embedding 与 LM Head 共享权重，并配合 `std=0.02` 初始化。这种组合背后的思路不是玄学，而是三点一起成立：

- 词表维度对 kernel/并行友好
- 输入输出共享参数，减少显存和参数量
- 初始尺度较小，早期 logits 不至于过大

---

## 替代方案与适用边界

weight tying 很主流，但不是绝对正确。它的前提是假设“输入端 token 表示”和“输出端 token 判别模板”应该落在同一个空间里。这个假设对标准自回归语言模型通常成立，但对某些任务未必最优。

下面是一个简化对比：

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Tied embedding | 标准语言模型、参数敏感场景 | 参数少、输入输出空间一致 | 限制输入输出解耦 |
| Untied embedding | 多任务、多语言、特殊输出头 | 表达更灵活 | 参数翻倍，语义空间可能漂移 |

什么情况下可以不绑？

第一类是多任务模型。比如输入是自然语言，输出却不完全是自然语言 token，而是任务标签、结构化符号或受约束的动作空间。这时输入和输出的统计结构不同，共享矩阵可能反而成为约束。

第二类是多语言或多模态场景。比如一个模型输入可以是多语言文本或图像离散码，但输出只面向某个特定词表。此时输入 embedding 需要兼容更广泛的分布，而输出层只服务于一部分目标空间，解耦更合理。

第三类是使用特殊 embedding 结构的系统，例如分组 embedding、哈希 embedding、稀疏 embedding 或 vector quantization。它们的共同点是为了降低参数、提高吞吐，或者引入额外离散约束。这些方案可以在超大词表或推荐系统里很有价值，但它们的目标与标准 LLM 的“输入输出共享语义空间”并不完全一致。

因此，是否 tied，不是审美问题，而是任务边界问题：

- 如果你在做标准 Transformer 语言模型，优先考虑 tied。
- 如果输入输出语义空间明显不同，untied 往往更稳妥。
- 如果主要瓶颈是超大词表存储，而不是语言建模语义一致性，可以考虑更激进的 embedding 压缩方案。

---

## 参考资料

- [PyTorch `nn.Embedding` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html)：查表语义、`padding_idx` 的梯度行为、pad 行默认初始化。
- [PyTorch `F.embedding` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.embedding)：函数式接口，对 `padding_idx` 的说明更直接。
- [Press & Wolf, 2017, Using the Output Embedding to Improve Language Models](https://aclanthology.org/E17-2025/)：weight tying 的经典论文，核心结论是输入 embedding 和输出 embedding 绑定可减少参数并改善语言模型表现。
- [vLLM `VocabParallelEmbedding` 文档](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/vocab_parallel_embedding/)：词表维度并行切分、padding 到可分片尺寸、LoRA 额外 token 的排布方式。
- [Karpathy GPT-2 复现讲解整理](https://tuananhbui89.github.io/blog/2025/karpathy-lec10/)：从实现角度解释 GPT-2 中的 weight tying、初始化和 `50304` 这类 padded vocab 设计。
- [Megatron-LM 相关 issue：非整除词表下的 TP 行为](https://github.com/nvidia/megatron-lm/issues/1754)：帮助理解为什么工程上常要求词表大小与 tensor parallel 规模对齐。

建议阅读顺序：

1. 先看 PyTorch `nn.Embedding`，明确“它就是查表”。
2. 再看 Press & Wolf，理解为什么输入输出可以共用一张表。
3. 然后看 Karpathy 或 vLLM 的实现资料，把“数学形式”对上“工程代码”。
4. 最后再看并行相关文档，理解为什么真实系统里还要处理 vocab padding。

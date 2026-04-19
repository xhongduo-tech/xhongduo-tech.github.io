## 核心结论

XLNet 是一种排列语言模型：它不改变句子里的真实词序，而是在训练时改变“预测这些词的顺序”，让模型在自回归框架里学习双向上下文。

它的核心不是“把句子打乱”。原句 `A B C` 仍然保持 `A B C`。被改变的是概率分解顺序。例如先预测 `B`，再预测 `A`，最后预测 `C`，对应的是：

$$
p(B) \cdot p(A \mid B) \cdot p(C \mid B, A)
$$

这样在预测 `A` 时，模型已经见过 `B`。如果换成另一个顺序，比如先预测 `A`，再预测 `C`，最后预测 `B`，对应：

$$
p(A) \cdot p(C \mid A) \cdot p(B \mid A, C)
$$

这时 `B` 又能同时利用左侧 `A` 和右侧 `C`。同一个 token 在不同排列中看到不同上下文，双向建模就来自这里。

XLNet 试图统一两类预训练模型的优点：自回归语言模型能建模完整联合概率，自编码模型能看双向上下文。BERT 用 `[MASK]` 遮住词再预测，XLNet 不遮输入词，而是通过排列预测顺序让每个位置都有机会从左右两侧获得信息。

| 模型 | 训练方式 | 能否看双向上下文 | 主要问题 |
|---|---|---:|---|
| 传统自回归 LM | 从左到右预测下一个词 | 否 | 右侧上下文不可见 |
| BERT | 遮住部分词再预测 | 是 | `[MASK]` 只在预训练出现，微调时没有 |
| XLNet | 采样不同预测顺序 | 是 | 实现复杂，训练成本更高 |

排列语言模型与掩码语言模型的区别可以压缩成一句话：BERT 改输入，XLNet 改预测顺序。

| 对比项 | 掩码语言模型 MLM | 排列语言模型 PLM |
|---|---|---|
| 代表模型 | BERT | XLNet |
| 输入是否被破坏 | 是，部分 token 替换为 `[MASK]` | 否，原始 token 保留 |
| 双向信息来源 | 被遮词两侧都可见 | 不同预测排列覆盖不同上下文 |
| 概率结构 | 更偏向条件独立的局部预测 | 保留自回归因子分解 |
| 典型风险 | 预训练与微调不一致 | mask 与两流注意力实现复杂 |

---

## 问题定义与边界

XLNet 要解决的问题是：如何在不丢掉自回归建模优势的前提下，让模型同时利用左侧和右侧上下文。

术语先说明。自回归语言模型是指把一句话的概率拆成一串条件概率，例如从左到右预测：

$$
p(x_1, x_2, x_3) = p(x_1)p(x_2 \mid x_1)p(x_3 \mid x_1, x_2)
$$

这种方式清晰、稳定，可以建模完整联合概率，但每一步只能看到前文。自编码模型是指模型根据被破坏或部分隐藏的输入恢复原内容，BERT 就是典型代表。它能利用左右上下文，但训练时引入了 `[MASK]`，而真实下游任务通常没有这个特殊符号。

新手版可以这样理解：BERT 会先把词遮住再猜；XLNet 不遮住词，而是换不同的猜词顺序，让每个词都有机会从左边和右边学习信息。

| 路线 | 解决什么 | 缺什么 |
|---|---|---|
| 传统自回归模型 | 联合概率建模清楚，适合生成 | 只能单向看上下文 |
| BERT | 双向理解能力强，适合分类、抽取 | `[MASK]` 带来预训练与微调差异 |
| XLNet | 在自回归框架中引入双向上下文 | 训练和实现明显更复杂 |

边界也要说清楚。

XLNet 适合做语言理解预训练，尤其是需要长上下文、概率分解结构、文档级理解的任务。它不是“通用替代所有双向模型”，也不是简单的“BERT 升级版”。如果任务只是短文本分类，现代 encoder 模型通常更直接。如果工程团队主要追求易部署、易微调、推理吞吐，XLNet 的复杂结构未必划算。

一个真实工程例子是合同审阅。合同条款经常跨段引用，模型需要理解前文定义、后文限制条件和当前条款之间的关系。XLNet 继承 Transformer-XL 的分段循环机制，可以缓存前一段内容的表示，对长文档抽取式问答或文档分类有实际意义。

---

## 核心机制与推导

XLNet 的核心目标是对所有可能的预测排列做期望最大化。设输入序列为 $x_{1:n}$，排列为 $z$，$z_t$ 表示第 $t$ 个被预测的位置，则训练目标为：

$$
L(\theta) = \mathbb{E}_{z \sim Z_n}\left[\sum_{t=1}^{n}\log p_\theta(x_{z_t} \mid x_{z_{<t}})\right]
$$

白话解释：每次训练先抽一个预测顺序，再让模型按这个顺序逐个预测 token。长期来看，每个 token 都会在不同训练样本中看到不同方向的上下文。

玩具例子：序列是 `A, B, C`。

| 排列 | 概率分解 | 说明 |
|---|---|---|
| `(2,1,3)` | `p(B) · p(A|B) · p(C|B,A)` | 预测 `A` 时可以利用右侧 `B` |
| `(1,3,2)` | `p(A) · p(C|A) · p(B|A,C)` | 预测 `B` 时可以利用左右两侧 |
| `(3,2,1)` | `p(C) · p(B|C) · p(A|C,B)` | 预测 `B` 时可以利用右侧 `C` |

这里的“排列”不是把输入变成 `B A C` 或 `A C B`。原始位置仍然保留，改变的是注意力允许访问哪些位置，也就是“当前预测能看哪些上下文”。

这会带来一个关键问题：如果模型在预测位置 `i` 的 token 时，输入表示里已经包含了 `x_i` 本身，就会目标泄露。目标泄露是指模型提前看到了要预测的答案，导致训练损失虚假变低，但学不到真正的语言规律。

XLNet 用两流自注意力解决这个问题。自注意力是 Transformer 中让一个位置从其他位置聚合信息的机制。两流指两套隐藏状态：`content stream` 和 `query stream`。

| 流 | 记号 | 能看什么 | 作用 |
|---|---|---|---|
| content stream | `h^c` | 当前 token 内容和允许的上下文 | 给后续位置提供内容信息 |
| query stream | `h^q` | 当前预测位置和允许的上下文，但不能看当前 token 内容 | 用来预测当前 token，防止目标泄露 |

可以把它理解为：`content stream` 负责“保存已经知道的词内容”，`query stream` 负责“站在某个位置上发问，但不能偷看答案”。最终预测某个 token 时，用的是 query stream 的表示，而不是已经包含答案的 content stream。

两流注意力示意表：

| 预测目标 | 允许上下文 | query stream 是否看目标词 | content stream 是否保存目标词 |
|---|---|---:|---:|
| `A` | `B` | 否 | 是 |
| `B` | `A, C` | 否 | 是 |
| `C` | `A, B` | 否 | 是 |

XLNet 还继承了 Transformer-XL 的分段循环机制。分段循环是指把长文本切成多个段，当前段处理时可以读取上一段缓存下来的隐藏状态。`mem` 就是这个缓存。

`mem` 机制流程：

```text
长文档 token
  ↓
切成 segment_1, segment_2, segment_3
  ↓
处理 segment_1，得到 hidden states
  ↓
截取一部分作为 mem
  ↓
处理 segment_2 时，把 mem 拼到当前上下文前面
  ↓
继续更新 mem，传给下一段
```

这让 XLNet 不必一次性把整篇长文放进模型，也能利用跨段上下文。配合相对位置编码，它能区分“两个 token 相隔多远”，而不是只依赖绝对下标。相对位置编码是指用位置之间的距离表示顺序关系，例如“当前词前面 3 个位置”，而不是“这是全局第 128 个词”。

---

## 代码实现

实现 XLNet 需要处理四件事：排列采样、两流注意力掩码、分段记忆、相对位置编码。完整模型很复杂，但可以先用一个简化版本理解“排列采样 + 损失计算”的核心。

训练伪代码如下：

```text
输入 token 序列 x
采样一个排列 z
根据 z 构造注意力 mask
构造 content stream 和 query stream
query stream 只拿位置，不拿目标 token 内容
Transformer 根据 mask 做注意力计算
对排列中需要预测的位置计算交叉熵损失
更新参数
```

新手版解释是：先决定今天让模型按什么顺序猜词，再把这个顺序变成注意力规则，最后只在允许看的信息范围内计算预测损失。

下面的 Python 代码不实现神经网络，只实现排列语言模型的“可见上下文”和一个玩具负对数似然计算。它可以直接运行，重点是看清楚排列如何改变条件上下文。

```python
from collections import defaultdict
import math

def contexts_from_permutation(tokens, perm):
    """返回每个目标 token 在该排列下能看到的上下文。perm 使用位置下标。"""
    seen = []
    result = {}
    for pos in perm:
        result[pos] = tuple(tokens[i] for i in seen)
        seen.append(pos)
    return result

def toy_probability(target, context):
    """一个玩具概率表：上下文越匹配，概率越高。"""
    rules = {
        ("A", ("B",)): 0.7,
        ("B", ("A", "C")): 0.8,
        ("C", ("B", "A")): 0.6,
    }
    return rules.get((target, context), 0.2)

def permutation_lm_loss(tokens, perm):
    contexts = contexts_from_permutation(tokens, perm)
    loss = 0.0
    for pos in perm:
        target = tokens[pos]
        context = contexts[pos]
        prob = toy_probability(target, context)
        loss += -math.log(prob)
    return loss, contexts

tokens = ["A", "B", "C"]

loss_213, ctx_213 = permutation_lm_loss(tokens, [1, 0, 2])
assert ctx_213[1] == ()
assert ctx_213[0] == ("B",)
assert ctx_213[2] == ("B", "A")

loss_132, ctx_132 = permutation_lm_loss(tokens, [0, 2, 1])
assert ctx_132[0] == ()
assert ctx_132[2] == ("A",)
assert ctx_132[1] == ("A", "C")

assert loss_213 > 0
assert loss_132 > 0
print(round(loss_213, 4), round(loss_132, 4))
```

掩码构造是工程实现里最容易错的部分。注意力 mask 是一个矩阵，用来规定“某个查询位置能不能看某个内容位置”。

| 目标位置 | 排列中已出现的位置 | 可见内容 | 不可见内容 |
|---|---|---|---|
| 第一个预测 token | 无 | 无或历史 `mem` | 当前目标及后续目标 |
| 中间预测 token | 排列中更早的位置 | 已预测 token、历史 `mem` | 当前目标内容、未来排列位置 |
| 最后预测 token | 前面所有排列位置 | 大部分上下文 | 当前目标内容 |

训练流程可以写成：

```text
tokens
  ↓
sample permutation z
  ↓
build attention mask from z
  ↓
content stream: token content + allowed context
query stream: position query + allowed context, no current token content
  ↓
Transformer-XL layers with mem
  ↓
logits for target positions
  ↓
cross entropy loss
```

真实代码中还会加入 `target_mapping`，它用于指定哪些位置需要被预测。论文实现里常只预测排列后半段，而不是每次预测全部 token，这样可以降低训练成本。

---

## 工程权衡与常见坑

XLNet 的收益来自机制组合，不是单独一个技巧。只说“随机排列”会误解它；只说“两流注意力”也不完整。它真正的结构是：排列目标负责引入双向上下文，两流注意力负责防止目标泄露，Transformer-XL 记忆机制负责长上下文。

常见坑如下：

| 现象 | 原因 | 规避方法 |
|---|---|---|
| 以为 XLNet 会打乱句子 | 混淆了词序和预测顺序 | 明确输入顺序不变，只改因子分解顺序 |
| 训练 loss 异常低 | query stream 偷看了目标 token | 检查 attention mask 和目标位置隔离 |
| 长文本训练显存爆掉 | `mem_len` 太大或 batch 太大 | 缩短 `mem_len`，降低 batch，使用梯度累积 |
| 速度明显下降 | 排列掩码和双流计算增加开销 | 控制 `perm_size`，只预测部分位置 |
| 微调效果不稳定 | 预训练目标和下游任务差异大 | 调整学习率、最大长度、层冻结策略 |
| 复现论文困难 | 官方实现细节多 | 先复现小数据流程，再扩大规模 |

几个关键参数需要重点看。

| 参数 | 含义 | 变大后的影响 | 风险 |
|---|---|---|---|
| `mem_len` | 缓存多少前文隐藏状态 | 可利用更长上下文 | 显存和计算增加 |
| `reuse_len` | 当前段中多少 token 会复用到后续段 | 提高跨段信息传递 | 设置不当会影响训练效率 |
| `perm_size` | 排列采样的局部窗口大小 | 上下文组合更多 | mask 更复杂，速度下降 |
| `seq_len` | 当前输入段长度 | 单段信息更多 | 显存增加 |
| `target_len` | 需要计算损失的目标长度 | 训练信号更多 | 计算成本增加 |

一个真实工程例子：在文档问答系统中，先用检索模型找到相关段落，再用 XLNet 类模型做答案抽取。假设合同定义在第 1 段，限制条件在第 8 段，问题问第 9 段中的责任范围。`mem_len` 增大后，模型更可能利用前文定义；但如果设得过大，训练速度会显著下降，显存也可能爆掉。新手可以理解为：缓存太多前文，机器记不住了。

另一个常见误区是把 `h^q` 理解成推理时必须保留的第二套完整表示。更准确地说，两流注意力主要服务于预训练目标，目的是在预测当前 token 时阻断目标内容。下游微调时如何使用表示，要看具体任务和实现。

---

## 替代方案与适用边界

如果任务主要依赖短文本理解，现代双向 encoder 通常更直接。它们训练和部署更成熟，生态工具更多。如果任务强调长上下文、生成式概率结构，或者需要研究自回归与双向理解的结合，XLNet 的思想仍然有参考价值。

| 模型路线 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| XLNet | 长文理解、抽取式问答、需要自回归概率结构的理解任务 | 双向上下文 + 自回归目标 + 长文本机制 | 训练复杂，推理和复现成本高 |
| BERT | 短文本分类、序列标注、抽取式问答 | 简单稳定，生态成熟 | `[MASK]` 带来预训练与微调差异 |
| 纯自回归 LM | 生成任务、下一个词预测 | 概率分解自然，适合生成 | 原始形式缺少右侧上下文 |
| 其他长上下文模型 | 长文摘要、长文问答、代码上下文 | 更适合现代长上下文硬件和任务 | 具体机制差异大，不能只看长度指标 |

文档问答场景中，XLNet 适合做“长文检索后抽取答案”的底座。例如先从几十页文档中检索出相关段落，再让模型在候选段落里定位答案。这类任务需要双向理解，也可能受益于跨段记忆。

但如果只是判断一句用户评论是正面还是负面，XLNet 未必比更简单的 encoder 模型划算。短文本分类通常更看重训练稳定性、推理速度和工具兼容性，而不是复杂的排列目标。

选择模型时可以按问题问自己三件事：

| 问题 | 倾向选择 |
|---|---|
| 是否主要做短文本理解？ | BERT 类 encoder |
| 是否需要生成或严格的自回归概率？ | 自回归 LM |
| 是否需要长上下文理解并希望保留自回归结构？ | XLNet 或借鉴其机制 |
| 是否只是追求工程效率？ | 优先选生态成熟的模型 |

---

## 参考资料

如果只读两篇论文，先读 XLNet，再读 Transformer-XL，因为 XLNet 的记忆机制和相对位置编码继承自后者。读代码时重点看排列目标、两流注意力和 `mem` 的实现位置。

| 机制 | 主要来源 | 对应内容 |
|---|---|---|
| 排列语言模型 | XLNet 论文 | permutation language modeling objective |
| 两流注意力 | XLNet 论文 | content stream 与 query stream |
| 分段循环记忆 | Transformer-XL 论文 | segment-level recurrence |
| 相对位置编码 | Transformer-XL 论文 | relative positional encoding |
| 工程实现 | XLNet 官方仓库 | 训练脚本、mask、memory 处理 |

1. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
2. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
3. [XLNet 官方代码仓库](https://github.com/zihangdai/xlnet)
4. [Transformer-XL 官方代码仓库](https://github.com/kimiyoung/transformer-xl)

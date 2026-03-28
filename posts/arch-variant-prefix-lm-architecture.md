## 核心结论

Prefix LM，中文可叫“前缀语言模型”，本质是给同一个 Transformer 施加一种混合注意力规则：输入部分叫 prefix，使用双向注意力；生成部分叫 suffix，使用因果注意力。双向注意力的白话解释是“这一段内部彼此都能看见”；因果注意力的白话解释是“当前位置只能看左边，不能偷看未来”。

它解决的问题不是“让模型更大”，而是“让模型在理解和生成之间做更合理的分工”。纯 Encoder-only 模型擅长读懂整段输入，但不适合自回归生成；纯 Decoder-only 模型能顺序生成，但对 prompt 的内部结构理解不如双向可见充分。Prefix LM 把两者拼起来：先把 prompt 当成“已知上下文”充分理解，再按左到右方式输出答案。

如果把总长度记为 $L$，前缀长度记为 $p$，常见的掩码矩阵 $M_{ij}$ 可以写成：

$$
M_{ij} =
\begin{cases}
1, & i < p,\ j < p \\
1, & i \ge p,\ j < p \\
1, & i \ge p,\ p \le j \le i \\
0, & \text{otherwise}
\end{cases}
$$

这三条分别对应：

1. prefix 内部全可见  
2. suffix 可以看见全部 prefix  
3. suffix 内部只能看见自己左边和自己

直观理解可以写成一句话：模型先“认真读懂你给的上下文”，再“按顺序写回答”。

在摘要、表格到文本、带说明的生成这类任务里，这种结构通常比纯 Decoder-only 更占优。公开结果里，GLM-130B 在一些摘要基准上相对同规模 GPT 风格模型取得了约 2 到 5 个 ROUGE 点的提升，说明“前缀双向理解”确实能转化成生成质量。

---

## 问题定义与边界

Prefix LM 讨论的是一种“同一模型内部的注意力可见性设计”，不是另一套完全独立的网络骨架。它通常仍然长得像 Decoder 栈，但 mask 不再是全局统一的下三角，而是“前半段双向，后半段因果”。

这里先明确边界：

| 架构 | 输入内部可见性 | 是否天然适合生成 | 典型用途 | 核心限制 |
|---|---|---|---|---|
| Encoder-only | 全双向 | 否 | 分类、检索、表征学习 | 不能直接做左到右生成 |
| Decoder-only | 全因果 | 是 | 对话、续写、代码生成 | 对输入理解依赖单向累积 |
| Prefix LM | prefix 双向，suffix 因果 | 是 | 摘要、条件生成、数据到文本 | prefix 处理成本更高 |
| Encoder-Decoder | encoder 双向，decoder 因果 | 是 | 翻译、摘要、结构化生成 | 系统复杂度更高 |

问题定义可以写成一句更工程化的话：给定一段条件输入 $x_{1:p}$ 和一段目标输出 $y_{1:T}$，希望模型既能充分利用条件输入内部的全局关系，又能保持自回归生成的训练和推理方式。

于是总序列通常被拼成：

$$
[x_1, x_2, \dots, x_p, y_1, y_2, \dots, y_T]
$$

其中 $L = p + T$。

“玩具例子”先看最简单的摘要任务：

- prefix：一段新闻正文
- suffix：一句摘要

这里 prefix 的价值很明显。新闻里前后句往往互相解释，模型如果让正文内部双向可见，就更容易找到主语、事件、时间和因果关系；而摘要输出仍然必须一个 token 一个 token 生成，所以 suffix 仍要因果掩码。

“真实工程例子”看电商表格转文案：

- prefix：商品标题、价格、规格、适用场景、禁用词要求
- suffix：生成营销文案

这类任务里，字段之间关系很强。比如“儿童可用”和“含酒精”不能同时出现，价格区间也会影响文案风格。如果只用纯 causal 方式顺着读，模型能做，但对字段间全局一致性的利用不如 prefix 双向可见充分。

需要注意的边界也很明确：

- 如果任务只需要生成，不太需要复杂理解，比如小说续写、长代码补全，纯 Decoder-only 往往更合适。
- 如果任务天然是“先编码输入，再单独解码输出”，比如高质量机器翻译，Encoder-Decoder 往往更直接。
- Prefix LM 最适合“一轮条件输入 + 一段输出”的场景，而不是特别长的多轮历史对话。

---

## 核心机制与推导

注意力，白话讲就是“每个位置决定该看哪些位置，以及看多大权重”。标准缩放点积注意力写成：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top + B}{\sqrt{d}}\right)V
$$

其中 $B$ 是 mask 对应的偏置矩阵。工程实现里，允许看的位置加 $0$，不允许看的位置加 $-\infty$，这样 softmax 后非法位置权重会变成 0。

Prefix LM 的关键不在公式本身，而在 $B$ 的形状。

设总长度 $L=6$，前缀长度 $p=3$。则 mask 可以写成：

| query\key | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| 2 | 1 | 1 | 1 | 0 | 0 | 0 |
| 3 | 1 | 1 | 1 | 1 | 0 | 0 |
| 4 | 1 | 1 | 1 | 1 | 1 | 0 |
| 5 | 1 | 1 | 1 | 1 | 1 | 1 |

逐行解释最容易看懂：

- 第 0 到 2 行是 prefix token，它们只能看 prefix 区域，但 prefix 内部彼此全可见。
- 第 3 行是第一个 suffix token，它能看全部 prefix，也能看自己，但不能看未来的 4、5。
- 第 4 行能看 prefix 和 suffix 中已经出现的 3、4。
- 第 5 行能看所有之前内容。

这比纯 causal mask 多出来的一块，就是左上角的 prefix 双向块。纯 Decoder-only 的左上角本来也只能是下三角，而 Prefix LM 把它放宽成了全 1。

这件事为什么有效？因为 prompt 的理解往往不是单向线性的。比如输入里前一句定义概念，后一句给约束条件，第三句给例外。双向可见后，prefix 中任意 token 的表示都能综合整段输入，而不是只依赖左边上下文。

从复杂度上看，Prefix LM 不是“更便宜”，而是“更合理地花算力”。

设 prefix 长度为 $p$，suffix 长度为 $s$，则一次完整前向中：

- prefix 内部注意力成本约为 $O(p^2)$
- suffix 对 prefix 的注意力成本约为 $O(sp)$
- suffix 内部因果注意力成本约为 $O(s^2)$

总量仍是二次型，只是可见性结构变了。

训练时这通常不是大问题，因为整段序列本来就要一起算。推理时差异才明显：

- 纯 causal LM 可以把历史 K/V 缓存起来，后续每一步只追加一个 token。
- Prefix LM 里，prefix 作为“特殊理解区”通常需要先完整计算，系统实现也更复杂。
- 如果 prompt 很长，prefix 部分就容易变成前向瓶颈。

这里再给一个更贴近工程的例子。做法律条款摘要时，输入常常包括：

- 条款正文
- 生效日期
- 例外条件
- 适用地区

这些信息在原文里相隔很远，但输出摘要要求它们被统一组织。Prefix LM 的优势就在于：模型在开始生成前，prefix 区域已经通过双向注意力把这些远距离关系整合到隐藏状态里了。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，只构造 Prefix LM 的 mask，并验证它满足规则。

```python
import numpy as np

def build_prefix_lm_mask(total_len: int, prefix_len: int) -> np.ndarray:
    assert 0 <= prefix_len <= total_len
    mask = np.zeros((total_len, total_len), dtype=bool)

    for i in range(total_len):
        for j in range(total_len):
            if i < prefix_len and j < prefix_len:
                mask[i, j] = True
            elif i >= prefix_len and j < prefix_len:
                mask[i, j] = True
            elif i >= prefix_len and prefix_len <= j <= i:
                mask[i, j] = True

    return mask

mask = build_prefix_lm_mask(total_len=6, prefix_len=3)

expected = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
], dtype=bool)

assert mask.shape == (6, 6)
assert np.array_equal(mask, expected)
assert mask[4, 5] == False   # suffix 不能看未来
assert mask[5, 2] == True    # suffix 可以看 prefix
assert mask[1, 2] == True    # prefix 内双向可见
print(mask.astype(int))
```

如果换成 PyTorch，核心逻辑也是一样，只是最终要把 `False` 的位置变成一个极小值偏置。伪代码如下：

```python
import torch

def build_prefix_lm_bias(total_len: int, prefix_len: int, device=None):
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    # prefix 内部全可见
    mask[:prefix_len, :prefix_len] = True

    # suffix 可以看全部 prefix
    mask[prefix_len:, :prefix_len] = True

    # suffix 内部因果可见
    causal = torch.tril(torch.ones(
        total_len - prefix_len,
        total_len - prefix_len,
        dtype=torch.bool,
        device=device
    ))
    mask[prefix_len:, prefix_len:] = causal

    # True 表示允许注意力，False 表示屏蔽
    bias = torch.zeros(total_len, total_len, device=device)
    bias = bias.masked_fill(~mask, float("-inf"))
    return bias

L, p = 8, 3
bias = build_prefix_lm_bias(L, p)
assert bias.shape == (8, 8)
assert torch.isinf(bias[0, 5])      # prefix 不能看 suffix
assert bias[6, 1] == 0              # suffix 能看 prefix
assert torch.isinf(bias[4, 7])      # suffix 不能看未来
```

在多头注意力里，这个 bias 一般会扩成：

$$
[batch,\ heads,\ L,\ L]
$$

或者直接利用广播机制加到 `attn_scores` 上。

生成时的实现要点比训练更重要：

1. prefix 先整体编码一遍，得到其各层 K/V  
2. suffix 第一步生成时，可以看全部 prefix K/V  
3. 之后每生成一个 token，只增量追加 suffix 自己的 K/V

这意味着缓存通常要分成两部分：

- prefix cache：固定不变，来自整段输入
- suffix cache：逐 token 追加，来自已生成输出

但要注意，很多实现里“prefix 无法像普通历史 token 那样免费得到”并不是说它不能缓存，而是说它的处理方式不如纯 causal 路径统一、简单，尤其在复杂 prompt 变换、packing、批量解码时更容易出现额外开销。

---

## 工程权衡与常见坑

Prefix LM 的最大优点是理解更强，最大代价是推理路径不如纯 causal 简洁。

先看常见风险和处理方式：

| 风险 | 现象 | 原因 | 常见对策 |
|---|---|---|---|
| prefix 重算或重处理成本高 | 首 token 延迟变大 | prefix 需要先完整建好表示 | 限制 prefix 长度，提前编码 |
| 显存增长 | 长 prompt 时注意力矩阵大 | prefix 内部是全连接 | 截断、分块、压缩历史 |
| 批处理效率下降 | 不同样本 prefix 长度差异大 | mask 结构不规则 | 按长度分桶 |
| 系统复杂度上升 | 训练和推理代码分叉 | 需要特殊 mask 与缓存逻辑 | 单独封装 mask builder |
| 长对话扩展性差 | 每轮历史都很长 | prefix 越长，理解成本越高 | 做摘要记忆或退回 causal |

新手最容易误解的一点是：“既然 suffix 还能缓存，那 Prefix LM 和 GPT 推理速度应该差不多吧？”不对。两者最常见的差异在首段处理成本。

一个直观例子：

- 纯 causal 模型处理 4k token 上下文时，本来也要先算一遍 prefill。
- Prefix LM 也要做类似 prefill，但 prefix 部分是更强的双向交互区。
- 当系统以“短输出、长输入”为主时，这个 prefill 成本占比更高，收益和代价都更明显。

“真实工程例子”是客服知识库摘要。假设把最近 4k token 对话记录作为 prefix，再让模型生成一段 80 token 的总结。此时用户真正感受到的往往不是“生成慢”，而是“开始出字前等待时间更长”。解决策略通常不是继续堆算力，而是控制 prefix 设计：

- 只保留最近最相关的 1k 到 1.5k token
- 把更早历史先摘要成几个 summary token
- 对长历史回退到纯 causal 解码

另一个坑是训练目标错配。Prefix LM 适合“输入条件 + 输出目标”式训练。如果拿纯无条件续写语料硬套，prefix/suffix 分界就会很随意，模型未必能学到稳定收益。它最适合那些天然有“先读后写”结构的数据。

还有一个常见实现问题是 mask 方向写反。判断标准很简单：

- prefix 不能看 suffix
- suffix 必须能看 prefix
- suffix 不能看未来 suffix

只要这三条里有一条不满足，模型行为就会偏。

---

## 替代方案与适用边界

Prefix LM 不是“全面替代 GPT”，而是一个更偏任务导向的折中方案。

| 方案 | 优势 | 劣势 | 更适合的任务 |
|---|---|---|---|
| 纯 causal decoder | 推理链路最简单，缓存友好 | 对复杂输入理解较弱 | 对话、续写、代码补全 |
| Prefix LM | 输入理解和生成能力平衡较好 | 长 prefix 成本高 | 摘要、数据到文本、单轮条件生成 |
| Encoder-Decoder | 条件建模清晰，输入输出分工明确 | 系统更复杂，部署栈更重 | 翻译、高质量摘要、结构化生成 |

和 T5 的关系也需要说清。T5 是 Encoder-Decoder，不是 Prefix LM；但它的 span corruption，也就是“掩盖若干片段后再让 decoder 生成缺失内容”，在功能上体现了类似思想：输入侧负责充分理解，输出侧负责因果生成。两者不是同一个结构，但在“理解区”和“生成区”分工上有相通之处。

什么时候优先选 Prefix LM？

- 输入信息密度高，字段关系复杂
- 输出长度相对较短
- 任务强调“先理解条件，再生成答案”
- 希望保留 decoder 风格的统一生成接口

什么时候不要优先选？

- 多轮长对话，历史越来越长
- 纯开放式生成，条件输入弱
- 对低延迟要求极高
- 系统已经围绕纯 causal cache 深度优化

可以把选择规则记成一句话：

- 如果任务像“读完材料再写结论”，Prefix LM 值得考虑。
- 如果任务像“顺着上下文一直续写”，纯 causal 往往更省事。
- 如果任务像“输入和输出天然是两个模块”，Encoder-Decoder 更直接。

一个很典型的对比是摘要与聊天：

- 摘要任务里，prefix 很像原文材料，suffix 很像摘要结果，Prefix LM 结构天然匹配。
- 长聊天里，所有历史都会不断增长，prefix 成本会持续放大，这时纯 causal 常常更合适。

---

## 参考资料

- [GLM-130B: An Open Bilingual Pre-trained Model](https://ar5iv.org/pdf/2210.02414)  
  用途：支持 Prefix LM 在摘要、NLG 等基准上的效果比较。文中可看到 GLM-130B 在 GEM、WikiLingua 等任务上相对 GPT 风格模型的提升，例如 WikiLingua 上 ROUGE-L 从 16.4 到 23.4 的对比。

- [Prefix Language Modeling: Combining Bidirectional Context with Causal Generation](https://mbrenndoerfer.com/writing/prefix-language-modeling-bidirectional-causal-generation?utm_source=openai)  
  用途：解释 Prefix LM 的核心 mask 结构，适合理解“前缀双向、后缀因果”的注意力可见性设计。

- [A Review of Current Trends in Large Language Models](https://www.mdpi.com/2076-3417/14/5/2074?utm_source=openai)  
  用途：给出 Prefix LM 在大模型架构谱系中的定位，说明它介于 Encoder-only 与 Decoder-only 之间。

- [Prefix LM in Practice](https://haileyschoelkopf.github.io/blog/2024/prefix-lm/?utm_source=openai)  
  用途：补充推理阶段的工程代价，尤其是 prefix 处理、缓存路径和长 prompt 成本问题。

- [UniLM: Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)  
  用途：说明“通过统一 mask 设计同时覆盖理解与生成任务”的代表思路。UniLM 不是只做 Prefix LM，但它把多种可见性模式统一到了一个预训练框架里。

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)  
  用途：帮助理解 T5 的 span corruption 与“先理解输入，再因果生成缺失片段”之间的关系，适合拿来与 Prefix LM 做边界比较。

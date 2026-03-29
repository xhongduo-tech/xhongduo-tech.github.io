## 核心结论

自回归语言模型（Autoregressive Language Model，简称 AR LM，白话讲就是“按从左到右顺序，一个 token 一个 token 往下接”的模型）本质上不是先“理解整句话”再一次性输出，而是把整个序列的联合概率拆成一串条件概率：

$$
P(x_1,x_2,\dots,x_n)=\prod_{t=1}^{n}P(x_t \mid x_1,\dots,x_{t-1})
$$

这条公式就是链式法则在语言建模里的直接应用。它说明一件事：模型在位置 $t$ 预测当前 token 时，只允许依赖前缀 $x_{<t}$，不允许读取未来 token。

这也是 GPT 一类模型的数学定义。它们通过因果掩码（causal mask，白话讲就是“强制右边信息不可见的遮罩”）把注意力矩阵的未来位置屏蔽掉，使训练时和生成时遵守同一条约束。训练目标通常写成：

$$
\max_\theta \sum_{t=1}^{n}\log P_\theta(x_t \mid x_{<t})
$$

等价地，也可以说是在最小化逐 token 的交叉熵损失。

最重要的工程含义有三点：

| 结论 | 含义 |
|---|---|
| 只能看左侧前缀 | 天然适合生成，不能直接偷看未来 |
| 概率按链式法则分解 | 可以逐 token 采样、贪心或 beam search |
| 推理必须串行 | 长文本生成延迟高，需要 KV cache 等优化 |

玩具例子可以直接看一句很短的话：“我 喜欢”。自回归模型此时只能基于“我 喜欢”预测下一个 token，可能是“学习”“你”“编程”，但不能先看后面已经存在的标准答案再回头预测。这个限制不是训练技巧，而是模型定义本身。

---

## 问题定义与边界

语言模型要解决的问题可以写成：给定前缀序列 $x_{<t}$，估计下一个 token 的条件概率分布 $P(x_t\mid x_{<t})$。这里 token（词元，白话讲就是“模型切分文本后处理的最小符号单位”）可能是一个字、一个词、一个子词，具体取决于分词方式。

如果把整个句子写成 $x_{1:n}$，自回归模型的输入输出边界非常明确：

- 输入：当前位置左边已经出现的 token
- 输出：当前位置 token 的概率分布
- 禁止：使用当前位置右边的未来信息
- 训练与推理约束一致：训练时不能看未来，生成时当然也不能看未来

这和“我已经知道完整句子，只是做标注”是两回事。自回归模型不是双向阅读器，而是单向生成器。

下面这个边界表最关键：

| 位置 | 可见信息 | 不可见信息 | 目标 |
|---|---|---|---|
| 第 1 个 token | 起始符 `<bos>` | 全部正文 | 预测 $x_1$ |
| 第 2 个 token | `<bos>, x_1$ | $x_3,\dots,x_n$ | 预测 $x_2$ |
| 第 $t$ 个 token | $x_1,\dots,x_{t-1}$ | $x_{t+1},\dots,x_n$ | 预测 $x_t$ |

这里有一个容易混淆的点：训练时我们手里明明有完整句子，为什么还说“不能看未来”？因为训练数据完整，不代表模型计算图允许访问未来。Transformer decoder 会在注意力层加下三角掩码，只让每个位置关注自己和左边。

一个最小例子：

- 序列：`<bos> 今 天 下 雨`
- 在预测“天”时，只能使用 `<bos> 今`
- 在预测“下”时，只能使用 `<bos> 今 天`
- 在预测“雨”时，只能使用 `<bos> 今 天 下`

所以，自回归的“问题定义”决定了它擅长续写、对话、代码补全这类左到右展开任务；它不擅长直接做“同时利用左右文做判别”的任务。

---

## 核心机制与推导

先从概率推导开始。任意序列联合概率都可以按链式法则展开：

$$
P(x_{1:n}) = P(x_1)P(x_2\mid x_1)P(x_3\mid x_1,x_2)\cdots P(x_n\mid x_{1:n-1})
$$

这不是 GPT 独有技巧，而是概率论基本定理。GPT 的关键在于：它把这个分解当成训练目标和生成规则。

取对数后，乘积变求和：

$$
\log P(x_{1:n})=\sum_{t=1}^{n}\log P(x_t\mid x_{<t})
$$

最大化整句概率，就等价于最大化每一步“下一个 token 的对数概率”之和。因为训练集里真实 token 已知，这个目标又等价于最小化交叉熵：

$$
\mathcal{L}=-\sum_{t=1}^{n}\log P_\theta(x_t\mid x_{<t})
$$

交叉熵（白话讲就是“真实答案在模型预测分布里有多不确定”）越小，说明模型越把真实下一个 token 赋予高概率。

接着看注意力如何实现“不能看未来”。标准缩放点积注意力可以写成：

$$
\text{Attention}(Q,K,V)=\text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中 $M$ 是掩码矩阵。自回归模型里，$M$ 使用因果掩码：

$$
M_{ij}=
\begin{cases}
0, & j\le i \\
-\infty, & j>i
\end{cases}
$$

意思是：第 $i$ 个位置只能看第 $1$ 到第 $i$ 个位置，所有未来位置 $j>i$ 的分数直接变成 $-\infty$。Softmax 后这些位置的权重就是 0。

一个 4 个位置的掩码可以写成：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

这就是常说的“下三角 mask”。

玩具例子可以把序列设成 `<bos> a b c`。它的联合概率分解是：

$$
P(<bos>,a,b,c)=P(<bos>)P(a\mid<bos>)P(b\mid<bos>,a)P(c\mid<bos>,a,b)
$$

如果模型给出：

- $P(a\mid<bos>)=0.6$
- $P(b\mid<bos>,a)=0.5$
- $P(c\mid<bos>,a,b)=0.2$

那么后三步的联合概率就是：

$$
0.6\times 0.5\times 0.2 = 0.06
$$

这个例子说明两件事：

1. 每一步只基于前缀更新。
2. 整句概率是逐步累计出来的，不是一次性算出的。

真实工程例子是代码补全。Copilot 或类似系统在你输入：

```text
def add(a, b):
    return
```

时，模型预测接下来最可能的 token 序列，可能先输出空格，再输出 `a`，再输出 `+`，再输出 `b`。每一步都只基于你已经输入的代码和刚生成的 token。它不能“先知道完整答案是 `a + b` 再回填”，因为生成机制就是左到右展开。

---

## 代码实现

下面用一个最小可运行 Python 例子演示两件事：如何构造因果掩码，以及如何验证联合概率等于条件概率乘积。

```python
import math

def build_causal_mask(seq_len: int):
    # True 表示这个位置应该被屏蔽
    mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(j > i)
        mask.append(row)
    return mask

def joint_prob_from_chain(cond_probs):
    p = 1.0
    for value in cond_probs:
        p *= value
    return p

mask = build_causal_mask(4)
expected = [
    [False, True,  True,  True],
    [False, False, True,  True],
    [False, False, False, True],
    [False, False, False, False],
]
assert mask == expected

# 玩具例子：P(a|<bos>)=0.6, P(b|<bos>,a)=0.5, P(c|<bos>,a,b)=0.2
p = joint_prob_from_chain([0.6, 0.5, 0.2])
assert abs(p - 0.06) < 1e-9

log_p = sum(math.log(x) for x in [0.6, 0.5, 0.2])
assert abs(log_p - math.log(0.06)) < 1e-9

print("causal mask and chain rule checks passed")
```

如果换成工程里更常见的深度学习伪代码，核心逻辑通常是这样：

```python
import torch

def causal_mask(seq_len: int, device=None):
    # 上三角为 True，表示未来位置不可见
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

def masked_self_attention_scores(scores: torch.Tensor):
    # scores shape: [batch, heads, seq_len, seq_len]
    seq_len = scores.size(-1)
    mask = causal_mask(seq_len, device=scores.device)
    return scores.masked_fill(mask, float("-inf"))
```

这里的 `scores` 是注意力打分矩阵，shape 一般是 `[B, H, T, T]`：

| 张量 | 形状 | 含义 |
|---|---|---|
| `Q` | `[B, H, T, D]` | query |
| `K` | `[B, H, T, D]` | key |
| `V` | `[B, H, T, D]` | value |
| `scores` | `[B, H, T, T]` | 每个位置对所有位置的注意力分数 |
| `mask` | `[T, T]` | 因果可见性约束 |

真实工程里还会加入 KV cache。KV cache（键值缓存，白话讲就是“把历史 token 计算过的 K、V 存起来，下一个 token 不再重复算”）的作用是降低推理成本。因为生成第 $t$ 个 token 时，前 $t-1$ 个 token 的表示已经算过，如果每次都从头重算，复杂度和延迟都会明显上升。

---

## 工程权衡与常见坑

自回归模型的优点很清晰：定义干净、训练目标直接、生成流程自然。但它的代价也很明确。

第一类问题是表示能力边界。因为每个位置只能看到左边，上下文编码天然受限。对于“需要同时利用左右文判断”的任务，比如完形填空、命名实体识别中的局部歧义消解，纯自回归通常不如双向模型。

第二类问题是推理串行。第 $t+1$ 个 token 必须等第 $t$ 个 token 生成完才能继续，这意味着延迟会随着输出长度增长。训练时可以并行计算所有位置的损失，推理时却不能完全并行。

第三类问题是 exposure bias。这个术语白话讲就是“训练时总喂真实前缀，推理时却要吃自己生成的前缀，二者分布不完全一样”。结果是前面一步错了，后面容易连锁偏移。

常见问题与规避方式可以总结成表：

| 问题 | 原因 | 常见规避手段 |
|---|---|---|
| 理解类任务偏弱 | 只能看左文 | 用双向编码器，或用 instruction tuning / RAG 补充信息 |
| 长文本生成慢 | 推理串行 | KV cache、speculative decoding、模型压缩 |
| 显存占用高 | 长上下文 K/V 累积 | 分页注意力、滑窗注意力、量化 cache |
| 错误逐步放大 | 生成依赖历史输出 | 采样策略控制、拒绝采样、后验重排 |
| 训练和推理不完全一致 | exposure bias | scheduled sampling 类方法，或靠大规模数据弱化问题 |

真实工程例子是对话系统。聊天模型在多轮对话中会不断把历史消息拼到上下文中，然后继续按自回归方式生成。如果不做 KV cache，每输出一个新 token 都要重算整段历史，成本非常高；如果上下文过长，缓存本身又会占用大量显存。因此线上系统常常同时做三件事：压缩上下文、缓存历史 K/V、控制最大输出长度。

一个常见坑是把“训练时并行”误解为“模型不是自回归”。训练阶段虽然可以把整段序列一次送入 Transformer，但每个位置依然被 mask 限制，只能依赖左边。并行的是硬件计算，不是概率依赖关系。

---

## 替代方案与适用边界

最常见的替代方案是 MLM，即 Masked Language Model，掩码语言模型，白话讲就是“把句子里某些词遮住，再让模型根据左右文一起猜出来”。BERT 属于这一类。

它和自回归的根本区别不是网络名字，而是条件结构不同：

| 方案 | 看到的信息 | 典型目标 | 优势 | 局限 |
|---|---|---|---|---|
| 自回归 LM | 只看左侧前缀 | 预测下一个 token | 生成自然，概率定义完整 | 推理串行，双向理解弱 |
| MLM | 看左右文但遮住当前位置 | 恢复被 mask 的 token | 理解类任务强 | 不能直接左到右生成 |
| 非自回归生成 | 尝试并行预测多个位置 | 一次或少次生成整句 | 速度快 | 质量和一致性常受损 |

为什么说 MLM 不能直接用于生成？因为它的训练问题不是“给前缀预测下一个 token”，而是“给部分缺失的序列恢复被遮住的位置”。这让它更适合理解和表征学习，而不是无缝续写。

非自回归模型则试图并行生成多个位置，以换取速度。但自然语言强依赖前后关系，并行生成容易出现重复、漏词或全局不一致，所以在开放式文本生成里通常仍不如强自回归模型稳定。

因此，选择边界可以概括为：

- 需要开放式生成、对话、代码补全、长文本续写：优先自回归
- 需要句子理解、分类、检索编码、局部判别：优先双向模型或编码器
- 极度关注延迟、允许一定质量折中：考虑非自回归或混合式方案

从数学上说，自回归模型最重要的价值不是“它会说话”，而是它对序列概率给出了一种严格、可训练、可采样的分解方式。这套分解天然连接了训练目标、注意力约束和生成过程，所以 GPT 一类系统能在同一套框架里完成预训练、续写和交互生成。

---

## 参考资料

1. MachinaLearning, *Autoregressive Models*  
   重点：链式法则分解、条件概率形式、训练目标定义。

2. Artificial-Intelligence-Wiki, *GPT and Decoder Models*  
   重点：Decoder-only 结构、训练时只看前缀、训练和推理一致性。

3. 1Cademy, *Step-by-step Example of Auto-Regressive Sequence Generation*  
   重点：逐步生成的最小例子，帮助理解联合概率如何累积。

4. Avichala, *Causal Attention Explained Simply*  
   重点：因果注意力、三角掩码、KV cache 与实际生成系统。

5. EmergentMind, *Masked Language Modeling (MLM)*  
   重点：MLM 的目标函数、与自回归模型在信息可见性上的根本区别。

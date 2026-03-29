## 核心结论

因果语言模型，英文常写作 Causal Language Model，简称 CLM。白话说，它是一种只能按“已经看到的内容”来预测下一个词的语言模型。

它的训练目标是把整段文本拆成一连串“根据前文预测当前词”的小任务。形式上：

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

这条公式的意思很直接：一句话的总概率，等于每个位置在“只看左边内容”条件下的条件概率连乘。

CLM 的关键不是“名字里有 causal”，而是它在注意力层用了严格的下三角掩码。掩码，mask，白话说就是“哪些位置允许看、哪些位置强制不许看”的规则。对位置 $t$ 来说，它只能访问 $\le t$ 的 token，不能访问未来 token，所以训练目标和推理过程天然一致。

这带来两个直接后果：

| 结论 | 含义 | 工程影响 |
| --- | --- | --- |
| 训练是自回归的 | 自回归，autoregressive，白话说就是“一个词接一个词往后写” | 适合续写、对话、代码补全 |
| 每个位置都能计算损失 | 损失，loss，白话说就是“模型在这个位置错了多少” | 训练信号密度高 |
| 只能看左侧上下文 | 上下文，context，白话说就是“模型当前可依赖的信息” | 理解类任务通常弱于双向模型 |

因此，GPT 系列本质上都是 CLM。它们在生成任务上天然占优，因为任务形式和模型结构完全一致；但在纯理解型任务上，同规模下通常不如 BERT 这类 MLM。

---

## 问题定义与边界

要理解 CLM，先要明确它解决的不是“看完整句话再判断”，而是“在当前位置继续往后写”。

问题定义可以写成：

给定前缀 $x_{<t}$，预测当前位置 $x_t$ 的分布。

这和 MLM，Masked Language Model，掩码语言模型，有根本差别。MLM 的白话解释是：把一句话里部分词遮住，再让模型根据左右两边一起恢复被遮住的词。BERT 就属于这一类。

一个最小玩具例子就能说明边界。

句子是：

`今天天气很好`

如果要预测“很”这个位置：

- 在 CLM 里，模型只能看到 `今天 天气` 以及此前已经生成出的内容，不能看右边的“好”。
- 在 MLM 里，如果把“很”遮住，模型可以同时看左边的“天气”和右边的“好”。

这意味着两类模型优化的是不同问题：

| 模型 | 可见上下文 | 训练问题 | 天然擅长 |
| --- | --- | --- | --- |
| CLM | 仅左侧前缀 | 预测下一个 token | 文本生成、对话、代码补全 |
| MLM | 左右双向 | 恢复被遮蔽 token | 分类、匹配、阅读理解 |

token，词元，白话说就是模型实际处理的最小文本单位，可能是字、词、子词，甚至标点。

所以 CLM 的边界非常清楚：它不是“更通用的语言模型”，而是“更贴近生成过程的语言模型”。如果任务要求模型严格按时间顺序展开内容，比如写回复、续写文章、补全代码，那么 CLM 是结构上正确的选择。反过来，如果任务核心是“看完整句后做判断”，比如情感分类、句子匹配、自然语言推断，那么纯 CLM 往往不是最优。

用户常见误解是：“语言模型都能做分类，所以 CLM 也应该一样强。”这不成立。能做，不代表训练目标最适合做。CLM 可以通过 prompt 或增加任务头完成理解任务，但其基础表示是单向约束的，这个约束不会自动消失。

---

## 核心机制与推导

CLM 的核心是链式分解和因果注意力。

先看目标函数。负对数似然，negative log-likelihood，白话说就是“如果真实答案概率越低，惩罚越大”。CLM 的训练损失通常写成：

$$
\mathcal{L}=-\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})
$$

这表示整段文本中，每个位置都参与训练。相比 MLM 通常只在被 mask 的一小部分位置上算损失，CLM 的训练利用率更高。

再看注意力。注意力，attention，白话说就是“当前位置应该把多少注意力分给哪些历史位置”。标准 self-attention 会先计算分数矩阵：

$$
\text{scores}_{t,u}=\frac{Q_tK_u^\top}{\sqrt{d}}
$$

其中 $Q$ 是 query，查询向量；$K$ 是 key，键向量；$d$ 是维度。白话说，这一步是在衡量“位置 $t$ 要不要关注位置 $u$”。

CLM 的关键修改是加入 causal mask：

$$
\text{scores}_{t,u}=\frac{Q_tK_u^\top}{\sqrt{d}}+\begin{cases}
0,& u\leq t\\
-\infty,& u>t
\end{cases}
$$

为什么是 $-\infty$？因为 softmax 之后，$e^{-\infty}=0$，未来位置的注意力权重就会变成 0，相当于完全不可见。

玩具例子：序列 `[A, B, C, D]`。

如果当前处理位置是 `C`，也就是第 3 个 token，那么 mask 后允许访问的是 `A, B, C`，禁止访问的是 `D`。分数矩阵对应这一行会变成：

$$
[\text{score}(C,A),\ \text{score}(C,B),\ \text{score}(C,C),\ -\infty]
$$

softmax 后，最后一个位置权重必为 0。于是模型在生成 `C` 相关表示时，不会偷看未来的 `D`。

这个约束解决了一个很重要的问题：训练和推理一致。

- 训练时，模型只能看历史 token。
- 推理时，模型本来也只有历史 token。

如果训练时允许偷看未来，推理时却看不到，就会产生暴露偏差之外更直接的目标错位。CLM 用结构约束避免了这个问题。

还可以从矩阵形状理解下三角 mask。长度为 4 时，mask 大致是：

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
$$

第 $t$ 行前 $t$ 列可见，后面全部不可见。这就是“下三角”名字的来源。

真实工程例子是代码补全。用户输入：

```text
def is_even(n):
    return
```

CLM 在生成 `n % 2 == 0` 时，只能依赖已经出现的函数名、参数名和前面代码。这和真实补全场景完全一致，因为编辑器里未来代码本来就不存在。反过来，MLM 依赖左右文恢复中间空位，这和“向右生成”的交互式补全并不完全一致。

---

## 代码实现

下面用纯 Python 写一个最小可运行例子，演示“下三角 mask 会屏蔽未来位置”。这个例子不依赖深度学习框架，但逻辑和 Transformer 一致。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def causal_mask(seq_len):
    return [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]

def apply_causal_mask(scores):
    n = len(scores)
    mask = causal_mask(n)
    masked = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(scores[i][j] if mask[i][j] == 1 else float("-inf"))
        masked.append(row)
    return masked

scores = [
    [1.0, 2.0, 3.0, 4.0],
    [1.5, 2.5, 3.5, 4.5],
    [2.0, 3.0, 4.0, 5.0],
    [2.5, 3.5, 4.5, 5.5],
]

masked = apply_causal_mask(scores)

# 第 3 行表示位置 2（从 0 开始）只能看见 0,1,2，看不见未来位置 3
assert masked[2][0] == 2.0
assert masked[2][1] == 3.0
assert masked[2][2] == 4.0
assert masked[2][3] == float("-inf")

weights = softmax(masked[2])

# softmax 后未来位置权重为 0
assert abs(weights[3] - 0.0) < 1e-12
assert abs(sum(weights) - 1.0) < 1e-12

print("causal mask works")
```

如果换成 PyTorch，核心逻辑通常只有几行：

```python
import math
import torch

def causal_attention_scores(Q, K):
    seq_len = Q.size(-2)
    head_dim = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return scores
```

这里的 `torch.tril` 就是“取下三角”。`masked_fill` 的含义是：凡是不允许看的位置，直接填成负无穷。

真实工程中，如果你用 Hugging Face 微调 GPT 类模型，通常不需要自己手写 mask，因为 `AutoModelForCausalLM` 已经内置了因果注意力约束。典型流程是：

1. 用 tokenizer 把文本切成 token。
2. 把 `input_ids` 右移一位后作为预测目标。
3. 使用 `AutoModelForCausalLM` 训练。
4. 推理时逐步生成，并缓存 KV。

KV cache，键值缓存，白话说就是“把前面算过的注意力中间结果存起来，后续生成时不用重复算”。它不改变模型结果，只是减少重复计算，所以在长文本生成里非常关键。

---

## 工程权衡与常见坑

CLM 最大的工程优势是训练目标简单，而且每个位置都有监督信号。假设序列长度是 $T$，那么一次前向通常能为接近 $T-1$ 个位置提供 loss；而 MLM 如果只 mask 15% token，那么大部分位置虽然参与了前向计算，却不直接贡献损失。

这就是“训练利用率”的差别。

但高利用率不等于对所有任务都更强。CLM 的根本限制是单向上下文。这个限制在生成里是优点，在理解里往往是缺点。

可以把常见权衡整理成表：

| 维度 | CLM | MLM |
| --- | --- | --- |
| 训练目标 | 下一 token 预测 | 被遮蔽 token 恢复 |
| 损失覆盖率 | 高，几乎每个位置都算 | 低，只在被 mask 位置算 |
| 上下文视野 | 左侧单向 | 双向 |
| 生成能力 | 强，天然匹配 | 弱，通常需改造 |
| 理解能力 | 同规模下常偏弱 | 通常更强 |

几个常见坑比理论本身更重要。

第一，很多人把 teacher forcing 和“模型看见未来”混为一谈。teacher forcing，教师强制，白话说就是训练时把真实前缀喂给模型，而不是喂模型自己上一步生成的结果。它不等于允许模型偷看未来。CLM 在 teacher forcing 下依然只能看左边，只是左边来自真实序列。

第二，数据构造时标签对齐经常出错。CLM 的标准做法不是“输入预测自己”，而是“输入预测右移一位后的目标”。如果对齐错了，loss 会看起来正常下降，但模型学到的是无意义映射。

第三，padding 处理容易和 causal mask 混淆。padding，填充，白话说就是为了把不同长度样本补成同一长度加入的占位符。实际训练里常常同时需要两种 mask：
- causal mask：禁止看未来
- padding mask：禁止看补齐出来的空位

只加前者不加后者，模型会把 padding 当成真实上下文；只加后者不加前者，又会破坏因果约束。

第四，不要把 CLM 在 NLU 上的劣势理解为“完全不能做理解”。更准确的说法是：同规模、同训练预算下，纯 CLM 在很多理解 benchmark 上通常低于双向 MLM，大致常见差距可到 5 到 10 个点，小模型上更明显。但如果模型足够大、数据足够多，或者通过 instruction tuning、任务格式改造，也能获得可用结果。

真实工程例子：做客服问答机器人时，底座模型通常选 CLM，因为它要持续生成回复；但如果还要做“工单分类”“意图识别”“相似问题匹配”，团队常常会额外部署一个双向编码器，或者采用混合训练目标，而不是指望一个纯 CLM 同时在所有理解任务上最优。

---

## 替代方案与适用边界

如果任务既要生成，又要强理解，只用纯 CLM 往往不够。这时常见替代方案有三类。

第一类是 MLM。它适合离线理解任务，比如文本分类、检索重排、句子匹配。它的问题不是能力弱，而是生成方式不自然，不适合直接做连续文本输出。

第二类是 encoder-decoder。编码器-解码器结构，白话说就是“前半部分负责双向理解输入，后半部分负责自回归生成输出”。T5、BART 属于这类。它适合翻译、摘要、问答这类“输入明确、输出也要生成”的任务。

第三类是混合目标训练。常见形式是：

$$
L_{\text{mix}}=\lambda L_{\text{CLM}}+(1-\lambda)L_{\text{MLM}},\quad \lambda\in[0,1]
$$

这里 $\lambda$ 表示 CLM 部分权重。$\lambda$ 越大，训练越偏向生成；越小，越偏向双向理解。

它的意义很实际：不是在“选边站”，而是在算力预算有限时，把不同能力合并到一个训练流程里。比如先用 CLM 进行 warm-up，让模型快速获得语言建模和生成能力；再插入 MLM 阶段，补足双向语义建模。也有一些工作会直接交替 batch 或交替阶段训练。

适用边界可以总结为：

| 场景 | 更合适的方案 | 原因 |
| --- | --- | --- |
| 聊天、写作、代码补全 | CLM | 推理过程就是逐 token 生成 |
| 情感分类、句子匹配 | MLM/双向编码器 | 需要完整上下文做判断 |
| 摘要、翻译、问答生成 | Encoder-decoder 或强 CLM | 既要理解输入又要生成输出 |
| 一套模型兼顾生成与理解 | 混合目标 | 用训练目标折中能力分布 |

因此，CLM 不是“语言模型的终极形式”，而是“在生成任务上最符合问题结构的形式”。如果你的任务本质是向右展开文本，优先选 CLM；如果你的任务本质是看完整信息再判断，优先考虑双向模型；如果两者都要，就用混合目标或分工架构。

---

## 参考资料

- Causal Language Models in NLP: https://www.emergentmind.com/topics/causal-language-models-clms
- Understanding Attention: A Code-First Journey Through Transformers: https://www.gada.space/posts/transformers-from-scratch/
- Hugging Face Transformers, Causal language modeling: https://huggingface.co/docs/transformers/v4.33.2/tasks/language_modeling
- Causal language modeling vs Masked language modeling: https://www.aimodels.fyi/research-topics/causal-language-modeling
- Should You Mask 15% in Masked Language Modeling?: https://www.emergentmind.com/papers/2202.08005
- Hybrid Masked-Causal Language Modeling: https://www.emergentmind.com/topics/hybrid-masked-causal-language-modeling
- AntLM: Bridging Causal and Masked Language Models: https://www.emergentmind.com/papers/2412.03275

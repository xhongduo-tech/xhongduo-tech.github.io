## 核心结论

Seq2Seq，完整写法是 Sequence-to-Sequence，指“序列到序列”模型：给定一个输入序列 $x_{1:T}$，模型按顺序生成一个输出序列 $y_{1:U}$。

它的核心是条件生成：

$$
p(y_{1:U}\mid x_{1:T})=\prod_{u=1}^{U}p(y_u\mid y_{<u},x_{1:T})
$$

白话解释：模型不是一次性吐出完整答案，而是在已生成前文的基础上，一个 token 一个 token 地生成后续内容。token 是文本模型处理的基本单位，可以是字、词或子词。

最经典的 Seq2Seq 结构由两部分组成：

```text
输入序列 -> 编码器 -> 表示向量 -> 解码器 -> 输出序列
```

编码器负责“读入输入”，解码器负责“生成输出”。新手可以先把它理解成“先读完再写出”：例如机器翻译时，模型先读完整句中文，再逐词生成英文。

| 版本 | 编码器输出如何被使用 | 优点 | 主要问题 |
|---|---|---|---|
| vanilla Seq2Seq | 通常只把最后隐状态 $h_T$ 作为固定长度向量 | 结构简单，适合入门 | 长输入信息容易被压丢 |
| Attention Seq2Seq | 解码每一步都重新查看所有输入隐状态 | 能处理更长输入，具备软对齐能力 | 计算更复杂，实现要处理 mask |
| Transformer | 不使用 RNN，主要依赖 self-attention 并行建模 | 长距离依赖和并行训练更强 | 对数据、算力和工程实现要求更高 |

vanilla Seq2Seq 的问题在于“固定长度瓶颈”：无论输入是 5 个词还是 100 个词，都要压进一个向量。注意力机制 attention 的作用是让解码器每一步都能回看输入的不同位置，而不是只依赖最后一个压缩状态。

---

## 问题定义与边界

Seq2Seq 解决的是映射问题：

$$
x_{1:T} \rightarrow y_{1:U}
$$

其中 $T$ 是输入长度，$U$ 是输出长度。二者不必相等。比如中文句子“我想取消订单”可以翻译成英文 “I want to cancel the order”，输入输出长度不同，但语义相关。

| 项目 | 含义 | 是否固定 |
|---|---|---|
| 输入 $x_{1:T}$ | 原始序列，如中文句子、语音特征、工单文本 | 不固定 |
| 输出 $y_{1:U}$ | 目标序列，如英文句子、摘要、回复文本 | 不固定 |
| 输入输出长度 | $T$ 和 $U$ 可以不同 | 不固定 |
| 对齐关系 | 输出某一步可能对应输入某几个位置 | 不一定显式给出 |

玩具例子：输入是 `["我", "爱", "机器学习"]`，输出是 `["I", "love", "machine", "learning"]`。输入有 3 个单位，输出有 4 个单位，Seq2Seq 允许这种长度变化。

真实工程例子：客服自动回复系统。输入是一段中文工单：“订单 8291 已付款，但地址填错了，希望改到上海市徐汇区。”输出可能是：“已为你记录修改地址请求，请确认新地址后等待客服处理。”输入和输出长度不固定，但输出必须受输入语义约束。

Seq2Seq 适合以下任务：

| 任务 | 输入 | 输出 | 是否常需要对齐 |
|---|---|---|---|
| 机器翻译 | 源语言句子 | 目标语言句子 | 是 |
| 文本摘要 | 长文本 | 短摘要 | 是 |
| 对话生成 | 用户问题 | 回复 | 不一定 |
| 文本纠错 | 含错误句子 | 修正句子 | 是 |
| 语音识别 | 声学特征序列 | 文本序列 | 是 |

边界也很明确：如果输入特别长，例如整篇合同、长篇论文、完整对话历史，vanilla Seq2Seq 的单向量表示会成为瓶颈。此时通常需要 attention、coverage、copy mechanism，或者直接换成 Transformer。coverage 是一种记录“哪些输入位置已经被关注过”的机制，用来减少重复生成和漏译。

---

## 核心机制与推导

先看 vanilla Seq2Seq。编码器通常是 RNN、LSTM 或 GRU。RNN 是循环神经网络，会按时间步处理序列，并把历史信息保存在隐状态中。隐状态可以理解为模型读到当前位置后形成的内部记忆。

编码器递推公式是：

$$
h_t=f_{enc}(x_t,h_{t-1})
$$

其中 $h_t$ 是第 $t$ 个输入位置的编码器隐状态。vanilla Seq2Seq 常用最后一个隐状态作为整段输入的表示：

$$
c=h_T
$$

然后解码器基于 $c$ 逐步生成：

$$
s_u=f_{dec}(y_{u-1},s_{u-1},c)
$$

$$
p(y_u\mid y_{<u},x)=softmax(W_o s_u)
$$

softmax 是把一组分数转成概率分布的函数，所有概率相加为 1。

LSTM 和 GRU 是经典的编码器、解码器选择。LSTM，长短期记忆网络，通过门控机制控制信息保留和遗忘：

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\quad h_t=o_t\odot tanh(c_t)
$$

GRU，门控循环单元，结构比 LSTM 更简洁：

$$
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}
$$

这里的“门”是取值在 0 到 1 之间的控制量，用来决定信息通过多少。

vanilla 结构的问题是：$c=h_T$ 必须承载全部输入信息。长句中前面的数字、专名、否定词很容易被压弱。Attention Seq2Seq 改成在每个解码步 $u$ 都计算一次上下文向量 $c_u$。

注意力打分：

$$
e_{u,t}=v^T tanh(W_s s_{u-1}+W_h h_t)
$$

归一化权重：

$$
\alpha_{u,t}=softmax_t(e_{u,t})
$$

上下文向量：

$$
c_u=\sum_t \alpha_{u,t}h_t
$$

输出分布：

$$
p(y_u\mid y_{<u},x)=softmax(W_o[s_u;c_u])
$$

其中 $\alpha_{u,t}$ 表示生成第 $u$ 个输出时，模型对第 $t$ 个输入位置的关注程度。$[s_u;c_u]$ 表示把解码器状态和上下文向量拼接起来。

| 解码步骤 | 使用的信息 | 计算结果 | 含义 |
|---|---|---|---|
| 1. 取状态 | $s_{u-1}$ 和所有 $h_t$ | 当前生成状态与输入表示 | 准备判断该看哪里 |
| 2. 打分 | $e_{u,t}$ | 每个输入位置一个分数 | 分数越高越相关 |
| 3. softmax | $\alpha_{u,t}$ | 权重分布 | 权重和为 1 |
| 4. 加权求和 | $c_u=\sum_t\alpha_{u,t}h_t$ | 当前上下文向量 | 本步重点参考的信息 |
| 5. 生成 | $softmax(W_o[s_u;c_u])$ | 下一个 token 概率 | 选择或采样输出 |

数值例子：设编码器输出 3 个状态：

$$
h_1=[1,0],\quad h_2=[0,2],\quad h_3=[1,1]
$$

某一步 attention 打分为 $e=[1,2,0]$，softmax 后约为：

$$
\alpha=[0.245,0.665,0.090]
$$

则：

$$
c_u=0.245h_1+0.665h_2+0.090h_3\approx[0.335,1.420]
$$

这说明模型主要关注第 2 个输入位置。新手可以把它理解为：做摘要时，不是只给整篇文章打一个“总分”，而是每写一句都回到原文，重新找当前最相关的位置。

---

## 代码实现

最小工程结构通常包括：

| 模块 | 作用 |
|---|---|
| `Encoder` | 把输入 token 序列编码成隐状态序列 |
| `Attention` | 根据解码器状态计算输入位置权重 |
| `Decoder` | 基于前一个输出和上下文生成下一个 token |
| `Seq2Seq.forward()` | 训练阶段串联编码器和解码器 |
| `train_step()` | 计算 loss、反向传播、更新参数 |
| `greedy_decode()` | 推理时每步选择概率最大的 token |
| `beam_search_decode()` | 保留多个候选序列，提高生成稳定性 |

下面是一个不依赖深度学习框架的 attention 玩具实现，用来验证“打分、softmax、加权求和”的核心逻辑：

```python
import math

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [v / total for v in exps]

def weighted_sum(weights, vectors):
    dim = len(vectors[0])
    return [
        sum(w * vec[i] for w, vec in zip(weights, vectors))
        for i in range(dim)
    ]

h = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
scores = [1.0, 2.0, 0.0]

alpha = softmax(scores)
context = weighted_sum(alpha, h)

assert abs(sum(alpha) - 1.0) < 1e-9
assert alpha[1] > alpha[0] > alpha[2]
assert round(context[0], 3) == 0.335
assert round(context[1], 3) == 1.421
```

PyTorch 中，编码器常写成 `nn.LSTM` 或 `nn.GRU`。下面是结构级伪代码，重点展示数据流：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        x = self.emb(src)
        outputs, hidden = self.rnn(x)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ws = nn.Linear(hidden_dim, hidden_dim)
        self.wh = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_mask):
        # decoder_state: [batch, hidden]
        # encoder_outputs: [batch, src_len, hidden]
        score = self.v(torch.tanh(
            self.ws(decoder_state).unsqueeze(1) + self.wh(encoder_outputs)
        )).squeeze(-1)

        score = score.masked_fill(src_mask == 0, -1e9)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn
```

这里的 `src_mask` 是 attention mask。mask 是遮罩，用来告诉模型哪些位置有效、哪些位置是 padding。padding 是为了把不同长度句子补齐成同一长度而加入的占位 token。没有 mask 时，模型可能把注意力分配给无意义的 padding。

| 阶段 | 输入给解码器的前缀 | 常用策略 | 风险 |
|---|---|---|---|
| 训练 | 真值 token，如真实答案的前一个词 | teacher forcing | 与推理阶段不一致 |
| 推理 | 模型上一步自己生成的 token | greedy 或 beam search | 错误会逐步累积 |

teacher forcing 是训练时把标准答案前缀喂给解码器，帮助模型更快学习。beam search 是推理时同时保留多个候选输出，而不是每一步只选一个最大概率 token。

---

## 工程权衡与常见坑

Seq2Seq 的第一个坑是长序列瓶颈。没有 attention 时，模型把输入压到固定向量 $c=h_T$，短句可能够用，长句容易丢信息。机器翻译中，输入越长，vanilla Seq2Seq 的漏译和错译通常越明显。

第二个坑是训练和推理不一致。训练时模型看到真值前缀，像老师一步步给标准提示；推理时模型只能看自己刚生成的内容。如果前面一个词错了，后面的状态也会偏，最后出现“前面对、后面崩”。

第三个坑是专名和数字丢失。真实客服回复中，订单号、金额、地址、姓名不能随意改写。例如输入包含“订单 8291”，输出却生成“订单 8921”，这在业务上是严重错误。常见补救包括 BPE、SentencePiece、copy mechanism 和 pointer network。BPE 和 SentencePiece 是子词切分方法，可以把未知词拆成更小单位；copy mechanism 是允许模型直接从输入复制片段的机制。

| 坑 | 典型现象 | 原因 | 规避方案 |
|---|---|---|---|
| 长序列瓶颈 | 长句漏译、摘要丢重点 | 固定向量容量不足 | attention / coverage / Transformer |
| 训练推理不一致 | 生成后半段质量下降 | teacher forcing 与自回归推理不同 | scheduled sampling / beam search |
| 数字专名丢失 | 订单号、地址、人名被改写 | 词表外词和低频词难建模 | BPE / SentencePiece / copy mechanism |
| padding 干扰 | 注意力落到空白位置 | 未使用 attention mask | attention mask / loss mask |
| 重复输出 | “已处理已处理已处理” | 解码偏好高频短循环 | length penalty / coverage penalty |

scheduled sampling 是训练时有时喂真值 token，有时喂模型自己的预测，让训练更接近推理。loss mask 是计算损失时忽略 padding 位置，避免模型为了预测 padding 而浪费学习能力。length penalty 是对过短或过长输出做惩罚，coverage penalty 是惩罚反复关注同一输入位置的行为。

真实工程例子：客服自动回复生成。输入工单可能包含“订单号、退款原因、地址、时间、商品名”等结构化信息。一个普通 Seq2Seq 模型能生成流畅回复，但可能漏掉订单号，或把“退款”误写成“换货”。工程上通常会使用 attention mask 过滤 padding，用子词切分处理低频词，用 copy/pointer 保留关键字段，再用 beam search 提升候选质量。

| 故障现象 | 直接原因 | 工程解决方案 |
|---|---|---|
| 输出很流畅但事实错误 | 生成模型偏语言流畅性，不保证复制事实 | 加 copy mechanism，或把关键字段结构化输入 |
| 短输入正常，长输入崩 | 编码压缩瓶颈或 attention 不稳定 | 加 attention、coverage，或改 Transformer |
| 训练 loss 下降但线上效果差 | 暴露偏差和数据分布不一致 | scheduled sampling，构造更接近线上输入的数据 |
| 批训练时效果异常 | padding 参与 attention 或 loss | 同时检查 attention mask 和 loss mask |
| beam search 输出重复 | 候选序列偏向高频模式 | length penalty、去重约束、coverage penalty |

---

## 替代方案与适用边界

Seq2Seq 不是当前所有序列任务的首选。现代 NLP 中，大规模翻译、摘要、对话和长上下文任务通常优先使用 Transformer。Transformer 是基于注意力机制的序列模型，不依赖 RNN 的逐步递推，因此更适合并行训练和长距离依赖建模。

但 Seq2Seq 仍然有价值。它概念清晰，能把“输入一句，输出一句”的生成逻辑讲得很直接。对零基础到初级工程师来说，先理解 Seq2Seq，再理解 attention，最后过渡到 Transformer，是一条清晰路径。

| 模型 | 核心结构 | 优点 | 适用边界 |
|---|---|---|---|
| Seq2Seq RNN | 编码器最后状态传给解码器 | 简单，适合教学和短序列 | 长序列信息瓶颈明显 |
| Seq2Seq + attention | 解码每一步回看输入状态 | 具备软对齐，翻译和摘要更强 | 仍然逐步解码，训练并行性有限 |
| Transformer | self-attention + feed-forward | 并行训练强，长距离依赖更好 | 工程复杂度和资源需求更高 |

选择建议：

| 场景 | 建议 |
|---|---|
| 学习文本生成基本原理 | 先选 Seq2Seq RNN |
| 输入输出需要明显对齐，如翻译、纠错 | 至少使用 Seq2Seq + attention |
| 输入较长，包含大量跨段依赖 | 优先考虑 Transformer |
| 资源受限、小数据、结构明确 | Seq2Seq + GRU/LSTM 仍可作为基线 |
| 需要严格复制数字、专名、地址 | Seq2Seq 需配合 copy/pointer 或结构化字段 |

新手版判断方法：如果目标只是理解“输入一段序列，输出另一段序列”的基本生成逻辑，Seq2Seq 是最直接的起点。如果目标是做长文档摘要、多轮对话、超长上下文问答，通常应该直接学习 Transformer。

Seq2Seq 的历史价值也很重要。它把神经网络从固定类别分类，推进到可变长度生成；attention 又把固定向量瓶颈改成动态查看输入。理解这条线，有助于理解后来的编码器-解码器 Transformer、机器翻译系统和现代文本生成模型。

---

## 参考资料

1. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural)
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
3. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/)
4. [PyTorch nn.LSTM 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
5. [PyTorch nn.GRU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)

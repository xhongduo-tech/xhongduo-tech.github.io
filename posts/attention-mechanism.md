## 核心结论

注意力机制（Attention）是一种按相关性分配信息权重的方法。它的核心不是把整句压缩成一个固定向量，而是在每个生成步骤动态读取输入序列中最相关的部分。

在机器翻译中，早期 Seq2Seq 模型会先用编码器把源句压成一个向量，再让解码器逐词生成目标句。这个固定向量必须同时保存词义、语序、依赖关系和上下文，句子越长，信息越容易丢失。Attention 的做法是：解码器每生成一个目标词，都重新查看源句所有位置，并根据当前需要分配不同权重。

统一公式是：

$$
e_{t,i}=score(q_t,k_i),\quad
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})},\quad
c_t=\sum_i \alpha_{t,i}v_i
$$

其中 `query` 是查询向量，表示“当前要找什么信息”；`key` 是键向量，表示“每个输入位置可被匹配的特征”；`value` 是值向量，表示“真正被取走的信息”。在经典 Seq2Seq 翻译中，通常有 `q_t = decoder state`，`k_i = v_i = encoder hidden state`。

| 对比项 | 固定向量编码 | 动态注意力读取 |
|---|---|---|
| 信息来源 | 编码器最后状态 | 所有编码器状态 |
| 是否每步变化 | 否 | 是 |
| 长句表现 | 容易丢前半句信息 | 能按步骤回看源句 |
| 对齐能力 | 隐式、弱 | 显式、软对齐 |
| 典型问题 | 固定长度瓶颈 | 计算量随序列长度增长 |

玩具例子：如果输入是“猫 坐在 垫子上”，模型生成英文单词 `cat` 时应主要关注“猫”；生成 `mat` 时应主要关注“垫子”。它不是一次性记住整句话，而是每次生成前重新按当前目标词读取源句信息。

---

## 问题定义与边界

Attention 首先解决的是 Seq2Seq 的固定长度瓶颈。Seq2Seq 是序列到序列模型，白话说就是把一个序列转换成另一个序列，例如把英文句子翻译成德文句子。早期结构通常由编码器和解码器组成：编码器读完整个源句，输出一个固定长度向量；解码器只依赖这个向量逐词生成目标句。

问题在于，固定长度向量对长句不友好。短句“hello world”可能容易压缩；长句包含多个从句、修饰语和远距离依赖时，一个向量很难保留所有细节。尤其在翻译中，目标词的生成常常只依赖源句中的局部片段，而不是整句的同等信息。

Attention 解决的是“信息选择与对齐”。信息选择指模型知道当前步骤应该重点读取哪些输入位置；对齐指目标词和源词之间建立对应关系。这里的对齐是软对齐，意思是权重可以分散到多个位置，而不是只能选一个词。

| 对象 | 含义 | 是否动态 | 在 Seq2Seq 中的典型来源 |
|---|---|---:|---|
| `q_t` | 当前解码步的查询，表示当前要生成词时需要什么 | 是 | decoder state |
| `k_i` | 第 `i` 个源位置的匹配特征 | 否，随源句编码固定 | encoder hidden state |
| `v_i` | 第 `i` 个源位置可提供的信息内容 | 否，随源句编码固定 | encoder hidden state |
| `α_{t,i}` | 当前步对第 `i` 个源位置的注意力权重 | 是 | softmax(score) |
| `c_t` | 当前步汇总得到的上下文向量 | 是 | 加权求和 |

边界也要明确：Attention 不等于模型真正理解了因果关系，也不等于彻底解决长序列问题。它让模型更容易访问输入信息，但全局 Attention 仍然需要在每个目标步比较所有源位置。当输入很长时，计算和内存开销仍然明显上升。

真实工程例子：英德机器翻译中，decoder 生成德语动词时，可能需要同时关注英文主语、助动词和句尾成分。固定向量模型容易把早期信息压缩掉；Attention 允许每个目标词生成时重新访问源句所有 encoder states，使长句翻译更稳定。

---

## 核心机制与推导

Attention 的统一流程是：打分、归一化、加权求和。

第一步，计算相关性分数：

$$
e_{t,i}=score(q_t,k_i)
$$

`score` 是打分函数，白话说就是判断“当前查询”和“某个输入位置”有多匹配。分数越高，表示当前解码步越应该关注这个位置。

第二步，用 softmax 归一化：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

softmax 是把一组实数转换成概率分布的方法。它保证所有权重非负，并且总和为 1。这样模型可以进行可微的软选择：不是硬选一个位置，而是对所有位置分配权重。

第三步，对 value 加权求和：

$$
c_t=\sum_i \alpha_{t,i}v_i
$$

`c_t` 是上下文向量，表示当前解码步从源句读取到的信息。

逐步推导可以写成：

```text
decoder 当前状态 q_t
        ↓
与所有 encoder key 计算相关性分数 e_{t,i}
        ↓
softmax 得到注意力权重 α_{t,i}
        ↓
对所有 value 加权求和
        ↓
得到当前上下文向量 c_t
```

最小数值例子：设

```text
q=[1,0]
k1=[1,0], k2=[0,1]
v1=[2,0], v2=[0,4]
```

用点积打分：

$$
e_1=q^\top k_1=1,\quad e_2=q^\top k_2=0
$$

softmax 后：

$$
softmax([1,0])=[0.7311,0.2689]
$$

上下文向量为：

$$
c=0.7311[2,0]+0.2689[0,4]=[1.4622,1.0756]
$$

结论是：模型更像 `k1`，所以从 `v1` 拿更多信息；但它不会完全丢弃 `v2`。这就是软选择。

不同 Attention 的主要差别在打分函数。

Bahdanau Attention 也叫 additive attention，白话说就是用一个小型前馈网络计算 `query` 和 `key` 的匹配分数：

$$
score(q_t,k_i)=v_a^\top\tanh(W_q q_t+W_k k_i)
$$

Luong Attention 也叫 multiplicative attention，白话说就是用向量乘法或带参数矩阵的乘法计算相似度：

$$
score(q_t,k_i)=q_t^\top k_i \quad \text{或}\quad q_t^\top W k_i
$$

Transformer 把这个机制统一成 `Q-K-V` 的矩阵形式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里除以 $\sqrt{d_k}$ 是为了缩放点积分数。`d_k` 是 key 向量维度；维度越大，点积数值越可能变大，softmax 容易过早变尖，导致梯度不稳定。缩放可以让训练更平滑。

---

## 代码实现

下面是一个可运行的 PyTorch 单头 scaled dot-product attention。单头是指只用一组 `Q/K/V` 做注意力计算；多头注意力则是并行使用多组投影，让模型从不同子空间读取信息。

| 张量 | 形状 | 含义 |
|---|---|---|
| `Q` | `[batch, tgt_len, d_k]` | 目标序列每个位置的查询 |
| `K` | `[batch, src_len, d_k]` | 源序列每个位置的键 |
| `V` | `[batch, src_len, d_v]` | 源序列每个位置的值 |
| `weights` | `[batch, tgt_len, src_len]` | 每个目标位置对源位置的权重 |
| `context` | `[batch, tgt_len, d_v]` | 每个目标位置读到的上下文 |

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights, V)
    return context, weights

# 玩具例子：q=[1,0], k1=[1,0], k2=[0,1], v1=[2,0], v2=[0,4]
Q = torch.tensor([[[1.0, 0.0]]])          # [batch=1, tgt_len=1, d_k=2]
K = torch.tensor([[[1.0, 0.0],
                   [0.0, 1.0]]])         # [batch=1, src_len=2, d_k=2]
V = torch.tensor([[[2.0, 0.0],
                   [0.0, 4.0]]])         # [batch=1, src_len=2, d_v=2]

# 为了复现 softmax([1,0])，这里不用 sqrt(d_k) 缩放，单独写一个未缩放版本。
def dot_attention_no_scale(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    weights = F.softmax(scores, dim=-1)
    context = torch.matmul(weights, V)
    return context, weights

context, weights = dot_attention_no_scale(Q, K, V)

assert weights.shape == (1, 1, 2)
assert context.shape == (1, 1, 2)
assert torch.allclose(weights[0, 0], torch.tensor([0.7311, 0.2689]), atol=1e-4)
assert torch.allclose(context[0, 0], torch.tensor([1.4621, 1.0758]), atol=1e-4)

# padding mask 例子：第二个源位置是 padding，不允许被关注。
mask = torch.tensor([[[1, 0]]])
context_masked, weights_masked = attention(Q, K, V, mask=mask)

assert torch.allclose(weights_masked[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
assert torch.allclose(context_masked[0, 0], torch.tensor([2.0, 0.0]), atol=1e-6)
```

代码中最容易混淆的是矩阵维度。`QK^T` 的结果是 `[batch, tgt_len, src_len]`，表示每个目标位置对每个源位置的分数。softmax 必须沿最后一维做，因为要让每个目标位置在所有源位置上重新归一化。

---

## 工程权衡与常见坑

Attention 提高了信息读取能力，但不是没有成本。Global attention 会让每个目标位置都和所有源位置计算相关性。若目标长度为 $T$，源长度为 $S$，点积注意力的主要计算规模约为 $O(TS)$。在 Transformer 自注意力中，若序列长度为 $n$，复杂度通常是 $O(n^2)$。

| 坑 | 表现 | 原因 | 规避方法 |
|---|---|---|---|
| 把权重当解释 | 看到某词权重大，就说模型“因为它”才输出结果 | Attention 权重只是相关性分配，不是因果证明 | 只能作为辅助观察，不能单独作为解释结论 |
| 忘记每步重新归一化 | 不同解码步共用旧权重 | 每个目标词需要的信息不同 | 每个 `t` 都重新计算 score 和 softmax |
| Bahdanau 和 Luong 的时刻混淆 | 复现论文时结果对不上 | Bahdanau 常用前一解码状态，Luong 常用当前状态 | 明确公式中的 decoder state 来自哪个时间步 |
| 忽略 padding mask | 模型关注到补齐位置 | padding 没有真实语义 | 在 softmax 前把 padding 分数置为 `-inf` |
| 误以为 attention 解决所有长序列问题 | 长文档输入仍然慢、显存高 | 全局比较仍随长度增长 | 使用截断、local attention、sparse attention 或分块处理 |

真实工程例子：在翻译系统中，如果源句被 padding 到同一长度，而没有加 mask，模型可能会给 `<pad>` 分配注意力权重。这会污染上下文向量，导致目标词生成不稳定。padding mask 不是优化项，而是正确性要求。

另一个常见误解是把 attention heatmap 当成模型解释。某个源词权重高，只能说明当前输出和它在注意力计算中更相关，不代表它是唯一原因。模型后续还有非线性层、残差连接、前馈网络和输出层，最终预测不是由 attention 权重单独决定。

---

## 替代方案与适用边界

不同方案适合不同规模和任务。Attention 很重要，但不是所有问题都必须使用复杂注意力结构。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 固定向量 Seq2Seq | 结构简单，计算较省 | 长句信息瓶颈明显，对齐能力弱 | 短序列、简单转换任务 |
| Bahdanau attention | 对齐能力强，缓解固定长度瓶颈 | 前馈打分计算较重 | 经典 RNN Seq2Seq 翻译 |
| Luong attention | 点积形式更简洁，计算更快 | 表达能力依赖向量空间质量 | 需要较高效率的 RNN 翻译模型 |
| Transformer attention | 统一 `Q-K-V`，并行能力强 | 自注意力对长序列开销高 | 机器翻译、文本生成、编码建模 |
| local/sparse attention | 降低长序列计算和显存 | 可能漏掉远距离依赖 | 长文档、语音、长上下文任务 |

短序列任务中，简单 pooling 或 RNN 最后状态可能已经足够。例如判断一句很短的评论是正面还是负面，平均池化加分类器可能能达到可接受效果。引入 Attention 会增加实现复杂度和调参成本。

长序列任务中，经典 global attention 不一定合适。例如长文档问答、长音频识别、长上下文翻译，输入长度很大，全局回看虽然信息完整，但代价高。local attention 只看附近窗口，计算更省；sparse attention 只选择部分位置参与计算，适合更长上下文；分块检索则适合工程系统中先召回相关片段，再让模型精读。

边界可以简化为一句话：全局 Attention 稳定但贵，局部或稀疏 Attention 更省但可能漏信息。具体选择取决于序列长度、对齐复杂度、延迟预算和可接受的误差类型。

---

## 参考资料

1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)  
Bahdanau et al., 2015。提出 additive attention，用可学习打分网络缓解固定长度向量瓶颈。

2. [Effective Approaches to Attention-based Neural Machine Translation](https://aclanthology.org/D15-1166/)  
Luong et al., 2015。系统比较 multiplicative attention、global attention 和 local attention 等变体。

3. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)  
Vaswani et al., 2017。提出 Transformer，用 `Q-K-V` 和 scaled dot-product attention 统一注意力计算。

4. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)  
Harvard NLP。用代码解释 Transformer attention 的实现细节，适合对照公式理解张量形状。

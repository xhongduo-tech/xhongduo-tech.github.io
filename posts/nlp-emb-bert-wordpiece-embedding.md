## 核心结论

BERT 的“词片嵌入”不是单一向量，而是三部分信息的叠加结果：WordPiece 词表嵌入、可学习的位置嵌入、段嵌入。WordPiece 可以理解为“把词拆成更小、可复用的子词单位”；位置嵌入表示“这个 token 在序列里的第几个位置”；段嵌入表示“这个 token 属于句子 A 还是句子 B”。三者维度相同，逐元素相加后，再经过 LayerNorm 和 Dropout，作为 Transformer 的输入。

这件事的意义很直接：自注意力本身只擅长看“内容之间的关系”，不天然知道顺序，也不知道两个 token 是否来自不同句子。把位置和段身份编码进输入后，BERT 才能同时感知“是什么词”“在什么位置”“属于哪句话”。

玩具例子可以先看一句很短的话：“今天 我 看书”。如果分成三个词片，每个词片先查到一个 768 维 WordPiece 向量，再加上“第 1、2、3 个位置”的位置向量，以及“都属于第一句话”的段向量，那么模型在进入注意力层之前，就已经知道内容、顺序和句子身份。

| 组成部分 | 维度 | 来源 | 作用 |
| --- | --- | --- | --- |
| WordPiece embedding | $H$ | 词表查表 | 表示子词内容 |
| Position embedding | $H$ | 可学习参数表 | 表示绝对位置 |
| Segment embedding | $H$ | 可学习参数表 | 表示句子 A/B 身份 |

BERT Base 中 $H=768$。这意味着每个输入 token 的最终表示都是一个 768 维向量。注意，这里的位置嵌入不是正弦函数生成的固定编码，而是随机初始化、在预训练中学出来的参数。这一点和很多早期 Transformer 教材里的“固定正弦位置编码”不同，也和 ELMo 这种依赖 BiLSTM 顺序传播的模型不同。

---

## 问题定义与边界

问题可以表述成一句话：原始文本怎样变成 Transformer 能处理的数值输入？

BERT 的答案是先做子词切分，再构造带有特殊标记的 token 序列，然后把每个位置映射成统一维度的向量。这里有几个边界条件必须明确。

第一，BERT 处理的不是“词”，而是 WordPiece 子词。子词的白话解释是“比完整单词更细的小块”，这样可以减少未登录词问题。比如英文里的 `playing` 可能被切成 `play` 和 `##ing`。中文实现里有时一个字就是一个 token，但底层机制相同。

第二，输入序列长度有限。标准 BERT 的位置嵌入最多只学到 512 个位置，所以有效输入上限是 512。这个长度包括 `[CLS]`、`[SEP]` 这些特殊标记。`[CLS]` 可以理解为“整段输入的汇总占位符”；`[SEP]` 是“句子边界标记”。

第三，句对任务必须显式区分两句话。比如自然语言推断、问答匹配、句子对分类，通常输入格式是：

`[CLS] sentence_A [SEP] sentence_B [SEP]`

其中第一句所有 token 的段 ID 设为 0，第二句所有 token 的段 ID 设为 1。这样模型才能知道“哪些 token 属于 A，哪些属于 B”。

| 输入限制/元素 | 具体规则 | 作用 |
| --- | --- | --- |
| 最大长度 | 不超过 512 | 避免位置索引越界 |
| `[CLS]` | 放在最前面 | 汇聚全局语义，供分类头使用 |
| `[SEP]` | 句间和句尾插入 | 标记边界，支持句对/NSP |
| 段 ID 0 | 第一段 | 标记句子 A |
| 段 ID 1 | 第二段 | 标记句子 B |

新手最容易忽略的是：这些 embedding 参数表本身也是模型参数。词表嵌入矩阵大约是 $|V| \times H$，位置嵌入矩阵大约是 $512 \times H$，段嵌入矩阵是 $2 \times H$。其中词表矩阵最大。以 BERT Base 为例，若词表大小约 30k，则词表嵌入参数量约为：

$$
30000 \times 768 \approx 2.3 \times 10^7
$$

这部分参数大约占整个模型总参数的 22% 左右。也就是说，词表设计不是小事，改动词表往往意味着要付出很高的重新训练成本。

---

## 核心机制与推导

形式上，BERT 每个位置的输入表示可以写成：

$$
E_{\text{input}} = E_{\text{tok}} + E_{\text{pos}} + E_{\text{seg}}
$$

如果写成矩阵查表形式，可以表示为：

$$
E_{\text{word}}=[O_{\text{tok}}\;O_{\text{seg}}\;O_{\text{pos}}]
\begin{bmatrix}
W_{\text{tok}}\\
W_{\text{seg}}\\
W_{\text{pos}}
\end{bmatrix}
$$

其中：

- $W_{\text{tok}} \in \mathbb{R}^{|V| \times H}$ 是词表嵌入矩阵
- $W_{\text{seg}} \in \mathbb{R}^{2 \times H}$ 是段嵌入矩阵
- $W_{\text{pos}} \in \mathbb{R}^{P \times H}$ 是位置嵌入矩阵，标准 BERT 中 $P=512$

这里的 $O$ 可以理解为 one-hot 选择向量，也就是“我现在要从表里取哪一行”。因为三张表输出的都是 $H$ 维，所以才能直接逐元素相加。

玩具例子最容易理解。假设某个 token 的三种向量都是 3 维，分别为：

- 词片向量：$[0.10, 0.20, 0.30]$
- 位置向量：$[0.02, 0.01, 0.00]$
- 段向量：$[0.50, 0.50, 0.50]$

那么相加得到：

$$
[0.10,0.20,0.30] + [0.02,0.01,0.00] + [0.50,0.50,0.50]
= [0.62,0.71,0.80]
$$

这个结果表示：同一个位置上的最终数值，已经同时混入了内容、位置和句子身份。后续自注意力层看到的，就不是“纯词义向量”，而是“带上下文定位信息的输入向量”。

流程可以压缩成下面这一条：

`WordPiece lookup -> + Position -> + Segment -> LayerNorm -> Dropout -> Transformer`

这里的 LayerNorm 可以理解为“按特征维度做标准化，让数值分布更稳定”；Dropout 是“训练时随机丢弃一部分信息，减少过拟合”。它们不改变语义结构，但会影响训练稳定性和泛化能力。

真实工程例子是句对分类。假设任务是判断“问题”和“答案候选”是否匹配：

- 句子 A：`用户忘记密码怎么办`
- 句子 B：`可以通过邮箱重置密码`

BERT 输入会被构造成：

`[CLS] 用户 忘记 密码 怎么办 [SEP] 可以 通过 邮箱 重置 密码 [SEP]`

对应段 ID：

`0 0 0 0 0 0 1 1 1 1 1 1`

这样做以后，自注意力既能看见“密码”在两句里都出现，也知道前半段是问题、后半段是回答。最后拿 `[CLS]` 位置的最终隐藏状态送给分类头，预测“是否匹配”。`[CLS]` 可以理解为“代表整段输入的聚合位”，它不是天然有汇总能力，而是在预训练和下游训练中被学出来承担这个角色。

与 ELMo 的差别也在这里。ELMo 依赖双向 LSTM，LSTM 可以理解为“按顺序一步一步传递状态的循环网络”，所以位置信息主要通过传播顺序间接体现。BERT 则把位置信息直接塞进输入表示里，因此注意力层能并行处理所有位置，不必像 RNN 那样逐步推进。

---

## 代码实现

下面用一个可运行的 Python 玩具实现，把“词向量 + 位置向量 + 段向量 + LayerNorm”这个过程写清楚。代码不依赖深度学习框架，只演示核心机制。

```python
from math import sqrt

def layer_norm(vec, eps=1e-12):
    mean = sum(vec) / len(vec)
    var = sum((x - mean) ** 2 for x in vec) / len(vec)
    return [(x - mean) / sqrt(var + eps) for x in vec]

def add_vectors(a, b):
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]

def prepare_token_embeddings(token_ids, segment_ids, token_table, position_table, segment_table):
    assert len(token_ids) == len(segment_ids)
    assert len(token_ids) <= len(position_table)

    output = []
    for position_id, (token_id, segment_id) in enumerate(zip(token_ids, segment_ids)):
        token_vec = token_table[token_id]
        pos_vec = position_table[position_id]
        seg_vec = segment_table[segment_id]

        hidden = add_vectors(token_vec, pos_vec)
        hidden = add_vectors(hidden, seg_vec)
        hidden = layer_norm(hidden)
        output.append(hidden)
    return output

token_table = {
    0: [0.10, 0.20, 0.30],   # [CLS]
    1: [0.40, 0.10, 0.20],   # 今天
    2: [0.20, 0.50, 0.10],   # 我
    3: [0.30, 0.20, 0.60],   # 看书
}

position_table = [
    [0.01, 0.01, 0.01],
    [0.02, 0.01, 0.00],
    [0.03, 0.01, 0.00],
    [0.04, 0.01, 0.00],
]

segment_table = {
    0: [0.50, 0.50, 0.50],   # 句子A
    1: [0.80, 0.80, 0.80],   # 句子B
}

token_ids = [0, 1, 2, 3]
segment_ids = [0, 0, 0, 0]

embeddings = prepare_token_embeddings(
    token_ids, segment_ids, token_table, position_table, segment_table
)

assert len(embeddings) == 4
assert len(embeddings[0]) == 3
assert all(abs(sum(vec) / len(vec)) < 1e-9 for vec in embeddings)

print(embeddings)
```

上面这段代码里：

- `token_ids` 是词表索引列表
- `segment_ids` 是段 ID 列表
- `position_id` 直接由位置顺序生成
- 每个位置都先查三张表，再做向量相加和 LayerNorm

如果换成真实 BERT 的句对输入，构造顺序通常是：

```python
tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
position_ids = list(range(len(tokens)))
```

| 输入名 | 含义 | 长度 |
| --- | --- | --- |
| `token_ids` | 每个 token 在词表中的编号 | $L$ |
| `segment_ids` | 每个 token 属于句子 A/B 的编号 | $L$ |
| `position_ids` | 每个 token 的位置编号 | $L$ |
| 输出 hidden states | 每个 token 的最终输入向量 | $L \times H$ |

真实工程中，一般不会自己手写这段逻辑，而是由 tokenizer 和模型内部 embedding 层自动完成。但理解这段构造过程非常重要，因为很多输入错误都发生在这里，而不是发生在 Transformer 层内部。

---

## 工程权衡与常见坑

第一类问题是词表成本高。BERT Base 的词表嵌入矩阵非常大，约 30000 × 768。它的好处是表达能力强，覆盖面广；代价是参数重、迁移成本高。如果你随意改词表，比如增加大量行业专有词，旧的 embedding 权重就不能直接完整复用，通常需要重新训练或至少做额外适配。

第二类问题是位置上限。标准 BERT 只学到了前 512 个位置。如果把长文档直接塞进去，超过部分不是“效果变差”，而是直接没有合法位置索引可查。工程上常见做法是截断、分块、滑动窗口。

第三类问题是句对边界构造错误。很多人把两个句子拼起来，却忘了插 `[SEP]` 或把所有段 ID 都设成 0。这样模型还能跑，但它失去了“句子边界”和“句子身份”这两条重要信号，尤其在 NSP、自然语言推断、检索排序等任务里会明显掉点。

第四类问题是误解 `[CLS]`。`[CLS]` 不是“天然最重要的词”，而是“被训练成汇总整段信息的特殊位置”。如果你的下游任务适合 token 级输出，比如序列标注、命名实体识别，就不应该只盯着 `[CLS]`，而应使用每个 token 的对应隐藏状态。

| 常见坑 | 典型症状 | 原因 | 规避方法 |
| --- | --- | --- | --- |
| 超过 512 长度 | 报错或被强制截断 | 位置表只有 512 行 | 分块、滑窗、换长文本模型 |
| 漏掉 `[SEP]` | 句对任务效果差 | 边界信号缺失 | 严格按模板拼接 |
| 段 ID 全为 0 | A/B 不可区分 | 句子身份信息丢失 | 第二句统一设为 1 |
| 改词表后直接微调 | 收敛慢或性能差 | embedding 权重不匹配 | 重新预训练或做增量适配 |
| 只看 `[CLS]` 做所有任务 | token 级任务结果差 | 任务目标不匹配 | 按任务选输出位 |

真实工程例子是在搜索排序里做 query-document 匹配。若把 query 和 document 直接拼接，但不标段 ID，模型虽然还能通过词共现学到一些相关性，但它不知道“前面是查询，后面是文档”。一旦 query 和 document 中出现相同词，模型很难准确建模“跨段交互”和“段内结构”的区别，排序质量通常会受影响。

一个实用建议是：如果你必须加行业词，不要先改词表，先测试原始 WordPiece 能否通过子词组合覆盖需求。因为“词表更大”不等于“一定更好”，很多场景下原有子词拆分已经够用，而你真正需要优化的是训练数据和任务头设计。

---

## 替代方案与适用边界

BERT 这套输入嵌入设计很经典，但它并不覆盖所有场景。最明显的边界就是长文本。只要文档显著超过 512 token，标准 BERT 的绝对位置嵌入就会成为硬限制。

一种保守方案是 sliding window，也就是“滑动窗口分块”。白话说，就是把长文本切成多个长度不超过 512 的片段，分别编码，再把结果聚合。这种方法实现简单，兼容现有模型，但跨块信息建模有限。

另一类方案是专门面向长文本的模型，比如 Longformer、BigBird 等。它们通常通过稀疏注意力或可扩展位置机制，把计算和长度限制放宽。适合文档分类、长报告问答、合同审查这类任务。

再看历史对比：

| 模型 | 位置处理方式 | 是否易于并行 | 适合场景 |
| --- | --- | --- | --- |
| BERT | 可学习绝对位置嵌入 | 是 | 中短文本编码、分类、匹配 |
| ELMo | 无显式 Transformer 位置表，靠 BiLSTM 顺序传播 | 否 | 早期上下文化表示 |
| Transformer-XL | 相对位置编码 | 较强 | 更长上下文建模 |
| Longformer | 可扩展位置机制 + 稀疏注意力 | 是 | 长文档任务 |

ELMo 的边界在于顺序计算。它通过 BiLSTM 获取上下文，表达力并不低，但难以像 Transformer 那样高效并行训练。BERT 通过可学习位置嵌入解决“顺序信息如何注入”这个问题，但代价是长度上限固定。Transformer-XL 用相对位置编码，核心思路是“关注相对距离而不是绝对槽位”，因此更适合长依赖建模。

对于小模型或多语言场景，另一个边界是 embedding 参数过大。词表越大，词嵌入越重。此时可以考虑更小词表、共享输入输出 embedding、使用更激进的子词拆分，或者采用字节级方案来减少词表维护成本。但这通常会增加序列长度，属于典型的工程权衡，而不是无成本优化。

所以选择标准 BERT 的前提通常是：

- 输入长度大多在 512 以内
- 任务需要成熟稳定的双向编码器
- 句对建模需求明确
- 不希望为长文本或超大词表付出额外系统复杂度

---

## 参考资料

1. Devlin 等人的 BERT 原论文，提供了 token embedding、segment embedding、position embedding 的原始定义，也是理解 `[CLS]`、`[SEP]` 与输入表示构造的基础来源。
2. Stack Overflow 上关于 BERT token embedding 的讨论，重点解释了三类 embedding 相加、`[CLS]` 作为分类输入、`[SEP]` 作为边界标记的实现含义。
3. LLM Foundation 相关教材页面，给出了更形式化的矩阵表达，便于从“查表相加”上升到线性代数视角理解输入表示。
4. 面向初学者的 BERT token embedding 教程文章，强调了 WordPiece、位置、段三类信息如何共同进入自注意力层，适合建立直觉。
5. 关于 BERT Base 参数构成的工程资料，说明了词表嵌入矩阵在总参数中占比较高，这对词表修改成本和模型压缩都很关键。
6. 关于 BERT 位置嵌入是否可训练的技术讨论，帮助区分“固定正弦位置编码”和“可学习绝对位置嵌入”这两条路线。
7. CSDN 等工程总结文章，提供了长度上限、词表调整、句对输入格式等常见坑的经验性说明，适合作为落地实践补充。

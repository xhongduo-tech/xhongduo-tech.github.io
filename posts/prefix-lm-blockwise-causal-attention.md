## 核心结论

Prefix LM 可以用一句话定义：**把一个序列拆成“条件前缀”和“生成后缀”，前缀内部允许双向注意力，后缀仍按标准自回归方式从左到右生成**。这里的“双向注意力”就是“前缀里的 token 可以互相看见彼此，不受左到右顺序限制”；“自回归”就是“当前位置只能根据已经出现的内容预测下一个 token”。

它不是一种全新的网络骨架，更准确地说，它是**对 Transformer 注意力掩码和训练目标的重写**。模型层数、参数形状、解码接口通常都可以不变，真正变化的是“哪些位置允许互相看见”和“哪些位置参与 loss”。

这件事的价值在于，它把两类需求放进了同一个 decoder 风格模型里：

- 前缀负责条件建模，也就是把题目、检索材料、系统指令、表单字段这些已知信息先融合起来。
- 后缀负责条件生成，也就是继续用 next-token prediction 逐步写答案。

对零基础读者，最直观的理解是：**把输入分成“已知材料”和“要写的答案”**。已知材料内部可以来回查阅，答案部分只能一边看左边已经写好的内容一边往后写。

下面这个对比能先把位置关系看清：

| 目标 | 前缀可见性 | 后缀可见性 | 是否适合条件生成 | 是否保留标准自回归解码 |
|---|---|---|---|---|
| Causal LM | 仅左侧 | 仅左侧 | 较弱 | 是 |
| Prefix LM | 前缀双向 | 仅左侧 | 强 | 是 |
| Encoder-Decoder | 编码器双向 | 解码器仅左侧 | 强 | 不完全是同一条链路 |

如果任务是“读材料后回答”“根据指令补全文档”“RAG 问答”，Prefix LM 往往比纯 causal LM 更自然；如果任务只是“继续写下去”，它的优势会明显减弱。

---

## 问题定义与边界

先把问题形式化。设输入序列被拆成两段：

$$
x = [p_1, p_2, \dots, p_m, y_1, y_2, \dots, y_n]
$$

其中：

- $p$ 是 prefix，也就是前缀，含义是“已知条件”。
- $y$ 是 suffix，也就是后缀，含义是“需要模型生成的内容”。

术语第一次出现时可以这样记：

- **prefix**：前面那段给模型参考的材料。
- **suffix**：后面那段要求模型写出来的答案。
- **loss**：训练时的误差，数值越小表示预测越接近目标。

Prefix LM 的训练目标通常写成：

$$
\mathcal{L} = - \sum_{t=1}^{n}\log p(y_t \mid p, y_{<t})
$$

这条公式的意思很直接：模型训练时只负责预测后缀 token，条件是“整个前缀 + 当前后缀左边已经生成出的 token”。注意，这里**前缀不参与预测目标**，它只提供条件。

这也是 Prefix LM 的边界。它解决的问题不是“让模型什么都更强”，而是更具体的：

**如何在同一条序列里，同时完成条件理解和条件生成。**

职责边界可以写成表格：

| 部分 | 作用 | 是否参与 loss | 是否允许看未来 |
|---|---|---|---|
| Prefix $p$ | 提供条件和上下文 | 否 | 在前缀内部双向可见 |
| Suffix $y$ | 需要生成的目标内容 | 是 | 只能看左侧 |

玩具例子先看一个最小版：

- 前缀：`[北京是中国首都, 问题：北京属于哪个国家？]`
- 后缀：`[中国]`

这里前缀里的两句话可以互相参考，因为它们都属于“已知条件”；但生成“中国”时，模型只能看前缀和后缀左侧，不能偷看还没生成出来的后续答案。

真实工程例子更典型：

- 前缀：`[检索文档1, 检索文档2, 检索文档3, 用户问题]`
- 后缀：`[最终答案]`

这就是很多 RAG 问答系统的标准结构。RAG 是 retrieval-augmented generation，白话讲就是“先查资料，再让模型基于资料作答”。Prefix LM 很适合它，因为多段检索证据都属于条件，应当允许它们先互相融合。

边界也要说清楚。如果任务只是纯续写，比如“继续写这篇小说”“继续补全这段代码”，输入里没有稳定的条件区和生成区，那么 Prefix LM 不一定带来显著收益，甚至只会增加掩码复杂度。

---

## 核心机制与推导

Prefix LM 的关键不在“模型多了一层什么结构”，而在**注意力掩码**。注意力掩码就是一张可见性表，决定位置 $i$ 能不能看见位置 $j$。

设前缀长度是 $m$，总长度是 $m+n$。Prefix LM 的可见性规则可写成：

$$
A_{ij} = 1 \iff
\begin{cases}
i \le m \land j \le m \\
\text{or } i > m \land j \le m \\
\text{or } i > m \land m < j \le i
\end{cases}
$$

否则 $A_{ij}=0$。

把这三条翻成自然语言：

1. 如果当前位置 $i$ 在前缀里，那么它可以看整个前缀。
2. 如果当前位置 $i$ 在后缀里，那么它可以看整个前缀。
3. 如果当前位置 $i$ 在后缀里，那么它还可以看后缀中自己左边以及自己当前位置的 token。

这个规则最重要的安全边界是：**前缀不能看后缀，后缀不能看自己右边**。前者防止条件区读到答案区造成信息泄漏，后者保证自回归生成成立。

看一个最小矩阵例子。令 $m=2, n=3$，序列为：

`A B | C D E`

其中 `A B` 是前缀，`C D E` 是后缀。可见矩阵是：

```text
A: 11000
B: 11000
C: 11100
D: 11110
E: 11111
```

逐行解释：

- `A` 能看 `A B`，不能看 `C D E`
- `B` 能看 `A B`，不能看 `C D E`
- `C` 能看 `A B C`
- `D` 能看 `A B C D`
- `E` 能看 `A B C D E`

这比纯 causal LM 多出来的能力只有一件事：**前缀内部变成双向可见**。如果是普通 causal LM，同样的 `A B | C D E` 会变成：

```text
A: 10000
B: 11000
C: 11100
D: 11110
E: 11111
```

区别只在前缀前两行。Prefix LM 允许 `A` 看见 `B`，普通 causal LM 不允许。看起来只是一个小变化，但它对条件建模很关键，因为前缀经常包含多段需要互相整合的信息，比如“系统指令 + 用户输入 + 检索证据”。

从 loss 角度看，这个结构也很干净：

- 掩码决定“能看谁”。
- label mask 决定“算谁的 loss”。

所以 Prefix LM 的本质不是“让前缀也生成”，恰恰相反，它强调的是：**前缀只负责被读，不负责被预测**。如果把前缀也纳入 loss，目标就会从“条件生成”偏成“把条件本身也当成续写内容来记忆”。

---

## 代码实现

工程里最常见的落地方式不是改 Transformer block，而是做三件事：

1. 拼接 prefix 和 suffix。
2. 构造 Prefix LM attention mask。
3. 只让 suffix 位置参与 loss。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只验证 mask 和 label 是否符合 Prefix LM 规则。

```python
def build_prefix_lm_mask(prefix_len: int, total_len: int):
    assert 0 <= prefix_len <= total_len
    mask = []
    for i in range(total_len):
        row = []
        for j in range(total_len):
            visible = False

            # 前缀位置：只能看整个前缀
            if i < prefix_len and j < prefix_len:
                visible = True

            # 后缀位置：能看整个前缀
            elif i >= prefix_len and j < prefix_len:
                visible = True

            # 后缀位置：能看后缀中自己左侧和自己
            elif i >= prefix_len and prefix_len <= j <= i:
                visible = True

            row.append(1 if visible else 0)
        mask.append(row)
    return mask


def build_labels(prefix_ids, suffix_ids, ignore_index=-100):
    return [ignore_index] * len(prefix_ids) + list(suffix_ids)


# 玩具例子：A B | C D E
prefix = [11, 12]
suffix = [21, 22, 23]

mask = build_prefix_lm_mask(prefix_len=len(prefix), total_len=len(prefix) + len(suffix))
labels = build_labels(prefix, suffix)

expected_mask = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
]

assert mask == expected_mask
assert labels == [-100, -100, 21, 22, 23]

# 边界检查
for i in range(len(prefix)):
    for j in range(len(prefix), len(prefix) + len(suffix)):
        assert mask[i][j] == 0  # 前缀不能看后缀

print("Prefix LM mask and labels are valid.")
```

这段代码体现了两个核心原则：

- `mask[i][j]` 控制可见性。
- `-100` 这样的 ignore index 用来让前缀不参与 loss。

如果换成 Hugging Face 风格，流程通常是下面这样：

| 步骤 | 输入 | 输出 | 容易错的点 |
|---|---|---|---|
| 拼接序列 | `prefix`, `suffix` | `input_ids` | 分词边界不一致 |
| 构造注意力掩码 | `prefix_len`, `total_len` | 2D/4D mask | 右上块泄漏 |
| 构造 labels | `input_ids` | `labels` | 前缀被错误计入 loss |
| 叠加 padding mask | `mask`, `attention_mask` | 最终 mask | padding 未屏蔽 |
| 训练或生成 | 最终 mask | logits / tokens | 训练推理边界不一致 |

一个更接近真实工程的例子是文档补全。假设产品需求是“给定合同模板和已填写字段，让模型生成剩余条款”：

- 前缀：合同模板、客户名称、金额、付款方式
- 后缀：待生成的风险条款或补充说明

这时 Prefix LM 的优势很明显。模板和字段本身都属于条件，应当允许互相读取；但待生成条款仍要保持标准左到右生成，方便复用现成的 decoder 推理接口、beam search、logits processor 和 KV cache。

---

## 工程权衡与常见坑

Prefix LM 的主要收益是：**在不放弃 decoder-only 生成接口的前提下，增强条件建模能力**。但代价也很现实：mask 更复杂，调试更容易出错，而且很多错误不会直接报错，而是 silently degrade，也就是“程序能跑，但效果变差”。

常见坑先放表格：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| prefix 参与 loss | 训练目标偏移 | 只对 suffix 计损失 |
| prefix 能看 suffix | 信息泄漏，验证虚高 | 右上块全部屏蔽 |
| suffix 因果方向写错 | 生成时偷看未来 | 检查下三角结构 |
| padding 未遮住 | 注意力污染 | 叠加 padding mask |
| 训练和推理边界不一致 | 分布漂移，效果不稳 | 固定模板和字段顺序 |
| KV cache 直接复用错误 | 生成异常或错位 | 边界变化时重建缓存 |

为什么“prefix 不参与 loss”这么重要？因为前缀代表已知条件。训练目标如果要求模型预测这些已知条件，本质上是在混合两件不同的事：

- 一件事是“理解条件”
- 一件事是“复现条件文本”

这会把训练重心从“基于条件生成答案”拉偏到“把输入也背下来”。在条件生成任务上，这通常不是你真正想优化的目标。

为什么“右上块必须全屏蔽”也这么重要？因为右上块对应“前缀位置看见后缀位置”。一旦允许，模型在处理条件区时就能读到答案区，相当于训练时泄题。离线指标可能很好看，但上线推理没有答案可看，性能会直接塌掉。

KV cache 也值得单独说。KV cache 就是生成阶段缓存历史 key/value，避免每次重复计算前文。Prefix LM 一般仍能使用它，因为后缀生成仍是标准自回归。但有一个条件：**前缀边界必须稳定**。如果训练时前缀是 `[文档, 问题]`，推理时却变成 `[问题, 文档]`，或者中间插入了新的系统字段，那不仅数据分布变了，缓存对应的位置语义也变了，复用就会不安全。

一个“正确/错误”对照可以概括很多问题：

| 写法 | 是否正确 | 原因 |
|---|---|---|
| `prefix=[证据, 问题], suffix=[答案]`，只算答案 loss | 正确 | 符合条件生成定义 |
| `prefix=[证据, 问题], suffix=[答案]`，证据和问题也算 loss | 错误 | 目标变成复述输入 |
| 前缀可读后缀 | 错误 | 训练信息泄漏 |
| 推理时改变前缀模板顺序 | 高风险 | 训练推理分布不一致 |

---

## 替代方案与适用边界

Prefix LM 不是“更高级的 causal LM”，也不是“简化版 encoder-decoder”。更准确的说法是：**它卡在两者之间，适合那些既要明确条件输入，又希望保留统一自回归解码链路的任务**。

先看三种方案的工程取向：

| 方案 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| Causal LM | 简单、成熟、推理接口统一 | 条件建模偏弱 | 纯续写、代码补全 |
| Prefix LM | 条件生成更自然，仍可统一解码 | mask 和数据构造更复杂 | RAG、指令跟随、文档补全 |
| Encoder-Decoder | 条件区和生成区职责清晰 | 编码和解码链路分离 | 翻译、摘要、结构化问答 |

可以给一个简化的选型规则：

1. 如果任务本质是“继续往后写”，优先 causal LM。
2. 如果任务本质是“读条件再生成”，且想沿用 decoder-only 推理接口，优先 Prefix LM。
3. 如果条件区很复杂，输入融合强、结构清晰、训练和部署都接受双塔式链路，encoder-decoder 也很合理。

玩具例子可以这样分：

- 纯续写一句话：`今天天气很好，接下来`  
  这是 causal LM 的主场。
- 阅读材料后答题：`材料 + 问题 -> 答案`  
  这是 Prefix LM 很自然的结构。
- 翻译任务：`英文句子 -> 中文句子`  
  这是 encoder-decoder 的经典场景。

真实工程例子还是 RAG 最典型。因为 RAG 的输入天然由“证据集合 + 用户问题”构成，这些内容都应该被视为条件前缀。Prefix LM 在这里的价值不是“绝对更强”，而是**结构上更贴近任务定义**：证据之间允许充分交互，答案仍可沿用标准 token-by-token 生成。

但也别把它用过头。如果你的数据里没有清晰稳定的 prefix/suffix 边界，或者后缀长度极短、前缀格式变化很大，那么 Prefix LM 的理论优势可能会被数据噪声和工程复杂度吃掉。

---

## 参考资料

1. [UniLM: Unified Language Model Pre-training for Natural Language Understanding and Generation](https://papers.nips.cc/paper_files/paper/2019/hash/c20bb2d9a50d5ac1f713f8b34d9aac5a-Abstract.html)
2. [What Language Model Architecture and Pretraining Objective Works Best for Zero-Shot Generalization?](https://proceedings.mlr.press/v162/wang22u.html)
3. [CausalLM is not optimal for in-context learning](https://openreview.net/forum?id=guRNebwZBb)
4. [Hugging Face Transformers: Language Modeling](https://huggingface.co/docs/transformers/tasks/language_modeling)
5. [Transformers Source: create_extended_attention_mask_for_decoder](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py)

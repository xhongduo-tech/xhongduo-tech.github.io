## 核心结论

掩码语言模型，Masked Language Model，简称 MLM，可以直接理解为“先把句子里一部分词挖空，再要求模型填回去”。BERT 的预训练目标函数，本质上就是这样一个定向填空任务。

先给结论：

1. BERT 不会对整句每个 token 都计算预测损失，只会对随机选中的约 15% token 计算交叉熵。交叉熵可以理解为“模型预测分布和正确答案之间的差距”。
2. 被选中的 15% token，不是全部替换成 `[MASK]`，而是采用 80-10-10 规则：
   - 80% 真正替换成 `[MASK]`
   - 10% 替换成随机词
   - 10% 保持原词不变
3. 这样设计的核心原因，不是为了形式复杂，而是为了减少“训练时总能看到 `[MASK]`、推理时却看不到 `[MASK]`”带来的分布不一致。
4. 相比自回归语言模型，Autoregressive Language Model，简称 AR LM，也就是“按从左到右一个个预测下一个词”的模型，MLM 更容易学习双向上下文；但它的训练利用率更低，因为每个 batch 真正产生监督信号的 token 大约只有 15%。

一个最直观的玩具例子：

一句话有 20 个词，随机选 3 个词参与训练。模型不会因为剩下 17 个词预测得不好而受罚，只会因为这 3 个被选中的位置预测错了而受罚。也就是说，MLM 的“训练目标”不是“把整句复述一遍”，而是“只修复被挖空的位置”。

下面这张简图把流程压缩成三步：

```text
原始序列 x
  |
  |  随机选 15% 位置，得到掩码集合 M
  v
污染序列 x~
  |-- 80% -> [MASK]
  |-- 10% -> 随机词
  |-- 10% -> 保持原词
  v
模型输出每个位置的词分布
  |
  |  只在 M 中的位置计算交叉熵
  v
L_MLM
```

这意味着 MLM 的目标函数不是“全量语言建模”，而是“稀疏监督的条件恢复”。

---

## 问题定义与边界

先把符号说清楚。

- 原始序列记为 $x=(x_1,x_2,\dots,x_n)$，也就是一串真实 token。
- token 可以理解为“模型实际处理的最小文本单位”，在 BERT 里通常是 WordPiece 子词，不一定等于自然语言里的完整单词。
- 掩码集合记为 $\mathcal{M}$，表示这次训练里被选中作为预测目标的位置集合。
- 污染序列记为 $\tilde{x}$，意思是“原序列经过 `[MASK]` / 随机词 / 原词保留之后，真正送进模型的输入”。

MLM 的任务不是恢复整句，而是：

> 已知被污染后的序列 $\tilde{x}$，预测 $\mathcal{M}$ 中各位置原本的真实 token。

这几个边界必须明确，不然后面容易混淆：

| 项目 | MLM 中的定义 | 不属于本文讨论 |
|---|---|---|
| 训练阶段 | BERT 预训练 | 下游分类、问答、检索微调 |
| 预测目标 | 只预测 $\mathcal{M}$ 内位置 | 不对全部 token 同时算损失 |
| 输入形式 | 被污染后的序列 $\tilde{x}$ | 不等于原始干净文本直接监督 |
| 监督来源 | 文本自身构造伪标签 | 不是人工标注任务 |

一个很小的中文例子：

原句是“今天天气很好”。如果这次随机选中了“天气”对应的位置，那么输入可能变成“今天 `[MASK]` 很好”，也可能变成“今天苹果很好”，也可能仍然是“今天天气很好”。无论输入长什么样，监督信号只有一个：那个被选中的位置原本应该是“天气”。其他位置“今天”“很好”不参与这一步损失。

再看一个更接近实现的边界表：

| 维度 | 典型设置 | 含义 |
|---|---|---|
| 选词比例 | 15% | 每个 batch 里约 15% token 成为监督目标 |
| 替换策略 | 80-10-10 | 控制输入污染方式，而不是改变正确标签 |
| 梯度参与 | 仅掩码位置 | 非选中位置一般用 `-100` 忽略 |

这里最容易误解的一点是：  
“保持原词不变”的那 10%，虽然输入看起来没变，但如果该位置被选进了 $\mathcal{M}$，它依然参与损失。也就是说，“是否参加监督”由“是否被选中”决定，不由“最终输入看起来是否变了”决定。

---

## 核心机制与推导

MLM 的目标函数通常写成：

$$
\mathcal{L}_{\text{MLM}}
=
-\sum_{i\in\mathcal{M}}
\log P_\theta(x_i \mid \tilde{x})
$$

这行公式可以拆成三层意思：

1. $P_\theta(x_i \mid \tilde{x})$  
   表示模型在看到污染序列 $\tilde{x}$ 后，对第 $i$ 个位置真实 token 的预测概率。
2. 取对数再加负号  
   表示使用负对数似然，也就是分类问题里常见的交叉熵形式。
3. 只对 $i \in \mathcal{M}$ 求和  
   表示只有被选中的位置才会贡献损失。

如果把它写得更像工程实现，会变成：

$$
\mathcal{L}_{\text{MLM}}
=
-\frac{1}{|\mathcal{M}|}
\sum_{i=1}^{n}
\mathbf{1}(i\in\mathcal{M})
\log P_\theta(x_i \mid \tilde{x})
$$

这里的 $\mathbf{1}(i\in\mathcal{M})$ 是指示函数，可以理解为“开关”：

- 如果位置 $i$ 在掩码集合里，开关为 1，这个位置参与损失。
- 如果不在，开关为 0，这个位置被忽略。

为什么只对 $\mathcal{M}$ 累加？原因很直接。

如果对所有位置都算损失，模型最容易学到的策略是“看到原词就复制原词”，这会把问题退化成接近恒等映射。恒等映射可以理解为“输入是什么就原样抄出来”，它几乎不需要真正理解上下文。MLM 的目标恰恰是强迫模型在局部信息缺失时，利用左右文恢复原词，因此必须把监督集中在被破坏的位置上。

### 15% 与 80-10-10 是如何配合的

先分两步看：

第一步，随机选位置。  
从长度为 $n$ 的 token 序列中，独立地以约 15% 概率选择目标位置，得到集合 $\mathcal{M}$。

第二步，对 $\mathcal{M}$ 内位置做输入污染。  
设某个位置 $i\in\mathcal{M}$，则：

- 80% 概率令 $\tilde{x}_i=[MASK]$
- 10% 概率令 $\tilde{x}_i=r$，其中 $r$ 是随机 token
- 10% 概率令 $\tilde{x}_i=x_i$，即输入保持原词

注意，监督标签始终是原始词 $x_i$，不是污染后的 $\tilde{x}_i$。

可以把这一点写成条件分布：

$$
\tilde{x}_i \sim
\begin{cases}
[MASK], & 0.8 \\
r \sim \text{Uniform}(V), & 0.1 \\
x_i, & 0.1
\end{cases}
\qquad i\in\mathcal{M}
$$

其中 $V$ 表示词表。

### 玩具例子：长度为 20 的句子

假设一句话切成 20 个 token，本次选中的位置是第 4、9、15 个。

- 第 4 个位置被替换成 `[MASK]`
- 第 9 个位置被替换成随机词 `apple`
- 第 15 个位置保持原词不变

那么送进模型的是污染后的句子 $\tilde{x}$，但损失只来自 4、9、15 这三个位置。其余 17 个位置即使也有输出概率分布，也不计入交叉熵。

这就是“稀疏监督”的准确含义。

### 为什么要有随机词和原词保留

如果 100% 都替换成 `[MASK]`，模型会强烈依赖一个训练时特有的显式信号：  
“这里有个 `[MASK]`，我知道你就是要我猜这个位置。”

问题在于，下游微调或真实推理时，输入通常没有 `[MASK]`。于是训练分布与推理分布出现明显偏差。分布偏差可以理解为“模型训练时见到的数据样子，和部署时实际看到的数据样子，不是同一种东西”。

10% 随机词的作用，是让模型学会在“当前位置看起来是个正常词，但其实它是错的”这种扰动下仍然依赖上下文。  
10% 原词保留的作用，是让模型即使看到正确词，也不能简单把“是否是 `[MASK]`”当成唯一信号，而必须学习更稳定的上下文表示。

换句话说，80-10-10 不是三个彼此独立的小技巧，而是同一个目标的三种约束：

- `[MASK]` 提供强监督入口
- 随机词防止模型把“不是 `[MASK]` 就不用猜”学成捷径
- 原词保留缓和训练输入与真实文本之间的落差

---

## 代码实现

在工程里，最常见的实现方式是用 Hugging Face 的 `DataCollatorForLanguageModeling` 自动生成 MLM 训练样本。collator 可以理解为“把一批样本拼成 batch，并顺手做 mask 变换的组件”。

核心逻辑有两步：

1. 根据 `mlm_probability=0.15` 选中需要参与预测的位置
2. 构造 `labels`：
   - 被选中的位置保留原始 token id，作为监督目标
   - 未选中的位置设为 `-100`，让 `CrossEntropyLoss` 忽略

一个简化的训练代码如下：

```python
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts = [
    "今天天气很好，我们去公园散步。",
    "掩码语言模型只在部分位置计算损失。"
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

masked_batch = collator([
    {"input_ids": ids, "attention_mask": mask}
    for ids, mask in zip(batch["input_ids"], batch["attention_mask"])
])

print(masked_batch["input_ids"].shape)
print(masked_batch["labels"].shape)
```

上面这段代码没有手写 80-10-10，因为 `DataCollatorForLanguageModeling` 已经替你做了。

如果想把原理彻底看透，手写一个最小可运行版本更直接。下面这个 `python` 代码块展示了：

- 如何按 15% 选择位置
- 如何应用 80-10-10 规则
- 如何把非掩码位置标成 `-100`
- 如何验证损失只在被选中位置上计算

```python
import random
import math

PAD_IGNORE = -100
MASK_ID = 103
VOCAB_SIZE = 1000

def mask_tokens(input_ids, mlm_probability=0.15, seed=7):
    random.seed(seed)
    labels = [PAD_IGNORE] * len(input_ids)
    corrupted = input_ids[:]
    selected_positions = []

    for i, token_id in enumerate(input_ids):
        if random.random() < mlm_probability:
            selected_positions.append(i)
            labels[i] = token_id

            p = random.random()
            if p < 0.8:
                corrupted[i] = MASK_ID
            elif p < 0.9:
                corrupted[i] = random.randint(0, VOCAB_SIZE - 1)
            else:
                corrupted[i] = token_id

    return corrupted, labels, selected_positions

def cross_entropy_for_selected(probs, labels):
    loss = 0.0
    count = 0
    for i, gold in enumerate(labels):
        if gold != PAD_IGNORE:
            loss -= math.log(probs[i][gold])
            count += 1
    return loss / count if count > 0 else 0.0

# 一个玩具输入
input_ids = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
corrupted, labels, positions = mask_tokens(input_ids, mlm_probability=0.3, seed=1)

# 构造一个假的概率输出：在被选中位置上给正确答案高概率，其他位置随便
probs = []
for i in range(len(input_ids)):
    dist = {token_id: 1e-6 for token_id in input_ids}
    if labels[i] != PAD_IGNORE:
        dist[labels[i]] = 0.9
    else:
        dist[input_ids[i]] = 0.9
    # 归一化
    s = sum(dist.values())
    dist = {k: v / s for k, v in dist.items()}
    probs.append(dist)

loss = cross_entropy_for_selected(probs, labels)

assert len(corrupted) == len(input_ids)
assert len(labels) == len(input_ids)
assert all(labels[i] == PAD_IGNORE for i in range(len(labels)) if i not in positions)
assert all(labels[i] == input_ids[i] for i in positions)
assert loss > 0

print("input_ids      =", input_ids)
print("corrupted      =", corrupted)
print("labels         =", labels)
print("masked_pos     =", positions)
print("mlm_loss       =", round(loss, 6))
```

这段代码最关键的语句不是 mask 本身，而是：

```python
labels = [PAD_IGNORE] * len(input_ids)
```

以及：

```python
if gold != PAD_IGNORE:
    loss -= math.log(probs[i][gold])
```

这就是“非掩码位置不参加梯度”的完整工程表达。

### 真实工程例子：Hugging Face 训练 BERT

真实工程里，流程通常是：

1. 文本先被 tokenizer 切成 token id
2. `DataCollatorForLanguageModeling` 在每个 batch 动态做 mask
3. `BertForMaskedLM` 输出每个位置的词表 logits
4. `labels == -100` 的位置被 `CrossEntropyLoss(ignore_index=-100)` 自动忽略

伪代码可以写成：

```python
batch = collator(samples)  # 自动生成 input_ids / labels
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
loss = outputs.loss
loss.backward()
```

这里 `outputs.loss` 并不是“整句每个 token 的平均损失”，而是“被选中位置的平均损失”。

从计算利用率角度看，如果平均只 mask 15%，那么一个长度为 512 的序列，真正提供监督的 token 大约只有：

$$
512 \times 0.15 = 76.8
$$

也就是说，虽然模型前向会处理全部 512 个位置，但直接产生标签约束的只有 77 个左右。这也是 MLM 训练效率不如 AR LM 的一个重要原因。

---

## 工程权衡与常见坑

MLM 的设计不是“标准答案”，而是一组工程折中。理解这些折中，比死记 80-10-10 更重要。

### 坑 1：把 100% 掩码都替换成 `[MASK]`

这是最常见的初学者直觉：既然要模型填空，那就全部换成 `[MASK]`，不是最干净吗？

问题在于，这样会导致训练和推理的输入分布严重偏离。

| `[MASK]` 比例 | 随机词比例 | 原词比例 | 训练稳定性 | 分布偏差 |
|---|---:|---:|---|---|
| 100% | 0% | 0% | 前期看起来直接 | 很高 |
| 80% | 10% | 10% | 通常更平衡 | 较低 |
| 60% | 20% | 20% | 扰动更强 | 更接近真实输入，但训练更难 |

如果每次模型都看到一个非常显眼的 `[MASK]`，它容易学会一种不稳的策略：

- 一旦看到 `[MASK]`，就启动“填空模式”
- 一旦没有 `[MASK]`，就缺少训练时熟悉的触发信号

可是真实下游任务，例如文本分类、语义检索、命名实体识别，输入往往全是真词，没有 `[MASK]`。于是模型部署后性能会掉。

### 坑 2：误以为“保持原词不变”就没有训练意义

这是第二个高频误解。

假设某个位置被选中了，但最终 10% 概率下保留原词。很多人会说：“输入和标签一样，那模型不是白学了吗？”

不对。这个位置仍然要求模型输出该词的高概率，仍然反向传播。它的意义在于：

- 让模型面对更接近真实文本的输入
- 避免模型把“被监督位置必然长得怪异”当成固定模式
- 让上下文表示在正常文本上也保持可用

### 坑 3：忽略“有效监督 token 只有 15%”带来的吞吐问题

MLM 并不是算得少。相反，它通常“前向处理很多，直接监督很少”。

一个真实工程观察是：

- 你仍然要对整段文本做 Transformer 编码
- 但只有少数位置计入损失
- 因此相同算力下，MLM 的标签利用率偏低

这会带来两个后果：

1. 预训练通常需要更多数据或更长训练时间
2. 小模型、小语料条件下，MLM 可能学不到足够稳定的表示

### 坑 4：随机词替换过强会引入无意义噪声

10% 随机词不是越多越好。随机替换本质上是在制造输入噪声。噪声可以理解为“故意让输入变脏，逼模型更依赖上下文”。

但如果随机比例太高，模型会频繁看到完全不自然的局部组合，比如：

- “数据库 在 蓝天 中 提交事务”
- “今天天气 编译器 很好”

这类噪声能起到正则化作用，但过量后会损伤语义连续性。正则化可以理解为“故意增加训练难度，避免模型记死数据”的方法。

### 一个真实工程例子

在中文搜索或推荐场景里，如果你用 BERT 做 query 编码器，预训练语料往往是真实用户查询日志。真实 query 通常很短，例如“苹果手机发热”“北京周末天气”“显卡驱动安装失败”。

如果你把短 query 中 100% 被选中位置都改成 `[MASK]`，模型会经常看到非常不自然的输入，如：

- `[MASK]` 手机发热
- 北京 `[MASK]` 天气
- 显卡驱动 `[MASK]` 失败

这会让模型过度依赖特殊标记，而不是学会真实查询中的共现模式。引入随机词和原词保留后，训练时更容易逼近真实线上文本分布，预训练得到的表示通常更稳。

---

## 替代方案与适用边界

MLM 不是唯一方案，也不是所有场景都最优。它更像一个家族的基线设计。

### 1. Whole Word Masking

Whole Word Masking，简称 WWM，可以理解为“如果一个词被切成多个子词，就一起遮掉”。  
这对中文和英文复合词都很有意义，因为子词级单独 mask 有时会泄漏太多局部线索。

例如英文单词 `playing` 可能被切成 `play` 和 `##ing`。  
普通 MLM 可能只遮住 `##ing`，模型很容易从 `play` 猜出完整词。  
WWM 会把整词对应的所有子词一起处理，使预测更接近“真正恢复一个词”。

### 2. Span Masking

Span Masking 可以理解为“不是随机打散地遮单点，而是遮一小段连续片段”。

玩具例子：

- 普通 MLM：遮“北京 今天 天气 很好”中的“今天”
- Span Masking：连续遮“今天 天气”

对初学者来说，可以把 Span 理解成“把连续两个或三个词一起挖掉”，而不是只挖一个洞。

这种方式更适合学习更长的语义依赖，因为连续缺失会迫使模型利用更大范围上下文。

### 3. ELECTRA 式替代目标

ELECTRA 不再让模型只恢复少量 `[MASK]` 位置，而是改成 replaced token detection，即“判断每个位置的词是不是被替换过”。

它的核心优势是：  
几乎每个位置都能产生训练信号，训练利用率更高。

这类方法在算力有限、语料有限时经常更高效，因为它不像 MLM 那样只在 15% 位置上直接监督。

下面做一个对比：

| 方法 | 训练信号位置 | 计算开销 | 语义连续性 | 适用边界 |
|---|---|---|---|---|
| MLM | 约 15% 位置 | 基线 | 中等 | 通用 BERT 预训练 |
| WWM | 约 15%，但按整词选 | 略高 | 更好 | 词边界重要的语言 |
| Span Masking | 连续片段 | 略高 | 更强 | 长片段恢复任务 |
| ELECTRA | 接近全位置 | 通常更高效 | 不同目标 | 算力敏感、强调效率 |

还可以再压缩成一个“何时选什么”的判断表：

| 场景 | 更适合的方法 | 原因 |
|---|---|---|
| 通用编码器预训练 | MLM | 实现成熟、生态完整 |
| 子词切分明显，担心泄漏 | WWM | 避免只靠残余子词猜词 |
| 强调片段级语义恢复 | Span Masking | 更符合连续语言结构 |
| 想提高样本利用率 | ELECTRA | 更多位置参与训练 |

因此，MLM 的适用边界也很清楚：

- 需要双向上下文编码时，它是很自然的起点
- 需要和 BERT 体系完全兼容时，它最省事
- 但如果你极度关注训练效率，或者语料较小、算力有限，纯 MLM 未必是最优解

---

## 参考资料

1. BERT 词条与预训练机制概述：<https://en.wikipedia.org/wiki/BERT_%28language_model%29>
2. Masked Language Modeling 直观说明：<https://dataopsschool.com/blog/masked-language-modeling/>
3. BERT 预训练与 MLM 公式说明：<https://mbrenndoerfer.com/writing/bert-pretraining-mlm-nsp-training-guide>
4. Hugging Face 论坛关于 `BertForMaskedLM` loss 的讨论：<https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607>
5. 关于 BERT masking scheme 分布偏差讨论：<https://stats.stackexchange.com/questions/464201/bert-masking-scheme>

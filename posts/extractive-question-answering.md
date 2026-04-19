## 核心结论

抽取式问答是指：在给定上下文中，直接预测答案片段的起始位置和结束位置。

它不是让模型重新写一句答案，而是让模型从原文里截取一段连续文本。新手可以把它理解成：先读问题和文章，再在文章里把答案那一段圈出来。

边界位置是抽取式问答的核心。假设上下文被切成词元，词元是模型处理文本的最小单位，可以近似理解为“字、词或子词”。模型最终要输出两个编号：

```text
问题：退款多久到账？
上下文：退款会在 3 到 5 个工作日内到账。

词元：  [退款] [会] [在] [3] [到] [5] [个] [工作日] [内] [到账]
位置：     1    2   3   4   5   6   7     8      9    10

答案：3 到 5 个工作日
起点：4
终点：8
```

抽取式问答和生成式问答的差异如下：

| 类型 | 输出方式 | 答案是否必须来自原文 | 典型能力 | 典型风险 |
|---|---|---:|---|---|
| 抽取式问答 | 选择原文中的连续片段 | 是 | 定位、引用、可追溯 | 找不到跨段或改写答案 |
| 生成式问答 | 生成一段新文本 | 否 | 总结、解释、改写 | 可能生成原文不存在的内容 |

核心结论可以压缩成一句话：抽取式问答的本质不是“会不会回答”，而是“能不能在材料中找到答案边界”。

---

## 问题定义与边界

抽取式问答的标准输入通常包含两部分：问题和上下文段落。上下文段落是模型允许查找答案的材料，答案必须来自这段材料的原文。

| 项目 | 含义 | 示例 |
|---|---|---|
| 问题 | 用户要问的内容 | 退款多久到账？ |
| 上下文 | 可查找答案的文本材料 | 退款申请通过后，款项会在 3 到 5 个工作日内到账。 |
| 输出 | 上下文中的连续答案片段 | 3 到 5 个工作日 |
| 边界 | 答案片段的开始和结束位置 | start=若干词元，end=若干词元 |

抽取式问答只适合连续 span。span 是文本中的一段连续片段，比如“3 到 5 个工作日”。如果答案分散在两句话里，或者需要把多个信息合并后重新表述，普通抽取式问答就不直接适用。

一个玩具例子：

```text
问题：小明买了什么？
上下文：小明昨天去了商店，买了一支铅笔。
答案：一支铅笔
```

这里答案已经在上下文中出现，并且是连续片段，所以适合抽取式问答。

一个真实工程例子是客服知识库。用户问“退款多久到账”，退款政策文档里写着“退款会在 3 到 5 个工作日内到账”。系统应返回“3 到 5 个工作日内到账”或“3 到 5 个工作日”，而不是自己编写“通常几天后会退回”。

适用边界可以分成三类：

| 类型 | 例子 | 是否适合抽取式问答 |
|---|---|---:|
| 可抽取 | 原文写着“3 到 5 个工作日内到账” | 是 |
| 不可抽取 | 问“这项政策是否合理” | 否 |
| 无答案 | 上下文只写发货规则，问题问退款时间 | 需要无答案机制 |

SQuAD 1.1 主要处理“答案一定存在”的抽取式问答。SQuAD 2.0 增加了无答案样本，要求模型判断问题是否能从上下文中回答。

---

## 核心机制与推导

BERT 类模型处理抽取式问答时，会把问题和段落拼接成一个序列：

$$
x = [CLS]\ 问题\ [SEP]\ 段落\ [SEP]
$$

`[CLS]` 是序列开头的特殊词元，常用于表示整段输入；`[SEP]` 是分隔符，用来区分问题和段落。模型编码后，每个词元得到一个隐藏状态 $h_i$。隐藏状态是模型为某个词元计算出的向量表示，里面包含这个词元和上下文之间的关系。

然后模型为每个词元分别计算两个分数：

$$
s_i = w_s^T h_i + b_s
$$

$$
e_i = w_e^T h_i + b_e
$$

其中 $s_i$ 是第 $i$ 个词元作为答案起点的分数，$e_i$ 是第 $i$ 个词元作为答案终点的分数。$w_s$ 和 $w_e$ 是可训练参数，线性层是最简单的神经网络输出层，可以理解为“把向量换算成一个分数”。

分数还不是概率，需要经过 softmax。softmax 是把一组任意实数转换成概率分布的函数：

$$
p_{start}(i)=\frac{\exp(s_i)}{\sum_k \exp(s_k)}
$$

$$
p_{end}(j)=\frac{\exp(e_j)}{\sum_k \exp(e_k)}
$$

训练时，假设真实答案起点是 $i^*$，真实答案终点是 $j^*$，损失函数通常写成：

$$
L=-\frac{1}{2}\left(\log p_{start}(i^*)+\log p_{end}(j^*)\right)
$$

损失函数是模型要最小化的目标。这里的含义很直接：让真实起点的概率变高，也让真实终点的概率变高。

推理时，不能只分别选最大起点和最大终点，还要保证答案合法。合法 span 至少要满足：

$$
i \le j
$$

通常还会限制最大答案长度，比如 $j-i+1 \le 30$，避免模型抽出一整段无关文本。

简化流程如下：

```text
问题 + 段落
   ↓
分词并拼接：[CLS] 问题 [SEP] 段落 [SEP]
   ↓
BERT 编码每个词元
   ↓
起点线性层：start logits
终点线性层：end logits
   ↓
softmax 得到边界概率
   ↓
搜索合法 span
   ↓
还原为原文答案
```

数值例子如下。假设段落只有 3 个词元，答案是第 2 到第 3 个词元：

```text
start logits = [0, 5, 1]
end logits   = [0, 1, 4]
```

起点分数最高的是第 2 个词元，终点分数最高的是第 3 个词元，所以模型倾向于输出 `(start=2, end=3)`。这正是答案边界。

---

## 代码实现

工程实现通常复用预训练模型。预训练模型是已经在大规模语料上学过语言表示的模型，微调时只需要在问答数据上继续训练。BERT 问答模型的输出通常包含 `start_logits` 和 `end_logits`，分别表示每个词元作为答案起点和终点的分数。

代码流程可以写成：

```text
数据预处理
   ↓
tokenizer 编码问题和上下文
   ↓
用 offset_mapping 对齐 token 和原文字符位置
   ↓
模型前向计算 start_logits / end_logits
   ↓
训练时计算边界损失
   ↓
推理时搜索最佳 span 并映射回原文
```

`offset_mapping` 是分词结果到原文字符位置的映射。它解决的问题是：模型预测的是 token 编号，但最终要返回原始字符串。

| token | offset_mapping | 原文片段 |
|---|---|---|
| 退款 | (0, 2) | 退款 |
| 会 | (2, 3) | 会 |
| 在 | (3, 4) | 在 |
| 3 | (4, 5) | 3 |
| 到 | (6, 7) | 到 |
| 5 | (8, 9) | 5 |

下面是一个不依赖深度学习框架的最小 Python 示例，用来演示边界概率、合法 span 搜索和断言验证：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def best_span(start_logits, end_logits, max_answer_len=10):
    start_probs = softmax(start_logits)
    end_probs = softmax(end_logits)

    best = None
    best_score = -1.0

    for i, ps in enumerate(start_probs):
        for j, pe in enumerate(end_probs):
            if i <= j and (j - i + 1) <= max_answer_len:
                score = ps * pe
                if score > best_score:
                    best_score = score
                    best = (i, j)

    return best, best_score

tokens = ["退款", "会", "在", "3", "到", "5", "个", "工作日", "内", "到账"]
start_logits = [0, 0, 0, 5, 1, 0, 0, 0, 0, 0]
end_logits   = [0, 0, 0, 0, 1, 0, 0, 5, 0, 0]

span, score = best_span(start_logits, end_logits)
answer = "".join(tokens[span[0]:span[1] + 1])

assert span == (3, 7)
assert answer == "3到5个工作日"
assert score > 0.5
```

在 Hugging Face 风格的实现中，关键点通常是：

```python
# 伪代码：展示核心接口，不要求直接运行
encoded = tokenizer(
    question,
    context,
    truncation="only_second",
    max_length=384,
    stride=128,
    return_offsets_mapping=True,
    return_overflowing_tokens=True,
)

outputs = model(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
```

`truncation="only_second"` 表示优先截断上下文，而不是截断问题。`stride` 是滑窗重叠长度，防止长文档被切块后丢失答案。

输出示意：

| token | start_logits | end_logits |
|---|---:|---:|
| 退款 | 0.1 | -0.3 |
| 在 | 0.2 | 0.0 |
| 3 | 5.1 | 0.4 |
| 到 | 0.9 | 0.8 |
| 工作日 | 0.3 | 4.8 |

这里 `3` 最像答案起点，`工作日` 最像答案终点。

---

## 工程权衡与常见坑

抽取式问答在工程中最常见的问题是长文档截断。BERT 类模型有最大输入长度限制，常见上限是 512 个 token。输入包含问题、分隔符和上下文，所以真正留给上下文的长度还会更短。

截断风险示意：

```text
最大长度 512
[CLS] 问题 [SEP] 上下文 token 1 ... token 508 [SEP]
                              ↑
                         答案在 token 510 之后，被截掉
```

解决方法是滑窗。滑窗是把长文档切成多个有重叠的小片段，每个片段分别送入模型：

```text
窗口 1：token 1   - token 384
窗口 2：token 257 - token 640
窗口 3：token 513 - token 896
```

中间重叠部分由 `doc stride` 控制。这样即使答案靠近边界，也有机会完整出现在某个窗口里。

常见坑如下：

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只截取前 512 个 token | 后半段答案永远找不到 | 使用滑窗和 `doc stride` |
| token 边界和字符边界错位 | 返回答案少字、多字或乱码 | 使用 `offset_mapping` |
| 预测出 `start > end` | span 非法 | 解码时过滤非法组合 |
| 答案过长 | 抽出整段无关文本 | 设置最大答案长度 |
| 忽略无答案样本 | 对无法回答的问题强行抽取 | 区分 SQuAD 1.1 和 2.0 |
| 训练答案位置标错 | 损失函数学习错误目标 | 预处理阶段校验答案文本 |

SQuAD 1.1 和 SQuAD 2.0 的区别：

| 数据集 | 是否包含无答案问题 | 模型需要做什么 |
|---|---:|---|
| SQuAD 1.1 | 否 | 总是在上下文中抽取答案 |
| SQuAD 2.0 | 是 | 同时判断是否无答案 |

无答案机制通常会利用 `[CLS]` 位置。如果模型认为 `[CLS]` 作为起点和终点的分数最高，就可以把它解释为“没有可抽取答案”。不同实现会有不同打分策略，但核心都是比较“有答案 span”和“无答案”的置信度。

合法 span 解码规则通常包括：

| 规则 | 含义 |
|---|---|
| `i <= j` | 起点不能在终点之后 |
| `j - i + 1 <= max_len` | 答案不能超过最大长度 |
| span 必须位于上下文区域 | 不能从问题文本中抽答案 |
| 排除特殊 token | `[CLS]`、`[SEP]` 通常不作为普通答案文本 |

抽取式问答的优势是可追溯。系统可以返回答案，同时给出答案来自哪篇文档、哪一段、哪几个字符位置。这对客服、法务、医疗说明书、内部知识库都很重要。但它的代价也明确：只要答案不在原文里，它就不能凭空补出正确答案。

---

## 替代方案与适用边界

当答案不在原文中、需要总结改写，或者答案不是连续片段时，抽取式问答就不合适。此时应考虑其他方案。

| 方案 | 核心方式 | 适合场景 | 不适合场景 |
|---|---|---|---|
| 抽取式问答 | 从上下文中选连续 span | 答案可定位、要求引用 | 需要总结、推理、改写 |
| 生成式问答 | 直接生成答案文本 | 解释、归纳、自然表达 | 强可追溯要求 |
| 检索增强生成 | 先检索文档，再生成答案 | 多文档综合回答 | 严格只能返回原文片段 |
| 规则系统 | 用规则或模板匹配答案 | 格式稳定、字段明确 | 表达变化多的自然语言 |

真实工程中，客服文档里的“退款多久到账”适合抽取式问答，因为文档中通常有明确原句。用户问“这项政策是否合理”则不适合，因为它需要评价和解释，不是从原文里截取一个片段。

适用场景：

| 场景 | 原因 |
|---|---|
| 客服知识库 | 答案通常写在 FAQ 或政策文档中 |
| 合同条款定位 | 需要返回原文证据 |
| 产品说明书问答 | 答案可在说明书中定位 |
| 内部制度查询 | 需要引用制度原句 |

不适用场景：

| 场景 | 原因 |
|---|---|
| 多段信息合并 | 答案不是一个连续 span |
| 主观评价 | 原文没有标准答案 |
| 长篇总结 | 需要压缩和重写信息 |
| 开放式咨询 | 需要推理、规划或生成 |

与生成式 QA 的差异可以进一步拆开看：

| 维度 | 抽取式 QA | 生成式 QA |
|---|---|---|
| 输出单位 | 原文 span | 新文本 |
| 可解释性 | 强，能定位来源 | 取决于引用机制 |
| 灵活性 | 低 | 高 |
| 幻觉风险 | 较低 | 较高 |
| 对标注数据的要求 | 需要答案边界 | 可用问答对或指令数据 |
| 典型模型 | BERT QA | T5、GPT 类模型 |

工程选择可以用一句话判断：文档里有现成答案，就用抽取式；需要解释、归纳、改写，就换生成式或检索增强生成。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| BERT 论文 | 理解 `[CLS] 问题 [SEP] 段落 [SEP]` 和问答微调方式 |
| SQuAD 官方页面 | 理解抽取式问答基准和无答案任务 |
| Transformers 文档 | 理解工程实现中的 tokenizer、offset、滑窗 |
| BertForQuestionAnswering 源码 | 理解 `start_logits`、`end_logits` 和损失计算 |

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://research.google/pubs/bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding/)
2. [The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
3. [Hugging Face Transformers: Question answering](https://huggingface.co/docs/transformers/main/tasks/question_answering)
4. [Hugging Face Transformers BertForQuestionAnswering 源码](https://huggingface.co/transformers/v4.7.0/_modules/transformers/models/bert/modeling_bert.html)
5. [BERT paper on arXiv](https://arxiv.org/abs/1810.04805)

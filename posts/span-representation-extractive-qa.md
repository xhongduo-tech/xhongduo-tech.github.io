## 核心结论

抽取式问答的目标，是从一段 passage 中直接截取一个连续片段作为答案。这里的 **span**，白话说，就是“从第几个 token 到第几个 token 的这段连续文本”。如果只分别预测起点和终点，模型容易出现“起点像答案、终点也像答案，但两者拼起来不是同一个正确短语”的问题。更稳健的做法，是把每个候选 span 当成一个整体建模。

| 机制 | 作用 | 收益 |
| --- | --- | --- |
| 起点/终点拼接 | 把 span 两端边界一起表示 | 直接建模边界一致性 |
| 问答注意力融合 | 让 passage token 带上问题信息 | 候选 span 不再脱离问题独立打分 |
| 全局 softmax | 所有候选 span 一次性比较 | 避免局部贪心组合错误 |
| 长度限制与剪枝 | 限制候选集合规模 | 让 $O(N^2)$ 枚举可落地 |

对于候选答案 $a$，经典 span 表示可以写成：

$$
h_a = [p^*_{\text{start}},\; p^*_{\text{end}}]
\qquad
P(a\mid q,p)=\frac{\exp(w_a \cdot \mathrm{FFNN}(h_a))}{\sum_{a'} \exp(w_a \cdot \mathrm{FFNN}(h_{a'}))}
$$

这里的 **FFNN**，白话说，就是一个普通前馈神经网络，用来把拼接后的向量映射成分数。

玩具例子：passage 是“猫 抓住 了 鼠标”。如果候选 span 是“抓住”，可以把“抓住”的起点表示和终点表示拼起来，过一个小网络得到分数，再和其它 span 一起做 softmax。这样模型比较的是“整段答案”，不是把起点分类和终点分类拆开各做一次。

---

## 问题定义与边界

抽取式问答的输入通常是问题 $q$ 和文本 $p$，输出是 $p$ 中一个连续 span。连续的意思是，答案必须是 passage 里原样出现的一段，而不是生成出来的新句子。

| 项 | 内容 |
| --- | --- |
| 输入 | question、passage |
| 输出 | 一个连续 span |
| 候选集合 | 所有满足长度约束的连续子串 |
| 常见约束 | span 长度 $\le L$，如 $L=30$ |
| 主要难点 | 候选数接近 $O(N^2)$，且需要结合问题语义 |

若 passage 长度为 $N$，所有连续子串的数量大约是 $N(N+1)/2$。如果限制答案最大长度为 $L$，候选数会降为大约 $O(NL)$。这就是工程上常见的做法，因为很多问答数据集中的答案都比较短。

一个新手版场景：设 passage 长 100 个 token，设最大答案长度 $L=30$。那么模型不会枚举所有长度，而是只枚举长度 1 到 30 的 span。每个 span 单独打分，最后在所有合法 span 上取概率最大的那个。

信息流可以粗略理解为：

`question -> attention pooling -> question-aware passage token -> span scoring`

其中 **attention pooling**，白话说，就是“根据 passage 某个位置与问题的相关性，从问题里加权取出最相关的信息”。

边界也要说清楚：

1. 这种方法默认答案在给定 passage 中可抽取。
2. 如果 passage 很长，通常先做截断、分块，或先做检索再抽取。
3. 如果真实答案跨句太长，固定的 $L$ 可能造成召回下降。
4. 如果任务允许“不存在答案”，还需要额外建模 null span 或 no-answer 分类。

---

## 核心机制与推导

核心不是“枚举 span”本身，而是“怎样让 span 的表示真正感知问题与上下文”。

第一步是构造 question-aware 的 passage 表示。RASOR 一类方法会先给 passage 中每个 token 建一个增强表示：

$$
p_i^* = [p_i,\; q^{\text{align}}_i,\; q^{\text{indep}}]
$$

这里：

- $p_i$ 是 passage 第 $i$ 个 token 的上下文表示。
- $q^{\text{align}}_i$ 是与第 $i$ 个 token 对齐的问题向量。
- $q^{\text{indep}}$ 是问题的全局向量。

**对齐向量**，白话说，就是“问题里和当前 passage 位置最相关的那部分信息”；**全局向量**，白话说，就是“整句问题的大意”。

然后通常再经过共享的 BiLSTM。**BiLSTM**，白话说，就是一个同时看左边和右边上下文的序列编码器。共享的意思是，所有 token 先统一编码一次，后面任意 span 都复用这套结果，不必为每个 span 重算。

接着定义 span 表示。若候选答案 $a=(l,r)$，即从位置 $l$ 到位置 $r$，则：

$$
h_a = [p_l^*,\; p_r^*]
$$

最简单的直觉是：一个 span 至少应该知道自己“从哪开始，到哪结束”。但只拼端点还不够，所以后面接 FFNN：

$$
s_a = w_a \cdot \mathrm{FFNN}(h_a)
$$

再对所有候选 span 做全局归一化：

$$
P(a\mid q,p)=\frac{\exp(s_a)}{\sum_{a'} \exp(s_{a'})}
$$

这里的关键不是公式形式，而是“所有候选答案在同一个概率空间里竞争”。这和独立预测 start/end 的差别很大。独立预测时，本质上是先学 $P(l\mid q,p)$ 和 $P(r\mid q,p)$，再做组合；span scoring 则是直接学 $P((l,r)\mid q,p)$。

玩具例子可以具体一点。问题是“猫抓住什么？”，passage 是“猫 抓住 了 鼠标”。经过问答注意力后，模型知道“什么”在问宾语，因此“鼠标”附近 token 会获得更高的相关表示。假设某个候选 span 的端点表示拼接后，经 FFNN 得分如下：

- “抓住” 的分数：1.3
- “鼠标” 的分数：1.1
- “抓住了” 的分数：0.4

那么：

$$
P(\text{“抓住”})=\frac{e^{1.3}}{e^{1.3}+e^{1.1}+e^{0.4}}\approx 0.44
$$

如果把“鼠标”的分数提高到 1.6，它就会成为最大概率答案。重点是，模型比较的是整段 span 的整体适配程度，而不是拆开的局部概率。

这和自然语言推理有共性。问答里的 span，和 NLI 里的短语片段，本质上都需要“短语级语义表示”。因此后来很多 span-aware 预训练模型，例如 SpanBERT，会强调短语边界与片段内部语义的一致建模。

---

## 代码实现

下面用一个可运行的 Python 玩具实现说明流程。它不依赖深度学习框架，只模拟“端点表示拼接 -> 线性打分 -> 全局 softmax”的主干逻辑。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def enumerate_spans(tokens, max_len):
    spans = []
    n = len(tokens)
    for start in range(n):
        for length in range(1, max_len + 1):
            end = start + length - 1
            if end < n:
                spans.append((start, end))
    return spans

def score_span(token_features, start, end):
    # h_a = [p*_start, p*_end]
    h = token_features[start] + token_features[end]
    # 一个极简“FFNN/线性层”模拟：边界相关性更高时得分更高
    return 0.7 * h[0] + 0.3 * h[1] - 0.2 * (end - start)

tokens = ["猫", "抓住", "了", "鼠标"]
# 每个 token 一个二维特征，可理解为 question-aware 表示的简化版
token_features = [
    [0.2, 0.1],  # 猫
    [0.6, 0.5],  # 抓住
    [0.1, 0.2],  # 了
    [0.9, 0.8],  # 鼠标
]

spans = enumerate_spans(tokens, max_len=2)
scores = [score_span(token_features, s, e) for s, e in spans]
probs = softmax(scores)

best_idx = max(range(len(spans)), key=lambda i: probs[i])
best_span = spans[best_idx]
answer = "".join(tokens[best_span[0]:best_span[1] + 1])

assert len(spans) == 7
assert abs(sum(probs) - 1.0) < 1e-9
assert answer in {"鼠标", "了鼠标"}

print("best span:", best_span, "answer:", answer)
```

这个例子刻意简化了 attention pooling 和 BiLSTM，但保留了工程骨架：先得到每个 token 的增强表示，再枚举 span，再对 span 统一打分。

实现时常见 Tensor 形状如下：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| $P$ | `(B, N, d)` | passage token 编码 |
| $Q$ | `(B, M, d)` | question token 编码 |
| $P^*$ | `(B, N, d')` | 融合问题后的 token 表示 |
| `span_index` | `(K, 2)` | 所有候选 span 的起止位置 |
| $H$ | `(B, K, 2d')` | 每个 span 的端点拼接表示 |
| `scores` | `(B, K)` | 每个候选 span 的分数 |

真实工程例子：在 SQuAD 或 NewsQA 中，通常流程是：

1. 对 question 和 passage 编码。
2. 用 attention pooling 生成 $q^{align}_i$。
3. 拼接得到 $p_i^*$，再用共享 BiLSTM 编码。
4. 枚举长度不超过 30 的所有 span。
5. 通过向量化 gather 一次性取出所有起点与终点表示。
6. 批量送入 FFNN 计算 `scores`。
7. 对合法候选做 masked softmax。

真正的优化点不在 Python 双层循环，而在张量化。循环只适合讲原理；线上实现必须依赖 batch 计算，否则 $O(NL)$ 的候选数会让吞吐下降得很快。

---

## 工程权衡与常见坑

span 表示学习的优势明确，但代价也明确。它比独立 start/end 更一致，却也更重。

| 坑 | 影响 | 方案 |
| --- | --- | --- |
| 只做 start/end 独立预测 | 容易选出不一致边界 | 改成 span-level 全局打分 |
| 候选过多 | 显存和延迟上升 | 限制最大长度 $L$，只保留合法 mask |
| 端点信息太弱 | 重叠短语分数接近，难分辨 | 用 FFNN 建模端点交互 |
| 长文档直接枚举 | 召回和速度都差 | 先检索/切块，再做抽取 |
| 剪枝过猛 | 直接漏掉正确答案 | 先保证覆盖率，再优化效率 |

一个常见误区是：既然答案是一个区间，那分别预测开始和结束不就够了吗？问题在于这隐含了较强的独立假设。模型可能把“猫抓住”附近看成很像开始，把“鼠标”附近看成很像结束，最后组合出并不合理的答案。span 表示把这两个边界放在同一个表示里评分，等于强制模型回答“这整段是不是答案”。

另一个坑是长度偏置。短 span 往往更容易拿高分，因为边界更集中、噪声更少；但真实答案未必总是短。工程上通常会加入长度约束、正则或更好的训练采样方式，避免模型只偏好短短几个 token。

文献里一个重要经验是：基于构成树的剪枝虽然能减少候选，但会丢失不少原本可答的 span，覆盖率不够稳定。相比之下，RASOR 这类“共享编码 + 全局 softmax”的方案虽然仍需控制复杂度，但在 SQuAD 一类任务上能更稳定地减少由搜索和边界组合带来的误差。

---

## 替代方案与适用边界

不同方案的差异，不只是速度快慢，更是“模型在什么层级上理解答案”。

| 方案 | 复杂度 | 覆盖率 | 适合场景 |
| --- | --- | --- | --- |
| 独立 start/end 预测 | 近似 $O(N)$ | 中等 | 资源紧、序列长、实现简单 |
| 枚举 span 统一打分 | 近似 $O(NL)$ | 高 | 需要答案一致性和更稳比较 |
| 基于树或规则剪枝的 span | 低于全枚举 | 依赖剪枝质量 | 结构先验强、可接受漏召回 |
| Span-aware 预训练微调 | 依赖底座模型 | 高 | 有算力、追求跨任务迁移 |

可以把两类主流程简化成：

`start/end pipeline: token scoring -> choose start -> choose end -> combine`

`span scoring pipeline: token encoding -> enumerate spans -> score each span -> global select`

前者轻，后者稳。什么时候选哪种，取决于边界条件：

1. 如果文本很长、系统对延迟极敏感，独立 start/end 仍然常用。
2. 如果业务更重视答案边界精确性，例如搜索摘要抽取、客服证据定位，span scoring 往往更合适。
3. 如果上游已经有强大的预训练模型，如 SpanBERT，短语级表示能力会更强，span-level 方法收益通常更明显。
4. 如果任务不是抽取式，而是生成式问答，span 表示就不再是主方法，因为答案可以不在原文中连续出现。

从任务共性看，span-aware 方法还适合迁移到 NLI、事件抽取、实体片段分类等任务，因为这些任务都需要“片段级表示”，而不是只看单个 token。

---

## 参考资料

1. Lee, He, Lewis, Zettlemoyer. *Learning Recurrent Span Representations for Extractive Question Answering*. arXiv:1611.01436.  
作用：本文的核心来源，给出 question-aware token 表示、端点拼接、FFNN 打分与全局 softmax 的经典形式。

2. Joshi et al. *SpanBERT: Improving Pre-training by Representing and Predicting Spans*. arXiv:1907.10529.  
作用：说明 span-aware 预训练的价值，帮助理解为什么短语级表示能迁移到问答、推理等任务。

3. Rajpurkar et al. *SQuAD: 100,000+ Questions for Machine Comprehension of Text*. arXiv:1606.05250.  
作用：提供抽取式问答的标准任务设定与评测背景，解释为什么答案边界精确性重要。

4. Trischler et al. *NewsQA: A Machine Comprehension Dataset*. arXiv:1611.09830.  
作用：补充真实工程风格的数据场景，说明在新闻与长文本中，长度限制和候选覆盖率是实际问题。

5. 相关扩展阅读：span-level NLI 与可解释性研究。  
作用：帮助建立“问答中的 span 表示”和“推理中的短语级语义表示”之间的共性认识。

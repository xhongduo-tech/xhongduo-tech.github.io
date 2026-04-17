## 核心结论

N-gram 语言模型的本质，是用“固定长度历史”近似“完整历史”。语言模型这个术语，白话讲就是“给一句话打概率分数的函数”，分数越高，说明这句话越像训练语料里常见的表达。它的核心假设是：

$$
P(w_t \mid w_1,\dots,w_{t-1}) \approx P(w_t \mid w_{t-n+1},\dots,w_{t-1})
$$

这就是 $(n-1)$ 阶马尔可夫假设。马尔可夫假设的白话解释是：预测下一个词时，不再看全部过去，只看最近几词。

对工程最重要的结论有三条：

1. N-gram LM 用计数表就能建模，训练和推理都直接、可解释，trigram，也就是三元模型，通常是准确率、内存、查询速度之间的默认折中。
2. 只用最大似然估计会出现零频问题。零频的白话解释是：训练里没见过某个短语，于是它的概率被算成 0，进而整句概率也变成 0。这在真实系统里不可接受，所以必须做平滑。
3. Kneser-Ney 平滑通常被认为是传统统计语言模型里效果最强的一类方法，因为它不是简单给所有未见词平均补一点概率，而是看“一个词能接在多少不同历史后面”，也就是续延概率。

一个最小玩具例子就能看清楚 trigram 的直觉。训练语料只有一句：

`我 爱 你`

则三元计数满足：

- $C(\text{“我 爱 你”}) = 1$
- $C(\text{“我 爱”}) = 1$

所以：

$$
P(\text{你} \mid \text{我 爱}) = \frac{C(\text{我 爱 你})}{C(\text{我 爱})} = 1
$$

这就是 N-gram 的工作方式：先数，再除。

| N | 使用的历史长度 | 常见名称 | 典型用途 |
| --- | --- | --- | --- |
| 1 | 0 个词 | unigram | 词频基线、回退最低层 |
| 2 | 1 个词 | bigram | 输入法、简单纠错 |
| 3 | 2 个词 | trigram | 语音识别 beam search、传统解码 |
| 4+ | 3 个及以上词 | higher-order n-gram | 更强局部模式，但更稀疏 |

---

## 问题定义与边界

语言模型要解决的问题很明确：给任意词序列 $w_1,\dots,w_N$ 计算概率。按照链式法则：

$$
P(w_1,\dots,w_N)=\prod_{i=1}^{N} P(w_i \mid w_1,\dots,w_{i-1})
$$

链式法则的白话解释是：整句概率可以拆成“每一步预测下一个词”的条件概率连乘。

但直接这么做有两个现实问题。

第一，条件太长，几乎不可能统计。比如要估计“今天 北京 天气 很好 所以 我们 去...”后面接什么，如果条件是前面所有词，那么训练集中同样上下文几乎不会重复。

第二，词表和阶数一上去，组合数爆炸。词表大小记为 $V$，如果做 trigram，理论上可能的三元组数量约为 $V^3$。即使大量组合根本不出现，存储和估计也已经非常重。

所以 N-gram 语言模型主动缩小问题边界：只建模局部历史，不试图覆盖长距离依赖。这种边界带来两个直接后果：

1. 优点是查表快，容易部署。
2. 缺点是看不见窗口以外的信息。

举一个玩具例子。训练集只包含两句：

- `我 爱 你`
- `我 爱 编程`

则在历史 `我 爱` 之后，模型只见过 `你` 和 `编程`。

| 历史 | 下一个词 | 计数 | 最大似然概率 |
| --- | --- | --- | --- |
| 我 爱 | 你 | 1 | 1/2 |
| 我 爱 | 编程 | 1 | 1/2 |
| 我 爱 | 他 | 0 | 0 |

这里边界很清楚：如果测试时出现 `我 爱 他`，最大似然 trigram 会直接给 0。问题不在于“他”这个词不存在，而在于“我 爱 他”这个三元组没见过。

这也解释了窗口大小为什么决定稀疏度。窗口越长，表达能力越强，但未见组合越多，零频越严重。

真实工程里，这个边界更明显。比如语音识别解码器需要在 beam search 中不断给候选序列打分。beam search 的白话解释是：每一步只保留若干最优候选，不枚举全部句子。此时语言模型必须非常快，因为它要被调用很多次。N-gram 胜在查询近似就是表查；但它也只能告诉你“最近两三个词之后什么更像”，不能理解更长的语义约束。

---

## 核心机制与推导

N-gram 的基础估计方法是最大似然估计：

$$
P(w \mid h)=\frac{C(h,w)}{C(h)}
$$

其中 $h$ 是 history，也就是历史上下文；$C(h,w)$ 是拼接后的 n-gram 计数；$C(h)$ 是历史本身出现的次数。

把它代回整句概率，就得到：

$$
P_n(w_1,\dots,w_N)\approx\prod_i \frac{C(w_{i-n+1},\dots,w_i)}{C(w_{i-n+1},\dots,w_{i-1})}
$$

这套公式没有数学障碍，真正难点在零频。任何一个位置的概率为 0，整句连乘就变成 0。于是需要平滑。平滑的白话解释是：把一部分概率质量从见过的事件挪给没见过的事件。

最简单的平滑是加一平滑：

$$
P(w \mid h)=\frac{C(h,w)+1}{C(h)+V}
$$

它容易理解，但效果通常不够好，因为它把补偿分得过于平均。语言分布并不平均，没见过的短语也不是同样可能。

Kneser-Ney 的关键改进在于：低阶分布不能只看词频，还要看词的“续延能力”。续延概率的白话解释是：一个词虽然总出现很多次，但如果只在少数固定搭配里出现，它作为回退候选未必应该高；反之，一个能接在很多不同历史后面的词，回退时更值得给高概率。

其核心形式可以写成：

$$
P_{KN}(w \mid h)=\frac{\max(C(h,w)-d,0)}{C(h)}+\lambda(h)P_{cont}(w)
$$

这里：

- $d$ 是折扣值，意思是从已见 n-gram 的计数里先扣掉一点；
- $\lambda(h)$ 是补偿系数，保证扣掉的概率质量被完整分配出去；
- $P_{cont}(w)$ 是续延概率，不看词本身出现多少次，而看它跟多少不同历史搭配过。

如果用你给出的例子解释回退过程，会更直观。假设三元组 `你 爱 他` 没见过，即：

$$
C(\text{你 爱 他})=0
$$

那么最大似然下：

$$
P(\text{他} \mid \text{你 爱})=0
$$

Kneser-Ney 不会直接给 0，而是分两部分：

1. 已见部分：如果某个高阶组合见过，就用折扣后的计数保留主贡献。
2. 未见部分：把留出来的概率质量，通过 $\lambda(h)$ 分给低阶模型。

于是即使 `你 爱 他` 没见过，也可以靠回退项得到非零概率：

$$
P_{KN}(\text{他} \mid \text{你 爱})=\frac{\max(C(\text{你 爱 他})-d,0)}{C(\text{你 爱})}+\lambda(\text{你 爱})P_{cont}(\text{他}\mid \text{爱})
$$

如果看成流程，就是：

- 先问：这个 trigram 见过吗；
- 见过，就用“折扣后的 trigram 概率”；
- 没见过，就退到更短历史；
- 退到更短历史时，不直接看普通词频，而是看续延概率。

这一步非常关键。比如“的”总频率很高，但很多高频是固定结构带来的；而某些词虽然总频率没那么高，却能接在很多不同上下文后面。Kneser-Ney 认为后者更适合作为回退分布的依据，这正是它比普通 backoff 或 add-$\lambda$ 更强的原因。

---

## 代码实现

先看一个最小可运行实现。这个版本不是完整 Kneser-Ney，而是一个能跑通“计数 -> 条件概率 -> 句子概率”的 trigram 原型，并用加性平滑避免零频。加性平滑的白话解释是：每个候选词先统一加一个很小的常数 $\alpha$。

```python
from collections import Counter, defaultdict
from math import prod

class NGramLM:
    def __init__(self, order=3, alpha=0.1):
        assert order >= 1
        assert alpha > 0
        self.order = order
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, corpus):
        for sentence in corpus:
            tokens = sentence.split()
            self.vocab.update(tokens)
            for i, word in enumerate(tokens):
                history = tuple(tokens[max(0, i - (self.order - 1)): i])
                self.counts[history][word] += 1
        assert len(self.vocab) > 0

    def prob(self, history, word):
        history = tuple(history[-(self.order - 1):])
        vocab_size = len(self.vocab)
        total = sum(self.counts[history].values()) + vocab_size * self.alpha
        return (self.counts[history][word] + self.alpha) / total

    def sentence_prob(self, sentence):
        tokens = sentence.split()
        probs = []
        for i, word in enumerate(tokens):
            history = tokens[max(0, i - (self.order - 1)): i]
            probs.append(self.prob(history, word))
        return prod(probs)

corpus = [
    "我 爱 你",
    "我 爱 编程",
    "你 爱 学习"
]

lm = NGramLM(order=3, alpha=0.1)
lm.fit(corpus)

p1 = lm.prob(["我", "爱"], "你")
p2 = lm.prob(["我", "爱"], "他")

assert p1 > p2
assert p2 > 0
assert lm.sentence_prob("我 爱 你") > lm.sentence_prob("我 爱 他")

print(round(p1, 4), round(p2, 4))
```

这个代码做了四件事：

1. 遍历语料，按历史窗口统计 `counts[history][word]`。
2. 查询时把历史截断为最近 `order-1` 个词。
3. 用平滑后的条件概率避免未见事件变成 0。
4. 用逐位置概率连乘得到整句分数。

如果只想理解机制，这已经够了。但真实工程一般不会自己手写纯 Python 实现，而会用 KenLM。原因不是“算法不同”，而是“规模不同”。真实语料可能有数亿词，模型要支持高效构建、压缩存储和低延迟查询。

典型流程是：

1. 准备分词后的纯文本语料。
2. 用 `lmplz` 统计各阶 n-gram 并估计 modified Kneser-Ney 参数，输出 ARPA。
3. 用 `build_binary` 转成二进制格式。
4. 在线系统中加载 probing 或 trie 结构，按候选历史查分。

一个真实工程例子是语音识别重打分。声学模型先给出若干候选转写，语言模型再参与最终分数：

$$
\text{final\_score}=\text{acoustic\_score}+\alpha \cdot \text{lm\_score}+\beta \cdot \text{len}
$$

其中：

- `acoustic_score` 是声学证据；
- `lm_score` 是语言模型分数；
- `len` 是长度补偿；
- $\alpha,\beta$ 是调参系数。

这时 trigram LM 的优势不是它最准，而是它够快，能在 beam search 的每一步都稳定查询。

---

## 工程权衡与常见坑

N-gram LM 的工程价值主要来自“低延迟、强可控、可解释”，但它的坑也非常固定。

第一，不能直接用最大似然上线。任何未见 n-gram 都会把整条候选打死。在低资源语料、领域迁移、口语文本中，这个问题一定出现。很多新手的第一个错误，就是训练出一个“看起来公式正确”的模型，然后发现测试句大量得分为 0。

第二，阶数不是越高越好。4-gram、5-gram 理论上更强，但数据稀疏会快速恶化，模型体积也更大。trigram 常被用作默认折中，不是因为它完美，而是因为它在多数工程里已经覆盖了足够多的局部搭配，同时查询成本仍然可控。

第三，数据结构选择直接影响线上表现。KenLM 常见两种结构：

| 数据结构 | 内存相对更省 | 查询相对更快 | 说明 |
| --- | --- | --- | --- |
| probing | 否 | 是 | 哈希查找快，适合追求延迟 |
| trie | 是 | 略慢 | 前缀树更紧凑，适合受限内存 |

如果系统部署在大内存服务器上，probling 通常更有吸引力；如果部署在内存严格受限环境，trie 更稳妥。这里不是算法精度权衡，而是系统资源权衡。

第四，词表处理会影响模型稳定性。训练和测试的分词规范不一致、特殊符号未统一、OOV 处理缺失，都会让大量查询落到未知词路径。OOV 是 out-of-vocabulary，白话讲就是“测试里出现了训练词表外的词”。如果不显式加入 `<unk>` 机制，线上分数会非常不稳定。

第五，句子边界符不能忽略。训练 trigram 时，通常需要加 `<s>` 和 `</s>` 这样的开始、结束标记。否则模型学不到“哪些词适合开头，哪些词适合结尾”。这会让生成评分和解码评分都变差。

再看一个真实工程里的常见坑。假设语音识别 beam search 中有两个候选：

- `今天 北京 天气 不错`
- `今天 北京 天气 布错`

如果训练语料几乎没见过“布错”这个字词组合，那么没有平滑时第二条候选很可能直接归零；这是好事。但如果第一条候选中某个 trigram 恰好也没见过，比如领域数据太少，正确候选也会被硬砍掉。平滑的意义不是“给错误答案机会”，而是“防止正确答案因数据稀疏被误杀”。

---

## 替代方案与适用边界

今天如果目标是纯准确率，神经语言模型通常明显优于 N-gram。perplexity，白话讲就是“模型对测试集有多困惑”，越低越好。在 PTB 这类基准上，Kneser-Ney trigram 常见大约在 150 到 200 的区间，而较强的 LSTM 类神经模型可以低到 60 多。

| 模型 | PTB PPL | 说明 |
| --- | --- | --- |
| Kneser-Ney 3-gram | 约 150.64 | 统计模型，查表快 |
| AWD-LSTM | 约 64.27 | 神经模型，能建模更长依赖 |

这说明两点。

第一，N-gram 不擅长长距离依赖。比如“虽然……但是……”“如果……那么……”这种跨较长跨度的结构，固定窗口很难完整建模。神经模型，尤其是 RNN、Transformer，更适合处理这种关系。

第二，N-gram 仍然没有退出工程舞台。原因不复杂：

- 它训练便宜，不需要 GPU。
- 它推理稳定，近似是 O(1) 表查。
- 它容易做在线融合，尤其适合 beam search。
- 它可解释，排错成本低。

因此适用边界可以这样理解：

| 场景 | 更适合 N-gram | 更适合神经 LM |
| --- | --- | --- |
| 低延迟解码 | 是 | 未必 |
| 小数据快速建模 | 是 | 往往不是最优 |
| 长距离依赖建模 | 否 | 是 |
| 强语义泛化 | 否 | 是 |
| 可解释性与排错 | 是 | 相对弱 |

一个务实结论是：不要把 N-gram 和神经 LM 理解成二选一。在很多系统里，N-gram 负责第一层快速打分或解码约束，神经 LM 负责后续重排序或增强。前者解决吞吐和延迟，后者解决表达能力。

如果你的任务是教学、基础研究入门、语音识别解码、输入法候选排序、受限设备文本评分，那么 N-gram 依然是值得掌握的核心工具。如果你的任务是开放域生成、长文本建模、复杂语义理解，那么它通常只能作为基线或辅助模块，而不是主力模型。

---

## 参考资料

- ScienceDirect, “n-gram language model”
- Wikipedia, “Kneser-Ney smoothing”
- Kenneth Heafield, “KenLM: Faster and Smaller Language Model Queries”
- KenLM 官方结构说明与 `lmplz` / `build_binary` 文档
- NVIDIA NeMo ASR language model customization 文档
- MIT Computational Linguistics 关于语言模型评估的论文
- NAACL 2024 关于 n-gram smoothing 与 label smoothing 关系的论文

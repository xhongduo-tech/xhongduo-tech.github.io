## 核心结论

文本质量分类器的目标不是“理解内容真假”，而是给文本片段一个“是否像高质量训练语料”的分数。这里的“分类器”可以先理解成一个自动打分器：输入一段文本，输出它更像高质量来源，还是更像低质量来源。

最实用的一套做法是：用 Wikipedia、部分高质量 Reddit/ELI5 问答作为高质量样本，用大规模随机网页作为低质量样本，训练一个轻量的 fastText 分类器。fastText 可以理解成“用词和字符片段做快速统计表示的小模型”，训练成本低、吞吐高，适合先把海量网页做第一轮清洗。

质量分数通常直接定义为：

$$
s(x)=\frac{p_{\text{HQ}}(x)}{p_{\text{HQ}}(x)+p_{\text{LQ}}(x)}
$$

其中 $p_{\text{HQ}}(x)$ 表示文本 $x$ 更像高质量集合的分值，$p_{\text{LQ}}(x)$ 表示更像低质量集合的分值。这个分数落在 $[0,1]$，越接近 1 越像高质量文本。校准良好时，直接设固定阈值筛选，比每次都重新跑 ROC 曲线搜索更省工程成本。

玩具例子可以先这样理解：准备两个桶，一个装 Wikipedia/高质量社区文本，一个装随机抓来的网页。分类器做的事，不是判断“这段话绝对好不好”，而是判断“它更像哪个桶”。如果一段文本更像优质桶，它就得到更高分。

真实工程里，这类质量分类器常用于预训练数据清洗。原因很直接：预训练最贵的是 token 和算力，先把明显低质量文本过滤掉，可以减少无效训练样本，降低成本，同时让模型更稳定地学到结构清晰、信息密度更高的语言模式。

---

## 问题定义与边界

问题可以定义为：给定一个文本片段 $x$，输出一个质量分数 $s(x)\in[0,1]$，再根据阈值决定是否保留。这里的“质量”不是审美评价，也不是事实核查，而是“该文本是否接近你定义的高质量训练分布”。

术语先说清楚：

- 高质量源：指你主动挑选的优质种子语料，比如 Wikipedia、经过挑选的 Reddit 讨论、ELI5 问答。
- 低质量源：指大规模网页抓取中未经过严格筛选的文本，比如随机网页、模板站、广告页、拼接页。
- 阈值：就是保留线，分数高于它就保留，低于它就丢弃。

对新手来说，可以把它看成“给网页片段打分，只把高分文本喂给大模型”。

下面这个表先把边界说明白：

| 维度 | 高质量源 HQ | 低质量源 LQ | 边界说明 |
|---|---|---|---|
| 典型来源 | Wikipedia、ELI5、精选 Reddit 话题 | 随机爬虫网页、站群页、模板页 | 来源只是代理变量，不是真理 |
| 常见特征 | 结构完整、句法稳定、主题集中 | 噪声多、重复多、跳转多、上下文断裂 | 只是统计倾向，不是绝对规则 |
| 分类目标 | 更接近“可作为训练语料”的风格 | 更接近“浪费训练预算”的风格 | 目标是数据清洗，不是内容审核 |
| 输出形式 | $[0,1]$ 质量分 | $[0,1]$ 质量分 | 最终都映射成统一分数 |
| 决策方式 | 超过阈值则保留 | 低于阈值则丢弃 | 阈值可全局设定，也可分语言设定 |

这个定义有两个重要边界。

第一，它不能替代事实性判断。一篇逻辑清楚但内容错误的文章，仍可能被打高分，因为分类器学到的是“文本分布”，不是外部世界的真值。

第二，它容易带上来源偏见。如果 HQ 全是英文百科和技术论坛，分类器就可能把“像百科英文”误学成“高质量”，从而错杀风格不同但同样有价值的语料，尤其是小语种、论坛体、口语问答体文本。

---

## 核心机制与推导

fastText 的核心思想很朴素：把文本拆成词和 n-gram，再把这些局部片段映射到向量，最后做一个轻量分类。这里的 n-gram 可以白话理解成“连续的小片段”，比如相邻词组或字符片段。它的价值在于，即使完整句子没见过，模型也能从局部模式上判断文本风格。

你可以把整个流程想成下面这个结构：

| 步骤 | 作用 | 白话解释 |
|---|---|---|
| 分词 / n-gram | 提取局部模式 | 把长文本拆成许多小片段 |
| 向量表示 | 把片段变成可计算数字 | 机器只能处理数字，不直接处理文字 |
| 线性分类层 | 计算 HQ/LQ 分值 | 判断“更像优质桶还是低质桶” |
| 分数归一化 | 输出 $s(x)$ | 把结果变成 0 到 1 的统一质量分 |

如果模型输出两个非负分值 $p_{\text{HQ}}(x)$ 和 $p_{\text{LQ}}(x)$，那么定义

$$
s(x)=\frac{p_{\text{HQ}}(x)}{p_{\text{HQ}}(x)+p_{\text{LQ}}(x)}
$$

有两个好处。

第一，分数天然在 $[0,1]$，便于工程系统统一处理。

第二，如果模型校准得足够好，这个分数就可以近似看作“属于高质量集合的后验概率”。“后验概率”可以白话理解成：看完这段文本后，模型更新后的相信程度。

推导并不复杂。假设模型给出的两个值分别对应 HQ 和 LQ 的相对支持度，那么文本属于 HQ 的比例就是 HQ 支持度占总支持度的那一部分，即上式。这和二分类 softmax 的直觉一致：哪个类支持度更高，归一化后的概率就更高。

玩具例子：

- 文本 A：$p_{\text{HQ}}=0.6,\ p_{\text{LQ}}=0.2$
- 则 $s(A)=0.6/(0.6+0.2)=0.75$

如果阈值设为 0.7，A 被保留。

- 文本 B：$p_{\text{HQ}}=0.1,\ p_{\text{LQ}}=0.5$
- 则 $s(B)=0.1/(0.1+0.5)\approx0.17$

B 会被丢弃。

这套公式背后的关键，不是数学本身，而是“校准”。校准可以理解成“0.8 分是不是真的大致意味着 80% 的高质量可能性”。如果模型没有校准好，0.8 只是一个排序分，能比较高低，但不能直接拿来解释概率，也不适合直接跨语言、跨领域复用同一个阈值。

阈值怎么选，通常有三种方式：

1. 先用 0.5 作为默认线。
2. 按业务成本调高或调低，比如宁可错杀也不放过时用 0.7。
3. 按语言或站点分布设百分位阈值，比如每种语言只保留前 30%。

第三种在多语言工程里尤其常见，因为不同语言的网页分布差异很大。如果你对英文、中文、低资源语言一刀切，最后往往只保留了“最像英文百科”的文本。

真实工程例子：在大规模预训练数据清洗里，常见流程是先用 fastText 对数十亿网页片段做粗筛，把明显低分的文本去掉，再对边界样本交给更强的 Transformer 分类器复查。这样做的原因不是 fastText 最准，而是它“足够快，先把最脏的一层筛掉”。

---

## 代码实现

下面给一个可以直接运行的最小 Python 示例。它不依赖 fastText 库，而是用“词表统计 + 线性打分”的玩具版本，目的只是把训练、打分、阈值过滤的接口说明白。

```python
from collections import Counter
import math

hq_texts = [
    "Python uses indentation to define code blocks and improve readability.",
    "Wikipedia is an encyclopedia with structured and edited articles.",
    "A good explanation defines the problem, assumptions, and mechanism clearly.",
]

lq_texts = [
    "click here buy now free free free limited offer",
    "asdf qwer broken html http http banner banner",
    "top 10 shocking tricks you will not believe !!!",
]

def tokenize(text: str):
    return [w.strip(".,!?").lower() for w in text.split() if w.strip(".,!?")]

def train_naive_quality_model(hq_samples, lq_samples):
    hq_counts = Counter()
    lq_counts = Counter()

    for text in hq_samples:
        hq_counts.update(tokenize(text))
    for text in lq_samples:
        lq_counts.update(tokenize(text))

    vocab = set(hq_counts) | set(lq_counts)
    hq_total = sum(hq_counts.values())
    lq_total = sum(lq_counts.values())
    alpha = 1.0

    def word_prob(word, counts, total):
        return (counts[word] + alpha) / (total + alpha * len(vocab))

    def predict_proba(text: str):
        tokens = tokenize(text)
        log_hq = math.log(0.5)
        log_lq = math.log(0.5)

        for w in tokens:
            log_hq += math.log(word_prob(w, hq_counts, hq_total))
            log_lq += math.log(word_prob(w, lq_counts, lq_total))

        p_hq = math.exp(log_hq)
        p_lq = math.exp(log_lq)
        score = p_hq / (p_hq + p_lq)
        return {"p_hq": p_hq, "p_lq": p_lq, "score": score}

    return predict_proba

predict_proba = train_naive_quality_model(hq_texts, lq_texts)

good_text = "This article explains the mechanism and assumptions clearly."
bad_text = "free banner click shocking offer now"

good_score = predict_proba(good_text)["score"]
bad_score = predict_proba(bad_text)["score"]

assert 0.0 <= good_score <= 1.0
assert 0.0 <= bad_score <= 1.0
assert good_score > bad_score

threshold = 0.5
accepted = [t for t in [good_text, bad_text] if predict_proba(t)["score"] >= threshold]

assert good_text in accepted
assert bad_text not in accepted

print("good_score =", round(good_score, 4))
print("bad_score =", round(bad_score, 4))
print("accepted =", accepted)
```

这段代码对应真实系统中的四个步骤：

1. 准备 HQ/LQ 标注样本。
2. 训练一个二分类模型。
3. 对新文本输出 `p_hq`、`p_lq` 和 `score`。
4. 按阈值决定 `accept/reject`。

如果换成真实 fastText 或 Transformer，接口形式基本不变，只是模型更强、特征更多。典型伪代码如下：

```python
def score_document(model, text, threshold=0.7):
    p_hq, p_lq = model.predict_proba(text)
    score = p_hq / (p_hq + p_lq)
    keep = score >= threshold
    return score, keep
```

几个关键超参数需要特别注意：

| 超参数 | 作用 | 常见影响 |
|---|---|---|
| `ngram` 范围 | 控制局部片段长度 | 太小抓不到风格，太大稀疏严重 |
| `min_count` | 过滤低频词 | 降噪，但可能伤害小语种 |
| `dim` | 向量维度 | 越大表达力越强，但训练更慢 |
| `threshold` | 保留线 | 越高精度越高，召回越低 |
| `class_balance` | HQ/LQ 采样比例 | 影响模型偏置和输出校准 |

真实工程例子：假设你要清洗一个 50TB 网页语料库。先把网页切成段落，去重、去模板后喂给 fastText。fastText 只保留得分前 20% 的段落，再把临界区间如 $[0.45, 0.75]$ 的样本交给小型 Transformer 复核。这样可以把大部分吞吐留给轻模型，把算力集中在最难判断的样本上。

---

## 工程权衡与常见坑

质量分类器最容易被误解的一点是：它学到的可能不是“高质量的本质”，而是“远离低质量网页的风格”。这两件事不等价。

举例说，如果 LQ 集合里充满乱码、广告和模板页，模型很容易学会“只要不像垃圾页就是高质量”。但真实世界里还存在很多中间态文本：论坛问答、实验日志、项目 issue、个人技术笔记。这些文本不一定像 Wikipedia，却可能对模型很有价值。

常见坑和对应策略如下：

| 坑 | 现象 | 风险 | 规避策略 |
|---|---|---|---|
| 阈值太高 | 只留下最“整洁”的文本 | 多样性下降，语域变窄 | 先看分布，再按保留率设阈值 |
| 英文中心偏置 | 英文得分高，小语种得分低 | 多语种覆盖被破坏 | 分语言建阈值，或做多语言 pooling |
| 来源泄漏 | 模型记住网站特征而非文本质量 | 泛化差 | 按域名切分训练/验证集 |
| 标注过粗 | 把全部随机网页都当 LQ | 中等质量文本被误杀 | 从 LQ 中抽样人工复核 |
| 概率不准 | 0.8 分并不真代表 80% | 阈值不可迁移 | 做温度缩放或保序校准 |
| 只看单段 | 上下文缺失导致误判 | 短段落召回差 | 合理切分窗口，保留局部上下文 |

对新手来说，最重要的工程认知有两个。

第一，不要执着于“统一最佳阈值”。不同语言、不同网站、不同切分策略，分数分布都会变。更稳妥的方法是先看每一类数据的分位数，再决定保留前多少。

第二，不要只盯准确率。数据清洗更关心的是保留后的训练效果、语料多样性、语言覆盖率、下游任务收益。一个把所有边缘样本都砍掉的分类器，离线指标可能很好，但预训练后的模型反而更窄、更脆弱。

真实工程里，多语言问题尤其明显。比如英文 Wikipedia 和中文技术博客的句法结构、标点习惯、术语密度都不同。如果 HQ 样本绝大多数来自英文，分类器很可能把“像英文书面语”当成高质量标准。这时常见做法是：

- 多语言 pooling：把多种语言的 HQ/LQ 一起训练，避免模型只记住单一语言风格。
- 分语言保留率调优：不是所有语言都用同一阈值，而是每种语言各保留一定比例，如 top 30%。
- 验证管道抽检：从每种语言的高分和低分样本里都抽样人工看，确认没有系统性误杀。

---

## 替代方案与适用边界

fastText 的优势是快、轻、便宜。它适合做第一层质量过滤，尤其适合超大规模语料的离线清洗。但它并不是所有场景的最优解。

如果你的 HQ/LQ 差异主要体现在局部词法和句式风格上，fastText 往往已经足够。如果差异依赖更深层的语义一致性、篇章结构、复杂上下文，Transformer 分类器通常更强。

三种常见方案可以直接比较：

| 方案 | 速度 | 资源消耗 | 精度上限 | 适用场景 |
|---|---|---|---|---|
| fastText-only | 很高 | 很低 | 中等 | 海量粗筛、快速迭代 |
| fastText + Transformer | 高 | 中等 | 高 | 工程上最常见的折中 |
| Transformer-only | 低 | 高 | 最高但成本大 | 中小规模高精度筛选 |

对新手最容易理解的做法是“两级门禁”：

- 第一级 fastText 粗筛，快速去掉明显低质量文本。
- 第二级小型 Transformer 复核边界样本，做二次把关。

这种组合方案的价值在于，它把计算预算放在最值得判断的地方。绝大多数垃圾页不需要大模型来识别，真正难的是那些“既不像垃圾页，也不完全像百科”的中间样本。

适用边界也要说清楚：

- 如果你只有几万条数据，没必要上复杂的两阶段系统，直接训一个小型 Transformer 可能更简单。
- 如果你做的是垂直领域，如医学、法律、代码，通用 HQ/LQ 定义往往不够，需要按领域重新构造高低质量样本。
- 如果你的目标是事实正确性、毒性控制、版权过滤，质量分类器只能做辅助，不能替代专门的过滤器。

所以，质量分类器最合适的位置不是“万能审核器”，而是“预训练数据管道中的一层便宜而稳定的统计过滤器”。

---

## 参考资料

1. Saada 等，《The Data-Quality Illusion: Rethinking Classifier-based Quality Filtering for LLM Pretraining》。核心贡献是重新定义基于分类器的质量过滤，给出 HQ/LQ 构造方式与质量分公式。
2. fastText 相关工程资料。核心内容是 n-gram 表示、层次 softmax 与轻量文本分类，说明为什么 fastText 适合大规模快速训练。
3. 多语言质量分类器相关 Workshop 资料。核心内容是多语言 pooling、分语言保留率调优、Q3 等验证策略。
4. FineWeb2 与相关工程仓库。核心内容是把 fastText 与更强分类器结合，用于超大规模多语种预训练数据清洗。
5. 质量分数校准与阈值实践资料。核心内容是把分类器输出变成可解释分数，并按成本或保留率选择阈值。

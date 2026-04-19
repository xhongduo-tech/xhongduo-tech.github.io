## 核心结论

对比学习文本表示的目标，不是直接“分类句子”，而是学习一个可以比较语义距离的向量空间。向量空间是指：每个句子都被编码成一组数字，数字之间可以用相似度计算远近。

新手版本可以直接理解为：模型要学会“同义句靠近，非同义句分开”。

例如：

- “怎么退订会员？”
- “会员取消订阅的方法是什么？”

这两句话表达的是同一类需求，编码后的向量应该接近。

而：

- “怎么退订会员？”
- “今天天气怎么样？”

语义无关，编码后的向量应该明显远离。

核心公式是：

$$
z_i=\frac{f_\theta(x_i)}{\|f_\theta(x_i)\|_2}
$$

其中 \(x_i\) 是输入句子，\(f_\theta\) 是文本编码器，\(z_i\) 是归一化后的句向量。归一化的意思是把向量长度缩放到统一尺度，方便比较方向相似度。最终比较的不是原始文本，而是向量相似度。

| 维度 | 分类学习 | 对比学习文本表示 |
|---|---|---|
| 训练目标 | 预测固定标签 | 学习可比较的语义向量 |
| 输出结果 | 类别概率 | 句子 embedding |
| 典型任务 | 情感分类、违规识别 | 检索、语义匹配、相似度排序 |
| 关注重点 | 标签是否预测正确 | 相似文本是否靠近 |
| 泛化方式 | 泛化到相同标签空间 | 泛化到新句子之间的语义比较 |
| 典型问题 | “这句话属于哪个类？” | “这两句话语义有多接近？” |

对比学习的真正价值在于“可比较的表示”，不是一次性分类结果。它适合召回、问答匹配、相似句排序、语义去重等场景。

---

## 问题定义与边界

在对比学习里，通常围绕一个 anchor 构造训练样本。anchor 是当前被比较的基准句子。

| 术语 | 白话解释 | 文本例子 |
|---|---|---|
| anchor | 当前要学习表示的基准句子 | “买手机推荐” |
| positive | 与 anchor 语义相同或接近的正样本 | “手机选购建议” |
| negative | 与 anchor 语义无关或明显不同的负样本 | “怎么煮米饭” |
| false negative | 实际相似却被当成负样本的错误负样本 | “手机怎么选”被误当成“买手机推荐”的负样本 |
| augmentation | 数据增强，即在尽量不改变语义的前提下改写文本 | 删除少量停用词、同义替换、轻微重排序 |

正样本可以来自人工标注的同义句，也可以来自数据增强。数据增强是指：对原句做轻微变化，让它看起来不同，但语义保持稳定。常见方式包括删除、替换、重排序。SimCSE 还提出一种更简单的方式：同一句子经过带 Dropout 的编码器两次，得到两个不同视图，把它们当成正样本。

Dropout 是神经网络训练中的随机丢弃机制，训练时会随机屏蔽一部分神经元输出。SimCSE 利用这个随机性，把同一句子的两次编码结果当作两个增强视图。

边界必须说清楚：对比学习不是“任何增强都有效”。文本和图像不同，图像轻微裁剪、颜色变化通常不改变主体，但文本里删除一个词就可能改变语义。

玩具例子：

- anchor：“买手机推荐”
- positive：“手机选购建议”
- negative：“怎么煮米饭”

这是合理的，因为正样本表达相似需求，负样本属于完全不同主题。

反例：

- 原句：“退款申请”
- 增强后：“退货申请”

这两个词在某些业务系统里不是同一件事。“退款”可能只涉及钱款退回，“退货”可能涉及商品回寄。把它们强行当作正样本，会让模型学到错误的语义边界。

另一个边界是 false negatives。假设一个 batch 里同时有：

- “上海到北京机票”
- “北京到上海机票”

它们方向不同，但都属于机票查询。如果任务只关心“是否机票查询”，它们很接近；如果任务关心“出发地和目的地”，它们又不同。负样本是否合理，取决于业务定义。

文本对比学习关注的是语义稳定性，不是字符级改写。只要改写破坏了业务语义，就不能作为可靠正样本。

---

## 核心机制与推导

对比学习文本表示的训练流程可以拆成六步：

输入句子 → 编码器 → 向量 → 相似度矩阵 → softmax → loss

编码器是把文本转换成向量的模型，可以是 BERT、RoBERTa，也可以是更简单的神经网络。embedding 是文本对应的向量表示。

先把句子 \(x_i\) 输入编码器：

$$
h_i=f_\theta(x_i)
$$

再做归一化：

$$
z_i=\frac{h_i}{\|h_i\|_2}
$$

归一化后，余弦相似度可以直接写成点积：

$$
s(u,v)=u^\top v
$$

余弦相似度是衡量两个向量方向是否接近的指标。值越大，方向越接近；值越小，语义通常越远。

对于 anchor \(x_i\)，正样本为 \(x_i^+\)，负样本集合为 \(\mathcal N_i\)，对比损失为：

$$
\ell_i=-\log \frac{\exp(s(z_i,z_i^+)/\tau)}
{\exp(s(z_i,z_i^+)/\tau)+\sum_{x_k\in\mathcal N_i}\exp(s(z_i,z_k)/\tau)}
$$

其中 \(\tau\) 是温度系数。温度系数控制 softmax 分布的尖锐程度：\(\tau\) 越小，高相似度样本会被放得越大，分布更尖；\(\tau\) 越大，分布更平。它会影响梯度强弱和训练稳定性。

这个公式也可以看成一个分类问题：给定 anchor，让模型从“一个正样本 + 多个负样本”里选出正确的正样本。只是最终目标不是分类器本身，而是训练出更好的句向量。

最小数值例子：

设某个 anchor 的相似度为：

- 正样本：\(s^+=0.8\)
- 负样本 1：\(s_1^-=0.2\)
- 负样本 2：\(s_2^-=0.0\)
- 温度：\(\tau=0.5\)

正样本概率为：

$$
p^+=\frac{e^{s^+/\tau}}{e^{s^+/\tau}+\sum e^{s_k^-/\tau}}
=\frac{e^{0.8/0.5}}{e^{0.8/0.5}+e^{0.2/0.5}+e^{0.0/0.5}}
$$

也就是：

$$
p^+=\frac{e^{1.6}}{e^{1.6}+e^{0.4}+1}\approx 0.664
$$

损失为：

$$
\ell=-\log(0.664)\approx 0.410
$$

新手版本：正样本越像 anchor，分子越大，正样本概率越高，loss 越小；负样本越像 anchor，分母越大，正样本概率越低，loss 越大。

SimCSE 的关键点是把“同一句子经过两次 Dropout”当成正样本对。这样不需要人工标注同义句，也能训练句向量。它的假设是：同一句子的两次随机网络视图应该保持语义一致，而 batch 内其他句子通常可以作为负样本。

---

## 代码实现

实现上可以拆成四块：数据构造、编码器前向、对比损失、评估与推理。

| 阶段 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 数据构造 | 原始句子或同义句对 | 两个视图 | 构造 anchor 和 positive |
| 增强 | 删除、替换、Dropout | 增强文本或随机视图 | 生成正样本 |
| 编码 | 文本 batch | 向量 batch | 得到句子表示 |
| 损失 | 相似度矩阵 | cross entropy loss | 拉近正样本、推远负样本 |
| 评估 | 句子对和人工分数 | 相关性或排序指标 | 判断表示质量 |
| 推理 | 用户查询、候选文档 | top-k 相似结果 | 用于检索或匹配 |

下面是一个可运行的最小 Python 例子。它不用深度学习框架，只演示 `normalize`、`similarity_matrix`、`cross_entropy` 和 `forward` 的核心计算路径。真实训练时可以把 `ToyEncoder` 换成 BERT 类模型。

```python
import math
from collections import Counter

def tokenize(text):
    return text.lower().replace("？", "").replace("?", "").split()

class ToyEncoder:
    def __init__(self, vocab):
        self.vocab = vocab

    def forward(self, sentences):
        vectors = []
        for sent in sentences:
            counts = Counter(tokenize(sent))
            vec = [float(counts.get(word, 0)) for word in self.vocab]
            vectors.append(vec)
        return vectors

def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def similarity_matrix(left_vectors, right_vectors):
    left = [normalize(v) for v in left_vectors]
    right = [normalize(v) for v in right_vectors]
    return [[dot(a, b) for b in right] for a in left]

def cross_entropy(logits, labels):
    losses = []
    for row, label in zip(logits, labels):
        max_logit = max(row)
        exps = [math.exp(x - max_logit) for x in row]
        denom = sum(exps)
        prob = exps[label] / denom
        losses.append(-math.log(prob))
    return sum(losses) / len(losses)

def contrastive_loss(view_a, view_b, encoder, temperature=0.5):
    a = encoder.forward(view_a)
    b = encoder.forward(view_b)
    sims = similarity_matrix(a, b)
    logits = [[score / temperature for score in row] for row in sims]
    labels = list(range(len(view_a)))
    return cross_entropy(logits, labels), sims

def build_batch_views(sentences):
    # 监督版可以直接传入人工同义句；无监督版可传入同句的两个视图。
    view_a = sentences
    view_b = sentences[:]
    return view_a, view_b

def evaluate_sts(encoder):
    s1 = ["cancel membership"]
    s2 = ["cancel membership"]
    s3 = ["weather today"]
    same = similarity_matrix(encoder.forward(s1), encoder.forward(s2))[0][0]
    diff = similarity_matrix(encoder.forward(s1), encoder.forward(s3))[0][0]
    return same, diff

vocab = ["cancel", "membership", "weather", "today", "phone", "buy"]
encoder = ToyEncoder(vocab)

sentences = ["cancel membership", "weather today", "buy phone"]
view_a, view_b = build_batch_views(sentences)
loss, sims = contrastive_loss(view_a, view_b, encoder)

same, diff = evaluate_sts(encoder)

assert loss >= 0
assert sims[0][0] > sims[0][1]
assert same > diff
```

监督版和无监督版的区别主要在正样本来源。

| 版本 | 正样本来源 | 负样本来源 | 适用条件 |
|---|---|---|---|
| 监督版 | 人工标注同义句对 | batch 内其他句子或显式负样本 | 有高质量标注数据 |
| 无监督版 | 同句两次 Dropout 或轻量增强 | batch 内其他句子 | 有大量未标注语料 |
| 弱监督版 | 点击日志、搜索行为、FAQ 关联 | 曝光未点击或 batch 内样本 | 有业务行为数据 |

真实工程例子：客服 FAQ 检索。离线阶段把历史问题和标准答案编码成向量，建立 ANN 索引。ANN 是近似最近邻检索，用来在大量向量中快速找相似项。线上用户输入“会员怎么取消自动续费”，系统先编码成 query 向量，再从 FAQ 向量库中召回最接近的若干问题，最后返回对应答案或交给重排模型。

训练时可以用未标注客服问题做 SimCSE，也可以用历史“用户问法-命中 FAQ”作为监督正样本。这个场景里，问法变化非常多，直接枚举分类规则很难覆盖长尾表达，而语义向量检索更合适。

---

## 工程权衡与常见坑

batch size、负样本质量、温度系数三者强相关。batch size 是一次训练送进模型的样本数量。对比学习常把 batch 内其他样本当负样本，所以 batch 太小会导致负样本不足；batch 太大可以提供更多负样本，但显存和训练成本会上升。

温度系数 \(\tau\) 也不能随便设。太大时，相似度分布过平，模型区分不出强弱；太小时，分布过尖，模型可能过度关注少数 hard negative，训练不稳定。hard negative 是看起来很像 anchor、但按任务定义应该区分开的负样本。

| 问题 | 影响 | 规避方法 |
|---|---|---|
| false negatives | 把相似句推远，破坏语义空间 | 去重、过滤近重复、按业务语义分桶采样 |
| 增强过强 | 正样本语义被改坏，训练目标错误 | 优先使用 Dropout、轻量替换、受控模板增强 |
| batch 太小 | 负样本数量不足，表示区分度弱 | 增大 batch、梯度累积、memory bank 或 queue |
| 温度没调 | 分布过平或过尖，训练不稳定 | 把 \(\tau\) 当超参搜索，结合验证集指标选择 |
| 只看训练 loss | loss 下降但下游检索效果不一定提升 | 评估 STS、Recall@K、MRR、业务点击率 |

两个常见坑需要单独强调。

第一，把“上海到北京机票”和“北京到上海机票”当成负样本是否合理，要看任务目标。如果任务是“识别出行类问题”，它们不该被强行推远；如果任务是“精确匹配航线”，它们确实不同。对比学习不能脱离业务语义定义。

第二，对短句做删除增强很危险。比如“不要退款”删除“不要”后变成“退款”，语义直接反转；“苹果手机”删除“苹果”后变成“手机”，品牌信息丢失。短句通常信息密度高，删除一个词就可能删掉核心约束。

排查清单：

| 检查项 | 判断标准 |
|---|---|
| 正样本是否真的同义 | 人工抽样检查，业务人员能接受 |
| 负样本是否混入同义句 | 近重复、同 FAQ、同意图样本需要过滤 |
| 增强是否改变关键槽位 | 地点、时间、金额、否定词、品牌不能乱删 |
| 指标是否覆盖下游任务 | 检索看 Recall@K，排序看 MRR，语义相似看 Spearman |
| 线上是否需要重排 | 召回阶段可用双塔向量，精排阶段可用交叉编码器 |

新手版本：不要让训练题目本身出错。如果正样本不真相似，负样本不真无关，模型越训练越会学偏。

---

## 替代方案与适用边界

对比学习适合学习通用语义表示，但不是所有任务都需要它。若任务是明确分类，监督分类通常更直接；若有高质量标注句对，双塔监督匹配可能更稳；若候选数量很少且追求最高精度，交叉编码器通常效果更好。

双塔匹配是指 query 和 candidate 分别编码成向量，再计算相似度。交叉编码器是指把两个句子拼在一起输入模型，让模型直接判断二者关系。

| 方法 | 数据要求 | 推理成本 | 召回能力 | 精度上限 | 适用场景 |
|---|---|---|---|---|---|
| 对比学习 | 可用未标注语料，也可用同义句对 | 低，可提前建向量索引 | 强 | 中高 | 开放域检索、语义相似度、长尾问法召回 |
| 监督分类 | 需要明确标签 | 低 | 弱，通常不负责召回 | 高，限固定标签空间 | 情感分类、违规识别、意图分类 |
| 双塔匹配 | 需要正负样本对或行为数据 | 低到中 | 强 | 中高 | 搜索召回、FAQ 匹配、推荐召回 |
| 交叉编码器 | 需要句对标注 | 高，每对都要重新计算 | 弱，不适合全库召回 | 高 | 精排、相似度判别、候选集重排序 |

客服 FAQ 是典型适用场景。用户问法高度多样，例如：

- “怎么退订会员？”
- “自动续费在哪里关？”
- “会员不想用了怎么取消？”
- “取消订阅入口在哪？”

这些句子字面差异很大，但可能指向同一个 FAQ。对比学习得到的 embedding 可以先做 ANN 召回，覆盖大量长尾表达。之后可以再用交叉编码器重排 top-k 候选，提高最终精度。

但如果任务是“是否违规”这类边界非常明确的二分类，直接监督分类可能更简单。输入一句话，输出“违规/不违规”，不一定需要先学习通用向量空间。新手版本：不是所有任务都要先学向量，有些任务直接判标签更省事。

对比学习的适用边界可以概括为：

| 适合 | 不适合 |
|---|---|
| 查询和候选都很多，需要快速检索 | 候选很少，可以逐对精算 |
| 问法长尾，标签难以穷举 | 标签空间稳定且定义清晰 |
| 需要复用同一套文本表示 | 只解决一个窄分类任务 |
| 有大量未标注文本 | 负样本定义极不稳定 |
| 召回阶段更重要 | 需要强解释性规则判断 |

工程上常见组合是：对比学习或双塔模型负责召回，交叉编码器负责精排，规则负责兜底。不要把一种方法强行用于所有阶段。

---

## 参考资料

论文：

1. [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://aclanthology.org/2021.emnlp-main.552/)：最贴近本文主题，重点说明如何用 Dropout 做无监督句向量对比学习。
2. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)：SimCLR 论文，用来理解通用对比学习范式、温度系数和批内负样本。
3. [CLEAR: Contrastive Learning for Sentence Representation](https://arxiv.org/abs/2012.15466)：从句子表示角度讨论对比学习，可作为 SimCSE 之外的延伸阅读。

工程实现：

4. [Princeton NLP / SimCSE 官方代码仓库](https://github.com/princeton-nlp/SimCSE)：用于查看 SimCSE 的训练脚本、模型加载方式和评估流程。
5. [Sentence Transformers Documentation](https://www.sbert.net/)：用于理解句向量训练、相似度计算、语义检索和工程封装。
6. [FAISS Documentation](https://faiss.ai/)：用于了解向量索引和 ANN 检索，适合和文本 embedding 一起落地到检索系统。

阅读顺序建议：若只看一篇，先看 SimCSE；若想理解通用对比学习范式，再看 SimCLR；若想了解句子表示的更广泛对照，再看 CLEAR。

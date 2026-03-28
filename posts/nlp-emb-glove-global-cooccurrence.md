## 核心结论

GloVe 的核心不是“在窗口里边扫边猜下一个词”，而是先把整个语料的词共现统计汇总成一张全局共现矩阵 $X$，再直接拟合它的对数。**共现矩阵**可以先理解成“哪两个词在同一段局部上下文里一起出现过多少次的统计表”。GloVe 学到的不是单次窗口的偶然关系，而是整份语料里稳定出现的统计结构。

它的目标函数是：

$$
J=\sum_{i,j} f(X_{ij})\left(w_i^\top \tilde w_j+b_i+\tilde b_j-\log X_{ij}\right)^2
$$

其中，$w_i,\tilde w_j$ 是两套向量，分别表示“词向量”和“上下文向量”；**偏置** $b_i,\tilde b_j$ 可以先理解成“专门吸收词频影响的补偿项”；$f(X_{ij})$ 是权重函数，用来控制不同频次词对的影响。

和 Word2Vec 相比，GloVe 的区别不在“最后都得到词向量”这一层，而在训练入口：

| 维度 | GloVe | Word2Vec |
| --- | --- | --- |
| 统计对象 | 全语料共现矩阵 | 局部窗口样本 |
| 训练目标 | 拟合 $\log X_{ij}$ 的加权最小二乘 | 预测上下文或中心词 |
| 数据流程 | 先统计，再训练 | 边扫描语料边采样训练 |
| 对采样顺序依赖 | 低，矩阵固定后流程更稳定 | 高，窗口顺序和负采样会影响结果 |
| 适合场景 | 离线大语料、需稳定复现 | 流式训练、持续增量数据 |

玩具例子可以这样理解：如果“猫”和“宠物”在全语料里共现 500 次，那么 $\log 500\approx 6.21$。GloVe 就会调整参数，让“猫”的向量和“宠物”的上下文向量内积，再加上两个偏置后，尽量逼近 6.21。也就是说，它直接把“全局统计量”压进向量空间里。

---

## 问题定义与边界

GloVe 要解决的问题是：**怎样把离散的词，压缩成连续向量，同时保留它们在整份语料中的语义关系。**“连续向量”可以先理解成“一个词在多个数值维度上的坐标表示”。

形式化地说，设 $X_{ij}$ 表示词 $j$ 出现在词 $i$ 上下文中的次数，那么 GloVe 希望满足：

$$
w_i^\top \tilde w_j+b_i+\tilde b_j \approx \log X_{ij}
$$

这里为什么是 $\log X_{ij}$，而不是直接拟合 $X_{ij}$？因为原始频次跨度极大。高频词对可能比低频词对大出几个数量级，直接拟合原值会让训练被极少数高频词主导。取对数后，尺度被压缩，优化更稳定。

边界也很明确。

第一，GloVe 依赖完整语料的共现统计，所以它天然更偏离线训练。如果你的数据每分钟都在增长，频繁重建共现矩阵会很重。

第二，共现矩阵非常稀疏。**稀疏**可以先理解成“矩阵很大，但绝大多数位置都是 0”。例如 20 万词表理论上有 $4\times 10^{10}$ 个词对位置，但真实出现过的只是一小部分。

第三，低频词对信号弱，高频词对又容易过强，所以必须引入权重函数平衡。比如“猫”和“宠物”共现 500 次，目标值是 $\log 500\approx 6.21$；但“的”和“是”可能共现几十万次，这种词对如果不降权，会把模型拉向“学词频”而不是“学语义”。

因此，GloVe 的问题定义不是“给词一个向量”这么简单，而是“用低维参数去逼近全局共现结构，同时控制频次分布带来的偏差”。

---

## 核心机制与推导

GloVe 最关键的设计有两步。

第一步，使用两套向量：$w_i$ 和 $\tilde w_j$。这不是重复存储，而是因为“目标词”和“上下文词”的角色不完全对称。训练完成后，常见做法是取 $w_i+\tilde w_i$ 作为最终词表示。

第二步，使用带权重的最小二乘。**最小二乘**可以先理解成“让预测值和目标值之间的平方误差尽量小”。完整目标是：

$$
J=\sum_{i,j} f(X_{ij})\left(w_i^\top \tilde w_j+b_i+\tilde b_j-\log X_{ij}\right)^2
$$

其中权重函数通常取：

$$
f(x)=
\begin{cases}
\left(\frac{x}{x_{\max}}\right)^\alpha,& x<x_{\max}\\
1,& x\ge x_{\max}
\end{cases}
$$

常见超参数是 $x_{\max}=100,\alpha=0.75$。

这个函数为什么合理，可以分三层看。

1. 当 $x=0$ 时，不参与训练。因为 $\log 0$ 没定义，而且“没观察到”不等于“强负样本”。
2. 当 $x$ 很小时，权重较小。低频共现容易是噪声，不该和高质量统计一样重要。
3. 当 $x$ 很大时，权重封顶为 1。这样高频词对不会无限放大梯度。

数值例子：若 $X_{ij}=16$、$x_{\max}=100$、$\alpha=0.75$，则

$$
f(16)=\left(\frac{16}{100}\right)^{0.75}\approx 0.4
$$

而目标值是：

$$
\log 16 \approx 2.77
$$

也就是说，这个词对的误差会以大约 0.4 的强度进入损失，而不是像频次 1000 的词对那样一股脑地主导优化。

再看它和“语义关系”之间的联系。Stanford 原始论文强调的不是单个概率，而是**共现概率之比**。白话说，“ice”比“steam”更常和“solid”一起出现，而“steam”比“ice”更常和“gas”一起出现。这种“比例差异”比单独看频次更能表达语义特征。由于 $\log \frac{a}{b}=\log a-\log b$，比例关系最后会变成向量空间里的差分关系，所以 GloVe 在类比任务上通常表现不错。

玩具例子：假设小语料只有三句话：

| 句子 | 以“猫”为中心看到的上下文 |
| --- | --- |
| 猫 喜欢 鱼 | 喜欢，鱼 |
| 宠物 猫 很 可爱 | 宠物，很，可爱 |
| 猫 是 宠物 | 是，宠物 |

如果窗口统计后发现“猫-宠物”共现比“猫-汽车”高很多，那么模型就会把“猫”向量推向“宠物”相关区域，而不是“汽车”区域。这里并没有人工写规则，关系完全来自全局统计。

---

## 代码实现

下面给一个可运行的最小 Python 版本。它不追求速度，只演示三件事：构建共现、计算权重、做一次梯度更新。

```python
import math
from collections import defaultdict

def build_cooccurrence(corpus, window_size=2):
    vocab = {}
    for sent in corpus:
        for token in sent:
            if token not in vocab:
                vocab[token] = len(vocab)

    X = defaultdict(float)
    for sent in corpus:
        ids = [vocab[t] for t in sent]
        for i, center in enumerate(ids):
            left = max(0, i - window_size)
            right = min(len(ids), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = ids[j]
                distance = abs(i - j)
                X[(center, context)] += 1.0 / distance
    return vocab, X

def weight(x, x_max=100.0, alpha=0.75):
    if x <= 0:
        return 0.0
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0

corpus = [
    ["猫", "喜欢", "鱼"],
    ["宠物", "猫", "很", "可爱"],
    ["猫", "是", "宠物"],
]

vocab, X = build_cooccurrence(corpus, window_size=2)

cat = vocab["猫"]
pet = vocab["宠物"]

assert (cat, pet) in X
assert X[(cat, pet)] > 0

# 1 维向量，方便看数值
w = [0.2] * len(vocab)
wt = [0.3] * len(vocab)
b = [0.0] * len(vocab)
bt = [0.0] * len(vocab)

x = X[(cat, pet)]
target = math.log(x)
pred = w[cat] * wt[pet] + b[cat] + bt[pet]
g = weight(x, x_max=10.0, alpha=0.75) * (pred - target)

lr = 0.1
old_error = abs(pred - target)

w_cat_old = w[cat]
wt_pet_old = wt[pet]

w[cat] -= lr * g * wt_pet_old
wt[pet] -= lr * g * w_cat_old
b[cat] -= lr * g
bt[pet] -= lr * g

new_pred = w[cat] * wt[pet] + b[cat] + bt[pet]
new_error = abs(new_pred - target)

assert weight(16, 100.0, 0.75) > 0.3
assert weight(16, 100.0, 0.75) < 0.5
assert new_error < old_error
```

这段代码对应真实训练流程的缩略版：

1. 扫一遍语料，按窗口建立稀疏共现表 $X_{ij}$。
2. 对每个非零词对计算 $\log X_{ij}$ 和权重 $f(X_{ij})$。
3. 更新 $w_i,\tilde w_j,b_i,\tilde b_j$。
4. 最终输出 $w_i+\tilde w_i$ 或只取其中一套向量。

真实工程例子：假设你在做中文智能客服意图分类。语料已经离线积累到几千万句，且领域词固定，比如“退款”“工单”“签收”“物流异常”。这时可以先对历史语料构建共现矩阵，训练 GloVe 向量，再把它初始化到分类模型或检索召回模型中。这样做的价值不是“参数更新更炫”，而是能快速得到一套稳定、可复现实验的词表示，适合反复比较下游效果。

---

## 工程权衡与常见坑

GloVe 最大的工程压力不在优化器，而在共现矩阵。

第一类坑是内存。词表一大，词对数会爆炸。正确做法通常是先过滤低频词，再过滤低频词对，最后用 COO 或 CSR 这类稀疏结构存。**CSR** 可以先理解成“只存非零项及其索引的压缩格式”。

第二类坑是高频词污染。停用词、标点、模板化短语会制造大量高频共现。如果你不做降权和清洗，模型学到的更多是“语料格式”，不是“词义关系”。

第三类坑是复现性。很多人说 GloVe “完全可复现”，这句话不严谨。更准确的说法是：在共现矩阵固定后，训练不再依赖原始语料的窗口流式采样顺序，因此比 Word2Vec 更容易控制复现实验；但初始化、样本遍历顺序、浮点并行仍然会带来微小差异。

常见坑可以总结成下面这张表：

| 常见坑 | 现象 | 规避策略 |
| --- | --- | --- |
| 共现矩阵过大 | 内存爆炸、统计过程很慢 | 先截断词表，丢弃频次 < 5 的词或词对，使用稀疏存储 |
| 高频词主导梯度 | 向量更像词频编码 | 设 $x_{\max}$、$\alpha$，清理停用词，限制窗口 |
| 窗口过大 | 引入大量弱相关共现 | 从 5 到 10 这类中等窗口起调 |
| 低频词学不稳 | 稀有词向量质量差 | 增加语料、并入子词方案、做词表裁剪 |
| 复现不稳定 | 多次训练结果略有漂移 | 固定 seed、固定遍历顺序、固定线程数 |

新手常见操作建议是：`min_count=5`，构建稀疏共现表，固定 `seed=42`，先用默认的 $x_{\max}=100,\alpha=0.75$ 跑基线，再看下游任务是否需要调大或调小高频压制强度。

---

## 替代方案与适用边界

如果你的语料是静态的、规模大、希望实验稳定，GloVe 很合适；如果你的语料在持续流入，Word2Vec 通常更灵活；如果你的任务里有大量未登录词、拼写变体、词形变化，FastText 往往更实用。**未登录词**可以先理解成“训练时没见过或见得很少的新词”。

| 方法 | 训练入口 | 是否依赖全局统计 | 对增量数据友好度 | 对子词信息支持 |
| --- | --- | --- | --- | --- |
| GloVe | 共现矩阵分解 | 是 | 一般 | 否 |
| Word2Vec | 局部窗口预测 | 否 | 高 | 否 |
| FastText | 局部窗口预测 + 子词 | 否 | 高 | 是 |

适用边界也要说清。

1. 如果你是做离线预训练，比如电商评论聚类、客服 FAQ 检索初始化、传统分类器特征输入，GloVe 很合适。
2. 如果你是做实时聊天系统，语料每天变化，且希望在线增量更新，Word2Vec 往往更省事。
3. 如果你处理的是多形态语言、拼写噪声很多的文本，如用户搜索词、社交媒体文本，FastText 常常比纯 GloVe 更稳。
4. 如果你已经进入上下文模型时代，比如 BERT、现代检索编码器，静态词向量不再是最终方案，但仍然可作为轻量基线或资源受限场景的初始化。

真实工程对比例子：一个在线客服平台如果每周离线汇总历史工单，再训练一版词向量给分类和召回模块使用，GloVe 合适；但一个实时聊天机器人如果需要把昨晚新出现的活动名、明星名、梗词快速吸收进模型，Word2Vec 这类流式训练方法通常更顺手。

---

## 参考资料

1. Stanford NLP, *GloVe: Global Vectors for Word Representation*  
   https://www-nlp.stanford.edu/projects/glove/  
   用于支持“基于全局词共现统计训练”“训练只在非零共现项上进行”“一次统计后后续迭代更快”等结论。

2. Pennington, Socher, Manning, *GloVe: Global Vectors for Word Representation*  
   https://www-nlp.stanford.edu/pubs/glove.pdf  
   用于支持原始模型推导、共现概率比直觉、加权最小二乘目标和类比结构说明。

3. APXML, *GloVe Explained: Global Vectors for Word Representation*  
   https://apxml.com/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/glove-word-representation  
   用于支持面向初学者的机制解释，以及与 Word2Vec 在训练流程上的对比。

4. Emergent Mind, *GloVe Embeddings: Global Word Vectors*  
   https://www.emergentmind.com/topics/glove-embeddings  
   用于支持目标函数、权重函数常见设定与 PMI 视角的补充说明。

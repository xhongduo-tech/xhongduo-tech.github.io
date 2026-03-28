## 核心结论

Word2Vec 可以直接看成一个单隐层神经网络。单隐层的意思是：输入层和输出层之间只有一层中间表示。它的输入不是词本身，而是词的 one-hot 向量。one-hot 是一种“只有一个位置为 1，其余全为 0”的离散编码，用来表示“当前就是这个词”。

这个网络的核心路径是：

`one-hot 输入 → 嵌入矩阵 W → d 维隐藏表示 → 输出矩阵 W' → softmax 或负采样目标`

训练完成后，真正被保留下来的是嵌入矩阵 $W$。$W$ 的每一行都对应一个词的向量表示，也就是常说的词向量。典型维度 $d=100, 200, 300$，其中 300 是经典配置之一。

Word2Vec 有两种常见训练方向：

| 方法 | 输入 | 预测目标 | 直观理解 | 常见特点 |
|---|---|---|---|---|
| Skip-gram | 中心词 | 周围上下文词 | 已知当前词，猜邻居 | 对低频词更友好 |
| CBOW | 周围上下文词 | 中心词 | 已知邻居，猜中间词 | 训练通常更快 |

Word2Vec 的工程价值不在“网络有多深”，而在“训练目标设计得足够高效”。完整 softmax 需要对整个词表做归一化，词表一大就很慢。负采样（Negative Sampling，简称 NEG）只拿少量随机负例替代全词表计算，训练速度通常能比全 softmax 高一个数量级，实践里常见接近约 100 倍的提升。

训练好的嵌入空间还会出现线性结构。线性结构的意思是：向量差值会保留某种稳定关系。经典例子是：

$$
v(\text{king}) - v(\text{man}) + v(\text{woman}) \approx v(\text{queen})
$$

白话说，把词变成向量后，某些语义关系可以转成向量加减法。

---

## 问题定义与边界

自然语言里的词是离散符号，模型不能直接对“fox”“quick”“database”这些字符串做线性运算，所以第一步必须把词映射到连续空间。连续空间可以理解为“每个词对应一个可计算的数字向量”。Word2Vec 解决的正是这个映射问题。

它要回答的问题不是“这个句子整体什么意思”，而是：

1. 如何让经常出现在相似上下文中的词，学到相近的向量。
2. 如何用可扩展的训练方式，在大语料上快速得到词级别表示。

这里的“上下文”通常指一个固定窗口内的邻近词。例如句子：

`the quick brown fox jumps`

如果窗口大小设为 2，那么中心词 `fox` 的上下文可能是 `quick`、`brown`、`jumps`。Skip-gram 会把 `(fox -> quick)`、`(fox -> brown)`、`(fox -> jumps)` 当成训练样本。对初学者来说，可以把它理解成“给一个词，让模型猜它旁边经常出现谁”。

CBOW 则反过来。它会把 `quick + brown + jumps` 这些上下文词合起来，去预测中心词 `fox`。因此两者的差别不在词向量格式，而在训练信号的方向。

Word2Vec 的适用边界也要说清楚：

| 维度 | Word2Vec 的边界 |
|---|---|
| 表示单位 | 主要是词级别，不直接建模整句结构 |
| 依赖信息 | 主要依赖局部上下文，不直接看全局语义图 |
| 结果类型 | 静态词向量，同一个词在不同句子里默认同一个向量 |
| 适用场景 | 检索、召回、聚类、传统 NLP 特征、轻量预训练 |
| 不擅长场景 | 一词多义强依赖上下文、复杂推理、长距离依赖理解 |

玩具例子可以看得更直白一些。假设只有三句话：

- `king is a man`
- `queen is a woman`
- `king and queen rule`

训练后，`king` 和 `queen` 会因为共享相似上下文而靠近；`man` 和 `woman` 也会形成另一组关系。于是向量空间里可能出现“男性到女性”的稳定方向，这就是后面线性类比成立的基础。

真实工程里，Word2Vec 常被放在“语义召回”阶段。比如搜索系统要把用户搜的“显卡驱动安装失败”和文档库里的“GPU driver setup error”匹配起来，精确字符串匹配不够用，这时词向量或平均句向量可以先做粗召回，再交给更重的排序模型处理。

---

## 核心机制与推导

先看最基础的神经网络视角。

设词表大小为 $V$，嵌入维度为 $d$。某个输入词 $w$ 的 one-hot 向量记为 $x \in \mathbb{R}^V$。嵌入矩阵为：

$$
W \in \mathbb{R}^{V \times d}
$$

输出矩阵为：

$$
W' \in \mathbb{R}^{d \times V}
$$

因为 $x$ 是 one-hot，所以隐藏层表示其实就是“取出 $W$ 的第 $w$ 行”：

$$
h = x^T W = W_{w,:}
$$

这一步非常关键。它说明 Word2Vec 的隐藏层不是复杂非线性变换，而是一次查表。所谓“嵌入层”，本质上就是从矩阵里取一行向量。

然后用隐藏表示去预测输出词分布：

$$
u = hW'
$$

对每个候选上下文词 $c$，其条件概率为：

$$
P(c|w)=\frac{\exp(u_c)}{\sum_{j=1}^{V}\exp(u_j)}
$$

这就是 softmax。softmax 的作用是把一组分数变成概率分布。

如果用 Skip-gram，那么目标是最大化真实上下文词的概率。窗口里的每个上下文词都提供一个训练信号。问题在于，分母里有整个词表的求和，复杂度和 $V$ 成正比。词表几十万时，这一步非常重。

所以工程上更常用 SGNS，也就是 Skip-Gram with Negative Sampling。负采样的思想是：不要每次和整个词表比，只和“一个正样本 + $k$ 个负样本”比。

其经典目标函数写成：

$$
\log \sigma(v_c \cdot v_w) + \sum_{i=1}^{k}\mathbb{E}_{c_i \sim P_n}\left[\log \sigma(-v_{c_i}\cdot v_w)\right]
$$

其中：

- $v_w$ 是中心词向量。
- $v_c$ 是真实上下文词向量。
- $v_{c_i}$ 是随机采样得到的负样本词向量。
- $\sigma(z)=\frac{1}{1+e^{-z}}$ 是 sigmoid 函数，作用是把分数压到 0 到 1。
- $P_n$ 是噪声分布，常用 $P_n(w)\propto f(w)^{3/4}$，也就是按词频的 $3/4$ 次幂采样。

这条式子可以分两部分理解：

1. $\log \sigma(v_c \cdot v_w)$ 希望正样本点积更大。
2. $\log \sigma(-v_{c_i}\cdot v_w)$ 希望负样本点积更小。

白话解释就是：中心词要更像真实上下文词，同时远离随机抽来的词。

可以把它画成一个很简化的机制图：

`中心词 w -> 拉近 -> 真实上下文 c`  
`中心词 w -> 推远 -> 随机负词 c1, c2, ..., ck`

这也是为什么负采样高效。假设词表有 10000 个词，处理正例 `(fox, quick)` 时：

- softmax 要对 10000 个输出分数做归一化；
- NEG 如果只取 $k=5$，只更新 `fox`、`quick` 和 5 个负词相关参数。

从“全表更新”变成“局部更新”，速度自然会快很多。

再看一个玩具例子。语料只有：

- `the cat sits`
- `the dog sits`

如果中心词是 `cat`，窗口词是 `the` 和 `sits`。训练多轮后，`cat` 和 `dog` 会因为都和 `the`、`sits` 共现，而被拉到比较近的位置。这里没有任何人工规则告诉模型“cat 和 dog 都是动物”，这个相近关系完全来自上下文统计。

线性类比也可以从这个角度理解。假设“男性到女性”的变化在向量空间里形成一个近似稳定方向，那么：

$$
v(\text{king}) - v(\text{man}) \approx v(\text{royal})
$$

进一步加上 $v(\text{woman})$，就会逼近 `queen`。这不是严格代数定理，而是统计训练后经常出现的近似结构。

---

## 代码实现

下面先给一个可运行的最小 Python 版本，演示 Skip-gram + Negative Sampling 的单步损失计算。它不是完整训练器，但能准确反映“找向量、算相似度、区分正负样本”的核心过程。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))

def sgns_loss(center_vec, pos_vec, neg_vecs):
    # 正样本希望点积大
    pos_score = dot(center_vec, pos_vec)
    pos_loss = -math.log(sigmoid(pos_score))

    # 负样本希望点积小
    neg_loss = 0.0
    for neg_vec in neg_vecs:
        neg_score = dot(center_vec, neg_vec)
        neg_loss += -math.log(sigmoid(-neg_score))

    return pos_loss + neg_loss

center = [0.5, 0.1, -0.3]
positive = [0.4, 0.0, -0.2]
negative_1 = [-0.6, 0.2, 0.1]
negative_2 = [-0.3, -0.5, 0.4]

loss = sgns_loss(center, positive, [negative_1, negative_2])

assert loss > 0
assert dot(center, positive) > dot(center, negative_1)
print(round(loss, 6))
```

这段代码里：

- `center_vec` 是中心词向量。
- `pos_vec` 是真实上下文词向量。
- `neg_vecs` 是随机负样本向量。
- `assert` 用来保证最基本的数值关系成立。

如果要写成训练流程，最小逻辑通常是：

```python
# 伪代码
for center_word, context_word in positive_pairs:
    center_vec = W[center_word]
    pos_vec = W_out[context_word]

    neg_words = sample_from_noise_distribution(k)
    neg_vecs = [W_out[n] for n in neg_words]

    loss = -log(sigmoid(center_vec · pos_vec))
    for neg_vec in neg_vecs:
        loss += -log(sigmoid(-(center_vec · neg_vec)))

    backprop_and_update(W, W_out)
```

这一步里的核心动作只有三类：

1. 找向量。
2. 算点积相似度。
3. 按正负样本方向更新参数。

如果使用 TensorFlow，工程实现会更直接。官方教程里常见的是用 `tf.nn.nce_loss` 或类似接口封装负采样目标。`NCE` 是 Noise Contrastive Estimation，和 NEG 很接近，都是把“全词表分类”转成“少量正负对比”。

```python
import tensorflow as tf

vocab_size = 10000
embedding_dim = 300
num_sampled = 10

embedding = tf.Variable(tf.random.uniform([vocab_size, embedding_dim], -1.0, 1.0))
nce_weights = tf.Variable(tf.random.truncated_normal([vocab_size, embedding_dim], stddev=1.0 / embedding_dim**0.5))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

center_ids = tf.constant([12, 57, 91], dtype=tf.int64)         # 中心词 id
target_ids = tf.constant([[34], [88], [77]], dtype=tf.int64)   # 正样本上下文 id

embed = tf.nn.embedding_lookup(embedding, center_ids)

loss = tf.reduce_mean(
    tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=target_ids,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=vocab_size
    )
)

assert float(loss.numpy()) > 0
```

关键超参数可以先按经验值起步：

| 超参数 | 含义 | 常见起点 |
|---|---|---|
| `embedding_dim` | 词向量维度 | 100 或 300 |
| `window_size` | 上下文窗口大小 | 2 到 5 |
| `num_sampled` | 负样本数 $k$ | 5 到 20 |
| `learning_rate` | 学习率 | 0.01 到 0.05 |
| `min_count` | 低频词截断阈值 | 5 |
| `subsample` | 高频词下采样阈值 | 1e-5 附近 |

真实工程例子：搜索、推荐、广告召回系统里，经常先在日志语料上训练 Word2Vec，把 query、商品词、标签词映射到同一向量空间。在线上，一个用户输入的 query 可以先转成向量，再去近邻检索相似词或相似商品标签，作为第一阶段召回特征。这类系统重视的是训练快、部署轻、推理成本低，而不是句级深层理解，所以 Word2Vec 依然常见。

---

## 工程权衡与常见坑

Word2Vec 的难点不在“能不能跑起来”，而在“怎样让向量质量稳定”。很多效果差并不是算法错，而是训练配置不合理。

先看几个最常见的权衡：

| 选择项 | 好处 | 代价 |
|---|---|---|
| 更大的维度 `d` | 表达能力更强 | 内存更高，训练更慢 |
| 更大的窗口 | 捕捉更宽语义关系 | 容易混入噪声 |
| 更多负样本 `k` | 对比信号更强 | 单步训练更慢 |
| 保留低频词 | 覆盖更全 | 数据稀疏，训练不稳 |
| 高频词下采样 | 减少停用词干扰 | 过强会损失部分句法信息 |

负采样里有两个经验参数尤其重要。

第一是负样本数 $k$。实践里常见范围是 5 到 20。语料很大时，$k$ 太小会导致区分能力不足；太大则训练变慢，而且收益递减。

第二是噪声分布指数 0.75。也就是：

$$
P_n(w)\propto f(w)^{0.75}
$$

这里的意思是，负样本既参考词频，又不会完全被高频词垄断。若直接按原始词频采样，像 `the`、`of`、`is` 这样的高频词会占掉太多负样本；若完全均匀采样，又会让大量极低频词过度出现。0.75 是一个经验上很稳的折中。

常见坑可以集中看下面这个表：

| 常见坑 | 原因 | 结果 | 规避方式 |
|---|---|---|---|
| 词表不过滤低频词 | 语料稀疏，很多词几乎没学到 | 向量噪声大 | 设 `min_count`，先截断极低频词 |
| 不做高频词下采样 | 停用词样本太多 | 语义词训练不足 | 对高频词做 subsampling |
| `k` 设得过小 | 负对比不够 | 相似词边界模糊 | 从 5、10、15 做网格试验 |
| 学习率过高 | 更新剧烈震荡 | 损失不稳定 | 用衰减学习率或更小初值 |
| 窗口过大 | 混入弱相关词 | 向量语义发散 | 按任务调到 2 到 5 |
| 直接拿静态词向量处理歧义词 | 一个词只有一个向量 | 多义词效果差 | 需要上下文模型时换 BERT 类方案 |

还有一个常被忽略的问题：评估方法。很多人只看“king-man+woman≈queen”这种演示就判断模型好坏，这不够。线性类比只是一个现象，不是唯一目标。工程上更应该看：

- 近义词检索是否符合业务语义。
- 下游任务指标是否提升。
- 低频词和长尾词是否有足够覆盖。
- 不同时间切片训练出的向量是否稳定。

如果要做参数搜索，建议先固定语料清洗和词表策略，再小范围 grid search：

- `embedding_dim`: 100 / 200 / 300
- `k`: 5 / 10 / 15
- `window_size`: 2 / 4 / 5
- `learning_rate`: 0.01 / 0.025 / 0.05

否则你很难判断效果变化到底来自模型，还是来自词表和采样分布变化。

---

## 替代方案与适用边界

Word2Vec 不是唯一的词向量方法。它适合“轻量、快速、词级静态表示”，但并不覆盖所有需求。

先和几个常见方案做横向对比：

| 方法 | 核心思想 | 输入单位 | 优势 | 局限 | 资源需求 |
|---|---|---|---|---|---|
| Word2Vec | 局部上下文预测 | 词 | 训练快，部署轻 | 静态向量，多义词处理弱 | 低 |
| GloVe | 全局共现矩阵分解 | 词 | 利用全局统计信息 | 仍是静态向量 | 中 |
| fastText | 在词向量上加入子词 n-gram | 词 + 子词 | 对未登录词、拼写变体更稳 | 仍缺上下文适配 | 中 |
| ELMo | 上下文相关表示 | 词在句中 | 同词异义可区分 | 成本更高 | 高 |
| BERT | 双向上下文编码 | 子词序列 | 语义表达强，适合下游微调 | 推理重、训练更复杂 | 很高 |

如果只看静态 embedding，Word2Vec、GloVe、fastText 的差别可以粗略理解为：

- Word2Vec：通过“局部猜词”学向量。
- GloVe：通过“全局共现统计”学向量。
- fastText：把词拆成更小的子词片段一起学。

举例来说，`playing`、`played`、`player` 这些词在 fastText 里会共享一部分子词信息，所以对低频形态变化更稳。Word2Vec 则把它们更像是彼此独立的词条。

再和 BERT 这类上下文模型比，边界会更明显。Word2Vec 对一个词只给一个固定向量。例如 `bank` 在“river bank”和“bank account”里向量默认相同；BERT 会根据句子重新计算表示，所以能区分不同语义。

因此选择标准可以简单归纳成：

| 任务需求 | 更合适的方法 |
|---|---|
| 需要轻量词向量、快速训练、传统模型特征 | Word2Vec |
| 需要更强的 OOV 和词形处理 | fastText |
| 需要静态词向量但更依赖全局统计 | GloVe |
| 需要句子级、多义词、上下文理解 | BERT/ELMo 类模型 |

对初学者来说，可以记一句足够准确的话：Word2Vec 学的是“词在大语料中的平均用法”，BERT 学的是“词在当前句子里的具体用法”。这两者并不冲突，只是目标不同。

---

## 参考资料

1. Wikipedia: Word2vec  
   用于查架构总览、Skip-gram/CBOW 定义、线性类比现象和常见参数背景。

2. Baeldung: Word2Vec Negative Sampling  
   用于理解负采样目标函数、SGNS 公式、采样分布与 softmax 的替代关系。

3. Chris McCormick: Word2Vec Tutorial Part 2 - Negative Sampling  
   适合看“为什么负采样只更新少量权重”这一工程直觉，讲得很清楚。

4. TensorFlow 官方教程：Word2Vec  
   适合看 `tf.nn.nce_loss` 一类 API 如何落地，偏工程实战。

5. Ivan Bercovich: Word2Vec from Scratch  
   适合补充从零实现视角，帮助把“嵌入矩阵查表”和训练目标连起来。

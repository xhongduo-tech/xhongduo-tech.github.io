## 核心结论

Word2Vec 的 Skip-gram 任务是：给定中心词，预测它周围的上下文词。这里的“中心词”就是窗口中间那个词，“上下文词”就是它前后若干位置的词。

原始做法使用 softmax。softmax 是一种把所有候选词变成概率分布的方法，要求对整个词表都算一遍分数再归一化。当词表大小 $|V|=100000$ 时，一次训练样本就要访问 10 万个词向量，代价很高。

负采样（Negative Sampling）的核心改写是：不再问“真实上下文在整个词表里是哪一个”，而是问“这个词对是真实共现，还是噪声伪造”。这样，多分类问题被改成了 $K+1$ 个二分类问题。对每个正样本，只需要再采样 $K$ 个负样本，训练复杂度近似从和 $|V|$ 成正比，降到和 $K$ 成正比。

目标函数写成：

$$
\log \sigma(v_c^\top v_o) + \sum_{k=1}^{K}\mathbb{E}_{w_k \sim P_n(w)}\left[\log \sigma(-v_c^\top v_k)\right]
$$

其中，$\sigma(x)=\frac{1}{1+e^{-x}}$ 是 Sigmoid 函数，作用是把任意实数压到 $(0,1)$ 区间；$v_c$ 是中心词向量，$v_o$ 是真实上下文向量，$v_k$ 是负样本向量。

玩具例子可以直接看出它为什么快。假设中心词是 `king`，真实上下文是 `queen`，采样到 3 个负样本 `the`、`a`、`of`。负采样只更新 `king`、`queen`、`the`、`a`、`of` 这 5 个向量；而 softmax 需要把 `king` 与整个词表的所有输出向量都做点积，再算分母。

---

## 问题定义与边界

Skip-gram 的目标不是生成一句话，而是学习词的分布式表示。所谓“分布式表示”，就是把一个词表示成一个稠密向量，向量方向和长度承载语义与上下文信息。

给定中心词 $c$ 和上下文词 $o$，原始 softmax 建模为：

$$
P(o \mid c)=\frac{\exp(v_o^\top v_c)}{\sum_{w\in V}\exp(v_w^\top v_c)}
$$

问题出在分母。它要遍历整个词表 $V$。如果训练集中有上亿个中心词对，这个分母会成为主要开销。

负采样的边界也要说清楚。它不是精确计算 $P(o\mid c)$，而是在训练时用局部对比逼近一个好的词向量空间。因此：

- 它适合“需要高效训练 embedding”的场景。
- 它不适合“必须得到精确归一化概率”的场景。
- 它优化的是表示质量，不是严格的条件概率校准。

下面用一个量级对比说明差异。

| 方法 | 单个正样本需要访问的词 | 每对样本更新次数 | 梯度信号来源 |
|---|---:|---:|---|
| 全 softmax | $|V|$ 个 | 约 $|V|$ 次相关计算 | 全词表竞争 |
| 负采样 | $K+1$ 个 | $K+1$ 次 | 1 个正样本 + $K$ 个噪声样本 |

如果词表是 100k，且设 $K=5$，那么：

- softmax：一次样本要和 100000 个词打分。
- 负采样：一次样本只处理 6 个词。
- 两者都能训练出词向量，但计算路径完全不同。

这也是负采样最重要的适用边界：它是近似方法，目标是用更少计算换到接近的表示效果。

---

## 核心机制与推导

先看正样本项：

$$
\log \sigma(v_c^\top v_o)
$$

点积 $v_c^\top v_o$ 越大，$\sigma(v_c^\top v_o)$ 越接近 1，说明模型越相信“中心词和这个上下文词确实共现”。最大化这一项，等价于把正样本向量拉近。

再看负样本项：

$$
\log \sigma(-v_c^\top v_k)
$$

这里前面多了一个负号。若 $v_c^\top v_k$ 很大，说明中心词和负样本过于相似；乘上负号后，Sigmoid 会变小，损失变差。最大化这一项，等价于让 $v_c^\top v_k$ 变小，也就是把负样本推远。

所以它的训练逻辑很直接：

- 正样本：增大点积，拉近。
- 负样本：减小点积，推远。

这就是“局部二分类逼近全局分布”的实质。

玩具数值例子如下。设：

$$
v_c=[1,0],\quad v_o=[0.8,0.6],\quad v_{k_1}=[-0.2,0.4],\quad v_{k_2}=[0.1,-0.5]
$$

那么：

$$
v_c^\top v_o = 0.8
$$

所以正项是：

$$
\log \sigma(0.8)\approx \log(0.69)\approx -0.37
$$

第一个负样本：

$$
v_c^\top v_{k_1} = -0.2,\quad \log \sigma(-(-0.2))=\log \sigma(0.2)\approx -0.60
$$

第二个负样本：

$$
v_c^\top v_{k_2} = 0.1,\quad \log \sigma(-0.1)\approx -0.74
$$

这个结果说明：

- 正样本点积已经偏正，方向是对的，但还可以更大。
- 第一个负样本和中心词方向相反，所以问题不大。
- 第二个负样本点积仍是正数，说明它离中心词太近，训练会把它推远。

为什么负样本通常按 unigram$^{3/4}$ 分布采样？“unigram 分布”就是按词频采样。直接按词频采样会让 `the`、`of`、`a` 这类高频词出现过多；完全均匀采样又会让极低频噪声过多。把词频做 $3/4$ 次幂：

$$
P_n(w)\propto U(w)^{3/4}
$$

相当于压平高频词优势，但不完全抹掉频率信息。这是经验上效果较好的折中。

真实工程例子是搜索日志或推荐日志中的查询词训练。词表可能达到百万级，日志规模达到十亿级。如果仍用全 softmax，单步训练会大量消耗在分母归一化上；负采样只对少量候选做更新，吞吐量会高很多，更适合离线大规模 embedding 训练。

---

## 代码实现

下面给出一个可运行的 Python 版本，演示单个 Skip-gram 正样本配两个负样本时的损失与梯度方向。它不是完整训练器，但足够体现机制。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def neg_sampling_loss(v_c, v_o, negatives):
    pos_score = dot(v_c, v_o)
    loss = -math.log(sigmoid(pos_score))
    for v_k in negatives:
        neg_score = dot(v_c, v_k)
        loss += -math.log(sigmoid(-neg_score))
    return loss

# 玩具例子
v_c = [1.0, 0.0]
v_o = [0.8, 0.6]
negatives = [
    [-0.2, 0.4],
    [0.1, -0.5],
]

loss = neg_sampling_loss(v_c, v_o, negatives)
print(round(loss, 4))

# 基本断言：正样本更对齐时，loss 应更小
better_v_o = [1.2, 0.0]
worse_v_o = [0.2, 0.0]

loss_better = neg_sampling_loss(v_c, better_v_o, negatives)
loss_worse = neg_sampling_loss(v_c, worse_v_o, negatives)

assert loss_better < loss_worse
assert loss > 0
```

如果用 TensorFlow，工程上通常分两步：

1. 用 `skipgrams` 从语料生成中心词-上下文词对。
2. 用 `tf.random.log_uniform_candidate_sampler` 采样负样本，再用 `sigmoid_cross_entropy` 计算损失。

示意代码如下：

```python
import tensorflow as tf

num_ns = 5
distortion = 0.75
vocab_size = 100000
embed_dim = 128

center_ids = tf.constant([[42]])     # shape: [batch_size, 1]
context_ids = tf.constant([[108]])   # shape: [batch_size, 1]

sampling_values = tf.random.log_uniform_candidate_sampler(
    true_classes=context_ids,
    num_true=1,
    num_sampled=num_ns,
    unique=True,
    range_max=vocab_size
)

negative_ids = sampling_values.sampled_candidates  # [num_ns]

in_embed = tf.Variable(tf.random.normal([vocab_size, embed_dim]))
out_embed = tf.Variable(tf.random.normal([vocab_size, embed_dim]))

center_vec = tf.nn.embedding_lookup(in_embed, tf.squeeze(center_ids, axis=1))
pos_vec = tf.nn.embedding_lookup(out_embed, tf.squeeze(context_ids, axis=1))
neg_vec = tf.nn.embedding_lookup(out_embed, negative_ids)

pos_logits = tf.reduce_sum(center_vec * pos_vec, axis=1, keepdims=True)
neg_logits = tf.einsum("bd,nd->bn", center_vec, neg_vec)

pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(pos_logits), logits=pos_logits
)
neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(neg_logits), logits=neg_logits
)

loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
```

这里 `num_ns` 就是负样本数量，常见范围是 2 到 20。`log_uniform_candidate_sampler` 本质上是在做近似高频优先采样，工程里常用来模拟负采样分布。

---

## 工程权衡与常见坑

负采样不是“参数越大越好”的算法，关键在于速度和近似质量的平衡。

| 场景 | 语料规模 | 建议 `num_ns` | 原因 |
|---|---:|---:|---|
| 小语料实验 | 10M 词 | 5-20 | 数据少，需更多负样本补足判别信号 |
| 大语料训练 | 1B 词 | 2-5 | 数据多，少量负样本已足够，优先吞吐 |

常见误区如下。

| 问题/误区 | 后果 | 缓解策略 |
|---|---|---|
| 负样本数小于 2 | 近似偏差大，词向量区分度不足 | 小语料设 `num_ns>=5` |
| 负样本数过大 | 训练变慢，收益递减 | 先从 5 或 10 开始调 |
| 直接按词频采样 | 高频虚词主导噪声 | 使用 unigram$^{3/4}$ |
| 词表清洗不足 | 噪声 token 太多，训练不稳定 | 先做低频截断和标准化 |
| 只看 loss 不看近邻词 | 可能 loss 降了但语义没学好 | 抽样检查最近邻结果 |

再给一个真实工程例子。假设你在做电商搜索召回，用用户搜索词、点击词、商品标题词一起训练 embedding。这个任务通常有三个特点：

- 词表很大，品牌词、型号词、错拼词都很多。
- 高频词很多，如“男”“女”“新款”“正品”。
- 训练样本量极大，日志按天滚动新增。

这种情况下，负采样非常合适，因为它把每个正样本的更新控制在少数词上，吞吐量高；同时通过 $3/4$ 采样分布，不至于让“新款”“包邮”这类高频词把负样本预算全部吃掉。

---

## 替代方案与适用边界

负采样不是唯一方案，常见替代方法有两个：全 softmax 和 hierarchical softmax。hierarchical softmax 是把词表组织成树，通过树路径做概率计算，把复杂度从 $O(|V|)$ 降到近似 $O(\log |V|)$。

可以用下面这个条件表做快速判断。

| 条件 | 更合适的方案 | 原因 |
|---|---|---|
| 词表小于 5k | 全 softmax | 实现简单，精确概率可得 |
| 词表 5k 到 20k | softmax 或 hierarchical softmax | 取决于是否关心速度 |
| 词表大于 20k | 负采样 | 训练成本更可控 |
| 词表百万级且只关心 embedding 质量 | 负采样 | 局部更新最省算力 |
| 需要严格概率输出 | 全 softmax | 负采样不是归一化概率模型 |

所以边界可以概括成一句话：

- 小词表、需要精确概率：优先 softmax。
- 大词表、只想高效学表示：优先负采样。
- 中间地带、想兼顾：可以考虑 hierarchical softmax。

例如，一个 5k 词表的教学任务或小型分类预训练，直接 softmax 更直观；但一个 1M 词表的日志系统，如果仍坚持全 softmax，训练代价通常没有工程意义。

---

## 参考资料

1. Tomas Mikolov 等，Word2Vec 原始论文与 Negative Sampling 相关论文。重点是 Skip-gram 目标、负采样形式和符号定义。访问时建议对照论文中的 $v_c$、$v_o$、$P_n(w)$ 记号阅读。  
2. TensorFlow 官方 `word2vec` 教程，重点是 `skipgrams`、`log_uniform_candidate_sampler`、`num_ns` 的工程实现方式。适合把数学公式映射到训练代码。https://www.tensorflow.org/text/tutorials/word2vec  
3. Vignesh 的 Word2Vec 数学推导文章，重点是从 softmax 到 negative sampling 的直观解释，以及“正样本拉近、负样本推远”的几何理解。https://vignesh.bearblog.dev/word2vec/  
4. ReadMedium 关于 Word2Vec 超参数的经验总结，重点是负样本数量和 unigram$^{3/4}$ 分布的实践建议。https://readmedium.com/the-word2vec-hyperparameters-e7b3be0d0c74

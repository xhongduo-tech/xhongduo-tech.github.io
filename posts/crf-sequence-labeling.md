## 核心结论

线性链 CRF（Conditional Random Field，条件随机场）是一种序列标注模型：给定输入序列 $x$，它直接建模整条标签序列 $y$ 的条件概率 $P(y|x)$。

它不是“逐 token 分类器”。逐 token 分类器会分别判断每个位置的标签，容易忽略标签之间的合法连接关系。CRF 会把“当前位置像不像某个标签”和“相邻标签能不能接上”放进同一个全局打分函数里。

在线性链 CRF 中，一条标签路径的概率通常写成：

$$
P(y|x)=\frac{1}{Z(x)}\exp\left(\sum_k \lambda_k F_k(x,y)\right)
$$

其中 $x=(x_1,\dots,x_T)$ 是输入序列，$y=(y_1,\dots,y_T)$ 是标签序列，$F_k(x,y)$ 是整条路径上的特征函数，$\lambda_k$ 是模型学到的权重，$Z(x)$ 是归一化常数。

中文命名实体识别中，`B-PER -> I-PER` 通常是合理路径，表示一个人名实体从开始到内部继续；`B-PER -> I-ORG` 通常是不合理路径，因为人名开头后面不应直接接组织名内部。CRF 可以学习这种转移约束，把合理路径分数抬高，把不合理路径分数压低。

| 对比项 | 局部独立分类 | 线性链 CRF |
|---|---|---|
| 建模对象 | 每个位置的标签 | 整条标签序列 |
| 是否建模相邻标签 | 通常不建模 | 建模 |
| 推理方式 | 每个位置取最高分 | 找全局最高分路径 |
| 典型问题 | 容易产生非法 BIO 序列 | 能减少非法转移 |
| 适合任务 | 标签弱相关任务 | NER、分词、词性标注 |

---

## 问题定义与边界

序列标注是指：输入一个长度为 $T$ 的序列 $x=(x_1,\dots,x_T)$，输出一个同长度的标签序列 $y=(y_1,\dots,y_T)$，每个输入位置都对应一个标签。

例如输入“我 爱 北 京 天 安 门”，可以按字标注：

| 字 | 我 | 爱 | 北 | 京 | 天 | 安 | 门 |
|---|---|---|---|---|---|---|---|
| 标签 | O | O | B-LOC | I-LOC | I-LOC | I-LOC | I-LOC |

BIO 是一种实体边界标注方法：`B-X` 表示类型为 `X` 的实体开始，`I-X` 表示类型为 `X` 的实体内部，`O` 表示不是实体。BIOES 在 BIO 基础上增加 `E-X` 表示实体结束，`S-X` 表示单字实体。

| 概念 | 定义 | 例子 | 说明 |
|---|---|---|---|
| 输入 x | 待标注的 token 序列 | 我 / 爱 / 北京 | 可以是字、词或子词 |
| 输出 y | 与输入等长的标签序列 | O / O / B-LOC | 每个位置一个标签 |
| 标签依赖 | 相邻标签之间的合法连接关系 | B-PER 后常接 I-PER | CRF 重点建模对象 |
| 典型任务 | 需要连续标注的任务 | NER、分词、词性标注 | 标签之间通常有结构 |

CRF 的边界也要明确。它解决的是“有标签依赖的判别式序列建模”。判别式模型是指直接建模输出在输入给定时的概率，比如 $P(y|x)$，而不是先建模输入和输出如何共同生成。

CRF 不负责生成输入句子，也不是所有深度模型的替代品。它擅长处理标签之间有明确转移约束的任务，但它本身不擅长从原始文本中自动学习复杂语义表示。真实工程里，CRF 常被放在 BiLSTM、Transformer 等编码器后面，用来约束最终标签序列。

---

## 核心机制与推导

线性链 CRF 的核心是“路径打分”。路径是指一整条候选标签序列，例如：

`B-PER -> I-PER -> O`

特征函数是一个判断条件是否成立的函数。白话说，它就是告诉模型“某个局部现象有没有出现”。例如：

- 当前位置字是“张”，当前标签是 `B-PER`
- 上一个标签是 `B-PER`，当前标签是 `I-PER`
- 上一个标签是 `B-PER`，当前标签是 `I-ORG`

CRF 把局部特征沿整条序列累加：

$$
F_k(x,y)=\sum_t f_k(y_t,y_{t-1},x,t)
$$

这里 $f_k(y_t,y_{t-1},x,t)$ 是第 $k$ 个局部特征函数，可能同时依赖当前标签 $y_t$、上一个标签 $y_{t-1}$、输入序列 $x$ 和当前位置 $t$。

整条路径的条件概率为：

$$
P(y|x)=\frac{1}{Z(x)}\exp\left(\sum_k \lambda_k F_k(x,y)\right)
$$

归一化常数为：

$$
Z(x)=\sum_{y'}\exp\left(\sum_k \lambda_k F_k(x,y')\right)
$$

$Z(x)$ 会遍历所有可能标签序列 $y'$，把它们的指数分数加起来。它的作用是把任意路径分数变成合法概率，使所有路径概率之和等于 1。

训练时，模型学习每个特征的权重 $\lambda_k$。推理时，模型不需要真的输出每条路径概率，只需要找到分数最高的路径：

$$
\hat{y}=\arg\max_y \sum_k \lambda_k F_k(x,y)
$$

这个 $\arg\max$ 通常用 Viterbi 算法求解。Viterbi 是一种动态规划算法，白话说就是：每个位置只保留“到达当前标签的最好历史路径”，逐步推进到最后一个位置，再回溯出全局最优标签序列。

路径打分与归一化流程可以写成：

```text
输入 x
  -> 枚举候选标签路径 y
  -> 计算每条路径的特征总分 score(x, y)
  -> exp(score) 得到非归一化权重
  -> 用 Z(x) 归一化为 P(y|x)
  -> 推理时选择 score 最高的路径
```

玩具例子：长度为 2 的序列，候选标签有 `B-PER`、`I-PER`、`I-ORG`、`O`。

| 路径 | 位置1分 | 位置2分 | 转移分 | 总分 |
|---|---:|---:|---:|---:|
| `B-PER -> I-PER` | 2 | 1 | 2 | 5 |
| `B-PER -> I-ORG` | 2 | 4 | -10 | -4 |
| `O -> O` | 0 | 0 | 0 | 0 |

只看第二个位置，`I-ORG` 的局部分数是 4，比 `I-PER` 的 1 更高。但 `B-PER -> I-ORG` 的转移分是 -10，整条路径总分变成 -4。全局最优反而是 `B-PER -> I-PER`，总分为 5。

这说明：单点分数高，不等于整条序列最优。CRF 的价值就在于按整条路径选标签。

---

## 代码实现

实现 CRF 序列标注时，通常分三步：构造特征、训练模型、解码预测。

数据格式是一句话对应一串 token 和一串标签：

```text
tokens = ["张", "三", "去", "北", "京"]
labels = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
```

特征模板是从当前位置和上下文中抽取可学习信号的方法。模板可以包含当前字、前一个字、后一个字、是否数字、是否中文、词性等。词性是指词在句子里的语法类别，例如名词、动词、地名词等。

下面代码不依赖外部 CRF 库，而是用一个最小 Viterbi 解码示例说明“局部分数 + 转移分数 = 全局路径分数”的推理过程。

```python
from math import inf

def extract_features(tokens, i):
    ch = tokens[i]
    return {
        "bias": 1,
        "char": ch,
        "prev": tokens[i - 1] if i > 0 else "<BOS>",
        "next": tokens[i + 1] if i + 1 < len(tokens) else "<EOS>",
        "is_digit": ch.isdigit(),
        "is_chinese": "\u4e00" <= ch <= "\u9fff",
    }

def viterbi(emission_scores, transition_scores, labels):
    n = len(emission_scores)
    dp = [{} for _ in range(n)]
    back = [{} for _ in range(n)]

    for y in labels:
        dp[0][y] = emission_scores[0].get(y, -inf)
        back[0][y] = None

    for t in range(1, n):
        for y in labels:
            best_prev = None
            best_score = -inf
            for prev_y in labels:
                score = (
                    dp[t - 1][prev_y]
                    + transition_scores.get((prev_y, y), -inf)
                    + emission_scores[t].get(y, -inf)
                )
                if score > best_score:
                    best_score = score
                    best_prev = prev_y
            dp[t][y] = best_score
            back[t][y] = best_prev

    last = max(labels, key=lambda y: dp[-1][y])
    path = [last]
    for t in range(n - 1, 0, -1):
        path.append(back[t][path[-1]])
    return list(reversed(path)), dp[-1][last]

tokens = ["张", "三"]
labels = ["B-PER", "I-PER", "I-ORG", "O"]
x_seq = [extract_features(tokens, i) for i in range(len(tokens))]

emission_scores = [
    {"B-PER": 2, "O": 0, "I-PER": -5, "I-ORG": -5},
    {"I-PER": 1, "I-ORG": 4, "O": 0, "B-PER": -5},
]
transition_scores = {
    ("B-PER", "I-PER"): 2,
    ("B-PER", "I-ORG"): -10,
    ("O", "O"): 0,
    ("B-PER", "O"): -1,
}

path, score = viterbi(emission_scores, transition_scores, labels)

assert path == ["B-PER", "I-PER"]
assert score == 5
assert x_seq[0]["char"] == "张"
```

真实工程里不会手写完整训练过程，通常使用 CRFsuite、sklearn-crfsuite 或深度学习框架里的 CRF 层。接口形态大致如下：

```python
train_sequences = [
    [extract_features(["张", "三", "去", "北", "京"], i) for i in range(5)]
]
train_labels = [
    ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
]

model.fit(train_sequences, train_labels)

test_x = [extract_features(["李", "四", "到", "上", "海"], i) for i in range(5)]
y_pred = model.predict([test_x])
```

真实工程例子：中文命名实体识别系统中，上游分词、词典、词性标注和字符特征已经比较稳定。CRF 接收这些特征后，学习 `B-LOC -> I-LOC`、`B-PER -> I-PER`、`O -> B-ORG` 等转移模式。相比单点分类，它能减少 `B-PER -> I-ORG`、`O -> I-LOC` 这类明显非法或低质量的输出。

---

## 工程权衡与常见坑

CRF 的效果高度依赖特征设计和标签规范。特征太少，模型只能看到局部表面信息；标签体系混乱，模型学到的转移权重也会混乱。

| 问题 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 只用单 token 特征 | 实体边界断裂 | 模型缺少上下文和转移信息 | 加入前后 token、词性、词典、转移特征 |
| 标签体系混用 | 边界识别不稳定 | BIO 和 BIOES 的转移规则不同 | 训练前统一标注规范 |
| 贪心解码 | 局部看似合理，全局不合法 | 每一步只看当前最高分 | 使用 Viterbi 全局解码 |
| 数据太少导致过拟合 | 训练集很好，测试集差 | 转移权重记住了少量样本 | 加正则、扩数据、合并稀有标签 |
| 特征模板过稀 | 很多特征只出现一次 | 权重难以可靠估计 | 控制模板粒度，过滤低频特征 |

BIO 和 BIOES 混用是常见坑。比如同一任务里既出现 `B-PER I-PER`，又出现 `B-PER E-PER`。前者属于 BIO，后者属于 BIOES。两套规则混在一起后，模型会同时学习两种边界解释，转移表难以稳定，最后常见结果是实体结束位置识别变差。

另一个坑是把 CRF 理解成“最后加一层就一定更好”。如果上游编码器已经很强、数据量很大、标签转移约束又不明显，CRF 的收益可能很小，甚至增加训练和推理复杂度。工程上要用验证集比较，而不是按经验固定加 CRF。

---

## 替代方案与适用边界

CRF 适合标注规则明确、标签依赖强、特征工程可控的传统序列任务，例如命名实体识别、中文分词、词性标注。它尤其适合标签结构稳定、非法转移需要显式压制的场景。

它不适合承担复杂语义表示学习。CRF 本身主要处理标签之间的结构约束。如果输入特征很弱，它不能凭空理解上下文语义。现代 NLP 中更常见的做法是：用 BiLSTM、BERT、RoBERTa 等模型先把文本编码成上下文表示，再用 CRF 约束最终标签序列。

| 方法 | 是否建模标签转移 | 是否判别式 | 是否适合复杂表示学习 | 适用场景 |
|---|---|---|---|---|
| HMM | 是 | 否 | 否 | 传统生成式序列建模 |
| 逐点分类器 | 否 | 是 | 取决于编码器 | 标签依赖弱的分类式标注 |
| 线性链 CRF | 是 | 是 | 否 | 特征工程可控、标签约束强的任务 |
| BiLSTM-CRF | 是 | 是 | 中等 | 需要上下文表示和标签约束的 NER |
| Transformer-CRF | 是 | 是 | 强 | 大规模语义表示 + 结构化输出 |

HMM 是隐马尔可夫模型，白话说，它假设隐藏标签先转移，再生成观测词。CRF 与 HMM 的关键区别是：CRF 直接建模 $P(y|x)$，可以自由使用输入侧的复杂特征，不需要建模输入如何生成。

逐点分类器适合标签之间关系不强的任务。如果每个位置的标签基本独立，CRF 的转移建模价值有限。

BiLSTM-CRF 和 Transformer-CRF 更适合现代工程。前者用循环神经网络学习上下文表示，后者用 Transformer 学习更强的语义表示，CRF 负责最后的标签路径约束。简单说：如果只想让模型记住标签之间的合法接法，CRF 很合适；如果还想让模型自己学复杂语义表示，通常要把 CRF 放到更强的编码器后面。

---

## 参考资料

1. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/handle/20.500.14332/6188)
2. [An Introduction to Conditional Random Fields](https://doi.org/10.1561/2200000013)
3. [CRFsuite Manual](https://www.chokkan.org/software/crfsuite/manual.html)
4. [CRFsuite Tagger API](https://www.chokkan.org/software/crfsuite/api/classCRFSuite_1_1Tagger.html)
5. [Apache MADlib CRF Documentation](https://madlib.apache.org/docs/v1.4/group__grp__crf.html)

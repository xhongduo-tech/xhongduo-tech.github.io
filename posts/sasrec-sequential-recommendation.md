## 核心结论

序列推荐的目标，是用用户已经发生的行为序列，预测下一步最可能发生的动作。这里的“序列”可以理解为按时间排好的点击、浏览、收藏、加购、播放等事件；“推荐”不是直接生成一句话，而是输出一个候选物品上的概率分布或打分列表。

Transformer 适合做这件事，因为它的自注意力机制可以让当前位置直接查看很久之前的行为，不需要像传统 RNN 那样一步一步传递状态。这里的“自注意力”可以白话理解为：序列里的每个位置都会主动判断，历史里哪些动作和当前预测最相关。

最关键的两件事是：

1. 因果掩码。它把未来位置全部遮住，保证模型在第 $t$ 步只能看 $1 \sim t-1$ 的历史，不能偷看答案。
2. 位置编码。它把“先发生”和“后发生”的顺序信息注入向量，否则模型只知道有哪些行为，不知道顺序。

可以先用一句最直白的话理解整件事：

历史序列 → Transformer（位置编码 + 因果 mask）→ 下一步物品分布

对新手而言，可以把它看成“把用户行为序列当作 token 串”。例如用户依次点击了 `手机壳 -> 充电器 -> 数据线`，模型会在只看这三步历史的前提下，预测第四步更可能是 `快充头`、`收纳包`，还是别的商品。

这种方法适合在线流量池中的召回补充、粗排、精排前特征建模，也常用作“下一物品预测”模型的主体。

---

## 问题定义与边界

我们先把问题写成一个清楚的数学形式。设用户到当前时刻的行为序列为：

$$
a_1, a_2, \dots, a_{t-1}
$$

目标是估计：

$$
P(a_t \mid a_1, a_2, \dots, a_{t-1})
$$

也就是：已知历史行为，下一步动作 $a_t$ 是什么。

这里的“动作”通常不是抽象事件，而是具体 item ID，譬如商品、视频、文章、歌曲。工程里最常见的做法，是把序列截成固定窗口长度 $L$，只保留最近 $L$ 个行为作为输入。原因很简单：显存、延迟、训练稳定性都有限，不可能无限长。

一个玩具例子：

用户在电商 App 中依次浏览了：

- 第 1 步：运动鞋
- 第 2 步：跑步袜
- 第 3 步：速干短裤

模型的任务不是解释这个用户为什么喜欢跑步，而是只根据前三步，计算第四步最可能点击哪些商品。比如：

- 跑步腰包：0.31
- 运动手表：0.22
- 压缩裤：0.18

边界必须明确，否则模型会“看起来很准，实际上不可上线”。

| 信息类型 | 是否允许进入当前步预测 | 原因 |
|---|---|---|
| 当前步之前的历史行为 | 允许 | 线上推理时真实可见 |
| 当前步之后的未来行为 | 禁止 | 属于答案泄露，训练和上线不一致 |
| 离线统计特征，如 item 热度 | 允许，但要按时间切分 | 只能使用当前时刻之前可观测到的统计 |
| 用整条序列反向编码出的未来信息 | 禁止用于线上 causal 预测 | 会导致离线效果虚高 |

这里的“causal”可以白话理解为“遵守时间因果方向”，只能从过去推现在，不能从未来倒灌信息。

真实工程里，这个边界经常被破坏。例如离线构造样本时，直接把完整用户序列送进双向 Transformer，再取中间位置做打分，表面 AUC 很高，但线上用户还没发生未来行为，这个模型实际上无法等价部署。

---

## 核心机制与推导

### 1. 输入表示

先把每个 item ID 映射成向量，得到输入矩阵：

$$
X \in \mathbb{R}^{L \times d}
$$

这里 $L$ 是序列长度，$d$ 是隐层维度。向量可以理解为“机器内部使用的数字表示”。

如果只有 $X$，模型并不知道第 1 个点击和第 10 个点击谁先谁后，所以要加入位置编码 $P$：

$$
H_0 = X + P
$$

“位置编码”可以白话理解为：给每个位置加一个专门表示顺序的信息，让模型知道这是第几个行为。

### 2. 自注意力计算

Transformer 的一层注意力会把输入映射成三组矩阵：

$$
Q = H_0 W_Q,\quad K = H_0 W_K,\quad V = H_0 W_V
$$

其中：

- $Q$ 是 Query，表示“我当前想找什么”
- $K$ 是 Key，表示“我这个位置有什么信息可供匹配”
- $V$ 是 Value，表示“真正要被聚合过去的内容”

单头注意力写成：

$$
\text{Att}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V
$$

这里的 $M$ 就是因果掩码。定义为：

$$
M_{ij}=
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

意思是：第 $i$ 个位置只能看自己和自己之前的位置，不能看后面的位置。

### 3. 最小数值例子

按题目给的最小例子，设：

$$
Q=K=V=
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

则相似度矩阵为：

$$
QK^\top=
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

缩放后得到：

$$
\frac{QK^\top}{\sqrt{2}}
=
\begin{bmatrix}
0.707 & 0 & 0.707 \\
0 & 0.707 & 0.707 \\
0.707 & 0.707 & 1.414
\end{bmatrix}
$$

如果加入因果掩码：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$

那么：

- 第 1 行只能看第 1 个位置
- 第 2 行只能看第 1、2 个位置
- 第 3 行可以看第 1、2、3 个位置

这就把“只看历史”的规则硬编码进 softmax 之前的分数里了。因为 $-\infty$ 经过 softmax 后概率会变成 0。

### 4. 为什么位置编码必须和 mask 一起看

如果没有位置编码，序列 `[A, B, C]` 和 `[C, B, A]` 在“只看集合不看顺序”的极端情况下可能很难区分。但推荐任务里，顺序本身就是信号：

- 先看入门书再看进阶书，和反过来，含义不同
- 先搜手机再搜手机壳，和先搜手机壳再搜手机，购买意图不同

所以在实践里，因果掩码负责“不能看未来”，位置编码负责“知道历史顺序”。两者缺一个都不完整。

位置编码示意可以写成：

`item embedding + position embedding -> masked self-attention -> next-item logits`

如果使用 RoPE，思路不是把位置向量直接加到输入上，而是在注意力里对 $Q$、$K$ 做旋转，让不同位置携带相对位置信息。RoPE 可以白话理解为：把“第几个位置”编码进向量方向里，而不是简单相加。

---

## 代码实现

下面给一个可运行的最小 Python 版本，演示“嵌入后的矩阵 + 因果 mask + softmax + 输出”。它不是完整训练代码，但足够说明核心逻辑。

```python
import math

def matmul(a, b):
    rows, cols, inner = len(a), len(b[0]), len(b)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(inner))
    return out

def transpose(x):
    return [list(row) for row in zip(*x)]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def causal_mask_scores(scores):
    n = len(scores)
    masked = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(scores[i][j] if j <= i else -1e9)
        masked.append(row)
    return masked

def attention(q, k, v):
    d = len(q[0])
    scores = matmul(q, transpose(k))
    scores = [[x / math.sqrt(d) for x in row] for row in scores]
    masked = causal_mask_scores(scores)
    weights = [softmax(row) for row in masked]
    output = matmul(weights, v)
    return weights, output

Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
K = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
V = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]

weights, out = attention(Q, K, V)

# 第1行只能看自己
assert abs(weights[0][0] - 1.0) < 1e-6
assert weights[0][1] < 1e-6
assert weights[0][2] < 1e-6

# 第2行不能看未来的第3个位置
assert weights[1][2] < 1e-6

# 第3行可以看全部历史
assert abs(sum(weights[2]) - 1.0) < 1e-6

print("attention weights:", weights)
print("outputs:", out)
```

新手可以这样理解这段代码：

1. 先准备好序列每个位置的向量表示。
2. 计算两两相似度分数。
3. 用因果 mask 把未来位置改成一个极小值。
4. 做 softmax，把分数变成概率。
5. 用这些概率对历史信息加权求和，得到当前位置的新表示。

如果把它扩成接近工程版的伪代码，大致是：

```python
# seq_ids: [item_1, item_2, ..., item_L]
x = item_embedding(seq_ids)
x = x + position_embedding(range(L))   # 或者对 q/k 使用 RoPE

mask = lower_triangular_mask(L)        # 只允许看 j <= i
h = transformer_blocks(x, attn_mask=mask)

last_hidden = h[-1]                    # 取最后一个位置代表“当前下一步预测”
logits = last_hidden @ item_embedding_table.T
probs = softmax(logits)
```

如果使用 RoPE，伪代码会变成：

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv

q = apply_rope(q, positions)
k = apply_rope(k, positions)

scores = (q @ k.T) / sqrt(d)
scores = scores + causal_mask
weights = softmax(scores)
out = weights @ v
```

这里要强调一个工程原则：训练时怎么构造位置和 mask，推理时就必须怎么构造。否则线上离线分布会错位。

真实工程例子可以放在短视频或电商场景里。比如首页推荐中，用户最近 50 次行为包括“搜索耳机、点击蓝牙耳机、查看降噪评测、收藏某型号”。序列 Transformer 会把这些动作编码成一个用户当前兴趣状态，再输出一个对全量商品或候选集合的打分。这个分数可以直接用于候选过滤，也可以和价格、CTR、库存等特征一起送入后续排序层。

---

## 工程权衡与常见坑

序列推荐不是“把 Transformer 套上去”就结束，真正难的是让训练目标、线上输入、长度策略一致。

| 坑 | 影响 | 解决方案 |
|---|---|---|
| 没有严格使用因果掩码 | 未来信息泄露，离线指标虚高 | 训练、验证、推理统一使用 causal mask |
| 位置编码和推理长度不匹配 | 长序列外推差，线上效果抖动 | 固定窗口、使用 RoPE/relative PE/ALiBi，并做长度外推验证 |
| 序列过长超出训练上限 | 早期行为被截断，长期兴趣丢失 | 滑窗、多尺度序列、长短兴趣塔结合 |

### 坑 1：未来泄露

最常见的错误是“为了省事直接调用默认 attention”，结果没有加下三角 mask。这样模型在第 $t$ 个位置训练时看到了 $t+1, t+2$ 的未来行为，等于提前抄答案。离线 recall 很可能好看，但线上必掉。

### 坑 2：位置外推失败

绝对位置编码在训练长度是 50、推理长度是 200 时，可能会出现没见过的位置索引，或者高位位置表达失真。RoPE 和相对位置编码通常更稳，但也不是自动解决，需要真实验证长序列外推。

### 坑 3：只建模最近兴趣，忘了长期偏好

序列模型天然偏向最近行为，因为最近 token 更密集、更强相关。但一些场景里长期偏好很重要，例如用户长期关注“摄影器材”，只是最近临时看了几次“旅行箱”。如果只保留短窗口，模型会过度响应短期噪声。

常见补法有两类：

- 滑窗内建模最近兴趣
- 额外引入长期兴趣向量，如用户画像、历史类目分布、长期塔输出

### 坑 4：训练目标和业务目标不完全一致

下一物品预测优化的是交叉熵，不一定等价于最终 GMV、时长、转化率。它通常适合作为强表征学习阶段，后面还需要与多目标排序结合。

---

## 替代方案与适用边界

Transformer 不是唯一方案，也不是所有场景都值得上。

| 方案 | 优点 | 局限 | 适用场景 |
|---|---|---|---|
| 序列 Transformer | 长距离依赖强，并行训练好 | 计算量高，对 mask/位置敏感 | 中长序列、流量较大、需要高表达能力 |
| RNN / GRU4Rec | 结构简单，推理成本低 | 长依赖较弱，并行性差 | 序列短、资源紧张、快速上线 |
| BERT4Rec | 双向建模强，离线表征好 | 线上下一步预测不能直接照搬 | 离线表征学习、重排特征、预训练 |

GRU4Rec 可以白话理解为“按时间一步步读序列的 RNN 推荐模型”。它在序列较短、延迟预算紧时仍然有价值，因为实现简单、成本更可控。

BERT4Rec 的重点是双向 self-attention。所谓“双向”，就是训练时一个位置可以同时看左边和右边上下文。这对离线建模很强，但对线上“预测下一步”有天然边界，因为线上没有右边未来。

给新手一个直白说法：

BERT4Rec 在训练时像做“完形填空”，会同时看左右；但真实线上下一步预测只能看左边历史，所以不能原样部署成 causal 推荐器。

因此，BERT4Rec 更适合：

- 离线预训练 item/user 表征
- 给排序模型提供序列特征
- 用在允许双向上下文的离线任务

而当你明确要做“当前时刻预测下一步点击”，causal Transformer 更直接，也更符合线上约束。

一个真实工程判断标准是：

- 如果你的序列长度通常只有 5 到 10，且服务延迟非常紧，GRU4Rec 或更简单的 session-based 模型可能更划算。
- 如果用户行为链较长，且“很久之前的一次行为会影响现在”，比如内容消费、电商复购、音乐连续收听，Transformer 的收益通常更明显。
- 如果你需要把序列表示接到双塔召回或精排模型，序列 Transformer 输出的用户状态向量往往更通用。

---

## 参考资料

1. Sun et al. *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*. CIKM 2019.  
   贡献：给出了把 Transformer 引入推荐序列建模的经典工程路线，尤其适合理解双向建模与 Cloze 训练。

2. *Transformers for time-series forecasting: A comprehensive survey through 2024*.  
   贡献：对自注意力、因果注意力、位置编码等机制有系统整理，可用于理解本文中的公式与 mask 推导。

3. *Behind RoPE: How Does Causal Mask Encode Positional Information?* 2024.  
   贡献：帮助理解 RoPE 与 causal mask 的交互，不把“位置编码”误认为只是简单加一个位置向量。

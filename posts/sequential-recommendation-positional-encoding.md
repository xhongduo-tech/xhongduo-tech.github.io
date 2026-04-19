## 核心结论

序列推荐的位置编码，是给序列模型补充“顺序”和“时间节奏”的信息。序列模型只看到一串物品 ID 时，并不知道哪个行为更早、两个行为之间隔了多久、这些行为是否发生在同一个晚间会话里。位置编码的作用就是把这些信息注入输入表示和注意力计算。

推荐场景里的“位置”不等于自然语言里的“第几个 token”。文本中第 10 个词和第 11 个词通常只表示相邻关系；推荐中第 10 次点击和第 11 次点击可能只隔 30 秒，也可能隔 3 天。前者更像一次短会话中的连续兴趣，后者可能已经跨过了兴趣变化、预算变化或场景变化。

一个更完整的输入表示可以写成：

$$
x_i = e(v_i) + p_i + h_i
$$

其中，$e(v_i)$ 是第 $i$ 个行为对应物品的向量，表示“点了什么”；$p_i$ 是绝对位置编码，表示“这是第几个行为”；$h_i$ 是层次时间编码，表示“发生在几点、星期几、哪一类周期里”。如果再在注意力里加入相对时间差 $\Delta t_{ij}$，模型就能同时理解“顺序”和“间隔”。

| 编码类型 | 解决的问题 | 推荐场景中的含义 |
|---|---|---|
| 绝对位置编码 | 第几个行为 | A 在 B 前面，C 是最近行为 |
| 相对时间编码 | 两个行为隔多久 | 5 分钟内连点与 5 天后再点不同 |
| 可学习位置编码 | 让模型自己学习位置模式 | 最近 5 个行为可能比更早行为重要 |
| 层次时间编码 | 日/周周期 | 晚间活跃、周末复购、工作日浏览 |

玩具例子：文本里的第 10 个词和第 11 个词，间隔基本恒定；推荐里的第 10 次点击和第 11 次点击，可能隔 30 秒，也可能隔 3 天。所以序列推荐不能只问“第几个”，还要问“隔了多久”。

真实工程例子：短视频推荐中，用户晚上 20:00 到 23:00 的点击密度通常更高，周末观看时长也可能更长。只用绝对位置编码，模型只能知道用户最近看过什么；加入相对时间差和 hour/day/week 编码后，模型才能区分“刚刚连续刷了 6 个篮球视频”和“上周也看过篮球视频”这两种信号。

---

## 问题定义与边界

本文讨论的是序列推荐中的位置与时间编码，重点是 Transformer 类 self-attention 模型。self-attention 是一种让序列中任意两个位置互相计算相关性的机制，它不会天然知道顺序，必须通过额外编码补充位置信息。

设一个用户的历史行为序列为：

$$
(v_1,t_1),(v_2,t_2),\dots,(v_n,t_n)
$$

其中 $v_i$ 是第 $i$ 个行为对应的物品，$t_i$ 是行为发生时间。模型的目标通常是预测下一个物品，或者从候选集中给出 Top-K 推荐结果。

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $v_i$ | 第 $i$ 个行为的物品 | 用户第几次点了哪个商品或视频 |
| $t_i$ | 第 $i$ 个行为的时间戳 | 这个行为发生在什么时候 |
| $\Delta t_{ij}=t_i-t_j$ | 两个行为的时间差 | 当前行为和历史行为隔了多久 |
| $p_i$ | 绝对位置编码 | 第 $i$ 个位置的顺序信息 |
| $h_i$ | 层次时间编码 | 小时、星期、月份等周期信息 |
| $a_{ij}$ | 注意力打分 | 位置 $i$ 对位置 $j$ 有多关注 |

新手例子：用户行为是 `A@08:00 -> B@08:05 -> C@21:00`，模型要预测 `C` 后可能点什么。只看顺序，模型知道 `C` 最靠后，`B` 在 `C` 前面，`A` 更早。但真正有用的信息还包括：`A` 和 `B` 是早晨连续行为，`C` 是晚上行为，`B` 到 `C` 间隔很长。

本文不展开所有推荐特征工程。例如用户画像、物品画像、召回策略、排序损失、多目标优化都不是重点。本文只讨论：在序列模型中，如何把位置、时间差和时间周期编码进模型。

---

## 核心机制与推导

最基础的 Transformer 输入通常是物品向量加位置向量：

$$
x_i = e(v_i) + p_i
$$

这里的物品向量 $e(v_i)$ 是 embedding，也就是把离散 ID 映射成连续向量的查表结果。绝对位置编码 $p_i$ 可以是固定的 sin/cos 编码，也可以是可学习参数。固定编码是预先定义好的函数；可学习编码是让模型在训练中自己学每个位置的向量。

但推荐行为的时间间隔不均匀，只加 $p_i$ 不够。更完整的输入可以加入层次时间编码：

$$
x_i = e(v_i) + p_i + h_i
$$

其中：

$$
h_i = E_h[\text{hour}(t_i)] + E_d[\text{day}(t_i)] + E_w[\text{week}(t_i)]
$$

$E_h$、$E_d$、$E_w$ 都是 embedding 表。`hour(t_i)` 表示小时，例如 0 到 23；`day(t_i)` 表示星期几；`week(t_i)` 可以表示一年中的第几周，或者更粗粒度的周期桶。

输入编码解决的是“每个位置自己带什么信息”。注意力偏置解决的是“两个位置之间有什么关系”。在 self-attention 中，位置 $i$ 对位置 $j$ 的基础打分通常来自 query 和 key 的点积：

$$
q_i^\top k_j
$$

query 是当前位置发出的查询向量，key 是历史位置被匹配的键向量。点积越大，表示两个位置越相关。为了让时间差参与打分，可以加入相对时间项：

$$
a_{ij} =
\frac{
q_i^\top k_j
+
q_i^\top r_{\phi(\Delta t_{ij})}
+
u^\top k_j
+
v^\top r_{\phi(\Delta t_{ij})}
}{\sqrt{d}}
$$

然后用 softmax 得到注意力权重：

$$
\alpha_{ij}=\text{softmax}_j(a_{ij})
$$

这里的 $\phi(\Delta t_{ij})$ 是时间差分桶函数。分桶就是把连续时间差映射成有限类别，例如 0 到 1 分钟、1 到 5 分钟、5 到 30 分钟、30 分钟到 2 小时、2 小时到 1 天。$r_{\phi(\Delta t_{ij})}$ 是这个时间差桶对应的向量。

为什么要分桶？因为原始秒级时间差跨度太大。30 秒、300 秒、30000 秒、3000000 秒直接作为数值输入，会让模型面对严重长尾。更稳定的方式是先做 clip 截断，再做 `log1p` 压缩或离散分桶。

玩具数值例子：用户行为为 `A@08:00 -> B@08:05 -> C@21:00`，现在预测 `C` 后的下一步。假设 `A` 和 `B` 与 `C` 的内容相似度都是 `0.3`，但时间偏置不同：

| 历史行为 | 与 C 的内容分 | 时间差 | 时间偏置 | 总分 |
|---|---:|---:|---:|---:|
| A@08:00 | 0.3 | 13h | 0.1 | 0.4 |
| B@08:05 | 0.3 | 12h55m | 0.4 | 0.7 |

如果时间桶把 `B` 和 `C` 归入更强相关的时间模式，模型就会更关注 `B`。这个例子不代表所有业务规则，只说明相对时间偏置可以改变注意力分配。

真实工程中，短视频、电商、新闻推荐都有明显的非均匀行为。用户可能在 10 分钟内连续点击相似内容，也可能隔几天才回访同一类商品。绝对位置能表达“最近”，相对时间能表达“刚刚”，层次时间能表达“晚上、周末、节假日附近”。三者解决的问题不同，通常不能简单互相替代。

---

## 代码实现

下面是一个最小可运行的 Python 例子。它不训练完整模型，只演示三件事：如何构造绝对位置、如何把时间差分桶、如何把 item embedding、position embedding 和 hour/day/week embedding 加起来。

```python
import math
import numpy as np
from datetime import datetime, timezone, timedelta

def bucketize_delta(seconds):
    """把时间差秒数映射到离散桶，避免直接使用长尾秒级数值。"""
    boundaries = [60, 5 * 60, 30 * 60, 2 * 3600, 12 * 3600, 24 * 3600, 7 * 24 * 3600]
    for idx, boundary in enumerate(boundaries):
        if seconds <= boundary:
            return idx
    return len(boundaries)

def make_embedding_table(size, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.02, size=(size, dim))

def encode_sequence(item_ids, timestamps, dim=8, max_len=16):
    item_table = make_embedding_table(100, dim, seed=1)
    pos_table = make_embedding_table(max_len, dim, seed=2)
    hour_table = make_embedding_table(24, dim, seed=3)
    day_table = make_embedding_table(7, dim, seed=4)

    xs = []
    for i, (item_id, ts) in enumerate(zip(item_ids, timestamps)):
        item_emb = item_table[item_id]
        pos_emb = pos_table[i]
        hour_emb = hour_table[ts.hour]
        day_emb = day_table[ts.weekday()]
        x = item_emb + pos_emb + hour_emb + day_emb
        xs.append(x)

    return np.stack(xs)

def attention_time_bias(timestamps):
    n = len(timestamps)
    bias = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            delta = abs((timestamps[i] - timestamps[j]).total_seconds())
            bias[i, j] = bucketize_delta(delta)
    return bias

tz = timezone(timedelta(hours=8))
timestamps = [
    datetime(2026, 4, 19, 8, 0, tzinfo=tz),
    datetime(2026, 4, 19, 8, 5, tzinfo=tz),
    datetime(2026, 4, 19, 21, 0, tzinfo=tz),
]
item_ids = [10, 20, 30]

x = encode_sequence(item_ids, timestamps)
bias = attention_time_bias(timestamps)

assert x.shape == (3, 8)
assert bias.shape == (3, 3)
assert bucketize_delta(30) == 0
assert bucketize_delta(3 * 24 * 3600) == 6
assert bias[0, 1] == bucketize_delta(5 * 60)
assert bias[0, 2] == bucketize_delta(13 * 3600)

print("encoded shape:", x.shape)
print("time bucket matrix:")
print(bias)
```

在真实模型中，`x` 会继续输入 Transformer encoder。Transformer encoder 是多层 self-attention 和前馈网络的组合，用来从历史行为中提取序列表示。`bias` 可以作为 attention bias 加到注意力 logits 上，或者查表得到相对时间向量后参与 $q_i^\top r_{\phi(\Delta t)}$ 计算。

工程实现时，关键不在代码有多复杂，而在离线训练和线上服务必须完全一致。`bucketize_delta` 的边界、时区转换、截断规则、缺失时间处理，都应该放进同一份可复用逻辑里。否则训练时第 3 桶代表 30 分钟内，线上第 3 桶代表 2 小时内，模型查到的 embedding 就会错位。

---

## 工程权衡与常见坑

位置编码设计的核心权衡是：信息越多，模型表达力越强；但特征越多，也越容易引入噪声、不一致和泄漏。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 只用绝对位置 | 5 分钟间隔和 5 天间隔被看得太像 | 加入相对时间差或会话边界 |
| 直接输入秒级时间戳 | 长尾严重，泛化差 | 使用 `clip`、`log1p`、bucket 分桶 |
| 时区不统一 | “晚上”被映射到错误小时 | 明确使用用户本地时区或统一业务时区 |
| 训练和线上分桶不一致 | embedding 语义错位 | 离线在线共用同一套分桶函数 |
| 层次特征过多 | hour/day/week 重复表达，过拟合 | 做消融实验，保留稳定增益特征 |
| 目标泄漏 | 模型提前看到未来信息 | 样本构造只保留预测点之前的行为 |
| padding 位置未 mask | 空位置参与注意力 | attention mask 和 position mask 同步处理 |

新手例子：如果一个用户在北京时间 21:00 点击，另一个用户在美国本地时间 21:00 点击，二者都被粗暴映射成同一个“晚上 21 点”，可能是合理的，也可能是错误的。合理与否取决于业务是否关注用户本地时间。如果线上只存 UTC，却在训练时使用本地时区，就会出现同一类行为被映射到不同 hour bucket 的问题。

真实工程例子：电商推荐中，用户可能在工作日午休浏览，在周末晚上下单。若模型只用绝对位置，它可能知道“最近看过某品牌耳机”；若加入 hour/day/week，模型可能进一步学到“周末晚上更容易购买”。但如果业务覆盖多个国家，时区处理错误会让这个信号变成噪声。

相对时间分桶也不能无限细。过细的桶会导致每个桶样本少，embedding 学不稳；过粗的桶会丢失行为节奏。通常可以从对数桶开始，例如：

$$
\phi(\Delta t)=\min(\lfloor \log(1+\Delta t) \rfloor, B)
$$

其中 $B$ 是最大桶编号。这个公式的含义是：短时间差保留更细粒度，长时间差逐渐压缩到粗粒度。推荐行为通常对“刚刚发生”的变化更敏感，所以这种设计比均匀切秒数更稳。

---

## 替代方案与适用边界

不同位置编码方案适合不同业务，不存在一个永远最优的设计。判断标准不是“复杂度越高越好”，而是时间信号是否稳定、是否能在线一致计算、是否带来可验证收益。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 只用绝对位置编码 | 简单、稳定、实现成本低 | 不知道行为间隔 | 短序列、时间不敏感任务 |
| 可学习位置编码 | 能学习“最近几个位置更重要” | 超出最大长度泛化较弱 | 固定长度历史窗口 |
| 固定 sin/cos 编码 | 对更长位置有一定外推能力 | 不一定贴合推荐行为 | 序列长度变化较大 |
| 相对时间偏置 | 能区分短期 burst 和长期偏好 | 需要稳定分桶和时间戳 | 新闻、短视频、电商点击流 |
| 层次时间编码 | 能表达日/周周期 | 时区和周期定义容易出错 | 外卖、出行、本地生活、复购 |
| 混合方案 | 表达力强 | 调参和排查成本更高 | 中大型排序或重排模型 |

新闻推荐和短视频推荐通常更需要相对时间和会话边界。用户刚刚连续点击某类内容，往往比几天前的一次点击更能代表当前兴趣。电商推荐则经常同时需要短期兴趣和长期偏好：刚刚浏览的商品影响强，但品牌偏好、价格带、复购周期也重要。

某些任务并不需要复杂时间建模。例如一个行为间隔稳定、序列很短、候选集变化慢的业务，只用绝对位置编码可能已经足够。过强的时间编码反而会让模型记住偶然模式，例如某个活动日的异常流量，而不是长期有效的偏好结构。

推荐的工程路线是：先实现绝对位置编码，建立稳定基线；再加入相对时间分桶，看短期指标和长周期指标是否都改善；最后加入 hour/day/week 等层次时间特征，并通过消融实验确认每一类特征是否真的有贡献。消融实验是指每次移除一个模块，观察指标变化，用来判断该模块是否必要。

---

## 参考资料

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
提出 Transformer 和位置编码，是理解固定 sin/cos 位置编码与 self-attention 的基础。

2. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)  
提出 SASRec，证明 self-attention 可以作为序列推荐的核心结构。

3. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)  
将双向 Transformer 用于序列推荐，可用于理解推荐场景中的序列建模范式。

4. [Time Interval Aware Self-Attention for Sequential Recommendation](https://doi.org/10.1145/3336191.3371786)  
提出 TiSASRec，把绝对位置和时间间隔一起建模，是相对时间编码的重要参考。

5. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)  
提供相对位置编码的重要机制来源，可帮助理解相对位置项如何进入注意力计算。

6. [TLSTSRec: Time-aware Long- and Short-Term Sequential Recommendation](https://doi.org/10.3233/IDA-240051)  
使用时间 duration 和 spectrum 构造时间相关表示，可作为层次化时间设计的扩展参考。

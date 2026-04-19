## 核心结论

SASRec，全称 Self-Attentive Sequential Recommendation，是一个用于序列推荐的 next-item 模型。next-item 模型的意思是：给定用户已经发生的一串行为，预测下一次最可能发生的 item。

它的核心做法是把 Transformer 的因果自注意力用于用户历史行为建模。自注意力是指模型会计算“当前位置应该关注历史序列中哪些位置”；因果是指当前位置只能看自己和过去，不能看未来答案。

SASRec 的价值主要有两点：

| 对比项 | RNN / GRU 序列推荐 | SASRec |
|---|---|---|
| 序列处理方式 | 按时间步逐个递推 | 一次处理整个序列 |
| 训练并行性 | 弱，后一步依赖前一步隐藏状态 | 强，不同位置可并行计算 |
| 长距离依赖 | 容易被较近行为覆盖 | 可直接关注远处关键行为 |
| 核心结构 | 循环状态转移 | 因果自注意力 |
| 常见问题 | 长序列训练慢、远距离信息衰减 | mask、padding、位置编码容易写错 |

新手版例子：用户最近看了“连衣裙 -> 外套 -> 靴子”，SASRec 不会把这三步当成固定链条顺着算，而是会判断“外套”和“靴子”对当前预测谁更重要，再预测下一件更可能是“围巾”还是“其他服饰”。

核心打分公式是：

$$
\text{下一项打分} = \text{最后位置表示} \cdot \text{候选 item embedding}
$$

写成推荐系统里的常用记号：

$$
r_{t,j}=h_t^\top M[j]
$$

其中 \(h_t\) 是用户历史序列最后一个位置的表示，\(M[j]\) 是候选 item \(j\) 的向量，\(r_{t,j}\) 是候选 item 的排序分数。

---

## 问题定义与边界

SASRec 解决的问题是：给定一个按时间排序的用户行为序列，预测下一步最可能发生的 item。

设用户历史行为序列为：

$$
s_{1:t}=(s_1,s_2,\dots,s_t)
$$

目标是预测：

$$
s_{t+1}
$$

这里的 \(s_i\) 是第 \(i\) 次交互的 item id，例如商品 id、视频 id、文章 id、游戏道具 id。item 是推荐系统里的候选对象，白话说就是“要被推荐的东西”。

新手版例子：电商里，用户最近点击了手机壳、充电器、耳机，模型要预测下一次更可能点什么商品。它学的是“下一次点击”，不是“这个用户一生会喜欢什么”。

SASRec 的输入输出边界如下：

| 边界项 | 内容 |
|---|---|
| 输入 | 按时间排序的历史 item 序列 |
| 输出 | 下一 item 的排序分数 |
| 学习目标 | 根据历史行为预测下一步行为 |
| 不包含 | 用户显式特征 |
| 不包含 | 上下文多模态信息，如图片、文本、地理位置 |
| 不包含 | 图结构关系，如好友关系、商品知识图谱 |
| 不直接负责 | 召回后的复杂重排序 |

数据上有两个硬约束。

第一，行为必须按时间排序。序列推荐依赖“先发生什么、后发生什么”，如果顺序错了，模型学到的转移规律就是错的。

第二，必须截断最近窗口。生产环境里用户历史可能很长，但 SASRec 的注意力复杂度通常随序列长度平方增长，即：

$$
O(L^2)
$$

其中 \(L\) 是序列长度。工程里常保留最近 \(N\) 条行为，例如最近 50、100 或 200 条。这个窗口不是越长越好，而是要在效果、延迟、显存和吞吐之间取平衡。

---

## 核心机制与推导

SASRec 的核心由三部分组成：

| 组件 | 白话解释 | 作用 |
|---|---|---|
| item embedding | 把 item id 转成向量 | 表达 item 的语义和协同关系 |
| position embedding | 给每个位置一个向量 | 让模型知道顺序 |
| causal self-attention | 只能看过去的自注意力 | 建模历史行为对当前位置的影响 |

embedding 是“向量表示”的意思。模型不能直接理解 item id 这种离散编号，所以要把每个 item id 映射成一个连续向量。

对序列中第 \(i\) 个 item，输入表示为：

$$
e_i=M[s_i]+P_i
$$

其中 \(M[s_i]\) 是 item embedding，\(P_i\) 是第 \(i\) 个位置的位置 embedding。

接着做自注意力。每个位置的输入向量 \(e_i\) 会被映射成 query、key、value：

$$
q_i=e_iW_Q,\quad k_i=e_iW_K,\quad v_i=e_iW_V
$$

query 可以理解为“当前位置想找什么信息”，key 是“历史位置提供什么索引”，value 是“历史位置真正被汇总的内容”。

注意力权重为：

$$
a_{ij}=\mathrm{softmax}_j\left(\frac{q_i k_j^\top}{\sqrt d}+m_{ij}\right)
$$

其中 causal mask 为：

$$
m_{ij}=
\begin{cases}
0,& j\le i\\
-\infty,& j>i
\end{cases}
$$

这个 mask 保证第 \(i\) 个位置只能看 \(1\) 到 \(i\) 的内容，不能看未来位置。最后当前位置表示为：

$$
h_i=\sum_{j\le i}a_{ij}v_j
$$

预测下一项时，用最后位置表示 \(h_t\) 给候选 item 打分：

$$
r_{t,j}=h_t^\top M[j]
$$

从输入到打分的流程可以写成：

```text
用户历史 item 序列
        |
        v
item id: [s1, s2, ..., st]
        |
        v
item embedding + position embedding
        |
        v
causal self-attention block
        |
        v
最后位置表示 h_t
        |
        v
h_t 与候选 item embedding 点积
        |
        v
候选 item 排序分数
```

新手版例子：序列 `[A, B, C]` 中，预测位置 3 时，只能看 `A、B、C` 自己和之前的项，不能偷看 `D`。如果 `B` 更能解释 `C` 的出现，模型就会给 `B` 更高权重。

一个 3 步手算玩具例子如下。假设对位置 3，因果 mask 后只能看位置 1、2、3。注意力原始打分为：

$$
[1,2,3]
$$

softmax 后近似为：

$$
[0.09,0.24,0.67]
$$

假设对应 value 是：

$$
[10,20,30]
$$

那么输出为：

$$
h_3 \approx 0.09\cdot10+0.24\cdot20+0.67\cdot30=25.8
$$

含义是：模型认为第三个位置最应该关注第三项，其次关注第二项，第一项影响较小。真实模型中的 value 不是一个数字，而是一个高维向量，但汇聚逻辑相同。

真实工程例子：在电商首页的“下一次点击”预测里，可以把用户最近 \(N\) 次点击、加购、收藏、购买按时间排序后输入 SASRec。模型输出每个候选商品的分数，然后取 top-K 商品进入后续排序或展示链路。它适合服饰、内容流、游戏推荐这类短期兴趣变化明显、但历史偏好仍然有用的场景。

---

## 代码实现

代码实现的重点不是把完整 Transformer 全搬过来，而是保证四件事正确：序列、mask、padding、位置编码。

训练时通常把序列向右错位：输入前 \(t\) 个 item，预测第 \(t+1\) 个 item。推理时用用户最近行为序列得到最后位置表示，再对候选集合打分排序。

新手版伪代码：

```python
items = item_embedding(seq) + position_embedding(seq)
h = transformer_block(items, causal_mask=True, pad_mask=True)
score = h_last @ item_embedding_table.T
```

这段代码的意思是：先把历史序列变成向量，再用带因果遮罩的注意力层编码，最后拿最后一个位置去给所有候选 item 打分。

数据流如下：

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 原始数据 | 用户行为日志 | 按用户分组、按时间排序 | item 序列 |
| 序列处理 | item id 序列 | 截断、padding | 固定长度序列 |
| 表示层 | 固定长度 item id | item embedding + position embedding | 向量序列 |
| 编码层 | 向量序列 | causal mask + padding mask + Transformer block | 每个位置的表示 |
| 打分层 | 最后位置表示 | 与候选 item embedding 点积 | 候选 item 分数 |

下面是一个可运行的最小 Python 例子，只演示 causal self-attention 的关键计算，不依赖 PyTorch：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]

def causal_attention_for_last(scores, values):
    # scores 和 values 只包含当前位置允许看到的历史项
    weights = softmax(scores)
    output = sum(w * v for w, v in zip(weights, values))
    return weights, output

scores = [1.0, 2.0, 3.0]
values = [10.0, 20.0, 30.0]

weights, h3 = causal_attention_for_last(scores, values)

assert len(weights) == 3
assert abs(sum(weights) - 1.0) < 1e-9
assert weights[2] > weights[1] > weights[0]
assert abs(h3 - 25.752103508360806) < 1e-9

candidate_embeddings = {
    "scarf": 0.9,
    "phone": 0.1,
}

scores_to_items = {
    item: h3 * emb
    for item, emb in candidate_embeddings.items()
}

assert scores_to_items["scarf"] > scores_to_items["phone"]
```

这个例子里，`scores` 对应注意力打分，`values` 对应历史项的信息，`h3` 对应最后位置表示。候选 item embedding 被简化成一个数字，所以最后用乘法模拟点积。

真实实现中还要处理以下关键点：

| 实现点 | 必要性 |
|---|---|
| 序列截断 | 控制计算成本，优先保留最近行为 |
| causal mask | 防止训练时看到未来答案 |
| padding mask | 防止 PAD 位置污染注意力 |
| position embedding | 保留顺序信息 |
| 负样本采样或全量 softmax | 决定训练成本和目标形式 |

负样本采样是指训练时不对所有 item 计算分数，而是采一部分“用户没有交互过的 item”作为负例。全量 softmax 是指对完整 item 集合计算概率，目标更直接，但当 item 很多时成本更高。

---

## 工程权衡与常见坑

SASRec 的效果很依赖数据顺序、mask 正确性和评估口径。工程上最容易出错的地方往往不在模型结构，而在数据管道和训练评估细节。

新手版例子：如果把用户一天内的行为顺序弄反，模型学到的就是“先买后看”的错误规律；如果忘了 mask，模型会在训练时偷看未来答案，离线指标会虚高，上线就掉。

常见坑如下：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 时间顺序错了 | 学到反向转移规律 | 按用户分组后严格按时间戳升序排序 |
| 未来信息泄漏 | 离线指标虚高，上线效果变差 | 训练和评估都使用 causal mask |
| 忘记 position embedding | 模型难以区分顺序 | item embedding 必须叠加位置 embedding |
| PAD 没有 mask | 空位置参与注意力计算 | 单独构造 padding mask |
| 负采样评估和全量评估混用 | 指标不可比较 | 明确记录评估候选集口径 |
| 序列过长不截断 | 显存、延迟、吞吐恶化 | 设置最大长度并监控效果变化 |
| 训练推理序列处理不一致 | 线上分布偏移 | 复用同一套截断和 padding 逻辑 |
| item id 映射不稳定 | embedding 对错 item | 固定词表版本，保留未知 item 策略 |

训练和推理一致性检查清单：

| 检查项 | 应确认的问题 |
|---|---|
| 时间排序 | 训练、验证、线上是否都按事件时间升序 |
| 序列窗口 | 最大长度是否一致 |
| padding 方向 | 左 padding 还是右 padding 是否一致 |
| position id | PAD 位置是否不会产生有效位置语义 |
| mask | causal mask 和 padding mask 是否同时生效 |
| 候选集 | 离线评估候选和线上召回候选是否口径一致 |
| item 词表 | 训练词表、线上词表、embedding 表是否同版本 |
| 特征时间 | 是否只使用预测时刻之前可获得的信息 |

一个常见工程取舍是序列长度。更长的历史窗口可能包含更多长期偏好，但计算更重，也可能引入噪声。很多业务中，最近 50 到 200 个行为已经覆盖主要兴趣变化。是否继续加长，要用离线指标、线上延迟和 A/B 实验共同判断。

另一个取舍是训练目标。负采样训练更便宜，适合 item 数量很大的系统；全量 softmax 更接近完整分类目标，但成本更高。评估时也要小心，采样负例上的 NDCG、HitRate 和全量排序上的指标不能直接横向对比。

---

## 替代方案与适用边界

SASRec 适合“短期兴趣会变化，但长期偏好仍重要”的场景。如果目标更偏静态画像、强上下文、多模态理解或复杂关系建模，单靠 SASRec 不一定最优。

新手版例子：如果用户行为序列很短，或者 item 间关系主要来自社交关系、知识图谱、共同购买图，而不是时间顺序，SASRec 可能不如图推荐模型；如果你只想要极快召回，双塔往往更简单、更便宜。

不同方案对比如下：

| 方法 | 是否能并行训练 | 是否擅长长依赖 | 是否依赖时间顺序 | 线上延迟 | 适用场景 |
|---|---|---|---|---|---|
| SASRec | 是 | 较强 | 强依赖 | 中等 | 下一点击、下一购买、内容流短期兴趣 |
| RNN / GRU4Rec | 较弱 | 中等 | 强依赖 | 中等 | 中短序列、工程复杂度较低的序列推荐 |
| CNN 序列模型 | 是 | 中等，受卷积窗口影响 | 依赖 | 较低 | 局部模式明显的序列 |
| 图推荐 | 通常可批量训练 | 取决于图结构 | 不一定 | 中高 | 社交关系、商品关系、知识图谱明显 |
| 双塔召回 | 是 | 弱到中等 | 不强制 | 低 | 大规模候选召回、低延迟检索 |

何时使用 SASRec，可以按下表判断：

| 条件 | 是否适合 SASRec |
|---|---|
| 用户有较丰富的历史行为序列 | 适合 |
| 下一步行为强依赖最近点击、浏览、购买 | 适合 |
| 业务需要建模长距离行为影响 | 适合 |
| 只有少量历史行为，很多用户是冷启动 | 不一定适合 |
| 主要信号来自图片、文本、地理位置 | 单独使用不够 |
| 主要目标是百万级候选快速召回 | 双塔可能更合适 |
| item 关系来自图结构，时间顺序较弱 | 图推荐可能更合适 |
| 线上延迟预算极低 | 需要压缩、蒸馏或改用更轻模型 |

SASRec 不是推荐系统的完整替代品。它更像一个序列兴趣编码器，可以用于召回、粗排或排序特征。实际生产系统常把它和双塔召回、热门召回、协同过滤、图召回、重排序模型组合使用。

---

## 参考资料

如果准备实现一个序列推荐系统，建议先读 Transformer 原论文理解 self-attention，再读 SASRec 论文理解为什么需要 causal mask，然后看官方代码确认位置编码、mask、训练目标是怎么落地的，最后查 PyTorch 文档避免 mask 维度写错。

建议阅读顺序：

1. Transformer 原论文
2. SASRec 论文
3. 官方代码
4. PyTorch mask 文档

参考链接：

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)
3. [kang205/SASRec](https://github.com/kang205/SASRec)
4. [PyTorch Transformer 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
5. [GRU4Rec: Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)

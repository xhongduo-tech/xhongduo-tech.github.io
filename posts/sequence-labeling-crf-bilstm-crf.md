## 核心结论

序列标注不是“每个位置各自分类”这么简单。它的核心难点是：当前位置的标签，往往依赖前一个甚至后一段位置的标签。例如命名实体识别里，`I-ORG` 表示“组织名内部”，白话说就是“这个词还在前面那个组织实体里面”；它通常不能凭空出现。

线性链 CRF（Conditional Random Field，条件随机场，白话说就是“对整条标签序列统一打分的模型”）解决的正是这个问题。它不直接逐位置输出最终标签，而是给整条标签路径一个总分：

$$
score(x,y)=\sum_{t=1}^{T}\left(A_{y_{t-1},y_t}+s_t(y_t)\right)
$$

其中，$s_t(y_t)$ 是第 $t$ 个位置取标签 $y_t$ 的发射分数，白话说就是“当前位置像不像这个标签”；$A_{y_{t-1},y_t}$ 是转移分数，白话说就是“前后两个标签连在一起合不合理”。

因此，CRF 学到的不是单点最优，而是全局最优：

$$
\log P(y|x)=score(x,y)-\log\sum_{y'}\exp(score(x,y'))
$$

BiLSTM-CRF 则是在 CRF 前面加一层 BiLSTM（双向长短期记忆网络，白话说就是“同时看左边和右边上下文的序列编码器”）。BiLSTM 负责把上下文压进每个位置的表示里，CRF 负责让最终标签序列满足结构约束。前者擅长“看懂词义和上下文”，后者擅长“保证输出连贯合法”。

玩具例子：如果一句话中某个位置单看词面像 `I-ORG`，独立 softmax 可能直接输出 `I-ORG`；但 CRF 会发现前一个标签是 `O`，而 `O -> I-ORG` 很不合理，于是整条路径总分下降，最后改成 `B-ORG` 或 `O`。

---

## 问题定义与边界

序列标注的输入是一串 token，可以是字、词或子词；输出是同长度标签序列。常见任务包括分词、词性标注、命名实体识别（NER）。

以 BIO 标注为例：

- `B-ORG`：实体开头
- `I-ORG`：实体内部
- `O`：不属于任何实体

例如“华为在深圳”可以标为：

| 位置 | 字词 | 标签 |
|---|---|---|
| 1 | 华为 | B-ORG |
| 2 | 在 | O |
| 3 | 深圳 | B-LOC |

如果按更细粒度字符级标注，也可能写成：

| 位置 | 字 | 标签 |
|---|---|---|
| 1 | 华 | B-ORG |
| 2 | 为 | I-ORG |
| 3 | 在 | O |
| 4 | 深 | B-LOC |
| 5 | 圳 | I-LOC |

这里的边界很明确：我们讨论的是**标签之间存在强依赖**的任务。也就是，后一个标签是否合法，不能只看当前位置输入，还要看前一个标签状态。

下面这个合法性表格就是 CRF 存在的直接原因：

| 前一标签 -> 当前标签 | 是否常见/合法 | 原因 |
|---|---|---|
| `O -> B-LOC` | 合法 | 新实体可以从非实体位置开始 |
| `B-ORG -> I-ORG` | 合法 | 组织实体继续延伸 |
| `I-ORG -> I-ORG` | 合法 | 仍在同一组织实体内部 |
| `O -> I-ORG` | 通常不合法 | 实体内部标签不能凭空出现 |
| `B-LOC -> I-ORG` | 通常不合法 | 实体类型突然切换 |
| `I-PER -> I-ORG` | 通常不合法 | 同一连续片段内部类型冲突 |

因此，问题不是“这个词像什么标签”，而是“整条标签链条是否同时局部合理、全局一致”。

---

## 核心机制与推导

先看独立 token 分类。它在每个位置都做一次 softmax：

$$
P(y_t|x)=softmax(s_t)
$$

这相当于假设各位置条件独立。问题在于，真实任务里这个假设通常不成立。

CRF 的做法是直接建模整条路径条件概率。对于输入序列 $x$ 和标签序列 $y$：

$$
score(x,y)=\sum_{t=1}^{T}(A_{y_{t-1},y_t}+s_t(y_t))
$$

$$
P(y|x)=\frac{\exp(score(x,y))}{\sum_{y'}\exp(score(x,y'))}
$$

分子是目标路径得分，分母是所有可能路径得分的指数和，也叫分区函数。白话说，模型不是问“当前位置像不像这个标签”，而是问“这整条路径在所有候选路径里占多大比例”。

训练时最大化对数似然：

$$
\log P(y|x)=score(x,y)-\log Z(x)
$$

其中

$$
Z(x)=\sum_{y'}\exp(score(x,y'))
$$

### 玩具例子

设序列长度为 3，标签集合为 $\{B, I\}$。发射分数如下：

- $s_1(B)=2,\ s_1(I)=0$
- $s_2(B)=0,\ s_2(I)=1$
- $s_3(B)=0.5,\ s_3(I)=1.2$

转移分数：

- $A_{B,B}=0.1,\ A_{B,I}=0.5$
- $A_{I,B}=-0.2,\ A_{I,I}=0.3$

假设忽略起始符号，路径 `B-I-I` 的总分是：

$$
score(x, BII)=2 + (0.5+1) + (0.3+1.2)=5.0
$$

路径 `B-B-I` 的总分是：

$$
score(x, BBI)=2 + (0.1+0) + (0.5+1.2)=3.8
$$

所以 `B-I-I` 更优。维特比算法（Viterbi，白话说就是“只保留到当前位置为止最好的前驱路径”）不会枚举全部路径，而是用动态规划求最优解。

训练时分母 $\log Z(x)$ 不能暴力枚举，一般用前向算法（forward algorithm）计算。定义前向量：

$$
\alpha_t(j)=\log\sum_{y_{1:t-1}}\exp(score(x,y_{1:t-1}, y_t=j))
$$

递推式：

$$
\alpha_t(j)=s_t(j)+\log\sum_i \exp(\alpha_{t-1}(i)+A_{i,j})
$$

这一步里的 `log-sum-exp` 很关键，因为直接算指数和容易溢出。

梯度本质上是“经验计数减模型期望”。白话说，正确路径里出现过的转移和标签应该被拉高，而模型在所有路径上平均偏爱的那些错误转移和标签应该被压低。前向-后向算法就是高效计算这种期望。

真实工程例子：医疗 NER 中，“2 型糖尿病”如果中间某个词单独看容易被判成 `O`，独立分类会导致实体断裂；BiLSTM 先看到完整上下文，CRF 再约束 `B-DISEASE -> I-DISEASE -> I-DISEASE` 这种连续路径，边界通常更稳。

---

## 代码实现

下面给一个可运行的最小 CRF 解码与打分实现。它不包含 BiLSTM 本体，但保留了 CRF 最关键的三部分：路径打分、前向算法、维特比解码。实际工程中，`emissions` 通常来自 BiLSTM 或 Transformer。

```python
import math

def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))

def path_score(emissions, transitions, path):
    score = emissions[0][path[0]]
    for t in range(1, len(path)):
        score += transitions[path[t - 1]][path[t]] + emissions[t][path[t]]
    return score

def forward_logZ(emissions, transitions):
    dp = emissions[0][:]
    for t in range(1, len(emissions)):
        new_dp = []
        for curr in range(len(emissions[t])):
            vals = [dp[prev] + transitions[prev][curr] for prev in range(len(dp))]
            new_dp.append(emissions[t][curr] + logsumexp(vals))
        dp = new_dp
    return logsumexp(dp)

def viterbi_decode(emissions, transitions):
    dp = emissions[0][:]
    backpointers = []

    for t in range(1, len(emissions)):
        new_dp = []
        bp_t = []
        for curr in range(len(emissions[t])):
            candidates = [dp[prev] + transitions[prev][curr] for prev in range(len(dp))]
            best_prev = max(range(len(candidates)), key=lambda i: candidates[i])
            new_dp.append(emissions[t][curr] + candidates[best_prev])
            bp_t.append(best_prev)
        dp = new_dp
        backpointers.append(bp_t)

    last = max(range(len(dp)), key=lambda i: dp[i])
    best_path = [last]
    for bp_t in reversed(backpointers):
        last = bp_t[last]
        best_path.append(last)
    best_path.reverse()
    return best_path

# 0 -> B, 1 -> I
emissions = [
    [2.0, 0.0],
    [0.0, 1.0],
    [0.5, 1.2],
]
transitions = [
    [0.1, 0.5],
    [-0.2, 0.3],
]

best_path = viterbi_decode(emissions, transitions)
best_score = path_score(emissions, transitions, best_path)
logZ = forward_logZ(emissions, transitions)

assert best_path == [0, 1, 1]   # B-I-I
assert abs(best_score - 5.0) < 1e-8
assert logZ > best_score         # 分区函数的 log 一定不小于最优路径分数
print(best_path, best_score, round(logZ, 4))
```

如果把它扩展成 BiLSTM-CRF，整体流程就是：

1. 输入 token 序列。
2. BiLSTM 输出每个位置的发射分数 $s_t(y)$。
3. CRF 用转移矩阵 $A$ 计算训练对数似然。
4. 推理时用维特比回溯最优路径。

伪代码可以写成：

```python
emissions = bilstm(inputs)              # [seq_len, num_tags]
loss = crf.neg_log_likelihood(emissions, gold_tags)
best_tags = crf.viterbi_decode(emissions)
```

---

## 工程权衡与常见坑

BiLSTM-CRF 的优点很明确，但代价也真实存在：训练和解码都比独立 softmax 更复杂。

| 问题 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 把序列标注当独立分类 | 出现 `O -> I-ORG` 这类非法跳变 | 没有建模标签依赖 | 使用 CRF 或显式约束解码 |
| 前向算法数值溢出 | loss 为 `inf` 或 `nan` | 直接算指数和 | 使用 `log-sum-exp` |
| 忘记 mask 补齐位 | padding 位置污染路径分数 | 不同样本长度不同 | 对真实长度之外位置做 mask |
| 标签集合定义不一致 | 训练正常但预测混乱 | `B/I/O` 编码映射错位 | 固定 `tag2id` 并持久化 |
| 只看 token 局部证据 | 边界断裂、实体类型漂移 | 上下文建模不足 | 用 BiLSTM/Transformer 生成 emission |

一个常见误区是认为“CRF 只是后处理”。这不准确。后处理只是把不合法标签改一改，而 CRF 是在训练目标里直接加入全局归一化，模型参数会主动学习哪些转移该奖励、哪些该惩罚，两者不是一个层次。

另一个坑是把 CRF 神化。CRF 只能建模**局部标签依赖**，标准线性链 CRF 关注的是相邻标签转移，不会自动解决长距离语义问题。长距离依赖主要还是靠 BiLSTM 或 Transformer 的表示能力。

---

## 替代方案与适用边界

如果任务里标签依赖很弱，CRF 不一定值得上。选择模型时要看收益是否覆盖复杂度。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 独立 softmax 分类 | 简单、快、容易部署 | 标签不连贯，无法显式约束转移 | 快速原型、标签近似独立 |
| BiLSTM-CRF | 上下文强，标签序列合法性好 | 训练和解码更复杂 | NER、分词、POS 等经典序列标注 |
| Transformer + CRF | 语义表示更强 | 参数更大、成本更高 | 数据较多、需要更强上下文建模 |
| Span-based | 直接预测实体区间 | 实现更复杂，不适合所有标注任务 | 嵌套实体、区间级抽取 |

新手最容易判断的一条边界是：如果输出标签之间有明确规则，例如 BIO/BMES 体系，那 CRF 往往有价值；如果每个位置标签几乎完全独立，例如某些简单 token 属性分类，CRF 增益可能有限。

真实工程里也常见替代：医疗 NER、法务实体抽取等高一致性任务，常用 `Encoder + CRF`；但在大模型特征已经很强、且推理速度敏感的场景，有时会退回独立分类或直接做 span 抽取，因为 CRF 的收益不足以覆盖复杂度。

---

## 参考资料

| 来源 | 内容简介 | 用途 |
|---|---|---|
| Springer: https://link.springer.com/article/10.1186/s12911-019-0865-1 | 解释 CRF 条件概率、分区函数、BiLSTM-CRF 在医学文本中的使用 | 理解公式与任务背景 |
| MindSpore 教程: https://www.mindspore.cn/tutorials/en/r2.6.0rc1/nlp/sequence_labeling.html | 用较直观方式说明序列标注为何不能做成独立决策 | 帮助初学者建立问题直觉 |
| ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0167865512001857 | 讨论 CRF 训练、梯度与数值稳定实现 | 理解前向-后向与工程实现 |
| Emergent Mind: https://www.emergentmind.com/topics/bilstm-crf-model | 总结 BiLSTM-CRF 架构、维特比解码和领域应用 | 连接理论与真实工程案例 |

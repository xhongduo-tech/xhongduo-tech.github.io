## 核心结论

RRHF，全称 Rank Responses to Align Human Feedback，是一种把“人类偏好排序”直接写进语言模型微调目标的对齐方法：给同一个 `prompt` 的多个候选回复排序，然后训练模型让高偏好回复的条件概率高于低偏好回复，同时保留监督微调目标。

它的核心不是“拟合奖励值”，而是“学习候选回复之间的相对顺序”。如果同一个问题有 4 个回答，人工或奖励模型排出第 1 到第 4 名，RRHF 不要求模型学会“1 分、2 分、3 分、4 分”这些绝对分数，而是要求模型学会：第 1 名的概率应该高于第 2、3、4 名，第 2 名的概率应该高于第 3、4 名。

总目标可以写成：

$$
L = L_{rank} + L_{sft}
$$

其中 $L_{rank}$ 负责学习偏好排序，$L_{sft}$ 负责保留最高质量回复上的监督微调信号。

RRHF 在几类对齐方法里的位置如下：

| 方法 | 学什么 | 数据形态 | 训练复杂度 | 典型用途 |
|---|---|---|---|---|
| SFT | 单一标准答案 | `prompt -> answer` | 低 | 指令微调、格式学习 |
| RRHF | 多个回复的相对排序 | `prompt -> 多候选 + 排序` | 中低 | 离线偏好对齐 |
| DPO | 成对偏好 | `prompt -> chosen/rejected` | 中 | 直接偏好优化 |
| PPO/RLHF | 奖励最大化策略 | reward model + rollout | 高 | 在线强化学习对齐 |

一个玩具例子是：用户问“如何安全地拒绝危险请求”，模型生成 4 个回答。第 1 个既拒绝危险操作又给出安全替代建议，第 2 个只拒绝，第 3 个含糊不清，第 4 个直接给出危险步骤。RRHF 训练的目标不是预测每个回答的具体分数，而是让模型以后更倾向生成第 1 类回复，远离第 4 类回复。

---

## 问题定义与边界

RRHF 的输入不是单个标准答案，而是一个问题和多个可比较的候选回复。

用符号表示：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $x$ | prompt | 用户输入的问题或指令 |
| $y_i$ | 第 $i$ 个候选回复 | 同一个问题下的一个备选答案 |
| $r_i$ | 外部偏好分数 | 人工或奖励模型给出的质量判断，越大越好 |
| $s_\theta(x,y_i)$ | 模型对回复的打分 | 模型认为这个回复有多“像自己应该生成的答案” |
| $y^*$ | 最高分回复 | 当前候选里最应该被模仿的回复 |

这里的监督信号来自排序标注或偏好分数，而不是唯一标准答案。也就是说，RRHF 处理的是“这些回答哪个更好”的问题，不是“唯一正确答案是什么”的问题。

真实工程例子：做“安全性 + 有用性”对齐时，可以对同一批安全相关 prompt 采样 6 个候选回复。其中 2 个来自人类示范，4 个来自模型采样。然后由人工标注员或奖励模型给这 6 个回复排序。RRHF 使用这个排序训练模型：更安全、更有用、更完整的回复获得更高条件概率；危险、空泛、跑题的回复被压低。

适用边界如下：

| 场景 | 是否适合 RRHF | 原因 |
|---|---:|---|
| 同一 prompt 有多个候选回复，并且有排序 | 适合 | 排序信号完整 |
| 有奖励模型分数，可以转成排序 | 适合 | 不需要直接拟合奖励值 |
| 只有单一标准答案 | 不适合 | SFT 更直接 |
| 只有零散文本，没有可比候选 | 不适合 | 无法构造排序对 |
| 需要在线探索新策略 | 不适合 | PPO 类方法更接近目标 |
| 已有 chosen/rejected 成对数据 | 可能适合 | 但 DPO 往往更直接 |

问题边界要说清楚：RRHF 是离线训练方法。离线训练是指候选回复和偏好标注在训练前已经准备好，训练过程中不依赖模型不断生成新回复再即时打分。它适合“已有多候选偏好数据，希望稳定微调”的场景，不是完整替代 RLHF 的在线探索框架。

---

## 核心机制与推导

RRHF 首先需要把一个候选回复变成模型分数。常用定义是长度归一化的 log probability：

$$
s_\theta(x, y_i) =
\frac{1}{|y_i|}
\sum_t \log p_\theta(y_{i,t} \mid x, y_{i,<t})
$$

这里的 log probability 是“模型给某个 token 的对数概率”。长度归一化是指除以回复 token 数，避免长回复因为 token 更多而在总 logprob 上天然更吃亏。

如果偏好分数满足 $r_i > r_j$，说明 $y_i$ 比 $y_j$ 更好。RRHF 就要求：

$$
s_\theta(x,y_i) > s_\theta(x,y_j)
$$

无 margin 的排序损失可以写成：

$$
L_{rank} =
\sum_{i,j:r_i > r_j}
\max(0, s_\theta(x,y_j) - s_\theta(x,y_i))
$$

这个公式的含义很直接：如果好回复的模型分数已经高于差回复，损失为 0；如果差回复分数反而更高，就产生惩罚。

再加上最高分回复 $y^*$ 的 SFT 项：

$$
L_{sft} =
-\sum_t \log p_\theta(y^*_t \mid x, y^*_{<t})
$$

最终：

$$
L = L_{rank} + L_{sft}
$$

一个最小数值例子：

| 回复 | 偏好分数 | 当前模型分数 |
|---|---:|---:|
| `y_good` | 0.9 | -1.2 |
| `y_bad` | 0.3 | -0.8 |

因为 `y_good` 更好，但模型当前给它的分数更低，所以排序错了：

$$
L_{rank} = \max(0, -0.8 - (-1.2)) = 0.4
$$

如果最高分回复还要做 SFT，假设 $L_{sft}=1.6$，那么：

$$
L = 0.4 + 1.6 = 2.0
$$

训练方向可以理解为：

| 当前关系 | 是否符合偏好 | 梯度方向 |
|---|---:|---|
| 好回复分数 > 差回复分数 | 是 | 排序项不更新或很弱 |
| 好回复分数 < 差回复分数 | 否 | 推高好回复，压低差回复 |
| 好回复也是 $y^*$ | 是 | SFT 项继续增强它 |
| 差回复更像模型原本输出 | 否 | 排序项抑制这种偏好 |

无 margin 版本只要求“好回复高于差回复”。有 margin 版本会要求“好回复至少高出一个间隔 $m$”：

$$
L_{rank}^{margin} =
\sum_{i,j:r_i > r_j}
\max(0, m + s_\theta(x,y_j) - s_\theta(x,y_i))
$$

margin 是“安全间隔”的意思。它能让排序差距更明确，但也会引入额外超参数。RRHF 论文里的关键设计是使用无 margin 排序损失，让目标更简单，减少调参负担。

---

## 代码实现

实现 RRHF 可以分成三步：数据准备、排序对构造、损失计算。关键是按 prompt 分组处理多个候选，而不是把所有候选回复完全打散。

伪代码结构如下：

```text
读取 prompt + candidates + rewards
对同一个 prompt 内的候选按 reward 排序
计算每个候选的 length-normalized logprob
构造所有 reward_i > reward_j 的 pair
计算 pairwise ranking loss
取最高 reward 的回复计算 SFT loss
L = L_rank + L_sft
反向传播
```

关键函数可以拆成：

| 函数 | 作用 |
|---|---|
| `score_response` | 计算某个回复的长度归一化 logprob |
| `build_pairs` | 在同一个 prompt 内构造偏好排序对 |
| `compute_rank_loss` | 计算排序损失 |
| `compute_sft_loss` | 计算最高分回复的 SFT 损失 |

下面是一段可运行的 Python 玩具实现。它不调用真实大模型，只演示 RRHF 损失的数值逻辑：

```python
from itertools import combinations

def build_pairs(rewards):
    pairs = []
    for i, j in combinations(range(len(rewards)), 2):
        if rewards[i] > rewards[j]:
            pairs.append((i, j))
        elif rewards[j] > rewards[i]:
            pairs.append((j, i))
    return pairs

def compute_rank_loss(scores, rewards, margin=0.0):
    loss = 0.0
    for better, worse in build_pairs(rewards):
        loss += max(0.0, margin + scores[worse] - scores[better])
    return loss

def compute_total_loss(scores, rewards, sft_loss):
    rank_loss = compute_rank_loss(scores, rewards)
    return rank_loss + sft_loss

# 同一 prompt 下两个候选：
# y_good 的偏好更高，但当前模型分数更低，所以产生排序损失。
scores = [-1.2, -0.8]   # length-normalized logprob
rewards = [0.9, 0.3]
rank_loss = compute_rank_loss(scores, rewards)
total_loss = compute_total_loss(scores, rewards, sft_loss=1.6)

assert abs(rank_loss - 0.4) < 1e-9
assert abs(total_loss - 2.0) < 1e-9

# 如果好回复分数已经更高，排序损失为 0。
fixed_scores = [-0.5, -1.1]
assert compute_rank_loss(fixed_scores, rewards) == 0.0
```

在真实训练里，`score_response` 会从模型输出 logits 计算每个 token 的 log probability。logits 是模型对词表里每个 token 的原始打分，经过 softmax 后变成概率。训练时要只统计回复部分的 token，不要把 prompt 部分也算进回复分数，否则模型可能学到输入模板的概率差异，而不是回复质量差异。

新手可以把 RRHF 理解成一句话：先在同一道题的多个答案里比大小，再对第一名做标准答案学习。

---

## 工程权衡与常见坑

RRHF 的效果高度依赖数据质量。损失函数只是把排序信号传给模型，如果候选回复本身太差、太相似，或者排序标注噪声很大，模型学不到稳定偏好。

常见坑如下：

| 问题现象 | 原因 | 规避办法 |
|---|---|---|
| 模型偏向长回复 | 长回复更容易被误判为完整，或未做长度归一化 | 使用 length-normalized logprob，并在人评中惩罚空话 |
| 排序损失下降但真实效果差 | 候选质量低，排序只在差答案之间比较 | 混入人工示范和强模型回复 |
| 模型语言能力退化 | 只做排序，不保留 SFT | 加入 $L_{sft}$，用最高分回复做锚点 |
| 对安全问题过度拒答 | 排序标准只奖励安全，不奖励有用 | 标注维度同时覆盖安全性和有用性 |
| 训练不稳定 | 权重、margin、采样策略过度复杂 | 先用简单等权版本，再小步调参 |
| reward 分数高但人评差 | 奖励模型存在偏差 | 保留人工抽检和多指标评估 |

一个典型错误是：候选里总是出现“长但空”的回复，例如回答安全问题时写很多原则性废话，却不给任何可执行的安全替代建议。如果标注或奖励模型偏爱这种长度，RRHF 会把“长文本”误学成“好回复”。正确做法是用长度归一化分数，并在候选集中混入高质量人工示范作为锚点。

另一个工程点是优先使用离线采样。在线自采样是指训练中的模型不断生成新回复再拿去打分，这会引入更多系统复杂度，也可能让模型逐步学会迎合奖励模型的模板。RRHF 更适合先离线收集候选、固定排序数据、训练后再做人工评估和迭代。

安全性和有用性的平衡也要在数据阶段解决。比如“如何处理家用清洁剂误混”的问题，最低质量回复可能给出危险操作；中等回复可能只说“不要这样做”；高质量回复应同时拒绝危险步骤、说明风险、建议通风、远离现场、联系专业机构。RRHF 可以学习这个排序，但前提是排序标准本身写得准确。

---

## 替代方案与适用边界

RRHF 不是唯一的偏好对齐方法。选择方法时，应先看数据形态，而不是先看算法名。

| 方法 | 数据形态 | 训练方式 | 工程复杂度 | 适用场景 |
|---|---|---|---:|---|
| SFT | 单一标准答案 | 最大化标准答案概率 | 低 | 有高质量示范数据 |
| RRHF | 同一 prompt 的多候选排序 | 排序损失 + SFT | 中低 | 有多候选和排序信号 |
| DPO | chosen/rejected 成对偏好 | 直接优化偏好概率 | 中 | 成对偏好数据充足 |
| PPO | reward model + 在线 rollout | 强化学习更新策略 | 高 | 需要在线探索和奖励最大化 |

如果你手上是“同一 prompt 采样 8 个候选，并且有人类完整排序”的数据，RRHF 很合适。它能直接利用多候选结构，不需要把排序强行拆成单一标准答案。

如果你手上只有“回答 A 比回答 B 更好”的成对数据，DPO 可能更省事。DPO，全称 Direct Preference Optimization，是一种直接用成对偏好训练模型的方法，不需要显式训练奖励模型，也不需要 PPO rollout。

如果任务必须让模型在线探索新策略，例如在交互环境里不断尝试新动作并根据反馈更新策略，PPO 更接近目标。PPO，全称 Proximal Policy Optimization，是一种强化学习算法，通常用于 RLHF 中的策略更新，但它需要奖励模型、价值模型、KL 控制和 rollout 管线，工程复杂度明显更高。

RRHF 的价值在于“离线、多候选、直接排序、实现相对简单”。它特别适合已有候选回复池、希望比 SFT 更懂偏好、但又不想引入 PPO 复杂训练系统的团队。

---

## 参考资料

1. [Rank Responses to Align Language Models with Human Feedback](https://arxiv.org/abs/2304.05302)
2. [GanjinZero/RRHF](https://github.com/GanjinZero/RRHF)
3. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
4. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

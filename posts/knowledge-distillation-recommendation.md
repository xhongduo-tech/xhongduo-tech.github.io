## 核心结论

知识蒸馏在推荐系统中，是把大模型 `Teacher` 学到的打分分布、排序关系和中间表示，迁移给小模型 `Student`，让小模型用更低的在线推理成本保住尽量接近的 Top-K 排序效果。

一句话说清楚：`Teacher` 先离线学得更准，`Student` 在线更轻量；短视频、信息流、电商推荐会采用蒸馏，是因为线上排序模型既要准，又要在几十毫秒甚至更短时间内完成推理。

推荐蒸馏的目标不是让 `Student` 完全复制 `Teacher`。线上服务真正关心的是：对同一个用户的一批候选内容，哪些应该排在前面。也就是说，推荐任务里最重要的通常不是单个 item 的绝对分数，而是候选集内部的相对顺序。

| 任务类型 | 主要目标 | 蒸馏重点 |
|---|---|---|
| 图像分类 | 判断样本属于哪个类别 | 类别概率分布 |
| 点击率预测 | 预测单个 item 是否点击 | pointwise 分数 |
| 推荐排序 | 把候选 item 按偏好排序 | Top-K 顺序、pairwise/listwise 关系 |

| 角色 | 特点 | 工程位置 |
|---|---|---|
| `Teacher` | 模型大、特征多、效果好、推理慢 | 离线训练或离线打分 |
| `Student` | 模型小、延迟低、容量有限 | 在线召回、粗排、精排 |

---

## 问题定义与边界

推荐蒸馏问题可以写成：给定用户上下文 `x` 和候选 item 集合 $\{i_1,i_2,\dots,i_n\}$，教师模型给每个 item 一个分数 $z_i^T=f_T(x,i)$，学生模型给出 $z_i^S=f_S(x,i)$。训练目标是让 `Student` 在保留真实点击学习能力的同时，尽量接近 `Teacher` 的排序行为。

术语说明：`Top-K` 是指从候选集中选出分数最高的 K 个 item，并按分数从高到低展示。

玩具例子：同一个用户面前有 3 个视频 A、B、C。目标不是孤立地预测 A 是否点击、B 是否点击、C 是否点击，而是把最可能点击的视频排在最前面。如果 `Teacher` 排序是 A > C > B，而 `Student` 学成 A > B > C，即使每个分数看起来差不多，推荐结果也可能变差。

| 符号 | 含义 |
|---|---|
| `x` | 用户上下文，例如用户历史、时间、设备、场景 |
| `i` | 候选 item，例如视频、商品、文章 |
| $z_i^T$ | `Teacher` 对 item `i` 的 logit 分数 |
| $z_i^S$ | `Student` 对 item `i` 的 logit 分数 |
| $\tau$ | 温度系数，用来调节软标签分布的平滑程度 |

| 推荐目标 | 解释 |
|---|---|
| CTR 预测 | 估计单个 item 被点击的概率 |
| Top-K 排序 | 从候选集中选出最值得展示的 K 个 item |
| 多目标排序 | 同时考虑点击、停留、转化、负反馈等目标 |

蒸馏不是所有推荐系统的必选项。它更适合模型较重、线上延迟敏感、候选集规模大、已有高质量教师模型的场景；如果系统还处在规则推荐或简单模型阶段，先做好样本、特征和评估通常收益更高。

| 适用场景 | 不适用场景 |
|---|---|
| 大模型离线效果明显更好 | `Teacher` 本身效果不稳定 |
| 线上服务有严格延迟约束 | 模型很小，延迟不是瓶颈 |
| 候选 item 多，排序质量重要 | 数据分布频繁漂移且无法及时重训 |
| 希望压缩模型但保住效果 | 训练和部署链路还不稳定 |

---

## 核心机制与推导

推荐蒸馏常见有三类信号：软标签蒸馏、中间层蒸馏、排序蒸馏。

软标签蒸馏是让 `Student` 学 `Teacher` 对候选 item 的概率分布。软标签是模型输出的非 0/1 概率，它比硬标签包含更多信息。例如真实点击标签只告诉你“点了 A”，而教师分布可能告诉你“A 明显最好，C 次之，B 很差”。

对候选集 logits 做带温度的 softmax：

$$
p_i^T=\frac{\exp(z_i^T/\tau)}{\sum_j \exp(z_j^T/\tau)},\quad
p_i^S=\frac{\exp(z_i^S/\tau)}{\sum_j \exp(z_j^S/\tau)}
$$

其中 $\tau$ 越大，分布越平滑；$\tau$ 越小，分布越接近硬排序。软标签损失通常用 KL 散度：

$$
L_{\text{soft}}=\tau^2\mathrm{KL}(p^T\|p^S)
=\tau^2\sum_i p_i^T\log\frac{p_i^T}{p_i^S}
$$

KL 散度的白话解释是：用来衡量两个概率分布有多不一致。这里衡量的是 `Student` 的候选分布和 `Teacher` 的候选分布差多少。

中间层蒸馏是让 `Student` 的隐藏表示接近 `Teacher`。隐藏表示是模型中间层生成的向量，可以理解为模型对用户和 item 的内部表达。由于两个模型维度可能不同，通常加一个投影矩阵 $W_l$：

$$
L_{\text{mid}}=\sum_l \|W_lh_l^S-h_l^T\|_2^2
$$

排序蒸馏关注 item 之间的相对关系。对 item `i` 和 `j`，教师认为 `i` 比 `j` 更好的概率可以写成：

$$
p_{ij}^T=\sigma((z_i^T-z_j^T)/\tau)
$$

对应的学生概率是 $p_{ij}^S$，pairwise 排序损失为：

$$
L_{\text{rank}}=\sum_{(i,j)}\mathrm{BCE}(p_{ij}^T,p_{ij}^S)
$$

`BCE` 是二元交叉熵，用来衡量两个二分类概率的差异。总损失通常写成：

$$
L=L_{\text{rec}}+\alpha L_{\text{soft}}+\beta L_{\text{mid}}+\gamma L_{\text{rank}}
$$

其中 $L_{\text{rec}}$ 是真实点击、转化等监督信号，不能丢。

数值例子：候选集有 3 个 item，`Teacher` logits 是 `[4, 1, 0]`，`Student` logits 是 `[3, 1, 0]`，温度 $\tau=2$。softmax 后，`Teacher` 分布约为 `[0.736, 0.164, 0.100]`，`Student` 分布约为 `[0.629, 0.231, 0.140]`。这说明 `Student` 虽然也把第 1 个 item 排第一，但把第 2 个 item 评得偏高。蒸馏损失会把这种偏差拉回去。

| 蒸馏信号 | 约束目标 | 适合解决的问题 |
|---|---|---|
| 软标签 | 候选集概率分布 | 学到教师的细粒度偏好 |
| 中间层表示 | 内部语义向量 | 小模型表示能力不足 |
| pairwise 排序 | 两两相对顺序 | Top-K 顺序不稳定 |
| listwise 排序 | 整个候选列表结构 | 列表级排序质量下降 |

对抗蒸馏也是一种变体。对抗蒸馏是让一个判别器区分表示来自 `Teacher` 还是 `Student`，同时让 `Student` 生成更像 `Teacher` 的表示。它更复杂，通常在普通蒸馏收益不足时再考虑。

---

## 代码实现

下面是一个最小可运行的 PyTorch 风格训练步骤，展示 soft loss、pairwise rank loss 和真实点击 loss 如何组合。代码里的模型用线性层模拟，重点是损失计算方式。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

batch_size = 2
num_items = 3
feat_dim = 4
temperature = 2.0

teacher = nn.Linear(feat_dim, 1)
student = nn.Linear(feat_dim, 1)

for p in teacher.parameters():
    p.requires_grad = False

features = torch.randn(batch_size, num_items, feat_dim)
labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

with torch.no_grad():
    teacher_logits = teacher(features).squeeze(-1)  # [B, N]

student_logits = student(features).squeeze(-1)      # [B, N]

teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
student_log_prob = F.log_softmax(student_logits / temperature, dim=1)

soft_loss = F.kl_div(
    student_log_prob,
    teacher_prob,
    reduction="batchmean"
) * (temperature ** 2)

rec_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

t_diff = teacher_logits.unsqueeze(2) - teacher_logits.unsqueeze(1)
s_diff = student_logits.unsqueeze(2) - student_logits.unsqueeze(1)

teacher_pair = torch.sigmoid(t_diff / temperature)
student_pair = torch.sigmoid(s_diff / temperature)

rank_loss = F.binary_cross_entropy(student_pair, teacher_pair)

alpha, gamma = 0.5, 0.2
loss = rec_loss + alpha * soft_loss + gamma * rank_loss

assert teacher_logits.shape == (batch_size, num_items)
assert student_logits.shape == (batch_size, num_items)
assert loss.ndim == 0
assert loss.item() > 0

loss.backward()
```

| 张量 | 形状 | 含义 |
|---|---:|---|
| `features` | `[B, N, D]` | B 个用户场景，每个有 N 个候选 item，每个 item D 维特征 |
| `teacher_logits` | `[B, N]` | 教师对候选 item 的打分 |
| `student_logits` | `[B, N]` | 学生对候选 item 的打分 |
| `teacher_pair` | `[B, N, N]` | 教师的两两排序偏好 |
| `loss` | `[]` | 标量总损失 |

| 步骤 | 操作 | 是否更新参数 |
|---|---|---|
| 1 | 加载或训练好 `Teacher` | 否 |
| 2 | 冻结 `Teacher`，离线或在线产出蒸馏信号 | 否 |
| 3 | `Student` 前向计算 logits | 是 |
| 4 | 计算真实标签损失和蒸馏损失 | 是 |
| 5 | 反向传播，只更新 `Student` | 是 |

真实工程例子：短视频推荐中，`Teacher` 可以使用更深的特征交叉、长行为序列、图特征和大 Transformer；`Student` 只保留轻量 MLP、小 embedding 或短序列模块，部署在 pre-rank 或 rank 阶段。`Teacher` 不直接上线，而是离线产出软标签和排序关系，`Student` 负责 10ms 级在线推理。

---

## 工程权衡与常见坑

蒸馏不是“越多越好”。软标签、中间层、pairwise、listwise 都会增加训练复杂度，有些还会显著增加样本构造和存储成本。工程上要看的不是单个 loss 是否更小，而是线上延迟、离线 AUC/NDCG、在线点击率、停留时长和稳定性是否共同变好。

| 问题 | 后果 | 规避 |
|---|---|---|
| 只蒸馏 pointwise 分数 | Top-K 顺序学不好 | 加 pairwise 或 listwise 蒸馏 |
| `Teacher` 和 `Student` 候选集不一致 | 学到错误相对关系 | 统一候选池或统一采样策略 |
| 负样本噪声大 | 学生被低质量样本带偏 | 使用 rank-aware sampling |
| 温度 $\tau$ 太低 | 软标签退化成硬标签 | 从 `2~4` 开始调参 |
| 只学 `Teacher`，丢掉真实标签 | 继承教师偏差 | 保留 $L_{\text{rec}}$ |
| 教师效果不稳定 | 学生上限受限 | 先验证教师校准和排序质量 |

一个常见错误是：只让 `Student` 拟合每个 item 的教师分数。例如 `Teacher` 对 A、B、C 打分 `[0.91, 0.90, 0.20]`，`Student` 学成 `[0.86, 0.88, 0.19]`。单点误差不大，但 A 和 B 的顺序反了。对于 Top-1 展示位，这就是实质错误。改法是加入 pairwise 蒸馏，让模型明确学习 A > B、B > C、A > C。

| 策略 | 收益 | 成本 |
|---|---|---|
| 软标签蒸馏 | 简单、稳定、易实现 | 对排序结构约束较弱 |
| 中间层蒸馏 | 改善表示能力 | 需要对齐层和维度 |
| pairwise 蒸馏 | 直接优化相对顺序 | 两两组合成本较高 |
| listwise 蒸馏 | 更接近推荐列表目标 | 实现和采样更复杂 |
| 对抗蒸馏 | 表示分布更接近 | 训练稳定性更难控制 |

| 超参数 | 建议起点 | 作用 |
|---|---:|---|
| $\tau$ | `2~4` | 控制软标签平滑程度 |
| $\alpha$ | `0.1~1.0` | 控制软标签蒸馏权重 |
| $\beta$ | `0.01~0.5` | 控制中间层蒸馏权重 |
| $\gamma$ | `0.05~0.5` | 控制排序蒸馏权重 |

---

## 替代方案与适用边界

知识蒸馏只是模型压缩的一种方式。推荐系统还可以用剪枝、量化、低秩分解、特征裁剪、缓存、召回/粗排/精排分层等方式降低成本。

| 方案 | 核心做法 | 优点 | 局限 |
|---|---|---|---|
| 模型剪枝 | 删除低重要性参数或结构 | 可直接减小模型 | 可能破坏排序能力 |
| 量化 | 用低精度数值表示参数 | 推理加速明显 | 精度可能下降 |
| 低秩分解 | 用更小矩阵近似大矩阵 | 适合大 embedding 或大线性层 | 需要额外调参 |
| 特征裁剪 | 删除低收益特征 | 降低特征计算成本 | 可能损失长尾信号 |
| 两阶段排序 | 粗排过滤，精排少量候选 | 工程收益大 | 链路更复杂 |
| 知识蒸馏 | 学习教师输出和排序 | 能补回压缩损失 | 依赖高质量教师 |

“只做模型量化”和“量化 + 蒸馏”的区别在于：量化主要改变模型的数值表示，让推理更快；蒸馏则重新训练 `Student`，让它主动学习 `Teacher` 的排序行为。实际工程中，一个小模型量化后可能 Top-K 效果下降，再加蒸馏训练，有机会把压缩造成的排序损失拉回来。

| 边界 | 判断标准 |
|---|---|
| 适合蒸馏 | 有强教师、线上延迟严格、排序质量重要 |
| 谨慎使用 | 教师和线上分布有偏差、候选集构造不一致 |
| 不优先使用 | 当前瓶颈是数据质量、特征缺失或评估体系混乱 |
| 不建议使用 | 教师模型本身弱，或业务目标频繁变化到无法稳定训练 |

选择建议：先确认线上瓶颈是不是模型推理；再训练或选择一个可靠 `Teacher`；然后从软标签蒸馏开始；如果 Top-K 顺序仍差，再加 pairwise/listwise；最后再考虑中间层和对抗蒸馏。不要一开始就把所有蒸馏项都堆上去。

---

## 参考资料

| 阅读顺序 | 内容 | 用途 |
|---|---|---|
| 1 | 基础蒸馏 | 先理解 soft label 和温度 |
| 2 | 中间层蒸馏 | 再理解表示对齐 |
| 3 | 推荐蒸馏 | 最后看排序、序列、多样性场景 |

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)：知识蒸馏的基础论文，适合先理解软标签和温度系数。
2. [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)：中间层蒸馏代表工作，用来理解隐藏表示对齐。
3. [Knowledge distillation meets recommendation: collaborative distillation for top-N recommendation](https://doi.org/10.1007/s10115-022-01667-8)：推荐蒸馏综述性质工作，适合理解 Top-N 推荐中的蒸馏方法。
4. [Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation](https://arxiv.org/abs/2311.01056)：序列推荐蒸馏论文，关注 item transition 知识迁移。
5. [Contextual Distillation Model for Diversified Recommendation](https://arxiv.org/abs/2406.09021)：多样性推荐中的上下文蒸馏方法，适合进阶阅读。

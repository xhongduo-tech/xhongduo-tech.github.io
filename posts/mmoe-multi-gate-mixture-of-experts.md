## 核心结论

MMoE（Multi-gate Mixture-of-Experts，多门控专家混合模型）是一种多任务学习结构：多个 Expert 共享底层输入表示，每个任务用自己的 Gate 为这些 Expert 分配权重，再把加权后的表示交给任务专属 Tower 输出预测结果。

它的本质不是“所有任务共享同一套表示”，而是“共享多个专家，由每个任务自己选择专家组合”。Expert 可以理解为一组候选特征提取器；Gate 是一个小网络，用来决定某个任务当前更该相信哪些 Expert；Tower 是任务自己的输出网络。

在推荐系统里，常见任务包括 `click`、`conversion`、`long_view`。它们相关，但不完全一致：点击样本多、转化样本少，点击偏即时兴趣，转化更接近购买意图，长观看重内容质量。MMoE 允许这些任务共享底层专家，同时让每个任务偏向不同专家，因此比完全共享底座更灵活，也比完全独立训练更能利用数据。

| 方法 | 表征共享方式 | 任务是否独立选择特征 | 主要优点 | 主要缺点 |
|---|---|---|---|---|
| Shared-Bottom | 所有任务共享同一个底层网络 | 否 | 结构简单，训练成本低 | 容易负迁移 |
| MMoE | 共享多个 Expert，每个任务有独立 Gate | 是 | 兼顾共享与任务差异 | 结构和调参更复杂 |
| 独立模型 | 每个任务单独训练一套模型 | 是 | 任务互不干扰 | 无法共享数据，成本高 |

负迁移是指一个任务的训练信号损害另一个任务的效果。MMoE 的主要价值就是缓解这种问题，尤其适合“任务相关但目标不完全一致”的任务集合。

---

## 问题定义与边界

多任务学习是指：同一份输入样本 $x$ 同时对应多个任务标签 $y_1, y_2, ..., y_K$，模型不是分别训练 $K$ 个独立模型，而是联合学习多个预测函数。

最小定义如下：

```text
输入：x
任务：k = 1..K
标签：y_1, y_2, ..., y_K
目标：联合学习多个任务预测函数
```

设有 $K$ 个任务，每个任务的预测函数为 $\hat{y}_k = F_k(x)$。联合训练目标通常写成：

$$
\mathcal{L} = \sum_{k=1}^{K} \alpha_k \mathcal{L}_k(\hat{y}_k, y_k)
$$

其中，$\mathcal{L}_k$ 是第 $k$ 个任务的损失函数，$\alpha_k$ 是该任务的损失权重。损失权重的作用是控制不同任务对总训练方向的影响。

MMoE 适用于任务之间存在共享信息，但又不能完全共用一套表示的场景。推荐系统中的 `click`、`conversion`、`long_view` 是典型例子：用户兴趣、商品质量、上下文特征都可能共享，但每个任务对这些特征的使用方式不同。

反例是“图像分类 + 股票预测”。这两个任务输入模态、业务目标、数据分布都不同，强行共享 Expert 往往收益有限，甚至会让训练更不稳定。

| 场景 | 任务相关性 | 是否适合 MMoE | 原因 |
|---|---:|---|---|
| 点击率、转化率、长观看预测 | 高但不完全一致 | 适合 | 共享用户和内容表示，同时保留任务差异 |
| 搜索排序中的点击、收藏、购买 | 中高 | 适合 | 不同目标都反映用户偏好 |
| 图像分类和股票预测 | 很低 | 不适合 | 表征空间和目标函数差异过大 |
| 两个几乎相同的点击任务 | 极高 | 不一定需要 | Shared-Bottom 可能已经足够 |
| 一个任务标签极稀疏且噪声大 | 不稳定 | 谨慎使用 | Gate 和 Tower 可能学不到稳定信号 |

边界要写清楚：MMoE 不是万能结构。它解决的是“相关任务之间如何共享又如何分化”的问题，不负责自动消除所有数据质量问题。如果任务无关、标签噪声很大、样本分布差异极端，MMoE 可能不如独立建模。

---

## 核心机制与推导

Shared-Bottom 的结构很直接：所有任务先经过同一个共享底座，再接各自的 Tower。问题是这个共享表示过于统一。任务 A 需要的特征，任务 B 未必需要；任务 B 的梯度也可能把共享底座推向对任务 A 不利的方向。

MMoE 的改法是：把“一个共享底座”拆成“多个共享 Expert”，再让每个任务通过自己的 Gate 选择 Expert 组合。这样模型仍然共享参数，但不是强制所有任务使用同一种表示。

第 $i$ 个专家输出为：

$$
f_i(x) = ReLU(W_i x + b_i)
$$

其中，$x$ 是输入特征，$W_i$ 和 $b_i$ 是第 $i$ 个 Expert 的参数，$f_i(x) \in R^d$ 是专家输出的 $d$ 维表示。$ReLU$ 是常用激活函数，会把负数截断为 0。

第 $k$ 个任务的 Gate 输出为：

$$
g^k(x) = softmax(W_g^k x + c^k)
$$

其中，$W_g^k$ 和 $c^k$ 是第 $k$ 个任务 Gate 的参数，$g^k(x) \in R^E$，$E$ 是 Expert 数量。$softmax$ 会把一组分数转成概率权重，因此满足：

$$
\sum_{i=1}^{E} g_i^k(x) = 1
$$

第 $k$ 个任务的混合表示为：

$$
z^k = \sum_{i=1}^{E} g_i^k(x) \cdot f_i(x)
$$

其中，$g_i^k(x)$ 是任务 $k$ 对第 $i$ 个 Expert 的权重，$z^k$ 是任务 $k$ 最终拿到的共享表示。

最后，任务输出为：

$$
\hat{y}_k = h_k(z^k)
$$

其中，$h_k$ 是第 $k$ 个任务的 Tower。Tower 是任务专属网络，负责把任务表示转成最终预测值，例如点击概率、转化概率或观看时长分数。

一个玩具例子可以直接看出 Gate 的作用。设 $E=2$，$K=2$，两个 Expert 输出都是标量：

```text
f1(x) = 2
f2(x) = 5
```

任务 A 的 Gate 权重为 `[0.881, 0.119]`：

```text
zA = 0.881×2 + 0.119×5 = 2.357
```

任务 B 的 Gate 权重为 `[0.119, 0.881]`：

```text
zB = 0.119×2 + 0.881×5 = 4.643
```

同一个输入、同一组 Expert，因为任务 Gate 不同，最后得到的任务表示不同。这就是 MMoE 的核心机制：Expert 共享，Gate 分化。

真实工程里，`click` 任务可能更依赖即时兴趣 Expert，`conversion` 任务可能更依赖购买意图 Expert，`long_view` 任务可能更依赖内容质量 Expert。MMoE 不需要人为指定哪个 Expert 服务哪个任务，而是通过训练让 Gate 学出这种偏好。

---

## 代码实现

实现 MMoE 通常分成三层：共享 Expert 层、任务 Gate 层、任务 Tower 层。工程上最重要的不是堆很多网络层，而是保证三件事：所有 Expert 输出维度一致；每个任务有独立 Gate；多个任务的 loss 能稳定联合训练。

张量形状通常如下：

```python
# x: [batch_size, input_dim]

# 1) experts
expert_outputs = [expert_i(x) for i in range(E)]  # each: [batch, d]
expert_stack = torch.stack(expert_outputs, dim=1) # [batch, E, d]

# 2) task gates
gate_weights_k = softmax(gate_k(x), dim=-1)       # [batch, E]

# 3) task-specific mixture
z_k = torch.sum(gate_weights_k.unsqueeze(-1) * expert_stack, dim=1)  # [batch, d]

# 4) task tower
y_k = tower_k(z_k)
```

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Expert | `x` | `f_i(x)` | 提取候选特征 |
| Gate | `x` | `g^k(x)` | 为任务分配专家权重 |
| Tower | `z^k` | `ŷ_k` | 输出最终任务结果 |

下面是一个最小可运行的 Python 例子，不依赖深度学习框架，只演示 Gate 如何把同一组 Expert 组合成不同任务表示：

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]

def mix(expert_outputs, gate_logits):
    weights = softmax(gate_logits)
    return sum(w * f for w, f in zip(weights, expert_outputs)), weights

# two experts, scalar outputs
expert_outputs = [2.0, 5.0]

# task A prefers expert 1
z_a, gate_a = mix(expert_outputs, [2.0, 0.0])

# task B prefers expert 2
z_b, gate_b = mix(expert_outputs, [0.0, 2.0])

assert round(gate_a[0], 3) == 0.881
assert round(gate_a[1], 3) == 0.119
assert round(z_a, 3) == 2.358

assert round(gate_b[0], 3) == 0.119
assert round(gate_b[1], 3) == 0.881
assert round(z_b, 3) == 4.642

print("task A:", z_a, gate_a)
print("task B:", z_b, gate_b)
```

训练时，总 loss 通常是多个任务 loss 的加权和：

```python
loss = sum(alpha_k * loss_k(y_pred_k, y_true_k) for k in tasks)
loss.backward()
optimizer.step()
```

这里的 `alpha_k` 很关键。真实推荐系统里，`click` 样本通常远多于 `conversion` 样本。如果直接相加，点击任务可能主导训练，导致转化任务的 Gate 学不到合适的专家偏好。

---

## 工程权衡与常见坑

MMoE 的效果高度依赖任务权重、Expert 数量、Gate 稳定性和任务相关性。它不是把 Shared-Bottom 替换成多专家结构就一定提升。

最常见的问题是 loss 不平衡。比如推荐系统同时训练 `click` 和 `conversion`，点击标签数量大、反馈密集，转化标签数量少、反馈稀疏。如果不做 loss scaling，模型可能主要优化点击，Gate 也会偏向服务点击任务的 Expert，最后转化收益不明显。

gate 塌缩也是常见问题。gate 塌缩是指模型最后几乎只使用一个 Expert，其他 Expert 权重长期接近 0。白话说，就是模型表面上有多个专家，实际只信一个专家，多专家结构没有发挥作用。

| 问题 | 典型表现 | 处理方式 |
|---|---|---|
| loss 不平衡 | 某个任务主导训练 | `loss scaling`、动态权重、分任务采样 |
| gate 塌缩 | 大部分样本只用一个 Expert | 熵正则、load balancing、调低 Gate 学习率 |
| Expert 太少 | 多任务共用能力不足，欠拟合 | 从 4 到 8 个 Expert 起步尝试 |
| Expert 太多 | 冗余、过拟合、训练慢 | 减少 Expert 数，加正则 |
| 任务相关性不足 | 多任务训练不如单任务 | 拆分模型或只共享部分特征 |
| 稀疏任务不稳定 | 小任务指标波动大 | loss scaling、warmup、增加样本权重 |

调参时可以按这个顺序排查：

1. 先确认单任务模型和 Shared-Bottom 基线，不要直接只看 MMoE。
2. Expert 数先从 4 到 8 试，任务少时不必一开始堆很多 Expert。
3. 先看每个任务的 loss 曲线，再看总 loss。
4. 监控 Gate 分布熵。熵越低，说明 Gate 越集中；过低可能意味着塌缩。
5. 先做 loss scaling，再判断结构是否有效。
6. 对稀疏任务单独看 AUC、校准、召回等业务指标，不要只看整体平均指标。

真实工程中还有一个容易被忽略的点：Gate 的输入是否应该和 Expert 的输入完全一样。很多实现会让 Gate 使用同一份输入 $x$，但在推荐系统里，也可能给 Gate 更多任务相关上下文，例如场景、流量入口、用户活跃度分桶。这样 Gate 更容易学会“什么样的样本该用什么专家”。

---

## 替代方案与适用边界

MMoE 不是多任务学习的唯一答案。选择结构时，核心依据是任务相关性、数据规模、是否存在强弱任务，以及训练稳定性要求。

| 方法 | 结构特点 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 独立模型 | 每个任务单独建模 | 互不干扰，调试简单 | 无法共享数据，维护成本高 | 任务相关性低 |
| Shared-Bottom | 所有任务共享底层网络 | 简单高效 | 负迁移风险高 | 任务高度相关 |
| MMoE | 多 Expert 共享，多 Gate 分任务选择 | 兼顾共享与分化 | 调参复杂，可能 Gate 塌缩 | 任务相关但不完全一致 |
| PLE | 区分共享专家和任务专属专家 | 分路更强，适合任务冲突 | 结构更复杂 | 任务冲突明显 |
| Cross-Stitch | 学习任务表示之间的线性组合 | 表示交互直观 | 扩展到大规模任务较复杂 | 小规模多任务 |

如果两个任务几乎完全一致，例如两个相近口径的点击任务，Shared-Bottom 可能已经足够。此时使用 MMoE 可能只是增加参数和训练成本。

如果任务冲突明显，例如一个任务偏短期点击，另一个任务强烈偏长期留存，MMoE 可能仍然不够，需要更强的分路结构。PLE（Progressive Layered Extraction，渐进式分层抽取）会显式区分共享专家和任务专属专家，适合处理更强的任务差异。

如果任务相关性太低，MMoE 可能不如独立建模。共享本身不是收益来源，只有共享到有效信息才是收益来源。低相关任务强行共享，会把噪声也一起共享。

如果数据极少，多专家结构也可能更不稳定。因为 Expert 和 Gate 都需要数据来学习，样本不足时，模型容量变大反而更容易过拟合。

如果追求极致可解释性，MMoE 的 Gate 只能提供有限解释。它能说明“某个任务更偏向哪些 Expert”，但不能直接说明“这个 Expert 为什么导致预测结果变化”，更不能把 Gate 权重解释成因果关系。

因此，工程选择可以简化成一句话：任务高度一致用 Shared-Bottom，任务相关但有差异用 MMoE，任务冲突明显考虑 PLE，任务基本无关就独立建模。

---

## 参考资料

| 类型 | 名称 | 作用 |
|---|---|---|
| 论文 | Google Research 原论文页面 | 理解 MMoE 的原始定义 |
| 代码 | `keras-mmoe` 参考实现 | 对照实现细节 |
| 教程 | TensorFlow 多任务推荐教程 | 理解工程落地 |

1. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)
2. [keras-mmoe GitHub Repository](https://github.com/drawbridge/keras-mmoe)
3. [keras-mmoe mmoe.py Reference Implementation](https://github.com/drawbridge/keras-mmoe/blob/master/mmoe.py)
4. [TensorFlow Recommenders Multi-task Recommenders](https://www.tensorflow.org/recommenders/examples/multitask/)

## 核心结论

冷启动迁移学习，是把源域学到的表示、参数或对齐约束迁到目标域，在目标域标注很少甚至没有时，减少从零训练所需的数据量和训练成本。

这里的源域，指数据比较充分、标签比较完整的原始场景；目标域，指新上线、数据少、标签少、分布可能变化的新场景。冷启动的关键不是“把旧模型直接搬过去”，而是在保留原任务能力的前提下，缩小源域和目标域之间的分布差异。

玩具例子：旧产线已经有 10 万张缺陷检测图片，模型学会了裂纹、污点、变形等视觉特征。新产线只有 200 张图片，直接从零训练很容易过拟合。更合理的做法是复用旧模型的特征提取能力，再用少量新产线数据微调，或者加入域对齐，让旧产线和新产线的特征分布更接近。

| 方案 | 迁移什么 | 目标域是否需要标注 | 核心风险 |
|---|---|---:|---|
| 从零训练 | 不迁移，重新学习全部参数 | 需要大量标注 | 冷启动阶段数据不足 |
| Fine-tuning | 迁移预训练参数 | 需要少量标注 | 学习率过大导致遗忘 |
| Domain Adaptation | 迁移并对齐特征分布 | 可少量或无标注 | 对齐过度会伤害分类边界 |
| DANN | 学习域不变特征 | 可使用无标注目标域 | 对抗训练不稳定 |

总览可以写成：

```text
源域数据 Ds ──训练──> 已有表示/参数
                         │
                         ▼
目标域数据 Dt ──微调或对齐──> 目标域可用模型
```

结论先行：有目标域标注时，优先从 fine-tuning 做起；目标域几乎无标注但源目标任务一致时，再考虑 Domain Adaptation、DANN 或 MMD 这类对齐方法；如果任务或标签空间已经变了，迁移收益会明显下降。

---

## 问题定义与边界

设源域为：

$$
D_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}
$$

目标域为：

$$
D_t = \{x_j^t\}_{j=1}^{n_t}
$$

其中 $x$ 是输入样本，例如图片、文本、用户行为序列；$y$ 是任务标签，例如“正常/缺陷”“点击/不点击”“类别 A/B/C”。当目标域有少量标注时，也可以写成：

$$
D_t^l = \{(x_j^t, y_j^t)\}_{j=1}^{m_t}, \quad m_t \ll n_s
$$

冷启动迁移学习的基本前提是：源域和目标域之间存在可复用知识。这个知识可能是视觉边缘、纹理、形状，也可能是文本语义、用户偏好模式或模型参数初始化。

新手版本的工程场景：

| 元素 | 示例 |
|---|---|
| 源域 | 旧相机拍摄的质检图片，带完整缺陷标签 |
| 目标域 | 新相机拍摄的质检图片，标签很少或没有 |
| 任务 | 仍然判断正常、裂纹、污点、变形 |
| 分布变化 | 相机、光照、背景、角度发生变化 |
| 迁移目标 | 让旧任务在新数据上继续有效 |

边界需要先说清楚：

| 情况 | 是否适合迁移 | 说明 |
|---|---:|---|
| 同任务、同标签空间、不同分布 | 适合 | 典型域适应问题 |
| 同任务、目标域少量标注 | 适合 | fine-tuning 是首选基线 |
| 无标签目标域 | 可尝试 | 需要无监督域适应或自监督方法 |
| 标签空间部分变化 | 谨慎 | 可能需要重定义输出层或样本重加权 |
| 任务完全变化 | 通常不适合直接迁移 | 源域知识只能作为弱初始化 |

场景判定清单：

| 问题 | 判断标准 |
|---|---|
| 是否同任务 | 源域和目标域预测目标是否一致 |
| 是否同标签空间 | 输出类别是否相同 |
| 是否有目标域标注 | 有多少可验证模型质量的样本 |
| 分布偏移是否可解释 | 偏移来自设备、时间、用户群、语言风格还是业务规则 |
| 是否存在 label shift | 标签比例是否明显变化，例如缺陷率从 1% 变成 20% |

label shift 指标签分布变化。比如老产线缺陷率是 1%，新产线因为设备不稳定，缺陷率变成 15%。这种情况下，强行让两个域“完全一样”可能会把真实业务差异抹掉。

---

## 核心机制与推导

迁移学习可以拆成三个组件：

```text
x -> G -> z -> C -> y
          │
          └-> D -> d
```

$G$ 是特征提取器，白话说就是把原始输入转成模型更容易处理的特征；$C$ 是任务分类器，用特征预测业务标签；$D$ 是域判别器，用特征判断样本来自源域还是目标域。

Fine-tuning 的核心是迁移参数初始化。先令：

$$
\theta \leftarrow \theta_{pretrained}
$$

再用目标域少量标注数据最小化目标任务损失：

$$
L_t(\theta) = \frac{1}{m_t}\sum_j \ell(f_\theta(x_j^t), y_j^t)
$$

Domain Adaptation 的核心是显式缩小域差异。域差异可以理解为：同一个模型看源域样本和目标域样本时，特征分布是否明显不同。如果源域特征集中在一个区域，目标域特征集中在另一个区域，源域训练出的分类边界迁到目标域就可能失效。

源域任务损失为：

$$
L_{task} = \frac{1}{n_s}\sum_i \ell(C(G(x_i^s)), y_i^s)
$$

域损失为：

$$
L_{dom} =
\frac{1}{n_s+n_t}\sum_i CE(D(G(x_i)), d_i)
$$

其中 $d_i$ 是域标签，源域可记为 0，目标域可记为 1。$CE$ 是交叉熵损失，白话说就是分类预测越错，惩罚越大。

DANN，即 Domain-Adversarial Training of Neural Networks，用对抗训练学习域不变特征。它的目标是：

$$
\min_{G,C} \max_D \ L_{task} - \lambda L_{dom}
$$

这句话可以拆开理解：域判别器 $D$ 要尽量分清样本来自哪个域，所以它最小化 $L_{dom}$；特征提取器 $G$ 要“骗过”域判别器，让 $D$ 分不清域，所以 $G$ 反向最大化域损失。GRL，即 Gradient Reversal Layer，中文常叫梯度反转层，它前向传播时什么也不做，反向传播时把传给 $G$ 的域损失梯度乘以 $-\lambda$：

$$
-\lambda \nabla_G L_{dom}
$$

玩具例子：把 $G$ 理解成“看图提特征的眼睛”，$C$ 是“判断类别的脑子”，$D$ 是“判断图片来自哪个产线的探测器”。DANN 希望 $C$ 仍然能判断缺陷类别，同时让 $D$ 看不出图片来自旧产线还是新产线。这样 $G$ 学到的特征更偏向“缺陷本身”，而不是“哪条产线拍的”。

域适应理论里有一个常见形式的目标误差上界：

$$
\epsilon_T(h) \le \epsilon_S(h) + \frac{1}{2}d_{H\Delta H}(D_s, D_t) + \lambda^*
$$

其中 $\epsilon_T(h)$ 是目标域误差，$\epsilon_S(h)$ 是源域误差，$d_{H\Delta H}$ 衡量源域和目标域的差异，$\lambda^*$ 表示两个域上共同最优模型仍然不可避免的误差。

这个式子的含义很直接：目标域误差受三部分控制。第一，源域上本来要学得好；第二，源域和目标域不能差太远；第三，两个域的任务本身要兼容。对齐方法主要作用在第二项上。如果 $d_{H\Delta H}$ 下降，在其他项不恶化的前提下，目标域误差上界也会下降。

数值例子：某分类器源域误差 $\epsilon_S=0.08$，域差异 $d_{H\Delta H}=0.30$，共同最优误差 $\lambda^*=0.05$，则：

$$
\epsilon_T \le 0.08 + 0.5 \times 0.30 + 0.05 = 0.28
$$

如果对齐后域差异降到 $0.10$：

$$
\epsilon_T \le 0.08 + 0.5 \times 0.10 + 0.05 = 0.18
$$

这不是说真实误差一定等于 0.18，而是说明“缩小域差异”有明确的理论动机。

---

## 代码实现

工程实现建议分两条线：先做 fine-tuning 的最小闭环，再扩展到 DANN 或其他域对齐方法。

| 模块 | 作用 | 新手优先级 |
|---|---|---:|
| 数据加载 | 分别读取源域、目标域、验证集 | 高 |
| 模型初始化 | 加载预训练 backbone 和分类头 | 高 |
| 冻结 / 解冻 | 控制哪些层参与训练 | 高 |
| 训练循环 | 计算损失、反向传播、更新参数 | 高 |
| 验证与保存 | 只用干净验证集选择模型 | 高 |
| 域判别器 | 判断样本来自源域还是目标域 | 中 |
| GRL | 反转域损失梯度 | 中 |

Fine-tuning 伪代码：

```text
θ ← θ_pretrained
replace classifier head
freeze(backbone)
optimize L_t(θ) on target labeled data
unfreeze top blocks
continue training with smaller learning rate
```

DANN 伪代码：

```text
for source_batch, target_batch:
    z_s = G(x_s)
    z_t = G(x_t)

    y_pred = C(z_s)
    L_task = CE(y_pred, y_s)

    d_pred = D(GRL(concat(z_s, z_t)))
    L_dom = CE(d_pred, domain_labels)

    L = L_task + L_dom
    backward(L)
```

下面是一个最小可运行的 Python 玩具实现，用线性模型模拟“从源域预训练，再到目标域微调”。它不依赖深度学习框架，重点展示迁移初始化确实能降低冷启动训练难度。

```python
import numpy as np

rng = np.random.default_rng(7)

def make_data(n, shift):
    x = rng.normal(size=(n, 2)) + shift
    y = (x[:, 0] + x[:, 1] > shift.sum()).astype(float)
    return x, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logreg(x, y, w=None, b=0.0, lr=0.2, steps=300):
    if w is None:
        w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = sigmoid(x @ w + b)
        grad_w = x.T @ (p - y) / len(y)
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def acc(x, y, w, b):
    pred = (sigmoid(x @ w + b) >= 0.5).astype(float)
    return np.mean(pred == y)

# 源域：样本多
xs, ys = make_data(2000, shift=np.array([0.0, 0.0]))
w_source, b_source = train_logreg(xs, ys, steps=500)

# 目标域：样本少，并且分布有偏移
xt_train, yt_train = make_data(20, shift=np.array([0.8, -0.4]))
xt_test, yt_test = make_data(1000, shift=np.array([0.8, -0.4]))

# 从零训练：目标域只有 20 个样本
w_scratch, b_scratch = train_logreg(xt_train, yt_train, steps=80)

# 迁移微调：用源域参数初始化，再用目标域少量样本更新
w_ft, b_ft = train_logreg(xt_train, yt_train, w=w_source.copy(), b=b_source, lr=0.05, steps=80)

scratch_acc = acc(xt_test, yt_test, w_scratch, b_scratch)
ft_acc = acc(xt_test, yt_test, w_ft, b_ft)

print("scratch:", scratch_acc)
print("fine_tuning:", ft_acc)

assert ft_acc >= 0.85
assert ft_acc >= scratch_acc - 0.05
```

真实工程例子：推荐系统新城市冷启动。源域是老城市的用户点击、收藏、购买行为，目标域是刚开城的新城市。用户和商品体系大体一致，但新城市用户偏好、价格敏感度、供给结构不同。可行方案是先用老城市数据训练召回或排序模型，再用新城市少量行为做 fine-tuning；如果新城市行为太少，可以加入域判别器，让模型表示减少“城市身份”信息，同时保留点击率预测损失。

分层学习率示意：

| 参数部分 | 学习率 | 原因 |
|---|---:|---|
| 预训练 backbone 底层 | 1e-5 | 保留通用特征 |
| backbone 高层 | 5e-5 | 允许适配目标域 |
| 新分类头 | 1e-3 | 随机初始化，需要更快学习 |
| 域判别器 | 1e-4 | 避免对抗训练过强 |

---

## 工程权衡与常见坑

冷启动迁移不是单纯追求“域不变”。模型需要同时满足两个目标：特征在源域和目标域之间足够接近，类别边界又足够清晰。如果只看域对齐，可能把不同类别压到一起；如果只看源域任务损失，模型可能继续依赖源域特有噪声。

| 常见坑 | 后果 | 规避策略 |
|---|---|---|
| 测试集混进对齐训练 | 指标虚高，线上失效 | 按时间、产线、用户、站点严格切分 |
| 只做域对齐，不保留任务损失 | 类别边界被抹平 | 始终保留源域监督损失 |
| label shift 下强行域不变 | 真实业务差异被压掉 | 先检查标签比例变化 |
| fine-tuning 学习率过大 | 灾难性遗忘 | 先冻结 backbone，再逐步解冻 |
| 冻结策略过死 | 目标域适配不足 | 观察验证集后解冻高层 |
| 只看总体准确率 | 少数类质量变差 | 同时看召回率、AUC、分组指标 |

灾难性遗忘，指模型在新数据上训练后，把原来学到的通用能力破坏掉。小数据 fine-tuning 特别容易出现这个问题，因为少量目标域样本可能不足以代表真实分布。

训练策略建议：

| 阶段 | 做法 | 监控指标 |
|---|---|---|
| 预热阶段 | 冻结 backbone，只训练分类头 | 目标域验证集损失 |
| 初步适配 | 解冻高层，小学习率训练 | AUC、F1、少数类召回 |
| 域对齐 | 加入 DANN 或 MMD，控制权重 | 域判别准确率、任务指标 |
| 收敛选择 | 早停并保存最佳模型 | 目标域验证集指标 |

MMD，即 Maximum Mean Discrepancy，中文常叫最大均值差异，它用统计量衡量两个分布在特征空间里的差别。相比 DANN，MMD 不需要训练域判别器，但需要选择合适的核函数和对齐层。

一个容易被忽略的问题是验证集设计。目标域样本少时，很多团队会把所有目标域数据都用于训练或对齐，然后用同一批数据报告效果。这会造成数据泄漏。正确做法是即使目标域很小，也要留出独立验证集，或者按时间滚动评估。例如用第 1 周新城市数据训练，用第 2 周数据验证，用第 3 周数据测试。

---

## 替代方案与适用边界

迁移学习不是唯一方案。目标域标注量、任务一致性、标签分布变化、域差异大小，都会影响方案选择。

| 方案 | 适用条件 | 优点 | 局限 |
|---|---|---|---|
| 从零训练 | 目标域标注充足 | 不受源域偏差影响 | 冷启动成本高 |
| Feature extraction | 目标域极少标注 | 简单稳定 | 适配能力有限 |
| Fine-tuning | 目标域有少量标注 | 工程首选基线 | 学习率和冻结策略敏感 |
| DANN | 目标域无标注或少标注 | 可利用无标签目标域 | 训练不稳定 |
| MMD 对齐 | 需要显式分布对齐 | 实现相对直接 | 对核和层选择敏感 |
| 自监督预训练 | 大量无标签目标域数据 | 提升表示能力 | 训练成本较高 |
| 半监督学习 | 少量标注加大量无标注 | 能扩大目标域监督信号 | 伪标签错误会累积 |

Feature extraction 指只把预训练模型当固定特征提取器，不更新 backbone，只训练最后的小模型。它比 fine-tuning 更稳定，但目标域差异较大时效果可能不足。

适用边界表：

| 判断项 | 推荐做法 |
|---|---|
| 同任务、目标域有少量标注 | fine-tuning |
| 同任务、目标域无标注 | DANN、MMD、自监督预训练 |
| 标签比例明显变化 | 先处理 label shift，再考虑对齐 |
| 源目标任务不同 | 重新定义任务，只把预训练作为初始化 |
| 域差异过大 | 增加目标域采样和标注，不能只靠算法 |

决策流程可以写成：

```text
是否同任务？
  否 -> 重新定义迁移目标，不能直接套域适应
  是 -> 目标域是否有标注？
        有 -> 先做 fine-tuning 基线
        无 -> 是否有大量无标签目标域数据？
              有 -> 自监督 / DANN / MMD
              无 -> 优先补采样和补标注
        再检查是否有明显 label shift
        最后决定是否组合 fine-tuning + 域对齐
```

新手版本反例：源域是“猫狗分类”，目标域变成“车辆类型分类”。这不是简单的域迁移，因为任务和标签空间都变了。旧模型学到的边缘、纹理等底层视觉特征可能还有用，但分类头、类别边界和业务目标都需要重新设计。此时直接上 DANN，通常不会解决核心问题。

真实工程里，最稳妥的推进顺序是：先建立从零训练和 fine-tuning 两个基线，再加入域对齐方法。只有当 fine-tuning 明显受限于分布偏移，并且验证集能证明对齐有效时，DANN 或 MMD 才值得进入主流程。

---

## 参考资料

| 资料 | 核心贡献 | 适合章节 | 适合读者层级 |
|---|---|---|---|
| Ben-David et al., A theory of learning from different domains | 给出域适应误差上界 | 核心机制与推导 | 初级到中级 |
| Ganin et al., Domain-Adversarial Training of Neural Networks | 提出 DANN 和梯度反转思想 | 核心机制、代码实现 | 初级到中级 |
| Hugging Face Docs, Fine-tuning | 展示预训练模型微调流程 | 代码实现 | 零基础到初级 |
| PyTorch Tutorial, Transfer Learning for Computer Vision | 展示计算机视觉迁移学习工程写法 | 代码实现 | 零基础到初级 |

1. [Ben-David et al., A theory of learning from different domains](https://research.google/pubs/a-theory-of-learning-from-different-domains/)
2. [Ganin et al., Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
3. [Hugging Face Docs, Fine-tuning](https://huggingface.co/docs/transformers/en/training)
4. [PyTorch Tutorial, Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer)

阅读顺序建议：

1. 先看问题定义，明确源域、目标域、任务和标签空间。
2. 再看 Ben-David 的理论界，理解目标域误差为什么和源域误差、域差异有关。
3. 再看 DANN，理解梯度反转和对抗式域适应。
4. 最后看 PyTorch 或 Hugging Face 文档，把 fine-tuning 跑成可复用工程流程。

## 核心结论

知识蒸馏是用一个较大的教师模型指导一个较小的学生模型训练。它的核心不是让学生只记住教师的最高分答案，而是让学生同时学习两类信息：真标签给出的正确答案，以及教师模型输出分布中隐含的类别关系。

例如教师模型对一张图片的判断是“猫 93%，狗 5%，狐狸 2%”。如果只用硬标签训练，学生只知道答案是“猫”。如果用蒸馏训练，学生还能知道“狗比狐狸更像猫”。这类相对关系就是软标签真正传递的信息。

软标签是指模型输出的概率分布，而不是只有一个类别为 1、其他类别为 0 的 one-hot 标签。温度 `T` 是控制 softmax 平滑程度的超参数：`T` 越高，分布越平滑，非最高分类别的概率越明显；`T=1` 只是普通 softmax，不等于硬标签。硬标签来自真实标注，软标签来自教师模型输出。

训练时常用损失函数是：

$$
L = \alpha \cdot CE(y, p_s^1) + (1 - \alpha) \cdot T^2 \cdot KL(p_t^T || p_s^T)
$$

其中，`CE` 让学生贴近真实标签，`KL` 让学生贴近教师分布，`T^2` 用来补偿高温下蒸馏梯度变小的问题。一个常见策略是温度从高到低调度：前期用较高 `T` 学类别关系，后期降低 `T` 收紧判别边界。

| 项目 | 作用 |
|---|---|
| 真标签 `CE(y, p_s^1)` | 保证学生不偏离正确答案 |
| 软标签 `KL(p_t^T || p_s^T)` | 传递类别相似性 |
| 温度 `T` | 控制分布平滑度 |
| `T^2` | 补偿高温下蒸馏梯度变小 |

---

## 问题定义与边界

知识蒸馏解决的是模型压缩问题：在算力、参数量、内存、时延受限时，把大模型的输出知识迁移到小模型，使小模型部署成本更低，同时尽量保留精度。

教师模型是训练好、能力更强的模型；学生模型是需要部署的小模型。logits 是模型在 softmax 之前输出的原始分数，它们保留了类别之间的相对强弱关系。蒸馏通常不是直接复制教师的参数，而是让学生的输出分布接近教师的输出分布。

移动端文本分类是一个典型例子。线上手机端只能运行轻量模型，因为大模型推理太慢、耗电太高。离线训练时，可以先用大模型给训练集打出软答案，再用这些软答案训练小模型。最终只把学生模型部署到手机上，推理更快，精度损失也更小。

| 维度 | 说明 |
|---|---|
| 输入 | 训练数据 + teacher logits |
| 目标 | 让学生接近 teacher 的输出结构 |
| 约束 | 学生模型更小、更快、更省算力 |
| 边界 | teacher 质量、类别数、任务难度、数据规模 |

蒸馏的边界也必须明确。第一，教师模型质量差时，学生会学习错误分布。第二，标签噪声很大时，硬标签和软标签可能相互冲突，需要重新检查数据质量。第三，类别极度不平衡时，教师可能对少数类给出过低概率，学生会进一步弱化这些类别。第四，如果学生模型容量太小，即使教师很好，学生也未必能表达教师学到的结构。

真实工程中，搜索意图识别经常用蒸馏。大模型可以离线判断“查天气”“买机票”“找餐厅”等意图之间的相似性，小模型负责在线低延迟推理。它适合蒸馏，因为线上请求量大，推理成本敏感，而且类别之间确实存在相近关系。

---

## 核心机制与推导

softmax 是把 logits 转成概率分布的函数。带温度的 softmax 定义为：

```text
p_i^T = exp(z_i / T) / Σ_j exp(z_j / T)

L = α · CE(y, p_s^1) + (1 - α) · T^2 · KL(p_t^T || p_s^T)
```

对应数学形式是：

$$
p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

当 `T` 增大时，所有 logits 都被除以更大的数，类别分数之间的差距会缩小，softmax 输出会更平坦。平坦不是说没有信息，而是 top-1 之外的类别也能获得可见概率。学生因此能看到更多类间关系。

玩具例子：教师 logits 是 `[4, 1, 0]`。`T=1` 时，输出几乎只强调第 1 类；`T=4` 时，其他两类的概率也变明显。这表示学生从“只有一个答案”的监督，变成了“答案之间有远近关系”的监督。

| 符号 | 含义 |
|---|---|
| `z_t, z_s` | 教师/学生 logits |
| `p_t^T, p_s^T` | 温度为 `T` 时的分布 |
| `y` | 真标签 one-hot |
| `α` | 硬标签权重 |
| `T^2` | 梯度尺度修正 |

KL 散度是衡量两个概率分布差异的指标。这里使用 `KL(p_t^T || p_s^T)`，意思是让学生在温度 `T` 下的分布接近教师在同一温度下的分布。硬标签交叉熵 `CE(y, p_s^1)` 则让学生在普通温度下仍然预测正确答案。

为什么蒸馏项要乘 `T^2`？直观上，温度越高，softmax 越平滑，输出概率对 logits 的变化越不敏感，蒸馏损失产生的梯度会变小。如果不做补偿，高温蒸馏项在总损失中可能变得太弱。乘 `T^2` 是 Hinton 等人在经典蒸馏设置中采用的工程约定，用来保持梯度量级更稳定。

需要注意，`T=1` 不是“退化为硬标签”。`T=1` 只是普通 softmax，教师仍然可能输出 `[0.936, 0.047, 0.017]` 这样的软分布。硬标签是 `[1, 0, 0]`。两者来源不同，含义也不同。

---

## 代码实现

下面是一个最小可运行的 Python 例子，用纯 Python 实现温度 softmax 和 KL，用断言检查高温会让分布更平滑。

```python
import math

def softmax_with_temperature(logits, T=1.0):
    scaled = [x / T for x in logits]
    max_v = max(scaled)
    exps = [math.exp(x - max_v) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]

def kl_divergence(p, q):
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0)

teacher_logits = [4.0, 1.0, 0.0]
student_logits = [3.0, 0.5, 0.0]

teacher_t1 = softmax_with_temperature(teacher_logits, T=1.0)
teacher_t4 = softmax_with_temperature(teacher_logits, T=4.0)
student_t4 = softmax_with_temperature(student_logits, T=4.0)

assert round(teacher_t1[0], 3) == 0.936
assert round(teacher_t4[0], 3) == 0.543
assert teacher_t4[1] > teacher_t1[1]
assert teacher_t4[2] > teacher_t1[2]

soft_loss = kl_divergence(teacher_t4, student_t4)
assert soft_loss >= 0
assert round(soft_loss, 3) == 0.005

print("teacher T=1:", [round(x, 3) for x in teacher_t1])
print("teacher T=4:", [round(x, 3) for x in teacher_t4])
print("KL:", round(soft_loss, 3))
```

在 PyTorch 中，蒸馏损失通常写成下面的形式：

```python
import torch
import torch.nn.functional as F

def distill_loss(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    hard_loss = F.cross_entropy(student_logits, labels)

    student_log_prob = F.log_softmax(student_logits / T, dim=-1)
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)

    soft_loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    ) * (T * T)

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

这里要把 teacher 当成只负责打分的裁判，student 当成正在学习的选手。裁判不参与更新，选手同时学习标准答案和裁判偏好。训练循环中，教师模型应使用 `eval()`，并放在 `torch.no_grad()` 下计算 logits。

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

for batch in train_loader:
    inputs, labels = batch

    with torch.no_grad():
        teacher_logits = teacher(inputs)

    student_logits = student(inputs)
    loss = distill_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        alpha=0.5,
        T=4.0,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

| 项目 | 要求 |
|---|---|
| teacher 模式 | `eval()` |
| teacher 参数 | `requires_grad=False` |
| student 输入 | 原始 logits |
| 蒸馏输入 | `log_softmax / softmax` |
| 归一化 | `batchmean` 更稳定 |

`KLDivLoss` 的输入格式是常见错误点。PyTorch 的 `F.kl_div(input, target)` 默认要求 `input` 是 log-probability，所以学生端应传 `F.log_softmax(student_logits / T)`；教师端通常传 `F.softmax(teacher_logits / T)`。如果直接把 raw logits 传进去，数值含义就错了。

---

## 工程权衡与常见坑

`T` 和 `α` 不是固定常量，需要用验证集调参。`T` 控制软标签信息密度，`α` 控制硬标签和软标签的权重。`α` 越大，越相信真实标签；`α` 越小，越相信教师分布。

如果 `T` 太高，所有类别概率都差不多，教师提示会变得像没重点的说明书；如果 `T` 太低，软分布接近普通尖锐分布，类别关系传递不足。如果 `α` 太小，学生会过度模仿教师，教师一旦犯错，学生也跟着错。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 直接对 raw logits 算 KL | 数值不正确 | 先做 `log_softmax / softmax` |
| 忘记乘 `T^2` | 蒸馏项太弱 | 统一在 soft loss 外乘 `T^2` |
| `T` 过高 | 信息被抹平 | 用验证集找中等温度 |
| `α` 过小 | 过拟合 teacher 错误 | 保留足够硬标签权重 |
| teacher 不冻结 | 目标漂移 | `eval()` + 禁止梯度 |
| 只看训练集 | 过拟合调参 | 必须看验证集 |

温度调度是一种常用训练技巧。前期学生还没有形成稳定边界，可以用较高温度，让它先学习类别之间的相对关系。中期保持蒸馏项和硬标签项平衡。后期降低温度，让输出分布更尖锐，帮助学生收紧决策边界。

| 阶段 | 建议 |
|---|---|
| 前期 | 较高 `T`，强化类间关系 |
| 中期 | 稳定蒸馏与硬标签平衡 |
| 后期 | 降低 `T`，收紧决策边界 |

真实工程例子：搜索意图识别中，类别可能包括“查天气”“查航班”“订酒店”“查景点”。大教师模型可能知道“查航班”和“订酒店”都接近旅行意图，而“查天气”在某些查询中也可能与旅行相关。学生模型如果只学硬标签，会把这些类别完全割裂；如果学教师软分布，就能保留一部分语义结构。上线时，学生模型推理快，适合高 QPS 场景。

常见调参起点可以设为 `T=2、4、8`，`α=0.3、0.5、0.7` 做网格搜索。不要只看训练损失，因为蒸馏损失下降不一定代表泛化更好。应该以验证集准确率、F1、校准误差或业务指标为准。

---

## 替代方案与适用边界

标准 KD 适合有强教师、学生需要压缩部署、任务输出空间清晰的场景。它实现简单，主要改动集中在损失函数。但如果任务更强调中间表示、模型校准、多教师融合或持续训练，就需要考虑其他方案。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 标准 KD | 常规压缩 | 简单、稳定 | 依赖 teacher 质量 |
| 温度调度 KD | 需要先宽后窄 | 兼顾关系学习和边界收敛 | 训练策略更复杂 |
| 特征蒸馏 | 中间表征重要 | 信息更丰富 | 实现复杂 |
| 在线蒸馏 | 多模型协同 | 可持续提升 | 训练成本高 |

特征蒸馏是让学生不仅学习教师最终输出，还学习教师中间层表示。中间层表示是模型内部产生的向量特征，通常包含更细的语义信息。它适合视觉模型、语音模型、检索模型等表示质量很重要的任务，但需要处理教师和学生层数、维度不一致的问题。

在线蒸馏是多个模型在训练过程中互相学习，而不是先训练好一个固定教师。它适合训练资源较充足、希望多模型协同提升的场景，但工程复杂度更高，训练稳定性也更难控制。

如果你要做搜索意图识别，普通蒸馏适合“先学大概关系再收紧边界”。如果数据很少、标签噪声又大，单纯模仿教师可能不如先做数据清洗、重采样、标签审计或更稳的正则化。蒸馏不是替代数据质量的工具，它只能把教师已有的判断结构传给学生。

温度调度 KD 的适用边界是：类别之间确实存在可学习的相似关系，并且学生容量足以吸收这些关系。如果类别彼此完全独立，或者教师输出本身过度自信，高温也未必能提供有效信息。此时更重要的是改善教师校准、扩充数据或调整任务定义。

---

## 参考资料

| 类型 | 资料 |
|---|---|
| 原始论文 | Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* |
| 官方文档 | PyTorch `torch.nn.KLDivLoss` |
| 变体研究 | Li et al. *Curriculum Temperature for Knowledge Distillation* |
| 参考实现 | `CTKD` GitHub 源码 |

如果只看论文，容易只记住公式；如果再看官方文档和开源实现，就能知道公式在代码里怎么落地，尤其是 `KLDivLoss` 的输入格式、`batchmean` 归一化和 `T^2` 修正这些容易写错的细节。本文中的基本损失形式来自原始蒸馏论文，PyTorch 写法来自官方 `KLDivLoss` 文档，温度调度思想可参考课程温度蒸馏相关研究和实现。

1. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. [PyTorch torch.nn.KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
3. [Curriculum Temperature for Knowledge Distillation](https://arxiv.org/abs/2211.16231)
4. [CTKD GitHub Repository](https://github.com/zhengli97/CTKD)

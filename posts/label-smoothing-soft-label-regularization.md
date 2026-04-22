## 核心结论

Label Smoothing 是一种软标签正则化方法：把分类任务里的 one-hot 硬标签，替换成一个保留主类、同时给其他类别少量概率的软目标分布。

硬标签是指“正确类别为 1，其他类别为 0”的标签形式。软标签是指“每个类别都有一个概率值，总和为 1”的标签形式。

核心公式是：

$$
q_c = (1 - \varepsilon) \cdot 1[c = y] + \frac{\varepsilon}{K}
$$

其中，$K$ 是类别数，$y$ 是真实类别，$\varepsilon$ 是平滑系数，通常取 `0.1`。训练损失仍然是交叉熵：

$$
L = -\sum_{c=1}^{K} q_c \log p_c
$$

这里的 $p_c$ 是模型对第 $c$ 类的预测概率。

Label Smoothing 的目的不是改变类别真相，而是降低模型对单一类别的过度自信。模型仍然要把正确类预测得最高，但不再被训练目标逼着把正确类概率推到 1、把其他类概率压到 0。

新手版玩具例子：四分类任务中，真实类别是第 2 类。硬标签是 `[0, 1, 0, 0]`。当 `ε = 0.1` 时，软标签变成 `[0.025, 0.925, 0.025, 0.025]`。含义是：第 2 类仍然最正确，但其他类别不再被当成绝对不可能。

| 对比项 | 硬标签 | Label Smoothing 软标签 |
|---|---:|---:|
| 第 1 类 | 0 | 0.025 |
| 第 2 类 | 1 | 0.925 |
| 第 3 类 | 0 | 0.025 |
| 第 4 类 | 0 | 0.025 |

| 训练目标 | 普通交叉熵 | Label Smoothing |
|---|---|---|
| 正确类目标 | 尽量接近 1 | 接近 $1-\varepsilon+\varepsilon/K$ |
| 非正确类目标 | 尽量接近 0 | 接近 $\varepsilon/K$ |
| 主要效果 | 分类边界更尖锐 | 输出分布更平滑 |
| 常见收益 | 训练集拟合更强 | 泛化和校准更稳 |

---

## 问题定义与边界

分类模型通常输出 logits。logits 是 softmax 之前的原始分数，数值越大，softmax 后对应类别概率越高。普通分类训练会把 one-hot 标签作为监督目标：

$$
y_c = 1[c = y]
$$

如果真实类别是第 2 类，四分类 one-hot 标签就是：

```text
[0, 1, 0, 0]
```

这会让交叉熵持续鼓励模型把第 2 类概率推高，把其他类别概率压低。训练久了以后，模型可能在训练集上损失很低，但在测试集上输出过度自信的概率。例如模型预测错误时仍然给出 `0.99` 的置信度，这会让线上阈值策略、人工复核策略和风险控制策略变得不稳定。

Label Smoothing 解决的是训练目标过硬的问题。它把目标改成：

$$
q_c = (1 - \varepsilon) \cdot 1[c = y] + \frac{\varepsilon}{K}
$$

在 `K = 4, ε = 0.1, y = 2` 时：

```text
[0, 1, 0, 0] -> [0.025, 0.925, 0.025, 0.025]
```

它不是推理期阈值规则。推理期阈值规则是指模型训练完成后，根据预测概率决定是否自动通过、拒识或转人工。Label Smoothing 发生在训练期，直接改变损失函数看到的目标分布。

它也不是数据增强。数据增强是改输入样本，例如裁剪图片、加噪声、随机翻转。Label Smoothing 不改输入，只改监督目标。

它也不是标签噪声修复。标签噪声是指数据标注本身错误。Label Smoothing 不能识别哪条样本标错了，只是降低模型把所有标签都当成绝对真理的强度。

| 场景 | 是否适合 Label Smoothing | 原因 |
|---|---|---|
| 单标签图像分类 | 适合 | one-hot 目标常导致过度自信 |
| 文本分类 | 通常适合 | 可改善泛化和置信度 |
| 线上按置信度决策 | 适合 | 预测概率更容易校准 |
| 标签已经是概率分布 | 谨慎 | 可能二次平滑 |
| 多标签分类 | 不能直接照搬 | 每个类别是独立二分类，不是单一 softmax |
| 少数类召回不足 | 不是首选 | 类不平衡应优先考虑重加权或采样 |

| 方法 | 标签形式 | 主要来源 | 典型用途 |
|---|---|---|---|
| 硬标签 | `[0, 1, 0, 0]` | 人工标注 | 标准分类训练 |
| Label Smoothing 软标签 | `[0.025, 0.925, 0.025, 0.025]` | 人工标签加平滑规则 | 正则化和校准 |
| 蒸馏软标签 | 如 `[0.05, 0.80, 0.10, 0.05]` | teacher 模型输出 | 迁移类别间相似性 |

---

## 核心机制与推导

softmax 是把 logits 转成概率分布的函数：

$$
p_c = softmax(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}
$$

普通交叉熵使用 one-hot 标签时，损失为：

$$
L = -\log p_y
$$

因为只有真实类别那一项的标签为 1，其他类别标签为 0。这个目标会持续推动 $p_y$ 接近 1。要让 $p_y$ 接近 1，模型通常会把真实类 logit 拉得很大，把其他类 logit 压得很小。

Label Smoothing 后，损失变成：

$$
L = -\sum_{c=1}^{K} q_c \log p_c
$$

展开后可以看到两个变化：

$$
L = -q_y \log p_y - \sum_{c \ne y} q_c \log p_c
$$

其中：

$$
q_y = 1 - \varepsilon + \frac{\varepsilon}{K}
$$

$$
q_c = \frac{\varepsilon}{K}, c \ne y
$$

这意味着真类项权重变小，非真类项不再为 0。模型仍然会提高真实类别概率，但不会被鼓励把其他类别概率压到绝对接近 0。

对 softmax 交叉熵来说，logits 的梯度可以写成：

$$
\frac{\partial L}{\partial z_c} = p_c - q_c
$$

普通 one-hot 下，真实类梯度是 $p_y - 1$，非真实类梯度是 $p_c - 0$。Label Smoothing 下，真实类梯度变成 $p_y - q_y$，非真实类梯度变成 $p_c - \varepsilon/K$。这会让梯度方向更温和，减少模型把 logits 拉到极端值的动力。

还是四分类玩具例子，真实类别为第 2 类：

| ε | 目标分布 |
|---:|---|
| 0 | `[0, 1, 0, 0]` |
| 0.05 | `[0.0125, 0.9625, 0.0125, 0.0125]` |
| 0.1 | `[0.025, 0.925, 0.025, 0.025]` |
| 0.2 | `[0.05, 0.85, 0.05, 0.05]` |

| 类别 | 普通交叉熵目标 | Label Smoothing 目标 | 梯度变化 |
|---|---:|---:|---|
| 真实类 | 1 | 小于 1 | 减少继续推高真实类 logit 的压力 |
| 非真实类 | 0 | 大于 0 | 减少继续压低非真实类 logit 的压力 |
| 整体分布 | 极尖锐 | 更平滑 | 降低过度自信 |

从正则化角度看，Label Smoothing 对输出分布施加了“不要太尖锐”的约束。正则化是指训练时加入限制，降低模型只记住训练集细节的倾向。它不保证一定提升 accuracy，但经常改善泛化和校准。

校准是指预测概率和真实正确率之间的一致性。例如模型对 1000 个样本都预测 `0.8` 置信度，如果其中约 800 个真的预测正确，这个模型就比较校准。

---

## 代码实现

下面是一个可运行的手写 Python 版本，只依赖标准库。它构造 Label Smoothing 目标分布，并计算交叉熵。

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]

def smooth_label(num_classes, target, epsilon):
    return [
        (1 - epsilon if i == target else 0.0) + epsilon / num_classes
        for i in range(num_classes)
    ]

def cross_entropy_with_soft_target(logits, target, epsilon):
    p = softmax(logits)
    q = smooth_label(len(logits), target, epsilon)
    loss = -sum(q_i * math.log(p_i) for q_i, p_i in zip(q, p))
    return loss, q, p

loss, q, p = cross_entropy_with_soft_target(
    logits=[0.2, 2.0, -0.5, 0.1],
    target=1,
    epsilon=0.1,
)

assert q == [0.025, 0.925, 0.025, 0.025]
assert abs(sum(q) - 1.0) < 1e-12
assert loss > 0
assert p[1] == max(p)
```

手写公式对应的是：

$$
L = -\sum_{c=1}^{K} q_c \log softmax(z)_c
$$

PyTorch 中可以直接使用 `CrossEntropyLoss(label_smoothing=...)`。注意：PyTorch 的 `CrossEntropyLoss` 接收的是 logits 和类别索引，不需要先手动 softmax。

```python
import torch
import torch.nn as nn

logits = torch.tensor([[0.2, 2.0, -0.5, 0.1]], dtype=torch.float32)
target = torch.tensor([1], dtype=torch.long)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, target)

assert loss.item() > 0
```

TensorFlow / Keras 中，`CategoricalCrossentropy` 支持 `label_smoothing`。它通常接收 one-hot 标签。如果输入是 logits，需要设置 `from_logits=True`。

```python
import tensorflow as tf

logits = tf.constant([[0.2, 2.0, -0.5, 0.1]], dtype=tf.float32)
target = tf.constant([[0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)

loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1,
)
loss = loss_fn(target, logits)

assert float(loss.numpy()) > 0
```

实现时要确认平滑定义。本文使用的是把 `ε` 分给所有类别的形式：

$$
q_c = (1-\varepsilon) \cdot 1[c=y] + \varepsilon/K
$$

还有一种常见写法是只把 `ε` 分给非真实类：

$$
q_y = 1-\varepsilon,\quad q_c = \frac{\varepsilon}{K-1}, c \ne y
$$

两者数值不同，不能混着比较实验结果。

| 定义 | 真类概率 | 非真类概率 | 特点 |
|---|---:|---:|---|
| `ε/K` | $1-\varepsilon+\varepsilon/K$ | $\varepsilon/K$ | 很多深度学习框架采用 |
| `ε/(K-1)` | $1-\varepsilon$ | $\varepsilon/(K-1)$ | 更直观地把平滑量只分给错误类 |

| 框架 | 常用 API | 输入标签 | 注意点 |
|---|---|---|---|
| PyTorch | `nn.CrossEntropyLoss(label_smoothing=ε)` | 类别索引或概率目标 | 通常传 logits，不要先 softmax |
| TensorFlow | `tf.keras.losses.CategoricalCrossentropy(label_smoothing=ε)` | one-hot 或概率分布 | logits 输入要设置 `from_logits=True` |
| 手写实现 | `log_softmax + soft target` | 自己构造 `q` | 要处理 padding、mask、ignore 样本 |

真实工程例子：商品图分类系统有 5000 个类，线上用模型置信度决定“自动归类”还是“人工复核”。普通训练后，模型可能对相似商品也输出 `0.99`，比如把“男款跑鞋”和“女款跑鞋”混淆时仍然极度自信。加入 `ε = 0.1` 后，top-1 类别通常仍然不变，但置信度分布会更保守，阈值策略更容易稳定运行。

---

## 工程权衡与常见坑

Label Smoothing 不是越大越好。`ε` 太小，效果不明显；`ε` 太大，真实类别的监督信号会被削弱，模型可能欠拟合。欠拟合是指模型连训练集规律都学不充分，表现为训练集和验证集效果都不理想。

经验上，`ε = 0.1` 是常见起点，但不是固定答案。ImageNet 分类中的 Inception-v3 使用 Label Smoothing 作为正则项，并报告了精度收益。具体任务仍然要通过验证集选择。

不要只看 accuracy。accuracy 是预测类别是否正确的比例，但它不反映概率是否可信。Label Smoothing 的收益经常体现在校准指标上。

常见指标包括：

$$
NLL = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
$$

NLL 是负对数似然，预测错且很自信时惩罚很大。

$$
Brier = \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{K}(p_{i,c}-y_{i,c})^2
$$

Brier score 衡量预测概率分布和真实标签分布之间的平方误差。

ECE 是 Expected Calibration Error，中文可理解为“期望校准误差”。它把预测按置信度分桶，比较每个桶里的平均置信度和真实准确率差异：

$$
ECE = \sum_{m=1}^{M}\frac{|B_m|}{N}|acc(B_m)-conf(B_m)|
$$

| 指标 | 看什么 | 为什么重要 |
|---|---|---|
| Accuracy | 类别是否预测正确 | 衡量分类结果 |
| NLL | 错误且自信的代价 | 衡量概率质量 |
| ECE | 置信度和准确率是否一致 | 衡量校准 |
| Brier score | 概率分布整体误差 | 衡量概率预测质量 |

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| `ε` 设得过大 | 真类学习变弱，top-1 下降 | 从 `0.05` 或 `0.1` 开始调参 |
| 忽略 API 定义差异 | 实验不可比 | 明确是 `ε/K` 还是 `ε/(K-1)` |
| 对 padding token 平滑 | 损失污染 | 使用 `ignore_index` 或 mask |
| 对 soft label 再平滑 | 信息被稀释 | 蒸馏数据和人工软标签单独处理 |
| 只看 accuracy | 忽略概率质量 | 同时看 NLL、ECE、Brier score |
| teacher 训练时加 LS | student 蒸馏收益可能下降 | 蒸馏前单独评估 teacher 输出分布 |

知识蒸馏场景要特别谨慎。知识蒸馏是指用大模型 teacher 的输出概率训练小模型 student。teacher 的软输出本来就包含类别相似性，例如“猫”和“狗”比“猫”和“汽车”更接近。如果 teacher 训练时使用过强 Label Smoothing，输出分布可能变得不够有信息量，student 学到的类别关系会变弱。

---

## 替代方案与适用边界

Label Smoothing 适合主要问题是“模型过度自信”的单标签分类任务。它不是所有训练问题的默认解法。

如果目标是让概率更可信，可以考虑 Label Smoothing 或温度缩放。温度缩放是训练后校准方法，不改模型参数主体，只在 logits 上除以温度 $T$：

$$
p = softmax(z / T)
$$

当 $T > 1$ 时，输出概率会变平滑；当 $T < 1$ 时，输出概率会更尖锐。它通常在验证集上拟合一个温度参数，用来改善校准。

如果目标是防止过拟合，可以考虑 dropout、weight decay、数据增强。dropout 是训练时随机关闭部分神经元，减少模型对局部特征的依赖。weight decay 是对参数大小加惩罚，限制模型复杂度。

如果目标是处理类别不平衡，Label Smoothing 通常不是第一选择。类别不平衡是指少数类样本远少于多数类样本。此时更直接的方法是重加权、重采样或 focal loss。focal loss 会降低易分类样本的损失权重，让模型更关注难样本：

$$
FL = -(1-p_t)^\gamma \log p_t
$$

| 方法 | 发生阶段 | 主要目标 | 适用问题 |
|---|---|---|---|
| Label Smoothing | 训练期 | 降低过度自信 | 单标签分类、校准改善 |
| 温度缩放 | 训练后 | 校准概率 | 已训练模型概率偏尖锐 |
| Dropout | 训练期 | 降低过拟合 | 模型容量大、数据较少 |
| Weight decay | 训练期 | 限制参数规模 | 通用正则化 |
| Focal loss | 训练期 | 关注难样本 | 类不平衡、检测任务 |

| 任务类型 | 推荐方案 | 说明 |
|---|---|---|
| 普通图像分类 | Label Smoothing + 数据增强 | 常见稳定组合 |
| 文本单标签分类 | 小 ε Label Smoothing | 注意不要削弱细粒度类别差异 |
| 多标签分类 | 独立二分类平滑或不用 | 不能直接套 softmax 单标签公式 |
| 类别极不平衡 | 重加权或 focal loss | 优先解决样本权重问题 |
| 蒸馏训练 | teacher soft label | 避免盲目二次平滑 |
| 只需后处理校准 | 温度缩放 | 不需要重新训练主模型 |

强依赖精确概率的场景要谨慎使用。例如医疗风险评分、金融违约预测、广告出价系统，概率值本身会进入决策公式。Label Smoothing 可能改善校准，也可能改变概率分布形态。上线前必须用独立验证集检查 NLL、ECE、Brier score，并确认阈值策略的业务影响。

新手版判断方法：如果你的目标是“模型不要动不动输出 0.99”，Label Smoothing 值得尝试。如果你的目标是“少数类召回更高”，它通常不是第一选择。如果你的标签已经是 teacher 模型给出的软分布，就不要默认再平滑一次。

---

## 参考资料

| 文献 | 作用 | 对应章节 |
|---|---|---|
| Rethinking the Inception Architecture for Computer Vision | 提出并应用 Label Smoothing，报告 ImageNet 收益 | 核心结论、工程权衡 |
| When Does Label Smoothing Help? | 分析 Label Smoothing 何时有效及对蒸馏的影响 | 核心机制、常见坑 |
| PyTorch `CrossEntropyLoss` docs | 说明 PyTorch 内置实现 | 代码实现 |
| TensorFlow `CategoricalCrossentropy` docs | 说明 TensorFlow / Keras 内置实现 | 代码实现 |

1. [Rethinking the Inception Architecture for Computer Vision](https://discovery.ucl.ac.uk/id/eprint/1503253/11/Wojna_1512.00567.pdf)
2. [When Does Label Smoothing Help?](https://papers.nips.cc/paper/8717-when-does-label-smoothing-help)
3. [PyTorch `CrossEntropyLoss` docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
4. [TensorFlow `CategoricalCrossentropy` docs](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)

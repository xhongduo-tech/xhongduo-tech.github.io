## 核心结论

损失函数是训练目标的数学表达，白话讲，就是模型每次预测后系统给出的“扣分规则”。分类、回归、排序三类任务虽然都在最小化一个数，但这个数衡量的对象完全不同：

| 任务类型 | 预测对象 | 常用损失 | 适合场景 | 主要优点 | 主要风险 |
|---|---|---|---|---|---|
| 分类 | 离散类别概率 | 交叉熵、Focal Loss、Label Smoothing | 图像分类、文本分类、检测分类头 | 概率解释清晰，优化成熟 | 类别不平衡、过度自信 |
| 回归 | 连续数值 | MSE、MAE、Huber Loss | 房价预测、CTR 预估、传感器标定 | 目标直接，易解释 | 对异常值敏感或梯度不稳定 |
| 排序 | 相对距离或先后关系 | Triplet Loss、Contrastive Loss | 检索、人脸识别、召回排序 | 直接优化相似性结构 | 采样难，易出现无效样本 |

最重要的选择规则只有两层：

1. 先按任务类型定大类。
2. 再按数据分布和优化难度选具体损失。

分类里，交叉熵是默认起点；正负样本极不平衡时优先考虑 Focal Loss；模型过度自信、概率校准差时考虑 Label Smoothing。回归里，噪声接近高斯分布可先用 MSE；离群点多时优先 MAE 或 Huber；Huber 是二者之间的折中。排序里，Triplet 和 Contrastive 都依赖“正样本更近、负样本更远”的约束，真正决定效果的往往不是公式本身，而是样本构造和 hard mining。

---

## 问题定义与边界

分类任务的目标是预测类别，输出通常是一个概率向量，例如三分类输出 $[0.7, 0.2, 0.1]$。回归任务的目标是预测一个连续值，例如温度、价格、时延。排序任务不关心绝对值，而关心相对关系，例如“相似图片应该更近，不相似图片应该更远”。

这三类任务的边界要分清：

- 如果标签是离散类，哪怕最后只输出一个编号，本质仍是分类。
- 如果标签是连续值，即使最后会被分桶展示，本质仍是回归。
- 如果监督信号是“谁比谁更像”或“谁排前面”，本质是排序或度量学习。

一个玩具例子可以快速说明差别：

- 分类：判断邮件是不是垃圾邮件。
- 回归：预测明天的 CPU 峰值负载是多少。
- 排序：给搜索结果按相关性重新排序。

同一个业务甚至可能同时用到三类损失。以电商搜索为例，召回阶段常用排序损失训练向量表示，粗排阶段可能做点击率回归，精排阶段还可能加入转化分类头。

分类任务最常见的边界问题是类别不平衡。比如目标检测里，大量 anchor 都是背景，若直接用普通交叉熵，模型很容易被“背景样本”主导。回归任务的边界问题是异常值。若少量坏数据误差极大，MSE 会被这些点拉着走。排序任务的边界问题是嵌入坍塌，白话讲，就是所有样本向量挤在一起，距离失去区分意义。

三分类例子里，若真实类别是第 0 类，预测是 $[0.7, 0.2, 0.1]$，交叉熵为：

$$
L=-\log 0.7 \approx 0.357
$$

如果模型把真实类概率压到 0.2，损失就会变成 $-\log 0.2 \approx 1.609$，惩罚显著增大。这说明交叉熵关心的不是“猜中没猜中”，而是“给正确类分了多大概率”。

---

## 核心机制与推导

### 1. 分类损失

交叉熵是概率分布差异的度量，白话讲，就是目标分布和预测分布越不一致，扣分越高。多分类交叉熵写成：

$$
L_{CE}=-\sum_{i=1}^{K} y_i \log p_i
$$

其中 $y_i$ 是目标分布，$p_i$ 是模型预测概率。对 one-hot 标签，只有真实类那一项保留下来，所以本质就是 $-\log p_t$，$p_t$ 表示真实类概率。

Focal Loss 在交叉熵前加了一个难样本权重：

$$
L_{FL}=-\alpha_t(1-p_t)^\gamma \log p_t
$$

这里 $\gamma$ 叫聚焦参数，白话讲，就是让“容易样本少管一点、困难样本多管一点”。当 $p_t$ 很大时，$(1-p_t)^\gamma$ 很小，说明样本已经学会，不必继续占用太多梯度。比如 $p_t=0.7,\gamma=2$，权重是 $(1-0.7)^2=0.09$，主项明显缩小。

Label Smoothing 把硬标签从 1 和 0 改成软标签，白话讲，就是别让模型把某一类当成绝对真理。常见形式是：

$$
y_k^{smooth}=(1-\varepsilon)\mathbf{1}_{k=t}+\frac{\varepsilon}{K}
$$

若三分类、$\varepsilon=0.1$，真实类软标签就是 $0.9333$，非真实类各是 $0.0333$。一些教材也用“把 0.1 分到其余类别”的写法，得到近似的 $0.8/0.1/0.1$ 示例，本质都是在削弱过硬标签。

### 2. 回归损失

MSE 是均方误差，白话讲，就是误差先平方再求平均：

$$
L_{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

平方的效果是大误差被放大，所以它对离群点特别敏感，但梯度平滑，优化稳定。

MAE 是平均绝对误差：

$$
L_{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

它对异常值更稳，因为误差不会被平方放大，但在 0 点附近不可导，实际训练中常表现为收敛较慢。

Huber Loss 是分段函数，可以理解成“误差小时像 MSE，误差大时像 MAE”：

$$
L_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le \delta \\
\delta(|r|-\frac{1}{2}\delta), & |r|>\delta
\end{cases}
$$

其中 $r=y-\hat{y}$。若 $\delta=1,r=-1$，则损失为 $0.5$。它的价值是把平滑优化和异常值鲁棒性拼在一起。

### 3. 排序损失

Triplet Loss 用三元组训练：anchor、positive、negative。白话讲，就是让锚点更靠近同类样本，远离异类样本。

$$
L_{triplet}=\max(0, d(a,p)-d(a,n)+m)
$$

$m$ 是 margin，白话讲，就是至少要留出的安全间隔。若 $d(a,p)=0.3,d(a,n)=0.8,m=0.5$，则损失为 0，说明间隔刚好满足要求。

Contrastive Loss 通常用于样本对：

$$
L=(1-Y)\frac{1}{2}D^2 + Y\frac{1}{2}\max(0,m-D)^2
$$

这里 $D$ 是两个嵌入的距离，$Y$ 表示是否为负对。它本质也是“相似的拉近，不相似的推远”。

真实工程例子是人脸识别。系统不直接记住“张三是谁”，而是把每张脸映射为一个向量。训练目标不是单纯分类，而是让同一个人的向量更近，不同人的向量更远。这样新用户即使没见过，也能靠距离做识别或检索。

---

## 代码实现

下面的代码只用 `python` 标准库实现核心逻辑，便于理解公式和检查边界。

```python
import math

def cross_entropy(probs, target_idx, eps=1e-12):
    p = max(min(probs[target_idx], 1 - eps), eps)
    return -math.log(p)

def focal_loss(probs, target_idx, alpha=1.0, gamma=2.0, eps=1e-12):
    p_t = max(min(probs[target_idx], 1 - eps), eps)
    return -alpha * ((1 - p_t) ** gamma) * math.log(p_t)

def label_smoothing_target(num_classes, target_idx, epsilon=0.1):
    smooth = [epsilon / num_classes] * num_classes
    smooth[target_idx] += 1.0 - epsilon
    return smooth

def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

def mae(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def huber(y_true, y_pred, delta=1.0):
    assert len(y_true) == len(y_pred)
    total = 0.0
    for a, b in zip(y_true, y_pred):
        r = a - b
        if abs(r) <= delta:
            total += 0.5 * r * r
        else:
            total += delta * (abs(r) - 0.5 * delta)
    return total / len(y_true)

def euclidean(x, y):
    assert len(x) == len(y)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def triplet_loss(anchor, positive, negative, margin=0.5):
    d_ap = euclidean(anchor, positive)
    d_an = euclidean(anchor, negative)
    return max(0.0, d_ap - d_an + margin)

def contrastive_loss(x1, x2, is_negative, margin=1.0):
    d = euclidean(x1, x2)
    if is_negative:
        return 0.5 * max(0.0, margin - d) ** 2
    return 0.5 * d ** 2

# 玩具断言
assert round(cross_entropy([0.7, 0.2, 0.1], 0), 3) == 0.357
assert round(mse([5], [4]), 3) == 1.000
assert round(mae([5], [4]), 3) == 1.000
assert round(huber([5], [4], delta=1.0), 3) == 0.500
assert triplet_loss([0, 0], [0.3, 0], [0.8, 0], margin=0.5) == 0.0
```

实现时有几个细节不能省：

- 分类损失要做数值稳定处理，避免 $\log(0)$。
- Label Smoothing 先构造软标签，再与预测概率做交叉熵。
- Huber 必须按残差分段。
- 排序损失真正难的是采样，不是这几行公式。

如果在 PyTorch 中写 Focal Loss，核心仍然是先取 `softmax` 概率，再抽出真实类概率 `p_t`，然后乘上 $(1-p_t)^\gamma$。公式不变，只是张量化实现更高效。

---

## 工程权衡与常见坑

| 问题 | 影响 | 缓解方案 |
|---|---|---|
| 类别极不平衡还用普通交叉熵 | 背景样本主导训练，前景召回低 | 改用 Focal，或配合重采样/OHEM |
| Focal 的 $\gamma$ 太大 | 大量样本梯度接近 0，训练变慢甚至学不动 | 从 1 到 2 小范围试验 |
| Label Smoothing 的 $\varepsilon$ 太高 | 决策边界被抹平，最终精度下降 | 常从 0.05 或 0.1 起试 |
| MSE 直接用于重尾噪声 | 少数异常点支配梯度 | 换 MAE/Huber，先做剪枝或归一化 |
| MAE 单独使用 | 梯度信息弱，早期收敛慢 | 用 Huber 折中，或先 MSE 后切换 |
| Triplet/Contrastive 不做 hard mining | 很快 loss=0，向量没有区分度 | 做 batch-hard、semi-hard 采样 |

真实工程里，RetinaNet 是 Focal Loss 的代表例子。它面对的是密集检测场景，绝大多数候选框都是背景。若不压低简单背景样本的权重，模型会把训练资源浪费在“已经会了的负样本”上。Focal Loss 的核心价值不是让公式更复杂，而是重新分配梯度预算。

回归里也有类似现象。金融时间序列或工业传感器数据里，经常混有尖峰噪声。若直接用 MSE，少数尖峰会迫使模型去拟合异常点，导致整体预测偏移。这时常见做法是切到 Huber，或者在数据侧增加截断和稳健归一化。

排序任务最容易被低估的坑是“样本组织”。Triplet Loss 看起来简单，但如果 negative 太容易，模型几步就把 loss 压到 0，训练表面稳定，实际向量空间没有形成有效结构。所以很多项目里真正花时间的不是改损失，而是设计 batch 采样器。

---

## 替代方案与适用边界

| 当前 loss | 典型问题 | 备选 loss | 适用条件 |
|---|---|---|---|
| 交叉熵 | 类别极不平衡 | Focal Loss | 前景少、背景多 |
| 交叉熵 | 概率过度自信 | Label Smoothing | 需要更好校准 |
| MSE | 被异常值主导 | MAE / Huber | 重尾噪声、坏点较多 |
| MAE | 收敛慢 | Huber | 既要鲁棒又要平滑梯度 |
| Triplet | 样本构造复杂 | Contrastive / InfoNCE | 有成对数据或大 batch |
| Triplet | 类别判别不够强 | ArcFace | 封闭集身份识别 |

几个边界要明确：

- Label Smoothing 不能替代 Focal。前者处理的是标签分布过硬，后者处理的是样本难度不均。
- Huber 不是万能回归损失。若数据几乎无异常值，MSE 往往更直接。
- Triplet 和 Contrastive 依赖明确的正负关系。如果标注里没有可靠的相似性定义，强行用排序损失通常效果不稳。
- ArcFace、InfoNCE 等方法常优于基础 Triplet，但它们通常需要更明确的 batch 组织、归一化策略或类别结构。

一个真实替代例子是人脸识别。早期方案常从 Triplet Loss 起步，因为概念直接。但当类别很多、训练集规模变大时，ArcFace 这类角度间隔损失通常更容易训练，也更适合封闭集身份分类。另一个例子是金融预测：若 MSE 被少数尖峰主导，可改成 MAE 或 Huber，再配合数据剪枝，否则模型会持续追逐异常样本。

---

## 参考资料

- Cross Entropy、Label Smoothing、Metric Learning 基础公式与概念：GeeksforGeeks 相关文章
- Focal Loss 原始论文：Lin et al., *Focal Loss for Dense Object Detection*, 2017
- RetinaNet 与 Focal Loss 的工程背景解析：RetinaNet 论文解读博客与 Hugging Face 论文页
- MSE、MAE、Huber 的性质与鲁棒性讨论：IBM、Britannica、Deepchecks 等资料
- Triplet Loss、Contrastive Loss 的入门说明与应用背景：GeeksforGeeks Metric Learning 条目

参考内容覆盖了本文涉及的核心公式、典型应用场景、参数含义与常见工程陷阱，足以支撑分类、回归、排序三类损失的入门选型。

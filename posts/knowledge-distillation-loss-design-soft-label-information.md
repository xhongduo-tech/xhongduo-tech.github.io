## 核心结论

知识蒸馏的核心不是“把正确答案再教一遍”，而是把教师模型对所有类别的偏好结构一起交给学生模型。软标签指经过温度 $T$ 平滑后的概率分布，它比 one-hot 硬标签包含更多类间关系信息。所谓“暗知识”，就是这些非最大概率项里编码的相似度结构，例如猫和狐狸都不像狗，但猫比飞机更像狐狸。

经典蒸馏损失写成：

$$
L=(1-\alpha)L_{\mathrm{CE}}(y,p^S)+\alpha T^2 D_{\mathrm{KL}}(q^T \| p^S)
$$

其中：

- $L_{\mathrm{CE}}$ 是硬标签交叉熵，作用是保证学生别偏离真实标签。
- $D_{\mathrm{KL}}$ 是 KL 散度，白话说是“比较两个概率分布差多少”的量，用来逼近教师的软标签分布。
- $T$ 是温度，白话说是 softmax 的平滑旋钮，$T$ 越大，分布越平。
- $\alpha$ 是权重，控制硬标签与软标签各占多少。

结论有三条：

1. 硬标签只告诉学生“谁是第一”，软标签还告诉学生“第二、第三和第一差多远”。
2. 温度升高后，低概率类别不再几乎为零，暗知识更容易被学生看到。
3. 但温度也会把 KL 项的梯度缩小到 $O(1/T^2)$，所以必须乘上 $T^2$ 做量级归一化，否则软标签信号会被硬标签淹没。

一个直观玩具例子是三分类：教师 logits 是 $[3,1,0]$。如果直接 softmax，最大类几乎统治分布；如果设 $T=5$，学生不只知道“第一个类最好”，还知道“第二个类比第三个类更接近正确答案”。这部分结构，才是蒸馏真正压缩下来的信息。

---

## 问题定义与边界

知识蒸馏要解决的问题是：教师模型容量大、预测更细腻，但部署成本高；学生模型容量小、速度快，但表达能力弱。蒸馏通过让学生模仿教师的输出分布，尽量把教师的判别结构迁移到一个更小的模型中。

这里的边界要先说清楚：

| 项目 | 含义 | 过低/过高的结果 |
|---|---|---|
| 温度 $T$ | 控制 softmax 平滑程度 | $T=1$ 时接近普通分类；$T$ 过高会把分布抹得过平 |
| KL 项 | 对齐教师与学生的软分布 | 忽略后只剩硬标签监督，暗知识消失 |
| $T^2$ 缩放 | 归一化软损失梯度量级 | 不加时高温下 KL 梯度过弱 |
| logits 零均值化 | 给高温近似提供条件 | 不处理时 “KL 近似 MSE” 的推导会失真 |

温度的作用可以先看一个简单表：

| 温度 $T$ | softmax 平滑程度 | 低频项是否保留 | KL 梯度原始量级 |
|---|---|---|---|
| 1 | 弱平滑 | 基本不保留 | 大 |
| 5 | 中等平滑 | 明显保留 | 变小 |
| 10 | 强平滑 | 大量保留，但区分度下降 | 更小 |

对初学者来说，可以把它理解成一份考试成绩单。硬标签只告诉你“第一名是谁”；软标签则告诉你“第一名 95 分，第二名 91 分，第三名 72 分”。如果学生模型只看第一名，它学不到“第二名其实也很强”这类近邻关系。

经验上，分类蒸馏里常见的温度范围是 $T \in [3,10]$。这不是数学定理，而是工程经验：太低看不到暗知识，太高则会让所有类别太像，反而不利于有限容量学生学习。

---

## 核心机制与推导

先写出温度 softmax。softmax 是把一组任意实数转换成概率分布的函数。

$$
q_i^T=\frac{\exp(v_i/T)}{\sum_j \exp(v_j/T)}, \qquad
p_i^T=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

其中 $v_i$ 是教师 logits，$z_i$ 是学生 logits。logits 可以理解为“归一化前的打分”。

蒸馏的软损失通常写为：

$$
C=T^2 D_{\mathrm{KL}}(q^T \| p^T)
= T^2 \sum_i q_i^T \log \frac{q_i^T}{p_i^T}
$$

对学生 logits $z_i$ 求导，可得：

$$
\frac{\partial C}{\partial z_i}
= T^2 \cdot \frac{1}{T}(p_i^T-q_i^T)
= T(p_i^T-q_i^T)
$$

这一步先说明了一个关键点：softmax 内部除以 $T$，会让梯度天然缩小一个量级。

接着看高温极限。假设类别数是 $N$，并且教师与学生 logits 都做了零均值化：

$$
\sum_i v_i = 0, \qquad \sum_i z_i = 0
$$

当 $T \to \infty$ 时，指数项可以做一阶展开：

$$
\exp(z_i/T) \approx 1 + z_i/T
$$

于是：

$$
p_i^T \approx \frac{1+z_i/T}{N}
= \frac{1}{N} + \frac{z_i}{NT}
$$

同理：

$$
q_i^T \approx \frac{1}{N} + \frac{v_i}{NT}
$$

两者相减：

$$
p_i^T-q_i^T \approx \frac{z_i-v_i}{NT}
$$

再代回梯度公式：

$$
\frac{\partial C}{\partial z_i}
\approx T \cdot \frac{z_i-v_i}{NT}
= \frac{z_i-v_i}{N}
$$

如果不把前面的 $T^2$ 提出来，而是看原始 KL 的梯度，那么它会变成：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial z_i}
\approx \frac{z_i-v_i}{NT^2}
$$

这就是文献里常说的结论：高温下，原始 KL 梯度衰减为 $O(1/T^2)$，因此要乘 $T^2$ 才能把量级拉回正常范围。

进一步看目标函数本身，高温下最小化 KL 等价于最小化 logits 的平方误差：

$$
\min D_{\mathrm{KL}}(q^T \| p^T)
\quad \Longrightarrow \quad
\min \frac{1}{2}\sum_i (z_i-v_i)^2
$$

这就是“高温极限下蒸馏退化为 MSE”的来源。但这个结论有条件：它不是任意 logits 都成立，而是建立在高温近似和零均值化前提上。

看一个玩具例子。教师 logits 是 $[3,1,0]$，学生 logits 是 $[2.6,1.2,0.2]$，在 $T=5$ 时：

- 教师分布约为 $[0.47,0.32,0.26]$ 左右的平滑结构。
- 学生分布也会接近，但若第二、第三类比例不对，KL 会继续推动它们对齐。

如果只用硬标签交叉熵，梯度只关心“第一类还不够高”；如果用软标签 KL，梯度还会关心“第二类应该比第三类更像第一类”，这就是暗知识的具体表现。

一个真实工程例子是移动端文本分类。教师模型是 12 层 BERT，学生模型只有 4 层。硬标签训练常常只保证 top-1 正确，但对“体育/财经”“投诉/售后”这类相近标签的混淆结构学不到。蒸馏后，学生即使容量更小，也会保留教师对相近标签的排序偏好，召回率通常比纯硬标签训练更稳定。

---

## 代码实现

下面给一个可运行的 Python 版本，演示三件事：

1. 温度如何改变分布。
2. 为什么原始 KL 梯度在高温下会变弱。
3. 为什么乘 $T^2$ 后量级恢复稳定。

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def zero_mean(logits):
    mu = sum(logits) / len(logits)
    return [x - mu for x in logits]

def kl_div(q, p):
    eps = 1e-12
    return sum(qi * math.log((qi + eps) / (pi + eps)) for qi, pi in zip(q, p))

def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)

teacher = [3.0, 1.0, 0.0]
student = [2.6, 1.2, 0.2]

# 零均值化，匹配高温近似前提
teacher_zm = zero_mean(teacher)
student_zm = zero_mean(student)

q1 = softmax(teacher_zm, T=1.0)
q5 = softmax(teacher_zm, T=5.0)

# T 越高，分布越平
assert max(q5) < max(q1)

# 原始 KL 在高温下通常更小
kl_t1 = kl_div(softmax(teacher_zm, 1.0), softmax(student_zm, 1.0))
kl_t20 = kl_div(softmax(teacher_zm, 20.0), softmax(student_zm, 20.0))
assert kl_t20 < kl_t1

# 乘 T^2 后，高温量级被拉回
scaled_kl_t20 = (20.0 ** 2) * kl_t20
assert scaled_kl_t20 > kl_t20

# 高温下，scaled KL 与 logits MSE 呈同阶关系
dist_mse = mse(teacher_zm, student_zm)
assert dist_mse > 0

print("T=1:", [round(x, 4) for x in q1])
print("T=5:", [round(x, 4) for x in q5])
print("KL(T=1):", round(kl_t1, 6))
print("KL(T=20):", round(kl_t20, 6))
print("T^2 * KL(T=20):", round(scaled_kl_t20, 6))
print("MSE(logits):", round(dist_mse, 6))
```

训练时的实现结构通常如下：

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=5.0, alpha=0.5):
    # 零均值化，便于高温近似稳定
    student_logits = student_logits - student_logits.mean(dim=-1, keepdim=True)
    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)

    hard_loss = F.cross_entropy(student_logits, labels)

    teacher_prob = F.softmax(teacher_logits / T, dim=-1)
    student_log_prob = F.log_softmax(student_logits / T, dim=-1)

    soft_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    loss = (1 - alpha) * hard_loss + alpha * (T ** 2) * soft_loss
    return loss
```

工程上还有一个常见问题：教师前向很贵。解决方法一般有两种。

| 方案 | 做法 | 适用场景 |
|---|---|---|
| 在线蒸馏 | 每个 batch 实时跑教师 | 教师不大，数据增强很多 |
| 离线缓存 | 预先把教师 logits 存盘 | 教师很大，训练要重复多轮 |

离线缓存时，通常保存每个样本的教师 logits，而不是保存 softmax 后的概率。原因是后续你还可以调整温度 $T$，不用重新跑教师模型。

---

## 工程权衡与常见坑

蒸馏在论文里很干净，在工程里却很容易“看起来写对了，实际上没起作用”。下面这些坑最常见。

| 问题 | 现象 | 诊断指标 | 缓解策略 |
|---|---|---|---|
| 忘记乘 $T^2$ | 训练几乎退化成普通 CE | soft loss 数值很小，梯度几乎无贡献 | 在 KL 项外显式乘 `T**2` |
| 教师学生温度不一致 | 分布对齐异常 | loss 抖动，收敛慢 | 两边都用同一个 $T$ |
| logits 未零均值化 | 高温近似不稳定 | KL 与 MSE 关系不明显 | 减均值后再蒸馏 |
| $T$ 过高 | 分布太平，区分度下降 | top-1 提升有限，校准变差 | 从 3、5、7 逐步搜索 |
| $\alpha$ 过大 | 软标签主导，偏离真实标签 | 训练集 CE 下降慢 | 联合调 $\alpha$ 与 $T$ |

再把核心公式重复一遍，因为它最容易被“写漏”：

$$
L=(1-\alpha)L_{\mathrm{CE}}+\alpha T^2 D_{\mathrm{KL}}(q^T \| p^T)
$$

这里的 $T^2$ 不是经验技巧，而是来自梯度量级分析。如果不加，温度越高，KL 越难影响更新方向。

一个真实工程坑是小模型蒸馏到移动端。比如 4 层学生版 BERT 做多标签文本分类，如果标签之间存在强混淆关系，只用 BCE 或 CE 往往让模型学成“谁大推谁”。加入高温蒸馏后，学生会学到教师对相近标签组合的排序结构，例如“退款申请”和“售后投诉”经常一起出现，二者都高于“物流咨询”。这类结构对召回很关键，但如果温度太高或没加 $T^2$，这部分信息几乎传不过去。

另一个常见误区是把蒸馏理解成“学生复制教师输出”。更准确的说法是：学生在有限容量下，优先吸收教师最有迁移价值的结构信息。容量太小的学生不可能完整复制教师，因此蒸馏总是带有信息筛选。

---

## 替代方案与适用边界

概率蒸馏不是唯一选择。常见替代方案有 logits 回归和特征对齐。

| 方法 | 对齐对象 | 优势 | 局限 | 适用边界 |
|---|---|---|---|---|
| Softmax KL 蒸馏 | 概率分布 | 直接保留类间相似度 | 依赖温度和缩放 | 分类任务最常用 |
| Logit 回归 | 最后一层 logits | 实现简单，高温下与 KL 近似 | 对平移与尺度更敏感 | 教师 logits 可直接访问 |
| 特征对齐 | hidden states / attention | 更强表达迁移 | 结构依赖更强 | 教师学生层结构可映射 |

如果你只关心最后输出层，并且教师 logits 已经稳定可用，那么直接做 logits MSE 是一个可行基线：

$$
L_{\mathrm{logit}}=\|z^T-z^S\|_2^2
$$

它可以理解为“让两个模型最后一层打分尽量一致”。优点是简单，缺点是缺少概率归一化后的解释性，而且对 logit 平移更敏感，所以通常要先减均值。

实际工程里，常见折中方案是 “KL + 辅助 MSE”：

```python
import torch
import torch.nn.functional as F

def kd_with_logit_mse(student_logits, teacher_logits, labels, T=5.0, alpha=0.6, beta=0.1):
    student_centered = student_logits - student_logits.mean(dim=-1, keepdim=True)
    teacher_centered = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)

    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(
        F.log_softmax(student_centered / T, dim=-1),
        F.softmax(teacher_centered / T, dim=-1),
        reduction="batchmean",
    )
    logit_mse = F.mse_loss(student_centered, teacher_centered)

    return (1 - alpha) * ce + alpha * (T ** 2) * kl + beta * logit_mse
```

什么时候不该只用概率蒸馏？当学生特别小、任务特别复杂时，输出层信号可能不够。这时常加入 attention 对齐、hidden state 对齐甚至中间层 hint loss。白话说，只靠“最终答案分布”教不动时，就要把“中间思路”也一起教。

---

## 参考资料

| 资料 | 核心贡献 | 何时查阅 |
|---|---|---|
| Hinton et al., *Distilling the Knowledge in a Neural Network* | 提出经典蒸馏框架、温度 softmax、$T^2$ 缩放 | 先看理论定义与基本公式 |
| *What Mechanisms Does Knowledge Distillation Distill?* | 分析蒸馏到底迁移了什么机制，强调结构信息而不只是标签 | 想理解“暗知识”到底是什么时查 |
| System Overflow 蒸馏训练指南 | 给出工程训练配方、调参经验和实现细节 | 开始落地训练、排查效果差时查 |

- Hinton 这篇是理论起点，适合先建立公式和推导框架。
- OpenReview 这篇更偏机制解释，适合回答“蒸馏到底学到了什么”。
- System Overflow 更偏实践，适合训练不稳定、温度和权重不会调时快速定位问题。

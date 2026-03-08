## 核心结论

Transformer 的知识蒸馏里，学生模型不是“把参数砍小”这么简单。学生的**深度**、**宽度**和**注意力头数**共同决定容量。这里的容量，指模型能表示多少模式、规则和中间状态。如果压缩过猛，学生即使拿到了教师的输出分布，也没有足够表达能力去复现教师的内部计算路径；如果压缩过于保守，部署收益又不够大，蒸馏的工程价值会下降。

一个实用结论是：**优先缩深度，其次谨慎缩宽度，最后再动头数**。原因不是“层数最不重要”，而是三者带来的代价不同。减少层数，通常最直接地降低串行计算路径和推理延迟，同时还能保留单层内部的表示能力。缩宽度会同时影响自注意力投影、前馈网络、残差通道和任务头，属于影响范围最大的改动。缩头数则要分情况讨论：在标准多头注意力里，如果 `d_model` 不变，仅仅减少头数，参数量和 FLOPs 并不会像很多人想的那样显著下降，因为 `Q/K/V/O` 的总投影维度通常仍然是 `d_model`。真正变化的是每个头的子空间划分方式，以及多种关系模式的并行建模能力。所以头数通常是最后调的结构旋钮，而不是第一优先级。

蒸馏损失不能只看最终输出。**软标签蒸馏**是让学生匹配教师输出的概率分布；它能把类别之间的相对关系一并传给学生。但仅靠 soft label 往往不够，因为学生容易学到“结果像”，却学不到“过程像”。因此在 Transformer 场景下，常见做法是把输出蒸馏和中间层蒸馏组合起来：

$$
L=\alpha L_{\text{hard}} + (1-\alpha)\tau^2 \mathrm{KL}\big(\sigma(z_T/\tau), \sigma(z_S/\tau)\big) + \lambda_{\text{attn}}L_{\text{attn}} + \lambda_{\text{hid}}L_{\text{hid}} + \lambda_{\text{emb}}L_{\text{emb}}
$$

这里的 $L_{\text{hard}}$ 是监督标签损失，$\mathrm{KL}$ 是教师分布和学生分布的差异；后面的注意力、隐藏态和嵌入对齐，用来把学生的内部表示路径拉向教师，而不是只在最后一层修正答案。

最典型的例子是 DistilBERT。它把 BERT-base 从 12 层压到 6 层，隐藏维度保持 768，头数保持 12，并用教师的隔层权重初始化学生，再用蒸馏损失、MLM 损失和余弦嵌入损失联合训练。这里真正重要的不是“参数从 110M 降到 66M”本身，而是**学生尽量保留教师的结构骨架，从而降低对齐难度**。TinyBERT 则更进一步，不只对齐输出，还对齐 attention、hidden state 和 embedding，并区分通用蒸馏与任务蒸馏两个阶段。它代表的是“结构化模仿”，而不仅是“答案模仿”。

下表先给出工程上最常见的设计对比：

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 参数规模 | 主要压缩方式 | 典型收益 |
|---|---:|---:|---:|---:|---|---|
| BERT-base | 12 | 768 | 12 | 约 110M | 基线 | 精度高，延迟高 |
| DistilBERT | 6 | 768 | 12 | 约 66M | 深度减半 | 推理更快，精度保留较好 |
| TinyBERT-6L | 6 | 768 或更小 | 常随宽度调整 | 更低 | 深度压缩 + 多层对齐 | 结构更灵活，适合分阶段蒸馏 |
| TinyBERT-4L | 4 | 更小 | 更小 | 更低 | 深度 + 宽度联合压缩 | 延迟更低，但精度风险更大 |

结论可以压缩成一句话：**学生模型的架构设计，第一优先级是可对齐性，第二优先级才是极限压缩率；同族结构缩减通常比完全换架构更稳。**

---

## 问题定义与边界

知识蒸馏要解决的问题，不是简单地“把大模型变小”，而是：**在给定部署预算下，设计一个学生 Transformer，使它尽量保留教师模型的语言理解能力、推理模式和任务性能。**

这里的部署预算通常包括三类：

| 约束项 | 白话解释 | 常见指标 |
|---|---|---|
| 延迟预算 | 一次推理最多能花多少时间 | CPU 单次请求 ms、P95 延迟、tokens/s |
| 资源预算 | 机器能给模型多少内存和算力 | 参数量、显存、RAM、峰值激活、FLOPs |
| 精度预算 | 允许掉多少效果 | GLUE、F1、EM、准确率下降不超过 1% 到 3% |

学生架构里能调的主要是三项：

| 可调维度 | 变化方式 | 对参数 / 计算的影响 | 主要风险 |
|---|---|---|---|
| 层数 depth | 12→6、12→4 | 最直接影响串行深度和延迟 | 推理链条变短，抽象层级减少 |
| 隐藏维度 width | 768→512、768→312 | 参数和计算大幅下降 | 表示空间变窄，中间层难对齐 |
| 头数 heads | 12→8、12→6 | 主要改变子空间划分；若 `d_model` 不变，节省通常有限 | 每头维度变化，多关系并行建模能力变弱 |

可以把架构选择看成一个双向约束问题：

```text
任务要求（精度下限）
        ↓
可接受性能下降 <= 1%~3%
        ↓
确定部署预算（延迟 / 参数 / 内存 / FLOPs）
        ↓
先试减层，再决定是否减宽，最后评估是否动头数
        ↓
决定蒸馏损失：仅 logits，还是 logits + hidden + attention + embedding
        ↓
验证训练稳定性与部署平台约束
```

### 玩具例子

假设教师模型做三分类任务，输出 logits 是：

- 教师：`[4.0, 3.6, 0.2]`
- 硬标签：第 1 类

如果只看硬标签，学生只知道“第 1 类是对的”。但软分布会告诉学生：第 2 类也很像，只是略差。这意味着教师认为前两类在语义上接近。对新手可以这样理解：软标签不是只给一个标准答案，而是把“其他选项分别错多少”也告诉学生。

取温度 $\tau=2$，教师 softmax 后的分布大约是：

$$
p_T \approx [0.50,\ 0.41,\ 0.09]
$$

这组数的重点不在绝对值，而在相对关系：第 2 类并不是“完全错误”，而是“次优候选”。学生如果学不到这个结构，就只能学到一个更硬、更窄的边界。

### 真实工程例子

在 CPU 后台服务里，很多团队的第一步不是直接把 BERT-base 改成 4 层、312 维，而是先做一个 **6 层、768 维、12 头** 的学生。原因通常有四个：

1. 输入嵌入维度不变，旧 tokenizer、词向量和任务头更容易复用。
2. 中间层蒸馏时不需要额外投影，训练更稳。
3. 深度减半已经能带来明显延迟收益，常常足够达到第一阶段上线目标。
4. 一旦 6 层模型验证通过，再继续减宽，会有更明确的性能基线。

这就是边界意识。如果你的上线目标是“精度最多掉 2%，延迟下降 30%”，那么先减层通常比同时大幅减宽更合理。

---

## 核心机制与推导

蒸馏的核心不是一句“让学生模仿教师”，而是让学生同时匹配教师的**输出结果**和**内部计算路径**。前者决定最终预测是否接近，后者决定学生是不是沿着与教师相似的表示轨迹完成计算。

### 1. 输出层匹配

设教师 logits 为 $z_T$，学生 logits 为 $z_S$，温度为 $\tau$。带温度的 softmax 定义为：

$$
\sigma_i(z/\tau)=\frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

当 $\tau > 1$ 时，分布会被拉平，类别之间原本很小的概率差异会更容易被观察到。蒸馏项通常写成：

$$
L_{\text{distill}}=\tau^2 \cdot \mathrm{KL}\big(\sigma(z_T/\tau), \sigma(z_S/\tau)\big)
$$

其中 $\mathrm{KL}$ 衡量教师分布和学生分布的差异。乘上 $\tau^2$ 的原因，是在反向传播时补偿温度放大带来的梯度缩小，避免高温下 soft loss 影响过弱。

再加上硬标签监督：

$$
L_{\text{hard}}=\mathrm{CE}(y,\sigma(z_S))
$$

组合后得到输出层损失：

$$
L_{\text{out}}=\alpha L_{\text{hard}} + (1-\alpha)L_{\text{distill}}
$$

这部分负责让学生学到“最终判断长什么样”。

### 2. 为什么 soft label 比 hard label 信息更多

看一个更具体的数值。教师 logits 为：

$$
z_T=[4.0,\ 3.6,\ 0.2]
$$

当 $\tau=1$ 时，softmax 近似为：

$$
\sigma(z_T)\approx[0.589,\ 0.395,\ 0.016]
$$

当 $\tau=2$ 时，softmax 近似为：

$$
\sigma(z_T/2)\approx[0.499,\ 0.409,\ 0.092]
$$

可以看到，高温把尾部类别抬起来了。硬标签只保留“第 1 类正确”，而软标签保留了“第 2 类与第 1 类接近、第 3 类差很多”。这类结构信息对小模型很重要，因为学生容量有限，最应该先学的是教师已经压缩过的相对关系，而不是从数据中重新发明一次全部边界。

### 3. 注意力对齐

注意力矩阵描述的是“每个 token 在当前层更关注哪些 token”。它不是最终答案，但它是模型构建依赖关系的中间路径。对于学生第 $l$ 层和教师映射层 $m(l)$，注意力对齐可写成：

$$
L_{\text{attn}}^{(l)}=\mathrm{MSE}(A_S^{(l)}, A_T^{(m(l))})
$$

若教师 12 层、学生 6 层，常见映射是：

| 学生层 | 教师层 |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
| 5 | 10 |
| 6 | 12 |

这一步的意义不是让数值逐元素完全一样，而是让学生学到类似的“关注路径”。对新手可以理解成：教师在某层里主要关注主语和谓语，学生也应该学会在相近层次上看向这些关键位置。

### 4. 隐藏态对齐

隐藏态是每层输出的向量表示。它可以理解为模型在某一层对整段输入的内部编码结果。若师生维度一致，可以直接做均方误差：

$$
L_{\text{hid}}^{(l)}=\mathrm{MSE}(H_S^{(l)}, H_T^{(m(l))})
$$

如果维度不一致，比如教师是 768，学生是 312，则不能直接比较。需要引入投影矩阵 $W_h$：

$$
L_{\text{hid}}^{(l)}=\mathrm{MSE}(H_S^{(l)}W_h,\ H_T^{(m(l))})
$$

这里的 $W_h \in \mathbb{R}^{d_S \times d_T}$。它的作用不是“做个形式上的适配”，而是把学生表示映射到教师所在的表示空间。否则两个向量连坐标系都不同，MSE 没有稳定意义。

### 5. 嵌入层对齐

TinyBERT 一类方法还会对齐输入嵌入：

$$
L_{\text{emb}}=\mathrm{MSE}(E_S W_e,\ E_T)
$$

其中 $W_e$ 是嵌入投影矩阵。它的作用是让学生从底层表示开始就尽量靠近教师，而不是只在顶部 logits 上强行纠正。

### 6. 完整损失与层映射

把上面几项合在一起，就得到常见的联合目标：

$$
L=\alpha L_{\text{hard}} + (1-\alpha)\tau^2 \mathrm{KL}\big(\sigma(z_T/\tau), \sigma(z_S/\tau)\big) + \lambda_{\text{attn}}\sum_l L_{\text{attn}}^{(l)} + \lambda_{\text{hid}}\sum_l L_{\text{hid}}^{(l)} + \lambda_{\text{emb}}L_{\text{emb}}
$$

各项作用可以用下表概括：

| 损失项 | 监督对象 | 解决的问题 | 典型收益 | 典型风险 |
|---|---|---|---|---|
| $L_{\text{hard}}$ | 真实标签 | 防止学生只学教师偏差 | 保证任务目标 | 数据少时不够强 |
| KL 蒸馏项 | logits 分布 | 传递类别相对关系 | 收敛更稳，泛化更好 | 单独使用时只学结果 |
| $L_{\text{attn}}$ | 注意力矩阵 | 学习 token 依赖模式 | 长句结构更稳 | 噪声大时约束过强 |
| $L_{\text{hid}}$ | 中间隐藏态 | 学习层级表示路径 | 提升上限 | 维度不一致时需投影 |
| $L_{\text{emb}}$ | 输入嵌入 | 从底层表示开始贴近教师 | 两阶段蒸馏更有效 | 训练更复杂 |

### 7. 分阶段蒸馏

TinyBERT 的关键设计是两阶段：

1. **General distillation**：在大规模通用语料上对齐 embedding、hidden、attention，让学生先学通用语言结构。
2. **Task-specific distillation**：在具体任务数据上继续蒸馏 prediction、task representation 和任务头。

这样做的原因是：如果一开始就只在小规模任务数据上训练，学生容易只记住标签边界，而学不到教师的通用语言知识。

### 8. DistilBERT 与 TinyBERT 的机制差别

| 方法 | 核心思路 | 主要对齐项 | 适合场景 |
|---|---|---|---|
| DistilBERT | 保留同类结构，先减半层数 | 输出分布 + MLM + 余弦嵌入损失 | 通用模型快速压缩 |
| TinyBERT | 细粒度模仿教师内部结构 | attention + hidden + embedding + prediction | 精度敏感任务、可接受双阶段训练 |

### 9. 为什么同结构缩减通常更稳

教师层可写成：

$$
H_T^{(l+1)}=\mathrm{TransformerBlock}(H_T^{(l)})
$$

如果学生仍然使用相同类型的 block，只是层数更少，那么蒸馏本质上是在逼近教师的一个“稀疏采样轨迹”。优化器面对的是**同构映射 + 少量缺失步骤**。而如果学生换成完全不同的结构，蒸馏就变成“用另一个函数族复现当前函数族的行为”，对齐空间会更不连续，优化难度会更高。

---

## 代码实现

下面先给一个**最小可运行**的纯 Python 实现。它不依赖深度学习框架，只演示蒸馏里最核心的三件事：温度 softmax、KL 损失，以及 hard label 交叉熵。

```python
import math

def softmax(logits, temperature=1.0):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def kl_divergence(p, q, eps=1e-12):
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return total

def cross_entropy_from_probs(target_index, probs, eps=1e-12):
    p = max(probs[target_index], eps)
    return -math.log(p)

def distillation_loss(
    teacher_logits,
    student_logits,
    label_index,
    alpha=0.5,
    temperature=2.0,
):
    teacher_probs_t = softmax(teacher_logits, temperature)
    student_probs_t = softmax(student_logits, temperature)
    student_probs_1 = softmax(student_logits, 1.0)

    hard_loss = cross_entropy_from_probs(label_index, student_probs_1)
    soft_loss = kl_divergence(teacher_probs_t, student_probs_t) * (temperature ** 2)
    total = alpha * hard_loss + (1.0 - alpha) * soft_loss

    return {
        "teacher_probs_t": teacher_probs_t,
        "student_probs_t": student_probs_t,
        "student_probs_1": student_probs_1,
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "total_loss": total,
    }

teacher_logits = [4.0, 3.6, 0.2]
student_logits_good = [3.8, 3.4, 0.5]
student_logits_bad = [4.5, 0.2, 0.1]
label_index = 0
tau = 2.0

good = distillation_loss(teacher_logits, student_logits_good, label_index, alpha=0.5, temperature=tau)
bad = distillation_loss(teacher_logits, student_logits_bad, label_index, alpha=0.5, temperature=tau)

assert abs(sum(good["teacher_probs_t"]) - 1.0) < 1e-9
assert good["soft_loss"] < bad["soft_loss"]

print("teacher probs @tau=2:", [round(x, 4) for x in good["teacher_probs_t"]])
print("good soft loss:", round(good["soft_loss"], 6))
print("bad soft loss :", round(bad["soft_loss"], 6))
print("good total loss:", round(good["total_loss"], 6))
print("bad total loss :", round(bad["total_loss"], 6))
```

这段代码表达的结论是：即使两个学生都把第 1 类排第一，那个更接近教师分布的学生，KL 更小，也更符合蒸馏目标。

下面给出一个**可运行的 PyTorch 最小例子**。它不依赖 Hugging Face 模型，只用随机张量模拟 logits、attention 和 hidden states，演示联合损失如何组织。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

def mse_loss(a, b):
    return ((a - b) ** 2).mean()

def build_fake_outputs(batch_size, seq_len, num_labels, num_layers, num_heads, hidden_dim):
    logits = torch.randn(batch_size, num_labels)
    hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers + 1)]
    attentions = [torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_layers)]
    return {
        "logits": logits,
        "hidden_states": hidden_states,
        "attentions": attentions,
    }

def distill_step(
    teacher_out,
    student_out,
    labels,
    teacher_layer_ids,
    proj_hidden=None,
    proj_embed=None,
    alpha=0.5,
    tau=2.0,
    lambda_attn=1.0,
    lambda_hid=1.0,
    lambda_emb=1.0,
):
    hard_loss = F.cross_entropy(student_out["logits"], labels)

    t_prob = F.softmax(teacher_out["logits"] / tau, dim=-1)
    s_log_prob = F.log_softmax(student_out["logits"] / tau, dim=-1)
    kl_loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (tau ** 2)

    attn_loss = torch.tensor(0.0)
    hid_loss = torch.tensor(0.0)

    for i, t_idx in enumerate(teacher_layer_ids):
        s_attn = student_out["attentions"][i]
        t_attn = teacher_out["attentions"][t_idx]
        attn_loss = attn_loss + mse_loss(s_attn, t_attn)

        s_hid = student_out["hidden_states"][i + 1]
        t_hid = teacher_out["hidden_states"][t_idx + 1]

        if proj_hidden is not None:
            s_hid = proj_hidden(s_hid)

        hid_loss = hid_loss + mse_loss(s_hid, t_hid)

    s_emb = student_out["hidden_states"][0]
    t_emb = teacher_out["hidden_states"][0]
    if proj_embed is not None:
        s_emb = proj_embed(s_emb)
    emb_loss = mse_loss(s_emb, t_emb)

    total = (
        alpha * hard_loss
        + (1.0 - alpha) * kl_loss
        + lambda_attn * attn_loss
        + lambda_hid * hid_loss
        + lambda_emb * emb_loss
    )

    return {
        "hard_loss": hard_loss.item(),
        "kl_loss": kl_loss.item(),
        "attn_loss": attn_loss.item(),
        "hid_loss": hid_loss.item(),
        "emb_loss": emb_loss.item(),
        "total_loss": total.item(),
    }

batch_size = 2
seq_len = 5
num_labels = 3

teacher_layers = 12
student_layers = 6
num_heads = 12
hidden_teacher = 768
hidden_student = 768

teacher_out = build_fake_outputs(batch_size, seq_len, num_labels, teacher_layers, num_heads, hidden_teacher)
student_out = build_fake_outputs(batch_size, seq_len, num_labels, student_layers, num_heads, hidden_student)
labels = torch.tensor([0, 2])

teacher_layer_ids = [1, 3, 5, 7, 9, 11]  # 12层教师 -> 6层学生

loss_dict = distill_step(
    teacher_out=teacher_out,
    student_out=student_out,
    labels=labels,
    teacher_layer_ids=teacher_layer_ids,
    proj_hidden=None,
    proj_embed=None,
    alpha=0.5,
    tau=2.0,
    lambda_attn=1e-3,  # 玩具例子里数值缩小，避免 attn MSE 主导
    lambda_hid=1.0,
    lambda_emb=1.0,
)

for k, v in loss_dict.items():
    print(f"{k}: {v:.6f}")
```

如果你要把它改成真实 Hugging Face 训练流程，最关键的是三件事：

1. `teacher(..., output_hidden_states=True, output_attentions=True)`  
   否则拿不到中间层监督信号。
2. 明确层映射关系  
   比如 12 层教师到 6 层学生，常用 `[1, 3, 5, 7, 9, 11]`。
3. 当师生宽度不同，必须加投影层  
   比如 `nn.Linear(d_student, d_teacher, bias=False)`。

### 初始化策略

如果学生是 6 层、教师是 12 层，常见初始化方式是按规则抽取教师层参数：

| 学生层 | 教师来源层 |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
| 5 | 10 |
| 6 | 12 |

这相当于先把学生放在教师轨道附近。没有这一步也能训练，但通常收敛更慢，且结果更不稳定。

### 参数配置建议

| 参数 | 作用 | 常见经验 |
|---|---|---|
| `tau` | 平滑教师分布 | 常用 2 到 5 |
| `alpha` | 平衡 hard / soft loss | 常见 0.3 到 0.7 |
| `lambda_attn` | 注意力对齐强度 | 任务复杂、长句较多时可适当提高 |
| `lambda_hid` | 隐藏态对齐强度 | 宽度变化时更关键 |
| `lambda_emb` | 嵌入对齐强度 | 两阶段蒸馏里较常见 |
| `proj_hidden` | 隐藏态投影 | 师生维度不同时必需 |

### 真实工程例子

如果你要把一个线上 BERT-base 分类服务迁移到蒸馏模型，最稳的路径通常是：

1. 先做 6 层、768 维学生，复用原 tokenizer、embedding 维度和分类头。
2. 用教师离线跑训练集，缓存 logits；如果资源允许，再缓存 hidden states 或 attention。
3. 第一阶段只训输出蒸馏，确认学生能收敛。
4. 第二阶段接入 hidden / attention 对齐，提升精度上限。
5. 线上压测 CPU 延迟和内存占用，再决定是否继续减宽。

这样做比一开始就上 4 层、312 维、6 头更工程化，因为你先解决“能上线”，再解决“压到极限”。

---

## 工程权衡与常见坑

学生架构设计最大的误区，是把“参数更少”误当成唯一目标。实际工程里，更重要的是**单位延迟、单位内存、单位算力下的有效能力**。一个 6 层 768 维学生，往往比 4 层 312 维学生更值得上线，因为前者虽然不是最小，但更稳、更容易复现教师性能。

下表总结常见问题：

| 问题 | 原因 | 规避策略 | 影响 |
|---|---|---|---|
| 只蒸馏 logits，效果不稳 | 学到结果，学不到过程 | 增加 hidden / attention 对齐 | 泛化偏弱 |
| 宽度直接从 768 降到 256 后 MSE 发散 | 师生表示不在同一空间 | 加线性投影 $W_h$、$W_e$ | 训练不收敛 |
| 头数缩太多 | 每头维度变化，关系模式变少 | 先保持头数，优先减层 | 长句依赖变差 |
| 压缩过猛 | 学生容量不足 | 分阶段蒸馏，逐步压缩 | 性能塌陷 |
| 层映射设计随意 | 监督不对应 | 使用规则映射或搜索映射 | 中间层损失噪声大 |
| 误以为减头数能大幅降参 | 标准 MHA 中总投影维度通常不变 | 把头数视为表达结构超参，而非主要压缩杠杆 | 架构判断失真 |

### 坑 1：过早缩宽度

这是最常见的错误。很多人看到 FFN 参数占比大，就先把隐藏维度从 768 砍到 256。参数确实掉得快，但训练难度会明显上升，因为：

- 注意力投影矩阵和 FFN 同时缩水；
- 每层表示空间变窄；
- hidden state 对齐必须依赖投影；
- 任务头也常常要重配。

如果没有非常强的压缩约束，先做“减层不减宽”通常更稳。

### 坑 2：忽略中间层监督

只做输出蒸馏时，学生可能在训练集上表现正常，但泛化能力不稳定。原因是学生学到的是最终类别边界，而不是教师内部如何组织语义表示。对新手可以把它理解成：学生背会了答案，但没学会解题步骤。

### 坑 3：维度不一致时直接做 MSE

假设教师 hidden dim 是 768，学生是 256。你如果直接拿两者做 MSE，不只是代码形状不匹配，更是目标定义错误。正确做法是：

$$
\hat{H}_S = H_S W_h,\quad L_{\text{hid}} = \mathrm{MSE}(\hat{H}_S, H_T)
$$

这个投影层不是装饰，而是必要的坐标变换。

### 坑 4：把所有损失权重都设成一样

蒸馏项不是越多越好。任务数据少时，过强的 hidden / attention loss 可能压制任务相关学习；任务数据多且教师质量高时，这些内部对齐又很关键。工程上通常要做小规模网格搜索，而不是固定死一套权重。

### 坑 5：忽略部署平台的真实瓶颈

如果目标平台是 CPU，层数对延迟的影响常常比参数量更直接，因为层数增加了串行深度。  
如果目标平台是显存敏感的 GPU 批处理服务，参数量和激活峰值可能更重要。  
如果目标平台是移动端，模型体积、内存映射和算子支持情况可能比论文里的 GLUE 分数更关键。

### 一个实际判断标准

如果你的目标是线上 CPU 服务，优先看：

1. 单样本平均延迟是否达到目标范围。
2. P95 延迟是否稳定。
3. 精度是否仍在业务容忍区间。
4. 模型大小是否方便部署、热更新和回滚。

如果你的目标是移动端或边缘端，模型体积和峰值内存可能比训练集指标更重要。这时可以接受更激进的压缩，但要明确：性能损失是架构决策的一部分，不是简单把锅推给“训练没调好”。

---

## 替代方案与适用边界

学生架构设计并不只有“减半层数”一种做法。不同部署场景，对“最优学生”的定义并不相同。

### 路线 1：同宽减层

这是 DistilBERT 风格。保留隐藏维度和头数，只减少层数。优点是对齐简单、初始化方便、训练稳定。缺点是参数虽然下降明显，但不是最小。

适用边界：
- 追求快速替换线上基座模型；
- 多任务共用一个通用学生；
- 对精度下降比较敏感。

### 路线 2：减层 + 减宽

这是更激进的压缩方式。优点是参数和 FLOPs 都降得更多，适合边缘设备。缺点是需要投影层、训练更难、效果更依赖蒸馏细节。

适用边界：
- 延迟和内存预算都很紧；
- 能接受更长的蒸馏训练周期；
- 有足够数据支撑中间层对齐。

### 路线 3：分阶段蒸馏

这更接近 TinyBERT。先做 general distillation，再做 task-specific distillation。优点是结构知识学得更完整。缺点是训练流程更长，工程链路更复杂。

适用边界：
- 下游任务复杂，且精度要求高；
- 希望学生保留通用语言能力；
- 工程团队能承担双阶段训练成本。

### 路线 4：同层数但改注意力结构

有些场景不会优先减层，而是保留深度、改局部注意力、共享参数、做低秩分解或 KV 共享。它们不一定属于经典蒸馏学生设计，但常与蒸馏叠加使用。

适用边界：
- 模型主要瓶颈是长序列注意力成本；
- 需要保留深层推理链；
- 可以接受改动基础算子。

下表做一个集中对比：

| 方案 | 阶段 | 对齐维度 | 初始化方式 | 适用场景 |
|---|---|---|---|---|
| DistilBERT 风格 | 单阶段为主 | 输出分布 + 辅助损失 | 教师层复制 / 抽样 | 通用压缩、快速部署 |
| TinyBERT 风格 | 两阶段 | attention + hidden + embedding + prediction | 层映射 + 投影 | 高精度需求、任务蒸馏 |
| 激进轻量学生 | 可单阶段或两阶段 | 输出为主，内部对齐按预算决定 | 常需投影和重设宽度 | 边缘设备、极低延迟 |
| 结构改造型学生 | 视方案而定 | 可与蒸馏叠加 | 结构重设计 | 长上下文、算子受限平台 |

### 如何选

可以用一个简单原则：

- **预算宽松，先减层。**
- **预算紧张，再减宽，但必须给 hidden / embedding 加投影对齐。**
- **任务对精度敏感，用多层对齐和分阶段蒸馏。**
- **需要快速上线，优先选同结构缩减。**
- **不要把减头数当成第一压缩杠杆，除非你同时调整 `d_model` 或注意力实现。**

### 真实工程例子

假设你要在边缘设备上部署问答模型，延迟预算极紧。此时 6 层 768 维可能仍然太大。可以选 TinyBERT 风格学生，再适度减少宽度和层数，并用两阶段蒸馏先迁移通用语言能力，再迁移任务能力。反过来，如果你只是想把已有的 CPU 文本分类服务提速，DistilBERT 风格通常更划算，因为结构改动更小、上线风险更低。

---

## 参考资料

| 来源 | 主题 | 核心贡献 | 阅读顺序建议 |
|---|---|---|---|
| Hinton, Vinyals, Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) | 知识蒸馏基础 | 提出 soft target、temperature 和 teacher-student 蒸馏框架，是蒸馏问题的起点 | 1，先建立 soft label 和温度的基本概念 |
| Devlin et al., [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://aclanthology.org/N19-1423/) | 教师结构基线 | 给出 BERT-base / large 的层数、隐藏维度、头数等标准配置，是讨论学生架构设计的直接基线 | 2，先明确教师结构长什么样 |
| Sanh et al., [*DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*](https://arxiv.org/abs/1910.01108) | DistilBERT | 展示“减层但保留宽度和头数”的稳定压缩路线，并介绍蒸馏 + MLM + cosine 的训练组合 | 3，重点看为什么先减层更稳 |
| Jiao et al., [*TinyBERT: Distilling BERT for Natural Language Understanding*](https://aclanthology.org/2020.findings-emnlp.372/) | TinyBERT | 提出 embedding / hidden / attention / prediction 的细粒度对齐，以及 general distillation + task-specific distillation 的双阶段流程 | 4，重点看层对齐和两阶段设计 |
| Hugging Face, [Transformers Model Outputs 文档](https://huggingface.co/docs/transformers/main_classes/output) | 工程实现 | 说明 `hidden_states`、`attentions`、`logits` 的输出形式，便于把论文目标落成训练代码 | 5，写代码前查接口细节 |
| Hugging Face, [DistilBERT 文档](https://huggingface.co/docs/transformers/model_doc/distilbert) | 工程落地 | 方便对照 DistilBERT 的配置、层数、接口和部署方式 | 6，做实验或替换线上模型时再回看 |

建议的阅读顺序是：先看 Hinton 的蒸馏基础，再看 BERT 的教师结构基线，然后看 DistilBERT 的“同结构减层”，最后看 TinyBERT 的“细粒度内部对齐”。这样更容易建立从原理到工程方案的完整链路。

## 核心结论

Prompt Tuning 学的是一串 **soft prompt**，即“不是文字，而是可训练向量”的提示。它把离散的自然语言 prompt 换成连续的 embedding，放在输入最前面或最前面一段位置，主模型参数全部冻结，只更新这串向量。

这件事的核心价值有两点。第一，训练参数极少，通常是几万到几十万；第二，部署很轻，一个基础模型可以挂很多任务专用 prompt 文件，切换任务时不必加载一份新的大模型。对于大模型，Lester 等人在 2021 年的结果表明，Prompt Tuning 会随着模型规模增大而更接近全量微调；但对中小模型、复杂生成任务、需要学到新行为模式的任务，它通常不如 LoRA 这类层内注入方法稳定。

先看一个最小对比：

| 方法 | 改哪里 | 常见可训练参数量 | 部署负担 | 典型特点 |
| --- | --- | ---: | --- | --- |
| Prompt Tuning | 只改输入端 soft prompt | $l \times d$，常见几万到几十万 | 很低，只保存 prompt | 便宜、切换快、依赖模型规模 |
| LoRA | 在注意力/MLP 权重旁边加低秩矩阵 | 常见几十万到几百万 | 中等，需合并或额外挂载权重 | 表达力更强，常更稳 |
| Adapter | 每层插小模块 | 常见百万级到千万级 | 中等偏高 | 模块化强，但延迟略增 |
| 全量微调 | 所有参数都训练 | 全模型参数量 | 高，每个任务一份模型 | 表达力最强，成本最高 |

玩具例子：把一个问答分类任务迁移到 BERT-large，只训练 20 个 soft prompt token。若 embedding 维度 $d=1024$，那可训练参数只有 $20 \times 1024 = 20{,}480$，而不是 3 亿多参数。

真实工程例子：一个企业有同一个基础模型，分别服务“客服问答检索”“合同条款归类”“内部知识问答”三类任务。用 Prompt Tuning 时，它可以保留一份冻结主模型，只给每个任务保存一小份 prompt 参数，推理时按任务 ID 挂载对应 prompt，模型切换成本接近常数。

---

## 问题定义与边界

Prompt Tuning 解决的问题不是“让模型整体学会新结构”，而是“在不改主模型内部权重的前提下，让模型被一串可优化输入向量引导到某个任务上”。

这里有三个边界要先说清楚。

第一，**embedding** 是“把 token 映射成向量的表示层”。Prompt Tuning 并不改 embedding 表本身，而是额外加一串和 embedding 同维度的可训练向量。

第二，**冻结模型** 是“前向照常跑，但参数不更新”。所以它不是缩小版全量微调，而是只在输入侧增加一个很薄的可学习接口。

第三，它和“写一句更好的自然语言提示”不是一回事。硬提示是人能读懂的文本；软提示是高维向量，人通常看不懂，只能通过梯度下降学出来。

结构可以画成这样：

```text
原始输入:
[token_1, token_2, token_3, ..., token_n]
   │
embedding
   ▼
[e_1, e_2, e_3, ..., e_n]
   │
frozen Transformer
   ▼
task head

Prompt Tuning 输入:
[p_1, p_2, ..., p_l, e_1, e_2, e_3, ..., e_n]
   │
frozen Transformer
   ▼
task head
```

BERT-base 做命名实体识别可以作为一个具体边界例子。假设输入句子是“Apple released a new iPhone in California”，传统做法是直接把 token embedding 喂给模型；Prompt Tuning 则先拼 30 个 soft prompt token，再接原句 embedding。标签预测逻辑不变，还是靠原模型输出每个位置的分类结果。也就是说，它改的是“模型看到输入时的上下文起点”，不是“模型内部怎么计算”。

这决定了它的优点和上限：

| 维度 | Prompt Tuning 的边界 |
| --- | --- |
| 参数效率 | 很强，因为只学输入端 |
| 新行为学习能力 | 有限，因为层内结构没变 |
| 多任务部署 | 很强，一个底座挂多个 prompt |
| 小模型效果 | 往往一般，容量不够时更明显 |
| 复杂生成任务 | 常不如 LoRA/Adapter |

---

## 核心机制与推导

Prompt Tuning 的参数量几乎可以直接写出来。设 soft prompt 长度为 $l$，模型输入 embedding 维度为 $d$，那么：

$$
\mathcal{P}_{prompt} = l \times d
$$

这个公式的含义非常直接：每个 soft prompt token 本质上是一条长度为 $d$ 的向量，一共有 $l$ 条，所以总参数就是一个 $l \times d$ 的矩阵。

以 BERT-large 为例，$d=1024$，如果只加 20 个 soft prompt token，则：

$$
\mathcal{P}_{prompt} = 20 \times 1024 = 20{,}480
$$

而 BERT-large 的总参数量大约是 3.35 亿，所以这部分只占一个极小比例。它之所以能起作用，不是因为参数多，而是因为这些参数直接作用在模型输入空间，能持续影响后续每一层的表示传播。

前向过程可以写成：

$$
H^{(0)} = [P; E(X)]
$$

其中：

- $X$ 是原始输入 token 序列
- $E(X)$ 是原始输入经过 embedding 后的向量序列
- $P \in \mathbb{R}^{l \times d}$ 是 soft prompt 参数
- $[P; E(X)]$ 表示按序列维度拼接

任务训练时，优化目标通常还是普通监督学习损失，比如交叉熵：

$$
\mathcal{L}(P) = -\frac{1}{B}\sum_{j=1}^{B}\log p_\theta(y^{(j)} \mid [P; X^{(j)}])
$$

这里 $\theta$ 是冻结的主模型参数，不更新；只有 $P$ 更新。于是反向传播虽然穿过整个网络，但最终只改 soft prompt。

可以把它理解成“输入端新增了一个参数化层”，但这个层非常特殊：它不依赖具体样本内容，所有样本共享同一组 prompt 参数。于是它更像一个任务级控制器，而不是样本级编码器。

推导链条可以压缩成四步：

1. 初始化一个 $P \in \mathbb{R}^{l \times d}$。
2. 每次把 $P$ 拼到输入 embedding 前面。
3. 用冻结主模型计算任务损失。
4. 只对 $P$ 求梯度并更新。

为什么大模型更有效？一个常见解释是：大模型原本已经学到大量可复用能力，soft prompt 不需要“教会”模型新知识，只需要“把已有能力调出来”。当主模型足够大时，这种输入侧控制就够用了；当模型较小或任务离预训练分布太远时，只改输入往往不够。

---

## 代码实现

下面先给一个最小可运行玩具实现。这个例子不依赖深度学习框架，只用 `numpy` 演示“冻结主模型，只训练 soft prompt”的机制。主模型被简化成：对拼接后的向量求均值，再做一个固定线性分类。这个模型很弱，但足够说明梯度只更新 prompt。

```python
import numpy as np

np.random.seed(42)

# 冻结的“主模型”参数
d = 8                  # embedding 维度
seq_len = 4            # 原始输入 token 数
prompt_len = 3         # soft prompt 长度
W = np.random.randn(d) # 固定分类头
b = -0.1               # 固定偏置

# 一条样本的原始 token embedding，视为冻结输入表示
x = np.random.randn(seq_len, d)
y = 1.0  # 二分类标签，目标是正类

# 只训练 soft prompt
P = np.random.randn(prompt_len, d) * 0.01

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(prompt):
    h = np.concatenate([prompt, x], axis=0)   # 拼接 soft prompt 和原始输入
    pooled = h.mean(axis=0)                   # 简化的 frozen encoder 输出
    logit = pooled @ W + b
    prob = sigmoid(logit)
    loss = -(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12))
    return loss, prob

loss0, prob0 = forward(P)

lr = 0.5
for _ in range(200):
    h = np.concatenate([P, x], axis=0)
    pooled = h.mean(axis=0)
    logit = pooled @ W + b
    prob = sigmoid(logit)

    # dLoss/dlogit for binary cross entropy
    dlogit = prob - y

    # logit = pooled @ W + b
    dpooled = dlogit * W

    # pooled = mean([P; x])
    dP = np.tile(dpooled / (prompt_len + seq_len), (prompt_len, 1))

    P -= lr * dP

loss1, prob1 = forward(P)

assert loss1 < loss0, (loss0, loss1)
assert prob1 > prob0, (prob0, prob1)

print("initial loss:", round(float(loss0), 4))
print("final loss:", round(float(loss1), 4))
print("initial prob:", round(float(prob0), 4))
print("final prob:", round(float(prob1), 4))
```

这个代码里的关键点是：

- `P` 就是 soft prompt，形状是 `(l, d)`。
- `x`、`W`、`b` 都冻结，不参与更新。
- 前向时先 `concatenate`，等价于把 prompt 放到输入最前面。
- 反向时只计算并更新 `dP`。

如果换成 PyTorch，工程上通常写成下面这样：

```python
import torch
import torch.nn as nn

l, d = 20, 1024
soft_prompt = nn.Parameter(torch.randn(l, d) * 0.02)

# 假设 model 是冻结好的预训练模型
for p in model.parameters():
    p.requires_grad = False

token_embeds = model.get_input_embeddings()(input_ids)   # [B, T, d]
prompt = soft_prompt.unsqueeze(0).expand(token_embeds.size(0), -1, -1)  # [B, l, d]
input_embeds = torch.cat([prompt, token_embeds], dim=1)  # [B, l+T, d]

outputs = model(inputs_embeds=input_embeds, labels=labels)
loss = outputs.loss
loss.backward()  # 梯度只会流到 soft_prompt
optimizer.step()
```

真实工程例子：做企业 FAQ 检索增强问答。你已经有一个冻结的生成模型和一套 RAG 流程。现在不同业务线只差“回答风格、领域优先级、术语偏好”。这种情况下，Prompt Tuning 可以给每条业务线单独训练一组 soft prompt，并在检索结果前统一拼接。底座模型和检索链路不改，线上只需要多维护几份小 prompt 文件。

---

## 工程权衡与常见坑

Prompt Tuning 最常见的问题不是“训不动”，而是“能训动，但效果不稳定”。原因通常集中在三个超参数：prompt 长度、初始化方式、学习率。

**初始化** 是“训练开始时 soft prompt 的初值”。它比很多人以为的更重要。完全随机初始化经常能跑通，但在中小模型上可能收敛慢、波动大；用词表 embedding、任务关键词 embedding 或特殊 token embedding 做初始化，常更稳。

**prompt 长度** 不是越长越好。太短，容量不够；太长，序列开销上升，优化也可能更难。因为 Transformer 的注意力计算对序列长度近似是平方复杂度，输入从 $n$ 变成 $n+l$ 后，单层注意力代价近似变成 $O((n+l)^2)$。

一个经验表如下：

| prompt 长度 / 初始化 | 简单分类任务 | 序列标注任务 | 复杂生成任务 |
| --- | --- | --- | --- |
| 短长度 + 随机初始化 | 可能可用 | 常不稳 | 往往较差 |
| 中长度 + 词表采样初始化 | 常见稳妥起点 | 较稳 | 仍需观察 |
| 较长 + 任务关键词初始化 | 有时收益有限 | 常更好 | 可能仍不如 LoRA |

一个典型坑的对照：

| 设置 | 现象 | 可能原因 | 调整方向 |
| --- | --- | --- | --- |
| 长度 5，随机初始化 | loss 几乎不降 | 容量太小，起点差 | 增到 20 或 50 |
| 学习率过低 | 训练很慢 | prompt 参数少，更新幅度不够 | 提高 5 到 10 倍 |
| 学习率过高 | 验证集抖动大 | 小参数集被过冲 | 降低学习率，配合 warmup |
| 小模型上做复杂生成 | 指标明显落后 | 输入侧控制不够 | 换 LoRA 或 P-Tuning v2 |

还有几个工程细节经常被忽略：

- Prompt Tuning 的参数文件很小，这是优点，但也意味着它强依赖“底座模型版本完全一致”。换了 tokenizer、embedding 表、主模型 checkpoint，旧 prompt 很可能不能直接复用。
- 线上如果用 `inputs_embeds` 路径，要检查推理框架是否支持缓存、批处理和导出；有些服务栈对 `input_ids` 路径优化更多。
- 多任务服务时，prompt 文件管理要做版本化。因为“任务 A 的 prompt”本质上就是模型参数，不是普通配置项。

---

## 替代方案与适用边界

如果任务是“轻量适配、共享底座、快速切换”，Prompt Tuning 很合适。如果任务是“让模型学到新的推理或生成行为”，层内方法通常更可靠。

LoRA 的白话解释是“不给原权重动刀，只在层里面旁挂低秩增量矩阵”。它直接作用于注意力或 MLP 权重，所以表达力通常强于纯输入侧 prompt。Adapter 则是在每层插小模块，参数更多，但模块化管理也更成熟。

下面给一个工程化选择表：

| 方法 | 适用场景 | 参数量级 | 推理成本 | 优势 | 边界 |
| --- | --- | ---: | --- | --- | --- |
| Prompt Tuning | 大模型、多任务切换、分类/检索/轻量生成 | 很低 | 低到中 | 参数最省，便于一底座多任务 | 小模型和复杂任务弱 |
| LoRA | 中大模型、复杂生成、需要更强适配 | 低到中 | 低 | 性能通常更稳，可接近全量微调 | 比 prompt 更重 |
| Adapter | 需要强模块化管理、老系统兼容 | 中 | 中 | 插拔式明确 | 延迟与参数更高 |
| Full FT | 任务差异大、资源充足、追求上限 | 高 | 高 | 表达力最强 | 存储和训练成本最高 |

关于规模效应，可以这样理解同一趋势。以 T5 系列为例，Lester 等人的结果显示，随着模型从较小规模增大到十亿级以上，Prompt Tuning 与全量微调的差距显著缩小。换句话说，在 T5-XL 这类较大模型上，Prompt Tuning 可能仍明显落后 LoRA；到 T5-XXL 这种 11B 级别模型时，差距会进一步缩小，某些任务上接近全量微调。这不是说 soft prompt 变强了，而是大模型自身已有能力足够丰富，输入侧引导更容易奏效。

因此可以把适用边界总结成一句话：**任务越像“激活已有能力”，Prompt Tuning 越合适；任务越像“学习新内部行为”，LoRA、Adapter 或全量微调越合适。**

---

## 参考资料

| 来源 | 覆盖内容 | 重点页面 |
| --- | --- | --- |
| IBM, *What is prompt tuning?* | 面向工程实践的定义、与 LoRA/Adapter 的对比、优缺点 | https://www.ibm.com/think/topics/prompt-tuning |
| Learn Prompting, *Prompt Tuning with Soft Prompts* | 软提示定义、训练流程、为什么便于多任务切换 | https://learnprompting.org/docs/trainable/soft_prompting |
| Lester et al., 2021, EMNLP | Prompt Tuning 原始代表论文，核心结论是“随模型规模增大而更接近全量微调” | https://aclanthology.org/2021.emnlp-main.243/ |
| Emergent Mind, *Soft-Prompt Tuning Overview* | 公式化描述、参数量估算、T5-XXL 等规模讨论与实践要点 | https://www.emergentmind.com/topics/soft-prompt-tuning |
| Google Research, *prompt-tuning* | 原始实现仓库，含训练配置、初始化方式、已发布 prompt | https://github.com/google-research/prompt-tuning |

其中，IBM 适合理解方法定位；Learn Prompting 适合理解训练流程；Lester 等人的论文适合理解“规模效应”这个最关键结论；Emergent Mind 适合查公式和参数量级；Google Research 仓库适合看实际工程配置。

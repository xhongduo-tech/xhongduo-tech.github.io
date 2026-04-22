## 核心结论

Prefix Tuning 是一种参数高效微调方法：冻结基础模型参数 $\phi$，只训练一组前缀参数 $\theta$。这里的“参数高效微调”指的是：不更新大模型主体，只更新很少的额外参数，让模型适配新任务。

它的核心不是在输入文本前面拼接一句提示词，而是在 Transformer 每一层注意力中插入可学习的 prefix key/value。prefix 可以理解为“虚拟 token”，但它不是词表里的真实 token，也不会被用户直接看到。它是一段连续向量，作用在模型内部。

新手版理解可以写成：模型原本像一个固定的大脑，Prefix Tuning 不是重训大脑，而是给它装一个可更换的任务记忆卡。不同任务只换记忆卡，不换整机。

```text
冻结参数: φ
可训练 prefix 参数: θ
prefix 长度: m
```

| 方法 | 训练对象 | 参数量 | 是否改基础模型 |
|---|---:|---:|---|
| 全量微调 | 全部参数 | 大 | 是 |
| Prefix Tuning | prefix 参数 | 小 | 否 |
| Prompt Tuning | 输入层连续 prompt | 小 | 否 |

Prefix Tuning 的优势在生成任务上更明显，例如摘要、翻译、客服回复、风格化生成。它通常只训练不到 $0.1\%$ 的参数，就可能达到接近全参数微调的效果。但 $0.1\%$ 不是保证值，实际比例取决于模型大小、prefix 长度 $m$、层数、隐藏维度以及是否使用额外投影层。

---

## 问题定义与边界

Prefix Tuning 解决的问题是：如何在保留大模型通用能力的前提下，用更低训练成本适配下游任务。下游任务指具体业务任务，例如“把长文摘要成三句话”“把用户问题改写成客服回复”“把英文翻译成中文”。

普通全量微调会更新模型的大量参数。这样效果可能好，但训练成本、显存占用、模型版本管理和部署成本都会上升。一个业务训练一份完整模型，十个业务就可能产生十份大模型副本。Prefix Tuning 的目标是让基础模型保持不变，每个任务只保存一份小 prefix。

客服回复是一个典型例子。传统文本 prompt 可能把“请礼貌回答用户问题”拼到输入前面。Prefix Tuning 不这样做。它训练一个只属于客服任务的 prefix。模型表面上仍然看到原始输入，但每层注意力已经被这段内部 prefix 引导。

```text
任务目标: 以少量可训练参数适配下游任务
适用对象: 自回归生成模型、Seq2Seq 模型
不适用或需谨慎: 不能直接修改注意力结构的模型、需要极强结构可解释性的场景
```

| 维度 | Prefix Tuning | 说明 |
|---|---|---|
| 参数更新 | 少量 | 只训练 prefix |
| 输入形式 | 连续参数 | 不是离散文本 |
| 任务迁移 | 强 | 每个任务可单独保存 prefix |
| 实现要求 | 中等 | 需要能在 attention 中拼接前缀 |

边界需要明确。第一，它不是普通 prompt。普通 prompt 是人写的文本，Prefix Tuning 是训练出来的连续向量。第二，它不是 LoRA。LoRA 是在线性层上学习低秩增量，Prefix Tuning 是在注意力的 key/value 侧加前缀。第三，它不是万能压缩技术。它适合让一个已有模型学习任务偏好，但不适合让模型凭空获得预训练阶段没有学到的复杂新知识。

---

## 核心机制与推导

Transformer 是一种基于注意力机制的神经网络结构。注意力机制的作用是：让当前位置根据相关性，从上下文位置读取信息。Prefix Tuning 改变的正是这个读取过程。

先看普通第 $l$ 层注意力。设 $H^l$ 是第 $l$ 层输入隐藏状态，$W_Q^l, W_K^l, W_V^l$ 是冻结模型里的投影矩阵：

```text
Q^l = H^l W_Q^l
K^l = H^l W_K^l
V^l = H^l W_V^l
```

普通 attention 可以写成：

$$
\text{Attn}^l = \text{softmax}\left(\frac{Q^l {K^l}^T}{\sqrt{d_k}}\right)V^l
$$

其中 $d_k$ 是 key 向量维度，用来缩放点积，避免数值过大导致 softmax 过尖。

Prefix Tuning 在每层额外生成前缀 key/value：

```text
P_K^l, P_V^l = g_θ(l)
```

$g_\theta(l)$ 是 prefix 参数到第 $l$ 层 prefix 的映射函数。它可以是直接查表，也可以是一个小的 prefix encoder。插入 prefix 后，注意力变成：

```text
Attn^l = softmax( Q^l [P_K^l; K^l]^T / sqrt(d_k) ) [P_V^l; V^l]
```

这里的 `[P_K^l; K^l]` 表示把前缀 key 拼在真实 token 的 key 前面，`[P_V^l; V^l]` 表示把前缀 value 拼在真实 token 的 value 前面。softmax 会同时在 prefix 位置和真实输入位置上分配权重。只要某些查询 $Q^l$ 对 prefix key 的匹配分数高，模型输出就会更多读取 prefix value，于是生成行为被改变。

| 符号 | 含义 |
|---|---|
| `φ` | 冻结的基础模型参数 |
| `θ` | 可训练 prefix 参数 |
| `m` | prefix 长度 |
| `H^l` | 第 `l` 层输入隐藏状态 |
| `P_K^l, P_V^l` | 第 `l` 层的前缀 key/value |

玩具例子：单头、单层注意力里，设查询和 key/value 如下：

```text
q = [1, 0]
k_prefix = [2, 0], k1 = [1, 1], k2 = [0, 1]
v_prefix = [10, 10], v1 = [1, 0], v2 = [0, 2]
```

点积分数是：

```text
[q·k_prefix, q·k1, q·k2] = [2, 1, 0]
```

softmax 后约为：

```text
[0.665, 0.245, 0.090]
```

输出约为：

```text
0.665*[10,10] + 0.245*[1,0] + 0.090*[0,2] ≈ [6.90, 6.83]
```

这说明 prefix 可以明显改变注意力输出。它不是“装饰性参数”，而是直接参与每层信息读取。

在翻译任务中，Prefix Tuning 的直观效果是：模型在翻译句子前，先读到一段任务相关的隐式上下文。这段上下文告诉模型“当前任务更像翻译，而不是摘要或续写”。在 Seq2Seq 场景下，原论文设定允许 encoder 和 decoder 都加入 prefix；实际工程实现里也可能只加在 decoder。复现论文或比较实验时，必须明确这一点。

---

## 代码实现

代码层面要处理三件事：初始化 prefix、把 prefix 注入 attention、只更新 prefix 参数。概念实现是“给每层 attention 拼接前缀”，工程实现通常由 `PrefixEncoder` 生成每层的 key/value 张量。

下面是一个可运行的最小 Python 例子，用数值模拟 prefix 如何改变 attention 输出：

```python
import numpy as np

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

q = np.array([1.0, 0.0])

k_prefix = np.array([2.0, 0.0])
k1 = np.array([1.0, 1.0])
k2 = np.array([0.0, 1.0])

v_prefix = np.array([10.0, 10.0])
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 2.0])

keys = np.stack([k_prefix, k1, k2])
values = np.stack([v_prefix, v1, v2])

scores = keys @ q
weights = softmax(scores)
output = weights @ values

assert np.allclose(scores, np.array([2.0, 1.0, 0.0]))
assert weights[0] > weights[1] > weights[2]
assert output[0] > 6.8 and output[1] > 6.7

print(weights.round(3))
print(output.round(3))
```

如果用 Hugging Face PEFT，典型流程是加载基础模型、冻结主干、创建 prefix 配置，再训练可学习 prefix：

```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
```

训练循环本身和普通微调类似：

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=1e-3)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

优化器只接收 `requires_grad=True` 的参数。这个细节很重要，因为 Prefix Tuning 的前提是基础模型参数冻结。如果误把基础模型也传给优化器，并且这些参数仍然允许梯度更新，训练就变成了混合微调，参数量和实验结论都会变。

真实工程例子：一个多业务摘要系统，底座统一使用 T5 或 BART。金融摘要、客服回复、产品介绍分别训练一份 prefix。上线时服务只加载一份基础模型，根据业务路由切换 prefix 文件。这样存储、分发、回滚都更轻。业务 A 的 prefix 出问题时，只回滚业务 A 的 prefix，不需要替换整套模型。

流程可以概括为：

```text
载入预训练模型
冻结主干参数
构建 prefix encoder
训练只作用于 prefix
保存 prefix
推理时加载对应任务 prefix
```

---

## 工程权衡与常见坑

Prefix Tuning 的工程效果不只由参数量决定。prefix 长度、初始化方式、注入位置、学习率、训练步数都会影响结果。

```text
常见风险:
1. prefix 太长 -> 过拟合
2. 初始化不稳 -> 训练震荡
3. 实现差异 -> 论文结果和复现结果不一致
4. 误把 prefix 当文本 prompt -> 概念混淆
```

| 问题 | 现象 | 规避方式 |
|---|---|---|
| prefix 太长 | 训练集好、验证集差 | 缩短 `m`，加早停 |
| 初始化随机 | loss 波动大 | 使用更稳定的初始化策略 |
| 实现不一致 | 论文和代码差异大 | 明确 encoder/decoder 注入位置 |
| 训练参数泄漏 | 基座模型被更新 | 检查 `requires_grad` |
| 学习率过大 | 指标震荡或生成退化 | 降低学习率并观察验证集 |

一个常见现象是：训练 loss 降得很快，但验证集指标反而下降。这通常是过拟合信号。此时不一定是模型太弱，更可能是 prefix 太长、训练太久、数据太少或初始化不稳定。

判断标准可以更具体一些。数据很少时，优先短 prefix，例如从 5、10、20 个 virtual tokens 开始试。任务跨度较大时，可以先试中等长度，再根据验证集调长。模型很大时，不要默认 Prefix Tuning 一定更稳定，因为大模型对少量参数的变化也可能很敏感。先跑小规模验证，再扩大训练。

还有一个复现坑：encoder-decoder 模型里，prefix 加在 encoder、decoder 还是两边，会影响实验结果。原论文在 Seq2Seq 场景下可以对 encoder 和 decoder 都加 prefix；部分工程实现为了简化，可能只在 decoder 侧加 prefix。比较论文结果、复现实验或写技术报告时，必须写清楚实现口径。

---

## 替代方案与适用边界

Prefix Tuning 属于 PEFT。PEFT 是 Parameter-Efficient Fine-Tuning 的缩写，意思是参数高效微调。这个家族里还有 Prompt Tuning、LoRA、Adapter 等方法。它们都减少训练参数，但改动位置不同。

| 方法 | 改动位置 | 优点 | 局限 |
|---|---|---|---|
| Prefix Tuning | 注意力前缀 | 参数少、任务切换方便 | 实现细节敏感 |
| Prompt Tuning | 输入层 prompt | 更简单 | 对部分任务表达力有限 |
| LoRA | 线性层低秩更新 | 通常效果强、适用广 | 参数量略高 |
| Adapter | 插入小模块 | 结构清晰 | 推理路径更复杂 |

Prompt Tuning 只在输入层加入连续 prompt，通常更容易接入，但表达力可能弱于每层都注入 prefix 的方法。LoRA 在权重矩阵上学习低秩增量，适用面很广，也是当前工程里非常常见的选择。Adapter 会在模型层之间插入小模块，结构直观，但会改变推理路径。

如果任务只是轻量文本分类，LoRA 可能更直接。如果是生成任务，并且希望每个业务单独保存一份任务状态，Prefix Tuning 更适合。新手版可以理解成：不同 PEFT 方法像不同类型的外挂模块，Prefix Tuning 更像内部上下文插件。

优先考虑 Prefix Tuning 的场景包括：多任务生成、基础模型复用要求高、部署时需要频繁切换任务、不同业务希望隔离保存任务参数。不优先考虑的场景包括：只做少量任务、训练框架不方便修改 attention、业务更关心极简实现、或者团队已经有成熟的 LoRA 训练和部署链路。

结论是：Prefix Tuning 不是“更先进就一定更好”的方法，而是一种明确偏向生成任务、参数隔离和模型复用的工程选择。它的价值来自“冻结大模型、训练小 prefix、按任务切换”，不是来自神秘的提示词技巧。

---

## 参考资料

| 类型 | 来源 | 用途 |
|---|---|---|
| 论文 | Prefix-Tuning: Optimizing Continuous Prompts for Generation | 原始定义、公式、实验结论 |
| 官方源码 | XiangLi1999/PrefixTuning | 参考实现细节 |
| 官方文档 | Hugging Face PEFT Prefix tuning | 工程接入与配置 |
| 任务指南 | Prefix tuning for conditional generation | Seq2Seq 场景使用方法 |

1. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
2. [XiangLi1999/PrefixTuning](https://github.com/XiangLi1999/PrefixTuning)
3. [Hugging Face PEFT Prefix tuning](https://huggingface.co/docs/peft/main/en/package_reference/prefix_tuning)
4. [Prefix tuning for conditional generation](https://huggingface.co/docs/peft/main/task_guides/seq2seq-prefix-tuning)

阅读顺序：先看论文摘要和方法部分，再看 attention 公式和实验设置，然后看 PEFT 文档中的配置项，最后对照源码确认 encoder、decoder 和 prefix encoder 的实现差异。

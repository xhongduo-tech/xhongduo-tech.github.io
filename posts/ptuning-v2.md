## 核心结论

P-Tuning v2 是一种深层提示调优方法：把可学习的 continuous prompt 插入 Transformer 的每一层，冻结 backbone，只训练少量 prompt 参数和任务头。

这里有三个术语需要先说清楚。continuous prompt 指“连续向量形式的提示”，不是人写的自然语言句子，而是一组模型可以训练的 embedding。Transformer 是当前大语言模型和 BERT 类模型常用的神经网络结构，核心是多层 self-attention。backbone 指预训练模型主体，例如 BERT、RoBERTa、DeBERTa 的主干参数。

P-Tuning v2 的核心结论是：只在输入层加 prompt，任务信号要穿过很多层才能影响最终表示；在每一层都加 prompt，可以让任务信号更稳定地进入深层计算，因此在小模型和序列标注任务上通常比原始 Prompt Tuning 更稳。

新手版玩具例子：把一个 12 层 Transformer 想成一栋 12 层办公楼。普通 Prompt Tuning 只在一楼贴一张“今天要做命名实体识别”的说明，信息要靠员工一层一层传上去。P-Tuning v2 是每一层都贴一张说明，每层员工处理信息时都能看到任务要求，所以任务目标不容易在传递中变弱。

| 方法 | prompt 放在哪里 | backbone 是否训练 | 典型优点 | 典型短板 |
|---|---:|---:|---|---|
| 普通 Prompt Tuning | 输入层 | 冻结 | 参数少，实现简单 | 深层任务信号可能变弱 |
| P-Tuning v2 | Transformer 每一层 | 冻结 | 小模型、NLU、序列标注更稳 | 需要改每层 forward 或 attention cache |
| 全参数微调 | 不加 prompt | 训练全部参数 | 上限高，使用成熟 | 显存和存储成本高 |

一句适用场景结论：P-Tuning v2 适合“希望接近全参数微调效果，但只想训练少量参数”的 NLU 任务，尤其是分类、命名实体识别、语义角色标注、抽取式理解等任务。

---

## 问题定义与边界

P-Tuning v2 解决的问题是：提示只在输入层注入时，任务信号在深层表示中不稳定，尤其会影响小模型和 token-level 任务。

token-level 任务指“每个 token 都要预测标签”的任务，例如命名实体识别。句子“张三在北京工作”中，“张三”要标成人名，“北京”要标成地点，模型不能只理解整句语义，还要保留每个 token 的局部信息。

为什么输入层 prompt 不够？因为 Transformer 是逐层变换的。输入层 prompt 先和原始 token embedding 拼在一起，经过第一层、第二层、第三层，直到最后一层。每层都会重新混合信息。对于简单分类任务，最后只需要一个句子级表示，输入层 prompt 可能已经够用；但对于 NER、SRL 这类任务，每个 token 的边界、词性、上下文依赖都很重要，单层 prompt 的影响可能在多层传播后被稀释。

中文 NER 例子：输入“张三在北京工作”。模型要判断“张三”是人名，“北京”是地点。如果只在最前面放 prompt，后面层可能逐渐把注意力转向句子整体语义，而不是每个 token 的标签边界。P-Tuning v2 在每层都加入 prompt，相当于每层都提醒模型“当前任务是 token 标注”，所以深层表示更容易保留任务相关信息。

| 任务类型 | 是否适合 P-Tuning v2 | 原因 |
|---|---:|---|
| 文本分类 | 适合 | 参数少，效果通常接近全参微调 |
| NER | 很适合 | 需要稳定的 token 级表示 |
| SRL | 很适合 | 需要深层语义和局部结构同时保留 |
| 抽取式阅读理解 | 适合 | 需要定位 span 边界 |
| 开放式长文本生成 | 不一定适合 | 生成任务更常用 Prefix-Tuning、LoRA 或指令微调 |
| 需要改变大量领域知识的任务 | 不一定适合 | 冻结 backbone 会限制知识更新能力 |

适用任务列表包括：文本分类、自然语言推理、命名实体识别、语义角色标注、抽取式问答、关系抽取、事件抽取。

边界也要明确：P-Tuning v2 不是让模型学习新知识的万能方法。它更像是在冻结的预训练模型上学习一组任务适配参数。如果模型本身完全没有某个领域知识，只靠 prompt 参数通常很难补齐。

---

## 核心机制与推导

P-Tuning v2 的核心机制只有两个：每层注入 prompt，冻结 backbone。

设第 $l$ 层隐藏状态为 $H^l$，该层可学习 prompt 为 $P^l \in \mathbb{R}^{m_l \times d}$，其中 $m_l$ 是 prompt 长度，$d$ 是隐藏维度。把 prompt 拼到当前层隐藏状态前面：

$$
\tilde H^l = [P^l; H^l], \quad H^{l+1} = T_l(\tilde H^l; \theta_l), \quad \theta_l \text{ frozen}
$$

这里 $T_l$ 表示第 $l$ 层 Transformer，$\theta_l$ 是该层原始参数，并且被冻结。冻结的意思是训练时不更新这些参数，只让 prompt 和任务头参与梯度下降。

从 attention 视角看，也可以把 prompt 理解成每层 attention 的前缀 key/value：

$$
K' = [K_p; K], \quad V' = [V_p; V]
$$

key 是 attention 用来匹配查询的信息，value 是匹配后被加权读取的信息。给每层加 $K_p$ 和 $V_p$，等价于让每层 attention 都能读到一组任务相关的虚拟上下文。

从输入层 prompt 到 layer-wise prompt 的流程图：

```text
普通 Prompt Tuning:
输入 token
  |
[输入 prompt + token embedding]
  |
Layer 1 -> Layer 2 -> Layer 3 -> ... -> Layer L
  |
任务头

P-Tuning v2:
输入 token
  |
token embedding
  |
[prompt 1 + H1] -> Layer 1
  |
[prompt 2 + H2] -> Layer 2
  |
[prompt 3 + H3] -> Layer 3
  |
...
  |
[prompt L + HL] -> Layer L
  |
任务头
```

小型推导可以从信号路径长度理解。普通 Prompt Tuning 的任务信号从输入层进入，影响最后一层要经过 $L$ 次非线性变换。路径越长，信息越可能被重写、压缩或稀释。P-Tuning v2 在每层都引入 $P^l$，所以第 $l$ 层的任务信号只需要影响本层和后续较短路径。对于最后几层，prompt 到输出的距离很短，因此更容易直接控制最终表示。

第 3 层的例子：普通 Prompt Tuning 中，第 3 层只能接收前两层处理后的结果，prompt 信息已经被混合过。P-Tuning v2 中，第 3 层收到的是“第 3 层专用提示 + 当前隐藏状态”，所以它不是在处理裸输入，而是在带着任务提示处理当前表示。

真实工程例子：用 RoBERTa-large 做中文 NER 时，实体边界经常依赖深层上下文。例如“苹果发布新产品”和“我吃了苹果”中，“苹果”的实体类型不同。P-Tuning v2 让每层都带有 NER 任务提示，模型在底层处理字词局部特征，在高层处理语义消歧时，都能读到任务条件。

---

## 代码实现

实现 P-Tuning v2 的重点不是改 backbone 参数，而是给每一层挂可训练 prompt，并确保 backbone 全部冻结。工程结构通常包含三部分：输入层负责接收 token，每层 prompt 负责调节各层表示，任务头负责输出分类或序列标签。

简化伪代码如下：

```python
freeze(backbone)
prompts = [Prompt(m, d) for _ in range(num_layers)]

for layer_idx, layer in enumerate(backbone.layers):
    h = concat(prompts[layer_idx], h)
    h = layer(h)

logits = head(h)
loss = task_loss(logits, labels)
update(prompts, head)
```

下面是一个可运行的 Python 参数量计算例子：

```python
def p_tuning_v2_params(num_layers, prompt_len, hidden_size):
    return num_layers * prompt_len * hidden_size

def ratio(trainable, backbone):
    return trainable / backbone

# BERT-base 近似配置：12 层，hidden size 768
prompt_params = p_tuning_v2_params(
    num_layers=12,
    prompt_len=10,
    hidden_size=768,
)

bert_base_params = 110_000_000
trainable_ratio = ratio(prompt_params, bert_base_params)

assert prompt_params == 92_160
assert 0.0008 < trainable_ratio < 0.0009

print(prompt_params, trainable_ratio)
```

参数量计算表：

| 配置 | 层数 | hidden size | 每层 prompt 长度 | prompt 参数量 |
|---|---:|---:|---:|---:|
| BERT-base 玩具配置 | 12 | 768 | 10 | 92,160 |
| BERT-base 更长 prompt | 12 | 768 | 20 | 184,320 |
| RoBERTa-large 近似配置 | 24 | 1024 | 10 | 245,760 |
| RoBERTa-large 更长 prompt | 24 | 1024 | 20 | 491,520 |

如果再加一个 NER 分类头，假设标签数为 10，分类头大约是 $1024 \times 10 + 10$ 级别，和大模型 backbone 相比仍然很小。

训练流程图：

```text
输入文本
  |
Tokenizer
  |
冻结的 embedding / backbone
  |
每层读取对应 prompt
  |
每层 Transformer 计算
  |
任务头 head
  |
loss
  |
只更新：prompt 参数 + head 参数
不更新：embedding + Transformer backbone
```

新手版 BERT NER 代码关系可以理解为：输入层把中文句子变成 token embedding；每层 prompt 插入一组长度为 $m$ 的可学习向量；任务头对每个 token 输出标签，例如 B-PER、I-PER、B-LOC、O。训练时 BERT 主体不动，只更新 prompt 和分类头。

实现时常见有两种方式。第一种是直接在每层 hidden states 前拼 prompt，这种方式直观，但要处理 prompt token 对真实 token 输出的影响。第二种是在每层 attention 中注入 prefix key/value，这更接近 Prefix-Tuning 的工程实现，也更容易避免真实 token 位置被改变。

---

## 工程权衡与常见坑

P-Tuning v2 的主要收益是参数高效，但它不是零成本。每层 prompt 会增加 attention 的序列长度，prompt 越长，显存和计算越高。它通常比全参数微调省显存和存储，但不一定比所有 PEFT 方法都便宜。

PEFT 是 parameter-efficient fine-tuning 的缩写，意思是参数高效微调，只训练少量新增参数或部分参数来适配任务。

| 问题 | 现象 | 处理方式 |
|---|---|---|
| prompt 太短 | 验证集欠拟合，NER 边界不稳 | 从 10、20、50 做搜索 |
| 学习率沿用全参微调 | loss 抖动或收敛慢 | 单独搜索 prompt/head 学习率 |
| 初始化不稳定 | 不同随机种子差异大 | 多种子实验，尝试正态初始化或从 embedding 初始化 |
| 没有正确冻结 backbone | 参数量对比失真 | 打印 trainable parameters |
| prompt token 影响标签对齐 | NER 标签位置错位 | 只对真实 token 计算 loss |
| attention mask 没改对 | prompt 不可见或 padding 混乱 | 明确扩展 mask 维度 |

参数量和显存开销对比：

| 方法 | 可训练参数 | 额外显存 | 存储多个任务的成本 | 备注 |
|---|---:|---:|---:|---|
| Full FT | 最高 | 最高 | 每个任务一份完整模型 | 效果上限强 |
| Prompt Tuning | 很低 | 低 | 每个任务一份 prompt | 简单任务合适 |
| Prefix-Tuning | 低 | 中 | 每个任务一份 prefix | 生成任务常见 |
| P-Tuning v2 | 低到中 | 中 | 每个任务一份深层 prompt | NLU 更稳 |
| LoRA | 低到中 | 中 | 每个任务一份 LoRA 权重 | 通用性强 |

训练超参清单：

| 超参 | 建议关注点 |
|---|---|
| prompt 长度 | 小模型和复杂任务不要过短，可从 10、20、50 搜索 |
| 学习率 | prompt/head 常需要比全参微调更大的学习率，但必须验证 |
| batch size | 太小会导致序列标注任务波动，必要时用梯度累积 |
| warmup | 可降低训练初期不稳定 |
| 初始化 | 随机初始化简单，从词向量初始化有时更稳 |
| weight decay | 对 prompt 不一定总有收益，需要实验确认 |

对比实验必须确认 backbone 是否冻结。很多错误结论来自“P-Tuning v2 只训练 prompt”与“全参微调训练全部参数”混在一起比较，却没有打印可训练参数。最简单的检查方式是训练前统计 `requires_grad=True` 的参数名和数量。

新手版常见坑：prompt 长度设成 5，模型可能连任务条件都表达不够，NER 上会表现为实体漏标、边界错位。更稳的做法是先试 10 或 20，再看验证集 F1，而不是只看训练 loss。

---

## 替代方案与适用边界

P-Tuning v2 不是唯一参数高效微调方案。选择方法时，要同时看任务类型、显存预算、参数存储预算和效果预期。

| 方法 | 核心做法 | 适合任务 | 优点 | 局限 |
|---|---|---|---|---|
| Full FT | 更新全部模型参数 | 数据足、预算足的任务 | 上限高，简单直接 | 成本最高，任务间存储重 |
| Prompt Tuning | 只在输入层训练 prompt | 简单分类、大模型场景 | 参数极少 | 小模型和序列标注不稳 |
| Prefix-Tuning | 每层 attention 加 prefix key/value | 生成任务、条件生成 | 控制生成较自然 | NLU token 标注未必最优 |
| P-Tuning v2 | 每层加入深层 prompt | NLU、NER、SRL、抽取 | 深层任务信号稳定 | 实现比单层 prompt 复杂 |
| LoRA | 给权重矩阵加低秩增量 | 通用微调、生成和理解 | 生态成熟，效果强 | 可训练参数通常高于极短 prompt |

适用边界表：

| 条件 | 更推荐的方法 |
|---|---|
| 只做简单二分类，数据不多 | Prompt Tuning、LoRA、Full FT 都可试 |
| 做中文 NER、SRL、抽取 | P-Tuning v2 或 LoRA |
| 重点是生成长文本 | Prefix-Tuning 或 LoRA |
| 预算允许且追求最高效果 | Full FT |
| 一个 backbone 服务很多任务 | P-Tuning v2、Prefix-Tuning、LoRA |
| 需要注入大量新知识 | Full FT、继续预训练、LoRA 更合适 |

为什么不用单层 prompt？反例可以看 token 标注。假设任务是识别“华为在深圳发布新品”里的组织名和地点。输入层 prompt 告诉模型“做 NER”，但后续层为了理解句子语义，会不断压缩和重组 token 信息。到高层时，“深圳”可能主要作为事件地点参与整体语义，而不是作为需要输出标签的 token。P-Tuning v2 在每层都注入任务条件，可以让高层语义处理时仍然保留“我要对每个 token 分类”的约束。

真实工程选择可以这样判断：如果你维护一个中文信息抽取系统，backbone 是 RoBERTa-large，任务包括 NER、关系抽取、事件抽取，每个任务都想单独存一份小参数，P-Tuning v2 是合理选择。如果你维护的是一个对话生成模型，要根据用户风格生成长回复，LoRA 或 Prefix-Tuning 往往更直接。

---

## 参考资料

建议阅读顺序：先读 P-Tuning v2 论文摘要和实验表，理解为什么要做 deep prompt；再看官方源码，确认每层 prompt 的实现方式；最后读 Prefix-Tuning 和 P-Tuning v1，补齐机制背景。

1. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)  
用途：主参考，说明 P-Tuning v2 的动机、实验结论和“深层提示调优”设计。

2. [THUDM/P-tuning-v2](https://github.com/THUDM/P-tuning-v2)  
用途：官方实现参考，适合查看 NLU 任务中 prompt、backbone、任务头如何组织。

3. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)  
用途：理解 attention prefix key/value 机制，帮助理解 $K'=[K_p;K]$ 和 $V'=[V_p;V]$。

4. [GPT Understands, Too: P-Tuning for Few-Shot Learning](https://arxiv.org/abs/2103.10385)  
用途：P-Tuning v1 背景资料，用来对比从输入层 prompt 到深层 prompt 的演进。

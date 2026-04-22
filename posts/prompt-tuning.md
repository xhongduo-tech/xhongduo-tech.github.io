## 核心结论

Prompt Tuning 是一种参数高效微调方法：冻结底座模型参数 `θ`，只在输入层前面添加一段可学习的连续向量 `P`，训练时只更新这段向量，不修改模型内部结构。

这里的“参数高效微调”指只训练少量新增参数，而不是更新整个大模型。Prompt Tuning 的核心形式是：

$$
Z = [P; E(x)]
$$

其中：

- `x` 是原始输入文本。
- `E(x)` 是输入文本经过词嵌入层后的向量序列。
- `P` 是可学习的软提示。
- `[P; E(x)]` 表示把软提示拼接到原始输入向量前面。

可以把它理解成给模型装一段可训练的前缀。这段前缀不是自然语言句子，而是一组连续向量；训练时模型只会调整这组向量，不会改动大模型本体。

| 对象 | 是否训练 | 作用 | 参数规模 |
|---|---:|---|---:|
| 底座模型 `θ` | 否 | 提供原有语言理解与生成能力 | 通常是百万到千亿级 |
| 词嵌入 `E(x)` | 否 | 把输入 token 转成向量 | 属于底座模型 |
| 软提示 `P` | 是 | 为特定任务提供可学习前缀 | 通常是几千到几十万级 |

Prompt Tuning 适合“同一个大模型要适配很多任务”的场景。它的收益不是让一个小模型突然学会全新能力，而是在共享底座模型的前提下，用很少的任务参数完成快速切换。多任务部署时，每个任务只保存自己的 prompt embedding，就能避免为每个任务保存一整份模型。

---

## 问题定义与边界

Prompt Tuning 解决的问题是：在不改底座模型的前提下，让模型更好地完成某个具体任务。

它不适合被理解成“让模型从零学会新知识”。如果底座模型完全没有某类能力，只靠一段软提示通常无法补齐能力缺口。它更像是在已有能力空间里找到更适合当前任务的输入方向。

需要先区分几个容易混淆的概念。

| 方法 | 是否可训练 | 训练参数量 | 部署成本 | 适用场景 |
|---|---:|---:|---:|---|
| 手工 prompt | 否 | 0 | 极低 | 快速试验、无训练数据 |
| Prompt Tuning | 是 | 很少，只训练输入前缀 | 低 | 多任务适配、共享大模型 |
| 全参数微调 | 是 | 更新整个模型 | 高 | 数据充足、追求效果上限 |
| LoRA | 是 | 训练低秩适配矩阵 | 中低 | 需要调整模型内部表示但预算有限 |

“手工 prompt”是人写的离散文本提示，例如：

```text
请判断下面这句话的情感是 positive 还是 negative：
这个产品很好用。
```

“离散文本”指由真实词语组成、tokenizer 可以直接编码的文本。

Prompt Tuning 则不是写一句更好的任务描述，而是让系统学习一组连续向量。例如同样做文本分类，手工 prompt 是写“判断情感”；Prompt Tuning 是初始化一段虚拟 token 的 embedding，然后通过反向传播自动优化。

“虚拟 token”指没有自然语言含义、只用于占位的可训练输入位置。模型看到的不是“请判断情感”这几个字，而是若干个向量。

玩具例子：做二分类任务，输入是“电影很好看”，标签是 `positive`。手工 prompt 靠人工设计模板；Prompt Tuning 会把 `[p1, p2, ..., pm]` 拼到输入前面，让模型学会这些向量应该如何引导它输出 `positive`。

真实工程例子：一个客服系统里有“退款意图识别”“工单优先级分类”“风险文本过滤”三个任务。全参数微调需要保存三份模型；Prompt Tuning 可以冻结同一个大模型，分别训练三份软提示。上线时只切换 prompt 参数文件，不切换底座模型。

---

## 核心机制与推导

输入序列记为：

$$
x = (x_1, x_2, ..., x_n)
$$

经过词嵌入层后得到：

$$
E(x) \in \mathbb{R}^{n \times d}
$$

这里的“词嵌入”是把 token 映射成向量的过程；`n` 是输入 token 数，`d` 是模型隐藏维度。

Prompt Tuning 额外引入一组可学习参数：

$$
P \in \mathbb{R}^{m \times d}
$$

其中 `m` 是虚拟 token 数，`d` 必须和模型隐藏维度一致。拼接后的输入是：

$$
Z = [P; E(x)] \in \mathbb{R}^{(m+n) \times d}
$$

训练目标可以写成：

$$
\min_P L(\theta, P) = -\log p_\theta(y \mid [P; E(x)])
$$

关键点是：`θ` 固定，只优化 `P`。

伪流程如下：

```text
输入文本 x
  -> tokenizer 得到 token ids
  -> embedding 层得到 E(x)
  -> 拼接软提示 P，得到 [P; E(x)]
  -> 冻结模型 θ 做前向传播
  -> 计算任务损失 L
  -> 反向传播
  -> 只更新 P，不更新 θ
```

参数量计算很直接：

$$
\text{prompt 参数量} = m \times d
$$

玩具例子：设隐藏维度 `d = 4`，软提示长度 `m = 3`。那么软提示参数只有：

$$
3 \times 4 = 12
$$

个标量。

如果输入句子只有 2 个 token，原始嵌入序列是：

```text
[e1, e2]
```

加入软提示后，模型实际看到的是：

```text
[p1, p2, p3, e1, e2]
```

序列长度从 2 变成 5。假设模型对标签的输出概率为：

$$
p(positive)=0.69,\quad p(negative)=0.31
$$

若真实标签是 `positive`，损失为：

$$
L = -\log(0.69) \approx 0.37
$$

反向传播会根据这个损失调整 `p1, p2, p3`，但不会调整底座模型内部的注意力层、前馈层或词嵌入矩阵。

这也是 Prompt Tuning 和 Prefix Tuning 的一个重要区别。Prompt Tuning 通常只在输入 embedding 层添加软提示；Prefix Tuning 则更常把可学习前缀注入 Transformer 每一层的 key/value 中，影响位置更深，参数量也通常更多。

---

## 代码实现

工程实现通常分三步：

1. 冻结底座模型参数。
2. 初始化 soft prompt embedding。
3. 在前向传播时拼接 soft prompt，只训练它。

下面是一个不依赖大模型库的最小 Python 例子，用线性分类器模拟“冻结模型 + 可训练 prompt”。它不是完整语言模型，但能准确展示 Prompt Tuning 的训练边界：底座参数不变，只更新 prompt。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

# toy setting
batch_size = 4
seq_len = 2
prompt_len = 3
hidden_dim = 4
num_classes = 2

# frozen base model: mean pooling + linear classifier
class FrozenToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs_embeds):
        pooled = inputs_embeds.mean(dim=1)
        return self.classifier(pooled)

model = FrozenToyModel()

# freeze model
for param in model.parameters():
    param.requires_grad = False

# init prompt embeddings
prompt = nn.Parameter(torch.randn(prompt_len, hidden_dim) * 0.02)

optimizer = torch.optim.SGD([prompt], lr=0.5)

# fake input embeddings E(x), usually produced by a tokenizer + embedding layer
x_embeds = torch.randn(batch_size, seq_len, hidden_dim)
labels = torch.tensor([1, 0, 1, 0])

before_weight = model.classifier.weight.detach().clone()
before_prompt = prompt.detach().clone()

# concat: [P; E(x)]
prompt_batch = prompt.unsqueeze(0).expand(batch_size, -1, -1)
inputs = torch.cat([prompt_batch, x_embeds], dim=1)

logits = model(inputs)
loss = F.cross_entropy(logits, labels)

loss.backward()
optimizer.step()

# base model unchanged
assert torch.allclose(model.classifier.weight, before_weight)

# prompt updated
assert not torch.allclose(prompt.detach(), before_prompt)

# sequence length changed from 2 to 5
assert inputs.shape == (batch_size, prompt_len + seq_len, hidden_dim)

print("loss:", round(loss.item(), 4))
print("prompt parameters:", prompt.numel())
```

真实工程中，分类任务常不直接接一个分类头，而是把标签映射成文本 token，再用生成式损失训练。这个映射叫 verbalizer，白话说就是“把类别名转成模型能生成的词”。例如：

| 分类标签 | verbalizer 文本 | 说明 |
|---|---|---|
| 正向情感 | `positive` | 模型生成 positive 表示正类 |
| 负向情感 | `negative` | 模型生成 negative 表示负类 |
| 退款意图 | `refund` | 适合客服意图分类 |
| 投诉意图 | `complaint` | 需要和 tokenizer 对齐 |

新手版训练流程可以理解为：先创建 20 个虚拟 token 的 embedding，把它们接到每个句子前面，然后冻结大模型，只训练这 20 个 token 对应的向量。训练完成后，不需要保存整个模型，只保存这组 prompt embedding。

伪代码如下：

```text
load pretrained model
freeze model parameters

init prompt embeddings P with shape [num_virtual_tokens, hidden_size]

for batch in dataloader:
    token_ids = tokenizer(batch.text)
    input_embeddings = model.embedding(token_ids)

    prompt_embeddings = expand(P, batch_size)
    inputs = concat(prompt_embeddings, input_embeddings)

    outputs = model(inputs_embeds=inputs, labels=batch.label_tokens)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

save P
```

工程上要特别注意 tokenizer 和 label 对齐。比如 `positive` 可能是一个 token，也可能被切成多个 token。标签 token 数不一致时，损失位置和 mask 处理都要明确，否则训练信号会偏。

---

## 工程权衡与常见坑

Prompt Tuning 的参数少，但不代表调参简单。软提示长度、初始化方式、标签映射和数据分布都会显著影响结果。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| prompt 长度太短 | 容量不足，验证集效果上不去 | 从 10、20、50、100 个 virtual tokens 做搜索 |
| prompt 长度太长 | 训练集很好，验证集变差 | 用验证集选长度，不盲目加大 |
| 初始化方式不合适 | 收敛慢，结果波动大 | 优先尝试文本初始化或词表采样初始化 |
| verbalizer 不一致 | 标签语义和 token 不匹配 | 固定标签词，并检查 tokenizer 切分 |
| 验证集选择错误 | 线上掉点但离线很好 | 验证集要覆盖真实业务分布 |
| 分布漂移 | 换业务线后效果明显下降 | 做跨域验证，必要时为新域单独训练 prompt |

“分布漂移”指训练数据和线上数据的统计特征发生变化。例如训练集主要来自电商退款，线上突然加入本地生活服务退款，用户表达方式、商品类型、处理流程都变了。

真实工程例子：退款意图识别在训练集上准确率很高，但换到新业务线后掉点。常见原因不是底座模型坏了，而是 prompt 学到的任务引导过度贴合旧业务数据。比如旧业务里“退货”“退款”“不想要了”很常见，新业务里用户更多说“预约取消”“服务没做”“费用退回”。这时需要重新构造验证集，或者为新业务线单独训练 prompt。

Prompt Tuning 对数据量也有要求。手工 prompt 可以零样本使用，但软提示是可训练参数，需要标注数据提供梯度。如果数据非常少，随机初始化的软提示可能学不到稳定方向。论文和实践中都观察到，随着模型规模增大，Prompt Tuning 的效果更接近全参数微调；但在小模型、少数据、强跨域场景下，它未必占优。

还要注意部署链路。虽然只保存 prompt 参数很轻，但线上推理必须保证底座模型版本、tokenizer 版本、prompt 长度、embedding 维度完全一致。底座模型升级后，旧 prompt 不一定还能保持原有效果，因为它依赖的是旧模型的向量空间。

---

## 替代方案与适用边界

Prompt Tuning 不是万能方案。它更适合任务多、参数预算紧、希望共享底座模型的环境。典型场景是多业务线文本分类：共享一个大模型，每个业务线保存一份很小的 prompt 参数。

如果任务数据很少、任务跨度很大，或者需要明显改写模型内部能力，Prompt Tuning 的上限可能不够。此时可以考虑 Prefix Tuning、LoRA 或全参数微调。

| 方法 | 改动位置 | 参数量 | 训练成本 | 效果上限 | 适用边界 |
|---|---|---:|---:|---:|---|
| Prompt Tuning | 输入层 soft prompt | 很低 | 低 | 中到高，依赖模型规模 | 多任务轻量适配 |
| Prefix Tuning | 多层 attention 的前缀 key/value | 低到中 | 中 | 通常高于简单 Prompt Tuning | 生成任务、需要更强控制 |
| LoRA | 注意力或线性层的低秩矩阵 | 中低 | 中 | 高 | 需要改变内部表示 |
| 全参数微调 | 整个模型 | 最高 | 高 | 通常最高 | 数据充足、效果优先 |

“低秩矩阵”指用两个较小矩阵近似一个大矩阵的更新，从而减少训练参数。LoRA 的思路是冻结原权重，只训练额外的低秩更新项。

选择时可以按问题边界判断：

| 场景 | 更推荐 |
|---|---|
| 无训练数据，只想快速试 | 手工 prompt |
| 很多相似分类任务，共享大模型 | Prompt Tuning |
| 生成任务需要更强格式控制 | Prefix Tuning 或 LoRA |
| 任务和底座能力差距较大 | LoRA 或全参数微调 |
| 数据充足且追求最高指标 | 全参数微调 |
| 线上存储和发布成本极敏感 | Prompt Tuning |

保守预期很重要。比如一个非常小的数据集只有几十条样本，却希望 Prompt Tuning 追平全参数微调，通常不现实。此时应该先尝试更强的手工任务描述、少量示例、数据增强，或者使用 LoRA 这类能影响模型内部层的方案。

Prompt Tuning 的价值不在于单任务一定最强，而在于系统层面的效率：一个底座模型，多个任务 prompt，低成本训练，低成本保存，快速切换。

---

## 参考资料

1. [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
2. [Google Research prompt-tuning GitHub Repository](https://github.com/google-research/prompt-tuning)
3. [Hugging Face PEFT Soft Prompts Conceptual Guide](https://huggingface.co/docs/peft/conceptual_guides/prompting)
4. [Hugging Face PEFT Prompt Tuning API Reference](https://huggingface.co/docs/peft/en/package_reference/prompt_tuning)
5. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

## 核心结论

困惑度（Perplexity, PPL）衡量语言模型对“下一个 token”的平均不确定性。token 是模型处理文本的最小单位，可以是字、词、子词或符号片段。PPL 的本质是平均负对数似然的指数形式。

$$
\mathrm{PPL}(x_{1:T})=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})\right)
$$

关键结论是：`PPL 越低`，表示模型给真实下一个 token 的预测概率越高，通常说明模型在这份文本上的语言建模能力更好。但 PPL 不等于下游任务能力。一个模型 PPL 更低，不必然说明它更会问答、更会推理、更会遵守指令。

新手版理解：如果模型每一步都更倾向于猜对答案，那么平均下来“需要排除的候选答案数”更少，PPL 就更低。比如看到“我今天去”，真实下一个 token 是“学校”，模型 A 给“学校”0.4 概率，模型 B 只给 0.1 概率，那么这一位置上 A 的损失更低。

解释版理解：两个模型都能生成通顺句子，但在验证集上，A 的 PPL 比 B 低，表示 A 对测试文本的逐 token 预测更自信、更接近真实序列。

| 指标 | 含义 | 越大/越小更好 |
|---|---|---|
| PPL | 平均不确定性 | 越小越好 |
| 交叉熵 | 平均负对数似然 | 越小越好 |
| 准确率 | 是否猜中某个离散答案 | 不适合直接替代 PPL |

---

## 问题定义与边界

PPL 主要定义在自回归语言模型（autoregressive language model）上。自回归语言模型是按顺序生成文本的模型，每一步根据前面的上下文预测下一个 token。它的核心对象是序列条件概率：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

这里的 \(x_{<t}\) 表示第 \(t\) 个 token 之前的所有上下文。PPL 评估的就是模型是否把较高概率分配给真实出现的下一个 token。

新手版例子：模型看到“我今天去”后，要预测下一个词是“了”“学校”“吃饭”还是别的。PPL 关注的不是模型最后生成了哪个词，而是它给真实下一个 token 分配了多大概率。

PPL 的边界也很明确。它评估的是概率分布质量，不是语义理解能力、对话能力，也不是生成质量的完整刻画。一个模型可以在困惑度上表现很好，但在问答、数学推理、工具调用、指令跟随上仍然不强，因为这些能力不只由 token 级预测决定。

| 项目 | 适用对象 | 不适用对象 |
|---|---|---|
| PPL | causal LM / autoregressive LM | 典型 masked LM 直接套用 |
| token 级预测 | 序列概率评估 | 语义理解总分 |
| 验证集/测试集 | 模型比较 | 单句主观好坏判断 |

masked LM 是“遮住部分 token 再预测”的模型，例如 BERT。它不是从左到右生成完整序列，因此不能把 causal LM 的 PPL 定义直接搬过去比较。

---

## 核心机制与推导

PPL 来自交叉熵与负对数似然。交叉熵在这里可以理解为：模型对真实答案分配的概率越低，惩罚越大。负对数似然是训练语言模型时常用的损失形式，真实 token 概率越高，损失越小。

推导链条分三步。

第一步，序列概率分解：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

第二步，取对数并求平均负值：

$$
L=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

第三步，对平均损失取指数：

$$
\mathrm{PPL}=e^L
$$

如果使用自然对数 \(\ln\)，损失单位是 `nats/token`。如果使用 \(\log_2\)，单位是 `bits/token`。底数不同，数值尺度不同，但含义一致，都是衡量每个 token 的平均预测难度。

| 符号 | 含义 |
|---|---|
| \(x_{1:T}\) | 长度为 \(T\) 的 token 序列 |
| \(p_\theta\) | 参数为 \(\theta\) 的语言模型 |
| \(x_{<t}\) | 第 \(t\) 个 token 之前的上下文 |
| \(\log p_\theta(x_t\mid x_{<t})\) | 真实 token 的对数概率 |

玩具例子：假设一段文本只有两个需要预测的位置，模型对真实 token 的概率分别是 `0.5` 和 `0.25`。

$$
L=-\frac{\ln 0.5+\ln 0.25}{2}=1.0397
$$

$$
\mathrm{PPL}=e^{1.0397}\approx 2.83
$$

这可以解释为：模型平均每一步大约像是在 2.83 个候选中选择真实 token。这个解释只是直观近似，不表示候选集合真的只有 2.83 个。

机制版例子：若某个 token 的真实概率从 `0.1` 提升到 `0.2`，对应损失从 \(-\ln 0.1\approx2.30\) 降到 \(-\ln 0.2\approx1.61\)。该位置损失下降，整段文本的平均损失下降，PPL 也会下降。

---

## 代码实现

训练和评估时，PPL 通常由 `CrossEntropyLoss` 间接计算。logits 是模型输出的未归一化分数，softmax 后才是概率分布。PyTorch 的 `cross_entropy` 等价于 `LogSoftmax + NLLLoss`，会取出真实标签对应位置的负对数概率并求平均。

新手版理解：如果模型输出的是每个位置上所有词的打分 `logits`，那就先让正确答案位置的概率尽量大，再把整段平均损失取指数，就得到 PPL。

```python
import math
import torch
import torch.nn.functional as F

# batch=1, seq_len=3, vocab_size=3
# labels 中 -100 表示该位置不参与 loss 和 PPL 计算
logits = torch.tensor([[
    [2.0, 0.0, 0.0],
    [0.0, 1.5, 0.0],
    [0.0, 0.0, 3.0],
]])

labels = torch.tensor([[0, 1, -100]])

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)

ppl = math.exp(loss.item())

print("loss:", loss.item())
print("ppl:", ppl)

assert loss.item() > 0
assert ppl > 1
assert abs(ppl - math.exp(loss.item())) < 1e-8
```

真实工程例子：做 causal LM 评估时，通常在验证集或测试集上算 PPL，用于模型选择、早停或比较不同 checkpoint。对固定上下文长度模型，不能简单把长文本切成互不重叠的块直接算。因为每个块开头的 token 会失去前文上下文，模型本来能利用的信息被人为删除，PPL 会被抬高。

更合理的做法是滑动窗口。滑动窗口是用重叠片段评估长文本的方法：每次给模型尽可能多的历史上下文，但只统计当前新 token 的 loss。

| 场景 | 做法 | 原因 |
|---|---|---|
| 长文本评估 | 滑动窗口 | 保留尽可能多的上下文 |
| 定长模型 | 分块重叠 | 避免上下文缺失 |
| 有 padding 的 batch | mask 掉无效 token | 防止损失被污染 |

还要注意 `label shift`。在 causal LM 中，第 \(t\) 个位置的 logits 通常用于预测第 \(t+1\) 个 token，所以实现时要确认 logits 和 labels 是否已经对齐。不同框架可能在模型内部做 shift，也可能要求调用方自己处理。

---

## 工程权衡与常见坑

PPL 是基础指标，但它很容易被误用。

第一，不同分词器下的 PPL 不可直接横向比较。分词器是把原始文本切成 token 的规则。同一句话，如果一个模型按字切，另一个按词切，token 数和每一步预测空间都变了，PPL 的单位不再一致。

第二，PPL 低不代表所有任务都强。验证集 PPL 下降了，但问答准确率没提升，说明模型可能更会“猜下一个 token”，但不一定更会完成任务。指令跟随、对话安全、复杂推理、代码执行正确性，都需要额外指标。

第三，padding 和 `-100` 必须正确处理。padding 是为了把不同长度样本拼成 batch 而补上的无效 token。如果把 padding 也算进 loss，平均损失会被污染。`-100` 是 PyTorch 里常用的忽略标记，表示该位置不参与交叉熵计算。

| 坑点 | 结果 | 规避方式 |
|---|---|---|
| 直接比较不同分词器 | 结论失真 | 固定 tokenizer 再比较 |
| 把 masked LM 当 causal LM 评估 | 指标不可比 | 使用对应模型定义 |
| 忽略 `-100` 和 padding | 平均损失被污染 | 正确 mask |
| 固定长度模型硬切块 | PPL 偏高 | 用滑动窗口 |
| 只看 PPL 不看任务指标 | 选错模型 | 联合看 downstream metrics |

工程上，PPL 更适合回答“这个模型在这份文本分布上预测下一个 token 的能力是否更好”。它不适合单独回答“这个模型是否更适合上线”。

---

## 替代方案与适用边界

PPL 适合做语言建模能力的基础比较，尤其适合预训练阶段、模型规模实验、验证集选择、数据清洗策略对比。但它不应该作为唯一指标。

新手版理解：你可以把 PPL 理解为“语言预测能力的基础分”。真正上线时，还要看模型会不会答题、会不会遵守指令、会不会输出安全内容、响应是否够快、成本是否可接受。

在摘要、问答、代码生成任务里，PPL 只能提供背景信息。摘要更关心内容覆盖和可读性，问答更关心答案是否正确，代码生成更关心能不能通过测试。此时要用任务指标补充 PPL。

| 指标 | 适用场景 | 与 PPL 的关系 |
|---|---|---|
| 准确率 / EM | 分类、抽取、QA | 直接任务指标 |
| ROUGE / BLEU | 摘要、翻译 | 结果相似度 |
| pass@k | 代码生成 | 任务成功率 |
| 人工评审 | 对话、写作 | 更接近真实体验 |
| 延迟 / 成本 | 在线部署 | 生产约束 |

场景版例子：如果训练一个客服问答模型，验证集 PPL 降低说明模型更贴近训练语料分布。但上线前仍要检查回答正确率、拒答边界、安全策略、响应延迟和单位请求成本。PPL 可以参与模型筛选，但不能替代完整评估。

---

## 参考资料

| 类型 | 来源 | 用途 |
|---|---|---|
| 经典定义 | Shannon, 1951 | 信息论视角 |
| 神经语言模型 | Bengio et al., 2003 | 序列概率建模 |
| 工程评估 | Hugging Face Transformers | 固定长度模型与滑动窗口 |
| 损失函数 | PyTorch `CrossEntropyLoss` | 训练与评估实现 |
| 指标实现 | Keras `Perplexity` | 交叉熵取指数的实践说明 |

1. [Prediction and Entropy of Printed English](https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/)
2. [A Neural Probabilistic Language Model](https://proceedings.neurips.cc/paper_files/paper/2000/hash/728f206c2a01bf572b5940d7d9a8fa4c-Abstract.html)
3. [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/v4.32.0/perplexity)
4. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html)
5. [Keras Hub Perplexity metric](https://keras.io/keras_hub/api/metrics/perplexity/)

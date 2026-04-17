## 核心结论

Tie Word Embeddings 的定义很直接：把输入词向量矩阵和输出层的词分类矩阵绑定成同一个参数。词向量矩阵可以理解为“把 token id 翻译成向量的查表本”；输出分类矩阵可以理解为“把隐藏状态映射回整个词表分数的打分表”。绑定之后，模型“读词”和“预测词”使用同一套词表示。

它解决的核心问题是大词表下的参数冗余。标准语言模型通常有两份与词表同规模的大矩阵，一份给输入 embedding，一份给输出 projection。词表大小记为 $V$，词向量维度记为 $d$，那么这两部分的参数量通常是 $2Vd$；如果共享，立刻降到 $Vd$。当 $V$ 很大时，这不是小优化，而是直接减少一整份大矩阵。

更重要的是，这种共享不是“把两份参数初始化成一样”，而是“让它们本来就是同一个参数对象”。前者训练几步后就会分叉，后者才是真正的 tie。工程上常见写法是输入侧仍然做查表，输出侧改成用隐藏状态与同一个 embedding 矩阵做点积，再加偏置项。

可以把它概括成一句话：Tie Word Embeddings 用一份词表矩阵同时承担输入编码和输出打分，主要收益是省参数、省显存、省 checkpoint 体积，并让输入空间和输出空间保持一致；代价是可能限制输出头的独立表达能力，在小模型或任务不匹配时可能损失效果。

---

## 问题定义与边界

先看标准语言模型在做什么。语言模型的目标是根据上下文预测下一个 token。token 可以理解为“词、子词或符号的编号单位”。这个任务至少涉及两步：

1. 把输入 token id 变成连续向量。
2. 把模型最后得到的隐藏状态变成对整个词表的打分。

如果这两步各自维护一套独立矩阵，就会出现重复。

| 项目 | 不共享 | 共享 |
|---|---:|---:|
| 输入嵌入 | $V \times d$ | $V \times d$ |
| 输出投影 | $V \times d$ | 与输入共享 |
| 总参数 | $2Vd$ | $Vd$ |

玩具例子先看最小情况。假设词表只有 4 个词，向量维度是 2：

| token | id | 向量 |
|---|---:|---|
| cat | 0 | $[1, 0]$ |
| dog | 1 | $[0, 1]$ |
| pet | 2 | $[1, 1]$ |
| runs | 3 | $[-1, 1]$ |

如果模型读入 `dog`，输入侧会查到 $[0,1]$。到了输出侧，如果隐藏状态是 $[2,1]$，模型可以用同一个矩阵里每个词向量与它做点积，得到所有词的分数。也就是说，模型不是“用一套字典读词，再用另一套字典猜词”，而是同一本字典同时负责两件事。

真实工程边界比这个例子更重要。Tie 不是任何模型都能直接套上，通常需要满足以下条件：

| 条件 | 说明 |
|---|---|
| 输出目标与输入词表一致 | 典型场景是 next-token prediction |
| 隐藏状态维度等于 embedding 维度 | 即 $H=d$ 时可直接共享 |
| 若 $H \ne d$ | 需要先加投影层把隐藏状态映射到 $d$ |
| 词表稳定 | 改 tokenizer 或 vocab 后旧权重通常不能直接复用 |

这里的“投影层”就是一个额外线性层，作用是把一种维度的向量变到另一种维度。它不改变 token 语义，只做维度对齐。

一个真实工程例子：设词表 $V=50{,}000$，维度 $d=768$。单份矩阵参数量是

$$
Vd = 50{,}000 \times 768 = 38{,}400{,}000
$$

两份就是 7680 万参数。若使用 fp32，每个参数 4 字节，那么节省的内存约为

$$
38.4\text{M} \times 4 \approx 153.6\text{MB}
$$

这部分节省会直接反映到模型权重大小、加载时间和显存占用上。

---

## 核心机制与推导

先定义符号。设词表大小为 $V$，embedding 维度为 $d$，输入词向量矩阵为

$$
E \in \mathbb{R}^{V \times d}
$$

这里的第 $i$ 行表示第 $i$ 个 token 的向量。

### 1. 输入侧：查表

当输入 token 为 $x_t$ 时，模型取出第 $x_t$ 行：

$$
e_t = E[x_t]
$$

这一步叫 embedding lookup，也就是“按 id 查表取向量”。

### 2. 输出侧：打分

如果模型最后得到的隐藏状态 $h_t \in \mathbb{R}^d$，那么未共享时通常写成：

$$
o_t = h_t W_{\text{out}}^T + b
$$

其中 $W_{\text{out}} \in \mathbb{R}^{V \times d}$ 是独立的输出矩阵，$b \in \mathbb{R}^{V}$ 是偏置。$o_t$ 是 logits，也就是“softmax 之前的原始分数向量”。

共享后，直接把 $W_{\text{out}}$ 换成 $E$：

$$
o_t = h_t E^T + b
$$

于是第 $i$ 个词的分数就是

$$
o_{t,i} = h_t \cdot E_i + b_i
$$

其中 $E_i$ 表示词表中第 $i$ 个词的向量。直观解释是：隐藏状态和哪个词向量更接近，哪个词得分就更高。最后再经过 softmax 变成概率：

$$
p(y_t=i \mid h_t) = \frac{\exp(o_{t,i})}{\sum_{j=1}^{V}\exp(o_{t,j})}
$$

### 3. 为什么这样做在语义上说得通

如果 embedding 空间已经学会了“哪些词彼此相近”，那么输出预测时继续使用这套空间，等于要求模型在同一个语义坐标系里完成输入和输出。这种约束往往是合理的，因为语言建模的输入词和输出词来自同一个词表。

### 4. 维度不一致时怎么办

如果隐藏状态维度是 $H$，而不是 $d$，就不能直接相乘。此时需要先投影：

$$
z_t = h_t P,\quad P \in \mathbb{R}^{H \times d}
$$

再做共享输出：

$$
o_t = z_t E^T + b
$$

所以真正的限制不是“模型不能 tie”，而是“最终送入输出头的向量维度必须和 embedding 维度一致”。

### 5. 参数量推导

未共享时，与词表直接相关的两大块参数是：

$$
Vd + Vd = 2Vd
$$

共享后变成：

$$
Vd
$$

也就是省掉一整份 $V \times d$ 矩阵。这也是它在大词表模型里最常见的原因。

---

## 代码实现

代码层面最重要的一点是：必须共享同一个参数对象，而不是复制数值。

下面先给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只用 `numpy` 展示输入查表和输出共享的数学过程。

```python
import numpy as np

# 词表大小 V=4, 维度 d=2
E = np.array([
    [1.0, 0.0],   # cat
    [0.0, 1.0],   # dog
    [1.0, 1.0],   # pet
    [-1.0, 1.0],  # runs
])

vocab = ["cat", "dog", "pet", "runs"]

def embed(token_id: int) -> np.ndarray:
    return E[token_id]

def tied_logits(hidden_state: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    logits = hidden_state @ E.T
    if bias is not None:
        logits = logits + bias
    return logits

# 输入 token: dog
x_t = 1
e_t = embed(x_t)
assert np.allclose(e_t, np.array([0.0, 1.0]))

# 隐藏状态
h_t = np.array([2.0, 1.0])
logits = tied_logits(h_t)

# 计算结果：[2, 1, 3, -1]
assert np.allclose(logits, np.array([2.0, 1.0, 3.0, -1.0]))

pred_id = int(np.argmax(logits))
assert vocab[pred_id] == "pet"

print("embedding:", e_t)
print("logits:", logits)
print("prediction:", vocab[pred_id])
```

这个例子说明了 tie 的本质：输入 `dog` 用的是 `E[1]`，输出打分也用的是同一个 `E` 的转置。

如果用 PyTorch，核心代码通常类似下面这样：

```python
import torch
import torch.nn as nn

class TiedLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        # 关键点：不是复制值，而是绑定同一权重对象
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, hidden_states):
        x = self.embed(input_ids)
        logits = self.lm_head(hidden_states)
        return x, logits

model = TiedLM(vocab_size=100, d_model=16)
assert model.lm_head.weight is model.embed.weight
```

如果隐藏状态维度和 embedding 维度不一致，要加投影层：

```python
import torch
import torch.nn as nn

class TiedLMWithProj(nn.Module):
    def __init__(self, vocab_size: int, h_dim: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(h_dim, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, hidden_states):
        x = self.embed(input_ids)
        z = self.proj(hidden_states)
        logits = self.lm_head(z)
        return x, logits
```

参数绑定关系可以总结成下面这张表：

| 组件 | 参数 | 是否共享 |
|---|---|---|
| Embedding | $E$ | 是 |
| Output head | $E^T$ | 是 |
| Projection | $P$ | 否 |
| Bias | $b$ | 否 |

真实工程例子可以用 GPT 类模型来理解。decoder-only 模型就是“只根据左侧上下文预测右侧 token 的模型”。它天然满足输入词表和输出词表一致，因此非常适合 tie。很多实现会在配置里提供 `tie_word_embeddings` 之类的选项，本质就是上面这件事。

---

## 工程权衡与常见坑

Tie 带来的收益很明确，但它不是无条件更优。

### 1. 收益

| 收益 | 原因 |
|---|---|
| 参数更少 | 少了一整份 $V \times d$ 矩阵 |
| checkpoint 更小 | 权重文件直接减少 |
| 显存占用更低 | 训练和推理都更轻 |
| 输入输出语义更一致 | 编码和解码在同一词空间中完成 |

对于大词表模型，这些收益通常很实在。仍以 $V=50{,}000, d=768$ 为例，少掉 3840 万参数。即使只看权重本身，也能节省约 153.6MB 的 fp32 存储；若考虑优化器状态，训练时实际节省通常更大。

### 2. 代价

输出头原本有独立参数，可以自由学习“怎样从隐藏状态映射到词表”。共享后，这个自由度被收紧了。简单说，模型被要求“输入怎么看词，输出也怎么区分词”。如果这个约束与任务结构匹配，通常是好事；如果模型本来容量就不足，或者输出头需要更强的独立表达，性能可能下降。

### 3. 常见坑

| 问题 | 表现 | 处理方式 |
|---|---|---|
| $H \ne d$ 直接 tie | shape mismatch | 先加投影层 |
| 只复制初值 | 训练后两套权重分叉 | 绑定同一参数对象 |
| tokenizer 变化 | 加载 checkpoint 时报错或语义错乱 | 重建词表并重新适配 |
| 共享后效果变差 | perplexity 上升 | 保留 bias、加 projection、必要时取消共享 |

这里的 perplexity 可以理解为“语言模型预测不确定性的常用指标，越低通常越好”。

一个很常见的误区是：代码里写了 `lm_head.weight.data.copy_(embed.weight.data)`，以为这就算共享。不是。这个写法只是在某个时间点把数值复制过去，之后两者仍然各自更新。真正的共享必须检查对象身份，例如在 PyTorch 中用：

```python
assert model.lm_head.weight is model.embed.weight
```

另一个坑是词表更新。比如旧模型用的是 50,000 个 token，新 tokenizer 扩成 52,000 个 token，那么原来的共享矩阵尺寸就变了，不能无缝沿用。即使强行截断或补零，也未必保留原有语义结构。

工程上的检查清单可以直接写成：

| 检查项 | 为什么要查 |
|---|---|
| 权重是否是同一对象 | 防止“假共享” |
| 输出层是否保留 bias | 某些模型需要它稳定训练 |
| tokenizer 与 vocab 是否一致 | 防止权重错位 |
| 是否监控 perplexity | 判断共享是否伤害效果 |

---

## 替代方案与适用边界

Tie Word Embeddings 属于参数复用，但它不是唯一的省参方法，也不是所有任务都应该优先使用。

先看对照表：

| 方案 | 省参位置 | 核心思路 | 适用边界 |
|---|---|---|---|
| Tie embeddings | 输入/输出词表矩阵 | 共享同一矩阵 | 语言建模、词表稳定 |
| Factorized embeddings | 词嵌入参数 | 用低秩分解减少大矩阵参数 | 超大词表、大模型 |
| Layer sharing | Transformer 层 | 多层复用同一层参数 | 轻量化模型 |
| 独立输出头 | 输出层 | 不共享，保留最大自由度 | 更重视表达能力 |

这里要特别澄清一个常见混淆。ALBERT 的确和“参数共享”有关，但它的代表性设计不只是 tie embeddings。它更核心的是 factorized embedding parameterization 和跨层参数共享。前者是把原本巨大的 embedding 参数拆成更节省的结构，后者是多个 Transformer 层复用参数。它们和 tie embeddings 属于同一大类思路，但不是同一件事。

适用建议可以归纳成三条：

1. 如果任务是标准语言建模，输入输出词表相同，且词表很大，优先考虑 tie。
2. 如果隐藏状态和 embedding 维度不一致，但又想省参，可以加投影层再 tie。
3. 如果模型容量已经偏紧，或者实验发现共享后困惑度明显变差，就不应机械坚持共享。

反过来说，以下场景通常不适合直接套用：

| 场景 | 原因 |
|---|---|
| 输入词表与输出词表不同 | 没有同一矩阵可共享 |
| 输出不是词表分类 | 例如分类标签空间与输入 token 无关 |
| 任务要求输出头强独立性 | 共享可能压缩过度 |
| 高频修改 tokenizer | 权重复用成本高 |

所以 tie 不是“更先进”的固定答案，而是一个有明确前提的工程选择：当输入和输出确实在同一语义空间中，且大词表参数开销值得被压缩时，它通常是高性价比方案。

---

## 参考资料

| 资料 | 作用 |
|---|---|
| Press & Wolf, 2017, Using the Output Embedding to Improve Language Models, https://aclanthology.org/E17-2025/ | 经典论文，系统讨论输入输出 embedding 共享 |
| Inan, Khosravi & Socher, 2016/2017, Tying Word Vectors and Word Classifiers, https://huggingface.co/papers/1611.01462 | 独立提出相关思路，从损失与词向量角度解释共享 |
| OpenAI GPT-2 官方代码 `src/model.py`, https://github.com/openai/gpt-2/blob/master/src/model.py | 真实工程实现，可观察输出头与词嵌入的关联方式 |
| Google Research ALBERT 官方仓库, https://github.com/google-research/albert | 用于对比“参数共享”在更大结构层面的做法 |
| Hugging Face `tie_word_embeddings` 相关配置文档, https://huggingface.co/transformers/v4.7.0/_modules/transformers/configuration_utils.html | 框架层如何暴露和管理该配置项 |

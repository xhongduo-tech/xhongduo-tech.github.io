## 核心结论

Embedding 与 Softmax 共享，指输入词向量矩阵和输出分类矩阵共用同一组参数。更准确地说，模型用同一张词表矩阵完成两件事：

- 输入阶段：把 token id 查成向量。
- 输出阶段：把隐藏状态映射成各个 token 的打分。

因此，同一行参数既表示“这个词在输入端长什么样”，也表示“当前隐藏状态像不像这个词”。

它最直接的收益是减少参数量。若词表大小为 $V$、隐藏维度为 $d$，不共享时需要两张矩阵：

- 输入 embedding：$E \in \mathbb{R}^{V \times d}$
- 输出 softmax 权重：$W \in \mathbb{R}^{V \times d}$

参数量为：

$$
\text{Params}_{\text{untied}} = 2 \times V \times d
$$

共享后只保留一张矩阵：

$$
\text{Params}_{\text{tied}} = V \times d
$$

因此 embedding 相关参数直接减少 50%。

这个结论可以立刻代入真实数量级。若词表大小 $V=50{,}000$，隐藏维度 $d=768$，单张矩阵参数量是：

$$
50{,}000 \times 768 = 38.4\text{M}
$$

不共享时输入表和输出表一共是：

$$
2 \times 38.4\text{M} = 76.8\text{M}
$$

共享后只剩：

$$
38.4\text{M}
$$

若权重用 FP16 存储，每个参数约 2 字节，则仅这部分参数就能节省约：

$$
38.4\text{M} \times 2 \approx 76.8\text{MB}
$$

如果还把优化器状态算进去，训练时节省的显存往往更大。对中大词表语言模型，这不是边角优化，而是会影响模型体积、训练吞吐和部署成本的结构决策。

它也不只是省参数。共享后，每个词向量同时接受输入端和输出端的梯度更新：

$$
\nabla E = \nabla E_{\text{input}} + \nabla E_{\text{output}}
$$

这可以看成一种隐式正则化。一个词向量不能只擅长“被读取”，也不能只擅长“被预测”；它必须同时服务这两件事。结果通常是输入语义空间和输出分类空间更一致，对中小规模数据尤其常见地会带来更稳定的训练和更好的泛化。

但共享不是默认最优。它有明确前提：

- 输入词表与输出词表必须严格一致。
- embedding 维度与输出投影维度必须一致，或者至少能无歧义地映射到同一空间。
- 模型假设“读这个词”和“写这个词”应该落在同一语义空间里。

如果任务本身希望输入表示和输出表示故意分离，那么共享会成为约束，而不是收益。

---

## 问题定义与边界

先把对象定义清楚。

- 输入 embedding：把 token id 映射成向量的查表矩阵，记为 $E \in \mathbb{R}^{V \times d}$。
- 输出 softmax 权重：把隐藏状态投影到词表得分的矩阵，记为 $W \in \mathbb{R}^{V \times d}$。实现里常写成 `lm_head.weight`，矩阵乘法时也可能以 $W^\top$ 的形式出现。

所谓共享，就是令：

$$
W = E
$$

或者在工程实现里，让 `lm_head.weight` 和 `embedding.weight` 指向同一个参数对象。

这里要强调一个新手常混淆的点：共享的不是“数值刚好一样”，而是“底层就是同一组参数”。如果训练开始前把两张矩阵初始化成一样，但后续分别更新，那不叫共享，只是“初值相同”。

这个设计成立，有三个硬边界。

| 条件 | 可共享 | 原因 |
|---|---|---|
| 输入词表与输出词表完全一致 | 是 | 同一行必须始终对应同一个 token |
| embedding 维度与输出投影维度一致 | 是 | 隐藏状态才能直接和共享矩阵做匹配 |
| encoder 与 decoder 词表不同 | 否 | 行索引语义不一致 |
| decoder hidden size 与 embedding size 不同 | 否 | 需要额外投影，不能直接 tied |
| 任务只做分类、不做词表生成 | 通常无意义 | 没有对应的词表 softmax 层 |

为什么“索引语义一致”这么关键，可以看一个最小例子。

假设词表第 10 行对应 token `"apple"`：

- 输入时，token id 为 10，模型查第 10 行得到 `"apple"` 的向量。
- 输出时，softmax 第 10 个 logit 也必须表示“预测为 `apple` 的分数”。

如果输入词表第 10 行是 `"apple"`，输出词表第 10 行却是 `"orange"`，那共享后就会出现直接的语义冲突：同一行参数一会儿代表 `"apple"`，一会儿又代表 `"orange"`。这不是训练难一点，而是定义本身已经错了。

因此，“词表一致”不是实现细节，而是共享成立的数学前提。

再看不同架构的边界。

| 模型类型 | 是否容易共享 | 原因 |
|---|---|---|
| Decoder-only | 最容易 | 输入和输出天然使用同一词表 |
| Encoder-decoder，统一 tokenizer | 可以 | 需进一步检查 encoder、decoder、lm head 的维度和词表 |
| Encoder-decoder，源语言和目标语言词表不同 | 通常不行 | 输入和输出空间不是同一索引体系 |
| 分类模型 | 通常不适用 | 输出不是词表分布 |

真实工程里，decoder-only 模型最容易做 embedding-softmax 共享，因为训练目标本来就是“根据前文预测下一个 token”。输入 token 和输出 token 属于同一套词表，语义绑定最自然。

encoder-decoder 模型则要多看一步：encoder、decoder、最终输出头是否共用同一套 tokenizer 和同一维度。像 T5 这种统一 text-to-text 架构，更适合做三路共享；而某些翻译系统若源语言和目标语言使用不同词表，或者业务上故意把输入输出词典拆开，就不能直接 tied。

---

## 核心机制与推导

先看最基础的单路共享。

对输入 token $t$，输入阶段是查表：

$$
h_{\text{in}} = E[t]
$$

这里 $E[t]$ 表示矩阵 $E$ 的第 $t$ 行，也就是 token $t$ 的向量表示。

经过若干层 Transformer 后，得到输出隐藏状态 $h_{\text{out}} \in \mathbb{R}^{d}$。若不共享，输出 logits 通常是：

$$
z = W h_{\text{out}} + b
$$

其中 $W \in \mathbb{R}^{V \times d}$，$b \in \mathbb{R}^{V}$。

若共享，则直接写成：

$$
z = E h_{\text{out}} + b
$$

第 $i$ 个词的 logit 为：

$$
z_i = \langle E_i, h_{\text{out}} \rangle + b_i
$$

其中 $E_i$ 是词表矩阵第 $i$ 行，$\langle \cdot,\cdot \rangle$ 是点积。

这件事的含义非常直接：输出层在做的事不是“重新发明一个分类器”，而是在问：

> 当前隐藏状态和词表里的哪一个词向量最相似？

把 softmax 展开后，预测概率为：

$$
p(y=i \mid h_{\text{out}}) = \frac{\exp(\langle E_i, h_{\text{out}} \rangle + b_i)}{\sum_{j=1}^{V}\exp(\langle E_j, h_{\text{out}} \rangle + b_j)}
$$

所以共享后的输出层，本质上是“用隐藏状态去检索词表向量”。

这时同一个词向量 $E_i$ 同时承担两种角色：

- 输入时，它是“词的表示”。
- 输出时，它是“词的分类模板”。

这就是共享成立的核心直觉：如果一个词在输入语义空间里由某个向量表示，那么模型在生成这个词时，也可以把“当前状态像不像这个向量”当作判断依据。

### 梯度为什么会变成隐式正则化

设训练损失为交叉熵 $\mathcal{L}$。共享后，矩阵 $E$ 的梯度来自两部分：

$$
\frac{\partial \mathcal{L}}{\partial E}
=
\left.\frac{\partial \mathcal{L}}{\partial E}\right|_{\text{input}}
+
\left.\frac{\partial \mathcal{L}}{\partial E}\right|_{\text{output}}
$$

这意味着同一组参数同时被两种学习信号推动：

- 输入端希望它更适合表示 token 的上下文语义。
- 输出端希望它更适合区分“该不该生成这个 token”。

若不共享，这两类信号分别去更新两张矩阵；模型可以把“理解词”和“生成词”学成两套坐标系。共享后，这种自由被压缩，模型被迫把两种功能对齐。对数据不够大、参数预算有限的场景，这通常是有利的，因为它减少了不必要的自由度。

### 一个极简数字例子

设词表只有 3 个词：`cat`、`dog`、`apple`，维度 $d=2$，共享矩阵为：

$$
E=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
$$

设三行分别对应：

- 第 0 行：`cat`
- 第 1 行：`dog`
- 第 2 行：`apple`

输入 `cat` 时，查表得到：

$$
E[\text{cat}] = (1, 0)
$$

假设某一步 decoder 输出隐藏状态为：

$$
h_{\text{out}} = (0.9, 0.1)
$$

则 logits 为：

$$
z = E h_{\text{out}} =
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}
\begin{bmatrix}
0.9\\
0.1
\end{bmatrix}
=
\begin{bmatrix}
0.9\\
0.1\\
1.0
\end{bmatrix}
$$

所以三个词的打分分别是：

- `cat`: $0.9$
- `dog`: $0.1$
- `apple`: $1.0$

若不考虑 bias，softmax 概率约为：

$$
\text{softmax}(z) \approx (0.388,\ 0.175,\ 0.437)
$$

因此模型更偏向预测 `apple`。原因不是 `apple` 被特殊照顾，而是当前隐藏状态和 `apple` 那一行向量的点积最大。

这个例子有两个作用：

- 它说明输出层不是黑箱分类器，而是“和词表向量逐行做相似度比较”。
- 它说明共享后，词向量几何结构会直接影响生成分布。

### 三路共享是什么

在 encoder-decoder 模型里，共享可以进一步扩展为：

$$
E_{\text{enc}} = E_{\text{dec}} = W_{\text{out}}
$$

也就是：

- encoder 输入 embedding
- decoder 输入 embedding
- decoder 输出 softmax 权重

三者共用一张表。

它的含义是：同一套词向量同时服务“读源文本”“读已生成文本”“预测下一词”。

这在统一 text-to-text 架构中很自然。T5 就是典型例子：无论做摘要、翻译还是问答，输入输出都被统一表示成文本序列，因此让 encoder、decoder 和输出头围绕同一词表空间工作，结构上是连贯的。

但共享越多，先验越强。先验对时，模型会更省参数、更稳；先验错时，它就会限制表示能力。比如某些任务里，编码阶段更偏“条件理解”，解码阶段更偏“目标生成”，两边未必总该被硬绑定在同一参数空间里。

---

## 代码实现

实现关键只有一句话：不是“把两份权重拷贝成一样”，而是“两个模块引用同一个参数对象”。

下面先给出一个可运行的 NumPy 玩具实现。它演示三件事：

- 共享前后参数量怎么变化。
- 输入查表和输出 logits 是否真的共用同一张权重。
- 修改共享权重后，输入端和输出端是否同时看到变化。

```python
import numpy as np


class TiedEmbeddingSoftmax:
    def __init__(self, vocab_size: int, hidden_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        # 共享权重: [V, d]
        self.shared_weight = rng.normal(
            loc=0.0, scale=0.02, size=(vocab_size, hidden_size)
        ).astype(np.float32)

        # 输出 bias 通常仍然单独保留
        self.bias = np.zeros(vocab_size, dtype=np.float32)

    def embed(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        return self.shared_weight[token_ids]

    def logits(self, hidden_states):
        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        return hidden_states @ self.shared_weight.T + self.bias


def untied_param_count(vocab_size, hidden_size):
    return 2 * vocab_size * hidden_size


def tied_param_count(vocab_size, hidden_size):
    return vocab_size * hidden_size


if __name__ == "__main__":
    # 参数量检查
    V, d = 50_000, 768
    assert untied_param_count(V, d) == 76_800_000
    assert tied_param_count(V, d) == 38_400_000

    # 构造模型
    model = TiedEmbeddingSoftmax(vocab_size=10, hidden_size=4, seed=42)

    # 输入查表
    x = model.embed([1, 3, 5])
    assert x.shape == (3, 4)

    # 输出 logits
    h = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
    z = model.logits(h)
    assert z.shape == (3, 10)

    # “共享”检查：修改同一块底层权重，输入端和输出端都应观察到变化
    before = float(model.shared_weight[1, 2])
    model.shared_weight[1, 2] += 1.0

    after_embed = float(model.embed([1])[0, 2])
    after_weight = float(model.shared_weight[1, 2])

    assert np.isclose(after_embed, before + 1.0)
    assert np.isclose(after_weight, before + 1.0)
    assert np.isclose(after_embed, after_weight)

    print("All checks passed.")
```

这段代码可以直接运行。你会看到 `All checks passed.`，说明输入端和输出端确实共享同一张矩阵，而不是两份内容相同的副本。

如果换成 PyTorch，核心写法通常是下面这样：

```python
import torch
import torch.nn as nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 关键：让两个模块引用同一个 Parameter
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids, hidden_states):
        x = self.embed(token_ids)
        logits = self.lm_head(hidden_states)
        return x, logits


if __name__ == "__main__":
    torch.manual_seed(0)

    model = TinyLM(vocab_size=10, hidden_size=4)

    token_ids = torch.tensor([1, 3, 5], dtype=torch.long)
    hidden_states = torch.randn(3, 4)

    x, logits = model(token_ids, hidden_states)

    assert x.shape == (3, 4)
    assert logits.shape == (3, 10)

    # 检查是否真的是同一个 Parameter 对象
    assert model.lm_head.weight is model.embed.weight

    # 修改 embedding 权重，lm_head 应同步变化
    with torch.no_grad():
        before = model.embed.weight[1, 2].item()
        model.embed.weight[1, 2].add_(1.0)
        after_embed = model.embed.weight[1, 2].item()
        after_head = model.lm_head.weight[1, 2].item()

    assert abs(after_embed - (before + 1.0)) < 1e-6
    assert abs(after_head - after_embed) < 1e-6

    print("Tied weights verified.")
```

这段 PyTorch 代码也可以直接运行。输出 `Tied weights verified.` 就说明共享关系已经建立。

工程上要重点检查三件事。

| 检查项 | 正确做法 | 错误做法 |
|---|---|---|
| 权重共享 | 两个模块指向同一 `Parameter` | 只做一次 `copy_`，后续各自更新 |
| 维度匹配 | `hidden_size == embedding_dim`，或显式加投影层 | 依赖临时 reshape 糊过去 |
| 词表一致 | 同一 tokenizer、同一 id 映射 | 输入输出各自维护不兼容词表 |

再补一个新手常见误区。下面这段写法**不是**真正的共享：

```python
self.lm_head.weight.data.copy_(self.embed.weight.data)
```

原因很简单：这只是把数值复制了一次。复制完成后，`lm_head.weight` 和 `embed.weight` 仍然是两块独立内存，训练中会分别接收梯度，最后一定分叉。

真实工程里，很多框架都会显式提供 `tie_weights()` 或类似逻辑。它的职责不是“再初始化一遍”，而是确保输出头和输入 embedding 真正绑定为同一组参数。

---

## 工程权衡与常见坑

共享最常见的收益和代价如下。

| 维度 | 共享 | 不共享 |
|---|---|---|
| 参数量 | 更少 | 更多 |
| 显存占用 | 更低 | 更高 |
| 小数据泛化 | 常更稳 | 更易过拟合 |
| 表达自由度 | 更低 | 更高 |
| 结构约束 | 更强 | 更弱 |
| 适合统一词表生成任务 | 是 | 也可，但更重 |

这里最重要的不是背表，而是理解每一项为什么会出现。

- 参数量更少：少掉整整一张 $V \times d$ 的矩阵。
- 泛化常更稳：少了一套独立自由度，输入和输出空间被强制对齐。
- 表达自由度更低：模型不能把“理解词”和“生成词”学成两套完全不同的表示。
- 结构约束更强：必须满足词表一致、维度一致、任务形式匹配。

下面看几个最常见的坑。

### 第一个坑：词表不一致

假设 encoder 词表有 32,000 个 token，decoder 因业务词典扩展到 33,000 个 token，多出的 1,000 个词只在输出端存在。此时就不能直接共享，因为连矩阵形状都不同：

$$
E_{\text{enc}} \in \mathbb{R}^{32000 \times d},\quad
W_{\text{out}} \in \mathbb{R}^{33000 \times d}
$$

它们不可能是同一张表。

常见处理方式只有三类：

| 处理方式 | 优点 | 代价 |
|---|---|---|
| 统一 tokenizer 和词表 | 可以直接共享 | 需要改数据和训练流程 |
| 保持 untied | 简单直接 | 参数更多，空间分离 |
| 建公共子空间后部分共享 | 兼顾灵活性 | 实现更复杂 |

### 第二个坑：维度不一致

有些模型把 embedding 维度设成 $d_e$，内部 hidden 维度设成 $d_h$。如果 $d_e \neq d_h$，就不能直接计算：

$$
z = E h_{\text{out}}
$$

因为矩阵乘法维度不合法。

此时常见做法是加一个投影矩阵 $P \in \mathbb{R}^{d_e \times d_h}$：

$$
z = E(P h_{\text{out}})
$$

或者等价地写成：

$$
z = (E P) h_{\text{out}}
$$

含义很直接：先把内部隐藏状态映射到 embedding 的坐标系，再去和共享词表向量做匹配。

这种方案有时被称为 joint projection 或 shared subspace。它保留了“输出仍然围绕输入词表空间工作”的好处，但不再要求模型主干维度和 embedding 维度完全相同。

### 第三个坑：梯度冲突

共享意味着输入端和输出端同时更新同一组词向量。很多时候这是收益，但它也可能带来冲突。

例如某个词在输入侧更需要承担“区分上下文语义”的职责，而在输出侧更需要承担“作为常见候选词被快速识别”的职责。这两类压力不一定完全同向。结果通常不是训练崩掉，而是模型在某些词上做了折中，导致：

- 判别性不够强
- 生成质量不够尖锐
- 高频词和低频词的表示被拉扯得不均衡

常见缓解手段如下。

| 手段 | 思路 | 适用情况 |
|---|---|---|
| 输出前加 LayerNorm | 稳定输出分布 | 训练不稳定或 logits 过尖 |
| 加投影层再共享 | 给主干留一层变换自由度 | 维度一致但任务拉扯明显 |
| Partial tying | 只共享最相关的部分 | encoder/decoder 角色差异大 |
| 完全 untied | 彻底解除约束 | 追求容量上限或任务差异极大 |

### 第四个坑：把不同层面的“参数共享”混为一谈

Embedding-softmax 共享，和 ALBERT 那类跨层参数共享，不是一回事。

| 共享类型 | 共享对象 | 主要影响 |
|---|---|---|
| Embedding-softmax 共享 | 词表相关矩阵 | 压缩词表参数，约束输入/输出语义空间 |
| 跨层共享 | 多层 Transformer block | 压缩主干参数，直接影响层间表达多样性 |

两者都在省参数，但风险等级不同。词表共享主要约束的是“词怎么表示、怎么生成”；跨层共享约束的是“不同层是否还能学出足够不同的功能”。后者通常对表达能力的影响更深，不应混为一谈。

---

## 替代方案与适用边界

不是所有模型都应该把 embedding 和 softmax 绑死。至少有四类常见方案。

| 方案 | 做法 | 适用场景 |
|---|---|---|
| Untied embedding | 输入表和输出表完全独立 | 大数据、输入输出角色差异明显 |
| Full tying | 输入表与输出表完全共享 | 同词表、参数敏感、生成式任务 |
| Partial tying | 只共享一部分，如 decoder embedding 与 lm head | encoder/decoder 分工明显 |
| Joint projection | 先投影到公共空间再共享 | 维度不一致但仍想保留共享收益 |

### Untied embedding

这是最自由的做法。模型可以分别学习：

- 词作为输入时应该如何表示
- 词作为输出类别时应该如何判别

如果数据量足够大、任务足够复杂，这种自由度可能换来更高上限。代价是参数量更大，也更容易过拟合，尤其在中小模型上更明显。

### Full tying

这是最强约束、也最省参数的做法。它最适合下面几类场景：

- 统一 tokenizer
- 统一文本生成接口
- 参数预算紧张
- 希望输入输出语义空间尽量一致

decoder-only 语言模型通常天然适合这一类，因为“读历史 token”和“预测下一个 token”本来就是同一词表体系里的两端动作。

### Partial tying

这是很实用的中间路线。最常见形式是：

$$
E_{\text{dec}} = W_{\text{out}}, \quad E_{\text{enc}} \text{ 独立}
$$

即只让 decoder embedding 与输出头共享，encoder 保持独立。

原因也很直接：输出生成直接发生在 decoder 侧，所以 decoder 输入和输出词表关系最紧；encoder 更偏向条件理解，可以保留自己的表示空间。

这类做法特别适合下面场景：

- encoder 负责理解多模态或结构化输入
- decoder 负责统一文本生成
- 编码侧和生成侧的需求相关，但不完全相同

### Joint projection

当你想保留共享收益，但主干维度和 embedding 维度又不同，joint projection 往往是折中方案。它的核心不是“完全共享”，而是“让输出最终仍落回 embedding 所在的语义空间”。

从工程角度看，它常出现在下面两类情况：

- embedding 维度为了词表效率被单独设定
- 主干隐藏维度为了模型容量被设得更大

### 什么时候更适合哪一种

可以用下面这个经验表判断。

| 条件 | 更倾向方案 |
|---|---|
| 小到中等数据规模 | Full tying 或 Partial tying |
| 显存和参数预算紧张 | Full tying |
| 统一 text-to-text、多任务生成 | Full tying |
| encoder 与 decoder 角色差异明显 | Partial tying |
| 输入输出词表不同 | Untied |
| hidden size 与 embedding size 不同 | Joint projection 或 Untied |
| 超大数据、追求容量上限 | 重新评估是否 Untied |

因此，共享不是“最佳实践模板”，而是一个带强先验的结构选择。你采用它，等于接受这样一个假设：

> 同一个词在输入空间和输出空间里，应该尽量由同一组向量来定义。

任务越符合这个假设，共享越有效；偏离越多，它越可能从收益变成束缚。

---

## 参考资料

1. Press, O. and Wolf, L. *Using the Output Embedding to Improve Language Models*. EACL 2017. 这是 embedding-softmax 共享最常被引用的基础论文，核心贡献是明确提出把输出 embedding 作为输入 embedding 使用，并分析其正则化效果。适合先读，用来建立定义、公式和动机。

2. Inan, H., Khosravi, K. and Socher, R. *Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling*. ICLR 2017. 这篇工作从语言模型损失和词向量空间的角度解释为什么 tying 合理，适合在理解基本机制后阅读，用来补足理论背景。

3. Vaswani, A. et al. *Attention Is All You Need*. NeurIPS 2017. Transformer 论文本身不是专门讲权重共享，但它给出了后续大规模生成模型使用统一 embedding 空间的基本结构背景。适合用来确认 embedding、hidden state、output projection 在标准 Transformer 中各自的位置。

4. Raffel, C. et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR 2020. T5 是三路共享讨论里的典型对象。适合在理解单路共享后看它如何把 encoder、decoder 和输出头放进统一 text-to-text 框架。

5. Lewis, M. et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. 适合对照理解 encoder-decoder 模型里的表示分工，以及为什么不是所有 seq2seq 模型都会把共享做得一样激进。

6. Lan, Z. et al. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR 2020. 适合扩展阅读，用来区分“词表参数共享”和“跨层参数共享”这两类完全不同的压缩策略。

7. Hugging Face Transformers 文档与对应模型源码。重点看 `tie_weights()`、`get_input_embeddings()`、`get_output_embeddings()` 这类接口，能直接看到工程里“同一参数对象”是如何绑定的。适合在理论理解后对照源码看真正落地方式。

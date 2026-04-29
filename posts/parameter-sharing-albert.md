## 核心结论

ALBERT 不是“更小号的 BERT”，而是对 BERT 的参数组织方式做了两次重构：一是把输入层的词嵌入拆成低维表征和高维投影，二是让多层 Transformer 共享同一套参数。前者解决词表矩阵过大的问题，后者解决层数增加时参数线性膨胀的问题。

它优化的核心指标是**参数效率**。参数效率的意思是“用更少的可训练参数，尽量保住原来的效果”。这和“算得更少”不是同一件事。ALBERT 可以让模型文件更小、训练时参数同步更轻、checkpoint 更省空间，但不保证前向计算天然更快。

可以先记住两条公式：

$$P_{emb}=V \times H \quad \rightarrow \quad P_{emb}=V \times E + E \times H$$

$$P_{enc}\approx L \times P_{layer} \quad \rightarrow \quad P_{enc}\approx P_{layer}$$

这里的直观含义是：原来 embedding 直接从词表连到隐藏维度，现在先连到一个更小的中间维度；原来每层各有一套参数，现在多层复用一套参数。

| 维度 | BERT | ALBERT |
|---|---|---|
| embedding | 直接用 `V × H` | 用 `V × E + E × H` 分解 |
| 编码器参数 | 每层独立 | 跨层共享 |
| 参数量 | 随层数明显增长 | 对层数不敏感 |
| 推理速度 | 取决于层数和宽度 | 不一定更快 |
| checkpoint 体积 | 大 | 小 |

---

## 问题定义与边界

先把问题说清楚。BERT 的参数主要堆在两块。

第一块是 **embedding**。embedding 可以理解为“把离散 token 变成连续向量的查表层”。如果词表大小是 $V$，隐藏维度是 $H$，那么仅词嵌入矩阵就有 $V \times H$ 个参数。词表大、隐藏维度大时，这一块非常重。

第二块是 **多层 Transformer 编码器**。Transformer 层可以理解为“重复执行的上下文建模模块”。普通 BERT 有 12 层、24 层时，每层都单独维护 attention 和前馈网络参数，所以总参数近似按层数 $L$ 线性增长。

下面先统一符号：

| 符号 | 含义 |
|---|---|
| `V` | 词表大小 |
| `E` | embedding 维度，即低维词向量大小 |
| `H` | hidden size，即主干网络隐藏维度 |
| `L` | Transformer 层数 |
| `P_layer` | 单个 Transformer 层的参数量 |

ALBERT 的边界也要说清楚。它解决的是“模型太大”这类问题，不是所有效率问题都一起解决。

| 指标 | 主要受什么影响 | ALBERT 是否直接优化 |
|---|---|---|
| 参数量 | embedding 设计、是否共享层参数 | 是 |
| 计算量 FLOPs | 层数、序列长度、隐藏维度、头数 | 不直接 |
| 训练显存 | 参数、激活、优化器状态 | 部分优化 |
| 多卡同步成本 | 参数规模、梯度规模 | 是 |
| 推理延迟 | 计算深度、宽度、硬件实现 | 不一定 |

一句话边界总结就是：**参数少，不等于算得少。**

真实工程里这个区别很重要。比如做多卡预训练时，常见痛点并不只是矩阵乘法慢，还包括显存吃紧、梯度同步耗时、checkpoint 动辄几 GB。此时 ALBERT 的价值主要体现在参数和存储侧。如果你的目标是线上极低延迟，比如移动端或高 QPS 服务，ALBERT 不一定是最优答案。

---

## 核心机制与推导

ALBERT 的第一个机制叫 **factorized embedding parameterization**，中文常叫“embedding 分解”。分解的意思是“把一个大矩阵拆成两个更小矩阵相乘”。

普通 BERT 的 embedding 参数量是：

$$P_{emb}^{bert}=V \times H$$

ALBERT 先把 token 映射到低维空间 $E$，再投影到主干隐藏维度 $H$：

$$P_{emb}^{albert}=V \times E + E \times H$$

只要 $E \ll H$，参数就会显著下降。

玩具例子可以直接算。取：

- `V = 30000`
- `H = 768`
- `E = 128`

那么：

- BERT 式 embedding：`30000 × 768 = 23,040,000`
- ALBERT 式 embedding：`30000 × 128 + 128 × 768 = 3,938,304`

只看 embedding，这里就少了约 1910 万参数，压缩比约为：

$$\frac{23,040,000}{3,938,304}\approx 5.85$$

第二个机制叫 **cross-layer parameter sharing**，即“跨层参数共享”。共享的意思是“多层重复使用同一套权重”，不是“把层删掉”。

普通堆叠时，编码器参数大致是：

$$P_{enc}^{bert}\approx L \times P_{layer}$$

ALBERT 中，如果所有层共享同一个 Transformer block 的参数，那么存储的参数更接近：

$$P_{enc}^{albert}\approx P_{layer}$$

注意这里减少的是“需要存几套参数”，不是“要做几次层计算”。如果模型仍然跑 12 层，那就是同一个 block 连续计算 12 次。

这正是很多初学者最容易误解的点：

- 错误理解：12 层共享参数，所以模型本质上只剩 1 层。
- 正确理解：仍然执行 12 次变换，只是这 12 次变换复用同一套参数。

可以把流程写成一条结构链：

`词表 -> 低维 embedding -> 高维投影 -> 共享 Transformer 层 × L -> 输出`

论文里的结果也能帮助理解这个机制的边界。ALBERT-base 大约 12M 参数，而 BERT-base 大约 108M 参数，压缩非常明显。但共享参数并不是没有代价。论文中的对比实验表明，`all-shared` 虽然最省参数，但和 `not-shared` 相比，表达能力会有边界，任务指标可能略有回落。这说明共享不是“零成本压缩”，而是在参数与表达力之间做取舍。

还有一个相关设计是 **SOP**，即 Sentence Order Prediction，中文可理解为“句子顺序预测”。它用来替代 BERT 的 NSP。这个设计不是本文主角，但它说明 ALBERT 并不只做参数压缩，也顺手调整了预训练目标，以减轻原始 NSP 信号过弱的问题。

---

## 代码实现

实现 ALBERT 时，最关键的是把“分解 embedding”和“共享层实例”分开理解。前者是两段映射，后者是一个模块反复调用。

下面先用可运行的 Python 写一个最小例子，只计算参数量，不依赖深度学习框架：

```python
def bert_embedding_params(vocab_size: int, hidden_size: int) -> int:
    return vocab_size * hidden_size

def albert_embedding_params(vocab_size: int, embedding_size: int, hidden_size: int) -> int:
    return vocab_size * embedding_size + embedding_size * hidden_size

def bert_encoder_params(num_layers: int, params_per_layer: int) -> int:
    return num_layers * params_per_layer

def albert_encoder_params(num_layers: int, params_per_layer: int) -> int:
    # 共享参数后，存储上仍只保留一套层参数
    return params_per_layer

V, E, H = 30000, 128, 768
L, P_layer = 12, 7_000_000

bert_emb = bert_embedding_params(V, H)
albert_emb = albert_embedding_params(V, E, H)

assert bert_emb == 23_040_000
assert albert_emb == 3_938_304
assert bert_emb > albert_emb

bert_enc = bert_encoder_params(L, P_layer)
albert_enc = albert_encoder_params(L, P_layer)

assert bert_enc == 84_000_000
assert albert_enc == 7_000_000
assert bert_enc > albert_enc

compression_ratio = bert_emb / albert_emb
assert round(compression_ratio, 2) == 5.85

print("ok")
```

如果换成 PyTorch 风格，结构大致如下：

```python
import torch.nn as nn

class AlbertEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, input_ids):
        x = self.word_embed(input_ids)
        return self.proj(x)

class SharedEncoder(nn.Module):
    def __init__(self, block, num_layers):
        super().__init__()
        self.block = block
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.block(x)
        return x
```

这里有三个关键点。

第一，`word_embed` 是低维查表，`proj` 是升维投影。也就是先查词，再送入主干隐藏空间。

第二，参数共享依赖“同一个模块实例被重复调用”。如果你在 `__init__` 里写一个 `ModuleList([TransformerBlock() for _ in range(num_layers)])`，那就不是共享，而是创建了多份独立参数。

第三，共享层后，反向传播会把来自不同层位置的梯度累加到同一套参数上。这会改变训练动力学，也就是参数更新时受到多个深度位置共同影响，这正是 ALBERT 能工作、也可能带来表达约束的原因。

| 实现点 | 错误写法 | 正确写法 |
|---|---|---|
| embedding 分解 | 直接 `Embedding(V, H)` | `Embedding(V, E)` 再 `Linear(E, H)` |
| 层共享 | 创建 `L` 个 block 实例 | 一个 block 循环调用 `L` 次 |
| 共享判断 | “循环多次就是共享” | “同一个模块实例才是共享” |

真实工程例子可以这样看：如果你在预训练一个中文编码器，模型深度希望保留 12 层以上，因为长上下文建模需要足够深的非线性变换；但你的集群带宽一般，checkpoint 上传也慢。此时 ALBERT 的共享设计可以明显减轻参数同步和模型存储压力，同时保留深层结构。它不一定让单次 step 更短，但经常能让训练系统更容易跑起来。

---

## 工程权衡与常见坑

ALBERT 的价值很明确，但误解也很集中。

| 常见坑 | 错误理解 | 正确理解 | 规避方式 |
|---|---|---|---|
| 把参数少当成速度快 | 参数减少就必然低延迟 | 延迟主要看计算路径和硬件实现 | 实测吞吐、延迟和 FLOPs |
| 把共享参数当成减层 | 12 层共享等于 1 层 | 仍然是 12 次层计算 | 分清“参数套数”和“计算次数” |
| embedding 压得过小 | `E` 越小越好 | 过小会伤词表示能力 | 从论文常用值如 `E=128` 起试 |
| 忽视表达力边界 | 共享不会损失能力 | 全共享可能带来任务回落 | 做下游任务验证 |
| 误用框架实现 | 重复实例化层也算共享 | 只有同一实例复用才共享 | 检查参数对象是否同源 |

一个实际的经验点是：`E` 不是越小越好。论文里的结果显示，base 配置下 `E=128` 是一个比较稳的折中。太小，词表信息在进入主干前就被压得过狠；太大，又会削弱分解带来的参数收益。

另一个权衡是训练稳定性。多层共享同一套参数后，这套参数必须同时服务浅层和深层的位置需求。浅层通常更偏局部模式，深层更偏抽象组合，这种“同参多职”天然更难。因此，ALBERT 体现的是一种精细折中：它不是不付代价地省参数，而是用一定表达力风险换来大规模参数压缩。

如果你的目标是线上超低延迟，通常应先看浅层蒸馏模型。因为浅层模型减少的是实际计算深度，而 ALBERT 更多减少的是参数冗余。

---

## 替代方案与适用边界

什么时候优先考虑 ALBERT？当你最关心的是参数预算，而不是把延迟压到最低。

下面直接对比：

| 方案 | 主要目标 | 参数量 | 推理速度 | 适合场景 |
|---|---|---|---|---|
| BERT | 标准基线 | 较大 | 中等 | 通用研究与基线系统 |
| ALBERT | 参数效率 | 很低 | 不一定更快 | 预训练、存储和同步受限 |
| DistilBERT/TinyBERT | 蒸馏压缩 | 低 | 往往更快 | 在线推理、低延迟服务 |
| 更浅模型 | 直接减层 | 低 | 更快 | 对时延极敏感的场景 |

可以把决策逻辑写成一个简表：

| 关注点 | 更适合的选择 |
|---|---|
| 参数预算紧 | ALBERT |
| checkpoint 太大 | ALBERT |
| 多卡同步压力大 | ALBERT |
| 线上低延迟优先 | 蒸馏模型或浅层模型 |
| 极限精度优先 | 先用标准大模型做上限 |

场景 A：你在做领域预训练，GPU 显存和模型存储都紧张，但仍想保留较深编码器结构。这时 ALBERT 很合适，因为它保住了“深”这个属性，同时大幅减参数。

场景 B：你在做线上分类服务，请求量高，要求 P99 延迟极低。这时更优先考虑 DistilBERT、TinyBERT 或者更浅的 backbone，因为这些方案直接减少了计算深度。

一句话总结这部分：**ALBERT 适合参数效率优先，不适合把速度当唯一目标。**

---

## 参考资料

1. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
2. [ALBERT 论文 HTML 版](https://ar5iv.labs.arxiv.org/html/1909.11942)
3. [Google Research ALBERT 仓库](https://github.com/google-research/albert)
4. [Hugging Face Transformers: ALBERT 文档](https://huggingface.co/docs/transformers/model_doc/albert)

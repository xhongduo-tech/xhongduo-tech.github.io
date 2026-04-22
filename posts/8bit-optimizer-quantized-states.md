## 核心结论

8-bit 优化器是一类把优化器状态压缩到 8 位整数存储的训练方法；在 Adam 中，核心被量化的是一阶动量 `m_t` 和二阶矩 `v_t`，不是模型权重本身。

新手版可以这样理解：模型参数像“当前答案”，优化器状态像“记笔记”；8-bit Adam 不是把答案写糊，而是把笔记本换成更省空间的格式。训练时该怎么算仍然怎么算，只是长期保存的历史梯度信息不再全部用 `fp32` 存。

| 对象 | 是否量化 | 作用 |
|---|---:|---|
| 模型权重 | 否 | 前向推理和反向传播的参数 |
| Adam 状态 `m_t, v_t` | 是 | 保存历史梯度信息 |
| 更新计算 | 否 | 仍在高精度中完成 |

标准 Adam 对每个参数通常要额外保存两份 `fp32` 状态：一份一阶动量 `m_t`，一份二阶矩 `v_t`。`fp32` 每个数占 4 字节，`int8` 每个数占 1 字节，所以仅从状态张量本体看，存储可以从 8 字节/参数降到 2 字节/参数，约节省 75%。

对大模型来说，这不是小优化。175B 参数模型的两份 Adam `fp32` 状态粗算是：

$$
2 \times 175B \times 4\text{ bytes} \approx 1.4\text{ TB}
$$

如果状态主体改成 8-bit，约为：

$$
2 \times 175B \times 1\text{ byte} \approx 350\text{ GB}
$$

中间差值约 1TB。考虑分块尺度等元数据后，论文与工程实现中常用的说法是可以节省约 75% 的优化器状态显存，在 175B 模型量级上约节省 700GB 以上显存，具体数值取决于是否分片、是否混合精度、哪些参数被保留为 32-bit。

---

## 问题定义与边界

大模型训练的显存瓶颈通常不是单一来源。至少有四类主要对象：模型权重、梯度、优化器状态、激活值。激活值是前向传播中临时保存、供反向传播使用的中间结果；优化器状态是 Adam 这类算法为了记住历史梯度趋势而长期保存的数据。

8-bit 优化器主要解决的是优化器状态过大，不直接解决激活值过大，也不直接解决多卡训练中的梯度通信开销。

| 解决的问题 | 8-bit 优化器是否直接解决 |
|---|---:|
| 优化器状态过大 | 是 |
| 激活值过大 | 否 |
| 梯度通信过大 | 否 |
| 模型权重本身过大 | 间接，有帮助但不是核心 |

玩具例子：假设一个模型只有 1,000,000 个参数。标准 Adam 的两份状态占用：

$$
2 \times 1,000,000 \times 4\text{ bytes} = 8\text{ MB}
$$

8-bit 状态主体占用：

$$
2 \times 1,000,000 \times 1\text{ byte} = 2\text{ MB}
$$

这说明 8-bit 优化器的收益来自“每个参数背后的历史状态变小了”，不是来自模型参数消失了。

真实工程例子：训练一个 13B 参数模型时，参数本身已经很大，但 Adam 还要额外保存两份状态。即使模型权重用 `bf16`，优化器状态仍可能以 `fp32` 保存。结果是显存卡死的根源常常不是某一层网络特别大，而是每个参数背后都带着两份长期历史状态。此时把 `torch.optim.Adam` 换成 `bitsandbytes` 的 `Adam8bit`，往往能把“完全放不下”变成“可以开始训练”。

边界也必须说清楚：如果显存主要被长序列激活值占满，8-bit 优化器的收益会有限；如果瓶颈是多机多卡通信，它也不是主解法。它适合的问题是：训练逻辑基本可行，但 Adam 状态占用太高。

---

## 核心机制与推导

8-bit Adam 的核心机制是 `block-wise quantization + dynamic quantization`。`block-wise quantization` 指按块量化：不把整个大张量共用一个尺度，而是把张量切成许多小块，每块单独计算缩放尺度。`dynamic quantization` 指动态量化：尺度不是训练前固定的，而是在训练过程中根据当前块内数值动态计算。

设某个状态张量的一块为 $B$，块内元素为 $x_i$。先计算块内最大绝对值：

$$
a_B = \max(|x_i|,\ i \in B)
$$

再把浮点数映射到 `int8`：

$$
q_i = round(127 \cdot x_i / a_B)
$$

需要参与计算时，再反量化：

$$
\hat{x}_i = a_B \cdot q_i / 127
$$

这里的 $a_B$ 是尺度。尺度是一句白话解释就是：把一组浮点数压进 -127 到 127 这个整数区间时使用的“换算比例”。

为什么要按块？如果整个张量只用一个最大值当尺度，一个极端大值会压缩其他普通值的分辨率。按块后，每个局部区域都有自己的尺度，普通数值更不容易被压扁。

可以把一块参数状态想成一箱温度计读数。先找这箱里绝对值最大的读数当“刻度尺”，再把箱内每个读数压成 8 位数字。下一步要用时，再按这把刻度尺还原成近似浮点值。这个例子只用于理解尺度，不替代数学定义。

Adam 的状态更新公式本身没有变：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

其中 `g_t` 是当前梯度；一阶动量 `m_t` 是梯度的指数滑动平均，用来估计更新方向；二阶矩 `v_t` 是梯度平方的指数滑动平均，用来估计不同参数的更新尺度。

8-bit Adam 的关键点是：`m_t` 和 `v_t` 可以用 `int8` 加尺度元数据存储；真正更新参数时，会把状态反量化成近似高精度值，再按 Adam 公式计算。

流程可以写成：

```text
当前梯度 g_t
    |
    v
读取 int8 状态 q(m), q(v) + 块尺度
    |
    v
反量化得到近似 m_{t-1}, v_{t-1}
    |
    v
按 Adam 公式更新 m_t, v_t 和参数
    |
    v
把新的 m_t, v_t 按块动态量化
    |
    v
保存 int8 状态 + 块尺度
```

这也是它和“把模型权重量化成 8-bit 推理”的根本区别：这里压缩的是训练过程中的优化器记忆，不是直接把模型计算主体改成低精度。

---

## 代码实现

工程上最常见的落地方式是使用 `bitsandbytes`，把 `torch.optim.Adam` 替换成 `bnb.optim.Adam8bit`。`bitsandbytes` 是一个提供低比特优化器、量化线性层和相关 CUDA 算子的库。

典型写法如下：

```python
import torch
import bitsandbytes as bnb
from bitsandbytes.nn import StableEmbedding

num_embeddings = 32000
embedding_dim = 768

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = StableEmbedding(num_embeddings, embedding_dim)
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.head = torch.nn.Linear(embedding_dim, 10)

    def forward(self, token_ids):
        x = self.embed_tokens(token_ids).mean(dim=1)
        x = self.ln(x)
        return self.head(x)

model = TinyModel()
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

`StableEmbedding` 是更稳定的词嵌入层实现；词嵌入层是把 token ID 映射成向量的层，在 NLP 训练中经常对数值稳定性比较敏感。

不是所有参数都适合 8-bit。通常大矩阵状态最值得压缩，因为省显存明显；`bias`、`LayerNorm`、小张量常保留 32-bit，因为它们本来占用不大，而且可能更影响稳定性。

| 参数类型 | 默认策略 | 原因 |
|---|---|---|
| 大矩阵状态 | 8-bit | 省显存明显 |
| bias | 32-bit | 太小，省不多且敏感 |
| LayerNorm | 32-bit | 稳定性优先 |
| embedding | 视情况，常配 `StableEmbedding` | NLP 中更稳 |

下面是一个可运行的纯 Python 玩具实现，用来验证“按块量化再反量化”的基本逻辑。它不是完整 Adam8bit，只演示状态量化的核心数学。

```python
def quantize_blockwise(values, block_size=4):
    quantized = []
    scales = []

    for start in range(0, len(values), block_size):
        block = values[start:start + block_size]
        scale = max(abs(x) for x in block) or 1.0
        scales.append(scale)

        for x in block:
            q = round(127 * x / scale)
            q = max(-127, min(127, q))
            quantized.append(q)

    return quantized, scales


def dequantize_blockwise(quantized, scales, block_size=4):
    values = []

    for block_index, scale in enumerate(scales):
        start = block_index * block_size
        block = quantized[start:start + block_size]

        for q in block:
            values.append(scale * q / 127)

    return values


state = [0.10, -0.20, 0.30, -0.40, 10.0, -8.0, 6.0, -4.0]
q, scales = quantize_blockwise(state, block_size=4)
restored = dequantize_blockwise(q, scales, block_size=4)

assert len(q) == len(state)
assert len(scales) == 2
assert all(isinstance(x, int) for x in q)
assert max(abs(a - b) for a, b in zip(state, restored)) < 0.05
```

真实工程中还会遇到参数分组、混合精度、分布式训练和特定层例外策略。最小心智模型可以保持简单：先定义模型，再替换优化器，再指定敏感层保持高精度。不要把 8-bit Adam 理解为全局一刀切的“所有张量都变 int8”。

---

## 工程权衡与常见坑

8-bit 优化器的最大收益是省显存，但它不保证训练更快。量化和反量化本身有计算开销，实际速度取决于 GPU、模型结构、batch size、显存压力和 kernel 实现。如果原来因为显存不够只能用很小 batch，省下显存后可以加大 batch，那么吞吐可能变好；如果原来显存不紧张，速度不一定提升。

| 常见误解 | 实际情况 |
|---|---|
| 8-bit 能解决所有显存问题 | 只能直接解决优化器状态 |
| 量化后一定更快 | 主要是省显存，速度不一定提升 |
| 所有层都该 8-bit | 小而敏感的层常保留 32-bit |
| 量化是固定规则 | 实际上有动态尺度和按块处理 |

第一个常见坑是把激活值问题误判成优化器状态问题。长上下文训练时，激活值可能占据显存大头。此时换 8-bit 优化器会有帮助，但不会像预期那样大幅降低总显存。更直接的手段可能是梯度检查点、减小序列长度、降低 batch size，或者使用更高效的注意力实现。

第二个常见坑是误解动态量化。8-bit Adam 不是先拿一批数据做离线校准，然后永远使用固定量化表。它是在训练中按块计算尺度，随着状态分布变化而变化。如果把它当成静态量化，很容易在调参时得出错误判断。

第三个常见坑是把敏感层也一起低精度处理。如果训练不稳定，先不要急着判断“8-bit 不行”，应该检查 `embedding`、`LayerNorm`、`bias`、小张量是否被不合适地量化。NLP 任务中，词嵌入层尤其值得关注，使用 `StableEmbedding` 通常更稳。

第四个常见坑是只看参数量，不看显存构成。一个模型能不能训，取决于参数、梯度、优化器状态、激活值、临时 buffer、分布式框架开销的总和。8-bit 优化器只是在其中一项上做强优化。

真实工程例子：微调 LLaMA 或 GPT 类模型时，团队可能先用普通 AdamW 跑不起来，报 CUDA out of memory。第一步可以换成 `bnb.optim.Adam8bit`，并配合 `StableEmbedding`。如果仍然放不下，再继续加梯度检查点、ZeRO 分片或 LoRA/QLoRA。这个顺序的好处是每一步都对应明确瓶颈，而不是同时打开一堆开关后不知道哪个有效。

---

## 替代方案与适用边界

8-bit 优化器不是唯一省显存手段。它和 AMP、BF16、ZeRO、QLoRA、梯度检查点解决的是不同问题。工程上经常需要组合，而不是只换优化器。

| 方法 | 主要省什么 | 适合场景 |
|---|---|---|
| 8-bit optimizer | 优化器状态 | Adam 显存过大 |
| AMP / BF16 | 计算和部分存储 | 通用训练加速与省显存 |
| ZeRO | 参数、状态、梯度分片 | 多卡大模型训练 |
| QLoRA | 权重和适配器微调 | 低显存微调 |
| 梯度检查点 | 激活值 | 激活显存瓶颈 |

AMP 是自动混合精度训练；白话说，就是在能用低精度的地方用低精度，在需要稳定的地方保留较高精度。BF16 是一种 16 位浮点格式，指数范围接近 `fp32`，训练大模型时常比 `fp16` 更稳。

ZeRO 是分布式训练中的状态分片方法；白话说，就是把参数、梯度、优化器状态拆到多张卡上，不让每张卡都保存完整副本。QLoRA 是参数高效微调方法；白话说，就是把基础模型权重量化后冻结，只训练很小的适配器参数。

选择时可以按瓶颈判断：

| 瓶颈现象 | 优先考虑 |
|---|---|
| Adam 状态占用太大 | 8-bit optimizer |
| 长序列导致显存暴涨 | 梯度检查点、FlashAttention、减小序列长度 |
| 单卡放不下完整训练状态 | ZeRO、FSDP |
| 只想低成本微调大模型 | LoRA、QLoRA |
| 训练吞吐低且硬件支持好 | BF16 / AMP |

微调大模型时，8-bit 优化器适合作为第一步降显存手段，因为它改动小、心智负担低、能保留常规 Adam 训练方式。如果还是放不下，再考虑更激进的分片、参数高效微调或权重量化方案。

边界也很明确：如果你要做纯推理，8-bit 优化器没有意义，因为推理没有 Adam 状态；如果你使用的是 SGD 且几乎没有额外状态，收益也不会像 Adam 那么明显；如果训练不稳定来自学习率、数据质量或损失缩放问题，换优化器状态精度也不一定解决。

---

## 参考资料

| 类型 | 链接 | 用途 |
|---|---|---|
| 论文 | `8-bit Optimizers via Block-wise Quantization` | 理论定义 |
| 官方文档 | `bitsandbytes 8-bit optimizers` | 使用方法 |
| 官方实现总览 | `bitsandbytes README` | 工程入口 |
| 细节文档 | `Optimizer overview` | 机制细节 |

1. [8-bit Optimizers via Block-wise Quantization](https://huggingface.co/papers/2110.02861)
2. [bitsandbytes 8-bit optimizers](https://huggingface.co/docs/bitsandbytes/optimizers)
3. [bitsandbytes README](https://github.com/bitsandbytes-foundation/bitsandbytes)
4. [Optimizer overview](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/optim_overview)

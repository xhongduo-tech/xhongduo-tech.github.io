## 核心结论

Qwen2.5 的主干仍然是 decoder-only Transformer，也就是“只看左侧上下文、按 token 逐步生成”的自回归架构。真正决定它工程可用性的，不是这一点本身，而是它在注意力、位置编码、前馈层和归一化上的一组组合式选择：GQA、RoPE、SwiGLU、Pre-RMSNorm，以及面向超长上下文的 YaRN 扩展。

这组设计解决的是同一个矛盾：上下文一旦从 8K、32K 拉到 128K 级别，先爆掉的通常不是理论算力，而是 KV Cache、显存带宽和长序列下的训练稳定性。Qwen2.5 的路线不是只靠“参数更多”硬推，而是把问题拆成几层分别处理：

- 用 GQA 压缩 Key/Value 头数量，直接降低 KV Cache。
- 用 RoPE 保留相对位置关系，并为长度外推提供更自然的基础。
- 用 YaRN 对 RoPE 做长上下文插值和再适配，把原生窗口扩展到更长范围。
- 用 SwiGLU 提高前馈层表达能力。
- 用 Pre-RMSNorm 让深层网络在训练和推理时更稳定。

如果只记一张表，先记这个：

| 模块 | 白话解释 | 解决的问题 | 直接影响 |
| --- | --- | --- | --- |
| GQA | 多个 Query 头共用较少的 K/V 头 | KV Cache 太大 | 长上下文推理更省显存和带宽 |
| RoPE | 把位置信息写进向量旋转角度 | 模型不知道 token 前后关系 | 比绝对位置表更适合长距离外推 |
| YaRN | 对 RoPE 的位置频率做插值并继续适配 | 原生窗口不够长 | 可把 32K 级窗口扩到 128K 级 |
| SwiGLU | 带门控的前馈激活 | 传统 FFN 表达力不够 | 一般比 ReLU/GELU-FFN 更有效 |
| Pre-RMSNorm | 子层计算前先归一化 | 深层训练不稳、残差易漂移 | 更稳、更容易扩深度和参数 |
| MoE | 每次只激活少数专家网络 | 参数容量和单次成本冲突 | 总容量更大，但单次激活成本更低 |

对初级工程师最重要的，不是记住“Qwen 很强”，而是理解它为什么在长文本和高吞吐场景下更实用。以 `Qwen2.5-32B-Instruct` 为例，官方模型卡给出的原始配置是 `max_position_embeddings = 32768`，并说明通过 `rope_scaling.type = "yarn"` 可以把可处理长度扩到 `131072` token。这个差异很关键，因为它意味着系统可以直接处理整份年报、长合同、长代码仓摘要，而不是必须先切块再拼接。

MoE 则是另一条路线。像 `Qwen2-57B-A14B` 这类模型，总参数量更大，但每个 token 只激活一部分专家，活跃参数量约为 `14B`。它的价值不是“白拿 57B 效果”，而是在容量和每次推理成本之间做折中。需要强调的是，Qwen2.5 开源主线是 dense 模型；MoE 在这里更适合拿来理解 Qwen 系列另一类工程权衡，而不是当作 Qwen2.5 主干的一部分。

---

## 问题定义与边界

本文讨论的不是“Qwen 所有版本的全部细节”，而是技术报告和官方模型卡里最关键的几个架构选择，以及这些选择分别在工程上解决什么问题。

先定义“长上下文”。这里主要指 `100K` 级输入，而不是 4K、8K 这种常规聊天长度。到了这个量级，第一瓶颈通常不是矩阵乘法本身，而是 KV Cache。KV Cache 可以理解为模型为历史 token 保留的注意力记忆。新 token 生成时，不需要把历史序列重新算一遍，而是复用已经存下来的 Key 和 Value。

如果使用标准多头注意力，KV Cache 大小近似正比于：

$$
\text{KV Cache Elements} \propto 2 \times L \times H_{kv} \times d_{head} \times N_{layer}
$$

其中：

- $L$ 是当前上下文长度。
- $H_{kv}$ 是 Key/Value 头数。
- $d_{head}$ 是每个头的维度。
- $N_{layer}$ 是 Transformer 层数。
- 前面的 $2$ 表示同时存 `K` 和 `V`。

如果再乘上每个元素占用的字节数 $b$，可以得到字节量近似式：

$$
\text{KV Cache Bytes} \approx 2 \times L \times H_{kv} \times d_{head} \times N_{layer} \times b
$$

这就是为什么长上下文问题本质上先变成“缓存问题”。当 $L$ 从 `8K` 变成 `128K`，只看长度这一项，缓存规模就会直接放大 `16` 倍。很多部署瓶颈不是模型算不动，而是显存先装不下，或者显存带宽先跟不上。

本文主要覆盖四类任务边界：

| 任务问题 | 典型输入长度 | 主要瓶颈 | 相关设计 |
| --- | --- | --- | --- |
| 普通问答、客服、助手 | 1K-8K | 首 token 延迟、生成质量 | 默认 RoPE，通常不必启用 YaRN |
| 长文档摘要、长合同审查、代码仓扫描 | 32K-128K | KV Cache、位置外推、吞吐下降 | GQA + RoPE + YaRN |
| 数学与代码推理 | 长短都可能 | 数据分布、推理链稳定性 | 结构 + 专项数据配比 |
| 大模型降本部署 | 任意 | 单次推理成本、并行通信 | Dense 与 MoE 取舍 |

还有三个边界必须说明。

第一，`32K 原生窗口` 和 `128K 扩展窗口` 不是同一件事。对 `Qwen2.5-32B-Instruct`，官方 `config.json` 中的 `max_position_embeddings` 是 `32768`。模型卡同时给出 YaRN 配置，用于把长文本处理能力扩到 `131072` token。这说明长上下文能力并不是“把最大长度参数改大”那么简单，而是位置编码策略和继续适配共同作用的结果。

第二，Qwen2.5 在代码和数学上的提升，不应简单归因于“结构更先进”。官方博客明确把 Qwen2.5、Qwen2.5-Coder、Qwen2.5-Math 分成不同方向的模型族。也就是说，结构提供的是能力上限和效率基础，实际表现仍然高度依赖训练数据和后训练策略。对初学者来说，这里最容易犯的错是把“架构选择”和“数据分布”混成一件事。

第三，MoE 讨论的是 Qwen 系列中的另一条设计路线。`Qwen2-57B-A14B` 官方模型卡显示它是 `57B` 总参数、`14B` 激活参数的 MoE 模型。它适合放在“容量与成本权衡”这一节讨论，但不应该误写成“Qwen2.5 所有模型都采用 MoE”。

一个典型工程边界可以这样理解：如果企业要处理一份 `13 万 token` 左右的年报，并直接让模型生成 `8K` 摘要，那么问题已经不是“能不能读一段长文字”，而是“能不能把整份材料一次送进去，并且系统仍然跑得动”。Qwen2.5 的价值正是在这里体现出来。

---

## 核心机制与推导

先看 GQA。GQA 全称是 Grouped Query Attention，可以直译为“分组查询注意力”。它的核心做法是：`Query` 头保留较多，但 `Key/Value` 头减少，并由多组 Query 共享。

标准多头注意力里，如果有 $H$ 个 Query 头，通常也会有 $H$ 个 Key 头和 $H$ 个 Value 头。GQA 把 K/V 头缩成 $H_{kv}$ 个，于是 KV Cache 的近似缩减比例就是：

$$
\frac{\text{KV Cache}_{\text{GQA}}}{\text{KV Cache}_{\text{MHA}}}
\approx
\frac{H_{kv}}{H}
$$

`Qwen2.5-32B-Instruct` 官方配置里：

- `num_attention_heads = 40`
- `num_key_value_heads = 8`

代入后得到：

$$
\frac{8}{40} = 0.2
$$

这表示在其他条件相同的情况下，GQA 的 KV Cache 大约只有标准多头注意力的 `20%`，也就是约 `5` 倍压缩。这里说“约”，是因为系统真实显存占用还会受到张量并行、缓存管理策略、数据类型和框架实现影响，但作为架构层面的主导项，这个结论是成立的。

对新手更容易理解的说法是：提问的人还是 40 个，但记笔记的人只保留 8 个。提问能力没有直接降成 8 份，但存储成本明显下降。

再看 RoPE。RoPE 全称 Rotary Position Embedding，旋转位置编码。它不再把“第几个 token”写成一个单独的位置向量加到词向量上，而是把位置信息编码成向量平面内的旋转角度。对每一对二维分量，可以写成：

$$
\begin{pmatrix}
x'_1 \\
x'_2
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}
$$

其中：

- $m$ 是位置索引。
- $\theta$ 是与维度相关的角频率。
- 旋转后的 $(x'_1, x'_2)$ 同时携带了原始内容和位置信息。

RoPE 的关键优点不是“公式更复杂”，而是它天然更适合表达相对位置信息。两个 token 的注意力关系，最终跟它们的相对位移有更直接的联系。这也是为什么 RoPE 比传统绝对位置表更适合作为长上下文扩展的基础。

但 RoPE 也不是无限可外推。模型如果只在 `32K` 范围内见过位置角度分布，直接把长度硬拉到 `128K`，高频部分通常会失真，注意力模式也会变乱。YaRN 的作用就在这里。

YaRN 可以概括成两步：

1. 对 RoPE 的位置频率分布做插值或重标定。
2. 用较少量的继续训练，让模型重新适应扩展后的长程位置关系。

它不是简单修改一个 `max_length` 参数，而是在“位置映射”和“继续适配”之间做配合。YaRN 论文给出的结论是：相对已有扩窗方法，它用更少的训练 token 和更少的训练步数，就能把上下文扩到更长。

再看 SwiGLU。标准 FFN 大致是：

$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x)
$$

SwiGLU 则可以写成：

$$
\text{SwiGLU}(x) = W_2 \big( \text{Swish}(W_g x) \odot (W_u x) \big)
$$

其中：

- $W_g x$ 走门控分支。
- `Swish(z) = z \cdot \sigma(z)`。
- $\odot$ 是逐元素乘法。
- $W_u x$ 是内容分支。

直观上，SwiGLU 的意思不是“把激活函数换成 SiLU”这么简单，而是“让一条分支决定另一条分支有多少信息通过”。这会让前馈层对复杂模式的选择更细，也就是表达能力更强。

Pre-RMSNorm 解决的是稳定性问题。RMSNorm 的定义可以写成：

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

它和 LayerNorm 的差别在于：RMSNorm 只做尺度归一化，不显式做均值中心化。计算更简单，开销更小。放在 Pre-Norm 结构里，子层输入先被归一化，再进入注意力或 FFN，残差流通常更稳定，深层训练也更容易收敛。

最后看 MoE。MoE 全称 Mixture of Experts，可以写成：

$$
p = \text{softmax}(G(x)), \qquad
y = \sum_{i \in \text{top}_k(p)} p_i E_i(x)
$$

这里：

- $G(x)$ 是路由器，决定把 token 发给哪些专家。
- $E_i(x)$ 是第 $i$ 个专家网络。
- `top-k` 表示只保留分数最高的几个专家参与计算。

它的工程意义是：总参数量可以很大，但每个 token 不需要激活全部参数。对容量和推理成本的折中，这比单纯把 dense 模型做大更灵活。但它也会引入额外问题，例如路由不均衡、专家热点、跨卡 all-to-all 通信。

把这些机制放在一起看，Qwen 系列的工程逻辑就很清楚了：

| 机制 | 本质动作 | 主要收益 | 代价 |
| --- | --- | --- | --- |
| GQA | 减少 K/V 头 | 降低 KV Cache、减轻带宽压力 | 头共享过多可能伤及质量 |
| RoPE | 用旋转编码位置 | 更自然表达相对位置 | 原生长度外推仍有限 |
| YaRN | 重标定位置频率并适配 | 把上下文扩到更长 | 短文本可能受影响 |
| SwiGLU | 用门控增强 FFN | 提高表达能力 | 参数与实现略复杂 |
| Pre-RMSNorm | 子层前归一化 | 深层训练更稳 | 不是性能提升的唯一来源 |
| MoE | 稀疏激活专家 | 更大容量/更低单次激活成本 | 路由和通信复杂 |

---

## 代码实现

如果目标是部署支持 `128K` 上下文的 Qwen2.5 变体，最常见的入口确实是模型配置里的 `rope_scaling`。但在动手之前，更值得先把几个基础量算清楚，否则只会把“长上下文很强”停留在口号层面。

先给一个完全可运行、无第三方依赖的 Python 例子，用来计算 GQA 的 KV Cache 压缩比例和近似缓存占用：

```python
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int  # bf16/fp16=2, fp32=4


def kv_cache_ratio(num_query_heads: int, num_kv_heads: int) -> float:
    assert num_query_heads > 0
    assert num_kv_heads > 0
    assert num_query_heads % num_kv_heads == 0
    return num_kv_heads / num_query_heads


def kv_cache_bytes(seq_len: int, cfg: AttentionConfig) -> int:
    assert seq_len > 0
    assert cfg.num_layers > 0
    assert cfg.num_kv_heads > 0
    assert cfg.head_dim > 0
    assert cfg.dtype_bytes > 0

    # 2 表示同时缓存 K 和 V
    return (
        2
        * seq_len
        * cfg.num_layers
        * cfg.num_kv_heads
        * cfg.head_dim
        * cfg.dtype_bytes
    )


def format_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


qwen25_32b = AttentionConfig(
    num_layers=64,
    num_query_heads=40,
    num_kv_heads=8,
    head_dim=128,   # 5120 hidden_size / 40 heads
    dtype_bytes=2,  # bf16
)

ratio = kv_cache_ratio(qwen25_32b.num_query_heads, qwen25_32b.num_kv_heads)
compression = 1 / ratio

cache_32k = kv_cache_bytes(32768, qwen25_32b)
cache_128k = kv_cache_bytes(131072, qwen25_32b)

assert abs(ratio - 0.2) < 1e-9
assert abs(compression - 5.0) < 1e-9
assert cache_128k == cache_32k * 4

print("GQA KV ratio:", ratio)
print("Compression factor:", compression)
print("Approx KV cache at 32K:  %.2f GiB" % format_gib(cache_32k))
print("Approx KV cache at 128K: %.2f GiB" % format_gib(cache_128k))
```

这段代码表达了两个工程事实：

1. `40` 个 Query 头配 `8` 个 KV 头时，缓存比例就是 `0.2`，也就是约 `5` 倍压缩。
2. 即便已经用了 GQA，长度从 `32K` 到 `128K`，KV Cache 仍然会再扩大 `4` 倍。

这也是为什么“有 GQA”并不等于“长上下文免费”。

再给一个最小 RoPE 旋转示例。它不是完整训练代码，但能帮助新手理解“位置如何写进向量角度”：

```python
import math


def rope_rotate_pair(x1: float, x2: float, position: int, theta: float):
    angle = position * theta
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y1 = cos_a * x1 - sin_a * x2
    y2 = sin_a * x1 + cos_a * x2
    return y1, y2


y1, y2 = rope_rotate_pair(1.0, 0.0, position=10, theta=0.01)
print(round(y1, 6), round(y2, 6))
```

如果你只看概念还不够，可以把它理解成：同样的内容向量，随着位置不同，会被旋转到不同方向；注意力层后续利用这种方向差异感知顺序和距离。

再看 YaRN 的部署配置。官方模型卡给出的典型写法是：

```json
{
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```

这几个字段分别表示：

- `original_max_position_embeddings`：原始训练窗口。
- `factor: 4.0`：把长度按比例扩展到约 4 倍。
- `type: "yarn"`：使用 YaRN 的 RoPE 扩展方式。

如果要把这个逻辑写成部署前的显式判断，可以这样做：

```python
def choose_rope_scaling(input_tokens: int):
    assert input_tokens > 0

    if input_tokens <= 32768:
        return None

    return {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }


assert choose_rope_scaling(8000) is None

cfg = choose_rope_scaling(100000)
assert cfg["type"] == "yarn"
assert cfg["factor"] == 4.0
assert cfg["original_max_position_embeddings"] == 32768
```

这个判断在工程里很重要，因为不少推理框架对 YaRN 的支持是静态的。官方模型卡也明确提醒：当前一些框架只支持 static YaRN，也就是缩放因子不会随着输入长度动态变化，因此短文本性能可能受影响。换句话说，你往往不能指望同一个实例在每个请求到来时自动选择“要不要扩窗”，而是需要在部署层显式分流。

如果希望给出一个更贴近真实部署的、可直接运行的 Hugging Face 示例，可以写成下面这样。它的重点不是演示性能，而是展示配置注入方式：

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

config = AutoConfig.from_pretrained(MODEL_NAME)
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype="auto",
    device_map="auto",
)

print("Loaded model with rope_scaling =", model.config.rope_scaling)
```

这段代码依赖 `transformers` 和真实模型权重，因此只有在本地环境满足条件时才能运行。它说明的是一个部署原则：长上下文能力通常是实例级配置，而不是每次请求里的随手开关。

一个更实际的系统方案通常是这样：

| 请求类型 | 输入长度 | 实例配置 | 原因 |
| --- | --- | --- | --- |
| 客服、FAQ、普通助手 | 1K-8K | 默认实例 | 吞吐更高，短文本质量更稳 |
| 长报告总结 | 32K-128K | YaRN 长上下文实例 | 避免切块和跨块丢信息 |
| 代码仓级审查 | 32K-128K | YaRN + 更大显存预算 | 长上下文和缓存压力更明显 |
| 高并发在线服务 | 混合 | 前置路由分流 | 避免所有请求都承担长窗代价 |

这类分流比“一个实例全包”更符合真实成本结构。

---

## 工程权衡与常见坑

第一类坑是：长上下文扩展不是白送的。YaRN 能把上下文从 `32K` 拉到 `128K` 级，但并不意味着所有任务都会同步变好。官方模型卡已经提示，static YaRN 可能影响较短输入的性能。原因不神秘，位置编码分布被重新拉伸之后，模型原本在短区间上学到的最优几何关系会被扰动。

因此，工程上的正确问法不是“能不能开 YaRN”，而是“我的请求分布值不值得长期承担这个代价”。

第二类坑是：GQA 省了显存，不等于系统整体一定更快。GQA 主要优化的是 KV Cache 和由此带来的带宽压力，所以在单卡长文本推理里，收益通常很直接。但多卡场景不一样。如果你同时用了张量并行、流水并行，甚至 MoE 专家并行，那么通信可能成为新瓶颈。尤其在没有 NVLink、NVSwitch 之类高速互联时，理论上的 FLOPs 节省很可能被 `AllGather`、`ReduceScatter` 或 `all-to-all` 抵消。

第三类坑是：把“模型更擅长代码和数学”误解成“任何代码和数学题都更稳”。官方博客的表述更接近“通过专门模型和训练过程增强相关能力”，这意味着结果仍然取决于任务形式。比如：

- 补全类代码任务，结构化上下文是否完整很关键。
- 数学推理任务，是否允许更长中间推导、是否使用工具调用也很关键。
- 即使底座相同，Qwen2.5、Qwen2.5-Coder、Qwen2.5-Math 的适用场景也不完全相同。

第四类坑是：把长上下文当成检索系统的替代品。`128K` 输入确实能减少切块和人工拼接，但它不自动替代检索、排序、去重、结构化预处理。对超长文档，先做段落筛选、再把高价值片段送给模型，很多时候仍然更稳、更便宜。

第五类坑是：只看参数量，不看激活参数和缓存结构。Dense 模型的参数量和每次激活参数量基本一致；MoE 模型不是这样。`57B-A14B` 里，“57B”告诉你总容量，“A14B”告诉你单 token 的活跃计算规模。部署估算时，这两个数都要看。

下面这张表可以直接拿来做部署前判断：

| 场景 | 建议 | 原因 | 常见误区 |
| --- | --- | --- | --- |
| 输入几乎都小于 32K | 关闭 YaRN | 保持短文本质量和吞吐 | 以为长上下文配置一定更强 |
| 输入经常接近 128K | 开启 YaRN | 避免切块、减少跨块遗漏 | 忽视吞吐下降和缓存膨胀 |
| 单卡长文本推理 | 优先关注 GQA 和缓存预算 | KV Cache 是一线瓶颈 | 只看参数量，不看缓存 |
| 多卡高并发部署 | 先测通信，再看理论算力 | 通信常常盖过计算收益 | 把理论 FLOPs 当实际延迟 |
| 使用 MoE | 重点测路由均衡和并行策略 | 热门专家会拖慢尾延迟 | 只看激活参数，不看通信 |

一个很实用的经验是：当你怀疑“模型为什么这么慢”时，先别盯着前向矩阵乘法，先把这四个问题问清楚：

1. 当前平均上下文多长？
2. KV Cache 占了多少显存？
3. 是否因为长窗配置牺牲了短文本吞吐？
4. 多卡时是不是通信在拖后腿？

很多问题到这里就已经定位了。

---

## 替代方案与适用边界

如果业务主要落在 `32K` 以下，最稳妥的方案通常不是启用 YaRN，而是保持默认配置。原因很直接：原生窗口已经够用，额外做位置插值只会引入质量波动和吞吐损失。

如果输入只是偶尔超长，而且任务对局部精确召回要求高，那么分块和滑动窗口仍然是强竞争方案。它的本质不是“老办法”，而是把计算预算集中在局部高价值片段上。比如客服日志、工单串联、短会议纪要，很多时候并不要求模型对 100K 全局同时建模，这时直接上 `128K` 反而是浪费。

如果你在做更大规模部署，还要在 Dense 和 MoE 之间做结构级选择：

| 方案 | 适用上下文 | 成本特点 | 部署复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| Dense 默认窗口 | 32K 内 | 成本可预测，吞吐稳定 | 低 | 常规聊天、客服、知识助手 |
| Dense + YaRN | 32K-128K | 长文本更有优势 | 中 | 长报告、长合同、代码仓级分析 |
| Chunking / 滑动窗口 | 任意 | 需要额外编排与汇总 | 中 | 可容忍分段处理的业务 |
| MoE | 任意 | 总容量更大，单次激活成本低 | 高 | 大规模服务、追求容量成本比 |

还可以把选型进一步压缩成三条规则：

- 如果大多数请求在 `8K` 内，不要为了极少数长文档把所有实例默认切到 YaRN。
- 如果长文本是核心业务，例如投研、法务审查、代码仓级审查，再考虑单独部署长上下文实例。
- 如果预算受限，但又希望保留更大的模型容量，可以考虑 MoE；前提是团队能处理好多卡调度、专家并行和通信诊断。

换句话说，Qwen2.5 的这些设计不是“必须全部打开”的开关，而是一组可组合的工程工具：

- GQA 更像基础设施优化，通常值得保留。
- RoPE 是长上下文的基础位置编码。
- YaRN 是扩窗工具，不应默认全开。
- SwiGLU 和 Pre-RMSNorm 主要服务于表达能力与稳定性。
- MoE 是另一条容量/成本折中路线，不是所有部署都需要。

真正成熟的系统，不是把这些词全堆进配置文件，而是知道什么时候启用、什么时候克制。

---

## 参考资料

- [Qwen2.5-32B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [Qwen2.5-32B-Instruct 官方 `config.json`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json)
- [Qwen 官方博客：Qwen2.5 发布说明](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2-57B-A14B 官方模型卡（MoE）](https://huggingface.co/Qwen/Qwen2-57B-A14B)
- [Qwen2-57B-A14B 官方 `config.json`](https://huggingface.co/Qwen/Qwen2-57B-A14B/blob/main/config.json)
- [GQA 原始论文：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [YaRN 原始论文：YaRN: Efficient Context Window Extension of Large Language Models](https://openreview.net/forum?id=wHBfxhZu1u)
- [RMSNorm 原始论文：Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [SwiGLU 参考论文：GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [NVIDIA 关于 NVLink / NVSwitch 与大模型推理通信的技术说明](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/)

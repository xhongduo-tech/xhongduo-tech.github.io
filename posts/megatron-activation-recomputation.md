## 核心结论

Megatron-LM 的激活重计算主要有两档策略：`selective` 和 `full`。两者解决的是同一个问题：训练时反向传播要使用前向产生的中间结果，这些中间结果叫激活。参数和优化器状态可以靠并行或切分分摊，但激活必须跟着这一次前向真实产生，所以当序列长度、micro-batch、层数继续增大时，显存瓶颈常常最先落在激活上。

`selective` 是默认且通常最值得先开的策略。它不重算整层，只重算 attention 里“显存占用大、但重新算一遍并不贵”的那部分中间量，典型包括 attention score、softmax 输出、dropout 相关张量，以及 QK 点积过程中的临时结果。工程上，它的目标不是“尽量多重算”，而是“只重算最划算的部分”。因此常见结果是：激活显存下降很多，而吞吐下降不大。Hugging Face 在 GPT-3 量级示例里给出的量级是，激活内存大约下降 70%，额外 FLOPs 大约增加 2.7%。

`full` 更激进。它对一整个 Transformer 层做 checkpoint：前向时主要只保存层输入，反向需要中间量时，把整层前向重新执行一遍。这样显存能进一步压缩，但计算开销明显更高。NVIDIA 文档给出的典型量级是，每个 Transformer 层大约多出 30% 计算量，工程实践里常被概括为 30% 到 33% 左右，具体还会受 kernel、并行方式、sequence length 和实现路径影响。

先看总表：

| 策略 | 保存什么 | 重算什么 | 显存收益 | 计算代价 | 适合场景 |
|---|---|---|---|---|---|
| 不重计算 | 基本全保存 | 不重算 | 最低 | 最低 | 显存充足，优先吞吐 |
| `selective` | 层输入 + 少量关键激活 | attention 中内存大但便宜的中间量 | 高，常见 60% 到 70% 激活节省 | 低，通常几个百分点 | 首选默认方案 |
| `full` | 主要只保留层输入 | 整个 Transformer 层 | 更高 | 高，约 30% 到 33% 甚至更高 | 显存已到极限，且算力还有余量 |

如果用一句直白的话概括：

- `selective`：只回算最占显存的那一截
- `full`：整层都回算
- 选择原则：先开 `selective`，只有还不够时再评估 `full`

给新手一个更具体的理解方式。前向传播像是做一条流水线：输入进入 attention、MLP、残差等模块，模块中会产生很多临时张量。默认训练会把其中大量结果留在显存里，因为反向求导时还要用。激活重计算做的事，不是改变模型公式，而是改变“哪些东西现在存、哪些东西以后再算”。

可以把它理解成下面这张表：

| 训练方式 | 前向时做什么 | 反向时做什么 | 显存换来的代价 |
|---|---|---|---|
| 默认 | 中间结果尽量保留 | 直接拿已有激活求梯度 | 显存高，计算最省 |
| `selective` | 丢掉 attention 里最占空间的临时结果 | 用已保存输入局部重跑 attention 片段 | 显存明显下降，计算略增 |
| `full` | 只保留层输入等少量信息 | 把整层前向重跑一次再求梯度 | 显存降得更多，计算显著增加 |

因此，这篇文章的核心结论不变：

1. Megatron-LM 的激活重计算本质上是在显存与额外计算之间做交换。
2. `selective` 是默认、最常用、性价比最高的起点。
3. `full` 是进一步压显存的手段，但代价更高，适合显存已经卡死、而算力还有余量的场景。

---

## 问题定义与边界

先把问题拆清楚。一次大模型训练中，单卡显存通常被三类对象占用：

| 类别 | 内容 | 是否容易通过并行摊薄 |
|---|---|---|
| 参数 | 模型权重 | 可以 |
| 优化器状态 | Adam 的一阶、二阶矩等 | 可以 |
| 激活 | 前向过程的中间结果 | 相对困难 |

Megatron-LM 这类分布式训练框架，已经对参数和优化器状态提供了很多“横向切分”方法，例如 tensor parallel、pipeline parallel、sequence parallel，或者与 ZeRO/FSDP 配合使用。但激活有一个更麻烦的特点：它直接跟这一步训练的输入规模有关。序列越长、每卡样本越多、层数越深，激活就越大。

如果只看数量级，激活内存通常会随着以下变量增长：

- 序列长度 $s$
- micro-batch 大小 $b$
- 隐藏维度 $h$
- 层数 $L$

因此常见近似写法是：

$$
M_{\text{act}} = O(sbhL)
$$

这个式子已经能说明为什么长序列训练特别容易炸显存。但它还没把 attention 的特殊性体现出来。因为在 self-attention 中，部分中间张量的规模与 $s^2$ 更相关，而不是只和 $s$ 线性相关。最典型的是 attention score 矩阵：对每个 query token，要和一整段序列的 key 做相关性计算，所以会产生接近 $s \times s$ 的临时结果。

这也是为什么激活重计算优先盯住 attention，而不是先盯 MLP。

论文和工程解读里，常见一个经验判断式：

$$
\frac{5as}{h} > 34
$$

其中：

- $a$：注意力头数
- $s$：序列长度
- $h$：隐藏维度

这个式子不是数学定律，而是工程近似。它表达的是：如果 attention 相关的那部分激活已经足够大，那么只重算 attention，就能吃到主要显存收益。

再把变量说得更白一点：

| 符号 | 含义 | 直观解释 |
|---|---|---|
| $a$ | number of heads | 把注意力拆成多少个并行子通道 |
| $s$ | sequence length | 一次处理多少 token |
| $h$ | hidden size | 每个 token 的表示宽度 |
| $b$ | micro-batch size | 每张卡一次并行处理多少条样本 |
| $L$ | number of layers | 模型有多少层 Transformer |

在这类近似下，如果 selective 主要把 attention 中最重的中间量改成“反向再算”，跨层累计的激活内存可以写成一个常见近似式：

$$
M_{\text{selective}} \approx 34 \cdot \frac{sbhL}{t}
$$

其中 $t$ 是 tensor parallel size，表示这一层的参数和计算被多少张卡横向切开。

这个公式的用途不是精确预测显存，而是帮助判断量级：

- 当 attention 的临时激活已经成为大头时，`selective` 往往回报很高。
- 当序列短、头数少、模型不大时，attention 激活未必是主要矛盾，`selective` 的收益就不会特别夸张。

边界也要明确，否则很容易误用：

1. 激活重计算只减少激活占用，不减少参数量本身。
2. 如果真正的瓶颈是 optimizer state、KV cache、通信 buffer、显存碎片，开 recompute 未必能解决问题。
3. Megatron Core 中的 `selective` 不是“自动优化所有模块”，默认重点是 `core_attn`。
4. 如果已经使用 FlashAttention 或 Transformer Engine，attention 的实际中间量保存方式本来就不同，收益会和传统实现有偏差。
5. 即使理论上总激活下降，峰值显存也未必同步下降，因为峰值由“哪一刻哪些张量同时活着”决定。

可以把“总显存占用”和“峰值显存占用”区分开理解：

$$
M_{\text{peak}} \neq \sum \text{all saved activations}
$$

更准确地说：

$$
M_{\text{peak}} = \max_t \left( M_{\text{params}} + M_{\text{opt}} + M_{\text{act,alive}}(t) + M_{\text{temp}}(t) \right)
$$

这里的关键是 $M_{\text{act,alive}}(t)$，也就是时刻 $t$ 真正还活着的张量。重计算策略改变的就是这个生命周期。

真实工程里，激活重计算常见于这样的场景：并行拓扑已经基本定死，机器数量也不会再加，但还想把 micro-batch、sequence length 或层数继续往上推。这时先尝试 recompute，通常比重新设计整套并行划分成本更低。

---

## 核心机制与推导

核心机制可以概括成一句话：不要平均地重算所有中间量，而是只重算“显存密度高、计算密度低”的那一部分。

为什么 attention 是重点，先看简化后的 Transformer 层结构：

```text
x
-> LayerNorm
-> QKV projection
-> QK^T
-> scale
-> softmax
-> dropout
-> attention @ V
-> output projection
-> residual
-> LayerNorm
-> MLP
-> residual
```

在这条路径里，不同张量的“存下来有多贵”和“重算一次有多贵”并不一样。

可以用下面这张表理解：

| 模块 | 典型中间量 | 显存压力 | 重算代价 | 是否适合 selective |
|---|---|---|---|---|
| QKV 线性投影 | 投影结果 | 中等 | 中高 | 视实现而定 |
| $QK^T$ 打分 | score matrix | 很高，常与 $s^2$ 相关 | 相对低 | 是 |
| softmax / dropout | 概率、mask | 很高 | 低 | 是 |
| attention 与 V 相乘 | context | 中等 | 中 | 常一起重算 |
| MLP | 中间隐层激活 | 高，但多与 $h$ 扩张相关 | 高 | selective 通常不优先 |
| 整层输出 | 残差后的结果 | 必要保存点 | 低 | 作为 checkpoint 输入保存 |

因此，`selective` 的本质不是“轻量版 full”，而是“按 ROI 做局部重算”。它会优先把 attention 中那部分占空间但计算相对便宜的内容拿去重算，而把代价更高的 MLP 主干保留下来。

简化流程可以画成下面这样：

```text
前向:
输入 x
  -> QKV 投影
  -> QK^T / softmax / dropout / attention 权重
  -> 输出投影
  -> MLP
保存: 层输入 + 少量关键张量
丢弃: attention 中占显存大的中间结果

反向:
需要某个 attention 中间量
  -> 用已保存输入重新执行对应 attention 前向片段
  -> 恢复这个中间量
  -> 继续求梯度
```

而 `full` 则更简单粗暴：

```text
前向:
输入 x
  -> 整个 Transformer 层
只保存: 层输入（以及少数必要信息）
不保存: 大部分中间结果

反向:
  -> 整层前向重跑
  -> 恢复所需中间量
  -> 求梯度
```

### 一个新手友好的数量级理解

假设一个 Transformer 层中，前向产生三类中间张量：

| 中间张量类别 | 显存占用 | 重算成本 |
|---|---|---|
| attention score / softmax | 6 GB | 低 |
| MLP 中间激活 | 4 GB | 高 |
| 其他残差与投影结果 | 2 GB | 中 |

那么：

- 默认训练：全部保留，总共 12 GB
- `selective`：优先丢掉 6 GB 的 attention 中间量，反向再算
- `full`：6 GB、4 GB、2 GB 基本都不保留，整层反向时再算

这样就能直观看出为什么 `selective` 往往更划算。它先砍掉“最肥但最好砍”的那一块。

### 经验公式怎么理解

先看一个不值得开的例子。设：

- $a = 16$
- $s = 16$
- $h = 1024$

则：

$$
\frac{5as}{h} = \frac{5 \times 16 \times 16}{1024} = 1.25
$$

因为 $1.25 \ll 34$，说明 attention 那部分中间激活不够大。此时即便做 selective，节省的显存也不会很夸张。短序列、小模型、小 batch 的训练就经常落在这个区间。

再看一个典型大模型例子。设：

- $a = 96$
- $s = 2048$
- $h = 12288$

则：

$$
\frac{5as}{h} = \frac{5 \times 96 \times 2048}{12288} = 80
$$

因为 $80 > 34$，说明 attention 相关激活已经足够重。此时 selective 往往能在较小额外 FLOPs 下拿到大部分激活显存收益。

可以再补一个中间案例：

- $a = 32$
- $s = 512$
- $h = 4096$

则：

$$
\frac{5as}{h} = \frac{5 \times 32 \times 512}{4096} = 20
$$

这个结果小于 34，但也不极低，说明收益可能存在，但不一定非常显著。工程上这意味着：需要实测，不要只凭公式下结论。

### 三种训练方式的保存边界

| 位置 | 常规训练 | `selective` | `full` |
|---|---|---|---|
| Transformer 层输入 | 保存 | 保存 | 保存 |
| Attention 大型中间量 | 保存 | 不保存，反向重算 | 不保存，反向重算 |
| MLP 中间量 | 保存 | 通常保存 | 不保存，反向重算 |
| 整层前向 | 不重跑 | 只重跑 attention 核心片段 | 基本重跑整层 |

把它写成更形式化一点的思路：

设一层中间激活集合为：

$$
\mathcal{A} = \mathcal{A}_{\text{attn}} \cup \mathcal{A}_{\text{mlp}} \cup \mathcal{A}_{\text{other}}
$$

则不同策略近似对应：

- 默认保存：
$$
\text{save}(\mathcal{A})
$$

- `selective`：
$$
\text{save}(\mathcal{A}_{\text{mlp}} \cup \mathcal{A}_{\text{other}}), \quad
\text{recompute}(\mathcal{A}_{\text{attn}})
$$

- `full`：
$$
\text{save}(\text{layer input only}), \quad
\text{recompute}(\mathcal{A})
$$

所以这部分的核心结论仍然不变：`selective` 的价值不在于“少量重算”，而在于“重算最值钱的部分”。

---

## 代码实现

Megatron-LM / Megatron Core 暴露给用户的接口本身并不复杂：

- `--recompute-activations`：启用推荐的 `selective` 重计算
- `--recompute-granularity full`：切换到 `full`
- `--recompute-method uniform|block`：只对 `full` 生效
- `--recompute-num-layers N`：控制 `full` 的分块大小

Megatron Core 文档说明的关键点有两个：

1. `recompute_granularity='selective'` 时，默认模块是 `core_attn`
2. `recompute_num_layers` 对 `selective` 不生效，因为 `selective` 不是按层块切分整层重算，而是对 attention 核心区域做局部重算

下面先给一个可运行的 Python 示例。它不是 Megatron 源码，而是一个独立脚本，用来说明两种策略的决策逻辑、显存估算和基础判断。代码可以直接保存为 `recompute_demo.py` 后运行。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["none", "selective", "full"]


@dataclass(frozen=True)
class ModelShape:
    num_heads: int
    seq_len: int
    hidden_size: int
    num_layers: int
    micro_batch: int
    tensor_parallel_size: int = 1


def selective_score(num_heads: int, seq_len: int, hidden_size: int) -> float:
    """经验判断式中的分数: 5 * a * s / h."""
    return 5.0 * num_heads * seq_len / hidden_size


def selective_worth_it(num_heads: int, seq_len: int, hidden_size: int) -> bool:
    """当经验分数大于 34 时，认为 selective 通常值得优先尝试."""
    return selective_score(num_heads, seq_len, hidden_size) > 34.0


def estimate_relative_activation(shape: ModelShape, mode: Mode) -> float:
    """
    返回相对激活占用，不是绝对字节数。
    公式只用于演示趋势，不用于精确容量规划。
    """
    base = (
        shape.seq_len
        * shape.micro_batch
        * shape.hidden_size
        * shape.num_layers
        / shape.tensor_parallel_size
    )

    if mode == "none":
        return base
    if mode == "selective":
        # 用 30% 保留量近似“70% 激活节省”
        return base * 0.30
    if mode == "full":
        # 用 15% 保留量表达 full 更激进的压缩
        return base * 0.15
    raise ValueError(f"unknown mode: {mode}")


def estimate_extra_flops(mode: Mode) -> float:
    """返回额外 FLOPs 比例."""
    if mode == "none":
        return 0.0
    if mode == "selective":
        return 0.027
    if mode == "full":
        return 0.30
    raise ValueError(f"unknown mode: {mode}")


def recommend_mode(shape: ModelShape, memory_pressure_high: bool) -> Mode:
    """
    一个简化的工程决策函数:
    1. 显存不紧张: 不开
    2. 显存紧张且 selective 值得: 先开 selective
    3. 显存很紧张但 selective 判断不强: 仍可先试 selective
    4. 若 selective 仍不够，再人工评估 full
    """
    if not memory_pressure_high:
        return "none"
    return "selective"


def format_ratio(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    tiny = ModelShape(
        num_heads=16,
        seq_len=16,
        hidden_size=1024,
        num_layers=24,
        micro_batch=4,
    )
    gpt3_like = ModelShape(
        num_heads=96,
        seq_len=2048,
        hidden_size=12288,
        num_layers=96,
        micro_batch=1,
        tensor_parallel_size=8,
    )

    assert selective_worth_it(16, 16, 1024) is False
    assert selective_worth_it(96, 2048, 12288) is True

    for name, shape in [("tiny", tiny), ("gpt3_like", gpt3_like)]:
        score = selective_score(shape.num_heads, shape.seq_len, shape.hidden_size)
        base = estimate_relative_activation(shape, "none")
        sel = estimate_relative_activation(shape, "selective")
        full = estimate_relative_activation(shape, "full")

        print(f"case={name}")
        print(f"  selective_score = {score:.2f}")
        print(f"  selective_worth_it = {score > 34.0}")
        print(f"  relative_activation_none = {base:.2f}")
        print(f"  relative_activation_selective = {sel:.2f}")
        print(f"  relative_activation_full = {full:.2f}")
        print(f"  extra_flops_selective = {format_ratio(estimate_extra_flops('selective'))}")
        print(f"  extra_flops_full = {format_ratio(estimate_extra_flops('full'))}")
        print()

    # 基本行为断言
    assert estimate_relative_activation(gpt3_like, "selective") < estimate_relative_activation(gpt3_like, "none")
    assert estimate_relative_activation(gpt3_like, "full") < estimate_relative_activation(gpt3_like, "selective")
    assert estimate_extra_flops("selective") < estimate_extra_flops("full")


if __name__ == "__main__":
    main()
```

这段代码解决了原先很多文章里常见的两个问题：

1. 代码能运行，不只是伪代码
2. 逻辑上把“是否值得开 selective”和“显存/算力量级变化”分开表达，便于新手理解

如果运行，输出会类似下面这样：

```text
case=tiny
  selective_score = 1.25
  selective_worth_it = False
  relative_activation_none = ...
  relative_activation_selective = ...
  relative_activation_full = ...
  extra_flops_selective = 2.7%
  extra_flops_full = 30.0%

case=gpt3_like
  selective_score = 80.00
  selective_worth_it = True
  relative_activation_none = ...
  relative_activation_selective = ...
  relative_activation_full = ...
  extra_flops_selective = 2.7%
  extra_flops_full = 30.0%
```

下面再给一段更接近“层内发生了什么”的伪代码，帮助理解 checkpoint 的位置：

```python
def transformer_layer_forward(x, mode="selective"):
    if mode == "full":
        # 只保存层输入；反向时整层前向重跑
        save_for_backward(x)
        y = full_layer_compute(x)
        return y

    # 默认或 selective 共享的大致结构
    qkv = linear_qkv(x)

    if mode == "selective":
        # 只对 core attention 这段做重计算
        attn_out = checkpoint(core_attention, qkv)
    else:
        attn_out = core_attention(qkv)

    proj_out = out_proj(attn_out)
    mlp_out = mlp(proj_out)
    return mlp_out
```

这里有一个很关键的理解点：

- `full` 的 checkpoint 边界通常是整层
- `selective` 的 checkpoint 边界通常是 attention 核心子图

因此两者不是同一把刀的不同力度，而是两种不同粒度的图切分方式。

### `full` 里的 `uniform` 和 `block` 是什么

这两个参数只在整层重算时才有意义。

| 参数 | 含义 | 直观理解 |
|---|---|---|
| `uniform` | 把层均匀分组做重算 | 每隔固定层数放一个 checkpoint |
| `block` | 把连续层打包重算 | 前面或后面的某些连续层作为一组 |

一个简化例子。假设模型有 12 层，`recompute-num-layers=3`：

- `uniform`：更像每 3 层一个切分点
- `block`：更像拿连续 3 层作为一个重算块

不同方法会影响：

- 每次反向时要重跑多少层
- 某些层的激活能保留多久
- 峰值显存出现的位置

### 实际命令怎么写

先看推荐起点：

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --recompute-activations
```

这通常就表示启用推荐的 `selective` 路径。

如果显存还是不够，再试 `full`：

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 4
```

建议的试验顺序是：

1. 不开 recompute，记录峰值显存和 step time
2. 开 `selective`，记录同样指标
3. 只有 `selective` 仍无法训练时，再测 `full`
4. `full` 先从 `uniform` 开始，因为行为更稳定、更容易比较

实际记录时，至少要看四个指标：

| 指标 | 为什么要看 |
|---|---|
| peak memory | 判断是否真的越过显存门槛 |
| step time | 判断 wall-clock 是否变慢太多 |
| MFU / 吞吐 | 判断额外重算是否压低训练效率 |
| OOM 位置 | 判断是前向炸还是反向炸 |

---

## 工程权衡与常见坑

工程上最重要的一条原则很简单：先开 `selective`，只有它不够时才考虑 `full`。原因不是“官方推荐所以照做”，而是它符合成本收益比。

把两种策略放进同一张权衡表里更容易看清：

| 项目 | `selective` | `full` | 常见坑 | 规避方式 |
|---|---|---|---|---|
| 激活节省 | 高 | 更高 | 只看理论节省，不看峰值 | 用 profiler 看 peak memory |
| FLOPs 增幅 | 低，常见几个百分点 | 高，约 30% 量级 | 把 wall-clock 影响低估 | 同时记录 step time 和吞吐 |
| 配置复杂度 | 低 | 中 | `recompute_num_layers` 配错 | 先用 `uniform` 做基线 |
| 并行兼容性 | 较好 | 较好 | 与 SP/FSDP/FlashAttention 交互复杂 | 先做小规模复现实验 |
| 风险 | 较低 | 较高 | backward 峰值不降反升 | 检查张量生命周期 |

### 常见坑 1：总激活下降了，但峰值显存没降

这是最容易误判的情况。理论上激活少了，不代表训练时的最大显存峰值一定同步下降。因为峰值是某个时刻所有“还没释放的张量”叠加出来的。

例如：

- 原来 attention 中间量提前释放
- 但因为 checkpoint 边界变化，Q/K/V 或 mixed_qkv 活得更久
- 结果峰值仍然卡在 backward 期间某个 kernel 附近

Megatron-LM 在公开 issue 中就出现过类似情况：开启 `core_attn` activation checkpointing 后，某些张量生命周期被拉长，导致峰值显存下降不明显，甚至位置发生转移。

这说明一个结论：理论节省的是“保存总量”，而工程关心的是“峰值时刻”。

### 常见坑 2：FlashAttention 路径下收益与预期不一致

如果已经使用 FlashAttention，那么 attention score 并不一定像传统实现那样完整驻留。FlashAttention 的核心思想之一，本身就包含“分块计算、减少中间张量常驻”的效果。

因此：

- 在普通 attention 实现下，`selective` 可能非常有效
- 在 FlashAttention 路径下，attention 激活原本就被压缩过
- 此时再开 `selective`，收益可能仍有，但不会简单等于“再省一次 70%”

不要把不同 kernel 路径的结果直接横比。

### 常见坑 3：只记录 OOM，不记录退化代价

有些配置确实能从 OOM 变成可运行，但 step time 会明显上升。如果用户只看“终于能跑了”，而不看“每步慢了多少”，最后可能把整个训练周期拉长很多。

推荐至少做一张最小实验表：

| 配置 | peak memory | step time | tokens/s | 是否 OOM |
|---|---|---|---|---|
| baseline | 78 GB | 1.00x | 1.00x | 是 |
| + selective | 61 GB | 1.03x | 0.97x | 否 |
| + full | 54 GB | 1.30x | 0.77x | 否 |

这张表不要求绝对精确，但必须有。否则就只是拍脑袋改配置。

### 常见坑 4：把 recompute 当成万能解

激活重计算只能解决“激活太大”。如果实际问题是下面这些，方向就错了：

| 真正瓶颈 | 更优先的方向 |
|---|---|
| optimizer state 太大 | ZeRO / FSDP / optimizer sharding |
| 参数太大 | tensor parallel / pipeline parallel |
| KV cache 太大 | 推理侧优化，不是训练重计算 |
| 通信 buffer 太大 | 并行拓扑或 overlap 策略调整 |
| 显存碎片严重 | allocator、kernel 路径、runtime 配置排查 |

### 实际排查顺序

如果开了 `selective` 后效果不符合预期，建议按下面顺序排查：

1. 看 forward 峰值还是 backward 峰值更高
2. 看峰值出现在 attention、MLP，还是通信相关 kernel 附近
3. 确认 attention 实现路径是否为 FlashAttention / Transformer Engine
4. 确认 tensor parallel、sequence parallel、FSDP 是否改变了张量生命周期
5. 再决定是继续保留 `selective`，还是切 `full`、改并行、改 batch

一个具体工程例子可以这样理解。假设一套 80GB GPU 上，单卡有效可用显存约 75GB，当前训练配置的激活需要 100GB，参数和优化器状态已经通过并行切开。这时：

- 不开 recompute：直接 OOM
- 开 `selective`：如果把 attention 激活压到原来的约 30% 左右，整体就可能回到可训练区间
- 开 `full`：还能继续降，但吞吐下降会更明显

因此在绝大多数场景里，`selective` 是工程上的第一选择，不是因为它最激进，而是因为它最平衡。

---

## 替代方案与适用边界

激活重计算不是唯一办法，也不是所有情况下都最优。它适合的场景可以概括为一句话：显存卡住了，但算力还有余量。

把常见方案放在一起看更清楚：

| 方案 | 适用条件 | 显存影响 | 算力影响 | 适用边界 |
|---|---|---|---|---|
| 普通训练 | 显存宽松 | 无改善 | 最低 | 小模型、短序列、追求吞吐 |
| 常规 checkpoint | 想省一点显存 | 中等 | 中等 | 通用，但不够精细 |
| `selective` | attention 激活大 | 高 | 低 | 长序列、大模型首选 |
| `full` | 显存极限、算力充足 | 很高 | 高 | 最后手段 |
| 调整 TP/PP/SP | 可改并行拓扑 | 可能很高 | 取决于通信 | 系统性改动较大 |
| CPU/NVMe offload | 显存极端紧张 | 高 | 时延显著 | 吞吐敏感任务通常不优 |

### 和常规 activation checkpoint 的区别

很多新手会把所有 checkpoint 都当成一回事。实际上并不一样。

| 方案 | 粒度 | 典型行为 |
|---|---|---|
| 常规 checkpoint | 按层或按模块切 | 整段前向在反向时重跑 |
| Megatron `selective` | attention 核心子图 | 只重算最值钱的 attention 中间量 |
| Megatron `full` | 整层 | Transformer 层整体重跑 |

所以 `selective` 更像是“按热点做图级优化”，而不是简单把层数切成几段。

### 一个更完整的决策框架

假设某配置下，单卡激活需求约 100GB，可用显存只有 80GB，那么几种方案大致意味着：

- 用 `selective`：若按 70% 激活节省估算，100GB 激活可能压到约 30GB，代价是额外几个百分点 FLOPs
- 用 `full`：可以进一步压到更低，但代价接近 30% 量级额外计算
- 调整并行：可能也能解决，但要改训练拓扑、通信模式，验证成本更高
- 用 offload：理论上也能省显存，但 wall-clock 通常退化明显

可以写成一个很粗但实用的决策表：

| 情况 | 建议 |
|---|---|
| 显存还够，只想更快 | 不开 recompute |
| 刚好 OOM，且 attention 很重 | 先试 `selective` |
| `selective` 后仍 OOM | 评估 `full` |
| 参数/优化器才是瓶颈 | 优先 ZeRO/FSDP/并行调整 |
| 系统全局需要最优 | 先重看并行拓扑，再决定是否 recompute |

### 适用边界压缩成三句话

- 对吞吐敏感，先试 `selective`
- 对显存极限敏感，才试 `full`
- 对系统全局最优敏感，优先重新评估并行策略，而不是只盯 recompute

这部分的核心意思是：激活重计算是局部最优工具，不是全局最优答案。它擅长解决“激活太大”，但不负责解决所有显存问题。

---

## 参考资料

1. Megatron Core `TransformerConfig` 文档：说明 `recompute_granularity` 支持 `selective` 与 `full`，默认 selective 模块为 `core_attn`，且 selective 作用于所有层。  
   https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html

2. NVIDIA NeMo Megatron Bridge 文档《Activation Recomputation》：说明 selective 针对 self-attention 的 memory-intensive 模块，full 会重算整个 Transformer 层，计算代价约 30%。  
   https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html

3. Hugging Face Accelerate 文档《Megatron-LM》：给出 GPT-3 量级示例，selective 激活重计算可带来约 70% 激活内存节省，额外约 2.7% FLOPs。  
   https://huggingface.co/docs/accelerate/main/en/usage_guides/megatron_lm

4. 论文《Reducing Activation Recomputation in Large Transformer Models》：提出 selective activation recomputation 与 sequence parallelism，是该策略的原始论据来源。  
   https://arxiv.org/abs/2205.05198

5. 博客园技术解读：整理了常见近似公式，包括 \(5as/h > 34\) 与 selective 的内存节省判断，适合作为阅读论文前的辅助材料。  
   https://www.cnblogs.com/fariver/p/18901293

6. Megatron-LM GitHub Issue #1886：展示一个真实工程坑，开启 `core_attn` 重计算后 Q/K/V 生命周期延长，峰值显存未必下降。  
   https://github.com/nvidia/megatron-lm/issues/1886

7. FlashAttention 论文与实现文档：帮助理解为什么在 fused attention 路径下，激活保存模式会与传统 attention 不同，因此 selective 的收益不能直接照搬普通实现。  
   https://arxiv.org/abs/2205.14135

8. NVIDIA Transformer Engine 文档：用于理解 fused kernel、mixed precision 与 activation 生命周期之间的实现差异，适合排查实际峰值显存为什么与理论不同。  
   https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html

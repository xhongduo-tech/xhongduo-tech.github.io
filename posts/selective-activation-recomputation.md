## 核心结论

Selective Activation Recomputation，中文可译为“选择性激活重计算”，指训练时不统一保存或统一丢弃所有中间激活，而是按“显存节省”和“重算代价”的比例决定哪些激活进入 checkpoint 策略。

它的本质是“按收益选 checkpoint”，不是把 Transformer 整段均匀切开后统一重算。普通 activation checkpointing 更像“整段录像都不存，回放时全部重拍”；Selective Activation Recomputation 更像“只删最占空间、而且最容易补回来的片段”，因此能省显存，同时减少额外计算。

技术上，决策原则不是“attention 固定重算、MLP 固定保存”，而是比较每个中间激活的：

$$
score_i = \frac{m_i}{c_i}
$$

其中 $m_i$ 是丢弃该激活可节省的显存字节数，$c_i$ 是反向传播时重新计算它的代价。优先选择 $m_i / c_i$ 更高的激活重算，优先保存“重算贵但省不多”的激活。

| 对比项 | 普通 checkpointing | selective recomputation |
|---|---|---|
| 基本策略 | 按层或按 block 均匀切分 | 按激活收益选择 |
| 保存对象 | 只保存 checkpoint 边界 | 保存重算不划算的激活 |
| 重算范围 | 通常较大 | 只重算高收益部分 |
| 决策依据 | 结构边界 | $m_i / c_i$ |
| 常见结果 | 显存下降明显，但额外 FLOPs 高 | 显存下降接近，同时速度更好 |

在 Transformer 中，attention 相关激活通常更占内存，尤其在序列长度 $S$ 较大时会出现明显的 $S^2$ 项；但 attention 的部分中间量重算相对便宜。MLP 的激活也占显存，但大矩阵乘法重算成本更高，所以更常被保存。这也是 selective 策略在相同显存预算下常比均匀 checkpointing 快 20-30% 的原因。

---

## 问题定义与边界

训练大模型时，前向传播会产生大量中间结果。激活，英文 activation，是神经网络某一层或某个算子输出的中间张量；反向传播需要它们来计算梯度。保存越多激活，反向越快，但显存压力越大；丢弃越多激活，显存越省，但反向时需要重跑部分前向计算。

Selective Activation Recomputation 解决的是“激活显存过高”的问题。它不是优化模型精度的方法，也不是替代所有 memory optimization 的总方案。它通常与 tensor parallel、FlashAttention、ZeRO、torch.compile 等技术共存。

新手需要先分清一点：你不是在“让模型算得更少”，而是在“决定哪些中间结果值得存，哪些值得算回来”。

| 术语 | 白话解释 | 在本文中的作用 |
|---|---|---|
| 激活 activation | 前向过程中产生、反向还要用的中间张量 | 显存压力来源 |
| checkpoint | 被保留下来的关键中间状态 | 反向重算的起点 |
| recomputation | 反向时重新执行一段前向计算 | 用算力换显存 |
| selective recomputation | 只对部分高收益激活做重算 | 降低无效重算 |

选择问题可以写成：

```text
max Σ m_i
subject to Σ c_i ≤ C
```

意思是在额外计算预算 $C$ 内，最大化节省的激活显存。

| 变量 | 含义 |
|---|---|
| `B` | 微批大小，即一次参与前后向的小批样本数 |
| `S` | 序列长度 |
| `H` | 隐藏维度 |
| `A` | attention 头数 |
| `m_i` | 第 `i` 个激活可节省的字节数 |
| `c_i` | 第 `i` 个激活的重算代价 |

玩具例子：两个中间张量都能省 100MB，一个重算只要 1 单位算力，另一个要 10 单位算力。显然先丢弃前者更合理，因为它的 $m_i / c_i$ 更高。

真实工程例子：长上下文 Transformer 训练中，attention 的中间量在 `S` 很大时快速膨胀。若直接全存，显存会先爆掉；selective 策略的目标就是把最不划算保存的激活挪到重算路径上。

---

## 核心机制与推导

核心机制是把每个激活看成候选项。每个候选项有两个属性：节省多少显存，重算需要多少代价。排序标准不是单独看显存，也不是单独看 FLOPs，而是看单位重算代价能换来多少显存下降。

```text
priority_i = m_i / c_i
```

如果两个激活都能省 100MB，但一个重算只要 1 单位算力，另一个要 10 单位算力，那么前者的优先级是 100，后者是 10。先重算前者，训练速度损失更小。

Transformer 中，attention 和 MLP 的激活形态不同。attention，白话说就是让每个 token 计算它应该关注哪些其他 token；它会产生和序列长度强相关的中间量。MLP，白话说就是每个 token 内部的非线性变换模块，主要由大矩阵乘法构成。

论文中的单层激活近似可写为：

```text
M_attn ≈ B S (11H + 5AS) bytes
M_mlp  ≈ 19BSH bytes
M_layer ≈ BSH (34 + 5AS/H) bytes
```

| 项 | 主要来源 | 量级特征 | 倾向 |
|---|---|---|---|
| `M_attn` | attention 中间量 | 含 `S^2` 项 | 更适合重算 |
| `M_mlp` | MLP / GEMM 中间量 | 主要是 `O(BSH)` | 更适合保存 |
| `M_layer` | 单层总激活 | 组合项 | 按收益拆分 |

其中 `5AS` 这一项乘上 `BS` 后会形成和 $BAS^2$ 相关的量。序列越长，attention 激活越容易成为显存峰值来源。MLP 的显存项主要随 $B S H$ 增长，但其重算通常包含昂贵的大 GEMM，GEMM 是 general matrix multiplication，即通用矩阵乘法，通常是 GPU 训练中的主要计算开销之一。

一个最小数值例子：

```text
B = 1, S = 1024, H = 4096, A = 32

M_attn ≈ 1 * 1024 * (11 * 4096 + 5 * 32 * 1024)
       = 213,909,504 bytes ≈ 204 MiB

M_mlp ≈ 19 * 1 * 1024 * 4096
      = 79,691,776 bytes ≈ 76 MiB
```

如果粗略设定 `C_attn = 2`、`C_mlp = 6`，则：

| 模块 | 可省显存 | 重算代价 | 性价比 |
|---|---:|---:|---:|
| attention | 204 MiB | 2 | 102 MiB / 单位 |
| MLP | 76 MiB | 6 | 12.7 MiB / 单位 |

结论是先重算 attention 更划算，MLP 更适合保存。

张量并行，英文 tensor parallel，是把同一层的矩阵或 attention 头切到多张 GPU 上并行计算。叠加 tensor parallel 后，论文给出的结论可简化理解为：

```text
激活内存可降到约 34BSH/t
```

其中 `t` 是 tensor parallel size。关键点是 selective recomputation 能把 attention 中高风险的 $S^2$ 激活项压掉，再由 tensor parallel 分摊剩余激活。

---

## 代码实现

在训练框架里，selective recomputation 通常不是用户手写每一层的保存逻辑，而是通过配置项打开，例如只对特定子模块启用重算。工程接口常见形式是：指定 recompute 粒度、指定模块范围、指定是否只重算 attention core。

伪代码如下：

```python
for module in transformer_block:
    if should_recompute(module):
        save_minimal_state(module)
    else:
        save_activation(module)

backward():
    for module in reversed(transformer_block):
        if not activation_saved(module):
            recompute_forward(module)
        run_backward(module)
```

更具体的模块级选择策略可以写成：

```python
if module_name in {"core_attn", "attention"}:
    recompute = True
else:
    recompute = False
```

下面是一个可运行的 Python 玩具实现。它不模拟真实 autograd，只演示如何按 $m_i / c_i$ 在计算预算内选出更值得重算的模块。

```python
from dataclasses import dataclass

@dataclass
class Activation:
    name: str
    memory_mib: float
    recompute_cost: float

def select_for_recompute(items, cost_budget):
    ranked = sorted(items, key=lambda x: x.memory_mib / x.recompute_cost, reverse=True)
    chosen = []
    total_cost = 0.0
    saved_memory = 0.0

    for item in ranked:
        if total_cost + item.recompute_cost <= cost_budget:
            chosen.append(item.name)
            total_cost += item.recompute_cost
            saved_memory += item.memory_mib

    return chosen, saved_memory, total_cost

items = [
    Activation("core_attn", 204, 2),
    Activation("mlp", 76, 6),
    Activation("layernorm", 8, 1),
]

chosen, saved, cost = select_for_recompute(items, cost_budget=3)

assert chosen == ["core_attn", "layernorm"]
assert saved == 212
assert cost == 3
```

| 框架 | 典型入口 | 含义 |
|---|---|---|
| Megatron-LM | `recompute_granularity="selective"` | 选择性重算 |
| NeMo | activation recomputation 配置 | 训练时控制重算粒度 |

真实工程例子：在 Megatron-LM / NeMo 训练大模型时，常见做法是只对 `core_attn` 等 attention 内部模块启用选择性重算，而不是对整个 Transformer block 一刀切。这样可以保留 MLP 中重算昂贵的矩阵乘激活，同时丢弃 attention 中占空间大的中间量。

---

## 工程权衡与常见坑

第一层权衡是“显存 vs 算力”。Selective recomputation 的目标不是把所有东西都重算，而是在显存下降明显的前提下，把额外 FLOPs 控制在可接受范围内。FLOPs 是 floating point operations，指浮点运算次数，常用于估算计算开销。

第二层权衡是“局部最优 vs 系统整体”。attention、MLP、FlashAttention、tensor parallel、`torch.compile` 可能彼此影响。某个模块单看最优，不代表组合后仍然最优。

| 常见坑 | 错误做法 | 更稳妥做法 |
|---|---|---|
| 只看张量大小 | 按 `m_i` 排序 | 按 `m_i / c_i` 排序 |
| block 级一刀切 | 整个 Transformer block checkpoint | 先 selective，再决定是否 full |
| 忽略长上下文 | 只用短序列 profile 结果 | 单独对长序列 profile |
| 工具叠加冲突 | 默认认为 FlashAttention / compile 不影响策略 | 前后向剖析，确认实际保存/重算路径 |

新手版判断：如果只看“哪个张量大”，可能会把重算代价很高的 MLP 也删掉。结果显存省了，但训练速度掉得更厉害。

工程版判断：长上下文时 attention 的重算成本会上升，原本“便宜”的判断可能失效。短序列上得到的经验不能直接搬到长序列训练。

一个简单风险判断公式是：

```text
若 c_i 上升快于 m_i，原先的 selective 选择就可能失效
```

实际落地时可以按这个清单检查：

| 检查项 | 需要确认的问题 |
|---|---|
| 显存峰值 | 峰值是否真的来自激活，而不是 optimizer state 或参数 |
| 重算路径 | backward 中是否确实触发了目标模块重算 |
| kernel 影响 | FlashAttention 是否已经减少了 attention 中间量 |
| 并行影响 | tensor parallel 是否改变了各模块显存占比 |
| 速度影响 | tokens/s 或 step time 是否优于 full checkpointing |

---

## 替代方案与适用边界

Selective recomputation 适合“大模型训练 + 显存紧张 + 模块代价差异明显”的场景。如果模型很小、序列很短，或者工程栈已经把内存压得很低，它的收益可能不明显。

不是所有场景都值得做复杂选择。如果你只有几个层、序列也短，直接 checkpoint 整段可能就够了。复杂策略会引入 profiling 成本，也会增加调试难度。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| full checkpointing | 简单 | 重算范围大 | 结构简单、实现优先 |
| selective recomputation | 性价比高 | 需要 profile 和策略选择 | 大模型、长序列 |
| 仅依赖 kernel 优化 | 实现透明 | 不一定解决激活峰值 | 计算瓶颈更明显时 |

边界判断列表：

- 长上下文是否显著放大 `S^2` 项
- 模块重算代价是否差异明显
- 是否已经叠加 tensor parallel / FlashAttention
- 是否愿意为更复杂策略增加 profiling 成本
- 当前瓶颈是显存、计算，还是通信

技术版例子：当 attention 计算已经被 FlashAttention 这类高效 kernel 大幅优化，而 MLP 重算仍然昂贵时，selective 的收益会下降，需要重新评估模块划分。此时“attention 更适合重算”仍是候选假设，不是无需验证的规则。

最终原则是：先用 profile 找出激活峰值来源，再用 $m_i / c_i$ 排序选择重算对象，最后用真实训练吞吐验证结果。只有同时降低显存并保持较高 tokens/s，selective recomputation 才算真正生效。

---

## 参考资料

1. [Reducing Activation Recomputation in Large Transformer Models (MLSys 2023)](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)
2. [PyTorch Blog: Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
3. [NVIDIA NeMo / Megatron Bridge: Activation Recomputation](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/activation-recomputation.html)
4. [Megatron-LM TransformerConfig 源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py)

| 顺序 | 目的 |
|---|---|
| 1 | 先理解 activation checkpointing 为什么能省显存 |
| 2 | 再理解 selective recomputation 为什么要按收益选择 |
| 3 | 最后看 Megatron-LM / NeMo 中如何配置和落地 |

论文结果中提到，在 530B GPT-3 风格训练上，选择性策略将 MFU 从 42.1% 提到 54.2%。MFU 是 model FLOPs utilization，即模型理论计算量被硬件有效执行的比例，可作为“这不是纯理论优化”的工程证据。

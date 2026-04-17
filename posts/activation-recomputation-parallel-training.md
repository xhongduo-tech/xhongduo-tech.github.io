## 核心结论

激活重计算（activation checkpointing，也常叫 activation recomputation，白话讲就是“前向中间结果先不全存，反向需要时再算一遍”）的本质，是用额外计算换显存。对大语言模型训练，瓶颈通常不是参数本身，而是前向传播留下来的激活值；一旦序列长度 $s$、微批大小 $b$、隐藏维度 $h$、层数 $L$ 同时变大，激活内存会近似按 $O(L\cdot s\cdot b\cdot h)$ 线性增长。

如果采用“全层重计算”，最常见的近似写法是：

$$
M_{\text{full}} \approx 2sbhL
$$

这里忽略了常数项、数据类型和框架细节，只抓主导量级。核心含义很直接：每层都保存输入或关键中间结果时，层数越多，显存越线性上升。

如果只在每 $t$ 层保留一个 checkpoint，其余层反向时重跑前向，那么显存从“跟层数线性相关”变成“主要跟单层规模相关”。对新手最重要的理解不是公式常数，而是趋势：激活占用可以从“几十层一起堆着”降成“只像一层或一个分段那样大”。

在并行训练里，更实用的不是把整层都重算，而是 selective checkpoint，也就是“只重算最吃显存、但重算相对便宜的模块”。Megatron Bridge / NeMo 默认就偏向这种做法，典型配置是只对 attention 主干做 selective recomputation，例如 `core_attn`。原因是 attention 里的 softmax、dropout、QK 相关中间量对显存很敏感，尤其随序列长度平方增长；但它们的重算代价往往低于整层 MLP 和大投影。

一个直观数字例子：假设每层仅保留输入，近似按 `2 × s × b × h` 个元素估算，取 $s=2048,b=1,h=4096,L=48$，若 48 层都保留，激活约为 3.2 GiB；若只保留单层级别的 checkpoint，则约 64 MiB，量级差接近 50 倍。工程上不一定真的只剩 64 MiB，因为还会有临时张量、并行通信 buffer、优化器状态，但“从数 GiB 到数十或数百 MiB”这个方向是成立的。

| 策略 | 保存内容 | 激活显存趋势 | 额外计算 | 典型场景 |
|---|---|---:|---:|---|
| 不重计算 | 全部激活 | $O(Lsbh)$ | 1.0x | 显存充裕、追求最短步时 |
| 全层重计算 | 每段输入 checkpoint | 近似降到分段级 | 约 1.30x 到 1.33x | 显存紧张、实现简单 |
| selective checkpoint | 只重算 `core_attn/QKV` 等 | 明显下降，常优于全层性价比 | 小于全层重算 | LLM 并行训练主流做法 |

---

## 问题定义与边界

先把问题说清楚。激活值（activation，白话讲就是“前向过程里每层临时算出来、反向还要用的中间结果”）为什么会成为瓶颈？因为参数只存一份，但激活跟 batch、序列长度、层数一起长。训练 GPT 一类模型时，参数显存、优化器显存、梯度显存都重要，但当序列长、层深时，激活常常先把卡打满。

目标不是“少存一点内存”这么模糊，而是两个更严格的目标：

1. 减少前向后需要长期驻留的激活显存。
2. 保持反向梯度与原始计算图一致，不能为了省内存改掉训练结果。

checkpoint（检查点，白话讲就是“故意保留下来的存档点”）就是为这个目标服务的。前向时只留少量检查点，中间很多张量立刻释放；反向时如果需要中间值，就从最近的 checkpoint 重新跑一次前向。

可以用一个玩具例子理解。假设你有 8 层网络，不做重计算时，相当于把 1 到 8 层所有房间都一直留着；做分段 checkpoint 时，只保留第 0、4、8 层这几个“公共走廊”，中间房间先拆掉，等反向经过时再按图纸临时重建。显存只为“走廊”长期买单，不为所有房间同时买单。

这件事的适用边界也要明确：

| 场景 | 适用性 | 原因 |
|---|---|---|
| 训练超大模型、长序列 | 很适合 | 激活显存大，省显存价值高 |
| GPU 算力较强、显存较紧 | 很适合 | 可以接受多算一点 |
| 推理阶段 | 通常不适合 | 推理没有反向，重算收益很小 |
| 显存本就充裕 | 收益有限 | 额外计算未必值得 |
| 强依赖低延迟单步训练 | 需谨慎 | 会增加训练步时 |
| 无法控制激活生命周期的老代码 | 易踩坑 | 可能“逻辑上重算了，内存上没省下来” |

这里还要区分一个常见混淆：很多资料把 activation checkpointing 和 gradient checkpointing 混着叫。工程语境里它们通常指的是同一类思想，即“保留少量前向状态、反向时重算”。真正要关注的不是名字，而是你到底 checkpoint 了什么粒度：整层、attention 子模块，还是更细的算子。

---

## 核心机制与推导

先看最基本的全层重计算。对一个 $L$ 层 Transformer，如果每层都要为反向保存输入激活，最粗略估算是：

$$
M_{\text{full}} \approx 2sbhL
$$

其中 $2$ 可以理解为“保存输入和必要副本/残差的粗略常数”，不同实现并不完全相同，但这个公式足够解释为什么层数一多就爆显存。

如果改成每 $t$ 层保存一次 checkpoint，那么任意时刻只需长期保留每个分段的边界输入，而不用把所有层的中间结果都压在显存里。于是主导项会从“按 $L$ 累加”转成“按分段宽度或单层上界”来算。很多工程文章会把这种趋势写成近似的 $O(sbh)$，重点不是说常数消失了，而是说“层数不再一层不落地乘进去”。

对于 selective checkpoint，分析会更细。论文和工程文档强调，attention 里的某些中间量对显存最敏感，尤其自注意力分数和 softmax/dropout 相关张量，其规模会带有 $s^2$ 项；但重算这些部分的 FLOPs 增量不一定最大。因此只对 self-attention 主干做重算，常常比整层重算更划算。

一些资料给出一个更具体的简化估计：

$$
M_{\text{selective}} \approx \frac{34sbhL}{t}
$$

这个式子里的常数 34 来自特定 Transformer 结构的激活分解，不是所有实现都通用。它有用的地方在于告诉你：一旦引入分段因子 $t$，激活内存不再按“所有层全量保存”增长，而会被分摊到每个 checkpoint 段上。写文章或做容量预估时，可以把它看成“结构化估算模板”，不能当成所有模型的精确真值。

把用户给定数值代进去，看量级变化更直观。设：

$$
s=2048,\quad b=1,\quad h=4096,\quad L=48
$$

则全层保存近似为：

$$
2 \times 2048 \times 1 \times 4096 \times 48
= 805{,}306{,}368
$$

若使用 FP32，每个元素 4 字节，约为：

$$
805{,}306{,}368 \times 4 \approx 3.0\ \text{GiB}
$$

若考虑实现中的额外张量和保守估计，写成约 3.2 GiB 是合理的工程口径。若只保留单层输入：

$$
2 \times 2048 \times 1 \times 4096 \times 4 \approx 64\ \text{MiB}
$$

这就是“数 GiB 对比数十 MiB”的来源。

把时间线画成一个简化过程，会更容易理解：

| 阶段 | 不重计算 | 分段 checkpoint |
|---|---|---|
| Forward 1-4 层 | 每层中间结果都保留 | 只保留第 0、4 层边界 |
| Forward 5-8 层 | 继续累积激活 | 只保留第 8 层边界 |
| Backward 8-5 层 | 直接用已存激活 | 从第 4 层 checkpoint 重跑 5-8 层前向 |
| Backward 4-1 层 | 直接用已存激活 | 从第 0 层 checkpoint 重跑 1-4 层前向 |

真实工程例子是 3D 并行训练。pipeline parallel（流水线并行，白话讲就是“把模型层切到多张卡、像流水线一样分段执行”）会把不同层放在不同 stage 上。此时如果每个 stage 内再做 `uniform` 或 `block` 方式的 checkpoint，就能让每张卡只承担本 stage 的少量激活驻留；再叠加 selective checkpoint 到 `core_attn`，通常能明显提升可训练序列长度或 micro-batch size。

---

## 代码实现

下面先给一个能运行的 Python 玩具程序，用来估算不同策略下的激活量级。它不是框架真实显存统计，但足够帮助新手建立数量级直觉。

```python
def gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)

def mib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)

def estimate_full_activation_bytes(s: int, b: int, h: int, L: int, bytes_per_elem: int = 4) -> int:
    # 粗略模型：每层保存约 2 * s * b * h 个元素
    return 2 * s * b * h * L * bytes_per_elem

def estimate_single_checkpoint_bytes(s: int, b: int, h: int, bytes_per_elem: int = 4) -> int:
    # 只保留单层级别 checkpoint
    return 2 * s * b * h * bytes_per_elem

def estimate_uniform_checkpoint_bytes(s: int, b: int, h: int, L: int, t: int, bytes_per_elem: int = 4) -> int:
    assert L > 0 and t > 0 and L % t == 0
    # 每 t 层一个 checkpoint，近似保存 L/t 个边界
    return 2 * s * b * h * (L // t) * bytes_per_elem

s, b, h, L = 2048, 1, 4096, 48

full_bytes = estimate_full_activation_bytes(s, b, h, L)
single_bytes = estimate_single_checkpoint_bytes(s, b, h)
uniform_bytes = estimate_uniform_checkpoint_bytes(s, b, h, L, t=8)

assert full_bytes > uniform_bytes > single_bytes
assert round(mib(single_bytes)) == 64
assert 2.9 < gib(full_bytes) < 3.1  # 纯公式值接近 3.0 GiB

print("full  :", round(gib(full_bytes), 2), "GiB")
print("uniform checkpoint (t=8):", round(mib(uniform_bytes), 2), "MiB")
print("single checkpoint:", round(mib(single_bytes), 2), "MiB")
```

在 PyTorch 里，最直接的写法是用 `torch.utils.checkpoint.checkpoint` 包住需要重算的模块。伪代码如下：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        qkv = self.qkv(x)
        # 这里省略 reshape / attention / softmax 细节
        out = qkv[..., : x.size(-1)]
        return self.proj(out)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, use_recompute=True):
        super().__init__()
        self.attn = AttentionBlock(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.use_recompute = use_recompute

    def forward(self, x):
        # 选择性重算 attention，而不是整层都重算
        if self.use_recompute and self.training:
            x = x + checkpoint(self.attn, x, use_reentrant=False)
        else:
            x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
```

要点有两个。

第一，checkpoint 不是“自动省显存魔法”。如果你在 checkpoint 外面仍然保留了 Q/K/V、中间 attention scores、某些 closure 捕获的张量，那么生命周期没有真正缩短，显存就不会明显下降。

第二，真正的工程代码往往不是手写 wrapper，而是框架配置。例如在 Megatron Bridge / NeMo 中，常见配置思路是：

```python
from megatron.bridge.models import GPTModelProvider

model_config = GPTModelProvider(
    recompute_granularity="selective",
    recompute_modules=["core_attn"],
)
```

如果要做整层重计算，则会看到类似 `full + uniform/block` 的配置思路。`uniform` 的意思是“按固定层数均匀分段”，`block` 的意思是“每个 pipeline stage 只对前若干层做重算”。后者适合做更细的工程折中，因为不是所有层都必须被纳入重算。

再给一个真实工程例子。假设一个 70B 级 GPT 模型使用 tensor parallel + pipeline parallel 训练，单卡显存卡在 80GB 边缘，长序列下 attention 中间量爆得最快。此时如果不改模型结构，只把 `recompute_granularity` 调到 `selective`、模块限定为 `core_attn`，很多时候就足以把 micro-batch 从 1 提到 2，或把序列长度从 4K 提到 8K，而精度保持不变，因为数学上前向函数没改，只是中间值改成“稍后再算”。

---

## 工程权衡与常见坑

激活重计算不是“越多越好”，而是一个明确的三方权衡：显存、额外计算、实现复杂度。

| 策略 | 显存节省 | 时间开销 | 实现复杂度 | 备注 |
|---|---:|---:|---:|---|
| 不重算 | 最低 | 最低 | 最低 | 最简单，但最吃显存 |
| 全层重算 | 很高 | 常见约 30%–33% | 中 | 适合先保命跑通 |
| selective `core_attn` | 高 | 通常显著低于全层 | 中高 | LLM 训练更常见 |
| selective + sequence parallel | 更高 | 额外开销可进一步压低 | 高 | 需要并行策略配合 |

sequence parallel（序列并行，白话讲就是“把序列维的一部分激活切分到多卡上分担”）和 selective recomputation 是经典搭配。MLSys 2023 论文给出的核心结论是：二者配合 tensor parallel 后，可以把重计算导致的执行时间开销大幅压低，很多情况下比传统全层重算快得多。工程讨论中常见一个经验数字：如果原来全层重算要付出接近 33% 的额外计算，那么配合 sequence parallel 与 selective checkpoint 后，额外时间可能下降到约 10% 量级，常见口径在 11% 左右，但这依赖模型大小、内核实现和并行拓扑，不能当成固定承诺。

最容易踩的坑有三类。

第一类是“以为 checkpoint 了，其实没释放”。典型反例是只把 `core_attn` 包进 checkpoint，但在外层代码里提前算好了 Q/K/V 并把它们继续引用着。这样反向确实会重算 attention 主干，但 Q/K/V 仍然常驻，显存不但没明显下降，还多了重算开销。

第二类是“全层重算过度使用”。如果你只是差 2GB 显存就能跑通，却把全部层都打开 full recomputation，那么换来的可能是明显更慢的 step time。此时更合理的是 block recomputation 或 selective checkpoint，只对几个最重的层段下手。

第三类是“忽略并行策略兼容性”。例如 Megatron Core 文档明确说明，`distribute_saved_activations=True` 这类分布式保存激活的选项，与 `sequence_parallel=True` 不是任意组合都兼容。并行训练里很多“显存优化开关”并不能随便叠加，需要看版本文档。

一个实用 checklist：

- 先用 profiler 或框架日志确认瓶颈是不是激活，而不是优化器状态或 KV cache。
- 先试 selective `core_attn`，再考虑 full recomputation。
- 检查 checkpoint 边界外是否仍引用中间张量。
- 区分训练与推理路径，不要把重算逻辑带进推理。
- 观察 step time、吞吐、显存峰值三项，不要只看显存。
- 在并行训练中核对 sequence parallel、pipeline、activation distribute 的兼容矩阵。

---

## 替代方案与适用边界

如果显存问题来自激活，激活重计算通常是第一选择；但它不是唯一选择。工程上还常把它和 pipeline parallel、ZeRO、offload 一起比较。

| 方案 | 主要解决什么 | 显存收益 | 计算代价 | 通信代价 | 更适合的瓶颈 |
|---|---|---:|---:|---:|---|
| Activation recomputation | 激活显存 | 高 | 中 | 低到中 | 长序列、深层模型 |
| Pipeline parallel | 单卡装不下全部层 | 中到高 | 低到中 | 中 | 模型层数很深 |
| ZeRO/FSDP | 参数、梯度、优化器状态 | 很高 | 低到中 | 高 | 参数状态过大 |
| CPU/NVMe offload | 本地显存不足 | 高 | 中到高 | 高 | 显存极紧但可忍受慢 |
| 无重算 | 无 | 无 | 最低 | 最低 | 显存很充裕 |

对初级工程师最重要的判断顺序可以写成一个简化决策树：

1. 如果爆显存主要出现在长序列训练，并且 profiler 显示 attention 激活占大头，优先用 selective activation recomputation。
2. 如果单卡连模型层都放不下，先上 pipeline parallel 或 tensor parallel，再考虑重算。
3. 如果参数、梯度、优化器状态占主要显存，优先考虑 ZeRO/FSDP，而不是只盯着激活。
4. 如果算力紧张、训练吞吐比显存更重要，不要先上 full recomputation。
5. 如果已经用了 sequence parallel 和 FlashAttention，再评估 selective checkpoint 是否仍有额外收益。

再看一个更贴近业务的对比。假设你要把上下文从 4K 拉到 8K：

| 方案组合 | batch 扩展能力 | 序列扩展能力 | 单步速度 | 工程复杂度 |
|---|---|---|---|---|
| 无重算 | 弱 | 弱 | 快 | 低 |
| activation recomputation + pipeline | 中到强 | 强 | 中 | 中 |
| ZeRO + offload | 中 | 中 | 慢 | 高 |
| selective recomputation + sequence parallel | 强 | 很强 | 中到较快 | 高 |

因此，激活重计算最合适的边界不是“凡是训练都开”，而是“当激活是主瓶颈，且你愿意拿少量额外计算换取更长序列、更大 batch 或更低卡数时，就应该优先考虑”。

---

## 参考资料

- [NVIDIA NeMo / Megatron Bridge Activation Recomputation 文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/activation_recomputation.html)：官方文档，说明 `full`、`selective`、`core_attn`、`uniform/block` 等配置含义。
- [Megatron Core TransformerConfig 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)：查看 `recompute_granularity`、`recompute_method`、`recompute_modules`、`distribute_saved_activations` 等参数定义。
- [Reducing Activation Recomputation in Large Transformer Models, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html)：核心论文，提出 selective activation recomputation 与 sequence parallel 的组合思路。
- [OpenReview: Reducing Activation Recomputation in Large Transformer Models](https://openreview.net/forum?id=wKv8jIyuqw)：论文页面，便于查看版本、作者和公开讨论。
- [AWS SageMaker Model Parallel Activation Checkpointing 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html)：从框架使用者视角解释 activation checkpointing 的基本机制。
- [Megatron-LM 相关讨论 issue](https://github.com/nvidia/megatron-lm/issues/1886)：工程问题讨论，适合了解 selective checkpoint、生命周期管理和实际 overhead 的坑点。

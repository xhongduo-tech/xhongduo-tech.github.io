## 核心结论

Activation Checkpointing，中文常译为“激活检查点”，白话讲就是：训练时不把每一层的中间结果都一直留在显存里，只保留少量关键边界，等反向传播真的需要时再把中间过程重算一遍。

它解决的是训练显存里的“激活内存”瓶颈。激活值就是前向传播过程中每层产生、并且反向传播求导时还要再次用到的中间张量。模型越深、序列越长、batch 越大，激活值越占显存。很多大模型训练先撞上的不是参数内存，而是激活内存。

这项技术的本质是“以计算换内存”。默认训练更省算力，因为前向算过的东西都存着；Checkpointing 更省显存，因为只存少量边界，代价是反向时要把丢掉的中间激活重演一次。工程上常见的结果是：额外增加约 20% 到 33% 计算开销，换来 2 倍到 10 倍量级的激活内存节省，从而支持更大的 batch size、更长上下文或更大的模型。

一个足够准确的心智模型是：把网络分段。假设一共有 4 段前向，不做 checkpoint 时，4 段的中间激活都保存到反向结束；做 checkpoint 后，只保存第 1、3 段这样的边界输出，第 2、4 段内部激活不保存。等反向传播走到第 4 段时，先根据最近一个已保存边界把第 4 段前向重跑出来，再立刻求梯度并释放。于是显存峰值下降，但反向阶段会多跑若干次前向。

理论上，若总深度为 $N$，每隔 $k$ 层设一个 checkpoint，则保存激活的量可写成近似的
$$
M_{\text{save}} = O(N/k + k)
$$
当 $k$ 取得合适时，空间复杂度可从朴素保存的 $O(N)$ 压到接近 $O(\sqrt{N})$ 级别。这个式子不是让你手算生产环境参数，而是帮助理解：checkpoint 太密，算力浪费；checkpoint 太稀，内存仍然太高。

---

## 问题定义与边界

先把问题说清楚。训练一个神经网络时，前向传播不仅要得到输出，还要为反向传播保留求导所需的上下文。自动求导框架默认会保存很多中间张量，因为反向传播要按链式法则，也就是“后面的导数要依赖前面的中间结果”来计算梯度。

因此，训练时显存大致由三部分组成：

| 部分 | 白话解释 | 典型变化趋势 |
|---|---|---|
| 参数 | 模型权重本身 | 随模型规模增长 |
| 优化器状态 | Adam 等优化器额外保存的统计量 | 通常比参数还大 |
| 激活 | 前向中间结果，供反向求导使用 | 随层数、batch、序列长度快速增长 |

Activation Checkpointing 主要作用在第三项。它不减少参数量，也不改变优化器状态；它针对的是“前向中间结果保存太多”的问题。

要注意边界。它只适合训练，不是推理阶段的主优化手段。推理通常没有完整反向传播，因此激活不会长期保留到 backward，瓶颈常常不在这里。另一个边界是：它不改变数学结果。只要实现正确，checkpoint 训练和普通训练的梯度在数值上应当等价，区别只是多了一次重算。

玩具例子可以先看 8 层网络。默认做法是 8 层的中间激活全保存。若每 2 层设一个 checkpoint，那么只保留第 2、4、6、8 层边界，层内细节不保存。这样显存显著下降，但反向传播每进入一段，都要把这一段重新前向一次。

真实工程例子更直观。21medien 给出的 Llama 3 8B 训练示例里，序列长度 2048 时，不开 checkpoint 激活内存约 40GB，batch size 只能到 4；开启后激活内存降到约 12GB，batch size 可到 16，训练时间从 100 分钟增加到 130 分钟。换句话说，单样本更慢了，但因为一次能塞进更多样本，系统整体吞吐未必更差。

把这件事抽象成资源边界，通常是下面这个对比：

| 策略 | 显存占用 | 额外计算 | 调试难度 | 适合场景 |
|---|---|---|---|---|
| 全部存激活 | 高 | 最低 | 最低 | 显存充裕、优先速度 |
| Activation Checkpointing | 中到低 | 中等 | 中等 | 显存紧张、训练大模型 |
| 更细粒度重算 | 更灵活 | 可调 | 更高 | 需要精细控制 speed/memory |

---

## 核心机制与推导

核心机制只有两步。

第一步，分段保存。把网络切成若干段，只保存段的输入或输出边界，不保存段内部全部激活。

第二步，反向重算。当反向传播走到某一段时，框架根据最近保存的边界，把这段前向再执行一次，临时恢复所需中间激活，随后立刻完成该段反向并释放内存。

为什么它能省内存？因为默认训练的峰值通常出现在 backward 刚开始前，那时前向所有层的激活都还没释放。Checkpointing 把大量“长期占着显存”的中间张量换成“需要时再临时生成”，所以峰值会下降。

用一个 4 段玩具例子说明：

1. 前向经过 A、B、C、D 四段。
2. 默认做法：A/B/C/D 的内部激活都保存。
3. Checkpoint 做法：只保存 A 的输出、C 的输出，以及整网输入。
4. 反向从 D 开始时，先利用 C 的输出把 D 重跑，拿到 D 内部激活，再算 D 的梯度。
5. 接着处理 C 段。若 C 内部没保存，就从 A 的输出重跑 C，再算梯度。
6. 这样一路回退。

所以节省的不是“完全不算”，而是“不提前保存”。

为什么常说它的复杂度近似是 $O(N/k+k)$？直觉如下：如果总共 $N$ 层、每段长度约为 $k$，那么需要长期保存的边界数量约是 $N/k$；与此同时，单次重算时最多保留一段内部临时激活，规模约是 $k$。两者叠加就是
$$
M_{\text{save}} \approx O(N/k + k)
$$
令两项量级接近，即 $N/k \approx k$，就得到 $k \approx \sqrt{N}$，此时空间近似最优，接近 $O(\sqrt{N})$。

但工程上不会真的按公式机械求最优。因为不同层的激活大小不一样，注意力层和 MLP 层的代价也不一样。真正做法是先看 profile，也就是性能与内存分析结果，再只对峰值显存区域下手。

真实工程里，NVIDIA NeMo 文档把它和 FSDP、LC-CE 组合使用。FSDP 是全分片数据并行，白话讲就是把参数和状态拆到多卡分担；LC-CE 是一种更省输出层内存的 loss 计算方法。NeMo 给出的例子中，Llama-3.2-1B 在 H100-80GB 上，基线最大显存约 53.03GB；加 FSDP 后约 47.59GB；再加 gradient checkpointing 后降到约 33.06GB。说明 checkpointing 针对的是 Transformer block 内部激活热点，和其他节省内存的方法往往能叠加。

---

## 代码实现

先给一个可运行的 Python 玩具实现，用“数字加法层”模拟分段重算。它不依赖深度学习框架，但能准确展示“保存边界、反向时重演”的思路。

```python
from dataclasses import dataclass

@dataclass
class Segment:
    delta: int

    def forward(self, x: int) -> int:
        return x + self.delta

def forward_full(segments, x):
    activations = [x]
    for seg in segments:
        x = seg.forward(x)
        activations.append(x)
    return x, activations

def forward_checkpointed(segments, x, checkpoint_every=2):
    checkpoints = {0: x}
    for i, seg in enumerate(segments, start=1):
        x = seg.forward(x)
        if i % checkpoint_every == 0 or i == len(segments):
            checkpoints[i] = x
    return x, checkpoints

def replay_segment(segments, checkpoints, start, end):
    x = checkpoints[start]
    for i in range(start, end):
        x = segments[i].forward(x)
    return x

segments = [Segment(1), Segment(2), Segment(3), Segment(4)]

y_full, activations = forward_full(segments, 0)
y_ckpt, checkpoints = forward_checkpointed(segments, 0, checkpoint_every=2)

assert y_full == 10
assert y_ckpt == 10
assert activations == [0, 1, 3, 6, 10]
assert checkpoints == {0: 0, 2: 3, 4: 10}

# 反向时如果需要第 3~4 层之间的中间值，可从最近 checkpoint=2 重演
replayed = replay_segment(segments, checkpoints, start=2, end=3)
assert replayed == 6
print("toy example ok")
```

上面这个例子里，`activations` 是“全部保存”，`checkpoints` 是“只保存边界”。`replay_segment` 就是 checkpointing 的核心动作。

真正工程里，PyTorch 原生接口通常这样用：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

class MyModel(nn.Module):
    def __init__(self, dim=256, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([MyLayer(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

`use_reentrant=False` 是 PyTorch 现在更推荐的路径。它的直观含义是：使用较新的非重入实现，兼容性和行为更清晰。若你在 Hugging Face Transformers 中训练模型，很多时候甚至不必手工包每层，直接调用：

```python
model.gradient_checkpointing_enable()
```

就能在库已接好的模块上启用。

如果需要更细粒度控制，PyTorch 2025 年博客还介绍了 Selective Activation Checkpointing。它允许你声明哪些算子必须保存、哪些更适合重算。最典型的策略是：矩阵乘法这类昂贵算子尽量保存，逐元素操作这类便宜算子更适合重算。

---

## 工程权衡与常见坑

第一，Checkpointing 不是“免费内存”。FairScale 文档明确给出的经验值是大约 33% 计算开销，NeMo 在 Transformer 场景里给出的实践值约 5% 到 10% wall-clock 开销。两者不矛盾，因为模型结构、分段粒度、并行方式不同，最终看到的是不同口径的代价。保守做法是把它当成“有明显算力加价”的技术，而不是默认开启。

第二，不要全模型无脑包。应优先包住显存热点区域，通常是 Transformer block、长序列注意力、超深堆叠层。若某段本来激活不大，却计算很重，checkpoint 只会白白拖慢训练。

第三，随机性要处理一致。FairScale 文档特别提到 RNG state，也就是随机数生成器状态。白话讲，如果一层里有 Dropout，前向和重算前向必须看到一致的随机状态，否则梯度会错。

第四，某些层有副作用时要谨慎。比如 BatchNorm 在训练态会更新统计量，而 checkpoint 会把前向执行两次。如果直接重跑，统计量可能被重复更新。FairScale 的建议是这类场景要冻结统计或避免这样包。

第五，profiling 会变难。因为你在 trace 里看到的前向不再只跑一次，真实耗时会分散到 backward 前后的重算过程。看 profile 时要区分“原始前向”和“反向触发的重算前向”。

可以把权衡总结成一张表：

| 维度 | 优点 | 缺点 | 常见处理 |
|---|---|---|---|
| 内存 | 激活内存显著下降 | 不能减少参数和优化器状态 | 与 FSDP、ZeRO、LC-CE 组合 |
| 速度 | 可换来更大 batch，整体吞吐未必变差 | 单 step 变慢 | 只包显存峰值区域 |
| 正确性 | 数学上与原训练等价 | Dropout、BatchNorm、状态层要小心 | 保证 RNG 和模块行为一致 |
| 调试 | 改动少时接入简单 | profile 与报错栈更难读 | 先小规模对比 loss 与梯度 |

真实工程例子是：在分布式 LLM 微调里，只对 Transformer blocks 开 `activation_checkpointing: true`，而 embedding、输出头、轻量辅助模块保持普通 forward。这样更容易把额外开销控制在可接受区间，同时避免全局重算导致吞吐塌陷。

---

## 替代方案与适用边界

Activation Checkpointing 不是唯一方案，它只是最通用、最容易理解的一类。

第一类替代方案是 `torch.compile` 的 min-cut rematerialization。rematerialization 白话讲就是“需要时重新物化中间值”。它会把前向和反向联合看成一张图，再自动决定哪些值值得保存、哪些值值得重算。PyTorch 官方博客指出，它默认更偏向节省运行时间，所以通常只重算一些便宜、可融合的 pointwise ops，而不是粗粒度重算整个大块区域。

第二类是 Selective Activation Checkpointing。它还是 checkpoint，但不再是“整段全重算”，而是“段内分算子选择”。这更接近工程最优解，因为 matmul、attention 这类昂贵算子不一定值得重跑。

第三类是 Memory Budget API。它让你给一个预算值，在 compile 路径下自动找 speed/memory 的 Pareto 折中点。Pareto 的白话解释是：在一组方案里，任何一个维度想再变好，都必须牺牲另一个维度。

三种策略的适用边界可以这样看：

| 方案 | 内存节省 | 速度影响 | 控制粒度 | 适合场景 |
|---|---|---|---|---|
| 默认 eager | 最弱 | 最快基线 | 无 | 显存够用，优先简单稳定 |
| 粗粒 Activation Checkpointing | 强 | 中等下降 | 段级 | 显存紧张，想快速落地 |
| Selective AC | 中到强 | 更可控 | 算子级 | 需要进一步压榨性能 |
| `torch.compile` + min-cut | 中等 | 往往更优 | 自动 | 想少改模型代码 |
| Memory Budget API | 可调 | 可调 | 预算级 | 需要系统化调参 |

如果你是零基础到初级工程师，一个可靠决策顺序是：

1. 先确认瓶颈是不是激活内存，而不是参数或优化器状态。
2. 若是激活内存，先启用粗粒度 checkpoint，验证 loss 与吞吐。
3. 若速度损失过大，再考虑 selective checkpoint 或 compile 路线。
4. 若仍不够，再叠加分片并行、低精度、输出层内存优化。

---

## 参考资料

- PyTorch Blog, *Current and New Activation Checkpointing Techniques in PyTorch*（2025-03-05）  
  https://pytorch.org/blog/activation-checkpointing-techniques/

- 21medien, *Gradient Checkpointing*  
  https://www.21medien.de/en/library/gradient-checkpointing

- FairScale Docs, *Enhanced Activation Checkpointing*  
  https://fairscale.readthedocs.io/en/latest/deep_dive/activation_checkpointing.html

- NVIDIA NeMo AutoModel, *Gradient (Activation) Checkpointing*  
  https://docs.nvidia.com/nemo/automodel/latest/guides/gradient-checkpointing.html

- Amazon SageMaker Docs, *PyTorch activation checkpointing*  
  https://docs.aws.amazon.com/en_us/sagemaker/latest/dg/model-parallel-extended-features-pytorch-activation-checkpointing.html

- Chen et al., *Training Deep Nets with Sublinear Memory Cost*  
  https://arxiv.org/abs/1604.06174

## 核心结论

Gradient Checkpointing，也叫 Activation Checkpointing，定义是：前向传播时只保存少量激活值，反向传播时重新计算被丢弃的中间激活。

一句话写成：

`Gradient Checkpointing = 保存少量激活 + 反向重算中间前向`

这里的“激活值”指每一层前向计算产生的中间结果，例如 Transformer block 的输出、注意力层中间张量、MLP 层输出。训练时反向传播需要这些中间结果来计算梯度，所以普通训练会把它们保留下来。Gradient Checkpointing 改变的是“哪些激活值必须在前向阶段保存”。

普通训练像是把每一步算出来的草稿都留着。Checkpointing 像是只留每一段的起点和终点，回头需要中间过程时再复算一遍。这个说法只是帮助理解，本质仍然是自动微分系统中的存储策略变化。

| 方案 | 前向保存内容 | 显存 | 计算量 |
|---|---|---:|---:|
| 普通训练 | 全部中间激活 | 高 | 低 |
| Checkpointing | 少量边界激活 | 低 | 高 |

核心结论有三点：

1. 它解决的是训练阶段的激活显存压力，不是压缩模型参数。
2. 它用额外前向重算换取更低显存，训练吞吐通常会下降。
3. 它最适合深层模型、长序列 Transformer、大 batch 微调等激活显存占主导的场景。

典型工程收益是：显存下降明显，训练速度变慢。实际开销和模型结构、切分粒度、硬件、框架实现有关，常见经验范围是约 20%-30% 额外计算开销，但不能把它当成固定常数。

---

## 问题定义与边界

训练一个神经网络时，显存不只用于保存模型参数。一次完整训练 step 至少涉及参数、梯度、优化器状态和激活值。

| 显存来源 | 作用 | 是否受 checkpointing 影响 |
|---|---|---|
| 参数 | 存模型权重 | 否 |
| 梯度 | 反向更新需要 | 否 |
| 优化器状态 | Adam 等状态 | 否 |
| 激活值 | 反向传播需要 | 是，主要优化对象 |

“参数”是模型本身的权重，例如线性层里的矩阵。  
“梯度”是损失函数对参数的导数，用来更新参数。  
“优化器状态”是优化器额外维护的变量，例如 Adam 的一阶矩和二阶矩。  
“激活值”是前向传播产生的中间张量，反向传播要用它们计算链式法则。

Gradient Checkpointing 只主要影响激活值。它不会让参数变少，也不会让 Adam 的优化器状态消失。如果一个模型的显存主要被参数和优化器状态占满，单独开启 checkpointing 的效果会有限。

一个真实工程例子是长上下文 Transformer 微调。假设参数通过混合精度已经可以放进 GPU，优化器状态也能通过 LoRA 或 ZeRO 控制住，但序列长度从 2K 提到 8K 后，注意力和 MLP 中间激活迅速变大，训练开始 OOM。此时开启 gradient checkpointing，框架不再保存所有 block 的内部激活，显存下降，可能让更长序列或更大 batch 放得下。代价是每个 step 变慢。

边界要明确：

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 训练深层模型 | 适合 | 层数多，激活累积明显 |
| 长序列 Transformer | 适合 | 激活随序列长度增长很快 |
| 大 batch 微调 | 适合 | batch 越大，激活越多 |
| 参数本身放不下 | 不主要解决 | checkpointing 不减少参数 |
| Adam 状态太大 | 不主要解决 | 优化器状态仍然存在 |
| 推理显存问题 | 通常不适合 | 推理一般不需要保存反向激活 |

所以它不是一个通用省显存按钮，而是训练阶段针对激活存储的策略。

---

## 核心机制与推导

设一个模型有 $n$ 层，第 $i$ 层输出记为 $a_i$。普通训练中，前向传播会保存大量中间激活：

$$
a_0, a_1, a_2, ..., a_n
$$

反向传播从 $a_n$ 开始向前计算梯度，并使用每一层对应的输入或输出。为了避免反向时缺少数据，普通自动微分会在前向阶段保存这些值。

Gradient Checkpointing 的做法是分段保存。假设每 $k$ 层保存一个 checkpoint，只保存：

$$
a_0, a_k, a_{2k}, a_{3k}, ...
$$

前向流程可以写成：

```text
[a0] -> ... -> [ak] -> ... -> [a2k] -> ... -> [a3k]
```

方括号表示被保存的边界激活，中间的激活在前向结束后可以丢弃。

反向传播到某一段时，例如要处理从第 $k+1$ 层到第 $2k$ 层，系统会从最近保存的 $a_k$ 重新执行这一段前向，临时补出：

$$
a_{k+1}, a_{k+2}, ..., a_{2k}
$$

然后再用这些临时激活计算梯度。计算完这一段后，中间激活又可以释放。

玩具例子：8 层网络，每层激活占 4 MB。

普通训练保存 8 个内部激活：

$$
8 \times 4\text{ MB} = 32\text{ MB}
$$

如果每 4 层做一次 checkpoint，只保存第 0、4、8 层边界，近似保存 3 个激活：

$$
3 \times 4\text{ MB} = 12\text{ MB}
$$

反向时，第 1-4 层会从 $a_0$ 重算一次，第 5-8 层会从 $a_4$ 重算一次。显存减少，计算增加。

更一般地，均匀分段时可以用一个启发式公式理解激活显存：

$$
M_{act} \approx O(n/k + k)
$$

其中 $n/k$ 表示保存的 checkpoint 数量，$k$ 表示反向处理单段时临时需要保留的段内激活。取：

$$
k \approx \sqrt{n}
$$

可以得到：

$$
M_{act} = O(\sqrt{n})
$$

这个推导不是说所有框架都会精确达到这个复杂度，而是说明一个核心趋势：如果分段合理，激活显存可以从随层数线性增长，降低到次线性增长。

下面是一个可运行的 Python 玩具例子，用数字模拟“全保存”和“每 4 层 checkpoint”的激活保存量：

```python
def activation_memory_full(num_layers: int, mb_per_activation: int) -> int:
    return num_layers * mb_per_activation

def activation_memory_checkpoint(num_layers: int, segment_size: int, mb_per_activation: int) -> int:
    # 保存 a0、每段边界，以及最后 an；这里假设 num_layers 能被 segment_size 整除。
    num_boundaries = num_layers // segment_size + 1
    return num_boundaries * mb_per_activation

full = activation_memory_full(num_layers=8, mb_per_activation=4)
checkpointed = activation_memory_checkpoint(num_layers=8, segment_size=4, mb_per_activation=4)

assert full == 32
assert checkpointed == 12
assert checkpointed < full
print(full, checkpointed)
```

这个例子没有实现自动微分，只展示存储量变化。真实框架里的 checkpointing 还要处理计算图、随机数状态、反向梯度和设备放置。

---

## 代码实现

在 PyTorch 中，核心接口是 `torch.utils.checkpoint.checkpoint`。它不是改变模型结构，而是告诉框架：这段前向的内部激活不要全部保存，反向传播时允许重新运行这段前向。

最小示例：

```python
import torch
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)

def run_block(block, x):
    return block(x)

x = torch.randn(2, 128, requires_grad=True)
block = Block(torch.nn.Linear(128, 128))

y = checkpoint(run_block, block, x, use_reentrant=False)
loss = y.sum()
loss.backward()

assert x.grad is not None
assert block.layer.weight.grad is not None
```

这里的 `use_reentrant=False` 是 PyTorch 当前推荐显式传入的参数。简单说，reentrant 是一种旧式重入式 checkpoint 实现路径；非 reentrant 版本在很多场景下行为更清晰，也支持更多 autograd 用法。实际工程中应按当前 PyTorch 文档要求显式选择，避免版本升级后出现警告或行为差异。

不要把 checkpoint 理解成“把整个模型随便一包”。更常见的做法是按较大的语义块切分，例如 Transformer block：

```text
forward:
  save boundary activations
  discard internal activations

backward:
  reload nearest boundary activation
  rerun forward segment
  compute gradients
```

真实工程例子：一个 Transformer 有 32 个 block。可以对每个 block 或每几个 block 使用 checkpoint。这样反向传播到某个 block 时，重新执行该 block 的前向，恢复注意力、MLP、归一化等中间激活，再计算梯度。通常不建议只给一个很小的 `Linear` 或 `LayerNorm` 单独 checkpoint，因为省下的激活有限，调度和重算开销仍然存在。

Hugging Face Transformers 中通常不需要手动包每个 block，可以调用：

```python
model.gradient_checkpointing_enable()
```

这个开关会使用模型内部已经适配好的 checkpoint 逻辑。对于训练脚本，也常见通过 `TrainingArguments` 或模型配置打开。使用时还要检查模型是否支持该功能，以及是否与缓存机制冲突。例如自回归模型训练时，常需要关闭 `use_cache`，因为缓存是推理加速用的，和训练时的重算策略目标不同。

JAX 中对应概念通常叫 `jax.checkpoint` 或 `jax.remat`。`remat` 是 rematerialization 的缩写，意思是“重新物化”：不保存某些中间值，需要时重新计算出来。不同框架接口不同，但底层思想一致：控制自动微分系统在前向阶段保存哪些值。

---

## 工程权衡与常见坑

主要权衡是显存下降和速度下降。Checkpoint 切分越激进，保存的激活越少，但反向重算越多。切得太粗，省显存不明显；切得太碎，速度损失和调度开销会增加。

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| `dropout` 等随机算子 | 重算不一致 | 保持 RNG 状态，使用框架默认机制 |
| `forward` 和重算路径不一致 | 梯度错误 | 避免依赖全局状态 |
| `in-place` 修改 | 破坏重算输入 | 尽量使用非原地操作 |
| 在 `run_fn` 中搬到新设备 | 触发异常或额外开销 | 保持计算路径稳定 |
| 模块切得太小 | 收益低于开销 | 按较大语义块切分 |
| 整个函数都 checkpoint | 省不到多少内存 | 避免把最后段或整个大函数一把包住 |

随机性是最容易被忽略的问题。`dropout` 是一种训练时随机丢弃部分神经元输出的正则化方法。如果第一次前向和反向重算时使用不同的 dropout mask，重算出来的激活就不等于原来的激活，梯度会偏离预期。PyTorch 的 checkpoint 默认会处理随机数状态保存和恢复，但如果手写复杂逻辑、跨设备移动张量，仍然可能破坏一致性。

全局状态也会造成问题。例如 `forward` 里根据某个全局计数器选择不同分支，第一次前向走 A 分支，重算时走 B 分支，梯度就不再对应原来的计算。Checkpoint 要求重算路径与原前向路径语义一致。

`in-place` 修改指直接修改已有张量内容，例如某些带下划线的 PyTorch 操作。它可能破坏 autograd 需要的值，也可能让重算输入不再可靠。不是所有原地操作都一定错误，但在 checkpoint 范围内应更保守。

实践建议是：优先 checkpoint 大块、重复计算相对便宜、激活多的模块，比如 Transformer block，而不是单个小算子。开启后要同时观察三类指标：峰值显存、每 step 时间、最终 loss 是否正常下降。

---

## 替代方案与适用边界

Gradient Checkpointing 不是唯一省显存方案。训练大模型时，它经常和混合精度、梯度累积、FlashAttention、LoRA、ZeRO、FSDP 一起使用。

| 方案 | 主要省什么 | 代价 | 适用场景 |
|---|---|---|---|
| Gradient Checkpointing | 激活显存 | 变慢 | 深层模型、长序列训练 |
| 混合精度 | 参数/激活/梯度部分 | 数值精度管理 | 通用训练 |
| 梯度累积 | 峰值 batch 显存 | 更慢 | batch 太大时 |
| FlashAttention | 注意力激活 | 实现依赖 | 长上下文 Transformer |
| LoRA / PEFT | 可训练参数与状态 | 表达受限 | 微调 |
| ZeRO / FSDP | 参数、梯度、状态 | 通信复杂 | 大模型分布式训练 |

“混合精度”是用 FP16、BF16 等低精度格式参与训练，减少显存和提升吞吐，但要处理数值稳定性。  
“梯度累积”是把一个大 batch 拆成多个小 micro-batch，多次反向后再更新参数，降低单次峰值显存。  
“FlashAttention”是更高效的注意力实现，减少注意力计算中的中间存储。  
“LoRA/PEFT”是参数高效微调方法，只训练少量新增参数。  
“ZeRO/FSDP”是分布式切分参数、梯度和优化器状态的方法。

新手判断可以直接看瓶颈来源：

| 瓶颈 | 优先考虑 |
|---|---|
| 激活太多，尤其是长序列或深层模型 | Gradient Checkpointing、FlashAttention |
| 参数和优化器状态太大 | LoRA、ZeRO、FSDP、8-bit optimizer |
| batch 太大 | 梯度累积、减小 micro-batch |
| 数值格式占用太高 | BF16/FP16 混合精度 |
| 训练速度已经很紧张 | 谨慎使用 checkpointing |

如果模型很浅、序列很短、batch 很小，激活占比不高，checkpointing 的收益可能不明显。此时开启后只会让训练变慢，却省不出关键显存。

如果训练任务对吞吐极其敏感，也要评估是否值得使用。例如同样一张 GPU，开启 checkpointing 后能把 batch 从 8 提到 12，但每 step 变慢 30%。最终吞吐是否提升，要看“样本数/秒”，不能只看 batch size。

一个实际决策顺序是：先用显存分析工具确认峰值来自哪里；如果激活占主导，再开启 checkpointing；如果参数或优化器状态占主导，先处理参数、状态和分布式切分问题。

---

## 参考资料

1. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
2. [PyTorch 文档：torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)
3. [JAX 文档：Gradient checkpointing with jax.checkpoint](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
4. [Hugging Face Transformers：Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)

| 资料 | 主要用途 |
|---|---|
| Chen et al. 论文 | 理解核心思想和次线性显存推导 |
| PyTorch 文档 | 查看 `checkpoint` 接口、参数和注意事项 |
| JAX 文档 | 理解 `remat/checkpoint` 的边界与实践建议 |
| Hugging Face 文档 | 了解训练任务中的实际开关与集成方式 |

建议阅读顺序：先看论文理解为什么能用重算换显存，再看 PyTorch 或 JAX 文档理解接口边界，最后看 Hugging Face 文档确认具体训练脚本中如何打开。

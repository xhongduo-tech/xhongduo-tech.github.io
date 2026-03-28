## 核心结论

梯度检查点（Gradient Checkpointing）是训练时的一种显存优化方法。白话说，它不把每一层的中间结果都长期留在显存里，而是只保留少数“里程碑”层，等反向传播真正需要时，再从最近的里程碑重跑那一小段前向计算。

训练显存大致由三部分构成：激活值、梯度、优化器状态。对大模型训练来说，激活值就是前向过程中每层输出的中间结果，反向传播要靠它求梯度；当层数、batch size、序列长度一起增大时，激活值会快速吃掉显存。梯度检查点主要压缩的就是这部分。

核心交换关系很直接：

$$
\text{更少显存} \Longleftrightarrow \text{更多重算}
$$

在常见分析里，若一个有 $n$ 层的顺序网络按大约每 $\sqrt{n}$ 层放一个检查点，那么激活显存可以从 $O(n)$ 降到约 $O(\sqrt{n})$，而总训练计算通常增加到约 $1.33\times$。更激进的策略还能继续省显存，但训练会更慢，极端情况下接近 $2\times$ 计算量。

一个新手最容易理解的玩具例子是 8 层模型。假设只保留第 0、3、6 层激活，其余层输出不长期保存。反向时如果需要第 5 层激活，就从第 3 层开始重跑第 4、5 层。这样显存峰值可以从“近似保存 8 层激活”下降到“近似保存 3 个里程碑激活”，而训练时间只增加大约 33%。可以把它理解成只在关键楼层放缓冲区，其余楼层临时重新走一遍。

| 方案 | 激活显存 | 额外计算 | 典型结论 |
|---|---:|---:|---|
| 普通训练 | $O(n)$ | 1.0x | 最快，但最吃显存 |
| 梯度检查点 | $O(\sqrt{n})$ | 约 1.33x | 工程上最常见的折中 |
| 几乎全重算 | 接近 $O(1)$ | 最多接近 2.0x | 只有极端显存压力才值得 |

长序列训练尤其依赖它。因为序列长度增长时，Transformer 块里的激活张量会同步变大，7B 以上模型、8K 甚至更长上下文时，单卡常常先被激活值撑爆，而不是先被参数本身撑爆。

---

## 问题定义与边界

问题很明确：训练大模型时，前向传播如果把每层激活都保存下来，显存占用会随层数近似线性增长；当模型足够深、序列足够长、batch 足够大时，就会直接 OOM，也就是“显存不够，程序被迫中断”。

这里要先划清边界。梯度检查点不是压缩参数本身，也不是减少优化器状态。它主要作用于“激活值这部分显存”。所以它最适合下面这类场景：

1. 模型已经能加载进显存，但一训练就 OOM。
2. 参数量不是唯一瓶颈，长序列和大 batch 让激活值成为主因。
3. 你愿意接受 10% 到 30% 甚至更高的训练变慢，换取更大的 batch 或更长上下文。

真实工程例子是 24GB 显存的 RTX 4090。训练 7B+ 模型或 8K 以上长上下文时，常见情况是模型能成功加载，但一进入训练，激活值立刻把显存打满。此时把 `gradient_checkpointing=True` 打开，经常能直接把“不能训练”变成“能训练”，并让 batch size 或上下文长度继续上调。这类收益通常比单纯去抠几个小张量有价值。

但也要看到边界：

- 如果模型很小、序列很短、显存本来就够，梯度检查点只会白白拖慢训练。
- 如果真正的瓶颈是优化器状态，例如全参数 AdamW 微调超大模型，那么它只能解决一部分问题，不能替代 ZeRO、FSDP、LoRA 之类方案。
- 如果你更在乎单位时间吞吐，而不是“能不能塞进显存”，那它未必是首选。

| 维度 | 普通保存激活 | 梯度检查点 | 全重算思路 |
|---|---|---|---|
| 显存复杂度 | $O(n)$ | 常见为 $O(\sqrt{n})$ | 可逼近 $O(1)$ |
| 训练速度 | 最快 | 中等变慢 | 最慢 |
| 工程复杂度 | 最低 | 中等 | 较高 |
| 适用场景 | 显存充足 | 显存紧张但还能接受变慢 | 极端显存受限 |

所以它本质上不是“让训练更高效”，而是“让原本放不下的训练任务放得下”。

---

## 核心机制与推导

机制可以分成两步理解。

第一步，为什么普通训练要存激活？  
因为反向传播求梯度时，需要前向经过的中间结果。比如第 11 层的梯度，通常要知道第 10 层输出、第 11 层输入等信息。如果前向结束后这些都丢了，反向就没法直接算。

第二步，检查点怎么减少保存量？  
它不保存所有中间激活，只保存少数检查点。反向传播需要某层激活时，就从最近的检查点重新执行那段前向。这就是“重算”。

以 16 层网络为例，假设在第 0、4、8、12、16 层设置检查点。现在反向传播处理到第 11 层，需要第 9、10、11 层的激活，但它们没有长期保存，于是系统从第 8 层检查点开始，重跑第 9、10、11 层前向，拿到需要的激活后继续求梯度。

可以画成一个简化图：

```text
前向层级:  0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 - 10 - 11 - 12 - 13 - 14 - 15 - 16
保存位置:  C               C               C                C                 C
反向重算:                                  8 -> 9 -> 10 -> 11
```

为什么会出现 $O(\sqrt{n})$？  
设总层数为 $n$，把网络切成 $k$ 段，每段长度约为 $n/k$。

- 你需要保存大约 $k$ 个检查点激活。
- 反向时每一段最多重算约 $n/k$ 层。

想让“保存的数量”和“重算段长度”都不要太大，常见做法是让两者数量级相同：

$$
k \approx \frac{n}{k}
$$

于是得到：

$$
k \approx \sqrt{n}
$$

所以显存复杂度变成：

$$
\text{Memory} \approx O(\sqrt{n})
$$

这不是说实际显存永远精确等于 $\sqrt{n}$，而是说随着层数增长，它的增长速度从线性降成了次线性。

时间为什么常说约 $1.33\times$？  
直观上，普通训练是“1 次前向 + 1 次反向”。用了检查点后，反向阶段会插入一些局部前向重算。若采用接近理论最优的均匀分段策略，额外重算开销常近似记成原前向成本的约三分之一，于是：

$$
T_{\text{total}} = T_{\text{forward}} + T_{\text{backward}} + T_{\text{recompute}}
$$

若把普通训练归一化成 1，则常见简化写法是：

$$
T_{\text{total}} \approx 1 + \frac{1}{3} \approx 1.33
$$

这只是工程近似，不是所有模型都严格等于 1.33。实际值会受分段粒度、是否只检查点部分层、注意力实现方式、通信开销等因素影响。更密集地设置检查点，比如几乎每层都做，显存会继续下降，但重算比例会上升，整体可能接近 $2\times$ 计算。

---

## 代码实现

在 PyTorch 里，最常用入口是 `torch.utils.checkpoint.checkpoint`。它会把一段前向函数包装成“可重算段”。白话说，前向时先别把中间过程全存下来，反向时缺什么再补算。

下面先用一个可运行的玩具 Python 例子模拟“分段保存”的思想。它不是完整 autograd 实现，但能帮助理解为什么检查点能把保存的激活数量压下来。

```python
import math

def checkpoint_schedule(num_layers: int):
    segment = max(1, int(math.sqrt(num_layers)))
    checkpoints = list(range(0, num_layers + 1, segment))
    if checkpoints[-1] != num_layers:
        checkpoints.append(num_layers)
    return checkpoints

def peak_saved_activations(num_layers: int, checkpoints):
    # 简化模型：只统计被长期保存的“里程碑”激活数量
    return len(checkpoints)

c8 = checkpoint_schedule(8)
c16 = checkpoint_schedule(16)

assert c8[0] == 0 and c8[-1] == 8
assert c16 == [0, 4, 8, 12, 16]
assert peak_saved_activations(8, c8) <= 4
assert peak_saved_activations(16, c16) == 5

print("8层检查点:", c8)
print("16层检查点:", c16)
```

真正训练时，一般这样写：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)

class ToyModel(nn.Module):
    def __init__(self, hidden_size=128, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(hidden_size) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = checkpoint(
                block,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return x

x = torch.randn(2, 128, requires_grad=True)
model = ToyModel()
y = model(x).sum()
y.backward()

assert x.grad is not None
```

新手可以先把它理解成下面这个最小模式：

```python
def checkpointed_block(x):
    def fn(x):
        return block(x)
    return checkpoint(fn, x, use_reentrant=False)

out = checkpointed_block(hidden)
```

如果你在 Hugging Face Transformers 里训练，通常不需要手动包每一层，直接打开模型级开关即可：

```python
from transformers import AutoModelForCausalLM, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("your-model")
model.gradient_checkpointing_enable()

args = TrainingArguments(
    output_dir="./out",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

两个常见参数要知道：

- `preserve_rng_state`：是否保留随机数状态。白话说，就是让 dropout 这类随机操作在重算时尽量和原前向一致。默认 `True` 更稳，但有额外开销。
- `use_reentrant`：选择旧版还是新版检查点实现。当前 PyTorch 文档建议显式传入，通常推荐 `False`，因为限制更少、行为更稳定。

---

## 工程权衡与常见坑

第一个权衡是速度。梯度检查点不是白拿显存，代价就是重算。经验上，训练变慢 5% 到 30% 都常见，具体取决于你检查点的密度。如果你已经有大量算力冗余，这个代价通常能接受；如果 GPU 计算本来就吃满，额外重算会直接拉低吞吐。

第二个权衡是可复现性。很多模型里有 dropout，它会依赖随机数状态。默认保存 RNG 状态能让重算阶段与原前向更一致，但保存和恢复本身也要成本。如果你的任务不要求严格复现实验，或者已经在别处控制随机性，可以评估是否关闭它。

第三个问题是“前向必须可重放”。白话说，检查点区段内的前向逻辑，重跑一遍必须还是同一个数学函数。如果它依赖全局变量、外部可变状态、临时 `torch.rand()` 结果、训练过程中会变化的分支条件，就可能出现前向和重算不一致，最后得到错误梯度。

错误示意：

```python
# 错误：段内依赖外部可变随机结果
def fn(x):
    noise = torch.rand_like(x)
    return x + noise
```

更稳妥的做法是：

```python
# 更稳妥：把随机性控制在段外，或使用受控 dropout
noise = torch.rand_like(x)
def fn(x, noise):
    return x + noise
```

另一个常见坑是 `detach()`。如果使用 `use_reentrant=True`，检查点段内不能随便把张量从计算图里断开，否则反向可能直接报错。错误和正确可以这样对比：

```python
# 错误：段内 detach
def fn(x):
    x = x.detach()
    return block(x)
```

```python
# 正确：如果确实要 detach，尽量放在 checkpoint 段外
x = x.detach()
out = block(x)
```

再补一个容易忽略的问题：并不是所有层都同样值得做检查点。Transformer 中注意力和 MLP 往往是激活开销大户，选择性地给这些块做检查点，通常比“每层全包”更平衡。工程上常见做法不是追求理论最小显存，而是追求“刚好不 OOM 且吞吐还能接受”。

---

## 替代方案与适用边界

梯度检查点不是唯一的显存优化手段。工程上更常见的是组合使用，而不是单独依赖。

| 方案 | 显存节省 | 计算开销 | 适用场景 |
|---|---|---|---|
| 梯度检查点 | 中到大 | 增加重算 | 激活值主导显存、长序列训练 |
| 混合精度 AMP/BF16 | 中等 | 通常很小 | 大多数训练任务的默认首选 |
| 梯度累积 | 间接降低单步显存 | 吞吐下降 | 显存不够放大 batch |
| FSDP/ZeRO | 很大 | 通信更复杂 | 多卡大模型训练 |
| LoRA/PEFT | 很大 | 训练目标受限 | 微调而非全参数训练 |

如果显存容量还比较宽裕，通常先开 AMP 或 BF16，因为它的收益稳定、额外成本小。若显存还是不够，再加梯度检查点。若是多卡大模型全参数训练，则往往要把检查点和 FSDP/ZeRO 一起用，因为一个解决激活值，一个解决参数与优化器状态，它们处理的热点不同。

什么时候不该用？

- 短序列、小模型、本来就跑得下。
- 延迟和吞吐是绝对第一优先级。
- 已经通过 AMP、FlashAttention、参数高效微调等手段解决了显存问题。

一句话概括适用边界：当“显存是第一瓶颈”时，梯度检查点很有价值；当“算力和吞吐是第一瓶颈”时，它通常不是首选。

---

## 参考资料

- FlashAttention 博客：公式与推导，解释了按 $\sqrt{n}$ 级别设置检查点时，激活显存从 $O(n)$ 降到 $O(\sqrt{n})$ 的直觉与实现。  
  https://flashattn.dev/blog/gradient-checkpointing-explained

- PyTorch `torch.utils.checkpoint` 文档：API 细节最重要，尤其是 `preserve_rng_state`、`use_reentrant`、`checkpoint_sequential` 的行为差异与限制。  
  https://docs.pytorch.org/docs/stable/checkpoint.html

- NVIDIA NeMo AutoModel 指南：真实工程配置示例，展示了激活检查点与 FSDP、内存友好 loss 组合时的显存变化。  
  https://docs.nvidia.com/nemo/automodel/latest/guides/gradient-checkpointing.html

- AI Wiki 梯度检查点入门文章：适合快速理解“只保留里程碑激活、反向重算”的直观解释。  
  https://artificial-intelligence-wiki.com/natural-language-processing/large-language-models/gradient-checkpointing-guide/

- Instagit 的 Hugging Face 训练示例：偏实践视角，强调在单卡显存受限时，`gradient_checkpointing=True` 如何把 OOM 任务变成可训练任务。  
  https://instagit.com/huggingface/transformers/how-does-gradient-checkpointing-reduce-memory-usage-during-training/

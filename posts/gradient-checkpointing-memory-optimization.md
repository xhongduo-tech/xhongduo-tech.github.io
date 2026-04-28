## 核心结论

Gradient Checkpointing，中文常叫“激活检查点”，意思是前向传播时只保存少量中间激活，反向传播需要用到时再把中间那段前向重算一次。它的本质是用额外计算换更低显存。

它主要优化的是**激活显存**。激活就是每一层前向计算后的中间结果，反向传播求梯度时要用到，所以训练框架通常会先把它们缓存起来。模型越深、序列越长、hidden size 越大，激活越容易成为显存峰值来源。

它不是免费优化。显存下降的同时，训练会变慢。工程实践里，常见量级是大约 `20%` 左右的吞吐损失，但具体幅度取决于模型结构、分段方式、序列长度和实现细节。

| 项目 | 普通训练 | Checkpointing |
|---|---|---|
| 前向保存 | 所有中间激活 | 只保存少量检查点 |
| 反向计算 | 直接读取缓存 | 先重算再求梯度 |
| 显存 | 高 | 低 |
| 速度 | 快 | 慢 |

一个新手可用的直觉是：普通训练像“每走一步都存档”，Gradient Checkpointing 像“只在关键路口存档”，回头需要时再把中间路段走一遍。

---

## 问题定义与边界

它解决的问题很具体：**训练时激活显存太大，导致 OOM，或者导致 batch size 和 sequence length 上不去。**

先把边界说清楚。训练显存通常由四类东西构成：

1. 参数
2. 梯度
3. 优化器状态
4. 激活

其中，Gradient Checkpointing 直接针对的是第 4 类。

| 显存来源 | checkpointing 是否直接降低 |
|---|---|
| 参数 | 否 |
| 梯度 | 否 |
| 优化器状态 | 否 |
| 激活 | 是 |

这意味着它并不是“通用省显存按钮”。如果你的瓶颈是参数本体，比如模型太大；或者是优化器状态，比如 Adam 为每个参数维护额外状态，那么只开 checkpointing 往往不够。

一个玩具例子：

假设有 4 层网络，每层激活占 `100 MB`。

- 普通训练：缓存 4 层激活，总共约 `400 MB`
- 每 2 层设一个 checkpoint：只保留边界激活，可能降到约 `200 MB`
- 代价：反向时把段内前向重新算一遍

这个例子说明它最适合“中间结果很多”的网络，而不是“权重本身大到放不下”的网络。

一个真实工程例子：

微调长上下文 Transformer 时，单卡 24GB 或 48GB 显存常常不是先被参数吃满，而是被每层的 attention/MLP 激活顶到峰值。此时开启 `gradient_checkpointing=True`，再配合 `gradient_accumulation_steps`，常常能把原本放不下的序列长度或 batch size 拉回可训练范围。

适用条件通常包括：

- 深层网络
- 长序列
- 大 batch
- hidden size 较大
- 单卡显存紧张
- 微调时希望在不改模型结构的前提下先把训练跑起来

---

## 核心机制与推导

设第 $i$ 层输出激活为 $h_i$。普通训练时，前向会一路计算

$$
h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \dots \rightarrow h_n
$$

并把大部分反向需要的中间结果都留在显存里。这样做的好处是反向快，因为梯度计算可以直接读取缓存；坏处是显存高。

Gradient Checkpointing 的做法是把网络切成若干段，只保留分段边界：

$$
h_0, h_k, h_{2k}, \dots
$$

段内的中间激活不长期保存。等反向传播走到某一段时，再从最近的检查点开始，把这一小段前向重新跑出来，恢复该段所需的中间激活，然后再继续求梯度。

4 层网络的最小直观例子如下。

普通训练：

- 前向保存 $h_1, h_2, h_3, h_4$
- 反向直接读取这些缓存

如果每 2 层放一个 checkpoint：

- 前向主要保留 $h_0, h_2, h_4$
- 反向计算第 4 层、第 3 层梯度前，需要从 $h_2$ 重新跑第 3 到第 4 层
- 计算前两层梯度时，再从 $h_0$ 重跑第 1 到第 2 层

这就是“只保留边界点，段内重算”的核心机制。

理论上，若把 $n$ 层网络做合适分段，可以把激活显存近似写成：

$$
M_{act} \approx O(k + n/k)
$$

这里的 $k$ 可以理解为分段尺度。第一项对应保存的检查点数量，第二项对应重算时需要暂时保留的段内激活量。令两部分平衡，即 $k \approx \sqrt{n}$，可得到常见的简化结论：

$$
M_{act} \approx O(\sqrt{n})
$$

这不是说所有模型都会严格达到 $\sqrt{n}$ 级别，而是说在抽象分层模型里，存在一种“次线性显存”的重算思路。真实工程里，收益取决于：

- 模型是否接近顺序结构
- 某些层是否特别大，比如 attention
- 实际框架保存了哪些张量
- checkpoint 分段是否合理

所以更准确的表述不是“它一定把显存降到多少”，而是“它显著减少激活缓存，代价是更多重算”。

---

## 代码实现

PyTorch 中最常用的接口是 `torch.utils.checkpoint.checkpoint(...)`。它把一个前向函数包起来，使该区域采用激活重算策略。术语“重入模式”对应 `use_reentrant` 参数，官方当前推荐 `use_reentrant=False`，因为它支持更完整的 autograd 行为，限制更少。

下面是一个可运行的最小示例。这个例子不依赖 GPU，重点展示“开启 checkpoint 后输出和梯度仍应正确”这一点。

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x

block = Block(8)

x1 = torch.randn(4, 8, requires_grad=True)
x2 = x1.clone().detach().requires_grad_(True)

# 普通前向
y1 = block(x1)
loss1 = y1.pow(2).mean()
loss1.backward()

# checkpoint 前向
def run_block(x):
    return block(x)

y2 = checkpoint(run_block, x2, use_reentrant=False)
loss2 = y2.pow(2).mean()
loss2.backward()

assert torch.allclose(y1, y2, atol=1e-6)
assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
```

如果你的模型基本是顺序堆叠结构，还可以用 `checkpoint_sequential(...)` 按段包装，而不是手动给每个 block 加壳。

再给一个“显存来源判断”的玩具代码。它不真实测 GPU 显存，只用数字模拟为什么 checkpointing 只影响激活，不影响参数和优化器状态。

```python
def estimate_memory(param_mb, grad_mb, optim_mb, act_mb, use_checkpoint=False, save_ratio=0.4):
    effective_act = act_mb * save_ratio if use_checkpoint else act_mb
    total = param_mb + grad_mb + optim_mb + effective_act
    return total

normal = estimate_memory(param_mb=6000, grad_mb=6000, optim_mb=12000, act_mb=8000, use_checkpoint=False)
ckpt = estimate_memory(param_mb=6000, grad_mb=6000, optim_mb=12000, act_mb=8000, use_checkpoint=True, save_ratio=0.45)

assert normal == 32000
assert ckpt == 27600
assert ckpt < normal
assert (normal - ckpt) == 4400
```

真实工程里，Hugging Face Transformers 通常直接开配置：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
)
```

这里的组合含义是：

- `gradient_checkpointing=True`：降低激活显存峰值
- `gradient_accumulation_steps=4`：把 4 个小 batch 的梯度累起来，形成更大的有效 batch

这个组合常见于大模型微调。目标不是提速，而是在有限显存下把训练稳定跑通。

---

## 工程权衡与常见坑

第一件要记住的事：它是显存优化，不是吞吐优化。反向期间需要重放部分前向，训练通常会变慢。Hugging Face 文档给出的常见经验量级是约 `20%` 变慢，这个数字适合作为预期，但不应当当成硬指标。

第二件要记住的事：它优化的是峰值显存中的“激活部分”。如果你已经被参数和优化器状态占满，收益会有限。

常见坑可以直接整理成表。

| 坑点 | 现象 | 规避 |
|---|---|---|
| 与“模型保存 checkpoint”混淆 | 概念错误，排障方向错 | 明确这里指激活重算 |
| `use_reentrant=True` 用错 | API 限制多，梯度行为更容易踩坑 | 优先 `use_reentrant=False` |
| checkpoint 函数里搬到新设备 | 结果不稳定或梯度异常 | 避免在函数内部切换到新设备 |
| 随机算子未处理 RNG | dropout 等结果不一致 | 默认保留 RNG 状态，必要时再关 |
| 误以为会提速 | 吞吐下降 | 只把它当显存优化 |
| 分段太碎 | 重算开销过大 | 先按 block 级别分段，再测收益 |

这里特别说两个工程细节。

第一，RNG，也就是随机数生成器状态。dropout 这类随机算子在重算时如果拿到不同随机数，前向和重算就不一致。PyTorch 默认会保存和恢复 RNG 状态，保证 checkpoint 区域的随机行为尽量与非 checkpoint 版本一致。但这也会有额外开销。如果你为了极限性能关闭 `preserve_rng_state`，就要接受结果可能不再严格复现。

第二，`gradient_accumulation` 不是吞吐魔法。它只是把多个小步的梯度累起来，得到更大的有效 batch。设单卡 batch 为 $b$，累积步数为 $s$，则有效 batch 近似为：

$$
B_{\text{effective}} = b \times s \times \text{data\_parallel\_world\_size}
$$

它能解决“单步放不下”的问题，但不能让训练速度超过真实一次性大 batch 的吞吐。

一个真实工程例子：

你在 24GB 显存单卡上做 7B 模型 LoRA 微调，序列长度从 2048 提到 4096 后 OOM。此时常见顺序不是先盲目减模型，而是先看瓶颈：

- 若是激活峰值高：开 gradient checkpointing
- 若 batch 太小导致不稳定：再加 gradient accumulation
- 若仍不够：再考虑混合精度、8-bit optimizer、序列裁剪或分布式分片

这个排障顺序的核心是先判断瓶颈属于哪一类显存。

---

## 替代方案与适用边界

Gradient Checkpointing 不是唯一方案，也不应该被当成默认第一选择。更好的做法是先判断：你卡的是激活，还是参数/优化器状态。

| 方案 | 主要减少什么 | 代价 | 适合场景 |
|---|---|---|---|
| Gradient checkpointing | 激活 | 变慢 | 长序列、大模型微调 |
| Mixed precision | 参数/激活/带宽 | 需关注数值稳定性 | 几乎所有训练 |
| Gradient accumulation | 单步峰值显存 | 训练步数变多 | 小显存但想要大有效 batch |
| FSDP / 分片 | 参数与优化器状态 | 通信与工程复杂度高 | 多卡大模型 |
| `torch.compile` 重计算策略 | 部分中间张量 | 依赖编译图与算子支持 | 希望兼顾性能与部分省显存 |

可以这样理解这些方案的分工：

- 如果主要是激活爆了，checkpointing 很对症。
- 如果主要是参数和优化器状态太大，优先看分片、量化、低精度优化器。
- 如果只是想在小显存上维持较大有效 batch，gradient accumulation 更直接。
- 如果模型和算子支持良好，`torch.compile` 有时也会做图级重计算，但它的目标通常不是像 checkpointing 那样激进地节省激活。

因此，适用边界可以压缩成一句话：**当瓶颈是激活显存，而你愿意接受更慢训练时，Gradient Checkpointing 是高性价比方案。**

---

## 参考资料

1. [Training Deep Nets with Sublinear Memory Cost](https://huggingface.co/papers/1604.06174)
2. [PyTorch 官方文档：torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint)
3. [PyTorch 官方博客：Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
4. [Hugging Face 官方文档：Gradient checkpointing](https://huggingface.co/docs/transformers/grad_checkpointing)
5. [Hugging Face 官方文档：Gradient accumulation](https://huggingface.co/docs/transformers/grad_accumulation)

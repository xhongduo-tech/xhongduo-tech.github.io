## 核心结论

梯度累积和梯度检查点解决的是同一个工程问题：显存不够，但方式不同。

梯度累积的本质是把一个大批次拆成多个微批。微批就是一次真正送进 GPU 计算的小批量数据。每个微批都做前向传播和反向传播，但先不更新参数，而是把梯度累加在 `.grad` 里，等累计到指定次数后再统一执行一次参数更新。因此，它降低的是“单次前向/反向所需的瞬时显存”，但不减少参数、优化器状态本身占用的显存。

梯度检查点的本质是少存中间激活值。激活值就是每一层前向传播产生、反向传播还要再用一次的中间结果。常规训练会把所有层的激活都存下来，显存开销随层数上升。检查点方法只保存少数关键层的激活，其余激活在反向传播时重新算一遍，用更多计算换更少显存。

两者经常一起用。前者让“批次变大成为可能”，后者让“深模型能装进显存”。在 Transformer 训练里，这两种技术几乎已经是基础工具，而不是可选优化项。

玩具例子最直观：目标批次大小是 32，但显卡一次只能放下 8。那就把 32 拆成 4 个微批，每次处理 8，做 4 次 `loss.backward()`，最后再 `optimizer.step()` 一次。这样等效批次还是 32，只是更新频率变低。

参数更新可以写成：

$$
w_{t+1}=w_t-\eta\sum_{i=1}^k\nabla\mathcal{L}(b_i)
$$

其中 $k$ 是累积步数，$b_i$ 是第 $i$ 个微批，$\eta$ 是学习率。

---

## 问题定义与边界

训练显存通常被四部分占用：参数、梯度、优化器状态、激活。优化器状态就是 Adam 这类优化器额外保存的动量和方差。对大模型来说，激活和优化器状态经常是两个主要瓶颈。

如果模型很深，或者序列长度很长，激活会迅速膨胀。比如在一张 48GB GPU 上训练一个 8B 量级 Transformer，参数本体不一定是第一瓶颈，真正卡住的往往是长序列下每一层的激活缓存。你会看到一种典型情况：模型能跑通 batch size=1，但一旦把微批调大，立刻 OOM，也就是“显存溢出”。

这里要区分三个概念：

| 概念 | 含义 | 是否直接决定单次显存峰值 |
|---|---|---|
| 微批大小 | 一次真正送进模型的样本数 | 是 |
| 累积步数 | 连续做多少次反向后再更新 | 否 |
| 等效批大小 | 微批大小 × 累积步数 × 数据并行卡数 | 否 |

所以，梯度累积不会让单次前向更大，只会让“若干次小前向合起来”表现得像一次大批次更新。它适合“想增大等效 batch，但显存只容得下小 micro-batch”的情况。

梯度检查点也有边界。它只能减少激活相关显存，不能减少优化器状态，也不能让参数凭空变小。对浅层模型收益有限，对深层 Transformer 收益明显。原因很简单：层越多，可少存的中间结果越多。

真实工程例子可以这样理解：在一张 48GB GPU 上训练 Llama 3 8B，如果序列长度较长，常规训练时激活可能就接近 40GB，几乎没有空间再扩大微批。此时只靠梯度累积不够，因为累积不降低单次激活峰值；开启梯度检查点后，激活占用下降，微批才有机会从 4 提高到 8 或 16。

还要注意边界模块。BatchNorm 依赖一个批次内部的统计量，统计量就是均值和方差。微批太小时，这些统计量会很噪，梯度累积也不能真正恢复 BatchNorm 在“大批次同时前向”时的统计行为。因此，梯度累积并不天然等价于所有模型上的大 batch 训练。Transformer 大多使用 LayerNorm，所以问题较小；卷积网络若大量依赖 BatchNorm，要单独评估。

下面这个表概括三类策略的差别：

| 策略 | 显存峰值 | 等效批 | 计算开销 |
|---|---:|---:|---:|
| 常规训练 | 高 | 受单卡显存限制 | 基线 |
| 梯度累积 | 接近单个微批的水平 | 提升明显 | 略增调度开销 |
| 梯度检查点 | 明显下降 | 可间接增大微批 | 常见增加 20%~33% |
| 两者组合 | 最低的可行峰值 | 最大 | 时间开销最高 |

---

## 核心机制与推导

先看梯度累积。

设微批大小为 $m$，累积步数为 $k$，则单次参数更新对应的等效批大小是：

$$
B_{\text{effective}} = m \times k
$$

如果还有数据并行，卡数为 $n$，则进一步变为：

$$
B_{\text{effective}} = m \times k \times n
$$

为什么它近似等效于大 batch？因为损失函数通常对 batch 做平均或求和。假设我们对每个微批都把 loss 除以 $k$，那么连续执行 $k$ 次反向传播后，`.grad` 中的总梯度就等价于一个更大 batch 的平均梯度。这里“等价”成立的前提是：模型在这 $k$ 次微批之间参数不变，且没有依赖批统计的副作用模块。

玩具例子：微批 16，累积步数 $k=2$。第一次喂入 16 个样本，做一次前向和反向，不更新；第二次再喂入 16 个样本，再做一次反向；此时 `.grad` 里已经累加了两次的梯度，再执行一次 `optimizer.step()`，等效批大小就是 32。

再看梯度检查点。

常规反向传播需要用到前向过程中的所有中间激活，因此若网络有 $L$ 层，激活存储大致随 $L$ 线性增长，记作 $O(L)$。检查点方法不是每层都存，而是每隔一段保存一个“检查点”，其余层的中间结果在反向时从最近的检查点重新前向计算出来。

经典结论是：如果合理选择检查点间隔，激活存储复杂度可以从 $O(L)$ 降到 $O(\sqrt{L})$。直觉上，这是在“存多少层”和“重算多少层”之间找平衡。存得太密，显存省不下来；存得太稀，反向时重算成本太高。将层划成约 $\sqrt{L}$ 个区间时，空间与重算的乘积达到一个更平衡的水平。

$$
\text{Activation Memory}: O(L)\rightarrow O(\sqrt{L})
$$

这不是说实际代码里一定严格达到 $\sqrt{L}$，因为框架实现、层结构、注意力缓存方式都会影响结果，但方向是明确的：少存，多算。

两种技术配合时，逻辑是：

1. 先用检查点降低单次微批的激活峰值。
2. 再用梯度累积把可容纳的微批叠加成更大的等效批。
3. 最后根据吞吐量和训练稳定性调学习率、序列长度和并行策略。

时间和显存的主要权衡如下：

| 技术 | 节省什么 | 代价是什么 | 适合什么模型 |
|---|---|---|---|
| 梯度累积 | 单次更新所需的大批显存 | 更新变慢，吞吐下降 | 显存紧、但模型本身能放下 |
| 梯度检查点 | 激活显存 | 反向时重算前向，训练更慢 | 深层 Transformer、长序列模型 |
| 组合使用 | 激活和批次双重约束 | 最复杂，调参更细 | 大模型微调和资源受限训练 |

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现，用线性回归模拟“梯度累积等效于大 batch”。这里不依赖深度学习框架，重点是把机制讲清楚。

```python
import numpy as np

# y = 2x 的玩具数据
x = np.array([1.0, 2.0, 3.0, 4.0])
y = 2.0 * x

def grad_on_batch(w, xb, yb):
    pred = w * xb
    # MSE: mean((pred - y)^2)
    # dL/dw = mean(2 * (pred - y) * x)
    return np.mean(2.0 * (pred - yb) * xb)

lr = 0.1
w0 = 0.0

# 方案 A: 一次性用 batch size = 4
grad_full = grad_on_batch(w0, x, y)
w_full = w0 - lr * grad_full

# 方案 B: 微批大小 = 2，累积 2 步
micro_batches = [(x[:2], y[:2]), (x[2:], y[2:])]
accum_grad = 0.0
for xb, yb in micro_batches:
    accum_grad += grad_on_batch(w0, xb, yb) / 2.0  # 除以 accum_steps

w_accum = w0 - lr * accum_grad

assert abs(w_full - w_accum) < 1e-9
print("w_full =", w_full)
print("w_accum =", w_accum)
```

这个例子说明：当参数在多个微批之间保持不变，并且 loss 进行了正确缩放时，累积梯度与大 batch 梯度是一致的。

在 PyTorch 里，最常见的新手写法如下：

```python
accum_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader, start=1):
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / accum_steps
    loss.backward()

    if step % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

这段代码里有三个关键点：

1. `loss / accum_steps` 不能漏，否则梯度会放大 `accum_steps` 倍。
2. `optimizer.step()` 只在累积满一轮后执行。
3. `optimizer.zero_grad()` 要放在更新后清空，而不是每个微批都清空。

如果还要配合混合精度训练，可以把 `backward` 和 `step` 放进 `GradScaler` 流程，但机制不变。

梯度检查点在 PyTorch 中通常不是手写重算逻辑，而是借助框架接口。对支持的 Transformer 模型，常见用法如下：

```python
model.gradient_checkpointing_enable()
```

在 Hugging Face Trainer 里，常见配置是：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
)
```

真实工程例子：训练或微调 GPT、BERT、T5 这类 Transformer 时，常见配置是“小微批 + 梯度累积 + 混合精度 + 梯度检查点”。例如单卡 48GB 环境中，若原始设置只能容纳 `per_device_train_batch_size=1`，可以先打开检查点把激活压下去，再把微批提升到 2 或 4，最后用 `gradient_accumulation_steps=8` 拼出更大的等效批。这样吞吐不一定最高，但能让任务从“完全跑不起来”变成“可训练”。

---

## 工程权衡与常见坑

最常见的误解是：“梯度累积等于大 batch，所以所有行为都完全一样。”这不准确。它只是在梯度求和层面近似等效，不保证所有训练细节一致。

第一个坑是 BatchNorm。它依赖当前微批的统计量，而不是等效批的统计量。微批过小会导致统计不稳定，表现为收敛变慢、波动变大，甚至最终精度下降。Transformer 里主流是 LayerNorm，LayerNorm 是对单个样本内部特征做归一化，不依赖批统计，所以更适合梯度累积场景。

第二个坑是忘记缩放 loss。如果你累积 8 步却没有 `loss = loss / 8`，那么最终梯度相当于放大 8 倍，训练可能直接发散，表现为 loss 爆炸或参数变成 NaN。

第三个坑是学习率调度器步数错位。很多调度器默认按 `optimizer.step()` 的次数推进，而不是按微批次数推进。开启梯度累积后，真正的更新步数变少了，如果调度器配置没同步调整，预热步数和衰减节奏都会错。

第四个坑是梯度检查点并非对所有模型自动生效。模型需要实现相应支持逻辑，常见描述是 `supports_gradient_checkpointing`。如果模型结构里有自定义模块，可能只对部分层有效，最终显存下降幅度低于预期。

第五个坑是时间预算。检查点会带来重复前向计算，实际训练常见增加 20% 到 33% 的时间，具体取决于层结构和序列长度。如果你的瓶颈本来就是算力而不是显存，盲目开启检查点可能得不偿失。

常见问题与规避如下：

| 问题 | 影响 | 解决 |
|---|---|---|
| 忘记除以 `accum_steps` | 梯度过大，训练发散 | 每个微批的 loss 先缩放 |
| 微批太小且使用 BatchNorm | 收敛不稳，精度下降 | 改用 LayerNorm 或 GroupNorm |
| 调度器按微批而非更新步工作 | 学习率曲线失真 | 以实际 `optimizer.step()` 次数配置 |
| 检查点未真正生效 | 显存几乎不降 | 确认模型支持并检查日志/显存曲线 |
| 开启检查点后训练过慢 | 吞吐下降明显 | 配合 AMP、Flash Attention、并行优化 |

在 GPT-family 模型里，一个稳妥经验是：使用 LayerNorm 结构、开启 `gradient_checkpointing=True`、再配合梯度累积。这样做的原因不是“流行”，而是这类模型本身就没有 BatchNorm 约束，且层数深、激活大，检查点收益通常更稳定。

---

## 替代方案与适用边界

梯度累积和梯度检查点不是唯一解，它们只是最通用的两种显存优化技术。

混合精度训练，也就是 AMP，本质是让一部分计算和张量用更低精度表示，例如 FP16 或 BF16。低精度就是用更少比特表示数字。它通常能同时减少显存和提高吞吐，因此几乎应该默认开启。但它对优化器状态和超大激活问题的缓解有限，不能替代检查点。

ZeRO 是另一类更激进的方案。它把优化器状态、梯度、参数分片到多卡或多进程上，本质是“把一份大状态拆散，不让每张卡都完整持有”。当模型已经大到单卡连参数或优化器状态都装不下时，仅靠梯度累积基本无解，因为累积不减少这些常驻状态。这时 ZeRO 往往比单纯累积更关键。

真实工程上，一个常见组合是 ZeRO Stage 2 或 Stage 3 加梯度检查点，再叠加混合精度。比如 65B 量级模型的训练或微调，如果目标是单卡 48GB 或少量 GPU 环境，单靠梯度累积往往仍然不够，因为参数和优化器状态已经超出上限；而 ZeRO 分片负责解决“模型状态放不下”，检查点负责解决“激活放不下”，两者职责不同。

下面是常见替代方案对比：

| 技术 | 主要节省哪类显存 | 额外成本 | 适配场景 |
|---|---|---|---|
| 梯度累积 | 大批训练时的瞬时显存 | 更新频率下降 | 单卡想提升等效批 |
| 梯度检查点 | 激活显存 | 训练变慢 | 深层、长序列 Transformer |
| AMP/BF16 | 激活和部分计算缓存 | 数值稳定性需验证 | 几乎所有现代训练 |
| ZeRO | 参数/梯度/优化器状态 | 分布式复杂度高 | 超大模型、多卡训练 |
| QLoRA/LoRA | 可训练参数量 | 适用全量训练有限 | 大模型参数高效微调 |

适用边界可以概括为：

1. 显存不够，但模型本体能放下，优先考虑梯度累积。
2. 模型能放下，但激活太大，优先考虑梯度检查点。
3. 参数和优化器状态都放不下，要考虑 ZeRO、LoRA、QLoRA。
4. 如果计算资源比显存更紧，检查点要谨慎，因为它省显存但更耗时。

---

## 参考资料

- Uplatz, *Gradient Accumulation: A Comprehensive Technical Guide to Training Large-Scale Models on Memory-Constrained Hardware*, 2025, https://uplatz.com/blog/gradient-accumulation-a-comprehensive-technical-guide-to-training-large-scale-models-on-memory-constrained-hardware/ （用于等效批次定义、更新公式、显存与时间权衡）
- InstagIT / Hugging Face Transformers 相关文章, *How does gradient checkpointing reduce memory usage during training?*, 2026, https://instagit.com/huggingface/transformers/how-does-gradient-checkpointing-reduce-memory-usage-during-training/ （用于框架接口、`gradient_checkpointing_enable()` 与 Trainer 配置说明）
- 21medien AI Library, *Gradient Checkpointing*, https://www.21medien.de/en/library/gradient-checkpointing （用于激活存储从 $O(L)$ 到 $O(\sqrt{L})$ 的机制说明与经验数字）
- Hugging Face Transformers Documentation, *TrainingArguments / gradient_checkpointing*, 官方文档，https://huggingface.co/docs/transformers/ （用于工程接口定义与训练参数语义核对）
- PyTorch Documentation, *torch.utils.checkpoint*, 官方文档，https://pytorch.org/docs/stable/checkpoint.html （用于检查点底层机制与框架实现入口）
